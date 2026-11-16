#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Interactive multi-camera checkerboard calibration (1–2 CSI cameras via picamera2).

Three-step workflow (see ``docs/optical/02_calibration_and_frames.md``):

1. ``--mode intrinsics`` — per-camera K/dist from live captures (SPACE/Q).
2. ``--mode extrinsics`` — ``T_world_cam`` per camera; **world = cam0** (identity for cam0);
   robust fusion + acceptance gates (``calibration.extrinsics``); the bundle is
   written only when the gates pass.
3. ``--mode bundle`` — merge existing caches into the bundle without recapture
   (also the explicit override after reviewing a gate failure).

Camera list and board geometry come from ``optical.yaml`` (``calibration:`` section).
Live preview is served over HTTP (Flask MJPEG, same pattern as ``camera_test.py``).
Operator keys are read from the terminal (SPACE capture, Q finish, ESC abort).

Intermediate results are cached next to the bundle as
``<bundle_stem>.intrinsics_cache.npz`` and ``<bundle_stem>.extrinsics_cache.npz``.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from optical.calibration.extrinsics import (
        BoardView,
        ExtrinsicsFit,
        ExtrinsicsGates,
        Snapshot,
        calibration_cache_paths,
        fuse_extrinsics,
        intrinsics_fingerprint,
        rejudge_cached_gates,
    )
    from optical.calibration.geometry import board_points_well_spread, invert_T
    from optical.calibration.io import CameraExtrinsics, CameraIntrinsics, write_calibration_bundle
    from optical.paths import calibration_bundle_path, default_path
    from optical.sync.frame_meta import CamFrameMeta
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from optical.calibration.extrinsics import (
        BoardView,
        ExtrinsicsFit,
        ExtrinsicsGates,
        Snapshot,
        calibration_cache_paths,
        fuse_extrinsics,
        intrinsics_fingerprint,
        rejudge_cached_gates,
    )
    from optical.calibration.geometry import board_points_well_spread, invert_T
    from optical.calibration.io import CameraExtrinsics, CameraIntrinsics, write_calibration_bundle
    from optical.paths import calibration_bundle_path, default_path
    from optical.sync.frame_meta import CamFrameMeta

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - dev machines without Pi stack
    Picamera2 = None  # type: ignore[misc, assignment]


def _now_utc() -> str:
    """ISO-8601 UTC timestamp for log messages."""
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str | Path, *, n_cameras: int | None = None) -> dict:
    """Load optical.yaml (calibration section) or a legacy calibration JSON."""
    from optical.config.loader import calibration_section
    from optical.config.resolve import load_resolved_config

    path = Path(path)
    doc = load_resolved_config(path, n_cameras=n_cameras)
    try:
        return calibration_section(doc, config_path=path)
    except KeyError:
        return doc


def _build_checkerboard_obj_points(inner_corners: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    """
    3D corner coordinates on the board plane (z=0), in millimeters.

    Args:
        inner_corners: (cols, rows) inner corner count, OpenCV convention.
        square_size_mm: Physical square edge length.

    Returns:
        (N, 3) float32 array for ``cv2.calibrateCamera`` / ``solvePnP``.
    """
    cols, rows = inner_corners
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)
    return objp


def _board_square_size_mm(obj_points: np.ndarray) -> float:
    """Shortest neighbor spacing on the board plane (mm) — one square edge."""
    objp = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    min_dist = float("inf")
    for i in range(len(objp)):
        for j in range(i + 1, len(objp)):
            dist = float(np.linalg.norm(objp[j, :2] - objp[i, :2]))
            if dist > 1e-6:
                min_dist = min(min_dist, dist)
    return min_dist if np.isfinite(min_dist) else float("nan")


def _median_board_square_px(
    obj_points_list: Sequence[np.ndarray],
    img_points_list: Sequence[np.ndarray],
    *,
    square_size_mm: float,
) -> float:
    """Median image-space length of one board square across calibration views."""
    pitches: list[float] = []
    tol = max(1e-3, 0.01 * float(square_size_mm))
    for objp, imgp in zip(obj_points_list, img_points_list):
        objp = np.asarray(objp, dtype=np.float64).reshape(-1, 3)
        imgp = np.asarray(imgp, dtype=np.float64).reshape(-1, 2)
        for i in range(len(objp)):
            for j in range(i + 1, len(objp)):
                dist_obj = float(np.linalg.norm(objp[j, :2] - objp[i, :2]))
                if abs(dist_obj - square_size_mm) > tol:
                    continue
                pitches.append(float(np.linalg.norm(imgp[j] - imgp[i])))
    if not pitches:
        return float("nan")
    return float(np.median(pitches))


def _intrinsics_rms_mm(
    rms_px: float,
    obj_points_list: Sequence[np.ndarray],
    img_points_list: Sequence[np.ndarray],
) -> float:
    """Approximate checkerboard-plane RMS (mm) from OpenCV ``calibrateCamera`` RMS (px)."""
    from optical.evaluation.metrics import reprojection_rms_mm

    square_mm = _board_square_size_mm(obj_points_list[0])
    pitch_px = _median_board_square_px(
        obj_points_list, img_points_list, square_size_mm=square_mm
    )
    return reprojection_rms_mm(
        reprojection_rms_px=float(rms_px),
        size_mm=square_mm,
        side_px=pitch_px,
    )


@dataclass
class BoardDetection:
    """Matched board points from one camera view (full or partial).

    Attributes:
        obj_points: (N, 3) float32 board coordinates in millimeters.
        img_points: (N, 1, 2) float32 pixel coordinates (``solvePnP`` layout).
        preview_points: (N, 2) float32 pixels to draw as an overlay.
    """

    obj_points: np.ndarray
    img_points: np.ndarray
    preview_points: np.ndarray


class BoardDetector(ABC):
    """Detect a calibration board and return matched 3D/2D correspondences.

    Both backends reduce to per-view matched points, so the same
    ``calibrateCamera`` / ``solvePnP`` code handles checkerboards (all-or-
    nothing) and ChArUco boards (partial views allowed).
    """

    @abstractmethod
    def detect(self, gray: np.ndarray) -> Optional[BoardDetection]:
        """Return correspondences for one grayscale frame, or ``None``."""


class CheckerboardDetector(BoardDetector):
    """Classic full-grid checkerboard (requires the whole board in view)."""

    def __init__(self, inner_corners: Tuple[int, int], square_size_mm: float) -> None:
        self._inner = inner_corners
        self._objp = _build_checkerboard_obj_points(inner_corners, square_size_mm)
        self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    def detect(self, gray: np.ndarray) -> Optional[BoardDetection]:
        found, corners = cv2.findChessboardCorners(
            gray,
            self._inner,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            return None
        refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)
        return BoardDetection(
            obj_points=self._objp.copy(),
            img_points=refined.astype(np.float32),
            preview_points=refined.reshape(-1, 2),
        )


class CharucoBoardDetector(BoardDetector):
    """ChArUco board: tolerates partial/occluded views via marker IDs."""

    def __init__(
        self,
        *,
        squares_x: int,
        squares_y: int,
        square_length_mm: float,
        marker_length_mm: float,
        dictionary_name: str,
        min_corners: int = 6,
    ) -> None:
        from optical.markers.config import make_charuco_board

        self._board = make_charuco_board(
            squares_x=squares_x,
            squares_y=squares_y,
            square_length_mm=square_length_mm,
            marker_length_mm=marker_length_mm,
            dictionary_name=dictionary_name,
        )
        self._all_obj = self._board.getChessboardCorners().astype(np.float32)
        self._min_corners = int(min_corners)
        self._detector = cv2.aruco.CharucoDetector(self._board)

    def detect(self, gray: np.ndarray) -> Optional[BoardDetection]:
        charuco_corners, charuco_ids, _marker_corners, _marker_ids = self._detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < self._min_corners:
            return None
        ids = charuco_ids.flatten()
        obj_points = self._all_obj[ids]
        if not board_points_well_spread(obj_points, min_points=self._min_corners):
            # Corners on (or near) one line: PnP / calibrateCamera return garbage
            # or assert. Treat as "no board" so it can never be captured.
            return None
        img_points = charuco_corners.reshape(-1, 1, 2).astype(np.float32)
        return BoardDetection(
            obj_points=obj_points,
            img_points=img_points,
            preview_points=charuco_corners.reshape(-1, 2),
        )


def _build_board_detector(board: dict) -> BoardDetector:
    """Construct a :class:`BoardDetector` from the ``board`` config block."""
    board_type = board.get("type", "checkerboard")
    if board_type == "checkerboard":
        inner = board["checkerboard_inner_corners"]
        return CheckerboardDetector((int(inner[0]), int(inner[1])), float(board["square_size_mm"]))
    if board_type == "charuco":
        return CharucoBoardDetector(
            squares_x=int(board["squares_x"]),
            squares_y=int(board["squares_y"]),
            square_length_mm=float(board["square_length_mm"]),
            marker_length_mm=float(board["marker_length_mm"]),
            dictionary_name=str(board.get("dictionary", "DICT_4X4_50")),
            min_corners=int(board.get("min_corners", 6)),
        )
    raise ValueError(f"Unsupported board.type: {board_type!r} (use 'checkerboard' or 'charuco')")


def _rvec_tvec_to_T_cam_obj(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Build homogeneous ``T_cam←obj`` from OpenCV ``solvePnP`` output.

    OpenCV expresses object points in the object frame; the result maps
    object coordinates into the camera frame.
    """
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def _camera_baseline_mm(T_world_cam_a: np.ndarray, T_world_cam_b: np.ndarray) -> float:
    """
    Distance between camera optical centers in a shared world frame (mm).

    Uses the translation columns of ``T_world_cam`` (camera origin in world).
    """
    ta = np.asarray(T_world_cam_a, dtype=np.float64)[:3, 3]
    tb = np.asarray(T_world_cam_b, dtype=np.float64)[:3, 3]
    return float(np.linalg.norm(ta - tb))


def _summarize_baselines(
    baselines_mm: List[float],
    *,
    expected_baseline_mm: Optional[float],
    bundle_baseline_mm: Optional[float],
) -> None:
    """Print per-snapshot baseline statistics after extrinsics capture."""
    if not baselines_mm:
        print("[extrinsics] No paired-camera snapshots for baseline logging.")
        return

    arr = np.asarray(baselines_mm, dtype=np.float64)
    print(
        "[extrinsics] Per-snapshot baseline: "
        f"n={len(arr)}  mean={arr.mean():.1f} mm  std={arr.std():.1f} mm  "
        f"min={arr.min():.1f} mm  max={arr.max():.1f} mm"
    )
    if expected_baseline_mm is not None:
        err = arr - float(expected_baseline_mm)
        print(
            f"[extrinsics] Expected baseline {expected_baseline_mm:.1f} mm: "
            f"mean error {err.mean():+.1f} mm  (|error| mean {np.abs(err).mean():.1f} mm)"
        )
    if bundle_baseline_mm is not None:
        print(f"[extrinsics] Bundle baseline (world=cam0): {bundle_baseline_mm:.1f} mm")
        if expected_baseline_mm is not None and len(arr) >= 1:
            bundle_err = bundle_baseline_mm - float(expected_baseline_mm)
            print(f"[extrinsics] Bundle vs expected: {bundle_err:+.1f} mm")
        if len(arr) >= 1:
            delta = abs(bundle_baseline_mm - float(arr.mean()))
            if delta > 10.0:
                print(
                    f"[extrinsics] WARNING: bundle baseline differs from per-snapshot mean "
                    f"by {delta:.1f} mm — check board geometry (square_length_mm) or outliers."
                )


@dataclass
class CamSpec:
    """One physical camera in the rig."""

    cam_id: str  # logical name, e.g. cam0 (used in HDF5 bundle)
    camera_num: int  # picamera2 camera index (0 or 1 on dual-CSI Pi setups)
    resolution: Tuple[int, int] = (1456, 1088)
    # When set (track / optical-track), capture returns the half-res lores Y plane
    # instead of full-res main — same path as ``jtsmm record --optical-track``.
    lores_size: Optional[Tuple[int, int]] = None


def half_lores_size(resolution: Tuple[int, int]) -> Tuple[int, int]:
    """Half-resolution lores size matching ``jtsmm.capture.capture_cam``."""
    w, h = int(resolution[0]), int(resolution[1])
    return (max(1, w // 2), max(1, h // 2))


def enable_track_lores(cams: List[CamSpec]) -> List[CamSpec]:
    """Set ``lores_size`` to half of ``resolution`` on each cam (in place)."""
    for c in cams:
        if c.lores_size is None:
            c.lores_size = half_lores_size(c.resolution)
    return cams


# Frames of one hardware trigger carry SensorTimestamps within well under 1 ms;
# adjacent triggers are >= 16.7 ms apart (60 Hz). 4 ms separates the two cases
# with margin at any trigger rate up to ~120 Hz.
DEFAULT_MAX_PAIR_SKEW_NS = 4_000_000


class CameraRig(ABC):
    """Capture backend shared by intrinsics and extrinsics workflows."""

    #: Counters for :meth:`capture_all_synced` (paired / resynced / dropped sets).
    sync_stats: Dict[str, int]

    @abstractmethod
    def capture_all(self) -> Dict[str, np.ndarray]:
        """Return the latest BGR frame per ``cam_id``."""

    def _capture_one(self, cam_id: str) -> Optional[Tuple[np.ndarray, CamFrameMeta]]:
        """Capture the next frame of one camera (used to re-align a lagging camera)."""
        return None

    def capture_all_synced(
        self,
        *,
        max_skew_ns: int = DEFAULT_MAX_PAIR_SKEW_NS,
        max_retries: int = 4,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        """Frames of all cameras from the *same* trigger, checked by SensorTimestamp.

        ``capture_all_with_meta`` reads the cameras one after another, so one
        camera can hand back frame k while the other returns k+1 (a stale queue
        entry). Here the camera(s) whose timestamp lags the newest by more than
        ``max_skew_ns`` are read again, up to ``max_retries`` times. Returns
        ``({}, {})`` when no aligned set was found — callers treat that like
        "no frame yet". With fewer than two cameras this is ``capture_all_with_meta``.
        """
        stats = self.__dict__.setdefault(
            "sync_stats", {"paired": 0, "resynced": 0, "dropped": 0, "last_skew_ns": 0}
        )
        frames, meta = self.capture_all_with_meta()
        if len(meta) < 2:
            return frames, meta
        resynced = False
        for _attempt in range(max_retries + 1):
            ts = {cid: int(m.sensor_timestamp_ns) for cid, m in meta.items()}
            newest = max(ts.values())
            stats["last_skew_ns"] = newest - min(ts.values())
            lagging = [cid for cid, t in ts.items() if newest - t > max_skew_ns]
            if not lagging:
                stats["paired"] += 1
                stats["resynced"] += int(resynced)
                return frames, meta
            resynced = True
            for cid in lagging:
                got = self._capture_one(cid)
                if got is None:
                    stats["dropped"] += 1
                    return {}, {}
                frames[cid], meta[cid] = got
        stats["dropped"] += 1
        return {}, {}

    def capture_all_with_meta(self) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        """Return frames plus per-camera metadata (default: host monotonic timestamps)."""
        frames = self.capture_all()
        t_ns = time.monotonic_ns()
        meta = {
            cam_id: CamFrameMeta(
                cam_id=cam_id,
                sensor_timestamp_ns=t_ns,
                t_host_monotonic_ns=t_ns,
            )
            for cam_id in frames
        }
        return frames, meta

    @abstractmethod
    def close(self) -> None:
        """Release camera resources."""


class PicamRig(CameraRig):
    """One or two CSI cameras via picamera2."""

    def __init__(self, cams: List[CamSpec]) -> None:
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not installed; use --backend v4l2 on non-Pi hosts.")
        if len(cams) > 2:
            raise ValueError("PicamRig supports at most 2 cameras.")
        from optical.sync.hat_trigger import hat_trigger

        # IMX296 stays in external-trigger mode: arm the HAT before picamera2
        # waits for the first frame.
        self._hat_trigger = hat_trigger()
        self._hat_trigger.__enter__()
        self._cams = {c.cam_id: c for c in cams}
        self._picams: Dict[str, Picamera2] = {}
        self._frame_idx: Dict[str, int] = {}
        try:
            self._open_cameras(cams)
        except BaseException:
            for picam in list(self._picams.values()):
                try:
                    picam.stop()
                except Exception:
                    pass
            self._picams.clear()
            self._hat_trigger.__exit__(*sys.exc_info())
            self._hat_trigger = None
            raise

    def _open_cameras(self, cams: List[CamSpec]) -> None:
        for c in cams:
            picam = Picamera2(int(c.camera_num))
            if c.lores_size is not None:
                # Match jtsmm record --optical-track: full-res main + half-res lores.
                # Tracking / preview consume the lores Y plane only.
                lw, lh = int(c.lores_size[0]), int(c.lores_size[1])
                config = picam.create_video_configuration(
                    main={"size": c.resolution, "format": "YUV420"},
                    lores={"size": (lw, lh), "format": "YUV420"},
                    buffer_count=4,
                )
            else:
                config = picam.create_preview_configuration(
                    # picamera2 names formats by byte order in reverse (its own
                    # FORMAT_TABLE maps "RGB888" -> "BGR"), so this hands back
                    # [B, G, R] arrays — the BGR that capture_all promises and that
                    # cv2 drawing and PreviewServer.update expect.
                    main={"size": c.resolution, "format": "RGB888"},
                )
            picam.configure(config)
            picam.start()
            self._picams[c.cam_id] = picam
            self._frame_idx[c.cam_id] = 0

    def capture_all(self) -> Dict[str, np.ndarray]:
        frames, _meta = self.capture_all_with_meta()
        return frames

    def capture_all_with_meta(self) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        frames: Dict[str, np.ndarray] = {}
        meta: Dict[str, CamFrameMeta] = {}
        for cam_id in list(self._picams):
            got = self._capture_one(cam_id)
            if got is not None:
                frames[cam_id], meta[cam_id] = got
        return frames, meta

    def _grab_request(self, cam_id: str):
        """Take the next completed request and its SensorTimestamp. No pixel copy."""
        picam = self._picams.get(cam_id)
        if picam is None:
            return None
        request = picam.capture_request()
        md = request.get_metadata()
        sensor_ts = int(md.get("SensorTimestamp", time.time_ns()))
        return request, sensor_ts

    def _frame_from_request(
        self,
        cam_id: str,
        request,
        sensor_ts: int,
    ) -> Tuple[np.ndarray, CamFrameMeta]:
        """Copy pixels out of a request that is still held."""
        from jtsmm.capture.capture_cam import apply_mount_rotation, y_plane

        cam = self._cams[cam_id]
        if cam.lores_size is not None:
            raw = request.make_array("lores")
            frame = apply_mount_rotation(y_plane(raw, cam.lores_size))
        else:
            frame = apply_mount_rotation(request.make_array("main"))
        self._frame_idx[cam_id] += 1
        meta = CamFrameMeta(
            cam_id=cam_id,
            sensor_timestamp_ns=sensor_ts,
            t_host_monotonic_ns=time.monotonic_ns(),
            frame_idx=self._frame_idx[cam_id],
        )
        return frame, meta

    def _capture_one(self, cam_id: str) -> Optional[Tuple[np.ndarray, CamFrameMeta]]:
        got = self._grab_request(cam_id)
        if got is None:
            return None
        request, sensor_ts = got
        try:
            return self._frame_from_request(cam_id, request, sensor_ts)
        finally:
            request.release()

    def capture_all_synced(
        self,
        *,
        max_skew_ns: int = DEFAULT_MAX_PAIR_SKEW_NS,
        max_retries: int = 4,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        """Pair requests by SensorTimestamp, then copy pixels.

        The base implementation copies a full frame before reading the next
        camera. That copy is longer than one trigger period, so the second
        camera has already moved on and a re-read of the first hops past it.
        Here both requests are held and only the lagging one is replaced,
        until the timestamps match. The full-res arrays are made after that.
        """
        if len(self._picams) < 2:
            return self.capture_all_with_meta()
        stats = self.__dict__.setdefault(
            "sync_stats", {"paired": 0, "resynced": 0, "dropped": 0, "last_skew_ns": 0}
        )
        held: Dict[str, tuple] = {}
        try:
            for cam_id in list(self._picams):
                got = self._grab_request(cam_id)
                if got is None:
                    stats["dropped"] += 1
                    return {}, {}
                held[cam_id] = got
            resynced = False
            for _attempt in range(max_retries + 1):
                ts = {cid: int(item[1]) for cid, item in held.items()}
                newest = max(ts.values())
                stats["last_skew_ns"] = newest - min(ts.values())
                lagging = [cid for cid, t in ts.items() if newest - t > max_skew_ns]
                if not lagging:
                    frames: Dict[str, np.ndarray] = {}
                    meta: Dict[str, CamFrameMeta] = {}
                    for cid, (request, sensor_ts) in held.items():
                        frames[cid], meta[cid] = self._frame_from_request(cid, request, sensor_ts)
                    stats["paired"] += 1
                    stats["resynced"] += int(resynced)
                    return frames, meta
                resynced = True
                for cid in lagging:
                    request, _sensor_ts = held.pop(cid)
                    request.release()
                    got = self._grab_request(cid)
                    if got is None:
                        stats["dropped"] += 1
                        return {}, {}
                    held[cid] = got
            stats["dropped"] += 1
            return {}, {}
        finally:
            for request, _sensor_ts in held.values():
                try:
                    request.release()
                except Exception:
                    pass

    def close(self) -> None:
        try:
            for picam in self._picams.values():
                try:
                    picam.stop()
                except Exception:
                    pass
            self._picams.clear()
            self._frame_idx.clear()
        finally:
            if getattr(self, "_hat_trigger", None) is not None:
                self._hat_trigger.__exit__(None, None, None)
                self._hat_trigger = None


class V4l2Rig(CameraRig):
    """Fallback for hosts without picamera2 (OpenCV VideoCapture)."""

    def __init__(self, cams: List[CamSpec]) -> None:
        self._caps: Dict[str, cv2.VideoCapture] = {}
        self._cams = {c.cam_id: c for c in cams}
        self._frame_idx: Dict[str, int] = {}
        for c in cams:
            cap = cv2.VideoCapture(int(c.camera_num))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {c.cam_id} at index={c.camera_num}")
            w, h = c.lores_size if c.lores_size is not None else c.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            self._caps[c.cam_id] = cap
            self._frame_idx[c.cam_id] = 0

    def capture_all(self) -> Dict[str, np.ndarray]:
        frames, _meta = self.capture_all_with_meta()
        return frames

    def capture_all_with_meta(self) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        frames: Dict[str, np.ndarray] = {}
        meta: Dict[str, CamFrameMeta] = {}
        for cam_id, cap in self._caps.items():
            ok, frame = cap.read()
            if not ok:
                continue
            t_ns = time.monotonic_ns()
            cam = self._cams[cam_id]
            if cam.lores_size is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames[cam_id] = gray
            else:
                frames[cam_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._frame_idx[cam_id] += 1
            meta[cam_id] = CamFrameMeta(
                cam_id=cam_id,
                sensor_timestamp_ns=t_ns,
                t_host_monotonic_ns=t_ns,
                frame_idx=self._frame_idx[cam_id],
            )
        return frames, meta

    def capture_all_synced(
        self,
        *,
        max_skew_ns: int = DEFAULT_MAX_PAIR_SKEW_NS,
        max_retries: int = 4,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        """No sensor timestamps (host read time only), so no pairing check: dev fallback."""
        return self.capture_all_with_meta()

    def close(self) -> None:
        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()
        self._frame_idx.clear()


def _open_rig(cams: List[CamSpec], backend: str) -> CameraRig:
    if backend == "picamera2":
        return PicamRig(cams)
    if backend == "v4l2":
        return V4l2Rig(cams)
    raise ValueError(f"Unknown camera backend: {backend}")


# Longest published JPEG size (width × height). Capture may be full-res or lores.
_PREVIEW_MAX_WH = (960, 360)
_PREVIEW_HZ = 20.0
_PREVIEW_PERIOD_S = 1.0 / _PREVIEW_HZ


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    """Normalize gray / RGB / BGR arrays to BGR for JPEG encode and stacking."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return arr


def _fit_preview(bgr: np.ndarray, max_wh: tuple[int, int] = _PREVIEW_MAX_WH) -> np.ndarray:
    """Downscale a BGR frame so it fits in ``max_wh`` (area interpolation)."""
    bgr = _as_bgr(bgr)
    h, w = bgr.shape[:2]
    max_w, max_h = max_wh
    scale = min(1.0, float(max_w) / float(w), float(max_h) / float(h))
    if scale >= 1.0:
        return bgr
    return cv2.resize(
        bgr,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


class PreviewServer:
    """MJPEG preview over HTTP (RGB frames; same pattern as ``camera_test.py``)."""

    def __init__(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._latest_frame: bytes | None = None
        self._version = 0
        self._last_publish_mono = 0.0
        self._lock = threading.Lock()
        # Imported here so calibration math / tests import without Flask installed.
        from flask import Flask, Response

        self._app = Flask(__name__)
        self._placeholder = self._make_placeholder_jpeg("Waiting for camera…")

        @self._app.route("/")
        def index() -> str:
            return (
                "<!doctype html><title>Camera preview</title>"
                "<h1>Camera preview</h1>"
                '<img src="/stream" style="max-width:min(100%,960px);height:auto"/>'
            )

        @self._app.route("/stream")
        def stream() -> "Response":
            return Response(self._generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @staticmethod
    def _make_placeholder_jpeg(text: str) -> bytes:
        img = np.zeros((_PREVIEW_MAX_WH[1], _PREVIEW_MAX_WH[0], 3), dtype=np.uint8)
        cv2.putText(
            img,
            text,
            (20, _PREVIEW_MAX_WH[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        return jpeg.tobytes()

    def _generate(self):
        while True:
            with self._lock:
                frame = self._latest_frame or self._placeholder
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(_PREVIEW_PERIOD_S)

    def update(self, bgr: np.ndarray, *, jpeg_quality: int = 50) -> None:
        """Publish one **BGR** frame to the MJPEG stream.

        cv2.imencode reads its input as BGR, so RGB frames (e.g. picamera2
        "BGR888", or jtsmm's ``frame.rgb``) must be swapped by the caller or the
        stream comes out with red and blue exchanged.

        Frames larger than ``_PREVIEW_MAX_WH`` are downscaled before encode so
        the MJPEG stays light; detection still runs on the capture-resolution copy.

        Publish rate is capped at ``_PREVIEW_HZ`` (encode + JPEG refresh).
        """
        now = time.monotonic()
        with self._lock:
            if self._latest_frame is not None and (now - self._last_publish_mono) < _PREVIEW_PERIOD_S:
                return
        bgr = _fit_preview(bgr)
        _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        with self._lock:
            if self._latest_frame is not None and (now - self._last_publish_mono) < _PREVIEW_PERIOD_S:
                return
            self._latest_frame = jpeg.tobytes()
            self._version += 1
            self._last_publish_mono = now

    def start(self) -> None:
        thread = threading.Thread(
            target=lambda: self._app.run(host=self._host, port=self._port, threaded=True, use_reloader=False),
            daemon=True,
        )
        thread.start()
        time.sleep(0.3)

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}/"


class _TerminalKeys:
    """Non-blocking single-key reads from the terminal (SPACE / Q / ESC)."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)

    def __enter__(self) -> _TerminalKeys:
        # cbreak (not raw) keeps ISIG enabled, so Ctrl+C still raises SIGINT.
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> int | None:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == " ":
            return ord(" ")
        if ch in ("q", "Q"):
            return ord("q")
        if ch == "\x1b":
            return 27
        if ch == "\x03":  # Ctrl+C if ISIG is somehow disabled
            raise KeyboardInterrupt
        return None


_STATUS_BAR_H = 20


def _draw_status_bar(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    """Overlay a compact top status strip (call on preview-sized frames)."""
    disp = frame.copy()
    cv2.rectangle(disp, (0, 0), (disp.shape[1], _STATUS_BAR_H), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        disp,
        text,
        (10, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        0,
        cv2.LINE_AA,
    )
    return disp


def _stack_preview(frames: Dict[str, np.ndarray], order: List[str]) -> np.ndarray | None:
    if len(order) >= 2:
        order = list(reversed(order))
    imgs = [_as_bgr(frames[cid]) for cid in order if cid in frames]
    if not imgs:
        return None
    if len(imgs) == 1:
        return imgs[0]
    h = min(im.shape[0] for im in imgs)
    resized = []
    for im in imgs:
        if im.shape[0] != h:
            scale = h / im.shape[0]
            im = cv2.resize(im, (int(im.shape[1] * scale), h))
        resized.append(im)
    return np.hstack(resized)


class CameraStream:
    """Background capture + preview thread, decoupled from slow detection.

    A dedicated daemon thread continuously grabs frames, draws the most recent
    (cheap) overlay — cached corners plus a status bar — and pushes them to the
    MJPEG preview at ``_PREVIEW_HZ``. The caller's loop runs slow checkerboard
    detection on :meth:`latest` frames without throttling capture.
    """

    def __init__(
        self,
        rig: CameraRig,
        preview: PreviewServer,
        cam_order: List[str],
    ) -> None:
        self._rig = rig
        self._preview = preview
        self._order = cam_order
        self._lock = threading.Lock()
        self._latest: Dict[str, np.ndarray] = {}
        self._latest_meta: Dict[str, CamFrameMeta] = {}
        self._seq = 0
        self._corners: Dict[str, Optional[np.ndarray]] = {}
        self._status_text = ""
        self._status_color: Tuple[int, int, int] = (0, 220, 0)
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _loop(self) -> None:
        while self._running:
            # Timestamp-checked: a snapshot pairs frames of the same trigger.
            frames, meta = self._rig.capture_all_synced()
            if not frames:
                self._warn_unsynced()
                time.sleep(0.001)
                continue
            with self._lock:
                self._latest = frames
                self._latest_meta = meta
                self._seq += 1
                corners = dict(self._corners)
                status_text = self._status_text
                status_color = self._status_color
            overlay: Dict[str, np.ndarray] = {}
            for cam_id, frame in frames.items():
                disp = frame.copy()
                pts = corners.get(cam_id)
                if pts is not None:
                    for x, y in np.asarray(pts).reshape(-1, 2):
                        cv2.circle(disp, (int(round(x)), int(round(y))), 4, (0, 255, 0), -1, cv2.LINE_AA)
                overlay[cam_id] = disp
            stacked = _stack_preview(overlay, self._order)
            if stacked is not None:
                stacked = _fit_preview(stacked)
                if status_text:
                    stacked = _draw_status_bar(stacked, status_text, status_color)
                self._preview.update(stacked)

    def _warn_unsynced(self) -> None:
        """At most every 5 s: say why no frame set arrives (cameras not on one trigger)."""
        stats = getattr(self._rig, "sync_stats", None)
        now = time.monotonic()
        if not stats or not stats.get("dropped") or now - getattr(self, "_last_sync_warn", 0.0) < 5.0:
            return
        self._last_sync_warn = now
        print(
            f"warning: camera frames not from the same trigger "
            f"(skew {stats['last_skew_ns'] / 1e6:.1f} ms > {DEFAULT_MAX_PAIR_SKEW_NS / 1e6:.0f} ms; "
            f"{stats['dropped']} sets dropped) — check the HAT trigger wiring / rate"
        )

    def latest(self) -> Dict[str, np.ndarray]:
        """Most recent raw frames per ``cam_id`` (copy of the dict)."""
        with self._lock:
            return dict(self._latest)

    def latest_with_meta(self) -> Tuple[int, Dict[str, np.ndarray], Dict[str, CamFrameMeta]]:
        """``(seq, frames, meta)`` of the most recent synced set; ``seq`` increments per set."""
        with self._lock:
            return self._seq, dict(self._latest), dict(self._latest_meta)

    def set_overlay(
        self,
        corners: Dict[str, Optional[np.ndarray]],
        status_text: str,
        status_color: Tuple[int, int, int],
    ) -> None:
        """Update the overlay the capture thread draws onto each frame."""
        with self._lock:
            self._corners = corners
            self._status_text = status_text
            self._status_color = status_color


def _parse_cams(cfg: dict) -> List[CamSpec]:
    cams: List[CamSpec] = []
    for c in cfg["cameras"]:
        res = c.get("resolution")
        resolution = (int(res[0]), int(res[1])) if res else (1456, 1088)
        camera_num = int(c.get("camera_num", c.get("device_index", 0)))
        cams.append(CamSpec(cam_id=c["id"], camera_num=camera_num, resolution=resolution))
    return cams


def calibrate_intrinsics_from_live(
    *,
    cam: CamSpec,
    detector: BoardDetector,
    min_images: int,
    backend: str,
    preview: PreviewServer,
) -> Tuple[CameraIntrinsics, float]:
    """
    Interactive intrinsics calibration for a single camera.

    Operator keys: SPACE capture, Q finish (after ``min_images``), ESC abort.

    Args:
        cam: Camera identity and picamera2 index.
        detector: Checkerboard or ChArUco board detector.
        min_images: Minimum captures before Q is accepted.
        backend: ``picamera2`` or ``v4l2``.
        preview: HTTP MJPEG preview server.

    Returns:
        ``(intrinsics, rms_mm)`` — solved intrinsics, global RMS in pixels, and
        the same error expressed on the board plane in millimetres.
    """
    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []

    rig = _open_rig([cam], backend)
    stream = CameraStream(rig, preview, [cam.cam_id])
    stream.start()
    image_size: Optional[Tuple[int, int]] = None
    print("=" * 70)
    print(f"[intrinsics] Camera {cam.cam_id} (camera_num {cam.camera_num})")
    print(f"Preview: {preview.url}")
    print("SPACE: capture | Q: solve+finish | ESC/Ctrl+C: abort")
    print("=" * 70)

    try:
        with _TerminalKeys() as keys:
            while True:
                frame = stream.latest().get(cam.cam_id)
                if frame is None:
                    time.sleep(0.01)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                image_size = (gray.shape[1], gray.shape[0])

                det = detector.detect(gray)
                if det is not None:
                    status = f"Board detected ({len(det.obj_points)} pts) — SPACE to capture"
                    color = (0, 220, 0)
                else:
                    status = "No board"
                    color = (0, 100, 220)

                stream.set_overlay(
                    {cam.cam_id: det.preview_points if det is not None else None},
                    f"{cam.cam_id} captures {len(obj_points)}/{min_images} | {status}",
                    color,
                )

                key = keys.poll()
                if key == 27:
                    raise SystemExit("Aborted.")
                if key == ord(" "):
                    if det is not None:
                        obj_points.append(det.obj_points)
                        img_points.append(det.img_points)
                        print(f"  captured {len(obj_points)}")
                if key == ord("q"):
                    if len(obj_points) < min_images:
                        print(f"Need at least {min_images} captures (have {len(obj_points)})")
                    else:
                        break
    finally:
        stream.stop()
        rig.close()

    if image_size is None:
        raise SystemExit(f"{cam.cam_id}: no frame received; nothing to calibrate.")
    sol = solve_intrinsics(obj_points, img_points, image_size)
    for msg in sol.dropped:
        print(f"  {cam.cam_id}: {msg}")
    std = sol.std_intrinsics
    if std.size >= 4:
        print(
            f"  {cam.cam_id}: {int(sol.kept.sum())}/{len(obj_points)} views used; "
            f"fx={sol.K[0, 0]:.1f}±{std[0]:.1f} fy={sol.K[1, 1]:.1f}±{std[1]:.1f} "
            f"cx={sol.K[0, 2]:.1f}±{std[2]:.1f} cy={sol.K[1, 2]:.1f}±{std[3]:.1f} px; "
            f"worst view {float(np.max(sol.per_view_err_px)):.2f} px"
        )
    kept_obj = [o for o, k in zip(obj_points, sol.kept) if k]
    kept_img = [p for p, k in zip(img_points, sol.kept) if k]
    rms_mm = _intrinsics_rms_mm(sol.rms_px, kept_obj, kept_img)
    intr = CameraIntrinsics(K=sol.K, dist=sol.dist.ravel(), image_size=image_size, rms_px=sol.rms_px)
    return intr, rms_mm


@dataclass
class IntrinsicsSolve:
    """Result of :func:`solve_intrinsics`."""

    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    kept: np.ndarray  # (n_views,) bool — views used in the final solve
    per_view_err_px: np.ndarray  # RMS per kept view (calibrateCameraExtended)
    std_intrinsics: np.ndarray  # 1-σ of (fx, fy, cx, cy, k1, k2, p1, p2, k3, …)
    dropped: List[str]  # human-readable reasons, one per dropped view


def _calibrate_views(obj_points, img_points, image_size):
    """``cv2.calibrateCameraExtended`` → (rms, K, dist, std_intrinsics, per_view_errors)."""
    rms, K, dist, _r, _t, std_i, _std_e, per_view = cv2.calibrateCameraExtended(
        list(obj_points), list(img_points), tuple(image_size), None, None
    )
    return float(rms), K, dist, np.asarray(std_i).ravel(), np.asarray(per_view).ravel()


def solve_intrinsics(
    obj_points: Sequence[np.ndarray],
    img_points: Sequence[np.ndarray],
    image_size: Tuple[int, int],
    *,
    outlier_factor: float = 3.0,
    outlier_floor_px: float = 1.0,
) -> IntrinsicsSolve:
    """Robust ``calibrateCamera``: one bad view must not cost the whole session.

    1. Views whose board points are degenerate (collinear) are dropped up front.
    2. If OpenCV still raises (e.g. an assertion on a near-degenerate view),
       each view is left out in turn and the first successful solve is kept.
    3. Views whose RMS exceeds ``max(outlier_floor_px, outlier_factor × median)``
       are dropped and the solve is repeated once, as long as at least half
       of the views (and ≥ 5) remain.

    Raises:
        SystemExit: fewer than 3 usable views, or no solve succeeded.
    """
    n = len(obj_points)
    kept = np.ones(n, dtype=bool)
    dropped: List[str] = []
    for i, obj in enumerate(obj_points):
        if not board_points_well_spread(obj):
            kept[i] = False
            dropped.append(f"view {i + 1}: dropped (board corners collinear / too few)")

    def run(mask: np.ndarray):
        idx = np.flatnonzero(mask)
        if idx.size < 3:
            raise SystemExit(f"Only {idx.size} usable calibration views; capture more.")
        return _calibrate_views([obj_points[i] for i in idx], [img_points[i] for i in idx], image_size)

    try:
        res = run(kept)
    except cv2.error as exc:
        res = None
        for i in np.flatnonzero(kept):
            trial = kept.copy()
            trial[i] = False
            try:
                res = run(trial)
            except cv2.error:
                continue
            kept = trial
            dropped.append(f"view {i + 1}: dropped (calibrateCamera failed with it: {str(exc).strip().splitlines()[-1]})")
            break
        if res is None:
            raise SystemExit(f"calibrateCamera failed on every subset of views: {exc}") from exc

    rms, K, dist, std_i, per_view = res
    idx = np.flatnonzero(kept)
    limit = max(outlier_floor_px, outlier_factor * float(np.median(per_view)))
    bad = per_view > limit
    if bad.any() and (idx.size - int(bad.sum())) >= max(5, (idx.size + 1) // 2):
        for i, e in zip(idx[bad], per_view[bad]):
            dropped.append(f"view {i + 1}: dropped (reprojection {e:.2f} px > {limit:.2f} px)")
        kept[idx[bad]] = False
        rms, K, dist, std_i, per_view = run(kept)

    return IntrinsicsSolve(
        K=K, dist=np.asarray(dist).ravel(), rms_px=rms, kept=kept,
        per_view_err_px=per_view, std_intrinsics=std_i, dropped=dropped,
    )


def _require_calibrated_size(frame: np.ndarray, intr: CameraIntrinsics, cam_id: str) -> None:
    """Refuse PnP on frames whose size differs from the intrinsics' image_size.

    K is only valid at the resolution it was solved for; PnP with a mismatched
    K yields a wrong board pose without any error.
    """
    h, w = np.asarray(frame).shape[:2]
    if (int(w), int(h)) != tuple(int(v) for v in intr.image_size):
        raise SystemExit(
            f"{cam_id}: frame is {w}x{h} but intrinsics were calibrated at "
            f"{intr.image_size[0]}x{intr.image_size[1]}. Capture extrinsics at the "
            "intrinsics resolution (calibration.cameras[].resolution) or redo intrinsics."
        )


def estimate_extrinsics_snapshots(
    *,
    cams: List[CamSpec],
    intrinsics: Dict[str, CameraIntrinsics],
    detector: BoardDetector,
    min_snapshots: int,
    backend: str,
    preview: PreviewServer,
    expected_baseline_mm: Optional[float] = None,
    gates: ExtrinsicsGates = ExtrinsicsGates(),
) -> Tuple[ExtrinsicsFit, Dict[str, np.ndarray]]:
    """
    Estimate ``T_world_cam`` per camera from synchronized board views.

    **World frame = cam0** (``T_world_cam0 = I``). Each SPACE snapshot stores one
    PnP board pose per camera; views above ``gates.max_pnp_rms_px`` are dropped
    immediately. On Q the snapshots are fused robustly and graded by the
    cross-view error (see :mod:`optical.calibration.extrinsics`).

    Per-snapshot baseline (cam0–cam1 distance) is logged when both cameras detect
    the board on the same SPACE press.

    Args:
        cams: All cameras in the rig (1 or 2 with picamera2).
        intrinsics: Precomputed intrinsics per ``cam_id``.
        detector: Checkerboard or ChArUco board detector.
        min_snapshots: Minimum paired snapshots (multi-cam) or per-cam captures.
        backend: ``picamera2`` or ``v4l2``.
        preview: HTTP MJPEG preview server.
        expected_baseline_mm: Optional physical baseline for log comparison.
        gates: PnP / outlier / cross-view acceptance thresholds.

    Returns:
        ``(fit, stats)`` where ``stats`` may include
        ``baseline_mm_per_snapshot`` (1D array).
    """
    rig = _open_rig(cams, backend)
    cam_order = [c.cam_id for c in cams]
    world_id = cam_order[0]
    stream = CameraStream(rig, preview, cam_order)
    stream.start()

    snapshots: List[Snapshot] = []
    baseline_log: List[float] = []
    baseline_pair: Optional[Tuple[str, str]] = None
    if len(cam_order) >= 2:
        baseline_pair = (cam_order[0], cam_order[1])

    def n_paired(cid: str) -> int:
        return sum(1 for s in snapshots if world_id in s and cid in s)

    def n_world() -> int:
        return sum(1 for s in snapshots if world_id in s)

    print("=" * 70)
    print(f"[extrinsics] Multi-camera snapshots (world = {world_id})")
    print(f"Preview: {preview.url}")
    print("SPACE: capture snapshot | Q: finish | ESC/Ctrl+C: abort")
    print(f"Views with board PnP RMS > {gates.max_pnp_rms_px:.2f} px are dropped.")
    if baseline_pair is not None:
        a, b = baseline_pair
        print(f"Per-snapshot baseline logged for {a}–{b} when both detect the board.")
        if expected_baseline_mm is not None:
            print(f"Expected physical baseline: {expected_baseline_mm:.1f} mm")
    print("=" * 70)

    try:
        with _TerminalKeys() as keys:
            while True:
                frames = stream.latest()
                if not frames:
                    time.sleep(0.01)
                    continue

                live_dets: Dict[str, Optional[BoardDetection]] = {}
                overlay: Dict[str, Optional[np.ndarray]] = {}
                for cam_id, frame in frames.items():
                    _require_calibrated_size(frame, intrinsics[cam_id], cam_id)
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    det = detector.detect(gray)
                    live_dets[cam_id] = det
                    overlay[cam_id] = det.preview_points if det is not None else None

                if len(cam_order) >= 2:
                    paired_n = min(n_paired(cid) for cid in cam_order[1:])
                    status = f"paired:{paired_n} (need {min_snapshots})"
                else:
                    status = f"{world_id}:{n_world()} (need {min_snapshots})"
                stream.set_overlay(
                    overlay,
                    f"Snapshots {status} | SPACE capture | Q finish",
                    (0, 220, 0),
                )

                key = keys.poll()
                if key == 27:
                    raise SystemExit("Aborted.")

                if key == ord(" "):
                    snap: Snapshot = {}
                    for cam in cams:
                        cam_id = cam.cam_id
                        det = live_dets.get(cam_id)
                        if det is None:
                            continue

                        intr = intrinsics[cam_id]
                        ok, rvec, tvec = cv2.solvePnP(
                            det.obj_points,
                            det.img_points,
                            intr.K,
                            intr.dist,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        if not ok:
                            continue
                        proj, _ = cv2.projectPoints(det.obj_points, rvec, tvec, intr.K, intr.dist)
                        err = np.linalg.norm(
                            det.img_points.reshape(-1, 2) - proj.reshape(-1, 2), axis=1
                        )
                        pnp_rms = float(np.sqrt(np.mean(err**2)))
                        if not (pnp_rms <= gates.max_pnp_rms_px):
                            print(
                                f"  {cam_id}: view dropped — board PnP RMS {pnp_rms:.2f} px "
                                f"> {gates.max_pnp_rms_px:.2f} px (blur / partial board?)"
                            )
                            continue
                        snap[cam_id] = BoardView(
                            obj_points=np.asarray(det.obj_points, dtype=np.float64).reshape(-1, 3),
                            img_points=np.asarray(det.img_points, dtype=np.float64).reshape(-1, 2),
                            T_cam_board=_rvec_tvec_to_T_cam_obj(rvec, tvec),
                            pnp_rms_px=pnp_rms,
                        )
                    if snap:
                        snapshots.append(snap)

                    if len(cam_order) >= 2:
                        paired_counts = {cid: n_paired(cid) for cid in cam_order[1:]}
                        print(f"snapshot captured; paired samples: {paired_counts}")
                    else:
                        print(f"snapshot captured; {world_id} samples: {n_world()}")

                    if baseline_pair is not None:
                        ca, cb = baseline_pair
                        if ca in snap and cb in snap:
                            T_world_cb = snap[ca].T_cam_board @ invert_T(snap[cb].T_cam_board)
                            baseline_mm = _camera_baseline_mm(np.eye(4), T_world_cb)
                            baseline_log.append(baseline_mm)
                            reproj = ", ".join(f"{c}={snap[c].pnp_rms_px:.2f}px" for c in (ca, cb))
                            line = (
                                f"  baseline {ca}–{cb}: {baseline_mm:.1f} mm  "
                                f"board reproj: {reproj}"
                            )
                            if expected_baseline_mm is not None:
                                line += (
                                    f"  (Δ vs expected "
                                    f"{baseline_mm - expected_baseline_mm:+.1f} mm)"
                                )
                            print(line)
                        else:
                            missing = [cid for cid in (ca, cb) if cid not in snap]
                            print(f"  baseline skipped — no usable board view in: {missing}")

                if key == ord("q"):
                    if len(cam_order) == 1:
                        ready = n_world() >= min_snapshots
                    else:
                        ready = all(n_paired(cid) >= min_snapshots for cid in cam_order[1:])
                    if ready:
                        break
                    print("Not enough snapshots yet:")
                    if len(cam_order) == 1:
                        print(f"  {world_id}: {n_world()}/{min_snapshots}")
                    else:
                        for cid in cam_order[1:]:
                            print(f"  paired {world_id}–{cid}: {n_paired(cid)}/{min_snapshots}")
    finally:
        stream.stop()
        rig.close()

    try:
        fit = fuse_extrinsics(snapshots, cam_order, intrinsics, gates)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    bundle_baseline_mm: Optional[float] = None
    if baseline_pair is not None:
        bundle_baseline_mm = _camera_baseline_mm(
            fit.extrinsics[baseline_pair[0]].T_world_cam,
            fit.extrinsics[baseline_pair[1]].T_world_cam,
        )

    print(f"[extrinsics] Fused world frame: {world_id} (T_world_{world_id} = identity)")
    for cid, ext in fit.extrinsics.items():
        t = ext.T_world_cam[:3, 3]
        print(
            f"  {cid}: T_world_cam translation = [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm  "
            f"cross-view rms = {ext.cross_view_rms_px:.2f} px  "
            f"(mean board PnP rms = {ext.rms_px:.3f} px)"
        )
    for line in fit.summary_lines():
        print(f"  {line}")
    for cid, inl in fit.inliers.items():
        for i in np.flatnonzero(fit.paired[cid] & ~inl):
            print(
                f"  rejected snapshot #{i + 1} ({world_id}->{cid}): "
                f"Δrot {fit.rot_dev_deg[cid][i]:.2f}°  Δt {fit.trans_dev_mm[cid][i]:.1f} mm  "
                f"cross-view {fit.per_snapshot_cross_view_px[cid][i]:.2f} px"
            )

    _summarize_baselines(
        baseline_log,
        expected_baseline_mm=expected_baseline_mm,
        bundle_baseline_mm=bundle_baseline_mm,
    )

    stats: Dict[str, np.ndarray] = {}
    if baseline_log:
        stats["baseline_mm_per_snapshot"] = np.asarray(baseline_log, dtype=np.float64)
    return fit, stats


def _savez_atomic(path: Path, **arrays: np.ndarray) -> None:
    """``np.savez`` to a temp file in the same directory, then rename into place."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _save_extrinsics_cache(
    extr_path: Path,
    *,
    fit: ExtrinsicsFit,
    stats: Dict[str, np.ndarray],
    intr: Dict[str, CameraIntrinsics],
    cam_order: Sequence[str],
    gates: ExtrinsicsGates,
) -> None:
    """Persist the fused extrinsics, their quality and the intrinsics fingerprint."""
    save: Dict[str, np.ndarray] = {}
    for cid, ext in fit.extrinsics.items():
        save[f"{cid}_T_world_cam"] = np.asarray(ext.T_world_cam, dtype=np.float64)
        save[f"{cid}_rms_px"] = np.asarray(ext.rms_px, dtype=np.float64)
        save[f"{cid}_cross_view_rms_px"] = np.asarray(ext.cross_view_rms_px, dtype=np.float64)
    for cid, arrays in fit.quality_arrays().items():
        for name, arr in arrays.items():
            save[f"{cid}_q_{name}"] = np.asarray(arr)
    save.update(stats)
    save["world_frame"] = np.array(str(cam_order[0]))
    save["intrinsics_sha256"] = np.array(intrinsics_fingerprint(intr, list(cam_order)))
    save["gates_passed"] = np.asarray(bool(fit.gates_passed))
    save["gate_problems"] = np.asarray(fit.problems if fit.problems else [""], dtype=str)
    save["gate_max_pnp_rms_px"] = np.asarray(gates.max_pnp_rms_px)
    save["gate_max_cross_view_rms_px"] = np.asarray(gates.max_cross_view_rms_px)
    save["gate_min_inlier_fraction"] = np.asarray(gates.min_inlier_fraction)
    _savez_atomic(extr_path, **save)


def _export_calibration_bundle(
    *,
    cfg: dict,
    cams: Sequence["CamSpec"],
    intr: Dict[str, CameraIntrinsics],
    extr_path: Path,
    bundle_path: Path,
    rig_id: str,
    gates: ExtrinsicsGates | None = None,
) -> Path:
    """Merge intrinsics + extrinsics caches into the calibration bundle (HDF5).

    Refuses an extrinsics cache that was solved with different intrinsics (or
    that predates the fingerprint): its board poses, and therefore
    ``T_world_cam``, belong to the other K / dist.

    A stored gate failure is re-checked against ``gates`` (the current
    ``calibration.capture`` thresholds). Raising ``max_cross_view_rms_px``
    after the solve therefore accepts the same cache without recapture.
    """
    cam_ids = [c.cam_id for c in cams]
    extr: Dict[str, CameraExtrinsics] = {}
    quality: Dict[str, Dict[str, np.ndarray]] = {}
    meta_attrs: Dict[str, object] = {"world_frame": cam_ids[0]}
    if extr_path.exists():
        data_e = np.load(extr_path)
        if any(f"{cid}_T_world_cam" in data_e for cid in cam_ids):
            stored = str(data_e["intrinsics_sha256"]) if "intrinsics_sha256" in data_e else None
            current = intrinsics_fingerprint(intr, cam_ids)
            if stored != current:
                raise SystemExit(
                    f"{extr_path.name} was solved with different intrinsics "
                    f"({'no fingerprint stored' if stored is None else 'fingerprint mismatch'}). "
                    "Run --mode extrinsics again with the current intrinsics."
                )
            if "world_frame" in data_e and str(data_e["world_frame"]) != cam_ids[0]:
                raise SystemExit(
                    f"{extr_path.name} uses world frame {str(data_e['world_frame'])!r}, "
                    f"but the first camera is {cam_ids[0]!r}. Re-run --mode extrinsics."
                )
            if "gates_passed" in data_e or any(
                f"{cid}_cross_view_rms_px" in data_e for cid in cam_ids[1:]
            ):
                judged = rejudge_cached_gates(data_e, cam_ids, gates or ExtrinsicsGates())
                if judged is None:
                    passed = bool(np.asarray(data_e["gates_passed"]))
                    problems = [str(p) for p in np.atleast_1d(data_e["gate_problems"]) if str(p)]
                else:
                    passed, problems = judged
                meta_attrs["extrinsics_gates_passed"] = passed
                meta_attrs["extrinsics_gate_problems"] = "; ".join(problems)
                if not passed:
                    print(
                        "WARNING: writing the bundle although the extrinsics acceptance gates "
                        "failed (operator override):"
                    )
                    for p in problems:
                        print(f"  - {p}")
        for cam in cams:
            cid = cam.cam_id
            if f"{cid}_T_world_cam" in data_e:
                cross_key = f"{cid}_cross_view_rms_px"
                extr[cid] = CameraExtrinsics(
                    T_world_cam=data_e[f"{cid}_T_world_cam"],
                    rms_px=float(np.asarray(data_e[f"{cid}_rms_px"])),
                    cross_view_rms_px=(
                        float(np.asarray(data_e[cross_key])) if cross_key in data_e else float("nan")
                    ),
                )
                prefix = f"{cid}_q_"
                q = {k[len(prefix):]: data_e[k] for k in data_e.files if k.startswith(prefix)}
                if q:
                    quality[cid] = q

    # A single-camera rig has no cross-camera geometry to solve: define
    # world = cam (identity). Multi-camera rigs still require extrinsics.
    missing = [c.cam_id for c in cams if c.cam_id not in extr]
    if missing:
        if len(cams) == 1:
            cid = missing[0]
            extr[cid] = CameraExtrinsics(
                T_world_cam=np.eye(4, dtype=np.float64),
                rms_px=float("nan"),
            )
            print(f"Single-camera rig: world = {cid} (identity T_world_cam); skipping extrinsics.")
        else:
            raise SystemExit(
                f"Missing extrinsics for {missing}. Run --mode extrinsics first "
                "(required for multi-camera rigs)."
            )

    fid_cfg = cfg.get("fiducials", {})
    fiducials = {
        "detector": fid_cfg.get("detector", "aruco"),
        "dictionary": fid_cfg.get("dictionary", "DICT_4X4_50"),
        "tag_size_mm": float(fid_cfg.get("tag_size_mm", 25.0)),
        "marker_config_path": str(
            fid_cfg.get(
                "marker_config_path",
                default_path("marker_config"),
            )
        ),
    }

    out = write_calibration_bundle(
        bundle_path,
        rig_id=rig_id,
        schema_version=str(cfg.get("schema_version", "1.0")),
        cameras_intrinsics=intr,
        cameras_extrinsics=extr,
        fiducials=fiducials,
        notes=str(cfg.get("notes", "")),
        extrinsics_quality=quality,
        meta_attrs=meta_attrs,
    )
    print(f"Wrote calibration bundle {out} at {_now_utc()}")
    return out


def _missing_cache_hint(path: Path, legacy_name: str) -> str:
    legacy = path.parent / legacy_name
    if legacy.exists():
        return (
            f" Found legacy {legacy.name} (shared by all camera profiles); if it belongs to "
            f"this profile, rename it to {path.name}, otherwise recapture."
        )
    return ""


def run_calibration(
    *,
    config_path: Path,
    mode: str,
    backend: str = "picamera2",
    preview_host: str = "0.0.0.0",
    preview_port: int = 5000,
) -> None:
    """
    Run intrinsics, extrinsics, or bundle export from optical.yaml / calibration JSON.

    ``extrinsics`` writes the cache and, only if the acceptance gates pass, the
    HDF5 bundle. Otherwise the cache is kept for review and ``--mode bundle``
    writes the bundle on explicit operator request.

    Caches are named after the bundle (``<bundle_stem>.intrinsics_cache.npz`` /
    ``<bundle_stem>.extrinsics_cache.npz``) so the 1- and 2-camera profiles do
    not overwrite each other.

    Args:
        config_path: Path to optical.yaml (uses ``calibration:``) or legacy JSON.
        mode: One of ``intrinsics``, ``extrinsics``, or ``bundle``.
        backend: Camera capture backend (``picamera2`` or ``v4l2``).
        preview_host: Flask MJPEG preview bind address.
        preview_port: Flask MJPEG preview port.
    """
    cfg = _load_json(config_path)
    # Resolve relative bundle path against repo root.
    from optical.paths import resolve_repo_path

    out_cfg = cfg.get("output", {})
    if out_cfg.get("bundle_path"):
        bp = Path(out_cfg["bundle_path"])
        if not bp.is_absolute():
            cfg.setdefault("output", {})["bundle_path"] = str(resolve_repo_path(bp))
    rig_id = str(cfg.get("rig_id", "rig"))
    cams = _parse_cams(cfg)
    if backend == "picamera2" and len(cams) > 2:
        raise SystemExit("picamera2 backend supports at most 2 cameras; use --backend v4l2 for more.")

    detector = _build_board_detector(cfg["board"])

    cap_cfg = cfg.get("capture", {})
    min_images = int(cap_cfg.get("min_images_intrinsics", 25))
    min_snaps = int(cap_cfg.get("min_snapshots_extrinsics", 40))
    gates = ExtrinsicsGates.from_capture_cfg(cap_cfg)
    ignored = [k for k in ("lock_exposure", "lock_focus") if k in cap_cfg]
    if ignored:
        # Never implemented: picamera2 runs its default AE here, and the IMX296
        # has no autofocus (manual lens). Say so instead of implying a lock.
        print(f"Note: calibration.capture.{'/'.join(ignored)} are not implemented and ignored.")
    expected_baseline_mm = cfg.get("expected_baseline_mm")
    if expected_baseline_mm is not None:
        expected_baseline_mm = float(expected_baseline_mm)

    out_cfg = cfg.get("output", {})
    bundle_path = Path(out_cfg.get("bundle_path", calibration_bundle_path(rig_id)))

    scratch_dir = bundle_path.parent
    scratch_dir.mkdir(parents=True, exist_ok=True)
    intr_path, extr_path = calibration_cache_paths(bundle_path)

    if mode == "intrinsics":
        from contextlib import nullcontext

        from optical.sync.hat_trigger import CALIBRATE_TRIGGER_RATE_HZ, hat_trigger

        with hat_trigger(rate_hz=CALIBRATE_TRIGGER_RATE_HZ) if backend == "picamera2" else nullcontext():
            preview = PreviewServer(preview_host, preview_port)
            preview.start()
            intr: Dict[str, CameraIntrinsics] = {}
            for cam in cams:
                intr_val, rms_mm = calibrate_intrinsics_from_live(
                    cam=cam,
                    detector=detector,
                    min_images=min_images,
                    backend=backend,
                    preview=preview,
                )
                intr[cam.cam_id] = intr_val
                print(
                    f"{cam.cam_id}: intrinsics rms={intr_val.rms_px:.3f}px "
                    f"({rms_mm:.3f}mm on board) size={intr_val.image_size}"
                )

            _savez_atomic(
                intr_path,
                **{f"{cid}_K": v.K for cid, v in intr.items()},
                **{f"{cid}_dist": v.dist for cid, v in intr.items()},
                **{f"{cid}_image_size": np.asarray(v.image_size, dtype=np.int32) for cid, v in intr.items()},
                **{f"{cid}_rms_px": np.asarray(v.rms_px, dtype=np.float64) for cid, v in intr.items()},
            )
            print(f"Wrote {intr_path}")
            if extr_path.exists():
                print(
                    f"Note: {extr_path.name} was solved with the previous intrinsics; "
                    "--mode bundle will refuse it until --mode extrinsics is re-run."
                )
        return

    if not intr_path.exists():
        raise SystemExit(
            f"Missing intrinsics cache {intr_path}. Run --mode intrinsics first."
            + _missing_cache_hint(intr_path, "intrinsics_cache.npz")
        )

    data = np.load(intr_path)
    intr = {}
    for cam in cams:
        cid = cam.cam_id
        if f"{cid}_K" not in data:
            raise SystemExit(f"{intr_path.name} has no intrinsics for {cid}. Run --mode intrinsics.")
        intr[cid] = CameraIntrinsics(
            K=data[f"{cid}_K"],
            dist=data[f"{cid}_dist"],
            image_size=tuple(int(x) for x in data[f"{cid}_image_size"]),
            rms_px=float(np.asarray(data[f"{cid}_rms_px"])),
        )

    if mode == "extrinsics":
        from contextlib import nullcontext

        from optical.sync.hat_trigger import CALIBRATE_TRIGGER_RATE_HZ, hat_trigger

        with hat_trigger(rate_hz=CALIBRATE_TRIGGER_RATE_HZ) if backend == "picamera2" else nullcontext():
            preview = PreviewServer(preview_host, preview_port)
            preview.start()
            fit, extr_stats = estimate_extrinsics_snapshots(
                cams=cams,
                intrinsics=intr,
                detector=detector,
                min_snapshots=min_snaps,
                backend=backend,
                preview=preview,
                expected_baseline_mm=expected_baseline_mm,
                gates=gates,
            )
            _save_extrinsics_cache(
                extr_path,
                fit=fit,
                stats=extr_stats,
                intr=intr,
                cam_order=[c.cam_id for c in cams],
                gates=gates,
            )
            print(f"Wrote {extr_path}")
        if not fit.gates_passed:
            print("Extrinsics acceptance gates FAILED:")
            for p in fit.problems:
                print(f"  - {p}")
            raise SystemExit(
                f"Bundle NOT written; {extr_path.name} kept for review. Recapture "
                "(board flat, sharp, fully visible in both cameras), or accept this "
                "result explicitly with --mode bundle."
            )
        _export_calibration_bundle(
            cfg=cfg,
            cams=cams,
            intr=intr,
            extr_path=extr_path,
            bundle_path=bundle_path,
            rig_id=rig_id,
            gates=gates,
        )
        return

    if mode == "bundle":
        if len(cams) > 1 and not extr_path.exists():
            raise SystemExit(
                f"Missing extrinsics cache {extr_path}. Run --mode extrinsics first."
                + _missing_cache_hint(extr_path, "extrinsics_cache.npz")
            )
        _export_calibration_bundle(
            cfg=cfg,
            cams=cams,
            intr=intr,
            extr_path=extr_path,
            bundle_path=bundle_path,
            rig_id=rig_id,
            gates=gates,
        )
        return

    raise SystemExit(f"Unknown mode {mode!r}; expected intrinsics|extrinsics|bundle")


def main(argv: list[str] | None = None) -> None:
    """Argparse entry for ``python -m optical.calibration.multicam``."""
    ap = argparse.ArgumentParser(
        description="Multi-camera checkerboard calibration (intrinsics + world extrinsics).",
    )
    ap.add_argument(
        "--config",
        required=True,
        help="optical.yaml (uses calibration: section; default: software/optical/etc/optical.yaml)",
    )
    ap.add_argument(
        "--mode",
        required=True,
        choices=["intrinsics", "extrinsics", "bundle"],
        help="intrinsics | extrinsics (writes bundle afterwards) | bundle",
    )
    ap.add_argument(
        "--backend",
        default="picamera2",
        choices=["picamera2", "v4l2"],
        help="Camera capture backend (default: picamera2)",
    )
    ap.add_argument("--preview-host", default="0.0.0.0", help="Flask preview bind address")
    ap.add_argument("--preview-port", type=int, default=5000, help="Flask preview port")
    args = ap.parse_args(argv)
    run_calibration(
        config_path=Path(args.config),
        mode=args.mode,
        backend=args.backend,
        preview_host=args.preview_host,
        preview_port=args.preview_port,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(130)
