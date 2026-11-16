# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Optical ArUco bridge for ``jtsmm record --optical-track``.

Owns tracker construction (lores-scaled intrinsics), an off-thread worker, and
the per-timestamp RGB bundle → HDF5 ``/optical`` path.

Stereo pairing is by **sensor time**, not MCU trigger id. The two IMX296s share
a hardware pulse, but picamera2 does not report that id, so sequence-consumed
``t_master_us`` labels can attach the same id to exposures seconds apart.
Frames whose ``SensorTimestamp`` values fall within half a frame period are
treated as one shutter; unmatched RGB is never triangulated.

``track_hz`` decimates detection only. Cameras and ``/cameras`` timestamps stay
at the trigger rate. A skipped pair is ``track_skipped``, not a drop: the
pixels were paired, then not sent to the detector. ``/optical`` stores only
the poses that were detected.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jtsmm.capture.capture_cam import CameraCapture


@dataclass
class _RgbFrame:
    cam_id: str
    sensor_ns: int
    t_master_us: int
    rgb: Any
    noted_ns: int


def nth(n: int) -> str:
    """``3`` → ``3rd``."""
    n = int(n)
    if n % 100 in (11, 12, 13):
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def track_stride(track_hz: float | None, camera_hz: float) -> int:
    """Stereo pairs per live pose. ``1`` detects every pair.

    ``track_hz`` at or above ``camera_hz``, or ``None``, stays at every pair.
    Otherwise the stride is the nearest integer to ``camera_hz / track_hz``
    (a tie rounds up, so 12 Hz on a 30 Hz camera is every 3rd pair).
    """
    if track_hz is None:
        return 1
    try:
        hz = float(track_hz)
        cam = float(camera_hz)
    except (TypeError, ValueError):
        return 1
    if hz <= 0 or cam <= 0 or hz >= cam:
        return 1
    return max(1, int(cam / hz + 0.5))


_EMPTY_QUALITY_SLOT = {"n_tags": 0, "n_cameras": 0, "reprojection_rms": float("nan")}
_EMPTY_QUALITY = {"mta": dict(_EMPTY_QUALITY_SLOT), "cra": dict(_EMPTY_QUALITY_SLOT), "ct": dict(_EMPTY_QUALITY_SLOT)}
_BODY_TO_SLOT = {"MP": "mta", "HP": "cra", "CT": "ct"}


def _quality_from_record(record: Any, pose_method: str = "") -> dict[str, dict[str, Any]]:
    """Per-slot quality for the HDF5 writer.

    The writer labels the ``reprojection_rms`` column by pose method
    (``rms_kind``: ``kabsch_mm`` for triangulation, ``reprojection_px``
    otherwise), and IMU mounting gates on that unit. The tracker reports the
    Kabsch residual as ``fit_rms_mm`` and keeps ``reprojection_rms_px`` in
    pixels, so pick the field that matches the label, never the other one.
    """
    out = {k: dict(v) for k, v in _EMPTY_QUALITY.items()}
    quality = getattr(record, "quality", None) or {}
    kabsch = str(pose_method) == "triangulation"
    for body, slot in _BODY_TO_SLOT.items():
        q = quality.get(body) or {}
        if kabsch:
            rms = q.get("fit_rms_mm", float("nan"))
        else:
            rms = q.get("reprojection_rms_px", q.get("reprojection_rms", float("nan")))
        try:
            rms_f = float(rms)
        except (TypeError, ValueError):
            rms_f = float("nan")
        out[slot] = {
            "n_tags": int(q.get("n_tags") or 0),
            "n_cameras": int(q.get("n_cameras") or 0),
            "reprojection_rms": rms_f,
        }
    return out


class OpticalTrackBridge:
    """Non-blocking ArUco pipeline used by :class:`SessionSupervisor`."""

    # Half of a 60 Hz period: later than one shutter's jitter, earlier than the
    # next pulse. Overridden from camera expect_rate_hz in __init__.
    _MATCH_NS = 8_333_000
    # Wall-clock safety if the other camera never produces a nearby frame.
    _PENDING_HOLD_NS = 1_000_000_000  # 1 s
    _PENDING_MAX = 120  # per camera

    def __init__(
        self,
        config_path: Path,
        cameras: list[CameraCapture],
        *,
        queue_size: int = 8,
        track_hz: float | None = None,
    ) -> None:
        from optical.calibration.io import CameraIntrinsics
        from optical.tracking.tracker import ArucoJawTracker, TrackerConfig

        active_ids = [c.cam_id for c in cameras if not c.is_mock]
        cfg = TrackerConfig.from_json(
            config_path,
            active_cam_ids=active_ids or None,
        )
        self.config_path = Path(config_path)
        self.tracker = ArucoJawTracker(cfg)
        self.cam_ids = set(self.tracker.cam_ids)
        self._profile_cam_ids = list(cfg.cam_ids)
        self._pose_method = cfg.pose_method
        self._scale_intrinsics(cameras, CameraIntrinsics)
        self._init_pairing(cameras)
        rates = [
            float(c.expect_rate_hz)
            for c in cameras
            if getattr(c, "expect_rate_hz", 0)
        ]
        self._camera_hz = max(rates) if rates else 30.0
        self._track_hz = None if track_hz is None else float(track_hz)
        self._track_stride = track_stride(self._track_hz, self._camera_hz)
        self._pairs_seen = 0

        self.stats: Counter[str] = Counter()
        self.stats["track_stride"] = int(self._track_stride)
        self._frame_idx = 0
        self._jobs: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        # (t_master_us, pose_mta, pose_cra, pose_ct, quality, corners)
        self._results: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._calibration_sent = False  # camera model handed to the writer once
        self._stopped = False
        self._overlay_lock = threading.Lock()
        self._overlay_markers: dict[str, list[tuple[int, np.ndarray]]] = {}
        self._overlay_src_wh: dict[str, tuple[int, int]] = {}
        self._worker = threading.Thread(
            target=self._worker_main, name="optical-track", daemon=True
        )
        self._worker.start()
        self._ct_tip_body_mm: np.ndarray | None = None

    def _init_pairing(self, cameras: list[CameraCapture] | None = None) -> None:
        rates = [
            float(c.expect_rate_hz)
            for c in (cameras or ())
            if getattr(c, "expect_rate_hz", 0)
        ]
        hz = max(rates) if rates else 60.0
        self._MATCH_NS = max(1_000_000, int(round(1_000_000_000.0 / (2.0 * hz))))
        self._pending: dict[str, list[_RgbFrame]] = {cid: [] for cid in self.cam_ids}
        self._newest_ns: dict[str, int] = {}

    def set_ct_tip_body_mm(self, tip_body_mm: np.ndarray | None) -> None:
        """Apply removable tip offset so ``pose_ct`` reports the tip pose."""
        if tip_body_mm is None:
            self._ct_tip_body_mm = None
            return
        self._ct_tip_body_mm = np.asarray(tip_body_mm, dtype=np.float64).reshape(3)

    def session_meta(self, cameras: list[CameraCapture], config_path: Path) -> dict[str, str]:
        meta = {
            "optical_tracking": "true",
            "optical_config": str(config_path),
            "optical_pose_method": str(self._pose_method),
            "optical_cam_ids": ",".join(self._profile_cam_ids),
            "optical_pair_mode": "sensor_time",
            "optical_pair_match_ns": str(int(self._MATCH_NS)),
            "optical_track_stride": str(int(self._track_stride)),
        }
        if self._track_hz is not None:
            meta["optical_track_hz"] = f"{self._track_hz:g}"
        if self._track_stride > 1:
            meta["optical_track_hz_effective"] = f"{self._camera_hz / self._track_stride:g}"
        lores = next((c.lores_size for c in cameras if c.lores_size), None)
        if lores is not None:
            meta["optical_rgb_stream"] = "lores_y"
            meta["optical_rgb_size"] = f"{lores[0]}x{lores[1]}"
            fmt = next((c.lores_format for c in cameras if c.lores_size), "YUV420")
            meta["optical_lores_format"] = str(fmt)
        if self._ct_tip_body_mm is not None:
            tip = self._ct_tip_body_mm
            meta["ct_tip_body_mm"] = f"{tip[0]:.6f},{tip[1]:.6f},{tip[2]:.6f}"
        return meta

    def wants_cam(self, cam_id: str) -> bool:
        return cam_id in self.cam_ids

    def overlay_markers(
        self, cam_id: str
    ) -> tuple[list[tuple[int, np.ndarray]], tuple[int, int]] | None:
        """Latest tracker corners for ``cam_id`` (preview boxes; no re-detect)."""
        with self._overlay_lock:
            markers = self._overlay_markers.get(cam_id)
            src_wh = self._overlay_src_wh.get(cam_id)
            if markers is None or src_wh is None:
                return None
            return markers, src_wh

    def _detected_corners(self) -> list[tuple[str, int, np.ndarray]]:
        """This frame's detections as ``(cam_id, tag_id, (4, 2) px)``.

        Read on the worker thread right after ``process_frame``, so they are the
        detections the frame's poses came from (full-image pixels: the tracker
        shifts region-of-interest hits back).
        """
        dets = getattr(self.tracker, "last_detections", None) or {}
        return [
            (str(cam_id), int(m.marker_id),
             np.array(m.corners, dtype=np.float32, copy=True).reshape(4, 2))
            for cam_id in sorted(dets)
            for m in getattr(dets[cam_id], "markers", ())
        ]

    def _store_overlay(self, frames: dict[str, Any]) -> None:
        dets = getattr(self.tracker, "last_detections", None) or {}
        overlay: dict[str, list[tuple[int, np.ndarray]]] = {}
        sizes: dict[str, tuple[int, int]] = {}
        for cam_id, det in dets.items():
            img = frames.get(cam_id)
            if img is None:
                continue
            overlay[cam_id] = [
                (int(m.marker_id), np.array(m.corners, dtype=np.float64, copy=True))
                for m in getattr(det, "markers", ())
            ]
            sizes[cam_id] = (int(img.shape[1]), int(img.shape[0]))
        with self._overlay_lock:
            self._overlay_markers = overlay
            self._overlay_src_wh = sizes

    def _ref_cam(self) -> str:
        for cid in getattr(self, "_profile_cam_ids", ()):
            if cid in self.cam_ids:
                return cid
        return sorted(self.cam_ids)[0]

    def note_rgb(
        self,
        *,
        cam_id: str,
        t_master_us: int,
        rgb: Any,
        sensor_timestamp_ns: int,
    ) -> None:
        """Buffer one camera's lores RGB and pair it with the other camera by time.

        A stereo bundle is enqueued only when every configured camera has a
        frame whose sensor timestamps fall within half a frame period.
        ``t_master_us`` is stored for the HDF5 stamp (reference camera); it is
        not the pairing key. ``rgb`` is the lores Y plane (HxW uint8).
        """
        if cam_id not in self.cam_ids or rgb is None:
            return
        self.stats["bundled"] += 1
        q = self._pending.setdefault(cam_id, [])
        frame = _RgbFrame(
            cam_id=cam_id,
            sensor_ns=int(sensor_timestamp_ns),
            t_master_us=int(t_master_us),
            rgb=rgb,
            noted_ns=time.monotonic_ns(),
        )
        q.append(frame)
        self._newest_ns[cam_id] = frame.sensor_ns
        self._try_pair()
        self._drop_stale_pending()

    def flush(self, *, force: bool = False) -> None:
        """Pair any complete stereo sets; drop leftover 1-camera frames by age / cap / shutdown."""
        self._try_pair()
        self._drop_stale_pending(force=force)

    def _try_pair(self) -> None:
        if not self.cam_ids:
            return
        if len(self.cam_ids) == 1:
            cam_id = next(iter(self.cam_ids))
            q = self._pending.setdefault(cam_id, [])
            while q:
                frame = q.pop(0)
                self._submit_pair(
                    frame.t_master_us,
                    {cam_id: frame.rgb},
                    {cam_id: frame.sensor_ns},
                )
            return

        ref = self._ref_cam()
        others = [cid for cid in sorted(self.cam_ids) if cid != ref]
        qref = self._pending.setdefault(ref, [])
        i = 0
        while i < len(qref):
            fr = qref[i]
            picked: dict[str, int] = {}
            ok = True
            for oc in others:
                best_j: int | None = None
                best_err: int | None = None
                for j, fo in enumerate(self._pending.get(oc, ())):
                    err = abs(int(fo.sensor_ns) - int(fr.sensor_ns))
                    if best_err is None or err < best_err:
                        best_err = err
                        best_j = j
                if best_j is None or best_err is None or best_err > int(self._MATCH_NS):
                    ok = False
                    break
                picked[oc] = best_j
            if not ok:
                i += 1
                continue
            frames = {ref: fr.rgb}
            t_cam = {ref: fr.sensor_ns}
            qref.pop(i)
            for oc in others:
                # Each other camera has its own list; indices are independent.
                fo = self._pending[oc].pop(picked[oc])
                frames[oc] = fo.rgb
                t_cam[oc] = fo.sensor_ns
            self._submit_pair(fr.t_master_us, frames, t_cam)

    def _submit_pair(
        self,
        t_master_us: int,
        frames: dict[str, Any],
        t_cam_ns: dict[str, int],
    ) -> None:
        """Count a stereo pair, then detect it or skip it.

        Skipped pairs are not drops: both cameras delivered the shutter, and
        live tracking only asked for every ``_track_stride``-th one.
        """
        self.stats["paired"] += 1
        stride = int(getattr(self, "_track_stride", 1) or 1)
        seen = int(getattr(self, "_pairs_seen", 0))
        self._pairs_seen = seen + 1
        if stride > 1 and (seen % stride) != 0:
            self.stats["track_skipped"] += 1
            return
        self._enqueue(t_master_us, frames, t_cam_ns)

    def _drop_stale_pending(self, *, force: bool = False) -> None:
        """Drop unmatched RGB. Never send 1-camera bundles to triangulation."""
        if force:
            for q in self._pending.values():
                self.stats["pair_dropped"] += len(q)
                q.clear()
            return
        now = time.monotonic_ns()
        hold_ns = int(self._PENDING_HOLD_NS)
        match_ns = int(self._MATCH_NS)
        max_n = int(self._PENDING_MAX)
        for cam_id, q in self._pending.items():
            keep: list[_RgbFrame] = []
            for frame in q:
                stale = (now - frame.noted_ns) > hold_ns
                if not stale:
                    for oc in self.cam_ids:
                        if oc == cam_id:
                            continue
                        newest = self._newest_ns.get(oc)
                        if newest is not None and newest > frame.sensor_ns + match_ns:
                            stale = True
                            break
                if stale:
                    self.stats["pair_dropped"] += 1
                else:
                    keep.append(frame)
            overflow = len(keep) - max_n
            if overflow > 0:
                self.stats["pair_dropped"] += overflow
                keep = keep[overflow:]
            self._pending[cam_id] = keep

    def corner_calibration(self) -> dict[str, dict[str, Any]]:
        """Camera model the tracker uses (intrinsics scaled to the tracked image)."""
        out: dict[str, dict[str, Any]] = {}
        intrinsics = getattr(self.tracker, "intrinsics", {}) or {}
        extrinsics = getattr(self.tracker, "extrinsics", {}) or {}
        for cam_id in getattr(self.tracker, "cam_ids", ()) or ():
            intr, extr = intrinsics.get(cam_id), extrinsics.get(cam_id)
            if intr is None or extr is None:
                continue
            out[str(cam_id)] = {
                "K": intr.K,
                "dist": intr.dist,
                "image_size": tuple(intr.image_size),
                "rms_px": getattr(intr, "rms_px", None),
                "T_world_cam": extr.T_world_cam,
                "extrinsics_rms_px": getattr(extr, "rms_px", None),
            }
        return out

    def _send_calibration(self, writer: Any) -> None:
        if getattr(self, "_calibration_sent", False):
            return
        self._calibration_sent = True
        store = getattr(writer, "set_optical_calibration", None)
        if store is None:
            return
        try:
            store(self.corner_calibration())
        except Exception as exc:  # noqa: BLE001 - the take goes on without it
            print(f"warning: optical calibration not stored with the corners: {exc}",
                  file=sys.stderr)

    def drain_to_writer(self, writer: Any) -> None:
        """Pull finished poses (and the corners they came from) into the HDF5 writer."""
        self._send_calibration(writer)
        while True:
            try:
                t_us, pose_mta, pose_cra, pose_ct, quality, corners = self._results.get_nowait()
            except queue.Empty:
                return
            writer.append_optical(
                t_master_us=t_us,
                pose_mta=pose_mta,
                pose_cra=pose_cra,
                pose_ct=pose_ct,
                quality=quality,
                corners=corners,
            )
            self.stats["written"] += 1

    def shutdown(self, writer: Any, *, timeout_s: float = 10.0) -> None:
        """Finish pending jobs and stop the worker (idempotent)."""
        if self._stopped:
            self.drain_to_writer(writer)
            return
        self._stopped = True
        self.flush(force=True)
        try:
            self._jobs.put(None, timeout=5.0)
        except Exception:
            pass
        deadline = time.monotonic() + timeout_s
        while self._worker.is_alive() and time.monotonic() < deadline:
            self.drain_to_writer(writer)
            self._worker.join(timeout=0.05)
        self.drain_to_writer(writer)
        if self._worker.is_alive():
            try:
                self._jobs.put_nowait(None)
            except Exception:
                pass
            self._worker.join(timeout=1.0)

    def _scale_intrinsics(self, cameras: list[CameraCapture], CameraIntrinsics: Any) -> None:
        """Scale each calibrated K to the lores stream the tracker actually sees.

        Uses optical's scale_camera_intrinsics: principal point
        ``c' = (c + 0.5)*s - 0.5`` (the old ``c*s`` was 0.25 px off at half
        resolution), scaled from the calibration's own image size (not the main
        stream's), and it refuses an aspect-ratio change (a crop / other sensor
        mode) instead of producing a silently wrong K.
        """
        from optical.calibration.io import scale_camera_intrinsics

        by_id = {c.cam_id: c for c in cameras}
        scaled: dict[str, Any] = {}
        for cam_id, intr in self.tracker.intrinsics.items():
            cam = by_id.get(cam_id)
            if cam is None or cam.lores_size is None:
                scaled[cam_id] = intr
                continue
            src = tuple(int(v) for v in (getattr(intr, "image_size", None) or (0, 0)))
            if len(src) != 2 or src[0] <= 0 or src[1] <= 0:
                # Older bundles without image_size: calibrated at the main stream size.
                intr = CameraIntrinsics(
                    K=np.asarray(intr.K, dtype=np.float64).reshape(3, 3).copy(),
                    dist=np.asarray(intr.dist, dtype=np.float64).copy(),
                    image_size=(int(cam.resolution[0]), int(cam.resolution[1])),
                    rms_px=intr.rms_px,
                )
            dst = (int(cam.lores_size[0]), int(cam.lores_size[1]))
            scaled[cam_id] = scale_camera_intrinsics(intr, dst)
        self.tracker.intrinsics = scaled

    def _enqueue(
        self,
        t_master_us: int,
        frames: dict[str, Any],
        t_cam_ns: dict[str, int],
    ) -> None:
        job = (self._frame_idx, int(t_master_us), frames, t_cam_ns)
        self._frame_idx += 1
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()
                self.stats["queue_dropped"] += 1
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                self.stats["queue_dropped"] += 1
                return
        self.stats["enqueued"] += 1
        depth = int(self._jobs.qsize())
        if depth > int(self.stats.get("queue_depth_max", 0)):
            self.stats["queue_depth_max"] = depth

    def _worker_main(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                break
            fidx, t_us, frames, t_cam = job
            try:
                record = self.tracker.process_frame(
                    frames,
                    frame_idx=fidx,
                    t_cam_ns=t_cam,
                    t_master_us=t_us,
                )
                self._store_overlay(frames)
                corners = self._detected_corners()
                pose_ct = record.pose_CT
                if self._ct_tip_body_mm is not None:
                    from optical.tracking.pivot import apply_tip_offset
                    from optical.tracking.pose import is_valid_pose

                    if is_valid_pose(pose_ct):
                        pose_ct = apply_tip_offset(pose_ct, self._ct_tip_body_mm)
                self._results.put(
                    (int(t_us), record.pose_MP, record.pose_HP, pose_ct,
                     _quality_from_record(record, self._pose_method), corners)
                )
                self.stats["processed"] += 1
            except Exception as exc:  # noqa: BLE001 — never kill the worker
                self.stats["failed"] += 1
                if self.stats["failed"] <= 3:
                    print(f"warning: optical process_frame failed: {exc}", file=sys.stderr)
                nan = np.full((4, 4), np.nan, dtype=np.float64)
                self._results.put(
                    (int(t_us), nan, nan, nan, {k: dict(v) for k, v in _EMPTY_QUALITY.items()}, [])
                )
