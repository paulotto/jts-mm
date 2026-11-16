#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Evaluate minimum usable ArUco marker size at a fixed working distance.

Runs two interactive captures against a test plate with multiple marker sizes
(typically 20–60 mm in 10 mm steps, each with a distinct ID):

1. **static** — plate held still (~50–60 cm): detection rate, pixel size, pose jitter
2. **motion** — slow movement (jaw-like): detection rate and triangulation reprojection

Requires a calibrated camera rig (HDF5 bundle + ``tracker_config``).
With **2 cameras**, triangulation metrics are reported; with **1 camera**,
only PnP metrics are used (triangulation skipped).

Example::

    python software/optical/optical/evaluation/run_marker_size_eval.py \\
        --config software/optical/etc/optical.yaml

    # Same eval at half-res (scales K; compare TriRep mm across runs)
    optical eval --config software/optical/etc/optical.yaml \\
        --resolution 728x544

    # Single-camera PnP (no triangulation)
    optical eval --config software/optical/etc/optical.yaml \\
        --n-cameras 1 --cameras cam0

Operator keys during capture: SPACE start phase, Q skip phase, ESC/Ctrl+C abort.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from optical.calibration.io import (
        CameraExtrinsics,
        CameraIntrinsics,
        load_cameras_from_bundle,
        read_calibration_bundle,
        scale_camera_intrinsics,
    )
    from optical.calibration.multicam import (
        CamSpec,
        PreviewServer,
        _TerminalKeys,
        _as_bgr,
        _draw_status_bar,
        _open_rig,
        _parse_cams,
        _stack_preview,
        enable_track_lores,
        half_lores_size,
    )
    from optical.evaluation.metrics import (
        MarkerEvalResult,
        MarkerFrameSample,
        MarkerTriSample,
        aggregate_marker_results,
        marker_side_px,
    )
    from optical.evaluation.triangulation_eval import triangulate_marker_pose
    from optical.paths import REPO_ROOT, resolve_config_path, resolve_repo_path
    from optical.tracking.aruco import ArucoDetector, DetectionResult, _detect_image
    from optical.tracking.pose import tag_object_points
    from optical.tracking.tracker import TrackerConfig
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from optical.calibration.io import (
        CameraExtrinsics,
        CameraIntrinsics,
        load_cameras_from_bundle,
        read_calibration_bundle,
        scale_camera_intrinsics,
    )
    from optical.calibration.multicam import (
        CamSpec,
        PreviewServer,
        _TerminalKeys,
        _as_bgr,
        _draw_status_bar,
        _open_rig,
        _parse_cams,
        _stack_preview,
        enable_track_lores,
        half_lores_size,
    )
    from optical.evaluation.metrics import (
        MarkerEvalResult,
        MarkerFrameSample,
        MarkerTriSample,
        aggregate_marker_results,
        marker_side_px,
    )
    from optical.evaluation.triangulation_eval import triangulate_marker_pose
    from optical.paths import REPO_ROOT, resolve_config_path, resolve_repo_path
    from optical.tracking.aruco import ArucoDetector, DetectionResult, _detect_image
    from optical.tracking.pose import tag_object_points
    from optical.tracking.tracker import TrackerConfig


@dataclass(frozen=True)
class MarkerSpec:
    marker_id: int
    size_mm: float
    label: str


def _load_json(path: Path) -> dict:
    from optical.config.loader import eval_section, load_config_file

    doc = load_config_file(path)
    try:
        return eval_section(doc, "eval")
    except KeyError:
        return doc


def parse_resolution(value: str | list | tuple | None) -> Optional[Tuple[int, int]]:
    """Parse ``WxH``, ``W,H``, or ``[W, H]`` into ``(width, height)``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"resolution must be [width, height], got {value!r}")
        w, h = int(value[0]), int(value[1])
    else:
        text = str(value).strip().lower().replace(" ", "")
        m = re.fullmatch(r"(\d+)[x,](\d+)", text)
        if not m:
            raise ValueError(
                f"resolution must look like 728x544 or 728,544, got {value!r}"
            )
        w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        raise ValueError(f"resolution dimensions must be positive, got {w}x{h}")
    return w, h


def _override_cam_resolution(cams: List[CamSpec], resolution: Tuple[int, int]) -> List[CamSpec]:
    return [replace(c, resolution=resolution) for c in cams]


def _scale_intrinsics_map(
    intrinsics: Dict[str, CameraIntrinsics],
    resolution: Tuple[int, int],
) -> Dict[str, CameraIntrinsics]:
    return {
        cid: scale_camera_intrinsics(intr, resolution)
        for cid, intr in intrinsics.items()
    }


def _resize_frames(
    frames: Dict[str, np.ndarray],
    resolution: Optional[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Force frames to ``resolution`` when the sensor returned a different size."""
    if resolution is None:
        return frames
    dst_w, dst_h = resolution
    out: Dict[str, np.ndarray] = {}
    for cam_id, rgb in frames.items():
        h, w = rgb.shape[:2]
        if (w, h) == (dst_w, dst_h):
            out[cam_id] = rgb
        else:
            out[cam_id] = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    return out


def _parse_markers(cfg: dict) -> List[MarkerSpec]:
    out: List[MarkerSpec] = []
    for entry in cfg["markers"]:
        mid = int(entry["id"])
        size_mm = float(entry["size_mm"])
        label = str(entry.get("label", f"{size_mm:g}mm"))
        out.append(MarkerSpec(marker_id=mid, size_mm=size_mm, label=label))
    return out


def _load_calibration(
    bundle_path: Path,
    cam_ids: List[str],
    *,
    require_extrinsics: bool = True,
):
    bundle = read_calibration_bundle(bundle_path)
    intrinsics, extrinsics = load_cameras_from_bundle(bundle)
    for cid in cam_ids:
        if cid not in intrinsics:
            raise KeyError(f"Camera {cid!r} missing intrinsics in {bundle_path}")
        if require_extrinsics and cid not in extrinsics:
            raise KeyError(f"Camera {cid!r} missing extrinsics in {bundle_path}")
    return intrinsics, extrinsics


def _parse_cameras_arg(value: str | list | tuple | None) -> Optional[List[str]]:
    """Parse ``cam0,cam1`` or a JSON list into camera ids."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        ids = [str(v).strip() for v in value if str(v).strip()]
    else:
        ids = [p.strip() for p in str(value).split(",") if p.strip()]
    if not ids:
        raise ValueError("cameras list is empty")
    if len(ids) > 2:
        raise ValueError(f"eval supports 1 or 2 cameras, got {ids!r}")
    return ids


def _resolve_cam_ids(
    *,
    cfg: dict,
    tracker_cam_ids: List[str],
    cal_cam_ids: List[str],
    cameras: str | list | tuple | None = None,
    n_cameras: int | None = None,
) -> List[str]:
    """
    Choose 1–2 cameras for the eval.

    Priority: ``--cameras`` → ``--n-cameras`` → config ``cam_ids`` →
    config ``n_cameras`` → tracker ``cam_ids`` → calibration cameras.
    """
    explicit = _parse_cameras_arg(cameras)
    if explicit is not None:
        return explicit

    n = n_cameras
    if n is None and cfg.get("n_cameras") is not None:
        n = int(cfg["n_cameras"])
    if n is not None:
        n = int(n)
        if n not in (1, 2):
            raise ValueError(f"n_cameras must be 1 or 2, got {n}")
        pool: List[str] = []
        for src in (cfg.get("cam_ids"), tracker_cam_ids, cal_cam_ids):
            if not src:
                continue
            for cid in src:
                if cid not in pool:
                    pool.append(str(cid))
        if len(pool) < n:
            raise ValueError(
                f"Need {n} cameras but only found {pool!r} "
                f"(tracker={tracker_cam_ids}, calibration={cal_cam_ids})"
            )
        return pool[:n]

    if cfg.get("cam_ids"):
        ids = _parse_cameras_arg(cfg["cam_ids"])
        assert ids is not None
        return ids

    pool = list(tracker_cam_ids) if tracker_cam_ids else list(cal_cam_ids)
    if not pool:
        raise ValueError("No cameras listed in tracker_config or calibration_config")
    if len(pool) > 2:
        pool = pool[:2]
    return pool


def _estimate_pose(
    corners: np.ndarray,
    *,
    size_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    obj_pts = tag_object_points(size_mm).astype(np.float32)
    img_pts = np.asarray(corners, dtype=np.float32).reshape(4, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None, None, float("inf")
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    err = np.linalg.norm(img_pts.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(err ** 2)))
    # Object points are in mm → tvec is in mm (same units as tag_object_points).
    return rvec.reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3), rms


class LivePreviewFeed:
    """Background capture for all rig cameras; stacked MJPEG preview."""

    def __init__(
        self,
        rig,
        cams: List[CamSpec],
        cam_order: List[str],
        detector: ArucoDetector,
        markers: List[MarkerSpec],
        preview: Optional[PreviewServer],
        *,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._rig = rig
        self._cam_order = cam_order
        self._detector = detector
        self._marker_labels = {m.marker_id: m.label for m in markers}
        self._preview = preview
        self._target_resolution = target_resolution
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: Dict[str, np.ndarray] = {}
        self._seq = 0  # increments once per captured (timestamp-paired) frame set
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status = "Live preview"
        self._warned_resize = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_status(self, text: str) -> None:
        self._status = text

    def latest_frames(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {cid: frame.copy() for cid, frame in self._latest.items()}

    def latest_frames_seq(
        self, after_seq: int = -1, timeout_s: float = 2.0
    ) -> Tuple[int, Dict[str, np.ndarray]]:
        """Wait for a frame set newer than ``after_seq``; return ``(seq, frames)``.

        On timeout returns ``(after_seq, {})``. Lets the recorder process every
        captured set at most once (``latest_frames`` can return the same set
        twice when detection is faster than capture).
        """
        with self._new_frame:
            self._new_frame.wait_for(lambda: self._seq > after_seq, timeout=timeout_s)
            if self._seq <= after_seq:
                return after_seq, {}
            return self._seq, {cid: frame.copy() for cid, frame in self._latest.items()}

    def _loop(self) -> None:
        while self._running:
            # Pair cameras by SensorTimestamp (same trigger), not by read order.
            frames, _meta = self._rig.capture_all_synced()
            if not frames:
                time.sleep(0.005)
                continue
            if self._target_resolution is not None:
                sample = next(iter(frames.values()))
                h, w = sample.shape[:2]
                if (w, h) != self._target_resolution and not self._warned_resize:
                    tw, th = self._target_resolution
                    print(
                        f"warning: camera returned {w}x{h}; "
                        f"resizing to {tw}x{th} for evaluation"
                    )
                    self._warned_resize = True
                frames = _resize_frames(frames, self._target_resolution)
            with self._new_frame:
                self._latest = frames
                self._seq += 1
                self._new_frame.notify_all()

            overlay: Dict[str, np.ndarray] = {}
            for cam_id in self._cam_order:
                rgb = frames.get(cam_id)
                if rgb is None:
                    continue
                det = _detect_image(self._detector, rgb)
                det_by_id = {m.marker_id: m.corners for m in det.markers}
                overlay[cam_id] = _draw_markers(rgb, det_by_id, self._marker_labels)

            if self._preview is not None:
                stacked = _stack_preview(overlay, self._cam_order)
                if stacked is not None:
                    preview_frame = _draw_status_bar(stacked, self._status, (0, 220, 0))
                    # Frames are BGR (picamera2 "RGB888") or gray -> _as_bgr; update() wants BGR.
                    self._preview.update(preview_frame)


def _draw_markers(
    rgb: np.ndarray,
    detections: Dict[int, np.ndarray],
    marker_labels: Dict[int, str],
) -> np.ndarray:
    """Draw marker quads on a single camera frame (gray frames become BGR first)."""
    disp = _as_bgr(rgb).copy()
    for mid, corners in detections.items():
        pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(disp, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        x, y = pts[0, 0]
        cv2.putText(
            disp,
            marker_labels.get(mid, str(mid)),
            (int(x), max(int(y) - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return disp


def _wait_for_key(prompt: str, *, accept: set[int]) -> int:
    print(prompt)
    with _TerminalKeys() as keys:
        while True:
            key = keys.poll()
            if key is None:
                time.sleep(0.02)
                continue
            if key in accept:
                return key
            if key == 27:
                raise SystemExit("Aborted.")


def _wait_start(phase_name: str, *, hint: str) -> bool:
    """Prompt operator; return True to run phase, False to skip."""
    key = _wait_for_key(
        f"\n[{phase_name}] {hint}\n  SPACE = starten, Q = überspringen",
        accept={ord(" "), ord("q"), ord("Q")},
    )
    return key == ord(" ")


def _record_frames(
    *,
    phase_name: str,
    n_frames: int,
    warmup_frames: int,
    feed: LivePreviewFeed,
    detector: ArucoDetector,
    markers: List[MarkerSpec],
    cam_ids: List[str],
    primary_cam_id: str,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    min_views: int,
    enable_triangulation: bool = True,
) -> tuple[
    Dict[int, List[bool]],
    Dict[int, List[MarkerFrameSample]],
    Dict[int, List[bool]],
    Dict[int, List[MarkerTriSample]],
]:
    detected_flags: Dict[int, List[bool]] = {m.marker_id: [] for m in markers}
    samples: Dict[int, List[MarkerFrameSample]] = {m.marker_id: [] for m in markers}
    tri_detected_flags: Dict[int, List[bool]] = {m.marker_id: [] for m in markers}
    tri_samples: Dict[int, List[MarkerTriSample]] = {m.marker_id: [] for m in markers}

    primary_intr = intrinsics[primary_cam_id]

    last_seq = -1
    if warmup_frames > 0:
        print(f"[{phase_name}] Warmup {warmup_frames} frames …")
        for _ in range(warmup_frames):
            last_seq, warm = feed.latest_frames_seq(last_seq)
            if not warm:  # camera stalled: don't wait 2 s per warmup frame
                break

    print(f"[{phase_name}] Aufnahme {n_frames} frames …")
    # Process n_frames *distinct* captured sets; each flag list entry is one set,
    # so detection rates use len(flags) as denominator.
    frame_idx = 0
    skipped = 0
    stalls = 0
    while frame_idx < n_frames:
        seq, frames = feed.latest_frames_seq(last_seq)
        if not frames:
            stalls += 1
            if stalls >= 5:
                print(f"[{phase_name}] no new frames for ~10 s — ending phase after {frame_idx} frames")
                break
            continue
        stalls = 0
        if last_seq >= 0 and seq > last_seq + 1:
            skipped += seq - last_seq - 1
        last_seq = seq

        feed.set_status(f"{phase_name} {frame_idx + 1}/{n_frames}")

        detections: Dict[str, DetectionResult] = {}
        det_by_cam: Dict[str, Dict[int, np.ndarray]] = {}
        for cam_id in cam_ids:
            rgb = frames.get(cam_id)
            if rgb is None:
                continue
            # Default capture is the lores Y plane (2-D); _detect_image handles gray and colour.
            det = _detect_image(detector, rgb)
            detections[cam_id] = det
            det_by_cam[cam_id] = {m.marker_id: m.corners for m in det.markers}

        for spec in markers:
            corners_primary = det_by_cam.get(primary_cam_id, {}).get(spec.marker_id)
            ok_primary = corners_primary is not None
            detected_flags[spec.marker_id].append(ok_primary)

            if ok_primary:
                rvec, tvec, rms = _estimate_pose(
                    corners_primary,
                    size_mm=spec.size_mm,
                    K=primary_intr.K,
                    dist=primary_intr.dist,
                )
                if rvec is not None and tvec is not None:
                    side_px = marker_side_px(corners_primary)
                    for cid in cam_ids:
                        c = det_by_cam.get(cid, {}).get(spec.marker_id)
                        if c is not None:
                            side_px = min(side_px, marker_side_px(c))
                    samples[spec.marker_id].append(
                        MarkerFrameSample(
                            frame_idx=frame_idx,
                            marker_id=spec.marker_id,
                            size_mm=spec.size_mm,
                            side_px=side_px,
                            tvec_mm=np.asarray(tvec, dtype=np.float64).reshape(3),
                            rvec=rvec,
                            reprojection_rms_px=rms,
                        )
                    )

            tri_ok = False
            if enable_triangulation:
                in_both = all(
                    spec.marker_id in det_by_cam.get(cid, {}) for cid in cam_ids
                )
                if in_both and len(detections) >= min_views:
                    tri = triangulate_marker_pose(
                        detections,
                        marker_id=spec.marker_id,
                        size_mm=spec.size_mm,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        min_views=min_views,
                    )
                    if tri is not None:
                        tri_ok = True
                        rvec_tag, _ = cv2.Rodrigues(tri.T_world_tag[:3, :3])
                        tri_samples[spec.marker_id].append(
                            MarkerTriSample(
                                frame_idx=frame_idx,
                                marker_id=spec.marker_id,
                                size_mm=spec.size_mm,
                                translation_mm=tri.translation_mm,
                                reprojection_rms_px=tri.reprojection_rms_px,
                                fit_rms_mm=tri.fit_rms_mm,
                                rotation_rvec=rvec_tag.reshape(3),
                            )
                        )
                tri_detected_flags[spec.marker_id].append(tri_ok)

        if (frame_idx + 1) % 50 == 0:
            print(f"  … {frame_idx + 1}/{n_frames}")
        frame_idx += 1

    if skipped:
        print(
            f"[{phase_name}] {skipped} camera frames arrived while detection was busy "
            "and were not evaluated (rates use evaluated frames only)."
        )
    return detected_flags, samples, tri_detected_flags, tri_samples


def _pass_keys(*, evaluate_jitter: bool, enable_triangulation: bool = True) -> tuple[str, ...]:
    if evaluate_jitter:
        keys = ("detection_rate", "translation_jitter", "pixel_size")
        if enable_triangulation:
            keys += (
                "tri_detection_rate",
                "tri_reprojection",
                "tri_translation_jitter",
                "tri_rotation_jitter",
            )
        return keys
    if enable_triangulation:
        return ("detection_rate", "tri_detection_rate", "tri_reprojection")
    return ("detection_rate",)


def _print_results(
    phase: str,
    results: List[MarkerEvalResult],
    thresholds: dict,
    tri_thresholds: Optional[dict],
    *,
    evaluate_jitter: bool = True,
    enable_triangulation: bool = True,
) -> None:
    pass_keys = _pass_keys(
        evaluate_jitter=evaluate_jitter,
        enable_triangulation=enable_triangulation,
    )
    if evaluate_jitter:
        if enable_triangulation:
            print(f"\n{'=' * 96}")
            print(f"Ergebnisse — {phase}")
            print(f"{'=' * 96}")
            print(
                f"{'Label':<7} {'mm':>4} {'Det%':>6} {'Tri%':>6} {'px':>5} "
                f"{'PnP J mm':>9} {'Tri J mm':>9} {'TriRepx':>8} {'TriFit':>7} {'TriRot°':>7} {'OK':>4}"
            )
            print("-" * 96)
            for r in sorted(results, key=lambda x: x.size_mm, reverse=True):
                ok = all(r.passes.get(k, False) for k in pass_keys)
                print(
                    f"{r.label:<7} {r.size_mm:4.0f} {r.detection_rate_pct:5.1f}% "
                    f"{r.tri_detection_rate_pct:5.1f}% {r.min_side_px:5.1f} "
                    f"{r.translation_std_mm:9.3f} {r.tri_translation_std_mm:9.3f} "
                    f"{r.tri_reprojection_rms_px_mean:8.3f} {r.tri_fit_rms_mm_mean:7.3f} "
                    f"{r.tri_rotation_jitter_deg:7.2f} {'yes' if ok else 'no':>4}"
                )
        else:
            print(f"\n{'=' * 72}")
            print(f"Ergebnisse — {phase} (1 Kamera / PnP)")
            print(f"{'=' * 72}")
            print(
                f"{'Label':<7} {'mm':>4} {'Det%':>6} {'px':>5} "
                f"{'PnP J mm':>9} {'PnPpx':>7} {'PnPmm':>7} {'OK':>4}"
            )
            print("-" * 72)
            for r in sorted(results, key=lambda x: x.size_mm, reverse=True):
                ok = all(r.passes.get(k, False) for k in pass_keys)
                print(
                    f"{r.label:<7} {r.size_mm:4.0f} {r.detection_rate_pct:5.1f}% "
                    f"{r.min_side_px:5.1f} {r.translation_std_mm:9.3f} "
                    f"{r.reprojection_rms_px_mean:7.3f} {r.reprojection_rms_mm_mean:7.3f} "
                    f"{'yes' if ok else 'no':>4}"
                )
        print(
            f"\nPnP ({thresholds.get('primary_cam', 'cam0')}): Det ≥ "
            f"{thresholds.get('min_detection_rate_pct', '?')}%, "
            f"Jitter ≤ {thresholds.get('max_translation_jitter_mm', '?')} mm"
        )
        if enable_triangulation and tri_thresholds:
            print(
                f"Triangulation ({', '.join(tri_thresholds.get('cam_ids', ['cam0', 'cam1']))}): "
                f"Det ≥ {tri_thresholds.get('min_detection_rate_pct', '?')}%, "
                f"Reproj ≤ {tri_thresholds.get('max_reprojection_rms_px', '?')} px, "
                f"Trans ≤ {tri_thresholds.get('max_translation_jitter_mm', '?')} mm, "
                f"Rot ≤ {tri_thresholds.get('max_rotation_jitter_deg', '?')}°"
            )
        return

    if enable_triangulation:
        print(f"\n{'=' * 80}")
        print(f"Ergebnisse — {phase}")
        print(f"{'=' * 80}")
        print(
            f"{'Label':<7} {'mm':>4} {'Det%':>6} {'Tri%':>6} {'Gap':>5} "
            f"{'PnPpx':>7} {'PnPmm':>7} {'Tripx':>7} {'Trimm':>7} {'OK':>4}"
        )
        print("-" * 80)
        for r in sorted(results, key=lambda x: x.size_mm, reverse=True):
            ok = all(r.passes.get(k, False) for k in pass_keys)
            print(
                f"{r.label:<7} {r.size_mm:4.0f} {r.detection_rate_pct:5.1f}% "
                f"{r.tri_detection_rate_pct:5.1f}% {r.longest_gap_frames:5d} "
                f"{r.reprojection_rms_px_mean:7.3f} {r.reprojection_rms_mm_mean:7.3f} "
                f"{r.tri_reprojection_rms_px_mean:7.3f} {r.tri_reprojection_rms_mm_mean:7.3f} "
                f"{'yes' if ok else 'no':>4}"
            )
        print(
            f"\nDet ≥ {thresholds.get('min_detection_rate_pct', '?')}%, "
            f"Tri-Det ≥ {tri_thresholds.get('min_detection_rate_pct', '?') if tri_thresholds else '?'}%, "
            f"Tri-Reproj ≤ {tri_thresholds.get('max_reprojection_rms_px', '?') if tri_thresholds else '?'} px "
            f"(compare Trimm across resolutions)"
        )
        return

    print(f"\n{'=' * 72}")
    print(f"Ergebnisse — {phase} (1 Kamera / PnP)")
    print(f"{'=' * 72}")
    print(
        f"{'Label':<7} {'mm':>4} {'Det%':>6} {'Gap':>5} "
        f"{'PnPpx':>7} {'PnPmm':>7} {'OK':>4}"
    )
    print("-" * 72)
    for r in sorted(results, key=lambda x: x.size_mm, reverse=True):
        ok = all(r.passes.get(k, False) for k in pass_keys)
        print(
            f"{r.label:<7} {r.size_mm:4.0f} {r.detection_rate_pct:5.1f}% "
            f"{r.longest_gap_frames:5d} "
            f"{r.reprojection_rms_px_mean:7.3f} {r.reprojection_rms_mm_mean:7.3f} "
            f"{'yes' if ok else 'no':>4}"
        )
    print(
        f"\nDet ≥ {thresholds.get('min_detection_rate_pct', '?')}% "
        f"(compare PnPmm across resolutions)"
    )


def _recommend_size(
    results: List[MarkerEvalResult],
    *,
    evaluate_jitter: bool = True,
    enable_triangulation: bool = True,
) -> Optional[float]:
    passing = []
    pass_keys = _pass_keys(
        evaluate_jitter=evaluate_jitter,
        enable_triangulation=enable_triangulation,
    )
    for r in results:
        if all(r.passes.get(k, False) for k in pass_keys):
            passing.append(r)
    if not passing:
        return None
    return min(r.size_mm for r in passing)


def _tri_thresholds(
    phase_thresholds: dict,
    cam_ids: List[str],
    *,
    evaluate_jitter: bool = True,
) -> dict:
    tri = dict(phase_thresholds.get("triangulation", {}))
    tri.setdefault("cam_ids", cam_ids)
    tri.setdefault("min_detection_rate_pct", phase_thresholds.get("min_detection_rate_pct", 90))
    tri.setdefault("max_reprojection_rms_px", 2.0)
    if evaluate_jitter:
        tri.setdefault("max_translation_jitter_mm", phase_thresholds.get("max_translation_jitter_mm", 0.5))
        tri.setdefault("max_rotation_jitter_deg", 1.0)
    return tri


def run_marker_size_eval(
    *,
    config: str | Path,
    backend: str = "picamera2",
    preview_host: str = "0.0.0.0",
    preview_port: int = 5000,
    output: str | Path | None = None,
    skip_motion: bool = False,
    resolution: str | list | tuple | None = None,
    cameras: str | list | tuple | None = None,
    n_cameras: int | None = None,
) -> Path:
    """Run static (+ optional motion) marker-size evaluation; write a JSON report."""
    cfg_path = resolve_config_path(config) if not Path(config).is_absolute() else Path(config)
    if not cfg_path.is_file():
        cfg_path = resolve_repo_path(config)
    cfg = _load_json(cfg_path)

    markers = _parse_markers(cfg)
    dictionary = str(cfg.get("dictionary", "DICT_4X4_50"))

    # Prefer the same optical.yaml for tracker + calibration sections.
    tracker_cfg_path = cfg_path
    if cfg.get("tracker_config"):
        tracker_cfg_path = resolve_config_path(cfg["tracker_config"])
    tcfg = TrackerConfig.from_json(tracker_cfg_path)

    cal_cfg_path = tracker_cfg_path
    if cfg.get("calibration_config"):
        cal_cfg_path = resolve_config_path(cfg["calibration_config"])
    from optical.config.loader import calibration_section, load_config_file

    try:
        cal_cfg = calibration_section(load_config_file(cal_cfg_path), config_path=cal_cfg_path)
    except KeyError:
        cal_cfg = _load_json(cal_cfg_path)
    all_cams = _parse_cams(cal_cfg)
    cal_cam_ids = [c.cam_id for c in all_cams]

    cam_ids = _resolve_cam_ids(
        cfg=cfg,
        tracker_cam_ids=list(tcfg.cam_ids),
        cal_cam_ids=cal_cam_ids,
        cameras=cameras,
        n_cameras=n_cameras,
    )
    enable_triangulation = len(cam_ids) >= 2
    if enable_triangulation and tcfg.pose_method != "triangulation":
        print(
            f"Warning: tracker pose_method={tcfg.pose_method!r}; "
            "2-camera evaluation still reports triangulation metrics."
        )
    min_views = tcfg.triangulation_min_views if enable_triangulation else 1
    primary_cam_id = str(cfg.get("camera_id", cam_ids[0]))
    if primary_cam_id not in cam_ids:
        primary_cam_id = cam_ids[0]

    cams = [c for c in all_cams if c.cam_id in cam_ids]
    if len(cams) != len(cam_ids):
        missing = set(cam_ids) - {c.cam_id for c in cams}
        raise KeyError(f"Cameras {missing} missing in {cal_cfg_path}")
    # Keep cam_ids order for stacking / reports.
    by_id = {c.cam_id: c for c in cams}
    cams = [by_id[cid] for cid in cam_ids]

    cap = cfg.get("capture", {})
    # CLI --resolution overrides config. Default (null / half / omitted) → lores
    # stream like optical track / jtsmm record --optical-track.
    target_resolution = parse_resolution(
        resolution if resolution is not None else cap.get("resolution", cfg.get("resolution"))
    )
    native_resolution = cams[0].resolution
    lores = half_lores_size(native_resolution)
    if target_resolution is None or target_resolution == lores:
        enable_track_lores(cams)
        capture_resolution = lores
        print(
            f"Capture stream: lores_y {capture_resolution[0]}x{capture_resolution[1]} "
            f"(main sensor {native_resolution[0]}x{native_resolution[1]})"
        )
    elif target_resolution == native_resolution:
        capture_resolution = native_resolution
        print(f"Capture stream: full-res main {capture_resolution[0]}x{capture_resolution[1]}")
    else:
        cams = _override_cam_resolution(cams, target_resolution)
        capture_resolution = target_resolution
        print(f"Capture stream: main override {capture_resolution[0]}x{capture_resolution[1]}")

    bundle_path = resolve_repo_path(
        cfg.get("calibration_bundle", str(tcfg.calibration_bundle_path))
    )
    intrinsics, extrinsics = _load_calibration(
        bundle_path,
        cam_ids,
        require_extrinsics=enable_triangulation,
    )
    calib_sizes = {cid: tuple(intr.image_size) for cid, intr in intrinsics.items() if cid in cam_ids}
    if any(sz != capture_resolution for sz in calib_sizes.values()):
        print(
            f"Scaling intrinsics from calibration sizes {calib_sizes} → "
            f"{capture_resolution[0]}x{capture_resolution[1]}"
        )
        intrinsics = _scale_intrinsics_map(
            {cid: intrinsics[cid] for cid in cam_ids},
            capture_resolution,
        )

    static_frames = int(cap.get("static_frames", 300))
    motion_frames = int(cap.get("motion_frames", 300))
    warmup = int(cap.get("warmup_frames", 30))

    thresholds_static = cfg.get("thresholds", {}).get("static", cfg.get("thresholds", {}))
    thresholds_static.setdefault("primary_cam", primary_cam_id)
    thresholds_motion = cfg.get("thresholds", {}).get(
        "motion",
        {
            "min_detection_rate_pct": thresholds_static.get("min_detection_rate_pct", 95) - 10,
            "min_marker_side_px": thresholds_static.get("min_marker_side_px", 20),
        },
    )
    thresholds_motion.setdefault("primary_cam", primary_cam_id)
    tri_static = (
        _tri_thresholds(thresholds_static, cam_ids, evaluate_jitter=True)
        if enable_triangulation
        else None
    )
    tri_motion = (
        _tri_thresholds(thresholds_motion, cam_ids, evaluate_jitter=False)
        if enable_triangulation
        else None
    )

    detector = ArucoDetector(dictionary)
    rig = _open_rig(cams, backend)

    preview: Optional[PreviewServer] = None
    if preview_port > 0:
        preview = PreviewServer(preview_host, preview_port)
        preview.start()
        print(f"Preview: {preview.url}")

    mode_label = (
        "2-Kamera Triangulation" if enable_triangulation else "1-Kamera PnP"
    )
    print("=" * 96)
    print(f"ArUco Marker-Größen Evaluation ({mode_label})")
    print(f"Kameras: {', '.join(cam_ids)} | PnP-Referenz: {primary_cam_id}")
    print(f"Auflösung: {capture_resolution[0]}x{capture_resolution[1]}")
    print(f"Marker: {', '.join(m.label for m in markers)}")
    print(f"Tracker: {tracker_cfg_path}")
    print(f"Kalibrierung: {bundle_path}")
    print("=" * 96)

    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "tracker_config": str(tracker_cfg_path),
        "calibration_bundle": str(bundle_path),
        "cam_ids": cam_ids,
        "n_cameras": len(cam_ids),
        "enable_triangulation": enable_triangulation,
        "primary_cam_id": primary_cam_id,
        "resolution": [int(capture_resolution[0]), int(capture_resolution[1])],
        "calibration_image_sizes": {cid: list(sz) for cid, sz in calib_sizes.items()},
        "min_views": min_views,
        "dictionary": dictionary,
        "markers": [{"id": m.marker_id, "size_mm": m.size_mm, "label": m.label} for m in markers],
        "phases": {},
    }

    feed = LivePreviewFeed(
        rig,
        cams,
        cam_ids,
        detector,
        markers,
        preview,
        target_resolution=capture_resolution,
    )
    feed.set_status("Live preview — SPACE startet Aufnahme")
    feed.start()

    visibility_hint = (
        "alle Größen in beiden Kameras sichtbar"
        if enable_triangulation
        else f"alle Größen in {primary_cam_id} sichtbar"
    )

    try:
        if _wait_start(
            "static",
            hint=f"Testplatte bei 50–60 cm ruhig halten ({visibility_hint}).",
        ):
            static_flags, static_samples, static_tri_flags, static_tri_samples = _record_frames(
                phase_name="static",
                n_frames=static_frames,
                warmup_frames=warmup,
                feed=feed,
                detector=detector,
                markers=markers,
                cam_ids=cam_ids,
                primary_cam_id=primary_cam_id,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                min_views=min_views,
                enable_triangulation=enable_triangulation,
            )
            static_results = [
                aggregate_marker_results(
                    marker_id=spec.marker_id,
                    size_mm=spec.size_mm,
                    label=spec.label,
                    total_frames=len(static_flags[spec.marker_id]),
                    detected_flags=static_flags[spec.marker_id],
                    samples=static_samples[spec.marker_id],
                    thresholds=thresholds_static,
                    tri_detected_flags=(
                        static_tri_flags[spec.marker_id] if enable_triangulation else None
                    ),
                    tri_samples=(
                        static_tri_samples[spec.marker_id] if enable_triangulation else None
                    ),
                    tri_thresholds=tri_static,
                )
                for spec in markers
            ]
            _print_results(
                "statisch (Platte still)",
                static_results,
                thresholds_static,
                tri_static,
                evaluate_jitter=True,
                enable_triangulation=enable_triangulation,
            )
            report["phases"]["static"] = {
                "frames": static_frames,
                "thresholds": thresholds_static,
                "tri_thresholds": tri_static,
                "markers": [r.to_dict() for r in static_results],
            }
        else:
            static_results = []
            print("Statische Phase übersprungen.")

        motion_results: List[MarkerEvalResult] = []
        if not skip_motion and static_results:
            if _wait_start(
                "motion",
                hint="Langsam bewegen (~0–20 mm Hub, leichte Neigung).",
            ):
                motion_flags, motion_samples, motion_tri_flags, motion_tri_samples = _record_frames(
                    phase_name="motion",
                    n_frames=motion_frames,
                    warmup_frames=0,
                    feed=feed,
                    detector=detector,
                    markers=markers,
                    cam_ids=cam_ids,
                    primary_cam_id=primary_cam_id,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    min_views=min_views,
                    enable_triangulation=enable_triangulation,
                )
                motion_results = [
                    aggregate_marker_results(
                        marker_id=spec.marker_id,
                        size_mm=spec.size_mm,
                        label=spec.label,
                        total_frames=len(motion_flags[spec.marker_id]),
                        detected_flags=motion_flags[spec.marker_id],
                        samples=motion_samples[spec.marker_id],
                        thresholds=thresholds_motion,
                        tri_detected_flags=(
                            motion_tri_flags[spec.marker_id] if enable_triangulation else None
                        ),
                        tri_samples=(
                            motion_tri_samples[spec.marker_id] if enable_triangulation else None
                        ),
                        tri_thresholds=tri_motion,
                        evaluate_jitter=False,
                    )
                    for spec in markers
                ]
                _print_results(
                    "Bewegung",
                    motion_results,
                    thresholds_motion,
                    tri_motion,
                    evaluate_jitter=False,
                    enable_triangulation=enable_triangulation,
                )
                report["phases"]["motion"] = {
                    "frames": motion_frames,
                    "thresholds": thresholds_motion,
                    "tri_thresholds": tri_motion,
                    "markers": [r.to_dict() for r in motion_results],
                }
            else:
                print("Bewegungs-Phase übersprungen.")

        rec_static = _recommend_size(
            static_results,
            evaluate_jitter=True,
            enable_triangulation=enable_triangulation,
        )
        rec_motion = (
            _recommend_size(
                motion_results,
                evaluate_jitter=False,
                enable_triangulation=enable_triangulation,
            )
            if motion_results
            else rec_static
        )
        mode_key = "tri" if enable_triangulation else "pnp"
        report["recommendation_mm"] = {
            f"static_min_passing_{mode_key}": rec_static,
            f"motion_min_passing_{mode_key}": rec_motion,
        }

        print(f"\n{'=' * 96}")
        metric_name = "Triangulation" if enable_triangulation else "PnP"
        if rec_static is not None:
            print(f"Empfehlung (statisch, {metric_name}): kleinster Marker ≥ {rec_static:g} mm")
        else:
            print(
                "Kein Marker erfüllt alle Kriterien — größere Marker oder bessere Belichtung testen."
            )
        if motion_results and rec_motion is not None:
            print(f"Empfehlung (Bewegung, {metric_name}): kleinster Marker ≥ {rec_motion:g} mm")
        print(f"{'=' * 96}")

    finally:
        feed.stop()
        rig.close()

    out_path = Path(output) if output else (
        REPO_ROOT / "data" / "recorder" / "evaluation" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_marker_size_eval.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    """Argparse entry for ``python -m optical.evaluation.run_marker_size_eval``."""
    ap = argparse.ArgumentParser(description="ArUco marker size evaluation at fixed distance.")
    ap.add_argument("--config", required=True, help="optical.yaml (uses eval: section)")
    ap.add_argument("--backend", default="picamera2", choices=["picamera2", "v4l2"])
    ap.add_argument("--preview-port", type=int, default=5000, help="MJPEG preview port (0 = off)")
    ap.add_argument("--preview-host", default="0.0.0.0")
    ap.add_argument("--output", help="Report JSON path (default: data/optical/evaluation/<timestamp>.json)")
    ap.add_argument("--skip-motion", action="store_true", help="Only run the static phase")
    ap.add_argument(
        "--resolution",
        default=None,
        help="Capture/eval resolution WxH (e.g. 728x544). Overrides config capture.resolution; "
        "scales calibration K. Omit to use camera config resolution.",
    )
    ap.add_argument(
        "--cameras",
        default=None,
        help="Camera ids to use, comma-separated (e.g. cam0 or cam0,cam1). "
        "1 camera → PnP only; 2 cameras → triangulation metrics.",
    )
    ap.add_argument(
        "--n-cameras",
        type=int,
        choices=[1, 2],
        default=None,
        help="Use the first N cameras from tracker/calibration config (1=PnP, 2=triangulation).",
    )
    args = ap.parse_args(argv)
    run_marker_size_eval(
        config=args.config,
        backend=args.backend,
        preview_host=args.preview_host,
        preview_port=args.preview_port,
        output=args.output,
        skip_motion=args.skip_motion,
        resolution=args.resolution,
        cameras=args.cameras,
        n_cameras=args.n_cameras,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(130)
