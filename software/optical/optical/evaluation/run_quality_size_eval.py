#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Evaluate ArUco marker sizes across an offline image-quality ladder.

Captures (or loads) full-resolution frames once, then degrades them across
resolution / blur / JPEG / noise profiles and re-runs detection + PnP
(and triangulation when ≥2 cameras are available). Use this to pick a lores
stream size that still detects your smallest markers reliably.

Example::

    optical eval-quality --config software/optical/etc/optical.yaml

    optical eval-quality --config ... --frames-dir data/optical/evaluation/frames_dump
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

try:
    from optical.calibration.io import (
        CameraExtrinsics,
        CameraIntrinsics,
        load_cameras_from_bundle,
        read_calibration_bundle,
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
    )
    from optical.evaluation.image_quality import (
        QualityProfile,
        degrade_rgb,
        resolve_profile_for_frame,
        scale_intrinsics,
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
    )
    from optical.evaluation.image_quality import (
        QualityProfile,
        degrade_rgb,
        resolve_profile_for_frame,
        scale_intrinsics,
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
        return eval_section(doc, "eval_quality")
    except KeyError:
        try:
            return eval_section(doc, "eval")
        except KeyError:
            return doc


def _parse_markers(cfg: dict) -> List[MarkerSpec]:
    out: List[MarkerSpec] = []
    for entry in cfg["markers"]:
        mid = int(entry["id"])
        size_mm = float(entry["size_mm"])
        label = str(entry.get("label", f"{size_mm:g}mm"))
        out.append(MarkerSpec(marker_id=mid, size_mm=size_mm, label=label))
    return out


def _load_calibration(bundle_path: Path, cam_ids: List[str]):
    bundle = read_calibration_bundle(bundle_path)
    intrinsics, extrinsics = load_cameras_from_bundle(bundle)
    for cid in cam_ids:
        if cid not in intrinsics:
            raise KeyError(f"Camera {cid!r} missing intrinsics in {bundle_path}")
        if cid not in extrinsics:
            raise KeyError(f"Camera {cid!r} missing extrinsics in {bundle_path}")
    return intrinsics, extrinsics


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
    rms = float(np.sqrt(np.mean(err**2)))
    return rvec.reshape(3), np.asarray(tvec, dtype=np.float64).reshape(3), rms


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


def _scaled_intrinsics_map(
    intrinsics: Dict[str, CameraIntrinsics],
    *,
    profile: QualityProfile,
) -> Dict[str, CameraIntrinsics]:
    """K for each profile size, always scaled from the size K was calibrated at.

    Scaling from the burst frame size instead silently applied the wrong
    factor whenever the burst was not captured at ``intr.image_size``.
    """
    out: Dict[str, CameraIntrinsics] = {}
    for cam_id, intr in intrinsics.items():
        K = scale_intrinsics(intr.K, src_wh=intr.image_size, dst_wh=profile.size)
        out[cam_id] = CameraIntrinsics(
            K=K,
            dist=np.asarray(intr.dist, dtype=np.float64).copy(),
            image_size=profile.size,
            rms_px=intr.rms_px,
        )
    return out


def _list_frame_indices(frames_dir: Path, cam_ids: Sequence[str]) -> List[int]:
    """Discover frame indices from ``{cam_id}_{idx:06d}.png`` dumps."""
    primary = cam_ids[0]
    indices: List[int] = []
    for path in sorted(frames_dir.glob(f"{primary}_*.png")):
        stem = path.stem  # cam0_000012
        try:
            idx = int(stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        indices.append(idx)
    return indices


def _load_frame_bundle(
    frames_dir: Path,
    cam_ids: Sequence[str],
    frame_idx: int,
) -> Dict[str, np.ndarray]:
    frames: Dict[str, np.ndarray] = {}
    for cam_id in cam_ids:
        path = frames_dir / f"{cam_id}_{frame_idx:06d}.png"
        if not path.is_file():
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")
        frames[cam_id] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return frames


def _save_frame_bundle(
    frames_dir: Path,
    frames: Dict[str, np.ndarray],
    frame_idx: int,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for cam_id, rgb in frames.items():
        bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
        path = frames_dir / f"{cam_id}_{frame_idx:06d}.png"
        cv2.imwrite(str(path), bgr)


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


class _LivePreviewFeed:
    """Background capture so MJPEG preview is live before SPACE starts the burst."""

    def __init__(
        self,
        rig: Any,
        cam_order: List[str],
        detector: ArucoDetector,
        markers: List[MarkerSpec],
        preview: Optional[PreviewServer],
    ) -> None:
        self._rig = rig
        self._cam_order = cam_order
        self._detector = detector
        self._marker_labels = {m.marker_id: m.label for m in markers}
        self._preview = preview
        self._lock = threading.Lock()
        self._latest: Dict[str, np.ndarray] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status = "Live preview — SPACE starts capture"

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="eval-quality-preview")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def set_status(self, text: str) -> None:
        self._status = text

    def latest_frames(self) -> Dict[str, np.ndarray]:
        with self._lock:
            return {cid: frame.copy() for cid, frame in self._latest.items()}

    def _loop(self) -> None:
        while self._running:
            frames = self._rig.capture_all()
            if not frames:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frames

            if self._preview is None:
                continue
            overlay: Dict[str, np.ndarray] = {}
            for cam_id in self._cam_order:
                rgb = frames.get(cam_id)
                if rgb is None:
                    continue
                det = _detect_image(self._detector, rgb)
                det_by_id = {m.marker_id: m.corners for m in det.markers}
                overlay[cam_id] = _draw_markers(rgb, det_by_id, self._marker_labels)
            stacked = _stack_preview(overlay, self._cam_order)
            if stacked is not None:
                self._preview.update(_draw_status_bar(stacked, self._status, (0, 220, 0)))


def _capture_fullres_burst(
    *,
    n_frames: int,
    warmup_frames: int,
    cams: List[CamSpec],
    cam_ids: List[str],
    backend: str,
    preview_host: str,
    preview_port: int,
    save_dir: Path | None,
    markers: List[MarkerSpec],
    dictionary: str,
) -> List[Dict[str, np.ndarray]]:
    """Interactive full-res capture; returns list of {cam_id: rgb}."""
    rig = _open_rig(cams, backend)
    preview: Optional[PreviewServer] = None
    if preview_port > 0:
        preview = PreviewServer(preview_host, preview_port)
        preview.start()
        print(f"Preview: {preview.url}")

    detector = ArucoDetector(dictionary)
    feed = _LivePreviewFeed(rig, cam_ids, detector, markers, preview)
    feed.start()

    try:
        _wait_for_key(
            "\n[capture] Hold the marker plate still (all sizes visible).\n"
            "  Preview is live — SPACE = start full-res burst, ESC = abort",
            accept={ord(" ")},
        )
        # Stop the background feeder so the burst owns the camera exclusively.
        feed.stop()

        if warmup_frames > 0:
            print(f"[capture] Warmup {warmup_frames} frames …")
            for _ in range(warmup_frames):
                rig.capture_all()
                time.sleep(0.01)

        burst: List[Dict[str, np.ndarray]] = []
        print(f"[capture] Recording {n_frames} full-res frames …")
        for i in range(n_frames):
            # Same-trigger frames only (SensorTimestamp check) so triangulation pairs match.
            frames, _meta = rig.capture_all_synced()
            if not frames:
                time.sleep(0.005)
                continue
            # Keep a contiguous copy so later degrade steps own the buffer.
            copied = {
                cid: np.ascontiguousarray(frames[cid][:, :, :3])
                for cid in cam_ids
                if cid in frames
            }
            if len(copied) < len(cam_ids):
                continue
            burst.append(copied)
            if save_dir is not None:
                _save_frame_bundle(save_dir, copied, len(burst) - 1)
            if preview is not None:
                overlay = {
                    cid: _draw_markers(
                        rgb,
                        {
                            m.marker_id: m.corners
                            for m in _detect_image(detector, rgb).markers
                        },
                        {m.marker_id: m.label for m in markers},
                    )
                    for cid, rgb in copied.items()
                }
                stacked = _stack_preview(overlay, cam_ids)
                if stacked is not None:
                    preview.update(
                        _draw_status_bar(
                            stacked,
                            f"capture {len(burst)}/{n_frames}",
                            (0, 220, 0),
                        )
                    )
            if (i + 1) % 25 == 0:
                print(f"  … {len(burst)}/{n_frames}")
        if not burst:
            raise RuntimeError("no frames captured")
        print(f"[capture] Kept {len(burst)} frames")
        return burst
    finally:
        # PreviewServer has no stop(): its Flask thread is a daemon and ends with
        # the process (the old preview.stop() only raised a swallowed AttributeError).
        feed.stop()
        rig.close()


def _load_burst_from_dir(
    frames_dir: Path,
    cam_ids: List[str],
    *,
    max_frames: int | None = None,
) -> List[Dict[str, np.ndarray]]:
    indices = _list_frame_indices(frames_dir, cam_ids)
    if not indices:
        raise FileNotFoundError(
            f"no frames matching {cam_ids[0]}_XXXXXX.png in {frames_dir}"
        )
    if max_frames is not None:
        indices = indices[: max(1, int(max_frames))]
    burst: List[Dict[str, np.ndarray]] = []
    for idx in indices:
        frames = _load_frame_bundle(frames_dir, cam_ids, idx)
        if len(frames) < len(cam_ids):
            continue
        burst.append(frames)
    if not burst:
        raise RuntimeError(f"no complete multi-cam frames in {frames_dir}")
    return burst


def _evaluate_quality(
    *,
    burst: List[Dict[str, np.ndarray]],
    profile: QualityProfile,
    markers: List[MarkerSpec],
    cam_ids: List[str],
    primary_cam_id: str,
    detector: ArucoDetector,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    min_views: int,
    thresholds: dict,
    do_triangulation: bool,
) -> tuple[List[MarkerEvalResult], dict[str, float]]:
    """Run detection/PnP(/tri) for one quality profile over the burst."""
    scaled_intr = _scaled_intrinsics_map(intrinsics, profile=profile)
    primary_intr = scaled_intr[primary_cam_id]

    detected_flags: Dict[int, List[bool]] = {m.marker_id: [] for m in markers}
    samples: Dict[int, List[MarkerFrameSample]] = {m.marker_id: [] for m in markers}
    tri_detected_flags: Dict[int, List[bool]] = {m.marker_id: [] for m in markers}
    tri_samples: Dict[int, List[MarkerTriSample]] = {m.marker_id: [] for m in markers}
    detect_ms: List[float] = []

    total_frames = len(burst)
    for frame_idx, frames in enumerate(burst):
        degraded: Dict[str, np.ndarray] = {}
        for cid in cam_ids:
            degraded[cid] = degrade_rgb(frames[cid], profile)

        detections: Dict[str, DetectionResult] = {}
        det_by_cam: Dict[str, Dict[int, np.ndarray]] = {}
        t0 = time.perf_counter()
        for cid in cam_ids:
            det = _detect_image(detector, degraded[cid])
            detections[cid] = det
            det_by_cam[cid] = {m.marker_id: m.corners for m in det.markers}
        detect_ms.append((time.perf_counter() - t0) * 1000.0)

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
                    side = marker_side_px(corners_primary)
                    for cid in cam_ids:
                        c = det_by_cam.get(cid, {}).get(spec.marker_id)
                        if c is not None:
                            side = min(side, marker_side_px(c))
                    samples[spec.marker_id].append(
                        MarkerFrameSample(
                            frame_idx=frame_idx,
                            marker_id=spec.marker_id,
                            size_mm=spec.size_mm,
                            side_px=side,
                            tvec_mm=np.asarray(tvec, dtype=np.float64).reshape(3),
                            rvec=rvec,
                            reprojection_rms_px=rms,
                        )
                    )

            tri_ok = False
            if do_triangulation and len(cam_ids) >= min_views:
                in_both = all(
                    spec.marker_id in det_by_cam.get(cid, {}) for cid in cam_ids
                )
                if in_both:
                    tri = triangulate_marker_pose(
                        detections,
                        marker_id=spec.marker_id,
                        size_mm=spec.size_mm,
                        intrinsics=scaled_intr,
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

    tri_thresholds = None
    if do_triangulation:
        tri_thresholds = dict(thresholds.get("triangulation", {}))
        tri_thresholds.setdefault("cam_ids", cam_ids)
        tri_thresholds.setdefault(
            "min_detection_rate_pct", thresholds.get("min_detection_rate_pct", 90)
        )
        tri_thresholds.setdefault("max_reprojection_rms_px", 2.0)
        tri_thresholds.setdefault(
            "max_translation_jitter_mm",
            thresholds.get("max_translation_jitter_mm", 0.5),
        )
        tri_thresholds.setdefault("max_rotation_jitter_deg", 1.0)

    results = [
        aggregate_marker_results(
            marker_id=spec.marker_id,
            size_mm=spec.size_mm,
            label=spec.label,
            total_frames=total_frames,
            detected_flags=detected_flags[spec.marker_id],
            samples=samples[spec.marker_id],
            thresholds=thresholds,
            tri_detected_flags=tri_detected_flags[spec.marker_id] if do_triangulation else None,
            tri_samples=tri_samples[spec.marker_id] if do_triangulation else None,
            tri_thresholds=tri_thresholds,
            evaluate_jitter=True,
        )
        for spec in markers
    ]
    # Quality-ladder pass gates: detectability + geometric fit in mm.
    # Absolute translation jitter is reported but not required (hand tremor).
    # Reprojection is gated in mm on the marker plane so resolutions compare fairly.
    max_reproj_mm = float(thresholds.get("max_reprojection_rms_mm", 0.5))
    for r in results:
        r.passes["reprojection"] = bool(
            np.isfinite(r.reprojection_rms_mm_mean)
            and r.reprojection_rms_mm_mean <= max_reproj_mm
        )
    timing = {
        "detect_ms_mean": float(np.mean(detect_ms)) if detect_ms else float("nan"),
        "detect_ms_p95": float(np.percentile(detect_ms, 95)) if detect_ms else float("nan"),
    }
    return results, timing


def _pass_keys(*, do_triangulation: bool) -> tuple[str, ...]:
    keys = ("detection_rate", "pixel_size", "reprojection")
    if do_triangulation:
        keys = keys + ("tri_detection_rate", "tri_reprojection")
    return keys


def _col(name: str, width: int = 12) -> str:
    """Fit a quality name into a fixed column."""
    if len(name) <= width:
        return f"{name:>{width}}"
    return f"{name[: width - 1]}~"


def _fail_abbrev(r: MarkerEvalResult, pass_keys: tuple[str, ...]) -> str:
    fails = [k for k in pass_keys if not r.passes.get(k, False)]
    if not fails:
        return "yes"
    short = {
        "detection_rate": "det",
        "pixel_size": "px",
        "reprojection": "rep",
        "translation_jitter": "jit",
        "tri_detection_rate": "tdet",
        "tri_reprojection": "trep",
        "tri_translation_jitter": "tjit",
        "tri_rotation_jitter": "trot",
    }
    return ",".join(short.get(k, k[:3]) for k in fails)


def _print_matrix(
    *,
    quality_names: List[str],
    markers: List[MarkerSpec],
    cells: Dict[tuple[str, float], MarkerEvalResult],
    timing: Dict[str, dict[str, float]],
    do_triangulation: bool,
    thresholds: dict,
) -> None:
    pass_keys = _pass_keys(do_triangulation=do_triangulation)
    w = 12

    print(f"\n{'=' * 100}")
    print("Detection rate (%) — rows=marker size, cols=quality")
    print(f"{'=' * 100}")
    header = f"{'size':>8}" + "".join(_col(q, w) for q in quality_names)
    print(header)
    print("-" * len(header))
    for spec in sorted(markers, key=lambda m: m.size_mm, reverse=True):
        row = f"{spec.label:>8}"
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            row += f"{r.detection_rate_pct:11.1f}%"
        print(row)

    print(f"\n{'=' * 100}")
    print("Min marker side (px) / detect_ms_mean")
    print(f"{'=' * 100}")
    header = f"{'size':>8}" + "".join(_col(q, w) for q in quality_names)
    print(header)
    print("-" * len(header))
    for spec in sorted(markers, key=lambda m: m.size_mm, reverse=True):
        row = f"{spec.label:>8}"
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            row += f"{r.min_side_px:12.1f}"
        print(row)
    time_row = f"{'det_ms':>8}"
    for q in quality_names:
        time_row += f"{timing[q]['detect_ms_mean']:12.2f}"
    print(time_row)

    print(f"\n{'=' * 100}")
    print("PnP translation jitter (mm, informational — not a pass gate)")
    print(f"{'=' * 100}")
    header = f"{'size':>8}" + "".join(_col(q, w) for q in quality_names)
    print(header)
    print("-" * len(header))
    for spec in sorted(markers, key=lambda m: m.size_mm, reverse=True):
        row = f"{spec.label:>8}"
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            row += f"{r.translation_std_mm:12.3f}"
        print(row)

    print(f"\n{'=' * 100}")
    print("PnP reprojection RMS (mm on marker plane; px × size_mm / side_px)")
    print(f"{'=' * 100}")
    header = f"{'size':>8}" + "".join(_col(q, w) for q in quality_names)
    print(header)
    print("-" * len(header))
    for spec in sorted(markers, key=lambda m: m.size_mm, reverse=True):
        row = f"{spec.label:>8}"
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            row += f"{r.reprojection_rms_mm_mean:12.3f}"
        print(row)

    print(f"\n{'=' * 100}")
    print(
        "Pass? (yes | fail reasons: det=detection, px=min side_px, rep=reprojection)"
    )
    print(
        f"  gates: det≥{thresholds.get('min_detection_rate_pct')}%, "
        f"side≥{thresholds.get('min_marker_side_px')}px, "
        f"reproj≤{thresholds.get('max_reprojection_rms_mm')}mm"
    )
    print(f"{'=' * 100}")
    header = f"{'size':>8}" + "".join(_col(q, w) for q in quality_names)
    print(header)
    print("-" * len(header))
    for spec in sorted(markers, key=lambda m: m.size_mm, reverse=True):
        row = f"{spec.label:>8}"
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            row += f"{_fail_abbrev(r, pass_keys):>12}"
        print(row)


def _recommendations(
    *,
    quality_names: List[str],
    markers: List[MarkerSpec],
    cells: Dict[tuple[str, float], MarkerEvalResult],
    do_triangulation: bool,
) -> dict:
    pass_keys = _pass_keys(do_triangulation=do_triangulation)
    min_passing_by_quality: Dict[str, Optional[float]] = {}
    for q in quality_names:
        passing = [
            spec.size_mm
            for spec in markers
            if all(cells[(q, spec.size_mm)].passes.get(k, False) for k in pass_keys)
        ]
        min_passing_by_quality[q] = min(passing) if passing else None

    best_quality_by_marker: Dict[str, Optional[str]] = {}
    for spec in markers:
        candidates: List[tuple[float, str]] = []
        for q in quality_names:
            r = cells[(q, spec.size_mm)]
            if all(r.passes.get(k, False) for k in pass_keys):
                candidates.append((r.min_side_px, q))
        if candidates:
            # Highest side_px ⇒ least degraded among passing set.
            best_quality_by_marker[spec.label] = max(candidates, key=lambda t: t[0])[1]
        else:
            best_quality_by_marker[spec.label] = None

    return {
        "min_passing_marker_mm_by_quality": min_passing_by_quality,
        "best_passing_quality_by_marker": best_quality_by_marker,
    }


def run_quality_size_eval(
    *,
    config: str | Path,
    backend: str = "picamera2",
    preview_host: str = "0.0.0.0",
    preview_port: int = 5000,
    output: str | Path | None = None,
    frames_dir: str | Path | None = None,
    save_frames: bool = False,
) -> Path:
    """Run size × quality evaluation; write a JSON report."""
    cfg_path = resolve_config_path(config) if not Path(config).is_absolute() else Path(config)
    if not cfg_path.is_file():
        cfg_path = resolve_repo_path(config)
    cfg = _load_json(cfg_path)

    markers = _parse_markers(cfg)
    dictionary = str(cfg.get("dictionary", "DICT_4X4_50"))
    quality_entries = list(cfg.get("qualities") or [])
    if not quality_entries:
        raise ValueError("config must define a non-empty 'qualities' list")

    tracker_cfg_path = cfg_path
    if cfg.get("tracker_config"):
        tracker_cfg_path = resolve_config_path(cfg["tracker_config"])
    tcfg = TrackerConfig.from_json(tracker_cfg_path)
    cam_ids = list(tcfg.cam_ids) if tcfg.cam_ids else []
    min_views = int(tcfg.triangulation_min_views or 2)
    primary_cam_id = str(cfg.get("camera_id", cam_ids[0] if cam_ids else "cam0"))

    cal_cfg_path = tracker_cfg_path
    if cfg.get("calibration_config"):
        cal_cfg_path = resolve_config_path(cfg["calibration_config"])
    from optical.config.loader import calibration_section, load_config_file

    try:
        cal_cfg = calibration_section(load_config_file(cal_cfg_path), config_path=cal_cfg_path)
    except KeyError:
        cal_cfg = _load_json(cal_cfg_path)
    all_cams = _parse_cams(cal_cfg)
    if not cam_ids:
        cam_ids = [c.cam_id for c in all_cams]
    if primary_cam_id not in cam_ids:
        primary_cam_id = cam_ids[0]
    do_triangulation = bool(cfg.get("triangulation", len(cam_ids) >= 2)) and len(cam_ids) >= 2

    cams = [c for c in all_cams if c.cam_id in cam_ids]
    if len(cams) != len(cam_ids):
        missing = set(cam_ids) - {c.cam_id for c in cams}
        raise KeyError(f"Cameras {missing} missing in {cal_cfg_path}")

    bundle_path = resolve_repo_path(
        cfg.get("calibration_bundle", str(tcfg.calibration_bundle_path))
    )
    intrinsics, extrinsics = _load_calibration(bundle_path, cam_ids)

    cap = cfg.get("capture", {})
    n_frames = int(cap.get("frames", cap.get("static_frames", 60)))
    warmup = int(cap.get("warmup_frames", 15))
    thresholds = dict(cfg.get("thresholds", {}))
    thresholds.setdefault("min_detection_rate_pct", 95)
    thresholds.setdefault("min_marker_side_px", 20)
    thresholds.setdefault("max_reprojection_rms_mm", 0.5)
    thresholds.setdefault("primary_cam", primary_cam_id)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO_ROOT / "data" / "optical" / "evaluation" / f"{stamp}_quality_size"
    dump_dir: Path | None = None
    if save_frames:
        dump_dir = run_dir / "frames"
        dump_dir.mkdir(parents=True, exist_ok=True)

    if frames_dir is not None:
        burst = _load_burst_from_dir(Path(frames_dir), cam_ids, max_frames=n_frames)
    else:
        burst = _capture_fullres_burst(
            n_frames=n_frames,
            warmup_frames=warmup,
            cams=cams,
            cam_ids=cam_ids,
            backend=backend,
            preview_host=preview_host,
            preview_port=preview_port,
            save_dir=dump_dir,
            markers=markers,
            dictionary=dictionary,
        )

    # Resolve quality profiles against the captured native size.
    sample0 = burst[0][primary_cam_id]
    src_h, src_w = sample0.shape[:2]
    for cid in cam_ids:
        calib_wh = tuple(int(v) for v in intrinsics[cid].image_size)
        h, w = burst[0][cid].shape[:2]
        if (w, h) != calib_wh:
            print(
                f"warning: {cid} frames are {w}x{h} but K was calibrated at "
                f"{calib_wh[0]}x{calib_wh[1]}; K is scaled from the calibration size "
                "(assumes the same field of view)"
            )
    profiles = [
        resolve_profile_for_frame(entry, src_wh=(src_w, src_h)) for entry in quality_entries
    ]

    detector = ArucoDetector(dictionary)

    print("=" * 100)
    print("ArUco marker size × image quality evaluation")
    print(f"Cameras: {', '.join(cam_ids)} | primary PnP: {primary_cam_id}")
    print(f"Markers: {', '.join(m.label for m in markers)}")
    print(f"Frames: {len(burst)} | native: {src_w}x{src_h}")
    print(f"Qualities: {', '.join(p.name for p in profiles)}")
    print(f"Triangulation: {'on' if do_triangulation else 'off'}")
    print("=" * 100)

    cells: Dict[tuple[str, float], MarkerEvalResult] = {}
    timing: Dict[str, dict[str, float]] = {}
    quality_block: List[dict] = []

    for profile in profiles:
        print(f"\n[quality] {profile.name} ({profile.width}x{profile.height}) …")
        results, tstats = _evaluate_quality(
            burst=burst,
            profile=profile,
            markers=markers,
            cam_ids=cam_ids,
            primary_cam_id=primary_cam_id,
            detector=detector,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            min_views=min_views,
            thresholds=thresholds,
            do_triangulation=do_triangulation,
        )
        timing[profile.name] = tstats
        for r in results:
            cells[(profile.name, r.size_mm)] = r
        quality_block.append(
            {
                "profile": profile.to_dict(),
                "timing": tstats,
                "markers": [r.to_dict() for r in results],
            }
        )

    quality_names = [p.name for p in profiles]
    _print_matrix(
        quality_names=quality_names,
        markers=markers,
        cells=cells,
        timing=timing,
        do_triangulation=do_triangulation,
        thresholds=thresholds,
    )
    rec = _recommendations(
        quality_names=quality_names,
        markers=markers,
        cells=cells,
        do_triangulation=do_triangulation,
    )

    print(f"\n{'=' * 100}")
    print("Recommendations")
    for q, mm in rec["min_passing_marker_mm_by_quality"].items():
        if mm is None:
            print(f"  {q}: no marker passes thresholds")
        else:
            print(f"  {q}: smallest passing marker ≥ {mm:g} mm")
    for label, qname in rec["best_passing_quality_by_marker"].items():
        print(f"  marker {label}: best passing quality = {qname or 'none'}")
    print(f"{'=' * 100}")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(cfg_path),
        "tracker_config": str(tracker_cfg_path),
        "calibration_bundle": str(bundle_path),
        "cam_ids": cam_ids,
        "primary_cam_id": primary_cam_id,
        "dictionary": dictionary,
        "native_resolution": [src_w, src_h],
        "n_frames": len(burst),
        "triangulation": do_triangulation,
        "markers": [
            {"id": m.marker_id, "size_mm": m.size_mm, "label": m.label} for m in markers
        ],
        "thresholds": thresholds,
        "qualities": quality_block,
        "recommendation": rec,
        "frames_dir": str(frames_dir) if frames_dir else (str(dump_dir) if dump_dir else None),
    }

    out_path = Path(output) if output else (run_dir / "report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")
    return out_path


def main(argv: list[str] | None = None) -> None:
    """Argparse entry for ``python -m optical.evaluation.run_quality_size_eval``."""
    ap = argparse.ArgumentParser(
        description="ArUco marker size × image quality (lores tradeoff) evaluation."
    )
    ap.add_argument("--config", required=True, help="Eval config JSON")
    ap.add_argument("--backend", default="picamera2", choices=["picamera2", "v4l2"])
    ap.add_argument("--preview-port", type=int, default=5000)
    ap.add_argument("--preview-host", default="0.0.0.0")
    ap.add_argument("--output", help="Report JSON path")
    ap.add_argument(
        "--frames-dir",
        help="Replay full-res frames from a prior dump (skips live capture)",
    )
    ap.add_argument(
        "--save-frames",
        action="store_true",
        help="Dump the full-res burst under data/optical/evaluation/<run>/frames/",
    )
    args = ap.parse_args(argv)
    run_quality_size_eval(
        config=args.config,
        backend=args.backend,
        preview_host=args.preview_host,
        preview_port=args.preview_port,
        output=args.output,
        frames_dir=args.frames_dir,
        save_frames=args.save_frames,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(130)
