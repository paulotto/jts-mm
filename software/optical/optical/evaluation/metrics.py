# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Metrics for ArUco marker size / detection quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from optical.calibration.geometry import rodrigues, rotation_rms_deg


def marker_side_px(corners: np.ndarray) -> float:
    """Mean edge length of a detected marker quad in pixels."""
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    return float(np.mean(edges))


def reprojection_rms_mm(
    *,
    reprojection_rms_px: float,
    size_mm: float,
    side_px: float,
) -> float:
    """Convert image-plane RMS (px) to approximate error on the marker plane (mm).

    Uses the local scale ``size_mm / side_px`` so values are comparable across
    image resolutions (raw px RMS shrinks under downscale even when geometry
    does not improve).
    """
    if side_px <= 0 or not np.isfinite(side_px):
        return float("nan")
    if not np.isfinite(reprojection_rms_px):
        return float("nan")
    return float(reprojection_rms_px) * float(size_mm) / float(side_px)


def rotation_magnitude_deg(rvecs: np.ndarray) -> np.ndarray:
    """Per-sample rotation magnitude in degrees from Rodrigues vectors."""
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    return np.linalg.norm(rvecs, axis=1) * 180.0 / np.pi


def rotation_jitter_deg(rvecs: np.ndarray) -> float:
    """RMS geodesic angle (deg) of each pose from the mean rotation.

    ``std(|rvec|)`` (the previous metric) only sees changes of the rotation
    *magnitude*: noise perpendicular to the rotation axis barely moves it, and
    for a tag facing the camera (≈180°) two of three axes are invisible. This
    uses ``‖log(R_meanᵀ R_i)‖`` about the chordal mean, which sees all axes.
    """
    rvecs = np.asarray(rvecs, dtype=np.float64).reshape(-1, 3)
    return rotation_rms_deg([rodrigues(r) for r in rvecs])


def longest_detection_gap(detected: List[bool]) -> int:
    """Longest run of consecutive missed detections."""
    best = 0
    current = 0
    for ok in detected:
        if ok:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


@dataclass
class MarkerFrameSample:
    """One detected marker in one frame (single-camera PnP)."""

    frame_idx: int
    marker_id: int
    size_mm: float
    side_px: float
    tvec_mm: np.ndarray  # (3,)
    rvec: np.ndarray  # (3,)
    reprojection_rms_px: float


@dataclass
class MarkerTriSample:
    """One triangulated marker pose in one frame (stereo)."""

    frame_idx: int
    marker_id: int
    size_mm: float
    translation_mm: np.ndarray  # (3,) world frame
    reprojection_rms_px: float
    fit_rms_mm: float
    rotation_rvec: np.ndarray  # (3,) from T_world_tag


@dataclass
class MarkerEvalResult:
    """Aggregated metrics for one physical marker size."""

    marker_id: int
    size_mm: float
    label: str
    total_frames: int
    detected_frames: int
    detection_rate_pct: float
    mean_side_px: float
    min_side_px: float
    translation_jitter_mm: Dict[str, float]
    translation_std_mm: float
    rotation_jitter_deg: float
    reprojection_rms_px_mean: float
    reprojection_rms_mm_mean: float
    longest_gap_frames: int
    tri_detected_frames: int = 0
    tri_detection_rate_pct: float = 0.0
    tri_reprojection_rms_px_mean: float = float("nan")
    tri_reprojection_rms_mm_mean: float = float("nan")
    tri_fit_rms_mm_mean: float = float("nan")
    tri_translation_std_mm: float = float("nan")
    tri_rotation_jitter_deg: float = float("nan")
    passes: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "marker_id": self.marker_id,
            "size_mm": self.size_mm,
            "label": self.label,
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "detection_rate_pct": self.detection_rate_pct,
            "mean_side_px": self.mean_side_px,
            "min_side_px": self.min_side_px,
            "translation_jitter_mm": self.translation_jitter_mm,
            "translation_std_mm": self.translation_std_mm,
            "rotation_jitter_deg": self.rotation_jitter_deg,
            "reprojection_rms_px_mean": self.reprojection_rms_px_mean,
            "reprojection_rms_mm_mean": self.reprojection_rms_mm_mean,
            "longest_gap_frames": self.longest_gap_frames,
            "tri_detected_frames": self.tri_detected_frames,
            "tri_detection_rate_pct": self.tri_detection_rate_pct,
            "tri_reprojection_rms_px_mean": self.tri_reprojection_rms_px_mean,
            "tri_reprojection_rms_mm_mean": self.tri_reprojection_rms_mm_mean,
            "tri_fit_rms_mm_mean": self.tri_fit_rms_mm_mean,
            "tri_translation_std_mm": self.tri_translation_std_mm,
            "tri_rotation_jitter_deg": self.tri_rotation_jitter_deg,
            "passes": self.passes,
        }


def aggregate_marker_results(
    *,
    marker_id: int,
    size_mm: float,
    label: str,
    total_frames: int,
    detected_flags: List[bool],
    samples: List[MarkerFrameSample],
    thresholds: dict,
    tri_detected_flags: Optional[List[bool]] = None,
    tri_samples: Optional[List[MarkerTriSample]] = None,
    tri_thresholds: Optional[dict] = None,
    evaluate_jitter: bool = True,
) -> MarkerEvalResult:
    """Summarize one marker ID over a capture run.

    Detection rates are computed over the evaluated frames (``len(flags)``), not
    ``total_frames``: a skipped / duplicate frame must neither count as a miss
    nor inflate the rate. ``total_frames`` is reported as given.
    """
    detected_frames = int(sum(detected_flags))
    n_eval = len(detected_flags)
    rate = 100.0 * detected_frames / n_eval if n_eval else 0.0

    if samples:
        sides = np.array([s.side_px for s in samples], dtype=np.float64)
        reproj = float(np.mean([s.reprojection_rms_px for s in samples]))
        reproj_mm_vals = [
            reprojection_rms_mm(
                reprojection_rms_px=s.reprojection_rms_px,
                size_mm=s.size_mm,
                side_px=s.side_px,
            )
            for s in samples
        ]
        reproj_mm_vals = [v for v in reproj_mm_vals if np.isfinite(v)]
        reproj_mm = float(np.mean(reproj_mm_vals)) if reproj_mm_vals else float("nan")
        if evaluate_jitter:
            tvecs = np.stack([s.tvec_mm for s in samples], axis=0)
            rvecs = np.stack([s.rvec for s in samples], axis=0)
            t_std = np.std(tvecs, axis=0)
            rot_std = rotation_jitter_deg(rvecs)
            trans_std = float(np.linalg.norm(t_std))
        else:
            t_std = np.full(3, np.nan, dtype=np.float64)
            rot_std = float("nan")
            trans_std = float("nan")
    else:
        sides = np.array([], dtype=np.float64)
        t_std = np.full(3, np.nan, dtype=np.float64)
        reproj = float("nan")
        reproj_mm = float("nan")
        rot_std = float("nan")
        trans_std = float("nan")

    tri_detected_frames = 0
    tri_rate = 0.0
    tri_reproj = float("nan")
    tri_reproj_mm = float("nan")
    tri_fit = float("nan")
    tri_trans_std = float("nan")
    tri_rot_std = float("nan")
    if tri_detected_flags is not None:
        tri_detected_frames = int(sum(tri_detected_flags))
        n_tri = len(tri_detected_flags)
        tri_rate = 100.0 * tri_detected_frames / n_tri if n_tri else 0.0
    if tri_samples:
        tri_reproj = float(np.mean([s.reprojection_rms_px for s in tri_samples]))
        # Convert via mean detected side length so mm values compare across resolutions.
        mean_side = float(np.mean(sides)) if len(sides) else 0.0
        tri_reproj_mm = reprojection_rms_mm(
            reprojection_rms_px=tri_reproj,
            size_mm=size_mm,
            side_px=mean_side,
        )
        if evaluate_jitter:
            tri_fit = float(np.mean([s.fit_rms_mm for s in tri_samples]))
            t_tri = np.stack([s.translation_mm for s in tri_samples], axis=0)
            tri_trans_std = float(np.linalg.norm(np.std(t_tri, axis=0)))
            tri_rot_std = rotation_jitter_deg(np.stack([s.rotation_rvec for s in tri_samples]))

    passes: Dict[str, bool] = {
        "detection_rate": rate >= float(thresholds.get("min_detection_rate_pct", 0)),
    }
    if evaluate_jitter:
        passes["translation_jitter"] = trans_std <= float(
            thresholds.get("max_translation_jitter_mm", 999)
        )
        passes["pixel_size"] = (
            float(np.min(sides)) >= float(thresholds.get("min_marker_side_px", 0))
            if len(sides)
            else False
        )
    if tri_thresholds is not None and tri_detected_flags is not None:
        passes["tri_detection_rate"] = tri_rate >= float(
            tri_thresholds.get("min_detection_rate_pct", thresholds.get("min_detection_rate_pct", 0))
        )
        if tri_samples:
            passes["tri_reprojection"] = tri_reproj <= float(
                tri_thresholds.get("max_reprojection_rms_px", 1.0)
            )
            if evaluate_jitter:
                passes["tri_translation_jitter"] = tri_trans_std <= float(
                    tri_thresholds.get("max_translation_jitter_mm", 0.5)
                )
                passes["tri_rotation_jitter"] = tri_rot_std <= float(
                    tri_thresholds.get("max_rotation_jitter_deg", 1.0)
                )

    return MarkerEvalResult(
        marker_id=marker_id,
        size_mm=size_mm,
        label=label,
        total_frames=total_frames,
        detected_frames=detected_frames,
        detection_rate_pct=rate,
        mean_side_px=float(np.mean(sides)) if len(sides) else 0.0,
        min_side_px=float(np.min(sides)) if len(sides) else 0.0,
        translation_jitter_mm={
            "x": float(t_std[0]),
            "y": float(t_std[1]),
            "z": float(t_std[2]),
        },
        translation_std_mm=trans_std,
        rotation_jitter_deg=rot_std,
        reprojection_rms_px_mean=reproj,
        reprojection_rms_mm_mean=reproj_mm,
        longest_gap_frames=longest_detection_gap(detected_flags),
        tri_detected_frames=tri_detected_frames,
        tri_detection_rate_pct=tri_rate,
        tri_reprojection_rms_px_mean=tri_reproj,
        tri_reprojection_rms_mm_mean=tri_reproj_mm,
        tri_fit_rms_mm_mean=tri_fit,
        tri_translation_std_mm=tri_trans_std,
        tri_rotation_jitter_deg=tri_rot_std,
        passes=passes,
    )
