# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Triangulation-based marker pose metrics for size evaluation (2-camera rig)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from optical.calibration.geometry import rotation_rms_deg
from optical.calibration.io import CameraExtrinsics, CameraIntrinsics
from optical.tracking.aruco import DetectionResult
from optical.tracking.pose import tag_object_points
from optical.tracking.triangulation import (
    collect_corner_observations,
    kabsch,
    triangulate_dlt,
)


def reprojection_residuals_px(
    point_world: np.ndarray,
    observations: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-view pixel residuals of a triangulated point (same projection as tracking)."""
    X = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    out: List[float] = []
    for P, uv in observations:
        proj = np.asarray(P, dtype=np.float64) @ X
        if abs(proj[2]) < 1e-12:
            continue
        out.append(float(np.hypot(proj[0] / proj[2] - uv[0], proj[1] / proj[2] - uv[1])))
    return np.asarray(out, dtype=np.float64)


@dataclass
class TriangulatedMarkerFrameSample:
    """One marker pose from multi-view corner triangulation + Kabsch fit."""

    frame_idx: int
    marker_id: int
    size_mm: float
    T_world_tag: np.ndarray  # (4, 4)
    translation_mm: np.ndarray  # (3,) from T_world_tag
    reprojection_rms_px: float  # RMS over all corner × view reprojection residuals (px)
    fit_rms_mm: float  # Kabsch residual (mm)


def triangulate_marker_pose(
    detections: Dict[str, DetectionResult],
    *,
    marker_id: int,
    size_mm: float,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    min_views: int = 2,
) -> Optional[TriangulatedMarkerFrameSample]:
    """
    Triangulate four tag corners (DLT) and fit ``T_world←tag`` via Kabsch.

    Uses the same projection / triangulation path as ``tracking/triangulation.py``.
    """
    world_corners: List[np.ndarray] = []
    residuals: List[np.ndarray] = []

    for corner_idx in range(4):
        obs = collect_corner_observations(
            detections,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            marker_id=marker_id,
            corner_index=corner_idx,
        )
        if len(obs) < min_views:
            return None
        pos = triangulate_dlt(obs)
        if pos is None:
            return None
        world_corners.append(pos)
        residuals.append(reprojection_residuals_px(pos, obs))

    model = tag_object_points(size_mm)
    observed = np.stack(world_corners, axis=0)
    T_world_tag, fit_rms_mm = kabsch(model, observed)

    return TriangulatedMarkerFrameSample(
        frame_idx=-1,
        marker_id=marker_id,
        size_mm=size_mm,
        T_world_tag=T_world_tag,
        translation_mm=T_world_tag[:3, 3].copy(),
        # True RMS over all corner × view residuals (was a mean of per-corner means,
        # reported under the name "rms" and gated by max_reprojection_rms_px).
        reprojection_rms_px=_rms(np.concatenate(residuals)),
        fit_rms_mm=fit_rms_mm,
    )


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2))) if values.size else float("inf")


def rotation_jitter_deg_from_poses(poses: List[TriangulatedMarkerFrameSample]) -> float:
    """RMS geodesic angle (deg) of triangulated ``T_world_tag`` rotations from their mean."""
    if len(poses) < 2:
        return float("nan")
    return rotation_rms_deg([p.T_world_tag[:3, :3] for p in poses])
