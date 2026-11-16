# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Multi-view triangulation and Kabsch rigid-body fitting (Approach B).

See ``docs/optical/03_pose_estimation_and_output.md`` §2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from optical.calibration.io import CameraExtrinsics, CameraIntrinsics
from optical.markers.config import MarkerConfig
from optical.tracking.aruco import DetectionResult
from optical.tracking.pose import invert_T, is_valid_pose, nan_pose, tag_object_points
from optical.tracking.pose_estimation import BODY_NAMES, WorldBodyPose, body_for_marker_id


@dataclass
class TriangulatedPoint:
    """One 3D point in the world frame."""

    marker_id: int
    corner_index: int  # 0..3 for tag corners; -1 for tag center
    position_mm: np.ndarray  # (3,)
    n_views: int
    reprojection_rms_px: float


# Per-view reprojection error above which a corner observation is dropped
# (lores pixels). Default of tracker.triangulation.max_corner_reprojection_px.
DEFAULT_MAX_CORNER_REPROJ_PX = 2.0
# Kabsch trimming: a point is an outlier above max(TRIM_K * median, TRIM_MIN_MM).
_TRIM_K = 3.0
_TRIM_MIN_MM = 1.0
_TRIM_ROUNDS = 2

_P_CACHE: Dict[bytes, np.ndarray] = {}
_P_CACHE_MAX = 32


def projection_matrix(K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """
    Build 3×4 projection matrix ``P`` such that ``λ [u,v,1]^T = P [X,Y,Z,1]^T``.

    ``T_world_cam`` maps camera coordinates to world (``p_world = T @ p_cam``).
    """
    T_cam_world = invert_T(T_world_cam)
    Rt = T_cam_world[:3, :]
    return np.asarray(K, dtype=np.float64) @ Rt


def cached_projection_matrix(K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """:func:`projection_matrix`, memoised on the (K, T) values.

    Calibration is fixed for a session, so rebuilding ``P`` (a 4×4 inverse) for
    every corner of every frame was pure overhead on the Pi.
    """
    K = np.ascontiguousarray(K, dtype=np.float64)
    T = np.ascontiguousarray(T_world_cam, dtype=np.float64)
    key = K.tobytes() + T.tobytes()
    P = _P_CACHE.get(key)
    if P is None:
        if len(_P_CACHE) >= _P_CACHE_MAX:
            _P_CACHE.clear()
        P = projection_matrix(K, T)
        P.setflags(write=False)
        _P_CACHE[key] = P
    return P


def undistort_points(
    points: np.ndarray,
    *,
    K: np.ndarray,
    dist: np.ndarray,
) -> np.ndarray:
    """Undistort 2D points to normalized image plane, then re-scale with K."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    d = np.asarray(dist, dtype=np.float64).ravel()
    if d.size == 0 or not np.any(d):
        return pts.reshape(-1, 2).copy()  # no distortion: identity, skip cv2
    import cv2

    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und.reshape(-1, 2)


def triangulate_dlt(
    observations: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> Optional[np.ndarray]:
    """
    Triangulate one 3D point from ≥2 (P, uv) pairs via DLT + SVD.

    Args:
        observations: List of ``(P_i, (u, v))`` where ``P_i`` is 3×4.

    Returns:
        3D point in world frame (mm), or ``None`` if underconstrained.
    """
    if len(observations) < 2:
        return None

    A_rows: List[np.ndarray] = []
    for P, uv in observations:
        u, v = float(uv[0]), float(uv[1])
        A_rows.append(u * P[2] - P[0])
        A_rows.append(v * P[2] - P[1])
    A = np.stack(A_rows, axis=0)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None
    return (Xh[:3] / Xh[3]).astype(np.float64)


def reprojection_errors_px(
    point_world: np.ndarray,
    observations: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Per-view reprojection error (pixels); ``inf`` for a view behind the camera."""
    X = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    errs = np.full(len(observations), np.inf, dtype=np.float64)
    for i, (P, uv) in enumerate(observations):
        proj = P @ X
        if proj[2] <= 1e-12:
            continue
        errs[i] = float(np.hypot(proj[0] / proj[2] - uv[0], proj[1] / proj[2] - uv[1]))
    return errs


def reprojection_error_px(
    point_world: np.ndarray,
    observations: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean reprojection error (pixels) for a triangulated world point."""
    errs = reprojection_errors_px(point_world, observations)
    errs = errs[np.isfinite(errs)]
    return float(np.mean(errs)) if errs.size else float("inf")


def kabsch(
    model_pts: np.ndarray,
    observed_pts: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Solve rigid transform ``T`` minimizing ``‖R @ model + t - observed‖²``.

    Args:
        model_pts: (N, 3) points in body frame.
        observed_pts: (N, 3) corresponding points in world frame.

    Returns:
        ``(T_world_body, rms_mm)`` where ``T_world_body`` is 4×4.
    """
    model_pts = np.asarray(model_pts, dtype=np.float64)
    observed_pts = np.asarray(observed_pts, dtype=np.float64)
    if model_pts.shape != observed_pts.shape or model_pts.shape[0] < 3:
        raise ValueError("kabsch requires ≥3 matched points with identical shapes")

    centroid_model = model_pts.mean(axis=0)
    centroid_obs = observed_pts.mean(axis=0)
    M = model_pts - centroid_model
    Q = observed_pts - centroid_obs

    H = M.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_obs - R @ centroid_model
    transformed = (R @ model_pts.T).T + t
    residual = observed_pts - transformed
    rms_mm = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, rms_mm


def kabsch_trimmed(
    model_pts: np.ndarray,
    observed_pts: np.ndarray,
    *,
    rounds: int = _TRIM_ROUNDS,
    min_points: int = 4,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Kabsch with up to ``rounds`` of residual trimming.

    Points whose residual exceeds ``max(3·median, 1 mm)`` are dropped and the
    fit repeated while at least ``min_points`` remain (one mis-triangulated
    corner otherwise tilts the whole body). Returns ``(T, rms_mm, keep_mask)``.
    """
    model_pts = np.asarray(model_pts, dtype=np.float64)
    observed_pts = np.asarray(observed_pts, dtype=np.float64)
    keep = np.ones(model_pts.shape[0], dtype=bool)
    T, rms = kabsch(model_pts, observed_pts)
    for _ in range(max(0, int(rounds))):
        res = np.linalg.norm(observed_pts - (model_pts @ T[:3, :3].T + T[:3, 3]), axis=1)
        thr = max(_TRIM_K * float(np.median(res[keep])), _TRIM_MIN_MM)
        new_keep = keep & (res <= thr)
        if new_keep.sum() == keep.sum() or new_keep.sum() < max(3, int(min_points)):
            break
        keep = new_keep
        T, rms = kabsch(model_pts[keep], observed_pts[keep])
    return T, rms, keep


def _undistorted_corners(
    detections: Dict[str, DetectionResult],
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
) -> Dict[str, Tuple[np.ndarray, Dict[int, np.ndarray]]]:
    """Per camera: ``(P, {marker_id: (4, 2) undistorted corners})``, one undistort call each."""
    out: Dict[str, Tuple[np.ndarray, Dict[int, np.ndarray]]] = {}
    for cam_id, detection in detections.items():
        if cam_id not in intrinsics or cam_id not in extrinsics or not detection.markers:
            continue
        intr = intrinsics[cam_id]
        ids: List[int] = []
        pts: List[np.ndarray] = []
        for marker in detection.markers:
            if marker.marker_id in ids:
                continue  # duplicate ID in one frame: keep the first
            ids.append(marker.marker_id)
            pts.append(np.asarray(marker.corners, dtype=np.float64).reshape(4, 2))
        und = undistort_points(np.concatenate(pts, axis=0), K=intr.K, dist=intr.dist)
        P = cached_projection_matrix(intr.K, extrinsics[cam_id].T_world_cam)
        out[cam_id] = (P, {mid: und[4 * i : 4 * i + 4] for i, mid in enumerate(ids)})
    return out


def collect_corner_observations(
    detections: Dict[str, DetectionResult],
    *,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    marker_id: int,
    corner_index: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Gather undistorted (P, uv) pairs for one tag corner across cameras."""
    observations: List[Tuple[np.ndarray, np.ndarray]] = []
    for cam_id, detection in detections.items():
        if cam_id not in intrinsics or cam_id not in extrinsics:
            continue
        for marker in detection.markers:
            if marker.marker_id != marker_id:
                continue
            uv = undistort_points(
                marker.corners[corner_index],
                K=intrinsics[cam_id].K,
                dist=intrinsics[cam_id].dist,
            )[0]
            P = cached_projection_matrix(intrinsics[cam_id].K, extrinsics[cam_id].T_world_cam)
            observations.append((P, uv))
    return observations


def triangulate_tag_corners(
    detections: Dict[str, DetectionResult],
    *,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    marker_config: MarkerConfig,
    min_views: int = 2,
    max_corner_reproj_px: float | None = DEFAULT_MAX_CORNER_REPROJ_PX,
) -> List[TriangulatedPoint]:
    """Triangulate all tag corners visible in ≥ ``min_views`` cameras.

    A view whose reprojection error exceeds ``max_corner_reproj_px`` is dropped
    and the corner re-triangulated from the rest; a corner that cannot get
    under the limit with ``min_views`` views is discarded (``None`` / 0 keeps
    every corner). ``reprojection_rms_px`` is the RMS over the views kept.
    """
    per_cam = _undistorted_corners(detections, intrinsics, extrinsics)
    limit = float(max_corner_reproj_px) if max_corner_reproj_px else 0.0
    min_views = max(2, int(min_views))
    seen_ids = sorted({mid for _P, corners in per_cam.values() for mid in corners})

    points: List[TriangulatedPoint] = []
    for marker_id in seen_ids:
        for corner_idx in range(4):
            obs = [
                (P, corners[marker_id][corner_idx])
                for P, corners in per_cam.values()
                if marker_id in corners
            ]
            while len(obs) >= min_views:
                pos = triangulate_dlt(obs)
                if pos is None:
                    break
                errs = reprojection_errors_px(pos, obs)
                worst = int(np.argmax(errs))
                if limit > 0.0 and errs[worst] > limit:
                    del obs[worst]
                    continue
                points.append(
                    TriangulatedPoint(
                        marker_id=marker_id,
                        corner_index=corner_idx,
                        position_mm=pos,
                        n_views=len(obs),
                        reprojection_rms_px=float(np.sqrt(np.mean(errs**2))),
                    )
                )
                break
    return points


def model_points_for_body(
    body: str,
    marker_config: MarkerConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Known 3D model points in body frame for Kabsch fitting.

    Returns:
        ``(model_pts, marker_corner_keys)`` where keys are ``(marker_id, corner_idx)``.
    """
    lo, hi = marker_config.id_ranges[body]
    obj_corners = tag_object_points(marker_config.tag_size_mm)

    model_list: List[np.ndarray] = []
    keys: List[Tuple[int, int]] = []
    for marker_id in range(lo, hi + 1):
        if marker_id not in marker_config.mounts_T_body_tag:
            continue
        T_body_tag = marker_config.mounts_T_body_tag[marker_id]
        for corner_idx, pt_tag in enumerate(obj_corners):
            pt_h = np.append(pt_tag, 1.0)
            pt_body = (T_body_tag @ pt_h)[:3]
            model_list.append(pt_body)
            keys.append((marker_id, corner_idx))
    if not model_list:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 2), dtype=object)
    return np.stack(model_list, axis=0), np.array(keys, dtype=object)


@dataclass
class KabschFit:
    """Outcome of fitting one body to triangulated corners (pose may be ``None``)."""

    body: str
    T_world_body: Optional[np.ndarray]
    rms_mm: float
    n_model: int
    n_matched: int
    n_used: int
    n_tags: int
    reprojection_rms_px: float
    fail_reason: Optional[str] = None


def fit_body_kabsch(
    triangulated: Sequence[TriangulatedPoint],
    *,
    body: str,
    marker_config: MarkerConfig,
    max_residual_mm: float = 5.0,
) -> KabschFit:
    """Trimmed Kabsch fit of ``body``; always returns diagnostics."""
    model_pts, keys = model_points_for_body(body, marker_config)
    n_model = int(model_pts.shape[0])
    tri_map = {(p.marker_id, p.corner_index): p for p in triangulated}
    idx = [i for i, key in enumerate(keys) if (int(key[0]), int(key[1])) in tri_map]
    matched = [tri_map[(int(keys[i][0]), int(keys[i][1]))] for i in idx]
    n_tags = len({p.marker_id for p in matched})

    def _fail(reason: str, rms: float = float("nan"), n_used: int = 0, reproj: float = float("nan")):
        return KabschFit(body, None, rms, n_model, len(matched), n_used, n_tags, reproj, reason)

    if n_model < 3:
        return _fail("model<3")
    if len(matched) < 3:
        return _fail(f"matched={len(matched)}<3")

    obs = np.stack([p.position_mm for p in matched], axis=0)
    T, rms, keep = kabsch_trimmed(model_pts[idx], obs)
    used = [p for p, k in zip(matched, keep) if k]
    reproj = float(np.sqrt(np.mean([p.reprojection_rms_px ** 2 for p in used])))
    if rms > max_residual_mm:
        return _fail(f"rms={rms:.1f}>{max_residual_mm:.1f}mm", rms, len(used), reproj)
    return KabschFit(body, T, rms, n_model, len(matched), len(used), n_tags, reproj)


def estimate_body_pose_kabsch(
    triangulated: Sequence[TriangulatedPoint],
    *,
    body: str,
    marker_config: MarkerConfig,
    max_residual_mm: float = 5.0,
) -> Optional[Tuple[np.ndarray, float, int]]:
    """
    Fit ``T_world←body`` from triangulated corners belonging to ``body``.

    Returns:
        ``(T_world_body, rms_mm, n_points)`` or ``None``.
    """
    fit = fit_body_kabsch(
        triangulated, body=body, marker_config=marker_config, max_residual_mm=max_residual_mm
    )
    if fit.T_world_body is None:
        return None
    return fit.T_world_body, fit.rms_mm, fit.n_used


def _cameras_seeing_body(
    detections: Dict[str, DetectionResult], body: str, marker_config: MarkerConfig
) -> int:
    return sum(
        1
        for det in detections.values()
        if any(body_for_marker_id(m.marker_id, marker_config.id_ranges) == body for m in det.markers)
    )


def estimate_world_body_poses_triangulation(
    detections: Dict[str, DetectionResult],
    *,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    marker_config: MarkerConfig,
    min_views: int = 2,
    max_residual_mm: float = 5.0,
    max_corner_reproj_px: float | None = DEFAULT_MAX_CORNER_REPROJ_PX,
    diagnostics: Optional[Dict[str, Any]] = None,
    bodies: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, WorldBodyPose], List[TriangulatedPoint]]:
    """
    Approach B: triangulate corners, trimmed-Kabsch-fit each rigid body.

    Quality is filled even when the pose is rejected (doc 03 §3.3):
    ``reprojection_rms_px`` is the corner reprojection RMS in pixels,
    ``fit_rms_mm`` the Kabsch residual, ``n_cameras`` the cameras that saw a
    tag of the body. ``diagnostics`` (optional dict) receives a
    :class:`KabschFit` per body.

    Returns body poses and the list of triangulated 3D points. The default
    bodies are ``HP``, ``MP``, ``CT``. ``bodies`` solves that list instead
    (experiment eval); ``jtsmm record`` leaves it unset.
    """
    triangulated = triangulate_tag_corners(
        detections,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        marker_config=marker_config,
        min_views=min_views,
        max_corner_reproj_px=max_corner_reproj_px,
    )

    out: Dict[str, WorldBodyPose] = {}
    for body in BODY_NAMES if bodies is None else bodies:
        if body not in marker_config.id_ranges:
            out[body] = WorldBodyPose(body, nan_pose(), 0, 0, float("nan"), False)
            continue
        fit = fit_body_kabsch(
            triangulated,
            body=body,
            marker_config=marker_config,
            max_residual_mm=max_residual_mm,
        )
        if diagnostics is not None:
            diagnostics[body] = fit
        ok = fit.T_world_body is not None
        out[body] = WorldBodyPose(
            body=body,
            T_world_body=fit.T_world_body if ok else nan_pose(),
            n_tags=fit.n_tags,
            n_cameras=_cameras_seeing_body(detections, body, marker_config),
            reprojection_rms_px=fit.reprojection_rms_px,
            valid=ok and is_valid_pose(fit.T_world_body),
            fit_rms_mm=fit.rms_mm,
        )
    return out, triangulated
