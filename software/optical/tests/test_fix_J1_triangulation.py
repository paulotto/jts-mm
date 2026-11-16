# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Triangulation + trimmed Kabsch (pure numpy: zero distortion skips cv2)."""

from __future__ import annotations

import numpy as np
import pytest

from optical.calibration.io import CameraExtrinsics, CameraIntrinsics
from optical.markers.config import MarkerConfig
from optical.tracking.aruco import DetectionResult, MarkerDetection
from optical.tracking.pose import rodrigues_to_matrix, rotation_angle_deg, tag_object_points
from optical.tracking.triangulation import (
    cached_projection_matrix,
    estimate_world_body_poses_triangulation,
    kabsch,
    kabsch_trimmed,
    projection_matrix,
    triangulate_dlt,
    triangulate_tag_corners,
)

K = np.array([[870.0, 0.0, 364.0], [0.0, 870.0, 272.0], [0.0, 0.0, 1.0]])
INTR = CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(728, 544), rms_px=0.3)


def _T(R=np.eye(3), t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _yaw(deg: float) -> np.ndarray:
    return rodrigues_to_matrix([0.0, np.radians(deg), 0.0])


# Three cameras on an arc, all looking at a point ~600 mm in front of cam0.
T_WORLD_CAM = {
    "cam0": np.eye(4),
    "cam1": _T(_yaw(-35.0), (400.0, 0.0, 80.0)),
    "cam2": _T(_yaw(35.0), (-400.0, 0.0, 80.0)),
}
EXTR = {cid: CameraExtrinsics(T_world_cam=T, rms_px=0.2) for cid, T in T_WORLD_CAM.items()}
INTRS = {cid: INTR for cid in T_WORLD_CAM}

MARKERS = MarkerConfig(
    dictionary="DICT_4X4_50",
    tag_size_mm=30.0,
    id_ranges={"HP": (0, 9), "MP": (10, 19), "CT": (20, 29)},
    mounts_T_body_tag={1: _T(t=(-40.5, 0, 0)), 8: _T(t=(40.5, 0, 0)), 13: _T(t=(0, 0, 0))},
)
T_WORLD_HP = _T(rodrigues_to_matrix([0.1, 0.3, -0.05]), (10.0, -5.0, 600.0))
T_WORLD_MP = _T(rodrigues_to_matrix([-0.1, 0.2, 0.0]), (-20.0, 60.0, 620.0))


def _uv(cam_id: str, X_world: np.ndarray) -> np.ndarray:
    P = projection_matrix(K, T_WORLD_CAM[cam_id])
    h = np.c_[X_world, np.ones(len(X_world))] @ P.T
    return h[:, :2] / h[:, 2:]


def _corners_world(T_world_body: np.ndarray, mid: int) -> np.ndarray:
    T = T_world_body @ MARKERS.mounts_T_body_tag[mid]
    return tag_object_points(30.0) @ T[:3, :3].T + T[:3, 3]


def _detections(rng, sigma=0.3, visible=None, corrupt=None):
    """visible: {cam: [ids]}; corrupt: (cam, marker_id, corner_idx, offset_px)."""
    visible = visible or {cid: [1, 8, 13] for cid in T_WORLD_CAM}
    dets = {}
    for cid, ids in visible.items():
        markers = []
        for mid in ids:
            T = T_WORLD_HP if mid < 10 else T_WORLD_MP
            uv = _uv(cid, _corners_world(T, mid)) + rng.normal(0, sigma, (4, 2))
            if corrupt and corrupt[0] == cid and corrupt[1] == mid:
                uv[corrupt[2]] += corrupt[3]
            markers.append(MarkerDetection(mid, uv))
        dets[cid] = DetectionResult(tuple(markers))
    return dets


def test_projection_matrix_is_cached():
    P1 = cached_projection_matrix(K, T_WORLD_CAM["cam1"])
    P2 = cached_projection_matrix(K.copy(), T_WORLD_CAM["cam1"].copy())
    assert P1 is P2
    assert np.allclose(P1, projection_matrix(K, T_WORLD_CAM["cam1"]))
    assert not P1.flags.writeable


def test_dlt_noiseless_is_exact():
    X = np.array([[12.0, -7.0, 610.0]])
    obs = [(projection_matrix(K, T_WORLD_CAM[c]), _uv(c, X)[0]) for c in ("cam0", "cam1")]
    assert np.allclose(triangulate_dlt(obs), X[0], atol=1e-6)


def test_bad_view_is_dropped_and_corner_retriangulated():
    rng = np.random.default_rng(0)
    dets = _detections(rng, corrupt=("cam2", 1, 0, np.array([15.0, -9.0])))
    pts = {(p.marker_id, p.corner_index): p for p in triangulate_tag_corners(
        dets, intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS)}
    p = pts[(1, 0)]
    assert p.n_views == 2
    assert p.reprojection_rms_px < 2.0
    assert np.linalg.norm(p.position_mm - _corners_world(T_WORLD_HP, 1)[0]) < 2.0
    assert pts[(1, 1)].n_views == 3
    # Without the per-view gate the corrupt view stays in and drags the point.
    raw = {(q.marker_id, q.corner_index): q for q in triangulate_tag_corners(
        dets, intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS, max_corner_reproj_px=0)}
    assert raw[(1, 0)].n_views == 3 and raw[(1, 0)].reprojection_rms_px > 3.0


def test_two_view_corner_over_limit_is_discarded():
    rng = np.random.default_rng(1)
    vis = {"cam0": [1, 8], "cam1": [1, 8]}
    dets = _detections(rng, visible=vis, corrupt=("cam1", 8, 2, np.array([0.0, 12.0])))
    keys = {(p.marker_id, p.corner_index) for p in triangulate_tag_corners(
        dets, intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS)}
    assert (8, 2) not in keys and len(keys) == 7


def test_kabsch_trimmed_rejects_outlier():
    rng = np.random.default_rng(2)
    model = np.r_[_corners_world(np.eye(4), 1), _corners_world(np.eye(4), 8)]
    R = rodrigues_to_matrix([0.2, -0.4, 0.1])
    t = np.array([5.0, 10.0, 600.0])
    obs = model @ R.T + t + rng.normal(0, 0.1, model.shape)
    obs[3] += np.array([0.0, 12.0, -6.0])  # one mis-triangulated corner
    T_plain, rms_plain = kabsch(model, obs)
    T, rms, keep = kabsch_trimmed(model, obs)
    assert not keep[3] and keep.sum() == 7
    assert rms < 0.3 < rms_plain
    assert rotation_angle_deg(R, T[:3, :3]) < 0.3 < rotation_angle_deg(R, T_plain[:3, :3])
    assert np.linalg.norm(T[:3, 3] - t) < 0.5


def test_kabsch_trimmed_keeps_clean_data():
    rng = np.random.default_rng(3)
    model = np.r_[_corners_world(np.eye(4), 1), _corners_world(np.eye(4), 8)]
    obs = model + np.array([1.0, 2.0, 3.0]) + rng.normal(0, 0.2, model.shape)
    _T_, _rms, keep = kabsch_trimmed(model, obs)
    assert keep.all()


def test_world_poses_quality_fields():
    rng = np.random.default_rng(4)
    vis = {"cam0": [1, 8, 13], "cam1": [1, 8, 13], "cam2": [13]}
    poses, tri = estimate_world_body_poses_triangulation(
        _detections(rng, visible=vis),
        intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS,
    )
    hp, mp, ct = poses["HP"], poses["MP"], poses["CT"]
    assert hp.valid and mp.valid and not ct.valid
    assert hp.n_cameras == 2 and mp.n_cameras == 3 and ct.n_cameras == 0
    assert hp.n_tags == 2 and mp.n_tags == 1
    # Pixels and millimetres are separate fields now.
    assert 0.0 < hp.reprojection_rms_px < 1.0
    assert 0.0 < hp.fit_rms_mm < 1.0
    assert rotation_angle_deg(hp.T_world_body[:3, :3], T_WORLD_HP[:3, :3]) < 1.0
    assert np.linalg.norm(hp.T_world_body[:3, 3] - T_WORLD_HP[:3, 3]) < 1.0
    assert len(tri) == 12


def test_rejected_pose_keeps_quality():
    rng = np.random.default_rng(5)
    poses, _ = estimate_world_body_poses_triangulation(
        _detections(rng, sigma=0.3),
        intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS,
        max_residual_mm=1e-6,  # reject everything
    )
    hp = poses["HP"]
    assert not hp.valid and np.all(np.isnan(hp.T_world_body))
    assert hp.n_tags == 2 and hp.n_cameras == 3
    assert np.isfinite(hp.fit_rms_mm) and np.isfinite(hp.reprojection_rms_px)


def test_single_camera_triangulates_nothing():
    rng = np.random.default_rng(6)
    poses, tri = estimate_world_body_poses_triangulation(
        _detections(rng, visible={"cam0": [1, 8]}),
        intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS,
    )
    assert tri == [] and not poses["HP"].valid and poses["HP"].n_cameras == 1


@pytest.mark.parametrize("min_views", [2, 3])
def test_min_views_respected(min_views):
    rng = np.random.default_rng(7)
    vis = {"cam0": [1], "cam1": [1], "cam2": []}
    pts = triangulate_tag_corners(
        _detections(rng, visible=vis),
        intrinsics=INTRS, extrinsics=EXTR, marker_config=MARKERS, min_views=min_views,
    )
    assert len(pts) == (4 if min_views == 2 else 0)
