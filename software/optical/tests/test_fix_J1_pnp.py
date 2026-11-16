# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Per-body PnP and flip resolution on synthetic projections.

Projections are numpy-only; the solver tests need cv2 and skip without it.
Geometry mirrors the lab rig: lores K (f = 870 px), 30 mm tags, HP layout of
two coplanar tags 81 mm apart, bodies ~600 mm from the camera.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from optical.calibration.io import CameraExtrinsics, CameraIntrinsics
from optical.markers.config import MarkerConfig
from optical.tracking.aruco import DetectionResult, MarkerDetection
from optical.tracking.pose import (
    matrix_to_rodrigues,
    rodrigues_to_matrix,
    rotation_angle_deg,
    tag_object_points,
)
from optical.tracking.pose_estimation import WorldBodyPose

K = np.array([[870.0, 0.0, 364.0], [0.0, 870.0, 272.0], [0.0, 0.0, 1.0]])
INTR = CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(728, 544), rms_px=0.3)


def _T(R=np.eye(3), t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot(axis: int, deg: float) -> np.ndarray:
    r = np.zeros(3)
    r[axis] = np.radians(deg)
    return rodrigues_to_matrix(r)


MARKERS = MarkerConfig(
    dictionary="DICT_4X4_50",
    tag_size_mm=30.0,
    id_ranges={"HP": (0, 9), "MP": (10, 19), "CT": (20, 29)},
    mounts_T_body_tag={1: _T(t=(-40.5, 0, 0)), 8: _T(t=(40.5, 0, 0)), 13: _T()},
)


def _project(T_cam_body: np.ndarray, T_body_tag: np.ndarray, rng, sigma: float) -> np.ndarray:
    pts = tag_object_points(30.0)
    T = T_cam_body @ T_body_tag
    pc = pts @ T[:3, :3].T + T[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:] + rng.normal(0.0, sigma, (4, 2))


def _detection(T_cam_body, ids, rng, sigma=0.5, corrupt=None) -> DetectionResult:
    markers = []
    for mid in ids:
        uv = _project(T_cam_body, MARKERS.mounts_T_body_tag[mid], rng, sigma)
        if corrupt is not None and mid == corrupt:
            uv = uv + np.array([[30.0, 0.0], [0.0, -20.0], [0.0, 0.0], [0.0, 0.0]])
        markers.append(MarkerDetection(marker_id=mid, corners=uv))
    return DetectionResult(markers=tuple(markers))


def _random_pose(rng, *, z=600.0, tilt=20.0) -> np.ndarray:
    R = _rot(1, tilt) @ _rot(0, rng.uniform(-5, 5))
    return _T(R, (rng.uniform(-60, 60), rng.uniform(-40, 40), z))


# ---------------------------------------------------------------- numpy only


def test_numpy_rodrigues_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r = rng.normal(size=3) * rng.uniform(0.0, 3.0)
        R = rodrigues_to_matrix(r)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(rodrigues_to_matrix(matrix_to_rodrigues(R)), R, atol=1e-9)


def test_prior_ages_unless_confirmed_by_fresh_evidence():
    from optical.tracking.tracker import ArucoJawTracker

    fake = SimpleNamespace(_prior_T={}, _prior_age={}, config=SimpleNamespace(pnp_prior_max_frames=2))

    def pose(by: str, valid: bool = True) -> dict:
        return {"HP": WorldBodyPose("HP", np.eye(4), 1, 1, 0.3, valid, resolved_by=by)}

    ArucoJawTracker._update_pose_prior(fake, pose("view"))
    assert fake._prior_age["HP"] == 0
    # pnp_prior_max_frames=2: the prior serves exactly the next two frames.
    assert "HP" in ArucoJawTracker.pose_prior(fake)  # frame N+1
    ArucoJawTracker._update_pose_prior(fake, pose("prior"))
    assert fake._prior_age["HP"] == 1
    assert "HP" in ArucoJawTracker.pose_prior(fake)  # frame N+2
    ArucoJawTracker._update_pose_prior(fake, pose("prior"))
    assert fake._prior_age["HP"] == 2
    assert ArucoJawTracker.pose_prior(fake) == {}  # expired: a flip cannot lock in
    ArucoJawTracker._update_pose_prior(fake, pose("cameras"))
    assert fake._prior_age["HP"] == 0
    ArucoJawTracker._update_pose_prior(fake, pose("none", valid=False))
    assert fake._prior_age["HP"] == 1


# ---------------------------------------------------------------- needs cv2


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2")


def test_default_bodies_stay_hp_mp_ct_until_caller_overrides():
    """jtsmm record omits ``bodies``; experiment eval passes its own list."""
    from optical.tracking.pose_estimation import BODY_NAMES, estimate_world_body_poses_pnp
    from optical.tracking.triangulation import estimate_world_body_poses_triangulation

    kwargs = dict(intrinsics={}, extrinsics={}, marker_config=MARKERS)
    pnp = estimate_world_body_poses_pnp({}, **kwargs)
    tri, _ = estimate_world_body_poses_triangulation({}, **kwargs)
    assert set(pnp) == set(BODY_NAMES) == {"HP", "MP", "CT"}
    assert set(tri) == set(BODY_NAMES)
    pnp_eval = estimate_world_body_poses_pnp({}, bodies=["eval_array"], **kwargs)
    tri_eval, _ = estimate_world_body_poses_triangulation({}, bodies=["eval_array"], **kwargs)
    assert set(pnp_eval) == {"eval_array"}
    assert set(tri_eval) == {"eval_array"}
    assert pnp_eval["eval_array"].valid is False
    assert tri_eval["eval_array"].valid is False


def _estimate(dets, extr, prior=None):
    from optical.tracking.pose_estimation import estimate_world_body_poses_pnp

    return estimate_world_body_poses_pnp(
        dets,
        intrinsics={cid: INTR for cid in dets},
        extrinsics=extr,
        marker_config=MARKERS,
        max_reprojection_rms_px=8.0,
        prior_T_world_body=prior,
    )


def test_two_tag_body_pnp_rotation_p95_below_5deg(cv2):
    """600 mm / 20° tilt / 0.5 px: per-tag IPPE had p95 ≈ 30°; one body solve < 5°."""
    rng = np.random.default_rng(1)
    extr = {"cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.0)}
    errs = []
    for _ in range(400):
        T = _random_pose(rng)
        out = _estimate({"cam0": _detection(T, (1, 8), rng)}, extr)["HP"]
        assert out.valid and out.n_tags == 2 and out.n_cameras == 1
        errs.append(rotation_angle_deg(T[:3, :3], out.T_world_body[:3, :3]))
    assert np.percentile(errs, 95) < 5.0


def test_single_tag_flip_resolved_by_prior(cv2):
    rng = np.random.default_rng(2)
    extr = {"cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.0)}
    with_prior, without = [], []
    for _ in range(400):
        T = _random_pose(rng)
        det = {"cam0": _detection(T, (13,), rng)}
        # Previous-frame pose: truth rotated by a few degrees of motion.
        prev = T.copy()
        prev[:3, :3] = T[:3, :3] @ rodrigues_to_matrix(rng.normal(0.0, np.radians(2.0), 3))
        a = _estimate(det, extr, prior={"MP": prev})["MP"]
        b = _estimate(det, extr)["MP"]
        with_prior.append(rotation_angle_deg(T[:3, :3], a.T_world_body[:3, :3]))
        without.append(rotation_angle_deg(T[:3, :3], b.T_world_body[:3, :3]))
    with_prior, without = np.asarray(with_prior), np.asarray(without)
    assert np.percentile(without, 95) > 15.0  # the flip is real without a prior
    assert np.mean(with_prior > 15.0) < 0.01  # ... and resolved with one
    # A single 30 mm tag at 600 mm is noise-limited at ~7° p95 even when the
    # branch is right; 10° bounds that without re-admitting flips.
    assert np.percentile(with_prior, 95) < 10.0


def test_single_tag_flip_resolved_across_cameras(cv2):
    rng = np.random.default_rng(3)
    # Stereo rig: cam1 450 mm to the right, yawed 40° toward the body (world = cam0).
    T_w_c1 = _T(_rot(1, -40.0), (450.0, 0.0, 0.0))
    extr = {
        "cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.0),
        "cam1": CameraExtrinsics(T_world_cam=T_w_c1, rms_px=0.0),
    }
    errs, flags = [], []
    for _ in range(300):
        T_w_b = _T(_rot(1, -20.0) @ _rot(0, rng.uniform(-5, 5)), (200 + rng.uniform(-30, 30), 0, 600))
        dets = {
            "cam0": _detection(T_w_b, (13,), rng),
            "cam1": _detection(np.linalg.inv(T_w_c1) @ T_w_b, (13,), rng),
        }
        out = _estimate(dets, extr)["MP"]
        assert out.n_cameras == 2
        errs.append(rotation_angle_deg(T_w_b[:3, :3], out.T_world_body[:3, :3]))
        flags.append(out.resolved_by)
    assert np.mean(np.asarray(errs) > 15.0) < 0.02
    assert set(flags) <= {"view", "cameras", "none"}


def test_corrupt_tag_is_dropped_not_fused(cv2):
    rng = np.random.default_rng(4)
    extr = {"cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.0)}
    T = _random_pose(rng, z=450.0, tilt=30.0)
    out = _estimate({"cam0": _detection(T, (1, 8), rng, sigma=0.2, corrupt=8)}, extr)["HP"]
    assert out.valid
    assert out.n_tags == 1
    assert out.reprojection_rms_px < 1.0
    assert np.linalg.norm(out.T_world_body[:3, 3] - T[:3, 3]) < 10.0


def test_no_mounted_tags_gives_nan_pose(cv2):
    rng = np.random.default_rng(5)
    extr = {"cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.0)}
    det = DetectionResult(markers=(MarkerDetection(3, _project(_random_pose(rng), np.eye(4), rng, 0.1)),))
    out = _estimate({"cam0": det}, extr)["HP"]
    assert not out.valid and out.n_cameras == 0


def test_prior_max_frames_zero_disables_the_prior():
    from optical.tracking.tracker import ArucoJawTracker

    fake = SimpleNamespace(_prior_T={}, _prior_age={}, config=SimpleNamespace(pnp_prior_max_frames=0))
    pose = {"HP": WorldBodyPose("HP", np.eye(4), 1, 1, 0.3, True, resolved_by="view")}
    ArucoJawTracker._update_pose_prior(fake, pose)
    assert ArucoJawTracker.pose_prior(fake) == {}  # optical.yaml: "0 = off"
