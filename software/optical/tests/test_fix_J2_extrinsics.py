# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Extrinsics degeneracy / outlier gates, cross-view error, cache identity (runs without cv2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from _fix_J2_imports import import_optical

geometry, extrinsics, io = import_optical(
    "optical.calibration.geometry", "optical.calibration.extrinsics", "optical.calibration.io"
)

SQ = 38.52
K = np.array([[1740.0, 0.0, 728.0], [0.0, 1740.0, 544.0], [0.0, 0.0, 1.0]])
DIST = np.array([-0.08, 0.05, 0.0005, -0.0003, 0.0])
FULL = np.array([[SQ * (i + 1), SQ * (j + 1), 0.0] for j in range(4) for i in range(6)])
ROW = FULL[:6]  # one row of inner corners: collinear
T_WORLD_C1 = geometry.make_T(geometry.rodrigues([0.0, 0.35, 0.0]), [450.0, 0.0, 40.0])


def _intr(fx: float = 1740.0) -> "io.CameraIntrinsics":
    Km = K.copy()
    Km[0, 0] = Km[1, 1] = fx
    return io.CameraIntrinsics(K=Km, dist=DIST.copy(), image_size=(1456, 1088), rms_px=0.3)


def _perturb(T: np.ndarray, rng, rot_deg: float, trans_mm: float) -> np.ndarray:
    dR = geometry.rodrigues(rng.normal(0.0, np.radians(rot_deg), 3))
    return geometry.make_T(dR @ T[:3, :3], T[:3, 3] + rng.normal(0.0, trans_mm, 3))


def _view(obj, T_true, T_est, rng, *, K_img=K, pnp_rms=0.3):
    img = geometry.project_points(obj, T_true, K_img, DIST) + rng.normal(0.0, 0.2, (len(obj), 2))
    return extrinsics.BoardView(obj_points=obj, img_points=img, T_cam_board=T_est, pnp_rms_px=pnp_rms)


def _snapshots(n=30, *, seed=0, outlier_idx=(), wild_idx=(), K_cam1_img=K, pnp_bad_idx=()):
    rng = np.random.default_rng(seed)
    snaps = []
    for k in range(n):
        R = geometry.rodrigues(rng.normal(0, 0.2, 3) + [np.pi, 0, 0])
        T_c0b = geometry.make_T(R, [rng.normal(150, 40), rng.normal(-60, 30), rng.uniform(550, 800)])
        T_c1b = geometry.invert_T(T_WORLD_C1) @ T_c0b
        v0 = _view(FULL, T_c0b, _perturb(T_c0b, rng, 0.01, 0.05), rng)
        if k in outlier_idx:
            # Near-collinear partial view: PnP fits its own points but the pose is off.
            v1 = _view(ROW, T_c1b, _perturb(T_c1b, rng, 2.0, 20.0), rng)
        elif k in wild_idx:
            v1 = _view(FULL, T_c1b, _perturb(T_c1b, rng, 5.0, 60.0), rng)
        else:
            v1 = _view(FULL, T_c1b, _perturb(T_c1b, rng, 0.01, 0.05), rng, K_img=K_cam1_img)
        if k in pnp_bad_idx:
            v1.pnp_rms_px = 3.0
            v1.T_cam_board = _perturb(T_c1b, rng, 5.0, 60.0)
        snaps.append({"cam0": v0, "cam1": v1})
    return snaps


def _pose_err(T_est: np.ndarray) -> tuple[float, float]:
    E = geometry.invert_T(T_WORLD_C1) @ T_est
    return float(np.linalg.norm(E[:3, 3])), float(np.degrees(geometry.rotation_angle_rad(E[:3, :3])))


def test_outlier_and_bad_pnp_are_rejected_and_fusion_is_accurate() -> None:
    snaps = _snapshots(outlier_idx=(0,), pnp_bad_idx=(5,))
    intr = {"cam0": _intr(), "cam1": _intr()}
    fit = extrinsics.fuse_extrinsics(snaps, ["cam0", "cam1"], intr)

    assert fit.gates_passed, fit.problems
    assert not fit.inliers["cam1"][0]  # the collinear-view snapshot
    assert not fit.paired["cam1"][5]  # dropped by the PnP gate
    assert fit.n_pnp_rejected["cam1"] == 1
    assert fit.inliers["cam1"].sum() >= 25
    dt, dr = _pose_err(fit.extrinsics["cam1"].T_world_cam)
    assert dt < 1.0 and dr < 0.05
    assert fit.extrinsics["cam1"].cross_view_rms_px < 1.0
    assert np.allclose(fit.extrinsics["cam0"].T_world_cam, np.eye(4))

    # The old plain average (quaternion mean + mean translation) is pulled off by
    # the single outlier — this is what the robust fusion fixes.
    Ts = [s["cam0"].T_cam_board @ geometry.invert_T(s["cam1"].T_cam_board) for s in snaps]
    naive = geometry.make_T(geometry.mean_rotation([T[:3, :3] for T in Ts]), np.mean([T[:3, 3] for T in Ts], axis=0))
    assert _pose_err(naive)[0] > 2.0 * dt + 0.5


def test_cross_view_gate_catches_inconsistent_intrinsics() -> None:
    # cam1 images were formed with a different focal length than the one supplied.
    snaps = _snapshots(K_cam1_img=np.array([[1650.0, 0, 728.0], [0, 1650.0, 544.0], [0, 0, 1]]))
    fit = extrinsics.fuse_extrinsics(snaps, ["cam0", "cam1"], {"cam0": _intr(), "cam1": _intr()})
    assert not fit.gates_passed
    assert any("cross-view" in p for p in fit.problems)
    assert fit.extrinsics["cam1"].cross_view_rms_px > 4.0


def test_inlier_fraction_gate() -> None:
    snaps = _snapshots(n=12, wild_idx=(0, 2, 4, 6, 8))  # 5/12 wild: 7 agree < ceil(0.6*12)
    fit = extrinsics.fuse_extrinsics(snaps, ["cam0", "cam1"], {"cam0": _intr(), "cam1": _intr()})
    assert not fit.gates_passed
    assert any("agree" in p for p in fit.problems)


def test_configurable_gates_from_capture_cfg() -> None:
    g = extrinsics.ExtrinsicsGates.from_capture_cfg(
        {"max_pnp_rms_px": 0.25, "max_cross_view_rms_px": 5.0, "min_snapshots_extrinsics": 40}
    )
    assert g.max_pnp_rms_px == 0.25 and g.max_cross_view_rms_px == 5.0
    assert extrinsics.ExtrinsicsGates.from_capture_cfg(None) == extrinsics.ExtrinsicsGates()
    # Every synthetic view has PnP RMS 0.3 > 0.25 -> nothing left to pair.
    with pytest.raises(ValueError, match="No snapshot"):
        extrinsics.fuse_extrinsics(
            _snapshots(n=10), ["cam0", "cam1"], {"cam0": _intr(), "cam1": _intr()}, g
        )


def test_single_camera_is_identity_and_passes() -> None:
    snaps = [{"cam0": s["cam0"]} for s in _snapshots(n=3)]
    fit = extrinsics.fuse_extrinsics(snaps, ["cam0"], {"cam0": _intr()})
    assert fit.gates_passed
    assert np.allclose(fit.extrinsics["cam0"].T_world_cam, np.eye(4))
    assert np.isnan(fit.extrinsics["cam0"].cross_view_rms_px)


def test_board_spread_rejects_collinear_and_diagonal() -> None:
    assert not geometry.board_points_well_spread(ROW)
    assert not geometry.board_points_well_spread(FULL[:, [1, 0, 2]][:4])  # one column
    diag = np.array([[SQ * i, SQ * i, 0.0] for i in range(1, 6)])
    assert not geometry.board_points_well_spread(diag)
    assert geometry.board_points_well_spread(FULL)
    assert geometry.board_points_well_spread(FULL[:12])  # two rows
    assert not geometry.board_points_well_spread(FULL[:3])  # too few


def test_cache_paths_are_per_bundle(tmp_path: Path) -> None:
    i1, e1 = extrinsics.calibration_cache_paths(tmp_path / "calibration_bundle.h5")
    i2, e2 = extrinsics.calibration_cache_paths(tmp_path / "calibration_bundle_2cams.h5")
    assert len({i1, e1, i2, e2}) == 4
    assert i1.parent == tmp_path and i1.name == "calibration_bundle.intrinsics_cache.npz"


def test_intrinsics_fingerprint_detects_change_and_survives_npz(tmp_path: Path) -> None:
    a = {"cam0": _intr(), "cam1": _intr()}
    h = extrinsics.intrinsics_fingerprint(a)
    np.savez(tmp_path / "c.npz", K=a["cam1"].K, dist=a["cam1"].dist)
    d = np.load(tmp_path / "c.npz")
    b = {"cam0": _intr(), "cam1": io.CameraIntrinsics(K=d["K"], dist=d["dist"], image_size=(1456, 1088), rms_px=9.0)}
    assert extrinsics.intrinsics_fingerprint(b) == h  # rms_px is not part of it
    c = {"cam0": _intr(), "cam1": _intr(fx=1740.001)}
    assert extrinsics.intrinsics_fingerprint(c) != h


def test_bundle_write_is_atomic_and_stores_cross_view(tmp_path: Path) -> None:
    bundle = tmp_path / "calibration_bundle.h5"
    ext = {
        "cam0": io.CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.2, cross_view_rms_px=0.7),
        "cam1": io.CameraExtrinsics(T_world_cam=T_WORLD_C1, rms_px=0.3, cross_view_rms_px=0.7),
    }
    kw = dict(
        rig_id="r", schema_version="1.0",
        cameras_intrinsics={"cam0": _intr(), "cam1": _intr()},
        cameras_extrinsics=ext, fiducials={},
        extrinsics_quality={"cam1": {"inlier_snapshot": np.array([True, False])}},
        meta_attrs={"extrinsics_gates_passed": True},
    )
    io.write_calibration_bundle(bundle, **kw)
    loaded = io.read_calibration_bundle(bundle)
    e1 = loaded["cameras"]["cam1"]["extrinsics"]
    assert e1["cross_view_rms_px"] == pytest.approx(0.7)
    assert list(e1["quality"]["inlier_snapshot"]) == [True, False]
    assert bool(loaded["meta"]["extrinsics_gates_passed"])
    _intr_map, ext_map = io.load_cameras_from_bundle(loaded)
    assert ext_map["cam1"].cross_view_rms_px == pytest.approx(0.7)

    # A failing write leaves the previous bundle untouched and no temp file behind.
    bad = dict(kw, cameras_extrinsics={"cam0": io.CameraExtrinsics(T_world_cam=np.eye(3), rms_px=0.1)})
    with pytest.raises(ValueError):
        io.write_calibration_bundle(bundle, **bad)
    assert io.read_calibration_bundle(bundle)["cameras"]["cam1"]["extrinsics"]["cross_view_rms_px"] == pytest.approx(0.7)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calibration_bundle.h5"]


def test_project_points_matches_cv2_when_available() -> None:
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(4)
    T = geometry.make_T(geometry.rodrigues([2.9, 0.2, -0.1]), [100.0, -50.0, 700.0])
    for dist in (DIST, np.array([-0.1, 0.02, 0.001, 0.0005, -0.01, 0.01, 0.002, 0.0005])):
        ours = geometry.project_points(FULL, T, K, dist)
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        ref, _ = cv2.projectPoints(FULL, rvec, T[:3, 3], K, dist)
        assert np.allclose(ours, ref.reshape(-1, 2), atol=1e-6)
    r = rng.normal(0, 1, 3)
    assert np.allclose(geometry.rodrigues(r), cv2.Rodrigues(r)[0], atol=1e-12)
