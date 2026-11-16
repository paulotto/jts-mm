# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Park–Martin IMU–array mounting rotation."""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from jtsmm.cli.imu_cmd import imu_group
from jtsmm.imu.kinematics import apply_mounting_quat, quat_from_rotation_matrix
from jtsmm.imu.mounting import (
    MAX_KABSCH_RMS_MM,
    MIN_AXIS3_RATIO,
    SELECT_N,
    MountingFitError,
    _select_pair_frames,
    fit_mounting_rotation,
    fit_session_slot,
    geodesic_deg,
    project_so3,
    rotation_from_quat,
    select_so3_fps,
    write_mounting_transform,
)


def _rx(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _wahba_opt_from_imu(R_opt: np.ndarray, R_imu: np.ndarray) -> np.ndarray:
    """Wrong absolute fit: R_opt ≈ R_w R_imu."""
    H = np.zeros((3, 3))
    for Ro, Ri in zip(R_opt, R_imu):
        H += Ro @ Ri.T
    return project_so3(H)


def _figure_eight(n: int = 60) -> list[np.ndarray]:
    out = []
    for i in range(n):
        u = i / max(n - 1, 1)
        roll = 50.0 * math.sin(2 * math.pi * u)
        pitch = 40.0 * math.sin(2 * math.pi * u + 1.1)
        yaw = 55.0 * math.sin(2 * math.pi * u + 2.2) + 25.0 * math.sin(4 * math.pi * u)
        out.append(_rz(yaw) @ _ry(pitch) @ _rx(roll))
    return out


def test_park_martin_recovers_mount_across_world_yaw():
    R_mount = _rz(22.0) @ _ry(-14.0) @ _rx(9.0)
    R_we = _rz(47.0)
    R_opt = np.stack(_figure_eight(50))
    R_imu = np.stack([R_we.T @ Ro @ R_mount for Ro in R_opt])
    t = np.arange(R_opt.shape[0], dtype=np.uint64) * 20_000
    poses = np.zeros((R_opt.shape[0], 4, 4), dtype=np.float64)
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])

    fit = fit_mounting_rotation(t, poses, t, quats, slot="imu_mta")
    assert geodesic_deg(fit.R_mount, R_mount) < 0.5
    assert fit.rms_deg < 0.5
    assert fit.axes_3d is True
    assert fit.axis_ratio_imu is not None and fit.axis_ratio_imu[1] >= 0.12
    assert np.allclose(fit.T[:3, 3], 0.0)
    assert np.allclose(fit.T[3, :], [0, 0, 0, 1])

    R_wrong = _wahba_opt_from_imu(R_opt, R_imu)
    assert geodesic_deg(R_wrong, R_mount) > 10.0


def test_nod_only_is_degenerate():
    R_mount = _rz(10.0)
    R_opt = np.stack([_rx(35.0 * math.sin(2 * math.pi * i / 40)) for i in range(40)])
    R_imu = np.stack([Ro @ R_mount for Ro in R_opt])
    t = np.arange(40, dtype=np.uint64) * 20_000
    poses = np.zeros((40, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])
    with pytest.raises(MountingFitError, match="collinear|degenerate|pairs"):
        fit_mounting_rotation(t, poses, t, quats)


def test_two_axis_motion_reports_not_3d():
    R_mount = _rz(10.0)
    R_opt = []
    for i in range(50):
        u = i / 49
        R_opt.append(_rx(45.0 * math.sin(2 * math.pi * u)) @ _ry(40.0 * math.sin(2 * math.pi * u + 1.0)))
    R_opt = np.stack(R_opt)
    R_imu = np.stack([Ro @ R_mount for Ro in R_opt])
    t = np.arange(50, dtype=np.uint64) * 150_000
    poses = np.zeros((50, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])
    fit = fit_mounting_rotation(t, poses, t, quats, strict=False)
    assert fit.axes_3d is False
    assert fit.axis_ratio_opt is not None and fit.axis_ratio_opt[1] < MIN_AXIS3_RATIO
    assert "2/3" in fit.summary()


def test_apply_mounting_quat_matches_matrix():
    R = _rz(90.0)                                    # body<-chip
    q_chip = quat_from_rotation_matrix(_rx(30.0))    # world<-chip
    q_body = apply_mounting_quat(q_chip, R)
    R_body = rotation_from_quat(q_chip) @ R.T        # world<-body
    assert geodesic_deg(rotation_from_quat(q_body), R_body) < 1e-5


def _write_slot_h5(path, *, R_mount, prefix_us, n=40):
    R_opt = np.stack(_figure_eight(n))
    R_imu = np.stack([Ro @ R_mount for Ro in R_opt])
    t = prefix_us + np.arange(n, dtype=np.uint64) * 20_000
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])
    return t, poses, quats


def test_cli_calibrate_mounting_two_phases(tmp_path):
    Rm = _ry(18.0) @ _rx(-7.0)
    Rc = _rz(-12.0) @ _ry(5.0)
    t_m, p_m, q_m = _write_slot_h5(tmp_path, R_mount=Rm, prefix_us=0)
    t_c, p_c, q_c = _write_slot_h5(tmp_path, R_mount=Rc, prefix_us=2_000_000)
    h5 = tmp_path / "session.h5"
    with h5py.File(h5, "w") as f:
        og = f.create_group("optical")
        t_all = np.concatenate([t_m, t_c])
        pose_mta = np.full((t_all.size, 4, 4), np.nan)
        pose_cra = np.full((t_all.size, 4, 4), np.nan)
        pose_mta[: t_m.size] = p_m
        pose_cra[t_m.size :] = p_c
        og.create_dataset("t_master_us", data=t_all)
        og.create_dataset("pose_mta", data=pose_mta)
        og.create_dataset("pose_cra", data=pose_cra)
        gm = f.create_group("imu_mta")
        gm.create_dataset("t_master_us_quat", data=t_m)
        gm.create_dataset("quat", data=q_m)
        gc = f.create_group("imu_cra")
        gc.create_dataset("t_master_us_quat", data=t_c)
        gc.create_dataset("quat", data=q_c)
        ph = f.create_group("phases")
        ph.create_dataset("name", data=np.array(["imu_mount_mta", "imu_mount_cra"], dtype="S"))
        ph.create_dataset(
            "t_start_us",
            data=np.array([int(t_m[0]), int(t_c[0])], dtype=np.uint64),
        )
        ph.create_dataset(
            "t_end_us",
            data=np.array([int(t_m[-1]), int(t_c[-1])], dtype=np.uint64),
        )

    runner = CliRunner()
    result = runner.invoke(
        imu_group,
        ["calibrate", "--mode", "fit", str(h5), "--no-persist-imu-calibration"],
    )
    assert result.exit_code == 0, result.output
    assert "imu_mta" in result.output
    assert "imu_cra" in result.output
    assert "axis coverage:" in result.output
    assert "3/3" in result.output
    with h5py.File(h5, "r") as f:
        Tm = np.asarray(f["calibration/imu_mta/mounting_transform"])
        Tc = np.asarray(f["calibration/imu_cra/mounting_transform"])
    assert geodesic_deg(Tm[:3, :3], Rm) < 1.0
    assert geodesic_deg(Tc[:3, :3], Rc) < 1.0
    assert np.allclose(Tm[:3, 3], 0.0)


def test_cli_calibrate_forces_optical_and_skips_emg(monkeypatch, tmp_config_path, tmp_path):
    captured: dict = {}
    h5 = tmp_path / "session.h5"
    h5.write_bytes(b"")

    def fake_run_session(*_args, **kwargs):
        captured.update(kwargs)
        return h5

    monkeypatch.setattr("jtsmm.capture.session.run_session", fake_run_session)
    runner = CliRunner()
    result = runner.invoke(
        imu_group,
        [
            "calibrate",
            "--mode",
            "record",
            "--config",
            str(tmp_config_path),
            "--mock",
            "--no-doctor",
            "--no-integrity",
            "--no-fit",
            "--no-persist-imu-calibration",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured.get("optical_track") is True
    assert captured.get("enable_emg") is False
    assert captured.get("ct_calibration") is False
    assert captured.get("persist_ct_calibration") is False
    assert captured.get("use_experiment_plan") is False
    assert captured.get("mvc") is False
    assert captured.get("experiment_plan") is not None
    assert captured.get("experiment_plan").name == "imu_mounting"
    assert captured.get("write_jts_config") is False
    assert captured.get("reproduce") is not None
    assert captured["reproduce"].imu_calibrate is True
    assert captured["reproduce"].optical_track is True
    assert captured["reproduce"].use_experiment_plan is False
    assert captured["reproduce"].mvc is False
    assert captured["reproduce"].experiment_plan is None


def test_missing_optical_pose_tells_operator_to_use_imu_calibrate(tmp_path):
    h5 = tmp_path / "cameras_only.h5"
    with h5py.File(h5, "w") as f:
        f.create_group("optical")
        g = f.create_group("imu_mta")
        g.create_dataset("t_master_us_quat", data=np.arange(10, dtype=np.uint64))
        g.create_dataset("quat", data=np.tile([1.0, 0, 0, 0], (10, 1)))
    runner = CliRunner()
    result = runner.invoke(
        imu_group,
        ["calibrate", "--mode", "fit", str(h5), "--imu", "mta", "--no-persist-imu-calibration"],
    )
    assert result.exit_code != 0
    assert "jtsmm imu calibrate" in result.output
    assert "pose_mta" in result.output


def test_write_roundtrip(tmp_path):
    R_mount = _rz(5.0)
    R_opt = np.stack(_figure_eight(30))
    t = np.arange(30, dtype=np.uint64) * 20_000
    poses = np.zeros((30, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ro @ R_mount) for Ro in R_opt])
    path = tmp_path / "s.h5"
    with h5py.File(path, "w") as f:
        og = f.create_group("optical")
        og.create_dataset("t_master_us", data=t)
        og.create_dataset("pose_mta", data=poses)
        g = f.create_group("imu_mta")
        g.create_dataset("t_master_us_quat", data=t)
        g.create_dataset("quat", data=quats)
    fit = fit_session_slot(path, "imu_mta")
    write_mounting_transform(path, fit)
    again = fit_session_slot(path, "imu_mta")
    assert geodesic_deg(again.R_mount, fit.R_mount) < 1e-6


def _slow_tumble_poses(n: int, *, dt_us: int = 150_000) -> tuple[np.ndarray, np.ndarray]:
    R_opt = np.stack(_figure_eight(n))
    t = np.arange(n, dtype=np.uint64) * np.uint64(dt_us)
    poses = np.zeros((n, 4, 4), dtype=np.float64)
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    return t, poses


def test_so3_fps_prefers_orientation_extremes_over_linspace():
    still = np.repeat(np.eye(3)[None, ...], 80, axis=0)
    motion = np.stack(_figure_eight(40))
    Rs = np.concatenate([still, motion], axis=0)
    lin = np.linspace(0, len(Rs) - 1, SELECT_N, dtype=int)
    fps = select_so3_fps(Rs, SELECT_N)
    assert int((fps >= 80).sum()) > int((lin >= 80).sum())


def test_quality_gate_drops_one_camera_views():
    t, poses = _slow_tumble_poses(80)
    outlier = np.zeros((40, 4, 4), dtype=np.float64)
    outlier[:, :3, :3] = _rz(90.0) @ _ry(90.0)
    outlier[:, 3, 3] = 1.0
    t2 = t[-1] + (np.arange(40, dtype=np.uint64) + 1) * np.uint64(150_000)
    t_all = np.concatenate([t, t2])
    poses_all = np.concatenate([poses, outlier], axis=0)
    n_cameras = np.concatenate([np.full(80, 2), np.full(40, 1)]).astype(np.uint8)
    n_tags = np.full(120, 4, dtype=np.uint8)
    rms = np.full(120, 0.4, dtype=np.float32)
    pick = _select_pair_frames(
        t_all.astype(np.int64),
        poses_all[:, :3, :3],
        n_tags=n_tags,
        n_cameras=n_cameras,
        rms=rms,
        rms_kind="kabsch_mm",
        k=SELECT_N,
    )
    assert pick.max() < 80
    assert pick.size == SELECT_N


def test_quality_gate_drops_high_kabsch_rms():
    t, poses = _slow_tumble_poses(80)
    n_cameras = np.full(80, 2, dtype=np.uint8)
    n_tags = np.full(80, 4, dtype=np.uint8)
    rms = np.full(80, 0.3, dtype=np.float32)
    rms[10:25] = MAX_KABSCH_RMS_MM + 1.0
    pick = _select_pair_frames(
        t.astype(np.int64),
        poses[:, :3, :3],
        n_tags=n_tags,
        n_cameras=n_cameras,
        rms=rms,
        rms_kind="kabsch_mm",
        k=SELECT_N,
    )
    assert not np.any((pick >= 10) & (pick < 25))


def test_omega_gate_keeps_slow_tumble_when_still_and_fast_exist():
    dt = 50_000
    n = 60
    still = np.repeat(np.eye(3)[None, ...], n, axis=0)
    slow = np.stack([_rz(40.0 * (i * dt) * 1e-6) for i in range(n)])
    fast = np.stack([_rz(250.0 * (i * dt) * 1e-6) for i in range(n)])
    Rs = np.concatenate([still, slow, fast], axis=0)
    t = np.arange(3 * n, dtype=np.int64) * dt
    pick = _select_pair_frames(t, Rs, k=SELECT_N)
    n_slow = int(((pick >= n) & (pick < 2 * n)).sum())
    assert n_slow >= SELECT_N // 2


def test_jump_gate_drops_orientation_flip():
    dt = 16_667
    n = 50
    Rs = np.stack([_rz(40.0 * (i * dt) * 1e-6) for i in range(n)])
    Rs[25] = _rz(40.0 * (25 * dt) * 1e-6 + 180.0)
    t = np.arange(n, dtype=np.int64) * dt
    pick = _select_pair_frames(t, Rs, k=min(SELECT_N, n - 1))
    assert 25 not in set(pick.tolist())


def test_fit_uses_quality_and_still_recovers_mount():
    R_mount = _rz(22.0) @ _ry(-14.0) @ _rx(9.0)
    t, poses = _slow_tumble_poses(50)
    R_opt = poses[:, :3, :3]
    R_we = _rz(47.0)
    R_imu = np.stack([R_we.T @ Ro @ R_mount for Ro in R_opt])
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])
    n_cameras = np.full(50, 2, dtype=np.uint8)
    n_tags = np.full(50, 4, dtype=np.uint8)
    rms = np.full(50, 0.5, dtype=np.float32)
    fit = fit_mounting_rotation(
        t,
        poses,
        t,
        quats,
        slot="imu_mta",
        n_tags=n_tags,
        n_cameras=n_cameras,
        reprojection_rms=rms,
        rms_kind="kabsch_mm",
    )
    assert geodesic_deg(fit.R_mount, R_mount) < 0.5
    assert fit.n_selected == SELECT_N or fit.n_samples <= SELECT_N
    assert fit.n_samples == 50
