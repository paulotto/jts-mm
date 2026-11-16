# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical↔IMU timing guard: catches a constant frame↔trigger misassignment."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from jtsmm.imu.kinematics import quat_from_rotation_matrix
from jtsmm.integrity.checker import check_session
from jtsmm.integrity.optical_imu_sync import estimate_lag

from tests.helpers import make_session_h5

OPT_HZ = 30.0
IMU_HZ = 330.0


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _attitude(t_s: np.ndarray) -> np.ndarray:
    """Figure-eight tumble with a non-constant angular speed."""
    roll = 0.9 * np.sin(2 * np.pi * 0.30 * t_s)
    pitch = 0.7 * np.sin(2 * np.pi * 0.23 * t_s + 1.1)
    yaw = 1.1 * np.sin(2 * np.pi * 0.17 * t_s + 2.2)
    return np.stack(
        [_rz(z) @ _ry(p) @ _rx(r) for r, p, z in zip(roll, pitch, yaw)]
    )


def _streams(duration_s: float = 30.0, *, optical_lag_us: int = 0):
    """Optical poses and IMU quats of one rigid body, mount rotation applied.

    ``optical_lag_us`` stamps the optical rows that many µs *earlier* than the
    motion they show — the signature of a stale trigger anchor.
    """
    t_opt = np.arange(0, duration_s, 1.0 / OPT_HZ)
    t_imu = np.arange(0, duration_s, 1.0 / IMU_HZ)
    R_opt = _attitude(t_opt)
    R_mount = _rz(0.4) @ _rx(-0.2)
    R_world = _rz(0.9)
    R_imu = np.stack([R_world.T @ R @ R_mount for R in _attitude(t_imu)])

    poses = np.zeros((t_opt.size, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(R) for R in R_imu])
    t_opt_us = (t_opt * 1e6).astype(np.int64) - int(optical_lag_us)
    t_imu_us = (t_imu * 1e6).astype(np.int64)
    return t_opt_us, poses, t_imu_us, quats


def test_estimate_lag_is_zero_when_synced():
    t_o, poses, t_i, quats = _streams()
    est = estimate_lag(t_o, poses, t_i, quats)
    assert est is not None
    assert abs(est.lag_us) < 10_000, est
    assert est.correlation > 0.9


def test_estimate_lag_recovers_injected_offset():
    for lag_ms in (-250, 100, 500):
        t_o, poses, t_i, quats = _streams(optical_lag_us=lag_ms * 1000)
        est = estimate_lag(t_o, poses, t_i, quats)
        assert est is not None, lag_ms
        assert abs(est.lag_us - lag_ms * 1000) < 15_000, (lag_ms, est)


def test_estimate_lag_inconclusive_when_static():
    t_opt = np.arange(0, 20.0, 1.0 / OPT_HZ)
    t_imu = np.arange(0, 20.0, 1.0 / IMU_HZ)
    poses = np.tile(np.eye(4), (t_opt.size, 1, 1))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (t_imu.size, 1))
    est = estimate_lag(
        (t_opt * 1e6).astype(np.int64), poses,
        (t_imu * 1e6).astype(np.int64), quats,
    )
    assert est is None


def _write_optical_imu(path: Path, *, optical_lag_us: int) -> None:
    make_session_h5(path, duration_s=1.0)
    t_o, poses, t_i, quats = _streams(optical_lag_us=optical_lag_us)
    base = 5_000_000
    with h5py.File(path, "a") as f:
        if "optical" in f:
            del f["optical"]
        og = f.create_group("optical")
        og.create_dataset("t_master_us", data=t_o + base)
        og.create_dataset("pose_mta", data=poses)
        og.create_dataset("pose_cra", data=poses)
        og.create_dataset("pose_ct", data=poses)
        for slot in ("imu_mta", "imu_cra"):
            g = f[slot]
            for key in ("t_master_us_quat", "quat"):
                if key in g:
                    del g[key]
            g.create_dataset("t_master_us_quat", data=t_i + base)
            g.create_dataset("quat", data=quats)


def test_check_session_flags_constant_trigger_offset(tmp_path: Path):
    path = tmp_path / "bad_anchor.h5"
    _write_optical_imu(path, optical_lag_us=500_000)
    report = check_session(path)
    assert not report.ok
    assert any("optical vs IMU timing offset" in e for e in report.issues)
    sync = report.summary["optical_imu_sync"]["imu_mta"]
    assert abs(sync["lag_ms"] - 500.0) < 20.0


def test_check_session_accepts_aligned_streams(tmp_path: Path):
    path = tmp_path / "good_anchor.h5"
    _write_optical_imu(path, optical_lag_us=0)
    report = check_session(path)
    assert not any("optical vs IMU timing offset" in e for e in report.issues)
    assert abs(report.summary["optical_imu_sync"]["imu_cra"]["lag_ms"]) < 15.0
