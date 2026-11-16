# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mounting fit persistence and robustness: accepted-only loading, gaps, so3_log at pi."""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from jtsmm.cli.imu_cmd import _fit_session
from jtsmm.imu.kinematics import quat_from_rotation_matrix
from jtsmm.imu.mounting import (
    MAX_PAIR_GAP_US,
    MountingFitResult,
    load_mounting_rotation,
    pair_optical_imu,
    so3_log,
    write_mounting_transform,
)


def _rot(axis, rad):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(rad) * k + (1 - math.cos(rad)) * (k @ k)


@pytest.mark.parametrize("axis", [(1, -1, 0), (0, 1, -1), (1, 0, -1), (1, -2, 3), (0, 0, 1)])
@pytest.mark.parametrize("angle", [math.pi, math.pi - 5e-5, math.pi - 2e-4])
def test_so3_log_near_pi_keeps_relative_signs(axis, angle):
    a = np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis)
    w = so3_log(_rot(a, angle))
    assert np.linalg.norm(w) == pytest.approx(angle, abs=1e-6)
    u = w / np.linalg.norm(w)
    if angle == math.pi:
        assert min(np.linalg.norm(u - a), np.linalg.norm(u + a)) < 1e-6   # +-a same rotation
    else:
        assert np.linalg.norm(u - a) < 1e-4


def _result(slot="imu_mta", *, accepted=True, error=None, phase=None, deg=10.0):
    R = _rot((0, 0, 1), math.radians(deg))
    T = np.eye(4)
    T[:3, :3] = R
    return MountingFitResult(slot=slot, R_mount=R, T=T, rms_deg=0.5, n_pairs=20,
                             n_samples=40, phase=phase, accepted=accepted, error=error)


def test_loader_ignores_rejected_fits(tmp_path):
    path = tmp_path / "s.h5"
    h5py.File(path, "w").close()
    write_mounting_transform(path, _result(accepted=False, error="rms too high"))
    assert load_mounting_rotation(path, "imu_mta") is None
    write_mounting_transform(path, _result(accepted=True))
    R = load_mounting_rotation(path, "mta")
    assert R is not None and np.allclose(R, _rot((0, 0, 1), math.radians(10.0)))


def test_rewrite_clears_stale_attrs(tmp_path):
    path = tmp_path / "s.h5"
    h5py.File(path, "w").close()
    write_mounting_transform(path, _result(accepted=False, error="bad", phase="imu_mount_mta"))
    write_mounting_transform(path, _result(accepted=True))
    with h5py.File(path, "r") as f:
        attrs = dict(f["calibration/imu_mta"].attrs)
    assert "error" not in attrs and "phase" not in attrs
    assert bool(attrs["accepted"]) is True


def test_pairing_does_not_interpolate_across_an_imu_gap():
    gap0, gap1 = 100_000, 100_000 + 2 * MAX_PAIR_GAP_US
    t_imu = np.concatenate([np.arange(0, gap0 + 1, 10_000), np.arange(gap1, gap1 + 100_001, 10_000)])
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (t_imu.size, 1))
    t_opt = np.arange(0, int(t_imu[-1]) + 1, 5_000, dtype=np.int64)
    poses = np.tile(np.eye(4), (t_opt.size, 1, 1))
    in_gap = (t_opt > gap0) & (t_opt < gap1)
    _, _, t_kept, _ = pair_optical_imu(t_opt, poses, t_imu, quats)
    assert in_gap.any()
    assert not np.any((t_kept > gap0) & (t_kept < gap1))
    assert t_kept.size == int(np.sum(~in_gap))


def _session(path, R_mount, n=40):
    R_opt = [_rot((1, 0.3, 0), 0.8 * math.sin(2 * math.pi * i / n))
             @ _rot((0, 1, 0.2), 0.7 * math.sin(4 * math.pi * i / n + 0.4))
             @ _rot((0.1, 0, 1), 0.6 * math.sin(2 * math.pi * i / n + 1.1)) for i in range(n)]
    t = np.arange(n, dtype=np.uint64) * np.uint64(150_000)
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :3] = np.stack(R_opt)
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(R @ R_mount) for R in R_opt])
    with h5py.File(path, "w") as f:
        og = f.create_group("optical")
        og.create_dataset("t_master_us", data=t)
        og.create_dataset("pose_mta", data=poses)
        g = f.create_group("imu_mta")
        g.create_dataset("t_master_us_quat", data=t)
        g.create_dataset("quat", data=quats)


def test_cli_fit_does_not_write_a_rejected_transform(tmp_path):
    path = tmp_path / "s.h5"
    _session(path, _rot((0, 0, 1), 0.3))
    write_mounting_transform(path, _result(accepted=True, deg=42.0))   # a good earlier fit
    rc, records = _fit_session(path, "mta", max_rms_deg=-1.0, output_path=None)
    assert rc == 1 and records["imu_mta"]["accepted"] is False
    R = load_mounting_rotation(path, "imu_mta")
    assert np.allclose(R, _rot((0, 0, 1), math.radians(42.0)))       # not overwritten
    rc, _ = _fit_session(path, "mta", max_rms_deg=3.0, output_path=None)
    assert rc == 0
    assert np.allclose(load_mounting_rotation(path, "imu_mta"), _rot((0, 0, 1), 0.3), atol=1e-9)
