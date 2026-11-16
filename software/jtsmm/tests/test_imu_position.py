# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical + accelerometer position fusion (``jtsmm.imu.position``) on a synthetic plate: a
known trajectory, an accelerometer in a tilted, rotating chip frame (gravity and a bias
included), the IMU world rotated against the optical world."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from jtsmm.imu import position as PO
from jtsmm.imu.fusion import Calibration, ImuTrack

FS_OPT, FS_ACC, T_END = 40.0, 250.0, 20.0


def _p(t):          # mm: jaw-like motion, a few cm at 1-2 Hz
    return np.column_stack([20 * np.sin(2 * np.pi * 1.1 * t), 8 * np.sin(2 * np.pi * 0.7 * t + 1),
                            15 * np.sin(2 * np.pi * 1.6 * t + 0.3)])


def _a(t):          # its second derivative, mm/s²
    w = 2 * np.pi * np.array([1.1, 0.7, 1.6])
    ph = np.array([0.0, 1.0, 0.3])
    amp = np.array([20.0, 8.0, 15.0])
    return -(amp * w ** 2) * np.sin(np.outer(t, w) + ph)


@pytest.fixture(scope="module")
def plate():
    rng = np.random.default_rng(4)
    t_opt = np.arange(0.0, T_END, 1 / FS_OPT)
    t_acc = np.arange(0.0, T_END, 1 / FS_ACC)
    Z = Rotation.from_rotvec([0.0, 0.0, 0.7]).as_matrix()       # IMU world -> optical world
    rot = Rotation.from_rotvec(np.column_stack([0.3 * np.sin(t_acc), 0.2 * np.cos(0.5 * t_acc),
                                                0.5 * np.sin(0.3 * t_acc)]))
    R_imu = rot.as_matrix()                                     # chip -> IMU world
    a_imuworld = (Z.T @ _a(t_acc).T).T / 1000.0                 # m/s²
    bias_chip = np.array([0.05, -0.08, 0.12])                   # m/s²
    f = np.einsum("nji,nj->ni", R_imu, a_imuworld + PO.G_MPS2) + bias_chip \
        + rng.normal(0, 0.02, (len(t_acc), 3))
    q = rot.as_quat()[:, [3, 0, 1, 2]]                          # w-first
    imu = ImuTrack.from_stream((t_acc * 1e6).astype(np.int64), q)
    cal = Calibration(Z=Z, X=np.eye(3), lag_s=0.0, drift_rad_s=np.zeros(3), t_ref_s=0.0)
    p_opt = _p(t_opt) + rng.normal(0, 0.2, (len(t_opt), 3))
    return {"t_opt": t_opt, "p_opt": p_opt, "t_acc": t_acc, "f": f, "imu": imu, "cal": cal}


def test_gravity_is_removed_and_the_world_mapped(plate):
    ta, a = PO.world_acceleration(plate["cal"], plate["imu"], plate["t_acc"], plate["f"])
    err = a - _a(ta)
    # residual = the chip bias seen in the world (|b| ≈ 150 mm/s²) plus noise -- not gravity
    assert np.max(np.abs(np.median(err, axis=0))) < 250.0
    assert np.abs(err).max() < 9806.0 * 0.1


def test_smoother_bridges_a_gap_far_better_than_interpolation(plate):
    t, p = plate["t_opt"], plate["p_opt"]
    ta, a = PO.world_acceleration(plate["cal"], plate["imu"], plate["t_acc"], plate["f"])
    rest = [(0.0, 0.0)]
    st = PO.settings_from_data(t, p, np.ones(len(t), bool), rest, ta, a)
    st.r_pos_mm = 0.2
    gap = (t >= 10.0) & (t < 10.5)
    ok = ~gap
    ps, pf = PO.position_kf(t, p, ok, ta, a, st, t)
    truth = _p(t)
    lin = np.column_stack([np.interp(t, t[ok], p[ok, i]) for i in range(3)])
    e_kf = np.linalg.norm(ps[gap] - truth[gap], axis=1).max()
    e_lin = np.linalg.norm(lin[gap] - truth[gap], axis=1).max()
    assert e_kf < 0.25 * e_lin and e_kf < 3.0
    assert np.isfinite(pf[gap]).all()
    # away from the gap the fused position stays at the optical accuracy
    assert np.sqrt(np.mean(np.sum((ps[~gap] - truth[~gap]) ** 2, axis=1))) < 0.5


def test_window_limits_the_filter(plate):
    t, p = plate["t_opt"], plate["p_opt"]
    ta, a = PO.world_acceleration(plate["cal"], plate["imu"], plate["t_acc"], plate["f"])
    st = PO.PositionSettings(0.2, 300.0 ** 2 / FS_OPT, 400.0, {})
    ps, _ = PO.position_kf(t, p, np.ones(len(t), bool), ta, a, st, t, window=(5.0, 8.0))
    inside = (t >= 5.0) & (t <= 8.0)
    assert np.isfinite(ps[inside]).all() and np.isnan(ps[~inside]).all()


def test_per_measurement_sd(plate):
    """``r_meas_mm``: equal to ``r_pos_mm`` everywhere gives the common-SD result; a much
    larger SD on part of the frames makes the filter follow those frames less."""
    t, p = plate["t_opt"], plate["p_opt"]
    ta, a = PO.world_acceleration(plate["cal"], plate["imu"], plate["t_acc"], plate["f"])
    st = PO.PositionSettings(0.2, 300.0 ** 2 / FS_OPT, 400.0, {})
    ok = np.ones(len(t), bool)
    ref, _ = PO.position_kf(t, p, ok, ta, a, st, t)
    same, _ = PO.position_kf(t, p, ok, ta, a, st, t, r_meas_mm=np.full(len(t), 0.2))
    np.testing.assert_allclose(same, ref)
    bad = (t >= 10.0) & (t < 10.5)
    p_off = p.copy()
    p_off[bad] += 20.0                                     # 20 mm off on those frames
    trusted, _ = PO.position_kf(t, p_off, ok, ta, a, st, t)
    weak, _ = PO.position_kf(t, p_off, ok, ta, a, st, t,
                             r_meas_mm=np.where(bad, 50.0, 0.2))
    truth = _p(t)
    e_t = np.linalg.norm(trusted[bad] - truth[bad], axis=1).mean()
    e_w = np.linalg.norm(weak[bad] - truth[bad], axis=1).mean()
    assert e_w < 0.25 * e_t
