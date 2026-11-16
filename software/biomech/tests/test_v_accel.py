# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU accelerometer vs Qualisys at the IMU site (VD17): a simulated body with a known lever
arm, chip rotation, gravity direction, bias and lag; leave-one-task-out scoring."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.validation import accel as AC
from biomech.validation.selection import Interval


def _pose(t):
    rv = np.column_stack([0.35 * np.sin(2 * np.pi * 1.1 * t), 0.25 * np.sin(2 * np.pi * 0.7 * t + 1),
                          0.30 * np.sin(2 * np.pi * 1.4 * t + 2)])
    p = np.column_stack([8 * np.sin(2 * np.pi * 1.2 * t), 6 * np.cos(2 * np.pi * 0.9 * t),
                         1200 + 10 * np.sin(2 * np.pi * 1.5 * t)])
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = Rotation.from_rotvec(rv).as_matrix(), p
    return T


def test_accelerometer_comparison_recovers_the_sensor():
    r_Q = np.array([20.0, -35.0, 15.0])
    R_true = Rotation.from_euler("xyz", [10, -100, 30], degrees=True).as_matrix()  # chip<-body
    g_Q = Rotation.from_euler("xy", [8, -5], degrees=True).apply([0, 0, -9806.65])
    bias, lag = np.array([60.0, -250.0, 120.0]), 0.010
    scale = np.array([1.02, 0.98, 1.01])                               # per-axis MEMS scale
    t_q = np.arange(0.0, 60.0, 0.005)                                   # Qualisys 200 Hz
    T_Q = _pose(t_q)
    t_a = np.arange(0.0, 60.0, 1 / 250.0)                               # accelerometer 250 Hz
    h = 1e-3
    def site(t):
        return _pose(t)[:, :3, 3] + np.einsum("nij,j->ni", _pose(t)[:, :3, :3], r_Q)
    tt = t_a - lag                                                      # f(τ) = model(τ − lag)
    acc_w = (site(tt + h) - 2 * site(tt) + site(tt - h)) / h ** 2
    R_b = _pose(tt)[:, :3, :3]
    f = scale * np.einsum("ij,nkj,nk->ni", R_true, R_b, acc_w - g_Q) + bias
    W = Rotation.from_euler("z", 40, degrees=True).as_matrix()          # IMU world vs Q world

    def R_imu_at(t):
        return np.einsum("ij,njk,lk->nil", W, _pose(t - lag)[:, :3, :3], R_true)

    names = ["open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion"]
    folds = [Interval(n, 2 + 10 * i, 10 + 10 * i) for i, n in enumerate(names)]
    rest = [Interval("rest", 52.0, 58.0)]
    R0 = Rotation.from_rotvec([0.05, -0.03, 0.04]).as_matrix() @ R_true   # prior a few ° off
    p = {**AC.DEFAULTS, "lag_grid_ms": [0.0, 25.0, 5.0], "fit_scale": True}
    rows, cal, out = AC.evaluate_body(t_q, T_Q, r_Q, t_a, f, R_imu_at, R0, folds, rest, p, 6.0)
    by = {(r["body_fold"], r["variant"]): r for r in rows}
    for n in names:
        s, nl, ow = by[(n, "sensor")], by[(n, "no_lever")], by[(n, "own_orientation")]
        assert s["rms_err_mm_s2"] < 0.05 * s["rms_ref_mm_s2"]               # sensor recovered
        assert s["r2_dyn"] > 0.99 and s["coherence"] > 0.95
        assert nl["rms_err_mm_s2"] > 3 * s["rms_err_mm_s2"]                # lever arm matters
        assert ow["rms_err_mm_s2"] < 0.05 * s["rms_ref_mm_s2"]              # perfect own tilt
    c = [x for x in cal if x["variant"] == "sensor"]
    assert all(abs(x["lag_ms"] - 10.0) < 1e-6 for x in c)
    assert all(abs(x["bias_y_mm_s2"] - bias[1]) < 20 for x in c)          # bias recovered
    assert all(abs(x["scale_x"] - scale[0]) < 0.005 for x in c)            # scale recovered
    assert all(x["prior_vs_fit_deg"] > 2.0 for x in c)                     # the prior moved
    assert np.isfinite(out["dyn_ref"]).any() and np.isfinite(out["dyn_own"]).any()


def test_primary_model_without_scale_on_an_ideal_sensor():
    """The primary model (no scale fit) is exact for an accelerometer without scale error."""
    t = np.arange(0.0, 40.0, 0.005)
    T = _pose(t)
    a = np.random.default_rng(0).normal(0, 300, (len(t), 3))
    R_true = Rotation.from_euler("xyz", [5, 60, -20], degrees=True).as_matrix()
    g = Rotation.from_euler("x", 12, degrees=True).apply([0, 0, -9806.65])
    f = np.einsum("ij,nkj,nk->ni", R_true, T[:, :3, :3], a - g) + np.array([30.0, 0, -80])
    R, g_hat, b, sc, rms = AC.fit(a, T[:, :3, :3], f, np.eye(3) @ R_true, 5)
    assert rms < 1e-6 and np.allclose(sc, 1.0) and np.allclose(g_hat, g, atol=1e-3)
