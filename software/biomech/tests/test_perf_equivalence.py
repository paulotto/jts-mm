# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Performance rewrites give the same (or better-converged) results (W3)."""

from __future__ import annotations

import numpy as np

from biomech.kinematics.coupling import hinge_fits
from biomech.kinematics.hinge_fit import _rt_predict, _unit, best_angles, rotate


def _rt_alternating(A, B, n, s, u, max_iter=20000, tol_mm=1e-13):
    """The pre-R3 angle/translation alternation, run to full convergence (reference)."""
    tau = np.zeros(A.shape[0])
    for _ in range(max_iter):
        th = best_angles(A - s, B - s - tau[:, None, None] * u, n)
        rot = rotate(A - s, n, th) + s
        new = ((B - rot) @ u).mean(axis=1)
        done = np.nanmax(np.abs(new - tau)) < tol_mm
        tau = new
        if done:
            break
    return rot + tau[:, None, None] * u, th, tau


def _synthetic(seed=0, F=200, K=5, noise=0.3):
    rng = np.random.default_rng(seed)
    n = _unit(np.array([0.1, 1.0, 0.05]))
    s = np.array([5.0, 0.0, -3.0])
    u = _unit(np.array([1.0, 0.0, -0.6]))
    P = rng.normal(size=(K, 3)) * 40
    th = np.radians(np.linspace(0, 30, F))
    tau = np.linspace(0, 8, F)
    A = np.broadcast_to(P, (F, K, 3)).copy()
    B = rotate(A - s, n, th) + s + tau[:, None, None] * u + rng.normal(size=(F, K, 3)) * noise
    return A, B, n, s, u


def test_rt_predict_equals_converged_alternation():
    A, B, n, s, u = _synthetic()
    for dn, ds, du in ((0, 0, 0), ([0.05, 0, 0.1], [3, 1, 4], [0, 0.2, 0.1]),
                       ([-0.2, 0.1, 0.3], [-10, 5, 2], [0.5, -0.3, 0.4])):
        nn, ss, uu = _unit(n + np.asarray(dn)), s + np.asarray(ds), _unit(u + np.asarray(du))
        p1, t1, z1 = _rt_alternating(A, B, nn, ss, uu)
        p2, t2, z2 = _rt_predict(A, B, nn, ss, uu)
        np.testing.assert_allclose(p2, p1, atol=1e-9)
        np.testing.assert_allclose(t2, t1, atol=1e-11)
        np.testing.assert_allclose(z2, z1, atol=1e-9)


def test_rt_predict_is_a_minimum_per_frame():
    A, B, n, s, u = _synthetic(seed=3)
    pred, th, tau = _rt_predict(A, B, n, s, u)
    cost = ((pred - B) ** 2).sum(axis=(1, 2))
    for d in (1e-4, -1e-4):
        rot = rotate(A - s, n, th + d) + s
        t2 = ((B - rot) @ u).mean(axis=1)
        c2 = ((rot + t2[:, None, None] * u - B) ** 2).sum(axis=(1, 2))
        assert (c2 >= cost - 1e-9).all()


def test_warm_started_fits_match_cold_fits():
    A, B, _, _, _ = _synthetic(seed=1, noise=0.2)
    # condyle-like points 3 and 4 for the cold start (hinge_fits uses them as the axis guess)
    cold = hinge_fits(A, B)
    warm = hinge_fits(A, B, init=cold)
    for k in cold:
        # Optimizer tolerance; rot_trans has a flat valley (axis height vs translation),
        # so the start may move along it without changing the fit.
        assert abs(warm[k].rms(0) - cold[k].rms(0)) < 1e-4
        if "cost" in cold[k].extra:
            assert warm[k].extra["cost"] <= cold[k].extra["cost"] * (1 + 1e-6)
