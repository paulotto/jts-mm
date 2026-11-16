# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Closed-form and refined X/Y registration (A = X B Y) and the frame-free jaw-in-head pose."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.validation import registration as REG


def _rand_pose(rng, rot=1.0, trans=100.0):
    return REG.make_pose(Rotation.from_rotvec(rng.normal(size=3) * rot).as_matrix(),
                         rng.normal(size=3) * trans)


def _noisy(T, rng, pos, rot_deg):
    n = len(T)
    out = T.copy()
    out[:, :3, :3] = T[:, :3, :3] @ Rotation.from_rotvec(
        rng.normal(size=(n, 3)) * np.radians(rot_deg) / np.sqrt(3)).as_matrix()
    out[:, :3, 3] += rng.normal(size=(n, 3)) * pos / np.sqrt(3)
    return out


def _scene(seed=0, n=800):
    rng = np.random.default_rng(seed)
    X, Yc, Ym = _rand_pose(rng, trans=500), _rand_pose(rng, 0.5, 50), _rand_pose(rng, 0.5, 20)
    B = REG.make_pose(Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.6).as_matrix(),
                      rng.normal(size=(n, 3)) * 50 + 800)
    Bm = REG.make_pose(Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.05).as_matrix(),
                       rng.normal(size=(n, 3)) * 2 + 700)
    A = REG.apply(X, B, Yc)
    Am = REG.apply(X, Bm, Ym)
    return rng, X, Yc, Ym, A, B, Am, Bm


def test_init_xy_exact():
    _, X, Yc, _, A, B, _, _ = _scene()
    X0, Y0 = REG.init_xy(A, B)
    for T, ref in ((X0, X), (Y0, Yc)):
        dp, ang = REG.pose_error(T[None], ref[None])
        assert np.linalg.norm(dp) < 1e-6 and ang[0] < 1e-6


def test_init_xy_robust_to_flipped_frames():
    rng, X, Yc, _, A, B, _, _ = _scene(1)
    B = B.copy()
    F = np.eye(4)
    F[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
    B[:300] = B[:300] @ F                                   # 37 % in another pose branch
    X0, Y0 = REG.init_xy(_noisy(A, rng, 0.05, 0.03), _noisy(B, rng, 0.2, 0.1))
    dp, ang = REG.pose_error(X0[None], X[None])
    assert np.linalg.norm(dp) < 1.0 and ang[0] < 0.2


def test_register_paired_refines_with_noise():
    rng, X, Yc, Ym, A, B, Am, Bm = _scene(2, n=3000)
    An, Bn = _noisy(A, rng, 0.05, 0.03), _noisy(B, rng, 0.2, 0.1)
    Amn, Bmn = _noisy(Am, rng, 0.05, 0.03), _noisy(Bm, rng, 0.2, 0.1)
    X0, Y0 = REG.init_xy(An, Bn)
    Ym0 = REG.y_given_x(Amn, Bmn, X0)
    reg = REG.register_paired({"CT": (An, Bn), "MP": (Amn, Bmn)}, X0, {"CT": Y0, "MP": Ym0},
                              sigma_pos_mm=0.2, sigma_rot_deg=0.1)
    for T, ref in ((reg.X, X), (reg.Y["CT"], Yc), (reg.Y["MP"], Ym)):
        dp, ang = REG.pose_error(T[None], ref[None])
        assert np.linalg.norm(dp) < 0.1 and ang[0] < 0.05
    se_r, se_t = reg.se("X")
    assert 0 < se_r < 0.05 and 0 < se_t < 0.1
    assert reg.rms_pos_mm["CT"] < 0.3


def test_relative_pose_pair_cancels_world():
    rng = np.random.default_rng(3)
    X, Yh, Yj = _rand_pose(rng, trans=500), _rand_pose(rng, 0.5, 20), _rand_pose(rng, 0.5, 20)
    n = 50
    head = REG.make_pose(Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.1).as_matrix(),
                         rng.normal(size=(n, 3)) * 5 + 600)
    jaw_rel = REG.make_pose(Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.2).as_matrix(),
                            rng.normal(size=(n, 3)) * 5 + [60.0, 0, -60])
    jaw = np.einsum("nij,njk->nik", head, jaw_rel)
    QH, QJ = REG.apply(X, head, Yh), REG.apply(X, jaw, Yj)
    QH[3] = np.nan
    rel_S, rel_Q = REG.relative_pose_pair(head, jaw, QH, QJ, Yh, Yj)
    dp, ang = REG.pose_error(rel_S, rel_Q)
    ok = np.isfinite(ang)
    assert ok.sum() == n - 1 and not se3.valid(rel_Q)[3]
    assert np.nanmax(np.linalg.norm(dp, axis=1)) < 1e-9 and np.nanmax(ang) < 1e-6


def test_robust_mean_rotation_ignores_minority_cluster():
    rng = np.random.default_rng(4)
    R0 = Rotation.from_rotvec([0.3, -0.2, 0.1])
    good = (R0 * Rotation.from_rotvec(rng.normal(size=(70, 3)) * 0.005)).as_matrix()
    bad = (R0 * Rotation.from_euler("x", 178, degrees=True)).as_matrix()[None].repeat(30, 0)
    M, inl = REG.robust_mean_rotation(np.concatenate([good, bad]))
    dp, ang = REG.pose_error(REG.make_pose(M, np.zeros(3))[None],
                             REG.make_pose(R0.as_matrix(), np.zeros(3))[None])
    assert ang[0] < 0.2 and inl[:70].all() and not inl[70:].any()
