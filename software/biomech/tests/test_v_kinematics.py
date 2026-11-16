# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5b / validation V2–V3: leave-one-task-out folds and the jaw-in-head mapping."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.validation import kinematics as VK
from biomech.validation.registration import make_pose
from biomech.validation.selection import Interval


def _rand_pose(rng, n=None, deg=20.0, mm=50.0):
    k = 1 if n is None else n
    R = Rotation.from_rotvec(np.radians(deg) * rng.normal(size=(k, 3))).as_matrix()
    T = np.repeat(np.eye(4)[None], k, axis=0)
    T[:, :3, :3], T[:, :3, 3] = R, mm * rng.normal(size=(k, 3))
    return T[0] if n is None else T


def test_folds_never_contain_the_evaluated_segment():
    held = [Interval(n, i * 10.0, i * 10.0 + 8.0) for i, n in enumerate(
        ["rest", "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion"])]
    folds = VK.loto_folds(held)
    assert set(folds) == {h.name for h in held}
    for name, train in folds.items():
        assert name not in {t.name for t in train}
        assert all(t.name != "rest" for t in train)          # rest never trains Y
    assert len(folds["rest"]) == 5 and len(folds["chewing"]) == 4


def test_jaw_in_head_agrees_when_the_systems_agree():
    """Qualisys poses generated from JTS-MM by T_Q = X T_S Y: zero error after mapping."""
    rng = np.random.default_rng(0)
    n = 50
    S_h, S_j = _rand_pose(rng, n), _rand_pose(rng, n)
    X, Y_h, Y_j = _rand_pose(rng), _rand_pose(rng), _rand_pose(rng)
    Q_h = np.einsum("ij,njk,kl->nil", X, S_h, Y_h)
    Q_j = np.einsum("ij,njk,kl->nil", X, S_j, Y_j)
    T_max_hp = make_pose(np.eye(3), np.array([5.0, 0, 3]))
    T_S, T_Q = VK.jaw_in_head(S_h, S_j, Q_h, Q_j, Y_h, Y_j, T_max_hp)
    inc = np.array([0.0, 19.0, -30.0])
    np.testing.assert_allclose(se3.apply(T_S, inc), se3.apply(T_Q, inc), atol=1e-9)
    # a wrong head Y shows up as an incisor error, growing with the rotation
    bad = Y_h @ make_pose(Rotation.from_euler("y", 2.0, degrees=True).as_matrix(), np.zeros(3))
    _, T_Q2 = VK.jaw_in_head(S_h, S_j, Q_h, Q_j, bad, Y_j, T_max_hp)
    assert np.linalg.norm(se3.apply(T_S, inc) - se3.apply(T_Q2, inc), axis=1).max() > 0.1
