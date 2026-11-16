# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for CT tip pivot OLS."""

from __future__ import annotations

import numpy as np
import pytest

from optical.tracking.pivot import apply_tip_offset, solve_pivot_ols


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def test_solve_pivot_ols_recovers_known_tip():
    tip = np.array([12.5, -3.0, 80.0], dtype=np.float64)
    pivot = np.array([100.0, 50.0, 200.0], dtype=np.float64)
    angles = [
        (_rot_x(10), _rot_y(5)),
        (_rot_x(-20), _rot_z(15)),
        (_rot_y(30), _rot_z(-10)),
        (_rot_x(40), _rot_y(-25)),
        (_rot_z(50), _rot_x(-15)),
        (_rot_y(-35), _rot_z(40)),
        (_rot_x(5), _rot_y(60)),
        (_rot_z(-45), _rot_y(20)),
        (_rot_x(-50), _rot_z(25)),
        (_rot_y(15), _rot_x(35)),
        (_rot_z(70), _rot_y(-40)),
        (_rot_x(25), _rot_z(-55)),
        (_rot_y(-10), _rot_z(5)),
        (_rot_x(-5), _rot_y(-50)),
        (_rot_z(20), _rot_x(45)),
        (_rot_y(55), _rot_z(-30)),
        (_rot_x(60), _rot_y(10)),
        (_rot_z(-20), _rot_y(35)),
        (_rot_x(-30), _rot_z(60)),
        (_rot_y(40), _rot_x(-40)),
    ]
    poses = []
    for Ra, Rb in angles:
        R = Ra @ Rb
        t = pivot - R @ tip
        poses.append(_T(R, t))

    result = solve_pivot_ols(poses)
    assert result.n_poses == 20
    np.testing.assert_allclose(result.tip_body_mm, tip, atol=1e-6)
    np.testing.assert_allclose(result.pivot_world_mm, pivot, atol=1e-6)
    assert result.rms_mm < 1e-6


def test_solve_pivot_ols_rejects_too_few():
    with pytest.raises(ValueError, match="≥3"):
        solve_pivot_ols([np.eye(4), np.eye(4)])


def test_apply_tip_offset():
    tip = np.array([1.0, 2.0, 3.0])
    T = _T(_rot_z(90), np.array([10.0, 0.0, 0.0]))
    out = apply_tip_offset(T, tip)
    # R_z(90) @ [1,2,3] + [10,0,0] = [-2,1,3] + [10,0,0] = [8,1,3]
    np.testing.assert_allclose(out[:3, 3], [8.0, 1.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(out[:3, :3], T[:3, :3])
