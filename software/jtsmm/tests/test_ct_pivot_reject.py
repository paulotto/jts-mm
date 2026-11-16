# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for CT tip OLS outlier pose rejection."""

from __future__ import annotations

import numpy as np

import pytest

# ct_pivot imports the optical preview stack, which needs OpenCV.
pytest.importorskip("cv2", reason="ct_pivot needs OpenCV (optical preview)")
pytest.importorskip("optical", reason="needs the in-repo optical package")

from jtsmm.capture.ct_pivot import MAX_PER_POSE_ERR_MM, reject_high_err_poses
from optical.tracking.pivot import solve_pivot_ols


def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R
    out[:3, 3] = t
    return out


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _clean_poses(n: int = 8) -> list[np.ndarray]:
    tip = np.array([0.0, -220.0, -5.0], dtype=np.float64)
    pivot = np.array([10.0, 20.0, 300.0], dtype=np.float64)
    poses: list[np.ndarray] = []
    for i in range(n):
        R = _rot_y(15.0 * i)
        t = pivot - R @ tip
        poses.append(_T(R, t))
    return poses


def test_reject_high_err_poses_drops_outlier():
    poses = _clean_poses(8)
    # Corrupt one pose translation so its residual is large after OLS.
    poses[3] = poses[3].copy()
    poses[3][:3, 3] += np.array([25.0, 0.0, 0.0], dtype=np.float64)
    frames = [{"cam0": np.zeros((4, 4), dtype=np.uint8)} for _ in poses]
    logs: list[str] = []

    initial = solve_pivot_ols(poses)
    assert initial.max_err_mm > MAX_PER_POSE_ERR_MM

    kept, result, kept_frames, n_rej = reject_high_err_poses(
        poses,
        initial,
        frames,
        max_err_mm=MAX_PER_POSE_ERR_MM,
        log=logs.append,
    )
    assert n_rej >= 1
    assert len(kept) == len(poses) - n_rej
    assert len(kept_frames) == len(kept)
    assert result.max_err_mm <= MAX_PER_POSE_ERR_MM + 1e-6
    assert any("rejecting pose" in line for line in logs)


def test_reject_high_err_poses_noop_when_clean():
    poses = _clean_poses(6)
    initial = solve_pivot_ols(poses)
    kept, result, _frames, n_rej = reject_high_err_poses(poses, initial)
    assert n_rej == 0
    assert len(kept) == 6
    assert result.n_poses == 6
