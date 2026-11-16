# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""solve_intrinsics: a degenerate or outlier view is dropped instead of losing the session."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from optical.calibration import multicam  # noqa: E402

SQ = 38.52
K_TRUE = np.array([[1740.0, 0, 728.0], [0, 1740.0, 544.0], [0, 0, 1]])
D_TRUE = np.array([-0.1, 0.05, 0.0, 0.0, 0.0])
FULL = np.array([[SQ * (i + 1), SQ * (j + 1), 0] for j in range(4) for i in range(6)], np.float32)


def _views(n: int = 14, seed: int = 1):
    rng = np.random.default_rng(seed)
    objs, imgs = [], []
    for _ in range(n):
        r = rng.normal(0, 0.35, 3) + [np.pi, 0, 0]
        t = np.array([rng.normal(-100, 40), rng.normal(-80, 40), rng.uniform(450, 800)])
        p, _ = cv2.projectPoints(FULL, r, t, K_TRUE, D_TRUE)
        objs.append(FULL.copy())
        imgs.append((p + rng.normal(0, 0.15, p.shape)).astype(np.float32))
    return objs, imgs


def test_collinear_and_outlier_views_are_dropped() -> None:
    objs, imgs = _views()
    # A collinear view (one row) — makes plain calibrateCamera assert.
    row = FULL[:6].copy()
    p, _ = cv2.projectPoints(row, np.array([0.2, -0.3, 0.1]), np.array([-100.0, -50.0, 600.0]), K_TRUE, D_TRUE)
    objs.insert(3, row)
    imgs.insert(3, p.astype(np.float32))
    # An outlier view: corners mismatched by several px (motion blur / wrong ids).
    bad = imgs[7].copy() + np.random.default_rng(9).normal(0, 4.0, imgs[7].shape).astype(np.float32)
    imgs[7] = bad

    sol = multicam.solve_intrinsics(objs, imgs, (1456, 1088))
    assert not sol.kept[3] and not sol.kept[7]
    assert sol.kept.sum() == len(objs) - 2
    assert len(sol.dropped) == 2
    assert sol.rms_px < 0.5
    assert abs(sol.K[0, 0] - 1740.0) < 15.0
    assert sol.std_intrinsics.size >= 4 and np.all(np.isfinite(sol.std_intrinsics[:4]))


def test_view_that_makes_opencv_raise_is_left_out() -> None:
    objs, imgs = _views(n=8)
    poison = 5
    real = multicam._calibrate_views

    def flaky(o, i, size):
        if any(x is imgs[poison] for x in i):
            raise cv2.error("simulated assertion")
        return real(o, i, size)

    with patch.object(multicam, "_calibrate_views", side_effect=flaky):
        sol = multicam.solve_intrinsics(objs, imgs, (1456, 1088))
    assert not sol.kept[poison] and sol.kept.sum() == 7
    assert "calibrateCamera failed" in sol.dropped[0]


def test_too_few_views_is_a_clear_error() -> None:
    objs, imgs = _views(n=2)
    with pytest.raises(SystemExit, match="usable calibration views"):
        multicam.solve_intrinsics(objs, imgs, (1456, 1088))
