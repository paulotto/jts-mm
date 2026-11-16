# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""known-signal tests for the R7 analysis additions -- E29 anti-alias filter,
the VD10 opening angle of a known rotation, and the VD11 "CMC undefined" case."""

from __future__ import annotations

import numpy as np


def test_e29_antialias_removes_content_above_the_coarse_nyquist():
    from biomech.imu.analysis import _antialias

    t_us = (np.arange(4000) * 5000).astype(np.int64)          # 200 Hz
    t = t_us * 1e-6
    slow, fast = np.sin(2 * np.pi * 3.0 * t), np.sin(2 * np.pi * 35.0 * t)
    y = _antialias(t_us, slow + fast, 16.0)
    core = slice(400, -400)
    assert np.std((y - slow)[core]) < 0.05                       # 35 Hz gone, 3 Hz kept
    # without it, 35 Hz on a 40 Hz grid folds onto 5 Hz -- the error E29 removed
    coarse = (slow + fast)[::5][80:-80] - slow[::5][80:-80]
    assert np.std(coarse) > 0.5


def test_vd10_opening_angle_of_a_known_rotation_about_y():
    from biomech.validation.evaluate import _opening_deg

    n = 200
    th = np.r_[np.zeros(50), np.linspace(0, 25, 150)]            # rest, then open to 25 deg
    M = np.tile(np.eye(4), (n, 1, 1))
    r = np.radians(th)
    M[:, 0, 0] = M[:, 2, 2] = np.cos(r)
    M[:, 0, 2], M[:, 2, 0] = np.sin(r), -np.sin(r)
    rest = np.r_[np.ones(50, bool), np.zeros(150, bool)]
    assert np.allclose(_opening_deg(M, rest), th, atol=1e-9)
    assert np.isnan(_opening_deg(M, np.zeros(n, bool))).all()   # no rest: no angle


def test_vd11_cmc_is_undefined_when_level_exceeds_shape():
    from biomech.stats.waveforms import cmc

    x = np.linspace(0, 2 * np.pi, 101)
    similar = np.vstack([np.sin(x), 1.05 * np.sin(x) + 0.02])
    assert 0.9 < cmc(similar) <= 1.0
    offset = np.vstack([0.1 * np.sin(x), 0.1 * np.sin(x) + 5.0])  # level >> shape
    assert np.isnan(cmc(offset))                                 # reported as "undefined"
