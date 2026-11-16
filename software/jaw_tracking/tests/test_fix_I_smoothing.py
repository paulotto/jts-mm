# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""TransformationFilter with tracking holes: smooth each finite run, keep NaN."""

import numpy as np

from jts.helper import TransformationFilter


def test_filter_keeps_holes_and_smooths_segments():
    rng = np.random.default_rng(0)
    n = 60
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.linspace(0.0, 10.0, n) + rng.normal(0.0, 0.2, n)
    T[20:30] = np.nan  # long hole
    T[45] = np.nan     # single dropped frame
    T[54:57] = np.nan  # leaves a 3-frame tail run, too short to filter
    out = TransformationFilter(window_length=11, poly_order=3)(T)

    hole = ~np.isfinite(T).reshape(n, -1).all(axis=1)
    assert np.isnan(out[hole]).all()
    assert np.isfinite(out[~hole]).all()
    # A long run is actually smoothed (noise reduced vs the raw line).
    ref = np.linspace(0.0, 10.0, n)
    seg = slice(0, 20)
    assert np.std(out[seg, 0, 3] - ref[seg]) < np.std(T[seg, 0, 3] - ref[seg])
    np.testing.assert_allclose(out[:, 3, 3][~hole], 1.0)


def test_filter_without_holes_unchanged_behaviour():
    T = np.tile(np.eye(4), (15, 1, 1))
    T[:, 1, 3] = np.arange(15.0) ** 2
    out = TransformationFilter(window_length=5, poly_order=2)(T)
    np.testing.assert_allclose(out[:, 1, 3], T[:, 1, 3], atol=1e-9)  # quadratic is exact
