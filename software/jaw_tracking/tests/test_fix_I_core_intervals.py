# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""MultiIntervalProcessor: NaN poses at a phase boundary must not abort a run."""

import numpy as np

from jts.core import FrameInterval, InterpolationMethod, MultiIntervalProcessor


def _track(start, n):
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = np.arange(start, start + n, dtype=float)
    return T


def _run(first, second, tf=4):
    data = {0: first, 20: second}
    proc = MultiIntervalProcessor(InterpolationMethod.CUBIC, transition_frames=tf)
    ivs = [FrameInterval(0, len(first) - 1, "a"), FrameInterval(20, 20 + len(second) - 1, "b")]
    return proc, proc.process_intervals(lambda iv: data[iv.start], ivs, connect=True)


def test_nan_at_boundary_uses_nearest_finite_frames():
    a, b = _track(0, 10), _track(20, 10)
    a[-2:] = np.nan  # lost lock at the end of phase a
    b[0] = np.nan    # ... and at the start of phase b
    proc, T = _run(a, b)
    assert T.shape[0] == 10 + 4 + 10
    bridge = T[10:14]
    assert np.isfinite(bridge).all()
    # Bridge runs between the last finite pose (x=7) and the first finite (x=21).
    assert 7.0 < bridge[0, 0, 3] < bridge[-1, 0, 3] < 21.0
    # Measured NaN frames stay NaN; they are not papered over.
    assert np.isnan(T[8:10]).all() and np.isnan(T[14]).all()
    np.testing.assert_array_equal(proc.source_frames[10:14], -1)


def test_all_nan_side_joins_without_transition():
    a, b = _track(0, 5), np.full((5, 4, 4), np.nan)
    proc, T = _run(a, b)
    assert T.shape[0] == 10
    np.testing.assert_array_equal(proc.source_frames, np.r_[np.arange(5), np.arange(20, 25)])
