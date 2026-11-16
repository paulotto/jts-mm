# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Camera label residual survives a Pi clock that NTP is correcting.

Bench take 2026-09-25 (2 kSPS): the Pi clock ran -3187 ppm settling to -321 ppm
against the MCU; one straight line over the take read a correct labelling as a
7.7 ms residual (error). Labels are checked step by step; the residual is per
~10 s window and the slew is reported.
"""

from __future__ import annotations

import numpy as np

from jtsmm.integrity.checker import _label_residual_rms

P_US = 1e6 / 30.0


def _take(rate_ppm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frames on every trigger; the Pi clock runs rate_ppm off the MCU per frame."""
    y = np.arange(rate_ppm.size) * P_US                         # MCU trigger times
    x = np.concatenate([[0.0], np.cumsum(P_US / (1 + rate_ppm[1:] * 1e-6))])  # Pi clock
    return x + np.random.default_rng(0).normal(0, 20, x.size), y


def test_correct_labels_on_a_slewing_clock_have_a_small_residual():
    n = 9000  # 300 s
    ppm = np.full(n, -321.0)
    ppm[:600], ppm[600:1200], ppm[1200:1800] = -3187.0, -708.0, 336.0
    x, y = _take(ppm)
    global_rms = np.sqrt(np.mean((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2))
    assert global_rms > 5000                      # what the old check reported
    rms, slew = _label_residual_rms(x, y, 1 / 30)
    assert rms < 1000                             # below the warning line
    assert slew is not None and slew < -2500      # the correction is reported


def test_steady_clock_reports_small_slew():
    x, y = _take(np.full(3000, -8.0))
    rms, slew = _label_residual_rms(x, y, 1 / 30)
    assert rms < 100 and abs(slew) < 50
