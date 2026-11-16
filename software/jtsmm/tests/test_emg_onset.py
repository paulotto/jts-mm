# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Onset detectors: offline accuracy on synthetic bursts, causal decision delay."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="the onset filters need SciPy (analysis extra)")

from jtsmm.emg.onset import OnsetParams, detect_onsets, first_onset, tkeo  # noqa: E402

FS = 4000.0


def _burst(onset_s: float, n_s: float = 3.0, amp: float = 40.0, noise: float = 3.0, seed: int = 0):
    from scipy.signal import butter, sosfiltfilt

    rng = np.random.default_rng(seed)
    n = int(n_s * FS)
    x = rng.standard_normal(n) * noise
    i0 = int(onset_s * FS)
    x[i0 : i0 + int(0.5 * FS)] += rng.standard_normal(int(0.5 * FS)) * amp
    sos = butter(4, [20 / 2000, 450 / 2000], btype="band", output="sos")
    return sosfiltfilt(sos, x), i0


def test_tkeo_of_sine_is_constant():
    t = np.arange(4000) / FS
    e = tkeo(np.sin(2 * np.pi * 100 * t))
    assert np.allclose(e[1:-1], np.sin(2 * np.pi * 100 / FS) ** 2, atol=1e-9)


@pytest.mark.parametrize("method,tol_s", [("tkeo", 0.010), ("hodges_bui", 0.020),
                                          ("envelope", 0.020)])
@pytest.mark.parametrize("seed", range(6))
def test_offline_onset_accuracy(method, tol_s, seed):
    x, i0 = _burst(1.5, seed=seed)
    on = first_onset(x, FS, slice(0, int(1.0 * FS)), OnsetParams(method=method))
    assert on is not None and abs(on - i0) / FS < tol_s


@pytest.mark.parametrize("method", ["tkeo", "hodges_bui", "envelope"])
def test_causal_decision_after_onset_and_bounded(method):
    x, i0 = _burst(1.5)
    p = OnsetParams(method=method)
    dec = first_onset(x, FS, slice(0, int(1.0 * FS)), p, causal=True)
    assert dec is not None and dec >= i0 and (dec - i0) / FS < 0.08


def test_no_onset_in_noise_and_masked_samples_ignored():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(int(3 * FS)) * 3
    assert detect_onsets(x, FS, slice(0, int(FS)), OnsetParams(k_sd=8.0)).size == 0
    y, i0 = _burst(1.5)
    y[i0 : i0 + int(0.5 * FS)] = np.nan
    assert first_onset(y, FS, slice(0, int(FS)), OnsetParams()) is None


def test_separate_baseline_signal_matches_in_window_baseline():
    x, i0 = _burst(1.5)
    a = first_onset(x, FS, slice(0, int(FS)), OnsetParams())
    b = first_onset(x[int(1.2 * FS):], FS, None, OnsetParams(), baseline_signal=x[: int(FS)])
    assert b is not None and abs((b + int(1.2 * FS)) - a) <= 2
