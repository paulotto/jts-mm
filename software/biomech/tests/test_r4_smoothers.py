# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 E17/E26: centred moving averages use odd windows (no half-sample shift)."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.emg import mains
from jtsmm.emg.onset import OnsetParams, detector_signal
from jtsmm.emg.onset import odd_window as odd_window_jtsmm

from biomech import timebase
from biomech.emg.artifacts import MaskParams, lowfreq_level, masked_envelope
from biomech.emg.pipeline import rms_envelope
from biomech.timebase import odd_window

FS = 1000.0
N = 801
C = N // 2


def _pulse(sigma: float = 30.0) -> np.ndarray:
    """Symmetric bell centred on sample C (wider than every window: a unique peak)."""
    return np.exp(-0.5 * ((np.arange(N) - C) / sigma) ** 2)


def _assert_centred(y: np.ndarray, c: int = C, k: int = 40) -> None:
    y = np.asarray(y, float).reshape(len(y), -1)[:, 0]
    assert int(np.nanargmax(y)) == c
    np.testing.assert_allclose(y[c - k:c], y[c + 1:c + k + 1][::-1], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("w_s, fs, n", [(0.15, 2000.0, 301), (0.02, 200.0, 5), (0.05, 1000.0, 51),
                                        (0.25, 2000.0, 501), (0.0049, 1000.0, 5),
                                        (1e-4, 100.0, 1), (0.0, 1000.0, 1)])
def test_odd_window_values_and_twin(w_s, fs, n):
    assert odd_window(w_s, fs) == n == odd_window_jtsmm(w_s, fs)
    assert odd_window(w_s, fs) % 2 == 1


def test_twins_agree_on_a_grid():
    for w in np.linspace(0.0, 0.3, 61):
        for fs in (100.0, 200.0, 1000.0, 2000.0, 4000.0):
            assert odd_window(w, fs) == odd_window_jtsmm(w, fs)


def test_rms_envelope_even_window_keeps_peak():
    _assert_centred(rms_envelope(_pulse(), FS, 0.02))          # 20 samples -> 21


def test_masked_envelope_even_window_keeps_peak():
    x = _pulse()[:, None]
    _assert_centred(masked_envelope(x, np.zeros_like(x, bool), FS, 0.02))


def test_lowfreq_level_even_window_is_symmetric():
    n = 8001                                     # long: the 1 Hz high-pass edges die out
    c = n // 2
    x = np.exp(-0.5 * ((np.arange(n) - c) / 20.0) ** 2)
    y = lowfreq_level(np.repeat(x[:, None], 2, axis=1), FS, MaskParams(lf_window_s=0.05))[:, 0]
    k = 200
    np.testing.assert_allclose(y[c - k:c], y[c + 1:c + k + 1][::-1], rtol=1e-6,
                               atol=1e-9 * y.max())


def test_gradient_smoothing_even_n_is_symmetric():
    from scipy.special import erf

    t_us = np.arange(N) * 5000.0                                  # 200 Hz
    x = erf((np.arange(N) - C) / 6.0)                             # derivative: Gaussian at C
    _assert_centred(timebase.gradient(t_us, x, smooth_n=4))        # 4 -> 5


def test_gradient_short_segment_uses_odd_length():
    t_us = np.arange(6) * 5000.0
    x = np.array([0.0, 0.0, 1.0, 3.0, 1.0, 0.0])                  # 6 samples, window 9 -> 5
    g = timebase.gradient(t_us, x, smooth_n=9)
    assert np.isfinite(g).all()


@pytest.mark.parametrize("method, p", [("tkeo", OnsetParams(method="tkeo", smooth_s=0.02)),
                                       ("envelope", OnsetParams(method="envelope",
                                                                envelope_s=0.05))])
def test_onset_detector_signal_offline_is_centred(method, p):
    _assert_centred(detector_signal(_pulse(), FS, p, causal=False))


def test_onset_causal_smoother_unchanged():
    """The causal branch keeps its nominal (trailing) window."""
    x = np.zeros(200)
    x[100] = 1.0
    s = detector_signal(x, FS, OnsetParams(method="envelope", envelope_s=0.02), causal=True)
    assert np.count_nonzero(s > 0) == 20


def test_mains_burst_envelope_kernel_is_centred():
    yb = (1.0 + 4.0 * _pulse())[:, None]
    d = np.ones((N, 1))
    sw = mains._robust_weights(d, yb, np.zeros((1, 1)), win=4, n_iter=1)
    _assert_centred(-sw[:, 0])          # weights are smallest at the burst peak
