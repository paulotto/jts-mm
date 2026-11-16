# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm emg check` measurements: timestamps, measured grid frequency, line peak.

Each case reproduces a number from the 2026-09 audit: with dropped samples and
a grid that is not exactly 50.000 Hz, the old check reported a quiet electrode
as hundreds of uV of "noise" and read the line tone at ~50% of its amplitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.emg.check import _bandpass_rms_no_mains, _capture_line_hz, _line_peak
from jtsmm.emg.mains import (
    estimate_line_hz,
    fit_tone_amplitude,
    harmonic_numbers,
    measure_line_hz,
    remove_mains,
)

FS = 2000.0


def _dropouts(n: int, frac: float, *, burst: int = 6, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = np.ones(n, dtype=bool)
    for s in rng.integers(0, n - burst, size=int(frac * n / burst)):
        keep[s : s + burst] = False
    return keep


def _hum_capture(*, f0=50.0, hum_uv=1000.0, noise_uv=3.0, seconds=5.0, drop=0.009, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    x = hum_uv * np.sin(2 * np.pi * f0 * t + 0.4) + rng.normal(0.0, noise_uv, t.size)
    keep = _dropouts(t.size, drop)
    return (t[keep] * 1e6).astype(np.int64), x[keep]


def test_timestamps_keep_dropouts_from_smearing_the_mains_fit():
    """Audit: 1 mV hum + 3 uV noise + 0.9% dropouts -> 570 uV without t_us."""
    t_us, x = _hum_capture()
    blind = _bandpass_rms_no_mains(x, FS, 20.0, 450.0, 50.0)
    timed = _bandpass_rms_no_mains(x, FS, 20.0, 450.0, 50.0, t_us)
    # White 3 uV RMS over 0-1 kHz carries sqrt(430/1000) of it in 20-450 Hz.
    truth = 3.0 * np.sqrt(430.0 / 1000.0)
    assert blind > 100.0                     # what the check used to report
    assert timed == pytest.approx(truth, rel=0.15)


def test_measured_grid_frequency_is_what_the_fit_needs():
    """Audit: grid at 50.016 Hz, fit at nominal 50.000 -> 72 uV residual vs 3 uV."""
    t_us, x = _hum_capture(f0=50.016, drop=0.0)
    at_nominal = _bandpass_rms_no_mains(x, FS, 20.0, 450.0, 50.0, t_us)
    f = _capture_line_hz(x[:, None], FS, 50.0, t_us, n_channels=1)
    assert f == pytest.approx(50.016, abs=0.003)
    at_measured = _bandpass_rms_no_mains(x, FS, 20.0, 450.0, f, t_us)
    truth = 3.0 * np.sqrt(430.0 / 1000.0)
    assert at_nominal > 5.0 * truth
    assert at_measured == pytest.approx(truth, rel=0.15)


@pytest.mark.parametrize("f0", [49.986, 50.020])
def test_measured_frequency_with_dropouts(f0):
    t_us, x = _hum_capture(f0=f0)
    f = measure_line_hz(x, FS, 50.0, t_us=t_us)
    assert f == pytest.approx(f0, abs=0.004)


def test_clean_room_falls_back_to_nominal():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 5.0, int(5 * FS))
    assert measure_line_hz(x, FS, 50.0) is None
    assert _capture_line_hz(x[:, None], FS, 50.0, None, n_channels=1) == 50.0


@pytest.mark.parametrize("seconds", [5.0, 20.0])
def test_log_power_interpolation_bias_is_small(seconds):
    """Parabola on linear power was biased up to 0.014 Hz at 5 s; log power ~0.002."""
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * FS)) / FS
    tol = 0.003 if seconds == 5.0 else 0.001
    for f0 in np.arange(49.98, 50.0301, 0.005):
        x = 1000.0 * np.sin(2 * np.pi * f0 * t + 0.3) + rng.normal(0.0, 3.0, t.size)
        assert estimate_line_hz(x, FS) == pytest.approx(f0, abs=tol), f0


@pytest.mark.parametrize("n", [1000, 2048, 4001, 10000, 10007])
def test_line_peak_reads_the_real_amplitude(n):
    """The zero-padded FFT divided by the padded length and read ~50%."""
    t = np.arange(n) / FS
    x = 100.0 * np.sin(2 * np.pi * 50.0 * t + 1.0) + 20_000.0  # electrode DC offset
    assert _line_peak(x, FS, 50.0) == pytest.approx(100.0, rel=0.01)


def test_line_peak_with_timestamps_and_dropouts():
    t_us, x = _hum_capture(hum_uv=100.0, noise_uv=0.0, drop=0.03)
    assert _line_peak(x, FS, 50.0, t_us) == pytest.approx(100.0, rel=0.01)


def test_fit_tone_amplitude_rejects_a_strong_neighbour_when_asked():
    t = np.arange(int(2 * FS)) / FS
    x = 50.0 * np.sin(2 * np.pi * 31.25 * t) + 2000.0 * np.sin(2 * np.pi * 50.013 * t)
    joint = fit_tone_amplitude(x, FS, 31.25, nuisance_hz=[50.013, 100.026])
    assert joint == pytest.approx(50.0, rel=1e-3)


def test_ninth_harmonic_of_a_fast_grid_is_still_removed():
    """50.02 Hz x 9 = 450.18 Hz: outside a strict < 450 cutoff, inside the band's leakage."""
    assert 9 in harmonic_numbers(50.02, FS)
    assert 9 in harmonic_numbers(50.0, FS)
    assert harmonic_numbers(50.0, 250.0) == [1, 2]      # never at/over Nyquist
    t = np.arange(int(5 * FS)) / FS
    x = 200.0 * np.sin(2 * np.pi * 9 * 50.02 * t)
    y, _ = remove_mains(x, FS, line_hz=50.02)
    assert np.std(y) < 1.0
