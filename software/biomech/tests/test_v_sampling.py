# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EV7: the movement-bandwidth spectrum of the unfiltered 200 Hz opening angle with the rest
noise floor subtracted, and the reported peak ratio (open_close opening velocity)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.validation import sampling as S

FS = 200.0
A2, A25, SIGMA = 10.0, 0.05, 0.05          # deg: 2 Hz motion, 25 Hz motion, white noise


def _signal(seed: int = 0):
    """0-15 s rest (noise only), 15-75 s task: 2 Hz + 25 Hz opening angle + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 75, 1 / FS)
    task = t >= 15
    th = np.where(task, A2 * np.sin(2 * np.pi * 2 * t) + A25 * np.sin(2 * np.pi * 25 * t), 0.0)
    return t, th + rng.normal(0, SIGMA, len(t))


def _expected_share() -> float:
    """Share above 20 Hz of the opening-rate power of the noise-free angle through the
    central difference (gain sin(2π f Δt)/Δt)."""
    dt = 1 / FS
    p2 = (A2 * np.sin(2 * np.pi * 2 * dt) / dt) ** 2 / 2
    p25 = (A25 * np.sin(2 * np.pi * 25 * dt) / dt) ** 2 / 2
    return p25 / (p2 + p25)


def test_central_rate():
    t = np.arange(0, 1, 0.01)
    r = S.central_rate(t, 3 * t ** 2)
    assert np.isnan(r[0]) and np.isnan(r[-1])
    assert np.allclose(r[1:-1], 6 * t[1:-1])


def test_unfiltered_spectrum_recovers_share_above_20hz():
    t, th = _signal()
    sp = S.spectrum_unfiltered("s01", t, th, [("open_close", 15.0, 75.0)], [(0.0, 15.0)])
    r = sp.iloc[0]
    true = _expected_share()                                     # about 0.0032
    assert r.noise_subtracted and r.noise_nperseg == 512 and r.n_segments > 40
    assert r.share_above_20hz_unfilt == pytest.approx(true, rel=0.25)
    assert r.f50_unfilt_hz == pytest.approx(2.0, abs=0.5)
    assert r.f95_unfilt_hz < 3.0 and r.f99_unfilt_hz < 26.0
    assert 0.0 < r.noise_power_frac < 0.05
    # without the floor the white angle noise (rate PSD ~ sin², mostly above 20 Hz) inflates it
    no = S.spectrum_unfiltered("s01", t, th, [("open_close", 15.0, 75.0)], [])
    assert not no.iloc[0].noise_subtracted
    assert no.iloc[0].share_above_20hz_unfilt > 2 * true


def test_unfiltered_spectrum_short_rest_and_gaps():
    """A rest shorter than one segment gives a shorter floor segment interpolated onto the
    task frequencies; NaN gaps split the task into runs (no bridging). Clipping the power
    below the floor at 0 keeps the positive noise fluctuations, so the share is biased
    upwards (conservative), more so with a short, noisy floor."""
    t, th = _signal(1)
    th = th.copy()
    th[(t > 40) & (t < 40.5)] = np.nan
    iv = [("open_close", 15.0, 75.0)]
    r = S.spectrum_unfiltered("s01", t, th, iv, [(0.0, 2.5)]).iloc[0]
    assert r.noise_subtracted and r.noise_nperseg == 256
    true = _expected_share()
    assert true < r.share_above_20hz_unfilt < 1.6 * true
    no = S.spectrum_unfiltered("s01", t, th, iv, []).iloc[0]
    assert no.share_above_20hz_unfilt > 2.5 * true


def test_filtered_spectrum_kept_under_its_name():
    """``spectrum`` keeps the pre-EV7 computation (on the given, filtered rate)."""
    t = np.arange(0, 6, 1 / FS)
    rate = 100 * np.sin(2 * np.pi * 3 * t)
    sp = S.spectrum("s01", t, rate, [("open_close", 0.0, 6.0)]).iloc[0]
    assert {"f50_hz", "f95_hz", "f99_hz", "share_above_20hz"} <= set(sp.index)


def test_peak_ratio_open_is_the_dataset_chain_open_close_opening_velocity():
    b = pd.DataFrame([
        {"pid": p, "task": task, "pipeline": pipe, "quantity": q, "n": 5,
         "ratio_median": 0.8, "ratio_ci_lo": 0.7, "ratio_ci_hi": 0.9}
        for p in ("s01", "s02") for task in ("open_close", "chewing")
        for pipe in ("ref40", "jtsmm")
        for q in ("peak_open_vel_dps", "peak_close_vel_dps", "peak_incisal_speed_mm_s")])
    r = S.peak_ratio_open(b)
    assert len(r) == 2 and list(r["pid"]) == ["s01", "s02"]
    assert set(r["task"]) == {"open_close"} and set(r["pipeline"]) == {"ref40"}
    assert set(r["quantity"]) == {"peak_open_vel_dps"}
    assert S.peak_ratio_open(pd.DataFrame()).empty
