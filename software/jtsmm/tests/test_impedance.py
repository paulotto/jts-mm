# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for lead-off tone impedance estimation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jtsmm.emg.impedance import (
    AC_LEADOFF_HZ,
    CONTACT_Z_MAX_KOHM,
    R_PROTECT_KOHM,
    contact_off,
    estimate_drl_impedance,
    estimate_impedance,
    goertzel_amplitude,
    leadoff_tone_hz,
)
from jtsmm.config.catalog import DRL_CHANNEL, DRL_STREAM_INDEX


def test_leadoff_tone_hz_is_fixed_31_25hz():
    # FLEAD_OFF=10 is fCLK/2^16 (2.048 MHz / 65536 = 31.25 Hz, "31.2 Hz" in the
    # datasheet), not derived from fDR, so it does not depend on the sample rate.
    assert leadoff_tone_hz() == pytest.approx(31.25)
    assert leadoff_tone_hz(2000) == pytest.approx(31.25)
    assert leadoff_tone_hz(250) == pytest.approx(31.25)
    assert AC_LEADOFF_HZ == 2.048e6 / 2**16


def test_goertzel_finds_sine():
    fs = 2000.0
    tone = AC_LEADOFF_HZ
    # 5000 samples = 78 exact cycles of 31.2 Hz at 2 kSPS (coherent sampling,
    # i.e. the tone lands exactly on a Goertzel bin) so this isolates the
    # Goertzel math itself from window scalloping loss (covered separately).
    n = 5000
    t = np.arange(n) / fs
    sig = 50.0 * np.sin(2 * math.pi * tone * t)
    amp = goertzel_amplitude(sig, fs, tone)
    assert amp == pytest.approx(50.0, rel=0.05)


def test_goertzel_bounded_scalloping_loss_off_bin():
    """A tone that does NOT land on an exact bin (the common case for a
    wall-clock capture window) must still be within the Hann window's ~15%
    worst-case scalloping loss, not the ~36% a rectangular window would give."""
    fs = 2000.0
    tone = AC_LEADOFF_HZ
    n = 4000  # 62.4 cycles -> deliberately off-bin (~0.4 of a bin width)
    t = np.arange(n) / fs
    sig = 50.0 * np.sin(2 * math.pi * tone * t)
    amp = goertzel_amplitude(sig, fs, tone)
    assert amp == pytest.approx(50.0, rel=0.2)


def test_estimate_impedance_known_tone():
    fs = 2000.0
    tone = AC_LEADOFF_HZ
    current_na = 6.0
    z_target = 20.0  # kΩ electrode + R_PROTECT_KOHM loop resistance
    v_peak = (z_target + R_PROTECT_KOHM) * current_na
    n = 5000  # coherent (see test_goertzel_finds_sine)
    t = np.arange(n) / fs
    col = v_peak * np.sign(np.sin(2 * math.pi * tone * t))  # +I / -I square (SBAS499 §9.3.2.4.3.2)
    samples = np.column_stack([col, col * 0.5])
    results = estimate_impedance(samples, fs, current_na=current_na, n_channels=2)
    assert results[0].z_kohm == pytest.approx(z_target, rel=0.2)
    assert results[0].quality in ("good", "fair")
    assert results[1].z_kohm == pytest.approx(10.0 - R_PROTECT_KOHM, rel=0.2, abs=0.05)


def test_estimate_impedance_silence_is_na_not_a_contact_verdict():
    """No tone = AC lead-off not injected; that says nothing about the contact."""
    fs = 2000.0
    samples = np.zeros((1000, 2))
    results = estimate_impedance(samples, fs, current_na=6.0, n_channels=2)
    assert all(r.z_kohm is None for r in results)
    assert all(r.quality == "n/a" for r in results)
    assert all(r.ok for r in results)


def test_contact_off_threshold():
    assert not contact_off(20.0)
    assert not contact_off(CONTACT_Z_MAX_KOHM)
    assert contact_off(CONTACT_Z_MAX_KOHM + 1.0)
    assert contact_off(None)


def test_estimate_drl_impedance_not_on_streamed_ch8():
    fs = 2000.0
    samples = np.zeros((1000, 8))
    result = estimate_drl_impedance(
        samples,
        fs,
        stream_index=DRL_STREAM_INDEX,
        channel=DRL_CHANNEL,
        current_na=6.0,
    )
    assert result.channel == DRL_CHANNEL
    assert result.z_kohm is None
    assert result.quality == "n/a"
    assert result.ok
    assert "bias node" in result.note


def test_low_z_is_good_quality():
    fs = 2000.0
    tone = AC_LEADOFF_HZ
    current_na = 6.0
    z_target = 3.0
    v_peak = (z_target + R_PROTECT_KOHM) * current_na
    n = 5000  # coherent (see test_goertzel_finds_sine)
    t = np.arange(n) / fs
    col = v_peak * np.sign(np.sin(2 * math.pi * tone * t))  # +I / -I square (SBAS499 §9.3.2.4.3.2)
    col += 200.0 * np.sin(2 * math.pi * 50.0 * t)  # strong 50 Hz (old SNR rule failed here)
    results = estimate_impedance(col[:, None], fs, current_na=current_na, n_channels=1)
    assert results[0].z_kohm == pytest.approx(z_target, rel=0.2, abs=0.1)
    assert results[0].quality == "good"


def test_format_impedance_table_drl_row():
    from jtsmm.emg.impedance import ImpedanceResult, format_impedance_table

    rows = [
        ImpedanceResult(1, 10.0, 60.0, "good"),
        ImpedanceResult(8, 12.0, 72.0, "good"),
    ]
    text = format_impedance_table(rows, channel_names=["ch1", "", "", "", "", "", "", "drl"])
    assert "DRL" in text
    assert "drl" in text


def test_estimate_impedance_high_z_is_poor_contact():
    fs = 250.0
    tone = AC_LEADOFF_HZ
    current_na = 6.0
    z_target = 80.0
    v_peak = z_target * current_na
    n = 1250  # 78 exact cycles of 31.2 Hz at 250 SPS (coherent sampling)
    t = np.arange(n) / fs
    col = v_peak * np.sign(np.sin(2 * math.pi * tone * t))  # +I / -I square (SBAS499 §9.3.2.4.3.2)
    results = estimate_impedance(col[:, None], fs, current_na=current_na, n_channels=1)
    assert results[0].quality == "poor"
    assert contact_off(results[0].z_kohm)


def test_r_protect_deembedded_from_total_loop_impedance():
    """A loop of only the on-board series network is ~0 kOhm after de-embedding
    -- not a skin contact, so it reads as "tone absent" (n/a), never "good 0.0"."""
    fs = 2000.0
    tone = AC_LEADOFF_HZ
    current_na = 6.0
    v_peak = R_PROTECT_KOHM * current_na  # pure series-resistor loop, no skin
    n = 5000  # coherent (see test_goertzel_finds_sine)
    t = np.arange(n) / fs
    col = v_peak * np.sign(np.sin(2 * math.pi * tone * t))  # +I / -I square (SBAS499 §9.3.2.4.3.2)
    results = estimate_impedance(col[:, None], fs, current_na=current_na, n_channels=1)
    assert results[0].z_kohm is None
    assert results[0].quality == "n/a"
    assert "tone absent" in results[0].note
