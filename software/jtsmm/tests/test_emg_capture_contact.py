# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for EMG contact helpers (no hardware)."""

from __future__ import annotations

import numpy as np

from jtsmm.emg.capture import (
    DRL_PROBE_OPEN_RMS_UV,
    comparator_contact_off,
    drl_comparator_off,
    drl_contact_off,
    drl_probe_floating,
    flat_rail_contact_off,
    leadoff_summary,
)
from jtsmm.config.catalog import lsb_uv


def test_comparator_contact_majority():
    ok = bytes([0xC0, 0x00, 0x00])
    off = bytes([0xC0, 0x10, 0x00])  # CH0 P lead-off
    words = [off] * 2 + [ok] * 8
    loff = leadoff_summary(words, n_channels=6)
    assert loff["any_off"][0]
    assert not comparator_contact_off(loff)[0]
    words2 = [off] * 3 + [ok] * 7
    loff2 = leadoff_summary(words2, n_channels=6)
    assert comparator_contact_off(loff2)[0]


def test_flat_rail_detects_dc_leadoff_open():
    fs_uv = (1 << 23) * lsb_uv(6)
    col = np.full(200, fs_uv, dtype=np.float64)
    assert flat_rail_contact_off(col[:, None], gain=6, n_channels=1)[0]


def test_drl_comparator_bit_is_masked():
    """K8: CH7/CH8 comparator bits are invalid on this board and ignored."""
    ok = bytes([0xC0, 0x00, 0x00])
    # CH8N lead-off: LOFF_STATN bit 7 → b1 low nibble = STATN[7:4] = 8
    off = bytes([0xC0, 0x08, 0x00])
    assert not drl_comparator_off([ok] * 10)
    assert not drl_comparator_off([off] * 10)


def test_drl_probe_floating_uses_rms_not_absolute_level():
    """CH8's absolute level is meaningless (IN8P is unwired) -- only the
    isolated probe's noise/RMS (antenna-like pickup on an open pin vs a
    damped skin contact) discriminates connected from disconnected here."""
    rng = np.random.default_rng(0)

    quiet = rng.normal(200000.0, 20.0, size=256)  # low RMS, arbitrary DC offset
    floating, rms = drl_probe_floating(quiet)
    assert not floating
    assert rms < DRL_PROBE_OPEN_RMS_UV

    noisy = rng.normal(200000.0, 3000.0, size=256)  # same DC offset, high RMS
    floating, rms = drl_probe_floating(noisy)
    assert floating
    assert rms >= DRL_PROBE_OPEN_RMS_UV


def test_drl_contact_off_uses_probe_rms_only():
    ok_status = bytes([0xC0, 0x00, 0x00])
    off_status = bytes([0xC0, 0x08, 0x00])  # LOFF_STATN bit7 -- masked (K8)
    rng = np.random.default_rng(1)
    quiet_col = rng.normal(0, 5.0, size=64)

    off, notes = drl_contact_off([ok_status] * 64, quiet_col)
    assert not off
    assert notes == []

    off, notes = drl_contact_off([off_status] * 64, quiet_col)
    assert not off
    assert notes == []

    noisy_col = rng.normal(0, 3000.0, size=64)
    off, notes = drl_contact_off([ok_status] * 64, noisy_col)
    assert off
    assert any("floating" in n for n in notes)


def test_floating_noise_detection_threshold():
    rng = np.random.default_rng(0)
    noisy = rng.normal(0, 3000.0, size=500)
    rms = float(np.sqrt(np.mean(noisy ** 2)))
    assert rms > 500.0
