# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""DRL / CH8: per-channel scaling (gain 24) and an informational bias readback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import jtsmm.emg.check as check_mod
from jtsmm.config.catalog import AUX_CHANNEL_GAIN, channel_lsb_uv, lsb_uv
from jtsmm.emg.capture import (
    DRL_RAIL_NA_FRAC,
    capture_emg_uv,
    drl_readback_status,
)
from jtsmm.protocol.types import RecordType


class _FakeCap:
    def __init__(self, rows):
        self._rows = rows

    def drain_records(self):
        rows, self._rows = self._rows, []
        return rows


def _rec(t_us, channels):
    return SimpleNamespace(
        record_type=int(RecordType.EMG_BLOCK),
        t_master_us=t_us,
        decoded={"channels": channels, "status": b"\xc0\x00\x00"},
    )


def test_capture_scales_ch7_ch8_at_gain_24_and_returns_timestamps():
    rows = [_rec(1000 + 500 * i, [100] * 8) for i in range(4)]
    uv, status, t_us = capture_emg_uv(_FakeCap(rows), 0.05, gain=6)
    assert uv.shape == (4, 8)
    assert uv[0, 0] == pytest.approx(100 * lsb_uv(6), rel=1e-6)
    assert uv[0, 7] == pytest.approx(100 * lsb_uv(AUX_CHANNEL_GAIN), rel=1e-6)
    assert list(t_us) == [1000, 1500, 2000, 2500]
    assert len(status) == 4


def test_readback_status_is_na_when_railed():
    assert drl_readback_status(0.0, DRL_RAIL_NA_FRAC + 0.01) == "n/a"
    assert drl_readback_status(0.0, 0.0) == "ok"
    assert drl_readback_status(50.0, 0.0) == "high"


def _fake_sessions(drl_uv: np.ndarray):
    """Muscle channels quiet; CH8 = drl_uv on the bias-on captures, quiet on the probe."""
    n = drl_uv.size
    emg = {"default_gain": 6, "sample_rate_sps": 2000.0}

    def _session(*_a, probe_drl=False, **_k):
        s = np.zeros((n, 8), dtype=np.float32)
        if not probe_drl:
            s[:, 7] = drl_uv
        t = (np.arange(n) * 500).astype(np.int64)
        return check_mod._Capture(s, [], emg, t)

    return _session


def test_swinging_bias_node_is_reported_but_does_not_fail_the_check(monkeypatch):
    t = np.arange(4000) / 2000.0
    swing = (60_000.0 * np.sqrt(2)) * np.sin(2 * np.pi * 50.0 * t)  # 60 mV RMS, unrailed
    monkeypatch.setattr(check_mod, "_hardware_session", _fake_sessions(swing.astype(np.float32)))
    res = check_mod.run_emg_check(
        {"mcu": {"emg": {"bias": {"enabled": True}}}}, port="/dev/null", with_impedance=False,
    )
    sat = res.drl_saturation
    assert sat is not None and sat.status == "high"
    assert sat.ac_mv == pytest.approx(60.0, rel=0.02)
    assert sat.ok is True
    assert res.passed is True           # informational until bench-verified
    assert "INFO high" in check_mod.format_check_table(res)


def test_railed_bias_node_reads_na_not_pass(monkeypatch):
    lsb8 = channel_lsb_uv(6)[7]
    pinned = np.full(4000, (1 << 23) * 0.99 * lsb8, dtype=np.float32)
    pinned[:100] = 0.0                  # 97.5% railed
    monkeypatch.setattr(check_mod, "_hardware_session", _fake_sessions(pinned))
    res = check_mod.run_emg_check(
        {"mcu": {"emg": {"bias": {"enabled": True}}}}, port="/dev/null", with_impedance=False,
    )
    assert res.drl_saturation.status == "n/a"
    assert res.drl_saturation.rail_frac > 0.9
    assert "n/a" in check_mod.format_check_table(res)


def test_viewer_scales_drl_column_and_judges_its_saturation_at_gain_24():
    from jtsmm.viewer.hub import Hub
    from jtsmm.viewer.sources import _counts_to_uv

    uv = _counts_to_uv(np.full((2, 8), 1000), 6)
    assert uv[0, 0] == pytest.approx(1000 * lsb_uv(6), rel=1e-6)
    assert uv[0, 7] == pytest.approx(1000 * lsb_uv(24), rel=1e-6)

    hub = Hub(emg_fs=500.0, emg_lsb_uv=lsb_uv(6), n_channels=7, display_fs=500.0,
              filter_enabled=False)
    # CH8's full scale in uV is a quarter of the muscle one, so judged with the
    # muscle LSB a railed CH8 looked like 25% of full scale and never saturated.
    rows = np.zeros((10, 7), dtype=np.float32)
    rows[:, 6] = 0.5 * (1 << 23) * lsb_uv(24)
    hub.append_emg_raw(np.arange(10, dtype=np.uint64) * 2000, rows)
    assert hub.snapshot().saturation[6] is False
    rows[:, 6] = 0.995 * (1 << 23) * lsb_uv(24)
    hub.append_emg_raw(np.arange(10, 20, dtype=np.uint64) * 2000, rows)
    snap = hub.snapshot()
    assert snap.saturation[6] is True
    assert snap.saturation[0] is False


def test_powered_down_channels_are_listed_as_not_checked(monkeypatch):
    """--active-channels 1,2,3: CH4-6 carry no signal; a flat trace must not
    read as a clean PASS."""
    monkeypatch.setattr(check_mod, "_hardware_session", _fake_sessions(np.zeros(4000, np.float32)))
    res = check_mod.run_emg_check(
        {"mcu": {"emg": {"bias": {"enabled": True}, "active_channels": [1, 2, 3]}}},
        port="/dev/null", with_impedance=False,
    )
    by_ch = {c.channel: c for c in res.channels}
    for ch in (4, 5, 6):
        assert by_ch[ch].noise_rms_uv is None
        assert any("not checked" in n for n in by_ch[ch].notes)
    for ch in (1, 2, 3):
        assert not any("not checked" in n for n in by_ch[ch].notes)
