# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""|Z| from the AC lead-off tone: exact 31.25 Hz, tone-presence gate, live n/a."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jtsmm.emg.impedance import (
    AC_LEADOFF_HZ,
    APPLY_SQUARE_WAVE_CORRECTION,
    R_PROTECT_KOHM,
    SQUARE_WAVE_FUNDAMENTAL_GAIN,
    estimate_impedance,
)
from jtsmm.viewer.hub import Hub

FS = 2000.0


def _tone_col(z_kohm, seconds, *, i_na=6.0, clock=1.0, noise=1.0, hum=300.0, seed=0, drop=0.0):
    """One channel with the tone at 31.25 Hz in ADC-clock time; t_us in host time."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    k = np.arange(n)
    t_adc = k / FS                                   # ADC clock time
    t_host = t_adc / clock                           # host sees fs * clock
    v = (z_kohm + R_PROTECT_KOHM) * i_na
    # The chip sources, then sinks I (SBAS499 §9.3.2.4.3.2): a +-I*Z square wave.
    x = (v * np.sign(np.sin(2 * math.pi * AC_LEADOFF_HZ * t_adc + 0.7))
         + hum * np.sin(2 * math.pi * 50.01 * t_host)
         + rng.normal(0.0, noise, n) + 15_000.0)
    keep = np.ones(n, dtype=bool)
    if drop:
        for s in rng.integers(0, n - 8, size=int(drop * n / 8)):
            keep[s : s + 8] = False
    return (t_host[keep] * 1e6).astype(np.int64), x[keep]


def test_constants():
    assert AC_LEADOFF_HZ == 31.25
    assert SQUARE_WAVE_FUNDAMENTAL_GAIN == pytest.approx(4 / math.pi)
    assert APPLY_SQUARE_WAVE_CORRECTION is True         # SBAS499 §9.3.2.4.3.2: +I / -I square


@pytest.mark.parametrize("seconds", [2.0, 2.3, 3.0, 3.7])
def test_no_scalloping_across_capture_lengths(seconds):
    """Integer-bin Goertzel: 2 s vs 3 s differed 13%. Exact frequency: <1%."""
    t_us, x = _tone_col(25.0, seconds)
    r = estimate_impedance(x[:, None], FS, current_na=6.0, n_channels=1, t_us=t_us)[0]
    assert r.z_kohm == pytest.approx(25.0, rel=0.01)
    assert r.quality == "good"


def test_internal_oscillator_offset_is_tracked_via_the_sample_clock():
    # ADC clock 0.8% fast: tone at 31.5 Hz in host time. Fitting 31.25 would lose ~40%.
    t_us, x = _tone_col(25.0, 3.0, clock=1.008, drop=0.02)
    r = estimate_impedance(x[:, None], FS, current_na=6.0, n_channels=1, t_us=t_us)[0]
    assert r.z_kohm == pytest.approx(25.0, rel=0.02)


def test_noise_only_is_tone_absent_not_good_zero():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 5.0, (int(2 * FS), 2)) + 300.0 * np.sin(
        2 * math.pi * 50.0 * np.arange(int(2 * FS)) / FS
    )[:, None]
    for r in estimate_impedance(x, FS, current_na=6.0, n_channels=2):
        assert r.z_kohm is None
        assert r.quality == "n/a"
        assert "tone absent" in r.note


def test_hub_shows_na_when_ac_tone_is_off():
    hub = Hub(emg_fs=FS, emg_lsb_uv=0.1, n_channels=7, display_fs=500.0,
              filter_enabled=False, leadoff_during_recording=True, leadoff_ac_tone=False)
    hub.set_impedance_enabled(True)
    t = (np.arange(4000) * 500).astype(np.uint64)
    hub.append_emg_raw(t, np.random.default_rng(0).normal(0, 5, (4000, 7)).astype(np.float32))
    rows = hub.snapshot().impedance
    assert rows is not None and len(rows) == 6
    assert all(r["z_kohm"] is None and r["quality"] == "n/a" for r in rows)


def test_hub_live_impedance_uses_timestamps():
    hub = Hub(emg_fs=FS, emg_lsb_uv=0.1, n_channels=7, display_fs=500.0,
              filter_enabled=False, leadoff_during_recording=True, leadoff_ac_tone=True)
    hub.set_impedance_enabled(True)
    t_us, x = _tone_col(30.0, 2.0, drop=0.02)
    rows = np.tile(x[:, None], (1, 7)).astype(np.float32)
    for s in range(0, x.size, 200):
        hub.append_emg_raw(t_us[s : s + 200].astype(np.uint64), rows[s : s + 200])
    imp = hub.snapshot().impedance
    assert imp is not None
    assert imp[0]["z_kohm"] == pytest.approx(30.0, rel=0.03)


def test_current_override_reaches_the_mcu_config(monkeypatch):
    import jtsmm.emg.check as check_mod

    seen = {}

    def _session(resolved, **_k):
        seen["na"] = resolved["mcu"]["emg"]["leadoff"]["current_na"]
        t_us, x = _tone_col(25.0, 2.0, i_na=24.0)
        s = np.zeros((x.size, 8), dtype=np.float32)
        s[:, 0] = x
        return check_mod._Capture(s, [], {"sample_rate_sps": FS}, t_us)

    monkeypatch.setattr(check_mod, "_hardware_session", _session)
    cfg = {"mcu": {"emg": {"leadoff": {"current_na": 6}, "bias": {"enabled": False}}}}
    results, meta = check_mod.run_impedance(cfg, port="/dev/null", current_na=24)
    assert seen["na"] == 24
    assert cfg["mcu"]["emg"]["leadoff"]["current_na"] == 6      # caller's dict untouched
    assert results[0].z_kohm == pytest.approx(25.0, rel=0.02)


def test_run_impedance_fits_the_configured_line_frequency(monkeypatch):
    """A 60 Hz site: the mains harmonics next to the tone must be fitted at 60 Hz."""
    import numpy as np

    import jtsmm.emg.check as check
    import jtsmm.emg.impedance as imp

    seen = {}

    def fake_session(resolved, **kw):
        n = 2000
        return check._Capture(np.zeros((n, 8)), [], {"sample_rate_sps": 1000}, np.arange(n) * 1000)

    def fake_estimate(samples, fs, **kw):
        seen["line_hz"] = kw.get("line_hz")
        return []

    monkeypatch.setattr(check, "_hardware_session", fake_session)
    monkeypatch.setattr(imp, "estimate_impedance", fake_estimate)
    monkeypatch.setattr(imp, "estimate_drl_impedance", lambda *a, **k: None)
    resolved = {"mcu": {"emg": {"bias": {"enabled": False}}}, "host": {"emg_check": {"line_hz": 60.0}}}
    check.run_impedance(resolved)
    assert seen["line_hz"] == 60.0
    check.run_impedance(resolved, line_hz=50.0)
    assert seen["line_hz"] == 50.0
