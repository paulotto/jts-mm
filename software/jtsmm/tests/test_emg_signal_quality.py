# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mains-domination and DRL-saturation detection on recorded EMG.

Calibrated against session_20260918T081708Z, which the pipeline passed as OK
while being 98% line interference with the bias amplifier railed 45% of the
time. The important half of these tests is the *quiet* direction: a resting
channel with a little hum, and a strong contraction, must not be flagged.
"""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.integrity.emg_signal import (
    DRL_AC_ERR_MV,
    MAINS_BAD_WINDOW_FRAC_ERR,
    analyse_emg_signal,
)

FS = 2000.0
LSB = 0.08940696716308594  # gain 6, as recorded
FULL_SCALE = 1 << 23


def _counts(uv: np.ndarray) -> np.ndarray:
    return np.asarray(uv / LSB, dtype=np.int32)


def _build(
    n_s: float = 24.0,
    *,
    emg_uv: float = 10.0,
    mains_uv: float = 0.0,
    mains_hz: float = 49.98,
    mains_from_s: float | None = None,
    drl: str = "midrange",
    seed: int = 7,
) -> np.ndarray:
    """Synthetic (N, 8) counts: broadband EMG on CH1, optional mains, chosen DRL state."""
    rng = np.random.default_rng(seed)
    n = int(n_s * FS)
    t = np.arange(n) / FS
    out = np.zeros((n, 8), dtype=np.int32)

    # Broadband "EMG": white noise band-limited to 20-450 Hz.
    x = rng.standard_normal(n)
    spec = np.fft.rfft(x)
    fr = np.fft.rfftfreq(n, 1 / FS)
    spec[(fr < 20) | (fr > 450)] = 0
    x = np.fft.irfft(spec, n)
    x = x / (np.std(x) or 1.0) * emg_uv

    if mains_uv > 0:
        hum = mains_uv * np.sqrt(2.0) * np.sin(2 * np.pi * mains_hz * t)
        if mains_from_s is not None:
            hum[t < mains_from_s] = 0.0
        x = x + hum
    out[:, 0] = _counts(x)

    if drl == "swinging":
        # The fault: bias node driven hard by common-mode. Measured 147-225 mV
        # RMS on an unusable take, against 0.0-3.9 mV on an excellent one.
        out[:, 7] = _counts(200_000.0 * np.sqrt(2.0) * np.sin(2 * np.pi * mains_hz * t))
    elif drl == "pinned":
        # The HEALTHY state on this board: parked at full scale, no AC at all,
        # because IN8P is unwired so the absolute level carries no information.
        out[:, 7] = int(FULL_SCALE * 0.99)
    elif drl == "midrange":
        out[:, 7] = _counts(0.2 * FULL_SCALE * LSB + 2_000.0 * np.sin(2 * np.pi * 3.0 * t))
    return out


def _run(samples, **kw):
    kw.setdefault("active_channels", [0])
    return analyse_emg_signal(samples, fs_hz=FS, lsb_uv=LSB, **kw)


# --- the quiet direction (must not fire) -------------------------------------


def test_resting_emg_with_a_little_hum_is_quiet():
    r = _run(_build(emg_uv=10.0, mains_uv=4.0))
    v = r["verdict"]
    assert not any(v.values()), r
    assert r["channels"][0]["bad_window_frac"] == 0.0


def test_strong_contraction_is_quiet():
    """400 uV of real EMG must not be mistaken for interference."""
    r = _run(_build(emg_uv=400.0, mains_uv=20.0))
    assert not any(r["verdict"].values()), r


def test_loud_channel_that_is_not_mains_is_quiet():
    """High amplitude alone is not enough -- the share must be high too."""
    r = _run(_build(emg_uv=3000.0, mains_uv=0.0))
    ch = r["channels"][0]
    assert ch["median_band_rms_uv"] > 1000.0
    assert ch["median_mains_share"] < 0.2
    assert not r["verdict"]["mains_error"] and not r["verdict"]["mains_warn"]


def test_quiet_channel_that_is_mostly_hum_is_not_an_error():
    """A resting channel whose little power happens to be hum is a warning at most."""
    r = _run(_build(emg_uv=2.0, mains_uv=30.0))
    assert not r["verdict"]["mains_error"], r


def test_pinned_drl_is_quiet_even_though_it_is_at_the_rail():
    """Regression for an inverted metric.

    Measured on one subject minutes apart, same electrodes: the EXCELLENT
    recording had CH8 pinned at full scale with ~0 mV AC (rail fraction 1.000),
    the unusable one had it swinging 219 mV with rail fraction 0.042. Judging
    rail occupancy ranked them exactly backwards.
    """
    r = _run(_build(drl="pinned"))
    assert r["drl"]["rail_frac"] > 0.9, r["drl"]
    assert r["drl"]["ac_mv"] < 1.0, r["drl"]
    assert not r["verdict"]["drl_error"] and not r["verdict"]["drl_warn"]


def test_midrange_drl_is_quiet():
    r = _run(_build(drl="midrange"))
    assert r["drl"]["ac_mv"] < DRL_AC_ERR_MV
    assert not r["verdict"]["drl_error"]


# --- the firing direction ----------------------------------------------------


def test_mains_dominated_session_is_an_error():
    """The rig case: ~7.6 mV in-band, ~99% of it at the line frequency."""
    r = _run(_build(emg_uv=10.0, mains_uv=7600.0))
    ch = r["channels"][0]
    assert ch["median_mains_share"] > 0.95, ch
    assert ch["median_band_rms_uv"] > 5000.0, ch
    assert ch["bad_window_frac"] == 1.0
    assert r["verdict"]["mains_error"]


def test_swinging_drl_is_graded_high_but_only_warns():
    """Hundreds of mV of AC on the bias node: graded "high", but informational
    (warn, never error) until the driven loop is bench-verified."""
    r = _run(_build(drl="swinging"))
    assert r["drl"]["ac_mv"] > 150.0, r["drl"]
    assert r["drl"]["rail_frac"] < 0.2, r["drl"]  # not railed, so the figure counts
    assert r["drl"]["status"] == "high"
    assert r["verdict"]["drl_warn"] and not r["verdict"]["drl_error"]


def test_intermittent_mains_reports_the_affected_fraction():
    """'Sometimes perfect, sometimes only noise' must read as a fraction, not a mean."""
    r = _run(_build(n_s=24.0, emg_uv=10.0, mains_uv=7600.0, mains_from_s=12.0))
    frac = r["channels"][0]["bad_window_frac"]
    assert 0.35 <= frac <= 0.65, frac
    assert r["verdict"]["mains_error"]  # >= 20% of the session is still an error


def test_a_small_bad_stretch_warns_but_does_not_fail():
    # 48 s = exactly 24 windows (so they are contiguous, not spread); the last
    # 4 s are bad -> 2/24 = 8%, above the warn threshold and below the error one.
    r = _run(_build(n_s=48.0, emg_uv=10.0, mains_uv=7600.0, mains_from_s=44.0))
    frac = r["channels"][0]["bad_window_frac"]
    assert 0.0 < frac < MAINS_BAD_WINDOW_FRAC_ERR, frac
    assert r["verdict"]["mains_warn"] and not r["verdict"]["mains_error"]


# --- configuration handling --------------------------------------------------


def test_probe_drl_session_skips_the_drl_test():
    """CH8 is a plain electrode input in the isolated probe session, not the bias readback."""
    r = _run(_build(drl="swinging"), drl_meaningful=False)
    assert r["drl"] is None
    assert not r["verdict"]["drl_error"]


def test_line_frequency_is_autodetected():
    assert _run(_build(mains_uv=5000.0, mains_hz=49.98))["line_hz"] == pytest.approx(50.0)
    assert _run(_build(mains_uv=5000.0, mains_hz=60.02))["line_hz"] == pytest.approx(60.0)


def test_explicit_line_frequency_is_honoured():
    r = _run(_build(mains_uv=7600.0, mains_hz=60.0), line_hz=50.0)
    # Told the wrong frequency, the 60 Hz energy is not attributed to mains.
    assert r["line_hz"] == pytest.approx(50.0)
    assert r["channels"][0]["median_mains_share"] < 0.3


def test_powered_down_channels_are_not_judged():
    s = _build(emg_uv=10.0, mains_uv=7600.0)
    r = _run(s, active_channels=[])          # everything powered down
    assert r["channels"] == {}
    assert not r["verdict"]["mains_error"]


def test_short_record_is_skipped_not_guessed():
    r = _run(_build(n_s=0.5))
    assert r["windows"] == 0
    assert not any(r["verdict"].values())


def test_window_count_is_bounded_for_long_sessions():
    r = _run(_build(n_s=200.0, emg_uv=10.0))
    assert r["windows"] <= 24


def test_real_mains_dominated_session_if_present():
    """Regression against a real recording, when one is available locally.

    Drop a session that shows the fault at ``data/session_mains.h5`` (untracked;
    subject data must not enter git). Calibrated on
    session_20260918T081708Z: CH1 7.6 mV RMS in-band, 99.8% of it at 49.98 Hz
    and harmonics, BIAS_DRN railed 45% of the time.
    """
    from pathlib import Path

    from jtsmm.integrity import check_session

    root = Path(__file__).resolve().parents[3]
    session = root / "data" / "session_mains.h5"
    if not session.is_file():
        pytest.skip("data/session_mains.h5 not present")

    report = check_session(session)
    sig = report.summary.get("emg_signal")
    assert sig is not None, "signal-quality check did not run"
    assert sig["drl_ac_mv"] > DRL_AC_ERR_MV, sig  # true CH8 scale (gain 24)
    assert any("DRL readback" in w for w in report.warnings), report.warnings
    assert not any("DRL" in e for e in report.issues), report.issues
    bad = [c for c, v in sig["channels"].items() if v["bad_window_frac"] > 0.5]
    assert bad, sig
    for name in bad:
        assert sig["channels"][name]["mains_share"] > 0.9, sig["channels"][name]
        assert sig["channels"][name]["band_rms_uv"] > 1000.0, sig["channels"][name]
    assert any("mains interference dominates" in e for e in report.issues), report.issues
