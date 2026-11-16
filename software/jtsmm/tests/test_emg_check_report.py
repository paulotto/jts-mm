# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Unit tests for `jtsmm.emg.check` report structures (no hardware)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jtsmm.emg.check import (
    ChannelCheck,
    CheckResult,
    check_result_to_dict,
    format_check_table,
    run_emg_check,
)


def _muscle_channel(channel: int = 1, ok: bool = True) -> ChannelCheck:
    return ChannelCheck(
        channel=channel,
        name=f"ch{channel}",
        leadoff_ok=ok,
        noise_rms_uv=5.0,
        noise_ok=ok,
        line_peak_uv=10.0,
        line_ok=ok,
    )


def test_check_result_to_dict_round_trips_through_json():
    drl = ChannelCheck(
        channel=8,
        name="drl",
        leadoff_ok=False,
        noise_rms_uv=None,
        noise_ok=True,
        line_peak_uv=None,
        line_ok=True,
        notes=["floating input (2510 µV RMS > 500)"],
    )
    result = CheckResult([_muscle_channel(1, ok=True)], False, drl=drl, note="best-effort")

    payload = check_result_to_dict(result)
    # Must be JSON-serializable (this is exactly what `record --emg-check`
    # writes into /meta/emg_check_json).
    text = json.dumps(payload)
    back = json.loads(text)

    assert back["passed"] is False
    assert back["note"] == "best-effort"
    assert back["channels"][0]["channel"] == 1
    assert back["drl"]["channel"] == 8
    assert back["drl"]["leadoff_ok"] is False


def test_check_result_to_dict_handles_missing_drl():
    result = CheckResult([_muscle_channel()], True, drl=None)
    payload = check_result_to_dict(result)
    assert payload["drl"] is None
    json.dumps(payload)  # must not raise


def test_run_emg_check_runtime_error_sets_note_not_drl(monkeypatch):
    """Regression test: the early-exit path used to pass the error message
    positionally, landing in `drl` (a ChannelCheck | None field) instead of
    `note` -- which crashes `format_check_table`/`failed_channels` the
    moment they touch `.leadoff_ok` on a plain string.

    `_hardware_session` is monkeypatched (rather than pointing at a bogus
    port) so this stays a pure unit test even when real jtsmm_hat hardware
    is attached to the test machine -- `find_hat_port` falls back to
    auto-detection and would otherwise happily talk to it.
    """
    import jtsmm.emg.check as check_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("No jtsmm_hat found on serial ports.")

    monkeypatch.setattr(check_mod, "_hardware_session", _boom)

    result = run_emg_check({"mcu": {"platform": {}}}, port="/dev/does-not-exist")
    assert result.passed is False
    assert result.drl is None
    assert isinstance(result.note, str) and result.note

    # Must not raise despite drl being None.
    table = format_check_table(result)
    assert "overall: FAIL" in table
    assert result.failed_channels == []


def test_run_emg_check_drl_rail_frac_uses_session_lsb(monkeypatch):
    """Regression: bias-amp scoring used a leftover `lsb` name after the
    noise capture started returning µV. That crashed `jtsmm record --mvc
    --force-emg-check` with NameError before any verdict was printed.
    """
    import numpy as np

    import jtsmm.emg.check as check_mod
    from jtsmm.config.catalog import channel_lsb_uv

    gain = 6
    lsb = channel_lsb_uv(gain)[7]   # CH8 runs at gain 24 whatever the muscle gain
    n, nch, fs = 2000, 8, 2000.0
    samples = np.zeros((n, nch), dtype=np.float32)
    # Bias-ON CH8 readback pinned at full scale (the MVC session).
    samples[:, 7] = np.float32((1 << 23) * 0.99 * lsb)
    emg = {"default_gain": gain, "sample_rate_sps": fs}

    def _fake_session(*_args, **_kwargs):
        return check_mod._Capture(samples, [], emg)

    monkeypatch.setattr(check_mod, "_hardware_session", _fake_session)

    result = run_emg_check(
        {"mcu": {"emg": {"bias": {"enabled": True}}}},
        port="/dev/null",
        with_impedance=False,
    )
    assert result.drl_saturation is not None
    assert result.drl_saturation.ac_mv < 0.1
    assert result.drl_saturation.rail_frac > 0.9
    # A railed column has no AC by construction: n/a, not PASS -- and the
    # readback is informational, so it never fails the check either way.
    assert result.drl_saturation.status == "n/a"
    assert result.drl_saturation.ok is True


def test_format_check_table_drl_row_uses_na_for_unmeasurable_columns():
    drl = ChannelCheck(
        channel=8,
        name="drl",
        leadoff_ok=True,
        noise_rms_uv=None,
        noise_ok=True,
        line_peak_uv=None,
        line_ok=True,
    )
    result = CheckResult([_muscle_channel()], True, drl=drl)
    table = format_check_table(result)
    drl_line = next(line for line in table.splitlines() if line.strip().startswith("DRL"))
    assert drl_line.count("n/a") == 3
    assert "PASS" in drl_line


def test_integrity_summary_names_failing_channels_not_the_drl_note():
    """Regression: the summary used to print the generic DRL caveat as the error.

    Mirrors the rig run with no electrodes attached: six muscle channels open
    (lead-off + line) and a floating DRL.
    """
    from jtsmm.emg.check import ChannelCheck, CheckResult, check_result_to_dict
    from jtsmm.integrity.summary import SessionIntegritySummary, set_emg_check

    names = [
        "masseter_l", "masseter_r", "temporalis_l",
        "temporalis_r", "digastric_l", "digastric_r",
    ]
    channels = [
        ChannelCheck(
            channel=i + 1, name=n, leadoff_ok=False, noise_rms_uv=0.0,
            noise_ok=True, line_peak_uv=670.5, line_ok=False,
            notes=["open input, DC lead-off tripped", "50 Hz 670.5 uV > 50"],
        )
        for i, n in enumerate(names)
    ]
    drl = ChannelCheck(
        channel=8, name="drl", leadoff_ok=False, noise_rms_uv=None,
        noise_ok=True, line_peak_uv=None, line_ok=True,
        notes=["floating input (909 uV RMS)"],
    )
    note = "DRL leadoff is best-effort: IN8P is unwired on this board, so neither ..."
    result = CheckResult(channels=channels, passed=False, drl=drl, note=note)

    summary = SessionIntegritySummary(session_dir=Path("."))
    set_emg_check(summary, result=check_result_to_dict(result), passed=False)

    assert summary.emg.ok is False
    errors = summary.emg.errors
    text = "\n".join(errors)
    # One entry per failing channel, carrying that channel's own notes.
    assert len(errors) == 7, errors
    assert "CH1 (masseter_l): open input, DC lead-off tripped" in text, text
    assert "DRL: floating input (909 uV RMS)" in text, text
    assert "IN8P is unwired" not in text, "the generic note must not be the error"


def test_integrity_summary_passing_emg_check_has_no_error():
    from jtsmm.emg.check import ChannelCheck, CheckResult, check_result_to_dict
    from jtsmm.integrity.summary import SessionIntegritySummary, set_emg_check

    ch = ChannelCheck(
        channel=1, name="masseter_l", leadoff_ok=True, noise_rms_uv=8.0,
        noise_ok=True, line_peak_uv=12.0, line_ok=True,
    )
    result = CheckResult(channels=[ch], passed=True, drl=None, note="note text")
    summary = SessionIntegritySummary(session_dir=Path("."))
    set_emg_check(summary, result=check_result_to_dict(result), passed=True)
    assert summary.emg.ok is True


# --- DRL / bias-amplifier saturation in `jtsmm emg check` ---------------------


def _sat(ac_mv: float, rail_frac: float = 0.0):
    from jtsmm.emg.capture import DRL_AC_ERR_MV, DRL_AC_WARN_MV
    from jtsmm.emg.check import DrlSaturation

    notes = [] if ac_mv < DRL_AC_WARN_MV else [f"bias node swinging {ac_mv:.0f} mV RMS"]
    return DrlSaturation(
        ac_mv=ac_mv, ok=ac_mv < DRL_AC_ERR_MV, rail_frac=rail_frac, notes=notes
    )


def test_drl_ac_separates_the_two_recorded_sessions():
    """Calibrated on one subject, two takes minutes apart, same electrodes.

    MVC (excellent):  CH8 pinned at full scale, ~0 mV AC, rail fraction 1.000
    take1 (unusable): CH8 swinging ~219 mV RMS,           rail fraction 0.042

    Rail occupancy ranks these backwards, so the verdict is on AC content.
    """
    import numpy as np

    from jtsmm.emg.capture import drl_ac_mv, drl_rail_fraction

    lsb = 0.089407
    full = 1 << 23
    t = np.arange(int(20 * 2000)) / 2000.0

    pinned = np.full(t.size, int(full * 0.99), dtype=np.float64)
    assert drl_ac_mv(pinned, lsb) < 0.1
    assert drl_rail_fraction(pinned) > 0.9          # railed, yet healthy

    swinging = (219_000.0 * np.sqrt(2.0) / lsb) * np.sin(2 * np.pi * 49.98 * t)
    assert drl_ac_mv(swinging, lsb) == pytest.approx(219.0, rel=0.05)
    assert drl_rail_fraction(swinging) < 0.2        # not railed, yet faulty

    assert drl_ac_mv(np.empty(0), lsb) == 0.0
    assert drl_ac_mv(pinned, 0.0) == 0.0


def test_saturated_bias_amp_fails_the_check_table():
    from jtsmm.emg.check import ChannelCheck, CheckResult, format_check_table

    ch = ChannelCheck(
        channel=1, name="masseter_l", leadoff_ok=True, noise_rms_uv=8.0,
        noise_ok=True, line_peak_uv=12.0, line_ok=True,
    )
    # Every electrode row passes; only the bias amp is saturated.
    result = CheckResult([ch], passed=False, drl=None, note="", drl_saturation=_sat(219.0))
    table = format_check_table(result)
    assert "bias amp (DRL)" in table
    assert "219" in table
    assert "overall: FAIL" in table


def test_healthy_bias_amp_row_passes():
    from jtsmm.emg.check import ChannelCheck, CheckResult, format_check_table

    ch = ChannelCheck(
        channel=1, name="masseter_l", leadoff_ok=True, noise_rms_uv=8.0,
        noise_ok=True, line_peak_uv=12.0, line_ok=True,
    )
    result = CheckResult([ch], passed=True, drl=None, note="", drl_saturation=_sat(0.0, rail_frac=1.0))
    table = format_check_table(result)
    assert "bias amp (DRL)" in table and "overall: PASS" in table
    assert "FAIL" not in table


def test_saturation_reaches_the_integrity_summary():
    from jtsmm.emg.check import ChannelCheck, CheckResult, check_result_to_dict
    from jtsmm.integrity.summary import SessionIntegritySummary, set_emg_check

    ch = ChannelCheck(
        channel=1, name="masseter_l", leadoff_ok=True, noise_rms_uv=8.0,
        noise_ok=True, line_peak_uv=12.0, line_ok=True,
    )
    result = CheckResult([ch], passed=False, drl=None, note="caveat", drl_saturation=_sat(219.0))
    d = check_result_to_dict(result)
    assert d["drl_saturation"]["ac_mv"] == pytest.approx(219.0)

    summary = SessionIntegritySummary(session_dir=Path("."))
    set_emg_check(summary, result=d, passed=False)
    text = "\n".join(summary.emg.errors)
    assert "bias amp (DRL)" in text, text
    assert "caveat" not in text


def test_saturation_is_skipped_when_bias_is_disabled():
    """With bias.enabled=false there is no DRL loop to judge."""
    from jtsmm.emg.check import ChannelCheck, CheckResult, check_result_to_dict

    ch = ChannelCheck(
        channel=1, name="masseter_l", leadoff_ok=True, noise_rms_uv=8.0,
        noise_ok=True, line_peak_uv=12.0, line_ok=True,
    )
    d = check_result_to_dict(CheckResult([ch], passed=True, drl=None, drl_saturation=None))
    assert d["drl_saturation"] is None


# --- hard contact failure vs. environmental warning --------------------------
#
# Calibrated on s01_f_30, which recorded 467 s with `emg_check_passed: false`
# because the old code warned and continued. The real AC |Z| in that session was
# 23-63 kOhm (four channels at or over the 50 kOhm limit) while noise and line
# hum were 4-5 orders of magnitude over their limits. Only the |Z| half is
# fixable at the rig, so only that half may abort.


def _ch(channel: int, *, name: str, hard: bool = False, noisy: bool = False) -> ChannelCheck:
    from jtsmm.emg.check import ChannelCheck as _CC

    return _CC(
        channel=channel,
        name=name,
        leadoff_ok=not hard,
        noise_rms_uv=400_000.0 if noisy else 5.0,
        noise_ok=not noisy,
        line_peak_uv=8_600.0 if noisy else 10.0,
        line_ok=not noisy,
        z_kohm=60.8 if hard else 24.8,
        z_quality="poor" if hard else "good",
        contact_hard_fail=hard,
    )


def test_noise_only_failure_is_not_a_hard_contact_failure():
    """The stranding case: mains hum must never block a subject session."""
    from jtsmm.emg.check import hard_contact_failures

    res = CheckResult([_ch(1, name="masseter_r", noisy=True)], False)
    assert res.passed is False
    assert hard_contact_failures(res) == []


def test_bad_contact_is_reported_by_name():
    from jtsmm.emg.check import hard_contact_failures

    res = CheckResult(
        [_ch(1, name="masseter_r"), _ch(2, name="temporalis_r", hard=True, noisy=True)],
        False,
    )
    assert hard_contact_failures(res) == ["temporalis_r"]


def test_all_good_has_no_hard_failures():
    from jtsmm.emg.check import hard_contact_failures

    res = CheckResult([_ch(1, name="masseter_r"), _ch(2, name="temporalis_r")], True)
    assert hard_contact_failures(res) == []


def test_hard_flag_survives_the_json_snapshot():
    """Regression: `ok` is a @property and asdict() drops it, which is how an
    earlier failure-reason loop silently did nothing. contact_hard_fail is a
    real field precisely so it reaches /meta/emg_check_json."""
    res = CheckResult([_ch(2, name="temporalis_r", hard=True)], False)
    doc = json.loads(json.dumps(check_result_to_dict(res)))
    assert doc["channels"][0]["contact_hard_fail"] is True


def test_block_on_contact_failure_defaults_on_and_is_overridable():
    from jtsmm.config.registry import record_emg_check_block_on_contact

    assert record_emg_check_block_on_contact({}) is True
    assert record_emg_check_block_on_contact(
        {"host": {"record": {"emg_check": {"block_on_contact_failure": False}}}}
    ) is False


# --- band RMS calibration ----------------------------------------------------
#
# _bandpass_rms returned RMS * sqrt(n) until 2026-09 (Parseval needs |X|^2/n^2,
# not /n, and it zero-padded to a power of two on top). At a 5 s / 2 kHz capture
# that is a factor of 128: the 10 uV noise limit became an unreachable 0.08 uV,
# every channel of every session failed the noise check, and the readings were
# six-figure microvolts that looked like a railed front end. s01_f_30 reported
# "noise 232259.6 uV" for a channel whose true in-band RMS was ~1.8 mV.


def _rms_tone(n_samples: int, rms_uv: float = 100.0, fs: float = 2000.0):
    import numpy as np

    t = np.arange(n_samples) / fs
    return rms_uv * np.sqrt(2) * np.sin(2 * np.pi * 100.0 * t)


def test_bandpass_rms_returns_actual_microvolts():
    from jtsmm.emg.check import _bandpass_rms

    for n in (2048, 2000, 10000):
        got = _bandpass_rms(_rms_tone(n), 2000.0, 20.0, 450.0)
        assert got == pytest.approx(100.0, rel=0.01), (n, got)


def test_bandpass_rms_does_not_scale_with_capture_length():
    """The actual bug: the reading grew as sqrt(n), so the threshold was moot."""
    from jtsmm.emg.check import _bandpass_rms

    short = _bandpass_rms(_rms_tone(2048), 2000.0, 20.0, 450.0)
    long = _bandpass_rms(_rms_tone(32768), 2000.0, 20.0, 450.0)
    assert long == pytest.approx(short, rel=0.01)


def test_bandpass_rms_rejects_out_of_band_energy():
    import numpy as np

    from jtsmm.emg.check import _bandpass_rms

    t = np.arange(20000) / 2000.0
    assert _bandpass_rms(100 * np.sin(2 * np.pi * 5 * t), 2000.0, 20.0, 450.0) < 1.0
    assert _bandpass_rms(100 * np.sin(2 * np.pi * 800 * t), 2000.0, 20.0, 450.0) < 1.0


def test_white_noise_band_fraction_is_right():
    """20-450 Hz of a 1 kHz Nyquist span is sqrt(430/1000) of the total."""
    import numpy as np

    from jtsmm.emg.check import _bandpass_rms

    x = np.random.default_rng(0).standard_normal(40000) * 12.0
    assert _bandpass_rms(x, 2000.0, 20.0, 450.0) == pytest.approx(
        12.0 * np.sqrt(430 / 1000), rel=0.05
    )


# --- mains-aware noise -------------------------------------------------------


def test_noise_measure_separates_the_electrode_from_the_room():
    """A quiet electrode in a hummy room must not read as a noisy electrode.

    This is what sent the operator to re-prep a 24.8 kOhm contact: the plain
    band RMS could not tell 8 uV of electrode noise under 3 mV of hum from 3 mV
    of electrode noise. `line_peak_uv` keeps the verdict on the mains itself.
    """
    import numpy as np

    from jtsmm.emg.check import _bandpass_rms, _bandpass_rms_no_mains

    fs = 2000.0
    t = np.arange(int(10 * fs)) / fs
    quiet = np.random.default_rng(1).standard_normal(t.size) * 8.0
    hum = sum((3000.0 / k) * np.sqrt(2) * np.sin(2 * np.pi * k * 50.0 * t) for k in (1, 2, 3))

    assert _bandpass_rms(quiet + hum, fs, 20.0, 450.0) > 1000.0        # room
    clean = _bandpass_rms_no_mains(quiet + hum, fs, 20.0, 450.0, 50.0)
    assert clean == pytest.approx(
        _bandpass_rms(quiet, fs, 20.0, 450.0), rel=0.15
    )                                                                  # electrode


def test_clean_noise_is_reported_alongside_the_total():
    from jtsmm.emg.check import ChannelCheck, check_result_to_dict, CheckResult

    c = ChannelCheck(1, "masseter_r", True, 9.0, True, 40.0, True, noise_total_uv=1800.0)
    doc = check_result_to_dict(CheckResult([c], True))
    assert doc["channels"][0]["noise_rms_uv"] == 9.0
    assert doc["channels"][0]["noise_total_uv"] == 1800.0
