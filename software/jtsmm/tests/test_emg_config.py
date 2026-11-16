# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG profiles, gains, leadoff policy."""

from __future__ import annotations

import copy

import pytest

from jtsmm.config import (
    apply_emg_overrides,
    build_mcu_apply_payload,
    resolve_profiles,
    validate_config,
)
from jtsmm.config.catalog import (
    EMG_GAINS,
    EMG_SAMPLE_RATES_SPS,
    channels_to_mask,
    mask_to_channels,
    parse_channel_spec,
)


def _base():
    from tests.helpers import load_operator_template

    return load_operator_template()


def test_acquisition_profile_defaults():
    doc = resolve_profiles(_base())
    emg = doc["mcu"]["emg"]
    assert emg["sample_rate_sps"] == 4000
    # DC comparator lead-off is on by default everywhere except
    # noise_characterization; the AC tone (the one that could contaminate the
    # signal) stays off regardless.
    assert emg["leadoff"]["during_recording"] is True
    assert emg["leadoff"]["ac_tone"] is False


def test_noise_characterization_profile_leadoff_off():
    doc = copy.deepcopy(_base())
    doc["mcu"]["emg"] = {"profile": "noise_characterization"}
    emg = resolve_profiles(doc)["mcu"]["emg"]
    assert emg["sample_rate_sps"] == 4000  # the default; 16 kSPS is rejected
    assert emg["leadoff"]["during_recording"] is False


def test_high_rate_profile():
    doc = copy.deepcopy(_base())
    doc["mcu"]["emg"]["profile"] = "high_rate"
    emg = resolve_profiles(doc)["mcu"]["emg"]
    assert emg["sample_rate_sps"] == 4000


def test_bench_profile_leadoff():
    doc = copy.deepcopy(_base())
    doc["mcu"]["emg"] = {"profile": "bench"}
    emg = resolve_profiles(doc)["mcu"]["emg"]
    assert emg["sample_rate_sps"] == 250
    assert emg["leadoff"]["during_recording"] is True


def test_invalid_gain_rejected():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["default_gain"] = 99
    errors = validate_config(doc)
    assert any("default_gain" in e for e in errors)


def test_catalog_rates_and_gains():
    assert 1000 in EMG_SAMPLE_RATES_SPS
    assert 6 in EMG_GAINS


# --- Channel / DRL selection (bringup-style args) -------------------------------


def test_parse_channel_spec_forms():
    assert parse_channel_spec("1,4") == [1, 4]
    assert parse_channel_spec("1-3") == [1, 2, 3]
    assert parse_channel_spec("3-1") == [1, 2, 3]
    assert parse_channel_spec([4, 4, 1]) == [1, 4]
    assert parse_channel_spec(None) == []


def test_parse_channel_spec_rejects_out_of_range():
    for bad in ("0", "7", "1,9"):
        with pytest.raises(ValueError):
            parse_channel_spec(bad)


def test_channels_to_mask_roundtrip():
    assert channels_to_mask([1, 4]) == 0b1001
    assert channels_to_mask("1-6") == 0x3F
    assert mask_to_channels(0b1001) == [1, 4]


def test_wire_active_channels_powers_down_and_derives_bias():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["active_channels"] = [4]
    doc["mcu"]["emg"]["bias"] = {"enabled": True}
    emg = build_mcu_apply_payload(doc)["emg"]
    assert "active_channels" not in emg  # translated to wire fields
    assert emg["pd_channels"] == 0x3F & ~(1 << 3)  # CH4 stays powered
    assert emg["bias"]["channels"] == (1 << 3)  # DRL derived from active set


def test_wire_explicit_bias_channels_override():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["active_channels"] = [1, 2, 4]
    doc["mcu"]["emg"]["bias"] = {"enabled": True, "channels": [1, 2]}
    emg = build_mcu_apply_payload(doc)["emg"]
    assert emg["bias"]["channels"] == 0b0011
    assert emg["pd_channels"] == 0x3F & ~0b1011  # CH1,2,4 powered


def test_wire_default_no_channel_fields():
    doc = resolve_profiles(_base())
    emg = build_mcu_apply_payload(doc)["emg"]
    assert "pd_channels" not in emg  # firmware default = all active
    assert "channels" not in emg.get("bias", {})  # firmware default 0x3F


def test_apply_emg_overrides_patches_resolved():
    doc = resolve_profiles(_base())
    apply_emg_overrides(doc, bias_drive=True, bias_channels="4", active_channels="4")
    emg = doc["mcu"]["emg"]
    assert emg["bias"]["enabled"] is True
    assert emg["bias"]["channels"] == [4]
    assert emg["active_channels"] == [4]


def test_apply_emg_overrides_gain():
    doc = resolve_profiles(_base())
    apply_emg_overrides(doc, default_gain=12)
    assert doc["mcu"]["emg"]["default_gain"] == 12


def test_apply_emg_overrides_gain_rejects_invalid():
    doc = resolve_profiles(_base())
    with pytest.raises(ValueError):
        apply_emg_overrides(doc, default_gain=99)


def test_apply_emg_overrides_no_leadoff_forces_off():
    doc = resolve_profiles(_base())
    apply_emg_overrides(doc, leadoff=False)
    assert doc["mcu"]["emg"]["leadoff"]["during_recording"] is False


def test_apply_emg_overrides_leadoff_forces_on_even_for_noise_profile():
    doc = copy.deepcopy(_base())
    doc["mcu"]["emg"]["profile"] = "noise_characterization"
    doc = resolve_profiles(doc)
    apply_emg_overrides(doc, leadoff=True)
    leadoff = doc["mcu"]["emg"]["leadoff"]
    assert leadoff["during_recording"] is True
    assert leadoff["enabled"] is True


def test_apply_emg_overrides_monitoring_forces_on_without_explicit_flag():
    doc = copy.deepcopy(_base())
    doc["mcu"]["emg"]["profile"] = "noise_characterization"
    doc = resolve_profiles(doc)
    apply_emg_overrides(doc, monitoring=True)
    assert doc["mcu"]["emg"]["leadoff"]["during_recording"] is True


def test_apply_emg_overrides_explicit_no_leadoff_wins_over_monitoring():
    doc = resolve_profiles(_base())
    apply_emg_overrides(doc, leadoff=False, monitoring=True)
    assert doc["mcu"]["emg"]["leadoff"]["during_recording"] is False


def test_apply_emg_overrides_bias_channels_requires_bias():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["bias"] = {"enabled": False}
    with pytest.raises(ValueError):
        apply_emg_overrides(doc, bias_channels="4")


def test_validate_rejects_bias_outside_active():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["bias"] = {"enabled": True, "channels": [5]}
    doc["mcu"]["emg"]["active_channels"] = [4]
    errors = validate_config(doc)
    assert any("not in active_channels" in e for e in errors)
    assert not any(e.startswith("warn:") and "active_channels" in e for e in errors)


def test_wire_strips_bias_channels_outside_active():
    """Even if a stale config lists powered-down channels for DRL, wire translation
    must intersect with the active set before the MCU ever sees it."""
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["active_channels"] = [1, 2, 4]
    doc["mcu"]["emg"]["bias"] = {"enabled": True, "channels": [1, 2, 3]}
    emg = build_mcu_apply_payload(doc)["emg"]
    assert emg["bias"]["channels"] == 0b0011  # CH3 powered down → excluded


def test_apply_emg_overrides_rejects_bias_outside_active():
    doc = resolve_profiles(_base())
    with pytest.raises(ValueError, match="not in active_channels"):
        apply_emg_overrides(doc, bias_drive=True, bias_channels="5", active_channels="4")


def test_validate_rejects_bad_channel():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["active_channels"] = [9]
    errors = validate_config(doc)
    assert any("active_channels" in e for e in errors)


def test_applied_config_hash_matches_integrity_reconstruction():
    """The persisted hash must equal what the integrity checker recomputes from
    the stored wire ``mcu_config`` + host bundle (regression: channel overrides)."""
    from jtsmm.config import applied_config_hash, config_hash, split_host_mcu

    doc = resolve_profiles(_base())
    apply_emg_overrides(doc, bias_drive=True, active_channels="4")
    mcu_cfg, host_bundle = split_host_mcu(doc)
    reconstructed = {
        "schema_version": mcu_cfg["schema_version"],
        "hat": host_bundle.get("hat", {}),
        "mcu": {k: mcu_cfg[k] for k in ("platform", "emg", "imu_mta", "imu_cra", "trigger")},
        "host": host_bundle.get("host", {}),
        "session": host_bundle.get("session", {}),
    }
    assert applied_config_hash(doc) == config_hash(reconstructed)
