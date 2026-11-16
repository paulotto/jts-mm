# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for `jtsmm.emg.check` MCU config construction (no hardware)."""

from __future__ import annotations

from jtsmm.emg.check import _check_config


def test_mute_muscle_powers_down_ch1_to_ch6():
    """DRL isolation session must exclude CH1-6 from LOFF_SENSP/LOFF_SENSN.

    Sent as the wire mask ``pd_channels = 0x3F``: ``active_channels: []`` read
    as "no selection" in ``_emg_to_wire`` and left all six powered.
    """
    from jtsmm.config.mcu_apply import build_mcu_apply_payload

    cfg = _check_config({"mcu": {"emg": {"active_channels": [4]}}}, leadoff_on=True, mute_muscle=True)
    assert cfg["mcu"]["emg"]["pd_channels"] == 0x3F
    assert "active_channels" not in cfg["mcu"]["emg"]
    assert build_mcu_apply_payload(cfg)["emg"]["pd_channels"] == 0x3F


def test_default_session_leaves_muscle_channels_powered():
    cfg = _check_config({}, leadoff_on=True)
    assert "active_channels" not in cfg["mcu"]["emg"]
    assert "pd_channels" not in cfg["mcu"]["emg"]


def test_mute_muscle_still_enables_leadoff_and_bias_path():
    cfg = _check_config({}, leadoff_on=True, mute_muscle=True)
    leadoff = cfg["mcu"]["emg"]["leadoff"]
    assert leadoff["enabled"] is True
    assert leadoff["during_recording"] is True
    assert leadoff["ac_tone"] is False
    assert "probe_drl" not in leadoff
    # mute_muscle alone must NOT touch bias -- only `probe_drl` does.
    assert "bias" not in cfg["mcu"]["emg"]


def test_probe_drl_disables_bias_and_forces_ch8_normal():
    """Probing DRL/IN8N requires disabling bias drive so CH8SET drops out of
    BIAS_DRN mode (see ads_write_spare_channels() in ads1299.c) -- otherwise
    the ADS1299's own bias amplifier actively drives IN8N and the lead-off
    comparator/ADC column just see its output, not the real electrode.
    """
    cfg = _check_config({}, leadoff_on=True, mute_muscle=True, probe_drl=True)
    emg = cfg["mcu"]["emg"]
    assert emg["bias"]["enabled"] is False
    assert emg["leadoff"]["probe_drl"] is True
    assert emg["pd_channels"] == 0x3F
