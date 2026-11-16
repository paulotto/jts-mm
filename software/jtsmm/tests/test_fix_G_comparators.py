# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Lead-off comparators: robust whether the firmware powers them or not (K8)."""

from __future__ import annotations

import numpy as np

import jtsmm.emg.check as check_mod
from jtsmm.emg.capture import (
    comparator_state,
    leadoff_summary,
    parse_config4,
    query_comparators_enabled,
)
from jtsmm.viewer.hub import LEADOFF_BAD_FRAMES, Hub

OK = bytes([0xC0, 0x00, 0x00])
CH1_P = bytes([0xC0, 0x10, 0x00])          # LOFF_STATP bit 0
CH8_N = bytes([0xC0, 0x08, 0x00])          # LOFF_STATN bit 7 (masked)
CH7_P = bytes([0xC8, 0x00, 0x00])          # LOFF_STATP bit 6 (masked)

DUMP = (
    b"ads_status:\r\n  CONFIG1=0x96 CONFIG2=0xC0 CONFIG3=0xEC (want 0xEC, bias_drive=1)\r\n"
    b"  CH1SET=0x60 (gain=24)\r\n"
    b"  CONFIG4=0x%02X CH8SET=0x67 LOFF=0x00 SENSP=0x3F SENSN=0x3F STATP=0x00 STATN=0x00\r\n"
)


class _Ser:
    def __init__(self, reply: bytes = b"", fail: bool = False):
        self.reply = reply
        self.fail = fail
        self.sent = b""

    def reset_input_buffer(self):
        pass

    def write(self, data):
        if self.fail:
            raise OSError("port gone")
        self.sent += data

    def flush(self):
        pass

    def read(self, n):
        out, self.reply = self.reply[:n], self.reply[n:]
        return out


def test_config4_readback_decides_comparator_state():
    assert parse_config4(DUMP % 0x02) == 0x02
    assert query_comparators_enabled(_Ser(DUMP % 0x02)) is True
    assert query_comparators_enabled(_Ser(DUMP % 0x00)) is False   # old firmware
    assert query_comparators_enabled(_Ser(b"ads_status busy\r\n"), timeout_s=0.05) is None
    assert query_comparators_enabled(_Ser(fail=True)) is None
    ser = _Ser(DUMP % 0x02)
    query_comparators_enabled(ser)
    assert ser.sent == b"ads_status\r\n"


def test_all_zero_status_is_unknown_not_connected():
    quiet = leadoff_summary([OK] * 20)
    assert comparator_state(None, quiet) == "unknown"
    assert comparator_state(False, quiet) == "off"
    assert comparator_state(True, quiet) == "on"
    # A tripped bit proves the comparators run, even without a readback.
    assert comparator_state(None, leadoff_summary([CH1_P] * 20)) == "on"


def test_ch7_ch8_bits_are_masked():
    s = leadoff_summary([CH7_P, CH8_N] * 10, n_channels=8)
    assert len(s["frac_off"]) == 6
    assert comparator_state(None, s) == "unknown"


def _sessions(status, comparators_on):
    emg = {"default_gain": 6, "sample_rate_sps": 2000.0}

    def _session(*_a, leadoff_on=False, **_k):
        s = np.zeros((2000, 8), dtype=np.float32)
        t = (np.arange(2000) * 500).astype(np.int64)
        return check_mod._Capture(
            s, status if leadoff_on else [], emg, t, comparators_on if leadoff_on else None
        )

    return _session


def test_comparator_hit_warns_but_does_not_hard_fail(monkeypatch):
    monkeypatch.setattr(check_mod, "_hardware_session", _sessions([CH1_P] * 20, True))
    res = check_mod.run_emg_check({"mcu": {"emg": {"bias": {"enabled": False}}}},
                                  port="/dev/null", with_impedance=False)
    ch1 = res.channels[0]
    assert ch1.comparator == "hit"
    assert any("WARN comparator" in n for n in ch1.notes)
    assert ch1.contact_hard_fail is False
    assert ch1.leadoff_ok is True
    assert res.channels[1].comparator == "clear"
    assert check_mod.hard_contact_failures(res) == []


def test_comparators_off_in_firmware_are_reported_not_trusted(monkeypatch):
    # Bits set, but CONFIG4 says the comparators are powered down: ignore them.
    monkeypatch.setattr(check_mod, "_hardware_session", _sessions([CH1_P] * 20, False))
    res = check_mod.run_emg_check({"mcu": {"emg": {"bias": {"enabled": False}}}},
                                  port="/dev/null", with_impedance=False)
    assert res.channels[0].comparator == "off"
    assert not any("comparator lead-off" in n for n in res.channels[0].notes)
    assert "powered down in firmware" in res.note


def test_unknown_comparators_leave_a_note(monkeypatch):
    monkeypatch.setattr(check_mod, "_hardware_session", _sessions([OK] * 20, None))
    res = check_mod.run_emg_check({"mcu": {"emg": {"bias": {"enabled": False}}}},
                                  port="/dev/null", with_impedance=False)
    assert all(c.comparator == "unknown" for c in res.channels)
    assert "state unknown" in res.note


def _hub(**kw):
    return Hub(emg_fs=500.0, emg_lsb_uv=0.1, n_channels=7, display_fs=500.0,
               filter_enabled=False, leadoff_during_recording=True, **kw)


def test_hub_all_zero_status_is_unknown():
    hub = _hub()
    for _ in range(20):
        hub.set_leadoff(OK)
    snap = hub.snapshot()
    assert snap.leadoff_cmp == "unknown"
    assert not any(snap.leadoff_p)


def test_hub_latches_a_hit_once_comparators_are_seen_running():
    hub = _hub()
    for _ in range(LEADOFF_BAD_FRAMES):
        hub.set_leadoff(CH1_P)
    snap = hub.snapshot()
    assert snap.leadoff_cmp == "on"
    assert snap.leadoff_p[0] is True


def test_hub_ignores_bits_when_config_says_comparators_off():
    hub = _hub(comparators_enabled=False)
    for _ in range(2 * LEADOFF_BAD_FRAMES):
        hub.set_leadoff(CH1_P)
    snap = hub.snapshot()
    assert snap.leadoff_cmp == "off"
    assert snap.leadoff_p[0] is False


def test_hub_never_reports_drl_from_the_masked_ch8_bit():
    hub = _hub(comparators_enabled=True)
    for _ in range(2 * LEADOFF_BAD_FRAMES):
        hub.set_leadoff(CH8_N)
    snap = hub.snapshot()
    assert snap.drl_leadoff is False
    assert snap.to_map()["leadoff"]["cmp"] == "on"
