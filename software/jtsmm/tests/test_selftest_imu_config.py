# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Preflight grades the selftest's IMU probe against the session config.

The firmware probes both slots whatever the config says; with no IMUs plugged in
it reports `[FAIL] IMU probe`, which used to block `jtsmm record` even with both
IMUs disabled. Transcript below: the HAT with no IMUs (2026-09-25).
"""

from __future__ import annotations

import pytest

from jtsmm.doctor.mcu import _grade_selftest

NO_IMUS = """=== jtsmm selftest ===
[PASS] LED blink
[PASS] ADS1299 SPI/CHIPID
  IMU probe MTA=0 CRA=0
[FAIL] IMU probe (no ACK on I2C)
[PASS] trigger pulse
  buttons USER1=1 USER2=1 (0=pressed)
[PASS] buttons readback
=== result: 4/5 ===
"""
MTA_ONLY = NO_IMUS.replace("MTA=0 CRA=0\n[FAIL] IMU probe (no ACK on I2C)",
                           "MTA=1 CRA=0\n[SKIP] CRA — no BNO085\n[PASS] IMU probe")


def test_disabled_imus_are_not_required():
    assert _grade_selftest(NO_IMUS, {"mta": False, "cra": False}) == []


@pytest.mark.parametrize("text,enabled,missing", [
    (NO_IMUS, {"mta": True, "cra": True}, ["MTA", "CRA"]),
    (NO_IMUS, {"mta": False, "cra": True}, ["CRA"]),
    (MTA_ONLY, {"mta": True, "cra": True}, ["CRA"]),
])
def test_enabled_imu_that_does_not_answer_is_an_error(text, enabled, missing):
    issues = _grade_selftest(text, enabled)
    assert [i.split()[2] for i in issues] == missing
    assert all(i.startswith("error:") and "mcu.imu_" in i for i in issues)


def test_mta_only_rig_with_cra_disabled_passes():
    assert _grade_selftest(MTA_ONLY, {"mta": True, "cra": False}) == []


def test_other_failures_still_reported():
    text = NO_IMUS.replace("[PASS] trigger pulse", "[FAIL] trigger pulse")
    issues = _grade_selftest(text, {"mta": False, "cra": False})
    assert issues == ["error: MCU selftest reported failure", "error: [FAIL] trigger pulse"]


def test_without_config_or_probe_line_the_firmware_verdict_stands():
    for text, cfg in ((NO_IMUS, None), (NO_IMUS.replace("  IMU probe MTA=0 CRA=0\n", ""),
                                       {"mta": False, "cra": False})):
        assert _grade_selftest(text, cfg) == [
            "error: MCU selftest reported failure", "error: [FAIL] IMU probe (no ACK on I2C)"]
