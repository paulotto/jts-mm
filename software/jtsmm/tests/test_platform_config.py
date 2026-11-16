# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Platform profiles, SRAM budget, SPI validation."""

from __future__ import annotations

import copy

from jtsmm.config import resolve_profiles, validate_config


def _base():
    from tests.helpers import load_operator_template

    return load_operator_template()


def test_platform_default_profile():
    plat = resolve_profiles(_base())["mcu"]["platform"]
    assert plat["profile"] == "default"
    assert plat["spi0"]["hz"] == 4_000_000


def test_sram_budget_error():
    doc = resolve_profiles(_base())
    # Byte budgets: 300 KB EMG + 2 x 60 KB IMU = 420 KB > ~400 KB SRAM budget.
    doc["mcu"]["platform"]["buffers"] = {
        "emg_ring": 300_000, "imu_ring": 60_000, "event_ring": 8192
    }
    errors = validate_config(doc)
    assert any("SRAM" in e for e in errors)


def test_emg_2khz_spi_warn():
    doc = resolve_profiles(_base())
    doc["mcu"]["emg"]["sample_rate_sps"] = 2000
    doc["mcu"]["platform"]["spi0"] = {"hz": 500_000}
    errors = validate_config(doc)
    assert any("ISR budget" in e for e in errors)


def test_high_throughput_profile_fragment():
    doc = copy.deepcopy(_base())
    doc["mcu"]["platform"]["profile"] = "high_throughput"
    plat = resolve_profiles(doc)["mcu"]["platform"]
    assert plat["profile"] == "high_throughput"
