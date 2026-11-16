# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Profiles: presets never replace explicit reports; host presets match the firmware."""

from __future__ import annotations

import copy

import pytest

from jtsmm.cli.doctor import _build_config
from jtsmm.config import resolve_profiles, validate_config
from jtsmm.config.catalog import BNO085_REPORT_CATALOG
from jtsmm.config.loader import template_path


def _hard(doc):
    return [e for e in validate_config(doc) if not e.startswith("warn:")]


def test_explicit_reports_survive_resolution(operator_config):
    doc = copy.deepcopy(operator_config)
    doc["mcu"]["imu_mta"]["reports"]["linear_acceleration"]["rate_hz"] = 200
    doc["mcu"]["imu_mta"]["reports"]["gyroscope"]["enabled"] = True
    out = resolve_profiles(doc)["mcu"]["imu_mta"]["reports"]
    assert out == doc["mcu"]["imu_mta"]["reports"]


def test_doctor_imu_rates_uses_config_reports_without_preset(operator_config):
    resolved = _build_config(template_path(), None, "both")
    for slot in ("imu_mta", "imu_cra"):
        assert resolved["mcu"][slot]["reports"] == operator_config["mcu"][slot]["reports"]


def test_doctor_imu_rates_preset_still_applies():
    resolved = _build_config(template_path(), "raw_only", "mta")
    enabled = {n for n, c in resolved["mcu"]["imu_mta"]["reports"].items() if c["enabled"]}
    assert enabled == {"accelerometer", "gyroscope"}


def test_slot_without_enabled_report_rejected(operator_config):
    doc = copy.deepcopy(operator_config)
    for cfg in doc["mcu"]["imu_cra"]["reports"].values():
        cfg["enabled"] = False
    assert any("imu_cra is enabled but has no enabled report" in e for e in _hard(doc))
    doc["mcu"]["imu_cra"]["enabled"] = False
    assert not any("imu_cra" in e for e in _hard(doc))


def test_datasheet_motion_label_without_reports_rejected(operator_config):
    # The firmware does not know datasheet_motion and would fall back to kinematics_default.
    doc = copy.deepcopy(operator_config)
    doc["mcu"]["imu_mta"]["profile"] = "datasheet_motion"
    doc["mcu"]["imu_mta"]["reports"] = {}
    assert any("no enabled report" in e for e in _hard(doc))


def test_more_reports_than_firmware_slots_rejected(operator_config):
    doc = copy.deepcopy(operator_config)
    slot = doc["mcu"]["imu_mta"]
    slot["profile"] = "custom"
    slot["reports"] = {
        name: {"enabled": True, "rate_hz": 10}
        for name in list(BNO085_REPORT_CATALOG)[:9]
    }
    assert any("the firmware keeps only the first 8" in e for e in _hard(doc))


@pytest.mark.parametrize(
    "profile,rate,width",
    [("imx296_dual_csi", 30, 12000), ("imx296_high_fps", 100, 2000), ("bench_scope", 10, 1000),
     ("track_a_led", 30, 200)],
)
def test_trigger_presets_match_firmware(operator_config, profile, rate, width):
    # firmware/jtsmm/src/domain/trigger_config.c trigger_config_resolve_profile
    doc = copy.deepcopy(operator_config)
    trig = {"profile": profile}
    doc["mcu"]["trigger"] = trig
    out = resolve_profiles(doc)["mcu"]["trigger"]
    assert (out["rate_hz"], out["pulse"]["width_us"]) == (rate, width)
