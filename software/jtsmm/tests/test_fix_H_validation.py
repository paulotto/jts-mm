# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Host validation must reject what the firmware would silently change or ignore."""

from __future__ import annotations

import copy
import json
import re

import pytest

from jtsmm.config import validate_config
from jtsmm.config.catalog import (
    FIRMWARE_FUSION_PRIMARIES,
    FIRMWARE_IMU_REPORTS,
    FIRMWARE_TRIGGER_PROFILES,
)
from jtsmm.config.loader import checkout_root, load_json_schema


def _hard(doc: dict) -> list[str]:
    return [e for e in validate_config(doc) if not e.startswith("warn:")]


@pytest.fixture
def cfg(operator_config) -> dict:
    return copy.deepcopy(operator_config)


def test_template_is_clean(cfg):
    assert _hard(cfg) == []


@pytest.mark.parametrize("rate,width,ok", [(30, 12000, True), (30, 33333, False), (30, 40000, False),
                                           (200, 4999, True), (200, 5000, False)])
def test_pulse_width_must_be_below_period(cfg, rate, width, ok):
    cfg["mcu"]["trigger"]["rate_hz"] = rate
    cfg["mcu"]["trigger"]["pulse"]["width_us"] = width
    for cam in cfg["host"]["cameras"]["list"]:
        cam["trigger"]["expect_rate_hz"] = rate
        cam["exposure_us"] = min(width, 1_000_000 // rate - 1)
    errs = [e for e in _hard(cfg) if "pulse.width_us" in e]
    assert (errs == []) is ok, errs


def test_arvr_fusion_primary_rejected(cfg):
    slot = cfg["mcu"]["imu_mta"]
    slot["fusion"]["primary"] = "arvr_stabilized_game_rotation_vector"
    # Rejected by the schema enum; the cross-check is the second line of defence.
    assert any("mcu.imu_mta.fusion.primary" in e for e in _hard(cfg))
    cross = validate_config(cfg, include_schema=False)
    assert any("not implemented by the firmware" in e for e in cross)


def test_fusion_primary_must_match_enabled_report(cfg):
    slot = cfg["mcu"]["imu_mta"]
    slot["fusion"]["primary"] = "rotation_vector"
    assert any("enabled fusion report is game_rotation_vector" in e for e in _hard(cfg))


def test_fusion_rate_must_match_enabled_report(cfg):
    cfg["mcu"]["imu_cra"]["fusion"]["rate_hz"] = 200
    assert any("fusion.rate_hz=200" in e for e in _hard(cfg))


@pytest.mark.parametrize("name", ["gravity", "raw_accelerometer", "arvr_stabilized_rotation_vector"])
def test_unimplemented_report_rejected(cfg, name):
    cfg["mcu"]["imu_mta"]["profile"] = "custom"
    cfg["mcu"]["imu_mta"]["reports"][name]["enabled"] = True
    assert any(f"reports.{name} is not implemented" in e for e in _hard(cfg))


def test_trigger_disabled_with_mcu_cameras(cfg):
    # The firmware refuses trig_start while disabled: MCU-triggered cameras get nothing.
    cfg["mcu"]["trigger"]["enabled"] = False
    assert any("mcu.trigger.enabled=false" in e for e in _hard(cfg))
    cfg["host"]["cameras"]["list"] = []
    assert not any("mcu.trigger.enabled=false" in e for e in _hard(cfg))


def test_unknown_trigger_profile_rejected(cfg):
    cfg["mcu"]["trigger"]["profile"] = "imx296_turbo"
    assert any("mcu.trigger.profile" in e for e in _hard(cfg))
    assert any(
        "unknown to the firmware" in e for e in validate_config(cfg, include_schema=False)
    )
    cfg["mcu"]["trigger"]["profile"] = "track_a_led"
    assert _hard(cfg) == []


def test_duplicate_camera_id_and_device(cfg):
    cams = cfg["host"]["cameras"]["list"]
    cams[1]["id"] = cams[0]["id"]
    assert any(".id='cam0' duplicates" in e for e in _hard(cfg))
    cams[1]["id"] = "cam1"
    cams[1]["device"] = "0"  # csi0 alias
    assert any("is the same camera" in e for e in _hard(cfg))


# --- parity with the firmware sources ------------------------------------------------


def _fw(rel: str) -> str:
    root = checkout_root()
    if root is None:
        pytest.skip("needs the repository checkout")
    return (root / "firmware" / "jtsmm" / rel).read_text(encoding="utf-8")


def _strcmp_names(src: str, func: str) -> set[str]:
    body = src.split(f"{func}(const char *s)", 1)[1].split("\n}\n", 1)[0]
    return set(re.findall(r'strcmp\(s, "([a-z0-9_]+)"\)', body))


def test_firmware_report_catalog_parity():
    src = _fw("src/domain/bno085_reports.c")
    names = set(re.findall(r'\{0x[0-9A-Fa-f]+, "([a-z_]+)"', src))
    assert names == set(FIRMWARE_IMU_REPORTS)


def test_firmware_fusion_primary_parity():
    src = _fw("src/runtime/config_json.c")
    # NULL/unknown -> IMU_FUSION_GAME_ROTATION_VECTOR, the one name without a strcmp.
    names = _strcmp_names(src, "parse_fusion_primary") | {"game_rotation_vector"}
    assert names == set(FIRMWARE_FUSION_PRIMARIES)
    enum = load_json_schema()["$defs"]["mcu_imu_slot"]["properties"]["fusion"]["properties"][
        "primary"
    ]["enum"]
    assert set(enum) == set(FIRMWARE_FUSION_PRIMARIES)


def test_firmware_trigger_profile_parity():
    src = _fw("src/runtime/config_json.c")
    names = _strcmp_names(src, "parse_trigger_profile") | {"imx296_dual_csi"}
    assert names == set(FIRMWARE_TRIGGER_PROFILES)
    enum = load_json_schema()["$defs"]["mcu_trigger"]["properties"]["profile"]["enum"]
    assert set(enum) == set(FIRMWARE_TRIGGER_PROFILES)


def test_schema_narrowed_to_firmware_integer_widths():
    schema = json.dumps(load_json_schema())
    assert '"decimation_factor": {"type": "integer", "minimum": 1, "maximum": 255}' in schema
    assert '"amplitude_mv": {"type": "integer", "minimum": 0, "maximum": 255}' in schema


@pytest.mark.parametrize(
    "path,value",
    [
        (("mcu", "platform", "system", "xosc_hz"), 24_000_000),
        (("mcu", "platform", "pio", "trigger", "clock_hz"), 2_000_000),
        (("mcu", "platform", "irq", "emg"), 1),
        (("mcu", "platform", "irq", "imu"), 0),
        (("mcu", "emg", "stream_channels"), 8),
        (("mcu", "emg", "bias", "ch7_readback"), True),
        (("mcu", "emg", "test_signal", "amplitude_mv"), 5),
        (("mcu", "trigger", "outputs", "csi2"), False),
        (("mcu", "trigger", "outputs", "sync_led"), True),
        (("mcu", "trigger", "start_on"), "manual"),
        (("host", "cameras", "marker_detector"), "aruco"),
        (("mcu", "emg", "custom"), {"default_gain": 12}),
    ],
)
def test_single_value_keys_rejected_with_reason(cfg, path, value):
    node = cfg
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value
    dotted = ".".join(path)
    errs = [e for e in validate_config(cfg, include_schema=False) if not e.startswith("warn:")]
    assert any(e.startswith(dotted) and ("refuses" in e or "no effect" in e or "remove" in e)
               for e in errs), errs


@pytest.mark.parametrize(
    "path,value",
    [(("mcu", "platform", "profile"), "debug"), (("mcu", "platform", "spi0", "hz"), 8_000_000),
     (("mcu", "platform", "irq", "trigger"), 4), (("mcu", "emg", "sample_rate_sps"), 16000)],
)
def test_schema_mirrors_firmware_limits(cfg, path, value):
    node = cfg
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    assert any(".".join(path) in e for e in _hard(cfg))
