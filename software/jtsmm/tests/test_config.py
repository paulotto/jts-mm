# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Config load, profile resolution, cross-validation."""

from __future__ import annotations

import copy

import yaml

from jtsmm.config import load, resolve_profiles, validate_config
from jtsmm.config.loader import load_yaml, repo_root, schema_dir


def test_schema_files_present():
    assert (schema_dir() / "jtsmm.schema.json").is_file()
    assert (repo_root() / "schema" / "v1" / "profiles" / "emg_acquisition.yaml").is_file()


def test_schema_camera_buffer_count_default_matches_code():
    """Schema default = capture_cam.CameraCapture default = etc template (6)."""
    import json

    from jtsmm.capture.capture_cam import CameraCapture

    schema = json.loads((schema_dir() / "jtsmm.schema.json").read_text())
    default = schema["$defs"]["host_camera_entry"]["properties"]["buffer_count"]["default"]
    assert default == CameraCapture.buffer_count == 6


def test_resolve_default_template(resolved_config):
    emg = resolved_config["mcu"]["emg"]
    assert emg["profile"] == "acquisition"
    assert emg["sample_rate_sps"] == 4000
    assert emg["default_gain"] == 6

    trig = resolved_config["mcu"]["trigger"]
    assert trig["rate_hz"] == 40
    assert trig["pulse"]["width_us"] == 12000

    imu = resolved_config["mcu"]["imu_mta"]
    assert "game_rotation_vector" in imu["reports"]
    # 390 Hz (not 400): the BNO085 GRV tops out at its native ~398 Hz, so the
    # template declares 390 to keep the strict 0.5% integrity drop budget robust.
    assert imu["reports"]["game_rotation_vector"]["rate_hz"] == 390


def test_validate_default_passes(resolved_config):
    hard = [e for e in validate_config(resolved_config) if not e.startswith("warn:")]
    assert hard == []


def test_trigger_camera_rate_mismatch(resolved_config):
    doc = copy.deepcopy(resolved_config)
    doc["mcu"]["trigger"]["rate_hz"] = 100
    errors = validate_config(doc)
    assert any("expect_rate_hz" in e for e in errors)


def test_exposure_exceeds_pulse_budget(resolved_config):
    doc = copy.deepcopy(resolved_config)
    # Pulse budget is width_us + IMX296_EXPOSURE_MARGIN_US (~14.26 µs).
    pulse_us = int(doc["mcu"]["trigger"]["pulse"]["width_us"])
    doc["host"]["cameras"]["list"][0]["exposure_us"] = pulse_us + 1000
    errors = validate_config(doc)
    assert any("exposure_us" in e for e in errors)


def test_invalid_emg_sample_rate(resolved_config):
    doc = copy.deepcopy(resolved_config)
    doc["mcu"]["emg"]["sample_rate_sps"] = 333
    errors = validate_config(doc)
    assert any("sample_rate_sps" in e for e in errors)


def test_custom_emg_profile(tmp_path, operator_config):
    doc = copy.deepcopy(operator_config)
    doc["mcu"]["emg"] = {
        "profile": "custom",
        "custom": {"sample_rate_sps": 2000, "default_gain": 12},
    }
    path = tmp_path / "custom.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    resolved = resolve_profiles(load_yaml(path))
    assert resolved["mcu"]["emg"]["sample_rate_sps"] == 2000
    assert resolved["mcu"]["emg"]["default_gain"] == 12


def test_load_from_template(tmp_config_path):
    cfg = load(tmp_config_path, validate=True)
    assert cfg["schema_version"] == 1
