# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU profile, fusion, sync mode, bus budget validation."""

from __future__ import annotations

from jtsmm.config import resolve_profiles, validate_config


def _base():
    from tests.helpers import load_operator_template

    return load_operator_template()


def test_kinematics_default_reports():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "kinematics_default"
    doc["mcu"]["imu_mta"].pop("reports")  # a preset only fills a slot without reports
    doc = resolve_profiles(doc)
    reports = doc["mcu"]["imu_mta"]["reports"]
    assert reports["game_rotation_vector"]["enabled"] is True
    assert reports["linear_acceleration"]["rate_hz"] == 250
    # Gyro is intentionally not in the default (bus-budget; keeps lin at rate).
    assert "gyroscope" not in reports


def test_emg_drdy_divisor_valid():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["sync"] = {"mode": "emg_drdy_divisor"}
    for name, cfg in doc["mcu"]["imu_mta"]["reports"].items():
        if cfg.get("enabled", True):
            cfg["rate_hz"] = 250
    doc["mcu"]["imu_mta"]["fusion"]["rate_hz"] = 250  # must agree with the fusion report
    errors = [e for e in validate_config(doc) if not e.startswith("warn:")]
    assert errors == []


def test_emg_drdy_divisor_rejects_non_integer_ratio():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "kinematics_default"
    doc = resolve_profiles(doc)
    doc["mcu"]["imu_mta"]["sync"] = {"mode": "emg_drdy_divisor"}
    # game_rv @ 390 Hz does not divide the 2000 Hz EMG DRDY evenly.
    doc["mcu"]["imu_mta"]["reports"]["game_rotation_vector"]["rate_hz"] = 390
    errors = validate_config(doc)
    assert any("emg_drdy_divisor" in e for e in errors)


def test_all_profile_matches_firmware_preset():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "all"
    doc["mcu"]["imu_mta"].pop("reports")
    doc = resolve_profiles(doc)
    reports = doc["mcu"]["imu_mta"]["reports"]
    # imu_config.c IMU_PROFILE_ALL: accel, gyro, mag, linear accel + the fusion primary.
    assert set(reports) == {
        "accelerometer", "gyroscope", "magnetometer", "linear_acceleration",
        "game_rotation_vector",
    }
    assert [e for e in validate_config(doc) if not e.startswith("warn:")] == []


def test_all_profile_bus_budget_error():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "all"
    doc["mcu"]["imu_mta"]["reports"] = {
        "accelerometer": {"enabled": True, "rate_hz": 500},
        "gyroscope": {"enabled": True, "rate_hz": 400},
        "magnetometer": {"enabled": True, "rate_hz": 100},
        "linear_acceleration": {"enabled": True, "rate_hz": 400},
        "game_rotation_vector": {"enabled": True, "rate_hz": 390},
    }
    doc = resolve_profiles(doc)
    errors = validate_config(doc)
    assert any("bus budget" in e for e in errors)


def test_unknown_fusion_primary():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["fusion"]["primary"] = "not_a_fusion_mode"
    errors = validate_config(doc)
    assert any("fusion.primary" in e for e in errors)


def test_int_driven_default_sync():
    doc = resolve_profiles(_base())
    assert doc["mcu"]["imu_mta"]["sync"]["mode"] == "int_driven"


# Rates the firmware boot resolver (firmware/jtsmm/src/domain/imu_config.c
# imu_config_resolve_profile) produces for each preset with fusion @ 390 Hz.
# raw_only is NOT in lockstep: the firmware uses the SH-2 catalog default
# (100 Hz) while the host preset asks for 400 Hz. It only matters for a boot
# without a host apply -- the host always sends an explicit report map.
_FIRMWARE_BOOT_RATES = {
    "fused_only": {"game_rotation_vector": 390},
    "kinematics_accel": {"game_rotation_vector": 390, "accelerometer": 250},
    "kinematics_default": {"game_rotation_vector": 390, "linear_acceleration": 250},
}


def test_host_preset_rates_match_firmware_boot():
    from jtsmm.config.catalog import IMU_FUSION_RATE_HZ, IMU_PROFILE_REPORTS

    assert IMU_FUSION_RATE_HZ == 390
    for preset, fw_rates in _FIRMWARE_BOOT_RATES.items():
        host = {
            name: cfg["rate_hz"]
            for name, cfg in IMU_PROFILE_REPORTS[preset].items()
            if cfg.get("enabled", True)
        }
        assert host == fw_rates, f"{preset}: host {host} != firmware {fw_rates}"


def test_multiple_fusion_rejected_outside_custom():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "custom"
    doc = resolve_profiles(doc)
    doc["mcu"]["imu_mta"]["reports"] = {
        "game_rotation_vector": {"enabled": True, "rate_hz": 390},
        "rotation_vector": {"enabled": True, "rate_hz": 200},
    }
    # Under custom it is a warning (operator opt-in), not a hard error.
    errors = [e for e in validate_config(doc) if not e.startswith("warn:")]
    assert not any("multiple fusion" in e for e in errors)

    doc["mcu"]["imu_mta"]["profile"] = "kinematics_default"
    hard = [e for e in validate_config(doc) if not e.startswith("warn:")]
    assert any("multiple fusion" in e for e in hard)


def test_geomagnetic_without_mag_warns():
    doc = resolve_profiles(_base())
    doc["mcu"]["imu_mta"]["profile"] = "custom"
    doc = resolve_profiles(doc)
    doc["mcu"]["imu_mta"]["reports"] = {
        "geomagnetic_rotation_vector": {"enabled": True, "rate_hz": 200},
    }
    errors = validate_config(doc)
    assert any("magnetometer" in e and e.startswith("warn:") for e in errors)


def test_default_is_kinematics_accel_everywhere():
    """Template, missing profile key and preset all mean game_rv + accelerometer."""
    from jtsmm.config import build_mcu_apply_payload
    from jtsmm.config.catalog import DEFAULT_IMU_PROFILE

    assert DEFAULT_IMU_PROFILE == "kinematics_accel"
    want = {"game_rotation_vector": 390, "accelerometer": 250}
    doc = _base()
    for slot in ("imu_mta", "imu_cra"):
        on = {n: c["rate_hz"] for n, c in build_mcu_apply_payload(doc)[slot]["reports"].items()}
        assert doc["mcu"][slot]["profile"] == "kinematics_accel"
        assert on == want
    doc["mcu"]["imu_mta"] = {"enabled": True}  # no profile, no reports
    on = {n: c["rate_hz"] for n, c in build_mcu_apply_payload(doc)["imu_mta"]["reports"].items()}
    assert on == want


def test_apply_payload_spells_out_preset_reports():
    """A preset without reports reaches the MCU as an explicit map, so the MCU
    runs the host's set (it has no copy of datasheet_motion, and used to fall
    back to kinematics_default silently)."""
    from jtsmm.config import build_mcu_apply_payload

    doc = _base()
    doc["mcu"]["imu_mta"] = {"enabled": True, "profile": "datasheet_motion"}
    doc["mcu"]["imu_cra"] = {"enabled": False, "profile": "kinematics_default"}
    p = build_mcu_apply_payload(doc)
    assert set(p["imu_mta"]["reports"]) == {
        "accelerometer", "gyroscope", "magnetometer", "linear_acceleration"}
    assert "reports" not in p["imu_cra"]  # a disabled slot is left alone
    doc["mcu"]["imu_mta"] = {"enabled": True, "profile": "all"}
    assert "reports" not in build_mcu_apply_payload(doc)["imu_mta"]  # firmware preset


def test_profile_label_that_disagrees_with_reports_warns():
    doc = resolve_profiles(_base())
    msgs = validate_config(doc)
    assert not any("differ from profile" in m for m in msgs)
    reports = doc["mcu"]["imu_mta"]["reports"]
    reports["accelerometer"]["enabled"] = False
    reports["linear_acceleration"] = {"enabled": True, "rate_hz": 250}
    msgs = validate_config(doc)
    hit = [m for m in msgs if "differ from profile" in m]
    assert hit and hit[0].startswith("warn:") and "kinematics_accel" in hit[0]
