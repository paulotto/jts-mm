# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Merge schema profile fragments into operator YAML."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


from .catalog import (
    BNO085_REPORT_CATALOG,
    DEFAULT_IMU_PROFILE,
    EMG_PROFILE_OVERRIDES,
    FUSION_REPORTS,
    IMU_ALL_PRESET_REPORTS,
    IMU_FUSION_RATE_HZ,
    IMU_PROFILE_REPORTS,
    PROFILE_FILES,
    TRIGGER_PROFILE_OVERRIDES,
    FUSION_PRIMARY_TO_REPORT,
)
from .loader import load_yaml, profiles_dir


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overlay into a copy of base."""
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_profile_fragment(filename: str) -> dict[str, Any]:
    path = profiles_dir() / filename
    if not path.is_file():
        return {}
    data = load_yaml(path)
    data.pop("profile", None)
    return data


def _resolve_block(
    block: dict[str, Any],
    *,
    domain: str,
    profile_map: dict[str, str],
    extra_overrides: dict[str, dict] | None = None,
) -> dict[str, Any]:
    block = copy.deepcopy(block or {})
    profile = block.get("profile", "default")
    custom = block.pop("custom", None)
    scalar_overrides = {k: v for k, v in block.items() if k != "profile"}

    resolved: dict[str, Any] = {"profile": profile}
    if profile != "custom":
        fragment_name = profile_map.get(profile)
        if fragment_name:
            resolved = deep_merge(resolved, _load_profile_fragment(fragment_name))
        if extra_overrides and profile in extra_overrides:
            resolved = deep_merge(resolved, extra_overrides[profile])
        resolved["profile"] = profile
    elif isinstance(custom, dict):
        resolved = deep_merge(resolved, custom)

    resolved = deep_merge(resolved, scalar_overrides)
    if profile == "custom" and isinstance(custom, dict):
        resolved = deep_merge(resolved, custom)
    return resolved


def _resolve_imu_slot(block: dict[str, Any]) -> dict[str, Any]:
    block = dict(block or {})
    block.setdefault("profile", DEFAULT_IMU_PROFILE)  # _resolve_block's own is "default"
    resolved = _resolve_block(
        block,
        domain="imu",
        profile_map=PROFILE_FILES["imu"],
    )
    profile = resolved["profile"]
    fusion = resolved.get("fusion") or {}
    primary = fusion.get("primary", "game_rotation_vector")
    fusion_rate = fusion.get("rate_hz", IMU_FUSION_RATE_HZ)
    explicit = (block or {}).get("reports")
    has_explicit = isinstance(explicit, dict) and any(
        isinstance(cfg, dict) and cfg.get("enabled", True) for cfg in explicit.values()
    )

    if has_explicit:
        # An explicit report map wins over the preset, exactly as on the MCU
        # (config_json.c switches the slot to custom when reports are given).
        # The preset only fills a slot that has no enabled report.
        reports = copy.deepcopy(explicit)
    elif profile == "all":
        # Same set as the firmware's boot preset (imu_config.c IMU_PROFILE_ALL):
        # accel + gyro + mag + linear accel, plus the fusion.primary quaternion.
        reports = {
            name: {"enabled": True, "rate_hz": BNO085_REPORT_CATALOG[name].default_rate_hz}
            for name in IMU_ALL_PRESET_REPORTS
        }
        report_name = FUSION_PRIMARY_TO_REPORT.get(primary)
        if report_name:
            reports[report_name] = {"enabled": True, "rate_hz": fusion_rate}
    elif profile == "custom":
        reports = copy.deepcopy(resolved.get("reports") or {})
        if not reports and primary in FUSION_PRIMARY_TO_REPORT:
            report_name = FUSION_PRIMARY_TO_REPORT[primary]
            reports[report_name] = {"enabled": True, "rate_hz": fusion_rate}
    else:
        reports = copy.deepcopy(IMU_PROFILE_REPORTS.get(profile, {}))
        # The fusion.primary/rate_hz block is *sugar* that selects which fusion
        # quaternion a fusion-bearing preset emits (and at what rate). Only apply
        # it to presets that already include a fusion report — never inject a
        # quaternion into fusion-free presets like datasheet_motion / raw_only.
        if any(name in FUSION_REPORTS for name in reports):
            report_name = FUSION_PRIMARY_TO_REPORT.get(primary)
            if report_name:
                for name in list(reports):
                    if name in FUSION_REPORTS and name != report_name:
                        reports.pop(name)
                entry = reports.setdefault(report_name, {})
                entry["enabled"] = True
                entry["rate_hz"] = fusion_rate

    resolved["reports"] = reports
    resolved.setdefault("enabled", True)
    resolved.setdefault("sync", {"mode": "int_driven"})
    return resolved


def resolve_profiles(doc: dict[str, Any]) -> dict[str, Any]:
    """Return a fully resolved config document."""
    out = copy.deepcopy(doc)
    mcu = out.setdefault("mcu", {})

    mcu["platform"] = _resolve_block(
        mcu.get("platform", {}),
        domain="platform",
        profile_map=PROFILE_FILES["platform"],
    )
    mcu["emg"] = _resolve_block(
        mcu.get("emg", {}),
        domain="emg",
        profile_map=PROFILE_FILES["emg"],
        extra_overrides=EMG_PROFILE_OVERRIDES,
    )
    mcu["imu_mta"] = _resolve_imu_slot(mcu.get("imu_mta", {}))
    mcu["imu_cra"] = _resolve_imu_slot(mcu.get("imu_cra", {}))
    mcu["trigger"] = _resolve_block(
        mcu.get("trigger", {}),
        domain="trigger",
        profile_map=PROFILE_FILES["trigger"],
        extra_overrides=TRIGGER_PROFILE_OVERRIDES,
    )
    return out


def load_profile_yaml(path: Path | str) -> dict[str, Any]:
    """Load and resolve a config file."""
    from .validate import validate_config

    raw = load_yaml(path)
    resolved = resolve_profiles(raw)
    errors = validate_config(resolved)
    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))
    return resolved
