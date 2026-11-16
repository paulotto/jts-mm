# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Configuration load, validation, registry, and MCU apply."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hash import config_hash
from .loader import default_config_path, load_yaml, note_loaded_config
from .overrides import apply_emg_overrides
from .mcu_apply import (
    apply_mcu,
    build_mcu_apply_payload,
    canonical_applied_config,
    format_apply_command,
    open_hat_serial,
    read_config_dump,
    start_mcu_sensor_streams,
    start_mcu_streams,
    start_mcu_trigger,
    stop_mcu_streams,
    stop_mcu_streams_robust,
    verify_mcu_config,
)
from .profiles import load_profile_yaml, resolve_profiles
from .strict import raise_on_errors, validate_strict
from .validate import validate_config

__all__ = [
    "applied_config_hash",
    "apply_emg_overrides",
    "apply_mcu",
    "build_mcu_apply_payload",
    "canonical_applied_config",
    "config_hash",
    "default_config_path",
    "format_apply_command",
    "load",
    "load_yaml",
    "open_hat_serial",
    "raise_on_errors",
    "read_config_dump",
    "start_mcu_sensor_streams",
    "start_mcu_streams",
    "start_mcu_trigger",
    "verify_mcu_config",
    "stop_mcu_streams",
    "stop_mcu_streams_robust",
    "resolve_profiles",
    "split_host_mcu",
    "validate_config",
    "validate_strict",
]


def load(
    path: Path | str | None = None,
    *,
    resolve: bool = False,
    validate: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load operator YAML from path (default: checkout etc, else ~/jtsmm.yaml).

    ``validate=False`` returns the document unchecked (doctor's debug path).
    """
    cfg_path = Path(path) if path else default_config_path()
    raw = load_yaml(cfg_path)
    note_loaded_config(cfg_path)
    doc = resolve_profiles(raw) if resolve else raw
    if validate:
        # strict=False keeps type/value and cross-subsystem checks but skips
        # completeness and demotes unknown keys to warnings.
        raise_on_errors(validate_strict(doc, path=cfg_path, strict=strict))
    return doc


def applied_config_hash(resolved: dict[str, Any]) -> str:
    """SHA-256 of the *effective applied* config (wire-form MCU blocks)."""
    return config_hash(canonical_applied_config(resolved))


def split_host_mcu(resolved: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    host = resolved.get("host") or {}
    session = resolved.get("session") or {}
    mcu_payload = build_mcu_apply_payload(resolved)
    return mcu_payload, {"host": host, "session": session, "hat": resolved.get("hat", {})}
