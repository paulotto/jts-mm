# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pre-flight checks before record / analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jtsmm.capture.serial_util import candidate_ports, probe_port
from jtsmm.config import config_hash, load, validate_config

from .mcu import run_mcu_selftest
from .mcu_info import query_mcu_info
from .sysfs import check_imx296_trigger_mode
from .preflight_issues import partition_preflight_issues
from .viewer import check_viewer_preflight


def _check_firmware_version(resolved: dict[str, Any], port: str | None) -> list[str]:
    issues: list[str] = []
    hat = resolved.get("hat") or {}
    spec = hat.get("firmware_version")
    if not spec:
        return issues
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version

        info = query_mcu_info(port or hat.get("serial_port"))
        if Version(info.firmware_version.split("-")[0]) not in SpecifierSet(str(spec)):
            issues.append(
                f"firmware {info.firmware_version} not in required range {spec}"
            )
    except Exception as exc:
        issues.append(f"warn: firmware version check skipped: {exc}")
    return issues


def run_preflight(
    resolved: dict[str, Any] | None = None,
    *,
    config_path: Path | str | None = None,
    check_serial: bool = True,
    check_mcu_selftest: bool = False,
    check_firmware: bool = True,
    require_hat: bool = False,
) -> list[str]:
    """Return list of issues (empty = pass). Warnings prefixed with 'warn:'.

    ``require_hat`` makes an undetected jtsmm_hat an error instead of a
    warning -- for a bare `jtsmm doctor`, whose exit code should say whether
    the rig is ready. Record preflight keeps it a warning (the session opens
    its own port and fails loudly there).
    """
    issues: list[str] = []
    if resolved is None:
        try:
            resolved = load(config_path, validate=False)
        except Exception as exc:
            return [f"config load failed: {exc}"]

    issues.extend(validate_config(resolved))
    issues.extend(check_imx296_trigger_mode(resolved))

    try:
        _ = config_hash(resolved)
    except Exception as exc:
        issues.append(f"config_hash computation failed: {exc}")

    hat_port = (resolved.get("hat") or {}).get("serial_port")
    if check_serial:
        found = False
        for p in candidate_ports(hat_port):
            if probe_port(p):
                found = True
                hat_port = p
                break
        if not found:
            msg = "jtsmm_hat serial port not detected"
            issues.append(msg if require_hat else f"warn: {msg}")

    if check_firmware and hat_port:
        issues.extend(_check_firmware_version(resolved, hat_port))

    if check_mcu_selftest:
        mcu = resolved.get("mcu") or {}
        imus = {slot: bool((mcu.get(f"imu_{slot}") or {}).get("enabled", True))
                for slot in ("mta", "cra")}
        issues.extend(run_mcu_selftest(hat_port, imus_enabled=imus))

    issues.extend(check_viewer_preflight(resolved))

    return issues
