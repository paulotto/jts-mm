# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MCU-side doctor helpers (selftest over USB)."""

from __future__ import annotations

import re
import time

import serial

from jtsmm.capture.serial_util import candidate_ports, probe_port


_IMU_PROBE_RE = re.compile(r"IMU probe MTA=(\d) CRA=(\d)")


def _grade_selftest(text: str, imus_enabled: dict[str, bool] | None) -> list[str]:
    """Issues from a finished selftest transcript.

    The firmware probes both IMU slots whatever the config says and fails only
    when neither answers. With ``imus_enabled`` (this session's
    ``mcu.imu_*.enabled``) the IMU part is graded against the config instead:
    a disabled slot is not required, an enabled one that does not answer is an
    error. Older firmware without the probe line keeps the firmware's verdict.
    """
    fails = [ln.strip() for ln in text.splitlines() if "[FAIL]" in ln]
    probe = _IMU_PROBE_RE.search(text)
    imu_issues: list[str] = []
    if imus_enabled is not None and probe is not None:
        fails = [ln for ln in fails if "IMU probe" not in ln]
        for slot, seen in (("mta", probe.group(1) == "1"), ("cra", probe.group(2) == "1")):
            if imus_enabled.get(slot, True) and not seen:
                imu_issues.append(
                    f"error: IMU {slot.upper()} is enabled in the config but does not answer "
                    f"on I2C (plug it in, or set mcu.imu_{slot}.enabled: false)"
                )
    issues = [f"error: {ln}" for ln in fails]
    if issues:
        issues.insert(0, "error: MCU selftest reported failure")
    return issues + imu_issues


def run_mcu_selftest(
    port: str | None = None,
    *,
    timeout_s: float = 8.0,
    imus_enabled: dict[str, bool] | None = None,
) -> list[str]:
    """Send `selftest` command; return issues (empty = pass).

    ``imus_enabled`` ({"mta": bool, "cra": bool}) grades the IMU probe against
    the session config (see :func:`_grade_selftest`).
    """
    issues: list[str] = []
    found_port: str | None = None
    for p in candidate_ports(port):
        if probe_port(p):
            found_port = p
            break
    if found_port is None:
        return ["error: jtsmm_hat serial port not detected for MCU selftest"]

    try:
        with serial.Serial(found_port, 115200, timeout=0.05) as ser:
            ser.reset_input_buffer()
            ser.write(b"selftest\r\n")
            ser.flush()
            deadline = time.monotonic() + timeout_s
            buf = bytearray()
            while time.monotonic() < deadline:
                n = ser.in_waiting
                if n:
                    buf.extend(ser.read(n))
                    text = bytes(buf).decode("utf-8", errors="replace")
                    if "=== result:" in text:
                        return _grade_selftest(text, imus_enabled)
                else:
                    time.sleep(0.05)
    except (serial.SerialException, OSError) as exc:
        return [f"error: MCU selftest serial error: {exc}"]

    issues.append("warn: MCU selftest timed out without result line")
    return issues
