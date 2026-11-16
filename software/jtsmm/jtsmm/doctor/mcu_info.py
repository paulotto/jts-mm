# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Query MCU firmware info over J_USB1 CDC."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import serial

from jtsmm.capture.serial_util import find_hat_port

_FW_RE = re.compile(rb"jtsmm\s+([\d.]+(?:-[\w]+)?)", re.I)


@dataclass
class McuInfo:
    port: str
    firmware_version: str
    raw_banner: str
    config_fingerprint: int | None = None


def query_mcu_info(port: str | None = None, *, timeout_s: float = 2.0) -> McuInfo:
    """Send `info` and parse firmware banner."""
    serial_port = find_hat_port(port)
    with serial.Serial(serial_port, 115200, timeout=0.05) as ser:
        ser.dtr = False
        time.sleep(0.05)
        ser.dtr = True
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.write(b"info\r\n")
        ser.flush()
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            n = ser.in_waiting
            if n:
                buf.extend(ser.read(n))
                m = _FW_RE.search(bytes(buf))
                if m:
                    return McuInfo(
                        port=serial_port,
                        firmware_version=m.group(1).decode("ascii"),
                        raw_banner=bytes(buf).decode("utf-8", errors="replace"),
                    )
            time.sleep(0.02)
    raise RuntimeError(f"No jtsmm banner on {serial_port}")
