# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Serial port discovery for jtsmm_hat USB-CDC (from emg_lab)."""

from __future__ import annotations

import glob
import time
from pathlib import Path

import serial

INFO_PROBE = b"info\r\n"
HAT_MARKER = b"jtsmm"


def candidate_ports(explicit: str | None = None) -> list[str]:
    ports: list[str] = []
    if explicit:
        ports.append(explicit)
    for path in ("/dev/jtsmm_hat",):
        if Path(path).exists() and path not in ports:
            ports.append(path)
    ports.extend(sorted(glob.glob("/dev/ttyACM*")))
    ports.extend(sorted(glob.glob("/dev/ttyUSB*")))
    seen: set[str] = set()
    out: list[str] = []
    for p in ports:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def probe_port(port: str, baud: int = 115200, timeout_s: float = 0.6) -> bool:
    try:
        with serial.Serial(port, baud, timeout=0.05) as ser:
            ser.dtr = False
            time.sleep(0.05)
            ser.dtr = True
            time.sleep(0.25)
            ser.reset_input_buffer()
            ser.write(INFO_PROBE)
            ser.flush()
            deadline = time.monotonic() + timeout_s
            buf = bytearray()
            while time.monotonic() < deadline:
                n = ser.in_waiting
                if n:
                    buf.extend(ser.read(n))
                    if HAT_MARKER in buf:
                        return True
                else:
                    time.sleep(0.02)
            return HAT_MARKER in buf
    except (serial.SerialException, OSError):
        return False


def find_hat_port(explicit: str | None = None) -> str:
    tried: list[str] = []
    for port in candidate_ports(explicit):
        tried.append(port)
        if probe_port(port):
            return port
    msg = "No jtsmm_hat found on serial ports."
    if tried:
        msg += f" Tried: {', '.join(tried)}."
    msg += " Connect J_USB1, install udev rules (software/jtsmm/etc/udev/99-jtsmm.rules), or pass --port /dev/ttyACM0."
    raise RuntimeError(msg)
