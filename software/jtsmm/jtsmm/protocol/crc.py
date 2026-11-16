# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CRC16-CCITT for host_link frames (poly 0x1021, init 0xFFFF)."""

from __future__ import annotations

_POLY = 0x1021


def crc16_ccitt(data: bytes | bytearray | memoryview, init: int = 0xFFFF) -> int:
    crc = init & 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc
