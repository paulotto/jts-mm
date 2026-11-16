# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""0x55 0xAA framed USB protocol — decode only (no I/O)."""

from .crc import crc16_ccitt
from .frame import FrameParser
from .records import decode_record
from .types import McuRecord, RecordType, SYNC

__all__ = [
    "FrameParser",
    "McuRecord",
    "RecordType",
    "SYNC",
    "crc16_ccitt",
    "decode_record",
]
