# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Protocol constants and decoded record containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

SYNC = b"\x55\xaa"
HEADER_SIZE = 4  # sync(2) + type + len
T_MASTER_SIZE = 8
CRC_SIZE = 2
MAX_PAYLOAD = 240


class RecordType(IntEnum):
    EMG_BLOCK = 0x10
    IMU_MTA_QUAT = 0x11
    IMU_MTA_RAW = 0x12
    IMU_MTA_LIN_ACCEL = 0x13
    IMU_CRA_QUAT = 0x14
    IMU_CRA_RAW = 0x15
    IMU_CRA_LIN_ACCEL = 0x16
    IMU_DP_QUAT = 0x17
    IMU_DP_RAW = 0x18
    IMU_DP_LIN_ACCEL = 0x19
    TRIGGER_EVENT = 0x20
    LED_FLASH_EVENT = 0x21
    BUTTON_EVENT = 0x22
    HEARTBEAT = 0x30
    PPS_REPLY = 0x40
    ADS_STATUS = 0x50
    STATS = 0x51
    LOG = 0x70
    CMD_ACK = 0xE0


RECORD_NAMES: dict[int, str] = {int(k): k.name for k in RecordType}


@dataclass(frozen=True)
class McuRecord:
    record_type: int
    t_master_us: int
    payload: bytes
    decoded: dict[str, Any] = field(default_factory=dict)

    @property
    def type_name(self) -> str:
        return RECORD_NAMES.get(self.record_type, f"0x{self.record_type:02X}")
