# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Incremental 0x55 0xAA frame parser with CRC16 resync."""

from __future__ import annotations

from struct import error as struct_error

from .crc import crc16_ccitt
from .records import decode_record
from .types import CRC_SIZE, HEADER_SIZE, MAX_PAYLOAD, SYNC, T_MASTER_SIZE, McuRecord


class FrameParser:
    """Parse framed USB records from a byte stream (pattern from emg_lab StreamParser)."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self.crc_errors = 0
        self.frames_ok = 0
        self.resyncs = 0

    def feed(self, data: bytes) -> list[McuRecord]:
        self._buf.extend(data)
        out: list[McuRecord] = []
        consume = 0
        buf = self._buf

        while True:
            if len(buf) - consume < HEADER_SIZE:
                break

            # Hunt sync
            if buf[consume] != 0x55 or buf[consume + 1] != 0xAA:
                consume += 1
                self.resyncs += 1
                continue

            if len(buf) - consume < HEADER_SIZE + T_MASTER_SIZE:
                break

            frame_start = consume
            rec_type = buf[consume + 2]
            payload_len = buf[consume + 3]
            if payload_len > MAX_PAYLOAD:
                consume += 1
                self.resyncs += 1
                continue

            frame_len = HEADER_SIZE + T_MASTER_SIZE + payload_len + CRC_SIZE
            if len(buf) - consume < frame_len:
                break

            frame = bytes(buf[consume : consume + frame_len])
            body = frame[2 : 2 + 1 + 1 + T_MASTER_SIZE + payload_len]
            crc_expected = int.from_bytes(frame[-2:], "little")
            crc_actual = crc16_ccitt(body)
            if crc_actual != crc_expected:
                self.crc_errors += 1
                consume += 1
                self.resyncs += 1
                continue

            t_off = HEADER_SIZE
            t_master_us = int.from_bytes(frame[t_off : t_off + T_MASTER_SIZE], "little")
            payload_off = t_off + T_MASTER_SIZE
            payload = frame[payload_off : payload_off + payload_len]
            try:
                decoded = decode_record(rec_type, payload)
            except (struct_error, ValueError):
                decoded = {"raw": payload}
            out.append(
                McuRecord(
                    record_type=rec_type,
                    t_master_us=t_master_us,
                    payload=payload,
                    decoded=decoded,
                )
            )
            self.frames_ok += 1
            consume = frame_start + frame_len

        if consume:
            del buf[:consume]
        return out
