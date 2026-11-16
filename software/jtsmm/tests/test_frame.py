# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Protocol frame parser: CRC16, resync, partial reads."""

from __future__ import annotations

import struct

from jtsmm.protocol import FrameParser
from jtsmm.protocol.crc import crc16_ccitt
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame, load_golden_vectors


def test_crc16_golden_vectors():
    for vec in load_golden_vectors()["crc_vectors"]:
        data = bytes.fromhex(vec["data_hex"])
        assert crc16_ccitt(data) == vec["crc"]


def test_crc16_empty_init():
    assert crc16_ccitt(b"") == 0xFFFF


def test_frame_roundtrip_emg():
    emg = bytearray(27)
    for i in range(8):
        val = (i + 1) & 0xFFFFFF
        off = 3 + i * 3
        emg[off] = val & 0xFF
        emg[off + 1] = (val >> 8) & 0xFF
        emg[off + 2] = (val >> 16) & 0xFF

    frame = build_frame(int(RecordType.EMG_BLOCK), 1_000_000, bytes(emg))
    parser = FrameParser()
    recs = parser.feed(frame)
    assert len(recs) == 1
    assert recs[0].record_type == int(RecordType.EMG_BLOCK)
    assert recs[0].t_master_us == 1_000_000
    assert len(recs[0].payload) == 27
    assert parser.crc_errors == 0


def test_partial_read_reassembly():
    payload = b"\x01\x02\x03\x04"
    frame = build_frame(int(RecordType.HEARTBEAT), 42, payload)
    parser = FrameParser()
    mid = len(frame) // 2
    assert parser.feed(frame[:mid]) == []
    recs = parser.feed(frame[mid:])
    assert len(recs) == 1
    assert recs[0].decoded.get("raw") == payload or recs[0].payload == payload


def test_resync_after_corruption():
    good = build_frame(int(RecordType.TRIGGER_EVENT), 100, b"\x00\x00\x00\x00" + struct.pack("<Q", 7))
    corrupt = b"\xff\xff" + good[2:]
    parser = FrameParser()
    recs = parser.feed(corrupt + good)
    assert len(recs) >= 1
    assert recs[-1].decoded.get("trigger_id") == 7
    assert parser.resyncs >= 1


def test_bad_crc_rejected():
    frame = bytearray(build_frame(int(RecordType.CMD_ACK), 0, b"\x01"))
    frame[-1] ^= 0xFF
    parser = FrameParser()
    assert parser.feed(bytes(frame)) == []
    assert parser.crc_errors == 1


def test_oversized_payload_resyncs():
    junk = b"\x55\xaa" + bytes([0x10, 250]) + b"\x00" * 8
    parser = FrameParser()
    parser.feed(junk)
    assert parser.resyncs >= 1
