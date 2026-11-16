# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Host-side CRC/frame encode mirroring firmware host_link.c."""

from __future__ import annotations

import struct

from jtsmm.protocol.crc import crc16_ccitt
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame, load_golden_vectors
from jtsmm.protocol import FrameParser


def _firmware_crc_reference(data: bytes) -> int:
    """Bit-exact reimplementation of host_link_crc16() in firmware."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte & 0xFF) << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def test_crc_matches_firmware_reference():
    samples = [b"", b"\x10", b"\x10\x1b" + b"\x00" * 8, bytes(range(32))]
    for data in samples:
        assert crc16_ccitt(data) == _firmware_crc_reference(data)


def test_golden_crc_vectors_match_firmware():
    for vec in load_golden_vectors()["crc_vectors"]:
        data = bytes.fromhex(vec["data_hex"])
        assert _firmware_crc_reference(data) == vec["crc"]
        assert crc16_ccitt(data) == vec["crc"]


def test_frame_encode_decode_roundtrip_all_golden():
    parser = FrameParser()
    for entry in load_golden_vectors()["frames"]:
        payload = bytes.fromhex(entry["payload_hex"])
        frame = build_frame(entry["type"], entry["t_master_us"], payload)
        body = frame[2:-2]
        assert crc16_ccitt(body) == int.from_bytes(frame[-2:], "little")
        recs = parser.feed(frame)
        assert len(recs) == 1
        assert recs[0].payload == payload


def test_trigger_frame_layout():
    payload = b"\x01\x00\x00\x00" + struct.pack("<Q", 42)
    frame = build_frame(int(RecordType.TRIGGER_EVENT), 50_000, payload)
    assert frame[:2] == b"\x55\xaa"
    assert frame[2] == int(RecordType.TRIGGER_EVENT)
    assert frame[3] == len(payload)
