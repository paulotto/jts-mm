# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""M8a gate: the native C++ frame parser must match the Python FrameParser.

These run only when the compiled ``_jtsmm_native`` extension is present (built via
``software/jtsmm/native/build.sh``); otherwise they skip so the Python-only
install still passes CI.
"""

from __future__ import annotations

import struct

import pytest

from jtsmm import native
from jtsmm.protocol import FrameParser as PyFrameParser
from jtsmm.protocol.records import STATS_FIELDS
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame

pytestmark = pytest.mark.skipif(not native.is_available(), reason="native extension not built")


def _stats_payload(**overrides: int) -> bytes:
    values = {name: i + 1 for i, name in enumerate(STATS_FIELDS)}
    values.update(overrides)
    return struct.pack(f"<{len(STATS_FIELDS)}I", *(values[n] for n in STATS_FIELDS))


def _mixed_stream() -> bytes:
    """One frame of every record type the decoder handles, back to back."""
    frames = [
        build_frame(int(RecordType.EMG_BLOCK), 1_000, bytes(range(27))),
        build_frame(int(RecordType.IMU_MTA_QUAT), 2_000, struct.pack("<4f", 1.0, 0.1, 0.2, 0.3)),
        build_frame(int(RecordType.IMU_CRA_QUAT), 3_000, struct.pack("<4f", 0.5, 0.5, 0.5, 0.5)),
        build_frame(int(RecordType.IMU_MTA_RAW), 4_000, struct.pack("<9h", *range(1, 10))),
        build_frame(int(RecordType.IMU_MTA_LIN_ACCEL), 5_000, struct.pack("<3f", 9.8, 0.0, -1.0)),
        build_frame(int(RecordType.TRIGGER_EVENT), 6_000, struct.pack("<4BQ", 1, 0, 1, 0, 42)),
        build_frame(int(RecordType.LED_FLASH_EVENT), 7_000, struct.pack("<I", 7)),
        build_frame(int(RecordType.PPS_REPLY), 8_000, struct.pack("<Q", 123456789)),
        build_frame(int(RecordType.STATS), 9_000, _stats_payload(tx_drops=99)),
        build_frame(int(RecordType.LOG), 10_000, b"hello watchdog"),
        build_frame(int(RecordType.CMD_ACK), 11_000, b"\x01"),
    ]
    return b"".join(frames)


def _as_comparable(rec) -> tuple:
    return (rec.record_type, rec.t_master_us, bytes(rec.payload), rec.decoded, rec.type_name)


def test_native_parser_matches_python():
    stream = _mixed_stream()
    py_recs = PyFrameParser().feed(stream)
    c_recs = native.FrameParser().feed(stream)

    assert len(c_recs) == len(py_recs) == 11
    for c, p in zip(c_recs, py_recs):
        assert _as_comparable(c) == _as_comparable(p)


def _stats_payload_n(n: int) -> bytes:
    return struct.pack(f"<{n}I", *range(100, 100 + n))


def _firmware_stream() -> bytes:
    """Every payload shape the firmware (and older firmware) actually emits."""
    q = struct.pack("<4f", 0.9, 0.1, -0.2, 0.3)
    frames = [
        # EMG: legacy 27 B and 31 B with the trailing u32 acquisition seq (K1).
        build_frame(0x10, 1_000, bytes(range(27))),
        build_frame(0x10, 1_500, bytes(range(27)) + struct.pack("<I", 0xFFFF_FFFE)),
        # QUAT: tagged + accuracy (18 B, what firmware sends), tagged (17), legacy (16).
        build_frame(0x11, 2_000, bytes([0x08]) + q + bytes([0x03])),
        build_frame(0x14, 2_100, bytes([0x05]) + q + bytes([0xFE])),  # acc masked to 2 bits
        build_frame(0x11, 2_200, bytes([0x2A]) + q),
        build_frame(0x14, 2_300, q),
        build_frame(0x11, 2_400, bytes([0x77]) + q + bytes([1])),  # unknown fusion id
    ]
    # RAW: tagged + accuracy (8 B) for every sensor id incl. an unknown one, tagged (7 B),
    # and legacy combined 18 B.
    for rt in (0x12, 0x15):
        for sid in (1, 2, 3, 9):
            frames.append(build_frame(rt, 3_000 + sid, bytes([sid]) + struct.pack("<3h", -5, 6, 32767) + bytes([sid & 3])))
            frames.append(build_frame(rt, 3_100 + sid, bytes([sid]) + struct.pack("<3h", 1, -2, -32768)))
        frames.append(build_frame(rt, 3_200, struct.pack("<9h", *range(-4, 5))))
    frames += [
        # LIN_ACCEL: with accuracy (13 B) and without (12 B).
        build_frame(0x13, 4_000, struct.pack("<3f", 0.5, -9.81, 1e-3) + bytes([2])),
        build_frame(0x16, 4_100, struct.pack("<3f", 1.0, 2.0, 3.0)),
        # Trigger: PIO stamp, timer fallback, unknown source.
        build_frame(0x20, 5_000, struct.pack("<4BQ", 1, 0, 0, 0, 7)),
        build_frame(0x20, 5_100, struct.pack("<4BQ", 2, 1, 0, 0, 8)),
        build_frame(0x20, 5_200, struct.pack("<4BQ", 0xAB, 0, 0, 0, 2**63 + 5)),
        build_frame(0x21, 6_000, struct.pack("<I", 0xDEADBEEF)),
        build_frame(0x22, 6_500, bytes([1, 1, 0, 0, 0])),  # BUTTON_EVENT (K7)
        build_frame(0x40, 7_000, struct.pack("<Q", 2**40)),
        # STATS: old firmware (28 u32), new firmware (36 u32, K2), and a future 40.
        build_frame(0x51, 8_000, _stats_payload_n(28)),
        build_frame(0x51, 8_100, _stats_payload_n(36)),
        build_frame(0x51, 8_200, _stats_payload_n(40)),
        # LOG: valid UTF-8, and a torn multi-byte sequence (must decode with U+FFFD).
        build_frame(0x70, 9_000, "PANEL: USER1 -> capture STOPPED µ".encode()),
        build_frame(0x70, 9_100, b"torn \xc3"),
        build_frame(0xE0, 10_000, b"\x00"),
        build_frame(0xE0, 10_100, b""),
        build_frame(0x30, 11_000, b"\x01\x02"),  # HEARTBEAT: raw passthrough
        # Malformed lengths must fall back to {"raw": ...} identically.
        build_frame(0x10, 12_000, bytes(28)),
        build_frame(0x11, 12_100, bytes(10)),
        build_frame(0x12, 12_200, bytes(5)),
        build_frame(0x13, 12_300, bytes(8)),
        build_frame(0x20, 12_400, bytes(11)),
        build_frame(0x22, 12_500, bytes(1)),
    ]
    return b"".join(frames)


def test_native_decode_matches_python_for_every_firmware_payload():
    stream = _firmware_stream()
    py_recs = PyFrameParser().feed(stream)
    c_recs = native.FrameParser().feed(stream)
    assert len(c_recs) == len(py_recs) > 40
    for c, p in zip(c_recs, py_recs):
        assert _as_comparable(c) == _as_comparable(p), p.type_name
    by_t = {p.t_master_us: p.decoded for p in py_recs}
    # Spot-check the fields the audit found missing on the native side.
    assert by_t[1_500]["seq"] == 0xFFFF_FFFE and "seq" not in by_t[1_000]
    assert by_t[2_000]["quat_accuracy"] == 3 and by_t[2_100]["quat_accuracy"] == 2
    assert "quat_accuracy" not in by_t[2_200]
    assert by_t[3_001]["accel_accuracy"] == 1 and by_t[3_003]["mag_accuracy"] == 3
    assert "gyro_accuracy" not in by_t[3_102]
    assert by_t[4_000]["linear_accel_accuracy"] == 2
    assert by_t[6_500] == {"button_id": 1, "pressed": True}
    assert len(by_t[8_000]) == 28 and len(by_t[8_100]) == len(STATS_FIELDS) == 36
    assert by_t[8_100]["stats_seq"] == 135
    assert by_t[9_100]["text"] == "torn \ufffd"
    assert by_t[12_000] == {"raw": bytes(28)}


def test_native_parser_reassembles_split_frames():
    """Bytes arriving in arbitrary chunks must yield the same records."""
    stream = _mixed_stream()
    parser = native.FrameParser()
    recs = []
    for i in range(0, len(stream), 7):  # 7-byte chunks straddle frame boundaries
        recs.extend(parser.feed(stream[i : i + 7]))

    py_recs = PyFrameParser().feed(stream)
    assert len(recs) == len(py_recs)
    for c, p in zip(recs, py_recs):
        assert _as_comparable(c) == _as_comparable(p)
    assert parser.crc_errors == 0


def test_native_parser_counts_crc_errors_and_recovers():
    good = build_frame(int(RecordType.EMG_BLOCK), 1_000, bytes(27))
    corrupt = bytearray(build_frame(int(RecordType.EMG_BLOCK), 2_000, bytes(27)))
    corrupt[-1] ^= 0xFF  # break the CRC
    good2 = build_frame(int(RecordType.IMU_MTA_QUAT), 3_000, struct.pack("<4f", 1, 0, 0, 0))

    parser = native.FrameParser()
    recs = parser.feed(bytes(good) + bytes(corrupt) + bytes(good2))

    # The good frames are recovered; the corrupt one is counted, not emitted.
    types = [r.record_type for r in recs]
    assert int(RecordType.EMG_BLOCK) in types
    assert int(RecordType.IMU_MTA_QUAT) in types
    assert parser.crc_errors >= 1


def test_drain_records_limit_leaves_the_remainder_queued():
    """Regression: the Python pump path silently discarded records.

    `_pump_native_python` drained the whole queue and then processed only
    `max_records`, so everything beyond that was lost -- it had already been
    popped. At ~3200 records/s the 128 cap is ~40 ms of data, which is why a
    488 s take showed 742 gaps and 2.95% of EMG missing while the native queue
    drop counters read zero.
    """
    from types import SimpleNamespace

    from jtsmm.capture.native_backend import NativeCapture

    class _Fake:
        def __init__(self, n):
            self._recs = [SimpleNamespace(record_type=0x10, t_master_us=i, decoded={})
                          for i in range(n)]

        def next_record(self, timeout_s=0.0):
            return self._recs.pop(0) if self._recs else None

    cap = NativeCapture.__new__(NativeCapture)
    cap._cap = _Fake(300)

    first = cap.drain_records(limit=128)
    assert len(first) == 128
    second = cap.drain_records(limit=128)
    assert len(second) == 128
    rest = cap.drain_records(limit=128)
    assert len(rest) == 44
    assert cap.drain_records(limit=128) == []
    # 300 in, 300 out -- nothing dropped on the floor.
    assert len(first) + len(second) + len(rest) == 300

    # limit=None keeps the old drain-everything behaviour for other callers.
    cap2 = NativeCapture.__new__(NativeCapture)
    cap2._cap = _Fake(300)
    assert len(cap2.drain_records()) == 300
