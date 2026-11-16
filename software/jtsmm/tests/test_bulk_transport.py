# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Bulk transport + REC_STATS decode coverage.

These tests exercise the host binary path without hardware: the ``0x55AA`` frame
parser, the full ``REC_STATS`` payload (14 counters incl. ``tx_drops``), and the
``McuCapture`` read/parse pipeline against a fake ``BulkReader``.
"""

from __future__ import annotations

import struct
import time

import pytest

from jtsmm.capture import capture_mcu
from jtsmm.capture.capture_mcu import McuCapture
from jtsmm.protocol import FrameParser
from jtsmm.protocol.records import STATS_FIELDS, decode_record
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame


def _stats_payload(**overrides: int) -> bytes:
    """Encode all 14 STATS counters, mirroring firmware core1_send_stats()."""
    values = {name: i + 1 for i, name in enumerate(STATS_FIELDS)}
    values.update(overrides)
    return struct.pack(f"<{len(STATS_FIELDS)}I", *(values[n] for n in STATS_FIELDS))


def test_stats_decode_includes_tx_drops():
    payload = _stats_payload(missed_drdy=7, emg_drops=3, tx_drops=99)
    decoded = decode_record(int(RecordType.STATS), payload)
    assert set(decoded) == set(STATS_FIELDS)
    assert decoded["missed_drdy"] == 7
    assert decoded["emg_drops"] == 3
    assert decoded["tx_drops"] == 99


def test_stats_frame_roundtrip():
    payload = _stats_payload(tx_drops=1234)
    frame = build_frame(int(RecordType.STATS), 555_000, payload)
    recs = FrameParser().feed(frame)
    assert len(recs) == 1
    assert recs[0].record_type == int(RecordType.STATS)
    assert recs[0].t_master_us == 555_000
    assert recs[0].decoded["tx_drops"] == 1234


class _FakeBulkReader:
    """Feeds pre-built byte chunks in place of the libusb vendor endpoint."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def drain(self, duration_ms: int = 300) -> int:
        return 0

    def read(self, timeout_ms: int = 50) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        time.sleep(timeout_ms / 1000.0)
        return b""

    def close(self) -> None:
        self.closed = True


def _collect(chunks: list[bytes], monkeypatch) -> list:
    reader = _FakeBulkReader(chunks)
    monkeypatch.setattr(capture_mcu, "BulkReader", lambda: reader)
    cap = McuCapture(port=None)
    cap.start()
    try:
        recs = list(cap.iter_records(timeout_s=1.0))
    finally:
        cap.stop()
    assert reader.opened and reader.closed
    assert cap.stats.crc_errors == 0
    return recs


def test_mcu_capture_mock_transport(monkeypatch):
    emg = build_frame(int(RecordType.EMG_BLOCK), 1_000, bytes(27))
    quat = build_frame(int(RecordType.IMU_MTA_QUAT), 2_000, struct.pack("<4f", 1.0, 0.0, 0.0, 0.0))
    stats = build_frame(int(RecordType.STATS), 3_000, _stats_payload(tx_drops=5))

    recs = _collect([emg + quat + stats], monkeypatch)

    by_type = {r.record_type: r for r in recs}
    assert int(RecordType.EMG_BLOCK) in by_type
    assert int(RecordType.IMU_MTA_QUAT) in by_type
    assert by_type[int(RecordType.STATS)].decoded["tx_drops"] == 5
    assert by_type[int(RecordType.IMU_MTA_QUAT)].decoded["quat"][0] == pytest.approx(1.0)


def test_mcu_capture_reassembles_split_frame(monkeypatch):
    frame = build_frame(int(RecordType.TRIGGER_EVENT), 9_000, struct.pack("<4BQ", 1, 0, 0, 0, 42))
    # Split mid-frame across two USB chunks; the parser must reassemble.
    recs = _collect([frame[:5], frame[5:]], monkeypatch)
    assert len(recs) == 1
    assert recs[0].record_type == int(RecordType.TRIGGER_EVENT)
    assert recs[0].decoded["trigger_id"] == 42
