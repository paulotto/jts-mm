# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Decode record payloads from golden byte blobs."""

from __future__ import annotations

import struct

import pytest

from jtsmm.protocol.records import decode_record
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame, load_golden_vectors
from jtsmm.protocol import FrameParser


@pytest.mark.parametrize("entry", load_golden_vectors()["frames"], ids=lambda e: e["name"])
def test_golden_frame_decode(entry):
    payload = bytes.fromhex(entry["payload_hex"])
    frame = build_frame(entry["type"], entry["t_master_us"], payload)
    recs = FrameParser().feed(frame)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.record_type == entry["type"]
    assert rec.t_master_us == entry["t_master_us"]

    if "expected_channels" in entry:
        assert rec.decoded["channels"] == entry["expected_channels"]
    if "expected_quat" in entry:
        for got, exp in zip(rec.decoded["quat"], entry["expected_quat"]):
            assert got == pytest.approx(exp)
    if "expected_trigger_id" in entry:
        assert rec.decoded["trigger_id"] == entry["expected_trigger_id"]
    if "expected_raw_len" in entry:
        assert len(rec.payload) == entry["expected_raw_len"]
    if "expected_quat_report_id" in entry:
        assert rec.decoded["quat_report_id"] == entry["expected_quat_report_id"]
    if "expected_fusion" in entry:
        assert rec.decoded["fusion"] == entry["expected_fusion"]
    if "expected_sensor_id" in entry:
        assert rec.decoded["sensor_id"] == entry["expected_sensor_id"]
    if "expected_accel" in entry:
        assert list(rec.decoded["accel"]) == entry["expected_accel"]
    if "expected_gyro" in entry:
        assert list(rec.decoded["gyro"]) == entry["expected_gyro"]


def test_emg_sign_extend():
    payload = bytearray(27)
    raw = 0x800001  # negative 24-bit (MSB-first)
    payload[3] = (raw >> 16) & 0xFF
    payload[4] = (raw >> 8) & 0xFF
    payload[5] = raw & 0xFF
    decoded = decode_record(int(RecordType.EMG_BLOCK), bytes(payload))
    assert decoded["channels"][0] < 0


def test_emg_big_endian_channel_order():
    """Regression: host must match firmware RDATA byte order (MSB first)."""
    payload = bytearray(27)
    # bytes fffdcb on CH4 → -565 counts, not millions.
    payload[3 + 3 * 3 + 0] = 0xFF
    payload[3 + 3 * 3 + 1] = 0xFD
    payload[3 + 3 * 3 + 2] = 0xCB
    decoded = decode_record(int(RecordType.EMG_BLOCK), bytes(payload))
    assert decoded["channels"][3] == -565


def test_imu_raw_decode_legacy_18b():
    payload = struct.pack("<9h", *range(9))
    decoded = decode_record(int(RecordType.IMU_MTA_RAW), payload)
    assert decoded["accel"] == (0, 1, 2)
    assert decoded["gyro"] == (3, 4, 5)
    assert decoded["mag"] == (6, 7, 8)


def test_imu_raw_decode_tagged_7b():
    for sensor_id, key in ((0x01, "accel"), (0x02, "gyro"), (0x03, "mag")):
        payload = bytes([sensor_id]) + struct.pack("<3h", -1, 2, -3)
        decoded = decode_record(int(RecordType.IMU_MTA_RAW), payload)
        assert decoded["sensor_id"] == sensor_id
        assert decoded[key] == (-1, 2, -3)


def test_imu_quat_decode_tagged_17b():
    payload = bytes([0x09]) + struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    decoded = decode_record(int(RecordType.IMU_CRA_QUAT), payload)
    assert decoded["quat"] == (1.0, 0.0, 0.0, 0.0)
    assert decoded["quat_report_id"] == 0x09
    assert decoded["fusion"] == "geomagnetic_rotation_vector"


def test_imu_quat_decode_legacy_16b():
    payload = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
    decoded = decode_record(int(RecordType.IMU_MTA_QUAT), payload)
    assert decoded["quat"] == (1.0, 0.0, 0.0, 0.0)
    assert decoded["quat_report_id"] is None


def test_cmd_ack_decode():
    assert decode_record(int(RecordType.CMD_ACK), b"\x01")["ok"] is True
    assert decode_record(int(RecordType.CMD_ACK), b"\x00")["ok"] is False


def test_stats_decode():
    values = list(range(13))
    payload = struct.pack("<13I", *values)
    decoded = decode_record(int(RecordType.STATS), payload)
    assert decoded["drdy_count"] == 0
    assert decoded["missed_drdy"] == 1
    assert decoded["emg_push"] == 2
    assert decoded["emg_drops"] == 3
    assert decoded["usb_tx_bytes"] == 10
    assert decoded["sensor_loop_hz"] == 11
    assert decoded["comms_loop_hz"] == 12


def test_tagged_records_decode_accuracy():
    from jtsmm.protocol.records import decode_lin_accel, decode_quat, decode_raw_imu

    q_payload = bytes([0x08]) + struct.pack("<4f", 1.0, 0.0, 0.0, 0.0) + bytes([0x03])
    q = decode_quat(q_payload)
    assert q["quat_accuracy"] == 3

    raw = bytes([0x02]) + struct.pack("<3h", 100, 0, 0) + bytes([0x02])
    r = decode_raw_imu(raw)
    assert r["gyro_accuracy"] == 2

    lin = struct.pack("<3f", 0.1, 0.0, 0.2) + bytes([0x01])
    l = decode_lin_accel(lin)
    assert l["linear_accel_accuracy"] == 1


def test_decode_trigger_stamp_source_flags():
    """flags[0] tags how the firmware timed the event; flags[1] the watchdog fallback."""
    from jtsmm.protocol.records import (
        TRIGGER_STAMP_PIO_IRQ,
        TRIGGER_STAMP_TIMER,
        decode_trigger,
    )

    tid = (7).to_bytes(8, "little")
    legacy = decode_trigger(bytes([0, 0, 0, 0]) + tid)
    assert legacy["trigger_id"] == 7
    assert legacy["stamp_source"] == "unknown"
    assert legacy["stamp_fallback"] is False

    edge = decode_trigger(bytes([TRIGGER_STAMP_PIO_IRQ, 0, 0, 0]) + tid)
    assert edge["stamp_source"] == "pio_irq"
    assert edge["stamp_fallback"] is False

    fallback = decode_trigger(bytes([TRIGGER_STAMP_TIMER, 1, 0, 0]) + tid)
    assert fallback["stamp_source"] == "timer"
    assert fallback["stamp_fallback"] is True
