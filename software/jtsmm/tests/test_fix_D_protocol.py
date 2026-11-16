# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Wire-contract decode (Python side): EMG seq (K1), STATS 36 fields (K2), BUTTON (K7)."""

from __future__ import annotations

import struct

from jtsmm.protocol import FrameParser
from jtsmm.protocol.records import STATS_FIELDS, decode_record
from jtsmm.protocol.types import RecordType

from tests.helpers import build_frame


def test_emg_block_accepts_27_and_31_bytes():
    base = bytes([0xC0, 0, 0]) + bytes(24)
    d27 = decode_record(int(RecordType.EMG_BLOCK), base)
    assert "seq" not in d27 and len(d27["channels"]) == 8
    d31 = decode_record(int(RecordType.EMG_BLOCK), base + struct.pack("<I", 123456))
    assert d31["seq"] == 123456
    assert d31["channels"] == d27["channels"] and d31["status"] == d27["status"]


def test_emg_block_other_lengths_fall_back_to_raw():
    for n in (26, 28, 30, 32):
        recs = FrameParser().feed(build_frame(int(RecordType.EMG_BLOCK), 1, bytes(n)))
        assert recs[0].decoded == {"raw": bytes(n)}


def test_stats_decode_by_count_old_and_new_firmware():
    assert len(STATS_FIELDS) == 36
    assert STATS_FIELDS[28:] == (
        "tx_drops_emg", "tx_drops_imu", "tx_drops_other", "imu_mta_seq_gaps",
        "imu_cra_seq_gaps", "imu_mta_resets", "imu_cra_resets", "stats_seq",
    )
    old = decode_record(int(RecordType.STATS), struct.pack("<28I", *range(28)))
    assert len(old) == 28 and "stats_seq" not in old
    new = decode_record(int(RecordType.STATS), struct.pack("<36I", *range(36)))
    assert new["tx_drops_emg"] == 28 and new["stats_seq"] == 35


def test_button_event_record():
    recs = FrameParser().feed(build_frame(0x22, 77, bytes([1, 1, 0, 0, 0])))
    assert recs[0].type_name == "BUTTON_EVENT"
    assert recs[0].decoded == {"button_id": 1, "pressed": True}
