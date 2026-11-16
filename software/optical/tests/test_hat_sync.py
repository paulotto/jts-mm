# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for HAT sync bridge (no hardware)."""

from __future__ import annotations

import struct
import threading

from jtsmm.protocol import FrameParser, RecordType
from jtsmm.protocol.crc import crc16_ccitt
from jtsmm.sync.clock_align import ClockAligner

from optical.sync.frame_meta import CamFrameMeta
from optical.sync.hat_sync import HatSyncBridge


def _trigger_rec(trig_id: int, t_us: int):
    payload = b"\x00\x00\x00\x00" + struct.pack("<Q", trig_id)
    body = bytes([int(RecordType.TRIGGER_EVENT) & 0xFF, len(payload) & 0xFF])
    body += int(t_us).to_bytes(8, "little")
    body += payload
    crc = crc16_ccitt(body).to_bytes(2, "little")
    return FrameParser().feed(b"\x55\xaa" + body + crc)[0]


def test_align_frame_bundle_maps_master_clock():
    bridge = HatSyncBridge.__new__(HatSyncBridge)
    bridge._aligner = ClockAligner()
    bridge._lock = threading.Lock()
    bridge._aligner.ingest_mcu(_trigger_rec(1, 16_666))

    meta = {
        "cam0": CamFrameMeta(
            cam_id="cam0",
            sensor_timestamp_ns=16_666_000,
            t_host_monotonic_ns=1,
            frame_idx=1,
        )
    }
    t_cam_ns, t_master_us = bridge.align_frame_bundle(meta, frame_idx=0, cam_order=["cam0"])
    assert t_cam_ns["cam0"] == 16_666_000
    assert t_master_us == 16_666
