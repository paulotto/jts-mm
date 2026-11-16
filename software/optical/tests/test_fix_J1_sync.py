# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trigger rate / ordering and the trigger-only HAT sync bridge (no cv2)."""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

import optical.sync.hat_trigger as ht
from optical.sync.hat_sync import HatSyncBridge


@pytest.fixture(autouse=True)
def _reset():
    ht.reset_hat_trigger_state()
    yield
    ht.reset_hat_trigger_state()


# ------------------------------------------------------------ trigger rate


def test_default_rate_is_30_not_60():
    assert ht.DEFAULT_RATE_HZ == 30


def test_rate_read_from_jtsmm_config(tmp_path):
    cfg = tmp_path / "jtsmm.yaml"
    cfg.write_text("mcu:\n  trigger:\n    rate_hz: 45\n")
    assert ht.configured_trigger_rate_hz(cfg) == 45


def test_rate_falls_back_when_config_unreadable(tmp_path):
    assert ht.configured_trigger_rate_hz(tmp_path / "missing.yaml") == ht.DEFAULT_RATE_HZ
    bad = tmp_path / "bad.yaml"
    bad.write_text("mcu: {trigger: {rate_hz: 0}}\n")
    assert ht.configured_trigger_rate_hz(bad) == ht.DEFAULT_RATE_HZ


def test_shipped_jtsmm_yaml_rate_is_40():
    from jtsmm.config.loader import template_path

    # Study rate (Oct 2026 pilot); calibration keeps its own 30 Hz.
    assert ht.configured_trigger_rate_hz(template_path()) == 40
    assert ht.CALIBRATE_TRIGGER_RATE_HZ == 30


def test_hat_trigger_without_rate_uses_configured(monkeypatch):
    monkeypatch.setattr(ht, "configured_trigger_rate_hz", lambda *a, **k: 25)
    calls = []
    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=lambda *a: calls.append(a)):
        with ht.hat_trigger():
            assert ht.active_trigger_rate_hz() == 25
    assert calls == [("--rate", "25"), ("--stop",)]


# ------------------------------------------------------------ deferred arming


def test_deferred_trigger_arms_after_cameras_and_suppresses_nested_start():
    calls = []
    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=lambda *a: calls.append(a)):
        with ht.hat_trigger(rate_hz=30, defer=True):
            assert calls == []
            # PicamRig enters hat_trigger() while opening cameras: must not pulse yet.
            with ht.hat_trigger():
                assert calls == []
            assert ht.arm_hat_trigger() is True
            assert ht.arm_hat_trigger() is False  # idempotent
            assert calls == [("--rate", "30")]
        assert calls == [("--rate", "30"), ("--stop",)]


def test_deferred_never_armed_still_sends_stop():
    calls = []
    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=lambda *a: calls.append(a)):
        with pytest.raises(RuntimeError, match="camera"):
            with ht.hat_trigger(rate_hz=30, defer=True):
                raise RuntimeError("camera open failed")
    assert calls == [("--stop",)]


def test_arm_outside_context_raises():
    with pytest.raises(RuntimeError):
        ht.arm_hat_trigger()


# ------------------------------------------------------------ HatSyncBridge


class _FakeMcu:
    def __init__(self, port):
        self.port = port
        self.on_record = None
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False


def test_hat_sync_start_opens_ingest_only(monkeypatch):
    import jtsmm.capture.capture_mcu as cm
    import jtsmm.capture.serial_util as su
    import jtsmm.config as jc

    monkeypatch.setattr(cm, "McuCapture", _FakeMcu)
    monkeypatch.setattr(su, "find_hat_port", lambda p: "/dev/fake")

    def _boom(*a, **k):
        raise AssertionError("HatSyncBridge must not start MCU streams")

    monkeypatch.setattr(jc, "start_mcu_streams", _boom)
    monkeypatch.setattr(jc, "open_hat_serial", _boom)
    monkeypatch.setattr(jc, "stop_mcu_streams_robust", _boom)
    bridge = HatSyncBridge()
    assert bridge.start() == "/dev/fake"
    assert bridge._mcu.started
    assert bridge.reset_trigger_backlog() == 0
    bridge.stop()


def _trigger_rec(trig_id: int, t_us: int):
    import struct

    from jtsmm.protocol import FrameParser, RecordType
    from jtsmm.protocol.crc import crc16_ccitt

    payload = b"\x00\x00\x00\x00" + struct.pack("<Q", trig_id)
    body = bytes([int(RecordType.TRIGGER_EVENT) & 0xFF, len(payload) & 0xFF])
    body += int(t_us).to_bytes(8, "little") + payload
    return FrameParser().feed(b"\x55\xaa" + body + crc16_ccitt(body).to_bytes(2, "little"))[0]


def test_reset_backlog_drops_pre_camera_pulses():
    from jtsmm.sync.clock_align import ClockAligner

    bridge = HatSyncBridge.__new__(HatSyncBridge)
    bridge._aligner = ClockAligner()
    bridge._lock = threading.Lock()
    bridge.bundle_trigger_id_mismatch = 0
    for i in range(1, 6):
        bridge._aligner.ingest_mcu(_trigger_rec(i, i * 33_333))
    assert bridge.reset_trigger_backlog() == 5
    report = bridge.sync_report()
    assert report["discarded_trigger_backlog"] == 5
    assert report["bundle_trigger_id_mismatch"] == 0
