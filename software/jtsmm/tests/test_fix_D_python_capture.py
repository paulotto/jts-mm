# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pure-Python (test-only) MCU capture: errors surface instead of silently ending."""

from __future__ import annotations

import time

import pytest

from jtsmm.capture import capture_mcu
from jtsmm.capture.capture_mcu import McuCapture, McuCaptureError

from tests.helpers import build_frame


class _Reader:
    def __init__(self, chunks, fail_after: bool = False) -> None:
        self._chunks = list(chunks)
        self._fail_after = fail_after
        self.closed = False

    def open(self) -> None:
        pass

    def drain(self, duration_ms: int = 300) -> int:
        return 0

    def read(self, timeout_ms: int = 50) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._fail_after:
            raise OSError("[Errno 19] No such device")
        time.sleep(timeout_ms / 1000.0)
        return b""

    def close(self) -> None:
        self.closed = True


def _capture(monkeypatch, reader) -> McuCapture:
    monkeypatch.setattr(capture_mcu, "BulkReader", lambda: reader)
    return McuCapture(port=None)


def test_read_error_is_raised_from_iter_records(monkeypatch):
    frames = [build_frame(0x10, i, bytes(27)) for i in range(3)]
    cap = _capture(monkeypatch, _Reader([b"".join(frames)], fail_after=True))
    cap.start()
    got = []
    try:
        with pytest.raises(McuCaptureError, match="No such device"):
            for rec in cap.iter_records(timeout_s=2.0):
                got.append(rec)
    finally:
        cap.stop()
    assert len(got) == 3  # data before the failure is still delivered
    assert cap.stats.read_errors == 1 and isinstance(cap.error, OSError)


def test_raising_on_record_does_not_kill_the_parse_thread(monkeypatch):
    frames = [build_frame(0x10, i, bytes(27)) for i in range(5)]
    cap = _capture(monkeypatch, _Reader(frames))

    def boom(rec):
        raise ValueError("consumer bug")

    cap.on_record = boom
    cap.start()
    try:
        recs = list(cap.iter_records(timeout_s=1.0))
    finally:
        cap.stop()
    assert [r.t_master_us for r in recs] == [0, 1, 2, 3, 4]
    assert cap.stats.callback_errors == 5
    assert cap.error is None
