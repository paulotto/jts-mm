# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""USB vendor-bulk MCU reader — framed record dispatch (pure-Python fallback).

Binary records arrive on the vendor bulk IN endpoint (see ``usb_bulk``); text and
commands live on the CDC serial port. This reader therefore uses libusb, not
pyserial.

Test / diagnostic only: recording, the viewer and the doctor use the native
capture (``jtsmm.native`` via ``capture.native_backend``). This path is kept for
unit tests and ``run_capture_loop``; it is not fast enough for a full-rate take.
Failures are surfaced, not swallowed: a USB read error ends the capture and is
re-raised from ``iter_records``; losses are counted in ``LinkStats``.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Full, Queue
from typing import Callable, Iterator

from jtsmm.protocol import FrameParser, McuRecord

from .usb_bulk import BulkReader


@dataclass
class LinkStats:
    bytes_rx: int = 0
    records_rx: int = 0
    crc_errors: int = 0
    resyncs: int = 0
    read_errors: int = 0  # USB read raised; the capture stopped (see McuCapture.error)
    raw_drops: int = 0  # raw chunks dropped: parser stalled past a full queue
    record_drops: int = 0  # parsed records dropped: consumer fell behind
    callback_errors: int = 0  # on_record raised (record still queued)
    started_monotonic: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict[str, float]:
        elapsed = max(time.monotonic() - self.started_monotonic, 1e-6)
        return {
            "bytes_per_s": self.bytes_rx / elapsed,
            "records_per_s": self.records_rx / elapsed,
            "crc_errors": float(self.crc_errors),
            "resyncs": float(self.resyncs),
        }


class McuCaptureError(RuntimeError):
    """The USB read loop failed; raised from ``iter_records`` once drained."""


class McuCapture:
    """Background bulk reader pushing parsed McuRecord into a queue."""

    def __init__(
        self,
        port: str | None = None,
        *,
        baud: int = 115200,
        queue_max: int = 16384,
    ) -> None:
        # ``port`` / ``baud`` are retained for API compatibility (callers still
        # pass the CDC port), but the binary stream is read over the vendor bulk
        # endpoint, so they are not used here.
        self.port = port
        self.baud = baud
        self._parser = FrameParser()
        self._queue: Queue[McuRecord] = Queue(maxsize=queue_max)
        # Raw byte chunks straight off the wire. The read thread only touches this
        # queue so it can immediately re-arm the next USB transfer; parsing (which
        # is comparatively slow in Python) happens on a separate thread. Without
        # this split the endpoint sits unpolled during parsing and the MCU vendor
        # FIFO overflows (dropped frames -> correlated EMG/IMU rate shortfalls).
        self._raw: Queue[bytes] = Queue(maxsize=queue_max)
        self._stats = LinkStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._parse_thread: threading.Thread | None = None
        self._reader: BulkReader | None = None
        self.on_record: Callable[[McuRecord], None] | None = None
        self.error: BaseException | None = None  # read-loop failure, if any

    @property
    def stats(self) -> LinkStats:
        return self._stats

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._reader = BulkReader()
        self._reader.open()
        # Discard stale buffered records so the session starts at "now".
        self._reader.drain()
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._read_loop, name="mcu-read", daemon=True)
        self._parse_thread = threading.Thread(
            target=self._parse_loop, name="mcu-parse", daemon=True
        )
        self._thread.start()
        self._parse_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._parse_thread:
            self._parse_thread.join(timeout=2.0)
        if self._reader:
            self._reader.close()
            self._reader = None

    def _read_loop(self) -> None:
        """Tight loop: pull bytes off the wire and hand them to the parser thread.

        Keeps this loop free of any per-record work so the next bulk transfer is
        submitted immediately, preventing MCU FIFO overflow.
        """
        assert self._reader is not None
        reader = self._reader
        while not self._stop.is_set():
            try:
                chunk = reader.read(timeout_ms=50)
            except Exception as exc:  # noqa: BLE001 - any USB failure ends the take
                # Used to `break` silently: the capture just stopped producing.
                self.error = exc
                self._stats.read_errors += 1
                print(f"warning: MCU bulk read failed, capture stopped: {exc}", file=sys.stderr)
                self._stop.set()
                break
            if not chunk:
                continue
            self._stats.bytes_rx += len(chunk)
            try:
                self._raw.put_nowait(chunk)
            except Full:
                # Parser is falling behind; block briefly rather than lose data,
                # but never forever (a dead parse thread used to hang this loop).
                try:
                    self._raw.put(chunk, timeout=1.0)
                except Full:
                    self._stats.raw_drops += 1

    def _parse_loop(self) -> None:
        while not (self._stop.is_set() and self._raw.empty()):
            try:
                chunk = self._raw.get(timeout=0.1)
            except Empty:
                continue
            for rec in self._parser.feed(chunk):
                self._stats.records_rx += 1
                self._stats.crc_errors = self._parser.crc_errors
                self._stats.resyncs = self._parser.resyncs
                if self.on_record:
                    # A raising callback used to kill this thread, after which the
                    # read thread blocked forever on the full raw queue.
                    try:
                        self.on_record(rec)
                    except Exception as exc:  # noqa: BLE001
                        self._stats.callback_errors += 1
                        if self._stats.callback_errors == 1:
                            print(f"warning: MCU on_record callback raised: {exc!r}",
                                  file=sys.stderr)
                try:
                    self._queue.put_nowait(rec)
                except Full:
                    try:
                        self._queue.get_nowait()
                        self._stats.record_drops += 1
                    except Empty:
                        pass
                    self._queue.put_nowait(rec)

    def iter_records(self, timeout_s: float = 0.5) -> Iterator[McuRecord]:
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while True:
            if self._stop.is_set() and self._queue.empty():
                if self.error is not None:
                    raise McuCaptureError(f"MCU bulk read failed: {self.error}") from self.error
                return
            try:
                wait = 0.1
                if deadline is not None:
                    wait = max(0.0, min(wait, deadline - time.monotonic()))
                    if wait == 0.0 and self._queue.empty():
                        return
                rec = self._queue.get(timeout=wait)
                yield rec
            except Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                continue


def run_capture_loop(
    port: str | None,
    handler: Callable[[McuRecord], None],
    *,
    duration_s: float | None = None,
) -> LinkStats:
    """Blocking capture for duration_s (None = until KeyboardInterrupt)."""
    cap = McuCapture(port=port)
    cap.on_record = handler
    cap.start()
    try:
        if duration_s is None:
            while True:
                time.sleep(0.2)
        else:
            time.sleep(duration_s)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
    return cap.stats
