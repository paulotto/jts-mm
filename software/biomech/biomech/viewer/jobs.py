# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Background jobs of the viewer (long exports), like :mod:`biomech.viewer.runs`.

One job at a time per :class:`Job`: the work runs in a daemon thread, reports progress
through ``progress(done, total, label)`` and polls ``cancelled()`` between items; a second
start while a job runs is refused (a double click never starts two exports). The page polls
:meth:`Job.snapshot` with a ``dcc.Interval``.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

Progress = Callable[[int, int, str], None]


@dataclass
class JobState:
    name: str = ""
    running: bool = False
    done: int = 0
    total: int = 0
    label: str = ""
    started: float | None = None
    finished: float | None = None
    message: str = ""
    error: str | None = None
    cancelled: bool = False


class Job:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = JobState()

    def start(self, name: str, work: Callable[[Progress, Callable[[], bool]], str]) -> bool:
        """Run ``work(progress, cancelled) -> message`` in a thread; False if one is running."""
        with self._lock:
            if self.state.running:
                return False
            self._cancel.clear()
            self.state = JobState(name=name, running=True, started=time.time())

        def progress(done: int, total: int, label: str = "") -> None:
            with self._lock:
                self.state.done, self.state.total, self.state.label = done, total, label

        def run() -> None:
            try:
                msg = work(progress, self._cancel.is_set)
                with self._lock:
                    self.state.message = msg or ""
                    self.state.cancelled = self._cancel.is_set()
            except Exception:
                with self._lock:
                    self.state.error = traceback.format_exc(limit=4)
            finally:
                with self._lock:
                    self.state.running = False
                    self.state.finished = time.time()

        self._thread = threading.Thread(target=run, daemon=True, name=f"biomech-{name}")
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> JobState:
        with self._lock:
            return JobState(**self.state.__dict__)

    def describe(self) -> dict[str, Any]:
        """Progress line for the page: text, value, max, running."""
        s = self.snapshot()
        if s.started is None:
            return {"text": "", "value": 0, "max": 1, "running": False}
        if s.running:
            text = f"{s.name}: {s.done}/{s.total or '?'}" + (f" · {s.label}" if s.label else "")
            if self._cancel.is_set():
                text += " (cancelling after this item)"
        elif s.error:
            text = f"{s.name} failed: {s.error.strip().splitlines()[-1]}"
        else:
            took = f" ({s.finished - s.started:.0f} s)" if s.finished else ""
            text = (f"{s.name} cancelled after {s.done} of {s.total} items" if s.cancelled
                    else f"{s.name} done{took}") + (f": {s.message}" if s.message else "")
        return {"text": text, "value": s.done, "max": max(s.total, 1), "running": s.running}
