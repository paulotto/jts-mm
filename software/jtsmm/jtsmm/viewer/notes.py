# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Operator notes typed into the viewer, stamped with the host clock on arrival.

The browser only knows its own display time (``t_s``, seconds since the first
WebSocket frame), which is not on any recorded clock. So every note is stamped
with ``time.monotonic_ns()`` when the server receives it -- the same host
domain the recorder uses for phase/event ``t_host_ns`` -- and the recorder maps
that to ``t_master_us`` like its other events (``SessionSupervisor``).

`NoteLog` is thread-safe: the web server adds notes from its event loop, the
recorder's main loop ``drain()``s the ones it has not taken yet. Standalone
(``jtsmm view`` / ``jtsmm replay``) nobody drains and ``saved_to`` stays None:
notes then live in server memory only.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

__all__ = ["MAX_NOTE_CHARS", "NoteLog"]

MAX_NOTE_CHARS = 2000


class NoteLog:
    def __init__(self, clock_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock_ns = clock_ns
        self._lock = threading.Lock()
        self._notes: list[dict[str, Any]] = []
        self._drained = 0
        # Set by the recorder to the session file name the notes are written to.
        self.saved_to: str | None = None

    def add(self, text: Any, t_s: Any = None) -> dict[str, Any]:
        """Stamp and store one note; raises ``ValueError`` on empty text / bad t_s."""
        t_host_ns = int(self._clock_ns())  # first: closest to the arrival
        # NUL cannot be stored in an HDF5 C string; newlines are not wanted in
        # a one-line note either.
        clean = " ".join(str(text if text is not None else "").replace("\x00", " ").split())
        if not clean:
            raise ValueError("empty note")
        clean = clean[:MAX_NOTE_CHARS]
        view_t: float | None = None
        if t_s is not None:
            view_t = float(t_s)  # ValueError/TypeError propagate to the caller
            if not math.isfinite(view_t):
                view_t = None  # JSON (the /api/session echo) cannot carry NaN/inf
        note = {"t_host_ns": t_host_ns, "t_s": view_t, "text": clean}
        with self._lock:
            self._notes.append(note)
        return dict(note)

    def recent(self, n: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(x) for x in self._notes[-n:]] if n > 0 else []

    def drain(self) -> list[dict[str, Any]]:
        """Notes added since the previous drain, in arrival order."""
        with self._lock:
            new = [dict(x) for x in self._notes[self._drained:]]
            self._drained = len(self._notes)
        return new

    def __len__(self) -> int:
        with self._lock:
            return len(self._notes)
