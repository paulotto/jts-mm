# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Background video renders for the Animation page (one local user, one render at a time).

The same path as ``biomech animate``: :func:`biomech.anim.make` builds the animation,
:func:`biomech.anim.render_video.render_video` writes it in a thread, frame progress is
collected for the progress bar, and Cancel stops the render and removes partial files.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VideoState:
    running: bool = False
    done: int = 0
    total: int = 0
    started: float | None = None
    finished: float | None = None
    path: str | None = None
    error: str | None = None
    cancelled: bool = False


class VideoJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = VideoState()

    def start(self, data_ctx: Any, out: Path, make_kw: dict[str, Any], size: str = "720p",
              n_jobs: int | None = None) -> bool:
        with self._lock:
            if self.state.running:
                return False
            self._cancel.clear()
            self.state = VideoState(running=True, started=time.time(), path=str(out))

        def progress(done: int, total: int) -> None:
            with self._lock:
                self.state.done, self.state.total = done, total

        def work() -> None:
            from biomech.anim import make
            from biomech.anim.render_video import Cancelled, render_video

            try:
                anim = make(data_ctx, **make_kw)
                with self._lock:
                    self.state.total = anim.timeline.n_frames
                render_video(anim, out, size=size, n_jobs=n_jobs, on_progress=progress,
                             cancel=self._cancel.is_set)
            except Cancelled:
                with self._lock:
                    self.state.cancelled = True
            except Exception:
                with self._lock:
                    self.state.error = traceback.format_exc(limit=4)
            finally:
                with self._lock:
                    self.state.running = False
                    self.state.finished = time.time()

        self._thread = threading.Thread(target=work, daemon=True, name="biomech-video")
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def snapshot(self) -> VideoState:
        with self._lock:
            return VideoState(**self.state.__dict__)
