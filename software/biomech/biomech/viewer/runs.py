# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Background analysis runs for the control center (one local user, one run at a time).

A thin wrapper around :func:`biomech.pipeline.run`: the run executes in a
thread, events are collected for the progress display, and Cancel sets the
same ``threading.Event`` the CLI path would use.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from biomech.config import StudyConfig
from biomech.pipeline import run


@dataclass
class RunState:
    running: bool = False
    started: float | None = None
    finished: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    error: str | None = None
    cancelled: bool = False
    stages: list[str] = field(default_factory=list)


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = RunState()

    def start(self, config: StudyConfig, stages: list[str] | None, pids: list[str] | None,
              force: bool = False, n_jobs: int = 4) -> bool:
        with self._lock:
            if self.state.running:
                return False
            self._cancel.clear()
            self.state = RunState(running=True, started=time.time(), stages=stages or [])

        def sink(ev: dict[str, Any]) -> None:
            with self._lock:
                self.state.events.append(ev)
                if ev.get("kind") == "run_start":
                    self.state.run_id = ev.get("run_id")

        def work() -> None:
            try:
                s = run(config, stages or None, pids or None, force=force, n_jobs=n_jobs,
                        on_event=sink, cancel=self._cancel)
                with self._lock:
                    self.state.cancelled = s.cancelled
            except Exception:
                with self._lock:
                    self.state.error = traceback.format_exc()
            finally:
                with self._lock:
                    self.state.running = False
                    self.state.finished = time.time()

        self._thread = threading.Thread(target=work, daemon=True, name="biomech-run")
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def snapshot(self) -> RunState:
        with self._lock:
            return RunState(**{**self.state.__dict__, "events": list(self.state.events)})

    def progress(self) -> tuple[int, int]:
        """(units done, units planned) from the event stream."""
        s = self.snapshot()
        total = sum(e.get("n_todo", 0) for e in s.events if e["kind"] == "stage_start")
        done = sum(1 for e in s.events if e["kind"] == "unit_done")
        return done, total


def describe_event(ev: dict[str, Any], label: Any = None) -> str:
    """One log line; ``label`` maps a unit to its shown name (the Overview: pseudonyms)."""
    k = ev["kind"]
    unit = (label or str)(ev["unit"]) if "unit" in ev else ""
    t = time.strftime("%H:%M:%S", time.localtime(ev.get("t", time.time())))
    if k == "stage_start":
        return f"{t}  [{ev['stage']}] {ev['n_todo']} to run, {ev['n_cached']} cached"
    if k == "unit_done":
        err = f"  {ev['error']}" if ev.get("error") else ""
        return f"{t}    {ev['stage']}/{unit}: {ev['status']} ({ev['elapsed_s']:.1f} s){err}"
    if k == "unit_skipped":
        return f"{t}    {ev['stage']}/{unit}: skipped ({ev['reason']})"
    if k in ("run_start", "run_end", "cancelled"):
        return f"{t}  {k.replace('_', ' ')}" + (f" ({ev.get('n_failed', 0)} failed)"
                                               if k != "run_start" else "")
    return f"{t}  {k}"
