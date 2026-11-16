# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""SPACE-controlled experiment phase walker for ``jtsmm record``.

Capture keeps running continuously; SPACE only stamps phase boundaries onto
``t_master_us`` for post-processing / JTS config generation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


StampFn = Callable[[], tuple[int, int, str]]
"""Return ``(t_host_ns, t_master_us, map_quality)`` for a SPACE edge."""


@dataclass
class PhaseEdgeEvent:
    """One phase start/end stamped onto the MCU master clock."""

    name: str
    kind: str
    edge: str  # "start" | "end"
    t_host_ns: int
    t_master_us: int
    map_quality: str


@dataclass
class TimedPhase:
    """Completed phase interval on the master clock."""

    name: str
    kind: str
    t_start_us: int
    t_end_us: int
    map_quality_start: str
    map_quality_end: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "t_start_us": int(self.t_start_us),
            "t_end_us": int(self.t_end_us),
            "map_quality_start": self.map_quality_start,
            "map_quality_end": self.map_quality_end,
        }


@dataclass
class PhaseWalker:
    """Walk calibration + motion phases; SPACE starts/ends each one."""

    phases: Sequence[Any]  # optical.tracking.experiment.Phase
    stamp_fn: StampFn
    events: list[PhaseEdgeEvent] = field(default_factory=list)
    recorded: list[TimedPhase] = field(default_factory=list)
    phase_index: int = 0
    active: bool = False
    finished: bool = False
    aborted: bool = False
    _t_start_us: int | None = None
    _q_start: str | None = None

    @classmethod
    def from_plan(cls, plan: Any, stamp_fn: StampFn) -> PhaseWalker:
        return cls(phases=list(plan.all_phases), stamp_fn=stamp_fn)

    @property
    def current(self) -> Any | None:
        if self.phase_index >= len(self.phases):
            return None
        return self.phases[self.phase_index]

    def status_line(self) -> str:
        if self.finished:
            return "phases done"
        cur = self.current
        if cur is None:
            return "phases done"
        n = len(self.phases)
        i = self.phase_index + 1
        extra = " EMG only" if getattr(cur, "kind", "") == "mvc" else f" ({cur.kind})"
        hint = getattr(cur, "hint", None)
        if hint:
            extra += f" → {hint}"
        if self.active:
            return f"Phase {i}/{n}: '{cur.name}'{extra} RECORDING — SPACE to end"
        return f"Phase {i}/{n}: '{cur.name}'{extra} — SPACE to start"

    def is_emg_only_slot(self) -> bool:
        """True while the current phase is MVC (waiting or recording)."""
        if self.finished:
            return False
        cur = self.current
        return cur is not None and getattr(cur, "kind", "") == "mvc"

    def is_emg_only_active(self) -> bool:
        """True while an MVC phase is recording (ArUco / preview paused)."""
        return bool(self.active and self.is_emg_only_slot())

    def print_prompt(self, *, file=sys.stderr) -> None:
        print(self.status_line(), file=file)

    def poll(self, key: int | None) -> list[PhaseEdgeEvent]:
        """Consume one keypress; return any new edge events (0–1 typically)."""
        if self.finished or key is None:
            return []
        if key in (27, ord("q")):
            return self._finish_early()
        if key != ord(" "):
            return []
        if not self.active:
            return [self._start()]
        ev = self._end()
        out = [ev]
        if self.phase_index >= len(self.phases):
            self.finished = True
            print("All phases complete.", file=sys.stderr)
        else:
            self.print_prompt()
        return out

    def _stamp(self) -> tuple[int, int, str]:
        return self.stamp_fn()

    def _start(self) -> PhaseEdgeEvent:
        cur = self.current
        assert cur is not None
        t_host, t_master, quality = self._stamp()
        self.active = True
        self._t_start_us = int(t_master)
        self._q_start = quality
        ev = PhaseEdgeEvent(
            name=cur.name,
            kind=cur.kind,
            edge="start",
            t_host_ns=int(t_host),
            t_master_us=int(t_master),
            map_quality=quality,
        )
        self.events.append(ev)
        print(
            f"  ▶ '{cur.name}' started @ t_master_us={t_master} ({quality}) — SPACE to end"
            + (" [EMG only: cameras/optical paused]" if cur.kind == "mvc" else ""),
            file=sys.stderr,
        )
        return ev

    def _end(self) -> PhaseEdgeEvent:
        cur = self.current
        assert cur is not None and self._t_start_us is not None and self._q_start is not None
        t_host, t_master, quality = self._stamp()
        # Ensure non-empty interval even if SPACE was double-tapped quickly.
        t_end = max(int(t_master), int(self._t_start_us) + 1)
        self.recorded.append(
            TimedPhase(
                name=cur.name,
                kind=cur.kind,
                t_start_us=int(self._t_start_us),
                t_end_us=t_end,
                map_quality_start=self._q_start,
                map_quality_end=quality,
            )
        )
        ev = PhaseEdgeEvent(
            name=cur.name,
            kind=cur.kind,
            edge="end",
            t_host_ns=int(t_host),
            t_master_us=t_end,
            map_quality=quality,
        )
        self.events.append(ev)
        print(
            f"  ■ '{cur.name}' ended @ t_master_us={t_end} "
            f"[{self._t_start_us}, {t_end}) ({quality})",
            file=sys.stderr,
        )
        self.active = False
        self._t_start_us = None
        self._q_start = None
        self.phase_index += 1
        return ev

    def _finish_early(self) -> list[PhaseEdgeEvent]:
        out: list[PhaseEdgeEvent] = []
        if self.active:
            out.append(self._end())
        self.finished = True
        self.aborted = True
        print("Finishing early — remaining phases skipped.", file=sys.stderr)
        return out


def optical_frame_intervals(
    timed: Sequence[TimedPhase],
    t_optical_us: Sequence[int] | Any,
    *,
    period_us: float | None = None,
) -> list[tuple[str, str, int, int]]:
    """Map master-time phases ``[t_start, t_end)`` to JTS frame intervals.

    Frame indices are on the **trigger grid**, not optical row numbers:
    dropped frames are absent rows, so frame ``k`` is the trigger at
    ``t0 + k * period`` with ``t0`` the first optical row
    (:func:`jtsmm.tools.timegrid.frame_grid_index`). ``jts.aruco_camera``
    lays the poses out on the same grid (NaN in the holes), so an interval
    spans the phase's real duration. ``period_us`` defaults to the period
    estimated from the optical rows themselves.

    Returns ``(name, kind, start_frame, end_frame)`` with **both ends
    inclusive** -- the JTS ``frame_interval`` convention (``[start, end]`` is
    ``end - start + 1`` frames; see ``jts.core.FrameInterval``). Phases with
    no optical row inside them are omitted.
    """
    import math

    import numpy as np

    from jtsmm.tools.timegrid import estimate_period_us, frame_grid_index

    t = np.asarray(t_optical_us, dtype=np.uint64)
    if t.size == 0:
        return []
    period = float(period_us) if period_us else estimate_period_us(t)
    t0 = float(t[0])
    n_grid = int(frame_grid_index(t, period)[-1]) + 1 if period else t.size
    out: list[tuple[str, str, int, int]] = []
    for ph in timed:
        start = int(np.searchsorted(t, np.uint64(ph.t_start_us), side="left"))
        stop = int(np.searchsorted(t, np.uint64(ph.t_end_us), side="left"))
        if stop <= start:
            continue
        if str(ph.kind) == "mvc":
            continue
        if period:
            # Grid frames whose trigger time lies in [t_start, t_end).
            start = max(0, math.ceil((float(ph.t_start_us) - t0) / period))
            stop = min(n_grid, math.ceil((float(ph.t_end_us) - t0) / period))
            if stop <= start:
                continue
        out.append((ph.name, ph.kind, start, stop - 1))
    return out
