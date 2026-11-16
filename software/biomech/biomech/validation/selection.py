# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Which samples each validation metric uses (analysis plan D0b, fixed before results).

* **Sync and registration fit:** only the CT sync windows -- the start window inside
  ``mand_point_1`` and the end window inside the last ``free_motion`` -- bounded by the CT
  visibility runs (not by the whole phases).
* **Kinematic agreement (held out from the fit):** the movement-task phases only,
  trimmed by 0.5 s at both ends; ``free_motion`` is cut at the first CT appearance
  (the end sync lies in it). Calibration and landmark holds are never used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from biomech import timebase

MOVEMENT_TASKS: tuple[str, ...] = ("open_close", "protrusion_retrusion", "left_right",
                                   "chewing", "free_motion")
STATIC_TASKS: tuple[str, ...] = ("rest",)
EXCLUDED_PREFIXES: tuple[str, ...] = ("mand_point_", "max_point_", "tmj_")


@dataclass(frozen=True)
class Interval:
    name: str
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


def _phase_bounds(phases: Iterable, name: str, last: bool = False) -> tuple[float, float] | None:
    hits = [(p.t_start_us * 1e-6, p.t_end_us * 1e-6) for p in phases if p.name == name]
    if not hits:
        return None
    return hits[-1] if last else hits[0]


def ct_runs_within(t_s: np.ndarray, ct_ok: np.ndarray, t0: float, t1: float,
                   min_run: int = 1) -> tuple[float, float] | None:
    """First and last CT-visible frame time inside ``[t0, t1]``, counting only visibility
    runs of at least ``min_run`` frames (a single spurious detection is not an appearance)."""
    inside = (t_s >= t0) & (t_s <= t1)
    ok = np.asarray(ct_ok, bool) & inside
    if min_run > 1:
        keep = np.zeros_like(ok)
        for a, b in timebase.segments(ok):
            if b - a >= min_run:
                keep[a:b] = True
        ok = keep
    if not ok.any():
        return None
    ts = t_s[ok]
    return float(ts[0]), float(ts[-1])


def ct_episodes(t_s: np.ndarray, ct_ok: np.ndarray, t0: float, t1: float,
                gap_s: float = 2.5) -> list[tuple[float, float]]:
    """CT visibility inside ``[t0, t1]`` grouped into episodes: runs separated by less than
    ``gap_s`` of invisibility belong together (on/off bursts + the continuous motion form
    one episode; a later hold after a longer pause is the next one)."""
    sel = (t_s >= t0) & (t_s <= t1) & np.asarray(ct_ok, bool)
    ts = t_s[sel]
    if ts.size == 0:
        return []
    cut = np.flatnonzero(np.diff(ts) > gap_s)
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut, ts.size - 1]
    return [(float(ts[a]), float(ts[b])) for a, b in zip(starts, ends)]


def sync_windows(phases: Sequence, t_s: np.ndarray, ct_ok: np.ndarray, *,
                 start_phase: str = "mand_point_1", end_phase: str = "free_motion",
                 pad_s: float = 0.5, episode_gap_s: float = 2.5) -> list[Interval]:
    """The start and end CT sync windows (master seconds): the first CT episode (bursts +
    continuous motion, :func:`ct_episodes`) inside the first ``start_phase`` and inside the
    last ``end_phase``, padded by ``pad_s`` on each side (clipped to the phase) so that the
    first and last visibility edges are inside. The landmark hold that follows the start
    motion after a pause is not part of the window (D0b)."""
    out: list[Interval] = []
    for label, name, last in (("sync_start", start_phase, False), ("sync_end", end_phase, True)):
        b = _phase_bounds(phases, name, last=last)
        if b is None:
            continue
        eps = ct_episodes(t_s, ct_ok, *b, gap_s=episode_gap_s)
        if not eps:
            continue
        r = eps[0]
        out.append(Interval(label, max(b[0], r[0] - pad_s), min(b[1], r[1] + pad_s)))
    return out


def movement_intervals(phases: Sequence, t_s: np.ndarray, ct_ok: np.ndarray, *,
                       tasks: Sequence[str] = MOVEMENT_TASKS + STATIC_TASKS,
                       trim_s: float = 0.5, ct_guard_s: float = 0.5,
                       min_duration_s: float = 1.0, ct_min_run: int = 3) -> list[Interval]:
    """Held-out evaluation intervals: each listed task phase trimmed by ``trim_s``; any phase
    in which the CT appears is cut ``ct_guard_s`` before its first appearance (the end sync
    in ``free_motion``). Intervals shorter than ``min_duration_s`` are dropped."""
    out: list[Interval] = []
    for p in phases:
        if p.name not in tasks or p.name.startswith(EXCLUDED_PREFIXES):
            continue
        a, b = p.t_start_us * 1e-6 + trim_s, p.t_end_us * 1e-6 - trim_s
        r = ct_runs_within(t_s, ct_ok, a, b, min_run=ct_min_run)
        if r is not None:
            b = min(b, r[0] - ct_guard_s)
        if b - a >= min_duration_s:
            out.append(Interval(p.name, a, b))
    return out
