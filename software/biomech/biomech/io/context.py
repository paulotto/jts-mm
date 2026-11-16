# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""DataContext: the one way views, the viewer and the animator reach the data.

* Cohort-aware: tables default to the *included* participants (never stale or
  excluded units), and :meth:`pseudonym` is one global pid -> 1..n map, so
  "participant 3" is the same person in every panel and figure.
* Take-aware: :meth:`take_of` finds which take (motion or MVC) holds a phase.
* Cached and thread-safe: EMG preparations, store arrays and tables are LRU-cached
  (keyed by the stored unit keys, so a new run invalidates them); evicted preparations
  close their HDF5 files (the viewer used to leak them). Tables are returned as copies.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from biomech.cohort import Cohort, load_cohort
from biomech.config import StudyConfig
from biomech.io.session import SessionData

TAKES = ("motion", "mvc")


class _LRU:
    """Thread-safe LRU; a value is built once per key (concurrent callers wait for it).

    ``on_evict`` is called for evicted values; values that hold resources should rather
    release them when garbage-collected (see :func:`_close_when_unused`), because another
    thread may still use an evicted value."""

    def __init__(self, size: int, on_evict: Callable[[Any], None] | None = None) -> None:
        self.size = size
        self.on_evict = on_evict
        self._d: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.RLock()
        self._building: dict[Any, threading.Lock] = {}

    def get(self, key: Any, make: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
            build = self._building.setdefault(key, threading.Lock())
        with build:                            # one builder per key
            with self._lock:
                if key in self._d:
                    self._d.move_to_end(key)
                    return self._d[key]
            val = make()
            with self._lock:
                self._d[key] = val
                self._building.pop(key, None)
                while len(self._d) > self.size:
                    _, old = self._d.popitem(last=False)
                    if self.on_evict:
                        self.on_evict(old)
                return val

    def clear(self) -> None:
        with self._lock:
            for v in self._d.values():
                if self.on_evict:
                    self.on_evict(v)
            self._d.clear()


def _close_sessions(*sessions: Any) -> None:
    for sd in sessions:
        if sd is not None:
            try:
                sd.close()
            except Exception:              # already closed
                pass


def _close_when_unused(prep: Any) -> Any:
    """Close the preparation's HDF5 sessions when the last reference to it is gone (not on
    LRU eviction: a viewer thread may still be reading an evicted preparation)."""
    import weakref

    weakref.finalize(prep, _close_sessions, getattr(prep, "motion", None),
                     getattr(prep, "mvc", None))
    return prep


EMG_CACHE_ENV = "BIOMECH_EMG_CACHE"     # EMG preparations kept per DataContext (int)


def emg_cache_size(n_participants: int, size: int | None = None) -> int:
    """Explicit ``size`` > ``$BIOMECH_EMG_CACHE`` > the cohort size (at least 4): views that
    visit every participant in turn then prepare each one's EMG once instead of cycling
    through a 4-entry cache (one preparation holds the participant's cleaned phases)."""
    import os

    if size is None:
        env = os.environ.get(EMG_CACHE_ENV, "").strip()
        size = int(env) if env.isdigit() else max(4, int(n_participants))
    return max(1, int(size))


class DataContext:
    def __init__(self, config: StudyConfig, cohort: Cohort | None = None,
                 emg_cache: int | None = None) -> None:
        from biomech.results import ResultStore

        self.config = config
        self.cohort = cohort or load_cohort(config.cohort_path, config.data_dir)
        self.store = ResultStore(config.read_dir)     # live store or a snapshot
        # Sessions closed by _close_when_unused on eviction.
        self._emg = _LRU(emg_cache_size(len(self.cohort.included()), emg_cache))
        self._arrays = _LRU(64)
        self._tables = _LRU(256)
        self._phases: dict[tuple[str, str], list[Any]] = {}
        self._phase_lock = threading.Lock()

    # -- cohort ---------------------------------------------------------------------------
    def pids(self) -> list[str]:
        return sorted(p.pid for p in self.cohort.included())

    def pseudonym(self, pid: str) -> int:
        """1-based index among the included participants (stable across all outputs)."""
        return self.pids().index(pid) + 1

    def pseudonyms(self) -> dict[str, int]:
        return {p: i + 1 for i, p in enumerate(self.pids())}

    def session_path(self, pid: str, take: str = "motion") -> Path | None:
        p = self.cohort.get(pid)
        name = p.motion if take == "motion" else p.mvc
        return self.cohort.session_path(name) if name else None

    # -- results ----------------------------------------------------------------------------
    def onset_rule(self) -> str:
        """Primary EMG onset rule of the timing tables (both are stored; analysis_plan D6)."""
        return str(self.config.params("timing").get("onset_rule", "drive"))

    def table(self, stage: str, name: str, pids: list[str] | None = None):
        """A stage table over the included participants (or the cohort unit)."""
        from biomech.pipeline import get_stage
        from biomech.results import COHORT_UNIT

        from biomech.results.access import note_table, tracking_frame

        units = ([COHORT_UNIT] if get_stage(stage).scope == "cohort"
                 else list(pids if pids is not None else self.pids()))
        keys = tuple((u, m.key if (m := self.store.meta(stage, u)) else None) for u in units)
        note_table(stage, name)                         # cached reads count too
        df = self._tables.get((stage, name, keys),
                              lambda: self.store.read_table(stage, name, units))
        # Under results.access.track() the columns a view uses are noted (completeness).
        return tracking_frame(df.copy(), stage, name)

    def arrays(self, stage: str, pid: str) -> dict[str, np.ndarray]:
        meta = self.store.meta(stage, pid)
        key = (stage, pid, meta.key if meta else None)
        return self._arrays.get(key, lambda: self.store.read_arrays(stage, pid))

    # -- sessions -------------------------------------------------------------------------
    def phases(self, pid: str, take: str = "motion") -> list[Any]:
        key = (pid, take)
        with self._phase_lock:
            if key not in self._phases:
                path = self.session_path(pid, take)
                if path is None:
                    self._phases[key] = []
                else:
                    with SessionData(path) as sd:
                        self._phases[key] = list(sd.phases)
            return self._phases[key]

    def take_of(self, pid: str, phase: str) -> str | None:
        """Which take holds ``phase`` (motion first)."""
        for take in TAKES:
            if any(ph.name == phase for ph in self.phases(pid, take)):
                return take
        return None

    def phase(self, pid: str, phase: str):
        take = self.take_of(pid, phase)
        if take is None:
            return None, None
        return take, next(ph for ph in self.phases(pid, take) if ph.name == phase)

    def emg(self, pid: str):
        """The EmgPrep of a participant (motion + MVC take), cached; files closed on eviction."""
        from biomech.emg.prepare import EmgParams, EmgPrep

        params = json.dumps(self.config.signals.get("emg", {}), sort_keys=True)

        def make():
            motion = SessionData(self.session_path(pid, "motion"))
            mp = self.session_path(pid, "mvc")
            mvc = SessionData(mp) if mp else None
            return _close_when_unused(EmgPrep(motion, mvc,
                                              EmgParams.from_dict(json.loads(params))))

        return self._emg.get((pid, params), make)

    def previews(self, pid: str, take: str = "motion") -> list[dict[str, Any]]:
        """Preview-frame index rows (t_master_us, cam_id, phase, path) of a take."""
        path = self.session_path(pid, take)
        if path is None or not (path / "preview_index.jsonl").is_file():
            return []
        rows = [json.loads(line) for line in
                (path / "preview_index.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for r in rows:
            r["abs_path"] = str(path / r["path"])
        return rows

    def close(self) -> None:
        self._emg.clear()
        self._arrays.clear()
        self._tables.clear()
