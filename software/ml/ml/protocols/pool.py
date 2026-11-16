# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One process pool for many fits: longest task first, no barriers, results by key.

The ML stages fit thousands of models of very different cost (an SVM with C = 0.01 takes
about three times as long as one with C = 10). :func:`run_tasks` keeps every worker busy:
tasks wait in a priority queue (highest expected ``cost`` first, then submission order) and
the next one starts as soon as any worker is free; ``on_result`` may add tasks (e.g. the
refit of a fold once its inner grid search is complete). Results are returned *by key*, so
the outcome never depends on completion order or on the number of workers.

Workers are private loky processes with one BLAS/OpenMP thread each (threadpoolctl), so a
pool never competes with another stage's pool. Large arrays reach them through
:class:`Shared`: written once as ``.npy`` files (``/dev/shm`` when available) and
memory-mapped read-only by every worker, instead of being pickled into every task.
``lanes`` give a task group its own executor (``ml_online`` replays on one reserved core).
"""

from __future__ import annotations

import heapq
import itertools
import os
import shutil
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable

import numpy as np

THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "BLIS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1"}
_LIMITER: Any = None


# -- shared arrays ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Ref:
    """A shared array: ``get()`` memory-maps it (read-only, once per process)."""

    path: str

    def get(self) -> np.ndarray:
        a = _MAPS.get(self.path)
        if a is None:
            a = np.load(self.path, mmap_mode="r", allow_pickle=False)
            if len(_MAPS) > 64:
                _MAPS.clear()
            _MAPS[self.path] = a
        return a


_MAPS: dict[str, np.ndarray] = {}


def _tmp_root() -> str | None:
    shm = Path("/dev/shm")
    try:
        if shm.is_dir() and os.access(shm, os.W_OK) and shutil.disk_usage(shm).free > 2 << 30:
            return str(shm)
    except OSError:
        pass
    return None


class Shared:
    """Arrays written once for all workers (context manager; the files go on exit)."""

    def __init__(self, prefix: str = "ml_pool_") -> None:
        self.dir = tempfile.mkdtemp(prefix=prefix, dir=_tmp_root())
        self._n = itertools.count()
        self._ids: dict[int, tuple[Any, Ref]] = {}

    def put(self, a: np.ndarray) -> Ref:
        """Store ``a`` (exact dtype and values; the same object is written once)."""
        hit = self._ids.get(id(a))
        if hit is not None and hit[0] is a:
            return hit[1]
        p = os.path.join(self.dir, f"{next(self._n)}.npy")
        np.save(p, np.ascontiguousarray(a), allow_pickle=False)
        ref = Ref(p)
        self._ids[id(a)] = (a, ref)
        return ref

    def close(self) -> None:
        for p in [k for k in _MAPS if k.startswith(self.dir)]:
            _MAPS.pop(p, None)
        self._ids.clear()
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self) -> "Shared":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# -- tasks ----------------------------------------------------------------------------------------
@dataclass
class Task:
    key: Hashable
    fn: Callable[..., Any]
    args: tuple = ()
    cost: float = 1.0
    lane: str = "main"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Lane:
    workers: int
    cpus: tuple[int, ...] | None = None        # CPU affinity of the lane's processes


def init_worker(cpus: tuple[int, ...] | None = None) -> None:
    """Worker start: one BLAS/OpenMP thread, optional CPU affinity."""
    global _LIMITER
    os.environ.update(THREAD_ENV)
    try:
        from threadpoolctl import threadpool_limits

        _LIMITER = threadpool_limits(limits=1)
    except Exception:                                   # pragma: no cover - optional
        _LIMITER = None
    if cpus:
        try:
            os.sched_setaffinity(0, set(cpus))
        except (AttributeError, OSError):             # pragma: no cover - non-Linux
            pass


def _executor(lane: Lane):
    from joblib.externals.loky.process_executor import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=lane.workers, initializer=init_worker,
                               initargs=(lane.cpus,), env=dict(THREAD_ENV))


def _call(fn: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
    return fn(*args, **kwargs)


def run_tasks(tasks: Iterable[Task], n_jobs: int | dict[str, Lane], *,
              on_result: Callable[[Hashable, Any], Iterable[Task] | None] | None = None,
              cancelled: Callable[[], bool] | None = None) -> dict[Hashable, Any]:
    """Run ``tasks`` (and those ``on_result`` adds), longest expected first; ``{key: result}``.

    ``n_jobs`` = workers of the one lane ``"main"`` (1 = in this process, in queue order),
    or ``{lane: Lane}``. ``cancelled`` is polled after every result: True shuts the pools
    down and raises ``RuntimeError("cancelled")``. A task's exception is raised here."""
    lanes = n_jobs if isinstance(n_jobs, dict) else {"main": Lane(max(int(n_jobs), 1))}
    seq = itertools.count()
    queues: dict[str, list] = {name: [] for name in lanes}
    keys: set[Hashable] = set()

    def push(ts: Iterable[Task] | None) -> None:
        for t in ts or ():
            if t.key in keys:
                raise ValueError(f"duplicate task key {t.key!r}")
            if t.lane not in queues:
                raise ValueError(f"task {t.key!r}: unknown lane {t.lane!r}")
            keys.add(t.key)
            heapq.heappush(queues[t.lane], (-float(t.cost), next(seq), t))

    def check() -> None:
        if cancelled is not None and cancelled():
            raise RuntimeError("cancelled")

    push(tasks)
    out: dict[Hashable, Any] = {}
    serial = all(lane.workers <= 1 and lane.cpus is None for lane in lanes.values()) \
        and len(lanes) == 1
    if serial:
        q = next(iter(queues.values()))
        while q:
            check()
            t = heapq.heappop(q)[2]
            out[t.key] = r = t.fn(*t.args, **t.kwargs)
            push(on_result(t.key, r) if on_result is not None else None)
        return out
    pools = {name: _executor(lane) for name, lane in lanes.items()}
    running: dict[Future, tuple[str, Task]] = {}
    ok = False
    try:
        def fill() -> None:
            for name, lane in lanes.items():
                busy = sum(1 for ln, _ in running.values() if ln == name)
                q = queues[name]
                while q and busy < lane.workers:
                    t = heapq.heappop(q)[2]
                    running[pools[name].submit(_call, t.fn, t.args, t.kwargs)] = (name, t)
                    busy += 1

        fill()
        while running:
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for fut in sorted(done, key=lambda f: running[f][1].cost, reverse=True):
                _, t = running.pop(fut)
                out[t.key] = r = fut.result()
                push(on_result(t.key, r) if on_result is not None else None)
            check()
            fill()
        ok = True
    finally:
        for ex in pools.values():
            if ok:
                ex.shutdown(wait=True)
            else:
                for f in running:
                    f.cancel()
                ex.shutdown(wait=False, kill_workers=True)
    return out


# -- CPU topology (ml_online's reserved replay core) ---------------------------------------------
def _read(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _cpu_list(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def reserve_core() -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """``(replay CPU, the other CPUs)``: one logical CPU of the fastest core (highest
    ``cpuinfo_max_freq``; P-core on hybrid Intel) for the replay, the remaining CPUs minus
    that core's SMT siblings for the pool. None without affinity support or < 4 CPUs."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 4:
        return None
    base = Path("/sys/devices/system/cpu")
    freq = {c: int(_read(base / f"cpu{c}/cpufreq/cpuinfo_max_freq") or 0) for c in allowed}
    best = max(freq.values())
    # the last fastest CPU (cpu0 takes interrupts)
    cpu = [c for c in allowed if freq[c] == best][-1]
    sib = _read(base / f"cpu{cpu}/topology/thread_siblings_list")
    core = set(_cpu_list(sib)) if sib else {cpu}
    rest = tuple(c for c in allowed if c not in core)
    return (cpu,), rest


__all__ = ["Lane", "Ref", "Shared", "Task", "init_worker", "reserve_core", "run_tasks"]
