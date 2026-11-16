# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ResultStore: the one place that knows how results are laid out on disk.

Every reader (CLI, figure specs, ``ml``, the Phase-2 viewer) goes through this
class; nothing else opens cache files. Layout under ``out_dir``::

    cache/<stage>/<unit>/meta.json        status, cache key, provenance
    cache/<stage>/<unit>/<table>.parquet  tidy tables
    cache/<stage>/<unit>/arrays.npz       signals (L1)
    runs/<run_id>/manifest.json           what ran, with which code and config
    runs/<run_id>/events.jsonl            progress/warning/error events

``unit`` is a participant id for per-participant stages and ``_cohort`` for
cohort stages. Writes are atomic: the new unit is built in a dot-prefixed temp
dir, the old unit is renamed aside, the new one renamed in, and only then is the
old one deleted. Readers therefore see either the old or the new unit, never a
gap or a half-written one, and dot-prefixed leftovers of a killed process are
invisible to readers and removed by :meth:`ResultStore.gc`.

One run at a time writes a store: :meth:`ResultStore.acquire_run_lock`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

from biomech.results.access import note_table, tracking

# The parquet engine is loaded here, once, in the importing (main) thread: pandas imports
# pyarrow.parquet lazily on the first to_parquet/read_parquet, and that first import from
# several pipeline worker threads at once deadlocked on the import lock (test suite hang).
pd.io.parquet.get_engine("auto")

COHORT_UNIT = "_cohort"
Status = Literal["missing", "failed", "stale", "up-to-date"]


@dataclass(frozen=True)
class UnitMeta:
    stage: str
    unit: str
    key: str
    status: str            # "ok" | "failed"
    created_utc: str
    tables: tuple[str, ...]
    has_arrays: bool
    run_id: str | None
    elapsed_s: float | None
    error: str | None
    extra: dict[str, Any]


class RunLock:
    """Exclusive lock file ``<out_dir>/.run.lock`` held by one run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def release(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ResultStore:
    def __init__(self, out_dir: str | Path) -> None:
        self.root = Path(out_dir)

    # -- paths ----------------------------------------------------------------------
    def _unit_dir(self, stage: str, unit: str) -> Path:
        return self.root / "cache" / stage / unit

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    # -- status ---------------------------------------------------------------------
    def meta(self, stage: str, unit: str) -> UnitMeta | None:
        p = self._unit_dir(stage, unit) / "meta.json"
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return UnitMeta(
            stage=d["stage"], unit=d["unit"], key=d["key"], status=d["status"],
            created_utc=d["created_utc"], tables=tuple(d.get("tables") or ()),
            has_arrays=bool(d.get("has_arrays")), run_id=d.get("run_id"),
            elapsed_s=d.get("elapsed_s"), error=d.get("error"), extra=d.get("extra") or {},
        )

    def status(self, stage: str, unit: str, expected_key: str | None = None) -> Status:
        m = self.meta(stage, unit)
        if m is None:
            return "missing"
        if expected_key is not None and m.key != expected_key:
            return "stale"
        return "failed" if m.status != "ok" else "up-to-date"

    def units(self, stage: str) -> list[str]:
        d = self.root / "cache" / stage
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir()
                      if not p.name.startswith(".") and (p / "meta.json").is_file())

    def stages(self) -> list[str]:
        d = self.root / "cache"
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))

    # -- maintenance ----------------------------------------------------------------
    def acquire_run_lock(self, run_id: str) -> RunLock:
        """Refuse a second concurrent run on this store; a dead owner's lock is taken over."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / ".run.lock"
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    owner = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    owner = {}
                pid = int(owner.get("pid", -1))
                if pid > 0 and _pid_alive(pid) and pid != os.getpid():
                    raise RuntimeError(f"another run ({owner.get('run_id')}, pid {pid}) is "
                                       f"writing {self.root}; wait for it or cancel it") from None
                if pid == os.getpid():
                    raise RuntimeError(f"a run ({owner.get('run_id')}) is already active in "
                                       "this process") from None
                path.unlink(missing_ok=True)      # stale lock of a dead process
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "run_id": run_id, "t": time.time()}, fh)
            return RunLock(path)
        raise RuntimeError(f"could not acquire {path}")

    def gc(self) -> list[Path]:
        """Remove dot-prefixed leftovers of interrupted writes. Not during a run."""
        lock = self.root / ".run.lock"
        if lock.exists():
            try:
                pid = int(json.loads(lock.read_text(encoding="utf-8")).get("pid", -1))
            except (OSError, ValueError):
                pid = -1
            if pid > 0 and _pid_alive(pid):
                raise RuntimeError("a run is active; gc refused")
        removed = []
        cache = self.root / "cache"
        if cache.is_dir():
            for d in cache.iterdir():
                if not d.is_dir():
                    continue
                for p in sorted(d.iterdir()):
                    if not (p.name.startswith(".") and p.is_dir()):
                        continue
                    # ".<unit>.old.<pid>.<ns>": the previous result moved aside by write();
                    # if the process died before the new one was renamed in, restore it.
                    parts = p.name[1:].split(".old.")
                    final = d / parts[0]
                    if len(parts) == 2 and not final.exists() and (p / "meta.json").is_file():
                        p.rename(final)
                        continue
                    shutil.rmtree(p, ignore_errors=True)
                    removed.append(p)
        return removed

    # -- write ----------------------------------------------------------------------
    def write(
        self,
        stage: str,
        unit: str,
        key: str,
        *,
        tables: dict[str, pd.DataFrame] | None = None,
        arrays: dict[str, np.ndarray] | None = None,
        status: str = "ok",
        error: str | None = None,
        run_id: str | None = None,
        elapsed_s: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        final = self._unit_dir(stage, unit)
        final.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{unit}.", dir=final.parent))
        try:
            names = []
            for name, df in (tables or {}).items():
                df.to_parquet(tmp / f"{name}.parquet", index=False)
                names.append(name)
            if arrays:
                np.savez_compressed(tmp / "arrays.npz", **arrays)
            meta = {
                "stage": stage, "unit": unit, "key": key, "status": status,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "tables": names, "has_arrays": bool(arrays), "run_id": run_id,
                "elapsed_s": elapsed_s, "error": error, "extra": extra or {},
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
            old = None
            if final.exists():
                old = final.parent / f".{unit}.old.{os.getpid()}.{time.monotonic_ns()}"
                final.rename(old)
            try:
                tmp.rename(final)
            except OSError:
                if old is not None and not final.exists():
                    old.rename(final)            # put the previous result back
                raise
            if old is not None:
                shutil.rmtree(old, ignore_errors=True)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    # -- read -----------------------------------------------------------------------
    def read_table(
        self, stage: str, name: str, units: Iterable[str] | None = None, *, ok_only: bool = True
    ) -> pd.DataFrame:
        """One table concatenated over units; adds a ``unit`` column if absent."""
        note_table(stage, name)
        frames = []
        for unit in units if units is not None else self.units(stage):
            m = self.meta(stage, unit)
            if m is None or (ok_only and m.status != "ok") or name not in m.tables:
                continue
            df = pd.read_parquet(self._unit_dir(stage, unit) / f"{name}.parquet")
            if "unit" not in df.columns:
                df.insert(0, "unit", unit)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def read_arrays(self, stage: str, unit: str) -> dict[str, np.ndarray]:
        p = self._unit_dir(stage, unit) / "arrays.npz"
        if not p.is_file():
            return tracking(stage, {})
        with np.load(p, allow_pickle=False) as z:
            return tracking(stage, {k: z[k] for k in z.files})

    # -- runs -----------------------------------------------------------------------
    def runs(self) -> list[str]:
        d = self.root / "runs"
        return sorted((p.name for p in d.iterdir() if p.is_dir()), reverse=True) if d.is_dir() else []

    def run_manifest(self, run_id: str) -> dict[str, Any]:
        p = self.run_dir(run_id) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    def run_events(self, run_id: str) -> list[dict[str, Any]]:
        p = self.run_dir(run_id) / "events.jsonl"
        if not p.is_file():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]
