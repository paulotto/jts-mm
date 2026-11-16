# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stage registry and the run API.

``run()`` is the only way analyses execute; the CLI and the Phase-2 control
center call it the same way. A stage is a function registered with
:func:`stage`; it receives a context and returns a :class:`StageResult`, and it
never touches the disk itself.

Cache keys chain: a unit's key hashes the stage name and version, its parameter
block, the participant record, the input files' fingerprints, and the keys of
the upstream units it read. Changing anything upstream therefore marks every
dependent unit ``stale`` without having to track it explicitly.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import pandas as pd

import biomech
from biomech.cohort import Cohort, Participant, load_cohort
from biomech.config import StudyConfig, stable_hash
from biomech.results import COHORT_UNIT, ResultStore
from biomech.results.provenance import git_state as _git_state
from biomech.results.provenance import package_versions as _versions

Scope = Literal["participant", "cohort"]


# -- results & contexts ----------------------------------------------------------------
@dataclass
class StageResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    arrays: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Context:
    config: StudyConfig
    cohort: Cohort
    params: dict[str, Any]
    store: ResultStore

    @property
    def signals(self) -> dict[str, Any]:
        return self.config.signals


class UnitCache:
    """Per-participant objects shared by the stages of one worker (participant-major runs).

    Opening a session and preparing its EMG (mains fit per phase) is the costly common
    part of ``qc``, ``emg``, ``timing`` and ``ml_windows``; within one participant's chain
    it is done once. Every entry is a pure function of (participant, config), so sharing
    never changes results; a stage-major run simply gets a fresh cache per unit.
    """

    def __init__(self, cohort: Cohort, participant: Participant) -> None:
        self.cohort = cohort
        self.participant = participant
        self._memo: dict[str, Any] = {}
        self._sessions: dict[str, Any] = {}

    def session(self, which: str = "motion") -> Any:
        """The open :class:`~biomech.io.session.SessionData` of the motion or MVC take
        (None for a missing MVC take). Do not close it; the cache does."""
        if which not in self._sessions:
            from biomech.io.session import SessionData

            rec = self.participant.motion if which == "motion" else self.participant.mvc
            self._sessions[which] = SessionData(self.cohort.session_path(rec)) if rec else None
        return self._sessions[which]

    def memo(self, key: str, make: Callable[[], Any]) -> Any:
        if key not in self._memo:
            self._memo[key] = make()
        return self._memo[key]

    def close(self) -> None:
        self._memo.clear()
        for sd in self._sessions.values():
            if sd is not None:
                sd.close()
        self._sessions.clear()


@dataclass
class ParticipantContext(_Context):
    participant: Participant
    cache: UnitCache | None = None

    @property
    def pid(self) -> str:
        return self.participant.pid

    def _cache(self) -> UnitCache:
        if self.cache is None:
            self.cache = UnitCache(self.cohort, self.participant)
        return self.cache

    def session(self, which: str = "motion") -> Any:
        """Shared open session (motion or mvc take); owned by the run, never close it."""
        return self._cache().session(which)

    def memo(self, key: str, make: Callable[[], Any]) -> Any:
        """Shared per-participant value (e.g. the pose chain); ``make`` must be pure."""
        return self._cache().memo(key, make)

    @property
    def seed(self) -> int:
        """Study seed combined with the participant id: reproducible, but no participant
        shares another's random draws (Monte Carlo, bootstraps)."""
        return participant_seed(self.config.seed, self.pid)

    def emg_prep(self) -> Any:
        """The participant's :class:`~biomech.emg.prepare.EmgPrep` under the run's EMG
        signal parameters, shared across stages (cleaned phases are computed once)."""
        from biomech.emg.prepare import EmgParams, EmgPrep

        emg = self.signals["emg"]
        return self.memo("emg_prep:" + stable_hash(emg),
                         lambda: EmgPrep(self.session("motion"), self.session("mvc"),
                                         EmgParams.from_dict(emg)))

    def motion_path(self) -> Path:
        return self.cohort.session_path(self.participant.motion)

    def mvc_path(self) -> Path | None:
        return self.cohort.session_path(self.participant.mvc) if self.participant.mvc else None

    def upstream(self, stage_name: str, table: str) -> pd.DataFrame:
        return self.store.read_table(stage_name, table, [self.pid])

    def upstream_arrays(self, stage_name: str) -> dict[str, Any]:
        return self.store.read_arrays(stage_name, self.pid)


@dataclass
class CohortContext(_Context):
    participants: list[Participant]
    # Included participants left out of this cohort stage, pid -> reason (failed or
    # missing upstream; see run()); the stage stores them in its "coverage" table.
    excluded: dict[str, str] = field(default_factory=dict)
    cancel: Any = None

    def cancelled(self) -> bool:
        """True once the run was cancelled: long cohort stages check it between steps."""
        return _is_cancelled(self.cancel)

    def upstream(self, stage_name: str, table: str) -> pd.DataFrame:
        units = [p.pid for p in self.participants]
        if get_stage(stage_name).scope == "cohort":
            units = [COHORT_UNIT]
        return self.store.read_table(stage_name, table, units)

    def upstream_arrays(self, stage_name: str, pid: str) -> dict[str, Any]:
        return self.store.read_arrays(stage_name, pid)

    def demographics(self) -> pd.DataFrame:
        """Per-participant sex/age for statistics only (never exported per person)."""
        return pd.DataFrame([{"pid": p.pid, "sex": p.sex, "age": p.age}
                             for p in self.participants])


def participant_seed(seed: int, pid: str) -> int:
    """A stable 32-bit seed per (study seed, participant)."""
    import zlib

    return int(zlib.crc32(f"{int(seed)}:{pid}".encode()) & 0x7FFFFFFF)


# -- registry --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    name: str
    func: Callable[..., StageResult]
    scope: Scope
    depends: tuple[str, ...]
    version: str
    description: str
    group: str = "dataset"          # study group (StudyConfig.groups): dataset | validation
    detach: bool = False            # scheduling only, see stage()
    partial: tuple[str, ...] = ()   # participant deps whose failed units do not exclude


_REGISTRY: dict[str, Stage] = {}


def stage(
    name: str,
    *,
    scope: Scope = "participant",
    depends: Iterable[str] = (),
    version: str = "1",
    description: str = "",
    group: str = "dataset",
    detach: bool = False,
    partial: Iterable[str] = (),
) -> Callable[[Callable[..., StageResult]], Callable[..., StageResult]]:
    """Register a stage. Bump ``version`` whenever its logic changes results. ``group``
    names the study kind that runs it (a study runs only its ``pipeline.groups``).

    ``detach`` (participant stages; scheduling only, never results): in parallel
    participant-major runs the stage runs as its own task after the rest of the
    participant's chain has returned, instead of at the end of that chain, so cohort stages
    that do not need it start without waiting for it (``uncertainty``: about 40 % of a chain
    and only the kinematics statistics read it). It opens its own session and cache.

    ``partial`` (cohort stages; a subset of ``depends``, participant scope): inputs that only
    some participants have (E32: ``imu``, absent without jaw-IMU data). A failed unit of
    such a dependency does not exclude the participant from the stage -- each input keeps
    its own participant set (``ctx.upstream`` reads the units that exist), and the
    ``coverage`` table names what the participant lacks. Stale or missing units still
    block, as for every dependency."""
    partial = tuple(partial)
    if bad := [d for d in partial if d not in tuple(depends)]:
        raise ValueError(f"stage {name!r}: partial dependencies {bad} not in depends")

    def deco(fn: Callable[..., StageResult]) -> Callable[..., StageResult]:
        _REGISTRY[name] = Stage(name, fn, scope, tuple(depends), version,
                                description or (fn.__doc__ or "").strip().split("\n")[0],
                                group, detach, partial)
        return fn

    return deco


_PLUGINS_LOADED = False


def registry() -> dict[str, Stage]:
    """Built-in stages plus those of installed plugins (entry-point group ``biomech.stages``)."""
    global _PLUGINS_LOADED
    import biomech.stages  # noqa: F401  (registers the built-in stages)

    if not _PLUGINS_LOADED:
        _PLUGINS_LOADED = True
        for ep in importlib.metadata.entry_points(group="biomech.stages"):
            try:
                ep.load()
            except Exception as exc:        # a broken plugin must not take the core down
                import warnings

                warnings.warn(f"biomech stage plugin {ep.name!r} not loaded: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)
    return dict(_REGISTRY)


def get_stage(name: str) -> Stage:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown stage {name!r}; known: {', '.join(sorted(reg))}")
    return reg[name]


def validate_config(config: StudyConfig) -> None:
    """Every ``stages`` block must belong to a registered stage (catches typos that would
    otherwise silently fall back to the defaults)."""
    from biomech.config import ConfigError

    reg = registry()
    unknown = sorted(set(config.study_stage_keys) - set(reg))
    if unknown:
        raise ConfigError(f"unknown stage block(s) {unknown} in the config; registered "
                          f"stages: {', '.join(sorted(reg))}")


def group_stages(groups: Iterable[str] = ("dataset",)) -> list[str]:
    """Registered stages of the given study groups, in registration order."""
    gs = set(groups)
    return [n for n, st in registry().items() if st.group in gs]


def resolve_order(names: Iterable[str] | None,
                  groups: Iterable[str] = ("dataset",)) -> list[str]:
    """Requested stages plus their dependencies, dependencies first (default: every stage
    of ``groups``). A requested stage or a dependency outside ``groups`` is an error: the
    dataset analysis and the system validation never run each other's stages."""
    gs = set(groups)
    wanted = list(names) if names else group_stages(gs)
    order: list[str] = []
    visiting: set[str] = set()

    def visit(n: str) -> None:
        if n in order:
            return
        if n in visiting:
            raise ValueError(f"stage dependency cycle at {n!r}")
        visiting.add(n)
        st = get_stage(n)
        if st.group not in gs:
            raise ValueError(f"stage {n!r} belongs to group {st.group!r}, not to this study's "
                             f"groups {sorted(gs)} (pipeline.groups)")
        for dep in st.depends:
            visit(dep)
        visiting.discard(n)
        order.append(n)

    for n in wanted:
        visit(n)
    return order


# -- keys ------------------------------------------------------------------------------
def _fingerprint(path: Path | None) -> list[Any] | None:
    if path is None:
        return None
    h5 = path / "session.h5"
    try:
        st = h5.stat()
    except OSError:
        return [h5.name, "missing"]
    return [h5.name, st.st_size, st.st_mtime_ns]


def _inputs_present(cohort: Cohort, p: Participant) -> bool:
    paths = [cohort.session_path(p.motion)] + ([cohort.session_path(p.mvc)] if p.mvc else [])
    return all((q / "session.h5").is_file() for q in paths)


_CODE_HASH: dict[str | None, str] = {}       # stage group (None: all) -> code hash
# Code that never changes results: presentation and orchestration. Matched on the
# path *relative to the biomech package*, never on a basename (emg/pipeline.py is
# analysis code; only the top-level pipeline.py is orchestration).
PRESENTATION = ("viz", "viewer", "results", "anim", "check")
NOT_ANALYSIS = {("cli.py",), ("pipeline.py",), ("catalog.py",), ("etc", "catalog.yaml"),
                # Reader wording: plan-code families and value vocabularies.
                ("etc", "codes.yaml"), ("etc", "vocab.yaml"), ("etc", "glossary.yaml"),
                ("etc", "tables.yaml"),
                # Literature reference values: read only by views (J7).
                ("etc", "literature.yaml"),
                ("papers.py",),
                # Parts manifest loader (hardware/mechanical): read by views only (J13).
                ("hwparts.py",),
                # Read-side helpers used only by views, the viewer and reports.
                ("io", "context.py"), ("ensembles.py",), ("report.py",),
                ("sensitivity.py",),
                # The resolved configuration enters every key (params_hash); the loader
                # code itself does not change results.
                ("config.py",)}
# Data files that change results (literature norms, defaults) are hashed with the code.
HASHED_SUFFIXES = (".py", ".yaml", ".yml", ".json")
# jtsmm: whole signal packages, plus single files the analysis imports from the viewer and
# tools packages (the rest of those is the live viewer and plotting scripts).
JTSMM_DIRS = ("config", "emg", "imu", "io")
JTSMM_FILES = ("viewer/processing.py", "tools/run_jts_analysis.py",
               "tools/plot_mp_trajectory.py", "tools/mock_session.py", "tools/timegrid.py")
# Subpackages of stage plugins (e.g. ml) that never change results: CLI, labelling app,
# training CLI runs.
PLUGIN_PRESENTATION = ("cli", "labeling", "training")


def _package_roots(mod: Any) -> list[Path]:
    """Directories of a (possibly namespace) package."""
    f = getattr(mod, "__file__", None)
    if f:
        return [Path(f).parent]
    return [Path(p) for p in getattr(mod, "__path__", [])]


def _is_presentation(rel: tuple[str, ...]) -> bool:
    return rel[0] in PRESENTATION or rel in NOT_ANALYSIS


def code_hash(group: str | None = None) -> str:
    """Hash of the analysis source: biomech (minus the viewer) and the jtsmm signal code.

    Part of every cache key, so editing analysis code can never leave results
    that still read as up to date. Over-invalidates (any analysis edit re-runs
    everything), which is cheap at this cohort size and always safe.

    ``group`` (a stage group, ``Stage.group``): only the source that group's stages can
    run (:func:`group_files`), so editing validation code leaves the dataset keys valid and
    vice versa. None: every analysis file (snapshot provenance).
    """
    if group not in _CODE_HASH:
        import hashlib

        files = hashed_files()
        if group is not None:
            mine = set(group_files(group, files))
            files = [(r, f) for r, f in files if f in mine]
        h = hashlib.sha256()
        for root, f in files:
            h.update(str(f.relative_to(root.parent)).encode())
            h.update(f.read_bytes())
        _CODE_HASH[group] = h.hexdigest()[:16]
    return _CODE_HASH[group]


def _module_name(root: Path, f: Path) -> str:
    """Dotted module name of source file ``f`` under package root ``root``."""
    top = root
    while (top.parent / "__init__.py").is_file():
        top = top.parent
    parts = list(f.relative_to(top.parent).with_suffix("").parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(f: Path, mod: str) -> set[str]:
    """Every module ``f`` imports anywhere (top level or inside functions; ``from a import
    b`` gives ``a`` and ``a.b``, which may be a submodule), relative imports resolved."""
    import ast

    pkg = mod if f.name == "__init__.py" else mod.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(f.read_bytes(), filename=str(f))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = pkg.split(".")[:len(pkg.split(".")) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            out.add(base)
            out.update(f"{base}.{a.name}" for a in node.names)
    return out


def group_files(group: str, files: list[tuple[Path, Path]] | None = None) -> list[Path]:
    """The hashed files (:func:`hashed_files`) a stage group depends on: the modules that
    define its stages, everything they import (statically, at any depth, with the parent
    packages' ``__init__``) -- but not the stage modules of other groups (the registry
    imports them all) -- plus every data file (yaml/json, read by path). A file no stage
    imports is in no group; any parse problem gives every file (over-invalidation is safe).
    """
    files = hashed_files() if files is None else files
    index: dict[str, Path] = {}
    data: list[Path] = []
    for root, f in files:
        if f.suffix == ".py":
            index[_module_name(root, f)] = f
        else:
            data.append(f)
    owners: dict[str, set[str]] = {}
    for st in registry().values():
        owners.setdefault(st.func.__module__, set()).add(st.group)
    todo = [m for m, g in owners.items() if group in g]
    seen: set[str] = set()
    try:
        while todo:
            m = todo.pop()
            if m in seen or m not in index or group not in owners.get(m, {group}):
                continue
            seen.add(m)
            parts = m.split(".")
            todo += [".".join(parts[:i]) for i in range(1, len(parts))]
            todo += [x for x in _imports(index[m], m) if x in index]
    except (SyntaxError, ValueError, OSError):
        return [f for _, f in files]
    return [f for _, f in files if f in data or f in {index[m] for m in seen}]


def hashed_files() -> list[tuple[Path, Path]]:
    """(package root, file) pairs that enter :func:`code_hash`."""
    import jtsmm

    broot = Path(biomech.__file__).parent
    roots = [broot]
    jroot = Path(jtsmm.__file__).parent
    # jtsmm.config holds the channel LSB scaling to µV; jtsmm.viewer the causal
    # filters the ml stage uses.
    roots += [jroot / d for d in JTSMM_DIRS]
    # Plugin packages that registered stages (e.g. ml) count as analysis code too.
    for st in registry().values():
        top = sys.modules.get(st.func.__module__.split(".")[0])
        if top is not None and top.__name__ not in ("biomech", "jtsmm"):
            roots += _package_roots(top)
    out = []
    for root in dict.fromkeys(roots):
        files = sorted(f for f in root.rglob("*") if f.is_file()
                       and f.suffix in HASHED_SUFFIXES and "__pycache__" not in f.parts)
        for f in files:
            # Presentation code (figures, viewer, CLI) never changes results, so
            # editing it must not invalidate analyses.
            rel = f.relative_to(root).parts
            if root == broot and _is_presentation(rel):
                continue
            if root != broot and root.parent != jroot and rel[0] in PLUGIN_PRESENTATION:
                continue
            out.append((root, f))
    # Single jtsmm files the analysis imports from otherwise presentation packages.
    out += [(jroot, jroot / f) for f in JTSMM_FILES if (jroot / f).is_file()]
    return out


def _depends_on_cohort(name: str) -> bool:
    """Is ``name`` a cohort stage or (transitively) downstream of one?"""
    st = get_stage(name)
    return st.scope == "cohort" or any(_depends_on_cohort(d) for d in st.depends)


class KeyResolver:
    """Expected cache keys, chained through the *expected* upstream keys.

    Using what is stored upstream instead would leave a cohort stage looking
    up to date after an upstream parameter change until something re-ran.
    """

    def __init__(self, config: StudyConfig, cohort: Cohort,
                 store: ResultStore | None = None) -> None:
        self.config = config
        self.cohort = cohort
        self.store = store
        self._memo: dict[tuple[str, str], str] = {}

    def coverage(self, stage_name: str) -> tuple[list[str], dict[str, str], list[str]]:
        """Participants a cohort stage uses: (available pids, excluded pid -> reason,
        blocking pids).

        A participant is *excluded* (stage runs on the others, reason stored) when its
        input session is missing or a participant-scope upstream unit failed with the
        current key -- except for the stage's ``partial`` dependencies, whose failed units
        leave the participant in (E32); it is *blocking* (stage waits) when such a unit is
        merely stale or not computed yet -- a cohort statistic is never computed silently
        from a subset because some participants simply were not run.
        """
        st = get_stage(stage_name)
        pdeps = [d for d in st.depends if get_stage(d).scope == "participant"]
        avail, excl, block = [], {}, []
        for p in self.cohort.included():
            if not _inputs_present(self.cohort, p):
                excl[p.pid] = "input session missing"
                continue
            reason, waits = None, False
            for d in pdeps:
                status = (self.store.status(d, p.pid, self.key(d, p.pid))
                          if self.store is not None else "up-to-date")
                if status == "failed":
                    if d in st.partial:
                        continue
                    m = self.store.meta(d, p.pid) if self.store is not None else None
                    how = "skipped (upstream failed)" if m and m.status == "skipped" \
                        else "failed"
                    reason = reason or f"{d} {how}"
                elif status != "up-to-date":
                    waits = True
            if waits:
                block.append(p.pid)
            elif reason:
                excl[p.pid] = reason
            else:
                avail.append(p.pid)
        return avail, excl, block

    def key(self, stage_name: str, unit: str) -> str:
        memo = self._memo.get((stage_name, unit))
        if memo is not None:
            return memo
        st = get_stage(stage_name)
        payload: dict[str, Any] = {
            "stage": st.name, "version": st.version, "code": code_hash(st.group),
            "params": self.config.params_hash(st.name), "seed": int(self.config.seed),
        }
        if st.scope == "participant":
            p = self.cohort.get(unit)
            payload["participant"] = asdict(p)
            payload["inputs"] = [
                _fingerprint(self.cohort.session_path(p.motion)),
                _fingerprint(self.cohort.session_path(p.mvc)) if p.mvc else None,
            ]
            payload["upstream"] = {d: self.key(d, unit if get_stage(d).scope == "participant"
                                               else COHORT_UNIT) for d in st.depends}
        else:
            units, excluded, _ = self.coverage(stage_name)
            payload["units"] = units
            payload["excluded"] = excluded
            payload["upstream"] = {
                d: [self.key(d, COHORT_UNIT)] if get_stage(d).scope == "cohort"
                else [self.key(d, u) for u in units]
                for d in st.depends
            }
        out = stable_hash(payload, 24)
        if not _depends_on_cohort(stage_name):
            self._memo[(stage_name, unit)] = out    # cohort keys follow the store state
        return out


# -- events ----------------------------------------------------------------------------
class _Events:
    def __init__(self, path: Path, sink: Callable[[dict[str, Any]], None] | None) -> None:
        self.path = path
        self.sink = sink
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, kind: str, **data: Any) -> None:
        ev = {"t": time.time(), "kind": kind, **data}
        with self._lock:                 # cohort stages emit from threads
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, default=str) + "\n")
            if self.sink is not None:
                self.sink(ev)


@dataclass
class RunSummary:
    run_id: str
    statuses: dict[str, dict[str, str]]  # stage -> unit -> ran|cached|failed|skipped
    cancelled: bool = False

    @property
    def n_failed(self) -> int:
        return sum(s == "failed" for d in self.statuses.values() for s in d.values())


# -- execution -------------------------------------------------------------------------
def _exec_participant(stage_name: str, config: StudyConfig, cohort: Cohort, pid: str,
                      key: str, run_id: str, single_thread: bool = False
                      ) -> tuple[str, str, float, str | None]:
    """Run one participant unit and store it. Top-level so joblib can pickle it.

    ``single_thread``: one BLAS thread inside a parallel worker -- n_jobs processes x
    multithreaded numpy oversubscribed the CPU (the reweighted mains fits slowed
    down severalfold).
    """
    if single_thread:
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                return _exec_participant(stage_name, config, cohort, pid, key, run_id, False)
        except ImportError:
            pass
    cache = UnitCache(cohort, cohort.get(pid))
    try:
        status, dt, err = _exec_unit(stage_name, config, cohort, pid, key, run_id, cache)
    finally:
        cache.close()
    return pid, status, dt, err


def _exec_unit(stage_name: str, config: StudyConfig, cohort: Cohort, pid: str, key: str,
               run_id: str, cache: UnitCache) -> tuple[str, float, str | None]:
    st = get_stage(stage_name)
    store = ResultStore(config.out_dir)
    ctx = ParticipantContext(config=config, cohort=cohort, params=config.params(stage_name),
                             store=store, participant=cohort.get(pid), cache=cache)
    t0 = time.perf_counter()
    try:
        res = st.func(ctx)
        dt = time.perf_counter() - t0
        # Inside the try: an unwritable result (type error in a table, full disk) marks
        # this unit failed instead of aborting the whole run.
        store.write(stage_name, pid, key, tables=res.tables, arrays=res.arrays, run_id=run_id,
                    elapsed_s=dt, extra=res.extra)
        return "ran", dt, None
    except Exception:
        err = traceback.format_exc()
        dt = time.perf_counter() - t0
        try:
            store.write(stage_name, pid, key, status="failed", error=err, run_id=run_id,
                        elapsed_s=dt)
        except Exception:                 # even the failure record cannot be written
            pass
        return "failed", dt, err.strip().splitlines()[-1]


def _exec_segment(plan: list[tuple[str, str, str | None]], config: StudyConfig, cohort: Cohort,
                  pid: str, run_id: str, single_thread: bool = False,
                  external: dict[str, bool | str] | None = None
                  ) -> tuple[str, list[tuple[str, str, float, str | None]]]:
    """One participant's chain of participant stages in dependency order, one UnitCache.

    ``plan`` = [(stage, "todo" | "cached", key)]; ``external`` = participant-scope
    dependencies outside this plan -> current? (bool), or -> their status in this run
    ("ran", "cached", "failed", "skipped") when they ran in an earlier task of the same
    chain (detached stages). A stage whose upstream failed, was skipped or is not current
    is skipped, and the skip is stored as a unit with status "skipped" (so cohort stages
    can tell "failed upstream" from "not run yet"). Returns per-stage
    (stage, status, elapsed_s, error-or-reason).
    """
    if single_thread:
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(limits=1):
                return _exec_segment(plan, config, cohort, pid, run_id, False, external)
        except ImportError:
            pass
    cache = UnitCache(cohort, cohort.get(pid))
    done: dict[str, str] = {d: ok if isinstance(ok, str) else ("cached" if ok else "not current")
                            for d, ok in (external or {}).items()}
    out: list[tuple[str, str, float, str | None]] = []
    store = ResultStore(config.out_dir)
    try:
        for name, what, key in plan:
            bad = [d for d in get_stage(name).depends
                   if get_stage(d).scope == "participant" and done.get(d, "not current")
                   not in ("ran", "cached")]
            if bad:
                done[name] = "skipped"
                reason = f"upstream {bad} " + ", ".join(sorted({done.get(d, "?") for d in bad}))
                if key:
                    try:
                        store.write(name, pid, key, status="skipped", error=reason,
                                    run_id=run_id)
                    except Exception:
                        pass
                out.append((name, "skipped", 0.0, reason))
            elif what == "cached":
                done[name] = "cached"
            else:
                status, dt, err = _exec_unit(name, config, cohort, pid, key or "", run_id, cache)
                done[name] = status
                out.append((name, status, dt, err))
    finally:
        cache.close()
    return pid, out


def _cohort_blocked(name: str, keys: KeyResolver, current: Callable[[str, str], bool]
                    ) -> str | None:
    """Why cohort stage ``name`` cannot run now (None = it can, on its available units)."""
    st = get_stage(name)
    stale = [d for d in st.depends if get_stage(d).scope == "cohort"
             and not current(d, COHORT_UNIT)]
    if stale:
        return f"upstream {stale}"
    avail, excluded, block = keys.coverage(name)
    if block:
        return f"participant units not current: {', '.join(block)}"
    if not avail:
        return f"no participant available ({len(excluded)} excluded)"
    return None


def _executor(n_jobs: int):
    """Worker pool for participant tasks. joblib's threading backend (tests) gives threads;
    otherwise spawned processes (independent of joblib/loky, which ml_eval uses)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    try:
        from joblib.parallel import ThreadingBackend, get_active_backend

        threading_backend = isinstance(get_active_backend()[0], ThreadingBackend)
    except ImportError:                                    # pragma: no cover
        threading_backend = False
    n = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
    if threading_backend:
        return ThreadPoolExecutor(max_workers=n)
    return ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("spawn"))


def _map_tasks(fn: Callable[..., Any], tasks: list[tuple[Any, ...]], n_jobs: int,
               on_result: Callable[[Any], None], cancel: Any,
               on_idle: Callable[[], None] | None = None) -> bool:
    """Run ``fn(*args)`` for every task (in-process for n_jobs == 1); returns True when
    cancelled. On cancel, tasks that have not started are dropped and running ones finish
    (a worker is never killed in the middle of a store write)."""
    from concurrent.futures import FIRST_COMPLETED, wait

    if n_jobs == 1:
        for args in tasks:
            on_result(fn(*args))
            if on_idle:
                on_idle()
            if _is_cancelled(cancel):
                return True
        return False
    cancelled = False
    with _executor(n_jobs) as pool:
        pending = {pool.submit(fn, *args) for args in tasks}
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for fut in done:
                if not fut.cancelled():
                    on_result(fut.result())
            if on_idle:
                on_idle()
            if not cancelled and _is_cancelled(cancel):
                cancelled = True
                for fut in pending:
                    fut.cancel()                    # not yet started: dropped
                pending = {f for f in pending if not f.cancelled()}
    return cancelled


def _is_cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if isinstance(cancel, threading.Event):
        return cancel.is_set()
    return bool(cancel())


def performance_cores() -> int | None:
    """Physical cores of the fastest kind (P-cores on hybrid Intel; every core otherwise),
    from sysfs; None when unknown."""
    base = Path("/sys/devices/system/cpu")
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None
    freq: dict[int, int] = {}
    for c in cpus:
        try:
            freq[c] = int((base / f"cpu{c}/cpufreq/cpuinfo_max_freq").read_text())
        except (OSError, ValueError):
            return None
    best = max(freq.values()) if freq else 0
    cores = set()
    for c in cpus:
        if freq[c] < 0.85 * best:          # favoured cores turbo a little higher
            continue
        try:
            cores.add(((base / f"cpu{c}/topology/physical_package_id").read_text().strip(),
                       (base / f"cpu{c}/topology/core_id").read_text().strip()))
        except OSError:
            return None
    return len(cores) or None


def default_jobs(n_units: int) -> int:
    """Default workers of a run: one per participant chain, at most the performance cores
    + 4 (a chain uses about 0.7 GB; SMT siblings and E-cores add little), at least 1."""
    p = performance_cores() or max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(int(n_units), p + 4))


def run(
    config: StudyConfig,
    stages: Iterable[str] | None = None,
    pids: Iterable[str] | None = None,
    *,
    force: bool | Iterable[str] = False,
    n_jobs: int | None = 1,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | Callable[[], bool] | None = None,
    scheduler: Literal["participant", "stage"] = "participant",
) -> RunSummary:
    """Bring the requested stages (and their dependencies) up to date.

    @param force   True re-runs everything requested; an iterable re-runs only those stages.
    @param n_jobs  workers for participant stages (1 = in-process; None =
                   :func:`default_jobs` of the participants run).
    @param cancel  checked between units; a cancelled run keeps what finished.
    @param scheduler  "participant" (default): each worker runs one participant's chain of
                   participant stages with shared sessions/EMG preparation, cohort stages
                   start as soon as their inputs are complete (concurrently); "stage":
                   one stage at a time across participants (the pre-R3 order). Both give
                   identical stores.
    """
    validate_config(config)
    cohort = load_cohort(config.cohort_path, config.data_dir)
    problems = cohort.validate()
    if problems:
        raise ValueError("cohort manifest invalid:\n  " + "\n  ".join(problems))
    order = resolve_order(stages, config.groups)
    forced = set(order) if force is True else set(force or ())
    participants = cohort.included(pids)
    if n_jobs is None:
        n_jobs = default_jobs(len(participants))
    store = ResultStore(config.out_dir)
    keys = KeyResolver(config, cohort, store)

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    rdir = store.run_dir(run_id)
    emit = _Events(rdir / "events.jsonl", on_event)
    manifest: dict[str, Any] = {
        "run_id": run_id, "study": config.study, "started_utc": run_id[:16],
        "config_source": str(config.source) if config.source else None,
        "config_hash": config.hash, "data_dir": str(config.data_dir),
        "out_dir": str(config.out_dir), "cohort": str(config.cohort_path),
        "stages": order, "participants": [p.pid for p in participants],
        "git": _git_state(), "versions": _versions(),
    }
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    emit("run_start", run_id=run_id, stages=order, participants=manifest["participants"])

    statuses: dict[str, dict[str, str]] = {}
    cancelled = False
    lock = store.acquire_run_lock(run_id)

    def current(stage_name: str, unit: str) -> bool:
        """Upstream unit exists *and* matches the key it would have now."""
        return store.status(stage_name, unit, keys.key(stage_name, unit)) == "up-to-date"

    try:
        runner = _run_participant_major if scheduler == "participant" else _run_stages
        cancelled = runner(order, participants, cohort, config, store, keys, forced,
                           n_jobs, run_id, emit, cancel, statuses, current)
    finally:
        lock.release()
    summary = RunSummary(run_id=run_id, statuses=statuses, cancelled=cancelled)
    manifest.update(finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    cancelled=cancelled, statuses=statuses)
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    emit("cancelled" if cancelled else "run_end", n_failed=summary.n_failed)
    return summary


def _run_stages(order: list[str], participants: list[Participant], cohort: Cohort,
                config: StudyConfig, store: ResultStore, keys: KeyResolver, forced: set[str],
                n_jobs: int, run_id: str, emit: _Events, cancel: Any,
                statuses: dict[str, dict[str, str]],
                current: Callable[[str, str], bool]) -> bool:
    """Execute the stages in order; returns True when cancelled."""
    for name in order:
        if _is_cancelled(cancel):
            return True
        st = get_stage(name)
        statuses[name] = {}
        if st.scope == "participant":
            todo: list[tuple[str, str]] = []
            for p in participants:
                if not _inputs_present(cohort, p):
                    statuses[name][p.pid] = "skipped"
                    emit("unit_skipped", stage=name, unit=p.pid, reason="input session missing")
                    continue
                blocked = [d for d in st.depends if not current(
                    d, p.pid if get_stage(d).scope == "participant" else COHORT_UNIT)]
                if blocked:
                    statuses[name][p.pid] = "skipped"
                    emit("unit_skipped", stage=name, unit=p.pid, reason=f"upstream {blocked}")
                    continue
                key = keys.key(name, p.pid)
                if name not in forced and store.status(name, p.pid, key) == "up-to-date":
                    statuses[name][p.pid] = "cached"
                else:
                    todo.append((p.pid, key))
            emit("stage_start", stage=name, n_todo=len(todo),
                 n_cached=sum(v == "cached" for v in statuses[name].values()))
            count = [0]

            def got(r: tuple[str, str, float, str | None], name: str = name) -> None:
                pid, status, dt, err = r
                count[0] += 1
                statuses[name][pid] = status
                emit("unit_done", stage=name, unit=pid, status=status,
                     elapsed_s=round(dt, 3), error=err, done=count[0], total=len(todo))

            stop = _map_tasks(_exec_participant,
                              [(name, config, cohort, pid, key, run_id, n_jobs != 1)
                               for pid, key in todo], n_jobs, got, cancel)
            if stop:
                for pid, _ in todo:
                    statuses[name].setdefault(pid, "cancelled")
                emit("stage_end", stage=name, statuses=statuses[name])
                return True
        else:
            # A cohort stage spans every included participant whose upstream is current;
            # participants with failed upstream / missing input are excluded and reported.
            why = _cohort_blocked(name, keys, current)
            if why:
                statuses[name][COHORT_UNIT] = "skipped"
                emit("unit_skipped", stage=name, unit=COHORT_UNIT, reason=why)
            else:
                _exec_cohort(name, config, cohort, store, keys, forced, run_id, emit, statuses,
                             cancel)
        emit("stage_end", stage=name, statuses=statuses[name])
    return False


def _exec_cohort(name: str, config: StudyConfig, cohort: Cohort, store: ResultStore,
                 keys: KeyResolver, forced: set[str], run_id: str, emit: _Events,
                 statuses: dict[str, dict[str, str]], cancel: Any = None) -> str:
    """Run (or reuse) one cohort stage on its available participants; returns its status.

    The stage sees only the available participants (``KeyResolver.coverage``); a
    ``coverage`` table (pid, used, reason) is stored with its results."""
    prep = _cohort_prepare(name, store, keys, forced, emit, statuses)
    if prep is None:
        return "cached"
    status, dt, err = _cohort_body(name, config, cohort, *prep, run_id, cancel)
    _cohort_finish(name, status, dt, err, emit, statuses)
    return status


def _cohort_prepare(name: str, store: ResultStore, keys: KeyResolver, forced: set[str],
                    emit: _Events, statuses: dict[str, dict[str, str]]
                    ) -> tuple[str, list[str], dict[str, str]] | None:
    """Parent side of a cohort unit: (key, available pids, excluded) to run, or None when
    it is cached (status recorded)."""
    key = keys.key(name, COHORT_UNIT)
    avail, excluded, _ = keys.coverage(name)
    if excluded:
        emit("cohort_excluded", stage=name, excluded=excluded)
    if name not in forced and store.status(name, COHORT_UNIT, key) == "up-to-date":
        statuses[name][COHORT_UNIT] = "cached"
        emit("stage_start", stage=name, n_todo=0, n_cached=1)
        return None
    emit("stage_start", stage=name, n_todo=1, n_cached=0)
    return key, avail, excluded


def _cohort_finish(name: str, status: str, dt: float, err: str | None, emit: _Events,
                   statuses: dict[str, dict[str, str]]) -> None:
    statuses[name][COHORT_UNIT] = status
    emit("unit_done", stage=name, unit=COHORT_UNIT, status=status, elapsed_s=round(dt, 3),
         error=err, done=1, total=1)


def _partial_note(store: ResultStore, st: Stage, pid: str) -> str:
    """Coverage reason of a used participant without a unit of a ``partial`` dependency
    (E32): it entered every input except those."""
    lacks = [d for d in st.partial
             if (m := store.meta(d, pid)) is None or m.status != "ok"]
    return f"used without {', '.join(lacks)} (no result)" if lacks else ""


def _cohort_body(name: str, config: StudyConfig, cohort: Cohort, key: str, avail: list[str],
                 excluded: dict[str, str], run_id: str, cancel: Any = None
                 ) -> tuple[str, float, str | None]:
    """Run one cohort unit and store it; (status, elapsed_s, error). Top-level so a worker
    process can run it (then without a cancel token)."""
    st = get_stage(name)
    store = ResultStore(config.out_dir)
    ctx = CohortContext(config=config, cohort=cohort, params=config.params(name),
                        store=store, participants=[cohort.get(p) for p in avail],
                        excluded=excluded, cancel=cancel)
    t0 = time.perf_counter()
    try:
        res = st.func(ctx)
        dt = time.perf_counter() - t0
        cov = pd.DataFrame([{"pid": p, "used": True, "reason": _partial_note(store, st, p)}
                            for p in avail]
                           + [{"pid": p, "used": False, "reason": r}
                              for p, r in excluded.items()])
        tables = {**res.tables, "coverage": cov}
        store.write(name, COHORT_UNIT, key, tables=tables, arrays=res.arrays,
                    run_id=run_id, elapsed_s=dt,
                    extra={**(res.extra or {}), "excluded": excluded})
        status, err = "ran", None
    except Exception:
        e = traceback.format_exc()
        dt = time.perf_counter() - t0
        if ctx.cancelled():
            # Stopped by the cancel token (check_cancelled): not a failure, and nothing is
            # written -- the unit stays as it was (missing or stale) and re-runs next time.
            status, err = "cancelled", None
        else:
            try:
                store.write(name, COHORT_UNIT, key, status="failed", error=e, run_id=run_id,
                            elapsed_s=dt)
            except Exception:
                pass
            status, err = "failed", e.strip().splitlines()[-1]
    return status, dt, err


class _TaskPool:
    """A ready queue in front of the worker pool: at most ``n_jobs`` tasks in flight; when a
    worker frees up, the ready task with the lowest priority tuple starts (ties: order
    added). Tasks never queue inside the executor, so "not started" means "still here" and
    is dropped on cancel. ``n_jobs == 1``: no pool, :meth:`wait` runs one task in-process.
    """

    def __init__(self, n_jobs: int) -> None:
        import heapq
        import itertools

        self._heapq = heapq
        self._seq = itertools.count()
        self.serial = n_jobs == 1
        self.n = n_jobs if n_jobs and n_jobs > 0 else (os.cpu_count() or 1)
        self._ex = None if self.serial else _executor(n_jobs)
        self._ready: list[tuple[Any, ...]] = []
        self.inflight: dict[Any, Any] = {}

    @property
    def processes(self) -> bool:
        from concurrent.futures import ProcessPoolExecutor

        return isinstance(self._ex, ProcessPoolExecutor)

    def add(self, prio: tuple[float, ...], tag: Any, fn: Callable[..., Any], *args: Any) -> None:
        self._heapq.heappush(self._ready, (prio, next(self._seq), tag, fn, args))

    def busy(self) -> bool:
        return bool(self._ready or self.inflight)

    def drop_ready(self) -> list[Any]:
        tags = [item[2] for item in self._ready]
        self._ready.clear()
        return tags

    def wait(self, timeout: float = 1.0) -> list[tuple[Any, Any]]:
        """Start ready tasks into free workers; return the (tag, result) pairs that finished
        within ``timeout`` (serial: run the next task)."""
        from concurrent.futures import FIRST_COMPLETED, wait

        if self.serial:
            if not self._ready:
                return []
            _, _, tag, fn, args = self._heapq.heappop(self._ready)
            return [(tag, fn(*args))]
        while self._ready and len(self.inflight) < self.n:
            _, _, tag, fn, args = self._heapq.heappop(self._ready)
            self.inflight[self._ex.submit(fn, *args)] = tag
        if not self.inflight:
            return []
        done, _ = wait(list(self.inflight), timeout=timeout, return_when=FIRST_COMPLETED)
        return [(self.inflight.pop(f), f.result()) for f in done]

    def close(self) -> None:
        if self._ex is not None:
            self._ex.shutdown(wait=True)              # running tasks finish, never killed


def _segment_writes(plan: list[tuple[str, str, str | None]], done: dict[str, str]) -> bool:
    """Would :func:`_exec_segment` run or store anything for ``plan`` (a todo stage, or a
    stage skipped because an upstream did not succeed)? ``done``: upstream statuses."""
    done = dict(done)
    for name, what, _ in plan:
        if any(get_stage(d).scope == "participant"
               and done.get(d, "not current") not in ("ran", "cached")
               for d in get_stage(name).depends) or what == "todo":
            return True
        done[name] = "cached"
    return False


def _previous_elapsed(store: ResultStore, stages: list[str], pids: list[str]
                      ) -> Callable[[list[str], str], float]:
    """Expected seconds of a participant's stages from the elapsed_s of the units stored by
    earlier runs (stage mean for units never run, 0 without any): longest tasks first keeps
    one slow chain from ending the wave alone."""
    prev: dict[tuple[str, str], float] = {}
    for n in stages:
        for pid in pids:
            m = store.meta(n, pid)
            if m is not None and m.elapsed_s:
                prev[(n, pid)] = float(m.elapsed_s)
    mean = {n: sum(v) / len(v) for n in stages
            if (v := [s for (st, _), s in prev.items() if st == n])}
    return lambda names, pid: sum(prev.get((n, pid), mean.get(n, 0.0)) for n in names)


def _plugin_cohort_todo(cstages: list[str], store: ResultStore, keys: KeyResolver,
                        forced: set[str]) -> bool:
    """Will a cohort stage of a plugin (e.g. ``ml_eval``, which runs its own worker pool)
    run in this run? (Not up to date at the start counts as yes.)"""
    for n in cstages:
        if get_stage(n).func.__module__.split(".")[0] == "biomech":
            continue
        if n in forced or store.status(n, COHORT_UNIT, keys.key(n, COHORT_UNIT)) != "up-to-date":
            return True
    return False


def _run_participant_major(order: list[str], participants: list[Participant], cohort: Cohort,
                           config: StudyConfig, store: ResultStore, keys: KeyResolver,
                           forced: set[str], n_jobs: int, run_id: str, emit: _Events,
                           cancel: Any, statuses: dict[str, dict[str, str]],
                           current: Callable[[str, str], bool]) -> bool:
    """Participant-major execution (see :func:`run`); returns True when cancelled.

    Participant stages are grouped into waves: wave k holds the participant stages whose
    cohort-stage dependencies all lie in earlier waves (with the built-in stages there is
    one wave). Per wave, every participant's chain is one task (worker processes, or
    in-process one at a time when ``n_jobs == 1``). In parallel runs a chain is split: its
    ``detach`` stages (and stages depending on them) form a second task, queued when the
    first part returns, so cohort stages that need only the first part (``ml_eval``) start
    once every first part is done; first parts run before second parts, each longest first
    by the elapsed times of the previous run (:class:`_TaskPool`). A cohort stage starts as
    soon as all units of its dependencies are done, in a thread -- or, when no plugin cohort
    stage (``ml_eval``, with its own worker pool) runs in this run, as a task in the worker
    pool (biomech cohort stages; no GIL contention between them). It runs on the
    participants whose upstream is current and records the others (``_exec_cohort``).
    Cancellation is checked about every second: tasks not yet started are dropped, running
    participant tasks and cohort stages finish (long cohort stages poll
    ``ctx.cancelled()``), and units that never ran are reported as "cancelled".
    """
    from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

    if _is_cancelled(cancel):
        return True
    pstages = [n for n in order if get_stage(n).scope == "participant"]
    cstages = [n for n in order if get_stage(n).scope == "cohort"]
    wave: dict[str, int] = {}
    for n in order:                                       # dependencies come first
        wave[n] = max([wave[d] + (get_stage(d).scope == "cohort")
                       for d in get_stage(n).depends if d in wave], default=0)
    n_waves = max([wave[n] for n in pstages], default=-1) + 1
    present = [p for p in participants if _inputs_present(cohort, p)]
    missing = [p for p in participants if p not in present]

    parallel = n_jobs != 1
    pool = _TaskPool(n_jobs)
    threads = ThreadPoolExecutor(max_workers=max(2, len(cstages)))
    running: dict[Future, str] = {}                       # cohort stages in threads
    in_pool: set[str] = set()                             # cohort stages in the worker pool
    cohort_tasks = (parallel and pool.processes
                    and not _plugin_cohort_todo(cstages, store, keys, forced))
    expect = _previous_elapsed(store, pstages, [p.pid for p in present]) if parallel \
        else (lambda names, pid: 0.0)
    started_c: set[str] = set()
    finished: set[str] = set()
    pending: dict[str, set[str]] = {}                     # stage -> pids not yet returned
    todo_total: dict[str, int] = {}
    task_stages: dict[tuple[str, str], list[str]] = {}
    tails: dict[str, tuple[list, dict, dict]] = {}        # pid -> (plan, ext, head plan)
    stopping = [False]

    def submit(kind: str, plan: list, pid: str, ext: dict) -> None:
        task_stages[(kind, pid)] = [n for n, _, _ in plan]
        todo = [n for n, what, _ in plan if what == "todo"]
        prio = (1 if kind == "head" else 2, -expect(todo, pid)) if parallel else (0,)
        pool.add(prio, (kind, pid), _exec_segment, plan, config, cohort, pid, run_id,
                 parallel, ext)

    def plan_wave(w: int) -> None:
        names = [n for n in pstages if wave[n] == w]
        blocked = {n for n in names for d in get_stage(n).depends
                   if get_stage(d).scope == "cohort" and not current(d, COHORT_UNIT)}
        detached: set[str] = set()
        if parallel:                                      # names are in dependency order
            for n in names:
                if get_stage(n).detach or any(d in detached for d in get_stage(n).depends):
                    detached.add(n)
        for n in names:
            statuses[n] = {}
            pending[n] = set()
            for p in missing:
                statuses[n][p.pid] = "skipped"
                emit("unit_skipped", stage=n, unit=p.pid, reason="input session missing")
        for p in present:
            pl, mine = [], []
            for n in names:
                if n in blocked:
                    continue
                key = keys.key(n, p.pid)
                cached = n not in forced and store.status(n, p.pid, key) == "up-to-date"
                pl.append((n, "cached" if cached else "todo", key))
                mine.append(n)
            # Participant-scope dependencies outside this plan (earlier waves, or stages
            # not requested in this run): current or not, checked here, not assumed.
            ext = {d: current(d, p.pid) for n in mine for d in get_stage(n).depends
                   if get_stage(d).scope == "participant" and d not in mine}
            if any(what == "todo" for _, what, _ in pl):
                for n in mine:
                    pending[n].add(p.pid)
                head = [x for x in pl if x[0] not in detached]
                tail = [x for x in pl if x[0] in detached]
                if any(what == "todo" for _, what, _ in head):
                    submit("head", head, p.pid, ext)
                    if tail:
                        tails[p.pid] = (tail, ext, head)
                else:                                     # nothing to run before the tail
                    submit("tail", tail, p.pid, {**ext, **{n: "cached" for n, _, _ in head}})
            for n, what, _ in pl:
                if what == "cached":
                    statuses[n][p.pid] = "cached"
        for n in names:
            if n in blocked:
                for p in present:
                    statuses[n][p.pid] = "skipped"
                    emit("unit_skipped", stage=n, unit=p.pid, reason="upstream cohort stage")
                continue
            todo = [pid for pid in pending[n] if statuses[n].get(pid) != "cached"]
            todo_total[n] = len(todo)
            emit("stage_start", stage=n, n_todo=len(todo), n_cached=len(present) - len(todo))

    def collect_segment(kind: str, r: tuple[str, list]) -> None:
        pid, results = r
        got: dict[str, str] = {}
        for name, status, dt, err in results:
            statuses[name][pid] = status
            got[name] = status
            if status == "skipped":
                emit("unit_skipped", stage=name, unit=pid, reason=err)
            else:
                emit("unit_done", stage=name, unit=pid, status=status, elapsed_s=round(dt, 3),
                     error=err, total=todo_total.get(name, 0),
                     done=sum(v in ("ran", "failed") for v in statuses[name].values()))
        for n in task_stages.pop((kind, pid)):
            pending[n].discard(pid)
        if kind == "head" and pid in tails:
            tail, ext, head = tails.pop(pid)
            done = {**ext, **{n: got.get(n, "cached") for n, _, _ in head}}
            if not stopping[0] and _segment_writes(tail, {d: v if isinstance(v, str) else (
                    "cached" if v else "not current") for d, v in done.items()}):
                submit("tail", tail, pid, done)
            elif not stopping[0]:
                for n, _, _ in tail:                      # all cached, nothing to store
                    pending[n].discard(pid)

    def end_finished_pstages() -> None:
        for n in pstages:
            if n in statuses and n not in finished and not pending.get(n):
                finished.add(n)
                emit("stage_end", stage=n, statuses=statuses[n])

    def launch_cohort() -> None:
        if stopping[0]:
            return
        for n in cstages:
            if n in started_c or not all(d in finished for d in get_stage(n).depends):
                continue
            started_c.add(n)
            statuses[n] = {}
            why = _cohort_blocked(n, keys, current)
            if why:
                statuses[n][COHORT_UNIT] = "skipped"
                emit("unit_skipped", stage=n, unit=COHORT_UNIT, reason=why)
            elif not parallel:
                _exec_cohort(n, config, cohort, store, keys, forced, run_id, emit, statuses,
                             cancel)
            elif cohort_tasks and get_stage(n).func.__module__.split(".")[0] == "biomech":
                prep = _cohort_prepare(n, store, keys, forced, emit, statuses)
                if prep is not None:
                    pool.add((0,), ("cohort", n), _cohort_body, n, config, cohort, *prep,
                             run_id)
                    in_pool.add(n)
                    continue
            else:
                running[threads.submit(_exec_cohort, n, config, cohort, store, keys, forced,
                                       run_id, emit, statuses, cancel)] = n
                continue
            finished.add(n)
            emit("stage_end", stage=n, statuses=statuses[n])

    def handle(tag: tuple[str, str], r: Any) -> None:
        kind, what = tag
        if kind == "cohort":
            _cohort_finish(what, *r, emit, statuses)
            in_pool.discard(what)
            finished.add(what)
            emit("stage_end", stage=what, statuses=statuses[what])
        else:
            collect_segment(kind, r)

    def reap(block: bool) -> None:
        """Collect finished cohort threads (waiting up to 1 s for one if ``block``)."""
        if not running:
            return
        done, _ = wait(list(running), timeout=1.0 if block else 0,
                       return_when=FIRST_COMPLETED)
        for fut in done:
            n = running.pop(fut)
            fut.result()
            finished.add(n)
            emit("stage_end", stage=n, statuses=statuses[n])

    def idle() -> None:
        end_finished_pstages()
        reap(block=False)
        launch_cohort()

    def step() -> None:
        """Wait about a second for a worker task (or, with none, a cohort thread)."""
        if pool.busy():
            for tag, r in pool.wait(1.0):
                handle(tag, r)
        elif running:
            reap(block=True)
        idle()

    def check_cancel() -> bool:
        if not stopping[0] and _is_cancelled(cancel):
            stopping[0] = True
            pool.drop_ready()                             # not yet started: dropped
        return stopping[0]

    cancelled = False
    clean = False
    try:
        for w in range(n_waves):
            plan_wave(w)
            idle()
            while pool.busy():
                step()
                check_cancel()
            if stopping[0]:
                break
            # Cohort stages that a later wave needs must finish first.
            while (running or in_pool) and any(get_stage(d).scope == "cohort"
                                               and d not in finished
                                               for n in pstages if wave[n] == w + 1
                                               for d in get_stage(n).depends):
                step()
                if check_cancel():
                    break
            if stopping[0]:
                break
        if not stopping[0]:
            idle()
            while running or pool.busy():
                step()
                if check_cancel():
                    break
        clean = True
    finally:
        cancelled = stopping[0]
        pool.drop_ready()
        try:
            while clean and pool.inflight:                # running tasks finish and are kept
                for tag, r in pool.wait(1.0):
                    handle(tag, r)
        finally:
            pool.close()
            if running:                                   # cohort threads finish, never killed
                wait(list(running))
                for fut, n in list(running.items()):
                    fut.result()
                    finished.add(n)
                    emit("stage_end", stage=n, statuses=statuses[n])
                running.clear()
            threads.shutdown(wait=True)
        if cancelled:
            for n in pstages:
                for pid in pending.get(n, set()):
                    statuses.setdefault(n, {}).setdefault(pid, "cancelled")
            for n in cstages:
                if n not in started_c:
                    statuses.setdefault(n, {})[COHORT_UNIT] = "cancelled"
    return cancelled


def stage_status(config: StudyConfig, stages: Iterable[str] | None = None, *,
                 read: bool = False) -> pd.DataFrame:
    """Status matrix (stage x unit) against the current config and code: the control-center
    view. ``read``: of the store the config reads (a frozen snapshot: is it still what the
    current code would compute?) instead of the live one."""
    validate_config(config)
    cohort = load_cohort(config.cohort_path, config.data_dir)
    store = ResultStore(config.read_dir if read else config.out_dir)
    keys = KeyResolver(config, cohort, store)
    missing = {p.pid for p in cohort.included() if not _inputs_present(cohort, p)}
    rows = []
    for name in resolve_order(stages, config.groups):
        st = get_stage(name)
        units = ([p.pid for p in cohort.included()] if st.scope == "participant"
                 else [COHORT_UNIT])
        for unit in units:
            if unit in missing:
                status = "missing-input"
            else:
                status = store.status(name, unit, keys.key(name, unit))
            rows.append({"stage": name, "unit": unit, "status": status})
    return pd.DataFrame(rows)
