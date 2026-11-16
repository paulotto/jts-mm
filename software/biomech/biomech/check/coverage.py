# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Which stored outputs does no view show? (every result in the viewer)

:func:`stored_outputs` lists what the pipeline wrote (tables as ``(stage, table)``, arrays
as ``(stage, family)``); :func:`view_reads` builds every registered view -- every declared
phase and perspective, the first participant -- under :func:`biomech.results.access.track`
and collects what they read. :func:`uncovered` is the difference minus :data:`INTERNAL`,
the outputs that are inputs of other stages rather than results (each with its reason).
``check-outputs`` reports the result and tests/test_view_coverage.py fails on it.
"""

from __future__ import annotations

import fnmatch
import warnings
from typing import Any

from biomech.results import COHORT_UNIT, ResultStore
from biomech.results.access import AccessLog, family, track

# (stage, table-or-array-family glob) -> why no view needs to show it.
INTERNAL: dict[tuple[str, str], str] = {
    ("ml_windows", "X_*"): "window feature matrices: input of ml_eval",
    ("ml_windows", "y"): "window labels: input of ml_eval",
    ("ml_windows", "t_end_us"): "window time stamps: input of ml_eval",
    ("ml_windows", "phase"): "window phase index: input of ml_eval",
    ("ml_windows", "norm_*"): "feature standardization: input of ml_eval",
    ("emg", "spectral__{task}__win"): "per-window spectral shape: input of stats_emg (SPM)",
    ("kinematics", "spec"): "specification-curve inputs: shown via variability:spec_values",
    ("emg", "spec"): "specification-curve inputs: shown via variability:spec_values",
    ("kinematics", "{task}__T"): "pose matrices: drawn as the 3-D jaw (data_jaw3d via landmarks)",
    ("kinematics", "T_max_hp"): "maxilla pose at the landmark hold: frame of the 3-D views",
    ("kinematics", "T_closed"): "closed-pose reference: frame of the hinge views",
    ("stats_emg", "spm1__left"): "test inputs behind spm1__z",
    ("stats_emg", "spm1__right"): "test inputs behind spm1__z",
    ("stats_emg", "spm2__openers"): "test inputs behind spm2__z (Fig. 2b shows the profiles)",
    ("stats_emg", "spm2__closers"): "test inputs behind spm2__z (Fig. 2b shows the profiles)",
    ("qc", "scene__*__*__*"): "per-phase tracking scene: data_tracking_scene draws the chosen "
                              "phase (check builds the whole take, scene__all__*)",
}


EXEMPT_PATH = __import__("pathlib").Path(__file__).with_name("coverage_exempt.yaml")


def _exempt_rules() -> list[tuple[str, str, str, str]]:
    import yaml

    doc = yaml.safe_load(EXEMPT_PATH.read_text(encoding="utf-8")) if EXEMPT_PATH.is_file() else {}
    out = []
    for r in (doc or {}).get("columns") or []:
        out.append((str(r.get("stage", "*")), str(r.get("table", "*")), str(r["column"]),
                    str(r.get("why", ""))))
    return out


def column_exempt(stage: str, table: str, column: str) -> str | None:
    """Why a stored column needs no artefact (bookkeeping: ids, keys, flags), else None."""
    for st, tb, col, why in _exempt_rules():
        if (fnmatch.fnmatchcase(stage, st) and fnmatch.fnmatchcase(table, tb)
                and fnmatch.fnmatchcase(column, col)):
            return why or "exempt"
    return None


def _internal(stage: str, name: str) -> str | None:
    for (st, pat), why in INTERNAL.items():
        if st == stage and fnmatch.fnmatchcase(name, pat):
            return why
    return None


def _stage_groups() -> dict[str, str]:
    from biomech.pipeline import registry

    return {n: s.group for n, s in registry().items()}


def stored_columns(store: ResultStore, tables: set[tuple[str, str]]
                   ) -> set[tuple[str, str, str]]:
    """Every column of the stored ``tables`` (union over units; read from the parquet
    schema, no data loaded)."""
    import pyarrow.parquet as pq

    out: set[tuple[str, str, str]] = set()
    for stage, name in tables:
        for unit in store.units(stage):
            f = store._unit_dir(stage, unit) / f"{name}.parquet"
            if f.is_file():
                out.update((stage, name, c) for c in pq.read_schema(f).names)
    return out


def stored_outputs(store: ResultStore, groups: tuple[str, ...] | None = None
                   ) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """(tables, array families) present in the store (ok units only); with ``groups`` only
    those of stages in these study groups (unregistered stages count)."""
    tables: set[tuple[str, str]] = set()
    arrays: set[tuple[str, str]] = set()
    sg = _stage_groups() if groups is not None else {}
    for stage in store.stages():
        if groups is not None and sg.get(stage, groups[0]) not in groups:
            continue
        for unit in store.units(stage):
            m = store.meta(stage, unit)
            if m is None or m.status != "ok":
                continue
            tables.update((stage, t) for t in m.tables)
            if m.has_arrays:
                arrays.update((stage, family(k)) for k in store.read_arrays(stage, unit).keys())
    return tables, arrays


def _variants(v: Any, ctx: Any, pids: list[str]) -> list[dict[str, Any]]:
    """Every phase x perspective at default pickers, plus every picker value once."""
    from biomech.viz.views import registry as R

    variants: list[dict[str, Any]] = []
    for ph in R.expand_phases(v, ctx):
        for persp in (v.perspectives or (None,)):
            kw: dict[str, Any] = {}
            if "pid" in v.params and pids:
                kw["pid"] = pids[0]
            if ph is not None:
                kw["phase"] = ph
            if persp is not None:
                kw["perspective"] = persp
            variants.append(kw)
    base = dict(variants[0]) if variants else {}
    for name in v.choices:
        for val in v.choice_values(name, ctx):
            variants.append({**base, name: val})
    return variants


def _view_reads_one(ctx: Any, vid: str, prebuilt: dict[tuple, tuple] | None = None
                    ) -> tuple[set, set, list[tuple[str, str]], set]:
    """Reads (tables, arrays), build errors and read columns of every variant of one view.
    A variant whose request was already built elsewhere (``prebuilt``: request key ->
    (tables, arrays, error, columns), from ``check_figures``) is not built again; its
    recorded reads count."""
    from biomech.check.outputs import _req_key
    from biomech.viz.views import registry as R

    v = R.get(vid)
    tables: set = set()
    arrays: set = set()
    columns: set = set()
    errors: list[tuple[str, str]] = []
    with track() as log:                          # picker values may read the store too
        variants = _variants(v, ctx, ctx.pids())
    tables |= log.tables
    arrays |= log.arrays
    columns |= log.columns
    for kw in variants:
        with track() as log:
            try:
                req = R.request(ctx, v.id, **kw)
                hit = (prebuilt or {}).get(_req_key(v.id, req))
                if hit is None:
                    R.build(v.id, req)
            except Exception as exc:              # reported, never hidden
                hit = None
                errors.append((f"{v.id} {kw}", f"{type(exc).__name__}: {exc}"))
        tables |= log.tables
        arrays |= log.arrays
        columns |= log.columns
        if hit is not None:
            tables |= hit[0]
            arrays |= hit[1]
            if hit[2] is not None:
                errors.append((f"{v.id} {kw}", hit[2]))
            if len(hit) > 3:
                columns |= hit[3]
    return tables, arrays, errors, columns


def _reads_task(config: Any, cohort: Any, vids: list[str], prebuilt: dict[tuple, tuple]
                ) -> list[tuple[str, tuple]]:
    from biomech.check.outputs import _worker_ctx

    ctx = _worker_ctx(config, cohort)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [(vid, _view_reads_one(ctx, vid, prebuilt)) for vid in vids]


def view_reads(ctx: Any, kinds: tuple[str, ...] = ("paper", "dashboard", "data"), *,
               prebuilt: dict[tuple, tuple] | None = None, n_jobs: int = 1,
               groups: Any = None) -> tuple[AccessLog, list[tuple[str, str]]]:
    """Build every view variant (views of the study's groups: ``groups``, default those of
    ``ctx``) and record its reads; returns (log, build errors).

    ``prebuilt`` (request key -> reads, from ``check_figures``): variants built there are
    not built again. ``n_jobs`` (1 = in-process, 0 = up to 8): views in worker processes;
    reads and errors are merged in view order (the same result)."""
    from biomech.viz.views import registry as R

    views = [v.id for v in R.views(groups=ctx if groups is None else groups)
             if v.kind in kinds]
    results: dict[str, tuple] = {}
    if n_jobs != 1 and len(views) > 1:
        from biomech.check.outputs import _pool

        pool, n = _pool(n_jobs)
        with pool:
            futs = [pool.submit(_reads_task, ctx.config, ctx.cohort, c, prebuilt or {})
                    for c in (views[k::n * 2] for k in range(n * 2)) if c]
            for f in futs:
                results.update(f.result())
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = {vid: _view_reads_one(ctx, vid, prebuilt) for vid in views}
    log = AccessLog()
    errors: list[tuple[str, str]] = []
    for vid in views:
        t, a, e, c = results[vid]
        log.tables |= t
        log.arrays |= a
        log.columns |= c
        for col in c:
            log.consumers.setdefault(col, set()).add(vid)
        errors += e
    return log, errors


def uncovered(store: ResultStore, log: AccessLog, groups: tuple[str, ...] | None = None
              ) -> dict[str, list[tuple[str, str]]]:
    """Stored tables / array families (of stages in ``groups``) that no view read and that
    are not INTERNAL."""
    tables, arrays = stored_outputs(store, groups)
    read_t = log.tables
    read_a = log.families()
    out_t = sorted(t for t in tables - read_t if not _internal(*t))
    out_a = sorted(a for a in arrays - read_a if not _internal(*a))
    # Columns: of the tables some view shows, every column must be
    # used by a view, unless it is bookkeeping (check/coverage_exempt.yaml).
    shown = {t for t in tables if t in read_t and not _internal(*t)}
    cols = stored_columns(store, shown)
    out_c = sorted(c for c in cols - log.columns if not column_exempt(*c))
    # Every stored table has its complete table view (etc/tables.yaml): a new stage table
    # must be declared with a caption before it counts as shown.
    from biomech.viz.views.tables import declarations

    decl = set(declarations())
    out_d = sorted(f"{st}:{t}" for st, t in tables if f"{st}:{t}" not in decl)
    return {"tables": out_t, "arrays": out_a, "columns": out_c, "undeclared": out_d}


def report(ctx: Any, *, prebuilt: dict[tuple, tuple] | None = None,
           n_jobs: int = 1, groups: Any = None) -> dict[str, Any]:
    """Build errors and uncovered outputs of the study's groups (``groups``, default those
    of ``ctx``): a study's outputs must be shown by views of its own groups."""
    from biomech.viz.views import registry as R

    g = R.study_groups(ctx if groups is None else groups)
    log, errors = view_reads(ctx, prebuilt=prebuilt, n_jobs=n_jobs, groups=g)
    return {"errors": errors, "matrix": log.consumers, **uncovered(ctx.store, log, g)}


def write_matrix(matrix: dict[tuple[str, str, str], set[str]], path: Any) -> int:
    """The coverage matrix as CSV: per stored column the number of artefacts using it and
    their ids; returns the number of columns."""
    import csv
    from pathlib import Path

    rows = sorted(matrix.items())
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stage", "table", "column", "n_artefacts", "artefacts"])
        for (st, t, c), views in rows:
            w.writerow([st, t, c, len(views), " ".join(sorted(views))])
    return len(rows)


__all__ = ["INTERNAL", "COHORT_UNIT", "report", "stored_outputs", "uncovered", "view_reads",
           "write_matrix"]
