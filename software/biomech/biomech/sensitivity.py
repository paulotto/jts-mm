# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sensitivity runs: the whole pipeline under an alternative analysis choice (an08 §1).

A variant is a named set of dotted config overrides in the ``sensitivity`` block of
``biomech.yaml`` (or the study file), e.g. ``lowfreq_mask`` = the pre-D14 masking, and/or a
cohort filter ``exclude_flags`` (participants carrying one of these manifest flags are
excluded, e.g. ``reporting_population`` drops the ``outside_protocol`` participant; the filtered manifest is
written to the variant's directory and the variant reads it). ``skip_stages`` names stages the
variant does not re-run (with every stage downstream of them, :func:`skipped_stages`): their
tables are left out of the comparison, which lists them as not run (:func:`not_run`), so a
missing result is never mistaken for an unchanged one. It runs
into ``<out_dir>/sensitivity/<name>`` with its own cache, so the primary store is never
touched; :func:`seed_cache` first copies the primary's units into it, so only the stages
the overrides change re-run. :func:`compare` lines up every cohort table of the two stores:

* key columns = the columns that identify a row in both versions: text columns whose value
  sets are identical in both (task, muscle, test, rule, ...; a text column that changed, e.g. a
  cluster list, is dropped) and the numeric parameter columns of :data:`PARAM_COLUMNS`
  (``drop_db``, ``hold_s``, ``primary``, ``k``, ...), plus the row's position within its key
  when keys repeat;
* value columns = the other numeric and boolean columns both tables share;
* one output row per (stage, table, key, column) with the primary and variant value and
  ``match`` (both / primary_only / variant_only); identity columns (pid, task, test, ...)
  are always keys, so a row missing in one run is reported, never paired by position.

The comparison is written to ``<variant out_dir>/comparison.parquet``; the viewer and the
analysis-plan deviation log read it from there.

Reading a variant (the dataset paper reports the ``reporting_population`` store):
:func:`store_config` is the one place that maps (variant, snapshot) to the store a reader
uses -- ``biomech export``, ``numbers``, ``verify-export``, ``snapshot``, ``check-outputs`` and
the paper sets (:meth:`biomech.papers.PaperSet.store_config`) all go through it. A variant's
snapshots live in its own store (``<primary>/sensitivity/<name>/snapshots/<snap>``, taken with
``biomech snapshot --variant <name> <snap>``, which freezes the filtered cohort manifest and the
variant's comparison with it); :func:`comparison_path` finds a comparison in either store.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomech.config import ConfigError, StudyConfig, load_config
from biomech.results import COHORT_UNIT, ResultStore

COMPARISON_FILE = "comparison.parquet"
PRIMARY = "primary"                   # name of the primary analysis where a variant is asked for


def variants(cfg: StudyConfig) -> dict[str, dict[str, Any]]:
    return dict(cfg.raw.get("sensitivity") or {})


def primary_out_dir(cfg: StudyConfig) -> Path:
    """Root of the primary store of ``cfg`` (for a variant configuration: the store the variant
    directory ``<primary>/sensitivity/<name>`` sits in)."""
    return Path(cfg.out_dir).parent.parent if getattr(cfg, "variant", None) else \
        Path(cfg.out_dir)


def variant_dir(cfg: StudyConfig, name: str) -> Path:
    """Store root of variant ``name`` (its own cache, runs, manifest and snapshots)."""
    return primary_out_dir(cfg) / "sensitivity" / name


def variant_config(config_path: str | Path, name: str, *, data_dir: str | Path | None = None,
                   out_dir: str | Path | None = None) -> StudyConfig:
    """The configuration of variant ``name`` of the study ``config_path``: its overrides
    applied, ``out_dir`` = :func:`variant_dir` (``out_dir`` names the primary store root),
    the cohort filtered by its ``exclude_flags`` (written to the variant directory) and its
    ``cohort_swap`` (participants analysed from a ``secondary`` take, e.g. a second day).

    A variant with ``base: <other variant>`` (D36) adds its change to the base's: the
    base's overrides, flags and swaps first, then its own; :func:`base_config` is the store it
    is seeded from and compared with."""
    base = load_config(config_path, data_dir=data_dir, out_dir=out_dir)
    spec = variants(base).get(name)
    if spec is None:
        raise ConfigError(f"unknown sensitivity variant {name!r}; known: {sorted(variants(base))}")
    sets, flags, swap = _changes(variants(base), name)
    if not sets and not flags and not swap:
        raise ConfigError(f"sensitivity variant {name!r} sets nothing")
    skipped_stages(base, name)                      # unknown stage names fail here
    vcfg = load_config(config_path, data_dir=data_dir, out_dir=variant_dir(base, name),
                       overrides=sets or None)
    vcfg = dataclasses.replace(vcfg, variant=name)
    if flags or swap:
        vcfg = dataclasses.replace(vcfg, cohort_path=filtered_cohort(
            vcfg.cohort_path, vcfg.out_dir / "cohort.yaml", flags, name, swap))
    return vcfg


def base_name(cfg: StudyConfig, name: str) -> str | None:
    """The variant ``name`` builds on (``base:``), None for a variant of the primary."""
    b = (variants(cfg).get(name) or {}).get("base")
    return str(b) if b else None


def _changes(specs: dict[str, dict[str, Any]], name: str, seen: tuple[str, ...] = ()
             ) -> tuple[dict[str, Any], list[str], dict[str, int]]:
    """(overrides, exclude flags, cohort swaps) of variant ``name`` with those of its base
    chain (the base's first; the variant's own win)."""
    if name in seen:
        raise ConfigError(f"sensitivity variant {name!r}: circular base {' -> '.join(seen)}")
    spec = specs.get(name)
    if spec is None:
        raise ConfigError(f"sensitivity variant {seen[-1]!r}: unknown base {name!r}")
    sets: dict[str, Any] = {}
    flags: list[str] = []
    swap: dict[str, int] = {}
    if spec.get("base"):
        sets, flags, swap = _changes(specs, str(spec["base"]), (*seen, name))
    sets.update(spec.get("set") or {})
    flags += [str(f) for f in (spec.get("exclude_flags") or []) if str(f) not in flags]
    swap.update({str(k): int(v) for k, v in (spec.get("cohort_swap") or {}).items()})
    return sets, flags, swap


def base_config(cfg: StudyConfig, name: str) -> StudyConfig:
    """The store variant ``name`` is seeded from and compared with: its ``base`` variant's
    (D36: the paper's sensitivity runs build on the 13-adult store), else the primary
    ``cfg``."""
    b = base_name(cfg, name)
    if b is None:
        return cfg
    if cfg.source is None:
        raise ConfigError(f"variant {name!r} builds on {b!r}: needs a study file")
    return variant_config(cfg.source, b, data_dir=cfg.data_dir, out_dir=primary_out_dir(cfg))


def store_config(cfg: StudyConfig, variant: str | None = None,
                 snapshot: str | None = None) -> StudyConfig:
    """The configuration that reads ``variant``'s store at ``snapshot``: the one
    place that resolves a reporting population and a frozen result set to a directory.

    ``variant``: a sensitivity variant (e.g. ``reporting_population``); None, "" or :data:`PRIMARY`: the
    primary analysis. ``snapshot``: a snapshot *of that store* (a variant's snapshots sit under
    its own directory, :func:`variant_dir`); None, "" or ``live``: its live store. ``cfg`` is
    returned unchanged when it already reads that store, and a live configuration of the same
    store only gains the snapshot (overrides it was loaded with stay); otherwise the study file
    (``cfg.source``) is loaded again for the primary root of ``cfg``.

    Raises :class:`ConfigError` for an unknown variant, a variant without results, or a
    configuration without a study file, and :class:`FileNotFoundError` for a missing snapshot
    (its message lists the snapshots of that store)."""
    from biomech.results.snapshot import LIVE

    variant = None if variant in (None, "", PRIMARY) else str(variant)
    snapshot = None if snapshot in (None, "", LIVE) else str(snapshot)
    if variant == cfg.variant:
        if snapshot == cfg.snapshot:
            return cfg
        if cfg.snapshot is None:
            return cfg.with_snapshot(snapshot)
    if cfg.source is None:
        raise ConfigError("reading another store (variant "
                          f"{variant or PRIMARY}, snapshot {snapshot or LIVE}) needs a "
                          "configuration loaded from a study file")
    root = primary_out_dir(cfg)
    new = load_config(cfg.source, data_dir=cfg.data_dir, out_dir=root)
    if variant is not None:
        if variant not in variants(new):
            raise ConfigError(f"unknown sensitivity variant {variant!r}; known: "
                              f"{sorted(variants(new))}")
        vdir = variant_dir(new, variant)
        if not vdir.is_dir():           # checked first: variant_config writes into it
            raise ConfigError(f"sensitivity variant {variant!r} has no results under {vdir} "
                              f"(run `biomech sensitivity {variant}` first)")
        new = variant_config(cfg.source, variant, data_dir=cfg.data_dir, out_dir=root)
    return new.with_snapshot(snapshot)


def comparison_path(cfg: StudyConfig, name: str) -> Path:
    """The comparison of variant ``name`` with the primary run, as the store ``cfg`` reads it:
    in the store of variant ``name`` itself (live or snapshot) at its root; otherwise under
    ``sensitivity/<name>/`` of a snapshot (primary or variant: frozen with it) or of the live
    primary store (a live variant store holds no other variant's comparison: those compare
    the primary run, e.g. the D20 low-pass variants)."""
    read = Path(getattr(cfg, "read_dir", None) or cfg.out_dir)
    variant = getattr(cfg, "variant", None)
    if variant == name:
        return read / COMPARISON_FILE
    root = primary_out_dir(cfg) if variant and not getattr(cfg, "snapshot", None) else read
    return root / "sensitivity" / name / COMPARISON_FILE


def skipped_stages(cfg: StudyConfig, name: str) -> list[str]:
    """The stages variant ``name`` does not run, in pipeline order: its ``skip_stages`` and
    every stage of the study's groups that depends on one of them, directly or not."""
    from biomech.pipeline import get_stage, registry, resolve_order

    spec = variants(cfg).get(name) or {}
    skip = {str(s) for s in (spec.get("skip_stages") or [])}
    unknown = sorted(skip - set(registry()))
    if unknown:
        raise ConfigError(f"sensitivity variant {name!r}: unknown skip_stages {unknown}")
    order = resolve_order(None, cfg.groups)        # dependencies first
    for st in order:
        if any(d in skip for d in get_stage(st).depends):
            skip.add(st)
    return [st for st in order if st in skip]


def variant_stages(cfg: StudyConfig, name: str) -> list[str] | None:
    """The stages the variant runs (None: all of the study's groups)."""
    from biomech.pipeline import group_stages

    skip = set(skipped_stages(cfg, name))
    return [st for st in group_stages(cfg.groups) if st not in skip] if skip else None


def not_run(comparison: pd.DataFrame) -> list[str]:
    """Stages the variant did not re-run (``skip_stages``), as recorded in its comparison."""
    return list(comparison.attrs.get("not_run") or [])


def filtered_cohort(src: str | Path, dst: str | Path, flags: list[str], name: str,
                    swap: dict[str, int] | None = None) -> Path:
    """Write the manifest ``src`` to ``dst`` with every included participant that carries one
    of ``flags`` excluded (reason names the variant and flag) and, per ``swap`` (pid -> day
    index, 1 = the first later day), the participant's motion and MVC takes replaced by its
    ``secondary`` takes of that day (S5.4 between-day reliability); returns ``dst``. The other
    rows, session names and manifest keys are copied unchanged."""
    import yaml

    doc = yaml.safe_load(Path(src).read_text(encoding="utf-8")) or {}
    want = {f.lower() for f in flags}
    for row in doc.get("participants") or []:
        hit = sorted(want & {str(f).lower() for f in (row.get("flags") or [])})
        if hit and row.get("include", True):
            row["include"] = False
            row["exclusion_reason"] = f"sensitivity {name}: flag {', '.join(hit)}"
        day = (swap or {}).get(str(row.get("pid")))
        if day:
            row.update(_secondary_day(row, int(day), name))
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    what = ", ".join(x for x in (f"exclude_flags: {', '.join(flags)}" if flags else "",
                                 f"cohort_swap: {swap}" if swap else "") if x)
    text = (f"# Generated by the sensitivity variant {name} from {Path(src).name} "
            f"({what}); do not edit.\n"
            + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    if not dst.is_file() or dst.read_text(encoding="utf-8") != text:
        dst.write_text(text, encoding="utf-8")
    return dst


def _secondary_day(row: dict[str, Any], day: int, name: str) -> dict[str, Any]:
    """The motion / MVC session names of a participant's ``day``-th later day (from its
    ``secondary`` takes, grouped by date; MVC takes carry ``_mvc_``)."""
    import re

    by_day: dict[str, dict[str, str]] = {}
    for s in row.get("secondary") or []:
        m = re.search(r"_(\d{8})T\d{6}Z", str(s))
        if m:
            by_day.setdefault(m.group(1), {})["mvc" if "_mvc_" in str(s) else "motion"] = str(s)
    days = [by_day[d] for d in sorted(by_day)]
    if not 1 <= day <= len(days) or "motion" not in days[day - 1]:
        raise ConfigError(f"sensitivity variant {name!r}: participant {row.get('pid')} has no "
                          f"secondary motion take for day {day}")
    d = days[day - 1]
    out = {"motion": d["motion"], "secondary": [str(x) for x in (row.get("secondary") or [])
                                                if x not in d.values()]}
    out["secondary"] += [x for x in (row.get("motion"), row.get("mvc") if "mvc" in d else None)
                         if x]
    if "mvc" in d:
        out["mvc"] = d["mvc"]
    return out


def seed_cache(cfg: StudyConfig, vcfg: StudyConfig) -> int:
    """Copy the primary's stage units into the variant cache wherever the primary unit
    has exactly the key the variant needs (in dependency order, so cohort keys see the
    copied participant units), replacing a missing or outdated variant unit. Keys are content
    addressed (code, stage parameters, seed, upstream keys), so the variant run then re-runs
    only what its overrides change. Copies, not hard links (see
    :func:`biomech.results.snapshot._copy_tree`). Returns the units copied."""
    from biomech.cohort import load_cohort
    from biomech.pipeline import KeyResolver, registry, resolve_order
    from biomech.results.snapshot import _copy_tree

    src, dst = Path(cfg.out_dir) / "cache", Path(vcfg.out_dir) / "cache"
    if not src.is_dir():
        return 0
    primary = ResultStore(cfg.out_dir)
    variant = ResultStore(vcfg.out_dir)
    cohort = load_cohort(vcfg.cohort_path, vcfg.data_dir)
    n = 0
    for name in resolve_order(None, vcfg.groups):
        if name not in registry():
            continue
        keys = KeyResolver(vcfg, cohort, variant)   # fresh: cohort keys read the coverage
        for unit in primary.units(name):
            m = primary.meta(name, unit)
            if m is None or m.status != "ok":
                continue
            try:
                want = keys.key(name, unit)
            except Exception:                       # unit outside the cohort, missing input
                continue
            if m.key != want or variant.status(name, unit, want) == "up-to-date":
                continue
            target = dst / name / unit
            if target.exists():
                shutil.rmtree(target)
            _copy_tree(src / name / unit, target)
            n += 1
    return n


def _split(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    keys, vals = [], []
    for c in df.columns:
        if c == "unit":
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c]):
            vals.append(c)
        else:
            keys.append(c)
    return keys, vals


def _keyed(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    for k in keys:
        df[k] = df[k].astype(str)
    df["_row"] = df.groupby(keys, sort=False).cumcount() if keys else np.arange(len(df))
    return df


# Text columns that identify a row (who / what): always keys, even when their value sets
# differ between the runs (a variant without one participant, a test only one run has) --
# rows are then matched by these keys and reported as only in one run, never paired by
# position (E33).
IDENTITY_COLUMNS = {"pid", "task", "muscle", "test", "rule", "detector", "direction", "metric",
                    "model", "feature_set", "phase", "side", "estimator", "family", "role",
                    "event", "variant", "protocol", "group", "key", "stage", "table", "column",
                    "outcome", "take", "body", "name", "kind", "quantity"}
# Numeric columns that are parameters of a row, not results (they identify the row).
PARAM_COLUMNS = {"drop_db", "hold_s", "primary", "k", "w_emg", "cycle", "window_s",
                 "detector_k", "seed", "fold"}


def _keys_and_values(a: pd.DataFrame, b: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Key columns identify a row in both versions: text columns whose values are the same
    set in both (a text column that changed, e.g. a cluster list, is neither key nor value)
    and the numeric parameter columns of :data:`PARAM_COLUMNS`. Every other shared numeric or
    boolean column is a value."""
    ka, va = _split(a)
    kb, vb = _split(b)
    keys, vals = [], []
    for c in [c for c in a.columns if c in b.columns and c != "unit"]:
        if c in ka and c in kb:
            if c in IDENTITY_COLUMNS or set(a[c].astype(str)) == set(b[c].astype(str)):
                keys.append(c)
        elif c in va and c in vb:
            (keys if c in PARAM_COLUMNS else vals).append(c)
    return keys, vals


def compare_tables(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Long comparison of two versions of one table (see module docstring)."""
    if a.empty and b.empty:
        return pd.DataFrame()
    keys, vals = _keys_and_values(a, b)
    if not vals:
        return pd.DataFrame()
    a, b = _keyed(a, keys), _keyed(b, keys)
    on = keys + ["_row"]
    m = a[on + vals].merge(b[on + vals], on=on, how="outer", suffixes=("__p", "__v"),
                           indicator=True)
    match = m["_merge"].map({"both": "both", "left_only": "primary_only",
                             "right_only": "variant_only"}).astype(str)
    key = (m[keys].astype(str).agg(lambda r: "|".join(f"{k}={v}" for k, v in r.items()), axis=1)
           if keys else pd.Series("", index=m.index))
    key = key.where(m["_row"] == 0, key + "|#" + m["_row"].astype(str))
    out = []
    for v in vals:
        p = pd.to_numeric(m[f"{v}__p"].astype("float64"), errors="coerce")
        q = pd.to_numeric(m[f"{v}__v"].astype("float64"), errors="coerce")
        out.append(pd.DataFrame({"key": key, "column": v, "primary": p, "variant": q,
                                 "match": match}))
    res = pd.concat(out, ignore_index=True)
    res["diff"] = res["variant"] - res["primary"]
    with np.errstate(invalid="ignore", divide="ignore"):
        res["rel_diff"] = res["diff"] / res["primary"].abs()
    return res


def compare(primary: ResultStore, variant: ResultStore,
            stages: list[str] | None = None) -> pd.DataFrame:
    """Every cohort table present in both stores, compared value by value."""
    frames = []
    for stage in stages or primary.stages():
        mp, mv = primary.meta(stage, COHORT_UNIT), variant.meta(stage, COHORT_UNIT)
        if mp is None or mv is None or mp.status != "ok" or mv.status != "ok":
            continue
        for name in sorted(set(mp.tables) & set(mv.tables)):
            c = compare_tables(primary.read_table(stage, name, [COHORT_UNIT]),
                               variant.read_table(stage, name, [COHORT_UNIT]))
            if not c.empty:
                c.insert(0, "table", name)
                c.insert(0, "stage", stage)
                frames.append(c)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["stage", "table", "key", "column", "primary", "variant", "diff", "rel_diff",
                 "match"])


def write_comparison(cfg: StudyConfig, name: str) -> pd.DataFrame:
    """Compare the variant with the primary run (with its ``base`` variant's store when it has
    one, D36) and write the comparison. Stages the variant
    skips are left out even when its store holds older units of them (a run before the skip
    was configured), and are recorded in ``attrs["not_run"]`` (kept in the parquet file)."""
    vdir = variant_dir(cfg, name)
    primary = ResultStore(base_config(cfg, name).out_dir)
    skip = skipped_stages(cfg, name)
    df = compare(primary, ResultStore(vdir),
                 [st for st in primary.stages() if st not in skip] if skip else None)
    df.attrs["not_run"] = skip
    vdir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(vdir / COMPARISON_FILE, index=False)
    return df


def read_comparison(out_dir: str | Path, name: str) -> pd.DataFrame:
    p = Path(out_dir) / "sensitivity" / name / COMPARISON_FILE
    return pd.read_parquet(p) if p.is_file() else pd.DataFrame()


def available(out_dir: str | Path) -> list[str]:
    d = Path(out_dir) / "sensitivity"
    return sorted(p.name for p in d.iterdir() if (p / COMPARISON_FILE).is_file()) \
        if d.is_dir() else []
