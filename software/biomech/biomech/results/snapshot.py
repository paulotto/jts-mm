# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frozen result snapshots: reproduce figures, tables and videos without re-running.

``biomech snapshot NAME`` freezes the current result store into
``<out_dir>/snapshots/NAME/``:

* ``cache/`` -- every stage unit, **copied** (not hard-linked: cloud sync clients such as
  MEGA do not sync hard links and ignore every linked file; a snapshot costs its size once);
* ``snapshot.json`` -- provenance: creation time, note, git state, code hash, package
  versions, the resolved configuration, the cohort file, the run ids whose units are
  included, and the stage/unit status at freeze time;
* ``catalog.yaml``, the study YAML and the cohort manifest as they were;
* ``sensitivity/<name>/comparison.parquet`` -- the sensitivity comparisons at freeze time.

A sensitivity variant's store is frozen the same way into its own directory
(``biomech snapshot --variant reporting_population NAME`` -> ``<out_dir>/sensitivity/reporting_population/
snapshots/NAME/``) with the variant's filtered cohort manifest, its ``comparison.parquet``
and the primary's comparisons; ``--out-dir`` pointing at the variant directory
would freeze the primary manifest instead. :func:`biomech.sensitivity.store_config`
resolves both kinds.

Every reading command takes ``--snapshot NAME`` (``view``, ``export``, ``check-outputs``,
``report``, ``animate``): its :class:`~biomech.io.context.DataContext` then reads the
snapshot instead of the live store. ``biomech compare A B`` lines up the cohort tables of
two snapshots (or a snapshot and the live store) with :func:`biomech.sensitivity.compare`.
"""

from __future__ import annotations

import json
import re
import os
import shutil
import time
from pathlib import Path
from typing import Any

from biomech.results.store import ResultStore

SNAPSHOT_DIR = "snapshots"
LIVE = "live"                     # name of the live store in comparisons


def root(out_dir: str | Path) -> Path:
    return Path(out_dir) / SNAPSHOT_DIR


def path(out_dir: str | Path, name: str) -> Path:
    """Directory of snapshot ``name`` (``live`` = the live store itself)."""
    if name == LIVE:
        return Path(out_dir)
    p = root(out_dir) / name
    if not (p / "snapshot.json").is_file():
        known = ", ".join(available(out_dir)) or "none"
        raise FileNotFoundError(f"no snapshot {name!r} under {root(out_dir)} (known: {known})")
    return p


def available(out_dir: str | Path) -> list[str]:
    r = root(out_dir)
    return sorted(p.name for p in r.iterdir() if (p / "snapshot.json").is_file()) \
        if r.is_dir() else []


def info(out_dir: str | Path, name: str) -> dict[str, Any]:
    return json.loads((path(out_dir, name) / "snapshot.json").read_text(encoding="utf-8"))


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy every file of ``src`` into ``dst`` (with its timestamps). Copies, not hard links:
    a cloud-synced results folder (MEGA) skips hard-linked files."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]   # no temp/.old units
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in filenames:
            s, d = Path(dirpath) / f, dst / rel / f
            shutil.copy2(s, d)
            n += 1
    return n


def create(config: Any, name: str, *, note: str = "", overwrite: bool = False) -> Path:
    """Freeze the live store of ``config`` as snapshot ``name``; returns its directory. For a
    variant configuration (:func:`biomech.sensitivity.variant_config`) that is the variant's
    store, its filtered cohort manifest and its comparison with the primary run."""
    from biomech.pipeline import code_hash, stage_status
    from biomech.results.provenance import git_state, package_versions

    if not name or name == LIVE or "/" in name or name.startswith("."):
        raise ValueError(f"invalid snapshot name {name!r}")
    out = Path(config.out_dir)
    dst = root(out) / name
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"snapshot {name!r} exists (use --overwrite)")
        shutil.rmtree(dst)
    store = ResultStore(out)
    dst.mkdir(parents=True)
    n = _copy_tree(out / "cache", dst / "cache") if (out / "cache").is_dir() else 0
    # Sensitivity comparisons (the variants' cohort tables lined up with the primary) are
    # read by result views: frozen with the snapshot, so views on it never mix in newer runs.
    # A variant's store (biomech snapshot --variant) holds its own comparison at its root and
    # freezes the primary's comparisons (the other variants compare the primary run) with it.
    from biomech.sensitivity import COMPARISON_FILE, primary_out_dir

    for comp in sorted((primary_out_dir(config) / "sensitivity").glob(f"*/{COMPARISON_FILE}")):
        d = dst / "sensitivity" / comp.parent.name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(comp, d / comp.name)
    if getattr(config, "variant", None) and (out / COMPARISON_FILE).is_file():
        shutil.copy2(out / COMPARISON_FILE, dst / COMPARISON_FILE)
    status = stage_status(config)
    runs = sorted({m.run_id for st in store.stages() for u in store.units(st)
                   if (m := store.meta(st, u)) is not None and m.run_id})
    for src, nm in ((config.source, "study.yaml"), (config.cohort_path, "cohort.yaml")):
        if src is not None and Path(src).is_file():
            shutil.copy2(src, dst / nm)
    from biomech.catalog import CATALOG_PATH

    shutil.copy2(CATALOG_PATH, dst / "catalog.yaml")
    meta = {
        "name": name, "note": note,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "study": config.study, "source_out_dir": str(out), "n_files": n,
        "git": git_state(), "code_hash": code_hash(), "versions": package_versions(),
        "config": config.raw, "runs": runs,
        "status": status.to_dict("records") if not status.empty else [],
        "stale_units": int((status.status != "up-to-date").sum()) if not status.empty else 0,
    }
    (dst / "snapshot.json").write_text(json.dumps(meta, indent=2, default=str),
                                       encoding="utf-8")
    from biomech.results.schema import describe

    (dst / "schema.json").write_text(json.dumps(describe(ResultStore(dst)), indent=1,
                                                default=str), encoding="utf-8")
    return dst


def delete(out_dir: str | Path, name: str) -> None:
    shutil.rmtree(path(out_dir, name))


# ``snapshot:`` (paper store, cite blocks) and ``record_snapshot:`` (the analysis of record)
PIN_RE = re.compile(r"^\s*(?:record_)?snapshot:\s*['\"]?([\w.-]+)", re.M)


def pinned(study_yaml: str | Path) -> dict[str, list[str]]:
    """Snapshot name -> the files that pin it: the ``snapshot:`` lines of every paper set and
    numbers file (``cite:`` blocks included) of all studies next to ``study_yaml``'s study
    (``experiments/*/paper_*.yaml``, ``numbers_*.yaml``). A pinned snapshot is never deleted
    (J1.4). Names are matched across studies: a name pinned by any study is kept."""
    root = Path(study_yaml).resolve().parent.parent
    out: dict[str, list[str]] = {}
    for f in sorted(root.glob("*/paper_*.yaml")) + sorted(root.glob("*/numbers_*.yaml")):
        for m in PIN_RE.finditer(f.read_text(encoding="utf-8")):
            out.setdefault(m.group(1), []).append(str(f.relative_to(root)))
    return out
