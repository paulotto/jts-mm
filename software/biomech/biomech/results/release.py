# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pseudonymized dataset release of the result store (G4; Scientific Data descriptor,
supplementary source data).

``biomech export-data --to DIR [--format csv|hdf5] [--arrays] [--snapshot NAME]`` writes every
stored table of the included participants and the cohort:

* study ids are replaced by the participants' pseudonyms (``P<n>``, the same numbering as in
  every figure); rows of participants outside the cohort are dropped, and any other study id
  found in a text cell becomes ``excluded``;
* per-participant demographic columns (age, sex, birth dates, ...) are never released --
  demographics are reported in aggregate only (``qc_cohort.demographics``);
* ``data_dictionary.csv``: one row per released column (label, unit, decimals, description
  from the metric catalog) and, with ``--arrays``, per array family (unit, shape,
  description from :data:`biomech.results.schema.ARRAYS`);
* ``release.json``: provenance (snapshot or live store, runs, git state, code hash) and the
  list of files.

``csv``: ``<stage>/<table>.csv``. ``hdf5``: one ``results.h5`` with ``/<stage>/<table>/<column>``
datasets (column attributes ``label``, ``unit``, ``description``) and, with ``--arrays``,
``/arrays/<stage>/<P<n>|cohort>/<key>``. The release is scanned for study ids before it is
reported complete; a leak raises.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomech.results import COHORT_UNIT

PID_RE = re.compile(r"(?<![A-Za-z0-9])s\d{2}(?![A-Za-z0-9])")
PRIVATE_COLUMNS = re.compile(r"^(age|sex|gender|birth|dob|height|weight|session)(_.*)?$",
                             re.I)
# Session folder names encode study id, sex, age and recording time
# (session_s01_f_31_20250302T101000Z_...): never released, in any cell or name (E29).
SESSION_RE = re.compile(r"session_[A-Za-z0-9]+_[fmdx]_\d{1,3}(?:_[A-Za-z0-9]+)*")
LEAKS = (PID_RE, SESSION_RE)


def _pseudo_map(ctx: Any) -> dict[str, str]:
    return {pid: f"P{n}" for pid, n in ctx.pseudonyms().items()}


def _sub(text: str, pm: dict[str, str]) -> str:
    text = SESSION_RE.sub("session (withheld)", text)
    return PID_RE.sub(lambda m: pm.get(m.group(0), "excluded"), text)


def pseudonymize(df: pd.DataFrame, pm: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Rows of included participants (and cohort rows), ids replaced, private columns
    dropped; returns (frame, dropped column names)."""
    df = df.copy()
    for col in ("unit", "pid"):
        if col in df.columns:
            ids = df[col].astype(str)
            keep = ids.isin(pm) | (ids == COHORT_UNIT) | ~ids.str.fullmatch(r"s\d{2}")
            df = df[keep.to_numpy()]
    # Per-person rows (a pid or participant unit) never carry demographics; cohort
    # aggregates (age_mean, n per sex, ...) are released.
    per_person = "pid" in df.columns or ("unit" in df.columns
                                         and (df["unit"].astype(str) != COHORT_UNIT).any())
    dropped = [c for c in df.columns if per_person and PRIVATE_COLUMNS.match(str(c))]
    df = df.drop(columns=dropped)
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: _sub(v, pm) if isinstance(v, str) else v)
    df.columns = [_sub(str(c), pm) for c in df.columns]
    return df.reset_index(drop=True), dropped


def export_data(ctx: Any, target: str | Path, fmt: str = "csv",
                arrays: bool = False) -> dict[str, Any]:
    from biomech.results.provenance import git_state
    from biomech.results.schema import array_entry, column_entry

    if fmt not in ("csv", "hdf5"):
        raise ValueError(f"unknown format {fmt!r}")
    if arrays and fmt != "hdf5":
        raise ValueError("arrays are released in the hdf5 format only")
    out = Path(target)
    out.mkdir(parents=True, exist_ok=True)
    store, pm = ctx.store, _pseudo_map(ctx)
    units_ok = set(pm) | {COHORT_UNIT}
    dictionary: list[dict[str, Any]] = []
    files: list[str] = []
    dropped_all: dict[str, list[str]] = {}
    h5 = None
    if fmt == "hdf5":
        import h5py

        h5 = h5py.File(out / "results.h5", "w")
        files.append("results.h5")
    try:
        for stage in store.stages():
            units = [u for u in store.units(stage) if u in units_ok]
            names: list[str] = []
            for u in units:
                m = store.meta(stage, u)
                if m is not None and m.status == "ok":
                    names += [t for t in m.tables if t not in names]
            for name in names:
                df, dropped = pseudonymize(store.read_table(stage, name, units), pm)
                if dropped:
                    dropped_all[f"{stage}.{name}"] = dropped
                if df.empty:
                    continue
                for c in df.columns:
                    e = column_entry(c)
                    dictionary.append({"kind": "column", "stage": stage, "table": name,
                                       "name": c, "label": e["label"], "unit": e["unit"],
                                       "decimals": e["decimals"], "dtype": str(df[c].dtype),
                                       "description": e["description"]})
                if fmt == "csv":
                    p = out / stage / f"{name}.csv"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(p, index=False)
                    files.append(p.relative_to(out).as_posix())
                else:
                    g = h5.require_group(f"{stage}/{name}")
                    for c in df.columns:
                        col = df[c]
                        if col.dtype == object or pd.api.types.is_string_dtype(col):
                            data = np.array(["" if v is None or (isinstance(v, float) and
                                                                 np.isnan(v)) else str(v)
                                             for v in col], dtype=h5py.string_dtype())
                        else:
                            data = col.to_numpy()
                        ds = g.create_dataset(str(c).replace("/", "_"), data=data)
                        e = column_entry(c)
                        ds.attrs.update(label=e["label"], unit=e["unit"],
                                        description=e["description"] or "")
            if arrays:
                from biomech.results.access import family

                seen: set[str] = set()
                for u in units:
                    m = store.meta(stage, u)
                    if m is None or m.status != "ok" or not m.has_arrays:
                        continue
                    who = "cohort" if u == COHORT_UNIT else pm[u]
                    for k, v in store.read_arrays(stage, u).items():
                        key = _sub(k, pm)
                        if "excluded" in key:
                            continue
                        if v.dtype.kind in ("U", "S", "O"):
                            import h5py

                            v = np.array([_sub(str(x), pm) for x in v.ravel()],
                                         dtype=h5py.string_dtype()).reshape(v.shape)
                        h5.create_dataset(f"arrays/{stage}/{who}/{key}", data=v,
                                          compression="gzip" if v.size > 1 else None)
                        f = family(k)
                        if f not in seen:
                            seen.add(f)
                            e = array_entry(stage, f) or {}
                            dictionary.append({"kind": "array", "stage": stage, "table": "",
                                               "name": _sub(f, pm), "label": "",
                                               "unit": e.get("unit", ""), "decimals": "",
                                               "dtype": str(v.dtype),
                                               "description": e.get("description", "")})
    finally:
        if h5 is not None:
            h5.close()
    with (out / "data_dictionary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["kind", "stage", "table", "name", "label", "unit", "decimals",
                                "dtype", "description"])
        w.writeheader()
        w.writerows(dictionary)
    files.append("data_dictionary.csv")
    cfg = ctx.config
    meta = {"study": cfg.study, "snapshot": getattr(cfg, "snapshot", None) or "live",
            "participants": len(pm), "pseudonyms": "P1..P%d (figure numbering)" % len(pm),
            "git": git_state(), "format": fmt, "arrays": arrays, "files": files,
            "dropped_private_columns": dropped_all}
    (out / "release.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    leaks = scan(out)
    if leaks:
        raise RuntimeError(f"study ids left in the release: {leaks[:10]}")
    return meta


def scan(out: str | Path) -> list[str]:
    """Study ids and session folder names found in the released text files and HDF5
    names/strings."""
    out = Path(out)
    found: set[str] = set()
    for p in out.rglob("*"):
        if p.suffix in (".csv", ".json"):
            text, rel = p.read_text("utf-8"), p.relative_to(out).as_posix()
            for rx in LEAKS:
                found |= {f"{p.name}:{m.group(0)}" for m in rx.finditer(text)}
                found |= {f"path:{m.group(0)}" for m in rx.finditer(rel)}
        elif p.suffix == ".h5":
            import h5py

            def visit(name: str, obj: Any) -> None:
                for rx in LEAKS:
                    found.update(f"h5:{m.group(0)}" for m in rx.finditer(name))
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in ("O", "S", "U"):
                    for v in obj.asstr()[...].ravel()[:100000]:
                        for rx in LEAKS:
                            found.update(f"h5:{m.group(0)}" for m in rx.finditer(v))

            with h5py.File(p, "r") as f:
                f.visititems(visit)
    return sorted(found)
