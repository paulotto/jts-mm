# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Figure and table specs as files: re-render any export without the store or the code that
computed it (G3), and give journals their "source data".

Next to every exported figure (``figures/<id>/``):

* ``spec.json`` -- the whole :class:`~biomech.viz.spec.FigureSpec` (panels, axes with units,
  series kinds and roles, captions, sources, perspective) with every array replaced by a
  reference ``{"$array": name}`` into
* ``spec.npz`` -- the arrays, exact dtype and shape (lossless round trip);
* ``source_data.csv`` -- the 1-D series in long form (panel, series, kind, label, x, y, ...),
  the plotted numbers as a reader can reuse them.

Next to every exported table (``tables/``): ``<id>.csv`` (header with units, plain text) and
``<id>.json`` (columns with units and decimals, rows, caption). Both kinds carry a
``provenance`` record (snapshot or live store, newest run, git state, time).

``biomech render figures/<id>/spec.json --to pgf|pdf|png|html`` re-renders a figure from these
files alone (:func:`load_figure`).
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biomech.viz.spec import Axis, Colorbar, Column, FigureSpec, Panel, Series, TableSpec

SPEC_VERSION = 1
_ONE_D = ("x", "y", "y2", "ylo", "c", "z")


def provenance(ctx: Any) -> dict[str, Any]:
    """Where an export's numbers come from: the store's variant (the reporting population, plan
    None = the primary analysis), snapshot (or the live store), newest run, git."""
    from biomech.results.provenance import git_state

    cfg = getattr(ctx, "config", None)
    root = Path(getattr(cfg, "read_dir", "") or "")
    runs = sorted(p.name for p in (root / "runs").iterdir()) \
        if (root / "runs").is_dir() else []
    if not runs and (root / "snapshot.json").is_file():   # a snapshot lists its runs (E35)
        runs = sorted(json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
                      .get("runs") or [])
    bundle = None
    if getattr(cfg, "snapshot", None):           # the snapshot's bundle, if written (J9.4)
        from biomech.results.bundle import bundle_path, info

        bundle = info(bundle_path(root))
    from biomech.results.provenance import package_versions

    # the versions that produced the numbers (the snapshot record; for the live store the
    # newest run's) and the ones that wrote the export
    produced = None
    snap_json = root / "snapshot.json"
    if snap_json.is_file():
        produced = json.loads(snap_json.read_text(encoding="utf-8")).get("versions")
    elif runs and (root / "runs" / runs[-1] / "manifest.json").is_file():
        produced = json.loads((root / "runs" / runs[-1] / "manifest.json")
                              .read_text(encoding="utf-8")).get("versions")
    return {"study": getattr(cfg, "study", None),
            "variant": getattr(cfg, "variant", None),
            "snapshot": getattr(cfg, "snapshot", None) or "live",
            "newest_run": runs[-1] if runs else None, "git": git_state(),
            "bundle": bundle, "versions_analysis": produced,
            "versions_export": package_versions(),
            "exported_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


# -- generic (de)serialization ------------------------------------------------------------------
class _Arrays:
    def __init__(self) -> None:
        self.store: dict[str, np.ndarray] = {}

    def put(self, name: str, a: np.ndarray) -> dict[str, str]:
        self.store[name] = np.asarray(a)
        return {"$array": name}


def _enc(v: Any, path: str, arrs: _Arrays) -> Any:
    if isinstance(v, np.ndarray):
        return arrs.put(path, v)
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return {"$float": repr(v)}
    if is_dataclass(v):
        return {f.name: _enc(getattr(v, f.name), f"{path}.{f.name}", arrs) for f in fields(v)}
    if isinstance(v, dict):
        return {str(k): _enc(x, f"{path}.{k}", arrs) for k, x in v.items()}
    if isinstance(v, tuple):
        return {"$tuple": [_enc(x, f"{path}.{i}", arrs) for i, x in enumerate(v)]}
    if isinstance(v, list):
        return [_enc(x, f"{path}.{i}", arrs) for i, x in enumerate(v)]
    return v


def _dec(v: Any, arrs: Any) -> Any:
    if isinstance(v, dict):
        if "$array" in v:
            return np.asarray(arrs[v["$array"]])
        if "$float" in v:
            return float(v["$float"])
        if "$tuple" in v:
            return tuple(_dec(x, arrs) for x in v["$tuple"])
        return {k: _dec(x, arrs) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x, arrs) for x in v]
    return v


def _build(cls: type, d: dict[str, Any], arrs: Any) -> Any:
    kw = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        if cls is Panel and f.name in ("x", "y", "z"):
            kw[f.name] = _build(Axis, val, arrs)
        elif cls is Panel and f.name == "colorbar":
            kw[f.name] = None if val is None else _build(Colorbar, val, arrs)
        elif cls is Panel and f.name == "series":
            kw[f.name] = [_build(Series, s, arrs) for s in val]
        elif cls is FigureSpec and f.name == "panels":
            kw[f.name] = [_build(Panel, p, arrs) for p in val]
        else:
            kw[f.name] = _dec(val, arrs)
    return cls(**kw)


# -- figures ------------------------------------------------------------------------------------
def figure_dict(spec: FigureSpec) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrs = _Arrays()
    d = {"spec_version": SPEC_VERSION, "kind": "figure",
         "figure": _enc(spec, "f", arrs)}
    return d, arrs.store


def write_figure(spec: FigureSpec, fdir: str | Path, prov: dict[str, Any] | None = None
                 ) -> Path:
    """``spec.json`` + ``spec.npz`` + ``source_data.csv`` in ``fdir``; returns the json."""
    fdir = Path(fdir)
    fdir.mkdir(parents=True, exist_ok=True)
    d, arrays = figure_dict(spec)
    d["provenance"] = prov or {}
    np.savez_compressed(fdir / "spec.npz", **arrays)
    path = fdir / "spec.json"
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    write_source_data(spec, fdir / "source_data.csv", prov)
    return path


def load_figure(path: str | Path) -> FigureSpec:
    """The FigureSpec of a ``spec.json`` (its ``spec.npz`` next to it)."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("kind") != "figure":
        raise ValueError(f"{path} is not a figure spec")
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    return _build(FigureSpec, d["figure"], arrs)


def write_source_data(spec: FigureSpec, path: str | Path,
                      prov: dict[str, Any] | None = None) -> Path:
    """The 1-D series of a figure in long form (one row per plotted point)."""
    from biomech.viz.layout import plain_text

    path = Path(path)
    head = ["panel", "panel_title", "series", "kind", "label", "role", "point",
            "x_label", "x", "y_label", "y", "y2", "ylo", "c", "z"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        if prov:
            fh.write(f"# {spec.id}: {plain_text(spec.title)}; source "
                     f"{prov.get('snapshot')}, run {prov.get('newest_run')}, "
                     f"exported {prov.get('exported_utc')}\n")
        w = csv.writer(fh)
        w.writerow(head)
        for pi, p in enumerate(spec.panels):
            for si, s in enumerate(p.series):
                if s.z is not None and np.ndim(s.z) == 2:
                    # a heatmap (2026-11-11): one row per cell, x / y the column / row
                    # coordinate or, where the axis names its cells, the tick label
                    for r in _heatmap_rows(p, s):
                        w.writerow([pi, plain_text(p.title), si, s.kind, s.label or "", s.role,
                                    *r])
                    continue
                cols = {k: np.asarray(getattr(s, k)) for k in _ONE_D
                        if getattr(s, k) is not None and np.ndim(getattr(s, k)) == 1}
                if not cols:
                    continue
                n = max(len(v) for v in cols.values())
                cols = {k: v for k, v in cols.items() if len(v) == n}
                for i in range(n):
                    w.writerow([pi, plain_text(p.title), si, s.kind, s.label or "", s.role, i,
                                p.x.text(), _num(cols.get("x"), i), p.y.text(),
                                _num(cols.get("y"), i), _num(cols.get("y2"), i),
                                _num(cols.get("ylo"), i), _num(cols.get("c"), i),
                                _num(cols.get("z"), i)])
    return path


def _heatmap_rows(p: Any, s: Any) -> list[list[str]]:
    """``[point, x_label, x, y_label, y, y2, ylo, c, z]`` per cell of a 2-D ``z`` (rows = y)."""
    z = np.asarray(s.z)

    def coords(ax: Any, v: Any, n: int) -> list[str]:
        if v is not None and np.ndim(v) == 1 and len(v) == n:
            vals = [float(x) for x in np.asarray(v, float)]
        else:
            vals = [float(i) for i in range(n)]
        lab = dict(zip([float(t) for t in (ax.ticks or [])], ax.ticklabels or []))
        return [str(lab[x]) if x in lab else repr(x) for x in vals]

    grid = all(v is not None and np.shape(v) == z.shape for v in (s.x, s.y))   # a surface
    xs, ys = coords(p.x, s.x, z.shape[1]), coords(p.y, s.y, z.shape[0])
    out, k = [], 0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            x = _num(np.asarray(s.x)[i], j) if grid else xs[j]
            y = _num(np.asarray(s.y)[i], j) if grid else ys[i]
            out.append([k, p.x.text(), x, p.y.text(), y, "", "", "", _num(z[i], j)])
            k += 1
    return out


def _num(a: np.ndarray | None, i: int) -> str:
    if a is None:
        return ""
    v = a[i]
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else repr(float(v))
    return str(v.item() if hasattr(v, "item") else v)


# -- tables -------------------------------------------------------------------------------------
def write_table(spec: TableSpec, out_dir: str | Path, prov: dict[str, Any] | None = None
                ) -> tuple[Path, Path]:
    """``tables/<id>.csv`` (plain headers with units) and ``tables/<id>.json``."""
    from biomech.viz.layout import plain_text
    from biomech.viz.style import plain_unit

    out = Path(out_dir) / "tables"
    out.mkdir(parents=True, exist_ok=True)
    arrs = _Arrays()
    # Table cells are small: array cells are written as lists (a table json has no npz).
    rows = [{c.key: _enc(_listify(r.get(c.key)), "r", arrs) for c in spec.columns}
            for r in spec.rows
            if not r.get("_rule")]
    d = {"spec_version": SPEC_VERSION, "kind": "table", "provenance": prov or {},
         "table": {"id": spec.id, "caption": spec.caption, "notes": spec.notes,
                   "sources": spec.sources, "width": spec.width,
                   "meta": _enc(spec.meta, "m", arrs),
                   "columns": [_enc(c, "c", arrs) for c in spec.columns], "rows": rows}}
    jp = out / f"{spec.id}.json"
    jp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    cp = out / f"{spec.id}.csv"
    with cp.open("w", newline="", encoding="utf-8") as fh:
        if prov:
            fh.write(f"# {spec.id}; source {prov.get('snapshot')}, run "
                     f"{prov.get('newest_run')}, exported {prov.get('exported_utc')}\n")
        w = csv.writer(fh)
        w.writerow([plain_text(c.header) + (f" ({plain_unit(c.unit)})" if c.unit else "")
                    for c in spec.columns])
        for r in spec.rows:
            if r.get("_rule"):
                continue
            w.writerow([_cell(r.get(c.key)) for c in spec.columns])
    return cp, jp


def _listify(v: Any) -> Any:
    return v.tolist() if isinstance(v, np.ndarray) else v


def _cell(v: Any) -> str:
    from biomech.viz.layout import plain_text

    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "" if not np.isfinite(v) else repr(float(v))
    if isinstance(v, (int, np.integer, bool, np.bool_)):
        return str(v.item() if hasattr(v, "item") else v)
    return plain_text(str(v))


def load_table(path: str | Path) -> TableSpec:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("kind") != "table":
        raise ValueError(f"{path} is not a table spec")
    t = d["table"]
    return TableSpec(t["id"], t["caption"], [Column(**_dec(c, {})) for c in t["columns"]],
                     [_dec(r, {}) for r in t["rows"]], t.get("notes", ""),
                     list(t.get("sources", [])), t.get("width", "column"),
                     _dec(t.get("meta", {}), {}))
