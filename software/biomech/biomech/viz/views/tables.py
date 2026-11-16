# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Complete result tables: one table view per stored stage table.

``etc/tables.yaml`` declares every stored table with a title and a caption; this module
registers one dashboard table view per entry (domain "Result tables"), in the group of the
stage that writes it. A table view shows every column: reader headers from the catalog,
cells through the vocabulary (yes/no, task and muscle names), private columns and session
or file names withheld (:func:`blocks.redact`). Curated paper tables (a selection of columns,
own layout) are separate views; these are the complete records behind them, so no stored
result stays without an artefact (``check-outputs``: column completeness).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Column, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views.registry import ViewRequest, register

TABLES_PATH = Path(__file__).resolve().parents[2] / "etc" / "tables.yaml"
MAX_ROWS = 400


@lru_cache(maxsize=1)
def declarations() -> dict[str, dict[str, Any]]:
    """``"stage:table"`` -> {id, title, caption, ...} of ``etc/tables.yaml``."""
    doc = yaml.safe_load(TABLES_PATH.read_text(encoding="utf-8")) or {}
    return {str(k): dict(v) for k, v in (doc.get("tables") or {}).items()}


def view_id(key: str, decl: dict[str, Any]) -> str:
    stage, name = key.split(":", 1)
    return str(decl.get("id") or f"tab_{stage}__{name}")


def _group(stage: str) -> str:
    """The stage group (dataset / validation) of a stage, also for plugin stages that are
    not installed (``ml_*``: dataset)."""
    try:
        from biomech.pipeline import get_stage

        return get_stage(stage).group
    except Exception:
        return "validation" if stage.startswith("ref_") else "dataset"


def _builder(key: str, decl: dict[str, Any]):
    stage, name = key.split(":", 1)
    vid = view_id(key, decl)
    max_rows = int(decl.get("max_rows", MAX_ROWS))

    def build(req: ViewRequest) -> TableSpec:
        d = B.pseudonymize_cells(B.redact(req.ctx.table(stage, name)), req.ctx)
        cap = CaptionSpec(str(decl["caption"]).rstrip("."),
                          n=str(decl.get("n", "")), statistic=str(decl.get("statistic", "")),
                          evidence=str(decl.get("evidence", "")),
                          notes=[str(x) for x in decl.get("notes", [])])
        if "pid" in d.columns and len(d):
            cap.n = f"n = {d['pid'].nunique()} participants"
        if len(d) > max_rows:
            cap.notes.append(f"First {max_rows} of {len(d)} rows; the complete table is in "
                             "the data export.")
        sort = [c for c in decl.get("sort", []) if c in d.columns]
        if sort:
            d = d.sort_values(sort, kind="stable")
        t = B.frame_table(d, vid, "", ctx=req.ctx, max_rows=max_rows, width="text")
        if not t.rows:                        # an empty result says so (no blank table)
            if not t.columns:
                t.columns = [Column("note", "Note", align="l")]
            t.rows = [{t.columns[0].key: "no rows in this study"}]
        t.sources = [key]
        return attach(t, cap)

    build.__doc__ = str(decl["caption"])
    build.__name__ = vid
    return build


for _key, _decl in declarations().items():
    register(view_id(_key, _decl), "dashboard", str(_decl["title"]), domain="tables",
             group=_group(_key.split(":", 1)[0]))(_builder(_key, _decl))
