# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sensitivity runs (an08 §1): how much an alternative analysis choice changes every result.

Reads ``<out_dir>/sensitivity/<variant>/comparison.parquet`` written by
``biomech sensitivity NAME``: every cohort table of the primary and the variant run, value by
value. Pickers: the variant and the table.
"""

from __future__ import annotations

import numpy as np

from biomech import catalog, sensitivity as S
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register


def _variants(ctx) -> list[str]:
    return S.available(ctx.config.read_dir)


def _tables(ctx) -> list[str]:
    out: list[str] = []
    for v in _variants(ctx):
        c = S.read_comparison(ctx.config.read_dir, v)
        out += [f"{a}:{b}" for a, b in c[["stage", "table"]].drop_duplicates().itertuples(
            index=False)] if not c.empty else []
    return sorted(dict.fromkeys(out))


def _table_name(tab: str | None) -> str:
    """Reader name of a compared ``stage:table`` (its ``etc/tables.yaml`` title)."""
    from biomech.viz.views.tables import declarations

    if not tab:
        return "every cohort table"
    st, tb = tab.split(":", 1)
    return str(declarations().get(tab, {}).get("title")
               or f"{V.term('stage', st)}: {tb.replace('_', ' ')}")


def _column_name(col: str) -> str:
    return catalog.get(col).label if V.known(col) else col.replace("_", " ")


def _field(f: str, v: str) -> str:
    """One field of a comparison row key in reader words (value through the vocabulary)."""
    return str(V.cell(f, v)) if f != "#" else V.rewrite_ids(v)


def _reader_key(key: str) -> str:
    """A comparison row key (``a=1|b=2``) in reader words: label and reader value of every
    non-empty field."""
    out = []
    for part in key.split("|"):
        f, eq, v = part.partition("=")
        if not eq:
            out.append(V.rewrite_ids(part))
        elif v:
            out.append(f"{_column_name(f)}: {_field(f, v)}")
    return "; ".join(out)


def _short_labels(d) -> list[str]:
    """Row labels from the key fields that vary within ``d`` (``a=1|b=2`` keys), then the
    column -- the shared prefix carries no information."""
    parts = [dict(f.split("=", 1) if "=" in f else ("#", f) for f in k.split("|"))
             for k in d.key]
    fields = sorted({f for p in parts for f in p})
    vary = [f for f in fields if len({p.get(f) for p in parts}) > 1]
    out = []
    for p, col in zip(parts, d.column):
        txt = " ".join(_field(f, str(p.get(f, ""))) for f in vary if p.get(f))
        out.append(f"{txt} · {_column_name(col)}"[:64] if txt else _column_name(col))
    return out


def _pseudo_keys(ctx, c):
    """Replace ``pid=sNN`` in the row keys by the pseudonym (``pid=P3``): views never show
    study ids."""
    if c is None or c.empty:
        return c
    m = {p: f"P{ctx.pseudonym(p)}" for p in ctx.pids()}
    c = c.copy()
    # any study-id token (pid=, left_out=, test names ...) becomes the pseudonym
    c["key"] = c.key.str.replace(r"(?<![A-Za-z0-9])(s\d{2})(?![0-9])",
                                 lambda g: m.get(g.group(1), "P?"), regex=True)
    return c


def _comparison(req: ViewRequest):
    ctx = req.ctx
    names = _variants(ctx)
    name = req.options.get("variant") or (names[0] if names else None)
    c = S.read_comparison(ctx.config.read_dir, name) if name else None
    return name, _pseudo_keys(ctx, c)


@register("sensitivity_changes", "dashboard", "Sensitivity run: largest changes",
          domain="variability", column="median",
          choices={"variant": _variants, "table": _tables,
                   "column": ("median", "mean", "p", "n", "all columns")})
def sensitivity_changes(req: ViewRequest) -> FigureSpec | TableSpec:
    """The values of one table that change most under the variant (primary vs variant)."""
    name, c = _comparison(req)
    tab = req.options.get("table")
    col = req.options.get("column", "median")
    if c is None or c.empty:
        return TableSpec("sensitivity_changes", "No sensitivity run found "
                         "(biomech sensitivity NAME).", [Column("note", "Note", align="l")],
                         [{"note": "none"}])
    if tab:
        st, tb = tab.split(":", 1)
        c = c[(c.stage == st) & (c.table == tb)]
    if col != "all columns" and (c.column == col).any():
        c = c[c.column == col]             # one statistic: comparable bars, no min/CI noise
    d = c[np.isfinite(c["diff"]) & (c["diff"] != 0) & np.isfinite(c.rel_diff)].copy()
    # Relative change only where the primary value is not ~0 (else it is meaningless).
    scale = c.groupby("column").primary.transform(
        lambda v: float(np.median(np.abs(v[np.isfinite(v)]))) if np.isfinite(v).any() else 0.0)
    d = d[d.primary.abs() > 0.05 * scale.reindex(d.index).fillna(0)]
    d = d.reindex(d.rel_diff.abs().sort_values(ascending=False).index).head(25)
    if d.empty:
        return TableSpec("sensitivity_changes", f"Sensitivity run "
                         f"'{V.term('sensitivity_variant', name)}': no value of "
                         f"{_table_name(tab)} ({col}) changes.",
                         [Column("note", "Note", align="l")], [{"note": "no change"}])
    lab = _short_labels(d)
    y = np.arange(len(lab), dtype=float)

    p = Panel(f"{_table_name(tab)}: largest relative changes", Axis("Change", "\\percent"),
              Axis(""), [Series("barh", 100 * d.rel_diff.to_numpy(float), y, role="primary"),
                         Series("vline", x=np.array([0.0]), role="reference")], legend="none")
    p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), lab, True
    n_all, n_diff = len(c), int((c["diff"].abs() > 1e-12).sum())
    return FigureSpec("sensitivity_changes", "Sensitivity run: largest changes", [p],
                      width="text", height_in=max(1.6, 0.2 * len(lab)),
                      caption=f"Variant '{V.term('sensitivity_variant', name)}' vs the "
                              f"primary run: the up to 25 values of {_table_name(tab)} with the "
                              "largest relative change "
                              "(variant − primary) / |primary|, values with a primary near 0 "
                              f"left out ({n_diff} of {n_all} compared values differ). Row "
                              "labels: the identifying fields that vary, then the column. "
                              "'Sensitivity run: comparison table' lists every value.",
                      meta={"variant": name})


@register("sensitivity_table", "dashboard", "Sensitivity run: comparison table",
          domain="variability", choices={"variant": _variants, "table": _tables})
def sensitivity_table(req: ViewRequest) -> TableSpec:
    """Every compared value of one table: primary, variant, difference."""
    name, c = _comparison(req)
    if c is None or c.empty:
        return TableSpec("sensitivity_table", "No sensitivity run found "
                         "(biomech sensitivity NAME).", [Column("note", "Note", align="l")],
                         [{"note": "none"}])
    tab = req.options.get("table")
    if tab:
        st, tb = tab.split(":", 1)
        c = c[(c.stage == st) & (c.table == tb)]
    rows = [{"key": _reader_key(r["key"]), "column": _column_name(r["column"]),
             **{k: r[k] for k in ("primary", "variant", "diff", "rel_diff")}}
            for r in c.head(400).to_dict("records")]
    cols = [Column("key", "Row", align="X"), Column("column", "Column", align="l"),
            Column("primary", "Primary", decimals=3), Column("variant", "Variant", decimals=3),
            Column("diff", "Δ", decimals=3), Column("rel_diff", "Δ rel.", decimals=3)]
    return TableSpec("sensitivity_table", f"Sensitivity run "
                     f"'{V.term('sensitivity_variant', name)}': {_table_name(tab)} "
                     "(first 400 values).", cols, rows, width="text")
