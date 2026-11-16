# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""In-text numbers of a paper: declared once, resolved from the frozen store.

A paper set names a numbers file (``numbers: numbers_dataset.yaml``); each entry is one
number of the manuscript::

    reqOpening:                         # macro name: letters only -> \\numReqOpening
      source: requirements:requirements # stage:table, or sensitivity:<variant> (the
                                        #   variant's comparison; filter on stage, table,
                                        #   column and "@<key part>", column primary|variant),
                                        #   or hw:<table> (hardware/mechanical/tables/
                                        #   <table>.csv: CAD masses, cross-check; J13),
                                        #   or bench:<table> (hardware/bench/tables/
                                        #   <table>.csv: bench checks such as B1),
                                        #   or derived:<name> (a table computed from the
                                        #   store with the figures' helpers, viz/derived.py)
      filter: {key: opening_mm}         # must select exactly one row
      column: value
      ci: [ci_lo, ci_hi]                # optional: CI columns
      n: n                              # optional: the row's n -- a column, an integer,
                                        #   or {distinct: pid} (distinct values in the rows)
      n_of: participants                # optional: what n counts (default participants;
                                        #   e.g. channels -- not bounded by the cohort size)
      fmt: ".1f"                        # optional (default: the catalog's decimals);
                                        #   "req": the requirements table's rounding (decimals
                                        #   from the row's design value, viz.views.paper)
      unit: "\\milli\\metre"            # optional siunitx unit (default: the catalog's)
      scale: 1000                       # optional factor (e.g. s -> ms)
      agg: median                       # optional: aggregate the filtered rows (median,
                                        #   mean, sum, count, max, min) instead of one row;
                                        #   n = distinct participants (``pid``) of those rows,
                                        #   else the number of rows
      allow_empty: true                 # optional, agg: count only: no matching rows = 0
      text: "..."                       # optional: what the number is (for the handoff)

    winTiming:                          # a method parameter (J8): the value the
      source: config:stages.timing.emg_pre_s   # store was computed with (the snapshot's frozen
      scale: 1000                       #   config); no column, filter or n
      fmt: ".0f"
      unit: "\\milli\\second"

    meanOpening:                        # n = 2 rule: a descriptive mean of
      mean_of: [openingPOne, openingPTwo]   #   other numbers of the file (same unit); n = the
      text: "..."                       #   number of operands unless ``n: <int>``; fmt and
                                        #   unit from the first operand unless given

    cite:                               # numbers of ANOTHER study, resolved from its pinned
      systems:                          #   snapshot (cited, not
        study: ../jtsmm_validation/study.yaml   # re-reported); one group per source
        snapshot: v7
        label: systems paper            # how the text names the source (numbers.json)
        from: ../jtsmm_validation/numbers_systems.yaml   # optional: declarations to reuse
        numbers:
          peakRatioRefMin: same         # the declaration of that key in ``from``
          jitterOptical: {source: ..., column: ...}      # or a full declaration

:func:`resolve` reads every entry from the store the context reads (the paper set's pinned
snapshot; a cited entry from its own study's snapshot) and fails on an unknown table, a
filter that does not select exactly one row, or a missing column -- a number in the text can
only be what the store holds. :func:`write` produces ``numbers.json`` (value, formatted text,
unit, CI, n, source, snapshot; cited numbers listed separately under ``cited`` with
``"cited": true`` and their study and snapshot) and ``numbers.tex``::

    \\newcommand{\\numReqOpening}{\\qty{39.7}{\\milli\\metre}}   value with unit
    \\newcommand{\\numReqOpeningVal}{39.7}                      bare value
    \\newcommand{\\numReqOpeningCI}{[\\num{35.2}, \\num{41.2}]\\,\\unit{\\milli\\metre}}
    \\newcommand{\\numReqOpeningN}{14}

(``siunitx`` v3). Values are rounded by the stated format only when written; ``numbers.json``
keeps the full value for the integrity check (:func:`verify`).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

KEY_RE = re.compile(r"^[A-Za-z]+$")


@dataclass
class Number:
    key: str
    value: float | str
    text: str
    unit: str = ""
    ci: tuple[float, float] | None = None
    ci_text: tuple[str, str] | None = None
    n: int | None = None
    source: str = ""
    filter: dict[str, Any] = field(default_factory=dict)
    column: str = ""
    what: str = ""
    cite: dict[str, Any] | None = None       # cited: {group, study, snapshot, label}
    mean_of: list[str] | None = None
    n_rows: int | None = None                # aggregated rows (agg), when n counts participants
    n_of: str = "participants"               # what n counts (``n_of:``; check-outputs bounds
                                             #   participant counts by the cohort size)

    def macro(self) -> str:
        return "num" + self.key[:1].upper() + self.key[1:]


CITE_KEY = "cite"
CITE_FIELDS = ("study", "snapshot", "variant", "label", "from", "numbers")


def _key_ok(path: Path, k: Any) -> str:
    if not KEY_RE.match(str(k)):
        raise ValueError(f"{path.name}: number key {k!r} must be letters only "
                         "(it becomes a LaTeX macro)")
    return str(k)


def load(path: str | Path) -> dict[str, dict[str, Any]]:
    """The declarations of a numbers file, key -> declaration. Cited entries (``cite:``)
    carry ``_cite`` = {group, study (absolute), snapshot, label}; relative paths resolve
    against the file."""
    path = Path(path).resolve()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in doc.items():
        if k == CITE_KEY:
            continue
        out[_key_ok(path, k)] = dict(v)
    for group, g in (doc.get(CITE_KEY) or {}).items():
        g = dict(g or {})
        bad = sorted(set(g) - set(CITE_FIELDS))
        if bad or not g.get("study") or not g.get("snapshot") or not g.get("numbers"):
            raise ValueError(f"{path.name}: cite group {group!r} needs study, snapshot and "
                             f"numbers (allowed: {', '.join(CITE_FIELDS)}; unknown: {bad})")
        study = (path.parent / str(g["study"])).resolve()
        reuse: dict[str, dict[str, Any]] = {}
        if g.get("from"):
            reuse = load((path.parent / str(g["from"])).resolve())
        meta = {"group": str(group), "study": str(study), "snapshot": str(g["snapshot"]),
                "label": str(g.get("label") or group)}
        if g.get("variant"):              # a snapshot of a sensitivity variant's store
            meta["variant"] = str(g["variant"])
        for k, v in (g["numbers"] or {}).items():
            k = _key_ok(path, k)
            if k in out:
                raise ValueError(f"{path.name}: number {k} declared twice (own and cited)")
            if v == "same" or (isinstance(v, dict) and v.get("same")):
                if k not in reuse:
                    raise ValueError(f"{path.name}: cited number {k}: 'same' needs the key in "
                                     f"the 'from' file of group {group!r}")
                d = {kk: vv for kk, vv in reuse[k].items() if not kk.startswith("_")}
                if isinstance(v, dict):
                    d.update({kk: vv for kk, vv in v.items() if kk != "same"})
            else:
                d = dict(v)
            if d.get("mean_of"):
                raise ValueError(f"{path.name}: cited number {k}: mean_of is resolved in the "
                                 "citing file (declare the mean there over cited numbers)")
            d["_cite"] = meta
            out[k] = d
    return out


def _select(df: Any, filt: dict[str, Any], key: str) -> Any:
    sel = df
    for c, v in (filt or {}).items():
        if c not in sel.columns:
            raise ValueError(f"{key}: filter column {c!r} not in the table")
        sel = sel[sel[c] == v] if not isinstance(v, list) else sel[sel[c].isin(v)]
    if len(sel) != 1:
        raise ValueError(f"{key}: the filter {filt} selects {len(sel)} rows, not 1")
    return sel.iloc[0]


AGGS = ("median", "mean", "sum", "count", "max", "min", "q1", "q3")


def _aggregate(df: Any, filt: dict[str, Any], col: str, agg: str, key: str,
               allow_empty: bool = False) -> Any:
    """One pseudo-row: ``agg`` of ``col`` over the rows the filter selects (n = rows). An
    empty selection is an error (a mistyped filter value), except for ``agg: count`` declared
    with ``allow_empty: true`` (a count that may be zero) when every filter column exists."""
    import pandas as pd

    if agg not in AGGS:
        raise ValueError(f"{key}: agg {agg!r} not in {AGGS}")
    sel = df
    for c, v in (filt or {}).items():
        if c not in sel.columns:
            raise ValueError(f"{key}: filter column {c!r} not in the table")
        sel = sel[sel[c] == v] if not isinstance(v, list) else sel[sel[c].isin(v)]
    if sel.empty:
        if allow_empty and agg == "count":
            return pd.Series({col: 0.0, "n": 0, "_rows": 0,
                              "_n_of": "participants" if "pid" in sel.columns else "rows"})
        raise ValueError(f"{key}: the filter {filt} selects no rows")
    x = pd.to_numeric(sel[col], errors="coerce")
    val = {"median": x.median, "mean": x.mean, "sum": x.sum, "count": lambda: x.notna().sum(),
           "max": x.max, "min": x.min, "q1": lambda: x.astype(float).quantile(0.25),
           "q3": lambda: x.astype(float).quantile(0.75)}[agg]()
    rows = int(x.notna().sum())
    # n counts participants where the table has them (a median over participants and tasks
    # has n = participants, not rows), else rows.
    n = int(sel.loc[x.notna(), "pid"].nunique()) if "pid" in sel.columns else rows
    return pd.Series({col: float(val), "n": n, "_rows": rows,
                      "_n_of": "participants" if "pid" in sel.columns else "rows"})


UNDEFINED = "--"                           # text of a NaN value (e.g. a test not run)
CI_UNDEFINED_TEX = "\\textup{n/a}"         # a CI macro whose interval is undefined


def _fmt(v: float, fmt: str) -> str:
    if isinstance(v, float) and math.isnan(v):
        return UNDEFINED
    t = format(int(round(float(v))), fmt) if fmt.endswith("d") else format(float(v), fmt)
    # a value that rounds to zero is printed without a sign ("-0.000" -> "0.000")
    return t[1:] if t.startswith("-") and not t.strip("-0.") else t


_P_COL = re.compile(r"^p(?:_|$)")


def _p_floor(col: str, val: float, fmt: str, text: str) -> str:
    """A p value that rounds to zero at its decimals prints as the bound ("< 0.001"), so the
    text needs no typed number; siunitx reads the comparator."""
    if not (_P_COL.match(col) and fmt.startswith(".") and fmt.endswith("f")):
        return text
    if not math.isfinite(val) or val < 0 or text.strip("0.") != "":
        return text
    dec = int(fmt[1:-1] or 0)
    return "< " + format(10.0 ** -dec, f".{dec}f") if dec > 0 else text


def open_cited(study: str, snapshot: str, variant: str | None = None) -> Any:
    """The context of a cited study at its snapshot (the default of ``resolve(open_ctx=)``);
    with ``variant``, the snapshot of that sensitivity variant's store (e.g. the data-set
    paper's ``reporting_population`` / ``r15a``; 2026-11-05)."""
    from biomech.config import load_config
    from biomech.io.context import DataContext

    if variant:
        from biomech import sensitivity as S

        return DataContext(S.store_config(load_config(study), variant, snapshot))
    return DataContext(load_config(study).with_snapshot(snapshot))


def _row_n(d: dict[str, Any], row: Any, sel_n: Any = None) -> int | None:
    """The n of a number: ``n:`` a column of the row, an integer, or {distinct: col}
    (``sel_n``: that count over the selected rows); aggregates default to their n."""
    nc = d.get("n")
    if isinstance(nc, bool):
        raise ValueError("n: true/false is not an n (a column, an integer or {distinct: col})")
    if isinstance(nc, int):
        return nc
    if isinstance(nc, dict):
        return sel_n
    nc = nc or ("n" if d.get("agg") else None)
    if nc and nc in row.index and row[nc] == row[nc]:
        return int(row[nc])
    return None


def resolve(ctx: Any, decl: dict[str, dict[str, Any]],
            open_ctx: Any = None) -> list[Number]:
    """Every declared number read from the store ``ctx`` reads (errors name the key); cited
    numbers from their own study's snapshot (``open_ctx(study, snapshot)``, default
    :func:`open_cited`), means (``mean_of``) from the numbers they name. Order: as declared."""
    own = {k: d for k, d in decl.items() if "_cite" not in d and not d.get("mean_of")}
    got = {n.key: n for n in _resolve_rows(ctx, own)}
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for k, d in decl.items():
        if "_cite" in d:
            groups.setdefault((d["_cite"]["study"], d["_cite"]["snapshot"],
                               d["_cite"].get("variant", "")), {})[k] = d
    for (study, snap, var), ds in groups.items():
        c = (open_ctx or open_cited)(study, snap, **({"variant": var} if var else {}))
        try:
            for n in _resolve_rows(c, ds):
                n.cite = dict(ds[n.key]["_cite"])
                got[n.key] = n
        except Exception as exc:
            raise ValueError(f"cited ({Path(study).parent.name} {snap}): {exc}") from exc
        finally:
            if c is not ctx and hasattr(c, "close"):
                c.close()
    for k, d in decl.items():
        if d.get("mean_of"):
            got[k] = _mean(k, d, got)
    return [got[k] for k in decl]


def _mean(key: str, d: dict[str, Any], got: dict[str, Number]) -> Number:
    """``mean_of``: the descriptive mean of other numbers of the file (n = 2)."""
    ops = [str(k) for k in d["mean_of"]]
    if len(ops) < 2:
        raise ValueError(f"{key}: mean_of needs at least two numbers")
    miss = [k for k in ops if k not in got]
    if miss:
        raise ValueError(f"{key}: mean_of names {miss}, not declared (or itself a mean)")
    xs = [got[k] for k in ops]
    if any(isinstance(x.value, str) for x in xs):
        raise ValueError(f"{key}: mean_of over a text value")
    units = {x.unit for x in xs}
    if len(units) > 1:
        raise ValueError(f"{key}: mean_of over different units {sorted(units)}")
    vals = [float(x.value) for x in xs]
    val = float("nan") if any(math.isnan(v) for v in vals) else sum(vals) / len(vals)
    fmt = str(d.get("fmt") or _fmt_of(xs[0]))
    n = d.get("n")
    if n is not None and (isinstance(n, bool) or not isinstance(n, int)):
        raise ValueError(f"{key}: n of a mean must be an integer")
    return Number(key, val, _fmt(val, fmt), str(d.get("unit", xs[0].unit)), None, None,
                  int(n) if n is not None else len(ops), "mean_of:" + ",".join(ops), {}, "",
                  str(d.get("text", "")), None, ops)


def _fmt_of(x: Number) -> str:
    """A format with the decimals of an already formatted number."""
    t = x.text
    return f".{len(t.split('.')[1])}f" if "." in t else ".0f"


def _resolve_rows(ctx: Any, decl: dict[str, dict[str, Any]]) -> list[Number]:
    from biomech import catalog

    out: list[Number] = []
    for key, d in decl.items():
        stage, table = str(d["source"]).split(":", 1)
        if stage == "config":
            out.append(_config_number(ctx, key, table, d))
            continue
        df = (sensitivity_table(ctx, table) if stage == "sensitivity"
              else hw_table(table) if stage == "hw"
              else bench_table(table) if stage == "bench"
              else _derived(ctx, table) if stage == "derived" else ctx.table(stage, table))
        if df.empty:
            raise ValueError(f"{key}: {stage}:{table} is empty or missing")
        col = str(d["column"])
        if col not in df.columns:
            raise ValueError(f"{key}: column {col!r} not in {stage}:{table}")
        if d.get("agg"):
            row = _aggregate(df, d.get("filter") or {}, col, str(d["agg"]), key,
                             bool(d.get("allow_empty")))
        else:
            row = _select(df, d.get("filter") or {}, key)
        sel_n = None
        if isinstance(d.get("n"), dict):
            dc = str((d["n"] or {}).get("distinct", ""))
            if dc not in df.columns:
                raise ValueError(f"{key}: n: {{distinct: {dc}}} -- no column {dc!r}")
            sel = df
            for c, v in (d.get("filter") or {}).items():
                sel = sel[sel[c] == v] if not isinstance(v, list) else sel[sel[c].isin(v)]
            sel_n = int(sel[dc].nunique())
        e = catalog.get(col)
        scale = float(d.get("scale", 1.0))
        raw = row[col]
        if isinstance(raw, str):
            num = Number(key, raw, raw, source=f"{stage}:{table}", filter=d.get("filter") or {},
                         column=col, what=str(d.get("text", "")))
            out.append(num)
            continue
        val = float(raw) * scale
        fmt = str(d.get("fmt") or f".{e.decimals}f")
        if fmt == "req":                # the requirements table's rounding (Table I = text)
            from biomech.viz.views.paper import req_decimals

            ref = float(row["value"]) * scale if "value" in row.index else val
            fmt = f".{req_decimals(ref)}f"
        unit = str(d.get("unit", e.unit))
        ci = ci_text = None
        if d.get("ci"):
            lo, hi = (float(row[c]) * scale for c in d["ci"])
            ci, ci_text = (lo, hi), (_fmt(lo, fmt), _fmt(hi, fmt))
        try:
            n = _row_n(d, row, sel_n)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
        rows = int(row["_rows"]) if "_rows" in row.index else None
        n_of = str(d.get("n_of") or (row["_n_of"] if "_n_of" in row.index and not d.get("n")
                                     else "participants"))
        out.append(Number(key, val, _p_floor(col, val, fmt, _fmt(val, fmt)), unit, ci, ci_text,
                          n, f"{stage}:{table}",
                          d.get("filter") or {}, col, str(d.get("text", "")),
                          n_rows=rows if rows != n else None, n_of=n_of))
    return out


def _derived(ctx: Any, name: str) -> Any:
    from biomech.viz import derived

    return derived.table(ctx, name)


def _config_number(ctx: Any, key: str, path: str, d: dict[str, Any]) -> Number:
    """``source: config:<dotted path>``: a method parameter as the store was computed with it
    (a snapshot's frozen configuration, else the loaded one); ``scale``, ``fmt``, ``unit``."""
    node: Any = getattr(ctx.config, "raw", None) or {}
    for part in path.split("."):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]                  # list element: bandpass_hz.0
            continue
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{key}: config:{path} not in the configuration")
        node = node[part]
    if isinstance(node, (list, dict)) or node is None:
        raise ValueError(f"{key}: config:{path} is not a single value ({node!r})")
    if isinstance(node, str):
        return Number(key, node, node, source=f"config:{path}", what=str(d.get("text", "")))
    val = float(node) * float(d.get("scale", 1.0))
    fmt = str(d.get("fmt") or ("g" if not float(val).is_integer() else ".0f"))
    return Number(key, val, _fmt(val, fmt), str(d.get("unit", "")), source=f"config:{path}",
                  what=str(d.get("text", "")), n_of="none")


HW_TABLES = Path(__file__).resolve().parents[4] / "hardware" / "mechanical" / "tables"


def hw_table(name: str) -> Any:
    """``source: hw:<table>``: a generated table of the mechanical parts
    (``hardware/mechanical/tables/<table>.csv``; J13); empty when absent."""
    import pandas as pd

    p = HW_TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


BENCH_TABLES = Path(__file__).resolve().parents[4] / "hardware" / "bench" / "tables"


def bench_table(name: str) -> Any:
    """``source: bench:<table>``: a table of a hardware bench check
    (``hardware/bench/tables/<table>.csv``, written by ``scripts/bench/``; e.g. the B1
    electrode-impedance scale check); empty when absent."""
    import pandas as pd

    p = BENCH_TABLES / f"{name}.csv"
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def sensitivity_table(ctx: Any, variant: str) -> Any:
    """``source: sensitivity:<variant>``: the variant's comparison with the primary run in the
    store ``ctx`` reads (:func:`biomech.sensitivity.comparison_path`: under
    ``sensitivity/<variant>/`` of a primary store, at the root of that variant's own store; a
    snapshot gives the comparison frozen with it). Its row keys ("key=x|task=y") become columns
    ``@key``, ``@task``, ... for the filter; the number is ``primary`` or ``variant``."""
    import pandas as pd

    from biomech.sensitivity import comparison_path

    path = comparison_path(ctx.config, variant)
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    parts = df["key"].astype(str).str.split("|")
    keyed = pd.DataFrame([dict(p.split("=", 1) for p in row if "=" in p) for row in parts],
                         index=df.index)
    return pd.concat([df, keyed.add_prefix("@")], axis=1)


def _tex(n: Number) -> list[str]:
    m = n.macro()
    if isinstance(n.value, str):
        from biomech.viz import render_pgf

        return [f"\\newcommand{{\\{m}}}{{{render_pgf._esc(n.text)}}}"]
    lines = [f"\\newcommand{{\\{m}}}{{\\qty{{{n.text}}}{{{n.unit}}}}}" if n.unit
             else f"\\newcommand{{\\{m}}}{{\\num{{{n.text}}}}}",
             f"\\newcommand{{\\{m}Val}}{{\\num{{{n.text}}}}}"]
    if n.ci_text:
        lo, hi = n.ci_text
        # "[35.2, 41.2] mm" (degrees without a space: "[19.8, 25.4]°")
        unit = "" if not n.unit else ("\\unit{" + n.unit + "}" if n.unit == "\\degree"
                                      else "\\,\\unit{" + n.unit + "}")
        if UNDEFINED in (lo, hi):
            # an interval that could not be computed (too few values, or every value the
            # same): one convention for every CI macro, never "\num{--}" (paper revision
            # 2026-11-09); numbers.json keeps the NaN bounds
            lines.append(f"\\newcommand{{\\{m}CI}}{{{CI_UNDEFINED_TEX}}}")
        else:
            lines.append(f"\\newcommand{{\\{m}CI}}{{[\\num{{{lo}}}, \\num{{{hi}}}]{unit}}}")
    if n.n is not None:
        lines.append(f"\\newcommand{{\\{m}N}}{{{n.n}}}")
    return lines


def write(out: str | Path, numbers: list[Number], provenance: dict[str, Any],
          source_file: str | Path | None = None) -> tuple[Path, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    def entry(n: Number) -> dict[str, Any]:
        e = {"macro": "\\" + n.macro(), "value": n.value, "text": n.text, "unit": n.unit,
             "ci": n.ci, "ci_text": n.ci_text, "n": n.n, "source": n.source,
             "filter": n.filter, "column": n.column, "what": n.what}
        if n.n is not None:
            e["n_of"] = n.n_of
        if n.n_rows is not None:
            e["n_rows"] = n.n_rows
        if n.mean_of:
            e["mean_of"] = n.mean_of
        if n.cite:
            e.update({"cited": True, "study": n.cite["study"], "snapshot": n.cite["snapshot"],
                      "cited_as": n.cite["label"]})
        return e

    doc = {"schema": 2, "provenance": provenance,
           "source_file": str(source_file) if source_file else None,
           "numbers": {n.key: entry(n) for n in numbers if not n.cite},
           "cited": {n.key: entry(n) for n in numbers if n.cite}}
    jp = out / "numbers.json"
    jp.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["% In-text numbers (biomech numbers), resolved from snapshot "
             f"{provenance.get('snapshot')}. Needs siunitx (v3).",
             "% \\numX value with unit, \\numXVal bare value, \\numXCI [lo, hi], \\numXN n."]
    for n in numbers:
        if not n.cite:
            lines += _tex(n)
    cited = [n for n in numbers if n.cite]
    for src in dict.fromkeys((n.cite["label"], n.cite["snapshot"]) for n in cited):
        lines.append(f"% Cited from the {src[0]} (snapshot {src[1]}): cite, do not re-report.")
        for n in cited:
            if (n.cite["label"], n.cite["snapshot"]) == src:
                lines += _tex(n)
    # One definition per macro: a key "XN" next to a key "X" with an n field would define
    # \numXN twice and LaTeX stops at the second \newcommand.
    names = [ln[len("\\newcommand{"):ln.index("}")] for ln in lines
             if ln.startswith("\\newcommand{")]
    dup = sorted({m for m in names if names.count(m) > 1})
    if dup:
        raise ValueError(f"numbers: macros defined twice: {', '.join(dup)} (a key and the "
                         "N / CI / Val macro of another key)")
    tp = out / "numbers.tex"
    tp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, tp


def written(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every number of a ``numbers.json`` document, own and cited."""
    return {**(doc.get("numbers") or {}), **(doc.get("cited") or {})}


def verify(out: str | Path, ctx: Any, open_ctx: Any = None) -> list[str]:
    """``numbers.json`` of an export against the store: every number re-resolved now from
    its declaration (the numbers file named in it) -- cited numbers from their study's
    snapshot, means from their operands -- must equal the written value, CI and n."""
    out = Path(out)
    jp = out / "numbers.json"
    if not jp.is_file():
        return []
    doc = json.loads(jp.read_text(encoding="utf-8"))
    src = doc.get("source_file")
    if not src or not Path(src).is_file():
        return [f"numbers.json: declaration file {src} not found"]
    try:
        now = {n.key: n for n in resolve(ctx, load(src), open_ctx)}
    except Exception as exc:
        return [f"numbers: {exc}"]
    probs = []
    old = written(doc)
    for k in sorted(set(old) | set(now)):
        if k not in now:
            probs.append(f"number {k}: no longer declared")
        elif k not in old:
            probs.append(f"number {k}: declared but not in numbers.json (export again)")
        elif old[k]["text"] != now[k].text or old[k].get("ci_text") != (
                list(now[k].ci_text) if now[k].ci_text else None):
            probs.append(f"number {k}: written {old[k]['text']}, store now {now[k].text}")
        elif old[k].get("n") != now[k].n:
            probs.append(f"number {k}: written n = {old[k].get('n')}, store now n = {now[k].n}")
        elif bool(old[k].get("cited")) != bool(now[k].cite) or (
                now[k].cite and old[k].get("snapshot") != now[k].cite["snapshot"]):
            probs.append(f"number {k}: cited source changed (export again)")
    return probs


MACRO_USE_RE = re.compile(r"\\(num[A-Z][A-Za-z]*)")
CAPTION_USE_RE = re.compile(r"\\fig(?:caption|sources)\{([A-Za-z]+)\}")


def check_manuscript(tex_dir: str | Path, export_dirs: list[str | Path]) -> dict[str, Any]:
    """Macros and caption labels a manuscript uses vs those the exports define:
    ``undefined`` (used, not defined: an error), ``unused`` (defined, never used),
    ``uncited`` (cited numbers without a citation in their sentence) and ``populations``
    (export folder -> the variant its store was, ``primary`` for the primary analysis, from
    the provenance of its numbers.json / manifest.json; one manuscript must report one
    population)."""
    defined: set[str] = set()
    cited: set[str] = set()
    labels: set[str] = set()
    populations: dict[str, str] = {}
    other_pop: set[str] = set()              # numbers of another population (D36)
    for d in export_dirs:
        d = Path(d)
        for f in ("numbers.json", "manifest.json"):
            if (d / f).is_file():
                prov = json.loads((d / f).read_text()).get("provenance") or {}
                populations.setdefault(str(d), prov.get("variant") or "primary")
        if (d / "numbers.json").is_file():
            doc = json.loads((d / "numbers.json").read_text())
            store_variant = (doc.get("provenance") or {}).get("variant")
            for v in written(doc).values():
                base = v["macro"].lstrip("\\")
                defined |= {base, base + "Val", base + "CI", base + "N"}
                if v.get("cited"):
                    cited.add(base)
                if store_variant and _other_population(v, store_variant):
                    other_pop.add(base)
        if (d / "manifest.json").is_file():
            labels |= {i["label"] for i in json.loads((d / "manifest.json").read_text())
                       ["items"] if i.get("label")}
    used: set[str] = set()
    used_labels: set[str] = set()
    uncited: set[str] = set()
    for f in Path(tex_dir).rglob("*.tex"):
        if any(str(f).startswith(str(Path(d).resolve())) for d in export_dirs):
            continue
        # uses only: not in TeX comments (documentation such as "\\numX, \\numXVal") and
        # not the \\newcommand definitions of a copied numbers file
        t = _uses_only(f.read_text(errors="ignore"))
        used |= set(MACRO_USE_RE.findall(t))
        used_labels |= set(CAPTION_USE_RE.findall(t))
        if cited:
            for sent in SENTENCE_RE.split(t):
                ms = {_base(m, cited) for m in MACRO_USE_RE.findall(sent)} & cited
                if ms and not CITE_RE.search(sent):
                    uncited |= ms
    bases = {m for m in defined if not m.endswith(("Val", "CI", "N"))}
    return {"undefined": sorted(used - defined) + sorted(f"figcaption{{{x}}}"
                                                         for x in used_labels - labels),
            "unused": sorted(b for b in bases
                             if not any(u.startswith(b) for u in used)),
            "uncited": sorted(uncited), "populations": populations,
            "other_population": sorted(b for b in other_pop
                                       if any(_base(u, {b}) == b for u in used))}


_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_DEFINE_RE = re.compile(r"\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\num[A-Za-z]*\}?")


def _uses_only(text: str) -> str:
    """TeX source without comments and without the names in \\newcommand{\\numX} (a
    definition is not a use)."""
    return _DEFINE_RE.sub("", _COMMENT_RE.sub("", text))


def _other_population(v: dict[str, Any], store_variant: str) -> bool:
    """A number of another reporting population than the export's store (D36): the
    primary column of the store variant's own comparison, i.e. the analysis of record with all
    participants (the ``…All`` keys), which a paper on ``store_variant`` must not quote."""
    src = str(v.get("source") or "")
    return src == f"sensitivity:{store_variant}" and v.get("column") == "primary"


# A cited number (another paper's result) needs a citation in its sentence.
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])|\n\s*\n")
CITE_RE = re.compile(r"\\(?:cite[a-zA-Z]*|ref)\{")


def _base(macro: str, bases: set[str]) -> str:
    """The number a macro belongs to (``numXCI`` -> ``numX`` when ``numX`` is in ``bases``)."""
    for suf in ("Val", "CI", "N"):
        if macro.endswith(suf) and macro[: -len(suf)] in bases:
            return macro[: -len(suf)]
    return macro
