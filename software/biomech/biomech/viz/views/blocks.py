# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared building blocks for views: one implementation per recurring panel form.

Every function returns a :class:`~biomech.viz.spec.Panel` (or a TableSpec) built from plain
arrays or a long DataFrame, so a new analysis view is mostly a call or two:

* :func:`strip` / :func:`strip_by` -- horizontal strip per category: participants as grey
  dots, the median as an accent diamond (optionally its interval), an optional reference;
* :func:`matrix_heat` / :func:`heat` -- participant (or task) x muscle heatmaps from a long
  table, fixed entity order, pseudonym rows, values printed, colourbar with unit;
* :func:`curves_with_band` -- grey per-participant curves, the cohort centre and band;
* :func:`spectrum` -- power spectra (log power) with marker lines (f50/f95, harmonics);
* :func:`spm_panel` -- an SPM{t}/{F} curve with its critical threshold and clusters;
* :func:`forest` -- estimates with intervals, one row each, a reference line;
* :func:`dumbbell` -- two values per row joined (e.g. before/after, two variants);
* :func:`event_raster` -- event times per row (onsets, decisions);
* :func:`histogram` -- a distribution (e.g. a null) with an observed value marked;
* :func:`strip_mark` -- a strip per category with a marked statistic (e.g. P95) and its CI;
* :func:`ecdf` -- empirical CDFs (steps), optionally a percentile with its CI marked;
* :func:`slopegraph` -- per-participant paired values across two conditions, joined, with
  the median and its interval per condition;
* :func:`paired_forest` -- two estimates with intervals per row (e.g. primary vs variant);
* :func:`spec_curve` -- a specification curve: sorted estimates with CIs over an indicator
  matrix of the analysis choices (two panels sharing the x axis);
* :func:`frame_table` -- a DataFrame as a TableSpec with catalog headers;
* uncertainty (R5 E): :func:`boot_curve_ci` / :func:`add_ci_band` / :func:`add_curves_ci`
  (pointwise bootstrap CI bands of a cohort curve), :func:`median_ci` / :func:`strip_ci` /
  :func:`strip_by_ci` (strips with the BCa CI of the median), :func:`paired_strip` (two
  conditions per row instead of a dumbbell of medians), :func:`rows_height` (n-aware height).
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd

from biomech import catalog
from biomech.viz.spec import Axis, Colorbar, Column, Panel, Series, TableSpec
from biomech.viz.views import vocab as V


def _finite(v: Any) -> np.ndarray:
    a = np.asarray(v, float).ravel()
    return a[np.isfinite(a)]


def pseudo_rows(ctx: Any, pids: Sequence[str]) -> list[str]:
    """Row labels for participants: pseudonyms only (never study ids in figures)."""
    return [f"P{ctx.pseudonym(p)}" for p in pids]


# -- strips -------------------------------------------------------------------------------
def strip(labels: Sequence[str], values: Sequence[Any], title: str, xaxis: Axis, *,
          ref: float | None = None, ref_label: str | None = None,
          intervals: Sequence[tuple[float, float]] | None = None,
          roles: Sequence[str] | None = None) -> Panel:
    """One row per category (labels on the y axis, never rotated)."""
    p = Panel(title, xaxis, Axis(""), [], legend="none")
    for i, v in enumerate(values):
        v = _finite(v)
        if intervals is not None and i < len(intervals) and all(
                np.isfinite(intervals[i])):
            lo, hi = intervals[i]
            p.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                   role="accent", style={"width": 1.6}))
        if not v.size:
            continue
        jit = np.linspace(-0.18, 0.18, v.size) if v.size > 1 else np.zeros(1)
        p.series.append(Series("scatter", v, np.full(v.size, float(i)) + jit,
                               role=roles[i] if roles else "context"))
        p.series.append(Series("scatter", np.array([np.median(v)]), np.array([float(i)]),
                               role="accent", style={"size": 1.5, "ms": "d"}))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.reverse = True
    return p


def strip_by(df: pd.DataFrame, value: str, by: str, title: str, *,
             order: Sequence[str] | None = None, labels: dict[str, str] | None = None,
             xaxis: Axis | None = None, **kw: Any) -> Panel:
    """:func:`strip` of a long table: one row per value of ``by`` (in ``order``)."""
    if df.empty or value not in df:
        return Panel(title, xaxis or V.axis(value), Axis(""), [], legend="none",
                     note="no data")
    cats = [c for c in (order or sorted(df[by].dropna().unique())) if c in set(df[by])]
    vals = [df.loc[df[by] == c, value].to_numpy(float) for c in cats]
    lab = [(labels or {}).get(c, V.task_label(c) if by == "task" else
                              V.muscle_label(c) if by == "muscle" else str(c)) for c in cats]
    return strip(lab, vals, title, xaxis or V.axis(value), **kw)


# -- heatmaps ---------------------------------------------------------------------------------
def heat(z: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, *,
         fmt: str = "{:.0f}", cb: Colorbar | None = None) -> Panel:
    z = np.asarray(z, float)
    p = Panel(title, Axis(""), Axis(""), [Series("heatmap", z=z, text=[
        fmt.format(v) if np.isfinite(v) else "" for v in z.ravel()])], legend="none",
              colorbar=cb)
    p.x.ticks, p.x.ticklabels = [float(i) for i in range(len(cols))], list(cols)
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(rows))], list(rows)
    return p


def matrix_heat(df: pd.DataFrame, row: str, col: str, value: str, title: str, *,
                rows: Sequence[str] | None = None, cols: Sequence[str] | None = None,
                row_labels: Sequence[str] | None = None,
                col_labels: Sequence[str] | None = None, scale: float = 1.0,
                fmt: str = "{:.0f}", cb: Colorbar | None = None,
                agg: str = "median") -> Panel:
    """Pivot of a long table (``agg`` over duplicates), fixed row/column order."""
    if df.empty or value not in df:
        return Panel(title, Axis(""), Axis(""), [], legend="none", note="no data")
    piv = df.pivot_table(index=row, columns=col, values=value, aggfunc=agg)
    rows = list(rows) if rows is not None else list(piv.index)
    cols = list(cols) if cols is not None else (
        [m for m in V.MUSCLES if m in piv.columns] if col == "muscle" else list(piv.columns))
    z = piv.reindex(index=rows, columns=cols).to_numpy(float) * scale
    rl = list(row_labels) if row_labels is not None else [
        V.task_label(r) if row == "task" else V.muscle_label(r) if row == "muscle" else str(r)
        for r in rows]
    cl = list(col_labels) if col_labels is not None else [
        V.muscle_label(c) if col == "muscle" else V.task_label(c) if col == "task" else str(c)
        for c in cols]
    return heat(z, rl, cl, title, fmt=fmt, cb=cb)


# -- curves -------------------------------------------------------------------------------------
def curves_with_band(x: np.ndarray, subjects: np.ndarray, title: str, xaxis: Axis,
                     yaxis: Axis, *, centre: str = "mean", label: str | None = None,
                     role: str = "accent", show_subjects: bool = True,
                     spread: np.ndarray | None = None) -> Panel:
    """Grey per-participant curves (rows of ``subjects``) + centre (mean/median) + band
    (SD or IQR, or an explicit ``spread`` (2, N) = lower/upper)."""
    p = Panel(title, xaxis, yaxis, [])
    add_curves(p, x, subjects, centre=centre, label=label, role=role,
               show_subjects=show_subjects, spread=spread)
    return p


def add_curves(p: Panel, x: np.ndarray, subjects: np.ndarray, *, centre: str = "mean",
               label: str | None = None, role: str = "accent", show_subjects: bool = True,
               spread: np.ndarray | None = None) -> Panel:
    S = np.atleast_2d(np.asarray(subjects, float))
    x = np.asarray(x, float)
    if S.size == 0 or not np.isfinite(S).any():
        return p
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if centre == "median":
            c = np.nanmedian(S, axis=0)
            lo, hi = np.nanpercentile(S, [25, 75], axis=0)
        else:
            c = np.nanmean(S, axis=0)
            sd = np.nanstd(S, axis=0, ddof=1) if len(S) > 1 else np.zeros_like(c)
            lo, hi = c - sd, c + sd
    if spread is not None:
        lo, hi = np.asarray(spread, float)
    if show_subjects:
        for row in S:
            p.series.append(Series("line", x, row, role="context"))
    if len(S) > 1 or spread is not None:
        p.series.append(Series("band", x, lo, y2=hi, role=role))
    p.series.append(Series("line", x, c, role=role, label=label))
    return p


# -- SPM shown as significance bars (user convention, Mohseni et al. BioRob 2026) ---------
def parse_clusters(v: Any) -> list[tuple[float, float, float]]:
    """Supra-threshold clusters as stored (a list, or its text form "[(a, b, p), ...]"):
    [(start, end, p)] in the test's own x units; [] for none, a count, or garbage."""
    import ast

    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(v, (list, tuple)):
        return []                                  # e.g. a stored cluster count
    out = []
    for c in v:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            p = float(c[2]) if len(c) > 2 and c[2] is not None else np.nan
            out.append((float(c[0]), float(c[1]), p))
    return out


def add_spm_bars(p: Panel, tests: Sequence[tuple[str, Any, str]]) -> int:
    """Thin significance bars over the supra-threshold clusters, above the curves of ``p``:
    one bar row per test ``(label, clusters, role)`` (clusters as stored, see
    :func:`parse_clusters`), coloured by ``role`` (the group or comparison it refers to),
    labelled "SPM: <label>" in the legend. The renderers reserve the head-room so bars never
    cover data (layout.SIGBAR_ROW). Returns the number of tests that have clusters."""
    n = 0
    for row, (label, clusters, role) in enumerate(tests):
        cl = parse_clusters(clusters)
        if not cl:
            continue
        n += 1
        p.series.append(Series("sigbar", np.array([c[0] for c in cl], float),
                               y2=np.array([c[1] for c in cl], float), role=role,
                               label=f"SPM: {label}" if label else None,
                               style={"row": row}))
    return n


SPM_BARS_CAPTION = ("bars above the curves: supra-threshold clusters of the SPM test "
                    "(p < 0.05")


def spm_caption(test: str, correction: str = "") -> str:
    """Caption fragment for the bars, e.g. '... (p < 0.05, paired SPM{t}, Holm)'."""
    return f"{SPM_BARS_CAPTION}, {test}{', ' + correction if correction else ''})"


def spectrum(freq: np.ndarray, curves: dict[str, np.ndarray], title: str, *,
             yaxis: Axis | None = None, markers: dict[str, float] | None = None,
             roles: dict[str, str] | None = None, log_power: bool = True,
             xlim: tuple[float, float] | None = None) -> Panel:
    """Power spectra over frequency; ``log_power`` plots 10 log10 (dB)."""
    f = np.asarray(freq, float)
    p = Panel(title, Axis("Frequency", V.HZ, lim=xlim),
              yaxis or Axis("Power", V.DB if log_power else ""), [])
    for i, (lab, y) in enumerate(curves.items()):
        y = np.asarray(y, float)
        if log_power:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = 10.0 * np.log10(np.where(y > 0, y, np.nan))
        p.series.append(Series("line", f, y, label=lab,
                               role=(roles or {}).get(lab, f"cat:{i}")))
    for lab, fx in (markers or {}).items():
        if fx is not None and np.isfinite(fx):
            p.series.append(Series("vline", x=np.array([float(fx)]), role="reference",
                                   label=lab))
    return p


def spm_panel(x: np.ndarray, z: np.ndarray, zstar: float, title: str, *,
              clusters: Sequence[tuple[float, float]] = (), xaxis: Axis | None = None,
              stat: str = "SPM{t}", two_tailed: bool = True) -> Panel:
    p = Panel(title, xaxis or Axis("Cycle", V.PCT), Axis(stat), [
        Series("line", np.asarray(x, float), np.asarray(z, float), role="primary",
               label=stat)])
    if np.isfinite(zstar):
        p.series.append(Series("hline", y=np.array([float(zstar)]), role="threshold",
                               label="critical"))
        if two_tailed:
            p.series.append(Series("hline", y=np.array([-float(zstar)]), role="threshold"))
    for a, b in clusters:
        p.series.append(Series("hspan", x=np.array([float(a)]), y2=np.array([float(b)]),
                               role="faint"))
    return p


def forest(labels: Sequence[str], est: Sequence[float], lo: Sequence[float],
           hi: Sequence[float], title: str, xaxis: Axis, *, ref: float | None = 0.0,
           ref_label: str | None = None, role: str = "primary") -> Panel:
    """Estimates (accent diamonds) with intervals (``role``), one row each."""
    p = Panel(title, xaxis, Axis(""), [], legend="none")
    for i, (e, a, b) in enumerate(zip(est, lo, hi)):
        if np.isfinite(a) and np.isfinite(b):
            p.series.append(Series("line", np.array([a, b], float), np.array([i, i], float),
                                   role=role, style={"width": 1.4}))
        if np.isfinite(e):
            p.series.append(Series("scatter", np.array([e], float), np.array([i], float),
                                   role="accent", style={"ms": "d", "size": 1.4}))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.reverse = True
    return p


def dumbbell(labels: Sequence[str], a: Sequence[float], b: Sequence[float], title: str,
             xaxis: Axis, *, label_a: str = "A", label_b: str = "B") -> Panel:
    a, b = np.asarray(a, float), np.asarray(b, float)
    y = np.arange(len(labels), dtype=float)
    p = Panel(title, xaxis, Axis(""), [])
    for i in range(len(labels)):
        if np.isfinite(a[i]) and np.isfinite(b[i]):
            p.series.append(Series("line", np.array([a[i], b[i]]), np.array([y[i], y[i]]),
                                   role="context"))
    p.series.append(Series("scatter", a, y, role="cat:0", label=label_a))
    p.series.append(Series("scatter", b, y, role="cat:1", label=label_b))
    p.y.ticks, p.y.ticklabels = list(y), list(labels)
    p.y.reverse = True
    return p


def event_raster(labels: Sequence[str], events: Sequence[Any], title: str, xaxis: Axis, *,
                 ref: float | None = 0.0, ref_label: str | None = None,
                 roles: Sequence[str] | None = None) -> Panel:
    p = Panel(title, xaxis, Axis(""), [], legend="none")
    for i, ev in enumerate(events):
        ev = _finite(ev)
        if ev.size:
            p.series.append(Series("scatter", ev, np.full(ev.size, float(i)),
                                   role=roles[i] if roles else "primary",
                                   style={"ms": "|", "size": 1.2}))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.reverse = True
    return p


def histogram(values: Any, title: str, xaxis: Axis, *, bins: int = 30,
              observed: float | None = None, observed_label: str = "observed",
              quantile: float | None = None) -> Panel:
    v = _finite(values)
    p = Panel(title, xaxis, Axis("Count"), [], legend="none")
    if v.size:
        cnt, edges = np.histogram(v, bins=min(bins, max(5, v.size // 3)))
        p.series.append(Series("bar", 0.5 * (edges[:-1] + edges[1:]), cnt.astype(float),
                               role="context", style={"bar_width": float(np.diff(edges)[0])}))
        if quantile is not None:
            p.series.append(Series("vline", x=np.array([np.quantile(v, quantile)]),
                                   role="threshold", label=f"P{100 * quantile:.0f}"))
    if observed is not None and np.isfinite(observed):
        p.series.append(Series("vline", x=np.array([float(observed)]), role="accent",
                               label=observed_label))
    return p


# -- distributions, paired values, specification curves (journal set) ---------------------------
def strip_mark(labels: Sequence[str], values: Sequence[Any], title: str, xaxis: Axis, *,
               marks: Sequence[tuple[float, float, float]] | None = None,
               mark_label: str = "P95 [95 % CI]", unit_label: str = "participants",
               flagged: Sequence[Any] | None = None, flagged_label: str | None = None,
               ref: float | None = None, ref_label: str | None = None,
               mark_role: str = "accent") -> Panel:
    """One row per category: participants as grey dots, a marked statistic per row (accent
    bar marker, e.g. the 95th percentile) with its interval (a one-sided lower bound, upper
    end NaN: drawn from the bound to the value); ``flagged`` values per row are drawn light
    (shown, not used for the mark)."""
    p = Panel(title, xaxis, Axis(""), [], legend="auto")
    first = True
    for i, v in enumerate(values):
        v = _finite(v)
        if v.size:
            jit = np.linspace(-0.16, 0.16, v.size) if v.size > 1 else np.zeros(1)
            p.series.append(Series("scatter", v, np.full(v.size, float(i)) + jit, role="context",
                                   label=unit_label if first else None))
            first = False
    lab_f = flagged_label
    for i, v in enumerate(flagged or []):
        v = _finite(v)
        if v.size:
            jit = np.linspace(-0.16, 0.16, v.size) if v.size > 1 else np.zeros(1)
            p.series.append(Series("scatter", v, np.full(v.size, float(i)) + jit, role="flagged",
                                   label=lab_f))
            lab_f = None
    lab_m = mark_label
    for i, m in enumerate(marks or []):
        val, lo, hi = (float(x) for x in m)
        # a one-sided lower bound (no upper end, E31): the interval runs from it to the value
        hi_ = hi if np.isfinite(hi) else val
        if np.isfinite(lo) and np.isfinite(hi_):
            p.series.append(Series("line", np.array([lo, hi_]), np.array([i, i], float),
                                   role=mark_role, style={"width": 1.4}))
        if np.isfinite(val):
            p.series.append(Series("scatter", np.array([val]), np.array([float(i)]),
                                   role=mark_role, style={"ms": "|", "size": 2.2}, label=lab_m))
            lab_m = None
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.lim = (-0.5, len(labels) - 0.5) if labels else None
    p.y.reverse = True
    return p


def ecdf_steps(values: Any) -> tuple[np.ndarray, np.ndarray]:
    """Step coordinates of the empirical CDF of the finite ``values`` in percent: x repeats
    each sorted value, y runs 0 -> 100 (the curve starts at 0 at the smallest value)."""
    v = np.sort(_finite(values))
    if not v.size:
        return np.empty(0), np.empty(0)
    y = 100.0 * np.arange(1, v.size + 1) / v.size
    return np.repeat(v, 2), np.r_[0.0, np.repeat(y, 2)[:-1]]


def ecdf(curves: dict[str | None, Any] | Any, title: str, xaxis: Axis, *,
         roles: dict[str | None, str] | Sequence[str] | None = None,
         mark: dict[str, Any] | None = None, points: bool = True,
         yaxis: Axis | None = None, ref: float | None = None,
         ref_label: str | None = None, mark_role: str = "accent") -> Panel:
    """Empirical CDF per series (share of units at or below x, %), steps; ``points`` puts a
    dot on every unit (small n). ``mark`` = ``{"p": 95, "value": v, "lo": l, "hi": h}`` marks a
    percentile: dashed lines at the value and at p %, the CI as a shaded x-range, the value as
    an accent diamond (``mark["label"]`` names it in the legend, ``mark["note"]`` goes to the
    panel note)."""
    if not isinstance(curves, dict):
        curves = {None: curves}
    p = Panel(title, xaxis, yaxis or Axis("Cumulative share", "\\percent", lim=(0.0, 104.0)),
              [], legend="auto")
    if mark is not None:
        val, lo, hi = (float(mark.get(k, np.nan)) for k in ("value", "lo", "hi"))
        hi = hi if np.isfinite(hi) else val          # one-sided lower bound: shade lo..value
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("hspan", x=np.array([lo]), y2=np.array([hi]), role="muted"))
    for i, (lab, vals) in enumerate(curves.items()):
        if isinstance(roles, dict):
            role = roles.get(lab, f"cat:{i}")
        elif roles is not None:
            role = roles[i]
        else:
            role = "accent" if len(curves) == 1 else f"cat:{i}"
        x, y = ecdf_steps(vals)
        if not x.size:
            continue
        p.series.append(Series("line", x, y, role=role, label=lab))
        if points:
            v = np.sort(_finite(vals))
            p.series.append(Series("scatter", v, 100.0 * np.arange(1, v.size + 1) / v.size,
                                   role=role, style={"size": 0.8}))
    if mark is not None and np.isfinite(float(mark.get("value", np.nan))):
        val, pct = float(mark["value"]), float(mark.get("p", 95.0))
        p.series += [Series("vline", x=np.array([val]), role="threshold"),
                     Series("hline", y=np.array([pct]), role="reference"),
                     Series("scatter", np.array([val]), np.array([pct]), role=mark_role,
                            style={"ms": "d", "size": 1.6}, label=mark.get("label"))]
        p.note = mark.get("note", "")
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    if not p.series:
        p.note = "no data"
    return p


def slopegraph(conditions: Sequence[str], a: Sequence[float], b: Sequence[float], title: str,
               yaxis: Axis, *, summary: Sequence[tuple[float, float, float]] | None = None,
               summary_label: str = "median [95 % CI]", unit_label: str = "participants",
               lone_label: str | None = "one condition only",
               annot: Sequence[Sequence[str]] | None = None, ref: float | None = None,
               ref_label: str | None = None) -> Panel:
    """Paired values per unit (participant) in two conditions: a grey line per unit with
    both values, units with one value as light dots, and per condition the summary
    (``(centre, lo, hi)``) as an accent diamond with its interval beside the column.
    ``annot``: text lines per condition stacked above the data (e.g. n and p)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    p = Panel(title, Axis(""), yaxis, [], legend="auto")
    both = np.isfinite(a) & np.isfinite(b)
    for va, vb in zip(a[both], b[both]):
        p.series.append(Series("line", np.array([0.0, 1.0]), np.array([va, vb]),
                               role="context"))
    if both.any():
        k = int(both.sum())
        p.series.append(Series("scatter", np.r_[np.zeros(k), np.ones(k)],
                               np.r_[a[both], b[both]], role="context", label=unit_label,
                               style={"size": 1.0}))
    lone_x = np.r_[np.zeros(int((np.isfinite(a) & ~both).sum())),
                   np.ones(int((np.isfinite(b) & ~both).sum()))]
    lone_y = np.r_[a[np.isfinite(a) & ~both], b[np.isfinite(b) & ~both]]
    if lone_y.size:
        p.series.append(Series("scatter", lone_x, lone_y, role="faint",
                               style={"ms": "s", "size": 1.0}, label=lone_label))
    lab = summary_label
    ys = [a, b]
    for j, s in enumerate(summary or []):
        c, lo, hi = (float(v) for v in s)
        xj = -0.2 if j == 0 else 1.2
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("line", np.array([xj, xj]), np.array([lo, hi]),
                                   role="accent", style={"width": 1.4}))
            ys.append(np.array([lo, hi]))
        if np.isfinite(c):
            p.series.append(Series("scatter", np.array([xj]), np.array([c]), role="accent",
                                   style={"ms": "d", "size": 1.5}, label=lab))
            lab = None
    if ref is not None:
        p.series.append(Series("hline", y=np.array([float(ref)]), role="reference",
                               label=ref_label))
        ys.append(np.array([float(ref)]))
    if annot:
        v = _finite(np.concatenate([np.ravel(y) for y in ys]))
        if v.size:
            top, span = float(v.max()), float(np.ptp(v)) or 1.0
            n = max(len(lines) for lines in annot)
            for j, lines in enumerate(annot):
                for k, t in enumerate(lines):
                    p.series.append(Series("text", np.array([float(j)]),
                                           np.array([top + span * (0.08 + 0.13 * (n - 1 - k))]),
                                           text=[t], style={"anchor": "above"}))
    p.x.ticks, p.x.ticklabels = [0.0, 1.0], list(conditions)
    p.x.lim = (-0.55, 1.55)
    if not both.any() and not lone_y.size:
        p.note = "no data"
    return p


def paired_forest(labels: Sequence[str], a: tuple[Any, Any, Any], b: tuple[Any, Any, Any],
                  title: str, xaxis: Axis, *, label_a: str, label_b: str,
                  roles: tuple[str, str] = ("accent", "muted"),
                  markers: tuple[str, str] = ("d", "s"), ref: float | None = None,
                  ref_label: str | None = None) -> Panel:
    """Two estimates with intervals per row (``a``/``b`` = (est, lo, hi) arrays), drawn just
    above and below the row line, e.g. the primary analysis and a sensitivity variant."""
    p = Panel(title, xaxis, Axis(""), [], legend="auto")
    for (est, lo, hi), off, role, ms, lab in ((a, -0.17, roles[0], markers[0], label_a),
                                              (b, 0.17, roles[1], markers[1], label_b)):
        est, lo, hi = (np.asarray(x, float) for x in (est, lo, hi))
        for i in range(len(labels)):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]):
                p.series.append(Series("line", np.array([lo[i], hi[i]]),
                                       np.array([i + off, i + off]), role=role,
                                       style={"width": 1.2}))
        ok = np.isfinite(est)
        if ok.any():
            p.series.append(Series("scatter", est[ok], np.arange(len(labels))[ok] + off,
                                   role=role, style={"ms": ms, "size": 1.4}, label=lab))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.lim = (-0.6, len(labels) - 0.4) if labels else None
    p.y.reverse = True
    return p


def grouped_forest(labels: Sequence[str],
                   groups: Sequence[tuple[str, tuple[Any, Any, Any]]], title: str,
                   xaxis: Axis, *, ref: float | None = None, ref_label: str | None = None,
                   spread: float = 0.36, marker: str = "d",
                   styles: Sequence[tuple[str, str]] | None = None) -> Panel:
    """Several estimates with intervals per row, dodged within the row and coloured by
    group (``groups`` = ``[(legend label, (est, lo, hi))]``, each array one value per row):
    e.g. rows = analyses, groups = outcomes, so a row holds every outcome of one analysis.
    ``styles`` = ``[(role, marker)]`` per group overrides the categorical colour and the one
    marker (e.g. colour = task, marker = rule)."""
    p = Panel(title, xaxis, Axis(""), [], legend="auto")
    k = len(groups)
    offs = np.linspace(-spread, spread, k) if k > 1 else np.zeros(1)
    for g, ((lab, (est, lo, hi)), off) in enumerate(zip(groups, offs)):
        est, lo, hi = (np.asarray(x, float) for x in (est, lo, hi))
        role, mk = styles[g] if styles else (f"cat:{g % 6}", marker)
        for i in range(len(labels)):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]):
                p.series.append(Series("line", np.array([lo[i], hi[i]]),
                                       np.array([i + off, i + off]), role=role,
                                       style={"width": 1.0}))
        ok = np.isfinite(est)
        if ok.any():
            p.series.append(Series("scatter", est[ok], np.arange(len(labels))[ok] + off,
                                   role=role, style={"ms": mk, "size": 1.1}, label=lab))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.lim = (-0.6, len(labels) - 0.4) if labels else None
    p.y.reverse = True
    return p


def spec_curve(est: Sequence[float], lo: Sequence[float], hi: Sequence[float],
               specs: Sequence[dict[str, str]], title: str, yaxis: Axis, *,
               dims: Sequence[tuple[str, Sequence[str]]],
               highlight: dict[int, tuple[str, str]] | None = None, ref: float | None = 0.0,
               ref_label: str | None = None, xlabel: str = "Specification, ranked",
               key: str = "spec", matrix_title: str = "") -> tuple[Panel, Panel]:
    """Specification curve: every specification's estimate with its CI, sorted, over an
    indicator matrix of the choices that define it (one row per level of each choice in
    ``dims`` = ``[(choice label, [levels in order])]``; ``specs[i][choice label]`` is the
    level of specification ``i``). ``highlight`` = ``{i: (legend label, marker)}`` draws
    specification ``i`` in the accent colour (e.g. the pre-specified analysis). Returns the
    curve panel and the matrix panel; put them in one column, curve above the matrix."""
    e, lw, hg = (np.asarray(x, float) for x in (est, lo, hi))
    keep = [i for i in np.argsort(np.where(np.isfinite(e), e, np.inf), kind="stable")
            if np.isfinite(e[i])]
    n = len(keep)
    x = np.arange(1, n + 1, dtype=float)
    hl = highlight or {}
    curve = Panel(title, Axis(""), yaxis, [], legend="auto", sharex=key)
    for xi, i in zip(x, keep):
        if np.isfinite(lw[i]) and np.isfinite(hg[i]):
            curve.series.append(Series("line", np.array([xi, xi]), np.array([lw[i], hg[i]]),
                                       role="accent" if i in hl else "context",
                                       style={"width": 1.1 if i in hl else 0.9}))
    rest = [k for k, i in enumerate(keep) if i not in hl]
    if rest:
        curve.series.append(Series("scatter", x[rest], e[[keep[k] for k in rest]],
                                   role="context", style={"size": 1.0}))
    pos = {i: k for k, i in enumerate(keep)}
    for i, (lab, ms) in hl.items():              # in the caller's order (legend order)
        if i in pos:
            curve.series.append(Series("scatter", np.array([x[pos[i]]]), np.array([e[i]]),
                                       role="accent", style={"ms": ms, "size": 1.6}, label=lab))
    if ref is not None:
        curve.series.append(Series("hline", y=np.array([float(ref)]), role="reference",
                                   label=ref_label))
    # Levels no drawn specification uses are dropped (an empty matrix row reads as a choice
    # that was tried; e.g. the other system's motion sources in a dataset study).
    used = {(d, specs[i].get(d)) for i in keep for d, _ in dims}
    dims = [(d, [lev for lev in levels if (d, lev) in used]) for d, levels in dims]
    dims = [(d, levels) for d, levels in dims if levels]
    rows: list[tuple[str, str]] = [(d, lev) for d, levels in dims for lev in levels]
    mat = Panel(matrix_title, Axis(xlabel), Axis(""), [], legend="none", sharex=key)
    on_x, on_y, off_x, off_y = [], [], [], []
    for xi, i in zip(x, keep):
        for r, (d, lev) in enumerate(rows):
            (on_x if specs[i].get(d) == lev else off_x).append(xi)
            (on_y if specs[i].get(d) == lev else off_y).append(float(r))
    if off_x:
        mat.series.append(Series("scatter", np.array(off_x), np.array(off_y), role="faint",
                                 style={"size": 0.7}))
    if on_x:
        mat.series.append(Series("scatter", np.array(on_x), np.array(on_y), role="accent",
                                 style={"size": 0.9}))
    r0 = 0
    for _, levels in dims[:-1]:                 # light separators between the choices
        r0 += len(levels)
        mat.series.append(Series("hline", y=np.array([r0 - 0.5]), role="faint"))
    mat.y.ticks = [float(r) for r in range(len(rows))]
    mat.y.ticklabels = [f"{d}: {lev}" for d, lev in rows]
    mat.y.lim = (-0.6, len(rows) - 0.4)
    mat.y.reverse = True
    for pp in (curve, mat):
        pp.x.lim = (0.4, n + 0.6) if n else (0.0, 1.0)
    if not n:
        curve.note = mat.note = "no data"
    return curve, mat


# -- number printing (one rule for the tables) --------------------------------------
MINUS = "\u2212"


def decimals_for(v: float, d: int = 2) -> int:
    """Decimals that suit the size of ``v``: none from 100, at most one from 10, else ``d``."""
    a = abs(v) if np.isfinite(v) else 0.0
    return 0 if a >= 100 else (min(d, 1) if a >= 10 else d)


def num(v: float, d: int = 2, *, adaptive: bool = False) -> str:
    """A number for print: a real minus, never a negative zero, ``--`` for NaN; ``adaptive``
    picks the decimals from the size (:func:`decimals_for`)."""
    if v is None or not np.isfinite(v):
        return "--"
    if adaptive:
        d = decimals_for(v, d)
    s = f"{v:.{d}f}"
    if s.startswith("-") and not s.strip("-0."):
        s = s[1:]                                   # -0.00 -> 0.00
    return s.replace("-", MINUS)


def snum(v: float, d: int = 2) -> str:
    """:func:`num` with an explicit sign: ``+0.4``, ``−0.4``; a value that rounds to zero
    prints unsigned (``0.0``)."""
    s = num(v, d)
    return s if s == "--" or s.startswith(MINUS) or not s.strip("0.") else "+" + s


def ci(lo: float, hi: float, d: int = 2) -> str:
    """``[lo, hi]`` with :func:`num` (empty when either end is missing)."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""
    return f"[{num(lo, d)},\u00a0{num(hi, d)}]"


def est_ci(v: float, lo: float, hi: float, d: int = 2, *, adaptive: bool = True) -> str:
    """``v [lo, hi]``: the estimate's decimals (adaptive by default) for all three."""
    dd = decimals_for(v, d) if adaptive else d
    c = ci(lo, hi, dd)
    return num(v, dd) + (f" {c}" if c else "")


def pval(p: float) -> str:
    """``< 0.001`` or three decimals (``--`` for NaN)."""
    return "--" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def ptext(p: float, name: str = "p") -> str:
    """``p = 0.123`` or ``p < 0.001`` for running text."""
    v = pval(p)
    return f"{name} {v}" if v.startswith("<") else f"{name} = {v}"


# -- tables -------------------------------------------------------------------------------------
# Never shown: session folder names (sex, age, date), recording file names (initials, date),
# wall-clock dates. Dropped by column, and any session name left in a text cell is withheld.
PRIVATE_COLUMNS = ("dataset_take", "dataset_mvc_take", "delsys_start", "qtm_timestamp", "date",
                   "source_path", "path", "session", "when", "stream_start_us",
                   "recording_utc")
_FILE_RE = r"\b[A-Za-z]{1,4}_\d{2}_\d{2}_\d{2}[A-Za-z0-9_]*(?:\.csv|\.mat|\.qtm)?"


def redact(df: pd.DataFrame) -> pd.DataFrame:
    """Private columns dropped; session folder names and dated recording file names in text
    cells replaced (results.release.SESSION_RE and the Delsys/QTM file pattern)."""
    from biomech.results.release import SESSION_RE

    def sub(v: str) -> str:
        return re.sub(_FILE_RE, "file (withheld)", SESSION_RE.sub("session (withheld)", v))

    d = df.drop(columns=[c for c in PRIVATE_COLUMNS if c in df.columns])
    for c in d.columns:
        if d[c].dtype == object:       # text cells only: booleans stay booleans (yes / no)
            d[c] = [sub(v) if isinstance(v, str) else v for v in d[c]]
    return d


def pseudonymize_cells(df: pd.DataFrame, ctx: Any) -> pd.DataFrame:
    """Study ids inside text cells of columns other than ``pid`` (``left_out: s02``, test
    names with an id) replaced by their pseudonyms (``P3``; ``excluded`` outside the
    cohort): no study id reaches an artefact through a cell either."""
    from biomech.results.release import PID_RE

    names = ctx.pseudonyms()
    def sub(m):
        return f"P{names[m.group(0)]}" if m.group(0) in names else "excluded"
    d = df.copy()
    for c in d.columns:
        if c != "pid" and d[c].dtype == object:
            d[c] = [PID_RE.sub(sub, v) if isinstance(v, str) else v for v in d[c]]
    return d


def reader_cell(column: str, value: Any) -> Any:
    """A table cell in reader form: booleans as yes/no, categorical values through the
    vocabulary (``etc/vocab.yaml``: task, muscle, estimator ... names), numbers unchanged."""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return V.cell(column, value)


def frame_table(df: pd.DataFrame, table_id: str, caption: str, *,
                columns: Sequence[str] | None = None, max_rows: int = 400,
                ctx: Any = None, width: str = "text") -> TableSpec:
    """A stored table as a TableSpec: catalog headers/units/rounding; ``pid`` shown as
    pseudonym (``ctx`` required for a table with participants; units outside the cohort
    read "excluded" -- never the study id) and the ``unit`` bookkeeping column dropped."""
    d = df.drop(columns=[c for c in ("unit",) if c in df.columns])
    if "pid" in d.columns:
        if ctx is None:
            raise ValueError(f"{table_id}: a table with participants needs the data context "
                             "(pseudonyms); study ids are never shown")
        d = d.copy()
        names = ctx.pseudonyms()
        d["pid"] = [f"P{names[p]}" if p in names else "excluded" for p in d["pid"]]
    cols = [c for c in (columns or list(d.columns)) if c in d.columns]
    out = []
    for c in cols:
        e = catalog.get(c)
        numeric = pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])
        out.append(Column(c, e.label, e.unit, e.decimals, "S" if numeric else "l"))
    rows = [{c: reader_cell(c, v) for c, v in r.items()}
            for r in d[cols].head(max_rows).to_dict("records")]
    return TableSpec(table_id, caption, out, rows, width=width)


# -- recorded phases on a timeline (R5b) ----------------------------------------------------
def phase_spans(t: np.ndarray, labels: Sequence[str]) -> list[tuple[str, float, float]]:
    """Runs of equal non-empty phase labels: ``(name, t0, t1)`` in the units of ``t``."""
    t = np.asarray(t, float)
    lab = np.asarray(labels, dtype=object)
    out: list[tuple[str, float, float]] = []
    if len(t) == 0:
        return out
    cut = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]):
        if lab[a]:
            out.append((str(lab[a]), float(t[a]), float(t[b - 1])))
    return out


def phase_lane(spans: Sequence[tuple[str, float, float]], title: str = "", *,
               extra: Sequence[tuple[str, float, float]] = (), short: bool = True) -> Panel:
    """A thin lane panel (share x with the data panels below): one band per phase in its
    phase colour (tasks, rest) or neutral (holds), a text label at each band centre.
    Labels of neighbouring short bands alternate between two rows. ``extra`` spans (e.g.
    the sync windows) are drawn as outlined neutral bands in the upper row."""
    p = Panel(title=title, x=Axis("", lim=None), y=Axis("", lim=(0.0, 1.0), ticks=[]),
              height=0.27, legend=None, sharex="time")
    by_role: dict[str, list[tuple[float, float]]] = {}
    for name, a, b in spans:
        by_role.setdefault(V.phase_role(name), []).append((a, b))
    for role, ab in by_role.items():
        a, b = np.array(ab).T
        p.series.append(Series("hspan", x=a, y2=b, role=role,
                               style={"alpha": 0.55 if role != "faint" else 0.35}))
    if extra:
        a, b = np.array([(s[1], s[2]) for s in extra]).T
        p.series.append(Series("hspan", x=a, y2=b, role="context", style={"alpha": 0.25}))
    xs, ys, ts = [], [], []
    width = (max(s[2] for s in spans) - min(s[1] for s in spans)) if spans else 1.0
    # Each label goes to the first of the two rows where it does not overlap the previous
    # label of that row, by its estimated extent (~1.1 % of the lane per character); the
    # band-width test alone let wide labels of neighbouring bands collide.
    char_w = 0.011 * width
    right_end = [-np.inf, -np.inf]
    # phase labels prefer the lower row, the extra (sync) labels the upper one; all in order
    labels = sorted([(0.5 * (a + b), V.phase_label(n, short=short), 0) for n, a, b in spans]
                    + [(0.5 * (a + b), V.phase_label(n, short=True), 1) for n, a, b in extra])
    for c, t, pref in labels:
        half = 0.5 * char_w * (len(t) + 1)
        free = [r for r in (pref, 1 - pref) if c - half >= right_end[r]]
        row = free[0] if free else int(np.argmin(right_end))
        right_end[row] = c + half
        xs.append(c), ys.append(0.06 if row == 0 else 0.56)
        ts.append(t)
    if ts:
        p.series.append(Series("text", x=np.array(xs), y=np.array(ys), text=ts,
                               style={"anchor": "above", "color": "ink"}))
    return p


def add_phase_bands(p: Panel, spans: Sequence[tuple[str, float, float]], *,
                    alpha: float = 0.10, holds: bool = False) -> None:
    """Faint phase bands behind a data panel (same colours as :func:`phase_lane`); the
    neutral holds only when ``holds``."""
    by_role: dict[str, list[tuple[float, float]]] = {}
    for name, a, b in spans:
        role = V.phase_role(name)
        if role == "faint" and not holds:
            continue
        by_role.setdefault(role, []).append((a, b))
    for role, ab in by_role.items():
        a, b = np.array(ab).T
        p.series.insert(0, Series("hspan", x=a, y2=b, role=role, style={"alpha": alpha}))



# -- uncertainty (E: bootstrap bands, CIs of the median, n-aware heights) -------------
# Views draw uncertainty they can derive deterministically from stored per-participant (or
# per-cycle) values: a fixed seed, the stats stages' own BCa routine for summary values, a
# pointwise percentile bootstrap for curves. Nothing here changes a stored result.
N_BOOT_CURVE = 2000          # resamples of a pointwise curve band
SD_BAND_ALPHA = 0.4          # the spread band (SD/IQR) drawn clearly lighter than the CI band


def boot_curve_ci(subjects: Any, *, centre: str = "mean", n_boot: int = N_BOOT_CURVE,
                  seed: int = 0, min_n: int = 3, level: float = 95.0
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise percentile bootstrap CI of the centre (``mean``/``median``) of the rows of
    ``subjects`` (units x samples, e.g. participants x cycle points): the units are resampled
    with replacement (``seed`` fixed, so a view draws the same band every time). Samples with
    fewer than ``min_n`` finite units are NaN."""
    S = np.atleast_2d(np.asarray(subjects, float))
    n_pts = S.shape[1] if S.ndim == 2 else 0
    nan = np.full(n_pts, np.nan)
    if S.size == 0:
        return nan, nan.copy()
    S = S[np.isfinite(S).any(axis=1)]
    m = len(S)
    if m < min_n:
        return nan, nan.copy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, (n_boot, m))
    f = np.nanmedian if centre == "median" else np.nanmean
    out = np.empty((n_boot, n_pts))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        step = max(1, int(4e6 // max(m * n_pts, 1)))
        for a in range(0, n_boot, step):
            out[a:a + step] = f(S[idx[a:a + step]], axis=1)
        a_lo = (100.0 - level) / 2.0
        lo, hi = np.nanpercentile(out, [a_lo, 100.0 - a_lo], axis=0)
    thin = np.isfinite(S).sum(axis=0) < min_n
    lo[thin], hi[thin] = np.nan, np.nan
    return lo, hi


def add_ci_band(p: Panel, x: Any, subjects: Any, *, centre: str = "mean", role: str = "band",
                seed: int = 0, where: Any = None, label: str | None = None,
                n_boot: int = N_BOOT_CURVE) -> bool:
    """Append the bootstrap CI band of the centre of ``subjects`` (rows) to ``p``; ``where``
    (bool per sample, e.g. where the drawn centre is finite) limits it. True if drawn."""
    lo, hi = boot_curve_ci(subjects, centre=centre, seed=seed, n_boot=n_boot)
    if where is not None:
        w = np.asarray(where, bool)
        lo, hi = np.where(w, lo, np.nan), np.where(w, hi, np.nan)
    if not (np.isfinite(lo) & np.isfinite(hi)).any():
        return False
    p.series.append(Series("band", np.asarray(x, float), lo, y2=hi, role=role, label=label))
    return True


def add_curves_ci(p: Panel, x: np.ndarray, subjects: np.ndarray, *, centre: str = "mean",
                  label: str | None = None, role: str = "accent", band_role: str | None = None,
                  show_subjects: bool = True, seed: int = 0) -> Panel:
    """:func:`add_curves` with two bands: the spread (± SD, or the IQR for a median) drawn
    light and the bootstrap 95 % CI of the centre over the rows drawn darker (the
    rows are the resampled units, see :func:`boot_curve_ci`)."""
    S = np.atleast_2d(np.asarray(subjects, float))
    n0 = len(p.series)
    add_curves(p, x, S, centre=centre, label=label, role=role, show_subjects=show_subjects)
    br = band_role or role
    for s in p.series[n0:]:
        if s.kind == "band":
            s.role = br
            s.style["alpha"] = SD_BAND_ALPHA
    line = p.series[-1] if len(p.series) > n0 else None
    if line is not None and line.kind == "line" and add_ci_band(
            p, x, S, centre=centre, role=br, seed=seed, where=np.isfinite(line.y)):
        p.series.append(p.series.pop(-2))           # the centre line stays on top
    return p


def curve_ci_caption(centre: str = "mean", units: str = "participants") -> str:
    """Caption fragment for :func:`add_curves_ci`."""
    spread = "IQR" if centre == "median" else "± SD"
    return (f"light band: {spread}; dark band: 95 % bootstrap CI of the {centre} "
            f"({N_BOOT_CURVE} resamples of {units}, pointwise percentile)")


def median_ci(values: Any, seed: int = 0) -> tuple[float, float]:
    """95 % BCa bootstrap CI of the median, by the stats stages' own routine
    (:func:`biomech.stats.core.describe`, same seed -> the same interval as a stored one);
    NaN for fewer than 4 values or no spread."""
    from biomech.stats.core import describe

    d = describe(_finite(values), stat="median", seed=seed)
    return float(d["ci_lo"]), float(d["ci_hi"])


def median_cis(values: Sequence[Any], seed: int = 0) -> list[tuple[float, float]]:
    return [median_ci(v, seed) for v in values]


MEDIAN_CI_CAPTION = "line: 95 % BCa bootstrap CI of the median"


def strip_ci(labels: Sequence[str], values: Sequence[Any], title: str, xaxis: Axis, *,
             seed: int = 0, **kw: Any) -> Panel:
    """:func:`strip` with every row's median CI (:func:`median_ci`) as the interval: all
    units as dots, the median as a diamond, its CI as a line."""
    kw.setdefault("intervals", median_cis(values, seed))
    return strip(labels, values, title, xaxis, **kw)


def strip_by_ci(df: pd.DataFrame, value: str, by: str, title: str, *, seed: int = 0,
                order: Sequence[str] | None = None, **kw: Any) -> Panel:
    """:func:`strip_by` with the median CI per row (see :func:`strip_ci`)."""
    if not df.empty and value in df and "intervals" not in kw:
        cats = [c for c in (order or sorted(df[by].dropna().unique())) if c in set(df[by])]
        kw["intervals"] = median_cis([df.loc[df[by] == c, value] for c in cats], seed)
    return strip_by(df, value, by, title, order=order, **kw)


def paired_strip(labels: Sequence[str], a: Sequence[Any], b: Sequence[Any], title: str,
                 xaxis: Axis, *, label_a: str, label_b: str,
                 roles: tuple[str, str] = ("cat:0", "cat:1"), seed: int = 0,
                 ref: float | None = None, ref_label: str | None = None) -> Panel:
    """Two conditions per row, each as a strip just above / below the row line: every unit
    as a small dot in the condition's colour, the median as a diamond with its 95 % BCa CI
    (:func:`median_ci`). Replaces a dumbbell of two bare medians."""
    p = Panel(title, xaxis, Axis(""), [], legend="auto")
    for vals, off, role, lab in ((a, -0.18, roles[0], label_a), (b, 0.18, roles[1], label_b)):
        xs, ys, mx, my = [], [], [], []
        for i in range(len(labels)):
            v = _finite(vals[i]) if i < len(vals) else np.empty(0)
            if not v.size:
                continue
            jit = np.linspace(-0.09, 0.09, v.size) if v.size > 1 else np.zeros(1)
            xs.append(v)
            ys.append(np.full(v.size, i + off) + jit)
            lo, hi = median_ci(v, seed)
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("line", np.array([lo, hi]), np.array([i + off] * 2),
                                       role=role, style={"width": 1.4}))
            mx.append(float(np.median(v)))
            my.append(i + off)
        if xs:
            p.series.append(Series("scatter", np.concatenate(xs), np.concatenate(ys), role=role,
                                   style={"size": 0.6, "alpha": 0.5}))
            p.series.append(Series("scatter", np.array(mx), np.array(my), role=role,
                                   style={"ms": "d", "size": 1.4}, label=lab))
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.lim = (-0.6, len(labels) - 0.4) if len(labels) else None
    p.y.reverse = True
    if not any(s.kind == "scatter" for s in p.series):
        p.note = "no data"
    return p


def rows_height(n: int, *, per: float = 0.17, extra: float = 0.6, lo: float = 1.3) -> float:
    """Axis height (inches, :attr:`FigureSpec.height_in`) for a panel with ``n`` category rows
    (participants, muscles, ...): grows with n so tick labels never collide."""
    return max(lo, per * max(int(n), 1) + extra)


def cfg(ctx: Any, stage: str, *keys: str, default: Any = None) -> Any:
    """A parameter the stage ran with (``config.stages.<stage>.<keys...>``), ``default`` when
    the context has no config or the key is missing: captions and reference lines state the
    configured value, never a typed copy (J1.2)."""
    try:
        d: Any = ctx.config.params(stage)
    except (AttributeError, TypeError):
        return default
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def latency_budget_ms(ctx: Any) -> float:
    """The latency budget of the online decoder (``ml_online.latency.budget_ms``)."""
    return float(cfg(ctx, "ml_online", "latency", "budget_ms", default=150.0))


def seed_of(ctx: Any) -> int:
    """The study seed (``StudyConfig.seed``, the one the stats stages use); 0 without one."""
    try:
        return int(getattr(ctx.config, "seed", 0) or 0)
    except (TypeError, ValueError):
        return 0
