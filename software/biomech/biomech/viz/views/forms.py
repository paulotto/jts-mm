# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Depictions of the same numbers: one drawer per form, chosen per item.

A view whose data are "one value per unit (participant, cycle) in each of several rows"
declares the forms it supports as a picker and draws with :func:`summary`::

    @register("x", ..., choices={"form": forms.SUMMARY}, form="strip")
    def x(req):
        panel, marks = forms.summary(req.options["form"], labels, values, title, axis)

The picker reaches the viewer (a "Depiction" dropdown), the export id (``--form-box``), the
paper set (``{id: x, form: box}``) and the gallery (``biomech export --gallery``, every form
side by side). Every drawer returns the panel and the meaning of its marks (for the
caption, :class:`biomech.viz.caption.CaptionSpec`), so a change of form changes the caption
with it. All drawers use only the existing series kinds (no renderer change) and the stats
stages' own interval routine (:func:`blocks.median_ci`, same seed), so a depiction never
computes a different number than the results.

Forms (``etc/vocab.yaml`` domain ``form`` names them for readers):

* ``strip``: every unit as a dot, the median as a diamond with its 95 % BCa CI;
* ``box``: interquartile box, median, whiskers to the most extreme value within 1.5 IQR,
  values beyond as dots;
* ``violin``: mirrored kernel density (Scott bandwidth), median and IQR line;
* ``raincloud``: half density above the row, box line and every unit as a dot below;
* ``ecdf``: one empirical cumulative distribution per row;
* ``bar``: the median as a bar with its 95 % BCa CI, every unit as a dot.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from biomech.viz.spec import Axis, Panel, Series
from biomech.viz.views import blocks as B

SUMMARY = ("strip", "box", "violin", "raincloud", "ecdf", "bar")
Marks = list[tuple[str, str]]


def _rows(labels: Sequence[str], values: Sequence[Any]) -> list[np.ndarray]:
    return [B._finite(values[i]) if i < len(values) else np.empty(0)
            for i in range(len(labels))]


def _yaxis(p: Panel, labels: Sequence[str]) -> Panel:
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.lim = (-0.6, len(labels) - 0.4) if len(labels) else None
    p.y.reverse = True
    return p


def _ref(p: Panel, ref: float | None, ref_label: str | None) -> None:
    if ref is not None:
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))


def _dots(p: Panel, v: np.ndarray, y: float, spread: float, role: str = "context",
          size: float = 0.7) -> None:
    if v.size:
        jit = np.linspace(-spread, spread, v.size) if v.size > 1 else np.zeros(1)
        p.series.append(Series("scatter", v, np.full(v.size, y) + jit, role=role,
                               style={"size": size}))


def _kde(v: np.ndarray, n: int = 64, log: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian KDE with Scott's bandwidth over the data range +- 1 bandwidth, peak = 1. On
    a log axis the density is estimated on log10 of the (positive) values, so it never
    reaches zero or below."""
    if log:
        v = np.log10(v[v > 0])
    if v.size < 2 or np.ptp(v) == 0:
        return np.empty(0), np.empty(0)
    bw = 1.06 * np.std(v, ddof=1) * v.size ** (-0.2)
    x = np.linspace(v.min() - bw, v.max() + bw, n)
    d = np.exp(-0.5 * ((x[:, None] - v[None, :]) / bw) ** 2).sum(1)
    return (10.0 ** x if log else x), d / d.max()


def _box(p: Panel, v: np.ndarray, y: float, h: float, role: str = "accent") -> None:
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo = v[v >= q1 - 1.5 * iqr].min()
    hi = v[v <= q3 + 1.5 * iqr].max()
    p.series.append(Series("fill", np.array([q1, q3, q3, q1]),
                           np.array([y - h, y - h, y + h, y + h]), role=role,
                           style={"alpha": 0.18}))
    p.series.append(Series("line", np.array([med, med]), np.array([y - h, y + h]), role=role,
                           style={"width": 1.6}))
    for a, b in ((lo, q1), (q3, hi)):
        p.series.append(Series("line", np.array([a, b]), np.array([y, y]), role=role))
    out = v[(v < lo) | (v > hi)]
    if out.size:
        p.series.append(Series("scatter", out, np.full(out.size, y), role="context",
                               style={"size": 0.7}))


def summary(form: str, labels: Sequence[str], values: Sequence[Any], title: str,
            xaxis: Axis, *, seed: int = 0, ref: float | None = None,
            ref_label: str | None = None, unit: str = "participants") -> tuple[Panel, Marks]:
    """One row per label, the units of each row drawn as ``form``; returns the panel and the
    meaning of its marks for the caption."""
    rows = _rows(labels, values)
    if form == "strip":
        p = B.strip_ci(labels, rows, title, xaxis, seed=seed, ref=ref, ref_label=ref_label)
        marks = [("dots", unit), ("diamond", "median"), ("line", "95 % BCa CI of the median")]
    elif form == "ecdf":
        p = B.ecdf({lab: v for lab, v in zip(labels, rows)}, title, xaxis, ref=ref,
                   ref_label=ref_label)
        p.legend = "below"
        marks = [("steps", f"share of {unit} at or below a value, one curve per row"),
                 ("dots", unit)]
        return p, marks
    else:
        p = Panel(title, xaxis, Axis(""), [], legend="none")
        for i, v in enumerate(rows):
            if not v.size:
                continue
            if form == "box":
                _box(p, v, float(i), 0.22)
            elif form == "violin":
                x, d = _kde(v, log=xaxis.log)
                if x.size:
                    p.series.append(Series("fill", np.r_[x, x[::-1]],
                                           np.r_[i - 0.32 * d, (i + 0.32 * d)[::-1]],
                                           role="accent", style={"alpha": 0.18}))
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                p.series.append(Series("line", np.array([q1, q3]), np.array([i, i], float),
                                       role="accent", style={"width": 1.6}))
                p.series.append(Series("scatter", np.array([med]), np.array([float(i)]),
                                       role="accent", style={"ms": "d", "size": 1.4}))
            elif form == "raincloud":
                x, d = _kde(v, log=xaxis.log)
                if x.size:
                    p.series.append(Series("fill", np.r_[x, x[::-1]],
                                           np.r_[np.full(x.size, i - 0.04),
                                                 (i - 0.04 - 0.36 * d)[::-1]],
                                           role="accent", style={"alpha": 0.18}))
                _box(p, v, float(i) + 0.08, 0.05)
                _dots(p, v, float(i) + 0.26, 0.08)
            elif form == "bar":
                if xaxis.log:
                    raise ValueError("form 'bar' needs a linear axis (a bar starts at 0)")
                med = float(np.median(v))
                lo, hi = B.median_ci(v, seed)
                p.series.append(Series("fill", np.array([0, med, med, 0]),
                                       np.array([i - 0.3, i - 0.3, i + 0.3, i + 0.3]),
                                       role="accent", style={"alpha": 0.25}))
                if np.isfinite(lo) and np.isfinite(hi):
                    p.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                           role="accent", style={"width": 1.4}))
                _dots(p, v, float(i), 0.15)
            else:
                raise ValueError(f"unknown form {form!r}; known: {SUMMARY}")
        _ref(p, ref, ref_label)
        _yaxis(p, labels)
        marks = {"box": [("box", "interquartile range"), ("vertical line", "median"),
                         ("whiskers", "most extreme values within 1.5 IQR"),
                         ("dots", f"{unit} beyond the whiskers")],
                 "violin": [("shape", f"kernel density of the {unit}"),
                            ("line", "interquartile range"), ("diamond", "median")],
                 "raincloud": [("shape", f"kernel density of the {unit}"),
                               ("box", "interquartile range with median and whiskers"),
                               ("dots", unit)],
                 "bar": [("bar", "median"), ("line", "95 % BCa CI of the median"),
                         ("dots", unit)]}[form]
    if not any(s.kind in ("scatter", "fill", "line") for s in p.series):
        p.note = "no data"
    return p, marks
