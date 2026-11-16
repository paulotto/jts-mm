# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Backend-neutral figure and table specifications (an08 §4).

A spec holds materialized data plus *meaning* (axes with units, series roles,
captions); renderers decide appearance. The same spec renders to matplotlib
(quick look, PNG/PDF), TikZ/pgfplots (paper) and Plotly (viewer), and the
geometry of all three comes from one layout engine (:mod:`biomech.viz.layout`),
so what is exported from the viewer is exactly what the paper shows.

Series kinds
------------
2-D: ``line``, ``scatter``, ``band`` (y..y2), ``bar`` (vertical) / ``barh``
(horizontal), ``errorbar`` (ylo..y2), ``hline``/``vline`` (reference lines,
labelled with ``label`` directly at the line), ``hspan`` (x..y2 shaded),
``heatmap`` (``z`` matrix), ``text``, ``colorline`` (a path coloured by the
per-point scalar ``c``), ``arrow`` (tails ``x, y``, vectors ``d``), ``fill``
(closed polygon), ``image`` (``img`` placed at ``extent``; a 2-D float ``img`` with
``style["cmap"]`` = "sequential"/"diverging" is a colour-mapped scalar map over
``style["lim"]`` -- NaN cells transparent, ``style["mask"]`` (bool, same shape) marks cells
drawn in the muted grey), ``sigbar`` (SPM
supra-threshold clusters ``x``..``y2`` as bars stacked at the top of the panel;
``style["row"]`` picks the row, the role its colour, ``label`` names the test
in the legend -- the layout reserves headroom so the bars never cover data).
3-D (``Panel.projection == "3d"``): ``line3d``, ``scatter3d``, ``arrow3d``
(``x, y, z`` plus ``d``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

SeriesKind = Literal["line", "scatter", "band", "bar", "barh", "errorbar", "hline", "vline",
                     "heatmap", "hspan", "text", "colorline", "arrow", "fill", "image",
                     "sigbar", "line3d", "scatter3d", "arrow3d", "surface3d"]
# surface3d (J2.2): x (nx,), y (ny,) grid vectors, z (ny, nx) heights; colour = z, or
# c (ny, nx) when given; style["ramp"] "sequential" (default) or "diverging"
KINDS_3D = ("line3d", "scatter3d", "arrow3d", "surface3d")
# Kinds that never get a legend entry (reference marks are labelled in place).
NO_LEGEND = ("hline", "vline", "hspan", "heatmap", "text", "image", "surface3d")  # sigbar: legend ok


@dataclass
class Series:
    kind: SeriesKind
    x: np.ndarray | None = None
    y: np.ndarray | None = None
    y2: np.ndarray | None = None          # band upper / errorbar upper / hspan end
    ylo: np.ndarray | None = None         # errorbar lower
    z: np.ndarray | None = None           # heatmap matrix (rows = y) or 3-D z coordinate
    c: np.ndarray | None = None           # per-point scalar of a colorline / coloured scatter
    d: np.ndarray | None = None           # arrow vectors (N, 2) or (N, 3)
    img: np.ndarray | None = None         # image (H, W) or (H, W, 3), uint8 or float 0..1
    extent: tuple[float, float, float, float] | None = None   # image x0, x1, y0, y1
    label: str | None = None              # legend entry (None = no entry)
    role: str = "primary"                 # style role (see style.ROLES)
    text: list[str] | None = None         # per-point labels / heatmap annotations
    style: dict[str, Any] = field(default_factory=dict)   # backend-neutral tweaks:
    # width (line width factor), size (marker size factor), alpha, ms (marker),
    # halo (scatter: width in pt of the surface-coloured outline; default 0.3-0.4),
    # anchor ("left"/"right"/"above"/"below" for text and reference labels)


@dataclass
class Axis:
    label: str = ""
    unit: str = ""                         # siunitx unit, e.g. "\\milli\\metre"
    lim: tuple[float, float] | None = None
    log: bool = False
    ticks: list[float] | None = None
    ticklabels: list[str] | None = None
    reverse: bool = False

    def text(self) -> str:
        """Plain-text label with unit, for non-LaTeX backends."""
        from biomech.viz.style import plain_unit

        u = plain_unit(self.unit)
        return f"{self.label} ({u})" if u else self.label


@dataclass
class Colorbar:
    label: str = ""
    unit: str = ""
    lim: tuple[float, float] | None = None
    ramp: Literal["sequential", "diverging"] = "sequential"
    ticks: list[float] | None = None       # explicit colorbar ticks (default: automatic)

    def text(self) -> str:
        from biomech.viz.style import plain_unit

        u = plain_unit(self.unit)
        return f"{self.label} ({u})" if u else self.label


# Legend policy: "auto" (none for a single labelled series -- the title names it;
# below the panel for two or more), "below", "none", or "inside:<pgf legend pos>"
# such as "inside:north east" (only when the caller knows that corner is empty).
LegendPolicy = str


def _legend_policy(v: str | None) -> str:
    if v is None or v == "none":
        return "none"
    if v in ("auto", "below") or v.startswith("inside:"):
        return v
    # Legacy pgf corner names ("north east", ...): a corner is only known to be empty
    # when the caller says so explicitly ("inside:..."), so they become "auto".
    return "auto"


_TAG_RE = re.compile(r"^\s*\(([A-Za-z])\)\s*")


def split_tag(title: str) -> tuple[str, str]:
    """``"(a) Opening angle"`` -> ``("a", "Opening angle")``; titles without a prefix keep it
    empty."""
    m = _TAG_RE.match(title or "")
    return (m.group(1).lower(), title[m.end():]) if m else ("", title or "")


@dataclass
class Panel:
    title: str = ""
    x: Axis = field(default_factory=Axis)
    y: Axis = field(default_factory=Axis)
    series: list[Series] = field(default_factory=list)
    legend: LegendPolicy | None = "auto"
    equal: bool = False                    # equal data aspect (mm-for-mm plane views)
    projection: Literal["2d", "3d"] = "2d"
    z: Axis = field(default_factory=Axis)  # 3-D only
    view: tuple[float, float] = (-55.0, 22.0)   # 3-D azimuth, elevation (deg)
    colorbar: Colorbar | None = None
    note: str = ""                         # small text in the panel's top-left corner
    sharex: str | None = None              # panels with the same key share the x range
    sharey: str | None = None              # ... and the y range (data and limits; 2026-11-05)
    # Axis labels a small-multiples grid removed ({"x": (label, unit), "y": ...}); the viewer
    # restores them when it reflows the panels into one column (narrow screens).
    hidden_labels: dict[str, tuple[str, str]] = field(default_factory=dict)
    # Panel letter ("a", "b", ...), drawn bold in a fixed corner at the top left of the panel
    # cell, outside the axis. Empty: derived from a "(a) ..." title prefix (:meth:`heading`),
    # so views keep writing "(a) Title" and every renderer shows the tag and the short title.
    tag: str = ""
    # Axis height relative to ``FigureSpec.height_in`` (2-D, non-equal panels): e.g. a thin
    # phase lane above the data panels (R5b).
    height: float = 1.0

    def __post_init__(self) -> None:
        self.legend = _legend_policy(self.legend)

    def heading(self) -> tuple[str, str]:
        """(tag, title) as drawn: the explicit tag, else the one in the title prefix."""
        tag, title = split_tag(self.title)
        return (self.tag or tag), title

    def legend_entries(self) -> list[Series]:
        return [s for s in self.series if s.label and s.kind not in NO_LEGEND]

    def legend_mode(self) -> str:
        """The effective policy: ``none``, ``below`` or ``inside:<pos>``."""
        n = len(self.legend_entries())
        if n == 0 or self.legend == "none":
            return "none"
        if self.legend == "auto":
            return "below" if n >= 2 else "none"
        return self.legend


@dataclass
class FigureSpec:
    id: str
    title: str
    panels: list[Panel]
    rows: int = 1
    cols: int = 1
    width: Literal["column", "text"] = "column"
    height_in: float = 2.2                 # per row: the axis box plus its title (inches)
    caption: str = ""
    sources: list[str] = field(default_factory=list)   # "stage:table" provenance
    perspective: str = ""                  # e.g. "cohort, cycle-normalized, median/IQR"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Column:
    key: str
    header: str                            # LaTeX header text
    unit: str = ""
    decimals: int = 2
    align: str = "S"                       # S (siunitx), l, c, r, X (wrapping text)
    group: str | None = None               # spanning header (cmidrule)


@dataclass
class TableSpec:
    id: str
    caption: str
    columns: list[Column]
    # Layout rows: {"_rule": True} draws a midrule; "_head": True renders the first column's
    # text as an italic heading line across the table (pgf; other renderers show the row).
    rows: list[dict[str, Any]]
    notes: str = ""
    sources: list[str] = field(default_factory=list)
    width: Literal["column", "text"] = "column"
    meta: dict[str, Any] = field(default_factory=dict)   # e.g. the structured caption
