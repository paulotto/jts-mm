# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One layout engine for every renderer: overlap-free by construction.

Given a :class:`FigureSpec` and a venue (``style.VENUES``: column / text width and font
sizes; default the active venue, IEEE) or an explicit width, :func:`compute` decides, in
inches from the figure's top-left corner,

* every panel's axis box, title band, bold panel tag (top-left corner of the panel cell,
  outside the axis; from ``Panel.tag`` or an "(a) ..." title prefix), tick-label and axis-label bands, the
  legend box below it (sized and wrapped into columns to fit the axis width)
  and the colorbar strip;
* every axis' limits, tick positions and tick *labels* (plain numbers, also
  on log axes: 0.1, 0.3, 1, 3, 10 -- never 2x10^0);
* equal-aspect (mm-for-mm) panels; with ``spec.meta["equal_scale"] = "shared"``
  all of them share one scale, so sagittal/frontal/horizontal projections line up.

The pgf, matplotlib and Plotly renderers place their axes at these boxes and
use these ticks, so the three outputs share geometry, and nothing is left for
a backend to auto-place on top of the data. Text sizes are estimated from a
per-character width table (a little generous); the layout tests measure real
text extents with matplotlib and real TeX boxes with ``standalone`` to keep
the estimate honest.
"""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field

import numpy as np

from biomech.viz import style
from biomech.viz.spec import KINDS_3D, Axis, FigureSpec, Panel

# Average glyph widths in em (sans and CM roman are close at these sizes).
_NARROW = set("iljtf.,:;'|!()[]{} ")
_WIDE = set("mwMW%@")
_UPPER = set("ABCDEFGHIJKLMNOPQRSTUVXYZ")
_OPERATOR = set("−+=±×<>≤≥")     # math-width signs (the minus is 0.78 em in CM, not a hyphen)
PAD = 0.04          # in, generic gap
TICK_LEN = 0.035    # in


CBAR_SHIFT_IN = 0.3 / 2.54          # pgfplots "colorbar right" offset (0.3 cm)

def text_width_in(s: str, pt: float) -> float:
    """Estimated rendered width of plain text ``s`` at ``pt`` points (inches)."""
    s = plain_text(s)
    em = 0.0
    for ch in s:
        if ch in _NARROW:
            em += 0.30
        elif ch in _WIDE:
            em += 0.85
        elif ch in _UPPER:
            em += 0.68
        elif ch in _OPERATOR:
            em += 0.78
        elif ch.isdigit():
            em += 0.52
        else:
            em += 0.52
    return 1.06 * em * pt / 72.0


def line_height_in(pt: float) -> float:
    return 1.25 * pt / 72.0


def plain_text(s: str) -> str:
    """Strip the LaTeX we allow in labels ($...$, \\pm, \\leq, \\si{...})."""
    s = re.sub(r"\\si\{([^}]*)\}", lambda m: style.plain_unit(m.group(1)), s)
    for a, b in (("\\pm", "±"), ("\\leq", "≤"), ("\\geq", "≥"), ("\\times", "×"),
                 ("\\theta", "θ"), ("\\omega", "ω"), ("\\alpha", "α"), ("\\%", "%"),
                 ("\\,", " "), ("\\_", "_"), ("^\\circ", "°"), ("\\mu", "µ")):
        s = s.replace(a, b)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")
    return style.unicode_minus(s)


# -- ticks ---------------------------------------------------------------------------------
def _nice_step(span: float, target: int) -> float:
    raw = span / max(target, 1)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def _decimals(step: float) -> int:
    """Decimals needed to print every multiple of ``step`` exactly (2.5 -> 1, 0.25 -> 2)."""
    frac = f"{abs(step):.6f}".rstrip("0").split(".")[1] if "." in f"{abs(step):.6f}" else ""
    return min(len(frac), 4)


def fmt_tick(v: float, dec: int) -> str:
    s = f"{v:.{dec}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−")


def frameless(p: Panel) -> bool:
    """A 2-D panel whose ticks are explicitly empty on both axes (a picture, a polar clock):
    drawn without axis lines and grid, which would only frame it (hardware renders,
    the activation clock)."""
    return (p.projection != "3d" and p.x.ticks is not None and p.y.ticks is not None
            and not p.x.ticks and not p.y.ticks)


def _clear_corner(p: Panel, ax: "AxisLayout", ay: "AxisLayout") -> None:
    """Blank the y tick label at the bottom-left corner when an x tick label also sits there:
    the two collide at print size ("−20" over "−600"). Numeric axes only."""
    if p.x.ticklabels is not None or p.y.ticklabels is not None or ax.rotate \
            or not ax.ticks or not ay.ticks:
        return
    xs, ys = ax.lim[1] - ax.lim[0], ay.lim[1] - ay.lim[0]
    if not (xs > 0 and ys > 0):
        return
    x0 = ax.lim[1] if p.x.reverse else ax.lim[0]
    yb = ay.lim[1] if p.y.reverse else ay.lim[0]
    jx = min(range(len(ax.ticks)), key=lambda i: abs(ax.ticks[i] - x0))
    jy = min(range(len(ay.ticks)), key=lambda i: abs(ay.ticks[i] - yb))
    if abs(ax.ticks[jx] - x0) <= 0.02 * xs and abs(ay.ticks[jy] - yb) <= 0.02 * ys:
        ay.labels = [("" if i == jy else t) for i, t in enumerate(ay.labels)]


def linear_ticks(lo: float, hi: float, target: int) -> tuple[list[float], int]:
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return [lo], 0
    want = min(3, max(target, 2))
    for t in range(max(target, 1), max(target, 1) + 8):
        # A nice step can leave a single tick in range (e.g. step 100 on -85..82):
        # densify until at least ``want`` ticks fall inside.
        step = _nice_step(hi - lo, t)
        start = math.ceil(lo / step - 1e-9) * step
        ticks = []
        v = start
        while v <= hi + 1e-9 * step:
            ticks.append(round(v, 10))
            v += step
        if len(ticks) >= want:
            break
    return ticks, _decimals(step)


def log_ticks(lo: float, hi: float) -> tuple[list[float], list[str]]:
    """Plain decimal ticks: 1-3 per decade, or decades only for wide ranges."""
    lo, hi = max(lo, 1e-12), max(hi, lo * 1.0001)
    d0, d1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    decades = math.log10(hi / lo)
    mult = (1, 2, 5) if decades <= 1.2 else ((1, 3) if decades <= 3.5 else (1,))
    out = []
    for d in range(d0, d1 + 1):
        for m in mult:
            v = float(f"{m * 10.0 ** d:.12g}")
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    labels = [(f"{v:.{max(0, -int(math.floor(math.log10(v))))}f}" if v < 1 else f"{v:g}")
              for v in out]
    return out, labels


SIGBAR_ROW = 0.055          # fraction of the axis height per row of significance bars


def sigbar_rows(p: Panel) -> int:
    rows = [int(s.style.get("row", 0)) for s in p.series if s.kind == "sigbar"]
    return max(rows) + 1 if rows else 0


def sigbar_y(row: int) -> float:
    """Axis-relative height (0..1) of the centre of significance-bar row ``row``."""
    return 1.0 - SIGBAR_ROW * (row + 0.5)


# -- data extents ----------------------------------------------------------------------------
def _vals(*arrs) -> np.ndarray:
    out = [np.asarray(a, float).ravel() for a in arrs if a is not None]
    if not out:
        return np.empty(0)
    v = np.concatenate(out)
    return v[np.isfinite(v)]


def data_extent(p: Panel) -> dict[str, tuple[float, float] | None]:
    xs, ys, zs = [], [], []
    for s in p.series:
        k = s.kind
        if k in ("line", "scatter", "colorline", "fill", "band"):
            x = np.asarray(s.x, float).ravel()
            y = np.asarray(s.y, float).ravel()
            ok = np.isfinite(x) & np.isfinite(y)          # x range only where data exists
            if k == "band":
                y2 = np.asarray(s.y2, float).ravel()
                ok &= np.isfinite(y2)
                ys.append(y2[ok])
            xs.append(x[ok])
            ys.append(y[ok])
        elif k == "errorbar":
            xs.append(_vals(s.x))
            ys.append(_vals(s.y, s.y2, s.ylo))
        elif k in ("bar",):
            xs.append(_vals(s.x) + 0.0)
            xs.append(_vals(s.x) - 0.45)
            xs.append(_vals(s.x) + 0.45)
            ys.append(np.r_[0.0, _vals(s.y)])
        elif k == "barh":
            ys.append(_vals(s.y) - 0.45)
            ys.append(_vals(s.y) + 0.45)
            xs.append(np.r_[0.0, _vals(s.x)])
        elif k == "hline":
            ys.append(_vals(s.y))
        elif k == "vline":
            xs.append(_vals(s.x))
        elif k in ("hspan", "sigbar"):
            xs.append(_vals(s.x, s.y2))
        elif k == "heatmap":
            z = np.asarray(s.z)
            xs.append(np.array([-0.5, z.shape[1] - 0.5]))
            ys.append(np.array([-0.5, z.shape[0] - 0.5]))
        elif k == "text":
            xs.append(_vals(s.x))
            ys.append(_vals(s.y))
        elif k == "arrow":
            d = np.asarray(s.d, float).reshape(-1, 2) if s.d is not None else np.zeros((0, 2))
            xs.append(_vals(s.x, np.asarray(s.x, float) + d[:, 0]))
            ys.append(_vals(s.y, np.asarray(s.y, float) + d[:, 1]))
        elif k == "image" and s.extent is not None:
            xs.append(np.asarray(s.extent[:2], float))
            ys.append(np.asarray(s.extent[2:], float))
        elif k in KINDS_3D:
            xs.append(_vals(s.x))
            ys.append(_vals(s.y))
            zs.append(_vals(s.z))
            if k == "arrow3d" and s.d is not None:
                d = np.asarray(s.d, float).reshape(-1, 3)
                xs.append(_vals(np.asarray(s.x, float) + d[:, 0]))
                ys.append(_vals(np.asarray(s.y, float) + d[:, 1]))
                zs.append(_vals(np.asarray(s.z, float) + d[:, 2]))

    def rng(parts: list[np.ndarray]) -> tuple[float, float] | None:
        v = np.concatenate(parts) if parts else np.empty(0)
        return (float(v.min()), float(v.max())) if v.size else None

    return {"x": rng(xs), "y": rng(ys), "z": rng(zs)}


# -- results --------------------------------------------------------------------------------
@dataclass
class Box:
    x0: float
    y0: float          # from the top
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x0 + self.w

    @property
    def y1(self) -> float:
        return self.y0 + self.h

    def overlaps(self, o: "Box", tol: float = 1e-6) -> bool:
        return not (self.x1 <= o.x0 + tol or o.x1 <= self.x0 + tol or
                    self.y1 <= o.y0 + tol or o.y1 <= self.y0 + tol)


@dataclass
class AxisLayout:
    lim: tuple[float, float]
    ticks: list[float]
    labels: list[str]
    log: bool = False
    label: str = ""                 # plain label incl. unit
    rotate: int = 0                 # tick-label rotation (deg) chosen to avoid overlap

    def band_height(self, pt: float) -> float:
        """Height of the x tick-label band (rotated labels need more)."""
        if not self.rotate:
            return line_height_in(pt)
        w = max((text_width_in(t, pt) for t in self.labels), default=0.0)
        return w + 0.02


@dataclass
class PanelLayout:
    panel: Panel
    axis: Box
    title: Box | None
    xticks_band: Box | None
    xlabel_band: Box | None
    yticks_band: Box | None
    ylabel_band: Box | None
    legend: Box | None
    legend_cols: int
    colorbar: Box | None
    x: AxisLayout
    y: AxisLayout
    z: AxisLayout | None = None
    mode_legend: str = "none"
    title_lines: list[str] = field(default_factory=list)
    tag: Box | None = None            # bold panel letter, top-left corner of the panel cell
    tag_text: str = ""

    def text_boxes(self) -> list[Box]:
        return [b for b in (self.title, self.xticks_band, self.xlabel_band, self.yticks_band,
                            self.ylabel_band, self.legend, self.tag) if b is not None]


@dataclass
class FigureLayout:
    width: float
    height: float
    panels: list[PanelLayout] = field(default_factory=list)
    font: dict[str, float] = field(default_factory=dict)
    venue: str = ""


# -- the engine ------------------------------------------------------------------------------
def _resolve_axis(a: Axis, ext: tuple[float, float] | None, target: int, pad: float = 0.04,
                  categorical: bool = False) -> AxisLayout:
    if a.ticks is not None and a.ticklabels is not None:
        lo, hi = (a.lim if a.lim else ((min(a.ticks) - 0.5, max(a.ticks) + 0.5) if a.ticks
                                       else (0.0, 1.0)))
        if ext is not None and a.lim is None:
            lo, hi = min(lo, ext[0]), max(hi, ext[1])
        return AxisLayout((lo, hi), list(a.ticks), [plain_text(t) for t in a.ticklabels],
                          a.log, a.text())
    if a.lim is not None:
        lo, hi = a.lim
    elif ext is None:
        lo, hi = (1.0, 10.0) if a.log else (0.0, 1.0)
    else:
        lo, hi = ext
        if a.log:
            lo, hi = max(lo, 1e-9), max(hi, lo * 1.01)
            f = (hi / lo) ** pad
            lo, hi = lo / f, hi * f
        else:
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            lo, hi = lo - pad * span, hi + pad * span
    if a.log:
        ticks, labels = log_ticks(lo, hi)
        if a.ticks is not None:
            ticks = list(a.ticks)
            labels = [f"{t:g}" for t in ticks]
        return AxisLayout((lo, hi), ticks, labels, True, a.text())
    if a.ticks is not None:
        ticks = list(a.ticks)
        step = min(np.diff(ticks)) if len(ticks) > 1 else 1.0
        dec = _decimals(step)
    else:
        ticks, dec = linear_ticks(lo, hi, target)
    return AxisLayout((lo, hi), ticks, [fmt_tick(t, dec) for t in ticks], False, a.text())


def compute(spec: FigureSpec, width_in: float | None = None,
            font: dict[str, float] | None = None, glyph: float | None = None,
            venue: str | style.Venue | None = None) -> FigureLayout:
    """Panel geometry and axis resolution for ``spec`` at ``width_in``.

    ``venue`` (default: the active one, :func:`style.use_venue`; IEEE unless set) maps
    ``spec.width`` to inches and gives the font sizes; ``font`` overrides single sizes (the
    viewer and the videos use their own). ``glyph`` scales the text-width estimate for the
    renderer's font (default: the venue's -- 1.0 for the paper's Computer Modern / Times,
    ~1.08 for a Helvetica-like sans; ~1.15 for matplotlib's DejaVu Sans).
    """
    V = style.venue(venue)
    W = width_in or V.width_in(spec.width)
    F = dict(V.font_pt, **(font or {}))
    F.setdefault("tag", F.get("title", 8.0))
    glyph = V.glyph if glyph is None else glyph
    rows, cols = spec.rows, spec.cols
    panels = spec.panels
    lh = {k: line_height_in(v) for k, v in F.items()}

    def tw(t: str, key: str) -> float:
        return text_width_in(t, F[key]) * glyph

    # Shared x ranges.
    exts = [data_extent(p) for p in panels]
    shared: dict[str, tuple[float, float]] = {}
    for p, e in zip(panels, exts):
        if p.sharex and e["x"]:
            lo, hi = shared.get(p.sharex, e["x"])
            shared[p.sharex] = (min(lo, e["x"][0]), max(hi, e["x"][1]))
    # Shared y ranges: the union of the members' data and set limits; the significance-bar
    # head-room is added once, for the most bar rows in the group.
    shared_y: dict[str, tuple[float, float]] = {}
    bar_rows_y: dict[str, int] = {}
    for p, e in zip(panels, exts):
        r = tuple(p.y.lim) if p.y.lim else e["y"]
        if p.sharey and r and not p.y.log:
            lo, hi = shared_y.get(p.sharey, r)
            shared_y[p.sharey] = (min(lo, r[0]), max(hi, r[1]))
            bar_rows_y[p.sharey] = max(bar_rows_y.get(p.sharey, 0), sigbar_rows(p))
    for k, (lo, hi) in list(shared_y.items()):
        span = (hi - lo) or 1.0
        lo, hi = lo - 0.04 * span, hi + 0.04 * span        # the auto axis padding
        if bar_rows_y.get(k):
            n = bar_rows_y[k]
            hi += (hi - lo) * SIGBAR_ROW * n / (1 - SIGBAR_ROW * n)
        shared_y[k] = (lo, hi)

    def fit_x(p: Panel, ax: AxisLayout, w_axis: float) -> AxisLayout:
        """Rotate categorical labels / thin numeric ticks so the x labels fit ``w_axis``."""
        if p.x.ticklabels is not None and len(ax.ticks) > 1:
            spacing = w_axis * (min(np.diff(sorted(ax.ticks))) / (ax.lim[1] - ax.lim[0]))
            widest = max(tw(t, "tick") for t in ax.labels)
            # Vertical, not slanted: slanted labels still collide at tight spacing. Views
            # with long names should use horizontal forms (labels on the y axis).
            ax.rotate = 90 if widest + 0.05 > spacing else 0
            return ax
        if ax.log:
            return ax
        n = len(ax.ticks) + 1
        while len(ax.ticks) > 2 and sum(tw(t, "tick") + 0.08 for t in ax.labels) > w_axis:
            n = max(2, min(n, len(ax.ticks)) - 1)
            ticks, dec = linear_ticks(ax.lim[0], ax.lim[1], n)
            ax = AxisLayout(ax.lim, ticks, [fmt_tick(t, dec) for t in ticks], False, ax.label)
            if n == 2:
                break
        # Very narrow axes (equal-aspect plane views): the sparsest nice ticks can still be 3
        # wide labels; keep every other one, down to a single tick (the one nearest 0).
        while len(ax.ticks) > 1 and sum(tw(t, "tick") + 0.08 for t in ax.labels) > w_axis:
            keep = list(range(0, len(ax.ticks), 2)) if len(ax.ticks) > 2 else \
                [int(np.argmin(np.abs(ax.ticks)))]
            ax = AxisLayout(ax.lim, [ax.ticks[i] for i in keep], [ax.labels[i] for i in keep],
                            False, ax.label)
        return ax

    def fit_y(p: Panel, ay: AxisLayout, h_axis: float) -> AxisLayout:
        if p.y.ticklabels is not None or ay.log:
            return ay
        n = len(ay.ticks) + 1
        while len(ay.ticks) > 2 and len(ay.ticks) * (lh["tick"] + 0.04) > h_axis:
            n = max(2, min(n, len(ay.ticks)) - 1)
            ticks, dec = linear_ticks(ay.lim[0], ay.lim[1], n)
            ay = AxisLayout(ay.lim, ticks, [fmt_tick(t, dec) for t in ticks], False, ay.label)
            if n == 2:                   # the sparsest nice ticks; never loop forever
                break
        return ay

    def resolve(p: Panel, e: dict, w_axis: float, h_axis: float):
        xe = shared.get(p.sharex) if p.sharex else e["x"]
        heat = any(s.kind == "heatmap" for s in p.series)
        # 3-D: the projected axes are short and oblique, so at most ~4 ticks each, as on z
        # (denser x/y labels overprint at journal width)
        d3 = p.projection == "3d"
        ax = _resolve_axis(p.x, xe, 4 if d3 else max(2, int(w_axis / 0.42)),
                           0.0 if heat else 0.04)
        ye = e["y"]
        rows = sigbar_rows(p)
        if p.sharey in shared_y:
            ye, rows = shared_y[p.sharey], 0
            p = dataclasses.replace(p, y=dataclasses.replace(p.y, lim=ye))
        if rows and ye is not None and p.y.lim is None and not p.y.log:
            span = (ye[1] - ye[0]) or 1.0           # headroom: the bars sit above the data
            ye = (ye[0], ye[1] + span * SIGBAR_ROW * rows / (1 - SIGBAR_ROW * rows))
        ay = _resolve_axis(p.y, ye, 4 if d3 else max(2, int(h_axis / 0.28)),
                           0.0 if heat else 0.06)
        az = _resolve_axis(p.z, e["z"], 4) if p.projection == "3d" else None
        fx, fy = fit_x(p, ax, w_axis), fit_y(p, ay, h_axis)
        if p.projection != "3d":
            _clear_corner(p, fx, fy)
        return fx, fy, az

    def margins(p: Panel, ax: AxisLayout, ay: AxisLayout) -> tuple[float, float, float]:
        if p.projection == "3d":
            # the projected box's tick labels and axis labels lie outside the axis box
            lab = tw("−00", "tick") + lh["label"] + PAD
            return lab, lab + 0.1, lab
        ytw = max((tw(t, "tick") for t in ay.labels), default=0.0)
        left = ytw + TICK_LEN + PAD + 0.02 + ((lh["label"] + PAD) if ay.label else 0.0)
        bottom = TICK_LEN + ax.band_height(F["tick"] * glyph) + PAD + (
            (lh["label"] + PAD) if ax.label else 0.0)
        # The last x tick label is centred on the axis end and overhangs it.
        right = max(0.07, 0.5 * tw(ax.labels[-1], "tick") + 0.04) \
            if ax.labels and not ax.rotate else 0.06
        if p.colorbar is not None:
            # pgfplots places "colorbar right" 0.3 cm (0.118 in) right of the axis, then the
            # bar, its tick labels (with their own separation) and the rotated label
            right += CBAR_SHIFT_IN + 0.09 + PAD + max(tw("000.0", "tick"), 0.2) \
                + lh["label"] + PAD
        return left, right, bottom

    # Panel letters: bold, a little wider than the regular estimate.
    heads = [p.heading() for p in panels]
    tag_w = [tw(t, "tag") * 1.12 if t else 0.0 for t, _ in heads]

    # 1. Provisional axes -> margins -> the common axis width per column.
    prov = [resolve(p, e, W / cols * 0.7, spec.height_in) for p, e in zip(panels, exts)]
    col_left, col_right = [0.0] * cols, [0.0] * cols
    for k, p in enumerate(panels):
        left, right, _ = margins(p, prov[k][0], prov[k][1])
        if tag_w[k]:                        # the letter sits left of the title, outside the axis
            left = max(left, tag_w[k] + 0.05)
        col_left[k % cols] = max(col_left[k % cols], left)
        col_right[k % cols] = max(col_right[k % cols], right)
    gap = 0.10
    axis_w = max((W - sum(col_left) - sum(col_right) - gap * (cols - 1)) / cols, 0.4)

    # 2. Final box sizes; equal-aspect panels fit their cell (or share one scale).
    eq = [i for i, p in enumerate(panels) if p.equal and p.projection == "2d"]
    scales: dict[int, float] = {}
    for i in eq:
        ax, ay = prov[i][0], prov[i][1]
        scales[i] = min(axis_w / (ax.lim[1] - ax.lim[0]),
                        spec.height_in * panels[i].height / (ay.lim[1] - ay.lim[0]))
    if eq and spec.meta.get("equal_scale") == "shared":
        m = min(scales.values())
        scales = {i: m for i in eq}
    sizes, final = [], []
    for k, p in enumerate(panels):
        if k in scales:
            ax, ay, az = prov[k]
            w = (ax.lim[1] - ax.lim[0]) * scales[k]
            h = (ay.lim[1] - ay.lim[0]) * scales[k]
            final.append((fit_x(p, ax, w), fit_y(p, ay, h), az))
        else:
            w, h = axis_w, spec.height_in * (p.height if p.projection == "2d" else 1.0)
            final.append(resolve(p, exts[k], w, h))
        sizes.append((w, h))

    # 3. Column x positions, titles and legends (now that widths are known).
    col_x, x = [], 0.0
    for c in range(cols):
        col_x.append(x + col_left[c])
        x += col_left[c] + axis_w + col_right[c] + gap
    titles, legs = [], []
    for k, p in enumerate(panels):
        c = k % cols
        # A title may run from its axis' left edge to the next column (or the figure edge).
        cell = (col_x[c + 1] - col_left[c + 1] - col_x[c]) if c < cols - 1 else W - col_x[c]
        lines = wrap(plain_text(heads[k][1]), cell - 0.02, F["title"] * glyph) \
            if heads[k][1] else []
        titles.append(lines)
        mode = p.legend_mode()
        if mode != "below":
            legs.append((mode, 0, 0.0))
            continue
        widths = [0.22 + tw(s.label or "", "legend") + 0.12 for s in p.legend_entries()]
        x0 = col_x[c] - 0.8 * col_left[c]
        span = min(sizes[k][0] + 0.8 * col_left[c] + col_right[c], W - x0)
        if spec.meta.get("legend_span") == "figure":
            # the only legend of its row may run under the neighbouring panels (2026-11-05)
            span = W - x0
        n = len(widths)
        # Row-major fill; a column is as wide as its widest entry (as in pgf/matplotlib).
        while n > 1 and sum(max(widths[j::n]) for j in range(n)) > span:
            n -= 1
        # A legend row holds its sample image too: a little taller than a text line.
        legs.append((mode, n, math.ceil(len(widths) / n) * (lh["legend"] + 0.035) + PAD))

    # 4. Row heights.
    row_axis_h, row_top, row_bottom = [0.0] * rows, [0.0] * rows, [0.0] * rows
    for k, p in enumerate(panels):
        r = k // cols
        _, _, bottom = margins(p, final[k][0], final[k][1])
        row_axis_h[r] = max(row_axis_h[r], sizes[k][1])
        row_top[r] = max(row_top[r], len(titles[k]) * lh["title"] + PAD if titles[k] else 0.0,
                         lh["tag"] + PAD if heads[k][0] else 0.0)
        row_bottom[r] = max(row_bottom[r], bottom + legs[k][2])
    row_y, y = [], 0.02
    for r in range(rows):
        row_y.append(y + row_top[r])
        y += row_top[r] + row_axis_h[r] + row_bottom[r] + (0.06 if r < rows - 1 else 0.02)
    out = FigureLayout(W, y, font=F, venue=V.name)

    # 5. Boxes.
    for k, p in enumerate(panels):
        r, c = k // cols, k % cols
        ax, ay, az = final[k]
        w, h = sizes[k]
        if p.projection == "3d":
            h = row_axis_h[r]
        a = Box(col_x[c], row_y[r], w, h)
        title_h = len(titles[k]) * lh["title"] + PAD if titles[k] else 0.0
        cell = (col_x[c + 1] - col_left[c + 1] - col_x[c]) if c < cols - 1 else W - col_x[c]
        title = Box(a.x0, a.y0 - title_h, cell - 0.02, title_h - PAD / 2) if title_h else None
        if p.projection == "3d":
            xt = xl = yt = yl = None
            yb = a.y1 + PAD
        else:
            xt = Box(a.x0, a.y1 + TICK_LEN, w, ax.band_height(F["tick"] * glyph))
            xl = Box(a.x0, xt.y1 + PAD, w, lh["label"]) if ax.label else None
            ytw = max((tw(t, "tick") for t in ay.labels), default=0.0)
            yt = Box(a.x0 - TICK_LEN - ytw - 0.01, a.y0, ytw, h)
            yl = Box(yt.x0 - PAD - lh["label"], a.y0, lh["label"], h) if ay.label else None
            yb = (xl.y1 if xl else xt.y1) + PAD
        mode, ncol, lh_leg = legs[k]
        legend = None
        if lh_leg:
            x0 = a.x0 - 0.8 * col_left[c]
            legend = Box(x0, yb, min(w + 0.8 * col_left[c] + col_right[c], W - x0),
                         lh_leg - PAD)
        cb = Box(a.x1 + CBAR_SHIFT_IN, a.y0, 0.09, h) if p.colorbar is not None else None
        tag = None
        if heads[k][0]:
            # Fixed corner: the left edge of the panel cell, on the first title line (the top
            # of the row band for a panel without a title).
            ty = title.y0 if title is not None else row_y[r] - row_top[r]
            tag = Box(col_x[c] - col_left[c], ty, tag_w[k], lh["tag"])
        out.panels.append(PanelLayout(p, a, title, xt, xl, yt, yl, legend, ncol, cb, ax, ay, az,
                                      mode, titles[k], tag, heads[k][0]))
    return out


def check(layout: FigureLayout) -> list[str]:
    """Overlaps between any text band and another panel's box or text (empty = clean)."""
    problems = []
    boxes = []
    for i, pl in enumerate(layout.panels):
        boxes.append((f"panel {i} axis", pl.axis))
        boxes += [(f"panel {i} text {j}", b) for j, b in enumerate(pl.text_boxes())]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, a), (nb, b) = boxes[i], boxes[j]
            if na.split(" text")[0].split(" axis")[0] == nb.split(" text")[0].split(" axis")[0]:
                continue                       # bands of one panel abut by design
            if a.overlaps(b, 0.005):
                problems.append(f"{na} overlaps {nb}")
    for pl in layout.panels:
        for b in pl.text_boxes() + [pl.axis]:
            if b.x0 < -0.01 or b.x1 > layout.width + 0.01:
                problems.append(f"{pl.panel.title!r}: box outside the figure width")
        # An x-axis label is centred on its axis: wider than the space to the figure edges it
        # runs off the page (JNER revision 2026-11-10: "(f95) (H" was cut off).
        if pl.xlabel_band is not None and layout.font:
            pt = layout.font.get("label", 8.0)
            V = style.venue(layout.venue or None)
            w = text_width_in(pl.panel.x.text(), pt) * V.glyph
            cx = pl.xlabel_band.x0 + pl.xlabel_band.w / 2
            if cx - w / 2 < -0.01 or cx + w / 2 > layout.width + 0.01:
                problems.append(f"{pl.panel.title!r}: x-axis label runs past the figure edge")
    return problems


def L_title(p: Panel) -> str:
    """The drawn title: without its "(a)" prefix (the tag is drawn separately)."""
    return plain_text(p.heading()[1])


def wrap(text: str, width_in: float, pt: float) -> list[str]:
    """Greedy word wrap of ``text`` to ``width_in`` (titles never run past their cell).

    ``pt`` may include the renderer's glyph factor. An explicit line break in ``text`` always
    starts a new line (2026-11-05: a statistic on its own title line).
    """
    if "\n" in text:
        return [ln for part in text.split("\n") for ln in wrap(part, width_in, pt)]
    # Tokens; a slash glued to the next word ("between/within") may also break there.
    tokens: list[tuple[str, bool]] = []          # (text, joins to the previous without space)
    for word in text.split():
        parts = re.split(r"(?<=/)(?=\S)", word)
        tokens += [(pt, j > 0) for j, pt in enumerate(parts)]
    lines, cur = [], ""
    for w, glued in tokens:
        cand = (cur + w if glued else f"{cur} {w}").strip()
        if cur and text_width_in(cand, pt) > width_in:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines
