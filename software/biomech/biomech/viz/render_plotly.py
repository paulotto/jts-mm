# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Render specs with Plotly for the viewer (same specs and layout as the paper export).

Panels sit at the domains of :func:`biomech.viz.layout.compute` (computed for
the viewer's width and font sizes), every panel has its own legend placed in
the band below it (or in the corner an ``inside:<corner>`` policy names), titles
are left-aligned annotations, panel notes sit in the axis' top-left corner, and the
template is built from the same light/dark tokens as the paper and the videos.

Colour: sequential / diverging ramps follow the theme (``style.ramp_values``: the dark
ramps never glare on the dark panel and keep the high end visible); a ``colorline`` is
drawn as coloured line segments (one trace per colour bin, :func:`colorline_bins`);
every panel with a ``Colorbar`` gets one at the layout's colourbar box (heatmaps on the
trace itself, colour lines / coloured scatters / colour-mapped images through an
invisible marker trace). NaN cells of heatmaps and maps are transparent.

Payload: line, scatter, band and colour-line series longer than :data:`POINT_BUDGET`
points are thinned per finite run (min/max per bin for monotone x, LTTB otherwise,
always keeping each run's ends and extremes), so a long recording never ships 100k
points to the browser.

``apply_zoom`` maps a Plotly ``relayoutData`` back onto the spec's axis limits,
so "export what I see" carries the current zoom into the TikZ output.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import plotly.graph_objects as go

from biomech.viz import layout as L
from biomech.viz import style
from biomech.viz.downsample import thin_index
from biomech.viz.spec import FigureSpec, Panel, Series

_DASH = {"solid": "solid", "dashed": "dash", "dotted": "dot"}
_SYM = {"o": "circle", "s": "square", "^": "triangle-up", "d": "diamond", "v": "triangle-down",
        "x": "x", "|": "line-ns-open", "oo": "circle-open", "none": "circle"}
# Viewer typography (pt -> px at 96 dpi, slightly larger than print).
VIEW_FONT = {"label": 10.0, "tick": 9.0, "title": 10.5, "legend": 9.0, "note": 8.5,
             "tag": 10.5}
PX_PER_IN = 96.0
# Text-width factor of the viewer font against the layout's estimate: Inter, and the wider
# fallbacks a browser may substitute (Noto Sans, DejaVu Sans: up to ~1.12 x the estimate on
# long labels), so a tick-label band sized by the layout always holds its longest label.
VIEW_GLYPH = 1.15
VIEW_WIDTH_IN = 11.0
# Viewer pages (assets/style.css): .page max-width 1500 px, page + card padding and borders.
PAGE_MAX_PX, PAGE_PAD_PX = 1500, 62
MAX_VIEW_WIDTH_IN = (PAGE_MAX_PX - PAGE_PAD_PX) / PX_PER_IN
FIT_MIN_PX, FIT_MAX_PX = 240, 1100          # height of an aspect-fitted figure (px)
POINT_BUDGET = 5000                         # points per series sent to the browser
COLOR_BINS = 24                             # colour steps of a colour line
THIN_ROLES = ("context", "muted", "faint")  # de-emphasized: thin, no hover
_LEGEND_SAMPLE_PX = 46                      # Plotly legend: symbol (itemwidth 30) + gaps


def template(theme: str = "light") -> go.layout.Template:
    t = style.THEMES[theme]
    ink, ink2, grid, axis = ("#" + t[k] for k in ("ink", "ink2", "grid", "axis"))
    ax = {"gridcolor": grid, "gridwidth": 0.6, "linecolor": axis, "linewidth": 0.8,
          "zeroline": False, "ticks": "outside", "tickcolor": axis, "ticklen": 3,
          "tickfont": {"color": ink2, "size": VIEW_FONT["tick"] * 1.333},
          "title": {"font": {"color": ink, "size": VIEW_FONT["label"] * 1.333}},
          "showline": True, "mirror": False}
    scene_ax = {"backgroundcolor": "rgba(0,0,0,0)", "gridcolor": grid, "zerolinecolor": grid,
                "showbackground": False, "tickfont": {"color": ink2, "size": 10},
                "title": {"font": {"color": ink2, "size": 11}}}
    return go.layout.Template(layout={
        # The figure sits on a viewer card (style.css --card = panel): the same colour
        # around and inside the axes, in both themes (no dark frame).
        "paper_bgcolor": "#" + t["panel"], "plot_bgcolor": "#" + t["panel"],
        "font": {"family": "Inter, system-ui, -apple-system, Segoe UI, sans-serif", "color": ink,
                 "size": 12},
        "xaxis": ax, "yaxis": ax,
        "scene": {"xaxis": scene_ax, "yaxis": scene_ax, "zaxis": scene_ax},
        "hoverlabel": {"bgcolor": "#" + t["panel"], "bordercolor": axis,
                       "font": {"color": ink, "size": 12}},
        "colorway": [style.hexcolor(c, theme) for c in style.CATEGORICAL],
    })


def _strip(s: str | None) -> str | None:
    return None if s is None else L.plain_text(s)


def _colorscale(ramp: str = "sequential", theme: str = "light") -> list[list[Any]]:
    """Plotly colourscale of the theme's ramp (dark: no glaring low end, visible high end)."""
    return [[p, c] for p, c in style.ramp(style.ramp_values(ramp, theme))]


def _ramp_color(v: float, lim: tuple[float, float], ramp: str = "sequential",
                theme: str = "light") -> str:
    """'#rrggbb' of value ``v`` on the theme's ramp over ``lim``."""
    f = 0.0 if lim[1] == lim[0] else (v - lim[0]) / (lim[1] - lim[0])
    return style.ramp_color(f, ramp, theme)


def _hover(pl: L.PanelLayout, name: str) -> str:
    """x and y with their axis titles and units (``Frequency (Hz): 12.5``)."""
    xl = _strip(pl.panel.x.text()) or "x"
    yl = _strip(pl.panel.y.text()) or "y"
    return (f"{xl}: %{{x:.4~g}}<br>{yl}: %{{y:.4~g}}<extra>{name}</extra>")


def _runs(*arrays: np.ndarray) -> list[slice]:
    """Slices of the stretches where every array is finite (a band never bridges a gap)."""
    ok = np.ones(len(arrays[0]), bool)
    for a in arrays:
        ok &= np.isfinite(np.asarray(a, float))
    if not ok.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


# -- payload budget ------------------------------------------------------------------------
def _thinned(s: Series) -> Series:
    """``s`` within the point budget (line, scatter, band, colour line); others unchanged."""
    if s.kind not in ("line", "scatter", "band", "colorline") or s.x is None or \
            np.size(s.x) <= POINT_BUDGET:
        return s
    if s.kind == "scatter" and s.text:            # per-point labels: never drop points
        return s
    ys = [s.y] + ([s.y2] if s.kind == "band" else [])
    idx = thin_index(np.asarray(s.x, float), *ys, budget=POINT_BUDGET)
    pick = {k: (np.asarray(getattr(s, k))[idx] if getattr(s, k) is not None
                and np.size(getattr(s, k)) == np.size(s.x) else getattr(s, k))
            for k in ("x", "y", "y2", "c")}
    return dataclasses.replace(s, **pick)


# -- colour lines --------------------------------------------------------------------------
def colorline_lim(s: Series, panel: Panel | None = None) -> tuple[float, float] | None:
    """Colour range of a colour line: the panel's colourbar, the series' ``lim``, the data."""
    cb = panel.colorbar if panel is not None else None
    lim = cb.lim if cb and cb.lim else s.style.get("lim")
    if lim:
        return float(lim[0]), float(lim[1])
    c = np.asarray(s.c, float)
    fin = c[np.isfinite(c)]
    if not fin.size:
        return None
    lo, hi = float(fin.min()), float(fin.max())
    if cb is not None and cb.ramp == "diverging":
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    return lo, hi


def colorline_bins(x: np.ndarray, y: np.ndarray, c: np.ndarray,
                   lim: tuple[float, float] | None, nbins: int = COLOR_BINS,
                   *, all_bins: bool = False) -> list[tuple[int, np.ndarray, np.ndarray,
                                                            np.ndarray]]:
    """A colour line as line pieces per colour bin: ``(bin, x, y, c)`` with NaN between pieces.

    Each segment takes the mean colour value of its two points (as matplotlib's
    LineCollection and pgfplots' ``mesh``); consecutive segments of one bin form one piece,
    which also contains the next point, so pieces of neighbouring bins join without a gap.
    Segments with a non-finite point or colour value are not drawn. ``all_bins``: one entry
    per bin, empty ones included (a fixed trace count for animation frames).
    """
    x, y, c = (np.asarray(v, float) for v in (x, y, c))
    n = len(x)
    pieces: dict[int, list[tuple[int, int]]] = {}
    if n >= 2 and lim is not None:
        cv = 0.5 * (c[:-1] + c[1:])
        ok = (np.isfinite(cv) & np.isfinite(x[:-1]) & np.isfinite(x[1:]) & np.isfinite(y[:-1])
              & np.isfinite(y[1:]))
        span = (lim[1] - lim[0]) or 1.0
        f = np.clip((np.where(ok, cv, lim[0]) - lim[0]) / span, 0.0, 1.0)
        b = np.minimum((f * nbins).astype(int), nbins - 1)
        b[~ok] = -1
        cut = np.flatnonzero(np.diff(b)) + 1
        for s0, e0 in zip(np.r_[0, cut], np.r_[cut, len(b)]):
            if b[s0] >= 0:
                pieces.setdefault(int(b[s0]), []).append((int(s0), int(e0) + 1))
    out = []
    for k in (range(nbins) if all_bins else sorted(pieces)):
        idx: list[np.ndarray] = []
        for a, e in pieces.get(k, []):
            idx += [np.arange(a, e), np.array([-1])]
        if idx:
            ii = np.concatenate(idx)[:-1]
            brk = ii < 0
            ii = np.where(brk, 0, ii)
            xs, ys, cs = (np.where(brk, np.nan, v[ii]) for v in (x, y, c))
        else:
            xs = ys = cs = np.empty(0)
        out.append((k, xs, ys, cs))
    return out


def _bin_color(k: int, nbins: int, ramp: str, theme: str) -> str:
    return style.ramp_color((k + 0.5) / nbins, ramp, theme)


def colorline_traces(s: Series, xa: str, ya: str, legend: str, theme: str,
                     lim: tuple[float, float] | None, *, name: str = "", show: bool = False,
                     width: float = 2.0, ramp: str = "sequential", hover_label: str = "value",
                     all_bins: bool = False) -> list[go.Scattergl]:
    """Plotly traces of a colour line: one coloured line trace per colour bin."""
    cc = np.asarray(s.c, float)
    fin = cc[np.isfinite(cc)]
    if not all_bins and lim and fin.size and float(fin.min()) == float(fin.max()):
        # A constant-colour line (e.g. a trial coloured by its order): one exact colour.
        return [go.Scattergl(x=s.x, y=s.y, mode="lines", name=name, showlegend=show,
                             line={"color": _ramp_color(float(fin[0]), lim, ramp, theme),
                                   "width": width * 0.8}, hoverinfo="skip", xaxis=xa,
                             yaxis=ya, legend=legend)]
    traces = []
    first = True
    for k, xs, ys, cs in colorline_bins(s.x, s.y, cc, lim, all_bins=all_bins):
        traces.append(go.Scattergl(
            x=xs, y=ys, mode="lines", name=name, legendgroup=f"cl-{id(s)}",
            showlegend=show and first and xs.size > 0, customdata=cs,
            line={"color": _bin_color(k, COLOR_BINS, ramp, theme), "width": width},
            hovertemplate=f"{hover_label}: %{{customdata:.3~g}}<extra>{name}</extra>",
            xaxis=xa, yaxis=ya, legend=legend))
        first = first and xs.size == 0
    return traces


# -- series ----------------------------------------------------------------------------------
def _coord(v: float, log: bool) -> float:
    """Annotation coordinate on a data axis: Plotly wants log10 values on log axes
    (shapes take data values on any axis type)."""
    v = float(v)
    if log:
        return float(np.log10(v)) if v > 0 else float("nan")
    return v


def _colorbar(pl: L.PanelLayout, size: tuple[float, float] | None) -> dict[str, Any]:
    """Plotly ``colorbar`` placed at the layout's colourbar box (``{}``: no colourbar)."""
    cb = pl.panel.colorbar
    if cb is None or pl.colorbar is None or not size:
        return {}
    W, H = size
    bx = pl.colorbar
    return {"x": bx.x0 / W, "xanchor": "left", "y": 1 - (bx.y0 + bx.y1) / 2 / H,
            "yanchor": "middle", "len": (bx.y1 - bx.y0) / H, "lenmode": "fraction",
            "thickness": max(8, int(bx.w * PX_PER_IN)),
            "title": {"text": _strip(cb.text()), "side": "right",
                      "font": {"size": VIEW_FONT["tick"] * 1.333}},
            "tickfont": {"size": VIEW_FONT["tick"] * 1.333},
            "outlinewidth": 0, **({"tickvals": cb.ticks} if cb.ticks else {})}


def _panel_clim(p: Panel) -> tuple[float, float]:
    """Colour range of a panel's colourbar: its limits, else the range of the colour data
    (colour lines, coloured scatters, maps), symmetric about 0 on a diverging ramp."""
    cb = p.colorbar
    if cb is not None and cb.lim:
        return float(cb.lim[0]), float(cb.lim[1])
    vals = []
    for s in p.series:
        if s.kind in ("colorline", "scatter") and s.c is not None:
            vals.append(np.ravel(np.asarray(s.c, float)))
            if s.style.get("lim"):
                vals.append(np.asarray(s.style["lim"], float))
        elif s.kind == "image" and s.style.get("cmap"):
            vals.append(np.ravel(np.asarray(s.style.get("lim") or s.img, float)))
    v = np.concatenate(vals) if vals else np.empty(0)
    v = v[np.isfinite(v)]
    lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
    if cb is not None and cb.ramp == "diverging":
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    return (lo, hi) if hi > lo else (lo - 0.5, hi + 0.5)


def _colorbar_trace(p: Panel, pl: L.PanelLayout, k: int, theme: str, legend: str,
                    size: tuple[float, float] | None) -> go.Scatter | None:
    """An invisible marker trace that carries the panel's colourbar (non-heatmap panels)."""
    bar = _colorbar(pl, size)
    if not bar:
        return None
    lo, hi = _panel_clim(p)
    xa, ya = ("x" if k == 0 else f"x{k + 1}"), ("y" if k == 0 else f"y{k + 1}")
    # One real (invisible) point inside the axis: Plotly draws no colourbar for a trace
    # without data.
    return go.Scatter(x=[pl.x.lim[0]], y=[pl.y.lim[0]], mode="markers", showlegend=False,
                      hoverinfo="skip",
                      marker={"color": [lo], "colorscale": _colorscale(p.colorbar.ramp, theme),
                              "cmin": lo, "cmax": hi, "showscale": True, "colorbar": bar,
                              "size": 0.1, "opacity": 0},
                      xaxis=xa, yaxis=ya, legend=legend)


def _cell_dark(f: np.ndarray, ramp: str, theme: str) -> np.ndarray:
    """Which heatmap cells (colour position ``f``) are dark enough to need light text."""
    if theme == "dark":
        return (f < 0.6) if ramp == "sequential" else (np.abs(f - 0.5) < 0.33)
    return (f > 0.55) if ramp == "sequential" else (np.abs(f - 0.5) > 0.32)


def _add(fig: go.Figure, s: Series, k: int, pl: L.PanelLayout, theme: str, legend: str,
         size: tuple[float, float] | None = None) -> None:
    """Add series ``s`` of panel ``k``. ``size``: the layout size (in) -- needed to place a
    heatmap's colourbar at the layout's colourbar box."""
    s = _thinned(s)
    colname, dash, mark = style.resolve(s)
    color = style.hexcolor(colname, theme)
    # A panel whose legend policy is "none" (one labelled series: the title names it) must
    # not feed Plotly's default legend either.
    show = bool(s.label) and getattr(pl, "mode_legend", "below") != "none"
    name = _strip(s.label) or ""
    xa, ya = ("x" if k == 0 else f"x{k + 1}"), ("y" if k == 0 else f"y{k + 1}")
    xlog, ylog = bool(pl.x.log), bool(pl.y.log)
    ref = {"xaxis": xa, "yaxis": ya, "legend": legend}
    thin = s.role in THIN_ROLES
    lw = (1.0 if thin else 2.0) * float(s.style.get("width", 1.0))
    msz = 7 * (style.marker_factor(s) if s.kind == "scatter"
               else float(s.style.get("size", 1.0)))
    surface = style.hexcolor("panel", theme)   # rings: the card colour
    # De-emphasized context is not hoverable: hovering it would hide the data it frames.
    hover: dict[str, Any] = ({"hoverinfo": "skip"} if thin else
                             {"hovertemplate": _hover(pl, name or "")})
    cb = pl.panel.colorbar
    if s.kind == "line":
        fig.add_trace(go.Scattergl(x=s.x, y=s.y, mode="lines", name=name, showlegend=show,
                                   line={"color": color, "width": lw, "dash": _DASH[dash]},
                                   opacity=float(s.style.get("alpha", 1.0)), **hover, **ref))
    elif s.kind == "scatter":
        sym = s.style.get("ms", mark)
        marker = {"color": color if s.c is None else s.c, "symbol": _SYM.get(sym, "circle"),
                  "size": msz, "line": {"color": surface,
                                        "width": 0.8 * (style.halo_pt(s) or 0.4) / 0.4}}
        if sym == "|":                                   # tick marks are drawn by their line
            marker.update(size=msz * 1.6, line={"color": color, "width": 1.5})
        if s.c is not None:
            ramp = cb.ramp if cb else "sequential"
            lo, hi = _panel_clim(pl.panel) if cb else (colorline_lim(s) or (0.0, 1.0))
            marker.update(colorscale=_colorscale(ramp, theme), showscale=False, cmin=lo,
                          cmax=hi)
        fig.add_trace(go.Scattergl(x=s.x, y=s.y, mode="markers", name=name, showlegend=show,
                                   marker=marker, **hover, **ref))
    elif s.kind in ("band", "fill"):
        if s.kind == "band":
            # One polygon per finite stretch: a band never bridges a gap (masked samples,
            # cycles without data).
            x = np.asarray(s.x, float)
            lo_, hi_ = np.asarray(s.y, float), np.asarray(s.y2, float)
            polys = [(np.r_[x[r], x[r][::-1]], np.r_[lo_[r], hi_[r][::-1]])
                     for r in _runs(x, lo_, hi_)]
        else:
            polys = [(np.asarray(s.x, float), np.asarray(s.y, float))]
        group = f"band-{id(s)}"
        for i, (xx, yy) in enumerate(polys):
            fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", fill="toself",
                                     fillcolor=style.rgba(colname, style.BAND_ALPHA * 1.3, theme),
                                     line={"width": 0.6 if s.kind == "fill" else 0,
                                           "color": color},
                                     name=name, showlegend=show and i == 0, legendgroup=group,
                                     hoverinfo="skip", **ref))
    elif s.kind == "errorbar":
        y = np.asarray(s.y, float)
        fig.add_trace(go.Scatter(x=s.x, y=y, mode="markers", name=name, showlegend=show,
                                 marker={"color": color, "symbol": _SYM[mark], "size": msz},
                                 error_y={"type": "data", "array": np.asarray(s.y2) - y,
                                          "arrayminus": y - np.asarray(s.ylo), "width": 0,
                                          "color": color}, **ref))
    elif s.kind in ("bar", "barh"):
        horiz = s.kind == "barh"
        fig.add_trace(go.Bar(x=s.x, y=s.y, orientation="h" if horiz else "v", name=name,
                             showlegend=show, marker={"color": color, "line": {"width": 0}},
                             width=float(s.style.get("bar_width", 0.62)), **ref))
    elif s.kind in ("hline", "vline"):
        vals = np.atleast_1d(s.y if s.kind == "hline" else s.x)
        for i, v in enumerate(vals):
            # Spans the whole axis in domain units, so zooming never clips the line.
            shape = {"type": "line", "line": {"color": color, "dash": _DASH[dash], "width": 1.2},
                     "layer": "below"}
            if s.kind == "hline":
                shape.update(x0=0, x1=1, y0=float(v), y1=float(v), xref=f"{xa} domain",
                             yref=ya)
            else:
                shape.update(y0=0, y1=1, x0=float(v), x1=float(v), xref=xa,
                             yref=f"{ya} domain")
            fig.add_shape(**shape)
            t = s.text[i] if s.text and i < len(s.text) else (s.label if i == 0 else None)
            if t:
                rev = pl.panel.y.reverse or any(q.kind == "heatmap" for q in pl.panel.series)
                left = s.kind == "hline" and s.style.get("anchor") == "left"
                fig.add_annotation(x=(0 if left else 1) if s.kind == "hline" else _coord(v, xlog),
                                   y=_coord(v, ylog) if s.kind == "hline" else (0 if rev else 1),
                                   xref=f"{xa} domain" if s.kind == "hline" else xa,
                                   yref=ya if s.kind == "hline" else f"{ya} domain",
                                   text=_strip(t), showarrow=False,
                                   xanchor=("left" if left else "right") if s.kind == "hline"
                                   else "left",
                                   yanchor="bottom" if s.kind == "hline" else "top",
                                   font={"size": VIEW_FONT["note"] * 1.333,
                                         "color": style.hexcolor("ink2", theme)})
    elif s.kind == "sigbar":
        yy = L.sigbar_y(int(s.style.get("row", 0)))
        bar_c = color if s.role not in THIN_ROLES else style.hexcolor("ink2", theme)
        for a, b in zip(np.atleast_1d(s.x), np.atleast_1d(s.y2)):
            fig.add_shape(type="line", x0=float(a), x1=float(b), y0=yy, y1=yy, xref=xa,
                          yref=f"{ya} domain", line={"color": bar_c, "width": 4})
        if show:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name,
                                     line={"color": bar_c, "width": 4}, **ref))
    elif s.kind == "hspan":
        for a, b in zip(np.atleast_1d(s.x), np.atleast_1d(s.y2)):
            fig.add_shape(type="rect", x0=float(a), x1=float(b), y0=0, y1=1,
                          xref=xa, yref=f"{ya} domain",
                          fillcolor=style.rgba(colname, float(s.style.get("alpha", 0.12)),
                                               theme),
                          line_width=0, layer="below")
    elif s.kind == "heatmap":
        _heatmap(fig, s, pl, theme, xa, ya, size)
    elif s.kind == "text":
        for xx, yy, t in zip(np.atleast_1d(s.x), np.atleast_1d(s.y), s.text or []):
            fig.add_annotation(x=_coord(xx, xlog), y=_coord(yy, ylog), xref=xa, yref=ya,
                               text=_strip(t), showarrow=False,
                               xanchor={"right": "left", "left": "right"}.get(
                                   s.style.get("anchor", "right"), "center"),
                               yanchor={"above": "bottom", "below": "top"}.get(
                                   s.style.get("anchor", "right"), "middle"),
                               font={"size": VIEW_FONT["note"] * 1.333,
                                     "color": style.hexcolor(s.style.get("color", "ink2"), theme)})
    elif s.kind == "colorline":
        lim = colorline_lim(s, pl.panel)
        label = _strip(cb.text()) if cb is not None else "value"
        fig.add_traces(colorline_traces(s, xa, ya, legend, theme, lim, name=name, show=show,
                                        width=lw * 0.8 if thin else lw * 0.75,
                                        ramp=cb.ramp if cb else "sequential",
                                        hover_label=label or "value"))
    elif s.kind == "arrow":
        d = np.asarray(s.d, float).reshape(-1, 2)
        xs, ys = [], []
        for x0, y0, (dx, dy) in zip(np.atleast_1d(s.x), np.atleast_1d(s.y), d):
            xs += [x0, x0 + dx, None]
            ys += [y0, y0 + dy, None]
        # Shafts as one line trace (legend entry) + heads as angle-following markers.
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, showlegend=show,
                                 line={"color": color, "width": lw}, hoverinfo="skip", **ref))
        tips_x = np.atleast_1d(s.x) + d[:, 0]
        tips_y = np.atleast_1d(s.y) + d[:, 1]
        hx = np.ravel(np.column_stack([np.atleast_1d(s.x), tips_x, np.full(len(d), np.nan)]))
        hy = np.ravel(np.column_stack([np.atleast_1d(s.y), tips_y, np.full(len(d), np.nan)]))
        sizes = np.ravel(np.column_stack([np.zeros(len(d)), np.full(len(d), 9.0),
                                          np.zeros(len(d))]))
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker={"symbol": "arrow", "angleref": "previous",
                                         "size": sizes, "color": color, "line": {"width": 0}},
                                 **ref))
    elif s.kind == "image":
        _image(fig, s, pl, theme, xa, ya)


def _heatmap(fig: go.Figure, s: Series, pl: L.PanelLayout, theme: str, xa: str, ya: str,
             size: tuple[float, float] | None) -> None:
    z = np.asarray(s.z, float)
    text = np.array([_strip(t) for t in s.text]).reshape(z.shape) if s.text else None
    cb = pl.panel.colorbar
    rows = list(pl.panel.y.ticklabels or range(z.shape[0]))
    cols = list(pl.panel.x.ticklabels or range(z.shape[1]))
    unit = f" {_strip(style.plain_unit(cb.unit))}" if cb and cb.unit else ""
    cd = np.array([[f"{rows[i] if i < len(rows) else i} · {cols[j] if j < len(cols) else j}"
                    for j in range(z.shape[1])] for i in range(z.shape[0])], dtype=object)
    bar = _colorbar(pl, size)
    ramp = cb.ramp if cb else "sequential"
    fin = np.isfinite(z)
    zlo, zhi = (cb.lim if cb and cb.lim else (np.nanmin(z) if fin.any() else 0,
                                               np.nanmax(z) if fin.any() else 1))
    if ramp == "diverging" and not (cb and cb.lim):
        m = max(abs(zlo), abs(zhi))
        zlo, zhi = -m, m
    # Cell values in a text colour that contrasts with the cell; empty (NaN, transparent)
    # cells take the theme's ink.
    per_cell = text is not None and z.size <= 600
    xlog, ylog = bool(pl.x.log), bool(pl.y.log)
    if per_cell:
        f = np.clip((z - zlo) / ((zhi - zlo) or 1.0), 0, 1)
        dark = _cell_dark(f, ramp, theme)
        ink = style.hexcolor("ink", theme)
        for (i, j), t in np.ndenumerate(text):
            if t:
                c = ink if not fin[i, j] else ("#FFFFFF" if dark[i, j] else "#1F2328")
                fig.add_annotation(x=_coord(j, xlog), y=_coord(i, ylog), xref=xa, yref=ya,
                                   text=t, showarrow=False,
                                   font={"size": VIEW_FONT["note"] * 1.333, "color": c})
    fig.add_trace(go.Heatmap(z=z, colorscale=_colorscale(ramp, theme),
                             showscale=bool(bar), text=None if per_cell else text,
                             customdata=cd,
                             texttemplate="%{text}" if text is not None and not per_cell
                             else None,
                             zmin=zlo, zmax=zhi, xaxis=xa, yaxis=ya,
                             hovertemplate=f"%{{customdata}}<br>%{{z:.4~g}}{unit}"
                                           "<extra></extra>",
                             **({"colorbar": bar} if bar else {})))


def _image(fig: go.Figure, s: Series, pl: L.PanelLayout, theme: str, xa: str,
           ya: str) -> None:
    img = np.asarray(s.img)
    ex = s.extent or (0, img.shape[1], img.shape[0], 0)
    if img.ndim == 2 and s.style.get("cmap"):
        # A colour-mapped scalar map: a heatmap (z, hover, colourbar, NaN transparent), never
        # a baked RGB picture. Row 0 sits at ex[3] (the top, as in matplotlib's imshow).
        z = np.asarray(img, float)
        Hh, Ww = z.shape
        xs = ex[0] + (np.arange(Ww) + 0.5) * (ex[1] - ex[0]) / Ww
        ys = ex[3] + (np.arange(Hh) + 0.5) * (ex[2] - ex[3]) / Hh
        flip = Hh > 1 and ys[1] < ys[0]
        if flip:
            ys, z = ys[::-1], z[::-1]
        lo, hi = s.style.get("lim") or _panel_clim(pl.panel)
        ramp = s.style.get("cmap") if s.style.get("cmap") in ("sequential", "diverging") \
            else "sequential"
        unit = ""
        cb = pl.panel.colorbar
        if cb is not None and cb.unit:
            unit = f" {_strip(style.plain_unit(cb.unit))}"
        mask = s.style.get("mask")
        if mask is not None:
            m = np.asarray(mask, bool)
            if flip:
                m = m[::-1]
            if m.any():
                muted = style.hexcolor("muted", theme)
                fig.add_trace(go.Heatmap(z=np.where(m, 1.0, np.nan), x=xs, y=ys,
                                         colorscale=[[0, muted], [1, muted]], showscale=False,
                                         hoverinfo="skip", xaxis=xa, yaxis=ya))
        fig.add_trace(go.Heatmap(z=z, x=xs, y=ys, colorscale=_colorscale(ramp, theme),
                                 zmin=lo, zmax=hi, showscale=False, xaxis=xa, yaxis=ya,
                                 hovertemplate=f"x %{{x:.3~g}}, y %{{y:.3~g}}: %{{z:.3~g}}"
                                               f"{unit}<extra></extra>"))
        return
    fig.add_trace(go.Image(z=_rgb(img), x0=ex[0], dx=(ex[1] - ex[0]) / img.shape[1],
                           y0=ex[3], dy=(ex[2] - ex[3]) / img.shape[0], xaxis=xa, yaxis=ya))


def _rgb(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(np.nan_to_num(a, nan=1.0), 0, 1) * 255).astype(np.uint8)
    return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a


def _add3d(fig: go.Figure, s: Series, scene: str, theme: str, legend: str) -> None:
    colname, dash, mark = style.resolve(s)
    color = style.hexcolor(colname, theme)
    thin = s.role in THIN_ROLES
    name = _strip(s.label) or ""
    skip = {"hoverinfo": "skip"} if thin else {}
    if s.kind == "line3d":
        fig.add_trace(go.Scatter3d(x=s.x, y=s.y, z=s.z, mode="lines", name=name,
                                   showlegend=bool(s.label), scene=scene, legend=legend,
                                   line={"color": color, "width": 2 if thin else 4}, **skip))
    elif s.kind == "scatter3d":
        fig.add_trace(go.Scatter3d(x=s.x, y=s.y, z=s.z, mode="markers", name=name,
                                   showlegend=bool(s.label), scene=scene, legend=legend,
                                   marker={"color": color, "size": 4,
                                           "symbol": "diamond" if mark == "d" else "circle"},
                                   **skip))
    elif s.kind == "surface3d":
        z = np.asarray(s.z, float)
        fig.add_trace(go.Surface(x=s.x, y=s.y, z=z,
                                 surfacecolor=np.asarray(s.c, float) if s.c is not None else z,
                                 colorscale=_colorscale(str(s.style.get("ramp", "sequential")),
                                                        theme),
                                 showscale=False, scene=scene, name=name,
                                 opacity=float(s.style.get("alpha", 1.0)), **skip))
    elif s.kind == "arrow3d":
        d = np.asarray(s.d, float).reshape(-1, 3)
        x0, y0, z0 = (np.atleast_1d(v) for v in (s.x, s.y, s.z))
        xs, ys, zs = [], [], []
        for i in range(len(d)):
            xs += [x0[i], x0[i] + d[i, 0], None]
            ys += [y0[i], y0[i] + d[i, 1], None]
            zs += [z0[i], z0[i] + d[i, 2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, scene=scene,
                                   showlegend=bool(s.label), legend=legend,
                                   line={"color": color, "width": 4}, **skip))


def reflow(spec: FigureSpec, cols: int) -> FigureSpec:
    """The same panels in ``cols`` columns (narrow screens stack them).

    Axis titles a small-multiples grid removed (``Panel.hidden_labels``) come back when the
    panels are stacked into one column: every panel then stands alone. The spec's panels
    are copied (axes only), never mutated -- specs are cached by the viewer.
    """
    rows = -(-len(spec.panels) // cols)
    panels = spec.panels
    if cols == 1 and any(p.hidden_labels for p in panels):
        panels = []
        for p in spec.panels:
            x, y = copy.copy(p.x), copy.copy(p.y)
            if "y" in p.hidden_labels and not y.label:
                y.label, y.unit = p.hidden_labels["y"]
            if "x" in p.hidden_labels and not x.label and not p.sharex:
                x.label, x.unit = p.hidden_labels["x"]
            panels.append(dataclasses.replace(p, x=x, y=y))
    return dataclasses.replace(spec, cols=cols, rows=rows, panels=panels)


def container_width_in(viewport_px: int) -> float:
    """Width (in) of a figure card on a viewer page ``viewport_px`` wide."""
    return min(MAX_VIEW_WIDTH_IN, max(3.0, (min(viewport_px, PAGE_MAX_PX) - PAGE_PAD_PX)
                                      / PX_PER_IN))


def paper_aspect(spec: FigureSpec) -> float:
    """Height / width of the figure at its paper width (IEEE column or text, paper fonts)."""
    lay = L.compute(spec)
    return lay.height / lay.width if lay.width > 0 else 0.5


def fit_height(spec: FigureSpec, width_in: float, *, min_px: float = FIT_MIN_PX,
               max_px: float = FIT_MAX_PX) -> FigureSpec:
    """``spec`` with its row height scaled so that, laid out at ``width_in`` with the viewer
    fonts, it keeps the paper aspect ratio (height clamped to ``min_px``..``max_px``).

    Without it a wide viewer card flattens every row to the paper's inch height; equal-aspect
    (mm-for-mm) and 3-D panels then shrink to a strip. Equal-aspect panels still never grow
    beyond what their column width allows (layout.compute), so they are not distorted.
    """
    if not spec.panels or spec.rows < 1 or spec.height_in <= 0:
        return spec
    target = min(max(width_in * paper_aspect(spec), min_px / PX_PER_IN), max_px / PX_PER_IN)
    lay = L.compute(spec, width_in, font=VIEW_FONT, glyph=VIEW_GLYPH)
    fixed = max(lay.height - spec.rows * spec.height_in, 0.0)   # titles, ticks, legends
    row_h = (target - fixed) / spec.rows
    k = min(max(row_h / spec.height_in, 0.6), 6.0)
    return dataclasses.replace(spec, height_in=spec.height_in * k)


# Plotly legend placement of a pgfplots ``legend pos`` corner (inside the axis box).
_CORNER = {"north east": ("x1", "y0", "right", "top"), "north west": ("x0", "y0", "left", "top"),
           "south east": ("x1", "y1", "right", "bottom"),
           "south west": ("x0", "y1", "left", "bottom"),
           "outer north east": ("x1", "y0", "left", "top")}


def _legend_layout(pl: L.PanelLayout, W: float, H: float) -> dict[str, Any]:
    b = pl.axis
    font = {"size": VIEW_FONT["legend"] * 1.333}
    base = {"bgcolor": "rgba(0,0,0,0)", "font": font, "itemwidth": 30, "tracegroupgap": 0}
    mode = pl.mode_legend
    if mode.startswith("inside:"):
        xk, yk, xa, ya = _CORNER.get(mode[7:], _CORNER["north east"])
        x = (b.x1 if xk == "x1" else b.x0) / W
        y = 1 - (b.y0 if yk == "y0" else b.y1) / H
        if mode[7:] == "outer north east":
            x += 0.01
        return {**base, "x": x, "y": y, "xanchor": xa, "yanchor": ya, "orientation": "v"}
    lb = pl.legend
    if lb is None:
        return {**base, "x": b.x1 / W, "y": 1 - b.y0 / H, "xanchor": "right", "yanchor": "top",
                "orientation": "h"}
    ncol = max(pl.legend_cols, 1)
    # Each entry as wide as the layout's legend column: Plotly then wraps the entries into
    # the same columns as the paper legend (entrywidth = text part, without the sample).
    entry = max(20.0, lb.w * PX_PER_IN / ncol - _LEGEND_SAMPLE_PX)
    return {**base, "x": lb.x0 / W, "y": 1 - lb.y0 / H, "xanchor": "left", "yanchor": "top",
            "orientation": "h", "entrywidth": entry, "entrywidthmode": "pixels"}


def to_plotly(spec: FigureSpec, *, theme: str = "light", width_in: float = VIEW_WIDTH_IN,
              height_px: int | None = None, viewport_px: int | None = None,
              fit: bool = False) -> go.Figure:
    """``viewport_px``: the page width; below ~640 px multi-column figures stack.

    ``fit``: lay the figure out at the card width with the paper's aspect ratio
    (:func:`fit_height`); ``fig.layout.meta["aspect"]`` is then width / height, for a
    container that keeps that ratio while the page is resized (viewer ``render``).
    """
    if viewport_px:
        width_in = (container_width_in(viewport_px) if fit
                    else min(VIEW_WIDTH_IN, max(3.0, (viewport_px - 48) / PX_PER_IN)))
        if width_in < 6.5 and spec.cols > 1:
            spec = reflow(spec, 1)
    if fit:
        spec = fit_height(spec, width_in)
    lay = L.compute(spec, width_in, font=VIEW_FONT, glyph=VIEW_GLYPH)
    W, H = lay.width, lay.height
    size = (W, H)                            # layout size (in): colourbar placement
    fig = go.Figure()
    lay_kw: dict[str, Any] = {}
    ink2 = style.hexcolor("ink2", theme)
    for k, pl in enumerate(lay.panels):
        p = pl.panel
        legend = "legend" if k == 0 else f"legend{k + 1}"
        b = pl.axis
        xdom = [max(0.0, b.x0 / W), min(1.0, b.x1 / W)]
        ydom = [max(0.0, 1 - b.y1 / H), min(1.0, 1 - b.y0 / H)]
        if p.projection == "3d":
            scene = "scene" if k == 0 else f"scene{k + 1}"
            for s in p.series:
                _add3d(fig, s, scene, theme, legend)
            spans = [pl.x.lim[1] - pl.x.lim[0], pl.y.lim[1] - pl.y.lim[0],
                     pl.z.lim[1] - pl.z.lim[0]]
            m = max(spans)
            lay_kw[scene] = {"domain": {"x": xdom, "y": ydom}, "aspectmode": "manual",
                             "aspectratio": {"x": spans[0] / m, "y": spans[1] / m,
                                             "z": spans[2] / m},
                             "xaxis": {"title": {"text": _strip(pl.x.label)}},
                             "yaxis": {"title": {"text": _strip(pl.y.label)}},
                             "zaxis": {"title": {"text": _strip(pl.z.label)}},
                             "camera": _camera(p.view)}
            if p.note:
                fig.add_annotation(text=_strip(p.note), xref="paper", yref="paper",
                                   x=b.x0 / W, y=1 - b.y0 / H, xanchor="left", yanchor="top",
                                   showarrow=False, align="left",
                                   font={"size": VIEW_FONT["note"] * 1.333, "color": ink2})
        else:
            for s in p.series:
                _add(fig, s, k, pl, theme, legend, size)
            heat = any(s.kind == "heatmap" for s in p.series)
            if p.colorbar is not None and not heat:
                tr = _colorbar_trace(p, pl, k, theme, legend, size)
                if tr is not None:
                    fig.add_trace(tr)
            sfx = "" if k == 0 else str(k + 1)
            for ax_name, a, dom in (("xaxis", pl.x, xdom), ("yaxis", pl.y, ydom)):
                d: dict[str, Any] = {"domain": dom, "anchor": ("y" if ax_name == "xaxis"
                                                              else "x") + sfx,
                                     "title": {"text": _strip(a.label), "standoff": 4},
                                     "tickvals": a.ticks, "ticktext": a.labels,
                                     "showgrid": not heat,
                                     # A card narrower than the layout width would squeeze the
                                     # label bands: let Plotly grow the outer margin instead of
                                     # clipping labels at the figure edge.
                                     "automargin": True}
                if a.log:
                    d["type"] = "log"
                    d["range"] = [np.log10(a.lim[0]), np.log10(a.lim[1])]
                else:
                    d["range"] = list(a.lim)
                if ax_name == "yaxis" and (p.y.reverse or heat):
                    d["range"] = d["range"][::-1]
                if ax_name == "xaxis" and a.rotate:
                    d["tickangle"] = -a.rotate
                if p.sharex and ax_name == "xaxis":
                    first = next(i for i, q in enumerate(spec.panels) if q.sharex == p.sharex)
                    if first != k:
                        d["matches"] = "x" if first == 0 else f"x{first + 1}"
                if ax_name == "yaxis" and p.equal:
                    d["scaleanchor"] = "x" + sfx
                    d["scaleratio"] = 1
                lay_kw[f"{ax_name}{sfx}"] = d
            if p.note:
                xa, ya = ("x" if k == 0 else f"x{k + 1}"), ("y" if k == 0 else f"y{k + 1}")
                fig.add_annotation(text=_strip(p.note), xref=f"{xa} domain",
                                   yref=f"{ya} domain", x=0.01, y=0.99, xanchor="left",
                                   yanchor="top", showarrow=False, align="left",
                                   font={"size": VIEW_FONT["note"] * 1.333, "color": ink2})
        if pl.title_lines:
            # add_annotation, never layout(annotations=...): that would drop the series'
            # own annotations (reference labels, text) added above.
            fig.add_annotation(text="<br>".join(pl.title_lines), xref="paper", yref="paper",
                               x=b.x0 / W, y=1 - b.y0 / H, xanchor="left", yanchor="bottom",
                               showarrow=False, align="left",
                               font={"size": VIEW_FONT["title"] * 1.333,
                                     "color": style.hexcolor("ink", theme)})
        if pl.tag is not None and pl.tag_text:
            # The panel letter: bold, fixed at the top-left corner of the panel cell.
            fig.add_annotation(text=f"<b>{pl.tag_text}</b>", xref="paper", yref="paper",
                               x=max(pl.tag.x0, 0.0) / W, y=1 - pl.tag.y0 / H, xanchor="left",
                               yanchor="top", showarrow=False, align="left",
                               font={"size": VIEW_FONT["tag"] * 1.333,
                                     "color": style.hexcolor("ink", theme)})
        if pl.mode_legend != "none":
            lay_kw[legend] = _legend_layout(pl, W, H)
    fig.update_layout(template=template(theme), height=height_px or int(H * PX_PER_IN),
                      # No extra margins: paper fractions are the layout's own fractions,
                      # so every band (tick labels, titles, legends) has its laid-out width.
                      margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0}, showlegend=True,
                      hovermode="x unified" if any(pl.panel.sharex for pl in lay.panels)
                      else "closest", **lay_kw)
    if fit:
        fig.update_layout(meta={"aspect": W / H if H > 0 else 2.0})
    return fig


def aspect_of(fig: go.Figure, default: float = 2.0) -> float:
    """Width / height recorded by ``to_plotly(fit=True)`` (else from the layout height)."""
    meta = fig.layout.meta
    if isinstance(meta, dict) and meta.get("aspect"):
        return float(meta["aspect"])
    return default


def _camera(view: tuple[float, float]) -> dict[str, Any]:
    az, el = np.radians(view[0]), np.radians(view[1])
    r = 1.9
    return {"eye": {"x": r * np.cos(el) * np.cos(az), "y": r * np.cos(el) * np.sin(az),
                    "z": r * np.sin(el)}, "projection": {"type": "orthographic"}}


def apply_zoom(spec: FigureSpec, relayout: dict[str, Any] | None) -> FigureSpec:
    """Copy of ``spec`` whose axis limits follow a Plotly relayoutData (zoom/pan).

    Handles both key forms (``xaxis2.range[0]``/``[1]`` and ``xaxis2.range`` as a
    list), autorange resets, log axes (Plotly reports log10 values) and reversed
    ranges (heatmaps), which are re-ordered.
    """
    if not relayout:
        return spec
    out = copy.deepcopy(spec)
    for k, p in enumerate(out.panels):
        sfx = "" if k == 0 else str(k + 1)
        for ax, a in (("xaxis", p.x), ("yaxis", p.y)):
            if relayout.get(f"{ax}{sfx}.autorange"):
                a.lim = None
                continue
            rng = relayout.get(f"{ax}{sfx}.range")
            lo = relayout.get(f"{ax}{sfx}.range[0]", rng[0] if rng else None)
            hi = relayout.get(f"{ax}{sfx}.range[1]", rng[1] if rng else None)
            if lo is None or hi is None:
                continue
            lo, hi = float(lo), float(hi)
            if a.log:
                lo, hi = 10 ** lo, 10 ** hi
            a.lim = (min(lo, hi), max(lo, hi))
    return out


def merge_relayout(current: dict[str, Any] | None, new: dict[str, Any] | None
                   ) -> dict[str, Any]:
    """Accumulate relayout events: a new range clears that axis' autorange and vice versa."""
    merged = dict(current or {})
    for key, val in (new or {}).items():
        axis = key.split(".")[0]
        if key.endswith(".autorange"):
            for k in [k for k in merged if k.startswith(axis + ".range")]:
                merged.pop(k)
        elif ".range" in key:
            merged.pop(f"{axis}.autorange", None)
        merged[key] = val
    return merged
