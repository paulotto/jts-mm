# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Render specs with matplotlib: PNG/PDF exports, layout tests and video frames.

Axes are placed exactly at the boxes of :func:`biomech.viz.layout.compute`, with
its limits, ticks and tick labels, so the matplotlib output has the same
geometry as the TikZ export. :func:`build` returns the live figure and axes
(the animator draws its moving layers on top of them).

No pyplot: figures are plain :class:`matplotlib.figure.Figure` objects with their own Agg
canvas, so concurrent renders (viewer threads, video workers) share no global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from biomech.viz import layout as L
from biomech.viz import style
from biomech.viz.spec import KINDS_3D, FigureSpec, Series

_LS = {"solid": "-", "dashed": (0, (4, 2.5)), "dotted": (0, (1, 1.6))}
_MK = {"o": "o", "s": "s", "^": "^", "d": "D", "v": "v", "x": "x", "|": "|", "oo": "o",
       "none": "o"}                                          # "oo": open circle
GLYPH = 1.16          # DejaVu Sans is wider than the layout's Computer Modern estimate
_LOC = {"north east": "upper right", "north west": "upper left", "south east": "lower right",
        "south west": "lower left", "outer north east": "upper left"}


@dataclass
class Built:
    fig: Any
    axes: list[Any]
    layout: L.FigureLayout
    theme: str


def _cmap(ramp: str = "sequential"):
    from matplotlib.colors import LinearSegmentedColormap

    vals = style.SEQUENTIAL if ramp == "sequential" else style.DIVERGING
    return LinearSegmentedColormap.from_list(f"jt-{ramp}", ["#" + v for v in vals])


def _c(name: str, theme: str) -> str:
    return style.hexcolor(name, theme)


def _draw(ax, s: Series, pl: L.PanelLayout, theme: str, scale: float,
          F: dict[str, float] | None = None) -> list[Any]:
    """Draw one series; returns legend handles for it (possibly empty)."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Patch, Polygon

    col, dash, mark = style.resolve(s)
    c = _c(col, theme)
    lw = style.LINE_PT * scale * float(s.style.get("width", 1.0))
    if s.role in ("context", "muted", "faint"):
        lw = style.THIN_PT * scale * float(s.style.get("width", 1.0))
    ms = 2 * style.marker_pt() * scale * (style.marker_factor(s) if s.kind == "scatter"
                                           else float(s.style.get("size", 1.0)))
    alpha = float(s.style.get("alpha", 1.0))
    ink2 = _c("ink2", theme)
    surface = _c("surface", theme)
    fs = L.plain_text
    k = s.kind
    handles: list[Any] = []
    if k == "line":
        (h,) = ax.plot(s.x, s.y, linestyle=_LS[dash], lw=lw, color=c, alpha=alpha,
                       solid_capstyle="round", zorder=3 if s.role not in ("context", "faint") else 2)
        handles = [h]
    elif k == "scatter":
        m = s.style.get("ms", mark)
        if s.c is not None:
            h = ax.scatter(s.x, s.y, c=s.c, cmap=_cmap(), s=ms ** 2 * 0.6, marker=_MK.get(m, "o"),
                           edgecolors=surface, linewidths=0.4 * scale, zorder=4)
        else:
            line_mark = m in ("|", "x", "oo")   # unfilled markers are drawn by their edge
            (h,) = ax.plot(s.x, s.y, linestyle="none", marker=_MK.get(m, "o"), ms=ms, color=c,
                           alpha=alpha, markeredgecolor=c if line_mark else surface,
                           markerfacecolor="none" if m == "oo" else c,
                           markeredgewidth=(1.2 if line_mark
                                            else (style.halo_pt(s) or 0.4)) * scale,
                           zorder=4)
        handles = [h]
    elif k == "band":
        ax.fill_between(s.x, s.y, s.y2, color=c, alpha=style.BAND_ALPHA * (alpha if alpha < 1
                                                                          else 1), lw=0, zorder=1)
        handles = [Patch(facecolor=c, alpha=style.BAND_ALPHA * 1.4, lw=0)]
    elif k == "fill":
        ax.add_patch(Polygon(np.column_stack([s.x, s.y]), closed=True, facecolor=c,
                             alpha=float(s.style.get("alpha", style.BAND_ALPHA)),
                             edgecolor=c if s.style.get("edge", True) else "none",
                             lw=style.THIN_PT * scale, zorder=1))
        handles = [Patch(facecolor=c, alpha=0.3, edgecolor=c, lw=0.5)]
    elif k == "errorbar":
        y = np.asarray(s.y, float)
        h = ax.errorbar(s.x, y, yerr=[y - np.asarray(s.ylo, float), np.asarray(s.y2, float) - y],
                        fmt=_MK.get(mark, "o"), ms=ms, capsize=0, elinewidth=lw, color=c,
                        markeredgecolor=surface, markeredgewidth=0.4 * scale, zorder=4)
        handles = [h]
    elif k == "bar":
        ax.bar(s.x, s.y, width=float(s.style.get("bar_width", 0.62)), color=c, alpha=0.85,
               lw=0, zorder=2)
        handles = [Patch(facecolor=c)]
    elif k == "barh":
        ax.barh(s.y, s.x, height=float(s.style.get("bar_width", 0.62)), color=c, alpha=0.85,
                lw=0, zorder=2)
        handles = [Patch(facecolor=c)]
    elif k == "hline":
        for i, v in enumerate(np.atleast_1d(s.y)):
            ax.axhline(v, color=c, ls=_LS[dash], lw=style.THIN_PT * scale * 1.4, zorder=2)
            t = (s.text[i] if s.text and i < len(s.text) else (s.label if i == 0 else None))
            if t:
                left = s.style.get("anchor") == "left"     # label at the left end of the line
                ax.text(pl.x.lim[0] if left else pl.x.lim[1], v,
                        (" " + fs(t)) if left else (fs(t) + " "),
                        ha="left" if left else "right", va="bottom",
                        fontsize=pl_font(pl, "note", F) * scale, color=ink2, zorder=5)
    elif k == "vline":
        for i, v in enumerate(np.atleast_1d(s.x)):
            ax.axvline(v, color=c, ls=_LS[dash], lw=style.THIN_PT * scale * 1.4, zorder=2)
            t = (s.text[i] if s.text and i < len(s.text) else (s.label if i == 0 else None))
            if t:
                rev = pl.panel.y.reverse or any(q.kind == "heatmap" for q in pl.panel.series)
                ax.text(v, pl.y.lim[0] if rev else pl.y.lim[1], " " + fs(t), ha="left",
                        va="top", rotation=0, fontsize=pl_font(pl, "note", F) * scale, color=ink2,
                        zorder=5)
    elif k == "hspan":
        for a, b in zip(np.atleast_1d(s.x), np.atleast_1d(s.y2)):
            ax.axvspan(a, b, color=c, alpha=float(s.style.get("alpha", 0.12)), lw=0,
                       zorder=0)
    elif k == "sigbar":
        from matplotlib.lines import Line2D
        from matplotlib.transforms import blended_transform_factory

        tr = blended_transform_factory(ax.transData, ax.transAxes)
        yy = L.sigbar_y(int(s.style.get("row", 0)))
        bar_c = c if s.role not in ("context", "muted", "faint") else _c("ink2", theme)
        for a, b in zip(np.atleast_1d(s.x), np.atleast_1d(s.y2)):
            ax.plot([a, b], [yy, yy], transform=tr, color=bar_c, lw=2.6 * scale,
                    solid_capstyle="butt", zorder=5, clip_on=False)
        handles = [Line2D([], [], color=bar_c, lw=2.6 * scale)]
    elif k == "heatmap":
        z = np.asarray(s.z, float)
        cb = pl.panel.colorbar
        ramp = cb.ramp if cb is not None else "sequential"
        lim = heat_lim(z, cb)
        ax.imshow(z, cmap=_cmap(ramp), aspect="auto", vmin=lim[0], vmax=lim[1], origin="upper",
                  interpolation="nearest", zorder=1)
        if s.text:
            span = (lim[1] - lim[0]) or 1.0
            for (yy, xx), t in zip(np.ndindex(*z.shape), s.text):
                f = (z[yy, xx] - lim[0]) / span
                # A dark cell needs light text: the high end of the sequential ramp, both
                # poles of the diverging one.
                dark = np.isfinite(z[yy, xx]) and (f > 0.6 if ramp == "sequential"
                                                   else abs(f - 0.5) > 0.32)
                ax.text(xx, yy, fs(t), ha="center", va="center",
                        fontsize=pl_font(pl, "note", F) * scale,
                        color="#FFFFFF" if dark else "#1F2328", zorder=3)
    elif k == "text":
        anchor = s.style.get("anchor", "right")
        ha, va = {"right": ("left", "center"), "left": ("right", "center"),
                  "above": ("center", "bottom"), "below": ("center", "top")}[anchor]
        for xx, yy, t in zip(np.atleast_1d(s.x), np.atleast_1d(s.y), s.text or []):
            ax.text(xx, yy, fs(t), ha=ha, va=va, fontsize=pl_font(pl, "note", F) * scale,
                    color=_c(s.style.get("color", "ink2"), theme), zorder=6)
    elif k == "colorline":
        P = np.column_stack([s.x, s.y]).astype(float)
        segs = np.stack([P[:-1], P[1:]], axis=1)
        cc = np.asarray(s.c, float)
        cv = 0.5 * (cc[:-1] + cc[1:])
        lim = pl.panel.colorbar.lim if pl.panel.colorbar and pl.panel.colorbar.lim else (
            s.style.get("lim") or (np.nanmin(cc), np.nanmax(cc)))
        lc = LineCollection(segs, cmap=_cmap(), norm=_norm(lim), linewidths=lw * 1.3,
                            capstyle="round", zorder=3)
        lc.set_array(cv)
        ax.add_collection(lc)
    elif k == "arrow":
        d = np.asarray(s.d, float).reshape(-1, 2)
        for x0, y0, (dx, dy) in zip(np.atleast_1d(s.x), np.atleast_1d(s.y), d):
            ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                        arrowprops={"arrowstyle": "-|>", "color": c, "lw": lw,
                                    "shrinkA": 0, "shrinkB": 0,
                                    "mutation_scale": 6 * scale}, zorder=4)
        handles = [ax.plot([], [], color=c, lw=lw)[0]]
    elif k == "image":
        ex = s.extent or (0, s.img.shape[1], s.img.shape[0], 0)
        if np.ndim(s.img) == 2 and s.style.get("cmap"):
            ax.imshow(scalar_rgba(s, theme), extent=ex, aspect="auto",
                      interpolation="nearest", zorder=0)
        else:
            ax.imshow(s.img, extent=ex, aspect="auto", cmap="gray", zorder=0)
    elif k in KINDS_3D:
        z = np.asarray(s.z, float)
        if k == "line3d":
            (h,) = ax.plot(s.x, s.y, z, color=c, lw=lw, alpha=alpha, linestyle=_LS[dash])
        elif k == "surface3d":
            X, Y = np.meshgrid(np.asarray(s.x, float), np.asarray(s.y, float))
            C = np.asarray(s.c if s.c is not None else z, float)
            lo, hi = (np.nanmin(C), np.nanmax(C)) if np.isfinite(C).any() else (0.0, 1.0)
            cm = _cmap(str(s.style.get("ramp", "sequential")))
            h = ax.plot_surface(X, Y, z, facecolors=cm((C - lo) / ((hi - lo) or 1.0)),
                                linewidth=0.1 * scale, edgecolor=surface, alpha=alpha,
                                shade=False, antialiased=True)
        elif k == "scatter3d":
            h = ax.scatter(s.x, s.y, z, color=c, s=ms ** 2 * 0.6, depthshade=False,
                           marker=_MK.get(s.style.get("ms", mark), "o"), edgecolors=surface,
                           linewidths=0.3 * scale)
        else:
            d = np.asarray(s.d, float).reshape(-1, 3)
            h = ax.quiver(s.x, s.y, z, d[:, 0], d[:, 1], d[:, 2], color=c, lw=lw,
                          arrow_length_ratio=0.15)
        handles = [h]
    return handles if s.label else []


def heat_lim(z: np.ndarray, cb) -> tuple[float, float]:
    """Colour limits of a heatmap: the colourbar's, else the data range -- symmetric about 0
    on a diverging ramp (zero stays the neutral centre)."""
    if cb is not None and cb.lim:
        return cb.lim
    fin = np.isfinite(z)
    lo, hi = (float(np.nanmin(z)), float(np.nanmax(z))) if fin.any() else (0.0, 1.0)
    if cb is not None and cb.ramp == "diverging":
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    return lo, hi


def scalar_rgba(s: Series, theme: str = "light") -> np.ndarray:
    """RGBA float image of a colour-mapped scalar ``image`` series (``style["cmap"]``):
    NaN cells transparent, ``style["mask"]`` cells in the muted grey."""
    z = np.asarray(s.img, float)
    fin = z[np.isfinite(z)]
    lim = s.style.get("lim") or ((float(fin.min()), float(fin.max())) if fin.size else (0, 1))
    ramp = s.style.get("cmap") if s.style.get("cmap") in ("sequential", "diverging") \
        else "sequential"
    rgba = _cmap(ramp)(_norm(lim)(np.ma.masked_invalid(z)))
    rgba[~np.isfinite(z)] = (0.0, 0.0, 0.0, 0.0)
    if s.style.get("mask") is not None:
        h = _c("muted", theme).lstrip("#")
        rgba[np.asarray(s.style["mask"], bool)] = (*(int(h[i:i + 2], 16) / 255
                                                      for i in (0, 2, 4)), 1.0)
    return rgba


def _norm(lim):
    from matplotlib.colors import Normalize

    return Normalize(vmin=lim[0], vmax=lim[1])


def pl_font(pl: L.PanelLayout, key: str, F: dict[str, float] | None = None) -> float:
    """Unscaled font size (pt) of ``key``: the venue's (``F``), else the default venue's."""
    return (F or _FONT).get(key, 6.5)


_FONT: dict[str, float] = dict(style.FONT_PT)


def _style_axes(ax, pl: L.PanelLayout, theme: str, scale: float,
                F: dict[str, float] | None = None) -> None:
    _FONT = F or globals()["_FONT"]
    p = pl.panel
    ink, ink2, axis_c, grid = (_c(k, theme) for k in ("ink", "ink2", "axis", "grid"))
    ax.set_facecolor(_c("panel", theme))
    for a, setter in ((pl.x, ax.set_xscale), (pl.y, ax.set_yscale)):
        if a.log:
            setter("log")
    ax.set_xlim(pl.x.lim)
    heat = any(s.kind == "heatmap" for s in p.series)
    ax.set_ylim(pl.y.lim[::-1] if (p.y.reverse or heat) else pl.y.lim)   # heatmap row 0 on top
    ax.set_xticks(pl.x.ticks)
    if pl.x.rotate:
        ax.set_xticklabels(pl.x.labels, rotation=pl.x.rotate, ha="center", va="top")
    else:
        ax.set_xticklabels(pl.x.labels)
    ax.set_yticks(pl.y.ticks)
    ax.set_yticklabels(pl.y.labels)
    ax.minorticks_off()
    ax.tick_params(labelsize=_FONT["tick"] * scale, colors=ink2, length=L.TICK_LEN * 72 * scale,
                   width=style.THIN_PT * scale, pad=1.5 * scale)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_c)
        ax.spines[side].set_linewidth(style.THIN_PT * scale)
        if L.frameless(p):
            ax.spines[side].set_visible(False)
    if not any(s.kind in ("heatmap", "image") for s in p.series) and not L.frameless(p):
        ax.grid(True, color=grid, lw=style.THIN_PT * scale, zorder=0)
        ax.set_axisbelow(True)
    ax.set_xlabel(L.plain_text(pl.x.label), fontsize=_FONT["label"] * scale, color=ink,
                  labelpad=2 * scale)
    ax.set_ylabel(L.plain_text(pl.y.label), fontsize=_FONT["label"] * scale, color=ink,
                  labelpad=2 * scale)
    if p.note:
        ax.text(0.01, 0.99, L.plain_text(p.note), transform=ax.transAxes, ha="left", va="top",
                fontsize=_FONT["note"] * scale, color=ink2)


def _style_3d(ax, pl: L.PanelLayout, theme: str, scale: float,
              F: dict[str, float] | None = None) -> None:
    _FONT = F or globals()["_FONT"]
    p = pl.panel
    ink, ink2, grid = (_c(k, theme) for k in ("ink", "ink2", "grid"))
    ax.view_init(elev=p.view[1], azim=p.view[0])
    for a, lim, setter in (("x", pl.x.lim, ax.set_xlim), ("y", pl.y.lim, ax.set_ylim),
                           ("z", pl.z.lim, ax.set_zlim)):
        setter(lim)
    spans = [pl.x.lim[1] - pl.x.lim[0], pl.y.lim[1] - pl.y.lim[0], pl.z.lim[1] - pl.z.lim[0]]
    ax.set_box_aspect(spans)                       # mm for mm
    for axis, lab in ((ax.xaxis, pl.x.label), (ax.yaxis, pl.y.label), (ax.zaxis, pl.z.label)):
        axis.set_pane_color((0, 0, 0, 0))
        axis._axinfo["grid"].update(color=grid, linewidth=style.THIN_PT * scale)
        axis.label.set_size(_FONT["note"] * scale)
        axis.label.set_color(ink2)
    ax.set_xlabel(L.plain_text(pl.x.label), labelpad=-6 * scale)
    ax.set_ylabel(L.plain_text(pl.y.label), labelpad=-6 * scale)
    ax.set_zlabel(L.plain_text(pl.z.label), labelpad=-6 * scale)
    ax.tick_params(labelsize=_FONT["note"] * scale * 0.9, colors=ink2, pad=-2 * scale)
    for axis, spec_ax, lay_ax in ((ax.xaxis, p.x, pl.x), (ax.yaxis, p.y, pl.y),
                                  (ax.zaxis, p.z, pl.z)):
        if spec_ax.ticks is not None:                # explicit ticks (e.g. category names)
            axis.set_ticks(list(spec_ax.ticks))
            if spec_ax.ticklabels is not None:
                axis.set_ticklabels([L.plain_text(str(x)) for x in spec_ax.ticklabels])
        elif lay_ax is not None and lay_ax.ticks:   # the layout's ticks, as in the PGF
            axis.set_ticks(list(lay_ax.ticks))
    ax.set_facecolor((0, 0, 0, 0))


def _heading(fig, pl: L.PanelLayout, W: float, H: float, theme: str, scale: float,
             F: dict[str, float]) -> None:
    """Title lines and the bold panel tag at the layout's boxes, on the same baselines as the
    pgf export (one text artist per line)."""
    ink = _c("ink", theme)

    def base(top: float, pt: float, i: int = 0) -> float:
        return 1 - (top + L.line_height_in(pt) * i + 0.95 * pt / 72.0) / H

    if pl.title is not None and pl.title_lines:
        pt = F["title"] * scale
        for i, t in enumerate(pl.title_lines):
            fig.text(pl.title.x0 / W, base(pl.title.y0, pt, i), t, ha="left", va="baseline",
                     fontsize=pt, color=ink, gid="title")
    if pl.tag is not None and pl.tag_text:
        pt = F.get("tag", F["title"]) * scale
        fig.text(pl.tag.x0 / W, base(pl.tag.y0, pt), pl.tag_text, ha="left", va="baseline",
                 fontsize=pt, color=ink, fontweight="bold", gid="tag")


def build(spec: FigureSpec, *, theme: str = "light", width_in: float | None = None,
          scale: float = 1.0, lay: L.FigureLayout | None = None,
          venue: str | style.Venue | None = None) -> Built:
    """Figure and axes at the layout's boxes. ``scale`` multiplies fonts and marks
    (videos are drawn at a larger physical size than the paper); ``venue`` (default: the
    active one) gives the widths and font sizes."""
    import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

    F = dict(style.venue(venue).font_pt)
    F.setdefault("tag", F["title"])
    font = {k: v * scale for k, v in F.items()}
    lay = lay or L.compute(spec, width_in, font=font, glyph=GLYPH, venue=venue)
    W, H = lay.width, lay.height
    # No pyplot (global figure manager, rcParams): a plain Figure with its own Agg canvas.
    # Text uses DejaVu Sans, matplotlib's default family (the layout's GLYPH factor is
    # calibrated for it).
    fig = Figure(figsize=(W, H))
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(_c("surface", theme))
    axes = []
    for pl in lay.panels:
        b = pl.axis
        rect = [b.x0 / W, 1 - b.y1 / H, b.w / W, b.h / H]
        p = pl.panel
        if p.projection == "3d":
            ax = fig.add_axes(rect, projection="3d")
            ax.set_proj_type("ortho")
        else:
            ax = fig.add_axes(rect)
        handles, labels = [], []
        for s in p.series:
            hs = _draw(ax, s, pl, theme, scale, F)
            if hs and s.label:
                handles.append(hs[0])
                labels.append(L.plain_text(s.label))
        if p.projection == "3d":
            _style_3d(ax, pl, theme, scale, F)
        else:
            _style_axes(ax, pl, theme, scale, F)
        _heading(fig, pl, W, H, theme, scale, F)
        if pl.mode_legend == "below" and handles and pl.legend is not None:
            lb = pl.legend
            fig.legend(handles, labels, loc="upper left",
                       bbox_to_anchor=(lb.x0 / W, 1 - lb.y0 / H), ncol=max(pl.legend_cols, 1),
                       frameon=False, fontsize=F["legend"] * scale, handlelength=1.5,
                       handletextpad=0.4, columnspacing=1.0, borderaxespad=0, borderpad=0,
                       labelcolor=_c("ink", theme))
        elif pl.mode_legend.startswith("inside:") and handles:
            ax.legend(handles, labels, loc=_LOC.get(pl.mode_legend[7:], "best"), frameon=False,
                      fontsize=F["legend"] * scale, labelcolor=_c("ink", theme))
        if pl.colorbar is not None and p.colorbar is not None:
            cb = pl.colorbar
            cax = fig.add_axes([cb.x0 / W, 1 - cb.y1 / H, cb.w / W, cb.h / H])
            vals = [s.c for s in p.series if s.c is not None] + [s.z for s in p.series
                                                                 if s.kind == "heatmap"]
            vals += [s.img for s in p.series if s.kind == "image" and s.style.get("cmap")]
            lim = heat_lim(np.concatenate([np.ravel(np.asarray(v, float)) for v in vals])
                           if vals else np.array([0.0, 1.0]), p.colorbar)
            cbar = Colorbar(cax, cmap=_cmap(p.colorbar.ramp), norm=_norm(lim))
            cbar.outline.set_visible(False)
            if p.colorbar.ticks is not None:
                cbar.set_ticks(p.colorbar.ticks)
            cbar.ax.tick_params(labelsize=F["tick"] * scale, colors=_c("ink2", theme),
                                length=2 * scale, width=style.THIN_PT * scale)
            cbar.set_label(L.plain_text(p.colorbar.text()), fontsize=F["label"] * scale,
                           color=_c("ink", theme))
        axes.append(ax)
    _keep_xlabels_inside(fig)
    return Built(fig, axes, lay, theme)


def _keep_xlabels_inside(fig: Any) -> None:
    """An x label wider than its axes in a narrow column may run past the canvas edge (the
    PGF output wraps and checks its own labels): align it to the axes edge it would cross."""
    r = fig.canvas.get_renderer()
    W = fig.bbox.width
    for ax in fig.axes:
        lab = ax.xaxis.label
        if not lab.get_text() or not lab.get_visible():
            continue
        bb = lab.get_window_extent(r)
        if bb.x1 > W + 1:
            lab.set_horizontalalignment("right")
            lab.set_x(1.0)
        elif bb.x0 < -1:
            lab.set_horizontalalignment("left")
            lab.set_x(0.0)


def render_figure(spec: FigureSpec, path: str | Path, dpi: int = 300, theme: str = "light",
                  width_in: float | None = None, venue: str | style.Venue | None = None) -> Path:
    """Save as PNG or PDF (by suffix). Thread-safe (no pyplot)."""
    b = build(spec, theme=theme, width_in=width_in, venue=venue)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b.fig.savefig(path, dpi=dpi, facecolor=b.fig.get_facecolor())
    return path


def text_overlaps(b: Built) -> list[str]:
    """Pairs of rendered text artists whose extents overlap (layout test helper)."""
    fig = b.fig
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for ax in b.axes:
        if hasattr(ax, "zaxis"):
            continue
        # Titles are left-aligned (loc="left"): matplotlib keeps those in _left_title.
        items += [("title", t) for t in (ax.title, getattr(ax, "_left_title", None),
                                         getattr(ax, "_right_title", None)) if t is not None]
        items += [("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
        items += [("text", t) for t in ax.texts]
        items += [("xtick", t) for t in ax.get_xticklabels()]
        items += [("ytick", t) for t in ax.get_yticklabels()]
    items += [(t.get_gid() or "figtext", t) for t in fig.texts]      # titles, panel tags
    items += [("legend", lg) for lg in fig.legends]
    boxes = []
    for name, art in items:
        if not art.get_visible() or (hasattr(art, "get_text") and not art.get_text()):
            continue
        bb = art.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            boxes.append((name, art, bb))
    out = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, aa, a), (nb, ab, bb) = boxes[i], boxes[j]
            if na == nb and na in ("xtick", "ytick") and aa.axes is ab.axes:
                if a.overlaps(bb) and a.intersection(a, bb).width > 1 and a.intersection(
                        a, bb).height > 1:
                    out.append(f"{na} '{aa.get_text()}' overlaps '{ab.get_text()}'")
                continue
            if a.overlaps(bb):
                inter = a.intersection(a, bb)
                if inter is not None and inter.width > 1.0 and inter.height > 1.0:
                    ta = getattr(aa, "get_text", lambda: na)()
                    tb = getattr(ab, "get_text", lambda: nb)()
                    out.append(f"{na} {ta!r} overlaps {nb} {tb!r}")
    # Nothing may leave the canvas.
    W, H = fig.bbox.width, fig.bbox.height
    for name, art, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            txt = getattr(art, "get_text", lambda: name)()
            out.append(f"{name} {txt!r} outside the figure")
    # Legends must also stay off every axis box.
    for lg in fig.legends:
        lb = lg.get_window_extent(r)
        for ax in b.axes:
            ab = ax.get_window_extent(r)
            inter = lb.intersection(lb, ab)
            if lb.overlaps(ab) and inter is not None and inter.width > 1 and inter.height > 1:
                out.append("legend overlaps an axis box")
    return out
