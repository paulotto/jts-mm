# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Interactive preview of an animation: Plotly frames + slider (an11 §6).

The preview uses the layout's compact single-spec form
(:func:`biomech.anim.layouts.preview`: full-window panels with a cursor instead
of scrolling ones), rendered with :func:`biomech.viz.render_plotly.to_plotly` --
the same template, tokens and layout engine as every viewer figure. The
dynamic layers contribute traces per frame from their backend-neutral
``series(i)``, so the preview shows the same numbers as the video. It is
downsampled (``fps``, ``max_frames``); the video is the full-quality output.

Viewer entry point::

    from biomech.anim import make
    from biomech.anim.render_plotly import preview_figure
    fig = preview_figure(make(ctx, "s01", "multimodal", "chewing"), theme="dark")
"""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np
import plotly.graph_objects as go

from biomech.anim.layouts import Animation, preview
from biomech.anim.source import PHASE_LABEL, Clip
from biomech.viz import layout as L
from biomech.viz import render_plotly as rp
from biomech.viz import style
from biomech.viz.spec import Series


def _data_uri(img: np.ndarray) -> str:
    from PIL import Image as PILImage

    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(a).save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _traces(s: Series, k: int, pl: L.PanelLayout, theme: str) -> list[Any]:
    """Plotly traces of one dynamic series on panel ``k`` (constant count per kind)."""
    xa, ya = ("x" if k == 0 else f"x{k + 1}"), ("y" if k == 0 else f"y{k + 1}")
    if s.kind == "vline":                       # a cursor must be a trace to animate
        x = float(np.atleast_1d(s.x)[0])
        return [go.Scatter(x=[x, x], y=list(pl.y.lim), mode="lines", showlegend=False,
                           hoverinfo="skip", xaxis=xa, yaxis=ya,
                           line={"color": style.hexcolor("ink", theme), "width": 1.6})]
    if s.kind == "image":
        img = s.img
        ex = s.extent or (0, img.shape[1], img.shape[0], 0)
        return [go.Image(source=_data_uri(img), x0=ex[0], dx=(ex[1] - ex[0]) / img.shape[1],
                         y0=ex[3], dy=(ex[2] - ex[3]) / img.shape[0], xaxis=xa, yaxis=ya,
                         hoverinfo="skip")]
    if s.kind == "colorline":
        # One trace per colour bin, empty ones included: the trace count is fixed whatever
        # the frame's colours are (a constant first frame must not change it).
        cb = pl.panel.colorbar
        lim = rp.colorline_lim(s, pl.panel)
        return rp.colorline_traces(s, xa, ya, "legend", theme, lim, all_bins=True,
                                   ramp=cb.ramp if cb else "sequential",
                                   width=2.0 * float(s.style.get("width", 1.0)) * 0.75)
    scratch = go.Figure()
    if pl.panel.projection == "3d":
        rp._add3d(scratch, s, "scene" if k == 0 else f"scene{k + 1}", theme, "legend")
    else:
        rp._add(scratch, s, k, pl, theme, "legend")
    out = list(scratch.data)
    for tr in out:
        tr.showlegend = False
        if hasattr(tr, "hoverinfo"):
            tr.hoverinfo = "skip"
    return out


def _nan_list(a) -> list:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(a, float).ravel()]


def _frame_data(s: Series, pl: L.PanelLayout) -> list[dict[str, Any]]:
    """Only the changing properties of :func:`_traces`' traces (same count and order);
    frames carry these plain dicts, which keeps the preview fast and small."""
    k = s.kind
    if k == "vline":
        x = float(np.atleast_1d(s.x)[0])
        return [{"x": [x, x]}]
    if k == "image":
        return [{"source": _data_uri(s.img)}]
    if k in ("line", "scatter"):
        return [{"x": _nan_list(s.x), "y": _nan_list(s.y)}]
    if k == "colorline":
        # The same bins as _traces (a segment without a colour value is not drawn).
        lim = rp.colorline_lim(s, pl.panel)
        return [{"x": _nan_list(xs), "y": _nan_list(ys), "customdata": _nan_list(cs)}
                for _, xs, ys, cs in rp.colorline_bins(s.x, s.y, s.c, lim, all_bins=True)]
    if k in ("line3d", "scatter3d"):
        return [{"x": _nan_list(s.x), "y": _nan_list(s.y), "z": _nan_list(s.z)}]
    if k == "surface3d":
        return [{"x": _nan_list(s.x), "y": _nan_list(s.y),
                 "z": [_nan_list(r) for r in np.asarray(s.z, float)]}]
    if k == "arrow3d":
        d = np.asarray(s.d, float).reshape(-1, 3)
        x0, y0, z0 = (np.atleast_1d(np.asarray(v, float)) for v in (s.x, s.y, s.z))
        xs, ys, zs = [], [], []
        for i in range(len(d)):
            xs += [x0[i], x0[i] + d[i, 0], np.nan]
            ys += [y0[i], y0[i] + d[i, 1], np.nan]
            zs += [z0[i], z0[i] + d[i, 2], np.nan]
        return [{"x": _nan_list(xs), "y": _nan_list(ys), "z": _nan_list(zs)}]
    if k in ("bar", "barh"):
        return [{"x": _nan_list(s.x), "y": _nan_list(s.y)}]
    raise ValueError(f"series kind {k!r} cannot be animated in the preview")


def preview_figure(anim: Animation, *, theme: str | None = None, fps: float = 10.0,
                   max_frames: int = 150, width_in: float = rp.VIEW_WIDTH_IN,
                   height_px: int | None = None) -> go.Figure:
    """Plotly figure with one frame per preview step, a play button and a time slider."""
    theme = theme or anim.theme
    spec, layers, _ = preview(anim)
    tl = anim.timeline
    n = int(min(max_frames, max(2, np.ceil(tl.duration_s * fps))))
    idx = np.unique(np.linspace(0, tl.n_frames - 1, n).round().astype(int))
    times = tl.frame_times(idx)
    segs = tl.frame_segments(idx)
    for ly in layers:
        ly.prepare(times, segs)
    fig = rp.to_plotly(spec, theme=theme, width_in=width_in, height_px=height_px)
    lay = L.compute(spec, width_in, font=rp.VIEW_FONT)
    clip = anim.clip
    secs = tl.clip_seconds(times, segs) if isinstance(clip, Clip) else (times * 1e-6) % 1.0

    def title(i: int) -> str:
        if isinstance(clip, Clip):
            ph = PHASE_LABEL.get(tl.label(int(segs[i])), tl.label(int(segs[i])))
            return f"{clip.label} · {ph} · t = {secs[i]:.2f} s"
        return f"All participants · {clip.task} · cycle {secs[i] * 100:.0f} %"

    first = []
    for ly in layers:
        k = int(ly.panel.split(":")[1])
        for s in ly.series(0):
            first += _traces(s, k, lay.panels[k], theme)
    base = len(fig.data)
    fig.add_traces(first)
    ids = list(range(base, base + len(first)))
    frames = []
    for j in range(len(idx)):
        data = []
        for ly in layers:
            k = int(ly.panel.split(":")[1])
            for s in ly.series(j):
                data += _frame_data(s, lay.panels[k])
        if len(data) != len(ids):
            raise RuntimeError("a layer changed its number of series between frames")
        for d, tr in zip(data, first):
            d["type"] = tr.type
        frames.append({"data": data, "traces": ids, "name": str(j),
                       "layout": {"title": {"text": title(j)}}})
    fig.frames = frames
    ink2 = style.hexcolor("ink2", theme)
    dur = int(1000 / fps)
    every = max(1, len(idx) // 10)              # ~10 labelled slider ticks
    play = {"frame": {"duration": dur, "redraw": True}, "fromcurrent": True,
            "transition": {"duration": 0}, "mode": "immediate"}
    fig.update_layout(
        title={"text": title(0), "x": 0.01, "xanchor": "left", "y": 0.995, "yanchor": "top",
               "font": {"size": 15}},
        margin={"t": 40, "b": 70},
        updatemenus=[{"type": "buttons", "direction": "left", "x": 0.0, "y": -0.02,
                      "xanchor": "left", "yanchor": "top", "showactive": False,
                      "pad": {"r": 6, "t": 0},
                      "buttons": [{"label": "▶", "method": "animate", "args": [None, play]},
                                  {"label": "❚❚", "method": "animate",
                                   "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                                     "mode": "immediate"}]}]}],
        sliders=[{"active": 0, "x": 0.08, "y": -0.02, "len": 0.92, "xanchor": "left",
                  "yanchor": "top", "pad": {"t": 0}, "currentvalue": {"visible": False},
                  "font": {"color": ink2, "size": 10}, "tickcolor": ink2,
                  "steps": [{"label": ("" if j % every else f"{secs[j]:.1f}" if isinstance(
                             clip, Clip) else f"{secs[j] * 100:.0f}%"), "method": "animate",
                             "args": [[str(j)], {"frame": {"duration": 0, "redraw": True},
                                                 "mode": "immediate"}]}
                            for j in range(len(idx))]}])
    fig.update_layout(meta={"anim": anim.manifest(), "frame_times_us": times.tolist()})
    return fig
