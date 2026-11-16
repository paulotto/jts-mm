# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Animations for LaTeX (an11 §7).

``filmstrip``
    Key frames as small multiples in one vector pgfplots figure: the layout's
    film panel (sagittal plane for ``multimodal``/``kinematics``, the optical
    scene for ``tracking``, ...) repeated at ``n`` instants, each with the
    dynamic layers' state at that instant (trail up to ``t``, mandible outline,
    rotation centre, cursor). With camera previews in the clip (``with_camera``),
    a second row shows the camera frame held at each instant (PNG next to the
    data). Rendered by :mod:`biomech.viz.render_pgf`, so it uses the paper's
    style file, fonts and layout engine.
``animate_frames``
    PNG frames for the ``animate`` package (beamer, PDF supplements) and the
    ``\\animategraphics`` snippet that plays them.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from biomech.anim.layers import Image
from biomech.anim.layouts import Animation, preview
from biomech.anim.source import Clip
from biomech.viz import render_pgf
from biomech.viz.spec import Axis, FigureSpec, Panel, Series


def key_frames(anim: Animation, n: int = 6, times_s: list[float] | None = None) -> np.ndarray:
    """Frame indices of the key frames: ``n`` evenly spaced, or at video times ``times_s``."""
    tl = anim.timeline
    if times_s:
        return np.clip(np.round(np.asarray(times_s) * tl.fps).astype(int), 0, tl.n_frames - 1)
    return np.unique(np.linspace(0, tl.n_frames - 1, n + 2)[1:-1].round().astype(int))


def filmstrip_spec(anim: Animation, n: int = 6, times_s: list[float] | None = None,
                   fig_id: str | None = None) -> FigureSpec:
    """The filmstrip as a FigureSpec (render it with any backend)."""
    spec, layers, film = preview(anim)
    tl = anim.timeline
    idx = key_frames(anim, n, times_s)
    times = tl.frame_times(idx)
    segs = tl.frame_segments(idx)
    film_layers = [ly for ly in layers if ly.panel == f"preview:{film}"]
    for ly in film_layers:
        ly.prepare(times, segs)
    clip = anim.clip
    secs = tl.clip_seconds(times, segs) if isinstance(clip, Clip) else (times * 1e-6) % 1.0
    base = spec.panels[film]
    n_k = len(idx)
    cols = n_k if n_k <= 4 else int(np.ceil(n_k / 2))
    cam = isinstance(clip, Clip) and clip.with_camera and clip.previews
    im = None
    if cam:
        c0 = sorted(clip.previews)[0]
        im = Image("cam", clip.previews[c0]["t_us"], clip.previews[c0]["paths"])
        im.prepare(times, segs)
    panels: list[Panel] = []
    for g in range(0, n_k, cols):                  # row groups: key frames (+ camera row)
        group = list(range(g, min(g + cols, n_k)))
        for j in group:
            p = copy.deepcopy(base)
            p.title = (f"t = {secs[j]:.2f} s" if isinstance(clip, Clip)
                       else f"{secs[j] * 100:.0f} % of the cycle")
            p.legend = "none"
            if j != group[0]:                      # one y label per row
                p.y = Axis("", "", lim=p.y.lim, log=p.y.log, ticks=p.y.ticks,
                           ticklabels=p.y.ticklabels, reverse=p.y.reverse)
            for ly in film_layers:
                p.series += [x for x in ly.series(j) if _drawable(x)]
            panels.append(p)
        panels += [_blank()] * (cols - len(group))
        if im is not None:
            for j in group:
                img = im.series(j)[0].img
                h, w = img.shape[:2]
                p = Panel("", Axis("", "", lim=(0, w), ticks=[], ticklabels=[]),
                          Axis("", "", lim=(0, h), ticks=[], ticklabels=[]), equal=True,
                          legend="none")
                p.series.append(Series("image", img=img, extent=(0, w, 0, h)))
                panels.append(p)
            panels += [_blank()] * (cols - len(group))
    rows = len(panels) // cols
    name = clip.label if isinstance(clip, Clip) else "cohort"
    title = f"{name}: {', '.join(tl.labels)}"
    return FigureSpec(fig_id or f"film_{anim.layout}", title, panels, rows=rows,
                      cols=cols, width="text", height_in=1.6,
                      caption=f"Key frames of the {anim.layout} animation ({title}).",
                      meta={"equal_scale": "shared"})


def _drawable(s: Series) -> bool:
    """A snapshot series with at least one finite point (an inactive helical axis or rotation
    centre is all-NaN and would be an empty pgfplots plot)."""
    if s.kind in ("image", "vline", "hline", "text"):
        return True
    arrs = [a for a in (s.x, s.y, s.z) if a is not None]
    if s.d is not None:
        arrs.append(s.d)
    if not arrs:
        return False
    ok = np.ones(np.size(arrs[0]), bool) if s.d is None else np.ones(len(np.atleast_2d(s.d)),
                                                                        bool)
    for a in arrs:
        a = np.asarray(a, float)
        ok &= np.isfinite(a.reshape(len(ok), -1)).all(axis=1)
    return bool(ok.any())


def _blank() -> Panel:
    """An empty cell (an incomplete last row)."""
    return Panel("", Axis("", "", lim=(0, 1), ticks=[], ticklabels=[]),
                 Axis("", "", lim=(0, 1), ticks=[], ticklabels=[]), legend="none")


def filmstrip(anim: Animation, out_dir: str | Path, n: int = 6,
              times_s: list[float] | None = None, fig_id: str | None = None,
              standalone: bool = True) -> Path:
    """Write the filmstrip as TikZ/pgfplots under ``out_dir/figures/<id>/fig.tex`` (plus the
    style file and, with ``standalone``, a ``standalone_<id>.tex`` that compiles it)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = filmstrip_spec(anim, n, times_s, fig_id)
    render_pgf.write_style(out_dir)
    path = render_pgf.render_figure(spec, out_dir)
    if standalone:
        render_pgf.standalone_figure(out_dir, spec.id)
    return path


ANIMATE_PREAMBLE = "\\usepackage{animate}"


def animate_frames(anim: Animation, out_dir: str | Path, *, fps: float = 10.0,
                   size: str | int | tuple = "720p", theme: str = "light",
                   name: str = "frame") -> Path:
    """PNG frames at ``fps`` plus ``animate.tex`` (the ``\\animategraphics`` command; needs
    ``\\usepackage{animate}``). Returns the snippet path."""
    from PIL import Image as PILImage

    out_dir = Path(out_dir)
    fdir = out_dir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for old in fdir.glob(f"{name}-*.png"):
        old.unlink()
    a = anim.with_(fps=fps, theme=theme)
    sc = a.scene(size, stamp=False)
    try:
        sc.prepare()
        n = a.timeline.n_frames
        for i in range(n):
            PILImage.fromarray(sc.frame(i)).save(fdir / f"{name}-{i}.png", optimize=True)
    finally:
        sc.close()
    snippet = out_dir / "animate.tex"
    snippet.write_text(
        "% Generated by biomech.anim.render_latex -- needs \\usepackage{animate}.\n"
        f"% {anim.layout}: {', '.join(a.timeline.labels)}, {a.speed:g}x real time, "
        f"{n} frames at {fps:g} fps.\n"
        f"\\animategraphics[controls,loop,autoplay=false,width=\\linewidth]{{{fps:g}}}"
        f"{{frames/{name}-}}{{0}}{{{n - 1}}}\n", encoding="utf-8")
    return snippet


def animate_demo(out_dir: str | Path) -> Path:
    """A compilable document around ``animate.tex`` (compile in ``out_dir``)."""
    out_dir = Path(out_dir)
    p = out_dir / "animate_demo.tex"
    p.write_text("\\documentclass{article}\n" + ANIMATE_PREAMBLE + "\n\\begin{document}\n"
                 "\\input{animate.tex}\n\\end{document}\n", encoding="utf-8")
    return p
