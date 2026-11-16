# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Named animation layouts, the frame compositor (:class:`Scene`) and :class:`Animation`.

A video frame is a 16:9 canvas of *regions*. Each region is a static
:class:`~biomech.viz.spec.FigureSpec` rendered once with
:func:`biomech.viz.render_mpl.build` (same tokens, fonts, ticks and layout engine
as every static figure, scaled for video) -- or a plain header/key figure -- plus
the dynamic layers (:mod:`biomech.anim.layers`) drawn on its axes. Per frame the
:class:`Scene` restores each region's cached background, draws only the animated
artists, and pastes the regions into one RGB frame (no PNG round trip), then
stamps the hidden timecode (:mod:`biomech.anim.timecode`).

Layouts (``LAYOUTS``):

``multimodal``  3-D jaw, sagittal plane, scrolling EMG (closers, openers), optical vs
                IMU opening angle, intent-label (+ ML posterior) and tracking-quality
                ribbons, optional camera views.
``kinematics``  3-D jaw plus sagittal, frontal and horizontal planes and the opening
                angle with a cursor.
``tracking``    the optical scene (CRA/MTA/CT frame triads in the camera frame),
                tag counts, rigid-fit residual (mm) and quality, optional camera views.
``emg``         activation raster (% of max recorded) and envelopes with a sweeping cursor, instantaneous bars.
``cohort``      every participant's mean sagittal incisor path (small multiples),
                played cycle-normalized.
``v1_hinge``, ``v2_intent``, ``v3_release``
                the D5 supplementary videos (:mod:`biomech.anim.supplementary`, registered
                here on import): "not a hinge", intent cues in simulated real time, and
                open-close vs chewing release behaviour.

Missing modalities drop their region; the others re-flow into the space.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from biomech.anim import timecode
from biomech.anim.layers import (LABEL_COLORS, OUTLINE, QUALITY_COLORS, Bars, Chip, Cursor,
                                 Image, Layer, Pose, Ribbon, RibbonRow, Scope, Text, Trail,
                                 Triads, categorical_rgba, magnitude_ramp, mpl_cmap,
                                 ramp_rgb, resample_cols, rgb, ribbon_image, speed_ramp)
from biomech.anim.source import (BODY_LABEL, MUSCLE_LABEL, MUSCLES, PHASE_LABEL, Clip,
                                 CohortClip)
from biomech.anim.timeline import Timeline
from biomech.viz import layout as L
from biomech.viz import style
from biomech.viz.views import vocab as V
from biomech.viz.spec import Axis, FigureSpec, Panel, Series

DESIGN_IN = (12.8, 7.2)          # every frame is laid out on this canvas (16:9), then rasterized
SCALE = 1.4                      # fonts and marks relative to the paper (legible at 1080p)
SIZES = {"1080p": (1920, 1080), "720p": (1280, 720), "4k": (3840, 2160),
         "print": (2148, 1208)}  # print: 7.16 in (IEEE text width) at 300 dpi
HEADER_H = 0.075
KEY_H = 0.06
TRACK_VIEW = (-110.0, -35.0)     # from behind and below the camera rig (y points down)
# The IMU angle next to the optical one: the IMU hue of every figure (feature set "IMU"),
# dashed, so the two blues never read as one line in the dark theme.
IMU_ROLE = "anim:imu"
style.ROLES.setdefault(IMU_ROLE, ("sky", "dashed", "s"))
CLOSERS = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l")
OPENERS = ("digastric_r", "digastric_l")
CLASS_NAMES = ("REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT")
MM = "\\milli\\metre"


def frame_size(size: str | int | tuple[int, int]) -> tuple[int, int]:
    """Even pixel size of a preset (1080p, 720p, 4k, print), a width or 'WxH'."""
    if isinstance(size, tuple):
        w, h = size
    elif isinstance(size, int) or (isinstance(size, str) and size.isdigit()):
        w = int(size)
        h = int(round(w * DESIGN_IN[1] / DESIGN_IN[0]))
    elif isinstance(size, str) and "x" in size:
        w, h = (int(v) for v in size.lower().split("x"))
    elif size in SIZES:
        w, h = SIZES[size]
    else:
        raise ValueError(f"unknown size {size!r}; presets: {', '.join(SIZES)}, a width or WxH")
    return w + (w % 2), h + (h % 2)


# -- composition --------------------------------------------------------------------------------
@dataclass
class Region:
    name: str
    box: tuple[float, float, float, float]        # x0, y0 (from the top), w, h: frame fractions
    spec: FigureSpec | None = None
    draw: Callable[[Any, str, float], None] | None = None   # static plain figure (header, key)
    align: str | None = None                      # regions sharing a key get aligned axes
    zoom: float = 1.1                             # 3-D panels: mplot3d zoom (fills the box)
    post: Callable[[Any, list[Any]], None] | None = None   # (fig, axes) hook after build


@dataclass
class Composition:
    name: str
    regions: list[Region]
    layers: list[Layer]
    timeline: Timeline
    meta: dict[str, Any] = field(default_factory=dict)


def _flow(items: list[tuple[str, float]], x0: float, y0: float, w: float, h: float,
          gap: float = 0.012) -> dict[str, tuple[float, float, float, float]]:
    """Stack the present items vertically, heights proportional to their weights."""
    items = [(n, wt) for n, wt in items if wt > 0]
    if not items:
        return {}
    tot = sum(wt for _, wt in items)
    avail = h - gap * (len(items) - 1)
    out, y = {}, y0
    for n, wt in items:
        hh = avail * wt / tot
        out[n] = (x0, y, w, hh)
        y += hh + gap
    return out


def _nice(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    e = 10 ** np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * e >= v:
            return float(m * e)
    return float(10 * e)


def _in_segments(t: np.ndarray, clip: Clip) -> np.ndarray:
    sel = np.zeros(len(t), bool)
    for _, a, b in clip.segments:
        sel |= (t >= a) & (t <= b)
    return sel


def _lim(vals: list[np.ndarray], pad: float, min_span: float = 10.0) -> tuple[float, float]:
    v = np.concatenate([np.ravel(x) for x in vals if x is not None and np.size(x)])
    v = v[np.isfinite(v)]
    if not v.size:
        return (-min_span / 2, min_span / 2)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < min_span:
        c = 0.5 * (lo + hi)
        lo, hi = c - min_span / 2, c + min_span / 2
    return (lo - pad, hi + pad)


def _breaks(t: np.ndarray, y: np.ndarray, gap_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN rows where the time steps over a gap (phase cuts, tracking loss)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    cut = np.flatnonzero(np.diff(t) > gap_s * 1e6) + 1
    if not cut.size:
        return t, y
    t2 = np.insert(t, cut, np.nan)
    y2 = np.insert(y, cut, np.nan, axis=0)
    return t2, y2


# -- panels: kinematics ----------------------------------------------------------------------------
_AX = {0: ("x (anterior)", MM), 1: ("y (left)", MM), 2: ("z (superior)", MM)}
_PLANE = {(0, 2): "Sagittal plane", (1, 2): "Frontal plane", (0, 1): "Horizontal plane"}


def _speed_vmax(clip: Clip) -> float:
    k = clip.kin
    sel = _in_segments(k["t_us"], clip)
    sp = k["speed"][sel]
    sp = sp[np.isfinite(sp)]
    return _nice(np.percentile(sp, 97) if sp.size else 50.0)


def _kin_window(clip: Clip) -> dict[str, np.ndarray]:
    k = clip.kin
    sel = _in_segments(k["t_us"], clip)
    return {n: k[n][sel] for n in (*OUTLINE, "t_us")}


def _jaw_limits(clip: Clip) -> list[tuple[float, float]]:
    w = _kin_window(clip)
    pts = [w[n] for n in OUTLINE]
    if clip.maxilla:
        pts.append(np.array(list(clip.maxilla.values())))
    P = np.concatenate([np.asarray(p).reshape(-1, 3) for p in pts])
    return [_lim([P[:, d]], 6.0, 20.0) for d in range(3)]


def _maxilla_series(clip: Clip, dims: tuple[int, ...]) -> list[Series]:
    m = clip.maxilla or {}
    out = []
    arch = [m[k] for k in ("left", "incisor", "right") if k in m]
    if len(arch) >= 2:
        A = np.array(arch)
        if len(dims) == 3:
            out.append(Series("line3d", x=A[:, 0], y=A[:, 1], z=A[:, 2], role="muted",
                              style={"width": 1.6}))
        else:
            out.append(Series("line", x=A[:, dims[0]], y=A[:, dims[1]], role="muted",
                              style={"width": 1.6}))
    fossae = [m[k] for k in ("tmj_left", "tmj_right") if k in m]
    if fossae:
        F = np.array(fossae)
        if len(dims) == 3:
            out.append(Series("scatter3d", x=F[:, 0], y=F[:, 1], z=F[:, 2], role="muted",
                              style={"ms": "s", "size": 0.9}))
        else:
            out.append(Series("scatter", x=F[:, dims[0]], y=F[:, dims[1]], role="muted",
                              style={"ms": "s", "size": 0.9}))
    return out


def _kin_layers(clip: Clip, key: str, dims: tuple[int, ...], tau: float, vmax: float,
                fha: bool = True) -> list[Layer]:
    k = clip.kin
    t = k["t_us"]
    out: list[Layer] = [Pose(key, k, dims=dims, fha=fha, bounds=tuple(_jaw_limits(clip)))]
    for name, role in (("tmj_left", "entity:tmj_left"), ("tmj_right", "entity:tmj_right")):
        out.append(Trail(key, t, k[name], dims, tau, role, width=1.6, size=0.8))
    out.append(Trail(key, t, k["incisor"], dims, tau, "entity:incisor", speed=k["speed"],
                     vmax=vmax, width=2.6, size=1.1))
    return out


def jaw3d_panel(clip: Clip, key: str, tau: float, vmax: float, title: str = "Mandible in 3-D"
                ) -> tuple[Panel, list[Layer]]:
    w = _kin_window(clip)
    lims = _jaw_limits(clip)
    p = Panel(title, Axis("x", MM, lim=lims[0]), Axis("y", MM, lim=lims[1]), projection="3d",
              z=Axis("z", MM, lim=lims[2]), view=(-58.0, 18.0), legend="none")
    for n in ("tmj_left", "tmj_right", "incisor"):
        tt, P = _breaks(w["t_us"], w[n])
        p.series.append(Series("line3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], role="faint"))
    p.series += _maxilla_series(clip, (0, 1, 2))
    return p, _kin_layers(clip, key, (0, 1, 2), tau, vmax)


def plane_panel(clip: Clip, key: str, dims: tuple[int, int], tau: float, vmax: float
                ) -> tuple[Panel, list[Layer]]:
    w = _kin_window(clip)
    lims = _jaw_limits(clip)
    a, b = dims
    p = Panel(_PLANE[dims], Axis(*_AX[a], lim=lims[a]), Axis(*_AX[b], lim=lims[b]), equal=True,
              legend="none")
    for n in ("tmj_left", "tmj_right", "incisor"):
        tt, P = _breaks(w["t_us"], w[n])
        p.series.append(Series("line", x=P[:, a], y=P[:, b], role="faint"))
    p.series += _maxilla_series(clip, dims)
    return p, _kin_layers(clip, key, dims, tau, vmax)


# -- panels: time series ---------------------------------------------------------------------------
def _time_axis(window: tuple[float, float] | None, label: bool = True) -> Axis:
    if window is None:
        return Axis("Time" if label else "", "\\second" if label else "")
    return Axis("Time relative to now" if label else "", "\\second" if label else "",
                lim=window)


def _now_line() -> Series:
    return Series("vline", x=np.array([0.0]), role="reference")


def emg_panel(clip: Clip, key: str, group: tuple[str, ...], window, t_ref: float,
              xlabel: bool, title: str) -> tuple[Panel, list[Layer]] | None:
    e = clip.emg
    if e is None:
        return None
    idx = [e["names"].index(m) for m in group if m in e["names"]]
    if not idx:
        return None
    sel = _in_segments(e["t_us"], clip)
    y = e["pct"][:, idx]
    hi = np.nanpercentile(y[sel], 99.5) if np.isfinite(y[sel]).any() else 10.0
    roles = [f"entity:{e['names'][i]}" for i in idx]
    # The title carries the unit: a rotated y label does not fit the short panel.
    p = Panel(title, _time_axis(window, xlabel), Axis("", "",
                                                       lim=(0.0, _nice(max(hi * 1.08, 5.0)))),
              legend="below", sharex="t")
    for i, r in zip(idx, roles):
        p.series.append(Series("line", x=np.array([]), y=np.array([]), role=r,
                               label=MUSCLE_LABEL.get(e["names"][i], e["names"][i])))
    if window is not None:
        p.series.append(_now_line())
        return p, [Scope(key, e["t_us"], y, roles, window)]
    tt, yy = _breaks(e["t_us"][sel], y[sel])
    for c, r in enumerate(roles):
        p.series.append(Series("line", x=(tt - t_ref) * 1e-6, y=yy[:, c], role=r))
    return p, [Cursor(key, t_ref)]


def angle_panel(clip: Clip, key: str, window, t_ref: float, xlabel: bool
                ) -> tuple[Panel, list[Layer]] | None:
    k, imu = clip.kin, clip.imu
    tracks = []
    if k is not None and np.isfinite(k["theta_deg"]).any():
        tracks.append((k["t_us"], k["theta_deg"], "primary", "Optical (reference)", 3.0))
    if imu is not None:
        tracks.append((imu["t_us"], imu["theta_quat"], IMU_ROLE, "IMU", 1.3))
    if not tracks:
        return None
    vals = [y[_in_segments(t, clip)] for t, y, _, _, _ in tracks]
    lo, hi = _lim(vals, 1.5, 6.0)
    p = Panel("Opening angle (°)", _time_axis(window, xlabel), Axis("", "", lim=(lo, hi)),
              legend="below", sharex="t")
    for _, _, r, lab, wd in tracks:
        p.series.append(Series("line", x=np.array([]), y=np.array([]), role=r, label=lab,
                               style={"width": wd / 1.5}))
    layers: list[Layer] = []
    if window is not None:
        p.series.append(_now_line())
        for t, y, r, _, wd in tracks:
            layers.append(Scope(key, t, y, [r], window, width=wd))
        return p, layers
    for t, y, r, _, wd in tracks:
        sel = _in_segments(t, clip)
        tt, yy = _breaks(t[sel], y[sel])
        p.series.append(Series("line", x=(tt - t_ref) * 1e-6, y=yy, role=r,
                               style={"width": wd / 1.5}))
    return p, [Cursor(key, t_ref)]


def _rows_axis(names: list[str]) -> Axis:
    R = len(names)
    return Axis("", "", lim=(-0.5, R - 0.5), ticks=[float(R - 1 - r) for r in range(R)],
                ticklabels=names)


def label_rows(clip: Clip, theme: str) -> list[RibbonRow]:
    rows = []
    if clip.labels is not None:
        rows.append(RibbonRow("Label", clip.labels["t_us"], clip.labels["label"],
                              categorical_rgba(LABEL_COLORS, theme)))
    if clip.posterior is not None:
        # The stored decoder output is one decision per window (one-hot, not calibrated
        # probabilities): one row coloured by the decided class, like the label row.
        P = np.asarray(clip.posterior["p"], float)
        dec = np.where(P.max(axis=1) > 0, P.argmax(axis=1), np.nan)
        rows.append(RibbonRow("Decoded", clip.posterior["t_us"], dec,
                              categorical_rgba(LABEL_COLORS, theme)))
    return rows


def quality_rows(clip: Clip, theme: str) -> list[RibbonRow]:
    o = clip.optical
    rows = []
    if o is None:
        return rows
    for b in ("cra", "mta", "ct"):
        if b not in o["poses"]:
            continue
        ncam = o["n_cameras"].get(b)
        T = o["poses"][b]
        lock = np.isfinite(T.reshape(len(T), -1)).all(axis=1)
        v = np.where(lock, np.minimum(ncam if ncam is not None else 2.0, 2.0), 0.0)
        if b == "ct" and not lock[_in_segments(o["t_us"], clip)].any():
            continue
        name = V.term("body", b, "short")
        rows.append(RibbonRow(name[:1].upper() + name[1:], o["t_us"], v,
                              categorical_rgba(QUALITY_COLORS, theme)))
    return rows


_CAT_ROLE = {"grey": "muted", "blue": "cat:0", "orange": "cat:1", "green": "cat:2",
             "vermillion": "cat:3", "purple": "cat:4", "sky": "cat:5",
             "lightgrey": "faint", "yellow": "entity:temporalis_l", "ink2": "reference"}
_QUALITY_NAME = {2: "2 cameras", 1: "1 camera", 0: "lost"}
CLASS_ROWS = ("Label", "Decoded")          # ribbon rows coloured by intent class


def _ribbon_legend(rows: list[RibbonRow], clip: Clip) -> list[Series]:
    """Legend entries (solid swatches) for the categories that occur in the clip."""
    out = []
    for kind, names, colors in (("Label", [n.title() for n in CLASS_NAMES], LABEL_COLORS),
                                ("quality", _QUALITY_NAME, QUALITY_COLORS)):
        sel = [r for r in rows if (r.name in CLASS_ROWS) == (kind == "Label")]
        present: set[int] = set()
        for r in sel:
            v = np.asarray(r.values, float)[_in_segments(np.asarray(r.t_src, float), clip)]
            present |= {int(x) for x in np.unique(v[np.isfinite(v)]) if int(x) in colors}
        for k in sorted(present, reverse=kind != "Label"):
            out.append(Series("bar", x=np.array([]), y=np.array([]),
                              role=_CAT_ROLE[colors[k]], label=names[k]))
    return out


def ribbon_panel(rows: list[RibbonRow], key: str, title: str, window, t_ref: float,
                 xlabel: bool, theme: str, clip: Clip) -> tuple[Panel, list[Layer]] | None:
    if not rows:
        return None
    p = Panel(title, _time_axis(window, xlabel), _rows_axis([r.name for r in rows]),
              legend="below", sharex="t")
    p.series += _ribbon_legend(rows, clip)
    if window is not None:
        bg = (rgb("panel", theme) * 255).round().astype(np.uint8)
        p.series.append(Series("image", img=np.tile(bg, (len(rows), 2, 1)),
                               extent=(window[0], window[1], -0.5, len(rows) - 0.5)))
        p.series.append(_now_line())
        return p, [Ribbon(key, rows, window, theme=theme)]
    t0, t1 = clip.t0_us, clip.t1_us
    grid = np.linspace(t0, t1, 1200)
    img = ribbon_image(rows, grid, theme)
    p.series.append(Series("image", img=img, extent=((t0 - t_ref) * 1e-6, (t1 - t_ref) * 1e-6,
                                                     -0.5, len(rows) - 0.5)))
    return p, [Cursor(key, t_ref)]


# -- header and key ----------------------------------------------------------------------------
def _header(a: "Animation", tl: Timeline) -> tuple[Region, list[Layer]]:
    clip = a.clip
    scale = SCALE
    name = clip.label if isinstance(clip, Clip) else "All participants"
    fs_title = style.FONT_PT["title"] * scale * 1.5
    x_phase = 0.012 + (L.text_width_in(name, fs_title) * 1.08 + 0.25) / DESIGN_IN[0]
    phase_names = [PHASE_LABEL.get(s, s.replace("_", " ")) for s in tl.labels] or [""]
    longest = max(phase_names, key=len)
    x_chip = x_phase + (L.text_width_in(longest, fs_title) + 0.35) / DESIGN_IN[0]
    speed = f"{a.speed:g}× real time" if isinstance(clip, Clip) else f"{1 / a.speed:g} s per cycle"

    if isinstance(clip, Clip):
        def clock(s): return f"t = {s['t_s']:.2f} s"
    else:
        def clock(s): return f"cycle {s['t_s'] * 100:.0f} %"

    items = [(0.012, 0.58, "left", lambda s: name, "title", "ink"),
             (x_phase, 0.58, "left", lambda s: s["phase"], "title", "ink2"),
             (0.80, 0.58, "right", clock, "title", "ink"),
             (0.988, 0.58, "right", lambda s: speed, "label", "ink2")]
    text = Text("header", items, phase_names=phase_names,
                t_ref=lambda t, seg: tl.clip_seconds(t, seg) if isinstance(clip, Clip)
                else (np.asarray(t, float) * 1e-6) % 1.0,
                weight=("0", "2"))
    layers: list[Layer] = [text]
    if isinstance(clip, Clip) and clip.labels is not None:
        layers.append(Chip("header", clip.labels["t_us"], clip.labels["label"],
                           list(CLASS_NAMES), x=x_chip, y=0.58))
    layers.append(_Progress("header:progress", tl))

    def draw(fig, theme, sc):
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.08])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhspan(0, 1, color=style.hexcolor("grid", theme), lw=0)
        v0 = tl.video_starts_s / tl.duration_s
        for x in v0[1:]:
            ax.axvline(x, color=style.hexcolor("muted", theme), lw=sc)
        fig._anim_axes = {"progress": ax}

    return Region("header", (0.0, 0.0, 1.0, HEADER_H), draw=draw), layers


@dataclass
class _Progress(Layer):
    """Progress bar along the header's bottom edge (segments = phase cuts)."""

    panel: str
    tl: Timeline

    def prepare(self, t_us, seg):
        # From the frame's segment (not tl.to_video): segments may overlap in master time
        # (onset_rules plays overlapping cycle windows); identical otherwise.
        j = np.clip(np.asarray(seg, int), 0, len(self.tl.segments) - 1)
        t0 = np.array([a for a, _ in self.tl.segments])[j]
        v = self.tl.video_starts_s[j] + (np.asarray(t_us, float) - t0) / (self.tl.speed * 1e6)
        self.f = np.nan_to_num(v) / self.tl.duration_s
        self.n = len(self.f)

    def mpl_init(self, ax, theme, scale):
        (self._bar,) = ax.plot([0, 0], [0.5, 0.5], color=style.hexcolor("blue", theme), lw=1.0,
                               solid_capstyle="butt")
        self._bar.set_animated(True)
        self._ax = ax
        return [self._bar]

    def mpl_update(self, i):
        h = self._ax.get_window_extent().height
        self._bar.set_linewidth(h * 72 / self._ax.figure.dpi)
        self._bar.set_data([0, self.f[i]], [0.5, 0.5])


def _key(entries: list[tuple[str, str, str]], vmax: float | None, theme_hint: str = "",
         gap_in: float = 0.3, bar_in: float = 1.3) -> Callable[[Any, str, float], None]:
    """A static key strip: the speed ramp (if ``vmax``, ``bar_in`` wide) and glyph entries
    (kind, colour token or role, label) ``gap_in`` apart; kinds: dot, line, dash, dotted,
    cross, square and the event markers v, |, d."""

    def draw(fig, theme, sc):
        W, H = fig.get_size_inches()
        fs = style.FONT_PT["legend"] * sc * 1.1
        ink = style.hexcolor("ink", theme)
        x = 0.012
        if vmax is not None:
            lab = "Incisal speed"
            fig.text(x, 0.5, lab, ha="left", va="center", fontsize=fs, color=ink)
            x += (L.text_width_in(lab, fs) + 0.12) / W
            ink2 = style.hexcolor("ink2", theme)
            fig.text(x, 0.5, "0", ha="left", va="center", fontsize=fs, color=ink2)
            x += (L.text_width_in("0", fs) + 0.06) / W
            bw = bar_in / W
            ax = fig.add_axes([x, 0.3, bw, 0.4])
            ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto",
                      cmap=mpl_cmap(speed_ramp(theme)), extent=(0, vmax, 0, 1))
            ax.set_axis_off()
            x += bw + 0.06 / W
            lab = f"{vmax:g} mm/s"
            fig.text(x, 0.5, lab, ha="left", va="center", fontsize=fs, color=ink2)
            x += (L.text_width_in(lab, fs) + gap_in + 0.15) / W
        for kind, col, lab in entries:
            c = style.hexcolor(style.role(col)[0] if ":" in col or col in style.ROLES else col,
                               theme)
            gw = 0.32 / W
            ax = fig.add_axes([x, 0.2, gw, 0.6])
            ax.set_axis_off()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if kind == "dot":
                ax.plot([0.5], [0.5], "o", ms=style.MARKER_PT * sc * 3.4, color=c,
                        markeredgecolor=ink, markeredgewidth=0.6 * sc)
            elif kind == "line":
                ax.plot([0.05, 0.95], [0.5, 0.5], color=c, lw=style.LINE_PT * sc * 2.4)
            elif kind == "dash":
                ax.plot([0.05, 0.95], [0.5, 0.5], color=c, lw=style.LINE_PT * sc * 1.8,
                        ls=(0, (5, 2.5)))
            elif kind == "cross":
                ax.plot([0.5], [0.5], "x", ms=style.MARKER_PT * sc * 4, mew=style.LINE_PT * sc * 2,
                        color=c)
            elif kind == "square":
                ax.plot([0.5], [0.5], "s", ms=style.MARKER_PT * sc * 2.4, color=c)
            elif kind == "dotted":
                ax.plot([0.05, 0.95], [0.5, 0.5], color=c, lw=style.LINE_PT * sc * 1.8,
                        ls=(0, (1, 1.6)))
            elif kind in ("v", "|", "d"):             # event markers (D5)
                ax.plot([0.5], [0.5], kind, ms=style.MARKER_PT * sc * 4, color=c,
                        mew=style.LINE_PT * sc * (2.2 if kind == "|" else 0.6),
                        markeredgecolor=c if kind == "|" else ink)
            x += gw + 0.06 / W
            fig.text(x, 0.5, lab, ha="left", va="center", fontsize=fs, color=ink)
            x += (L.text_width_in(lab, fs) + gap_in) / W

    return draw


KIN_KEY = [("line", "ink2", "Mandible"), ("dot", "entity:tmj_left", "Condyle L"),
           ("dot", "entity:tmj_right", "Condyle R"), ("cross", "vermillion",
                                                       "Rotation centre"),
           ("dash", "vermillion", "Helical axis"), ("square", "muted", "Maxilla")]


# -- layouts ------------------------------------------------------------------------------------
def _spec(name: str, panels: list[Panel], rows: int, cols: int = 1, **meta) -> FigureSpec:
    return FigureSpec(name, "", panels, rows=rows, cols=cols, width="text", meta=meta)


def multimodal(a: "Animation", tl: Timeline) -> Composition:
    clip: Clip = a.clip
    th, win, tau = a.theme, a.window_s, a.trail_s
    head, layers = _header(a, tl)
    regions = [head]
    t_ref = clip.t0_us
    body_y, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024 - KEY_H
    has_kin = clip.kin is not None
    cams = sorted(clip.previews) if clip.with_camera and clip.previews else []
    right_x = 0.44 if has_kin else 0.0
    right = []
    # Right column: scrolling panels, aligned axes.
    cand = []
    em_c = emg_panel(clip, "emg:0", CLOSERS, win, t_ref, False, "Closer activation (% of max recorded)")
    em_o = emg_panel(clip, "emg:1", OPENERS, win, t_ref, False, "Opener activation (% of max recorded)")
    emg_panels = [x for x in (em_c, em_o) if x is not None]
    for k, (_, ls) in enumerate(emg_panels):
        for ly in ls:
            ly.panel = f"emg:{k}"
    if emg_panels:
        cand.append(("emg", 2.0 * len(emg_panels), emg_panels))
    ang = angle_panel(clip, "angle:0", win, t_ref, False)
    if ang:
        cand.append(("angle", 1.5, [ang]))
    lr = label_rows(clip, th)
    qr = quality_rows(clip, th)
    parts = [t for t, r in (("intent label", clip.labels is not None),
                            ("decoder decision", clip.posterior is not None),
                            ("optical tracking", bool(qr))) if r]
    title = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] \
        if parts else ""
    rib = ribbon_panel(lr + qr, "ribbon:0", title[:1].upper() + title[1:], win, t_ref, False,
                       th, clip)
    if rib:
        cand.append(("ribbon", 0.7 + 0.3 * len(lr + qr), [rib]))
    if cand:                        # the bottom panel carries the time-axis label
        last = cand[-1][2][-1][0]
        last.x = _time_axis(win, True)
    boxes = _flow([(n, w) for n, w, _ in cand], right_x + 0.01, body_y, 1 - right_x - 0.02,
                  body_h, gap=0.008)
    for n, _, pls in cand:
        sp = _spec(n, [p for p, _ in pls], rows=len(pls))
        regions.append(Region(n, boxes[n], sp, align="right"))
        for _, ls in pls:
            layers += ls
        right.append(n)
    # Left column: 3-D jaw on top, sagittal plane (and cameras) below.
    if has_kin:
        vmax = _speed_vmax(clip)
        lw = right_x
        p3, l3 = jaw3d_panel(clip, "jaw:0", tau, vmax)
        regions.append(Region("jaw", (0.0, body_y, lw, body_h * 0.56), _spec("jaw", [p3], 1),
                              zoom=1.25))
        layers += l3
        y2, h2 = body_y + body_h * 0.57, body_h * 0.43
        ps, lsag = plane_panel(clip, "planes:0", (0, 2), tau, vmax)
        layers += lsag
        if cams:                     # sagittal plane | camera views
            sag_w = lw * 0.6
            regions.append(Region("planes", (0.0, y2, sag_w, h2), _spec("planes", [ps], 1)))
            regions.append(_camera_region(clip, cams, (sag_w, y2, lw - sag_w, h2), layers))
        else:
            regions.append(Region("planes", (0.0, y2, lw, h2), _spec("planes", [ps], 1)))
        regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H),
                              draw=_key(KIN_KEY, vmax)))
    elif cams:
        regions.append(_camera_region(clip, cams, (0.0, body_y, 0.3, body_h), layers))
    if not right and not has_kin:
        raise ValueError(f"{clip.label}: no data to animate in {clip.segments}")
    return Composition("multimodal", regions, layers, tl)


def _recolor_colorbar(ramp: tuple[str, ...]) -> Callable[[Any, list[Any]], None]:
    """Give the region's colorbar the theme-oriented ramp its image uses."""

    def post(fig, axes):
        cmap = mpl_cmap(ramp)
        for ax in fig.axes:
            if ax in axes:
                continue
            for art in [*ax.collections, *ax.images]:
                if hasattr(art, "set_cmap"):
                    art.set_cmap(cmap)

    return post


def _camera_region(clip: Clip, cams: list[str], box, layers: list[Layer]) -> Region:
    pv = clip.previews
    panels = []
    for k, cam in enumerate(cams[:2]):
        from biomech.anim.layers import _Images

        h, w = _Images().get(pv[cam]["paths"][0]).shape[:2]
        p = Panel(f"Camera {k + 1}", Axis("", "", lim=(0, w), ticks=[], ticklabels=[]),
                  Axis("", "", lim=(0, h), ticks=[], ticklabels=[], reverse=True), equal=True,
                  legend="none")
        p.series.append(Series("image", img=np.zeros((2, 2, 3), np.uint8), extent=(0, w, h, 0)))
        panels.append(p)
        layers.append(Image(f"cam:{k}", pv[cam]["t_us"], pv[cam]["paths"]))
    return Region("cam", box, _spec("cam", panels, rows=len(panels)))


def kinematics(a: "Animation", tl: Timeline) -> Composition:
    clip: Clip = a.clip
    if clip.kin is None:
        raise ValueError(f"{clip.label}: no kinematics for {clip.segments}")
    head, layers = _header(a, tl)
    regions = [head]
    vmax = _speed_vmax(clip)
    body_y, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024 - KEY_H
    p3, l3 = jaw3d_panel(clip, "jaw:0", a.trail_s, vmax)
    regions.append(Region("jaw", (0.0, body_y, 0.44, body_h), _spec("jaw", [p3], 1), zoom=1.0))
    layers += l3
    panels = []
    for k, dims in enumerate(((0, 2), (1, 2), (0, 1))):
        p, ls = plane_panel(clip, f"planes:{k}", dims, a.trail_s, vmax)
        panels.append(p)
        layers += ls
    ang = angle_panel(clip, "planes:3", None, clip.t0_us, True)
    if ang:
        panels.append(ang[0])
        layers += ang[1]
    regions.append(Region("planes", (0.45, body_y, 0.54, body_h),
                          _spec("planes", panels, rows=2, cols=2, equal_scale="shared")))
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(KIN_KEY, vmax)))
    return Composition("kinematics", regions, layers, tl)


def tracking(a: "Animation", tl: Timeline) -> Composition:
    clip: Clip = a.clip
    o = clip.optical
    if o is None or not o["poses"]:
        raise ValueError(f"{clip.label}: no optical tracking in this session")
    th, win = a.theme, a.window_s
    head, layers = _header(a, tl)
    regions = [head]
    body_y, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024 - KEY_H
    sel = _in_segments(o["t_us"], clip)
    bodies = {b: T for b, T in o["poses"].items()
              if np.isfinite(T[sel].reshape(sel.sum(), -1)).any()}
    origins = [T[sel][:, :3, 3] for T in bodies.values()]
    lims = [_lim([P[:, d] for P in origins], 30.0, 60.0) for d in range(3)]
    p = Panel("Optical scene (camera frame)", Axis("x", MM, lim=lims[0]),
              Axis("y", MM, lim=lims[1]), projection="3d", z=Axis("z", MM, lim=lims[2]),
              view=TRACK_VIEW, legend="none")
    for T in bodies.values():
        _, P = _breaks(o["t_us"][sel], T[sel][:, :3, 3])
        p.series.append(Series("line3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], role="faint"))
    regions.append(Region("scene", (0.0, body_y, 0.48, body_h), _spec("scene", [p], 1),
                          zoom=0.95))
    layers.append(Triads("scene:0", o["t_us"], bodies, {b: b.upper() for b in bodies}))
    for b in bodies:
        layers.append(Trail("scene:0", o["t_us"], bodies[b][:, :3, 3], (0, 1, 2), a.trail_s,
                            "muted", head=False, width=1.4))
    cams = sorted(clip.previews) if clip.with_camera and clip.previews else []
    roles = {"cra": "cat:0", "mta": "cat:1", "ct": "cat:2"}
    cand = []
    nt = [(b, o["n_tags"][b]) for b in bodies if b in o["n_tags"]]
    if nt:
        hi = max(np.nanmax(v[sel]) for _, v in nt if np.isfinite(v[sel]).any()) \
            if any(np.isfinite(v[sel]).any() for _, v in nt) else 10
        pt = Panel("Tags detected", _time_axis(win, False), Axis("Tags", "", lim=(0, hi + 1)),
                   legend="below", sharex="t")
        pt.series += [Series("line", x=np.array([]), y=np.array([]), role=roles.get(b, "muted"),
                             label=BODY_LABEL.get(b, b)) for b, _ in nt]
        pt.series.append(_now_line())
        cand.append(("tags", 1.6, pt, [Scope("tags:0", o["t_us"], v, [roles.get(b, "muted")],
                                             win) for b, v in nt]))
    rm = [(b, o["rms"][b]) for b in bodies if b in o["rms"]]
    if rm:
        vals = [v[sel] for _, v in rm]
        hi = _nice(max(np.nanpercentile(v, 99) for v in vals if np.isfinite(v).any()) * 1.1) \
            if any(np.isfinite(v).any() for v in vals) else 1.0
        pr = Panel("Tracking fit residual", _time_axis(win, False), Axis("RMS", "\\milli\\metre",
                                                                      lim=(0, hi)),
                   legend="below", sharex="t")
        pr.series += [Series("line", x=np.array([]), y=np.array([]), role=roles.get(b, "muted"),
                             label=BODY_LABEL.get(b, b)) for b, _ in rm]
        pr.series.append(_now_line())
        cand.append(("rms", 1.6, pr, [Scope("rms:0", o["t_us"], v, [roles.get(b, "muted")], win)
                                      for b, v in rm]))
    rib = ribbon_panel(quality_rows(clip, th), "quality:0", "Cameras seeing the body", win,
                       clip.t0_us, False, th, clip)
    if rib:
        cand.append(("quality", 1.3, rib[0], rib[1]))
    if cand:
        cand[-1][2].x = _time_axis(win, True)
    x0, w = 0.5, 0.49
    y0, h = body_y, body_h
    if cams:
        regions.append(_camera_region_row(clip, cams, (x0, y0, w, body_h * 0.42), layers))
        y0, h = body_y + body_h * 0.44, body_h * 0.56
    boxes = _flow([(n, wt) for n, wt, _, _ in cand], x0, y0, w, h, gap=0.008)
    for n, _, pnl, ls in cand:
        regions.append(Region(n, boxes[n], _spec(n, [pnl], 1), align="right"))
        layers += ls
    return Composition("tracking", regions, layers, tl)


def _camera_region_row(clip: Clip, cams: list[str], box, layers: list[Layer]) -> Region:
    r = _camera_region(clip, cams, box, layers)
    r.spec.rows, r.spec.cols = 1, len(r.spec.panels)
    return r


def emg(a: "Animation", tl: Timeline) -> Composition:
    clip: Clip = a.clip
    e = clip.emg
    if e is None:
        raise ValueError(f"{clip.label}: no EMG for {clip.segments}")
    head, layers = _header(a, tl)
    regions = [head]
    body_y, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024
    t_ref = clip.t0_us
    names = [m for m in MUSCLES if m in e["names"]]
    idx = [e["names"].index(m) for m in names]
    sel = _in_segments(e["t_us"], clip)
    Y = e["pct"][:, idx]
    hi = _nice(max(np.nanpercentile(Y[sel], 99.5) * 1.05, 5.0)) if np.isfinite(Y[sel]).any() \
        else 100.0
    # Raster: muscles x time, sequential ramp (same as the static heatmaps).
    from biomech.viz.spec import Colorbar

    grid = np.linspace(clip.t0_us, clip.t1_us, 1400)
    Z = resample_cols(e["t_us"], Y, grid, max_gap_s=0.25).T                # (C, M)
    ramp = magnitude_ramp(a.theme)        # zero blends into the panel in both themes
    img = (ramp_rgb((Z / hi).ravel(), ramp).reshape(Z.shape + (3,)) * 255)
    img[~np.isfinite(Z)] = rgb("lightgrey", a.theme) * 255             # masked: neutral grey
    R = len(names)
    raster = Panel("Activation raster", _time_axis(None), _rows_axis(
        [MUSCLE_LABEL[m] for m in names]), legend="none", sharex="t",
        colorbar=Colorbar("Activation", "\\percent", lim=(0.0, hi)))
    raster.series.append(Series("image", img=img.round().astype(np.uint8),
                                extent=(0.0, (clip.t1_us - t_ref) * 1e-6, -0.5, R - 0.5)))
    layers.append(Cursor("raster:0", t_ref))
    env = []
    for k, (grp, title) in enumerate(((CLOSERS, "Closer activation (% of max recorded)"),
                                      (OPENERS, "Opener activation (% of max recorded)"))):
        r = emg_panel(clip, f"env:{k}", grp, None, t_ref, k == 1, title)
        if r:
            for ly in r[1]:
                ly.panel = f"env:{len(env)}"
            env.append(r)
    regions.append(Region("raster", (0.0, body_y, 0.68, body_h * 0.36),
                          _spec("raster", [raster], 1), align="left",
                          post=_recolor_colorbar(ramp)))
    if env:
        regions.append(Region("env", (0.0, body_y + body_h * 0.38, 0.68, body_h * 0.62),
                              _spec("env", [p for p, _ in env], rows=len(env)), align="left"))
        for _, ls in env:
            layers += ls
    bars = Panel("Activation now", Axis("Activation", "\\percent", lim=(0.0, hi)),
                 _rows_axis([MUSCLE_LABEL[m] for m in names]), legend="none")
    regions.append(Region("bars", (0.70, body_y, 0.29, body_h * 0.60), _spec("bars", [bars], 1)))
    layers.append(Bars("bars:0", e["t_us"], Y, [f"entity:{m}" for m in names]))
    return Composition("emg", regions, layers, tl)


def cohort(a: "Animation", tl: Timeline) -> Composition:
    c: CohortClip = a.clip
    if not len(c.means):
        raise ValueError(f"no {c.task} cycles in the cohort")
    head, layers = _header(a, tl)
    regions = [head]
    n = len(c.labels)
    allp = np.concatenate([t.reshape(-1, 3) for t in c.trials] + [c.means.reshape(-1, 3)])
    lx, lz = _lim([allp[:, 0]], 2.0, 10.0), _lim([allp[:, 2]], 2.0, 10.0)
    # The grid that gives the largest mm-for-mm panels on the 16:9 body.
    bw, bh = DESIGN_IN[0], DESIGN_IN[1] * (1 - HEADER_H - KEY_H)
    cols = max(range(1, n + 1), key=lambda k: min(bw / k / (lx[1] - lx[0] + 8),
                                                  bh / np.ceil(n / k) / (lz[1] - lz[0] + 8)))
    rows = int(np.ceil(n / cols))
    vmax = _nice(np.nanpercentile(c.speed, 97))
    panels = []
    grid_us = np.linspace(0, 1e6, c.means.shape[1])
    for k, lab in enumerate(c.labels):
        bottom, left = k + cols >= n, k % cols == 0
        p = Panel(lab, Axis("x (anterior)" if bottom else "", MM if bottom else "", lim=lx),
                  Axis("z (superior)" if left else "", MM if left else "", lim=lz), equal=True,
                  legend="none")
        for tr in c.trials[k]:
            p.series.append(Series("line", x=tr[:, 0], y=tr[:, 2], role="faint"))
        p.series.append(Series("line", x=c.means[k, :, 0], y=c.means[k, :, 2], role="context"))
        panels.append(p)
        layers.append(Trail(f"grid:{k}", grid_us, c.means[k], (0, 2), 0.3, "entity:incisor",
                            speed=c.speed[k], vmax=vmax, width=2.6))
    body_y, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024 - KEY_H
    regions.append(Region("grid", (0.0, body_y, 1.0, body_h),
                          _spec("grid", panels, rows=rows, cols=cols, equal_scale="shared")))
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H),
                          draw=_key([("line", "context", "Participant mean"),
                                     ("line", "faint", "Single cycles")], vmax)))
    layers = [ly for ly in layers if not isinstance(ly, Chip)]
    return Composition("cohort", regions, layers, tl)


LAYOUTS: dict[str, Callable[["Animation", Timeline], Composition]] = {
    "multimodal": multimodal, "kinematics": kinematics, "tracking": tracking, "emg": emg,
    "cohort": cohort}
# Per-layout hooks, filled by layouts that need them (biomech.anim.supplementary, D5):
EXTRAS: dict[str, tuple[str, ...]] = {}          # derived modalities make() loads (source.EXTRAS)
TIMELINES: dict[str, Callable[["Animation"], Timeline]] = {}   # a layout's own timeline
PREVIEWS: dict[str, Callable[["Animation"], tuple]] = {}       # its compact preview form
DEFAULT_PHASES: dict[str, tuple[str, ...]] = {}  # phases played when none are given
DEFAULT_SPEED: dict[str, float] = {}             # playback speed when none is given
BANNERS: dict[str, str] = {}                     # fixed burned-in banner (also in the manifest)
STILLS: set[str] = set()                         # layouts that get a PNG filmstrip next to the video
STILL_WIDTH: dict[str, int] = {}                 # px per filmstrip frame (default 640)
# Layouts that play all participants (a CohortClip, no --pid), and their loaders
# ``(ctx, task) -> CohortClip`` (default: :func:`biomech.anim.source.load_cohort_clip`).
COHORT_LAYOUTS: set[str] = {"cohort"}
COHORT_LOADERS: dict[str, Callable[[Any, str], CohortClip]] = {}


# -- preview (one spec: Plotly frames, LaTeX filmstrip) --------------------------------------------
def preview(a: "Animation") -> tuple[FigureSpec, list[Layer], int]:
    """A compact single-spec version of the layout (full-window panels with cursors
    instead of scrolling ones): ``(spec, layers, film_panel)``. Layers address
    ``preview:<k>``; ``film_panel`` is the panel the filmstrip repeats."""
    clip = a.clip
    tau = a.trail_s
    panels: list[Panel] = []
    layers: list[Layer] = []

    def add(res):
        if res is None:
            return
        p, ls = res
        k = len(panels)
        panels.append(p)
        for ly in ls:
            ly.panel = f"preview:{k}"
            layers.append(ly)

    film = 0
    if a.layout in PREVIEWS:
        return PREVIEWS[a.layout](a)
    if a.layout == "cohort":
        comp = cohort(a, a.timeline)
        grid = next(r for r in comp.regions if r.name == "grid")
        spec = copy.deepcopy(grid.spec)
        lys = [ly for ly in comp.layers if ly.panel.startswith("grid:")]
        for ly in lys:
            ly.panel = "preview:" + ly.panel.split(":")[1]
        return spec, lys, 0
    if a.layout == "tracking":
        comp = tracking(a, a.timeline)
        scene = next(r for r in comp.regions if r.name == "scene")
        add((copy.deepcopy(scene.spec.panels[0]),
             [ly for ly in comp.layers if ly.panel == "scene:0"]))
        add(ribbon_panel(quality_rows(clip, a.theme), "", "Cameras seeing the body", None,
                         clip.t0_us, True, a.theme, clip))
        return _spec("preview", panels, rows=1, cols=len(panels)), layers, 0
    t_ref = clip.t0_us
    if clip.kin is not None and a.layout in ("multimodal", "kinematics"):
        vmax = _speed_vmax(clip)
        add(jaw3d_panel(clip, "", tau, vmax))
        film = len(panels)
        add(plane_panel(clip, "", (0, 2), tau, vmax))
        if a.layout == "kinematics":
            add(plane_panel(clip, "", (1, 2), tau, vmax))
            add(plane_panel(clip, "", (0, 1), tau, vmax))
    if a.layout in ("multimodal", "emg"):
        add(emg_panel(clip, "", CLOSERS, None, t_ref, True, "Closer activation (% of max recorded)"))
        if a.layout == "emg":
            add(emg_panel(clip, "", OPENERS, None, t_ref, True, "Opener activation (% of max recorded)"))
    if a.layout in ("multimodal", "kinematics"):
        add(angle_panel(clip, "", None, t_ref, True))
    if a.layout == "multimodal":
        add(ribbon_panel(label_rows(clip, a.theme), "", "Intent label", None, t_ref, True,
                         a.theme, clip))
    if clip.with_camera and clip.previews and a.layout == "multimodal":
        tmp: list[Layer] = []
        reg = _camera_region(clip, sorted(clip.previews)[:1], (0, 0, 1, 1), tmp)
        add((reg.spec.panels[0], tmp))
    if not panels:
        raise ValueError(f"{clip.label}: nothing to preview for layout {a.layout!r}")
    cols = 2 if len(panels) > 1 else 1
    rows = int(np.ceil(len(panels) / cols))
    return _spec("preview", panels, rows=rows, cols=cols), layers, film


# -- the scene ------------------------------------------------------------------------------------
def _fit(spec: FigureSpec, w_in: float, h_in: float, font: dict[str, float]) -> L.FigureLayout:
    """Layout of ``spec`` at width ``w_in`` with the tallest rows that fit ``h_in``."""
    lo, hi = 0.2, h_in
    best = None
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        spec.height_in = mid / max(spec.rows, 1)
        lay = L.compute(spec, w_in, font=font)
        if lay.height <= h_in:
            best, lo = (mid, lay), mid
        else:
            hi = mid
    if best is None:
        spec.height_in = 0.2
        return L.compute(spec, w_in, font=font)
    spec.height_in = best[0] / max(spec.rows, 1)
    return best[1]


def _align(lays: list[L.FigureLayout]) -> None:
    """Give the axes of several stacked regions common left and right edges (time axes line
    up across the column)."""
    pls = [pl for lay in lays for pl in lay.panels if pl.panel.projection == "2d"]
    if not pls:
        return
    x0 = max(pl.axis.x0 for pl in pls)
    x1 = min(pl.axis.x1 for pl in pls)
    for pl in pls:
        pl.axis.w = x1 - x0
        pl.axis.x0 = x0
        if pl.title is not None:
            pl.title.x0, pl.title.w = x0, x1 - x0


class Scene:
    """Built regions plus prepared layers for a range of frames; :meth:`frame` returns RGB."""

    def __init__(self, comp: Composition, size=(1920, 1080), theme: str = "dark",
                 stamp: bool = True) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from biomech.viz import render_mpl

        self.comp = comp
        self.theme = theme
        self.W, self.H = frame_size(size)
        self.dpi = self.W / DESIGN_IN[0]
        self.stamp = stamp
        font = {k: v * SCALE for k, v in style.FONT_PT.items()}
        self.regions: list[dict[str, Any]] = []
        lays: dict[str, L.FigureLayout] = {}
        groups: dict[str, list[str]] = {}
        for r in comp.regions:
            if r.spec is not None:
                spec = copy.deepcopy(r.spec)
                r_spec = spec
                lays[r.name] = _fit(spec, r.box[2] * DESIGN_IN[0], r.box[3] * DESIGN_IN[1], font)
                r.spec = r_spec
                if r.align:
                    groups.setdefault(r.align, []).append(r.name)
        for names in groups.values():
            _align([lays[n] for n in names])
        for r in comp.regions:
            bw, bh = r.box[2] * DESIGN_IN[0], r.box[3] * DESIGN_IN[1]
            if r.spec is not None:
                lay = lays[r.name]
                b = render_mpl.build(r.spec, theme=theme, width_in=lay.width, scale=SCALE,
                                     lay=lay)
                fig, axes = b.fig, b.axes
                if r.post is not None:
                    r.post(fig, axes)
                for ax in axes:
                    if hasattr(ax, "zaxis"):          # fill the panel (mplot3d leaves a margin)
                        ax.set_box_aspect(ax.get_box_aspect(), zoom=r.zoom)
                        ax.xaxis.labelpad = ax.yaxis.labelpad = 1.0 * SCALE
                        ax.zaxis.labelpad = -3.0 * SCALE
                w_in, h_in = lay.width, lay.height
                dx = 0.0
                dy = 0.0 if r.align else max(0.0, (bh - h_in) / 2)
            else:
                fig = Figure(figsize=(bw, bh))
                FigureCanvasAgg(fig)
                fig.patch.set_facecolor(style.hexcolor("surface", theme))
                axes = []
                w_in, h_in, dx, dy = bw, bh, 0.0, 0.0
                if r.draw is not None:
                    r.draw(fig, theme, SCALE)
            pw, ph = int(round(w_in * self.dpi)), int(round(h_in * self.dpi))
            fig.set_dpi(self.dpi)
            fig.set_size_inches(pw / self.dpi, ph / self.dpi)
            x = int(round((r.box[0] * DESIGN_IN[0] + dx) * self.dpi))
            y = int(round((r.box[1] * DESIGN_IN[1] + dy) * self.dpi))
            self.regions.append({"name": r.name, "fig": fig, "axes": axes, "xy": (x, y),
                                 "size": (pw, ph), "bg": None, "artists": []})
        self._by = {d["name"]: d for d in self.regions}
        self.base = np.empty((self.H, self.W, 3), np.uint8)
        self.base[:] = (rgb("surface", theme) * 255).round().astype(np.uint8)
        self.times: np.ndarray | None = None

    # -- targets ----------------------------------------------------------------------------
    def target(self, key: str):
        name, _, idx = key.partition(":")
        d = self._by[name]
        if not idx:
            return d, d["fig"]
        if idx.isdigit():
            return d, d["axes"][int(idx)]
        return d, d["fig"]._anim_axes[idx]

    def prepare(self, frames: range | np.ndarray | None = None) -> None:
        tl = self.comp.timeline
        frames = range(tl.n_frames) if frames is None else frames
        self.frames = np.asarray(list(frames) if isinstance(frames, range) else frames, int)
        self.times = tl.frame_times(self.frames)
        segs = tl.frame_segments(self.frames)
        for ly in self.comp.layers:
            ly.prepare(self.times, segs)
        for ly in self.comp.layers:
            d, tgt = self.target(ly.panel)
            arts = ly.mpl_init(tgt, self.theme, SCALE)
            d["artists"].append((ly, arts))
        for d in self.regions:
            d["fig"].canvas.draw()
            d["bg"] = d["fig"].canvas.copy_from_bbox(d["fig"].bbox)

    def frame(self, i: int) -> np.ndarray:
        """RGB (H, W, 3) uint8 of the ``i``-th prepared frame."""
        out = self.base.copy()
        for d in self.regions:
            canvas = d["fig"].canvas
            canvas.restore_region(d["bg"])
            for ly, arts in d["artists"]:
                ly.mpl_update(i)
                for a in arts:
                    if hasattr(a, "do_3d_projection") and a.axes is not None:
                        a.do_3d_projection()
                    if a.axes is not None:
                        a.axes.draw_artist(a)
                    else:
                        a.figure.draw_artist(a)
            buf = np.asarray(canvas.buffer_rgba())
            x, y = d["xy"]
            h = min(buf.shape[0], self.H - y)
            w = min(buf.shape[1], self.W - x)
            if h > 0 and w > 0:
                out[y:y + h, x:x + w] = buf[:h, :w, :3]
        if self.stamp:
            timecode.stamp(out, self.times[i], bg=self.base[0, 0])
        return out

    def close(self) -> None:
        import matplotlib.pyplot as plt

        for d in self.regions:
            plt.close(d["fig"])


# -- the animation --------------------------------------------------------------------------------
@dataclass
class Animation:
    """A picklable recipe: data + layout + playback options. Workers rebuild their
    :class:`Scene` from it, so nothing unpicklable crosses process boundaries."""

    clip: Clip | CohortClip
    layout: str = "multimodal"
    fps: float = 30.0
    speed: float = 0.5
    theme: str = "dark"
    window_s: tuple[float, float] = (-3.0, 1.0)
    trail_s: float = 1.0
    loops: int = 2                      # cohort: cycles played
    captions: tuple[tuple[float, float, str], ...] | None = None   # D5: cues in video s

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ValueError(f"unknown layout {self.layout!r}; known: {', '.join(LAYOUTS)}")
        if self.theme not in style.THEMES:
            raise ValueError(f"unknown theme {self.theme!r}")
        if (self.layout in COHORT_LAYOUTS) != isinstance(self.clip, CohortClip):
            raise ValueError(f"the cohort layouts ({', '.join(sorted(COHORT_LAYOUTS))}) play "
                             "a CohortClip; the others a Clip")

    @property
    def timeline(self) -> Timeline:
        if self.layout in TIMELINES:
            return TIMELINES[self.layout](self)
        if isinstance(self.clip, CohortClip):
            # Pseudo clock: one cycle = 1 s of "master" time (0..100 %).
            return Timeline(tuple((0.0, 1e6) for _ in range(max(1, self.loops))), self.fps,
                            self.speed, tuple(self.clip.task for _ in range(max(1, self.loops))))
        return Timeline(tuple((a, b) for _, a, b in self.clip.segments), self.fps, self.speed,
                        tuple(p for p, _, _ in self.clip.segments))

    def compose(self) -> Composition:
        return LAYOUTS[self.layout](copy.copy(self), self.timeline)

    def scene(self, size="1080p", stamp: bool = True) -> Scene:
        return Scene(self.compose(), size, self.theme, stamp)

    def with_(self, **kw: Any) -> "Animation":
        return replace(self, **kw)

    def manifest(self) -> dict[str, Any]:
        tl = self.timeline
        c = self.clip
        d = {"layout": self.layout, "fps": self.fps, "speed": self.speed, "theme": self.theme,
             "window_s": list(self.window_s), "trail_s": self.trail_s,
             "n_frames": tl.n_frames, "duration_s": tl.duration_s,
             "segments": [list(s) for s in tl.segments], "labels": list(tl.labels)}
        if self.layout in BANNERS:
            d["banner"] = BANNERS[self.layout]
        if self.captions is not None:
            d["captions"] = [list(c) for c in self.captions]
        if isinstance(c, Clip):
            d.update(pseudonym=c.label, with_camera=bool(c.with_camera and c.previews),
                     modalities=c.modalities(), snapshot=c.meta.get("snapshot"))
        else:
            d.update(task=c.task, participants=c.labels, with_camera=False,
                     snapshot=c.meta.get("snapshot"))
        if c.meta.get("variant"):
            d["variant"] = c.meta["variant"]
        return d


def video_stem(layout: str, pid: str | None, phases: list[str] | tuple[str, ...],
               with_camera: bool = False, *, pseudonym: int | str | None = None,
               t0_s: float | None = None, t1_s: float | None = None,
               speed: float | None = None, theme: str | None = None,
               size: str | None = None, window_s: tuple[float, float] | None = None,
               trail_s: float | None = None, fps: float | None = None) -> str:
    """File stem of a rendered video (CLI and viewer): layout, participant (pseudonym only
    -- a file name never carries the study id), phases, and every setting that changes the
    frames (window ``t0_s``-``t1_s`` -- seconds from the start of *each* phase, see
    :func:`biomech.anim.source.load_clip` --, speed, frame rate when not 30 fps, theme, size,
    scrolling window, trail), so two renders with different settings never overwrite each
    other."""
    if layout == "cohort":
        stem = f"cohort_{phases[0] if phases else 'open_close'}"
    elif layout in COHORT_LAYOUTS:            # all participants, no task (workspace3d)
        stem = layout
    else:
        if pseudonym is None:
            raise ValueError("video_stem needs the participant's pseudonym (file names never "
                             "carry study ids)")
        who = f"P{pseudonym}"
        stem = f"{layout}_{who}_{'+'.join(phases)}" + ("_camera" if with_camera else "")
    if t0_s is not None or t1_s is not None:
        stem += f"_t{'' if t0_s is None else f'{t0_s:g}'}-{'' if t1_s is None else f'{t1_s:g}'}"
    if speed is not None and speed != 0.5:
        stem += f"_{speed:g}x"
    if fps is not None and float(fps) != 30.0:
        stem += f"_{float(fps):g}fps"
    if theme is not None and theme != "dark":
        stem += f"_{theme}"
    if size is not None and size != "1080p":
        stem += f"_{size}"
    if (window_s is not None and tuple(window_s) != (-3.0, 1.0)) or \
            (trail_s is not None and trail_s != 1.0):
        w = window_s or (-3.0, 1.0)
        stem += f"_w{abs(w[0]):g}-{w[1]:g}_tr{1.0 if trail_s is None else trail_s:g}"
    return stem


def default_phases(layout: str) -> list[str]:
    """Phases a layout plays when none are given (CLI, viewer)."""
    return list(DEFAULT_PHASES.get(layout, ("chewing",)))


def default_speed(layout: str) -> float:
    return float(DEFAULT_SPEED.get(layout, 0.5))


def make(ctx, pid: str | None = None, layout: str = "multimodal",
         phase: str | list[str] | None = "chewing", t0_s: float | None = None,
         t1_s: float | None = None, *, speed: float | None = 0.5, fps: float = 30.0,
         theme: str = "dark", with_camera: bool = False,
         window_s: tuple[float, float] = (-3.0, 1.0), trail_s: float = 1.0,
         task: str = "open_close", loops: int = 2,
         captions: tuple[tuple[float, float, str], ...] | None = None) -> Animation:
    """Load the data and return the :class:`Animation` (the viewer's and the CLI's entry).
    ``phase=None`` / ``speed=None``: the layout's defaults (:func:`default_phases`,
    :func:`default_speed`)."""
    from biomech.anim.source import load_clip, load_cohort_clip

    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; known: {', '.join(LAYOUTS)}")
    speed = default_speed(layout) if speed is None else speed
    if layout in COHORT_LAYOUTS:
        clip = COHORT_LOADERS.get(layout, load_cohort_clip)(ctx, task)
    else:
        if pid is None:
            raise ValueError("a participant id is required for this layout")
        clip = load_clip(ctx, pid, phase or default_phases(layout), t0_s, t1_s,
                         with_camera=with_camera, extras=EXTRAS.get(layout, ()))
    return Animation(clip, layout, fps, speed, theme, tuple(window_s), trail_s, loops,
                     tuple(tuple(c) for c in captions) if captions is not None else None)
