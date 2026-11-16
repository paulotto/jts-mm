# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Dynamic layers: what moves on top of a static FigureSpec background (an11 §3).

Every layer follows one protocol:

``prepare(t_us, seg)``
    Precompute the whole per-frame state for the frame times ``t_us`` (master
    clock, us) at once, vectorized through :mod:`biomech.timebase` (linear
    interpolation, sample-and-hold, slerp for rotations). Nothing is
    interpolated per frame afterwards.
``series(i)``
    The state of frame ``i`` as backend-neutral :class:`~biomech.viz.spec.Series`
    -- the Plotly preview frames and the LaTeX filmstrip are built from these,
    so all outputs show the same numbers. The number and kinds of series are
    the same for every frame.
``mpl_init(ax, theme, scale)`` / ``mpl_update(i)``
    Fast path for video: create animated matplotlib artists once, then only set
    their data. The scene blits them onto a cached background.

Layers know nothing about files, figures or layouts; :mod:`biomech.anim.layouts`
wires them to clips and panels.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from biomech import timebase
from biomech.viz import style
from biomech.viz.spec import Series

GAP_S = 0.25            # never interpolate across a gap longer than this (tracking loss, cut)
LABEL_COLORS = {0: "grey", 1: "orange", 2: "blue", 3: "green", 4: "vermillion", 5: "purple",
                6: "sky"}  # REST, OPEN (openers), CLOSE (closers), PROTRUSION, RETRUSION, L, R
# Cameras seeing a body: quiet when fine, loud when broken (no hue of the label classes).
QUALITY_COLORS = {0: "ink2", 1: "yellow", 2: "lightgrey"}


def resample_cols(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray, **kw) -> np.ndarray:
    """:func:`timebase.resample` channel by channel: a NaN (masked) sample in one channel
    must not blank the others (``resample`` drops rows with any non-finite value)."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        return timebase.resample(t_src, y, t_dst, **kw)
    return np.column_stack([timebase.resample(t_src, y[:, c], t_dst, **kw)
                            for c in range(y.shape[1])])


# -- colour helpers --------------------------------------------------------------------------
def rgb(name: str, theme: str) -> np.ndarray:
    h = style.hexcolor(name, theme).lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def role_rgb(role: str, theme: str) -> np.ndarray:
    return rgb(style.role(role)[0], theme)


def speed_ramp(theme: str) -> tuple[str, ...]:
    """The sequential token ramp oriented so that *faster* has more contrast with the
    background (light theme: light -> dark blue; dark theme: mid blue -> near white)."""
    seq = style.SEQUENTIAL
    return seq[3:] if theme == "light" else tuple(reversed(seq[1:7]))


def magnitude_ramp(theme: str) -> tuple[str, ...]:
    """Ramp for magnitude rasters: zero blends into the panel, large values stand out
    (light: the token ramp white -> navy; dark: panel -> navy -> white)."""
    if theme == "light":
        return style.SEQUENTIAL
    return (style.THEMES["dark"]["panel"],) + tuple(reversed(style.SEQUENTIAL[1:]))


NAN_RGB = (0.62, 0.64, 0.67)          # missing values: a neutral grey, never a ramp colour


def ramp_rgb(values: np.ndarray, ramp: tuple[str, ...]) -> np.ndarray:
    """(N,) values in 0..1 -> (N, 3) colours on a token ramp. NaN (no data) gets
    :data:`NAN_RGB`, so a gap never reads as the lowest value."""
    stops = np.array([[int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] for h in ramp])
    raw = np.asarray(values, float)
    v = np.clip(np.nan_to_num(raw), 0, 1) * (len(stops) - 1)
    k = np.minimum(v.astype(int), len(stops) - 2)
    f = (v - k)[:, None]
    out = stops[k] * (1 - f) + stops[k + 1] * f
    out[~np.isfinite(raw)] = NAN_RGB
    return out


def break_gaps(t_s: np.ndarray, y: np.ndarray, gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert a NaN row wherever consecutive samples are more than ``gap_s`` apart, so a
    line is never drawn across missing data (``t_s`` in seconds, ``y`` (N, C))."""
    t_s = np.asarray(t_s, float)
    if t_s.size < 2:
        return t_s, y
    cut = np.flatnonzero(np.diff(t_s) > gap_s) + 1
    if not cut.size:
        return t_s, y
    t2 = np.insert(t_s, cut, np.nan)
    y2 = np.insert(np.asarray(y, float), cut, np.nan, axis=0)
    return t2, y2


def mpl_cmap(ramp: tuple[str, ...]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("anim", ["#" + h for h in ramp])


# -- base -------------------------------------------------------------------------------------
class Layer:
    """Base class; subclasses have a ``panel`` field naming the axes ("region:index", or
    "region" for the region's figure) they draw on; ``n`` is the number of prepared frames."""

    def prepare(self, t_us: np.ndarray, seg: np.ndarray) -> None:
        raise NotImplementedError

    def series(self, i: int) -> list[Series]:
        return []

    def mpl_init(self, ax: Any, theme: str, scale: float) -> list[Any]:
        return []

    def mpl_update(self, i: int) -> None:
        pass


def _set3d(line: Any, xs, ys, zs) -> None:
    """``Line3D.set_data_3d`` with float arrays (mplot3d's NaN masking needs arrays)."""
    line.set_data_3d(np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float))


def _anim(artist: Any) -> Any:
    artist.set_animated(True)
    return artist


# -- cursor -----------------------------------------------------------------------------------
@dataclass
class Cursor(Layer):
    """Vertical time line on a full-window time-series panel (x = seconds since ``t_ref``)."""

    panel: str
    t_ref_us: float

    def prepare(self, t_us, seg):
        self.x = (np.asarray(t_us, float) - self.t_ref_us) * 1e-6
        self.n = len(self.x)

    def series(self, i):
        return [Series("vline", x=np.array([self.x[i]]), role="reference")]

    def mpl_init(self, ax, theme, scale):
        self._line = _anim(ax.axvline(self.x[0], color=style.hexcolor("ink", theme),
                                      lw=style.LINE_PT * scale * 1.2, alpha=0.9, zorder=8))
        return [self._line]

    def mpl_update(self, i):
        self._line.set_xdata([self.x[i], self.x[i]])


# -- scrolling time series ----------------------------------------------------------------
@dataclass
class Scope(Layer):
    """Scrolling time series: the samples in ``[t + w0, t + w1]`` drawn against the time
    relative to now (s); a dot marks each channel's value at the current instant."""

    panel: str
    t_src: np.ndarray
    y: np.ndarray                     # (N,) or (N, C)
    roles: list[str]
    window: tuple[float, float] = (-3.0, 1.0)
    width: float = 1.5                # line width factor

    def __post_init__(self):
        self.t_src = np.asarray(self.t_src, np.float64)
        self.y = np.asarray(self.y, np.float64).reshape(len(self.t_src), -1)

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.t = t
        self.i0 = np.searchsorted(self.t_src, t + self.window[0] * 1e6, "left")
        self.i1 = np.searchsorted(self.t_src, t + self.window[1] * 1e6, "right")
        self.now = resample_cols(self.t_src, self.y, t, max_gap_s=GAP_S).reshape(len(t), -1)
        self.n = len(t)

    def window_data(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        sl = slice(self.i0[i], self.i1[i])
        return break_gaps((self.t_src[sl] - self.t[i]) * 1e-6, self.y[sl], GAP_S)

    def series(self, i):
        x, y = self.window_data(i)
        out = [Series("line", x=x, y=y[:, c], role=r, style={"width": self.width / 1.5})
               for c, r in enumerate(self.roles)]
        out += [Series("scatter", x=np.array([0.0]), y=np.array([self.now[i, c]]), role=r)
                for c, r in enumerate(self.roles)]
        return out

    def mpl_init(self, ax, theme, scale):
        surface = style.hexcolor("panel", theme)
        self._lines, self._dots = [], []
        from biomech.viz.render_mpl import _LS

        for r in self.roles:
            col, ls, _ = style.role(r)
            c = style.hexcolor(col, theme)
            (ln,) = ax.plot([], [], color=c, lw=style.LINE_PT * scale * self.width,
                            ls=_LS.get(ls, "-"), solid_capstyle="round", zorder=4)
            (dt,) = ax.plot([], [], ls="none", marker="o", ms=style.MARKER_PT * scale * 3.2,
                            color=c, markeredgecolor=surface, markeredgewidth=scale, zorder=6)
            self._lines.append(_anim(ln))
            self._dots.append(_anim(dt))
        return self._lines + self._dots

    def mpl_update(self, i):
        x, y = self.window_data(i)
        for c, (ln, dt) in enumerate(zip(self._lines, self._dots)):
            ln.set_data(x, y[:, c])
            v = self.now[i, c]
            dt.set_data([0.0], [v] if np.isfinite(v) else [np.nan])


# -- moving landmarks -------------------------------------------------------------------------
@dataclass
class Trail(Layer):
    """A moving point and its fading trail over ``[t - tau, t]`` (``tau = 0``: marker only).

    ``P`` is (N, 3) in mm; ``dims`` picks the plotted coordinates (two for a plane, three
    for a 3-D panel). With ``speed`` the trail is coloured by it on the token ramp;
    otherwise it wears the role colour. Opacity falls linearly with age.
    """

    panel: str
    t_src: np.ndarray
    P: np.ndarray
    dims: tuple[int, ...] = (0, 2)
    tau_s: float = 1.0
    role: str = "primary"
    speed: np.ndarray | None = None
    vmax: float = 1.0
    head: bool = True
    width: float = 2.2                # line width factor
    size: float = 1.0                 # head marker size factor

    def __post_init__(self):
        self.t_src = np.asarray(self.t_src, np.float64)
        self.P = np.asarray(self.P, np.float64)

    @property
    def is3d(self) -> bool:
        return len(self.dims) == 3

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.t = t
        self.head_p = timebase.resample(self.t_src, self.P, t, max_gap_s=GAP_S)
        if self.tau_s > 0:
            self.tail_p = timebase.resample(self.t_src, self.P, t - self.tau_s * 1e6,
                                            max_gap_s=GAP_S)
            # Interior samples strictly inside (t - tau, t).
            self.i0 = np.searchsorted(self.t_src, t - self.tau_s * 1e6, "right")
            self.i1 = np.searchsorted(self.t_src, t, "left")
            if self.speed is not None:
                sp = np.asarray(self.speed, np.float64)
                self.head_s = timebase.resample(self.t_src, sp, t, max_gap_s=GAP_S)
                self.tail_s = timebase.resample(self.t_src, sp, t - self.tau_s * 1e6,
                                                max_gap_s=GAP_S)
        self.n = len(t)

    def trail(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Points (M, 3), their times (us) and speeds of frame ``i``'s trail: the
        interpolated tail at ``t - tau``, the samples inside, the head at ``t``."""
        sl = slice(self.i0[i], max(self.i0[i], self.i1[i]))
        pts = np.vstack([self.tail_p[i], self.P[sl], self.head_p[i]])
        tt = np.r_[self.t[i] - self.tau_s * 1e6, self.t_src[sl], self.t[i]]
        sp = None
        if self.speed is not None:
            sp = np.r_[self.tail_s[i], np.asarray(self.speed, float)[sl], self.head_s[i]]
        return pts, tt, sp

    def _kind(self, k2: str, k3: str) -> str:
        return k3 if self.is3d else k2

    def series(self, i):
        out = []
        d = list(self.dims)
        if self.tau_s > 0:
            pts, _, sp = self.trail(i)
            if self.is3d:
                out.append(Series("line3d", x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                                  role=self.role, style={"width": self.width}))
            elif sp is not None:
                out.append(Series("colorline", x=pts[:, d[0]], y=pts[:, d[1]], c=sp,
                                  role=self.role, style={"width": self.width}))
            else:
                out.append(Series("line", x=pts[:, d[0]], y=pts[:, d[1]], role=self.role,
                                  style={"width": self.width}))
        if self.head:
            h = self.head_p[i]
            if self.is3d:
                out.append(Series("scatter3d", x=h[[0]], y=h[[1]], z=h[[2]], role=self.role,
                                  style={"size": 1.6 * self.size, "ms": "o"}))
            else:
                out.append(Series("scatter", x=h[[d[0]]], y=h[[d[1]]], role=self.role,
                                  style={"size": 1.6 * self.size, "ms": "o"}))
        return out

    def mpl_init(self, ax, theme, scale):
        from matplotlib.collections import LineCollection

        self._theme = theme
        self._base = role_rgb(self.role, theme)
        self._ramp = speed_ramp(theme)
        self._panel_rgb = rgb("panel", theme)
        arts = []
        lw = style.LINE_PT * scale * self.width
        if self.tau_s > 0:
            if self.is3d:
                from mpl_toolkits.mplot3d.art3d import Line3DCollection

                self._lc = Line3DCollection([np.zeros((2, 3))], linewidths=lw,
                                            capstyle="butt")
                ax.add_collection3d(self._lc, autolim=False)
            else:
                self._lc = LineCollection([], linewidths=lw, capstyle="butt", zorder=5)
                ax.add_collection(self._lc)
            arts.append(_anim(self._lc))
        if self.head:
            c = style.hexcolor(style.role(self.role)[0], theme)
            kw = dict(ls="none", marker="o", ms=style.MARKER_PT * scale * 3.6 * self.size,
                      color=c, markeredgecolor=style.hexcolor("ink", theme),
                      markeredgewidth=0.8 * scale, zorder=7)
            if self.is3d:
                (self._dot,) = ax.plot([np.nan], [np.nan], [np.nan], **kw)
            else:
                (self._dot,) = ax.plot([np.nan], [np.nan], **kw)
            arts.append(_anim(self._dot))
        return arts

    def colors(self, tt: np.ndarray, sp: np.ndarray | None, t_now: float) -> np.ndarray:
        """RGBA of each trail segment (age fade, speed ramp)."""
        mid_t = 0.5 * (tt[:-1] + tt[1:])
        age = np.clip((t_now - mid_t) / max(self.tau_s * 1e6, 1.0), 0, 1)
        alpha = 0.08 + 0.92 * (1 - age) ** 1.3
        if sp is not None:
            v = 0.5 * (sp[:-1] + sp[1:]) / max(self.vmax, 1e-9)
            col = ramp_rgb(v, self._ramp)
        else:
            col = np.repeat(self._base[None, :], len(mid_t), axis=0)
        return np.column_stack([col, alpha])

    def mpl_update(self, i):
        d = list(self.dims)
        if self.tau_s > 0:
            pts, tt, sp = self.trail(i)
            q = pts[:, d]
            segs = np.stack([q[:-1], q[1:]], axis=1)
            ok = np.isfinite(segs).all(axis=(1, 2))
            self._lc.set_segments(segs[ok])
            self._lc.set_color(self.colors(tt, sp, self.t[i])[ok])
        if self.head:
            h = self.head_p[i]
            if self.is3d:
                _set3d(self._dot, [h[0]], [h[1]], [h[2]])
            else:
                self._dot.set_data([h[d[0]]], [h[d[1]]])


def Marker(panel: str, t_src: np.ndarray, P: np.ndarray, dims=(0, 2), role="primary",
           size: float = 1.0) -> Trail:
    """A moving landmark without a trail."""
    return Trail(panel, t_src, P, dims, 0.0, role, size=size)


# -- rigid pose ---------------------------------------------------------------------------------
OUTLINE = ("tmj_left", "left", "incisor", "right", "tmj_right")


@dataclass
class Pose(Layer):
    """The mandible as a ghost outline (condyle - molar - incisor - molar - condyle) with
    a translucent dental arch, its frame triad at the incisor, and the instantaneous
    helical axis (finite helical axis of the current velocity screw) when the jaw rotates
    faster than ``omega_min``. In a plane (``dims`` of length 2) the outline is projected
    and the axis is shown as its pierce point through the midsagittal plane (the
    instantaneous centre of rotation)."""

    panel: str
    kin: dict[str, np.ndarray]
    dims: tuple[int, ...] = (0, 1, 2)
    triad_mm: float = 12.0              # 0: no triad
    omega_min_dps: float = 20.0
    fha: bool = True
    bounds: tuple[tuple[float, float], ...] | None = None   # 3-D box the axis is clipped to

    @property
    def is3d(self) -> bool:
        return len(self.dims) == 3

    def prepare(self, t_us, seg):
        k = self.kin
        t = np.asarray(t_us, np.float64)
        ts = np.asarray(k["t_us"], np.float64)
        L = np.stack([k[n] for n in OUTLINE], axis=1)                       # (N, 5, 3)
        self.outline = timebase.resample(ts, L, t, max_gap_s=GAP_S)        # (F, 5, 3)
        self.R = timebase.resample_rotations(ts, k["R"], t, max_gap_s=GAP_S)
        self.n = len(t)
        self.axis_seg = np.full((len(t), 2, 3), np.nan)
        self.pierce = np.full((len(t), 3), np.nan)
        if not self.fha:
            return
        w = np.radians(timebase.resample(ts, k["omega"], t, max_gap_s=GAP_S))
        v = timebase.resample(ts, k["incisor_vel"], t, max_gap_s=GAP_S)
        r = self.outline[:, 2]
        wn = np.linalg.norm(w, axis=1)
        good = np.isfinite(wn) & (wn >= np.radians(self.omega_min_dps))
        n = w / np.where(wn > 0, wn, np.nan)[:, None]
        q = r + np.cross(w, v) / np.where(wn > 0, wn ** 2, np.nan)[:, None]
        c = 0.5 * (self.outline[:, 0] + self.outline[:, 4])                 # condyle midpoint
        q = q + n * np.einsum("ij,ij->i", c - q, n)[:, None]                # closest to it
        half = 0.65 * np.linalg.norm(self.outline[:, 0] - self.outline[:, 4], axis=1)
        good &= np.linalg.norm(q - c, axis=1) < 150.0
        seg_ = np.stack([q - n * half[:, None], q + n * half[:, None]], axis=1)
        if self.bounds is not None:                 # mplot3d does not clip lines to the box
            seg_, inside = clip_segments(seg_, self.bounds)
            good &= inside
        self.axis_seg[good] = seg_[good]
        ny = n[:, 1]
        ok = good & (np.abs(ny) > 0.3)
        s = -q[:, 1] / np.where(ok, ny, 1.0)
        self.pierce[ok] = (q + n * s[:, None])[ok]

    def series(self, i):
        d = list(self.dims)
        o = self.outline[i]
        if self.is3d:
            out = [Series("line3d", x=o[:, 0], y=o[:, 1], z=o[:, 2], role="context",
                          style={"width": 2.5})]
            R = self.R[i]
            for a, rl in enumerate(("axis:x", "axis:y", "axis:z")):
                dv = R[:, a] * self.triad_mm if np.isfinite(R).all() else np.full(3, np.nan)
                out.append(Series("arrow3d", x=o[[2], 0], y=o[[2], 1], z=o[[2], 2],
                                  d=dv[None, :], role=rl))
            s = self.axis_seg[i]
            out.append(Series("line3d", x=s[:, 0], y=s[:, 1], z=s[:, 2], role="threshold",
                              style={"width": 1.5}))
            return out
        out = [Series("line", x=o[:, d[0]], y=o[:, d[1]], role="context",
                      style={"width": 2.5})]
        p = self.pierce[i]
        out.append(Series("scatter", x=p[[d[0]]], y=p[[d[1]]], role="threshold",
                          style={"ms": "x", "size": 1.6}))
        return out

    def mpl_init(self, ax, theme, scale):
        ghost = style.hexcolor("ink2", theme)
        lw = style.LINE_PT * scale
        arts = []
        if self.is3d:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            (self._out,) = ax.plot([np.nan], [np.nan], [np.nan], color=ghost, lw=lw * 2.4,
                                   alpha=0.9, solid_joinstyle="round", marker="o",
                                   ms=style.MARKER_PT * scale * 2.2, markerfacecolor=ghost,
                                   markeredgecolor="none")
            self._arch = Poly3DCollection([np.zeros((3, 3))], facecolors=[(*rgb("ink2", theme),
                                                                             0.16)],
                                          edgecolors="none")
            ax.add_collection3d(self._arch, autolim=False)
            self._triad = []
            for rl in ("axis:x", "axis:y", "axis:z"):
                (ln,) = ax.plot([np.nan], [np.nan], [np.nan], lw=lw * 2.2,
                                color=style.hexcolor(style.role(rl)[0], theme),
                                solid_capstyle="round")
                self._triad.append(_anim(ln))
            (self._fha,) = ax.plot([np.nan], [np.nan], [np.nan], lw=lw * 1.8,
                                   color=style.hexcolor("vermillion", theme), ls=(0, (5, 2.5)))
            arts = [_anim(self._arch), _anim(self._out), *self._triad, _anim(self._fha)]
        else:
            (self._out,) = ax.plot([np.nan], [np.nan], color=ghost, lw=lw * 2.4, alpha=0.9,
                                   solid_joinstyle="round", marker="o",
                                   ms=style.MARKER_PT * scale * 2.2, markerfacecolor=ghost,
                                   markeredgecolor="none", zorder=4)
            (self._icr,) = ax.plot([np.nan], [np.nan], ls="none", marker="x",
                                   ms=style.MARKER_PT * scale * 4.5,
                                   mew=lw * 2.0, color=style.hexcolor("vermillion", theme),
                                   zorder=7)
            arts = [_anim(self._out), _anim(self._icr)]
        return arts

    def mpl_update(self, i):
        d = list(self.dims)
        o = self.outline[i]
        if self.is3d:
            _set3d(self._out, o[:, 0], o[:, 1], o[:, 2])
            tri = o[[1, 2, 3]]
            self._arch.set_verts([tri] if np.isfinite(tri).all() else [np.zeros((3, 3))])
            self._arch.set_alpha(None)
            R = self.R[i]
            for a, ln in enumerate(self._triad):
                if self.triad_mm <= 0:                  # triad off (triad_mm = 0)
                    _set3d(ln, [np.nan], [np.nan], [np.nan])
                    continue
                e = o[2] + R[:, a] * self.triad_mm
                _set3d(ln, [o[2, 0], e[0]], [o[2, 1], e[1]], [o[2, 2], e[2]])
            s = self.axis_seg[i]
            _set3d(self._fha, s[:, 0], s[:, 1], s[:, 2])
            self._arch.do_3d_projection()
        else:
            self._out.set_data(o[:, d[0]], o[:, d[1]])
            p = self.pierce[i]
            self._icr.set_data([p[d[0]]], [p[d[1]]])


def clip_segments(seg: np.ndarray, bounds) -> tuple[np.ndarray, np.ndarray]:
    """Clip segments (N, 2, 3) to an axis-aligned box (Liang-Barsky); returns the clipped
    segments and whether any part lies inside."""
    p0, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    t0 = np.zeros(len(seg))
    t1 = np.ones(len(seg))
    for k, (lo, hi) in enumerate(bounds):
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lo - p0[:, k]) / d[:, k]
            b = (hi - p0[:, k]) / d[:, k]
        par = d[:, k] == 0
        out = par & ((p0[:, k] < lo) | (p0[:, k] > hi))
        t0 = np.where(par, t0, np.maximum(t0, np.minimum(a, b)))
        t1 = np.where(par, t1, np.minimum(t1, np.maximum(a, b)))
        t1 = np.where(out, -1.0, t1)
    inside = t1 > t0
    res = np.stack([p0 + d * t0[:, None], p0 + d * t1[:, None]], axis=1)
    return res, inside


# -- rigid bodies in the camera frame -------------------------------------------------------------
@dataclass
class Triads(Layer):
    """Frame triads of tracked bodies (world/camera frame, tracking layout), with a
    name tag; a body that is not tracked at the instant disappears."""

    panel: str
    t_src: np.ndarray
    poses: dict[str, np.ndarray]                 # body -> (N, 4, 4)
    names: dict[str, str]
    length_mm: float = 25.0

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        ts = np.asarray(self.t_src, np.float64)
        self.state = {}
        for b, T in self.poses.items():
            o = timebase.resample(ts, T[:, :3, 3], t, max_gap_s=GAP_S)
            R = timebase.resample_rotations(ts, T[:, :3, :3], t, max_gap_s=GAP_S)
            self.state[b] = (o, R)
        self.n = len(t)

    def series(self, i):
        out = []
        for o, R in self.state.values():
            for a, rl in enumerate(("axis:x", "axis:y", "axis:z")):
                out.append(Series("arrow3d", x=o[i, [0]], y=o[i, [1]], z=o[i, [2]],
                                  d=(R[i][:, a] * self.length_mm)[None, :], role=rl))
        return out

    def mpl_init(self, ax, theme, scale):
        lw = style.LINE_PT * scale * 2.4
        self._arts = {}
        arts = []
        for b in self.state:
            lines = []
            for rl in ("axis:x", "axis:y", "axis:z"):
                (ln,) = ax.plot([np.nan], [np.nan], [np.nan], lw=lw, solid_capstyle="round",
                                color=style.hexcolor(style.role(rl)[0], theme))
                lines.append(_anim(ln))
            txt = _anim(ax.text(np.nan, np.nan, np.nan, " " + self.names.get(b, b.upper()),
                                fontsize=style.FONT_PT["label"] * scale,
                                color=style.hexcolor("ink", theme), ha="left", va="bottom"))
            self._arts[b] = (lines, txt)
            arts += lines + [txt]
        return arts

    def mpl_update(self, i):
        for b, (o, R) in self.state.items():
            lines, txt = self._arts[b]
            p = o[i]
            for a, ln in enumerate(lines):
                e = p + R[i][:, a] * self.length_mm
                _set3d(ln, [p[0], e[0]], [p[1], e[1]], [p[2], e[2]])
            txt.set_position_3d(p if np.isfinite(p).all() else (np.nan, np.nan, np.nan))
            txt.set_visible(bool(np.isfinite(p).all()))


# -- camera ---------------------------------------------------------------------------------
class _Images:
    """Small LRU of decoded preview JPEGs (RGB uint8)."""

    def __init__(self, size: int = 32) -> None:
        self.size = size
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: str) -> np.ndarray:
        if path in self._d:
            self._d.move_to_end(path)
            return self._d[path]
        from PIL import Image as PILImage

        with PILImage.open(path) as im:
            a = np.asarray(im.convert("RGB"))
        self._d[path] = a
        while len(self._d) > self.size:
            self._d.popitem(last=False)
        return a


@dataclass
class Image(Layer):
    """A camera preview: the latest frame at or before ``t`` (sample-and-hold, the
    previews are ~8 Hz), blank when the latest one is older than ``hold_s``."""

    panel: str
    t_src: np.ndarray
    paths: list[str]
    hold_s: float = 0.6
    caption: str = ""

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        ts = np.asarray(self.t_src, np.float64)
        k = np.searchsorted(ts, t, "right") - 1
        ok = (k >= 0) & (t - ts[np.clip(k, 0, None)] <= self.hold_s * 1e6)
        self.idx = np.where(ok, k, -1)
        self.n = len(t)
        self._cache = _Images()
        self._shape = self._cache.get(self.paths[0]).shape if self.paths else (240, 320, 3)

    def image(self, i: int) -> np.ndarray | None:
        k = self.idx[i]
        return None if k < 0 else self._cache.get(self.paths[k])

    def series(self, i):
        img = self.image(i)
        if img is None:
            img = np.zeros(self._shape, np.uint8)
        h, w = img.shape[:2]
        return [Series("image", img=img, extent=(0, w, h, 0))]

    def mpl_init(self, ax, theme, scale):
        h, w = self._shape[:2]
        blank = np.zeros((h, w, 3), np.uint8) + (rgb("panel", theme) * 255).astype(np.uint8)
        self._blank = blank
        self._im = _anim(ax.imshow(blank, extent=(0, w, h, 0), aspect="auto",
                                   interpolation="bilinear", zorder=1))
        self._txt = _anim(ax.text(0.5, 0.5, "no frame", transform=ax.transAxes, ha="center",
                                  va="center", color=style.hexcolor("muted", theme),
                                  fontsize=style.FONT_PT["note"] * scale))
        return [self._im, self._txt]

    def mpl_update(self, i):
        img = self.image(i)
        self._im.set_data(self._blank if img is None else img)
        self._txt.set_visible(img is None)


# -- ribbons ----------------------------------------------------------------------------------
@dataclass
class RibbonRow:
    name: str
    t_src: np.ndarray
    values: np.ndarray
    to_rgba: Callable[[np.ndarray], np.ndarray]     # (M,) values -> (M, 4) RGBA 0..1
    method: str = "nearest"


def categorical_rgba(colors: dict[int, str], theme: str, alpha: float = 0.95
                     ) -> Callable[[np.ndarray], np.ndarray]:
    lut = {k: np.r_[rgb(c, theme), alpha] for k, c in colors.items()}

    def f(v: np.ndarray) -> np.ndarray:
        out = np.zeros((len(v), 4))
        vv = np.asarray(v, float)
        for k, c in lut.items():
            out[vv == k] = c
        return out

    return f


def alpha_rgba(color: str, theme: str) -> Callable[[np.ndarray], np.ndarray]:
    c = rgb(color, theme)

    def f(v: np.ndarray) -> np.ndarray:
        a = np.clip(np.nan_to_num(np.asarray(v, float)), 0, 1)
        return np.column_stack([np.repeat(c[None, :], len(a), 0), a])

    return f


SUB = 10          # image rows per ribbon row; the first and last stay empty (row gaps)


def _gapped(img: np.ndarray) -> np.ndarray:
    """(R, M, 4) -> (R * SUB, M, 4) with a transparent line between rows."""
    out = np.repeat(img, SUB, axis=0)
    out[0::SUB, :, 3] = 0
    out[SUB - 1::SUB, :, 3] = 0
    return out


def ribbon_image(rows: list[RibbonRow], t_grid: np.ndarray, theme: str) -> np.ndarray:
    """RGB image (R * SUB, M, 3) of the rows on ``t_grid`` over the panel colour."""
    bg = rgb("panel", theme)
    img = np.empty((len(rows), len(t_grid), 4))
    for r, row in enumerate(rows):
        v = timebase.resample(row.t_src, np.asarray(row.values, float), t_grid,
                              method=row.method, max_gap_s=GAP_S)
        img[r] = row.to_rgba(v)
    c = _gapped(img)
    return ((c[..., :3] * c[..., 3:] + bg * (1 - c[..., 3:])) * 255).round().astype(np.uint8)


@dataclass
class Ribbon(Layer):
    """Scrolling coloured rows (intent labels, ML posteriors, tracking quality) over the
    window ``[t + w0, t + w1]``, one image column per ``(w1 - w0) / cols`` seconds.
    Row ``r`` sits at ``y = R - 1 - r`` (first row on top)."""

    panel: str
    rows: list[RibbonRow]
    window: tuple[float, float] = (-3.0, 1.0)
    cols: int = 360
    theme: str = "dark"

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        off = np.linspace(self.window[0], self.window[1], self.cols) * 1e6
        grid = (t[:, None] + off[None, :]).ravel()
        R = len(self.rows)
        self.vals = np.empty((len(t), R, self.cols), np.float32)
        for r, row in enumerate(self.rows):
            v = timebase.resample(row.t_src, np.asarray(row.values, float), grid,
                                  method=row.method, max_gap_s=GAP_S)
            self.vals[:, r, :] = v.reshape(len(t), self.cols)
        self.n = len(t)

    def rgba(self, i: int) -> np.ndarray:
        img = np.zeros((len(self.rows), self.cols, 4))
        for r, row in enumerate(self.rows):
            img[r] = row.to_rgba(self.vals[i, r])
        return _gapped(img)

    def series(self, i):
        bg = rgb("panel", self.theme)
        c = self.rgba(i)
        img = ((c[..., :3] * c[..., 3:] + bg * (1 - c[..., 3:])) * 255).round().astype(np.uint8)
        R = len(self.rows)
        return [Series("image", img=img, extent=(self.window[0], self.window[1], -0.5, R - 0.5))]

    def mpl_init(self, ax, theme, scale):
        R = len(self.rows)
        self._im = _anim(ax.imshow(np.zeros((R * SUB, self.cols, 4)), aspect="auto",
                                   interpolation="nearest", origin="upper",
                                   extent=(self.window[0], self.window[1], -0.5, R - 0.5),
                                   zorder=1))
        return [self._im]

    def mpl_update(self, i):
        self._im.set_data(self.rgba(i))


# -- bars -----------------------------------------------------------------------------------------
@dataclass
class Bars(Layer):
    """Horizontal bars of each channel's value at ``t`` (e.g. % of max recorded per muscle); bar ``c``
    sits at ``y = C - 1 - c``."""

    panel: str
    t_src: np.ndarray
    y: np.ndarray                      # (N, C)
    roles: list[str]

    def prepare(self, t_us, seg):
        self.v = resample_cols(np.asarray(self.t_src, float), self.y, np.asarray(t_us, float),
                               max_gap_s=GAP_S)
        self.n = len(self.v)

    def series(self, i):
        C = len(self.roles)
        return [Series("barh", x=np.array([np.nan_to_num(self.v[i, c])]),
                       y=np.array([C - 1 - c], float), role=r)
                for c, r in enumerate(self.roles)]

    def mpl_init(self, ax, theme, scale):
        C = len(self.roles)
        cols = [style.hexcolor(style.role(r)[0], theme) for r in self.roles]
        self._bars = ax.barh(np.arange(C)[::-1], np.zeros(C), height=0.62, color=cols,
                             alpha=0.9, zorder=3)
        for b in self._bars:
            _anim(b)
        return list(self._bars)

    def mpl_update(self, i):
        for b, v in zip(self._bars, self.v[i]):
            b.set_width(float(v) if np.isfinite(v) else 0.0)


# -- text -----------------------------------------------------------------------------------------
@dataclass
class Text(Layer):
    """Per-frame strings at fixed figure positions (phase, clock, current label).

    ``items``: (x, y, ha, fmt, size_key, colour token) with x/y in figure fractions;
    ``fmt(state) -> str`` receives a dict of per-frame values (``t_s``, ``phase``,
    ``frame``, plus whatever ``extra`` holds per frame)."""

    panel: str
    items: list[tuple[float, float, str, Callable[[dict], str], str, str]]
    extra: dict[str, np.ndarray] | None = None
    phase_names: list[str] | None = None
    t_ref: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    weight: tuple[str, ...] = ()

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        ts = self.t_ref(t, seg) if self.t_ref else (t - t[0]) * 1e-6
        self.strings = []
        for i in range(len(t)):
            st = {"t_s": ts[i], "frame": i, "seg": int(seg[i]),
                  "phase": (self.phase_names or [""])[int(seg[i])] if self.phase_names else ""}
            if self.extra:
                st.update({k: v[i] for k, v in self.extra.items()})
            self.strings.append([f(st) for (_, _, _, f, _, _) in self.items])
        self.n = len(t)

    def text(self, i: int) -> list[str]:
        return self.strings[i]

    def mpl_init(self, fig, theme, scale):
        self._arts = []
        for k, (x, y, ha, _, size, col) in enumerate(self.items):
            t = fig.text(x, y, "", ha=ha, va="center", color=style.hexcolor(col, theme),
                         fontsize=style.FONT_PT.get(size, 8.0) * scale * (
                             1.5 if size == "title" else 1.0),
                         fontweight="bold" if str(k) in self.weight else "normal",
                         family="DejaVu Sans Mono" if "mono" in self.weight else None)
            self._arts.append(_anim(t))
        return self._arts

    def mpl_update(self, i):
        for t, s in zip(self._arts, self.strings[i]):
            t.set_text(s)


@dataclass
class Chip(Layer):
    """The current intent label as a coloured chip (header)."""

    panel: str
    t_src: np.ndarray
    labels: np.ndarray
    names: list[str]
    x: float = 0.5
    y: float = 0.5
    method: str = "nearest"           # "hold": the latest value at or before t (causal)
    size: float = 1.15

    def prepare(self, t_us, seg):
        self.v = timebase.resample(np.asarray(self.t_src, float), np.asarray(self.labels, float),
                                   np.asarray(t_us, float), method=self.method, max_gap_s=GAP_S)
        self.n = len(self.v)

    def label(self, i: int) -> int:
        v = self.v[i]
        return int(v) if np.isfinite(v) else -1

    def mpl_init(self, fig, theme, scale):
        self._theme = theme
        self._t = _anim(fig.text(self.x, self.y, "", ha="left", va="center",
                                 fontsize=style.FONT_PT["label"] * scale * self.size,
                                 fontweight="bold", color=style.hexcolor("surface", theme),
                                 bbox={"boxstyle": "round,pad=0.35,rounding_size=0.5",
                                       "fc": "#888888", "ec": "none"}))
        return [self._t]

    def mpl_update(self, i):
        k = self.label(i)
        if k < 0 or k >= len(self.names):
            self._t.set_visible(False)
            return
        self._t.set_visible(True)
        self._t.set_text(self.names[k])
        self._t.get_bbox_patch().set_facecolor(style.hexcolor(LABEL_COLORS.get(k, "grey"),
                                                              self._theme))


# -- D5 supplementary-video layers ------------------------------------------------------------
@dataclass
class Outline(Layer):
    """A moving polyline through ``K`` points, ``pts`` (N, K, 3) in mm: a model's ghost of
    the mandible outline, or (K = 2) a vector from one point to another (e.g. the hinge
    residual at the incisor). ``head`` marks the last point, ``nodes`` every point."""

    panel: str
    t_src: np.ndarray
    pts: np.ndarray
    dims: tuple[int, ...] = (0, 2)
    role: str = "secondary"
    dashed: bool = True
    width: float = 2.0
    head: bool = False
    nodes: bool = True

    def __post_init__(self):
        self.t_src = np.asarray(self.t_src, np.float64)
        self.pts = np.asarray(self.pts, np.float64)

    @property
    def is3d(self) -> bool:
        return len(self.dims) == 3

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.p = timebase.resample(self.t_src, self.pts, t, max_gap_s=GAP_S)     # (F, K, 3)
        self.n = len(t)

    def series(self, i):
        q = self.p[i]
        d = list(self.dims)
        st = {"width": self.width}
        if self.is3d:
            out = [Series("line3d", x=q[:, 0], y=q[:, 1], z=q[:, 2], role=self.role, style=st)]
            if self.head:
                out.append(Series("scatter3d", x=q[[-1], 0], y=q[[-1], 1], z=q[[-1], 2],
                                  role=self.role, style={"size": 1.2, "ms": "o"}))
            return out
        out = [Series("line", x=q[:, d[0]], y=q[:, d[1]], role=self.role, style=st)]
        if self.head:
            out.append(Series("scatter", x=q[[-1], d[0]], y=q[[-1], d[1]], role=self.role,
                              style={"size": 1.2, "ms": "o"}))
        return out

    def mpl_init(self, ax, theme, scale):
        from biomech.viz.render_mpl import _LS

        c = style.hexcolor(style.role(self.role)[0], theme)
        kw = dict(color=c, lw=style.LINE_PT * scale * self.width,
                  ls=_LS["dashed"] if self.dashed else "-", solid_joinstyle="round",
                  marker="o" if self.nodes else None, ms=style.MARKER_PT * scale * 2.0,
                  markerfacecolor=c, markeredgecolor="none", zorder=6)
        nan = [np.nan]
        (self._ln,) = ax.plot(nan, nan, nan, **kw) if self.is3d else ax.plot(nan, nan, **kw)
        arts = [_anim(self._ln)]
        if self.head:
            hk = dict(ls="none", marker="o", ms=style.MARKER_PT * scale * 3.0, color=c,
                      markeredgecolor=style.hexcolor("ink", theme),
                      markeredgewidth=0.6 * scale, zorder=7)
            (self._hd,) = ax.plot(nan, nan, nan, **hk) if self.is3d else ax.plot(nan, nan, **hk)
            arts.append(_anim(self._hd))
        return arts

    def mpl_update(self, i):
        q = self.p[i]
        d = list(self.dims)
        if self.is3d:
            _set3d(self._ln, q[:, 0], q[:, 1], q[:, 2])
            if self.head:
                _set3d(self._hd, q[[-1], 0], q[[-1], 1], q[[-1], 2])
        else:
            self._ln.set_data(q[:, d[0]], q[:, d[1]])
            if self.head:
                self._hd.set_data(q[[-1], d[0]], q[[-1], d[1]])


@dataclass
class Events(Layer):
    """Event markers in a scrolling window: each event sits at x = its time relative to now
    (s) while it lies inside ``[t + w0, t + w1]`` -- with ``w1 = 0`` an event appears only
    once it has happened. ``y``: one level for every event, or one value per event."""

    panel: str
    t_events: np.ndarray
    y: float | np.ndarray = 0.0
    window: tuple[float, float] = (-3.0, 0.0)
    role: str = "threshold"
    marker: str = "v"
    size: float = 1.0

    def __post_init__(self):
        te = np.asarray(self.t_events, np.float64).ravel()
        o = np.argsort(te, kind="stable")
        self.t_events = te[o]
        yy = np.asarray(self.y, np.float64)
        self._y = yy.ravel()[o] if yy.ndim and yy.size == te.size else np.full(te.size,
                                                                                float(yy))

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.t = t
        self.i0 = np.searchsorted(self.t_events, t + self.window[0] * 1e6, "left")
        self.i1 = np.searchsorted(self.t_events, t + self.window[1] * 1e6, "right")
        self.n = len(t)

    def window_data(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        sl = slice(self.i0[i], self.i1[i])
        return (self.t_events[sl] - self.t[i]) * 1e-6, self._y[sl]

    def series(self, i):
        x, y = self.window_data(i)
        return [Series("scatter", x=x, y=y, role=self.role,
                       style={"ms": self.marker, "size": 1.6 * self.size})]

    def mpl_init(self, ax, theme, scale):
        c = style.hexcolor(style.role(self.role)[0], theme)
        line = self.marker in ("|", "_", "x", "+")          # line markers: edge = colour
        (self._m,) = ax.plot([], [], ls="none", marker=self.marker,
                             ms=style.MARKER_PT * scale * 4.0 * self.size, color=c,
                             markeredgecolor=c if line else style.hexcolor("ink", theme),
                             markeredgewidth=(1.6 if line else 0.5) * scale, zorder=8,
                             clip_on=False)
        return [_anim(self._m)]

    def mpl_update(self, i):
        x, y = self.window_data(i)
        self._m.set_data(x, y)


def _played(t_ev: np.ndarray, segments, seg: np.ndarray, t: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """Event indices in play order (events inside the segments, segment by segment) and,
    per frame, how many of them have been played: every event of the earlier segments and
    those of the current segment up to ``t``."""
    order, base, cur_lo = [], [0], []
    o = np.argsort(t_ev, kind="stable")
    ts = t_ev[o]
    for a, b in segments:
        lo, hi = np.searchsorted(ts, a, "left"), np.searchsorted(ts, b, "right")
        order.append(o[lo:hi])
        cur_lo.append(lo)
        base.append(base[-1] + hi - lo)
    seg = np.asarray(seg, int)
    upto = np.searchsorted(ts, t, "right")
    lo = np.asarray(cur_lo)[seg]
    hi = np.array([lo_ + len(order[j]) for j, lo_ in zip(seg, lo)]) if len(seg) else lo
    n = np.asarray(base)[seg] + np.clip(upto, lo, hi) - lo
    return (np.concatenate(order) if order else np.empty(0, int)), n


@dataclass
class Tally(Text):
    """Live counters (D5): running counts of events up to the current instant. ``events``:
    name -> (t_us (M,), values (M,)); only events inside the played ``segments`` count (the
    earlier segments in play order fully, the current one up to now). Each frame's state
    (passed to the ``items`` formatters, as for :class:`Text`) holds per name
    ``<name>_n`` (count), ``<name>_last`` (value of the latest event, NaN before the
    first) and ``<name>_median`` (median of the finite values so far)."""

    events: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    segments: tuple[tuple[float, float], ...] = ()

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        extra: dict[str, np.ndarray] = {}
        for name, (te, v) in (self.events or {}).items():
            te = np.asarray(te, np.float64)
            v = np.asarray(v, np.float64)
            order, n = _played(te, self.segments, seg, t)
            vals = v[order]
            last = np.array([vals[k - 1] if k > 0 else np.nan for k in n])
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.array([np.nanmedian(vals[:k]) if k > 0 and np.isfinite(
                        vals[:k]).any() else np.nan for k in n])
            extra[f"{name}_n"], extra[f"{name}_last"], extra[f"{name}_median"] = n, last, med
        self.extra = extra
        super().prepare(t_us, seg)


@dataclass
class Caption(Text):
    """Burned-in captions (D5): ``cues`` (v0_s, v1_s, text) in *video* seconds; the cue
    playing at a frame's video time is shown (one line, or wrapped at ``wrap`` characters)."""

    cues: tuple[tuple[float, float, str], ...] = ()
    starts_s: tuple[float, ...] = ()            # the timeline's video start of each segment
    seg_t0_us: tuple[float, ...] = ()           # and its master-clock start
    speed: float = 1.0
    wrap: int = 140

    def video_s(self, t_us: np.ndarray, seg: np.ndarray) -> np.ndarray:
        seg = np.asarray(seg, int)
        return (np.asarray(self.starts_s)[seg]
                + (np.asarray(t_us, float) - np.asarray(self.seg_t0_us)[seg])
                / (self.speed * 1e6))

    def cue_at(self, v_s: float) -> str:
        for v0, v1, txt in self.cues:
            if v0 - 1e-9 <= v_s < v1 - 1e-9:
                return txt
        return ""

    def prepare(self, t_us, seg):
        import textwrap

        v = self.video_s(t_us, seg)
        cue = np.empty(len(v), dtype=object)
        cue[:] = ["\n".join(textwrap.wrap(self.cue_at(x), self.wrap)[:2]) for x in v]
        self.extra = {"caption": cue}           # items read it as state["caption"]
        super().prepare(t_us, seg)


class Shifted(Layer):
    """Another layer shown at master time ``t + offset_us`` (D5 side-by-side phases: the
    right column plays a second phase in step with the first)."""

    def __init__(self, inner: Layer, offset_us: float) -> None:
        self.inner = inner
        self.offset_us = float(offset_us)

    @property
    def panel(self) -> str:
        return self.inner.panel

    @panel.setter
    def panel(self, v: str) -> None:
        self.inner.panel = v

    def prepare(self, t_us, seg):
        self.inner.prepare(np.asarray(t_us, np.float64) + self.offset_us, seg)
        self.n = self.inner.n

    def series(self, i):
        return self.inner.series(i)

    def mpl_init(self, ax, theme, scale):
        return self.inner.mpl_init(ax, theme, scale)

    def mpl_update(self, i):
        self.inner.mpl_update(i)


# -- J2.7 layers (workspace rotation, spectrum strips, cycle explainer) ---------------
@dataclass
class Spin(Layer):
    """A slow rotation of a whole 3-D panel: azimuth ``azim0 + 360 deg x (t / period)``
    (``turns`` full turns per period), the elevation ``elev`` with an optional gentle wobble
    of ``elev_amp`` degrees. The panel's axes are animated as a whole (mplot3d has to redraw
    its panes, ticks and every static series at a new view), so the panel's static content
    is drawn each frame; keep it light (thinned point clouds)."""

    panel: str
    period_us: float
    azim0: float = -58.0
    elev: float = 18.0
    elev_amp: float = 0.0
    turns: float = 1.0

    def angles(self, t_us: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ph = (np.asarray(t_us, np.float64) / float(self.period_us)) % 1.0
        return (self.azim0 + 360.0 * self.turns * ph,
                self.elev + self.elev_amp * np.sin(2 * np.pi * ph))

    def prepare(self, t_us, seg):
        self.azim, self.elev_ = self.angles(t_us)
        self.n = len(self.azim)

    def mpl_init(self, ax, theme, scale):
        self._ax = _anim(ax)                 # excluded from the cached background
        return [ax]

    def mpl_update(self, i):
        self._ax.view_init(elev=float(self.elev_[i]), azim=float(self.azim[i]))


@dataclass
class ScrollImage(Layer):
    """A scrolling image strip (e.g. a short-time spectrum): column ``k`` of ``rgb``
    (rows = y from ``y_lim[0]`` at the bottom to ``y_lim[1]``, M columns, RGB 0..1) is centred
    at ``t_cols[k]``; the columns inside ``[t + w0, t + w1]`` are drawn against the time
    relative to now (s)."""

    panel: str
    t_cols: np.ndarray
    rgb: np.ndarray                    # (F, M, 3)
    y_lim: tuple[float, float]
    window: tuple[float, float] = (-3.0, 1.0)

    def __post_init__(self):
        self.t_cols = np.asarray(self.t_cols, np.float64)
        self.rgb = np.asarray(self.rgb, np.float32)
        d = np.diff(self.t_cols)
        self.dt_us = float(np.median(d)) if d.size else 1e5

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.t = t
        self.i0 = np.searchsorted(self.t_cols, t + self.window[0] * 1e6, "left")
        self.i1 = np.searchsorted(self.t_cols, t + self.window[1] * 1e6, "right")
        self.n = len(t)

    def window_data(self, i: int) -> tuple[np.ndarray | None, tuple[float, float, float, float]]:
        a, b = int(self.i0[i]), int(self.i1[i])
        if b <= a:
            return None, (self.window[0], self.window[1], *self.y_lim)
        x0 = (self.t_cols[a] - self.dt_us / 2 - self.t[i]) * 1e-6
        x1 = (self.t_cols[b - 1] + self.dt_us / 2 - self.t[i]) * 1e-6
        return self.rgb[:, a:b], (x0, x1, *self.y_lim)

    def series(self, i):
        img, ext = self.window_data(i)
        if img is None:
            img = np.zeros((2, 2, 3), np.float32)
        u8 = (np.clip(img, 0, 1) * 255).round().astype(np.uint8)[::-1]
        return [Series("image", img=u8, extent=ext)]

    def mpl_init(self, ax, theme, scale):
        self._blank = np.tile(rgb("panel", theme)[None, None, :], (2, 2, 1))
        self._im = _anim(ax.imshow(self._blank, extent=(self.window[0], self.window[1],
                                                        *self.y_lim),
                                   aspect="auto", origin="lower", interpolation="nearest",
                                   zorder=1))
        return [self._im]

    def mpl_update(self, i):
        img, ext = self.window_data(i)
        self._im.set_data(self._blank if img is None else img)
        self._im.set_extent(ext)


@dataclass
class Profile(Layer):
    """Curves over a fixed x grid that change with time (e.g. the short-time spectrum at the
    current instant): ``Y`` (M, C, X) sampled at ``t_src`` (M,); frame ``i`` shows the latest
    sample at or before ``t`` (held for at most ``hold_s``), one line per channel ``C``."""

    panel: str
    t_src: np.ndarray
    x: np.ndarray
    Y: np.ndarray
    roles: list[str]
    hold_s: float = 0.25
    width: float = 1.6

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        ts = np.asarray(self.t_src, np.float64)
        k = np.searchsorted(ts, t, "right") - 1
        ok = (k >= 0) & (t - ts[np.clip(k, 0, None)] <= self.hold_s * 1e6)
        self.idx = np.where(ok, k, -1)
        self.n = len(t)

    def values(self, i: int) -> np.ndarray:
        k = int(self.idx[i])
        Y = np.asarray(self.Y)
        return np.full(Y.shape[1:], np.nan) if k < 0 else Y[k]

    def series(self, i):
        v = self.values(i)
        return [Series("line", x=np.asarray(self.x, float), y=v[c], role=r,
                       style={"width": self.width / 1.5}) for c, r in enumerate(self.roles)]

    def mpl_init(self, ax, theme, scale):
        from biomech.viz.render_mpl import _LS

        self._lines = []
        for r in self.roles:
            col, ls, _ = style.role(r)
            (ln,) = ax.plot([], [], color=style.hexcolor(col, theme), ls=_LS.get(ls, "-"),
                            lw=style.LINE_PT * scale * self.width, zorder=4)
            self._lines.append(_anim(ln))
        return self._lines

    def mpl_update(self, i):
        v = self.values(i)
        for c, ln in enumerate(self._lines):
            ln.set_data(np.asarray(self.x, float), v[c])


def _seg_index(t_src: np.ndarray, t_ref: np.ndarray, lo_s: float, hi_s: float
               ) -> tuple[np.ndarray, np.ndarray]:
    a = np.searchsorted(t_src, np.asarray(t_ref, float) + lo_s * 1e6, "left")
    b = np.searchsorted(t_src, np.asarray(t_ref, float) + hi_s * 1e6, "right")
    return a, b


@dataclass
class SegmentTrace(Layer):
    """One window per timeline segment (e.g. one cycle each): segment ``j`` shows ``y`` over
    ``[t_ref[j] + lo_s, t_ref[j] + hi_s]`` against the time relative to ``t_ref[j]`` (in units
    of ``x_scale`` s: 1e-3 for ms), the whole window faint and the part up to now in the role
    colour (the cursor reveals it); ``cursor`` adds the vertical now line."""

    panel: str
    t_src: np.ndarray
    y: np.ndarray
    t_ref: np.ndarray                  # (S,) reference instant per segment (us)
    lo_s: float
    hi_s: float
    role: str = "primary"
    x_scale: float = 1e-3
    width: float = 1.4
    cursor: bool = False

    def __post_init__(self):
        self.t_src = np.asarray(self.t_src, np.float64)
        self.y = np.asarray(self.y, np.float64).ravel()
        self.t_ref = np.asarray(self.t_ref, np.float64)
        self._a, self._b = _seg_index(self.t_src, self.t_ref, self.lo_s, self.hi_s)

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        self.seg = np.clip(np.asarray(seg, int), 0, len(self.t_ref) - 1)
        self.t = t
        self.upto = np.searchsorted(self.t_src, t, "right")
        self.n = len(t)

    def data(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        j = int(self.seg[i])
        a, b = int(self._a[j]), int(self._b[j])
        x = (self.t_src[a:b] - self.t_ref[j]) * 1e-6 / self.x_scale
        y = self.y[a:b]
        k = int(np.clip(self.upto[i], a, b)) - a
        now = (self.t[i] - self.t_ref[j]) * 1e-6 / self.x_scale
        return x, y, x[:k], y[:k], now

    def series(self, i):
        x, y, xr, yr, now = self.data(i)
        out = [Series("line", x=x, y=y, role="faint"),
               Series("line", x=xr, y=yr, role=self.role, style={"width": self.width / 1.5})]
        if self.cursor:
            out.append(Series("vline", x=np.array([now]), role="reference"))
        return out

    def mpl_init(self, ax, theme, scale):
        col = style.hexcolor(style.role(self.role)[0], theme)
        (self._all,) = ax.plot([], [], color=style.hexcolor("faint", theme),
                               lw=style.LINE_PT * scale * self.width * 0.7, zorder=2)
        (self._rev,) = ax.plot([], [], color=col, lw=style.LINE_PT * scale * self.width,
                               solid_capstyle="round", zorder=4)
        arts = [_anim(self._all), _anim(self._rev)]
        if self.cursor:
            self._cur = _anim(ax.axvline(0.0, color=style.hexcolor("ink", theme),
                                         lw=style.LINE_PT * scale * 1.1, alpha=0.85, zorder=8))
            arts.append(self._cur)
        return arts

    def mpl_update(self, i):
        x, y, xr, yr, now = self.data(i)
        self._all.set_data(x, y)
        self._rev.set_data(xr, yr)
        if self.cursor:
            self._cur.set_xdata([now, now])


@dataclass
class SegmentEvents(Layer):
    """One event per segment (NaN: none): segment ``j``'s marker sits at x = (event −
    ``t_ref[j]``) in units of ``x_scale`` s and y = ``y[j]``, shown once the frame's time has
    reached the event (the markers appear in turn as the cursor passes them)."""

    panel: str
    t_ev: np.ndarray                   # (S,) event time per segment (us), NaN: none
    y: np.ndarray                      # (S,) marker height
    t_ref: np.ndarray                  # (S,)
    role: str = "accent"
    marker: str = "v"
    size: float = 1.0
    x_scale: float = 1e-3

    def prepare(self, t_us, seg):
        t = np.asarray(t_us, np.float64)
        j = np.clip(np.asarray(seg, int), 0, len(self.t_ref) - 1)
        te = np.asarray(self.t_ev, np.float64)[j]
        self.show = np.isfinite(te) & (t >= np.nan_to_num(te, nan=np.inf))
        self.x = (te - np.asarray(self.t_ref, np.float64)[j]) * 1e-6 / self.x_scale
        self.yv = np.asarray(self.y, np.float64)[j]
        self.n = len(t)

    def point(self, i: int) -> tuple[list[float], list[float]]:
        return ([float(self.x[i])], [float(self.yv[i])]) if self.show[i] else ([], [])

    def series(self, i):
        x, y = self.point(i)
        return [Series("scatter", x=np.asarray(x, float), y=np.asarray(y, float),
                       role=self.role, style={"ms": self.marker, "size": 1.6 * self.size})]

    def mpl_init(self, ax, theme, scale):
        c = style.hexcolor(style.role(self.role)[0], theme)
        (self._m,) = ax.plot([], [], ls="none", marker=self.marker,
                             ms=style.MARKER_PT * scale * 4.4 * self.size, color=c,
                             markeredgecolor=style.hexcolor("ink", theme),
                             markeredgewidth=0.6 * scale, zorder=9, clip_on=False)
        return [_anim(self._m)]

    def mpl_update(self, i):
        self._m.set_data(*self.point(i))
