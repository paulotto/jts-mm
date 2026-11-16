# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Data atlas: views of the recorded data itself (W6), registered with kind ``data``.

Every view is a :class:`FigureSpec` builder over a :class:`DataContext`, so the
viewer (Plotly), the quick look (matplotlib) and the paper/report (TikZ) show
the same thing. Conventions:

* coordinates are in the **maxillary frame** (x anterior, y left, z up, origin
  at the upper incisor), an occlusal-plane-like frame, *not* Frankfort;
  condyle points are skin-palpated (A4);
* participants appear only as pseudonyms (P1..Pn), never with pid, age or sex;
* where trials (cycles) exist the view honours the W6a perspective
  (``trials`` / ``trials+summary`` / ``participant`` / ``cohort``), the
  alignment and the statistic; every summary comes from
  :func:`biomech.ensembles.aggregate` over ``ok`` cycles, cohort statistics over
  participant centres;
* colour by job: one sequential ramp for magnitudes (speed, activation, trial
  order, density), the fixed categorical order for tasks, side hues for
  left/right, the emphasis form (accent over grey) for summaries; references
  are the only dashed lines and are labelled in place;
* a missing modality or array gives an annotated empty panel, never an
  exception.

Numbers printed on the views that are not outcomes of the frozen plan (demand
percentiles, condylar inclinations, median frequencies, optical-vs-IMU residuals)
are marked *exploratory* in the captions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
import pandas as pd

from biomech import ensembles as E
from biomech.viz.downsample import thin_index
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series
from biomech.viz.views import vocab as V
from biomech.viz.views.vocab import MUSCLE_LABEL, MUSCLES
from biomech.viz.views.registry import REGISTRY, ViewRequest, register

MM, MMS = "\\milli\\metre", "\\milli\\metre\\per\\second"
MS2 = "\\metre\\per\\second\\squared"
DEG, DPS, DPS2 = "\\degree", "\\degree\\per\\second", "\\degree\\per\\second\\squared"
PCT, SEC, MSEC, HZ, DB, UV = ("\\percent", "\\second", "\\milli\\second", "\\hertz",
                              "\\decibel", "\\micro\\volt")

TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
TASK_LABEL = V.TASK_LABEL
TASK_ROLE = {t: f"cat:{i}" for i, t in enumerate(TASKS)}   # fixed categorical order
CYCLE_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing")
PHASES_SHOWN = ("rest",) + TASKS        # recorded segments with stored kinematics (R5b)
XYZ = {0: "Anterior", 1: "Left", 2: "Superior"}
PLANES = {"sagittal": (0, 2), "frontal": (1, 2), "horizontal": (0, 1)}
FRAME_NOTE = ("Maxillary frame (x anterior, y left, z up; origin upper incisor; "
              "occlusal-plane-like, not Frankfort).")
WAVE_PERSP = ("trials", "trials+summary", "participant", "cohort")
PARAMS = ("pid", "phase", "perspective", "align", "stat", "window")
STAT_LABEL = V.STAT_LABEL
CENTRE = {"mean_sd": "mean", "median_iqr": "median", "mean_ci": "mean"}
G0 = 9.80665


# == small helpers ==========================================================================
def _pid(req: ViewRequest) -> str | None:
    pids = req.ctx.pids()
    if req.pid in pids:
        return req.pid
    return pids[0] if pids else None


def _who(ctx, pid: str) -> str:
    return f"P{ctx.pseudonym(pid)}"


def _arr(a: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    v = a.get(key)
    return None if v is None else np.asarray(v, dtype=np.float64)


def _empty(title: str, msg: str, xlabel: str = "", ylabel: str = "") -> Panel:
    """An annotated empty panel (a missing modality, array or table)."""
    p = Panel(title, Axis(xlabel, lim=(0.0, 1.0)), Axis(ylabel, lim=(0.0, 1.0)),
              [Series("text", np.array([0.5]), np.array([0.45]), text=[msg],
                      style={"anchor": "above"})], legend="none")
    p.x.ticks, p.x.ticklabels = [], []
    p.y.ticks, p.y.ticklabels = [], []
    return p


def _empty_fig(fid: str, title: str, msg: str, width: str = "column") -> FigureSpec:
    return FigureSpec(fid, title, [_empty("", msg)], width=width, height_in=1.2,
                      caption=f"{title}: {msg}.")


def _caption(text: str, perspective: str = "", exploratory: str = "") -> str:
    out = text.rstrip()
    if perspective:
        out += f" Shown: {perspective}."
    if exploratory:
        out += f" Exploratory: {exploratory}."
    return out


def _window(req: ViewRequest, t_s: np.ndarray) -> np.ndarray:
    if req.window_s is None:
        return np.ones(t_s.shape, bool)
    return (t_s >= req.window_s[0]) & (t_s <= req.window_s[1])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _segments(t_s: np.ndarray, max_gap: float) -> list[tuple[float, float]]:
    """Covered [t0, t1] intervals of a sample-time vector (gaps > max_gap split them)."""
    t = np.asarray(t_s, float)
    if t.size == 0:
        return []
    br = np.flatnonzero(np.diff(t) > max_gap)
    starts = np.r_[0, br + 1]
    stops = np.r_[br, t.size - 1]
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, stops)]


def _seg_line(segs: list[tuple[float, float]], y: float) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for a, b in segs:
        xs += [a, b, np.nan]
        ys += [y, y, np.nan]
    return np.array(xs, float), np.array(ys, float)


def _lowpass(t_s: np.ndarray, x: np.ndarray, fc: float = 6.0) -> np.ndarray:
    """Zero-phase 2nd-order Butterworth per finite run (NaN gaps stay NaN)."""
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    if x.size < 16:
        return out
    fs = 1.0 / float(np.median(np.diff(t_s)))
    if fc >= 0.5 * fs:
        return x.copy()
    sos = butter(2, fc / (0.5 * fs), output="sos")
    for a, b in _runs(np.isfinite(x)):
        if b - a > 15:
            out[a:b] = sosfiltfilt(sos, x[a:b])
    return out


def _hull(P: np.ndarray) -> np.ndarray | None:
    """Closed convex-hull polygon (M, 2) of the finite points, or None."""
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(P, float)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 3:
        return None
    try:
        h = ConvexHull(P)
    except QhullError:
        return None
    v = P[h.vertices]
    return np.vstack([v, v[:1]])


def _ellipse(pts: np.ndarray, n: int = 40) -> np.ndarray | None:
    """1-SD ellipse (n, 2) of 2-D points from their covariance."""
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    w, V = np.linalg.eigh(np.cov(pts.T))
    w = np.sqrt(np.clip(w, 0, None))
    a = np.linspace(0, 2 * np.pi, n)
    return c + (np.column_stack([np.cos(a) * w[0], np.sin(a) * w[1]]) @ V.T)


def fit_equal(panels: list[Panel], *, min_span: float = 0.0, square: bool = False,
              common: bool = False, same: bool = False, pad: float = 0.06) -> None:
    """Explicit limits for equal-aspect (mm-for-mm) panels.

    ``min_span``: no axis spans less (a nearly straight path keeps a readable box);
    ``square``: both axes of a panel span the same; ``common``: every axis of every
    panel spans the same (square boxes at one scale); ``same``: identical x and y
    limits on every panel (small multiples).
    """
    from biomech.viz.layout import data_extent

    eq = [p for p in panels if p.equal and p.projection == "2d" and p.series]
    if not eq:
        return
    ext = [data_extent(p) for p in eq]

    def span(r):
        return (r[1] - r[0]) if r else 0.0

    if same:
        for key, attr in (("x", "x"), ("y", "y")):
            rs = [e[key] for e in ext if e[key]]
            if not rs:
                continue
            lo, hi = min(r[0] for r in rs), max(r[1] for r in rs)
            s_ = max((hi - lo) * (1 + 2 * pad), min_span)
            c = 0.5 * (lo + hi)
            for p in eq:
                getattr(p, attr).lim = (c - s_ / 2, c + s_ / 2)
        return
    S = max(max(span(e["x"]), span(e["y"])) for e in ext)
    for p, e in zip(eq, ext):
        for ax, key in ((p.x, "x"), (p.y, "y")):
            r = e[key]
            if r is None or ax.lim is not None:
                continue
            s_ = S if common else (max(span(e["x"]), span(e["y"])) if square else span(r))
            s_ = max(s_ * (1 + 2 * pad), min_span)
            c = 0.5 * (r[0] + r[1])
            ax.lim = (c - s_ / 2, c + s_ / 2)


def _bottom_xlabels(panels: list[Panel], cols: int) -> None:
    """Stacked (sharex) panels: the x-axis label only on the bottom row."""
    for p in panels[:max(len(panels) - cols, 0)]:
        p.x.label, p.x.unit = "", ""


def _grid_labels(panels: list[Panel], cols: int) -> None:
    """Small multiples with common axes: x label on the bottom row, y label on the first column."""
    n = len(panels)
    for k, p in enumerate(panels):
        if k + cols < n:
            if p.x.label:
                p.hidden_labels["x"] = (p.x.label, p.x.unit)     # restored when reflowed
            p.x.label, p.x.unit = "", ""
        if k % cols:
            if p.y.label:
                p.hidden_labels["y"] = (p.y.label, p.y.unit)
            p.y.label, p.y.unit = "", ""


def _nanline(parts: list[np.ndarray]) -> np.ndarray:
    """Stack (k, 2) pieces separated by NaN rows (one series, line breaks)."""
    out = []
    for p in parts:
        out += [p, np.full((1, p.shape[1]), np.nan)]
    return np.vstack(out) if out else np.empty((0, 2))


def _nice_cap(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return float(m * mag)
    return float(10 * mag)


def _cycles(ctx, task: str) -> pd.DataFrame:
    cyc = ctx.table("kinematics", "cycles")
    return cyc[cyc.task == task] if not cyc.empty else cyc


def _bounds(c: pd.Series) -> tuple[float, float]:
    """Movement onset..offset when known (the same window as ensembles.kinematic)."""
    t0 = c.get("t_on_us", c.t_start_us)
    t1 = c.get("t_off_us", c.t_end_us)
    t0 = c.t_start_us if t0 is None or not np.isfinite(t0) else t0
    t1 = c.t_end_us if t1 is None or not np.isfinite(t1) else t1
    return float(t0), float(t1)


def custom_ensemble(ctx, task: str, fn: Callable[[str], tuple[np.ndarray, np.ndarray] | None],
                    channels: list[str], unit: str = "", name: str = "") -> E.Ensemble:
    """Cycle-normalized ensemble of any per-participant signal on the kinematic cycles.

    ``fn(pid) -> (t_us (N,), values (N, C))`` or None; each detected cycle of
    ``task`` is cut at movement onset..offset and resampled to 101 points, exactly
    like :func:`biomech.ensembles.kinematic` (so several signals of one cycle stay
    aligned and can be combined, e.g. a path coloured by activation).
    """
    from biomech.kinematics.cycles import time_normalize

    cyc = _cycles(ctx, task)
    rows, data = [], []
    for pid in ctx.pids():
        if cyc.empty:
            break
        got = fn(pid)
        if got is None:
            continue
        t, V = np.asarray(got[0], float), np.asarray(got[1], float)
        if V.ndim == 1:
            V = V[:, None]
        for _, c in cyc[cyc.pid == pid].iterrows():
            t0, t1 = _bounds(c)
            sel = (t >= t0) & (t <= t1)
            if sel.sum() < 3:
                continue
            data.append(time_normalize(V[sel], E.N_CYCLE))
            rows.append({"pid": pid, "task": task, "side": c.get("side", ""),
                         "cycle": int(c.cycle), "ok": bool(c.ok), "flags": c.get("flags", ""),
                         "t_start_us": int(t0), "t_peak_us": float(c.t_peak_us),
                         "duration_s": (t1 - t0) * 1e-6})
    D = np.stack(data) if data else np.zeros((0, E.N_CYCLE, len(channels)))
    trials = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pid", "task", "cycle", "ok"])
    return E.Ensemble(name or f"{task}/custom", np.linspace(0, 100, E.N_CYCLE), D, trials,
                      list(channels), "cycle", unit)


def _kin_fn(*keys: str, scale: dict[str, float] | None = None):
    """fn(pid) for custom_ensemble from kinematics arrays ``<task>__<key>`` (columns joined)."""
    def make(ctx, task):
        def fn(pid):
            a = ctx.arrays("kinematics", pid)
            t = a.get(f"{task}__t_us")
            cols = []
            for k in keys:
                v = _arr(a, f"{task}__{k.split(':')[0]}")
                if t is None or v is None:
                    return None
                if ":" in k:                       # "incisor_vel:speed" etc.
                    op = k.split(":")[1]
                    v = np.linalg.norm(v, axis=1) if op == "norm" else v[:, "xyz".index(op)]
                v = v * (scale or {}).get(k, 1.0)
                cols.append(v if v.ndim == 2 else v[:, None])
            return t, np.column_stack(cols)
        return fn
    return make


@dataclass
class Drawn:
    """What a perspective draws from an ensemble (see W6a)."""

    level: str
    trials: list[tuple[np.ndarray, int, bool]] = field(default_factory=list)  # data, order, ok
    members: list[tuple[str, np.ndarray]] = field(default_factory=list)       # pid, centre
    center: np.ndarray | None = None
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None
    cloud: np.ndarray | None = None        # (M, S, C) points behind the centre (ellipses)
    highlight: str | None = None
    n_part: int = 0
    n_trials: int = 0
    desc: str = ""
    stat: str = "mean_sd"

    @property
    def K(self) -> int:
        return max((k for _, k, _ in self.trials), default=1)

    @property
    def clim(self) -> tuple[float, float]:
        # Trial order on the sequential ramp, early = light; the lightest third of the
        # ramp is skipped so the first cycle stays visible on a white surface.
        K = self.K
        return (1.0 - 0.6 * max(K - 1, 1), float(max(K, 2)))

    def colorbar(self) -> Colorbar:
        return Colorbar("Cycle", lim=self.clim, ticks=sorted({1.0, float(self.K)}))

    def empty(self) -> bool:
        return not self.trials and not self.members and self.center is None


def resolve(ens: E.Ensemble, perspective: str, stat: str, pid: str | None, who: str = "",
            align: str | None = None) -> Drawn:
    """The perspective's lines and summaries; numbers come from ensembles.aggregate only."""
    tr = ens.trials
    p = perspective if perspective in WAVE_PERSP else "cohort"
    d = Drawn(p, stat=stat)
    if p in ("trials", "trials+summary"):
        e = ens.select((tr.pid == pid).to_numpy()) if len(tr) else ens
        if e.n_trials:
            start = (e.trials.t_start_us.to_numpy(float) if "t_start_us" in e.trials
                     else np.arange(e.n_trials, dtype=float))
            order = np.argsort(np.argsort(start, kind="stable")) + 1
            d.trials = [(e.data[k], int(order[k]), bool(e.trials.ok.iloc[k]))
                        for k in range(e.n_trials)]
        ok = e.ok()
        d.n_part, d.n_trials = (1 if ok.n_trials else 0), ok.n_trials
        if p == "trials+summary" and ok.n_trials:
            s = E.aggregate(ok, "participant", stat)
            d.center, d.lo, d.hi = s.center[0], s.lo[0], s.hi[0]
            d.cloud = ok.data
    elif p == "participant":
        s = E.aggregate(ens, "participant", stat)
        d.members = list(zip(s.groups, s.center))
        d.highlight = pid
        if pid in s.groups:
            i = s.groups.index(pid)
            d.lo, d.hi = s.lo[i], s.hi[i]
        d.n_part, d.n_trials = len(s.groups), int(np.sum(s.n))
    else:
        s = E.aggregate(ens, "cohort", stat)
        if s.groups:
            d.members = list(zip(ens.ok().pids(), s.members))
            d.center, d.lo, d.hi = s.center[0], s.lo[0], s.hi[0]
            d.cloud = s.members
            d.n_part, d.n_trials = int(s.n[0]), int(s.n_trials[0])
    level = {"trials": "trials", "trials+summary": "trials+summary"}.get(p, p)
    d.desc = E.describe_perspective(level, align or ens.align, stat, d.n_part, d.n_trials)
    if p in ("trials", "trials+summary") and who:
        d.desc = f"{who}, " + d.desc
    if p == "participant" and who:
        d.desc += f"; {who} highlighted"
    return d


def _wave(panel: Panel, grid: np.ndarray, d: Drawn, ch: int = 0, *, f=None,
          role: str | None = None, labels: bool = False, label: str | None = None,
          band: bool = True) -> None:
    """Series of one waveform channel under perspective ``d`` (W6a waveform column)."""
    f = f or (lambda v: v)
    accent = role or "accent"
    first = True
    for data, k, ok in d.trials:
        y = f(data[:, ch])
        if d.level == "trials" and role is None:
            if ok:
                panel.series.append(Series("colorline", grid, y, c=np.full(grid.size, float(k)),
                                           style={"lim": d.clim, "width": 0.6}))
            else:
                panel.series.append(Series("line", grid, y, role="noise",
                                           label="flagged cycle" if labels and first else None))
        else:
            r = role or "faint"
            st = {"width": 0.5, "alpha": 0.35} if role else {}
            panel.series.append(Series("line", grid, y, role=r if ok else "noise", style=st,
                                       label=("cycles" if labels and first and role is None
                                              else None)))
            first = False
    first = True
    for pid, m in d.members:
        hi = pid == d.highlight
        panel.series.append(Series("line", grid, f(m[:, ch]),
                                   role=accent if hi else ("faint" if role else "context"),
                                   style={"width": 1.2} if hi else {},
                                   label=("participant " + CENTRE[d.stat] + "s"
                                          if labels and first and not hi and role is None
                                          else None)))
        first = first and hi
    if band and d.lo is not None and d.hi is not None:
        panel.series.append(Series("band", grid, f(d.lo[:, ch]), f(d.hi[:, ch]),
                                   role=role or "band"))
    if d.center is not None:
        lab = label or (STAT_LABEL[d.stat] if labels else None)
        panel.series.append(Series("line", grid, f(d.center[:, ch]), role=accent,
                                   style={"width": 1.3}, label=lab))


def _path(panel: Panel, d: Drawn, ix: int, iy: int, *, labels: bool = False,
          ellipses: bool = True) -> None:
    """2-D path series (columns ix, iy of the ensemble channels) under perspective ``d``."""
    first = True
    for data, k, ok in d.trials:
        x, y = data[:, ix], data[:, iy]
        if d.level == "trials":
            if ok:
                panel.series.append(Series("colorline", x, y, c=np.full(x.size, float(k)),
                                           style={"lim": d.clim, "width": 0.6}))
            else:
                panel.series.append(Series("line", x, y, role="noise"))
        else:
            panel.series.append(Series("line", x, y, role="faint" if ok else "noise",
                                       label="cycles" if labels and first else None))
            first = False
    first = True
    for pid, m in d.members:
        hi = pid == d.highlight
        panel.series.append(Series("line", m[:, ix], m[:, iy], role="accent" if hi else "context",
                                   style={"width": 1.2} if hi else {},
                                   label=("participant " + CENTRE[d.stat] + "s"
                                          if labels and first and not hi else None)))
        first = first and hi
    if d.center is not None:
        panel.series.append(Series("line", d.center[:, ix], d.center[:, iy], role="accent",
                                   style={"width": 1.3},
                                   label=f"{CENTRE[d.stat]} path" if labels else None))
        if ellipses and d.cloud is not None and len(d.cloud) >= 3:
            parts = []
            for k in range(0, d.cloud.shape[1], 10):
                el = _ellipse(d.cloud[:, k][:, [ix, iy]])
                if el is not None:
                    parts.append(el)
            if parts:
                P = _nanline(parts)
                panel.series.append(Series("line", P[:, 0], P[:, 1], role="accent",
                                           style={"width": 0.45},
                                           label="1-SD ellipses, 10 % steps" if labels else None))


# == kinematics helpers =====================================================================
def _phases_present(a: dict[str, np.ndarray], tasks=TASKS) -> list[str]:
    return [t for t in tasks if f"{t}__incisor" in a and f"{t}__t_us" in a]


def _closed_frame(a: dict[str, np.ndarray]) -> tuple[str, int] | None:
    """(phase, index) of the closed reference pose (theta = 0, else highest incisor)."""
    th = _arr(a, "open_close__theta_deg")
    if th is not None and np.isfinite(th).any():
        return "open_close", int(np.nanargmin(np.abs(th)))
    inc = _arr(a, "open_close__incisor")
    if inc is not None and np.isfinite(inc[:, 2]).any():
        return "open_close", int(np.nanargmax(inc[:, 2]))
    return None


def _open_frame(a: dict[str, np.ndarray]) -> tuple[str, int] | None:
    """(phase, index) of the largest opening of any task."""
    best, where = -np.inf, None
    for t in _phases_present(a):
        th = _arr(a, f"{t}__theta_deg")
        v = th if th is not None else (None if _arr(a, f"{t}__incisor") is None
                                       else -_arr(a, f"{t}__incisor")[:, 2])
        if v is not None and np.isfinite(v).any() and np.nanmax(v) > best:
            best, where = float(np.nanmax(v)), (t, int(np.nanargmax(v)))
    return where


def _maxillary_landmarks(ctx, pid: str, a: dict[str, np.ndarray]) -> np.ndarray | None:
    """Static upper incisor and molars in the maxillary frame (3, 3)."""
    lm = ctx.table("kinematics", "landmarks", [pid])
    T = a.get("T_max_hp")
    if lm.empty or T is None:
        return None
    pts = []
    for role in ("incisor", "left", "right"):
        r = lm[(lm.jaw == "maxilla") & (lm.role_used == role) & (lm.body == "hp")]
        if r.empty:
            continue
        p = r[["x", "y", "z"]].to_numpy(float)[0]
        pts.append(np.asarray(T, float)[:3, :3] @ p + np.asarray(T, float)[:3, 3])
    return np.array(pts) if pts else None


def _ghost(a: dict[str, np.ndarray], where: tuple[str, int]) -> np.ndarray | None:
    """Mandible 'ghost': incisor-molar triangle plus rami to the condyle points (one line)."""
    ph, i = where
    pts = {r: _arr(a, f"{ph}__{r}") for r in ("incisor", "left", "right", "tmj_left",
                                               "tmj_right")}
    if any(v is None for v in pts.values()):
        return None
    P = {r: v[i] for r, v in pts.items()}
    seq = [P["incisor"], P["left"], P["tmj_left"], P["tmj_right"], P["right"], P["incisor"],
           np.full(3, np.nan), P["left"], P["right"]]
    return np.array(seq)


def _xyz_label(i: int) -> Axis:
    return Axis(f"{XYZ[i]} ({'xyz'[i]})", MM)


def _plane_panel(title: str, plane: str) -> Panel:
    ix, iy = PLANES[plane]
    return Panel(title, _xyz_label(ix), _xyz_label(iy), [], equal=True)


# == 1. kinematics, space ===================================================================
@register("data_orthographic", "data", "Orthographic projections", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("time", "path"), perspective="trials", align="time",
          multi_phase=True)
def orthographic(req: ViewRequest) -> FigureSpec:
    """Sagittal (Posselt-type envelope), frontal and horizontal (gothic-arch-type) projections."""
    return _ortho(req, "planes")


@register("data_jaw3d", "data", "Jaw motion in 3-D", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("time", "path"), perspective="trials", align="time",
          multi_phase=True)
def jaw3d(req: ViewRequest) -> FigureSpec:
    """3-D paths of incisor, molars and condyle points, maxillary landmarks, ghost mandible."""
    return _ortho(req, "3d")


def _task_selection(phase) -> list[str] | None:
    """Tasks of a (multi-)phase argument in task order; None = every task present."""
    if isinstance(phase, (list, tuple)):
        sel = [t for t in PHASES_SHOWN if t in phase]
        return sel or None
    return [phase] if phase in PHASES_SHOWN else None


def _ortho(req: ViewRequest, part: str) -> FigureSpec:
    ctx = req.ctx
    pid = _pid(req)
    fid, title = (("data_orthographic", "Jaw motion, orthographic projections") if part == "planes"
                  else ("data_jaw3d", "Jaw motion in 3-D"))
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    a = ctx.arrays("kinematics", pid)
    who = _who(ctx, pid)
    raw = req.align == "time" and req.perspective == "trials"
    chosen = _task_selection(req.phase)
    tasks = chosen or _phases_present(a, PHASES_SHOWN)
    if not tasks or not _phases_present(a, tasks):
        return _empty_fig(fid, title, f"no kinematics for {who}", "text")
    ps = _plane_panel("(a) Sagittal", "sagittal")
    pf = _plane_panel("(b) Frontal", "frontal")
    ph = _plane_panel("(c) Horizontal", "horizontal")
    p3 = Panel("", _xyz_label(0), _xyz_label(1), [], projection="3d", z=_xyz_label(2),
               view=(-60.0, 20.0), legend="below")
    persp = f"{who}, all frames of each task"
    if chosen and len(chosen) > 1:
        persp = f"{who}, all frames of " + ", ".join(TASK_LABEL[t] for t in chosen)
    if raw:
        union = []
        for t in tasks:
            P = _arr(a, f"{t}__incisor")
            if P is None:
                continue
            union.append(P[:, [0, 2]])
            for pan, plane in ((pf, "frontal"), (ps, "sagittal"), (ph, "horizontal")):
                ix, iy = PLANES[plane]
                pan.series.append(Series("line", P[:, ix], P[:, iy], role=V.phase_role(t),
                                         style={"width": 0.7},
                                         label=TASK_LABEL[t] if plane == "horizontal" else None))
            p3.series.append(Series("line3d", P[:, 0], P[:, 1], z=P[:, 2], role=V.phase_role(t),
                                    style={"width": 0.6}, label=TASK_LABEL[t]))
        hull = _hull(np.vstack(union)) if union else None
        if hull is not None:
            ps.series.insert(0, Series("fill", hull[:, 0], hull[:, 1], role="context",
                                       style={"alpha": 0.12}, label="envelope (convex hull)"))
    else:
        task = chosen[0] if chosen else "open_close"     # one task per cycle ensemble
        ens = E.kinematic(ctx, task, "incisor", "path")
        d = resolve(ens, req.perspective, req.stat, pid, who, "path")
        if d.empty():
            return _empty_fig(fid, title, f"no {TASK_LABEL[task]} cycles", "text")
        for pan, plane in ((pf, "frontal"), (ps, "sagittal"), (ph, "horizontal")):
            ix, iy = PLANES[plane]
            _path(pan, d, ix, iy, labels=plane == "horizontal")
        if d.level == "trials":
            ps.colorbar = d.colorbar()
        for data, _, ok in d.trials:
            p3.series.append(Series("line3d", data[:, 0], data[:, 1], z=data[:, 2],
                                    role="context" if ok else "faint", style={"width": 0.5}))
        for mp, m in d.members:
            p3.series.append(Series("line3d", m[:, 0], m[:, 1], z=m[:, 2],
                                    role="accent" if mp == d.highlight else "context",
                                    style={"width": 0.6}))
        if d.center is not None:
            p3.series.append(Series("line3d", d.center[:, 0], d.center[:, 1], z=d.center[:, 2],
                                    role="accent", label=f"incisor, {CENTRE[d.stat]} path"))
        pts = [x for x, _, _ in d.trials] + [m for _, m in d.members]
        if d.level in ("trials", "trials+summary") and pts:
            hull = _hull(np.vstack([x[:, [0, 2]] for x in pts]))
            if hull is not None:
                ps.series.insert(0, Series("fill", hull[:, 0], hull[:, 1], role="context",
                                           style={"alpha": 0.12}, label="envelope (convex hull)"))
        persp = d.desc + f", {TASK_LABEL[task]}"
    # 3-D anatomy: molar and condyle paths, static maxillary landmarks, ghost mandible.
    if raw:
        for t in tasks:
            for r in ("left", "right", "tmj_left", "tmj_right"):
                P = _arr(a, f"{t}__{r}")
                if P is not None:
                    p3.series.append(Series("line3d", P[:, 0], P[:, 1], z=P[:, 2], role="context",
                                            style={"width": 0.4},
                                            label="molar and condyle paths"
                                            if (t, r) == (tasks[0], "left") else None))
    mx = _maxillary_landmarks(ctx, pid, a)
    if mx is not None:
        p3.series.append(Series("scatter3d", mx[:, 0], mx[:, 1], z=mx[:, 2], role="reference",
                                label="maxillary landmarks", style={"ms": "d"}))
    for where, rl, lab in ((_closed_frame(a), "context", "mandible closed"),
                           (_open_frame(a), "faint", "mandible max. open")):
        g = _ghost(a, where) if where else None
        if g is not None:
            p3.series.append(Series("line3d", g[:, 0], g[:, 1], z=g[:, 2], role=rl,
                                    style={"width": 0.8}, label=lab))
    src = ["kinematics:arrays", "kinematics:landmarks", "kinematics:cycles"]
    if part == "3d":
        return FigureSpec(
            fid, title, [p3], width="column", height_in=2.9,
            caption=_caption(
                "Paths of the incisal point (by task), the lower molars and the skin-palpated "
                "condyle points, the static maxillary landmarks (diamonds) and the mandible "
                "(incisor-molar triangle, rami to the condyle points) at closure and at maximal "
                "opening; orthographic projection, equal mm axes. " + FRAME_NOTE, persp),
            sources=src, perspective=persp, meta={"pid": pid})
    fit_equal([ps, pf, ph], min_span=12.0)
    ps.legend = "below"                      # the envelope entry; the paths' legend is under (c)
    return FigureSpec(
        fid, title, [ps, pf, ph], rows=1, cols=3, width="text", height_in=2.6,
        caption=_caption(
            "Incisal point in the sagittal, frontal and horizontal planes on one shared mm scale "
            "(sagittal and frontal share the vertical axis, sagittal and horizontal the "
            "anterior one). The sagittal envelope is the convex hull of all incisal positions "
            "(Posselt-type); the horizontal plane shows the gothic-arch-type tracing of "
            "protrusion and laterotrusion. " + FRAME_NOTE, persp),
        sources=src, perspective=persp, meta={"equal_scale": "shared", "pid": pid})


def _take_series(a: dict[str, np.ndarray], phases: list[str]) -> dict[str, np.ndarray]:
    """The stored per-phase kinematics stitched on the master clock (NaN between phases):
    time (s), incisor (N, 3) and the rotation vector since the closed pose (deg)."""
    from biomech.kinematics import se3

    Tc = a.get("T_closed")
    parts = []
    for ph in sorted(phases, key=lambda q: float(np.nanmin(a[f"{q}__t_us"]))):
        t = np.asarray(a[f"{ph}__t_us"], float) * 1e-6
        inc = np.asarray(a[f"{ph}__incisor"], float)
        T = a.get(f"{ph}__T")
        rv = (np.degrees(se3.rotvec_rel(np.asarray(T, float), np.asarray(Tc, float)))
              if T is not None and Tc is not None else np.full((len(t), 3), np.nan))
        parts.append((t, inc, rv))
        parts.append((np.array([np.nan]), np.full((1, 3), np.nan), np.full((1, 3), np.nan)))
    if not parts:
        return {}
    return {"t": np.concatenate([p[0] for p in parts]),
            "incisor": np.vstack([p[1] for p in parts]), "rot": np.vstack([p[2] for p in parts])}


@register("data_kin_timeseries", "data", "Kinematics over the take", params=("pid",),
          perspectives=("trials",), perspective="trials")
def kin_timeseries(req: ViewRequest) -> FigureSpec:
    """Incisal point and mandibular rotation over the whole take, every recorded segment
    coloured and labelled (phase lane on top, faint bands behind the traces)."""
    from biomech.viz.views import blocks as B

    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_kin_timeseries", "Kinematics over the take"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    a = ctx.arrays("kinematics", pid)
    ser = _take_series(a, _phases_present(a, PHASES_SHOWN))
    if not ser:
        return _empty_fig(fid, title, f"no kinematics for {who}", "text")
    phases = ctx.phases(pid, "motion")
    t0 = min([ph.t_start_us * 1e-6 for ph in phases] + [float(np.nanmin(ser["t"]))])
    spans = [(ph.name, ph.t_start_us * 1e-6 - t0, ph.t_end_us * 1e-6 - t0) for ph in phases]
    t = ser["t"] - t0
    lane = B.phase_lane(spans)
    pi = Panel("(a) Incisal point", Axis(""), Axis("Position", MM), [], sharex="time",
               legend="below")
    pr = Panel("(b) Mandibular rotation since closure", Axis("Time in take", SEC),
               Axis("Rotation", DEG), [], sharex="time", legend="below")
    for k, ax in enumerate("xyz"):
        pi.series.append(Series("line", t, ser["incisor"][:, k], role=f"axis:{ax}",
                                style={"width": 0.6}, label=f"{XYZ[k].lower()} ({ax})"))
        pr.series.append(Series("line", t, ser["rot"][:, k], role=f"axis:{ax}",
                                style={"width": 0.6}, label=f"about {ax}"))
    for pan in (pi, pr):
        B.add_phase_bands(pan, spans)
    persp = f"{who}, motion take"
    return FigureSpec(fid, title, [lane, pi, pr], rows=3, cols=1, width="text", height_in=1.7,
                      caption=_caption(
                          "Incisal point (maxillary frame) and the rotation vector of the "
                          "mandible since the closed pose over the take. Coloured bands: "
                          "rest and the five movement tasks; grey: calibration and landmark "
                          "holds (no stored kinematics, labelled in the lane). " + FRAME_NOTE,
                          persp),
                      sources=["kinematics:arrays", "session:phases"], perspective=persp,
                      meta={"pid": pid})


@register("data_orient_pairs", "data", "Rotation component pairs", params=("pid",),
          perspectives=("trials",), perspective="trials")
def orient_pairs(req: ViewRequest) -> FigureSpec:
    """Rotation components against each other (opening vs lateral and vs tilt), one colour
    per recorded segment."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_orient_pairs", "Rotation component pairs"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    a = ctx.arrays("kinematics", pid)
    phs = _phases_present(a, PHASES_SHOWN)
    if not phs:
        return _empty_fig(fid, title, f"no kinematics for {who}", "text")
    pa = Panel("(a) Opening vs lateral rotation", Axis("About z (lateral)", DEG),
               Axis("About y (opening)", DEG), [])
    pb = Panel("(b) Opening vs tilt", Axis("About x (tilt)", DEG),
               Axis("About y (opening)", DEG), [], legend="below")
    for ph in phs:
        rv = _take_series(a, [ph])["rot"]
        pa.series.append(Series("line", rv[:, 2], rv[:, 1], role=V.phase_role(ph),
                                style={"width": 0.6}))
        pb.series.append(Series("line", rv[:, 0], rv[:, 1], role=V.phase_role(ph),
                                style={"width": 0.6}, label=TASK_LABEL[ph]))
    persp = f"{who}, all frames of each segment"
    return FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=2.4,
                      caption=_caption(
                          "Components of the mandibular rotation vector since the closed pose "
                          "(maxillary frame), one colour per recorded segment. " + FRAME_NOTE,
                          persp),
                      sources=["kinematics:arrays"], perspective=persp, meta={"pid": pid})


def _speed_ensemble(ctx, task):
    return custom_ensemble(ctx, task, _kin_fn("incisor", "incisor_vel:norm")(ctx, task),
                           ["x", "y", "z", "speed"], MM, f"{task}/incisor+speed")


@register("data_speed_path", "data", "Speed-coloured paths", params=PARAMS,
          perspectives=("trials", "participant", "cohort"), aligns=("time", "path"),
          perspective="trials", align="time", phase="open_close")
def speed_path(req: ViewRequest) -> FigureSpec:
    """Incisal path coloured by tangential speed (sagittal and frontal)."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_speed_path", "Incisal path coloured by speed"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase if req.phase in TASKS else "open_close"
    a = ctx.arrays("kinematics", pid)
    ps = _plane_panel("(a) Sagittal", "sagittal")
    pf = _plane_panel("(b) Frontal", "frontal")
    lines: list[tuple[np.ndarray, np.ndarray]] = []       # (P (N,3), speed (N,))
    context: list[np.ndarray] = []
    if req.align == "time" and req.perspective == "trials":
        P, V = _arr(a, f"{task}__incisor"), _arr(a, f"{task}__incisor_vel")
        if P is None or V is None:
            msg = "no incisor velocity in the store" if P is not None else "no kinematics"
            return FigureSpec(fid, title, [_empty("(a) Sagittal", msg), _empty("(b) Frontal", msg)],
                              rows=1, cols=2, width="column", height_in=1.4,
                              caption=f"{title}: {msg}.")
        t = _arr(a, f"{task}__t_us")
        sel = _window(req, (t - t[0]) * 1e-6)
        lines.append((P[sel], np.linalg.norm(V[sel], axis=1)))
        persp = f"{who}, {TASK_LABEL[task]}, all frames"
    else:
        ens = _speed_ensemble(ctx, task)
        d = resolve(ens, "participant" if req.perspective == "participant" else
                    ("cohort" if req.perspective == "cohort" else "trials"), req.stat, pid, who,
                    "path")
        if d.empty():
            return _empty_fig(fid, title, f"no {TASK_LABEL[task]} cycles")
        if d.level == "trials":
            lines += [(x[:, :3], x[:, 3]) for x, _, _ in d.trials]
        elif d.level == "participant":
            for mp, m in d.members:
                (lines if mp == d.highlight else context).append(
                    (m[:, :3], m[:, 3]) if mp == d.highlight else m[:, :3])
        else:
            context += [m[:, :3] for _, m in d.members]
            lines.append((d.center[:, :3], d.center[:, 3]))
        persp = d.desc + f", {TASK_LABEL[task]}"
    allc = np.concatenate([c for _, c in lines]) if lines else np.array([1.0])
    lim = (0.0, _nice_cap(float(np.nanpercentile(allc, 99)) if np.isfinite(allc).any() else 1.0))
    for pan, plane in ((ps, "sagittal"), (pf, "frontal")):
        ix, iy = PLANES[plane]
        for m in context:
            pan.series.append(Series("line", m[:, ix], m[:, iy], role="faint"))
        for P, c in lines:
            pan.series.append(Series("colorline", P[:, ix], P[:, iy], c=c,
                                     style={"lim": lim}))
    pf.colorbar = Colorbar("Speed", MMS, lim=lim)
    fit_equal([ps, pf], min_span=12.0)
    return FigureSpec(fid, title, [ps, pf], rows=1, cols=2, width="column", height_in=2.2,
                      caption=_caption("Incisal path coloured by tangential speed "
                                       "(Savitzky-Golay derivative, 6 Hz bandwidth); fast "
                                       "segments dark, hesitations light. " + FRAME_NOTE, persp),
                      sources=["kinematics:arrays", "kinematics:cycles"], perspective=persp,
                      meta={"equal_scale": "shared", "pid": pid, "speed_lim": lim})


def _rel(P: np.ndarray) -> np.ndarray:
    """Displacement from the first finite sample."""
    ok = np.flatnonzero(np.isfinite(P).all(axis=1))
    return P - P[ok[0]] if ok.size else P


def _inclination(P: np.ndarray) -> float:
    """Path inclination (deg below the x axis) from start to the most anterior point."""
    if P.size == 0 or not np.isfinite(P[:, 0]).any():
        return float("nan")
    k = int(np.nanargmax(P[:, 0]))
    return float(np.degrees(np.arctan2(-P[k, 2], P[k, 0]))) if P[k, 0] > 0.3 else float("nan")


@register("data_condylar_paths", "data", "Condylar paths", params=PARAMS,
          perspectives=("trials", "trials+summary", "cohort"), aligns=("path",),
          perspective="trials+summary", align="path")
def condylar_paths(req: ViewRequest) -> FigureSpec:
    """Axiography-style condyle-point paths: opening, protrusion (sagittal), laterotrusion."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_condylar_paths", "Condylar paths"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    panels, notes, persp = [], [], ""
    spec_ = (("open_close", "(a) Opening-closing, sagittal", (0, 2)),
             ("protrusion_retrusion", "(b) Protrusion, sagittal", (0, 2)),
             ("left_right", "(c) Laterotrusion to the left", (0, 1)))
    for task, ttl, (ix, iy) in spec_:
        p = Panel(ttl, Axis(XYZ[ix], MM), Axis(XYZ[iy], MM),
                  [], equal=True)
        fn = _kin_fn("tmj_left", "tmj_right")(ctx, task)
        ens = custom_ensemble(ctx, task, fn, ["lx", "ly", "lz", "rx", "ry", "rz"], MM)
        # Displacement from each cycle's start (condyle point at movement onset).
        if ens.n_trials:
            ens.data = ens.data - ens.data[:, :1, :]
            # Left and right excursions would cancel in a summary: left excursions only.
            if task == "left_right" and (ens.trials.side == "left").any():
                ens = ens.select((ens.trials.side == "left").to_numpy())
        d = resolve(ens, req.perspective, req.stat, pid, who, "path")
        if d.empty():
            panels.append(_empty(ttl, f"no {TASK_LABEL[task]} cycles"))
            continue
        persp = d.desc
        incl = []
        for side, off, rl in (("L", 0, "side:left"), ("R", 3, "side:right")):
            # Sides are named in the first panel's legend (labels at the paths collide).
            name = ("left" if side == "L" else "right") + " condyle point" \
                if not any(q.series for q in panels if isinstance(q, Panel)) else None
            for data, _, ok in d.trials:
                p.series.append(Series("line", data[:, off + ix], data[:, off + iy],
                                       role=rl if ok else "noise",
                                       style={"width": 0.5, "alpha": 0.4 if d.center is not None
                                              else 0.8},
                                       label=name if d.center is None and ok else None))
                if d.center is None and ok:
                    name = None
            for mp, m in d.members:
                p.series.append(Series("line", m[:, off + ix], m[:, off + iy], role="faint"))
            with E._quiet():
                src = d.center if d.center is not None else (
                    np.nanmedian(np.stack([x for x, _, ok in d.trials if ok]), axis=0)
                    if any(ok for _, _, ok in d.trials) else None)
            if d.center is not None:
                p.series.append(Series("line", d.center[:, off + ix], d.center[:, off + iy],
                                       role=rl, style={"width": 1.3}, label=name))
            if src is not None and iy == 2:
                incl.append(f"{side} {_inclination(src[:, off:off + 3]):.0f}°")
        if incl and task == "protrusion_retrusion":
            p.note = "inclination " + ", ".join(incl)
        panels.append(p)
        if incl and task == "protrusion_retrusion":
            notes.append("sagittal condylar path inclination to the x axis, start to most "
                         "anterior point")
    if task == "left_right":
        notes.append("side shift of the condyle points during laterotrusion to the left "
                     "(Bennett-type; horizontal plane)")
    fit_equal(panels, min_span=3.0, common=True)
    return FigureSpec(fid, title, panels, rows=1, cols=3, width="text", height_in=2.0,
                      caption=_caption(
                          "Paths of the skin-palpated condyle points (left blue, right orange) "
                          "as displacement from the movement onset of each cycle; thin lines "
                          "are cycles, thick lines the summary. " + FRAME_NOTE, persp,
                          "; ".join(notes)),
                      sources=["kinematics:arrays", "kinematics:cycles"], perspective=persp,
                      meta={"equal_scale": "shared", "pid": pid})


def _polar_median(hulls: list[np.ndarray], n: int = 90) -> np.ndarray | None:
    """Median envelope: per direction, the median hull radius about the common centre."""
    if len(hulls) < 2:
        return None
    c = np.mean([h[:-1].mean(axis=0) for h in hulls], axis=0)
    ang = np.linspace(-np.pi, np.pi, n, endpoint=False)
    R = []
    for h in hulls:
        r = np.full(n, np.nan)
        P = h - c
        for k, a in enumerate(ang):
            u = np.array([np.cos(a), np.sin(a)])
            best = np.nan
            for p0, p1 in zip(P[:-1], P[1:]):          # ray-segment intersection
                M = np.column_stack([u, p0 - p1])
                if abs(np.linalg.det(M)) < 1e-12:
                    continue
                s, w = np.linalg.solve(M, p0)
                if s > 0 and -1e-9 <= w <= 1 + 1e-9:
                    best = s if not np.isfinite(best) else max(best, s)
            r[k] = best
        R.append(r)
    with E._quiet():
        rm = np.nanmedian(np.array(R), axis=0)
    out = c + np.column_stack([np.cos(ang) * rm, np.sin(ang) * rm])
    return np.vstack([out, out[:1]])


@register("data_cohort_atlas", "data", "Cohort atlas", params=("phase", "perspective"),
          perspectives=("compare",), perspective="compare")
def cohort_atlas(req: ViewRequest) -> FigureSpec:
    """Small multiples of every participant's sagittal envelope plus a cohort overlay."""
    ctx = req.ctx
    pids = ctx.pids()
    fid, title = "data_cohort_atlas", "Sagittal envelopes, cohort atlas"
    tasks = [req.phase] if req.phase in TASKS else list(TASKS)
    panels, hulls = [], []
    for pid in pids:
        a = ctx.arrays("kinematics", pid)
        pts = [_arr(a, f"{t}__incisor")[:, [0, 2]] for t in _phases_present(a, tasks)]
        ttl = _who(ctx, pid)
        if not pts:
            panels.append(_empty(ttl, "no kinematics"))
            continue
        P = np.vstack(pts)
        p = Panel(ttl, Axis("x", MM), Axis("z", MM), [], equal=True)
        h = _hull(P)
        if h is not None:
            hulls.append(h)
            p.series.append(Series("fill", h[:, 0], h[:, 1], role="band", style={"alpha": 0.15}))
        for q in pts:
            keep = thin_index(q[:, 0], q[:, 1], budget=max(300, 1500 // len(pts)))
            p.series.append(Series("line", q[keep, 0], q[keep, 1], role="context"))
        panels.append(p)
    if not panels:
        return _empty_fig(fid, title, "no participants", "text")
    ov = Panel("Overlay", Axis("x", MM), Axis("z", MM), [], equal=True)
    for k, h in enumerate(hulls):
        ov.series.append(Series("line", h[:, 0], h[:, 1], role="context",
                                label="participant envelopes" if k == 0 else None))
    med = _polar_median(hulls)
    if med is not None:
        ov.series.append(Series("line", med[:, 0], med[:, 1], role="accent", style={"width": 1.3},
                                label="median envelope"))
    panels.append(ov)
    fit_equal(panels, same=True)
    cols = min(len(panels), 7)
    _grid_labels(panels, cols)
    rows = int(np.ceil(len(panels) / cols))
    persp = (f"small multiples per participant, n = {len(hulls)} participants, "
             f"{'all tasks' if len(tasks) > 1 else TASK_LABEL[tasks[0]]}")
    return FigureSpec(fid, title, panels, rows=rows, cols=cols, width="text", height_in=1.5,
                      caption=_caption(
                          "Sagittal incisal envelope (convex hull of all incisal positions) of "
                          "each participant on shared axes, and all envelopes overlaid with the "
                          "median envelope (median hull radius per direction about the common "
                          "centre). " + FRAME_NOTE, persp, "the median envelope"),
                      sources=["kinematics:arrays"], perspective=persp,
                      meta={"equal_scale": "shared"})


def _activation_fn(ctx, task):
    """fn(pid): incisor xyz + openers/closers activation (% of max recorded) interpolated onto the kinematic frames."""
    def fn(pid):
        a = ctx.arrays("kinematics", pid)
        t, P = a.get(f"{task}__t_us"), _arr(a, f"{task}__incisor")
        e = ctx.arrays("emg", pid)
        te, N = e.get(f"{task}__t_us"), _arr(e, f"{task}__norm_pct")
        names = _emg_names(ctx, pid, e)
        if t is None or P is None or te is None or N is None or not names:
            return None
        cols = []
        for group in (("digastric_l", "digastric_r"), ("masseter_l", "masseter_r")):
            idx = [names.index(m) for m in group if m in names]
            if not idx:
                return None
            with E._quiet():
                g = np.nanmean(N[:, idx], axis=1)
            cols.append(np.interp(t, te, g, left=np.nan, right=np.nan))
        return t, np.column_stack([P, *cols])
    return fn


@register("data_activation_path", "data", "Activation on the path", params=PARAMS,
          perspectives=("trials", "trials+summary", "participant", "cohort"),
          aligns=("time", "path"), perspective="trials", align="time", phase="open_close")
def activation_path(req: ViewRequest) -> FigureSpec:
    """Sagittal incisal path coloured by the activation of the openers (digastric) and closers (masseter)."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_activation_path", "Muscle activation along the path"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase if req.phase in TASKS else "open_close"
    pa = _plane_panel("(a) Digastric", "sagittal")
    pb = _plane_panel("(b) Masseter", "sagittal")
    lines: list[np.ndarray] = []         # (N, 5): x, y, z, openers, closers
    context: list[np.ndarray] = []
    if req.align == "time" and req.perspective == "trials":
        got = _activation_fn(ctx, task)(pid)
        if got is None:
            return FigureSpec(fid, title, [_empty(pa.title, "no EMG or kinematics"),
                                           _empty(pb.title, "no EMG or kinematics")],
                              rows=1, cols=2, width="column", height_in=1.4,
                              caption=f"{title}: no EMG or kinematics for {who}.")
        t, X = got
        lines.append(X[_window(req, (t - t[0]) * 1e-6)])
        persp = f"{who}, {TASK_LABEL[task]}, all frames"
    else:
        ens = custom_ensemble(ctx, task, _activation_fn(ctx, task),
                              ["x", "y", "z", "openers", "closers"], PCT, f"{task}/activation")
        d = resolve(ens, req.perspective, req.stat, pid, who, "path")
        if d.empty():
            return _empty_fig(fid, title, f"no {TASK_LABEL[task]} cycles with EMG")
        if d.level == "trials":
            lines += [x for x, _, ok in d.trials]
        elif d.level == "trials+summary":
            context += [x for x, _, _ in d.trials]
            lines.append(d.center)
        elif d.level == "participant":
            for mp, m in d.members:
                (lines if mp == d.highlight else context).append(m)
        else:
            context += [m for _, m in d.members]
            lines.append(d.center)
        persp = d.desc + f", {TASK_LABEL[task]}"
    vals = np.concatenate([x[:, 3:5].ravel() for x in lines]) if lines else np.array([])
    vals = vals[np.isfinite(vals)]
    lim = (0.0, _nice_cap(float(np.percentile(vals, 99))) if vals.size else 1.0)
    for pan, ch, on_top in ((pa, 3, True), (pb, 4, False)):
        for m in context:
            pan.series.append(Series("line", m[:, 0], m[:, 2], role="faint"))
        for X in lines:
            # Grey underlay: masked EMG leaves the path visible but uncoloured.
            pan.series.append(Series("line", X[:, 0], X[:, 2], role="faint"))
            # Opening and closing retrace nearly the same sagittal path: the stroke in which
            # the group works (openers: opening, closers: closing) is drawn on top.
            with np.errstate(invalid="ignore"):
                opening = np.gradient(X[:, 2]) < 0 if len(X) > 1 else np.ones(len(X), bool)
            opening = opening | np.r_[opening[1:], False]       # keep the joints
            for part in ((~opening, opening) if on_top else (opening, ~opening)):
                pan.series.append(Series("colorline", np.where(part, X[:, 0], np.nan),
                                         np.where(part, X[:, 2], np.nan), c=X[:, ch],
                                         style={"lim": lim}))
    pb.colorbar = Colorbar("Activation re max", PCT, lim=lim)
    fit_equal([pa, pb], min_span=12.0)
    return FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="column", height_in=2.2,
                      caption=_caption(
                          "Sagittal incisal path coloured by the activation (% of max recorded, mean of left "
                          "and right) of the jaw openers (digastric) and closers (masseter); "
                          "the stroke in which each group acts (opening, closing) is drawn on "
                          "top; "
                          "masked EMG samples leave the path grey. " + FRAME_NOTE, persp),
                      sources=["kinematics:arrays", "emg:arrays"], perspective=persp,
                      meta={"equal_scale": "shared", "pid": pid})


# == 2. kinematics, time ====================================================================
STACK = (  # (key, title, axis label, unit, scale, kind)
    ("theta", "(a) Opening angle", "Angle", DEG, 1.0, "theta_deg"),
    ("disp", "(b) Incisal displacement", "Displacement", MM, 1.0, "incisor"),
    ("theta_vel", "(c) Angular velocity", "Velocity", DPS, 1.0, "theta_vel"),
    ("speed", "(d) Incisal speed", "Speed", MMS, 1.0, "incisor_vel"),
    ("theta_acc", "(e) Angular acceleration", "Acceleration", DPS2, 1.0, "theta_acc"),
    ("acc", "(f) Incisal acceleration magnitude", "Acceleration", MS2, 1e-3, "incisor_acc"),
)


def _stack_series(a: dict[str, np.ndarray], task: str, key: str) -> np.ndarray | None:
    if key in ("theta", "theta_vel", "theta_acc"):
        return _arr(a, f"{task}__{dict(theta='theta_deg').get(key, key)}")
    if key == "disp":
        P, cf = _arr(a, f"{task}__incisor"), _closed_frame(a)
        if P is None or cf is None:
            return None
        ref = _arr(a, f"{cf[0]}__incisor")[cf[1]]
        return np.linalg.norm(P - ref, axis=1)
    if key == "speed":
        V = _arr(a, f"{task}__incisor_vel")
        return None if V is None else np.linalg.norm(V, axis=1)
    A = _arr(a, f"{task}__incisor_acc")
    return None if A is None else np.linalg.norm(A, axis=1) * 1e-3


@register("data_kinematic_stack", "data", "Kinematic stack", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("time", "cycle"), perspective="trials", align="time",
          phase="open_close")
def kinematic_stack(req: ViewRequest) -> FigureSpec:
    """Opening angle, angular velocity and acceleration next to incisal displacement, speed, |a|."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_kinematic_stack", "Kinematic stack"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    task = req.phase if req.phase in TASKS else "open_close"
    a = ctx.arrays("kinematics", pid)
    panels = []
    if req.align == "time" and req.perspective == "trials":
        t = a.get(f"{task}__t_us")
        if t is None:
            return _empty_fig(fid, title, f"no {TASK_LABEL[task]} kinematics for {who}", "text")
        t0 = float(t[0])
        ts = (np.asarray(t, float) - t0) * 1e-6
        sel = _window(req, ts)
        cyc = _cycles(ctx, task)
        cyc = cyc[cyc.pid == pid] if not cyc.empty else cyc
        for key, ttl, lab, unit, _, _ in STACK:
            y = _stack_series(a, task, key)
            xa = Axis("Time", SEC)
            if y is None:
                p = _empty(ttl, "not in the store (re-run kinematics)")
                p.sharex = "t"
                panels.append(p)
                continue
            p = Panel(ttl, xa, Axis(lab, unit), [], legend="none", sharex="t")
            if not cyc.empty:
                on = np.array([_bounds(c)[0] for _, c in cyc.iterrows()])
                pk = cyc.t_peak_us.to_numpy(float)
                keep = (on - t0) * 1e-6 <= ts[sel][-1]
                p.series.append(Series("hspan", (on[keep] - t0) * 1e-6, y2=(pk[keep] - t0) * 1e-6,
                                       role="band"))
            if key in ("theta_vel", "theta_acc"):
                p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
            p.series.append(Series("line", ts[sel], y[sel], role="accent", style={"width": 0.8}))
            panels.append(p)
        persp = f"{who}, {TASK_LABEL[task]}, absolute time"
        cap = ("Shaded: opening part (movement onset to peak) of each detected cycle.")
    else:
        persp = ""
        for i, (key, ttl, lab, unit, scale, _) in enumerate(STACK):
            if key == "disp":
                fn = _kin_fn("incisor")(ctx, task)
                ens = custom_ensemble(ctx, task, fn, ["x", "y", "z"], MM)
                if ens.n_trials:
                    ens.data = np.linalg.norm(ens.data - ens.data[:, :1], axis=2)[:, :, None]
            else:
                sig = {"theta": "theta", "theta_vel": "theta_vel", "theta_acc": "theta_acc",
                       "speed": "incisor_speed", "acc": "incisor_acc"}[key]
                ens = E.kinematic(ctx, task, sig, "cycle")
            d = resolve(ens, req.perspective, req.stat, pid, who, "cycle")
            if d.empty():
                p = _empty(ttl, "no cycles")
                p.sharex = "c"
                panels.append(p)
                continue
            persp = persp or d.desc
            p = Panel(ttl, Axis("Movement cycle", PCT), Axis(lab, unit), [], sharex="c")
            _wave(p, ens.grid, d, f=lambda v, s=scale: v * s, labels=i == len(STACK) - 2)
            if d.level == "trials" and i == 0:
                p.colorbar = d.colorbar()
            panels.append(p)
        persp += f", {TASK_LABEL[task]}"
        cap = "Cycles cut at movement onset and offset and time-normalized to 0-100 %."
    _bottom_xlabels(panels, 2)
    return FigureSpec(fid, title, panels, rows=3, cols=2, width="text", height_in=1.05,
                      caption=_caption(
                          "Opening angle (sagittal rotation from the closed reference), angular "
                          "velocity and acceleration (left) and incisal displacement from the "
                          "closed position, speed and acceleration magnitude (right). Derivatives "
                          "are Savitzky-Golay estimates matched to the 6 Hz pose bandwidth. "
                          + cap, persp),
                      sources=["kinematics:arrays", "kinematics:cycles"], perspective=persp,
                      meta={"pid": pid})


def min_jerk_velocity(tau: np.ndarray, amplitude: float, duration_s: float) -> np.ndarray:
    """Minimum-jerk velocity of a point-to-point move of ``amplitude`` over ``duration_s``
    at normalized time ``tau`` (0..1); it integrates to ``amplitude``."""
    s = np.clip(np.asarray(tau, float), 0.0, 1.0)
    return amplitude / duration_s * 30.0 * s ** 2 * (1.0 - s) ** 2


@register("data_cycle_profiles", "data", "Cycle-normalized profiles", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("cycle",), perspective="cohort", align="cycle",
          stat="median_iqr", phase="open_close")
def cycle_profiles(req: ViewRequest) -> FigureSpec:
    """Gait-style cycle profiles of opening angle, peak-normalized velocity (+ min jerk), accel."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_cycle_profiles", "Cycle-normalized kinematic profiles"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    th = E.kinematic(ctx, task, "theta", "cycle")
    vel = E.kinematic(ctx, task, "theta_vel", "cycle")
    acc = E.kinematic(ctx, task, "theta_acc", "cycle")
    if vel.n_trials:
        with np.errstate(all="ignore"):
            pk = np.nanmax(vel.data[:, :, 0], axis=1)
        vel.data = vel.data / np.where(pk > 0, pk, np.nan)[:, None, None]
    grid = th.grid
    panels, persp, meta = [], "", {"pid": pid}
    for i, (ens, ttl, ax) in enumerate((
            (th, "(a) Opening angle", Axis("Angle", DEG)),
            (vel, "(b) Angular velocity / peak", Axis("Normalized velocity")),
            (acc, "(c) Angular acceleration", Axis("Acceleration", DPS2)))):
        d = resolve(ens, req.perspective, req.stat, pid, who, "cycle")
        if d.empty():
            panels.append(_empty(ttl, "no cycles"))
            continue
        persp = persp or d.desc
        p = Panel(ttl, Axis("Movement cycle", PCT), ax, [], sharex="c")
        _wave(p, grid, d, labels=i == 1)
        if d.level == "trials" and i == 0:
            p.colorbar = d.colorbar()
        if i == 1:
            # Minimum-jerk reference: opening 0..p % and closing p..100 % with the same
            # amplitude; normalized like the data (by the opening peak).
            # The reference follows the summary shown: this participant's centre for the
            # trial levels, the highlighted participant, or the cohort centre.
            p.legend = "below"
            lvl = "trials+summary" if d.level == "trials" else d.level
            ref = resolve(th, lvl, req.stat, pid)
            mth = ref.center[:, 0] if ref.center is not None else next(
                (m[:, 0] for mp, m in ref.members if mp == pid), None)
            ok = th.ok()
            tr = ok.trials
            durs = (tr.duration_s[tr.pid == pid] if lvl != "cohort" else
                    tr.groupby("pid").duration_s.median()).to_numpy(float) if len(tr) else []
            if mth is not None and np.isfinite(mth).any():
                kp = int(np.nanargmax(mth))
                p_pct = float(grid[min(max(kp, 1), grid.size - 2)])
                A = float(mth[kp] - mth[0])
                dur = float(np.nanmedian(durs)) if len(durs) else float("nan")
                if not np.isfinite(dur) or dur <= 0:
                    dur = 1.0
                t_o, t_c = dur * p_pct / 100.0, dur * (1 - p_pct / 100.0)
                peak = 1.875 * A / t_o
                y = np.where(grid <= p_pct, min_jerk_velocity(grid / p_pct, A, t_o),
                             -min_jerk_velocity((grid - p_pct) / (100 - p_pct), A, t_c)) / peak
                p.series.append(Series("line", grid, y, role="reference", label="minimum jerk"))
                meta["min_jerk"] = {"amplitude_deg": A, "t_open_s": t_o, "t_close_s": t_c,
                                    "peak_dps": peak, "p_pct": p_pct, "duration_s": dur}
            p.series.append(Series("hline", y=np.array([0.0]), role="faint"))
        panels.append(p)
    persp += f", {TASK_LABEL[task]}"
    return FigureSpec(fid, title, panels, rows=1, cols=3, width="text", height_in=1.7,
                      caption=_caption(
                          "Movement cycles (onset to offset) time-normalized to 0-100 %. The "
                          "velocity of each cycle is divided by its peak opening velocity and "
                          "compared with the minimum-jerk profile of the same amplitude and "
                          "timing (dashed), the natural reference for exoskeleton trajectory "
                          "generation.", persp, "the minimum-jerk comparison"),
                      sources=["kinematics:arrays", "kinematics:cycles"], perspective=persp,
                      meta=meta)


@register("data_phase_portrait", "data", "Phase portrait", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("cycle",), perspective="trials+summary",
          align="cycle", stat="median_iqr", phase="open_close")
def phase_portrait(req: ViewRequest) -> FigureSpec:
    """Opening angle vs angular velocity (and vertical opening vs its velocity), loop per cycle."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_phase_portrait", "Phase portrait"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    panels, persp = [], ""
    for keys, ttl, ax, ay, flip in (
            (("theta_deg", "theta_vel"), "(a) Angular", Axis("Opening angle", DEG),
             Axis("Angular velocity", DPS), (1, 1)),
            (("incisor:z", "incisor_vel:z"), "(b) Incisal, vertical", Axis("Opening", MM),
             Axis("Opening velocity", MMS), (-1, -1))):
        ens = custom_ensemble(ctx, task, _kin_fn(*keys)(ctx, task), ["x", "v"])
        if ens.n_trials:
            ens.data = ens.data * np.array(flip, float)
            if flip[0] < 0:                      # opening from the cycle start
                ens.data[:, :, 0] -= ens.data[:, :1, 0]
        d = resolve(ens, req.perspective, req.stat, pid, who, "cycle")
        if d.empty():
            panels.append(_empty(ttl, "no cycles"))
            continue
        persp = persp or d.desc
        p = Panel(ttl, ax, ay, [], legend="auto")
        _path(p, d, 0, 1, labels=len(panels) == 0, ellipses=False)
        p.series.append(Series("hline", y=np.array([0.0]), role="faint"))
        if d.level == "trials" and not panels:
            p.colorbar = d.colorbar()
        panels.append(p)
    persp += f", {TASK_LABEL[task]}"
    return FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.0,
                      caption=_caption(
                          "Position against velocity, one loop per cycle: the upper half is "
                          "opening, the lower half closing. Loop overlap shows repeatability, "
                          "smooth ellipse-like loops smoothness, and the shape difference of the "
                          "halves opening/closing asymmetry.", persp),
                      sources=["kinematics:arrays", "kinematics:cycles"], perspective=persp,
                      meta={"pid": pid})


DEMAND = (  # row: (title, x key, y key, x label, y label)
    ("angular", "Angular velocity", DPS, "Angular acceleration", DPS2),
    ("incisal", "Incisal speed", MMS, "Tangential acceleration", MS2),
)
DEMAND_TASKS = ("open_close", "chewing", "free_motion")


def demand_values(a: dict[str, np.ndarray], task: str, kind: str
                  ) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-frame (x, y) of the demand map: (theta_vel, theta_acc) or (speed, tangential acc)."""
    if kind == "angular":
        v, acc = _arr(a, f"{task}__theta_vel"), _arr(a, f"{task}__theta_acc")
        if v is None or acc is None:
            return None
        x, y = v, acc
    else:
        V, A = _arr(a, f"{task}__incisor_vel"), _arr(a, f"{task}__incisor_acc")
        if V is None or A is None:
            return None
        x = np.linalg.norm(V, axis=1)
        with np.errstate(all="ignore"):
            y = np.einsum("ij,ij->i", V, A) / np.where(x > 1e-9, x, np.nan) * 1e-3
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def _hdr_contour(H: np.ndarray, xc: np.ndarray, yc: np.ndarray, frac: float = 0.95
                 ) -> np.ndarray | None:
    """Iso-line of the (smoothed) density enclosing ``frac`` of the frames, as a NaN-broken line."""
    import contourpy
    from scipy.ndimage import gaussian_filter

    S = gaussian_filter(H, 1.2)
    if S.sum() <= 0:
        return None
    v = np.sort(S.ravel())[::-1]
    c = np.cumsum(v) / v.sum()
    level = float(v[min(int(np.searchsorted(c, frac)), v.size - 1)])
    gen = contourpy.contour_generator(xc, yc, S.T)
    lines = gen.lines(level)
    if not lines:
        return None
    return _nanline([np.asarray(L, float) for L in lines if len(L) > 2])


@register("data_demand_map", "data", "Kinematic demand map", params=("pid", "perspective"),
          perspectives=("trials", "cohort"), perspective="trials")
def demand_map(req: ViewRequest) -> FigureSpec:
    """2-D density of velocity vs acceleration over all frames per task, with p95 limits."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_demand_map", "Kinematic demand map"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    cohort = req.perspective == "cohort"
    pids = ctx.pids() if cohort else [pid]
    arrays = {p: ctx.arrays("kinematics", p) for p in pids}
    tasks = [t for t in DEMAND_TASKS if any(f"{t}__t_us" in a for a in arrays.values())]
    if not tasks:
        return _empty_fig(fid, title, "no kinematics", "text")
    panels, meta = [], {"p95": {}, "pid": pid}
    for kind, xl, xu, yl, yu in DEMAND:
        row = []
        for task in tasks:
            vals = {p: demand_values(a, task, kind) for p, a in arrays.items()}
            vals = {p: v for p, v in vals.items() if v is not None and v[0].size}
            ttl = f"{TASK_LABEL[task]}, {kind}"
            if not vals:
                row.append(_empty(ttl, "no derivatives in the store", f"{xl}", f"{yl}"))
                continue
            X = np.concatenate([v[0] for v in vals.values()])
            Y = np.concatenate([v[1] for v in vals.values()])
            W = np.concatenate([np.full(v[0].size, 1.0 / v[0].size) for v in vals.values()])
            if cohort:
                p95x = float(np.median([np.percentile(np.abs(v[0]), 95) for v in vals.values()]))
                p95y = float(np.median([np.percentile(np.abs(v[1]), 95) for v in vals.values()]))
            else:
                p95x = float(np.percentile(np.abs(X), 95))
                p95y = float(np.percentile(np.abs(Y), 95))
            meta["p95"][(kind, task)] = (p95x, p95y)
            # Full range plus an empty margin, so the 95 % iso-line closes inside the map.
            mx, my = np.abs(X).max() * 1.2, np.abs(Y).max() * 1.2
            x0 = 0.0 if kind == "incisal" else -mx
            nb = 25 if X.size < 3000 else 40          # ~5+ frames per occupied cell
            xe, ye = np.linspace(x0, mx, nb + 1), np.linspace(-my, my, nb + 1)
            H, _, _ = np.histogram2d(X, Y, bins=[xe, ye], weights=W)
            H = 100.0 * H / H.sum()
            row.append((ttl, X, Y, H, xe, ye, p95x, p95y, kind, len(vals)))
        caps = [r[3].max() for r in row if isinstance(r, tuple)]
        lim = (0.0, _nice_cap(float(np.percentile([c for c in caps], 50)) * 0.5)) if caps \
            else (0.0, 1.0)
        for k, r in enumerate(row):
            if not isinstance(r, tuple):
                panels.append(r)
                continue
            ttl, X, Y, H, xe, ye, p95x, p95y, kind_, npart = r
            p = Panel(ttl, Axis(xl, xu), Axis(yl, yu), [], legend="none")
            # A colour-mapped map (heatmap in the viewer, NaN = empty cells transparent),
            # not a baked RGB picture: row 0 is the top (highest acceleration).
            img = np.where(H > 0, H, np.nan).T[::-1]
            p.series.append(Series("image", img=img, extent=(xe[0], xe[-1], ye[0], ye[-1]),
                                   style={"cmap": "sequential", "lim": lim}))
            xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
            C = _hdr_contour(H, xc, yc)
            if C is not None and len(C):
                p.series.append(Series("line", C[:, 0], C[:, 1], role="quaternary",
                                       style={"width": 0.8}))
            xs = [p95x] if kind_ == "incisal" else [-p95x, p95x]
            p.series.append(Series("vline", x=np.array(xs), role="reference"))
            p.series.append(Series("hline", y=np.array([-p95y, p95y]), role="reference"))
            from biomech.viz.style import plain_unit

            p.note = f"p95 {p95x:.3g} {plain_unit(xu)}, {p95y:.3g} {plain_unit(yu)}"
            if k == len(row) - 1:
                p.colorbar = Colorbar("Frames", "\\percent", lim=lim)
            panels.append(p)
    persp = (f"pooled frames of {len(pids)} participants, equal weight per participant; p95 = "
             "median of the participants' 95th percentiles" if cohort
             else f"{who}, all frames")
    return FigureSpec(fid, title, panels, rows=2, cols=len(tasks), width="text", height_in=1.6,
                      caption=_caption(
                          "Share of frames per velocity-acceleration cell (sequential ramp). "
                          "Vermilion line: the smallest region holding 95 % of the frames; "
                          "dashed (values top left, in the axis units): 95th percentiles of "
                          "|velocity| and |acceleration|, i.e. the "
                          "speed and acceleration an actuator must deliver for 95 % of the "
                          "observed motion.", persp, "all demand numbers"),
                      sources=["kinematics:arrays"], perspective=persp, meta=meta)


@register("data_imu_accel", "data", "Optical vs IMU acceleration",
          params=("pid", "phase", "window"), needs_session=True, perspectives=("trials",), aligns=("time",), perspective="trials", align="time",
          phase="open_close")
def imu_accel(req: ViewRequest) -> FigureSpec:
    """Relative linear acceleration of the mandible: optical prediction vs IMU, plus |a| - g."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_imu_accel", "Optical vs IMU acceleration"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase if req.phase in TASKS else "open_close"
    a, im = ctx.arrays("kinematics", pid), ctx.arrays("imu", pid)
    t_k, T = a.get(f"{task}__t_us"), _arr(a, f"{task}__T")
    t_i, acc = im.get(f"{task}__t_us"), _arr(im, f"{task}__acc_rel_ms2")
    titles = ("(a) Relative linear acceleration", "(b) Residual, IMU minus optical",
              "(c) Accelerometer |a| - 1 g")
    if t_k is None or T is None or t_i is None or acc is None:
        msg = "no IMU stage arrays" if t_i is None or acc is None else "no optical poses"
        return FigureSpec(fid, title, [_empty(s, msg) for s in titles], rows=3, cols=1,
                          width="column", height_in=0.9, caption=f"{title}: {msg} for {who}.")
    from biomech.kinematics.derivatives import derivatives

    t0 = float(t_k[0])
    _, A_opt = derivatives(t_k, T[:, :3, 3])
    a_opt = np.linalg.norm(A_opt, axis=1) * 1e-3
    ts_k = (np.asarray(t_k, float) - t0) * 1e-6
    ts_i = (np.asarray(t_i, float) - t0) * 1e-6
    a_opt_f = _lowpass(ts_k, a_opt)
    a_imu_f = _lowpass(ts_i, acc)
    a_imu_k = np.interp(ts_k, ts_i, a_imu_f, left=np.nan, right=np.nan)
    sel_k, sel_i = _window(req, ts_k), _window(req, ts_i)
    pa = Panel(titles[0], Axis("Time", SEC), Axis("|a|", MS2), [
        Series("line", ts_k[sel_k], a_opt_f[sel_k], role="reference",
               label="optical, mandibular array origin"),
        Series("line", ts_i[sel_i], a_imu_f[sel_i], role="accent", label="IMU")], sharex="t")
    res = a_imu_k - a_opt_f
    pb = Panel(titles[1], Axis("Time", SEC), Axis("Residual", MS2), [
        Series("hline", y=np.array([0.0]), role="faint"),
        Series("line", ts_k[sel_k], res[sel_k], role="accent")], sharex="t")
    rms = float(np.sqrt(np.nanmean(res[sel_k] ** 2))) if np.isfinite(res[sel_k]).any() else np.nan
    pb.note = f"RMS {rms:.2f} m/s²"
    pc = Panel(titles[2], Axis("Time", SEC), Axis("|a| - g", MS2), [], sharex="t")
    try:
        sd = ctx.emg(pid).motion
        _, phase = ctx.phase(pid, task)
        for slot, rl, lab in (("mta", "side:right", "mandible IMU"), ("cra", "context",
                                                                      "head IMU")):
            st = sd.imu(slot)
            if st is None or st.accel is None or phase is None:
                continue
            m = (st.accel.t_us >= phase.t_start_us) & (st.accel.t_us <= phase.t_end_us)
            tt = (st.accel.t_us[m].astype(float) - t0) * 1e-6
            g = np.linalg.norm(st.accel.data[m].astype(float), axis=1) - G0
            w = _window(req, tt)
            pc.series.append(Series("line", tt[w], _lowpass(tt, g)[w], role=rl, label=lab))
    except Exception:                              # session files absent: stage arrays only
        pass
    if not pc.series:
        pc = _empty(titles[2], "no raw accelerometer data")
        pc.sharex = "t"
    else:
        pc.series.insert(0, Series("hline", y=np.array([0.0]), role="faint"))
    _bottom_xlabels([pa, pb, pc], 1)
    return FigureSpec(fid, title, [pa, pb, pc], rows=3, cols=1, width="column", height_in=0.95,
                      caption=_caption(
                          "Acceleration of the mandible relative to the head, both low-passed at "
                          "6 Hz: the IMU's gravity-free linear acceleration (mandible minus head "
                          "sensor) against the optical prediction, the second derivative of the "
                          "mandibular-array origin in the maxillary frame. The IMU calibration "
                          "holds only rotations (no sensor lever arm), so the optical prediction "
                          "is evaluated at the array origin, not at the sensor; the residual "
                          "therefore includes the unknown lever-arm terms. (c): raw "
                          "accelerometer magnitude minus 1 g.", f"{who}, {TASK_LABEL[task]}",
                          "the optical-IMU residual"),
                      sources=["kinematics:arrays", "imu:arrays", "session:imu"],
                      perspective=f"{who}, {TASK_LABEL[task]}", meta={"pid": pid})


# == 3. EMG =================================================================================
def _emg_names(ctx, pid: str, e: dict[str, np.ndarray] | None = None) -> list[str]:
    e = e if e is not None else ctx.arrays("emg", pid)
    if "profile__names" in e:
        return [str(n) for n in e["profile__names"]]
    refs = ctx.table("emg", "refs", [pid])
    return list(refs.muscle) if not refs.empty else []


def _session_or_none(ctx, pid: str):
    """The motion take of a participant, or None (no recorded session, e.g. a snapshot)."""
    try:
        return ctx.emg(pid).motion
    except Exception:
        return None


def _qc_stored(ctx, pid: str, *tables: str):
    """qc tables of one participant (G2); empty frames for a store without them."""
    return [ctx.table("qc", t, [pid]) for t in tables]


def _emg_params(ctx):
    from biomech.emg.prepare import EmgParams

    return EmgParams.from_dict(ctx.config.signals.get("emg") or {})


def _chain_stored(ctx, pid: str, task: str, window_s) -> dict | None:
    """The stored chain example of the emg stage (G2), when it covers the request."""
    e = ctx.arrays("emg", pid)
    k = f"chain__{task}__"
    names = _emg_names(ctx, pid, e)
    if f"{k}t_us" not in e or not names:
        return None
    t_us = np.asarray(e[f"{k}t_us"], np.int64)
    t = (t_us.astype(float) - float(e[f"{k}t0_us"][0])) * 1e-6
    if window_s is not None and (window_s[0] < t[0] - 1e-3 or window_s[1] > t[-1] + 1e-3):
        return None
    lf = e.get(f"mains__{task}__line_hz")
    return {"t": t, "names": names, "raw": e[f"{k}raw"],
            "dem": e[f"{k}demained"], "cleaned": e[f"{k}cleaned"], "norm": e[f"{k}norm"],
            "mask": np.asarray(e[f"{k}masked"]).astype(bool),
            "f0": float(lf[0]) if lf is not None else float("nan"),
            "window": (float(window_s[0]), float(window_s[1])) if window_s is not None
            else (float(t[0]), float(t[-1]))}


def _chain_session(ctx, pid: str, task: str, window_s) -> dict | None:
    """The chain recomputed from the recorded session (any window; needs the session)."""
    from jtsmm.emg.mains import remove_mains

    from biomech.emg.analysis import chain_window

    prep, pe = _emg_phase(ctx, pid, task)
    if pe is None:
        return None
    t = (pe.t_us.astype(float) - pe.t_us[0]) * 1e-6
    w0, w1 = window_s if window_s is not None else chain_window(
        pe, ctx.table("kinematics", "cycles", [pid]), task)
    raw = pe.raw - np.median(pe.raw, axis=0)
    dem, f0 = remove_mains(raw, pe.fs, t_us=pe.t_us,
                           line_hz=prep.phase_line_hz(prep.motion, task),
                           robust=prep.p.robust_mains)
    return {"t": t, "names": list(pe.names), "raw": raw, "dem": dem, "cleaned": pe.cleaned,
            "norm": pe.norm, "mask": np.asarray(pe.mask, bool),
            "f0": float("nan") if f0 is None else float(f0), "window": (w0, w1)}


def _emg_phase(ctx, pid: str, phase: str):
    """PhaseEmg of a participant (raw, cleaned, masks, envelope, % of max recorded) or None."""
    try:
        prep = ctx.emg(pid)
        if not prep.names or prep.motion.phase(phase) is None:
            return None, None
        return prep, prep.phase(phase)
    except Exception:
        return None, None


@register("data_emg_chain", "data", "EMG processing chain", params=("pid", "phase", "window"),
          perspectives=("trials",), aligns=("time",), perspective="trials", align="time",
          phase="open_close")
def emg_chain(req: ViewRequest) -> FigureSpec:
    """Raw -> mains-removed -> band-passed -> envelope (% of max recorded) for one channel over 2 s."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_emg_chain", "EMG processing chain"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase or "open_close"
    titles = ("(a) Raw", "(b) Mains removed", "(c) Band-passed", "(d) Envelope")
    # The stored example (G2) when it covers the request; else the recorded session.
    ch_ = _chain_stored(ctx, pid, task, req.window_s)
    src = "emg:chain"
    if ch_ is None:
        ch_, src = _chain_session(ctx, pid, task, req.window_s), "session:emg"
    if ch_ is None:
        return FigureSpec(fid, title, [_empty(s, "no EMG") for s in titles], rows=4, cols=1,
                          width="column", height_in=0.8,
                          caption=f"{title}: no stored example and no recorded session for "
                                  f"{who}, {TASK_LABEL.get(task, task)}.")
    ep = _emg_params(ctx)
    t, names = ch_["t"], ch_["names"]
    w0, w1 = ch_["window"]
    sel = (t >= w0) & (t <= w1)
    norm_all, mask_all = ch_["norm"], ch_["mask"]
    ch = req.options.get("channel")
    if ch not in names:
        # Default: the most active channel in the window among those with at most 10 %
        # masked samples (else the least masked), so the example shows bursts.
        frac = {n: float(mask_all[sel, i].mean()) if sel.any() else 1.0
                for i, n in enumerate(names)}
        with np.errstate(all="ignore"):
            act = {n: float(np.nan_to_num(np.nanpercentile(norm_all[sel, i], 90)))
                   if sel.any() and np.isfinite(norm_all[sel, i]).any() else 0.0
                   for i, n in enumerate(names)}
        good = [n for n in names if frac[n] <= 0.1]
        ch = max(good, key=lambda n: act[n]) if good else min(names, key=lambda n: frac[n])
    ci = names.index(ch)
    raw, dem = np.asarray(ch_["raw"][:, ci], float), np.asarray(ch_["dem"][:, ci], float)
    cleaned, norm = np.asarray(ch_["cleaned"][:, ci], float), np.asarray(norm_all[:, ci], float)
    mask, f0 = mask_all[:, ci], ch_["f0"]
    panels = []
    # One amplitude scale for the three signal rows, so each step's effect is visible.
    with np.errstate(all="ignore"):
        amp = _nice_cap(float(np.nanmax(np.abs(np.r_[raw[sel], dem[sel]]))) * 1.05) \
            if sel.any() else 1.0
    for ttl, y, unit, rl in ((titles[0], raw, UV, "context"), (titles[1], dem, UV, "context"),
                             (titles[2], cleaned, UV, "accent")):
        panels.append(Panel(ttl, Axis("Time", SEC), Axis("EMG", unit, lim=(-amp, amp)),
                            [Series("line", t[sel], y[sel], role=rl, style={"width": 0.5})],
                            sharex="t"))
    pd_ = Panel(titles[3], Axis("Time", SEC), Axis("Activation re max", PCT), [],
                sharex="t", legend="auto")
    runs = [(a_, b_) for a_, b_ in _runs(mask & sel)]
    if runs:
        pd_.series.append(Series("hspan", np.array([t[a_] for a_, _ in runs]),
                                 y2=np.array([t[b_ - 1] for _, b_ in runs]), role="threshold"))
    pd_.series.append(Series("line", t[sel], norm[sel], role="accent"))
    n_m = int((mask & sel).sum())
    if n_m:
        pd_.note = f"masked {100 * n_m / max(int(sel.sum()), 1):.1f} % (shaded)"
    panels.append(pd_)
    _bottom_xlabels(panels, 1)
    persp = f"{who}, {MUSCLE_LABEL.get(ch, ch)}, {TASK_LABEL.get(task, task)}, {w1 - w0:.0f} s"
    return FigureSpec(fid, title, panels, rows=4, cols=1, width="column", height_in=0.8,
                      caption=_caption(
                          f"One channel through the offline chain: raw (median removed); after "
                          f"removing the {f0:.1f} Hz mains fundamental and harmonics (robust "
                          f"fit); band-passed {ep.bandpass_hz[0]:.0f}-"
                          f"{ep.bandpass_hz[1]:.0f} Hz (zero phase); the masked RMS "
                          f"envelope ({1000 * ep.envelope_s:.0f} ms) in % of the largest "
                          "activity recorded for the channel. Shaded: samples excluded by "
                          "the analysis mask (rail, lead-off, envelope).", persp),
                      sources=[src, "emg:refs"], perspective=persp,
                      meta={"pid": pid, "channel": ch})


def median_frequency(f: np.ndarray, P: np.ndarray, band: tuple[float, float]) -> float:
    m = (f >= band[0]) & (f <= band[1])
    if not m.any() or not np.isfinite(P[m]).all() or P[m].sum() <= 0:
        return float("nan")
    c = np.cumsum(P[m])
    return float(f[m][np.searchsorted(c, 0.5 * c[-1])])


@register("data_emg_spectra", "data", "EMG spectra", params=("pid", "phase"),
          perspectives=("trials",), perspective="trials", phase="open_close")
def emg_spectra(req: ViewRequest) -> FigureSpec:
    """Welch PSD per channel before and after mains removal, harmonics and median frequency."""
    from scipy.signal import welch

    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_emg_spectra", "EMG spectra"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    task = req.phase or "open_close"
    e = ctx.arrays("emg", pid)
    src = "emg:mains"
    if f"mains__{task}__raw_psd" in e:
        # Stored by the emg stage (G2): (C, F) PSDs on mains__freq.
        f = np.asarray(e["mains__freq"], float)
        P_raw = np.asarray(e[f"mains__{task}__raw_psd"], float).T
        P_dem = np.asarray(e[f"mains__{task}__demained_psd"], float).T
        f0 = float(e[f"mains__{task}__line_hz"][0])
        names = _emg_names(ctx, pid, e)
        fs_half = float(f[-1])
    else:
        from jtsmm.emg.mains import remove_mains

        prep, pe = _emg_phase(ctx, pid, task)
        if pe is None:
            return _empty_fig(fid, title, f"no stored spectra and no session for {who}",
                              "text")
        raw = pe.raw - np.median(pe.raw, axis=0)
        dem, f0 = remove_mains(raw, pe.fs, t_us=pe.t_us,
                               line_hz=prep.phase_line_hz(prep.motion, task),
                               robust=prep.p.robust_mains)
        nper = int(min(pe.fs, raw.shape[0]))
        f, P_raw = welch(raw, pe.fs, nperseg=nper, axis=0)
        _, P_dem = welch(dem, pe.fs, nperseg=nper, axis=0)
        names, fs_half, src = list(pe.names), pe.fs / 2, "session:emg"
    band = _emg_params(ctx).bandpass_hz
    fmax = min(500.0, fs_half)
    m = f <= fmax
    db = lambda P: 10 * np.log10(np.maximum(P, 1e-12))       # noqa: E731
    order = [n for n in MUSCLES if n in names] + [n for n in names if n not in MUSCLES]
    panels, mdfs = [], {}
    # The detected mains frequency can be missing (NaN) or 0 (nothing removed): no harmonic
    # lines then, and the caption says so.
    mains_ok = f0 is not None and np.isfinite(f0) and f0 > 0
    harm = np.arange(1, int(fmax // f0) + 1) * f0 if mains_ok else np.empty(0)
    leg_k = 3 * ((len(order) - 1) // 3)            # first panel of the last row
    for k, name in enumerate(order):
        ci = names.index(name)
        mdf = median_frequency(f, P_dem[:, ci], band)
        mdfs[name] = mdf
        p = Panel(MUSCLE_LABEL.get(name, name), Axis("Frequency", HZ, lim=(0.0, fmax)),
                  Axis("PSD", "\\decibel"), ([Series("vline", x=harm, role="faint")]
                                              if harm.size else []) + [
                      Series("line", f[m], db(P_raw[m, ci]), role="context",
                             label="raw" if k == leg_k else None),
                      Series("line", f[m], db(P_dem[m, ci]), role="accent", style={"width": 0.8},
                             label="mains removed" if k == leg_k else None)],
                  legend="below" if k == leg_k else "none")
        p.note = f"MDF {mdf:.0f} Hz"
        panels.append(p)
    _grid_labels(panels, 3)
    persp = f"{who}, {TASK_LABEL.get(task, task)}, whole phase"
    return FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 3)), cols=3,
                      width="text", height_in=1.25,
                      caption=_caption(
                          "Welch power spectral density (1-s segments, dB re 1 µV²/Hz) of the "
                          "raw signal and after mains removal "
                          + (f"({f0:.1f} Hz and harmonics, dotted)" if mains_ok else
                             "(no mains frequency detected: nothing removed)")
                          + ". MDF: median frequency of the mains-removed spectrum in the "
                          f"{band[0]:.0f}-{band[1]:.0f} Hz band."
                          + (f" Grey verticals: the {f0:.1f} Hz mains frequency and its "
                             "harmonics." if mains_ok else ""), persp,
                          "median frequencies"),
                      sources=[src], perspective=persp,
                      meta={"pid": pid, "line_hz": f0, "mdf_hz": mdfs})


@register("data_activation_raster", "data", "Activation raster", params=("pid",),
          perspectives=("trials",), perspective="trials", bin_s=0.25)
def activation_raster(req: ViewRequest) -> FigureSpec:
    """Muscles x time heatmap of the activation (% of max recorded) over every task of the session."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_activation_raster", "Activation raster"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    e = ctx.arrays("emg", pid)
    names = _emg_names(ctx, pid, e)
    tasks = [t for t in ("rest",) + TASKS if f"{t}__t_us" in e and f"{t}__norm_pct" in e]
    if not tasks or not names:
        return _empty_fig(fid, title, f"no EMG for {who}", "text")
    order = [n for n in MUSCLES if n in names]
    b = float(req.options.get("bin_s", 0.25))
    cols, ticks, labels, x = [], [], [], 0.0
    gap = 1.0
    for t in tasks:
        tt = (e[f"{t}__t_us"].astype(float) - float(e[f"{t}__t_us"][0])) * 1e-6
        N = np.asarray(e[f"{t}__norm_pct"], float)
        edges = np.arange(0.0, tt[-1] + b, b)
        idx = np.digitize(tt, edges) - 1
        Z = np.full((len(order), len(edges) - 1), np.nan)
        for j in range(len(edges) - 1):
            m = idx == j
            if m.any():
                with E._quiet():
                    Z[:, j] = np.nanmean(N[m][:, [names.index(n) for n in order]], axis=0)
        cols.append(Z)
        ticks.append(x + 0.5 * Z.shape[1] * b)
        labels.append(TASK_LABEL[t])
        x += Z.shape[1] * b
        if t != tasks[-1]:
            cols.append(np.full((len(order), int(round(gap / b))), -1.0))   # gap marker
            x += gap
    Z = np.hstack(cols)
    vals = Z[np.isfinite(Z) & (Z >= 0)]
    lim = (0.0, _nice_cap(float(np.percentile(vals, 99))) if vals.size else 1.0)
    # Colour-mapped scalar map: masked bins (NaN inside a task) in the muted grey, the gaps
    # between tasks transparent (the panel shows through in either theme).
    img = np.where(Z >= 0, Z, np.nan)
    masked = ~np.isfinite(Z)
    n = len(order)
    p = Panel("", Axis("Task (session order)"), Axis(""), [
        Series("image", img=img, extent=(0.0, x, -0.5, n - 0.5),
               style={"cmap": "sequential", "lim": lim, "mask": masked})], legend="none",
        colorbar=Colorbar("Activation re max", PCT, lim=lim))
    p.x.ticks, p.x.ticklabels, p.x.lim = ticks, labels, (0.0, x)
    p.y.ticks = [float(n - 1 - i) for i in range(n)]
    p.y.ticklabels = [MUSCLE_LABEL.get(m, m) for m in order]
    p.y.lim = (-0.5, n - 0.5)
    persp = f"{who}, all tasks, {b:.2f}-s bins"
    return FigureSpec(fid, title, [p], width="text", height_in=0.22 * n + 0.3,
                      caption=_caption(
                          "Activation (% of max recorded, mean per bin) of every muscle over every task "
                          "in session order; empty gaps separate the tasks, grey bins are "
                          "masked samples.", persp),
                      sources=["emg:arrays"], perspective=persp, meta={"pid": pid, "lim": lim})


PAIRS = (("digastric", "(b) Digastric"), ("masseter", "(c) Masseter"),
         ("temporalis", "(d) Temporalis"))


def _profile_ensembles(ctx, task: str) -> tuple[E.Ensemble, E.Ensemble]:
    return E.profile_ensembles(ctx, task)


@register("data_emg_cycle", "data", "Cycle-aligned activation", params=PARAMS,
          perspectives=WAVE_PERSP, aligns=("cycle",), perspective="trials+summary",
          align="cycle", phase="open_close")
def emg_cycle(req: ViewRequest) -> FigureSpec:
    """Opener and closer envelopes over the normalized cycle, with the opening angle; L/R pairs."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_emg_cycle", "Cycle-aligned activation"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    emg, theta = _profile_ensembles(ctx, task)
    if emg.n_trials == 0:
        return _empty_fig(fid, title, f"no {TASK_LABEL[task]} EMG profiles")
    grid = emg.grid
    panels = []
    d = resolve(theta, req.perspective, req.stat, pid, who, "cycle")
    p = Panel("(a) Opening angle", Axis("Movement cycle", PCT), Axis("Angle", DEG), [],
              sharex="c")
    _wave(p, grid, d)
    if d.level == "trials":
        p.colorbar = d.colorbar()
    panels.append(p)
    persp = d.desc
    for i, (m, ttl) in enumerate(PAIRS):
        q = Panel(ttl, Axis("Movement cycle", PCT), Axis("Activation re max", PCT), [],
                  sharex="c")
        for side, rl in (("l", "side:left"), ("r", "side:right")):
            name = f"{m}_{side}"
            if name not in emg.channels:
                continue
            dm = resolve(emg.channel(name), req.perspective, req.stat, pid, who, "cycle")
            last = i == len(PAIRS) - 1
            _wave(q, grid, dm, role=rl, band=False,
                  label=("left" if side == "l" else "right") if last else None)
            if dm.center is None and last:
                q.series.append(Series("line", np.array([np.nan]), np.array([np.nan]), role=rl,
                                       label="left" if side == "l" else "right"))
        finite = any(np.isfinite(np.asarray(sr.y, float)).any() for sr in q.series
                     if sr.y is not None)
        panels.append(q if finite else _empty(ttl, "no channels" if not q.series
                                              else "all samples masked"))
    _bottom_xlabels(panels, 1)
    return FigureSpec(fid, title, panels, rows=4, cols=1, width="column", height_in=0.85,
                      caption=_caption(
                          "Envelopes (% of max recorded) of the openers (digastric) and closers "
                          "(masseter, temporalis), time-normalized to the cycle (start to "
                          "end, as stored by the EMG stage) "
                          "together with the opening angle of the same cycles; left (blue) and "
                          "right (orange) to read symmetry.", persp + f", {TASK_LABEL[task]}"),
                      sources=["emg:arrays", "kinematics:cycles"],
                      perspective=persp + f", {TASK_LABEL[task]}", meta={"pid": pid})


# == 4. IMU =================================================================================
@register("data_imu_estimators", "data", "IMU opening-angle estimators", params=("pid", "phase", "window"),
          perspectives=("trials",), aligns=("time",), perspective="trials", align="time",
          phase="open_close")
def imu_estimators(req: ViewRequest) -> FigureSpec:
    """IMU opening angle (quaternion, tilt, complementary) vs optical, the difference, dθ/dt."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_imu_estimators", "IMU opening-angle estimators"
    if pid is None:
        return _empty_fig(fid, title, "no participants")
    who = _who(ctx, pid)
    task = req.phase if req.phase in ("rest",) + TASKS else "open_close"
    im = ctx.arrays("imu", pid)
    t = im.get(f"{task}__t_us")
    titles = ("(a) Opening angle", "(b) Estimator minus optical", "(c) Opening rate (quaternion)")
    if t is None:
        return FigureSpec(fid, title, [_empty(s, "no IMU stage arrays") for s in titles], rows=3,
                          cols=1, width="column", height_in=0.9,
                          caption=f"{title}: no IMU data for {who}.")
    ts = (np.asarray(t, float) - float(t[0])) * 1e-6
    sel = _window(req, ts)
    opt = _arr(im, f"{task}__theta_opt")
    est = (("quat", "quaternion", "cat:0"), ("tilt", "tilt only", "cat:1"),
           ("comp", "complementary", "cat:2"))
    pa = Panel(titles[0], Axis("Time", SEC), Axis("Angle", DEG), [], sharex="t")
    pb = Panel(titles[1], Axis("Time", SEC), Axis("Difference", DEG),
               [Series("hline", y=np.array([0.0]), role="faint")], sharex="t", legend="none")
    for k, lab, rl in est:
        v = _arr(im, f"{task}__theta_{k}")
        if v is None:
            continue
        pa.series.append(Series("line", ts[sel], v[sel], role=rl, style={"width": 0.8},
                                label=lab))
        if opt is not None:
            pb.series.append(Series("line", ts[sel], (v - opt)[sel], role=rl,
                                    style={"width": 0.8}))
    if opt is not None:
        pa.series.append(Series("line", ts[sel], opt[sel], role="reference", label="optical"))
    w = _arr(im, f"{task}__omega_dps")
    pc = (Panel(titles[2], Axis("Time", SEC), Axis("Rate", DPS),
                [Series("hline", y=np.array([0.0]), role="faint"),
                 Series("line", ts[sel], w[sel], role="cat:0", style={"width": 0.8})], sharex="t")
          if w is not None else _empty(titles[2], "not in the store"))
    persp = f"{who}, {TASK_LABEL.get(task, task)}"
    _bottom_xlabels([pa, pb, pc], 1)
    return FigureSpec(fid, title, [pa, pb, pc], rows=3, cols=1, width="column", height_in=0.95,
                      caption=_caption(
                          "Sagittal opening angle from the head and mandible IMUs (quaternion "
                          "relative rotation, accelerometer tilt only, and their complementary "
                          "fusion) against the optical reference (dashed), the difference, and "
                          "the opening rate from the quaternion estimate (no gyroscope was "
                          "recorded).", persp),
                      sources=["imu:arrays"], perspective=persp, meta={"pid": pid})


@register("data_imu_quality", "data", "IMU sample intervals and coverage", params=("pid",),
          perspectives=("trials",), perspective="trials")
def imu_quality(req: ViewRequest) -> FigureSpec:
    """Sample-interval histogram per IMU stream and coverage per phase."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_imu_quality", "IMU sample intervals and coverage"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    iv, cv = _qc_stored(ctx, pid, "imu_intervals", "imu_coverage")
    src = "qc:imu_intervals"
    if iv.empty:
        from biomech.qc.streams import imu_streams

        sd = _session_or_none(ctx, pid)
        iv, cv = imu_streams(sd, pid) if sd is not None else (iv, cv)
        src = "session:imu"
    if iv.empty:
        return _empty_fig(fid, title, f"no IMU streams for {who}", "text")
    iv = iv.sort_values("order")
    streams = list(iv.stream)
    # Interval distribution per stream: 1st-99th percentile (bar), median (diamond),
    # longest interval (dot); the y labels carry the rate and the share of gaps.
    pa = Panel("(a) Sample intervals", Axis("Interval", MSEC, log=True), Axis(""), [],
               legend="below")
    ylabels, lo_all, hi_all = [], [], []
    for k, r in enumerate(iv.itertuples()):
        med = getattr(r, "interval_median_ms", np.nan)
        if not np.isfinite(med):
            ylabels.append(f"{r.stream}, no data")
            continue
        q1, q99, mx = r.interval_p01_ms, r.interval_p99_ms, r.interval_max_ms
        nm = int(getattr(r, "nonmono_n", 0) or 0)
        ylabels.append(f"{r.stream} {r.rate_hz:.0f} Hz" + (f", {nm} out of order" if nm else ""))
        y = np.array([float(k)])
        pa.series += [
            Series("line", np.array([q1, q99]), np.r_[y, y], role="context",
                   style={"width": 4.0}, label="1st-99th percentile" if k == 0 else None),
            Series("scatter", np.array([med]), y, role="accent", style={"ms": "d"},
                   label="median" if k == 0 else None),
            Series("scatter", np.array([mx]), y, role="quaternary",
                   label="longest" if k == 0 else None),
            Series("text", np.array([mx * 1.4]), y,
                   text=[f"{r.long_interval_pct:.1f} % long"])]
        lo_all.append(q1)
        hi_all.append(mx)
    if lo_all:
        pa.x.lim = (min(lo_all) / 1.6, max(max(hi_all), 2 * min(lo_all)) * 12.0)
    pa.y.ticks, pa.y.ticklabels = [float(k) for k in range(len(streams))], ylabels
    pa.y.lim, pa.y.reverse = (-0.6, len(streams) - 0.4), True
    phases = list(dict.fromkeys(cv.sort_values("phase_order").phase)) if not cv.empty else []
    Z = np.full((len(phases), len(streams)), np.nan)          # phases x streams
    for r in cv.itertuples():
        Z[phases.index(r.phase), streams.index(r.stream)] = r.coverage_pct
    pb = Panel("(b) Coverage per phase (%)", Axis(""), Axis(""), [Series(
        "heatmap", z=Z, text=[f"{v:.0f}" if np.isfinite(v) else "" for v in Z.ravel()])],
        legend="none", colorbar=Colorbar("Coverage", "\\percent", lim=(80.0, 100.0),
                                         ticks=[80.0, 90.0, 100.0]))
    pb.y.ticks, pb.y.ticklabels = list(range(len(phases))), [TASK_LABEL.get(ph, ph)
                                                             for ph in phases]
    pb.x.ticks = list(range(len(streams)))
    pb.x.ticklabels = [s_.replace(" quat", "-Q").replace(" accel", "-A") for s_ in streams]
    return FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=1.7,
                      caption=_caption(
                          "Data quality of the IMU streams: distribution of the intervals between "
                          "consecutive samples (log axis; % long: intervals longer than 1.5 "
                          "times the median, i.e. dropped samples or clock jitter) and the share of expected samples present in each phase "
                          "(Q quaternion, A accelerometer stream).",
                          f"{who}, motion take"),
                      sources=[src, "qc:imu_coverage"], perspective=f"{who}, motion take",
                      meta={"pid": pid})


# == 5. tracking and synchronization ========================================================
@register("data_timeline", "data", "Session timeline", params=("pid",), perspectives=("trials",),
          perspective="trials")
def timeline(req: ViewRequest) -> FigureSpec:
    """Swimlanes of every stream on the master clock: coverage, gaps, rates, lock, phases."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_timeline", "Session timeline"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    lt, sg = _qc_stored(ctx, pid, "timeline_lanes", "timeline_segments")
    arr = ctx.arrays("qc", pid)
    src = "qc:timeline_lanes"
    if lt.empty:
        from biomech.qc.streams import timeline as stream_timeline

        sd = _session_or_none(ctx, pid)
        if sd is None:
            return _empty_fig(fid, title, f"no stored timeline and no session for {who}",
                              "text")
        lt, sg, arr = stream_timeline(sd, pid)
        src = "session"
    lanes: list[tuple[str, list[Series]]] = []
    rms_max = 0.0
    calib_labelled = stagger = False
    for ln in lt.sort_values("order").itertuples():
        seg = sg[sg.lane == ln.lane]
        if ln.kind == "phases":
            ser, tx, tt = [], [], []
            for r in seg.itertuples():
                x, y = _seg_line([(r.t0_s, r.t1_s)], 0.0)
                role = V.phase_role(r.phase)
                lab = TASK_LABEL.get(r.phase) if role != "faint" else None
                if role == "faint" and not calib_labelled:
                    lab, calib_labelled = "calibration / landmark holds", True
                ser.append(Series("line", x, y, role=role, style={"width": 5.0}, label=lab))
                tx.append(0.5 * (r.t0_s + r.t1_s))
                tt.append(V.phase_label(r.phase, short=True))
            if tx:      # every phase labelled on the lane (holds have no hue of their own)
                # labels of neighbouring short phases alternate between two rows (as the
                # phase lane of the validation views) so that e.g. "TMJ L"/"TMJ R" never touch
                span = max(float(seg.t1_s.max() - seg.t0_s.min()), 1e-9)
                ty, row = [], 0
                for r in seg.itertuples():
                    row = (row + 1) % 2 if (r.t1_s - r.t0_s) < 0.04 * span else 0
                    ty.append(-0.38 if row == 0 else -0.86)
                    stagger |= row == 1
                ser.append(Series("text", np.array(tx), np.array(ty), text=tt,
                                  style={"anchor": "above", "color": "ink"}))
            lanes.append(("phases", ser))
            continue
        if seg.empty:
            lanes.append((f"{ln.lane} (none)", []))
            continue
        x, y = _seg_line(list(zip(seg.t0_s, seg.t1_s)), 0.0)
        role = "context" if ln.kind in ("previews", "triggers") else "accent"
        ser = [Series("line", x, y, role=role, style={"width": 5.0})]
        label = f"{ln.lane} {ln.rate_hz:.0f} Hz"
        if ln.kind == "optical":
            body = ln.lane.split()[-1].lower()
            label += f", lock {ln.lock_pct:.0f} %"
            ts, r = arr.get(f"timeline__{body}__t_s"), arr.get(f"timeline__{body}__fit_rms_mm")
            if ts is not None and r is not None and np.isfinite(r).any():
                rms_max = max(rms_max, float(np.nanpercentile(r, 99)))
                ser.append(Series("line", np.asarray(ts, float), np.asarray(r, float),
                                  role="quaternary", style={"width": 0.4, "rms": True}))
        lanes.append((label, ser))
    # Lanes top to bottom (y axis reversed); the fit residual RMS drawn inside its lane.
    series = []
    rms_max = rms_max or 1.0
    for i, (_, ser) in enumerate(lanes):
        for sr in ser:
            if sr.style.pop("rms", False):
                y = i + 0.38 - 0.76 * np.clip(np.asarray(sr.y, float) / rms_max, 0, 1)
                series.append(Series("line", sr.x, y, role=sr.role, style=sr.style))
            else:
                series.append(Series(sr.kind, sr.x, np.asarray(sr.y, float) + i, role=sr.role,
                                     style=sr.style, label=sr.label, text=sr.text))
    p = Panel("", Axis("Session time (master clock)", SEC), Axis(""), series, legend="below")
    p.y.ticks = [float(i) for i in range(len(lanes))]
    p.y.ticklabels = [lab for lab, _ in lanes]
    top = (-1.7 if stagger else -1.25) if lanes and lanes[0][0] == "phases" else -0.7
    p.y.lim = (top, len(lanes) - 0.3)
    p.y.reverse = True
    persp = f"{who}, motion take"
    return FigureSpec(fid, title, [p], width="text",
                      height_in=0.17 * len(lanes) + (0.1 if stagger else 0.0),
                      caption=_caption(
                          "Every stream of the motion take on the master clock. Bars: covered "
                          "time (a gap longer than three sample intervals splits a bar), with the "
                          "effective rate; optical lanes: frames with a pose (lock), the thin "
                          f"line is the rigid-fit residual RMS (lane height = {rms_max:.2g} mm). Top "
                          "lane: protocol phases.", persp),
                      sources=[src, "qc:timeline_segments"], perspective=persp,
                      meta={"pid": pid})


def _triad(T: np.ndarray, L: float) -> list[Series]:
    out = []
    for k, rl in enumerate(("axis:x", "axis:y", "axis:z")):
        d = T[:, :3, k] * L
        out.append(Series("arrow3d", T[:, 0, 3], T[:, 1, 3], z=T[:, 2, 3], d=d, role=rl))
    return out


@register("data_tracking_scene", "data", "Tracking scene", params=("pid", "phase"),
          perspectives=("trials",), perspective="trials")
def tracking_scene(req: ViewRequest) -> FigureSpec:
    """World-frame poses of headpiece, mandibular array and CT pointer with triads, camera."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_tracking_scene", "Tracking scene"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    name = req.phase or "all"
    arr = ctx.arrays("qc", pid)
    src = "qc:scene"
    if not any(k.startswith("scene__") for k in arr):
        from biomech.qc.streams import scene

        sd = _session_or_none(ctx, pid)
        arr = scene(sd) if sd is not None else {}
        src = "session:optical"
    if not any(k.startswith(f"scene__{name}__") for k in arr):
        return _empty_fig(fid, title, f"no optical data for {who}", "text")
    panels = []
    bodies = (("cra", "headpiece (CRA)", "cat:0"), ("mta", "mandibular array (MTA)", "cat:1"),
              ("ct", "CT pointer tip", "cat:2"))
    for zoom in (False, True):
        p = Panel("(b) Head region" if zoom else "(a) Camera frame", Axis("x camera", MM),
                  Axis("y camera", MM), [], projection="3d", z=Axis("z camera (depth)", MM),
                  view=(-35.0, 25.0) if not zoom else (-60.0, 20.0),
                  legend="none" if zoom else "below")
        for body, lab, rl in bodies:
            Q = arr.get(f"scene__{name}__{body}__xyz")
            if Q is None:
                continue
            Q = np.asarray(Q, float)
            if body == "ct":
                ok = np.isfinite(Q).all(axis=1)
                if ok.any():
                    p.series.append(Series("scatter3d", Q[ok, 0], Q[ok, 1], z=Q[ok, 2],
                                           role=rl, style={"size": 0.5},
                                           label=None if zoom else lab))
                continue
            p.series.append(Series("line3d", Q[:, 0], Q[:, 1], z=Q[:, 2], role=rl,
                                   style={"width": 0.6}, label=None if zoom else lab))
            key = arr.get(f"scene__{name}__{body}__key")
            if key is not None:
                p.series += _triad(np.asarray(key, float), 15.0 if zoom else 40.0)
        if not zoom:
            # A cube from the camera to the bodies (equal mm axes, the depth not squeezed).
            pts = np.vstack([np.asarray(arr[f"scene__{name}__{b}__xyz"], float)
                             for b in ("cra", "mta") if f"scene__{name}__{b}__xyz" in arr]
                            + [np.zeros((1, 3))])
            pts = pts[np.isfinite(pts).all(axis=1)]
            c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
            h = 0.55 * float(np.max(pts.max(axis=0) - pts.min(axis=0)))
            p.x.lim, p.y.lim, p.z.lim = ((c[0] - h, c[0] + h), (c[1] - h, c[1] + h),
                                         (c[2] - h, c[2] + h))
            p.series += _triad(np.eye(4)[None], 80.0)
            p.series.append(Series("scatter3d", np.array([0.0]), np.array([0.0]),
                                   z=np.array([0.0]), role="reference", style={"ms": "s"},
                                   label="reference camera"))
        panels.append(p)
    persp = f"{who}, {req.phase or 'whole motion take'}"
    return FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.6,
                      caption=_caption(
                          "Poses in the reference camera's frame (origin, square; z = viewing "
                          "direction): paths of the headpiece and mandibular-array origins and the "
                          "CT pointer tip where tracked, with frame triads (x vermilion, y green, "
                          "z blue) at the start, middle and end. The second camera is not drawn: "
                          "the session files carry no stereo calibration.", persp),
                      sources=[src], perspective=persp, meta={"pid": pid})


LM_SHORT = V.labels("landmark", "title")


def tip_cloud(opt, ph, body: str, trim: float = 0.2) -> tuple[np.ndarray, np.ndarray] | None:
    """CT-tip positions in the body frame during a hold and the 'central part' mask
    (the same rule as :func:`biomech.frames.landmarks.digitize`)."""
    t = opt.t_us.astype(np.int64)
    span = ph.t_end_us - ph.t_start_us
    sel = (t >= ph.t_start_us) & (t <= ph.t_end_us)
    tip, B = opt.poses["ct"][sel], opt.poses[body][sel]
    ok = np.isfinite(tip.reshape(len(tip), -1)).all(1) & np.isfinite(B.reshape(len(B), -1)).all(1)
    if ok.sum() < 5:
        return None
    local = np.einsum("nji,nj->ni", B[ok, :3, :3], tip[ok, :3, 3] - B[ok, :3, 3])
    tt = t[sel][ok]
    central = (tt >= ph.t_start_us + trim * span) & (tt <= ph.t_end_us - trim * span)
    return local, central


@register("data_landmarks", "data", "Landmark digitization", params=("pid",),
          needs_session=True, perspectives=("trials",), perspective="trials")
def landmark_digitization(req: ViewRequest) -> FigureSpec:
    """Per landmark: CT-tip cloud in the hold, median and MAD; camera frame of the hold."""
    from biomech.frames.landmarks import LANDMARKS

    ctx = req.ctx
    pid = _pid(req)
    fid, title = "data_landmarks", "Landmark digitization"
    if pid is None:
        return _empty_fig(fid, title, "no participants", "text")
    who = _who(ctx, pid)
    try:
        sd = ctx.emg(pid).motion
        opt = sd.optical()
    except Exception:
        opt, sd = None, None
    if opt is None or "ct" not in opt.poses:
        return _empty_fig(fid, title, f"no optical CT pointer data for {who}", "text")
    panels, mads = [], {}
    body_of = {"mp": "mta", "hp": "cra"}
    for name, spec in LANDMARKS.items():
        ph = sd.phase(name)
        ttl = LM_SHORT[name]
        got = tip_cloud(opt, ph, body_of[spec.body]) if ph is not None else None
        if got is None:
            panels.append(_empty(ttl, "no hold"))
            continue
        local, central = got
        med = np.median(local[central], axis=0)
        dev = local - med
        mad = float(np.median(np.linalg.norm(dev[central], axis=1)))
        mads[name] = mad
        w, V = np.linalg.eigh(np.cov(dev[central].T))
        V = V[:, ::-1]                                  # principal axes, largest first
        D = dev @ V[:, :2]
        p = Panel(ttl, Axis("PC1", MM), Axis("PC2", MM), [
            Series("scatter", D[~central, 0], D[~central, 1], role="faint",
                   style={"size": 0.6}),
            Series("scatter", D[central, 0], D[central, 1], role="context", style={"size": 0.6}),
        ], equal=True, legend="none")
        a = np.linspace(0, 2 * np.pi, 60)
        p.series.append(Series("line", mad * np.cos(a), mad * np.sin(a), role="accent",
                               style={"width": 0.7}))
        p.series.append(Series("scatter", np.array([0.0]), np.array([0.0]), role="accent",
                               style={"ms": "d"}))
        p.note = f"MAD {mad:.2f} mm, n {int(central.sum())}"
        # Square box per panel around the central part (stray trimmed points may fall out).
        r = max(3.0 * mad, float(np.percentile(np.linalg.norm(D[central], axis=1), 98)), 0.05)
        p.x.lim = p.y.lim = (-1.25 * r, 1.25 * r)
        panels.append(p)
    _grid_labels(panels, 4)
    cols, n_img = 4, 0
    # Camera frames show the participant's face: only on request (participant report).
    if req.options.get("filmstrip", False):
        prev = ctx.previews(pid)
        for name in LANDMARKS:
            ph = sd.phase(name)
            rows_ = [r for r in prev if r.get("phase") == name and r["cam_id"] == "cam0"]
            if ph is None or not rows_:
                continue
            mid = 0.5 * (ph.t_start_us + ph.t_end_us)
            r = min(rows_, key=lambda r: abs(r["t_master_us"] - mid))
            try:
                from PIL import Image

                img = np.asarray(Image.open(r["abs_path"]).convert("L"), float) / 255.0
            except Exception:
                continue
            q = Panel(LM_SHORT[name] + ", cam0", Axis(""), Axis(""), [
                Series("image", img=img, extent=(0, img.shape[1], img.shape[0], 0))],
                legend="none")
            q.x.ticks, q.x.ticklabels, q.x.lim = [], [], (0, img.shape[1])
            q.y.ticks, q.y.ticklabels, q.y.lim = [], [], (0, img.shape[0])
            q.y.reverse = True
            panels.append(q)
            n_img += 1
    persp = f"{who}, calibration holds"
    return FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=1.15,
                      caption=_caption(
                          "CT-pointer tip during each landmark hold, in the frame of the body the "
                          "landmark belongs to, as deviation from the median along the cloud's "
                          "principal axes (grey: central part used for the median; light: "
                          "trimmed ends); each panel has its own scale. Circle: the 3-D median "
                          "absolute deviation (MAD)."
                          + (" Bottom: the camera frame at mid-hold." if n_img else ""), persp),
                      sources=["session:optical", "kinematics:landmarks"], perspective=persp,
                      meta={"pid": pid, "mad_mm": mads})


def data_views() -> list[str]:
    """Ids of every registered data view (report order)."""
    from biomech.viz.views.registry import views

    return [v.id for v in views("data")]


# Views that take a statistic default to mean +/- SD unless registered otherwise, so an
# exported id only carries a statistic that was chosen explicitly.
for _v in REGISTRY.values():
    if _v.kind == "data" and "stat" in _v.params:
        _v.defaults.setdefault("stat", "mean_sd")


def _drop_empty(spec: FigureSpec) -> FigureSpec:
    """Remove data series without a single finite point (fully masked EMG, empty
    path parts): they draw nothing and only make pgfplots warn."""
    for p in spec.panels:
        keep = []
        for sr in p.series:
            if sr.kind in ("line", "scatter", "colorline", "band", "line3d", "scatter3d"):
                cols = [sr.x, sr.y] + ([sr.z] if sr.kind.endswith("3d") else []) + (
                    [sr.y2] if sr.kind == "band" else [])
                ok = np.ones(np.size(sr.x), bool)
                for c in cols:
                    ok &= np.isfinite(np.asarray(c, float)).ravel()
                if sr.kind == "colorline" and sr.c is not None:
                    ok &= np.isfinite(np.asarray(sr.c, float))
                if not ok.any():
                    continue
            keep.append(sr)
        p.series = keep
    return spec


def _wrapped(fn):
    def build(req: ViewRequest) -> FigureSpec:
        return _drop_empty(fn(req))

    build.__doc__ = fn.__doc__
    return build


for _k, _v in list(REGISTRY.items()):
    if _v.kind == "data" and _v.builder.__module__ == __name__ and \
            _v.builder.__name__ != "build":
        REGISTRY[_k] = replace(_v, builder=_wrapped(_v.builder))
