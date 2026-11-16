# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Views of the system-validation study (group ``validation``; master plan 4.15, R5b §2,
R5d.4): sync diagnostics, the anatomical-frame jaw-in-head comparison with every recorded
segment colored and labelled, cycle agreement with SPM bars, robustness, decomposition,
reference quality and the error budget, and the CT tip verification.

Every view reads stored results only (stages ``ref_sync``, ``ref_tip``, ``ref_quality``,
``ref_kinematics``, ``ref_agreement``); pseudonyms only.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register

MM, DEG, SEC, MSEC, PPM = "\\milli\\metre", "\\degree", "\\second", "\\milli\\second", ""
G = {"group": "validation", "domain": "validation"}
SYS = {"s": ("JTS-MM", "system:jtsmm", None), "q": ("Qualisys", "system:qualisys", None)}
TOL = {"metrology": (0.5, 0.5), "application": (2.0, 1.2)}        # (mm, deg), V8
XYZ = ("x (anterior)", "y (left)", "z (superior)")


# -- helpers -----------------------------------------------------------------------------------
def _pid(req: ViewRequest) -> str | None:
    pids = list(req.ctx.pids())
    return req.pid if req.pid in pids else (pids[0] if pids else None)


def _who(ctx: Any, pid: str) -> str:
    from biomech.viz.views.results_kinematics import who

    return who(ctx, pid)


def n_text(ctx: Any, pids: Any, extra: str = "") -> str:
    """The sample of a caption from the participants shown: "n = 2 participants (P1, P2)",
    followed by ``extra`` ("5 held-out tasks each") when given."""
    names = sorted({_who(ctx, p) for p in pids})
    k = len(names)
    out = f"n = {k} participant{'' if k == 1 else 's'} ({', '.join(names)})"
    return f"{out}, {extra}" if extra else out


def _empty(fid: str, title: str, msg: str) -> FigureSpec:
    p = Panel("", Axis(""), Axis(""), [], note=msg)
    return FigureSpec(fid, title, [p], width="column", height_in=1.0, caption=msg)


def _arrays(ctx: Any, stage: str, pid: str) -> dict[str, np.ndarray]:
    try:
        return ctx.arrays(stage, pid)
    except (KeyError, FileNotFoundError, LookupError):
        return {}


def _table(ctx: Any, stage: str, name: str, pids: list[str] | None = None) -> pd.DataFrame:
    try:
        return ctx.table(stage, name, pids)
    except (KeyError, FileNotFoundError, LookupError):
        return pd.DataFrame()


def _spans(a: dict[str, np.ndarray]) -> list[tuple[str, float, float]]:
    t = np.asarray(a["t_s"], float)
    return B.phase_spans(t, a["phase"])


def _sync_spans(a: dict[str, np.ndarray]) -> list[tuple[str, float, float]]:
    t = np.asarray(a["t_s"], float)
    w = np.asarray(a.get("in_sync_window", np.zeros(len(t), bool)), bool)
    from biomech import timebase

    return [("sync_start" if k == 0 else "sync_end", float(t[a_]), float(t[b - 1]))
            for k, (a_, b) in enumerate(timebase.segments(w))]


def _line(p: Panel, t: np.ndarray, y: np.ndarray, sys_: str, role: str | None = None,
          label: str | None = None, width: float = 0.6) -> None:
    lab, r, dash = SYS[sys_]
    st: dict[str, Any] = {"width": width}
    if dash:
        st["dash"] = dash
    p.series.append(Series("line", t, y, role=role or r, style=st, label=label))


def shared_legend(panels: list[Panel]) -> list[Panel]:
    """One legend per distinct set of legend entries: panels repeating a set the
    figure already shows lose their legend; the last panel of each set keeps it (bottom of a
    grid). In place; returns ``panels``."""
    seen: set[tuple[str, ...]] = set()
    for p in reversed(panels):
        key = tuple(s.label for s in p.legend_entries())
        if not key or p.legend_mode() == "none":
            continue
        if key in seen:
            p.legend = "none"
        seen.add(key)
    return panels


# -- participant encoding (user 2026-11-04) ----------------------------------------
# Everywhere in the systems paper: marker shape = participant (circle the first, square the
# second), line style = participant where lines are drawn (solid / dashed); colour = system,
# estimator or task. Values that belong to no participant (bench, dataset references, ratios
# of the two) keep a neutral marker.
PARTICIPANT_MARK = "os"
POINT_KINDS = ("scatter", "errorbar")


def pmark(i: int) -> str:
    """Marker of the ``i``-th participant (sorted pids)."""
    return PARTICIPANT_MARK[i % 2]


def shape_by_pid(panels: Sequence[Panel], i: int) -> None:
    """A figure (or panel set) of one participant: every point marker in that participant's
    shape."""
    for p in panels:
        for s in p.series:
            if s.kind in POINT_KINDS and not s.style.get("fixed_marker"):
                s.style = {**s.style, "ms": pmark(i), "marker": pmark(i)}


def pshape_text(ctx: Any, pids: Sequence[Any]) -> tuple[str, str]:
    """Caption encoding entry of the participant shapes."""
    ps = sorted(dict.fromkeys(pids))
    return ("circle / square", " / ".join(_who(ctx, x) for x in ps[:2]) or "participants")


def shape_by_rows(p: Panel, row_pid: Sequence[Any], pids: Sequence[Any]) -> None:
    """Point markers by the participant of their row (``row_pid[i]`` for y row i, rows at
    integer y with small offsets): a series spanning rows of both participants is split in
    two, the legend label kept on the first part."""
    out: list[Series] = []
    for s in p.series:
        if s.kind not in POINT_KINDS or s.y is None:
            out.append(s)
            continue
        y = np.asarray(s.y, float)
        rows = np.clip(np.rint(y).astype(int), 0, max(len(row_pid) - 1, 0))
        who = np.array([pids.index(row_pid[r]) if row_pid[r] in pids else 0 for r in rows])
        parts = sorted(set(who.tolist()))
        if len(parts) == 1:
            s.style = {**s.style, "ms": pmark(parts[0]), "marker": pmark(parts[0])}
            out.append(s)
            continue
        label = s.label
        for w in parts:
            k = who == w
            kw = {f: (np.asarray(getattr(s, f))[k] if getattr(s, f) is not None
                      and np.ndim(getattr(s, f)) and len(np.asarray(getattr(s, f))) == len(y)
                      else getattr(s, f))
                  for f in ("x", "y", "y2", "ylo")}
            # a part without a finite point would be an empty pgfplots plot (a compile
            # problem); the legend label goes to the first part that draws something
            if not (np.isfinite(np.asarray(kw["x"], float))
                    & np.isfinite(np.asarray(kw["y"], float))).any():
                continue
            out.append(Series(s.kind, kw["x"], kw["y"], y2=kw["y2"], ylo=kw["ylo"],
                              role=s.role,
                              style={**s.style, "ms": pmark(w), "marker": pmark(w)},
                              label=label))
            label = None
    p.series = out


def _tol_lines(p: Panel, unit: str) -> None:
    """Both bounds (magnitudes, so no ±): the application bound dashed and labeled with its
    value at the right end; the metrology bound a thin grey line, named with its value in the
    caption (:data:`TOL_TEXT`). In a 0.6-in row the two lines are a few points apart, and a
    label on the lower line was struck through by the upper one."""
    for name, (mm, deg) in TOL.items():
        v = mm if unit == "mm" else deg
        lab = f"{name} {v:g}" + ("\u00a0mm" if unit == "mm" else "°")
        p.series.append(Series("hline", y=np.array([v]),
                               role="reference" if name == "application" else "muted",
                               label=lab if name == "application" else None))


# == sync ======================================================================================
# rising / falling CT visibility edges: two hues and two shapes, the same in every sync figure
# rise / fall by marker shape in ink (colour harmonisation 2026-11-05: vermillion is P1 and green
# a variant elsewhere in the paper); the shape rule X1 does not overwrite them
EDGE_STYLE = {"rise": ("accent", "^"), "fall": ("accent", "v")}

@register("val_sync_overview", "dashboard", "Sync: CT visibility and edges", params=("pid",),
          choices={"form": ("panels", "compact")}, form="panels", **G)
def sync_overview(req: ViewRequest) -> FigureSpec:
    """CT visibility in both systems on the master clock, the phase lane and the sync
    windows; matched visibility edges with their residuals. Form ``compact``:
    ``validation_r8`` (J2.6)."""
    if str(req.options.get("form", "panels")) == "compact":
        from biomech.viz.views.validation_r8 import sync_overview_compact

        return sync_overview_compact(req)
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "val_sync_overview", "Sync: CT visibility and edges"
    if pid is None:
        return _empty(fid, title, "no participants")
    a = _arrays(ctx, "ref_sync", pid)
    k = _arrays(ctx, "ref_kinematics", pid)
    if "ct_t_s" not in a:
        return _empty(fid, title, f"no ref_sync arrays for {_who(ctx, pid)}")
    t0 = float(a["ct_t_s"][0])
    lane = B.phase_lane(_spans(k) if k else [], extra=_sync_spans(k) if k else ())
    pv = Panel("(a) CT visible", Axis(""), Axis("", lim=(-0.6, 1.6), ticks=[0.0, 1.0],
                                                ticklabels=["Qualisys", "JTS-MM"]),
               [], sharex="time")
    for y, t, v, r in ((1.0, a["ct_t_s"], a["ct_visible_s"], "system:jtsmm"),
                       (0.0, a["ct_t_q_master_s"], a["ct_visible_q"], "system:qualisys")):
        tt = np.asarray(t, float) - t0
        vv = np.where(np.asarray(v, bool), y, np.nan)
        # solid for both systems: Qualisys' dash pattern looked like dropouts
        pv.series.append(Series("line", tt, vv, role=r, style={"width": 4.0, "dash": "solid"}))
    sy = _table(ctx, "ref_sync", "sync", [pid])
    if not sy.empty:          # detected visibility edges (Qualisys mapped through the clock)
        r0 = sy.iloc[0]
        c, al, tr = r0["offset_s"], r0["alpha_ppm"] * 1e-6, r0["t_ref_s"]
        for y, sysk in ((1.0, "s"), (0.0, "q")):
            for kind, (role, mk) in EDGE_STYLE.items():
                e = np.asarray(a.get(f"edges_{sysk}_{kind}", np.zeros(0)), float)
                if sysk == "q":
                    e = (e - c + al * tr) / (1.0 + al)
                if e.size:
                    # rises above falls: in the waving bursts they are fractions of a second
                    # apart and the falls hid the rises
                    dy = 0.4 if kind == "rise" else 0.22
                    pv.series.append(Series("scatter", e - t0, np.full(e.size, y + dy),
                                            role=role, style={"ms": mk, "size": 1.6,
                                                              "fixed_marker": True},
                                            label=(f"{kind} edge" if y == 1.0 else None)))
    pv.legend = "below"
    pe = Panel("(b) Residual lag under the fitted clock", Axis("Time in take", SEC),
               Axis("Qualisys − clock model", MSEC), [], sharex="time", legend="below")
    ep = np.asarray(a.get("edge_pairs", np.zeros((0, 3))), float)
    if "sliding_t_s" in a:
        pe.series.append(Series("scatter", np.asarray(a["sliding_t_s"], float) - t0,
                                np.asarray(a["sliding_resid_ms"], float), role="context",
                                style={"size": 0.8},
                                label="sliding 15-s windows (all bodies, unfiltered)"))
    if ep.size and not sy.empty:
        r0 = sy.iloc[0]
        lag = r0["offset_s"] + r0["alpha_ppm"] * 1e-6 * (ep[:, 0] - r0["t_ref_s"])
        res_ms = (ep[:, 1] - ep[:, 0] - lag) * 1e3       # t_Q − model(t_S)
        for typ, kind in ((1.0, "rise"), (-1.0, "fall")):
            k = ep[:, 2] == typ
            if k.any():
                role, mk = EDGE_STYLE[kind]
                pe.series.append(Series("scatter", ep[k, 0] - t0, res_ms[k], role=role,
                                        style={"ms": mk, "size": 1.6, "fixed_marker": True},
                                        label=f"matched {kind} edge"))
        pe.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    if pid in ctx.pids():                 # X1: shape = participant; rise / fall by colour
        shape_by_pid([pv, pe], sorted(ctx.pids()).index(pid))
    persp = f"{_who(ctx, pid)}, comparison take"
    spec = FigureSpec(fid, title, [lane, pv, pe], rows=3, cols=1, width="text", height_in=1.3,
                      sources=["ref_sync:arrays", "ref_kinematics:arrays"], perspective=persp,
                      meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "how the two systems are put on one clock: (a) CT visibility in "
        "JTS-MM and Qualisys on the master clock with the recorded phases and the detected "
        "visibility edges; (b) residual lag of the matched edges (coarse initialization only) "
        "and of the sliding 15-s windows under the fitted affine clock (flat around 0 = the "
        "clock model holds over the take)",
        [("lane", "recorded phases, sync windows outlined; L1–L3 / U1–U3: lower / upper "
                  "landmark holds, TMJ L / R: condyle holds"),
         ("thick bars (a)", "CT visible, JTS-MM (blue, top) and Qualisys (black, bottom)"),
         ("up / down triangles", "rising / falling visibility edges (a: detected, "
                                 "b: matched; rises drawn above falls in (a))"),
         ("other markers", "the participant's shape (circle P1, square P2)"),
         ("grey dots (b)", "sliding 15-s windows"),
         ("horizontal line (b)", "fitted clock")],
        persp, statistic="raw signals of one take, no summary: one residual per matched edge "
                         "and per sliding window"))


@register("val_sync_windows", "dashboard", "Sync: CT angular speed and cross-correlation",
          params=("pid",), **G)
def sync_windows(req: ViewRequest) -> FigureSpec:
    """CT |ω| of both systems in the start and end sync windows on the master clock after
    the fine fit, and the |ω| cross-correlation curve of each window."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "val_sync_windows", "Sync: CT angular speed and cross-correlation"
    a = _arrays(ctx, "ref_sync", pid) if pid else {}
    win = _table(ctx, "ref_sync", "windows", [pid]) if pid else pd.DataFrame()
    if "omega_s" not in a or win.empty:
        return _empty(fid, title, "no sync arrays")
    panels = []
    fitw = win[win["role"] == "fit"]
    for r in fitw.itertuples():
        p = Panel(f"{V.phase_label(r.window)}", Axis("Time", SEC), Axis("|ω|", "\\degree\\per\\second"),
                  [], legend="below")
        for key_t, key_w, s in (("omega_s_t", "omega_s", "s"), ("omega_q_t_master", "omega_q", "q")):
            t, w = np.asarray(a[key_t], float), np.asarray(a[key_w], float)
            k = (t >= r.t0_s) & (t <= r.t1_s)
            # stored as |ω|² in (°/s)² (ref_sync speed_series, power 2): drawn as |ω|
            _line(p, t[k] - r.t0_s, np.sqrt(np.clip(w[k], 0, None)), s, label=SYS[s][0])
        panels.append(p)
        lags, rho = a.get(f"xcorr_{r.window}_lags_s"), a.get(f"xcorr_{r.window}_rho")
        px = Panel("Cross-correlation", Axis("Lag − coarse lag", MSEC), Axis("ρ"), [])
        if lags is not None and rho is not None:
            lg = np.asarray(lags, float)
            px.series.append(Series("line", (lg - lg[np.nanargmax(rho)]) * 1e3,
                                    np.asarray(rho, float), role="accent"))
        panels.append(px)
    persp = f"{_who(ctx, pid)}, sync windows"
    return FigureSpec(fid, title, panels, rows=len(fitw), cols=2, width="text", height_in=1.6,
                      caption="Angular speed of the calibration tool in both systems after the "
                              "fine fit (Qualisys on the master clock; 100-ms difference "
                              "quotient) and the cross-correlation curve of |ω|² per window "
                              "(peak at 0 = the coarse lag).",
                      sources=["ref_sync:arrays", "ref_sync:windows"], perspective=persp,
                      meta={"pid": pid})


# reader gloss of the clock variants (validation.sync.clock_variants), for captions
CLOCK_GLOSS = {
    "ct": "fitted to the calibration-tool motion alone (primary)",
    "joint": "fitted to the calibration tool and the head and jaw bodies jointly",
    "two_stage": "calibration tool first, then refitted with the head and jaw frames whose "
                 "residual lies within 3 robust SD",
    "hybrid": "clock of the primary fit with the world registration of the two-stage fit",
    "tukey": "primary fit with a robust (Tukey) loss",
    "sliding_omega": "time model from the sliding windows of the angular speed",
    "sliding_pose": "time model from the sliding windows of the registered poses"}


@register("val_sync_lag", "dashboard", "Sync: sliding-window lag and clock variants",
          params=("pid",), **G)
def sync_lag(req: ViewRequest) -> FigureSpec:
    """Sliding-window residual lags over the take against the affine model, and the clock
    variants (offset change and α) with their pre-registered criteria."""
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "val_sync_lag", "Sync: sliding-window lag and clock variants"
    sl = _table(ctx, "ref_sync", "sliding_lag", [pid]) if pid else pd.DataFrame()
    cv = _table(ctx, "ref_sync", "clock_variants", [pid]) if pid else pd.DataFrame()
    cc = _table(ctx, "ref_sync", "clock_criteria", [pid]) if pid else pd.DataFrame()
    if sl.empty:
        return _empty(fid, title, "no sliding-lag table")
    t0 = float(sl["t_centre_s"].min())
    p = Panel("(a) Residual lag", Axis("Time in take", SEC), Axis("Residual lag", MSEC), [],
              legend="below")
    for body, role in (("joint", "accent"), ("pose", "context")):
        d = sl[(sl["body"] == body) & sl["ok"]] if "ok" in sl else sl[sl["body"] == body]
        if not d.empty:
            p.series.append(Series("errorbar", d["t_centre_s"].to_numpy() - t0,
                                   d["resid_ms"].to_numpy(), y2=d["resid_ms"].to_numpy()
                                   + d.get("se_ms", pd.Series(0, index=d.index)).to_numpy(),
                                   ylo=d["resid_ms"].to_numpy()
                                   - d.get("se_ms", pd.Series(0, index=d.index)).to_numpy(),
                                   role=role, label={"joint": "angular speed, head and jaw",
                                                     "pose": "registered poses"}[body]))
    # unlabelled: the label sat on the data (the windows span the take); the caption names it
    p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    panels = [p]
    if not cv.empty:
        crit = cc.set_index("variant")["pass"] if not cc.empty else pd.Series(dtype=bool)
        labels = [f"{V.term('clock_variant', v, 'short')}{' ✓' if bool(crit.get(v, False)) else ''}"
                  for v in cv["variant"]]
        panels.append(B.strip(labels, [[x] for x in cv["d_offset_ms"]], "(b) Clock variants",
                              Axis("Offset − primary", MSEC), ref=0.0))
        panels.append(B.strip(labels, [[x] for x in cv["alpha_ppm"]], "(c) Drift α",
                              Axis("α", "ppm")))
    if pid in ctx.pids():                 # X1: shape = participant (also in val_setup)
        shape_by_pid(panels, sorted(ctx.pids()).index(pid))
    persp = f"{_who(ctx, pid)}, whole take"
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text", height_in=1.9,
                      sources=["ref_sync:sliding_lag", "ref_sync:clock_variants",
                               "ref_sync:clock_criteria"], perspective=persp, meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "(a) Sliding 15-s window lags (from the squared angular speed of head and jaw, and from "
        "the registered poses) minus the fitted affine clock: flat means the clock model holds "
        "over the take. (b, c) Clock variants: offset relative to the primary clock and "
        "drift α. Variants: " + "; ".join(
            f"{V.term('clock_variant', v, 'short')}: {CLOCK_GLOSS.get(v, V.term('clock_variant', v))}"
            for v in (cv["variant"] if not cv.empty else [])),
        [("black markers with whiskers (a)", "lag of one window from the angular speed ± its "
                                             "standard error"),
         ("grey markers with whiskers (a)", "the same from the registered poses"),
         ("horizontal line (a)", "fitted clock"),
         ("markers (b, c)", "one estimate per clock variant"),
         ("marker shape", "the participant's (circle P1, square P2)"),
         ("✓", "all pre-registered criteria passed; no ✓: at least one criterion not met "
               "(criteria per variant in the supplementary source data)")],
        persp, statistic="lag per 15-s window with its standard error; one estimate per clock "
                         "variant, no summary"))


@register("val_sync_ct3d", "dashboard", "Sync: CT path in both systems", params=("pid",),
          choices={"form": ("paths", "geometry")}, form="paths", **G)
def sync_ct3d(req: ViewRequest) -> FigureSpec:
    """The CT path in the sync windows, both systems in the Qualisys world after the sync and
    registration (3-D and sagittal projection). Form ``geometry``: with the CT marker
    triangle (``validation_r8``, J2.4)."""
    if str(req.options.get("form", "paths")) == "geometry":
        from biomech.viz.views.validation_r8 import sync_ct3d_geometry

        return sync_ct3d_geometry(req)
    ctx = req.ctx
    pid = _pid(req)
    fid, title = "val_sync_ct3d", "Sync: CT path in both systems"
    a = _arrays(ctx, "ref_sync", pid) if pid else {}
    if "ct_path_s" not in a:
        return _empty(fid, title, "no CT path arrays")
    Ps, Pq = np.asarray(a["ct_path_s"], float), np.asarray(a["ct_path_q"], float)
    c = np.nanmean(Pq, axis=0)
    Ps, Pq = Ps - c, Pq - c
    p3 = Panel("", Axis("x", MM), Axis("y", MM), [], projection="3d", z=Axis("z", MM),
               legend="below")
    p2 = Panel("(b) Projection", Axis("x", MM), Axis("z", MM), [], equal=True)
    for P, s in ((Ps, "s"), (Pq, "q")):
        lab, r, dash = SYS[s]
        st = {"width": 0.5, **({"dash": dash} if dash else {})}
        p3.series.append(Series("line3d", P[:, 0], P[:, 1], z=P[:, 2], role=r, style=st,
                                label=lab))
        p2.series.append(Series("line", P[:, 0], P[:, 2], role=r, style=st))
    err = np.linalg.norm(Ps - Pq, axis=1)
    tp = np.asarray(a.get("ct_path_t_s", np.arange(len(err))), float)
    pd_ = Panel("(c) Tip distance", Axis("Time in take", SEC), Axis("|Δ|", MM), [])
    pd_.series.append(Series("scatter", tp - float(a["ct_t_s"][0]), err, role="accent",
                             style={"size": 0.4}))
    persp = f"{_who(ctx, pid)}, sync windows"
    return FigureSpec(fid, title, [p3, p2, pd_], rows=1, cols=3, width="text", height_in=2.4,
                      caption=f"Calibration-tool tip in the sync windows, both systems in the "
                              f"Qualisys world (chosen clock and registration); median distance "
                              f"{np.nanmedian(err):.2f} mm.",
                      sources=["ref_sync:arrays"], perspective=persp, meta={"pid": pid})


@register("val_sync_residuals", "dashboard", "Sync: held-out residuals", **G)
def sync_residuals(req: ViewRequest) -> FigureSpec:
    """Held-out per-body residuals (position, rotation) per movement interval, both
    participants (world frame via X, Y: descriptive, secondary)."""
    ctx = req.ctx
    h = _table(ctx, "ref_sync", "heldout")
    fid, title = "val_sync_residuals", "Sync: held-out residuals"
    if h.empty:
        return _empty(fid, title, "no held-out table")
    h = h[h["body"].isin(["MP", "HP"]) & (h["n"] > 0)]
    labels, vp, vr = [], [], []
    for (pid, body), d in h.groupby(["pid", "body"]):
        labels.append(f"{_who(ctx, pid)} {body}")
        vp.append(d["pos_rms_mm"].to_numpy())
        vr.append(d["rot_rms_deg"].to_numpy())
    pa = B.strip(labels, vp, "(a) Position", Axis("RMS", MM))
    pb = B.strip(labels, vr, "(b) Rotation", Axis("RMS", DEG))
    return FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=1.6,
                      caption="Per-body held-out residuals of the registered poses per movement "
                              "interval (dots; diamond = median). Secondary: depends on X and Y; "
                              "the primary outcome compares anatomical frames.",
                      sources=["ref_sync:heldout"])


# == kinematics on anatomical frames ==========================================================
def _kin(req: ViewRequest, fid: str, title: str):
    ctx = req.ctx
    pid = _pid(req)
    if pid is None:
        return None, None, _empty(fid, title, "no participants")
    a = _arrays(ctx, "ref_kinematics", pid)
    if "incisor_s_mm" not in a:
        return None, None, _empty(fid, title, f"no ref_kinematics arrays for {_who(ctx, pid)}")
    return pid, a, None


@register("val_kin_timeseries", "dashboard", "Jaw-in-head over the take, both systems",
          params=("pid",), **G)
def kin_timeseries(req: ViewRequest) -> FigureSpec:
    """Lower incisor (x, y, z) and mandibular rotation of both systems between their own
    anatomical frames over the whole take, every recorded segment colored and labelled,
    and the disagreement with the bounds."""
    fid, title = "val_kin_timeseries", "Jaw-in-head over the take, both systems"
    pid, a, e = _kin(req, fid, title)
    if e:
        return e
    t = np.asarray(a["t_s"], float)
    spans = _spans(a)
    lane = B.phase_lane(spans, extra=_sync_spans(a))
    panels = [lane]
    for k in range(3):
        p = Panel(f"Incisor {XYZ[k]}", Axis(""), Axis(XYZ[k][0], MM), [], sharex="time",
                  legend="below" if k == 0 else None)
        for s in ("s", "q"):
            _line(p, t, a[f"incisor_{s}_mm"][:, k], s, label=SYS[s][0] if k == 0 else None)
        B.add_phase_bands(p, spans)
        panels.append(p)
    pr = Panel("Rotation since rest (opening, about y)", Axis(""), Axis("ry", DEG), [],
               sharex="time")
    for s in ("s", "q"):
        _line(pr, t, a[f"rot_{s}_deg"][:, 1], s)
    B.add_phase_bands(pr, spans)
    pe = Panel("Disagreement", Axis("Time in take", SEC), Axis("Incisor", MM), [],
               sharex="time", legend="below")
    pe.series.append(Series("line", t, a["err_incisor_mm"], role="accent",
                            style={"width": 0.6}, label="|Δ incisor|, own frames"))
    if "err_incisor_aligned_mm" in a:
        pe.series.append(Series("line", t, a["err_incisor_aligned_mm"], role="cat:2",
                                style={"width": 0.6}, label="|Δ incisor|, frames aligned (leave-one-task-out)"))
    g = np.asarray(a.get("gate", np.zeros(len(t), bool)), bool)
    if g.any():
        from biomech import timebase

        runs = timebase.segments(g)
        pe.series.insert(0, Series("hspan", np.array([t[r[0]] for r in runs]),
                                   y2=np.array([t[r[1] - 1] for r in runs]), role="context",
                                   style={"alpha": 0.3}))
    if "tracking_only_err_incisor_mm" in a:
        pe.series.append(Series("line", t, a["tracking_only_err_incisor_mm"], role="cat:1",
                                style={"width": 0.5}, label="tracking only (leave-one-task-out)"))
    _tol_lines(pe, "mm")
    B.add_phase_bands(pe, spans)
    prr = Panel("Rotation disagreement", Axis("Time in take", SEC), Axis("|Δr|", DEG), [],
                sharex="time", legend="below")
    for key, role, lab in (("err_rot_deg", "accent", "own frames"),
                           ("err_rot_aligned_deg", "cat:2", "frames aligned (leave-one-task-out)"),
                           ("tracking_only_err_rot_deg", "cat:1", "tracking only (leave-one-task-out)")):
        if key in a:
            prr.series.append(Series("line", t, a[key], role=role, style={"width": 0.5},
                                     label=lab))
    B.add_phase_bands(prr, spans)
    pe.x = Axis("")
    panels += [pr, pe, prr]
    persp = f"{_who(req.ctx, pid)}, comparison take"
    # 0.75 in per row keeps the seven rows within the 8.5-in page limit (R7 size check)
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=0.75,
                      sources=["ref_kinematics:arrays"], perspective=persp, meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "lower incisor in the maxillary frame and opening rotation of both systems over the "
        "whole take, each between its own anatomical frames. Bottom two rows: incisor and "
        "rotation disagreement |Δ| in own frames, after the frame alignment (leave-one-task-out) "
        "and tracking-only (legend), with the metrology and application bounds",
        [("solid", "JTS-MM"), ("dashed", "Qualisys"),
         ("colored bands", "rest and movement tasks"), ("grey", "landmark holds"),
         ("outlined", "sync windows (all labeled in the lane)"),
         ("shaded (incisor disagreement)", "frames failing the Qualisys quality gate"),
         ("dashed horizontal line", "application bound (labeled with its value)"),
         ("thin grey horizontal line", f"metrology bound ({TOL['metrology'][0]:g} mm, "
                                       f"{TOL['metrology'][1]:g}°)")],
        persp, statistic="raw per-frame signals of one take, no summary"))


def _movement_window(a: dict[str, np.ndarray], t: np.ndarray, pad_s: float = 3.0,
                     tail_s: float = 12.0) -> tuple[float, float] | None:
    """Time range from the start of rest to the last evaluated movement-task frame, plus
    ``tail_s`` without data on the right, where the bound labels sit."""
    from biomech.validation.selection import MOVEMENT_TASKS

    ph = np.asarray(a.get("phase", []), dtype=str)
    if ph.size != t.size:
        return None
    rest = ph == "rest"
    task = np.isin(ph, list(MOVEMENT_TASKS))
    if "evaluated" in a and np.asarray(a["evaluated"]).size == t.size:
        task &= np.asarray(a["evaluated"], bool)
    keep = rest | task
    if not task.any():
        return None
    return (float(t[keep].min() - pad_s), float(t[task].max() + tail_s))


def _clip_spans(spans: list[tuple[str, float, float]], win: tuple[float, float] | None
                ) -> list[tuple[str, float, float]]:
    """The spans that overlap ``win``, clipped to it (all of them without a window)."""
    if win is None:
        return spans
    lo, hi = win
    return [(n, max(a, lo), min(b, hi)) for n, a, b in spans if b > lo and a < hi]


def _seg_paths(a: dict[str, np.ndarray], key: str):
    """(segment name, mask) for every labeled segment with data."""
    ph = np.asarray(a["phase"]).astype(str)
    for name in dict.fromkeys(ph):
        if name:
            yield name, ph == name


PLANE_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing")


@register("val_kin_planes", "dashboard", "Incisor paths by segment, both systems",
          params=("pid",), choices={"form": ("segments", "paper_grid")}, form="segments",
          **G)
def kin_planes(req: ViewRequest) -> FigureSpec:
    """Sagittal, frontal and horizontal incisor paths of both systems, one color per
    recorded segment (dashed = Qualisys). Form ``paper_grid``: :func:`kin_planes_grid`."""
    from biomech.viz.views.data import PLANES, _plane_panel, fit_equal

    if req.options.get("form", "segments") == "paper_grid":
        return kin_planes_grid(req)
    fid, title = "val_kin_planes", "Incisor paths by segment, both systems"
    pid, a, e = _kin(req, fid, title)
    if e:
        return e
    # small multiples: one column per recorded segment, one row per plane (all segments in
    # one panel overplotted each other, so solid and dashed could not be told apart)
    segs = [(n_, k) for n_, k in _seg_paths(a, "incisor") if V.phase_role(n_) != "faint"]
    planes = (("sagittal", "Sagittal"), ("frontal", "Frontal"), ("horizontal", "Horizontal"))
    grid: list[Panel] = []
    for r, (pl, pl_lab) in enumerate(planes):
        ix, iy = PLANES[pl]
        for j, (name, k) in enumerate(segs):
            pan = _plane_panel(V.phase_label(name, short=True) if r == 0 else "", pl)
            if j:                     # axis names in the first column only
                pan.hidden_labels = {"y": (pan.y.label, pan.y.unit),
                                     "x": (pan.x.label, pan.x.unit)}
                pan.y, pan.x = Axis("", ""), Axis("", "")
            else:
                pan.y = Axis(f"{pl_lab}: {pan.y.label}", pan.y.unit)
            for s_ in ("s", "q"):
                P = np.where(k[:, None], a[f"incisor_{s_}_mm"], np.nan)
                pan.series.append(Series(
                    "line", P[:, ix], P[:, iy],
                    role=V.phase_role(name) if s_ == "s" else SYS["q"][1],
                    style={"width": 0.5 if s_ == "s" else 0.6},
                    label=(None if r < len(planes) - 1 or j else SYS[s_][0])))
            grid.append(pan)
    if not grid:
        return _empty(fid, title, "no incisor paths")
    fit_equal(grid, same=True, min_span=12.0)
    grid[(len(planes) - 1) * len(segs)].legend = "below"
    persp = f"{_who(req.ctx, pid)}, rest and movement tasks"
    spec = FigureSpec(fid, title, grid, rows=len(planes), cols=len(segs), width="text",
                      height_in=1.15,
                      sources=["ref_kinematics:arrays"], perspective=persp,
                      meta={"equal_scale": "shared", "pid": pid})
    return attach(spec, CaptionSpec(
        "paths of the lower incisor in each system's maxillary frame, one column per recorded "
        "rest and movement segment (landmark holds omitted) and one row per plane (sagittal, "
        "frontal, horizontal); one mm scale for all panels",
        [("colored solid line", "JTS-MM, color = segment"), ("black dashed line", "Qualisys")],
        persp, statistic="raw per-frame paths of one take, no summary"))


def kin_planes_grid(req: ViewRequest) -> FigureSpec:
    """The main agreement picture (2026-11-05): incisor paths of JTS-MM (frames aligned,
    blue solid) and Qualisys (black dashed, drawn on top) in the sagittal and frontal plane
    for the four movement tasks, both takes with equal weight: one row per take, a column
    pair (sagittal, frontal) per task; one mm scale for all panels (two rows keep the figure
    within 0.45 of a page at text width)."""
    from biomech.viz.views.data import PLANES, _plane_panel, fit_equal

    fid, title = "val_kin_planes", "Incisor paths of both takes, both systems"
    ctx = req.ctx
    pids = sorted(ctx.pids())
    arrs = {p: _arrays(ctx, "ref_kinematics", p) for p in pids}
    pids = [p for p in pids if "incisor_s_aligned_mm" in arrs[p] and "phase" in arrs[p]]
    if not pids:
        return _empty(fid, title, "no aligned incisor paths")
    tasks = [t for t in PLANE_TASKS
             if all(np.any(np.asarray(arrs[p]["phase"]).astype(str) == t) for p in pids)]
    planes = (("sagittal", "sagittal"), ("frontal", "frontal"))
    grid: list[Panel] = []
    for pid in pids:
        a = arrs[pid]
        ph = np.asarray(a["phase"]).astype(str)
        for j, task in enumerate(tasks):
            for r, (pl, pl_lab) in enumerate(planes):
                c = 2 * j + r
                head = (f"{V.phase_label(task, short=True)}, {pl_lab}" if pid == pids[0]
                        else "")
                pan = _plane_panel(head, pl)
                xl = ("x" if pl == "sagittal" else "y", MM)
                if pid != pids[-1]:
                    pan.hidden_labels["x"] = (pan.x.label, pan.x.unit)
                    pan.x = Axis("", "")
                else:
                    pan.x = Axis(*xl)
                if c:
                    pan.hidden_labels["y"] = (pan.y.label, pan.y.unit)
                    pan.y = Axis("", "")
                else:
                    pan.y = Axis(f"{_who(ctx, pid)}: z", MM)
                k = ph == task
                for s_, key in (("s", "incisor_s_aligned_mm"), ("q", "incisor_q_mm")):
                    ix, iy = PLANES[pl]
                    P = np.where(k[:, None], a[key], np.nan)
                    pan.series.append(Series(
                        "line", P[:, ix], P[:, iy], role=SYS[s_][1],
                        style={"width": 0.5, **({"dash": "dashed"} if s_ == "q" else {})},
                        label=SYS[s_][0] + (" (frames aligned)" if s_ == "s" else "")
                        if (pid == pids[-1] and c == 0) else None))
                grid.append(pan)
    # one mm scale; x limits per plane (the frontal paths are narrow), z limits common
    for r in range(len(planes)):
        fit_equal([q for i, q in enumerate(grid) if i % 2 == r], same=True, min_span=12.0)
    from biomech.viz import layout as L

    zs = [q.y.lim for q in grid if q.y.lim]
    zl = (min(z[0] for z in zs), max(z[1] for z in zs))
    zt, dec = L.linear_ticks(zl[0], zl[1], 3)
    cols = 2 * len(tasks)
    for i, q in enumerate(grid):
        # the z range is common: tick labels in the first column only, so the panels can
        # use the width (the shared mm scale is limited by the narrowest cell)
        q.y.lim, q.y.ticks = zl, list(zt)
        q.y.ticklabels = [L.fmt_tick(v, dec) if i % cols == 0 else "" for v in zt]
    grid[len(grid) - cols].legend = "below"
    spec = FigureSpec(fid, title, grid, rows=len(pids), cols=cols, width="text",
                      height_in=1.8, sources=["ref_kinematics:arrays"],
                      perspective=f"{', '.join(_who(ctx, p) for p in pids)}, movement tasks",
                      meta={"equal_scale": "shared", "pids": pids, "legend_span": "figure"})
    who = " and ".join(_who(ctx, p) for p in pids)
    return attach(spec, CaptionSpec(
        "paths of the lower incisor in the maxillary frame for the four movement tasks, "
        "sagittal and frontal plane (a column pair per task), both comparison takes with "
        f"equal weight (rows: {who}); JTS-MM with its maxillary and mandibular frames aligned "
        "to Qualisys' by constant offsets fitted on the other movement tasks; one mm scale "
        "for all panels (x anterior, y left, z superior)",
        [("blue solid line", "JTS-MM (frames aligned)"),
         ("black dashed line", "Qualisys (drawn on top)")],
        f"{who}, one comparison take each",
        statistic="raw per-frame paths, no summary; the agreement statistics are in the "
                  "tables"))


@register("val_kin_orient", "dashboard", "Rotation component pairs, both systems",
          params=("pid",), **G)
def kin_orient(req: ViewRequest) -> FigureSpec:
    """Rotation components (opening vs lateral, opening vs tilt) of both systems by segment."""
    fid, title = "val_kin_orient", "Rotation component pairs, both systems"
    pid, a, e = _kin(req, fid, title)
    if e:
        return e
    pa = Panel("(a) Opening vs lateral", Axis("About z", DEG), Axis("About y", DEG), [])
    pb = Panel("(b) Opening vs tilt", Axis("About x", DEG), Axis("About y", DEG), [],
               legend="below")
    for name, k in _seg_paths(a, "rot"):
        if V.phase_role(name) == "faint":
            continue
        for s in ("s", "q"):
            R = np.where(k[:, None], a[f"rot_{s}_deg"], np.nan)
            st = {"width": 0.5, **({"dash": "dashed"} if s == "q" else {})}
            pa.series.append(Series("line", R[:, 2], R[:, 1], role=V.phase_role(name), style=st))
            pb.series.append(Series("line", R[:, 0], R[:, 1], role=V.phase_role(name), style=st,
                                    label=V.phase_label(name) if s == "s" else None))
    persp = f"{_who(req.ctx, pid)}, rest and movement tasks"
    return FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=2.3,
                      caption="Mandibular rotation vector since rest, each system in its own "
                              "anatomical frames; solid JTS-MM, dashed Qualisys.",
                      sources=["ref_kinematics:arrays"], perspective=persp, meta={"pid": pid})


@register("val_kin_3d", "dashboard", "Incisor and condyle paths in 3-D, both systems",
          params=("pid",), **G)
def kin_3d(req: ViewRequest) -> FigureSpec:
    """3-D incisor (and condyle-point) paths of both systems, colored by segment."""
    fid, title = "val_kin_3d", "Incisor and condyle paths in 3-D, both systems"
    pid, a, e = _kin(req, fid, title)
    if e:
        return e
    p3 = Panel("", Axis("x (anterior)", MM), Axis("y (left)", MM), [], projection="3d",
               z=Axis("z (superior)", MM), view=(-60.0, 20.0), legend="below")
    for name, k in _seg_paths(a, "incisor"):
        if V.phase_role(name) == "faint":
            continue
        for s in ("s", "q"):
            P = np.where(k[:, None], a[f"incisor_{s}_mm"], np.nan)
            st = {"width": 0.5, **({"dash": "dashed"} if s == "q" else {})}
            p3.series.append(Series("line3d", P[:, 0], P[:, 1], z=P[:, 2],
                                    role=V.phase_role(name), style=st,
                                    label=V.phase_label(name) if s == "s" else None))
            for n in ("tmj_left", "tmj_right"):
                if f"{n}_{s}_mm" in a:
                    C = np.where(k[:, None], a[f"{n}_{s}_mm"], np.nan)
                    p3.series.append(Series("line3d", C[:, 0], C[:, 1], z=C[:, 2],
                                            role="context", style={"width": 0.4}))
    persp = f"{_who(req.ctx, pid)}, rest and movement tasks"
    return FigureSpec(fid, title, [p3], width="column", height_in=2.9,
                      caption="Incisor (colored by segment) and condyle-point (grey) paths, "
                              "solid JTS-MM, dashed Qualisys, each in its own maxillary frame.",
                      sources=["ref_kinematics:arrays"], perspective=persp, meta={"pid": pid})


def _n_cycles(w: pd.DataFrame, task: str) -> str:
    """"5 " (the cycle count of one task, with a trailing space) from the stored waveform
    table; empty when not stored."""
    if w.empty or not {"task", "n_cycles"} <= set(w.columns):
        return ""
    n = w.loc[w["task"] == task, "n_cycles"].dropna()
    if n.empty:
        return ""
    lo, hi = int(n.min()), int(n.max())
    return f"{lo if lo == hi else f'{lo}–{hi}'} "


def diff_ylim(a: dict, key: str, u: str) -> tuple[float, float]:
    """Symmetric y range of a difference panel: the bounds, the mean difference and the band
    up to three times the application bound -- a CI spike at a cycle end (few cycles there)
    is clipped at the axis instead of flattening the mean difference."""
    app = TOL["application"][0 if u == "mm" else 1]
    m = np.abs(np.asarray(a.get(f"{key}__diff_mean", []), float))
    band = np.abs(np.r_[np.asarray(a.get(f"{key}__diff_lo", []), float),
                        np.asarray(a.get(f"{key}__diff_hi", []), float)])
    m_max = float(np.nanmax(m)) if np.isfinite(m).any() else 0.0
    b_max = float(np.nanmax(band)) if np.isfinite(band).any() else 0.0
    lim = max(1.25 * app, 1.15 * m_max, min(b_max * 1.05, 3.0 * app))
    return (-lim, lim)


def _visible_clusters(clusters: Any, min_pct: float = 2.0) -> list[tuple[float, float, float]]:
    """SPM clusters widened to at least ``min_pct`` % of the cycle around their centre, so a
    single-node cluster is not a speck at print size; drawing only."""
    out = []
    for a, b, p_ in B.parse_clusters(clusters):
        if b - a < min_pct:
            m = 0.5 * (a + b)
            a, b = max(0.0, m - min_pct / 2), min(100.0, m + min_pct / 2)
        out.append((a, b, p_))
    return out


@register("val_kin_waveforms", "dashboard", "Cycle waveforms with SPM and equivalence",
          params=("pid", "phase"), phases=("open_close", "chewing", "protrusion_retrusion",
                                           "left_right"),
          choices={"mode": ["disp", "abs"], "frames": ["aligned", "own"],
                   "form": ["panels", "curves", "difference"]}, form="panels", **G)
def kin_waveforms(req: ViewRequest) -> FigureSpec:
    """Time-normalized cycles of both systems (mean ± SD) with SPM significance bars, and
    the mean difference with its effective-n band against the metrology and application
    bounds (equivalence view). Forms ``curves`` / ``difference``: only one of the two
    columns, 2 × 2 (``validation_gallery``)."""
    form = str(req.options.get("form", "panels"))
    if form in ("curves", "difference"):
        from biomech.viz.views.validation_gallery import kin_waveforms_form

        return kin_waveforms_form(req, form)
    fid, title = "val_kin_waveforms", "Cycle waveforms with SPM and equivalence"
    ctx = req.ctx
    pid = _pid(req)
    task = req.phase if isinstance(req.phase, str) else "open_close"
    mode = str(req.options.get("mode", "disp"))
    frames = str(req.options.get("frames", "aligned"))
    sfx = "" if mode == "abs" else "__disp"
    a0 = _arrays(ctx, "ref_agreement", pid) if pid else {}
    pre = "aligned__" if frames == "aligned" else ""
    a = {k[len(pre):]: v for k, v in a0.items() if k.startswith(pre)} if pre else \
        {k: v for k, v in a0.items() if not k.startswith("aligned__")}
    w = _table(ctx, "ref_agreement", "waveforms" + ("_aligned" if pre else ""), [pid]) \
        if pid else pd.DataFrame()
    if not w.empty and "mode" in w:
        w = w[w["mode"] == mode]
    chans = [("incisor_z", "Incisor z", MM, "mm"), ("incisor_x", "Incisor x", MM, "mm"),
             ("incisor_y", "Incisor y", MM, "mm"), ("rot_y", "Opening rotation", DEG, "deg")]
    if not any(f"{task}__{c}{sfx}__mean_s" in a for c, *_ in chans):
        return _empty(fid, title, f"no {V.phase_label(task)} cycles")
    x = np.linspace(0, 100, 101)
    panels = []
    for c, lab, unit, u in chans:
        key = f"{task}__{c}{sfx}"
        if f"{key}__mean_s" not in a:
            continue
        first = not panels
        p = Panel(lab, Axis("Cycle", "\\percent"), Axis(lab, unit), [],
                  legend="below" if first else "none")
        for s in ("s", "q"):
            m, sd = a[f"{key}__mean_{s}"], a[f"{key}__sd_{s}"]
            p.series.append(Series("band", x, m - sd, y2=m + sd, role=SYS[s][1]))
            _line(p, x, m, s, label=SYS[s][0] if first else None, width=1.0)
        row = w[w["channel"] == c]
        if not row.empty:
            B.add_spm_bars(p, [("JTS-MM vs Qualisys" if first else "",
                                _visible_clusters(row.iloc[0]["clusters"]), "accent")])
        pd_ = Panel(f"Δ {lab}", Axis("Cycle", "\\percent"), Axis("Δ", unit), [])
        pd_.series.append(Series("band", x, a[f"{key}__diff_lo"], y2=a[f"{key}__diff_hi"],
                                 role="accent"))
        pd_.series.append(Series("line", x, a[f"{key}__diff_mean"], role="accent"))
        pd_.y.lim = diff_ylim(a, key, u)
        for name, (mm, deg) in TOL.items():
            v = mm if u == "mm" else deg
            for sign in (-1, 1):
                pd_.series.append(Series("hline", y=np.array([sign * v]),
                                         role="reference" if name == "application" else "context"))
        panels += [p, pd_]
    persp = f"{_who(ctx, pid)}, {V.phase_label(task)} cycles"
    # one legend and 1.1 in per row: four channel rows and the caption on one page
    spec = FigureSpec(fid, title, panels, rows=len(panels) // 2, cols=2, width="text",
                      height_in=1.1,
                      sources=["ref_agreement:arrays", "ref_agreement:waveforms"],
                      perspective=persp, meta={"pid": pid})
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of JTS-MM and Qualisys, {_who(ctx, pid)}, "
        + ("frames aligned" if frames == "aligned" else "each system in its own anatomical "
           "frames") + ". Left: time-normalized cycles per channel; right: their difference "
        "JTS-MM − Qualisys against the application and metrology bounds. "
        + ("Frames aligned: constant maxillary and mandibular offsets fitted on the other "
           "tasks. " if frames == "aligned" else "")
        + ("Displacement from the cycle start (the constant offset of the two systems' "
           "anatomical frames removed; it is reported separately)" if mode == "disp" else
           "Absolute, including the constant offset of the frame definitions"),
        [("solid line and band (left)", "JTS-MM"), ("dashed line and band (left)", "Qualisys"),
         ("bars above the curves", "supra-threshold clusters of the SPM test (drawn at least "
                                   "2 % of the cycle wide)"),
         ("line and band (right)", "mean difference and its 95 % band (clipped at the axis "
                                   "where few cycles reach a node)"),
         ("dashed lines", f"application bound (±{TOL['application'][0]:g} mm, "
                          f"±{TOL['application'][1]:g}°)"),
         ("thin grey lines", f"metrology bound (±{TOL['metrology'][0]:g} mm, "
                             f"±{TOL['metrology'][1]:g}°)")],
        f"{_who(ctx, pid)}, {_n_cycles(w, task)}{V.phase_label(task)} cycles",
        statistic="left: mean ± SD over the cycles; right: mean difference with a t-interval "
                  "per node using the effective number of cycles",
        test="paired SPM t-test over cycles, p < 0.05",
        notes=["Equivalence where the band lies inside a bound."]))


@register("val_kin_agreement", "dashboard", "Cycle-feature agreement",
          choices={"frames": ["aligned", "own"], "form": ["forest", "bland_altman"]},
          form="forest", **G)
def kin_agreement(req: ViewRequest) -> FigureSpec:
    """Bias of each cycle feature with its bootstrap CI per participant and task, against
    the bounds. Form ``bland_altman``: per-cycle differences against the means with the
    stored bias and limits (``validation_gallery``)."""
    if req.options.get("form", "forest") == "bland_altman":
        from biomech.viz.views.validation_gallery import kin_agreement_ba

        return kin_agreement_ba(req)
    fid, title = "val_kin_agreement", "Cycle-feature agreement"
    ctx = req.ctx
    frames = str(req.options.get("frames", "aligned"))
    f = _table(ctx, "ref_agreement", "features_aligned" if frames == "aligned" else "features")
    if f.empty:
        return _empty(fid, title, "no feature agreement")
    panels = []
    for feat, lab, unit in (("amp_mm", "Task amplitude", MM),
                            ("excursion_mm", "Incisor excursion", MM),
                            ("rot_excursion_deg", "Rotation excursion", DEG)):
        d = f[f["feature"] == feat]
        if d.empty:
            continue
        labels = [f"{_who(ctx, r.pid)} {V.phase_label(r.task, short=True)} ({int(r.n)})"
                  for r in d.itertuples()]
        v = TOL["application"][0 if unit == MM else 1]
        # an interval far beyond the bounds (few effective cycles) would squeeze the other
        # rows: the axis ends at 4 bounds, a clipped end is marked with its value
        lim = 4.0 * v
        lo_, hi_ = d["bias_ci_lo"].to_numpy(float), d["bias_ci_hi"].to_numpy(float)
        p = B.strip(labels, [[b] for b in d["bias"]], lab, Axis("Bias", unit),
                    intervals=list(zip(np.clip(lo_, -lim, lim), np.clip(hi_, -lim, lim))))
        p.series.insert(0, Series("vline", x=np.array([0.0]), role="context",
                                  style={"width": 0.8}))
        for i, (a, b) in enumerate(zip(lo_, hi_)):
            for end, side in ((a, -1), (b, 1)):
                if np.isfinite(end) and abs(end) > lim:
                    p.series.append(Series("text", x=np.array([side * 0.75 * lim]),
                                           y=np.array([float(i)]),
                                           text=[("← " if side < 0 else "") + B.num(end, 1)
                                                 + (" →" if side > 0 else "")],
                                           style={"anchor": "above"}))
        p.x.lim = (-lim, lim)
        for sgn in (-1, 1):
            # the bound dashed (threshold), the zero line solid (reference)
            p.series.append(Series("vline", x=np.array([sgn * v]), role="threshold"))
        shape_by_rows(p, list(d["pid"]), sorted(f["pid"].unique()))    # X1
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=max(1.4, 0.16 * len(f) / 3),
                      sources=["ref_agreement:features_aligned" if frames == "aligned"
                               else "ref_agreement:features"])
    fr = ("frames aligned (tracking only: a constant registration between the systems' body "
          "frames removes the marking offset)" if frames == "aligned" else
          "each system in its own anatomical frames (end to end, the marking offset included)")
    return attach(spec, CaptionSpec(
        f"bias of cycle features (JTS-MM − Qualisys) per participant and task, {fr}; the "
        "margin of a feature is the distance of its CI to the nearer bound",
        [("dashed lines", f"application bound (±{TOL['application'][0]:g} mm, "
                          f"±{TOL['application'][1]:g}°)"),
         ("solid grey line", "zero bias"),
         ("rows", "participant, task and number of cycles"), pshape_text(ctx, f["pid"]),
         ("arrow with value", "a CI end beyond the axis (few effective cycles)")],
        f"n = {f['pid'].nunique()} participants",
        statistic="bias with its 95 % CI (t-interval with the effective number of cycles)",
        notes=_missing_cycles(ctx, f)))


def _missing_cycles(ctx: Any, f: pd.DataFrame) -> list[str]:
    """The reason for participant-tasks without a row (validation_part_i)."""
    from biomech.viz.views.validation_part_i import missing_cycle_notes

    return missing_cycle_notes(ctx, f, sorted(f["pid"].unique())) if "pid" in f else []


@register("val_robustness", "dashboard", "Robustness to clock and tip", **G)
def robustness(req: ViewRequest) -> FigureSpec:
    """The pooled primary outcome under every clock variant and tip variant."""
    fid, title = "val_robustness", "Robustness to clock and tip"
    ctx = req.ctx
    r = _table(ctx, "ref_kinematics", "robustness")
    if r.empty:
        return _empty(fid, title, "no robustness table")
    labels = [f"{_who(ctx, x.pid)} {x.factor}: {V.term(x.factor + '_variant', x.variant, 'short')}"
              for x in r.itertuples()]
    pa = B.strip(labels, [[v] for v in r["incisor_rms_mm"]], "(a) Incisor", Axis("RMS", MM))
    pb = B.strip(labels, [[v] for v in r["rot_rms_deg"]], "(b) Rotation", Axis("RMS", DEG))
    for q in (pa, pb):                      # X1: shape = participant
        shape_by_rows(q, list(r["pid"]), sorted(r["pid"].unique()))
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=max(1.6, 0.15 * len(r)),
                      sources=["ref_kinematics:robustness"])
    return attach(spec, CaptionSpec(
        "how much the jaw-in-head disagreement of JTS-MM and Qualisys (each system in its own "
        "anatomical frames) depends on the clock model and on the CT tip: "
        "(a) incisor and (b) rotation RMS under each clock variant (CT-only = primary) and "
        "each CT-tip variant",
        [("rows", "participant, factor (clock or tip) and variant"),
         ("marker", "the participant's value under that variant"), pshape_text(ctx, r["pid"])],
        n_text(ctx, r["pid"]),
        statistic="per participant the RMS over all movement-task frames that pass the "
                  "Qualisys quality gate (tasks pooled within the participant)"))


@register("val_decomposition", "dashboard", "End-to-end vs tracking-only", **G)
def decomposition(req: ViewRequest) -> FigureSpec:
    """End-to-end disagreement (own anatomical frames) vs tracking-only (leave-one-task-out
    registration between the body frames) per segment."""
    fid, title = "val_decomposition", "End-to-end vs tracking-only"
    ctx = req.ctx
    d = _table(ctx, "ref_kinematics", "decomposition")
    if d.empty:
        return _empty(fid, title, "no decomposition table")
    labels = [f"{_who(ctx, r.pid)} {V.phase_label(r.segment, short=True)}" for r in d.itertuples()]
    p = Panel("Incisor RMS", Axis("RMS", MM), Axis(""), [], legend="below")
    y = np.arange(len(d), dtype=float)
    p.series.append(Series("scatter", d["end_to_end_incisor_rms_mm"].to_numpy(), y,
                           role="primary", label="end-to-end (own frames)"))
    p.series.append(Series("scatter", d["tracking_only_incisor_rms_mm"].to_numpy(), y,
                           role="secondary", label="tracking only (leave-one-task-out)"))
    p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
    shape_by_rows(p, list(d["pid"]), sorted(d["pid"].unique()))    # X1: shape = participant
    spec = FigureSpec(fid, title, [p], width="column", height_in=max(1.6, 0.15 * len(d)),
                      sources=["ref_kinematics:decomposition"])
    return attach(spec, CaptionSpec(
        "incisor disagreement of JTS-MM and Qualisys end-to-end (each system in its own "
        "anatomical frames) and tracking-only (leave-one-task-out registration between the "
        "body frames) per participant and segment",
        [("blue", "end-to-end (own frames)"), ("orange", "tracking-only"),
         pshape_text(ctx, d["pid"])],
        n_text(ctx, d["pid"]),
        statistic="per participant and segment the RMS over the segment's frames (end-to-end "
                  "values without the Qualisys quality gate)",
        notes=["The difference approximates the share of landmark digitization (each "
               "system's own digitization of the same teeth)."]))


# == reference quality =========================================================================
def _gate_mm() -> float:
    """The inter-marker distance threshold of the Qualisys quality gate."""
    from biomech.validation import quality as Q

    return float(Q.GATE_DIST_MM)


@register("val_ref_quality", "dashboard", "Qualisys data quality over the take", params=("pid",),
          **G)
def ref_quality(req: ViewRequest) -> FigureSpec:
    """Qualisys rigid-body consistency (max inter-marker distance deviation) of the head and
    jaw bodies over the take, with the phase lane and the V8 gate threshold."""
    fid, title = "val_ref_quality", "Qualisys data quality over the take"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_quality", pid) if pid else {}
    k = _arrays(ctx, "ref_kinematics", pid) if pid else {}
    if "t_qm_s" not in a:
        return _empty(fid, title, "no ref_quality arrays")
    t0 = float(k["t_master_s"][0]) if k else float(a["t_qm_s"][0])
    t = np.asarray(a["t_qm_s"], float) - t0
    panels = [B.phase_lane(_spans(k), extra=_sync_spans(k))] if k else []
    from biomech import timebase

    def gate_spans(p: Panel, tt: np.ndarray, g: np.ndarray, role: str = "context",
                   alpha: float = 0.25, label: str | None = None) -> None:
        runs = timebase.segments(np.asarray(g, bool))
        if runs:
            p.series.insert(0, Series("hspan", np.array([tt[r[0]] for r in runs]),
                                      y2=np.array([tt[r[1] - 1] for r in runs]),
                                      role=role, style={"alpha": alpha}, label=label))

    for body, lab in (("HP", "Head body"), ("MP", "Jaw body"), ("CT", "Calibration tool")):
        if f"{body}_dev_max_mm" not in a:
            continue
        p = Panel(lab, Axis(""), Axis("max |Δd|", MM), [], sharex="time")
        if f"{body}_gate" in a:
            gate_spans(p, t, a[f"{body}_gate"])
        if f"{body}_label_swapped" in a:            # QTM label swaps corrected (EV1)
            gate_spans(p, t, a[f"{body}_label_swapped"], "cat:4", 0.2, "labels corrected")
            gate_spans(p, t, a[f"{body}_label_undetermined"], "cat:5", 0.3,
                       "labels undetermined")
            p.legend = "below"
        p.series.append(Series("line", t, a[f"{body}_dev_max_mm"], role="accent",
                               style={"width": 0.4}))
        p.series.append(Series("hline", y=np.array([0.5]), role="reference",
                               label="gate 0.5 mm" if body == "HP" else None))
        if k:
            B.add_phase_bands(p, _spans(k), holds=True)
        panels.append(p)
    if "dev_max_at_s" in a and k:
        ts_ = np.asarray(k["t_master_s"], float) - t0
        if len(ts_) == len(a["dev_max_at_s"]):
            p = Panel("Head and jaw at the JTS-MM frames", Axis("Time in take", SEC),
                      Axis("max |Δd|", MM), [], sharex="time")
            gate_spans(p, ts_, a.get("gate_at_s", np.zeros(len(ts_), bool)))
            p.series.append(Series("line", ts_, a["dev_max_at_s"], role="accent",
                                   style={"width": 0.4}))
            p.series.append(Series("hline", y=np.array([0.5]), role="reference"))
            B.add_phase_bands(p, _spans(k), holds=True)
            panels.append(p)
    panels[-1].x = Axis("Time in take", SEC)
    persp = f"{_who(ctx, pid)}, Qualisys frames"
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=1.2,
                      sources=["ref_quality:arrays", "ref_kinematics:arrays"], perspective=persp,
                      meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "quality of the Qualisys data over the take: largest deviation of the three "
        "inter-marker distances of each Qualisys rigid body from their medians (should be "
        "constant). Bottom: the head and jaw gate at the JTS-MM frame times, as used by the "
        "outcome",
        [("black line", "largest deviation per frame"),
         ("dashed line", f"gate threshold {_gate_mm():g} mm"),
         ("grey", "gated frames (any criterion: missing or gap-filled marker, distance, "
                  "residual)"),
         *([("colored shading (CT)", "frames with corrected or undetermined marker labels")]
           if any(k.endswith("_label_swapped") for k in a) else []),
         ("bands", "recorded phases (lane on top)")],
        persp, statistic="raw per-frame values of one take, no summary",
        notes=["The bodies break up while the CT moves next to the face (sync windows)."]))


@register("val_ref_angle", "dashboard", "Qualisys consistency vs viewing direction",
          params=("pid",), **G)
def ref_angle(req: ViewRequest) -> FigureSpec:
    """Median inter-marker deviation binned by each camera's viewing direction in the body
    frame (azimuth × elevation, 30° bins), movement tasks: the flat-marker angle signature."""
    fid, title = "val_ref_angle", "Qualisys consistency vs viewing direction"
    ctx = req.ctx
    pid = _pid(req)
    v = _table(ctx, "ref_quality", "viewing", [pid]) if pid else pd.DataFrame()
    if v.empty:
        return _empty(fid, title, "no viewing-direction table")
    panels = []
    for (body, cam), d in v.groupby(["body", "camera"]):
        az = sorted(d["az_deg"].unique())
        el = sorted(d["el_deg"].unique())
        Z = np.full((len(el), len(az)), np.nan)
        for r in d.itertuples():
            Z[el.index(r.el_deg), az.index(r.az_deg)] = r.dev_med_mm
        panels.append(B.heat(Z, [f"{e:g}°" for e in el], [f"{x:g}°" for x in az],
                             f"{body}, camera {cam}"))
    n_b = v["body"].nunique()
    return FigureSpec(fid, title, panels, rows=n_b, cols=max(1, -(-len(panels) // n_b)),
                      width="text", height_in=1.9,
                      caption="Median inter-marker distance deviation (mm) by the viewing "
                              "direction of each camera in the body frame (rows elevation, "
                              "columns azimuth), movement tasks.",
                      sources=["ref_quality:viewing"], meta={"pid": pid})


@register("val_error_budget", "dashboard", "Error budget and Qualisys adequacy",
          choices={"form": ("dots", "ratio", "heatmap")}, form="dots", **G)
def error_budget(req: ViewRequest) -> FigureSpec:
    """σ_obs, σ_Q and the corrected σ_JTS per segment and outcome, with the 3:1 adequacy
    criterion (V9). Forms ``ratio`` / ``heatmap``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "dots"))
    if form != "dots":
        from biomech.viz.views.validation_r8 import error_budget_form

        return error_budget_form(req, form)
    fid, title = "val_error_budget", "Error budget and Qualisys adequacy"
    ctx = req.ctx
    b = _table(ctx, "ref_kinematics", "budget")
    if b.empty:
        return _empty(fid, title, "no budget table")
    b = b[b["kind"].isin(["movement", "static"])]
    panels = []
    for oc, unit in (("incisor", MM), ("rotation", DEG), ("condyle", MM)):
        d = b[b["outcome"] == oc]
        if d.empty:
            continue
        labels = [f"{_who(ctx, r.pid)} {V.phase_label(r.segment, short=True)}"
                  f"{'' if r.adequate else ' ✗'}" for r in d.itertuples()]
        p = Panel(oc.capitalize(), Axis("σ", unit), Axis(""), [], legend="below")
        y = np.arange(len(d), dtype=float)
        # colour = term (three clearly different hues: grey σ_Q and black σ_JTS were hard
        # to tell apart), shape = participant (X1)
        for col, role, lab in (("sigma_obs", "cat:1", "σ_obs"),
                               ("sigma_ref", "muted", "σ_Q (Qualisys)"),
                               ("sigma_jts_corrected", "system:jtsmm", "σ_JTS corrected")):
            p.series.append(Series("scatter", d[col].to_numpy(), y, role=role, label=lab,
                                   style={"size": 1.2}))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
        shape_by_rows(p, list(d["pid"]), sorted(b["pid"].unique()))
        panels.append(p)
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=max(1.6, 0.15 * len(b) / 3),
                      sources=["ref_kinematics:budget"])
    return attach(spec, CaptionSpec(
        "error budget per participant and segment (rest and movement tasks) for the incisor, "
        "the rotation and the condyle points: the observed bias-removed disagreement σ_obs of "
        "JTS-MM and Qualisys, the propagated Qualisys uncertainty σ_Q and the corrected "
        "σ_JTS = √(σ_obs² − σ_Q²)",
        [("colors", "σ_obs orange, σ_Q (Qualisys) grey, the corrected σ_JTS blue "
                    "(legend)"), pshape_text(ctx, b["pid"]),
         ("✗", "Qualisys not adequate (σ_obs/σ_Q < 3)"),
         ("no σ_JTS point", "not resolvable: σ_Q ≥ σ_obs")],
        n_text(ctx, b["pid"]),
        statistic="one value per participant and segment, no summary; σ_obs is the "
                  "bias-removed RMS over the segment's frames that pass the Qualisys quality "
                  "gate"))


@register("val_availability", "dashboard", "Tracking availability, both systems", **G)
def availability(req: ViewRequest) -> FigureSpec:
    """Share of frames with both head and jaw poses per segment, both systems."""
    fid, title = "val_availability", "Tracking availability, both systems"
    ctx = req.ctx
    av = _table(ctx, "ref_quality", "availability")
    if av.empty:
        return _empty(fid, title, "no availability table")
    av = av[av["kind"] != "hold"]
    labels, vs, roles = [], [], []
    for (pid, seg), d in av.groupby(["pid", "segment"], sort=False):
        for r in d.itertuples():
            labels.append(f"{_who(ctx, pid)} {V.phase_label(seg, short=True)} "
                          f"{'JTS-MM' if r.system == 'jtsmm' else 'Qualisys'}")
            vs.append([100 * r.valid_frac])
            roles.append("primary" if r.system == "jtsmm" else "reference")
    p = B.strip(labels, vs, "Availability", Axis("Frames with both bodies", "\\percent"),
                roles=roles)
    return FigureSpec(fid, title, [p], width="column", height_in=max(1.6, 0.12 * len(labels)),
                      caption="Frames with valid head and jaw poses per segment. In the sync "
                              "windows the CT occludes and confuses both systems.",
                      sources=["ref_quality:availability"])


@register("val_tip", "dashboard", "CT tip estimates", **G)
def tip(req: ViewRequest) -> FigureSpec:
    """Every CT tip estimate (Qualisys CT frame) with its hold-bootstrap SE, and the
    decision (V7)."""
    fid, title = "val_tip", "CT tip estimates"
    ctx = req.ctx
    t = _table(ctx, "ref_tip", "tip_estimates")
    dec = _table(ctx, "ref_tip", "tip_decision")
    if t.empty:
        return _empty(fid, title, "no tip table")
    panels = []
    for ax in "xyz":
        labels = [f"{_who(ctx, r.pid)} {V.term('method', r.method)}" for r in t.itertuples()]
        se = t[f"se_{ax}_mm"].to_numpy()
        val = t[f"d{ax}_mm"].to_numpy()
        panels.append(B.strip(labels, [[v] for v in val], f"d{ax}", Axis(f"d{ax}", MM),
                              ref=0.0, intervals=list(zip(val - 1.96 * se, val + 1.96 * se))))
    cap = "; ".join(f"{_who(ctx, r.pid)}: {V.rewrite_ids(str(r.primary))}"
                    for r in dec.itertuples()) if not dec.empty else ""
    return FigureSpec(fid, title, panels, rows=1, cols=3, width="text",
                      height_in=max(1.6, 0.15 * len(t)),
                      caption="CT tip in the Qualisys CT frame from the shared registration and "
                              "the data-based estimates (95 % hold-bootstrap intervals); 0 = the "
                              f"QTM body origin. Primary tip (V7): {cap}.",
                      sources=["ref_tip:tip_estimates", "ref_tip:tip_decision"])


# == tables ====================================================================================
def _tab(req: ViewRequest, stage: str, name: str, tid: str, caption: str | CaptionSpec,
         where: dict[str, Any] | None = None, columns: list[str] | None = None,
         kinds: tuple[str, ...] | None = None, mean_of: bool = False) -> TableSpec:
    d = _table(req.ctx, stage, name)
    for k, v in (where or {}).items():
        if k in d.columns:
            d = d[d[k] == v]
    if kinds and "kind" in d.columns:
        d = d[d["kind"].isin(kinds)]
    if columns:
        d = d[[c for c in columns if c in d.columns]]
    cap = caption if isinstance(caption, CaptionSpec) else None
    t = B.frame_table(d, tid, "" if cap else str(caption), ctx=req.ctx)
    t.sources = [f"{stage}:{name}"]
    if mean_of and "pid" in d.columns and d["pid"].nunique() > 1:
        add_summary_rows(t, "pid", f"mean of {d['pid'].nunique()}")
    if cap is None:
        return t
    if not cap.n and "pid" in d.columns and len(d):          # the sample of the shown rows
        cap.n = n_text(req.ctx, d["pid"])
    return attach(t, cap)


def add_summary_rows(t: TableSpec, label_key: str, label: str,
                     group: tuple[str, ...] = (), how: str = "mean") -> TableSpec:
    """Append the descriptive summary of a per-participant table (the n = 2 rule: per
    participant plus their mean, never pooled): per value of the ``group`` columns
    (none: one row) the mean (or median) of every numeric column, ``label_key`` reading
    ``label``. Rows with a missing value give NaN for that column. In place."""
    rows = [r for r in t.rows if not r.get("_rule") and not r.get("_head")]
    num = [c.key for c in t.columns if c.key not in (label_key, *group)
           and any(isinstance(r.get(c.key), (int, float, np.floating)) and
                   not isinstance(r.get(c.key), bool) for r in rows)]
    # groups in the order they first appear (the table's own task order)
    keys = list(dict.fromkeys(tuple(r.get(g) for g in group) for r in rows)) if group \
        else [()]
    fn = np.nanmedian if how == "median" else np.mean
    out = []
    for k in keys:
        sel = [r for r in rows if tuple(r.get(g) for g in group) == k]
        row: dict[str, Any] = {label_key: label, **dict(zip(group, k))}
        for c in num:
            v = [r.get(c) for r in sel]
            ok = [float(x) for x in v if isinstance(x, (int, float, np.floating))
                  and not isinstance(x, bool) and np.isfinite(float(x))]
            row[c] = float(fn(ok)) if ok and (how == "median" or len(ok) == len(v)) \
                else np.nan
        out.append(row)
    t.rows = t.rows + out
    return t


FEATURE_UNIT = {"mm": "mm", "mm/s": "mm/s", "deg": "°"}
# the summary of the cycle-feature tables (ref_agreement:features*)
FEATURE_STAT = ("over the cycles of each participant and task: bias = mean difference JTS-MM "
                "− Qualisys with a t-interval using the effective number of cycles; CCC and "
                "ICC(A,1) over the paired cycle values")


def _equiv_test() -> str:
    mm, deg = TOL["application"]
    return (f"equivalence against the pre-specified application bound (±{mm:.1f} mm / "
            f"±{deg:.1f}°)")


def feature_units(t: TableSpec, d: pd.DataFrame) -> TableSpec:
    """Cycle-feature tables mix mm, mm/s and ° rows: the value columns carry no unit in the
    header and a "Unit" column after the feature names each row's unit (stored ``unit``)."""
    if "unit" not in d.columns or "feature" not in [c.key for c in t.columns]:
        return t
    units = [FEATURE_UNIT.get(str(u), str(u)) for u in d["unit"].head(len(t.rows))]
    for c in t.columns:
        if c.key not in ("pid", "task", "feature"):
            c.unit = ""
    i = [c.key for c in t.columns].index("feature") + 1
    t.columns.insert(i, Column("feature_unit", "Unit", align="l"))
    # the decimals of val_table_features_both: speeds (mm/s) to 0.1, other
    # values to 0.01, CCC / ICC to 0.01; cells as text with the real minus
    value_keys = {"bias", "bias_ci_lo", "bias_ci_hi", "equiv_margin", "loa_lo", "loa_hi",
                  "sd_diff", "mean_s", "mean_q"}
    agree_keys = {"ccc", "icc_a1", "pearson_r"}
    for r, u in zip(t.rows, units):
        r["feature_unit"] = u
        for k, v in list(r.items()):
            if isinstance(v, (float, np.floating)) and not isinstance(v, bool):
                if k in agree_keys:
                    r[k] = B.num(float(v), 2)
                elif k in value_keys:
                    r[k] = B.num(float(v), 1 if u == "mm/s" else 2)
    for c in t.columns:
        if c.key in value_keys | agree_keys:
            c.align = "r"
    return t


@register("val_table_segments", "dashboard", "Table: jaw-in-head agreement per segment", **G)
def table_segments(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_kinematics", "segments", "val_table_segments", CaptionSpec(
                "jaw-in-head disagreement of JTS-MM and Qualisys per participant and segment, "
                "each system in its own anatomical frames: RMS of the incisor difference, its "
                "bias-removed part, the constant offset along z (bias) and the RMS rotation "
                "difference (also bias-removed)",
                statistic="over the frames of each segment that pass the Qualisys quality "
                          "gate (n frames): RMS, and the mean for the bias; the pooled row takes "
                          "all movement-task frames of the participant together"),
                {"gated": True},
                columns=["pid", "segment", "n", "incisor_rms_mm", "incisor_dyn_rms_mm",
                         "incisor_bias_z_mm", "rot_rms_deg", "rot_dyn_rms_deg"],
                kinds=("movement", "static", "pooled"))


@register("val_table_segments_aligned", "dashboard", "Table: segments aligned", **G)
def table_segments_aligned(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_kinematics", "segments_aligned", "val_table_segments_aligned",
                CaptionSpec(
                    "frame-aligned agreement of JTS-MM and Qualisys per participant and "
                    "movement task: the Qualisys pose is mapped onto the JTS-MM pose through "
                    "a constant maxillary and a constant mandibular frame offset, fitted on "
                    "the other movement tasks "
                    "(leave-one-task-out); incisor, rotation (also bias-removed) and "
                    "condyle-point RMS",
                    statistic="RMS over the frames of each task that pass the Qualisys quality "
                              "gate (n frames); the pooled row takes all movement-task frames "
                              "of the participant together",
                    notes=["Per-axis bias and SD and the bias-removed incisor RMS are in "
                           "the supplementary source data."]),
                columns=["pid", "segment", "n", "incisor_rms_mm", "rot_rms_deg",
                         "rot_dyn_rms_deg", "tmj_left_rms_mm", "tmj_right_rms_mm"])


@register("val_table_features", "dashboard", "Table: cycle-feature agreement", **G)
def table_features(req: ViewRequest) -> TableSpec:
    return feature_units(_tab(req, "ref_agreement", "features_aligned", "val_table_features",
                CaptionSpec(
                    "agreement of the cycle features of JTS-MM and Qualisys per participant "
                    "and task, frames aligned: Bland–Altman bias with its 95 % CI, CCC, "
                    "ICC(A,1) and the equivalence margin (smallest symmetric bound the CI lies "
                    "inside, mm or °)",
                    statistic=FEATURE_STAT, test=_equiv_test(),
                    notes=["Limits of agreement in the supplementary source data.",
                           *_missing_cycles(req.ctx, _table(req.ctx, "ref_agreement",
                                                            "features_aligned"))]),
                columns=["pid", "task", "feature", "bias", "bias_ci_lo", "bias_ci_hi", "ccc",
                         "icc_a1", "equiv_margin", "equivalent_application"]),
        _table(req.ctx, "ref_agreement", "features_aligned"))


@register("val_table_segments_aligned_freemotion", "dashboard",
          "Table: segments aligned on free motion only", **G)
def table_segments_aligned_freemotion(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_kinematics", "segments_aligned_freemotion",
                "val_table_segments_aligned_freemotion", CaptionSpec(
                    "sensitivity of the frame-aligned agreement of JTS-MM and Qualisys: the "
                    "constant frame offsets fitted on free motion alone (a fixed calibration "
                    "segment, never scored) instead of leave-one-task-out, applied to the four "
                    "protocol tasks; incisor, rotation (also bias-removed) and condyle-point RMS",
                    statistic="per participant and task the RMS over the frames that pass the "
                              "Qualisys quality gate (n frames); the pooled row takes the four "
                              "protocol tasks together"),
                columns=["pid", "segment", "n", "incisor_rms_mm", "rot_rms_deg",
                         "rot_dyn_rms_deg", "tmj_left_rms_mm", "tmj_right_rms_mm"])


@register("val_table_features_own", "dashboard", "Table: cycle-feature agreement, own frames",
          **G)
def table_features_own(req: ViewRequest) -> TableSpec:
    return feature_units(_tab(req, "ref_agreement", "features", "val_table_features_own",
                CaptionSpec(
                    "agreement of the cycle features of JTS-MM and Qualisys per participant "
                    "and task, each system in its own anatomical frames (of equal value with "
                    "the frame-aligned table): Bland–Altman bias with its 95 % CI, CCC, "
                    "ICC(A,1) and the equivalence margin (smallest symmetric bound the CI lies "
                    "inside, mm or °)",
                    statistic=FEATURE_STAT, test=_equiv_test(),
                    notes=_missing_cycles(req.ctx, _table(req.ctx, "ref_agreement",
                                                          "features"))),
                columns=["pid", "task", "feature", "bias", "bias_ci_lo", "bias_ci_hi", "ccc",
                         "icc_a1", "equiv_margin", "equivalent_application"]),
        _table(req.ctx, "ref_agreement", "features"))


@register("val_table_sync", "dashboard", "Table: clock and criteria", **G)
def table_sync(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_sync", "clock_criteria", "val_table_sync",
                "Clock variants and the pre-registered criteria.")


@register("val_table_quality", "dashboard", "Table: Qualisys data quality", **G)
def table_quality(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_quality", "consistency", "val_table_quality",
                "Qualisys rigid-body consistency per body and segment.")


# == optical + IMU fusion (V4; secondary, VD8) =====================================================
# reader names only (no estimator codes F0-F2P in legends or ticks)
EST = {"F0": ("optical only", "context"), "F1": ("complementary filter", "cat:1"),
       "F2": ("Kalman smoother", "cat:2")}
# all estimators of ref_fusion (time series, spectra, dropouts); F2P adds the position (VD6)
EST_ALL = {**EST, "F0_hold": ("optical, hold last pose", "faint"),
           "I": ("IMU alone", "accent"),
           "F2_causal": ("Kalman filter, causal (no smoother)", "cat:2:dot"),
           "F2P": ("Kalman smoother + position", "cat:4"),
           "F2P_causal": ("Kalman + position, causal", "cat:4:dot"),
           "F1_precal": ("complementary, hand-eye mounting", "cat:1:alt"),
           "F2_precal": ("Kalman smoother, hand-eye mounting", "cat:2:alt")}
# short reader names for ticks and table cells, the same in every fusion item
EST_SHORT = {"F0": "optical", "F0_lp6": "6-Hz low-pass", "F1": "complementary",
             "F2": "Kalman", "F2P": "Kalman + position", "F2P_causal": "Kalman + position, causal"}


@register("val_fusion_timeseries", "dashboard", "Fusion: rotation difference over the take",
          params=("pid",), **G)
def fusion_timeseries(req: ViewRequest) -> FigureSpec:
    """Jaw-in-head rotation error against Qualisys for optical-only and the fused
    estimators over the evaluated (held-out) movement tasks, with the phase lane."""
    fid, title = "val_fusion_timeseries", "Fusion: rotation difference over the take"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_fusion", pid) if pid else {}
    k = _arrays(ctx, "ref_kinematics", pid) if pid else {}
    if "err_rot_F0_deg" not in a or not k:
        return _empty(fid, title, "no fusion arrays")
    t = np.asarray(k["t_s"], float)
    spans = _spans(k)
    est_all = EST_ALL
    panels = [B.phase_lane(spans)]
    for key, lab, unit, ax in (("err_rot_{e}_deg", "Rotation difference, own frames", DEG, "|Δr|"),
                               ("err_rot_aligned_{e}_deg", "Rotation difference, frames aligned",
                                DEG, "|Δr|"),
                               ("err_incisor_{e}_mm", "Incisor difference, own frames", MM, "|Δ|"),
                               ("err_incisor_aligned_{e}_mm", "Incisor difference, frames aligned",
                                MM, "|Δ|")):
        p = Panel(lab, Axis(""), Axis(ax, unit), [], sharex="time",
                  legend="below" if key.startswith("err_incisor") else None)
        for e, (elab, role) in est_all.items():
            k_ = key.format(e=e)
            if k_ in a:
                p.series.append(Series("line", t, a[k_], role=role, style={"width": 0.5},
                                       label=elab))
        B.add_phase_bands(p, spans)
        panels.append(p)
    panels[-1].x = Axis("Time in take", SEC)
    return FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=1.3,
                      caption="Jaw-in-head disagreement with Qualisys of optical-only and fused "
                              "estimates in the held-out movement tasks (each task evaluated with "
                              "a calibration fitted without it; secondary), in each "
                              "system's own anatomical frames and after the frame alignment "
                              "(fitted on the optical-only poses of the other tasks).",
                      sources=["ref_fusion:arrays", "ref_kinematics:arrays"], meta={"pid": pid})


@register("val_fusion_segments", "dashboard", "Fusion: per-segment difference, optical vs fused",
          choices={"form": ("lines", "paired", "dumbbell")}, form="lines", **G)
def fusion_segments(req: ViewRequest) -> FigureSpec:
    """Per held-out task: rotation and incisor RMS of optical-only vs F1/F2, at the optical
    frames and at the 200-Hz reference frames; paired differences with block-bootstrap CIs.
    Forms ``paired`` / ``dumbbell``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "lines"))
    if form != "lines":
        from biomech.viz.views.validation_r8 import fusion_segments_form

        return fusion_segments_form(req, form)
    fid, title = "val_fusion_segments", "Fusion: per-segment difference, optical vs fused"
    ctx = req.ctx
    s = _table(ctx, "ref_fusion", "segments")
    pr = _table(ctx, "ref_fusion", "paired")
    if s.empty:
        return _empty(fid, title, "no fusion table")
    panels = []
    order = ["F0", "F1", "F2", "F2P"]
    d = s[(s["query"] == "frames") & s["estimator"].isin(order)]
    order = [e for e in order if e in set(d["estimator"])]
    xs = np.arange(len(order), dtype=float)
    for col, lab, unit in (("incisor_rms_mm", "(a) Incisor, own frames", MM),
                           ("incisor_aligned_rms_mm", "(b) Incisor, frames aligned", MM),
                           ("rot_rms_deg", "(c) Rotation, own frames", DEG),
                           ("rot_aligned_rms_deg", "(d) Rotation, frames aligned", DEG)):
        if col not in d or not d[col].notna().any():
            continue
        first = not panels
        p = Panel(lab, Axis("", ticks=list(xs), ticklabels=[EST_SHORT.get(e, e) for e in order]),
                  Axis("RMS", unit), [], legend="below" if first else "none")
        pids = sorted(d["pid"].unique())
        seen: set[str] = set()
        for (pid, seg), g in d.groupby(["pid", "segment"]):
            g = g.set_index("estimator").reindex(order)
            lab_ = V.phase_label(seg) if first and seg not in seen else None
            seen.add(seg)
            # task = color, participant = line style (P1 solid, P2 dashed)
            st = {"width": 0.6, **({"dash": "dashed"} if pids.index(pid) else {})}
            p.series.append(Series("line", xs, g[col].to_numpy(), role=V.phase_role(seg),
                                   style=st, label=lab_))
        panels.append(p)
    stat, stored = "", "the values at the 200-Hz Qualisys frames"
    if not pr.empty:
        r = pr[pr["estimator"].isin(["F2", "F2P"]) & pr["outcome"].isin(["inc_al", "rot_al"])]
        name = {"inc": "incisor, own", "inc_al": "incisor, aligned", "rot": "rotation, own",
                "rot_al": "rotation, aligned"}
        unit_ = {"inc": "mm", "inc_al": "mm", "rot": "°", "rot_al": "°"}
        unit_ = {k: (u if u == "°" else f"\u00a0{u}") for k, u in unit_.items()}
        stat = "paired difference fused − optical, frame-aligned (95 % block-bootstrap CI): " \
            + "; ".join(
            f"{_who(ctx, x.pid)} {EST_ALL[x.estimator][0]}, {name[x.outcome]} "
            f"{B.num(x.mean_diff)}{unit_[x.outcome]} {B.ci(x.ci_lo, x.ci_hi)}"
            for x in r.itertuples())
        stored += " and the own-frame paired differences"
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text", height_in=1.8,
                      sources=["ref_fusion:segments", "ref_fusion:paired"])
    return attach(spec, CaptionSpec(
        "optical-only and fused jaw-in-head RMS against Qualisys per participant and held-out "
        "task at the optical frames, in each system's own anatomical frames and frame-aligned; "
        "fused: complementary filter, Kalman smoother, Kalman smoother with position fusion",
        [("line color", "task (legend)"), ("solid / dashed", f"{_who(ctx, sorted(d['pid'].unique())[0])} / "
                                                         f"{_who(ctx, sorted(d['pid'].unique())[-1])}")],
        f"n = {d['pid'].nunique()} participants, {d['segment'].nunique()} held-out tasks",
        statistic=stat, notes=[f"{stored[0].upper() + stored[1:]} are in the supplementary "
                               "source data."]))


@register("val_fusion_spectrum", "dashboard", "Fusion: difference spectrum", params=("pid",), **G)
def fusion_spectrum(req: ViewRequest) -> FigureSpec:
    """Spectrum of the rotation error per estimator over the held-out tasks: where the IMU
    adds bandwidth."""
    from scipy.signal import welch

    from biomech import timebase

    fid, title = "val_fusion_spectrum", "Fusion: difference spectrum"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_fusion", pid) if pid else {}
    if "err_rot_F0_deg" not in a:
        return _empty(fid, title, "no fusion arrays")
    t = np.asarray(a["t_master_s"], float)
    fs = 1.0 / float(np.median(np.diff(t)))
    p = Panel("Rotation difference PSD", Axis("Frequency", "\\hertz"),
              Axis("PSD", "\\degree\\squared\\per\\hertz", log=True), [], legend="below")
    for e, (lab, role) in EST_ALL.items():
        if f"err_rot_{e}_deg" not in a:
            continue
        x = np.asarray(a.get(f"err_rot_dyn_{e}_deg", a[f"err_rot_{e}_deg"]), float)
        segs = [x[s:b] for s, b in timebase.segments(np.isfinite(x)) if b - s > 64]
        if not segs:
            continue
        P = [welch(sg - sg.mean(), fs, nperseg=64)[1] for sg in segs]
        f = welch(segs[0] - segs[0].mean(), fs, nperseg=64)[0]
        p.series.append(Series("line", f, np.mean(P, axis=0), role=role, label=lab))
    return FigureSpec(fid, title, [p], width="column", height_in=1.8,
                      caption="Welch spectrum of the rotation difference to Qualisys (held-out tasks).",
                      sources=["ref_fusion:arrays"], meta={"pid": pid})


@register("val_fusion_tau", "dashboard", "Fusion: complementary-filter difference against τ", **G)
def fusion_tau(req: ViewRequest) -> FigureSpec:
    """Descriptive: rotation RMS of F1 for a grid of time constants (the used τ comes from
    the sensors, never from this curve)."""
    fid, title = "val_fusion_tau", "Fusion: complementary-filter difference against τ"
    ctx = req.ctx
    t = _table(ctx, "ref_fusion", "tau")
    cal = _table(ctx, "ref_fusion", "calibration")
    if t.empty:
        return _empty(fid, title, "no τ table")
    p = Panel("Complementary filter, rotation RMS", Axis("τ", SEC, log=True), Axis("RMS", DEG),
              [], legend="below")
    for (pid, seg), g in t.groupby(["pid", "segment"]):
        p.series.append(Series("line", g["tau_s"].to_numpy(), g["rot_dyn_rms_deg"].to_numpy(),
                               role=V.phase_role(seg), style={"width": 0.6}))
    if not cal.empty:
        for v in cal["tau_s"].dropna().unique()[:4]:
            p.series.append(Series("vline", x=np.array([float(v)]), role="reference"))
    return FigureSpec(fid, title, [p], width="column", height_in=1.8,
                      caption="Lines: participant × held-out task; vertical: τ used (derived from "
                              "the sensors).", sources=["ref_fusion:tau", "ref_fusion:calibration"])


@register("val_fusion_dropout", "dashboard", "Fusion: bridging simulated optical dropouts",
          choices={"form": ("curves", "strip")}, form="curves", **G)
def fusion_dropout(req: ViewRequest) -> FigureSpec:
    """Rotation error on the dropped frames against the simulated gap length: holding the
    last optical pose, SLERP between frames, F1 and F2. Form ``strip``: every participant ×
    task at the shortest and longest gap (``validation_gallery``)."""
    if req.options.get("form", "curves") == "strip":
        from biomech.viz.views.validation_gallery import fusion_dropout_strip

        return fusion_dropout_strip(req)
    fid, title = "val_fusion_dropout", "Fusion: bridging simulated optical dropouts"
    ctx = req.ctx
    d = _table(ctx, "ref_fusion", "dropout")
    if d.empty:
        return _empty(fid, title, "no dropout table")
    roles = {"F0_hold": ("hold last pose", "faint"), "F0": ("SLERP / linear", "context"),
             "F1": EST["F1"], "F2": EST["F2"], "F2P": EST_ALL["F2P"],
             "F2P_causal": EST_ALL["F2P_causal"]}
    panels = []
    for col, lab, unit in (("rot_rms_deg", "(a) Rotation, own frames", DEG),
                           ("rot_aligned_rms_deg", "(b) Rotation, frames aligned", DEG),
                           ("incisor_rms_mm", "(c) Incisor, own frames", MM),
                           ("incisor_aligned_rms_mm", "(d) Incisor, frames aligned", MM)):
        if col not in d:
            continue
        # EV8: per participant the median over the held-out tasks (markers), and the
        # descriptive mean of the participants (line), as the dropout macros.
        per = d.groupby(["estimator", "dropout_s", "pid"])[col].median().reset_index()
        g = per.groupby(["estimator", "dropout_s"])[col].mean().reset_index()
        p = Panel(lab, Axis("Gap", SEC, log=True), Axis("RMS", unit), [],
                  legend="below" if not panels else "none")
        for e, (lab_e, role) in roles.items():
            x = g[g["estimator"] == e]
            if x.empty:
                continue
            p.series.append(Series("line", x["dropout_s"].to_numpy(), x[col].to_numpy(),
                                   role=role, label=lab_e))
            for i, pid in enumerate(sorted(per["pid"].unique())):
                xp = per[(per["estimator"] == e) & (per["pid"] == pid)]
                # black, larger participant markers on the estimator's line
                p.series.append(Series("scatter", xp["dropout_s"].to_numpy(), xp[col].to_numpy(),
                                       role=role, style={"ms": "o" if i == 0 else "s",
                                                         "size": 1.3, "color": "ink"}))
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=2, cols=2, width="text", height_in=1.7,
                      sources=["ref_fusion:dropout"])
    return attach(spec, CaptionSpec(
        "how well each estimator bridges simulated optical dropouts: RMS difference to "
        "Qualisys on the dropped frames against the gap length, (a, b) rotation and (c, d) "
        "incisor, in each system's own anatomical frames and frame-aligned",
        [("lines", "mean of the participants' medians, one color per estimator (legend)"),
         *[(f"black {'circles' if i == 0 else 'squares'}", f"{_who(ctx, p_)}'s median")
           for i, p_ in enumerate(sorted(d["pid"].unique())[:2])]],
        n_text(ctx, d["pid"], f"{d['segment'].nunique()} held-out tasks each"),
        statistic="per participant the median over the held-out tasks; line: the descriptive "
                  "mean of the participants' medians (no interval)",
        notes=["Hold last pose, SLERP / linear, the complementary filter and the Kalman smoother "
               "bridge the position linearly between the optical frames; the Kalman smoother "
               "with position fusion carries it with the accelerometer (gravity removed with "
               "the chip orientation). Its causal form (filter only, no backward pass) "
               "integrates the accelerometer without a later optical frame to pull it back, so "
               "its incisor difference to Qualisys grows with the gap length (c, d); a secondary result."]))


@register("val_table_fusion", "dashboard", "Table: fusion calibration and diagnostics", **G)
def table_fusion(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_fusion", "calibration", "val_table_fusion",
                "IMU-to-optical calibration per fold and body (lag, residual, drift, τ).")


@register("val_digitization", "dashboard", "Landmark digitization in both systems", **G)
def digitization(req: ViewRequest) -> FigureSpec:
    """Frame-free digitization QC: incisor-to-left vs incisor-to-right first-molar distance
    (should be nearly symmetric) and the within-body landmark-distance differences between
    the systems, per CT-tip variant."""
    fid, title = "val_digitization", "Landmark digitization in both systems"
    ctx = req.ctx
    sym = _table(ctx, "ref_tip", "landmark_symmetry")
    ld = _table(ctx, "ref_tip", "landmark_distances")
    if sym.empty:
        return _empty(fid, title, "no digitization tables")
    def _sys(s: str) -> str:
        return (f"Qualisys, {V.term('tip_variant', s.split(':', 1)[1], 'short')}"
                if s.startswith("qualisys:") else V.term("system", s))

    labels = [f"{_who(ctx, r.pid)} {V.term('jaw', r.jaw, 'short')}, {_sys(r.system)}"
              for r in sym.itertuples()]
    pa = B.strip(labels, [[v] for v in sym["asymmetry_mm"]], "(a) Incisor–molar asymmetry",
                 Axis("left − right", MM), ref=0.0)
    pids = sorted(sym["pid"].unique())
    shape_by_rows(pa, list(sym["pid"]), pids)          # X1: shape = participant
    panels = [pa]
    if not ld.empty:
        lab2, vals = [], []
        for (pid, v), d in ld.groupby(["pid", "variant"]):
            lab2.append(f"{_who(ctx, pid)}, {V.term('tip_variant', v, 'short')}")
            vals.append(d["diff_mm"].to_numpy())
        pb = B.strip(lab2, vals, "(b) Within-body distances", Axis("JTS-MM − Qualisys", MM),
                     ref=0.0)
        shape_by_rows(pb, [pid for pid, _ in ld.groupby(["pid", "variant"]).groups], pids)
        panels.append(pb)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=max(1.6, 0.14 * len(labels)),
                      sources=["ref_tip:landmark_symmetry", "ref_tip:landmark_distances"])
    return attach(spec, CaptionSpec(
        "landmark digitization of both systems without any frame: (a) asymmetry of the "
        "digitized incisor-to-first-molar distances, left minus right, per jaw (anatomically "
        "near 0), for JTS-MM and for Qualisys with each calibration-tool tip variant; (b) "
        "differences JTS-MM − Qualisys of all within-body landmark distances",
        [("black markers", "(a) the row's value; (b) the median of the row"),
         ("small grey markers (b)", "one landmark pair"), pshape_text(ctx, sym["pid"]),
         ("vertical line", "0, no difference"),
         ("tip variants", "; ".join(f"{V.term('tip_variant', k, 'short')}: "
                                    f"{V.term('tip_variant', k)}"
                                    for k in ("primary", "shared", "own_origin")))],
        n_text(ctx, sym["pid"]),
        statistic="(a) one value per participant, jaw and system, no summary; (b) median over "
                  "the landmark pairs of a participant and tip variant",
        evidence="descriptive"))


@register("val_kin_aligned_timeseries", "dashboard", "Jaw-in-head over the take, frames aligned",
          params=("pid",), choices={"form": ("timeseries", "multiples")}, form="timeseries",
          **G)
def kin_aligned_timeseries(req: ViewRequest) -> FigureSpec:
    """Frame-aligned result (VD3; of equal value with the own-frame result, user
    2026-10-23): JTS-MM jaw-in-head motion after the constant frame alignment
    (fitted on the other tasks) against Qualisys in the movement tasks -- incisor x, y, z,
    opening rotation and the disagreement; frames outside the evaluated tasks and with the
    CT in view are shaded. Form ``multiples``: one column per task (``validation_r8``)."""
    if str(req.options.get("form", "timeseries")) == "multiples":
        from biomech.viz.views.validation_r8 import kin_aligned_multiples

        return kin_aligned_multiples(req)
    fid, title = "val_kin_aligned_timeseries", "Jaw-in-head over the take, frames aligned"
    pid, a, e = _kin(req, fid, title)
    if e:
        return e
    if "incisor_s_aligned_mm" not in a:
        return _empty(fid, title, "no aligned arrays")
    from biomech import timebase

    t = np.asarray(a["t_s"], float)
    crop = _movement_window(a, t)
    spans = _clip_spans(_spans(a), crop)
    panels = [B.phase_lane(spans, extra=_clip_spans(_sync_spans(a), crop))]
    shade = []
    for key in ("evaluated", "ct_visible"):
        if key in a:
            m = ~np.asarray(a[key], bool) if key == "evaluated" else np.asarray(a[key], bool)
            shade.append(timebase.segments(m))
    for k in range(3):
        p = Panel(f"Incisor {XYZ[k]}", Axis(""), Axis(XYZ[k][0], MM), [], sharex="time",
                  legend="below" if k == 0 else None)
        _line(p, t, a["incisor_s_aligned_mm"][:, k], "s", label="JTS-MM (aligned)" if k == 0
              else None)
        _line(p, t, a["incisor_q_mm"][:, k], "q", label="Qualisys" if k == 0 else None)
        B.add_phase_bands(p, spans)
        panels.append(p)
    pr = Panel("Opening rotation", Axis(""), Axis("ry", DEG), [], sharex="time")
    _line(pr, t, a["rot_s_aligned_deg"][:, 1], "s")
    _line(pr, t, a.get("rot_q_aligned_deg", a["rot_q_deg"])[:, 1], "q")
    B.add_phase_bands(pr, spans)
    pe = Panel("Incisor disagreement after alignment", Axis(""), Axis("|Δ|", MM), [],
               sharex="time")
    pe.series.append(Series("line", t, a["err_incisor_aligned_mm"], role="accent",
                            style={"width": 0.5}))
    _tol_lines(pe, "mm")
    # the condyle points in their own row (in one panel the incisor trace hid them, or they
    # hid it), in their landmark colors
    pc = Panel("Condyle-point disagreement after alignment", Axis(""), Axis("|Δ|", MM), [],
               sharex="time", legend="below")
    for n_ in ("tmj_left", "tmj_right"):
        if f"{n_}_s_aligned_mm" in a and f"{n_}_q_mm" in a:
            d = np.linalg.norm(a[f"{n_}_s_aligned_mm"] - a[f"{n_}_q_mm"], axis=1)
            # the right trace drawn second and translucent, so the left one stays visible
            pc.series.append(Series("line", t, d, role=f"entity:{n_}",
                                    style={"width": 0.6,
                                           **({"alpha": 0.6} if n_ == "tmj_right" else {})},
                                    label=f"condyle point {n_[4:]}"))
    _tol_lines(pc, "mm")
    pa = Panel("Rotation disagreement after alignment", Axis("Time in take", SEC),
               Axis("|Δr|", DEG), [], sharex="time")
    pa.series.append(Series("line", t, a["err_rot_aligned_deg"], role="cat:2",
                            style={"width": 0.5}))
    _tol_lines(pa, "deg")
    for q in (pe, pc, pa):
        for runs in shade:
            if runs:
                q.series.insert(0, Series("hspan", np.array([t[r[0]] for r in runs]),
                                          y2=np.array([t[r[1] - 1] for r in runs]),
                                          role="context", style={"alpha": 0.15}))
        B.add_phase_bands(q, spans)
    panels += [pr, pe] + ([pc] if pc.series[:-2] else []) + [pa]
    # Only the movement tasks carry the alignment: the time axis is cropped to them (rest
    # included for context), so the landmark holds and sync windows no longer take most of
    # the width.
    if crop is not None:
        for q in panels:
            q.x.lim = crop
    persp = f"{_who(req.ctx, pid)}, comparison take"
    # 0.55 in per row: the eight rows and the caption on one page
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=0.55,
                      sources=["ref_kinematics:arrays"], perspective=persp, meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "jaw-in-head motion of JTS-MM, its maxillary and mandibular frames aligned to "
        "Qualisys' by constant offsets fitted on the other movement tasks, and of Qualisys, from "
        "rest through the movement tasks (holds and sync windows before are not shown): "
        "incisor x, y, z, opening rotation, and the incisor, condyle-point and rotation "
        "disagreement with the bounds",
        [("solid", "JTS-MM (aligned)"), ("dashed", "Qualisys"),
         ("colored lines (condyle row)", "left and right condyle point (legend)"),
         ("dashed horizontal line", "application bound (labeled with its value)"),
         ("thin grey horizontal line", f"metrology bound ({TOL['metrology'][0]:g} mm, "
                                       f"{TOL['metrology'][1]:g}°)"),
         ("colored bands", "rest and movement tasks (lane on top)"),
         ("grey", "not evaluated or calibration tool (CT) in view")],
        persp, statistic="raw per-frame signals of one take, no summary",
        # an example take, not a selection: say so and where the other is (2026-11-05)
        notes=[f"An example take; the same figure for "
               f"{', '.join(_who(req.ctx, q) for q in sorted(req.ctx.pids()) if q != pid)} "
               "is in the supplement, and the agreement statistics of both takes are in the "
               "tables"] if len(req.ctx.pids()) > 1 else []))



# == (I) point marking (VD4) =====================================================================
LM_GLOSS = ("L1 / U1: lower / upper incisor, L2 / U2: lower / upper left molar, L3 / U3: "
            "lower / upper right molar, TMJ L / R: left / right condyle, (head): condyle point "
            "held in the head frame")


def _pid_rule(panels: list[Panel], pids: Sequence[str]) -> None:
    """A thin rule between the rows of consecutive participants (rows sorted by participant)."""
    pids = list(pids)
    cuts = [i - 0.5 for i in range(1, len(pids)) if pids[i] != pids[i - 1]]
    for p in panels:
        for c in cuts:
            p.series.append(Series("hline", y=np.array([c]), role="context",
                                   style={"width": 0.6}))


def _lm_label(name: str) -> str:
    """Short landmark label; condyle points held in the head frame are marked "(head)"."""
    base = name[:-3] if name.endswith("_hp") else name
    return V.phase_label(base, short=True) + (" (head)" if name.endswith("_hp") else "")


@register("val_marking_accuracy", "dashboard", "Point marking: difference per landmark",
          choices={"form": ("components", "arrows3d", "ellipsoids3d")}, form="components",
          **G)
def marking_accuracy(req: ViewRequest) -> FigureSpec:
    """JTS-MM point-marking error per landmark (Qualisys = reference, EV1/EV2 applied):
    components along the Qualisys anatomical axes and |e|, each with its 95 % CI; σ_Q of
    the reference landmark beside |e|. Form ``arrows3d``: the error vectors at the landmarks
    in 3-D (``validation_gallery``); ``ellipsoids3d``: with their CI ellipsoids
    (``validation_r8``, J2.4)."""
    if req.options.get("form", "components") == "ellipsoids3d":
        from biomech.viz.views.validation_r8 import marking_ellipsoids3d

        return marking_ellipsoids3d(req)
    if req.options.get("form", "components") == "arrows3d":
        from biomech.viz.views.validation_gallery import marking_arrows3d

        return marking_arrows3d(req)
    fid, title = "val_marking_accuracy", "Point marking: difference per landmark"
    ctx = req.ctx
    e = _table(ctx, "ref_landmarks", "landmark_errors")
    if e.empty:
        return _empty(fid, title, "no landmark errors")
    labels = [f"{_who(ctx, r.pid)} {_lm_label(r.landmark)}" for r in e.itertuples()]
    panels = []
    for ax, lab in (("anterior", "(a) Anterior"), ("left", "(b) Left"),
                    ("superior", "(c) Superior")):
        panels.append(B.forest(labels, e[f"e_{ax}_mm"], e[f"e_{ax}_lo_mm"],
                               e[f"e_{ax}_hi_mm"], lab, Axis("JTS-MM − Qualisys", MM),
                               role="accent"))
    pn = B.forest(labels, e["e_mm"], e["e_lo_mm"], e["e_hi_mm"], "(d) |e| and σ_Q",
                  Axis("Distance", MM), ref=None, role="accent")
    pn.series.append(Series("scatter", e["sigma_ref_mm"].to_numpy(float),
                            np.arange(len(e), dtype=float), role="muted",
                            style={"ms": "s", "size": 1.0}, label="σ_Q (Qualisys)"))
    pn.legend = "below"
    panels.append(pn)
    for p in panels[1:]:
        p.y.ticklabels = [""] * len(labels)
    _pid_rule(panels, list(e["pid"]))
    for p in panels:                        # X1: shape = participant
        shape_by_rows(p, list(e["pid"]), sorted(e["pid"].unique()))
    spec = FigureSpec(fid, title, panels, rows=1, cols=4, width="text",
                      height_in=B.rows_height(len(e), per=0.14, extra=0.7, lo=1.8),
                      sources=["ref_landmarks:landmark_errors"])
    return attach(spec, CaptionSpec(
        "point-marking difference of JTS-MM from Qualisys: the JTS-MM landmark mapped into the "
        "Qualisys body frame through the motion-based body registration, minus the Qualisys "
        "landmark, along the Qualisys anatomical axes (a–c) and as distance |e| (d)",
        [("black markers with whiskers", "difference with its 95 % CI"),
         ("grey markers (d)", "σ_Q of the Qualisys landmark (marker noise and tip SE)"),
         pshape_text(ctx, e["pid"]),
         ("thin horizontal rule", "between the participants"),
         ("rows", LM_GLOSS)],
        f"n = {e['pid'].nunique()} participants, {len(e)} landmarks",
        statistic="95 % CIs (hold-frame and registration block bootstrap)",
        notes=["Both systems digitize the same still window; calibration-tool label swaps "
               "corrected."]))


@register("val_marking_frames", "dashboard", "Point marking: anatomical frames",
          choices={"form": ("forest", "dumbbell")}, form="forest", **G)
def marking_frames(req: ViewRequest) -> FigureSpec:
    """Angle and origin errors of the anatomical frames built from the marked points
    (95 % CI), next to the constant offsets the frame alignment estimates from motion (VD3)
    -- two independent estimates of the same marking difference. Form ``dumbbell``: the two
    angle estimates joined per frame."""
    if req.options.get("form", "forest") == "dumbbell":
        from biomech.viz.views.validation_gallery import marking_frames_dumbbell

        return marking_frames_dumbbell(req)
    fid, title = "val_marking_frames", "Point marking: anatomical frames"
    ctx = req.ctx
    f = _table(ctx, "ref_landmarks", "frame_errors")
    s = _table(ctx, "ref_landmarks", "marking_summary")
    if f.empty:
        return _empty(fid, title, "no frame errors")
    lab = {"F_max": "maxillary", "F_mand": "mandibular"}
    labels = [f"{_who(ctx, r.pid)} {lab.get(r.frame, r.frame)}" for r in f.itertuples()]
    pa = B.paired_forest(labels, (f["angle_deg"], f["angle_lo_deg"], f["angle_hi_deg"]),
                         (f["vd3_rot_deg"], np.full(len(f), np.nan), np.full(len(f), np.nan)),
                         "(a) Frame angle", Axis("Angle", DEG),
                         label_a="from the marked points", label_b="from motion")
    pb = B.forest(labels, f["origin_mm"], f["origin_lo_mm"], f["origin_hi_mm"],
                  "(b) Frame origin", Axis("Distance", MM), ref=None, role="accent")
    pb.y.ticklabels = [""] * len(labels)
    for q in (pa, pb):                      # X1: shape = participant
        shape_by_rows(q, list(f["pid"]), sorted(f["pid"].unique()))
    notes = []
    if not s.empty:
        notes.append("Offset at rest caused by the marking alone (incisor displacement / "
                     "jaw-in-head rotation angle): " + "; ".join(
            f"{_who(ctx, r.pid)} {r.rest_incisor_offset_mm:.1f} mm / "
            f"{r.rest_rotation_offset_deg:.1f}°" for r in s.itertuples()) + ".")
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(f), per=0.3, extra=0.8, lo=1.4),
                      sources=["ref_landmarks:frame_errors", "ref_landmarks:marking_summary"])
    return attach(spec, CaptionSpec(
        "how far apart the anatomical frames are that JTS-MM and Qualisys build from their "
        "marked points: (a) rotation angle and (b) origin distance per participant and frame "
        "(maxillary, mandibular), next to the constant frame offset estimated independently "
        "from the motion data by the frame alignment",
        [("black marker with whiskers", "from the marked points, with its 95 % CI"),
         ("grey marker (a)", "from motion"), pshape_text(ctx, f["pid"])],
        n_text(ctx, f["pid"], f"{f['frame'].nunique()} frames each"),
        statistic="marked points: point estimate with a 95 % bootstrap CI over the hold frames "
                  "and the registration; motion: median over the leave-one-task-out folds",
        notes=notes))


@register("val_marking_precision", "dashboard", "Point marking: static precision", **G)
def marking_precision(req: ViewRequest) -> FigureSpec:
    """Within-hold dispersion of the tip in the body frame (MAD), both systems, and the
    frame-free within-body distance differences."""
    fid, title = "val_marking_precision", "Point marking: static precision"
    ctx = req.ctx
    pr = _table(ctx, "ref_landmarks", "landmark_precision")
    dd = _table(ctx, "ref_landmarks", "landmark_distances")
    if pr.empty:
        return _empty(fid, title, "no precision table")
    w = pr.pivot_table(index=["pid", "landmark"], columns="system", values="mad_mm").reset_index()
    labels = [f"{_who(ctx, r.pid)} {_lm_label(r.landmark)}" for r in w.itertuples()]
    nan = np.full(len(w), np.nan)
    pa = B.paired_forest(labels, (w.get("jtsmm", nan), nan, nan),
                         (w.get("qualisys", nan), nan, nan), "(a) Within-hold MAD",
                         Axis("MAD", MM), label_a="JTS-MM", label_b="Qualisys",
                         roles=("system:jtsmm", "system:qualisys"))
    # X1: shape = participant, colour = system
    pids = sorted(pr["pid"].unique())
    shape_by_rows(pa, list(w["pid"]), pids)
    panels = [pa]
    if not dd.empty:
        dl = [f"{_who(ctx, r.pid)} {_lm_label(r.a)}–{_lm_label(r.b)}" for r in dd.itertuples()]
        pb = B.strip(dl, [[v] for v in dd["diff_mm"]], "(b) Distance difference",
                     Axis("JTS-MM − Qualisys", MM), ref=0.0)
        shape_by_rows(pb, list(dd["pid"]), pids)
        panels.append(pb)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=B.rows_height(max(len(w), len(dd)), per=0.12, extra=0.7, lo=1.6),
                      sources=["ref_landmarks:landmark_precision",
                               "ref_landmarks:landmark_distances"])
    return attach(spec, CaptionSpec(
        "static precision and frame-free consistency of the point marking: (a) per landmark "
        "hold the dispersion of the tip in each system over the common still window; (b) "
        "differences JTS-MM − Qualisys of the within-body landmark distances (no registration "
        "involved)",
        [("blue (a)", "JTS-MM"), ("black (a)", "Qualisys"), pshape_text(ctx, pids),
         ("marker (b)", "one landmark pair"), ("vertical line (b)", "0, no difference")],
        n_text(ctx, w["pid"], f"{len(w)} landmarks"),
        statistic="(a) MAD: median distance of the tip from its median over the hold's "
                  "frames; (b) one value per landmark pair, no summary"))


@register("val_tip_holds", "dashboard", "Point marking: what explains the hold differences", **G)
def tip_holds(req: ViewRequest) -> FigureSpec:
    """Per hold: the tooth-hold tip residual and the marking error against the covariates
    (CT orientation disagreement × lever, JTS-MM CT reprojection RMS, the smallest Qualisys
    viewing obliquity of the CT plate); descriptive."""
    fid, title = "val_tip_holds", "Point marking: what explains the hold differences"
    ctx = req.ctx
    at = _table(ctx, "ref_landmarks", "hold_attribution")
    hr = _table(ctx, "ref_tip", "hold_residuals")
    if at.empty:
        return _empty(fid, title, "no hold attribution")
    panels = []
    if not hr.empty:
        labels = [f"{_who(ctx, r.pid)} {_lm_label(r.landmark)}" for r in hr.itertuples()]
        pa = B.strip(labels, [[v] for v in hr["r_mm"]], "(a) Tooth-hold tip residual",
                     Axis("|r|", MM), ref=0.0)
        shape_by_rows(pa, list(hr["pid"]), sorted(at["pid"].unique()))     # X1
        panels.append(pa)
    # short titles (they wrapped in the narrow panels); the axis says what is plotted
    for col, lab, ax, unit in (
            ("predicted_tip_mm", "Tool orientation", "|ΔR| × lever", MM),
            ("jtsmm_ct_reprojection_rms", "JTS-MM reprojection", "Reprojection RMS",
             "\\pixel"),
            ("qualisys_min_obliquity_deg", "Qualisys view angle", "Smallest view angle", DEG)):
        if col not in at:
            continue
        p = Panel(f"({chr(97 + len(panels))}) {lab}", Axis(ax, unit),
                  Axis("Marking difference |e|", MM), [], legend="below")
        for i, (pid, d) in enumerate(at.groupby("pid")):
            p.series.append(Series("scatter", d[col].to_numpy(float), d["e_mm"].to_numpy(float),
                                   role="accent", style={"ms": "os"[i % 2]},
                                   label=_who(ctx, pid)))
        panels.append(p)
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=2.0,
                      sources=["ref_landmarks:hold_attribution", "ref_tip:hold_residuals"])
    return attach(spec, CaptionSpec(
        "what explains the point-marking differences, per landmark hold: (a) the residual of the "
        "joint tip fit on the tooth holds; (b–d) the marking difference against candidate causes: "
        "(b) the calibration-tool orientation disagreement between the systems times the tip "
        "lever, (c) the JTS-MM reprojection RMS of the calibration tool and (d) the smallest "
        "angle between a Qualisys camera's line of sight and the calibration-tool marker plate",
        [("markers", "one hold"), pshape_text(ctx, at["pid"])],
        n_text(ctx, at["pid"], f"{len(at)} holds"),
        statistic="one value per hold, no summary", evidence="descriptive"))


@register("val_table_marking", "dashboard", "Table: point-marking accuracy", **G)
def table_marking(req: ViewRequest) -> TableSpec:
    return _tab(req, "ref_landmarks", "marking_summary", "val_table_marking", CaptionSpec(
                "point-marking accuracy of JTS-MM against Qualisys per participant: RMS "
                "difference of the within-body tooth distances (registration-free), the "
                "maxillary and mandibular anatomical-frame angle differences, the incisor offset at "
                "rest they cause, and the static precision (MAD) of both systems",
                statistic="RMS over the tooth-distance pairs; frame angles and rest offset are "
                          "single estimates; precision: median over the landmark holds of the "
                          "within-hold MAD",
                notes=["Registered point differences and σ_Q in the supplementary source data.",
                       "Last row: the descriptive mean of the participants (not pooled)."]),
                columns=["pid", "rms_dist_diff_teeth_mm", "frame_max_deg", "frame_mand_deg",
                         "rest_incisor_offset_mm", "precision_jtsmm_mad_mm",
                         "precision_qualisys_mad_mm"], mean_of=True)
