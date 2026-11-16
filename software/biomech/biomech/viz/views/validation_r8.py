# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper perspectives and claim-gap views of (J2.4 systems 3-D, J2.5 systems
gaps, J2.6 forms); group ``validation``.

Alternative forms of existing validation views (reached through each view's ``form`` picker;
the original view keeps its drawing as the default and dispatches here):

* ``val_bench``: ``dumbbell`` (both systems joined per condition) and ``strip`` (every step);
* ``val_sync_overview``: ``compact`` (residual lag over the take and its distribution);
* ``val_kin_aligned_timeseries``: ``multiples`` (one column per movement task);
* ``val_budget``: ``stacked`` (variance shares) and ``dot`` (the three terms joined, log);
* ``val_hinge``: ``dumbbell`` and ``difference`` (what tracking adds);
* ``val_camerafree``: ``ladder`` (rungs in order) and ``slope`` (optics to rung);
* ``val_emg_profiles``: ``overlay`` (all muscles of a group) and ``difference`` (with SPM);
* ``val_sampling``: ``quantities`` (one panel per peak quantity, acceleration included when
  stored) and ``bandwidth`` (filtered vs unfiltered f95 and the power above Nyquist);
* ``val_imu_alone``: ``dumbbell`` and ``limits`` (bias ± 1.96 SD);
* ``val_error_budget``: ``ratio`` (adequacy ratio) and ``heatmap``;
* ``val_fusion_segments``: ``paired`` (stored paired differences) and ``dumbbell``;
* ``val_emg_side``: ``heatmap`` and ``correlations`` (the four shape correlations);
* ``val_marking_accuracy``: ``ellipsoids3d`` (error arrows with their CI ellipsoids, J2.4);
* ``val_sync_ct3d``: ``geometry`` (CT path and the CT marker triangle, J2.4).

New views (J2.5 / J2.4): ``val_frame_intervals``, ``val_label_swaps``,
``val_errors_vs_effects`` (the key figure: each systems error next to the dataset effect it
must resolve; dataset values from the dataset paper's pinned store), ``val_fusion_accuracy``,
``val_mounting``, ``val_marking_causes``, ``val_paths3d_error``.

Stored results only; where a stage output only arrives with a later run (``ref_diagnostics``,
``ref_sampling:peak_ratio_acc``) the view says so in an empty panel, or (frame timing) applies
the stage's own routine to the stored frame times and says so. n = 2: every value per
participant, at most a descriptive mean of the two, never pooled. Pseudonyms only.
"""

from __future__ import annotations

import re

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.citations import cite_label, tidy
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import (DEG, MM, MSEC, SEC, SYS, TOL, XYZ, _arrays, _empty,
                                          _pid, _sync_spans, _table, _who, n_text)
from biomech.viz.views.validation import pshape_text, shape_by_pid, shape_by_rows

G = {"group": "validation", "domain": "validation"}
PCT, HZ, US = "\\percent", "\\hertz", "\\micro\\second"
DESCRIPTIVE = "descriptive"
SYS_ROLE = {"jtsmm": "system:jtsmm", "qualisys": "system:qualisys", "delsys": "system:delsys"}
SYS_NAME = {"jtsmm": "JTS-MM", "qualisys": "Qualisys", "delsys": "Delsys"}
# one marker for every system: colour is the system, shape the participant
SYS_MARK = {"jtsmm": "o", "qualisys": "o", "delsys": "o"}
EFFECT_ROLE = "tertiary"            # dataset effects (green) next to the JTS-MM errors (blue)


# == helpers =======================================================================================
def _pids(d: pd.DataFrame) -> list[str]:
    return sorted(d["pid"].unique()) if "pid" in d and len(d) else []


def _pm(i: int) -> str:
    """Participant marker: circle for the first, square for the second."""
    return "os"[i % 2]


def _yrows(p: Panel, labels: list[str], blank: bool = False) -> None:
    y = [float(i) for i in range(len(labels))]
    p.y.ticks, p.y.reverse = y, True
    p.y.ticklabels = [""] * len(labels) if blank else list(labels)


def _xcats(p: Panel, labels: list[str]) -> None:
    p.x.ticks = [float(i) for i in range(len(labels))]
    p.x.ticklabels = list(labels)
    p.x.lim = (-0.5, len(labels) - 0.5)


def _tags(panels: list[Panel]) -> None:
    for i, p in enumerate(panels):
        _, t = p.heading()
        p.title, p.tag = t, chr(97 + i)


def _who_list(ctx: Any, pids: Any) -> str:
    return ", ".join(sorted(_who(ctx, p) for p in set(pids)))


def _pending(fid: str, title: str, shows: str, what: str, sources: list[str]) -> FigureSpec:
    """An informative empty figure for a stage output this store does not hold yet."""
    msg = (f"{what} not in this result store: the stage output is stored from the next "
           "validation re-run on")
    p = Panel("", Axis("", lim=(0.0, 1.0), ticks=[]), Axis("", lim=(0.0, 1.0), ticks=[]), [],
              note="Not in this result store yet (next re-run).")
    spec = FigureSpec(fid, title, [p], width="column", height_in=1.0, sources=sources)
    return attach(spec, CaptionSpec(
        shows, [("empty panel", msg)], "no stored values in this result store",
        evidence=DESCRIPTIVE,
        notes=["The figure fills in from the stored stage output without a change of the "
               "view."]))


def _interval(p: Panel, x: Any, y: float, lo: Any, hi: Any, role: str, ms: str,
              label: str | None = None, size: float = 1.2) -> None:
    """One estimate with its interval on a horizontal row."""
    if np.isfinite(lo) and np.isfinite(hi):
        p.series.append(Series("line", np.array([lo, hi], float), np.array([y, y]), role=role,
                               style={"width": 1.1}))
    p.series.append(Series("scatter", np.array([x], float), np.array([y]), role=role,
                           style={"ms": ms, "size": size}, label=label))


def _join(p: Panel, a: float, b: float, y: float) -> None:
    if np.isfinite(a) and np.isfinite(b):
        p.series.append(Series("line", np.array([a, b], float), np.array([y, y]),
                               role="context"))


# == J2.6 forms ====================================================================================
# -- val_bench -------------------------------------------------------------------------------------
def _conds(tt: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted(dict.fromkeys(zip(tt["orientation"], tt["distance"])))


def _cond_label(c: tuple[str, str]) -> str:
    return f"{c[0].capitalize()}, {c[1]}"


def bench_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``dumbbell`` / ``strip`` of ``val_bench``."""
    from biomech.viz.views.validation_emg import _cohort_arrays, _ladder_n

    fid, title = "val_bench", "Bench trueness and precision"
    ctx = req.ctx
    tt = _table(ctx, "ref_trueness", "trueness")
    if tt.empty:
        return _empty(fid, title, "no bench results")
    conds = _conds(tt)
    systems = [s for s in ("jtsmm", "qualisys") if (tt["system"] == s).any()]
    n = _ladder_n(tt)
    if form == "dumbbell":
        labels = [_cond_label(c) for c in conds]
        panels = []
        for col, lo, hi, lab, ax, ref in (
                ("trueness_step_mean_mm", "trueness_ci_lo_mm", "trueness_ci_hi_mm",
                 "Trueness", Axis("Mean step error", MM), 0.0),
                ("accuracy_rms_mm", None, None, "Accuracy", Axis("RMS", MM), None),
                ("precision_static_sd_3d_mm", None, None, "Precision",
                 Axis("Static 3-D SD", MM), None)):
            if col not in tt:
                continue
            first = not panels
            p = Panel(lab, ax, Axis(""), [], legend="below" if first else "none")
            vals: dict[str, list[tuple[float, float, float]]] = {}
            for s in systems:
                g = tt[tt["system"] == s].set_index(["orientation", "distance"])
                vals[s] = [(float(g.loc[c, col]) if c in g.index else np.nan,
                            float(g.loc[c, lo]) if lo and c in g.index else np.nan,
                            float(g.loc[c, hi]) if hi and c in g.index else np.nan)
                           for c in conds]
            for i in range(len(conds)):
                if len(systems) == 2:
                    _join(p, vals[systems[0]][i][0], vals[systems[1]][i][0], float(i))
            for s in systems:
                for i, (v, a, b) in enumerate(vals[s]):
                    _interval(p, v, float(i), a, b, SYS_ROLE[s], SYS_MARK[s],
                              label=SYS_NAME[s] if first and i == 0 else None)
            if ref is not None:
                p.series.append(Series("vline", x=np.array([ref]), role="reference"))
            _yrows(p, labels, blank=not first)
            panels.append(p)
        _tags(panels)
        spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                          height_in=B.rows_height(len(conds), per=0.3, extra=0.7, lo=1.5),
                          sources=["ref_trueness:trueness"])
        return attach(spec, CaptionSpec(
            "bench trueness, accuracy and precision of both systems on the hand-measured "
            "ladder per orientation and camera distance, the two systems joined per condition: "
            "(a) mean step error against the nominal step, (b) RMS distance of the static "
            "samples from the fitted ladder line, (c) static 3-D SD within the slots",
            [("blue", "JTS-MM"), ("black", "Qualisys"),
             ("whiskers (a)", "95 % CI of the mean step error"),
             ("grey line", "joins the two systems of a condition"),
             ("vertical line (a)", "0, no error")],
            n, statistic="per condition: mean over the steps with its 95 % CI (a), RMS over the "
                         "static samples (b), SD pooled within the slots (c)"))
    a = _cohort_arrays(ctx, "ref_trueness")
    rows, labels = [], []
    for c in conds:
        for s in systems:
            k = f"ladder__{s}__{c[0]}__{c[1]}"
            if f"{k}__step_dev_mm" in a:
                rows.append((c, s, k))
                labels.append(f"{_cond_label(c)}, {SYS_NAME[s]}")
    if not rows:
        return _empty(fid, title, "no ladder arrays")
    pa = Panel("Step error", Axis("Measured − nominal step", MM), Axis(""), [], legend="below")
    pb = Panel("Residual to the ladder line", Axis("Along the ladder", MM), Axis(""), [],
               legend="none")
    for i, (c, s, k) in enumerate(rows):
        step = np.asarray(a[f"{k}__step_dev_mm"], float)
        res = np.asarray(a[f"{k}__resid_mm"], float)
        r = tt[(tt.system == s) & (tt.orientation == c[0]) & (tt.distance == c[1])]
        u = r.iloc[0][["u_x", "u_y", "u_z"]].to_numpy(float) if len(r) and "u_x" in r else None
        along = res @ u if u is not None else np.linalg.norm(res, axis=1)
        for p, v in ((pa, step), (pb, along)):
            v = v[np.isfinite(v)]
            jit = np.linspace(-0.18, 0.18, v.size) if v.size > 1 else np.zeros(v.size)
            p.series.append(Series("scatter", v, i + jit, role=SYS_ROLE[s],
                                   style={"ms": SYS_MARK[s], "size": 0.6, "alpha": 0.7},
                                   label=SYS_NAME[s] if p is pa and i < len(systems) else None))
        if len(r):
            r = r.iloc[0]
            _interval(pa, r.trueness_step_mean_mm, float(i), r.trueness_ci_lo_mm,
                      r.trueness_ci_hi_mm, "accent", "d",
                      label="mean step error, 95 % CI" if i == 0 else None, size=1.4)
            if "line_resid_rms_mm" in r:
                pb.series.append(Series("scatter", np.array([r.line_resid_rms_mm], float),
                                        np.array([float(i)]), role="accent",
                                        style={"ms": "d", "size": 1.4}))
    for p in (pa, pb):
        p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
    _yrows(pa, labels)
    _yrows(pb, labels, blank=True)
    _tags([pa, pb])
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(rows), per=0.2, extra=0.7, lo=1.6),
                      sources=["ref_trueness:arrays", "ref_trueness:trueness"])
    return attach(spec, CaptionSpec(
        "every step of the hand-measured ladder, both systems, per orientation and camera "
        "distance: (a) measured minus nominal step at the far slot of each step, (b) residual "
        "of the static slot means along the ladder to the metric ladder line (scale fixed to "
        "the ruler)",
        [("dots", "one step (a) or slot (b), color = system (legend)"),
         ("black diamond with whiskers (a)", "mean step error and its 95 % CI"),
         ("black diamond (b)", "RMS of the residuals to the line"),
         ("vertical line", "0, no error")],
        n, statistic="(a) mean over the steps with its 95 % CI; (b) RMS over the slots; every "
                     "step and slot drawn"))


# -- val_sync_overview ---------------------------------------------------------------------------
def sync_overview_compact(req: ViewRequest) -> FigureSpec:
    """Form ``compact`` of ``val_sync_overview``: the residual lag under the fitted clock over
    the take (sync windows shaded) and its distribution, one row."""
    fid, title = "val_sync_overview", "Sync: CT visibility and edges"
    ctx = req.ctx
    pid = _pid(req)
    if pid is None:
        return _empty(fid, title, "no participants")
    a = _arrays(ctx, "ref_sync", pid)
    k = _arrays(ctx, "ref_kinematics", pid)
    sy = _table(ctx, "ref_sync", "sync", [pid])
    if "ct_t_s" not in a or "sliding_t_s" not in a:
        return _empty(fid, title, f"no ref_sync arrays for {_who(ctx, pid)}")
    t0 = float(a["ct_t_s"][0])
    pa = Panel("Residual lag over the take", Axis("Time in take", SEC),
               Axis("t_Q − model(t_S)", MSEC), [], legend="below")
    for _, s0, s1 in (_sync_spans(k) if k else []):
        pa.series.append(Series("hspan", np.array([s0]), y2=np.array([s1]), role="context",
                                style={"alpha": 0.2}))
    st, sr = np.asarray(a["sliding_t_s"], float) - t0, np.asarray(a["sliding_resid_ms"], float)
    pa.series.append(Series("scatter", st, sr, role="faint", style={"size": 0.5},
                            label="sliding 15-s windows"))
    ep = np.asarray(a.get("edge_pairs", np.zeros((0, 3))), float)
    edge_res = np.zeros(0)
    if ep.size and not sy.empty:
        r0 = sy.iloc[0]
        lag = r0["offset_s"] + r0["alpha_ppm"] * 1e-6 * (ep[:, 0] - r0["t_ref_s"])
        edge_res = (ep[:, 1] - ep[:, 0] - lag) * 1e3
        pa.series.append(Series("scatter", ep[:, 0] - t0, edge_res, role="accent",
                                style={"size": 0.9}, label="matched CT visibility edges"))
    pa.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    pb = B.histogram(sr, "Distribution of the window residuals", Axis("t_Q − model(t_S)", MSEC))
    pb.series.append(Series("vline", x=np.array([0.0]), role="reference"))
    if pid in ctx.pids():                 # X1: shape = participant
        shape_by_pid([pa], sorted(ctx.pids()).index(pid))
    _tags([pa, pb])
    persp = f"{_who(ctx, pid)}, comparison take"
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=1.7,
                      sources=["ref_sync:arrays", "ref_sync:sync", "ref_kinematics:arrays"],
                      perspective=persp, meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "whether the fitted affine clock (offset and drift) holds over the whole take: (a) "
        "residual lag of the sliding 15-s windows and of the matched calibration-tool (CT) "
        "visibility edges under the clock, sync windows shaded; (b) the distribution of the "
        "window residuals (centered on 0 and flat over time = the clock model holds)",
        [("faint markers (a)", "sliding 15-s windows"), ("black markers (a)", "matched CT edges"),
         ("marker shape", "the participant's (circle P1, square P2)"),
         ("grey bands (a)", "sync windows (CT waved in view)"),
         ("bars (b)", "count of window residuals"), ("lines", "0, the fitted clock")],
        persp, statistic="raw residuals of one take, no summary; one residual per window and "
                         "per matched edge"))


# -- val_kin_aligned_timeseries ------------------------------------------------------------------
def kin_aligned_multiples(req: ViewRequest) -> FigureSpec:
    """Form ``multiples``: one column per movement task (time from the task start): incisor
    z of both systems (frames aligned), opening rotation, and the incisor disagreement."""
    from biomech.validation.selection import MOVEMENT_TASKS

    fid, title = "val_kin_aligned_timeseries", "Jaw-in-head over the take, frames aligned"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_kinematics", pid) if pid else {}
    if "incisor_s_aligned_mm" not in a:
        return _empty(fid, title, "no aligned arrays")
    t = np.asarray(a["t_s"], float)
    ph = np.asarray(a["phase"]).astype(str)
    ev = np.asarray(a.get("evaluated", np.ones(len(t), bool)), bool)
    tasks = [x for x in V.TASKS if x in MOVEMENT_TASKS and (ph == x).any()]
    rq = a.get("rot_q_aligned_deg", a["rot_q_deg"])
    rows = (("Incisor z", MM, (a["incisor_s_aligned_mm"][:, 2], a["incisor_q_mm"][:, 2])),
            ("Opening rotation", DEG, (a["rot_s_aligned_deg"][:, 1], rq[:, 1])),
            ("|Δ incisor|", MM, (a["err_incisor_aligned_mm"], None)))
    grid: list[list[Panel]] = [[] for _ in rows]
    lims: list[list[float]] = [[] for _ in rows]
    for j, task in enumerate(tasks):
        m = (ph == task) & ev
        if not m.any():
            m = ph == task
        tt = t[m] - t[m][0]
        for r, (lab, unit, (ys, yq)) in enumerate(rows):
            p = Panel(V.phase_label(task) if r == 0 else "",
                      Axis("Time in task" if r == len(rows) - 1 else "", SEC if r == len(rows)
                           - 1 else ""),
                      Axis(lab if j == 0 else "", unit if j == 0 else ""), [],
                      legend="below" if (r == 0 and j == 0) else "none")
            if yq is not None:
                p.series.append(Series("line", tt, np.asarray(yq, float)[m],
                                       role=SYS["q"][1], style={"width": 0.6},
                                       label="Qualisys" if j == 0 else None))
                p.series.append(Series("line", tt, np.asarray(ys, float)[m],
                                       role=SYS["s"][1], style={"width": 0.6},
                                       label="JTS-MM (aligned)" if j == 0 else None))
                lims[r] += [np.nanmin(np.r_[ys[m], yq[m]]), np.nanmax(np.r_[ys[m], yq[m]])]
            else:
                p.series.append(Series("line", tt, np.asarray(ys, float)[m], role="accent",
                                       style={"width": 0.35, "alpha": 0.8}))
                for name, (mm, _deg) in TOL.items():
                    p.series.append(Series("hline", y=np.array([mm]),
                                           role="threshold" if name == "application"
                                           else "reference"))
                lims[r] += [0.0, max(np.nanpercentile(ys[m], 99.5), TOL["application"][0])]
            if j:
                p.hidden_labels = {"y": (lab, unit)}
            grid[r].append(p)
    if not tasks:
        return _empty(fid, title, "no movement tasks")
    for r, lim in enumerate(lims):
        v = np.asarray(lim, float)
        v = v[np.isfinite(v)]
        if v.size:
            pad = 0.05 * (v.max() - v.min() or 1.0)
            for p in grid[r]:
                p.y.lim = (float(v.min() - pad), float(v.max() + pad))
    panels = [p for row in grid for p in row]
    persp = f"{_who(ctx, pid)}, movement tasks"
    spec = FigureSpec(fid, title, panels, rows=len(rows), cols=len(tasks), width="text",
                      height_in=1.0, sources=["ref_kinematics:arrays"], perspective=persp,
                      meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "jaw-in-head motion per movement task after the constant alignment of the JTS-MM "
        "maxillary and mandibular frames to Qualisys' (fitted on the other movement tasks), one "
        "column per task with the time from the task start: incisor z (superior), opening "
        "rotation and the incisor disagreement; shared y range per row",
        [("solid blue", "JTS-MM (aligned)"), ("dashed black", "Qualisys"),
         ("black line (bottom)", "|Δ incisor| per frame"),
         ("dashed horizontal lines (bottom)", "application bound (upper) and metrology "
                                              "bound (lower)")],
        f"{persp}, every evaluated frame of each task",
        statistic="raw per-frame signals of one take, no summary"))


# -- val_budget ------------------------------------------------------------------------------------
def budget_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``stacked`` (shares of the observed variance) and ``dot`` (the three terms joined,
    log axis) of ``val_budget``."""
    from biomech.viz.views import validation_part_i as P

    fid, title = "val_budget", "Error budget per participant"
    ctx = req.ctx
    b = _table(ctx, "ref_budget", "budget")
    if b.empty:
        return _empty(fid, title, "no ref_budget results")
    lines = P.budget_lines(b)
    part_lab = {"marking": "marking", "motion": "motion"}

    def label(r: Any) -> str:
        what = ", ".join(x for x in (P.OUTCOME_SHORT[r.outcome] or "rotation",
                                     P.LINE_SHORT[r.line]) if x)
        return (f"{_who(ctx, r.pid)} {part_lab.get(r.part, r.part)}: {what}"
                + (f" {P.NOT_RESOLVED}" if not np.isfinite(r.sigma_jts_corrected) else ""))

    order = {"marking": 0, "motion": 1}
    if form == "stacked":
        d = lines.sort_values(["pid", "part"], key=lambda c: c.map(order) if c.name == "part"
                              else c, kind="stable").reset_index(drop=True)
        y = np.arange(len(d), dtype=float)
        share = np.clip(d["ref_var_share"].to_numpy(float), 0.0, 1.0) * 100.0
        p = Panel("Shares of the observed variance", Axis("Share of σ_obs²", PCT,
                                                          lim=(0.0, 100.0)),
                  Axis(""), [], legend="below")
        p.series.append(Series("barh", np.full(len(d), 100.0), y, role=P.TERMS[
            "sigma_jts_corrected"][1], label="JTS-MM part (σ_JTS²)", style={"bar_width": 0.6}))
        p.series.append(Series("barh", share, y, role=P.TERMS["sigma_ref"][1],
                               label="Qualisys part (σ_Q²)", style={"bar_width": 0.6}))
        _yrows(p, [label(r) for r in d.itertuples()])
        spec = FigureSpec(fid, title, [p], rows=1, cols=1, width="column",
                          height_in=B.rows_height(len(d), per=0.15, extra=0.8),
                          sources=["ref_budget:budget"])
        return attach(spec, CaptionSpec(
            "error budget per participant as shares of the observed variance: σ_obs² = σ_JTS² "
            "+ σ_Q², the Qualisys part σ_Q²/σ_obs² (capped at 100 %) and the JTS-MM part "
            "the rest, for the point marking (teeth and condyle points) and the motion at rest "
            "and in the movement tasks (incisor, condyle points, rotation)",
            [("black bar", "Qualisys part"), ("blue bar", "JTS-MM part (the rest)"),
             (P.NOT_RESOLVED, "JTS-MM part not resolvable (σ_Q ≥ σ_obs, all Qualisys)")],
            n_text(ctx, b["pid"]),
            statistic="per participant, no pooling; tasks = median over the movement tasks of "
                      "the Qualisys share",
            evidence=P.EVIDENCE))
    panels = []
    for lab, part, outs, unit in P.BUDGET_PANELS:
        d = lines[(lines["part"] == part) & lines["outcome"].isin(outs)]
        if d.empty:
            continue
        d = d.sort_values(["pid", "outcome"], key=lambda c: c.map(
            {o: i for i, o in enumerate(outs)}) if c.name == "outcome" else c, kind="stable")
        first = not panels
        p = Panel(lab.split(") ", 1)[-1], Axis("σ", unit, log=True), Axis(""), [],
                  legend="below" if first else "none")
        for i, r in enumerate(d.itertuples()):
            v = [x for x in (r.sigma_obs, r.sigma_ref, r.sigma_jts_corrected) if np.isfinite(x)]
            if len(v) > 1:
                _join(p, min(v), max(v), float(i))
        for col in ("sigma_obs", "sigma_ref", "sigma_jts_corrected"):
            lab_t, role, ms = P.TERMS[col]
            p.series.append(Series("scatter", d[col].to_numpy(float),
                                   np.arange(len(d), dtype=float),
                                   role="muted" if col == "sigma_obs" else role,
                                   style={"ms": ms or "o", "size": 1.2},
                                   label=lab_t if first else None))
        _yrows(p, [P._budget_panel(ctx, d.iloc[[k]], "", unit).y.ticklabels[0]
                   for k in range(len(d))])
        shape_by_rows(p, list(d["pid"]), sorted(lines["pid"].unique()))     # X1
        panels.append(p)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=B.rows_height(max(len(lines) // 3, 4), per=0.2),
                      sources=["ref_budget:budget"])
    return attach(spec, CaptionSpec(
        "error budget per participant as a dot plot on log axes: the observed JTS-MM vs "
        "Qualisys difference σ_obs, its Qualisys part σ_Q and the JTS-MM part σ_JTS = "
        "√(σ_obs² − σ_Q²) for (a) the point marking and (b, c) the motion at rest and in the "
        "movement tasks",
        [("grey", "observed difference"), ("blue", "JTS-MM part"),
         ("black", "Qualisys part"), pshape_text(ctx, lines["pid"]),
         ("grey line", "spans the terms of a row"),
         (P.NOT_RESOLVED, "JTS-MM part not resolvable (no blue dot)")],
        n_text(ctx, b["pid"]),
        statistic="per participant, no pooling: marking = RMS point difference; rest = bias-removed "
                  "RMS over the gated rest frames; tasks = median over the movement tasks of "
                  "each term",
        evidence=P.EVIDENCE))


# -- val_hinge -------------------------------------------------------------------------------------
def hinge_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``dumbbell`` (both systems joined per participant and model) and ``difference``
    (JTS-MM − Qualisys, what tracking adds) of ``val_hinge``."""
    from biomech.viz.views import validation_part_i as P

    fid, title = "val_hinge", "Hinge models on both systems"
    ctx = req.ctx
    h = _table(ctx, "ref_hinge", "hinge")
    if h.empty or "hinge_fit_mm" not in h:
        return _empty(fid, title, "no ref_hinge results")
    tasks = [t for t in V.TASKS if (h["task"] == t).any()]
    rows = [(pid, col) for pid in _pids(h) for col, _ in P.HINGE_MODELS]
    labels = [f"{_who(ctx, pid)} {dict(P.HINGE_MODELS)[col]}" for pid, col in rows]
    model_role = {"hinge_fit_mm": "model:hinge_fit", "hinge_tmj_mm": "model:hinge_tmj"}
    panels = []
    for k, task in enumerate(tasks):
        first = not panels
        if form == "dumbbell":
            p = Panel(V.term("task", task, "title"), Axis("Incisor residual", MM), Axis(""), [],
                      legend="below" if first else "none")
            vals = {}
            for s in ("jtsmm", "qualisys"):
                vals[s] = np.array([float(h[(h.pid == pid) & (h.task == task) & (h.system == s)]
                                          [col].iloc[0])
                                    if ((h.pid == pid) & (h.task == task) & (h.system == s)).any()
                                    else np.nan for pid, col in rows])
            for i in range(len(rows)):
                _join(p, vals["jtsmm"][i], vals["qualisys"][i], float(i))
            for s in ("qualisys", "jtsmm"):
                p.series.append(Series("scatter", vals[s], np.arange(len(rows), dtype=float),
                                       role=SYS_ROLE[s], style={"ms": SYS_MARK[s],
                                                                "size": 1.4 if s == "qualisys"
                                                                else 1.0},
                                       label=SYS_NAME[s] if first else None))
            shape_by_rows(p, [pid for pid, _ in rows], _pids(h))     # X1
        else:
            d = _table(ctx, "ref_hinge", "hinge_diff")
            p = Panel(V.term("task", task, "title"), Axis("JTS-MM − Qualisys", MM), Axis(""), [],
                      legend="below" if first else "none")
            for col, name in P.HINGE_MODELS:
                dc = "d_" + col
                x = np.array([float(d[(d.pid == pid) & (d.task == task)][dc].iloc[0])
                              if not d.empty and ((d.pid == pid) & (d.task == task)).any()
                              else np.nan for pid, c in rows if c == col])
                yy = np.array([float(i) for i, (_, c) in enumerate(rows) if c == col])
                p.series.append(Series("barh", x, yy, role=model_role[col], label=name
                                       if first else None, style={"bar_width": 0.6}))
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        _yrows(p, labels, blank=k > 0)
        panels.append(p)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=max(1, len(panels)), width="text",
                      height_in=B.rows_height(len(rows), per=0.3),
                      sources=["ref_hinge:hinge"] + (["ref_hinge:hinge_diff"]
                                                     if form == "difference" else []))
    notes = P._hinge_gate_notes(ctx, h)
    if form == "dumbbell":
        cap = CaptionSpec(
            "the dataset's two fixed-hinge models fitted to the same openings on the JTS-MM and "
            "the Qualisys trajectories, each system in its own head frame, joined per "
            "participant and model: the Qualisys residual bounds the anatomical non-hinge "
            "motion, the gap to JTS-MM is what its tracking adds",
            [("blue", "JTS-MM"), ("black", "Qualisys"), pshape_text(ctx, _pids(h)),
             ("grey line", "joins the two systems")],
            n_text(ctx, h["pid"]),
            statistic="per participant, task and model the incisor RMS residual over frames "
                      "sampled from each opening (start to peak), no pooling over participants",
            evidence="exploratory", notes=notes)
    else:
        cap = CaptionSpec(
            "what tracking with JTS-MM adds to the dataset's fixed-hinge residuals: the incisor RMS "
            "residual on JTS-MM minus that on Qualisys over the same openings, per participant, "
            "task and hinge model (best-fit fixed axis, axis through the condyle points)",
            [("bars", "difference JTS-MM − Qualisys, color = hinge model (legend)"),
             ("vertical line", "0, tracking adds nothing")],
            n_text(ctx, h["pid"]),
            statistic="per participant, task and model the difference of the two RMS residuals, "
                      "no pooling over participants", evidence="exploratory", notes=notes)
    return attach(spec, cap)


# -- val_camerafree ------------------------------------------------------------------------------
CF_ORDER = ("F0", "E1", "E2a", "E2b", "E3a", "E3b", "E4hinge", "E4own", "E5", "E6")
CF_OUT = {"opening_abs_err_deg": ("rmse_open_deg", "Opening angle", DEG, False),
          "incisor_err_mm": ("rmse_inc_mm", "Incisor", MM, True)}


def camerafree_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``ladder`` (the rungs in ladder order, one line per participant) and ``slope``
    (optics to each tested rung, one small panel per rung) of ``val_camerafree``."""
    from biomech.viz.views.camerafree import rung

    fid, title = "val_camerafree", "Camera-free estimators vs Qualisys"
    ctx = req.ctx
    d = _table(ctx, "ref_camerafree", "cf_val_segments")
    ni = _table(ctx, "ref_camerafree", "cf_val_noninferiority")
    if d.empty or "estimator" not in d:
        return _empty(fid, title, "no ref_camerafree results")
    d = d[d["segment"] == "movement_pooled"]
    pids = _pids(d)
    margins = ({o: float(g["margin"].iloc[0]) for o, g in ni.groupby("outcome")}
               if "margin" in ni else {})
    panels: list[Panel] = []
    if form == "ladder":
        for o, (col, lab, unit, log) in CF_OUT.items():
            tested = set(ni.loc[ni["outcome"] == o, "estimator"]) if "outcome" in ni else set()
            ests = [e for e in CF_ORDER if (d["estimator"] == e).any()
                    and (e == "F0" or e in tested or not tested)]
            y = np.arange(len(ests), dtype=float)
            first = not panels
            p = Panel(lab, Axis("RMSE vs Qualisys", unit, log=log), Axis(""), [],
                      legend="below" if first else "none")
            for i, pid in enumerate(pids):
                g = d[d["pid"] == pid].set_index("estimator")
                xv = np.array([float(g.loc[e, col]) if e in g.index else np.nan for e in ests])
                # X1: participant = marker shape and line style, not colour
                p.series.append(Series("line", xv, y, role="accent",
                                       style={"width": 0.8, **({"dash": "dashed"} if i else {})}))
                p.series.append(Series("scatter", xv, y, role="accent",
                                       style={"ms": _pm(i), "size": 1.1},
                                       label=_who(ctx, pid) if first else None))
                v = d[(d.estimator == "F0") & (d.pid == pid)][col]
                if o in margins and len(v):
                    p.series.append(Series("vline", x=np.array([float(v.iloc[0]) + margins[o]]),
                                           role="reference" if i else "muted"))
            _yrows(p, [rung(e) for e in ests])
            panels.append(p)
        _tags(panels)
        spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                          height_in=B.rows_height(6, per=0.3),
                          sources=["ref_camerafree:cf_val_segments",
                                   "ref_camerafree:cf_val_noninferiority"])
        marks = [("lines with markers", "one participant (legend; circle and solid line P1, "
                                        "square and dashed line P2), rungs in ladder order "
                                        "from the top"),
                 ("vertical lines", "each participant's optics RMSE plus the non-inferiority "
                                    "margin (grey solid P1, dashed P2)")]
    else:
        tested = list(dict.fromkeys(zip(ni["outcome"], ni["estimator"]))) if not ni.empty \
            else []
        cols = max(1, max((sum(1 for o, _ in tested if o == oo) for oo in CF_OUT), default=1))
        grid: list[Panel] = []
        for o, (col, lab, unit, log) in CF_OUT.items():
            ests = [e for e in CF_ORDER if (o, e) in tested]
            for j, e in enumerate(ests):
                first = not grid
                p = Panel(rung(e), Axis(""), Axis("RMSE vs Qualisys" if j == 0 else "",
                                                  unit if j == 0 else "", log=log), [],
                          legend="below" if first else "none")
                for i, pid in enumerate(pids):
                    g = d[d["pid"] == pid].set_index("estimator")
                    yv = np.array([float(g.loc[x, col]) if x in g.index else np.nan
                                   for x in ("F0", e)])
                    r = ni[(ni.pid == pid) & (ni.estimator == e) & (ni.outcome == o)]
                    ok = bool(r["noninferior"].iloc[0]) if len(r) else False
                    # X1: participant = shape and line style; the decision by colour
                    role = "accent" if ok else "muted"
                    p.series.append(Series("line", np.array([0.0, 1.0]), yv, role=role,
                                           style={"width": 1.0,
                                                  **({"dash": "dashed"} if i else {})}))
                    p.series.append(Series("scatter", np.array([0.0, 1.0]), yv,
                                           role=role, style={"ms": _pm(i), "size": 1.0},
                                           label=_who(ctx, pid) if first else None))
                p.x.ticks, p.x.ticklabels, p.x.lim = [0.0, 1.0], ["optics", "rung"], (-0.4, 1.4)
                if j:
                    p.hidden_labels = {"y": ("RMSE vs Qualisys", unit)}
                grid.append(p)
            if ests and len(grid) % cols and o != list(CF_OUT)[-1]:
                grid += [Panel("", Axis("", ticks=[]), Axis("", ticks=[]), [], legend="none")
                         for _ in range(cols - len(grid) % cols)]
        panels = grid
        _tags([p for p in panels if p.title])
        spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                          width="text", height_in=1.35,
                          sources=["ref_camerafree:cf_val_segments",
                                   "ref_camerafree:cf_val_noninferiority"])
        marks = [("lines", "one participant from the optics to the rung (circle and solid "
                           "line P1, square and dashed line P2)"),
                 ("black / grey", "non-inferior / not non-inferior to the optics")]
    if not panels:
        return _empty(fid, title, "no camera-free rungs")
    m_txt = " and ".join(f"{v:g} {'°' if o.endswith('deg') else 'mm'}"
                         for o, v in margins.items())
    return attach(spec, CaptionSpec(
        "the camera-free estimators on the comparison takes against Qualisys, pooled over the "
        "movement tasks, next to the JTS-MM optics (frames aligned as for the IMU alone; every "
        "estimate and Qualisys through the dataset's low-pass): "
        + ("opening-angle and incisor RMSE along the ladder of rungs" if form == "ladder" else
           "the change from the optics to each tested rung")
        + (f"; non-inferiority margins {m_txt}" if m_txt else "") + "; incisor on a log axis",
        marks, f"n = {len(pids)} participants ({_who_list(ctx, pids)}; descriptive)",
        statistic="per participant and rung the RMSE over all movement-task frames (tasks "
                  "pooled within the participant), no pooling over participants",
        test="non-inferior: upper block-bootstrap 95 % CI of the paired per-frame difference "
             "to the optics below the margin", evidence="exploratory"))


# -- val_emg_profiles -----------------------------------------------------------------------------
def emg_profiles_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``overlay`` (every muscle of a group in one panel, muscle colors, both systems)
    and ``difference`` (Delsys − JTS-MM mean with its pointwise 95 % band and the SPM bars)
    of ``val_emg_profiles``."""
    from biomech.validation.emg import finite_cycles, profile_array_key, profile_key
    from biomech.viz.style import ENTITY
    from biomech.viz.views.validation_emg import CROSS_SESSION, _muscles

    fid, title = "val_emg_profiles", "EMG: normalized activation profiles, both systems"
    ctx = req.ctx
    pid = _pid(req)
    task = req.phase if isinstance(req.phase, str) else "open_close"
    a = _arrays(ctx, "ref_emg", pid) if pid else {}
    if "names" not in a:
        return _empty(fid, title, "no ref_emg arrays")
    names = [str(n) for n in a["names"]]
    keys = [k for k in ([task] if task != "left_right" else
                        [profile_key(task, "left"), profile_key(task, "right")])
            if profile_array_key("jtsmm", k) in a or profile_array_key("delsys", k) in a]
    if not keys:
        return _empty(fid, title, f"no {V.phase_label(task)} profiles")
    clus = _table(ctx, "ref_emg", "spm_clusters", [pid])
    x = np.linspace(0, 100, 101)

    def cycles(s: str, key: str, i: int) -> np.ndarray:
        Y = a.get(profile_array_key(s, key))
        return finite_cycles(np.asarray(Y, float)[:, :, i]) if Y is not None else np.zeros((0,
                                                                                             101))

    def clusters(key: str, m: str) -> list[tuple[float, float, float]]:
        if clus.empty:
            return []
        c = clus[(clus["profile"] == key) & (clus["muscle"] == m)]
        return [(float(q.start_pct), float(q.end_pct), float(q.p)) for q in c.itertuples()]

    panels: list[Panel] = []
    ms = _muscles(names)
    if form == "overlay":
        groups = (("Closers", [m for m in ms if m in V.CLOSERS]),
                  ("Openers", [m for m in ms if m in V.OPENERS]))
        for key in keys:
            side = key.partition("__")[2]
            for gname, gm in groups:
                if not gm:
                    continue
                p = Panel(gname + (f", to the {side}" if side else ""), Axis("Cycle", PCT),
                          Axis("Activation", PCT), [], legend="below")
                for row, m in enumerate(gm):
                    i = names.index(m)
                    for s in ("jtsmm", "delsys"):
                        y = cycles(s, key, i)
                        if not len(y):
                            continue
                        p.series.append(Series("line", x, y.mean(axis=0), role=SYS_ROLE[s],
                                               style={"color": ENTITY.get(m, "ink"),
                                                      "width": 1.0},
                                               label=(V.muscle_label(m) if s == "jtsmm"
                                                      and key == keys[0] else None)))
                    cl = clusters(key, m)
                    if cl:
                        p.series.append(Series("sigbar", np.array([c[0] for c in cl]),
                                               y2=np.array([c[1] for c in cl]),
                                               role=SYS_ROLE["jtsmm"],
                                               style={"row": row,
                                                      "color": ENTITY.get(m, "ink")}))
                if not panels:
                    for s in ("jtsmm", "delsys"):
                        p.series.append(Series("line", np.array([np.nan]), np.array([np.nan]),
                                               role=SYS_ROLE[s], style={"color": "ink2"},
                                               label=f"{SYS_NAME[s]} ("
                                                     + ("solid" if s == "jtsmm" else "dotted")
                                                     + ")"))
                panels.append(p)
        _tags(panels)
        cols = 2
        marks = [("line color", "muscle (legend)"), ("solid", "JTS-MM mean"),
                 ("dotted", "Delsys mean"),
                 ("bars above the curves", "supra-threshold clusters of the SPM test, one row "
                                           "per muscle in its color")]
        stat = "mean over the cycles of each system (spread in the depiction with bands)"
    else:
        for m in ms:
            i = names.index(m)
            for key in keys:
                side = key.partition("__")[2]
                ys, yd = cycles("jtsmm", key, i), cycles("delsys", key, i)
                first = not panels
                p = Panel(V.muscle_label(m) + (f", to the {side}" if side else ""),
                          Axis("Cycle", PCT), Axis("Delsys − JTS-MM", PCT), [],
                          legend="below" if first else "none")
                if len(ys) > 1 and len(yd) > 1:
                    dm = yd.mean(axis=0) - ys.mean(axis=0)
                    se = np.sqrt(yd.var(axis=0, ddof=1) / len(yd)
                                 + ys.var(axis=0, ddof=1) / len(ys))
                    p.series.append(Series("band", x, dm - 1.96 * se, y2=dm + 1.96 * se,
                                           role="accent", label="95 % band" if first else None))
                    p.series.append(Series("line", x, dm, role="accent",
                                           label="mean difference" if first else None))
                p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
                B.add_spm_bars(p, [("JTS-MM vs Delsys" if first else "", clusters(key, m),
                                    "accent")])
                panels.append(p)
        cols = 2
        marks = [("line", "mean difference Delsys − JTS-MM (sign as in the agreement table)"),
                 ("band", "pointwise 95 % band of the difference of two independent means "
                          "(unequal variances, normal approximation)"),
                 ("bars above the curves", "supra-threshold clusters of the SPM test"),
                 ("horizontal line", "0, no difference")]
        stat = "difference of the means over the cycles of each system"
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=1.5,
                      sources=["ref_emg:arrays", "ref_emg:spm_clusters"],
                      perspective=f"{_who(ctx, pid)}, {V.phase_label(task)}", meta={"pid": pid})
    nj = sum(len(cycles("jtsmm", k, 0)) for k in keys)
    nd = sum(len(cycles("delsys", k, 0)) for k in keys)
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of each muscle, both EMG systems, each normalized to its "
        "own maximum: " + ("the means of all closers and of the openers in one panel each"
                           if form == "overlay" else
                           "the difference of the two systems' mean profiles per muscle"),
        marks, f"{_who(ctx, pid)}, {nj} JTS-MM and {nd} Delsys cycles (first muscle; masked "
               "channel-cycles drop out per muscle)",
        statistic=stat, test="two-sample SPM{t} over cycles, unequal variances, p < 0.05",
        notes=[CROSS_SESSION]))


# -- val_sampling ---------------------------------------------------------------------------------
def _frame_rate(ctx: Any) -> float:
    """The JTS-MM frame rate of the comparison takes (median frame interval, ref_kinematics)."""
    r = []
    for pid in ctx.pids():
        t = np.asarray(_arrays(ctx, "ref_kinematics", pid).get("t_s", np.zeros(0)), float)
        if t.size > 1:
            r.append(1.0 / float(np.median(np.diff(t[np.isfinite(t)]))))
    return float(np.median(r)) if r else np.nan


def _chain_text() -> str:
    from biomech.validation.sampling import PIPELINES

    (lp_r, w_r), (lp_d, w_d) = PIPELINES["ref200"], PIPELINES["ref40"]
    return (f"dataset chain: {lp_d:g} Hz zero-phase low-pass and a {w_d:g}-s Savitzky–Golay "
            f"derivative; Qualisys at its own rate: {lp_r:g} Hz low-pass, "
            f"{1e3 * w_r:.0f}-ms window")


def sampling_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``quantities`` (one panel per peak quantity; the peak angular acceleration from
    the stored ratios when present) and ``bandwidth`` (f95 of the 200 Hz reference with and
    without its low-pass, and the share of power above the Nyquist frequency of the optics)."""
    from biomech.validation.sampling import ACC_QUANTITIES, VEL_QUANTITIES
    from biomech.viz.views.validation_r7 import PIPE, QUANTITY, _tasks

    fid, title = "val_sampling", "Sampling: peak velocities and bandwidth"
    ctx = req.ctx
    b = _table(ctx, "ref_sampling", "bias")
    fr = _frame_rate(ctx)
    nyq = fr / 2.0
    if form == "quantities":
        acc = _table(ctx, "ref_sampling", "peak_ratio_acc")
        bb = pd.concat([b, acc], ignore_index=True).drop_duplicates(
            ["pid", "task", "pipeline", "quantity"]) if not acc.empty else b
        if bb.empty:
            return _empty(fid, title, "no ref_sampling tables")
        qlab = {**QUANTITY, "peak_abs_acc_dps2": "angular acceleration"}
        panels = []
        for q in (*VEL_QUANTITIES, *ACC_QUANTITIES):
            d = bb[bb["quantity"] == q] if "quantity" in bb else bb.iloc[:0]
            first = not panels
            p = Panel(f"Peak {qlab.get(q, q)}", Axis("Ratio to Qualisys at 200 Hz", ""), Axis(""), [],
                      legend="below" if first else "none")
            if d.empty:
                p.note = "not stored yet (next re-run)"
                p.x.ticks, p.y.ticks = [], []
                panels.append(p)
                continue
            rows = sorted({(r.pid, r.task) for r in d.itertuples()},
                          key=lambda x: (x[0], _tasks(d.task).index(x[1])))
            for k, (pipe, (lab, role, ms)) in enumerate(PIPE.items()):
                off = -0.15 if k == 0 else 0.15
                for i, (pid, task) in enumerate(rows):
                    r = d[(d.pid == pid) & (d.task == task) & (d.pipeline == pipe)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    _interval(p, r.ratio_median, i + off, r.ratio_ci_lo, r.ratio_ci_hi, role, ms,
                              label=lab if first and i == 0 else None)
            p.series.append(Series("vline", x=np.array([1.0]), role="reference"))
            _yrows(p, [f"{_who(ctx, pid)} {V.phase_label(task)}" for pid, task in rows])
            shape_by_rows(p, [pid for pid, _ in rows], sorted({pid for pid, _ in rows}))
            panels.append(p)
        _tags(panels)
        nrow = max(len({(r.pid, r.task) for r in bb.itertuples()}) // 2, 2)
        spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                          width="text", height_in=B.rows_height(nrow, per=0.25),
                          sources=["ref_sampling:bias", "ref_sampling:peak_ratio_acc"])
        return attach(spec, CaptionSpec(
            "per cycle the peak of each kinematic quantity through the dataset processing "
            f"(JTS-MM frames at {fr:.0f} Hz; {_chain_text()}) relative to Qualisys at 200 Hz, for the "
            "chain applied to Qualisys resampled at the JTS-MM frame times and to JTS-MM; one "
            "panel per quantity (the angular acceleration is the second derivative of the same "
            "fit)",
            [("black", "dataset chain on Qualisys"), ("blue", "dataset chain on JTS-MM"),
             ("whiskers", "95 % cycle-bootstrap CI"),
             ("vertical line", "ratio 1, equal to Qualisys at 200 Hz")],
            n_text(ctx, _pids(bb), "cycles per row as stored"),
            statistic="per participant and task the median ratio over the cycles with its 95 % "
                      "cycle-bootstrap CI, no pooling",
            evidence="post hoc (descriptive)",
            notes=["A ratio below 1 is what the dataset chain misses; a caveat, not a "
                   "correction."]))
    sp = _table(ctx, "ref_sampling", "spectrum")
    su = _table(ctx, "ref_sampling", "spectrum_unfiltered")
    if sp.empty and su.empty:
        return _empty(fid, title, "no ref_sampling spectra")
    pids = sorted(set(_pids(sp)) | set(_pids(su)))
    tasks = _tasks(list(sp.get("task", [])) + list(su.get("task", [])))
    rows = [(pid, t) for pid in pids for t in tasks]
    labels = [f"{_who(ctx, pid)} {V.phase_label(t)}" for pid, t in rows]

    def val(df: pd.DataFrame, pid: str, t: str, col: str) -> float:
        r = df[(df.pid == pid) & (df.task == t)] if not df.empty and col in df else df.iloc[:0]
        return float(r[col].iloc[0]) if len(r) else np.nan

    pa = Panel("f95 of the opening-rate power", Axis("f95", HZ), Axis(""), [], legend="below")
    pb = Panel("Power above the Nyquist frequency", Axis("Share", PCT), Axis(""), [],
               legend="none")
    for i, (pid, t) in enumerate(rows):
        f_f, f_u = val(sp, pid, t, "f95_hz"), val(su, pid, t, "f95_unfilt_hz")
        _join(pa, f_f, f_u, float(i))
        s_f, s_u = val(sp, pid, t, "share_above_20hz"), val(su, pid, t, "share_above_20hz_unfilt")
        _join(pb, 100 * s_f, 100 * s_u, float(i))
        for p, (vf, vu) in ((pa, (f_f, f_u)), (pb, (100 * s_f, 100 * s_u))):
            p.series.append(Series("scatter", np.array([vf]), np.array([float(i)]),
                                   role="cat:0", style={"ms": "o"},
                                   label="with the Qualisys low-pass" if i == 0 and p is pa
                                   else None))
            p.series.append(Series("scatter", np.array([vu]), np.array([float(i)]),
                                   role="cat:1", style={"ms": "s"},
                                   label="unfiltered, noise floor removed" if i == 0 and p is pa
                                   else None))
    if np.isfinite(nyq):
        pa.series.append(Series("vline", x=np.array([nyq]), role="threshold",
                                label=f"Nyquist frequency of the optics ({nyq:.0f} Hz)"))
    _yrows(pa, labels)
    _yrows(pb, labels, blank=True)
    for q in (pa, pb):                     # X1: shape = participant, colour = computation
        shape_by_rows(q, [pid for pid, _ in rows], list(pids))
    _tags([pa, pb])
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(rows), per=0.2),
                      sources=["ref_sampling:spectrum", "ref_sampling:spectrum_unfiltered"])
    return attach(spec, CaptionSpec(
        "movement bandwidth of the 200 Hz Qualisys data per participant and task: (a) the "
        "frequency below which 95 % of the opening-rate power lies and (b) the share of that "
        "power above the Nyquist frequency of the JTS-MM optics, computed with the Qualisys "
        "low-pass and on the unfiltered opening angle with the rest-phase noise floor removed",
        [("blue", "with the Qualisys low-pass"),
         ("orange", "unfiltered, noise floor removed"), pshape_text(ctx, pids),
         ("grey line", "joins the two"),
         ("dashed line (a)", "Nyquist frequency of the optics")],
        n_text(ctx, pids),
        statistic="one Welch-spectrum value per participant and task, no pooling",
        evidence="post hoc (descriptive)",
        notes=["In protrusion and laterotrusion the opening angle hardly moves, so their "
               "spectrum reflects Qualisys noise."]))


# -- val_imu_alone ---------------------------------------------------------------------------------
def imu_alone_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``dumbbell`` (IMU alone and optics against Qualisys joined per participant and
    task, the ``metric`` picked) and ``limits`` (bias ± 1.96 SD per comparison)."""
    from biomech.viz.views.validation_r7 import COMPARISONS, METRICS, _tasks

    fid, title = "val_imu_alone", "IMU alone vs Qualisys: opening angle"
    ctx = req.ctx
    d = _table(ctx, "ref_fusion", "imu_opening")
    if d.empty:
        return _empty(fid, title, "no ref_fusion imu_opening table")
    pids = _pids(d)
    if form == "dumbbell":
        metric = req.options.get("metric", "rmse_deg")
        mlab, mdesc = METRICS[metric]
        rows = [(pid, t) for pid in pids for t in _tasks(d[d.pid == pid]["segment"])]
        p = Panel(mlab, Axis(mlab, DEG), Axis(""), [], legend="below")
        comps = COMPARISONS[:2]
        vals = {c: np.array([float(d[(d.pid == pid) & (d.segment == t) & (d.comparison == c)]
                                   [metric].iloc[0])
                             if ((d.pid == pid) & (d.segment == t) & (d.comparison == c)).any()
                             else np.nan for pid, t in rows]) for c, *_ in comps}
        for i in range(len(rows)):
            _join(p, vals[comps[0][0]][i], vals[comps[1][0]][i], float(i))
        for c, lab, role, ms in comps:
            p.series.append(Series("scatter", vals[c], np.arange(len(rows), dtype=float),
                                   role=role, style={"ms": ms, "size": 1.3}, label=lab))
        if metric == "bias_deg":
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        _yrows(p, [f"{_who(ctx, pid)} {V.phase_label(t)}" for pid, t in rows])
        shape_by_rows(p, [pid for pid, _ in rows], pids)          # X1
        spec = FigureSpec(fid, title, [p], width="column",
                          height_in=B.rows_height(len(rows), per=0.2),
                          sources=["ref_fusion:imu_opening"])
        return attach(spec, CaptionSpec(
            f"opening-angle agreement with Qualisys per participant and movement task "
            f"({mdesc}): the IMU alone (the calibrated head and mouth IMUs, no optical "
            "correction) next to the JTS-MM optics, joined per row",
            [("green", "IMU alone vs Qualisys"), ("blue", "JTS-MM optics vs Qualisys"),
             pshape_text(ctx, pids), ("grey line", "joins the two of a row")],
            n_text(ctx, pids, "all frames of each task"),
            statistic=f"per participant and task the {mdesc} over the task's frames, no "
                      "pooling", evidence="post hoc (descriptive)"))
    panels = []
    for k, pid in enumerate(pids):
        g = d[d.pid == pid]
        tasks = _tasks(g["segment"])
        p = Panel(_who(ctx, pid), Axis("Bias ± 1.96 SD", DEG), Axis(""), [],
                  legend="below" if k == len(pids) - 1 else "none")
        for j, (c, lab, role, ms) in enumerate(COMPARISONS):
            for i, t in enumerate(tasks):
                r = g[(g.segment == t) & (g.comparison == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                _interval(p, r.bias_deg, i + (j - 1) * 0.2, r.bias_deg - 1.96 * r.sd_deg,
                          r.bias_deg + 1.96 * r.sd_deg, role, ms,
                          label=lab if i == 0 else None, size=1.1)
        p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        _yrows(p, [V.phase_label(t) for t in tasks], blank=k > 0)
        shape_by_pid([p], k)                                       # X1
        panels.append(p)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text", height_in=2.1,
                      sources=["ref_fusion:imu_opening"])
    return attach(spec, CaptionSpec(
        "opening-angle bias and spread per participant and movement task: the IMU alone and the "
        "JTS-MM optics against Qualisys and the IMU against the optics, as the mean difference "
        "with its limits of agreement",
        [("markers", "bias (mean difference, first minus second)"),
         ("whiskers", "bias ± 1.96 SD of the per-frame difference"),
         ("green / blue / grey", "IMU vs Qualisys / optics vs Qualisys / IMU vs optics"),
         ("circle / square", "panel of " + " / ".join(_who(ctx, x) for x in pids))],
        n_text(ctx, pids, "all frames of each task"),
        statistic="per participant and task: mean and SD of the per-frame difference (the "
                  "stored bias and SD), limits = bias ± 1.96 SD; frames are autocorrelated, so "
                  "the limits describe the spread, not a CI", evidence="post hoc (descriptive)"))


# -- val_error_budget ------------------------------------------------------------------------------
def error_budget_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``ratio`` (σ_obs/σ_Q with the adequacy line, log axis) and ``heatmap``
    (participant × segment × outcome) of ``val_error_budget``."""
    from biomech.validation.budget import DEFAULTS as BUDGET

    fid, title = "val_error_budget", "Error budget and Qualisys adequacy"
    ctx = req.ctx
    b = _table(ctx, "ref_kinematics", "budget")
    if b.empty or "ratio" not in b:
        return _empty(fid, title, "no budget table")
    b = b[b["kind"].isin(["movement", "static"])]
    adeq = float(B.cfg(ctx, "ref_budget", "adequacy_ratio",
                       default=BUDGET["adequacy_ratio"]))
    outs = [o for o in ("incisor", "rotation", "condyle") if (b["outcome"] == o).any()]
    keys = list(dict.fromkeys(zip(b["pid"], b["segment"])))
    labels = [f"{_who(ctx, p)} {V.phase_label(s, short=True)}" for p, s in keys]
    if form == "heatmap":
        z = np.array([[float(b[(b.pid == p) & (b.segment == s) & (b.outcome == o)]["ratio"]
                             .iloc[0]) if ((b.pid == p) & (b.segment == s) & (b.outcome == o))
                       .any() else np.nan for o in outs] for p, s in keys])
        p = B.heat(z, labels, [o.capitalize() for o in outs], "σ_obs / σ_Q", fmt="{:.1f}",
                   cb=Colorbar("σ_obs / σ_Q", "", lim=(0.0, float(np.nanmax(z)) if
                                                       np.isfinite(z).any() else 1.0)))
        spec = FigureSpec(fid, title, [p], width="column",
                          height_in=B.rows_height(len(keys), per=0.17),
                          sources=["ref_kinematics:budget"])
        marks = [("cells", "adequacy ratio per participant, segment and outcome (value "
                           "printed)")]
    else:
        panels = []
        for k, o in enumerate(outs):
            d = b[b["outcome"] == o].set_index(["pid", "segment"])
            x = np.array([float(d.loc[key, "ratio"]) if key in d.index else np.nan
                          for key in keys])
            p = Panel(o.capitalize(), Axis("σ_obs / σ_Q", "", log=True), Axis(""), [],
                      legend="below" if k == 0 else "none")
            for i, (pid, seg) in enumerate(keys):
                pi = sorted({q for q, _ in keys}).index(pid)
                p.series.append(Series("scatter", np.array([x[i]]), np.array([float(i)]),
                                       role=V.phase_role(seg), style={"ms": _pm(pi),
                                                                      "size": 1.2}))
            p.series.append(Series("vline", x=np.array([adeq]), role="threshold",
                                   label=f"adequacy {adeq:g} : 1" if k == 0 else None))
            _yrows(p, labels, blank=k > 0)
            panels.append(p)
        _tags(panels)
        spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                          height_in=B.rows_height(len(keys), per=0.17),
                          sources=["ref_kinematics:budget"])
        marks = [("markers", "adequacy ratio, color = segment, circle / square = participant"),
                 ("dashed line", f"adequacy criterion {adeq:g} : 1")]
    return attach(spec, CaptionSpec(
        "how well Qualisys resolves the JTS-MM − Qualisys difference per participant, segment "
        "(rest and movement tasks) and outcome: the ratio of the observed disagreement σ_obs to "
        "the propagated Qualisys uncertainty σ_Q",
        marks, n_text(ctx, b["pid"]),
        statistic="one ratio per participant, segment and outcome, no summary; σ_obs is the "
                  "bias-removed RMS over the segment's gated frames",
        notes=[f"Qualisys adequate where the ratio reaches {adeq:g}."]))


# -- val_fusion_segments ---------------------------------------------------------------------------
FUSION_OUT = (("inc", "Incisor, own frames", MM), ("inc_al", "Incisor, frames aligned", MM),
              ("rot", "Rotation, own frames", DEG), ("rot_al", "Rotation, frames aligned", DEG))


def fusion_segments_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``paired`` (the stored paired differences fused − optical with their block-
    bootstrap CIs) and ``dumbbell`` (optical-only and F2 per participant and task)."""
    from biomech.viz.views.validation import EST_ALL

    fid, title = "val_fusion_segments", "Fusion: per-segment difference, optical vs fused"
    ctx = req.ctx
    panels = []
    if form == "paired":
        pr = _table(ctx, "ref_fusion", "paired")
        if pr.empty:
            return _empty(fid, title, "no paired fusion table")
        ests = [e for e in ("F1", "F2", "F2P") if (pr["estimator"] == e).any()]
        rows = [(pid, e) for pid in _pids(pr) for e in ests]
        for k, (oc, lab, unit) in enumerate(FUSION_OUT):
            d = pr[pr["outcome"] == oc]
            if d.empty:
                continue
            p = Panel(lab, Axis("Fused − optical", unit), Axis(""), [], legend="none")
            for i, (pid, e) in enumerate(rows):
                r = d[(d.pid == pid) & (d.estimator == e)]
                if r.empty:
                    continue
                r = r.iloc[0]
                _interval(p, r.mean_diff, float(i), r.ci_lo, r.ci_hi, EST_ALL[e][1],
                          _pm(_pids(pr).index(pid)))
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
            _yrows(p, [f"{_who(ctx, pid)} {EST_ALL[e][0]}" for pid, e in rows],
                   blank=len(panels) % 2 == 1)
            panels.append(p)
        _tags(panels)
        spec = FigureSpec(fid, title, panels, rows=2, cols=2, width="text",
                          height_in=B.rows_height(len(rows), per=0.2),
                          sources=["ref_fusion:paired"])
        return attach(spec, CaptionSpec(
            "what the IMU fusion changes against optical-only per participant: the paired "
            "per-frame change of the jaw-in-head difference to Qualisys (fused minus optical) over the "
            "held-out movement tasks, incisor and rotation, own frames and frames aligned "
            "(negative: the fusion is closer to Qualisys)",
            [("markers with whiskers", "mean paired difference and its 95 % block-bootstrap "
                                       "CI, color = estimator, circle / square = participant"),
             ("vertical line", "0, no change")],
            n_text(ctx, pr["pid"], "frames of the held-out tasks"),
            statistic="per participant and estimator the mean paired difference with a "
                      "block-bootstrap 95 % CI, no pooling", evidence="secondary"))
    s = _table(ctx, "ref_fusion", "segments")
    if s.empty:
        return _empty(fid, title, "no fusion table")
    d = s[(s["query"] == "frames") & s["estimator"].isin(["F0", "F2"])]
    keys = sorted(dict.fromkeys(zip(d["pid"], d["segment"])),
                  key=lambda x: (x[0], V.TASKS.index(x[1]) if x[1] in V.TASKS else 99))
    for oc, lab, unit in FUSION_OUT:
        col = {"inc": "incisor_rms_mm", "inc_al": "incisor_aligned_rms_mm",
               "rot": "rot_rms_deg", "rot_al": "rot_aligned_rms_deg"}[oc]
        if col not in d or not d[col].notna().any():
            continue
        first = not panels
        p = Panel(lab, Axis("RMS", unit), Axis(""), [], legend="below" if first else "none")
        g = d.set_index(["pid", "segment", "estimator"])[col]
        a0 = np.array([float(g.get((pid, sg, "F0"), np.nan)) for pid, sg in keys])
        a2 = np.array([float(g.get((pid, sg, "F2"), np.nan)) for pid, sg in keys])
        for i in range(len(keys)):
            _join(p, a0[i], a2[i], float(i))
        y = np.arange(len(keys), dtype=float)
        p.series.append(Series("scatter", a0, y, role=EST_ALL["F0"][1], style={"ms": "o"},
                               label="optical only" if first else None))
        p.series.append(Series("scatter", a2, y, role=EST_ALL["F2"][1], style={"ms": "d"},
                               label=EST_ALL["F2"][0] if first else None))
        _yrows(p, [f"{_who(ctx, pid)} {V.phase_label(sg, short=True)}" for pid, sg in keys],
               blank=len(panels) % 2 == 1)
        shape_by_rows(p, [pid for pid, _ in keys], sorted({pid for pid, _ in keys}))   # X1
        panels.append(p)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=B.rows_height(len(keys), per=0.17),
                      sources=["ref_fusion:segments"])
    return attach(spec, CaptionSpec(
        "optical-only and fused (Kalman filter with smoother) jaw-in-head RMS against Qualisys "
        "per participant and held-out movement task at the optical frames, joined per row; "
        "incisor and rotation, own frames and frames aligned",
        [("grey", "optical only"), ("green", "fused"),
         pshape_text(ctx, [pid for pid, _ in keys]),
         ("grey line", "joins the two of a row")],
        n_text(ctx, d["pid"], f"{d['segment'].nunique()} held-out tasks each"),
        statistic="per participant and task the RMS over the task's optical frames, no "
                  "pooling", evidence="secondary"))


# -- val_emg_side ---------------------------------------------------------------------------------
def emg_side_form(req: ViewRequest, form: str) -> FigureSpec:
    """Forms ``heatmap`` (side index per task and muscle pair, one panel per participant) and
    ``correlations`` (the four shape correlations behind it) of ``val_emg_side``."""
    from biomech.viz.views.validation_part_i import _profile_label

    fid, title = "val_emg_side", "EMG: side check of the montage against Delsys"
    ctx = req.ctx
    d = _table(ctx, "ref_emg_side", "side_check")
    if d.empty:
        return _empty(fid, title, "no ref_emg_side results")
    d = d.assign(side=d["side"].fillna(""))
    keys = list(dict.fromkeys(zip(d["task"], d["side"])))
    order = {t: i for i, t in enumerate(V.TASKS)}
    keys.sort(key=lambda k: (order.get(k[0], len(order)), k[1]))
    informative = {(r.task, r.side) for r in d[d["side_informative"].astype(bool)].itertuples()}
    labels = [_profile_label(t, s) + (" *" if (t, s) in informative else "") for t, s in keys]
    pairs = list(dict.fromkeys(d["muscle"]))
    pids = _pids(d)
    panels = []
    if form == "heatmap":
        for k, pid in enumerate(pids):
            g = d[d.pid == pid].set_index(["task", "side", "muscle"])["side_index"]
            z = np.array([[float(g.get((t, s, m), np.nan)) for m in pairs] for t, s in keys])
            p = B.heat(z, labels if k == 0 else [""] * len(keys),
                       [V.term("muscle", m, "title") for m in pairs], _who(ctx, pid),
                       fmt="{:+.1f}", cb=Colorbar("Side index", "", lim=(-2.0, 2.0),
                                                  ramp="diverging"))
            panels.append(p)
        marks = [("cells", "side index per task and muscle pair (value printed); diverging "
                           "colors: +2 sides as in the montage, −2 swapped, 0 grey"),
                 ("*", "side-informative task (laterotrusion)")]
        stat = "one side index per participant, task and muscle pair from the four correlations"
        height = B.rows_height(len(keys), per=0.22)
    else:
        rows = [(pid, t, s) for pid in pids for t, s in keys]
        for k, m in enumerate(pairs):
            p = Panel(V.term("muscle", m, "title"), Axis("Shape correlation r", "",
                                                         lim=(-1.05, 1.05)), Axis(""), [],
                      legend="below" if k == 0 else "none")
            g = d[d.muscle == m].set_index(["pid", "task", "side"])
            for col, role, ms, lab in (("r_rr", "tertiary", "o", "same side (R–R, L–L)"),
                                       ("r_ll", "tertiary", "o", None),
                                       ("r_rl", "quaternary", "s", "opposite side (R–L, L–R)"),
                                       ("r_lr", "quaternary", "s", None)):
                x = np.array([float(g.loc[r, col]) if r in g.index else np.nan for r in rows])
                p.series.append(Series("scatter", x, np.arange(len(rows), dtype=float),
                                       role=role, style={"ms": ms, "size": 1.0},
                                       label=lab if k == 0 else None))
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
            _yrows(p, [f"{_who(ctx, pid)} {_profile_label(t, s)}"
                       + (" *" if (t, s) in informative else "") for pid, t, s in rows],
                   blank=k > 0)
            shape_by_rows(p, [pid for pid, _, _ in rows], list(pids))   # X1
            panels.append(p)
        marks = [("green", "same-side correlations r(R,R), r(L,L)"),
                 ("vermillion", "opposite-side correlations r(R,L), r(L,R)"),
                 pshape_text(ctx, pids),
                 ("*", "side-informative task (laterotrusion)")]
        stat = "Pearson r over the 101 time-normalized points of the two ensemble-mean profiles"
        height = B.rows_height(len(rows), per=0.15)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=height, sources=["ref_emg_side:side_check"])
    return attach(spec, CaptionSpec(
        "side check of the JTS-MM electrode montage against the side-labeled Delsys sensors: "
        "shape correlations r of the ensemble-mean cycle profiles between JTS-MM side a and "
        "Delsys side b, and the side index ½[r(R,R) + r(L,L) − r(R,L) − r(L,R)]",
        marks, n_text(ctx, d["pid"]), statistic=stat + ", no summary",
        evidence="post hoc (descriptive)"))


# -- val_marking_accuracy: error ellipsoids (J2.4) ----------------------------------------------
def _ellipse_loops(c: np.ndarray, r: np.ndarray, n: int = 40) -> list[np.ndarray]:
    """The three principal sections of an axis-aligned ellipsoid (centre ``c``, semi-axes
    ``r``) as closed 3-D loops."""
    t = np.linspace(0.0, 2 * np.pi, n)
    cs, sn = np.cos(t), np.sin(t)
    z = np.zeros(n)
    return [c + np.column_stack([r[0] * cs, r[1] * sn, z]),
            c + np.column_stack([r[0] * cs, z, r[2] * sn]),
            c + np.column_stack([z, r[1] * cs, r[2] * sn])]


def marking_ellipsoids3d(req: ViewRequest) -> FigureSpec:
    """Form ``ellipsoids3d`` of ``val_marking_accuracy``: the error arrows of ``arrows3d`` and,
    at each arrow tip, the 95 % CI of the error vector along the anatomical axes as an
    axis-aligned ellipsoid (its three principal sections)."""
    from biomech.viz.views.validation_gallery import marking_arrows3d

    spec = marking_arrows3d(req)
    e = _table(req.ctx, "ref_landmarks", "landmark_errors")
    if e.empty or not any(s.kind == "arrow3d" for p in spec.panels for s in p.series):
        return spec
    pids = sorted(e["pid"].unique())
    drawn = 0
    for p, pid in zip(spec.panels, pids):
        d = e[e["pid"] == pid]
        for s in [s for s in p.series if s.kind == "arrow3d"]:
            P = np.column_stack([s.x, s.y, s.z])
            for k, D in enumerate(np.asarray(s.d, float)):
                m = d[np.isclose(d["e_anterior_mm"], D[0]) & np.isclose(d["e_left_mm"], D[1])
                      & np.isclose(d["e_superior_mm"], D[2])]
                if m.empty:
                    continue
                r = m.iloc[0]
                semi = np.array([(r[f"e_{a}_hi_mm"] - r[f"e_{a}_lo_mm"]) / 2.0
                                 for a in ("anterior", "left", "superior")], float)
                if not np.isfinite(semi).all():
                    continue
                for L in _ellipse_loops(P[k] + D, semi):
                    p.series.append(Series("line3d", L[:, 0], L[:, 1], z=L[:, 2],
                                           role="quaternary", style={"width": 0.9},
                                           label=("95 % CI of the difference vector"
                                                  if drawn == 0 else None)))
                    drawn += 1
    cap = spec.meta.get("caption", {})
    return attach(spec, CaptionSpec(
        "point-marking difference of JTS-MM from Qualisys at each landmark with its uncertainty, "
        "in the Qualisys anatomical frames (positions from the stored Qualisys landmarks)",
        [("lines", "dental arch through the two molars and the incisor"),
         ("dots", "Qualisys landmarks"),
         ("arrows", "difference vector JTS-MM − Qualisys at true scale"),
         ("vermillion loops at the arrow tip", "95 % CI of the difference along the anterior, left and "
                                         "superior axes, drawn as the three principal sections "
                                         "of an axis-aligned ellipsoid")],
        cap.get("n", n_text(req.ctx, pids)),
        statistic="one difference vector per landmark with its per-axis 95 % CI (hold-frame and "
                  "registration block bootstrap)",
        notes=["The axis-aligned ellipsoid ignores correlations between the axes."]))


# -- val_sync_ct3d: CT geometry (J2.4) -----------------------------------------------------------
def sync_ct3d_geometry(req: ViewRequest) -> FigureSpec:
    """Form ``geometry`` of ``val_sync_ct3d``: the CT path of both systems in 3-D, the CT
    marker triangle from its stored mean inter-marker distances (why labels can swap), and
    the tip distance over the sync windows."""
    from biomech.viz.views.validation import sync_ct3d

    fid, title = "val_sync_ct3d", "Sync: CT path in both systems"
    ctx = req.ctx
    pid = _pid(req)
    base = sync_ct3d(replace(req, options={**req.options, "form": "paths"}))
    geo = _table(ctx, "ref_tip", "ct_geometry", [pid]) if pid else pd.DataFrame()
    if len(base.panels) < 3:
        return base
    p3, _, ptip = base.panels
    p3.title = "CT path, both systems"
    panels = [p3]
    if not geo.empty and {"dist_12_mm", "dist_13_mm", "dist_23_mm"} <= set(geo.columns):
        g = geo.iloc[0]
        d12, d13, d23 = (float(g[c]) for c in ("dist_12_mm", "dist_13_mm", "dist_23_mm"))
        x3 = (d12 ** 2 + d13 ** 2 - d23 ** 2) / (2 * d12)
        y3 = float(np.sqrt(max(d13 ** 2 - x3 ** 2, 0.0)))
        P = np.array([[0.0, 0.0], [d12, 0.0], [x3, y3]])
        c = P.mean(axis=0)
        P = P - c
        pt = Panel("CT marker triangle", Axis("", MM), Axis("", MM), [], equal=True,
                   legend="none")
        loop = np.vstack([P, P[:1]])
        pt.series.append(Series("fill", loop[:, 0], loop[:, 1], role="faint"))
        pt.series.append(Series("line", loop[:, 0], loop[:, 1], role="accent"))
        pt.series.append(Series("scatter", P[:, 0], P[:, 1], role="accent", style={"size": 1.2}))
        for (i, j), dd in (((0, 1), d12), ((0, 2), d13), ((1, 2), d23)):
            mid = 0.5 * (P[i] + P[j])
            out = mid / (np.linalg.norm(mid) or 1.0)
            side = abs(out[0]) > abs(out[1])
            anchor = ("right" if out[0] > 0 else "left") if side else (
                "above" if out[1] >= 0 else "below")
            pt.series.append(Series("text", np.array([mid[0] + 3 * out[0]]),
                                    np.array([mid[1] + 3 * out[1]]), text=[f"{dd:.1f} mm"],
                                    style={"anchor": anchor}))
        if np.isfinite(g.get("label_margin_mm", np.nan)):
            pt.note = f"label margin {float(g['label_margin_mm']):.1f} mm"
        pt.y.lim = (float(P[:, 1].min()) - 12.0, float(P[:, 1].max()) + 22.0)
        panels.append(pt)
    ptip.title = "Tip distance between the systems"
    panels.append(ptip)
    _tags(panels)
    persp = f"{_who(ctx, pid)}, sync windows"
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text", height_in=2.3,
                      sources=["ref_sync:arrays", "ref_tip:ct_geometry"], perspective=persp,
                      meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "the calibration tool (CT) that links the two systems: (a) its tip path in the sync "
        "windows in both systems (Qualisys world after the sync and registration), (b) its "
        "marker triangle drawn from the mean inter-marker distances in Qualisys (a nearly "
        "isosceles triangle lets the motion-capture software swap two labels and still fit "
        "the body) and (c) the tip distance between the systems",
        [("solid blue / dashed black (a)", "JTS-MM / Qualisys"),
         ("triangle (b)", "CT markers with the mean side lengths"),
         ("label margin (b)", "smallest change of the largest inter-marker distance over the "
                              "marker permutations"),
         ("dots (c)", "tip distance per frame")],
        persp, statistic="raw paths and per-frame distances of one take; mean inter-marker "
                         "distances over the CT frames",
        notes=["The positions of the cameras are not part of the stored results, so they are "
               "not drawn."]))


# == J2.5 / J2.4 new views =========================================================================
@register("val_frame_intervals", "dashboard", "JTS-MM camera-frame intervals", **G)
def val_frame_intervals(req: ViewRequest) -> FigureSpec:
    """The JTS-MM camera-frame intervals of the comparison takes: every interval by its length
    in frame periods (gaps are whole dropped frames) and the trigger jitter of the regular
    intervals. Stored ``ref_diagnostics:frame_timing / frame_gaps`` when present, else the
    stage's routine on the stored ``ref_kinematics`` frame times."""
    from biomech.validation import diagnostics as DG

    fid, title = "val_frame_intervals", "JTS-MM camera-frame intervals"
    ctx = req.ctx
    factor = float(B.cfg(ctx, "ref_diagnostics", "long_interval_factor",
                         default=DG.DEFAULTS["long_interval_factor"]))
    ft, fg = _table(ctx, "ref_diagnostics", "frame_timing"), _table(ctx, "ref_diagnostics",
                                                                   "frame_gaps")
    stored = not ft.empty and "frame_interval_ms" in ft
    dts: dict[str, np.ndarray] = {}
    timing, gaps = [], []
    for pid in ctx.pids():
        t = np.asarray(_arrays(ctx, "ref_kinematics", pid).get("t_s", np.zeros(0)), float)
        t = t[np.isfinite(t)]
        if t.size < 3:
            continue
        dts[pid] = np.diff(t)
        if not stored:
            a, b = DG.frame_timing(pid, t, factor)
            timing.append(a)
            gaps.append(b)
    if not stored:
        ft = pd.concat(timing, ignore_index=True) if timing else pd.DataFrame()
        fg = pd.concat(gaps, ignore_index=True) if gaps else pd.DataFrame()
    if ft.empty:
        return _empty(fid, title, "no frame times")
    pids = _pids(ft)
    lengths = sorted({1} | set(fg.get("interval_frames", pd.Series(dtype=int)).astype(int)))
    pa = Panel("Interval length", Axis("Interval", "", ticks=[float(v) for v in lengths],
                                       ticklabels=[f"{v} ({v - 1} dropped)" if v > 1 else
                                                   "1 (regular)" for v in lengths]),
               Axis("Intervals", "", log=True), [], legend="below")
    pb = Panel("Trigger jitter of the regular intervals", Axis("Interval − median", US),
               Axis("Intervals", ""), [], legend="none")
    w = 0.8 / max(len(pids), 1)
    for i, pid in enumerate(pids):
        r = ft[ft.pid == pid].iloc[0]
        g = fg[fg.pid == pid] if not fg.empty else fg
        cnt = {1: float(r["n_intervals"] - r["n_long"])}
        cnt.update({int(x.interval_frames): float(x.n) for x in g.itertuples()})
        xs = np.array([v + (i - (len(pids) - 1) / 2) * w for v in lengths])
        yv = np.array([cnt.get(v, np.nan) for v in lengths])
        for xx, yy in zip(xs, yv):
            pa.series.append(Series("line", np.array([xx, xx]), np.array([1.0, yy]),
                                    role="context", style={"width": 2.0}))
        pa.series.append(Series("scatter", xs, yv, role="accent",
                                style={"ms": _pm(i), "size": 1.2}, label=_who(ctx, pid)))
        if pid in dts:
            dt = dts[pid]
            med = float(np.median(dt))
            reg = (dt[dt <= factor * med] - med) * 1e6
            if reg.size:
                lim = float(np.ceil(np.nanpercentile(np.abs(reg), 99.9))) or 1.0
                cnt_j, edges = np.histogram(reg, bins=np.arange(-lim - 0.5, lim + 1.0, 1.0))
                pb.series.append(Series("line", 0.5 * (edges[:-1] + edges[1:]),
                                        cnt_j.astype(float), role="accent",
                                        style={"width": 1.0, "dash": "dashed" if i else
                                               "solid"}, label=_who(ctx, pid)))
    pa.x.lim = (0.4, max(lengths) + 0.6)
    top = max((float(np.nanmax(s_.y)) for s_ in pa.series if s_.kind == "scatter"), default=10)
    pa.y.lim = (0.7, 3.0 * top)
    _tags([pa, pb])
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text", height_in=1.8,
                      sources=["ref_diagnostics:frame_timing", "ref_diagnostics:frame_gaps",
                               "ref_kinematics:arrays"])
    notes = [] if stored else [
        "The frame-timing stage output is not in this result store; its routine was applied "
        "to the stored frame times (the same computation as the methods table)."]
    return attach(spec, CaptionSpec(
        "the timing of the JTS-MM camera frames in the comparison takes: (a) every frame "
        "interval by its length in frame periods, so a long interval is a whole number of "
        "dropped frames, and (b) the trigger jitter of the regular intervals around their "
        "median",
        [("stems (a)", "count of intervals per length, circle P1, square P2 (log axis)"),
         ("lines (b)", "histogram of the regular intervals' deviation from the median in 1-µs "
                       "bins, P1 solid, P2 dashed")],
        n_text(ctx, pids, "every frame interval of the comparison take"),
        statistic=f"counts per participant, no pooling; regular = not longer than {factor:g} "
                  "times the median interval, length = interval / median rounded to whole "
                  "frames", evidence=DESCRIPTIVE, notes=notes))


@register("val_label_swaps", "dashboard", "Motion-capture label swaps of the calibration tool",
          **G)
def val_label_swaps(req: ViewRequest) -> FigureSpec:
    """The QTM label correction: per participant and Qualisys body the label margin (how far a
    marker permutation is from fitting), the share of CT frames with corrected and
    undetermined labels per segment, and where in the take they lie."""
    from biomech import timebase

    fid, title = "val_label_swaps", "Motion-capture label swaps of the calibration tool"
    ctx = req.ctx
    lc = _table(ctx, "ref_quality", "label_correction")
    cons = _table(ctx, "ref_quality", "consistency")
    if lc.empty:
        return _empty(fid, title, "no label-correction table")
    pids = _pids(lc)
    body = {"CT": "calibration tool", "HP": "head body", "MP": "jaw body"}
    rows = [(pid, b) for pid in pids for b in ("CT", "HP", "MP") if
            ((lc.pid == pid) & (lc.body == b)).any()]
    pa = Panel("Label margin", Axis("Margin", MM, log=True), Axis(""), [], legend="below")
    named: set[bool] = set()
    for i, (pid, b) in enumerate(rows):
        r = lc[(lc.pid == pid) & (lc.body == b)].iloc[0]
        amb = bool(r["ambiguous"])
        pa.series.append(Series("scatter", np.array([float(r["margin_mm"])]),
                                np.array([float(i)]), role="quaternary" if amb else "context",
                                style={"ms": _pm(pids.index(pid)), "size": 1.3},
                                label=(("label-ambiguous" if amb else "unambiguous")
                                       if amb not in named else None)))
        named.add(amb)
    _yrows(pa, [f"{_who(ctx, pid)} {body[b]}" for pid, b in rows])
    panels = [pa]
    if not cons.empty and "frac_label_swapped" in cons:
        c = cons[(cons["body"] == "CT") & ((cons["frac_label_swapped"] > 0)
                                           | (cons["frac_label_undetermined"] > 0))]
        if not c.empty:
            c = c.sort_values(["pid", "kind", "segment"])
            pb = Panel("Calibration-tool frames per segment", Axis("Frames", PCT,
                                                                   lim=(0.0, 100.0)),
                       Axis(""), [], legend="below")
            y = np.arange(len(c), dtype=float)
            pb.series.append(Series("barh", 100 * c["frac_label_swapped"].to_numpy(float),
                                    y - 0.17, role="cat:4", label="labels corrected",
                                    style={"bar_width": 0.32}))
            pb.series.append(Series("barh", 100 * c["frac_label_undetermined"].to_numpy(float),
                                    y + 0.17, role="cat:5", label="labels undetermined",
                                    style={"bar_width": 0.32}))
            _yrows(pb, [f"{_who(ctx, r.pid)} {V.phase_label(r.segment)}"
                        for r in c.itertuples()])
            panels.append(pb)
    pt = Panel("Where in the take", Axis("Time in take", SEC), Axis(""), [], legend="none")
    for i, pid in enumerate(pids):
        a = _arrays(ctx, "ref_quality", pid)
        k = _arrays(ctx, "ref_kinematics", pid)
        if "CT_label_swapped" not in a:
            continue
        t0 = float(k["t_master_s"][0]) if "t_master_s" in k else float(a["t_qm_s"][0])
        t = np.asarray(a["t_qm_s"], float) - t0
        for key, role, dy in (("CT_label_swapped", "cat:4", -0.12),
                              ("CT_label_undetermined", "cat:5", 0.12)):
            m = np.asarray(a.get(key, np.zeros(len(t), bool)), bool)
            for s, e in timebase.segments(m):
                pt.series.append(Series("line", np.array([t[s], t[e - 1]]),
                                        np.array([i + dy, i + dy]), role=role,
                                        style={"width": 4.0}))
    _yrows(pt, [_who(ctx, p) for p in pids])
    pt.y.lim, pt.height = (-0.5, len(pids) - 0.5), 0.5
    panels.append(pt)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=B.rows_height(len(rows), per=0.16, lo=1.1),
                      sources=["ref_quality:label_correction", "ref_quality:consistency",
                               "ref_quality:arrays"])
    return attach(spec, CaptionSpec(
        "why and where the motion-capture software swapped marker labels: (a) the label margin "
        "of each Qualisys body (a small margin means a marker permutation fits almost as well, "
        "so swapped labels go unnoticed), (b) the share of calibration-tool frames per segment "
        "whose labels were corrected or stayed undetermined, and (c) where in the take those "
        "frames lie; the correction is applied before every result",
        [("markers (a)", "one body, circle / square = participant, color = label-ambiguous "
                         "or not (legend)"),
         ("bars (b)", "corrected and undetermined frames (legend)"),
         ("thick segments (c)", "runs of corrected (upper) and undetermined (lower) frames, "
                                "same colors")],
        n_text(ctx, pids),
        statistic="per participant and body one margin; shares over each segment's frames; "
                  "no pooling", evidence=DESCRIPTIVE))


# -- the key figure: errors next to the effects -------------------------------------------------
REPO = Path(__file__).resolve().parents[5]
DATASET_STUDY = REPO / "experiments" / "icorr2027" / "study.yaml"
DATASET_PAPER = REPO / "experiments" / "icorr2027" / "paper_dataset.yaml"
# (row label, unit text, systems feature, the task it is matched to, dataset requirement key)
EFFECTS = (("Condyle-point excursion", "mm", "tmj_excursion_mm", "open_close", "translation_mm"),
           ("Incisal opening", "mm", "amp_mm", "open_close", "opening_mm"),
           ("Protrusion", "mm", "amp_mm", "protrusion_retrusion", "protrusion_mm"),
           ("Laterotrusion", "mm", "amp_mm", "left_right", "laterotrusion_mm"),
           ("Opening rotation", "°", "rot_excursion_deg", "open_close", "opening_angle_deg"))


@lru_cache(maxsize=1)
def dataset_requirements() -> tuple[pd.DataFrame, str]:
    """The dataset paper's requirements table from its pinned store (paper set variant and
    snapshot), and a label of that store; empty when it cannot be opened."""
    try:
        from biomech import papers
        from biomech.config import load_config
        from biomech.io.context import DataContext

        ps = papers.load(DATASET_PAPER)
        c = DataContext(ps.store_config(load_config(DATASET_STUDY)))
        try:
            r = c.table("requirements", "requirements")
        finally:
            close = getattr(c, "close", None)
            if close:
                close()
        return r, str(ps.snapshot or "live")
    except Exception:                       # the dataset store is optional for this view
        return pd.DataFrame(), ""


@register("val_errors_vs_effects", "dashboard", "System differences next to the dataset effects",
          choices={"frames": ("aligned", "own"), "margin": ("median", "task")},
          frames="aligned", margin="median", **G)
def val_errors_vs_effects(req: ViewRequest) -> FigureSpec:
    """Each JTS-MM error (equivalence margin of a cycle feature, per participant) next to the
    dataset effect it must resolve (median and P95 across the dataset participants): condyle
    margin vs condyle-point excursion, incisor margin vs the workspace, rotation margin vs the
    opening rotation; and their ratio."""
    fid, title = "val_errors_vs_effects", "System differences next to the dataset effects"
    ctx = req.ctx
    frames = str(req.options.get("frames", "aligned"))
    how = str(req.options.get("margin", "median"))
    f = _table(ctx, "ref_agreement", "features_aligned" if frames == "aligned" else "features")
    if f.empty or "equiv_margin" not in f:
        return _empty(fid, title, "no feature agreement")
    req_t, snap = dataset_requirements()
    req_t = req_t.set_index("key") if not req_t.empty and "key" in req_t else req_t
    pids = _pids(f)
    rows = [e for e in EFFECTS if (f["feature"] == e[2]).any()]
    labels = [f"{lab} ({u})" for lab, u, *_ in rows]
    pa = Panel("Margin and effect", Axis("Size (mm or °)", "", log=True), Axis(""), [],
               legend="below")
    pb = Panel("Effect / margin", Axis("Dataset median / JTS-MM margin", "", log=True), Axis(""),
               [], legend="none")
    for i, (lab, u, feat, task, key) in enumerate(rows):
        d = f[f["feature"] == feat]
        eff_med = eff_p95 = np.nan
        if not req_t.empty and key in req_t.index:
            eff_med, eff_p95 = float(req_t.loc[key, "median"]), float(req_t.loc[key, "value"])
            pa.series.append(Series("line", np.array([eff_med, eff_p95]), np.array([i, i],
                                                                                    float),
                                    role=EFFECT_ROLE, style={"width": 2.2}))
            pa.series.append(Series("scatter", np.array([eff_med]), np.array([float(i)]),
                                    role=EFFECT_ROLE, style={"ms": "d", "size": 1.5},
                                    label="dataset median" if i == 0 else None))
            pa.series.append(Series("scatter", np.array([eff_p95]), np.array([float(i)]),
                                    role=EFFECT_ROLE, style={"ms": "|", "size": 1.8},
                                    label="dataset P95" if i == 0 else None))
        for j, pid in enumerate(pids):
            g = d[d["pid"] == pid]
            v = (float(g.loc[g["task"] == task, "equiv_margin"].iloc[0])
                 if how == "task" and (g["task"] == task).any()
                 else float(g["equiv_margin"].median()) if len(g) else np.nan)
            y = i + (j - 0.5) * 0.25
            pa.series.append(Series("scatter", np.array([v]), np.array([y]),
                                    role=SYS_ROLE["jtsmm"], style={"ms": _pm(j), "size": 1.2},
                                    label=f"JTS-MM margin, {_who(ctx, pid)}" if i == 0 else None))
            if np.isfinite(eff_med) and np.isfinite(v) and v > 0:
                pb.series.append(Series("scatter", np.array([eff_med / v]), np.array([y]),
                                        role=SYS_ROLE["jtsmm"],   # as in (a)
                                        style={"ms": _pm(j), "size": 1.2}))
    pb.series.append(Series("vline", x=np.array([1.0]), role="reference"))
    _yrows(pa, labels)
    _yrows(pb, labels, blank=True)
    panels = [pa, pb]
    if req_t.empty:
        pa.note = "dataset values not available (dataset store not found)"
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(rows), per=0.32, extra=0.8),
                      sources=[f"ref_agreement:features{'_aligned' if frames == 'aligned' else ''}",
                               "requirements:requirements (dataset paper)"])
    n_ds = int(req_t["n"].max()) if not req_t.empty and "n" in req_t else 0
    fr_txt = ("frames aligned" if frames == "aligned" else "each system in its own frames")
    m_txt = ("the median over the movement tasks" if how == "median" else
             "in the task the dataset quantity comes from (open-close for the opening, the "
             "condyle points and the rotation)")
    return attach(spec, CaptionSpec(
        "whether the JTS-MM − Qualisys differences are small next to the effects the companion data-set study "
        "measures: "
        "(a) per participant the equivalence margin of the matching cycle feature (" + fr_txt
        + "; the smallest symmetric bound the bias CI lies inside, " + m_txt + ") next to the "
        "dataset's value of the same quantity across its participants; (b) the dataset median "
        "divided by each margin (how many times larger the effect is)",
        [("green diamond and bar (a)", "dataset median to P95 across its participants"),
         ("blue circle / square (a)", "JTS-MM margin per participant (legend)"),
         ("blue circle / square (b)", "ratio per participant"),
         ("vertical line (b)", "1, effect as large as the margin")],
        n_text(ctx, pids) + (f"; dataset values: the companion data-set study's cohort of {n_ds} "
                             "(cited)"
                             if n_ds else ""),
        statistic="systems: one margin per participant (no pooling); dataset: median and P95 "
                  "(Harrell–Davis) across its participants, as reported in the companion "
                  "data-set study",
        evidence=DESCRIPTIVE,
        notes=["Condyle row: the dataset's passive translation is the condyle-point excursion "
               "in opening (larger side); workspace rows: incisal displacement per task."]))


@register("val_fusion_accuracy", "dashboard", "Fusion: trajectory accuracy against Qualisys", **G)
def val_fusion_accuracy(req: ViewRequest) -> FigureSpec:
    """Incisor trajectory accuracy per estimator (raw optics, optics with the dataset low-pass,
    rotation fusion, rotation and position fusion) against Qualisys at its frames, per
    participant and held-out task, and the error content above the dataset low-pass."""
    from biomech.validation.evaluate import SMOOTH_HZ
    from biomech.viz.views.validation_r7 import SMOOTH

    fid, title = "val_fusion_accuracy", "Fusion: trajectory accuracy against Qualisys"
    ctx = req.ctx
    sm = _table(ctx, "ref_fusion", "smoothness")
    if sm.empty or "err_rms_mm" not in sm:
        return _empty(fid, title, "no smoothness table")
    from biomech.viz.views.validation import EST_SHORT

    names = EST_SHORT                    # the same short names in every fusion item
    long = {k: lab for k, lab, _ in SMOOTH}
    ests = [e for e in ("F0", "F0_lp6", "F2", "F2P") if (sm["estimator"] == e).any()]
    x = np.arange(len(ests), dtype=float)
    pids = _pids(sm)
    panels = []
    for col, lab in (("err_rms_mm", "RMS difference to Qualisys"),
                     ("jitter_err_rms_mm", f"Difference content above {SMOOTH_HZ:g} Hz")):
        first = not panels
        p = Panel(lab, Axis(""), Axis("RMS", MM), [], legend="below" if first else "none")
        for j, pid in enumerate(pids):
            g = sm[sm.pid == pid]
            for task in [t for t in V.TASKS if (g["segment"] == t).any()]:
                h = g[g.segment == task].set_index("estimator")[col]
                p.series.append(Series("line", x, np.array([float(h.get(e, np.nan))
                                                            for e in ests]),
                                       role=V.phase_role(task),
                                       # X1: line style = participant (P2 dashed)
                                       style={"width": 0.5, "alpha": 0.7,
                                              **({"dash": "dashed"} if j else {})},
                                       label=(V.phase_label(task, short=True)
                                              if first and j == 0 else None)))
            med = np.array([float(g[g.estimator == e][col].median()) for e in ests])
            p.series.append(Series("line", x, med, role="accent",
                                   style={"width": 1.4, **({"dash": "dashed"} if j else {})}))
            p.series.append(Series("scatter", x, med, role="accent",
                                   style={"ms": _pm(j), "size": 1.4},
                                   label=f"{_who(ctx, pid)} median over tasks" if first
                                   else None))
        _xcats(p, [names.get(e, long.get(e, e)) for e in ests])
        panels.append(p)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.0,
                      sources=["ref_fusion:smoothness"])
    return attach(spec, CaptionSpec(
        "how close each incisor trajectory comes to Qualisys at its own frames (frames aligned, "
        "held-out tasks): the raw JTS-MM optics, the optics through the dataset low-pass, and "
        "the IMU fusion of the rotation and of the rotation and position; (a) RMS difference, (b) "
        f"the part of the difference above {SMOOTH_HZ:g} Hz (frame-to-frame noise)",
        [("x axis", f"optical: raw JTS-MM optical trajectory; {SMOOTH_HZ:g}-Hz low-pass: the "
                    "optics through the dataset low-pass; Kalman / Kalman + position: the "
                    "optics fused with the IMUs in the Kalman smoother for the rotation / for "
                    "the rotation and the position"),
         ("thin colored lines", "one held-out task, color = task (legend), solid / dashed "
                                "= first / second participant"),
         ("black line with circles / squares", "per participant the median over the tasks")],
        n_text(ctx, pids, f"{sm['segment'].nunique()} held-out tasks each"),
        statistic="per participant and task the RMS over the task's 200-Hz Qualisys frames; the "
                  "participant's median over the tasks, no pooling over participants",
        evidence="secondary"))


@register("val_mounting", "dashboard", "IMU mounting diagnostics", **G)
def val_mounting(req: ViewRequest) -> FigureSpec:
    """The IMU mounting between the hand-eye sessions (a remount shows as a large angle), the
    calibration residual with each mounting per participant and IMU, and the fold-to-fold
    spread of the in-take mounting (``ref_diagnostics``)."""
    fid, title = "val_mounting", "IMU mounting diagnostics"
    ctx = req.ctx
    ses = _table(ctx, "ref_diagnostics", "mounting_sessions")
    res = _table(ctx, "ref_diagnostics", "mounting_residual")
    spr = _table(ctx, "ref_diagnostics", "mounting_spread")
    shows = ("the IMU mounting diagnostics of the comparison takes: the rotation of each IMU "
             "mounting between the hand-eye sessions, the IMU-to-optical calibration residual "
             "with the mounting fitted in the take and with each hand-eye mounting, and the "
             "fold-to-fold spread of the in-take mounting")
    srcs = ["ref_diagnostics:mounting_sessions", "ref_diagnostics:mounting_residual",
            "ref_diagnostics:mounting_spread"]
    if ses.empty and res.empty:
        return _pending(fid, title, shows, "The mounting diagnostics are", srcs)
    imu = {"HP": "head IMU", "MP": "jaw IMU"}
    # hand-eye sessions by number, never by their calendar date (no wall-clock dates in print)
    names = sorted(set(ses.get("session_a", [])) | set(ses.get("session_b", []))
                   | {m for m in res.get("mounting", []) if m != "in_take"})
    sess = {n_: f"session {k + 1}" for k, n_ in enumerate(names)}
    panels = []
    if not ses.empty:
        s = ses.drop_duplicates(["body", "session_a", "session_b"])
        p = Panel("Mounting change between the sessions", Axis("Rotation", DEG), Axis(""), [],
                  legend="none", height=0.35)
        p.series.append(Series("barh", s["mounting_change_deg"].to_numpy(float),
                               np.arange(len(s), dtype=float), role="accent",
                               style={"bar_width": 0.5}))
        _yrows(p, [f"{imu.get(r.body, r.body)}, {sess.get(r.session_a, r.session_a)} → "
                   f"{sess.get(r.session_b, r.session_b)}" for r in s.itertuples()])
        panels.append(p)
    if not res.empty:
        pids = _pids(res)
        rows = [(r.pid, r.body, r.mounting, bool(r.own)) for r in res.itertuples()]
        p = Panel("Calibration residual per mounting", Axis("RMS residual", DEG, log=True),
                  Axis(""), [], legend="below")
        named: set[str] = set()
        for i, (pid, b, m, own) in enumerate(rows):
            r = res[(res.pid == pid) & (res.body == b) & (res.mounting == m)].iloc[0]
            kind = "in_take" if m == "in_take" else ("own" if own else "other")
            role, lab = {"in_take": ("system:jtsmm", "fitted in the take"),
                         "own": ("cat:1", "hand-eye, assigned session"),
                         "other": ("muted", "hand-eye, other session")}[kind]
            _interval(p, r.rms_median_deg, float(i), r.rms_min_deg, r.rms_max_deg, role,
                      _pm(pids.index(pid)), label=lab if kind not in named else None)
            named.add(kind)
        _yrows(p, [f"{_who(ctx, pid)} {imu.get(b, b)}, "
                   + ("in take" if m == "in_take" else f"{sess.get(m, m)}{' *' if own else ''}")
                   for pid, b, m, own in rows])
        panels.append(p)
    if not spr.empty:
        rows = list(spr.itertuples())
        p = Panel("In-take mounting across the folds", Axis("Angle", DEG), Axis(""), [],
                  legend="below", height=0.5)
        y = np.arange(len(rows), dtype=float)
        # colour = quantity, shape = participant (X1); larger markers
        # variant order orange, green, purple: blue is JTS-MM in (a) (2026-11-05)
        for col, role, lab in (("spread_median_deg", "cat:4", "median fold-to-fold"),
                               ("spread_max_deg", "cat:2", "largest fold-to-fold"),
                               ("intake_vs_handeye_median_deg", "cat:1",
                                "in take vs assigned hand-eye")):
            p.series.append(Series("scatter", np.array([getattr(r, col) for r in rows], float),
                                   y, role=role, style={"size": 1.6}, label=lab))
        _yrows(p, [f"{_who(ctx, r.pid)} {imu.get(r.body, r.body)}" for r in rows])
        shape_by_rows(p, [r.pid for r in rows], sorted({r.pid for r in rows}))
        panels.append(p)
    _tags(panels)
    n_rows = max(len(res), 4)
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=B.rows_height(n_rows, per=0.13, lo=1.0), sources=srcs)
    pids = sorted(set(_pids(res)) | set(_pids(ses)) | set(_pids(spr)))
    return attach(spec, CaptionSpec(
        shows + "; a wrong calibration date shows as a large residual",
        [("bars (a)", "rotation of a mounting between two hand-eye sessions (a remount is a "
                      "large angle)"),
         ("markers with whiskers (b)", "median and range over the leave-one-task-out folds, "
                                       "color = mounting (legend), circle / square = first / "
                                       "second participant, * = the session assigned to the "
                                       "take"),
         ("markers (c)", "pairwise angles between the folds' in-take mountings and the angle "
                         "to the assigned hand-eye mounting, color = quantity (legend), "
                         "circle / square = first / second participant"),
         ("sessions", "the hand-eye calibration sessions in time order")],
        n_text(ctx, pids, "five leave-one-task-out folds each"),
        statistic="per participant and IMU: median and range over the folds; no pooling",
        evidence=DESCRIPTIVE))


@register("val_marking_causes", "dashboard", "Point marking: what differs between the "
          "participants", **G)
def val_marking_causes(req: ViewRequest) -> FigureSpec:
    """The participant with the larger marking frame error against the one with the smaller:
    the frame errors and every candidate cause as a ratio (reference side: σ_Q of the teeth,
    the Qualisys head-cluster geometry; both systems: the CT orientation disagreement), and
    per hold the marking error against the tip error the CT orientation predicts."""
    from biomech import catalog

    fid, title = "val_marking_causes", "Point marking: what differs between the participants"
    ctx = req.ctx
    gap = _table(ctx, "ref_budget", "marking_gap")
    holds = _table(ctx, "ref_budget", "marking_holds")
    if gap.empty:
        return _empty(fid, title, "no marking-gap table")
    group = {"frame_max_deg": "outcome", "frame_mand_deg": "outcome", "rms_teeth_mm": "outcome",
             "sigma_ref_teeth_mm": "reference", "head_min_height_mm": "reference",
             "head_sv_ratio": "reference", "ct_rot_maxillary_deg": "both",
             "ct_tip_maxillary_mm": "both", "ct_rot_mandibular_deg": "both",
             "ct_tip_mandibular_mm": "both"}
    role = {"outcome": "system:jtsmm", "reference": "system:qualisys", "both": "accent"}
    # one shape (diamond), three colors: circle / square mean the participants in (b)
    mark = {"outcome": "d", "reference": "d", "both": "d"}
    glab = {"outcome": "marking result", "reference": "Qualisys side",
            "both": "calibration tool seen by both systems"}
    g = gap[gap["quantity"].isin(list(group)) & np.isfinite(gap["larger_over_smaller"]
                                                           .astype(float))]
    g = g.assign(_o=g["quantity"].map(list(group).index)).sort_values("_o")
    lo, hi = _who(ctx, gap["pid_smaller"].iloc[0]), _who(ctx, gap["pid_larger"].iloc[0])
    pa = Panel(f"{hi} relative to {lo}", Axis(f"{hi} / {lo}", "", log=True), Axis(""), [],
               legend="below")
    named: set[str] = set()
    labels = []
    for i, r in enumerate(g.itertuples()):
        e = catalog.get(str(r.quantity))
        labels.append(e.label.replace("CT ", "calibration-tool "))
        k = group[r.quantity]
        pa.series.append(Series("scatter", np.array([float(r.larger_over_smaller)]),
                                np.array([float(i)]), role=role[k],
                                style={"ms": mark[k], "size": 1.3},
                                label=glab[k] if k not in named else None))
        named.add(k)
    pa.series.append(Series("vline", x=np.array([1.0]), role="reference"))
    _yrows(pa, labels)
    panels = [pa]
    if not holds.empty and "predicted_tip_mm" in holds:
        pb = Panel("Marking difference per hold", Axis("Predicted tip difference", MM),
                   Axis("Marking difference", MM), [],
                   legend="below")
        hp = _pids(holds)
        for j, pid in enumerate(hp):
            h = holds[holds.pid == pid]
            sw = h["label_swapped"].astype(bool).to_numpy()
            # shape = participant (X1), swapped labels by colour
            for m, lab, role in ((~sw, _who(ctx, pid), "accent"),
                                 (sw, f"{_who(ctx, pid)}, labels swapped", "cat:3")):
                if m.any():
                    pb.series.append(Series("scatter", h["predicted_tip_mm"].to_numpy(float)[m],
                                            h["e_mm"].to_numpy(float)[m], role=role,
                                            style={"ms": _pm(j), "size": 1.2}, label=lab))
        v = np.r_[holds["predicted_tip_mm"].to_numpy(float), holds["e_mm"].to_numpy(float)]
        top = float(np.nanmax(v)) * 1.05 if np.isfinite(v).any() else 1.0
        pb.series.append(Series("line", np.array([0.0, top]), np.array([0.0, top]),
                                role="reference"))
        pb.x.lim = pb.y.lim = (0.0, top)
        panels.append(pb)
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=B.rows_height(len(g), per=0.2, lo=2.0),
                      sources=["ref_budget:marking_gap", "ref_budget:marking_holds"])
    return attach(spec, CaptionSpec(
        f"why the point marking differs between the participants: (a) {hi} relative to {lo}, "
        "the marking results and their candidate causes on the Qualisys side (tooth "
        "uncertainty, head-cluster geometry: smallest triangle height and singular-value "
        "ratio, small = nearly collinear) and seen by both systems (calibration-tool "
        "orientation disagreement and the tip difference it predicts); (b) per marking hold the "
        "point difference against the tip difference the calibration-tool orientation disagreement "
        "predicts over the calibration-tool lever",
        [("diamonds (a)", "ratio of the two participants' values, color = kind of quantity "
                          "(legend)"),
         ("vertical line (a)", "1, equal"),
         ("markers (b)", f"one hold, circle / square = {' / '.join(sorted({lo, hi}))}, "
                         "vermillion = swapped calibration-tool labels"),
         ("diagonal (b)", "difference equal to the prediction")],
        "n = 2 participants (" + ", ".join(sorted({lo, hi})) + ")",
        statistic="(a) ratio of two single values per quantity (median over the holds where "
                  "the quantity is per hold); (b) one value per hold, no summary",
        evidence="post hoc (descriptive)",
        notes=["The calibration-tool lever and the swapped-hold counts are in the marking-gap "
               "table."]))


@register("val_paths3d_error", "dashboard", "Incisor paths in 3-D colored by the difference", **G)
def val_paths3d_error(req: ViewRequest) -> FigureSpec:
    """Per participant the frame-aligned JTS-MM incisor path in 3-D with each frame colored
    by its distance to Qualisys (classes at the metrology and application bounds), the
    Qualisys path beside it, and the sagittal projection with the same classes."""
    from biomech.viz.views.validation_gallery import _movement_mask

    fid, title = "val_paths3d_error", "Incisor paths in 3-D colored by the difference"
    ctx = req.ctx
    met, app = TOL["metrology"][0], TOL["application"][0]
    # sky / orange / dark grey (2026-11-05): every pair OKLab dE >= 23.9 in normal vision and
    # under protan / deutan / tritan simulation (blue / orange / purple: 8.5 for orange /
    # purple under tritan; orange / vermillion: 13.1); no blue (= JTS-MM elsewhere), and the
    # class beyond the application bound is the darkest
    classes = ((0.0, met, "sky", f"≤ {met:g} mm (metrology bound)"),
               (met, app, "orange", f"{met:g}–{app:g} mm"),
               (app, np.inf, "ink2", f"> {app:g} mm (beyond the application bound)"))
    panels = []
    shown = []
    shares: list[str] = []
    data = []
    for pid in ctx.pids():
        a = _arrays(ctx, "ref_kinematics", pid)
        if "incisor_s_aligned_mm" not in a:
            continue
        m = _movement_mask(a)
        S, Q = np.asarray(a["incisor_s_aligned_mm"], float), np.asarray(a["incisor_q_mm"], float)
        err = np.asarray(a["err_incisor_aligned_mm"], float)
        m &= np.isfinite(err) & np.isfinite(S).all(1) & np.isfinite(Q).all(1)
        if not m.any():
            continue
        step = max(1, int(m.sum() // 1000))
        idx = np.flatnonzero(m)[::step]
        c = np.nanmean(Q[idx], axis=0)
        data.append((pid, S[idx] - c, Q[idx] - c, err[idx]))
    for j, (pid, S, Q, err) in enumerate(data):
        first = j == 0
        p3 = Panel(_who(ctx, pid), Axis(XYZ[0], MM), Axis(XYZ[1], MM), [],
                   projection="3d", z=Axis(XYZ[2], MM), view=(-60.0, 20.0),
                   legend="below" if first else "none")
        p3.series.append(Series("line3d", Q[:, 0], Q[:, 1], z=Q[:, 2], role="system:qualisys",
                                style={"width": 0.3, "alpha": 0.35},
                                label="Qualisys path" if first else None))
        ps = Panel(f"{_who(ctx, pid)}, sagittal", Axis(XYZ[0], MM), Axis(XYZ[2], MM), [],
                   equal=True, legend="none")
        ps.series.append(Series("line", Q[:, 0], Q[:, 2], role="system:qualisys",
                                style={"width": 0.3, "alpha": 0.35}))
        pct = []
        for lo, hi, col, lab in classes:
            k = (err > lo) & (err <= hi) if lo > 0 else err <= hi
            pct.append(f"{100.0 * k.mean():.0f}")
            if not k.any():
                continue
            p3.series.append(Series("scatter3d", S[k, 0], S[k, 1], z=S[k, 2], role="primary",
                                    style={"color": col, "size": 0.6},
                                    label=f"JTS-MM, {lab}" if first else None))
            ps.series.append(Series("scatter", S[k, 0], S[k, 2], role="primary",
                                    style={"color": col, "size": 0.6}))
        # the class shares in the caption, not as text on the data
        shares.append(f"{_who(ctx, pid)} " + " / ".join(pct) + " %")
        panels += [p3, ps]
        shown.append(pid)
    if not panels:
        return _empty(fid, title, "no aligned arrays")
    _tags(panels)
    spec = FigureSpec(fid, title, panels, rows=len(shown), cols=2, width="text", height_in=2.6,
                      sources=["ref_kinematics:arrays"])
    return attach(spec, CaptionSpec(
        "where in the workspace JTS-MM and Qualisys disagree: per participant the lower-incisor "
        "path of JTS-MM after the constant frame alignment (leave-one-task-out) in the movement "
        "tasks, each frame colored by its distance to Qualisys in classes at the "
        "pre-specified metrology and application bounds, (left) in 3-D and (right) in the "
        "sagittal plane; the Qualisys path beside it; coordinates centered on the Qualisys path",
        [("dots", "JTS-MM frames, color = distance class (legend: sky blue within the metrology "
                  "bound, orange within the application bound, dark grey beyond it)"),
         ("faint dashed black line", "Qualisys path")],
        n_text(ctx, shown, "every movement-task frame that passes the Qualisys quality gate, thinned "
                           "evenly for drawing"),
        statistic="raw per-frame distances, no summary; share of the drawn frames per class "
                  "(within metrology / within application / beyond): " + "; ".join(shares),
        evidence=DESCRIPTIVE))


# == J7: related work of jaw trackers ==============================================================
RELATED_WORK = Path(__file__).resolve().parents[5] / "experiments" / "jtsmm_validation" \
    / "related_work.yaml"
PRINCIPLE_ORDER = ("optical", "optoelectronic", "electromagnetic", "magnetic", "ultrasonic",
                   "inertial", "mixed")
ACCURACY_KIND = {"trueness": "trueness", "precision": "precision", "rmse": "RMS error",
                 "loa": "limits of agreement", "agreement": "agreement",
                 "event_timing": "event timing", "none": "not reported"}


def related_work_rows(path: Any = None) -> list[dict[str, Any]]:
    """Rows of ``related_work.yaml`` (J7), grouped by principle; empty without it."""
    import yaml

    p = Path(path) if path is not None else RELATED_WORK
    if not p.is_file():
        return []
    rows = [r for r in (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("rows", [])
            if isinstance(r, dict)]
    rank = {k: i for i, k in enumerate(PRINCIPLE_ORDER)}
    return sorted(rows, key=lambda r: (rank.get(str(r.get("principle")), 99), str(r["system"])))


def _this_work_row(ctx: Any) -> dict[str, Any] | None:
    """JTS-MM's own row from the results it reads (never typed by hand): bench accuracy
    (RMS to the fitted ladder, range over the conditions) and the in-vivo agreement with
    Qualisys (frames aligned, all movement tasks; per participant)."""
    t = _table(ctx, "ref_trueness", "summary")
    a = _table(ctx, "ref_kinematics", "segments_aligned")
    parts = []
    if not t.empty and "accuracy_rms_mm" in t:
        b = t[t.system == "jtsmm"].accuracy_rms_mm.to_numpy(float)
        if np.isfinite(b).any():
            parts.append(f"bench RMS {np.nanmin(b):.2f}–{np.nanmax(b):.2f} mm (ladder)")
    if not a.empty and "segment" in a:
        p = a[a.segment == "movement_pooled"].sort_values("pid")
        if len(p):
            inc = " / ".join(f"{v:.1f}" for v in p.incisor_rms_mm)
            rot = " / ".join(f"{v:.1f}" for v in p.rot_rms_deg)
            who = " / ".join(_who(ctx, q) for q in p.pid)
            parts.append(f"in vivo incisor RMS {inc} mm, rotation {rot}° ({who}, frames "
                         "aligned)")
    if not parts:
        return None
    # achieved camera-frame rate of the comparison takes (median frame interval, as in the
    # methods table), the same for every participant or given as a range
    rates = []
    for pid in _pids(a):
        tt = np.asarray(_arrays(ctx, "ref_kinematics", pid).get("t_s", np.zeros(0)), float)
        dt = np.diff(tt[np.isfinite(tt)])
        if dt.size:
            rates.append(1.0 / float(np.median(dt)))
    rate = ("" if not rates else f"{min(rates):.0f}" if round(min(rates)) == round(max(rates))
            else f"{min(rates):.0f}–{max(rates):.0f}")
    return {"system": "JTS-MM (this work)", "principle": "optical + inertial", "dof": "6",
            "rate": rate, "accuracy": "; ".join(parts) + " (bench: RMS error; in vivo: RMS difference)",
            "reference": "Qualisys (bench ladder and in vivo)", "worn": "no", "vivo": "yes",
            "source": "this work"}


def _rw_text(text: Any) -> str:
    """Curated literature text for print: the reader glyphs (:func:`tidy`), ° instead of
    "deg", en dashes in number ranges ("20-40 mm" -> "20–40 mm"); "None" -> not reported."""
    t = "" if text is None or str(text).strip() in ("", "None") else tidy(str(text))
    t = re.sub(r"(?<=\d)\s*deg\b", "°", t)
    return re.sub(r"(?<=\d)-(?=\d)", "\u2013", t)


RW_OPTICAL = ("optical", "optoelectronic")      # table 1; every other principle in table 2


def _related_work_table(ctx: Any, fid: str, optical: bool) -> TableSpec:
    """One of the two related-work tables (one table ran off the page): the optical
    trackers with this work's row first, or every other principle."""
    from biomech.viz.spec import Column

    rows: list[dict[str, Any]] = []
    own = _this_work_row(ctx) if optical else None
    if own:
        rows.append({**own, "specs": f"{own['dof']} DoF" + (f", {own['rate']} Hz"
                                                             if own.get("rate") else ""),
                     "where": f"{own['worn']} / {own['vivo']}"})
    yn = {True: "yes", False: "no"}
    for r in related_work_rows():
        if (str(r.get("principle")) in RW_OPTICAL) != optical:
            continue
        acc = _rw_text(r.get("accuracy"))
        kind = ACCURACY_KIND.get(str(r.get("accuracy_kind")), "")
        specs = ", ".join(x for x in (f"{r['dof']} DoF" if r.get("dof") is not None else "",
                                      f"{r['rate_hz']:g} Hz" if r.get("rate_hz") is not None
                                      else "") if x)
        rows.append({"system": _rw_text(r.get("system", "")),
                     "principle": str(r.get("principle", "")), "specs": specs,
                     "accuracy": (f"{acc} ({kind})" if acc and kind and kind != "not reported"
                                  else acc or "not reported"),
                     "reference": _rw_text(r.get("reference_system", "")),
                     "where": f"{yn.get(r.get('wearable'), '–')} / {yn.get(r.get('in_vivo'), '–')}",
                     "source": cite_label(str(r.get("key", ""))), "_key": str(r.get("key", ""))})
    if not rows:
        rows = [{"system": "no related-work table found"}]
    cols = [Column("system", "System", align="X"), Column("principle", "Principle", align="l"),
            Column("specs", "DoF, rate", align="l"),
            Column("accuracy", "Reported accuracy (kind)", align="X"),
            Column("reference", "Compared with", align="X"),
            Column("where", "Worn / in vivo", align="l"), Column("source", "Source", align="l")]
    t = TableSpec(fid, "", cols, rows, width="text",
                  notes="DoF: degrees of freedom tracked, rate: sampling rate; empty where the "
                        "source does not state it. Accuracy figures differ in kind (static "
                        "trueness on a phantom, precision, agreement in vivo) and are not "
                        "comparable across rows. Sources as first author and year (the BibTeX "
                        "keys are in the table's JSON for the manuscript's citations).",
                  sources=["literature:related_work"])
    n_pub = len([r for r in rows if r.get("principle") and r.get("source") != "this work"])
    what = ("optical and optoelectronic jaw-motion trackers in the literature, with this work "
            "in the first row (its values from this study)" if optical else
            "jaw-motion trackers in the literature that are not optical: electromagnetic, "
            "magnetic, ultrasonic, inertial and mixed")
    return attach(t, CaptionSpec(
        f"{what}: the accuracy each source reports, what it was compared with and whether the "
        "system is worn and was tested in people",
        n=f"n = {n_pub} published systems" + (", plus this work" if own else ""),
        statistic="as reported by each source (kind in brackets)", evidence="descriptive"))


@register("val_table_related_work", "dashboard", "Jaw-motion trackers in the literature", **G)
def val_table_related_work(req: ViewRequest) -> TableSpec:
    """Published optical jaw-motion trackers and this work: principle, degrees of freedom and
    rate, the reported accuracy with its kind, the reference and whether worn and in vivo --
    never ranked by the number (the kinds differ). JTS-MM's own row comes from the results.
    The other principles: ``val_table_related_work_other``."""
    return _related_work_table(req.ctx, "val_table_related_work", optical=True)


@register("val_table_related_work_other", "dashboard",
          "Jaw-motion trackers in the literature (other principles)", **G)
def val_table_related_work_other(req: ViewRequest) -> TableSpec:
    """Published jaw-motion trackers that are not optical (electromagnetic, magnetic,
    ultrasonic, inertial, mixed), as ``val_table_related_work``."""
    return _related_work_table(req.ctx, "val_table_related_work_other", optical=False)
