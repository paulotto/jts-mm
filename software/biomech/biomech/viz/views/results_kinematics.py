# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result views of the kinematics, uncertainty and variability stages (Results page).

Every stored kinematic outcome gets a view here: per-participant values as one dot per
participant (pseudonym rows), per-cycle arrays as grey cycles under the participant
median, cohort arrays as participant medians under the cohort median and its IQR,
Monte-Carlo draws as distributions and intervals, long tables as catalog tables. Views
only read stored results; nothing is recomputed beyond quantiles of stored draws.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from biomech import catalog
from biomech.viz.spec import Axis, Colorbar, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.results import COHORT_UNIT

MODELS = V.labels("hinge_model")
SIDE = {"left": "L", "right": "R"}
PHASES = V.TASKS


# -- helpers -------------------------------------------------------------------------------------
def _pid(req: ViewRequest) -> str | None:
    pids = req.ctx.pids()
    return req.pid if req.pid in pids else (pids[0] if pids else None)


def who(ctx: Any, pid: str) -> str:
    """Pseudonym of an included participant; anything else is never named."""
    return f"P{ctx.pseudonym(pid)}" if pid in ctx.pids() else "not included"


def _f(a: dict, key: str) -> np.ndarray | None:
    v = a.get(key)
    return None if v is None else np.asarray(v, float)


def numeric_columns(df: pd.DataFrame, skip: tuple[str, ...] = ("pid", "unit")) -> list[str]:
    return [c for c in df.columns if c not in skip and (
        pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))]


def _nanmedian(S: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.atleast_2d(S), axis=axis)


def _empty(p: Panel, msg: str = "no data") -> Panel:
    p.note = msg
    return p


def _desc(column: str) -> str:
    d = catalog.get(column).desc
    return f" {d}." if d else ""


def _label(column: str) -> str:
    """Catalog label, else the S1 metric label of stats_kinematics, else the name."""
    if V.known(column):
        return V.label(column)
    from biomech.stats.cohort_kinematics import METRICS

    return METRICS[column][0] if column in METRICS else column.replace("_", " ")


SUFFIX_UNIT = {"_mm": V.MM, "_deg": V.DEG, "_dps": V.DPS, "_hz": V.HZ, "_s": V.SEC,
               "_ms": V.MSEC, "_pct": V.PCT, "_uv": V.UV, "_mm_s": V.MMS}


def axis_of(column: str) -> Axis:
    """Catalog axis; for a column without a catalog entry the S1 label and the unit its
    name suffix states (``_mm``, ``_deg``, ...)."""
    if V.known(column):
        return V.axis(column)
    base = re.sub(r"_(raw|smooth)(_[xyz])?$|_[xyz]$", "", column)
    suf = next((k for k in sorted(SUFFIX_UNIT, key=len, reverse=True) if base.endswith(k)), "")
    lab = _label(column)
    if lab == column.replace("_", " ") and suf:          # no label anywhere: drop the unit
        lab = column.replace(suf, "", 1).replace("_", " ") if column.endswith(suf) else \
            column.replace(suf + "_", "_", 1).replace("_", " ")
    return Axis(lab, SUFFIX_UNIT.get(suf, ""))


def per_participant(ctx: Any, df: pd.DataFrame, column: str, title: str, *,
                    xaxis: Axis | None = None, ref: float | None = None,
                    ref_label: str | None = None, ci: bool = True) -> Panel:
    """One dot per included participant (pseudonym rows); the cohort median as a line and,
    with ``ci``, its 95 % BCa bootstrap CI as a faint band (:func:`blocks.median_ci`)."""
    pids = ctx.pids()
    s = (df.drop_duplicates("pid").set_index("pid")[column].reindex(pids)
         if not df.empty and column in df else pd.Series(np.nan, index=pids))
    v = pd.to_numeric(s.astype(object).where(s.notna(), np.nan), errors="coerce").to_numpy(float)
    p = Panel(title, xaxis or axis_of(column), Axis(""), [], legend="none")
    y = np.arange(len(pids), dtype=float)
    if np.isfinite(v).any():
        if ci:
            lo, hi = B.median_ci(v, B.seed_of(ctx))
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("hspan", x=np.array([lo]), y2=np.array([hi]),
                                       role="faint"))
        p.series.append(Series("vline", x=np.array([float(np.nanmedian(v))]),
                               role="context", label="median"))
        p.series.append(Series("scatter", v, y, role="accent"))
    else:
        _empty(p)
    if ref is not None and np.isfinite(ref):
        p.series.append(Series("vline", x=np.array([float(ref)]), role="reference",
                               label=ref_label))
    p.y.ticks, p.y.ticklabels = list(y), B.pseudo_rows(ctx, pids)
    p.y.reverse = True
    return p


PP_CAPTION = "(dots; line: cohort median, band: its 95 % BCa bootstrap CI)"


def _h(n: int, per: float = 0.17, lo: float = 1.4) -> float:
    return max(lo, per * n)


def _rel_s(t_us: np.ndarray) -> np.ndarray:
    t = np.asarray(t_us, float)
    return (t - t[0]) * 1e-6 if t.size else t


# == kinematics ==================================================================================
def _summary_metrics(ctx: Any) -> list[str]:
    from biomech.kinematics.analysis import SUMMARY_METRICS

    s = ctx.table("kinematics", "summary")
    have = set(s.metric) if "metric" in s else set()
    return [m for m in SUMMARY_METRICS if m in have]


@register("kin_summary", "dashboard", "Cycle metrics per task", domain="kinematics",
          choices={"metric": _summary_metrics}, metric="opening_mm")
def kin_summary(req: ViewRequest) -> FigureSpec:
    """Participant medians and cycle-to-cycle CV of one cycle metric, per task and side."""
    ctx, metric = req.ctx, req.options.get("metric") or "opening_mm"
    s = ctx.table("kinematics", "summary")
    s = s[s.metric == metric].copy() if not s.empty else s
    if not s.empty:
        s["row"] = [t + (f"|{d}" if d else "") for t, d in zip(s.task, s.side.fillna(""))]
        s["cv_pct"] = 100 * s["cv"]
    order = [t + (f"|{d}" if d else "") for t in V.TASKS for d in ("", "left", "right")]
    labels = {r: V.task_label(r.split("|")[0]) + (f" {SIDE[r.split('|')[1]]}" if "|" in r else "")
              for r in order}
    seed = B.seed_of(ctx)
    pa = B.strip_by_ci(s, "median", "row", "Participant median", order=order, labels=labels,
                       xaxis=axis_of(metric), seed=seed)
    pb = B.strip_by_ci(s, "cv_pct", "row", "Cycle-to-cycle variation", order=order,
                       labels=labels, xaxis=Axis("Coefficient of variation", V.PCT), seed=seed)
    return FigureSpec("kin_summary", "Cycle metrics per task", [pa, pb], rows=1, cols=2,
                      width="text", height_in=_h(len(pa.y.ticks or []), 0.22),
                      caption=f"{_label(metric)} per task and side: median over each "
                              "participant's good cycles (dots: participants, diamond: cohort "
                              f"median, {B.MEDIAN_CI_CAPTION}) and its cycle-to-cycle "
                              "coefficient of variation (only "
                              f"for strictly positive metrics).{_desc(metric)}",
                      sources=["kinematics:summary"])


def _anatomy_columns(ctx: Any) -> list[str]:
    return numeric_columns(ctx.table("kinematics", "anatomy"))


@register("kin_anatomy", "dashboard", "Anatomy and noise floor", domain="kinematics",
          choices={"column": _anatomy_columns}, column="occlusal_pitch_deg")
def kin_anatomy(req: ViewRequest) -> FigureSpec:
    """One anatomy, landmark-check or rest noise-floor quantity per participant."""
    ctx = req.ctx
    an = ctx.table("kinematics", "anatomy")
    col = req.options.get("column") or "occlusal_pitch_deg"
    if col not in an and numeric_columns(an):
        col = numeric_columns(an)[0]
    p = per_participant(ctx, an, col, _label(col))
    return FigureSpec("kin_anatomy", "Anatomy and noise floor", [p], width="column",
                      height_in=_h(len(ctx.pids())),
                      caption=f"{_label(col)} per participant {PP_CAPTION}."
                              + _desc(col), sources=["kinematics:anatomy"])


@register("kin_anatomy_table", "dashboard", "Anatomy table", domain="kinematics")
def kin_anatomy_table(req: ViewRequest) -> TableSpec:
    """Every anatomy / landmark / noise-floor column, one column per participant."""
    ctx = req.ctx
    an = ctx.table("kinematics", "anatomy")
    pids = [p for p in ctx.pids() if not an.empty and p in set(an.pid)]
    a = an.drop_duplicates("pid").set_index("pid").reindex(pids) if pids else an
    rows = []
    for c in [c for c in an.columns if c not in ("pid", "unit")]:
        e = catalog.get(c)
        r = {"quantity": _label(c), "unit": V.unit(c) if V.known(c) else ""}
        for i, p in enumerate(pids):
            v = a.at[p, c]
            r[f"p{i}"] = (V.cell(c, v) if isinstance(v, str) else "" if v is None
                          else "yes" if isinstance(v, (bool, np.bool_)) and v
                          else "no" if isinstance(v, (bool, np.bool_)) else e.fmt(float(v)))
        rows.append(r)
    cols = [Column("quantity", "Quantity", align="l"), Column("unit", "Unit", align="l")] + [
        Column(f"p{i}", who(ctx, p), align="r") for i, p in enumerate(pids)]
    return TableSpec("kin_anatomy_table", "Anatomy, landmark checks and rest noise floor per "
                     "participant (maxillary frame; smooth: after the pose low-pass, raw: "
                     "before)", cols, rows, sources=["kinematics:anatomy"], width="text")


@register("kin_opening_model", "dashboard", "Opening model", domain="kinematics")
def kin_opening_model(req: ViewRequest) -> FigureSpec:
    """Subject coupling model opening = a1 θ + a2 θ² (used by the IMU stage) and its fit."""
    ctx = req.ctx
    om = ctx.table("kinematics", "opening_model")
    th = np.linspace(0.0, 40.0, 81)
    pa = Panel("Model per participant", Axis("Opening angle", V.DEG), Axis("Opening", V.MM),
               [], legend="none")
    if not om.empty:
        curves = np.array([a1 * th + a2 * th ** 2 for a1, a2 in
                           zip(om.a1_mm_per_deg, om.a2_mm_per_deg2)])
        for c in curves:
            pa.series.append(Series("line", th, c, role="context"))
        pa.series.append(Series("line", th, _nanmedian(curves), role="accent"))
    else:
        _empty(pa)
    pb = per_participant(ctx, om, "rmse_mm", "Fit residual", xaxis=Axis("RMSE", V.MM))
    return FigureSpec("kin_opening_model", "Opening model", [pa, pb], rows=1, cols=2,
                      width="text", height_in=B.rows_height(len(ctx.pids()), per=0.15,
                                                            extra=0.2, lo=1.6),
                      caption="Incisal opening as a quadratic function of the opening angle, "
                              "fitted through the origin on each participant's open-close "
                              "openings (grey: participants, accent: median curve), and the "
                              f"RMS residual of that fit {PP_CAPTION}. The IMU stage converts "
                              "IMU angles "
                              "to opening with it.", sources=["kinematics:opening_model"])


@register("kin_coupling", "dashboard", "Coupling per cycle", params=("pid",),
          domain="kinematics")
def kin_coupling(req: ViewRequest) -> FigureSpec:
    """One participant's openings on the angle grid: incisal opening and condylar travel."""
    ctx, pid = req.ctx, _pid(req)
    a = ctx.arrays("kinematics", pid) if pid else {}
    g = _f(a, "coupling__grid_deg")
    xa = Axis("Opening angle", V.DEG)
    specs = [("opening_mm", "Incisal opening", "Opening"),
             ("tmj_mid_ant_mm", "Condylar midpoint, anterior", "Anterior translation"),
             ("tmj_mid_disp_mm", "Condylar midpoint, 3-D", "Displacement")]
    panels = []
    for key, title, ylab in specs:
        c = _f(a, f"coupling__{key}")
        p = Panel(title, xa, Axis(ylab, V.MM), [], legend="none", sharex="th")
        if g is not None and c is not None and c.size:
            B.add_curves_ci(p, g, c, centre="median", show_subjects=True, seed=B.seed_of(ctx))
        else:
            _empty(p)
        panels.append(p)
    p = Panel("Condyles, 3-D", xa, Axis("Displacement", V.MM), [], legend="auto", sharex="th")
    for side in ("left", "right"):
        c = _f(a, f"coupling__tmj_{side}_disp_mm")
        if g is not None and c is not None and c.size:
            for row in np.atleast_2d(c):
                p.series.append(Series("line", g, row, role="faint"))
            p.series.append(Series("line", g, _nanmedian(c), role=f"side:{side}",
                                   label=f"{side} condyle"))
    if not p.series:
        _empty(p)
    panels.append(p)
    n = int(np.atleast_2d(_f(a, "coupling__opening_mm")).shape[0]) \
        if a.get("coupling__opening_mm") is not None else 0
    return FigureSpec("kin_coupling", "Coupling per cycle", panels, rows=2, cols=2,
                      width="text", height_in=1.6,
                      caption=f"{who(ctx, pid) if pid else ''}: {n} openings of open-close on "
                              "the opening-angle grid (grey: openings, accent: median; "
                              + B.curve_ci_caption("median", "openings") + "); "
                              "condylar points are skin-palpated, displacement from the "
                              "start of each opening.", sources=["kinematics:coupling"])


@register("kin_coupling_cohort", "dashboard", "Coupling, cohort band", domain="kinematics")
def kin_coupling_cohort(req: ViewRequest) -> FigureSpec:
    """Cohort coupling curves: participant medians, cohort median and IQR (C1)."""
    arr = req.ctx.arrays("stats_kinematics", COHORT_UNIT)
    g = _f(arr, "grid_deg")
    panels = []
    n_sub = 0
    for key, title, ylab in (("opening_mm", "Incisal opening", "Opening"),
                             ("tmj_mid_ant_mm", "Condylar midpoint, anterior",
                              "Anterior translation"),
                             ("tmj_mid_disp_mm", "Condylar midpoint, 3-D", "Displacement")):
        p = Panel(title, Axis("Opening angle", V.DEG), Axis(ylab, V.MM), [], legend="none")
        S, med = _f(arr, f"{key}__subjects"), _f(arr, f"{key}__median")
        q1, q3 = _f(arr, f"{key}__q1"), _f(arr, f"{key}__q3")
        if g is None or S is None or med is None:
            panels.append(_empty(p))
            continue
        for row in np.atleast_2d(S):
            p.series.append(Series("line", g, row, role="context"))
        if q1 is not None and q3 is not None:
            p.series.append(Series("band", g, q1, y2=q3, role="band",
                                   style={"alpha": B.SD_BAND_ALPHA}))
        B.add_ci_band(p, g, S, centre="median", seed=B.seed_of(req.ctx),
                      where=np.isfinite(med))
        p.series.append(Series("line", g, med, role="accent"))
        n_sub = max(n_sub, int(np.atleast_2d(S).shape[0]))
        panels.append(p)
    return FigureSpec("kin_coupling_cohort", "Coupling, cohort band", panels, rows=1, cols=3,
                      width="text", height_in=1.6,
                      caption="Incisal opening and condylar-midpoint travel over the opening "
                              f"angle: participant medians over openings (grey, {n_sub} "
                              "participants), cohort median (accent; "
                              + B.curve_ci_caption("median") + "), shown where at least half "
                              "of the participants contribute. Condyles skin-palpated.",
                      sources=["stats_kinematics:coupling"])


@register("kin_hinge_frames", "dashboard", "Hinge fit per frame", params=("pid",),
          domain="kinematics")
def kin_hinge_frames(req: ViewRequest) -> FigureSpec:
    """Best-fit hinge residual and the translation of the hinge + translation model."""
    ctx, pid = req.ctx, _pid(req)
    a = ctx.arrays("kinematics", pid) if pid else {}
    th, res, tau = (_f(a, k) for k in ("hinge__theta_deg", "hinge__residual_incisor",
                                       "hinge__tau_mm"))
    xa = Axis("Opening angle (best-fit hinge)", V.DEG)
    pa = Panel("Best-fit hinge: incisal residual", xa, Axis("Residual", V.MM), [],
               legend="none", sharex="th")
    pb = Panel("Hinge + translation: translation", xa, Axis("Translation", V.MM), [],
               legend="none", sharex="th")
    if th is not None and res is not None:
        pa.series.append(Series("scatter", th, res, role="primary", style={"size": 0.5}))
    else:
        _empty(pa)
    if th is not None and tau is not None and tau.size == th.size:
        pb.series.append(Series("scatter", th, tau, role="primary", style={"size": 0.5}))
    else:
        _empty(pb)
    h = ctx.table("kinematics", "hinge")
    h = h[h.pid == pid] if not h.empty else h
    if not h.empty:
        rms = {r.model: r.rms_incisor_mm for r in h.itertuples()}
        pa.note = ", ".join(f"{m.replace('_', ' ')} {rms[m]:.2f} mm" for m in MODELS if m in rms)
        tr = h[h.model == "rot_trans"].translation_range_mm
        if not tr.empty and np.isfinite(tr.iloc[0]):
            pb.note = f"range {tr.iloc[0]:.1f} mm"
    return FigureSpec("kin_hinge_frames", "Hinge fit per frame", [pa, pb], rows=1, cols=2,
                      width="text", height_in=1.6,
                      caption=f"{who(ctx, pid) if pid else ''}: incisal misalignment of the "
                              "best-fit hinge per fitted frame (note: RMS of each model) and "
                              "the translation along the axis-normal direction of the hinge + "
                              "translation model (2-DoF), over the opening angle.",
                      sources=["kinematics:hinge"])


@register("kin_theta_raw", "dashboard", "Opening angle: raw vs smoothed",
          params=("pid", "phase"), phase="open_close", phases=PHASES, domain="kinematics")
def kin_theta_raw(req: ViewRequest) -> FigureSpec:
    """Opening angle before and after the pose low-pass and the angular velocity."""
    ctx, pid, task = req.ctx, _pid(req), req.phase or "open_close"
    a = ctx.arrays("kinematics", pid) if pid else {}
    t = _f(a, f"{task}__t_us")
    raw, sm, om = (_f(a, f"{task}__{k}") for k in ("theta_raw_deg", "theta_deg", "omega"))
    xa = Axis("Time in phase", V.SEC)
    pa = Panel("Opening angle", xa, Axis("Angle", V.DEG), [], sharex="t")
    pb = Panel("Raw minus smoothed", xa, Axis("Difference", V.DEG), [], legend="none",
               sharex="t")
    pc = Panel("Angular velocity (maxillary frame)", xa, Axis("Angular velocity", V.DPS), [],
               sharex="t")
    if t is not None and raw is not None and sm is not None:
        ts = _rel_s(t)
        pa.series += [Series("line", ts, raw, role="context", label="raw"),
                      Series("line", ts, sm, role="accent", label="low-pass")]
        pb.series.append(Series("line", ts, raw - sm, role="primary"))
        pb.note = f"RMS {np.sqrt(np.nanmean((raw - sm) ** 2)):.3f}°"
    else:
        _empty(pa)
        _empty(pb)
    if t is not None and om is not None and om.ndim == 2:
        for i, ax in enumerate("xyz"):
            pc.series.append(Series("line", _rel_s(t), om[:, i], role=f"axis:{ax}",
                                    label=f"about {ax}"))
    else:
        _empty(pc)
    return FigureSpec("kin_theta_raw", "Opening angle: raw vs smoothed", [pa, pb, pc], rows=3,
                      cols=1, width="text", height_in=1.25,
                      caption=f"{who(ctx, pid) if pid else ''}, {V.task_label(task)}: opening "
                              "angle from the unsmoothed poses and after the pose low-pass "
                              "(the bandwidth noise reference), their difference, and the "
                              "spatial angular velocity of the mandible (x anterior, y left, "
                              "z up).", sources=["kinematics:theta", "kinematics:omega"])


@register("kin_fha_cohort", "dashboard", "Helical-axis migration", domain="kinematics")
def kin_fha_cohort(req: ViewRequest) -> FigureSpec:
    """Per-participant migration of the finite helical axis and the cohort median (C1)."""
    ctx = req.ctx
    fs = ctx.table("stats_kinematics", "fha_subject")
    sm = ctx.table("stats_kinematics", "fha_summary")
    items = [("pierce_x_range", "anterior range of piercing point"),
             ("pierce_z_range", "vertical range of piercing point"),
             ("dist_tmj", "distance to condylar midpoint")]
    vals = [fs[k].to_numpy(float) if k in fs else np.empty(0) for k, _ in items]
    ci = []
    for k, _ in items:
        r = sm[sm.metric == k] if not sm.empty else sm
        ci.append((float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])) if not r.empty
                  else (np.nan, np.nan))
    p = B.strip([lab for _, lab in items], vals, "Finite helical axis per opening",
                Axis("Distance", V.MM), intervals=ci)
    return FigureSpec("kin_fha_cohort", "Helical-axis migration", [p], width="column",
                      height_in=1.4,
                      caption="Finite helical axis in steps of the opening angle: range of the "
                              "sagittal piercing point within an opening and its mean distance "
                              "to the palpated condylar midpoint; per participant the median "
                              "over openings (dots), cohort median (diamond) with 95 % "
                              "bootstrap CI (line).",
                      sources=["stats_kinematics:fha_subject", "stats_kinematics:fha_summary"])


def _cs_label(metric: str) -> str:
    if metric.startswith("hinge_rms_incisor_"):
        m = metric.removeprefix("hinge_rms_incisor_")
        return "incisal RMS, " + (MODELS.get(m) or V.term("text_term", m))
    return {"translation_range_mm": "translation range (hinge + transl.)",
            "rms_condyles_mm": "condylar RMS (hinge + transl.)"}.get(metric, _label(metric))


@register("kin_condyle_sensitivity", "dashboard", "Condyle plausibility sensitivity",
          domain="kinematics")
def kin_condyle_sensitivity(req: ViewRequest) -> FigureSpec:
    """Condyle-based outcomes for all participants vs only plausible condyles (A4)."""
    from biomech.stats.cohort_kinematics import GEOMETRY_METRICS

    cs = req.ctx.table("stats_kinematics", "condyle_sensitivity")
    # The jaw geometry rows (offsets, ratio, angles; D21) are not positive millimetres for
    # this log axis: table_geometry shows them.
    cs = cs[~cs.metric.isin(GEOMETRY_METRICS)] if not cs.empty else cs
    groups = [("all", "cat:0", "all participants"),
              ("plausible_condyles", "cat:1", "plausible condyles only"),
              ("condyles_unknown", "cat:2", "condyles unknown")]
    metrics = list(dict.fromkeys(cs.metric)) if not cs.empty else []
    p = Panel("Median with 95 % CI", Axis("Value", V.MM, log=True), Axis(""), [])
    for gi, (g, role, lab) in enumerate(groups):
        d = cs[cs.group == g].set_index("metric").reindex(metrics) if metrics else cs
        if d.empty or not d["median"].notna().any():
            continue
        y = np.arange(len(metrics)) + (gi - 0.5) * 0.25
        for yi, lo, hi in zip(y, d.ci_lo, d.ci_hi):
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("line", np.array([lo, hi]), np.array([yi, yi]),
                                       role=role))
        p.series.append(Series("scatter", d["median"].to_numpy(float), y, role=role,
                               label=f"{lab} (n = {int(d.n.max())})"))
    if not metrics:
        _empty(p)
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(metrics))], [
        _cs_label(m) for m in metrics]
    p.y.reverse = True
    return FigureSpec("kin_condyle_sensitivity", "Condyle plausibility sensitivity", [p],
                      width="text", height_in=_h(len(metrics), 0.24),
                      caption="Condyle-based outcomes over all participants and over those "
                              "whose skin-palpated condyles pass the plausibility check "
                              "(flagged, not excluded): median and 95 % bootstrap CI.",
                      sources=["stats_kinematics:condyle_sensitivity"])


@register("kin_norms", "dashboard", "Comparison with literature norms", domain="kinematics")
def kin_norms(req: ViewRequest) -> FigureSpec:
    """Participants relative to published reference means (percent difference)."""
    ns = req.ctx.table("stats_kinematics", "norms_subjects")
    su = req.ctx.table("stats_kinematics", "norms_summary")
    metrics = list(dict.fromkeys(ns.metric)) if not ns.empty else []
    vals, iv = [], []
    for m in metrics:
        g = ns[ns.metric == m]
        vals.append(100 * g["diff"].to_numpy(float) / g.ref_mean.to_numpy(float))
        r = su[su.metric == m] if not su.empty else su
        if not r.empty and np.isfinite(r.sd_diff.iloc[0]):
            c, s, ref = r.mean_diff.iloc[0], r.sd_diff.iloc[0], r.ref_mean.iloc[0]
            iv.append((100 * (c - s) / ref, 100 * (c + s) / ref))
        else:
            iv.append((np.nan, np.nan))
    p = B.strip([_label(m) for m in metrics], vals, "Difference to the reference mean",
                Axis("Difference", V.PCT), ref=0.0, ref_label="reference", intervals=iv)
    src = "; ".join(dict.fromkeys(su.source)) if not su.empty and "source" in su else ""
    return FigureSpec("kin_norms", "Comparison with literature norms", [p], width="column",
                      height_in=_h(len(metrics), 0.25),
                      caption="Participant values relative to the published healthy reference "
                              "mean (dots: participants, diamond: median, line: mean ± SD of "
                              f"the differences). References: {src or 'none matched'}.",
                      sources=["stats_kinematics:norms_subjects",
                               "stats_kinematics:norms_summary"])


@register("kin_norms_table", "dashboard", "Literature norms table", domain="kinematics")
def kin_norms_table(req: ViewRequest) -> TableSpec:
    su = req.ctx.table("stats_kinematics", "norms_summary")
    t = B.frame_table(su, "kin_norms_table", "Cohort vs published reference means: mean and "
                      "SD of the participant differences", width="text",
                      columns=["metric", "ref_mean", "n", "mean_diff", "sd_diff", "source"], ctx=req.ctx)
    t.sources = ["stats_kinematics:norms_summary"]
    return t


@register("kin_influence", "dashboard", "Leave-one-out influence", domain="kinematics")
def kin_influence(req: ViewRequest) -> FigureSpec:
    """Change of the cohort mean when one (older) participant is left out."""
    ctx = req.ctx
    inf = ctx.table("stats_kinematics", "influence")
    if not inf.empty:
        inf = inf.assign(who=[who(ctx, p) for p in inf.left_out])
        inf = inf[inf.who != "not included"]
    metrics = list(dict.fromkeys(inf.metric)) if not inf.empty else []
    rows = sorted(set(inf.who), key=lambda w: int(w[1:])) if not inf.empty else []
    p = B.matrix_heat(inf, "who", "metric", "change_pct", "Change of the cohort mean",
                      rows=rows, row_labels=rows, cols=metrics,
                      col_labels=[_label(m) for m in metrics], fmt="{:+.1f}",
                      cb=Colorbar("Change", V.PCT, ramp="diverging"))
    if inf.empty:
        p.note = "no left-out participant among the included"
    return FigureSpec("kin_influence", "Leave-one-out influence", [p], width="column",
                      height_in=_h(len(set(inf.who)) if not inf.empty else 1, 0.3),
                      caption="Change of the cohort mean (%) when the participant of the row is "
                              "left out (configured older participants that are included).",
                      sources=["stats_kinematics:influence"])


def _subject_columns(ctx: Any) -> list[str]:
    return numeric_columns(ctx.table("stats_kinematics", "subjects"))


@register("kin_subjects", "dashboard", "Subject-level metrics", domain="kinematics",
          choices={"metric": _subject_columns}, metric="opening_selfpaced_mm")
def kin_subjects(req: ViewRequest) -> FigureSpec:
    """One subject-level metric (S1) per participant and the cohort mean with its CI."""
    ctx = req.ctx
    su = ctx.table("stats_kinematics", "subjects")
    m = req.options.get("metric") or "opening_selfpaced_mm"
    if m not in su and numeric_columns(su):
        m = numeric_columns(su)[0]
    rv = ctx.table("stats_kinematics", "reference_values")
    r = rv[(rv.metric == m) & (rv.group == "all")] if not rv.empty else rv
    p = per_participant(ctx, su, m, _label(m), ci=r.empty or not np.isfinite(r.ci_lo.iloc[0]))
    if not r.empty and np.isfinite(r.ci_lo.iloc[0]):
        p.series = [s for s in p.series if s.label != "median"]   # the mean is the reference
        p.series.insert(0, Series("hspan", x=np.array([float(r.ci_lo.iloc[0])]),
                                  y2=np.array([float(r.ci_hi.iloc[0])]), role="faint"))
        p.series.append(Series("vline", x=np.array([float(r["mean"].iloc[0])]),
                               role="reference", label="mean"))
    return FigureSpec("kin_subjects", "Subject-level metrics", [p], width="column",
                      height_in=_h(len(ctx.pids())),
                      caption=f"{_label(m)} per participant (dots; "
                              + ("dashed: cohort mean, band: its 95 % BCa CI" if not r.empty
                                 and np.isfinite(r.ci_lo.iloc[0])
                                 else PP_CAPTION[7:-1]) + ")." + _desc(m),
                      sources=["stats_kinematics:subjects", "stats_kinematics:reference_values"])


@register("kin_reliability", "dashboard", "Cycle-to-cycle reliability", domain="kinematics")
def kin_reliability(req: ViewRequest) -> FigureSpec:
    """ICC(3,1) with CI and MDC95 across cycles, per task and metric."""
    rel = req.ctx.table("stats_kinematics", "reliability")
    labels = [f"{V.task_label(r.task)}{' ' + SIDE.get(r.side, '') if r.side else ''}: "
              f"{_label(r.metric)}" for r in rel.itertuples()] if not rel.empty else []
    p = B.forest(labels, rel.icc31 if labels else [], rel.icc_lo if labels else [],
                 rel.icc_hi if labels else [], "ICC(3,1), 95 % CI", Axis("ICC"), ref=None)
    p.series.append(Series("vline", x=np.array([0.75]), role="threshold"))
    if labels:
        mdc = [f"{v:.1f}" if np.isfinite(v) else "" for v in rel.mdc95]
        p.series.append(Series("text", np.full(len(labels), 1.02), np.arange(len(labels),
                               dtype=float), text=mdc, style={"anchor": "left"}))
        p.x.lim = (min(0.0, float(np.nanmin(rel.icc_lo))) if rel.icc_lo.notna().any()
                   else 0.0, 1.25)
    return FigureSpec("kin_reliability", "Cycle-to-cycle reliability", [p], width="column",
                      height_in=_h(len(labels), 0.24),
                      caption="Agreement of a metric across the first cycles within the "
                              "session: ICC(3,1) with 95 % CI (dotted: 0.75, good "
                              "reliability), and the minimal detectable change MDC95 printed "
                              "on the right (in the metric's unit).",
                      sources=["stats_kinematics:reliability"])


@register("kin_hinge_summary", "dashboard", "Joint-model summary table", domain="kinematics")
def kin_hinge_summary(req: ViewRequest) -> TableSpec:
    hs = req.ctx.table("stats_kinematics", "hinge_summary")
    if not hs.empty:
        hs = hs.assign(model=[MODELS.get(m, m) for m in hs.model],
                       metric=[_label(m) for m in hs.metric])
    t = B.frame_table(hs, "kin_hinge_summary", "Misalignment of the three exoskeleton joint "
                      "models over participants: median [IQR], 95 % bootstrap CI (mm)",
                      columns=["model", "metric", "n", "median", "q1", "q3", "ci_lo", "ci_hi"],
                      width="text", ctx=req.ctx)
    t.sources = ["stats_kinematics:hinge_summary"]
    return t


# == uncertainty =================================================================================
@register("unc_null", "dashboard", "Hinge residual vs noise", params=("pid",),
          domain="uncertainty")
def unc_null(req: ViewRequest) -> FigureSpec:
    """H0 distributions of the best-fit hinge residual with the observed value (G'1)."""
    ctx, pid = req.ctx, _pid(req)
    a = ctx.arrays("uncertainty", pid) if pid else {}
    hn = ctx.table("uncertainty", "hinge_noise")
    r = hn[hn.pid == pid] if not hn.empty else hn
    obs = float(r.obs_hinge_fit_mm.iloc[0]) if not r.empty else np.nan
    obs_teeth = float(r.obs_hinge_teeth_mm.iloc[0]) \
        if not r.empty and "obs_hinge_teeth_mm" in r else np.nan
    obs_tmj = float(r.obs_hinge_tmj_mm.iloc[0]) \
        if not r.empty and "obs_hinge_tmj_mm" in r else np.nan
    panels = []
    nulls = [("bench", "Noise: bench precision"), ("invivo", "Noise: in-vivo floor")]
    # D24 / D25 sensitivities and the D33 condyle hinge, where computed (the motion null only
    # when configured)
    nulls += [(k, t) for k, t in (("invivo_slow", "Sensitivity: slow rest noise (3 s)"),
                                  ("invivo_motion", "Sensitivity: rest noise scaled to motion"),
                                  ("invivo_teeth", "Sensitivity: teeth-only fit"),
                                  ("bench_tmj", "Condyle hinge: bench precision"),
                                  ("invivo_tmj", "Condyle hinge: in-vivo floor"))
              if f"null__{k}" in a]
    for k, title in nulls:
        o = (obs_teeth if k.endswith("_teeth") else obs_tmj if k.endswith("_tmj") else obs)
        p = B.histogram(a.get(f"null__{k}", []), title, Axis("Incisal RMS residual", V.MM),
                        observed=o, quantile=0.99)
        null = _f(a, f"null__{k}")
        if null is None:
            _empty(p)
        elif np.isfinite(o) and null.size:
            lo, hi = min(np.nanmin(null), o), max(np.nanmax(null), o)
            p.x.lim = (max(0.0, lo - 0.1 * (hi - lo)), hi + 0.3 * (hi - lo))
        elif not r.empty and f"ratio_{k}" in r:
            p.note = f"ratio to P99 {r[f'ratio_{k}'].iloc[0]:.1f}"
        panels.append(p)
    ncol = 2 if len(panels) <= 2 else 3
    return FigureSpec("unc_null", "Hinge residual vs noise", panels,
                      rows=-(-len(panels) // ncol), cols=ncol,
                      width="text", height_in=1.5,
                      caption=f"{who(ctx, pid) if pid else ''}: incisal RMS residual of the "
                              "best-fit hinge under H0 (a perfect hinge rebuilt from the "
                              "participant's own openings plus pose noise, same filter and "
                              "fit), with its 99th percentile (dotted) and the observed residual "
                              "(accent). Above P99: more than noise. Further panels: the "
                              "slow-noise, motion-scaled and teeth-only sensitivities, where "
                              "computed.",
                      sources=["uncertainty:null", "uncertainty:hinge_noise"])


@register("unc_ci", "dashboard", "Hinge residual CI", domain="uncertainty")
def unc_ci(req: ViewRequest) -> FigureSpec:
    """Observed residual per participant and model with its bootstrap 95 % interval."""
    ctx = req.ctx
    pids = ctx.pids()
    hn = ctx.table("uncertainty", "hinge_noise")
    hn = hn.drop_duplicates("pid").set_index("pid").reindex(pids) if not hn.empty else hn
    draws = {p: ctx.arrays("uncertainty", p) for p in pids}
    panels = []
    for m, lab in MODELS.items():
        est, lo, hi = [], [], []
        for p in pids:
            d = _f(draws[p], f"ci__{m}")
            d = d[np.isfinite(d)] if d is not None else np.empty(0)
            q = np.percentile(d, [2.5, 97.5]) if d.size else (np.nan, np.nan)
            lo.append(q[0])
            hi.append(q[1])
            est.append(float(hn.at[p, f"obs_{m}_mm"]) if not hn.empty and p in hn.index
                       else np.nan)
        pa = B.forest(B.pseudo_rows(ctx, pids), est, lo, hi, lab,
                      Axis("Incisal RMS residual", V.MM), ref=None)
        if m == "hinge_fit" and not hn.empty and "null_invivo_p99_mm" in hn:
            pa.series.append(Series("scatter", hn.null_invivo_p99_mm.to_numpy(float),
                                    np.arange(len(pids), dtype=float), role="threshold",
                                    style={"ms": "|", "size": 1.4}))
            pa.note = "red tick: in-vivo noise P99"
        panels.append(pa)
    return FigureSpec("unc_ci", "Hinge residual CI", panels, rows=1, cols=3, width="text",
                      height_in=_h(len(pids), 0.2, 1.5),
                      caption="Observed incisal RMS residual per participant and joint model "
                              "(diamond) with the 95 % interval of the bootstrap over cycles "
                              "and landmark perturbation (line); tick: 99th percentile of the "
                              "in-vivo noise null for the best-fit hinge.",
                      sources=["uncertainty:ci", "uncertainty:hinge_noise"])


@register("unc_hinge_noise", "dashboard", "Hinge noise table", domain="uncertainty")
def unc_hinge_noise(req: ViewRequest) -> TableSpec:
    t = B.frame_table(req.ctx.table("uncertainty", "hinge_noise"), "unc_hinge_noise",
                      "Monte-Carlo noise floor of the hinge residual per participant: observed "
                      "residuals with bootstrap CI, null medians and 99th percentiles, ratio "
                      "observed / P99 (mm)", ctx=req.ctx, width="text")
    t.sources = ["uncertainty:hinge_noise"]
    return t


# == variability =================================================================================
def _outcomes(ctx: Any) -> list[str]:
    s = ctx.table("variability", "spec_values")
    return sorted(set(s.outcome)) if "outcome" in s else []


def _outcome_axis(o: str) -> Axis:
    head = o.split("|")[0]
    if head == "median_pct":
        return Axis("Median activation", V.PCT)
    if o.endswith("|lead") or "lead" in o:
        return Axis("EMG lead", V.MSEC)
    return axis_of(head)


@register("var_spec_values", "dashboard", "Processing variants", domain="variability",
          choices={"outcome": _outcomes}, outcome="mio_mm")
def var_spec_values(req: ViewRequest) -> FigureSpec:
    """Participant values of one outcome under every defensible processing variant (S2)."""
    ctx = req.ctx
    s = ctx.table("variability", "spec_values")
    o = req.options.get("outcome") or "mio_mm"
    g = s[s.outcome == o] if not s.empty else s
    sc = ctx.table("variability", "spec_curve")
    r = sc[sc.outcome == o] if not sc.empty else sc
    prim = r.primary_variant.iloc[0] if not r.empty else None
    variants = sorted(set(g.variant), key=lambda v: (v != prim, v)) if not g.empty else []
    short = [V.term_in(V.vocabulary()[1]["variant"], v, "short") for v in variants]
    p = B.strip_ci([lab + (" (primary)" if v == prim else "") for lab, v in zip(short, variants)],
                   [g[g.variant == v].value.to_numpy(float) for v in variants],
                   V.rewrite_ids(o), _outcome_axis(o),
                   ref=float(r.primary_mean.iloc[0]) if not r.empty else None,
                   ref_label="primary mean" if not r.empty else None, seed=B.seed_of(ctx))
    return FigureSpec("var_spec_values", "Processing variants", [p], width="text",
                      height_in=_h(len(variants), 0.17),
                      caption="Values of one outcome per participant (dots) and their median "
                              f"(diamond; {B.MEDIAN_CI_CAPTION}) under each processing variant "
                              "of the specification curve; dashed: cohort mean of the primary "
                              "variant"
                              + (f"; spread / between-SD = {r.spread_over_sd.iloc[0]:.2f}"
                                 if not r.empty else "") + ".",
                      sources=["variability:spec_values", "variability:spec_curve"])


@register("var_cycles_needed", "dashboard", "Cycles needed", domain="variability")
def var_cycles_needed(req: ViewRequest) -> FigureSpec:
    """Share of participants whose mean reaches ± MDC/2 after n cycles, per metric and side."""
    cn = req.ctx.table("variability", "cycles_needed")
    keys = list(dict.fromkeys(zip(cn.task, cn.metric))) if not cn.empty else []
    panels = []
    for task, metric in keys:
        g = cn[(cn.task == task) & (cn.metric == metric)]
        p = Panel(f"{V.task_label(task)}: {_label(metric)}", Axis("Cycles"),
                  Axis("Participants reaching target", V.PCT, lim=(0, 105)), [],
                  sharex="n")
        n_max = int(np.nanmax(g.n_available)) if g.n_available.notna().any() else 10
        x = np.arange(1, n_max + 1, dtype=float)
        sides = [s for s in dict.fromkeys(g.side.fillna(""))]
        for s in sides:
            gs = g[g.side.fillna("") == s]
            n = gs.n_needed.to_numpy(float)
            y = 100 * np.array([(n <= k).sum() for k in x]) / max(len(n), 1)
            p.series.append(Series("line", x, y, role=f"side:{s}" if s else "accent",
                                   label=f"{s} side" if s else None))
        p.note = f"target ± {g.target_halfwidth.iloc[0]:.2g}" if len(sides) == 1 else ""
        panels.append(p)
    if not panels:
        panels = [_empty(Panel("Cycles needed", Axis("Cycles"), Axis("Participants", V.PCT)))]
    cols = min(3, len(panels))
    return FigureSpec("var_cycles_needed", "Cycles needed", panels,
                      rows=int(np.ceil(len(panels) / cols)), cols=cols, width="text",
                      height_in=1.45,
                      caption="Share of participants whose cycle mean has a 95 % CI half-width "
                              "of at most MDC95/2 after n cycles (participants that never reach "
                              "it within their recorded cycles count as not reached); "
                              "laterotrusion per side with its own MDC.",
                      sources=["variability:cycles_needed"])


@register("var_protocol", "dashboard", "Protocol heterogeneity", domain="variability")
def var_protocol(req: ViewRequest) -> TableSpec:
    pr = req.ctx.table("variability", "protocol")
    if not pr.empty:
        pr = pr.assign(item=[i.replace("|", ": ").replace("_", " ") for i in pr["item"]])
    t = B.frame_table(pr, "var_protocol", "What varied across participants: phase durations "
                      "(s), good cycles per task, median cycle duration (s), masked MVC share",
                      columns=["item", "min", "median", "max", "n"], ctx=req.ctx)
    t.sources = ["variability:protocol"]
    return t


@register("var_spec_curve", "dashboard", "Specification curve table", domain="variability")
def var_spec_curve(req: ViewRequest) -> TableSpec:
    sc = req.ctx.table("variability", "spec_curve")
    t = B.frame_table(sc, "var_spec_curve", "Specification curve: cohort means across "
                      "processing variants, their spread and the between-participant SD of "
                      "the primary variant", width="text",
                      columns=["outcome", "n_variants", "primary_variant", "primary_mean",
                               "variant_min", "variant_max", "spread", "sd_between",
                               "spread_over_sd"], ctx=req.ctx)
    t.sources = ["variability:spec_curve"]
    return t
