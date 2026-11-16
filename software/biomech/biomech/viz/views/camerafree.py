# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Views of the camera-free study X5 (dataset) and VD20 (validation)

Dataset (``camerafree``, ``camerafree_cohort``):

* ``cf_ladder``: the estimator ladder and the camera use of every rung, with the cohort
  result of each rung (concept figure, H4.6 Fig. 1);
* ``cf_opening_ladder``: opening-angle RMSE per participant across the rotation rungs with
  the 1.2° bound (Fig. 2);
* ``cf_drift``: one participant's signed opening error over the take, E2 against E3, with
  the anchor residuals and the optical scatter of the anchors (Fig. 3);
* ``cf_incisor_axes``: incisor RMSE per axis by task group with the 2 mm bound (Fig. 4);
* ``cf_traces``: one participant's traces of every estimator (all stored signals);
* ``cf_table_results``: cohort results of every rung with its contrast (T2);
* ``cf_decomposition``: the oracle analyses next to E4own per task -- the incisor error with
  optical rotation + own model (translation-model error), with camera-free rotation +
  optical translation (rotation-induced error), the task-conditioned model (oracle class)
  and the full E4own;
* ``cf_table_oracle``: their cohort medians per task and the paired contrasts;
* ``cf_oracle_tasks``: the per-participant rows behind them (participant x task heatmap of
  one oracle combination).

Validation (``ref_camerafree``): ``val_camerafree`` and ``val_table_camerafree``, per
participant against Qualisys with the non-inferiority margins.

Stored results only; nothing is recomputed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Colorbar, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register

DS = {"group": "dataset", "domain": "imu"}
VAL = {"group": "validation", "domain": "validation"}
OPEN_BOUND, INC_BOUND = 1.2, 2.0
EVIDENCE = "exploratory (X5)"
# rung -> (rotation, translation, camera use) --
LADDER = {
    "E0": ("optics", "optics", "continuous (reference)"),
    "E1": ("in-take 7-parameter fit on free motion (C3a)", "-", "dynamic, in-take"),
    "E2a": ("hand-eye mountings, landmark frame, heading from the optical closed pose", "-",
            "static (holds, closed pose)"),
    "E2b": ("as E2a, heading from the IMU-detected closed poses", "-", "landmark frame only"),
    "E3": ("E2 reset at every IMU-detected closed pose", "-", "as E2"),
    "E4hinge": ("E3", "rotation about the condyle midpoint", "static"),
    "E4own": ("E3", "own ridge model (free motion)", "15 s dynamic"),
    "E4gen": ("E3", "generic ridge model (other participants)", "static"),
    "E5": ("E3", "accelerometers between closed-pose anchors", "static + hand-eye lever arm"),
    "E6": ("E3", "E5 with E4own as pseudo-measurements", "as E4own"),
    "E7": ("E3", "E4 model chosen by movement class", "as E4"),
}
CAMERA = ("continuous optics", "in-take dynamic block", "optical closed pose",
          "landmark holds", "hand-eye mountings", "hand-eye lever arm",
          "15 s optical calibration", "other participants' optics")
USES = {"E0": (0, 3), "E1": (1, 2, 3), "E2a": (2, 3, 4), "E2b": (3, 4), "E3": (3, 4),
        "E4hinge": (3, 4), "E4own": (3, 4, 6), "E4gen": (3, 4, 7), "E5": (3, 4, 5),
        "E6": (3, 4, 5, 6), "E7": (3, 4, 6, 7)}
ROT_RUNGS = ("E1", "E2a", "E2b", "E3", "E3_slerp")


def rung(est: str) -> str:
    """Reader name of a ladder rung or stored estimator. The rung codes (E1 ... E7) are
    analysis-plan names and the export drops "E<n>" as an internal code (a tick label or
    table cell holding only the code would come out empty), so reader-facing labels name
    what the rung does (vocabulary domain ``estimator``, short form)."""
    return OPTICS if est == "F0" else V.term("estimator", est, "short")


OPTICS = "JTS-MM optics"
# Short tick labels of the rotation rungs (column-width figure).
# written out ("opt. heading", "resets offl." read as code and forced 90° labels)
RUNG_TICK = {"E1": "in-take fit", "E2a": "optical heading", "E2b": "IMU heading",
             "E3": "resets", "E3_slerp": "resets, offline"}
# Row labels of the results table, as in Table I of the ICORR paper (2026-11-12), and
# its row order (translation steps: condyle hinge, generic model, own model)
TABLE_LABEL = {"E1": "In-take fit", "E2a": "Optical heading", "E2b": "IMU heading",
               "E3": "Resets", "E4hinge": "Condyle hinge", "E4gen": "Generic model",
               "E4own": "Own model", "E5": "Accelerometers", "E6": "Accelerometers + own model",
               "E7": "Model by movement class"}
TABLE_ORDER = ("E1", "E2a", "E2b", "E3", "E4hinge", "E4gen", "E4own", "E5", "E6", "E7")


def _table_ref(est: str) -> str:
    """A step named inside a cell ("vs resets"): its table label, lower case except IMU."""
    lab = TABLE_LABEL.get(est, rung(est))
    return lab if lab.startswith("IMU") else lab[0].lower() + lab[1:]


# Camera use of the rungs whose LADDER text points at another rung by its code.
CAMERA_TEXT = {"E3": "as mountings, IMU heading", "E6": "as own model",
               "E7": "as own and generic model"}
NBSP = "\u00a0"                # keeps a CI on one line in a wrapping table cell
# reader names of the VD20 non-inferiority outcomes (cf_val_noninferiority.outcome)
NI_OUTCOME = {"opening_abs_err_deg": "opening angle |difference| (°)",
              "incisor_err_mm": "incisor difference (mm)"}
TR_RUNGS = ("E4hinge", "E4gen", "E4own", "E5", "E6")
# short names of the protocol phases for the bands of the drift figure (column width)
PHASE_SHORT = {"rest": "rest", "open_close": "open-close", "protrusion_retrusion": "protrusion",
               "left_right": "laterotrusion", "chewing": "chewing",
               "free_motion": "free motion"}
INC_XLIM = (0.6, 40.0)         # incisor-error axis of the paper figures (mm, logarithmic)
# no tick labels of our own: the layout rotates explicit labels by a linear spacing rule
INC_TICKS = [1.0, 2.0, 5.0, 10.0, 20.0]
# row labels of the translation steps in the column-narrow panels of the incisor figure
TR_TICK = {"E6": "accel. + own model"}
# oracle analyses next to E4own (X5b decomposition, X5c task-conditioned model)
ORACLE = ("X5b_model", "X5b_rotation", "X5c", "E4own")
ORACLE_TASKS = ("open_close", "chewing", "protrusion_retrusion", "left_right")
# table headers written out (the figure short forms "opt. rot. + model" read as code)
ORACLE_HEADER = {"X5b_model": "Optical rotation + own model",
                 "X5b_rotation": "IMU rotation + optical translation",
                 "X5c": "Task model", "E4own": "Own model"}
ORACLE_EVIDENCE = "exploratory oracle analysis (optics in the loop; not a camera-free result)"


def _table(ctx: Any, stage: str, name: str, pids: list[str] | None = None) -> pd.DataFrame:
    try:
        return ctx.table(stage, name, pids)
    except (KeyError, FileNotFoundError, LookupError):
        return pd.DataFrame()


def _arrays(ctx: Any, stage: str, pid: str | None) -> dict[str, np.ndarray]:
    if pid is None:
        return {}
    try:
        return ctx.arrays(stage, pid)
    except (KeyError, FileNotFoundError, LookupError):
        return {}


def _empty(fid: str, title: str, msg: str) -> FigureSpec:
    """A card for a missing result: the message inside a tick-free unit box (a default
    axis would put its tick labels outside the small figure)."""
    p = Panel("", Axis("", lim=(0.0, 1.0)), Axis("", lim=(0.0, 1.0)),
              [Series("text", np.array([0.5]), np.array([0.45]), text=[msg],
                      style={"anchor": "above"})], legend="none")
    p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
    return attach(FigureSpec(fid, title, [p], width="column", height_in=1.2),
                  CaptionSpec(msg, evidence=EVIDENCE))


def _who(ctx: Any, pid: str) -> str:
    from biomech.viz.views.results_kinematics import who

    return who(ctx, pid)


def _prim(ctx: Any) -> str:
    s = _table(ctx, "camerafree", "cf_summary")
    if "primary_e3" in s and s["primary_e3"].notna().any():
        return str(s["primary_e3"].dropna().iloc[0])
    return "E3b"


def _rung_name(est: str, prim: str) -> str:
    """Stored estimator name of a ladder rung (E3: the primary, E3_slerp its offline twin)."""
    return {"E3": prim, "E3_slerp": f"{prim}_slerp"}.get(est, est)


def _cf_pid(req: ViewRequest) -> str | None:
    """The requested participant, else the first one with camera-free arrays."""
    pids = list(req.ctx.pids())
    if req.pid in pids:
        return req.pid
    s = _table(req.ctx, "camerafree", "cf_summary")
    ok = list(s.loc[s["configured"].astype(bool), "pid"]) if "configured" in s else []
    return next((p for p in pids if p in ok), pids[0] if pids else None)


def _ci(r: Any, col: str = "median", dec: int = 2) -> str:
    """``median [lo, hi]`` with the shared print rule (decimals by size, real minus, no
    negative zero: the accelerometer rung reaches hundreds of mm)."""
    if r is None or not np.isfinite(r.get(col, np.nan)):
        return "--"
    return B.est_ci(r[col], r.get("ci_lo", np.nan), r.get("ci_hi", np.nan), dec)


def _cohort_row(c: pd.DataFrame, est: str, outcome: str) -> dict[str, Any] | None:
    if c.empty or "estimator" not in c:
        return None
    r = c[(c["estimator"] == est) & (c["outcome"] == outcome)]
    return None if r.empty else r.iloc[0].to_dict()


def _n_pid(d: pd.DataFrame) -> int:
    return int(d["pid"].nunique()) if "pid" in d and len(d) else 0


# -- concept ---------------------------------------------------------------------------------
@register("cf_ladder", "dashboard", "Camera-free ladder: camera use per rung", **DS)
def cf_ladder(req: ViewRequest) -> FigureSpec:
    """X5 concept: the rungs of the estimator ladder, what each takes from the cameras, and
    the cohort median (95 % BCa CI) of its primary error."""
    fid, title = "cf_ladder", "Camera-free ladder: camera use per rung"
    rungs = list(LADDER)
    y = np.arange(len(rungs), dtype=float)
    pa = Panel("(a) Camera use", Axis(""), Axis(""), [], legend="none")
    for j, cam in enumerate(CAMERA):
        ys = np.array([i for i, r in enumerate(rungs) if j in USES[r]], float)
        if ys.size:
            pa.series.append(Series("scatter", np.full(ys.size, float(j)), ys, role="accent",
                                    style={"ms": "s", "size": 1.4}))
    pa.x.ticks, pa.x.ticklabels = [float(j) for j in range(len(CAMERA))], list(CAMERA)
    pa.y.ticks, pa.y.ticklabels, pa.y.reverse = list(y), [rung(r) for r in rungs], True
    c = _table(req.ctx, "camerafree_cohort", "cf_cohort")
    pb = Panel("(b) Cohort error against the optics", Axis("Median RMSE"), Axis(""), [],
               legend="below")
    n = 0
    for oc, lab, role, unit in (("rmse_open_deg", "opening angle (deg)", "primary", V.DEG),
                                ("rmse_inc_mm", "incisor (mm)", "secondary", V.MM)):
        xs, lo, hi, yy = [], [], [], []
        for i, r in enumerate(rungs):
            row = _cohort_row(c, r, oc)
            if row is not None and np.isfinite(row.get("median", np.nan)):
                xs.append(row["median"])
                lo.append(row.get("ci_lo", np.nan))
                hi.append(row.get("ci_hi", np.nan))
                yy.append(i)
                n = max(n, int(row.get("n", 0) or 0))
        if xs:
            pb.series.append(Series("scatter", np.array(xs), np.array(yy, float), role=role,
                                    style={"ms": "d", "size": 1.3}, label=lab))
            for x0, x1, yi in zip(lo, hi, yy):
                if np.isfinite(x0) and np.isfinite(x1):
                    pb.series.append(Series("line", np.array([x0, x1]), np.array([yi, yi], float),
                                            role=role))
    # bound lines unlabelled in the panel (1.2 and 2 sit too close for two labels); the
    # caption names them
    pb.series.append(Series("vline", x=np.array([OPEN_BOUND]), role="threshold"))
    pb.series.append(Series("vline", x=np.array([INC_BOUND]), role="reference"))
    pb.y.ticks, pb.y.ticklabels, pb.y.reverse = list(y), [rung(r) for r in rungs], True
    # x from 0, with room right of the larger bound for its label (no cohort rows: only the
    # bounds would set the range and the label would sit outside the figure)
    vals = [v for s_ in pb.series if s_.kind in ("scatter", "line")
            for v in np.asarray(s_.x, float) if np.isfinite(v)]
    top = max(vals, default=0.0)
    pb.x.lim = (0.0, top * 1.05 if top > 1.4 * INC_BOUND else 1.6 * INC_BOUND)
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=2.6, sources=["camerafree_cohort:cf_cohort"])
    return attach(spec, CaptionSpec(
        "the estimator ladder of the camera-free study: (a) what each rung takes from the "
        "cameras (the optics themselves; an in-take optical calibration; from the hand-eye "
        "rungs on only static calibration: the landmark holds and the hand-eye IMU mountings, "
        "and per rung the additions named); (b) the cohort median of each rung's error against "
        "the optics (opening angle over all scored tasks; incisor over all scored tasks)",
        [("squares", "camera information used"), ("diamonds", "cohort median"),
         ("lines", "95 % BCa CI"), ("left vertical line", "opening bound 1.2°"),
         ("right vertical line", "incisor bound 2 mm")],
        f"n = {n} participants with jaw IMU" if n else "no camera-free results",
        "median over the participants of each participant's median over the scored tasks",
        evidence=EVIDENCE,
        notes=["The class-selected translation model was not computed (see the results "
               "table)."]))


# -- opening RMSE across the ladder ---------------------------------------------------------
LADDER_FORMS = ("lines", "heat", "share", "rows")
LADDER_SHOWS = ("opening-angle RMSE of the IMU rotation against the optics per rung: the in-take "
                "calibration (in-take fit), the hand-eye mountings and the landmark frame with the "
                "heading from the optical closed pose (optical heading) or from the "
                "IMU-detected closed poses (IMU heading), the latter with a reset at every "
                "IMU-detected closed pose (resets; causal, the primary rung) and its offline "
                "variant (resets, offline)")


@register("cf_opening_ladder", "dashboard", "Camera-free ladder: opening RMSE",
          choices={"form": LADDER_FORMS, "offline": ("yes", "no")}, form="lines", offline="yes",
          **DS)
def cf_opening_ladder(req: ViewRequest) -> FigureSpec:
    """Opening-angle RMSE against the optics per participant across the rotation rungs
    (median over the scored tasks), cohort median with 95 % BCa CI, the 1.2° bound. Forms
    (J2.1): ``lines`` one line per participant; ``heat`` participants x rungs with
    the RMSE in each cell; ``share`` per rung the participants within the bound."""
    fid, title = "cf_opening_ladder", "Camera-free ladder: opening RMSE"
    form = req.options.get("form") or "lines"
    part = _table(req.ctx, "camerafree_cohort", "cf_participant")
    c = _table(req.ctx, "camerafree_cohort", "cf_cohort")
    if part.empty or "outcome" not in part:
        return _empty(fid, title, "no camera-free cohort results")
    prim = _prim(req.ctx)
    d = part[part["outcome"] == "rmse_open_deg"]
    rungs = [r for r in ROT_RUNGS if (d["estimator"] == _rung_name(r, prim)).any()]
    if req.options.get("offline") == "no":   # causal rungs only (JNER sensing (c), 2026-11-10)
        rungs = [r for r in rungs if r != "E3_slerp"]
    x = np.arange(len(rungs), dtype=float)
    srcs = ["camerafree_cohort:cf_participant", "camerafree_cohort:cf_cohort"]
    ticks = [RUNG_TICK.get(r, r) for r in rungs]
    pids = sorted(d.pid.unique(), key=lambda q: req.ctx.pseudonym(q))
    z = np.array([[float(d[(d.pid == q) & (d.estimator == _rung_name(r, prim))]["value"]
                         .median()) if ((d.pid == q) & (d.estimator == _rung_name(r, prim)))
                   .any() else np.nan for r in rungs] for q in pids])
    who = f"n = {_n_pid(d)} participants with jaw IMU"
    stat = "per participant the median over the scored tasks of the task RMSE"
    if form == "heat":
        p = B.heat(z, B.pseudo_rows(req.ctx, pids), ticks, "", fmt="{:.1f}",
                   cb=Colorbar("Opening RMSE", V.DEG))
        spec = FigureSpec(fid, title, [p], width="column", height_in=3.0, sources=srcs)
        return attach(spec, CaptionSpec(
            LADDER_SHOWS, [("rows", "participants"), ("columns", "rungs"),
                           ("cell colour and number", "opening RMSE in degrees")],
            who, stat, evidence=EVIDENCE))
    if form == "share":
        ok = np.sum(z <= OPEN_BOUND, axis=0)
        tot = np.sum(np.isfinite(z), axis=0)
        share = np.where(tot > 0, 100.0 * ok / np.maximum(tot, 1), np.nan)
        p = Panel("", Axis(""), Axis("Participants within the bound", V.PCT, lim=(0, 118),
                                     ticks=[0.0, 25.0, 50.0, 75.0, 100.0]),
                  [Series("bar", x, share, role="primary"),
                   Series("text", x, share + 3.0, text=[f"{a}/{b}" for a, b in zip(ok, tot)],
                          role="primary", style={"anchor": "above"})], legend="none")
        p.x.ticks, p.x.ticklabels = list(x), ticks
        spec = FigureSpec(fid, title, [p], width="column", height_in=2.2, sources=srcs)
        return attach(spec, CaptionSpec(
            LADDER_SHOWS + f"; per rung the share of participants whose RMSE is at most the "
            f"{OPEN_BOUND:g}° bound",
            [("bars", "share of participants within the bound"),
             ("numbers", "participants within the bound / participants with the rung")],
            who, stat + "; count per rung (descriptive)", evidence=EVIDENCE))
    if form == "rows":
        # rungs as rows (readable at any width); each participant a thin path across
        # the rungs, the cohort median and its 95 % BCa CI per rung
        # readability at column width (2026-11-15, ICORR figure round): the region within the
        # bound shaded, participants as visible dots, the cohort median and CI heavy, no
        # legend (the caption names the marks), a compact height
        y = np.arange(len(rungs), dtype=float)
        p = Panel("", Axis("Opening RMSE", V.DEG), Axis(""), [], legend="none")
        lo_x = min(0.3, float(np.nanmin(z)) - 0.1) if np.isfinite(z).any() else 0.3
        hi_x = (float(np.nanmax(z)) if np.isfinite(z).any() else OPEN_BOUND) + 0.15
        p.series.append(Series("hspan", x=np.array([lo_x]), y2=np.array([OPEN_BOUND]),
                               role="faint"))
        for pid, g in d.groupby("pid"):
            v = g.set_index("estimator")["value"]
            xv = np.array([float(v.get(_rung_name(r, prim), np.nan)) for r in rungs])
            p.series.append(Series("line", xv, y, role="faint", style={"width": 0.8}))
            p.series.append(Series("scatter", xv, y, role="muted", style={"size": 1.15}))
        for i, r in enumerate(rungs):
            row = _cohort_row(c, _rung_name(r, prim), "rmse_open_deg") or {}
            lo_, hi_ = row.get("ci_lo", np.nan), row.get("ci_hi", np.nan)
            if np.isfinite(lo_) and np.isfinite(hi_):
                p.series.append(Series("line", np.array([lo_, hi_]), np.array([i + 0.3] * 2),
                                       role="accent", style={"width": 2.2}))
            p.series.append(Series("scatter", np.array([row.get("median", np.nan)]),
                                   np.array([i + 0.3]), role="accent",
                                   style={"ms": "d", "size": 1.9}))
        p.series.append(Series("vline", x=np.array([OPEN_BOUND]), role="threshold",
                               label=f"{OPEN_BOUND:g}° bound"))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), ticks, True
        p.y.lim = (-0.55, len(rungs) - 0.35)
        p.x.lim = (lo_x, hi_x)
        spec = FigureSpec(fid, title, [p], width="column", height_in=1.75, sources=srcs)
        return attach(spec, CaptionSpec(
            LADDER_SHOWS,
            [("grey lines and dots", "participants"), ("diamonds and bars", "cohort median "
             "and 95 % BCa CI"), ("dashed line", f"bound {OPEN_BOUND:g}°"),
             ("shaded range", "within the bound")],
            who, stat, evidence=EVIDENCE))
    p = Panel("", Axis(""), Axis("Opening RMSE", V.DEG), [], legend="below")
    for pid, g in d.groupby("pid"):
        v = g.set_index("estimator")["value"]
        yv = np.array([float(v.get(_rung_name(r, prim), np.nan)) for r in rungs])
        p.series.append(Series("line", x, yv, role="faint"))
        p.series.append(Series("scatter", x, yv, role="muted", style={"size": 0.8}))
    med, lo, hi = [], [], []
    for r in rungs:
        row = _cohort_row(c, _rung_name(r, prim), "rmse_open_deg") or {}
        med.append(row.get("median", np.nan))
        lo.append(row.get("ci_lo", np.nan))
        hi.append(row.get("ci_hi", np.nan))
    p.series.append(Series("errorbar", x + 0.12, np.array(med, float), y2=np.array(hi, float),
                           ylo=np.array(lo, float), role="accent", label="cohort median, 95 % CI"))
    p.series.append(Series("hline", y=np.array([OPEN_BOUND]), role="threshold",
                           label="1.2° bound"))
    p.x.ticks = list(x)
    p.x.ticklabels = [RUNG_TICK.get(r, r) for r in rungs]
    # text width: the written-out rung names fit horizontally
    spec = FigureSpec(fid, title, [p], width="text", height_in=2.2, sources=srcs)
    return attach(spec, CaptionSpec(
        LADDER_SHOWS,
        [("grey lines", "participants"), ("black markers", "cohort median"),
         ("bars", "95 % BCa CI"), ("dashed line", "bound 1.2°")],
        f"n = {_n_pid(d)} participants with jaw IMU",
        "per participant the median over the scored tasks of the task RMSE",
        evidence=EVIDENCE))


# -- drift ------------------------------------------------------------------------------------
@register("cf_drift", "dashboard", "Camera-free drift: without and with resets",
          params=("pid",), **DS)
def cf_drift(req: ViewRequest) -> FigureSpec:
    """One participant: signed opening error over the take for the mountings alone (E2, no
    resets) and the closed-pose resets (E3, reset at every closed pose), the residual each
    anchor removed, and the optical scatter of the anchors."""
    fid, title = "cf_drift", "Camera-free drift: without and with resets"
    ctx, pid = req.ctx, _cf_pid(req)
    a = _arrays(ctx, "camerafree", pid)
    if "t_s" not in a:
        return _empty(fid, title, "no camera-free arrays for this participant")
    prim = _prim(ctx)
    t = np.asarray(a["t_s"], float)
    t = t - t[0]
    ref = np.asarray(a["open_deg__opt"], float)
    pa = Panel("(a) Signed opening error", Axis("Time in take", V.SEC),
               Axis("IMU - optics", V.DEG), [], legend="below", sharex="t")
    k = slice(None, None, 10)                       # 20 Hz is enough to show drift
    # legend in the words of the paper (2026-11-15, ICORR figure round): the steps are
    # "optical heading", "IMU heading" and "resets", an anchor is a detected rest pose
    lines = [("E2a", "secondary", "optical heading"),
             ("E2b", "tertiary", "IMU heading"),
             (prim, "primary", "resets")]
    if all(f"open_deg__{e}" in a for e in ("E2a", "E2b")) and np.allclose(
            np.asarray(a["open_deg__E2a"], float), np.asarray(a["open_deg__E2b"], float),
            atol=1e-9, equal_nan=True):
        # identical in this take: one line (the second was hidden under the first)
        lines = [("E2a", "secondary", "optical heading (here equal to the IMU heading)"),
                 lines[2]]
    err = {e: (np.asarray(a[f"open_deg__{e}"], float) - ref) for e, _, _ in lines
           if f"open_deg__{e}" in a}
    an = _table(ctx, "camerafree", "cf_anchors", [pid])
    an = an[an["estimator"] == prim] if "estimator" in an else an
    # the protocol phases as shaded bands with their names: the argument of the figure is
    # about which task a reset falls into
    spans = []
    if "phase" in a and "phase_names" in a:
        ph = np.asarray(a["phase"])
        for i, name in enumerate([str(x) for x in a["phase_names"]]):
            m = ph == i
            if m.any():
                spans.append((float(t[m][0]), float(t[m][-1]), PHASE_SHORT.get(name, name)))
    fin = [v[np.isfinite(v)] for v in err.values()]
    top = max([float(v.max()) for v in fin if v.size] + [1.0])
    for j, (t0, t1, name) in enumerate(spans):
        pa.series.append(Series("hspan", x=np.array([t0]), y2=np.array([t1]), role="faint"))
        # two rows of names, alternating, so that neighbours do not collide at column width
        pa.series.append(Series("text", np.array([0.5 * (t0 + t1)]),
                                np.array([top + 0.2 + 0.75 * (j % 2)]),
                                text=[name], style={"anchor": "above"}))
    for e, role, lab in lines:
        if e in err:
            pa.series.append(Series("line", t[k], err[e][k], role=role, label=lab,
                                    style={"width": 1.3}))
    if not an.empty:
        pa.series.append(Series("scatter", an["t0_s"].to_numpy(float),
                                an["resid_open_deg"].to_numpy(float), role="accent",
                                style={"ms": "v", "size": 1.5},
                                label="error removed by a reset"))
    pa.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    if spans:
        lo_y = min([float(v.min()) for v in fin if v.size] + [-1.0])
        pa.y.lim = (lo_y - 0.3, top + 1.9)
    # (b) and (c): one quantity per axis (they shared one unlabelled axis in deg and mm)
    pb = Panel("(b) Optical opening at the detected rest poses", Axis("Time in take", V.SEC),
               Axis("Opening", V.DEG), [], legend="none", sharex="t")
    pc = Panel("(c) Optical incisor displacement at these poses",
               Axis("Time in take", V.SEC), Axis("Incisor", V.MM), [], legend="none",
               sharex="t")
    if not an.empty:
        pb.series.append(Series("scatter", an["t0_s"].to_numpy(float),
                                an["opt_open_deg"].to_numpy(float), role="primary",
                                style={"size": 1.4}))
        pc.series.append(Series("scatter", an["t0_s"].to_numpy(float),
                                an["opt_inc_mm"].to_numpy(float), role="primary",
                                style={"ms": "s", "size": 1.4}))
    for q in (pb, pc):
        for t0, t1, _ in spans:
            q.series.insert(0, Series("hspan", x=np.array([t0]), y2=np.array([t1]),
                                      role="faint"))
        q.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    pb.height = pc.height = 0.6
    spec = FigureSpec(fid, title, [pa, pb, pc], rows=3, cols=1, width="text", height_in=1.9,
                      sources=["camerafree:cf_anchors"])
    return attach(spec, CaptionSpec(
        f"participant {_who(ctx, pid)}: (a) the IMU opening angle minus the optical one over "
        "the take for the mountings alone (one reference closed pose at rest) and with "
        "resets at every IMU-detected closed pose, each after the ground truth's low-pass; "
        "triangles: the error each anchor removed; (b) the optical opening angle and (c) the "
        "incisor displacement at each anchor against the optical closed pose (how closed the "
        "anchors really were)",
        [("lines", "signed opening error"), ("triangles", "anchor residuals"),
         ("dots (b, c)", "one anchor each"), ("dashed lines", "zero"),
         ("shaded bands", "protocol phases")],
        f"n = {len(an)} anchors", "signals of one take, one value per anchor (no summary)",
        evidence=EVIDENCE))


# -- incisor error per axis ------------------------------------------------------------------
@register("cf_incisor_axes", "dashboard", "Camera-free translation: incisor error per axis",
          **DS)
def cf_incisor_axes(req: ViewRequest) -> FigureSpec:
    """Incisor RMSE per axis of each translation rung by task group (rotational: open-close,
    chewing; translational: protrusion, laterotrusion), participants and cohort median, with
    the 2 mm bound."""
    fid, title = "cf_incisor_axes", "Camera-free translation: incisor error per axis"
    tr = _table(req.ctx, "camerafree", "cf_translation")
    gen = _table(req.ctx, "camerafree_cohort", "cf_generic")
    d = pd.concat([x for x in (tr, gen) if not x.empty], ignore_index=True) \
        if not (tr.empty and gen.empty) else pd.DataFrame()
    if d.empty or "task_group" not in d:
        return _empty(fid, title, "no camera-free translation results")
    d = d[d["scored"].astype(bool) & d["task_group"].isin(["rotational", "translational"])]
    rungs = [r for r in TR_RUNGS if (d["estimator"] == r).any()]
    part = _table(req.ctx, "camerafree_cohort", "cf_participant")
    c = _table(req.ctx, "camerafree_cohort", "cf_cohort")
    panels = []
    # 2026-11-15 (ICORR figure round): the axis ends at 40 mm, so that the steps near the
    # bound are resolved; a step whose median lies beyond (the accelerometers alone, by orders
    # of magnitude) is named as off scale with its value. The bound is defined for the 3-D
    # error, so each row shows the 3-D error (participants, cohort median and CI) and below
    # it the cohort median per axis.
    for grp, lab, oc in (("rotational", "(a) Open-close and chewing", "rmse_inc_mm_rot"),
                         ("translational", "(b) Protrusion and laterotrusion",
                          "rmse_inc_mm_trans")):
        g = d[d["task_group"] == grp]
        p = Panel(lab, Axis("Incisor RMSE", V.MM, log=True, lim=INC_XLIM, ticks=INC_TICKS),
                  Axis(""), [], legend="below")
        # one legend per panel: the 3-D marks under (a), the axes under (b)
        first = grp == "rotational"
        by_axis = grp == "translational"
        for i, e in enumerate(rungs):
            row = _cohort_row(c, e, oc) or {}
            med = float(row.get("median", np.nan))
            if np.isfinite(med) and med > INC_XLIM[1]:
                p.series.append(Series("text", np.array([INC_XLIM[1] * 0.92]),
                                       np.array([float(i)]),
                                       text=[f"off scale: {med:.0f} mm"],
                                       style={"anchor": "left"}))
                continue
            v = part[(part["estimator"] == e) & (part["outcome"] == oc)]["value"] \
                if not part.empty else pd.Series(dtype=float)
            v = v.to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                p.series.append(Series("scatter", np.clip(v, None, INC_XLIM[1] * 0.97),
                                       np.full(v.size, i - 0.2), role="muted",
                                       style={"size": 1.0},
                                       label="participants, 3-D error" if first else None))
            lo_, hi_ = row.get("ci_lo", np.nan), row.get("ci_hi", np.nan)
            if np.isfinite(lo_) and np.isfinite(hi_):
                p.series.append(Series("line", np.array([lo_, min(hi_, INC_XLIM[1])]),
                                       np.array([i - 0.2] * 2), role="accent",
                                       style={"width": 2.0}))
            p.series.append(Series("scatter", np.array([med]), np.array([i - 0.2]),
                                   role="accent", style={"ms": "d", "size": 1.8},
                                   label="cohort median and 95 % CI, 3-D error" if first
                                   else None))
            first = False
        for k, ax in enumerate("xyz"):
            col = f"rmse_inc_{ax}_mm"
            per = g.groupby(["pid", "estimator"])[col].median().reset_index() \
                if col in g else pd.DataFrame(columns=["pid", "estimator", col])
            med = [float(per[per["estimator"] == e][col].median()) if (per["estimator"] == e)
                   .any() else np.nan for e in rungs]
            med = np.array([m if np.isfinite(m) and m <= INC_XLIM[1] else np.nan for m in med])
            p.series.append(Series("scatter", med, np.arange(len(rungs)) + 0.12 + 0.13 * k,
                                   role=f"axis:{ax}", style={"ms": "os^"[k], "size": 1.3},
                                   label=f"{ax} axis, cohort median" if by_axis else None))
        p.series.append(Series("vline", x=np.array([INC_BOUND]), role="threshold",
                               label="2 mm bound"))
        p.y.ticks = [float(i) for i in range(len(rungs))]
        p.y.ticklabels, p.y.reverse = [TR_TICK.get(r, rung(r)) for r in rungs], True
        p.y.lim = (-0.6, len(rungs) - 0.35)
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.0,
                      sources=["camerafree:cf_translation", "camerafree_cohort:cf_generic",
                               "camerafree_cohort:cf_participant",
                               "camerafree_cohort:cf_cohort"])
    return attach(spec, CaptionSpec(
        "incisor error against the optics per maxillary axis (x anterior, y left, z up) for "
        "the translation rungs, all on the anchored IMU rotation (closed-pose resets): "
        "rotation about the condyle midpoint (condyle hinge), the generic and the own "
        "rotation-to-translation model, accelerometer bridging between closed-pose anchors "
        "(accelerometers) and both fused (accelerometers + own model); (a) the rotational "
        "tasks, (b) the translational tasks; per step the three-dimensional error and below "
        "it the error per axis; log axis up to 40 mm (accelerometer bridging alone drifts by "
        "orders of magnitude and is named as off scale)",
        [("grey dots", "participants, 3-D error (median over the group's tasks)"),
         ("black diamonds and bars", "cohort median of the 3-D error with 95 % BCa CI"),
         ("coloured marks", "cohort median per axis"), ("dashed line", "bound 2 mm")],
        f"n = {_n_pid(d)} participants with jaw IMU",
        "per participant the median RMS error over the group's tasks; the cohort median per "
        "axis", evidence=EVIDENCE))


# -- traces ---------------------------------------------------------------------------------
@register("cf_traces", "dashboard", "Camera-free traces", params=("pid",), **DS)
def cf_traces(req: ViewRequest) -> FigureSpec:
    """One participant's camera-free signals over the take: the opening angle of the optics
    and the rotation rungs, the opening error of each rung, the incisor error of each
    translation rung and the off-axis rotation of the primary E3 (before the low-pass) against the
    optics; closed poses and anchors marked."""
    fid, title = "cf_traces", "Camera-free traces"
    ctx, pid = req.ctx, _cf_pid(req)
    a = _arrays(ctx, "camerafree", pid)
    if "t_s" not in a:
        return _empty(fid, title, "no camera-free arrays for this participant")
    t = np.asarray(a["t_s"], float)
    t = t - t[0]
    k = slice(None, None, 10)
    names = [str(x) for x in a["phase_names"]]
    ph = np.asarray(a["phase"], int)
    labels = np.array([names[i] if i >= 0 else "" for i in ph], dtype=object)
    lane = B.phase_lane(B.phase_spans(t, labels))
    ref = np.asarray(a["open_deg__opt"], float)
    pa = Panel("(a) Opening angle", Axis("Time in take", V.SEC), Axis("Opening", V.DEG), [],
               legend="below", sharex="time")
    pb = Panel("(b) Opening error", Axis("Time in take", V.SEC), Axis("IMU - optics", V.DEG),
               [], legend="below", sharex="time")
    pc = Panel("(c) Incisor error", Axis("Time in take", V.SEC), Axis("3-D error", V.MM), [],
               legend="below", sharex="time")
    pd_ = Panel("(d) Off-axis rotation, closed-pose resets", Axis("Time in take", V.SEC),
                Axis("IMU - optics", V.DEG), [], legend="below", sharex="time")
    pa.series.append(Series("line", t[k], ref[k], role="accent", label="optics"))
    rot = sorted(x.split("__", 1)[1] for x in a if x.startswith("open_deg__")
                 and not x.endswith("__opt"))
    for i, e in enumerate(rot):
        y = np.asarray(a[f"open_deg__{e}"], float)
        if e in ("E1", "E2a", _prim(ctx)):
            pa.series.append(Series("line", t[k], y[k], role=f"cat:{i % 6}", label=rung(e)))
        pb.series.append(Series("line", t[k], (y - ref)[k], role=f"cat:{i % 6}",
                                label=rung(e)))
    inc_opt = np.asarray(a["incisor_mm__opt"], float)
    tr = sorted(x.split("__", 1)[1] for x in a if x.startswith("incisor_mm__")
                and not x.endswith("__opt"))
    for i, e in enumerate(tr):
        err = np.linalg.norm(np.asarray(a[f"incisor_mm__{e}"], float) - inc_opt, axis=1)
        pc.series.append(Series("line", t[k], err[k], role=f"cat:{i % 6}", label=rung(e)))
    d_rv = np.asarray(a["rotvec_deg__E3"], float) - np.asarray(a["rotvec_deg__opt"], float)
    for j, ax in ((0, "x"), (2, "z")):
        pd_.series.append(Series("line", t[k], d_rv[k, j], role=f"axis:{ax}",
                                 label=f"about {ax}"))
    for name, role in (("closed", "faint"), ("anchor", "muted")):
        m = np.asarray(a[name], bool)
        runs = B.phase_spans(t, np.where(m, name, ""))
        if runs:
            pb.series.append(Series("hspan", x=np.array([r[1] for r in runs]),
                                    y2=np.array([r[2] for r in runs]), role=role,
                                    style={"alpha": 0.3}))
    pc.series.append(Series("hline", y=np.array([INC_BOUND]), role="threshold", label="2 mm"))
    spec = FigureSpec(fid, title, [lane, pa, pb, pc, pd_], rows=5, cols=1, width="text",
                      height_in=1.5, sources=["camerafree:cf_summary"])
    return attach(spec, CaptionSpec(
        f"participant {_who(ctx, pid)}: the camera-free estimates over the take against the "
        "optics (every estimate after the ground truth's low-pass, except (d)): (a) opening "
        "angle, (b) opening error of every rotation rung with the IMU-detected closed runs "
        "(light) and the anchor windows (darker), (c) incisor 3-D error of every translation "
        "rung, (d) rotation about the maxillary x and z axes of the primary rung (closed-pose "
        "resets) before the low-pass minus the optics",
        [("lane", "task phases")], "one participant, 20 Hz for display", evidence=EVIDENCE))


# -- results table --------------------------------------------------------------------------
@register("cf_table_results", "dashboard", "Camera-free results (cohort)", **DS)
def cf_table_results(req: ViewRequest) -> TableSpec:
    """Per step: cohort median opening and incisor RMSE with 95 % BCa CI and the verdict
    against the bound in the same cell, and the paired contrast to the step it builds on
    (Hodges–Lehmann with exact CI, exact Wilcoxon p, Holm within the family). Row labels and
    order as Table I of the ICORR paper; the camera use of each step is stated there."""
    ctx = req.ctx
    c = _table(ctx, "camerafree_cohort", "cf_cohort")
    k = _table(ctx, "camerafree_cohort", "cf_contrasts")
    prim = _prim(ctx)
    rows, n = [], 0

    def with_verdict(r: Any) -> str:
        v = (r or {}).get("verdict", "") or ""
        return _ci(r) + (f", {v}" if v and _ci(r) != "--" else "")

    for r in TABLE_ORDER:
        est = _rung_name(r, prim) if r == "E3" else r
        ro = _cohort_row(c, "E3" if r == "E3" else est, "rmse_open_deg")
        rr = _cohort_row(c, est, "rmse_inc_mm_rot")
        rt = _cohort_row(c, est, "rmse_inc_mm_trans")
        kk = k[k["a"] == r] if "a" in k else pd.DataFrame()
        con = kk.iloc[0].to_dict() if len(kk) else {}
        n = max([n] + [int(x.get("n", 0) or 0) for x in (ro, rr, rt) if x])
        note = next((x.get("note") for x in (ro, rr, rt) if x and x.get("note")), "")
        hl = "--"
        if con and np.isfinite(con.get("hl", np.nan)):
            hl = B.est_ci(con["hl"], con.get("hl_ci_lo", np.nan), con.get("hl_ci_hi", np.nan))
            b = str(con.get("b", "") or "")
            hl = f"vs {_table_ref(b)}: {hl}" if b else hl
        if note and all(v == "--" for v in (with_verdict(ro), with_verdict(rr), _ci(rt))):
            continue                         # all dashes; the caption names it
        rows.append({
            "rung": TABLE_LABEL[r] + (" (primary)" if r == "E3" else "")
                    + (" (not computed)" if note else ""),
            "open": with_verdict(ro), "inc_rot": with_verdict(rr), "inc_trans": _ci(rt),
            "hl": hl,
            "p_holm": float(con.get("p_holm", np.nan)) if con else np.nan})
    cols = [Column("rung", "Step", align="X"),
            Column("open", "Opening, all tasks", V.DEG, align="X",
                   group="RMSE: median [95 % CI], verdict"),
            Column("inc_rot", "Incisor, rotational tasks", V.MM, align="X",
                   group="RMSE: median [95 % CI], verdict"),
            Column("inc_trans", "Incisor, translational tasks", V.MM, align="X",
                   group="RMSE: median [95 % CI], verdict"),
            Column("hl", "Contrast: HL [95 % CI]", align="X"),
            Column("p_holm", "p (Holm)", decimals=3)]
    t = TableSpec("cf_table_results", "", cols, rows, width="text",
                  sources=["camerafree_cohort:cf_cohort", "camerafree_cohort:cf_contrasts"])
    return attach(t, CaptionSpec(
        "camera-free steps against the optics: per step the cohort median of the "
        "participants' median task RMSE (opening angle over all scored tasks; incisor over "
        "open-close and chewing, and over protrusion and laterotrusion), with the verdict "
        "against the bound (opening 1.2°, incisor 2 mm), and the paired contrast to the "
        "step it builds on, named in the cell (rotation and translation steps are separate "
        "families; negative: the step is better)",
        [("step", "in-take fit: in-take 7-parameter fit on free motion; optical and IMU "
                  "heading: hand-eye mountings and landmark frame, heading from the optical or "
                  "the IMU-detected closed pose; resets: reset at every IMU-detected closed "
                  "pose; condyle hinge: rotation about the condyle midpoint; generic and own "
                  "model: ridge model of the other participants or of the participant's free "
                  "motion; accelerometers: bridging between closed-pose anchors, with the own "
                  "model as pseudo-measurements in the last step")],
        n=f"n = {n} participants with jaw IMU" if n else "no camera-free results",
        statistic="median with 95 % BCa CI; verdict: within (upper CI below the bound), "
                  "compatible (median below), exceeds (lower CI above)",
        test="exact Wilcoxon signed-rank, Hodges-Lehmann (HL) estimate with exact CI, Holm "
             "within family",
        evidence=EVIDENCE,
        notes=["The movement-class model choice was not computed"]))


# -- oracle decomposition and task-conditioned model ------------------------------------------
def _oracle_outcomes() -> list[tuple[str, str]]:
    """(outcome, reader label) of the oracle summaries: each movement task, then all."""
    return [(f"rmse_inc_mm__{t}", V.task_label(t)) for t in ORACLE_TASKS] + \
        [("rmse_inc_mm", "all scored tasks")]


DECOMP_FORMS = ("medians", "dominance")


@register("cf_decomposition", "dashboard", "Camera-free incisor error: rotation vs model",
          choices={"form": DECOMP_FORMS}, form="medians", **DS)
def cf_decomposition(req: ViewRequest) -> FigureSpec:
    """Incisor RMSE per task of the oracle combinations next to E4own: optical rotation + own
    model (translation-model error alone), camera-free rotation + optical translation
    (rotation-induced error alone), the task-conditioned model (oracle class) and the full
    E4own; participants and cohort median with 95 % BCa CI, the 2 mm bound for orientation.
    Form ``dominance`` (J2.1): per task one panel, each participant's
    rotation-induced error against its translation-model error with the identity line --
    which half dominates, participant by participant."""
    fid, title = "cf_decomposition", "Camera-free incisor error: rotation vs model"
    part = _table(req.ctx, "camerafree_cohort", "cf_oracle_participant")
    c = _table(req.ctx, "camerafree_cohort", "cf_oracle_cohort")
    if part.empty or "outcome" not in part:
        return _empty(fid, title, "no oracle decomposition results")
    outs = _oracle_outcomes()
    srcs = ["camerafree_cohort:cf_oracle_participant", "camerafree_cohort:cf_oracle_cohort"]
    if (req.options.get("form") or "medians") == "dominance":
        panels = []
        hi_all = 0.0
        for i, (oc, lab) in enumerate(outs):
            def vals(e: str) -> pd.Series:
                g = part[(part["estimator"] == e) & (part["outcome"] == oc)]
                return g.groupby("pid")["value"].median()
            rot, mod = vals("X5b_rotation"), vals("X5b_model")
            both = pd.concat([rot.rename("rot"), mod.rename("mod")], axis=1).dropna()
            q = Panel(f"({'abcde'[i]}) {lab}", Axis("Rotation-induced", V.MM),
                      Axis("Translation model", V.MM), [], legend="none", equal=True)
            if not both.empty:
                hi_all = max(hi_all, float(both.to_numpy().max()))
                q.series.append(Series("scatter", both["rot"].to_numpy(float),
                                       both["mod"].to_numpy(float), role="primary",
                                       text=[f"P{req.ctx.pseudonym(x)}" for x in both.index]))
            panels.append(q)
        lim = (0.0, max(1.0, hi_all * 1.08))
        for q in panels:
            q.x.lim = q.y.lim = lim
            q.series.insert(0, Series("line", np.array(lim), np.array(lim), role="reference"))
        spec = FigureSpec(fid, title, panels, rows=2, cols=3, width="text", height_in=3.6,
                          sources=srcs)
        return attach(spec, CaptionSpec(
            "which half of the camera-free incisor estimate dominates its error, participant "
            "by participant: the incisor RMSE against the optics with the anchored IMU "
            "rotation and the optical condyle translation (rotation-induced error alone, x) "
            "against the RMSE with the optical rotation and the own translation model "
            "(translation-model error alone, y), per task and over all scored tasks; above "
            "the diagonal the translation model dominates, below it the rotation",
            [("dots", "participants"), ("diagonal", "equal error of both halves")],
            f"n = {_n_pid(part)} participants with jaw IMU",
            "per participant the task RMSE (all scored tasks: the median over the tasks)",
            evidence=ORACLE_EVIDENCE, codes=["X5b"]))
    ests = [e for e in ORACLE if (part["estimator"] == e).any()]
    # 2026-11-15 (ICORR figure round): tasks as rows with the error on the horizontal axis
    # (task names read normally at column width), one marker shape and colour per
    # combination, the names of the results table in the legend, a rule before the row of
    # all scored tasks. Log axis: a few participants reach 28 mm.
    roles = ("primary", "secondary", "tertiary", "quaternary")
    shapes = ("o", "s", "^", "d")
    p = Panel("", Axis("Incisor RMSE", V.MM, log=True, lim=INC_XLIM, ticks=INC_TICKS),
              Axis(""), [], legend="below")
    width = 0.72 / max(1, len(ests))
    for k, e in enumerate(ests):
        off = (k - 0.5 * (len(ests) - 1)) * width
        lab = ORACLE_HEADER.get(e, V.term("estimator", e))
        lab = lab if lab.startswith("IMU") else lab[0].lower() + lab[1:]
        for i, (oc, _) in enumerate(outs):
            v = part[(part["estimator"] == e) & (part["outcome"] == oc)]["value"]
            v = v.to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                p.series.append(Series("scatter", np.clip(v, INC_XLIM[0], INC_XLIM[1] * 0.97),
                                       np.full(v.size, i + off), role="faint",
                                       style={"size": 0.8}))
            row = _cohort_row(c, e, oc) or {}
            lo_, hi_ = row.get("ci_lo", np.nan), row.get("ci_hi", np.nan)
            if np.isfinite(lo_) and np.isfinite(hi_):
                p.series.append(Series("line", np.array([lo_, hi_]), np.array([i + off] * 2),
                                       role=roles[k % 4], style={"width": 1.6}))
            p.series.append(Series("scatter", np.array([row.get("median", np.nan)]),
                                   np.array([i + off]), role=roles[k % 4],
                                   style={"ms": shapes[k % 4],
                                          "size": (1.5, 1.5, 2.1, 2.1)[k % 4]},
                                   label=lab if i == 0 else None))
    p.series.append(Series("vline", x=np.array([INC_BOUND]), role="threshold",
                           label="2 mm bound"))
    p.series.append(Series("hline", y=np.array([len(outs) - 1.5]), role="reference"))
    p.y.ticks = [float(i) for i in range(len(outs))]
    p.y.ticklabels, p.y.reverse = [lab for _, lab in outs], True
    p.y.lim = (-0.55, len(outs) - 0.45)
    spec = FigureSpec(fid, title, [p], width="column", height_in=2.9, sources=srcs)
    return attach(spec, CaptionSpec(
        "where the camera-free incisor error comes from: the incisor RMSE against the optics "
        "per task when the optics replace one half of the own-model estimate -- optical "
        "rotation with the own translation model driven by it (translation-model error "
        "alone) and the anchored IMU rotation with the optical condyle translation "
        "(rotation-induced error alone) -- next to the task-conditioned model (the own "
        "model refitted per movement class, applied with the true task: the best a movement "
        "classifier could do) and the full own-model estimate; the errors do not add, the "
        "full estimate also contains the model's response to the rotation error",
        [("small grey dots", "participants"), ("markers", "cohort median"),
         ("x axis", "logarithmic"),
         ("bars", "95 % BCa CI"), ("dashed line", "2 mm, for orientation"),
         ("thin line", "separates the row of all scored tasks")],
        f"n = {_n_pid(part)} participants with jaw IMU",
        "per participant the task RMSE (all scored tasks: the median over the tasks)",
        evidence=ORACLE_EVIDENCE, codes=["X5b", "X5c"]))


GROUP_OUTCOMES = (("rmse_inc_mm_rot", "open-close and chewing"),
                  ("rmse_inc_mm_trans", "protrusion and laterotrusion"))


def _hl_p(k: pd.DataFrame, family: str, outcome: str) -> str:
    """'HL [CI], p' of one oracle contrast, '--' when absent."""
    if k.empty or "family" not in k:
        return "--"
    r = k[(k["family"] == family) & (k["outcome"] == outcome)]
    if r.empty or not np.isfinite(r.iloc[0].get("hl", np.nan)):
        return "--"
    r = r.iloc[0]
    pv = r.get("p_holm", np.nan)
    p = ("" if not np.isfinite(pv) else
         f", p\u00a0{B.pval(pv)}" if pv < 0.001 else f", p\u00a0=\u00a0{B.pval(pv)}")
    return B.est_ci(r["hl"], r.get("hl_ci_lo", np.nan), r.get("hl_ci_hi", np.nan)) + p


@register("cf_table_oracle", "dashboard", "Camera-free incisor error decomposition (table)",
          choices={"rows": ("all", "groups")}, rows="all", **DS)
def cf_table_oracle(req: ViewRequest) -> TableSpec:
    """Per task and task group: cohort median (95 % BCa CI) of the incisor RMSE of the oracle
    combinations and E4own, with the paired contrasts (rotation-induced − model error; task
    model − own model): Hodges–Lehmann with exact CI and Holm-adjusted p."""
    ctx = req.ctx
    c = _table(ctx, "camerafree_cohort", "cf_oracle_cohort")
    k = _table(ctx, "camerafree_cohort", "cf_oracle_contrasts")
    outs = _oracle_outcomes()
    outs = outs[:-1] + list(GROUP_OUTCOMES) + outs[-1:]
    if req.options.get("rows") == "groups":   # the two task groups only (ICORR, 2026-11-12)
        outs = list(GROUP_OUTCOMES)
    rows, n = [], 0
    for oc, lab in outs:
        row: dict[str, Any] = {"task": lab}
        for e in ORACLE:
            r = _cohort_row(c, e, oc)
            row[e] = _ci(r)
            n = max(n, int((r or {}).get("n", 0) or 0))
        row["c_rot"], row["c_task"] = _hl_p(k, "X5b", oc), _hl_p(k, "X5c", oc)
        rows.append(row)
    cols = [Column("task", "Tasks", align="X")] + [
        Column(e, ORACLE_HEADER.get(e, rung(e)), V.MM, align="X", group="Median [95 % CI]")
        for e in ORACLE] + [
        Column("c_rot", "Rotation − model", V.MM, align="X", group="HL [95 % CI], p (Holm)"),
        Column("c_task", "Task model − own model", V.MM, align="X",
               group="HL [95 % CI], p (Holm)")]
    t = TableSpec("cf_table_oracle", "", cols, rows, width="text",
                  sources=["camerafree_cohort:cf_oracle_cohort",
                           "camerafree_cohort:cf_oracle_contrasts"])
    return attach(t, CaptionSpec(
        "incisor RMSE against the optics per task and task group with the optics replacing "
        "one half of the own-model estimate (optical rotation + own model: translation-model "
        "error; IMU rotation + optical translation: rotation-induced error), the "
        "task-conditioned model (oracle class) and the full own-model estimate, with the "
        "paired contrasts rotation-induced minus model error and task-conditioned minus own "
        "model (negative: the first is smaller)",
        n=f"n = {n} participants with jaw IMU" if n else "no oracle results",
        statistic="median with 95 % BCa CI of the participant values (descriptive, no bound); "
                  "Hodges-Lehmann (HL) estimate of the paired difference with exact 95 % CI",
        test="exact Wilcoxon signed-rank, Holm within each analysis",
        evidence=ORACLE_EVIDENCE, codes=["X5b", "X5c"]))


@register("cf_oracle_tasks", "dashboard", "Camera-free incisor error per participant and task",
          **DS, choices={"estimator": ORACLE})
def cf_oracle_tasks(req: ViewRequest) -> FigureSpec:
    """The per-participant rows behind the oracle summaries: incisor RMSE per participant
    and movement task of one oracle combination (or E4own), with the class model's
    calibration samples for the task-conditioned model."""
    ctx = req.ctx
    est = req.options.get("estimator") or ORACLE[0]
    fid, title = "cf_oracle_tasks", "Camera-free incisor error per participant and task"
    rows = pd.concat([_table(ctx, "camerafree", n) for n in ("cf_decomposition",
                                                             "cf_task_model")],
                     ignore_index=True)
    d = rows[rows["estimator"] == est] if "estimator" in rows else rows.iloc[:0]
    if d.empty or "rmse_inc_mm" not in d:
        return _empty(fid, title, "no oracle rows per participant")
    pids = [p for p in ctx.pids() if (d["pid"] == p).any()]
    name = "full own-model estimate" if est == "E4own" else V.term("estimator", est)
    p = B.matrix_heat(d, "pid", "task", "rmse_inc_mm", name,
                      rows=pids, cols=list(ORACLE_TASKS), row_labels=B.pseudo_rows(ctx, pids),
                      fmt="{:.1f}", cb=Colorbar("Incisor RMSE", V.MM))
    notes = []
    if est == "X5c" and "n_cal_samples" in d:
        cal = d[d["task"].isin(ORACLE_TASKS)].dropna(subset=["n_cal_samples"]) \
            .groupby("task")["n_cal_samples"].median()
        notes.append("Median calibration samples of the class models: " + "; ".join(
            f"{V.task_label(t)} {int(v)}" for t, v in cal.items()))
    if "note" in d:
        miss = d.loc[d["note"].fillna("").astype(str) != "", "note"]
        if len(miss):
            notes.append(f"{len(miss)} participant-task rows without a value (the reason is "
                         "in the complete table)")
    spec = FigureSpec(fid, title, [p], width="text", height_in=B.rows_height(len(pids)),
                      sources=["camerafree:cf_decomposition", "camerafree:cf_task_model"])
    return attach(spec, CaptionSpec(
        f"incisor RMSE against the optics per participant and movement task of the "
        f"{name} (the per-participant values behind the cohort "
        "decomposition), after the ground truth's low-pass",
        [("cells", "participant x task RMSE; empty: no value")],
        f"n = {len(pids)} participants with jaw IMU", evidence=ORACLE_EVIDENCE,
        notes=notes, codes=["X5b", "X5c"]))


# -- validation (VD20) ------------------------------------------------------------------------
@register("val_camerafree", "dashboard", "Camera-free estimators vs Qualisys",
          choices={"form": ("offset_dots", "ladder", "slope")}, form="offset_dots", **VAL)
def val_camerafree(req: ViewRequest) -> FigureSpec:
    """VD20: per participant the opening-angle and incisor RMSE of each camera-free rung
    against Qualisys next to the optics (F0), with the non-inferiority margins (tick labels
    and caption name the rungs and the optics, not their codes). Forms ``ladder`` /
    ``slope``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "offset_dots"))
    if form != "offset_dots":
        from biomech.viz.views.validation_r8 import camerafree_form

        return camerafree_form(req, form)
    fid, title = "val_camerafree", "Camera-free estimators vs Qualisys"
    d = _table(req.ctx, "ref_camerafree", "cf_val_segments")
    ni = _table(req.ctx, "ref_camerafree", "cf_val_noninferiority")
    if d.empty or "estimator" not in d:
        return _empty(fid, title, "no ref_camerafree results")
    d = d[d["segment"] == "movement_pooled"]
    ests_all = [e for e in ("F0", "E1", "E2a", "E2b", "E3a", "E3b", "E4hinge", "E4own", "E5",
                            "E6") if (d["estimator"] == e).any()]
    panels = []
    # each panel lists only the rungs that estimate its outcome (the rotation rungs have no
    # incisor); a rung far off the others (the accelerometer double integration, metres) is
    # an arrow with its value at the axis end, not a log axis to 10^4
    for col, lab, unit, sym, margin in (("rmse_open_deg", "(a) Opening angle", V.DEG, "°", 0.5),
                                        ("rmse_inc_mm", "(b) Incisor", V.MM, "\u00a0mm", 1.0)):
        if col not in d:
            continue
        has = d.groupby("estimator")[col].apply(lambda v: v.notna().any())
        ests = [e for e in ests_all if bool(has.get(e, False))]
        y = np.arange(len(ests), dtype=float)
        vals = d[d["estimator"].isin(ests)][col].to_numpy(float)
        near = d[d["estimator"].isin([e for e in ests if e not in NO_MEAN_RUNGS])][col]
        hi = 1.3 * float(np.nanmax(np.r_[near.to_numpy(float), margin])) if len(near) else None
        p = Panel(lab, Axis("RMSE vs Qualisys", unit,
                            lim=(0.0, hi) if hi and np.nanmax(vals) > hi else None),
                  Axis(""), [], legend="below")
        for i, pid in enumerate(sorted(d["pid"].unique())):
            g = d[d["pid"] == pid].set_index("estimator")
            x = np.array([float(g.loc[e, col]) if e in g.index and col in g else np.nan
                          for e in ests])
            yy = y + (i - 0.5) * 0.2
            off = np.isfinite(x) & (x > hi) if hi else np.zeros(len(x), bool)
            p.series.append(Series("scatter", np.where(off, np.nan, x), yy, role="accent",
                                   style={"ms": "os"[i % 2], "size": 1.2},
                                   label=_who(req.ctx, pid)))
            for k in np.flatnonzero(off):
                p.series.append(Series("text", x=np.array([hi]), y=np.array([yy[k]]),
                                       text=[f"{_who(req.ctx, pid)} {B.num(x[k], 0)}{sym} →"],
                                       style={"anchor": "left"}))
        f0 = d[d["estimator"] == "F0"][col]
        if f0.notna().any():
            p.series.append(Series("vline", x=np.array([float(f0.mean()) + margin]),
                                   role="threshold", label=f"optics + {margin:g}{sym}"))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), [rung(e) for e in ests], True
        p.y.lim = (-0.6, len(ests) - 0.4)
        panels.append(p)
    panels[0].legend = "none"
    n_ok = int(ni["noninferior"].astype(bool).sum()) if "noninferior" in ni else 0
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.4,
                      sources=["ref_camerafree:cf_val_segments",
                               "ref_camerafree:cf_val_noninferiority"])
    return attach(spec, CaptionSpec(
        "the camera-free ladder on the comparison takes against Qualisys (frames aligned as "
        "for the IMU alone, every estimate and Qualisys through the dataset's 6 Hz "
        "low-pass), pooled over the movement tasks, next to the JTS-MM optics; the margin "
        "line is the optics' mean plus the non-inferiority margin (0.5°, 1.0 mm). Rungs: "
        + "; ".join(f"{rung(e)}: {V.term('estimator', e)}" for e in ests_all if e != "F0"),
        [("markers", "participants (circle / square, legend)"),
         ("dashed line", "optics mean + margin"),
         ("value with arrow", "a value beyond the axis end")],
        f"n = {_n_pid(d)} participants ("
        + ", ".join(sorted(_who(req.ctx, p) for p in d["pid"].unique())) + "; descriptive)",
        statistic="per participant and rung the RMSE over all movement-task frames (tasks "
                  "pooled within the participant); dashed line: the mean of the participants' "
                  "optics RMSE plus the margin",
        test=f"non-inferior estimator-outcome pairs: {n_ok} of {len(ni)} "
             "(upper block-bootstrap 95 % CI of the paired difference below the margin)",
        evidence="exploratory"))


# Rungs whose two-participant mean says nothing: the accelerometer double integration
# diverges by orders of magnitude between participants (12 m vs 0.2 m on the comparison takes).
NO_MEAN_RUNGS = ("E5",)


def _sig3(v: float) -> float | str:
    """At least 3 significant digits without spurious decimals: values of 100 and more as
    whole numbers (a 12-m error is not 12 522.55 mm), others as stored (column decimals)."""
    return f"{v:.0f}" if np.isfinite(v) and abs(v) >= 100 else v


@register("val_table_camerafree", "dashboard", "Camera-free vs Qualisys: non-inferiority (table)",
          **VAL)
def val_table_camerafree(req: ViewRequest) -> TableSpec:
    """VD20: per participant and rung the paired difference to F0 with its CI and the
    non-inferiority decision, and the two-participant mean (descriptive). The drift test is
    ``val_table_camerafree_drift`` (one table per question)."""
    ctx = req.ctx
    ni = _table(ctx, "ref_camerafree", "cf_val_noninferiority")
    rows = []
    skipped = []
    for (e, o), g in (ni.groupby(["estimator", "outcome"], sort=False) if not ni.empty else []):
        for r in g.to_dict("records"):
            rows.append({"pid": _who(ctx, r["pid"]), "estimator": rung(e),
                         "outcome": NI_OUTCOME.get(o, o),
                         "mean_diff": _sig3(r.get("mean_diff", np.nan)),
                         "ci_lo": _sig3(r.get("ci_lo", np.nan)),
                         "ci_hi": _sig3(r.get("ci_hi", np.nan)),
                         "margin": r.get("margin", np.nan),
                         "noninferior": "yes" if r.get("noninferior") else "no"})
        if g["pid"].nunique() > 1:
            if e in NO_MEAN_RUNGS:
                skipped.append(rung(e))
                continue
            rows.append({"pid": f"mean of {g['pid'].nunique()}", "estimator": rung(e),
                         "outcome": rows[-1]["outcome"],
                         "mean_diff": _sig3(float(g["mean_diff"].mean())),
                         "ci_lo": np.nan, "ci_hi": np.nan, "margin": rows[-1]["margin"],
                         "noninferior": ""})
    cols = [Column("pid", "Participant", align="l"), Column("estimator", "Rung", align="l"),
            Column("outcome", "Outcome", align="l"),
            Column("mean_diff", "Rung − optics"), Column("ci_lo", "CI low"),
            Column("ci_hi", "CI high"), Column("margin", "Margin"),
            Column("noninferior", "Non-inferior", align="l")]
    t = TableSpec("val_table_camerafree", "", cols, rows, width="text",
                  sources=["ref_camerafree:cf_val_noninferiority"])
    notes = [f"No mean for the {', '.join(skipped)} rung: its two values differ by orders of "
             "magnitude (double integration of the accelerations diverges)."] if skipped else []
    n = _n_pid(ni)
    return attach(t, CaptionSpec(
        "camera-free rungs against Qualisys: per frame the absolute difference to Qualisys, "
        "paired against the JTS-MM optics and pooled over the movement tasks (rung minus optics, "
        "in the outcome's unit; negative = the rung is closer to Qualisys)",
        n=f"n = {n} participants; the mean of the two is descriptive (no CI)",
        statistic="mean difference with a 1-s block-bootstrap 95 % CI",
        test="non-inferior: upper CI below the margin (0.5° opening, 1.0 mm incisor)",
        evidence="exploratory", notes=notes))


@register("val_table_camerafree_drift", "dashboard", "Camera-free vs Qualisys: drift (table)",
          **VAL)
def val_table_camerafree_drift(req: ViewRequest) -> TableSpec:
    """VD20 drift: per participant the signed opening-error slope of the mountings alone
    against time since calibration and of the closed-pose resets against time since the last
    anchor, with its CI, and the two-participant mean (descriptive)."""
    ctx = req.ctx
    dr = _table(ctx, "ref_camerafree", "cf_val_drift")
    rows = []
    for (e, c), g in (dr.groupby(["estimator", "clock"], sort=False) if not dr.empty else []):
        since = {"calibration": "since calibration", "anchor": "since the last anchor"}.get(c, c)
        for r in g.to_dict("records"):
            rows.append({"pid": _who(ctx, r["pid"]), "estimator": rung(e), "since": since,
                         "slope": B.est_ci(r.get("slope_dpm", np.nan), r.get("ci_lo", np.nan),
                                           r.get("ci_hi", np.nan))})
        if g["pid"].nunique() > 1:
            rows.append({"pid": f"mean of {g['pid'].nunique()}",
                         "estimator": rung(e), "since": since,
                         "slope": B.num(float(g["slope_dpm"].mean()), 2, adaptive=True)})
    # one value cell (slope [CI]) and a wrapping rung column: six columns overflowed the
    # column width by 89 pt
    cols = [Column("pid", "Participant", align="l"),
            Column("estimator", "Rung", align="X"),
            Column("since", "Time", align="X"),
            Column("slope", "Slope [95 % CI]", unit="°/min", align="l")]
    t = TableSpec("val_table_camerafree_drift", "", cols, rows, width="column",
                  sources=["ref_camerafree:cf_val_drift"])
    return attach(t, CaptionSpec(
        "drift of the camera-free opening angle against Qualisys: the signed slope of the "
        "opening difference of the mountings alone against the time since calibration, and of the "
        "closed-pose resets against the time since the last anchor (0 = no drift)",
        n=f"n = {_n_pid(dr)} participants; the mean of the two is descriptive (no CI)",
        statistic="least-squares slope with a 1-s block-bootstrap 95 % CI",
        evidence="exploratory"))
