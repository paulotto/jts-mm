# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result views of quality control, IMU calibration/cycles, labels and validation.

The per-participant QC tables (``qc.sessions|phases|imu|optical|emg``) each get one heatmap
view, participant x entity (take, phase, IMU stream, tracked body, muscle), with a picker
over the table's numeric columns. Cohort QC tables are shown as tables (demographics only
as the stored aggregate row). Participants appear as pseudonyms only; session folder
names (which carry study ids, sex and age) are never shown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.results_kinematics import _pid, _rel_s, numeric_columns, who

# table -> (entity column(s), entity label, default column, title)
QC_TABLES = {
    "emg": (("muscle",), "muscle", "snr_db", "EMG channel quality"),
    "emg_check": (("take", "muscle"), "take and muscle", "z_kohm", "Pre-take EMG check"),
    "imu": (("slot", "stream"), "IMU stream", "coverage", "IMU stream quality"),
    "optical": (("phase",), "phase", "lock_frac", "Optical tracking lock"),
    "phases": (("take", "phase"), "phase", "phase_s", "Phase durations"),
    "sessions": (("take",), "take", "duration_s", "Session takes"),
}
SKIP = ("pid", "unit", "t_start_us", "t_end_us", "line_hz")


def _qc_columns(table: str):
    def f(ctx: Any) -> list[str]:
        d = ctx.table("qc", table)
        return [c for c in numeric_columns(d, SKIP) if d[c].notna().any()]
    return f


def _fmt(v: np.ndarray) -> str:
    v = v[np.isfinite(v)]
    if not v.size:
        return "{:.2f}"
    if np.allclose(v, np.round(v)):
        return "{:.0f}"
    m = np.nanmax(np.abs(v))
    return "{:.3f}" if m < 1 else "{:.2f}" if m < 10 else "{:.1f}" if m < 1000 else "{:.0f}"


def _qc_heat(table: str):
    ent, ent_label, default, title = QC_TABLES[table]

    def build(req: ViewRequest) -> FigureSpec:
        ctx = req.ctx
        d = ctx.table("qc", table)
        col = req.options.get("column") or default
        cols_ok = _qc_columns(table)(ctx)
        if col not in cols_ok and cols_ok:
            col = cols_ok[0]
        pids = ctx.pids()
        panels = []
        groups = ([(b, d[d.body == b]) for b in sorted(set(d.body))]
                  if table == "optical" and not d.empty else [("", d)])
        for g_name, g in groups:
            if g.empty or col not in g:
                panels.append(Panel(g_name or title, Axis(""), Axis(""), [], legend="none",
                                    note="no data"))
                continue
            g = g.assign(entity=["/".join(str(r[c]) for c in ent) for _, r in g.iterrows()],
                         value=pd.to_numeric(g[col].astype(float), errors="coerce"))
            order = (list(dict.fromkeys(g.entity)) if table not in ("emg", "emg_check")
                     else [m for m in V.MUSCLES if m in set(g.entity)] if table == "emg"
                     else [f"{t}/{m}" for t in ("motion", "mvc") for m in V.MUSCLES
                           if f"{t}/{m}" in set(g.entity)])
            labels = [V.muscle_label(e) if table == "emg" else
                      f"{V.term('take', e.split('/')[0], 'short')} "
                      f"{V.muscle_label(e.split('/')[1])}" if table == "emg_check" else
                      e.replace("motion/", "").replace("mvc/", "MVC ").replace("_", " ")
                      for e in order]
            z = g.pivot_table(index="pid", columns="entity", values="value", aggfunc="median")
            z = z.reindex(index=pids, columns=order).to_numpy(float)
            cb = V.colorbar(col) if V.known(col) else None
            panels.append(B.heat(z, B.pseudo_rows(ctx, pids), labels,
                                 f"Body {g_name}" if g_name else V.label(col)
                                 if V.known(col) else col.replace("_", " "),
                                 fmt=_fmt(z.ravel()), cb=cb))
        n_ent = max((len(p.x.ticks or []) for p in panels), default=1)
        return FigureSpec(f"qc_heatmap_{table}", title, panels, rows=len(panels), cols=1,
                          width="text" if n_ent > 6 else "column",
                          height_in=max(1.3, 0.22 * len(pids) + 0.6),
                          caption=f"{title}: {V.label(col) if V.known(col) else col} per "
                                  f"participant (rows) and {ent_label} (columns), from "
                                  "the quality-control results" + (", one panel per tracked body" if
                                                    table == "optical" else "")
                                  + (" (1 = yes, 0 = no)" if col in d and d[col].dtype == bool
                                     else "")
                                  + ".", sources=[f"qc:{table}"])

    build.__doc__ = f"Heatmap of one numeric column of qc.{table}, participant x {ent_label}."
    return build


for _t, (_e, _l, _d, _title) in QC_TABLES.items():
    register(f"qc_heatmap_{_t}", "dashboard", _title, domain="qc",
             choices={"column": _qc_columns(_t)}, column=_d)(_qc_heat(_t))


def _table(req: ViewRequest, stage: str, name: str, view_id: str, caption: str,
           **kw: Any) -> TableSpec:
    t = B.frame_table(req.ctx.table(stage, name), view_id, caption, ctx=req.ctx, **kw)
    t.sources = [f"{stage}:{name}"]
    return t


@register("qc_overview", "dashboard", "QC overview", domain="qc")
def qc_overview(req: ViewRequest) -> TableSpec:
    return _table(req, "qc_cohort", "overview", "qc_overview",
                  "Per participant: EMG channels unusable for amplitude or noise-limited at "
                  "rest, mean masked task share, optical lock and quaternion coverage per "
                  "body/IMU in the motion phases", width="text")


@register("qc_emg_flags", "dashboard", "EMG flags", domain="qc")
def qc_emg_flags(req: ViewRequest) -> TableSpec:
    f = req.ctx.table("qc_cohort", "emg_flags")
    if not f.empty:
        f = f.assign(muscle=[V.muscle_label(m) for m in f.muscle])
    t = B.frame_table(f, "qc_emg_flags", "EMG channels with a usability flag or more than 1 % "
                      "masked task samples (flagged, not excluded)", ctx=req.ctx, width="text")
    t.sources = ["qc_cohort:emg_flags"]
    if f.empty:
        t.notes = "No channel flagged."
    return t


@register("qc_demographics", "dashboard", "Cohort demographics", domain="qc")
def qc_demographics(req: ViewRequest) -> TableSpec:
    """The stored aggregate demographics row (never per-person sex or age)."""
    return _table(req, "qc_cohort", "demographics", "qc_demographics",
                  "Included participants: number, sex counts and age (aggregate only)")


@register("qc_validation", "dashboard", "Validation status", domain="qc")
def qc_validation(req: ViewRequest) -> TableSpec:
    ctx = req.ctx
    s = ctx.table("validation", "status")
    if "pids" in s:                                  # study ids -> pseudonyms
        s = s.assign(pids=[", ".join(who(ctx, p) for p in str(v).split(",") if p)
                           for v in s.pids.fillna("")])
    t = B.frame_table(s, "qc_validation", "Reference-system validation (Qualisys/Delsys "
                      "subset) or the backup G' that applies", width="text", ctx=ctx)
    t.sources = ["validation:status"]
    return t


# == IMU ===========================================================================================
@register("imu_calibration", "dashboard", "IMU calibration", domain="imu")
def imu_calibration(req: ViewRequest) -> FigureSpec:
    """Hand-eye calibration residual, heading offset and quaternion drift per participant."""
    from biomech.viz.views.results_kinematics import PP_CAPTION, per_participant

    ctx = req.ctx
    c = ctx.table("imu", "calibration")
    panels = [per_participant(ctx, c, "rms_deg", "Calibration residual",
                              xaxis=Axis("RMS", V.DEG)),
              per_participant(ctx, c, "psi_deg", "Heading offset", xaxis=Axis("ψ", V.DEG)),
              per_participant(ctx, c, "drift_quat_deg_per_min", "Quaternion drift",
                              xaxis=Axis("Drift per minute", V.DEG), ref=0.0)]
    ph = V.task_label(str(c.phase.iloc[0])) if not c.empty and "phase" in c else "calibration"
    return FigureSpec("imu_calibration", "IMU calibration", panels, rows=1, cols=3,
                      width="text", height_in=max(1.4, 0.17 * len(ctx.pids())),
                      caption=f"IMU-to-optical calibration on the {ph} phase: RMS rotation "
                              "residual of the fitted mountings, heading offset ψ between the "
                              "two IMU worlds, and the drift of the quaternion opening angle "
                              f"against optical over the session, per participant {PP_CAPTION}.",
                      sources=["imu:calibration"])


ESTIMATORS = (("quat", "quaternion", "cat:0"), ("comp", "complementary", "cat:1"),
              ("tilt", "tilt only", "cat:2"))


@register("imu_cycles", "dashboard", "IMU peak opening per cycle", domain="imu",
          choices={"task": lambda ctx: [t for t in V.TASKS if t in set(
              ctx.table("imu", "cycles").get("task", []))]}, task="open_close")
def imu_cycles(req: ViewRequest) -> FigureSpec:
    """Per-cycle peak opening angle and opening from the IMU estimators vs optical."""
    c = req.ctx.table("imu", "cycles")
    task = req.options.get("task") or "open_close"
    c = c[c.task == task] if not c.empty else c
    pa = Panel("Peak opening angle", Axis("Optical", V.DEG), Axis("IMU", V.DEG), [])
    pb = Panel("Peak opening (subject model)", Axis("Optical", V.MM), Axis("IMU", V.MM), [])
    for p, ref, fmt in ((pa, "theta_peak_opt_deg", "theta_peak_{}_deg"),
                        (pb, "opening_opt_mm", "opening_subject_{}_mm")):
        if c.empty or ref not in c:
            p.note = "no data"
            continue
        x = c[ref].to_numpy(float)
        lim = [np.nanmin(x), np.nanmax(x)] if np.isfinite(x).any() else [0.0, 1.0]
        p.series.append(Series("line", np.array(lim), np.array(lim), role="reference",
                               label="identity"))
        for e, lab, role in ESTIMATORS:
            if fmt.format(e) in c:
                p.series.append(Series("scatter", x, c[fmt.format(e)].to_numpy(float),
                                       role=role, label=lab))
    if not c.empty:
        rm = {e: np.nanmedian(c[f"rmse_cycle_{e}_deg"]) for e, _, _ in ESTIMATORS
              if f"rmse_cycle_{e}_deg" in c}
        pa.note = "median RMSE " + ", ".join(f"{e} {v:.2f}°" for e, v in rm.items())
    return FigureSpec("imu_cycles", "IMU peak opening per cycle", [pa, pb], rows=1, cols=2,
                      width="text", height_in=2.0,
                      caption=f"{V.task_label(task)}: peak opening angle per good cycle from "
                              "the three IMU estimators against optical (tilt only while both "
                              "IMUs are quasi-static), and the opening the subject coupling "
                              "model gives from it; all participants pooled.",
                      sources=["imu:cycles"])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[0], np.asarray(mask, bool).astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


@register("imu_quasi_static", "dashboard", "IMU quasi-static spans", params=("pid", "phase"),
          phase="open_close", phases=V.TASKS, domain="imu")
def imu_quasi_static(req: ViewRequest) -> FigureSpec:
    """Where the tilt estimate is valid: quasi-static spans over the opening angle."""
    ctx, pid, task = req.ctx, _pid(req), req.phase or "open_close"
    a = ctx.arrays("imu", pid) if pid else {}
    t, qs = a.get(f"{task}__t_us"), a.get(f"{task}__quasi_static")
    p = Panel("Opening angle and quasi-static spans", Axis("Time in phase", V.SEC),
              Axis("Opening angle", V.DEG), [])
    if t is None or qs is None:
        p.note = "no IMU data"
    else:
        ts = _rel_s(t)
        for i, j in _runs(qs):
            p.series.append(Series("hspan", x=np.array([ts[i]]), y2=np.array([ts[j]]),
                                   role="faint"))
        for k, lab, role in (("theta_opt", "optical", "context"),
                             ("theta_tilt", "IMU tilt", "cat:2"),
                             ("theta_quat", "IMU quaternion", "accent")):
            if a.get(f"{task}__{k}") is not None:
                p.series.append(Series("line", ts, np.asarray(a[f"{task}__{k}"], float),
                                       role=role, label=lab))
        p.note = f"quasi-static {100 * np.mean(np.asarray(qs, bool)):.0f} % of the phase"
    return FigureSpec("imu_quasi_static", "IMU quasi-static spans", [p], width="text",
                      height_in=1.8,
                      caption=f"{who(ctx, pid) if pid else ''}, {V.task_label(task)}: opening "
                              "angle from optical and the IMU estimators; shaded: samples "
                              "where both IMUs are quasi-static (accelerometer norm within "
                              "tolerance of g), the only samples the tilt estimate and the "
                              "complementary filter's correction use.",
                      sources=["imu:quasi_static"])


# == labels ========================================================================================
@register("labels_timeline", "dashboard", "Intent label timeline", params=("pid", "phase"),
          phase="open_close", phases=V.TASKS, domain="ml")
def labels_timeline(req: ViewRequest) -> FigureSpec:
    """Auto labels (C3b ground truth) over time with the opening angle above."""
    from biomech.labels.auto_labels import CLASSES

    ctx, pid, task = req.ctx, _pid(req), req.phase or "open_close"
    la = ctx.arrays("labels", pid) if pid else {}
    ka = ctx.arrays("kinematics", pid) if pid else {}
    xa = Axis("Time in phase", V.SEC)
    pa = Panel("Opening angle", xa, Axis("Angle", V.DEG), [], legend="none", sharex="t")
    pb = Panel("Label", xa, Axis(""), [], legend="none", sharex="t")
    t, lab = la.get(f"{task}__t_us"), la.get(f"{task}__label")
    t0 = float(np.asarray(t)[0]) if t is not None and len(t) else 0.0
    kt, th = ka.get(f"{task}__t_us"), ka.get(f"{task}__theta_deg")
    if kt is not None and th is not None:
        pa.series.append(Series("line", (np.asarray(kt, float) - t0) * 1e-6,
                                np.asarray(th, float), role="primary"))
    else:
        pa.note = "no data"
    names = list(CLASSES) + ["UNTRACKED"]
    if t is not None and lab is not None:
        ts = (np.asarray(t, float) - t0) * 1e-6
        lab = np.asarray(lab, int)
        for k, name in enumerate(names):
            code = k if name != "UNTRACKED" else -1
            y = np.where(lab == code if code >= 0 else lab < 0, float(k), np.nan)
            if np.isfinite(y).any():
                pb.series.append(Series("line", ts, y, role="noise" if code < 0 else
                                        f"cat:{k % 7}", style={"width": 3.0}))
    else:
        pb.note = "no labels"
    pb.y.ticks, pb.y.ticklabels = [float(i) for i in range(len(names))], [
        n.lower() for n in names]
    pb.y.reverse = True
    c = ctx.table("labels", "counts")
    c = c[(c.pid == pid) & (c.phase == task)] if not c.empty else c
    if not c.empty:
        sec = c.set_index("label").seconds
        pb.note = ", ".join(f"{n.lower()} {sec[n]:.1f} s" for n in names
                            if n in sec and sec[n] > 0)
    return FigureSpec("labels_timeline", "Intent label timeline", [pa, pb], rows=2, cols=1,
                      width="text", height_in=1.6,
                      caption=f"{who(ctx, pid) if pid else ''}, {V.task_label(task)}: intent "
                              "labels derived from the optical incisal velocity (the decoding "
                              "ground truth), under the opening angle; note: seconds per "
                              "label.", sources=["labels:label", "labels:counts"])


@register("labels_counts", "dashboard", "Label time per participant", params=("phase",),
          phase="open_close", phases=V.TASKS + ("rest",), domain="ml")
def labels_counts(req: ViewRequest) -> FigureSpec:
    """Seconds per intent label, participant x label, in one phase."""
    from biomech.labels.auto_labels import CLASSES

    ctx, task = req.ctx, req.phase or "open_close"
    c = ctx.table("labels", "counts")
    c = c[c.phase == task] if not c.empty else c
    names = [n for n in list(CLASSES) + ["UNTRACKED"]
             if not c.empty and c[c.label == n].seconds.sum() > 0]
    p = B.matrix_heat(c, "pid", "label", "seconds", "Time per label",
                      rows=ctx.pids(), row_labels=B.pseudo_rows(ctx, ctx.pids()), cols=names,
                      col_labels=[n.lower() for n in names], fmt="{:.1f}",
                      cb=Colorbar("Time", V.SEC))
    return FigureSpec("labels_counts", "Label time per participant", [p],
                      width="column" if len(names) <= 5 else "text",
                      height_in=max(1.3, 0.22 * len(ctx.pids()) + 0.6),
                      caption=f"{V.task_label(task)}: seconds per intent label and participant "
                              "(classes absent in this phase omitted).",
                      sources=["labels:counts"])
