# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper depictions for the gallery.

Alternative forms of existing validation views (reached through each view's ``form`` picker,
drawn here so the original modules keep their current drawing as the default) and the new
items:

* ``val_setup``: the clock link of the two systems without hardware sync -- CT visibility,
  the CT angular speed in the first sync window and the residual lag over the take; the
  ``schematic`` form adds renders of the JTS-MM hardware (no participant);
* ``val_kin_error_ecdf``: the per-frame incisor and rotation disagreement of the movement
  tasks, own frames vs frames aligned, both participants (``ecdf``), or the per-task RMS
  of the two frame forms (``dumbbell``);
* ``val_bench``: bench trueness and precision in one figure;
* ``val_table_methods``: sensors, nominal vs achieved rates, timing and the clock model;
* ``val_table_segments_both``, ``val_table_features_both``: own frames and frames aligned
  side by side.

Stored results only. Two light derivations use the pipeline's own routines on stored values
and say so in the caption: the landmark positions of ``arrows3d`` in the Qualisys anatomical
frames (:func:`biomech.validation.anatomy.anatomical_frames` on the stored Qualisys
landmarks), and the achieved JTS-MM frame rate from the stored frame times. The per-frame
ECDF uses exactly the frames of the stored pooled RMS (movement tasks, reference gate
passed); the caption states that the RMS of the drawn frames equals the table value.
"""

from __future__ import annotations

import dataclasses

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import forms
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import (DEG, MM, SEC, MSEC, SYS, TOL, _arrays, _empty,
                                          _n_cycles, _pid, _table, _who, add_summary_rows,
                                          diff_ylim, n_text)
from biomech.viz.views.validation import pshape_text, shape_by_pid, shape_by_rows

G = {"group": "validation", "domain": "validation"}
SYSTEM_ROLE = {"jtsmm": "system:jtsmm", "qualisys": "system:qualisys", "delsys": "system:delsys"}
SYSTEM_NAME = {"jtsmm": "JTS-MM", "qualisys": "Qualisys", "delsys": "Delsys"}
FRAMES = (("own", "own frames", "system:jtsmm"), ("aligned", "frames aligned", "cat:2"))
# setup bodies of the schematic form (hardware/mechanical assemblies; biomech.hwparts)
RENDERS = (("head_piece", "Head piece"), ("mouth_piece", "Mouth piece"),
           ("calibration_tool", "Calibration tool"))


def _root() -> Path:
    return Path(__file__).resolve().parents[5]


def _pids(ctx: Any) -> list[str]:
    return list(ctx.pids())


def _movement_mask(a: dict[str, np.ndarray]) -> np.ndarray:
    """The frames of the stored pooled movement RMS: movement tasks, evaluated, reference
    gate passed (``ref_kinematics:segments``, ``movement_pooled``, gated)."""
    from biomech.validation.selection import MOVEMENT_TASKS

    ph = np.asarray(a["phase"]).astype(str)
    ev = np.asarray(a.get("evaluated", np.ones(len(ph), bool)), bool)
    gate = np.asarray(a.get("gate", np.zeros(len(ph), bool)), bool)
    return ev & ~gate & np.isin(ph, MOVEMENT_TASKS)


# == (I) marking: 3-D arrows ====================================================================
def marking_arrows3d(req: ViewRequest) -> FigureSpec:
    """Form ``arrows3d`` of ``val_marking_accuracy``: the stored error vector of every
    landmark (along the Qualisys anatomical axes) drawn at the landmark, in the Qualisys
    anatomical frames, one 3-D panel per participant, true scale."""
    from biomech.validation import anatomy as AN
    from biomech.validation.evaluate import holds_from_table

    fid, title = "val_marking_accuracy", "Point marking: difference per landmark"
    ctx = req.ctx
    e = _table(ctx, "ref_landmarks", "landmark_errors")
    lm = _table(ctx, "ref_tip", "landmarks")
    if e.empty or lm.empty:
        return _empty(fid, title, "no landmark errors")
    panels = []
    for pid in sorted(e["pid"].unique()):
        hq = holds_from_table(lm[lm["pid"] == pid], "qualisys", "primary")
        try:
            fr = AN.anatomical_frames(hq)
        except LookupError:
            continue
        inv = {"hp": np.linalg.inv(fr.F_max), "mp": np.linalg.inv(fr.F_mand)}
        pos = {n: inv[h.body][:3, :3] @ h.point + inv[h.body][:3, 3] for n, h in hq.items()
               if h.body in inv}
        p = Panel(_who(ctx, pid), Axis("Anterior", MM), Axis("Left", MM), [], legend="below",
                  projection="3d", z=Axis("Superior", MM), view=(-60.0, 20.0))
        for arch, role, lab in ((("max_point_2", "max_point_1", "max_point_3"), "context",
                                 "maxillary landmarks"),
                                (("mand_point_2", "mand_point_1", "mand_point_3"), "faint",
                                 "mandibular landmarks")):
            xyz = np.array([pos[n] for n in arch if n in pos])
            if len(xyz):
                p.series.append(Series("line3d", xyz[:, 0], xyz[:, 1], z=xyz[:, 2], role=role,
                                       label=lab if not panels else None))
        d = e[e["pid"] == pid]
        P, D = [], []
        for r in d.itertuples():
            if r.landmark in pos:
                P.append(pos[r.landmark])
                D.append([r.e_anterior_mm, r.e_left_mm, r.e_superior_mm])
        if P:
            P, D = np.array(P), np.array(D, float)
            p.series.append(Series("scatter3d", P[:, 0], P[:, 1], z=P[:, 2], role="system:qualisys",
                                   label="Qualisys landmark" if not panels else None))
            p.series.append(Series("arrow3d", P[:, 0], P[:, 1], z=P[:, 2], d=D,
                                   role="system:jtsmm",
                                   label="JTS-MM − Qualisys (difference)" if not panels else None))
        panels.append(p)
    if not panels:
        return _empty(fid, title, "no landmark frames")
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text", height_in=2.6,
                      sources=["ref_landmarks:landmark_errors", "ref_tip:landmarks"])
    return attach(spec, CaptionSpec(
        "point-marking difference of JTS-MM from Qualisys at each landmark, in the Qualisys "
        "anatomical frames (maxillary landmarks in the maxillary frame, mandibular landmarks "
        "and the condyle points held on the mouth piece in the mandibular frame; positions from "
        "the stored Qualisys landmarks)",
        [("lines", "dental arch through the two molars and the incisor"),
         ("dots", "Qualisys landmarks"),
         ("arrows", "difference vector JTS-MM − Qualisys at true scale (the components and their "
                    "95 % CIs of the depiction with the anatomical axes)")],
        n_text(ctx, e["pid"], f"{len(e)} landmarks"),
        statistic="one difference vector per landmark (point estimate), no summary"))


def marking_frames_dumbbell(req: ViewRequest) -> FigureSpec:
    """Form ``dumbbell`` of ``val_marking_frames``: the frame angle from the marked points and
    the constant offset estimated from motion, joined per frame."""
    fid, title = "val_marking_frames", "Point marking: anatomical frames"
    ctx = req.ctx
    f = _table(ctx, "ref_landmarks", "frame_errors")
    if f.empty:
        return _empty(fid, title, "no frame errors")
    lab = {"F_max": "maxillary", "F_mand": "mandibular"}
    labels = [f"{_who(ctx, r.pid)} {lab.get(r.frame, r.frame)}" for r in f.itertuples()]
    p = B.dumbbell(labels, f["angle_deg"], f["vd3_rot_deg"], "Frame angle", Axis("Angle", DEG),
                   label_a="from the marked points", label_b="from motion")
    p.legend = "below"
    shape_by_rows(p, list(f["pid"]), sorted(f["pid"].unique()))     # X1: shape = participant
    spec = FigureSpec(fid, title, [p], rows=1, cols=1, width="column",
                      height_in=B.rows_height(len(f), per=0.3, extra=0.6, lo=1.3),
                      sources=["ref_landmarks:frame_errors"])
    return attach(spec, CaptionSpec(
        "rotation between the anatomical frames that JTS-MM and Qualisys build from their marked "
        "points, next to the constant frame offset the frame alignment estimates from the motion "
        "data: two independent estimates of the same marking difference",
        [("blue", "from the marked points"),
         ("orange", "from motion (median over the leave-one-task-out folds)"),
         pshape_text(ctx, f["pid"]),
         ("line", "joins the two estimates of a frame")],
        n_text(ctx, f["pid"], f"{f['frame'].nunique()} frames each"),
        statistic="marked points: point estimate; motion: median over the leave-one-task-out "
                  "folds",
        notes=["The 95 % CIs of the marked-point angles are in the depiction with intervals."]))


# == (II) motion: per-frame error ===============================================================
@register("val_kin_error_ecdf", "dashboard", "Per-frame disagreement, own vs aligned frames",
          choices={"form": ("ecdf", "dumbbell")}, form="ecdf", **G)
def kin_error_ecdf(req: ViewRequest) -> FigureSpec:
    """Distribution of the per-frame incisor and rotation disagreement in the movement tasks,
    each system's own anatomical frames vs frames aligned, both participants."""
    fid, title = "val_kin_error_ecdf", "Per-frame disagreement, own vs aligned frames"
    ctx = req.ctx
    form = str(req.options.get("form", "ecdf"))
    if form == "dumbbell":
        return _kin_task_dumbbell(req, fid, title)
    pids = _pids(ctx)
    panels = []
    n_txt = []
    shown: set[str] = set()
    for key, lab, unit, u in (("incisor", "(a) Incisor", MM, "mm"),
                              ("rot", "(b) Rotation", DEG, "deg")):
        curves: dict[str, np.ndarray] = {}
        roles: dict[str, str] = {}
        for i, pid in enumerate(pids):
            a = _arrays(ctx, "ref_kinematics", pid)
            if "phase" not in a:
                continue
            m = _movement_mask(a)
            for fr, fr_lab, _ in FRAMES:
                col = f"err_{key}_mm" if key == "incisor" else "err_rot_deg"
                if fr == "aligned":
                    col = col.replace("_mm", "_aligned_mm").replace("_deg", "_aligned_deg")
                if col not in a:
                    continue
                v = np.asarray(a[col], float)[m]
                name = f"{_who(ctx, pid)}, {fr_lab}"
                curves[name] = v[np.isfinite(v)]
                # color = frame form, line style = participant (P1 solid, P2 dashed)
                roles[name] = f"cat:{2 if fr == 'aligned' else 1}" + (":alt" if i % 2 else "")
                shown.add(pid)
                if key == "incisor":
                    n_txt.append(f"{_who(ctx, pid)} {fr_lab}: {np.isfinite(v).sum()} frames")
        p = B.ecdf(curves, lab, Axis("Disagreement", unit, lim=None), roles=roles, points=False)
        bound = TOL["application"][0 if u == "mm" else 1]
        p.series.append(Series("vline", x=np.array([bound]), role="reference",
                               label=f"application bound {bound:g}" + ("\u00a0mm" if u == "mm" else "°")))
        # head-room above 100 % for the bound label, so it does not sit on the curves
        p.y = Axis("Cumulative share", "\\percent", lim=(0.0, 116.0),
                   ticks=[0.0, 25.0, 50.0, 75.0, 100.0])
        p.legend = "below" if key == "incisor" else "none"
        panels.append(p)
    if not any(s.kind == "line" for p in panels for s in p.series):
        return _empty(fid, title, "no ref_kinematics arrays")
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=1.8,
                      sources=["ref_kinematics:arrays", "ref_kinematics:segments",
                               "ref_kinematics:segments_aligned"])
    return attach(spec, CaptionSpec(
        "per-frame disagreement of JTS-MM and Qualisys in the movement tasks (lower incisor "
        "distance and rotation angle between the jaw-in-head poses), each system in its own "
        "anatomical frames and after the constant frame alignment (leave-one-task-out)",
        [("steps", "share of frames at or below a disagreement, one curve per participant and "
                   "frame form"),
         ("color", "frame form (orange own frames, green frames aligned)"),
         ("line style", "participant (solid P1, dashed P2)"),
         ("vertical line", "application bound (labeled)")],
        n_text(ctx, shown) + "; frames per participant: " + "; ".join(n_txt),
        statistic="empirical cumulative distribution of the per-frame values, per participant "
                  "(no pooling over participants)",
        notes=["The frames are those of the pooled movement RMS in the agreement tables "
               "(Qualisys quality gate passed); the RMS of the drawn frames equals the table value."]))


def _kin_task_dumbbell(req: ViewRequest, fid: str, title: str) -> FigureSpec:
    ctx = req.ctx
    own = _table(ctx, "ref_kinematics", "segments")
    al = _table(ctx, "ref_kinematics", "segments_aligned")
    if own.empty or al.empty:
        return _empty(fid, title, "no segment tables")
    own = own[own["gated"].astype(bool) & own["kind"].isin(["movement"])]
    al = al[al["kind"].isin(["movement"])]
    m = own.merge(al, on=["pid", "segment"], suffixes=("_own", "_al"))
    order = {t: i for i, t in enumerate(V.TASKS)}
    m = m.sort_values(["pid", "segment"], key=lambda c: c.map(order) if c.name == "segment"
                      else c, kind="stable")
    labels = [f"{_who(ctx, r.pid)} {V.phase_label(r.segment, short=True)}" for r in m.itertuples()]
    panels = []
    for col, lab, unit, b in (("incisor_rms_mm", "(a) Incisor", MM, TOL["application"][0]),
                              ("rot_rms_deg", "(b) Rotation", DEG, TOL["application"][1])):
        p = Panel(lab, Axis("RMS", unit), Axis(""), [], legend="below" if not panels else "none")
        y = np.arange(len(m), dtype=float)
        for i in range(len(m)):
            p.series.append(Series("line", np.array([m[f"{col}_own"].iloc[i],
                                                     m[f"{col}_al"].iloc[i]]),
                                   np.array([y[i], y[i]]), role="context"))
        for fr, fr_lab, role in FRAMES:
            c = f"{col}_own" if fr == "own" else f"{col}_al"
            p.series.append(Series("scatter", m[c].to_numpy(float), y, role=role,
                                   label=fr_lab if not panels else None))
        p.series.append(Series("vline", x=np.array([b]), role="reference"))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
        shape_by_rows(p, list(m["pid"]), sorted(m["pid"].unique()))    # X1
        if panels:
            p.y.ticklabels = [""] * len(labels)
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(m), per=0.15, extra=0.6, lo=1.6),
                      sources=["ref_kinematics:segments", "ref_kinematics:segments_aligned"])
    return attach(spec, CaptionSpec(
        "RMS disagreement per participant and movement task, each system in its own anatomical "
        "frames and after the constant frame alignment (leave-one-task-out); the drop is the "
        "marking difference the alignment removes",
        [("blue", "own frames"), ("green", "frames aligned"), pshape_text(ctx, m["pid"]),
         ("vertical line", "application bound")],
        f"n = {m['pid'].nunique()} participants", "RMS over the gated frames of each task"))


# == (II) motion: waveform forms ================================================================
def kin_waveforms_form(req: ViewRequest, form: str) -> FigureSpec | None:
    """Forms ``curves`` (only the cycles of both systems with SPM bars) and ``difference``
    (only the mean difference with its band against the bounds) of ``val_kin_waveforms``,
    four channels in a 2 × 2 grid."""
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
    spm_named = False
    for c, lab, unit, u in chans:
        key = f"{task}__{c}{sfx}"
        if f"{key}__mean_s" not in a:
            continue
        first = not panels
        tag = f"({chr(97 + len(panels))}) "
        if form == "curves":
            p = Panel(tag + lab, Axis("Cycle", "\\percent"), Axis(lab, unit), [],
                      legend="below")
            for s in ("s", "q"):
                m, sd = a[f"{key}__mean_{s}"], a[f"{key}__sd_{s}"]
                p.series.append(Series("band", x, m - sd, y2=m + sd, role=SYS[s][1]))
                p.series.append(Series("line", x, m, role=SYS[s][1], style={"width": 1.0},
                                       label=SYS[s][0] if first else None))
            row = w[w["channel"] == c] if not w.empty else w
            if not row.empty:
                spm_named |= bool(B.add_spm_bars(p, [(
                    "" if spm_named else "JTS-MM vs Qualisys", row.iloc[0]["clusters"],
                    "accent")]))
        else:
            p = Panel(f"{tag}Δ {lab}", Axis("Cycle", "\\percent"),
                      Axis("JTS-MM − Qualisys", unit), [], legend="below")
            p.series.append(Series("band", x, a[f"{key}__diff_lo"], y2=a[f"{key}__diff_hi"],
                                   role="accent", label="95 % band" if first else None))
            p.series.append(Series("line", x, a[f"{key}__diff_mean"], role="accent",
                                   label="mean difference" if first else None))
            p.y.lim = diff_ylim(a, key, u)
            for name, (mm, deg) in TOL.items():
                v = mm if u == "mm" else deg
                for sign in (-1, 1):
                    p.series.append(Series("hline", y=np.array([sign * v]),
                                           role="reference" if name == "application"
                                           else "context"))
            row = w[w["channel"] == c] if not w.empty else w
            if not row.empty:
                spm_named |= bool(B.add_spm_bars(p, [(
                    "" if spm_named else "JTS-MM vs Qualisys", row.iloc[0]["clusters"],
                    "accent")]))
        panels.append(p)
    for p in panels:                    # a legend only where a panel has labelled entries
        if not p.legend_entries():
            p.legend = "none"
    persp = f"{_who(ctx, pid)}, {V.phase_label(task)} cycles"
    frames_txt = ("frames aligned (constant maxillary and mandibular offsets fitted on the "
                  "other tasks)" if frames == "aligned" else "each system in its own "
                  "anatomical frames")
    mode_txt = ("displacement from the cycle start" if mode == "disp"
                else "absolute, including the constant offset of the frame definitions")
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=1.5,
                      sources=["ref_agreement:arrays", "ref_agreement:waveforms"],
                      perspective=persp, meta={"pid": pid})
    if form == "curves":
        cap = CaptionSpec(
            f"time-normalized {V.phase_label(task)} cycles of both systems, {frames_txt}, "
            f"{mode_txt}",
            [("lines and bands", "mean ± SD over cycles, JTS-MM and Qualisys"),
             ("bars above the curves", "supra-threshold clusters of the paired SPM t-test over "
                                       "cycles (p < 0.05)")],
            f"{_who(ctx, pid)}, {_n_cycles(w, task)}{V.phase_label(task)} cycles",
            statistic="mean ± SD over the cycles of each system",
            test="paired SPM t-test over cycles, p < 0.05")
    else:
        cap = CaptionSpec(
            f"mean difference JTS-MM − Qualisys over the time-normalized {V.phase_label(task)} "
            f"cycles, {frames_txt}, {mode_txt}",
            [("line", "mean difference"), ("band", "95 % band (t-interval per node with the effective number of cycles)"),
             ("dashed lines", f"application bound (±{TOL['application'][0]:g} mm, "
                              f"±{TOL['application'][1]:g}°)"),
             ("thin grey lines", f"metrology bound (±{TOL['metrology'][0]:g} mm, "
                                 f"±{TOL['metrology'][1]:g}°)"),
             ("bars above the curves", "supra-threshold clusters of the paired SPM t-test "
                                       "(p < 0.05)")],
            f"{_who(ctx, pid)}, {_n_cycles(w, task)}{V.phase_label(task)} cycles",
            statistic="mean difference over the cycles with a t-interval per node using the "
                      "effective number of cycles",
            test="paired SPM t-test over cycles, p < 0.05",
            notes=["Equivalence where the band lies inside a bound."])
    return attach(spec, cap)


# == (II) motion: Bland–Altman of the cycle features ============================================
FEATURES = (("amp_mm", "Task amplitude", MM, "mm"),
            ("excursion_mm", "Incisor excursion", MM, "mm"),
            ("peak_speed_mm_s", "Peak incisal speed", "\\milli\\metre\\per\\second", None),
            ("rot_excursion_deg", "Rotation excursion", DEG, "deg"))


def kin_agreement_ba(req: ViewRequest) -> FigureSpec:
    """Form ``bland_altman`` of ``val_kin_agreement``: per cycle the difference against the
    mean of the two systems, colored by task, with the stored bias and limits of agreement
    of each participant and task."""
    from biomech.viz.views.validation_part_i import missing_cycle_notes

    fid, title = "val_kin_agreement", "Cycle-feature agreement"
    ctx = req.ctx
    frames = str(req.options.get("frames", "aligned"))
    cyc = _table(ctx, "ref_agreement", "cycles_aligned" if frames == "aligned" else "cycles")
    f = _table(ctx, "ref_agreement", "features_aligned" if frames == "aligned" else "features")
    if cyc.empty or f.empty:
        return _empty(fid, title, "no cycle features")
    tasks = [t for t in V.TASKS if t in set(cyc["task"])]
    pids = sorted(cyc["pid"].unique())
    panels = []
    for feat, lab, unit, u in FEATURES:
        col = feat
        if f"{col}_s" not in cyc:
            continue
        first = not panels
        p = Panel(f"({chr(97 + len(panels))}) {lab}", Axis("Mean of both systems", unit),
                  Axis("JTS-MM − Qualisys", unit), [], legend="below" if first else "none")
        for ti, task in enumerate(tasks):
            for pi, pid in enumerate(pids):
                d = cyc[(cyc["task"] == task) & (cyc["pid"] == pid)]
                if d.empty:
                    continue
                s, q = d[f"{col}_s"].to_numpy(float), d[f"{col}_q"].to_numpy(float)
                mean, diff = (s + q) / 2, s - q
                p.series.append(Series("scatter", mean, diff, role=f"cat:{ti}",
                                       style={"ms": "o" if pi == 0 else "s", "size": 0.8},
                                       label=(f"{V.phase_label(task, short=True)}"
                                              if first and pi == 0 else None)))
                r = f[(f["pid"] == pid) & (f["task"] == task) & (f["feature"] == feat)]
                if r.empty or not np.isfinite(mean).any():
                    continue
                r = r.iloc[0]
                xr = np.array([np.nanmin(mean), np.nanmax(mean)])
                p.series.append(Series("line", xr, np.full(2, r["bias"]), role=f"cat:{ti}",
                                       style={"width": 1.4}))
                for k in ("loa_lo", "loa_hi"):
                    p.series.append(Series("line", xr, np.full(2, r[k]), role=f"cat:{ti}",
                                           style={"width": 0.5}))
        p.series.append(Series("hline", y=np.array([0.0]), role="context"))
        if u is not None:
            v = TOL["application"][0 if u == "mm" else 1]
            for sgn in (-1, 1):
                p.series.append(Series("hline", y=np.array([sgn * v]), role="reference"))
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=2, cols=2, width="text", height_in=1.7,
                      sources=["ref_agreement:cycles" + ("_aligned" if frames == "aligned"
                                                         else ""),
                               "ref_agreement:features" + ("_aligned" if frames == "aligned"
                                                           else "")])
    return attach(spec, CaptionSpec(
        "Bland–Altman agreement of the cycle features, " + (
            "frames aligned" if frames == "aligned" else "each system in its own frames")
        + ": per cycle the difference JTS-MM − Qualisys against the mean of the two systems",
        [("dots", f"cycles, color = task, circle = {_who(ctx, pids[0])}"
                  + (f", square = {_who(ctx, pids[1])}" if len(pids) > 1 else "")),
         ("thick segment", "bias of the participant and task"),
         ("thin segments", "95 % limits of agreement"),
         ("dark horizontal lines", "application bound (no bound for the peak speed)")],
        n_text(ctx, pids, f"{len(cyc)} cycles"),
        statistic="per participant and task: bias = mean of the cycle differences, limits of "
                  "agreement = bias ± 1.96 SD of the cycle differences",
        notes=["Bias CIs, CCC and ICC are in the cycle-feature table.",
               *missing_cycle_notes(ctx, f, sorted(f["pid"].unique()))]))


# == fusion: dropouts as dots ===================================================================
def fusion_dropout_strip(req: ViewRequest) -> FigureSpec:
    """Form ``strip`` of ``val_fusion_dropout``: at the shortest and the longest simulated gap,
    every participant × held-out task as a dot per estimator, with the median and its CI."""
    from biomech.viz.views.validation import EST, EST_ALL

    fid, title = "val_fusion_dropout", "Fusion: bridging simulated optical dropouts"
    ctx = req.ctx
    d = _table(ctx, "ref_fusion", "dropout")
    if d.empty:
        return _empty(fid, title, "no dropout table")
    names = {"F0_hold": "hold last pose", "F0": "SLERP / linear", "F1": EST["F1"][0],
             "F2": EST["F2"][0], "F2P": EST_ALL["F2P"][0], "F2P_causal": EST_ALL["F2P_causal"][0]}
    ests = [e for e in names if e in set(d["estimator"])]
    gaps = sorted(d["dropout_s"].unique())
    show = [gaps[0], gaps[-1]] if len(gaps) > 1 else gaps
    seed = B.seed_of(ctx)
    panels = []
    for col, lab, unit in (("rot_aligned_rms_deg", "Rotation", DEG),
                           ("incisor_aligned_rms_mm", "Incisor", MM)):
        if col not in d:
            continue
        for g in show:
            x = d[d["dropout_s"] == g]
            vals = [x.loc[x["estimator"] == e, col].to_numpy(float) for e in ests]
            p, marks = forms.summary("strip", [names[e] for e in ests], vals,
                                     f"({chr(97 + len(panels))}) {lab}, {g:g}-s gap",
                                     Axis("RMS", unit), seed=seed,
                                     unit="participant × task values")
            if len(panels) % len(show):
                p.y.ticklabels = [""] * len(ests)
            panels.append(p)
    n_pid, n_task = d["pid"].nunique(), d["segment"].nunique()
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / len(show))),
                      cols=len(show), width="text",
                      height_in=B.rows_height(len(ests), per=0.2, extra=0.6, lo=1.4),
                      sources=["ref_fusion:dropout"])
    return attach(spec, CaptionSpec(
        "difference to Qualisys on the dropped frames, frames aligned, at the shortest and the longest simulated "
        "gap, per estimator",
        marks, f"n = {n_pid} participants × {n_task} held-out tasks",
        "median and its 95 % BCa CI",
        notes=["The curves over every gap length are in the line depiction."]))


# == EMG ========================================================================================
def emg_timing_raincloud(req: ViewRequest) -> FigureSpec:
    """Form ``raincloud`` of ``val_emg_timing``: the leads of every usable channel-cycle per
    participant, task, direction and system as a raincloud."""
    from biomech.viz.views.validation_emg import SYSTEMS

    fid, title = "val_emg_timing", "EMG: onset timing, both systems"
    ctx = req.ctx
    on = _table(ctx, "ref_emg", "timing_onsets")
    if on.empty:
        return _empty(fid, title, "no ref_emg onsets")
    rule = ctx.onset_rule()
    u = on[on["usable"].astype(bool) & (on["detector"] == "tkeo") & (on["rule"] == rule)]
    src_lab = {"jtsmm_optical": "JTS-MM optics", "qualisys": "Qualisys"}
    labels, vals = [], []
    for (pid, task, direction), g in u.groupby(["pid", "task", "direction"]):
        for (sys_, src), h in g.groupby(["system", "mech_source"]):
            labels.append(f"{_who(ctx, pid)} {V.phase_label(task, short=True)} {direction}: "
                          f"{SYSTEMS[sys_][0]} ({src_lab.get(src, src)})")
            vals.append(h["lead_ms"].to_numpy(float))
    p, marks = forms.summary("raincloud", labels, vals, "Leads per channel-cycle",
                             Axis("EMG lead", MSEC), seed=B.seed_of(ctx), ref=0.0,
                             unit="channel-cycles")
    spec = FigureSpec(fid, title, [p], rows=1, cols=1, width="text",
                      height_in=B.rows_height(len(labels), per=0.22, extra=0.5, lo=1.6),
                      sources=["ref_emg:timing_onsets"])
    return attach(spec, CaptionSpec(
        f"EMG onset (TKEO, rule '{rule}') to the optical mechanical onset per usable "
        "channel-cycle (> 0: EMG first), per participant, task, direction and system",
        marks + [("vertical line", "simultaneous onset")],
        n_text(ctx, u["pid"], f"{len(u)} usable channel-cycles"),
        statistic="distribution of the per-channel-cycle leads per row, no pooling over "
                  "participants",
        notes=["Cross-session comparison: agreement in distribution, not per cycle; the "
               "difference of the medians with its CI is in the depiction with dots."]))


def emg_quality_dumbbell(req: ViewRequest) -> FigureSpec:
    """Form ``dumbbell`` of ``val_emg_quality``: rest noise, SNR and MVC level per muscle,
    JTS-MM and Delsys joined."""
    from biomech.viz.views.validation_emg import UV, DB, _muscles

    fid, title = "val_emg_quality", "EMG: signal quality, both systems"
    ctx = req.ctx
    pid = _pid(req)
    q = _table(ctx, "ref_emg", "quality", [pid]) if pid else pd.DataFrame()
    mv = _table(ctx, "ref_emg", "mvc", [pid]) if pid else pd.DataFrame()
    if q.empty:
        return _empty(fid, title, "no ref_emg quality table")
    ms = _muscles(list(q["muscle"].unique()))
    labels = [V.muscle_label(m) for m in ms]

    def vals(df: pd.DataFrame, col: str, s: str) -> np.ndarray:
        return np.array([float(df[(df.system == s) & (df.muscle == m)][col].median())
                         if not df.empty and ((df.system == s) & (df.muscle == m)).any()
                         else np.nan for m in ms])

    panels = []
    for df, col, lab, ax in ((q, "rest_rms_uv", "(a) Rest noise", Axis("RMS", UV, log=True)),
                             (q, "snr_db", "(b) SNR", Axis("MVC p95 / rest", DB)),
                             (mv, "mvc_max_uv", "(c) MVC level",
                              Axis("Envelope maximum", UV, log=True))):
        if df.empty or col not in df:
            continue
        first = not panels
        p = Panel(lab, ax, Axis(""), [], legend="below" if first else "none")
        y = np.arange(len(ms), dtype=float)
        a, b = vals(df, col, "jtsmm"), vals(df, col, "delsys")
        for i in range(len(ms)):
            if np.isfinite(a[i]) and np.isfinite(b[i]):
                p.series.append(Series("line", np.array([a[i], b[i]]), np.array([y[i], y[i]]),
                                       role="context"))
        p.series.append(Series("scatter", a, y, role="system:jtsmm", style={"ms": "o"},
                               label="JTS-MM" if first else None))
        p.series.append(Series("scatter", b, y, role="system:delsys", style={"ms": "s"},
                               label="Delsys" if first else None))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels if first else [""] * len(ms), True
        panels.append(p)
    if pid in ctx.pids():                 # X1: shape = participant, colour = system
        shape_by_pid(panels, sorted(ctx.pids()).index(pid))
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=B.rows_height(len(ms), per=0.2, extra=0.6, lo=1.5),
                      sources=["ref_emg:quality", "ref_emg:mvc"], perspective=_who(ctx, pid),
                      meta={"pid": pid})
    return attach(spec, CaptionSpec(
        f"signal quality per muscle of both EMG systems, {_who(ctx, pid)}: rest noise, "
        "signal-to-noise ratio (MVC p95 over the rest envelope) and the MVC level, same "
        "processing chain for both systems",
        [("blue", "JTS-MM"), ("black", "Delsys"), ("line", "joins the two systems"),
         ("marker", "the participant's shape (circle P1, square P2)")],
        f"{_who(ctx, pid)}, one value per muscle and system",
        statistic="rest values from the rest phase; MVC level: maximum of the masked envelope "
                  "over the MVC phases",
        notes=["JTS-MM: dataset take; Delsys: comparison take (different days, electrodes and "
               "sites)."]))


# == bench ======================================================================================
@register("val_bench", "dashboard", "Bench trueness and precision",
          choices={"form": ("panels", "dumbbell", "strip", "merged")}, form="panels", **G)
def bench(req: ViewRequest) -> FigureSpec:
    """Trueness (step error against the nominal 20-mm ladder) and precision (static SD) of
    both systems in one figure (the separate views keep the drift and the traces). Forms
    ``dumbbell`` / ``strip``: ``validation_r8`` (J2.6); ``merged``:
    :func:`bench_merged`."""
    form = str(req.options.get("form", "panels"))
    if form == "merged":
        return bench_merged(req)
    if form != "panels":
        from biomech.viz.views.validation_r8 import bench_form

        return bench_form(req, form)
    from biomech.viz.views.validation_emg import precision_view, trueness_view

    fid, title = "val_bench", "Bench trueness and precision"
    t = trueness_view(req)
    pr = precision_view(req)
    tp = [p for p in t.panels if any(s.kind in ("line", "scatter") for s in p.series)]
    pp = [p for p in pr.panels if p.title.startswith(("(a)", "(b)"))]
    if not tp and not pp:
        return _empty(fid, title, "no bench results")
    panels = []
    for i, p in enumerate(tp + pp):
        tag_free = p.title.split(") ", 1)[-1]
        p.title = f"({chr(97 + i)}) {tag_free}"
        p.legend = "below" if i == 0 or (i >= len(tp) and p.legend_entries()) else "none"
        panels.append(p)
    if tp and pp:
        # The line keys go under the first precision panel, at the bottom of the figure: under
        # (a) they split the trueness grid. Legend-only entries (no points).
        keys = [Series(s.kind, x=np.array([np.nan]), y=np.array([np.nan]), role=s.role,
                       style=dict(s.style), label=s.label)
                for s in tp[0].legend_entries()]
        for s in tp[0].series:
            s.label = None
        tp[0].legend = "none"
        pp[0].series += keys
        pp[0].legend = "below"
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=1.45,
                      sources=sorted(set(t.sources) | set(pr.sources)))
    from biomech.viz.views.validation_emg import _ladder_n

    tt = _table(req.ctx, "ref_trueness", "trueness")
    prt = _table(req.ctx, "ref_trueness", "precision")
    n = "; ".join(x for x in (_ladder_n(tt), f"{len(prt)} static takes of 30 s" if len(prt)
                              else "") if x)
    return attach(spec, CaptionSpec(
        "trueness and precision of both systems on the hand-measured 20-mm ladder (bench, no "
        "participant), per orientation and camera distance",
        [("solid blue / dashed black", "JTS-MM / Qualisys (legend)"),
         ("lines with dots", "measured minus nominal step at the far slot of each step "
                             "(trueness)"),
         ("light lines", "residual to the metric ladder line (scale fixed to the ruler), in "
                         "the system's color and line style"),
         ("markers (precision panels)", "static 3-D SD per condition or take in the system "
                                        "color, JTS-MM rows first (grey: the 30-s "
                                        "take not detrended)")],
        n, statistic="trueness: one step error per step, no summary; precision: 3-D SD pooled "
                     "within the slot takes of a condition, and of each detrended 30-s take",
        notes=["Mean step errors with CIs, the drift and the 30-s traces are in the bench "
               "table and the precision figure."]))


BENCH_MARK = {"near": "o", "far": "^"}


def bench_merged(req: ViewRequest) -> FigureSpec:
    """Form ``merged`` of ``val_bench`` (2026-11-05): near and far camera distance in one
    trueness panel per orientation (4 -> 2 panels): system by colour and line style as
    everywhere (JTS-MM blue solid, Qualisys black dashed), distance by marker (near circles,
    far triangles); the residual lines along the ladder stay in the four-panel form. The
    precision panels as in ``panels``."""
    from biomech.viz.views.validation_emg import _ladder_n, precision_view, trueness_view

    fid, title = "val_bench", "Bench trueness and precision"
    t = trueness_view(req)
    pr = precision_view(req)
    tp = [p for p in t.panels if any(s.kind in ("line", "scatter") for s in p.series)]
    pp = [p for p in pr.panels if p.title.startswith(("(a)", "(b)"))]
    if not tp and not pp:
        return _empty(fid, title, "no bench results")
    orient: dict[str, Panel] = {}
    for p in tp:
        o, _, dist = p.title.split(") ", 1)[-1].partition(", ")
        dist = dist.strip().lower()
        q = orient.get(o)
        if q is None:
            q = orient[o] = Panel(o, p.x, p.y, [], legend="none")
        for s in p.series:
            if "residual" in (s.label or "") or (s.style or {}).get("alpha"):
                continue                          # the light residual lines: four-panel form
            if s.kind == "scatter":
                sysname = "JTS-MM" if s.role == "system:jtsmm" else "Qualisys"
                q.series.append(dataclasses.replace(
                    s, style={**(s.style or {}), "ms": BENCH_MARK.get(dist, "o"), "size": 1.6},
                    label=f"{sysname}, {dist}"))
            else:
                q.series.append(dataclasses.replace(s, label=None))
    panels = list(orient.values())
    for i, p in enumerate(panels + pp):
        p.title = f"({chr(97 + i)}) " + p.title.split(") ", 1)[-1]
    panels[0].legend = "below"
    for p in pp:
        p.legend = "below" if p.legend_entries() else "none"
    panels += pp
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=1.3,
                      sources=sorted(set(t.sources) | set(pr.sources)),
                      meta={"legend_span": "figure"})
    tt = _table(req.ctx, "ref_trueness", "trueness")
    prt = _table(req.ctx, "ref_trueness", "precision")
    n = "; ".join(x for x in (_ladder_n(tt), f"{len(prt)} static takes of 30 s" if len(prt)
                              else "") if x)
    return attach(spec, CaptionSpec(
        "trueness and precision of both systems on the hand-measured 20-mm ladder (bench, no "
        "participant), per orientation; near and far camera distance in one panel",
        [("solid blue / dashed black", "JTS-MM / Qualisys"),
         ("circles / triangles", "near / far camera distance"),
         ("lines with markers", "measured minus nominal step at the far slot of each step "
                                "(trueness)"),
         ("markers (precision panels)", "static 3-D SD per condition or take in the system "
                                        "color, JTS-MM rows first (grey: the 30-s "
                                        "take not detrended)")],
        n, statistic="trueness: one step error per step, no summary; precision: 3-D SD pooled "
                     "within the slot takes of a condition, and of each detrended 30-s take",
        notes=["The residuals along the ladder per distance, the mean step errors with CIs, "
               "the drift and the 30-s traces are in the supplement."]))


# == setup and clock link =======================================================================
def _render(study: str, assembly: str, title: str, max_px: int = 480) -> Panel | None:
    """The used version of a setup body (original JTS render while not specified)."""
    from biomech.viz.views.hardware import assembly_image, image_panel

    path = assembly_image(study, assembly)
    return image_panel(path, title, max_px) if path is not None else None


@register("val_setup", "dashboard", "Setup: two systems on one clock", params=("pid",),
          choices={"form": ("signals", "schematic")}, form="signals", **G)
def setup(req: ViewRequest) -> FigureSpec:
    """The clock link of JTS-MM and Qualisys without hardware sync: CT visibility in both
    systems over the take, the CT angular speed of both systems in the first sync window after
    the fit, and the residual lag of sliding windows under the fitted clock. ``schematic``
    adds renders of the JTS-MM head piece, mouth piece and calibration tool."""
    from biomech.viz.views.validation import sync_lag, sync_overview, sync_windows

    fid, title = "val_setup", "Setup: two systems on one clock"
    ctx = req.ctx
    pid = _pid(req)
    form = str(req.options.get("form", "signals"))
    if pid is None:
        return _empty(fid, title, "no participants")
    ov = sync_overview(req)
    wn = sync_windows(req)
    lg = sync_lag(req)
    sig: list[Panel] = []
    vis = [p for p in ov.panels if p.title.startswith("(a)")]
    if vis:
        v = vis[0]
        v.title, v.legend = "CT visible", "none"
        v.x = Axis("Time in take", SEC)
        v.series = [s for s in v.series if s.kind == "line"]
        sig.append(v)
    om = [p for p in wn.panels if p.y.label == "|ω|"]
    if om:
        o = om[0]
        o.title, o.legend = "CT speed, start window", "below"
        sig.append(o)
    rl = [p for p in lg.panels if p.title.startswith("(a)")]
    if rl:
        r = rl[0]
        r.title, r.legend = "Residual lag over the take", "below"
        sig.append(r)
    if not sig:
        return _empty(fid, title, f"no sync results for {_who(ctx, pid)}")
    panels: list[Panel] = []
    study = str(getattr(ctx.config, "study", "") or "")
    if form == "schematic":
        for name, lab in RENDERS:
            q = _render(study, name, lab)
            if q is not None:
                q.equal = False             # keep the grid cell aspect of the old panels
                panels.append(q)
    n_img = len(panels)
    cols = 3
    # signal panels each take a full row of the grid (empty cells after them)
    grid: list[Panel] = list(panels)
    for p in sig:
        grid.append(p)
    for i, p in enumerate(grid):
        _, t = p.heading()
        p.title, p.tag = f"({chr(97 + i)}) {t}", ""
    spec = FigureSpec(fid, title, grid, rows=int(np.ceil(len(grid) / cols)), cols=cols,
                      width="text", height_in=1.5,
                      sources=sorted(set(ov.sources) | set(wn.sources) | set(lg.sources)),
                      perspective=f"{_who(ctx, pid)}, comparison take", meta={"pid": pid})
    marks = [("thick lines", "calibration tool (CT) visible in JTS-MM (top) and Qualisys "
                             "(bottom) on the master clock"),
             ("lines", "CT angular speed of both systems in the start sync window after the "
                       "clock fit (solid JTS-MM, dashed Qualisys)"),
             ("markers with whiskers", "lag of sliding 15-s windows minus the fitted affine "
                                       "clock "
                                    "(± SE), black from the angular speed, grey from the "
                                    "registered poses; flat = the clock model holds over the "
                                    "take"),
             ("dashed horizontal line", "the fitted clock (zero residual lag)")]
    shows = ("how JTS-MM and Qualisys share one clock without a hardware sync line: the "
             "calibration tool is waved in view of both systems at the start and the end of "
             "the take, and an affine clock (offset and drift) is fitted to its motion")
    notes: list[str] = []
    if n_img:
        from biomech.viz.views.hardware import usage_note

        shows = ("the JTS-MM hardware (renders: head piece, mouth piece and calibration tool) "
                 "and " + shows)
        notes = usage_note(study)
    return attach(spec, CaptionSpec(
        shows, marks, f"{_who(ctx, pid)}, comparison take",
        statistic="raw signals of one take, no summary; lag per 15-s window with its standard "
                  "error", notes=notes))


# == tables =====================================================================================
@register("val_table_methods", "dashboard", "Table: sensors, rates and clock", **G)
def table_methods(req: ViewRequest) -> TableSpec:
    """Sensors, nominal and achieved rates, timing and the clock model of the comparison
    takes, per participant."""
    ctx = req.ctx
    sy = _table(ctx, "ref_sync", "sync")
    tl = _table(ctx, "ref_emg", "timeline")
    ho = _table(ctx, "ref_sync", "heldout")
    rows = []
    mixed = []                  # participants whose Delsys sensors ran at different rates
    head_rot = []               # pooled held-out head-body rotation, for the note
    for pid in _pids(ctx):
        a = _arrays(ctx, "ref_kinematics", pid)
        t = np.asarray(a.get("t_s", np.zeros(0)), float)
        dt = np.diff(t[np.isfinite(t)]) if t.size > 1 else np.zeros(0)
        med = float(np.median(dt)) if dt.size else np.nan
        long_ = dt > 1.5 * med if dt.size else np.zeros(0, bool)
        r = sy[sy["pid"] == pid].iloc[0] if not sy.empty and (sy["pid"] == pid).any() else None
        e = tl[tl["pid"] == pid].iloc[0] if not tl.empty and (tl["pid"] == pid).any() else None
        delsys = "–"
        if e is not None and isinstance(e.get("delsys_rates_hz"), str):
            per = {x.split("=")[0]: float(x.split("=")[1])
                   for x in e["delsys_rates_hz"].split(";") if "=" in x}
            rates = sorted(set(per.values()))
            delsys = ", ".join(f"{v:.0f}" for v in rates)
            if len(rates) > 1:
                main = max(rates, key=list(per.values()).count)
                odd = [f"{V.muscle_label(m)} {v:.0f} Hz" for m, v in per.items()
                       if v != main]
                mixed.append(f"{_who(ctx, pid)}: {', '.join(odd)}, the other sensors "
                             f"{main:.0f} Hz")
        rows.append({
            "pid": f"P{ctx.pseudonym(pid)}",
            "jtsmm_rate": f"{1.0 / med:.1f}" if dt.size else "–",
            "jtsmm_jitter": f"{1e6 * np.std(dt[~long_]):.1f}" if dt.size else "–",
            "jtsmm_missing": f"{100 * long_.mean():.1f}" if dt.size else "–",
            "qualisys_rate": f"{r['qtm_fs_hz']:.0f}" if r is not None else "–",
            "delsys_rate": delsys,
            "emg_grid": f"{e['resample_hz']:.0f}" if e is not None else "–",
            "drift": (f"{r['alpha_ppm']:.1f} ± {r['se_alpha_ppm_hac']:.1f}"
                      if r is not None else "–"),
            "offset_se": f"{r['se_offset_ms_hac']:.2f}" if r is not None else "–",
            "heldout_body": (f"{r['heldout_mp_pos_rms_mm']:.2f} / "
                             f"{r['heldout_hp_pos_rms_mm']:.2f}" if r is not None else "–"),
            "heldout": (f"{r['heldout_rel_pos_rms_mm']:.2f} / {r['heldout_rel_rot_rms_deg']:.2f}"
                        if r is not None else "–")})
        rot = _heldout_rot(ho, pid, "HP")
        if np.isfinite(rot):
            head_rot.append(f"{_who(ctx, pid)} {rot:.1f}°")
    cols = [Column("pid", "Participant", align="l"),
            Column("jtsmm_rate", "JTS-MM optics (Hz)", align="r", group="Achieved rate"),
            Column("jtsmm_jitter", "interval SD (µs)", align="r", group="Achieved rate"),
            Column("jtsmm_missing", "missing frames (%)", align="r",
                   group="Achieved rate"),
            Column("qualisys_rate", "Qualisys (Hz)", align="r", group="Achieved rate"),
            Column("delsys_rate", "Delsys EMG (Hz)", align="r", group="Achieved rate"),
            Column("emg_grid", "EMG grid (Hz)", align="r"),
            Column("drift", "drift α (ppm)", align="r", group="Clock model"),
            Column("offset_se", "offset SE (ms)", align="r", group="Clock model"),
            Column("heldout_body", "MP / HP position (mm)", align="r",
                   group="Held-out intervals"),
            Column("heldout", "jaw in head (mm / °)", align="r",
                   group="Held-out intervals")]
    # the column definitions as a table note under the table, so the caption stays short
    #
    defs = ("Achieved rate: median of the recorded camera-frame intervals (JTS-MM), the rate in "
            "the Qualisys and Delsys file headers"
            + (" (Delsys sensors at different rates: " + "; ".join(mixed) + ")" if mixed
               else "")
            + ". Interval SD: SD of the JTS-MM frame intervals without the intervals longer "
            "than 1.5 times the median (hardware-triggered cameras). Missing frames: share of "
            "JTS-MM frame intervals longer than 1.5 times the median. EMG grid: common rate "
            "both electromyography (EMG) systems are resampled to. Drift α: clock drift ± its heteroscedasticity- "
            "and autocorrelation-consistent (HAC) standard error, from the per-window offsets. "
            "Offset SE: HAC standard error of the clock offset. Held-out intervals: RMS "
            "residual on the movement intervals not used for the clock fit; MP / HP position: "
            "each registered body, mandible plate (MP) and head plate (HP), against Qualisys "
            "at its body origin; jaw in head: the mandible-plate pose relative to the head plate, position "
            "/ rotation. The jaw-in-head position is taken at the mandible-plate origin in the "
            "head-plate frame, so an orientation difference of the head plate between the two "
            "systems turns into a position offset proportional to the distance between the two "
            "body origins"
            + (" (head-plate rotation residual " + ", ".join(head_rot) + ")" if head_rot
               else "")
            + "; it is a frame diagnostic, not a timing error. The head-plate rotation "
            "difference lies almost entirely about the long axis of the Qualisys head-marker "
            "cluster, which nearly coincides with the tag baseline of the JTS-MM head array, "
            "the weakest axis of both; it is not assigned to either system.")
    t = TableSpec("val_table_methods", "", cols, rows, notes=defs, width="text",
                  sources=["ref_kinematics:arrays", "ref_sync:sync", "ref_sync:heldout",
                           "ref_emg:timeline"])
    return attach(t, CaptionSpec(
        "the two measurement systems in the comparison takes: achieved sampling rates, the "
        "timing of the JTS-MM camera frames, and the affine clock that maps Qualisys and Delsys "
        "time onto the JTS-MM master clock (columns defined below the table)", [],
        n_text(ctx, _pids(ctx)),
        statistic="per participant one take: median and SD of the frame intervals, share of "
                  "long intervals; clock drift and offset from the affine fit with HAC "
                  "standard errors",
        notes=["IMUs (BNO085): game rotation vector requested at 390 Hz and accelerometer "
               "at 250 Hz (session configuration); their delivered rates are not stored in "
               "the validation results",
               "The held-out residuals are evaluated at the body origins, so they contain "
               "the world and body registrations: a registration diagnostic, not a measure of "
               "the clock (the clock's quality is the offset SE); the motion agreement at the "
               "incisor and condyles is reported separately"]))


def _heldout_rot(ho: pd.DataFrame, pid: str, body: str) -> float:
    """n-weighted RMS rotation residual of one body over the held-out movement intervals
    (pooled like the ``heldout_*`` columns of ``ref_sync:sync``)."""
    if ho.empty:
        return float("nan")
    d = ho[(ho["pid"] == pid) & (ho["body"] == body) & (ho["n"] > 0)
           & ho["interval"].isin(V.TASKS)]
    if d.empty:
        return float("nan")
    return float(np.sqrt(np.sum(d["n"] * d["rot_rms_deg"] ** 2) / d["n"].sum()))


def _merge_both(own: pd.DataFrame, al: pd.DataFrame, keys: list[str], cols: list[str]
                ) -> pd.DataFrame:
    m = own[keys + cols].merge(al[keys + cols], on=keys, how="outer", suffixes=("", "__al"))
    return m


def _both_table(req: ViewRequest, tid: str, own: pd.DataFrame, al: pd.DataFrame,
                keys: list[str], cols: list[str], cap: CaptionSpec, sources: list[str]
                ) -> TableSpec:
    ctx = req.ctx
    m = _merge_both(own, al, keys, cols)
    order = {t: i for i, t in enumerate(V.TASKS)}
    sort_key = "segment" if "segment" in keys else "task"
    m = m.sort_values(["pid", sort_key], key=lambda c: c.map(lambda v: order.get(v, 99))
                      if c.name == sort_key else c, kind="stable")
    base = B.frame_table(m[keys + cols], tid, "", ctx=ctx)
    al_tab = B.frame_table(m[keys + [c + "__al" for c in cols]].rename(
        columns={c + "__al": c for c in cols}), tid, "", ctx=ctx)
    columns = []
    for c in base.columns:
        if c.key in keys:
            columns.append(c)
    for grp, tab, sfx in (("own frames", base, ""), ("frames aligned", al_tab, "__al")):
        for c in tab.columns:
            if c.key in keys:
                continue
            columns.append(Column(c.key + sfx, c.header, c.unit, c.decimals, c.align, grp))
    rows = []
    for r0, r1 in zip(base.rows, al_tab.rows):
        row = dict(r0)
        row.update({k + "__al": v for k, v in r1.items() if k not in keys})
        rows.append(row)
    t = TableSpec(tid, "", columns, rows, width="text", sources=sources)
    return attach(t, cap)


@register("val_table_segments_both", "dashboard",
          "Table: jaw-in-head agreement per task, own and aligned frames", **G)
def table_segments_both(req: ViewRequest) -> TableSpec:
    """Per participant and movement task: incisor and rotation RMS in each system's own
    anatomical frames (gated) and after the frame alignment, side by side."""
    ctx = req.ctx
    own = _table(ctx, "ref_kinematics", "segments")
    al = _table(ctx, "ref_kinematics", "segments_aligned")
    keys, cols = ["pid", "segment"], ["incisor_rms_mm", "rot_rms_deg"]
    if not own.empty:
        own = own[own["gated"].astype(bool) & own["kind"].isin(["movement", "pooled"])]
    if own.empty or al.empty:
        return B.frame_table(pd.DataFrame(), "val_table_segments_both", "no segment tables")
    t = _both_table(req, "val_table_segments_both", own, al, keys, cols, CaptionSpec(
        "jaw-in-head disagreement of JTS-MM and Qualisys per movement task, each system in its "
        "own anatomical frames and after the constant frame alignment (leave-one-task-out)",
        [("incisor", "RMS of the lower-incisor distance"),
         ("rotation", "RMS of the rotation angle between the poses")],
        n_text(ctx, own["pid"]),
        statistic="per participant and task the RMS over the task's frames that pass the "
                  "Qualisys quality gate; the pooled row takes all movement-task frames of the "
                  "participant together",
        notes=["Per-axis bias and SD, "
               "bias-removed RMS and the condyle points are in the separate tables.",
               "Last rows: per task the descriptive mean of the participants (not pooled)."]),
        ["ref_kinematics:segments", "ref_kinematics:segments_aligned"])
    n_p = own["pid"].nunique()
    return add_summary_rows(t, "pid", f"mean of {n_p}", group=("segment",)) if n_p > 1 else t


@register("val_table_features_both", "dashboard",
          "Table: cycle-feature agreement, own and aligned frames", **G)
def table_features_both(req: ViewRequest) -> TableSpec:
    """Cycle-feature bias with its CI, CCC and the equivalence margin in each system's own
    frames and after the frame alignment, side by side (one unit per row)."""
    from biomech.viz.views.validation import FEATURE_UNIT
    from biomech.viz.views.validation_part_i import missing_cycle_notes

    ctx = req.ctx
    own = _table(ctx, "ref_agreement", "features")
    al = _table(ctx, "ref_agreement", "features_aligned")
    keys = ["pid", "task", "feature"]
    cols = ["bias", "bias_ci_lo", "bias_ci_hi", "ccc", "equiv_margin"]
    if own.empty or al.empty:
        return B.frame_table(pd.DataFrame(), "val_table_features_both", "no feature tables")
    m = own[keys + ["unit"] + cols].merge(al[keys + cols], on=keys, how="outer",
                                          suffixes=("", "__al"))
    order = {t: i for i, t in enumerate(V.TASKS)}
    m = m.sort_values(["pid", "task"], key=lambda c: c.map(lambda v: order.get(v, 99))
                      if c.name == "task" else c, kind="stable")

    def num(v: Any, d: int) -> str:
        return "–" if v is None or not np.isfinite(float(v)) else \
            f"{float(v):.{d}f}".replace("-", "−")

    rows = []
    for r in m.to_dict("records"):
        d = 1 if r["unit"] == "mm/s" else 2
        row = {"pid": f"P{ctx.pseudonym(r['pid'])}", "task": V.phase_label(r["task"]),
               "feature": B.reader_cell("feature", r["feature"]),
               "feature_unit": FEATURE_UNIT.get(str(r["unit"]), str(r["unit"]))}
        for sfx in ("", "__al"):
            row["bias" + sfx] = (f"{num(r['bias' + sfx], d)} [{num(r['bias_ci_lo' + sfx], d)}, "
                                 f"{num(r['bias_ci_hi' + sfx], d)}]")
            row["ccc" + sfx] = num(r["ccc" + sfx], 2)
            row["margin" + sfx] = num(r["equiv_margin" + sfx], 2)
        rows.append(row)
    # the n = 2 rule: per task and feature the descriptive mean of the participants,
    # only where every participant has the row
    n_p = m["pid"].nunique()
    if n_p > 1:
        for (task, feat), g in m.groupby(["task", "feature"], sort=False):
            if g["pid"].nunique() < n_p:
                continue
            d = 1 if g["unit"].iloc[0] == "mm/s" else 2
            row = {"pid": f"mean of {n_p}", "task": V.phase_label(task),
                   "feature": B.reader_cell("feature", feat),
                   "feature_unit": FEATURE_UNIT.get(str(g["unit"].iloc[0]),
                                                    str(g["unit"].iloc[0]))}
            for sfx in ("", "__al"):
                row["bias" + sfx] = num(g["bias" + sfx].mean(), d)
                row["ccc" + sfx] = num(g["ccc" + sfx].mean(), 2)
                row["margin" + sfx] = ""
            rows.append(row)
    columns = [Column("pid", "Participant", align="l"), Column("task", "Task", align="l"),
               Column("feature", "Feature", align="l"), Column("feature_unit", "Unit", align="l")]
    for grp, sfx in (("own frames", ""), ("frames aligned", "__al")):
        columns += [Column("bias" + sfx, "Bias (CI)", align="r", group=grp),
                    Column("ccc" + sfx, "CCC", align="r", group=grp),
                    Column("margin" + sfx, "Margin", align="r", group=grp)]
    t = TableSpec("val_table_features_both", "", columns, rows, width="text",
                  sources=["ref_agreement:features", "ref_agreement:features_aligned"])
    return attach(t, CaptionSpec(
        "cycle-feature agreement of JTS-MM and Qualisys, each system in its own anatomical "
        "frames and after the constant frame alignment",
        [("bias", "Bland–Altman mean difference JTS-MM − Qualisys with its 95 % CI (t-interval "
                  "with the effective number of cycles), in the unit of the row"),
         ("CCC", "Lin's concordance correlation"),
         ("margin", "smallest symmetric bound the CI lies inside (mm or °; none for the peak "
                    "speed, which has no pre-specified bound)")],
        n_text(ctx, own["pid"]),
        statistic="per participant and task over the cycles: mean difference with a "
                  "t-interval using the effective number of cycles; CCC over the paired cycle "
                  "values; mean rows: the descriptive mean of the participants (no interval)",
        notes=["Limits of agreement, ICC(A,1) and equivalence against the pre-specified "
               "bounds are in the separate tables.",
               *missing_cycle_notes(ctx, own, sorted(own["pid"].unique()))]))
