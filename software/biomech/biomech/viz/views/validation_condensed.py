# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Condensed systems-paper figures (IEEE Sensors, about eight main figures; 2026-11-05).

Each view composes panels of existing views into one figure for the paper; the originals stay
unchanged (supporting / supplement items and gallery forms):

* ``val_marking_combined`` -- the point-marking result in one figure: per-landmark distance
  error (``val_marking_accuracy`` (d)), its error budget (``val_budget`` (a)) and the frame
  errors from the marks vs from motion (``val_marking_frames``);
* ``val_motion_agreement`` -- the motion agreement in one figure: per-frame disagreement in
  own vs aligned frames (``val_kin_error_ecdf``), the motion error budget (``val_budget`` (b))
  and the cycle-feature biases against the bounds (``val_kin_agreement``);
* ``val_sync_combined`` -- the clock link in one figure: phase lane and calibration-tool
  visibility with its edges (``val_sync_overview`` (a)), the CT speed of both systems in the
  start window (``val_sync_windows``) and the residual lag over the take (``val_sync_lag``).

Tables that replace hand-built tables of the draft:

* ``val_table_integrity`` -- recording integrity per comparison take, from what the store keeps
  (dropped camera frames, frame-to-trigger assignment drift, valid poses of both systems);
* ``val_table_emg_bench`` -- the sEMG front end on the bench per channel (B1 impedance scale,
  B2 shorted-input noise; ``hardware/bench/tables``, read like the ``bench:`` numbers);
* ``val_table_fitness`` -- fitness for purpose in one table: the dataset's hinge models on both
  systems, the camera-free opening and the dataset's sampling chain, per participant + mean.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Column, FigureSpec, Panel, TableSpec
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import _empty, _pid, _table, _who, n_text

G = {"group": "validation", "domain": "validation"}


def _cap(spec: Any) -> dict[str, Any]:
    """The caption parts of a built view (``attach`` keeps them in ``meta["caption"]``)."""
    return dict((getattr(spec, "meta", None) or {}).get("caption") or {})


def _letter(panels: list[Panel]) -> None:
    """(a), (b), ... in order, over whatever letter or title the source view gave."""
    for i, p in enumerate(panels):
        _, t = p.heading()
        p.title, p.tag = f"({chr(97 + i)}) {t}", ""


def _panel(spec: Any, start: str) -> Panel | None:
    """The panel of ``spec`` whose title (without its letter) starts with ``start``."""
    for p in getattr(spec, "panels", []):
        if p.heading()[1].startswith(start):
            return p
    return None


def _rows_from(dst: Panel | None, src: Panel | None) -> None:
    """Give ``dst`` the row labels of ``src`` (the source views share them across a row)."""
    if dst is not None and src is not None and src.y.ticklabels:
        dst.y.ticks, dst.y.ticklabels = list(src.y.ticks), list(src.y.ticklabels)


def _keys_from(dst: Panel | None, src: Panel | None) -> None:
    """Legend-only copies of ``src``'s legend entries on ``dst`` (its legend below)."""
    from biomech.viz.spec import Series

    if dst is None or src is None:
        return
    have = {s.label for s in dst.legend_entries()}         # never a key twice
    dst.series += [Series(s.kind, x=np.array([np.nan]), y=np.array([np.nan]), role=s.role,
                          style=dict(s.style), label=s.label) for s in src.legend_entries()
                   if s.label not in have]
    dst.legend = "below"


def _sources(*specs: Any) -> list[str]:
    return sorted({s for sp in specs for s in (getattr(sp, "sources", None) or [])})


@register("val_marking_combined", "dashboard", "Point marking in one figure", **G)
def marking_combined(req: ViewRequest) -> FigureSpec:
    """Point marking for the paper: (a) distance error per landmark with the reference noise,
    (b) its error budget per participant, (c, d) the frame errors from the marked points next
    to those estimated from motion."""
    from biomech.viz.views.validation import marking_accuracy, marking_frames
    from biomech.viz.views.validation_part_i import val_budget

    fid, title = "val_marking_combined", "Point marking: accuracy, budget and frames"
    acc, frm = marking_accuracy(req), marking_frames(req)
    bud = val_budget(req)
    a = _panel(acc, "|e|")
    b = _panel(bud, "Point marking")
    c, d = _panel(frm, "Frame angle"), _panel(frm, "Frame origin")
    _rows_from(a, _panel(acc, "Anterior"))           # (d) of the source shares (a)'s rows
    _rows_from(d, c)
    if a is not None:
        a.title = "Landmark difference |e|"
    if b is not None:
        b.title = "Error budget of the points"
    panels = [p for p in (a, b, c, d) if p is not None]
    if not panels:
        return _empty(fid, title, "no point-marking results")
    if a is not None:
        a.height = 1.0
    if b is not None:
        b.height = 1.0
    for p in (c, d):
        if p is not None:
            p.height = 0.5
    _letter(panels)
    spec = FigureSpec(fid, title, panels, rows=2, cols=2, width="text", height_in=2.3,
                      sources=_sources(acc, frm, bud))
    ca, cf, cb = _cap(acc), _cap(frm), _cap(bud)
    marks = [("black markers with whiskers (a)", "distance between the JTS-MM and the Qualisys "
                                                 "landmark with its 95 % CI"),
             ("grey markers (a)", "σ_Q of the Qualisys landmark (marker noise and tip SE)"),
             ("wide bar / narrow bar / black marker (b)", "observed difference / JTS-MM part / "
                                                          "Qualisys part "
                                                          "(σ_obs² = σ_JTS² + σ_Q²)"),
             ("black / grey markers (c, d)", "frame difference from the marked points (with its 95 % "
                                             "CI) / the constant frame offset estimated from "
                                             "motion"),
             ("circle / square", "P1 / P2")]
    return attach(spec, CaptionSpec(
        "point marking of JTS-MM against Qualisys: (a) distance of each JTS-MM landmark from "
        "the Qualisys landmark after the motion-based body registration (rows: L1–L3 / U1–U3 "
        "lower / upper incisor, left and right molar; TMJ L / R the condyles, '(head)' marked "
        "on the head piece); (b) the error budget of the teeth and condyle points; (c) rotation "
        "and (d) origin distance of the anatomical frames built from the marks, next to the "
        "frame offset found independently from the motion data; the differences along the "
        "anatomical axes are in the supplementary marking figure",
        marks, ca.get("n") or cf.get("n") or "",
        statistic="; ".join(x for x in (ca.get("statistic"), cb.get("statistic"),
                                          cf.get("statistic")) if x),
        evidence=ca.get("evidence") or cf.get("evidence") or "",
        notes=list(dict.fromkeys((ca.get("notes") or []) + (cf.get("notes") or [])))))


@register("val_motion_agreement", "dashboard", "Motion agreement in one figure", **G)
def motion_agreement(req: ViewRequest) -> FigureSpec:
    """Motion agreement for the paper: (a, b) per-frame disagreement in own vs aligned frames,
    (c) the motion error budget, (d–f) cycle-feature biases against the application bounds."""
    from biomech.viz.views.validation import kin_agreement
    from biomech.viz.views.validation_gallery import kin_error_ecdf
    from biomech.viz.views.validation_part_i import val_budget

    fid, title = "val_motion_agreement", "Motion agreement: frames, budget and cycle features"
    ec, bud, ag = kin_error_ecdf(req), val_budget(req), kin_agreement(req)
    mb = _panel(bud, "Motion: incisor")
    _keys_from(mb, _panel(bud, "Point marking"))     # the budget legend sits on its (a)
    if mb is not None:
        mb.title = "Error budget, motion"             # one line at a third of the width
    top = [p for p in (_panel(ec, "Incisor"), _panel(ec, "Rotation"), mb) if p is not None]
    low = list(getattr(ag, "panels", []))[:3]
    panels = top + low
    if not panels:
        return _empty(fid, title, "no motion-agreement results")
    for p in top:
        p.height = 1.0
    for p in low:
        p.height = 0.9
        p.legend = "none"
    _letter(panels)
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 3)), cols=3,
                      width="text", height_in=1.55, sources=_sources(ec, bud, ag))
    ce, cg, cb = _cap(ec), _cap(ag), _cap(bud)
    marks = [("steps (a, b)", "share of frames at or below a disagreement; color: frame form "
                              "(orange own frames, green frames aligned); line style: "
                              "participant (solid P1, dashed P2)"),
             ("wide bar / narrow bar / black marker (c)", "observed difference / JTS-MM part / "
                                                          "Qualisys part, at rest and in the "
                                                          "tasks"),
             ("markers with whiskers (d–f)", "bias JTS-MM − Qualisys with its 95 % CI, frames "
                                             "aligned; rows: participant, task and number of "
                                             "cycles; arrow with value: a CI end beyond the "
                                             "axis"),
             ("dashed lines", "application bound (2 mm, 1.2°)"),
             ("circle / square", "P1 / P2")]
    return attach(spec, CaptionSpec(
        "agreement of the jaw-in-head motion of JTS-MM and Qualisys in the movement tasks: "
        "(a, b) per-frame incisor and rotation disagreement, each system in its own anatomical "
        "frames and after the constant frame alignment (leave-one-task-out); (c) the error "
        "budget of the incisor and condyle motion split into the Qualisys and the JTS-MM part "
        "(the rotation budget is given in the Supplementary Material); (d–f) bias of the cycle "
        "features task "
        "amplitude, incisor excursion and rotation excursion, frames aligned",
        marks, ce.get("n") or cg.get("n") or "",
        statistic="; ".join(x for x in (cg.get("statistic"), cb.get("statistic")) if x),
        evidence=cg.get("evidence") or ce.get("evidence") or "",
        notes=list(dict.fromkeys((cg.get("notes") or []) + (ce.get("notes") or [])))))


SYNC_FORMS = ("both_lag", "one_take")


@register("val_sync_combined", "dashboard", "Clock link in one figure", params=("pid",),
          choices={"form": SYNC_FORMS}, form="both_lag", **G)
def sync_combined(req: ViewRequest) -> FigureSpec:
    """The clock link for the paper: phase lane and CT visibility with its edges, the CT
    angular speed of both systems in the start window after the clock fit (the example take
    ``pid``), and the residual lag of sliding windows over the take -- form ``both_lag``
    (default, 2026-11-05): of both takes on one axis (marker shape = participant), since the
    takes differ in the pre-specified clock checks; ``one_take``: of the example take only."""
    from biomech.viz.views import registry as R
    from biomech.viz.views.validation import sync_lag, sync_overview, sync_windows

    fid, title = "val_sync_combined", "Two systems on one clock"
    ctx = req.ctx
    pid = _pid(req)
    if pid is None:
        return _empty(fid, title, "no participants")
    ov, wn, lg = sync_overview(req), sync_windows(req), sync_lag(req)
    lane = next((p for p in ov.panels if not p.heading()[1] and not p.title), None)
    vis = _panel(ov, "CT visible")
    om = next((p for p in wn.panels if p.y.label == "|ω|"), None)
    rl = next((p for p in lg.panels if p.title.startswith("(a)")), None)
    if vis is None and om is None and rl is None:
        return _empty(fid, title, f"no sync results for {_who(ctx, pid)}")
    if vis is not None:
        vis.title, vis.height = "CT visible, with the detected edges", 0.75
    if om is not None:
        om.title, om.legend, om.height = "CT speed in the start window", "below", 0.6
        om.sharex = None
    others = [q for q in sorted(ctx.pids()) if q != pid] \
        if req.options.get("form", "both_lag") == "both_lag" else []
    if rl is not None:
        rl.title, rl.legend, rl.height = "Residual lag over the take", "below", 0.6
        if others:
            # the angular-speed lag (black in the one-take form) in the take's colour, the
            # pose lag grey; marker shape = participant as everywhere (X1)
            def take(series: list, j: int, who: str) -> list:
                out = []
                for s in series:
                    if s.kind not in ("errorbar", "scatter"):
                        continue
                    role = f"participant:{j}" if s.role == "accent" else s.role
                    out.append(dataclasses.replace(
                        s, role=role, label=f"{s.label}, {who}" if s.label else None))
                return out

            keep = [s for s in rl.series if s.kind not in ("errorbar", "scatter")]
            rl.series = take(rl.series, 0, _who(ctx, pid)) + keep
            for j, q in enumerate(others, start=1):
                lq = R.build("val_sync_lag", R.request(ctx, "val_sync_lag", pid=q))
                src = next((p for p in lq.panels if p.title.startswith("(a)")), None)
                rl.series += take(src.series if src is not None else [], j, _who(ctx, q))
            rl.title = "Residual lag over the take, both takes"
    lettered = [p for p in (vis, om, rl) if p is not None]
    _letter(lettered)
    panels = ([lane] if lane is not None else []) + lettered
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=1.35, sources=_sources(ov, wn, lg),
                      perspective=f"{_who(ctx, pid)}, comparison take", meta={"pid": pid})
    both = bool(others) and rl is not None
    ex = _who(ctx, pid)
    rest = ", ".join(_who(ctx, q) for q in others)
    co = _cap(ov)
    marks = [("lane", "recorded phases, sync windows outlined; L1–L3 / U1–U3: lower / upper "
                      "landmark holds, TMJ L / R: condyle holds"),
             ("thick bars (a)", "calibration tool (CT) visible in JTS-MM (blue, top) and "
                                "Qualisys (black, bottom) on the master clock"),
             ("up / down triangles (a)", "rising / falling visibility edges (rises drawn "
                                         "above falls)"),
             ("lines (b)", "CT angular speed of both systems after the clock fit (solid "
                           "JTS-MM, dashed Qualisys)"),
             ("markers with whiskers (c)", "lag of sliding 15-s windows minus the fitted affine "
                                           "clock (± SE), "
                                           + (f"from the angular speed vermillion circles {ex}, "
                                              f"sky-blue squares {rest}; grey from the registered "
                                              "poses (circle / square by take)" if both else
                                              "black from the angular speed, grey from the "
                                              "registered poses") + "; flat = the clock model "
                                           "holds"),
             ("dashed horizontal line (c)", "the fitted clock (zero residual lag)")]
    return attach(spec, CaptionSpec(
        "how JTS-MM and Qualisys share one clock without a hardware sync line: the calibration "
        "tool is waved in view of both systems at the start and the end of the take, and an "
        "affine clock (offset and drift) is fitted to its motion; the edge-matched coarse "
        "initialization is in the supplementary sync figure"
        + (f". (a, b) show the example take {ex}; the raw sync signals of {rest} are in the "
           f"supplement; (c) shows both takes" if both else ""),
        marks, (f"(a, b) {ex}; (c) {ex} and {rest}, comparison takes" if both
                else f"{_who(ctx, pid)}, comparison take"),
        statistic="raw signals of one take, no summary; lag per 15-s window with its standard "
                  "error",
        evidence=co.get("evidence") or "", notes=co.get("notes") or []))


# -- tables --------------------------------------------------------------------------------------
def _fmt(v: Any, d: int) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "–"
    return f"{x:.{d}f}" if np.isfinite(x) else "–"


def _mean_cell(vals: list[float], d: int) -> str:
    v = [x for x in vals if np.isfinite(x)]
    return _fmt(np.mean(v), d) if len(v) == len(vals) and v else "–"


def _pick(df: pd.DataFrame, col: str, **filt: Any) -> float:
    """The single value of ``col`` where every filter matches (a list matches any of it);
    NaN when absent or ambiguous."""
    if df.empty or col not in df:
        return float("nan")
    m = np.ones(len(df), bool)
    for k, v in filt.items():
        if k not in df:
            return float("nan")
        m &= df[k].isin(v if isinstance(v, (list, tuple)) else [v]).to_numpy()
    d = df.loc[m, col]
    return float(d.iloc[0]) if len(d) == 1 else float("nan")


def _valid_share(av: pd.DataFrame, pid: str, system: str) -> float:
    """Frames with a valid pose (both plates) over the movement tasks, n-weighted, %."""
    if av.empty:
        return float("nan")
    d = av[(av["pid"] == pid) & (av["system"] == system) & av["segment"].isin(V.TASKS)]
    return float(100 * np.sum(d["n"] * d["valid_frac"]) / d["n"].sum()) if len(d) else np.nan


@register("val_table_integrity", "dashboard", "Table: recording integrity per take", **G)
def table_integrity(req: ViewRequest) -> TableSpec:
    """Recording integrity of the comparison takes from what the validation store keeps; the
    recorder's own sample-loss counters are named in a note (not stored)."""
    ctx = req.ctx
    pids = list(ctx.pids())
    ft, gp = _table(ctx, "ref_diagnostics", "frame_timing"), _table(ctx, "ref_diagnostics",
                                                                   "frame_gaps")
    sy, av = _table(ctx, "ref_sync", "sync"), _table(ctx, "ref_quality", "availability")
    keys = ("dropped", "longest", "drift", "valid_jtsmm", "valid_qualisys")
    dec = {"dropped": 1, "longest": 0, "drift": 1, "valid_jtsmm": 1, "valid_qualisys": 1}
    rows, vals = [], {k: [] for k in keys}
    for pid in pids:
        n_int = _pick(ft, "n_intervals", pid=pid)
        miss = (float(np.sum(gp.loc[gp["pid"] == pid, "missing_frames"]
                             * gp.loc[gp["pid"] == pid, "n"])) if not gp.empty else np.nan)
        v = {"dropped": 100 * miss / (n_int + miss) if np.isfinite(n_int) else np.nan,
             "longest": _pick(ft, "long_max_ms", pid=pid),
             "drift": _pick(sy, "cam_assign_drift_ppm", pid=pid),
             "valid_jtsmm": _valid_share(av, pid, "jtsmm"),
             "valid_qualisys": _valid_share(av, pid, "qualisys")}
        for k in keys:
            vals[k].append(v[k])
        rows.append({"pid": _who(ctx, pid), **{k: _fmt(v[k], dec[k]) for k in keys}})
    if len(pids) > 1:
        rows.append({"_rule": True})
        rows.append({"pid": "Mean", **{k: _mean_cell(vals[k], dec[k]) for k in keys}})
    cols = [Column("pid", "Participant", align="l"),
            Column("dropped", "dropped (%)", align="r", group="JTS-MM camera frames"),
            Column("longest", "longest gap (ms)", align="r", group="JTS-MM camera frames"),
            Column("drift", "assignment drift (µs/s)", align="r",
                   group="JTS-MM camera frames"),
            Column("valid_jtsmm", "JTS-MM", align="r", group="Valid poses (%)"),
            Column("valid_qualisys", "Qualisys", align="r",
                   group="Valid poses (%)")]
    notes = ("Dropped: camera frames missing from the hardware-triggered sequence (intervals "
             "of two or three frame periods), share of the expected frames over the take. "
             "Longest gap: longest frame interval. Assignment drift: slope of the residual "
             "between the camera frames and the trigger pulses they are assigned to (cam0, "
             "the recorder's sync report). Valid poses: frames with both plates tracked on the "
             "movement tasks. Not stored in the validation results (checked at recording, "
             "not reported here): sEMG and IMU sample loss and the median frame-to-trigger "
             "assignment residual.")
    t = TableSpec("val_table_integrity", "", cols, rows, notes=notes, width="column",
                  sources=["ref_diagnostics:frame_timing", "ref_diagnostics:frame_gaps",
                           "ref_sync:sync", "ref_quality:availability"])
    return attach(t, CaptionSpec(
        "recording integrity of the two comparison takes: dropped JTS-MM camera frames, the "
        "drift of the frame-to-trigger assignment, and the share of valid poses of both "
        "systems on the movement tasks (columns defined below the table)", [],
        n_text(ctx, pids, "one take each"),
        statistic="per take: shares and extremes over the take; mean of the participants",
        evidence="descriptive"))


def _first(d: pd.DataFrame, c: str) -> Any:
    """The first value of column ``c`` (NaN when absent)."""
    return d[c].iloc[0] if len(d) and c in d else np.nan


def _bench(name: str) -> pd.DataFrame:
    from biomech.viz.numbers import bench_table

    return bench_table(name)


@register("val_table_emg_bench", "dashboard", "Table: sEMG front end on the bench", **G)
def table_emg_bench(req: ViewRequest) -> TableSpec:
    """The recorder's sEMG front end per channel: B1 impedance scale at the dataset's lead-off
    current (6 nA) and B2 shorted-input noise against the in-vivo pre-take checks."""
    imp, noi = _bench("emg_impedance_channels"), _bench("emg_noise_channels")
    tid = "val_table_emg_bench"
    if imp.empty and noi.empty:
        return TableSpec(tid, "", [Column("x", "–", align="l")], [{"x": "no bench tables"}])
    imp = imp[imp["current_na"] == 6] if not imp.empty else imp
    chans = sorted({str(c) for d in (imp, noi) if not d.empty for c in d["channel"]},
                   key=lambda c: (c == "all", c))
    rows = []
    for ch in chans:
        if ch == "all":
            rows.append({"_rule": True})
        i = imp[imp["channel"].astype(str) == ch] if not imp.empty else imp
        n = noi[noi["channel"].astype(str) == ch] if not noi.empty else noi
        k = (f"{_fmt(_first(i, 'k'), 2)} ({_fmt(_first(i, 'k_min'), 2)}–{_fmt(_first(i, 'k_max'), 2)})"
             if ch == "all" else _fmt(_first(i, "k"), 2))
        noise = (_fmt(_first(n, "noise_median_uv"), 1) if ch == "all" else
                 f"{_fmt(_first(n, 'noise_median_uv'), 1)} ({_fmt(_first(n, 'noise_min_uv'), 1)}–"
                 f"{_fmt(_first(n, 'noise_max_uv'), 1)})")
        rows.append({"channel": "All channels" if ch == "all" else f"CH{ch}",
                     "k": k, "err": _fmt(_first(i, "max_abs_err_pct"), 1),
                     "repeat": _fmt(_first(i, "repeat_max_pct"), 1), "noise": noise,
                     "limit": _fmt(_first(n, "limit_uv"), 0),
                     "vivo": _fmt(_first(n, "vivo_noise_median_uv"), 1),
                     "ratio": _fmt(_first(n, "ratio_bench_vivo"), 2)})
    cols = [Column("channel", "Channel", align="l"),
            Column("k", "scale k", align="r", group="B1 impedance (6 nA)"),
            Column("err", "max |error| (%)", align="r", group="B1 impedance (6 nA)"),
            Column("repeat", "repeat (%)", align="r", group="B1 impedance (6 nA)"),
            Column("noise", "bench (µV)", align="r", group="B2 noise RMS"),
            Column("limit", "check limit (µV)", align="r", group="B2 noise RMS"),
            Column("vivo", "in vivo (µV)", align="r", group="B2 noise RMS"),
            Column("ratio", "bench / in vivo", align="r", group="B2 noise RMS")]
    k24 = _bench("emg_impedance_channels")
    k24 = (k24[(k24["current_na"] == 24) & (k24["channel"].astype(str) == "all")]
           if not k24.empty else k24)
    notes = ("Scale k: the recorder's impedance |Z| against the multimeter value of known "
             "resistors (10–100 kΩ), least squares through the origin; all channels: median "
             "(range). Max |error|: largest relative error of a single capture. Repeat: largest "
             "difference between two captures of one resistor after re-seating. Noise RMS: root "
             "mean square (RMS) of the pre-take check's 20–450 Hz band with mains removed, "
             "inputs shorted; median (range) over the captures; all channels: median. In vivo: the same check before "
             "the participants' motion takes (median over participants). Bench / in vivo: "
             "median over the channels of the per-channel ratio"
             + (f". At 24 nA (not the dataset's current) the scale is "
                f"{_fmt(k24['k'].iloc[0], 2)}" if len(k24) else "") + ".")
    t = TableSpec(tid, "", cols, rows, notes=notes, width="text",
                  sources=["bench:emg_impedance_channels", "bench:emg_noise_channels"])
    n_vivo = int(noi.loc[noi["channel"].astype(str) == "all", "n_vivo"].iloc[0]) \
        if not noi.empty and (noi["channel"].astype(str) == "all").any() else 0
    return attach(t, CaptionSpec(
        "the recorder's sEMG front end on the bench, per channel: (B1) the impedance scale "
        "against known resistors at the dataset's lead-off current and (B2) the noise with "
        "shorted inputs next to the same check in vivo (columns defined below the table)", [],
        "no participants for B1 and the bench noise (hardware of the setup); in vivo: "
        f"{n_vivo} pre-take checks", statistic="per channel: least-squares scale, largest "
        "errors; medians and ranges over the captures", evidence="descriptive",
        notes=["The dataset's impedances are reported as recorded; the sensitivity to the "
               "per-channel scale is given in the text."]))


# quantity rows of the fitness table: (heading, label with unit, decimals (as the numbers),
# stage, table, column, filter); one value per participant
FITNESS: tuple[tuple[str, str, int, str, str, str, dict[str, Any]], ...] = (
    ("Hinge models of the dataset (open-close, incisor residual)",
     "best-fit fixed axis, JTS-MM (mm)", 2, "ref_hinge", "hinge", "hinge_fit_mm",
     {"task": "open_close", "system": "jtsmm"}),
    ("", "best-fit fixed axis, Qualisys (mm)", 2, "ref_hinge", "hinge", "hinge_fit_mm",
     {"task": "open_close", "system": "qualisys"}),
    ("", "axis through the condyles, JTS-MM (mm)", 2, "ref_hinge", "hinge", "hinge_tmj_mm",
     {"task": "open_close", "system": "jtsmm"}),
    ("", "axis through the condyles, Qualisys (mm)", 2, "ref_hinge", "hinge", "hinge_tmj_mm",
     {"task": "open_close", "system": "qualisys"}),
    ("Camera-free opening (pooled movement tasks, RMSE against Qualisys)",
     "opening, JTS-MM optics (°)", 2, "ref_camerafree", "cf_val_segments", "rmse_open_deg",
     {"segment": "movement_pooled", "estimator": "F0"}),
    ("", "opening, anchored IMU rotation (°)", 2, "ref_camerafree", "cf_val_segments",
     "rmse_open_deg", {"segment": "movement_pooled", "estimator": "E3b"}),
    ("", "opening, calibrated IMU rotation (°)", 2, "ref_camerafree", "cf_val_segments",
     "rmse_open_deg", {"segment": "movement_pooled", "estimator": "E2b"}),
    ("", "incisor, JTS-MM optics (mm)", 1, "ref_camerafree", "cf_val_segments", "rmse_inc_mm",
     {"segment": "movement_pooled", "estimator": "F0"}),
    ("", "incisor, per-user translation model (mm)", 1, "ref_camerafree", "cf_val_segments",
     "rmse_inc_mm", {"segment": "movement_pooled", "estimator": "E4own"}),
    ("Sampling chain of the dataset",
     "peak opening velocity, 40 Hz chain / 200 Hz", 2, "ref_sampling", "peak_ratio_open",
     "ratio_median", {"task": "open_close", "pipeline": "ref40",
                      "quantity": "peak_open_vel_dps"}),
    ("", "incisor RMS after the pose chain, aligned (mm)", 2, "ref_kinematics",
     "segments_filtered", "incisor_rms_mm",
     {"chain": "both", "frames": "aligned", "segment": "movement_pooled"}),
    ("", "rotation RMS after the pose chain, aligned (°)", 2, "ref_kinematics",
     "segments_filtered", "rot_rms_deg",
     {"chain": "both", "frames": "aligned", "segment": "movement_pooled"}),
)


@register("val_table_fitness", "dashboard", "Table: fitness for purpose", **G)
def table_fitness(req: ViewRequest) -> TableSpec:
    """Fitness for purpose of the dataset's analyses in one table, per participant + mean:
    the hinge models on both systems (VD21), the camera-free opening (VD20) and the dataset's
    sampling chain (VD9, VD12)."""
    ctx = req.ctx
    pids = list(ctx.pids())
    who = [_who(ctx, p) for p in pids]
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    rows, srcs = [], []
    for head, label, d, stage, table, col, filt in FITNESS:
        if (stage, table) not in cache:
            cache[(stage, table)] = _table(ctx, stage, table)
            srcs.append(f"{stage}:{table}")
        df = cache[(stage, table)]
        if head:
            rows.append({"quantity": head, "_head": True})
        v = [_pick(df, col, pid=p, **filt) for p in pids]
        rows.append({"quantity": label, **{f"p{i}": _fmt(x, d) for i, x in enumerate(v)},
                     "mean": _mean_cell(v, d)})
    cols = ([Column("quantity", "Quantity", align="X")]
            + [Column(f"p{i}", w, align="r") for i, w in enumerate(who)]
            + [Column("mean", "Mean", align="r")])
    notes = ("Hinge models: incisor RMS residual of the dataset's fixed-axis models over the "
             "same openings in both systems. Camera-free: the opening angle and incisor of the "
             "estimators that need no camera, against Qualisys; the optics (JTS-MM) as the "
             "comparator. Sampling chain: the dataset's 40 Hz pose chain run on Qualisys, its "
             "peak opening velocity over Qualisys at 200 Hz (median per-cycle ratio), and the "
             "agreement of both systems after the chain.")
    t = TableSpec("val_table_fitness", "", cols, rows, notes=notes, width="column",
                  sources=srcs)
    return attach(t, CaptionSpec(
        "fitness of JTS-MM for the analyses of the dataset: the hinge models on both systems, "
        "the camera-free opening against Qualisys and the effect of the dataset's sampling "
        "chain (definitions below the table)", [], n_text(ctx, pids, "one take each"),
        statistic="per participant: RMS / RMSE over the pooled movement tasks or the "
                  "open-close openings, median per-cycle ratio; mean of the participants",
        evidence="exploratory"))

