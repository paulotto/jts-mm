# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pseudo-online (simulated real-time) evaluation views (X4, exploratory; ml08).

The recorded sessions replayed in 20 ms chunks through a strictly causal chain (stage
``ml_online``): stream decoding accuracy and latency, causal release detection, compute time
per chunk and the agreement of stream features with the offline windows. No exoskeleton is
involved: these views show intent detection only, without closed-loop effects.
"""

from __future__ import annotations

import numpy as np

from biomech.results import COHORT_UNIT
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register

CLASSES = ("REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT")
MODE_LABEL = V.labels("calibration_mode")
SIM_NOTE = ("Simulated real time: recorded streams replayed in 20-ms chunks through the "
            "causal chain; no exoskeleton and no closed loop")
NOTE = SIM_NOTE + " (exploratory)."


def _summary(ctx):
    s = ctx.table("ml_online", "summary")
    return s[s.get("applicable", True).astype(bool)] if not s.empty and "applicable" in s \
        else s


def _rows(s):
    """(label, frame) per feature set x calibration mode, in a fixed order."""
    out = []
    for fs in [f for f in ("emg_imu", "imu", "emg") if f in set(s.feature_set)]:
        for mode in [m for m in MODE_LABEL if m in set(s["mode"])]:
            g = s[(s.feature_set == fs) & (s["mode"] == mode)]
            if not g.empty:
                out.append((f"{V.FSET_LABEL.get(fs, fs)}, {MODE_LABEL[mode]}", g))
    return out


@register("online_accuracy", "dashboard", "Simulated real-time decoding", domain="ml")
def online_accuracy(req: ViewRequest) -> FigureSpec:
    """Stream balanced accuracy, decision latency and false activations per configuration."""
    ctx = req.ctx
    s = _summary(ctx)
    rows = _rows(s) if not s.empty else []
    lab = [r[0] for r in rows]
    p1 = B.strip(lab, [g.ba_raw.to_numpy(float) for _, g in rows], "Balanced accuracy (stream)",
                 Axis("Balanced accuracy", lim=(0.0, 1.0)), ref=1 / 7, ref_label="chance")
    off = [g.balanced_accuracy_offline.to_numpy(float) for _, g in rows]
    for i, v in enumerate(off):
        v = v[np.isfinite(v)]
        if v.size:
            p1.series.append(Series("scatter", np.array([np.median(v)]), np.array([float(i)]),
                                    role="cat:1", style={"ms": "s", "size": 1.2}))
    budget = B.latency_budget_ms(ctx)
    p2 = B.strip(lab, [g.latency_ms_median.to_numpy(float) for _, g in rows],
                 "Decision latency after the label onset", Axis("Latency", V.MSEC), ref=budget,
                 ref_label=f"{budget:g} ms")
    p3 = B.strip(lab, [g.fa_per_min_rest.to_numpy(float) for _, g in rows],
                 "False activations at rest", Axis("False activations", V.PER_MIN))
    dec = ctx.table("ml_online", "decisions")
    lat = dec.latency_ms.to_numpy(float) if not dec.empty and "latency_ms" in dec else []
    p4 = B.histogram(lat, "Latency per detected onset (all configurations)",
                     Axis("Latency", V.MSEC), observed=budget, observed_label=f"{budget:g} ms")
    return FigureSpec("online_accuracy", "Simulated real-time decoding", [p1, p2, p3, p4],
                      rows=2, cols=2, width="text", height_in=max(1.5, 0.22 * len(lab)),
                      caption="Per participant (dots; diamond = median) and decoder "
                              "configuration: balanced accuracy of the stream decisions "
                              "(7 classes; square = offline decoding median), latency from the "
                              "optical label onset to the first "
                              "decision entering that class, false activations per minute at "
                              "rest; the histogram pools every scored onset. " + NOTE,
                      sources=["ml_online:summary", "ml_online:decisions"])


def _online_pids(ctx) -> list[str]:
    try:
        a = ctx.arrays("ml_online", COHORT_UNIT)
    except Exception:
        return []
    return [p for p in ctx.pids() if f"online__{p}__t_us" in a]


@register("online_timeline", "dashboard", "Simulated real-time decision timeline", domain="ml",
          params=("pid",))
def online_timeline(req: ViewRequest) -> FigureSpec:
    """True class, raw and smoothed stream decisions, and causal release decisions over time."""
    ctx = req.ctx
    pids = _online_pids(ctx)
    pid = req.pid if req.pid in pids else (pids[0] if pids else None)
    a = ctx.arrays("ml_online", COHORT_UNIT)
    p = Panel("Intent class over time", Axis("Time from stream start", V.SEC), Axis(""), [])
    # Never substitute silently: say whose stream is shown instead of the requested one.
    sub = ""
    if req.pid is not None and req.pid != pid:
        asked = (f"P{ctx.pseudonym(req.pid)}" if req.pid in ctx.pids()
                 else "The requested participant")
        sub = (f"{asked} has no data; showing P{ctx.pseudonym(pid)}" if pid is not None
               else f"{asked} has no data")
        p.note = sub
    if pid is not None:
        # Decisions at their emission time; the window ends mark where each decision's data
        # stops (their gap is the emission delay, <= 10 ms in the synthetic study).
        t = np.asarray(a[f"online__{pid}__t_us"], float)
        t_end = np.asarray(a.get(f"online__{pid}__t_end_us", t), float)
        t0 = t.min() if t.size else 0.0
        x = (t - t0) * 1e-6
        ph = np.asarray(a.get(f"online__{pid}__phase", np.zeros(t.size)))
        cut = np.flatnonzero(np.diff(ph.astype(float)) != 0) + 1 if ph.size else []
        edges = np.r_[0, cut, t.size] if t.size else np.array([], int)
        for k, (i0, i1) in enumerate(zip(edges[:-1], edges[1:])):
            if k % 2 == 1 and i1 > i0:                 # every other phase shaded
                p.series.append(Series("hspan", x=np.array([(t_end[i0] - t0) * 1e-6]),
                                       y2=np.array([(t_end[i1 - 1] - t0) * 1e-6]),
                                       role="faint"))
        for key, lab, role, st in (("y", "label (optical)", "context", {"width": 2.0}),
                                   ("yhat_raw", "decision (raw)", "cat:1", {}),
                                   ("yhat", "decision (smoothed)", "cat:0", {})):
            y = np.array(a.get(f"online__{pid}__{key}", np.full(t.size, np.nan)), float)
            y[y < 0] = np.nan
            kind = "line" if key == "y" else "scatter"
            p.series.append(Series(kind, x, y + (0.0 if key == "y" else
                                                 0.12 if key == "yhat" else -0.12),
                                   label=lab, role=role, style={**st, "size": 0.6}))
        rel = np.asarray(a.get(f"online__{pid}__release_us", []), float)
        if rel.size:
            p.series.append(Series("vline", x=(rel - t0) * 1e-6, role="reference",
                                   label="causal release decision"))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(CLASSES))], list(CLASSES)
    who = f"P{ctx.pseudonym(pid)}" if pid is not None else "none"
    return FigureSpec("online_timeline", "Simulated real-time decision timeline", [p],
                      width="text", height_in=2.4,
                      caption=(f"{sub}. " if sub else "") +
                      f"Participant {who}: optical intent label, the decoder's raw "
                              "window decisions and after smoothing (majority of 3 windows with "
                              "hysteresis), offset vertically for visibility; dashed: causal "
                              "closer-release decisions; shading alternates with the protocol "
                              "phases. " + NOTE,
                      sources=["ml_online:arrays"], meta={"pid": pid, "substituted": bool(sub)})


@register("online_compute", "dashboard", "Simulated real-time compute budget", domain="ml")
def online_compute(req: ViewRequest) -> FigureSpec:
    """Compute time per 20-ms chunk (median, P99) and the real-time factor per participant."""
    s = _summary(req.ctx)
    one = s.drop_duplicates("pid") if not s.empty else s
    # one-line titles with panel letters; the reference lines named in the caption (their
    # in-plot labels spilled into the gutter); "largest" as in jnl_online
    p1 = B.strip(["median", "P99", "largest"],
                 [one[c].to_numpy(float) if c in one else [] for c in
                  ("chunk_ms_median", "chunk_ms_p99", "chunk_ms_max")],
                 "(a) Compute per chunk", Axis("Time", V.MSEC), ref=20.0)
    p2 = B.strip(["real-time factor"], [one.realtime_factor.to_numpy(float)
                                         if "realtime_factor" in one else []],
                 "(b) Compute / signal time", Axis("Factor"), ref=1.0)
    p3 = B.strip(["calibration", "largest emission delay"],
                 [one[c].to_numpy(float) if c in one else [] for c in
                  ("calibration_ms", "emit_delay_ms_max")],
                 "(c) One-off and emission delays", Axis("Time", V.MSEC))
    spec = FigureSpec("online_compute", "Simulated real-time compute budget", [p1, p2, p3],
                      rows=1, cols=3, width="text", height_in=1.3, sources=["ml_online:summary"])
    return attach(spec, CaptionSpec(
        "(a) single-thread compute per 20-ms chunk on the analysis PC; (b) its ratio to "
        "signal time (< 1: faster than real time); (c) the one-off calibration after the rest "
        "phase and the largest delay between a window's end and its decision",
        [("dots", "participants"), ("diamonds", "median"),
         ("dashed line", "the chunk length (20 ms) in (a), real time in (b)")],
        f"n = {len(one)} participants",
        "per participant the median, the 99th percentile, and the maximum over the chunks of "
        "the replay",
        evidence="exploratory", notes=[SIM_NOTE]))


@register("online_release", "dashboard", "Causal release detection, stream vs offline",
          domain="timing")
def online_release(req: ViewRequest) -> FigureSpec:
    """X3 on the causal chain vs the offline (zero-phase) chain, per participant."""
    ctx = req.ctx
    r = ctx.table("ml_online", "release")
    r = r[r.task == "open_close"] if not r.empty and "task" in r else r
    pids = [p for p in ctx.pids() if not r.empty and (r.pid == p).any()]
    lab = B.pseudo_rows(ctx, pids)
    r = r.set_index("pid").reindex(pids) if pids else r
    panels = []
    for col, ttl, ax in (("detected_frac", "(a) Openings detected", Axis("Share", lim=(0, 1))),
                         ("anticipation_ms", "(b) Anticipation", Axis("Time", V.MSEC)),
                         ("false_per_min", "(c) False releases", Axis("Rate", V.PER_MIN))):
        a = r[f"{col}_offline"].to_numpy(float) if pids else np.array([])
        b = r[col].to_numpy(float) if pids else np.array([])
        panels.append(B.dumbbell(lab, a, b, ttl, ax, label_a="offline chain",
                                 label_b="causal stream"))
    spec = FigureSpec("online_release", "Causal release detection, stream vs offline", panels,
                      rows=1, cols=3, width="text", height_in=max(1.4, 0.18 * len(pids)),
                      sources=["ml_online:release"])
    return attach(spec, CaptionSpec(
        "open-close: the causal closer-release detector on the offline (zero-phase) chain and "
        "on the causal stream, per participant: (a) share of openings detected, (b) decision "
        "time before the opening onset, (c) false releases per minute",
        [("blue circles / orange squares", "offline chain / causal stream, per participant"),
         ("lines", "join a participant's two values"),
         ("a square alone", "both chains give the same value (the circle lies under it)")],
        n=f"n = {len(pids)} participants",
        statistic="per participant the share of openings detected and the median decision "
                  "time over cycles", evidence="exploratory", notes=[SIM_NOTE]))


@register("online_feature_deviation", "dashboard", "Stream vs offline features", domain="ml")
def online_feature_deviation(req: ViewRequest) -> FigureSpec:
    """How far the stream window features are from the offline ml_windows features."""
    ctx = req.ctx
    d = ctx.table("ml_online", "feature_deviation")
    rows = [p for p in ctx.pids() if not d.empty and (d.pid == p).any()]
    p1 = B.matrix_heat(d, "pid", "block", "median_abs_z", "Median |Δ| (SD units)",
                       rows=rows, row_labels=B.pseudo_rows(ctx, rows), fmt="{:.3f}")
    p2 = B.matrix_heat(d, "pid", "block", "p99_abs_z", "P99 |Δ| (SD units)",
                       rows=rows, row_labels=B.pseudo_rows(ctx, rows), fmt="{:.2f}")
    return FigureSpec("online_feature_deviation", "Stream vs offline features", [p1, p2],
                      rows=1, cols=2, width="text", height_in=max(1.3, 0.2 * len(rows)),
                      caption="Absolute difference between the stream and the offline window "
                              "features in units of the feature SD, per participant and block "
                              "(continuous causal filter vs the offline per-phase restart; "
                              "differences concentrate in the first windows of a phase, ml08).",
                      sources=["ml_online:feature_deviation"])
