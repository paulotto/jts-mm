# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Timing and intent-cue results (stages ``timing`` and ``stats_timing``) as dashboards.

One sign convention throughout: times are *before* the earliest mechanical onset when
positive (EMG lead, causal margin), as in Fig. 2c. Single-cycle onsets per detector and
onset rule (C2, D6) are rasters per participant; the causal closer-release detector and
its parameter sweep are exploratory (X3), the ensemble events too (X2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomech.timing.lead import DETECTORS, ONSET_RULES
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register

TIMING_TASKS = ("open_close", "chewing")
DIRECTIONS = ("opening", "closing")
DET_LABEL = V.labels("detector")
RULE_LABEL = V.labels("rule")
RULE_SHORT = V.labels("rule", "short")
AGONIST = {"opening": "digastric", "closing": "masseter, temporalis"}
LEAD = Axis("EMG lead before motion onset", V.MSEC)


def _rules(ctx) -> list[str]:
    """Onset rules, the primary one (``ctx.onset_rule()``) first."""
    prim = ctx.onset_rule()
    return [prim] + [r for r in ONSET_RULES if r != prim]


def timing_params(ctx, block: str) -> dict:
    """``stages.timing.<block>`` (``release``: the causal detector, ``ensemble``: X2,
    ``c2e``: X6) as the timing stage ran it, with the plan defaults where the config does not
    say (C2-E: those of :class:`biomech.stats.cohort_c2e.CohortParams`)."""
    defaults = {"release": {"drop_db": 3.0, "hold_s": 0.03},
                "ensemble": {"rise_frac": 0.25, "release_frac": 0.5}, "c2e": {}}[block]
    try:
        got = dict((ctx.config.params("timing") or {}).get(block) or {})
    except (AttributeError, TypeError):
        got = {}
    return {**defaults, **got}


def release_setting(ctx) -> str:
    """'3 dB below its lagged baseline for 30 ms' of the pre-set (primary) setting."""
    r = timing_params(ctx, "release")
    return f"{float(r['drop_db']):g} dB below its lagged baseline for {1e3 * float(r['hold_s']):g} ms"


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    for k, v in kw.items():
        if df.empty or k not in df:
            return df.iloc[:0]
        df = df[df[k] == v]
    return df


@register("timing_onsets", "dashboard", "Single-cycle EMG onsets", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS,
          choices={"detector": DETECTORS, "rule": _rules})
def timing_onsets(req: ViewRequest) -> FigureSpec:
    """EMG onset of every cycle relative to the mechanical onset, per participant."""
    ctx, task = req.ctx, req.phase or "open_close"
    det = req.options.get("detector") or DETECTORS[0]
    rule = req.options.get("rule") or ctx.onset_rule()
    on = _sel(ctx.table("timing", "onsets"), task=task, detector=det, rule=rule)
    pids = [p for p in ctx.pids() if not on.empty and (on.pid == p).any()]
    rows = B.pseudo_rows(ctx, pids)
    panels = []
    for d in DIRECTIONS:
        g = _sel(on, direction=d)
        ok, bad = g[g.usable.astype(bool)] if not g.empty else g, \
            g[~g.usable.astype(bool)] if not g.empty else g
        p = B.strip(rows, [ok.loc[ok.pid == q, "lead_ms"] for q in pids],
                    f"{d.capitalize()} ({AGONIST[d]})", LEAD, ref=0.0)
        for i, q in enumerate(pids):
            v = bad.loc[bad.pid == q, "lead_ms"].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                p.series.append(Series("scatter", v, np.full(v.size, float(i)), role="faint"))
        if g.empty:
            p.note = "no cycles"
        else:
            p.note = f"usable {len(ok)}/{len(g)} channel-cycles"
        panels.append(p)
    return FigureSpec("timing_onsets", "Single-cycle EMG onsets", panels, rows=1, cols=2,
                      width="text", height_in=max(1.3, 0.22 * len(pids) + 0.6),
                      caption=f"{V.task_label(task)}: lead of the single-cycle EMG onset of "
                              f"the agonists over the earliest mechanical onset (> 0: EMG "
                              f"first), detector {DET_LABEL.get(det, det)}, onset taken as the "
                              f"{RULE_LABEL.get(rule, rule)}"
                              + (" (primary rule)" if rule == ctx.onset_rule() else "")
                              + ". Dots: usable channel-cycles, diamond: participant median; light grey dots: onsets not usable "
                                "(active at the window start, weak burst or gaps).",
                      sources=["timing:onsets"])


@register("timing_release", "dashboard", "Causal closer release", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS)
def timing_release(req: ViewRequest) -> FigureSpec:
    """Per-cycle causal release decisions and detection rate vs chance (X3, exploratory)."""
    ctx, task = req.ctx, req.phase or "open_close"
    rel = _sel(ctx.table("timing", "release"), task=task)
    rs = ctx.table("timing", "release_summary")
    rs = rs[(rs.task == task) & rs.primary.astype(bool)] if not rs.empty else rs
    pids = [p for p in ctx.pids() if not rel.empty and (rel.pid == p).any()]
    rows = B.pseudo_rows(ctx, pids)
    pa = B.strip(rows, [rel.loc[rel.pid == q, "anticipation_ms"] for q in pids],
                 "Release decision per opening",
                 V.axis("anticipation_ms", "Decision before motion onset"), ref=0.0)
    if not rel.empty:
        pa.note = f"detected {int(rel.detected.sum())}/{len(rel)} openings"
    else:
        pa.note = "no openings"
    s = rs.set_index("pid").reindex(pids) if not rs.empty else pd.DataFrame(index=pids)
    det = 100 * s.get("detected_frac", pd.Series(np.nan, index=pids)).to_numpy(float)
    ch = 100 * s.get("chance_detected_frac", pd.Series(np.nan, index=pids)).to_numpy(float)
    pb = B.dumbbell(rows, det, ch, "Detection rate vs chance", Axis("Openings detected", V.PCT,
                                                                    lim=(0.0, 100.0)),
                    label_a="real onsets", label_b="random pseudo-onsets")
    if not np.isfinite(ch).any():
        pb.note = "chance level not available (task too short)"
    fpm = s.get("false_per_min", pd.Series(dtype=float)).to_numpy(float)
    if np.isfinite(fpm).any():
        pb.note = (pb.note + "; " if pb.note else "") + \
            f"false releases {np.nanmedian(fpm):.1f}/min (median)"
    return FigureSpec("timing_release", "Causal closer release", [pa, pb], rows=1, cols=2,
                      width="text", height_in=max(1.3, 0.22 * len(pids) + 0.6),
                      caption=f"{V.task_label(task)}: a causal detector of the closer release "
                              f"(group envelope {release_setting(ctx)}, the "
                              "pre-set parameters) per opening: time of its decision before "
                              "the mechanical opening onset (> 0: before), and per participant "
                              "the share of openings detected against the rate at random "
                              "pseudo-onsets ≥ 1 s away from real ones (chance). "
                              "Exploratory.",
                      sources=["timing:release", "timing:release_summary"])


def _cb(z: np.ndarray, label: str, unit: str, diverging: bool) -> Colorbar:
    """Colorbar with a non-degenerate range (symmetric about 0 when diverging)."""
    v = z[np.isfinite(z)]
    if diverging:
        m = max(float(np.abs(v).max()) if v.size else 1.0, 1.0)
        return Colorbar(label, unit, lim=(-m, m), ramp="diverging")
    lo, hi = (float(v.min()), float(v.max())) if v.size else (0.0, 1.0)
    return Colorbar(label, unit, lim=(lo, hi) if hi > lo else (lo - 1.0, hi + 1.0))


def _grid(df: pd.DataFrame, value: str, drops: list[float], holds: list[float]) -> np.ndarray:
    if df.empty or value not in df:
        return np.full((len(drops), len(holds)), np.nan)
    piv = df.pivot_table(index="drop_db", columns="hold_s", values=value, aggfunc="median")
    return piv.reindex(index=drops, columns=holds).to_numpy(float)


@register("timing_release_sweep", "dashboard", "Release detector sweep", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS)
def timing_release_sweep(req: ViewRequest) -> FigureSpec:
    """Detection above chance and causal margin over the drop x hold sweep (X3)."""
    ctx, task = req.ctx, req.phase or "open_close"
    rs = _sel(ctx.table("timing", "release_summary"), task=task)
    rc = _sel(ctx.table("stats_timing", "release_cohort"), task=task)
    drops = sorted(set(rs.drop_db)) if not rs.empty else []
    holds = sorted(set(rs.hold_s)) if not rs.empty else []
    has_chance = not rs.empty and rs.chance_detected_frac.notna().any()
    d = rs.assign(v=100 * (rs.detected_frac - rs.chance_detected_frac) if has_chance
                  else 100 * rs.detected_frac) if not rs.empty else rs
    rows = [f"{x:g} dB" for x in drops]
    cols = [f"{1e3 * h:.0f} ms" for h in holds]
    za, zb = _grid(d, "v", drops, holds), _grid(rc, "median", drops, holds)
    pa = B.heat(za, rows, cols, "Detected − chance (median)" if has_chance
                else "Detected (median)", cb=_cb(za, "Openings", V.PCT, has_chance))
    pb = B.heat(zb, rows, cols, "Causal margin (median)",
                # signed (negative: after the onset): a diverging map about 0 when the
                # margins change sign
                cb=_cb(zb, "Before onset", V.MSEC,
                       bool(np.nanmin(zb) < 0 < np.nanmax(zb)) if np.isfinite(zb).any()
                       else False))
    prim = rs[rs.primary.astype(bool)] if not rs.empty else rs
    for p in (pa, pb):
        p.x.label, p.y.label = "Hold time", "Drop below baseline"
        if not drops:
            p.note = "no data"
        elif not prim.empty:
            i, j = drops.index(prim.drop_db.iloc[0]), holds.index(prim.hold_s.iloc[0])
            p.series[0].text[i * len(holds) + j] += " *"
    if not has_chance and drops:
        pa.note = "chance level not available"
    n = rs.pid.nunique() if not rs.empty and "pid" in rs else 0
    spec = FigureSpec("timing_release_sweep", "Release detector sweep", [pa, pb], rows=1,
                      cols=2, width="text", height_in=1.6,
                      sources=["timing:release_summary", "stats_timing:release_cohort"])
    return attach(spec, CaptionSpec(
        f"{V.task_label(task)}: sensitivity of the causal closer-release detector to its drop "
        "threshold and hold time. (a) Median over participants of the share of openings "
        "detected minus the chance rate at random pseudo-onsets; (b) cohort median of the "
        "per-participant median time of the decision before the opening onset",
        [("*", "pre-set setting"), ("cell colour and text", "the value of each setting")],
        f"n = {n} participants",
        "per participant the share detected (or the median decision time); median over the "
        "participants", evidence="exploratory"))


@register("timing_usability", "dashboard", "Onset usability", domain="timing")
def timing_usability(req: ViewRequest) -> FigureSpec:
    """Share of usable single-cycle onsets and of channels active at the window start."""
    ctx = req.ctx
    u = ctx.table("stats_timing", "timing_usability")
    keys = [(d, r) for d in DETECTORS for r in _rules(ctx)]
    cols = [(t, d) for t in TIMING_TASKS for d in DIRECTIONS]
    if not u.empty:
        u = u.assign(row=u.detector + "|" + u.rule, col=u.task + "|" + u.direction)
    rl = [f"{DET_LABEL.get(d, d)}, {RULE_LABEL.get(r, r)}" for d, r in keys]
    cl = [f"{V.task_label(t)} {d}" for t, d in cols]
    panels = [B.matrix_heat(u, "row", "col", val, title, rows=["|".join(k) for k in keys],
                            cols=["|".join(c) for c in cols], row_labels=rl,
                            col_labels=cl, scale=100.0, cb=Colorbar("Share", V.PCT, lim=(0, 100)))
              for val, title in (("usable_frac", "Usable onsets (median)"),
                                 ("active_at_start", "Active at window start (median)"))]
    return FigureSpec("timing_usability", "Onset usability", panels, rows=2, cols=1,
                      width="text", height_in=max(1.6, 0.17 * len(keys) + 0.6),
                      caption="Per detector and onset rule: median over participants of the "
                              "share of channel-cycles with a usable EMG onset, and of those "
                              "whose detector signal is already above threshold at the start "
                              "of the search window (onset not identifiable).",
                      sources=["stats_timing:timing_usability"])


@register("timing_cohort", "dashboard", "Cohort EMG lead", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS)
def timing_cohort(req: ViewRequest) -> FigureSpec:
    """Cohort median lead per detector and onset rule, and the ensemble events (X2)."""
    ctx, task = req.ctx, req.phase or "open_close"
    tc = _sel(ctx.table("stats_timing", "timing_cohort"), task=task, metric="lead_ms")
    ec = _sel(ctx.table("stats_timing", "ensemble_cohort"), task=task)
    keys = [(d, r) for d in DETECTORS for r in _rules(ctx)]
    lab = [f"{DET_LABEL.get(d, d)}, {RULE_SHORT.get(r, r)}"
           + (" *" if r == ctx.onset_rule() and d == "tkeo" else "") for d, r in keys]
    panels = []
    for dr in DIRECTIONS:
        g = _sel(tc, direction=dr)
        s = (g.set_index(["detector", "rule"]).reindex(keys) if not g.empty
             else pd.DataFrame(index=pd.MultiIndex.from_tuples(keys),
                               columns=["median", "ci_lo", "ci_hi"], dtype=float))
        p = B.forest(lab, s["median"].to_numpy(float), s.ci_lo.to_numpy(float),
                     s.ci_hi.to_numpy(float), f"{dr.capitalize()} ({AGONIST[dr]})", LEAD)
        if g.empty:
            p.note = "no data"
        panels.append(p)
    ev = [(d, r) for d in DIRECTIONS for r in ("agonist", "antagonist")]
    s = (ec.set_index(["direction", "role"]).reindex(ev) if not ec.empty else
         pd.DataFrame(index=pd.MultiIndex.from_tuples(ev),
                      columns=["median", "ci_lo", "ci_hi", "n_before", "n_found"], dtype=float))
    elab = [f"{d[:4]}. {'rise' if r == 'agonist' else 'release'}"
            + (f" {s.loc[(d, r), 'n_before']:.0f}/{s.loc[(d, r), 'n_found']:.0f}"
               if np.isfinite(s.loc[(d, r), "n_found"]) else "") for d, r in ev]
    pe = B.forest(elab, s["median"].to_numpy(float), s.ci_lo.to_numpy(float),
                  s.ci_hi.to_numpy(float), "Ensemble events",
                  Axis("Before onset", V.MSEC))
    if ec.empty:
        pe.note = "no data"
    panels.append(pe)
    ens_p = timing_params(ctx, "ensemble")
    return FigureSpec("timing_cohort", "Cohort EMG lead", panels, rows=1, cols=3, width="text",
                      height_in=max(1.6, 0.17 * len(keys) + 0.7),
                      caption=f"{V.task_label(task)}: (a, b) cohort median (diamond) and 95 % BCa CI of "
                              "the per-participant lead of the single-cycle agonist onset over "
                              "the earliest mechanical onset, per detector and onset rule (* "
                              "primary: TKEO with the primary rule; driving: the burst that "
                              "drives the movement; in-task thr.: driving burst with the "
                              "threshold from the cycle's own pre-movement samples); (c) "
                              "events of the cycle-aligned "
                              f"envelopes ({100 * float(ens_p['rise_frac']):g} % agonist rise, "
                              f"{100 * float(ens_p['release_frac']):g} % antagonist release; "
                              "exploratory) of opening and closing, "
                              "with the number of participants whose event precedes the motion "
                              "of those where it was found.",
                      sources=["stats_timing:timing_cohort", "stats_timing:ensemble_cohort"])


@register("timing_cohort_table", "dashboard", "Cohort timing statistics (table)",
          params=("phase",), phase="open_close", domain="timing", phases=TIMING_TASKS)
def timing_cohort_table(req: ViewRequest) -> TableSpec:
    """Every timing metric of the cohort: median, CI and the one-sided signed-rank test."""
    ctx, task = req.ctx, req.phase or "open_close"
    tc = _sel(ctx.table("stats_timing", "timing_cohort"), task=task)
    cols = ["direction", "detector", "rule", "metric", "n", "median", "q1", "q3", "ci_lo",
            "ci_hi", "W", "p", "r"]
    t = B.frame_table(tc, "timing_cohort_table",
                      f"{V.task_label(task)}: cohort statistics of the per-participant timing "
                      "metrics (ms; > 0: before the mechanical onset) per direction, detector "
                      "and onset rule: median [IQR], 95 % BCa CI of the median, Wilcoxon "
                      "signed-rank test against 0 (one-sided, lead and margin only).",
                      columns=cols, width="text", ctx=ctx)
    t.sources = ["stats_timing:timing_cohort"]
    return t
