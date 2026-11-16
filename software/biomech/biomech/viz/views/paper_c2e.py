# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Views of the post hoc family C2-E (X6 (change log)): block C of the
dataset paper and dashboards of its per-cycle tables.

Sign convention of the analysis: a lead > 0 means EMG before the mechanical onset; on a time
axis an EMG event lies at -lead (ms; negative = before the motion event). Participants enter
a cohort row with at least ``min_usable`` usable channel-cycles (``stages.timing.c2e``), as in
the cohort tests; the motion events are the relative-quaternion angular-velocity onsets.

* ``c2e_sequence`` (paper): the event sequence per task -- closer release before the opening
  onset, the opening onset (0), the digastric onset, the closing onset and the closer onset
  before it -- every participant as a dot, the cohort median with its 95 % BCa CI. Forms:
  ``timeline`` (one axis from the opening onset; the closing onset at each participant's
  median opening duration), ``aligned`` (each EMG event against its own motion onset) and
  ``raincloud`` (the aligned rows as densities);
* ``c2e_locking`` (paper): (a) locking of the closers before closing (both tasks pooled):
  each participant's lead IQR against the median IQR of the circular-shift surrogates;
  (b) the EMG-to-acceleration lag per task (continuous coupling), residualised and raw.
  Forms ``dumbbell`` / ``slopegraph``;
* ``c2e_table`` (paper table): the cohort rows of ``stats_timing:c2e_cohort`` -- the primary
  hypothesis, the two gated release hypotheses (with the post hoc Holm-over-H1–H3
  sensitivity next to the gatekeeping), the supporting block, the structure rows
  (descriptive, cohort p withheld) and the decision;
* ``c2e_cycles`` (dashboard): every channel-cycle lead per participant (task, lead source);
* ``c2e_release`` (dashboard): every cycle's closer release and the release depth against the
  surrogates.

Stored results only (``timing:c2e_null``, ``c2e_cycles``, ``c2e_release``, ``coupling``,
``stats_timing:c2e_cohort``); the cohort median CI of an event time is the stats stages' BCa
routine (:func:`blocks.median_ci`); no test is computed here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.stats.cohort_c2e import CohortParams
from biomech.timing.c2e import C2EParams
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import forms
from biomech.viz.views import vocab as V
from biomech.viz.views.paper_journal import _ci, _f, _n, _peq, _sel, _t
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.results_timing import TIMING_TASKS, timing_params

G = {"group": "dataset"}
EVIDENCE = ("post hoc (specified before computing and validated on synthetic data before "
            "the real data were analysed)")
SOURCES = ["timing:c2e_null", "timing:c2e_cycles", "stats_timing:c2e_cohort"]
TIME_AX = Axis("Time from the opening onset", V.MSEC)
EVENT_AX = Axis("Time from the motion onset", V.MSEC)
LEAD_AX = Axis("EMG lead before the motion onset", V.MSEC)
IQR_AX = Axis("Lead IQR", V.MSEC, log=True)
LAG_AX = Axis("EMG-to-acceleration lag", V.MSEC)
SEQUENCE_FORMS = ("timeline", "aligned", "raincloud")
# Lead sources of c2e_cycles / c2e_null (picker "lead"): column of the per-cycle lead.
LEAD_SOURCES = ("omega", "alpha15", "alpha10", "alpha20", "opt", "cp", "sym")
GROUP_LABEL = {"closing": "closers (masseter, temporalis) before closing",
               "opening": "digastric before opening"}
# Reader wording of the cohort rows (stats.cohort_c2e row keys); never a plan code.
ROW_LABEL = {
    "h1": "Closer activity precedes the closing onset (both tasks)",
    "h2": "Closer release precedes the opening onset, open-close",
    "h3": "Closer release precedes the opening onset, chewing",
    "p4_alpha": "Closers before the angular-acceleration onset",
    "p6_changepoint": "Changepoint onsets before the closing onset",
    "optical": "Closers before the optical closing onset",
    "symmetric_window": "Symmetric window: median lead minus its surrogate median",
    "p5_open_close": "EMG-to-acceleration lag, open-close",
    "p5_chewing": "EMG-to-acceleration lag, chewing",
    "digastric": "Digastric locked to the opening onset",
    "s2_h1": "Closer locking beyond the cycle rhythm (cycle shuffle)",
    "s2_release_open_close": "Release slope against the cycle shuffle, open-close",
    "s2_release_chewing": "Release slope against the cycle shuffle, chewing",
    "decision": "Decision"}
BLOCK_LABEL = {"primary": "Primary hypothesis", "secondary": "Gated hypotheses (serial)",
               "supporting": "Supporting analyses", "structure": "Structure (descriptive)",
               "decision": ""}
# statistic of a cohort row -> (unit, decimals); a row without one of these reads as ms
STAT_UNIT = {"log(IQR / cycle-shuffle IQR)": ("", 2), "Theil-Sen b": ("", 2)}


# -- shared helpers ---------------------------------------------------------------------------
def c2e_params(ctx: Any) -> CohortParams:
    """The cohort part of ``stages.timing.c2e`` as the timing stage ran it."""
    return CohortParams.from_dict(timing_params(ctx, "c2e"))


def _cohort(ctx: Any) -> pd.DataFrame:
    return _t(ctx, "stats_timing", "c2e_cohort")


def _row(coh: pd.DataFrame, key: str) -> dict[str, Any]:
    r = _sel(coh, key=key)
    return {} if r.empty else r.iloc[0].to_dict()


def per_pid(null: pd.DataFrame, cp: CohortParams, scope: str, direction: str, source: str,
            col: str) -> pd.Series:
    """Participant values (index pid) of one c2e_null statistic, participants with at least
    ``min_usable`` usable channel-cycles only (the cohort tests' rule); NaN dropped."""
    g = _sel(null, scope=scope, direction=direction, source=source)
    if g.empty or col not in g:
        return pd.Series(dtype=float)
    g = g[g["n_usable"].fillna(0) >= cp.min_usable]
    return g.set_index("pid")[col].astype(float).dropna()


def opening_durations(cyc: pd.DataFrame, task: str) -> pd.Series:
    """Per participant the median time (ms) from the opening onset to the closing onset of
    the cycles (closing rows: window event minus its anchor, the opening onset)."""
    g = _sel(cyc, task=task, direction="closing")
    if g.empty:
        return pd.Series(dtype=float)
    g = g.drop_duplicates(["pid", "cycle"])
    d = (g["t_event_us"] - g["anchor_us"]) / 1e3
    return d.groupby(g["pid"]).median().dropna()


def _events(task: str, cp: CohortParams, null: pd.DataFrame) -> dict[str, pd.Series]:
    """Participant event times of one task relative to their own motion onset (ms)."""
    return {"release": -per_pid(null, cp, task, "opening", "release", "median_ms"),
            "opener": -per_pid(null, cp, task, "opening", "omega", "mu_ms"),
            "closer": -per_pid(null, cp, task, "closing", "omega", "mu_ms")}


def _est(r: dict[str, Any], unit: str = "ms", d: int = 0) -> str:
    """'46 ms [44, 54]' of a cohort row; '--' without an estimate."""
    m = _f(r.get("median"))
    if not np.isfinite(m):
        return "--"
    ci = _ci(_f(r.get("ci_lo")), _f(r.get("ci_hi")), d)
    return f"{_n(m, d)}{' ' + unit if unit else ''}" + (f" {ci}" if ci else "")


def h_text(coh: pd.DataFrame) -> str:
    """The primary and gated hypotheses in reader words (caption statistic)."""
    h1 = _row(coh, "h1")
    if not h1:
        return ""
    n = _f(h1.get("n"))
    s = (f"closer activity before the closing onset (both tasks pooled): locked lead "
         f"{_est(h1)}, n = {int(n) if np.isfinite(n) else 0} participants; locking p "
         f"{_peq(_f(h1.get('p_lock')))}, location p {_peq(_f(h1.get('p_loc')))} against the "
         f"bias margin of {_n(_f(h1.get('bias_margin_ms', h1.get('delta_ms'))))} ms, combined p "
         f"{_peq(_f(h1.get('p')))}")
    rel = []
    for key, task in (("h2", "open_close"), ("h3", "chewing")):
        r = _row(coh, key)
        if r:
            pa = _f(r.get("p_adj"))
            tail = (f"adjusted p {_peq(pa)}" if np.isfinite(pa)
                    else "not tested (primary hypothesis not rejected)")
            rel.append(f"{V.task_label(task)} {_est(r)}, {tail}")
    if rel:
        s += "; closer release before the opening onset: " + "; ".join(rel)
    return s


def decision_note(coh: pd.DataFrame) -> list[str]:
    d = _row(coh, "decision")
    if not d:
        return []
    h = str(d.get("hypothesis", ""))
    return [f"Decision of the hypothesis family: {DECISION_TEXT.get(h, h)}",
            "With back-to-back cycles, activity locked to the closing onset cannot be told "
            "apart from activity that keeps the cycle rhythm: the result is an order within "
            "the cycle, not a causal drive"]


def _empty_fig(fid: str, title: str, msg: str) -> FigureSpec:
    p = Panel("", Axis(""), Axis(""), [], note=msg)
    return attach(FigureSpec(fid, title, [p], width="column", height_in=1.0),
                  CaptionSpec(msg, evidence=EVIDENCE, codes=["X6"]))


# -- event sequence (paper, block C) ------------------------------------------------------------
TIMELINE_ROWS = ("closer release", "digastric onset", "closing onset (motion)",
                 "closer onset")
# Rows coloured by what they time: closer events blue, the digastric (opener) vermilion, the
# motion onset grey.
TIMELINE_ROLES = ("group:closers", "group:openers", "context", "group:closers")
EVENT_ROLES = {"release": "group:closers", "opener": "group:openers", "closer": "group:closers"}
ROW_COLOURS = [("blue", "closer (masseter, temporalis) events"),
               ("vermilion", "digastric onset"), ("grey", "motion onset")]


@register("c2e_sequence", "paper", "Order of muscle and motion events in a cycle", **G,
          choices={"form": SEQUENCE_FORMS}, form="timeline")
def c2e_sequence(req: ViewRequest) -> FigureSpec:
    """Per task the cycle's event sequence: closer release, opening onset, digastric onset,
    closing onset and closer onset; participants and the cohort median [95 % BCa CI]."""
    ctx = req.ctx
    form = req.options.get("form", "timeline")
    fid, title = "c2e_sequence", "Order of muscle and motion events in a cycle"
    null, cyc = _t(ctx, "timing", "c2e_null"), _t(ctx, "timing", "c2e_cycles")
    if null.empty or "source" not in null:
        return _empty_fig(fid, title, "no event-order results in this store")
    cp, seed = c2e_params(ctx), B.seed_of(ctx)
    ev = {t: _events(t, cp, null) for t in TIMING_TASKS}
    pids: set[str] = set()
    for e in ev.values():
        for s in e.values():
            pids |= set(s.index)
    marks = [("dots", "participants"), ("diamond", "cohort median"),
             ("line", "95 % BCa CI of the median")]
    if form == "timeline":
        panels = []
        for k, task in enumerate(TIMING_TASKS):
            e = ev[task]
            dur = opening_durations(cyc, task)
            closer = (dur + e["closer"]).dropna()          # closer onset on the opening axis
            vals = [e["release"], e["opener"], dur, closer]
            p = B.strip_ci(list(TIMELINE_ROWS), [v.to_numpy(float) for v in vals],
                           f"({'ab'[k]}) {V.task_label(task)}", TIME_AX, seed=seed, ref=0.0,
                           ref_label="opening onset", roles=TIMELINE_ROLES)
            p.sharex = "t"
            panels.append(p)
        spec = FigureSpec(fid, title, panels, cols=2, width="text",
                          height_in=B.rows_height(len(TIMELINE_ROWS), per=0.3),
                          sources=SOURCES)
        shows = ("the order of muscle and motion events within a movement cycle, per task, "
                 "on one time axis from the mechanical opening onset (0): the closer release "
                 "and the digastric onset against the opening onset, the closing onset at "
                 "each participant's median opening duration, and the closer onset before it "
                 "(closing onset minus the participant's locked closer lead)")
    else:
        labels, vals, roles = [], [], []
        for task in TIMING_TASKS:
            e = ev[task]
            for key, lab in (("release", "closer release, to opening"),
                             ("opener", "digastric onset, to opening"),
                             ("closer", "closer onset, to closing")):
                labels.append(f"{V.task_label(task)}: {lab}")
                vals.append(e[key].to_numpy(float))
                roles.append(EVENT_ROLES[key])
        allv = np.concatenate([np.zeros(1)] + [v[np.isfinite(v)] for v in vals])
        pad = 0.08 * (float(np.ptp(allv)) or 100.0)
        ax = Axis(EVENT_AX.label, EVENT_AX.unit,         # room right of 0 for the line label
                  lim=(float(allv.min()) - pad, float(allv.max()) + 3.5 * pad))
        if form == "raincloud":
            p, marks = forms.summary("raincloud", labels, vals, "", ax, seed=seed,
                                     ref=0.0, ref_label="motion onset")
        else:
            p = B.strip_ci(labels, vals, "", ax, seed=seed, ref=0.0,
                           ref_label="motion onset", roles=roles)
        spec = FigureSpec(fid, title, [p], width="text",
                          height_in=B.rows_height(len(labels), per=0.24), sources=SOURCES)
        shows = ("the muscle events of a movement cycle against their own mechanical onset, "
                 "per task: the closer release and the digastric onset against the opening "
                 "onset, the closer onset against the closing onset (negative: before)")
    coh = _cohort(ctx)
    return attach(spec, CaptionSpec(
        shows, marks + [("vertical line", "the motion onset")]
        + {"timeline": ROW_COLOURS, "raincloud": []}.get(form, ROW_COLOURS[:2]),
        f"n = {len(pids)} participants with jaw IMU; a participant enters a row with at least "
        f"{cp.min_usable} usable channel-cycles",
        "per participant the locked lead of the onsets and the closer release of the "
        "cycle-averaged envelope; the tests and their p values are in the event-order table",
        "locking and location tests (serial gatekeeping, Holm)",
        EVIDENCE, decision_note(coh)[:1], codes=["X6"]))


# -- locking and coupling -----------------------------------------------------------------------
LOCKING_FORMS = ("dumbbell", "slopegraph")


@register("c2e_locking", "paper", "Time-locking and coupling of the closers", **G,
          choices={"form": LOCKING_FORMS}, form="dumbbell")
def c2e_locking(req: ViewRequest) -> FigureSpec:
    """(a) Per participant the closer lead IQR before closing (both tasks) against the median
    IQR of the circular-shift surrogates; (b) the EMG-to-acceleration lag per task."""
    ctx = req.ctx
    form = req.options.get("form", "dumbbell")
    fid, title = "c2e_locking", "Time-locking and coupling of the closers"
    null, coup = _t(ctx, "timing", "c2e_null"), _t(ctx, "timing", "coupling")
    if null.empty or "source" not in null:
        return _empty_fig(fid, title, "no event-order results in this store")
    cp, seed = c2e_params(ctx), B.seed_of(ctx)
    n_sur = int(timing_params(ctx, "c2e").get("n_surrogates", C2EParams.n_surrogates))
    obs = per_pid(null, cp, "pooled", "closing", "omega", "iqr_obs")
    sur = per_pid(null, cp, "pooled", "closing", "omega", "iqr_null_median")
    pids = [p for p in ctx.pids() if p in obs.index and p in sur.index]
    a, b = obs.reindex(pids).to_numpy(float), sur.reindex(pids).to_numpy(float)
    a_title = "(a) Locking before closing"
    if form == "slopegraph":
        pa = B.slopegraph(["surrogates", "observed"], b, a, a_title, IQR_AX,
                          summary=[(float(np.median(v)) if v.size else np.nan,
                                    *B.median_ci(v, seed)) for v in (b, a)])
        pa_marks = [("grey lines in (a)", "join a participant's observed and surrogate IQR"),
                    ("diamonds in (a)", "median [95 % BCa CI]")]
    else:
        pa = B.dumbbell(B.pseudo_rows(ctx, pids), a, b, a_title, IQR_AX,
                        label_a="observed", label_b="surrogates (median)")
        pa_marks = [("rows in (a)", "participants, the observed IQR and the median IQR of the "
                                    "circular-shift surrogates joined")]
    # (b) coupling lag per task, residualised next to raw
    tasks = [t for t in TIMING_TASKS if not _sel(coup, task=t).empty]
    res = [_sel(coup, task=t)["emd_lag_ms"].to_numpy(float) for t in tasks]
    raw = [_sel(coup, task=t)["emd_lag_raw_ms"].to_numpy(float) for t in tasks]
    pb = B.paired_strip([V.task_label(t) for t in tasks], res, raw,
                        "(b) EMG-to-acceleration lag", LAG_AX, label_a="residualised",
                        label_b="raw", seed=seed, ref=0.0)
    coh = _cohort(ctx)
    h1 = _row(coh, "h1")
    p5 = [(t, _row(coh, f"p5_{t}")) for t in TIMING_TASKS]
    stat = (f"(a) cohort locking p {_peq(_f(h1.get('p_lock')))} (one-sided signed-rank test "
            "of the log ratio of observed to surrogate IQR below 0)" if h1 else "")
    lag = "; ".join(f"{V.task_label(t)} {_est(r, d=0)}, Holm p {_peq(_f(r.get('p_adj')))}"
                    for t, r in p5 if r)
    if lag:
        stat += ("; " if stat else "") + "(b) residualised lag " + lag
    spec = FigureSpec(fid, title, [pa, pb], cols=2, width="text",
                      height_in=B.rows_height(max(len(pids), 4)), sources=[
                          "timing:c2e_null", "timing:coupling", "stats_timing:c2e_cohort"])
    return attach(spec, CaptionSpec(
        "(a) how tightly the closer onsets lock to the closing onset: per participant the "
        "interquartile range of the per-cycle leads (both tasks) and the median of the same "
        f"range when the EMG is circularly shifted against the motion ({n_sur} surrogates; "
        "smaller observed range: time-locked). (b) Continuous coupling: lag of the peak "
        "correlation between the rectified closer EMG and the closing angular acceleration "
        "(> 0: EMG first), on cycle-residualised signals and raw; the raw lag is not an "
        "onset order",
        pa_marks + [("dots in (b)", "participants"), ("diamonds in (b)", "median with its "
                                                                         "95 % BCa CI")],
        f"n = {len(pids)} participants in (a) (at least {cp.min_usable} usable "
        f"channel-cycles), {coup['pid'].nunique() if 'pid' in coup else 0} in (b)",
        stat, "", EVIDENCE + "; supporting analyses, no claim rests on them",
        codes=["X6"]))


# -- cohort table -------------------------------------------------------------------------------
def _p_cell(v: Any) -> str:
    v = _f(v)
    return "--" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"{v:.3f}")


# Reader words of the stored family decision (stats.cohort_c2e.decide): "not shown" read as an
# omitted cell.
DECISION_TEXT = {"not shown": "event order not demonstrated",
                 "shown (post hoc)": "event order demonstrated (post hoc)",
                 "inconclusive (underpowered)": "inconclusive (underpowered)",
                 "delta not fixed": "no bias margin fixed (no decision)"}


def _result(r: dict[str, Any]) -> str:
    fam, key = r.get("family"), r.get("key")
    if fam == "structure":
        return "descriptive; cohort p withheld"
    if fam == "decision":
        h = str(r.get("hypothesis", ""))
        return DECISION_TEXT.get(h, h)
    if fam == "secondary" and not np.isfinite(_f(r.get("p_adj"))):
        return "not tested (gatekeeping)"
    out = "rejected" if bool(r.get("rejected")) else "not rejected"
    if bool(r.get("veto")):
        out += "; veto (CI below 0)"
    if key == "digastric":
        out += " (locking)"
    return out


HOLM_FAMILY_ROWS = ("h1", "h2", "h3")          # rows of the post hoc Holm sensitivity


@register("c2e_table", "paper", "Order of muscle and motion events: cohort tests", **G)
def c2e_table(req: ViewRequest) -> TableSpec:
    """The cohort rows of the event-order family under a heading row per block: estimate
    [95 % BCa CI], the p values of the locking and location parts (one cell), the adjusted p
    with the post hoc Holm over H1–H3 in parentheses, and the result; then the decision."""
    ctx = req.ctx
    coh = _cohort(ctx)
    rows: list[dict[str, Any]] = []
    block = None
    for r in (coh.to_dict("records") if not coh.empty and "key" in coh else []):
        unit, d = STAT_UNIT.get(str(r.get("statistic")), ("ms", 0))
        dec = r.get("family") == "decision"
        n, nf = _f(r.get("n")), _f(r.get("n_flagged"))
        lab = BLOCK_LABEL.get(str(r.get("family")), "")
        if lab != block:                      # a new block: rule, then its heading row
            if rows:
                rows.append({"_rule": True})
            if lab:
                rows.append({"question": lab, "n": "", "estimate": "", "p_parts": "",
                             "p_adj": "", "result": "", "_head": True})
            block = lab
        lock, loc = _p_cell(r.get("p_lock")), _p_cell(r.get("p_loc"))
        holm = (_p_cell(r.get("p_holm_family")) if r.get("key") in HOLM_FAMILY_ROWS else "")
        rows.append({
            "question": ROW_LABEL.get(str(r.get("key")), str(r.get("hypothesis", ""))),
            "n": "" if dec or not np.isfinite(n) else
                 f"{int(n)}" + (f" ({int(nf)} flagged)" if np.isfinite(nf) and nf > 0 else ""),
            "estimate": (f"margin {_n(_f(r.get('bias_margin_ms', r.get('delta_ms'))))} ms" if dec
                         else _est(r, unit, d)),
            "p_parts": "" if dec else ("--" if lock == loc == "--" else f"{lock} / {loc}"),
            "p_adj": "" if dec else _p_cell(r.get("p_adj")) + (f" ({holm})" if holm else ""),
            "result": _result(r)})
    if not rows:
        rows = [{"question": "no event-order cohort results in this store"}]
    cols = [Column("question", "Question", align="X"),
            Column("n", "n", align="l"),
            Column("estimate", "Estimate [95 % CI]", align="l"),
            Column("p_parts", "Lock. / loc.", align="l", group="p"),
            Column("p_adj", "Adj. (Holm over the three)", align="l", group="p"),
            Column("result", "Result", align="X")]
    t = TableSpec("c2e_table", "", cols, rows, width="text",
                  notes="Lock.: locking part, one-sided signed-rank test of the log ratio of "
                        "the observed to the surrogate lead spread. Loc.: location part, "
                        "one-sided signed-rank test of the lead minus the bias margin. Adj.: "
                        "adjusted p; the primary unadjusted, the release rows by serial "
                        "gatekeeping with Holm, the supporting rows by Holm within their "
                        "block. In parentheses: post hoc sensitivity, Holm over the three "
                        "event-order hypotheses whatever the gatekeeping decided, chosen "
                        "after the results were seen (not the pre-specified procedure).",
                  sources=["stats_timing:c2e_cohort"])
    h1 = _row(coh, "h1")
    return attach(t, CaptionSpec(
        "does muscle activity precede jaw motion: the cohort tests of the event order "
        "(closers before the closing onset, closer release before the opening onset), the "
        "supporting analyses and the structure rows, then the decision of the hypothesis family",
        [("estimate", "median over participants of the locked lead, the release lead, the "
                      "lag, or the ratio, with its 95 % BCa CI (ms unless a ratio or slope)")],
        f"n = {int(_f(h1.get('n')))} participants in the primary row" if h1 and
        np.isfinite(_f(h1.get("n"))) else "",
        "a hypothesis holds only if both parts reject (intersection-union, p = the larger "
        "of the two)",
        "", EVIDENCE,
        # the Holm sensitivity in parentheses is explained in the table note (the
        # caption said it twice)
        ["Structure rows (cycle shuffle) report the effect size only: their cohort p is "
         "withheld, as the shuffle is not calibrated at the cohort level and has no power "
         "for back-to-back cycles",
         f"Flagged: participants with {cp_flag_text(ctx)} usable channel-cycles, counted "
         "but not tested"] + decision_note(coh)[1:], codes=["X6"]))


def cp_flag_text(ctx: Any) -> str:
    """'3–4': the usable channel-cycle counts that are flagged, not tested."""
    cp = c2e_params(ctx)
    return f"{cp.flag_usable}–{cp.min_usable - 1}"


# -- dashboards: per-cycle tables ---------------------------------------------------------------
def _lead_col(src: str) -> str:
    return "lead_omega_ms" if src == "omega" else f"lead_{src}_ms"


@register("c2e_cycles", "dashboard", "Muscle leads per cycle (event order)", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS, **G,
          choices={"lead": LEAD_SOURCES})
def c2e_cycles(req: ViewRequest) -> FigureSpec:
    """Every channel-cycle lead per participant (closers before closing, digastrics before
    opening) for one task and lead source, with the participant's locked lead."""
    ctx, task = req.ctx, req.phase or "open_close"
    src = req.options.get("lead") or "omega"
    fid, title = "c2e_cycles", "Muscle leads per cycle (event order)"
    cyc, null = _t(ctx, "timing", "c2e_cycles"), _t(ctx, "timing", "c2e_null")
    col = _lead_col(src)
    if cyc.empty or col not in cyc:
        return _empty_fig(fid, title, "no event-order channel-cycles in this store")
    ucol = f"usable_{src}" if src in ("cp", "sym") else "usable"
    cp = c2e_params(ctx)
    g = _sel(cyc, task=task)
    pids = [p for p in ctx.pids() if (g["pid"] == p).any()]
    rows = B.pseudo_rows(ctx, pids)
    panels, n_use = [], 0
    for direction in ("closing", "opening"):
        d = _sel(g, direction=direction)
        use = d[d[ucol].astype(bool)] if not d.empty else d
        bad = d[~d[ucol].astype(bool)] if not d.empty else d
        mu = per_pid(null, cp, task, direction, src, "mu_ms")
        n_use += len(use)
        panels.append(B.strip_mark(
            rows, [use.loc[use["pid"] == q, col] for q in pids],
            GROUP_LABEL[direction].capitalize(), LEAD_AX,
            marks=[(_f(mu.get(q)), np.nan, np.nan) for q in pids],
            mark_label="locked lead (mixture)", unit_label="usable channel-cycles",
            flagged=[bad.loc[bad["pid"] == q, col] for q in pids],
            flagged_label="not usable", ref=0.0))
    spec = FigureSpec(fid, title, panels, cols=2, width="text",
                      height_in=B.rows_height(len(pids)),
                      sources=["timing:c2e_cycles", "timing:c2e_null"])
    return attach(spec, CaptionSpec(
        f"{V.task_label(task)}: every channel-cycle's {V.choice_label('lead', src)} "
        "per participant (> 0: EMG before the motion onset), the closers before closing and "
        "the digastrics before opening, with the participant's locked lead",
        [("dots", "usable channel-cycles"), ("grey triangles", "not usable (no onset, "
                                                                "active at the window start "
                                                                "or below the burst size)"),
         ("bar", "the participant's locked lead (mixture location)")],
        f"n = {len(pids)} participants, {n_use} usable channel-cycles",
        evidence=EVIDENCE, codes=["X6"]))


@register("c2e_release", "dashboard", "Closer release per cycle (event order)", params=("phase",),
          phase="open_close", domain="timing", phases=TIMING_TASKS, **G)
def c2e_release(req: ViewRequest) -> FigureSpec:
    """Every cycle's closer release before the opening onset per participant, and the release
    depth of the participant's averaged closer envelope against the surrogates."""
    ctx, task = req.ctx, req.phase or "open_close"
    fid, title = "c2e_release", "Closer release per cycle (event order)"
    rel, null = _t(ctx, "timing", "c2e_release"), _t(ctx, "timing", "c2e_null")
    g = _sel(rel, task=task)
    if g.empty:
        return _empty_fig(fid, title, "no closer-release results in this store")
    cp = c2e_params(ctx)
    pids = [p for p in ctx.pids() if (g["pid"] == p).any()]
    rows = B.pseudo_rows(ctx, pids)
    med = per_pid(null, cp, task, "opening", "release", "median_ms")
    pa = B.strip_mark(rows, [g.loc[g["pid"] == q, "lead_release_ms"] for q in pids],
                      "(a) Release before the opening onset",
                      Axis("Release lead before the opening onset", V.MSEC),
                      marks=[(_f(med.get(q)), np.nan, np.nan) for q in pids],
                      mark_label="release of the averaged envelope", unit_label="cycles",
                      ref=0.0)
    r = _sel(null, scope=task, direction="opening", source="release")
    r = r.set_index("pid").reindex(pids) if not r.empty else pd.DataFrame(index=pids)
    depth = r["depth_db"].to_numpy(float) if "depth_db" in r else np.full(len(pids), np.nan)
    dnull = (r["depth_null_median_db"].to_numpy(float) if "depth_null_median_db" in r
             else np.full(len(pids), np.nan))
    pb = B.dumbbell(rows, depth, dnull, "(b) Release depth", Axis("Release depth", V.DB),
                    label_a="observed", label_b="surrogates (median)")
    spec = FigureSpec(fid, title, [pa, pb], cols=2, width="text",
                      height_in=B.rows_height(len(pids)),
                      sources=["timing:c2e_release", "timing:c2e_null"])
    return attach(spec, CaptionSpec(
        f"{V.task_label(task)}: (a) the closer release of every cycle before the mechanical "
        "opening onset (> 0: release first), with the release of the participant's averaged "
        "envelope; (b) the depth of that release against the median depth when the EMG is "
        "circularly shifted against the motion",
        [("dots in (a)", "cycles"), ("bar in (a)", "release of the averaged envelope"),
         ("rows in (b)", "participants, observed and surrogate depth joined")],
        f"n = {len(pids)} participants, {len(g)} cycles", evidence=EVIDENCE, codes=["X6"]))
