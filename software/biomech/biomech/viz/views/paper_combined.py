# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Combined figures for the JNER manuscript (2026-11-05): fewer, denser main figures built from
the panels of existing items, so the paper can move the single-topic originals to the
Additional files without losing anything. The originals stay unchanged paper items.
Captions name no other item version and do not open with a title (the manuscript sets the
title); ``meta["combines"]`` records the source views.

* ``jnl_timing_overview``: the timing story in one figure -- opener / closer activity around
  the motion onset and the opening lead under both onset rules (``jnl_c2`` a, b), the order of
  muscle and motion events in the cycle (``c2e_sequence`` timeline) and the causal
  closer-release detection (``jnl_x3`` a, b).
* ``jnl_sensing``: what the worn sensors recover without the camera -- the IMU opening
  agreement (``icorr_imu`` Bland-Altman, generic and own model) next to the camera-free
  opening-angle ladder (``cf_opening_ladder``).
"""

from __future__ import annotations

import re
from typing import Any

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import FigureSpec, Panel
from biomech.viz.views import registry as R
from biomech.viz.views.registry import ViewRequest, register

_PREFIX = re.compile(r"^\(\w+\)\s*")


def _sub(req: ViewRequest, view: str, **kw: Any) -> tuple[FigureSpec, dict[str, Any]]:
    """A sub-figure and its caption parts."""
    spec = R.build(view, R.request(req.ctx, view, **kw))
    return spec, (spec.meta or {}).get("caption") or {}


def _letter(p: Panel, letter: str, title: str | None = None) -> Panel:
    p.title = f"({letter}) " + (title if title is not None else _PREFIX.sub("", p.title or ""))
    p.tag = ""
    return p


TIMING_FORMS = ("lead", "events", "all")


def _timing_marks_a(ctx, pa: Panel) -> list[tuple[str, str]]:
    """(a) the median closer release (X2, average envelope) and the median digastric rise on
    the time axis (JNER revision 3.6), as group-coloured triangles under the envelopes."""
    import numpy as np

    from biomech.viz.spec import Series
    from biomech.viz.views.timing_j18 import cue_leads

    rq = ctx.table("requirements", "requirements") if "requirements" in ctx.store.stages() \
        else None
    rel = float(rq.loc[rq.key == "closer_release_ms", "value"].iloc[0]) \
        if rq is not None and (rq.key == "closer_release_ms").any() else np.nan
    try:
        rise = cue_leads(ctx, "open_close")["dig_rise"][0]
    except Exception:  # noqa: BLE001 -- a store without the J18 audit
        rise = np.empty(0)
    rise_t = -float(np.median(rise)) if rise.size else np.nan
    ys = [float(v) for sr in pa.series if sr.kind in ("band", "line") and sr.y is not None
          for v in np.concatenate([np.asarray(sr.y, float),
                                   np.asarray(sr.y2 if sr.y2 is not None else [], float)])
          if np.isfinite(v)]
    lo = min(ys) if ys else -5.0
    y = lo - 1.2
    marks = []
    for t, role, ms, lab in ((-rel, "group:closers", "^", "closer release"),
                             (rise_t, "group:openers", "^", "digastric rise")):
        if np.isfinite(t):
            pa.series.append(Series("scatter", np.array([t]), np.array([y]), role=role,
                                    style={"ms": ms, "size": 1.7}))
            pa.series.append(Series("text", np.array([t]), np.array([y]), text=[lab],
                                    style={"anchor": "below"}))
    if np.isfinite(rel) or np.isfinite(rise_t):
        pa.y.lim = (y - 2.6, pa.y.lim[1] if pa.y.lim else max(ys) + 1.0)
        marks.append(("triangles under (a)", "median closer release of the average envelope and "
                                             "median rise of the cycle-averaged digastric "
                                             "envelope"))
    return marks


def _lead_nulls(ctx, pb: Panel) -> list[tuple[str, str]]:
    """(b) the circular-shift null of each onset rule (D23) as a short dashed line over its
    column, labelled (JNER revision 3.6: the C2 argument is that the lead is no larger)."""
    import numpy as np

    from biomech.viz.spec import Series

    rq = ctx.table("requirements", "requirements") if "requirements" in ctx.store.stages() \
        else None
    if rq is None or "lead_null_ms" not in rq:
        return []
    out = False
    vals = []
    for i, key in enumerate(("lead_first_ms", "lead_drive_ms")):
        r = rq[rq.key == key]
        v = float(r.lead_null_ms.iloc[0]) if len(r) else np.nan
        if np.isfinite(v):
            pb.series.append(Series("line", np.array([i - 0.22, i + 0.22]), np.array([v, v]),
                                    role="threshold", style={"width": 1.4, "dash": "dashed"}))
            # the label centred above its line: beside it, it crossed the participant
            # lines
            lab = Series("text", np.array([float(i)]), np.array([v]),
                         text=[f"shift null {v:.0f} ms"], style={"anchor": "above"})
            pb.series.append(lab)
            vals.append(v)
            out = True
    if out:
        ys = [float(np.nanmax(sr.y)) for sr in pb.series if sr.y is not None
              and sr.kind in ("line", "scatter") and np.isfinite(sr.y).any()]
        lo = min(float(np.nanmin(sr.y)) for sr in pb.series if sr.y is not None
                 and sr.kind in ("line", "scatter") and np.isfinite(sr.y).any())
        top = max(ys)
        span = max(top - lo, 1.0)
        # the n and Holm p lines (stacked by the slopegraph above the data) move up past the
        # null labels
        ann = [sr for sr in pb.series if sr.kind == "text" and sr.text
               and (sr.text[0].startswith("n =") or sr.text[0].startswith("Holm"))]
        for sr in pb.series:          # clear of the dashed line and a dot on it
            if sr.kind == "text" and sr.text and sr.text[0].startswith("shift null"):
                sr.y = sr.y + 0.05 * span
        base = max(top, max(vals) + 0.19 * span)
        for sr in ann:
            k = 0 if sr.text[0].startswith("Holm") else 1
            sr.y = np.array([base + span * (0.08 + 0.13 * k)])
        new_top = base + span * 0.40
        lim = pb.y.lim
        pb.y.lim = (lim[0] if lim is not None else lo - 0.08 * span,
                    max(lim[1] if lim is not None else top, new_top))
    return [("dashed line over each rule in (b)", "its circular-shift null, the lead a "
                                                  "window alone produces")] if out else []


@register("jnl_timing_overview", "paper", "EMG and motion timing in one figure",
          choices={"form": TIMING_FORMS}, form="all")
def jnl_timing_overview(req: ViewRequest) -> FigureSpec:
    """Form ``lead`` (JNER revision 3.6, 2026-11-10: the figure split in two, each at full
    width): (a) opener and closer activity around the motion onset with the median closer
    release and digastric rise marked, (b) the opening lead per participant under both onset
    rules with the shift null of each. Form ``events``: (a, b) the event order in the cycle
    (open-close, chewing), (c) openings detected by the causal closer-release detector
    against chance, participants by detection, (d) its anticipation against chance. Form
    ``all``: the earlier six panels."""
    form = req.options.get("form", "all")
    c2, c2c = _sub(req, "jnl_c2")
    seq, seqc = _sub(req, "c2e_sequence", form="timeline")
    x3, x3c = _sub(req, "jnl_x3")
    by = {_PREFIX.sub("", q.title or ""): q for q in c2.panels}
    pa, pb = by["Opener and closer activity"], by["Opening lead per participant"]
    pc, pd = seq.panels[0], seq.panels[1]
    pe, pf = x3.panels[0], x3.panels[1]
    if form == "lead":
        return _timing_lead(req, pa, pb, c2, c2c)
    if form == "events":
        return _timing_events(req, pc, pd, pe, seq, seqc, x3, x3c)
    for p, letter in zip((pa, pb, pc, pd, pe, pf), "abcdef"):
        _letter(p, letter)
    _letter(pc, "c", "Event order, open-close")
    _letter(pd, "d", "Event order, chewing")
    _letter(pe, "e", "Release detected vs chance")
    _letter(pf, "f", "Release anticipation")
    panels = [pa, pb, pc, pd, pe, pf]
    sources = list(dict.fromkeys((c2.sources or []) + (seq.sources or []) + (x3.sources or [])))
    spec = FigureSpec("jnl_timing_overview", "EMG and motion timing", panels, rows=3, cols=2,
                      width="text", height_in=1.75, sources=sources,
                      meta={"combines": ["jnl_c2", "c2e_sequence", "jnl_x3"]})
    marks = [("bands in (a)", "IQR over participants"),
             ("grey lines in (b)", "participants; diamonds: median [95 % CI]"),
             ("dots in (c, d)", "participants; diamond and line: median with its 95 % BCa CI; "
                                "blue closer events, vermilion digastric onset, grey motion "
                                "onset"),
             ("(e)", "per participant, black: real openings, grey: random pseudo-onsets"),
             ("lines in (f)", "join a participant's offline and stream value"),
             ("dashed lines", "the motion onset (a, c, d) and no lead (b, f)")]
    return attach(spec, CaptionSpec(
        "(a) Opener and closer activity around the "
        "mechanical opening onset. (b) Opening lead of the digastric EMG per participant under "
        "the pre-specified first-onset rule and the post hoc driving-burst rule. (c, d) Order "
        "of muscle and motion events within the cycle, on one time axis from the opening "
        "onset. (e) Openings detected by the causal closer-release detector against its chance "
        "level. (f) Its anticipation of the opening onset, offline and in simulated real time",
        marks,
        "; ".join(x for x in (f"(a, b) {c2c.get('n', '')}".strip(),
                              f"(c, d) {seqc.get('n', '')}".strip(),
                              f"(e, f) {x3c.get('n', '')}".strip()) if len(x) > 7),
        "(b) median lead per rule, Holm over the confirmatory family; (c, d) per participant "
        "the locked lead of the onsets and the closer release of the cycle-averaged envelope; "
        "(e, f) per participant the share detected and the median decision time",
        evidence="(b) pre-specified, C2 not confirmed; (c, d) post hoc, event order not "
                 "demonstrated (tests in the event-order table); (a, e, f) exploratory",
        codes=["C2", "X6", "X3"]))


def _timing_lead(req: ViewRequest, pa: Panel, pb: Panel, c2: FigureSpec,
                 c2c: dict[str, Any]) -> FigureSpec:
    _letter(pa, "a")
    _letter(pb, "b")
    pa.legend = "inside:north west"            # not below the panel (3.0)
    pb.legend = "none"                         # the caption names the marks
    # the cohort medians in the openers' colour (the lead is the digastric's; blue is the
    # closers in (a)), not black
    for sr in pb.series:
        if sr.role == "accent":
            sr.role = "group:openers"
    marks = _timing_marks_a(req.ctx, pa) + _lead_nulls(req.ctx, pb)
    spec = FigureSpec("jnl_timing_overview", "Opening lead of the digastric EMG", [pa, pb],
                      rows=1, cols=2, width="text", height_in=2.2,
                      sources=list(c2.sources or []) + ["requirements:requirements"],
                      meta={"combines": ["jnl_c2"]})
    return attach(spec, CaptionSpec(
        "(a) Opener (digastric) and closer (masseter, temporalis) activity around the "
        "mechanical opening onset in open-close, cohort median of the participant envelopes. "
        "(b) Opening lead of the digastric EMG per participant under the pre-specified "
        "first-onset rule and the post hoc driving-burst rule",
        [("vermilion and blue bands in (a)", "IQR over participants"),
         ("dashed line in (a)", "the motion onset")] + marks
        + [("grey lines in (b)", "join a participant's two values, light squares a "
                                 "participant usable under one rule only"),
           ("vermilion diamonds in (b)", "cohort median with its 95 % BCa CI")],
        c2c.get("n", ""),
        "(b) median lead per rule, Holm over the confirmatory family",
        evidence="(b) confirmatory, not confirmed (the driving-burst rule is post hoc). "
                 "Descriptive: (a) and its marks",
        codes=["C2"]))


def _timing_events(req: ViewRequest, pc: Panel, pd: Panel, pe: Panel, seq: FigureSpec,
                   seqc: dict[str, Any], x3: FigureSpec, x3c: dict[str, Any]) -> FigureSpec:
    import numpy as np

    from biomech.viz.spec import Axis, Series
    from biomech.viz.views import blocks as B
    from biomech.viz.views import vocab as V

    ctx = req.ctx
    _letter(pc, "a", "Event order, open-close")
    _letter(pd, "b", "Event order, chewing")
    _letter(pe, "c", "Release detected against chance")
    # (c) participants ordered by the share of real openings detected (3.0)
    pts = [sr for sr in pe.series if sr.kind == "scatter" and sr.role == "accent"]
    if pts and pe.y.ticklabels:
        det = np.asarray(pts[0].x, float)
        order = np.argsort(-np.where(np.isfinite(det), det, -1.0), kind="stable")
        newy = np.empty(len(order))
        newy[order] = np.arange(len(order), dtype=float)
        for sr in pe.series:
            if sr.y is not None and sr.kind in ("scatter", "line"):
                sr.y = newy[np.asarray(sr.y, float).round().astype(int)]
        pe.y.ticklabels = [pe.y.ticklabels[i] for i in order]
    pe.legend = "inside:north west"
    for sr in pe.series:                       # short keys fit the empty corner
        if sr.label == "random pseudo-onsets":
            sr.label = "chance"
        elif sr.label == "real openings":
            sr.label = "openings"
    # (d) anticipation of the detector against chance (offline causal chain), 3.6
    rs = ctx.table("timing", "release_summary")
    prim = rs[(rs.task == "open_close") & rs.primary.astype(bool)] if not rs.empty else rs
    pids = [p for p in ctx.pids() if not prim.empty and (prim.pid == p).any()]
    s = prim.set_index("pid").reindex(pids)
    real = s.anticipation_ms.to_numpy(float) if pids else np.empty(0)
    cha = s.chance_anticipation_ms.to_numpy(float) if pids else np.empty(0)
    rc = ctx.table("stats_timing", "release_cohort")
    r = rc[(rc.task == "open_close") & rc.primary.astype(bool)] if not rc.empty else rc

    def med_ci(v: np.ndarray) -> tuple[float, float, float]:
        from biomech.stats.core import describe

        d = describe(v[np.isfinite(v)], seed=B.seed_of(ctx), stat="median")
        return float(d["median"]), float(d["ci_lo"]), float(d["ci_hi"])

    pf = B.slopegraph(["chance", "openings"], cha, real,
                      "(d) Anticipation against chance",
                      Axis("Decision before onset", V.MSEC),
                      summary=[med_ci(cha), med_ci(real)], summary_label="median [95 % CI]",
                      lone_label=None, ref=0.0,
                      annot=[[], [f"p = {float(r.p_vs_chance.iloc[0]):.3f}"]] if len(r)
                      and "p_vs_chance" in r else None)
    pf.legend = "none"
    pf.tag = ""
    spec = FigureSpec("jnl_timing_overview", "Event order and closer release", [pc, pd, pe, pf],
                      rows=2, cols=2, width="text", height_in=1.85,
                      sources=list(dict.fromkeys((seq.sources or []) + (x3.sources or [])
                                                 + ["timing:release_summary"])),
                      meta={"combines": ["c2e_sequence", "jnl_x3"]})
    return attach(spec, CaptionSpec(
        "(a, b) Order of muscle and motion events within the cycle, on one time axis from the "
        "opening onset, in open-close and chewing. (c) Per participant the share of openings "
        "the causal closer-release detector finds before the motion onset, against the share "
        "at random pseudo-onsets (chance), participants ordered by the share found. (d) Time "
        "of the detector's decision before the opening onset at real openings, against the "
        "time at random pseudo-onsets (chance), offline replay of the causal chain",
        [("dots in (a, b)", "participants, blue closer events, vermilion digastric onset, grey "
                            "motion onset"),
         ("diamond and line in (a, b)", "median with its 95 % BCa CI"),
         ("black dots and grey squares in (c)", "real openings and random pseudo-onsets"),
         ("grey lines in (d)", "join a participant's two values"),
         ("diamonds in (d)", "median with its 95 % BCa CI"),
         ("dashed lines", "the opening onset (a, b) and no anticipation (d)")],
        "; ".join(x for x in (f"(a, b) {seqc.get('n', '')}".strip(),
                              f"(c, d) {x3c.get('n', '')}".strip()) if len(x) > 7),
        "(a, b) per participant the locked lead of the onsets and the closer release of the "
        "cycle-averaged envelope. (c, d) per participant the share detected and the median "
        "decision time, Wilcoxon signed-rank test of the anticipation against its chance level",
        evidence="(a, b) exploratory (planned), event order not demonstrated (tests in the "
                 "event-order table). (c, d) exploratory (post hoc)",
        codes=["X6", "X3"]))


def _translation_panel(ctx) -> Panel:
    """(d) Camera-free incisor position (JNER revision 3.4, 2026-11-10): incisor RMSE against
    the optics of the three translation models and the accelerometer fusion, per participant
    (grey) with the cohort median and its 95 % BCa CI (summary colour), the 2 mm bound."""
    import numpy as np

    from biomech.viz.spec import Axis, Series
    from biomech.viz.views import vocab as V
    from biomech.viz.views.camerafree import INC_BOUND, _cohort_row, _table, rung

    part = _table(ctx, "camerafree_cohort", "cf_participant")
    c = _table(ctx, "camerafree_cohort", "cf_cohort")
    p = Panel("(d) Camera-free incisor position", Axis("Incisor RMSE", V.MM), Axis(""), [],
              legend="none")
    rows = [r for r in TRANSLATION_RUNGS
            if not part.empty and ((part.outcome == "rmse_inc_mm") & (part.estimator == r)).any()]
    for i, r in enumerate(rows):
        v = part[(part.outcome == "rmse_inc_mm") & (part.estimator == r)]["value"] \
            .to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            jit = np.linspace(-0.17, 0.17, v.size) if v.size > 1 else np.zeros(1)
            p.series.append(Series("scatter", v, np.full(v.size, float(i)) + jit,
                                   role="context"))
        row = _cohort_row(c, r, "rmse_inc_mm") or {}
        lo, hi = row.get("ci_lo", np.nan), row.get("ci_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                   role="summary", style={"width": 1.6}))
        if np.isfinite(row.get("median", np.nan)):
            p.series.append(Series("scatter", np.array([row["median"]]), np.array([float(i)]),
                                   role="summary", style={"ms": "d", "size": 1.3}))
    p.series.append(Series("vline", x=np.array([INC_BOUND]), role="threshold"))
    p.series.append(Series("text", np.array([INC_BOUND]), np.array([-0.55]),
                           text=[f"{INC_BOUND:g} mm bound"], style={"anchor": "right"}))
    p.y.ticks = [float(i) for i in range(len(rows))]
    p.y.ticklabels = [TRANSLATION_TICK.get(r, rung(r)) for r in rows]
    p.y.lim = (-0.75, len(rows) - 0.4) if rows else None
    p.y.reverse = True
    vals = [x for sr in p.series if sr.kind in ("scatter", "line")
            for x in np.asarray(sr.x, float) if np.isfinite(x)]
    p.x.lim = (0.0, max(vals + [2.5 * INC_BOUND]) * 1.05)
    if not rows:
        p.note = "no camera-free results"
    return p


# the three translation models and the fusion (not the accelerometers alone, which drift
# by orders of magnitude between anchors)
TRANSLATION_RUNGS = ("E4hinge", "E4gen", "E4own", "E6")
TRANSLATION_TICK = {"E4hinge": "condyle hinge", "E4gen": "generic model",
                    "E4own": "own model", "E6": "fusion with own model"}


def _larger_loa(ctx) -> tuple[float, float]:
    """The larger absolute limit of agreement of the IMU opening (generic, own model)."""
    import numpy as np

    ba = ctx.table("stats_imu", "imu_opening_ba") if "stats_imu" in ctx.store.stages() \
        else None
    out = []
    for m in ("loso", "subject"):
        r = ba[(ba.task == "open_close") & (ba.estimator == "quat") & (ba.model == m)] \
            if ba is not None and not ba.empty else None
        out.append(float(max(abs(r.loa_lo.iloc[0]), abs(r.loa_hi.iloc[0])))
                   if r is not None and len(r) else np.nan)
    return out[0], out[1]


@register("jnl_sensing", "paper", "Camera-free sensing in one figure")
def jnl_sensing(req: ViewRequest) -> FigureSpec:
    """(a, b) agreement of the IMU-estimated incisal opening with the optics (generic model
    for an unseen user, the participant's own model), (c) the camera-free opening-angle
    RMSE along the calibration ladder, (d) the camera-free incisor position of the
    translation models and the fusion against the 2 mm bound (JNER revision 3.4: the
    headline is "rotation yes, translation not yet")."""
    imu, imuc = _sub(req, "icorr_imu")
    # without the offline resets variant: the paper does not use it
    cf, cfc = _sub(req, "cf_opening_ladder", form="rows", offline="no")
    pa, pb = imu.panels[0], imu.panels[1]
    pc = cf.panels[0]
    _letter(pa, "a", "IMU opening, generic model")
    _letter(pb, "b", "IMU opening, own model")
    pa.x.label, pa.x.unit = pb.x.label, pb.x.unit        # the stacked original drops it on (a)
    _letter(pc, "c", "Camera-free opening angle")
    # (a, b): bias and limits of agreement in the summary colour; the proportional-bias line
    # neutral dark (orange is the EMG colour, 3.0)
    for p in (pa, pb):
        p.height = 1.0
        for sr in p.series:
            if sr.kind == "hline":
                sr.style = {**sr.style, "color": "blue"}
            elif sr.kind == "band" and sr.role == "band":
                sr.role = "summary:fill"
                sr.style = {**sr.style, "alpha": 0.5}
    # (c): no lines joining the participants (they dominated the panel, 3.4), participants
    # grey, the cohort median in the summary colour
    pc.series = [sr for sr in pc.series if not (sr.kind == "line" and sr.role == "faint")]
    dots = [sr for sr in pc.series if sr.role == "muted" and sr.kind == "scatter"]
    for k, sr in enumerate(dots):              # a small vertical jitter per participant
        sr.y = sr.y - 0.05 - 0.22 * (k / max(len(dots) - 1, 1))
    for sr in pc.series:
        if sr.role == "muted" and sr.kind == "scatter":
            sr.role, sr.style = "context", {}
        elif sr.role == "accent":
            sr.role, sr.label = "summary", None
    pc.legend = "none"
    pd_ = _translation_panel(req.ctx)
    sources = list(dict.fromkeys((imu.sources or []) + (cf.sources or [])))
    spec = FigureSpec("jnl_sensing", "Camera-free sensing", [pa, pb, pc, pd_], rows=2, cols=2,
                      width="text", height_in=1.75, sources=sources,
                      meta={"combines": ["icorr_imu", "cf_opening_ladder"]})
    gen, own = _larger_loa(req.ctx)
    loa_txt = (f" The larger absolute limit of agreement is {gen:.1f} mm with the generic model "
               f"and {own:.1f} mm with the own model" if gen == gen and own == own else "")
    return attach(spec, CaptionSpec(
        "(a, b) Bland–Altman agreement of "
        "the incisal opening estimated from the two IMUs with the optical measurement, per "
        "open-close cycle and participant mean, (a) with a model fitted on the other "
        "participants (an unseen user) and (b) with the participant's own model." + loa_txt
        + ". (c) Opening-angle RMSE of the IMU rotation against the optics along the "
        "calibration ladder, from an in-take calibration to camera-free resets. (d) Incisor "
        "position RMSE of the camera-free translation estimates (a hinge about the condyle "
        "points, the generic and the own angle-to-translation model, and the fusion of the "
        "accelerometers with the own model) against the optics",
        [("grey dots", "open-close cycles in (a, b), participants in (c, d)"),
         ("black dots in (a, b)", "participant means"),
         ("blue dashed lines in (a, b)", "bias and repeated-measures 95 % limits of "
                                         "agreement, the blue bands their 95 % CIs"),
         ("dark line in (a, b)", "proportional bias, the regression of the difference on the "
                                 "mean, with its slope and 95 % CI in the panel"),
         ("blue diamonds and bars in (c, d)", "cohort median and 95 % BCa CI"),
         ("dashed lines in (c, d)", "the opening-angle bound (1.2°) and the incisor bound "
                                    "(2 mm)")],
        "; ".join(x for x in (f"(a, b) {imuc.get('n', '')}".strip(),
                              f"(c, d) {cfc.get('n', '')}".strip()) if len(x) > 7),
        "(a, b) bias and limits of agreement with participant-cluster bootstrap CIs. (c, d) "
        "per participant the median over the scored tasks of the task RMSE",
        evidence="(a, b) confirmatory (estimation). (c, d) exploratory (camera-free study)",
        codes=["C3a", "X5"]))
