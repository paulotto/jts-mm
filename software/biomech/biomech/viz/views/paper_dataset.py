# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Dataset-paper items of the dataset paper (paper_dataset.yaml).

* ``table_methods`` (T1): the recorded modalities with their achieved sampling rates, timing
  quality and processing -- every number from the stored QC tables or the study
  configuration, so the methods section states what the data are, not what the sensors
  were set to.
* ``emg_landscape`` (F5, D19): which muscles are active in which task, their signal quality
  and the scale of the reference contractions.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from biomech.kinematics.derivatives import WINDOW_S as DERIV_WINDOW_S
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Colorbar, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views.registry import ViewRequest, register

G = {"group": "dataset"}
STREAMS = (("IMU, head piece", "CRA"), ("IMU, mouth piece", "MTA"))


def _range(x: pd.Series, fmt: str = ".0f") -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "–"
    lo, med, hi = (format(float(v), fmt) for v in (x.min(), x.median(), x.max()))
    return med if lo == hi else f"{med} ({lo}–{hi})"


@register("table_methods", "paper", "Recorded modalities, sampling, and processing", **G)
def table_methods(req: ViewRequest) -> TableSpec:
    """T1: modality, signal, achieved rate, timing quality and processing per stream."""
    ctx = req.ctx
    lanes = ctx.table("qc", "timeline_lanes")
    iv = ctx.table("qc", "imu_intervals")
    opt = ctx.table("qc", "optical")
    cfg = ctx.config
    emg = (getattr(cfg, "signals", {}) or {}).get("emg", {})
    kin = (getattr(cfg, "signals", {}) or {}).get("kinematics", {})
    imu = (getattr(cfg, "stages", {}) or {}).get("imu", {})
    band = emg.get("bandpass_hz", [np.nan, np.nan])

    def lane(name: str) -> pd.Series:
        return lanes.loc[lanes.lane == name, "rate_hz"] if not lanes.empty else pd.Series([])

    rows: list[dict[str, Any]] = [{
        "modality": "EMG", "signal": "6 bipolar channels (masseter, temporalis, digastric; "
                                    "both sides)",
        "rate": _range(lane("EMG")), "timing": "one ADC clock for all channels",
        "processing": f"mains model at the measured line frequency; {band[0]:g}–{band[1]:g} Hz "
                      f"band-pass (order {emg.get('filter_order', 4)}, zero phase); "
                      f"{1000 * float(emg.get('envelope_s', np.nan)):.0f} ms RMS envelope; "
                      "% of the largest activity recorded"}]
    for label, slot in STREAMS:
        for kind, sig, proc in (
                ("quat", "orientation without magnetometer (game rotation vector)",
                 f"resampled to {float(imu.get('grid_hz', 200)):g} Hz (slerp); opening angle "
                 "from the relative orientation"),
                ("accel", "accelerometer (gravity included)",
                 "tilt-based opening estimate; gravity removed for the position fusion")):
            name = f"{slot} {kind}"
            r = iv[iv.stream == name] if not iv.empty and "stream" in iv else pd.DataFrame()
            timing = (f"interval P99 {_range(r['interval_p99_ms'], '.1f')} ms; long intervals "
                      f"{_range(r['long_interval_pct'], '.1f')} %") if not r.empty else "–"
            rows.append({"modality": label, "signal": sig,
                         "rate": _range(lane(f"IMU {name}")), "timing": timing,
                         "processing": proc})
    for label, body in (("Optical, head piece", "cra"), ("Optical, mouth piece", "mta")):
        o = opt[(opt.body == body) & (opt.kind == "experiment")] if not opt.empty else opt
        rows.append({
            "modality": label, "signal": "ArUco tag rigid body, 2 hardware-triggered cameras",
            "rate": _range(lane(f"optical {body.upper()}")),
            "timing": f"valid pose in {_range(100 * o['lock_frac'], '.1f')} % of the frames "
                      "of the movement tasks" if not o.empty else "–",
            "processing": f"pose low-pass {float(kin.get('lowpass_hz', np.nan)):g} Hz (order "
                          f"{kin.get('lowpass_order', 2)}, zero phase); gaps up to "
                          f"{float(kin.get('max_gap_s', np.nan)):g} s bridged; Savitzky–Golay "
                          f"derivatives ({DERIV_WINDOW_S:g} s window)"})
    cols = [Column("modality", "Modality", align="l"), Column("signal", "Signal", align="X"),
            Column("rate", "Rate (Hz)", align="l"), Column("timing", "Timing", align="X"),
            Column("processing", "Processing", align="X")]
    n = lanes.pid.nunique() if not lanes.empty and "pid" in lanes else 0
    t = TableSpec("table_methods", "", cols, rows, width="text",
                  sources=["qc:timeline_lanes", "qc:imu_intervals", "qc:optical",
                           "config:signals"])
    return attach(t, CaptionSpec(
        "the recorded modalities: achieved sampling rate, timing quality, and processing",
        [("rate", "median over the participants (range)"),
         ("timing", "sample-interval 99th percentile and share of intervals longer than "
                    "1.5 times the median (IMU); share of camera frames with a valid pose "
                    "(optical), median (range)")],
        f"n = {n} participants", "median (range) over the participants",
        evidence="descriptive"))


# Dot-matrix size classes of the median task P95 (% of the largest activity).
DOT_CLASSES = ((0.0, 10.0, 1.2, "below 10 %"), (10.0, 25.0, 2.0, "10–25 %"),
               (25.0, 50.0, 2.9, "25–50 %"), (50.0, np.inf, 3.8, "50 % and more"))


@register("emg_landscape", "paper", "Which muscles are active in which task", **G,
          choices={"form": ("strip", "box", "raincloud", "ecdf", "dotmatrix")}, form="strip")
def emg_landscape(req: ViewRequest) -> FigureSpec:
    """F5 (D19): (a) task x muscle: median over the participants of the task 95th percentile
    of the normalized envelope, with the participants in whom the muscle is active; (b) SNR
    of each muscle (maximal contraction vs rest); (c) the reference contraction relative to
    the largest task activity (scale of the tasks)."""
    from biomech.viz.views import blocks as B
    from biomech.viz.views import forms
    from biomech.viz.views import vocab as V

    ctx = req.ctx
    rel = ctx.table("emg_relevance", "relevance")
    qc = ctx.table("qc", "emg")
    refs = ctx.table("emg", "refs")
    form = req.options.get("form", "strip")
    muscles = [m for m in V.MUSCLES if not rel.empty and m in set(rel.muscle)]
    tasks = [t for t in V.TASKS if not rel.empty and t in set(rel.task)]
    # Muscles are the rows of every panel (rows line up across the figure); tasks the columns.
    z = np.full((len(muscles), len(tasks)), np.nan)
    text = []
    for i, m in enumerate(muscles):
        for j, t in enumerate(tasks):
            g = rel[(rel.task == t) & (rel.muscle == m)]
            z[i, j] = float(np.nanmedian(g.p95_pct)) if len(g) else np.nan
            text.append(f"{int(g.active.sum())}" if len(g) else "")
    if form == "dotmatrix":
        # size classes of the median task P95 (dot size), colour = share of participants in
        # whom the muscle is active
        share = np.full(z.shape, np.nan)
        for i, m in enumerate(muscles):
            for j, t in enumerate(tasks):
                g = rel[(rel.task == t) & (rel.muscle == m)]
                share[i, j] = 100.0 * g.active.mean() if len(g) else np.nan
        pa = Panel("(a) Task activity", Axis(""), Axis(""), [], legend="below",
                   colorbar=Colorbar("Participants active", V.PCT, lim=(0.0, 100.0)))
        jj, ii = np.meshgrid(np.arange(len(tasks), dtype=float),
                             np.arange(len(muscles), dtype=float))
        for lo, hi, size, lab in DOT_CLASSES:
            k = (z >= lo) & (z < hi) & np.isfinite(share)
            if k.any():
                # a grey ring behind each dot keeps cells with few active participants (the
                # light end of the colour ramp) visible; the size legend is drawn in grey
                pa.series.append(Series("scatter", jj[k], ii[k], role="muted",
                                        style={"size": size * 1.25}, label=lab))
                pa.series.append(Series("scatter", jj[k], ii[k], c=share[k], role="accent",
                                        style={"size": size}))
        pa.x.lim = (-0.6, len(tasks) - 0.4)
        pa.y.lim = (-0.6, len(muscles) - 0.4)
        pa.y.reverse = True
        form = "strip"
    else:
        # colour and number named in the title and the colour bar in words (JNER revision 3.5)
        pa = Panel("(a) Colour: activation. Number: participants active",
                   Axis(""), Axis(""), [Series("heatmap", z=z, text=text)],
                   legend="none", colorbar=Colorbar("Activation (% of the largest)", ""))
    pa.x.ticks, pa.x.ticklabels = [float(j) for j in range(len(tasks))], \
        [V.phase_label(t) for t in tasks]                 # written out
    pa.y.ticks, pa.y.ticklabels = [float(i) for i in range(len(muscles))], \
        [V.muscle_label(m) for m in muscles]
    labels = [V.muscle_label(m) for m in muscles]
    pb, marks = forms.summary(form, labels, [qc.loc[qc.muscle == m, "snr_db"].to_numpy(float)
                                             for m in muscles] if not qc.empty else [],
                              "(b) Signal-to-noise ratio", V.axis("snr_db"), seed=B.seed_of(ctx))
    ax_c = V.axis("mvc_over_task")
    ax_c.log = True                                   # a ratio: symmetric on a log axis
    pc, _ = forms.summary(form, labels, [refs.loc[refs.muscle == m, "mvc_over_task"]
                                         .to_numpy(float) for m in muscles]
                          if not refs.empty else [],
                          "(c) Reference ratio", ax_c,
                          seed=B.seed_of(ctx), ref=1.0, ref_label="equal")
    # rows coloured by muscle group (JNER revision 3.5): the summary of each row (median and
    # CI) in the group hue, closers blue, openers vermilion; participants stay grey
    for p in (pb, pc):
        for sr in p.series:
            if sr.role != "accent" or sr.y is None or not len(sr.y):
                continue
            i = int(round(float(np.asarray(sr.y, float)[0])))
            if 0 <= i < len(muscles):
                sr.role = "group:openers" if muscles[i].startswith("digastric") \
                    else "group:closers"
                if sr.kind == "scatter":
                    sr.style = {**sr.style, "ms": sr.style.get("ms", "d")}
    marks = list(marks) + [("blue and vermilion in (b, c)", "closers (masseter, temporalis) "
                                                            "and openers (digastric)")]
    n = rel.pid.nunique() if "pid" in rel else 0
    head = ("(a) Per muscle and task the median over the participants of the 95th percentile of "
            "the normalized EMG envelope (colour, % of the largest activity recorded); the number "
            "in each cell is not that value but the count of participants (of all with the "
            "channel) in whom the muscle is active in the task (its "
            "95th percentile reaches the rest mean plus three times the rest standard deviation "
            "and 5 % of the largest activity)")
    if req.options.get("form") == "dotmatrix":
        head = ("(a) per muscle and task the median over the participants of the 95th "
                "percentile of the normalized EMG envelope (dot size, in the classes of the "
                "legend, % of the largest activity recorded) and the share of participants in "
                "whom the muscle is active in the task (colour; active: its 95th percentile "
                "reaches the rest mean plus three times the rest standard deviation and 5 % of "
                "the largest activity)")
    spec = FigureSpec("emg_landscape", "Which muscles are active in which task", [pa, pb, pc],
                      rows=1, cols=3, width="text", height_in=2.4,
                      sources=["emg_relevance:relevance", "qc:emg", "emg:refs"])
    return attach(spec, CaptionSpec(
        head + ". (b) Signal-to-noise ratio of each "
        "muscle (reference contraction against rest). (c) The reference contraction (MVC for "
        "the closers, resisted opening for the openers) relative to the largest activity in "
        "the tasks (below 1: a task exceeded the reference)", marks, f"n = {n} participants",
        "(a) medians over the participants; (b, c) one value per participant with the "
        "summary of the chosen form", evidence="descriptive"))


@register("emg_opener_detectability", "dashboard", "Opener detectability vs signal quality",
          **{**G, "domain": "emg_time"})
def emg_opener_detectability(req: ViewRequest) -> FigureSpec:
    """per participant and side the share of digastric channel-cycles usable
    for the opening onset (primary detector and onset rule) against the channel's SNR --
    whether unloaded openers are detectable at all, and whether SNR explains when not."""
    from biomech.viz.views import vocab as V

    ctx = req.ctx
    o = ctx.table("timing", "onsets")
    qc = ctx.table("qc", "emg")
    from biomech.stats.cohort_misc import PRIMARY

    det = PRIMARY["detector"]                  # the C2 primary detector (TKEO)
    # thresholds from the run's parameters (E33.3: a variant must not leave them stale)
    snr_db = float((ctx.config.params("qc").get("usability") or {}).get("snr_min_db", 6.0))
    burst_db = float(ctx.config.params("timing").get("burst_min_db", 6.0))
    tasks = ("open_close", "chewing")
    panels = []
    for task in tasks:
        p = Panel(V.phase_label(task, short=False)[:1].upper() + V.phase_label(task)[1:],
                  Axis("SNR", "\\decibel"), Axis("Usable cycles", "\\percent", lim=(0.0, 100.0)),
                  [], legend="below")
        for m, ms in (("digastric_r", "o"), ("digastric_l", "s")):
            g = o[(o.task == task) & (o.muscle == m) & (o.direction == "opening") &
                  (o.detector == det) & (o.rule == "first")] if not o.empty else o
            share = g.groupby("pid").usable.mean() * 100.0 if not g.empty else None
            if share is None:
                continue
            snr = qc[qc.muscle == m].set_index("pid").snr_db if not qc.empty else None
            pids = [q for q in share.index if snr is not None and q in snr.index]
            p.series.append(Series("scatter", np.array([snr[q] for q in pids], float),
                                   np.array([share[q] for q in pids], float),
                                   role=f"entity:{m}", style={"ms": ms, "size": 1.2},
                                   label=V.muscle_label(m)))
        p.series.append(Series("vline", x=np.array([snr_db]), role="threshold",
                               label=f"{snr_db:g} dB: usable for shape"))
        panels.append(p)
    panels[0].legend = "none"                  # one legend, under the last panel
    n = o.pid.nunique() if not o.empty and "pid" in o else 0
    spec = FigureSpec("emg_opener_detectability", "Opener detectability vs signal quality",
                      panels, rows=1, cols=2, width="text", height_in=2.2,
                      sources=["timing:onsets", "qc:emg"])
    return attach(spec, CaptionSpec(
        "per participant and side the share of digastric channel-cycles usable for the "
        "opening onset (not active at the window start, burst at least "
        f"{burst_db:g} dB above rest, window more than 90 % finite; primary detector, "
        "first-onset rule) against the "
        "channel's signal-to-noise ratio (reference contraction vs rest). The openers "
        "worked against gravity only (no resisted opening in the protocol)",
        [("circles", "digastric R"), ("squares", "digastric L"),
         ("line", f"the {snr_db:g} dB shape-usability threshold")],
        f"n = {n} participants", evidence="post hoc (descriptive)"))


# -- D21: electrode impedance and jaw geometry --------------------------------------------------
def _minus(s: str) -> str:
    """Typographic minus for negative numbers (not for hyphens in words)."""
    return re.sub(r"-(?=\d)", "−", s)


def _mid(label: str) -> str:
    """A catalog label inside a phrase: lower-case first letter unless an acronym."""
    return label[:1].lower() + label[1:] if label[1:2].islower() else label


def _assoc(a: pd.DataFrame, x: str, y: str) -> pd.Series | None:
    r = a[(a.check_quantity == x) & (a.recorded_quantity == y)] if not a.empty else a
    return r.iloc[0] if len(r) else None


# emg_impedance (b): colour = muscle group, marker = muscle and side
IMP_MARK = {"masseter_r": "o", "masseter_l": "s", "temporalis_r": "^", "temporalis_l": "v",
            "digastric_r": "o", "digastric_l": "s"}


@register("emg_impedance", "dashboard", "Electrode impedance and rest signal",
          choices={"form": ("panels", "matrix")}, form="panels", **{**G, "domain": "qc"})
def emg_impedance(req: ViewRequest) -> FigureSpec:
    """D21: (a) the electrode impedance of every channel before each take, by muscle (no
    limit line: the check's contact limit is not a recording criterion and has no source;
    2026-11-09); (b) impedance against the low-frequency rest level of the
    motion take -- the one recorded quantity it predicts with participant and muscle held
    fixed (``qc_cohort:impedance_assoc``). Form ``matrix`` (J2.1): participants x
    muscles, the impedance before the motion take in each cell (which electrode was high)."""
    from biomech.viz.views import blocks as B
    from biomech.viz.views import vocab as V
    ctx = req.ctx
    chk = ctx.table("qc", "emg_check")
    if req.options.get("form") == "matrix":
        g = chk[chk["take"] == "motion"] if not chk.empty else chk
        muscles = [m for m in V.MUSCLES if not g.empty and m in set(g.muscle)]
        pids = sorted(g.pid.unique(), key=lambda q: ctx.pseudonym(q)) if not g.empty else []
        z = np.array([[float(g[(g.pid == q) & (g.muscle == m)].z_kohm.median())
                       if ((g.pid == q) & (g.muscle == m)).any() else np.nan
                       for m in muscles] for q in pids]) if pids else np.zeros((0, 0))
        p = B.heat(z, B.pseudo_rows(ctx, pids), [V.term("muscle", m, "short") for m in muscles],
                   "", fmt="{:.0f}", cb=Colorbar("Impedance", "\\kilo\\ohm",
                                                 lim=(0.0, float(np.nanmax(z))
                                                      if np.isfinite(z).any() else 1.0)))
        spec = FigureSpec("emg_impedance", "Electrode impedance and rest signal", [p],
                          width="column", height_in=3.0, sources=["qc:emg_check"])
        return attach(spec, CaptionSpec(
            "the electrode impedance of every channel before the motion take, per "
            "participant and muscle",
            [("rows", "participants"), ("columns", "muscles"),
             ("cell colour and number", "impedance in kΩ")],
            f"n = {len(pids)} participants", "one measurement per channel (no summary)",
            evidence="descriptive"))
    qc = ctx.table("qc", "emg")
    assoc = ctx.table("qc_cohort", "impedance_assoc")
    z = chk.z_kohm.to_numpy(float) if not chk.empty else np.array([])
    top = (float(np.nanmax(z)) if np.isfinite(z).any() else 1.0) * 1.08
    zax = Axis("Impedance", "\\kilo\\ohm", lim=(0.0, top))
    muscles = [m for m in V.MUSCLES if not chk.empty and m in set(chk.muscle)]
    pa = Panel("(a) Before each take", Axis("", ticks=[float(i) for i in range(len(muscles))],
                                            ticklabels=[V.term("muscle", m, "short")
                                                        for m in muscles]),
               zax, [], legend="below")
    # takes in the variant order (orange, green): blue is the closers in (b) (2026-11-05)
    for take, dx, role in (("motion", -0.15, "cat:1"), ("mvc", 0.15, "cat:2")):
        g = chk[chk["take"] == take] if not chk.empty else chk
        if g.empty:
            continue
        x = np.array([muscles.index(m) + dx for m in g.muscle], float)
        pa.series.append(Series("scatter", x, g.z_kohm.to_numpy(float), role=role,
                                style={"size": 0.9, "alpha": 0.8},
                                label=V.term("take", take)))
    pb = Panel("(b) Motion take", Axis("Rest low-frequency level", "\\micro\\volt", log=True),
               zax, [], legend="below")
    m = (chk[chk["take"] == "motion"].drop(columns="unit", errors="ignore")
         .merge(qc.drop(columns="unit", errors="ignore"), on=["pid", "muscle"])
         if not chk.empty and not qc.empty else pd.DataFrame())
    # colour = muscle group, marker = side: six muscle hues were too many to tell apart
    #
    for mu in muscles:
        g = m[m.muscle == mu] if not m.empty else m
        if not g.empty:
            grp = "openers" if mu.startswith("digastric") else "closers"
            pb.series.append(Series("scatter", g.rest_lf_uv.to_numpy(float),
                                    g.z_kohm.to_numpy(float), role=f"group:{grp}",
                                    style={"size": 1.0, "ms": IMP_MARK.get(mu, "o"),
                                           "alpha": 0.85},
                                    label=V.muscle_label(mu)))
    r = _assoc(assoc, "z_kohm", "rest_lf_uv")
    s = _assoc(assoc, "z_kohm", "snr_db")
    stat = []
    if r is not None:
        stat.append(f"impedance vs rest low-frequency level: pooled Spearman ρ = "
                    f"{B.num(r.rho_pooled, 2)} (95 % CI {B.ci(r.rho_pooled_lo, r.rho_pooled_hi, 2)}), "
                    f"rank slope with participant and muscle fixed {B.num(r.rho_adj, 2)} "
                    f"({B.ptext(r.p_adj)}, Holm {B.pval(r.p_holm)})")
    if s is not None:
        stat.append(f"vs SNR: pooled ρ = {B.num(s.rho_pooled, 2)}, adjusted slope {B.num(s.rho_adj, 2)} "
                    f"({B.ptext(s.p_adj)})")
    stat = [_minus(x) for x in stat]
    n = chk.pid.nunique() if not chk.empty else 0
    spec = FigureSpec("emg_impedance", "Electrode impedance and rest signal", [pa, pb],
                      rows=1, cols=2, width="text", height_in=2.4,
                      sources=["qc:emg_check", "qc:emg", "qc_cohort:impedance_assoc"])
    return attach(spec, CaptionSpec(
        "electrode–skin impedance of each EMG channel from the AC lead-off tone (31.25 Hz) "
        "of the check run before each take, (a) by muscle for both takes and (b) against the "
        "low-frequency (1–20 Hz) rest level of the motion take, log scale. No channel was "
        "excluded on the check",
        [("circles/squares in (a)", "motion / MVC take"),
         ("colour in (b)", "muscle group (blue closers: masseter, temporalis; vermilion "
                           "openers: digastric)"),
         ("marker in (b)", "muscle and side (masseter and digastric: circle right, square "
                           "left; temporalis: up / down triangle right / left)")],
        f"n = {n} participants, 6 channels per take", statistic="; ".join(stat),
        evidence="post hoc (descriptive)"))


@register("table_impedance", "paper", "Impedance and recorded signal quality", **G)
def table_impedance(req: ViewRequest) -> TableSpec:
    """D21: does the pre-take check predict the recorded rest signal? Pooled and adjusted
    rank associations (``qc_cohort:impedance_assoc``) under the impedance summary."""
    from biomech.viz.views import vocab as V

    from biomech.viz.views import blocks as B

    ctx = req.ctx
    a = ctx.table("qc_cohort", "impedance_assoc")
    rows = []
    for _, r in a.iterrows() if not a.empty else []:
        rows.append({"rel": f"{V.label(r.check_quantity)} vs {_mid(V.label(r.recorded_quantity))}",
                     "rho": r.rho_pooled,
                     "ci": B.ci(r.rho_pooled_lo, r.rho_pooled_hi, 2),
                     "wp": r.rho_within_pid, "wm": r.rho_within_muscle, "b": r.rho_adj,
                     "p": r.p_holm})
    cols = [Column("rel", "Impedance check vs recorded signal", align="l"),
            Column("rho", "$\\rho_s$", decimals=2, group="Pooled"),
            Column("ci", "95 % CI", align="c", group="Pooled"),
            Column("wp", "Participants", decimals=2, group="$\\rho$ within"),
            Column("wm", "Muscles", decimals=2, group="$\\rho$ within"),
            Column("b", "Slope", decimals=2, group="Participant and muscle fixed"),
            Column("p", "$p_{\\mathrm{Holm}}$", decimals=3, group="Participant and muscle fixed")]
    n = int(a.n.max()) if not a.empty else 0
    k = int(a.n_pid.max()) if not a.empty else 0
    t = TableSpec("table_impedance", "", cols, rows,
                  notes="Pooled: Spearman ρ over all channels, CI from resampling "
                        "participants; within: ranks centred per participant or per "
                        "muscle; both fixed: rank regression on participant and muscle, "
                        "Holm over the relations. Descriptive (post hoc).",
                  sources=["qc_cohort:impedance_assoc"], width="text")
    return attach(t, CaptionSpec(
        "does the pre-take electrode check predict the rest signal recorded in the motion take? "
        "Each row relates one check quantity to one recorded quantity across all channels",
        n=f"n = {n} channels of {k} participants",
        statistic="Spearman rank correlation pooled over channels with its 95 % CI, the "
                  "correlation within participants and within muscles, and the slope of a "
                  "rank regression with participant and muscle as fixed effects",
        test="Holm over the relations", evidence="post hoc (descriptive)"))


GEOMETRY_ROWS = ("bonwill_side_mm", "bonwill_ratio", "balkwill_deg", "lower_arch_depth_mm",
                 "upper_arch_depth_mm", "condyle_height_mm", "hinge_radius_mm",
                 "hinge_axis_dx_mm", "hinge_axis_dz_mm", "hinge_axis_offset_mm",
                 "hinge_axis_angle_deg")


@register("table_geometry", "paper", "Jaw geometry", **G)
def table_geometry(req: ViewRequest) -> TableSpec:
    """D21: per-participant jaw geometry summarized over the cohort, all participants and
    those with plausible palpated condyles (condyle-based rows only)."""
    from biomech.viz.style import plain_unit
    from biomech.viz.views import vocab as V

    ctx = req.ctx
    rv = ctx.table("stats_kinematics", "reference_values")
    cs = ctx.table("stats_kinematics", "condyle_sensitivity")
    geo = ctx.table("kinematics", "geometry")

    def num(v: float, d: int) -> str:
        return _minus(f"{v:.{d}f}")

    def iqr(r: pd.Series, d: int) -> str:
        return f"{num(r['median'], d)} [{num(r.q1, d)}, {num(r.q3, d)}]"

    rows = []
    for key in GEOMETRY_ROWS:
        a = rv[(rv.metric == key) & (rv.group == "all")] if not rv.empty else rv
        if a.empty:
            continue
        d = 2 if key == "bonwill_ratio" else 1
        p = cs[(cs.metric == key) & (cs.group == "plausible_condyles")] if not cs.empty else cs
        g = geo[key].dropna() if key in geo else pd.Series([], dtype=float)
        rows.append({"q": V.label(key), "u": plain_unit(V.unit(key)), "all": iqr(a.iloc[0], d),
                     "pl": iqr(p.iloc[0], d) if not p.empty else "--",
                     "rng": f"{num(g.min(), d)} to {num(g.max(), d)}" if len(g) else ""})
    n = int(rv[(rv.group == "all") & rv.metric.isin(GEOMETRY_ROWS)].n.max()) if not rv.empty \
        else 0
    npl = int(cs[(cs.group == "plausible_condyles") & cs.metric.isin(GEOMETRY_ROWS)].n.max()) \
        if not cs.empty and cs.metric.isin(GEOMETRY_ROWS).any() else 0
    cols = [Column("q", "Quantity", align="l"), Column("u", "Unit", align="l"),
            Column("all", f"All (n\u00a0=\u00a0{n})", align="c", group="Median [IQR]"),
            Column("pl", f"Plausible condyles (n\u00a0=\u00a0{npl})", align="c",
                   group="Median [IQR]"),
            Column("rng", "Range", align="c")]
    t = TableSpec("table_geometry", "", cols, rows,
                  notes="Maxillary frame (x anterior, z up); occlusal plane, arch depths, and "
                        "intermolar widths through the incisor and the second molars. "
                        "Condyle-based rows rest on "
                        "skin-palpated condyle points, so the plausible-condyle column "
                        "is the one to read; the arch depths and the hinge radius do not "
                        "use the condyles (--). The best-fit hinge axis is the equivalent "
                        "fixed axis of opening, not the anatomical condylar axis.",
                  sources=["stats_kinematics:reference_values",
                           "stats_kinematics:condyle_sensitivity", "kinematics:geometry"],
                  width="text")
    return attach(t, CaptionSpec(
        "jaw geometry of the participants, computed from the digitized landmarks (incisors, "
        "palpated condyles) and the recorded openings: the classic articulator measures, the "
        "arch depths, and the location of the best-fit opening axis",
        n=f"n = {n} participants ({npl} with anatomically plausible condyle points)",
        statistic="per participant one value; median [interquartile range (IQR)] and range "
                  "over the participants", evidence="descriptive"))


HINGE_AXIS_FORMS = ("scatter", "arrows", "offset")


@register("kin_hinge_axis_location", "dashboard", "Best-fit opening axis vs palpated condyles",
          choices={"form": HINGE_AXIS_FORMS}, form="scatter",
          **{**G, "domain": "kinematics"})
def kin_hinge_axis_location(req: ViewRequest) -> FigureSpec:
    """D21: where the best-fit hinge axis of opening pierces the sagittal plane, relative to
    the palpated condyle midpoint, per participant (plausible condyles filled). Forms
    (J2.1): ``scatter`` the pierce points; ``arrows`` one arrow per participant from
    the condyle midpoint to its axis; ``offset`` the offset distance and the axis angle as
    strips with the median and its CI, plausible vs implausible condyles."""
    from biomech.viz.views import blocks as B
    from biomech.viz.views import vocab as V
    from biomech.viz.views.results_kinematics import who

    ctx = req.ctx
    form = req.options.get("form") or "scatter"
    geo = ctx.table("kinematics", "geometry")
    an = ctx.table("kinematics", "anatomy")
    ok = set(an.loc[pd.to_numeric(an.condyles_plausible, errors="coerce").eq(1), "pid"]) \
        if not an.empty else set()
    groups = []
    for flag, role, lab in ((True, "primary", "plausible condyles"),
                            (False, "secondary", "implausible condyles")):
        g = geo[geo.pid.isin(ok) == flag].dropna(subset=["hinge_axis_dx_mm"]) \
            if not geo.empty and "hinge_axis_dx_mm" in geo else pd.DataFrame()
        groups.append((g, role, lab))
    n = int(geo.hinge_axis_dx_mm.notna().sum()) if "hinge_axis_dx_mm" in geo else 0
    srcs = ["kinematics:geometry", "kinematics:anatomy"]
    shows = ("where the best-fit fixed hinge axis of the openings pierces the sagittal plane, "
             "relative to the palpated condyle midpoint at closure (maxillary frame). The fixed "
             "axis that best reproduces rotation plus condylar translation lies below and "
             "behind the condyles, about perpendicular to their path")
    if form == "offset":
        cols = (("hinge_axis_offset_mm", "Axis offset from the condyle midpoint", "Offset",
                 V.MM),
                ("hinge_axis_angle_deg", "Axis angle to the condyle line", "Angle", V.DEG))
        seed = B.seed_of(ctx)
        panels = []
        for k, (col, title, axlab, unit) in enumerate(cols):
            labs = [lab for g, _, lab in groups if not g.empty and col in g]
            vals = [pd.to_numeric(g[col], errors="coerce").to_numpy(float)
                    for g, _, lab in groups if not g.empty and col in g]
            q = B.strip_ci(labs, vals, f"({'ab'[k]}) {title}", Axis(axlab, unit), seed=seed,
                           roles=[r for g, r, _ in groups if not g.empty and col in g])
            panels.append(q)
        spec = FigureSpec("kin_hinge_axis_location", "Best-fit opening axis vs palpated "
                          "condyles", panels, rows=2, cols=1, width="column", height_in=1.7,
                          sources=srcs)
        return attach(spec, CaptionSpec(
            shows + "; (a) distance of the axis from the condyle midpoint, (b) angle between "
            "the axis and the line through the two condyle points",
            [("dots", "participants"), ("diamond and line", "median with its 95 % CI"),
             ("rows", "participants with plausible / implausible condyle points")],
            f"n = {n} participants", "median with bootstrap 95 % CI over participants",
            evidence="post hoc (descriptive)"))
    p = Panel("", Axis("Anterior offset", V.MM), Axis("Superior offset", V.MM), [],
              legend="below", equal=True)
    for g, role, lab in groups:
        if g.empty:
            continue
        x, z = g.hinge_axis_dx_mm.to_numpy(float), g.hinge_axis_dz_mm.to_numpy(float)
        if form == "arrows":
            p.series.append(Series("arrow", np.zeros_like(x), np.zeros_like(z),
                                   d=np.column_stack([x, z]), role=role, label=lab))
        p.series.append(Series("scatter", x, z, role=role,
                               label=None if form == "arrows" else lab,
                               text=[who(ctx, q) for q in g.pid]))
    p.series.append(Series("scatter", np.array([0.0]), np.array([0.0]), role="accent",
                           style={"ms": "x", "size": 1.6}, label="palpated condyle midpoint"))
    spec = FigureSpec("kin_hinge_axis_location", "Best-fit opening axis vs palpated condyles",
                      [p], width="column", height_in=2.6, sources=srcs)
    marks = [("cross", "palpated condyle midpoint"),
             ("blue dots", "participants with anatomically plausible condyle points"),
             ("orange squares", "participants with implausible condyle points")]
    if form == "arrows":
        marks.insert(1, ("arrows", "from the condyle midpoint to each participant's axis"))
    return attach(spec, CaptionSpec(shows, marks, f"n = {n} participants",
                                    "per participant one axis (no summary)",
                                    evidence="post hoc (descriptive)"))


@register("table_emg_activation", "paper", "Muscle activation and signal quality", **G)
def table_emg_activation(req: ViewRequest) -> TableSpec:
    """T6 (D19, A3.2-A3.4): per muscle, the task activity (median over the participants of
    the task 95th percentile, % of the largest activity) with the participants in whom the
    muscle is active, the SNR and the reference contraction relative to the task maximum --
    the numbers behind F5 (``emg_landscape``) as a table."""
    from biomech.viz.views import blocks as B
    from biomech.viz.views import vocab as V

    ctx = req.ctx
    rel = ctx.table("emg_relevance", "relevance")
    qc = ctx.table("qc", "emg")
    refs = ctx.table("emg", "refs")
    muscles = [m for m in V.MUSCLES if not rel.empty and m in set(rel.muscle)]
    tasks = [t for t in V.TASKS if not rel.empty and t in set(rel.task)]
    seed = B.seed_of(ctx)
    rows = []
    for m in muscles:
        row: dict[str, Any] = {"muscle": V.muscle_label(m)}
        for t in tasks:
            g = rel[(rel.task == t) & (rel.muscle == m)]
            row[t] = (f"{np.nanmedian(g.p95_pct):.0f} ({int(g.active.sum())}/{len(g)})"
                      if len(g) else "–")
        snr = qc.loc[qc.muscle == m, "snr_db"].to_numpy(float) if not qc.empty else []
        snr = np.asarray(snr, float)[np.isfinite(snr)]
        if snr.size:
            lo, hi = B.median_ci(snr, seed)
            row["snr"] = f"{np.median(snr):.0f} [{lo:.0f}, {hi:.0f}]"
        else:
            row["snr"] = "–"
        r = refs.loc[refs.muscle == m, "mvc_over_task"].to_numpy(float) if not refs.empty \
            else []
        r = np.asarray(r, float)[np.isfinite(r)]
        row["mvc"] = (f"{np.median(r):.2f} ({int((r < 1).sum())}/{r.size})" if r.size
                      else "–")
        rows.append(row)
    cols = ([Column("muscle", "Muscle", align="l")]
            + [Column(t, V.phase_label(t)[:1].upper() + V.phase_label(t)[1:], align="c",
                      group="Task activity, % of max "
                      "(participants active)") for t in tasks]
            + [Column("snr", "SNR (dB)", align="c"),
               Column("mvc", "Reference / task max", align="c")])
    n = rel.pid.nunique() if "pid" in rel else 0
    t = TableSpec("table_emg_activation", "", cols, rows, width="text",
                  sources=["emg_relevance:relevance", "qc:emg", "emg:refs"])
    return attach(t, CaptionSpec(
        "per muscle the activity in each task, its signal-to-noise ratio, and the scale of the "
        "reference contraction",
        [("task activity", "median over the participants of the task 95th percentile of the "
                           "normalized envelope (% of the largest activity recorded); in "
                           "brackets the participants in whom the muscle is active in the task "
                           "(95th percentile at least the rest mean plus three rest SD and 5 % "
                           "of the largest activity)"),
         ("SNR", "reference contraction vs rest, median [95 % BCa CI]"),
         ("reference / task max", "the reference contraction (MVC, closers; resisted "
                                  "opening, openers) relative to the largest task activity, "
                                  "median; in brackets the participants whose tasks exceeded "
                                  "the reference")],
        f"n = {n} participants",
        "medians over the participants; signal-to-noise ratio (SNR) with its 95 % BCa CI",
        evidence="post hoc (descriptive)"))
