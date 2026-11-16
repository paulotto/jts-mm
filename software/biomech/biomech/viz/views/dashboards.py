# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Analysis dashboards (viewer) as specs; every one is exportable to TikZ like a paper figure.

Forms follow the data's job: heatmaps for participant x muscle grids (sequential
ramp, values printed), horizontal dot/strip plots where category names are long
(labels on the y axis, never rotated), the emphasis form (accent median over
grey individuals) for distributions.
"""

from __future__ import annotations

import numpy as np

from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.views import blocks as B
from biomech.viz.views import forms
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.vocab import EMG_NORM_SHORT, MUSCLE_LABEL, MUSCLES, TASK_LABEL
from biomech.results import COHORT_UNIT

MUSCLES = list(MUSCLES)                       # kept for importers of the old names


def _heat(z: np.ndarray, rows: list[str], cols: list[str], title: str, fmt: str = "{:.0f}",
          cb: Colorbar | None = None) -> Panel:
    return B.heat(z, rows, cols, title, fmt=fmt, cb=cb)


def _curve_marks(centre: str = "mean") -> list[tuple[str, str]]:
    """The band marks of :func:`B.add_curves_ci` as caption (glyph, meaning) pairs."""
    return [tuple(x.split(": ", 1)) for x in B.curve_ci_caption(centre).split("; ")]


def _strip(labels: list[str], values: list[np.ndarray], title: str, xaxis: Axis,
           ref: float | None = None, ref_label: str | None = None) -> Panel:
    """Horizontal strip: one row per category, grey dots + accent median."""
    return B.strip(labels, values, title, xaxis, ref=ref, ref_label=ref_label)


@register("landmark_qc", "dashboard", "Landmark quality", domain="qc")
def landmark_qc(req: ViewRequest) -> FigureSpec:
    """Hold quality of all eight landmarks and plausibility of the palpated condyles."""
    import pandas as pd

    ctx = req.ctx
    names = ["mand_point_1", "mand_point_2", "mand_point_3", "max_point_1", "max_point_2",
             "max_point_3", "tmj_left", "tmj_right"]
    short = ["lower inc.", "lower L6", "lower R6", "upper inc.", "upper L6", "upper R6",
             "condyle L", "condyle R"]
    pids = ctx.pids()
    lm = ctx.table("kinematics", "landmarks")
    # Empty stage tables (kinematics not run, no participant) give empty panels, not errors.
    if lm.empty or not {"pid", "name", "mad_mm"} <= set(lm.columns):
        z = pd.DataFrame(np.nan, index=pids, columns=names)
    else:
        lm = lm[~lm.name.astype(str).str.endswith("_hp")]
        z = lm.pivot_table(index="pid", columns="name", values="mad_mm", aggfunc="first"
                           ).reindex(pids).reindex(columns=names)
    an = ctx.table("kinematics", "anatomy")
    an = an.set_index("pid").reindex(pids) if "pid" in an.columns else pd.DataFrame(index=pids)
    left = an.get("condyle_incisor_left_mm", pd.Series(np.nan, index=an.index)).to_numpy(float)
    right = an.get("condyle_incisor_right_mm", pd.Series(np.nan, index=an.index)
                   ).to_numpy(float)
    idx = np.array([ctx.pseudonym(p) for p in pids], float)
    try:
        lim = float((ctx.config.signals.get("kinematics") or {})["min_condyle_incisor_mm"])
    except (AttributeError, KeyError, TypeError, ValueError):
        lim = 70.0
    p2 = Panel("Condyle-incisor distance", Axis("Participant"),
               Axis("Distance", "\\milli\\metre"),
               [Series("scatter", idx - 0.12, left, label="left condyle", role="side:left"),
                Series("scatter", idx + 0.12, right, label="right condyle", role="side:right"),
                Series("hline", y=np.array([lim]), role="threshold",
                       label="plausibility limit")])
    p2.x.ticks = [float(i) for i in idx]
    p2.x.ticklabels = [str(int(i)) for i in idx]
    if not np.isfinite(np.r_[left, right]).any():
        p2.note = "no anatomy table (kinematics not run)"
    n_low = int(np.sum(np.r_[left, right] < lim))
    return FigureSpec("landmark_qc", "Landmark quality", [
        _heat(z.to_numpy(float), B.pseudo_rows(ctx, pids), short, "Hold MAD (mm)", "{:.2f}"),
        p2], rows=1, cols=2, width="text", height_in=B.rows_height(len(pids), per=0.16,
                                                                   extra=0.9, lo=2.2),
        caption=f"Landmark digitization quality of the {len(pids)} participants: median "
                "absolute deviation of each landmark over its hold (mm), and the distance "
                "of each skin-palpated condyle to the lower incisor against the "
                f"plausibility limit ({lim:g} mm; {n_low} condyle(s) below it).",
        sources=["kinematics:landmarks", "kinematics:anatomy"])


@register("qc_emg", "dashboard", "EMG quality", domain="qc")
def qc_emg(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    e = ctx.table("qc", "emg")
    pids = ctx.pids()
    rows = B.pseudo_rows(ctx, pids)
    cols = [MUSCLE_LABEL[m] for m in MUSCLES]
    snr = e.pivot(index="pid", columns="muscle", values="snr_db").reindex(pids).reindex(
        columns=MUSCLES)
    mk = e.pivot(index="pid", columns="muscle", values="mask_frac_task").reindex(pids).reindex(
        columns=MUSCLES)
    return FigureSpec("qc_emg", "EMG quality", [
        _heat(snr.to_numpy(float), rows, cols, "SNR, reference vs rest (dB)"),
        _heat(100 * mk.to_numpy(float), rows, cols, "Masked task samples (%)")],
        rows=1, cols=2, width="text",
        height_in=B.rows_height(len(pids), per=0.16, extra=0.9, lo=2.0),
        caption=f"EMG channel quality of the {len(pids)} participants: SNR of the reference "
                "contraction over rest (dB) and the share of task samples masked by the "
                "analysis artifact rules (%).", sources=["qc:emg"])


@register("activation_map", "dashboard", "Activation map", domain="emg_time",
          choices={"form": ("heat", "lines")}, form="heat")
def activation_map(req: ViewRequest) -> FigureSpec:
    """Cohort median of the per-participant median activation, task x muscle (amplitude-usable
    channels only, ``stats_emg:activation_cohort``). Form ``lines`` (J2.1): one line
    per muscle across the tasks, coloured by muscle."""
    a = req.ctx.table("stats_emg", "activation_cohort")
    tasks = [t for t in TASK_LABEL if t in set(a.task)]
    z = a.pivot(index="task", columns="muscle", values="median").reindex(tasks).reindex(
        columns=MUSCLES)
    n = int(a.n.max()) if not a.empty and "n" in a else 0
    if req.options.get("form") == "lines":
        x = np.arange(len(tasks), dtype=float)
        p = Panel("", Axis(""), Axis(f"Median activation ({EMG_NORM_SHORT})"), [],
                  legend="right")
        for m in MUSCLES:
            if m in z.columns and np.isfinite(z[m].to_numpy(float)).any():
                p.series.append(Series("line", x, z[m].to_numpy(float), role=f"entity:{m}",
                                       label=MUSCLE_LABEL[m], style={"ms": "o"}))
        p.x.ticks, p.x.ticklabels = list(x), [TASK_LABEL[t] for t in tasks]
        spec = FigureSpec("activation_map", "Activation map", [p], width="text",
                          height_in=2.2, sources=["stats_emg:activation_cohort"])
        return attach(spec, CaptionSpec(
            "which muscle is active in which task: per task and muscle the typical level of "
            f"the normalized EMG envelope ({EMG_NORM_SHORT}), one line per muscle",
            [("lines", "muscles (colour)"), ("x axis", "tasks")],
            n=f"n = up to {n} participants per point (amplitude-usable channels only)",
            statistic="per participant the median of the envelope over the task; per point "
                      "the median over the participants", evidence="descriptive"))
    spec = FigureSpec("activation_map", "Activation map", [
        _heat(z.to_numpy(float), [TASK_LABEL[t] for t in tasks],
              [MUSCLE_LABEL[m] for m in MUSCLES], f"Median activation ({EMG_NORM_SHORT})")],
        width="column", height_in=2.0, sources=["stats_emg:activation_cohort"])
    return attach(spec, CaptionSpec(
        "which muscle is active in which task: per task and muscle the typical level of the "
        f"normalized EMG envelope ({EMG_NORM_SHORT})",
        [("cell colour", "the activation level (darker: higher)"),
         ("cell text", "the value")],
        n=f"n = up to {n} participants per cell (amplitude-usable channels only)",
        statistic="per participant the median of the envelope over the task; per cell the "
                  "median over the participants", evidence="descriptive"))


@register("imu_agreement", "dashboard", "IMU agreement", domain="imu",
          choices={"form": ("strip", "box", "violin", "raincloud", "ecdf")}, form="strip")
def imu_agreement(req: ViewRequest) -> FigureSpec:
    a = req.ctx.table("imu", "agreement")
    a = a[~a.is_calibration]
    tasks = [t for t in TASK_LABEL if t in set(a.task) and t != "rest"]
    form = req.options.get("form", "strip")
    panels, marks = [], []
    for est in ("quat", "comp", "tilt"):
        col = f"rmse_{est}_deg"
        if col not in a:
            continue
        el = V.term('estimator', est, 'short')
        p, marks = forms.summary(form, [TASK_LABEL[t] for t in tasks],
                                 [a[a.task == t][col].to_numpy() for t in tasks],
                                 f"({'abc'[len(panels)]}) {el[:1].upper() + el[1:]}",
                                 Axis("RMSE vs optical", "\\degree", log=True),
                                 seed=B.seed_of(req.ctx))
        p.sharex = "rmse"                     # one log axis for the three estimators
        panels.append(p)
    n = a.pid.nunique() if "pid" in a else 0
    spec = FigureSpec("imu_agreement", "IMU agreement", panels, rows=1, cols=len(panels),
                      width="text", height_in=B.rows_height(len(tasks), per=0.25, lo=1.6),
                      sources=["imu:agreement"])
    return attach(spec, CaptionSpec(
        "error of the IMU opening angle against the optical angle per task, for the "
        "(a) quaternion, (b) complementary and (c) accelerometer-tilt estimators (one log "
        "axis for all panels)", marks,
        f"n = {n} participants (RMSE over each participant's cycles of the task)",
        "median and 95 % BCa CI"))


@register("timing_detectors", "dashboard", "Detector sensitivity", domain="timing")
def timing_detectors(req: ViewRequest) -> FigureSpec:
    from biomech.timing.lead import TASKBASE_S

    t = req.ctx.table("stats_timing", "timing_spec")
    t = t[t.outcome == "open_close|opening|lead"]
    variants = sorted(t.variant.unique())
    labels = [v.replace("detector=", "").replace("|mech=", ", ").replace("lead_", "")
              .replace("_ms", "").replace("hodges_bui", "Hodges-Bui").replace("tkeo", "TKEO")
              .replace("omega", "ω").replace(", ms", ", earliest")
              .replace("|onset=first", ", first")
              .replace("|onset=drive_taskbase", ", drive, in-task thr.")
              .replace("|onset=drive", ", drive")
              for v in variants]
    p = B.strip_ci(labels, [t[t.variant == v].value.to_numpy() for v in variants],
                   "Opening lead per processing variant",
                   Axis("EMG lead before motion", "\\milli\\second"), ref=0.0,
                   seed=B.seed_of(req.ctx))
    n = t.pid.nunique() if "pid" in t else 0
    spec = FigureSpec("timing_detectors", "Detector sensitivity", [p], width="text",
                      height_in=B.rows_height(len(variants), per=0.16, extra=0.3, lo=1.6),
                      sources=["stats_timing:timing_spec"])
    return attach(spec, CaptionSpec(
        f"open-close opening lead under each of the {len(variants)} processing variants "
        "(detector, mechanical onset source, onset rule; > 0: EMG first)",
        [("dots", "participants"), ("diamond", "median"),
         ("line", "95 % BCa bootstrap CI of the median"),
         ("dashed line", "zero lead")], f"n = {n} participants",
        "per participant the median lead over usable channel-cycles; cohort median with its "
        "95 % BCa CI", evidence="exploratory (specification sensitivity)",
        notes=["Row labels: detector (TKEO: Teager–Kaiser energy operator; envelope; "
               "Hodges–Bui), mechanical onset (earliest: the earliest mechanical onset; acc: "
               "IMU angular acceleration; ω: IMU angular velocity; opt: optical opening "
               "velocity) and onset rule (first: first onset; drive: driving burst; in-task "
               f"thr.: driving burst with the threshold from the first {TASKBASE_S:g} s of the "
               "onset search window)"]))


@register("ml_overview", "dashboard", "Decoding overview", domain="ml")
def ml_overview(req: ViewRequest) -> FigureSpec:
    lo = req.ctx.table("ml_eval", "loso")
    combos = sorted({(m, f) for m, f in zip(lo.model, lo.feature_set)})
    labels = [f"{m.upper()}, {f.replace('_', '+').replace('nodig', 'no dig.')}"
              for m, f in combos]
    p = B.strip_ci(labels, [lo[(lo.model == m) & (lo.feature_set == f)].balanced_accuracy
                            .to_numpy() for m, f in combos], "Unseen-subject balanced accuracy",
                   Axis("Balanced accuracy", lim=(0.0, 1.0)), ref=1 / 7, ref_label="chance",
                   seed=B.seed_of(req.ctx))
    n = lo.pid.nunique() if "pid" in lo else 0
    return FigureSpec("ml_overview", "Decoding overview", [p], width="column",
                      height_in=B.rows_height(len(combos), per=0.16, extra=0.3, lo=1.6),
                      caption="Balanced accuracy of 7-class intent decoding for an unseen "
                              f"subject (leave-one-subject-out, {n} folds) per model and "
                              "feature set; chance 1/7. Dots: participants, diamond: median, "
                              f"{B.MEDIAN_CI_CAPTION}.", sources=["ml_eval:loso"])


# -- frequency domain (X1, exploratory) --------------------------------------------------------
def _bin_hz(grid) -> str:
    """'10-Hz' from the stored spectral grid (the bin width of the spectral shape)."""
    g = np.asarray(grid if grid is not None else [], float)
    return f"{float(np.median(np.diff(g))):g}-Hz" if g.size > 1 else "frequency"



@register("emg_spectra", "dashboard", "EMG spectral shape", params=("phase",), phase="chewing",
          domain="emg_freq", phases=("chewing", "open_close", "mvc_closing"))
def emg_spectra(req: ViewRequest) -> FigureSpec:
    """Spectral shape per muscle: participants (grey), cohort mean ± SD, SPM bars."""

    task = req.phase or "chewing"
    arr = req.ctx.arrays("stats_emg", COHORT_UNIT)
    grid = arr.get("spectral__grid_hz")
    spm = req.ctx.table("stats_emg", "spectral_spm")
    panels = []
    n_sub = 0
    for m in MUSCLES:
        p = Panel(MUSCLE_LABEL[m], Axis("Frequency", "\\hertz"), Axis("log10 power share"),
                  legend="none", sharex="f")
        key = f"spectral__{task}__{m}"
        if grid is not None and f"{key}__subjects" in arr:
            S = np.atleast_2d(arr[f"{key}__subjects"])
            n_sub = max(n_sub, len(S))
            for row in S:
                p.series.append(Series("line", grid, row, role="context"))
            mu, sd = arr[f"{key}__mean"], arr[f"{key}__sd"]
            p.series.append(Series("band", grid, mu - sd, mu + sd, role="band",
                                   style={"alpha": B.SD_BAND_ALPHA}))
            B.add_ci_band(p, grid, S, seed=B.seed_of(req.ctx), where=np.isfinite(mu))
            p.series.append(Series("line", grid, mu, role="accent"))
            if m.startswith("masseter") and task == "chewing" and not spm.empty:
                r = spm[spm.test == "masseter_L_vs_R_chewing_spectrum"]
                if not r.empty:
                    B.add_spm_bars(p, [("masseter L ≠ R", r.clusters_hz.iloc[0], "context")])
        else:
            p.note = "no data"
        panels.append(p)
    sig = any(s.kind == "sigbar" for q in panels for s in q.series)
    spec = FigureSpec("emg_spectra", "EMG spectral shape", panels, rows=2, cols=3, width="text",
                      height_in=1.35, sources=["emg:spectral", "stats_emg:spectral_spm"])
    return attach(spec, CaptionSpec(
        f"spectral shape of the cleaned EMG ({task.replace('_', ' ')}): log10 share of in-band "
        f"power per {_bin_hz(grid)} bin",
        [("grey lines", "participants"), ("black line", "cohort mean")] + _curve_marks()
        + ([("bars", "SPM masseter L vs R, p < 0.05")] if sig else []),
        f"n = up to {n_sub} shape-usable participants per muscle",
        "per participant the mean spectrum over the task's cycles; cohort mean ± SD and the "
        "95 % CI of the mean" + ("; paired SPM t-test, cluster p < 0.05" if sig
                                  else ""),
        evidence="exploratory"))


@register("spectral_variability", "dashboard", "Spectral variability (MNF)",
          params=("phase",), phase="chewing", domain="emg_freq",
          phases=("chewing", "open_close", "mvc_closing"))
def spectral_variability(req: ViewRequest) -> FigureSpec:
    """Mean frequency: spread between participants vs within a participant (S2)."""
    task = req.phase or "chewing"
    sc = req.ctx.table("stats_emg", "spectral_cohort")
    sp = req.ctx.table("emg", "spectral")
    sc = sc[sc.task == task] if not sc.empty else sc
    order = [m for m in MUSCLES if not sc.empty and (sc.muscle == m).any()]
    p1 = B.strip_ci([MUSCLE_LABEL[m] for m in order],
                    [sp[(sp.task == task) & (sp.muscle == m)].mnf_hz.to_numpy() for m in order],
                    "Mean frequency per participant", Axis("MNF", "\\hertz"),
                    seed=B.seed_of(req.ctx))
    p2 = Panel("Between / within participants", Axis("SD of MNF", "\\hertz"), Axis(""),
               legend="auto")
    for i, m in enumerate(order):
        r = sc[sc.muscle == m].iloc[0]
        p2.series.append(Series("line", np.array([r.mnf_sd_within_hz, r.mnf_sd_between_hz]),
                                np.array([i, i], float), role="context"))
    if order:
        s = sc.set_index("muscle").loc[order]
        p2.series += [Series("scatter", s.mnf_sd_within_hz.to_numpy(), np.arange(len(order)),
                             role="cat:1", label="within (0.5-s windows)"),
                      Series("scatter", s.mnf_sd_between_hz.to_numpy(), np.arange(len(order)),
                             role="cat:0", label="between participants")]
        p2.y.ticks, p2.y.ticklabels = list(range(len(order))), [MUSCLE_LABEL[m] for m in order]
        p2.y.reverse = True
    ratio = (sc.mnf_sd_between_hz / sc.mnf_sd_within_hz).to_numpy(float) if not sc.empty \
        else np.array([])
    ratio = ratio[np.isfinite(ratio)]
    rtxt = (f"; the between-participant SD is {ratio.min():.1f}–{ratio.max():.1f} times the "
            "within-participant SD across muscles" if ratio.size else "")
    return FigureSpec("spectral_variability", "Spectral variability", [p1, p2], rows=1, cols=2,
                      width="text", height_in=B.rows_height(len(order), per=0.2, lo=1.4),
                      caption=f"EMG mean frequency (MNF) in {TASK_LABEL.get(task, task)} per "
                              "participant (dots; diamond: median, "
                              f"{B.MEDIAN_CI_CAPTION}) and the between- vs "
                              f"within-participant SD{rtxt}: spectral shape depends on electrode "
                              "placement and tissue, an argument for standardized electrode "
                              "protocols (exploratory).",
                      sources=["emg:spectral", "stats_emg:spectral_cohort"])


@register("movement_bandwidth", "dashboard", "Movement bandwidth", domain="imu")
def movement_bandwidth(req: ViewRequest) -> FigureSpec:
    """Frequencies holding 50/95/99 % of the jaw angular-velocity power, per task: raw IMU,
    and the part confirmed by the optical measurement (coherence-weighted, D12)."""
    ctx = req.ctx
    bw = ctx.table("imu", "bandwidth")
    tasks = [t for t in TASK_LABEL if t != "rest" and not bw.empty and (bw.task == t).any()]
    labels = [TASK_LABEL[t] for t in tasks]
    xax = Axis("Frequency", "\\hertz")

    def rows(p: Panel, cols: list[tuple[str, str, str, str]], g, dy: float = 0.22) -> None:
        # cols: (column, role, marker, legend); one marker per participant, offset per column
        off = np.linspace(-dy, dy, len(cols)) if len(cols) > 1 else [0.0]
        for (c, role, ms, lab), o in zip(cols, off):
            x, y = [], []
            for i, t in enumerate(tasks):
                v = g.loc[g.task == t, c].to_numpy(float) if c in g else np.array([])
                v = v[np.isfinite(v)]
                x.append(v)
                y.append(np.full(v.size, i + o))
            if any(v.size for v in x):
                p.series.append(Series("scatter", np.concatenate(x), np.concatenate(y),
                                       role=role, label=lab, style={"ms": ms, "size": 1.1}))

    p1 = Panel("Raw IMU", xax, Axis(""), [], legend="below")
    rows(p1, [("f50_hz", "cat:0", "o", "f50"), ("f95_hz", "cat:1", "s", "f95"),
              ("f99_hz", "cat:2", "^", "f99")], bw)
    p2 = Panel("Confirmed by optical", xax, Axis(""), [], legend="below")
    from biomech.imu.analysis import COHERENCE_MIN, MIN_COHERENT_BAND_HZ

    edge_txt = ""
    if not bw.empty and "coh_band_hz" in bw:
        ok = ~bw.noise_limited.astype(bool) if "noise_limited" in bw else \
            np.ones(len(bw), bool)
        e = bw.loc[ok & bw.task.isin(tasks), "coh_band_hz"].to_numpy(float)
        e = e[np.isfinite(e)]
        if e.size:
            edge_txt = (f" Where not noise-limited, the coherent band ends at {np.median(e):.1f} Hz "
                        f"(median; {e.min():.1f}–{e.max():.1f} Hz): above it the IMU power is "
                        "not confirmed by the optical motion")
    if not bw.empty and "f95_coh_hz" in bw:
        nl = bw.noise_limited.astype(bool) if "noise_limited" in bw else np.zeros(len(bw), bool)
        cols = [("f95_coh_hz", "cat:1", "s", "f95 of C′·P"),
                ("coh_band_hz", "cat:3", "d", "coherent band edge")]
        rows(p2, cols, bw[~nl])
        rows(p2, [(c, "flagged", "v", None) for c, *_ in cols], bw[nl])
        nl_s = [s for s in p2.series if s.role == "flagged"]
        if nl_s:
            nl_s[0].label = "noise-limited"
        n_nl = int((nl & bw.task.isin(tasks)).sum())
        p2.note = f"{n_nl} noise-limited participant-task" + ("s" if n_nl != 1 else "")
    for p in (p1, p2):
        p.y.ticks, p.y.ticklabels, p.y.reverse = [float(i) for i in range(len(tasks))], labels, True
        p.sharex = "f"
    n = bw.pid.nunique() if not bw.empty and "pid" in bw else 0
    spec = FigureSpec("movement_bandwidth", "Movement bandwidth", [p1, p2], rows=1, cols=2,
                      width="text", height_in=max(1.6, 0.32 * len(tasks) + 0.6),
                      sources=["imu:bandwidth"])
    return attach(spec, CaptionSpec(
        "jaw angular-velocity bandwidth per task and participant. (a) Frequencies below "
        "which 50, 95, and 99 % of the raw IMU power lie. (b) The part shared with the "
        "optical measurement: the 95 % frequency of the coherence-weighted power C′·P and the "
        "upper edge of the band where the bias-corrected IMU-optical coherence C′ is at least "
        f"{COHERENCE_MIN:g}",
        [("grey triangles", "noise-limited participant-tasks (coherent band below "
                            f"{MIN_COHERENT_BAND_HZ:g} Hz: the opening angle does not carry "
                            "the motion; not used for the bandwidth requirement)")],
        f"n = {n} participants", edge_txt.strip(), evidence="exploratory"))


def _ensemble_params(ctx) -> dict:
    """The X2 ensemble settings the timing stage ran with (config ``stages.timing.ensemble``)."""
    try:
        e = dict((ctx.config.params("timing") or {}).get("ensemble") or {})
    except (AttributeError, TypeError):
        e = {}
    e.setdefault("base_ms", [-600.0, -300.0])
    e.setdefault("envelope_s", 0.05)
    e.setdefault("rise_frac", 0.25)
    e.setdefault("release_frac", 0.5)
    return e


@register("timing_ensemble", "dashboard", "EMG around motion onset", params=("phase",),
          phase="open_close", domain="timing", phases=("open_close", "chewing"))
def timing_ensemble(req: ViewRequest) -> FigureSpec:
    """Cycle-aligned envelopes around the mechanical onset, detector-free (X2, exploratory)."""
    ctx = req.ctx
    task = req.phase or "open_close"
    ens = ctx.table("timing", "ensemble")
    par = _ensemble_params(ctx)
    b0, b1 = (float(v) for v in par["base_ms"])
    seed = B.seed_of(ctx)
    panels, ev_txt = [], []
    n_max = 0
    tl = TASK_LABEL.get(task, task)
    for direction, roles in (("opening", (("agonist", "openers (digastric)"),
                                          ("antagonist", "closers (masseter, temporalis)"))),
                             ("closing", (("agonist", "closers"), ("antagonist", "openers")))):
        for role, lab in roles:
            # panel letter and task in the title
            p = Panel(f"({'abcd'[len(panels)]}) {tl[:1].upper() + tl[1:]}, {direction}: {lab}",
                      Axis("Time from motion onset", "\\milli\\second"),
                      Axis("Envelope re pre-level", "\\decibel"), [], legend="none")
            curves, grid = [], None
            for pid in ctx.pids():
                a = ctx.arrays("timing", pid)
                y = a.get(f"ensemble__{task}__{direction}__{role}")
                if y is None or "ensemble__grid_ms" not in a:
                    continue
                grid = a["ensemble__grid_ms"]
                y = np.asarray(y, float)
                pre = (grid >= b0) & (grid <= b1)
                if not np.isfinite(y[pre]).any():
                    continue
                y = y - np.nanmedian(y[pre])
                curves.append(y)
            if curves:
                n_max = max(n_max, len(curves))
                B.add_curves_ci(p, grid, np.array(curves), centre="median", seed=seed)
                if not ens.empty:
                    e = ens[(ens.task == task) & (ens.direction == direction)
                            & (ens.role == role)].t_ms.to_numpy(float)
                    e = e[np.isfinite(e)]
                    if e.size:
                        # the event time as a line, its value in the caption (a panel note
                        # covered the traces)
                        p.series.append(Series("vline", x=np.array([float(np.median(e))]),
                                               role="group:openers" if lab.startswith("openers")
                                               else "group:closers"))
                        ev_txt.append(f"({'abcd'[len(panels)]}) "
                                      f"{B.snum(float(np.median(e)), 0)} ms ({e.size} "
                                      "participants)")
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
            panels.append(p)
    env_ms = 1e3 * float(par["envelope_s"])
    spec = FigureSpec("timing_ensemble", "EMG around motion onset", panels, rows=2, cols=2,
                      width="text", height_in=1.7, sources=["timing:ensemble"])
    return attach(spec, CaptionSpec(
        f"{TASK_LABEL.get(task, task)}: per-participant median envelope over cycles "
        f"({env_ms:g} ms RMS, dB re the level {-b0:g}–{-b1:g} ms before onset), aligned to "
        "the earliest mechanical onset",
        [("dashed vertical line", "the earliest mechanical onset"),
         ("solid vertical line", "median event time over participants (vermilion: openers, "
                                 "blue: closers)"),
         ("grey lines", "participants"),
         ("black line", "their median")] + _curve_marks("median"),
        f"n = up to {n_max} participants",
        "per participant the median over cycles; cohort median with its 95 % CI band"
        + (f"; median event times: {', '.join(ev_txt)}" if ev_txt else ""),
        evidence="exploratory",
        notes=[f"Agonist rise ({100 * float(par['rise_frac']):g} % of the way to the peak) and "
               f"antagonist release ({100 * float(par['release_frac']):g} % of the way to the "
               "minimum) times as in the detector-free ensemble timing"]))
