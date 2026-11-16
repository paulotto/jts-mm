# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frequency-domain views: EMG spectra and spectral descriptors, mains interference, the
low-frequency mask check, IMU movement spectra and the chewing rate.

All of them read stored stage outputs only (emg, stats_emg, qc, imu, kinematics,
stats_kinematics); nothing is recomputed. Spectral analyses are exploratory (X1, D12, D14).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from biomech.viz.spec import Axis, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.results import COHORT_UNIT

# Phases with stored EMG spectra (motion tasks, rest and the MVC takes).
EMG_PHASES = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion",
              "rest", "mvc_opening", "mvc_closing")
BANDS = (("share_20_50", "20–50"), ("share_50_100", "50–100"), ("share_100_200", "100–200"),
         ("share_200_450", "200–450"))
PSD_DB = "Power (dB re 1 µV²/Hz)"


def _pid(req: ViewRequest) -> str | None:
    pids = req.ctx.pids()
    return req.pid if req.pid in pids else (pids[0] if pids else None)


def _empty_fig(fid: str, title: str, msg: str, sources: list[str]) -> FigureSpec:
    p = Panel("", Axis("", lim=(0.0, 1.0)), Axis("", lim=(0.0, 1.0)),
              [Series("text", np.array([0.5]), np.array([0.45]), text=[msg],
                      style={"anchor": "above"})], legend="none")
    p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
    return FigureSpec(fid, title, [p], height_in=1.2, caption=f"{title}: {msg}.",
                      sources=sources)


def _grid_labels(panels: list[Panel], cols: int) -> None:
    """Small multiples: x label on the bottom row, y label on the first column only."""
    for k, p in enumerate(panels):
        if k + cols < len(panels):
            if p.x.label:
                p.hidden_labels["x"] = (p.x.label, p.x.unit)     # restored when reflowed
            p.x.label, p.x.unit = "", ""
        if k % cols:
            if p.y.label:
                p.hidden_labels["y"] = (p.y.label, p.y.unit)
            p.y.label, p.y.unit = "", ""


def _median(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def _order(muscles) -> list[str]:
    s = set(muscles)
    return [m for m in V.MUSCLES if m in s]


def _band(ctx) -> str:
    """The EMG band-pass of the analysis ('20–450 Hz'), from ``signals.emg.bandpass_hz``."""
    try:
        lo, hi = (float(v) for v in ctx.config.signals["emg"]["bandpass_hz"])
        return f"{lo:g}–{hi:g} Hz"
    except (AttributeError, KeyError, TypeError, ValueError):
        return "20–450 Hz"


# -- EMG spectra ------------------------------------------------------------------------------
@register("emg_psd", "dashboard", "EMG power spectra", params=("pid", "phase"), phase="chewing",
          domain="emg_freq", phases=EMG_PHASES)
def emg_psd(req: ViewRequest) -> FigureSpec:
    """Welch PSD of the cleaned EMG per muscle (stored), with the mains harmonics."""
    ctx, task = req.ctx, req.phase or "chewing"
    fid, title = "emg_psd", "EMG power spectra"
    src = ["emg:spectral", "qc:emg"]
    pid = _pid(req)
    a = ctx.arrays("emg", pid) if pid else {}
    f, P = a.get("spectral__freq"), a.get(f"spectral__{task}__psd")
    who = f"P{ctx.pseudonym(pid)}" if pid else ""
    if f is None or P is None:
        return _empty_fig(fid, title, f"no stored EMG spectrum for {who}, "
                                      f"{V.task_label(task)}", src)
    sp = ctx.table("emg", "spectral", [pid])
    sp = sp[sp.task == task].set_index("muscle")
    names = list(sp.index)                        # channel order of the stored PSD rows
    if len(names) != len(P):
        refs = ctx.table("emg", "refs", [pid])
        names = list(refs.muscle)
    qc = ctx.table("qc", "emg", [pid])
    f0 = _median(qc.line_hz) if "line_hz" in qc else np.nan
    f = np.asarray(f, float)
    harm = f0 * np.arange(1, int(f.max() // f0) + 1) if np.isfinite(f0) else np.array([])
    panels = []
    for m in V.MUSCLES:
        lab = V.muscle_label(m)
        if m not in names:
            panels.append(Panel(lab, Axis("Frequency", V.HZ), Axis(PSD_DB), [],
                                legend="none", note="no channel"))
            continue
        p = B.spectrum(f, {lab: P[names.index(m)]}, lab, yaxis=Axis(PSD_DB),
                       roles={lab: f"entity:{m}"}, xlim=(0.0, float(f.max())))
        p.legend, p.sharex = "none", "f"
        if harm.size:
            p.series.insert(0, Series("vline", x=harm, role="faint"))
        if m in sp.index:
            r = sp.loc[m]
            p.note = f"MNF {r.mnf_hz:.0f} Hz · MDF {r.mdf_hz:.0f} Hz"
        panels.append(p)
    _grid_labels(panels, 3)
    mains = f"{f0:.2f} Hz" if np.isfinite(f0) else "n/a"
    return FigureSpec(fid, title, panels, rows=2, cols=3, width="text", height_in=1.35,
                      caption=f"{who}, {V.task_label(task)}: Welch power spectral density of "
                              f"the cleaned EMG (mains removed, {_band(ctx)} band-pass; 0.5-s "
                              "segments, masked samples left out), 10 log10 of the power. Grey "
                              f"verticals: mains frequency of this recording ({mains}) and its "
                              "harmonics. MNF/MDF: mean and median frequency (exploratory).",
                      sources=src, perspective=f"{who}, {V.task_label(task)}",
                      meta={"pid": pid, "line_hz": f0})


@register("emg_spectral_descriptors", "dashboard", "EMG spectral descriptors",
          params=("phase",), phase="chewing", domain="emg_freq", phases=EMG_PHASES)
def emg_spectral_descriptors(req: ViewRequest) -> FigureSpec:
    """MNF and MDF per muscle (participants, median) and the band shares (cohort median)."""
    task = req.phase or "chewing"
    fid, title = "emg_spectral_descriptors", "EMG spectral descriptors"
    sp = req.ctx.table("emg", "spectral")
    sp = sp[sp.task == task] if not sp.empty else sp
    if sp.empty:
        return _empty_fig(fid, title, f"no spectra for {V.task_label(task)}", ["emg:spectral"])
    use = sp[sp.usable_shape.astype(bool)] if "usable_shape" in sp else sp
    order = _order(use.muscle)
    kw = dict(by="muscle", order=order, seed=B.seed_of(req.ctx))
    p1 = B.strip_by_ci(use, "mnf_hz", title="Mean frequency", xaxis=Axis("MNF", V.HZ), **kw)
    p2 = B.strip_by_ci(use, "mdf_hz", title="Median frequency", xaxis=Axis("MDF", V.HZ), **kw)
    long = use.melt(id_vars=["muscle"], value_vars=[c for c, _ in BANDS if c in use],
                    var_name="band", value_name="share")
    p3 = B.matrix_heat(long, "muscle", "band", "share", "Band share of power (%)",
                       rows=order, cols=[c for c, _ in BANDS],
                       col_labels=[lab + " Hz" for _, lab in BANDS], scale=100.0)
    n_flag = int(len(sp) - len(use))
    nwin = _median(use.n_windows) if "n_windows" in use else np.nan
    return FigureSpec(fid, title, [p1, p2, p3], rows=1, cols=3, width="text",
                      height_in=B.rows_height(len(order), per=0.2, extra=0.6, lo=1.6),
                      caption=f"{V.task_label(task)}: (a) mean (MNF) and (b) median frequency (MDF) of "
                              "the cleaned-EMG Welch spectrum per participant (grey) and their "
                              f"median (diamond; {B.MEDIAN_CI_CAPTION}); (c) share of the "
                              f"{_band(req.ctx)} power per band, "
                              "median over participants. Channels flagged unusable for spectral "
                              f"shape (SNR) are not shown ({n_flag} channel(s)); median "
                              f"{nwin:.0f} 0.5-s windows per spectrum (exploratory).",
                      sources=["emg:spectral"])


@register("emg_spectral_spm", "dashboard", "EMG spectral SPM", domain="emg_freq")
def emg_spectral_spm(req: ViewRequest) -> FigureSpec:
    """SPM along frequency: masseter L vs R in chewing, and where participants differ."""
    fid, title = "emg_spectral_spm", "EMG spectral SPM"
    src = ["stats_emg:spectral_spm"]
    spm = req.ctx.table("stats_emg", "spectral_spm")
    arr = req.ctx.arrays("stats_emg", COHORT_UNIT)
    grid, z = arr.get("spectral__grid_hz"), arr.get("spectral_spm1__z")
    rows = {} if spm.empty else {r.test: r for r in spm.itertuples()}
    t1 = rows.get("masseter_L_vs_R_chewing_spectrum")
    an = rows.get("anova1_masseter_r_chewing_spectrum")

    def clusters(r) -> list[tuple[float, float]]:
        return [(a, b) for a, b, _ in B.parse_clusters(getattr(r, "clusters_hz", None))]

    if grid is None or (t1 is None and an is None):
        return _empty_fig(fid, title, "no spectral SPM stored", src)
    grid = np.asarray(grid, float)
    fax = Axis("Frequency", V.HZ, lim=(float(grid.min()), float(grid.max())))

    def pct(v) -> str:
        v = float(v)
        return f", {100 * v:.0f} % supra-threshold" if np.isfinite(v) else ""

    if t1 is not None and z is not None and len(z) == len(grid):
        p1 = B.spm_panel(grid, z, float(t1.zstar), "Masseter L vs R, chewing",
                         clusters=clusters(t1), xaxis=fax)
        p1.legend = "none"
        p1.note = f"n = {t1.n}{pct(t1.supra_frac)}"
    else:
        p1 = Panel("Masseter L vs R, chewing", fax, Axis("SPM{t}"), [], legend="none",
                   note="SPM{t} not computed: n = "
                        f"{t1.n if t1 is not None else 0} paired participants (< 5)")
    p1.sharex = "f"
    key = "spectral__chewing__masseter_r__subjects"
    p2 = Panel("Participants differ (ANOVA), masseter R", fax, Axis("log10 power share"), [],
               legend="none", sharex="f")
    if key in arr:
        B.add_curves_ci(p2, grid, arr[key], label="cohort mean", seed=B.seed_of(req.ctx))
    meta: dict = {"t_clusters_hz": clusters(t1) if t1 is not None else []}
    if an is not None:
        cl = meta["anova_clusters_hz"] = clusters(an)
        B.add_spm_bars(p2, [("participants differ", cl, "secondary")])
        zs = meta["anova_zstar"] = float(an.zstar)
        anova = (f" ANOVA: {an.n} participants" + (f", F* = {zs:.2f}" if np.isfinite(zs) else "")
                 + pct(an.supra_frac) + ".")
    else:
        anova = ""
    return FigureSpec(fid, title, [p1, p2], rows=1, cols=2, width="text", height_in=1.9,
                      caption="SPM along the frequency axis on the per-window spectral shape "
                              "(log10 share of the in-band power per 10-Hz bin), chewing. (a) "
                              "paired SPM{t} masseter left vs right on participant means with "
                              "its critical threshold (dotted) and supra-threshold clusters "
                              "(shaded; needs at least 5 participants). (b) Participant mean "
                              "shapes of the right masseter (grey), cohort mean ("
                              + B.curve_ci_caption() + "), and the "
                              "frequency ranges where participants differ (bar: one-way SPM "
                              "ANOVA, windows as repetitions)." + anova + " Exploratory.",
                      sources=src + ["stats_emg:spectral_spm1__z",
                                     "stats_emg:spectral__chewing__masseter_r__subjects"],
                      meta=meta)


def _cohort_frame(sc: pd.DataFrame) -> pd.DataFrame:
    tasks = [t for t in V.TASK_LABEL if t in set(sc.task)]
    sc = sc.assign(_t=sc.task.map({t: i for i, t in enumerate(tasks)}),
                   _m=sc.muscle.map({m: i for i, m in enumerate(V.MUSCLES)}))
    return sc.sort_values(["_t", "_m"]).drop(columns=["_t", "_m"])


@register("emg_spectral_cohort", "dashboard", "EMG spectral cohort summary", domain="emg_freq")
def emg_spectral_cohort(req: ViewRequest) -> FigureSpec:
    """Cohort MNF and the between/within-participant SD ratio, task x muscle."""
    fid, title = "emg_spectral_cohort", "EMG spectral cohort summary"
    sc = req.ctx.table("stats_emg", "spectral_cohort")
    if sc.empty:
        return _empty_fig(fid, title, "no spectral cohort table", ["stats_emg:spectral_cohort"])
    tasks = [t for t in V.TASK_LABEL if t in set(sc.task)]
    kw = dict(rows=tasks)
    p1 = B.matrix_heat(sc, "task", "muscle", "mnf_mean_hz", "Mean MNF (Hz)", **kw)
    p2 = B.matrix_heat(sc, "task", "muscle", "mdf_mean_hz", "Mean MDF (Hz)", **kw)
    p3 = B.matrix_heat(sc, "task", "muscle", "between_over_within", "MNF SD between / within",
                       **kw, fmt="{:.1f}")
    return FigureSpec(fid, title, [p1, p2, p3], rows=1, cols=3, width="text",
                      height_in=0.22 * len(tasks) + 1.0,
                      caption="Spectral descriptors across the cohort per task and muscle "
                              "(shape-usable channels): mean MNF and MDF over participants, and "
                              "the between-participant SD of the MNF over the pooled "
                              "within-participant SD across 0.5-s windows (> 1: participants "
                              "differ more than one participant's windows; exploratory). "
                              "All columns: the complete spectral cohort table.",
                      sources=["stats_emg:spectral_cohort"])


@register("emg_spectral_cohort_table", "dashboard", "EMG spectral cohort table",
          domain="emg_freq")
def emg_spectral_cohort_table(req: ViewRequest) -> TableSpec:
    """All columns of stats_emg.spectral_cohort, task x muscle."""
    sc = req.ctx.table("stats_emg", "spectral_cohort")
    if not sc.empty:
        sc = _cohort_frame(sc)
        sc = sc.assign(task=sc.task.map(V.task_label), muscle=sc.muscle.map(V.muscle_label))
    t = B.frame_table(sc, "emg_spectral_cohort_table",
                      "EMG mean (MNF) and median frequency (MDF) across participants per task "
                      "and muscle, the between-participant SD of the MNF, the pooled "
                      "within-participant SD across 0.5-s windows and their ratio "
                      "(exploratory).", ctx=req.ctx)
    t.sources = ["stats_emg:spectral_cohort"]
    return t


# -- the low-frequency mask check (D14) and mains interference ------------------------------
@register("emg_mask_validation", "dashboard", "Low-frequency flag check", domain="emg_freq")
def emg_mask_validation(req: ViewRequest) -> FigureSpec:
    """Windows hit by the low-frequency rule vs clean windows: 20-40 Hz share and MNF."""
    fid, title = "emg_mask_validation", "Low-frequency flag check"
    src = ["emg:mask_validation", "stats_emg:mask_validation"]
    mv = req.ctx.table("emg", "mask_validation")
    cm = req.ctx.table("stats_emg", "mask_validation")
    if mv.empty:
        return _empty_fig(fid, title, "no channel-task with low-frequency hits", src)
    order = _order(mv.muscle)
    lab = [V.muscle_label(m) for m in order]
    g = mv.groupby("muscle")
    vals = lambda c, k=1.0: [k * g.get_group(m)[c].to_numpy(float) for m in order]  # noqa: E731
    seed = B.seed_of(req.ctx)
    p1 = B.paired_strip(lab, vals("share_20_40_masked", 100.0), vals("share_20_40_clean", 100.0),
                        "20–40 Hz share", Axis(f"Share of {_band(req.ctx)} power", V.PCT),
                        label_a="flagged windows", label_b="clean windows", seed=seed)
    p2 = B.paired_strip(lab, vals("mnf_masked_hz"), vals("mnf_clean_hz"), "Mean frequency",
                        Axis("MNF", V.HZ), label_a="flagged windows", label_b="clean windows",
                        seed=seed)
    p1.legend, p2.legend = "none", "below"
    cons = cm.set_index("muscle") if not cm.empty else pd.DataFrame()
    cl = (["all"] if "all" in cons.index else []) + [m for m in order if m in cons.index]
    p3 = B.forest(["all channels" if m == "all" else V.muscle_label(m) for m in cl],
                  [100 * float(cons.loc[m, "conservative_frac"]) for m in cl],
                  [np.nan] * len(cl), [np.nan] * len(cl), "Flag looks like clean EMG",
                  Axis("Channel-tasks", V.PCT, lim=(0.0, 100.0)), ref=None)
    if cl:
        p3.note = f"n = {int(cons.loc[cl[0], 'n'])} channel-tasks"
    from biomech.emg.analysis import MASK_WIN_S

    return FigureSpec(fid, title, [p1, p2, p3], rows=1, cols=3, width="text",
                      height_in=B.rows_height(len(cl) or len(order), per=0.3, extra=0.6, lo=1.6),
                      caption="Check of the low-frequency rule (a QC flag since a post hoc change, no longer an "
                              f"exclusion). Per channel and task, {MASK_WIN_S:g}-s windows of the "
                              "cleaned EMG with > 50 % flagged samples vs windows without any "
                              f"masked sample ({len(mv)} channel-tasks with hits): the median "
                              "20–40 Hz share and mean frequency of each channel-task (small "
                              f"dots) and their median (diamond; {B.MEDIAN_CI_CAPTION}). "
                              "(c) Share of channel-tasks where both medians "
                              "of the flagged windows lie inside the clean windows' IQR "
                              "(conservative: the flag hits EMG that looks clean; exploratory).",
                      sources=src)


@register("emg_mains", "dashboard", "Mains interference", domain="emg_freq")
def emg_mains(req: ViewRequest) -> FigureSpec:
    """Measured mains frequency, mains share before/after cleaning, rest RMS."""
    ctx = req.ctx
    fid, title = "emg_mains", "Mains interference"
    q = ctx.table("qc", "emg")
    if q.empty:
        return _empty_fig(fid, title, "no EMG QC table", ["qc:emg"])
    pids = [p for p in ctx.pids() if p in set(q.pid)]
    lhz = q.groupby("pid").line_hz.median().reindex(pids).to_numpy(float)
    nominal = np.where(np.abs(lhz - 60.0) < np.abs(lhz - 50.0), 60.0, 50.0)   # grid: 50/60 Hz
    p1 = B.forest([f"{r} ({n:.0f} Hz)" for r, n in zip(B.pseudo_rows(ctx, pids), nominal)],
                  1e3 * (lhz - nominal), [np.nan] * len(pids), [np.nan] * len(pids),
                  "Mains frequency", Axis("Measured − nominal (mHz)"), ref=0.0)
    order = _order(q.muscle)
    lab = [V.muscle_label(m) for m in order]
    g = q.groupby("muscle")
    seed = B.seed_of(ctx)
    p2 = B.paired_strip(lab, [100 * g.get_group(m).mains_share_raw.to_numpy(float)
                              for m in order],
                        [100 * g.get_group(m).mains_share_clean.to_numpy(float) for m in order],
                        "Mains share at rest", Axis("Power at line harmonics", V.PCT),
                        label_a="raw", label_b="cleaned", seed=seed)
    try:
        thr = float(ctx.config.params("qc")["usability"]["rest_noise_uv"])
    except (KeyError, TypeError, ValueError):
        thr = None
    p3 = B.strip_by_ci(q, "rest_rms_uv", "muscle", "Rest RMS (cleaned)", order=order,
                       xaxis=Axis("RMS", V.UV), ref=thr, seed=seed)
    return FigureSpec(fid, title, [p1, p2, p3], rows=1, cols=3, width="text",
                      height_in=max(1.6, 0.16 * len(pids) + 0.9),
                      caption="Mains interference: grid frequency measured per recording, "
                              "as the deviation from the nominal grid frequency (in brackets) "
                              f"in mHz; share of the {_band(ctx)} rest "
                              "power at the line harmonics before and after mains removal "
                              "(small dots: participants); RMS of the cleaned rest EMG per "
                              "participant (grey)"
                              + (f", dashed: the limit ({thr:g} µV) above which a channel is "
                                 "reported as noise-limited at rest" if thr is not None else "")
                              + f". Diamond: median, {B.MEDIAN_CI_CAPTION}.",
                      sources=["qc:emg"])


# -- IMU movement spectra (D12) -----------------------------------------------------------------
@register("imu_spectra", "dashboard", "IMU movement spectrum", params=("pid", "phase"),
          phase="chewing", domain="imu", phases=("rest",) + V.TASKS)
def imu_spectra(req: ViewRequest) -> FigureSpec:
    """Jaw angular-velocity PSD with its bandwidth quantiles, and IMU-optical coherence."""
    from biomech.imu.analysis import COHERENCE_MIN

    ctx, task = req.ctx, req.phase or "chewing"
    fid, title = "imu_spectra", "IMU movement spectrum"
    src = ["imu:bandwidth"]
    pid = _pid(req)
    who = f"P{ctx.pseudonym(pid)}" if pid else ""
    a = ctx.arrays("imu", pid) if pid else {}
    k = f"bandwidth__{task}__"
    f, P = a.get(k + "freq"), a.get(k + "psd")
    if f is None or P is None:
        return _empty_fig(fid, title, f"no IMU spectrum for {who}, {V.task_label(task)}", src)
    bw = ctx.table("imu", "bandwidth", [pid])
    bw = bw[bw.task == task]
    r = bw.iloc[0] if not bw.empty else pd.Series(dtype=float)
    val = lambda c: float(r.get(c, np.nan))                              # noqa: E731
    f = np.asarray(f, float)
    pos = f > 0
    xlim = (float(f[pos].min()), float(f.max()))
    unit = "(°/s)²/Hz"
    yax = Axis(f"Power (dB re 1 {unit})")
    p1 = B.spectrum(f[pos], {"IMU": np.asarray(P, float)[pos]}, "Angular-velocity PSD",
                    yaxis=yax, roles={"IMU": "primary"})
    p1.x.log, p1.x.lim, p1.sharex = True, xlim, "f"
    fc, C, P40 = a.get(k + "coh_freq"), a.get(k + "coh"), a.get(k + "psd_imu40")
    for fx, rl in ((val("f50_hz"), "reference"), (val("f95_hz"), "reference"),
                   (val("f99_hz"), "reference"), (val("f95_coh_hz"), "threshold")):
        if np.isfinite(fx):
            p1.series.append(Series("vline", x=np.array([fx]), role=rl))
    raw = " · ".join(f"{c[:3]} {val(c):.1f}" for c in ("f50_hz", "f95_hz", "f99_hz")
                     if np.isfinite(val(c)))
    p1.note = "; ".join(t for t in (f"{raw} Hz (dashed)" if raw else "",
                                    f"f95,coh {val('f95_coh_hz'):.1f} Hz (dotted)"
                                    if np.isfinite(val("f95_coh_hz")) else "") if t)
    p2 = Panel("IMU-optical coherence", Axis("Frequency", V.HZ, log=True, lim=xlim),
               Axis("C′ (bias-corrected)", lim=(0.0, 1.0)), [], legend="none", sharex="f")
    if fc is not None and C is not None:
        fc, C = np.asarray(fc, float), np.asarray(C, float)
        ok = fc > 0
        edge = val("coh_band_hz")
        if np.isfinite(edge) and edge > 0:
            p2.series.append(Series("hspan", x=np.array([xlim[0]]), y2=np.array([edge]),
                                    role="faint"))
        p2.series += [Series("line", fc[ok], C[ok], role="primary"),
                      Series("hline", y=np.array([COHERENCE_MIN]), role="threshold",
                             label=f"C′ = {COHERENCE_MIN:g}")]
        if P40 is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                y = 10 * np.log10(np.where(C * P40 > 0, C * P40, np.nan))
            p1.series.insert(1, Series("line", fc[ok], y[ok], role="secondary",
                                       label="confirmed by optical (C′·P)"))
            p1.series[0].label = "IMU (raw)"
        nl = bool(r.get("noise_limited", True))
        p2.note = ((f"coherent band ≤ {edge:.2f} Hz" if edge > 0 else "no coherent band")
                   + ("; noise-limited" if nl else ""))
    else:
        p2.note = "no optical reference for this task"
    p1.legend = "auto"
    return FigureSpec(fid, title, [p1, p2], rows=1, cols=2, width="text", height_in=1.9,
                      caption=f"{who}, {V.task_label(task)}: (a) Welch PSD (4-s windows) of the IMU "
                              "jaw opening angular velocity up to 50 Hz with the frequencies "
                              "below which 50/95/99 % of its power lie (dashed) and the 95 % "
                              "frequency of the optically confirmed power (dotted). (b) "
                              "bias-corrected magnitude-squared coherence C′ of the IMU and the "
                              "unsmoothed optical opening velocity on a common 40 Hz grid, the "
                              f"{COHERENCE_MIN:g} threshold and the coherent band from the lowest "
                              "frequencies up (shaded); C′·P is the IMU power the optical "
                              "measurement confirms (exploratory).",
                      sources=src, perspective=f"{who}, {V.task_label(task)}",
                      meta={"pid": pid})


# -- chewing rate ---------------------------------------------------------------------------------
@register("chewing_rate", "dashboard", "Chewing rate", domain="kinematics")
def chewing_rate(req: ViewRequest) -> FigureSpec:
    """Chewing frequency per participant and the per-cycle rate distribution."""
    ctx = req.ctx
    fid, title = "chewing_rate", "Chewing rate"
    src = ["stats_kinematics:subjects", "kinematics:cycles"]
    cyc = ctx.table("kinematics", "cycles")
    subj = ctx.table("stats_kinematics", "subjects")
    if cyc.empty or "duration_s" not in cyc:
        return _empty_fig(fid, title, "no movement cycles", src)
    ch = cyc[(cyc.task == "chewing") & cyc.ok.astype(bool) & (cyc.duration_s > 0)]
    pids = [p for p in ctx.pids() if p in set(ch.pid)]
    rates = [1.0 / ch[ch.pid == p].duration_s.to_numpy(float) for p in pids]
    xax = Axis("Chewing rate", V.HZ)
    p1 = B.strip(B.pseudo_rows(ctx, pids), rates, "Per participant", xax)
    fq = (subj.set_index("pid").chewing_freq_hz.reindex(pids).to_numpy(float)
          if "chewing_freq_hz" in subj else np.full(len(pids), np.nan))
    med = next((s for s in p1.series if s.role == "accent"), None)
    if med is not None:
        med.label = "median of cycle rates"
    p1.series.append(Series("scatter", fq, np.arange(len(pids), dtype=float), role="cat:1",
                            label="1 / median cycle duration", style={"ms": "s", "size": 1.2}))
    p1.legend = "below"
    allr = np.concatenate(rates) if rates else np.array([])
    p2 = B.histogram(allr, "All chewing cycles", xax, observed=_median(fq),
                     observed_label="cohort median")
    p2.note = f"{allr.size} cycles, {len(pids)} participants"
    return FigureSpec(fid, title, [p1, p2], rows=1, cols=2, width="text",
                      height_in=max(1.6, 0.16 * len(pids) + 0.9),
                      caption="Chewing rate: (a) per accepted chewing cycle 1 / cycle duration "
                              "(grey dots, diamond: median per participant) and the participant "
                              "chewing frequency 1 / median cycle duration (square, "
                              "the stored cohort value); (b) distribution over all accepted "
                              "cycles with the cohort median of the participant frequencies.",
                      sources=src)
