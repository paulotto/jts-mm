# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Paper figures and tables (ICORR 2027), registered as ``paper`` views.

Each figure argues one claim and picks its form for that job:

* Fig. 1 (C1, not a hinge): incisal paths as participant medians (grey) under
  the cohort path (accent); helical-axis migration early -> late per
  participant (arrows); per-model misalignment as a strip against the
  noise-only band; condylar translation coupled to rotation (grey individuals,
  IQR wash, median accent).
* Fig. 2 (C2, muscle activity and motion onset): opening angle and opener/closer
  activation over the cycle with SPM clusters; per-participant strips around zero:
  single-cycle lead under both onset rules (D6), ensemble opener rise and closer
  release (X2), causal margin.
* Fig. 3 (C3, wearable sensing): Bland-Altman with labelled limits;
  unseen- vs same-subject decoding as dumbbells; row-normalized confusion.
* Fig. 4 (S2, variability): processing spread relative to the
  between-subject SD (lollipop); between/within variance along the cycle.

Participants appear only as pseudonymous indices (``DataContext.pseudonym``),
never with age or sex. ``perspective``/``stat`` switch the trial-based panels
(Fig. 1a/1d, Fig. 2a/2b) between participant medians, one participant's
cycles and the cohort summary; the paper yaml fixes the published choice.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from biomech import ensembles as E
from biomech.kinematics import se3
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views.registry import ViewRequest, register
from biomech.results import COHORT_UNIT

MM, DEG, DPS, MS = "\\milli\\metre", "\\degree", "\\degree\\per\\second", "\\milli\\second"
PCT = "\\percent"
CLASSES = ("REST", "OPEN", "CLOSE", "PROTR.", "RETR.", "LEFT", "RIGHT")
TRIAL_VIEWS = ("cohort", "participant", "trials", "trials+summary")


def _empty(p: Panel, msg: str = "no data") -> Panel:
    p.note = msg
    return p


def _summary_series(ens: E.Ensemble, req: ViewRequest, ch: int = 0, *, role: str = "accent",
                    label: str | None = None) -> list[Series]:
    """Series for one channel of an ensemble at the requested perspective.

    cohort: participant centres (context) + cohort centre (accent) + spread band;
    participant / trials: the selected participant's cycles (+ its centre and band).
    """
    xx = ens.grid
    out: list[Series] = []
    if not ens.n_trials:
        return out
    if req.perspective in ("participant", "trials", "trials+summary") and req.pid:
        e = ens.select((ens.trials.pid == req.pid).to_numpy())
        if not e.n_trials:
            return out
        if req.perspective != "participant":
            for k in range(e.n_trials):
                out.append(Series("line", xx, e.data[k, :, ch], role="context"))
        if req.perspective != "trials":
            s = E.aggregate(e, "participant", req.stat)
            out.append(Series("band", xx, s.lo[0, :, ch], s.hi[0, :, ch], role="band"))
            out.append(Series("line", xx, s.center[0, :, ch], role=role, label=label))
        return out
    s = E.aggregate(ens, "cohort", req.stat)
    for row in s.members:
        out.append(Series("line", xx, row[:, ch], role="context"))
    c = s.center[0, :, ch]
    if req.stat == "mean_ci":                     # the band is the CI already
        out.append(Series("band", xx, s.lo[0, :, ch], s.hi[0, :, ch], role="band"))
    else:                                         # spread light, CI of the centre dark (R5 E)
        out.append(Series("band", xx, s.lo[0, :, ch], s.hi[0, :, ch], role="band",
                          style={"alpha": B.SD_BAND_ALPHA}))
        tmp = Panel()
        if B.add_ci_band(tmp, xx, s.members[:, :, ch], seed=B.seed_of(req.ctx),
                         centre="median" if req.stat == "median_iqr" else "mean",
                         where=np.isfinite(c)):
            out += tmp.series
    out.append(Series("line", xx, c, role=role, label=label))
    return out


def _persp_caption(ens: E.Ensemble, req: ViewRequest) -> str:
    if not ens.n_trials:
        return "no data"
    if req.perspective == "cohort" or not req.pid:
        txt = E.describe_perspective("cohort", ens.align, req.stat, len(ens.pids()),
                                     int(ens.trials.ok.sum()))
        if req.stat in ("mean_sd", "median_iqr"):
            txt += "; " + B.curve_ci_caption("median" if req.stat == "median_iqr" else "mean")
        return txt
    n_t = int(((ens.trials.pid == req.pid) & ens.trials.ok).sum())
    return E.describe_perspective(req.perspective, ens.align, req.stat, 1, n_t)


# -- Figure 1: C1 kinematics -------------------------------------------------------------------
@register("fig_kinematics", "paper", "Jaw opening is not a hinge",
          params=("perspective", "stat", "pid"), perspectives=TRIAL_VIEWS,
          perspective="cohort", stat="median_iqr")
def fig_kinematics(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    # (a) sagittal incisal paths, cycle-normalized, relative to the cycle start
    ens = E.kinematic(ctx, "open_close", "incisor_disp", "path")
    pa = Panel("(a) Incisal opening path", Axis("Anterior", MM), Axis("Superior", MM),
               legend="none", equal=True)
    if ens.n_trials:
        if req.perspective in ("participant", "trials", "trials+summary") and req.pid:
            e = ens.select((ens.trials.pid == req.pid).to_numpy())
            for k in range(e.n_trials):
                pa.series.append(Series("line", e.data[k, :, 0], e.data[k, :, 2], role="context"))
            if e.n_trials:
                s = E.aggregate(e, "participant", req.stat)
                pa.series.append(Series("line", s.center[0, :, 0], s.center[0, :, 2],
                                        role="accent"))
            pa.note = "grey: cycles"
        else:
            s = E.aggregate(ens, "cohort", req.stat)
            for row in s.members:
                pa.series.append(Series("line", row[:, 0], row[:, 2], role="context"))
            pa.series.append(Series("line", s.center[0, :, 0], s.center[0, :, 2], role="accent"))
    else:
        _empty(pa)
    # (b) helical-axis migration: early (<= 10 deg) -> late piercing point per participant
    fha = ctx.table("kinematics", "fha")
    lm = ctx.table("kinematics", "landmarks")
    pb = Panel("(b) Helical-axis migration", Axis("Anterior of condyles", MM),
               Axis("Superior", MM), legend="auto", equal=True)
    tails, vecs = [], []
    for pid in ctx.pids():
        g = fha[fha.pid == pid] if not fha.empty else fha
        a = ctx.arrays("kinematics", pid)
        L = lm[(lm.pid == pid) & lm.name.isin(["tmj_left", "tmj_right"])][["x", "y", "z"]] \
            if not lm.empty else lm
        if g.empty or "T_closed" not in a or L.empty:
            continue
        mid = se3.apply(a["T_closed"], L.to_numpy(float).mean(axis=0))
        early = g[g.theta_to_deg <= 10]
        late = g[g.theta_to_deg > 10]
        if early.empty or late.empty:
            continue
        e0 = np.array([early.pierce_x_mm.median() - mid[0], early.pierce_z_mm.median() - mid[2]])
        e1 = np.array([late.pierce_x_mm.median() - mid[0], late.pierce_z_mm.median() - mid[2]])
        tails.append(e0)
        vecs.append(e1 - e0)
    if tails:
        T, V = np.array(tails), np.array(vecs)
        pb.series += [Series("scatter", T[:, 0], T[:, 1], role="context",
                             label="early opening (≤ 10°)"),
                      Series("arrow", T[:, 0], T[:, 1], d=V, role="accent",
                             label="shift with further opening"),
                      Series("scatter", np.array([0.0]), np.array([0.0]), role="reference",
                             style={"ms": "d", "size": 1.4}),
                      Series("text", np.array([0.0]), np.array([0.0]), text=["condyles"],
                             style={"anchor": "right"})]
    else:
        _empty(pb)
    # (c) misalignment per joint model vs the noise-only band
    h = ctx.table("kinematics", "hinge")
    nz = ctx.table("uncertainty", "hinge_noise")
    models = [("hinge_tmj", "hinge through palpated condyles"), ("hinge_fit", "best-fit hinge"),
              ("rot_trans", "hinge + translation")]
    pc = Panel("(c) Misalignment at the incisor", Axis("RMS residual", MM, log=True),
               Axis(""), legend="none")
    if not nz.empty and "null_invivo_p99_mm" in nz:
        lo, hi = np.nanmin(nz.null_invivo_p99_mm), np.nanmax(nz.null_invivo_p99_mm)
        pc.series.append(Series("hspan", x=np.array([max(lo * 0.98, 1e-3)]), y2=np.array([hi]),
                                role="muted"))
        pc.series.append(Series("text", np.array([hi]), np.array([len(models) - 0.55]),
                                text=["noise only"], style={"anchor": "right"}))
    for i, (m, _lab) in enumerate(models):
        v = h[h.model == m].rms_incisor_mm.to_numpy(float) if not h.empty else np.empty(0)
        v = v[np.isfinite(v)]
        if v.size:
            jit = np.linspace(-0.16, 0.16, v.size) if v.size > 1 else np.zeros(1)
            pc.series.append(Series("scatter", v, i + jit, role="context"))
            lo, hi = B.median_ci(v, B.seed_of(ctx))
            if np.isfinite(lo) and np.isfinite(hi):
                pc.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                        role="accent", style={"width": 1.4}))
            pc.series.append(Series("scatter", np.array([np.median(v)]), np.array([float(i)]),
                                    role="accent", style={"ms": "d", "size": 1.5}))
    pc.y.ticks = list(range(len(models)))
    pc.y.ticklabels = [lab for _, lab in models]
    pc.y.reverse = True
    # (d) condylar translation vs opening angle (midpoint of the palpated condyles)
    ens_l = E.kinematic(ctx, "open_close", "tmj_left", "angle")
    ens_r = E.kinematic(ctx, "open_close", "tmj_right", "angle")
    pd_ = Panel("(d) Condylar translation with opening", Axis("Opening angle", DEG),
                Axis("Anterior condylar transl.", MM), legend="none")
    if ens_l.n_trials and ens_r.n_trials:
        x = 0.5 * (ens_l.data[:, :, :1] + ens_r.data[:, :, :1])
        mid = E.Ensemble("open_close/tmj_mid", ens_l.grid, x - x[:, :1], ens_l.trials, ["x"],
                         "angle", MM, DEG)
        pd_.series += _summary_series(mid, req)
        ys = pd_.series[-1].y if pd_.series else None
        last = np.flatnonzero(np.isfinite(ys)) if ys is not None else []
        if len(last):
            k = last[-1]
            pd_.series.append(Series("text", np.array([ens_l.grid[k]]), np.array([ys[k]]),
                                     text=["median" if req.stat == "median_iqr" else "mean"],
                                     style={"anchor": "above", "color": "ink"}))
    else:
        _empty(pd_)
    return FigureSpec("fig_kinematics", "Jaw opening is not a hinge", [pa, pb, pc, pd_], rows=2,
                      cols=2, width="text", height_in=1.75,
                      caption="(a) Sagittal incisal path of opening-closing; (b) shift of the "
                              "finite helical axis from early to further opening, relative to "
                              "the palpated condyles (one arrow per participant); (c) RMS "
                              "misalignment at the incisal point of three exoskeleton joint "
                              "models (dots: participants, diamond: median, line: its 95 % BCa "
                              "bootstrap CI; grey band: "
                              "noise-only 99th percentile); (d) anterior condylar translation "
                              f"coupled to rotation. (a, d): {_persp_caption(ens, req)}.",
                      perspective=req.perspective,
                      sources=["kinematics:cycles", "kinematics:fha", "kinematics:hinge",
                               "uncertainty:hinge_noise"])


# -- Figure 2: C2 EMG timing --------------------------------------------------------------------
@register("fig_emg_timing", "paper", "Muscle activity and motion onset",
          params=("perspective", "stat", "pid"), perspectives=TRIAL_VIEWS,
          perspective="cohort", stat="mean_sd")
def fig_emg_timing(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    # The angle stored with the EMG profiles: the same cycles and the same start..end cut
    # as panel (b) and the SPM test (kinematic(..., "cycle") cuts at movement onset..offset).
    _, th = E.profile_ensembles(ctx, "open_close")
    pa = Panel("(a) Opening angle", Axis("Open-close cycle", PCT), Axis("Angle", DEG),
               legend="none", sharex="cycle")
    pa.series += _summary_series(th, req) if th.n_trials else []
    if not pa.series:
        _empty(pa)
    emg = E.emg_profiles(ctx, "open_close")
    pb = Panel("(b) Opener and closer activation", Axis("Open-close cycle", PCT),
               Axis("Activation (peak = 1)"), legend="auto", sharex="cycle")
    spm = ctx.table("stats_emg", "spm")
    row = spm[spm.test == "openers_vs_closers_open_close"] if not spm.empty else spm
    if not row.empty:                            # SPM supra-threshold clusters as bars
        from biomech.viz.views import blocks as B

        B.add_spm_bars(pb, [("openers ≠ closers", row.clusters.iloc[0], "context")])
    if emg.n_trials:
        refs = ctx.table("emg", "refs")
        for group, lab, rl in ((("digastric_l", "digastric_r"), "openers (digastric)",
                                "group:openers"),
                               (("masseter_r", "masseter_l", "temporalis_r", "temporalis_l"),
                                "closers (masseter, temporalis)", "group:closers")):
            idx = [emg.channels.index(m) for m in group if m in emg.channels]
            if not idx:
                continue
            data = emg.data[:, :, idx].copy()
            if not refs.empty:                     # shape-usable channels only (an01 §3)
                for k, pid in enumerate(emg.trials.pid):
                    use = set(refs[(refs.pid == pid) & refs.usable_shape].muscle)
                    for j, m in enumerate([emg.channels[i] for i in idx]):
                        if m not in use:
                            data[k, :, j] = np.nan
            with E._quiet():
                y = np.nanmean(data, axis=2)
                y = y / np.nanmax(y, axis=1, keepdims=True)
            g = E.Ensemble("grp", emg.grid, y, emg.trials, [lab], "cycle")
            if req.perspective == "cohort" or not req.pid:
                s = E.aggregate(g, "cohort", req.stat)
                c = s.center[0, :, 0]
                ci = req.stat in ("mean_sd", "median_iqr")
                pb.series.append(Series("band", g.grid, s.lo[0, :, 0], s.hi[0, :, 0], role=rl,
                                        style={"alpha": B.SD_BAND_ALPHA} if ci else {}))
                if ci:
                    B.add_ci_band(pb, g.grid, s.members[:, :, 0], role=rl,
                                  centre="median" if req.stat == "median_iqr" else "mean",
                                  seed=B.seed_of(ctx), where=np.isfinite(c))
                pb.series.append(Series("line", g.grid, c, role=rl, label=lab))
            else:
                e = g.select((g.trials.pid == req.pid).to_numpy())
                if not e.n_trials:
                    continue
                if req.perspective in ("trials", "trials+summary"):
                    for k in range(e.n_trials):
                        pb.series.append(Series("line", g.grid, e.data[k, :, 0], role=rl,
                                                style={"width": 0.4, "alpha": 0.45}))
                if req.perspective != "trials":
                    s = E.aggregate(e, "participant", req.stat)
                    pb.series += [Series("band", g.grid, s.lo[0, :, 0], s.hi[0, :, 0], role=rl),
                                  Series("line", g.grid, s.center[0, :, 0], role=rl, label=lab)]
    if not pb.series:
        _empty(pb)
    sm = ctx.table("timing", "summary")
    g = sm[(sm.task == "open_close") & (sm.direction == "opening") & (sm.detector == "tkeo")] \
        if not sm.empty else sm
    rule = ctx.onset_rule()
    ens = ctx.table("timing", "ensemble")
    ens = ens[(ens.task == "open_close") & (ens.direction == "opening")] if not ens.empty else ens
    # Rows: single-cycle lead under both onset rules (frozen plan, D6), the detector-free
    # ensemble events (X2), then the causal margin.
    rows = [("first", "lead_ms", "first onset"),
            ("drive", "lead_ms", "driving burst"),
            ("agonist", "ens", "opener rise"),
            ("antagonist", "ens", "closer release"),
            ("release", "rel", "causal release"),
            (rule, "anticipation_ms", "causal margin")]
    rel = ctx.table("timing", "release_summary")
    rel = rel[(rel.task == "open_close") & rel.primary.astype(bool)] if not rel.empty else rel
    pc = Panel("(c) EMG event before motion onset", Axis("Time", MS), Axis(""), legend="none")
    if not g.empty:
        for i, (rl, col, _) in enumerate(rows):
            if col == "ens":
                v = ens[ens.role == rl].lead_ms.to_numpy(float) if not ens.empty else np.empty(0)
            elif col == "rel":
                v = rel.anticipation_ms.to_numpy(float) if not rel.empty else np.empty(0)
            else:
                sub = g[g.rule == rl] if "rule" in g else g
                v = sub.groupby("pid")[col].mean().to_numpy(float)
            v = v[np.isfinite(v)]
            jit = np.linspace(-0.15, 0.15, v.size) if v.size > 1 else np.zeros(v.size)
            pc.series.append(Series("scatter", v, i + jit, role="context"))
            if v.size:
                lo, hi = B.median_ci(v, B.seed_of(ctx))
                if np.isfinite(lo) and np.isfinite(hi):
                    pc.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                            role="accent", style={"width": 1.4}))
                pc.series.append(Series("scatter", np.array([np.median(v)]), np.array([float(i)]),
                                        role="accent", style={"ms": "d", "size": 1.5}))
        pc.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        pc.y.ticks, pc.y.ticklabels = list(range(len(rows))), [r[2] for r in rows]
        pc.y.reverse = True
    else:
        _empty(pc)
    rule_txt = "driving burst" if rule == "drive" else "first onset"
    from biomech.viz.views.results_timing import release_setting, timing_params

    ens_p = timing_params(ctx, "ensemble")
    return FigureSpec("fig_emg_timing", "Muscle activity and motion onset", [pa, pb, pc], rows=1,
                      cols=3, width="text", height_in=1.6,
                      caption="(a) Opening angle over the open-close cycle; (b) peak-normalized "
                              "opener and closer activation (bars: SPM supra-threshold "
                              "clusters, p < 0.05, paired t, Holm-corrected); (c) time by which "
                              "an EMG event "
                              "precedes the earliest mechanical opening onset (> 0: EMG first): "
                              "single-cycle digastric onset (TKEO) taken as the first one in the "
                              "window (pre-specified) or as the burst that drives the movement "
                              "(post hoc); detector-free events of the cycle-aligned "
                              "envelopes, every cycle included: "
                              f"{100 * float(ens_p['rise_frac']):g} % rise of the openers and "
                              f"{100 * float(ens_p['release_frac']):g} % "
                              "release of the closers (exploratory); a causal closer-release "
                              f"detector (envelope {release_setting(ctx)}, exploratory); and "
                              "the margin left after "
                              f"a causal detector's decision ({rule_txt}). Dots: participants, "
                              "diamond: median, line: its 95 % BCa bootstrap CI. "
                              f"(a, b): {_persp_caption(th, req)}.",
                      perspective=req.perspective,
                      sources=["kinematics:cycles", "emg:profiles", "stats_emg:spm",
                               "timing:summary", "timing:ensemble", "timing:release_summary"])


# -- Figure 3: C3 wearable sensing ---------------------------------------------------------------
@register("fig_wearable", "paper", "Wearable sensing recovers opening and intent")
def fig_wearable(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    c = ctx.table("stats_imu", "imu_opening_cycles")
    c = c[c.task == "open_close"] if not c.empty else c
    pa = Panel("(a) IMU vs optical opening", Axis("Mean of IMU and optical", MM),
               Axis("IMU − optical", MM), legend="none")
    col = "opening_subject_quat_mm"
    if not c.empty and col in c:
        mean = ((c[col] + c.opening_opt_mm) / 2).to_numpy(float)
        diff = (c[col] - c.opening_opt_mm).to_numpy(float)
        pa.series.append(Series("scatter", mean, diff, role="context", style={"size": 0.8}))
        ba = ctx.table("stats_imu", "imu_opening_ba")
        r = ba[(ba.task == "open_close") & (ba.estimator == "quat") & (ba.model == "subject")]
        if not r.empty:
            b, lo, hi = (float(r[k].iloc[0]) for k in ("bias", "loa_lo", "loa_hi"))
            pa.series.append(Series("hline", y=np.array([b]), role="reference",
                                    text=[f"bias {B.snum(b, 1)}"]))
            pa.series.append(Series("hline", y=np.array([lo, hi]), role="threshold",
                                    text=[f"LoA {B.snum(lo, 1)}", f"LoA {B.snum(hi, 1)}"]))
    else:
        _empty(pa)
    sets = ["emg", "imu", "emg_imu", "emg_nodig", "emg_nodig_imu"]
    names = {"emg": "EMG", "imu": "IMU", "emg_imu": "EMG + IMU", "emg_nodig": "EMG, no digastric",
             "emg_nodig_imu": "no digastric + IMU"}
    pb = Panel("(b) Intent decoding", Axis("Balanced accuracy", lim=(0.0, 1.0)), Axis(""),
               legend="auto")
    has_ml = "ml_eval" in ctx.store.stages()
    lo_t = ctx.table("ml_eval", "loso") if has_ml else pd.DataFrame()
    bl_t = ctx.table("ml_eval", "blocked") if has_ml else pd.DataFrame()
    model = "svm"
    have = [s for s in sets if not lo_t.empty and (lo_t.feature_set == s).any()]
    for i, s in enumerate(have):
        u = lo_t[(lo_t.model == model) & (lo_t.feature_set == s)].balanced_accuracy.to_numpy()
        w = bl_t[(bl_t.model == model) & (bl_t.feature_set == s)].balanced_accuracy.to_numpy() \
            if not bl_t.empty else np.empty(0)
        if u.size:
            pb.series.append(Series("scatter", u, np.full(u.size, i - 0.12), role="faint"))
        if w.size:
            pb.series.append(Series("scatter", w, np.full(w.size, i + 0.12), role="faint"))
        if u.size and w.size:
            pb.series.append(Series("line", np.array([np.median(u), np.median(w)]),
                                    np.array([i - 0.12, i + 0.12]), role="context"))
        for vals, off, rl in ((u, -0.12, "protocol:loso"), (w, 0.12, "protocol:blocked")):
            lo, hi = B.median_ci(vals, B.seed_of(ctx)) if vals.size else (np.nan, np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                pb.series.append(Series("line", np.array([lo, hi]), np.array([i + off] * 2),
                                        role=rl, style={"width": 1.2}))
        if u.size:
            pb.series.append(Series("scatter", np.array([np.median(u)]), np.array([i - 0.12]),
                                    role="protocol:loso", style={"size": 1.4},
                                    label="unseen subject" if i == 0 else None))
        if w.size:
            pb.series.append(Series("scatter", np.array([np.median(w)]), np.array([i + 0.12]),
                                    role="protocol:blocked", style={"size": 1.4},
                                    label="same subject" if i == 0 else None))
    if have:
        pb.series.append(Series("vline", x=np.array([1 / 7]), role="threshold", label="chance"))
        pb.y.ticks, pb.y.ticklabels = list(range(len(have))), [names[s] for s in have]
        pb.y.reverse = True
    else:
        _empty(pb)
    arr = ctx.arrays("ml_eval", COHORT_UNIT) if has_ml else {}
    conf = arr.get(f"confusion__emg_imu|{model}")
    pc = Panel("(c) Confusion, unseen subject", Axis("Predicted"), Axis("Actual"), legend="none")
    if conf is not None:
        Z = 100 * conf / np.maximum(conf.sum(1, keepdims=True), 1)
        pc.series.append(Series("heatmap", z=Z, text=[f"{v:.0f}" for v in Z.ravel()]))
        pc.x.ticks = pc.y.ticks = list(range(7))
        pc.x.ticklabels = pc.y.ticklabels = list(CLASSES)
    else:
        _empty(pc)
    return FigureSpec("fig_wearable", "Wearable sensing recovers opening and intent",
                      [pa, pb, pc], rows=1, cols=3, width="text", height_in=1.55,
                      caption="(a) Bland-Altman agreement of IMU-estimated and optical opening "
                              "per cycle (repeated-measures limits of agreement); (b) balanced "
                              "accuracy of 7-class intent decoding (SVM) for unseen subjects "
                              "(leave-one-subject-out) vs the same subject (blocked), medians "
                              "joined (lines: their 95 % BCa bootstrap CIs), dots: "
                              "participants; (c) row-normalized confusion (%), "
                              "EMG + IMU, unseen subject.",
                      sources=["stats_imu:imu_opening_cycles", "stats_imu:imu_opening_ba",
                               "ml_eval:loso", "ml_eval:blocked", "ml_eval:confusion"])


# -- Figure 4: S2 variability ----------------------------------------------------------------------
SPEC_NAMES = {"mio_mm": "largest opening", "open_close|opening|lead": "opening lead",
              "open_close|closing|lead": "closing lead", "chewing|opening|lead": "chewing lead",
              "median_pct|open_close|digastric_l": "digastric L, open-close",
              "median_pct|open_close|digastric_r": "digastric R, open-close",
              "median_pct|chewing|masseter_r": "masseter R, chewing",
              "median_pct|chewing|masseter_l": "masseter L, chewing"}


@register("fig_variability", "paper", "Variability and processing sensitivity")
def fig_variability(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    sc = ctx.table("variability", "spec_curve")
    keep = sc[sc.outcome.isin(SPEC_NAMES)].sort_values("spread_over_sd").reset_index(drop=True) \
        if not sc.empty else sc
    pa = Panel("(a) Processing spread / between-subject SD", Axis("Ratio"), Axis(""),
               legend="none")
    if not keep.empty:
        y = np.arange(len(keep), dtype=float)
        v = keep.spread_over_sd.to_numpy(float)
        for yi, vi in zip(y, v):
            pa.series.append(Series("line", np.array([0.0, vi]), np.array([yi, yi]),
                                    role="context"))
        pa.series.append(Series("scatter", v, y, role="accent"))
        pa.series.append(Series("vline", x=np.array([1.0]), role="threshold", label="1 SD"))
        pa.y.ticks = list(y)
        pa.y.ticklabels = [SPEC_NAMES[o] for o in keep.outcome]
    else:
        _empty(pa)
    arr = ctx.arrays("stats_emg", COHORT_UNIT)
    x = np.linspace(0, 100, 101)
    pb = Panel("(b) Between / within-subject variance", Axis("Cycle", PCT),
               Axis("Variance ratio", log=True), legend="auto")
    from biomech.viz.views.results_emg import variability_inputs, varratio_ci

    inputs = variability_inputs(ctx)
    n_ci = 0
    for key, lab, rl in (("theta", "opening angle", "accent"),
                         ("masseter_r_chewing", "masseter R, chewing", "entity:masseter_r")):
        k = f"varratio__{key}__ratio"
        if k in arr:
            ci = varratio_ci(ctx, key, arr[k], seed=B.seed_of(ctx), inputs=inputs)
            if ci is not None:
                pb.series.append(Series("band", x, ci[0], y2=ci[1],
                                        role="band" if rl == "accent" else rl))
                n_ci += 1
    for key, lab, rl in (("theta", "opening angle", "accent"),
                         ("masseter_r_chewing", "masseter R, chewing", "entity:masseter_r")):
        k = f"varratio__{key}__ratio"
        if k in arr:
            pb.series.append(Series("line", x, arr[k], label=lab, role=rl))
    pb.series.append(Series("hline", y=np.array([1.0]), role="reference", label="equal"))

    spm = ctx.table("stats_emg", "spm")
    an = {r.test: r.clusters for r in spm.itertuples()} if not spm.empty else {}
    n_bars = B.add_spm_bars(pb, [
        ("participants differ, angle", an.get("anova1_opening_angle_open_close"), "accent"),
        ("participants differ, masseter R", an.get("anova1_masseter_r_chewing"),
         "entity:masseter_r")])
    return FigureSpec("fig_variability", "Variability and processing sensitivity", [pa, pb],
                      rows=1, cols=2, width="text", height_in=1.5,
                      caption="(a) Spread of cohort means across defensible processing variants, "
                              "relative to the between-participant SD (above 1: the processing "
                              "choice matters more than who was measured); (b) pointwise ratio "
                              "of between- to within-participant variance along the cycle"
                              + (" (bands: pointwise 95 % bootstrap CIs, participants "
                                 "resampled)" if n_ci else "")
                              + ("; " + B.spm_caption("one-way ANOVA across participants")
                                 if n_bars else "") + ".",
                      sources=["variability:spec_curve", "stats_emg:varratio",
                               "stats_emg:spm"])


# -- Table I: reference values ---------------------------------------------------------------------
REF_ROWS = [
    ("opening_selfpaced_mm", "Self-paced opening", MM), ("max_observed_opening_mm",
                                                          "Largest observed opening", MM),
    ("opening_angle_deg", "Opening angle", DEG), ("protrusion_mm", "Protrusion", MM),
    ("laterotrusion_left_mm", "Laterotrusion L", MM), ("laterotrusion_right_mm",
                                                       "Laterotrusion R", MM),
    ("opening_deviation_mm", "Opening deviation", MM),
    ("condyle_left_disp_mm", "Condylar path L", MM),
    ("peak_open_vel_dps", "Peak opening velocity", DPS),
    ("chewing_freq_hz", "Chewing frequency", "\\hertz"),
]
REL_KEY = {"opening_selfpaced_mm": ("open_close", "opening_mm", ""),
           "opening_angle_deg": ("open_close", "theta_y_deg", ""),
           "protrusion_mm": ("protrusion_retrusion", "protrusion_mm", ""),
           "laterotrusion_left_mm": ("left_right", "lateral_mm", "left"),
           "laterotrusion_right_mm": ("left_right", "lateral_mm", "right"),
           "peak_open_vel_dps": ("open_close", "peak_open_vel_dps", "")}


def _u(unit: str) -> str:
    from biomech.viz.style import plain_unit

    return plain_unit(unit)


@register("table_reference", "paper", "Reference values")
def table_reference(req: ViewRequest) -> TableSpec:
    ctx = req.ctx
    rv = ctx.table("stats_kinematics", "reference_values")
    rel = ctx.table("stats_kinematics", "reliability")
    rows = []
    for key, label, unit in REF_ROWS:
        g = rv[rv.metric == key].set_index("group")
        if "all" not in g.index:
            continue
        a = g.loc["all"]
        r = {"metric": f"{label} ({_u(unit)})", "mean": a["mean"], "sd": a["sd"],
             "ci": B.ci(a.ci_lo, a.ci_hi, 1),
             "f": g.loc["f", "mean"] if "f" in g.index else np.nan,
             "m": g.loc["m", "mean"] if "m" in g.index else np.nan,
             "g": g.loc["f_minus_m", "g"] if "f_minus_m" in g.index else np.nan}
        if key in REL_KEY:
            task, metric, side = REL_KEY[key]
            side_col = rel.side.fillna("") if "side" in rel else ""
            q = rel[(rel.task == task) & (rel.metric == metric) & (side_col == side)]
            r["icc"] = q.icc31.iloc[0] if not q.empty else np.nan
            r["mdc"] = q.mdc95.iloc[0] if not q.empty else np.nan
        rows.append(r)
    n_sex = {k: int(rv[rv.group == k].n.max()) if (rv.group == k).any() else 0 for k in ("f", "m")}
    cols = [Column("metric", "Metric", align="l"), Column("mean", "Mean", decimals=1),
            Column("sd", "SD", decimals=1), Column("ci", "95 % CI", align="c"),
            Column("f", f"Female ($n = {n_sex['f']}$)", decimals=1, group="Mean by sex"),
            Column("m", f"Male ($n = {n_sex['m']}$)", decimals=1, group="Mean by sex"),
            Column("g", "Hedges' $g$", decimals=2), Column("icc", "ICC", decimals=2),
            Column("mdc", "MDC", decimals=1)]
    n = int(rv[rv.group == "all"].n.max()) if not rv.empty else 0
    t = TableSpec("table_reference", "", cols, rows,
                  # the statistics are named in the caption (2026-11-13: the note repeated them)
                  notes="g: female minus male, exploratory. ICC and MDC: across the cycles of "
                        "the session, for laterotrusion per side.",
                  sources=["stats_kinematics:reference_values", "stats_kinematics:reliability"],
                  width="text")
    return attach(t, CaptionSpec(
        "reference values of jaw function in adults without jaw-joint symptoms (self-report "
        "in the consent form): range of motion and opening speed, with their reliability across "
        "the cycles of one session",
        n=f"n = {n} participants",
        statistic="per participant one value; mean, standard deviation (SD) and 95 % BCa CI of "
                  "the mean over the participants; means by sex with Hedges' g (exploratory); "
                  "ICC(3,1) and minimal detectable change (MDC) at the 95 % "
                  "level across cycles", evidence="descriptive"))


GROUP_LABEL = {"workspace": "Workspace (lower bounds)", "kinematics": "Kinematics",
               "dynamics": "Dynamics", "intent": "Intent detection", "sensing": "Wearable sensing"}
# Requirement groups per table scope: the conference paper carries workspace,
# speed, bandwidth, alignment (passive translation), C1 (fixed-hinge misalignment) and C3a
# (wearable sensing); the timing (C2, X2, X3) and decoding rows are journal-only.
REQ_SCOPES = {"full": tuple(GROUP_LABEL),
              "conference": ("workspace", "kinematics", "dynamics", "sensing"),
              # journal (2026-11-04): the full table ran off the page and lost its last
              # rows, so the journal splits it into the capacities (with the post hoc
              # envelope rows) and the intent-detection table (table_requirements_intent)
              # 2026-11-10: the journal table is the clean specification, the
              # P95 capacity rows only; the hinge rows are in table_compatibility and the IMU
              # rows in the sensing results
              "capacities": ("workspace", "kinematics", "dynamics"),
              "intent": ("intent",)}
# C2 rows: "not confirmed" is the user's decision (2026-10-15): the result depends on the onset
# rule (first onset significant, driving burst not). The label is refused once both rules
# are significant (the premise of the decision no longer holds; ask the user).
C2_KEYS = {"lead_first_ms": "first", "lead_drive_ms": "drive"}
ALPHA = 0.05


def _evidence_label(value: str) -> str:
    """Reader label of an evidence level (vocab domain ``evidence``)."""
    from biomech.viz.views import vocab as V

    return V.labels("evidence").get(value, value)


def _c2_holm_p(ctx, rule: str) -> float | None:
    """Holm p of the C2 opening lead (open-close, TKEO) for onset ``rule``, from the stored
    timing statistics; None when not stored."""
    if "stats_timing" not in ctx.store.stages():
        return None
    tc = ctx.table("stats_timing", "timing_cohort")
    if tc.empty or "p_holm" not in tc:
        return None
    q = tc[(tc.task == "open_close") & (tc.direction == "opening") & (tc.detector == "tkeo")
           & (tc.metric == "lead_ms") & (tc.rule == rule)]
    return float(q.p_holm.iloc[0]) if len(q) == 1 and pd.notna(q.p_holm.iloc[0]) else None


def _shift_null(ctx, key: str) -> str:
    """The cohort median of the C2 shift null (D23) for the two lead rows, else ''."""
    if key not in C2_KEYS or "stats_timing" not in ctx.store.stages():
        return ""
    tc = ctx.table("stats_timing", "timing_cohort")
    q = tc[(tc.task == "open_close") & (tc.direction == "opening") & (tc.detector == "tkeo")
           & (tc.metric == "lead_ms") & (tc.rule == C2_KEYS[key])] if "lead_null_ms" in tc \
        else tc.iloc[:0]
    return f"{_num(float(q.lead_null_ms.iloc[0]), 0)} ms" if len(q) == 1 \
        and pd.notna(q.lead_null_ms.iloc[0]) else ""


def requirement_evidence(ctx, key: str, stored: str) -> str:
    """The evidence label a requirement row shows (presentation only; the stored table keeps
    its value). ``etc/vocab.yaml`` domain ``requirement_evidence`` maps a requirement key to
    the evidence level the paper states (C3a generic model: the pre-specified primary
    outcome; C2: pre-specified, not confirmed, because the result depends on the onset
    rule). A C2 "not confirmed" label is refused when the stored Holm p of both onset rules
    is below 0.05 (the result would no longer depend on the rule)."""
    from biomech.viz.views import vocab as V

    ev = V.labels("requirement_evidence").get(key, stored)
    if key in C2_KEYS and ev.startswith("not_confirmed"):
        ps = [_c2_holm_p(ctx, rule) for rule in C2_KEYS.values()]
        if all(p is not None and p < ALPHA for p in ps):
            raise ValueError(f"table_requirements: {key} is labelled 'not confirmed' "
                             "(etc/vocab.yaml requirement_evidence), but both onset rules "
                             f"are significant (Holm p = {ps[0]:.3f}, {ps[1]:.3f})")
    return _evidence_label(ev)
# Reader names that differ from the stored requirement label (presentation only).
REQ_NAME = {"bandwidth_hz": "Movement bandwidth",
            "closer_release_ms": "Closer release before opening, average envelope",
            "c2e_h2_ms": "Closer release before opening, per cycle against the shift null"}


UNIT_TEXT = {"mm": " mm", "deg": "°", "deg/s": "°/s", "deg/s^2": "°/s²", "Hz": " Hz",
             "ms": " ms", "": ""}


def _num(v: float, dec: int) -> str:
    from biomech.viz.views import blocks as B

    return B.num(v, dec)          # real minus, no negative zero


def req_decimals(value: float) -> int:
    """Decimals of a requirement row (its design value, CI, bound and maximum alike): one below
    20, none above. Shared with the in-text macros (numbers ``fmt: req``), so Table I and the
    text print the same digits (2026-11-05)."""
    return 1 if np.isfinite(value) and abs(value) < 20 else 0


def _value_text(r: pd.Series) -> str:
    """Design value, unit and CI, all with the value's precision (:func:`req_decimals`)."""
    dec = req_decimals(float(r.value))
    u = UNIT_TEXT.get(r.unit, f" {r.unit}")
    if np.isfinite(r.ci_lo) and np.isfinite(r.ci_hi):
        ci = f" [{_num(r.ci_lo, dec)}, {_num(r.ci_hi, dec)}]"
    elif np.isfinite(r.ci_hi):
        ci = f" (upper CI {_num(r.ci_hi, dec)})"
    elif np.isfinite(r.ci_lo):            # P95 design rows: one-sided lower 95 % bound (E31)
        ci = f" (bound {_num(r.ci_lo, dec)})"
    else:
        ci = ""
    return f"{_num(r.value, dec)}{u}{ci}"


_CODE_REF = re.compile(r"\s*\((?:[A-Z]\d+[a-z]?(?:,\s*)?)+\)|,\s*[A-Z]\d+[a-z]?(?=[);]|$)")


def _plain_basis(text: str) -> str:
    """A derivation without analysis-plan codes ("(X2)", ", D10)") and with known identifiers
    in reader words (``drive_taskbase`` -> "driving burst, in-task threshold") -- the table
    states the evidence level in its own column."""
    from biomech.viz.views import vocab as V

    t = re.sub(r"\(\s*\)", "", _CODE_REF.sub("", text)).replace(" ;", ";").strip()
    t = V.rewrite_ids(t).replace(">=", "≥").replace("<=", "≤").replace("->", "→")
    for a, b in _BASIS_WORDS:
        t = re.sub(a, b, t)
    if "not tested (the preceding hypothesis was not rejected" in t:
        # a p computed for a hypothesis the gatekeeping never tested reads as a result
        t = t.replace("intersection-union test p =", "intersection-union test p (descriptive) =")
    # "anticipation a ms vs chance b ms (paired p = x): not above chance": say which way
    t = re.sub(r"anticipation (\d+(?:\.\d+)?) ms vs chance (\d+(?:\.\d+)?) ms \(paired p = "
               r"([\d.]+)\): not above chance",
               lambda m: (f"anticipation {m[1]} ms vs chance {m[2]} ms (paired p = {m[3]}; "
                          f"{'shorter' if float(m[1]) < float(m[2]) else 'not longer'} than "
                          "chance)"), t)
    return t.strip("; ")


# Reader wording of the stored derivation strings: test names spelled out, internal
# hypothesis / parameter labels dropped, p and zero formatted for print.
_BASIS_WORDS = (
    # serial comma in stored derivations (the store keeps its own text, 2026-11-12)
    (r"open-close, chewing and free motion", "open-close, chewing, and free motion"),
    # abbreviation used only in this note, spelled out (2026-11-13)
    (r"\bno MIO recorded\b", "no maximal interincisal opening recorded"),
    (r"\bIUT p\b", "intersection-union test p"),
    (r"\s*\(exploratory P\d+\)", " (exploratory)"),
    (r"not tested \(H\d not rejected, serial gatekeeping\)",
     "not tested (the preceding hypothesis was not rejected; serial gatekeeping)"),
    (r";?\s*event-order decision: not shown", ""),
    (r"\bp = 0\.000\b", "p < 0.001"),
    (r"(?<![\d.])[−-]0\.(0+)(?!\d)", r"0.\1"),
    (r"\blocking 0\.000\b", "locking < 0.001"),
    (r"\s*\((?:[A-Z]\d+[a-z]?\s+)?P\d+\)", ""),          # "(X6 P3)": plan prediction labels
    (r"\(X2 parameters\)", "(the event-aligned analysis's parameters)"),
    (r"\bthe S1 surrogate background\b", "the circular-shift surrogate background"),
    (r"\bS1 surrogates?\b", "circular-shift surrogates"),
    (r"(?<![\w.\])])-(?=\d)", "−"),                       # a real minus before a number
)


def _decoding_rows(ctx) -> list[dict]:
    """C3b from ml_eval (an optional plugin): medians with their stored CIs."""
    if "ml_eval" not in ctx.store.stages():
        return []
    out = []
    ms = ctx.table("ml_eval", "summary")
    s = ms[(ms.protocol == "loso") & (ms.model == "svm") & (ms.feature_set == "emg_imu")
           & (ms.metric == "balanced_accuracy")]
    if not s.empty:
        r = s.iloc[0]
        out.append({"req": "Intent decoding, unseen user",
                    "value": f"BA {r['median']:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]",
                    "stat": "median",
                    "basis": f"{len(CLASSES)} classes, leave-one-subject-out, SVM, EMG + IMU; "
                             f"chance {1 / len(CLASSES):.2f}",
                    "evidence": _evidence_label("confirmatory")})
    vs = ctx.table("ml_eval", "variants_summary")
    if not vs.empty:
        v = vs[(vs.variant == "few_shot") & (vs.feature_set == "emg_imu")
               & (vs.metric == "balanced_accuracy")]
        if not v.empty:
            v = v[v.k == v.k.max()].iloc[0]
            stride = float((ctx.config.params("ml_windows").get("window") or {})
                           .get("stride_s", 0.05))
            out.append({"req": "Intent decoding, after calibration",
                        "value": f"BA {v['median']:.2f} (IQR {v.q1:.2f}–{v.q3:.2f})",
                        "stat": "median",
                        "basis": f"plus the user's first {int(v.k)} windows "
                                 f"({int(v.k) * stride:g} s) per movement",
                        "evidence": _evidence_label("exploratory")})
    return out


REQ_FORMS = ("columns", "notes")


# Written table notes of the journal's capacities table (2026-11-13): what one participant's
# value is, in a few words. The stored derivations repeated "Lower bound" (which collided with
# the column of that name, the confidence bound of the P95) and "Post hoc" (the Evidence
# column carries the level); both are said once, in the caption and in REQ_NOTE_TAIL.
REQ_NOTE_SHORT = {
    "opening_mm": "Largest opening in any task (no maximal interincisal opening recorded)",
    "protrusion_mm": "Peak anterior incisal displacement in the protrusion task",
    "laterotrusion_mm": "Peak lateral incisal displacement in the left-right task",
    "opening_angle_deg": "Peak angle in self-paced open-close",
    "posterior_excursion_mm": "99th percentile of the incisal displacement behind the rest "
                              "position over the five movement tasks",
    "translation_mm": "Condyle-point excursion in opening, larger side",
    "peak_vel_dps": "Largest cycle peak in self-paced opening",
    "close_vel_dps": "Largest cycle peak in self-paced closing",
    "free_peak_open_vel_dps": "Largest opening velocity in free motion",
    "peak_acc_dps2": "Largest |angular acceleration| over open-close, chewing, and free motion",
    "bandwidth_hz": "F95 of the IMU angular-velocity power confirmed by the optical "
                    "measurement (coherence), largest over the opening tasks",
}
REQ_NOTE_TAIL = ("Each note states the value of one participant. The velocities and the "
                 "acceleration underestimate the true peaks (40-Hz optics, 6-Hz low-pass).")


def _notes_layout(rows: list[dict], text_key: str, ev_key: str,
                  name_key: str) -> tuple[list[dict], str] | None:
    """Form ``notes`` of a requirements table: each row's derivation becomes a lettered table
    note (rows with the same text share a letter) and its evidence label a short code with a
    key, so a page-high table keeps every word at full text width (2026-11-05, the JNER
    tables ran off the page). None when there are more texts than note letters."""
    from biomech.viz.caption import _sentence
    from biomech.viz.render_pgf import SUP_LETTERS
    from biomech.viz.views import vocab as V

    texts = list(dict.fromkeys(r[text_key] for r in rows
                               if not r.get("_rule") and r.get(text_key)))
    if len(texts) > len(SUP_LETTERS):
        return None
    letter = dict(zip(texts, SUP_LETTERS))
    out = []
    for r in rows:
        r = dict(r)
        txt = r.pop(text_key, "")
        if txt:
            r[name_key] = f"{r[name_key]}{letter[txt]}"
        out.append(r)
    key = ""          # evidence in words since 2026-11-10 (no code key)
    notes = " ".join(f"{letter[x]}{_sentence(x)}" for x in texts)
    return out, " ".join(x for x in (key, notes) if x)


@register("table_requirements", "paper", "Exoskeleton requirements",
          choices={"scope": tuple(REQ_SCOPES), "form": REQ_FORMS}, scope="full",
          form="columns")
def table_requirements(req: ViewRequest) -> TableSpec:
    """Table I: every requirement of the ``requirements`` stage (design value, 95 % CI,
    statistic, derivation, evidence level; capacity values are the 95th percentile across
    participants, stats.requirements) plus the decoding results of ml_eval. ``scope``:
    ``full`` (every group), ``conference`` (workspace, speed, bandwidth, alignment, C1,
    C3a; no timing or decoding rows), ``capacities`` (the same groups with the
    post hoc envelope rows, the journal's Table I) or ``intent`` (see
    :func:`table_requirements_intent`). ``form``: ``columns`` (derivation and evidence as
    columns) or ``notes`` (:func:`_notes_layout`)."""
    return _requirements_table(req, str(req.options.get("scope", "full")), "table_requirements")


@register("table_requirements_intent", "paper", "Intent-detection requirements",
          choices={"form": REQ_FORMS}, form="columns")
def table_requirements_intent(req: ViewRequest) -> TableSpec:
    """The intent-detection rows of :func:`table_requirements` (EMG timing, closer release,
    decoding) as their own table: with the capacities in one table the journal's Table I ran
    off the page. ``form`` as in :func:`table_requirements`."""
    return _requirements_table(req, "intent", "table_requirements_intent")


def _requirements_table(req: ViewRequest, scope: str, fid: str) -> TableSpec:
    ctx = req.ctx
    if scope not in REQ_SCOPES:
        raise ValueError(f"table_requirements: scope {scope!r} not in {tuple(REQ_SCOPES)}")
    groups = REQ_SCOPES[scope]
    rq = ctx.table("requirements", "requirements") if "requirements" in ctx.store.stages() \
        else pd.DataFrame()
    rows: list[dict] = []
    if rq.empty:
        rows.append({"req": "requirements stage not run", "value": "", "stat": "",
                     "basis": "", "evidence": ""})
    for g, glabel in GROUP_LABEL.items():
        if g not in groups:
            continue
        part = rq[rq.group == g] if not rq.empty else rq
        if scope == "conference" and not part.empty and "evidence" in part:
            part = part[part.evidence != "post_hoc_descriptive"]   # X7 rows: journal only
        extra = _decoding_rows(ctx) if g == "intent" else []
        if part.empty and not extra:
            continue
        if rows:
            rows.append({"_rule": True})
        if scope == "capacities":
            # the caption says it; "(lower bounds)" beside the column "Lower bound" (the
            # confidence bound of the P95) read as the same thing (2026-11-13)
            glabel = glabel.replace(" (lower bounds)", "")
        rows.append({"req": glabel, "value": "", "stat": "", "basis": "", "evidence": ""})
        if scope == "capacities" and not part.empty:
            part = part[part.statistic.astype(str).str.startswith("P95")]
        for _, r in part.iterrows():
            basis = _plain_basis(str(r.derivation) + (f"; {r.note}" if isinstance(r.note, str)
                                                      and r.note else ""))
            if scope == "capacities" and req.options.get("form", "columns") == "notes":
                basis = REQ_NOTE_SHORT.get(str(r.key), basis)
            dec = req_decimals(float(r.value))
            u = UNIT_TEXT.get(r.unit, f" {r.unit}")
            rows.append({"req": REQ_NAME.get(str(r.key), r.requirement),
                         "value": _value_text(r),
                         "p95": f"{_num(r.value, dec)}{u}",
                         "lower": _num(r.ci_lo, dec) if np.isfinite(r.ci_lo) else "--",
                         "largest": _num(r.get("sample_max", np.nan), dec)
                         if np.isfinite(float(r.get("sample_max", np.nan))) else "--",
                         "null": _shift_null(ctx, str(r.key)),
                         "stat": "P95" if str(r.statistic).startswith("P95") else "median",
                         "basis": basis,
                         "evidence": requirement_evidence(ctx, str(r.key), str(r.evidence))})
        rows += extra
    shown = rq[rq.group.isin(groups)] if not rq.empty else rq
    n_part = int(shown.n.max()) if not shown.empty and "n" in shown else len(ctx.pids())
    cols = [Column("req", "Outcome" if scope == "intent" else "Requirement", align="X"),
            Column("value", "Value [95 % CI]", align="l"),
            Column("stat", "Statistic", align="l"), Column("basis", "Derivation", align="X"),
            Column("evidence", "Evidence", align="X")]
    notes = ""
    compact = _notes_layout(rows, "basis", "evidence", "req") \
        if req.options.get("form", "columns") == "notes" else None
    if compact is not None:
        rows, notes = compact
        if scope == "capacities":
            notes = f"{notes} {REQ_NOTE_TAIL}"
        cols = [cols[0], cols[1], cols[2], Column("evidence", "Evidence", align="l")]
        if scope == "intent":            # the shift null of the two C2 lead rows (D23)
            # every intent row is a cohort median: the statistic goes into the caption
            #
            cols = [cols[0], cols[1], Column("null", "Shift null", align="l"), cols[3]]
    if scope == "capacities":
        # 2026-11-10: the design value, its exact one-sided lower 95 % bound and the largest
        # participant value as columns (the P95 rows only)
        cols = [Column("req", "Requirement", align="X"),
                Column("p95", "Design value (P95)", align="l"),
                Column("lower", "Lower bound", align="l"),
                Column("largest", "Largest value", align="l"),
                Column("evidence", "Evidence", align="l")]
    t = TableSpec(fid, "", cols, rows, notes=notes,
                  sources=["requirements:requirements"] + (
                      ["ml_eval:summary", "ml_eval:variants_summary"] if "intent" in groups
                      else []), width="text")
    if scope == "intent":
        return attach(t, CaptionSpec(
            # the paper sets its own bold title: the caption starts with the content
            #
            "the timing of muscle activity relative to the jaw motion, the closer release, and "
            "the decoding of the movement class", n=f"n = {n_part} participants",
            statistic="every value is a cohort median with its 95 % BCa CI in brackets, or "
                      "its IQR where no CI is stored. The shift null is "
                      "the cohort median lead that the same detector finds in EMG shifted at "
                      "random against the motion",
            notes=["TKEO: Teager–Kaiser energy operator (onset detector). BA: balanced "
                   "accuracy. SVM: support vector machine. IQR: interquartile range."]))
    if scope == "capacities":
        return attach(t, CaptionSpec(
            "the 95th-percentile rows of every requirement group, with the bound and the "
            "largest participant that frame each design value", n=f"n = {n_part} participants",
            statistic="design value: 95th percentile across the participants (Harrell–Davis "
                      "estimate). Lower bound: its exact one-sided lower 95 % confidence bound "
                      "(an order statistic; the two-sided interval is not estimable at "
                      f"n = {n_part}). Largest value: the largest participant",
            notes=["DoF: degrees of freedom.",
                   "Workspace values are lower bounds (no maximal opening recorded)."]))
    return attach(t, CaptionSpec(
        "derived requirements for a jaw exoskeleton", n=f"n = {n_part} participants",
        statistic="capacity values: 95th percentile across the participants (Harrell–Davis "
                  "estimate with its one-sided lower 95 % bound, an order statistic; with "
                  f"n = {n_part} it lies near the largest participants); performance values: "
                  "cohort medians (BCa CI)",
        notes=["MIO: maximal incisal opening; DoF: degrees of freedom; RMS: root mean "
               "square.",
               "Workspace values are lower bounds (no maximal opening recorded).",
               "P95 design values: the bound in parentheses is the one-sided lower 95 % "
               "confidence bound of the P95 (its two-sided interval is not estimable at this "
               "n); medians: 95 % CI in brackets."]))
