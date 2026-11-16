# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Intent-decoding results (stages ``ml_windows`` and ``ml_eval`` of the ml package).

biomech never imports ml at module level (ml depends on biomech); the class names come
from ``ml.config.classes`` when it is installed. Participants appear as pseudonyms; the
stored array keys (``pred__{pid}__...``) are internal. Late fusion, the reduced EMG
block, few-shot calibration and the D13 fusion diagnostics are exploratory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.results import COHORT_UNIT

PRIMARY = ("emg_imu", "svm")
BA = Axis("Balanced accuracy", lim=(0.0, 1.0))
# Decoding metrics of ml_eval:summary without a catalog entry (label, unit).
METRIC = {"free_balanced_accuracy": ("Balanced accuracy, free motion", ""),
          "latency_ms_median": ("Detection latency (median)", V.MSEC),
          "detected_frac": ("Movements detected", ""),
          "fa_frac": ("Rest windows with a false activation", ""),
          "fa_per_min": ("False activations at rest", V.PER_MIN)}


def _metric_axis(m: str) -> Axis:
    if V.known(m):
        return V.axis(m)
    lab, unit = METRIC.get(m, (m.replace("_", " "), ""))
    return Axis(lab, unit)


def _classes() -> list[str]:
    try:
        from ml.config.classes import CLASS_NAMES
    except ImportError:                       # ml not installed: the README §4 order
        return ["REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT"]
    return [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]


def _has(ctx) -> bool:
    return "ml_eval" in ctx.store.stages()


def _arrays(ctx) -> dict:
    return ctx.arrays("ml_eval", COHORT_UNIT) if _has(ctx) else {}


def _fset(f: str) -> str:
    return V.term("feature_set", f)


def _config_label(c: str) -> str:
    f, _, m = c.partition("|")
    return f"{_fset(f)}, {V.MODEL_LABEL.get(m, m)}"


def _configs(ctx) -> list[str]:
    """Confusion-matrix configurations (feature set|model), the primary one first."""
    keys = sorted(k.split("__", 1)[1] for k in _arrays(ctx).keys() if k.startswith("confusion__"))
    return sorted(keys, key=lambda k: k != "|".join(PRIMARY))


def _pred_fsets(ctx) -> list[str]:
    """Feature sets with a stored window-level decision timeline, primary first."""
    fs = sorted({k.split("__")[2] for k in _arrays(ctx).keys()
                 if k.startswith("pred__") and k.endswith("__yhat")})
    return sorted(fs, key=lambda f: f != PRIMARY[0])


@register("ml_confusion", "dashboard", "Confusion matrix", domain="ml",
          choices={"config": _configs})
def ml_confusion(req: ViewRequest) -> FigureSpec:
    """Row-normalized LOSO confusion matrix of one configuration (all participants)."""
    ctx = req.ctx
    cfg = req.options.get("config") or "|".join(PRIMARY)
    conf = _arrays(ctx).get(f"confusion__{cfg}")
    cls = _classes()
    p = Panel("Unseen subject", Axis("Predicted"), Axis("Actual"), [], legend="none")
    if conf is not None:
        C = np.asarray(conf, float)
        Z = 100 * C / np.maximum(C.sum(1, keepdims=True), 1)
        p = B.heat(Z, cls[:len(Z)], cls[:Z.shape[1]], "Unseen subject (% of true class)")
        p.x.label, p.y.label = "Predicted", "Actual"
        p.note = f"{int(C.sum())} windows"
    else:
        p.note = "no data"
    return FigureSpec("ml_confusion", "Confusion matrix", [p], width="column", height_in=2.4,
                      caption=f"{_config_label(cfg)}: confusion of the 7-class intent decoder, "
                              "leave-one-subject-out, windows of all held-out participants "
                              "pooled, row-normalized (% of the true class).",
                      sources=[f"ml_eval:confusion__{cfg}"])


@register("ml_timeline", "dashboard", "Decision timeline", params=("pid",), domain="ml",
          choices={"fset": _pred_fsets})
def ml_timeline(req: ViewRequest) -> FigureSpec:
    """True vs predicted class over time for one held-out participant (primary model)."""
    ctx = req.ctx
    pid = req.pid if req.pid in ctx.pids() else (ctx.pids() or [None])[0]
    fs = req.options.get("fset") or PRIMARY[0]
    a = _arrays(ctx)
    base = f"pred__{pid}__{fs}"
    t, y, yh = (a.get(f"{base}__{k}") for k in ("t_us", "y", "yhat"))
    cls = _classes()
    p = Panel("Decoder output", Axis("Time in recording", V.SEC), Axis("Class"), [])
    if t is not None and y is not None and yh is not None and len(t):
        ts = (np.asarray(t, float) - float(t[0])) / 1e6
        o = np.argsort(ts, kind="stable")
        ts, yy, yp = ts[o], np.asarray(y, float)[o], np.asarray(yh, float)[o]
        yy[yy < 0] = np.nan                      # unlabelled windows
        gap = np.r_[False, np.diff(ts) > 1.0]   # do not join across phase breaks
        tl, yl = np.insert(ts, np.flatnonzero(gap), np.nan), np.insert(yy, np.flatnonzero(gap),
                                                                       np.nan)
        p.series += [Series("line", tl, yl, role="context", label="true class",
                            style={"width": 1.6}),
                     Series("scatter", ts, yp, role="accent", label="predicted",
                            style={"size": 0.8})]
        ok = np.isfinite(yy)
        p.note = f"window accuracy {100 * np.mean(yy[ok] == yp[ok]):.0f} %" if ok.any() else ""
    else:
        p.note = "no decoder output"
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(cls))], cls
    who = f"P{ctx.pseudonym(pid)}" if pid else "no participant"
    return FigureSpec("ml_timeline", "Decision timeline", [p], width="text", height_in=2.0,
                      caption=f"{who}, {_fset(fs)}, {V.MODEL_LABEL.get(PRIMARY[1])}: "
                              "window-level output of the leave-one-subject-out decoder "
                              "(trained without this participant) against the labelled class, "
                              "over the recording (phase breaks not joined).",
                      sources=[f"ml_eval:pred__{{pid}}__{fs}__t_us",
                               f"ml_eval:pred__{{pid}}__{fs}__y",
                               f"ml_eval:pred__{{pid}}__{fs}__yhat"])


@register("ml_summary", "dashboard", "Decoding summary", domain="ml",
          choices={"metric": lambda ctx: _metrics(ctx)})
def ml_summary(req: ViewRequest) -> FigureSpec:
    """Cohort median and CI of one decoding metric per configuration and protocol."""
    ctx = req.ctx
    m = req.options.get("metric") or "balanced_accuracy"
    s = ctx.table("ml_eval", "summary") if _has(ctx) else pd.DataFrame()
    s = s[s.metric == m].sort_values(["protocol", "feature_set", "model"]) if not s.empty else s
    lab = [f"{'same subject' if r.protocol == 'blocked' else 'unseen'}: {_fset(r.feature_set)}, "
           f"{V.MODEL_LABEL.get(r.model, r.model)}" for r in s.itertuples()]
    ax = _metric_axis(m)
    p = B.forest(lab, s["median"].to_numpy(float) if not s.empty else [],
                 s.ci_lo.to_numpy(float) if not s.empty else [],
                 s.ci_hi.to_numpy(float) if not s.empty else [], ax.label, ax,
                 ref=1 / 7 if "balanced_accuracy" in m else None,
                 ref_label="chance" if "balanced_accuracy" in m else None)
    if s.empty:
        p.note = "no data"
    return FigureSpec("ml_summary", "Decoding summary", [p], width="text",
                      height_in=max(1.4, 0.15 * len(lab) + 0.6),
                      caption=f"Cohort median (diamond) and 95 % BCa CI over participants of "
                              f"{_metric_axis(m).label.lower()} per feature set and model; "
                              "unseen: "
                              "leave-one-subject-out, same subject: blocked within-subject "
                              "cross-validation.",
                      sources=["ml_eval:summary"])


def _metrics(ctx) -> list[str]:
    s = ctx.table("ml_eval", "summary") if _has(ctx) else pd.DataFrame()
    if s.empty:
        return []
    ms = list(dict.fromkeys(s.metric))
    return sorted(ms, key=lambda m: m != "balanced_accuracy")


@register("ml_permutation", "dashboard", "Permutation test and block importance", domain="ml")
def ml_permutation(req: ViewRequest) -> FigureSpec:
    """Null distribution of the cohort BA and the BA drop when a block is permuted."""
    ctx = req.ctx
    null = _arrays(ctx).get("permutation_null")
    tests = ctx.table("ml_eval", "tests") if _has(ctx) else pd.DataFrame()
    row = tests[tests.test.astype(str).str.startswith("permutation|")] if not tests.empty \
        else tests
    obs = float(row.stat.iloc[0]) if not row.empty else np.nan
    # balanced accuracy cannot be negative: the axis starts at 0 and ends just past the null
    # and the observed value
    top = max(([float(np.nanmax(null))] if null is not None and np.size(null) else [])
              + ([obs] if np.isfinite(obs) else []), default=1.0)
    pa = B.histogram(null if null is not None else [], "(a) Label-shift null",
                     Axis("Cohort mean balanced accuracy", lim=(0.0, min(1.0, 1.25 * top))),
                     observed=obs,
                     observed_label="observed", quantile=0.95)
    hmax = max((float(np.nanmax(sr.y)) for sr in pa.series
                if sr.kind == "bar" and np.size(sr.y) and np.isfinite(sr.y).any()), default=0.0)
    if hmax >= 1:                             # counts: whole-number ticks
        step = max(1, int(np.ceil(hmax / 4)))
        pa.y.ticks = [float(t) for t in range(0, int(hmax) + 1, step)]
    p_txt = (f"; {B.ptext(float(row.p.iloc[0]))} ({int(row.n.iloc[0])} permutations)"
             if not row.empty else "")              # in the caption: a note hits the P95 label
    if row.empty and null is None:
        pa.note = "no permutation test"
    imp = ctx.table("ml_eval", "permutation_importance") if _has(ctx) else pd.DataFrame()
    blocks = sorted(set(imp.block)) if not imp.empty else []
    pb = B.strip_by(imp, "ba_drop", "block", "(b) Block importance", order=blocks,
                    labels={b: _fset(b) for b in blocks},
                    xaxis=Axis("Drop of balanced accuracy"), ref=0.0)
    cfg = (f"{_fset(imp.feature_set.iloc[0])}, {V.MODEL_LABEL.get(imp.model.iloc[0])}"
           if not imp.empty else "")
    tm = row.test.iloc[0].split("|") if not row.empty else []
    tcfg = f"{_fset(tm[2])}, {V.MODEL_LABEL.get(tm[1], tm[1])}" if len(tm) == 3 else ""
    n = imp.pid.nunique() if not imp.empty and "pid" in imp else 0
    spec = FigureSpec("ml_permutation", "Permutation test and block importance", [pa, pb],
                      rows=1, cols=2, width="text", height_in=1.5,
                      sources=["ml_eval:permutation_null", "ml_eval:tests",
                               "ml_eval:permutation_importance"])
    return attach(spec, CaptionSpec(
        f"(a) cohort mean leave-one-subject-out balanced accuracy ({tcfg}) against its null "
        "distribution, each participant's label sequence circularly shifted (keeps the label "
        "autocorrelation), with the 95th percentile. (b) Drop of the held-out balanced "
        f"accuracy when one feature block is permuted ({cfg})",
        [("dots in (b)", "participants")], f"n = {n} participants",
        "(a) the observed cohort mean against the null distribution" + p_txt
        + "; (b) per participant the drop in balanced accuracy",
        test="permutation test (one-sided)",
        # which decoder each panel uses
        notes=["(a) uses the linear discriminant analysis (LDA) decoder of the same features, "
               "refitted for every permutation; (b) the primary support vector machine (SVM) "
               "decoder."],
        evidence="pre-specified"))


@register("ml_variants", "dashboard", "Decoding variants", domain="ml")
def ml_variants(req: ViewRequest) -> FigureSpec:
    """Few-shot calibration, late-fusion weights and the reduced EMG block (exploratory)."""
    ctx = req.ctx
    v = ctx.table("ml_eval", "variants") if _has(ctx) else pd.DataFrame()
    vs = ctx.table("ml_eval", "variants_summary") if _has(ctx) else pd.DataFrame()
    lo = ctx.table("ml_eval", "loso") if _has(ctx) else pd.DataFrame()
    fsv = v[v.variant == "few_shot"] if not v.empty else v
    # seconds per movement, as in jnl_decoding (k windows x the window stride)
    stride = float((ctx.config.params("ml_windows").get("window") or {}).get("stride_s", 0.05))
    pa = Panel("(a) Few-shot calibration", Axis("Calibration data per movement", V.SEC), BA,
               [])
    for i, fs in enumerate(sorted(set(fsv.feature_set)) if not fsv.empty else []):
        g = fsv[fsv.feature_set == fs]
        for _, q in g.groupby("pid"):
            q = q.sort_values("k")
            pa.series.append(Series("line", q.k.to_numpy(float) * stride,
                                    q.balanced_accuracy.to_numpy(float), role="context"))
        s = vs[(vs.variant == "few_shot") & (vs.feature_set == fs)
               & (vs.metric == "balanced_accuracy")].sort_values("k") if not vs.empty else vs
        role = f"set:{fs}" if f"set:{fs}" in _roles() else f"cat:{i}"
        if not s.empty:
            k = s.k.to_numpy(float) * stride
            pa.series += [Series("band", k, s.q1.to_numpy(float), y2=s.q3.to_numpy(float),
                                 role=role),
                          Series("line", k, s["median"].to_numpy(float), role=role,
                                 label=_fset(fs))]
    if not pa.series:
        pa.note = "no few-shot variant"
    lf = v[v.variant == "late_fusion"] if not v.empty else v
    # (b) one dot per fold (a histogram of 13 folds at one weight was a single bar)
    w = lf.w_emg.dropna().round(3) if "w_emg" in lf else pd.Series(dtype=float)
    pb = B.strip(["chosen weight"], [w.to_numpy(float)], "(b) Late fusion: EMG weight",
                 Axis("Weight of the EMG posterior w", lim=(-0.05, 1.05)))
    lf_txt = ""
    if not lf.empty:
        lf_txt = (f"; (b) late fusion: {len(lf)} folds, balanced accuracy median "
                  f"{B.num(lf.balanced_accuracy.median(), 2)}"
                  + (f", every fold chose w = {B.num(float(w.iloc[0]), 2)}"
                     if w.nunique() == 1 else ""))
    else:
        pb.note = "no late-fusion variant"
    rd = v[v.variant == "reduced_emg"] if not v.empty else v
    full = lo[(lo.feature_set == PRIMARY[0]) & (lo.model == PRIMARY[1])].set_index("pid") \
        if not lo.empty else pd.DataFrame()
    pids = [p for p in ctx.pids() if not rd.empty and (rd.pid == p).any()]
    r = rd.set_index("pid").reindex(pids) if not rd.empty else pd.DataFrame(index=pids)
    f = full.reindex(pids) if not full.empty else pd.DataFrame(index=pids)
    pc = B.dumbbell(B.pseudo_rows(ctx, pids),
                    f.get("balanced_accuracy", pd.Series(np.nan, index=pids)).to_numpy(float),
                    r.get("balanced_accuracy", pd.Series(np.nan, index=pids)).to_numpy(float),
                    "(c) Reduced EMG block", BA, label_a="all EMG features",
                    label_b=(V.term("feature_set", rd.feature_set.iloc[0])
                             if not rd.empty else "reduced"))
    if rd.empty:
        pc.note = "no reduced-EMG variant"
    n = v.pid.nunique() if not v.empty and "pid" in v else 0
    spec = FigureSpec("ml_variants", "Decoding variants", [pa, pb, pc], rows=1, cols=3,
                      width="text", height_in=max(1.7, 0.2 * len(pids) + 0.8),
                      sources=["ml_eval:variants", "ml_eval:variants_summary", "ml_eval:loso"])
    return attach(spec, CaptionSpec(
        "decoding variants (SVM, the LOSO folds of the main run, rest-relative features). "
        "(a) Few-shot calibration, balanced accuracy on the held-out participant after "
        "adding its first seconds of every phase to the training set (0: plain LOSO). "
        "(b) Weight of the EMG posterior chosen per fold for late fusion of EMG and IMU "
        "classifiers. (c) EMG + IMU with all EMG features vs a reduced EMG block, per "
        "participant",
        [("grey lines in (a)", "participants"),
         ("line and band in (a)", "median and IQR"),
         ("dots in (b)", "folds (one per held-out participant)")],
        f"n = {n} participants",
        "per participant the held-out balanced accuracy; median and interquartile range over "
        "the participants" + lf_txt, evidence="exploratory"))


# X10 left-out groups (ml_eval variant muscle_ablation), in reading order
ABLATION = {"masseter": "masseter", "temporalis": "temporalis", "digastric": "digastric",
            "digastric_r": "digastric R", "digastric_l": "digastric L"}


def ablation_deltas(ctx) -> pd.DataFrame:
    """Per participant and left-out group (X10): the balanced accuracy without that group
    minus the main ``svm|emg_imu`` run's (columns pid, group, ba, ba_full, delta)."""
    v = ctx.table("ml_eval", "variants") if _has(ctx) else pd.DataFrame()
    lo = ctx.table("ml_eval", "loso") if _has(ctx) else pd.DataFrame()
    if v.empty or lo.empty or "variant" not in v:
        return pd.DataFrame(columns=["pid", "group", "ba", "ba_full", "delta"])
    a = v[v.variant == "muscle_ablation"]
    full = lo[(lo.feature_set == PRIMARY[0]) & (lo.model == PRIMARY[1])]
    d = a.assign(group=a.feature_set.str.removeprefix("emg-").str.removesuffix("+imu"))[
        ["pid", "group", "balanced_accuracy"]].rename(columns={"balanced_accuracy": "ba"})
    d = d.merge(full[["pid", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "ba_full"}), on="pid", how="inner")
    return d.assign(delta=d.ba - d.ba_full)


@register("ml_ablation", "dashboard", "Decoding without one muscle group", domain="ml")
def ml_ablation(req: ViewRequest) -> FigureSpec:
    """X10 (exploratory): what each muscle group adds to EMG + IMU decoding, per participant."""
    ctx = req.ctx
    d = ablation_deltas(ctx)
    order = [g for g in ABLATION if g in set(d.group)] + sorted(set(d.group) - set(ABLATION))
    labels = {g: f"{ABLATION.get(g, g)}, {int((d[d.group == g].delta < 0).sum())}/"
                 f"{int(d[d.group == g].delta.notna().sum())} lower" for g in order}
    pa = B.strip_by(d, "delta", "group", "Without one muscle group", order=order,
                    labels=labels, xaxis=Axis("Change in balanced accuracy"), ref=0.0)
    if d.empty:
        pa.note = "no muscle-ablation variant (X10) in this store"
    n = d.pid.nunique() if not d.empty else 0
    spec = FigureSpec("ml_ablation", "Decoding without one muscle group", [pa], rows=1, cols=1,
                      width="column", height_in=max(1.6, 0.3 * len(order) + 0.7),
                      sources=["ml_eval:variants", "ml_eval:loso"])
    return attach(spec, CaptionSpec(
        "what each muscle group adds to intent decoding given the others: per participant the "
        "held-out balanced accuracy of EMG + IMU (SVM, leave-one-subject-out) with the EMG "
        "features of one group left out, minus that with all six channels (same folds, "
        "training windows, and hyper-parameters). Masseter, temporalis, and digastric rows leave "
        "out both sides; the digastric sides follow the checked electrode montage. Row labels give the "
        "participants whose accuracy dropped, of those with both runs",
        [("dots", "participants"), ("diamond", "median"), ("vertical line", "no change")],
        f"n = {n} participants",
        "per participant the paired difference in balanced accuracy; median over the "
        "participants", evidence="exploratory"))


def _roles() -> dict:
    from biomech.viz.style import ROLES

    return ROLES


@register("ml_fusion", "dashboard", "Feature identity before and after rest-relative normalization", domain="ml")
def ml_fusion(req: ViewRequest) -> FigureSpec:
    """Between/within-participant variance ratio of each EMG feature before and after the
    label-free rest-relative normalization (D13, exploratory)."""
    ctx = req.ctx
    fd = ctx.table("ml_eval", "fusion_diagnostics") if _has(ctx) else pd.DataFrame()
    if not fd.empty:
        order = {m: i for i, m in enumerate(V.MUSCLES)}
        fd = fd.assign(_o=fd.channel.map(order).fillna(99)).sort_values(["_o", "kind",
                                                                          "feature"])
    lab = [f"{V.muscle_label(c)} {V.label(f.split('|', 1)[-1])}" for c, f in
           zip(fd.channel, fd.feature)] if not fd.empty else []
    eps = 1e-6
    a = np.maximum(fd.ratio_before.to_numpy(float), eps) if not fd.empty else []
    b = np.maximum(fd.ratio_after.to_numpy(float), eps) if not fd.empty else []
    p = B.dumbbell(lab, a, b, "Between / within-participant variance",
                   Axis("Variance ratio", log=True), label_a="before rest-relative normalization", label_b="after")
    if fd.empty:
        p.note = "no diagnostics"
    else:
        p.series.append(Series("vline", x=np.array([1.0]), role="reference", label="equal"))
        p.note = f"{int(fd.n_windows.iloc[0])} windows, {int(fd.n_participants.iloc[0])} " \
                 "participants"
    return FigureSpec("ml_fusion", "Feature identity before and after rest-relative normalization", [p], width="text",
                      height_in=max(1.6, 0.11 * len(lab) + 0.8),
                      caption="Per EMG window feature: variance of the participant means "
                              "relative to the mean within-participant variance, before and "
                              "after the label-free rest-relative normalization (post hoc). A "
                              "large ratio means the feature mostly encodes who the "
                              "participant is, which a leave-one-subject-out decoder cannot "
                              "learn (exploratory).",
                      sources=["ml_eval:fusion_diagnostics"])


@register("ml_tests", "dashboard", "Decoding tests (table)", domain="ml")
def ml_tests(req: ViewRequest) -> TableSpec:
    """Friedman, Holm-Wilcoxon and permutation tests of the decoding comparison."""
    ctx = req.ctx
    t = ctx.table("ml_eval", "tests") if _has(ctx) else pd.DataFrame()
    out = B.frame_table(t, "ml_tests", "Feature-set comparison of the primary model "
                        "(Friedman; Wilcoxon of EMG + IMU against each other set, Holm) and "
                        "the label-shift permutation test of the cohort mean balanced "
                        "accuracy (stat: observed mean; null mean and 95th percentile).",
                        ctx=ctx)
    out.sources = ["ml_eval:tests"]
    return out


# reader labels of the stored items (the catalog would compose "Correlation with age vs Balanced
# accuracy, smoothed" from the item key)
GENERALIZATION_ITEM = {"spearman_age_vs_ba": "Correlation of balanced accuracy with age"}


@register("ml_generalization", "dashboard", "Decoding by age and sex (table)", domain="ml")
def ml_generalization(req: ViewRequest) -> TableSpec:
    """Aggregate dependence of the unseen-subject accuracy on age and sex."""
    ctx = req.ctx
    g = ctx.table("ml_eval", "generalization") if _has(ctx) else pd.DataFrame()
    empty = g.drop(columns=["unit"], errors="ignore").empty
    shown = g.assign(item=g["item"].map(GENERALIZATION_ITEM).fillna(g["item"])) if not empty else g
    out = B.frame_table(pd.DataFrame([{"item": "not computed"}]) if empty else shown, "ml_generalization",
                        "Unseen-subject balanced accuracy (EMG + IMU, primary model) against "
                        "age (Spearman) and by sex (median, n); aggregates only, computed with "
                        "at least five participants.", ctx=ctx)
    out.sources = ["ml_eval:generalization", "ml_eval:loso"]   # loso: the per-participant values
    out.notes = "Not computed: fewer than five participants." if empty else ""
    if empty:
        return out
    ns = pd.to_numeric(g.set_index("item")["n"], errors="coerce") if "n" in g else pd.Series()
    nf, nm = (int(ns.get(k)) if np.isfinite(ns.get(k, np.nan)) else 0
              for k in ("median_ba_f", "median_ba_m"))
    return attach(out, CaptionSpec(
        "unseen-subject balanced accuracy of the primary decoder (EMG + IMU) against age "
        "(Spearman correlation) and by sex (median per group); aggregates only, no value per "
        "participant",
        n=f"n = {nf + nm} participants ({nf} female, {nm} male)",
        statistic="Spearman's rank correlation with its p value; group medians",
        evidence="descriptive (no claim; too few participants for an effect of age or sex)",
        notes=[out.notes] if out.notes else []))


@register("ml_windows_table", "dashboard", "Decoding windows (table)", domain="ml")
def ml_windows_table(req: ViewRequest) -> TableSpec:
    """Windows, rejections, references and feature counts per participant."""
    ctx = req.ctx
    meta = ctx.table("ml_windows", "meta")
    refs = ctx.table("ml_windows", "refs")
    feats = ctx.table("ml_windows", "features")
    rows = []
    for pid in ctx.pids():
        m = meta[meta.pid == pid] if not meta.empty else meta
        if m.empty:
            continue
        r = m.iloc[0].to_dict()
        q = refs[refs.pid == pid] if not refs.empty else refs
        src = q.source.value_counts() if not q.empty else pd.Series(dtype=int)
        f = feats[feats.unit == pid] if not feats.empty else feats
        nf = f.groupby("block").size() if not f.empty else pd.Series(dtype=int)
        rows.append({"pid": pid, **{k: r.get(k) for k in ("n_windows", "n_rest_windows",
                                                            "masked", "label", "gap",
                                                            "nonfinite")},
                     # reader words, so the cell needs no vocabulary lookup
                     "normalization": f"{V.term('emg_norm', r.get('emg_norm'), 'short')}; "
                                      f"reference {V.term('emg_norm', r.get('ref_mode'), 'short')}"
                                      f", {V.term('ref_chain', r.get('ref_chain'), 'short')}",
                     "ref_source": ", ".join(f"{k} {v}" for k, v in src.items()),
                     "ref_median_uv": float(q.ref_uv.median()) if not q.empty else np.nan,
                     "features": ", ".join(f"{k} {v}" for k, v in nf.items())})
    df = pd.DataFrame(rows)
    out = B.frame_table(df, "ml_windows_table",
                        "Decoding windows per participant: windows kept and rejected (masked, "
                        "unlabelled, gaps, non-finite), the EMG normalization and the source "
                        "of the causal reference per channel, the median reference (µV) and "
                        "the number of features per block.", ctx=ctx, width="text")
    out.sources = ["ml_windows:meta", "ml_windows:refs", "ml_windows:features"]
    short = {"pid": "", "n_windows": "Windows", "n_rest_windows": "Rest", "masked": "Masked",
             "label": "Unlab.", "gap": "Gap", "nonfinite": "Non-fin.", "normalization": "Normalization",
             "ref_source": "Ref. source", "ref_median_uv": "Ref. (µV)", "features": "Features"}
    for c in out.columns:
        c.header, c.unit = short.get(c.key, c.header), ""
        if c.key in ("normalization", "ref_source", "features"):
            c.align = "X"
        if c.key == "ref_median_uv":
            c.decimals = 0
    return out
