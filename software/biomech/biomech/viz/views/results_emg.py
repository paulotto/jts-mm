# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG time-domain results (stages ``emg`` and ``stats_emg``) as dashboards.

Participant x muscle grids are heatmaps (pseudonym rows, values printed), distributions
over participants are strips (grey dots, accent median, cohort CI where the stats stage
gives one), SPM curves use :func:`blocks.spm_panel`. Channels or participants that an
analysis flags as unusable are shown faint or marked, never dropped silently.
"""

from __future__ import annotations

import re
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.results import COHORT_UNIT

PROFILE_TASKS = ("open_close", "chewing")
NORM_CB = Colorbar("Activation", V.PCT)          # % of max recorded (D11)
INDEX_LABEL = {"asym_masseter_pct": "masseter asymmetry (R − L)",
               "asym_temporalis_pct": "temporalis asymmetry (R − L)",
               "activity_index_pct": "activity index (masseter − temporalis)"}


def _pids(ctx, df: pd.DataFrame) -> list[str]:
    have = set(df.pid) if not df.empty and "pid" in df else set()
    return [p for p in ctx.pids() if p in have]


def _pheat(ctx, df: pd.DataFrame, value: str, title: str, *, fmt: str = "{:.1f}",
           cb: Colorbar | None = None, flag: set[tuple[str, str]] = frozenset(),
           mark: str = "†") -> Panel:
    """Participant x muscle heatmap; cells in ``flag`` ((pid, muscle)) get ``mark``."""
    pids = _pids(ctx, df)
    p = B.matrix_heat(df, "pid", "muscle", value, title, rows=pids,
                      row_labels=B.pseudo_rows(ctx, pids), fmt=fmt, cb=cb)
    if flag and p.series:
        cols = [m for m in V.MUSCLES if m in set(df.muscle)]
        txt = p.series[0].text
        for i, pid in enumerate(pids):
            for j, m in enumerate(cols):
                if (pid, m) in flag and txt[i * len(cols) + j]:
                    txt[i * len(cols) + j] += mark
    return p


def _strip_flagged(labels, ok, flagged, title, xaxis, **kw) -> Panel:
    """:func:`blocks.strip` of the usable values; flagged values as faint dots."""
    p = B.strip(labels, ok, title, xaxis, **kw)
    for i, v in enumerate(flagged):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        if v.size:
            p.series.append(Series("scatter", v, np.full(v.size, float(i)), role="faint"))
    return p


def _ci(stats: pd.DataFrame, key: str, keys: list[str], **sel) -> list[tuple[float, float]]:
    s = stats
    for k, v in sel.items():
        s = s[s[k] == v] if not s.empty else s
    s = s.set_index(key) if not s.empty else s
    return [(float(s.loc[k, "ci_lo"]), float(s.loc[k, "ci_hi"])) if k in s.index
            else (np.nan, np.nan) for k in keys]


# -- participant level --------------------------------------------------------------------------
@register("emg_activation", "dashboard", "Activation per participant", params=("phase",),
          phase="open_close", domain="emg_time", phases=("rest",) + V.TASKS)
def emg_activation(req: ViewRequest) -> FigureSpec:
    """Median and 95th-percentile activation per participant and muscle in one task."""
    ctx, task = req.ctx, req.phase or "open_close"
    a = ctx.table("emg", "activation")
    refs = ctx.table("emg", "refs")
    a = a[a.task == task] if not a.empty else a
    flag = set() if refs.empty else {(r.pid, r.muscle) for r in refs.itertuples()
                                     if not r.usable_amplitude}
    panels = [_pheat(ctx, a, "median_pct", "Median activation", cb=NORM_CB, flag=flag),
              _pheat(ctx, a, "p95_pct", "95th percentile", cb=NORM_CB, flag=flag)]
    n = max(len(_pids(ctx, a)), 1)
    return FigureSpec("emg_activation", "Activation per participant", panels, rows=1, cols=2,
                      width="text", height_in=max(1.4, 0.22 * n + 0.6),
                      caption=f"{V.task_label(task)}: median and 95th percentile of the "
                              f"normalized EMG envelope ({V.EMG_NORM_SHORT}) per "
                              "participant and muscle over the whole task; † channel not "
                              "amplitude-usable (excluded from the cohort statistics).",
                      sources=["emg:activation", "emg:refs"])


@register("emg_cycles", "dashboard", "Activation per cycle", params=("phase",),
          phase="open_close", domain="emg_time", phases=PROFILE_TASKS)
def emg_cycles(req: ViewRequest) -> FigureSpec:
    """Per-cycle mean activation, masked share and co-contraction; cycle profiles."""
    ctx, task = req.ctx, req.phase or "open_close"
    c = ctx.table("emg", "cycles")
    c = c[c.task == task] if not c.empty else c
    ms = [m for m in V.MUSCLES if f"mean_pct_{m}" in c]
    lab = [V.muscle_label(m) for m in ms]
    pa = B.strip(lab, [c[f"mean_pct_{m}"] for m in ms], "Cycle mean activation",
                 Axis("Activation", V.PCT))
    pm = B.strip(lab, [100 * c[f"masked_{m}"] for m in ms], "Masked share per cycle",
                 Axis("Masked", V.PCT))
    pids = _pids(ctx, c)
    rows = B.pseudo_rows(ctx, pids)
    pb, pc = (B.strip(rows, [c.loc[c.pid == p, col] for p in pids], title,
                      Axis("CCI", V.PCT, lim=(0.0, 100.0)))
              for col, title in (("cci_opening", "Co-contraction, opening half"),
                                 ("cci_closing", "Co-contraction, closing half")))
    # Cycle-normalized profiles: participant means, cohort mean per muscle.
    x = np.linspace(0, 100, 101)
    pp = Panel("Activation profile (cohort mean)", Axis("Cycle", V.PCT),
               Axis("Activation", V.PCT), [], sharex="cyc")
    seed = B.seed_of(ctx)
    pt = Panel("Opening angle", Axis("Cycle", V.PCT), Axis("Angle", V.DEG), [], legend="none",
               sharex="cyc")
    means: dict[str, list[np.ndarray]] = {m: [] for m in V.MUSCLES}
    th, n_cyc = [], 0
    for pid in ctx.pids():
        a = ctx.arrays("emg", pid)
        P = a.get(f"profile__{task}__pct")
        if P is None:
            continue
        names = [str(n) for n in a.get("profile__names", [])]
        n_cyc += len(a.get(f"profile__{task}__cycle", []))
        with _quiet():
            for m in V.MUSCLES:
                if m in names:
                    means[m].append(np.nanmean(P[:, :, names.index(m)], axis=0))
            T = a.get(f"profile__{task}__theta_deg")
            if T is not None and len(T):
                th.append(np.nanmean(np.asarray(T, float), axis=0))
    for m in V.MUSCLES:                      # CI bands first, every mean line above them
        if means[m]:
            with _quiet():
                c = np.nanmean(np.array(means[m]), axis=0)
            B.add_ci_band(pp, x, np.array(means[m]), role=f"entity:{m}", seed=seed,
                          where=np.isfinite(c))
    for m in V.MUSCLES:
        if means[m]:
            with _quiet():
                pp.series.append(Series("line", x, np.nanmean(np.array(means[m]), axis=0),
                                        role=f"entity:{m}", label=V.muscle_label(m)))
    if th:
        B.add_curves_ci(pt, x, np.array(th), seed=seed)
    pp.note = f"{n_cyc} cycles" if n_cyc else "no profiles"
    for p in (pa, pm, pb, pc, pt):
        if not p.series:
            p.note = "no cycles"
    # n-aware: the participant rows (bottom) set the height; the muscle rows (top) their own.
    h = B.rows_height(len(pids), per=0.2, lo=1.5)
    for p in (pa, pm, pp):
        p.height = B.rows_height(len(ms), per=0.2, lo=1.5) / h
    return FigureSpec("emg_cycles", "Activation per cycle", [pa, pm, pp, pb, pc, pt], rows=2,
                      cols=3, width="text", height_in=h,
                      caption=f"{V.task_label(task)}: mean normalized activation "
                              f"({V.EMG_NORM_SHORT}) and masked share of every kinematic cycle "
                              "(channel-cycles masked above the cycle limit have no mean); "
                              "cycle-normalized activation profiles (mean of the participant "
                              "means, shape-usable or not; bands: 95 % bootstrap CI of the "
                              f"mean over the {len(pids)} participants); opener/closer "
                              "co-contraction index "
                              "(CCI) of the opening and closing half of each cycle, "
                              "shape-usable channels only; opening angle over the same cycles "
                              "(participants grey, mean; " + B.curve_ci_caption() + "). "
                              "Dots: cycles; diamond: median.",
                      sources=["emg:cycles", "emg:profile__{task}__pct",
                               "emg:profile__{task}__theta_deg"])


@contextmanager
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@register("emg_refs", "dashboard", "Normalization references", domain="emg_time")
def emg_refs(req: ViewRequest) -> FigureSpec:
    """Where each channel's normalization reference comes from (D11)."""
    ctx = req.ctx
    r = ctx.table("emg", "refs")
    task_ref = set() if r.empty else {(x.pid, x.muscle) for x in r.itertuples()
                                      if x.ref_source != "mvc"}
    pa = _pheat(ctx, r, "mvc_over_task", "MVC / task maximum", fmt="{:.2f}", flag=task_ref,
                mark=" T")
    pb = _pheat(ctx, r, "ref_uv", "Normalization reference (µV)", fmt="{:.0f}")
    n = max(len(_pids(ctx, r)), 1)
    return FigureSpec("emg_refs", "Normalization references", [pa, pb], rows=1, cols=2,
                      width="text", height_in=max(1.4, 0.22 * n + 0.6),
                      caption="Normalization reference per channel (the largest masked "
                              "150-ms envelope recorded, MVC/RVC or any task). (a) Largest "
                              "envelope in the reference contraction divided by the largest "
                              "in the tasks; T: a task exceeded the reference contraction, "
                              "which then is not the maximum (reference source 'task'). "
                              "(b) The reference value.",
                      sources=["emg:refs"])


def _compact(t: TableSpec, headers: dict[str, tuple[str, str]], wrap: tuple[str, ...] = (),
             decimals: dict[str, int] | None = None) -> TableSpec:
    """Short headers (header, unit), decimals and wrapping text columns, so wide tables
    fit the text width."""
    for c in t.columns:
        if c.key in headers:
            c.header, c.unit = headers[c.key]
        if c.key in wrap:
            c.align = "X"
        c.decimals = (decimals or {}).get(c.key, c.decimals)
    return t


@register("emg_refs_table", "dashboard", "Normalization references (table)",
          domain="emg_time")
def emg_refs_table(req: ViewRequest) -> TableSpec:
    """Every column of the reference provenance table."""
    r = req.ctx.table("emg", "refs")
    if not r.empty:
        order = {p: i for i, p in enumerate(req.ctx.pids())}
        r = r.assign(_o=r.pid.map(order), _m=r.muscle.map({m: i for i, m in
                                                            enumerate(V.MUSCLES)}))
        r = r.sort_values(["_o", "_m"]).drop(columns=["_o", "_m"])
        r = r.assign(muscle=r.muscle.map(V.muscle_label),
                     task_max_phase=r.task_max_phase.map(V.task_label),
                     **{c: r[c].map({True: "yes", False: "no"})
                        for c in ("usable_amplitude", "usable_shape")})
    t = B.frame_table(r, "emg_refs_table", "EMG normalization references per channel: "
                      "reference phase, its p95 and largest 1-s mean, masked share, the "
                      "normalization reference (largest recorded) and its source, the largest "
                      "envelope in the reference contraction and in the tasks (and which "
                      "task), their ratio, and the amplitude/shape usability flags. "
                      "Amplitudes in µV.", ctx=req.ctx, width="text")
    t.sources = ["emg:refs"]
    return _compact(t, {"pid": ("", ""), "muscle": ("Muscle", ""), "ref_phase": ("Phase", ""),
                        "p95_uv": ("p95", ""), "max1s_uv": ("1 s", ""),
                        "mask_frac": ("Masked", ""), "ref_uv": ("Ref.", ""),
                        "ref_source": ("Source", ""), "mvc_max_uv": ("MVC", ""),
                        "task_max_uv": ("Task", ""), "task_max_phase": ("in", ""),
                        "mvc_over_task": ("MVC/task", ""), "usable_amplitude": ("Amp.", ""),
                        "usable_shape": ("Shape", "")},
                    decimals={"p95_uv": 0, "max1s_uv": 0, "ref_uv": 0, "mvc_max_uv": 0,
                              "task_max_uv": 0, "mask_frac": 2, "mvc_over_task": 2})


@register("emg_mask_sensitivity", "dashboard", "Artifact-mask sensitivity", params=("phase",),
          phase="open_close", domain="emg_time", phases=V.TASKS)
def emg_mask_sensitivity(req: ViewRequest) -> FigureSpec:
    """Cycles kept, masked share and activation under the primary and alternative mask
    rules (D14)."""
    ctx, task = req.ctx, req.phase or "open_close"
    s = ctx.table("emg", "mask_sensitivity")
    s = s[s.task == task] if not s.empty else s
    ms = [m for m in V.MUSCLES if not s.empty and (s.muscle == m).any()]
    lab = [V.muscle_label(m) for m in ms]
    rules = s.rules.iloc[0] if not s.empty else ""
    alt = s.alt_rules.iloc[0] if not s.empty else ""
    added = sorted(set(alt.split(",")) - set(rules.split(",")))
    removed = sorted(set(rules.split(",")) - set(alt.split(",")))
    alt_lab = "alternative (" + ", ".join([f"+{r}" for r in added]
                                          + [f"−{r}" for r in removed]) + ")"
    g = s.groupby("muscle") if not s.empty else None

    def agg(col: str, how: str) -> np.ndarray:
        if g is None:
            return np.empty(0)
        v = g[col].sum() if how == "sum" else g[col].median()
        return v.reindex(ms).to_numpy(float)

    n = agg("n_cycles", "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        ka, kb = 100 * agg("kept", "sum") / n, 100 * agg("kept_alt", "sum") / n
    pa = B.dumbbell(lab, ka, kb, "Cycles kept", Axis("Cycles kept", V.PCT),
                    label_a="primary", label_b=alt_lab)
    if not np.isfinite(ka).any():
        pa.note = "no kinematic cycles in this task"
    else:
        pa.note = f"{int(np.nansum(n) / max(len(ms), 1))} channel-cycles per muscle"
    seed = B.seed_of(req.ctx)
    col = (lambda c: [100 * s.loc[s.muscle == m, c].to_numpy(float) for m in ms]) \
        if not s.empty else (lambda c: [])
    pb = B.paired_strip(lab, col("mask_frac"), col("mask_frac_alt"), "Masked samples",
                        Axis("Masked", V.PCT), label_a="primary", label_b=alt_lab, seed=seed)
    d = s.assign(d=s.median_pct_alt - s.median_pct) if not s.empty else s
    pc = B.strip_by_ci(d, "d", "muscle", "Activation shift (alternative − primary)", order=ms,
                       xaxis=Axis("Δ median activation", V.PCT), ref=0.0, seed=seed)
    n_p = int(s.pid.nunique()) if not s.empty else 0
    return FigureSpec("emg_mask_sensitivity", "Artifact-mask sensitivity", [pa, pb, pc],
                      rows=1, cols=3, width="text",
                      height_in=B.rows_height(len(ms), per=0.3, extra=0.7, lo=1.5),
                      caption=f"{V.task_label(task)}: artifact masks of the analysis "
                              f"(primary: {rules.replace(',', ', ')}) against the sensitivity "
                              f"counterpart ({alt.replace(',', ', ')}). (a) Share of "
                              "channel-cycles kept (masked share at most the cycle limit), "
                              f"all {n_p} participants pooled; (b) masked share of each "
                              "participant's channel (small dots) under each rule; (c) "
                              "change of each participant's median activation "
                              f"({V.EMG_NORM_SHORT}, each variant with its own reference). "
                              f"Diamond: median, {B.MEDIAN_CI_CAPTION}.",
                      sources=["emg:mask_sensitivity"])


def _runs(mask: np.ndarray, t: np.ndarray, gap_s: float = 0.2, cap: int = 150
          ) -> list[tuple[float, float]]:
    """Contiguous masked runs as (start, end) times, gaps < ``gap_s`` merged."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
    out: list[list[float]] = []
    for a, b in zip(t[starts], t[ends]):
        if out and a - out[-1][1] < gap_s:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out[:cap]]


@register("emg_envelope", "dashboard", "EMG envelope and mask", params=("pid", "phase"),
          phase="open_close", domain="emg_time", phases=("rest",) + V.TASKS,
          choices={"units": ("uv", "norm")}, units="uv")
def emg_envelope(req: ViewRequest) -> FigureSpec:
    """One participant's envelope per muscle over a task, masked stretches shaded."""
    ctx, task = req.ctx, req.phase or "open_close"
    pid = req.pid if req.pid in ctx.pids() else (ctx.pids() or [None])[0]
    units = req.options.get("units", "uv")
    a = ctx.arrays("emg", pid) if pid else {}
    t = a.get(f"{task}__t_us")
    y = a.get(f"{task}__env_uv" if units == "uv" else f"{task}__norm_pct")
    mk = a.get(f"{task}__masked")
    names = [str(n) for n in a.get("profile__names", [])]
    if not names and pid:
        r = ctx.table("emg", "refs")
        names = list(r[r.pid == pid].muscle) if not r.empty else []
    yax = Axis("Envelope", V.UV) if units == "uv" else Axis("Activation", V.PCT)
    panels = []
    ts = (np.asarray(t, float) - float(t[0])) / 1e6 if t is not None and len(t) else None
    for m in V.MUSCLES:
        p = Panel(V.muscle_label(m), Axis("Time in task", V.SEC), yax, [], legend="none",
                  sharex="t")
        if ts is None or y is None or m not in names:
            p.note = "no data"
        else:
            i = names.index(m)
            if mk is not None:
                for s0, s1 in _runs(mk[:, i], ts):
                    p.series.append(Series("hspan", x=np.array([s0]), y2=np.array([s1]),
                                           role="faint"))
                p.note = f"masked {100 * float(np.mean(mk[:, i])):.1f} %"
            p.series.append(Series("line", ts, np.asarray(y[:, i], float), role="primary"))
        panels.append(p)
    q = Panel("Channels masked", Axis("Time in task", V.SEC), Axis("Share", V.PCT, lim=(0, 100)),
              [], legend="none", sharex="t")
    if ts is not None and mk is not None:
        q.series.append(Series("line", ts, 100 * np.asarray(mk, float).mean(axis=1),
                               role="accent"))
    else:
        q.note = "no data"
    panels.append(q)
    who = f"P{ctx.pseudonym(pid)}" if pid else "no participant"
    return FigureSpec("emg_envelope", "EMG envelope and mask", panels, rows=len(panels), cols=1,
                      width="text", height_in=0.75,
                      caption=f"{who}, {V.task_label(task)}: "
                              + ("envelope of the cleaned EMG (µV)" if units == "uv" else
                                 f"normalized envelope ({V.EMG_NORM_SHORT})")
                              + " per muscle, resampled to 100 Hz; grey: samples masked by "
                                "the analysis artifact rules (rail, lead-off, envelope); "
                                "bottom: share of the six channels masked.",
                      sources=["emg:{task}__env_uv", "emg:{task}__norm_pct", "emg:{task}__masked"])


# -- cohort level ------------------------------------------------------------------------------
@register("emg_indices", "dashboard", "Asymmetry and activity indices", domain="emg_time")
def emg_indices(req: ViewRequest) -> FigureSpec:
    """Clench asymmetry and activity index per participant with the cohort CI."""
    ctx = req.ctx
    ind = ctx.table("emg", "indices")
    st = ctx.table("stats_emg", "indices")
    keys = list(INDEX_LABEL)
    ok = ind[ind.usable.astype(bool)] if not ind.empty else ind
    bad = ind[~ind.usable.astype(bool)] if not ind.empty else ind
    p = _strip_flagged([INDEX_LABEL[k] for k in keys],
                       [ok[k] if k in ok else [] for k in keys],
                       [bad[k] if k in bad else [] for k in keys],
                       "Clench indices", Axis("Index", V.PCT), ref=0.0,
                       intervals=_ci(st, "metric", keys))
    n_bad = int(len(bad))
    return FigureSpec("emg_indices", "Asymmetry and activity indices", [p], width="column",
                      height_in=1.4,
                      caption="Asymmetry (R − L)/(R + L) of masseter and temporalis and the "
                              "activity index (masseter − temporalis)/(masseter + temporalis) "
                              "of the clench references (p95). Dots: participants, diamond: "
                              "median, line: 95 % BCa CI of the mean (usable participants); "
                              f"light grey dots: not usable ({n_bad}; a closer's maximum came from a "
                              "task).",
                      sources=["emg:indices", "stats_emg:indices"])


@register("emg_rest_activity", "dashboard", "Rest activity", domain="emg_time")
def emg_rest_activity(req: ViewRequest) -> FigureSpec:
    """Rest RMS (µV) and rest activation (% of max recorded) per muscle."""
    ctx = req.ctx
    a = ctx.table("emg", "activation")
    a = a[a.task == "rest"] if not a.empty else a
    refs = ctx.table("emg", "refs")
    st = ctx.table("stats_emg", "rest_activity")
    if not a.empty and not refs.empty:
        a = a.merge(refs[["pid", "muscle", "usable_amplitude"]], on=["pid", "muscle"],
                    how="left")
    use = a.usable_amplitude.fillna(False).astype(bool) if "usable_amplitude" in a else \
        pd.Series(True, index=a.index)
    ms = [m for m in V.MUSCLES if not a.empty and (a.muscle == m).any()]
    lab = [V.muscle_label(m) for m in ms]
    pa = B.strip(lab, [a.loc[a.muscle == m, "rest_rms_uv"] for m in ms], "Rest RMS",
                 V.axis("rest_rms_uv"), intervals=_ci(st, "muscle", ms, metric="rest_rms_uv"))
    pb = _strip_flagged(lab, [a.loc[(a.muscle == m) & use, "median_pct"] for m in ms],
                        [a.loc[(a.muscle == m) & ~use, "median_pct"] for m in ms],
                        "Rest activation", Axis("Activation", V.PCT),
                        intervals=_ci(st, "muscle", ms, metric="rest_pct"))
    return FigureSpec("emg_rest_activity", "Rest activity", [pa, pb], rows=1, cols=2,
                      width="text", height_in=max(1.4, 0.2 * len(ms) + 0.5),
                      caption="Activity at rest per muscle: RMS of the cleaned EMG and the "
                              f"median normalized envelope ({V.EMG_NORM_SHORT}). Dots: "
                              "participants, diamond: median, line: 95 % BCa CI of the "
                              "median; light grey dots: channels not amplitude-usable (not in the "
                              "cohort statistics).",
                      sources=["emg:activation", "emg:refs", "stats_emg:rest_activity"])


@register("emg_cci", "dashboard", "Co-contraction by task", domain="emg_time")
def emg_cci(req: ViewRequest) -> FigureSpec:
    """Participant-median co-contraction index per task with the cohort CI."""
    ctx = req.ctx
    c = ctx.table("emg", "cycles")
    st = ctx.table("stats_emg", "cci")
    per = (c.groupby(["pid", "task"])[["cci_opening", "cci_closing"]].median().reset_index()
           if not c.empty else c)
    tasks = [t for t in PROFILE_TASKS if not per.empty and (per.task == t).any()]
    panels = [B.strip_by(per, col, "task", title, order=tasks, xaxis=Axis("CCI", V.PCT),
                         intervals=_ci(st, "task", tasks, metric=col))
              for col, title in (("cci_opening", "Opening half"),
                                 ("cci_closing", "Closing half"))]
    return FigureSpec("emg_cci", "Co-contraction by task", panels, rows=1, cols=2,
                      width="text", height_in=max(1.2, 0.3 * len(tasks) + 0.6),
                      caption="Opener/closer co-contraction index (overlap of the normalized "
                              "group envelopes, 0-100 %) per participant (median over "
                              "cycles); diamond: median, line: 95 % BCa CI of the mean.",
                      sources=["emg:cycles", "stats_emg:cci"])


def _term(t: str) -> str:
    parts = re.findall(r"C\((\w+)\)\[T\.([^\]]+)\]", t)
    if not parts:
        return t
    return " × ".join(V.task_label(v) if k == "task" else V.muscle_label(v) for k, v in parts)


@register("emg_activation_lmm", "dashboard", "Activation mixed model", domain="emg_time")
def emg_activation_lmm(req: ViewRequest) -> FigureSpec:
    """Fixed effects of the task x muscle mixed model on log activation."""
    ctx = req.ctx
    lm = ctx.table("stats_emg", "activation_lmm")
    p = Panel("Fixed effects on log activation", Axis("Coefficient (log ratio)"), Axis(""),
              legend="none", note="no model (fewer than 5 participants)")
    cap = ""
    if not lm.empty and "term" in lm:
        f = lm[lm.term != "Intercept"].reset_index(drop=True)
        p = B.forest([_term(t) for t in f.term], f.coef.to_numpy(float),
                     f.ci_lo.to_numpy(float), f.ci_hi.to_numpy(float),
                     "Fixed effects on log activation", Axis("Coefficient (log ratio)"))
        lv = {k: sorted({v for t in lm.term for kk, v in re.findall(
            r"C\((\w+)\)\[T\.([^\]]+)\]", t) if kk == k}) for k in ("task", "muscle")}
        act = ctx.table("emg", "activation")
        ref = []
        for k, lab in (("task", V.task_label), ("muscle", V.muscle_label)):
            all_ = sorted(set(act[k]) - {"rest"}) if not act.empty else []
            ref += [lab(v) for v in all_ if v not in lv[k]][:1]
        icpt = lm[lm.term == "Intercept"]
        cap = (f" Reference levels: {', '.join(ref)}" if ref else "") + (
            f"; intercept {float(icpt.coef.iloc[0]):.2f}" if not icpt.empty else "") + (
            f"; optimizer {lm.optimizer.iloc[0]}" if "optimizer" in lm else "") + "."
    n = max(len(p.y.ticks or []), 1)
    return FigureSpec("emg_activation_lmm", "Activation mixed model", [p], width="column",
                      height_in=max(1.4, 0.13 * n + 0.6),
                      caption="Linear mixed model log(median activation) ~ task × muscle, "
                              "random intercept per participant (REML), amplitude-usable "
                              "channels, rest excluded: fixed-effect estimates with 95 % "
                              "Wald CIs (exp(coef) = activation ratio)." + cap,
                      sources=["stats_emg:activation_lmm", "emg:activation"])


SPM_TESTS = {"masseter_L_vs_R_chewing": ("spm1", "Masseter L vs R, chewing",
                                         ("left", "right"), ("masseter L", "masseter R"),
                                         Axis("Activation", V.PCT)),
             "openers_vs_closers_open_close": ("spm2", "Openers vs closers, open-close",
                                               ("openers", "closers"), ("openers", "closers"),
                                               Axis("Activation (peak = 1)"))}


def _clusters(s: str) -> list[tuple[float, float]]:
    return [(a, b) for a, b, _ in B.parse_clusters(s)]


@register("emg_spm", "dashboard", "SPM tests", domain="emg_time",
          choices={"test": tuple(SPM_TESTS)})
def emg_spm(req: ViewRequest) -> FigureSpec:
    """A confirmatory SPM{t} curve with its threshold and clusters, and its inputs."""
    ctx = req.ctx
    test = req.options.get("test") or next(iter(SPM_TESTS))
    key, title, (ka, kb), (la, lb), yax = SPM_TESTS[test]
    arr = ctx.arrays("stats_emg", COHORT_UNIT)
    spm = ctx.table("stats_emg", "spm")
    row = spm[spm.test == test] if not spm.empty else spm
    z = np.asarray(arr.get(f"{key}__z", []), float)
    zstar = float(row.zstar.iloc[0]) if not row.empty and pd.notna(row.zstar.iloc[0]) else np.nan
    cl = _clusters(row.clusters.iloc[0]) if not row.empty else []
    if z.size:
        pa = B.spm_panel(np.linspace(0, 100, z.size), z, zstar, "SPM{t}", clusters=cl)
        pa.legend = "none"
        if not row.empty and "p_holm" in row and pd.notna(row.p_holm.iloc[0]):
            pa.note = f"p (Holm) = {float(row.p_holm.iloc[0]):.3f}"
    else:
        n = int(row.n.iloc[0]) if not row.empty and pd.notna(row.n.iloc[0]) else 0
        pa = Panel("SPM{t}", Axis("Cycle", V.PCT), Axis("SPM{t}"), [], legend="none",
                   note=f"test not computed (n = {n})")
    pb = Panel("Test inputs (participant means)", Axis("Cycle", V.PCT), yax, [])
    for k, lab, role in ((ka, la, "cat:0"), (kb, lb, "cat:1")):
        A = np.asarray(arr.get(f"{key}__{k}", np.empty((0, 0))), float)
        if A.ndim == 2 and A.size:
            B.add_curves_ci(pb, np.linspace(0, 100, A.shape[1]), A, label=lab, role=role,
                            show_subjects=False, seed=B.seed_of(ctx))
    B.add_spm_bars(pb, [(title.split(",")[0], [(a, b) for a, b in cl], "context")])
    if not pb.series:
        pb.note = "no data"
    n_in = int(np.asarray(arr.get(f"{key}__{ka}", np.empty((0, 0)))).shape[0]) \
        if np.ndim(arr.get(f"{key}__{ka}", [])) == 2 else 0
    return FigureSpec("emg_spm", "SPM tests", [pa, pb], rows=1, cols=2, width="text",
                      height_in=1.6,
                      caption=f"{title} (confirmatory, paired SPM{{t}} on participant means, "
                              "two-tailed, Holm over the two tests): the test statistic over "
                              "the cycle with the critical threshold and supra-threshold "
                              "clusters (grey), and the two compared curves (mean over the "
                              f"{n_in} participants; " + B.curve_ci_caption() + ").",
                      sources=["stats_emg:spm", f"stats_emg:{key}__z"])


@register("emg_spm_table", "dashboard", "SPM and variability tests (table)", domain="emg_time")
def emg_spm_table(req: ViewRequest) -> TableSpec:
    """Every row of the SPM table: confirmatory, variability, single-participant, CMC."""
    ctx = req.ctx
    spm = ctx.table("stats_emg", "spm")
    pids = set(ctx.pids())
    cm = cmc_ci(ctx, seed=B.seed_of(ctx)) if not spm.empty else None
    if cm is not None and "test" in spm:            # CMC with its bootstrap CI (R5 E)
        is_cmc = spm.test == "cmc_opening_angle"
        ci = lambda v: f"[{v[1]:.2f}, {v[2]:.2f}]" if np.isfinite(v[1]) else ""  # noqa: E731
        spm = spm.assign(cmc_within_ci=np.where(is_cmc, ci(cm["median_within"]), ""),
                         cmc_between_ci=np.where(is_cmc, ci(cm["between"]), ""))
    if not spm.empty:
        spm = spm.assign(test=[re.sub(r"\|([^|]+)$", lambda m: f"|P{ctx.pseudonym(m[1])}"
                                      if m[1] in pids else m[0], t).replace("_", " ")
                               .replace("|", " | ") for t in spm.test],
                         family=spm.family.str.replace("_", " "))
    t = B.frame_table(spm, "emg_spm_table",
                      "SPM tests on the EMG and angle waveforms: confirmatory paired tests "
                      "(Holm), participants ANOVA (variability), single-participant masseter "
                      "L vs R tests (non-parametric) and the coefficient of multiple "
                      "correlation (CMC) of the opening angle"
                      + (" with 95 % bootstrap CIs (cycles resampled within participants, "
                         f"{cm['n_boot']} resamples)" if cm is not None else "")
                      + ". Clusters: (start %, end %, p), "
                      "for the ANOVAs their number; supra: share of the cycle above the "
                      "threshold.", ctx=ctx, width="text")
    t.sources = ["stats_emg:spm"]
    return _compact(t, {"family": ("Family", ""), "test": ("Test", ""), "h0reject": ("H0 rej.", ""),
                        "zstar": ("z*", ""), "clusters": ("Clusters", ""),
                        "p_min_cluster": ("p min", ""), "p_holm": ("p Holm", ""),
                        "supra_frac": ("Supra", ""), "n_significant": ("n sig.", ""),
                        "cmc_within_median": ("CMC within", ""),
                        "cmc_between": ("CMC between", ""),
                        "cmc_within_ci": ("CI", ""), "cmc_between_ci": ("CI", "")},
                    wrap=("test", "clusters"))


# -- variability: bootstrap CIs from the stored per-cycle profiles (R5 E) ----------------------
def variability_inputs(ctx) -> dict[str, dict[str, np.ndarray]]:
    """The per-participant cycle profiles the variability analysis of ``stats_emg`` uses
    (stages.stats_emg -> cohort_emg.spm_tests): the open-close opening angle and the
    right-masseter chewing activation of shape-usable channels, cycles x 101."""
    refs = ctx.table("emg", "refs")
    ok = {} if refs.empty else {p: set(g.muscle[g.usable_shape.astype(bool)])
                                for p, g in refs.groupby("pid")}
    theta, mas = {}, {}
    for pid in ctx.pids():
        a = ctx.arrays("emg", pid)
        if "profile__open_close__theta_deg" in a:
            theta[pid] = np.asarray(a["profile__open_close__theta_deg"], float)
        names = [str(n) for n in a.get("profile__names", V.MUSCLES)]
        P = a.get("profile__chewing__pct")
        if P is not None and "masseter_r" in names and "masseter_r" in ok.get(pid, set()):
            mas[pid] = np.asarray(P, float)[:, :, names.index("masseter_r")]
    return {"theta": theta, "masseter_r_chewing": mas}


def varratio_ci(ctx, key: str, stored: np.ndarray | None, *, seed: int = 0,
                n_boot: int = B.N_BOOT_CURVE, inputs: dict | None = None
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """Pointwise 95 % bootstrap CI (participants resampled) of the stored between/within
    variance ratio ``varratio__<key>__ratio``; None when the stored curve cannot be
    reproduced from the stored profiles (then no band is drawn)."""
    from biomech.stats.waveforms import _clean_rows, variance_ratio

    if stored is None or not np.size(stored):
        return None
    prof = (inputs or variability_inputs(ctx)).get(key, {})
    vr = variance_ratio(prof)
    if not vr or len(vr["ratio"]) != len(stored) or not np.allclose(
            vr["ratio"], np.asarray(stored, float), rtol=1e-9, equal_nan=True):
        return None
    P = [v for v in (_clean_rows(x) for x in prof.values()) if len(v) >= 3]
    M = np.stack([v.mean(0) for v in P])
    W = np.stack([v.var(0, ddof=1) for v in P])
    idx = np.random.default_rng(seed).integers(0, len(P), (n_boot, len(P)))
    r = M[idx].var(axis=1, ddof=1) / np.maximum(W[idx].mean(axis=1), 1e-12)
    lo, hi = np.percentile(r, [2.5, 97.5], axis=0)
    return lo, hi


def _cmc_batch(Y: np.ndarray) -> np.ndarray:
    """CMC (Kadaba 1989, as stats.waveforms.cmc) of a batch of curve sets (..., m, n)."""
    m, n = Y.shape[-2:]
    mt = Y.mean(axis=-2, keepdims=True)
    gr = Y.mean(axis=(-2, -1), keepdims=True)
    num = ((Y - mt) ** 2).sum(axis=(-2, -1)) / (n * (m - 1))
    den = ((Y - gr) ** 2).sum(axis=(-2, -1)) / (m * n - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        val = 1.0 - num / np.where(den > 0, den, np.nan)
        return np.where(val >= 0, np.sqrt(np.clip(val, 0.0, None)), np.nan)


def cmc_ci(ctx, *, seed: int = 0, n_boot: int = B.N_BOOT_CURVE, inputs: dict | None = None
           ) -> dict | None:
    """CMC of the open-close opening angle with 95 % bootstrap CIs (cycles resampled within
    each participant): per participant (within), the median of those and the CMC of the
    participant means (between). The point estimates are the stored ones
    (``stats_emg.spm`` row ``cmc_opening_angle``); None when they cannot be reproduced."""
    from biomech.stats.waveforms import _clean_rows, cmc

    theta = (inputs or variability_inputs(ctx))["theta"]
    spm = ctx.table("stats_emg", "spm")
    row = spm[spm.test == "cmc_opening_angle"] if not spm.empty and "test" in spm else spm
    if row.empty or not theta:
        return None
    within = {pid: cmc(P) for pid, P in theta.items()}
    med = float(np.nanmedian(list(within.values())))
    between = cmc(np.array([np.nanmean(P, 0) for P in theta.values()]))
    if not (np.isclose(med, float(row.cmc_within_median.iloc[0]), equal_nan=True)
            and np.isclose(between, float(row.cmc_between.iloc[0]), equal_nan=True)):
        return None
    rng = np.random.default_rng(seed)
    wb, means = {}, []
    for pid, P in theta.items():
        Y = _clean_rows(P)
        if len(Y) < 2:
            wb[pid] = np.full(n_boot, np.nan)
            continue
        Yb = Y[rng.integers(0, len(Y), (n_boot, len(Y)))]
        wb[pid] = _cmc_batch(Yb)
        means.append(Yb.mean(axis=1))
    q = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).any() \
        else (np.nan, np.nan)                                            # noqa: E731
    with _quiet():
        med_b = np.nanmedian(np.stack(list(wb.values())), axis=0)
    betw_b = _cmc_batch(np.stack(means, axis=1)) if len(means) >= 2 else np.full(n_boot, np.nan)
    return {"within": {pid: (within[pid], *q(wb[pid])) for pid in theta},
            "median_within": (med, *q(med_b)), "between": (between, *q(betw_b)),
            "n_boot": n_boot}


@register("emg_varratio", "dashboard", "Between- vs within-participant variance",
          domain="emg_time")
def emg_varratio(req: ViewRequest) -> FigureSpec:
    """Pointwise between/within-participant variance of angle and masseter activation, and
    the CMC of the opening angle, with bootstrap CIs."""
    ctx = req.ctx
    seed = B.seed_of(ctx)
    arr = ctx.arrays("stats_emg", COHORT_UNIT)
    spm = ctx.table("stats_emg", "spm")
    an = {r.test: r.clusters for r in spm.itertuples()} if not spm.empty else {}
    tests = {"theta": "anova1_opening_angle_open_close",
             "masseter_r_chewing": "anova1_masseter_r_chewing"}
    inputs = variability_inputs(ctx)
    panels = []
    n_bars, n_ci = 0, 0
    for key, lab, unit in (("theta", "Opening angle, open-close", "°"),
                           ("masseter_r_chewing", "Masseter R, chewing", V.EMG_NORM_SHORT)):
        pv = Panel(f"{lab}: variance", Axis("Cycle", V.PCT), Axis(f"Variance ({unit})²"),
                   [], sharex="c")
        pr = Panel(f"{lab}: ratio", Axis("Cycle", V.PCT), Axis("Between / within", log=True),
                   [], legend="none", sharex="c")
        for part, role in (("between", "cat:0"), ("within", "cat:1")):
            v = arr.get(f"varratio__{key}__{part}")
            if v is not None and np.size(v):
                pv.series.append(Series("line", np.linspace(0, 100, len(v)),
                                        np.asarray(v, float), role=role,
                                        label=f"{part} participants"))
        r = arr.get(f"varratio__{key}__ratio")
        if r is not None and np.size(r):
            xr = np.linspace(0, 100, len(r))
            ci = varratio_ci(ctx, key, r, seed=seed, inputs=inputs)
            if ci is not None:
                pr.series.append(Series("band", xr, ci[0], y2=ci[1], role="band"))
                n_ci += 1
            pr.series += [Series("line", xr, np.asarray(r, float), role="accent"),
                          Series("hline", y=np.array([1.0]), role="reference", label="equal")]
            n_bars += B.add_spm_bars(pr, [("participants differ", an.get(tests[key]),
                                           "secondary")])
        for p in (pv, pr):
            if not p.series:
                p.note = "no data"
        panels += [pv, pr]
    # CMC of the opening angle with bootstrap CIs over cycles.
    cm = cmc_ci(ctx, seed=seed, inputs=inputs)
    cax = Axis("CMC", lim=None)
    if cm is not None:
        pids = [p for p in ctx.pids() if p in cm["within"]]
        w = [cm["within"][p] for p in pids]
        pw = B.forest(B.pseudo_rows(ctx, pids), [e for e, _, _ in w], [lo for _, lo, _ in w],
                      [hi for _, _, hi in w], "Opening angle: CMC within participants", cax,
                      ref=None)
        ps = B.forest(["median within", "between"],
                      [cm["median_within"][0], cm["between"][0]],
                      [cm["median_within"][1], cm["between"][1]],
                      [cm["median_within"][2], cm["between"][2]],
                      "Opening angle: CMC, cohort", cax, ref=None)
        mw, bw = cm["median_within"], cm["between"]
        cmc_txt = (f"; coefficient of multiple correlation (CMC) of the opening angle within "
                   f"each participant (diamond, line: 95 % bootstrap CI over cycles, "
                   f"{cm['n_boot']} resamples), its median {mw[0]:.2f} [{mw[1]:.2f}, "
                   f"{mw[2]:.2f}] and the CMC of the participant means {bw[0]:.2f} "
                   f"[{bw[1]:.2f}, {bw[2]:.2f}] (cycles resampled within participants)")
    else:
        pids = []
        why = ("no CMC stored" if spm.empty or "cmc_opening_angle" not in set(spm.test)
               else "stored CMC not reproduced from the stored profiles (stale run?)")
        pw = Panel("Opening angle: CMC within participants", cax, Axis(""), [],
                   legend="none", note=why)
        ps = Panel("Opening angle: CMC, cohort", cax, Axis(""), [], legend="none", note=why)
        cmc_txt = ""
    h = 1.4
    pw.height = B.rows_height(len(pids), per=0.15, extra=0.4, lo=1.0) / h
    ps.height = pw.height
    panels += [pw, ps]
    n_p = len(inputs["theta"])
    return FigureSpec("emg_varratio", "Between- vs within-participant variance", panels,
                      rows=3, cols=2, width="text", height_in=h,
                      caption="Variance decomposition along the cycle: variance of the "
                              "participant means (between) and mean within-participant "
                              "variance over cycles (within), and their ratio, for the "
                              "opening angle in open-close and the right masseter activation "
                              f"in chewing ({V.EMG_NORM_SHORT}; {n_p} participants)"
                              + ("; band: pointwise 95 % bootstrap CI of the ratio "
                                 f"({B.N_BOOT_CURVE} resamples of participants)" if n_ci else "")
                              + ("; " + B.spm_caption("one-way ANOVA across participants")
                                 if n_bars else "") + cmc_txt + ".",
                      sources=["stats_emg:varratio__theta", "stats_emg:varratio__masseter_r_chewing",
                               "stats_emg:spm", "emg:profile__open_close__theta_deg",
                               "emg:profile__chewing__pct"])
