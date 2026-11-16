# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Views of the validation study's EMG comparability (stage ``ref_emg``, plan D5) and
trueness/precision (stage ``ref_trueness``, D6); group ``validation`` only.

EMG: JTS-MM = the dataset take of the subject, Delsys = the comparison take (cross-session,
different electrodes and sites); each system normalized to its own MVC (D11), so the views
compare shapes, normalized levels, timing distributions and signal quality, never samples.
Every stored table of both stages also has a table view (``val_table_emg_*``,
``val_table_trueness_*``). Pseudonyms only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import (_arrays, _empty, _pid, _table, _who, n_text,
                                          shared_legend)
from biomech.viz.views.validation import pshape_text, shape_by_pid, shape_by_rows

G = {"group": "validation", "domain": "validation"}
MM, UV, PCT, MSEC, SEC, HZ, DB = ("\\milli\\metre", "\\micro\\volt", "\\percent",
                                  "\\milli\\second", "\\second", "\\hertz", "\\decibel")
# system -> (label, role, dash)
SYSTEMS = {"jtsmm": ("JTS-MM", "system:jtsmm", None), "delsys": ("Delsys", "system:delsys", None),
           "qualisys": ("Qualisys", "system:qualisys", None)}
MUSCLE_ORDER = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l", "digastric_r",
                "digastric_l")
PROFILE_TASKS = ("open_close", "chewing", "protrusion_retrusion", "left_right")
CROSS_SESSION_RAW = ("JTS-MM: dataset take; Delsys: comparison take (different days, "
                     "electrodes and sites)")
CROSS_SESSION = CROSS_SESSION_RAW + ", each normalized to its own maximum"


def _log_ticks(values: Any, extra: tuple[float, ...] = ()) -> tuple[list[float], list[str]]:
    """1-2-5 ticks over the finite range of ``values`` on a log axis, at least three (
    a log axis inside one decade showed a single tick label)."""
    v = np.asarray([x for x in np.r_[np.ravel(values), extra] if np.isfinite(x) and x > 0])
    if not v.size:
        return [], []
    lo, hi = float(v.min()), float(v.max())
    cand = [m * 10.0 ** e for e in range(int(np.floor(np.log10(lo))) - 1,
                                        int(np.ceil(np.log10(hi))) + 2) for m in (1, 2, 5)]
    inside = [c for c in cand if lo / 1.3 <= c <= hi * 1.3]
    if len(inside) < 3:                    # widen to the neighbouring ticks
        below = [c for c in cand if c < lo / 1.3][-1:]
        above = [c for c in cand if c > hi * 1.3][:1]
        inside = below + inside + above
    return inside, [f"{c:g}" for c in inside]


def _style(sys_: str, width: float = 1.0) -> dict[str, Any]:
    st: dict[str, Any] = {"width": width}
    if SYSTEMS[sys_][2]:
        st["dash"] = SYSTEMS[sys_][2]
    return st


def _muscles(names: list[str]) -> list[str]:
    return [m for m in MUSCLE_ORDER if m in names] + [m for m in names if m not in MUSCLE_ORDER]


def _rows_two(p: Panel, labels: list[str], vals: dict[str, list[float]]) -> None:
    """One row per label, the two systems as offset markers (legend once)."""
    for k, (sys_, v) in enumerate(vals.items()):
        y = np.arange(len(labels), dtype=float) + (-0.15 if k == 0 else 0.15)
        p.series.append(Series("scatter", np.asarray(v, float), y, role=SYSTEMS[sys_][1],
                               label=SYSTEMS[sys_][0],
                               style={"ms": "o" if k == 0 else "s"}))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], list(labels)
    p.y.reverse = True


# == EMG =======================================================================================
@register("val_emg_profiles", "dashboard", "EMG: normalized activation profiles, both systems",
          params=("pid", "phase"), phases=PROFILE_TASKS,
          choices={"form": ("panels", "overlay", "difference")}, form="panels", **G)
def emg_profiles(req: ViewRequest) -> FigureSpec:
    """Time-normalized cycles of each muscle (mean ± SD, % of each system's own maximum) with
    the two-sample SPM bars, r and CMC of the ensemble means. Forms ``overlay`` /
    ``difference``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "panels"))
    if form != "panels":
        from biomech.viz.views.validation_r8 import emg_profiles_form

        return emg_profiles_form(req, form)
    fid, title = "val_emg_profiles", "EMG: normalized activation profiles, both systems"
    ctx = req.ctx
    pid = _pid(req)
    task = req.phase if isinstance(req.phase, str) else "open_close"
    a = _arrays(ctx, "ref_emg", pid) if pid else {}
    if "names" not in a:
        return _empty(fid, title, "no ref_emg arrays")
    from biomech.validation.emg import finite_cycles, profile_array_key, profile_key

    names = [str(n) for n in a["names"]]
    keys = [k for k in ([task] if task != "left_right" else
                        [profile_key(task, "left"), profile_key(task, "right")])
            if profile_array_key("jtsmm", k) in a or profile_array_key("delsys", k) in a]
    if not keys:
        return _empty(fid, title, f"no {V.phase_label(task)} profiles")
    prof = _table(ctx, "ref_emg", "profiles", [pid])
    clus = _table(ctx, "ref_emg", "spm_clusters", [pid])
    x = np.linspace(0, 100, 101)
    panels = []
    for m in _muscles(names):
        i = names.index(m)
        for key in keys:
            side = key.partition("__")[2]
            lab = V.muscle_label(m) + (f", to the {side}" if side else "")
            note = ""
            r = prof[(prof["profile"] == key) & (prof["muscle"] == m)] if not prof.empty \
                else prof
            if not r.empty:
                rr = r.iloc[0]
                note = (f"r = {rr['r']:.2f}, CMC = {rr['cmc']:.2f}" if np.isfinite(rr["cmc"])
                        else f"r = {rr['r']:.2f}, CMC undefined*")
            p = Panel(f"{lab} ({note})" if note else lab, Axis("Cycle", PCT),
                      Axis("Activation", PCT), [], legend="below")
            for s in ("jtsmm", "delsys"):
                Y = a.get(profile_array_key(s, key))
                if Y is None:
                    continue
                y = finite_cycles(np.asarray(Y, float)[:, :, i])
                ids = a.get(profile_array_key(s, key).removesuffix("__pct") + "__cycle")
                n_all = len(ids) if ids is not None else len(Y)
                if not len(y):
                    continue
                mu, sd = y.mean(axis=0), y.std(axis=0, ddof=1) if len(y) > 1 else 0 * y[0]
                # activation cannot be negative: the SD band is clipped at 0
                p.series.append(Series("band", x, np.maximum(mu - sd, 0.0), y2=mu + sd,
                                       role=SYSTEMS[s][1]))
                p.series.append(Series("line", x, mu, role=SYSTEMS[s][1], style=_style(s),
                                       label=f"{SYSTEMS[s][0]} (n = {len(y)}"
                                             + (f" of {n_all}" if n_all != len(y) else "")
                                             + ")"))
            if not clus.empty:
                c = clus[(clus["profile"] == key) & (clus["muscle"] == m)]
                cl = [(float(q.start_pct), float(q.end_pct), float(q.p)) for q in c.itertuples()]
                B.add_spm_bars(p, [("JTS-MM vs Delsys", cl, "accent")])
            panels.append(p)
    ncol = 2
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / ncol)), cols=ncol,
                      width="text", height_in=1.5,
                      sources=["ref_emg:arrays", "ref_emg:profiles", "ref_emg:spm_clusters"],
                      perspective=f"{_who(ctx, pid)}, {V.phase_label(task)}", meta={"pid": pid})
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of each muscle, both systems",
        [("panel titles", "Pearson r and CMC (coefficient of multiple correlation) of the two "
                          "ensemble means"),
         ("*", "CMC undefined, because the curves differ more in level than they vary in "
               "shape (typical of a muscle near rest)"),
         ("bars above the curves", "supra-threshold clusters of the SPM test")],
        f"{_who(ctx, pid)}, cycle count n per system in the legend",
        statistic="mean ± SD per system (band clipped at 0 %)",
        test="two-sample SPM{t} over cycles, unequal variances, p < 0.05",
        notes=[CROSS_SESSION]))


@register("val_emg_mvc", "dashboard", "EMG: MVC levels and MVC/task maximum, both systems", **G)
def emg_mvc(req: ViewRequest) -> FigureSpec:
    """MVC level (µV, the D11 MVC maximum of the masked envelope) per muscle and system, and
    the MVC/task-maximum ratio (< 1: a task exceeded the MVC, ``ref_source = task``)."""
    fid, title = "val_emg_mvc", "EMG: MVC levels and MVC/task maximum, both systems"
    ctx = req.ctx
    d = _table(ctx, "ref_emg", "mvc")
    if d.empty:
        return _empty(fid, title, "no ref_emg mvc table")
    panels = []
    for pid in sorted(d["pid"].unique()):
        g = d[d["pid"] == pid]
        ms = _muscles(list(g["muscle"].unique()))
        labels = [V.muscle_label(m) for m in ms]
        for col, lab, ax in (("mvc_max_uv", "MVC level", Axis("Envelope maximum", UV, log=True)),
                             ("mvc_over_task", "MVC / task maximum",
                              Axis("Ratio", "", log=True))):
            vals = {s: [float(g[(g.system == s) & (g.muscle == m)][col].iloc[0])
                        if ((g.system == s) & (g.muscle == m)).any() else np.nan for m in ms]
                    for s in ("jtsmm", "delsys")}
            p = Panel(f"{_who(ctx, pid)}: {lab}", ax, Axis(""), [], legend="below")
            _rows_two(p, labels, vals)
            tk, tl = _log_ticks(list(vals.values()), (1.0,) if col == "mvc_over_task" else ())
            if tk:
                p.x = Axis(ax.label, ax.unit, log=True, lim=(tk[0], tk[-1]), ticks=tk,
                           ticklabels=tl)
            if col == "mvc_over_task":
                p.series.append(Series("vline", x=np.array([1.0]), role="reference",
                                       label="task = MVC"))
            shape_by_pid([p], sorted(d["pid"].unique()).index(pid))   # X1: shape = participant
            panels.append(p)
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=len(panels) // 2, cols=2, width="text",
                      height_in=1.7, sources=["ref_emg:mvc"])
    return attach(spec, CaptionSpec(
        "maximal effort of each muscle in both EMG systems, one row of panels per participant. "
        "Left: MVC level per muscle (maximum of the masked 150-ms RMS envelope over the MVC "
        "phases; Delsys: the stronger of the MVC files). Right: MVC maximum over the largest "
        "task activity",
        [("blue", "JTS-MM"), ("black", "Delsys"), pshape_text(ctx, d["pid"]),
         ("vertical line (right)", "task maximum = MVC maximum")],
        n_text(ctx, d["pid"]),
        statistic="one value per participant, muscle and system, no summary (log axes)",
        notes=["A ratio below 1 makes the task the normalization reference.",
               CROSS_SESSION_RAW]))


@register("val_emg_quality", "dashboard", "EMG: signal quality, both systems", params=("pid",),
          choices={"form": ("offset_dots", "dumbbell")}, form="offset_dots", **G)
def emg_quality(req: ViewRequest) -> FigureSpec:
    """Rest noise, SNR (MVC p95 / rest envelope), mains share, and MNF/MDF of the task
    phases per muscle and system. Form ``dumbbell``: rest noise, SNR and MVC level with the
    two systems joined (``validation_gallery``)."""
    if req.options.get("form", "offset_dots") == "dumbbell":
        from biomech.viz.views.validation_gallery import emg_quality_dumbbell

        return emg_quality_dumbbell(req)
    fid, title = "val_emg_quality", "EMG: signal quality, both systems"
    ctx = req.ctx
    pid = _pid(req)
    q = _table(ctx, "ref_emg", "quality", [pid]) if pid else pd.DataFrame()
    sp = _table(ctx, "ref_emg", "spectra", [pid]) if pid else pd.DataFrame()
    if q.empty:
        return _empty(fid, title, "no ref_emg quality table")
    ms = _muscles(list(q["muscle"].unique()))
    labels = [V.muscle_label(m) for m in ms]

    def vals(df: pd.DataFrame, col: str) -> dict[str, list[float]]:
        return {s: [float(df[(df.system == s) & (df.muscle == m)][col].median())
                    if ((df.system == s) & (df.muscle == m)).any() else np.nan for m in ms]
                for s in ("jtsmm", "delsys")}

    panels = []
    for col, lab, ax in (("rest_rms_uv", "(a) Rest noise", Axis("RMS", UV)),
                         ("snr_db", "(b) SNR", Axis("MVC p95 / rest", DB)),
                         ("mains_share_raw", "(c) Mains share at rest (raw)", Axis("Share", ""))):
        p = Panel(lab, ax, Axis(""), [], legend="below")
        _rows_two(p, labels, vals(q, col))
        panels.append(p)
    if not sp.empty:
        task = sp[(sp["take"] == "motion") & (sp["phase"].isin(["open_close", "chewing",
                                                               "protrusion_retrusion",
                                                               "left_right"]))]
        for col, lab in (("mnf_hz", "(d) Mean frequency, tasks"),
                         ("mdf_hz", "(e) Median frequency, tasks")):
            p = Panel(lab, Axis(lab.split(") ")[1].split(",")[0], HZ), Axis(""), [],
                      legend="below")
            _rows_two(p, labels, vals(task, col))
            panels.append(p)
        mvc = sp[sp["take"] != "motion"]
        p = Panel("(f) 200–450 Hz band, MVC", Axis("Share of 20–450 Hz power", ""), Axis(""),
                  [], legend="below")
        _rows_two(p, labels, vals(mvc, "share_200_450"))
        panels.append(p)
    if pid in ctx.pids():                 # X1: shape = participant, colour = system
        shape_by_pid(panels, sorted(ctx.pids()).index(pid))
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 3)), cols=3,
                      width="text", height_in=1.6,
                      sources=["ref_emg:quality", "ref_emg:spectra"],
                      perspective=_who(ctx, pid), meta={"pid": pid})
    return attach(spec, CaptionSpec(
        "EMG signal quality per muscle of both systems, same processing chain: (a) rest noise, "
        "(b) SNR, the 95th percentile of the MVC envelope over the rest envelope, (c) mains "
        "share at rest, (d, e) mean and median frequency in the cycle tasks, (f) share of the "
        "200–450 Hz band in the MVC",
        [("blue", "JTS-MM"), ("black", "Delsys"),
         ("circle" if sorted(ctx.pids()).index(pid) % 2 == 0 else "square",
          f"{_who(ctx, pid)} (the participant's marker in every systems figure)")]
        if pid in ctx.pids() else [("blue", "JTS-MM"), ("black", "Delsys")],
        f"{_who(ctx, pid)}, one value per muscle and system",
        statistic="rest values from the rest phase; spectra: Welch over 0.5-s mask-free "
                  "segments, median over the cycle tasks (d, e) and over the MVC recordings (f)",
        notes=[CROSS_SESSION_RAW]))


@register("val_emg_delta", "dashboard", "EMG: Delsys-QTM start latency δ check",
          params=("pid",), **G)
def emg_delta(req: ViewRequest) -> FigureSpec:
    """Cross-correlation of the Delsys opener/closer envelopes with the Qualisys jaw angle
    and speed (δ = 0 assumed) and the lead windows each candidate δ implies."""
    fid, title = "val_emg_delta", "EMG: Delsys-QTM start latency δ check"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_emg", pid) if pid else {}
    tl = _table(ctx, "ref_emg", "timeline", [pid]) if pid else pd.DataFrame()
    dc = _table(ctx, "ref_emg", "delta_check", [pid]) if pid else pd.DataFrame()
    if "delta_lags_s" not in a:
        return _empty(fid, title, "no δ check arrays")
    lags = np.asarray(a["delta_lags_s"], float)
    tasks = sorted({k.split("__")[1] for k in a if k.startswith("delta__")})
    cands = sorted(float(c.split("_d")[-1]) for c in dc.columns
                   if c.startswith("plausible_d")) if not dc.empty else [0.0]
    panels = []
    for task in tasks:
        p = Panel(V.phase_label(task), Axis("Lag (EMG first > 0), δ = 0 assumed", SEC),
                  Axis("r"), [], legend="below")
        for k, d in enumerate(cands):
            p.series.append(Series("hspan", x=np.array([0.0 - d]), y2=np.array([0.25 - d]),
                                   role="muted" if k else "context"))
            p.series.append(Series("text", x=np.array([0.125 - d]), y=np.array([1.0]),
                                   text=[f"δ = {d:g} s"], role="context",
                                   style={"anchor": "below"}))
        for j, (grp, sig) in enumerate((("openers", "angle_deg"), ("openers", "speed_dps"),
                                        ("closers", "angle_deg"), ("closers", "speed_dps"))):
            r = a.get(f"delta__{task}__{grp}__{sig}")
            if r is None:
                continue
            p.series.append(Series("line", lags, np.asarray(r, float), role=f"cat:{j}",
                                   style={"width": 0.9},
                                   label=f"{grp}, {'angle' if sig == 'angle_deg' else 'speed'}"))
        panels.append(p)
    if not panels:
        return _empty(fid, title, "no δ cross-correlation arrays (re-run ref_emg)")
    cap = ""
    if not tl.empty:
        r0 = tl.iloc[0]
        cap = (f" Chosen δ = {r0['delta_s']:g} s ({r0['delta_status']}; votes "
               + ", ".join(f"δ = {d:g}: {int(r0.get(f'votes_d{d:g}', 0))}" for d in cands)
               + f"). Delsys − QTM duration: {r0['duration_diff_s']:+.2f} s.")
    return FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=1.8,
                      caption="Pearson r between the Delsys group envelopes (each channel over "
                              "its window median) and the Qualisys jaw angle and speed, "
                              "Delsys time taken as QTM time. Shaded: the physiological "
                              "lead window (0-250 ms) under each candidate latency δ; the openers "
                              "vote." + cap,
                      sources=["ref_emg:arrays", "ref_emg:delta_check", "ref_emg:timeline"],
                      perspective=_who(ctx, pid), meta={"pid": pid})


@register("val_emg_timing", "dashboard", "EMG: onset timing, both systems",
          choices={"form": ("strip", "raincloud")}, form="strip", **G)
def emg_timing(req: ViewRequest) -> FigureSpec:
    """EMG-to-motion leads of the usable channel-cycles (TKEO, primary onset rule) per system
    and mechanical source, and the difference of the medians with its bootstrap CI. Form
    ``raincloud``: the leads as rainclouds (``validation_gallery``)."""
    if req.options.get("form", "strip") == "raincloud":
        from biomech.viz.views.validation_gallery import emg_timing_raincloud

        return emg_timing_raincloud(req)
    fid, title = "val_emg_timing", "EMG: onset timing, both systems"
    ctx = req.ctx
    on = _table(ctx, "ref_emg", "timing_onsets")
    cmp_ = _table(ctx, "ref_emg", "timing_compare")
    if on.empty:
        return _empty(fid, title, "no ref_emg onsets")
    rule = ctx.onset_rule()
    u = on[on["usable"].astype(bool) & (on["detector"] == "tkeo") & (on["rule"] == rule)]
    labels, vals, roles = [], [], []
    # row = participant, task and phase, then the EMG system and the motion source of the
    # mechanical onset, with the channel-cycle count
    src_lab = {"jtsmm_optical": "optics", "qualisys": "Qualisys"}
    for (pid, task, direction), g in u.groupby(["pid", "task", "direction"]):
        for (sys_, src), h in g.groupby(["system", "mech_source"]):
            labels.append(f"{_who(ctx, pid)} {V.phase_label(task, short=True)} {direction} | "
                          f"{SYSTEMS[sys_][0]}, {src_lab.get(src, src)} ({len(h)})")
            vals.append(h["lead_ms"].to_numpy(float))
            roles.append(SYSTEMS[sys_][1])
    pa = B.strip(labels, vals, "(a) Channel-cycles", Axis("EMG lead", MSEC), ref=0.0,
                 roles=roles)
    pids = sorted(u["pid"].unique())
    row_pid = [pid for (pid, _, _), g in u.groupby(["pid", "task", "direction"])
               for _ in g.groupby(["system", "mech_source"])]
    shape_by_rows(pa, row_pid, pids)              # X1: shape = participant, colour = system
    panels = [pa]
    if not cmp_.empty:
        lab2 = [f"{_who(ctx, r.pid)} {V.phase_label(r.task, short=True)} {r.direction} | "
                f"{src_lab.get(r.delsys_mech_source, r.delsys_mech_source)} "
                f"({int(r.n_jtsmm)}/{int(r.n_delsys)})"
                for r in cmp_.itertuples()]
        pb = B.strip(lab2, [[v] for v in cmp_.get("diff_ms", pd.Series(np.nan,
                                                                       index=cmp_.index))],
                     "(b) Delsys − JTS-MM", Axis("Difference of medians", MSEC), ref=0.0,
                     intervals=list(zip(cmp_.get("diff_ci_lo_ms", pd.Series(np.nan, index=cmp_.index)),
                                        cmp_.get("diff_ci_hi_ms", pd.Series(np.nan, index=cmp_.index)))))
        shape_by_rows(pb, list(cmp_["pid"]), pids)
        panels.append(pb)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=max(1.8, 0.14 * len(labels)),
                      sources=["ref_emg:timing_onsets", "ref_emg:timing_compare"])
    empty = cmp_[~np.isfinite(cmp_["diff_ms"].astype(float))] if "diff_ms" in cmp_ else cmp_
    notes = ["Delsys onsets are placed on the JTS-MM clock through the Delsys start latency "
             "δ and the clock model."]
    if len(empty):
        notes.append("Rows of (b) without a marker: only " + ", ".join(sorted(
            {str(int(n)) for n in empty["n_jtsmm"]})) + " usable JTS-MM channel-cycles, too "
            "few for a difference of medians.")
    per_pid = "; ".join(f"{_who(ctx, p_)} {int((u['pid'] == p_).sum())}"
                        for p_ in sorted(u["pid"].unique()))
    return attach(spec, CaptionSpec(
        "lead of the EMG onset over the mechanical onset of the movement per usable "
        "channel-cycle, both EMG systems (> 0: EMG first), and (b) the difference of the "
        "medians Delsys − JTS-MM (different sessions: agreement in distribution, not per "
        "cycle). Onset detector: TKEO (Teager–Kaiser energy operator), onset rule: "
        f"{V.term('rule', rule)}",
        [("small blue markers (a)", "JTS-MM EMG, one channel-cycle"),
         ("small black markers (a)", "Delsys EMG, one channel-cycle"),
         ("large black markers", "medians (a), difference of the medians (b)"),
         pshape_text(ctx, u["pid"]),
         ("whiskers (b)", "95 % cycle-bootstrap CI"),
         ("rows", "participant, task and phase | EMG system ((a) only) and the motion source "
                  "of the Delsys mechanical onset (optics: JTS-MM optics; JTS-MM EMG always "
                  "with the JTS-MM optics), channel-cycles in parentheses ((b): JTS-MM / "
                  "Delsys)")],
        f"n = {u['pid'].nunique()} participants, usable channel-cycles {per_pid}",
        statistic="cycle-bootstrap 95 % CI of the difference of the medians",
        notes=notes))


# == trueness / precision ======================================================================
def _cohort_arrays(ctx: Any, stage: str) -> dict[str, np.ndarray]:
    from biomech.results import COHORT_UNIT

    return _arrays(ctx, stage, COHORT_UNIT)


def _ladder_n(d: pd.DataFrame) -> str:
    """The bench sample of a ladder table: conditions (system × orientation × distance) and
    slots per condition."""
    if d.empty or "n_slots" not in d or d["n_slots"].isna().all():
        return ""
    v = sorted(int(x) for x in d["n_slots"].dropna().unique())
    slots = f"{v[0]}–{v[-1]}" if len(v) > 1 else str(v[0])
    return f"{len(d)} conditions (system × orientation × camera distance), {slots} slots each"


@register("val_trueness", "dashboard", "Ladder: step error against the nominal distance", **G)
def trueness_view(req: ViewRequest) -> FigureSpec:
    """Per orientation and camera distance: deviation of each measured 20-mm step from the
    nominal step and the residual to the metric ladder line, per system."""
    fid, title = "val_trueness", "Ladder: step error against the nominal distance"
    ctx = req.ctx
    a = _cohort_arrays(ctx, "ref_trueness")
    tt = _table(ctx, "ref_trueness", "trueness")
    keys = sorted({k.rsplit("__", 1)[0] for k in a if k.startswith("ladder__")})
    if not keys:
        return _empty(fid, title, "no ladder arrays")
    conds = sorted({tuple(k.split("__")[2:4]) for k in keys})
    panels = []
    for ori, dist in conds:
        p = Panel(f"{ori.capitalize()}, {dist}", Axis("Nominal distance", MM),
                  Axis("Error", MM), [], legend="below")
        for k in keys:
            _, sys_, o, d = k.split("__")
            if (o, d) != (ori, dist):
                continue
            nom = np.asarray(a[f"{k}__nominal_mm"], float)
            step = np.asarray(a[f"{k}__step_dev_mm"], float)
            res = np.asarray(a[f"{k}__resid_mm"], float)
            role = SYSTEMS.get(sys_, (sys_, "accent", None))[1]
            lab = SYSTEMS.get(sys_, (sys_,))[0]
            p.series.append(Series("line", nom[1:], step, role=role, style={"width": 1.0},
                                   label=f"{lab}: step error"))
            # bench: no participant, so one marker for both systems (colour = system; X1)
            p.series.append(Series("scatter", nom[1:], step, role=role, style={"ms": "o"}))
            u = None
            if not tt.empty:
                r = tt[(tt.system == sys_) & (tt.orientation == ori) & (tt.distance == dist)]
                if not r.empty and "u_x" in r:
                    u = r.iloc[0][["u_x", "u_y", "u_z"]].to_numpy(float)
            along = res @ u if u is not None else np.linalg.norm(res, axis=1)
            # the residual in a light version of the system's line (color and dash), so the
            # two systems' residuals stay apart (thick vs thin grey did not)
            p.series.append(Series("line", nom, along, role=role,
                                   style={"width": 0.7, "alpha": 0.4},
                                   label=f"{lab}: residual along the ladder"))
        p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
        panels.append(p)
    stat = ""
    if not tt.empty and "trueness_step_mean_mm" in tt:
        stat = "mean step error [95 % CI]: " + "; ".join(
            f"{SYSTEMS.get(r.system, (r.system,))[0]} {r.orientation} {r.distance} "
            f"{B.num(r.trueness_step_mean_mm)} {B.ci(r.trueness_ci_lo_mm, r.trueness_ci_hi_mm)} mm"
            for r in tt.itertuples() if np.isfinite(r.trueness_step_mean_mm))
    shared_legend(panels)                 # one legend per distinct entry set
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=1.6,
                      sources=["ref_trueness:arrays", "ref_trueness:trueness"])
    return attach(spec, CaptionSpec(
        "trueness on the hand-measured ladder (20-mm steps; central 80 % of each static take), "
        "one panel per orientation and camera distance: measured minus nominal step at the far "
        "slot of each step, and the residual to the metric line (scale fixed to the ruler, "
        "direction free)",
        [("solid blue / dashed black lines with dots", "step error of JTS-MM / Qualisys"),
         ("light lines", "residual along the ladder, in the system's color and line style"),
         ("horizontal line", "0, no error")],
        n=_ladder_n(tt), statistic=stat or "one value per step, no summary"))


@register("val_precision", "dashboard", "Ladder: precision and drift", **G)
def precision_view(req: ViewRequest) -> FigureSpec:
    """SD of the static samples (within the slot takes, pooled; and the detrended 30-s
    takes), the drift, and the detrended 30-s traces."""
    fid, title = "val_precision", "Ladder: precision and drift"
    ctx = req.ctx
    tt = _table(ctx, "ref_trueness", "trueness")
    pr = _table(ctx, "ref_trueness", "precision")
    a = _cohort_arrays(ctx, "ref_trueness")
    if tt.empty and pr.empty:
        return _empty(fid, title, "no precision results")
    panels = []
    # rows JTS-MM first in every panel, markers in the system colors
    order = {"jtsmm": 0, "qualisys": 1}
    if not tt.empty:
        tt = tt.sort_values("system", key=lambda c: c.map(order), kind="stable")
    if not pr.empty:
        pr = pr.sort_values("system", key=lambda c: c.map(order), kind="stable")

    def by_system(p: Panel, systems: list[str]) -> Panel:
        """One value per row: the strip's median diamond in the row's system color."""
        for s_ in p.series:
            if s_.kind == "scatter" and s_.role == "accent" and s_.style.get("ms") == "d":
                s_.role = SYSTEMS.get(systems[int(round(float(s_.y[0])))], ("", "accent"))[1]
        return p

    if not tt.empty and "precision_static_sd_3d_mm" in tt:
        lab = [f"{SYSTEMS.get(r.system, (r.system,))[0]} {r.orientation} {r.distance}"
               for r in tt.itertuples()]
        panels.append(by_system(B.strip(lab, [[v] for v in tt["precision_static_sd_3d_mm"]],
                                        "(a) Static SD, slot takes", Axis("3-D SD", MM)),
                                list(tt["system"])))
    if not pr.empty and "sd_3d_mm" in pr:
        lab = [f"{SYSTEMS.get(r.system, (r.system,))[0]} {r.distance}" for r in pr.itertuples()]
        p = by_system(B.strip(lab, [[v] for v in pr["sd_3d_mm"]], "(b) 30-s takes, detrended",
                              Axis("3-D SD", MM)), list(pr["system"]))
        p.series.append(Series("scatter", pr["sd_3d_raw_mm"].to_numpy(float),
                               np.arange(len(pr), dtype=float), role="muted",
                               label="not detrended"))
        p.legend = "below"
        panels.append(p)
        panels.append(by_system(B.strip(lab, [[v] for v in pr["drift_mm_per_min"]], "(c) Drift",
                                        Axis("Linear drift", "\\milli\\metre\\per\\minute")),
                                list(pr["system"])))
    tr = Panel("(d) Detrended 30-s traces", Axis("Time", SEC), Axis("|p − trend|", MM), [],
               legend="below")
    for k in sorted(x for x in a if x.startswith("precision__") and x.endswith("__t_s")):
        base = k[: -len("__t_s")]
        _, sys_, _, dist = base.split("__")
        r = np.linalg.norm(np.asarray(a[f"{base}__detrended_mm"], float), axis=1)
        # colour = system (JTS-MM blue, Qualisys ink), line style = distance (near solid, far
        # dashed): the paper's system colours (2026-11-05; was one category colour each)
        tr.series.append(Series("line", np.asarray(a[k], float), r,
                                role="system:jtsmm" if sys_ == "jtsmm" else "system:qualisys",
                                style={"width": 0.5,
                                       "dash": "solid" if dist == "near" else "dashed"},
                                label=f"{SYSTEMS.get(sys_, (sys_,))[0]} {dist}"))
    panels.append(tr)
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / 2)), cols=2,
                      width="text", height_in=1.6,
                      sources=["ref_trueness:trueness", "ref_trueness:precision",
                               "ref_trueness:arrays"])
    n = "; ".join(x for x in (_ladder_n(tt), f"{len(pr)} static takes of 30 s" if len(pr) else "")
                  if x)
    return attach(spec, CaptionSpec(
        "precision (repeatability) of both systems on the bench: (a) the static samples within "
        "each ladder slot, (b) the 30-s static takes after removing a linear trend per axis, "
        "(c) that trend (drift) and (d) the detrended traces",
        [("markers (a–c)", "one value per condition or take, in the system color (JTS-MM "
                           "blue, Qualisys black)"),
         ("grey (b)", "without detrending"),
         ("lines (d)", "distance from the trend, one color per system and distance")],
        n, statistic="(a) 3-D SD pooled over the slots of a condition; (b) 3-D SD of each "
                     "take; (c) slope of the linear trend; (d) raw traces, no summary",
        notes=["The trend is the drift."]))


# == tables ====================================================================================
redact = B.redact
PRIVATE_COLUMNS = B.PRIVATE_COLUMNS


@register("val_table_emg_agreement", "dashboard", "Table: EMG profile agreement (paper)", **G)
def table_emg_agreement(req: ViewRequest) -> TableSpec:
    """Paper table of the D5 profile comparison: open-close and chewing, per participant and
    muscle -- r and CMC of the two systems' ensemble means, RMS and mean difference, SPM."""
    d = _table(req.ctx, "ref_emg", "profiles")
    if not d.empty:
        d = d[d["profile"].isin(["open_close", "chewing"])].copy()
        order = {m: i for i, m in enumerate(V.MUSCLES)}
        torder = {"open_close": 0, "chewing": 1}           # as named in the caption
        d = d.sort_values(["pid", "task", "muscle"], key=lambda c: c.map(order)
                          if c.name == "muscle" else c.map(torder) if c.name == "task"
                          else c, kind="stable")
    t = B.frame_table(redact(d), "val_table_emg_agreement", "",
                      columns=["pid", "task", "muscle", "r", "cmc", "rms_diff_pct",
                               "mean_diff_pct", "spm_h0reject"], ctx=req.ctx)
    t.sources = ["ref_emg:profiles"]
    if "pid" in d and len(d):           # the numbers the claim quotes: median over muscles
        from biomech.viz.views.validation import add_summary_rows

        add_summary_rows(t, "muscle", "median over muscles", group=("pid", "task"),
                         how="median")
        # and the descriptive mean of the participants' medians per task (n = 2 rule)
        med = [r for r in t.rows if r.get("muscle") == "median over muscles"]
        for task in dict.fromkeys(r.get("task") for r in med):
            sel = [r for r in med if r.get("task") == task]
            if len(sel) < 2:
                continue
            row: dict[str, Any] = {"pid": f"mean of {len(sel)}", "task": task,
                                   "muscle": "median over muscles"}
            for k in ("r", "cmc", "rms_diff_pct", "mean_diff_pct"):
                v = [r.get(k) for r in sel]
                ok = [float(x) for x in v if isinstance(x, (int, float, np.floating))
                      and np.isfinite(float(x))]
                row[k] = float(np.mean(ok)) if len(ok) == len(v) else np.nan
            t.rows.append(row)
    return attach(t, CaptionSpec(
        "EMG profile agreement of JTS-MM and Delsys (open-close and chewing): Pearson r and CMC "
        "of the ensemble-mean profiles, RMS and mean difference (% of each system's own "
        "maximum) and whether SPM finds a difference",
        [("–", "CMC undefined, because the two curves differ more in level than they vary in "
               "shape")],
        n_text(req.ctx, d["pid"]) if "pid" in d and len(d) else "",
        statistic="per participant, task and muscle over the 101 time-normalized points of "
                  "the two ensemble means (mean over the cycles of each system); difference = "
                  "Delsys − JTS-MM; median over muscles per participant and task, and the "
                  "descriptive mean of the two participants' medians",
        test="SPM: two-sample SPM{t} over the cycles, unequal variances, p < 0.05",
        notes=[CROSS_SESSION]))


@register("val_table_ladder", "dashboard", "Table: bench ladder (paper)", **G)
def table_ladder(req: ViewRequest) -> TableSpec:
    """Paper table of the D6 bench ladder: trueness, scale error, accuracy and static
    precision per system, orientation and distance."""
    d = _table(req.ctx, "ref_trueness", "summary")
    t = B.frame_table(redact(d), "val_table_ladder", "",
                      columns=["system", "orientation", "distance", "trueness_step_mean_mm",
                               "scale_error_pct", "accuracy_rms_mm",
                               "precision_static_sd_3d_mm"], ctx=req.ctx)
    t.sources = ["ref_trueness:summary"]
    if {"trueness_ci_lo_mm", "trueness_ci_hi_mm"} <= set(d.columns):    # the CI in the cell
        for r, (_, x) in zip(t.rows, d.iterrows()):
            v, lo, hi = (float(x[k]) for k in ("trueness_step_mean_mm", "trueness_ci_lo_mm",
                                                "trueness_ci_hi_mm"))
            if np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi):
                # the shared number rule (real minus), 2 decimals as in val_trueness
                r["trueness_step_mean_mm"] = B.est_ci(v, lo, hi, 2, adaptive=False)
        for c in t.columns:
            if c.key == "trueness_step_mean_mm":
                c.header, c.align = "Trueness and 95\u00a0%\u00a0CI", "l"
    return attach(t, CaptionSpec(
        "bench ladder (20-mm steps): mean step error (trueness), scale error, accuracy (RMS) and "
        "static precision (3-D SD), per system, orientation and camera distance",
        n=_ladder_n(d),
        statistic="per condition: mean over the steps (trueness), slope of the slot means "
                  "against the nominal distance (scale), RMS of every static sample's distance "
                  "from its slot's point on the fitted ladder line (accuracy), 3-D SD pooled over the slots "
                  "(precision)",
        notes=["Trueness: mean step error with its 95 % CI over the steps; the residual to the "
               "fitted line and the drift of the 30-s static takes are in the supplementary source "
               "data."]))
