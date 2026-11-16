# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Compact summary of the C2 timing cues (J18, post hoc / exploratory; paper revision
2026-11-09): every cue relative to the mechanical onset in one figure, open-close and chewing
side by side.

* (a, b) cues as EMG leads before their motion onset (> 0: before), every participant as a
  dot, the cohort median as a diamond with its 95 % BCa CI, n at the row end:
  the digastric onset and the closer onset with the symmetric ±0.4 s window (frozen comparison
  of ``c2_symmetric_window``, driving-burst rule; their shift null as an open circle), the
  digastric rise of the group-averaged envelope (J18.4 rule), the closer release relative to
  t_ω and to t_α (``timing:c2e_release``, ``c2e_cycles``), and the causal IMU detection
  (J18.5: opening-angle rate beyond rest mean + 3 SD, held 25 ms; lead = -(t_IMU - t_mech));
* (c, d) the continuous EMG-to-acceleration lag (J18.2; > 0: EMG first): closers against the
  closing acceleration, digastric against the opening acceleration; the median of the
  participants' centred shift nulls as an open circle.

Stored results only (the J18 audit files through ``viz.derived_dataset``, the frozen
comparison). Colours follow the other timing figures (Okabe-Ito: closers blue, digastric
vermilion, the IMU grey).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Series
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.results_timing import TIMING_TASKS

LEAD_AX = Axis("EMG lead before the motion onset", V.MSEC)
LAG_AX = Axis("EMG-to-acceleration lag", V.MSEC)
CUES = (("dig_onset", "digastric onset", "group:openers"),
        ("dig_rise", "digastric rise (average)", "group:openers"),
        ("rel_omega", "closer release (velocity onset)", "group:closers"),
        ("rel_alpha", "closer release (acceleration onset)", "group:closers"),
        ("closer_onset", "closer onset, to closing", "group:closers"),
        ("imu", "IMU detection", "context"))
LAGS = (("closing", "closers, closing", "group:closers"),
        ("opening", "digastric, opening", "group:openers"))


def _sym(ctx: Any, task: str, direction: str, rule: str = "drive"
         ) -> tuple[np.ndarray, float]:
    """Per-participant C2 leads with the symmetric window and the cohort median null."""
    from biomech.sensitivity import comparison_path
    from biomech.viz import derived
    from biomech.viz.derived_dataset import _keyparts

    try:
        d = pd.read_parquet(comparison_path(ctx.config, "c2_symmetric_window"))
        s = derived.table(ctx, "j18_symmetric")
    except FileNotFoundError:
        return np.empty(0), np.nan
    d = d[(d["table"] == "timing_null") & (d["column"] == "lead_ms")]
    want = {"task": task, "direction": direction, "detector": "tkeo", "rule": rule}
    sel = d[d["key"].map(lambda k: all(_keyparts(k).get(a) == b for a, b in want.items()))]
    row = s[(s.task == task) & (s.direction == direction) & (s.rule == rule)]
    return (sel["variant"].dropna().to_numpy(float),
            float(row["lead_null_ms"].iloc[0]) if not row.empty else np.nan)


def _audit(ctx: Any, name: str) -> pd.DataFrame:
    from biomech.viz.derived_dataset import _j18_file

    try:
        return _j18_file(ctx, name)
    except FileNotFoundError:
        return pd.DataFrame()


def cue_leads(ctx: Any, task: str) -> dict[str, tuple[np.ndarray, float]]:
    """Per cue the participants' leads (ms, > 0 before the motion onset) and the null (NaN
    where the cue has none)."""
    out: dict[str, tuple[np.ndarray, float]] = {}
    out["dig_onset"] = _sym(ctx, task, "opening")
    out["closer_onset"] = _sym(ctx, task, "closing")
    rise = _audit(ctx, "rise")
    r = rise[(rise.task == task) & (rise.cue == "digastric_rise")] if not rise.empty else rise
    out["dig_rise"] = (r["lead_ms"].dropna().to_numpy(float) if not r.empty else np.empty(0),
                       np.nan)
    try:
        rel, cyc = ctx.table("timing", "c2e_release"), ctx.table("timing", "c2e_cycles")
    except Exception:  # noqa: BLE001 -- a store without C2-E
        rel, cyc = pd.DataFrame(), pd.DataFrame()
    out["rel_omega"] = out["rel_alpha"] = (np.empty(0), np.nan)
    if not rel.empty:
        a = (cyc[cyc["direction"] == "opening"].dropna(subset=["t_alpha_us"])
             .drop_duplicates(["pid", "task", "cycle"])[["pid", "task", "cycle", "t_alpha_us"]])
        g = rel[rel.task == task].dropna(subset=["t_release_us"]).merge(
            a, on=["pid", "task", "cycle"], how="left")
        g["lead_alpha_ms"] = (g["t_alpha_us"] - g["t_release_us"]) / 1e3
        out["rel_omega"] = (g.groupby("pid")["lead_release_ms"].median().dropna()
                            .to_numpy(float), np.nan)
        out["rel_alpha"] = (g.dropna(subset=["lead_alpha_ms"]).groupby("pid")["lead_alpha_ms"]
                            .median().to_numpy(float), np.nan)
    dev = _audit(ctx, "device")
    dv = dev[(dev.task == task) & (dev.detector == "digastric_onset")] if not dev.empty else dev
    out["imu"] = ((-(dv["t_imu_k3_us"] - dv["t_mech_us"]) / 1e3).groupby(dv["pid"]).median()
                  .dropna().to_numpy(float) if not dv.empty else np.empty(0), np.nan)
    return out


def lag_values(ctx: Any, task: str) -> dict[str, tuple[np.ndarray, float]]:
    """Per direction the participants' lags (ms) and the median of their centred nulls."""
    cp = _audit(ctx, "coupling")
    out = {}
    for direction, _, _ in LAGS:
        g = cp[(cp.task == task) & (cp.direction == direction)] if not cp.empty else cp
        out[direction] = ((g["lag_ms"].dropna().to_numpy(float),
                           float(g["null_lag_median_ms"].median()))
                          if not g.empty else (np.empty(0), np.nan))
    return out


def _panel(rows: list[tuple[str, np.ndarray, float, str]], title: str, ax: Axis,
           lim: tuple[float, float], seed: int, show_labels: bool, share: str,
           ref_label: str, height: float = 1.0) -> Any:
    labels = [r[0] for r in rows]
    p = B.strip_ci(labels, [r[1] for r in rows], title,
                   Axis(ax.label, ax.unit, lim=lim), seed=seed, ref=0.0, ref_label=ref_label,
                   roles=[r[3] for r in rows])
    nulls = [(i, r[2]) for i, r in enumerate(rows) if np.isfinite(r[2])]
    if nulls:
        p.series.append(Series("scatter", np.array([v for _, v in nulls]),
                               np.array([float(i) for i, _ in nulls]), role="accent",
                               style={"ms": "oo", "size": 1.4}))
    p.series.append(Series("text", np.full(len(rows), lim[1]),
                           np.arange(len(rows), dtype=float),
                           text=[f"n = {int(np.isfinite(r[1]).sum())}" for r in rows],
                           role="faint", style={"anchor": "left"}))   # left of the axis end
    if not show_labels:
        p.y.ticklabels = [""] * len(labels)
    p.sharex = share
    p.height = height
    return p


def _lim(values: list[np.ndarray], extra: list[float]) -> tuple[float, float]:
    """x range of all values, with room at the right end for the n labels."""
    v = np.concatenate([np.asarray(x, float) for x in values] + [np.asarray(extra, float)])
    v = v[np.isfinite(v)]
    lo, hi = (float(v.min()), float(v.max())) if v.size else (-1.0, 1.0)
    span = (hi - lo) or 100.0
    return lo - 0.05 * span, hi + 0.22 * span


@register("jnl_timing_cues", "paper", "Timing cues relative to the motion onset",
          group="dataset")
def jnl_timing_cues(req: ViewRequest) -> Any:
    from biomech.viz import derived

    ctx = req.ctx
    seed = B.seed_of(ctx)
    cues = {t: cue_leads(ctx, t) for t in TIMING_TASKS}
    lags = {t: lag_values(ctx, t) for t in TIMING_TASKS}
    lim_c = _lim([v for t in cues for v, _ in cues[t].values()],
                 [n for t in cues for _, n in cues[t].values()] + [0.0])
    lim_l = _lim([v for t in lags for v, _ in lags[t].values()],
                 [n for t in lags for _, n in lags[t].values()] + [0.0])
    # JNER revision 3.8 (2026-11-10): three rows per panel -- the opener cues with the IMU
    # detection, then the closer cues -- and a legend inside the figure for the open circles
    groups = ((("dig_onset", "dig_rise", "imu"), "Opener cues"),
              (("rel_omega", "rel_alpha", "closer_onset"), "Closer cues"))
    by = {key: (lab, role) for key, lab, role in CUES}
    panels, letters = [], iter("abcdef")
    for keys, gname in groups:
        for k, task in enumerate(TIMING_TASKS):
            rows = [(by[key][0], cues[task][key][0], cues[task][key][1], by[key][1])
                    for key in keys]
            panels.append(_panel(rows, f"({next(letters)}) {gname}, {V.task_label(task)}",
                                 LEAD_AX, lim_c, seed, k == 0, "cue", "motion onset",
                                 height=0.62))
    for k, task in enumerate(TIMING_TASKS):
        rows = [(lab, lags[task][d][0], lags[task][d][1], role) for d, lab, role in LAGS]
        panels.append(_panel(rows, f"({next(letters)}) Lag, {V.task_label(task)}", LAG_AX,
                             lim_l, seed, k == 0, "lag", "no lag", height=0.45))
    nan = np.array([np.nan])
    key = panels[2]          # (c): its left part is empty
    key.series += [Series("scatter", nan, nan, role="context", label="participants"),
                   Series("scatter", nan, nan, role="accent", style={"ms": "d", "size": 1.4},
                          label="median, 95 % CI"),
                   Series("scatter", nan, nan, role="accent", style={"ms": "oo", "size": 1.4},
                          label="shift null")]
    key.legend = "inside:north west"
    try:
        fam = int(derived.table(ctx, "j18_timing")["family_size"].iloc[0])
    except (FileNotFoundError, IndexError, KeyError):
        fam = 0
    spec = FigureSpec("jnl_timing_cues", "Timing cues relative to the motion onset", panels,
                      rows=3, cols=2, width="text",
                      height_in=B.rows_height(len(CUES), per=0.24),
                      sources=["sensitivity:c2_symmetric_window", "timing:c2e_release",
                               "timing:c2e_cycles", "derived:j18_timing",
                               "derived:j18_symmetric"])
    return attach(spec, CaptionSpec(
        "(a to d) EMG leads before the motion onset, (a, b) of the opener cues with the time a "
        "causal IMU detector fires and (c, d) of the closer cues (digastric and closer onsets: "
        "driving-burst rule, symmetric ±0.4 s window; rise of the cycle-averaged digastric "
        "envelope; closer release to the angular-velocity and angular-acceleration onsets). "
        "(e, f) Lag of the peak correlation between the EMG and the acceleration it drives "
        "(> 0: EMG first)",
        [("dots", "participants"), ("diamond", "median with its 95 % BCa CI"),
         ("open circle", "shift null"), ("vermilion", "digastric"), ("blue", "closers"),
         ("grey", "IMU")],
        "the number of participants of each row is given at its end",
        f"tests in the text, Holm-adjusted over the {fam} timing tests of this analysis"
        if fam else "", "",
        "exploratory (planned); definitions fixed in the analysis plan before computing"))
