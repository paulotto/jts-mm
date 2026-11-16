# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Journal figure and table set (Springer profile, 85 / 170 mm).

Thin paper views over the stored tables; the forms come from ``views.blocks``. The ICORR
views ``icorr_setup``, ``icorr_hinge`` and ``icorr_imu`` are reused unchanged (listed in
``paper_journal.yaml``); the journal adds:

* ``jnl_requirements``: the requirements as a graphic -- incisal workspace per participant
  with the 95th percentile, the peak opening velocity ECDF with its P95 [CI], the optically
  confirmed movement bandwidth per participant, and a latency budget on one ms axis;
* ``jnl_c2``: onset-aligned opener and closer envelopes, the per-participant opening lead
  under both onset rules (pre-specified first onset, post hoc driving burst; 3-channel-cycle
  minimum, Holm), and the specification curve of the lead over detector, motion source and
  onset rule;
* ``jnl_x3``: the causal closer-release detector (exploratory): detection vs chance per
  participant, offline (causal chain) vs stream anticipation and false releases, and the
  parameter trade-off;
* ``jnl_decoding``: EMG + IMU minus IMU per participant without and with calibration,
  the few-shot calibration curve, per-class recall and block permutation importance;
* ``jnl_online``: simulated real-time decoding (exploratory): stream vs offline balanced
  accuracy, decision latency ECDF, false activations and compute per chunk;
* ``jnl_robustness``: processing choices as a standardized specification curve, primary
  vs sensitivity runs, and the plausible-condyle subset of the C1 outcomes;
* ``jnl_table_decoding`` (Table III) and ``jnl_table_robustness`` (Table IV).

Views compute nothing heavy and no test: medians and IQRs of stored participant values; every
p value is read from the stored tables (the few-shot EMG + IMU vs IMU tests are rows
``fewshot|svm|emg_imu_vs_imu|k=<k>`` of ``ml_eval.tests``). Participants appear as pseudonyms
only.
"""

from __future__ import annotations

import re

import warnings
from typing import Any

import numpy as np
import pandas as pd

from biomech.labels.auto_labels import CLASSES
from biomech.results import COHORT_UNIT
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.style import MINUS
from biomech.viz.views import blocks as B
from biomech.viz.views import forms
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.results_timing import release_setting

RULES = (("first", "first onset", "pre-specified"), ("drive", "driving burst", "post hoc"))
LEAD_SPEC = "detector=tkeo|mech=lead_ms|onset={}"
BANDWIDTH_TASKS = ("open_close", "chewing", "free_motion")
FSET = V.labels("feature_set")
MODE = V.labels("calibration_mode", "short")
CLASS_LABEL = ("rest", "open", "close", "protrude", "retrude", "left", "right")
VARIANT_SHORT = V.labels("sensitivity_variant", "short")
RULE_SHORT = V.labels("rule", "short")
VARIANT_LABEL = V.labels("sensitivity_variant")
DET = V.labels("detector")
MECH = V.labels("mech_source", "short")
RULE = V.labels("rule")
LEAD_AX = Axis("Lead before onset", V.MSEC)
BA_AX = Axis("Balanced accuracy", lim=(0.0, 1.0))


# -- helpers ------------------------------------------------------------------------------------

def series(items: list[str], conj: str = "and") -> str:
    """A list in reader text with the serial comma: "a", "a and b", "a, b, and c"."""
    items = [str(x) for x in items]
    if len(items) <= 2:
        return f" {conj} ".join(items)
    return ", ".join(items[:-1]) + f", {conj} " + items[-1]

def _t(ctx, stage: str, name: str) -> pd.DataFrame:
    """A stored table, empty when the stage is absent (optional plugins, older snapshots)."""
    try:
        if stage not in ctx.store.stages():
            return pd.DataFrame()
        return ctx.table(stage, name)
    except KeyError:                              # stage not registered (plugin missing)
        return pd.DataFrame()


def _arr(ctx, stage: str, unit: str) -> dict:
    try:
        return ctx.arrays(stage, unit) if stage in ctx.store.stages() else {}
    except KeyError:
        return {}


def _sel(df: pd.DataFrame, **kw: Any) -> pd.DataFrame:
    for k, v in kw.items():
        if df.empty or k not in df:
            return df.iloc[:0]
        df = df[df[k] == v]
    return df


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _kappa_min(lab_agree: pd.DataFrame) -> str:
    """'= 1.00' when every kappa is 1 (kappa cannot exceed 1), else '≥ <min>'."""
    k = _f(lab_agree.kappa.min())
    return f"= {k:.2f}" if round(k, 2) >= 1 else f"≥ {k:.2f}"


def _n(v: float, d: int = 0) -> str:
    """A number with a real minus; '--' for NaN."""
    if not np.isfinite(v):
        return "--"
    s = f"{v:.{d}f}"
    return s.replace("-", MINUS) if s.startswith("-") else s


def _p(p: float) -> str:
    return "--" if not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def _peq(p: float) -> str:
    """'= 0.023' or '< 0.001' (to follow a 'p')."""
    t = _p(p)
    return t if t.startswith("<") else f"= {t}"


def _ci(lo: float, hi: float, d: int = 0) -> str:
    return f"[{_n(lo, d)}, {_n(hi, d)}]" if np.isfinite(lo) and np.isfinite(hi) else ""


def _med_iqr(v: Any) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (np.nan,) * 3
    q1, m, q3 = np.percentile(v, [25, 50, 75])
    return float(m), float(q1), float(q3)


def _min_cycles(ctx) -> int:
    return int(ctx.config.params("stats_timing").get("min_cycles", 3))


def _excl(row: dict, short: bool = False) -> str:
    k = row.get("excl", np.nan)
    if not np.isfinite(k):
        return "excl. not stored" if short else "exclusions not stored"
    return f"{int(k)} excl." if short else f"{int(k)} excluded"


def _c2_rows(ctx, task: str = "open_close") -> dict[str, dict[str, float]]:
    """Cohort opening lead (digastric TKEO, earliest mechanical onset) per onset rule:
    median, CI, n, participants excluded by the channel-cycle minimum, Holm p of the family."""
    tc = _sel(_t(ctx, "stats_timing", "timing_cohort"), task=task, direction="opening",
              detector="tkeo", metric="lead_ms")
    out: dict[str, dict[str, float]] = {}
    for rule, _, _ in RULES:
        r = _sel(tc, rule=rule)
        if r.empty:
            out[rule] = {k: np.nan for k in ("median", "ci_lo", "ci_hi", "n", "excl", "p_holm",
                                              "q1", "q3")}
            continue
        r = r.iloc[0]
        out[rule] = {"median": _f(r["median"]), "ci_lo": _f(r.ci_lo), "ci_hi": _f(r.ci_hi),
                     "n": _f(r.n), "excl": _f(r.get("n_excluded_min_cycles", np.nan)),
                     "p_holm": _f(r.get("p_holm", np.nan)), "q1": _f(r.q1), "q3": _f(r.q3),
                     "family": str(r.get("family", ""))}
    return out


def _c2_text(c: dict[str, float], k: int) -> str:
    """'108 ms [74, 248], n = 9 with >= 3 usable channel-cycles (3 excluded), Holm p = 0.023'."""
    return (f"{_n(c['median'])} ms {_ci(c['ci_lo'], c['ci_hi'])}, n = "
            f"{_n(c['n'])} with ≥ {k} usable channel-cycles ({_excl(c)}), Holm p {_peq(c['p_holm'])}")


def _c2_verdict(c2: dict[str, dict[str, float]]) -> str:
    """The C2 conclusion as the stored Holm p values give it (never fixed text, E37)."""
    sig = {r: _f(c2.get(r, {}).get("p_holm", np.nan)) < 0.05 for r in ("first", "drive")}
    if sig["first"] and not sig["drive"]:
        return ("only the pre-specified rule is significant after Holm, so the lead depends "
                "on which burst is taken as the onset and the lead hypothesis is not "
                "confirmed")
    if sig["first"] and sig["drive"]:
        return "both rules are significant after Holm"
    if sig["drive"]:
        return ("only the driving-burst rule is significant after Holm; the pre-specified "
                "rule is not, so the lead hypothesis is not confirmed")
    return "neither rule is significant after Holm; the lead hypothesis is not confirmed"


def _stride(ctx) -> float:
    """Decoder window stride (s): k calibration windows per movement = k x stride seconds."""
    return float((ctx.config.params("ml_windows").get("window") or {}).get("stride_s", 0.05))


def _cal_s(ctx, k: int) -> str:
    """Calibration amount k (windows per movement) as seconds at the window stride."""
    return f"{k * _stride(ctx):g} s"


def _requirement(rq: pd.DataFrame, key: str, values: Any) -> tuple[float, float, float, bool]:
    """(design value, CI low, CI high, from the requirements stage). Without the stage the
    value is the 95th percentile of ``values`` (pandas linear quantile) and has no CI."""
    r = rq[rq.key == key] if not rq.empty and "key" in rq else pd.DataFrame()
    if not r.empty:
        r = r.iloc[0]
        return _f(r["value"]), _f(r.ci_lo), _f(r.ci_hi), True
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 95)) if v.size else np.nan), np.nan, np.nan, False


def _empty(p: Panel, msg: str = "no data") -> Panel:
    p.note = msg
    return p


# -- requirements -------------------------------------------------------------------------------
def _latency_rows(ctx) -> tuple[list[tuple[str, float, float, float]], int, dict, list[float]]:
    """The latency budget: (label, median, CI low, CI high) in ms relative to the mechanical
    opening onset in open-close (negative: before), the minimum cycles k, the C2 rows and the
    participants behind each row."""
    k = _min_cycles(ctx)
    c2 = _c2_rows(ctx)
    ens = _sel(_t(ctx, "stats_timing", "ensemble_cohort"), task="open_close",
               direction="opening", role="antagonist")
    rel = _t(ctx, "stats_timing", "release_cohort")
    rel = rel[(rel.task == "open_close") & rel.primary.astype(bool)] if not rel.empty else rel
    ms = _sel(_t(ctx, "ml_eval", "summary"), protocol="loso", model="svm",
              feature_set="emg_imu", metric="latency_ms_median")
    on = _sel(_t(ctx, "ml_online", "summary"), feature_set="emg_imu", mode="none")

    def before(df: pd.DataFrame) -> tuple[float, float, float]:
        if df.empty:
            return (np.nan,) * 3
        r = df.iloc[0]
        return -_f(r["median"]), -_f(r.ci_hi), -_f(r.ci_lo)

    lat = [("closer release (envelopes)", *before(ens)),
           ("causal release decision", *before(rel)),
           ("EMG onset, first-onset rule", -c2["first"]["median"], -c2["first"]["ci_hi"],
            -c2["first"]["ci_lo"]),
           ("EMG onset, driving-burst rule", -c2["drive"]["median"], -c2["drive"]["ci_hi"],
            -c2["drive"]["ci_lo"]),
           ("decoder decision, offline", *((_f(ms["median"].iloc[0]), _f(ms.ci_lo.iloc[0]),
                                            _f(ms.ci_hi.iloc[0])) if not ms.empty
                                           else (np.nan,) * 3)),
           ("decoder decision, stream", *(_med_iqr(on.latency_ms_median) if not on.empty
                                          and "latency_ms_median" in on else (np.nan,) * 3))]
    def first(df: pd.DataFrame, c: str) -> float:
        return _f(df[c].iloc[0]) if not df.empty and c in df else np.nan

    ns = [first(ens, "n"), first(rel, "n_participants"), c2["first"]["n"], c2["drive"]["n"],
          first(ms, "n"), float(on.latency_ms_median.notna().sum()) if not on.empty
          and "latency_ms_median" in on else np.nan]
    return lat, k, c2, ns


def _latency_caption(lat: list, ns: list[float], c2: dict, k: int) -> dict[str, str]:
    """The caption parts of the latency budget: what it shows, its participants, the EMG
    onset statistics with the C2 verdict and the evidence of each time."""
    return {"shows": "the closer release of the onset-aligned envelopes and the decision of a "
                     "causal closer-release detector; the digastric EMG onset under the "
                     "first-onset and the driving-burst rule; and the intent decoder's first "
                     "correct decision after the label onset (EMG + IMU, unseen participant), "
                     "offline and in simulated real time (median and IQR over participants)",
            "n": "participants per time: " + "; ".join(f"{r[0]} {_n(m)}"
                                                        for r, m in zip(lat, ns)),
            "statistic": "EMG onset lead, first-onset rule: " + _c2_text(c2["first"], k)
                         + "; driving-burst rule: " + _c2_text(c2["drive"], k) + ": "
                         + _c2_verdict(c2),
            "evidence": "closer release, causal detector and decoder exploratory; EMG onset "
                        "under the first-onset rule pre-specified, under the driving-burst "
                        "rule post hoc"}


def _budget_figure(ctx) -> FigureSpec:
    """The latency budget alone: each time as a bar from the motion onset (0) to its cohort
    median, with the 95 % CI; bars left of 0 are lead time a controller gains, bars right of
    0 are delay it must absorb."""
    lat, k, c2, ns = _latency_rows(ctx)
    p = Panel("Latency budget", Axis("Time from motion onset", V.MSEC), Axis(""), [],
              legend="none")
    for i, (_lab, med, lo, hi) in enumerate(lat):
        if np.isfinite(med):
            p.series.append(Series("fill", np.array([0.0, med, med, 0.0]),
                                   np.array([i - 0.3, i - 0.3, i + 0.3, i + 0.3]),
                                   role="cat:0" if med < 0 else "cat:1", style={"alpha": 0.35}))
            p.series.append(Series("scatter", np.array([med]), np.array([float(i)]),
                                   role="accent", style={"ms": "d", "size": 1.3}))
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                   role="accent", style={"width": 1.2}))
    p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
    p.y.ticks = [float(i) for i in range(len(lat))]
    p.y.ticklabels = [r[0] for r in lat]
    p.y.reverse = True
    if not any(np.isfinite(r[1]) for r in lat):
        _empty(p)
    spec = FigureSpec("jnl_requirements", "Requirements from asymptomatic jaw function", [p],
                      width="column", height_in=B.rows_height(len(lat), per=0.24, extra=0.6),
                      sources=["stats_timing:timing_cohort", "stats_timing:ensemble_cohort",
                               "stats_timing:release_cohort", "ml_eval:summary",
                               "ml_online:summary"])
    lc = _latency_caption(lat, ns, c2, k)
    return attach(spec, CaptionSpec(
        "latency budget of an intent-driven jaw exoskeleton in open-close: times relative to "
        "the mechanical opening onset, cohort medians with 95 % CIs, of " + lc["shows"],
        [("dashed line", "mechanical opening onset"),
         ("diamond, bar from the onset", "cohort median"),
         ("blue bars (left of the onset)", "lead time available to a controller"),
         ("orange bars (right)", "delay it must absorb")],
        lc["n"], lc["statistic"], evidence=lc["evidence"]))


REQ_FORMS = ("envelope_planes", "panels", "ecdf", "budget")


def _hull2(P: np.ndarray) -> np.ndarray | None:
    """Closed outline (convex hull) of 2-D points, or None for fewer than 3 / collinear."""
    from scipy.spatial import ConvexHull, QhullError

    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 3:
        return None
    try:
        h = ConvexHull(P)
    except (QhullError, ValueError):
        return None
    v = P[h.vertices]
    return np.vstack([v, v[:1]])


def _union_outline(polys: list[np.ndarray], res: float = 0.2,
                   keep: Any = None) -> list[np.ndarray]:
    """Closed outlines of the union of the polygons ``polys`` (each (k, 2), closed), from a
    ``res``-mm raster (no polygon library in the environment); ``keep(X, Y) -> bool mask``
    restricts the area (e.g. outside the design box). Outermost rings, largest first."""
    import contourpy
    from matplotlib.path import Path as MPath

    if not polys:
        return []
    allp = np.vstack(polys)
    lo, hi = allp.min(axis=0) - 2 * res, allp.max(axis=0) + 2 * res
    xs = np.arange(lo[0], hi[0] + res, res)
    ys = np.arange(lo[1], hi[1] + res, res)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    inside = np.zeros(len(pts), bool)
    for P in polys:
        inside |= MPath(P).contains_points(pts, radius=1e-9)
    m = inside.reshape(X.shape)
    if keep is not None:
        m &= keep(X, Y)
    if not m.any():
        return []
    lines = contourpy.contour_generator(xs, ys, m.astype(float)).lines(0.5)
    rings = [np.asarray(L, float) for L in lines if len(L) >= 4]

    def area(R: np.ndarray) -> float:
        return 0.5 * abs(np.dot(R[:, 0], np.roll(R[:, 1], 1)) - np.dot(R[:, 1], np.roll(R[:, 0], 1)))

    rings = [np.vstack([R, R[:1]]) for R in rings if area(R) > 4 * res * res]
    return sorted(rings, key=area, reverse=True)


PLANES = (("sagittal", (0, 2), "Anterior", "Superior"), ("frontal", (1, 2), "Left", "Superior"))


def envelope_outlines(ctx) -> dict[str, Any]:
    """Per plane the participants' convex outlines of the incisal positions (all five tasks,
    from the rest position; ``kin_r8.incisor_cloud``), the cohort envelope (their union), the
    part of it outside the P95 design box, the box (``kin_r8.design_box``) and per participant
    the share of positions outside the box. The data of :func:`jnl_requirements` (a, b) and of
    the additional data file ``workspaceEnvelope`` (JNER revision 5.7)."""
    from biomech.viz.views.kin_r8 import _outside, _requirements, design_box, incisor_cloud

    rq = _requirements(ctx)
    box = design_box(rq)
    per: dict[str, dict[str, np.ndarray]] = {}
    out: list[float] = []
    for pid in ctx.pids():
        try:
            a = ctx.arrays("kinematics", pid)
        except (KeyError, FileNotFoundError, ValueError):
            continue
        P, _ = incisor_cloud(a)
        if len(P) < 5:
            continue
        out.append(_outside(P, box))
        per[pid] = {}
        for plane, (i, j), _, _ in PLANES:
            o = _hull2(P[:, [i, j]])
            if o is not None:
                per[pid][plane] = o
    cohort, outside = {}, {}
    for plane, (i, j), _, _ in PLANES:
        polys = [d[plane] for d in per.values() if plane in d]
        cohort[plane] = _union_outline(polys)
        if box is not None:
            lo, hi = box

            def keep(X, Y, i=i, j=j, lo=lo, hi=hi):
                return ~((X >= lo[i]) & (X <= hi[i]) & (Y >= lo[j]) & (Y <= hi[j]))

            outside[plane] = _union_outline(polys, keep=keep)
        else:
            outside[plane] = []
    v = np.asarray([x for x in out if np.isfinite(x)], float)
    back = _f(rq["value"].get("posterior_excursion_mm", np.nan)) if not rq.empty \
        and "value" in rq else np.nan
    return {"per": per, "cohort": cohort, "outside": outside, "box": box, "n": len(out),
            "median": float(np.median(v)) if v.size else np.nan,
            "max": float(v.max()) if v.size else np.nan, "back_mm": back}


def _envelope_panels(ctx) -> tuple[Panel, Panel, dict[str, Any]]:
    """(a) sagittal and (b) frontal projections (JNER revision 3.2, 2026-11-10): the
    participants' outlines thin and light, the cohort envelope (their union) as a filled area
    with an outline in the summary colour, the part of it outside the P95 design box shaded,
    the box dashed, the rest position as a marker and (a) the posterior excursion as an
    arrow."""
    env = envelope_outlines(ctx)
    box = env["box"]
    ps = Panel("(a) Sagittal envelope", Axis("Anterior", V.MM), Axis("Superior", V.MM), [],
               equal=True, legend="none")
    pf = Panel("(b) Frontal envelope", Axis("Left", V.MM), Axis("Superior", V.MM), [],
               equal=True, legend="none")
    for p, (plane, (i, j), _, _) in zip((ps, pf), PLANES):
        for d in env["per"].values():
            if plane in d:
                p.series.append(Series("line", d[plane][:, 0], d[plane][:, 1], role="faint",
                                       style={"width": 0.8}))
        for k, R in enumerate(env["cohort"][plane]):
            p.series.append(Series("fill", R[:, 0], R[:, 1], role="summary:fill",
                                   style={"alpha": 0.14}))
            p.series.append(Series("line", R[:, 0], R[:, 1], role="summary",
                                   style={"width": 1.3}))
        for R in env["outside"][plane]:
            p.series.append(Series("fill", R[:, 0], R[:, 1], role="summary:out",
                                   style={"alpha": 0.55}))
        if box is not None:
            lo, hi = box
            x = np.array([lo[i], hi[i], hi[i], lo[i], lo[i]])
            y = np.array([lo[j], lo[j], hi[j], hi[j], lo[j]])
            p.series.append(Series("line", x, y, role="threshold", style={"width": 1.2}))
        p.series.append(Series("scatter", np.array([0.0]), np.array([0.0]), role="accent",
                               style={"ms": "x", "size": 1.6}))
    if box is not None:
        lo, hi = box
        # direct labels instead of a legend below (3.0): box, posterior excursion
        for p, i in ((ps, 0), (pf, 1)):
            p.series.append(Series("text", np.array([hi[i] - 0.4]), np.array([lo[2] + 1.2]),
                                   text=["P95 box"], style={"anchor": "left"}))
        top = hi[2] + 4.5
        if np.isfinite(env["back_mm"]) and env["back_mm"] > 0:
            ps.series.append(Series("arrow", np.array([0.0]), np.array([top]),
                                    d=np.array([[-env["back_mm"], 0.0]]), role="accent",
                                    style={"width": 1.1}))
            ps.series.append(Series("text", np.array([0.5]), np.array([top]),
                                    text=["posterior excursion"], style={"anchor": "right"}))
        # no "outside" label: it sat inside the box next to the vermilion area; the caption
        # names that area (2026-11-13)
        for p in (ps, pf):
            p.y.lim = (lo[2] - 2.5, top + 3.0)
    if not env["per"]:
        _empty(ps)
        _empty(pf)
    return ps, pf, env


# Every P95 row of tabRequirements, per participant (JNER revision 3.2): (key, reader name,
# participant values from stats_kinematics.subjects or the bandwidth rule)
REQ_ROWS_WORKSPACE = (("opening_mm", "Opening"), ("protrusion_mm", "Protrusion"),
                      ("laterotrusion_mm", "Laterotrusion"),
                      ("posterior_excursion_mm", "Posterior excursion"),
                      ("translation_mm", "Condyle translation"))
REQ_ROWS_MOTION = (("opening_angle_deg", "Opening rotation"),
                   ("peak_vel_dps", "Opening velocity"), ("close_vel_dps", "Closing velocity"),
                   ("free_peak_open_vel_dps", "Free-motion velocity"),
                   ("peak_acc_dps2", "Angular acceleration"),
                   ("bandwidth_hz", "Movement bandwidth"))


def requirement_participant_values(ctx) -> dict[str, np.ndarray]:
    """Per P95 requirement key the participant values its design value is the P95 of (the same
    columns as :func:`biomech.stats.requirements.requirements_table`)."""
    from biomech.stats.requirements import bandwidth_per_participant

    s = _t(ctx, "stats_kinematics", "subjects")
    out: dict[str, np.ndarray] = {}

    def col(c: str) -> np.ndarray:
        return s[c].to_numpy(float) if c in s else np.empty(0)

    if not s.empty:
        out["opening_mm"] = col("max_observed_opening_mm")
        out["protrusion_mm"] = col("protrusion_mm")
        if {"laterotrusion_left_mm", "laterotrusion_right_mm"} <= set(s):
            out["laterotrusion_mm"] = s[["laterotrusion_left_mm", "laterotrusion_right_mm"]] \
                .max(axis=1).to_numpy(float)
        out["opening_angle_deg"] = col("opening_angle_deg")
        if {"condyle_left_disp_mm", "condyle_right_disp_mm"} <= set(s):
            out["translation_mm"] = s[["condyle_left_disp_mm", "condyle_right_disp_mm"]] \
                .max(axis=1).to_numpy(float)
        out["peak_vel_dps"] = col("peak_open_vel_max_dps")
        out["close_vel_dps"] = col("peak_close_vel_max_dps")
        out["free_peak_open_vel_dps"] = col("free_peak_open_vel_dps")
        out["peak_acc_dps2"] = col("peak_acc_dps2")
        out["posterior_excursion_mm"] = col("posterior_excursion_mm")
    bw = _t(ctx, "imu", "bandwidth")
    if not bw.empty:
        out["bandwidth_hz"] = bandwidth_per_participant(bw).to_numpy(float)
    return out


def _req_rows_panel(ctx, rows: tuple, title: str) -> tuple[Panel, int]:
    """One row per requirement: participants (grey) relative to the design value (P95 = 100 %),
    the P95 (summary diamond) with the line from its one-sided lower 95 % bound; the design
    value in the row label."""
    from biomech.viz.views.paper import UNIT_TEXT, _num, req_decimals

    rq = _t(ctx, "requirements", "requirements")
    vals = requirement_participant_values(ctx)
    p = Panel(title, Axis("Relative to design value", V.PCT), Axis(""),
              [], legend="none", sharex="req_rel")
    labels, k = [], 0
    for key, name in rows:
        r = rq[rq.key == key] if not rq.empty else rq
        v = vals.get(key, np.empty(0))
        if r.empty or not np.isfinite(_f(r.value.iloc[0])):
            continue
        r = r.iloc[0]
        val, lo = _f(r.value), _f(r.ci_lo)
        u = UNIT_TEXT.get(str(r.unit), f" {r.unit}")
        labels.append(f"{name}, {_num(val, req_decimals(val))}{u}")
        x = 100.0 * v[np.isfinite(v)] / val
        if x.size:
            jit = np.linspace(-0.17, 0.17, x.size) if x.size > 1 else np.zeros(1)
            p.series.append(Series("scatter", x, np.full(x.size, float(k)) + jit,
                                   role="context"))
        if np.isfinite(lo):
            p.series.append(Series("line", np.array([100.0 * lo / val, 100.0]),
                                   np.array([k, k], float), role="summary",
                                   style={"width": 1.6}))
        # the size of the P95 diamond of panel (c) (blocks.ecdf), so the figure has one
        # P95 marker (2026-11-13: at 1.3 it was smaller than the grey dots)
        p.series.append(Series("scatter", np.array([100.0]), np.array([float(k)]),
                               role="summary", style={"ms": "d", "size": 1.6}))
        k += 1
    p.series.append(Series("vline", x=np.array([100.0]), role="reference"))
    p.y.ticks, p.y.ticklabels = [float(i) for i in range(len(labels))], labels
    p.y.lim = (-0.6, len(labels) - 0.4) if labels else None
    p.y.reverse = True
    if not labels:
        _empty(p)
    return p, k


@register("req_envelope_data", "paper", "Workspace envelope coordinates (additional file)",
          group="dataset")
def req_envelope_data(req: ViewRequest) -> TableSpec:
    """The cohort workspace envelope of :func:`jnl_requirements` (a, b) as coordinates (JNER
    revision 5.7, 2026-11-10): per plane the outline of the cohort envelope (union of the
    participants' convex outlines), the part outside the P95 design box, the box and each
    participant's outline, in mm from the rest position (maxillary frame: anterior, left,
    superior). Written as ``additional/workspaceEnvelope.csv``."""
    env = envelope_outlines(req.ctx)
    rows: list[dict[str, Any]] = []
    pseud = getattr(req.ctx, "pseudonym", None)

    def add(plane: str, what: str, ring: int, R: np.ndarray) -> None:
        for vtx, (h, v) in enumerate(R):
            rows.append({"plane": plane, "outline": what, "ring": ring, "vertex": vtx,
                         "h": round(float(h), 2), "v": round(float(v), 2)})

    for plane, (i, j), _, _ in PLANES:
        for k, R in enumerate(env["cohort"][plane]):
            add(plane, "cohort envelope", k, R)
        for k, R in enumerate(env["outside"][plane]):
            add(plane, "cohort envelope outside the P95 box", k, R)
        if env["box"] is not None:
            lo, hi = env["box"]
            add(plane, "P95 design box", 0, np.array(
                [[lo[i], lo[j]], [hi[i], lo[j]], [hi[i], hi[j]], [lo[i], hi[j]], [lo[i], lo[j]]]))
        for pid, d in env["per"].items():
            if plane in d:
                add(plane, f"participant P{pseud(pid)}" if callable(pseud) else pid, 0,
                    d[plane])
    cols = [Column("plane", "Plane", align="l"), Column("outline", "Outline", align="l"),
            Column("ring", "Ring", decimals=0), Column("vertex", "Vertex", decimals=0),
            Column("h", "Horizontal (sagittal: anterior, frontal: left)", "\\milli\\metre"),
            Column("v", "Vertical (superior)", "\\milli\\metre")]
    return TableSpec("req_envelope_data", "Incisal workspace envelope of the cohort as "
                     "coordinates, in mm from the rest position (maxillary frame). Sagittal "
                     "plane: anterior and superior. Frontal plane: left and superior. The cohort "
                     "envelope is the union of the participants' convex outlines of the incisal "
                     "positions over the five movement tasks.", cols, rows, width="text",
                     sources=["kinematics:arrays", "requirements:requirements"])


@register("jnl_requirements", "paper", "Requirements from asymptomatic jaw function (journal)",
          choices={"form": REQ_FORMS}, form="envelope_planes")
def jnl_requirements(req: ViewRequest) -> FigureSpec:
    """Workspace, peak velocity and movement bandwidth. Form ``envelopes`` (default,
    2026-11-05): the workspace as sagittal and frontal incisal envelopes of every participant
    with the P95 design box; ``panels`` / ``ecdf``: the workspace as per-participant ranges
    plus (d) the latency budget (the old figure); ``budget``: the latency budget alone."""
    ctx = req.ctx
    form = req.options.get("form", "envelope_planes")
    if form == "budget":
        return _budget_figure(ctx)
    rq = _t(ctx, "requirements", "requirements")
    subj = _t(ctx, "stats_kinematics", "subjects")
    n = len(subj)
    staged = []
    # The design rows carry a one-sided lower 95 % bound since E31 (no upper end): name the
    # interval as stored (the legend promised a two-sided CI that was never drawn).
    one_sided = (not rq.empty and "ci_hi" in rq and rq.key.eq("opening_mm").any()
                 and not np.isfinite(_f(rq.loc[rq.key == "opening_mm", "ci_hi"].iloc[0])))
    bound = "one-sided lower 95 % bound" if one_sided else "95 % CI"
    mlab = "P95 and its lower bound" if one_sided else "P95 [95 % CI]"
    # (a) incisal workspace per participant with the 95th percentile
    rows = [("opening_mm", "opening", "max_observed_opening_mm"),
            ("protrusion_mm", "protrusion", "protrusion_mm"),
            ("laterotrusion_mm", "laterotrusion", None)]
    vals, marks = [], []
    for key, _, col in rows:
        if subj.empty:
            v = np.empty(0)
        elif col is None:
            v = subj[["laterotrusion_left_mm", "laterotrusion_right_mm"]].max(axis=1).to_numpy(
                float)
        else:
            v = subj[col].to_numpy(float)
        val, lo, hi, st = _requirement(rq, key, v)
        staged.append(st)
        vals.append(v)
        marks.append((val, lo, hi))
    if form == "ecdf":
        pa = B.ecdf({r[1]: v for r, v in zip(rows, vals)}, "(a) Incisal workspace",
                    Axis("Largest incisal displacement", V.MM),
                    yaxis=Axis("Participants at or below", V.PCT, lim=(0.0, 104.0)))
        for i, (val, lo, hi) in enumerate(marks):
            if np.isfinite(lo) and np.isfinite(hi):
                pa.series.append(Series("line", np.array([lo, hi]), np.array([95.0, 95.0]),
                                        role=f"cat:{i}", style={"width": 1.4}))
            if np.isfinite(val):
                pa.series.append(Series("scatter", np.array([val]), np.array([95.0]),
                                        role=f"cat:{i}", style={"ms": "d", "size": 1.6}))
        pa.series.append(Series("hline", y=np.array([95.0]), role="reference"))
    else:
        pa = B.strip_mark([r[1] for r in rows], vals, "(a) Incisal workspace",
                          Axis("Largest incisal displacement", V.MM), marks=marks,
                          mark_label=mlab)
    if subj.empty:
        _empty(pa)
    # (b) peak opening angular velocity: ECDF with the 95th percentile
    vel = subj.peak_open_vel_max_dps.to_numpy(float) if "peak_open_vel_max_dps" in subj \
        else np.empty(0)
    val, lo, hi, st = _requirement(rq, "peak_vel_dps", vel)
    staged.append(st)
    # headroom above 100 %: the P95 note (top left) sits above the 95 % line, not on it
    # JNER revision 3.0 / 3.2: participants grey, the P95 in the summary colour, the note
    # names the lower bound as well
    pb = B.ecdf(vel, "(b) Peak opening velocity", Axis("Peak angular velocity", V.DPS),
                yaxis=Axis("Participants at or below", V.PCT, lim=(0.0, 118.0)),
                roles=["context"], mark_role="summary",
                mark={"p": 95, "value": val, "lo": lo, "hi": hi,
                      "note": (f"P95 {_n(val)}°/s, lower bound {_n(lo)}°/s" if one_sided
                               and np.isfinite(lo) else
                               f"P95 {_n(val)}°/s {_ci(lo, hi)}".strip())})
    pb.y.ticks = [0.0, 25.0, 50.0, 75.0, 100.0]
    # (c) optically confirmed movement bandwidth per task and the largest per participant
    bw = _t(ctx, "imu", "bandwidth")
    ok = bw[bw.task.isin(BANDWIDTH_TASKS)] if not bw.empty and "f95_coh_hz" in bw else \
        pd.DataFrame(columns=["pid", "task", "f95_coh_hz", "noise_limited"])
    nl = ok.noise_limited.astype(bool) if not ok.empty else pd.Series(dtype=bool)
    per = ok[~nl].groupby("pid").f95_coh_hz.max() if not ok.empty else pd.Series(dtype=float)
    val, lo, hi, st = _requirement(rq, "bandwidth_hz", per.to_numpy(float))
    staged.append(st)
    labels = [V.task_label(t) for t in BANDWIDTH_TASKS] + ["largest per participant"]
    used = [ok[(ok.task == t) & ~nl].f95_coh_hz.to_numpy(float) if not ok.empty
            else np.empty(0) for t in BANDWIDTH_TASKS] + [per.to_numpy(float)]
    flagged = [ok[(ok.task == t) & nl].f95_coh_hz.to_numpy(float) if not ok.empty
               else np.empty(0) for t in BANDWIDTH_TASKS] + [np.empty(0)]
    n_nl = int(nl.sum()) if not ok.empty else 0
    if form == "ecdf":
        pc = B.ecdf(per.to_numpy(float), "(c) Movement bandwidth",
                    Axis("Largest optically confirmed bandwidth (f95)", V.HZ),
                    yaxis=Axis("Participants at or below", V.PCT, lim=(0.0, 104.0)),
                    mark={"p": 95, "value": val, "lo": lo, "hi": hi,
                          "note": f"P95 {val:.2f} Hz {_ci(lo, hi, 2)}".strip()})
    else:
        pc = B.strip_mark(labels, used, "(c) Movement bandwidth",
                          Axis("Bandwidth f95", V.HZ),
                          marks=[(np.nan,) * 3] * 3 + [(val, lo, hi)],
                          mark_label=mlab, flagged=flagged if n_nl else None,
                          flagged_label="noise-limited, not used", mark_role="summary")
        if form == "envelope_planes" and np.isfinite(val):
            # direct label instead of a legend below the panel (3.0)
            pc.legend = "none"
            pc.series.append(Series("text", np.array([val]), np.array([len(labels) - 1.22]),
                                    text=["P95"], style={"anchor": "above"}))
    if ok.empty:
        _empty(pc)
    if form == "envelope_planes":
        return _requirements_envelopes(ctx, pb, pc, n, staged, bound, one_sided, n_nl)
    # (d) latency budget on one ms axis (negative: before the mechanical opening onset)
    lat, k, c2, ns = _latency_rows(ctx)
    pd_ = B.forest([r[0] for r in lat], [r[1] for r in lat], [r[2] for r in lat],
                   [r[3] for r in lat], "(d) Latency budget",
                   Axis("Time from motion onset", V.MSEC), ref=0.0, role="accent")
    if not any(np.isfinite(r[1]) for r in lat):
        _empty(pd_)
    lc = _latency_caption(lat, ns, c2, k)
    ecdf = form == "ecdf"
    marks = [("dots", "participants")]
    if ecdf:
        marks += [("diamonds and lines at 95 % in (a)", f"P95 with its {bound}"),
                  ("diamond and dashed lines in (b, c)", "P95"),
                  ("shaded in (b, c)", f"from its {bound} to the P95" if one_sided
                   else f"its {bound}")]
    else:
        marks += [("bar markers and lines in (a, c)", f"P95, the line from its {bound}"
                   if one_sided else f"P95 with its {bound}"),
                  ("diamond and dashed lines in (b)", "P95"),
                  ("shaded in (b)", f"from its {bound} to the P95" if one_sided
                   else f"its {bound}")]
        if n_nl:
            marks.append(("grey triangles in (c)", f"{n_nl} noise-limited participant-tasks "
                                                    "(coherent band below 2 Hz), shown but not "
                                                    "used"))
    marks.append(("dashed line in (d)", "onset"))
    if all(staged):
        marks.append(("brackets and bars", "95 % participant-bootstrap CI"))
    spec = FigureSpec("jnl_requirements", "Requirements from asymptomatic jaw function",
                      [pa, pb, pc, pd_], rows=2, cols=2, width="text", height_in=1.45,
                      sources=["requirements:requirements", "stats_kinematics:subjects",
                               "imu:bandwidth", "stats_timing:timing_cohort",
                               "stats_timing:ensemble_cohort", "stats_timing:release_cohort",
                               "ml_eval:summary", "ml_online:summary"],
                      meta={"requirements_stage": bool(all(staged))})
    return attach(spec, CaptionSpec(
        "requirements derived from the participants; the design value of a capacity is the "
        "95th percentile across participants (P95). (a) "
        + ("Empirical distribution of the largest" if ecdf else "Largest")
        + " incisal opening, protrusion and laterotrusion (larger side) per participant; "
        "lower bounds, as no maximal opening was recorded. (b) "
        "Empirical distribution of the per-participant peak opening angular velocity. (c) "
        "Frequency below which 95 % of the jaw angular-velocity power shared with the optics "
        "lies (coherence-weighted), "
        + ("as the empirical distribution of the largest per participant" if ecdf
           else "per opening task and as the largest per participant")
        + ". (d) Times relative to the mechanical opening onset in open-close (negative: "
        "before), cohort medians with 95 % CIs: " + lc["shows"],
        marks, f"n = {n} participants; (d) " + lc["n"],
        # the (d) test results are in the requirements table; here only the verdict
        "P95 with its one-sided lower 95 % bound; (d) EMG onset lead: " + _c2_verdict(c2)
        + " (tests in the table of intent requirements)",
        evidence="(a) pre-specified description; (c) exploratory; (d) " + lc["evidence"],
        notes=([f"(c) Not used: {n_nl} noise-limited participant-tasks (coherent band below "
                "2 Hz)."] if ecdf and n_nl else [])
        + ([] if all(staged) else ["95 % CIs come from the requirements stage, which this "
                                   "result set does not contain; the percentiles are computed "
                                   "from the participant values."])))


def _requirements_envelopes(ctx, pb: Panel, pc: Panel, n: int, staged: list, bound: str,
                            one_sided: bool, n_nl: int) -> FigureSpec:
    """Form ``envelope_planes`` of :func:`jnl_requirements` (JNER revision 3.2, 2026-11-10):
    (a, b) the incisal envelopes with the cohort envelope, the part outside the P95 box and
    the posterior excursion, (c) the peak velocity, (d) the movement bandwidth, (e, f) every
    P95 requirement of the table relative to its design value (no latency budget: the intent
    side is the table of intent requirements, form ``budget``)."""
    ps, pf, st = _envelope_panels(ctx)
    pb.title = re.sub(r"^\(\w\)\s*", "(c) ", pb.title)
    pc.title = re.sub(r"^\(\w\)\s*", "(d) ", pc.title)
    pb.tag = pc.tag = ""
    pe, _ = _req_rows_panel(ctx, REQ_ROWS_WORKSPACE, "(e) Workspace and translation")
    pf_, _ = _req_rows_panel(ctx, REQ_ROWS_MOTION, "(f) Rotation and speed")
    spec = FigureSpec("jnl_requirements", "Requirements from asymptomatic jaw function",
                      [ps, pf, pb, pc, pe, pf_], rows=3, cols=2, width="text", height_in=1.55,
                      sources=["requirements:requirements", "kinematics:arrays",
                               "stats_kinematics:subjects", "imu:bandwidth"],
                      meta={"requirements_stage": bool(all(staged))})
    ps.height = pf.height = 1.25
    pe.height = pf_.height = 0.95
    out = (f" Outside the box lie a median of {st['median']:.0f} % and up to {st['max']:.0f} % "
           "of a participant's incisal positions, as the envelope is curved, not box-shaped"
           if np.isfinite(st["median"]) else "")
    marks = [("light grey outlines in (a, b)", "each participant's convex outline of the "
                                               "incisal positions over the five movement "
                                               "tasks, from the rest position (cross)"),
             ("blue area", "cohort envelope, inside any participant's outline"),
             ("vermilion area", "part of it outside the design box"),
             ("dashed box", "P95 design workspace, forward by the protrusion, "
                            + ("backward by the posterior incisal excursion (arrow), "
                               if np.isfinite(st["back_mm"]) else "")
                            + "sideways by the laterotrusion and down by the opening"),
             ("grey dots", "participants"),
             ("blue diamonds and bar marker in (c to f)", "P95"),
             ("blue lines in (d to f), shaded range in (c)", "from the lower bound to the P95")]
    if n_nl:
        marks.append(("dark triangles in (d)", f"{n_nl} noise-limited participant-tasks "
                                               "(coherent band below 2 Hz), not used"))
    return attach(spec, CaptionSpec(
        # starts with the first panel: the paper sets its own bold title (paper revision
        # 2026-11-10)
        "(a, b) Incisal workspace as the sagittal "
        "and frontal projection of the incisal envelope with the P95 design box. The design "
        "value of a capacity is the 95th percentile across participants (P95). The workspace "
        "values are lower bounds, as no maximal opening was recorded." + out + ". (c) "
        "Empirical distribution of the per-participant peak opening angular velocity. (d) "
        "Frequency below which 95 % of the jaw angular-velocity power shared with the optics "
        "lies (coherence-weighted), per opening task and as the largest per participant. "
        "(e, f) Every P95 requirement of the requirements table, each participant's value "
        "relative to the design value given in the row label",
        marks, f"n = {n} participants" + (f" ({st['n']} with incisal kinematics)"
                                          if st["n"] != n else ""),
        "P95 with its one-sided lower 95 % bound",
        evidence="(a, b, e) descriptive, the outside share and the posterior excursion post "
                 "hoc. (d) and the movement bandwidth row exploratory",
        notes=[] if all(staged) else ["95 % CIs come from the requirements stage, which "
                                      "this result set does not contain. The percentiles "
                                      "are computed from the participant values."]))


# -- C2 -------------------------------------------------------------------------------------------
def _envelopes(ctx, task: str = "open_close") -> tuple[Panel, int]:
    """Per-participant median envelopes around the opening onset (dB re the configured
    pre-movement level, ``timing.ensemble.base_ms``), cohort median and IQR of the openers
    (agonists) and closers (antagonists); the panel and the participants behind the larger
    group."""
    from biomech.viz.views.dashboards import _ensemble_params

    b0, b1 = sorted(float(v) for v in _ensemble_params(ctx)["base_ms"])
    p = Panel("(a) Opener and closer activity", Axis("Time from motion onset", V.MSEC),
              Axis("Envelope re pre-movement", V.DB), [], legend="auto")
    n_used = 0
    for role, lab, rl in (("agonist", "openers (digastric)", "group:openers"),
                          ("antagonist", "closers (masseter, temporalis)", "group:closers")):
        curves, grid = [], None
        for pid in ctx.pids():
            a = _arr(ctx, "timing", pid)
            y = a.get(f"ensemble__{task}__opening__{role}")
            if y is None or "ensemble__grid_ms" not in a:
                continue
            grid = np.asarray(a["ensemble__grid_ms"], float)
            y = np.asarray(y, float)
            pre = (grid >= b0) & (grid <= b1)
            if not np.isfinite(y[pre]).any():
                continue
            curves.append(y - np.nanmedian(y[pre]))
        if not curves:
            continue
        n_used = max(n_used, len(curves))
        S = np.array(curves)
        win = (grid >= b0) & (grid <= 400)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmedian(S, axis=0)
            q1, q3 = np.nanpercentile(S, [25, 75], axis=0)
        p.series += [Series("band", grid[win], q1[win], y2=q3[win], role=rl),
                     Series("line", grid[win], c[win], role=rl, label=lab)]
    p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
    if n_used == 0:
        _empty(p)
    p.x.lim = (-600.0, 400.0)
    return p, n_used


def _lead_specs(ctx, task: str = "open_close") -> tuple[list[float], list[float], list[float],
                                                         list[dict[str, str]], dict]:
    """The 36 single-cycle lead specifications (detector x motion source x onset rule) of
    the opening lead: cohort median and BCa CI from ``stats_timing.timing_cohort``."""
    tc = _sel(_t(ctx, "stats_timing", "timing_cohort"), task=task, direction="opening")
    tc = tc[tc.metric.isin(list(MECH))] if not tc.empty else tc
    est, lo, hi, specs, hl = [], [], [], [], {}
    for i, r in enumerate(tc.itertuples()):
        est.append(_f(r.median))
        lo.append(_f(r.ci_lo))
        hi.append(_f(r.ci_hi))
        specs.append({"detector": DET.get(r.detector, r.detector),
                      "motion": MECH.get(r.metric, r.metric), "rule": RULE.get(r.rule, r.rule)})
        if r.detector == "tkeo" and r.metric == "lead_ms" and r.rule in ("first", "drive"):
            hl[r.rule] = i
    lab = {"first": ("pre-specified analysis", "d"), "drive": ("driving-burst rule (post hoc)",
                                                               "s")}
    return est, lo, hi, specs, {hl[r]: lab[r] for r in ("first", "drive") if r in hl}


LEAD_DIMS = [("detector", list(DET.values())), ("motion", list(MECH.values())),
             ("rule", list(RULE.values()))]


C2_FORMS = ("slopegraph", "raincloud", "rugs")


@register("jnl_c2", "paper", "EMG activity and motion onset (journal)",
          choices={"form": C2_FORMS}, form="slopegraph")
def jnl_c2(req: ViewRequest) -> FigureSpec:
    """Envelopes around the opening onset, the per-participant lead under both onset rules,
    and the specification curve of the lead."""
    ctx = req.ctx
    form = req.options.get("form", "slopegraph")
    k = _min_cycles(ctx)
    pa, n_env = _envelopes(ctx)
    # (b) per-participant opening lead under both rules (slopegraph)
    ts = _sel(_t(ctx, "stats_timing", "timing_spec"), outcome="open_close|opening|lead")
    c2 = _c2_rows(ctx)
    pids = ctx.pids()
    per = {rule: (ts[ts.variant == LEAD_SPEC.format(rule)].set_index("pid").value
                  .reindex(pids).to_numpy(float) if not ts.empty else np.full(len(pids), np.nan))
           for rule, _, _ in RULES}
    pb = B.slopegraph([RULES[0][1], RULES[1][1]], per["first"], per["drive"],
                      "(b) Opening lead per participant", LEAD_AX,
                      summary=[(c2[r]["median"], c2[r]["ci_lo"], c2[r]["ci_hi"])
                               for r, _, _ in RULES],
                      # short form: two columns of text must fit the Springer panel too
                      annot=[[f"n = {_n(c2[r]['n'])} ({_excl(c2[r], short=True)})",
                              f"Holm p {_peq(c2[r]['p_holm'])}"] for r, _, _ in RULES],
                      ref=0.0)
    form_marks = [("grey lines in (b)", "join a participant's two values"),
                  ("light squares in (b)", "participants usable under one rule only"),
                  ("diamonds in (b)", "cohort median [95 % BCa CI]")]
    if form == "raincloud":
        pb, rc_marks = forms.summary("raincloud", [RULES[0][1], RULES[1][1]],
                                     [per["first"], per["drive"]],
                                     "(b) Opening lead per participant", LEAD_AX,
                                     seed=B.seed_of(ctx), ref=0.0)
        form_marks = [("per rule in (b)", "the kernel density of the participants' leads, a "
                                          "box (interquartile range, median, whiskers to the "
                                          "most extreme value within 1.5 IQR) and every "
                                          "participant as a dot")]
    if form == "rugs":
        # the per-participant EMG onsets (= − lead) as rugs under the envelopes, one row
        # per rule
        lo_y = min((float(np.nanmin(sr.y)) for sr in pa.series
                    if sr.kind in ("band", "line") and sr.y is not None
                    and np.isfinite(sr.y).any()), default=0.0)
        for i, (rule, lab, _) in enumerate(RULES):
            v = -per[rule][np.isfinite(per[rule])]
            if v.size:
                pa.series.append(Series("scatter", v, np.full(v.size, lo_y - 1.5 - 1.5 * i),
                                        role=("accent", "context")[i], style={"ms": "|", "size": 1.6},
                                        label=f"EMG onset per participant, {lab}"))
        pa.legend = "below"
        form_marks.append(("rugs below the envelopes in (a)", "each participant's EMG onset "
                                                              "relative to the motion onset "
                                                              "(= − lead) under the two rules"))
    # (c) specification curve of the opening lead
    est, lo, hi, specs, hl = _lead_specs(ctx)
    pc, pm = B.spec_curve(est, lo, hi, specs, "(c) Specification curve of the lead", LEAD_AX,
                          dims=LEAD_DIMS, highlight=hl, ref=0.0,
                          xlabel="Specification, ranked by lead", key="lead")
    # (c): legend inside the empty upper-left corner, so the matrix sits right below the curve;
    # (a): below the panel (inside it covered the traces).
    pc.legend = "inside:north west"
    pa.legend = "below"
    n_pos = int(sum(1 for a in lo if np.isfinite(a) and a > 0))
    n_b = int((np.isfinite(per["first"]) | np.isfinite(per["drive"])).sum())   # either rule
    spec = FigureSpec("jnl_c2", "EMG activity and motion onset", [pa, pc, pb, pm], rows=2,
                      cols=2, width="text", height_in=1.45,
                      sources=["timing:arrays", "stats_timing:timing_spec",
                               "stats_timing:timing_cohort"])
    from biomech.viz.views.dashboards import _ensemble_params

    b0, b1 = (abs(float(v)) for v in _ensemble_params(ctx)["base_ms"])
    base = f"{max(b0, b1):g}–{min(b0, b1):g} ms"
    return attach(spec, CaptionSpec(
        "open-close, opening. (a) EMG envelopes aligned to the mechanical opening onset, per "
        f"participant the median over all cycles in dB relative to the level {base} before "
        "onset (detector-free). (b) Per-participant lead of the digastric EMG onset (TKEO) "
        "over the earliest mechanical onset (> 0: EMG first) under the first-onset rule and "
        "the driving-burst rule. (c) Cohort median [95 % BCa CI] of the opening lead for all "
        f"{len(est)} combinations of onset detector, mechanical onset source, and onset rule, "
        f"ranked; {n_pos} of {len(est)} intervals lie entirely above zero",
        [("dashed line in (a)", "mechanical opening onset"),
         ("lines and bands in (a)", "median and IQR across participants")] + form_marks
        + [("dots below (c)", "each specification's choices")],
        f"n = {n_env} participants in (a), {n_b} in (b) (per rule: first onset "
        f"{int(np.isfinite(per['first']).sum())}, driving burst "
        f"{int(np.isfinite(per['drive']).sum())}); participants need ≥ {k} usable "
        "channel-cycles",
        "(b) cohort median [95 % BCa CI] per rule; the four confirmatory tests (open-close and "
        "chewing, both rules) with their values are in the text and the requirements table",
        "Holm correction over the confirmatory family of four tests (open-close and chewing "
        "under both rules)",
        "(a) exploratory; (b) first-onset rule pre-specified, driving-burst rule post hoc",
        [f"In open-close {_c2_verdict(c2)}."], codes=["C2"]))


# -- X3: causal closer release --------------------------------------------------------------------
@register("jnl_x3", "paper", "Causal closer-release detection (journal)",
          choices={"form": ("panels", "slopegraph")}, form="panels")
def jnl_x3(req: ViewRequest) -> FigureSpec:
    """Detection vs chance per participant, offline vs stream anticipation and false
    releases, and the parameter trade-off (exploratory)."""
    ctx = req.ctx
    form = req.options.get("form", "panels")
    rs = _t(ctx, "timing", "release_summary")
    prim = rs[(rs.task == "open_close") & rs.primary.astype(bool)] if not rs.empty else rs
    pids = [p for p in ctx.pids() if not prim.empty and (prim.pid == p).any()]
    s = prim.set_index("pid").reindex(pids) if pids else pd.DataFrame()
    det = 100 * s.detected_frac.to_numpy(float) if pids else np.empty(0)
    cha = 100 * s.chance_detected_frac.to_numpy(float) if pids else np.empty(0)
    rows = B.pseudo_rows(ctx, pids)
    n_nochance = int((~np.isfinite(cha)).sum()) if pids else 0
    pa = Panel("(a) Openings detected vs chance", Axis("Openings detected", V.PCT,
                                                        lim=(-4.0, 104.0)), Axis(""), [],
               legend="auto")
    y = np.arange(len(pids), dtype=float)
    for i in range(len(pids)):
        if np.isfinite(det[i]) and np.isfinite(cha[i]):
            pa.series.append(Series("line", np.array([cha[i], det[i]]), np.array([y[i], y[i]]),
                                    role="context"))
    if pids:
        okc = np.isfinite(cha)
        pa.series += [Series("scatter", cha[okc], y[okc], role="context",
                             style={"ms": "s", "size": 1.1}, label="random pseudo-onsets"),
                      Series("scatter", det, y, role="accent", style={"size": 1.3},
                             label="real openings")]
        pa.y.ticks, pa.y.ticklabels, pa.y.reverse = list(y), rows, True
        pa.y.lim = (-0.6, len(pids) - 0.4)
    else:
        _empty(pa)
    if form == "slopegraph" and pids:
        pa = B.slopegraph(["random pseudo-onsets", "real openings"], cha, det,
                          "(a) Openings detected vs chance",
                          Axis("Openings detected", V.PCT, lim=(-4.0, 104.0)),
                          summary=[_med_iqr(cha), _med_iqr(det)],
                          summary_label="median [IQR]", lone_label=None)
    # (b, c) offline replay of the causal chain vs the stream, per participant
    on = _t(ctx, "ml_online", "release")
    on = on[on.task == "open_close"] if not on.empty and "task" in on else on
    on = on.set_index("pid").reindex([p for p in ctx.pids() if not on.empty
                                      and p in set(on.pid)]) if not on.empty else on
    panels = []
    for col, title, ax in (("anticipation_ms", "(b) Anticipation",
                            Axis("Decision before onset", V.MSEC)),
                           ("false_per_min", "(c) False releases",
                            Axis("False releases", V.PER_MIN))):
        a = on[f"{col}_offline"].to_numpy(float) if not on.empty else np.empty(0)
        b = on[col].to_numpy(float) if not on.empty else np.empty(0)
        if form == "slopegraph":
            p, _ = forms.summary("raincloud", ["offline", "stream"], [a, b], title, ax,
                                 seed=B.seed_of(ctx),
                                 ref=0.0 if col == "anticipation_ms" else None)
        else:
            p = B.slopegraph(["offline", "stream"], a, b, title, ax,
                             summary=[_med_iqr(a), _med_iqr(b)], summary_label="median [IQR]",
                             lone_label=None, ref=0.0 if col == "anticipation_ms" else None)
        panels.append(p)
    pb, pc = panels
    # (d) trade-off over the drop x hold sweep (cohort medians)
    rc = _t(ctx, "stats_timing", "release_cohort")
    rc = rc[rc.task == "open_close"] if not rc.empty else rc
    pd_ = Panel("(d) Detector settings", Axis("False releases (median)", V.PER_MIN),
                Axis("Detected minus chance", V.PCT), [], legend="auto")
    if not rc.empty and "detected_minus_chance_median" in rc:
        pr = rc.primary.astype(bool).to_numpy()
        x, yy = rc.false_per_min_median.to_numpy(float), 100 * rc.detected_minus_chance_median \
            .to_numpy(float)
        pd_.series += [Series("hline", y=np.array([0.0]), role="reference", label="chance"),
                       Series("scatter", x[~pr], yy[~pr], role="context",
                              style={"size": 1.2}, label="other settings"),
                       Series("scatter", x[pr], yy[pr], role="accent",
                              style={"ms": "d", "size": 1.6}, label="pre-set setting")]
    else:
        _empty(pd_)
    rp = rc[rc.primary.astype(bool)] if not rc.empty else rc
    cw = _t(ctx, "stats_timing", "release_cohort")
    cw = cw[(cw.task == "chewing") & cw.primary.astype(bool)] if not cw.empty else cw

    def pct(df: pd.DataFrame, c: str) -> str:
        return f"{100 * _f(df[c].iloc[0]):.0f} %" if not df.empty and c in df else "--"

    stat, chew = "", []
    if not rp.empty:
        r = rp.iloc[0]
        stat = (f"cohort (pre-set setting, {int(_f(r.n_participants))} participants): "
                f"{pct(rp, 'detected_frac_median')} of openings detected vs "
                f"{pct(rp, 'chance_detected_frac_median')} at chance (medians); decision "
                f"{_n(_f(r['median']))} ms before onset {_ci(_f(r.ci_lo), _f(r.ci_hi))} "
                f"(median, 95 % BCa CI; before onset in {int(_f(r.n_before))}/{int(_f(r.n))}); "
                f"{_n(_f(r.false_per_min_median))} false releases per minute")
    if not cw.empty:
        margin = (_f(cw.iloc[0].detected_frac_median)
                  - _f(cw.iloc[0].chance_detected_frac_median))
        near = not np.isfinite(margin) or margin < 0.10
        chew = [f"In chewing the detector is {'close to' if near else 'above'} chance "
                f"({pct(cw, 'detected_frac_median')} vs "
                f"{pct(cw, 'chance_detected_frac_median')})"
                + (": the rhythm releases the closers continuously." if near else ".")]
    slope = form == "slopegraph"
    rain = ("the kernel density of the participants, a box (interquartile range, median, "
            "whiskers to the most extreme value within 1.5 IQR) and every participant as a dot")
    marks = ([("grey lines in (a)", "join a participant"), ("diamonds in (a)", "median [IQR]"),
              ("per condition in (b, c)", rain)] if slope else
             [("grey lines in (b, c)", "join a participant"),
              ("diamonds in (b, c)", "median [IQR]")])
    spec = FigureSpec("jnl_x3", "Causal closer-release detection", [pa, pb, pc, pd_], rows=2,
                      cols=2, width="text", height_in=1.6,
                      sources=["timing:release_summary", "ml_online:release",
                               "stats_timing:release_cohort"])
    return attach(spec, CaptionSpec(
        f"a causal detector of the closer release (closer-group envelope {release_setting(ctx)}, "
        "set before the data were analysed) in open-close. (a) Per "
        "participant, the share of openings with a release decision before the motion onset, "
        "against the share at random pseudo-onsets (chance). (b, c) Anticipation and false "
        "releases per minute of the same detector on the offline replay of the causal "
        "processing chain and on the simulated real-time stream (20-ms chunks; no device, no "
        "closed loop). (d) Cohort medians for every drop threshold and hold time of the "
        "sensitivity sweep: detection above chance against false releases",
        marks,
        f"n = {len(pids)} participants"
        + (f"; chance not available for {n_nochance} participants whose task left no room for "
           "pseudo-onsets" if n_nochance else ""),
        stat, evidence="exploratory", notes=chew, codes=["X3"]))


# -- decoding ---------------------------------------------------------------------------------------
def _few_shot(ctx) -> dict[int, dict[str, Any]]:
    """Per k: per-participant EMG + IMU minus IMU (few-shot variant, same evaluation set) and
    the stored test of that k (``ml_eval.tests`` row ``fewshot|svm|emg_imu_vs_imu|k=<k>``:
    exact Wilcoxon p, Holm over the calibration amounts, participants better); NaN p when the
    store has no such row (results computed before ml_eval stored it)."""
    v = _t(ctx, "ml_eval", "variants")
    f = v[v.variant == "few_shot"] if not v.empty else v
    if f.empty:
        return {}
    t = _t(ctx, "ml_eval", "tests")
    w = f.pivot_table(index="pid", columns=["feature_set", "k"], values="balanced_accuracy")
    out: dict[int, dict[str, Any]] = {}
    for k in sorted({int(c[1]) for c in w.columns}):
        if ("emg_imu", k) in w.columns and ("imu", k) in w.columns:
            d = (w[("emg_imu", k)] - w[("imu", k)]).dropna()
            r = t[t.test == f"fewshot|svm|emg_imu_vs_imu|k={k}"] if "test" in t else t.iloc[:0]
            r = r.iloc[0] if len(r) else {}
            out[k] = {"d": d, "better": int(r.get("n_better", (d > 0).sum())),
                      "n": int(r.get("n", d.size)), "p": _f(r.get("p", np.nan)),
                      "p_holm": _f(r.get("p_holm", np.nan)),
                      "n_amounts": int(r.get("n_amounts", 0) or 0)}
    return out


def _n_amounts(fsh: dict[int, dict[str, Any]]) -> int:
    """Size of the Holm family of the few-shot tests (stored; else the amounts shown)."""
    return max((v["n_amounts"] for v in fsh.values()), default=0) or len(fsh)


def _loso_diff(ctx, fs: str = "emg_imu", base: str = "imu", table: str = "loso"
               ) -> pd.Series:
    lo = _t(ctx, "ml_eval", table)
    lo = lo[lo.model == "svm"] if not lo.empty else lo
    if lo.empty:
        return pd.Series(dtype=float)
    w = lo.pivot_table(index="pid", columns="feature_set", values="balanced_accuracy")
    if fs not in w or base not in w:
        return pd.Series(dtype=float)
    return (w[fs] - w[base]).dropna()


def _test_p(ctx, other: str) -> float:
    t = _t(ctx, "ml_eval", "tests")
    r = t[t.test == f"wilcoxon|svm|emg_imu_vs_{other}"] if not t.empty else t
    return _f(r.p_holm.iloc[0]) if not r.empty and "p_holm" in r else np.nan


def _recall_per_participant(ctx, fs: str) -> np.ndarray | None:
    """(participants, classes) recall in % of the unseen-participant SVM from the stored
    held-out predictions (``pred__<pid>__<set>__y`` / ``__yhat``; their sum is the stored
    pooled confusion); NaN where a participant has no window of that class (J22.6)."""
    a = _arr(ctx, "ml_eval", COHORT_UNIT)
    pids = sorted({k.split("__")[1] for k in a
                   if k.startswith("pred__") and k.endswith(f"__{fs}__y")})
    if not pids:
        return None
    k = len(CLASSES)
    out = np.full((len(pids), k), np.nan)
    for i, p in enumerate(pids):
        y = np.asarray(a[f"pred__{p}__{fs}__y"], int)
        yh = np.asarray(a[f"pred__{p}__{fs}__yhat"], int)
        for c in range(k):
            m = y == c
            if m.any():
                out[i, c] = 100.0 * float(np.mean(yh[m] == c))
    return out


def _recall(ctx, fs: str) -> np.ndarray | None:
    """Per class the median over participants of their recall (%; J22.6)."""
    r = _recall_per_participant(ctx, fs)
    if r is None:
        return None
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmedian(r[:, c]) if np.isfinite(r[:, c]).any() else np.nan
                         for c in range(r.shape[1])])


@register("jnl_decoding", "paper", "Intent decoding across participants (journal)",
          choices={"form": ("strip", "slopegraph", "with_online")}, form="strip")
def jnl_decoding(req: ViewRequest) -> FigureSpec:
    """EMG + IMU minus IMU per participant, few-shot calibration, per-class recall and
    block importance."""
    ctx = req.ctx
    form = req.options.get("form", "strip")
    loso = _loso_diff(ctx)
    fsh = _few_shot(ctx)
    lab = [f"no calibration ({int((loso > 0).sum())}/{loso.size})"]
    vals = [loso.to_numpy(float)]
    for k in sorted(fsh):
        if k > 0:
            lab.append(f"calibration {_cal_s(ctx, k)} ({fsh[k]['better']}/{fsh[k]['n']})")
            vals.append(fsh[k]["d"].to_numpy(float))
    pa = B.strip(lab, vals, "(a) EMG + IMU minus IMU",
                 Axis("Difference in balanced accuracy"), ref=0.0)
    if not loso.size:
        _empty(pa)
    a_txt = ("per participant the balanced accuracy with EMG + IMU features minus that with "
             "IMU features alone")
    a_marks = [("dots", "participants"), ("diamond in (a)", "median"),
               ("in brackets in (a)", "participants better with EMG")]
    if form == "slopegraph":
        lo = _t(ctx, "ml_eval", "loso")
        lo = lo[lo.model == "svm"] if not lo.empty else lo
        w = lo.pivot_table(index="pid", columns="feature_set", values="balanced_accuracy") \
            if not lo.empty else pd.DataFrame()
        if {"imu", "emg_imu"} <= set(w.columns):
            a, b = w["imu"].to_numpy(float), w["emg_imu"].to_numpy(float)
            seed = B.seed_of(ctx)
            summ = [(float(np.nanmedian(v)), *B.median_ci(v, seed)) for v in (a, b)]
            pa = B.slopegraph([FSET["imu"], FSET["emg_imu"]], a, b,
                              "(a) Unseen participant, no calibration", BA_AX, summary=summ,
                              lone_label=None)
            pa.series.append(Series("hline", y=np.array([1 / 7]), role="threshold",
                                    label="chance"))
        a_txt = ("per participant the balanced accuracy with IMU features alone and with "
                 "EMG + IMU features")
        a_marks = [("grey lines in (a)", "participants"),
                   ("diamonds in (a)", "median with 95 % BCa CI"),
                   ("dots in (d)", "participants")]
    # (b) few-shot calibration curve
    vs = _t(ctx, "ml_eval", "variants_summary")
    pb = Panel("(b) Few-shot calibration", Axis("Calibration data per movement", V.SEC),
               BA_AX, [], legend="auto")
    for fs in ("emg_imu", "imu"):
        s = vs[(vs.variant == "few_shot") & (vs.feature_set == fs)
               & (vs.metric == "balanced_accuracy")].sort_values("k") if not vs.empty else vs
        if s.empty:
            continue
        x = s.k.to_numpy(float) * _stride(ctx)            # windows -> s per movement
        pb.series += [Series("band", x, s.q1.to_numpy(float), y2=s.q3.to_numpy(float),
                             role=f"set:{fs}"),
                      Series("line", x, s["median"].to_numpy(float), role=f"set:{fs}"),
                      # the legend key on the markers, so it shows the set's marker
                      Series("scatter", x, s["median"].to_numpy(float), role=f"set:{fs}",
                             style={"size": 1.1}, label=FSET[fs])]
    if pb.series:
        pb.series.append(Series("hline", y=np.array([1 / 7]), role="threshold",
                                label="chance"))
    else:
        _empty(pb)
    # (c) recall per class (unseen subject): median over participants (J22.6)
    pc = Panel("(c) Recall per class", Axis("Recall, unseen participant", V.PCT,
                                            lim=(0.0, 100.0)), Axis(""), [], legend="auto")
    offs = {"emg_imu": -0.22, "imu": 0.0, "emg": 0.22}
    for fs, off in offs.items():
        r = _recall(ctx, fs)
        if r is None:
            continue
        pc.series.append(Series("scatter", r, np.arange(r.size) + off, role=f"set:{fs}",
                                style={"size": 1.3}, label=FSET[fs]))
    if pc.series:
        pc.series.append(Series("vline", x=np.array([100 / 7]), role="threshold",
                                label="chance"))
        pc.y.ticks = [float(i) for i in range(len(CLASS_LABEL))]
        pc.y.ticklabels, pc.y.reverse = list(CLASS_LABEL), True
        pc.y.lim = (-0.6, len(CLASS_LABEL) - 0.4)
    else:
        _empty(pc)
    # (d) block permutation importance
    imp = _t(ctx, "ml_eval", "permutation_importance")
    blocks = [b for b in ("emg", "imu") if not imp.empty and (imp.block == b).any()]
    pd_ = B.strip_by(imp, "ba_drop", "block", "(d) Feature-block importance", order=blocks,
                     labels={b: FSET[b] for b in blocks},
                     xaxis=Axis("Drop in balanced accuracy when permuted"), ref=0.0)
    p0 = _test_p(ctx, "imu")
    k_max = max(fsh) if fsh else None
    few = (f"with {_cal_s(ctx, k_max) if k_max else '--'} of calibration data per movement "
           "EMG + IMU "
           f"is better than IMU alone in {fsh[k_max]['better']}/{fsh[k_max]['n']} participants "
           f"(exact Wilcoxon p {_peq(fsh[k_max]['p'])}, Holm-adjusted over {_n_amounts(fsh)} "
           f"calibration amounts including none, p {_peq(fsh[k_max]['p_holm'])})" if k_max
           else "the calibration variant is not available")
    amounts = [_cal_s(ctx, k) for k in sorted(fsh) if k > 0]
    amounts_txt = series(amounts, "or") if amounts else "--"
    help0 = ("improves decoding" if np.isfinite(p0) and p0 < 0.05
             else "does not improve decoding significantly")
    med = (imp.groupby("block").ba_drop.median() if not imp.empty and "block" in imp
           else pd.Series(dtype=float))
    relies = (f"across participants the decoder relies most on the "
              f"{FSET.get(med.idxmax(), med.idxmax())} block" if med.size else "")
    slope = form == "slopegraph"
    shows = (
        "seven-class intent decoding (SVM). (a) " + a_txt[:1].upper() + a_txt[1:] + " for an "
        "unseen participant (leave-one-subject-out), without calibration"
        + (" (the calibrated amounts are in b)" if slope else
           f" and after adding the new participant's first {amounts_txt} of every movement "
           "to the training set")
        + ". (b) Balanced accuracy of the calibrated decoder over the amount of calibration "
        "data (chance 1/7). (c) Recall per movement class of the unseen-participant decoder, "
        "median over participants of each held-out participant's recall. (d) Drop in held-out balanced accuracy "
        "when one feature block of the EMG + IMU decoder is permuted"
        + (f": {relies}" if relies else ""))
    # every form (2026-11-10: `decoding` replaced `decodingOnline` in the JNER main text):
    # (b) the median curves without the overlapping IQR bands (revision 3.7), legends inside
    pb.series = [sr for sr in pb.series if sr.kind != "band"]
    pb.legend = "inside:north west"
    pc.legend = "inside:north east"            # in room right of 100 %
    pc.x.lim, pc.x.ticks = (0.0, 135.0), [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    marks = a_marks + [("lines in (b)", "median across participants"),
                       ("dots and diamond in (d)", "participants and their median")]
    evidence = ("(a) confirmatory comparison; (b) exploratory (post hoc)" if slope else
                "(a) without calibration confirmatory comparison, with calibration "
                "exploratory (post hoc); (b) exploratory (post hoc)") + \
        "; (c) descriptive; (d) pre-specified"
    srcs = ["ml_eval:loso", "ml_eval:variants", "ml_eval:variants_summary",
            "ml_eval:confusion", "ml_eval:permutation_importance", "ml_eval:tests"]
    panels, rows = [pa, pb, pc, pd_], 2
    if form == "with_online":
        # JNER revision 3.7 (2026-11-10): (e) without calibration only (the calibration-block
        # rows scored below "no calibration" and read as an error; all modes are in the
        # simulated real-time figure of the additional file), (b) the median curves without
        # the overlapping IQR bands, the stream legend inside the panel
        on = jnl_online(ViewRequest(ctx, options={"modes": "none"}))
        pe, pf = on.panels[0], on.panels[1]
        pe.title, pf.title = "(e) Simulated real time", "(f) Decision latency"
        pe.y.ticklabels = [t.split(",")[0] for t in (pe.y.ticklabels or [])]
        pe.legend = "inside:south west"
        pf.legend = "inside:north west"
        panels, rows = panels + [pe, pf], 3
        shows += (". (e, f) Simulated real time (every recording replayed in 20-ms chunks "
                  "through a strictly causal chain; no device, no closed loop), without "
                  "calibration: (e) balanced accuracy of the stream decisions per feature "
                  "set; (f) latency from the optical label onset to the first decision "
                  "entering that class, where a negative latency is a decision before the "
                  "label onset")
        marks = [("dots", "participants"), ("diamond in (a, e)", "median")] + a_marks[2:] + [
            ("lines in (b)", "median across participants"), marks[-1],
            ("squares in (e)", "the same decoder offline"),
            ("dashed line in (f)", f"the latency budget, "
                                   f"{B.latency_budget_ms(ctx):g} ms")]
        evidence += "; (e, f) exploratory"
        srcs += on.sources
    spec = FigureSpec("jnl_decoding", "Intent decoding across participants",
                      panels, rows=rows, cols=2, width="text", height_in=1.45, sources=srcs)
    return attach(spec, CaptionSpec(
        shows, marks, f"n = {loso.size} participants",
        f"without calibration EMG {help0} (Holm p {_peq(p0)}); {few}", evidence=evidence))


# -- simulated real time -----------------------------------------------------------------------------
def _online_rows(s: pd.DataFrame) -> list[tuple[str, str, str, pd.DataFrame]]:
    out = []
    for fs in ("emg_imu", "imu"):
        for mode in MODE:
            g = _sel(s, feature_set=fs, mode=mode)
            if not g.empty:
                out.append((fs, mode, f"{FSET[fs]}, {MODE[mode]}", g))
    return out


@register("jnl_online", "paper", "Simulated real-time decoding (journal)",
          choices={"form": ("panels", "compact")}, form="panels")
def jnl_online(req: ViewRequest) -> FigureSpec:
    """Stream vs offline balanced accuracy, decision latency, false activations, compute."""
    ctx = req.ctx
    form = req.options.get("form", "panels")
    s = _t(ctx, "ml_online", "summary")
    if not s.empty and "applicable" in s:
        s = s[s.applicable.astype(bool)]
    rows = _online_rows(s)
    modes = req.options.get("modes")
    if modes:                       # e.g. "none": only those calibration modes (JNER 3.7)
        keep = set(str(modes).split(","))
        rows = [r for r in rows if r[1] in keep]
    lab = [r[2] for r in rows]

    def strip_with_offline(col: str, off: str, title: str, ax: Axis, ref=None, ref_label=None):
        p = B.strip(lab, [g[col].to_numpy(float) for *_, g in rows], title, ax, ref=ref,
                    ref_label=ref_label)
        p.legend = "auto"
        first = True
        for i, (*_, g) in enumerate(rows):
            m = _med_iqr(g[off])[0] if off in g else np.nan
            if np.isfinite(m):
                # just below the row line: on it, the square hid the stream diamond
                p.series.append(Series("scatter", np.array([m]), np.array([float(i) + 0.3]),
                                       role="muted", style={"ms": "s", "size": 1.4},
                                       label="offline, same decoder" if first else None))
                first = False
        # the stream median (accent diamond) as a legend entry too
        for sr in p.series:
            if sr.role == "accent" and sr.kind == "scatter":
                sr.label = "stream, median"
                break
        if not rows:
            _empty(p)
        return p

    pa = strip_with_offline("ba_raw", "balanced_accuracy_offline", "(a) Balanced accuracy",
                            BA_AX, ref=1 / len(CLASSES), ref_label="chance")
    dec = _t(ctx, "ml_online", "decisions")
    hits = dec[(dec.kind == "hit") & dec.scored.astype(bool) & (dec["mode"] == "none")] \
        if not dec.empty and "latency_ms" in dec else pd.DataFrame()
    curves = {FSET[fs]: hits[hits.feature_set == fs].latency_ms.to_numpy(float)
              for fs in ("emg_imu", "imu") if not hits.empty and (hits.feature_set == fs).any()}
    budget = B.latency_budget_ms(ctx)
    pb = B.ecdf(curves, "(b) Decision latency", Axis("Latency after label onset", V.MSEC),
                roles={FSET["emg_imu"]: "set:emg_imu", FSET["imu"]: "set:imu"}, points=False,
                yaxis=Axis("Onsets at or below", V.PCT, lim=(0.0, 104.0)),
                ref=budget)
    for sr in pb.series:                      # IMU dashed: the two hues are close
        if sr.role == "set:imu" and sr.kind == "line":
            sr.style["dash"] = "dashed"
    le = {fs: float(np.mean(hits[hits.feature_set == fs].latency_ms <= budget)) * 100
          for fs in ("emg_imu", "imu") if not hits.empty and (hits.feature_set == fs).any()}
    pc = strip_with_offline("fa_per_min_rest", "fa_per_min_offline", "(c) False activations",
                            Axis("False activations at rest", V.PER_MIN))
    one = s.drop_duplicates("pid") if not s.empty else s
    cols = [c for c in ("chunk_ms_median", "chunk_ms_p99", "chunk_ms_max") if c in one]
    pd_ = B.strip([{"chunk_ms_median": "median", "chunk_ms_p99": "P99",
                    "chunk_ms_max": "largest"}[c] for c in cols],
                  [one[c].to_numpy(float) for c in cols], "(d) Compute per 20-ms chunk",
                  Axis("Compute time", V.MSEC), ref=20.0)
    if one.empty:
        _empty(pd_)
    rtf = _med_iqr(one.realtime_factor)[0] if "realtime_factor" in one else np.nan
    n = int(one.pid.nunique()) if "pid" in one else 0
    compact = form == "compact"
    shows = (
        "simulated real time: every recording replayed in 20-ms chunks through a strictly "
        "causal processing chain and the leave-one-subject-out decoder (SVM); no device and no "
        "closed loop. (a) Balanced accuracy of the stream window decisions per feature set and "
        "calibration mode (none; a calibration block after rest; the first windows of every "
        "phase). (b) Latency from the optical label onset to the first decision entering that "
        "class, all detected onsets without calibration"
        + ("" if compact else
           ". (c) False activations per minute at rest (not defined for the calibration "
           "block, which leaves no rest windows after calibration). (d) Single-thread compute "
           "per 20-ms chunk on the analysis PC per participant (median, 99th percentile, "
           "largest)"))
    marks = [("dots in (a)", "participants"), ("diamond in (a)", "median"),
             ("squares in (a)", "median of the same decoder on the offline windows (just "
                                "below each row)"),
             ("dashed line in (b)", f"the latency budget, {budget:g} ms")]
    if not compact:
        marks.append(("dashed line in (d)", "the chunk length"))
    within = (f"(b) detected onsets decided within {budget:g} ms: "
              + "; ".join(f"{FSET[k]} {v:.0f} %" for k, v in le.items()) + "; " if le else "")
    srcs = ["ml_online:summary", "ml_online:decisions"]
    if compact:
        # one row: accuracy and latency; the false activations and compute go to the text
        spec = FigureSpec("jnl_online", "Simulated real-time decoding", [pa, pb], rows=1,
                          cols=2, width="text", height_in=1.6, sources=srcs)
        return attach(spec, CaptionSpec(
            shows, marks, f"n = {n} participants",
            within + f"real-time factor {_n(rtf, 3)} (median)", evidence="exploratory",
            notes=["False activations at rest and the compute per chunk are reported in the "
                   "text."]))
    spec = FigureSpec("jnl_online", "Simulated real-time decoding", [pa, pb, pc, pd_], rows=2,
                      cols=2, width="text", height_in=1.4, sources=srcs)
    return attach(spec, CaptionSpec(
        shows, marks, f"n = {n} participants",
        within + f"real-time factor {_n(rtf, 3)} (median)", evidence="exploratory"))


# -- robustness ---------------------------------------------------------------------------------------
KEY_OUTCOMES = {"mio_mm": "amplitude", "open_close|opening|lead": "lead",
                "median_pct|chewing|masseter_r": "masseter R",
                "median_pct|open_close|digastric_r": "digastric R"}
OPENING_ALT = {"rest|vertical|lowpass": "rest ref.", "cycle_start|3d|lowpass": "3-D",
               "cycle_start|vertical|raw": "unfiltered"}
NORM_ALT = {"norm=p95": "MVC P95", "norm=max1s": "max 1 s"}
# Onset-rule levels of the specification matrix (short: the matrix rows are labelled
# "rule: <level>"); spec values and matrix levels must be the same strings.
RULE_MATRIX = {"first": "first onset", "drive": "driving burst",
               "drive_taskbase": "in-task threshold"}


def _parse(variant: str) -> dict[str, str]:
    return dict(f.split("=", 1) for f in variant.split("|") if "=" in f)


def _standardized_specs(ctx) -> tuple[list, list, list, list[dict[str, str]], dict]:
    """Every non-primary processing variant of the key outcomes: mean paired change vs the
    primary variant in between-participant SD units, with a t interval."""
    sv = _t(ctx, "variability", "spec_values")
    sc = _t(ctx, "variability", "spec_curve")
    est, lo, hi, specs, hl = [], [], [], [], {}
    if sv.empty or sc.empty:
        return est, lo, hi, specs, hl
    from scipy.stats import t as tdist

    for o, olab in KEY_OUTCOMES.items():
        r = sc[sc.outcome == o]
        g = sv[sv.outcome == o]
        if r.empty or g.empty:
            continue
        prim, sd = str(r.primary_variant.iloc[0]), _f(r.sd_between.iloc[0])
        w = g.pivot_table(index="pid", columns="variant", values="value")
        if prim not in w or not np.isfinite(sd) or sd <= 0:
            continue
        pp = _parse(prim)
        for v in w.columns:
            if v == prim:
                continue
            d = (w[v] - w[prim]).dropna().to_numpy(float) / sd
            if d.size < 2:
                continue
            m, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
            q = float(tdist.ppf(0.975, d.size - 1))
            spec = {"outcome": olab}
            pv = _parse(v)
            if pv:                                # detector / mech / onset / norm fields
                for key, dim, names in (("detector", "detector", DET), ("mech", "motion", MECH),
                                        ("onset", "rule", RULE_MATRIX)):
                    if key in pv and pv[key] != pp.get(key):
                        spec[dim] = names.get(pv[key], pv[key])
                if "norm" in pv and pv["norm"] != pp.get("norm"):
                    spec["normalization"] = NORM_ALT.get(v, pv["norm"])
            else:
                spec["opening"] = OPENING_ALT.get(v, v)
            if o.endswith("|lead") and pv == {"detector": "tkeo", "mech": "lead_ms",
                                              "onset": "first"}:
                hl[len(est)] = ("pre-specified onset rule", "d")
            est.append(m)
            lo.append(m - q * se)
            hi.append(m + q * se)
            specs.append(spec)
    return est, lo, hi, specs, hl


def _comparison(ctx, name: str) -> pd.DataFrame:
    from biomech import sensitivity as S

    c = S.read_comparison(ctx.config.read_dir, name)
    if c.empty:
        return c
    c = c.copy()
    c["_k"] = [dict(f.split("=", 1) for f in str(k).split("|") if "=" in f) for k in c.key]
    return c


def _cmp(c: pd.DataFrame, table: str, column: str, **fields: str) -> tuple[float, float]:
    """(primary, variant) of one compared value; the row key must hold ``fields``."""
    if c.empty:
        return np.nan, np.nan
    q = c[(c.table == table) & (c.column == column)]
    q = q[[all(k.get(f) == v for f, v in fields.items()) for k in q._k]]
    if q.empty:
        return np.nan, np.nan
    return _f(q.primary.iloc[0]), _f(q.variant.iloc[0])


C2_KEYS = [("open_close", "first"), ("open_close", "drive"), ("chewing", "first"),
           ("chewing", "drive")]


def _lead_fields(task: str, rule: str) -> dict[str, str]:
    return {"task": task, "direction": "opening", "detector": "tkeo", "rule": rule,
            "metric": "lead_ms"}


VARIANT_TEXT = {"lowfreq_mask": "EMG masking before the low-frequency rule became a flag",
                "c2_muscle_mean": "the per-muscle mean without the channel-cycle minimum",
                "reporting_population": "the cohort without the excluded participant",
                "labels_p95": "free-motion label axes without the scale floor",
                "lowpass_8hz": "the pose low-pass at 8 Hz",
                "lowpass_10hz": "the pose low-pass at 10 Hz",
                "lowpass_12hz": "the pose low-pass at 12 Hz"}


NUM_WORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten"}


def _runs_text(names: list[str]) -> str:
    """'two complete sensitivity runs: A (a), and B' from the runs that exist (E37); the
    parenthesis is the short label the panel rows carry."""
    num = NUM_WORD
    if not names:
        return "no sensitivity run (none available)"
    parts = [f"{VARIANT_TEXT.get(n) or VARIANT_LABEL.get(n, n)} ({VARIANT_SHORT.get(n, n)})"
             for n in names]
    body = series(parts)
    return (f"{num.get(len(names), len(names))} complete sensitivity "
            f"run{'s' if len(names) > 1 else ''}: {body}")


def _variants(ctx) -> list[str]:
    from biomech import sensitivity as S

    def natural(v: str) -> list:                 # lowpass_8hz before lowpass_10hz
        return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", v)]

    avail = sorted(S.available(ctx.config.read_dir), key=natural)
    based = _based_on_store(ctx, avail)
    if based:                            # D36: only the runs on this store's population
        avail = based
    return [v for v in avail if v in VARIANT_LABEL] + [v for v in avail if v not in VARIANT_LABEL]


def _based_on_store(ctx, names: list[str]) -> list[str]:
    """The variants whose ``base`` is the store the paper reads (D36: the sensitivity
    runs of the 13-adult paper build on, and are compared with, the reporting_population store); empty
    when none is (runs compared with the primary analysis, D35)."""
    from biomech import sensitivity as S

    v = getattr(ctx.config, "variant", None)
    if not v:
        return []
    try:
        return [n for n in names if S.base_name(ctx.config, n) == v]
    except Exception:                   # no sensitivity block in this configuration
        return []


def _current(ctx, task: str, rule: str) -> float:
    return _c2_rows(ctx, task)[rule]["median"]


def _run_items(ctx) -> list[tuple[str, str, dict[str, str], float]]:
    """(label, table, key fields, current median) of the outcomes drawn per sensitivity run:
    the four opening leads and the closer release before opening."""
    items = [(f"{V.task_label(t)}, {RULE_SHORT[r]}", "timing_cohort", _lead_fields(t, r),
              _current(ctx, t, r)) for t, r in C2_KEYS]
    ens = _sel(_t(ctx, "stats_timing", "ensemble_cohort"), task="open_close",
               direction="opening", role="antagonist")
    items.append(("closer release", "ensemble_cohort",
                  {"task": "open_close", "direction": "opening", "role": "antagonist"},
                  _f(ens["median"].iloc[0]) if not ens.empty else np.nan))
    return items


RunRow = tuple[str, dict[str, tuple[float, float]], bool]


def _run_rows(ctx) -> tuple[list[tuple[str, list[RunRow]]], list[str]]:
    """Per sensitivity run (that differs from the primary somewhere) the drawn outcomes:
    (label, {median|ci_lo|ci_hi: (primary, variant)}, stale), and the runs identical to the
    primary on every value (caption only)."""
    out: list[tuple[str, list[RunRow]]] = []
    same: list[str] = []
    items = _run_items(ctx)
    for name in _variants(ctx):
        c = _comparison(ctx, name)
        if c.empty:
            continue
        rows_v: list[RunRow] = []
        for lab, table, fields, now in items:
            vals = {col: _cmp(c, table, col, **fields) for col in ("median", "ci_lo", "ci_hi")}
            stale = bool(np.isfinite(now) and np.isfinite(vals["median"][0])
                         and abs(vals["median"][0] - now) > 1e-6)
            rows_v.append((lab, vals, stale))
        if all(np.allclose(v[col][0], v[col][1], atol=1e-6, equal_nan=True)
               for _, v, _ in rows_v for col in ("median", "ci_lo", "ci_hi")):
            same.append(name)
            continue
        out.append((name, rows_v))
    return out, same


def _runs_blocks(runs_data: list[tuple[str, list[RunRow]]], title: str) -> Panel:
    """The long form: one block per run (a header row, then each outcome with the primary
    and the run's value)."""
    labels: list[str] = []
    A: tuple[list, list, list] = ([], [], [])
    Bv: tuple[list, list, list] = ([], [], [])
    for name, rows_v in runs_data:
        labels.append(f"{VARIANT_SHORT.get(name, name)}:")
        for arr in (*A, *Bv):
            arr.append(np.nan)
        for lab, vals, stale in rows_v:
            labels.append(lab + (" †" if stale else ""))
            for j, col in enumerate(("median", "ci_lo", "ci_hi")):
                A[j].append(vals[col][0])
                Bv[j].append(vals[col][1])
    return B.paired_forest(labels, A, Bv, title, LEAD_AX, label_a="primary analysis",
                           label_b="sensitivity run", ref=0.0)


def _run_style(label: str) -> tuple[str, str]:
    """(role, marker) of a sensitivity-run outcome: colour = task (open-close orange, chewing
    green, the closer release pink; blue stays the closers'), marker = onset rule (first
    onset circle, driving burst diamond)."""
    task = ("cat:1" if label.startswith(V.task_label("open_close")) else
            "cat:2" if label.startswith(V.task_label("chewing")) else "cat:4")
    mark = ("o" if label.endswith(RULE_SHORT.get("first", "first")) else
            "d" if label.endswith(RULE_SHORT.get("drive", "driving")) else "s")
    return task, mark


def _runs_grouped(runs_data: list[tuple[str, list[RunRow]]], title: str) -> Panel:
    """The compact form: one row per analysis (the primary once, then each run), the five
    outcomes dodged within the row (colour = task, marker = onset rule). Runs compared with different primary
    values get one primary row each (in the order first met)."""
    if not runs_data:
        return B.grouped_forest([], [], title, LEAD_AX, ref=0.0)
    outs = [lab for lab, _, _ in runs_data[0][1]]
    labels: list[str] = []
    cols: list[list[tuple[float, float, float]]] = [[] for _ in outs]
    prim_seen: list[tuple] = []

    def key(rows_v: list[RunRow]) -> tuple:
        return tuple(round(float(v[c][0]), 6) if np.isfinite(v[c][0]) else None
                     for _, v, _ in rows_v for c in ("median", "ci_lo", "ci_hi"))

    groups: dict[tuple, list[tuple[str, list[RunRow]]]] = {}
    for name, rows_v in runs_data:
        k = key(rows_v)
        if k not in groups:
            prim_seen.append(k)
            groups[k] = []
        groups[k].append((name, rows_v))
    for k in prim_seen:
        members = groups[k]
        first = members[0][1]
        stale = any(st for _, st_rows in members for _, _, st in st_rows)
        labels.append("primary analysis" + (" †" if stale else ""))
        for j, (_, vals, _) in enumerate(first):
            cols[j].append(tuple(vals[c][0] for c in ("median", "ci_lo", "ci_hi")))
        for name, rows_v in members:
            labels.append(VARIANT_SHORT.get(name) or V.rewrite_ids(name))
            for j, (_, vals, _) in enumerate(rows_v):
                cols[j].append(tuple(vals[c][1] for c in ("median", "ci_lo", "ci_hi")))
    groups_arr = [(lab, tuple(np.array([r[i] for r in cols[j]], float) for i in range(3)))
                  for j, lab in enumerate(outs)]
    p = B.grouped_forest(labels, groups_arr, title, LEAD_AX, ref=0.0,
                         styles=[_run_style(lab) for lab in outs])
    prim_rows = [i for i, lab in enumerate(labels) if lab.startswith("primary analysis")]
    for i in sorted({i + 0.5 for i in prim_rows} | {i - 0.5 for i in prim_rows if i > 0}):
        p.series.append(Series("hline", y=np.array([i]), role="faint"))  # primary rows set off
    return p


@register("jnl_robustness", "paper", "Robustness of the findings (journal)",
          choices={"form": ("speccurve", "ranked", "speccurve_blocks", "per_outcome", "runs")},
          form="speccurve")
def jnl_robustness(req: ViewRequest) -> FigureSpec:
    """Standardized specification curve, primary vs sensitivity runs, plausible condyles.

    Forms: ``speccurve`` (one row per analysis in (c), the primary once and each run below
    it, the five timing outcomes dodged, colour = task and marker = rule: fits one page),
    ``speccurve_blocks`` (the same values as one block per run, primary and run value per
    outcome: long) and ``ranked`` ((a) as a ranked dot plot). Every value of (c) is also in
    ``jnl_table_robustness``."""
    ctx = req.ctx
    form = req.options.get("form", "speccurve")
    est, lo, hi, specs, hl = _standardized_specs(ctx)
    dims = [("outcome", [v for k, v in KEY_OUTCOMES.items()
                         if any(s["outcome"] == v for s in specs)] or list(KEY_OUTCOMES.values())),
            ("rule", [RULE_MATRIX["first"], RULE_MATRIX["drive_taskbase"]]),
            ("detector", ["envelope", "Hodges–Bui"]),
            ("motion", ["IMU velocity", "IMU accel.", "optical"]),
            ("normalization", list(NORM_ALT.values())), ("opening", list(OPENING_ALT.values()))]
    pa, pm = B.spec_curve(est, lo, hi, specs, "(a) Processing choices",
                          Axis("Change (between-participant SD)"), dims=dims,
                          highlight=hl, ref=0.0, xlabel="Processing variant, ranked",
                          key="robust")
    pa.legend = "inside:north west"
    big = int(sum(1 for e in est if abs(e) > 1))
    where = sorted({sp["outcome"] for e, sp in zip(est, specs) if abs(e) > 1})
    long = {"amplitude": "largest opening", "lead": "opening lead",
            "masseter R": "masseter R activation", "digastric R": "digastric R activation"}
    big_where = f", all of them in the {' and '.join(long.get(w, w) for w in where)}" \
        if where else ""
    # (b) primary vs sensitivity runs on the C2 / X2 outcomes (ms)
    runs_data, same = _run_rows(ctx)       # same: identical to the primary on every row
    stale_any = any(st for _, rows_v in runs_data for _, _, st in rows_v)
    blocks = form == "speccurve_blocks"
    pb = (_runs_blocks if blocks else _runs_grouped)(runs_data, "(c) Sensitivity runs")
    n_rows = len(pb.y.ticklabels or [])
    if not runs_data:
        _empty(pb, "no sensitivity run")
    # (c) C1 outcomes: all participants vs plausible palpated condyles
    cs = _t(ctx, "stats_kinematics", "condyle_sensitivity")
    mets = [("hinge_rms_incisor_hinge_fit", "best-fit hinge"),
            ("hinge_rms_incisor_hinge_tmj", "hinge through condyles"),
            ("condyle_left_disp_mm", "condyle excursion L"),
            ("condyle_right_disp_mm", "condyle excursion R")]

    def grp(g: str) -> tuple[list[float], list[float], list[float], int]:
        e, lw, hg, n = [], [], [], 0
        for m, _ in mets:
            r = _sel(cs, metric=m, group=g)
            e.append(_f(r["median"].iloc[0]) if not r.empty else np.nan)
            lw.append(_f(r.ci_lo.iloc[0]) if not r.empty else np.nan)
            hg.append(_f(r.ci_hi.iloc[0]) if not r.empty else np.nan)
            n = max(n, int(_f(r.n.iloc[0])) if not r.empty else 0)
        return e, lw, hg, n

    ea, la, ha, na = grp("all")
    ep, lp, hp, npl = grp("plausible_condyles")
    pc = B.paired_forest([m[1] for m in mets], (ea, la, ha), (ep, lp, hp),
                         "(b) Plausible condyles", Axis("Median", V.MM),
                         label_a=f"all participants (n = {na})",
                         label_b=f"plausible condyles (n = {npl})")
    pc.legend = "inside:north east"        # rows start at the left; keeps row 1 compact
    if cs.empty:
        _empty(pc)
    hn = _t(ctx, "uncertainty", "hinge_noise")
    subj = _t(ctx, "stats_kinematics", "subjects")
    c1 = ""
    if not hn.empty and "above_invivo" in hn:
        from biomech.stats.cohort_kinematics import hinge_noise_counts

        plaus = set(subj[subj.condyles_plausible.astype(bool)].pid) if \
            "condyles_plausible" in subj else set()
        k, n = hinge_noise_counts(hn, "invivo")
        kp, np_ = hinge_noise_counts(hn[hn.pid.isin(plaus)], "invivo")
        c1 = (f"the best-fit hinge residual exceeds the participant's in-vivo noise in "
              f"{k}/{n} tested participants and in {kp}/{np_} with plausible condyles")
    lab_agree = _t(ctx, "labels", "label_agreement")
    ran_p95 = "labels_p95" in _variants(ctx)
    d16 = (f"Removing the free-motion label-axis floor changes "
           f"{100 * _f(lab_agree.changed_frac.max()):.0f} % of the labels (Cohen's kappa "
           f"{_kappa_min(lab_agree)}): identical by construction"
           + (" (confirmed by its run)." if ran_p95 else " and was not run.")
           if not lab_agree.empty and "changed_frac" in lab_agree and "labels_p95" not in same
           else "")                         # a run identical to the primary is named in (c)
    same_txt = ("; identical to the primary analysis on every value shown (not drawn): "
                + ", ".join(VARIANT_TEXT.get(n, n) for n in same) if same else "")
    ranked = form == "ranked"
    lr, lc = ("b", "c") if ranked else ("c", "b")     # letters: sensitivity runs, condyles
    nv = len(_variants(ctx))
    runs = (f"({lr}) Cohort median [95 % CI] of the opening lead (TKEO) and of the closer "
            f"release before opening, primary analysis and {NUM_WORD.get(nv, nv)} sensitivity "
            "runs (defined in the robustness table)" + same_txt)
    cond = (f"({lc}) Median [95 % BCa CI] of the hinge residuals and the condyle excursion for "
            "all participants and for those whose palpated condyle points are anatomically "
            "plausible")
    cap = CaptionSpec(
        "(a) Every non-primary processing variant of four key outcomes (largest opening; "
        "opening lead in open-close; masseter R activation in chewing; digastric R activation "
        "in open-close): mean paired change of the participant values relative to the "
        "primary processing, in units of the between-participant SD, with its 95 % t "
        "interval"
        + (", one row per variant ranked by the size of the change and labelled with the "
           "outcome and what differs from the primary" if ranked else "")
        + " (lead reference: TKEO, earliest mechanical onset, driving-burst rule; the "
        "pre-specified first-onset rule is among the other rules; normalization: largest "
        "activity recorded). "
        + (f"{runs}. {cond}" if ranked else f"{cond}. {runs}"),
        ([("dashed lines in (a)", "± 1 SD")] if ranked else
         [("dots below (a)", "the outcome and what differs from the primary")])
        + ([] if blocks else
           [(f"rows in ({lr})", "the primary analysis, then each sensitivity run"),
            (f"colour in ({lr})", "task (orange open-close, green chewing, pink release)"),
            (f"marker in ({lr})", "rule (circle first onset, diamond driving burst)")]),
        f"n = {na} participants, {npl} with plausible condyles",
        f"(a) {big} of {len(est)} variants move the cohort mean by more than one "
        f"between-participant SD{big_where}" + (f"; ({lc}) {c1}" if c1 else ""),
        evidence="(a) pre-specified description",
        notes=(["† The run was compared with an earlier primary run, so its primary value "
                "differs from the current one."] if stale_any else []) + ([d16] if d16 else []))
    if form in ("per_outcome", "runs"):
        return _robustness_split(ctx, form, est, lo, hi, specs, hl, pc, pb, cap, n_rows,
                                 runs, cond, na, npl, c1, big, len(est), stale_any, d16)
    if form == "ranked":
        # (a) as a ranked dot plot: one labelled row per variant, largest change first
        order = sorted(range(len(est)), key=lambda q: -abs(est[q]) if np.isfinite(est[q])
                       else 0.0)
        names = [specs[q]["outcome"] + ": " + ", ".join(
            f"{v}" for kk, v in specs[q].items() if kk != "outcome") for q in order]
        pr = B.forest(names, [est[q] for q in order], [lo[q] for q in order],
                      [hi[q] for q in order], "(a) Processing choices, ranked",
                      Axis("Change (between-participant SD)"), ref=0.0, role="accent")
        for v in (-1.0, 1.0):
            pr.series.append(Series("vline", x=np.array([v]), role="threshold"))
        # (a) and (c) side by side (both long row lists), (b) below (inches: height_in = 1)
        pr.height = pb.height = max(1.0, 0.105 * len(order),
                                    (0.105 if blocks else 0.3) * n_rows)
        pc.height = 1.1
        pb.title, pc.title = "(b) Sensitivity runs", "(c) Plausible condyles"
        pc.legend = "below"
        return attach(FigureSpec("jnl_robustness", "Robustness of the findings", [pr, pb, pc],
                                 rows=2, cols=2, width="text", height_in=1.0,
                                 sources=["variability:spec_values", "variability:spec_curve",
                                          "sensitivity:comparison",
                                          "stats_kinematics:condyle_sensitivity",
                                          "uncertainty:hinge_noise", "labels:label_agreement"]),
                      cap)
    # Row 1: (a) and (b) at 2.1 in; row 2: the choice matrix and (c), which grows with its
    # rows (compact: about 0.24 in per analysis, five outcomes each; blocks: 0.12 in per
    # row). Sized so that figure and caption fit one text page.
    h = 1.8
    pm.height = max(1.0, 0.1 * len(pm.y.ticklabels or []) / h)
    pb.height = max(pm.height, (0.12 if blocks else 0.24) * n_rows / h)
    pb.legend = "below"
    return attach(FigureSpec("jnl_robustness", "Robustness of the findings", [pa, pc, pm, pb],
                             rows=2, cols=2, width="text", height_in=h,
                             sources=["variability:spec_values", "variability:spec_curve",
                                      "sensitivity:comparison",
                                      "stats_kinematics:condyle_sensitivity",
                                      "uncertainty:hinge_noise", "labels:label_agreement"]), cap)


ROBUST_OUTCOME = (("amplitude", "(b) Largest opening"),
                  ("masseter R", "(c) Masseter R activation, chewing"),
                  ("digastric R", "(d) Digastric R activation, open-close"))
ROBUST_SRC = ["variability:spec_values", "variability:spec_curve", "sensitivity:comparison",
              "stats_kinematics:condyle_sensitivity", "uncertainty:hinge_noise",
              "labels:label_agreement"]


def _robustness_split(ctx, form, est, lo, hi, specs, hl, pc, pb, cap, n_rows, runs, cond,
                      na, npl, c1, big, n_est, stale_any, d16) -> FigureSpec:
    """JNER revision 3.8 (2026-11-10): the specification curve mixed four outcomes on one
    axis. Form ``per_outcome``: (a) the opening lead (its 35 processing variants) as a
    specification curve over its choice matrix, (b to d) one small panel per other outcome
    with each variant as a row, (e) plausible condyles; all changes on one SD axis. Form
    ``runs``: the sensitivity runs alone (the former (c)), at full width."""
    if form == "runs":
        pb.title = ""
        pb.tag = ""
        pb.legend = "below"
        h = max(2.0, 0.22 * n_rows)
        spec = FigureSpec("jnl_robustness", "Sensitivity runs", [pb], rows=1, cols=1,
                          width="text", height_in=h, sources=ROBUST_SRC)
        txt = runs.split(") ", 1)[1].replace("lead (TKEO)", "lead with the TKEO detector")
        return attach(spec, CaptionSpec(
            txt[:1].upper() + txt[1:],
            [("rows", "the primary analysis, then each sensitivity run"),
             ("colour", "task (orange open-close, green chewing, pink release)"),
             ("marker", "rule (circle first onset, diamond driving burst)")],
            f"n = {na} participants", "cohort median with its 95 % BCa CI",
            evidence="descriptive",
            notes=(["† The run was compared with an earlier primary run, so its primary value "
                    "differs from the current one."] if stale_any else [])))
    lead = [i for i, sp in enumerate(specs) if sp["outcome"] == "lead"]
    dims = [("rule", [RULE_MATRIX["first"], RULE_MATRIX["drive_taskbase"]]),
            ("detector", ["envelope", "Hodges–Bui"]),
            ("motion", ["IMU velocity", "IMU accel.", "optical"])]
    sub = {k: i for k, i in enumerate(lead)}
    pa, pm = B.spec_curve([est[i] for i in lead], [lo[i] for i in lead], [hi[i] for i in lead],
                          [specs[i] for i in lead], "(a) Opening lead, open-close",
                          Axis("Change (between-participant SD)"), dims=dims,
                          highlight={k: hl[i] for k, i in sub.items() if i in hl}, ref=0.0,
                          xlabel="Processing variant, ranked", key="robust")
    pa.legend = "inside:north west"
    small = []
    for out, title in ROBUST_OUTCOME:
        ix = [i for i, sp in enumerate(specs) if sp["outcome"] == out]
        ix.sort(key=lambda i: est[i])
        names = [", ".join(str(v) for k, v in specs[i].items() if k != "outcome") for i in ix]
        p = B.forest(names, [est[i] for i in ix], [lo[i] for i in ix], [hi[i] for i in ix],
                     title, Axis("Change (SD)"), ref=0.0, role="context")
        p.sharex = "robust_sd"
        p.y.lim = (-0.6, len(names) - 0.4)
        small.append(p)
    pc.title = "(e) Plausible condyles"
    pc.legend = "inside:north east"
    xs = [v for p in small for sr in p.series if sr.kind in ("line", "scatter")
          for v in np.asarray(sr.x, float) if np.isfinite(v)]
    if xs:                              # one range for (b to d), from their own values
        span = max(xs) - min(min(xs), 0.0)
        lim = (min(min(xs), 0.0) - 0.15 * span, max(xs) + 0.15 * span)
        for p in small:
            p.x.lim = lim
    h = 1.55
    pm.height = max(0.75, 0.1 * len(pm.y.ticklabels or []) / h)
    for p in small[1:]:
        p.height = pm.height
    panels = [pa, small[0], pm, small[1], small[2], pc]
    # row 3: (d) and (e); heights from their rows
    small[2].height = pc.height = 0.75
    spec = FigureSpec("jnl_robustness", "Robustness of the findings", panels, rows=3, cols=2,
                      width="text", height_in=h, sources=ROBUST_SRC)
    return attach(spec, CaptionSpec(
        "every non-primary processing variant of four key outcomes, as the mean paired change "
        "of the participant values relative to the primary processing, in units of the "
        "between-participant SD, with its 95 % t interval. (a) Opening lead in open-close (lead "
        "reference: TKEO, earliest mechanical onset, driving-burst rule; the pre-specified "
        "first-onset rule is among the other rules), ranked, over the choices that define each "
        "variant. (b to d) Largest opening, masseter R activation in chewing, and digastric R "
        "activation in open-close, one row per variant (normalization reference: largest "
        "activity recorded). " + cond.replace("(b) ", "(e) ").replace("(c) ", "(e) "),
        [("dots below (a)", "what differs from the primary"),
         ("diamond in (a)", "the pre-specified onset rule"),
         ("dashed lines", "no change")],
        f"n = {na} participants, {npl} with plausible condyles",
        f"(a to d) {big} of {n_est} variants move the cohort mean by more than one "
        "between-participant SD" + (f". (e) {c1}" if c1 else ""),
        evidence="descriptive (pre-specified)",
        notes=[d16] if d16 else []))


# -- tables -------------------------------------------------------------------------------------------
DEC_FORMS = ("all_protocols", "offline_protocols", "realtime_protocols")


@register("jnl_table_decoding", "paper", "Intent decoding results (journal table)",
          choices={"form": DEC_FORMS}, form="all_protocols")
def jnl_table_decoding(req: ViewRequest) -> TableSpec:
    """Table III: feature set x protocol (unseen, same participant, calibrated, stream).
    ``form``: every protocol, the offline protocols only, or the simulated real time only
    (the split keeps each part within a page; 2026-11-05)."""
    ctx = req.ctx
    form = req.options.get("form", "all_protocols")
    offline, online = form != "realtime_protocols", form != "offline_protocols"
    sm = _t(ctx, "ml_eval", "summary")
    vs = _t(ctx, "ml_eval", "variants_summary")
    on = _t(ctx, "ml_online", "summary")
    if not on.empty and "applicable" in on:
        on = on[on.applicable.astype(bool)]
    fsh = _few_shot(ctx)

    def summ(protocol: str, fs: str, metric: str) -> pd.Series | None:
        r = _sel(sm, protocol=protocol, feature_set=fs, model="svm", metric=metric)
        return r.iloc[0] if not r.empty else None

    def ba_ci(r: pd.Series | None) -> str:
        return "--" if r is None else f"{r['median']:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"

    def num(r: pd.Series | None, d: int = 0) -> str:
        return "--" if r is None else _n(_f(r["median"]), d)

    rows: list[dict[str, Any]] = []
    blank = {"fs": "", "ba": "", "recall": "", "lat": "", "fa": "", "better": "", "p": ""}
    # evidence per feature set (2026-11-10, Methods 2.1 wording): the three
    # sets of the pre-specified comparison are confirmatory under leave-one-subject-out and
    # descriptive under the pre-specified secondary blocked CV; the sets without the
    # digastric were added with the montage fix after outcomes were known (post hoc). The
    # evidence sits in the group heading (constant within a group) and, where a row differs,
    # after its feature set: an evidence column squeezed the first column to one letter per
    # line (2026-11-10, P1)
    EV_ROW = {"emg_nodig": "exploratory (post hoc)", "emg_nodig_imu": "exploratory (post hoc)"}
    groups = [("Unseen participant (leave-one-subject-out)", "loso", ("emg_imu", "imu", "emg", "emg_nodig",
                                              "emg_nodig_imu"), None),
              ("Same participant (blocked cross-validation)", "blocked",
               ("emg_imu", "imu", "emg"), "descriptive")]
    groups[0] = (*groups[0][:3], "confirmatory")
    for title, proto, sets, ev in groups if offline else ():
        have = [fs for fs in sets if summ(proto, fs, "balanced_accuracy") is not None]
        if not have:
            continue
        if rows:
            rows.append({"_rule": True})
        rows.append({**blank, "fs": f"{title}, {ev}"})
        d_all = {fs: _loso_diff(ctx, fs, "imu", proto) for fs in have if fs != "imu"}
        for fs in have:
            rec = _recall(ctx, fs) if proto == "loso" else None
            d = d_all.get(fs)
            lab = FSET[fs] + (f", {EV_ROW[fs]}" if proto == "loso" and fs in EV_ROW else "")
            rows.append({**blank, "fs": lab, "ba": ba_ci(summ(proto, fs, "balanced_accuracy")),
                         "recall": (f"{np.nanmin(rec):.0f}–{np.nanmax(rec):.0f}"
                                    if rec is not None else "--"),
                         "lat": num(summ(proto, fs, "latency_ms_median")) if proto == "loso"
                         else "--",
                         "fa": num(summ(proto, fs, "fa_per_min"), 1) if proto == "loso"
                         else "--",
                         "better": (f"{int((d > 0).sum())}/{d.size}" if d is not None and d.size
                                    else "--"),
                         "p": _p(_test_p(ctx, "imu")) if (proto == "loso" and fs == "emg_imu")
                         else "--"})
    if offline and fsh and not vs.empty:
        rows += [{"_rule": True},
                 {**blank, "fs": f"Unseen, calibrated with {_cal_s(ctx, max(fsh))} per "
                                 "movement, exploratory (post hoc)"}]
        k = max(fsh)
        for fs in ("emg_imu", "imu"):
            r = _sel(vs, variant="few_shot", feature_set=fs, k=k, metric="balanced_accuracy")
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({**blank, "fs": FSET[fs],
                         "ba": f"{r['median']:.2f} (IQR {r.q1:.2f}–{r.q3:.2f})",
                         "better": f"{fsh[k]['better']}/{fsh[k]['n']}" if fs == "emg_imu"
                         else "--",
                         "p": _p(fsh[k]["p_holm"]) if fs == "emg_imu" else "--"})
    if online and not on.empty:
        rows += [{"_rule": True}] if rows else []
        rows += [{**blank, "fs": "Unseen, simulated real time, exploratory (post hoc)"}]
        piv = on.pivot_table(index="pid", columns=["feature_set", "mode"], values="ba_raw")
        for fs, mode, lab, g in _online_rows(on):
            m, q1, q3 = _med_iqr(g.ba_raw)
            better = "--"
            if fs == "emg_imu" and ("imu", mode) in piv and (fs, mode) in piv:
                d = (piv[(fs, mode)] - piv[("imu", mode)]).dropna()
                better = f"{int((d > 0).sum())}/{d.size}"
            rows.append({**blank, "fs": f"{FSET[fs]}, {MODE[mode]}",
                         "ba": f"{m:.2f} (IQR {q1:.2f}–{q3:.2f})",
                         "lat": _n(_med_iqr(g.latency_ms_median)[0]),
                         "fa": _n(_med_iqr(g.fa_per_min_rest)[0], 1), "better": better})
    if not rows:
        rows.append({**blank, "fs": "decoding results not available"})
    # a data row shows "--" for every value it does not have; only the group headings stay
    # blank
    for r in rows:
        if not r.get("_rule") and r.get("ba"):
            for k in ("recall", "lat", "fa", "better", "p"):
                if r.get(k) == "":
                    r[k] = "--"
    neg_lat = any(str(r.get("lat", "")).startswith("\u2212") for r in rows)
    cols = [Column("fs", "Protocol, features", align="X"),
            Column("ba", "Balanced accuracy", align="l"), Column("recall", "Recall (%)",
                                                                  align="l"),
            Column("lat", "Lat. (ms)", align="l"), Column("fa", "FA (/min)", align="l"),
            Column("better", "> IMU", align="l"), Column("p", "Holm p", align="l")]
    if not offline:                       # no recall or test in the real-time rows
        cols = [c for c in cols if c.key not in ("recall", "p")]
    note_off = ("EMG features are relative to each participant's own rest phase, and the "
                "within-participant blocks are cut inside every phase (both post hoc).")
    note_on = ("Simulated real time replays the recordings through a causal chain. No device "
               "was used. Calibration block: calibrated on a block of windows after the rest "
               "phase. Per phase: calibrated on the first windows of every phase.")
    n = int(_f(summ("loso", "emg_imu", "balanced_accuracy").n)) if summ(
        "loso", "emg_imu", "balanced_accuracy") is not None else 0
    t = TableSpec("jnl_table_decoding", "", cols, rows,
                  notes=" ".join(x for x in (note_off if offline else "",
                                             note_on if online else "") if x)
                        + " --: not defined or not tested for that row."
                        + (" Negative latency: the median first correct decision came "
                           "before the optical label onset. For a decoder near chance (EMG "
                           "without the digastric), it and the false activations reflect "
                           "guessing, not anticipation." if neg_lat else ""),
                  sources=["ml_eval:summary", "ml_eval:loso", "ml_eval:blocked",
                           "ml_eval:variants", "ml_eval:variants_summary",
                           "ml_eval:tests", "ml_online:summary"], width="text")
    marks = [("recall", "range over the seven classes of the per-class recall, each the "
                        "median over participants (leave-one-subject-out)"),
             ("lat.", "median time from the optical label onset to the first correct "
                      "decision"),
             ("FA", "false activations per minute at rest"),
             ("> IMU", "participants whose balanced accuracy is higher than with IMU "
                       "features alone")]
    if not offline:
        marks = [m for m in marks if m[0] != "recall"]
    n_on = int(on.pid.nunique()) if not on.empty and "pid" in on else 0
    return attach(t, CaptionSpec(
        f"{NUM_WORD.get(len(CLASSES), len(CLASSES))}-class intent decoding (SVM; chance "
        f"level 1/{len(CLASSES)})" + (", simulated real time" if not offline else
                                      ", offline" if not online else ""),
        marks, f"n = {n if offline else n_on} participants",
        "balanced accuracy: median [95 % BCa CI] over participants, or median (IQR) where no "
        "CI is stored" if offline else "balanced accuracy: median (IQR) over participants",
        "Holm p: EMG + IMU vs IMU, Wilcoxon signed-rank (calibrated: exact test, Holm over the "
        f"{_n_amounts(fsh) or 'stored'} calibration amounts including none)" if offline
        else ""))


def _robustness_matrix(matrix: list[tuple[str, bool, list[dict[str, Any]]]], n_primary: float,
                       based: bool, stale_any: bool, no_dec_txt: str, n_runs: int,
                       lab_agree: pd.DataFrame) -> TableSpec:
    """The ``run_matrix`` form of ``jnl_table_robustness``: rows = the primary analysis and each
    run, columns = outcomes; a run cell is marked when its conclusion differs."""
    outs = list(dict.fromkeys(e["outcome"] for _, _, block in matrix for e in block))
    keys = [f"o{i}" for i in range(len(outs))]
    prim_row: dict[str, Any] = {"run": "Primary analysis" + (
        f" ({_n(n_primary)} adults)" if based and np.isfinite(n_primary) else "")}
    rows: list[dict[str, Any]] = [prim_row]
    for name, stale, block in matrix:
        by = {e["outcome"]: e for e in block}
        row: dict[str, Any] = {"run": (VARIANT_SHORT.get(name) or VARIANT_LABEL.get(name, name))
                               + (" †" if stale else "")}
        for k, o in zip(keys, outs):
            e = by.get(o)
            if e is None:
                row[k] = ""
                continue
            prim_row.setdefault(k, e["short_p"])
            cell = "–" if e["short_v"] == "not re-run" else e["short_v"]
            row[k] = cell + (" *" if e["same"] == "no" else "")
        rows.append(row)
    rows.insert(1, {"_rule": True})
    cols = [Column("run", "Analysis", align="X")] + [
        Column(k, _matrix_head(o), align="r") for k, o in zip(keys, outs)]
    # 2026-11-13: without what the caption says (cohort medians, participants with a value,
    # the meaning of the asterisk)
    notes = ["Lead and release: ms before the mechanical onset. Release detection: detected "
             "vs chance (%). Activation: each participant's median envelope in the task, % of "
             "the largest recorded activity (the activation table gives the 95th "
             "percentile). Decoding: "
             "balanced accuracy for an unseen participant. Hinge: participants whose residual "
             "is above their in-vivo noise. Conclusion: the decision at p < 0.05 (detection: "
             "above chance; noise comparison: at most one participant below its noise). The "
             "CIs and p values are given in the robustness figure."]
    if no_dec_txt:
        notes.append(no_dec_txt.replace("was not re-run", "was not re-run (–)"))
    if (not lab_agree.empty and "changed_frac" in lab_agree
            and "labels_p95" not in {name for name, _, _ in matrix}):
        notes.append(f"{VARIANT_LABEL['labels_p95'][:1].upper()}"
                     f"{VARIANT_LABEL['labels_p95'][1:]}: "
                     f"{100 * _f(lab_agree.changed_frac.max()):.0f} % of the free-motion labels "
                     f"change (Cohen's kappa {_kappa_min(lab_agree)}); identical by "
                     "construction, not drawn as a row.")
    if stale_any:
        notes.append("† compared with an earlier primary run, so its values and conclusions "
                     "refer to that run, not to the current analysis.")
    t = TableSpec("jnl_table_robustness", "", cols, rows, notes=" ".join(notes),
                  sources=["sensitivity:comparison", "labels:label_agreement",
                           "stats_timing:timing_cohort", "stats_timing:ensemble_cohort"],
                  width="text")
    return attach(t, CaptionSpec(
        "key outcomes under the primary analysis and under "
        f"{NUM_WORD.get(n_runs, n_runs)} sensitivity run{'s' if n_runs != 1 else ''}, one row "
        "per analysis, labelled by what differs from the primary analysis (every run is "
        "defined in the analysis plan)",
        [("*", "conclusion differs from the primary analysis")],
        n=(f"n = {_n(n_primary)} participants in the primary analysis and in each run"
           if np.isfinite(n_primary) else ""),
        statistic="cohort medians; participants with a value in parentheses",
        test="the conclusion uses the one-sided signed-rank test (Holm-adjusted where stored)"))


def _decide(kind: str, prim: dict[str, float], var: dict[str, float]) -> str:
    def sig(d: dict[str, float]) -> bool | None:
        if kind == "descriptive":
            return None
        p = d.get("p_holm", np.nan)
        p = p if np.isfinite(p) else d.get("p", np.nan)
        if kind == "above_chance":
            v = d.get("detected_minus_chance_median", np.nan)
            return None if not np.isfinite(v) else bool(v > 0)
        if kind == "count":
            return None if not np.isfinite(d.get("v", np.nan)) else bool(d["v"] >= d["n"] - 1)
        return None if not np.isfinite(p) else bool(p < 0.05)

    a, b = sig(prim), sig(var)
    if kind == "descriptive":
        return "descriptive"
    if a is None or b is None:
        return "--"
    return "yes" if a == b else "no"


# Matrix form of the robustness table: short column headers per outcome (the long labels of
# the block form are the keys).
MATRIX_HEAD = {"Closer release before opening": "Closer release (ms)",
               "Causal release detection": "Release detected vs chance (%)",
               "Masseter R activation, chewing": "Masseter R, chewing (%)",
               "Decoding, EMG + IMU, unseen": "Decoding, unseen (balanced accuracy)",
               "Hinge residual above in-vivo noise": "Hinge above noise"}


def _matrix_head(label: str) -> str:
    if label.startswith("Opening lead, "):          # "Opening lead, <task>, <rule>"
        _, task, rule = (x.strip() for x in label.split(",", 2))
        return f"{task[:1].upper()}{task[1:]} lead, {rule} (ms)"
    return MATRIX_HEAD.get(label, label)


@register("jnl_table_robustness", "paper", "Robustness of the findings (journal table)",
          choices={"form": ("run_matrix", "run_blocks"), "detector": ("yes", "no")},
          form="run_matrix", detector="yes")
def jnl_table_robustness(req: ViewRequest) -> TableSpec:
    """Table IV: key outcome x sensitivity variant, primary vs variant, conclusion unchanged.

    Forms: ``run_matrix`` (default: one row per analysis -- the primary, then every run -- and
    one column per outcome; compact cells, a mark where the conclusion differs) and
    ``run_blocks`` (one block per run with the primary and the run value of every outcome,
    with CIs and p values: about five pages). ``detector="no"`` leaves out the causal
    release detection (a paper that does not report it)."""
    ctx = req.ctx
    form = req.options.get("form", "run_matrix")
    matrix: list[tuple[str, bool, list[dict[str, Any]]]] = []  # (run, stale, block)
    rows: list[dict[str, Any]] = []
    stale_any = False
    ens = _sel(_t(ctx, "stats_timing", "ensemble_cohort"), task="open_close",
               direction="opening", role="antagonist")
    hs = _t(ctx, "stats_kinematics", "hinge_noise_summary")
    cur_ens = {"median": _f(ens["median"].iloc[0]), "n": _f(ens.n.iloc[0]),
               "p": _f(ens.p.iloc[0])} if not ens.empty else {}
    cur_hinge = {"above_invivo_n": _f(hs.above_invivo_n.iloc[0])} \
        if not hs.empty and "above_invivo_n" in hs else {}
    blank = {"outcome": "", "primary": "", "value": "", "same": ""}
    from biomech import sensitivity as S

    ml_sum = _t(ctx, "ml_eval", "summary")
    names = _variants(ctx)
    # A variant store (the paper's reporting_population, D35): the sensitivity runs are run on, and
    # compared with, the primary analysis (all participants), so the stale check takes the
    # current primary values from this store's own comparison (the same primary run), not
    # from the variant's tables; the primary column then shows the primary analysis.
    # D36: runs built on this store compare with it, so the current values are this
    # store's own and the primary column is this store's population.
    based = bool(_based_on_store(ctx, names))
    base = _comparison(ctx, ctx.config.variant) if ctx.config.variant and not based \
        else pd.DataFrame()
    if base.empty:
        demo = _t(ctx, "qc_cohort", "demographics")
        n_primary = float(demo.n.iloc[0]) if not demo.empty and "n" in demo else np.nan
    else:
        n_primary = _cmp(base, "demographics", "n")[0]
    skipped_ml: list[str] = []     # variants that did not re-run the decoding (skip_stages)
    for name in names:
        c = _comparison(ctx, name)
        if c.empty:
            continue
        no_ml = "ml_eval" in S.not_run(c)
        # (label, table, key fields, decision kind, current values of the columns that must
        # match for the comparison's primary to be the current analysis)
        items: list[tuple[str, str, dict[str, str], str, dict[str, float]]] = []
        for t, r in C2_KEYS:
            cur = _c2_rows(ctx, t)[r]
            items.append((f"Opening lead, {V.task_label(t)}, {RULE[r]}", "timing_cohort",
                          _lead_fields(t, r), "test",
                          {k: cur[k] for k in ("median", "n", "p_holm")}))
        items.append(("Closer release before opening", "ensemble_cohort",
                      {"task": "open_close", "direction": "opening", "role": "antagonist"},
                      "test", cur_ens))
        if req.options.get("detector", "yes") != "no":   # "no": a paper without X3 (2026-11-12)
            items.append(("Causal release detection", "release_cohort",
                          {"task": "open_close", "primary": "True"}, "above_chance", {}))
        items.append(("Masseter R activation, chewing", "activation_cohort",
                      {"task": "chewing", "muscle": "masseter_r"}, "descriptive", {}))
        items.append(("Decoding, EMG + IMU, unseen", "summary",
                      {"protocol": "loso", "feature_set": "emg_imu", "model": "svm",
                       "metric": "balanced_accuracy"}, "descriptive", {}))
        items.append(("Hinge residual above in-vivo noise", "hinge_noise_summary", {}, "count",
                      cur_hinge))
        block: list[dict[str, Any]] = []
        stale_block = False
        cols = ("median", "ci_lo", "ci_hi", "n", "p", "p_holm", "detected_frac_median",
                "chance_detected_frac_median", "detected_minus_chance_median",
                "above_invivo_n")
        for lab, table, fields, kind, now in items:
            if table == "summary" and no_ml:
                cur = _sel(ml_sum, **fields)
                prim = f"{_f(cur['median'].iloc[0]):.3f}" if not cur.empty else "--"
                block.append({"outcome": lab, "primary": prim, "value": "not re-run",
                              "same": "--", "short_p": prim[:-1] if prim != "--" else prim,
                              "short_v": "not re-run"})
                skipped_ml.append(name)
                continue
            if not base.empty:
                now = {col: _cmp(base, table, col, **fields)[0] for col in now}
            v = {col: _cmp(c, table, col, **fields) for col in cols}
            P = {col: v[col][0] for col in cols}
            Q = {col: v[col][1] for col in cols}
            if not any(np.isfinite(x) for x in P.values()):
                continue
            stale_block |= any(np.isfinite(val) and np.isfinite(P[col])
                               and abs(P[col] - val) > 1e-6 * max(1.0, abs(val))
                               for col, val in now.items())
            if kind == "count":
                for d, j in ((P, 0), (Q, 1)):
                    d["v"], d["n"] = d["above_invivo_n"], _cmp(c, table, "n")[j]

            def fmt(d: dict[str, float]) -> str:
                if kind == "count":
                    return f"{_n(d['v'])}/{_n(d['n'])}"
                if table == "release_cohort":
                    return (f"{100 * d['detected_frac_median']:.0f} % vs "
                            f"{100 * d['chance_detected_frac_median']:.0f} % chance")
                if table == "summary":
                    return f"{d['median']:.3f}"
                if table == "activation_cohort":
                    return f"{d['median']:.1f} %"
                txt = f"{_n(d['median'])} ms {_ci(d['ci_lo'], d['ci_hi'])}".strip()
                if np.isfinite(d["n"]):
                    txt += f", n = {_n(d['n'])}"
                pv = d["p_holm"] if np.isfinite(d["p_holm"]) else d["p"]
                if np.isfinite(pv):
                    txt += f", {'Holm ' if np.isfinite(d['p_holm']) else ''}p {_peq(pv)}"
                return txt

            def short(d: dict[str, float]) -> str:
                """Matrix cell: the value without CI and p (those are in the block form)."""
                if kind == "count":
                    return f"{_n(d['v'])}/{_n(d['n'])}"
                if table == "release_cohort":
                    return (f"{100 * d['detected_frac_median']:.0f} vs "
                            f"{100 * d['chance_detected_frac_median']:.0f}")
                if table == "summary":
                    return f"{d['median']:.2f}"
                if table == "activation_cohort":
                    return f"{d['median']:.1f}"
                txt = _n(d["median"])
                return txt + (f" ({_n(d['n'])})" if np.isfinite(d["n"]) else "")

            block.append({"outcome": lab, "primary": fmt(P), "value": fmt(Q),
                          "same": _decide(kind, P, Q), "short_p": short(P),
                          "short_v": short(Q)})
        if not block:
            continue
        if rows:
            rows.append({"_rule": True})
        vlab = VARIANT_LABEL.get(name, name)
        rows.append({**blank, "outcome": f"Sensitivity run: {vlab}" + (" †" if stale_block
                                                                        else "")})
        stale_any |= stale_block
        rows += block
        matrix.append((name, stale_block, block))
    lab_agree = _t(ctx, "labels", "label_agreement")
    if not lab_agree.empty and "changed_frac" in lab_agree:
        if rows:
            rows.append({"_rule": True})
        rows.append({**blank, "outcome": f"Sensitivity run: {VARIANT_LABEL['labels_p95']}"})
        rows.append({"outcome": "Free-motion labels (decoding)",
                     "primary": f"{100 * _f(lab_agree.changed_frac.max()):.0f} % of labels "
                                f"changed, Cohen's kappa {_kappa_min(lab_agree)}",
                     "value": "identical by construction, not run", "same": "yes"})
    if not rows:
        rows.append({**blank, "outcome": "no sensitivity run available"})
    cols_t = [Column("outcome", "Outcome", align="X"),
              Column("primary", "Primary analysis" + (
                  (f" ({_n(n_primary)} adults)" if based else f" (all {_n(n_primary)})")
                  if np.isfinite(n_primary) else ""),
                     align="l"),
              Column("value", "Sensitivity run", align="l"),
              Column("same", "Same conclusion", align="l")]
    no_dec = [VARIANT_SHORT.get(n, n) for n in dict.fromkeys(skipped_ml)]
    no_dec_txt = ("The decoding was not re-run in the " + (
        series(no_dec))
        + f" run{'s' if len(no_dec) > 1 else ''} (skipped for computing time)."
        if no_dec else "")
    n_runs = sum(1 for n in names if not _comparison(ctx, n).empty)
    if form == "run_matrix" and matrix:
        return _robustness_matrix(matrix, n_primary, based, stale_any, no_dec_txt, n_runs,
                                  lab_agree)
    t = TableSpec("jnl_table_robustness", "", cols_t, rows,
                  notes=("† compared with an earlier primary run (before the channel-cycle minimum, "
                         "the confirmatory family of four tests or the code corrections), so its "
                         "primary values and its conclusions refer to that run, not to the "
                         "current analysis."
                         if stale_any else ""),
                  sources=["sensitivity:comparison", "labels:label_agreement",
                           "stats_timing:timing_cohort", "stats_timing:ensemble_cohort"],
                  width="text")
    return attach(t, CaptionSpec(
        "key outcomes under the primary analysis and under "
        f"{NUM_WORD.get(n_runs, n_runs)} sensitivity run{'s' if n_runs != 1 else ''} (one block "
        "each, labelled by what differs from the primary analysis)",
        [("same conclusion", "the same decision at p < 0.05 (detection: above chance; noise "
                             "comparison: at most one participant below its noise)")],
        n=(f"n = {_n(n_primary)} participants in the primary analysis and in each run"
           if np.isfinite(n_primary) else ""),
        statistic="lead and release: cohort median [95 % CI] in ms before the mechanical "
                  "onset and participants",
        test="p of the one-sided signed-rank test (Holm-adjusted where stored)",
        notes=[x for x in (
            "" if based or not np.isfinite(n_primary) else
            f"Primary and sensitivity runs include all {_n(n_primary)} participants (the participant recorded outside the protocol, "
            "participant included; the paper's main analyses report the adults)",
            no_dec_txt) if x]))
