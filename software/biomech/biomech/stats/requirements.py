# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Derived jaw-exoskeleton requirements (Table I / journal Table II): every number computed
once, with its statistic, interval, derivation and evidence level (stage ``requirements``).

**Percentile policy** (one rule for every capacity requirement; E31): the design value is
the **95th percentile across participants** of the participant-level metric (the device
should accommodate 95 % of users). With n = 14 (13 for the IMU bandwidth and for the
13 adults the dataset paper reports, the ``reporting_population`` variant, D35) P95 lies between the
two largest participants, and no finite two-sided 95 % interval exists: P(sample max > P95) =
1 - 0.95^14 = 0.51, and a percentile bootstrap of so extreme a quantile can never reach above
the sample maximum. Per capacity row:

* ``value``: the Harrell-Davis estimate of P95 (``scipy.stats.mstats.hdquantiles``), a
  weighted mean of all order statistics; ``value_linear``: the linear-interpolation P95
  (``numpy.percentile``) used up to r9, stored next to it;
* ``ci_lo``: the exact distribution-free one-sided lower 95 % bound, the order statistic
  X(k) with the largest k such that P(X(k) <= P95) = P(Binomial(n, 0.95) >= k) >= 0.95
  (``lower_k``, its achieved confidence ``lower_conf``; X(12) at n = 14 with 0.970, X(11) at
  n = 13 with 0.975); ``ci_hi`` is NaN and ``upper_note`` says "none finite at n = ..."
  (a one-sided upper bound needs P(X(n) >= P95) = 1 - 0.95^n >= 0.95, i.e. n >= 59);
* ``sample_max`` with its coverage statement: with confidence ``max_conf`` = 1 - 0.8^n
  (0.956 at n = 14) it exceeds the ``max_q`` = 80th percentile of the population;
* ``median`` for orientation.

Workspace values are **lower bounds**: no maximal opening was recorded (D1). Performance
values (misalignment, latency) are cohort medians with BCa CIs (``stats.core.describe``);
the IMU opening rows are the largest absolute repeated-measures limit of agreement with a
MOVER interval (D26, ``stats.cohort_misc.max_abs_loa_ci``).

Each input keeps its own participant set (E32): optical rows use every participant with
optical data, IMU rows those with IMU data (``n`` per row).

Evidence levels: one map, :data:`biomech.stats.evidence.REQUIREMENT_EVIDENCE` (J5; checked against the analysis plan's evidence register), stored with every row -- e.g.
the best-fit hinge ``confirmatory``, the hinge through the palpated condyles
``secondary_post_hoc`` (A4, D33), the C2 leads ``not_confirmed`` (the result depends on the
onset rule), C3a's generic model ``primary``.

D27 sensitivity (:func:`requirements_sensitivity`, table ``requirements_sensitivity``):
every workspace and speed row also with one uniform per-participant statistic (the largest
cycle of the task), and the bandwidth f95 from the IMU alone and from the optics alone next
to the coherence-weighted primary; D29: the bandwidth with the noise-limited rule at its
fixed neighbours (coherent band below 1.5 / 3 Hz; primary 2 Hz).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from biomech.stats import evidence as EV
from biomech.stats.core import describe

P_DESIGN = 95.0
CONF = 0.95                   # one-sided confidence of the order-statistic bound (E31)
MAX_Q = 0.80                  # population quantile the sample maximum is stated to exceed
BANDWIDTH_TASKS = ("open_close", "chewing", "free_motion")   # opening rotation is the motion
ALPHA = 0.05
COLUMNS = ["key", "group", "requirement", "value", "ci_lo", "ci_hi", "median", "unit",
           "statistic", "n", "derivation", "evidence", "implication", "note",
           # E31 (capacity rows)
           "value_linear", "lower_k", "lower_conf", "upper_note", "sample_max", "max_q",
           "max_conf",
           # D23 / D31 (C2 latency-budget rows; the D31 per-source nulls were withdrawn by
           # D34: the onset sources only re-reference the same EMG onset)
           "lead_null_ms", "lead_minus_null_ms", "p_null", "p_null_holm",
           "chance_distinguishable", "lead_omega_ms", "lead_opt_ms",
           # D26 (IMU opening rows)
           "ci_method", "ci_lo_boot", "ci_hi_boot", "value_insample"]
SENS_COLUMNS = ["key", "variant", "requirement", "value", "value_linear", "ci_lo", "lower_k",
                "sample_max", "median", "n", "unit", "primary_value", "derivation", "note"]


# -- E31: P95 with an exact order-statistic bound -----------------------------------------
def order_stat_lower(n: int, p: float = P_DESIGN / 100, conf: float = CONF
                     ) -> tuple[int, float]:
    """(k, achieved confidence) of the one-sided lower ``conf`` bound X(k) of the population
    ``p``-quantile from ``n`` values: the largest k with P(X(k) <= Q_p) =
    P(Binomial(n, p) >= k) >= ``conf``. (0, NaN) when no order statistic qualifies."""
    best = (0, np.nan)
    for k in range(1, int(n) + 1):
        c = float(sps.binom.sf(k - 1, n, p))
        if c >= conf:
            best = (k, c)
    return best


def order_stat_upper(n: int, p: float = P_DESIGN / 100, conf: float = CONF
                     ) -> tuple[int, float]:
    """(k, achieved confidence) of the one-sided upper ``conf`` bound X(k): the smallest k
    with P(X(k) >= Q_p) = P(Binomial(n, p) <= k - 1) >= ``conf``; (0, NaN) when none exists
    (n < 59 for the 95th percentile at 95 %)."""
    for k in range(1, int(n) + 1):
        c = float(sps.binom.cdf(k - 1, n, p))
        if c >= conf:
            return k, c
    return 0, np.nan


def harrell_davis(x: np.ndarray, p: float = P_DESIGN / 100) -> float:
    """Harrell-Davis estimate of the ``p``-quantile of the finite values (NaN if none; the
    single value if one)."""
    from scipy.stats.mstats import hdquantiles

    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    if v.size == 1:
        return float(v[0])
    return float(np.asarray(hdquantiles(v, prob=[p])).ravel()[0])


COVERAGE_N = range(13, 31)
COVERAGE_Q = (0.80, 0.90)


def coverage_table(ns: Any = COVERAGE_N, qs: tuple[float, ...] = COVERAGE_Q,
                   conf: float = CONF) -> pd.DataFrame:
    """X8 (post hoc, planning only): the chance that the largest of n participants
    exceeds the population q-quantile, ``p_max_exceeds`` = 1 - q^n, for every n and q, and per
    q the smallest n that reaches ``conf`` (row with ``smallest_n``). Uses no data: what more
    participants buy."""
    rows = [{"q": float(q), "n": int(n), "p_max_exceeds": 1.0 - float(q) ** int(n),
             "smallest_n": False} for q in qs for n in ns]
    for q in qs:
        need = int(np.ceil(np.log(1.0 - conf) / np.log(float(q))))
        rows.append({"q": float(q), "n": need, "p_max_exceeds": 1.0 - float(q) ** need,
                     "smallest_n": True})
    out = pd.DataFrame(rows)
    out["conf"] = conf
    return out


def p95_design(x: np.ndarray, p: float = P_DESIGN) -> dict[str, Any]:
    """The E31 design-value block of one capacity requirement (module docstring)."""
    v = np.sort(np.asarray(x, float))
    v = v[np.isfinite(v)]
    n = int(v.size)
    out: dict[str, Any] = {"value": np.nan, "value_linear": np.nan, "ci_lo": np.nan,
                           "ci_hi": np.nan, "lower_k": 0, "lower_conf": np.nan,
                           "upper_note": "", "sample_max": np.nan, "max_q": MAX_Q,
                           "max_conf": np.nan, "median": np.nan, "n": n}
    if n == 0:
        return out
    q = p / 100
    k, c = order_stat_lower(n, q)
    ku, cu = order_stat_upper(n, q)
    out.update(value=harrell_davis(v, q), value_linear=float(np.percentile(v, p)),
               ci_lo=float(v[k - 1]) if k else np.nan, lower_k=k, lower_conf=c,
               ci_hi=float(v[ku - 1]) if ku else np.nan,
               upper_note="" if ku else f"none finite at n = {n}",
               sample_max=float(v[-1]), max_conf=float(1 - MAX_Q ** n),
               median=float(np.median(v)))
    return out


def _blank() -> dict[str, Any]:
    return {c: np.nan for c in COLUMNS}


def _cap(key: str, group: str, req: str, x: np.ndarray, unit: str, derivation: str,
         implication: str, note: str = "") -> dict[str, Any]:
    d = p95_design(x)
    stat = (f"P{P_DESIGN:.0f} across participants (Harrell-Davis; one-sided lower 95 % bound "
            f"X({d['lower_k']}) of {d['n']}; maximum covers >= {100 * MAX_Q:.0f} % with "
            f"confidence {d['max_conf']:.3f})")
    return {**_blank(), "key": key, "group": group, "requirement": req, **d, "unit": unit,
            "statistic": stat, "derivation": derivation, "evidence": EV.requirement(key),
            "implication": implication, "note": note}


def _perf(key: str, group: str, req: str, x: np.ndarray, unit: str, derivation: str,
          implication: str, seed: int, note: str = "") -> dict[str, Any]:
    d = describe(np.asarray(x, float), seed=seed, stat="median")
    return {**_blank(), "key": key, "group": group, "requirement": req, "value": d["median"],
            "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "median": d["median"], "unit": unit,
            "statistic": "median across participants (BCa CI)", "n": d["n"],
            "derivation": derivation, "evidence": EV.requirement(key),
            "implication": implication,
            "note": note}


def _envelope_rows(s: pd.DataFrame) -> list[dict[str, Any]]:
    """X7 (post hoc description): closing speed, free-motion opening speed and peak
    angular acceleration, each with the S1 design rule; lower bounds of the 40-Hz / 6-Hz chain
    (VD9, VD24 give the attenuation against 200-Hz optics)."""
    rows = []
    for key, col, req, unit, how, imp in (
            ("close_vel_dps", "peak_close_vel_max_dps", "Peak closing angular velocity", "deg/s",
             "self-paced closing, per participant the largest cycle peak",
             "actuator speed when closing"),
            ("free_peak_open_vel_dps", "free_peak_open_vel_dps",
             "Peak opening angular velocity, free motion", "deg/s",
             "free-motion take, per participant the largest opening velocity",
             "actuator speed in unconstrained use"),
            ("peak_acc_dps2", "peak_acc_dps2", "Peak angular acceleration", "deg/s^2",
             "per participant the largest |angular acceleration| over open-close, chewing and "
             "free motion", "actuator torque per unit inertia")):
        if col in s and s[col].notna().any():
            rows.append(_cap(key, "dynamics", req, s[col].to_numpy(float), unit, how, imp,
                             "lower bound (40-Hz optics, 6-Hz low-pass); post hoc"))
    return rows


def _get(r: pd.Series | None, col: str) -> float:
    if r is None or col not in r.index:
        return np.nan
    v = r[col]
    return float(v) if pd.notna(v) else np.nan


def _lead_row(timing: pd.DataFrame, rule: str, lab: str) -> dict[str, Any] | None:
    """C2 latency-budget row of one onset rule (open_close opening, TKEO) with its D23 null
    and the D31 onset-source sensitivity (lead against the angular-velocity and the optical
    onset). No per-source null (D34: a source only re-references the same EMG onset)."""
    base = timing[(timing.task == "open_close") & (timing.direction == "opening")
                  & (timing.detector == "tkeo") & (timing.rule == rule)]

    def metric(m: str) -> pd.Series | None:
        q = base[base.metric == m]
        return q.iloc[0] if len(q) else None

    r = metric("lead_ms")
    if r is None:
        return None
    om, op = metric("lead_omega_ms"), metric("lead_opt_ms")
    p_null_holm = _get(r, "p_null_holm")
    distinct = (bool(p_null_holm < ALPHA) if np.isfinite(p_null_holm) else np.nan)
    note = [f"Holm p = {r.p_holm:.3f}"] if "p_holm" in r.index and pd.notna(r.p_holm) else []
    if np.isfinite(_get(r, "lead_null_ms")):
        note.append(f"chance null median {_get(r, 'lead_null_ms'):.0f} ms, Holm p vs chance "
                    f"= {p_null_holm:.3f}")
        if distinct is False:
            note.append("not distinguishable from chance")
    return {**_blank(), "key": f"lead_{rule}_ms", "group": "intent",
            "requirement": f"EMG-to-motion lead, opening ({lab})",
            "value": r["median"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
            "median": r["median"], "unit": "ms",
            "statistic": "median across participants (BCa CI)", "n": int(r.n),
            "derivation": "digastric TKEO onset to mechanical opening onset, "
                          "participants with >= 3 usable cycles (D15)",
            # Both rules are in the confirmatory Holm family (D15); the driving-burst
            # rule itself was defined post hoc (D6). The paper label comes from vocab.
            "evidence": EV.requirement(f"lead_{rule}_ms"),
            "implication": "time available before motion for an EMG trigger",
            "note": "; ".join(note),
            "lead_null_ms": _get(r, "lead_null_ms"),
            "lead_minus_null_ms": _get(r, "lead_minus_null_ms"),
            "p_null": _get(r, "p_null"), "p_null_holm": p_null_holm,
            "chance_distinguishable": distinct,
            "lead_omega_ms": _get(om, "median"), "lead_opt_ms": _get(op, "median")}


def _release_note(r: pd.Series) -> str:
    """X3 note (D34): the detection rate against its chance level carries the statement; the
    anticipation is compared with the chance anticipation (paired test, D23), and an
    anticipation not above chance is said to be so."""
    note = (f"detected {100 * r.detected_frac_median:.0f} % vs chance "
            f"{100 * r.chance_detected_frac_median:.0f} %, "
            f"{r.false_per_min_median:.0f} false/min")
    ch, pv = _get(r, "chance_anticipation_ms_median"), _get(r, "p_vs_chance")
    if np.isfinite(ch):
        note += (f"; anticipation {r['median']:.0f} ms vs chance {ch:.0f} ms"
                 + (f" (paired p = {pv:.3f})" if np.isfinite(pv) else ""))
        if not r["median"] > ch:
            note += ": not above chance, only the detection rate carries a statement"
    return note


MIN_BAND_HZ = 2.0             # primary noise-limited rule (imu.analysis.MIN_COHERENT_BAND_HZ)
MIN_BAND_NEIGHBOURS = (1.5, 3.0)   # D29: the rule's fixed neighbours, never tuned


def bandwidth_per_participant(bandwidth: pd.DataFrame, col: str = "f95_coh_hz",
                              min_band_hz: float = MIN_BAND_HZ) -> pd.Series:
    """Per participant the largest ``col`` over the opening tasks that are not
    noise-limited (D12): the participant value of the bandwidth requirement.

    Noise-limited (D29): the coherent band ``coh_band_hz`` ends below ``min_band_hz``; a NaN
    band (no coherence computed) counts as noise-limited. The default 2.0 Hz reproduces the
    stored ``noise_limited`` flag of the imu stage (the primary); without a ``coh_band_hz``
    column that flag is used."""
    if bandwidth.empty or col not in bandwidth:
        return pd.Series(dtype=float)
    if "coh_band_hz" in bandwidth:
        band = bandwidth.coh_band_hz.astype(float)
        limited = ~(band >= float(min_band_hz))            # NaN -> noise-limited
    else:
        limited = bandwidth.noise_limited.astype(bool)
    use = bandwidth[bandwidth.task.isin(BANDWIDTH_TASKS) & ~limited]
    return use.groupby("pid")[col].max()


def _c2e_rows(c2e: pd.DataFrame | None) -> list[dict[str, Any]]:
    """C2-E rows (X6, evidence ``post_hoc``) from ``stats_timing.c2e_cohort``: closer
    activation before closing (H1: the mixture location mu, BCa CI) and the null-calibrated
    closer release before opening (H2, open_close: the release lead of the t_omega-aligned
    median curve, its depth tested against the S1 null). The note carries the IUT p, the
    gatekeeping and the X6 decision."""
    if c2e is None or c2e.empty or "key" not in c2e:
        return []
    k = c2e.set_index("key")
    decision = str(k.loc["decision", "hypothesis"]) if "decision" in k.index else ""
    rows = []
    for key, req, deriv, imp in (
            ("h1", "Closer activation before closing",
             "masseter/temporalis TKEO onset (drive_taskbase) to the angular-velocity closing "
             "onset, both tasks; location of the time-locked component (mixture against the "
             "S1 surrogate background, X6)", "time for a closing trigger"),
            ("h2", "Closer release before opening (null-calibrated)",
             "closer group envelope aligned to the angular-velocity opening onset (X2 "
             "parameters), release depth against S1 surrogates (X6 P3)",
             "earliest EMG cue of an opening")):
        if key not in k.index:
            continue
        r = k.loc[key]
        p_adj = _get(r, "p_adj")
        note = (f"IUT p = {_get(r, 'p'):.3f} (locking {_get(r, 'p_lock'):.3f}, location "
                f"{_get(r, 'p_loc'):.3f})")
        if key == "h1":
            note += f"; event-order decision: {decision}"
        elif not np.isfinite(p_adj):
            note += "; not tested (H1 not rejected, serial gatekeeping)"
        else:
            note += f"; Holm p = {p_adj:.3f}"
        rows.append({**_blank(), "key": f"c2e_{key}_ms", "group": "intent",
                     "requirement": req, "value": _get(r, "median"),
                     "ci_lo": _get(r, "ci_lo"), "ci_hi": _get(r, "ci_hi"),
                     "median": _get(r, "median"), "unit": "ms",
                     "statistic": "median across participants (BCa CI)",
                     "n": int(r["n"]) if pd.notna(r["n"]) else 0, "derivation": deriv,
                     "evidence": EV.requirement(f"c2e_{key}_ms"), "implication": imp,
                     "note": note})
    return rows


def requirements_table(subjects: pd.DataFrame, hinge: pd.DataFrame, hinge_noise: pd.DataFrame,
                       bandwidth: pd.DataFrame, timing: pd.DataFrame,
                       ensemble: pd.DataFrame, release: pd.DataFrame, imu_agree: pd.DataFrame,
                       imu_ba: pd.DataFrame, seed: int, c2e: pd.DataFrame | None = None
                       ) -> pd.DataFrame:
    """One row per requirement (see the module docstring for the statistics).

    ``hinge``: per-participant ``kinematics.hinge``; ``hinge_noise``: per-participant
    ``uncertainty.hinge_noise``; ``timing``: ``stats_timing.timing_cohort`` (its rows carry
    the CIs and the D23 null); ``release``: ``stats_timing.release_cohort``; ``imu_ba``:
    ``stats_imu.imu_opening_ba``; ``c2e``: ``stats_timing.c2e_cohort`` (X6, post hoc rows
    "closer activation before closing" and "closer release before opening (null-calibrated)").
    Decoding (C3b) is not here: ``ml`` is an optional plugin,
    and the table view adds it from ``ml_eval.summary``.
    """
    rows: list[dict[str, Any]] = []
    s = subjects
    # -- workspace and kinematic capacity (S1, descriptive) -------------------------------------
    if not s.empty:
        rows.append(_cap("opening_mm", "workspace", "Incisal opening", s.max_observed_opening_mm,
                         "mm", "largest opening observed in any task (no MIO recorded, D1)",
                         "lower bound of the vertical incisal range",
                         "lower bound"))
        rows.append(_cap("protrusion_mm", "workspace", "Protrusion", s.protrusion_mm, "mm",
                         "protrusion task, peak anterior incisal displacement",
                         "anterior range of the incisal guide", "lower bound"))
        lat = s[["laterotrusion_left_mm", "laterotrusion_right_mm"]].max(axis=1)
        rows.append(_cap("laterotrusion_mm", "workspace", "Laterotrusion (larger side)", lat,
                         "mm", "left-right task, peak lateral incisal displacement",
                         "lateral range", "lower bound"))
        rows.append(_cap("opening_angle_deg", "workspace", "Opening rotation",
                         s.opening_angle_deg, "deg", "self-paced open-close, peak angle",
                         "rotational range of the hinge joint",
                         "self-paced, lower bound"))
        cond = s[["condyle_left_disp_mm", "condyle_right_disp_mm"]].max(axis=1)
        rows.append(_cap("translation_mm", "kinematics", "Passive translational DoF", cond,
                         "mm", "condyle-point excursion in opening, larger side (model-free, "
                         "D10)", "translation the joint must allow without "
                         "load"))
        rows.append(_cap("peak_vel_dps", "dynamics", "Peak opening angular velocity",
                         s.peak_open_vel_max_dps, "deg/s", "self-paced opening, per participant "
                         "the largest cycle peak", "actuator speed"))
        rows += _envelope_rows(s)
        if "posterior_excursion_mm" in s and s.posterior_excursion_mm.notna().any():
            rows.append(_cap("posterior_excursion_mm", "workspace",
                             "Posterior incisal excursion", s.posterior_excursion_mm, "mm",
                             "per participant the 99th percentile of the incisal displacement "
                             "behind the rest position over the five movement tasks",
                             "backward range of the incisal guide (opening carries the "
                             "incisor backward)", "lower bound; post hoc"))
    # -- C1: misalignment of a fixed hinge (confirmatory) ---------------------------------------
    if not hinge.empty:
        for m, lab, imp in (("hinge_fit", "Fixed-hinge misalignment (best-fit axis)",
                             "a fixed hinge misaligns the incisal point"),
                            ("hinge_tmj", "Hinge through palpated condyles",
                             "palpation-based alignment is worse")):
            v = hinge[hinge.model == m].rms_incisor_mm.to_numpy(float)
            note = ""
            if m == "hinge_fit" and not hinge_noise.empty and "above_invivo" in hinge_noise:
                from biomech.stats.cohort_kinematics import hinge_noise_counts

                k, n = hinge_noise_counts(hinge_noise, "invivo")
                note = f"above in-vivo noise P99 in {k}/{n}"
            rows.append(_perf(f"misalign_{m}_mm", "kinematics", lab, v, "mm",
                              "RMS incisal residual over opening cycles", imp, seed + 6,
                              note))
    # -- bandwidth (X1, D12) --------------------------------------------------------------------
    per = bandwidth_per_participant(bandwidth)
    if len(per):
        rows.append(_cap("bandwidth_hz", "dynamics", "Control bandwidth", per.to_numpy(float),
                         "Hz", "f95 of the IMU angular-velocity power confirmed by the optical "
                         "measurement (coherence), largest over the opening tasks (D12)",
                         "closed-loop bandwidth"))
    # -- C2 / X2 / X3: latency budget -----------------------------------------------------------
    if not timing.empty:
        for rule, lab in (("first", "first onset"), ("drive", "driving burst")):
            r = _lead_row(timing, rule, lab)
            if r is not None:
                rows.append(r)
    if not ensemble.empty:
        q = ensemble[(ensemble.task == "open_close") & (ensemble.direction == "opening")
                     & (ensemble.role == "antagonist")]
        if not q.empty:
            r = q.iloc[0]
            rows.append({**_blank(), "key": "closer_release_ms", "group": "intent",
                         "requirement": "Closer release before opening",
                         "value": r["median"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                         "median": r["median"], "unit": "ms",
                         "statistic": "median across participants (BCa CI)", "n": int(r.n),
                         "derivation": "onset-aligned group envelopes, every cycle (X2)",
                         "evidence": EV.requirement("closer_release_ms"),
                         "implication": "earliest EMG cue of an opening",
                         "note": f"before onset in {int(r.n_before)}/{int(r.n_found)}"})
    if not release.empty:
        q = release[(release.task == "open_close") & release.primary.astype(bool)]
        if not q.empty:
            r = q.iloc[0]
            rows.append({**_blank(), "key": "causal_release_ms", "group": "intent",
                         "requirement": "Causal closer-release detection",
                         "value": r["median"], "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
                         "median": r["median"], "unit": "ms",
                         "statistic": "median anticipation across participants (BCa CI)",
                         "n": int(r.n),
                         "derivation": "causal detector on the causal EMG chain (X3, E27)",
                         "evidence": EV.requirement("causal_release_ms"),
                         "implication": "achievable real-time anticipation",
                         "note": _release_note(r)})
    rows += _c2e_rows(c2e)
    # -- C3a: IMU ------------------------------------------------------------------------------
    if not imu_agree.empty:
        q = imu_agree[(imu_agree.task == "open_close") & (imu_agree.estimator == "quat")]
        if not q.empty:
            r = q.iloc[0]
            rows.append({**_blank(), "key": "imu_angle_rmse_deg", "group": "sensing",
                         "requirement": "IMU opening angle error", "value": r["median"],
                         "ci_lo": r.ci_lo, "ci_hi": r.ci_hi, "median": r["median"],
                         "unit": "deg", "statistic": "median RMSE across participants (BCa CI)",
                         "n": int(r.n), "derivation": "quaternion estimator vs optical, "
                                                      "one-time calibration (C3a)",
                         "evidence": EV.requirement("imu_angle_rmse_deg"),
                         "implication": "wearable angle sensing suffices", "note": ""})
    if not imu_ba.empty:
        rows += _imu_loa_rows(imu_ba)
    return pd.DataFrame(rows, columns=COLUMNS)


def _imu_loa_rows(imu_ba: pd.DataFrame) -> list[dict[str, Any]]:
    """C3a opening rows (D26): the generic leave-one-subject-out model first (the
    pre-specified C3a outcome), then the participant's own model fitted leave-one-cycle-out;
    the own model's in-sample value in ``value_insample`` (sensitivity)."""
    def pick(model: str) -> pd.Series | None:
        q = imu_ba[(imu_ba.task == "open_close") & (imu_ba.estimator == "quat")
                   & (imu_ba.model == model)]
        return q.iloc[0] if len(q) else None

    rows = []
    insample = pick("subject_insample")
    for model, lab, how in (("loso", "generic model", "leave-one-subject-out"),
                            ("subject", "own model", "leave-one-cycle-out")):
        r = pick(model)
        if r is None:
            continue
        # The error a user must expect: the larger absolute limit of agreement (bias
        # included, E34), with its own interval (D26).
        worst = _get(r, "max_abs_loa")
        if not np.isfinite(worst):
            worst = max(abs(r.loa_lo), abs(r.loa_hi))
        note = (f"bias {r.bias:+.2f} mm [{_get(r, 'bias_ci_lo'):+.2f}, "
                f"{_get(r, 'bias_ci_hi'):+.2f}]")
        ins = np.nan
        if model == "subject" and insample is not None:
            ins = _get(insample, "max_abs_loa")
            if not np.isfinite(ins):
                ins = max(abs(insample.loa_lo), abs(insample.loa_hi))
            note += f"; in-sample own model {ins:.2f} mm"
        if model == "loso" and np.isfinite(_get(r, "loso_n_others")):
            note += f"; model from {int(_get(r, 'loso_n_others'))} other participants"
        rows.append({**_blank(), "key": f"imu_opening_loa_{model}_mm", "group": "sensing",
                     "requirement": f"IMU incisal opening, {lab}", "value": worst,
                     "ci_lo": _get(r, "max_abs_loa_ci_lo"),
                     "ci_hi": _get(r, "max_abs_loa_ci_hi"), "median": np.nan, "unit": "mm",
                     "statistic": "largest absolute 95 % limit of agreement (repeated "
                                  "measures, bias included; MOVER 95 % CI)",
                     "n": int(r.n_subjects),
                     "derivation": f"IMU angle -> opening model ({how}) vs optical, per "
                                   "cycle (D7, D26)",
                     "evidence": EV.requirement(f"imu_opening_loa_{model}_mm"),
                     "implication": "opening from a wearable without optical tracking",
                     "note": note, "ci_method": "mover_zou2013",
                     "ci_lo_boot": _get(r, "max_abs_loa_boot_ci_lo"),
                     "ci_hi_boot": _get(r, "max_abs_loa_boot_ci_hi"), "value_insample": ins})
    return rows


# -- D27: one per-participant statistic, bandwidth from each sensor -------------------------
def _largest(cycles: pd.DataFrame, task: str, col: str, side: str | None = None,
             sign: float = 1.0) -> pd.Series:
    g = cycles[(cycles.task == task) & cycles.ok.astype(bool)]
    if side is not None:
        g = g[g.side == side]
    v = sign * g[col].astype(float)
    return v.groupby(g.pid).max()


UNIFORM = (  # key, requirement, unit, extractor description
    ("opening_mm", "Incisal opening", "mm", "largest open_close cycle opening"),
    ("protrusion_mm", "Protrusion", "mm", "largest protrusion cycle (as the primary)"),
    ("laterotrusion_mm", "Laterotrusion (larger side)", "mm",
     "largest left-right cycle, larger side (as the primary)"),
    ("opening_angle_deg", "Opening rotation", "deg", "largest open_close cycle peak angle"),
    ("peak_vel_dps", "Peak opening angular velocity", "deg/s",
     "fastest open_close cycle (as the primary)"),
)


def uniform_per_participant(cycles: pd.DataFrame) -> dict[str, pd.Series]:
    """D27: per participant the largest cycle value of the task, for every workspace and
    speed requirement (kinematics ``cycles``, ``ok`` cycles only)."""
    if cycles.empty:
        return {}
    lat = pd.concat([_largest(cycles, "left_right", "lateral_mm", "left"),
                     _largest(cycles, "left_right", "lateral_mm", "right", -1.0)],
                    axis=1).max(axis=1)
    return {"opening_mm": _largest(cycles, "open_close", "opening_mm"),
            "protrusion_mm": _largest(cycles, "protrusion_retrusion", "protrusion_mm"),
            "laterotrusion_mm": lat,
            "opening_angle_deg": _largest(cycles, "open_close", "theta_y_deg"),
            "peak_vel_dps": _largest(cycles, "open_close", "peak_open_vel_dps")}


def requirements_sensitivity(cycles: pd.DataFrame, bandwidth: pd.DataFrame,
                             primary: pd.DataFrame | None = None) -> pd.DataFrame:
    """D27 rows under the E31 rule, each with the primary design value next to it
    (``primary_value``, from the ``requirements`` table):

    * variant ``largest_cycle``: workspace and speed rows from the largest cycle of the task
      per participant (``cycles``: kinematics ``cycles``);
    * variants ``imu_only`` / ``optics_only``: the bandwidth f95 from the IMU angular-velocity
      power alone (``f95_hz``) and from the optical opening-angle velocity alone
      (``f95_opt_hz``), same tasks and noise-limited rule as the coherence-weighted primary;
    * variants ``min_band_1.5hz`` / ``min_band_3hz`` (D29): the coherence-weighted primary
      with the noise-limited rule at 1.5 and 3 Hz instead of 2 Hz.
    """
    prim = (primary.set_index("key").value if primary is not None and not primary.empty
            else pd.Series(dtype=float))
    rows = []

    def add(key: str, variant: str, req: str, per: pd.Series, unit: str, deriv: str) -> None:
        d = p95_design(per.to_numpy(float))
        rows.append({"key": key, "variant": variant, "requirement": req, "value": d["value"],
                     "value_linear": d["value_linear"], "ci_lo": d["ci_lo"],
                     "lower_k": d["lower_k"], "sample_max": d["sample_max"],
                     "median": d["median"], "n": d["n"], "unit": unit,
                     "primary_value": float(prim.get(key, np.nan)), "derivation": deriv,
                     "note": d["upper_note"]})

    uni = uniform_per_participant(cycles)
    for key, req, unit, deriv in UNIFORM:
        if key in uni:
            add(key, "largest_cycle", req, uni[key], unit, deriv)
    for col, variant, deriv in (("f95_hz", "imu_only", "f95 of the IMU angular-velocity "
                                 "power alone"),
                                ("f95_opt_hz", "optics_only", "f95 of the optical opening-"
                                 "angle velocity power alone (40 Hz, unsmoothed)")):
        per = bandwidth_per_participant(bandwidth, col)
        if len(per):
            add("bandwidth_hz", variant, "Control bandwidth", per, "Hz", deriv)
    # D29: the noise-limited rule (coherent band below 2 Hz) at its fixed neighbours.
    for x in MIN_BAND_NEIGHBOURS:
        per = bandwidth_per_participant(bandwidth, "f95_coh_hz", x)
        if len(per):
            add("bandwidth_hz", f"min_band_{x:g}hz", "Control bandwidth", per, "Hz",
                f"coherence-weighted f95 as the primary, tasks noise-limited when the coherent "
                f"band ends below {x:g} Hz (primary 2 Hz, D29)")
    return pd.DataFrame(rows, columns=SENS_COLUMNS)
