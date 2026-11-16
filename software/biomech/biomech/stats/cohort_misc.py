# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cohort statistics for timing (C2), IMU proxies (C3a), variability (S2), validation (G)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.stats.core import (bland_altman_rm, bland_altman_rm_ci, describe, holm,
                                 wilcoxon_one_sample)
from biomech.stats.waveforms import cycles_needed

# -- C2 timing ---------------------------------------------------------------------------
PRIMARY = {"task": "open_close", "direction": "opening", "detector": "tkeo"}


def ensemble_cohort(ens: pd.DataFrame, seed: int) -> pd.DataFrame:
    """X2: cohort median of the per-participant ensemble event leads (> 0: before motion)."""
    rows = []
    if ens is None or ens.empty:
        return pd.DataFrame(rows)
    for (task, direction, role), g in ens.groupby(["task", "direction", "role"]):
        v = g.lead_ms.to_numpy(float)
        rows.append({"task": task, "direction": direction, "role": role,
                     "event": g.event.iloc[0], "n_found": int(np.isfinite(v).sum()),
                     "n_participants": int(len(g)),
                     "n_before": int((v[np.isfinite(v)] > 0).sum()),
                     **describe(v, seed=seed, stat="median"),
                     **wilcoxon_one_sample(v, "greater")})
    return pd.DataFrame(rows)


def release_cohort(rel: pd.DataFrame, seed: int) -> pd.DataFrame:
    """X3: per task and detector setting, cohort summaries of the per-participant causal
    release anticipation (median over detected cycles), detection rate and false releases."""
    rows = []
    if rel is None or rel.empty:
        return pd.DataFrame(rows)
    for (task, drop, hold), g in rel.groupby(["task", "drop_db", "hold_s"]):
        a = g.anticipation_ms.to_numpy(float)
        rows.append({"task": task, "drop_db": drop, "hold_s": hold,
                     "primary": bool(g.primary.iloc[0]), "n_participants": int(len(g)),
                     "n_before": int((a[np.isfinite(a)] > 0).sum()),
                     "detected_frac_median": float(g.detected_frac.median()),
                     "chance_detected_frac_median": float(g.chance_detected_frac.median())
                     if "chance_detected_frac" in g else np.nan,
                     "detected_minus_chance_median": float((g.detected_frac
                                                            - g.chance_detected_frac).median())
                     if "chance_detected_frac" in g else np.nan,
                     "false_per_min_median": float(g.false_per_min.median()),
                     **describe(a, seed=seed, stat="median"),
                     **wilcoxon_one_sample(a, "greater"),
                     **_vs_chance(g)})
    return pd.DataFrame(rows)


def _vs_chance(g: pd.DataFrame) -> dict[str, Any]:
    """D23 (X3): observed vs chance anticipation per participant (the median anticipation at
    the detected pseudo-onsets, ``chance_anticipation_ms``): cohort medians and a two-sided
    paired signed-rank test over the participants (pairs with both values)."""
    if "chance_anticipation_ms" not in g:
        return {}
    a = g.anticipation_ms.to_numpy(float)
    c = g.chance_anticipation_ms.to_numpy(float)
    d = a - c
    w = wilcoxon_one_sample(d, "two-sided")
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return {"chance_anticipation_ms_median": float(np.nanmedian(c)),
                    "anticipation_minus_chance_ms_median": float(np.nanmedian(d)),
                    "n_vs_chance": int(np.isfinite(d).sum()),
                    "W_vs_chance": w["W"], "p_vs_chance": w["p"], "r_vs_chance": w["r"]}


TIMING_METRICS = ["lead_ms", "lead_omega_ms", "lead_acc_ms", "lead_opt_ms", "causal_delay_ms",
                  "anticipation_ms"]
TIMING_GROUPS = {"opening": ("digastric_l", "digastric_r"),
                 "closing": ("masseter_l", "masseter_r", "temporalis_l", "temporalis_r")}
CONFIRMATORY_TASKS = ("open_close", "chewing")
# Both onset rules are confirmatory (D6: the frozen "first" rule is never dropped; "drive"
# is its D15 companion), so a C2 claim needs the pre-specified rule *and* the driving burst.
CONFIRMATORY_RULES = ("first", "drive")
AGGREGATIONS = ("pooled", "muscle_mean")


N_PERM_NULL = 10_000          # sign-flip permutations of the observed - null test (D23)


def sign_flip_p(d: np.ndarray, n_perm: int = N_PERM_NULL, seed: int = 0) -> float:
    """One-sided (> 0) sign-flip permutation p of the mean of the paired differences ``d``:
    p = (1 + #{permuted mean >= observed}) / (1 + n_perm), signs flipped at random (seeded).
    NaN below 5 pairs (as the signed-rank test)."""
    v = np.asarray(d, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan
    rng = np.random.default_rng(seed)
    obs = float(v.mean())
    perm = (rng.choice((-1.0, 1.0), size=(n_perm, v.size)) * np.abs(v)).mean(axis=1)
    tol = 1e-9 * max(1.0, abs(obs))
    return float((1 + np.sum(perm >= obs - tol)) / (1 + n_perm))


def null_pairs(per: pd.DataFrame, lead_null: pd.DataFrame | None, task: str, direction: str,
               det: str, rule: str, aggregation: str = "pooled", min_cycles: int = 3
               ) -> pd.DataFrame:
    """Per participant the observed lead (``per``, :func:`per_participant_timing`) next to its
    chance null (``timing.lead_null``, D23): ``lead_null_ms`` (the pooled null median, or the
    mean of the per-muscle null medians under ``muscle_mean``) and ``lead_minus_null_ms``.
    A null with fewer than ``min_cycles`` usable pseudo channel-cycles is NaN."""
    cols = ["pid", "lead_ms", "lead_null_ms", "lead_minus_null_ms", "usable_null_frac",
            "n_null", "n_null_usable"]
    if lead_null is None or lead_null.empty or "lead_ms" not in per:
        return pd.DataFrame(columns=cols)
    g = lead_null[(lead_null.task == task) & (lead_null.direction == direction)
                  & (lead_null.detector == det) & (lead_null.rule == rule)]
    src = "lead_null_muscle_mean_ms" if aggregation == "muscle_mean" else "lead_null_ms"
    nul = g.set_index("pid")
    out = pd.DataFrame({"pid": per.index, "lead_ms": per["lead_ms"].to_numpy(float)})
    out["lead_null_ms"] = out.pid.map(nul[src]) if src in nul else np.nan
    for c in ("usable_null_frac", "n_null", "n_null_usable"):
        out[c] = out.pid.map(nul[c]) if c in nul else np.nan
    out.loc[~(out.n_null_usable >= min_cycles), "lead_null_ms"] = np.nan
    out["lead_minus_null_ms"] = out.lead_ms - out.lead_null_ms
    return out[cols]


def null_test(pairs: pd.DataFrame, seed: int) -> dict[str, Any]:
    """D23 calibrated test, reported next to the frozen test against 0: cohort medians of
    the null and of observed - null, the one-sided signed-rank test of observed - null > 0
    (``W_null``, ``p_null``, ``r_null``) and the sign-flip permutation test
    (``p_null_perm``)."""
    d = pairs.lead_minus_null_ms.to_numpy(float) if len(pairs) else np.empty(0)
    nul = pairs.lead_null_ms.to_numpy(float) if len(pairs) else np.empty(0)
    w = wilcoxon_one_sample(d, "greater")
    fin = np.isfinite(d)
    return {"lead_null_ms": float(np.median(nul[np.isfinite(nul)]))
            if np.isfinite(nul).any() else np.nan,
            "lead_minus_null_ms": float(np.median(d[fin])) if fin.any() else np.nan,
            "n_null_pairs": int(fin.sum()),
            "W_null": w["W"], "p_null": w["p"], "r_null": w["r"],
            "p_null_perm": sign_flip_p(d, seed=seed)}


def per_participant_timing(summary: pd.DataFrame, onsets: pd.DataFrame, task: str,
                           direction: str, det: str, rule: str, aggregation: str = "pooled",
                           min_cycles: int = 3) -> tuple[pd.DataFrame, int]:
    """Per-participant timing values of one (task, direction, detector, rule) and the number
    of participants excluded for fewer than ``min_cycles`` usable channel-cycles (D15).

    * ``pooled`` (primary, D15): median over the participant's usable channel-cycles of the
      direction's muscle group, pooled over channels (each cycle weighs the same);
    * ``muscle_mean`` (pre-D15, sensitivity ``c2_muscle_mean``): unweighted mean of the
      per-muscle medians (``summary``).
    A participant needs ``min_cycles`` usable channel-cycles in the group (counted over all
    its muscles); participants with onset rows but fewer usable ones are excluded and
    counted.
    """
    group = TIMING_GROUPS[direction]
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown stats_timing.aggregation {aggregation!r}; "
                         f"known: {AGGREGATIONS}")
    has_on = onsets is not None and not onsets.empty and "usable" in onsets
    cand: set = set()
    n_use = pd.Series(dtype=float)
    if has_on:
        o = onsets
        if "rule" in o:
            o = o[o.rule == rule]
        o = o[(o.task == task) & (o.direction == direction) & (o.detector == det)
              & o.muscle.isin(group)]
        cand = set(o.pid)
        u = o[o.usable.astype(bool)]
        n_use = u.groupby("pid").size()
    g = summary
    if "rule" in g:
        g = g[g.rule == rule]
    g = g[(g.task == task) & (g.direction == direction) & (g.detector == det)
          & g.muscle.isin(group)]
    cols = [c for c in TIMING_METRICS if c in g.columns]
    if aggregation == "pooled" and has_on:
        per = u.groupby("pid")[[c for c in TIMING_METRICS if c in u.columns]].median()
    else:
        per = g.groupby("pid")[cols].mean()
        if not has_on:
            cand = set(g.pid)
            n_use = g.groupby("pid")["n"].sum() if "n" in g else g.groupby("pid").size()
    keep = [pid for pid in per.index if n_use.get(pid, 0) >= min_cycles]
    n_excl = int(sum(1 for pid in cand | set(per.index) if n_use.get(pid, 0) < min_cycles))
    return per.loc[keep], n_excl


def _timing_family(rows: list[dict[str, Any]],
                   rules: tuple[str, ...] = CONFIRMATORY_RULES) -> None:
    """Holm families in place (D15). Confirmatory: opening lead (``lead_ms``), tasks
    open_close and chewing, primary detector (tkeo), each onset rule of ``rules`` (default
    both: the frozen "first" rule and the D6 driving burst): Holm across those four tests.
    Exploratory: every other tested row, Holm within its metric (all detector x rule x task
    x direction combinations of that metric). Untested rows: ``descriptive``."""
    def confirm(r: dict[str, Any]) -> bool:
        return (r["metric"] == "lead_ms" and r["direction"] == "opening"
                and r["detector"] == "tkeo" and r["rule"] in rules
                and r["task"] in CONFIRMATORY_TASKS)

    for r in rows:
        tested = "p" in r
        r["family"] = "confirmatory" if confirm(r) and tested else (
            "exploratory" if tested else "descriptive")
        r["p_holm"] = np.nan
    fams: dict[tuple[str, str], list[int]] = {}
    for k, r in enumerate(rows):
        if r["family"] != "descriptive":
            key = (r["family"], "" if r["family"] == "confirmatory" else r["metric"])
            fams.setdefault(key, []).append(k)
    for idx in fams.values():
        for k, pa in zip(idx, holm([rows[k]["p"] for k in idx])):
            rows[k]["p_holm"] = pa
        # D23: the calibrated tests (observed - null) get Holm within the same family, i.e.
        # the confirmatory one across the same four tests as C2.
        for col in ("p_null", "p_null_perm"):
            have = [k for k in idx if col in rows[k]]
            if have:
                for k, pa in zip(have, holm([rows[k][col] for k in have])):
                    rows[k][f"{col}_holm"] = pa


def timing_tables(summary: pd.DataFrame, onsets: pd.DataFrame, seed: int,
                  ensemble: pd.DataFrame | None = None,
                  release: pd.DataFrame | None = None, *,
                  params: dict[str, Any] | None = None,
                  onset_rule: str | None = None,
                  lead_null: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """C2 cohort tables. ``params`` = the ``stages.stats_timing`` block (``aggregation``:
    pooled | muscle_mean, ``min_cycles``, ``confirmatory_rules``; D15); ``onset_rule`` =
    ``stages.timing.onset_rule`` (the rule figures show first; not used for the families).
    ``lead_null`` = ``timing.lead_null`` (D23): the ``lead_ms`` rows also get the calibrated
    test of observed - null (:func:`null_test`, Holm within the same families), and
    ``timing_null`` lists the per-participant pairs. The frozen test against 0 (``p``,
    ``p_holm``) stays the primary and is unchanged.
    """
    params = dict(params or {})
    aggregation = str(params.get("aggregation", "pooled"))
    min_cycles = int(params.get("min_cycles", 3))
    onset_rule = str(onset_rule or params.get("onset_rule", "drive"))
    out: dict[str, pd.DataFrame] = {}
    if ensemble is not None and not ensemble.empty:
        out["ensemble_cohort"] = ensemble_cohort(ensemble, seed)
    if release is not None and not release.empty:
        out["release_cohort"] = release_cohort(release, seed)
    if summary.empty:
        return out
    rows, spec, null_rows = [], [], []
    if "rule" not in summary:
        summary = summary.assign(rule="first")
    for (task, direction, det, rule), _ in summary.groupby(
            ["task", "direction", "detector", "rule"]):
        if direction not in TIMING_GROUPS:
            continue
        per, n_excl = per_participant_timing(summary, onsets, task, direction, det, rule,
                                             aggregation, min_cycles)
        has_null = lead_null is not None and not lead_null.empty
        pairs = null_pairs(per, lead_null, task, direction, det, rule, aggregation,
                           min_cycles) if has_null else None
        for col in per.columns:
            d = describe(per[col].to_numpy(), seed=seed, stat="median")
            row = {"task": task, "direction": direction, "detector": det, "rule": rule,
                   "metric": col, **d, "n_excluded_min_cycles": n_excl}
            if col in ("lead_ms", "anticipation_ms"):
                row.update(wilcoxon_one_sample(per[col].to_numpy(), "greater"))
            if col == "lead_ms" and pairs is not None:
                row.update(null_test(pairs, seed))
            rows.append(row)
        if pairs is not None and len(pairs):
            null_rows += pairs.assign(task=task, direction=direction, detector=det,
                                      rule=rule).to_dict("records")
        for src in ("lead_ms", "lead_omega_ms", "lead_acc_ms", "lead_opt_ms"):
            if src not in per:
                continue
            for pid, v in per[src].items():
                spec.append({"pid": pid, "outcome": f"{task}|{direction}|lead",
                             "variant": f"detector={det}|mech={src}|onset={rule}",
                             "value": v})
    _timing_family(rows, tuple(params.get("confirmatory_rules", CONFIRMATORY_RULES)))
    out["timing_cohort"] = pd.DataFrame(rows)
    out["timing_spec"] = pd.DataFrame(spec)
    if lead_null is not None and not lead_null.empty:
        keys = ["pid", "task", "direction", "detector", "rule"]
        tn = pd.DataFrame(null_rows)
        out["timing_null"] = tn[keys + [c for c in tn.columns if c not in keys]] \
            if len(tn) else tn
    if not onsets.empty:
        keys = ["pid", "task", "direction", "detector"] + (["rule"] if "rule" in onsets else [])
        u = (onsets.groupby(keys)
             .agg(usable_frac=("usable", "mean"), active_at_start=("active_at_start", "mean"),
                  n=("usable", "size")).reset_index())
        out["timing_usability"] = u
    return out


# -- C3a IMU -----------------------------------------------------------------------------
LOA_Z = 1.96


def _ba_anova(d: np.ndarray, g: np.ndarray) -> dict[str, Any] | None:
    """One-way ANOVA pieces of the Bland & Altman (2007) repeated-measures limits."""
    labels, inv, m = np.unique(g, return_inverse=True, return_counts=True)
    k, n = len(labels), d.size
    if k < 2 or n - k < 1:
        return None
    means = np.bincount(inv, weights=d) / m
    grand = float(d.mean())
    msb = float(np.sum(m * (means - grand) ** 2) / (k - 1))
    msw = float(np.sum((d - means[inv]) ** 2) / (n - k))
    div = float((n ** 2 - np.sum(m ** 2)) / ((k - 1) * n))
    return {"mu": grand, "msb": msb, "msw": msw, "div": div, "k": k, "n": n, "m": m}


def max_abs_loa_mover(d: np.ndarray, g: np.ndarray, alpha: float = 0.05
                      ) -> tuple[float, float, float]:
    """(max |LoA|, CI low, CI high) of the largest absolute repeated-measures limit of
    agreement, max(|mu - z s|, |mu + z s|) = |mu| + z s, by the MOVER (method of variance
    estimates recovery; Zou 2013, Stat Methods Med Res 22:630, for limits of agreement with
    several observations per participant; Donner & Zou 2012 for the general method).

    Why not the percentile cluster bootstrap: with 13 participant clusters it under-covers
    the limits of agreement (Part H audit: 0.84 for the upper limit; for max |LoA| 0.85
    with 14 % upper-tail misses against MOVER's 0.97 with 2 %, nominal 2.5 %; test
    ``test_max_abs_loa_interval_coverage_13_clusters``), the bootstrap of a
    variance from few clusters being too narrow and biased low. The MOVER interval keeps the
    exact chi-square distributions of the two variance components and needs no resampling.

    Pieces (the B&A 2007 one-way ANOVA, as ``stats.core.bland_altman_rm``):
    s^2 = MSB/D + (1 - 1/D) MSW (= s_b^2 + s_w^2 whenever s_b^2 >= 0), MSB with k - 1 and
    MSW with N - k degrees of freedom; the MOVER limits of a sum of scaled chi-square
    variances (each term's exact limit l_i, u_i):
    L = s^2 - sqrt(sum (t_i - l_i)^2), U = s^2 + sqrt(sum (u_i - t_i)^2), t_i the term,
    [l_i, u_i] = t_i df_i / [chi2_{1-a/2, df_i}, chi2_{a/2, df_i}];
    the bias mu (cycle-weighted grand mean) with a t interval on k - 1 df and the variance
    s_b^2 sum(m_i^2)/N^2 + s_w^2/N. The interval of |mu| is [0 or min(|l|, |u|), max(|l|, |u|)],
    and for theta = |mu| + z s:
    theta_L = theta - sqrt((|mu| - l_|mu|)^2 + z^2 (s - l_s)^2),
    theta_U = theta + sqrt((u_|mu| - |mu|)^2 + z^2 (u_s - s)^2).
    When the between-participant variance estimate is negative the point estimate falls back
    to MSW (as ``bland_altman_rm``) and the interval to the MSW term alone.
    """
    d = np.asarray(d, float)
    g = np.asarray(g)
    ok = np.isfinite(d)
    a = _ba_anova(d[ok], g[ok])
    if a is None:
        return np.nan, np.nan, np.nan
    from scipy import stats as sps

    k, n, m, div = a["k"], a["n"], a["m"], a["div"]
    var_b = max((a["msb"] - a["msw"]) / div, 0.0)
    # (estimate of the term, its degrees of freedom); a term's exact limits are
    # est * df / chi2 quantiles, ordered (a negative coefficient swaps them, D < 1).
    terms = ([(a["msb"] / div, k - 1), (a["msw"] * (1 - 1 / div), n - k)] if var_b > 0
             else [(a["msw"], n - k)])
    s2 = sum(c for c, _ in terms)
    lims = [sorted((c * df / sps.chi2.ppf(1 - alpha / 2, df), c * df / sps.chi2.ppf(alpha / 2, df)))
            for c, df in terms]
    lo2 = s2 - np.sqrt(sum((c - lo) ** 2 for (c, _), (lo, _) in zip(terms, lims)))
    hi2 = s2 + np.sqrt(sum((hi - c) ** 2 for (c, _), (_, hi) in zip(terms, lims)))
    s, ls, us = np.sqrt(s2), np.sqrt(max(lo2, 0.0)), np.sqrt(hi2)
    mu = a["mu"]
    se_mu = np.sqrt(var_b * np.sum(m ** 2) / n ** 2 + a["msw"] / n)
    t = sps.t.ppf(1 - alpha / 2, k - 1)
    lmu, umu = mu - t * se_mu, mu + t * se_mu
    amu = abs(mu)
    u_a = max(abs(lmu), abs(umu))
    l_a = 0.0 if lmu <= 0.0 <= umu else min(abs(lmu), abs(umu))
    theta = amu + LOA_Z * s
    return (float(theta),
            float(theta - np.sqrt((amu - l_a) ** 2 + LOA_Z ** 2 * (s - ls) ** 2)),
            float(theta + np.sqrt((u_a - amu) ** 2 + LOA_Z ** 2 * (us - s) ** 2)))


def max_abs_loa_ci(est: np.ndarray, ref: np.ndarray, groups: np.ndarray, seed: int,
                   n_boot: int = 2000) -> dict[str, float]:
    """D26: the largest absolute repeated-measures limit of agreement ``max_abs_loa`` =
    max(|loa_lo|, |loa_hi|) of ``stats.core.bland_altman_rm`` with

    * ``max_abs_loa_ci_lo``/``_hi``: the MOVER 95 % interval (:func:`max_abs_loa_mover`),
      the one reported;
    * ``max_abs_loa_boot_ci_lo``/``_hi``: the participant-cluster percentile bootstrap of
      max |LoA| itself (the D26 wording; same draws as ``bland_altman_rm_ci`` with the same
      seed), stored for comparison -- it under-covers with 13 clusters (see
      :func:`max_abs_loa_mover`).
    """
    e, r, g = np.asarray(est, float), np.asarray(ref, float), np.asarray(groups)
    ok = np.isfinite(e) & np.isfinite(r)
    e, r, g = e[ok], r[ok], g[ok]
    out = {"max_abs_loa": np.nan, "max_abs_loa_ci_lo": np.nan, "max_abs_loa_ci_hi": np.nan,
           "max_abs_loa_boot_ci_lo": np.nan, "max_abs_loa_boot_ci_hi": np.nan}
    ba = bland_altman_rm(e, r, g)
    if not np.isfinite(ba["loa_lo"]):
        return out
    out["max_abs_loa"] = float(max(abs(ba["loa_lo"]), abs(ba["loa_hi"])))
    _, out["max_abs_loa_ci_lo"], out["max_abs_loa_ci_hi"] = max_abs_loa_mover(e - r, g)
    labels = np.unique(g)
    if len(labels) < 3:
        return out
    idx = {lab: np.flatnonzero(g == lab) for lab in labels}
    rng = np.random.default_rng(seed)
    draws = np.full(n_boot, np.nan)
    for b in range(n_boot):
        pick = rng.choice(labels, len(labels), replace=True)
        sel = np.concatenate([idx[lab] for lab in pick])
        gg = np.concatenate([np.full(len(idx[lab]), j) for j, lab in enumerate(pick)])
        q = bland_altman_rm(e[sel], r[sel], gg)
        draws[b] = max(abs(q["loa_lo"]), abs(q["loa_hi"]))
    v = draws[np.isfinite(draws)]
    if v.size:
        out["max_abs_loa_boot_ci_lo"], out["max_abs_loa_boot_ci_hi"] = (
            float(x) for x in np.percentile(v, [2.5, 97.5]))
    return out


# Opening models of the C3a Bland-Altman, in report order (D26): the generic model is the
# pre-specified C3a outcome; the own model is out of sample (leave-one-cycle-out, imu stage
# column opening_subject_*); its in-sample fit (r9) is kept as a sensitivity.
OPENING_MODELS = ("loso", "subject", "subject_insample")


def imu_tables(agree: pd.DataFrame, calib: pd.DataFrame, cycles: pd.DataFrame,
               models: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """C3a cohort tables. ``agree``/``calib``/``cycles``: ``imu`` tables (participants with
    IMU data); ``models``: ``kinematics.opening_model`` of **every participant with optical
    data** (E32): the generic (leave-one-subject-out) opening model of a participant is the
    mean of the coefficients of all other participants with optics, including those without
    IMU data (``loso_n_others`` in ``imu_opening_ba``)."""
    out: dict[str, pd.DataFrame] = {}
    test = agree[~agree.is_calibration]
    rows = []
    for (task, est), g in test.melt(
            id_vars=["pid", "task"], value_vars=[c for c in test if c.startswith("rmse_")],
            var_name="estimator", value_name="rmse").groupby(["task", "estimator"]):
        rows.append({"task": task, "estimator": est.replace("rmse_", "").replace("_deg", ""),
                     **describe(g.rmse.to_numpy(), seed=seed, stat="median")})
    for col in ("rms_deg", "drift_quat_deg_per_min"):
        if col not in calib:
            continue
        rows.append({"task": "calibration", "estimator": col,
                     **describe(calib[col].to_numpy(), seed=seed, stat="median")})
    rows.append({"task": "all", "estimator": "lag_ms",
                 **describe(test.lag_ms.to_numpy(), seed=seed, stat="median")})
    out["imu_agreement_cohort"] = pd.DataFrame(rows)

    # Opening (mm) from the IMU angle: generic (LOSO), own (leave-one-cycle-out) and own
    # in-sample models.
    ba_rows = []
    if not cycles.empty and not models.empty:
        m = models.drop_duplicates("pid").set_index("pid")
        c = cycles.copy()
        others = c.pid.map(lambda p: int((m.index != p).sum()))
        for est in ("quat", "comp"):
            th = c[f"theta_peak_{est}_deg"]
            a1_loso = c.pid.map(lambda p: m.drop(index=p, errors="ignore").a1_mm_per_deg.mean())
            a2_loso = c.pid.map(lambda p: m.drop(index=p, errors="ignore").a2_mm_per_deg2.mean())
            c[f"opening_loso_{est}_mm"] = a1_loso * th + a2_loso * th ** 2
            for task in ("open_close", "chewing"):
                g = c[c.task == task]
                for model in OPENING_MODELS:
                    col = f"opening_{model}_{est}_mm"
                    if col not in g:
                        continue
                    e, r, pid = g[col].to_numpy(), g.opening_opt_mm.to_numpy(), g.pid.to_numpy()
                    # Cycles repeat within participants: repeated-measures limits (D7).
                    ba = bland_altman_rm(e, r, pid)
                    ba |= bland_altman_rm_ci(e, r, pid, seed)
                    ba |= max_abs_loa_ci(e, r, pid, seed)
                    per = (g[col] - g.opening_opt_mm).groupby(g.pid).mean()
                    ba_rows.append({"task": task, "estimator": est, "model": model, **ba,
                                    "subject_bias_sd": float(per.std(ddof=1)),
                                    "subject_abs_bias_median": float(per.abs().median()),
                                    "loso_n_others": float(others[g.index].median())
                                    if model == "loso" and len(g) else np.nan})
        out["imu_opening_cycles"] = c
    out["imu_opening_ba"] = pd.DataFrame(ba_rows)
    return out


# -- S2 variability ----------------------------------------------------------------------
def spec_curve(spec: pd.DataFrame, primary: dict[str, str]) -> pd.DataFrame:
    """Spread of cohort means across processing variants vs between-participant SD."""
    rows = []
    for outcome, g in spec.groupby("outcome"):
        means = g.groupby("variant")["value"].mean()
        prim = primary.get(outcome.split("|")[0], None)
        prim_var = next((v for v in means.index if prim and v.startswith(prim)), means.index[0])
        sd_between = float(g[g.variant == prim_var]["value"].std(ddof=1))
        rows.append({"outcome": outcome, "n_variants": int(len(means)),
                     "primary_variant": prim_var, "primary_mean": float(means[prim_var]),
                     "variant_min": float(means.min()), "variant_max": float(means.max()),
                     "spread": float(means.max() - means.min()), "sd_between": sd_between,
                     "spread_over_sd": float((means.max() - means.min()) / sd_between)
                     if sd_between > 0 else np.nan})
    return pd.DataFrame(rows)


def cycles_needed_table(cycles: pd.DataFrame, mdc: dict[tuple, float],
                        seed: int = 0) -> pd.DataFrame:
    """Cycles needed per participant until the CI half-width of the mean is <= MDC/2.

    ``mdc`` is keyed (task, metric) or (task, metric, side); a side restricts the cycles
    to that side (laterotrusion), and the side's own MDC is the target.
    """
    rows = []
    for key, target in mdc.items():
        task, metric = key[0], key[1]
        side = key[2] if len(key) > 2 else ""
        g = cycles[(cycles.task == task) & cycles.ok]
        if side and "side" in g:
            g = g[g.side == side]
        for pid, gp in g.groupby("pid"):
            v = gp[metric].abs().to_numpy() if task == "left_right" else gp[metric].to_numpy()
            if np.isfinite(v).sum() < 4 or not np.isfinite(target):
                continue
            _, _, n = cycles_needed(v, target / 2, seed=seed)
            rows.append({"task": task, "metric": metric, "side": side, "pid": pid,
                         "target_halfwidth": target / 2, "n_available": int(np.isfinite(v).sum()),
                         "n_needed": n if n is not None else np.nan})
    return pd.DataFrame(rows)


def protocol_heterogeneity(phases: pd.DataFrame, cycles: pd.DataFrame,
                           emg_refs: pd.DataFrame) -> pd.DataFrame:
    """What actually varied across participants in this data set (§5.3.6)."""
    rows = []
    ph = phases[phases.kind.isin(["experiment", "mvc"])]
    for name, g in ph.groupby("phase"):
        rows.append({"item": f"duration_s|{name}", "min": g.phase_s.min(),
                     "median": g.phase_s.median(), "max": g.phase_s.max(), "n": len(g)})
    ok = cycles[cycles.ok]
    for task, g in ok.groupby("task"):
        n = g.groupby("pid").size()
        rows.append({"item": f"cycles|{task}", "min": n.min(), "median": n.median(),
                     "max": n.max(), "n": len(n)})
        d = g.groupby("pid").duration_s.median()
        rows.append({"item": f"cycle_duration_s|{task}", "min": d.min(), "median": d.median(),
                     "max": d.max(), "n": len(d)})
    if not emg_refs.empty:
        mf = emg_refs.groupby("pid").mask_frac.max()
        rows.append({"item": "mvc_reference_masked_frac_max", "min": mf.min(),
                     "median": mf.median(), "max": mf.max(), "n": len(mf)})
    return pd.DataFrame(rows)


# -- G validation --------------------------------------------------------------------------
def validation_status(participants: list[Any]) -> pd.DataFrame:
    ref = [p.pid for p in participants if "reference_subset" in p.flags]
    if not ref:
        return pd.DataFrame([{"status": "no_reference_data",
                              "note": "no participant flagged reference_subset; backup G' "
                                      "(Monte Carlo, noise floor, cross-modal) applies",
                              "n": 0}])
    return pd.DataFrame([{"status": "reference_subset_present", "n": len(ref),
                          "pids": ",".join(ref)}])
