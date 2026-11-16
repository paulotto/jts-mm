# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Statistics used by the cohort stages (analysis_plan §5, an07).

Unit of analysis is the participant: callers pass one value per participant
(or per cycle with a participant id for mixed models / reliability).
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats


def describe(x: np.ndarray, *, n_boot: int = 4000, seed: int = 0, stat: str = "mean"
             ) -> dict[str, float]:
    """mean ± SD, median [IQR], 95 % BCa bootstrap CI of ``stat`` (mean or median).

    ``ci_percentile`` is True where the BCa interval was undefined and the percentile
    interval is reported instead (E25). (A boolean, not a text column: text columns act as
    row keys in the sensitivity comparison.)
    """
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    out = {"n": int(v.size), "mean": np.nan, "sd": np.nan, "median": np.nan, "q1": np.nan,
           "q3": np.nan, "min": np.nan, "max": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "ci_percentile": False}
    if v.size == 0:
        return out
    out.update(mean=float(v.mean()), sd=float(v.std(ddof=1)) if v.size > 1 else np.nan,
               median=float(np.median(v)), q1=float(np.percentile(v, 25)),
               q3=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))
    if v.size >= 4 and np.ptp(v) > 0:
        fn = np.mean if stat == "mean" else np.median
        lo, hi, method = _boot_ci(v, fn, n_boot, seed)
        out.update(ci_lo=lo, ci_hi=hi, ci_percentile=method == "percentile")
    return out


def _boot_ci(v: np.ndarray, fn: Callable[..., Any], n_boot: int, seed: int
             ) -> tuple[float, float, str]:
    """95 % BCa bootstrap CI, falling back to the percentile CI (E25).

    scipy returns NaN bounds (with a suppressed ``DegenerateDataWarning``) when the BCa
    acceleration/bias cannot be computed, e.g. a median whose bootstrap distribution is
    mostly one value; that NaN used to be reported as the CI. Returns (lo, hi, method).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("BCa", "percentile"):
            try:
                r = stats.bootstrap((v,), fn, n_resamples=n_boot, method=method,
                                    random_state=np.random.default_rng(seed))
            except Exception:
                continue
            lo, hi = float(r.confidence_interval.low), float(r.confidence_interval.high)
            if np.isfinite(lo) and np.isfinite(hi):
                return lo, hi, method
    return np.nan, np.nan, ""


def hedges_g(a: np.ndarray, b: np.ndarray, *, n_boot: int = 4000, seed: int = 0
             ) -> dict[str, float]:
    """Hedges' g (a - b, independent groups) with percentile bootstrap CI."""
    a = np.asarray(a, float)[np.isfinite(a)]
    b = np.asarray(b, float)[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"g": np.nan, "g_lo": np.nan, "g_hi": np.nan}

    def g(x: np.ndarray, y: np.ndarray) -> float:
        sp = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                     / (x.size + y.size - 2))
        j = 1 - 3 / (4 * (x.size + y.size) - 9)
        return float(j * (x.mean() - y.mean()) / sp) if sp > 0 else np.nan

    # The resample indices are drawn in the former order (a, then b, per resample) and g is
    # evaluated for all resamples at once: row-wise mean/var reduce each row exactly as the
    # 1-D calls did, so the draws are identical to the former per-resample loop.
    rng = np.random.default_rng(seed)
    ia, ib = np.empty((n_boot, a.size), np.int64), np.empty((n_boot, b.size), np.int64)
    for k in range(n_boot):
        ia[k] = rng.integers(0, a.size, a.size)      # == rng.choice(a, a.size) indices
        ib[k] = rng.integers(0, b.size, b.size)
    x, y = a[ia], b[ib]
    sp = np.sqrt(((a.size - 1) * x.var(ddof=1, axis=1) + (b.size - 1) * y.var(ddof=1, axis=1))
                 / (a.size + b.size - 2))
    j = 1 - 3 / (4 * (a.size + b.size) - 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = np.where(sp > 0, j * (x.mean(axis=1) - y.mean(axis=1)) / sp, np.nan)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"g": g(a, b), "g_lo": float(lo), "g_hi": float(hi)}


def holm(p: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def icc_sem_mdc(df: pd.DataFrame, value: str, subject: str = "pid", rep: str = "rep",
                k: int | None = None) -> dict[str, float]:
    """ICC(3,1) across repetitions (cycles) with SEM and MDC95 (balanced on the first k reps)."""
    import pingouin as pg

    d = df[[subject, rep, value]].dropna()
    counts = d.groupby(subject)[rep].nunique()
    k = int(k or counts.min()) if len(counts) else 0
    keep = counts[counts >= k].index
    d = d[d[subject].isin(keep)]
    d = d[d.groupby(subject)[rep].rank(method="first") <= k]
    out = {"icc31": np.nan, "icc_lo": np.nan, "icc_hi": np.nan, "sem": np.nan, "mdc95": np.nan,
           "n_subjects": int(len(keep)), "k_reps": k}
    if len(keep) < 3 or k < 2:
        return out
    d = d.assign(_r=d.groupby(subject).cumcount())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = pg.intraclass_corr(d, targets=subject, raters="_r", ratings=value)
    t = t.set_index("Type")
    # pingouin < 0.7 labels ICC(3,1) "ICC3" with column "CI95%"; >= 0.7 "ICC(C,1)" / "CI95".
    row = t.loc["ICC(C,1)"] if "ICC(C,1)" in t.index else t.loc["ICC3"]
    ci = row["CI95"] if "CI95" in row.index else row["CI95%"]
    icc = float(row["ICC"])
    sd = float(d[value].std(ddof=1))
    sem = sd * np.sqrt(max(1 - icc, 0.0))
    out.update(icc31=icc, icc_lo=float(ci[0]), icc_hi=float(ci[1]),
               sem=float(sem), mdc95=float(1.96 * np.sqrt(2) * sem))
    return out


def bland_altman(est: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    d = np.asarray(est, float) - np.asarray(ref, float)
    d = d[np.isfinite(d)]
    if d.size < 3:
        return {"n": int(d.size), "bias": np.nan, "loa_lo": np.nan, "loa_hi": np.nan,
                "rmse": np.nan}
    sd = d.std(ddof=1)
    return {"n": int(d.size), "bias": float(d.mean()), "loa_lo": float(d.mean() - 1.96 * sd),
            "loa_hi": float(d.mean() + 1.96 * sd), "rmse": float(np.sqrt(np.mean(d ** 2)))}


def bland_altman_rm(est: np.ndarray, ref: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Limits of agreement for repeated measures per participant (Bland & Altman 2007,
    "true value varies"): the SD of a single difference combines the within- and the
    between-participant variance from a one-way ANOVA of the differences. Pooling all
    cycles as if independent (plain :func:`bland_altman`) makes the limits too narrow.
    """
    d = np.asarray(est, float) - np.asarray(ref, float)
    g = np.asarray(groups)
    ok = np.isfinite(d)
    d, g = d[ok], g[ok]
    out = {"n": int(d.size), "n_subjects": int(len(np.unique(g))), "bias": np.nan,
           "loa_lo": np.nan, "loa_hi": np.nan, "rmse": np.nan, "sd_within": np.nan,
           "sd_between": np.nan}
    labels, inv, m = np.unique(g, return_inverse=True, return_counts=True)
    k, n = len(labels), d.size
    if k < 2 or n - k < 1:
        return out
    means = np.bincount(inv, weights=d) / m
    grand = d.mean()
    msb = np.sum(m * (means - grand) ** 2) / (k - 1)
    msw = np.sum((d - means[inv]) ** 2) / (n - k)
    divisor = (n ** 2 - np.sum(m ** 2)) / ((k - 1) * n)
    var_b = max((msb - msw) / divisor, 0.0)
    sd = float(np.sqrt(var_b + msw))
    out.update(bias=float(grand), loa_lo=float(grand - 1.96 * sd),
               loa_hi=float(grand + 1.96 * sd), rmse=float(np.sqrt(np.mean(d ** 2))),
               sd_within=float(np.sqrt(msw)), sd_between=float(np.sqrt(var_b)))
    return out


def bland_altman_rm_ci(est: np.ndarray, ref: np.ndarray, groups: np.ndarray, seed: int,
                       n_boot: int = 2000) -> dict[str, float]:
    """95 % CIs of the repeated-measures bias and limits of agreement and the proportional
    bias (OLS slope of the difference on the mean, cycle level) by a participant-cluster
    bootstrap: participants are resampled with replacement (all their cycles kept), so the
    CIs carry the between-participant uncertainty that dominates with few participants."""
    est, ref, g = np.asarray(est, float), np.asarray(ref, float), np.asarray(groups)
    ok = np.isfinite(est) & np.isfinite(ref)
    est, ref, g = est[ok], ref[ok], g[ok]
    keys = ("bias", "loa_lo", "loa_hi", "prop_slope")
    out = {f"{k}_ci_{s}": np.nan for k in keys for s in ("lo", "hi")} | {"prop_slope": np.nan}
    labels = np.unique(g)
    if len(labels) < 3:
        return out

    def slope(e: np.ndarray, r: np.ndarray) -> float:
        m, d = (e + r) / 2, e - r
        return float(np.polyfit(m, d, 1)[0]) if np.ptp(m) > 0 else np.nan

    out["prop_slope"] = slope(est, ref)
    idx = {lab: np.flatnonzero(g == lab) for lab in labels}
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, 4), np.nan)
    for b in range(n_boot):
        pick = rng.choice(labels, len(labels), replace=True)
        sel = np.concatenate([idx[lab] for lab in pick])
        # Resampled participants get distinct group labels (a participant drawn twice
        # counts as two participants in the ANOVA).
        gg = np.concatenate([np.full(len(idx[lab]), k) for k, lab in enumerate(pick)])
        r = bland_altman_rm(est[sel], ref[sel], gg)
        draws[b] = (r["bias"], r["loa_lo"], r["loa_hi"], slope(est[sel], ref[sel]))
    for j, k in enumerate(keys):
        v = draws[:, j][np.isfinite(draws[:, j])]
        if v.size:
            out[f"{k}_ci_lo"], out[f"{k}_ci_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    return out


MIXED_METHODS = ("lbfgs", "bfgs", "powell", "nm")


def mixed_model(df: pd.DataFrame, formula: str, group: str = "pid") -> pd.DataFrame:
    """Linear mixed model (random intercept per participant, REML); fixed-effect table.

    statsmodels can report ``converged`` for a diverged fit (log-likelihood +inf, group
    variance on the boundary, a coefficient stuck at its start with an infinite SE), and
    only says so in a warning. Each optimizer in ``MIXED_METHODS`` is tried in turn; a fit
    counts only with a finite log-likelihood and finite standard errors. The chosen
    optimizer and the group variance are returned as columns.
    """
    import statsmodels.formula.api as smf

    tried = []
    for method in MIXED_METHODS:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = smf.mixedlm(formula, df, groups=df[group]).fit(reml=True, method=method)
            except (np.linalg.LinAlgError, ValueError) as exc:
                tried.append(f"{method}: {exc}")
                continue
        se = np.asarray(res.bse.drop(index="Group Var", errors="ignore"), float)
        if np.isfinite(res.llf) and np.isfinite(se).all() and se.max() < 1e3:
            break
        tried.append(f"{method}: llf={res.llf}, max SE={np.nanmax(se):.3g} "
                     f"({len(caught)} warnings)")
    else:
        raise RuntimeError("mixed model did not converge with any optimizer: " + "; ".join(tried))
    ci = res.conf_int()
    t = pd.DataFrame({"coef": res.params, "se": res.bse, "z": res.tvalues, "p": res.pvalues,
                      "ci_lo": ci[0], "ci_hi": ci[1]})
    t = t.drop(index=[i for i in t.index if i == "Group Var"], errors="ignore")
    t = t.reset_index().rename(columns={"index": "term"})
    return t.assign(optimizer=method, group_var=float(np.asarray(res.cov_re)[0, 0]),
                    scale=float(res.scale), llf=float(res.llf))


def signed_rank_z(v: np.ndarray) -> tuple[float, int]:
    """Normal-approximation z of the Wilcoxon signed-rank statistic T+ (zeros dropped,
    average ranks for ties, tie-corrected variance) and the number of non-zero values.

    Computed from the statistic itself, so it is valid for any ``alternative`` (the
    p value of a one-sided test cannot be turned back into z with ``isf(p / 2)``).
    """
    d = np.asarray(v, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n == 0:
        return np.nan, 0
    rk = stats.rankdata(np.abs(d))
    t_plus = rk[d > 0].sum()
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(rk, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - np.sum(counts ** 3 - counts) / 48.0
    return (float((t_plus - mean) / np.sqrt(var)) if var > 0 else np.nan), n


def wilcoxon_one_sample(x: np.ndarray, alternative: str = "two-sided") -> dict[str, Any]:
    """Signed-rank test against 0; effect size r = z / sqrt(n), signed (|r| < 1)."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return {"W": np.nan, "p": np.nan, "r": np.nan}
    r = stats.wilcoxon(v, alternative=alternative)
    z, n = signed_rank_z(v)
    return {"W": float(r.statistic), "p": float(r.pvalue),
            "r": float(z / np.sqrt(n)) if n else np.nan}


def nan_quiet(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Silence the expected all-NaN reduction warnings *inside* ``fn`` only."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
            warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
            return fn(*args, **kwargs)

    return wrapper
