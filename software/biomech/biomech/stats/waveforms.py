# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Waveform statistics: SPM1d tests and variability decomposition (analysis_plan §5.2-5.3).

Profiles are (cycles, 101) arrays, one per participant (time-normalized cycles).
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

# spm1d's permutation tests draw from numpy's *global* RNG. Unseeded, their critical
# thresholds changed from run to run; under concurrent cohort stages the draws of two
# tests could interleave. Every permutation test therefore runs under this lock with the
# global state seeded (and restored afterwards).
_GLOBAL_RNG = threading.Lock()


@contextmanager
def seeded_global_rng(seed: int) -> Iterator[None]:
    with _GLOBAL_RNG:
        state = np.random.get_state()
        np.random.seed(int(seed) % 2 ** 32)
        try:
            yield
        finally:
            np.random.set_state(state)


def _clusters(ti: Any) -> list[dict[str, float]]:
    out = []
    for c in getattr(ti, "clusters", []) or []:
        a, b = c.endpoints
        out.append({"start_pct": float(a), "end_pct": float(b), "p": float(c.P)})
    return out


def _clean_rows(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, float)
    return Y[np.isfinite(Y).all(axis=1)]


def paired_t(YA: np.ndarray, YB: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """Parametric SPM paired t-test on subject means (rows paired)."""
    import spm1d

    ok = np.isfinite(YA).all(1) & np.isfinite(YB).all(1)
    YA, YB = YA[ok], YB[ok]
    if len(YA) < 5:
        return {"n": int(len(YA)), "h0reject": None, "zstar": np.nan, "clusters": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ti = spm1d.stats.ttest_paired(YA, YB).inference(alpha, two_tailed=True)
    return {"n": int(len(YA)), "h0reject": bool(ti.h0reject), "zstar": float(ti.zstar),
            "z": ti.z.tolist(), "clusters": _clusters(ti)}


def paired_t_nonparam(YA: np.ndarray, YB: np.ndarray, alpha: float = 0.05,
                      iterations: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Nonparametric (permutation) SPM paired t-test: single-participant use (cycles paired).

    Deterministic for a given ``seed`` (see :func:`seeded_global_rng`)."""
    import spm1d

    ok = np.isfinite(YA).all(1) & np.isfinite(YB).all(1)
    YA, YB = YA[ok], YB[ok]
    if len(YA) < 6:
        return {"n": int(len(YA)), "h0reject": None, "clusters": []}
    iters = min(iterations, 2 ** len(YA))
    with warnings.catch_warnings(), seeded_global_rng(seed):
        warnings.simplefilter("ignore")
        ti = spm1d.stats.nonparam.ttest_paired(YA, YB).inference(
            alpha, two_tailed=True, iterations=iters)
    return {"n": int(len(YA)), "h0reject": bool(ti.h0reject), "zstar": float(ti.zstar),
            "clusters": _clusters(ti)}


def anova1_participants(profiles: dict[str, np.ndarray], alpha: float = 0.05) -> dict[str, Any]:
    """One-way SPM ANOVA across participants, cycles as repetitions (descriptive, §5.3.1)."""
    import spm1d

    Y, A = [], []
    for k, (pid, P) in enumerate(sorted(profiles.items())):
        P = _clean_rows(P)
        if len(P) >= 3:
            Y.append(P)
            A += [k] * len(P)
    if len(Y) < 3:
        return {"h0reject": None, "n_groups": len(Y), "clusters": []}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fi = spm1d.stats.anova1(np.vstack(Y), np.array(A)).inference(alpha)
    return {"h0reject": bool(fi.h0reject), "n_groups": len(Y), "zstar": float(fi.zstar),
            "supra_frac": float(np.mean(fi.z > fi.zstar)), "clusters": _clusters(fi)}


def variance_ratio(profiles: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pointwise between-participant variance of the means vs mean within-participant variance."""
    P = {k: _clean_rows(v) for k, v in profiles.items()}
    P = {k: v for k, v in P.items() if len(v) >= 3}
    if len(P) < 3:
        return {}
    means = np.stack([v.mean(0) for v in P.values()])
    within = np.stack([v.var(0, ddof=1) for v in P.values()]).mean(0)
    between = means.var(0, ddof=1)
    return {"between": between, "within": within,
            "ratio": between / np.maximum(within, 1e-12)}


def cmc(Y: np.ndarray) -> float:
    """Coefficient of multiple correlation (Kadaba 1989) of a set of curves (rows)."""
    Y = _clean_rows(Y)
    m, n = Y.shape if Y.ndim == 2 else (0, 0)
    if m < 2:
        return np.nan
    mean_t = Y.mean(0)
    grand = Y.mean()
    num = ((Y - mean_t) ** 2).sum() / (n * (m - 1))
    den = ((Y - grand) ** 2).sum() / (m * n - 1)
    val = 1 - num / den if den > 0 else np.nan
    return float(np.sqrt(val)) if val >= 0 else np.nan


def cycles_needed(values: np.ndarray, target_halfwidth: float, n_boot: int = 2000,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Bootstrap 95 % CI half-width of the mean vs number of cycles; first n meeting target."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    ns = np.arange(1, v.size + 1)
    hw = np.full(ns.size, np.nan)
    for i, n in enumerate(ns):
        if n < 2:
            continue
        means = rng.choice(v, (n_boot, n)).mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        hw[i] = (hi - lo) / 2
    hit = np.flatnonzero(hw <= target_halfwidth)
    return ns, hw, int(ns[hit[0]]) if hit.size else None
