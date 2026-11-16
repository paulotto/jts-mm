# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared cohort test building blocks of the post hoc families (code-quality
rule "one shared test path"): C1 (D24 / D33, ``stats.cohort_kinematics``), C2-E (X6,
``stats.cohort_c2e``) and camera-free.

* :func:`wilcoxon_signed` -- one-sided / two-sided signed-rank test with the exact null
  distribution where scipy allows it (no zeros or ties), else the normal approximation;
  :func:`wilcoxon_greater` is its ``greater`` form (D24, unchanged).
* :func:`log_ratio_test` -- the cohort test of per-participant log(observed / null) (D24:
  hinge residual / noise P99 > 0; X6: locking IQR / S1 IQR < 0).
* :func:`location_test` -- per-participant locations against a margin: BCa CI of the
  median (``stats.core.describe``) and the signed-rank test of value - margin.
* :func:`iut` -- intersection-union test: a hypothesis made of several parts holds only if
  every part rejects; p = max of the parts (X6 B5).
* :func:`gatekeeping` -- serial gatekeeping: the secondary family is tested (Holm) only when
  the primary is rejected.
* :func:`block_holm` -- Holm within a block of supporting rows.
* :func:`paired_contrast` -- a paired contrast a − b (camera-free ladder, X5b / X5c): the
  two-sided signed-rank test of the differences (zeros dropped) and the Hodges–Lehmann
  estimate with its exact CI (:func:`hodges_lehmann`, :func:`signed_rank_null`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sps

from biomech.stats.core import describe, holm


def wilcoxon_signed(x: np.ndarray, alternative: str = "greater", min_n: int = 2
                    ) -> tuple[float, float, bool]:
    """Signed-rank test of median(x) against 0 over the finite values: (W+, p, exact).
    Exact null distribution where scipy allows it (no zeros or ties), else the normal
    approximation; NaN with fewer than ``min_n`` values."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < max(min_n, 2):
        return np.nan, np.nan, False
    exact = bool(np.all(v != 0) and np.unique(np.abs(v)).size == v.size)
    r = sps.wilcoxon(v, alternative=alternative, method="exact" if exact else "asymptotic")
    return float(r.statistic), float(r.pvalue), exact


def wilcoxon_greater(x: np.ndarray) -> tuple[float, float, bool]:
    """One-sided Wilcoxon signed-rank test of median(x) > 0 (D24): :func:`wilcoxon_signed`."""
    return wilcoxon_signed(x, "greater")


def log_ratio_test(log_ratio: np.ndarray, alternative: str, seed: int, min_n: int = 5
                   ) -> dict[str, Any]:
    """Cohort test of per-participant log ratios (``alternative`` "less": observed below its
    null, e.g. a locking IQR; "greater": above). ``n``, the median log ratio with its BCa CI
    (``lr_median``, ``lr_ci_lo``, ``lr_ci_hi``), ``W``, ``p``, ``exact``."""
    v = np.asarray(log_ratio, float)
    d = describe(v, seed=seed, stat="median")
    w, p, ex = wilcoxon_signed(v, alternative, min_n)
    return {"n": d["n"], "lr_median": d["median"], "lr_ci_lo": d["ci_lo"],
            "lr_ci_hi": d["ci_hi"], "W": w, "p": p, "exact": ex}


def location_test(values: np.ndarray, margin: float | None, seed: int, min_n: int = 5,
                  alternative: str = "greater") -> dict[str, Any]:
    """Per-participant locations (ms): ``n``, ``median`` with its BCa CI (``ci_lo``,
    ``ci_hi``; ``stats.core.describe``) and the signed-rank test of value - ``margin``
    (``alternative``; NaN p without a margin)."""
    v = np.asarray(values, float)
    d = describe(v, seed=seed, stat="median")
    w, p, ex = ((np.nan, np.nan, False) if margin is None or not np.isfinite(margin)
                else wilcoxon_signed(v - margin, alternative, min_n))
    return {"n": d["n"], "median": d["median"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
            "W": w, "p": p, "exact": ex}


def iut(*p: float) -> float:
    """Intersection-union p: the largest part p (NaN if any part is NaN)."""
    v = np.asarray(p, float)
    return float(v.max()) if v.size and np.isfinite(v).all() else np.nan


def gatekeeping(p_primary: float, p_secondary: list[float], alpha: float = 0.05
                ) -> tuple[bool, list[float]]:
    """Serial gatekeeping: (primary rejected, Holm-adjusted secondary p values); the
    secondary family is NaN (not tested) unless ``p_primary`` <= ``alpha``."""
    rejected = bool(np.isfinite(p_primary) and p_primary <= alpha)
    if not rejected:
        return False, [np.nan] * len(p_secondary)
    return True, holm(list(p_secondary))


def block_holm(p: list[float]) -> list[float]:
    """Holm within one block of supporting rows (NaN kept)."""
    return holm(list(p))


def signed_rank_null(n: int) -> np.ndarray:
    """Exact null distribution (counts) of the Wilcoxon T+ for n untied ranks."""
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1.0
    for r in range(1, n + 1):
        c[r:] = c[r:] + c[:-r].copy()
    return c


def hodges_lehmann(d: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """One-sample Hodges–Lehmann estimate (median of the Walsh averages) with the exact CI
    from the signed-rank distribution, over the finite values (NaN CI when n is too small for
    ``level``)."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return np.nan, np.nan, np.nan
    i, j = np.triu_indices(n)
    w = np.sort(0.5 * (d[i] + d[j]))
    est = float(np.median(w))
    cdf = np.cumsum(signed_rank_null(n)) / 2.0 ** n
    below = np.flatnonzero(cdf <= (1 - level) / 2)
    if below.size == 0:
        return est, np.nan, np.nan
    # c = largest k with P(T+ <= k) <= alpha/2: the interval [W_(c+1), W_(M-c)] (1-based
    # order statistics of the M Walsh averages) has coverage 1 - 2 P(T+ <= c) >= level
    c = int(below[-1])
    return est, float(w[c]), float(w[len(w) - 1 - c])


def paired_contrast(d: np.ndarray, level: float = 0.95) -> dict[str, Any]:
    """Paired contrast of the differences ``d`` = a − b: ``p`` two-sided signed-rank
    (:func:`wilcoxon_signed` of the finite non-zero differences; ``method`` "exact", "approx"
    with ties, "" below 2 values) and ``hl`` with its exact CI (``hl_ci_lo``, ``hl_ci_hi``;
    :func:`hodges_lehmann` of the finite differences, zeros included)."""
    v = np.asarray(d, float)
    _, p, exact = wilcoxon_signed(v[np.isfinite(v) & (v != 0)], "two-sided")
    hl, lo, hi = hodges_lehmann(v, level)
    return {"hl": hl, "hl_ci_lo": lo, "hl_ci_hi": hi, "p": p,
            "method": ("exact" if exact else "approx") if np.isfinite(p) else ""}
