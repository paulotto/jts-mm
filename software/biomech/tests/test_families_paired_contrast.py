# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The camera-free paired contrasts on the shared test path ("one shared test
path"): :func:`biomech.stats.families.paired_contrast` (signed-rank p via
:func:`~biomech.stats.families.wilcoxon_signed`, Hodges–Lehmann with exact CI) replaces the
module-local ``wilcoxon_exact`` / ``hodges_lehmann`` of ``stats.cohort_camerafree``.

``_old_*`` below are the removed functions, frozen verbatim as the reference: the shared path
must give identical values (bit for bit) on untied inputs, zeros and NaN included. With tied
|d| the old code passed the ties to scipy's ``method="exact"`` (a conservatively rounded
no-ties distribution, labelled "exact"); the shared path uses the normal approximation and
says so (``method`` "approx"), as for C1 and C2-E.

Correction 2026-11-01: the old interval cut one Walsh average too many at each end
(coverage 0.943 at n = 13 instead of >= 0.95); the reference below carries the corrected
index, the estimate and the p values are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from biomech.stats.families import hodges_lehmann, paired_contrast, signed_rank_null


# -- frozen reference: stats/cohort_camerafree.py before 2026-10-31 --------------------------
def _old_signed_rank_null(n: int) -> np.ndarray:
    c = np.zeros(n * (n + 1) // 2 + 1)
    c[0] = 1.0
    for r in range(1, n + 1):
        c[r:] = c[r:] + c[:-r].copy()
    return c


def _old_hodges_lehmann(d: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return np.nan, np.nan, np.nan
    i, j = np.triu_indices(n)
    w = np.sort(0.5 * (d[i] + d[j]))
    est = float(np.median(w))
    cdf = np.cumsum(_old_signed_rank_null(n)) / 2.0 ** n
    below = np.flatnonzero(cdf <= (1 - level) / 2)
    if below.size == 0:
        return est, np.nan, np.nan
    c = int(below[-1])               # corrected (was + 1): [W_(c+1), W_(M-c)], 1-based
    return est, float(w[c]), float(w[len(w) - 1 - c])


def _old_wilcoxon_exact(d: np.ndarray) -> tuple[float, str]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size < 2:
        return np.nan, ""
    try:
        return float(sps.wilcoxon(d, method="exact").pvalue), "exact"
    except (ValueError, TypeError):
        return float(sps.wilcoxon(d, method="approx").pvalue), "approx"


def _same(a: float, b: float) -> bool:
    return (np.isnan(a) and np.isnan(b)) or a == b


def _draws():
    for seed in range(300):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(0, 25))
        d = rng.normal(rng.normal(0, 0.5), rng.gamma(2, 1), n)
        d[rng.random(n) < 0.1] = 0.0                       # exact zeros (dropped by the test)
        d[rng.random(n) < 0.1] = np.nan
        yield seed, d


def test_signed_rank_null_is_the_old_one():
    for n in range(0, 30):
        np.testing.assert_array_equal(signed_rank_null(n), _old_signed_rank_null(n))


@pytest.mark.parametrize("level", [0.95, 0.9])
def test_hodges_lehmann_identical(level):
    for seed, d in _draws():
        new, old = hodges_lehmann(d, level), _old_hodges_lehmann(d, level)
        assert all(_same(a, b) for a, b in zip(new, old)), (seed, new, old)


def test_paired_contrast_identical_without_ties():
    for seed, d in _draws():
        r = paired_contrast(d)
        p, method = _old_wilcoxon_exact(d)
        assert _same(r["p"], p) and r["method"] == method, (seed, r, p, method)
        assert (r["hl"], r["hl_ci_lo"], r["hl_ci_hi"]) == pytest.approx(
            _old_hodges_lehmann(d), nan_ok=True)


def test_paired_contrast_ties_use_the_normal_approximation():
    d = np.array([1.0, 2.0, 2.0, -3.0, 4.0, 5.0])
    r = paired_contrast(d)
    assert r["method"] == "approx"
    assert r["p"] == pytest.approx(sps.wilcoxon(d, method="approx").pvalue)
    assert _old_wilcoxon_exact(d)[1] == "exact"             # the old label, for the record


def test_paired_contrast_small_n():
    assert np.isnan(paired_contrast(np.array([0.0, 1.0]))["p"])      # one non-zero value
    assert paired_contrast(np.array([0.0, 1.0]))["method"] == ""
    r = paired_contrast(np.array([]))
    assert np.isnan(r["p"]) and np.isnan(r["hl"]) and r["method"] == ""


def test_hodges_lehmann_interval_coverage():
    """The exact interval holds its level (t5 data, n = 13 as in the cohort)."""
    rng = np.random.default_rng(7)
    hits = 0
    for _ in range(2000):
        _, lo, hi = hodges_lehmann(rng.standard_t(5, 13))
        hits += lo <= 0.0 <= hi
    assert hits / 2000 >= 0.94                 # nominal >= 0.95; 2000 draws, SE 0.005

