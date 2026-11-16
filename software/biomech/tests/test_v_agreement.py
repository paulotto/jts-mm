# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cycle-level agreement (validation plan §5): CCC, ICC(A,1), bootstrap BA, cycles on the
reference signal, equivalence and waveform SPM."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.validation import agreement as AG
from biomech.validation.selection import Interval


def test_agreement_indices():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 200)
    assert AG.ccc(a, a) == pytest.approx(1.0)
    assert AG.icc_a1(a, a) == pytest.approx(1.0)
    b = a + 1.0                                          # constant bias lowers both
    assert AG.ccc(a, b) < 0.95 and AG.icc_a1(a, b) < 0.95
    ba = AG.ba_boot(a + rng.normal(0.2, 0.1, 200), a)
    assert ba["bias"] == pytest.approx(0.2, abs=0.03)
    assert ba["bias_ci_lo"] < 0.2 < ba["bias_ci_hi"]


def test_run_on_synthetic_cycles():
    fs = 40.0
    t = np.arange(0, 60, 1 / fs)
    z = -15.0 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2           # 30 opening cycles
    rng = np.random.default_rng(1)
    inc_q = np.c_[0.2 * z, np.zeros_like(t), z] + rng.normal(0, 0.05, (len(t), 3))
    inc_s = inc_q + rng.normal(0, 0.1, inc_q.shape) + [0.0, 0.0, 0.3]
    rot_q = np.c_[np.zeros_like(t), 0.6 * z, np.zeros_like(t)] + rng.normal(0, 0.02, (len(t), 3))
    arr = {"t_master_s": t, "incisor_q_mm": inc_q, "incisor_s_mm": inc_s, "rot_q_deg": rot_q,
           "rot_s_deg": rot_q + rng.normal(0, 0.05, rot_q.shape), "gate": np.zeros(len(t), bool)}
    tables, arrays = AG.run_ref_agreement(arr, [Interval("open_close", 0.5, 59.5)], "s99")
    assert len(tables["cycles"]) >= 25
    f = tables["features"].set_index("feature")
    assert f.loc["amp_mm", "bias"] == pytest.approx(0.0, abs=0.1)     # offset cancels
    assert f.loc["amp_mm", "equivalent_application"]
    # VD13: the margin is the smallest symmetric bound around the bias CI
    m = f.loc["amp_mm"]
    assert m["equiv_margin"] == pytest.approx(max(abs(m["bias_ci_lo"]), abs(m["bias_ci_hi"])))
    assert m["equivalent_application"] == (m["equiv_margin"] < AG.TOL["application"]["mm"])
    w = tables["waveforms"]
    assert set(w["mode"]) == {"abs", "disp"}
    w = w[w["mode"] == "abs"].set_index("channel")
    assert w.loc["incisor_z", "spm_h0reject"]                         # the 0.3 mm offset
    assert w.loc["incisor_z", "frac_inside_application"] == pytest.approx(1.0)
    # the 0.3 mm offset sits inside the whole band: margin above it, below the bound
    assert 0.3 < w.loc["incisor_z", "equiv_margin"] < AG.TOL["application"]["mm"]
    assert "open_close__incisor_z__diff_mean" in arrays
    assert "open_close__incisor_z__disp__diff_mean" in arrays
    wd = tables["waveforms"].query("mode == 'disp'").set_index("channel")
    assert wd.loc["incisor_z", "frac_inside_metrology"] > 0.9       # offset removed


# -- EV6: bias CI of the cycle features ---------------------------------------------------------
def test_bias_t_interval_arithmetic():
    """t-interval d̄ ± t(.975, n_eff − 1) s/√n_eff with n_eff = n (1 − ρ1)/(1 + ρ1)."""
    from scipy.stats import t as T

    d = np.array([0.1, 0.4, 0.2, 0.5, 0.3, 0.6])          # rising: ρ1 > 0
    x = d - d.mean()
    rho = np.sum(x[1:] * x[:-1]) / np.sum(x * x)
    assert AG.lag1_autocorr(d) == pytest.approx(rho)
    r, ne = AG.n_effective(d)
    assert r == pytest.approx(max(rho, 0.0))
    assert ne == pytest.approx(6 * (1 - r) / (1 + r))
    ci = AG.bias_t_interval(d)
    df = max(ne - 1, 1)
    half = T.ppf(0.975, df) * d.std(ddof=1) / np.sqrt(ne)
    assert ci["bias_ci_lo"] == pytest.approx(d.mean() - half)
    assert ci["bias_ci_hi"] == pytest.approx(d.mean() + half)
    assert ci["t_df"] == pytest.approx(df) and ci["n_eff"] == pytest.approx(ne)


def test_negative_autocorrelation_is_clipped():
    d = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]) + 0.2  # alternating: ρ1 < 0
    assert AG.lag1_autocorr(d) < 0
    r, ne = AG.n_effective(d)
    assert r == 0.0 and ne == pytest.approx(len(d))
    from scipy.stats import t as T
    ci = AG.bias_t_interval(d)
    half = T.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))   # plain t-interval
    assert ci["bias_ci_hi"] - ci["bias_ci_lo"] == pytest.approx(2 * half)


def test_ba_boot_time_order_and_sensitivities():
    """The CI is computed in time order (``order``), the bootstraps are kept as *_boot and
    *_mbb, the margin is the primary t-interval's."""
    d = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    perm = np.array([3, 0, 6, 1, 7, 2, 5, 4])
    a = AG.ba_boot(d, np.zeros_like(d))
    b = AG.ba_boot(d[perm], np.zeros_like(d), order=perm)  # start frames restore the order
    for k in ("bias_ci_lo", "bias_ci_hi", "rho1", "n_eff"):
        assert a[k] == pytest.approx(b[k])
    c = AG.ba_boot(d[perm], np.zeros_like(d))               # shuffled: other ρ1
    assert c["rho1"] != pytest.approx(a["rho1"])
    for k in ("bias_ci_lo_boot", "bias_ci_hi_boot", "bias_ci_lo_mbb", "bias_ci_hi_mbb",
              "loa_lo_ci_lo", "loa_hi_ci_hi"):
        assert np.isfinite(a[k])
    assert a["bias_ci_lo_boot"] < a["bias"] < a["bias_ci_hi_boot"]
    assert AG.margin(-0.3, 0.1) == pytest.approx(0.3)


def test_moving_block_means_uses_successive_pairs():
    d = np.arange(5, dtype=float)
    rng = np.random.default_rng(0)
    m = AG.moving_block_means(d, rng, 200, block=2)
    # every resample is 3 blocks (i, i+1) cut to 5 values: (0,1,0,1,0) .. (3,4,3,4,3)
    assert m.shape == (200,) and m.min() >= 0.4 - 1e-12 and m.max() <= 3.4 + 1e-12
    assert AG.moving_block_means(d, rng, 50, block=1).std() > 0


def test_features_margin_from_primary_and_sensitivity_columns():
    fs = 40.0
    t = np.arange(0, 40, 1 / fs)
    z = -15.0 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2
    rng = np.random.default_rng(3)
    inc_q = np.c_[0.2 * z, np.zeros_like(t), z] + rng.normal(0, 0.05, (len(t), 3))
    inc_s = inc_q + rng.normal(0, 0.1, inc_q.shape)
    rot_q = np.c_[np.zeros_like(t), 0.6 * z, np.zeros_like(t)]
    arr = {"t_master_s": t, "incisor_q_mm": inc_q, "incisor_s_mm": inc_s, "rot_q_deg": rot_q,
           "rot_s_deg": rot_q + rng.normal(0, 0.05, rot_q.shape)}
    tab, _ = AG.run_ref_agreement(arr, [Interval("open_close", 0.5, 39.5)], "s99")
    f = tab["features"].set_index("feature").loc["amp_mm"]
    assert f["equiv_margin"] == pytest.approx(max(abs(f["bias_ci_lo"]), abs(f["bias_ci_hi"])))
    assert f["equiv_margin_boot"] == pytest.approx(max(abs(f["bias_ci_lo_boot"]),
                                                       abs(f["bias_ci_hi_boot"])))
    assert f["equiv_margin_mbb"] == pytest.approx(max(abs(f["bias_ci_lo_mbb"]),
                                                      abs(f["bias_ci_hi_mbb"])))
    assert 0 <= f["rho1"] <= 0.99 and f["n_eff"] <= f["n"]


def _ar1(rng, n, phi):
    e = np.empty(n)
    e[0] = rng.normal() / np.sqrt(1 - phi ** 2)
    z = rng.normal(size=n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + z[i]
    return e


@pytest.mark.slow
def test_bias_ci_coverage_ar1():
    """EV6 coverage by simulation: AR(1) cycle errors (φ = 0, 0.3, 0.5), n = 5 / 8 / 18,
    4000 replicates each. Recorded 2026-10-29 (seed 1): primary t-interval
    φ=0: 0.96 / 0.96 / 0.96 (target 0.93–0.97 met); φ=0.3: 0.90 / 0.90 / 0.91;
    φ=0.5: 0.85 / 0.86 / 0.88 (below target: ρ̂1 from 5–18 cycles is biased low and
    clipped at 0); independent-cycle bootstrap 0.60–0.93, block bootstrap 0.54–0.91."""
    rng = np.random.default_rng(1)
    R = 4000
    cov = {}
    for phi in (0.0, 0.3, 0.5):
        for n in (5, 8, 18):
            hit = np.zeros(3)
            for r in range(R):
                b = AG.ba_boot(_ar1(rng, n, phi) + 0.3, np.zeros(n), n_boot=400, seed=r)
                hit += [b[f"bias_ci_lo{s}"] <= 0.3 <= b[f"bias_ci_hi{s}"]
                        for s in ("", "_boot", "_mbb")]
            cov[(phi, n)] = hit / R
    print({k: np.round(v, 3).tolist() for k, v in cov.items()})
    for n in (5, 8, 18):
        assert 0.93 <= cov[(0.0, n)][0] <= 0.97                     # target, independent
        for phi in (0.0, 0.3, 0.5):
            assert cov[(phi, n)][0] > cov[(phi, n)][1]               # beats the old CI
            assert cov[(phi, n)][0] >= 0.82


def test_waveform_band_is_the_effective_n_interval_per_node():
    """EV14: the waveform band is bias_t_interval node by node; with positively correlated
    successive cycles it is wider than the independent-cycle bootstrap band."""
    import numpy as np

    from biomech.validation.agreement import bias_t_interval, pointwise_t_band

    rng = np.random.default_rng(3)
    n, k = 20, 11
    e = np.zeros((n, k))
    for i in range(1, n):                                   # AR(1) over cycles, rho 0.6
        e[i] = 0.6 * e[i - 1] + rng.normal(0, 1, k)
    lo, hi = pointwise_t_band(e)
    for j in (0, 5, 10):
        b = bias_t_interval(e[:, j])
        assert np.isclose(lo[j], b["bias_ci_lo"]) and np.isclose(hi[j], b["bias_ci_hi"])
    boot = e[rng.integers(0, n, (4000, n))].mean(axis=1)
    blo, bhi = np.percentile(boot, [2.5, 97.5], axis=0)
    assert np.mean((hi - lo) > (bhi - blo)) > 0.8
    flat = np.zeros((5, 3))                                 # no spread: the mean itself
    assert np.allclose(pointwise_t_band(flat), 0.0)
