# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R10 fixes of in the requirements chain (E31, E32, D26, D27, D31).

* E31: the P95 design value (Harrell-Davis next to the linear value), the exact
  order-statistic lower bound, no finite upper bound, the maximum's coverage statement;
  coverage simulation of the bound (``slow``);
* E32: the requirements and stats_imu stages read ``imu`` as a *partial* dependency -- a
  participant without IMU data enters the optical rows and the generic opening model
  (toy stages through the real scheduler, and the functions on a synthetic participant
  without IMU);
* D26: the own opening model leave-one-cycle-out, and the MOVER interval of the largest
  absolute limit of agreement with its coverage against the cluster bootstrap (``slow``);
* D27: the uniform per-participant statistic and the bandwidth from each sensor alone;
* D31/D23: the C2 latency-budget rows carry the chance null and the onset-source leads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from biomech.stats.requirements import (COLUMNS, harrell_davis, order_stat_lower,
                                        order_stat_upper, p95_design, requirements_sensitivity,
                                        requirements_table)


# -- E31 -------------------------------------------------------------------------------------
def _brute_lower(n: int, p: float = 0.95, conf: float = 0.95) -> tuple[int, float]:
    """P(X(k) <= Q_p) = P(at least k of n values <= Q_p), summed term by term."""
    from math import comb

    best = (0, np.nan)
    for k in range(1, n + 1):
        c = sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
        if c >= conf:
            best = (k, c)
    return best


@pytest.mark.parametrize("n", [5, 9, 12, 13, 14, 20, 40])
def test_order_stat_lower_matches_the_binomial_sum(n):
    k, c = order_stat_lower(n)
    kb, cb = _brute_lower(n)
    assert k == kb and c == pytest.approx(cb, abs=1e-12)


def test_order_stat_bounds_at_the_cohort_sizes():
    assert order_stat_lower(14) == (12, pytest.approx(0.9699, abs=1e-4))
    assert order_stat_lower(13) == (11, pytest.approx(0.9755, abs=1e-4))
    # the next order statistic would not reach 95 %
    assert sps.binom.sf(12, 14, 0.95) < 0.95 and sps.binom.sf(11, 13, 0.95) < 0.95
    # no finite one-sided upper bound below n = 59 (1 - 0.95^n < 0.95)
    assert order_stat_upper(14) == (0, pytest.approx(np.nan, nan_ok=True))
    assert order_stat_upper(58)[0] == 0 and order_stat_upper(59)[0] == 59


def test_p95_design_block():
    rng = np.random.default_rng(1)
    x = rng.normal(30, 5, 14)
    d = p95_design(np.r_[x, np.nan])
    s = np.sort(x)
    from scipy.stats.mstats import hdquantiles

    assert d["n"] == 14 and d["value_linear"] == pytest.approx(np.percentile(x, 95))
    assert d["value"] == pytest.approx(float(np.asarray(hdquantiles(x, prob=[0.95]))[0]))
    assert d["value"] == pytest.approx(harrell_davis(x))
    assert d["ci_lo"] == s[11] and d["lower_k"] == 12 and np.isnan(d["ci_hi"])
    assert d["upper_note"] == "none finite at n = 14"
    assert d["sample_max"] == s[-1] and d["max_q"] == 0.8
    assert d["max_conf"] == pytest.approx(1 - 0.8 ** 14) and round(d["max_conf"], 3) == 0.956
    # reporting-population size: the same rule at n = 13
    d13 = p95_design(x[:13])
    assert d13["lower_k"] == 11 and d13["ci_lo"] == np.sort(x[:13])[10]
    assert d13["upper_note"] == "none finite at n = 13"


def test_p95_design_small_samples():
    assert np.isnan(p95_design(np.array([np.nan]))["value"])
    one = p95_design(np.array([3.0]))
    assert one["value"] == 3.0 and one["lower_k"] == 1 and one["ci_lo"] == 3.0


@pytest.mark.slow
@pytest.mark.parametrize("n", [13, 14])
def test_order_stat_lower_bound_coverage(n, record_property):
    """Coverage of the one-sided lower bound of the population P95 (lognormal, 20 000
    samples): the exact value is P(Bin(n, .95) >= k) (0.970 at n = 14, 0.975 at n = 13),
    whatever the distribution; and the maximum exceeds the population P80 with probability
    1 - 0.8^n."""
    rng = np.random.default_rng(31 + n)
    q95, q80 = np.exp(sps.norm.ppf([0.95, 0.80]) * 0.6)
    x = np.sort(rng.lognormal(0.0, 0.6, (20000, n)), axis=1)
    k, conf = order_stat_lower(n)
    cov = float(np.mean(x[:, k - 1] <= q95))
    cov_max = float(np.mean(x[:, -1] > q80))
    record_property("coverage_lower", cov)
    print(f"n = {n}: lower-bound coverage {cov:.4f} (exact {conf:.4f}); "
          f"max > P80 {cov_max:.4f} (exact {1 - 0.8 ** n:.4f})")
    assert cov == pytest.approx(conf, abs=0.006)
    assert cov_max == pytest.approx(1 - 0.8 ** n, abs=0.006)


# -- D31 / D23: the C2 rows -----------------------------------------------------------------
def _timing(p_null_holm: float) -> pd.DataFrame:
    rows = []
    for rule, med in (("first", 108.0), ("drive", 49.0)):
        base = {"task": "open_close", "direction": "opening", "detector": "tkeo", "rule": rule,
                "n": 9, "ci_lo": 70.0, "ci_hi": 200.0}
        rows.append({**base, "metric": "lead_ms", "median": med, "p": 0.006, "p_holm": 0.02,
                     "lead_null_ms": 190.0, "lead_minus_null_ms": -60.0, "p_null": 0.6,
                     "p_null_holm": p_null_holm})
        rows.append({**base, "metric": "lead_omega_ms", "median": med + 33})
        rows.append({**base, "metric": "lead_opt_ms", "median": med - 15})
    return pd.DataFrame(rows)


def test_c2_rows_carry_the_null_and_the_onset_sources():
    e = pd.DataFrame()
    t = requirements_table(e, e, e, e, _timing(0.9), e, e, e, e, seed=0)
    assert list(t.columns) == COLUMNS
    r = t.set_index("key").loc["lead_first_ms"]
    assert r.value == 108.0 and r.lead_null_ms == 190.0 and r.p_null_holm == 0.9
    assert r.lead_omega_ms == 141.0 and r.lead_opt_ms == 93.0
    assert r.chance_distinguishable is False or r.chance_distinguishable == 0
    assert "not distinguishable from chance" in r.note and "Holm p = 0.020" in r.note
    # D34: the never-computed D31 per-source nulls are gone
    assert not {"lead_omega_null_ms", "lead_opt_null_ms", "p_null_holm_omega",
                "p_null_holm_opt"} & set(COLUMNS)
    t2 = requirements_table(e, e, e, e, _timing(0.01), e, e, e, e, seed=0)
    r2 = t2.set_index("key").loc["lead_drive_ms"]
    assert bool(r2.chance_distinguishable) and "not distinguishable" not in r2.note
    # the stored evidence comes from the one map (J5 / E43)
    assert set(t.evidence) == {"not_confirmed", "not_confirmed_posthoc_rule"}


def test_x3_note_states_anticipation_against_chance():
    """D34: the causal-release row says whether the anticipation exceeds its chance level;
    only the detection rate carries a statement when it does not."""
    e = pd.DataFrame()
    rel = pd.DataFrame([{"task": "open_close", "primary": True, "median": 103.0, "ci_lo": 60.0,
                         "ci_hi": 150.0, "n": 13, "detected_frac_median": 0.80,
                         "chance_detected_frac_median": 0.61, "false_per_min_median": 20.0,
                         "chance_anticipation_ms_median": 229.0, "p_vs_chance": 0.017}])
    t = requirements_table(e, e, e, e, e, e, rel, e, e, seed=0)
    note = t.set_index("key").loc["causal_release_ms", "note"]
    assert "detected 80 % vs chance 61 %" in note
    assert "anticipation 103 ms vs chance 229 ms (paired p = 0.017)" in note
    assert "not above chance" in note
    rel2 = rel.assign(chance_anticipation_ms_median=50.0)
    note2 = requirements_table(e, e, e, e, e, e, rel2, e, e, seed=0).set_index(
        "key").loc["causal_release_ms", "note"]
    assert "vs chance 50 ms" in note2 and "not above chance" not in note2


# -- E32: optical rows from every participant with optics ------------------------------------
def test_optical_rows_and_imu_rows_keep_their_own_n():
    pids = [f"s{i:02d}" for i in range(1, 15)]
    rng = np.random.default_rng(2)
    hinge = pd.DataFrame({"pid": pids * 2, "model": ["hinge_fit"] * 14 + ["hinge_tmj"] * 14,
                          "rms_incisor_mm": rng.normal(2, 0.4, 28)})
    imu_pids = [p for p in pids if p != "s09"]                       # s09: no jaw IMU
    bw = pd.DataFrame({"pid": imu_pids, "task": "open_close", "noise_limited": False,
                       "f95_coh_hz": rng.uniform(3, 8, 13)})
    e = pd.DataFrame()
    t = requirements_table(e, hinge, e, bw, e, e, e, e, e, seed=0).set_index("key")
    assert t.loc["misalign_hinge_fit_mm", "n"] == 14
    assert t.loc["misalign_hinge_fit_mm", "value"] == pytest.approx(
        np.median(hinge[hinge.model == "hinge_fit"].rms_incisor_mm))
    assert t.loc["bandwidth_hz", "n"] == 13 and t.loc["bandwidth_hz", "lower_k"] == 11


def _imu_inputs(n_pid: int = 5, no_imu: str = "s05"):
    rng = np.random.default_rng(7)
    models = pd.DataFrame({"pid": [f"s{i:02d}" for i in range(1, n_pid + 1)],
                           "a1_mm_per_deg": rng.uniform(1.0, 2.0, n_pid),
                           "a2_mm_per_deg2": rng.uniform(0.0, 0.02, n_pid)})
    rows = []
    for p in models.pid:
        if p == no_imu:
            continue
        for c in range(6):
            th = rng.uniform(10, 20)
            rows.append({"pid": p, "task": "open_close", "cycle": c, "theta_peak_quat_deg": th,
                         "theta_peak_comp_deg": th + 0.3, "opening_opt_mm": 1.5 * th,
                         "opening_subject_quat_mm": 1.5 * th + rng.normal(0, 0.5),
                         "opening_subject_insample_quat_mm": 1.5 * th + rng.normal(0, 0.3)})
    cyc = pd.DataFrame(rows)
    agree = pd.DataFrame({"pid": cyc.pid.unique(), "task": "open_close", "is_calibration": False,
                          "rmse_quat_deg": 0.7, "lag_ms": 5.0})
    return agree, pd.DataFrame({"pid": cyc.pid.unique(), "rms_deg": 0.3}), cyc, models


def test_generic_model_uses_every_participant_with_optics():
    from biomech.stats.cohort_misc import imu_tables

    agree, calib, cyc, models = _imu_inputs()
    out = imu_tables(agree, calib, cyc, models, seed=0)
    c = out["imu_opening_cycles"]
    m = models.set_index("pid")
    r = c[c.pid == "s01"].iloc[0]
    others = m.drop(index="s01")                                    # includes s05 (no IMU)
    want = others.a1_mm_per_deg.mean() * r.theta_peak_quat_deg \
        + others.a2_mm_per_deg2.mean() * r.theta_peak_quat_deg ** 2
    assert r.opening_loso_quat_mm == pytest.approx(want)
    ba = out["imu_opening_ba"]
    q = ba[(ba.task == "open_close") & (ba.estimator == "quat")]
    assert list(q.model) == ["loso", "subject", "subject_insample"]        # D26 order
    assert q.set_index("model").loc["loso", "loso_n_others"] == 4
    assert (q.n_subjects == 4).all()
    for _, row in q.iterrows():
        assert row.max_abs_loa == pytest.approx(max(abs(row.loa_lo), abs(row.loa_hi)))
        assert row.max_abs_loa_ci_lo <= row.max_abs_loa <= row.max_abs_loa_ci_hi


def test_requirements_imu_rows_generic_first_and_insample_as_sensitivity():
    from biomech.stats.cohort_misc import imu_tables

    agree, calib, cyc, models = _imu_inputs()
    ba = imu_tables(agree, calib, cyc, models, seed=0)["imu_opening_ba"]
    e = pd.DataFrame()
    t = requirements_table(e, e, e, e, e, e, e, e, ba, seed=0)
    assert list(t.key) == ["imu_opening_loa_loso_mm", "imu_opening_loa_subject_mm"]
    own = t.iloc[1]
    ref = ba[(ba.task == "open_close") & (ba.estimator == "quat")].set_index("model")
    assert own.value == pytest.approx(ref.loc["subject", "max_abs_loa"])
    assert own.ci_hi == pytest.approx(ref.loc["subject", "max_abs_loa_ci_hi"])
    assert own.ci_hi_boot == pytest.approx(ref.loc["subject", "max_abs_loa_boot_ci_hi"])
    assert own.value_insample == pytest.approx(ref.loc["subject_insample", "max_abs_loa"])
    assert own.ci_method == "mover_zou2013" and "in-sample own model" in own.note


@pytest.fixture
def partial_stages():
    """t_opt (participant, every pid) and t_imu (participant, fails for s02) feeding a
    cohort stage t_req that declares t_imu partial, as requirements/stats_imu do."""
    from biomech import pipeline
    from biomech.pipeline import StageResult, stage

    pipeline.registry()
    saved = dict(pipeline._REGISTRY)

    @stage("t_opt", version="1")
    def opt(ctx):
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [1.0]})})

    @stage("t_imu", depends=("t_opt",), version="1")
    def imu(ctx):
        if ctx.pid == "s02":
            raise LookupError("IMU streams missing")
        return StageResult(tables={"y": pd.DataFrame({"pid": [ctx.pid], "w": [2.0]})})

    @stage("t_req", scope="cohort", depends=("t_opt", "t_imu"), partial=("t_imu",))
    def req(ctx):
        return StageResult(tables={"n": pd.DataFrame({
            "n_opt": [ctx.upstream("t_opt", "x").pid.nunique()],
            "n_imu": [ctx.upstream("t_imu", "y").pid.nunique()]})})

    @stage("t_strict", scope="cohort", depends=("t_opt", "t_imu"))
    def strict(ctx):
        return StageResult(tables={"n": pd.DataFrame({
            "n_opt": [ctx.upstream("t_opt", "x").pid.nunique()]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def test_partial_dependency_keeps_a_participant_without_imu(study, partial_stages):
    from biomech.pipeline import run
    from biomech.results import COHORT_UNIT, ResultStore

    cfg = study()
    s = run(cfg, ["t_req", "t_strict"])
    assert s.statuses["t_imu"]["s02"] == "failed"
    assert s.statuses["t_req"] == {COHORT_UNIT: "ran"}
    store = ResultStore(cfg.out_dir)
    n = store.read_table("t_req", "n").iloc[0]
    assert n.n_opt == 2 and n.n_imu == 1                   # each input its own participants
    cov = store.read_table("t_req", "coverage").set_index("pid")
    assert bool(cov.loc["s02", "used"]) and "t_imu" in cov.loc["s02", "reason"]
    assert cov.loc["s01", "reason"] == ""
    # without the partial declaration the participant is dropped from every input (r9)
    assert store.read_table("t_strict", "n").iloc[0].n_opt == 1


def test_partial_must_be_a_dependency():
    from biomech.pipeline import stage

    with pytest.raises(ValueError, match="partial"):
        stage("t_bad", scope="cohort", depends=("qc",), partial=("imu",))


def test_requirements_and_stats_imu_read_imu_as_partial():
    from biomech.pipeline import get_stage

    for name in ("requirements", "stats_imu"):
        st = get_stage(name)
        assert st.partial == ("imu",) and "kinematics" in st.depends


# -- D26: own model out of sample ------------------------------------------------------------
def test_opening_model_loco_refits_without_each_cycle():
    from biomech.kinematics.analysis import _opening_fit, opening_model_loco

    rng = np.random.default_rng(3)
    th, op, cy = [], [], []
    for c, a1 in enumerate((1.4, 1.5, 1.6, 1.8)):
        t = np.linspace(0, 20, 30)
        th.append(t)
        op.append(a1 * t + 0.005 * t ** 2 + rng.normal(0, 0.1, t.size))
        cy.append(np.full(t.size, c + 2))
    th, op, cy = map(np.concatenate, (th, op, cy))
    loco = opening_model_loco(th, op, cy, "s01")
    assert list(loco.cycle) == [2, 3, 4, 5] and (loco.n == 90).all()
    for _, r in loco.iterrows():
        coef, _ = _opening_fit(th[cy != r.cycle], op[cy != r.cycle])
        assert r.a1_mm_per_deg == pytest.approx(coef[0]) and r.a2_mm_per_deg2 == pytest.approx(
            coef[1])
    one = opening_model_loco(th[cy == 2], op[cy == 2], cy[cy == 2], "s01")
    assert len(one) == 1 and np.isnan(one.a1_mm_per_deg.iloc[0])     # nothing left to fit


def test_own_model_openings_are_out_of_sample_for_fitted_cycles():
    from biomech.imu.analysis import own_model_openings

    cyc = pd.DataFrame({"task": ["open_close", "open_close", "open_close", "chewing"],
                        "cycle": [0, 1, 7, 0], "theta_peak_quat_deg": [10.0, 12.0, 11.0, 5.0]})
    full = pd.DataFrame({"a1_mm_per_deg": [1.5], "a2_mm_per_deg2": [0.01]})
    loco = pd.DataFrame({"cycle": [0, 1], "a1_mm_per_deg": [1.4, np.nan],
                         "a2_mm_per_deg2": [0.02, np.nan]})
    out = own_model_openings(cyc.copy(), full, loco, estimators=("quat",))
    assert out.opening_subject_quat_mm[0] == pytest.approx(1.4 * 10 + 0.02 * 100)
    assert np.isnan(out.opening_subject_quat_mm[1])          # fitted, no LOCO model possible
    assert out.opening_subject_quat_mm[2] == pytest.approx(1.5 * 11 + 0.01 * 121)  # not fitted
    assert out.opening_subject_quat_mm[3] == pytest.approx(1.5 * 5 + 0.01 * 25)    # chewing
    assert out.opening_subject_insample_quat_mm[0] == pytest.approx(1.5 * 10 + 0.01 * 100)


def test_mover_on_a_large_sample_brackets_the_truth_tightly():
    from biomech.stats.cohort_misc import max_abs_loa_mover

    rng = np.random.default_rng(12)
    k = 2000
    m = rng.integers(2, 9, k)
    g = np.repeat(np.arange(k), m)
    d = -0.4 + rng.normal(0, 1.0, k)[g] + rng.normal(0, 0.6, g.size)
    true = 0.4 + 1.96 * np.sqrt(1.36)
    v, lo, hi = max_abs_loa_mover(d, g)
    assert lo < true < hi and hi - lo < 0.2 and v == pytest.approx(true, abs=0.08)


def test_mover_without_between_variance_uses_the_within_term():
    from biomech.stats.cohort_misc import max_abs_loa_mover
    from biomech.stats.core import bland_altman_rm

    g = np.repeat(np.arange(4), 3)
    d = np.array([0.0, 1, 2, 0.1, 1.1, 1.9, -0.1, 1, 2.1, 0, 0.9, 2])   # equal means
    v, lo, hi = max_abs_loa_mover(d, g)
    ba = bland_altman_rm(d, np.zeros_like(d), g)
    assert v == pytest.approx(max(abs(ba["loa_lo"]), abs(ba["loa_hi"])))
    assert lo < v < hi


@pytest.mark.slow
def test_max_abs_loa_interval_coverage_13_clusters(record_property):
    """MOVER vs the percentile cluster bootstrap for max |LoA| with 13 clusters (5-12
    cycles each, between SD 1.0, within SD 0.8, bias 0.5), 300 samples. Measured
    (2026-10-29, seeds as written): MOVER 0.970 (upper-tail misses 0.023), bootstrap 0.850
    (upper-tail misses 0.143; nominal 0.025): the bootstrap upper bound is too low."""
    from biomech.stats.cohort_misc import max_abs_loa_ci

    rng = np.random.default_rng(2026)
    sb, sw, bias = 1.0, 0.8, 0.5
    true = abs(bias) + 1.96 * np.sqrt(sb ** 2 + sw ** 2)
    reps = 300
    hit_m = hit_b = above_m = above_b = 0
    for _ in range(reps):
        m = rng.integers(5, 13, 13)
        g = np.repeat(np.arange(13), m)
        d = bias + rng.normal(0, sb, 13)[g] + rng.normal(0, sw, g.size)
        ref = rng.normal(30, 5, g.size)
        ci = max_abs_loa_ci(ref + d, ref, g, seed=int(rng.integers(1 << 30)), n_boot=1000)
        hit_m += ci["max_abs_loa_ci_lo"] <= true <= ci["max_abs_loa_ci_hi"]
        hit_b += ci["max_abs_loa_boot_ci_lo"] <= true <= ci["max_abs_loa_boot_ci_hi"]
        above_m += true > ci["max_abs_loa_ci_hi"]
        above_b += true > ci["max_abs_loa_boot_ci_hi"]
    cov_m, cov_b = hit_m / reps, hit_b / reps
    record_property("coverage_mover", cov_m)
    record_property("coverage_boot", cov_b)
    print(f"max |LoA|, 13 clusters: MOVER {cov_m:.3f} (upper misses {above_m / reps:.3f}), "
          f"bootstrap {cov_b:.3f} (upper misses {above_b / reps:.3f})")
    assert cov_m >= 0.92 and above_m / reps <= 0.04
    assert above_m < above_b


# -- D27 ------------------------------------------------------------------------------------
def test_requirements_sensitivity_uniform_statistic_and_sensor_bandwidths():
    rng = np.random.default_rng(5)
    rows = []
    for k in range(14):
        p = f"s{k + 1:02d}"
        for c in range(4):
            rows.append({"pid": p, "task": "open_close", "side": "", "ok": c < 3,
                         "opening_mm": 30 + k + c, "theta_y_deg": 15 + c,
                         "peak_open_vel_dps": 60 + c, "protrusion_mm": np.nan,
                         "lateral_mm": np.nan})
        rows.append({"pid": p, "task": "protrusion_retrusion", "side": "", "ok": True,
                     "opening_mm": np.nan, "theta_y_deg": np.nan, "peak_open_vel_dps": np.nan,
                     "protrusion_mm": 8.0 + k, "lateral_mm": np.nan})
        for side, lat in (("left", 10.0 + k), ("right", -(11.0 + k))):
            rows.append({"pid": p, "task": "left_right", "side": side, "ok": True,
                         "opening_mm": np.nan, "theta_y_deg": np.nan,
                         "peak_open_vel_dps": np.nan, "protrusion_mm": np.nan,
                         "lateral_mm": lat})
    cyc = pd.DataFrame(rows)
    bw = pd.DataFrame({"pid": [f"s{k + 1:02d}" for k in range(13)] * 2,
                       "task": ["open_close"] * 13 + ["protrusion_retrusion"] * 13,
                       "noise_limited": [False] * 13 + [True] * 13,
                       "f95_coh_hz": rng.uniform(3, 6, 26), "f95_hz": rng.uniform(5, 9, 26),
                       "f95_opt_hz": rng.uniform(4, 12, 26)})
    prim = pd.DataFrame({"key": ["opening_mm", "bandwidth_hz"], "value": [40.0, 5.0]})
    t = requirements_sensitivity(cyc, bw, prim)
    op = t[(t.key == "opening_mm")].iloc[0]
    largest = np.array([32.0 + k for k in range(14)])                 # ok cycles only
    assert op.variant == "largest_cycle" and op.n == 14 and op.primary_value == 40.0
    assert op.value == pytest.approx(harrell_davis(largest))
    assert op.sample_max == 45.0 and op.lower_k == 12
    lat = t[t.key == "laterotrusion_mm"].iloc[0]
    assert lat.sample_max == 24.0                                     # right side, sign flipped
    assert t[t.key == "opening_angle_deg"].iloc[0].sample_max == 17.0
    b = t[t.key == "bandwidth_hz"].set_index("variant")
    assert set(b.index) == {"imu_only", "optics_only", "min_band_1.5hz", "min_band_3hz"}
    assert (b.n == 13).all()
    use = bw[bw.task == "open_close"]
    assert b.loc["imu_only", "value"] == pytest.approx(harrell_davis(use.f95_hz))
    assert b.loc["optics_only", "sample_max"] == pytest.approx(use.f95_opt_hz.max())
    assert b.loc["optics_only", "primary_value"] == 5.0


def test_movement_bandwidth_reports_the_optics_alone():
    from biomech.imu.analysis import movement_bandwidth

    rng = np.random.default_rng(0)
    T = 60.0
    ti = np.arange(0, T, 1 / 200.0)
    to = np.arange(0, T, 1 / 40.0)
    f0 = 1.5
    th = lambda t: 10 * np.sin(2 * np.pi * f0 * t)                    # noqa: E731
    omega = np.gradient(th(ti), ti) + rng.normal(0, 2, ti.size)
    arrays = {"open_close__omega_dps": omega, "open_close__t_us": (ti * 1e6).astype(np.int64)}
    kin = {"open_close__t_us": (to * 1e6).astype(np.int64),
           "open_close__theta_raw_deg": th(to) + rng.normal(0, 0.05, to.size)}
    r = movement_bandwidth("s01", arrays, 200.0, kin).iloc[0]
    assert np.isfinite(r.f95_opt_hz) and r.f95_opt_hz == pytest.approx(f0, abs=0.75)


def test_bandwidth_min_band_threshold_d29():
    """D29: noise-limited = coherent band below the threshold, NaN counts as noise-limited;
    the default 2 Hz equals the stored flag (primary unchanged)."""
    from biomech.stats.requirements import bandwidth_per_participant

    bw = pd.DataFrame({"pid": ["a", "a", "b", "b", "c"],
                       "task": ["open_close", "chewing", "open_close", "chewing", "open_close"],
                       "coh_band_hz": [1.8, 3.5, 2.5, np.nan, np.nan],
                       "f95_coh_hz": [4.0, 6.0, 5.0, 9.0, 7.0]})
    bw["noise_limited"] = ~(bw.coh_band_hz >= 2.0)
    p2 = bandwidth_per_participant(bw)
    assert p2.to_dict() == {"a": 6.0, "b": 5.0}                       # c: NaN band only
    assert p2.equals(bandwidth_per_participant(bw.drop(columns="coh_band_hz")))
    assert bandwidth_per_participant(bw, min_band_hz=1.5).to_dict() == {"a": 6.0, "b": 5.0}
    assert bandwidth_per_participant(bw, min_band_hz=3.0).to_dict() == {"a": 6.0}
    t = requirements_sensitivity(pd.DataFrame(), bw)
    v = t.set_index("variant")
    assert v.loc["min_band_3hz", "n"] == 1 and v.loc["min_band_1.5hz", "n"] == 2
