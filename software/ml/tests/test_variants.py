# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Exploratory decoding variants (ml.protocols.variants): late fusion, reduced EMG, few-shot.

All data is synthetic: per participant two phases in time order, classes in runs of
10 windows (stride 50 ms); EMG and IMU depend on the class, EMG also carries a
participant-specific offset (poor transfer across people, as in the pilot).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.core.dataset import WindowDataset, concat
from ml.core.windows import label_runs
from ml.models import registry as reg
from ml.protocols import evaluate as ev
from ml.protocols import variants as va
from ml.protocols.folds import make_folds

CH = ("masseter_r", "digastric_l")
EMG_F = ("log_mav", "log_rms", "mnf", "zc")
STRIDE_US = 50_000
P = {"C": 1.0, "gamma_mult": 1.0}


def _participant(pid: str, seed: int, n_run: int = 6) -> WindowDataset:
    rng = np.random.default_rng(seed)
    y, phase = [], []
    for name, cyc in (("open_close", (0, 1, 0, 2)), ("left_right", (0, 5, 0, 6))):
        lab = np.repeat(np.tile(cyc, n_run // 2), 10)
        y.append(lab)
        phase.append(np.full(lab.size, name))
    y, phase = np.concatenate(y), np.concatenate(phase)
    n = y.size
    t = 1_000_000 + np.arange(n, dtype=np.int64) * STRIDE_US
    t[phase == "left_right"] += 3_000_000                    # pause between the phases
    centres = rng.normal(size=(7, len(CH) * len(EMG_F)))
    emg = centres[y] * 1.5 + rng.normal(size=(n, len(CH) * len(EMG_F))) \
        + rng.normal(scale=2.0, size=len(CH) * len(EMG_F))  # participant offset
    imu_c = np.random.default_rng(99).normal(size=(7, 4)) * 2.0
    imu = imu_c[y] + rng.normal(size=(n, 4))
    return WindowDataset(
        blocks={"emg": emg, "imu": imu}, y=y, pid=np.full(n, pid), session=np.full(n, pid),
        phase=phase, t_end_us=t, groups=label_runs(y, phase),
        names={"emg": [f"{c}|{f}" for c in CH for f in EMG_F],
               "imu": [f"i{j}" for j in range(4)]})


@pytest.fixture(scope="module")
def ds() -> WindowDataset:
    return concat([_participant(f"s0{i}", i) for i in range(1, 5)])


def _main(ds: WindowDataset) -> dict:
    return {fs: {p: dict(P) for p in np.unique(ds.pid)} for fs in ("emg", "imu", "emg_imu")}


def test_config_defaults_and_switches():
    c = va.config({})
    assert c["hyper"] == "main" and c["few_shot"]["k"] == [0, 20, 50, 100]
    assert c["late_fusion"]["w_step"] == 0.25 and c["reduced_emg"]["features"] == ["log_rms", "mnf"]
    assert va.config(None) is None and va.config(False) is None
    assert va.config({"late_fusion": None})["late_fusion"] is None
    with pytest.raises(ValueError, match="unknown"):
        va.config({"few_shots": {}})
    with pytest.raises(ValueError, match="unknown"):
        va.config({"few_shot": {"stratified": True}})


def test_main_params_from_loso_rows():
    rows = [{"pid": "s01", "feature_set": "imu", "model": "svm",
             "best_params": str({"C": 0.1, "gamma_mult": 4.0})},
            {"pid": "s01", "feature_set": "imu", "model": "lda", "best_params": "{}"}]
    assert va.main_params(rows) == {"imu": {"s01": {"C": 0.1, "gamma_mult": 4.0}}}


def test_every_variant_runs_and_gives_finite_ba(ds):
    cfg = va.config({"few_shot": {"k": [0, 10, 20]}})
    df = va.run_variants(ds, cfg, _main(ds), seed=0, inner=2, max_train=500)
    assert set(df.variant) == {"late_fusion", "reduced_emg", "few_shot"}
    assert np.isfinite(df.balanced_accuracy).all() and np.isfinite(df.macro_f1).all()
    for v, g in df.groupby("variant"):
        assert set(g.pid) == set(np.unique(ds.pid)), v
    fs = df[df.variant == "few_shot"]
    assert set(fs.feature_set) == {"imu", "emg_imu"} and set(fs.k) == {0, 10, 20}
    assert (fs.groupby(["feature_set", "pid"]).n_test.nunique() == 1).all()   # common eval set
    lf = df[df.variant == "late_fusion"]
    assert lf.w_emg.isin([0.0, 0.25, 0.5, 0.75, 1.0]).all()
    assert (df.hyper.str.startswith("main_loso")).all()
    red = df[df.variant == "reduced_emg"]
    assert (red.n_emg_features == len(CH) * 2).all()
    s = va.summarize_variants(df)
    assert {"median", "q1", "q3", "iqr", "n"} <= set(s.columns)
    ba = s[(s.metric == "balanced_accuracy")]
    assert (ba.n == 4).all() and np.isfinite(ba["median"]).all()
    assert {"ba_gain_vs_k0", "ba_emg_imu_minus_imu"} <= set(s.metric)
    # Without a main-run choice a fold gets one inner search.
    one = va.run_variants(ds, va.config({"late_fusion": None, "few_shot": None}), {}, seed=0,
                          inner=2, max_train=300)
    assert (one.hyper == "inner_search").all() and np.isfinite(one.balanced_accuracy).all()


def test_few_shot_k0_equals_plain_loso(ds):
    X, sl = ds.matrix("emg_imu")
    plain = {r.fold.unit: r for r in ev.run(ds, "emg_imu", "svm", "loso_subject", seed=3,
                                            search=False, params=P, max_train=400)}
    for f in make_folds(ds, "loso_subject"):
        res = va.few_shot_predictions(ds, f, X, sl, P, ks=[0, 20, 50], guard_s=0.35,
                                      weight="balanced", seed=3, max_train=400)
        r0 = res[0]
        assert r0["k"] == 0 and r0["cal"].size == 0 and r0["w_user"] == 0.0
        pos = np.searchsorted(f.test, r0["test"])
        assert np.array_equal(f.test[pos], r0["test"])
        np.testing.assert_array_equal(r0["y_pred"], plain[f.unit].y_pred[pos])


def test_few_shot_first_mode_uses_the_recording_start(ds):
    ks, guard = [0, 20, 50, 100], 0.35
    for f in make_folds(ds, "loso_subject"):
        cals, test = va.few_shot_split(ds, f.test, ks, guard, mode="first")
        order = f.test[np.lexsort((ds.t_end_us[f.test], ds.session[f.test]))]
        for k in ks:
            assert np.array_equal(cals[k], order[:k])
        gap = ds.t_end_us[test].min() - ds.t_end_us[order[max(ks) - 1]]
        assert gap > guard * 1e6


def test_few_shot_never_trains_on_evaluated_windows(ds):
    """per_phase (default): the first k windows of every phase; evaluation windows of each
    phase start > guard after its K-th window; calibration and evaluation never overlap."""
    X, sl = ds.matrix("imu")
    ks, guard = [0, 20, 50, 100], 0.35
    for f in make_folds(ds, "loso_subject"):
        cals, test = va.few_shot_split(ds, f.test, ks, guard)
        assert not np.isin(ds.pid[f.train], [f.unit]).any()
        phases = np.unique(ds.phase[f.test])
        for k in ks:
            cal = cals[k]
            assert not np.isin(cal, test).any() and (ds.pid[cal] == f.unit).all()
            for ph in phases:                                # k windows of every phase
                inph = f.test[ds.phase[f.test] == ph]
                n = min(k, inph.size)
                assert (ds.phase[cal] == ph).sum() == n
                first = inph[np.argsort(ds.t_end_us[inph])][:n]
                assert set(first) == set(cal[ds.phase[cal] == ph])
        for ph in phases:                                    # guard after the K-th window
            inph = np.sort(ds.t_end_us[f.test[ds.phase[f.test] == ph]])
            tp = ds.t_end_us[test[ds.phase[test] == ph]]
            if inph.size > max(ks) and tp.size:
                assert tp.min() - inph[max(ks) - 1] > guard * 1e6
        res = va.few_shot_predictions(ds, f, X, sl, P, ks=ks, guard_s=guard,
                                      weight="balanced", seed=0, max_train=None)
        for r in res:
            assert np.array_equal(np.sort(r["cal"]), np.sort(cals[r["k"]]))
            assert not np.isin(r["cal"], r["test"]).any()
            if r["cal"].size:
                assert np.isclose(r["w_user"], f.train.size / (np.unique(ds.y[f.train]).size
                                                               * r["cal"].size))
            np.testing.assert_array_equal(r["test"], test)


def test_svm_sample_weight_matters():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    Xn, yn = np.array([[0.3, 0.0]] * 5), np.zeros(5, int)            # 5 "user" windows
    Xa, ya = np.vstack([X, Xn]), np.r_[y, yn]
    plain = reg.get("svm").make(0, C=1.0).fit(Xa, ya)
    heavy = reg.get("svm").make(0, C=1.0).fit(Xa, ya, sample_weight=np.r_[np.ones(200),
                                                                      np.full(5, 100.0)])
    assert plain.predict(Xn).mean() > 0.5 and (heavy.predict(Xn) == 0).all()


def test_late_fusion_w0_equals_imu_model_posterior(ds):
    f = make_folds(ds, "loso_subject")[0]
    X, sl = ds.matrix("emg+imu")
    p_emg, p_imu = {"C": 1.0, "gamma_mult": 1.0}, {"C": 10.0, "gamma_mult": 0.25}
    tr, te = f.train, f.test
    m = va.late_fusion_model(X[tr], ds.y[tr], f.groups, sl, [p_emg, p_imu], w_step=0.25,
                             inner=2, seed=1)
    assert len(m.weight_scores_) == 5 and m.weights_[0] in (0.0, 0.25, 0.5, 0.75, 1.0)
    m.weights_ = np.array([0.0, 1.0])
    imu = reg.get("svm").make(1, probability=True, **p_imu).fit(X[tr][:, sl[1]], ds.y[tr],
                                                                groups=f.groups)
    np.testing.assert_allclose(m.predict_proba(X[te]), imu.predict_proba(X[te][:, sl[1]]))
    m.weights_ = np.array([1.0, 0.0])
    emg = reg.get("svm").make(1, probability=True, **p_emg).fit(X[tr][:, sl[0]], ds.y[tr],
                                                                groups=f.groups)
    np.testing.assert_allclose(m.predict_proba(X[te]), emg.predict_proba(X[te][:, sl[0]]))


def test_summary_iqr_over_participants():
    df = pd.DataFrame({"variant": "few_shot", "feature_set": "imu", "k": [0] * 4 + [20] * 4,
                       "pid": ["a", "b", "c", "d"] * 2,
                       "balanced_accuracy": [.4, .5, .6, .7, .5, .6, .7, .8],
                       "macro_f1": 0.5, "fit_time_s": 1.0})
    s = va.summarize_variants(df).set_index(["k", "metric"])
    r = s.loc[(0, "balanced_accuracy")]
    assert np.isclose(r["median"], 0.55) and np.isclose(r.iqr, np.subtract(*np.percentile(
        [.4, .5, .6, .7], [75, 25])))
    assert np.isclose(s.loc[(20, "ba_gain_vs_k0"), "median"], 0.1)


def test_resolve_jobs():
    assert ev.resolve_jobs(0) >= 1 and ev.resolve_jobs("auto") >= 1 and ev.resolve_jobs(3) == 3


def test_flat_parallel_search_equals_serial(ds):
    """run(n_jobs > 1) searches all folds' grids in one pool: same choice, same predictions."""
    grid = {"C": [0.1, 1.0, 10.0], "gamma_mult": [0.25, 1.0]}
    a = ev.run(ds, "emg_imu", "svm", "loso_subject", seed=5, inner=2, max_train=300,
               grid=grid, n_jobs=1)
    b = ev.run(ds, "emg_imu", "svm", "loso_subject", seed=5, inner=2, max_train=300,
               grid=grid, n_jobs=3, flat_search=True)
    assert [r.fold.unit for r in a] == [r.fold.unit for r in b]
    for ra, rb in zip(a, b):
        assert ra.best_params == pytest.approx(rb.best_params)
        np.testing.assert_array_equal(ra.y_pred, rb.y_pred)


def test_muscle_ablation_leaves_out_one_group_and_pairs_with_the_main_run(ds):
    """X10: off unless configured; per group the EMG columns of its muscles are left out
    (IMU kept), the fold's main parameters reused; ba_minus_full pairs per participant."""
    assert va.config({})["muscle_ablation"] is None
    cfg = va.config({"late_fusion": None, "reduced_emg": None, "few_shot": None,
                     "muscle_ablation": {"drop": {"masseter": ["masseter_r"],
                                                  "digastric_l": ["digastric_l"]}}})
    df = va.run_variants(ds, cfg, _main(ds), seed=0, inner=2, max_train=400)
    assert set(df.variant) == {"muscle_ablation"}
    assert set(df.feature_set) == {"emg-masseter+imu", "emg-digastric_l+imu"}
    assert (df.n_emg_features == len(EMG_F)).all()
    assert set(df.dropped) == {"masseter_r", "digastric_l"}
    assert (df.hyper.str.startswith("main_loso")).all() and np.isfinite(df.balanced_accuracy).all()
    X, sl, names = va.ablation_matrix(ds, ["masseter_r"])
    assert all(n.startswith("digastric_l|") for n in names)
    assert X.shape[1] == len(EMG_F) + 4 and sl[1] == slice(len(EMG_F), len(EMG_F) + 4)
    with pytest.raises(ValueError, match="no EMG columns"):
        va.ablation_matrix(ds, ["temporalis_r"])
    with pytest.raises(ValueError, match="leaves no EMG"):
        va.ablation_matrix(ds, list(CH))
    with pytest.raises(ValueError, match="at least one muscle"):
        va.config({"muscle_ablation": {"drop": {"masseter": []}}})
    pids = sorted(set(df.pid))
    loso = pd.DataFrame({"pid": pids, "model": "svm", "feature_set": "emg_imu",
                         "balanced_accuracy": 1.0})
    d = va.ablation_deltas(df, loso).set_index("feature_set")
    a = df[df.feature_set == "emg-masseter+imu"].balanced_accuracy.to_numpy() - 1.0
    assert d.loc["emg-masseter+imu", "median"] == pytest.approx(np.median(a))
    assert d.loc["emg-masseter+imu", "n"] == len(pids)
    assert d.loc["emg-masseter+imu", "n_lower"] == int((a < 0).sum())
    assert (d.metric == "ba_minus_full").all()
