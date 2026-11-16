# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result-identical speed-ups of ml_eval / ml_online (bit-identical gates).

* the batched exact SVM prediction (:mod:`ml.models.fast_svm`) gives ``SVC.predict``'s
  labels on random problems and on the folds x grid points x inner splits of a synthetic
  study, and ``loso`` / ``grouped_search`` choose and predict the same with it on or off;
* the shared pool (:mod:`ml.protocols.pool`): results by key, dependencies, longest first,
  cancellation, errors, memory-mapped arrays;
* ``loso_many`` equals the per-fold reference (:func:`ml.protocols.evaluate.run`) for any
  number of workers; ``run_variants`` (late fusion split into its fits, few-shot per k,
  inner searches as tasks) equals the serial pieces;
* the model cache (:mod:`ml.models.cache`): a hit is the same model, every input is in the key;
* the few-shot EMG + IMU vs IMU rows of ``ml_eval.tests``.

All data is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.svm import SVC

from ml.core.dataset import concat
from ml.models import cache as mc
from ml.models import fast_svm
from ml.models import registry as reg
from ml.protocols import evaluate as ev
from ml.protocols import pool as pl
from ml.protocols import variants as va
from ml.protocols.folds import make_folds
from tests.test_variants import _participant


@pytest.fixture(scope="module")
def ds():
    return concat([_participant(f"s0{i}", i, n_run=8) for i in range(1, 5)])


# -- fast SVM prediction ------------------------------------------------------------------
def _random_problem(rng):
    k = int(rng.integers(2, 8))
    d = int(rng.integers(1, 25))
    n = int(rng.integers(k * 3, 300))
    X = rng.normal(size=(n, d))
    y = rng.integers(0, k, n)
    y[:k] = np.arange(k)
    X += rng.normal(scale=float(rng.choice([0.0, 0.5, 2.0])), size=(k, d))[y]
    labels = rng.choice([0, 1, 2, 3, 4, 5, 6, 9, -2], size=k, replace=False)
    svc = SVC(C=float(rng.choice([0.01, 0.1, 1.0, 10.0])),
              gamma=float(rng.choice([0.01, 0.1, 1.0, 5.0])) / d,
              class_weight=rng.choice([None, "balanced"])).fit(X, labels[y])
    Xt = np.vstack([rng.normal(size=(300, d)) * float(rng.choice([0.5, 1.0, 3.0])), X[:40]])
    return svc, Xt


def test_fast_predict_equals_libsvm_on_random_problems(monkeypatch):
    rng = np.random.default_rng(0)
    seen = {"tied": 0}
    for i in range(120):
        svc, Xt = _random_problem(rng)
        monkeypatch.setattr(fast_svm, "CHUNK_ELEMS", int(rng.choice([500, 2_000_000])))
        st: dict[str, int] = {}
        np.testing.assert_array_equal(fast_svm.predict(svc, Xt, stats=st), svc.predict(Xt),
                                      err_msg=f"problem {i}")
        seen["tied"] += st["tied"]
    assert seen["tied"] > 0                      # the tie path is exercised


def test_libsvm_tie_rule_is_reproduced_without_fallback(monkeypatch):
    """Ties need no fallback: the vote's first maximum is libsvm's choice (checked with the
    tie fallback off; ambiguous decision values still go to libsvm)."""
    monkeypatch.setattr(fast_svm, "TIE_FALLBACK", False)
    rng = np.random.default_rng(5)
    for _ in range(60):
        svc, Xt = _random_problem(rng)
        np.testing.assert_array_equal(fast_svm.predict(svc, Xt), svc.predict(Xt))


def test_ambiguous_rows_go_to_libsvm(monkeypatch):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    svc = SVC(gamma=0.5).fit(X, y)
    # rows exactly on the decision boundary have |dec| ~ 0: force them through libsvm
    called = []
    real = svc.predict

    def spy(Z):
        called.append(len(Z))
        return real(Z)

    monkeypatch.setattr(svc, "predict", spy)
    monkeypatch.setattr(fast_svm, "_EPS", 1.0)            # every row "ambiguous"
    Xt = rng.normal(size=(50, 3))
    np.testing.assert_array_equal(fast_svm.predict(svc, Xt), real(Xt))
    assert called == [50]


def test_svm_classifier_uses_libsvm_for_small_batches(monkeypatch, ds):
    X, sl = ds.matrix("emg_imu")
    est = reg.get("svm").make(0, slices=sl).fit(X, ds.y)
    monkeypatch.setattr(fast_svm, "predict", lambda *a, **k: pytest.fail("fast path used"))
    est.predict(X[:fast_svm.MIN_ROWS - 1])                # in-loop decoding size: libsvm


def test_fast_predict_on_every_fold_grid_point_and_inner_split(ds):
    """The gate on the study's own combinations: LOSO folds x the frozen grid x inner splits
    (the ``max_train`` subsample), with ``SvmClassifier`` as the search fits it."""
    for fset in ("emg_imu", "imu"):
        X, sl = ds.matrix(fset)
        for f in make_folds(ds, "loso_subject"):
            idx = ev.subsample(f.train.size, 700, 3)
            Xf, yf, gf = X[f.train][idx], ds.y[f.train][idx], f.groups[idx]
            for tr, te in GroupKFold(3).split(Xf, yf, gf):
                for pt in ParameterGrid(reg.get("svm").grid):
                    est = reg.get("svm").make(3, slices=sl, **pt).fit(Xf[tr], yf[tr],
                                                                       groups=gf[tr])
                    with fast_svm.disabled():
                        ref = est.predict(Xf[te])
                    np.testing.assert_array_equal(est.predict(Xf[te]), ref)
                    np.testing.assert_array_equal(est.predict(X[f.test]),
                                                  est.pipe_.predict(X[f.test]))


def test_search_and_loso_identical_with_and_without_fast_path(ds):
    X, sl = ds.matrix("emg_imu")
    f = make_folds(ds, "loso_subject")[0]
    a = ev.grouped_search("svm", X[f.train], ds.y[f.train], f.groups, reg.get("svm").grid,
                          inner=3, seed=1, slices=sl)
    rows = ev.loso(ds, "emg_imu", "svm", max_train=600, seed=1, importance=2)
    with fast_svm.disabled():
        assert not fast_svm.enabled()
        b = ev.grouped_search("svm", X[f.train], ds.y[f.train], f.groups,
                              reg.get("svm").grid, inner=3, seed=1, slices=sl)
        ref = ev.loso(ds, "emg_imu", "svm", max_train=600, seed=1, importance=2)
    assert fast_svm.enabled()
    assert a == b
    _same_rows(rows, ref)


def _same_rows(a: list[dict], b: list[dict]) -> None:
    assert len(a) == len(b)
    for r, s in zip(a, b):
        assert r.keys() == s.keys()
        for k in r:
            if k == "_pred":
                for q in ("t_end_us", "y", "yhat", "rows"):
                    np.testing.assert_array_equal(r[k][q], s[k][q])
            elif isinstance(r[k], np.ndarray):
                np.testing.assert_array_equal(r[k], s[k])
            elif isinstance(r[k], float) and np.isnan(r[k]):
                assert np.isnan(s[k]), k
            else:
                assert r[k] == s[k], k


# -- pool -----------------------------------------------------------------------------------
def _square(x):
    return x * x


def _boom(x):
    raise ZeroDivisionError("boom")


def _sum_ref(ref, rows):
    return float(ref.get()[rows].sum())


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_pool_results_by_key_with_dependencies(n_jobs):
    order = []

    def on_result(key, r):
        order.append(key)
        if key[0] == "a":
            return [pl.Task(("b", key[1]), _square, (r,), cost=0.5)]
        return None

    tasks = [pl.Task(("a", i), _square, (i,), cost=float(i)) for i in range(6)]
    out = pl.run_tasks(tasks, n_jobs, on_result=on_result)
    assert out == {**{("a", i): i * i for i in range(6)}, **{("b", i): i ** 4 for i in range(6)}}
    if n_jobs == 1:                     # longest first; dependants queue by their own cost
        assert order == [("a", i) for i in (5, 4, 3, 2, 1)] + \
            [("b", i) for i in (5, 4, 3, 2, 1)] + [("a", 0), ("b", 0)]
    else:
        assert sorted(order) == sorted(out)


def test_pool_serial_order_is_longest_first():
    seen = []
    pl.run_tasks([pl.Task(i, _square, (i,), cost=c) for i, c in enumerate([1, 5, 3, 5])], 1,
                 on_result=lambda k, r: seen.append(k))
    assert seen == [1, 3, 2, 0]                         # equal cost: submission order


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_pool_errors_and_cancel(n_jobs):
    with pytest.raises(ZeroDivisionError):
        pl.run_tasks([pl.Task(0, _boom, (1,)), pl.Task(1, _square, (2,))], n_jobs)
    with pytest.raises(RuntimeError, match="cancelled"):
        pl.run_tasks([pl.Task(i, _square, (i,)) for i in range(4)], n_jobs,
                     cancelled=lambda: True)
    with pytest.raises(ValueError, match="duplicate"):
        pl.run_tasks([pl.Task(0, _square, (1,)), pl.Task(0, _square, (2,))], n_jobs)


def test_shared_arrays_are_mapped_read_only_and_removed():
    a = np.arange(1000, dtype=np.float64).reshape(100, 10)
    with pl.Shared() as sh:
        ref = sh.put(a)
        assert sh.put(a) is ref                         # one file per array object
        m = ref.get()
        assert not m.flags.writeable and np.array_equal(m, a) and m.dtype == a.dtype
        rows = np.array([3, 1, 7])
        out = pl.run_tasks([pl.Task(0, _sum_ref, (ref, rows))], 2)
        assert out[0] == float(a[rows].sum())
        d = sh.dir
    import os

    assert not os.path.exists(d)


# -- nested LOSO in one pool ----------------------------------------------------------------
def _reference_rows(ds, model, fset, *, max_train, seed, inner, importance=0):
    """The per-fold reference: run() with one search + refit per fold (old serial path)."""
    res = ev.run(ds, fset, model, "loso_subject", seed=seed, inner=inner, max_train=max_train,
                 importance=importance)
    return ev._loso_rows(ds, fset, model, res, y=None, max_train=max_train, seed=seed)


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_loso_many_equals_the_per_fold_reference(ds, n_jobs):
    plan = [("svm", "emg_imu", 2), ("lda", "imu", 0), ("mlp", "emg", 0), ("svm", "imu", 0)]
    got = ev.loso_many(ds, plan, max_train=500, seed=4, inner=2, n_jobs=n_jobs)
    for (model, fset, imp), rows in zip(plan, got):
        _same_rows(rows, _reference_rows(ds, model, fset, max_train=500, seed=4, inner=2,
                                         importance=imp))
    # the flat-search path of run() chooses the same as well
    flat = ev.run(ds, "imu", "svm", "loso_subject", seed=4, inner=2, max_train=500,
                  n_jobs=2, flat_search=True)
    assert [str({k: v for k, v in r.best_params.items() if k != "inner_macro_f1"})
            for r in flat] == [r["best_params"] for r in got[3]]


def _search_tasks(monkeypatch) -> list:
    """Spy on the pool: the inner-search tasks ``loso_many`` submits."""
    seen: list = []
    real = pl.run_tasks

    def spy(tasks, *a, **k):
        tasks = list(tasks)
        seen.extend(t for t in tasks if t.key[0] == "search")
        return real(tasks, *a, **k)

    monkeypatch.setattr(pl, "run_tasks", spy)
    return seen


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_loso_many_fits_shared_inner_splits_once(ds, monkeypatch, n_jobs):
    """Outer folds whose inner splits train on the very same rows (equal-sized synthetic
    subjects make that common) share one fit per grid point, scored on each fold's inner
    test rows; the rows equal the per-fold reference, where every split is its own fit."""
    seen = _search_tasks(monkeypatch)
    got = ev.loso_many(ds, [("svm", "imu", 0), ("mlp", "emg", 0)], max_train=None, seed=4,
                       inner=2, n_jobs=n_jobs)
    n_folds = len(make_folds(ds, "loso_subject"))
    for model, rows in zip(("svm", "mlp"), got):
        tasks = [t for t in seen if t.args[4] == model]
        n_points = len(ParameterGrid(reg.get(model).grid))
        assert sum(len(t.args[-1]) for t in tasks) == n_folds * n_points * 2
        assert len(tasks) < n_folds * n_points * 2                  # some fits are shared
    _same_rows(got[0], _reference_rows(ds, "svm", "imu", max_train=None, seed=4, inner=2))
    _same_rows(got[1], _reference_rows(ds, "mlp", "emg", max_train=None, seed=4, inner=2))


def test_loso_many_with_label_override_and_cancel(ds):
    y = ev.shift_labels(ds, np.random.default_rng(0))
    a = ev.loso_many(ds, [("lda", "emg_imu", 0)], max_train=None, seed=1, y=y)[0]
    b = ev.run(ds, "emg_imu", "lda", "loso_subject", seed=1, max_train=None, y=y)
    assert [r["balanced_accuracy"] for r in a] == [ev.fold_scores(ds, r, y)["balanced_accuracy"]
                                                   for r in b]
    with pytest.raises(RuntimeError, match="cancelled"):
        ev.loso_many(ds, [("svm", "imu", 0)], max_train=300, cancelled=lambda: True)


# -- model cache ---------------------------------------------------------------------------
def test_model_cache_hit_is_the_same_model(ds, tmp_path):
    X, sl = ds.matrix("emg_imu")
    f = make_folds(ds, "loso_subject")[1]
    a, info_a = ev.fit("svm", X[f.train], ds.y[f.train], f.groups, seed=3, max_train=500,
                       search=False, params={"C": 1.0, "gamma_mult": 4.0}, slices=sl,
                       cache_dir=str(tmp_path))
    assert not a.fit_cache_hit_ and len(list(tmp_path.glob("*.joblib"))) == 1
    b, info_b = ev.fit("svm", X[f.train], ds.y[f.train], f.groups, seed=3, max_train=500,
                       search=False, params={"C": 1.0, "gamma_mult": 4.0}, slices=sl,
                       cache_dir=str(tmp_path))
    assert b.fit_cache_hit_ and b.fit_time_s_ == a.fit_time_s_ and info_a == info_b
    with fast_svm.disabled():
        np.testing.assert_array_equal(a.predict(X[f.test]), b.predict(X[f.test]))
    np.testing.assert_array_equal(
        a.pipe_.named_steps["clf"].dual_coef_, b.pipe_.named_steps["clf"].dual_coef_)
    # anything that determines the fit changes the key
    est = reg.get("svm").make(3, slices=sl, C=1.0, gamma_mult=4.0)
    Xf, yf, gf = X[:200], ds.y[:200], ds.groups[:200]
    k0 = mc.fit_key(est, Xf, yf, gf)
    assert mc.fit_key(est, Xf, yf, gf) == k0
    Xd = Xf.copy()
    Xd[0, 0] = np.nextafter(Xd[0, 0], np.inf)
    keys = {mc.fit_key(reg.get("svm").make(3, slices=sl, C=10.0, gamma_mult=4.0), Xf, yf, gf),
            mc.fit_key(reg.get("svm").make(4, slices=sl, C=1.0, gamma_mult=4.0), Xf, yf, gf),
            mc.fit_key(reg.get("svm").make(3, slices=None, C=1.0, gamma_mult=4.0), Xf, yf, gf),
            mc.fit_key(reg.get("svm").make(3, slices=sl, C=1.0, gamma_mult=4.0,
                                           probability=True), Xf, yf, gf),
            mc.fit_key(est, Xd, yf, gf), mc.fit_key(est, Xf, yf[::-1].copy(), gf),
            mc.fit_key(est, Xf, yf, gf[::-1].copy()), mc.fit_key(est, Xf, yf, gf, np.ones(200)),
            mc.fit_key(est, Xf.astype(np.float32), yf, gf)}
    assert len(keys) == 9 and k0 not in keys


def test_model_cache_ignores_broken_entries(ds, tmp_path):
    X, sl = ds.matrix("imu")
    est = reg.get("lda").make(0, slices=sl)
    key = mc.fit_key(est, X, ds.y, ds.groups)
    (tmp_path / f"{key}.joblib").write_bytes(b"not a pickle")
    m, _, hit = mc.fit(est, X, ds.y, groups=ds.groups, cache_dir=tmp_path)
    assert not hit and m.predict(X[:5]).shape == (5,)
    m2, _, hit2 = mc.fit(reg.get("lda").make(0, slices=sl), X, ds.y, groups=ds.groups,
                         cache_dir=tmp_path)
    assert hit2


def test_loso_many_fills_the_cache_ml_online_reads(ds, tmp_path):
    from ml.online import decoder as dc

    rows = ev.loso_many(ds, [("svm", "emg_imu", 0), ("svm", "imu", 0)], max_train=400, seed=5,
                        inner=2, cache_dir=str(tmp_path), cache_models={("svm", "emg_imu")})
    assert len(list(tmp_path.glob("*.joblib"))) == 4        # one per fold, emg_imu only
    main = va.main_params(rows[0])
    X, sl = ds.matrix("emg_imu")
    for f in make_folds(ds, "loso_subject"):
        p = main["emg_imu"][f.unit]
        cached, _ = dc.fit_fold(X[f.train], ds.y[f.train], f.groups, model="svm", params=p,
                                slices=sl, seed=5, max_train=400, inner=2,
                                cache_dir=str(tmp_path))
        fresh, _ = dc.fit_fold(X[f.train], ds.y[f.train], f.groups, model="svm", params=p,
                               slices=sl, seed=5, max_train=400, inner=2)
        assert cached.fit_cache_hit_ and not fresh.fit_cache_hit_
        np.testing.assert_array_equal(cached.predict(X), fresh.predict(X))
        yhat = next(r for r in rows[0] if r["pid"] == f.unit)["_pred"]["yhat"]
        np.testing.assert_array_equal(cached.predict(X[f.test]), yhat)


# -- variants ----------------------------------------------------------------------------------
def _variants_cfg(hyper="main"):
    return va.config({"hyper": hyper, "late_fusion": {"w_step": 0.25},
                      "reduced_emg": {"feature_set": "emg_imu", "features": ["log_rms", "mnf"]},
                      "few_shot": {"feature_sets": ["imu", "emg_imu"], "k": [0, 5, 10]}})


@pytest.mark.parametrize("hyper", ["main", "search"])
def test_variants_pool_equals_serial_pieces(ds, hyper):
    rows = []
    for fs in ("emg", "imu", "emg_imu"):
        rows += ev.loso(ds, fs, "svm", max_train=400, seed=2, inner=2)
    main, pred = va.main_params(rows), va.main_predictions(rows)
    cfg = _variants_cfg(hyper)
    a = va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=400, n_jobs=1, main_pred=pred)
    b = va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=400, n_jobs=3, main_pred=pred)
    cols = [c for c in a.columns if c != "fit_time_s"]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)
    assert set(a.variant) == {"late_fusion", "reduced_emg", "few_shot"}
    # late fusion: the pieces assemble to the classifier's own fit
    X, sl = ds.matrix("emg+imu")
    for f in make_folds(ds, "loso_subject"):
        r = a[(a.variant == "late_fusion") & (a.pid == f.unit)].iloc[0]
        bp = eval(r["params"])                                     # noqa: S307 - own repr
        idx = ev.subsample(f.train.size, 400, 2)
        m = va.late_fusion_model(X[f.train][idx], ds.y[f.train][idx], f.groups[idx], sl, bp,
                                 w_step=0.25, inner=2, seed=2)
        yhat = m.predict(X[f.test])
        assert r["balanced_accuracy"] == va._score(ds.y[f.test], yhat)["balanced_accuracy"]
        assert r["w_emg"] == float(m.weights_[0])
        if hyper == "main":
            assert bp == [main["emg"][f.unit], main["imu"][f.unit]]
        # few-shot: the per-k pool tasks equal the serial few_shot_predictions
        for fs in ("imu", "emg_imu"):
            Xs, sls = ds.matrix(fs)
            q = pred[fs][f.unit]
            p = main[fs][f.unit] if hyper == "main" else eval(
                a[(a.variant == "few_shot") & (a.pid == f.unit) & (a.feature_set == fs)]
                .params.iloc[0])                                   # noqa: S307
            ser = va.few_shot_predictions(ds, f, Xs, sls, p, ks=[0, 5, 10], guard_s=0.35,
                                          weight="balanced", seed=2, max_train=400,
                                          k0_pred=(q["rows"], q["yhat"]) if hyper == "main"
                                          else None)
            got = a[(a.variant == "few_shot") & (a.pid == f.unit) & (a.feature_set == fs)]
            assert [va._score(ds.y[s["test"]], s["y_pred"])["balanced_accuracy"]
                    for s in ser] == got.balanced_accuracy.tolist()


# -- few-shot test rows ------------------------------------------------------------------------
def _view_reference(v: pd.DataFrame) -> dict[int, dict]:
    """The computation of the paper view (paper_journal._few_shot) restated here."""
    from scipy.stats import wilcoxon

    from biomech.stats.core import holm

    f = v[v.variant == "few_shot"]
    w = f.pivot_table(index="pid", columns=["feature_set", "k"], values="balanced_accuracy")
    out = {}
    for k in sorted({int(c[1]) for c in w.columns}):
        d = (w[("emg_imu", k)] - w[("imu", k)]).dropna().to_numpy()
        p = float(wilcoxon(d).pvalue) if d.size >= 5 and np.any(d != 0) else np.nan
        out[k] = {"d": d, "p": p}
    for k, ph in zip(out, holm([o["p"] for o in out.values()])):
        out[k]["p_holm"] = ph
    return out


def test_few_shot_tests_rows_match_the_view():
    pytest.importorskip("biomech")
    rng = np.random.default_rng(3)
    rows = []
    for pid in [f"s{i:02d}" for i in range(1, 13)]:
        for k in (0, 20, 50, 100):
            base = rng.uniform(0.3, 0.6)
            rows.append({"variant": "few_shot", "feature_set": "imu", "k": k, "pid": pid,
                         "balanced_accuracy": base})
            rows.append({"variant": "few_shot", "feature_set": "emg_imu", "k": k, "pid": pid,
                         "balanced_accuracy": base + rng.normal(0.02 * (k > 0), 0.03)})
    rows.append({"variant": "late_fusion", "feature_set": "emg+imu", "k": 0, "pid": "s01",
                 "balanced_accuracy": 0.5})
    v = pd.DataFrame(rows)
    got = va.few_shot_tests(v)
    ref = _view_reference(v)
    assert [r["test"] for r in got] == [f"fewshot|svm|emg_imu_vs_imu|k={k}"
                                        for k in (0, 20, 50, 100)]
    for r, (k, o) in zip(got, ref.items()):
        assert r["p"] == o["p"] and r["p_holm"] == o["p_holm"]
        assert r["stat"] == float(np.median(o["d"])) and r["n"] == 12
        assert r["n_better"] == int((o["d"] > 0).sum()) and r["n_amounts"] == 4
    assert va.few_shot_tests(pd.DataFrame()) == []
    small = va.few_shot_tests(v[v.pid.isin(["s01", "s02", "s03"])])
    assert all(np.isnan(r["p"]) and np.isnan(r["p_holm"]) and r["n_amounts"] == 0
               for r in small)


# -- blocked CV in one pool -----------------------------------------------------------------
def _old_blocked(ds, fset, model, *, k, seed, params_by_pid=None):
    """The pre-2026-10-31 ``blocked`` (frozen): one ``run`` per configuration."""
    res = ev.run(ds, fset, model, "blocked", seed=seed, search=False, k=k,
                 fold_params=params_by_pid)
    out = []
    for pid in sorted({r.fold.pid for r in res}):
        rs = [r for r in res if r.fold.pid == pid]
        s = ev.scores(np.concatenate([ds.y[r.fold.test] for r in rs]),
                      np.concatenate([r.y_pred for r in rs]))
        chosen = params_by_pid is not None and pid in params_by_pid
        out.append({"pid": pid, "feature_set": fset, "model": model,
                    "balanced_accuracy": s["balanced_accuracy"], "macro_f1": s["macro_f1"],
                    "n": s["n"], "hyper": "loso" if chosen else "default",
                    "best_params": str(params_by_pid[pid]) if chosen else "{}"})
    return out


@pytest.mark.parametrize("n_jobs", [1, 3])
def test_blocked_many_equals_separate_runs_and_shares_fold_fits(ds, monkeypatch, n_jobs):
    pids = sorted(set(ds.pid))
    svm_choice = {pids[0]: {"C": 0.1, "gamma_mult": 1.0}, pids[1]: {"C": 10.0}}
    lda_choice = {p: {} for p in pids}                  # no grid: the LOSO choice is {}
    specs = [("imu", "svm", svm_choice), ("imu", "svm", None),
             ("emg_imu", "lda", lda_choice), ("emg_imu", "lda", None)]
    seen: list = []
    real = pl.run_tasks

    def spy(tasks, *a, **k):
        tasks = list(tasks)
        seen.extend(tasks)
        return real(tasks, *a, **k)

    monkeypatch.setattr(pl, "run_tasks", spy)
    got = ev.blocked_many(ds, specs, k=3, seed=1, n_jobs=n_jobs)
    monkeypatch.setattr(pl, "run_tasks", real)
    for (fset, model, by_pid), rows in zip(specs, got):
        assert rows == _old_blocked(ds, fset, model, k=3, seed=1, params_by_pid=by_pid)
    n_folds = len(make_folds(ds, "blocked", k=3))
    n_svm_chosen = sum(1 for f in make_folds(ds, "blocked", k=3) if f.pid in svm_choice)
    # svm: the default folds + the chosen folds of two subjects; lda: one fit per fold
    assert len(seen) == n_folds + n_svm_chosen + n_folds
