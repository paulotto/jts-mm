# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ML-side review fixes (E20, A5, B2, B3): permutation null fitted like the observed run,
onset latency within segments, ml_online uses ml_eval's subsample, few-shot k = 0 reuse,
parallel blocked CV, matrix cache, one filter pass for windows and reference.

All data is synthetic. Equality tests compare the fast path with the reference path.
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.config.classes import PHASES
from ml.core import signals as S
from ml.core.dataset import WindowConfig, build_windows, concat, filter_phases
from ml.models import registry as reg
from ml.protocols import evaluate as ev
from ml.protocols import variants as va
from tests.test_variants import _participant

FS = 2000.0
CH = ["masseter_r", "digastric_l"]


@pytest.fixture(scope="module")
def ds():
    return concat([_participant(f"s0{i}", i) for i in range(1, 5)])


# -- E20: permutation test --------------------------------------------------------------
def test_permutation_null_uses_the_observed_inner_and_max_train(ds, monkeypatch):
    seen = []
    real = ev.run

    def spy(*a, **kw):
        seen.append((kw["inner"], kw["max_train"]))
        return real(*a, **kw)

    monkeypatch.setattr(ev, "run", spy)
    null = ev.permutation(ds, "imu", "lda", 2, inner=2, max_train=123, seed=0, n_jobs=1)
    assert null.shape == (2,) and np.isfinite(null).all()
    assert seen == [(2, 123), (2, 123)]
    with pytest.raises(TypeError):
        ev.permutation(ds, "imu", "lda", 1)                    # inner / max_train required


def test_permutation_batches_give_the_same_null_and_cancel(ds):
    a = ev.permutation(ds, "imu", "lda", 3, inner=2, max_train=None, seed=4, n_jobs=1)
    b = ev.permutation(ds, "imu", "lda", 3, inner=2, max_train=None, seed=4, n_jobs=1,
                       cancelled=lambda: False)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(RuntimeError, match="cancelled"):
        ev.permutation(ds, "imu", "lda", 3, inner=2, max_train=None, n_jobs=1,
                       cancelled=lambda: True)


def _stage_ctx(ds, configs, n_perm=2, cancelled=None):
    """A cohort context for ml_eval_stage over the synthetic participants."""
    pids = sorted(set(ds.pid))
    feats = pd.DataFrame([{"unit": p, "block": b, "feature": n} for p in pids
                          for b in ("emg", "imu") for n in ds.names[b]])
    arrays = {}
    for p in pids:
        sel = ds.pid == p
        arrays[p] = {"X_emg": ds.blocks["emg"][sel].astype(np.float32),
                     "X_imu": ds.blocks["imu"][sel].astype(np.float32),
                     "y": ds.y[sel].astype(np.int8), "t_end_us": ds.t_end_us[sel],
                     "phase": np.array([PHASES.index(x) for x in ds.phase[sel]], np.int8)}
    params = {"configs": configs, "blocked": [], "primary_model": "svm", "inner_folds": 2,
              "max_train": 300, "n_perm": n_perm, "n_jobs": 1, "variants": None,
              "diagnostics": {"fusion_variance": False, "importance": None}}
    ctx = SimpleNamespace(
        params=params, config=SimpleNamespace(seed=0),
        participants=[SimpleNamespace(pid=p) for p in pids],
        upstream=lambda st, t: feats, upstream_arrays=lambda st, pid: arrays[pid],
        demographics=lambda: pd.DataFrame({"pid": pids, "sex": "f", "age": 30}))
    if cancelled is not None:
        ctx.cancelled = cancelled
    return ctx


def test_ml_eval_skips_the_permutation_test_without_observed_rows(ds, monkeypatch):
    pytest.importorskip("biomech")
    from ml.biomech_stages import ml_eval_stage

    called = []
    monkeypatch.setattr(ev, "permutation", lambda *a, **k: called.append(1) or np.zeros(2))
    res = ml_eval_stage(_stage_ctx(ds, {"svm": ["imu"]}))       # no lda|emg_imu rows
    assert not called
    t = res.tables["tests"]
    assert t.empty or not t.test.str.startswith("permutation").any()
    assert "permutation_null" not in res.arrays


def test_ml_eval_permutation_row_and_cancel(ds, monkeypatch):
    pytest.importorskip("biomech")
    from ml.biomech_stages import ml_eval_stage

    got = {}

    def fake(d, fset, model, n_perm, **kw):
        got.update(kw)
        return np.full(n_perm, 0.2)

    monkeypatch.setattr(ev, "permutation", fake)
    res = ml_eval_stage(_stage_ctx(ds, {"lda": ["emg_imu"]}))
    row = res.tables["tests"].set_index("test").loc["permutation|lda|emg_imu"]
    assert row.p == pytest.approx(1 / 3) and got["inner"] == 2 and got["max_train"] == 300
    with pytest.raises(RuntimeError, match="cancelled"):
        ml_eval_stage(_stage_ctx(ds, {"lda": ["emg_imu"]}, cancelled=lambda: True))


# -- latency within segments -------------------------------------------------------------
def test_latency_respects_phase_boundaries_and_window_gaps():
    stride = 50_000
    # phase A: REST x4, class 1 x4; phase B (directly adjacent rows, 3 s later): class 2 x4
    t = np.r_[np.arange(8) * stride, 3_000_000 + np.arange(4) * stride]
    y = np.r_[[0] * 4, [1] * 4, [2] * 4]
    p = y.copy()
    seg = np.r_[["A"] * 8, ["B"] * 4]
    assert ev.latency(t, y, p, segment=seg) == [0.0]             # the REST -> 1 onset
    # A REST row closing phase A makes the first row of phase B look like an onset by
    # adjacency; it is not one (phase boundary, and a 3 s gap).
    y2 = np.r_[[0] * 4, [1] * 3, [0], [2] * 4]
    p2 = y2.copy()
    assert len(ev.latency(t, y2, p2)) == 1                      # the gap alone cuts it
    assert len(ev.latency(t, y2, p2, segment=seg)) == 1
    t_same = np.r_[np.arange(8) * stride, 8 * stride + np.arange(4) * stride]
    assert len(ev.latency(t_same, y2, p2)) == 2                 # no gap, no key: adjacent
    assert len(ev.latency(t_same, y2, p2, segment=seg)) == 1    # the phase key cuts it
    # Dropped windows: REST, then 5 missing windows, then class 1: no onset (unknown time);
    # the confirmation run never bridges the gap either.
    t3 = np.r_[np.arange(4) * stride, 9 * stride + np.arange(4) * stride]
    y3 = np.r_[[0] * 4, [1] * 4]
    assert ev.latency(t3, y3, y3) == []
    # Onset inside a segment; prediction confirmed only across a gap -> not detected.
    t4 = np.r_[np.arange(6) * stride, 20 * stride + np.arange(3) * stride]
    y4 = np.r_[[0] * 3, [1] * 3, [1] * 3]
    p4 = np.r_[[0] * 5, [1], [1] * 3]
    lat = ev.latency(t4, y4, p4)
    assert len(lat) == 1 and np.isnan(lat[0])
    # The search start never reaches back into the previous segment.
    t5 = np.r_[np.arange(3) * stride, 10 * stride + np.arange(6) * stride]
    y5 = np.r_[[1] * 3, [0] * 2, [1] * 4]
    p5 = np.r_[[1] * 3, [0] * 2, [0, 1, 1, 1]]
    assert ev.latency(t5, y5, p5, pre_s=1.0) == [50.0]


def test_latency_unchanged_on_a_clean_sequence():
    rng = np.random.default_rng(0)
    t = np.arange(400) * 50_000
    y = np.repeat(rng.choice([0, 1, 0, 2], 40), 10)
    p = np.where(rng.random(400) < 0.8, y, 0)
    seg = np.zeros(400, int)
    assert ev.latency(t, y, p) == ev.latency(t, y, p, segment=seg)
    assert ev.segments(t, seg).max() == 0


# -- A5 ----------------------------------------------------------------------------------
def test_online_config_has_no_own_max_train():
    pytest.importorskip("biomech")
    from biomech.config import ConfigError

    from ml.online import stage as st

    assert "max_train" not in st.DEFAULTS
    with pytest.raises(ConfigError, match="ml_eval.max_train"):
        st.config({"max_train": 8000})


def test_user_weight_counts_all_calibration_windows():
    # 3 phases x k = 20 calibration windows weigh together as one average class
    w = va.user_weight("balanced", 7000, 7, 60)
    assert w * 60 == pytest.approx(7000 / 7)
    assert va.user_weight("balanced", 7000, 7, 0) == 0.0 and va.user_weight(2.0, 1, 1, 5) == 2.0


def test_loso_requires_max_train(ds):
    with pytest.raises(TypeError):
        ev.loso(ds, "imu", "lda")


def test_line_hz_agrees_with_biomech():
    pytest.importorskip("biomech")
    from biomech.emg.pipeline import take_line_hz

    rng = np.random.default_rng(2)
    fs = 2000.0
    t = np.arange(int(4 * fs)) / fs
    for f0 in (50.02, 59.97):
        uv = rng.normal(size=(t.size, 3)) * 5.0
        uv[:, 1] += 40.0 * np.sin(2 * np.pi * f0 * t)
        t_us = (t * 1e6).astype(np.int64)
        assert S.line_hz(uv, fs, t_us) == take_line_hz(uv, fs, t_us)


# -- B2 ----------------------------------------------------------------------------------
def test_few_shot_k0_reuses_the_main_predictions(ds):
    rows = ev.loso(ds, "emg_imu", "svm", seed=2, inner=2, max_train=300)
    cfg = va.config({"late_fusion": None, "reduced_emg": None,
                     "few_shot": {"feature_sets": ["emg_imu"], "k": [0, 20]}})
    main = va.main_params(rows)
    a = va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=300)
    b = va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=300,
                        main_pred=va.main_predictions(rows))
    cols = [c for c in a.columns if c != "fit_time_s"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert b[b.k == 0].fit_time_s.isna().all() and b[b.k > 0].fit_time_s.notna().all()
    # A different subsample size must not reuse them.
    c = va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=250,
                        main_pred=va.main_predictions(rows))
    assert c[c.k == 0].fit_time_s.notna().all()
    with pytest.raises(RuntimeError, match="cancelled"):
        va.run_variants(ds, cfg, main, seed=2, inner=2, max_train=300,
                        cancelled=lambda: True)


def test_blocked_in_parallel_equals_serial(ds):
    a = ev.blocked(ds, "emg_imu", "svm", k=3, seed=1)
    b = ev.blocked(ds, "emg_imu", "svm", k=3, seed=1, n_jobs=2)
    assert a == b


def test_blocked_uses_the_loso_choice_per_subject(ds, monkeypatch):
    """Like-for-like blocked CV: each subject's folds get the hyper-parameters the nested
    LOSO search chose for the fold that held that subject out; subjects without a choice
    keep the defaults; serial and parallel agree."""
    seen: dict[str, list] = {}
    real = ev._run_fold

    def spy(f, X, y, model, slices, kw, *a, **k):
        seen.setdefault(f.pid, []).append((kw["search"], dict(kw["params"] or {})))
        return real(f, X, y, model, slices, kw, *a, **k)

    monkeypatch.setattr(ev, "_run_fold", spy)
    pids = sorted(set(ds.pid))
    chosen = {pids[0]: {"C": 0.1}, pids[1]: {"C": 10.0}}
    rows = ev.blocked(ds, "imu", "svm", k=3, seed=1, params_by_pid=chosen)
    assert all(not srch for v in seen.values() for srch, _ in v)
    assert all(p == {"C": 0.1} for _, p in seen[pids[0]])
    assert all(p == {"C": 10.0} for _, p in seen[pids[1]])
    assert all(p == {} for pid in pids[2:] for _, p in seen[pid])
    by = {r["pid"]: r for r in rows}
    assert by[pids[0]]["hyper"] == "loso" and by[pids[0]]["best_params"] == "{'C': 0.1}"
    assert by[pids[2]]["hyper"] == "default"
    monkeypatch.setattr(ev, "_run_fold", real)
    assert ev.blocked(ds, "imu", "svm", k=3, seed=1, params_by_pid=chosen, n_jobs=2) == \
        ev.blocked(ds, "imu", "svm", k=3, seed=1, params_by_pid=chosen)


def test_matrix_cache(ds):
    d = concat([_participant("s01", 1), _participant("s02", 2)])
    X1, s1 = d.matrix("emg_imu")
    X2, s2 = d.matrix("emg_imu")
    assert X1 is X2 and s1 == s2 and not X1.flags.writeable
    fresh = np.concatenate([d.blocks["emg"], d.blocks["imu"]], axis=1).astype(np.float64)
    np.testing.assert_array_equal(X1, fresh)
    assert pickle.loads(pickle.dumps(d))._matrix_cache == {}
    sub = d.subset(np.arange(10))
    assert sub._matrix_cache == {} and sub.matrix("emg_imu")[0].shape[0] == 10


def test_svm_kernel_cache_does_not_change_the_fit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int)
    est = reg.get("svm").make(0, C=1.0)
    est.fit(X, y)
    assert est.pipe_.named_steps["clf"].cache_size == reg.SVC_CACHE_MB
    small = reg.get("svm").make(0, C=1.0)
    small.fit(X, y)
    small.pipe_.named_steps["clf"].set_params(cache_size=200)
    small.pipe_.fit(X, y)
    np.testing.assert_array_equal(est.predict(X), small.predict(X))
    np.testing.assert_array_equal(est.pipe_.named_steps["clf"].dual_coef_,
                                  small.pipe_.named_steps["clf"].dual_coef_)


# -- B3: one filter pass for windows and reference ----------------------------------------
def _take(rng):
    from scipy.signal import butter, sosfilt

    n = int(16 * FS)
    t = (10_000_000 + np.arange(n) * 1e6 / FS).astype(np.int64)
    sos = butter(4, [20.0, 450.0], btype="band", fs=FS, output="sos")
    x = sosfilt(sos, rng.normal(size=(n, 2)), axis=0) * 10.0
    x[int(9 * FS):int(12 * FS)] *= 8.0
    phases = [("rest", int(t[int(2 * FS)]), int(t[int(7 * FS)])),
              ("open_close", int(t[int(8 * FS)]), int(t[-1]))]
    labels = {"open_close": (np.arange(phases[1][1], phases[1][2], 25_000),
                             np.zeros(len(range(phases[1][1], phases[1][2], 25_000)), int))}
    labels["open_close"][1][len(labels["open_close"][1]) // 3:] = 1
    masks = {}
    for name, a, b in phases:
        m = np.zeros((int(((t >= a) & (t <= b)).sum()), 2), bool)
        m[100:180, 0] = True                                  # a rail / lead-off stretch
        masks[name] = m
    return t, x, phases, labels, masks


@pytest.mark.parametrize("emg_norm", ["rest_relative", "mvc"])
def test_prefiltered_windows_are_identical(emg_norm):
    t, x, phases, labels, masks = _take(np.random.default_rng(5))
    cfg = WindowConfig(warmup_s=1.0, emg_norm=emg_norm)
    kw = dict(pid="s01", session="s01", phases=phases, emg_t=t, emg_uv=x, emg_fs=FS,
              line_hz=50.0, ref_uv=np.array([37.0, 81.0]), masks=masks, channels=CH,
              labels=labels, cfg=cfg)
    a = build_windows(**kw)
    filt = filter_phases(phases, t, x, FS, 50.0, masks, CH, cfg)
    b = build_windows(**kw, filtered=filt)
    for blk in a.blocks:
        np.testing.assert_array_equal(a.blocks[blk], b.blocks[blk])
    np.testing.assert_array_equal(a.y, b.y)
    assert a.meta["thr"] == b.meta["thr"] and a.meta["norm"] == b.meta["norm"]


def test_reference_from_the_window_filter_pass_is_identical():
    pytest.importorskip("biomech")
    from ml.biomech_stages import _causal_segment, _reusable

    t, x, phases, _, masks = _take(np.random.default_rng(6))
    cfg = WindowConfig(warmup_s=1.0)
    by = {n: (a, b) for n, a, b in phases}
    sd = SimpleNamespace(
        emg_fs=FS,
        phase=lambda n: SimpleNamespace(t_start_us=by[n][0], t_end_us=by[n][1]),
        emg_window=lambda a, b, names: (t[(t >= a) & (t <= b)], x[(t >= a) & (t <= b)], names))
    t0 = phases[0][1] - int(cfg.warmup_s * 1e6)
    sel = t >= t0
    filt = filter_phases(phases, t[sel], x[sel], FS, 50.0, masks, CH, cfg)
    for name, _, _ in phases:
        seg = _causal_segment(sd, name, CH, 50.0, masks[name], cfg)
        np.testing.assert_array_equal(seg, filt[name][1])
    # Only phases whose decoding mask equals the reference mask are reused.
    other = {"rest": masks["rest"].copy(), "open_close": masks["open_close"].copy()}
    other["open_close"][500:600, 1] = True                    # e.g. the lowfreq rule added
    assert set(_reusable(filt, masks, other)) == {"rest"}
    assert set(_reusable(filt, masks, masks)) == {"rest", "open_close"}
    ref_a = S.emg_reference({}, {"open_close": _causal_segment(sd, "open_close", CH, 50.0,
                                                               masks["open_close"], cfg)},
                            CH, FS, mode="max_recorded")
    ref_b = S.emg_reference({}, {"open_close": filt["open_close"][1]}, CH, FS,
                            mode="max_recorded")
    np.testing.assert_array_equal(ref_a[0], ref_b[0])


def test_rest_relative_features_depend_on_the_reference_only_at_eps_level():
    """Why B3 keeps the true reference: the D13 features are not bit-identical under a
    different ``ref_uv`` (EPS in log(amp + EPS), rounding), only nearly so."""
    t, x, phases, labels, masks = _take(np.random.default_rng(7))
    cfg = WindowConfig(warmup_s=1.0)
    kw = dict(pid="s01", session="s01", phases=phases, emg_t=t, emg_uv=x, emg_fs=FS,
              line_hz=50.0, masks=masks, channels=CH, labels=labels, cfg=cfg)
    filt = filter_phases(phases, t, x, FS, 50.0, masks, CH, cfg)
    a = build_windows(**kw, ref_uv=np.array([37.0, 81.0]), filtered=filt)
    b = build_windows(**kw, ref_uv=np.array([1.0, 1.0]), filtered=filt)
    assert np.allclose(a.blocks["emg"], b.blocks["emg"], atol=1e-3)
