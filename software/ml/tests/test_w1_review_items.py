# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One regression test per open review item of plan W1 (a, g-s), plus loso naming and config.

All data is synthetic (tests/synth.py); nothing under data/ is read.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ml.config.classes import UNLABELLED
from ml.config.loader import describe_source, load_dataset_config
from ml.core import signals as S
from ml.core.dataset import WindowDataset, concat
from ml.datasets.builder import build_dataset
from ml.evaluation.metrics import macro_f1
from ml.io.labels import read_labels
from ml.protocols import evaluate as ev
from ml.protocols.folds import make_folds, normalize_protocol
from tests import synth

ML_ROOT = Path(__file__).resolve().parents[1] / "ml"


@pytest.fixture
def take(tmp_path):
    h5 = synth.write_session(synth.folder(tmp_path, "s01"))
    return tmp_path, h5, load_dataset_config(synth.rest_config(tmp_path))


def _pool(tmp_path, n=3, **kw):
    cfg = load_dataset_config(synth.rest_config(tmp_path))
    parts = []
    for i in range(n):
        h5 = synth.write_session(synth.folder(tmp_path, f"s0{i + 1}"), seed=i)
        synth.write_sidecar(h5, **kw)
        parts.append(build_dataset(h5, config=cfg))
    return concat(parts)


# (a) old sidecars: bout_id -1 for REST -> recomputed; groups unique across sessions
def test_a_negative_bout_ids_recomputed_and_unique_groups(tmp_path):
    h5 = synth.write_session(synth.folder(tmp_path, "s01"))
    lp = synth.write_sidecar(h5)
    with h5py.File(lp, "a") as f:
        y = f["labels/y"][()]
        b = f["labels/bout_id"][()]
        b[y == 0] = -1
        f["labels/bout_id"][...] = b
    assert (read_labels(lp).bout_id > 0).all()
    d = _pool(tmp_path, 2)
    for s in np.unique(d.session):
        other = d.groups[d.session != s]
        assert not set(d.groups[d.session == s]) & set(other)


# (g) within-subject CV: per-phase blocks with a guard band
def test_g_blocked_folds_have_guard_and_all_phases(tmp_path):
    d = _pool(tmp_path, 1)
    folds = make_folds(d, "blocked", k=5, guard_s=0.35)
    assert len(folds) == 5
    for f in folds:
        assert set(d.phase[f.test]) == set(d.phase[f.train]) == {"rest", "open_close",
                                                                   "left_right"}
        gap = np.abs(d.t_end_us[f.train][:, None] - d.t_end_us[f.test][None, :]).min()
        assert gap > 0.35e6
        assert not np.isin(f.train, f.test).any()


# (h) no whole-recording statistics: robust removed, rest/MVC references, causal fill
def test_h_normalization_is_label_free_and_causal(take, tmp_path):
    _, h5, cfg = take
    synth.write_sidecar(h5)
    cfg_path = tmp_path / "robust.yaml"
    cfg_path.write_text("dataset: {emg: {normalization: robust}}\n")
    with pytest.raises(ValueError, match="robust"):
        load_dataset_config(cfg_path)
    a = build_dataset(h5, config=cfg)
    with h5py.File(h5, "a") as f:            # change EMG after the rest phase only
        t = f["emg/t_master_us"][()].astype(np.int64)
        rest_end = int(f["phases/t_end_us"][0])
        x = f["emg/samples_uv"][()]
        x[t > rest_end + 2_000_000, :6] *= 3.0
        f["emg/samples_uv"][...] = x
    b = build_dataset(h5, config=cfg)
    assert np.allclose(a.meta["ref_uv"], b.meta["ref_uv"])     # reference from rest only
    mvc = synth.write_mvc(synth.folder(tmp_path, "s01", mvc=True))
    synth.write_sidecar(h5, mvc=mvc)
    c = build_dataset(h5)                                      # packaged default: mvc
    assert c.meta["normalization"] == "mvc" and c.meta["ref_source"].endswith("session.h5")
    x = np.zeros((6, 1))
    m = np.zeros((6, 1), bool)
    m[2:4] = True
    x[4:] = 9.0                                                # future values never leak back
    assert S.hold_fill(x, m)[2:4, 0].tolist() == [0.0, 0.0]


def test_h_mvc_required_or_explicit(take):
    _, h5, _ = take
    synth.write_sidecar(h5)
    with pytest.raises(ValueError, match="normalization 'mvc' needs an MVC take"):
        build_dataset(h5)


# (i) label at the window end by default; majority + purity as an option/diagnostic
def test_i_label_at_end_default_with_purity_diagnostic(take):
    _, h5, cfg = take
    synth.write_sidecar(h5)
    assert cfg.window.label_at == "end"
    d = build_dataset(h5, config=cfg)
    assert d.purity is not None and (d.purity < 0.6).any()    # transition windows kept
    from dataclasses import replace

    m = build_dataset(h5, config=replace(cfg, window=replace(cfg.window, label_at="majority")))
    assert len(m) < len(d) and np.nanmin(m.purity) >= 0.6


# (j) unpainted = 255 = ignored, never REST
def test_j_unlabelled_is_ignored(take):
    _, h5, cfg = take
    synth.write_sidecar(h5, unlabel=("left_right",))
    assert UNLABELLED in read_labels(h5.with_name("session_labels.h5")).y
    d = build_dataset(h5, config=cfg)
    assert "left_right" not in set(d.phase) and 6 not in set(d.y)   # RIGHT only in left_right


def test_j_labeling_app_defaults_to_unlabelled():
    from ml.labeling.constants import PAINT_IDS
    from ml.labeling.phases import PhaseView, reset_phase_labels
    from ml.io.session import PhaseInterval

    assert UNLABELLED in PAINT_IDS
    view = PhaseView(0, 1, PhaseInterval("p", "experiment", 0, 1), np.arange(3), np.zeros(3),
                     np.zeros(3))
    assert reset_phase_labels([1, 2, 3, 4], view) == [UNLABELLED] * 3 + [4]
    src = (ML_ROOT / "labeling" / "app.py").read_text()
    assert "np.full(n, UNLABELLED" in src and "np.zeros(n, dtype=np.uint8)" not in src


# (k) windows are time based: a window over a recording gap is dropped
def test_k_gap_windows_rejected(tmp_path):
    cfg = load_dataset_config(synth.rest_config(tmp_path))
    h5 = synth.write_session(synth.folder(tmp_path, "s01"), emg_gap_s=0.3)
    synth.write_sidecar(h5)
    d = build_dataset(h5, config=cfg)
    assert d.meta["rejected"]["gap"] > 0
    w = cfg.window.window_s * 1e6
    assert np.all(np.diff(d.t_end_us[d.phase == "open_close"]) > 0) and w > 0


# (l) EMG chain = jtsmm CausalEmgFilter with the tracking mains canceller
def test_l_causal_tracking_mains_removal():
    fs = 1000.0
    t = (np.arange(int(20 * fs)) * 1000).astype(np.int64)
    x = 200 * np.sin(2 * np.pi * 50.2 * t * 1e-6)[:, None] + np.random.default_rng(0).normal(
        size=(t.size, 1))
    line = S.line_hz(x, fs, t, 50.0)                          # measured grid frequency
    assert abs(line - 50.2) < 0.01
    y = S.causal_emg(t, x, fs, line)
    assert y[-5000:].std() < 0.01 * x[-5000:].std()
    x2 = x.copy()
    x2[15000:] += 500.0
    assert np.allclose(S.causal_emg(t, x2, fs, line)[:15000], y[:15000])
    assert "mains_mode=\"track\"" in (ML_ROOT / "core" / "signals.py").read_text()


# (m) a trajectory that does not match the optical grid raises instead of shifting
def test_m_trajectory_mismatch_raises():
    from ml.labeling.trajectory import _align_trajectory_to_optical

    opt = np.arange(100) * 25_000
    xyz = np.zeros((90, 3))
    with pytest.raises(ValueError, match="shifted"):
        _align_trajectory_to_optical(xyz, xyz, opt, np.arange(100), {}, sample_rate=40.0)
    cfg = {"analysis": {"experiment": {"frame_interval": [5, 94]}}}
    out = _align_trajectory_to_optical(xyz, xyz, opt, np.arange(100), cfg, sample_rate=40.0)
    assert out[3][0] == 5
    with pytest.raises(ValueError, match="frame_interval"):
        _align_trajectory_to_optical(xyz[:80], xyz[:80], opt, np.arange(100), cfg,
                                     sample_rate=40.0)


# (n) subject id from the folder name when /meta has none; error when unresolvable
def test_n_subject_id_from_folder_or_error(tmp_path):
    from ml.io.session import resolve_subject_id

    cfg = load_dataset_config(synth.rest_config(tmp_path))
    h5 = synth.write_session(synth.folder(tmp_path, "s09"))
    synth.write_sidecar(h5)
    pytest.importorskip("biomech.cohort")
    assert build_dataset(h5, config=cfg).pid[0] == "s09"
    other = tmp_path / "take"
    other.mkdir()
    h5b = synth.write_session(other)
    with h5py.File(h5b, "a") as f:
        del f["meta"].attrs["experiment_name"]
    assert resolve_subject_id(h5b, {}) == ""
    synth.write_sidecar(h5b)
    with pytest.raises(ValueError, match="no subject id"):
        build_dataset(h5b, config=cfg)
    synth.write_sidecar(h5b, subject="p9")                   # explicit sidecar subject wins
    assert build_dataset(h5b, config=cfg).pid[0] == "p9"


# (o) one macro-F1 over present labels; grouped SVM calibration; fold SD ddof = 1
def test_o_scorer_calibration_and_ddof(tmp_path):
    assert macro_f1(np.array([0, 0, 1]), np.array([0, 0, 1])) == 1.0  # absent classes ignored
    rng = np.random.default_rng(0)
    y = np.tile([0, 1, 2], 100)
    X = rng.normal(size=(300, 4)) + y[:, None]
    from ml.models import registry as reg

    m = reg.get("svm").make(0, probability=True).fit(X, y, groups=np.arange(300) // 50)
    assert m.proba_method_ == "sigmoid_grouped_cv"
    m2 = reg.get("svm").make(0, probability=True).fit(X, y)
    assert m2.proba_method_ == "softmax_decision"
    from ml.training.artifacts import write_aggregate_artifacts

    folds = [{"macro_f1": v} for v in (0.2, 0.4, 0.6)]
    agg = write_aggregate_artifacts(tmp_path, fold_metrics=folds, y_true_all=y, y_pred_all=y)
    assert np.isclose(agg["std"]["macro_f1"], np.std([0.2, 0.4, 0.6], ddof=1))
    src = (ML_ROOT / "protocols" / "evaluate.py").read_text()
    assert "macro_f1(y[te]" in src and "scoring=" not in src


# (p) predictions are mapped by row index; identical timestamps across takes are fine
def test_p_eval_maps_by_row_index(tmp_path):
    from ml.config.loader import load_train_config
    from ml.evaluation.evaluate import evaluate_run_models, evaluate_run_predictions
    from ml.training.run import train

    d = _pool(tmp_path, 3)
    assert len(np.unique(d.t_end_us)) < len(d)               # every take shares the clock
    path = d.save(tmp_path / "ds")
    cfg = load_train_config(synth.rest_config(tmp_path))
    run = train(path, cfg, runs_dir=tmp_path / "runs")
    saved = evaluate_run_predictions(run)
    again = evaluate_run_models(run, WindowDataset.load(path))
    assert np.array_equal(saved.predictions.indices, again.predictions.indices)
    assert np.array_equal(saved.predictions.y_pred, again.predictions.y_pred)
    z = dict(np.load(run / "per_fold" / "fold_0" / "predictions.npz"))
    z.pop("indices")
    np.savez(run / "per_fold" / "fold_0" / "predictions.npz", **z)
    with pytest.raises(ValueError, match="row indices"):
        evaluate_run_predictions(run)


# (q) one model registry and one feature registry; no hard-coded name lists elsewhere
def test_q_single_registries():
    from ml.models.registry import REGISTRY, get

    assert list(REGISTRY) == ["lda", "svm", "mlp", "gru"]
    with pytest.raises(KeyError, match="known"):
        get("rf")
    offenders = []
    for py in ML_ROOT.rglob("*.py"):
        if py.name == "registry.py":
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                vals = {e.value for e in node.elts if isinstance(e, ast.Constant)}
                if {"svm", "mlp"} <= vals or {"svm", "gru"} <= vals:
                    offenders.append(f"{py.relative_to(ML_ROOT)}:{node.lineno}")
    assert not offenders, offenders
    assert not (ML_ROOT / "features").exists() and not (ML_ROOT / "arrays").exists()


# (r) no stubs: IMU path and fusion are implemented
def test_r_fusion_and_imu_implemented(take):
    _, h5, cfg = take
    synth.write_sidecar(h5)
    d = build_dataset(h5, config=cfg)
    assert d.blocks["imu"].shape[1] == 25 and np.isfinite(d.blocks["imu"]).all()
    assert not (ML_ROOT / "preprocessing").exists()
    for py in ML_ROOT.rglob("*.py"):
        assert "NotImplementedError(\"" not in py.read_text(), py
    rows = ev.loso(_pool(Path(h5).parents[1] / "p", 3), "emg+imu", "late_mean:lda", max_train=None)
    assert len(rows) == 3


# (s) vectorized features equal a plain per-window loop
def test_s_vectorized_equals_loop():
    from ml.core import features as F

    x = np.random.default_rng(1).normal(size=(3000, 3))
    thr = np.array([0.5, 1.0, 1.5])
    ends = np.arange(399, 3000, 100)
    fast, _ = F.emg_window_features(x, 1000.0, ends, 400, thr)
    for i, e in enumerate(ends):
        w = x[e - 399:e + 1].T                                 # (C, L)
        mav = np.abs(w).mean(1)
        d = np.diff(w, axis=1)
        zc = ((w[:, :-1] * w[:, 1:] < 0) & (np.abs(d) >= thr[:, None])).mean(1)
        got = fast[i].reshape(3, -1)
        assert np.allclose(got[:, 0], np.log(mav + 1e-6)) and np.allclose(got[:, 3], zc)


# protocol names: "loso" is ambiguous
def test_loso_rename():
    with pytest.raises(ValueError, match="loso_subject.*loso_session"):
        normalize_protocol("loso")
    assert normalize_protocol("leave_one_subject_out") == "loso_subject"
    with pytest.raises(ValueError, match="blocked"):
        normalize_protocol("groupkfold")


def test_loso_session_vs_subject_folds(tmp_path):
    d = _pool(tmp_path, 2)
    d.session[d.pid == "s01"] = np.where(np.arange((d.pid == "s01").sum()) < 100, "a", "b")
    with pytest.warns(UserWarning, match="loso_subject"):
        fs = make_folds(d, "loso_session")
    assert len(fs) == 3
    assert {f.unit for f in make_folds(d, "loso_subject")} == {"s01", "s02"}


# ~/ml.yaml no longer overrides the packaged config silently
def test_home_config_is_ignored_and_source_printed(monkeypatch, tmp_path):
    import ml.config.loader as L

    home = tmp_path / "ml.yaml"
    home.write_text("dataset: {emg: {normalization: robust}}\n")
    monkeypatch.setattr(L, "HOME_YAML", home)
    assert L.default_config_path().parent.name == "etc"
    msg = describe_source(None)
    assert "packaged default" in msg and "ignored" in msg
    assert load_dataset_config().emg.normalization == "mvc"
    assert "--config" in describe_source(home)


def test_unknown_or_moved_config_keys_fail(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("dataset: {windowing: {window_ms: 200}}\n")
    with pytest.raises(ValueError, match="renamed to 'window'"):
        load_dataset_config(p)
    p.write_text("dataset: {emg: {causal: false}}\n")
    with pytest.raises(ValueError, match="causal"):
        load_dataset_config(p)
    p.write_text(json.dumps({"train": {"protocol": "loso"}}))
    from ml.config.loader import load_train_config

    with pytest.raises(ValueError, match="ambiguous"):
        load_train_config(p)
