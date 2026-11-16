# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Regression tests for the decoding errata (2026-10-16 review) and D13.

* ZC/SSC/WAMP thresholds on the sample-difference scale (features informative);
* SVM grid (C down to 0.01, gamma multiples of "scale"); LDA uniform priors;
* per-participant feature names checked when pooling the biomech stage output;
* normalization reference on the causal chain of the windows;
* D13 rest-relative EMG features: gain invariant, rest windows centred;
* exploratory fusion diagnostics and block permutation importance.

All data is synthetic.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, sosfilt

from ml.core import features as F
from ml.core import signals as S
from ml.core.dataset import WindowConfig, WindowDataset, build_windows, concat
from ml.evaluation import diagnostics as dg
from ml.models import registry as reg
from ml.protocols import evaluate as ev

ML_ROOT = Path(__file__).resolve().parents[1] / "ml"
FS = 2000.0
CH = ["masseter_r", "digastric_l"]


def _emg(n: int, rng: np.random.Generator, fs: float = FS) -> np.ndarray:
    sos = butter(4, [20.0, 450.0], btype="band", fs=fs, output="sos")
    return sosfilt(sos, rng.normal(size=(n, len(CH))), axis=0)


# -- ZC / SSC / WAMP thresholds ---------------------------------------------------------
def test_zc_ssc_wamp_thresholds_on_difference_scale_are_informative():
    fs, win = 4000.0, 800
    rng = np.random.default_rng(0)
    x = _emg(int(6 * fs), rng, fs)
    burst = slice(int(3 * fs), int(4.5 * fs))
    x[burst] *= 10.0                                    # known burst, 10 x rest amplitude
    rest = x[: int(3 * fs)]
    thr = F.rest_thresholds(rest)
    assert np.allclose(thr, np.nanstd(np.diff(rest, axis=0), axis=0))
    names = ("zc", "ssc", "wamp")
    ends = np.arange(win - 1, x.shape[0], 200)
    f, _ = F.emg_window_features(x, fs, ends, win, thr, names)
    f = f.reshape(len(ends), len(CH), len(names))
    in_rest = ends < int(3 * fs)
    in_burst = (ends - win + 1 >= burst.start) & (ends < burst.stop)
    r, b = f[in_rest].mean(axis=(0, 1)), f[in_burst].mean(axis=(0, 1))
    assert 0.15 < r[2] < 0.6                            # WAMP fires at rest ~ P(|z| > 1)
    assert np.all(r > 0.01) and np.all(b - r > 0.02)    # all three rise with the burst
    assert np.all(f.reshape(-1, len(names)).std(axis=0) > 0.01)   # not near-constant
    # The erratum: an amplitude-SD threshold hardly ever fires at rest (near-constant).
    old, _ = F.emg_window_features(x, fs, ends, win, np.nanstd(rest, axis=0), names)
    assert old.reshape(len(ends), len(CH), 3)[in_rest][..., 2].mean() < 0.5 * r[2]


# -- models -----------------------------------------------------------------------------
def test_svm_grid_and_gamma_multiples_of_scale():
    g = reg.get("svm").grid
    assert g["C"] == [0.01, 0.1, 1.0, 10.0] and g["gamma_mult"] == [0.25, 1.0, 4.0]
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 5)) * [1, 2, 3, 4, 5]
    y = np.tile([0, 1, 2], 40)
    X[:, 0] += y
    m1 = reg.get("svm").make(0).fit(X, y)
    m4 = reg.get("svm").make(0, gamma_mult=4.0).fit(X, y)
    Xs = m1.pipe_.named_steps["scale"].transform(X)
    assert np.isclose(m1.gamma_, 1.0 / (X.shape[1] * Xs.var()))      # == sklearn "scale"
    assert np.isclose(m4.gamma_, 4.0 * m1.gamma_)
    assert m4.pipe_.named_steps["clf"].gamma == m4.gamma_


def test_lda_priors_uniform_by_default():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(300, int), np.ones(30, int), np.full(30, 2)]   # REST-dominated
    X = rng.normal(size=(y.size, 4)) + y[:, None]
    m = reg.get("lda").make(0).fit(X, y)
    assert np.allclose(m.pipe_.named_steps["clf"].priors_, 1 / 3)
    e = reg.get("lda").make(0, priors="empirical").fit(X, y)
    assert e.pipe_.named_steps["clf"].priors_[0] > 0.8
    with pytest.raises(ValueError, match="priors"):
        reg.get("lda").make(0, priors="rest").fit(X, y)


# -- pooling: per-participant names -----------------------------------------------------
def _part(pid: str, names: list[str]) -> WindowDataset:
    n = 6
    return WindowDataset(blocks={"emg": np.zeros((n, len(names)))}, y=np.zeros(n, int),
                         pid=np.full(n, pid), session=np.full(n, pid),
                         phase=np.full(n, "rest"), t_end_us=np.arange(n),
                         groups=np.zeros(n, int), names={"emg": names})


def test_concat_rejects_differing_feature_names():
    with pytest.raises(ValueError, match="feature names"):
        concat([_part("s01", ["a|mav", "b|mav"]), _part("s02", ["b|mav", "a|mav"])])


def test_pool_uses_per_participant_names():
    pytest.importorskip("biomech")
    from ml.biomech_stages import _pool

    feats = pd.DataFrame({"unit": ["s01", "s01", "s02", "s02"], "block": ["emg"] * 4,
                          "feature": ["a|mav", "b|mav", "b|mav", "a|mav"]})
    arrays = {pid: {"X_emg": np.zeros((4, 2), np.float32), "y": np.zeros(4, np.int8),
                    "t_end_us": np.arange(4), "phase": np.zeros(4, np.int8)}
              for pid in ("s01", "s02")}
    ctx = SimpleNamespace(participants=[SimpleNamespace(pid="s01"), SimpleNamespace(pid="s02")],
                          upstream=lambda st, t: feats,
                          upstream_arrays=lambda st, pid: arrays[pid])
    with pytest.raises(ValueError, match="feature names"):
        _pool(ctx)                                     # swapped channel order is caught
    feats.loc[2:, "feature"] = ["a|mav", "b|mav"]
    d = _pool(ctx)
    assert d.names["emg"] == ["a|mav", "b|mav"] and len(d) == 8


# -- causal reference -------------------------------------------------------------------
def test_stage_reference_uses_causal_chain():
    pytest.importorskip("biomech")
    from ml.biomech_stages import _causal_segment

    src = (ML_ROOT / "biomech_stages.py").read_text(encoding="utf-8")
    calls = {n.func.attr for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "ref_uv" not in calls                       # biomech's zero-phase reference unused
    rng = np.random.default_rng(3)
    t = (np.arange(int(4 * FS)) * 1e6 / FS).astype(np.int64)
    uv = rng.normal(size=(t.size, 2)) * 20.0
    t0, t1 = int(t[int(2 * FS)]), int(t[-1])
    sd = SimpleNamespace(
        emg_fs=FS, phase=lambda name: SimpleNamespace(t_start_us=t0, t_end_us=t1),
        emg_window=lambda a, b, names: (t[(t >= a) & (t <= b)], uv[(t >= a) & (t <= b)], names))
    cfg = WindowConfig(warmup_s=1.0)
    got = _causal_segment(sd, "mvc_closing", CH, 50.0, None, cfg)
    sel = t >= t0 - 1_000_000
    want = S.causal_emg(t[sel], uv[sel], FS, 50.0)[t[sel] >= t0]
    assert np.allclose(got, want)


def test_emg_reference_modes():
    rng = np.random.default_rng(4)
    mvc_close = rng.normal(size=(4000, 2)) * [50.0, 1.0]
    mvc_open = rng.normal(size=(4000, 2)) * [1.0, 40.0]
    task = rng.normal(size=(4000, 2)) * [80.0, 10.0]     # masseter exceeds its MVC in a task
    mvc = {"mvc_closing": mvc_close, "mvc_opening": mvc_open}
    p95, src = S.emg_reference(mvc, {"chewing": task}, CH, FS, mode="p95")
    env = S.causal_rms_envelope(mvc_close[:, :1], FS, 0.15)[:, 0]
    assert np.isclose(p95[0], np.nanpercentile(env, 95)) and list(src) == ["mvc", "mvc"]
    mx, src = S.emg_reference(mvc, {"chewing": task}, CH, FS, mode="max_recorded")
    assert list(src) == ["task", "mvc"] and mx[0] > 70.0 and 35.0 < mx[1] < 60.0
    assert np.allclose(S.mvc_reference(mvc, CH, FS), p95)
    env_c = S.causal_rms_envelope(np.ones((10, 1)), 1000.0, 0.004)
    assert np.isnan(env_c[0, 0]) and np.allclose(env_c[2:, 0], 1.0)  # trailing, needs 50 %


# -- D13 rest-relative normalization ----------------------------------------------------
def _take(gain: float, *, seed: int = 5, emg_norm: str = "rest_relative", pid: str = "s01"):
    rng = np.random.default_rng(seed)
    n = int(8 * FS)
    t = (1_000_000 + np.arange(n) * 1e6 / FS).astype(np.int64)
    x = _emg(n, rng) * 5.0
    act = (t >= 4_000_000) & (((t - 4_000_000) // 500_000) % 2 == 1)
    x[act] *= [8.0, 3.0]
    phases = [("rest", 1_500_000, 3_500_000), ("open_close", 4_000_000, int(t[-1]))]
    labels = {}
    for name, a, b in phases:
        lt = np.arange(a, b, 25_000)
        ly = np.where((name == "open_close") & (((lt - 4_000_000) // 500_000) % 2 == 1), 1, 0)
        labels[name] = (lt, ly)
    cfg = WindowConfig(warmup_s=0.5, emg_norm=emg_norm,
                       emg_features=("log_mav", "log_rms", "mav", "zc", "ssc", "wamp", "mnf",
                                     "band_20_100"))
    return build_windows(pid=pid, session=pid, phases=phases, emg_t=t, emg_uv=gain * x,
                         emg_fs=FS, line_hz=50.0, ref_uv=np.array([10.0, 10.0]), masks={},
                         channels=CH, labels=labels, cfg=cfg)


def test_rest_relative_features_are_gain_invariant():
    a, b = _take(1.0), _take(7.0)
    for blk in ("emg", "seq"):
        assert np.allclose(a.blocks[blk], b.blocks[blk], atol=1e-3), blk
    ma, mb = _take(1.0, emg_norm="mvc"), _take(7.0, emg_norm="mvc")
    j = ma.names["emg"].index("masseter_r|log_rms")
    assert np.allclose(mb.blocks["emg"][:, j] - ma.blocks["emg"][:, j], np.log(7.0), atol=1e-3)
    with pytest.raises(ValueError, match="emg_norm"):
        WindowConfig.from_dict({"emg_norm": "zscore"})


def test_rest_windows_centred_after_rest_relative():
    d = _take(3.0)
    rest = d.blocks["emg"][d.phase == "rest"]
    assert d.meta["n_rest_windows"] == rest.shape[0] and d.meta["emg_norm"] == "rest_relative"
    kinds = np.array([F.feature_kind(n) for n in d.names["emg"]])
    assert np.allclose(rest[:, kinds == "z"].mean(0), 0.0, atol=1e-9)
    assert np.allclose(np.median(rest[:, kinds != "z"], axis=0), 0.0, atol=1e-9)
    assert np.all(np.abs(rest.mean(0)) < 0.25)
    move = d.blocks["emg"][d.y == 1]
    j = d.names["emg"].index("masseter_r|log_rms")
    assert move[:, j].mean() > 1.5                     # activity: log(8) above rest


def test_fusion_diagnostics_before_after():
    parts = [_take(g, seed=s, pid=p) for g, s, p in ((1.0, 5, "s01"), (4.0, 6, "s02"),
                                                    (15.0, 7, "s03"))]
    d = concat(parts)
    before, after = dg.norm_views(d)
    mvc = _take(4.0, seed=6, emg_norm="mvc", pid="s02")
    rows = d.pid == "s02"
    assert np.allclose(before[rows], F.to_norm_scale(mvc.blocks["emg"], mvc.names["emg"]),
                       atol=1e-6)                      # stored statistics invert exactly
    assert np.allclose(after, d.blocks["emg"])
    t = dg.fusion_diagnostics(d).set_index("feature")
    r = t.loc["masseter_r|log_rms"]
    assert r.ratio_before > 1.0 and r.ratio_after < 0.1 * r.ratio_before
    # Windows built with emg_norm mvc give the same "after" from their rest windows.
    m = concat([_take(g, seed=s, pid=p, emg_norm="mvc")
                for g, s, p in ((1.0, 5, "s01"), (4.0, 6, "s02"), (15.0, 7, "s03"))])
    assert np.allclose(dg.norm_views(m)[1], after, atol=1e-6)


def test_block_permutation_importance():
    rng = np.random.default_rng(8)
    y = np.tile([0, 1, 2], 100)
    X = np.hstack([rng.normal(size=(300, 3)) + 2.0 * y[:, None], rng.normal(size=(300, 2))])
    sl = [slice(0, 3), slice(3, 5)]
    m = reg.get("lda").make(0, slices=sl).fit(X, y)
    ba = ev.block_importance(m, X, y, sl, n_repeats=3, seed=0)
    assert ba[0] < 0.5 and ba[1] > 0.9                 # informative block matters, noise not
    rows = [{"pid": "s01", "feature_set": "emg_imu", "model": "lda", "balanced_accuracy": 0.95,
             "perm_ba|emg": ba[0], "perm_ba|imu": ba[1]}]
    t = ev.importance_table(rows)
    assert list(t.block) == ["emg", "imu"] and "perm_ba|emg" not in rows[0]
    assert np.isclose(t.ba_drop.iloc[0], 0.95 - ba[0])
