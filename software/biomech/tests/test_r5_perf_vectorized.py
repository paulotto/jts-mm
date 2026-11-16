# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5 performance rewrites (plan B3) are bit-identical to the code they replace.

Each test keeps the former implementation as the reference and compares with
``np.array_equal`` (NaN == NaN, signed zeros distinguished), never with a tolerance.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest


def _same(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return (a.shape == b.shape and a.dtype == b.dtype
            and bool(np.array_equal(a, b, equal_nan=True))
            and bool(np.array_equal(np.signbit(a), np.signbit(b))))


# -- jtsmm.imu.batch.quat_continuous (B3.1) -----------------------------------------------
def test_quat_continuous_equals_the_sequential_rule():
    from jtsmm.imu.batch import _quat_continuous_loop, quat_continuous, quat_normalize

    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(0, 300))
        if trial % 2:                       # smooth stream with random sign flips
            q = np.cumsum(rng.normal(scale=0.05, size=(n, 4)), 0) + [1, 0, 0, 0]
            q *= rng.choice([-1.0, 1.0], size=(n, 1))
        else:
            q = rng.normal(size=(n, 4))
        q = quat_normalize(q) if n else q
        if n > 6:
            q[rng.integers(0, n, 3)] = np.nan            # NaN dot: parity restarts
            q[rng.integers(0, n, 2)] = 0.0               # zero quaternion: dot exactly 0
            i = int(rng.integers(1, n))
            q[i - 1], q[i] = [1, 0, 0, 0], [0, 1, 0, 0]  # orthogonal neighbours
        assert _same(quat_continuous(q), _quat_continuous_loop(q)), trial


def test_quat_resample_prepared_equals_unprepared():
    from jtsmm.imu.batch import quat_prepare, quat_resample

    rng = np.random.default_rng(1)
    t = np.cumsum(rng.integers(2000, 3000, 500)).astype(np.int64)
    q = np.cumsum(rng.normal(scale=0.05, size=(500, 4)), 0) + [1, 0, 0, 0]
    q *= rng.choice([-1.0, 1.0], size=(500, 1))
    td = np.linspace(t[0] - 5000, t[-1] + 5000, 777).astype(np.int64)
    assert _same(quat_resample(t, quat_prepare(q), td, prepared=True), quat_resample(t, q, td))


def test_imu_streams_are_loaded_once_per_session_and_read_only(monkeypatch):
    from biomech.imu import preprocess as P

    calls = []

    def fake(sd, slot):
        calls.append(slot)
        a = np.arange(3.0)
        a.flags.writeable = False
        return SimpleNamespace(q=a)

    monkeypatch.setattr(P, "_load", fake)

    class SD:                                         # any weak-referenceable session
        pass

    sd, other = SD(), SD()
    assert P.load(sd, "mta") is P.load(sd, "mta") and calls == ["mta"]
    P.load(sd, "cra")
    P.load(other, "mta")
    assert calls == ["mta", "cra", "mta"]
    with pytest.raises(ValueError):
        P.load(sd, "mta").q[0] = 1.0


def test_imu_stream_quat_at_equals_quat_resample():
    from jtsmm.imu.batch import quat_resample

    from biomech.imu.preprocess import ImuStream

    rng = np.random.default_rng(2)
    t = np.cumsum(rng.integers(2000, 3000, 300)).astype(np.int64)
    q = np.cumsum(rng.normal(scale=0.05, size=(300, 4)), 0) + [1, 0, 0, 0]
    s = ImuStream("mta", t, q, t, q[:, :3], 0, 0)
    for k in range(3):                              # cached prepared stream on repeat calls
        td = np.sort(rng.integers(t[0], t[-1], 200)).astype(np.int64)
        assert _same(s.quat_at(td), quat_resample(t, q, td))


# -- jtsmm.emg.onset (B3.4, B3.5) -----------------------------------------------------------
@pytest.mark.parametrize("method", ["tkeo", "hodges_bui", "envelope"])
@pytest.mark.parametrize("causal", [False, True])
def test_precomputed_rest_threshold_gives_the_same_onsets(method, causal):
    from jtsmm.emg.onset import OnsetParams, baseline_stats, detect_onsets

    rng = np.random.default_rng(3)
    fs = 2000.0
    op = OnsetParams(method=method, k_sd=3.0)
    rest = rng.normal(size=4000)
    rest[500:700] = np.nan                           # masked rest: pooled runs (E18)
    for trial in range(20):
        x = rng.normal(size=1500) * (1 + 8 * (np.arange(1500) > rng.integers(300, 1200)))
        if trial % 3 == 0:
            x[100:130] = np.nan
        st = baseline_stats(rest, fs, op, causal=causal)
        a = detect_onsets(x, fs, None, op, causal=causal, baseline_signal=rest)
        b = detect_onsets(x, fs, None, op, causal=causal, baseline_stats=st)
        assert _same(a, b)
    short = np.full(4000, np.nan)
    short[:10] = 1.0                                 # < 8 baseline samples: no onsets
    assert baseline_stats(short, fs, op, causal=causal) is None
    assert detect_onsets(x, fs, None, op, causal=causal, baseline_stats=None).size == 0


def test_cached_hodges_bui_filter_equals_a_fresh_design():
    from scipy.signal import butter

    from jtsmm.emg.onset import _hb_sos

    for fs in (1000.0, 2000.0, 4000.0):
        assert _same(_hb_sos(fs), butter(2, 50.0 / (0.5 * fs), output="sos"))
        assert _hb_sos(fs) is _hb_sos(fs) and not _hb_sos(fs).flags.writeable


# -- biomech.timing.ensemble bootstrap (B3.3) ---------------------------------------------
def test_batched_bootstrap_medians_equal_the_per_resample_loop():
    from biomech.timing.ensemble import _boot_medians

    warnings.simplefilter("ignore", RuntimeWarning)
    rng0 = np.random.default_rng(4)
    for trial in range(80):
        n, G = int(rng0.integers(3, 40)), int(rng0.integers(1, 120))
        y = rng0.normal(size=(n, G)) * 5
        y[rng0.random((n, G)) < rng0.random() * 0.6] = np.nan
        y[:, : G // 4] = np.nan                       # all-NaN lanes
        if trial % 2:                                 # +0 / -0 ties at the median
            y[rng0.random((n, G)) < 0.15] = 0.0
            y[rng0.random((n, G)) < 0.15] = -0.0
        if trial % 5 == 0:
            y[rng0.random((n, G)) < 0.05] = np.inf
        if trial % 4 == 0:
            y = np.round(y)
        if trial % 7 == 0:
            y = y.astype(np.float32)
        nb = (0, 1, 63, 64, 65, 200)[trial % 6]
        r1, r2 = np.random.default_rng(trial), np.random.default_rng(trial)
        ref = [np.nanmedian(y[r1.integers(0, n, n)], axis=0) for _ in range(nb)]
        new = _boot_medians(y, nb, r2)
        assert len(new) == nb and all(_same(a, b) for a, b in zip(ref, new)), trial
        assert r1.bit_generator.state == r2.bit_generator.state     # same stream after


def test_ensemble_events_unchanged():
    from biomech.timing import ensemble as E

    warnings.simplefilter("ignore", RuntimeWarning)
    p = E.EnsembleParams(n_boot=150)
    g = E.grid_ms(p)
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(size=(12, g.size)), axis=1) * 0.3
    y[:, g > 0] += 12.0
    y[rng.random(y.shape) < 0.1] = np.nan

    def old(y, kind, rng):
        fn = E.rise_time if kind == "rise" else E.release_time
        t, change = fn(np.nanmedian(y, axis=0), g, p)
        boots = np.array([fn(np.nanmedian(y[rng.integers(0, len(y), len(y))], axis=0), g, p)[0]
                          for _ in range(p.n_boot)])
        fin = boots[np.isfinite(boots)]
        lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > p.n_boot / 2
                  else (np.nan, np.nan))
        return {"t_ms": t, "lead_ms": -t, "ci_lo_ms": float(lo), "ci_hi_ms": float(hi),
                "change_db": change, "boot_found": float(fin.size / max(boots.size, 1))}

    for kind in ("rise", "release"):
        a = E.ensemble_events(y, g, kind, p, np.random.default_rng(9))
        b = old(y, kind, np.random.default_rng(9))
        assert a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a), kind


# -- biomech.stats.core.hedges_g (B3.6) ----------------------------------------------------
def test_vectorized_hedges_g_equals_the_per_resample_loop():
    from biomech.stats.core import hedges_g

    def old(a, b, n_boot=4000, seed=0):
        a = np.asarray(a, float)[np.isfinite(a)]
        b = np.asarray(b, float)[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            return {"g": np.nan, "g_lo": np.nan, "g_hi": np.nan}

        def g(x, y):
            sp = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1))
                         / (x.size + y.size - 2))
            j = 1 - 3 / (4 * (x.size + y.size) - 9)
            return float(j * (x.mean() - y.mean()) / sp) if sp > 0 else np.nan

        rng = np.random.default_rng(seed)
        boots = [g(rng.choice(a, a.size), rng.choice(b, b.size)) for _ in range(n_boot)]
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        return {"g": g(a, b), "g_lo": float(lo), "g_hi": float(hi)}

    warnings.simplefilter("ignore", RuntimeWarning)
    rng = np.random.default_rng(6)
    for t in range(60):
        na, nb = int(rng.integers(0, 25)), int(rng.integers(0, 25))
        a = rng.normal(size=na) * 10 ** rng.uniform(-3, 3)
        b = rng.normal(size=nb) + 0.3
        if t % 4 == 0:
            a = np.round(a)                          # ties: zero-variance resamples
        if t % 6 == 0:
            a = np.full(na, 2.0)
        r1, r2 = old(a, b, 500, t), hedges_g(a, b, n_boot=500, seed=t)
        assert all(_same(r1[k], r2[k]) for k in r1), t


# -- biomech.kinematics.se3.lowpass_batch / montecarlo.null_distribution (B3.2) -----------
def _ideal(rng, N):
    from biomech.kinematics import se3
    from biomech.uncertainty.montecarlo import _hinge_pose

    th = np.cumsum(np.abs(rng.normal(0, 0.01, N)))
    return se3.compose(_hinge_pose(np.array([0.0, 1.0, 0.0]), np.array([-80.0, 0.0, 10.0]), th),
                       np.eye(4)[None])


def test_lowpass_batch_equals_lowpass_per_sequence():
    from biomech.kinematics import se3
    from biomech.uncertainty.montecarlo import _noisy

    rng = np.random.default_rng(7)
    for trial in range(15):
        N, K = int(rng.integers(5, 200)), int(rng.integers(1, 40))
        T0 = _ideal(rng, N)
        if trial % 3 == 0:
            T0[rng.integers(0, N, 3)] = np.nan          # several valid segments
        Ts = np.stack([_noisy(T0, 0.1, 0.07, rng) for _ in range(K)])
        a = se3.lowpass_batch(Ts, 40.0, 6.0, 2)
        b = np.stack([se3.lowpass(T, 40.0, 6.0, 2) for T in Ts])
        assert _same(a, b), trial
    Ts[1, 0] = np.nan                                   # differing validity: per-sequence
    assert _same(se3.lowpass_batch(Ts, 40.0, 6.0, 2),
                 np.stack([se3.lowpass(T, 40.0, 6.0, 2) for T in Ts]))


@pytest.mark.parametrize("resampled", [False, True])
def test_null_distribution_equals_the_per_simulation_loop(resampled):
    from biomech.kinematics import se3
    from biomech.kinematics.coupling import HINGE_POINTS, Opening
    from biomech.kinematics.hinge_fit import fit_hinge
    from biomech.uncertainty import montecarlo as mc

    rng = np.random.default_rng(8)
    pts = {r: rng.normal(size=3) * 30 for r in HINGE_POINTS}
    jm = SimpleNamespace(points_mp=pts, fs=40.0)
    ops = []
    for c in range(4):
        n = int(rng.integers(12, 30))
        th = np.linspace(0, 20 + 3 * c, n)
        if c == 2:
            th[3] = np.nan
        T = mc.se3.compose(mc._hinge_pose(np.array([0.0, 1.0, 0.0]), np.zeros(3),
                                          np.radians(np.nan_to_num(th))), np.eye(4)[None])
        ops.append(Opening(c, T, th))
    n_axis, s_axis = np.array([0.0, 1.0, 0.0]), np.array([-80.0, 0.0, 10.0])
    noise = mc.ResidualNoise([rng.normal(size=(60, 6)) * 0.05], 8) if resampled else None

    def old(n_sims, seed):
        r = np.random.default_rng(seed)
        out = np.full(n_sims, np.nan)
        P = np.stack([jm.points_mp[k] for k in HINGE_POINTS])
        n_pad = int(round(mc.PAD_S * jm.fs))
        ideal, cuts = [], []
        for op in ops:
            th = np.radians(mc._padded_theta(op.theta, n_pad))
            pad = (th.size - len(op.theta)) // 2
            T = np.full((len(th), 4, 4), np.nan)
            ok = np.isfinite(th)
            T[ok] = se3.compose(mc._hinge_pose(n_axis, s_axis, th[ok]), op.T[0][None])
            ideal.append(T)
            cuts.append(slice(pad, pad + len(op.theta)))
        for k in range(n_sims):
            Ts = [se3.lowpass(mc._noisy(T, 0.1, 0.07, r, noise), jm.fs, 6.0, 2)[c]
                  for T, c in zip(ideal, cuts)]
            A, B = mc._stack(P, Ts, 30)
            if A.shape[0] < mc.MIN_FRAMES:
                continue
            out[k] = fit_hinge(A, B, s0=s_axis, n0=n_axis).rms(0)
        return out

    new = mc.null_distribution(jm, ops, n_axis, s_axis, 0.1, 0.07, 6, 6.0, 30, 11, 2,
                               noise=noise)
    assert _same(new, old(6, 11))


# -- biomech.io.context EMG cache (B3.7) -----------------------------------------------------
def test_emg_cache_size(monkeypatch):
    from biomech.io.context import EMG_CACHE_ENV, emg_cache_size

    monkeypatch.delenv(EMG_CACHE_ENV, raising=False)
    assert emg_cache_size(14) == 14 and emg_cache_size(2) == 4 and emg_cache_size(14, 6) == 6
    monkeypatch.setenv(EMG_CACHE_ENV, "3")
    assert emg_cache_size(14) == 3 and emg_cache_size(14, 9) == 9
