# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 timing fixes: E18 masked envelopes / per-run rest thresholds, E21 causal baseline fill
and seed, B4 vectorized release detector, A3 crash guards, E27 causal chain, A6 zero-phase
onset spread."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from jtsmm.emg.onset import OnsetParams, baseline_threshold, detect_onsets, detector_signal

from biomech.timing import ensemble as ens
from biomech.timing import lead
from biomech.timing import release as rel
from biomech.timing.release import ReleaseParams

FS = 2000.0


# -- reference copies of the pre-R4 release detector (B4 equality) ------------------------
def _old_causal_db(t_us, cleaned, mask, rest_level, names, window_s):
    fs = 1e6 / np.median(np.diff(t_us))
    n = max(int(round(window_s * fs)), 1)
    idx = [names.index(m) for m in rel.CLOSERS if m in names]
    x = np.where(mask[:, idx], 0.0, np.nan_to_num(cleaned[:, idx]))
    valid = (~mask[:, idx]).astype(float)
    k = np.ones(n)
    num = np.stack([np.convolve(x[:, j] ** 2, k)[: len(x)] for j in range(len(idx))], axis=1)
    den = np.stack([np.convolve(valid[:, j], k)[: len(x)] for j in range(len(idx))], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.sqrt(num / den)
        env[den < 0.5 * n] = np.nan
        db = 20.0 * np.log10(np.maximum(env, 1e-9) / rest_level[idx][None, :])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(db, axis=1)


def _old_detect(t_us, e_db, p):
    from scipy.ndimage import median_filter

    t = np.asarray(t_us, dtype=np.int64)
    fs = 1e6 / np.median(np.diff(t))
    lo, hi = int(round(p.base_lo_s * fs)), int(round(p.base_hi_s * fs))
    hold, refr = max(int(round(p.hold_s * fs)), 1), int(round(p.refractory_s * fs))
    step = max(int(round(0.01 * fs)), 1)
    ed = e_db[::step]
    w = max((hi - lo) // step, 1)
    filled = np.where(np.isfinite(ed), ed, np.nanmedian(ed) if np.isfinite(ed).any() else 0.0)
    med = median_filter(filled, size=w, mode="nearest", origin=(w - 1) // 2)
    base_d = np.full(ed.size, np.nan)
    shift = lo // step
    base_d[shift:] = med[: ed.size - shift] if shift else med
    base = np.repeat(base_d, step)[: e_db.size]
    below = np.isfinite(e_db) & np.isfinite(base) & (e_db <= base - p.drop_db)
    out, run, last = [], 0, -refr - 1
    for i in range(e_db.size):
        run = run + 1 if below[i] else 0
        if run == hold and i - last > refr:
            out.append(int(t[i]))
            last = i
    return np.asarray(out, dtype=np.int64)


def _old_loop(below, hold, refr):
    out, run, last = [], 0, -refr - 1
    for i in range(below.size):
        run = run + 1 if below[i] else 0
        if run == hold and i - last > refr:
            out.append(i)
            last = i
    return out


@pytest.mark.parametrize("seed", range(6))
def test_b4_decision_runs_equal_old_loop(seed):
    rng = np.random.default_rng(seed)
    below = rng.random(20000) < rng.uniform(0.3, 0.9)
    for hold, refr in ((1, 0), (3, 10), (60, 600), (5, 0)):
        assert rel._decisions(below, hold, refr).tolist() == _old_loop(below, hold, refr)


@pytest.mark.parametrize("seed", range(4))
def test_b4_detect_and_causal_db_equal_old_on_finite_signals(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, 20_000_000, 500, dtype=np.int64)
    amp = np.exp(np.cumsum(rng.normal(0, 0.01, t.size)))          # wandering level
    names = ["masseter_r", "masseter_l", "temporalis_r", "temporalis_l", "digastric_l"]
    x = rng.normal(size=(t.size, 5)) * amp[:, None] * 20
    mask = np.zeros(x.shape, bool)
    mask[rng.integers(0, t.size, 30)] = True                      # isolated masked samples
    rest = np.full(5, 10.0)
    p = ReleaseParams()
    e_new = rel.causal_db(t, x, mask, rest, names, p.window_s)
    e_old = _old_causal_db(t, x, mask, rest, names, p.window_s)
    np.testing.assert_allclose(e_new, e_old, rtol=0, atol=1e-9)
    # The detector on a NaN-free envelope (the first window/2 samples are NaN by design).
    ok = np.flatnonzero(np.isfinite(e_old))
    assert ok[0] < 100 and np.isfinite(e_old[ok[0]:]).all()
    tt, ee = t[ok[0]:], e_old[ok[0]:]
    for drop, hold in ((2.0, 0.02), (3.0, 0.03), (6.0, 0.05)):
        q = replace(p, drop_db=drop, hold_s=hold)
        d_new, d_old = rel.detect(tt, ee, q), _old_detect(tt, ee, q)
        assert d_new.tolist() == d_old.tolist()


def test_e21_detect_has_no_look_ahead_across_masked_stretch():
    fs = 1000.0
    t = (np.arange(0, 15, 1 / fs) * 1e6).astype(np.int64)
    ts = t / 1e6
    e = np.full(t.size, 10.0)
    e[(ts >= 2.0) & (ts < 3.0)] = np.nan                           # masked: no envelope
    e[(ts >= 3.2) & (ts < 3.4)] = 6.0                               # a 4 dB release
    p = ReleaseParams()
    t_cut = 4_000_000
    d1 = rel.detect(t, e, p)
    e2 = e.copy()
    e2[ts >= 5.0] = 0.0                                             # a different future
    d2 = rel.detect(t, e2, p)
    assert d1[d1 < t_cut].tolist() == d2[d2 < t_cut].tolist()
    # The release after the gap is found against the last valid level (10 dB), not the
    # whole-phase median (which the different future pulls to 0 dB).
    assert d1[d1 < t_cut].size == 1 and abs(d1[0] / 1e6 - (3.2 + p.hold_s)) < 0.01
    assert _old_detect(t, e2, p)[_old_detect(t, e2, p) < t_cut].size == 0   # old: looked ahead


def test_e21_no_decision_before_any_valid_baseline():
    fs = 1000.0
    t = (np.arange(0, 4, 1 / fs) * 1e6).astype(np.int64)
    ts = t / 1e6
    e = np.full(t.size, 10.0)
    e[ts < 1.0] = np.nan
    e[(ts >= 1.1) & (ts < 1.25)] = 5.0                              # drop before a baseline exists
    p = ReleaseParams()
    d = rel.detect(t, e, p)
    assert not (d < (1.0 + p.base_lo_s) * 1e6).any()


def test_e21_seed_threaded_to_chance_level(monkeypatch):
    seen = []
    real = rel.chance_level                     # D23: analyse calls chance_level

    def spy(*a, **k):
        seen.append(k.get("seed"))
        return real(*a, **k)

    monkeypatch.setattr(rel, "chance_level", spy)
    t = np.arange(0, 30_000_000, 500, dtype=np.int64)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(t.size, 2)) * 10
    rel.analyse(["masseter_r", "masseter_l"], {"open_close": (t, x, np.zeros(x.shape, bool))},
                np.full(2, 10.0), {"open_close": np.array([5e6, 15e6])}, "s01",
                ReleaseParams(sweep_drop_db=(3.0,), sweep_hold_s=(0.03,)), seed=1234)
    assert seen == [1234]


def test_e18_ensemble_envelope_not_lowered_next_to_mask():
    t = np.arange(0, 20_000_000, 500, dtype=np.int64)
    rng = np.random.default_rng(1)
    x = rng.normal(size=(t.size, 2)) * 10.0                         # constant level, RMS 10
    mask = np.zeros(x.shape, bool)
    events = np.arange(3e6, 18e6, 1.5e6)
    for ev in events:                                               # mask 0.2 s before each event
        mask[(t >= ev - 0.2e6) & (t < ev)] = True
    p = ens.EnsembleParams(n_boot=10)
    rows, arrays = ens.analyse(["masseter_r", "digastric_l"], {"open_close": (t, x, mask)},
                               np.full(2, 10.0), {("open_close", "opening"): events}, "s01", p)
    for role in ("agonist", "antagonist"):
        curve = arrays[f"ensemble__open_close__opening__{role}"].astype(float)
        fin = curve[np.isfinite(curve)]
        # 0 dB everywhere (RMS = rest level); zero-filling dipped up to ~-3 dB at mask edges.
        assert fin.size and fin.min() > -1.0 and fin.max() < 1.0


def test_e18_rest_threshold_pools_runs_instead_of_concatenating():
    """A smooth (correlated) rest signal with many mask gaps: concatenating the runs puts a
    discontinuity at every seam, whose TKEO spikes raised the threshold."""
    rng = np.random.default_rng(2)
    n = int(6 * FS)
    ts = np.arange(n) / FS
    x = 10.0 * np.sin(2 * np.pi * 30.0 * ts) + rng.normal(size=n) * 0.05
    for a in rng.choice(np.arange(200, n - 400), 30, replace=False):
        x[a:a + int(rng.integers(100, 200))] = np.nan               # 50-100 ms masked
    op = OnsetParams(method="tkeo")
    thr = lead._threshold(x, FS, op)
    from jtsmm.emg.onset import _runs

    runs = [x[a:b] for a, b in _runs(np.isfinite(x)) if b - a >= 100]
    pooled = np.concatenate([detector_signal(r, FS, op) for r in runs])
    assert thr == pytest.approx(pooled.mean() + op.k_sd * pooled.std())
    s_cat = detector_signal(x[np.isfinite(x)], FS, op)               # the old concatenation
    assert thr < 0.5 * (s_cat.mean() + op.k_sd * s_cat.std())


def test_e18_detect_onsets_baseline_signal_nan_runs_equal_pooled():
    rng = np.random.default_rng(3)
    base = rng.normal(size=int(2 * FS)) * 5.0
    base_nan = base.copy()
    base_nan[1000:1100] = np.nan
    x = rng.normal(size=int(1.5 * FS)) * 5.0
    x[int(0.8 * FS):] *= 8.0
    on = detect_onsets(x, FS, None, OnsetParams(), baseline_signal=base_nan)
    assert on.size >= 1 and abs(on[0] - int(0.8 * FS)) < 0.02 * FS


def test_a3_short_rest_gives_nan_threshold_and_no_crash():
    rng = np.random.default_rng(4)
    x = np.full(int(5 * FS), np.nan)
    x[100:300] = rng.normal(size=200)                               # 0.1 s of valid rest
    for d in lead.DETECTORS:
        op = OnsetParams(method=d)
        assert np.isnan(lead._threshold(x, FS, op))
        assert np.isnan(baseline_threshold(np.full(50, np.nan), FS, op))
        # detection against such a baseline returns nothing instead of raising
        assert detect_onsets(rng.normal(size=500), FS, None, op, baseline_signal=x[:310]).size == 0
        assert detect_onsets(np.ones(5), FS, None, op, baseline_signal=x).size == 0


def test_a3_mechanical_onsets_all_nan_velocity():
    t = np.arange(0, 2_000_000, 5000, dtype=np.int64)
    nan = np.full(t.size, np.nan)
    th = nan.copy()
    th[::2] = np.linspace(0, 10, th[::2].size)                      # every other sample missing
    base = {"omega_noise": 0.0, "acc_thr": 1.0}
    p = {"vel_frac": 0.05}
    out = lead.mechanical_onsets(t, nan, nan, th, int(t[0]), int(t[-1]), "opening", base, p)
    assert np.isnan(out["t_omega"]) and np.isnan(out["t_opt"])
    om = np.sin(np.linspace(0, np.pi, t.size)) * 100
    acc = nan.copy()
    acc[-5:] = 3.0                                                   # finite only after the peak
    out = lead.mechanical_onsets(t, om, acc, nan, int(t[0]), int(t[-1]), "opening", base, p)
    assert np.isfinite(out["t_omega"]) and np.isnan(out["t_acc"])


# -- E27 causal chain ---------------------------------------------------------------------
def _burst_signal(fs=FS, dur=6.0, t_on=4.0, seed=5):
    rng = np.random.default_rng(seed)
    t = (np.arange(0, dur, 1 / fs) * 1e6).astype(np.int64)
    ts = t / 1e6
    x = rng.normal(size=(t.size, 2)) * 5.0 + 200.0 * np.sin(2 * np.pi * 50.0 * ts)[:, None]
    x[ts >= t_on] += rng.normal(size=(int((ts >= t_on).sum()), 2)) * 60.0
    return t, x, ts


def test_e27_causal_clean_has_no_look_ahead():
    from biomech.emg.causal import causal_clean

    t, x, ts = _burst_signal()
    t0 = 1_000_000                                                   # 1 s warm-up
    keep = t >= t0
    mask = np.zeros((int(keep.sum()), 2), bool)
    mask[4000:4100, 0] = True
    y1 = causal_clean(x, t_us=t, fs=FS, line_hz=50.0, t0_us=t0, mask=mask)
    x2 = x.copy()
    cut = int(np.searchsorted(t, 3_000_000))
    x2[cut:] = np.random.default_rng(9).normal(size=x2[cut:].shape) * 500.0
    y2 = causal_clean(x2, t_us=t, fs=FS, line_hz=50.0, t0_us=t0, mask=mask)
    k = cut - int((~keep).sum())                                     # index in the kept rows
    np.testing.assert_array_equal(y1[:k], y2[:k])
    assert not np.allclose(y1[k:], y2[k:])
    assert np.isnan(y1[4000:4100, 0]).all() and np.isfinite(y1[4000:4100, 1]).all()
    assert y1.shape == (int(keep.sum()), 2)


def test_e27_causal_envelope_lags_zero_phase():
    from biomech.emg.causal import causal_clean
    from biomech.emg.pipeline import bandpass, rms_envelope

    t, x, ts = _burst_signal(t_on=4.0)
    t0 = 2_000_000
    xc = causal_clean(x, t_us=t, fs=FS, line_hz=50.0, t0_us=t0)
    tk = ts[t >= t0]
    xz = bandpass(x - 200.0 * np.sin(2 * np.pi * 50.0 * ts)[:, None], FS)[t >= t0]
    ec, ez = rms_envelope(xc, FS, 0.05), rms_envelope(xz, FS, 0.05)
    half = lambda e: tk[np.flatnonzero(e[:, 0] > 0.5 * (np.median(e[tk > 4.5, 0])))[0]]  # noqa: E731
    assert half(ec) > half(ez)
    # Causal output before the burst does not know the burst.
    pre = (tk > 3.0) & (tk < 3.999)
    assert np.std(xc[pre, 0]) < 10.0


def _a6_errors(burst_sd: float, n_rep: int = 40) -> np.ndarray:
    from biomech.emg.pipeline import bandpass

    errs = []
    for seed in range(n_rep):
        rng = np.random.default_rng(100 + seed)
        n, on = int(3 * FS), int(2 * FS)
        x = rng.normal(size=n) * 5.0
        x[on:] += rng.normal(size=n - on) * burst_sd
        xb = bandpass(x[:, None], FS)[:, 0]
        k = detect_onsets(xb[int(1.2 * FS):], FS, slice(0, int(0.6 * FS)), OnsetParams())
        if k.size:
            errs.append((k[0] + int(1.2 * FS) - on) / FS * 1e3)
    return np.asarray(errs)


def test_a6_zero_phase_tkeo_onset_spread():
    """A6 (measurement): offline onset (zero-phase 20-450 Hz band-pass + TKEO, k = 3 SD of a
    0.6-s quiet baseline) minus the true step onset of a white-noise burst in white noise,
    2 kHz. Negative = detected early (zero-phase pre-ringing + TKEO refinement window)."""
    for sd, label in ((50.0, "20 dB"), (15.8, "10 dB")):
        e = _a6_errors(sd)
        print(f"A6 burst SNR {label}: onset error median {np.median(e):.1f} ms, "
              f"IQR [{np.percentile(e, 25):.1f}, {np.percentile(e, 75):.1f}] ms, "
              f"range [{e.min():.1f}, {e.max():.1f}] ms, early in {np.mean(e < 0):.0%}, "
              f"n={e.size}")
        assert e.size >= 36
        assert np.median(e) <= 0.5 and np.median(e) > -15.0
