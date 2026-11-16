# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pseudo-online evaluation (ml.online): causality, chunk invariance, offline equivalence,
metric definitions, smoothing. Small synthetic takes (1 kHz EMG, 100 Hz IMU)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ml.config.classes import MUSCLE_CHANNEL_NAMES
from ml.core.dataset import Imu, WindowConfig, WindowDataset, build_windows
from ml.online import decoder as dc
from ml.online import metrics as mt
from ml.online.stream import (CausalChain, StreamConfig, StreamInput, causal_dilate,
                              causal_flatline, replay)
from ml.protocols import evaluate as ev

FS = 1000.0
IMU_HZ = 100.0
OPT_HZ = 40.0
GAIN = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 8, 8, 1, 1], [6, 6, 1, 1, 6, 6],
                 [3, 1, 3, 1, 3, 1], [1, 3, 1, 3, 1, 3], [8, 1, 1, 1, 1, 8],
                 [1, 8, 1, 1, 8, 1]], dtype=float)
# phase, start (s), duration (s), classes cycled every 0.5 s
PLAN = (("rest", 2.5, 3.0, (0,)), ("open_close", 6.0, 4.0, (0, 1, 0, 2)),
        ("left_right", 10.5, 4.0, (0, 5, 0, 6)), ("free_motion", 15.0, 2.5, (0, 1, 0, 5)))
T_END_S = 18.0
WCFG = WindowConfig()


def _labels():
    out = {}
    opt = np.arange(0, int(T_END_S * 1e6), int(1e6 / OPT_HZ), dtype=np.int64)
    for name, a, d, classes in PLAN:
        t0, t1 = int(a * 1e6), int((a + d) * 1e6)
        t = opt[(opt >= t0) & (opt <= t1)]
        out[name] = (t, np.asarray(classes)[((t - t0) // 500_000) % len(classes)].astype(np.int64))
    return out


def _session(seed: int = 0, hum: float = 30.0) -> tuple[StreamInput, dict]:
    rng = np.random.default_rng(seed)
    t = np.arange(500_000, int(T_END_S * 1e6), int(1e6 / FS), dtype=np.int64)  # rest - 2 s
    labels = _labels()
    cls = np.zeros(t.size, dtype=int)
    for lt, ly in labels.values():
        j = np.searchsorted(lt, t, side="right") - 1
        inside = (j >= 0) & (t >= lt[0]) & (t <= lt[-1] + 25_000)
        cls[inside] = ly[j[inside]]
    emg = rng.normal(size=(t.size, 6)) * 10.0 * GAIN[cls]
    emg += hum * np.sin(2 * np.pi * 50.0 * t[:, None] * 1e-6 + np.arange(6)) + 40.0
    ti = np.arange(0, int(T_END_S * 1e6), int(1e6 / IMU_HZ), dtype=np.int64)
    ci = cls[np.clip(np.searchsorted(t, ti), 0, t.size - 1)]
    ang = np.cumsum(np.select([ci == 1, ci == 2, ci == 5, ci == 6], [0.01, -0.01, 0.02, -0.02],
                              0.0))
    streams = {}
    for slot, k in (("mta", 1.0), ("cra", 0.1)):
        q = np.column_stack([np.cos(k * ang / 2), np.sin(k * ang / 2), np.zeros_like(ang),
                             np.zeros_like(ang)])
        a = np.tile([0.0, 0.0, 9.80665], (ti.size, 1)) + rng.normal(size=(ti.size, 3)) * 0.05
        streams[slot] = (q, a)
    imu = Imu((ti, streams["mta"][0]), (ti, streams["cra"][0]), (ti + 3000, streams["mta"][1]),
              (ti + 3000, streams["cra"][1]))
    phases = [(n, int(a * 1e6), int((a + d) * 1e6)) for n, a, d, _ in PLAN]
    inp = StreamInput(t_us=t, uv=emg, fs=FS, channels=list(MUSCLE_CHANNEL_NAMES), phases=phases,
                      line_hz=50.0, imu=imu)
    return inp, labels


def _predict(xe, xi):
    """A fixed decision rule on the features (stands in for a fitted model)."""
    v = np.r_[xe[::11][:6], 0.0]
    return int(np.argmax(v)) if np.isfinite(v).all() else 0


@pytest.fixture(scope="module")
def take():
    return _session(0)


@pytest.fixture(scope="module")
def stream20(take):
    inp, _ = take
    return replay(inp, WCFG, StreamConfig(chunk_s=0.02), predict=_predict,
                  release_params=_release_params())


def _release_params():
    from biomech.timing.release import ReleaseParams

    return ReleaseParams()


# -- causal pieces -------------------------------------------------------------------------
def test_causal_dilate_hold_only_and_carried():
    hit = np.zeros((20, 2), bool)
    hit[5, 0] = True
    hit[18, 1] = True
    m, carry = causal_dilate(hit, 3)
    assert np.flatnonzero(m[:, 0]).tolist() == [5, 6, 7, 8]       # never sample 4
    assert np.flatnonzero(m[:, 1]).tolist() == [18, 19]
    m2, _ = causal_dilate(np.zeros((4, 2), bool), 3, carry)        # continues into next block
    assert np.flatnonzero(m2[:, 1]).tolist() == [0, 1]
    assert not m2[:, 0].any()
    # Same result in one block or in pieces.
    big = np.random.default_rng(1).random((300, 3)) < 0.01
    whole, _ = causal_dilate(big, 7)
    parts, c = [], None
    for a in range(0, 300, 17):
        p, c = causal_dilate(big[a:a + 17], 7, c)
        parts.append(p)
    assert np.array_equal(np.vstack(parts), whole)


def test_causal_flatline_marks_only_after_the_stretch():
    x = np.random.default_rng(0).normal(size=(400, 1))
    x[100:300] = 2.0
    m = causal_flatline(np.zeros((0, 1)), x, 1000.0, 0.05)
    assert not m[:149].any() and m[149:300].all() and not m[301:].any()


def test_causal_chain_equals_offline_chain_with_gaps():
    from ml.core.signals import causal_emg

    rng = np.random.default_rng(3)
    t = np.arange(0, 6_000_000, 1000, dtype=np.int64)
    keep = np.ones(t.size, bool)
    keep[2100:2140] = False                      # a 40-ms recording gap
    keep[4000:4005] = False
    t = t[keep]
    x = rng.normal(size=(t.size, 3)) * 20 + 50 * np.sin(2 * np.pi * 50.03 * t[:, None] * 1e-6)
    off = causal_emg(t, x, FS, 50.0)
    for n in (7, 20, 64, 333):
        ch = CausalChain(FS, 3, 50.0)
        on = np.vstack([ch.process(t[a:a + n], x[a:a + n]) for a in range(0, t.size, n)])
        np.testing.assert_allclose(on, off, rtol=0, atol=1e-4)


# -- (1) chunk-size invariance --------------------------------------------------------------
def test_chunk_size_invariance(take, stream20):
    inp, labels = take
    ref = stream20
    for chunk in (0.01, 0.05):
        r = replay(inp, WCFG, StreamConfig(chunk_s=chunk), predict=_predict,
                   release_params=_release_params())
        assert np.array_equal(r.windows["t_end_us"], ref.windows["t_end_us"])
        np.testing.assert_allclose(r.X_emg, ref.X_emg, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(r.X_imu, ref.X_imu, rtol=0, atol=1e-12)
        assert np.array_equal(r.windows["yhat_inloop"], ref.windows["yhat_inloop"])
        y = r.attach_labels(labels, WCFG)
        s_r = dc.smooth(r.windows["yhat_inloop"], r.windows["yhat_inloop"] >= 0,
                        r.windows["phase"])
        s_0 = dc.smooth(ref.windows["yhat_inloop"], ref.windows["yhat_inloop"] >= 0,
                        ref.windows["phase"])
        assert np.array_equal(s_r, s_0) and y.size == s_r.size
        assert np.array_equal(r.release_us, ref.release_us)
        # Only the emission time depends on the chunking (it is the chunk's last sample).
        d = (r.windows["t_dec_us"] - r.windows["t_end_us"])[~r.windows["retro"]]
        assert d.min() >= 0 and d.max() <= chunk * 1e6 + 20_000


# -- (2) no lookahead ------------------------------------------------------------------------
def test_no_lookahead(take, stream20):
    inp, _ = take
    seen = {}

    def hook(proc, ch):
        if ch.index == CUT:
            seen["n"] = len(proc.out["t_end_us"])
            seen["T"] = int(ch.t[-1])
            seen["rel"] = list(proc.release.decisions) if proc.release else []

    CUT = int(11.2 / 0.02)                       # inside left_right, after the calibration
    a = replay(inp, WCFG, StreamConfig(), predict=_predict, release_params=_release_params(),
               on_chunk=hook)
    T, n = seen["T"], seen["n"]
    rng = np.random.default_rng(9)
    after = inp.t_us > T
    uv = inp.uv.copy()
    uv[after] = rng.normal(size=(int(after.sum()), uv.shape[1])) * 500.0
    lead = np.zeros(uv.shape, bool)
    lead[after & (inp.t_us < T + 300_000)] = True
    imu = inp.imu

    def bend(s):
        tt, v = s
        v = v.copy()
        v[tt > T] = v[tt > T][::-1]
        return tt, v
    pert = replace(inp, uv=uv, leadoff=lead,
                   imu=Imu(bend(imu.q_mta), bend(imu.q_cra), bend(imu.a_mta), bend(imu.a_cra)))
    b = replay(pert, WCFG, StreamConfig(), predict=_predict, release_params=_release_params())
    # Windows emitted up to chunk CUT (in emission order) are identical, bit for bit.
    ea = np.argsort(a.windows["chunk"], kind="stable")
    eb = np.argsort(b.windows["chunk"], kind="stable")
    sel_a = ea[a.windows["chunk"][ea] <= CUT]
    sel_b = eb[b.windows["chunk"][eb] <= CUT]
    assert sel_a.size == n and sel_b.size == n
    assert np.array_equal(a.windows["t_end_us"][sel_a], b.windows["t_end_us"][sel_b])
    assert np.array_equal(a.X_emg[sel_a], b.X_emg[sel_b], equal_nan=True)
    assert np.array_equal(a.X_imu[sel_a], b.X_imu[sel_b], equal_nan=True)
    assert np.array_equal(a.windows["yhat_inloop"][sel_a], b.windows["yhat_inloop"][sel_b])
    assert [d for d in b.release_us if d <= T] == seen["rel"]
    # ... and the perturbation did change later outputs (the test has teeth).
    later_a = ea[a.windows["chunk"][ea] > CUT][:20]
    later_b = eb[b.windows["chunk"][eb] > CUT][:20]
    assert not np.allclose(a.X_emg[later_a], b.X_emg[later_b], equal_nan=True)
    assert stream20.calibration.t_us < T


def test_calibration_uses_rest_only(take, stream20):
    inp, _ = take
    c = stream20.calibration
    rest = inp.phases[0]
    assert c.rest_precedes_tasks and c.error == ""
    assert rest[2] <= c.t_us < inp.phases[1][1]
    w = stream20.windows
    assert w["retro"][w["phase"] == "rest"].all() and not w["retro"][w["phase"] != "rest"].any()
    assert (w["t_dec_us"][w["phase"] == "rest"] == c.t_us).all()


# -- (3) stream windows == offline windows on the same causal chain --------------------------
def _offline(inp, labels, ref):
    return build_windows(pid="x", session="x", phases=inp.phases, emg_t=inp.t_us, emg_uv=inp.uv,
                         emg_fs=inp.fs, line_hz=inp.line_hz, ref_uv=ref, masks={},
                         channels=inp.channels, labels=labels, imu=inp.imu, cfg=WCFG)


def test_stream_features_equal_offline_windows(take):
    inp, labels = take
    r = replay(inp, WCFG, StreamConfig(filter_mode="phase_warmup"))
    off = _offline(inp, labels, r.calibration.ref_uv)
    y = r.attach_labels(labels, WCFG)
    ts = r.windows["t_end_us"]
    pos = np.searchsorted(ts, off.t_end_us)
    assert np.array_equal(ts[pos], off.t_end_us)                 # every offline window streamed
    np.testing.assert_allclose(r.X_emg[pos], off.blocks["emg"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(r.X_imu[pos], off.blocks["imu"], rtol=0, atol=1e-9)
    assert np.array_equal(y[pos], off.y)
    np.testing.assert_allclose(r.calibration.center, off.meta["norm"]["emg"]["center"], atol=1e-9)
    np.testing.assert_allclose(r.calibration.thr, off.meta["thr"], rtol=1e-12)
    # The streamed windows the offline builder dropped are exactly the undecodable/unlabelled.
    keep = np.zeros(ts.size, bool)
    keep[pos] = True
    assert not (r.decodable() & (y >= 0) & ~keep).any()


def test_continuous_filter_deviation_is_a_phase_start_transient(take, stream20):
    """Intended difference: one filter from the stream start instead of a fresh filter 2 s
    before each phase; the offline mains canceller has not fully converged at a phase start."""
    inp, labels = take
    off = _offline(inp, labels, stream20.calibration.ref_uv)
    ts = stream20.windows["t_end_us"]
    pos = np.searchsorted(ts, off.t_end_us)
    z = np.abs(stream20.X_emg[pos] - off.blocks["emg"]) / np.maximum(off.blocks["emg"].std(0),
                                                                    1e-9)
    t0 = {n: a for n, a, _ in inp.phases}
    age = (off.t_end_us - np.array([t0[p] for p in off.phase])) / 1e6
    assert np.nanmax(z[(off.phase == "rest")]) < 1e-9          # same start for the rest phase
    assert np.percentile(z[age > 2.0], 99) < 0.05


# -- (4) metrics on streams with known onsets ---------------------------------------------------
def _known_stream():
    t_end = np.arange(1_000_000, 6_000_000, 50_000, dtype=np.int64)
    t_dec = t_end + 10_000
    phase = np.array(["open_close"] * t_end.size)
    lt = np.arange(1_000_000, 6_000_000, 25_000, dtype=np.int64)
    ly = np.zeros(lt.size, dtype=np.int64)
    ly[(lt >= 2_000_000) & (lt < 2_500_000)] = 1             # onsets at 2.0 s and 4.0 s
    ly[(lt >= 4_000_000) & (lt < 4_600_000)] = 2
    y = ly[np.searchsorted(lt, t_end, side="right") - 1]
    s = np.zeros(t_end.size, dtype=np.int64)
    s[(t_end >= 2_100_000) & (t_end < 2_600_000)] = 1          # +100 ms (+10 ms emission)
    s[(t_end >= 3_200_000) & (t_end < 3_300_000)] = 5          # a false onset at 3.2 s
    return t_end, t_dec, phase, y, s, {"open_close": (lt, ly)}


def test_decision_latency_and_detection():
    t_end, t_dec, phase, y, s, labels = _known_stream()
    on = mt.label_onsets(labels)
    assert [(o["t_on_us"], o["cls"], o["t_rest_us"]) for o in on] == [
        (2_000_000, 1, 1_000_000), (4_000_000, 2, 2_500_000)]
    lat = mt.decision_latency(on, t_end, t_dec, s, phase, np.ones(t_end.size, bool), 0.5)
    assert lat[0]["detected"] and lat[0]["latency_ms"] == pytest.approx(110.0)
    assert lat[0]["latency_end_ms"] == pytest.approx(100.0)
    assert not lat[1]["detected"] and np.isnan(lat[1]["latency_ms"])
    # A state held over from the previous movement is not a detection of the next onset.
    s_held = s.copy()
    s_held[(t_end >= 3_500_000) & (t_end < 4_500_000)] = 2
    s_held[(t_end >= 2_400_000) & (t_end < 3_500_000)] = 0
    lat_h = mt.decision_latency(on, t_end, t_dec, s_held, phase, np.ones(t_end.size, bool), 0.5)
    assert lat_h[1]["latency_ms"] == pytest.approx(-490.0)       # entry at 3.5 s (lo = 3.5 s)
    s_held[(t_end >= 3_400_000) & (t_end < 3_500_000)] = 2       # entered before the window
    lat_h = mt.decision_latency(on, t_end, t_dec, s_held, phase, np.ones(t_end.size, bool), 0.5)
    assert not lat_h[1]["detected"]
    # The window-end latency is ml_eval's definition (evaluate.latency).
    rows = np.arange(t_end.size)
    assert mt.window_latency(t_end, y, s, phase, rows) == ev.latency(t_end, y, s)
    assert mt.window_latency(t_end, y, s, phase, rows)[0] == pytest.approx(100.0)


def test_false_activations_and_event_kinds():
    t_end, t_dec, _, y, s, labels = _known_stream()
    phase = np.array(["free_motion"] * t_end.size)
    on = mt.label_onsets({"free_motion": labels["open_close"]})
    ev_idx = dc.onsets(s, phase)
    assert t_end[ev_idx].tolist() == [2_100_000, 3_200_000]
    fa = mt.false_activations_free(ev_idx, t_end, y, phase, np.ones(t_end.size, bool),
                                   np.ones(t_end.size, bool), on, 0.05, guard_s=0.5)
    rest_min = (y == 0).sum() * 0.05 / 60
    assert fa["fa_free_n"] == 1 and fa["fa_per_min_free"] == pytest.approx(1 / rest_min)
    lat = mt.decision_latency(on, t_end, t_dec, s, phase, np.ones(t_end.size, bool))
    kinds = mt.classify_events(ev_idx, {r["hit"] for r in lat if r["detected"]}, t_end, y, on)
    assert kinds == ["hit", "false"]
    # An onset 0.3 s before a labelled onset is early, not false.
    s2 = s.copy()
    s2[(t_end >= 3_700_000) & (t_end < 3_800_000)] = 2
    ev2 = dc.onsets(s2, phase)
    fa2 = mt.false_activations_free(ev2, t_end, y, phase, np.ones(t_end.size, bool),
                                    np.ones(t_end.size, bool), on, 0.05, guard_s=0.5)
    assert fa2["fa_free_n"] == 1
    r = mt.rest_false_activations(t_end, s, np.arange(t_end.size))
    assert r == ev.false_activations(t_end, s)


def test_release_metrics_known_decisions():
    from biomech.timing.release import ReleaseParams, chance_detection, evaluate

    p = ReleaseParams()
    onsets = np.array([3e6, 6e6, 9e6])
    dec = np.array([2.85e6, 6.1e6, 12.0e6], dtype=np.int64)    # 150 ms early, 100 ms late, false
    rows = mt.release_metrics(dec, {"open_close": (0, 15_000_000)}, {"open_close": onsets}, p)
    r = rows[0]
    assert r["n_openings"] == 3 and r["detected_frac"] == pytest.approx(2 / 3)
    assert r["anticipation_ms"] == pytest.approx(np.median([150.0, -100.0]))
    assert r["n_false"] == 1 and r["false_per_min"] == pytest.approx(1 / 0.25)
    # E42: the rhythmic tasks take the circular-shift chance level as offline, the others
    # pseudo-onsets
    assert r["chance_detected_frac"] == chance_detection(dec, onsets, 0.0, 15e6, p, shift=True)
    other = mt.release_metrics(dec, {"protrusion_retrusion": (0, 15_000_000)},
                               {"protrusion_retrusion": onsets}, p)[0]
    assert other["chance_detected_frac"] == chance_detection(dec, onsets, 0.0, 15e6, p)
    assert evaluate(dec, onsets, p)[1] == 1


def test_stream_release_equals_batch_detector():
    from biomech.timing.release import ReleaseParams, causal_db, detect

    rng = np.random.default_rng(5)
    fs = 2000.0
    t = np.arange(0, 20_000_000, 500, dtype=np.int64)
    amp = np.ones(t.size)
    for k in range(1, 9):                                      # bursts with sharp releases
        amp[(t > k * 2.2e6) & (t < k * 2.2e6 + 1.2e6)] = 6.0
    x = rng.normal(size=(t.size, 4)) * amp[:, None] * 10
    names = ["masseter_r", "masseter_l", "temporalis_r", "temporalis_l"]
    rest = np.full(4, 10.0)
    p = ReleaseParams()
    batch = detect(t, causal_db(t, x, np.zeros(x.shape, bool), rest, names, p.window_s), p)
    from ml.online.release import StreamRelease

    for n in (37, 40, 400):
        sr = StreamRelease(fs, rest, p)
        for a in range(0, t.size, n):
            sr.feed(t[a:a + n], x[a:a + n])
        assert sr.decisions == batch.tolist()
    assert len(batch) >= 5
    # Masked stretches (all closers NaN), also at the start: last valid value held (E21).
    xm = x.copy()
    for a0, a1 in ((0, 1500), (9000, 9600), (21000, 21900), (30500, 30700)):
        xm[a0:a1] = np.nan
    mask = ~np.isfinite(xm)
    batch_m = detect(t, causal_db(t, np.nan_to_num(xm), mask, rest, names, p.window_s), p)
    for n in (37, 400):
        sr = StreamRelease(fs, rest, p)
        for a in range(0, t.size, n):
            sr.feed(t[a:a + n], xm[a:a + n])
        assert sr.decisions == batch_m.tolist()
    assert len(batch_m) >= 4


# -- (5) smoothing and hysteresis --------------------------------------------------------------
def test_smoothing_majority_and_hysteresis():
    seg = np.zeros(12, dtype=int)
    ok = np.ones(12, bool)
    yhat = np.array([0, 1, 0, 1, 1, 0, 0, 0, 2, 2, 1, 2])
    s = dc.smooth(yhat, ok, seg, dc.Smoothing(3, 2, 1))
    # enter OPEN with 2 of 3; hold while >= 1 of 3; CLOSE wins with 2 of 3 and more than OPEN.
    assert s.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2]
    # A single stray decision never activates; an undecodable window holds the state.
    assert dc.smooth(np.array([0, 3, 0, 0, 4, 0]), np.ones(6, bool), np.zeros(6)).max() == 0
    held = dc.smooth(np.array([1, 1, -1, -1, -1, 0, 0, 0]), np.array([1, 1, 0, 0, 0, 1, 1, 1],
                                                                      bool), np.zeros(8))
    assert held.tolist() == [0, 1, 1, 1, 1, 1, 1, 0]
    # The state starts at REST in every phase.
    s2 = dc.smooth(np.array([1, 1, 1, 1]), np.ones(4, bool), np.array([0, 0, 1, 1]))
    assert s2.tolist() == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        dc.Smoothing.from_dict({"n": 3, "enter": 1, "hold": 2})


def test_onsets_include_class_switches():
    s = np.array([0, 1, 1, 2, 2, 0, 1])
    assert dc.onsets(s, np.zeros(7)).tolist() == [1, 3, 6]


# -- decoder plumbing -----------------------------------------------------------------------------
def test_fset_columns_match_dataset_matrix():
    names = {"emg": [f"{c}|log_rms" for c in MUSCLE_CHANNEL_NAMES],
             "imu": [f"i{k}" for k in range(3)]}
    rng = np.random.default_rng(0)
    E, M = rng.normal(size=(5, 6)), rng.normal(size=(5, 3))
    one = np.zeros(5, dtype=np.int64)
    ds = WindowDataset(blocks={"emg": E, "imu": M}, y=one, pid=np.full(5, "a"),
                       session=np.full(5, "a"), phase=np.full(5, "x"), t_end_us=one,
                       groups=one, names=names)
    for f in ("emg", "imu", "emg_imu", "emg_nodig", "emg_nodig_imu"):
        X, _ = dc.stream_matrix(E, M, names, f)
        assert np.array_equal(X, ds.matrix(f)[0])


class _Const:
    def __init__(self, c):
        self.c = c

    def predict(self, X):
        return np.full(len(X), self.c)


def test_calibration_plans_are_causal(stream20, take):
    _, labels = take
    w = stream20.windows
    y = stream20.attach_labels(labels, WCFG)
    dec = stream20.decodable()
    cp = dc.CalibrationPlan(first_k=10, per_phase_k=5)
    for mode in ("first", "per_phase"):
        mp = dc.plan(mode, w["t_end_us"], w["phase"], y, dec, cp)
        assert mp.applicable and mp.portion.any()
        assert all(np.diff(mp.t_switch) > 0)
        for ts, cal in zip(mp.t_switch, mp.cal):
            assert (w["t_end_us"][cal] < ts).all()                 # calibration precedes use
            assert dec[cal].all() and (y[cal] >= 0).all()
        # Scored windows come after the switch that calibrated their phase.
        assert (w["t_end_us"][mp.portion] > mp.t_switch[0]).all()
    mp = dc.plan("per_phase", w["t_end_us"], w["phase"], y, dec, cp)
    assert [len(c) for c in mp.cal] == [5 * (i + 1) for i in range(len(mp.cal))]
    models = [_Const(i) for i in range(len(mp.cal) + 1)]
    out = dc.decode(models, mp, np.zeros((len(y), 1)), w["t_end_us"], dec)
    which = np.searchsorted(mp.t_switch, w["t_end_us"], side="left")
    assert np.array_equal(out[dec], which[dec]) and (out[~dec] == -1).all()
    first = dc.plan("first", w["t_end_us"], w["phase"], y, dec, cp)
    assert set(w["phase"][first.cal[0]]) == {"open_close"}          # after the rest block


def test_line_hz_finds_a_60_hz_grid():
    from ml.core import signals as S

    fs = 2000.0
    t = np.arange(int(20 * fs)) / fs
    rng = np.random.default_rng(1)
    x = (40 * np.sin(2 * np.pi * 60.02 * t) + rng.normal(0, 3, t.size))[:, None]
    assert abs(S.line_hz(x, fs, (t * 1e6).astype(np.int64), 50.0) - 60.02) < 0.05
