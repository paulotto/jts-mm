# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The vectorised display filters equal the per-sample recursions they replaced."""

from __future__ import annotations

import math

import numpy as np

from jtsmm.viewer.processing import (
    Biquad,
    CausalEmgFilter,
    HarmonicCanceller,
    _highpass,
    _lowpass,
    _notch,
    line_notch_freqs,
)


def _biquad_ref(bq: Biquad, x, z1, z2):
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        xi = x[i]
        yi = bq.b0 * xi + z1
        z1 = bq.b1 * xi - bq.a1 * yi + z2
        z2 = bq.b2 * xi - bq.a2 * yi
        out[i] = yi
    return out, z1, z2


def _canceller_ref(hc: HarmonicCanceller, x, st):
    th, dth, a = st["th"].copy(), hc._dtheta.copy(), hc._alpha
    i1, q1, i2, q2 = (st[k].copy() for k in ("i1", "q1", "i2", "q2"))
    out = np.empty_like(x, dtype=np.float64)
    for n in range(x.shape[0]):
        sn = np.sin(th)[:, None]
        cs = np.cos(th)[:, None]
        xi = x[n]
        out[n] = xi - 2.0 * (i2 * sn + q2 * cs).sum(axis=0)
        i1 += a * (xi[None, :] * sn - i1)
        q1 += a * (xi[None, :] * cs - q1)
        i2 += a * (i1 - i2)
        q2 += a * (q1 - q2)
        th += dth
    return out, {"th": np.mod(th, 2 * math.pi), "i1": i1, "q1": q1, "i2": i2, "q2": q2}


def _signal(n, c=7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 2000.0
    return (rng.normal(0, 20, (n, c)) + 800 * np.sin(2 * np.pi * 50.013 * t)[:, None]
            + 150 * np.sin(2 * np.pi * 150.04 * t + 1)[:, None] + 20_000.0)


def _close(a, b, scale):
    assert np.max(np.abs(a - b)) <= 1e-9 * max(1.0, scale), np.max(np.abs(a - b))


def test_biquads_match_reference_across_odd_block_sizes():
    x = _signal(5000)
    for bq in (_highpass(2000.0, 20.0, math.sqrt(0.5), 7), _lowpass(2000.0, 450.0, 0.7071, 7),
               _notch(2000.0, 50.0, 10.0, 7), _notch(2000.0, 400.0, 80.0, 7)):
        z1 = np.zeros(7)
        z2 = np.zeros(7)
        s = 0
        for size in (1, 7, 63, 64, 65, 200, 1000, 3600):
            blk = x[s : s + size]
            want, z1, z2 = _biquad_ref(bq, blk, z1, z2)
            got = bq.process(blk)
            _close(got, want, np.max(np.abs(want)))
            s += size
        _close(bq._z1, z1, 1e5)
        _close(bq._z2, z2, 1e5)


def test_harmonic_canceller_matches_reference_including_relock():
    fs = 2000.0
    x = _signal(12_000, seed=1)
    hc = HarmonicCanceller(fs, 7, line_notch_freqs(fs, 50.0))
    st = {"th": hc._theta.copy(), "i1": hc._i.copy(), "q1": hc._q.copy(),
          "i2": hc._i2.copy(), "q2": hc._q2.copy()}
    s = 0
    for size in (5, 64, 129, 500, 40, 2000, 700, 3000, 5562):
        blk = x[s : s + size]
        want, st = _canceller_ref(hc, blk, st)
        got = hc.process(blk)
        _close(got, want, np.max(np.abs(want)))
        # The relock step (run inside process) retunes dtheta identically for both.
        s += size
    assert hc.locked_offset_hz != 0.0
    _close(hc._i2, st["i2"], 1e5)


def test_full_chain_equals_blockwise_reference_order():
    """Chunking must not change an LTI chain's output (state carried exactly).

    (The tracking canceller retunes at block ends by design, so the notch chain
    is the LTI case; the canceller itself is compared sample-exact above.)
    """
    x = _signal(8000, seed=2)
    a = CausalEmgFilter(2000.0, 7, mains_mode="notch")
    b = CausalEmgFilter(2000.0, 7, mains_mode="notch")
    fa, ea = a.process(x)
    parts = [b.process(x[i : i + 37]) for i in range(0, x.shape[0], 37)]
    fb = np.vstack([p[0] for p in parts])
    eb = np.vstack([p[1] for p in parts])
    assert np.max(np.abs(fa - fb)) < 1e-3          # float32 outputs
    assert np.max(np.abs(ea - eb)) < 1e-3


def test_hub_decimation_matches_the_bucket_loop():
    from jtsmm.viewer.hub import Hub

    rng = np.random.default_rng(4)
    x = rng.normal(0, 50, (1003, 7)).astype(np.float32)
    t = (np.arange(1003) * 500).astype(np.uint64)
    hub = Hub(emg_fs=2000.0, emg_lsb_uv=0.1, n_channels=7, display_fs=500.0,
              filter_enabled=False)
    hub.append_emg_raw(t, x)
    snap = hub.snapshot()
    d = 4
    want_lo = np.stack([x[b : b + d].min(axis=0) for b in range(0, 1003, d)])
    want_hi = np.stack([x[b : b + d].max(axis=0) for b in range(0, 1003, d)])
    assert np.array_equal(snap.emg_lo, want_lo)
    assert np.array_equal(snap.emg_hi, want_hi)
    assert snap.emg_t.shape[0] == want_lo.shape[0]
