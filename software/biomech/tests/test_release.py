# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""X3 causal release detector: latency and false alarms on constructed envelopes."""

from __future__ import annotations

import numpy as np

from biomech.timing.release import ReleaseParams, detect, evaluate


def _db(t_s, levels):
    e = np.full(t_s.size, levels[0][1])
    for t0, v in levels[1:]:
        e[t_s >= t0] = v
    return e


def test_release_detected_with_causal_latency():
    fs = 1000.0
    t = np.arange(0, 6, 1 / fs)
    e = _db(t, [(0, 20.0), (2.0, 10.0), (2.8, 20.0), (4.0, 10.0)])   # two drops of 10 dB
    p = ReleaseParams()
    dec = detect((t * 1e6).astype(np.int64), e, p)
    # decision = drop + hold (causal, the run must last hold_s); refractory keeps one per drop
    np.testing.assert_allclose(dec / 1e6, [2.0 + p.hold_s - 1 / fs, 4.0 + p.hold_s - 1 / fs],
                               atol=2e-3)
    rows, false = evaluate(dec, np.array([2.1e6, 4.1e6]), p)
    assert all(r["detected"] for r in rows) and false == 0
    np.testing.assert_allclose([r["anticipation_ms"] for r in rows], [71.0, 71.0], atol=2)


def test_no_release_without_a_drop_and_false_alarm_counted():
    fs = 1000.0
    t = np.arange(0, 4, 1 / fs)
    rng = np.random.default_rng(0)
    e = 20.0 + rng.normal(0, 0.3, t.size)                           # steady activity
    p = ReleaseParams()
    assert detect((t * 1e6).astype(np.int64), e, p).size == 0
    e2 = _db(t, [(0, 20.0), (1.5, 10.0)])
    dec = detect((t * 1e6).astype(np.int64), e2, p)
    _, false = evaluate(dec, np.array([3.5e6]), p)                  # no opening near 1.5 s
    assert dec.size == 1 and false == 1


def test_chance_level_reflects_false_alarm_rate():
    from biomech.timing.release import chance_detection

    p = ReleaseParams()
    dec = np.arange(0, 60e6, 1e6).astype(np.int64)          # one decision per second
    hi = chance_detection(dec, np.array([30e6]), 0.0, 60e6, p)
    lo = chance_detection(np.array([30e6 - 1e5], np.int64), np.array([30e6]), 0.0, 60e6, p)
    assert hi > 0.7 and lo == 0.0
