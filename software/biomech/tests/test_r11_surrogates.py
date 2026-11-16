# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""(X6 code-quality rule "one surrogate engine"): the D23 chance null and the
X3 chance level run on :mod:`biomech.timing.surrogates`; their values are byte-identical to
the code before the refactor.

The reference digests below were computed from the pre-refactor code (commit d027a80,
2026-10-30) on the self-contained generator :func:`_participant` (independent of
``testing.synth``, so later simulator changes cannot move them). A digest is the SHA-256 of
the table written with full float precision.
"""

from __future__ import annotations

import functools
import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from biomech.io.session import CLOSERS, OPENERS
from biomech.timing import lead as L
from biomech.timing import release as rel
from biomech.timing import surrogates as S

NAMES = list(OPENERS + CLOSERS)
FS = 1000.0
P = {"tasks": ["open_close", "chewing"], "onset_rule": "drive", "vel_frac": 0.05,
     "acc_k": 3.0, "pre_s": 0.3, "emg_pre_s": 0.6, "emg_post_s": 0.2, "k_sd": 3.0,
     "burst_min_db": 6.0,
     "ensemble": {"n_boot": 20},
     "release": {}}


def _participant(seed: int = 3):
    """(prep, imu_arrays, kin_cycles): rest 8 s, open_close 20 s (1.4 s period), chewing
    12 s (0.85 s period); opener/closer bursts 60 ms before each opening / closing; a
    duck-typed :class:`~biomech.emg.prepare.EmgPrep` (cleaned = causal chain)."""
    rng = np.random.default_rng(seed)
    spans = {"rest": (0.0, 8.0), "open_close": (9.0, 29.0), "chewing": (30.0, 42.0)}
    period = {"open_close": (1.4, 1.0), "chewing": (0.85, 0.7)}
    phases, imu, cyc = {}, {}, []
    for name, (a, b) in spans.items():
        t = np.arange(a, b, 1.0 / FS)
        env = np.full((t.size, len(NAMES)), 3.0)
        tg = np.arange(a, b, 0.005)
        theta = np.zeros(tg.size)
        if name in period:
            per, dur = period[name]
            s, k = a + 0.8, 0
            while s + dur < b - 0.4:
                pk = s + dur / 2
                cyc.append({"pid": "s90", "task": name, "cycle": k, "side": "",
                            "t_start_us": s * 1e6, "t_peak_us": pk * 1e6,
                            "t_end_us": (s + dur) * 1e6, "ok": True})
                u = (tg - s) / dur
                m = (u >= 0) & (u <= 1)
                theta[m] += 20.0 * 0.5 * (1 - np.cos(2 * np.pi * u[m]))
                for cols, t0, t1 in (([0, 1], s - 0.06, pk), ([2, 3, 4, 5], pk - 0.06, s + dur)):
                    w = np.clip(np.minimum((t - t0) / 0.02, (t1 - t) / 0.02), 0, 1)
                    env[:, cols] = np.maximum(env[:, cols], 50.0 * w[:, None])
                s += per
                k += 1
        x = rng.normal(size=env.shape) * env
        mask = np.zeros(x.shape, bool)
        if name == "open_close":
            mask[int(5.0 * FS):int(5.2 * FS), 2] = True          # one masked stretch
        t_us = np.round(t * 1e6).astype(np.int64)
        phases[name] = SimpleNamespace(t_us=t_us, fs=FS, cleaned=x, mask=mask,
                                       env=np.where(mask, np.nan, np.abs(x)))
        tgu = np.round(tg * 1e6).astype(np.int64)
        om = np.gradient(theta, 0.005) + rng.normal(0, 0.5, tg.size)
        imu[f"{name}__t_us"] = tgu
        imu[f"{name}__omega_dps"] = om
        imu[f"{name}__acc_rel_ms2"] = np.abs(np.gradient(om, 0.005)) * 0.01 \
            + np.abs(rng.normal(0, 0.05, tg.size))
        imu[f"{name}__theta_opt"] = theta + rng.normal(0, 0.05, tg.size)
        imu[f"{name}__theta_quat"] = theta
    prep = SimpleNamespace(
        names=NAMES, phase=lambda n: phases[n],
        causal=lambda n: np.where(phases[n].mask, np.nan, phases[n].cleaned))
    return prep, imu, pd.DataFrame(cyc)


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False, float_format="%.17g").encode()).hexdigest()


# Pre-refactor values (see the module docstring).
REF_LEAD_NULL = "63c5f3f29dd4471c96893e02e6acd862b09c229f4f74ccf9633ee23bf88b0270"
REF_RELEASE_SUMMARY = "2c4f44a9be944509a994cb411b76528fd39287dbf8bd4886b6bb493ad2911d02"
# The other per-participant timing tables (their pre-refactor columns): the detector set-up
# moved into ``lead.build_detector`` and ``WindowOnsets`` gained options.
REF_OTHER = {
    "onsets": ("3ba67bdcc24fd7cf9ee6d3724abf548f51866252309564e6a86ac288b9e39d89",
               ["pid", "task", "cycle", "side", "direction", "muscle", "detector", "rule",
                "t_mech_us", "t_omega_us", "t_acc_us", "t_opt_us", "peak_vel_dps",
                "t_onset_us", "t_decision_us", "lead_ms", "lead_omega_ms", "lead_acc_ms",
                "lead_opt_ms", "causal_delay_ms", "anticipation_ms", "burst_db",
                "active_at_start", "usable"]),
    "summary": ("775d624570e830fe2ac429b4c84ea53f3bb635c5f6aaba8e200c70356260a0d3",
                ["pid", "task", "direction", "muscle", "detector", "rule", "n", "lead_ms",
                 "lead_omega_ms", "lead_acc_ms", "lead_opt_ms", "causal_delay_ms",
                 "anticipation_ms", "n_total"]),
    "ensemble": ("4d63ed0410cb5a0ccfe965ef463e18b1e7cb1dd98c13e49505d54646b5ac6c8d",
                 ["pid", "task", "direction", "role", "event", "n_cycles", "t_ms", "lead_ms",
                  "ci_lo_ms", "ci_hi_ms", "change_db", "boot_found"]),
    "release": ("06a0a6e7023b8f995b18590a0fe5634da94daf51a38233bda23538d6b98a1974",
                ["pid", "task", "cycle", "t_mech_us", "t_release_us", "anticipation_ms",
                 "detected"]),
}
REF_CHANCE = {
    (False, 0): "{'detected_frac': 0.6, 'anticipation_ms': 354.5292169347406, 'n': 50}",
    (False, 2): "{'detected_frac': 0.6, 'anticipation_ms': 339.0989456629436, 'n': 50}",
    (False, 11): "{'detected_frac': 0.54, 'anticipation_ms': 141.90006515928357, 'n': 50}",
    (True, 0): "{'detected_frac': 0.5892307692307692, 'anticipation_ms': 270.9119579209685, "
               "'n': 50}",
    (True, 2): "{'detected_frac': 0.5728205128205128, 'anticipation_ms': 220.1599260789752, "
               "'n': 50}",
    (True, 11): "{'detected_frac': 0.4276923076923077, 'anticipation_ms': 281.5796181592997, "
                "'n': 50}",
}


@functools.lru_cache(maxsize=1)
def _tables():
    prep, imu, cyc = _participant()
    tables, _ = L.analyse(prep, "s90", imu, cyc, P, seed=7, n_null=40)
    return tables


def test_d23_null_byte_identical():
    t = _tables()
    assert _digest(t["lead_null"]) == REF_LEAD_NULL


def test_x3_chance_level_byte_identical():
    t = _tables()
    cols = ["pid", "task", "drop_db", "hold_s", "chance_detected_frac",
            "chance_anticipation_ms", "detected_frac", "anticipation_ms"]
    assert _digest(t["release_summary"][cols]) == REF_RELEASE_SUMMARY


def test_other_timing_tables_byte_identical():
    t = _tables()
    for k, (ref, cols) in REF_OTHER.items():
        assert _digest(t[k][cols]) == ref, k


def _chance_cases():
    p = rel.ReleaseParams()
    ev = np.arange(1.0e6, 59.1e6, 1.5e6)
    rng = np.random.default_rng(5)
    dec = np.sort(ev - rng.uniform(0, 300_000, ev.size)).astype(np.int64)
    sparse = np.array([10e6, 30e6, 45e6])                # pseudo-onset branch
    out = {}
    for shift in (False, True):
        for seed in (0, 2, 11):
            out[(shift, seed)] = rel.chance_level(dec, sparse if not shift else ev, 0.0, 60e6,
                                                  p, n=50, seed=seed, shift=shift)
    return out


def test_chance_level_direct_byte_identical():
    got = _chance_cases()
    assert set(got) == set(REF_CHANCE)
    for k, v in REF_CHANCE.items():
        assert repr(got[k]) == v, k


def test_shift_helpers():
    rng = np.random.default_rng(0)
    s = S.uniform_shifts(rng, 10.0, 5, 1.0)
    assert ((s >= 1.0) & (s <= 9.0)).all()
    i_e, sel = S.shifted_windows(np.array([5, 7]), np.array([3, 9]), np.arange(-2, 3), 10)
    assert i_e.tolist() == [2, 8]
    assert sel.tolist() == [[0, 1, 2, 3, 4], [6, 7, 8, 9, 0]]
    t = S.shift_times(np.array([1.0, 4.0]), 0.0, 5.0, 2.0)
    assert t.tolist() == [3.0, 1.0]
