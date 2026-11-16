# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J34: the C2 start gate (gate flow, in-task threshold leads) and the label-shift null
of the C3b permutation test (the ceiling and position decoder of the audit script)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from biomech.viz.derived_dataset import c2_gate_flow, c2_taskbase_leads

ROOT = Path(__file__).resolve().parents[3]


def _onsets() -> pd.DataFrame:
    """Four opener channel-cycles under rule first: usable, active at the start, burst too
    small, no onset; plus a closing row and a second detector that must be ignored."""
    base = {"pid": "s1", "task": "open_close", "direction": "opening", "detector": "tkeo",
            "rule": "first", "t_mech_us": 1e6, "burst_db": 12.0, "active_at_start": False}
    rows = [dict(base, cycle=0, muscle="digastric_r", t_onset_us=0.9e6, usable=True),
            dict(base, cycle=1, muscle="digastric_r", t_onset_us=0.9e6, usable=False,
                 active_at_start=True),
            dict(base, cycle=2, muscle="digastric_l", t_onset_us=0.9e6, usable=False,
                 burst_db=3.0),
            dict(base, cycle=3, muscle="digastric_l", t_onset_us=np.nan, usable=False),
            dict(base, cycle=0, muscle="masseter_r", direction="closing", t_onset_us=0.9e6,
                 usable=True),
            dict(base, cycle=0, muscle="digastric_r", detector="envelope", t_onset_us=0.9e6,
                 usable=False, active_at_start=True)]
    return pd.DataFrame(rows)


def test_gate_flow_counts_each_channel_cycle_at_its_first_failing_gate():
    o = _onsets()
    ctx = SimpleNamespace(table=lambda s, n, pids=None: o, config=None)
    r = c2_gate_flow(ctx).iloc[0]
    assert r["n"] == 4
    assert (r["usable"], r["active_start"], r["burst"], r["no_onset"], r["masked"]) == \
        (1, 1, 1, 1, 0)
    assert r["active_start_share"] == pytest.approx(0.25)


def test_taskbase_leads_per_task():
    rows = []
    for task, meds in (("open_close", (-40.0, -18.0, -60.0)), ("chewing", (-5.0, 38.0, 9.0))):
        for k, m in enumerate(meds):
            v = f"detector=d{k}|mech=lead_ms|onset=drive_taskbase"
            n = 3 if k else 2
            rows += [{"pid": f"s{i}", "outcome": f"{task}|opening|lead", "variant": v,
                      "value": m + d} for i, d in enumerate((-1.0, 0.0, 1.0)[:n])]
        rows.append({"pid": "s0", "outcome": f"{task}|opening|lead",
                     "variant": "detector=d0|mech=lead_ms|onset=first", "value": 500.0})
    t = pd.DataFrame(rows)
    ctx = SimpleNamespace(table=lambda s, n: t)
    out = c2_taskbase_leads(ctx).set_index("task")
    assert out.loc["open_close", "n_variants"] == 3
    assert out.loc["open_close", "median_max"] == pytest.approx(-18.0)
    assert out.loc["open_close", "above_zero"] == 0
    assert (out.loc["open_close", "n_min"], out.loc["open_close", "n_max"]) == (2, 3)
    assert out.loc["chewing", "above_zero"] == 2
    assert out.loc["chewing", "median_max"] == pytest.approx(38.0)


def _script():
    if not (ROOT / "scripts" / "audit_recompute" / "j34_label_shift.py").exists():
        pytest.skip("the audit script is not in this checkout (public release)")
    sys.path.insert(0, str(ROOT / "scripts" / "audit_recompute"))
    try:
        import j34_label_shift
    finally:
        sys.path.pop(0)
    return j34_label_shift


def test_shift_ceiling_constant_class_is_chance_and_ceiling_matches_brute_force():
    from sklearn.metrics import balanced_accuracy_score

    js = _script()
    rng = np.random.default_rng(0)
    y = np.repeat(rng.integers(0, 4, 12), rng.integers(3, 9, 12))  # blocks, as task phases
    y[5] = -1                                                       # one untracked window
    ceiling, const = js.shift_ceiling(y, 4)
    present = np.unique(y[y >= 0]).size
    assert const == pytest.approx(1 / present, rel=1e-12)
    # Brute force: the expected balanced accuracy over every allowed shift of the best
    # per-window class equals the closed form.
    n = len(y)
    shifts = range(n // 4, 3 * n // 4)
    ok = y >= 0
    q = np.zeros((n, 4))
    for s in shifts:
        ys = np.roll(y, s)
        q[np.flatnonzero(ys >= 0), ys[ys >= 0]] += 1
    nk = np.bincount(y[ok], minlength=4)
    pred = np.argmax(np.divide(q, nk, out=np.zeros_like(q), where=nk > 0), axis=1)
    brute = np.mean([balanced_accuracy_score(np.roll(y, s)[np.roll(ok, s)],
                                             pred[np.roll(ok, s)]) for s in shifts])
    assert ceiling == pytest.approx(brute, rel=1e-9)
    assert ceiling > 1 / present


def test_position_decoder_reads_the_other_participants():
    from ml.evaluation.metrics import balanced_accuracy

    js = _script()
    y = np.concatenate([np.repeat([0, 1, 2], 10)] * 3)              # same order, 3 people
    idx = {q: np.arange(30) + 30 * k for k, q in enumerate("abc")}
    pos = {q: np.minimum(np.arange(30) * js.N_BINS // 30, js.N_BINS - 1) for q in idx}
    assert js.position_ba(y, idx, pos, balanced_accuracy) == pytest.approx(1.0)
    y2 = y.copy()
    y2[30:60] = np.roll(y[30:60], 10)                               # one person shifted
    assert js.position_ba(y2, idx, pos, balanced_accuracy) < 1.0
