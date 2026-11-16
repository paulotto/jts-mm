# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D23: the chance null of the C2 lead and the X3 release decision.

The EMG onset is searched from 600 ms before to 200 ms after the mechanical onset, so a burst
unrelated to the movement still yields a positive "first" lead. The null repeats the
per-window analysis with the EMG circularly shifted against the motion (open_close and
chewing; D23a: open_close was first planned with pseudo-onsets >= 1 s from every real onset,
which the continuous cycling of the real protocol does not leave; the pseudo-onset branch is
kept for tasks with pauses); ``stats_timing`` tests observed - null next to the frozen test.

Fast: the pseudo-onset draw, the window bookkeeping of the null (stub detector), the
sign-flip test, the cohort columns and Holm families, the X3 chance anticipation.
Slow (synthetic pipeline, one fixture, ~25 s): true leads 0 and -80 ms recovered by the drive
rule; EMG decoupled from the motion: the calibrated test does not reject while "first"
against 0 is biased positive.

Synthetic numbers at the time of writing (6 decoupled participants, TKEO, opening; seeds
fixed, so they are reproducible):

  | task       | "first" observed (cohort median) | frozen p vs 0 | chance null | p_null |
  |------------|----------------------------------|---------------|-------------|--------|
  | open_close | +151 ms                          | 0.078         | +310 ms     | 0.92   |
  | chewing    | +284 ms                          | 0.016         | +235 ms     | 0.22   |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.stats import cohort_misc as cm
from biomech.stats.core import holm
from biomech.timing import lead as L
from biomech.timing import release as rel
from biomech.timing.release import ReleaseParams

P = {"emg_pre_s": 0.6, "emg_post_s": 0.2}


# -- pseudo-onsets ------------------------------------------------------------------------
def test_allowed_intervals_and_uniform_draw():
    excl = np.array([5e6, 5.5e6, 12e6])
    iv = L.allowed_intervals(0.0, 20e6, excl, 1e6)
    assert iv == [(1e6, 4e6), (6.5e6, 11e6), (13e6, 19e6)]
    t = L.draw_uniform(iv, 5000, np.random.default_rng(0))
    assert t.size == 5000
    assert (np.abs(t[:, None] - excl[None, :]) >= 1e6).all()
    assert (t >= 1e6).all() and (t <= 19e6).all()
    # uniform on the union: share per interval ~ its length (3 + 4.5 + 6 = 13.5 s)
    share = [np.mean((t >= a) & (t <= b)) for a, b in iv]
    np.testing.assert_allclose(share, [3 / 13.5, 4.5 / 13.5, 6 / 13.5], atol=0.03)
    assert L.allowed_intervals(0.0, 1.5e6, np.empty(0), 1e6) == []
    assert L.draw_uniform([], 10, np.random.default_rng(0)).size == 0
    # back-to-back onsets 1.4 s apart leave no admissible time
    assert L.allowed_intervals(0.0, 12e6, np.arange(1e6, 12e6, 0.7e6), 1e6) == []


class _StubDetector:
    """Records the windows and reports one usable onset at the first window sample."""

    fs = 1000.0

    def __init__(self):
        self.seen = []

    def __call__(self, t_mech, tw, sel, x, xc_all, cleaned, mask, group):
        self.seen.append((t_mech, np.asarray(tw), np.asarray(sel), group))
        for m in group[:2]:
            for rule in L.ONSET_RULES:
                yield m, "tkeo", rule, float(tw[0]), float(tw[0]) + 20e3, 20.0, False, True


def _phase(T_s=30.0, fs=1000.0):
    t = (np.arange(int(T_s * fs)) * (1e6 / fs)).astype(np.int64) + 10_000_000
    z = np.zeros((t.size, 6))
    return (t, z, z, z, np.zeros(z.shape, bool))


def test_shift_tasks_d23a():
    assert set(L.NULL_SHIFT_TASKS) == {"open_close", "chewing"}


def test_pseudo_onset_null_draws_away_from_real_onsets():
    """The pseudo-onset branch (tasks with pauses; none of the study's since D23a)."""
    ph = _phase()
    det = _StubDetector()
    real = np.array([15e6, 22e6, 30e6])
    rows = L.pseudo_onset_null(det, "s01", "pause_task", ph, {"opening": real}, real, P, 3,
                               n=50)
    tm = np.array([s[0] for s in det.seen])
    assert len(det.seen) == 100                                  # 50 x both directions
    assert (np.abs(tm[:, None] - real[None, :]) >= 1e6).all()
    assert (tm >= ph[0][0] + 1e6).all() and (tm <= ph[0][-1] - 1e6).all()
    r = pd.DataFrame(rows)
    assert set(r.direction) == {"opening", "closing"} and set(r.rule) == set(L.ONSET_RULES)
    assert (r.null_kind == "pseudo_onset").all() and (r.n_null == 50).all()
    # onset at the window start: lead = emg_pre_s (to a sample), the causal decision 20 ms on
    np.testing.assert_allclose(r.lead_null_ms, 600.0, atol=1.0)
    np.testing.assert_allclose(r.anticipation_null_ms, 580.0, atol=1.0)
    assert (r.usable_null_frac == 1.0).all() and (r.n_null_usable == 100).all()
    # the same seed and task: the same pseudo-onsets
    det2 = _StubDetector()
    L.pseudo_onset_null(det2, "s01", "pause_task", ph, {"opening": real}, real, P, 3, n=50)
    assert np.array_equal(tm, [s[0] for s in det2.seen])


def test_pseudo_onset_null_chewing_circular_shift_wraps():
    ph = _phase(T_s=10.0)
    t = ph[0]
    det = _StubDetector()
    real = {"opening": np.array([12e6, 14e6]), "closing": np.array([12.4e6])}
    rows = L.pseudo_onset_null(det, "s01", "chewing", ph, real, np.empty(0), P, 5, n=200)
    assert len(det.seen) == 400
    n_win = 600 + 200 + 1
    wrapped = 0
    for tm, tw, sel, _ in det.seen:
        assert sel.dtype.kind == "i" and sel.size == n_win and tw.size == n_win
        assert tw[600] == tm                                     # anchor = sample 600
        assert np.array_equal(np.diff(sel) % t.size, np.ones(n_win - 1))   # consecutive mod N
        wrapped += int((np.diff(sel) < 0).any())
    assert wrapped > 0                                           # some windows cross the edge
    # each shift is in [1 s, T - 1 s]: the pseudo EMG time differs from the real onset
    # (mod T) by at least 1 s
    T = t.size / 1000.0 * 1e6
    for (tm, *_), k in zip(det.seen[:200], range(200)):
        r_on = real["opening"][k % 2]
        d = (r_on - tm) % T
        assert 1e6 - 1e3 <= d <= T - 1e6 + 1e3
    r = pd.DataFrame(rows)
    assert (r.null_kind == "circular_shift").all() and (r.n_null == 200).all()
    np.testing.assert_allclose(r.lead_null_ms, 600.0, atol=1e-6)


# -- cohort tests -------------------------------------------------------------------------
def test_sign_flip_p():
    rng = np.random.default_rng(1)
    assert np.isnan(cm.sign_flip_p(np.ones(4)))
    assert cm.sign_flip_p(np.arange(1.0, 15.0), seed=0) < 0.001          # all 14 positive
    p6 = cm.sign_flip_p(np.arange(1.0, 7.0), seed=0)                     # exact: 1/64
    assert abs(p6 - 1 / 64) < 0.005
    assert cm.sign_flip_p(-np.arange(1.0, 7.0), seed=0) > 0.98
    assert cm.sign_flip_p(rng.normal(size=12), seed=3) == cm.sign_flip_p(
        np.random.default_rng(1).normal(size=12), seed=3)               # seeded
    # calibration: under the symmetric null, P(p <= 0.05) ~ 0.05
    ps = [cm.sign_flip_p(rng.normal(size=10), n_perm=2000, seed=k) for k in range(300)]
    assert 0.02 <= np.mean(np.asarray(ps) <= 0.05) <= 0.09


def _toy(n_pid=7, null_shift=0.0, seed=0):
    """Onset rows of n_pid participants: 4 usable cycles x 2 digastrics per task/rule."""
    rng = np.random.default_rng(seed)
    rows, nul = [], []
    for k in range(n_pid):
        pid = f"s{k + 1:02d}"
        for task in ("open_close", "chewing"):
            for rule in ("first", "drive", "drive_taskbase"):
                base = {"first": 200.0, "drive": 40.0, "drive_taskbase": -20.0}[rule]
                for c in range(4):
                    for m in ("digastric_l", "digastric_r"):
                        v = base + rng.normal(0, 30)
                        rows.append({"pid": pid, "task": task, "direction": "opening",
                                     "muscle": m, "detector": "tkeo", "rule": rule,
                                     "cycle": c, "lead_ms": v, "anticipation_ms": v - 50,
                                     "usable": True, "active_at_start": False})
                nul.append({"pid": pid, "task": task, "direction": "opening",
                            "detector": "tkeo", "rule": rule,
                            "lead_null_ms": base + null_shift + rng.normal(0, 10),
                            "lead_null_muscle_mean_ms": base + null_shift,
                            "usable_null_frac": 0.4, "n_null": 200, "n_null_usable": 160})
    on = pd.DataFrame(rows)
    keys = ["pid", "task", "direction", "muscle", "detector", "rule"]
    sm = on.groupby(keys).agg(n=("lead_ms", "size"), lead_ms=("lead_ms", "median"),
                              anticipation_ms=("anticipation_ms", "median")).reset_index()
    return sm, on, pd.DataFrame(nul)


def test_timing_tables_null_columns_holm_and_primary_unchanged():
    sm, on, nul = _toy()
    base = cm.timing_tables(sm, on, 7)
    out = cm.timing_tables(sm, on, 7, lead_null=nul)
    a, b = base["timing_cohort"], out["timing_cohort"]
    # the frozen test and every pre-D23 column are unchanged
    pd.testing.assert_frame_equal(a, b[a.columns])
    new = {"lead_null_ms", "lead_minus_null_ms", "n_null_pairs", "W_null", "p_null", "r_null",
           "p_null_perm", "p_null_holm", "p_null_perm_holm"}
    assert new <= set(b.columns)
    lead = b[b.metric == "lead_ms"]
    assert lead.p_null.notna().all() and b[b.metric != "lead_ms"].p_null.isna().all()
    # Holm over the same four tests as C2 (opening lead, TKEO, both tasks x first/drive)
    conf = lead[lead.family == "confirmatory"]
    assert len(conf) == 4 and set(conf.rule) == {"first", "drive"}
    np.testing.assert_allclose(conf.p_null_holm, holm(conf.p_null.tolist()))
    np.testing.assert_allclose(conf.p_null_perm_holm, holm(conf.p_null_perm.tolist()))
    expl = lead[lead.family == "exploratory"]
    np.testing.assert_allclose(expl.p_null_holm, holm(expl.p_null.tolist()))
    # observed ~ null: nothing to reject; the per-participant pairs are listed
    assert (conf.p_null > 0.05).all()
    tn = out["timing_null"]
    assert len(tn) == 7 * 2 * 3
    np.testing.assert_allclose(tn.lead_minus_null_ms, tn.lead_ms - tn.lead_null_ms)
    # a null 150 ms below the observed lead: rejected by both tests
    out2 = cm.timing_tables(sm, on, 7, lead_null=_toy(null_shift=-150.0)[2])
    c2 = out2["timing_cohort"]
    c2 = c2[(c2.metric == "lead_ms") & (c2.family == "confirmatory")]
    assert (c2.p_null < 0.01).all() and (c2.p_null_perm < 0.01).all()
    assert (c2.lead_minus_null_ms > 100).all()


def test_null_pairs_needs_usable_null_and_muscle_mean_source():
    sm, on, nul = _toy(n_pid=5)
    per, _ = cm.per_participant_timing(sm, on, "open_close", "opening", "tkeo", "first")
    nul.loc[nul.pid == "s01", "n_null_usable"] = 2                      # < min_cycles
    pr = cm.null_pairs(per, nul, "open_close", "opening", "tkeo", "first")
    assert np.isnan(pr.set_index("pid").loc["s01", "lead_null_ms"])
    assert pr.lead_null_ms.notna().sum() == 4
    mm = cm.null_pairs(per, nul, "open_close", "opening", "tkeo", "first", "muscle_mean")
    np.testing.assert_allclose(mm.dropna().lead_null_ms, 200.0)
    t = cm.null_test(pr, 0)
    assert t["n_null_pairs"] == 4 and np.isnan(t["p_null"])             # n < 5: no test


# -- X3 release vs chance -------------------------------------------------------------------
def test_chance_level_circular_shift_without_free_second():
    """D23a: back-to-back openings (one per 1.5 s) leave no pseudo-onset >= 1 s from every
    onset; the circular shift still gives a chance level, and a decision train locked to the
    onsets scores above it."""
    p = ReleaseParams()
    ev = np.arange(1.0e6, 59.1e6, 1.5e6)                 # 1.0 ... 58.0 s
    dec = ev - 100_000                                   # 100 ms before every opening
    assert rel.chance_level(dec, ev, 0.0, 60e6, p, seed=2)["n"] == 0
    ch = rel.chance_level(dec, ev, 0.0, 60e6, p, n=100, seed=2, shift=True)
    assert ch["n"] == 100 and np.isfinite(ch["detected_frac"])
    rows, _ = rel.evaluate(dec, ev, p)
    assert np.mean([r["detected"] for r in rows]) == 1.0
    assert ch["detected_frac"] < 1.0
    again = rel.chance_level(dec, ev, 0.0, 60e6, p, n=100, seed=2, shift=True)
    assert again == ch


def test_chance_level_anticipation_and_cohort_columns():
    p = ReleaseParams()
    dec = np.arange(0, 60_000_000, 1_000_000, dtype=np.int64) + 300_000   # one per second
    ch = rel.chance_level(dec, np.array([30e6]), 0.0, 60e6, p, seed=2)
    assert ch["n"] == 200
    assert ch["detected_frac"] == rel.chance_detection(dec, np.array([30e6]), 0.0, 60e6, p,
                                                       seed=2)
    assert -200 <= ch["anticipation_ms"] <= 600
    rows = []
    for k in range(8):
        rows.append({"pid": f"s{k}", "task": "open_close", "drop_db": 3.0, "hold_s": 0.03,
                     "primary": True, "detected_frac": 0.9, "chance_detected_frac": 0.4,
                     "chance_anticipation_ms": 250.0 + k, "anticipation_ms": 100.0 + 3 * k,
                     "false_per_min": 1.0})
    c = cm.release_cohort(pd.DataFrame(rows), 0).iloc[0]
    assert c.chance_anticipation_ms_median == 253.5
    assert c.anticipation_minus_chance_ms_median == pytest.approx(-150 + 2 * 3.5)
    assert c.n_vs_chance == 8 and c.p_vs_chance < 0.01                  # two-sided, all below


# -- synthetic pipeline: truth and decoupled EMG ---------------------------------------------
TRUTH = {"s50": 0.0, "s51": -80.0}
DECOUPLED = [f"s6{k}" for k in range(1, 7)]
POSITIVE = "s52"                  # D34 regression case: a true, time-locked lead of +50 ms


@pytest.fixture(scope="module")
def c2_synth(tmp_path_factory):
    """Two participants with true leads 0 and -80 ms, one with +50 ms (D34) and six with
    EMG bursts at random times; open_close with 2-s pauses (3-s period) so pseudo-onsets >= 1 s from every real
    onset exist. Stages up to stats_timing."""
    from biomech.config import load_config
    from biomech.pipeline import run
    from biomech.results import ResultStore
    from biomech.testing.synth import Spec, write_study

    kw = dict(emg_fs=2000, open_close_period_s=3.0,
              task_s={"open_close": 30.0, "chewing": 15.0})
    specs = [Spec(pid="s50", lead_ms=TRUTH["s50"], seed=11, **kw),
             Spec(pid="s51", lead_ms=TRUTH["s51"], seed=12, **kw),
             Spec(pid=POSITIVE, lead_ms=50.0, seed=13, **kw)]
    specs += [Spec(pid=pid, emg_decoupled=True, seed=21 + k, **kw)
              for k, pid in enumerate(DECOUPLED)]
    root = tmp_path_factory.mktemp("c2_null")
    path, _ = write_study(root, specs)
    cfg = load_config(path)
    run(cfg, ["stats_timing"], n_jobs=4)
    return cfg, ResultStore(cfg.out_dir)


@pytest.mark.slow
def test_drive_rule_recovers_true_leads_0_and_minus_80(c2_synth):
    """D23: the drive rule recovers true leads of 0 and -80 ms within +-20 ms (every
    detector; open_close both directions, chewing closing). Chewing opening is not checked:
    in the synthetic back-to-back chewing (0.15 s between cycles) the acceleration onset
    still sits above its rest threshold from the previous closing in some cycles, so the
    mechanical onset falls 300-450 ms early and the lead reads about +450...+530 ms under
    every rule (a mechanical-onset issue, not an onset-rule one; reported with D23)."""
    _, store = c2_synth
    sm = store.read_table("timing", "summary")
    for pid, truth in TRUTH.items():
        for task, direction in (("open_close", "opening"), ("open_close", "closing"),
                                ("chewing", "closing")):
            g = sm[(sm.pid == pid) & (sm.task == task) & (sm.direction == direction)
                   & (sm.rule == "drive")]
            leads = g.groupby("detector").lead_ms.median()
            assert len(leads) == 3, (pid, task, direction)
            np.testing.assert_allclose(leads.to_numpy(), truth, atol=20.0,
                                       err_msg=f"{pid} {task} {direction}")
    # D23a: open_close nulls by circular shift, every shift evaluated
    nul = store.read_table("timing", "lead_null")
    oc = nul[nul.pid.isin(list(TRUTH)) & (nul.task == "open_close")]
    assert (oc.null_kind == "circular_shift").all() and (oc.n_null == L.N_NULL).all()


@pytest.mark.slow
def test_decoupled_emg_calibrated_test_does_not_reject_first_is_biased(c2_synth):
    """D23: EMG unrelated to the motion. The calibrated test (observed - null) must not
    reject at 0.05 (all four confirmatory tests, both test forms, raw and Holm); the frozen
    "first" rule against 0 is biased positive: cohort median > +100 ms in both tasks and
    the null itself sits > +200 ms (the size of the bias of a lead from a random burst)."""
    _, store = c2_synth
    on = store.read_table("timing", "onsets")
    sm = store.read_table("timing", "summary")
    nul = store.read_table("timing", "lead_null")
    sel = lambda d: d[d.pid.isin(DECOUPLED)]                   # noqa: E731
    out = cm.timing_tables(sel(sm), sel(on), 7, lead_null=sel(nul))
    c = out["timing_cohort"]
    c = c[(c.metric == "lead_ms") & (c.family == "confirmatory")].set_index(["task", "rule"])
    assert len(c) == 4 and (c.n_null_pairs == 6).all()
    for col in ("p_null", "p_null_perm", "p_null_holm", "p_null_perm_holm"):
        assert (c[col] > 0.05).all(), c[col]
    for task in ("open_close", "chewing"):
        first = c.loc[(task, "first")]
        assert first["median"] > 100.0, (task, first["median"])
        assert first.lead_null_ms > 200.0, (task, first.lead_null_ms)
    # the frozen test against 0 rejects (unadjusted) for chewing although nothing is coupled
    assert c.loc[("chewing", "first"), "p"] < 0.05
    # the stage wrote the same columns
    st = store.read_table("stats_timing", "timing_cohort")
    assert {"lead_null_ms", "lead_minus_null_ms", "p_null", "p_null_perm",
            "p_null_holm"} <= set(st.columns)
    assert not store.read_table("stats_timing", "timing_null").empty
    # every pseudo-onset of a decoupled participant was evaluated
    dn = sel(nul)
    assert (dn.n_null == L.N_NULL).all()
    assert (dn[dn.rule.isin(["first", "drive"])].usable_null_frac > 0).all()
    assert set(dn.null_kind) == {"circular_shift"}


@pytest.mark.slow
def test_release_vs_chance_columns(c2_synth):
    """X3 (D23): the chance anticipation per participant and the paired cohort test."""
    _, store = c2_synth
    r = store.read_table("timing", "release_summary")
    assert "chance_anticipation_ms" in r
    oc = r[(r.task == "open_close") & r.primary]
    assert oc.chance_anticipation_ms.notna().all()
    c = store.read_table("stats_timing", "release_cohort")
    c = c[(c.task == "open_close") & c.primary].iloc[0]
    pairs = np.isfinite(oc.anticipation_ms - oc.chance_anticipation_ms).sum()
    assert c.n_vs_chance == pairs >= 5 and np.isfinite(c.p_vs_chance)


@pytest.mark.slow
def test_positive_lead_has_no_power_against_the_d23_null(c2_synth):
    """D34 regression case: a true, time-locked lead of +50 ms. The drive rule recovers it,
    but the D23 null median (the window bias of an unrelated burst, ~200-330 ms) lies far above
    it, so observed - null < 0 in every confirmatory cell and the participant's one-sided test
    of observed - null > 0 (signed-rank over its usable channel-cycles against its null median)
    has p_null > 0.5: the D23 test is calibrated but has no power for physiological leads."""
    from biomech.stats.core import wilcoxon_one_sample

    _, store = c2_synth
    on = store.read_table("timing", "onsets")
    nul = store.read_table("timing", "lead_null")
    sm = store.read_table("timing", "summary")
    g = sm[(sm.pid == POSITIVE) & (sm.task == "open_close") & (sm.direction == "opening")
           & (sm.rule == "drive")]
    np.testing.assert_allclose(g.lead_ms.median(), 50.0, atol=20.0)
    # open_close only: in the synthetic back-to-back chewing too few opening channel-cycles
    # are usable for a per-participant test (see test_drive_rule_recovers_true_leads_0_and_minus_80).
    for task in ("open_close",):
        for rule in ("first", "drive"):
            o = on[(on.pid == POSITIVE) & (on.task == task) & (on.direction == "opening")
                   & (on.detector == "tkeo") & (on.rule == rule) & on.usable.astype(bool)]
            n0 = nul[(nul.pid == POSITIVE) & (nul.task == task) & (nul.direction == "opening")
                     & (nul.detector == "tkeo") & (nul.rule == rule)].lead_null_ms.iloc[0]
            assert n0 > 150.0, (task, rule, n0)
            d = o.lead_ms.to_numpy(float) - n0
            assert d.size >= 5 and np.median(d) < 0, (task, rule)
            p_null = wilcoxon_one_sample(d, "greater")["p"]
            assert p_null > 0.5, (task, rule, p_null)


@pytest.mark.slow
def test_c2e_tables_written_by_the_stages(c2_synth):
    """X6 wiring: the timing stage writes the C2-E tables, stats_timing the cohort table
    (while delta is not fixed the decision says so)."""
    cfg, store = c2_synth
    for t in ("c2e_cycles", "c2e_null", "c2e_release", "coupling"):
        assert not store.read_table("timing", t).empty, t
    c = store.read_table("stats_timing", "c2e_cohort").set_index("key")
    assert {"h1", "h2", "h3", "decision"} <= set(c.index)
    delta = cfg.params("timing")["c2e"].get("delta_ms")
    if delta is None:
        assert c.loc["decision", "hypothesis"] == "delta not fixed"
