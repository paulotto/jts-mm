# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""C2-E building blocks (X6 (change log)): fast unit tests.

The synthetic gates G1-G6 (slow) are in ``test_r11_c2e_synth.py``; the byte-identity of the
D23 null and the X3 chance level after the surrogate refactor in ``test_r11_surrogates.py``.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from biomech.stats import cohort_c2e as CC
from biomech.stats import families as F
from biomech.testing import synth as S
from biomech.timing import c2e, changepoint, coupling, locking, surrogates
from biomech.timing import lead as L

P = {"tasks": ["open_close", "chewing"], "vel_frac": 0.05, "acc_k": 3.0, "pre_s": 0.3,
     "emg_pre_s": 0.6, "emg_post_s": 0.2, "k_sd": 3.0, "burst_min_db": 6.0, "ensemble": {},
     "release": {}}
QUICK = {"n_surrogates": 20, "n_s2": 20, "coupling": {"n_null": 10}}
SPEC = dict(emg_fs=1000, task_s={"open_close": 12.0, "chewing": 8.0})


# -- synthetic participants stay byte-identical ------------------------------------------------
def _h5digest(path: Path) -> str:
    h = hashlib.sha256()
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                v = np.asarray(obj[()])
                h.update(name.encode())
                h.update(repr(v.tolist()).encode() if v.dtype.kind in "OSU"
                         else np.ascontiguousarray(v).tobytes())
        f.visititems(visit)
    return h.hexdigest()


# SHA-256 of every dataset of the motion and MVC takes, written by the synth module before
# the C2-E extensions (commit d027a80): DEFAULT_SPECS and the two kinds of D23 fixture specs.
# Updated for the montage fix of 2026-11-03 (an03 §1): the generator draws in its old muscle
# order and writes the columns in the recorder's new channel order (CH3 digastric_r, CH4
# digastric_l), and the pre-take check carries the new channel numbers. Every muscle's data,
# lead-off and impedance are unchanged by name (checked against the d027a80-era output).
REF_H5 = {
    "s01": ("66e9172cc85fec97ae18a2f031a6ddbc3572449c9e03e685a408f0cb86ac5e27",
            "ccfb3f79a914ae66423449b1c9da453cabdb330fa201110c245090d5abea6abc"),
    "s02": ("fbc0604799aff53d78bbd03cf0f1f60fe95840488cd701bdf13cddd49f765ec7",
            "3d95d51e2c070842d97ab1acd02a37d8f6f18fdf9d2264678fdc0d91ca57b16e"),
    "s03": ("61955e178467900bfbae0a1c958433a1f881aa35babfef4063527abcaa5be89c",
            "f904e639d64fb4c91cc2f1b3d0b0bdb69be219943b3f08bf477fae8814f60c5d"),
    "s04": ("a8d6508f73855fd94406408ef1235156fd538a72c4d643821815aeed338897da",
            "e5d5c3d9ddc154c61d605406757dc3378dd2fa274cf16c620574026c458b811d"),
    "s50": ("c283e5a79bb98d81b319ffc12a45914d9b500f5ef0cc0d42b99cca8359995d7e",
            "597c0f6a72ea11a023c396377e8026f6c2359af804c52e5b4b84f407d15cb803"),
    "s61": ("dcc7868ab8d09d7f418e769c42f49d13f7b0cf60f2280d40901823da31a30dce",
            "273a544d383e853de5c7b82237e6341e8767f2497a7626f2b5daa1e7e0cbae2d"),
}


def test_default_specs_and_d23_fixtures_byte_identical():
    kw = dict(emg_fs=2000, open_close_period_s=3.0,
              task_s={"open_close": 30.0, "chewing": 15.0})
    specs = list(S.DEFAULT_SPECS) + [S.Spec(pid="s50", lead_ms=0.0, seed=11, **kw),
                                     S.Spec(pid="s61", emg_decoupled=True, seed=21, **kw)]
    for sp in specs:
        with tempfile.TemporaryDirectory() as d:
            S.write_participant(d, sp)
            paths = sorted(Path(d).glob("*/session.h5"), key=lambda p: "mvc" in p.parent.name)
            assert tuple(_h5digest(p) for p in paths) == REF_H5[sp.pid], sp.pid


def test_writer_refuses_simulator_only_fields(tmp_path):
    with pytest.raises(ValueError, match="timing_arrays only"):
        S.write_participant(tmp_path, S.Spec(quat_noise_deg=0.1))
    with pytest.raises(ValueError, match="emg_lock"):
        S.timing_arrays(S.Spec(emg_lock="sometimes"))


# -- the in-memory simulator -------------------------------------------------------------------
def test_timing_arrays_truth_and_extensions():
    ta = S.timing_arrays(S.Spec(seed=3, closer_lead_ms=50.0, opener_lead_ms=20.0, **SPEC))
    assert set(ta.prep.names) == set(S.MUSCLES)
    assert {"rest__theta_quat", "open_close__omega_dps", "chewing__acc_rel_ms2"} \
        <= set(ta.imu_arrays)
    tr = ta.truth["open_close"]
    assert len(tr) == (ta.kin_cycles.task == "open_close").sum()
    for b in tr:
        s0, pk, _ = b["cycle"]
        assert b["closer"][0] == pytest.approx(pk - 0.05)
        assert b["opener"][0] == pytest.approx(s0 - 0.02)
    # period jitter: own generator, varying periods; the default schedule untouched
    j = S.timing_arrays(S.Spec(seed=3, period_jitter_s=0.2, **SPEC))
    per = np.diff([b["cycle"][0] for b in j.truth["open_close"]])
    assert per.std() > 0.05
    # no IMU: the s09 case
    assert S.timing_arrays(S.Spec(seed=3, **SPEC), imu=False).imu_arrays == {}
    # previous-event locking keeps the mean lead, not the per-cycle one
    pe = S.timing_arrays(S.Spec(seed=3, period_jitter_s=0.2, closer_lead_ms=40.0,
                                emg_lock="previous_event", **SPEC))
    d = np.array([b["cycle"][1] - b["closer"][0] for b in pe.truth["open_close"]])
    assert d.mean() == pytest.approx(0.04, abs=0.01) and d.std() > 0.01


# -- the vectorised onsets equal the scalar ones -----------------------------------------------
def test_batch_onsets_equal_scalar_window_onsets():
    ta = S.timing_arrays(S.Spec(seed=5, lead_jitter_ms=30.0, **SPEC))
    det = L.build_detector(ta.prep, P)[0].restricted()
    pe = ta.prep.phase("open_close")
    x = np.where(pe.mask, np.nan, pe.cleaned)
    x[3000:3100, 0] = np.nan                      # a masked stretch: the scalar fallback
    N, fs = len(pe.t_us), pe.fs
    rng = np.random.default_rng(1)
    _, sel = surrogates.shifted_windows(rng.integers(700, N - 300, 60),
                                        surrogates.s1_sample_shifts(rng, N, fs, 60, 1.0),
                                        np.arange(-600, 201), N)
    sel[0] = np.arange(2700, 3501)                # contains the mask
    muscles = ("masseter_r", "temporalis_r", "masseter_l")
    b = det.batch(x, pe.cleaned, pe.mask, sel, 600, muscles)
    tw = np.arange(801) * (1e6 / fs)
    for m in muscles:
        for j in range(len(sel)):
            (_, _, _, t_on, _, bd, act, use), = det(tw[600], tw, sel[j], x, x, pe.cleaned,
                                                    pe.mask, (m,))
            k = int(np.searchsorted(tw, t_on)) if np.isfinite(t_on) else -1
            assert k == b[m]["k_on"][j] and act == b[m]["active"][j]
            assert use == b[m]["usable"][j]
            assert bd == b[m]["burst_db"][j] or (np.isnan(bd) and np.isnan(b[m]["burst_db"][j]))
    with pytest.raises(ValueError):
        L.build_detector(ta.prep, P)[0].batch(x, pe.cleaned, pe.mask, sel, 600, muscles)


# -- one lagged-correlation helper: the C3a IMU lag is unchanged -------------------------------
def _lag_ms_legacy(t_us, ref, x, max_ms=100.0, min_pairs=50):
    """imu.analysis._lag_ms before the refactor (verbatim)."""
    from biomech import timebase

    ok = np.isfinite(ref) & np.isfinite(x)
    if ok.sum() < min_pairs:
        return np.nan
    t = np.asarray(t_us, dtype=np.float64)
    dt = np.median(np.diff(t)) * 1e-6
    a = timebase.gradient(t, np.where(ok, ref, np.nan))
    b = timebase.gradient(t, np.where(ok, x, np.nan))
    k = int(round(max_ms * 1e-3 / dt))
    lags = np.arange(-k, k + 1)
    n = len(a)
    c = np.full(lags.size, np.nan)
    for j, Lg in enumerate(lags):
        aa = a[max(0, -Lg): n - max(0, Lg)]
        bb = b[max(0, Lg): n - max(0, -Lg)]
        v = np.isfinite(aa) & np.isfinite(bb)
        if v.sum() < min_pairs:
            continue
        av, bv = aa[v], bb[v]
        den = np.sqrt(np.dot(av, av) * np.dot(bv, bv))
        if den > 0:
            c[j] = np.dot(av, bv) / den
    if not np.isfinite(c).any():
        return np.nan
    return float(lags[int(np.nanargmax(c))] * dt * 1e3)


def test_imu_lag_uses_the_shared_helper_unchanged():
    from biomech.imu.analysis import _lag_ms

    rng = np.random.default_rng(4)
    t = np.arange(4000) * 5000
    ref = np.sin(np.arange(4000) / 30.0) * 10
    for shift in (0, 3, -4):
        x = np.roll(ref, shift) + rng.normal(0, 0.1, 4000)
        x[500:520] = np.nan
        assert _lag_ms(t, ref, x) == _lag_ms_legacy(t, ref, x)
    lags, c = coupling.lagged_corr(ref, np.roll(ref, 6), 10, center=True)
    assert coupling.peak_lag(lags, c)[0] == 6.0          # b late by 6 samples
    assert np.isnan(coupling.peak_lag(lags, np.full(lags.size, np.nan))[0])


# -- surrogates, locking, changepoint ----------------------------------------------------------
def test_s2_shuffle_keeps_anchors_and_offsets():
    a = np.array([0.0, 10.0, 20.0, 30.0])
    e = a + np.array([1.0, 2.0, 3.0, 4.0])
    perms = surrogates.s2_permutations(np.random.default_rng(0), 4, 5)
    sh = surrogates.s2_shuffle(a, e, perms)
    assert np.allclose(np.sort(sh - a, axis=1), [1, 2, 3, 4])
    assert surrogates.mc_p(np.arange(10.0), 3.5, "less") == pytest.approx(5 / 11)
    assert np.isnan(surrogates.mc_p(np.array([]), 1.0))
    assert surrogates.s1_sample_shifts(np.random.default_rng(0), 1500, 1000.0, 5, 1.0).size == 0


def test_mixture_recovers_location_against_a_background():
    rng = np.random.default_rng(2)
    lo, hi = -200.0, 600.0
    bg = rng.uniform(lo, hi, 4000)
    obs = np.concatenate([rng.normal(40, 15, 30), rng.uniform(lo, hi, 10)])
    m = locking.mixture_location(obs, bg, lo, hi)
    assert m["reason"] == "" and abs(m["mu_ms"] - 40) < 10 and 0.6 < m["pi"] <= 1.0
    none = locking.mixture_location(rng.uniform(lo, hi, 40), bg, lo, hi)
    assert none["pi"] < 0.5
    assert locking.mixture_location(obs[:3], bg, lo, hi)["reason"] == "few_leads"


def test_locking_theil_sen_and_s2_slope():
    rng = np.random.default_rng(3)
    null = [rng.normal(0, 100, 30) for _ in range(99)]
    r = locking.locking(rng.normal(0, 10, 30), null)
    assert r["log_ratio"] < -1 and r["p_lock"] == pytest.approx(0.01)
    assert locking.theil_sen([0, 1, 2, 3], [1, 3, 5, 7]) == 2.0
    a = np.cumsum(rng.uniform(1.5e6, 2.0e6, 20))
    m = a + rng.uniform(0.2e6, 0.6e6, 20)
    locked = locking.s2_slope(a, m, m - 80e3, np.random.default_rng(0), 199)
    anchored = locking.s2_slope(a, m, a + 300e3, np.random.default_rng(0), 199)
    assert locked["b"] == pytest.approx(1.0) and locked["p_s2"] < 0.01
    assert anchored["b"] == pytest.approx(0.0) and anchored["p_s2"] > 0.05


def test_variance_changepoint():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 500), rng.normal(0, 5, 300)])
    X = np.vstack([x, rng.normal(0, 1, 800), np.where(np.arange(800) < 100, np.nan, x)])
    k, llr, db = changepoint.variance_changepoints(X, 1000.0, 0.05)
    assert abs(k[0] - 500) <= 5 and db[0] > 10
    assert db[1] < 6                   # no real change: below the 6-dB usability gate
    assert abs(k[2] - 500) <= 5                        # masked samples left out


# -- alpha onset -------------------------------------------------------------------------------
def test_alpha_signal_and_onset():
    t = np.arange(0, 4_000_000, 5000)
    s = t * 1e-6 - 1.0
    th = np.where((s >= 0) & (s <= 1), 10 * (1 - np.cos(2 * np.pi * s)), 0.0)
    al = L.alpha_signal(t, th, 15.0)
    mid = (s > 0.1) & (s < 0.9)
    expect = 10 * (2 * np.pi) ** 2 * np.cos(2 * np.pi * s)
    assert np.nanmax(np.abs(al[mid] - expect[mid])) < 0.05 * np.nanmax(np.abs(expect))
    om = np.gradient(th, t * 1e-6)
    base = {"omega_noise": 0.0, "acc_thr": np.inf, "alpha_noise": 0.0}
    mech = L.mechanical_onsets(t, om, np.zeros_like(om), th, int(0.6e6), int(1.5e6), "opening",
                               base, {"vel_frac": 0.05}, alpha=al)
    assert abs(mech["t_alpha"] - 1.0e6) <= 25_000 and mech["t_alpha"] < mech["t_omega"]
    cl = L.mechanical_onsets(t, om, np.zeros_like(om), th, int(1.4e6), int(2.0e6), "closing",
                             base, {"vel_frac": 0.05}, alpha=al, alpha_t0=mech["t_omega"])
    assert abs(cl["t_alpha"] - 1.25e6) <= 30_000            # braking from the velocity peak
    with pytest.raises(ValueError):
        L.alpha_signal(t, th, 150.0)


# -- the per-participant analysis: edge cases ---------------------------------------------------
@pytest.fixture(scope="module")
def one():
    sp = S.Spec(seed=8, closer_lead_ms=40.0, lead_jitter_ms=20.0, period_jitter_s=0.15,
                **SPEC)
    return sp, S.timing_arrays(sp)


def test_c2e_tables_and_reasons(one):
    sp, ta = one
    out = c2e.analyse(ta.prep, "s1", ta.imu_arrays, ta.kin_cycles, {**P, "c2e": QUICK},
                      seed=1, kin_arrays=ta.kin_arrays)
    assert set(out) == {"c2e_cycles", "c2e_null", "c2e_release", "coupling"}
    cyc, nul = out["c2e_cycles"], out["c2e_null"]
    assert set(cyc.direction) == {"opening", "closing"}
    assert (cyc.usable | (cyc.reason != "")).all()
    assert {"lead_omega_ms", "lead_alpha15_ms", "lead_opt_ms", "lead_cp_ms",
            "lead_sym_ms"} <= set(cyc.columns)
    pooled = nul[(nul.scope == "pooled") & (nul.direction == "closing")
                 & (nul.source == "omega")].iloc[0]
    assert 15 <= pooled.n_surr <= QUICK["n_surrogates"] and np.isfinite(pooled.mu_ms)
    assert set(nul[nul.source == "release"].scope) == {"open_close", "chewing"}
    assert set(out["coupling"].task) == {"open_close", "chewing"}
    # deterministic
    again = c2e.analyse(ta.prep, "s1", ta.imu_arrays, ta.kin_cycles, {**P, "c2e": QUICK},
                        seed=1, kin_arrays=ta.kin_arrays)
    pd.testing.assert_frame_equal(nul, again["c2e_null"])


def test_c2e_without_imu_uses_the_optical_onset(one):
    sp, _ = one
    ta = S.timing_arrays(sp, imu=False)
    out = c2e.analyse(ta.prep, "s09", {}, ta.kin_cycles, {**P, "c2e": QUICK}, seed=1,
                      kin_arrays=ta.kin_arrays)
    assert set(out["c2e_null"].event_source.dropna()) <= {"opt", ""}
    assert "omega" not in set(out["c2e_null"].source)
    assert out["coupling"].empty
    # neither IMU nor optics: a reason row, no crash
    none = c2e.analyse(ta.prep, "s09", {}, ta.kin_cycles, {**P, "c2e": QUICK}, seed=1)
    assert set(none["c2e_null"].flag) == {"no_mechanical_signal"}


def test_c2e_no_cycles_all_nan_channel_one_channel(one):
    _, ta = one
    empty = c2e.analyse(ta.prep, "s1", ta.imu_arrays, ta.kin_cycles.iloc[:0],
                        {**P, "c2e": QUICK}, seed=1)
    assert set(empty["c2e_null"].flag) == {"no_phase_or_cycles"}
    # one closer completely masked (rest included: no threshold) and the others dropped
    ph = {n: ta.prep.phase(n) for n in ("rest", "open_close", "chewing")}
    for p in ph.values():
        p.mask = p.mask.copy()
        p.mask[:, ta.prep.names.index("masseter_r")] = True
    prep = S.SynthPrep(ta.prep.names, ph)
    out = c2e.analyse(prep, "s1", ta.imu_arrays, ta.kin_cycles,
                      {**P, "c2e": {**QUICK, "directions": ["closing"]}}, seed=1)
    cyc = out["c2e_cycles"]
    m = cyc[cyc.muscle == "masseter_r"]
    assert not m.usable.any() and (m.reason == "no_onset").all()
    single = S.SynthPrep(["masseter_l"], {n: S._PhaseArrays(
        p.t_us, p.fs, p.cleaned[:, [ta.prep.names.index("masseter_l")]],
        np.zeros((len(p.t_us), 1), bool), p.env[:, [ta.prep.names.index("masseter_l")]])
        for n, p in ta.prep._phases.items()})
    out1 = c2e.analyse(single, "s1", ta.imu_arrays, ta.kin_cycles,
                       {**P, "c2e": {**QUICK, "p5": False}}, seed=1)
    assert set(out1["c2e_cycles"].muscle) == {"masseter_l"}


def test_c2e_params_reject_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        c2e.C2EParams.from_dict({"n_surogates": 3})
    assert c2e.C2EParams.from_dict({"gates_passed": True, "alpha": 0.05}).n_surrogates == 200


# -- cohort ------------------------------------------------------------------------------------
def _null_table(n=13, mu=60.0, lr=-1.0, rel=100.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        pid = f"s{k:02d}"
        base = {"pid": pid, "n_usable": 20, "flag": "ok"}
        rows.append({**base, "scope": "pooled", "direction": "closing", "source": "omega",
                     "log_ratio": lr + rng.normal(0, 0.2), "mu_ms": mu + rng.normal(0, 10),
                     "median_minus_null_ms": 30 + rng.normal(0, 5)})
        for src in ("alpha15", "cp", "opt", "sym"):
            rows.append({**base, "scope": "pooled", "direction": "closing", "source": src,
                         "mu_ms": mu + rng.normal(0, 10),
                         "median_minus_null_ms": 30 + rng.normal(0, 5)})
        rows.append({**base, "scope": "pooled", "direction": "opening", "source": "omega",
                     "log_ratio": -0.5 + rng.normal(0, 0.2), "mu_ms": -40.0})
        for task in ("open_close", "chewing"):
            rows.append({**base, "scope": task, "direction": "closing", "source": "omega",
                         "iqr_obs": 20.0, "iqr_s2_median": 40 + rng.normal(0, 3)})
            rows.append({**base, "scope": task, "direction": "opening", "source": "release",
                         "log_depth_ratio": 1 + rng.normal(0, 0.2),
                         "median_ms": rel + rng.normal(0, 10), "b_s2": 0.9, "p_b_s2": 0.01,
                         "x_sd_ms": 150.0})
    rows[0]["n_usable"] = 4                                      # flagged, not tested
    return pd.DataFrame(rows)


def test_cohort_iut_gatekeeping_and_decision():
    t = CC.c2e_cohort(_null_table(), None, {"delta_ms": 12.0, "gates_passed": True}, 0)
    k = t.set_index("key")
    assert list(t.columns) == CC.COLUMNS
    h1 = k.loc["h1"]
    assert h1.n == 12 and h1.n_flagged == 1
    assert h1.p == max(h1.p_lock, h1.p_loc) and h1.rejected
    assert np.isfinite(k.loc["h2", "p_adj"]) and k.loc["h2", "rejected"]
    assert k.loc["decision", "hypothesis"] == "shown (post hoc)"
    # H1 not rejected: H2/H3 not tested; underpowered gate -> inconclusive
    t2 = CC.c2e_cohort(_null_table(lr=0.5), None, {"delta_ms": 12.0, "gates_passed": True,
                                                   "gate_power_40ms": 0.3}, 0).set_index("key")
    assert not t2.loc["h1", "rejected"] and np.isnan(t2.loc["h2", "p_adj"])
    assert t2.loc["decision", "hypothesis"] == "inconclusive (underpowered)"
    # a veto row (alpha-referenced location entirely below 0) blocks "shown"
    nt = _null_table()
    nt.loc[nt.source == "alpha15", "mu_ms"] = -150.0 + np.arange(13.0)
    t3 = CC.c2e_cohort(nt, None, {"delta_ms": 12.0, "gates_passed": True}, 0).set_index("key")
    assert t3.loc["p4_alpha", "veto"] and t3.loc["decision", "hypothesis"] == "not shown"
    # delta not fixed: no verdict
    t4 = CC.c2e_cohort(_null_table(), None, {}, 0).set_index("key")
    assert t4.loc["decision", "hypothesis"] == "delta not fixed"
    assert CC.c2e_cohort(pd.DataFrame(), None, {}, 0).empty


def test_holm_over_h1_h3_post_hoc_sensitivity():
    """r11 for orientation: combined p 0.6152 / 0.0014 / 0.0156 -> Holm 0.6152 / 0.0042 /
    0.0312; computed although the gatekeeping stops at H1; gatekeeping columns unchanged."""
    rows = [{"key": k, "p": p, "note": ""} for k, p in (("h1", 0.6152), ("h2", 0.0014),
                                                         ("h3", 0.0156))]
    CC.holm_family(rows)
    assert [r["p_holm_family"] for r in rows] == pytest.approx([0.6152, 0.0042, 0.0312])
    assert all(r["note"] == CC.HOLM_FAMILY_NOTE for r in rows)
    t = CC.c2e_cohort(_null_table(lr=0.5), None, {"delta_ms": 12.0, "gates_passed": True},
                      0).set_index("key")
    assert not t.loc["h1", "rejected"] and np.isnan(t.loc["h2", "p_adj"])   # gate stops
    hs = t.loc[["h1", "h2", "h3"]]
    assert hs["p_holm_family"].tolist() == pytest.approx(CC.holm(hs["p"].tolist()))
    assert np.isfinite(hs["p_holm_family"]).all()
    assert hs["note"].str.contains("post hoc sensitivity: Holm over H1–H3").all()
    others = t.drop(index=["h1", "h2", "h3"])
    assert others["p_holm_family"].isna().all()


def test_families_helpers():
    assert F.iut(0.01, 0.2) == 0.2 and np.isnan(F.iut(0.01, np.nan))
    rej, adj = F.gatekeeping(0.2, [0.01, 0.02])
    assert not rej and all(np.isnan(adj))
    rej, adj = F.gatekeeping(0.01, [0.01, 0.04])
    assert rej and adj == [0.02, 0.04]
    w, p, exact = F.wilcoxon_signed(np.arange(1.0, 8.0), "greater")
    assert exact and p == pytest.approx(1 / 128)
    assert F.location_test(np.arange(10.0), None, 0)["p"] != F.location_test(
        np.arange(10.0), 0.0, 0)["p"]


def test_requirements_c2e_rows():
    from biomech.stats.requirements import requirements_table

    e = pd.DataFrame()
    c = CC.c2e_cohort(_null_table(), None, {"delta_ms": 12.0, "gates_passed": True}, 0)
    t = requirements_table(e, e, e, e, e, e, e, e, e, seed=0, c2e=c).set_index("key")
    assert {"c2e_h1_ms", "c2e_h2_ms"} <= set(t.index)
    assert set(t.loc[["c2e_h1_ms", "c2e_h2_ms"], "evidence"]) == {"post_hoc"}
    assert "shown (post hoc)" in t.loc["c2e_h1_ms", "note"]
