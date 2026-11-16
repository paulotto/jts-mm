# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""End-to-end: the full stage graph on synthetic participants with a known truth.

The participants come from :mod:`biomech.testing.synth`. Each assertion compares
an analysis output with the value injected into the data, so this test checks
that the pipeline *measures what it claims*, not only that it runs.

  s01  pure hinge                         lead  60 ms
  s02  hinge + 4 mm delayed translation   lead  90 ms, lead-off on digastric_r
  s03  hinge + 6 mm, larger opening       lead 120 ms
  s04  hinge + 3 mm, 60 Hz mains          lead  75 ms
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.pipeline import registry
from biomech.results import COHORT_UNIT

pytestmark = pytest.mark.slow

@pytest.fixture(scope="module")
def synth(synth_study):
    return synth_study


def test_every_unit_ran(synth):
    _, _, _, summary = synth
    bad = {(st, u): s for st, d in summary.statuses.items() for u, s in d.items() if s != "ran"}
    assert not bad, bad


# -- anatomy and landmarks -----------------------------------------------------------
def test_anatomy_recovered(synth):
    _, store, truths, _ = synth
    an = store.read_table("kinematics", "anatomy").set_index("pid")
    for pid, t in truths.items():
        a = an.loc[pid]
        lm = t.landmarks_max
        up = np.linalg.norm(np.subtract(lm["upper_left"], lm["upper_right"]))
        lo = np.linalg.norm(np.subtract(lm["lower_left"], lm["lower_right"]))
        ic = np.linalg.norm(np.subtract(lm["tmj_left"], lm["tmj_right"]))
        assert a.upper_intermolar_mm == pytest.approx(up, abs=0.3)
        assert a.lower_intermolar_mm == pytest.approx(lo, abs=0.3)
        assert a.intercondylar_mm == pytest.approx(ic, abs=0.3)
        for side in ("left", "right"):
            got = [a[f"frame_tmj_{side}_{ax}"] for ax in "xyz"]
            np.testing.assert_allclose(got, lm[f"tmj_{side}"], atol=0.3)
        assert a.roles_upper_geometry_agrees == 1 and a.roles_lower_geometry_agrees == 1
        assert a.roles_relabelled == 0 and a.condyles_plausible == 1
        assert a.frame_lr_ok == 1 and a.frame_posterior_ok == 1 and a.frame_superior_ok == 1


# -- cycles and clinical metrics -----------------------------------------------------
def _match(cyc: pd.DataFrame, peaks_s: list[float]) -> pd.DataFrame:
    """Detected cycles matched to the true peaks (nearest peak within 0.25 s)."""
    t = cyc.t_peak_us.to_numpy() * 1e-6
    rows = []
    for k, tp in enumerate(peaks_s):
        j = int(np.argmin(np.abs(t - tp))) if t.size else -1
        if j >= 0 and abs(t[j] - tp) < 0.25:
            rows.append((k, j))
    return pd.DataFrame(rows, columns=["true", "found"])


def test_cycles_and_clinical_metrics(synth):
    _, store, truths, _ = synth
    cyc = store.read_table("kinematics", "cycles")
    for pid, t in truths.items():
        c = cyc[cyc.pid == pid]
        oc = c[c.task == "open_close"].reset_index(drop=True)
        peaks = [p[1] for p in t.cycles["open_close"]]
        m = _match(oc, peaks)
        assert len(m) == len(peaks) and oc.ok.all()
        exp = np.array(t.peak_incisor_mm["open_close"])[m.true]
        got = oc.loc[m.found]
        np.testing.assert_allclose(got.opening_mm, -exp[:, 2], atol=0.5)
        np.testing.assert_allclose(got.theta_y_deg, np.array(t.peak_theta_deg["open_close"])[
            m.true], atol=0.5)

        pr = c[c.task == "protrusion_retrusion"].reset_index(drop=True)
        m = _match(pr, [p[1] for p in t.cycles["protrusion_retrusion"]])
        exp = np.array(t.peak_incisor_mm["protrusion_retrusion"])[m.true]
        np.testing.assert_allclose(pr.loc[m.found].protrusion_mm, exp[:, 0], atol=0.5)

        lr = c[c.task == "left_right"]
        for side, sgn in (("left", 1), ("right", -1)):
            s = lr[lr.side == side]
            assert (sgn * s[s.ok].lateral_mm).median() == pytest.approx(
                t.spec["lateral_mm"], rel=0.12)


# -- C1: hinge models and the noise null ---------------------------------------------
def test_hinge_models_and_noise_null(synth):
    _, store, truths, _ = synth
    h = store.read_table("kinematics", "hinge").set_index(["pid", "model"])
    un = store.read_table("uncertainty", "hinge_noise").set_index("pid")
    for pid, t in truths.items():
        tr = t.spec["translation_mm"]
        assert h.loc[(pid, "rot_trans"), "rms_incisor_mm"] < 0.15     # hinge + translation fits
        if tr == 0:
            hf = h.loc[(pid, "hinge_fit")]
            assert hf.rms_incisor_mm < 0.15
            assert hf.axis_y == pytest.approx(1.0, abs=1e-3)
            assert hf.axis_pierce_x_mm == pytest.approx(t.hinge_axis_s[0], abs=1.0)
            assert hf.axis_pierce_z_mm == pytest.approx(t.hinge_axis_s[2], abs=1.0)
            assert h.loc[(pid, "rot_trans"), "translation_range_mm"] < 0.5
            assert not un.loc[pid, "above_invivo"] and not un.loc[pid, "above_bench"]
        else:
            assert h.loc[(pid, "hinge_fit"), "rms_incisor_mm"] > 0.3
            assert h.loc[(pid, "rot_trans"), "translation_range_mm"] > 2.0
            assert un.loc[pid, "above_invivo"] and un.loc[pid, "above_bench"]
    # The hinge residual orders participants by their injected translation.
    order = sorted(truths, key=lambda p: truths[p].spec["translation_mm"])
    res = [h.loc[(p, "hinge_fit"), "rms_incisor_mm"] for p in order]
    assert res == sorted(res)


# -- C3a: IMU calibration and estimators ---------------------------------------------
def test_imu_calibration_and_agreement(synth):
    _, store, truths, _ = synth
    cal = store.read_table("imu", "calibration").set_index("pid")
    ag = store.read_table("imu", "agreement")
    for pid, t in truths.items():
        c = cal.loc[pid]
        assert c.rms_deg < 0.1
        X = Rotation.from_rotvec(np.radians([c[f"X_rv{a}_deg"] for a in "xyz"]))
        err = (X * Rotation.from_matrix(np.array(t.X_mta_mp)).inv()).magnitude()
        assert np.degrees(err) < 1.0
        g = ag[(ag.pid == pid) & (ag.task != "rest")]
        assert (g.rmse_quat_deg < 0.15).all()


# -- C2: EMG precedes motion ---------------------------------------------------------
def test_emg_lead_recovered(synth):
    _, store, truths, _ = synth
    sm = store.read_table("timing", "summary")
    oc = sm[sm.task == "open_close"]
    assert set(oc.rule) == {"first", "drive", "drive_taskbase"}   # all onset rules stored
    for pid, t in truths.items():
        for rule in ("drive", "first"):                    # one clean burst: the rules agree
            g = oc[(oc.pid == pid) & (oc.rule == rule)]
            for direction in ("opening", "closing"):
                leads = g[g.direction == direction].groupby("detector").lead_ms.median()
                assert len(leads) == 3
                np.testing.assert_allclose(leads.to_numpy(), t.spec["lead_ms"], atol=20.0)
    on = store.read_table("timing", "onsets")
    use = on[on.task == "open_close"].groupby("pid").usable.mean()
    assert (use > 0.8).all()


def test_release_tables(synth):
    """X3: the causal release detector ran for every participant and both tasks."""
    _, store, truths, _ = synth
    r = store.read_table("timing", "release_summary")
    assert set(r.pid) == set(truths) and r.primary.any()
    assert r.groupby(["pid", "task"]).size().eq(9).all()          # 3 drops x 3 holds
    assert (r.false_per_min >= 0).all() and r.detected_frac.between(0, 1).all()
    c = store.read_table("stats_timing", "release_cohort")
    assert c.primary.any()


def test_ensemble_timing_recovers_lead(synth):
    """X2: the detector-free agonist rise of the aligned envelopes recovers the injected lead."""
    _, store, truths, _ = synth
    e = store.read_table("timing", "ensemble")
    ag = e[(e.task == "open_close") & (e.direction == "opening") & (e.role == "agonist")]
    for pid, t in truths.items():
        r = ag[ag.pid == pid].iloc[0]
        assert r.n_cycles >= 5 and r.change_db > 10
        assert abs(r.lead_ms - t.spec["lead_ms"]) < 40, (pid, r.lead_ms, t.spec["lead_ms"])
        assert r.ci_lo_ms <= r.t_ms <= r.ci_hi_ms
    a = store.read_arrays("timing", next(iter(truths)))
    assert a["ensemble__grid_ms"].size == a["ensemble__open_close__opening__agonist"].size
    c = store.read_table("stats_timing", "ensemble_cohort")
    row = c[(c.task == "open_close") & (c.direction == "opening") & (c.role == "agonist")].iloc[0]
    assert row.n_found == len(truths) and row["median"] > 0


# -- EMG references, mains, lead-off -------------------------------------------------
def test_emg_references_and_mains(synth):
    _, store, truths, _ = synth
    refs = store.read_table("emg", "refs")
    op = refs[refs.ref_phase == "mvc_opening"].p95_uv
    cl = refs[refs.ref_phase == "mvc_closing"].p95_uv
    assert op.between(200, 330).all() and cl.between(240, 390).all()
    q = store.read_table("qc", "emg")
    assert (q.mains_share_raw > 0.5).all() and (q.mains_share_clean < 0.08).all()
    lead = q.set_index(["pid", "muscle"]).leadoff_frac
    assert lead.loc[("s02", "digastric_r")] > 0 and lead.drop(("s02", "digastric_r")).eq(0).all()
    assert q.set_index("pid").line_hz.loc["s04"].mean() == pytest.approx(60.0, abs=0.05)


def test_mask_roles_and_sensitivity_table(synth):
    """D14: lowfreq is reported (QC flag, sensitivity table) but not part of the analysis mask."""
    _, store, _, _ = synth
    q = store.read_table("qc", "emg")
    assert {"mask_lowfreq_frac", "lowfreq_flag"} <= set(q.columns)
    ms = store.read_table("emg", "mask_sensitivity")
    assert not ms.empty and (ms.rules == "rail,leadoff,envelope").all()
    assert (ms.alt_rules == "rail,leadoff,envelope,lowfreq").all()
    # The alternative adds a rule: it can only mask more and keep fewer cycles.
    assert (ms.mask_frac_alt >= ms.mask_frac - 1e-12).all()
    assert (ms.kept_alt <= ms.kept).all() and (ms.kept <= ms.n_cycles).all()
    # Lead-off (s02 digastric_r) is an analysis rule in both.
    d = ms[(ms.pid == "s02") & (ms.muscle == "digastric_r")]
    assert (d.mask_frac > 0).any()


# -- labels and cohort stages ----------------------------------------------------------
def test_labels_cover_all_classes(synth):
    _, store, _, _ = synth
    c = store.read_table("labels", "counts")
    secs = c.groupby("label").seconds.sum()
    for lab in ("REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT"):
        assert secs[lab] > 1.0, lab


def test_cohort_stages_finite(synth):
    _, store, truths, _ = synth
    rv = store.read_table("stats_kinematics", "reference_values")
    assert rv[rv.group == "all"]["mean"].notna().any()
    tc = store.read_table("stats_timing", "timing_cohort")
    lead = tc[(tc.task == "open_close") & (tc.metric == "lead_ms")]
    assert lead["n"].max() == len(truths) and np.isfinite(lead["median"]).all()
    assert lead.groupby(["direction", "detector"]).rule.nunique().ge(2).all()
    assert store.status("variability", COHORT_UNIT) == "up-to-date"


def test_ml_stages_when_installed(synth):
    if "ml_eval" not in registry():
        pytest.skip("ml plugin not installed")
    _, store, _, _ = synth
    assert store.status("ml_eval", COHORT_UNIT) == "up-to-date"


# -- frequency domain (X1) --------------------------------------------------------------
def test_spectral_shape_and_bandwidth(synth):
    _, store, truths, _ = synth
    sp = store.read_table("emg", "spectral")
    chew = sp[(sp.task == "chewing") & (sp.muscle == "masseter_r")].set_index("pid")
    others = chew.drop(index="s04").mnf_hz
    assert chew.loc["s04", "mnf_hz"] < others.min() - 30          # narrower band, lower MNF
    spm = store.read_table("stats_emg", "spectral_spm", ["_cohort"]).set_index("test")
    assert bool(spm.loc["anova1_masseter_r_chewing_spectrum", "h0reject"])
    bw = store.read_table("imu", "bandwidth")
    chew_bw = bw[bw.task == "chewing"]
    rate = {p: 1 / np.median(np.diff([c[0] for c in t.cycles["chewing"]]))
            for p, t in truths.items()}
    for _, r in chew_bw.iterrows():                        # chewing rhythm inside the band
        assert r.f50_hz <= 2.5 * rate[r.pid] and r.f95_hz >= rate[r.pid] * 0.9
    # D12: the optical-confirmed (coherence-weighted) bandwidth holds the rhythm and never
    # exceeds the raw IMU f95 by more than a bin; open-close is not noise-limited.
    for _, r in bw[bw.task.isin(["chewing", "open_close"])].iterrows():
        assert np.isfinite(r.f95_coh_hz) and r.f95_coh_hz <= r.f95_hz + 0.5
        assert not bool(r.noise_limited) and r.coh_band_hz >= 2.0
    for _, r in chew_bw.iterrows():
        assert r.f95_coh_hz >= 0.9 * rate[r.pid]
