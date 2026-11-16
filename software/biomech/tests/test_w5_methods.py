# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Audit W5 regressions: onset selection, time base, cycles, agreement, noise, cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

from biomech import timebase
from biomech.cohort import load_cohort
from biomech.kinematics.clinical import pose_noise
from biomech.kinematics.cycles import detect
from biomech.stats import cohort_kinematics as ck
from biomech.stats.core import bland_altman, bland_altman_rm
from biomech.timing.lead import select_onset
from biomech.uncertainty.montecarlo import _noisy


# -- EMG onset selection (D6) ----------------------------------------------------------
def _active(n, runs):
    a = np.zeros(n, bool)
    for s, e in runs:
        a[s:e] = True
    return a


def test_onset_ignores_early_unrelated_burst():
    n, i_mech = 1000, 700
    active = _active(n, [(50, 150), (600, 900)])        # distractor ends long before motion
    assert select_onset(np.array([50, 600]), active, i_mech, n) == 600


def test_onset_of_long_burst_containing_motion():
    n, i_mech = 1000, 700
    active = _active(n, [(100, 950)])
    assert select_onset(np.array([100]), active, i_mech, n) == 100


def test_onset_after_motion_gives_negative_lead():
    n, i_mech = 1000, 500
    active = _active(n, [(50, 120), (560, 800)])
    assert select_onset(np.array([50, 560]), active, i_mech, n) == 560


def test_no_onset():
    assert select_onset(np.array([], dtype=int), np.zeros(10, bool), 5, 10) is None


# -- time base -------------------------------------------------------------------------
def test_gradient_never_spikes_at_gaps():
    t = np.arange(0, 2_000_000, 10_000).astype(np.int64)     # 100 Hz, 2 s
    x = 5.0 * t * 1e-6                                       # 5 units/s
    x[80:90] = np.nan
    g = timebase.gradient(t, x, smooth_n=5)
    assert np.isnan(g[80:90]).all()
    np.testing.assert_allclose(g[np.isfinite(g)], 5.0, atol=1e-9)


def test_gradient_splits_time_gaps():
    t = np.r_[np.arange(0, 100) * 10_000, 5_000_000 + np.arange(0, 100) * 10_000]
    x = np.r_[np.zeros(100), np.ones(100) * 50.0]
    g = timebase.gradient(t, x, max_gap_s=0.05)
    np.testing.assert_allclose(g, 0.0, atol=1e-9)        # no step across the 4-s gap


def test_resample_methods():
    t = np.array([0, 10, 20, 30], dtype=float) * 1e3
    x = np.array([0.0, 1.0, np.nan, 3.0])
    td = np.array([5, 15, 25, 30, 40], dtype=float) * 1e3
    lin = timebase.resample(t, x, td, method="linear")
    assert lin[0] == pytest.approx(0.5)
    assert np.isnan(lin[1]) and np.isnan(lin[2])            # gap stays a gap
    assert np.isnan(lin[4])                                  # outside the span
    hold = timebase.resample(t, x, td, method="hold")
    assert hold.tolist()[:4] == [0.0, 1.0, 1.0, 3.0]
    near = timebase.resample(t, x, td, method="nearest", max_gap_s=0.006)
    assert near[0] == 0.0 and np.isnan(near[4])


def test_resample_rotations_slerp():
    t = np.array([0.0, 1e6])
    R = Rotation.from_rotvec([[0, 0, 0], [0, 0, np.pi / 2]]).as_matrix()
    out = timebase.resample_rotations(t, R, np.array([0.5e6]))
    ang = Rotation.from_matrix(out[0]).as_euler("xyz", degrees=True)[2]
    assert ang == pytest.approx(45.0)


# -- cycles ----------------------------------------------------------------------------
def test_edge_and_non_returning_cycles_flagged_not_dropped():
    fs = 50.0
    t = np.arange(0, 10, 1 / fs)
    x = 5.0 * (1 - np.cos(2 * np.pi * 0.5 * t + np.pi / 2))   # starts half open
    cyc = detect(x, fs, min_period_s=0.4, prominence_frac=0.3, return_frac=0.2)
    assert len(cyc) >= 4
    assert not cyc[0].ok and {"edge", "no_return"} & set(cyc[0].flags)
    assert all(c.ok for c in cyc[1:-1])
    # Without the return rule (alternating tasks) only the cut edge is flagged.
    alt = detect(x, fs, min_period_s=0.4, prominence_frac=0.3)
    assert alt[0].flags == ("edge",) and all(c.ok for c in alt[1:-1])


def test_untracked_cycle_flag_uses_parameter():
    fs = 50.0
    t = np.arange(0, 6, 1 / fs)
    x = (1 - np.cos(2 * np.pi * 0.5 * t)) * 5
    x[60:66] = np.nan                                        # ~6 % of one cycle
    strict = detect(x, fs, max_nan_frac=0.01)
    loose = detect(x, fs, max_nan_frac=0.5)
    assert any("untracked" in c.flags for c in strict)
    assert not any("untracked" in c.flags for c in loose)


# -- agreement -------------------------------------------------------------------------
def test_repeated_measures_loa_wider_than_pooled():
    rng = np.random.default_rng(1)
    pids = np.repeat(np.arange(10), 20)
    bias_subject = rng.normal(0, 2.0, 10)[pids]             # between-participant bias
    d = bias_subject + rng.normal(0, 0.5, pids.size)
    ref = rng.normal(30, 5, pids.size)
    rm = bland_altman_rm(ref + d, ref, pids)
    pooled = bland_altman(ref + d, ref)
    assert rm["sd_between"] == pytest.approx(2.0, rel=0.5)
    assert rm["sd_within"] == pytest.approx(0.5, rel=0.2)
    assert (rm["loa_hi"] - rm["loa_lo"]) >= (pooled["loa_hi"] - pooled["loa_lo"]) - 1e-9


def test_reliability_split_by_side():
    pytest.importorskip("pingouin")
    rng = np.random.default_rng(2)
    rows = []
    for k in range(8):
        base = rng.normal(10, 2)
        for side, sgn in (("left", 1), ("right", -1)):
            for c in range(10):
                rows.append({"pid": f"s{k:02d}", "task": "left_right", "side": side, "cycle": c,
                             "ok": True, "lateral_mm": sgn * (base + rng.normal(0, 0.3))})
    rel = ck.reliability(pd.DataFrame(rows))
    lr = rel[rel.task == "left_right"]
    assert sorted(lr.side) == ["left", "right"]
    assert (lr.k_reps == 8).all() and (lr.icc31 > 0.9).all()


def test_condyle_sensitivity_nan_is_unknown():
    subj = pd.DataFrame({"pid": ["a", "b", "c", "d"],
                         "condyles_plausible": [1.0, 0.0, np.nan, 1.0],
                         "intercondylar_mm": [110.0, 120, 130, 115]})
    hinge = pd.DataFrame({"pid": ["a", "b", "c", "d"], "model": "rot_trans",
                          "rms_incisor_mm": 0.5, "translation_range_mm": 3.0,
                          "rms_condyles_mm": 0.4})
    out = ck.condyle_sensitivity(subj, hinge, seed=0)
    q = out[out.metric == "intercondylar_mm"].set_index("group")
    assert q.loc["plausible_condyles", "n"] == 2
    assert q.loc["condyles_unknown", "n"] == 1


# -- Monte Carlo noise -----------------------------------------------------------------
def test_pose_noise_recovers_per_axis_sd():
    rng = np.random.default_rng(3)
    n = 4000
    T = np.tile(np.eye(4), (n, 1, 1))
    sd_r = np.array([0.05, 0.10, 0.02])
    sd_t = np.array([0.03, 0.01, 0.05])
    T = _noisy(T, sd_t, sd_r, rng)
    r, t = pose_noise(T)
    np.testing.assert_allclose(r, sd_r, rtol=0.08)
    np.testing.assert_allclose(t, sd_t, rtol=0.08)


def test_noisy_keeps_untracked_frames_untracked():
    T = np.tile(np.eye(4), (10, 1, 1))
    T[3] = np.nan
    out = _noisy(T, 0.1, 0.1, np.random.default_rng(0))
    assert np.isnan(out[3]).all() and np.isfinite(out[4]).all()


# -- cohort manifest -------------------------------------------------------------------
def test_manifest_include_must_be_bool(tmp_path):
    doc = {"participants": [{"pid": "s01", "sex": "f", "age": 31, "motion": None,
                             "include": "false"}]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="include must be true or false"):
        load_cohort(p, tmp_path)
    doc["participants"][0]["include"] = True
    p.write_text(yaml.safe_dump(doc))
    c = load_cohort(p, tmp_path)
    assert c.participants[0].motion == ""
    assert any("no motion session" in m for m in c.validate())


def test_pose_noise_separates_rotation_from_origin_motion():
    """Rotation noise far from the origin must not be counted again as translation noise."""
    rng = np.random.default_rng(4)
    T = np.tile(np.eye(4), (3000, 1, 1))
    T[:, :3, 3] = [80.0, -20.0, -60.0]                 # pose origin far from the frame origin
    T = _noisy(T, 0.0, np.array([0.05, 0.05, 0.05]), rng)
    r, t = pose_noise(T)
    np.testing.assert_allclose(r, 0.05, rtol=0.08)
    assert (t < 1e-6).all()


def test_movement_bounds_start_where_motion_starts():
    from biomech.kinematics.clinical import movement_bounds

    t = np.linspace(0, 2.0, 201)
    x = np.where((t > 0.5) & (t < 1.5), 0.5 * (1 - np.cos(2 * np.pi * (t - 0.5))), 0.0)
    v = np.abs(np.gradient(x, t))
    on, off = movement_bounds(v, int(np.argmax(x)))
    assert abs(t[on] - 0.5) < 0.04 and abs(t[off] - 1.5) < 0.04
    assert v[on + 1] < 0.15 * v.max()                 # velocity still low at the onset
