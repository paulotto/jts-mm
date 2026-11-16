# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""QTM label swaps of a near-isosceles rigid body (erratum EV1): detection, relabelling and
pose refit (``io.reference.fix_label_swaps``)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.io.reference import (QtmBody, QtmData, _kabsch, fix_label_swaps,
                                  label_ambiguity)

# the CT triangle (sides 60.0, 74.3, 75.9 mm) and an unambiguous cluster
CT = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [27.997, 68.823, 0.0]])
HP = np.array([[0.0, 0.0, 0.0], [75.65, 0.0, 0.0], [25.97, 37.48, 0.0]])   # s10 head cluster


def _pose(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _take(swap_runs=(1, 3), distorted_run=2, n_run=200, gap=20, seed=0):
    """Five runs separated by marker gaps; QTM labels swapped in ``swap_runs`` (its pose
    then is the fit to the swapped labels, ~180° off); ``distorted_run`` has one marker
    stretched by 3 mm along the base (both labellings misfit: no informative frames) and
    continues run 1 across a short gap."""
    rng = np.random.default_rng(seed)
    n = 5 * (n_run + gap)
    t = np.arange(n) / 200.0
    rots = Rotation.from_rotvec(np.outer(np.sin(t / 3.0), [0.3, 0.2, 0.4]))
    pos = np.stack([100 * np.sin(t), 50 * np.cos(t / 2), 20 * t], axis=1)
    true = np.zeros((n, 4, 4))
    M = {f"CT - {k + 1}": np.full((n, 3), np.nan) for k in range(3)}
    H = {f"HP - {k + 1}": np.full((n, 3), np.nan) for k in range(3)}
    Tq = np.full((n, 4, 4), np.nan)
    Th = np.full((n, 4, 4), np.nan)
    swapped = np.zeros(n, bool)
    for r in range(5):
        a, b = r * (n_run + gap), r * (n_run + gap) + n_run
        for i in range(a, b):
            T = _pose(rots[i].as_matrix(), pos[i])
            true[i] = T
            m = CT @ T[:3, :3].T + T[:3, 3] + rng.normal(0, 0.2, (3, 3))
            if r == distorted_run:
                m[0] += np.array([-3.0, 0.0, 0.0]) @ T[:3, :3].T
            if r in swap_runs:
                m = m[[1, 0, 2]]
                swapped[i] = True
            R, tt = _kabsch(CT, m)                      # what QTM reports for these labels
            Tq[i] = _pose(R, tt)
            for k in range(3):
                M[f"CT - {k + 1}"][i] = m[k]
            h = HP @ T[:3, :3].T + T[:3, 3] + rng.normal(0, 0.2, (3, 3))
            Th[i] = T
            for k in range(3):
                H[f"HP - {k + 1}"][i] = h[k]
    q = QtmData(path=None, take="t", file="", timestamp="", start_frame=1, n_frames=n,
                fs_hz=200.0, t_s=t, bodies={"CT": QtmBody("CT", Tq, np.zeros(n)),
                                            "HP": QtmBody("HP", Th, np.zeros(n))},
                markers={**M, **H})
    return q, true, swapped


def _angle(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def test_ambiguity_margin_separates_ct_from_unambiguous_clusters():
    m_ct, cand = label_ambiguity(CT)
    assert m_ct < 3.0 and (1, 0, 2) in cand
    m_hp, cand_hp = label_ambiguity(HP)
    assert m_hp > 10.0 and not cand_hp


def test_swapped_runs_are_relabelled_and_refit():
    q, true, swapped = _take(swap_runs=(1, 3), distorted_run=4)
    hp_before = q.bodies["HP"].T.copy()
    flag = fix_label_swaps(q, "CT")
    ok = np.isfinite(q.bodies["CT"].T).all(axis=(1, 2))
    assert np.array_equal(flag[ok], swapped[ok])
    # the corrected pose matches the truth (body frame = definition frame)
    ang = np.array([_angle(q.bodies["CT"].T[i, :3, :3], true[i, :3, :3])
                    for i in np.flatnonzero(ok)])
    assert np.percentile(ang, 99) < 2.0
    assert q.label_info["CT"]["n_swapped"] == int(swapped[ok].sum())
    # the unambiguous body is untouched
    fix_label_swaps(q, "HP")
    assert q.label_info["HP"]["ambiguous"] is False
    assert np.array_equal(q.bodies["HP"].T, hp_before, equal_nan=True)


def test_distorted_run_is_decided_by_continuity():
    # run 2 is distorted (no informative frames) and swapped; it follows run 1 (decided)
    q, true, swapped = _take(swap_runs=(1, 2), distorted_run=2)
    fix_label_swaps(q, "CT")
    info = q.label_info["CT"]
    assert info["runs_by_continuity"] >= 1
    run2 = slice(2 * 220, 2 * 220 + 200)
    assert q.label_swapped["CT"][run2].all()
    ang = [_angle(q.bodies["CT"].T[i, :3, :3], true[i, :3, :3])
           for i in range(run2.start, run2.stop)]
    assert np.median(ang) < 3.0


def test_undecidable_run_is_flagged_not_changed():
    # a distorted run after a long gap: neither cue decides -> QTM labels kept, flagged
    q, true, swapped = _take(swap_runs=(), distorted_run=2, gap=400)
    before = q.bodies["CT"].T.copy()
    fix_label_swaps(q, "CT")
    run2 = slice(2 * 600, 2 * 600 + 200)
    assert q.label_swapped["CT__undetermined"][run2].all()
    assert np.array_equal(q.bodies["CT"].T[run2], before[run2])


def test_tooth_holds_recovers_tip_and_world_offset():
    from biomech.validation import tip as TP

    rng = np.random.default_rng(3)
    d_true, c_true = np.array([1.0, -2.0, 0.5]), np.array([4.0, -3.0, 2.0])
    holds = {}
    for h in range(8):
        R0 = Rotation.from_rotvec(rng.normal(0, 1.2, 3)).as_matrix()
        R = np.stack([R0 @ Rotation.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix()
                      for _ in range(30)])
        t = rng.normal(0, 100, 3) + rng.normal(0, 0.1, (30, 3))
        p = np.einsum("nij,j->ni", R, d_true) + t - c_true + rng.normal(0, 0.2, (30, 3))
        holds[f"h{h}"] = (R, t, p)
    est, res = TP.tooth_holds(holds)
    assert np.allclose(est.d, d_true, atol=0.3) and est.cond < 50
    assert max(np.linalg.norm(r) for r in res.values()) < 1.0
    assert est.rms_after_mm < est.rms_before_mm


def test_common_windows_intersect_still_runs():
    from biomech.validation.anatomy import Hold, common_windows

    a = {"max_point_1": Hold("max_point_1", "hp", np.zeros(3), 0, 10, 10.0, 14.0),
         "tmj_left_hp": Hold("tmj_left_hp", "hp", np.zeros(3), 0, 10, 0.0, 1.0)}
    b = {"max_point_1": Hold("max_point_1", "hp", np.zeros(3), 0, 10, 12.0, 15.0)}
    assert common_windows(a, b) == {"max_point_1": (12.0, 14.0)}
    b["max_point_1"].t0_s = 13.8                     # overlap 0.2 s < 0.5 s: no common window
    assert common_windows(a, b) == {}
