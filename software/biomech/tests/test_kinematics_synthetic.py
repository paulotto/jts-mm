# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Kinematics on synthetic jaw motion with known ground truth."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.frames.landmarks import dental_roles
from biomech.frames.maxillary import check, maxillary_frame
from biomech.kinematics import se3
from biomech.kinematics.cycles import detect
from biomech.kinematics.helical import screw, stepped_screws
from biomech.kinematics.hinge_fit import fit_hinge, fit_rot_trans, hinge_through

# Mandibular points in maxillary-like coordinates at closure (mm): incisor,
# left/right first molars, left/right condyles.
P = np.array([[0.0, 0.0, 0.0], [-35.0, 22.0, 2.0], [-35.0, -22.0, 2.0],
              [-85.0, 50.0, 25.0], [-85.0, -50.0, 25.0]])
AXIS_PT = np.array([-80.0, 0.0, 20.0])


def _pose(theta_rad: float, trans: np.ndarray = np.zeros(3)) -> np.ndarray:
    R = Rotation.from_rotvec([0.0, theta_rad, 0.0]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = AXIS_PT - R @ AXIS_PT + trans
    return T


def _opening(n: int = 60, max_deg: float = 25.0, translate_mm: float = 0.0,
             onset_deg: float = 10.0):
    """Rotation first; condylar translation only beyond ``onset_deg`` (as in vivo).

    A translation strictly proportional to the angle would be indistinguishable
    from rotation about a shifted axis: the hinge residual measures axis
    migration, not translation as such.
    """
    th = np.radians(np.linspace(0, max_deg, n))
    on = np.radians(onset_deg)
    frac = np.clip((th - on) / (th[-1] - on), 0, 1)
    T = np.stack([_pose(t, translate_mm * f * np.array([1.0, 0, -0.3])) for t, f in zip(th, frac)])
    A = np.broadcast_to(se3.apply(T[0][None], P[None]), (n, len(P), 3))
    B = np.einsum("fij,kj->fki", T[:, :3, :3], P) + T[:, None, :3, 3]
    return T, np.ascontiguousarray(A), B, np.degrees(th)


def test_screw_recovers_pure_hinge():
    sc = screw(_pose(0.0), _pose(np.radians(20)))
    assert np.isclose(sc.phi_deg, 20.0) and abs(sc.t_mm) < 1e-9
    assert np.allclose(np.abs(sc.n), [0, 1, 0], atol=1e-9)
    assert np.allclose(sc.pierce(), AXIS_PT, atol=1e-6)


def test_stepped_screws_spacing():
    T, _, _, deg = _opening()
    steps = stepped_screws(T, deg, 5.0)
    assert len(steps) == 4 and all(sc.phi_deg >= 5.0 - 1e-9 for _, _, sc in steps)


def test_hinge_fit_zero_residual_for_pure_hinge():
    _, A, B, _ = _opening()
    res = fit_hinge(A, B, s0=AXIS_PT + np.array([5.0, 0, -5.0]))
    assert res.rms(0) < 1e-3 and np.allclose(np.abs(res.n), [0, 1, 0], atol=1e-4)


def test_translation_breaks_hinge_but_not_rot_trans():
    _, A, B, _ = _opening(translate_mm=8.0)
    hinge = fit_hinge(A, B, s0=AXIS_PT)
    rt = fit_rot_trans(A, B, hinge)
    assert hinge.rms(0) > 0.3 and rt.rms(0) < 0.05
    assert np.ptp(rt.tau_mm) > 7.0
    tmj = hinge_through(A, B, A[0, 3], A[0, 4])
    assert tmj.rms(0) > hinge.rms(0)


def test_dental_roles_and_frame_from_geometry():
    tl, tr = P[3] + [0, 15, 0], P[4] - [0, 15, 0]
    pts = {"p1": P[1], "p2": P[0], "p3": P[2]}       # incisor deliberately not first
    roles = dental_roles(pts, tl, tr)
    assert (roles.incisor, roles.left, roles.right) == ("p2", "p1", "p3") and roles.geometry_ok
    T = maxillary_frame(P[1], P[2], P[0])
    c = check(se3.inv(T), {"tmj_left": tl, "tmj_right": tr})
    assert c["lr_ok"] and c["posterior_ok"] and c["superior_ok"]


def test_cycle_detection():
    fs = 40.0
    t = np.arange(0, 12, 1 / fs)
    x = 20 * (1 - np.cos(2 * np.pi * 0.5 * t)) / 2
    x[100:104] = np.nan
    cyc = detect(x, fs)
    assert len(cyc) == 6 and all(abs(x[c.peak] - 20) < 0.5 for c in cyc if c.ok)


def test_auto_labels_open_close_and_free_motion():
    from biomech.labels.auto_labels import CODE, cohen_kappa, label_phase

    fs = 40.0
    t = (np.arange(0, 8, 1 / fs) * 1e6).astype(np.int64)
    z = -15 * (1 - np.cos(2 * np.pi * 0.5 * t * 1e-6))
    z[(t > 6e6)] = z[np.searchsorted(t, 6e6)]          # hold still at the end
    inc = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    p = {"vel_cutoff_hz": 4.0, "hi_frac": 0.2, "lo_frac": 0.1, "min_seg_s": 0.1}
    lab = label_phase("open_close", t, inc, fs, p)
    opening = (np.gradient(z) < -0.5)
    assert np.mean(lab[opening] == CODE["OPEN"]) > 0.9
    assert np.mean(lab[t > 6.5e6] == CODE["REST"]) > 0.9
    lab_fm = label_phase("free_motion", t, inc, fs, p)
    assert cohen_kappa(lab, lab_fm) > 0.9


def test_roles_follow_convention_and_flag_swaps():
    from biomech.frames.landmarks import condyle_check

    tl, tr = P[3] + [0, 15, 0], P[4] - [0, 15, 0]
    pts = {"m1": P[0], "m2": P[1], "m3": P[2]}       # incisor, left, right: as documented
    conv = {"incisor": "m1", "left": "m2", "right": "m3"}
    r = dental_roles(pts, tl, tr, conv)
    assert (r.incisor, r.left, r.right) == ("m1", "m2", "m3") and r.agrees and not r.relabelled
    swapped = {"m1": P[0], "m2": P[2], "m3": P[1]}   # operator touched the molars in reverse
    r = dental_roles(swapped, tl, tr, conv)
    assert (r.left, r.right) == ("m3", "m2") and r.relabelled and not r.agrees
    c = condyle_check(P[0], P[3], P[4])
    assert c.plausible and abs(c.intercondylar_mm - 100.0) < 1e-9
    assert not condyle_check(P[0], np.array([-20.0, 50, 0]), P[4]).plausible


def test_landmarks_match_the_operator_prompts():
    import pytest

    from biomech.frames.landmarks import LANDMARKS
    exp = pytest.importorskip("optical.tracking.experiment")
    hints = getattr(exp, "LANDMARK_HINTS", None)
    if hints is None:
        pytest.skip("optical without LANDMARK_HINTS")
    assert {k: v.description for k, v in LANDMARKS.items()} == dict(hints)
