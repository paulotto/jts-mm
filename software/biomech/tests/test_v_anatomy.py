# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-system anatomical frames (V5), CT tip verification (V7) and reference quality (V8/V9)
on a synthetic two-system recording: the same physical teeth digitized by a CT whose pose
each system measures in its own (arbitrary) frames."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.validation import anatomy as AN
from biomech.validation import quality as QY
from biomech.validation import tip as TP
from biomech.validation.registration import make_pose

FS = 50.0
# physical landmarks: teeth in the jaw / head body (true body = a reference frame)
JAW = {"mand_point_1": [60.0, 0.0, -45.0], "mand_point_2": [25.0, 22.0, -40.0],
       "mand_point_3": [25.0, -22.0, -40.0], "tmj_left": [-35.0, 55.0, 5.0],
       "tmj_right": [-35.0, -55.0, 5.0]}
HEAD = {"max_point_1": [62.0, 0.0, -38.0], "max_point_2": [27.0, 23.0, -33.0],
        "max_point_3": [27.0, -23.0, -33.0]}
ORDER = ["mand_point_1", "mand_point_2", "mand_point_3", "max_point_1", "max_point_2",
         "max_point_3", "tmj_left", "tmj_right"]


def _rp(rng, s_rot, t):
    return make_pose(Rotation.from_rotvec(rng.normal(size=3) * s_rot).as_matrix(), np.asarray(t))


def _recording(seed=0, d_q=(0.0, 0.0, 0.0), noise=0.0):
    """Returns per-system dicts (t, tip_world, T_head, T_jaw), the phases and the truth."""
    rng = np.random.default_rng(seed)
    hold = 6.0
    t = np.arange(0.0, hold * len(ORDER) + 40.0, 1 / FS)
    n = len(t)
    # true head/jaw motion (world): head slowly moving, jaw opening in the last 40 s
    H = np.repeat(make_pose(np.eye(3), np.array([0.0, 0.0, 800.0]))[None], n, axis=0)
    H[:, :3, :3] = Rotation.from_rotvec(np.c_[0.05 * np.sin(0.3 * t), 0.03 * np.cos(0.2 * t),
                                              0.02 * np.sin(0.5 * t)]).as_matrix()
    ang = np.where(t > hold * len(ORDER), 0.3 * np.abs(np.sin(1.5 * t)), 0.0)
    rel = np.repeat(np.eye(4)[None], n, axis=0)
    rel[:, :3, :3] = Rotation.from_rotvec(np.c_[np.zeros(n), ang, np.zeros(n)]).as_matrix()
    J = np.einsum("nij,njk->nik", H, rel)
    # CT tip world position per hold; handle orientation wobbles around the tip
    tip = np.full((n, 3), np.nan)
    R_ct = Rotation.from_rotvec(rng.normal(size=(n, 3)) * 0.4).as_matrix()
    phases = []
    for k, name in enumerate(ORDER):
        a, b = k * hold, (k + 1) * hold - 0.5
        sel = (t >= a) & (t < b)
        body, pts = (J, JAW) if name in JAW else (H, HEAD)
        p = np.asarray(pts[name])
        tip[sel] = np.einsum("nij,j->ni", body[sel, :3, :3], p) + body[sel, :3, 3]
        wob = Rotation.from_rotvec(np.radians(12) * np.c_[np.sin(3 * t[sel]), np.cos(2 * t[sel]),
                                                           0.5 * np.sin(5 * t[sel])])
        R_ct[sel] = (Rotation.from_euler("x", 40 * k, degrees=True) * wob).as_matrix()
        phases.append(SimpleNamespace(name=name, t_start_us=int(a * 1e6), t_end_us=int(b * 1e6)))
    T_ct = np.repeat(np.eye(4)[None], n, axis=0)
    T_ct[:, :3, :3] = R_ct
    T_ct[:, :3, 3] = tip                                   # true CT frame origin = tip
    # system frames: world X_s, body frames Y_s (arbitrary, different per system)
    sys_ = {}
    for name, d in (("S", np.zeros(3)), ("Q", np.asarray(d_q))):
        X = _rp(rng, 1.0, rng.normal(size=3) * 300)
        Yh, Yj = _rp(rng, 1.0, rng.normal(size=3) * 40), _rp(rng, 1.0, rng.normal(size=3) * 40)
        Yc = make_pose(np.eye(3), -d)        # system's CT origin sits at -d from the tip
        Th = np.einsum("ij,njk,kl->nil", X, H, Yh)
        Tj = np.einsum("ij,njk,kl->nil", X, J, Yj)
        Tc = np.einsum("ij,njk,kl->nil", X, T_ct, Yc)
        if noise:
            for T_ in (Th, Tj, Tc):
                T_[:, :3, 3] += rng.normal(size=(n, 3)) * noise
        sys_[name] = dict(t=t, T_head=Th, T_jaw=Tj, T_ct=Tc, X=X)
    return sys_, phases, rel


def test_anatomical_frames_agree_although_body_frames_differ():
    sys_, phases, rel = _recording()
    M = {}
    for k, s in sys_.items():
        holds = AN.digitize(s["t"], s["T_ct"][:, :3, 3], s["T_head"], s["T_jaw"], phases)
        assert set(ORDER) <= set(holds)
        fr = AN.anatomical_frames(holds)
        M[k] = AN.jaw_in_head(s["T_head"], s["T_jaw"], fr)
    np.testing.assert_allclose(M["S"], M["Q"], atol=1e-6)
    assert np.nanmax(AN.rotation_error_deg(M["S"], M["Q"])) < 1e-4


def test_tip_offset_recovered_and_decided():
    d_true = np.array([1.5, -2.0, 3.0])
    sys_, phases, _ = _recording(seed=1, d_q=d_true, noise=0.05)
    S, Q = sys_["S"], sys_["Q"]
    # Qualisys landmarks with a zero tip are biased; the hold pivot finds d
    holds_q = AN.digitize(Q["t"], Q["T_ct"][:, :3, 3], Q["T_head"], Q["T_jaw"], phases)
    rel_h = []
    for name in ORDER:
        Tb = Q["T_jaw"] if AN.BODY_OF[name] == "mp" else Q["T_head"]
        idx = holds_q[name].idx
        rel_h.append(np.einsum("nij,njk->nik", se3.inv(Tb[idx]), Q["T_ct"][idx]))
    piv = TP.hold_pivot(rel_h)
    np.testing.assert_allclose(piv.d, d_true, atol=0.1)
    assert piv.rms_after_mm < piv.rms_before_mm
    # cross-system: X maps the JTS-MM world into the Qualisys world
    Xsq = Q["X"] @ se3.inv(S["X"])
    idx = np.concatenate([h.idx for h in holds_q.values() if not h.name.endswith("_hp")])
    cs = TP.cross_system(Xsq, S["T_ct"][idx, :3, 3], Q["T_ct"][idx])
    np.testing.assert_allclose(cs.d, d_true, atol=0.1)
    shared = TP.TipEstimate("shared", d_true + 0.05, np.eye(3) * 0.01, np.nan, np.nan, np.nan, 0)
    dec = TP.decide(shared, [piv, cs])
    assert dec["primary"] == "own_verified"
    far = TP.TipEstimate("shared", d_true + 2.0, np.eye(3) * 0.01, np.nan, np.nan, np.nan, 0)
    assert TP.decide(far, [piv, cs])["primary"] == "shared"


def test_reference_quality_gate_and_sigma():
    rng = np.random.default_rng(3)
    n = 400
    tmpl = np.array([[0.0, 0, 0], [30.0, 0, 0], [0.0, 60, 0]])
    T = np.repeat(make_pose(np.eye(3), np.array([0.0, 0.0, 900.0]))[None], n, axis=0)
    mk = {f"B - {k + 1}": se3.apply(T, tmpl[k]) + rng.normal(0, 0.05, (n, 3))
          for k in range(3)}
    mk["B - 2"][100] += [2.0, 0.0, 0.0]     # one corrupted frame (along a pair: distances see it)
    mt = {k: np.ones(n, np.int8) for k in mk}
    mt["B - 3"][200] = QY.GAPFILLED
    q = QY.consistency(mk, None, mt)
    assert q["gate"].iloc[100] and q["gate"].iloc[200] and q["gate"].sum() <= 3
    s = QY.sigma_marker(q.drop(index=[100]))
    assert s == pytest.approx(0.05, rel=0.25)
    # σ_ref: zero noise -> zero; scales with σ
    rel = np.repeat(np.eye(4)[None], 20, axis=0)
    F = make_pose(np.eye(3), np.zeros(3))
    z = QY.sigma_ref(tmpl, tmpl, 0.0, 0.0, rel, F, F, {"tmj_left": np.array([-80.0, 50, 0])})
    assert z["incisor_mm"] < 1e-9
    a = QY.sigma_ref(tmpl, tmpl, 0.1, 0.1, rel, F, F, {"tmj_left": np.array([-80.0, 50, 0])})
    b = QY.sigma_ref(tmpl, tmpl, 0.2, 0.2, rel, F, F, {"tmj_left": np.array([-80.0, 50, 0])})
    assert b["incisor_mm"] == pytest.approx(2 * a["incisor_mm"], rel=0.2)
    assert a["tmj_mm"] > a["incisor_mm"]                     # lever arm to the condyle
    bud = QY.budget(2.0, 0.5)
    assert bud["adequate"] and bud["sigma_jts_corrected"] == pytest.approx(np.sqrt(3.75))
