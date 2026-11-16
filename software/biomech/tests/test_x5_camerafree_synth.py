# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic acceptance gate of the camera-free study (dataset X5/D28, VD20).

The five acceptance tests of H4.4, on takes from :mod:`biomech.testing.synth_camerafree`
with a known truth; all are required before any real-data run. The bounds are the plan's: a
failing bound is fixed in the method (:mod:`biomech.imu.camerafree`), never in the bound.

Opening and rotation errors are RMS over the scored tasks after the ground truth's 6 Hz
low-pass (``camerafree.score``), unless a per-sample bound is stated.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.imu import camerafree as cf
from biomech.imu.calibration import ImuCalibration
from biomech.testing.synth_camerafree import CFSpec, CFSynth, generate, head_tilt_range_deg

SCORED = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
MOVING_HEAD = dict(head_pitch_deg=6.0, head_roll_deg=3.0, head_yaw_deg=4.0, head_sway_mm=3.0)
STATIC_HEAD = dict(head_pitch_deg=0.0, head_roll_deg=0.0, head_yaw_deg=0.0, head_sway_mm=0.0)
EZ = np.array([0.0, 0.0, 1.0])


def _ladder(syn: CFSynth) -> cf.RotationLadder:
    tr = syn.truth
    return cf.rotation_ladder(tr.t_us, syn.cra, syn.mta, tr.A, tr.mask("rest"), X=tr.X,
                              R_closed=tr.T_max_mp_closed[:3, :3])


def _geometry(syn: CFSynth) -> cf.JawGeometry:
    tr = syn.truth
    return cf.jaw_geometry(tr.T_max_hp, tr.T_max_mp_closed, tr.lever_cra_mm, tr.lever_mta_mm,
                           tr.incisor0_mm, tr.condyle0_mm)


def _quats(syn: CFSynth) -> tuple[np.ndarray, np.ndarray]:
    return syn.cra.quat_at(syn.truth.t_us), syn.mta.quat_at(syn.truth.t_us)


def _errors_5deg(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=3)
    return Rotation.from_rotvec(np.radians(5.0) * v / np.linalg.norm(v)).as_matrix()


@pytest.fixture(scope="module")
def noise_free() -> tuple[CFSynth, cf.RotationLadder]:
    """Everything exact, the head moving; no translation outside the E4 model family except
    in protrusion_retrusion and left_right (free_translation_mm = 0)."""
    syn = generate(CFSpec(**MOVING_HEAD, free_translation_mm=0.0))
    return syn, _ladder(syn)


@pytest.fixture(scope="module")
def static_head() -> CFSynth:
    return generate(CFSpec())


@pytest.fixture(scope="module")
def drifting() -> tuple[CFSynth, cf.RotationLadder]:
    """Realistic sensors, natural head tilt, heading drift 1°/min (cra +0.6, mta −0.4) plus a
    random walk, two rounds of the protocol (≈ 2.7 min)."""
    syn = generate(CFSpec.realistic(n_rounds=2, heading_drift_dpm=(0.6, -0.4),
                                    heading_rw_deg=(0.2, 0.2)))
    return syn, _ladder(syn)


# -- 1. noise-free -----------------------------------------------------------------------------
def test_1_noise_free_rotation(noise_free):
    """E2/E3 < 0.05° (opening and geodesic RMS), every rung and mode."""
    syn, lad = noise_free
    tr = syn.truth
    for name, R in lad.estimates.items():
        s = cf.score(tr.t_s, R, tr.R_rel, mask=tr.mask(*SCORED))
        assert s["rmse_open_deg"] < 0.05, (name, s)
        assert s["rmse_geo_deg"] < 0.05, (name, s)
    assert abs(np.degrees(lad.psi0["a"] - tr.psi_rad[0])) < 0.01
    assert lad.psi_b.identifiable and abs(np.degrees(lad.psi_b.psi_rad - tr.psi_rad[0])) < 0.01


def test_1_noise_free_e4own(noise_free):
    """E4own < 0.1 mm (incisor 3-D RMS, open_close and chewing) when the model family
    contains the truth: the ridge model fitted on the optical (true) free_motion block, then
    driven by the E3 rotation."""
    syn, lad = noise_free
    tr = syn.truth
    t = tr.t_s
    cal = tr.mask("free_motion")
    model = cf.TranslationModel.fit(tr.rotvec_deg[cal],
                                    cf.theta_rate_dps(t, tr.rotvec_deg[:, 1])[cal], tr.tau_mm[cal])
    R3 = lad.estimates["E3b"]
    rv = np.degrees(Rotation.from_matrix(R3).as_rotvec())
    tau = model.predict(rv, cf.theta_rate_dps(t, rv[:, 1]))
    geom = _geometry(syn)
    inc = cf.incisor_position(R3, tau, geom.incisor0, geom.condyle0)
    s = cf.score(t, R3, tr.R_rel, inc, tr.incisor_mm, tr.mask("open_close", "chewing"))
    assert s["rmse_inc_mm"] < 0.1, s
    hinge = cf.incisor_position(R3, None, geom.incisor0, geom.condyle0)
    assert cf.score(t, R3, tr.R_rel, hinge, tr.incisor_mm,
                    tr.mask("open_close", "chewing"))["rmse_inc_mm"] > 1.0   # τ matters


def test_1_noise_free_e5(noise_free):
    """E5 < 0.2 mm (incisor 3-D RMS, every scored task): accelerometer bridging between the
    E3 anchor windows (closed-pose anchors, rest-phase noise)."""
    syn, lad = noise_free
    tr = syn.truth
    an = lad.anchored["E3b"]
    br = cf.accel_bridge(tr.t_s, an.R_rel, an.psi_rad, syn.cra, syn.mta, tr.A, _geometry(syn),
                         an.anchor_mask(), tr.mask("rest"))
    for task in SCORED:
        s = cf.score(tr.t_s, an.R_rel, tr.R_rel, br.incisor_mm, tr.incisor_mm, tr.mask(task))
        assert s["rmse_inc_mm"] < 0.2, (task, s)


# -- 2. injected mounting / heading errors -----------------------------------------------------
def test_2_x_error_leaves_opening_unchanged(noise_free):
    """A 5° error of X (three axes) changes the opening by < 0.01°: the relative rotation of
    the calibration model, ``predict(t) predict(c)ᵀ``, does not contain X (head moving)."""
    syn, _ = noise_free
    tr = syn.truth
    qc, qm = _quats(syn)
    rest = tr.mask("rest")
    psi = float(tr.psi_rad[0])

    def opening(X: np.ndarray) -> np.ndarray:
        P = ImuCalibration(tr.A, psi, X, 0.0, 0, "synthetic").predict(qc, qm)
        Pc = Rotation.from_matrix(P[rest]).mean().as_matrix()
        return cf.opening_deg(P @ Pc.T)

    ref = opening(tr.X)
    th = cf.opening_deg(tr.R_rel)
    for seed in range(3):
        o = opening(tr.X @ _errors_5deg(seed))
        assert np.max(np.abs(o - ref)) < 0.01
        assert np.max(np.abs(o - th)) < 0.01


def test_2_a_error_only_tilts_the_axis(noise_free):
    """A 5° error of A: per sample |Δθ| ≤ θ(1 − cos 5°) + 0.02° in open_close (pure opening
    rotation; head moving -- an A error is a constant conjugation)."""
    syn, _ = noise_free
    tr = syn.truth
    qc, qm = _quats(syn)
    psi = float(tr.psi_rad[0])
    M_ref = cf.closed_reference(qc, qm, psi, tr.mask("rest"))
    th = cf.opening_deg(tr.R_rel)
    m = tr.mask("open_close")
    bound = np.abs(th[m]) * (1 - np.cos(np.radians(5.0))) + 0.02
    for seed in range(3):
        o = cf.opening_deg(cf.relative_rotation(qc, qm, tr.A @ _errors_5deg(seed), psi, M_ref))
        assert np.all(np.abs(o - th)[m] <= bound)


def test_2_psi_error_static_head(static_head):
    """A 5° error of ψ: per sample |Δθ| ≤ θ(1 − cos 5°) + 0.02° in open_close, with the head
    keeping its tilt (the premise of H4.0: then the ψ error is a constant conjugation)."""
    syn = static_head
    tr = syn.truth
    qc, qm = _quats(syn)
    th = cf.opening_deg(tr.R_rel)
    m = tr.mask("open_close")
    bound = np.abs(th[m]) * (1 - np.cos(np.radians(5.0))) + 0.02
    for d in (5.0, -5.0):
        psi = float(tr.psi_rad[0]) + np.radians(d)
        o = cf.opening_deg(cf.relative_rotation(qc, qm, tr.A, psi,
                                                cf.closed_reference(qc, qm, psi, tr.mask("rest"))))
        assert np.all(np.abs(o - th)[m] <= bound)


def test_2_psi_error_couples_with_head_tilt(noise_free):
    """Documents why ψ is re-fitted at every anchor: with head tilt φ since the reference,
    a ψ error δ adds ≈ δ·sin φ (Rz(ψ) turns about the head's vertical, which tilts with the
    head), so the static bound does not hold; δ·φ bounds the excess."""
    syn, _ = noise_free
    tr = syn.truth
    qc, qm = _quats(syn)
    rest = tr.mask("rest")
    th = cf.opening_deg(tr.R_rel)
    v = np.einsum("nji,j->ni", tr.H, EZ)                       # head vertical, maxillary frame
    v_ref = v[rest].mean(0) / np.linalg.norm(v[rest].mean(0))
    phi = np.arccos(np.clip(v @ v_ref, -1, 1))
    psi = float(tr.psi_rad[0]) + np.radians(5.0)
    M_ref = cf.closed_reference(qc, qm, psi, rest)
    o = cf.opening_deg(cf.relative_rotation(qc, qm, tr.A, psi, M_ref))
    m = tr.mask("open_close")
    static = np.abs(th[m]) * (1 - np.cos(np.radians(5.0))) + 0.02
    assert np.any(np.abs(o - th)[m] > static)                 # the static bound is exceeded
    assert np.all(np.abs(o - th)[m] <= static + np.radians(5.0) * np.degrees(phi[m]))


# -- 3. drift ----------------------------------------------------------------------------------
def test_3_drift_e3_below_0p3(drifting):
    """Heading drift 1°/min with head tilts ≥ 10°: E3 < 0.3° (opening RMS, every scored
    task), on E2a and on E2b, causal and offline."""
    syn, lad = drifting
    tr = syn.truth
    assert head_tilt_range_deg(tr) >= 10.0
    drift = np.degrees(tr.psi_rad[-1] - tr.psi_rad[0])
    assert abs(drift) > 2.0                                    # the drift is there
    for name in ("E3a", "E3a_slerp", "E3b", "E3b_slerp"):
        s = cf.score(tr.t_s, lad.estimates[name], tr.R_rel, mask=tr.mask(*SCORED))
        assert s["rmse_open_deg"] < 0.3, (name, s)


def test_3_e2b_recovers_psi(drifting):
    """E2b recovers ψ within 0.5° (against the circular mean of the true, drifting ψ over the
    closures)."""
    syn, lad = drifting
    tr = syn.truth
    est = lad.psi_b
    assert est.identifiable and est.tilt_range_deg >= cf.PSI_MIN_TILT_RANGE_DEG
    truth = np.angle(np.mean(np.exp(1j * tr.psi_rad[tr.closed_mask()])))
    assert abs(np.degrees(cf._wrap(est.psi_rad - truth))) < 0.5


def test_3_psi_not_identifiable_with_static_head():
    """A static head: "not identifiable" (NaN ψ), E2b falls back to E2a (H4.8)."""
    syn = generate(CFSpec.realistic(**STATIC_HEAD, heading_drift_dpm=(0.6, -0.4)))
    tr = syn.truth
    qc, qm = _quats(syn)
    est = cf.psi_consistency(tr.t_s, qc, qm, tr.A, tr.mask("rest"))
    assert not est.identifiable and np.isnan(est.psi_rad)
    assert est.tilt_range_deg < cf.PSI_MIN_TILT_RANGE_DEG
    lad = _ladder(syn)
    assert lad.flags["psi_b_fallback"] and lad.psi0["b"] == lad.psi0["a"]


# -- 4. detector -------------------------------------------------------------------------------
def test_4_detector_finds_closures(drifting):
    """The detector finds ≥ 90 % of the closures (a closure is found when a detected run
    overlaps it). Every anchor window is a closed pose: it overlaps a true closure, and the
    true jaw pose averaged over the window (what the anchor is set from) lies within 0.05°
    and 0.05 mm (incisor) of that closure's pose. A window may include tens of milliseconds
    of the closing tail: in laterotrusion the jaw slides ≈ 4 mm per degree of opening, below
    the opening's noise."""
    syn, lad = drifting
    tr = syn.truth
    t = tr.t_s
    for name in ("E3a", "E3b"):
        an = lad.anchored[name]
        found = [an.closed.mask[(t >= a) & (t <= b)].any() for a, b in tr.closures]
        assert np.mean(found) >= 0.9, (name, np.mean(found))
        for w0, w1 in zip(an.anchors["w0"], an.anchors["w1"]):
            over = [(min(b, t[w1 - 1]) - max(a, t[w0]), a, b) for a, b in tr.closures]
            ov, a, b = max(over)
            assert ov > 0, (name, t[w0])
            mid = int(np.argmin(np.abs(t - 0.5 * (a + b))))
            d_rot = np.linalg.norm(tr.rotvec_deg[w0:w1].mean(0) - tr.rotvec_deg[mid])
            d_inc = np.linalg.norm(tr.incisor_mm[w0:w1].mean(0) - tr.incisor_mm[mid])
            assert d_rot < 0.05 and d_inc < 0.05, (name, t[w0], d_rot, d_inc)


# -- 5. drift signatures -----------------------------------------------------------------------
MECHANISMS = {
    "none": {},
    "heading": {"heading_drift_dpm": (0.6, -0.4)},
    "tilt": {"tilt_drift_dpm": 1.0},
    "slip": {"slip_deg": 3.0},
}


@pytest.mark.parametrize("mech", list(MECHANISMS))
def test_5_drift_mechanism_signature(mech):
    """Each drift mechanism leaves its own anchor-residual signature (``drift_signature``):

    * heading: heading part ≈ the drift, tilt part and fusion-tilt change small;
    * tilt (jaw IMU fusion): tilt part ≈ the fusion-vs-accelerometer tilt change;
    * slip: one step (≥ 50 % of the net change between two anchors), fusion unchanged;
    * none: net residual below the floor.
    """
    syn = generate(CFSpec.realistic(n_rounds=2, **MECHANISMS[mech]))
    tr = syn.truth
    qc, qm = _quats(syn)
    rest = tr.mask("rest")
    psi0 = cf.psi_static(qc[rest], qm[rest], tr.A, tr.X, tr.T_max_mp_closed[:3, :3])
    an = cf.anchored_rotation(tr.t_s, qc, qm, tr.A, psi0, cf.closed_reference(qc, qm, psi0, rest),
                              rest)
    sig = cf.drift_signature(an, tr.t_s, syn.cra, syn.mta)
    assert sig["label"] == mech, sig
    span_min = (an.anchors["t0_s"][-1] - an.anchors["t0_s"][0]) / 60.0
    if mech == "none":
        assert sig["net_deg"] < cf.SIGNATURE["floor_deg"]
    elif mech == "heading":
        assert sig["heading_deg"] > 0.7 * span_min * 1.0
        assert sig["tilt_deg"] < 0.25 * sig["heading_deg"]
        assert sig["fusion_tilt_deg"] < cf.SIGNATURE["floor_deg"]
    elif mech == "tilt":
        assert sig["tilt_deg"] > 0.7 * span_min * 1.0
        assert abs(sig["fusion_tilt_deg"] - sig["tilt_deg"]) < 0.3 * sig["tilt_deg"]
        assert sig["heading_deg"] < 0.25 * sig["tilt_deg"]
    else:
        assert sig["step_frac"] >= cf.SIGNATURE["step_frac"]
        assert sig["fusion_tilt_deg"] < cf.SIGNATURE["floor_deg"]
        assert 0.7 * 3.0 < sig["net_deg"] < 1.3 * 3.0


# -- building blocks ---------------------------------------------------------------------------
def test_mounting_prior_reproduces_the_calibration_model(noise_free):
    """``mounting_prior`` builds A and X of ``ImuCalibration`` from hand-eye mountings and the
    landmark frame; with them and the true ψ the model reproduces R_max<-mp."""
    syn, _ = noise_free
    tr = syn.truth
    mp = cf.mounting_prior(tr.R_mount_cra, tr.R_mount_mta, tr.T_max_hp)
    assert np.allclose(mp.A, tr.A) and np.allclose(mp.X, tr.X)
    qc, qm = _quats(syn)
    R = ImuCalibration(mp.A, float(tr.psi_rad[0]), mp.X, 0.0, 0, "synthetic").predict(qc, qm)
    assert np.allclose(R, tr.R_rel @ tr.T_max_mp_closed[:3, :3], atol=1e-4)


def test_generic_translation_model_scales(noise_free):
    """E4gen: pooled over participants of different size (τ scaled with the incisor–condyle
    distance), the generic model predicts a new participant's τ from its distance."""
    syn, _ = noise_free
    tr = syn.truth
    t = tr.t_s
    cal = tr.mask("free_motion")
    th_d = cf.theta_rate_dps(t, tr.rotvec_deg[:, 1])
    d0 = float(np.linalg.norm(tr.incisor0_mm - tr.condyle0_mm))
    blocks = [(tr.rotvec_deg[cal], th_d[cal], tr.tau_mm[cal] * k, d0 * k) for k in (0.9, 1.0, 1.15)]
    gen = cf.TranslationModel.fit_generic(blocks)
    m = tr.mask("open_close", "chewing")
    pred = gen.predict(tr.rotvec_deg[m], th_d[m], scale_mm=1.05 * d0)
    assert np.sqrt(np.mean(np.sum((pred - 1.05 * tr.tau_mm[m]) ** 2, axis=1))) < 0.1
    with pytest.raises(ValueError):
        gen.predict(tr.rotvec_deg[m], th_d[m])
