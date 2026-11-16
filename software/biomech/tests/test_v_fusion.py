# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Adapter for the optical + IMU fusion (biomech.validation.fusion) on a synthetic take."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from jtsmm.imu import fusion as FU

from biomech.validation import fusion as VF
from biomech.validation import selection as SEL
from biomech.validation.kinematics import mask_of

T0_US = 1_000_000_000          # master clock offset
EXPO = 0.006                   # half exposure: frame time = trigger + 6 ms


@dataclass
class _Phase:
    name: str
    t_start_us: int
    t_end_us: int


def _us(t_s: float) -> int:
    return int(round(t_s * 1e6)) + T0_US


class _FakeSession:
    """Two bodies simulated independently on one optical grid, a CT visible in the start
    and end sync windows, phases as in the comparison takes (seconds from the start)."""

    def __init__(self) -> None:
        sims = {"cra": FU.simulate(90.0, seed=11, amp_deg=4.0),
                "mta": FU.simulate(90.0, seed=12)}
        t = sims["cra"]["t_opt_s"]
        n = len(t)
        self.sims, self.t_rel = sims, t
        poses, quality = {}, {}
        for body, sim in sims.items():
            T = np.full((n, 4, 4), np.nan)
            T[:, :3, :3] = sim["R_opt"]
            T[:, :3, 3] = np.array([10.0, 20.0, 300.0]) + np.sin(t)[:, None]
            T[:, 3, :] = (0, 0, 0, 1)
            T[~np.isfinite(sim["R_opt"].reshape(n, -1)).all(1)] = np.nan
            poses[body] = T
            quality[body] = sim["quality"]
        ct = np.full((n, 4, 4), np.nan)
        vis = ((t >= 11.0) & (t <= 14.0)) | ((t >= 85.0) & (t <= 88.0))
        ct[vis] = np.eye(4)
        poses["ct"] = ct
        t_us = (np.round((t - EXPO) * 1e6).astype(np.int64) + T0_US).astype(np.uint64)
        self._opt = SimpleNamespace(t_us=t_us, poses=poses, quality=quality)
        self.phases = [_Phase("mand_point_1", _us(0.0), _us(15.0)),
                       _Phase("rest", _us(1.5), _us(10.5)),
                       _Phase("open_close", _us(16.0), _us(35.0)),
                       _Phase("chewing", _us(36.0), _us(55.0)),
                       _Phase("free_motion", _us(56.0), _us(89.0))]

    def optical(self):
        return self._opt

    def imu(self, slot: str):
        sim = self.sims[slot]
        t = np.asarray(sim["imu_t_us"], np.int64) + T0_US
        return SimpleNamespace(t_us=t, quat=sim["imu_q"], quat_accuracy=np.full(len(t), 3),
                               n=len(t))

    def truth(self, body: str, t_abs_s: np.ndarray) -> np.ndarray:
        return self.sims[body]["truth"](t_abs_s - T0_US * 1e-6)


def _rms(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


@pytest.fixture(scope="module")
def run():
    sd = _FakeSession()
    tf = sd.t_rel + T0_US * 1e-6
    ct_ok = np.isfinite(sd._opt.poses["ct"].reshape(len(tf), -1)).all(1)
    win = SEL.sync_windows(sd.phases, tf, ct_ok)
    held = SEL.movement_intervals(sd.phases, tf, ct_ok)
    chew = [h for h in held if h.name == "chewing"][0]
    train = win + [h for h in held if h.name in ("open_close", "free_motion")]
    tq = np.arange(chew.t0, chew.t1, 0.005)
    ctx = VF.prepare(sd, {}, lambda t: mask_of(t, train), lambda t: mask_of(t, [chew]))
    out = VF.fuse(ctx, tq)
    return {"sd": sd, "ctx": ctx, "out": out, "tq": tq, "chew": chew, "train": train,
            "win": win}


def test_windows_and_masks_are_as_designed(run):
    assert [w.name for w in run["win"]] == ["sync_start", "sync_end"]
    ctx = run["ctx"]
    assert not np.any(ctx.train_mask & ctx.eval_mask)
    assert ctx.eval_mask.sum() > 500 and ctx.train_mask.sum() > 1000
    assert ctx.rest_intervals and ctx.rest_intervals[0][1] - ctx.rest_intervals[0][0] > 7.0


def test_overlapping_training_and_evaluation_is_refused(run):
    sd, chew = run["sd"], run["chew"]
    with pytest.raises(ValueError, match="overlap"):
        VF.prepare(sd, {}, lambda t: mask_of(t, run["train"] + [chew]),
                   lambda t: mask_of(t, [chew]))


def test_output_structure(run):
    out, tq = run["out"], run["tq"]
    for B in ("HP", "MP"):
        o = out[B]
        # the synthetic take has no accelerometer stream: no position fusion (F2P)
        no_acc = o["pos_settings"] is None
        assert set(o["R"]) == set(VF.ESTIMATORS) - ({"F2P", "F2P_causal"} if no_acc else set())
        for est, R in o["R"].items():
            assert R.shape == (len(tq), 3, 3)
            assert o["T"][est].shape == (len(tq), 4, 4)
        assert o["p"].shape == (len(tq), 3) and o["p_gap_s"].shape == (len(tq),)
        cal = o["calibration"]
        assert {"Z", "X", "lag_ms", "lag_se_ms", "rms_deg", "drift_heading_deg_min",
                "inlier_frac"} <= set(cal)
        assert {"tau_s", "sigma_opt_deg_axis", "q_theta", "derivation"} <= set(o["settings"])
        d2 = o["diagnostics"]["F2"]
        assert {"nis", "gated", "n_gated", "nis_mean", "nis_frac95", "resets"} <= set(d2)
        assert o["imu"]["rate_hz"] == pytest.approx(350, rel=0.05)
        assert o["imu"]["per_segment"]["chewing"]["coverage"] == 1.0
    assert out["HP"]["body"] == "cra" and out["MP"]["body"] == "mta"


def test_pooled_lag_and_calibration(run):
    ctx = run["ctx"]
    lag = ctx.bodies["MP"].imu_info["lag"]
    assert lag["mode"] == "pooled"
    assert lag["lag_ms"] == pytest.approx(10.0, abs=1.5)
    assert ctx.bodies["HP"].calibration.lag_s == ctx.bodies["MP"].calibration.lag_s
    for B, body in (("HP", "cra"), ("MP", "mta")):
        sim = run["sd"].sims[body]
        assert FU.rot_error_deg(ctx.bodies[B].calibration.X, sim["X"]) < 0.5


def test_fused_rotation_beats_optical_only_between_frames(run):
    out, tq, sd = run["out"], run["tq"], run["sd"]
    for B, body in (("HP", "cra"), ("MP", "mta")):
        Rt = sd.truth(body, tq)
        e0 = FU.rot_error_deg(out[B]["R"]["F0"], Rt)
        flips = sd.sims[body]["flips"]
        near = np.zeros(len(tq), bool)
        for k in flips:
            near |= np.abs(tq - (sd.t_rel[k] + T0_US * 1e-6)) < 0.03
        for est in ("F1", "F2", "F2_causal"):
            e = FU.rot_error_deg(out[B]["R"][est], Rt)
            assert _rms(e) < 0.7 * _rms(e0[~near]), (B, est)


def test_position_stays_optical(run):
    ctx = run["ctx"]
    t = ctx.t_frames_s
    k = np.flatnonzero(ctx.eval_mask)[::7]
    out = VF.fuse(ctx, t[k], estimators=("F0",))
    p_opt = ctx.bodies["MP"].p_opt[k]
    ok = np.isfinite(p_opt).all(1)
    assert np.allclose(out["MP"]["p"][ok], p_opt[ok])


def test_dropout_masks_and_bridging(run):
    ctx, chew, sd = run["ctx"], run["chew"], run["sd"]
    t = ctx.t_frames_s
    masks = VF.dropout_masks(t, [chew])
    assert set(masks) == set(VF.DROPOUT_LENGTHS_S)
    dt = float(np.median(np.diff(t)))
    for L, m in masks.items():
        assert m.any() and not np.any(m & ~ctx.eval_mask)
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])))[::2]
        assert np.all(np.abs(runs * dt - L) <= dt + 1e-9)
    again = VF.dropout_masks(t, [(chew.t0, chew.t1)])
    assert all(np.array_equal(again[L], masks[L]) for L in masks)
    m = masks[1.0]
    out = VF.fuse(ctx, t[m], drop_mask=m)
    Rt = sd.truth("mta", t[m])
    e_hold = _rms(FU.rot_error_deg(out["MP"]["R"]["F0_hold"], Rt))
    e_f2 = _rms(FU.rot_error_deg(out["MP"]["R"]["F2"], Rt))
    assert e_f2 < 0.3 * e_hold
    assert out["drop_mask"] is not None


def test_run_fusion_defaults_to_frame_times(run):
    sd, chew = run["sd"], run["chew"]
    out = VF.run_fusion(sd, {"lag_mode": 0.010, "noise_source": "rest"},
                        lambda t: mask_of(t, run["train"]),
                        eval_mask_fn=lambda t: mask_of(t, [chew]))
    assert len(out["t_query_s"]) == len(out["t_frames_s"])
    assert out["MP"]["calibration"]["lag_ms"] == pytest.approx(10.0)
    assert out["MP"]["settings"]["derivation"]["source"] == "rest"
    assert out["train_mask"].sum() > 0 and out["eval_mask"].sum() > 0


def test_load_mounting_reads_accepted_rotations(tmp_path):
    import json

    from scipy.spatial.transform import Rotation

    from biomech.validation.fusion import load_mounting

    R = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    f = tmp_path / "imu_mounting.json"
    f.write_text(json.dumps({"imu_mta": {"R_mount": R.tolist(), "accepted": True},
                             "imu_cra": {"T": T.tolist(), "accepted": False}}))
    np.testing.assert_allclose(load_mounting(f, "mta"), R)
    assert load_mounting(f, "cra") is None                      # rejected fit is never used
    g = tmp_path / "slots.json"
    g.write_text(json.dumps({"slots": {"cra": {"T": T.tolist()}}}))
    np.testing.assert_allclose(load_mounting(g, "cra"), R)
