# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD24: stored diagnostics of the validation study on synthetic inputs -- frame
timing, mounting helpers, the acceleration peak ratio of VD9, and the dynamic-calibration
audit against the independent script ``scripts/audit_recompute/imu_dyncal.py``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.validation import diagnostics as DG
from biomech.validation import sampling as S

ROOT = Path(__file__).resolve().parents[3]


# -- frame timing ---------------------------------------------------------------------------------
def test_frame_timing_separates_jitter_from_dropped_frames():
    rng = np.random.default_rng(0)
    n = 4000
    t = np.arange(n) * 0.025 + rng.normal(0, 3e-6, n)
    keep = np.ones(n, bool)
    keep[100:101] = False                           # one 50-ms gap
    keep[[500, 900, 1300]] = False                  # three more
    keep[2000:2002] = False                         # one 75-ms gap
    t = np.r_[t[keep], np.nan]                      # a NaN frame time is ignored
    summ, gaps = DG.frame_timing("s01", t)
    r = summ.iloc[0]
    assert r.n_intervals == keep.sum() - 1
    assert abs(r.frame_interval_ms - 25.0) < 1e-3 and abs(r.frame_rate_hz - 40.0) < 1e-2
    assert 3.0 < r.interval_sd_us < 5.5                     # diff of 3 µs jitter: √2 · 3 µs
    assert r.n_long == 5 and abs(r.long_share - 5 / r.n_intervals) < 1e-12
    assert abs(r.long_max_ms - 75.0) < 0.1
    g = gaps.set_index("interval_frames")
    assert g.loc[2, "n"] == 4 and g.loc[2, "missing_frames"] == 1
    assert g.loc[3, "n"] == 1 and abs(g.loc[3, "interval_ms"] - 75.0) < 0.1


def test_frame_timing_matches_the_methods_table_rule():
    """The same numbers as ``val_table_methods`` computes in the view."""
    rng = np.random.default_rng(1)
    t = np.cumsum(np.where(rng.random(3000) < 0.03, 0.05, 0.025) + rng.normal(0, 2e-6, 3000))
    dt = np.diff(t)
    med = np.median(dt)
    long_ = dt > 1.5 * med
    r = DG.frame_timing("s01", t)[0].iloc[0]
    assert r.interval_sd_us == pytest.approx(1e6 * np.std(dt[~long_]))
    assert r.long_share == pytest.approx(long_.mean())


# -- mounting helpers -----------------------------------------------------------------------------
def test_session_label_and_rotation_between_sessions():
    assert DG.session_label("x/session_imu_mounting_20250314T110000Z_0000000d/session.h5") \
        == "2025-03-14"
    assert DG.session_label("some_folder") == "some_folder"
    R0 = Rotation.from_rotvec([0.1, 0.2, 0.3]).as_matrix()
    turn = Rotation.from_rotvec(np.radians(176.0) * np.array([0, 0, 1.0])).as_matrix()
    small = Rotation.from_rotvec(np.radians(1.2) * np.array([1.0, 0, 0])).as_matrix()
    he = {"2025-03-14": {"cra": R0, "mta": R0},
          "2025-03-15": {"cra": R0 @ turn, "mta": R0 @ small}}
    tab = DG.session_rotations("s10", he).set_index("body")
    assert tab.loc["HP", "mounting_change_deg"] == pytest.approx(176.0, abs=1e-6)
    assert tab.loc["MP", "mounting_change_deg"] == pytest.approx(1.2, abs=1e-6)
    assert tab.loc["HP", "session_a"] == "2025-03-14"


def test_excitation_singular_values_see_single_axis_motion():
    t = np.arange(2000) / 40.0
    one_axis = Rotation.from_rotvec(np.column_stack(
        [np.radians(10) * np.sin(2 * np.pi * 0.5 * t), 0 * t, 0 * t])).as_matrix()
    sv = DG.excitation_sv_deg(one_axis, 25)
    assert sv[0] > 1.0 and sv[1] < 1e-6 and sv[2] < 1e-6
    rng = np.random.default_rng(0)
    three = Rotation.from_rotvec(np.radians(10) * np.sin(
        2 * np.pi * np.outer(t, [0.5, 0.37, 0.61]) + rng.random(3))).as_matrix()
    sv3 = DG.excitation_sv_deg(three, 25)
    assert sv3[2] > 0.2 * sv3[0]
    assert np.isnan(DG.excitation_sv_deg(one_axis[:10], 25)).all()


def test_pairwise_spread_and_nearest_quaternion():
    Xs = [Rotation.from_rotvec(np.radians(a) * np.array([0, 1.0, 0])).as_matrix()
          for a in (0.0, 5.0, 14.0)]
    mx, md = DG.pairwise_spread(Xs)
    assert mx == pytest.approx(14.0) and md == pytest.approx(9.0)
    assert np.isnan(DG.pairwise_spread(Xs[:1])).all()
    ts = np.array([0.0, 0.01, 0.02, 0.10])
    q = np.arange(16, dtype=float).reshape(4, 4)
    out = DG.nearest_quat(ts, q, np.array([0.004, 0.016, 0.06]), 0.02)
    assert np.array_equal(out[0], q[0]) and np.array_equal(out[1], q[2])
    assert np.isnan(out[2]).all()                               # 40 ms from the nearest


# -- VD9 acceleration ratio ----------------------------------------------------------------------
F_HZ, AMP_DEG = 3.0, 12.0


def _poses(t: np.ndarray) -> np.ndarray:
    th = AMP_DEG * (1 - np.cos(2 * np.pi * F_HZ * t)) / 2
    M = np.tile(np.eye(4), (len(t), 1, 1))
    M[:, :3, :3] = Rotation.from_rotvec(np.column_stack([0 * th, np.radians(th), 0 * th])
                                        ).as_matrix()
    M[:, :3, 3] = np.column_stack([90 * np.sin(np.radians(th)), 0 * th,
                                   -90 * (1 - np.cos(np.radians(th)))])
    return M


def test_chain_acceleration_is_the_second_derivative_of_the_same_fit():
    t200, t40 = np.arange(0, 6, 1 / 200), np.arange(0, 6, 1 / 40)
    true_acc = AMP_DEG / 2 * (2 * np.pi * F_HZ) ** 2                 # max |θ̈|, deg/s²
    v200, sp200, a200 = S._chain(t200, _poses(t200), *S.PIPELINES["ref200"], t200 < 0.01,
                                 accel=True)
    v_only = S._chain(t200, _poses(t200), *S.PIPELINES["ref200"], t200 < 0.01)
    assert len(v_only) == 2 and np.array_equal(v_only[0], v200, equal_nan=True)
    _, _, a40 = S._chain(t40, _poses(t40), *S.PIPELINES["ref40"], t40 < 0.01, accel=True)
    k200, k40 = (t200 > 1) & (t200 < 5), (t40 > 1) & (t40 < 5)
    assert abs(np.nanmax(np.abs(a200[k200])) / true_acc - 1) < 0.05
    assert np.nanmax(np.abs(a40[k40])) / true_acc < 0.9               # the chain attenuates


def test_acceleration_rows_leave_the_velocity_bootstrap_unchanged():
    t = np.arange(0, 6, 1 / 200)
    rng = np.random.default_rng(3)
    rate = AMP_DEG * np.pi * F_HZ * np.sin(2 * np.pi * F_HZ * t)
    acc = np.gradient(rate, t)
    noise = 1 + 0.05 * rng.standard_normal(len(t))
    sig_v = {"ref200": (t, rate, np.abs(rate)), "ref40": (t, 0.8 * rate * noise,
                                                          0.8 * np.abs(rate))}
    sig_a = {"ref200": (t, rate, np.abs(rate), acc),
             "ref40": (t, 0.8 * rate * noise, 0.8 * np.abs(rate), 0.5 * acc)}
    cyc = pd.DataFrame({"task": ["open_close"] * 4 + ["left_right"] * 2, "side": [""] * 6,
                        "cycle": range(6), "start": [200, 400, 600, 800, 900, 1000],
                        "end": [266, 466, 666, 866, 966, 1066]})
    pv, pa = S.cycle_peaks("s01", sig_v, cyc, t), S.cycle_peaks("s01", sig_a, cyc, t)
    assert pv["peak_abs_acc_dps2"].isna().all()
    assert pa.loc[pa.task == "left_right", "peak_abs_acc_dps2"].isna().all()
    pd.testing.assert_frame_equal(S.bias(pv), S.bias(pa))         # same draws, same rows
    ba = S.bias(pa, quantities=S.ACC_QUANTITIES)
    assert set(ba.quantity) == {"peak_abs_acc_dps2"}
    r = S.peak_ratio_acc(pd.concat([S.bias(pa), ba], ignore_index=True)).iloc[0]
    assert r.pipeline == "ref40" and r.n == 4 and r.ratio_median == pytest.approx(0.5)
    assert S.peak_ratio_acc(pd.DataFrame()).empty


# -- dynamic calibration: stage version == audit script -------------------------------------------
@dataclass
class _Phase:
    name: str
    t_start_us: int
    t_end_us: int


class _FakeSession:
    def __init__(self, phases: list[_Phase]):
        self.phases = phases

    def phase_names(self) -> list[str]:
        return [p.name for p in self.phases]

    def phase(self, name: str) -> _Phase | None:
        return next((p for p in self.phases if p.name == name), None)


def _synthetic_take(seed: int = 0):
    """100 Hz IMU grid with still and moving stretches, phases (one name twice) and a 40 Hz
    error trace with a drift, a step after a still period and a step during motion."""
    rng = np.random.default_rng(seed)
    tg = (1_000_000_000 + np.arange(0, 120, 0.01) * 1e6).astype(np.int64)
    ts = (tg - tg[0]) * 1e-6
    moving = ((ts > 10) & (ts < 30)) | ((ts > 45) & (ts < 70)) | ((ts > 80) & (ts < 110))
    w = np.where(moving[:, None], np.radians(25.0) * np.sin(2 * np.pi * np.outer(
        ts, [0.4, 0.3, 0.5])), 0.0) + np.radians(0.05) * rng.standard_normal((len(ts), 3))
    rv_c = np.cumsum(w * 0.01, axis=0)
    rv_m = rv_c + np.cumsum(np.where(moving[:, None], 0.5 * w, 0.0) * 0.01, axis=0)
    qc = Rotation.from_rotvec(rv_c).as_quat()[:, [3, 0, 1, 2]]
    qm = Rotation.from_rotvec(rv_m).as_quat()[:, [3, 0, 1, 2]]
    us = lambda s: int(tg[0] + s * 1e6)   # noqa: E731
    phases = [_Phase("rest", us(2), us(9)), _Phase("open_close", us(11), us(29)),
              _Phase("chewing", us(46), us(69)), _Phase("free_motion", us(81), us(109)),
              _Phase("free_motion", us(111), us(118))]
    t = tg[0] * 1e-6 + np.arange(0, 120, 0.025)
    rel = t - tg[0] * 1e-6
    e = 0.02 * rel + 0.05 * rng.standard_normal(len(t))
    e[rel > 72] += 2.0                                       # after a still period
    e[rel > 95] -= 1.5                                       # during motion
    e[(rel > 32) & (rel < 44)] = np.nan                      # no reference
    return tg, qc, qm, phases, t, e


def _script():
    if not (ROOT / "scripts" / "audit_recompute" / "imu_dyncal.py").exists():
        pytest.skip("the audit script is not in this checkout (public release)")
    sys.path.insert(0, str(ROOT / "scripts" / "audit_recompute"))
    try:
        import imu_dyncal
    finally:
        sys.path.pop(0)
    return imu_dyncal


@pytest.mark.parametrize("seed", [0, 1])
def test_dyncal_stage_version_matches_the_audit_script(seed):
    A = _script()
    tg, qc, qm, phases, t, e = _synthetic_take(seed)
    sd = _FakeSession(phases)
    per = A.still_periods(tg, qc, qm)
    assert per, "the synthetic take has still periods"
    prow = A.period_rows(sd, tg, qc, qm, 0.0, per)
    per_s = [(r["t0_s"], r["t1_s"]) for r in prow]
    tr = A.trace_summary(t, e, per_s)
    spans = [(n, sd.phase(n).t_start_us * 1e-6, sd.phase(n).t_end_us * 1e-6)
             for n in dict.fromkeys(sd.phase_names())]
    jumps = A.transitions(t, e, spans, per_s, tr.get("drift_all_dpm", np.nan))
    assert tr["n_steps"] >= 1 and jumps

    first: dict[str, _Phase] = {}
    for ph in phases:
        first.setdefault(ph.name, ph)
    ph_list = [(n, float(p.t_start_us), float(p.t_end_us)) for n, p in first.items()]
    summ, periods, trans = DG.dyncal_trace("s01", tg, qc, qm, ph_list, t, e, "trace")
    s = summ.iloc[0]
    ren = {"n_steps": "n_err_steps", "n_steps_aligned": "n_err_steps_aligned"}
    for k, v in tr.items():
        got = s[ren.get(k, k)]
        assert got == pytest.approx(v, nan_ok=True, rel=1e-12, abs=1e-12), k
    assert s.n_still == len(prow)
    assert s.still_s == pytest.approx(sum(r["duration_s"] for r in prow))
    assert s.rel_rate_median_dpm == pytest.approx(np.median([r["rel_rate_dpm"] for r in prow]))
    assert s.max_abs_residual_jump_deg == pytest.approx(
        np.max(np.abs([j["residual_deg"] for j in jumps])))
    assert len(periods) == len(prow)
    for (_, a), b in zip(periods.iterrows(), prow):
        sel = (t >= b["t0_s"]) & (t <= b["t1_s"])
        for k in ("t0_s", "t1_s", "duration_s", "rel_net_deg", "rel_rate_dpm"):
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k
        assert a["phase"] == b["phase"]
        assert a["err_change_deg"] == pytest.approx(A._edge_change(t[sel], e[sel]), nan_ok=True)
    assert len(trans) == len(jumps)
    for (_, a), b in zip(trans.iterrows(), jumps):
        assert (a["from_phase"], a["to_phase"]) == (b["from"], b["to"])
        for k in ("gap_s", "still_in_gap_s", "jump_deg", "expected_deg", "residual_deg",
                  "window_sd_deg"):
            assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), k


def test_dataset_transition_test_groups_by_stillness():
    import pandas as pd

    from biomech.imu.stage_dyncal import transition_test

    tr = pd.DataFrame({"pid": ["a", "a", "b", "b", "b"],
                       "still_in_gap_s": [0.5, 3.0, 2.0, 0.0, 5.0],
                       "residual_deg": [0.1, -0.6, 0.4, -0.2, 0.5]})
    r = transition_test(tr).iloc[0]
    assert (r.n_still, r.n_other, r.n_participants) == (3, 2, 2)
    assert r.jump_still_median_deg == 0.5 and r.jump_other_median_deg == pytest.approx(0.15)
    assert 0 < r.p_mannwhitney <= 1
    assert transition_test(pd.DataFrame()).empty
