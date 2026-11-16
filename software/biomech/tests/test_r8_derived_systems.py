# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Derived tables of the systems paper (``viz/derived_systems.py``, 2026-11-05) on synthetic
contexts, and cited numbers from a sensitivity variant's snapshot (``cite: variant:``)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from biomech.viz import derived
from biomech.viz import numbers as N


class Ph:
    def __init__(self, name, t0, t1):
        self.name, self.t_start_us, self.t_end_us = name, t0, t1


class Ctx:
    def __init__(self, tables=None, arrays=None, sessions=None):
        self._t, self._a, self._s = tables or {}, arrays or {}, sessions or {}

    def pids(self):
        return ["s10", "s14"]

    def table(self, stage, name, pids=None):
        return self._t.get((stage, name), pd.DataFrame())

    def arrays(self, stage, pid):
        return self._a.get((stage, pid), {})

    def phases(self, pid):
        return [Ph("rest", 0, 10_000_000), Ph("open_close", 10_000_000, 26_000_000)]

    def session_path(self, pid):
        return self._s.get(pid)


def test_imu_lag_pools_the_fusion_lag_and_takes_the_accelerometer_median():
    cal = pd.DataFrame([{"pid": p, "fold": f, "body": b, "variant": v, "lag_ms": lag,
                         "lag_se_ms": 1.0, "drift_heading_deg_min": dr}
                        for p, lag in (("s10", 8.0), ("s14", 2.5))
                        for f in ("open_close", "chewing")
                        for b, dr in (("HP", 4.5), ("MP", 2.7))
                        for v in (None, "precal")])
    acc = pd.DataFrame([{"pid": p, "fold": f, "body": b, "primary": True, "variant": v,
                         "lag_ms": 5.0 if v == "sensor" else 99.0}
                        for p in ("s10", "s14") for f in ("open_close", "chewing")
                        for b in ("HP", "MP") for v in ("sensor", "no_lever")])
    t = derived.table(Ctx({("ref_fusion", "calibration"): cal,
                           ("ref_imu_accel", "calibration"): acc}), "imu_lag")
    assert len(t) == 4                                   # the precal rows are left out
    r = t[(t.pid == "s10") & (t.fold == "chewing")].iloc[0]
    assert r.lag_opt_ms == 8.0 and r.lag_ref_ms == 5.0
    assert r.drift_hp_deg_min == 4.5 and r.drift_mp_deg_min == 2.7


def test_lp_agreement_recovers_slope_intercept_and_nrmse():
    rng = np.random.default_rng(0)
    n = 400
    q = np.column_stack([np.linspace(0, 20, n), np.linspace(-5, 5, n), np.linspace(0, -30, n)])
    s = 1.1 * q + 0.5 + rng.normal(0, 0.01, q.shape)
    arr = {"phase": np.array(["open_close"] * n), "evaluated": np.ones(n, bool),
           "incisor_s_aligned_mm": s, "incisor_q_mm": q,
           "rot_s_aligned_deg": q / 2, "rot_q_aligned_deg": q / 2}
    t = derived.table(Ctx(arrays={("ref_kinematics", "s10"): arr}), "lp_agreement")
    x = t[(t.pid == "s10") & (t.axis == "incisor_x")].iloc[0]
    assert x.lp_slope == pytest.approx(1.1, abs=0.01)
    assert x.lp_intercept == pytest.approx(0.5, abs=0.05)
    rot = t[(t.pid == "s10") & (t.axis == "rot_y")].iloc[0]
    assert rot.lp_slope == pytest.approx(1.0) and rot.nrmse_pct == pytest.approx(0.0)
    assert set(t.pid) == {"s10"}                         # s14 has no arrays


def test_recorder_reads_the_session_metadata(tmp_path):
    h5py = pytest.importorskip("h5py")
    d = tmp_path / "session_s10"
    d.mkdir()
    with h5py.File(d / "session.h5", "w") as h:
        m = h.create_group("meta")
        m.attrs["mcu_config_json"] = json.dumps({"trigger": {"rate_hz": 40,
                                                             "pulse": {"width_us": 12000}}})
        m.attrs["optical_rgb_size"] = "728x544"
        cal = h.create_group("optical/corners/calibration")
        for cam, tx in (("cam0", 0.0), ("cam1", -437.0)):
            g = cal.create_group(cam)
            g["K"] = np.array([[900.0, 0, 364], [0, 900.0, 272], [0, 0, 1]])
            T = np.eye(4)
            T[0, 3] = tx
            g["T_world_cam"] = T
            g.attrs["image_size"] = np.array([728, 544])
        P = np.repeat(np.eye(4)[None], 5, axis=0).astype(np.float32)
        P[:, 2, 3] = 700.0
        h["optical/pose_mta"] = P
        h.create_group("sync").attrs["report_json"] = json.dumps(
            {"mcu_stats_start": {"missed_drdy": 1, "drdy_count": 10},
             "mcu_stats_end": {"missed_drdy": 2, "drdy_count": 4010, "poll_max_us": 520,
                               "max_drdy_gap_us": 430}})
        for body in ("mta", "cra"):
            g = h.create_group(f"imu_{body}")
            g["t_master_us_quat"] = np.arange(0, 1_000_000, 2_500)
            g["t_master_us_accel"] = np.arange(0, 1_000_000, 4_000)
    t = derived.table(Ctx(sessions={"s10": d}), "recorder").iloc[0]
    assert t.trigger_rate_hz == 40 and t.trigger_pulse_us == 12000
    assert (t.stream_w_px, t.stream_h_px) == (728, 544)
    assert t.baseline_mm == pytest.approx(437.0) and t.dist_cam0_mm == pytest.approx(700.0)
    assert t.hfov_deg == pytest.approx(np.degrees(2 * np.arctan(364 / 900)))
    assert t.poll_max_us == 520 and t.missed_drdy_take == 1 and t.drdy_take == 4000
    assert t.quat_rate_mta_hz == pytest.approx(400.0) and t.accel_rate_cra_hz == 250.0


def test_phase_durations_and_chewing_opening():
    t = derived.table(Ctx(), "phase_durations")
    assert t[(t.pid == "s10") & (t.phase == "open_close")].duration_s.iloc[0] == 16.0
    cyc = pd.DataFrame({"pid": ["a", "a", "a", "b", "b"],
                        "task": ["chewing", "chewing", "open_close", "chewing", "chewing"],
                        "ok": [True, True, True, True, False],
                        "opening_mm": [4.0, 6.0, 40.0, 10.0, 1.0]})
    c = derived.table(Ctx({("kinematics", "cycles"): cyc}), "chewing_opening")
    assert dict(zip(c.pid, c.chew_opening_mm)) == {"a": 5.0, "b": 10.0}


def test_cite_group_may_name_a_variant(tmp_path):
    f = tmp_path / "numbers_x.yaml"
    (tmp_path / "study.yaml").write_text("{}")
    f.write_text("cite:\n  ds:\n    study: study.yaml\n    snapshot: r15a\n"
                 "    variant: reporting_population\n    numbers:\n"
                 "      chewMin: {source: 'derived:chewing_opening', column: chew_opening_mm,"
                 " agg: min, fmt: '.1f'}\n")
    decl = N.load(f)
    assert decl["chewMin"]["_cite"]["variant"] == "reporting_population"
    seen = {}
    cyc = pd.DataFrame({"pid": ["a", "b"], "task": ["chewing"] * 2, "ok": [True] * 2,
                        "opening_mm": [4.4, 9.0]})

    def opener(study, snap, variant=None):
        seen["args"] = (snap, variant)
        return Ctx({("kinematics", "cycles"): cyc})

    got = {n.key: n for n in N.resolve(Ctx(), decl, opener)}
    assert seen["args"] == ("r15a", "reporting_population") and got["chewMin"].text == "4.4"


def test_imu_mounting_follows_the_vd14_rule(tmp_path):
    """``imu_mounting``: a participant's own mounting session overrides the study-level one;
    tumble durations from the phases, fit RMS and pairs from /calibration."""
    import h5py

    def session(name, d_mta, d_cra, rms):
        f = tmp_path / "data" / name / "session.h5"
        f.parent.mkdir(parents=True)
        with h5py.File(f, "w") as h:
            ph = h.create_group("phases")
            ph["name"] = np.array([b"imu_mount_mta", b"imu_mount_cra"])
            ph["t_start_us"] = np.array([0, 30_000_000])
            ph["t_end_us"] = np.array([d_mta * 1e6, 30_000_000 + d_cra * 1e6]).astype(np.int64)
            for body in ("mta", "cra"):
                g = h.create_group(f"calibration/imu_{body}")
                g.attrs["rms_deg"], g.attrs["n_selected"], g.attrs["n_pairs"] = rms, 40, 700
        return f"{name}/session.h5"

    early, late = session("m_early", 18.2, 13.8, 2.75), session("m_late", 23.7, 17.5, 1.75)
    ref = tmp_path / "reference.yaml"
    ref.write_text(f"imu_mounting: {late}\nparticipants:\n  s10: {{imu_mounting: {early}}}\n"
                   "  s14: {}\n")
    cfg = type("C", (), {"data_dir": tmp_path / "data", "source": tmp_path / "study.yaml",
                         "params": staticmethod(lambda s: {"reference": "reference.yaml"})})()
    ctx = Ctx()
    ctx.config = cfg
    t = derived.table(ctx, "imu_mounting").set_index("pid")
    assert t.loc["s10", "session"] == "m_early" and t.loc["s14", "session"] == "m_late"
    assert t.loc["s10", "mount_mta_s"] == pytest.approx(18.2)
    assert t.loc["s14", "mount_cra_s"] == pytest.approx(17.5)
    assert t.loc["s10", "mount_mta_rms_deg"] == 2.75 and t.loc["s14", "mount_cra_n_selected"] == 40
