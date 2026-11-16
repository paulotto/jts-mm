# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG comparability (biomech.validation.emg, plan D5) on synthetic signals only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.emg.prepare import EmgParams, EmgPrep
from biomech.io.reference import read_delsys_csv
from biomech.io.session import CLOSERS, MUSCLES, OPENERS
from biomech.validation import emg as E
from biomech.validation.synthetic import write_trigno_csv

FS = 2000.0
NAMES = list(MUSCLES)


# -- synthetic signals -------------------------------------------------------------------------
def _cycles(t0: float, n: int, period: float = 2.0, rise: float = 0.7, fall: float = 0.7):
    """(start, peak, end) seconds of n open-close cycles."""
    return [(t0 + k * period, t0 + k * period + rise, t0 + k * period + rise + fall)
            for k in range(n)]


def _emg(t: np.ndarray, cycles, rng, lead_s: float = 0.08, amp_op: float = 60.0,
         amp_cl: float = 40.0, noise: float = 3.0) -> np.ndarray:
    """Openers burst from ``lead_s`` before each cycle start to its peak, closers from the
    peak (minus the lead) to the end; white-noise carrier, a small 50-Hz hum."""
    x = noise * rng.standard_normal((len(t), len(NAMES)))
    for a, pk, b in cycles:
        op = (t >= a - lead_s) & (t < pk - lead_s)
        cl = (t >= pk - lead_s) & (t < b - lead_s)
        for i, m in enumerate(NAMES):
            if m in OPENERS:
                x[op, i] += amp_op * rng.standard_normal(op.sum())
            else:
                x[cl, i] += amp_cl * rng.standard_normal(cl.sum())
    return x + 2.0 * np.sin(2 * np.pi * 50.0 * t)[:, None]


def _theta(t: np.ndarray, cycles, amp: float = 20.0) -> np.ndarray:
    th = np.zeros_like(t)
    for a, pk, b in cycles:
        s = (t >= a) & (t <= b)
        th[s] = amp * np.sin(np.pi * (t[s] - a) / (b - a)) ** 2
    return th


def _motion_take(seed: int, scale: float = 1.0, lead_s: float = 0.08):
    """A take: rest 0-6 s, open_close 6-28 s (10 cycles), chewing 28-40 s (12 fast cycles);
    plus its cycle table and opening-angle arrays (40 Hz)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 40.0, 1 / FS)
    oc = _cycles(7.0, 10)
    ch = _cycles(29.0, 12, period=0.9, rise=0.35, fall=0.4)
    x = scale * _emg(t, oc + ch, rng, lead_s=lead_s)
    ph = [E.Phase("rest", "experiment", 0, 6_000_000),
          E.Phase("open_close", "experiment", 6_000_000, 28_000_000),
          E.Phase("chewing", "experiment", 28_000_000, 40_000_000 - 1000)]
    take = E.EmgTake("synthetic", (t * 1e6).astype(np.int64), x, NAMES, FS, ph)
    rows = []
    for task, cyc in (("open_close", oc), ("chewing", ch)):
        for k, (a, pk, b) in enumerate(cyc):
            rows.append({"task": task, "side": "", "cycle": k, "ok": True,
                         "t_start_us": int(a * 1e6), "t_peak_us": int(pk * 1e6),
                         "t_end_us": int(b * 1e6)})
    tk = np.arange(0, 40.0, 1 / 40.0)
    th = _theta(tk, oc + ch)
    kin = {}
    for p in ph:
        s = (tk * 1e6 >= p.t_start_us) & (tk * 1e6 <= p.t_end_us)
        kin[f"{p.name}__t_us"] = (tk[s] * 1e6).astype(np.int64)
        kin[f"{p.name}__theta_deg"] = th[s]
    return take, pd.DataFrame(rows), kin


def _mvc_take(seed: int, opening_first: bool = True, scale_cl: float = 1.0) -> E.EmgTake:
    """A Delsys-style MVC file: one effort per group (opening 1-11 s, closing 15-25 s)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, 28.0, 1 / FS)
    x = 3.0 * rng.standard_normal((len(t), len(NAMES)))
    first, second = (1 <= t) & (t < 11), (15 <= t) & (t < 25)
    op, cl = (first, second) if opening_first else (second, first)
    for i, m in enumerate(NAMES):
        sel = op if m in OPENERS else cl
        amp = 150.0 if m in OPENERS else 120.0 * scale_cl
        x[sel, i] += amp * rng.standard_normal(sel.sum())
    return E.EmgTake("mvc", (t * 1e6).astype(np.int64), x, NAMES, FS)


# -- the SessionData stand-in ------------------------------------------------------------------
def test_emg_take_runs_dataset_emgprep():
    take, _, _ = _motion_take(1)
    mvc = _mvc_take(2)
    mvc.phases, info = E.segment_mvc(mvc)
    prep = EmgPrep(take, mvc, EmgParams())
    ref = prep.reference
    assert (ref["source"] == "mvc").all()                    # MVC 150/120 > task 60/40
    pe = prep.phase("open_close")
    assert pe.env.shape == (len(pe.t_us), len(NAMES))
    assert np.nanmax(pe.norm) <= 100.0 + 1e-9                 # D11: <= 100 %
    t, uv, names = take.emg_window(6_000_000, 6_000_999)
    assert len(t) == 2 and names == NAMES                     # [t0, t1] inclusive, like h5
    assert not take.leadoff_window(0, 10_000_000).any()
    assert np.isinf(take.emg_full_scale_uv()).all()


def test_segment_mvc_order_and_split():
    for first in (True, False):
        mvc = _mvc_take(3, opening_first=first)
        ph, info = E.segment_mvc(mvc)
        assert info["order"] == ("opening_first" if first else "closing_first")
        assert 10.5 < info["split_s"] < 15.5                   # in the pause between efforts
        names = [p.name for p in ph]
        assert names == (["mvc_opening", "mvc_closing"] if first
                         else ["mvc_closing", "mvc_opening"])
        assert ph[0].t_end_us < ph[1].t_start_us


def test_multi_mvc_takes_largest_and_matches_single():
    take, _, _ = _motion_take(4)
    a, b = _mvc_take(5), _mvc_take(6, scale_cl=2.0)        # file b: closers twice as strong
    for m in (a, b):
        m.phases, _ = E.segment_mvc(m)
    single = EmgPrep(take, a, EmgParams())
    one = E.MultiMvcPrep(take, [a], EmgParams())
    for k in ("max_uv", "p95_uv", "max1s_uv"):
        np.testing.assert_array_equal(one.refs[k], single.refs[k])
    two = E.MultiMvcPrep(take, [a, b], EmgParams())
    cl = [i for i, n in enumerate(NAMES) if n in CLOSERS]
    assert (two.refs["file"][cl] == 1).all()
    assert np.all(two.refs["max_uv"][cl] > 1.5 * one.refs["max_uv"][cl])
    pe = two.mvc_phase(1, "mvc_closing")
    assert pe.env.shape[1] == len(NAMES)
    assert two.mvc is a                                       # restored after the call


def test_delsys_take_resamples_and_maps(tmp_path: Path):
    """4370-Hz and 2148-Hz channels -> one 4000-Hz grid; amplitude kept at 100 Hz (cubic
    interpolation), the time map applied to the sample times."""
    chans = []
    for lab, fs in (("Dig_Left", 4370.3704), ("Dig_Right", 2148.1481), ("Mas_Left", 4370.3704)):
        tt = np.arange(int(20 * fs)) / fs
        chans.append((lab, "Avanti", 84, fs, 0.05 * np.sin(2 * np.pi * 100.0 * tt)))  # 50 µV
    path = tmp_path / "d.csv"
    write_trigno_csv(path, chans)
    d = read_delsys_csv(path)
    take = E.delsys_take(d, NAMES, 4000.0, time_map=lambda td: td * (1 + 10e-6) + 1000.0)
    present = {"digastric_l", "digastric_r", "masseter_l"}
    assert take.names == [m for m in MUSCLES if m in present]        # MUSCLES order, present
    assert take.fs == 4000.0
    t_s = take.t_us * 1e-6
    assert abs(t_s[0] - 1000.0) < 1e-3
    assert np.allclose(np.diff(t_s), (1 + 10e-6) / 4000.0, atol=2e-6)
    mid = slice(4000, len(t_s) - 4000)
    amp = np.sqrt(2) * take.uv[mid].std(axis=0)
    np.testing.assert_allclose(amp, 50.0, rtol=0.01)


def test_select_and_clip_phases():
    ph = [E.Phase("mand_point_1", "calibration_mandibular", 0, 10_000_000),
          E.Phase("rest", "experiment", 10_000_000, 20_000_000),
          E.Phase("free_motion", "experiment", 20_000_000, 60_000_000)]
    t = np.arange(0, 60_000_000, 25_000)
    ok = np.zeros(len(t), bool)
    ok[(t > 2_000_000) & (t < 8_000_000)] = True              # sync in the hold: ignored
    ok[(t >= 45_000_000)] = True                              # end sync in free_motion
    out = E.select_phases(ph, t, ok, guard_s=0.5)
    fm = next(p for p in out if p.name == "free_motion")
    assert fm.t_end_us == 44_500_000
    assert next(p for p in out if p.name == "mand_point_1").t_end_us == 10_000_000
    cl = E.clip_phases(out, 15_000_000, 50_000_000)
    assert [p.name for p in cl] == ["rest", "free_motion"]
    assert cl[0].t_start_us == 15_000_000


# -- Qualisys jaw signals and the δ test --------------------------------------------------------
class _Qtm:
    def __init__(self, t, R_mp):
        self.t_s, self.fs_hz = t, 1 / np.median(np.diff(t))
        n = len(t)
        self._b = {"HP": np.tile(np.eye(4), (n, 1, 1)), "MP": np.tile(np.eye(4), (n, 1, 1))}
        self._b["MP"][:, :3, :3] = R_mp

    def pose(self, b):
        return self._b[b]


def test_qualisys_jaw_speed_and_signed_angle():
    t = np.arange(0, 20, 1 / 200.0)
    ang = np.radians(15.0 * np.sin(2 * np.pi * 0.5 * t) ** 2)     # 0 .. 15 deg, about +y
    R = Rotation.from_rotvec(np.c_[0 * t, -ang, 0 * t]).as_matrix()  # sign: either way
    jaw = E.qualisys_jaw(_Qtm(t, R), ref_window_s=(0.0, 0.02), axis_window_s=(0, 20))
    mid = slice(10, -10)
    np.testing.assert_allclose(jaw["theta_deg"][mid], np.degrees(ang)[mid], atol=0.05)
    rate = np.abs(np.gradient(np.degrees(ang), t))
    np.testing.assert_allclose(jaw["speed_dps"][mid], rate[mid], atol=0.6)


@pytest.mark.parametrize("delta_true", [0.0, 0.74])
def test_delta_check_recovers_latency(delta_true: float):
    """The opener envelope follows the opening angle 80 ms ahead (the digastric holds the
    jaw open), the closers the closing speed; the Delsys clock runs δ ahead of QTM."""
    tq = np.arange(0, 60, 1 / 200.0)
    cyc = _cycles(1.0, 29, period=2.0, rise=0.7, fall=0.7)
    th = _theta(tq, cyc)
    jaw = {"t_s": tq, "angle_deg": th, "speed_dps": np.abs(np.gradient(th, tq)),
           "theta_deg": th, "ok": np.ones(len(tq), bool)}
    td = np.arange(0, 61, 1 / 500.0)                       # Delsys clock: t_D = t_Q + δ
    tq_of_d = td - delta_true
    th_ahead = _theta(tq_of_d + 0.08, cyc)
    op = 2.5 * th_ahead
    cl = 0.5 * np.maximum(-np.gradient(th_ahead, td), 0.0)
    envs = {m: (td, (op if m in OPENERS else cl) + 3.0) for m in NAMES}
    tab, arr, info = E.delta_check(envs, jaw, {"open_close": (5.0, 50.0)}, [0.0, 0.74])
    assert info["delta_s"] == delta_true and info["status"] == "decided"
    row = tab[(tab.group == "openers") & (tab.signal == "angle_deg")].iloc[0]
    assert row[f"lead_ms_d{delta_true:g}"] == pytest.approx(80.0, abs=60.0)
    assert len(arr["delta_lags_s"]) == 301


# -- profiles, SPM, the full comparison ------------------------------------------------------
def _profiles(rng, n, shift=0.0, noise=1.0):
    x = np.linspace(0, 1, 101)
    base = 10 + 30 * np.exp(-((x - 0.3 - shift) / 0.1) ** 2)
    Y = base[None, :, None] + noise * rng.standard_normal((n, 101, 2))
    return Y, np.arange(n)


def test_compare_profiles_same_vs_shifted():
    rng = np.random.default_rng(7)
    a = {"open_close": _profiles(rng, 12)}
    b = {"open_close": _profiles(rng, 12)}
    tab, clus, arr = E.compare_profiles(a, b, ["m1", "m2"], "sX")
    assert (tab["r"] > 0.99).all() and (tab["rms_diff_pct"] < 1.0).all()
    assert (tab["cmc"] > 0.99).all()
    c = {"open_close": _profiles(rng, 12, shift=0.15)}
    tab2, clus2, _ = E.compare_profiles(a, c, ["m1", "m2"], "sX")
    assert tab2["spm_h0reject"].all() and len(clus2) >= 2
    assert {"start_pct", "end_pct", "p"} <= set(clus2.columns)
    assert arr == {} and tab2["spm_zstar"].notna().all()


def test_lead_comparison_bootstrap():
    rows = []
    for sysm, src, mu in (("jtsmm", "jtsmm_optical", 50.0), ("delsys", "jtsmm_optical", 80.0)):
        for k in range(20):
            rows.append({"system": sysm, "mech_source": src, "task": "open_close",
                         "direction": "opening", "detector": "tkeo", "rule": "drive",
                         "usable": True, "lead_ms": mu + (k - 10)})
    out = E.lead_comparison(pd.DataFrame(rows), "sX", seed=1, n_boot=500)
    r = out.iloc[0]
    assert r["diff_ms"] == pytest.approx(30.0)
    assert r["diff_ci_lo_ms"] < 30.0 < r["diff_ci_hi_ms"]


def test_compare_end_to_end_synthetic():
    """Both sides from the same generator (different seeds and gain): shapes agree, the
    normalization removes the gain, timing finds the 80-ms opener lead in both."""
    ta, ca, ka = _motion_take(11)
    tb, cb, kb = _motion_take(12, scale=0.5)
    ma, mb = _mvc_take(13), _mvc_take(14)
    mb.uv = 0.5 * mb.uv
    for m in (ma, mb):
        m.phases, _ = E.segment_mvc(m)
    sides = E.EmgSides(E.MultiMvcPrep(ta, [ma], EmgParams()),
                       E.MultiMvcPrep(tb, [mb], EmgParams()), ca, cb, ka, kb,
                       mvc_files=["b.csv"])
    from biomech.config import load_config

    cfg = load_config(None, overrides={"paths.cohort": "x.yaml"})
    dp = {"emg": cfg.signals["emg"], "timing": cfg.params("timing"), "cycle_mask_max": 0.1,
          "usability": cfg.params("qc")["usability"]}
    tables, arrays = E.compare(sides, "sX", dp, dict(E.DEFAULTS), seed=3)
    prof = tables["profiles"]
    oc = prof[prof.profile == "open_close"]
    assert len(oc) == len(NAMES) and (oc["r"] > 0.9).all()
    mvc = tables["mvc"]
    ratio = (mvc[mvc.system == "delsys"].set_index("muscle")["mvc_max_uv"]
             / mvc[mvc.system == "jtsmm"].set_index("muscle")["mvc_max_uv"])
    assert np.allclose(ratio, 0.5, rtol=0.1)                   # µV levels differ ...
    act = tables["activation"]
    a = act[(act.task == "open_close")].pivot(index="muscle", columns="system",
                                               values="median_pct")
    assert np.allclose(a["delsys"], a["jtsmm"], rtol=0.25)     # ... normalized levels agree
    q = tables["quality"]
    assert set(q.system) == {"jtsmm", "delsys"} and "snr_db" in q
    on = tables["timing_onsets"]
    u = on[on.usable & (on.detector == "tkeo") & (on.rule == "drive")
           & (on.task == "open_close") & (on.direction == "opening")]
    med = u.groupby("system")["lead_ms"].median()
    assert set(med.index) == {"jtsmm", "delsys"}
    assert ((med > 20) & (med < 200)).all()
    assert not tables["timing_compare"].empty
    assert "profile__delsys__open_close__pct" in arrays


def test_ref_emg_stage_registered():
    from biomech.pipeline import get_stage

    st = get_stage("ref_emg")
    assert st.group == "validation" and st.scope == "participant"
    assert st.depends == ("ref_sync",)
