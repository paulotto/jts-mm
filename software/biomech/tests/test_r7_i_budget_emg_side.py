# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""systems paper: VD22 error budget (``validation.budget``, stage
``ref_budget``) and VD23 EMG side check (``validation.emg_side``, stage ``ref_emg_side``) on
synthetic tables and profiles; the new in-text macros of ``numbers_systems.yaml`` resolve
against a store holding those tables; the new views build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomech.io.session import MUSCLES
from biomech.validation import budget as BU
from biomech.validation import emg_side as ES
from biomech.validation.emg import profile_array_key, profile_r

NUMBERS = Path(__file__).resolve().parents[3] / "experiments" / "jtsmm_validation" / \
    "numbers_systems.yaml"
PIDS = ("s10", "s14")


# -- VD23: side index -----------------------------------------------------------------------
def _stack(shapes: dict[str, np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """(cycles, points, channels) % profiles: each channel's shape with a per-cycle gain and a
    level offset (neither changes the shape correlation of the means)."""
    Y = np.full((n, 101, len(MUSCLES)), 5.0)
    for name, s in shapes.items():
        Y[:, :, MUSCLES.index(name)] = 20 + rng.uniform(0.5, 1.5, (n, 1)) * 10 * s[None]
    return Y


def _arrays(swap: bool, n: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    """left_right__left with distinct R/L shapes per pair (R = −L); open_close with equal
    shapes on both sides. ``swap``: the Delsys sides exchanged."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 101)
    a = np.sin(2 * np.pi * t)
    lr = {f"{p}_{s}": (a if s == "r" else -a) for p in ES.DEFAULTS["pairs"] for s in "rl"}
    oc = {f"{p}_{s}": np.sin(np.pi * t) for p in ES.DEFAULTS["pairs"] for s in "rl"}
    dl = {k.replace("_r", "_X").replace("_l", "_r").replace("_X", "_l"): v
          for k, v in lr.items()} if swap else lr
    return {"names": np.array(MUSCLES),
            profile_array_key("jtsmm", "left_right__left"): _stack(lr, n, rng),
            profile_array_key("delsys", "left_right__left"): _stack(dl, n + 1, rng),
            profile_array_key("jtsmm", "open_close"): _stack(oc, n, rng),
            profile_array_key("delsys", "open_close"): _stack(oc, n, rng)}


@pytest.mark.parametrize("swap,expected", [(False, 2.0), (True, -2.0)])
def test_side_index_correct_and_swapped(swap, expected):
    t = ES.side_check(_arrays(swap), "s10")
    lr = t[t["task"] == "left_right"]
    assert set(lr["muscle"]) == set(ES.DEFAULTS["pairs"]) and lr["side_informative"].all()
    assert np.allclose(lr["side_index"], expected, atol=1e-9)
    assert (lr["side"] == "left").all() and (lr["reason"] == "").all()
    oc = t[t["task"] == "open_close"]                       # same shape both sides: SI = 0
    assert np.allclose(oc["side_index"], 0.0, atol=1e-9) and not oc["side_informative"].any()


def test_side_index_formula_and_shape_r():
    assert ES.side_index(1, 1, -1, -1) == 2.0 and ES.side_index(-1, -1, 1, 1) == -2.0
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=101), rng.normal(size=101)
    za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    assert profile_r(a, b) == pytest.approx(np.mean(za * zb))           # z-scored product
    assert profile_r(3 * a + 7, b) == pytest.approx(profile_r(a, b))    # level and gain free


def test_side_check_edge_cases():
    arr = _arrays(False, n=1)                                           # one cycle per system
    t = ES.side_check(arr, "s10")
    assert t["side_index"].isna().all() and (t["reason"] == "too few cycles").all()
    arr = _arrays(False)
    arr["names"] = np.array([m for m in MUSCLES if m != "digastric_l"] + ["x"])
    t = ES.side_check(arr, "s10")
    assert (t[t["muscle"] == "digastric"]["reason"] == "no channel").all()
    empty = ES.side_check({}, "s10")
    assert empty.empty and list(empty.columns) == ES.COLUMNS
    one_sys = {k: v for k, v in _arrays(False).items() if "delsys" not in k}
    assert ES.side_check(one_sys, "s10").empty


# -- VD22: error budget ----------------------------------------------------------------------
def _budget_inputs() -> tuple[pd.DataFrame, ...]:
    kin = pd.DataFrame([
        {"pid": p, "segment": seg, "kind": kind, "outcome": oc, "sigma_obs": so,
         "sigma_ref": sr, "sigma_jts_corrected": np.sqrt(max(so ** 2 - sr ** 2, 0)),
         "ratio": so / sr, "adequate": so / sr >= 3}
        for p, scale in zip(PIDS, (1.0, 2.0))
        for seg, kind in (("rest", "static"), ("open_close", "movement"), ("chewing", "movement"))
        for oc, so, sr in (("incisor", 2.0, 0.3 * scale), ("rotation", 1.0, 0.2 * scale),
                           ("condyle", 2.5, 0.5 * scale))])
    marking = pd.DataFrame({"pid": PIDS, "rms_teeth_mm": [8.0, 7.0], "rms_condyle_mm": [11, 15],
                            "sigma_ref_teeth_mm": [0.7, 2.2], "sigma_ref_condyle_mm": [0.6, 0.9],
                            "frame_max_deg": [3.5, 7.5], "frame_mand_deg": [6.5, 10.0]})
    lever = 200.0
    holds = pd.DataFrame([
        {"pid": p, "landmark": lm, "e_mm": 5.0 + i, "ct_rot_disagreement_deg": ang,
         "predicted_tip_mm": np.radians(ang) * lever}
        for p, ang in zip(PIDS, (3.0, 1.5))
        for i, lm in enumerate(("max_point_1", "max_point_2", "max_point_3", "mand_point_1",
                                "tmj_left", "tmj_right"))])
    cons = pd.DataFrame([
        {"pid": "s14", "body": "CT", "segment": lm, "kind": "hold", "frac_label_swapped": f}
        for lm, f in (("max_point_1", 1.0), ("max_point_2", 0.9), ("max_point_3", 0.1),
                      ("tmj_left", 1.0))] + [
        {"pid": "s10", "body": "CT", "segment": "max_point_1", "kind": "hold",
         "frac_label_swapped": 0.3},
        {"pid": "s10", "body": "HP", "segment": "max_point_1", "kind": "hold",
         "frac_label_swapped": 1.0}])                     # not the CT: never counted
    geo = pd.DataFrame({"pid": [*PIDS, *PIDS], "body": ["HP", "HP", "MP", "MP"],
                        "sv1_mm": [55.0, 68.0, 67.0, 67.0], "sv2_mm": [30.0, 5.0, 17.0, 17.0],
                        "min_height_mm": [37.0, 6.4, 22.0, 21.0]})
    return kin, marking, holds, cons, geo


def test_budget_tables():
    tabs = BU.run_ref_budget(*_budget_inputs())
    b = tabs["budget"]
    assert list(b.columns) == BU.BUDGET_COLUMNS
    m = b[b["part"] == "marking"]
    assert len(m) == 4 and set(m["outcome"]) == {"teeth", "condyle"}
    ok = b["sigma_obs"] >= b["sigma_ref"]                 # quadrature: obs² = ref² + JTS²
    assert np.allclose(b.loc[ok, "sigma_obs"] ** 2, b.loc[ok, "sigma_ref"] ** 2
                       + b.loc[ok, "sigma_jts_corrected"] ** 2)
    assert np.allclose(b["ref_var_share"], b["sigma_ref"] ** 2 / b["sigma_obs"] ** 2)
    assert b["separable"].all() and (b["reason"] == "").all()
    assert set(b.loc[b["outcome"] == "rotation", "unit"]) == {"deg"}
    h = tabs["marking_holds"]
    assert h.loc[(h.pid == "s14") & (h.landmark == "max_point_1"), "label_swapped"].item()
    assert not h.loc[(h.pid == "s10") & (h.landmark == "max_point_1"), "label_swapped"].item()
    c = tabs["marking_causes"].set_index("pid")
    assert c.loc["s14", "n_swapped_maxillary"] == 2 and c.loc["s14", "n_swapped_condyle"] == 1
    assert c.loc["s10", "n_swapped_maxillary"] == 0 and c.loc["s10", "n_holds_maxillary"] == 3
    assert c.loc["s10", "ct_lever_mm"] == pytest.approx(200.0)
    assert c.loc["s14", "head_sv_ratio"] == pytest.approx(5 / 68)
    g = tabs["marking_gap"].set_index("quantity")
    assert list(tabs["marking_gap"].columns) == BU.GAP_COLUMNS
    assert (g["pid_larger"] == "s14").all() and (g["pid_smaller"] == "s10").all()
    assert g.loc["frame_max_deg", "larger_minus_smaller"] == pytest.approx(4.0)
    assert g.loc["frame_max_deg", "larger_over_smaller"] == pytest.approx(7.5 / 3.5)
    assert np.isnan(g.loc["n_swapped_maxillary", "larger_over_smaller"])  # smaller value 0


def test_budget_not_separable():
    """σ_ref ≥ σ_obs (v9, P2 hold segments): no JTS-MM part (NaN, not 0), share capped at 1,
    flagged with the reason; σ_ref = σ_obs is not separable either; missing σ says so."""
    kin, marking, holds, cons, geo = _budget_inputs()
    extra = pd.DataFrame([
        {"pid": "s14", "segment": seg, "kind": "hold", "outcome": "incisor", "sigma_obs": so,
         "sigma_ref": sr, "sigma_jts_corrected": np.sqrt(max(so ** 2 - sr ** 2, 0)),
         "ratio": so / sr, "adequate": False}
        for seg, so, sr in (("max_point_1", 0.4, 1.0), ("sync_start", 0.5, 0.5),
                            ("mand_point_1", np.nan, 0.5))])
    b = BU.run_ref_budget(pd.concat([kin, extra], ignore_index=True), marking, holds, cons,
                          geo)["budget"].set_index("segment")
    assert b["ref_var_share"].max() <= 1.0
    for seg in ("max_point_1", "sync_start"):
        r = b.loc[seg]
        assert not r["separable"] and np.isnan(r["sigma_jts_corrected"])
        assert r["ref_var_share"] == 1.0 and r["reason"] == BU.NOT_SEPARABLE
    r = b.loc["mand_point_1"]
    assert not r["separable"] and r["reason"] == BU.MISSING and np.isnan(r["ref_var_share"])
    # views: task medians skip the empty JTS-MM part; the line counts what was resolvable
    from biomech.viz.views.validation_part_i import _resolvable, budget_lines

    mov = b.reset_index()
    mov.loc[(mov.pid == "s14") & (mov.segment == "chewing") & (mov.outcome == "incisor"),
            ["sigma_ref", "sigma_jts_corrected", "separable"]] = [3.0, np.nan, False]
    li = budget_lines(mov)
    r = li[(li.pid == "s14") & (li.line == "movement") & (li.outcome == "incisor")].iloc[0]
    assert r["n_rows"] == 2 and r["n_separable"] == 1 and _resolvable(r) == "1 of 2"
    ok = kin[(kin.pid == "s14") & (kin.segment == "open_close") & (kin.outcome == "incisor")]
    assert r["sigma_jts_corrected"] == pytest.approx(ok["sigma_jts_corrected"].item())


def test_budget_degenerate_inputs():
    kin, marking, *_ = _budget_inputs()
    empty = pd.DataFrame()
    tabs = BU.run_ref_budget(kin, marking.iloc[:1], empty, empty, empty)
    assert tabs["marking_holds"].empty and tabs["marking_gap"].empty     # one participant
    c = tabs["marking_causes"].iloc[0]
    assert np.isnan(c["ct_lever_mm"]) and np.isnan(c["head_sv_ratio"])
    assert BU.error_budget(empty, empty).empty


# -- the in-text macros resolve against a store holding the new tables -------------------------
class Ctx:
    """Tables by "stage:name"; pseudonyms s10 -> 1, s14 -> 2."""

    def __init__(self, tables):
        self.tables = tables

    def table(self, stage, name, pids=None):
        d = self.tables.get(f"{stage}:{name}", pd.DataFrame()).copy()
        return d[d["pid"].isin(pids)] if pids and "pid" in d else d

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1

    def pseudonyms(self):
        return {p: i + 1 for i, p in enumerate(PIDS)}


def _store() -> Ctx:
    t = {f"ref_budget:{k}": v for k, v in BU.run_ref_budget(*_budget_inputs()).items()}
    t["ref_emg_side:side_check"] = pd.concat(
        [ES.side_check(_arrays(False, seed=i), p) for i, p in enumerate(PIDS)],
        ignore_index=True)
    # the side check's left_right__right profile: the same synthetic data mirrored
    sc = t["ref_emg_side:side_check"]
    right = sc[sc["task"] == "left_right"].assign(profile="left_right__right", side="right")
    t["ref_emg_side:side_check"] = pd.concat([sc, right], ignore_index=True)
    return Ctx(t)


def test_vd22_vd23_macros_resolve():
    from biomech.viz import numbers as N

    decl = N.load(NUMBERS)
    new = {k: d for k, d in decl.items()
           if str(d.get("source", "")).split(":")[0] in ("ref_budget", "ref_emg_side")}
    new.update({k: d for k, d in decl.items()
                if d.get("mean_of") and all(o in new for o in d["mean_of"])})
    assert len(new) > 40
    got = {n.key: n for n in N.resolve(_store(), new)}
    assert set(got) == set(new)
    assert got["sideIndexTemporalisLeftPOne"].value == pytest.approx(2.0)
    assert got["gapFrameMax"].value == pytest.approx(4.0)
    assert got["causeSwappedMaxPTwo"].text == "2"
    assert got["budgetShareIncisorTasksMean"].n == 2


def test_no_pooled_validation_macros_left():
    """EV8: the pooled P1+P2 medians are gone (the jaw-IMU lever arm stays one value)."""
    from biomech.viz import numbers as N

    decl = N.load(NUMBERS)
    # cited numbers aggregate the other study's cohort (2026-11-05: the data-set's chewing)
    pooled = [k for k, d in decl.items() if d.get("agg") and "pid" not in (d.get("filter") or {})
              and "_cite" not in d
              and str(d.get("source", "")).split(":")[0] not in ("ref_imu_lever", "ref_trueness",
                                                                  "ref_budget", "ref_emg_side",
                                                                  "hw")]   # print geometry, no participants
    assert pooled == []
    assert not [k for k in decl if k.startswith("openImuOptical")]


# -- the views build ----------------------------------------------------------------------------
VIEWS = ("val_budget", "val_table_budget", "val_table_marking_gap", "val_emg_side",
         "val_table_emg_side", "val_hinge", "val_table_hinge")


def test_new_views_build_and_empty():
    from biomech.viz.spec import FigureSpec, TableSpec
    from biomech.viz.views import registry as R

    hinge = pd.DataFrame([{"pid": p, "task": t, "system": s, "n_openings": 5,
                           "hinge_fit_mm": 1.0 + (s == "jtsmm"), "hinge_tmj_mm": 4.0,
                           "axis_to_condyles_mm": 30.0}
                          for p in PIDS for t in ("open_close", "chewing")
                          for s in ("jtsmm", "qualisys")])
    ctx = _store()
    ctx.tables["ref_hinge:hinge"] = hinge
    for vid in VIEWS:
        spec = R.build(vid, R.ViewRequest(ctx))
        assert isinstance(spec, (FigureSpec, TableSpec)) and spec.caption, vid
        if isinstance(spec, TableSpec):
            assert spec.rows, vid
            assert not any("s10" in str(v) or "s14" in str(v)
                           for r in spec.rows for v in r.values()), vid   # pseudonyms only
        else:
            assert spec.panels and spec.meta.get("caption"), vid
    for vid in VIEWS:                                                  # nothing stored
        R.build(vid, R.ViewRequest(Ctx({})))


def test_stages_registered():
    from biomech.pipeline import get_stage

    b, s = get_stage("ref_budget"), get_stage("ref_emg_side")
    assert b.scope == "cohort" and b.group == s.group == "validation"
    assert set(b.depends) == {"ref_kinematics", "ref_landmarks", "ref_quality"}
    assert s.depends == ("ref_emg",) and s.scope == "participant"
