# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trueness/precision on the ladder (biomech.validation.trueness, plan D6): the core on
synthetic slot positions, the Qualisys adapter on synthetic QTM exports, the JTS-MM adapter
on synthetic recording leaves with an injected pose loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from biomech.validation import trueness as TR
from biomech.validation.registration import make_pose
from biomech.validation.synthetic import write_qtm_mat

U = np.array([0.6, 0.0, 0.8])                         # ladder direction (unit)
C0 = np.array([100.0, -50.0, 900.0])
NOMINAL = np.arange(0.0, 201.0, 20.0)


def _slot_points(mm: float, n: int, rng, sd: float, scale: float = 1.0,
                 err: float = 0.0, drift_mm_per_s: float = 0.0, fs: float = 200.0):
    t = np.arange(n) / fs
    p = C0 + (scale * mm + err) * U + sd * rng.standard_normal((n, 3))
    p += drift_mm_per_s * t[:, None] * np.array([1.0, 0.0, 0.0])
    return t, p


def test_metric_line_recovers_direction():
    pts = C0 + NOMINAL[:, None] * U
    c, u = TR.metric_line(NOMINAL, pts)
    np.testing.assert_allclose(u, U, atol=1e-12)
    np.testing.assert_allclose(c, C0, atol=1e-9)
    with pytest.raises(ValueError):
        TR.metric_line(np.zeros(3), pts[:3])


def test_ladder_metrics_scale_error_and_precision():
    """A 0.5 % scale error gives a 0.1-mm mean step deviation; the static SD is recovered."""
    rng = np.random.default_rng(0)
    slots = []
    for mm in NOMINAL:
        t, p = _slot_points(mm, 2000, rng, sd=0.02, scale=1.005)
        p[:30] += 5.0                                  # handling at the start: trimmed (80 %)
        slots.append(TR.SlotSamples("sysA", "frontal", "near", f"d{int(mm):03d}", mm, t, p))
    stab, ttab, arr = TR.ladder_metrics(slots)
    r = ttab.iloc[0]
    assert r["n_slots"] == 11 and r["n_steps"] == 10
    assert r["trueness_step_mean_mm"] == pytest.approx(0.1, abs=0.005)
    assert r["trueness_ci_lo_mm"] < 0.1 < r["trueness_ci_hi_mm"]
    assert r["scale_error_pct"] == pytest.approx(0.5, abs=0.02)
    assert r["distance_error_end_mm"] == pytest.approx(1.0, abs=0.02)
    assert r["precision_static_sd_3d_mm"] == pytest.approx(0.02 * np.sqrt(3), rel=0.05)
    assert r["accuracy_rms_mm"] >= r["line_resid_rms_mm"]
    assert len(stab) == 11 and (stab["n_static"] == 1600).all()
    assert "ladder__sysA__frontal__near__resid_mm" in arr


def test_ladder_metrics_absorbs_tilt_and_sees_a_bad_slot():
    rng = np.random.default_rng(1)
    slots = []
    for mm in NOMINAL:
        err = 0.8 if mm == 100 else 0.0               # one slot 0.8 mm off along the ladder
        t, p = _slot_points(mm, 500, rng, sd=0.01, err=err)
        slots.append(TR.SlotSamples("s", "lateral", "far", f"d{int(mm):03d}", mm, t, p))
    stab, ttab, _ = TR.ladder_metrics(slots)
    steps = stab.set_index("nominal_mm")["step_dev_mm"]
    assert steps[100.0] == pytest.approx(0.8, abs=0.01)
    assert steps[120.0] == pytest.approx(-0.8, abs=0.01)
    assert abs(ttab.iloc[0]["trueness_step_mean_mm"]) < 0.01      # telescopes to ~0
    worst = stab.loc[stab["resid_mm"].idxmax(), "nominal_mm"]
    assert worst == 100.0


def test_precision_detrends_and_reports_drift():
    rng = np.random.default_rng(2)
    t, p = _slot_points(0.0, 6000, rng, sd=0.03, drift_mm_per_s=0.01)   # 0.6 mm/min
    tab, arr = TR.precision_metrics([TR.PrecisionSamples("s", "near", t, p)])
    r = tab.iloc[0]
    assert r["drift_mm_per_min"] == pytest.approx(0.6, abs=0.01)
    assert r["drift_total_mm"] == pytest.approx(0.3, abs=0.01)          # over 30 s
    assert r["sd_3d_mm"] == pytest.approx(0.03 * np.sqrt(3), rel=0.05)
    assert r["sd_3d_raw_mm"] > r["sd_3d_mm"]
    assert "precision__s__any__near__detrended_mm" in arr


def _write_q(path: Path, p: np.ndarray, body: str = "EvalArraz"):
    T = make_pose(np.tile(np.eye(3), (len(p), 1, 1)), p)
    empty = np.zeros_like(T)
    write_qtm_mat(path, path.stem.replace("-", "_"), 200.0,
                  {"CT": empty, "HP": empty, "MP": empty, body: T})


def _qualisys_folder(root: Path, rng) -> Path:
    folder = root / "trueness_precision"
    folder.mkdir(parents=True)
    for ori in ("frontal", "lateral"):
        for mm in (0.0, 20.0, 40.0, 60.0):
            _, p = _slot_points(mm, 400, rng, sd=0.01, scale=1.002)
            _write_q(folder / f"trueness-{ori}-near-d{int(mm):03d}.mat", p)
    _, p = _slot_points(0.0, 1200, rng, sd=0.02)
    _write_q(folder / "precision-near.mat", p)
    (folder / "trueness-lateral-far-d2000015.qtm").write_bytes(b"raw")    # stray, no .mat
    (folder / "trueness-frontal-near-d000.qtm").write_bytes(b"raw")       # raw next to export
    _write_q(folder / "notes-take.mat", p)                                 # unknown name
    return folder


def test_qualisys_adapter_reads_and_logs(tmp_path: Path):
    folder = _qualisys_folder(tmp_path, np.random.default_rng(3))
    slots, prec, log = TR.qualisys_ladder(folder, ignore=["trueness-lateral-far-d2000015"])
    assert len(slots) == 8 and len(prec) == 1
    s = next(x for x in slots if x.orientation == "lateral" and x.slot == "d040")
    assert s.nominal_mm == 40.0 and s.p_mm.shape == (400, 3) and s.distance == "near"
    ign = log[log.status == "ignored"].set_index("file")["reason"]
    assert ign["trueness-lateral-far-d2000015"] == "in the ignore list"
    assert "not a ladder slot" in ign["notes-take"]
    assert "trueness-frontal-near-d000" not in ign.index                  # .qtm + .mat: used


def test_parse_leaf_names():
    assert TR.parse_leaf("eval_near_lateral_trueness_d040_20261001T120000Z")["mm"] == 40.0
    a = TR.parse_leaf("trueness_near_lateral_d040_20261001T120000Z")
    assert a == {"kind": "trueness", "distance": "near", "orientation": "lateral", "mm": 40.0}
    b = TR.parse_leaf("trueness_frontal_far_200mm_20261001T120000Z")
    assert b["orientation"] == "frontal" and b["distance"] == "far" and b["mm"] == 200.0
    c = TR.parse_leaf("precision_far_20261001T120000Z")
    assert c == {"kind": "precision", "distance": "far", "orientation": ""}
    assert TR.parse_leaf("calibration_20261001T120000Z") is None


def test_run_ref_trueness_both_adapters(tmp_path: Path):
    """Qualisys from synthetic exports; JTS-MM from synthetic leaves through a fake loader
    with a known 0.2-mm noise; both systems appear side by side."""
    rng = np.random.default_rng(4)
    data = tmp_path / "data"
    _qualisys_folder(data, rng)
    ladder = data / "jtsmm_ladder"
    for ori in ("frontal", "lateral"):
        for mm in (0, 20, 40, 60):
            for stamp in ("20261001T120000Z", "20261001T130000Z"):     # repeated: newest used
                (ladder / f"trueness_near_{ori}_d{mm:03d}_{stamp}").mkdir(parents=True)
    (ladder / "precision_near_20261001T140000Z").mkdir()
    (ladder / "trueness_broken_20261001T140000Z").mkdir()

    def loader(leaf: Path, kind: str):
        info = TR.parse_leaf(leaf.name)
        n = 400 if kind == "trueness" else 1200
        t, p = _slot_points(info.get("mm", 0.0), n, rng, sd=0.2)
        return t, p

    doc = {"qualisys_ladder_dir": "trueness_precision", "jtsmm_ladder_dir": "jtsmm_ladder"}
    tables, arrays = TR.run_ref_trueness(doc, data, {}, jtsmm_loader=loader)
    tt = tables["trueness"].set_index(["system", "orientation"])
    assert set(tables["trueness"]["system"]) == {"qualisys", "jtsmm"}
    assert tt.loc[("qualisys", "frontal"), "scale_error_pct"] == pytest.approx(0.2, abs=0.05)
    assert tt.loc[("jtsmm", "lateral"), "precision_static_sd_3d_mm"] == \
        pytest.approx(0.2 * np.sqrt(3), rel=0.1)
    files = tables["files"]
    j = files[files.system == "jtsmm"]
    assert (j.status == "used").sum() == 9                               # 8 slots + precision
    assert (j.reason == "a newer leaf of the same condition exists").sum() == 8
    assert "trueness_broken_20261001T140000Z" in set(j[j.status == "ignored"].file)
    used = tables["slots"][tables["slots"].system == "jtsmm"]["source"]
    assert all("130000Z" in s for s in used)
    summ = tables["summary"]
    assert {"precision_30s_sd_3d_mm", "accuracy_rms_mm"} <= set(summ.columns)
    assert summ["precision_30s_sd_3d_mm"].notna().all()


def test_run_ref_trueness_jtsmm_not_configured(tmp_path: Path):
    data = tmp_path / "data"
    _qualisys_folder(data, np.random.default_rng(5))
    doc = {"qualisys_ladder_dir": "trueness_precision", "jtsmm_ladder_dir": None}
    tables, _ = TR.run_ref_trueness(doc, data, {})
    assert set(tables["trueness"]["system"]) == {"qualisys"}
    notes = tables["notes"].set_index("system")["note"]
    assert "not set" in notes["jtsmm"]


def test_ref_trueness_stage_registered():
    from biomech.pipeline import get_stage

    st = get_stage("ref_trueness")
    assert st.group == "validation" and st.scope == "cohort" and st.depends == ()
