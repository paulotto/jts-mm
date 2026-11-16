# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper depictions for the gallery: every form draws the stored values
(no recomputation), the defaults keep the existing export ids, and the merged tables put
each value in the right frame column with the row's own unit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R
from biomech.viz.views import validation as VA
from biomech.viz.views import validation_gallery as VG

PIDS = ("s01", "s02")
TASKS = ("rest", "open_close", "chewing", "max_point_1")


class Ctx:
    """Serves synthetic validation tables and arrays like a DataContext."""

    def __init__(self, tables: dict, arrays: dict):
        self._t, self._a = tables, arrays
        self.config = type("C", (), {"seed": 0})()

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1

    def pseudonyms(self):
        return {p: i + 1 for i, p in enumerate(PIDS)}

    def table(self, stage, name, pids=None):
        d = self._t.get((stage, name), pd.DataFrame())
        return d[d["pid"].isin(pids)] if pids is not None and "pid" in d else d

    def arrays(self, stage, pid):
        return self._a.get((stage, pid), {})


def _kin_arrays(rng) -> dict:
    n = 400
    ph = np.repeat(np.array(TASKS), n // len(TASKS))
    gate = np.zeros(n, bool)
    gate[150:160] = True                       # some movement frames fail the gate
    return {"phase": ph, "evaluated": ph != "", "gate": gate, "t_s": np.arange(n) * 0.025,
            "err_incisor_mm": rng.gamma(3, 1, n), "err_incisor_aligned_mm": rng.gamma(2, 1, n),
            "err_rot_deg": rng.gamma(4, 1, n), "err_rot_aligned_deg": rng.gamma(1, 1, n)}


def _features(shift: float) -> pd.DataFrame:
    rows = []
    for pid in PIDS:
        for feat, unit in (("amp_mm", "mm"), ("peak_speed_mm_s", "mm/s"),
                           ("rot_excursion_deg", "deg")):
            b = shift + (100 if unit == "mm/s" else 1.0)
            rows.append({"unit": pid, "pid": pid, "task": "open_close", "feature": feat,
                         "bias": b, "bias_ci_lo": b - 0.5, "bias_ci_hi": b + 0.5,
                         "loa_lo": b - 2, "loa_hi": b + 2, "ccc": 0.9, "icc_a1": 0.9,
                         "equiv_margin": np.nan if unit == "mm/s" else abs(b) + 0.5,
                         "equivalent_application": True})
    d = pd.DataFrame(rows)
    d["unit_"] = d["feature"].map({"amp_mm": "mm", "peak_speed_mm_s": "mm/s",
                                   "rot_excursion_deg": "deg"})
    return d.drop(columns="unit").rename(columns={"unit_": "unit"})


@pytest.fixture()
def ctx():
    rng = np.random.default_rng(1)
    arrays = {("ref_kinematics", p): _kin_arrays(rng) for p in PIDS}
    cyc = pd.DataFrame([{"pid": p, "task": "open_close", "cycle": k,
                         "amp_mm_s": 30 + k, "amp_mm_q": 29.5 + k,
                         "peak_speed_mm_s_s": 200.0 + k, "peak_speed_mm_s_q": 120.0 + k,
                         "rot_excursion_deg_s": 18.0, "rot_excursion_deg_q": 16.0}
                        for p in PIDS for k in range(5)])
    tables = {("ref_agreement", "features"): _features(0.5),
              ("ref_agreement", "features_aligned"): _features(0.0),
              ("ref_agreement", "cycles_aligned"): cyc}
    return Ctx(tables, arrays)


def test_defaults_keep_the_existing_export_ids():
    for vid, default in (("val_marking_accuracy", "components"), ("val_marking_frames", "forest"),
                         ("val_kin_waveforms", "panels"), ("val_kin_agreement", "forest"),
                         ("val_fusion_dropout", "curves"), ("val_emg_timing", "strip"),
                         ("val_emg_quality", "offset_dots"), ("val_setup", "signals"),
                         ("val_kin_error_ecdf", "ecdf")):
        v = R.get(vid)
        assert v.defaults["form"] == default and default in v.choice_values("form")
        assert len(v.choice_values("form")) >= 2


def test_ecdf_draws_exactly_the_frames_of_the_pooled_rms(ctx):
    spec = R.get("val_kin_error_ecdf").builder(R.ViewRequest(ctx, options={"form": "ecdf"}))
    assert isinstance(spec, FigureSpec) and len(spec.panels) == 2
    a = ctx.arrays("ref_kinematics", "s01")
    m = VG._movement_mask(a)
    assert not m[150:160].any() and not m[a["phase"] == "rest"].any()
    steps = [s for s in spec.panels[0].series if s.kind == "line"]
    # the first curve (P1, own frames) ends at the largest drawn error and has n frames
    x = steps[0].x
    assert np.isclose(x.max(), a["err_incisor_mm"][m].max())
    assert len(x) == 2 * m.sum()
    assert "equals the table value" in spec.caption


def test_bland_altman_draws_the_stored_bias_and_limits(ctx):
    spec = VG.kin_agreement_ba(R.ViewRequest(ctx, options={"frames": "aligned"}))
    p = spec.panels[0]                           # task amplitude
    d = [s for s in p.series if s.kind == "scatter"][0]
    assert np.allclose(d.y, 0.5)                 # 30 + k − (29.5 + k)
    seg = [s for s in p.series if s.kind == "line" and s.style.get("width") == 1.4]
    stored = ctx.table("ref_agreement", "features_aligned")
    b = stored[(stored.pid == "s01") & (stored.feature == "amp_mm")].bias.iloc[0]
    assert np.allclose(seg[0].y, b)


def test_merged_feature_table_has_row_units_and_both_frame_forms(ctx):
    t = VG.table_features_both(R.ViewRequest(ctx))
    assert isinstance(t, TableSpec)
    keys = [c.key for c in t.columns]
    assert keys[:4] == ["pid", "task", "feature", "feature_unit"]
    assert {c.group for c in t.columns if c.group} == {"own frames", "frames aligned"}
    r = {(row["pid"], row["feature_unit"]): row for row in t.rows}
    assert r[("P1", "mm")]["bias"].startswith("1.50 [1.00, 2.00]")      # own = shift 0.5
    assert r[("P1", "mm")]["bias__al"].startswith("1.00 [0.50, 1.50]")
    assert r[("P1", "mm/s")]["margin"] == "–" and r[("P1", "°")]["feature_unit"] == "°"


def test_original_feature_tables_name_each_rows_unit(ctx):
    t = VA.table_features(R.ViewRequest(ctx))
    cols = {c.key: c for c in t.columns}
    assert "feature_unit" in cols and cols["bias"].unit == ""
    assert [row["feature_unit"] for row in t.rows][:3] == ["mm", "mm/s", "°"]


def test_ecdf_task_dumbbell_reads_the_segment_tables(ctx):
    seg = pd.DataFrame([{"pid": p, "segment": "open_close", "kind": "movement", "gated": g,
                         "incisor_rms_mm": 3.0 + g, "rot_rms_deg": 5.0}
                        for p in PIDS for g in (False, True)])
    al = pd.DataFrame([{"pid": p, "segment": "open_close", "kind": "movement",
                        "incisor_rms_mm": 1.5, "rot_rms_deg": 1.0} for p in PIDS])
    ctx._t[("ref_kinematics", "segments")] = seg
    ctx._t[("ref_kinematics", "segments_aligned")] = al
    spec = VG.kin_error_ecdf(R.ViewRequest(ctx, options={"form": "dumbbell"}))
    dots = [s for s in spec.panels[0].series if s.kind == "scatter"]
    # the markers are split per participant (shape = participant): select by frame form
    own = np.concatenate([s.x for s in dots if s.role == "system:jtsmm"])
    aligned = np.concatenate([s.x for s in dots if s.role != "system:jtsmm"])
    assert np.allclose(own, 4.0) and np.allclose(aligned, 1.5)   # gated own, aligned
