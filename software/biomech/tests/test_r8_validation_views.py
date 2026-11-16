# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper perspectives of (J2.4 / J2.5 / J2.6, ``viz/views/validation_r8``):
every new form keeps the existing default (and its export id), draws the stored values, and
the views of stage outputs that arrive with the next run (``ref_diagnostics``,
``ref_sampling:peak_ratio_acc``) say so instead of failing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.check.text import caption_rubric_issues, text_issues
from biomech.viz.spec import FigureSpec
from biomech.viz.views import registry as R
from biomech.viz.views import validation_r8 as R8
from biomech.viz.views import vocab as V

PIDS = ("s01", "s02")
FORMS = {"val_bench": ("panels", "dumbbell", "strip", "merged"),
         "val_sync_overview": ("panels", "compact"),
         "val_kin_aligned_timeseries": ("timeseries", "multiples"),
         "val_budget": ("bars", "stacked", "dot"),
         "val_hinge": ("offset_dots", "dumbbell", "difference"),
         "val_camerafree": ("offset_dots", "ladder", "slope"),
         "val_emg_profiles": ("panels", "overlay", "difference"),
         "val_sampling": ("panels", "quantities", "bandwidth"),
         "val_imu_alone": ("offset_dots", "dumbbell", "limits"),
         "val_error_budget": ("dots", "ratio", "heatmap"),
         "val_fusion_segments": ("lines", "paired", "dumbbell"),
         "val_emg_side": ("offset_dots", "heatmap", "correlations"),
         "val_marking_accuracy": ("components", "arrows3d", "ellipsoids3d"),
         "val_sync_ct3d": ("paths", "geometry")}
NEW = ("val_frame_intervals", "val_label_swaps", "val_errors_vs_effects", "val_fusion_accuracy",
       "val_mounting", "val_marking_causes", "val_paths3d_error")


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


def _kin(rng: np.random.Generator) -> dict:
    n = 600
    ph = np.repeat(np.array(["rest", "open_close", "protrusion_retrusion", "chewing"]), n // 4)
    t = np.arange(n) * 0.025
    t[300:] += 0.025                              # one dropped frame
    q = np.column_stack([np.sin(t), np.cos(t), -10 + 10 * np.sin(2 * t)])
    s = q + rng.normal(0, 0.5, q.shape)
    err = np.linalg.norm(s - q, axis=1)
    rot = np.column_stack([0 * t, 10 * np.sin(2 * t), 0 * t])
    return {"t_s": t, "t_master_s": t + 100.0, "phase": ph, "evaluated": ph != "rest",
            "gate": np.zeros(n, bool), "incisor_s_aligned_mm": s, "incisor_q_mm": q,
            "err_incisor_aligned_mm": err, "rot_s_aligned_deg": rot + 0.1, "rot_q_deg": rot,
            "rot_q_aligned_deg": rot}


def _features() -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(PIDS):
        for task in ("open_close", "protrusion_retrusion", "left_right"):
            for feat, unit in (("amp_mm", "mm"), ("tmj_excursion_mm", "mm"),
                               ("rot_excursion_deg", "deg")):
                rows.append({"pid": pid, "task": task, "feature": feat, "unit": unit,
                             "equiv_margin": 1.0 + i + (task == "open_close")})
    return pd.DataFrame(rows)


@pytest.fixture()
def ctx():
    rng = np.random.default_rng(3)
    arrays = {("ref_kinematics", p): _kin(rng) for p in PIDS}
    budget = pd.DataFrame([{"pid": p, "part": part, "segment": seg, "kind": kind,
                            "outcome": oc, "unit": "mm", "sigma_obs": 2.0, "sigma_ref": r,
                            "sigma_jts_corrected": np.sqrt(4 - r * r) if r < 2 else np.nan,
                            "ref_var_share": min(r * r / 4, 1.0),
                            "separable": r < 2}
                           for p in PIDS for r in (0.5, 2.5)
                           for part, seg, kind, oc in (("marking", "teeth", "marking", "teeth"),
                                                       ("motion", "rest", "static", "incisor"))]
                          ).drop_duplicates(["pid", "part", "outcome"])
    hinge = pd.DataFrame([{"pid": p, "task": "open_close", "system": s, "n_openings": 3,
                           "hinge_fit_mm": 2.0 + (s == "jtsmm"), "hinge_tmj_mm": 4.0,
                           "axis_to_condyles_mm": 5.0}
                          for p in PIDS for s in ("jtsmm", "qualisys")])
    hdiff = pd.DataFrame([{"pid": p, "task": "open_close", "d_hinge_fit_mm": 1.0,
                           "d_hinge_tmj_mm": 0.0} for p in PIDS])
    smooth = pd.DataFrame([{"pid": p, "segment": t, "estimator": e, "n": 100,
                            "jitter_rms_mm": 0.5, "jitter_err_rms_mm": j,
                            "err_rms_mm": er}
                           for p in PIDS for t in ("open_close", "chewing")
                           for e, er, j in (("F0", 1.8, 0.8), ("F0_lp6", 1.5, 0.3),
                                            ("F2", 0.9, 0.2), ("F2P", 0.88, 0.1))])
    labels = pd.DataFrame([{"pid": p, "body": b, "margin_mm": m, "ambiguous": m < 3,
                            "n_frames": 100, "n_swapped": 10 * (b == "CT"),
                            "n_undetermined": 0} for p in PIDS
                           for b, m in (("CT", 1.0), ("HP", 13.0), ("MP", 17.0))])
    gap = pd.DataFrame([{"quantity": "frame_max_deg", "pid_smaller": "s01", "pid_larger": "s02",
                         "value_smaller": 3.0, "value_larger": 6.0, "larger_minus_smaller": 3.0,
                         "larger_over_smaller": 2.0},
                        {"quantity": "head_sv_ratio", "pid_smaller": "s01", "pid_larger": "s02",
                         "value_smaller": 0.5, "value_larger": 0.1,
                         "larger_minus_smaller": -0.4, "larger_over_smaller": 0.2}])
    holds = pd.DataFrame([{"pid": p, "landmark": "max_point_1", "group": "maxillary",
                           "e_mm": 5.0, "predicted_tip_mm": 6.0, "label_swapped": p == "s02"}
                          for p in PIDS])
    tables = {("ref_budget", "budget"): budget, ("ref_hinge", "hinge"): hinge,
              ("ref_hinge", "hinge_diff"): hdiff, ("ref_fusion", "smoothness"): smooth,
              ("ref_quality", "label_correction"): labels,
              ("ref_agreement", "features_aligned"): _features(),
              ("ref_budget", "marking_gap"): gap, ("ref_budget", "marking_holds"): holds,
              ("ref_sampling", "bias"): pd.DataFrame(
                  [{"pid": p, "task": "open_close", "pipeline": pipe, "quantity": q, "n": 8,
                    "ratio_median": 0.8, "ratio_ci_lo": 0.7, "ratio_ci_hi": 0.9}
                   for p in PIDS for pipe in ("ref40", "jtsmm")
                   for q in ("peak_open_vel_dps", "peak_close_vel_dps",
                             "peak_incisal_speed_mm_s")])}
    return Ctx(tables, arrays)


def _build(ctx, vid, **params):
    return V.reader_spec(R.build(vid, R.request(ctx, vid, **params)))


def test_forms_registered_and_defaults_unchanged():
    for vid, forms in FORMS.items():
        v = R.get(vid)
        assert tuple(v.choices["form"]) == forms, vid
        assert v.defaults["form"] == forms[0], vid           # the current depiction stays
        assert R.default_choice(v, "form", None) == forms[0]
    for vid in NEW:
        assert R.get(vid).group == "validation"


def test_default_form_keeps_the_export_id(ctx):
    spec = R.build("val_hinge", R.request(ctx, "val_hinge"))
    assert spec.id == "val_hinge"
    alt = R.build("val_hinge", R.request(ctx, "val_hinge", form="dumbbell"))
    assert alt.id == "val_hinge--form-dumbbell"


@pytest.mark.parametrize("vid,params", [
    ("val_budget", {"form": "stacked"}), ("val_budget", {"form": "dot"}),
    ("val_hinge", {"form": "dumbbell"}), ("val_hinge", {"form": "difference"}),
    ("val_kin_aligned_timeseries", {"pid": "s01", "form": "multiples"}),
    ("val_sampling", {"form": "quantities"}),
    ("val_frame_intervals", {}), ("val_label_swaps", {}), ("val_fusion_accuracy", {}),
    ("val_mounting", {}), ("val_marking_causes", {}), ("val_paths3d_error", {})])
def test_views_build_with_structured_readable_captions(ctx, vid, params):
    spec = _build(ctx, vid, **params)
    assert isinstance(spec, FigureSpec) and spec.panels
    assert spec.meta.get("caption"), vid                         # CaptionSpec
    assert not text_issues(spec), text_issues(spec)
    assert not caption_rubric_issues(spec), caption_rubric_issues(spec)


def test_budget_stacked_draws_the_stored_reference_share(ctx):
    spec = _build(ctx, "val_budget", form="stacked")
    ref = [s for s in spec.panels[0].series if s.label and "Qualisys part" in s.label][0]
    b = ctx.table("ref_budget", "budget")
    assert sorted(np.round(ref.x, 6)) == sorted(np.round(100 * b["ref_var_share"], 6))


def test_hinge_difference_reads_the_stored_differences(ctx):
    spec = _build(ctx, "val_hinge", form="difference")
    bars = [s for s in spec.panels[0].series if s.kind == "barh"]
    assert {float(v) for s in bars for v in s.x} == {1.0, 0.0}


def test_frame_intervals_fallback_uses_the_stage_routine(ctx):
    from biomech.validation import diagnostics as DG

    spec = _build(ctx, "val_frame_intervals")
    assert "not in this result store" in spec.caption
    ft, fg = DG.frame_timing("s01", ctx.arrays("ref_kinematics", "s01")["t_s"])
    pts = [s for s in spec.panels[0].series if s.kind == "scatter"][0]
    want = [ft["n_intervals"].iloc[0] - ft["n_long"].iloc[0], *fg["n"]]
    assert list(pts.y) == want
    # stored tables win (and the note goes)
    ctx._t[("ref_diagnostics", "frame_timing")] = pd.concat(
        [DG.frame_timing(p, ctx.arrays("ref_kinematics", p)["t_s"])[0] for p in PIDS])
    ctx._t[("ref_diagnostics", "frame_gaps")] = pd.concat(
        [DG.frame_timing(p, ctx.arrays("ref_kinematics", p)["t_s"])[1] for p in PIDS])
    assert "not in this result store" not in _build(ctx, "val_frame_intervals").caption


def test_mounting_pending_then_drawn(ctx):
    spec = _build(ctx, "val_mounting")
    assert len(spec.panels) == 1 and "result store" in spec.caption
    ctx._t[("ref_diagnostics", "mounting_sessions")] = pd.DataFrame(
        [{"pid": p, "body": b, "slot": "cra", "session_a": "2025-03-20",
          "session_b": "2025-03-21", "mounting_change_deg": a}
         for p in PIDS for b, a in (("HP", 176.0), ("MP", 1.2))])
    ctx._t[("ref_diagnostics", "mounting_residual")] = pd.DataFrame(
        [{"pid": p, "body": "HP", "mounting": m, "own": m == "2025-03-20", "n_folds": 5,
          "rms_median_deg": 3.0, "rms_min_deg": 2.0, "rms_max_deg": 4.0,
          "angle_to_intake_median_deg": 10.0} for p in PIDS
         for m in ("in_take", "2025-03-20", "2025-03-21")])
    spec = _build(ctx, "val_mounting")
    assert len(spec.panels) == 2
    bar = [s for s in spec.panels[0].series if s.kind == "barh"][0]
    assert sorted(bar.x) == [1.2, 176.0]                  # one row per session pair and IMU


def test_sampling_acceleration_panel_waits_for_the_stored_ratios(ctx):
    spec = _build(ctx, "val_sampling", form="quantities")
    acc = spec.panels[-1]
    assert "not stored" in acc.note and not acc.series
    ctx._t[("ref_sampling", "peak_ratio_acc")] = pd.DataFrame(
        [{"pid": p, "task": "open_close", "pipeline": pipe, "quantity": "peak_abs_acc_dps2",
          "n": 8, "ratio_median": 0.6, "ratio_ci_lo": 0.5, "ratio_ci_hi": 0.7}
         for p in PIDS for pipe in ("ref40", "jtsmm")])
    acc = _build(ctx, "val_sampling", form="quantities").panels[-1]
    assert any(s.kind == "scatter" and np.isclose(s.x, 0.6).all() for s in acc.series)


def test_errors_vs_effects_ratio_is_effect_over_margin(ctx, monkeypatch):
    req = pd.DataFrame([{"key": k, "median": 10.0, "value": 20.0, "n": 13}
                        for k in ("translation_mm", "opening_mm", "protrusion_mm",
                                  "laterotrusion_mm", "opening_angle_deg")])
    monkeypatch.setattr(R8, "dataset_requirements", lambda: (req, "r13a"))
    for how, s01 in (("median", 1.0), ("task", 2.0)):
        spec = _build(ctx, "val_errors_vs_effects", margin=how)
        ratios = [float(s.x[0]) for s in spec.panels[1].series if s.kind == "scatter"]
        assert np.isclose(ratios[0], 10.0 / s01)                 # first row, P1
        assert "cohort of 13" in spec.caption and not caption_rubric_issues(spec)
    monkeypatch.setattr(R8, "dataset_requirements", lambda: (pd.DataFrame(), ""))
    spec = _build(ctx, "val_errors_vs_effects")
    assert "not available" in spec.panels[0].note and not spec.panels[1].series[:-1]


def test_paths3d_classes_partition_the_drawn_frames(ctx):
    spec = _build(ctx, "val_paths3d_error")
    sag = spec.panels[1]
    n = sum(len(s.x) for s in sag.series if s.kind == "scatter")
    a = ctx.arrays("ref_kinematics", "s01")
    from biomech.viz.views.validation_gallery import _movement_mask

    m = _movement_mask(a)
    step = max(1, int(m.sum() // 1000))
    assert n == len(np.flatnonzero(m)[::step])


def test_gallery_enumerates_perspective_pickers():
    """J2.6: the gallery shows every form x perspective (frames, mode, margin, ...); a picker
    the paper entry fixes stays fixed."""
    from biomech.viz import export as E

    g = E.gallery_items(None, [{"id": "val_kin_waveforms", "frames": "own"},
                               "val_errors_vs_effects"])
    wave = [x for x in g if x["id"] == "val_kin_waveforms"]
    assert {x["frames"] for x in wave} == {"own"}
    assert {(x["form"], x["mode"]) for x in wave} == {(f, m) for f in ("panels", "curves",
                                                                       "difference")
                                                      for m in ("disp", "abs")}
    eff = [x for x in g if x["id"] == "val_errors_vs_effects"]
    assert {(x["frames"], x["margin"]) for x in eff} == {("aligned", "median"), ("aligned", "task"),
                                                         ("own", "median"), ("own", "task")}


def test_gallery_never_lists_one_depiction_twice():
    """An expanded picker can reproduce another entry of the set (frames own): it is listed
    once, so two compiles never race on the same files."""
    from biomech.viz import export as E

    g = E.gallery_items(None, [{"id": "val_kin_agreement"},
                               {"id": "val_kin_agreement", "frames": "own"}])
    sig = [tuple(sorted((k, str(v)) for k, v in x.items())) for x in g]
    assert len(sig) == len(set(sig)) == 4


def test_paper_forms_2026_11_05_registered():
    """Sensors figure requests: the plane grid of both takes and the merged bench stay
    alternative forms (defaults unchanged); the error classes avoid the JTS-MM blue."""
    assert R.get("val_kin_planes").choices["form"] == ("segments", "paper_grid")
    assert R.default_choice(R.get("val_kin_planes"), "form", None) == "segments"
    assert "merged" in R.get("val_bench").choices["form"]
    import inspect

    from biomech.viz.views import validation_r8

    src = inspect.getsource(validation_r8.val_paths3d_error)
    assert '"sky"' in src and '"ink2"' in src and '"purple"' not in src
