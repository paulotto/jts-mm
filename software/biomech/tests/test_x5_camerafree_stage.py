# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""X5 pipeline stages on the synthetic study: ``camerafree`` and
``camerafree_cohort`` end to end, with a hand-eye mounting file written from the synthetic
truth, and the camera-free views built on the resulting store.

The synthetic participants (``biomech.testing.synth``) have a static head and noise-free
quaternions; their optics are made noise-free here, so every rotation rung must agree with
the optics up to the 40 Hz sampling of the optical reference.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.slow

LEVER_MTA = [10.0, 5.0, -8.0]          # synth.write_participant: IMU site on the chin array


@pytest.fixture(scope="module")
def cf_study(tmp_path_factory):
    from jtsmm.imu.mounting import update_mounting_file

    from biomech.config import load_config
    from biomech.pipeline import run
    from biomech.results import ResultStore
    from biomech.testing.synth import DEFAULT_SPECS, write_study

    root = tmp_path_factory.mktemp("cf_study")
    specs = [replace(s, emg_fs=2000, pose_noise_mm=0.0, pose_noise_deg=0.0)
             for s in DEFAULT_SPECS[:3]]
    path, truths = write_study(root, specs)
    cfg = load_config(path)
    s0 = run(cfg, ["kinematics"], n_jobs=3)
    assert s0.n_failed == 0
    store = ResultStore(cfg.out_dir)
    by_pid = {}
    for pid, tr in truths.items():
        T_max_hp = np.asarray(store.read_arrays("kinematics", pid)["T_max_hp"], float)
        A, X = np.asarray(tr.A_max_cra), np.asarray(tr.X_mta_mp)
        f = root / f"imu_mounting_{pid}.json"
        update_mounting_file(f, {"imu_cra": {"accepted": True,
                                             "R_mount": (T_max_hp[:3, :3].T @ A).tolist()},
                                 "imu_mta": {"accepted": True, "R_mount": X.T.tolist()}}, "synth")
        by_pid[pid] = f.name
    doc = yaml.safe_load(path.read_text())
    doc["stages"]["camerafree"] = {"mounting_by_pid": by_pid, "lever": "param",
                                   "lever_mm": {"cra": [0.0, 0.0, 0.0], "mta": LEVER_MTA},
                                   "lever_source": "synthetic truth"}
    doc["stages"]["camerafree_cohort"] = {"n_boot": 200}
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    cfg = load_config(path)
    summary = run(cfg, ["camerafree_cohort"], n_jobs=3)
    return cfg, ResultStore(cfg.out_dir), truths, summary


def test_stages_run(cf_study):
    from biomech.results import COHORT_UNIT

    _, store, truths, summary = cf_study
    assert summary.statuses["camerafree_cohort"] == {COHORT_UNIT: "ran"}, summary.statuses
    assert set(summary.statuses["camerafree"].values()) == {"ran"}
    s = store.read_table("camerafree", "cf_summary")
    assert s["configured"].all() and len(s) == len(truths)
    assert (s["e1_vs_c3a_max_deg"] < 1e-3).all()          # E1 is C3a's in-take model
    assert (s["closed_vs_tclosed_deg"] < 0.2).all()
    assert not s["psi_b_identifiable"].any() and s["psi_b_fallback"].all()   # static head
    assert (s["lever_source"] == "synthetic truth").all()


def test_noise_free_rotation_is_tiny(cf_study):
    """Noise-free synthetic data: every camera-free rotation rung agrees with the optics
    within the sampling limit of the 40 Hz optical reference."""
    _, store, _, _ = cf_study
    r = store.read_table("camerafree", "cf_rotation")
    sc = r[r["task"] == "scored"].set_index(["pid", "estimator"])
    for est in ("E2a", "E2b", "E3a", "E3b", "E3b_slerp"):
        v = sc.xs(est, level="estimator")
        assert (v["rmse_open_deg"] < 0.05).all(), (est, v["rmse_open_deg"])
        assert (v["rmse_geo_deg"] < 0.05).all(), (est, v["rmse_geo_deg"])
    s = store.read_table("camerafree", "cf_summary").set_index("pid")
    tr_psi = {}
    for pid in s.index:
        a = store.read_arrays("camerafree", pid)
        assert {"t_s", "phase", "open_deg__opt", "open_deg__E3b", "incisor_mm__E4own",
                "rotvec_deg__E3", "anchor"} <= set(a)
        tr_psi[pid] = s.loc[pid, "n_anchors"]
    assert min(tr_psi.values()) >= 5                     # closed poses found between cycles


def test_translation_and_cohort_tables(cf_study):
    _, store, _, _ = cf_study
    t = store.read_table("camerafree", "cf_translation")
    assert {"E4hinge", "E4own", "E5", "E6"} <= set(t["estimator"])
    # s01 is a pure hinge, about an axis 4.5 mm from the condyle midpoint (synth.Motion):
    # rotating about the midpoint misses by about 1 mm at 25 deg
    s01 = t[(t["pid"] == "s01") & (t["task"] == "open_close")].set_index("estimator")
    assert s01.loc["E4hinge", "rmse_inc_mm"] < 1.5
    # synth's motion is only C1 (raised-cosine steps): the accelerometer bridge is not exact
    # here (the C3 gate generator is), but the E4 pseudo-measurements keep E6 bounded
    sc = t[t["task"] == "scored"].pivot_table(index="pid", columns="estimator",
                                              values="rmse_inc_mm")
    assert (sc["E6"] < sc["E5_causal"]).all() and np.isfinite(sc.to_numpy()).all()
    for name in ("cf_cohort", "cf_participant", "cf_contrasts", "cf_generic", "cf_cycles_ba",
                 "cf_drift_slopes", "cf_drift_cohort", "coverage"):
        assert not store.read_table("camerafree_cohort", name).empty, name
    c = store.read_table("camerafree_cohort", "cf_cohort")
    e3 = c[(c["estimator"] == "E3") & (c["outcome"] == "rmse_open_deg")].iloc[0]
    assert e3["n"] == 3 and e3["median"] < 0.05
    assert (c[c["estimator"] == "E7"]["n"] == 0).all()
    k = store.read_table("camerafree_cohort", "cf_contrasts")
    assert set(k["family"]) >= {"rotation", "translation"}
    assert set(store.read_table("camerafree_cohort", "cf_generic")["estimator"]) == {"E4gen"}


def test_oracle_tables(cf_study):
    """X5b / X5c in the stage and the cohort: the noise-free rotation leaves almost no
    rotation-induced incisor error; E4own in the decomposition is the cf_translation row."""
    _, store, truths, _ = cf_study
    d = store.read_table("camerafree", "cf_decomposition")
    assert set(d["estimator"]) == {"X5b_model", "X5b_rotation", "E4own"}
    sc = d[d["task"] == "scored"].pivot_table(index="pid", columns="estimator",
                                              values="rmse_inc_mm")
    assert (sc["X5b_rotation"] < 0.5).all(), sc
    t = store.read_table("camerafree", "cf_translation")
    e4 = t[(t["estimator"] == "E4own") & (t["task"] == "scored")].set_index("pid")
    assert np.allclose(sc["E4own"], e4.loc[sc.index, "rmse_inc_mm"])
    x = store.read_table("camerafree", "cf_task_model")
    assert set(x["estimator"]) == {"X5c"} and set(x["pid"]) == set(truths)
    assert {"label_class", "n_cal_samples", "note"} <= set(x.columns)
    for name in ("cf_oracle_participant", "cf_oracle_cohort", "cf_oracle_contrasts"):
        assert not store.read_table("camerafree_cohort", name).empty, name


def test_views_build(cf_study):
    from biomech.io.context import DataContext
    from biomech.viz.views import registry as R

    cfg, *_ = cf_study
    ctx = DataContext(cfg)
    try:
        ids = [v.id for v in R.views(groups=("dataset",)) if v.id.startswith("cf_")]
        assert {"cf_ladder", "cf_opening_ladder", "cf_drift", "cf_incisor_axes",
                "cf_table_results", "cf_traces", "cf_decomposition",
                "cf_table_oracle"} <= set(ids)
        for vid in ids:
            spec = R.build(vid, R.request(ctx, vid))
            assert spec.caption, vid
            if hasattr(spec, "panels"):
                assert any(p.series for p in spec.panels), vid
            else:
                assert spec.rows, vid
    finally:
        ctx.close()


def test_unconfigured_stage_notes_and_skips(tmp_path):
    """Without a hand-eye mounting the stage stores only its reason (the shared synthetic
    study runs it this way)."""
    from biomech.pipeline import ParticipantContext
    from biomech.imu.stage_camerafree import run_camerafree

    class Cfg:
        source = None
        signals: dict = {}

    ctx = ParticipantContext.__new__(ParticipantContext)
    ctx.params = {}
    ctx.config = Cfg()
    ctx.participant = type("P", (), {"pid": "s01"})()
    res = run_camerafree(ctx)
    assert set(res.tables) == {"cf_summary"} and not res.tables["cf_summary"]["configured"][0]


def test_unconfigured_study_runs_and_views_cover(tmp_path):
    """The shared synthetic study runs the X5 stages without a mounting: every unit runs, the
    stored tables are declared, and every cf view builds (an empty card or the note)."""
    from biomech.config import load_config
    from biomech.io.context import DataContext
    from biomech.pipeline import run
    from biomech.testing.synth import Spec, write_study
    from biomech.viz.views import registry as R
    from biomech.viz.views.tables import declarations

    path, _ = write_study(tmp_path, [Spec(pid="s01", emg_fs=2000, seed=1)])
    cfg = load_config(path)
    s = run(cfg, ["camerafree_cohort"], n_jobs=1)
    assert {u for d in s.statuses.values() for u in d.values()} <= {"ran"}, s.statuses
    ctx = DataContext(cfg)
    try:
        from biomech.check import coverage as C

        tables, _ = C.stored_outputs(ctx.store)
        mine = {f"{st}:{t}" for st, t in tables if st.startswith("camerafree")}
        assert mine and mine <= set(declarations())
        for vid in [v.id for v in R.views(groups=("dataset",)) if v.id.startswith("cf_")
                    or v.id.startswith("tab_camerafree")]:
            assert R.build(vid, R.request(ctx, vid)).caption, vid
    finally:
        ctx.close()
