# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every experiment script imports, and the session-based ones run end to end.

The cohort is synthetic (``test_w3_synth.make_cohort``: two participants, MVC +
motion take each). Scripts are loaded from their file like ``python script.py``
would (their own folder first on ``sys.path``, nothing else added) and driven
through ``main(argv)``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from test_w3_synth import make_cohort

REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "experiments"

# Session-based scripts that run here; the rest need hardware recordings.
RUNNABLE = {
    "5_3_1_trigger_frame_assoc_stats/eval_assoc_stats.py",
    "5_3_2_timing_residuals/eval_timing_residuals.py",
    "5_4_1_emg_properties/eval_emg_properties.py",
    "5_4_2_motion_trajectory_analysis/eval_motion_trajectories.py",
    "5_4_3_inter_subject_variability/eval_spm.py",
    "5_4_4_imu_only_motion_quality/eval_imu_only.py",
    "data/check_subject.py",
    "data/preview_imu_rotation.py",
}
NOT_RUN = {
    "5_1_1_pnp_vs_triangulation/eval_precision.py": "needs persisted optical frame recordings",
    "5_1_1_pnp_vs_triangulation/eval_trueness.py": "needs persisted optical frame recordings",
    "5_1_1_pnp_vs_triangulation/eval_accuracy.py": "needs eval_precision/eval_trueness reports",
    "5_1_2_marker_size_tradeoff/eval_marker_size.py": "needs persisted optical frame recordings",
    "5_2_1_pivot_calibration_repeatability/eval_repeatability.py": "needs CT pivot recordings",
    "5_2_1_pivot_calibration_repeatability/analyze_repeatability.py": "needs CT pivot recordings",
    "5_2_2_imu_optical_handeye/eval_handeye.py": "needs IMU hand-eye recordings",
    "5_2_2_imu_optical_handeye/eval_repeatability.py": "alias of eval_handeye",
    "data/eval_trajectory_quality.py": "needs a JTS-calibrated take (landmark phases + jts config)",
    "data/plot_mp_in_hp_all.py": "needs a JTS-calibrated take (landmark phases + jts config)",
    "data/plot_tmj_subjects.py": "needs a JTS-calibrated take (landmark phases + jts config)",
    "data/run_jts_plots.py": "runs the JTS pipeline in subprocesses on a calibrated take",
}
SCRIPTS = sorted(RUNNABLE | set(NOT_RUN))


def load(rel: str, monkeypatch):
    """Import ``experiments/<rel>`` as ``python <rel>`` would (script dir on sys.path)."""
    path = EXP / rel
    monkeypatch.syspath_prepend(str(path.parent))
    name = "w3_" + rel.replace("/", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_every_script_is_covered():
    found = {str(p.relative_to(EXP)) for p in EXP.glob("[0-9]_*/*.py")}
    found |= {str(p.relative_to(EXP)) for p in EXP.glob("data/*.py")}
    found.discard("5_1_1_pnp_vs_triangulation/array_pose.py")      # helper module, no main
    assert found == set(SCRIPTS)


def test_no_sys_path_edits_or_pi_paths_in_experiments():
    bad = []
    for p in EXP.rglob("*.py"):
        if "icorr2027" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        for needle in ("sys.path", "/home/pi", "_lib", "session_index"):
            if needle in text:
                bad.append(f"{p.relative_to(EXP)}: {needle}")
    assert not bad


@pytest.mark.parametrize("rel", SCRIPTS)
def test_script_imports_and_has_main(rel, monkeypatch):
    if rel.startswith("5_1_"):                    # optical bench scripts: need optical + OpenCV
        pytest.importorskip("cv2")
        pytest.importorskip("optical")
    mod = load(rel, monkeypatch)
    assert callable(getattr(mod, "main", None)), rel
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("w3_cohort")
    make_cohort(root)
    return root


@pytest.fixture
def env(tmp_path, monkeypatch, cohort):
    """Scripts must not touch ~/experiments or ~/jtsmm_data in tests."""
    monkeypatch.setenv("BIOMECH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("JTSMM_DATA_DIR", str(cohort))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    return tmp_path


def test_5_3_1_assoc(env, monkeypatch, cohort):
    mod = load("5_3_1_trigger_frame_assoc_stats/eval_assoc_stats.py", monkeypatch)
    assert mod.main(["--subject", "s01"]) == 0           # data dir from $JTSMM_DATA_DIR
    out = env / "results" / "5_3_1_trigger_frame_assoc_stats" / "results"
    rows = json.loads((out / "derived" / "assoc_sessions.json").read_text())
    assert rows and {r["subject_id"] for r in rows} == {"s01_f_31"}
    prov = json.loads((out / "derived" / "provenance.json").read_text())
    assert prov["n_sessions"] == 1 and prov["data_dir"] == str(cohort)
    assert (out / "assoc_shares.png").is_file()


def test_5_3_2_timing(env, monkeypatch, cohort):
    mod = load("5_3_2_timing_residuals/eval_timing_residuals.py", monkeypatch)
    out = env / "o532"
    assert mod.main(["--data-dir", str(cohort), "--out-dir", str(out), "--no-plot"]) == 0
    doc = json.loads((out / "residual_summary.json").read_text())
    assert doc["pool"]["n"] > 0 and len(doc["sessions"]) == 4      # 2 takes x 2 cameras


def test_5_4_1_emg(env, monkeypatch, cohort):
    import csv

    mod = load("5_4_1_emg_properties/eval_emg_properties.py", monkeypatch)
    out = env / "o541"
    assert mod.main(["--data-dir", str(cohort), "--out-dir", str(out)]) == 0
    rows = list(csv.DictReader((out / "emg_properties.csv").open()))
    assert len(rows) == 2 * 6 * 5                       # takes x muscles x motion phases
    assert {r["mvc_source"] for r in rows} == {"mvc"}
    assert {"mvc_below_recording_max", "recording_max_uv", "mask_frac"} <= set(rows[0])
    snr = [float(r["snr"]) for r in rows]
    assert min(snr) > 1.5                                # tasks are louder than rest
    scales = json.loads((out / "derived" / "mvc_scales.json").read_text())
    assert set(scales) == {"s01_f_31", "s02_m_31"}
    assert (out / "pct_mvc.png").is_file()


def test_5_4_1_flags_low_mvc_instead_of_substituting(tmp_path, monkeypatch):
    from test_w3_synth import make_take

    from biomech.io.inventory import scan

    make_take(tmp_path, "session_s05_f_30_mvc_20250310T090000Z_aa", kind="mvc", burst_uv=12.0)
    make_take(tmp_path, "session_s05_f_30_20250310T091000Z_bb", kind="motion")
    mod = load("5_4_1_emg_properties/eval_emg_properties.py", monkeypatch)
    takes = scan(tmp_path)
    motion = next(t for t in takes if t.take_kind == "motion")
    mvc = next(t for t in takes if t.take_kind == "mvc")
    rows, refs = mod.eval_motion(motion, mvc)
    assert any(r["mvc_below_recording_max"] for r in rows)
    for r in rows:      # the MVC stays the denominator; the recording max is only reported
        assert r["mvc_uv"] == refs[r["channel"]] and r["mvc_source"] == "mvc"
        if r["mvc_below_recording_max"]:
            assert r["recording_max_uv"] > r["mvc_uv"]


def test_5_4_2_then_5_4_3(env, monkeypatch, cohort):
    m542 = load("5_4_2_motion_trajectory_analysis/eval_motion_trajectories.py", monkeypatch)
    assert m542.main(["--data-dir", str(cohort), "--subject", "s01,s02"]) == 0
    out542 = env / "results" / "5_4_2_motion_trajectory_analysis" / "results"
    docs = sorted((out542 / "derived").glob("cycles_*.json"))
    assert len(docs) == 2
    doc = json.loads(docs[0].read_text())
    assert doc["cycle_definition"] == "peak_to_peak"
    assert len(doc["cycles"]["open_close"]) >= 5
    assert all(len(c) == 101 for c in doc["cycles"]["open_close"])
    assert (out542 / "cycle_xy_paths.png").is_file()

    pytest.importorskip("spm1d")
    m543 = load("5_4_3_inter_subject_variability/eval_spm.py", monkeypatch)
    out = env / "o543"
    assert m543.main(["--data-dir", str(cohort), "--out-dir", str(out), "--no-plot"]) == 0
    summary = json.loads((out / "spm_summary.json").read_text())   # cycles dir from env root
    assert summary["n_docs"] == 2
    assert summary["cycle_reps_mm"][0]["ok"] is True
    closers = summary["emg_mvc_closers"][0]
    assert closers["ok"] is True and closers["subjects"] == ["s01_f_31", "s02_m_31"]
    prov = json.loads((out / "derived" / "provenance.json").read_text())
    assert all(v for v in prov["extra"]["mvc_sessions"].values())


def test_5_4_4_imu_only(env, monkeypatch, cohort):
    mod = load("5_4_4_imu_only_motion_quality/eval_imu_only.py", monkeypatch)
    out = env / "o544"
    rc = mod.main(["--data-dir", str(cohort), "--out-dir", str(out),
                   "--mounting-dir", str(env / "no_mounting"), "--no-plot"])
    assert rc == 0
    doc = json.loads((out / "imu_only_summary.json").read_text())
    assert doc["n"] == 2 and doc["rows"][0]["mode"] == "relative"
    assert doc["rows"][0]["R_mount_mta"] is False


def test_5_4_4_reads_latest_mounting(env, monkeypatch, tmp_path):
    import numpy as np

    mod = load("5_4_4_imu_only_motion_quality/eval_imu_only.py", monkeypatch)
    rec = tmp_path / "rec"
    for stamp, angle in (("20250301T000000Z", 0.0), ("20250302T000000Z", 90.0)):
        leaf = rec / f"handeye_run_{stamp}"
        leaf.mkdir(parents=True)
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        R = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
        (leaf / "imu_mounting.json").write_text(json.dumps({"slots": {"imu_cra": {"R_mount": R}}}))
    R, src = mod.load_latest_mounting("imu_cra", rec)
    assert src.endswith("20250302T000000Z/imu_mounting.json") and np.isclose(R[1, 0], 1.0)
    assert mod.load_latest_mounting("imu_mta", rec)[0] is None
    assert mod.load_latest_mounting("imu_cra", tmp_path / "none") == (None, None)


def test_check_subject(env, monkeypatch, cohort, capsys):
    mod = load("data/check_subject.py", monkeypatch)
    assert mod.main(["s01_f_31", "--data-dir", str(cohort)]) in (0, 1)
    text = capsys.readouterr().out
    assert "best SNR" in text and "MVC masseter_r" in text
    assert mod.main(["s1", "--data-dir", str(cohort)]) == 2   # exact: s1 is not s01
    assert mod.main(["bogus"]) == 2


def test_preview_imu_rotation(env, monkeypatch, cohort):
    mod = load("data/preview_imu_rotation.py", monkeypatch)
    out = env / "prev"
    assert mod.main(["--data-dir", str(cohort), "--out-dir", str(out), "--subject", "s02"]) == 0
    assert [p.name for p in out.glob("*.png")] == ["s02_m_31.png"]
