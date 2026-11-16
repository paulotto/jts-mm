# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Tests for monolithic optical.yaml loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from optical.config.loader import (
    calibration_section,
    default_config_path,
    eval_section,
    experiment_section,
    load_config_file,
    markers_section,
    tracker_section,
)
from optical.markers.config import load_marker_config
from optical.tracking.experiment import ExperimentPlan
from optical.tracking.tracker import TrackerConfig


def test_default_optical_yaml_exists():
    assert default_config_path().is_file()


def test_sections_round_trip():
    path = default_config_path()
    doc = load_config_file(path)
    assert "detector" in markers_section(doc)
    assert tracker_section(doc)["pose_method"] in ("pnp", "triangulation")
    cal = calibration_section(doc, config_path=path)
    assert cal["cameras"]
    exp = experiment_section(doc)
    assert exp["phases"]
    assert exp.get("name")
    assert exp.get("folder")
    assert exp.get("condition")
    assert "use_experiment_plan" in exp
    assert eval_section(doc, "eval")["markers"]
    assert eval_section(doc, "eval_quality")["qualities"]


def test_marker_and_tracker_from_yaml():
    path = default_config_path()
    m = load_marker_config(path)
    assert m.tag_size_mm > 0
    t = TrackerConfig.from_path(path)
    assert t.marker_config_path == path
    assert t.cam_ids
    plan = ExperimentPlan.from_json(path)
    assert plan.phases
    assert all(p.kind == "experiment" for p in plan.phases)


def test_mvc_experiment_plan_is_emg_only():
    from optical.tracking.experiment import mvc_experiment_plan

    plan = mvc_experiment_plan()
    assert [p.kind for p in plan.phases] == ["mvc", "mvc"]
    assert [p.name for p in plan.all_phases] == ["mvc_opening", "mvc_closing"]
    assert plan.needs_optical is False


def test_from_dict_skips_mvc_names():
    with pytest.raises(ValueError, match="jtsmm record --mvc"):
        ExperimentPlan.from_dict(
            {"name": "s", "phases": ["mvc_opening", {"name": "mvc_closing", "kind": "mvc"}]}
        )
    mixed = ExperimentPlan.from_dict({"name": "s", "phases": ["mvc_opening", "chewing"]})
    names = [p.name for p in mixed.all_phases]
    assert names[0] == "mand_point_1"
    assert "mvc_opening" not in names
    assert names[-1] == "chewing"


def test_free_recording_plan_skips_landmarks():
    from optical.tracking.experiment import free_recording_plan

    plan = free_recording_plan(name="precision_test_0", condition="static_near")
    assert [p.name for p in plan.phases] == ["static_near"]
    # all_phases would insert calib for motion plans — free mode must not use it.
    assert len(plan.all_phases) == 1 + 8  # documents why run.py uses plan.phases
    assert [p.name for p in plan.all_phases][-3:] == [
        "tmj_left",
        "tmj_right",
        "static_near",
    ]


def test_resolve_session_plan_free_mode(tmp_path: Path):
    from optical.tracking.run import _resolve_session_plan

    cfg = tmp_path / "optical.yaml"
    cfg.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: precision_test_0",
                "  condition: static_near",
                "  use_experiment_plan: false",
                "  phases: [open_close]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan, phases, use_plan = _resolve_session_plan(
        tracker_config_path=cfg,
        experiment_plan_path=None,
        condition=None,
        use_experiment_plan=None,
    )
    assert use_plan is False
    assert plan.name == "precision_test_0"
    assert [p.name for p in phases] == ["static_near"]
    assert all(p.kind == "experiment" for p in phases)


def test_no_landmark_calibration_walks_motion_phases_only():
    from optical.tracking.experiment import ExperimentPlan

    plan = ExperimentPlan.from_dict(
        {
            "name": "trueness",
            "no_landmark_calibration": True,
            "phases": ["d00", "d02"],
        }
    )
    assert plan.no_landmark_calibration is True
    assert [p.name for p in plan.all_phases] == ["d00", "d02"]
    assert all(p.kind == "experiment" for p in plan.all_phases)


def test_resolve_session_plan_skips_landmarks(tmp_path: Path):
    from optical.tracking.run import _resolve_session_plan

    cfg = tmp_path / "optical.yaml"
    cfg.write_text(
        "\n".join(
            [
                "experiment:",
                "  name: trueness_test_1",
                "  condition: near_lateral_d00",
                "  use_experiment_plan: true",
                "  no_landmark_calibration: true",
                "  phases: [d00, d02]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan, phases, use_plan = _resolve_session_plan(
        tracker_config_path=cfg,
        experiment_plan_path=None,
        condition=None,
        use_experiment_plan=None,
    )
    assert use_plan is True
    assert plan.no_landmark_calibration is True
    assert [p.name for p in phases] == ["d00", "d02"]


def test_imu_mount_plan_skips_landmarks():
    plan = ExperimentPlan.from_dict(
        {"name": "imu_mounting", "phases": ["imu_mount_mta", "imu_mount_cra"]}
    )
    assert [p.name for p in plan.all_phases] == ["imu_mount_mta", "imu_mount_cra"]
    assert plan.needs_optical is True


def test_landmark_phases_carry_tooth_hint():
    from optical.tracking.experiment import CALIBRATION_PHASES, LANDMARK_HINTS, Phase

    assert [p.name for p in CALIBRATION_PHASES] == list(LANDMARK_HINTS)
    assert CALIBRATION_PHASES[0].hint.startswith("lower central incisor")
    assert CALIBRATION_PHASES[4].hint == "upper left first molar"
    assert "left condyle" in Phase("tmj_left", "calibration_tmj").hint
    assert Phase("open_close").hint is None
