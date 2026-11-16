# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""session.use_experiment_plan + inline session.experiment resolution."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from jtsmm.capture.session import load_experiment_plan, load_mvc_plan
from jtsmm.cli.record import record_cmd
from jtsmm.config import registry as reg
from tests.helpers import load_operator_template


def _cfg(**session_over: object) -> dict:
    doc = copy.deepcopy(load_operator_template())
    doc["session"].update(session_over)
    return doc


def test_template_has_inline_experiment():
    cfg = load_operator_template()
    exp = reg.session_experiment(cfg)
    assert exp is not None
    assert exp["phases"]
    if not cfg["session"].get("use_experiment_plan"):
        assert load_experiment_plan(cfg) is None


def test_legacy_path_without_switch_still_enables():
    cfg = _cfg()
    del cfg["session"]["use_experiment_plan"]
    cfg["session"]["experiment_plan"] = "/tmp/plan.json"
    assert reg.session_use_experiment_plan(cfg) is True


def test_switch_false_ignores_legacy_path():
    cfg = _cfg(use_experiment_plan=False, experiment_plan="/tmp/plan.json")
    assert load_experiment_plan(cfg) is None


def test_inline_plan_when_switch_on():
    cfg = _cfg(
        use_experiment_plan=True,
        experiment_plan=None,
        optical_config=None,
        experiment={
            "name": "subject01_session1",
            "description": "Jaw motion analysis experiment",
            "phases": ["open_close", "left_right", "protrusion_retrusion", "chewing"],
        },
    )
    plan = load_experiment_plan(cfg)
    assert plan is not None
    assert plan.name == "subject01_session1"
    assert [p.name for p in plan.phases] == [
        "open_close",
        "left_right",
        "protrusion_retrusion",
        "chewing",
    ]
    assert all(p.kind == "experiment" for p in plan.phases)
    assert len(plan.all_phases) == 12  # 8 landmark + 4 motion
    assert [p.name for p in plan.all_phases if "calibration" in p.kind][-2:] == [
        "tmj_left",
        "tmj_right",
    ]


def test_imu_mount_plan_skips_landmarks():
    cfg = _cfg(
        use_experiment_plan=True,
        experiment_plan=None,
        optical_config=None,
        experiment={
            "name": "imu_mounting",
            "phases": ["imu_mount_mta", "imu_mount_cra"],
        },
    )
    plan = load_experiment_plan(cfg)
    assert plan is not None
    assert [p.name for p in plan.all_phases] == ["imu_mount_mta", "imu_mount_cra"]
    assert plan.needs_optical is True


def test_mvc_names_are_not_part_of_the_motion_plan():
    cfg = _cfg(
        use_experiment_plan=True,
        experiment_plan=None,
        optical_config=None,
        experiment={
            "name": "mvc_only",
            "description": "MVC",
            "phases": ["mvc_opening", "mvc_closing"],
        },
    )
    with pytest.raises(ValueError, match="jtsmm record --mvc"):
        load_experiment_plan(cfg)


def test_mvc_names_stripped_from_mixed_plan():
    cfg = _cfg(
        use_experiment_plan=True,
        experiment_plan=None,
        optical_config=None,
        experiment={
            "name": "mixed",
            "phases": ["mvc_opening", "mvc_closing", "open_close"],
        },
    )
    plan = load_experiment_plan(cfg)
    names = [p.name for p in plan.all_phases]
    assert "mvc_opening" not in names
    assert names[0] == "mand_point_1"
    assert names[-1] == "open_close"
    assert [p.name for p in plan.phases] == ["open_close"]
    assert plan.needs_optical is True


def test_mvc_plan_uses_yaml_experiment_name():
    cfg = _cfg(
        experiment={
            "name": "s01_f_30_take4",
            "description": "Jaw motion analysis experiment",
            "phases": ["open_close"],
        }
    )
    plan = load_mvc_plan(cfg)
    assert plan.name == "s01_f_30_take4_mvc"
    assert [p.name for p in plan.all_phases] == ["mvc_opening", "mvc_closing"]
    assert plan.needs_optical is False


def test_mvc_plan_falls_back_without_experiment_name():
    cfg = _cfg(experiment=None)
    plan = load_mvc_plan(cfg)
    assert plan.name == "mvc"


def test_cli_path_implies_enable(tmp_path: Path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        '{"name": "from_file", "description": "x", "phases": ["chewing"]}\n',
        encoding="utf-8",
    )
    cfg = _cfg(use_experiment_plan=False, experiment_plan=None, optical_config=None)
    plan = load_experiment_plan(cfg, plan_path=plan_file)
    assert plan is not None
    assert plan.name == "from_file"
    assert [p.name for p in plan.phases] == ["chewing"]


def test_cli_no_use_overrides_path(tmp_path: Path):
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(
        '{"name": "from_file", "phases": ["chewing"]}\n',
        encoding="utf-8",
    )
    cfg = _cfg(use_experiment_plan=True)
    assert load_experiment_plan(cfg, use_plan=False, plan_path=plan_file) is None


def test_missing_plan_raises():
    cfg = _cfg(
        use_experiment_plan=True,
        experiment=None,
        experiment_plan=None,
        optical_config=None,
    )
    with pytest.raises(RuntimeError, match="no plan found"):
        load_experiment_plan(cfg)


def test_record_cmd_switch_off_ignores_inline_plan(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    doc = yaml.safe_load(tmp_config_path.read_text(encoding="utf-8"))
    doc["session"]["use_experiment_plan"] = True
    tmp_config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    session_h5 = tmp_path / "session.h5"
    session_h5.write_bytes(b"")
    captured: dict = {}

    def _fake_run_session(*_args, **kwargs):
        captured.update(kwargs)
        return session_h5

    monkeypatch.setattr("jtsmm.cli.record.run_session", _fake_run_session)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            "--duration",
            "0.1",
            "--no-doctor",
            "--no-integrity",
            "--no-emg-check",
            "--no-use-experiment-plan",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Experiment plan" not in result.output
    assert captured.get("experiment_plan") is None
    assert captured.get("use_experiment_plan") is False


def test_record_cmd_use_plan_without_source_fails(tmp_config_path: Path):
    doc = yaml.safe_load(tmp_config_path.read_text(encoding="utf-8"))
    doc["session"]["experiment"] = None
    doc["session"]["experiment_plan"] = None
    doc["session"]["optical_config"] = None
    doc["session"]["optical_track"] = False
    doc["session"]["ct_calibration"] = False
    tmp_config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            "--duration",
            "0.1",
            "--no-doctor",
            "--no-integrity",
            "--no-emg-check",
            "--use-experiment-plan",
        ],
    )
    assert result.exit_code == 1
    assert "no plan found" in result.output


def test_record_cmd_mvc_uses_dedicated_plan(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    session_h5 = tmp_path / "session.h5"
    session_h5.write_bytes(b"")
    captured: dict = {}

    def _fake_run_session(*_args, **kwargs):
        captured.update(kwargs)
        return session_h5

    monkeypatch.setattr("jtsmm.cli.record.run_session", _fake_run_session)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            "--no-doctor",
            "--no-integrity",
            "--no-emg-check",
            "--mvc",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = captured.get("experiment_plan")
    assert plan is not None
    assert plan.name.endswith("_mvc")
    assert "mvc_opening" in [p.name for p in plan.phases]
    assert [p.name for p in plan.phases] == ["mvc_opening", "mvc_closing"]
    assert captured.get("mvc") is True
    assert captured.get("use_experiment_plan") is False
    assert captured.get("optical_track") is False
    assert captured.get("ct_calibration") is False
    assert captured.get("write_jts_config") is False
    assert "MVC EMG-only" in result.output


def test_record_cmd_mvc_conflicts_with_experiment_plan(tmp_config_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            "--no-doctor",
            "--no-integrity",
            "--no-emg-check",
            "--mvc",
            "--use-experiment-plan",
        ],
    )
    assert result.exit_code == 1
    assert "cannot be combined" in result.output
