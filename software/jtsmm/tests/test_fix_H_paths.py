# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Paths from jtsmm.yaml must not depend on the directory `jtsmm` is run from."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from jtsmm.config import load
from jtsmm.config import registry as reg
from jtsmm.config.loader import checkout_root


def test_repo_relative_paths_resolve_from_any_cwd(operator_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = checkout_root()
    assert root is not None
    opt = reg.session_optical_config(operator_config)
    tip = reg.session_ct_tip_calibration(operator_config)
    assert opt == str(root / "software" / "optical" / "etc" / "optical.yaml")
    assert Path(opt).is_file()
    assert tip == str(root / "software" / "optical" / "etc" / "ct_tip_calibration.json")


def test_home_paths_are_expanded(operator_config, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = copy.deepcopy(operator_config)
    cfg["host"]["paths"]["data_dir"] = "~/rig_data"
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    assert reg.paths_data_dir(cfg) == str(tmp_path / "rig_data")
    rec = reg.session_ct_calibration_recordings_dir(operator_config)
    assert rec is not None and not rec.startswith("~") and Path(rec).is_absolute()
    assert Path(reg.session_imu_calibration_recordings_dir(operator_config)).is_absolute()


def test_non_repo_relative_path_resolves_next_to_config(operator_config, tmp_path, monkeypatch):
    cfg = copy.deepcopy(operator_config)
    cfg["session"]["experiment_plan"] = "plans/today.json"
    cfg_dir = tmp_path / "cfgdir"
    cfg_dir.mkdir()
    path = cfg_dir / "jtsmm.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    loaded = load(path)
    monkeypatch.chdir(tmp_path)
    assert reg.session_experiment_plan(loaded) == str(cfg_dir / "plans" / "today.json")


def test_absolute_and_unset_paths_pass_through(operator_config):
    cfg = copy.deepcopy(operator_config)
    cfg["session"]["optical_config"] = "/abs/optical.yaml"
    cfg["session"]["experiment_plan"] = None
    cfg["session"]["ct_tip_calibration"] = ""
    assert reg.session_optical_config(cfg) == "/abs/optical.yaml"
    assert reg.session_experiment_plan(cfg) is None
    assert reg.session_ct_tip_calibration(cfg) is None
