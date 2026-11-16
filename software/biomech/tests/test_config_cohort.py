# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from biomech.cohort import discover_sessions, draft_manifest, load_cohort, parse_session_name
from biomech.config import ConfigError, load_config


def test_parse_session_name():
    info = parse_session_name("session_s02_f_44_mvc_20250303T100000Z_00000006")
    assert info["pid"] == "s02" and info["sex"] == "f" and info["age"] == 44
    assert info["kind"] == "mvc"
    assert parse_session_name("session_test_imu_mounting") is None


def test_discover_ignores_foreign_folders(data_dir: Path):
    names = [s.name for s in discover_sessions(data_dir)]
    assert "not_a_session" not in names and len(names) == 7


def test_draft_takes_first_session_as_primary(data_dir: Path):
    doc = draft_manifest(data_dir)
    s02 = next(r for r in doc["participants"] if r["pid"] == "s02")
    assert s02["motion"].endswith("20250303T101000Z_00000004")
    assert s02["secondary"] == ["session_s02_f_44_20250306T101000Z_00000005"]


def test_validate_catches_inconsistencies(tmp_path: Path, data_dir: Path):
    doc = draft_manifest(data_dir)
    doc["participants"][0]["age"] = 99                # disagrees with folder name
    doc["participants"][1]["include"] = False          # no reason given
    doc["participants"][2]["mvc"] = doc["participants"][2]["motion"]  # wrong take kind
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(doc))
    problems = load_cohort(p, data_dir).validate()
    assert any("disagrees" in x for x in problems)
    assert any("without a reason" in x for x in problems)
    assert any("listed as mvc" in x for x in problems)


def test_demographics_cover_included_only(study):
    cfg = study()
    d = load_cohort(cfg.cohort_path, cfg.data_dir).demographics()
    assert d["n"] == 2 and d["age_min"] == 31 and d["age_max"] == 44


def test_config_rejects_unknown_keys(tmp_path: Path, data_dir: Path):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"paths": {"data_dir": str(data_dir), "cohort": "c.yaml",
                                           "typo": 1}}))
    with pytest.raises(ConfigError, match="paths.typo"):
        load_config(p)


def test_config_resolves_relative_paths_and_env(tmp_path: Path, data_dir: Path, monkeypatch):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump({"study": "x", "paths": {"data_dir": "nowhere",
                                                         "cohort": "c.yaml"}}))
    monkeypatch.setenv("JTSMM_DATA_DIR", str(data_dir))
    cfg = load_config(p)
    assert cfg.data_dir == data_dir and cfg.cohort_path == tmp_path / "c.yaml"
    assert cfg.out_dir.parts[-2:] == ("analysis", "x")


def test_stage_typo_and_type_change_rejected(study):
    from biomech.config import ConfigError
    from biomech.pipeline import validate_config

    cfg = study({"kinematic": {"x": 1}})               # typo of "kinematics"
    with pytest.raises(ConfigError, match="unknown stage block"):
        validate_config(cfg)
    with pytest.raises(ConfigError, match="expected dict"):
        study({"kinematics": 3})
    with pytest.raises(ConfigError, match="expected number"):
        study({"emg": {"cycle_mask_max": "a lot"}})


def test_defaults_are_package_data():
    from biomech.config import DEFAULTS_PATH, PACKAGE_DIR

    assert DEFAULTS_PATH.is_file() and DEFAULTS_PATH.parent == PACKAGE_DIR / "etc"
