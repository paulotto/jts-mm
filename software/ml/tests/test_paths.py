# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for ~/ml_data path resolution."""

from __future__ import annotations

from pathlib import Path

from ml.config.loader import load_dataset_config, load_ml_config
from ml.config.paths import (
    evals_dir,
    expand_path,
    session_data_dir,
    ml_data_dir,
    processed_dir,
    resolve_paths_section,
    runs_dir,
    sessions_dir,
)


def test_default_ml_data_under_home(tmp_path: Path) -> None:
    paths = {"data_dir": str(tmp_path / "ml_data")}
    assert ml_data_dir(paths) == tmp_path / "ml_data"
    assert processed_dir(paths) == tmp_path / "ml_data" / "processed"
    assert runs_dir(paths) == tmp_path / "ml_data" / "ml_runs"
    assert evals_dir(paths) == tmp_path / "ml_data" / "ml_evals"
    assert sessions_dir(paths) == tmp_path / "ml_data" / "sessions"


def test_ml_data_dir_env_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path / "custom_ml"))
    assert ml_data_dir({}) == tmp_path / "custom_ml"
    assert processed_dir({}) == tmp_path / "custom_ml" / "processed"


def test_resolve_paths_section_from_package_yaml(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path / "ml_data"))
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    doc = load_ml_config()
    resolved = resolve_paths_section(doc)
    assert resolved["processed_dir"] == str(tmp_path / "ml_data" / "processed")
    assert resolved["runs_dir"] == str(tmp_path / "ml_data" / "ml_runs")
    assert resolved["evals_dir"] == str(tmp_path / "ml_data" / "ml_evals")
    assert resolved["session_data_dir"] == str(Path.home() / "jtsmm_data")


def test_load_dataset_config_resolves_processed_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path / "ml_data"))
    cfg = load_dataset_config()
    assert cfg.output.processed_dir == str(tmp_path / "ml_data" / "processed")
    assert cfg.output.runs_dir == str(tmp_path / "ml_data" / "ml_runs")
    assert cfg.output.evals_dir == str(tmp_path / "ml_data" / "ml_evals")


def test_expand_path_tilde(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert expand_path("~/foo/bar") == tmp_path / "foo" / "bar"


def test_session_data_dir_from_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    paths = {"session_data_dir": str(tmp_path / "recordings")}
    assert session_data_dir(paths) == tmp_path / "recordings"


def test_resolve_paths_section_includes_session_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path / "ml_data"))
    doc = load_ml_config()
    doc = dict(doc)
    doc["paths"] = dict(doc.get("paths") or {})
    doc["paths"]["session_data_dir"] = str(tmp_path / "jtsmm_data")
    resolved = resolve_paths_section(doc)
    assert resolved["session_data_dir"] == str(tmp_path / "jtsmm_data")
