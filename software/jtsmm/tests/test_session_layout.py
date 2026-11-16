# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Session storage path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from jtsmm.io import session_layout


def test_resolve_storage_base_uses_config_when_writable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    base = tmp_path / "data"
    base.mkdir()
    resolved = {"host": {"storage": {"base_dir": str(base)}}}
    assert session_layout.resolve_storage_base(resolved) == base


def test_resolve_storage_base_falls_back_when_opt_not_writable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("JTSMM_DATA_DIR", raising=False)
    session_layout._FALLBACK_WARNED = False
    fallback = tmp_path / "fallback"
    monkeypatch.setattr(session_layout, "_dev_data_dir", lambda: fallback)
    resolved = {"host": {"storage": {"base_dir": "/opt/jtsmm/data"}}}
    assert session_layout.resolve_storage_base(resolved) == fallback


def test_resolve_storage_base_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = tmp_path / "envdata"
    base.mkdir()
    monkeypatch.setenv("JTSMM_DATA_DIR", str(base))
    resolved = {"host": {"storage": {"base_dir": "/opt/jtsmm/data"}}}
    assert session_layout.resolve_storage_base(resolved) == base


def test_session_dir_includes_experiment_name(tmp_path: Path):
    path = session_layout.session_dir(
        tmp_path,
        stamp="20260819T132719Z",
        sid="df1a0755",
        experiment_name="20260819_ds_session01",
    )
    assert path.name == "session_20260819_ds_session01_20260819T132719Z_df1a0755"


def test_session_dir_generic_without_experiment(tmp_path: Path):
    path = session_layout.session_dir(
        tmp_path, stamp="20260819T132719Z", sid="df1a0755"
    )
    assert path.name == "session_20260819T132719Z_df1a0755"


def test_session_output_from_config_uses_yaml_experiment_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    monkeypatch.setenv("JTSMM_DATA_DIR", str(tmp_path))
    resolved = {
        "session": {"experiment": {"name": "20260819_ds_session01", "phases": ["mvc_opening"]}}
    }
    sdir, h5 = session_layout.session_output_from_config(resolved)
    assert sdir.parent == tmp_path
    assert sdir.name.startswith("session_20260819_ds_session01_")
    assert h5.name == "session.h5"


def test_session_dir_mvc_appends_to_experiment_name(tmp_path: Path):
    path = session_layout.session_dir(
        tmp_path,
        stamp="20260902T120000Z",
        sid="deadbeef",
        experiment_name="s01_f_30_take4_mvc",
    )
    assert path.name == "session_s01_f_30_take4_mvc_20260902T120000Z_deadbeef"
