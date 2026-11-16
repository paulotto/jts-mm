# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""CLI record session meta from YAML tests."""

from __future__ import annotations

import copy
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from jtsmm.cli.main import main as cli_main
from jtsmm.config.registry import session_meta_from_config
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL, load_operator_template


def test_session_meta_yaml_only():
    doc = copy.deepcopy(load_operator_template())
    doc["session"]["subject_id"] = "hash-abc"
    doc["session"]["protocol"] = "battery_v1"
    meta = session_meta_from_config(doc)
    assert meta["subject_id"] == "hash-abc"
    assert meta["protocol"] == "battery_v1"


def test_cli_subject_overrides_yaml(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    doc["session"]["subject_id"] = "from-yaml"
    doc.setdefault("host", {}).setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    path = tmp_path / "jtsmm.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    captured: dict = {}

    def fake_run_session(config_path, *, meta=None, **kwargs):
        captured["meta"] = dict(meta or {})
        from pathlib import Path
        out = tmp_path / "session.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("stub", encoding="utf-8")
        return out

    runner = CliRunner()
    with patch("jtsmm.cli.record.run_session", side_effect=fake_run_session):
        result = runner.invoke(
            cli_main,
            [
                "record",
                "--config",
                str(path),
                "--mock",
                *MOCK_RECORD_DISABLE_OPTICAL,
                "--duration",
                "1",
                "--skip-preflight",
                "--no-emg-check",
                "--no-integrity",
                "--subject",
                "cli-wins",
            ],
        )
    assert result.exit_code == 0, result.output
    assert captured["meta"]["subject_id"] == "cli-wins"


def test_config_init_creates_file(tmp_path):
    runner = CliRunner()
    dest = tmp_path / "new.yaml"
    result = runner.invoke(
        cli_main,
        ["config", "init", "--dest", str(dest)],
    )
    assert result.exit_code == 0
    assert dest.is_file()
    loaded = yaml.safe_load(dest.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
