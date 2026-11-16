# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Integration: CLI aborts on incomplete config."""

from __future__ import annotations

import copy

import yaml
from click.testing import CliRunner

from jtsmm.cli.main import main as cli_main
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL, load_operator_template


def test_record_aborts_on_incomplete_config(tmp_path, monkeypatch):
    doc = copy.deepcopy(load_operator_template())
    del doc["host"]["capture"]["settle_ms"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["record", "--config", str(path), "--mock", "--duration", "1", "--skip-preflight"],
    )
    assert result.exit_code != 0
    text = (result.output or "") + str(getattr(result, "exception", "") or "")
    assert "settle_ms" in text or "validation" in text.lower()


def test_config_apply_dry_run_ok(tmp_path, operator_config):
    operator_config.setdefault("host", {}).setdefault("paths", {})["data_dir"] = str(
        tmp_path / "data"
    )
    path = tmp_path / "jtsmm.yaml"
    path.write_text(yaml.safe_dump(operator_config), encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["config", "apply", "--config", str(path), "--dry-run"],
    )
    assert result.exit_code == 0
    assert "config apply" in result.output
