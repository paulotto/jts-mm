# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Coverage manifest and audit tests."""

from __future__ import annotations

from click.testing import CliRunner

from jtsmm.config.coverage import load_coverage_manifest, run_audit, template_path
from jtsmm.config.loader import load_yaml
from jtsmm.cli.main import main as cli_main


def test_coverage_manifest_non_empty():
    params = load_coverage_manifest()
    assert len(params) >= 40
    paths = {p["path"] for p in params}
    assert "host.capture.settle_ms" in paths
    assert "session.subject_id" in paths


def test_audit_template_mostly_ok():
    doc = load_yaml(template_path())
    report = run_audit(doc)
    ok = sum(1 for e in report.entries if e.status == "OK")
    assert ok >= len(report.entries) * 0.8


def test_audit_detects_missing_yaml_key():
    doc = load_yaml(template_path())
    del doc["host"]["capture"]
    report = run_audit(doc)
    assert any(e.status == "MISSING_YAML" for e in report.entries)


def test_config_audit_cli_reports_on_template():
    # Exit codes (0 on a clean file, 1 on gaps) are asserted in test_fix_H_audit.py.
    runner = CliRunner()
    result = runner.invoke(cli_main, ["config", "audit", "--config", str(template_path())])
    assert result.exit_code == 0
    assert "config audit:" in result.output


def test_config_validate_cli_ok():
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["config", "validate", "--config", str(template_path())],
    )
    assert result.exit_code == 0
    assert "OK:" in result.output


def test_config_show_cli():
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["config", "show", "--config", str(template_path())],
    )
    assert result.exit_code == 0
    assert "schema_version" in result.output


def test_config_show_mcu_json():
    runner = CliRunner()
    result = runner.invoke(
        cli_main,
        ["config", "show", "--config", str(template_path()), "--mcu-json"],
    )
    assert result.exit_code == 0
    assert '"emg"' in result.output


def test_sync_config_template_check():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "sync_config_template.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(root),
    )
    assert proc.returncode == 0, proc.stderr
