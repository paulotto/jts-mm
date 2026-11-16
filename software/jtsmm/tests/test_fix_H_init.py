# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""`jtsmm config init` creates the user's file; the template fallback is announced."""

from __future__ import annotations

from click.testing import CliRunner

from jtsmm.cli.main import main as cli_main
from jtsmm.config import loader


def test_init_writes_home_config_then_refuses_then_forces(_isolate_home):
    runner = CliRunner()
    dst = _isolate_home / "jtsmm.yaml"
    assert not dst.exists()
    res = runner.invoke(cli_main, ["config", "init"])
    assert res.exit_code == 0, res.output
    assert f"wrote {dst}" in res.output
    assert dst.read_bytes() == loader.template_path().read_bytes()

    res = runner.invoke(cli_main, ["config", "init"])
    assert res.exit_code == 0 and f"exists: {dst}" in res.output

    dst.write_text("edited: true\n", encoding="utf-8")
    res = runner.invoke(cli_main, ["config", "init", "--force"])
    assert res.exit_code == 0, res.output
    assert dst.read_bytes() == loader.template_path().read_bytes()
    # A checkout loads software/jtsmm/etc/jtsmm.yaml, not the home copy init wrote.
    assert loader.default_config_path() == loader.template_path()


def test_init_honours_jtsmm_config(monkeypatch, tmp_path):
    target = tmp_path / "rig" / "cfg.yaml"
    monkeypatch.setenv("JTSMM_CONFIG", str(target))
    res = CliRunner().invoke(cli_main, ["config", "init"])
    assert res.exit_code == 0, res.output
    assert target.is_file()
    assert loader.default_config_path() == target


def test_init_refuses_to_overwrite_the_template():
    res = CliRunner().invoke(
        cli_main, ["config", "init", "--force", "--dest", str(loader.template_path())]
    )
    assert res.exit_code != 0
    assert "shipped template itself" in res.output


def test_template_fallback_warns_once(monkeypatch, capsys, tmp_path):
    """Wheel install, no operator file: packaged template, announced once."""
    packaged = tmp_path / "etc" / "jtsmm.yaml"
    packaged.parent.mkdir()
    packaged.write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr(loader, "_TEMPLATE_WARNED", False)
    monkeypatch.setattr(loader, "checkout_root", lambda: None)
    monkeypatch.setattr(loader, "template_path", lambda: packaged)
    monkeypatch.setattr(loader, "_LEGACY_CONFIG_PATH", loader.user_config_path().parent / "none")
    assert loader.default_config_path() == packaged
    loader.default_config_path()
    err = capsys.readouterr().err
    assert err.count("warning: no ") == 1
    assert "jtsmm config init" in err


def test_checkout_template_beats_home_config(_isolate_home, monkeypatch, capsys):
    monkeypatch.setattr(loader, "_TEMPLATE_WARNED", False)
    (_isolate_home / "jtsmm.yaml").write_text("x: 1\n", encoding="utf-8")
    assert loader.default_config_path() == loader.template_path()
    assert "warning" not in capsys.readouterr().err
