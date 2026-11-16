# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm ct-calibrate --port X` must preflight port X, not hat.serial_port."""

from __future__ import annotations

from click.testing import CliRunner

from jtsmm.cli.ct_cmd import ct_calibrate_cmd


def test_preflight_probes_the_port_override(monkeypatch, tmp_config_path):
    seen: dict = {}

    def fake_preflight(resolved=None, **kwargs):
        seen["port"] = (resolved or {}).get("hat", {}).get("serial_port")
        return ["stop here"]  # an error: the command exits before touching hardware

    monkeypatch.setattr("jtsmm.cli.ct_cmd.run_preflight", fake_preflight)
    res = CliRunner().invoke(
        ct_calibrate_cmd, ["--config", str(tmp_config_path), "--port", "/dev/ttyACM7"]
    )
    assert res.exit_code == 1, res.output
    assert seen["port"] == "/dev/ttyACM7"


def test_preflight_uses_config_port_without_override(monkeypatch, tmp_config_path):
    seen: dict = {}

    def fake_preflight(resolved=None, **kwargs):
        seen["port"] = (resolved or {}).get("hat", {}).get("serial_port")
        return ["stop here"]

    monkeypatch.setattr("jtsmm.cli.ct_cmd.run_preflight", fake_preflight)
    CliRunner().invoke(ct_calibrate_cmd, ["--config", str(tmp_config_path)])
    assert seen["port"] == "/dev/jtsmm_hat"
