# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""doctor: viewer checks respect host.viewer.enabled; exit code reflects a missing HAT."""

from __future__ import annotations

import socket

from click.testing import CliRunner

import jtsmm.doctor as doctor_mod
import jtsmm.doctor.viewer as viewer_mod
from jtsmm.cli.doctor import doctor_cmd
from jtsmm.doctor.preflight_issues import partition_preflight_issues


def _busy_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, s.getsockname()[1]


def _cfg(enabled, port):
    return {"host": {"viewer": {"enabled": enabled, "bind_host": "127.0.0.1", "port": port}}}


def test_disabled_viewer_only_warns_about_missing_deps_and_busy_port(monkeypatch):
    monkeypatch.setattr(viewer_mod, "_check_viewer_deps", lambda: ["viewer extra not installed"])
    sock, port = _busy_port()
    try:
        errors, warnings, _ = partition_preflight_issues(
            viewer_mod.check_viewer_preflight(_cfg(False, port))
        )
        assert errors == []
        assert any("not installed" in w for w in warnings)
        assert any(str(port) in w for w in warnings)

        errors, _, _ = partition_preflight_issues(
            viewer_mod.check_viewer_preflight(_cfg(True, port))
        )
        assert any("not installed" in e for e in errors)
        assert any(str(port) in e for e in errors)
    finally:
        sock.close()


def test_missing_hat_is_an_error_only_when_required(monkeypatch):
    monkeypatch.setattr(doctor_mod, "probe_port", lambda _p: False)
    cfg = _cfg(False, 28790)
    errs, warns, _ = partition_preflight_issues(
        doctor_mod.run_preflight(cfg, check_serial=True, check_firmware=False)
    )
    assert not any("not detected" in e for e in errs)          # record preflight: warn
    assert any("not detected" in w for w in warns)
    errs, _, _ = partition_preflight_issues(
        doctor_mod.run_preflight(cfg, check_serial=True, check_firmware=False, require_hat=True)
    )
    assert any("not detected" in e for e in errs)


def test_bare_doctor_exits_nonzero_without_a_hat(monkeypatch):
    monkeypatch.setattr(doctor_mod, "probe_port", lambda _p: False)
    monkeypatch.setattr(doctor_mod, "validate_config", lambda _r: [])
    monkeypatch.setattr(doctor_mod, "check_imx296_trigger_mode", lambda _r: [])
    monkeypatch.setattr(doctor_mod, "check_viewer_preflight", lambda _r: [])
    result = CliRunner().invoke(doctor_cmd, [])
    assert result.exit_code == 1, result.output
    result = CliRunner().invoke(doctor_cmd, ["--no-serial"])
    assert result.exit_code == 0, result.output
