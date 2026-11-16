# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Doctor viewer preflight tests."""

from __future__ import annotations

from jtsmm.config.viewer import viewer_settings_from_config
from jtsmm.doctor.viewer import _check_port_bindable, check_viewer_preflight


def test_doctor_preflight_import_chain():
    """doctor CLI imports must not fail (regression for lan_urls syntax)."""
    from jtsmm.doctor import run_preflight  # noqa: F401

    assert callable(run_preflight)


def test_viewer_settings_from_config_defaults():
    cfg = viewer_settings_from_config({})
    assert cfg == {"enabled": False, "bind_host": "0.0.0.0", "port": 8770}


def test_viewer_settings_from_config_yaml():
    cfg = viewer_settings_from_config(
        {"host": {"viewer": {"enabled": True, "bind_host": "0.0.0.0", "port": 9000}}}
    )
    assert cfg["enabled"] is True
    assert cfg["port"] == 9000


def test_port_bindable_high_port():
    issues = _check_port_bindable("127.0.0.1", 28771)
    assert issues == []


def test_check_viewer_preflight_info_lines():
    issues = check_viewer_preflight(
        {"host": {"viewer": {"enabled": True, "bind_host": "127.0.0.1", "port": 28770}}}
    )
    assert any(i.startswith("info:") and "28770" in i for i in issues)
