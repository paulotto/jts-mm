# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Resolve host.viewer settings from YAML."""

from __future__ import annotations

from typing import Any


def viewer_settings_from_config(resolved: dict[str, Any]) -> dict[str, Any]:
    """Return enabled/bind_host/port from ``host.viewer`` with defaults."""
    viewer = (resolved.get("host") or {}).get("viewer") or {}
    return {
        "enabled": bool(viewer.get("enabled", False)),
        "bind_host": str(viewer.get("bind_host", "0.0.0.0")),
        "port": int(viewer.get("port", 8770)),
    }
