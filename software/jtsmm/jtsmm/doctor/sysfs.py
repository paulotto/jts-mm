# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Host sysfs / device preflight probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def check_imx296_trigger_mode(resolved: dict[str, Any]) -> list[str]:
    """Verify IMX296 external trigger mode when Track B cameras are configured."""
    host = resolved.get("host") or {}
    cameras = host.get("cameras") or {}
    if cameras.get("track") != "B":
        return []
    cam_list = cameras.get("list") or []
    if not cam_list:
        return []

    path = Path("/sys/module/imx296/parameters/trigger_mode")
    if not path.is_file():
        return ["warn: imx296 trigger_mode sysfs not present (camera module not loaded?)"]

    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        return [f"warn: cannot read imx296 trigger_mode: {exc}"]

    if value != "1":
        return [
            f"error: imx296 trigger_mode={value!r}, expected '1' for Track B external trigger "
            "(run: echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode)"
        ]
    return []
