# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer preflight checks for jtsmm doctor."""

from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any

from jtsmm.net.lan_urls import format_viewer_urls, list_lan_addresses

WIFI_STATE_FILE = Path("/var/lib/jtsmm/wifi_disabled_until")


def _viewer_config(resolved: dict[str, Any] | None) -> dict[str, Any]:
    host = (resolved or {}).get("host") or {}
    viewer = host.get("viewer") or {}
    return {
        "enabled": bool(viewer.get("enabled", False)),
        "bind_host": str(viewer.get("bind_host", "0.0.0.0")),
        "port": int(viewer.get("port", 8770)),
    }


def _check_viewer_deps() -> list[str]:
    issues: list[str] = []
    missing: list[str] = []
    for mod in ("fastapi", "uvicorn", "msgpack"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        issues.append(
            "viewer extra not installed (missing: "
            + ", ".join(missing)
            + "); install with: pip install 'jtsmm[viewer]'"
        )
    return issues


def _check_port_bindable(host: str, port: int) -> list[str]:
    bind_host = host if host not in ("", "::") else "0.0.0.0"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
    except OSError as exc:
        return [f"viewer port {port} not bindable on {bind_host}: {exc}"]
    return []


def _check_ufw(port: int) -> list[str]:
    try:
        proc = subprocess.run(
            ["ufw", "status"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0 or "Status: active" not in proc.stdout:
        return []
    if f"{port}/tcp" not in proc.stdout and f"{port} " not in proc.stdout:
        return [f"warn: ufw is active but port {port}/tcp may not be allowed"]
    return []


def _check_wifi_state() -> list[str]:
    issues: list[str] = []
    if WIFI_STATE_FILE.is_file():
        try:
            raw = WIFI_STATE_FILE.read_text(encoding="utf-8").strip()
            until = int(raw.split()[0]) if raw else 0
            if until > 0:
                import time

                now = int(time.time())
                if until > now:
                    remaining = until - now
                    mins = remaining // 60
                    issues.append(
                        f"warn: Wi-Fi disabled until epoch {until} (~{mins} min remaining)"
                    )
        except (OSError, ValueError):
            issues.append("warn: could not read Wi-Fi disable state file")
    try:
        proc = subprocess.run(
            ["nmcli", "-t", "-f", "WIFI", "radio"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and "enabled" in proc.stdout.lower():
            issues.append("info: viewer Wi-Fi radio enabled")
        elif proc.returncode == 0:
            issues.append("info: viewer Wi-Fi radio disabled")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return issues


def check_viewer_preflight(resolved: dict[str, Any] | None = None) -> list[str]:
    """Return viewer subsection issues for jtsmm doctor (warnings prefixed warn:).

    Missing viewer deps or a busy port are errors only when ``host.viewer.enabled``
    is true. With the viewer disabled they are warnings: they used to block
    `jtsmm record` preflight on a rig that never starts the viewer.
    """
    cfg = _viewer_config(resolved)
    issues: list[str] = []
    blocking = _check_viewer_deps() + _check_port_bindable(cfg["bind_host"], cfg["port"])
    if not cfg["enabled"]:
        return [f"warn: {msg} (viewer disabled)" for msg in blocking]
    issues.extend(blocking)
    issues.extend(_check_ufw(cfg["port"]))
    issues.extend(_check_wifi_state())

    urls = format_viewer_urls(cfg["bind_host"], cfg["port"], mdns_hostname="jtsmm")
    if urls:
        issues.append("info: viewer local URL " + urls[0])
    for u in urls[1:]:
        issues.append("info: viewer LAN URL " + u)
    if list_lan_addresses():
        issues.append("info: viewer LAN addresses " + ", ".join(list_lan_addresses()))
    return issues
