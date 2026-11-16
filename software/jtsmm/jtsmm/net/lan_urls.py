# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""LAN URL helpers for the live viewer (stdlib-only)."""

from __future__ import annotations

import socket
import sys
from typing import TextIO


def list_lan_addresses() -> list[str]:
    """Return non-loopback IPv4 addresses on local interfaces."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    return sorted(addrs)


def format_viewer_urls(
    host: str,
    port: int,
    *,
    mdns_hostname: str | None = None,
) -> list[str]:
    """Human-readable viewer URLs for local and LAN access."""
    urls: list[str] = []
    local_host = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    urls.append(f"http://{local_host}:{port}/")
    if host in ("0.0.0.0", "", "::"):
        for ip in list_lan_addresses():
            urls.append(f"http://{ip}:{port}/")
    if mdns_hostname:
        urls.append(f"http://{mdns_hostname}.local:{port}/")
    return urls


def format_ssh_tunnel_hint(port: int, *, remote_user: str = "pi") -> str:
    """SSH port-forward command for blocked direct LAN access."""
    addrs = list_lan_addresses()
    target = addrs[0] if addrs else "jtsmm.local"
    return f"ssh -L {port}:127.0.0.1:{port} {remote_user}@{target}"


def print_viewer_startup(
    host: str,
    port: int,
    *,
    stream: TextIO | None = None,
    mdns_hostname: str | None = "jtsmm",
    prefix: str = "viewer",
) -> None:
    """Print bind address, local/LAN URLs, and SSH tunnel hint to stderr."""
    out = stream or sys.stderr
    bind = f"{host}:{port}" if host not in ("", "::") else f"0.0.0.0:{port}"
    print(f"{prefix}: bound {bind}", file=out)
    urls = format_viewer_urls(host, port, mdns_hostname=mdns_hostname)
    if urls:
        print(f"{prefix}: open on this machine:  {urls[0]}", file=out)
    lan = [u for u in urls[1:] if ".local" not in u]
    mdns = [u for u in urls if ".local" in u]
    for u in lan:
        print(f"{prefix}: open from your laptop: {u}", file=out)
    for u in mdns:
        print(f"{prefix}: mDNS: {u}", file=out)
    print(f"{prefix}: SSH tunnel: {format_ssh_tunnel_hint(port)}", file=out)
