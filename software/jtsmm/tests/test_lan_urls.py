# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for LAN URL helpers."""

from __future__ import annotations

import jtsmm.net.lan_urls as lan_urls
from jtsmm.net import format_ssh_tunnel_hint, format_viewer_urls, list_lan_addresses


def test_net_package_imports():
    """Package __init__ re-exports must import without syntax errors."""
    assert callable(lan_urls.print_viewer_startup)
    assert callable(format_viewer_urls)


def test_format_viewer_urls_localhost():
    urls = format_viewer_urls("127.0.0.1", 8770, mdns_hostname="jtsmm")
    assert urls == ["http://127.0.0.1:8770/", "http://jtsmm.local:8770/"]


def test_format_viewer_urls_all_interfaces():
    urls = format_viewer_urls("0.0.0.0", 8770, mdns_hostname="jtsmm")
    assert urls[0] == "http://127.0.0.1:8770/"
    assert any("jtsmm.local" in u for u in urls)


def test_list_lan_addresses_returns_list():
    addrs = list_lan_addresses()
    assert isinstance(addrs, list)
    assert all("." in a for a in addrs)


def test_ssh_tunnel_hint():
    hint = format_ssh_tunnel_hint(8770)
    assert "ssh -L 8770:127.0.0.1:8770" in hint
