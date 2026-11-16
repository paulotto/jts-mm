# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Host networking helpers (LAN URLs, etc.)."""

from jtsmm.net.lan_urls import (
    format_ssh_tunnel_hint,
    format_viewer_urls,
    list_lan_addresses,
    print_viewer_startup,
)

__all__ = [
    "format_ssh_tunnel_hint",
    "format_viewer_urls",
    "list_lan_addresses",
    "print_viewer_startup",
]
