# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Marker layout and printable fiducials (doc 01).

See ``docs/optical/01_marker_setup.md``.
"""

from optical.markers.config import (
    MarkerConfig,
    generate_charuco_board_png,
    generate_marker_png,
    get_aruco_dictionary,
    load_marker_config,
    make_charuco_board,
    validate_marker_config,
)

__all__ = [
    "MarkerConfig",
    "generate_charuco_board_png",
    "generate_marker_png",
    "get_aruco_dictionary",
    "load_marker_config",
    "make_charuco_board",
    "validate_marker_config",
]
