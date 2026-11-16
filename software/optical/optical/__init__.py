# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Optical: ArUco camera tracking to replace Qualisys jaw tracking.

Subpackages (code):

- :mod:`optical.markers` — doc 01: marker layout and printable fiducials
- :mod:`optical.calibration` — doc 02: multi-camera calibration bundles
- :mod:`optical.tracking` — doc 03: pose estimation and JTS output

Guides: ``docs/optical/``. Config: ``software/optical/etc/optical.yaml``. Artifacts: ``data/optical/``.
"""

from optical.calibration import (
    CameraExtrinsics,
    CameraIntrinsics,
    read_calibration_bundle,
    write_calibration_bundle,
)
from optical.markers import MarkerConfig, load_marker_config
from optical.tracking import ArucoJawTracker, FrameRecord, TrackerConfig

__all__ = [
    "ArucoJawTracker",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "FrameRecord",
    "MarkerConfig",
    "TrackerConfig",
    "load_marker_config",
    "read_calibration_bundle",
    "write_calibration_bundle",
]
