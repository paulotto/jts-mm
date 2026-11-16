# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical package configuration (monolithic optical.yaml)."""

from optical.config.loader import (
    calibration_bundle_from_doc,
    calibration_section,
    default_config_path,
    eval_section,
    experiment_section,
    is_monolithic,
    load_config_file,
    markers_section,
    package_etc_dir,
    tracker_section,
)
from optical.config.resolve import (
    detect_picamera_count,
    load_resolved_config,
    resolve_optical_doc,
)

__all__ = [
    "calibration_bundle_from_doc",
    "calibration_section",
    "default_config_path",
    "detect_picamera_count",
    "eval_section",
    "experiment_section",
    "is_monolithic",
    "load_config_file",
    "load_resolved_config",
    "markers_section",
    "package_etc_dir",
    "resolve_optical_doc",
    "tracker_section",
]
