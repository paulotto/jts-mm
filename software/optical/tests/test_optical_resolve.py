# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for optical.yaml profile resolution (1 vs 2 cameras)."""

from __future__ import annotations

from optical.config.loader import default_config_path, load_config_file, tracker_section
from optical.config.resolve import resolve_optical_doc
from optical.tracking.tracker import TrackerConfig


def test_resolve_1cam_profile():
    path = default_config_path()
    doc = load_config_file(path)
    resolved = resolve_optical_doc(doc, n_cameras=1, quiet=True)
    t = tracker_section(resolved)
    assert t["pose_method"] == "pnp"
    assert t["cam_ids"] == ["cam0"]
    assert len(resolved["calibration"]["cameras"]) == 1
    assert resolved["paths"]["calibration_bundle"].endswith("calibration_bundle.h5")
    assert "profiles" not in resolved


def test_resolve_2cam_profile():
    path = default_config_path()
    doc = load_config_file(path)
    resolved = resolve_optical_doc(doc, n_cameras=2, quiet=True)
    t = tracker_section(resolved)
    assert t["pose_method"] == "triangulation"
    assert t["cam_ids"] == ["cam0", "cam1"]
    assert len(resolved["calibration"]["cameras"]) == 2
    assert resolved["paths"]["calibration_bundle"].endswith("calibration_bundle_2cams.h5")


def test_profile_overrides_board_geometry():
    """Board geometry declared in a profile wins over the shared defaults."""
    doc = load_config_file(default_config_path())
    doc = {**doc, "profiles": {**doc["profiles"]}}
    doc["profiles"]["2"] = {
        **doc["profiles"]["2"],
        "calibration": {"board": {"square_length_mm": 34.5, "marker_length_mm": 25.0}},
    }
    resolved = resolve_optical_doc(doc, n_cameras=2, quiet=True)
    board = resolved["calibration"]["board"]
    assert board["square_length_mm"] == 34.5
    assert board["marker_length_mm"] == 25.0
    # Keys absent from the profile still come from the shared board block.
    assert board["squares_x"] == doc["calibration"]["board"]["squares_x"]
    assert board["dictionary"] == doc["calibration"]["board"]["dictionary"]


def test_resolve_active_cam_ids_overrides_count():
    path = default_config_path()
    doc = load_config_file(path)
    resolved = resolve_optical_doc(doc, active_cam_ids=["cam0"], quiet=True)
    t = tracker_section(resolved)
    assert t["cam_ids"] == ["cam0"]
    assert t["pose_method"] == "pnp"


def test_tracker_config_from_path_with_active_cams():
    path = default_config_path()
    one = TrackerConfig.from_path(path, active_cam_ids=["cam0"])
    two = TrackerConfig.from_path(path, active_cam_ids=["cam0", "cam1"])
    assert one.pose_method == "pnp"
    assert one.cam_ids == ["cam0"]
    assert two.pose_method == "triangulation"
    assert two.cam_ids == ["cam0", "cam1"]
