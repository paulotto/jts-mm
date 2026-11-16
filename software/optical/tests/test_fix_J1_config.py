# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Marker / tracker config validation (pure numpy; runs without cv2)."""

from __future__ import annotations

import numpy as np
import pytest

from optical.config.loader import package_etc_dir
from optical.markers.config import MarkerConfig, load_marker_config, validate_marker_config
from optical.tracking.tracker import TrackerConfig


def _cfg(**kw) -> MarkerConfig:
    base = dict(
        dictionary="DICT_4X4_50",
        tag_size_mm=30.0,
        id_ranges={"HP": (0, 9), "MP": (10, 19), "CT": (20, 29)},
        mounts_T_body_tag={1: np.eye(4), 13: np.eye(4)},
    )
    base.update(kw)
    return MarkerConfig(**base)


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def test_shipped_optical_yaml_is_valid():
    cfg = load_marker_config(package_etc_dir() / "optical.yaml")
    assert validate_marker_config(cfg) == []
    assert cfg.corner_refinement == "subpix"
    assert cfg.corner_refine_win_px == 3


def test_valid_config_has_no_errors():
    assert validate_marker_config(_cfg(mounts_T_body_tag={1: _rot_z(37.0), 22: np.eye(4)})) == []


def test_overlapping_id_ranges_rejected():
    errs = validate_marker_config(_cfg(id_ranges={"HP": (0, 10), "MP": (10, 19), "CT": (20, 29)}))
    assert any("overlaps" in e for e in errs)


def test_non_orthonormal_and_reflected_mounts_rejected():
    scaled = np.eye(4)
    scaled[:3, :3] *= 1.01
    mirrored = np.diag([1.0, 1.0, -1.0, 1.0])
    errs = validate_marker_config(_cfg(mounts_T_body_tag={1: scaled, 13: mirrored}))
    assert sum("orthonormal" in e for e in errs) == 2


def test_mount_outside_ranges_rejected():
    errs = validate_marker_config(_cfg(mounts_T_body_tag={1: np.eye(4), 42: np.eye(4)}))
    assert any("[42]" in e and "outside" in e for e in errs)


def test_bad_corner_refinement_rejected():
    errs = validate_marker_config(_cfg(corner_refinement="bogus", corner_refine_win_px=0))
    assert len(errs) == 2


def _write_yaml(tmp_path, tracker: str):
    p = tmp_path / "optical.yaml"
    p.write_text(
        "paths:\n  calibration_bundle: /tmp/none.h5\n"
        "markers:\n  detector: {dictionary: DICT_4X4_50, tag_size_mm: 30}\n"
        "  id_ranges: {HP: [0, 9]}\n"
        f"tracker:\n{tracker}"
    )
    return p


def test_pose_method_typo_rejected(tmp_path):
    with pytest.raises(ValueError, match="pose_method"):
        TrackerConfig.from_path(_write_yaml(tmp_path, "  pose_method: triangualtion\n"))


def test_pose_method_normalised(tmp_path):
    cfg = TrackerConfig.from_path(_write_yaml(tmp_path, "  pose_method: ' PnP '\n"))
    assert cfg.pose_method == "pnp"


def test_min_views_below_two_rejected(tmp_path):
    with pytest.raises(ValueError, match="min_views"):
        TrackerConfig.from_path(
            _write_yaml(tmp_path, "  pose_method: triangulation\n  triangulation: {min_views: 1}\n")
        )
