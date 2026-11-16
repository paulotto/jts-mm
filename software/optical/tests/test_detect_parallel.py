# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Parallel stereo ArUco detect matches sequential results."""

from __future__ import annotations

import numpy as np
import pytest

from optical.markers.config import get_aruco_dictionary
from optical.tracking.aruco import ArucoDetector, detect_frames

cv2 = pytest.importorskip("cv2")


def _synthetic_marker_rgb(*, marker_id: int, canvas_wh: tuple[int, int] = (320, 240), side_px: int = 80):
    w, h = canvas_wh
    d = get_aruco_dictionary("DICT_4X4_50")
    tag = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(d, marker_id, side_px, tag, 1)
    gray = np.full((h, w), 220, dtype=np.uint8)
    x0 = (w - side_px) // 2
    y0 = (h - side_px) // 2
    gray[y0 : y0 + side_px, x0 : x0 + side_px] = tag
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_detect_frames_parallel_matches_sequential():
    detector = ArucoDetector("DICT_4X4_50")
    frames = {
        "cam0": _synthetic_marker_rgb(marker_id=2),
        "cam1": _synthetic_marker_rgb(marker_id=7),
    }
    sequential = detect_frames(detector, frames, parallel=False)
    parallel = detect_frames(detector, frames, parallel=True)
    for cam_id in frames:
        seq_ids = [m.marker_id for m in sequential[cam_id].markers]
        par_ids = [m.marker_id for m in parallel[cam_id].markers]
        assert seq_ids == par_ids
        assert seq_ids, f"expected a marker in {cam_id}"
    assert sequential["cam0"].markers[0].marker_id == 2
    assert sequential["cam1"].markers[0].marker_id == 7


def test_detect_frames_single_camera_skips_pool():
    detector = ArucoDetector("DICT_4X4_50")
    frames = {"cam0": _synthetic_marker_rgb(marker_id=3)}
    out = detect_frames(detector, frames, parallel=True)
    assert [m.marker_id for m in out["cam0"].markers] == [3]


def test_detect_frames_gray_matches_rgb():
    detector = ArucoDetector("DICT_4X4_50")
    rgb = _synthetic_marker_rgb(marker_id=5)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    from_rgb = detect_frames(detector, {"cam0": rgb}, parallel=False)["cam0"]
    from_gray = detect_frames(detector, {"cam0": gray}, parallel=False)["cam0"]
    assert [m.marker_id for m in from_rgb.markers] == [5]
    assert [m.marker_id for m in from_gray.markers] == [5]
