# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Search-window math for live detection. No camera and no OpenCV."""

from __future__ import annotations

import numpy as np

from optical.tracking.aruco import (
    MarkerDetection,
    DetectionResult,
    plan_roi,
    roi_window,
    shift_detection,
)


def _marker(x0: float, y0: float, size: float = 40.0, mid: int = 3) -> MarkerDetection:
    return MarkerDetection(
        marker_id=mid,
        corners=np.array(
            [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size]],
            dtype=np.float64,
        ),
    )


def test_roi_window_is_twice_the_marker_box_and_clipped():
    window = roi_window((_marker(100, 80),), (544, 728))
    assert window is not None
    x0, y0, x1, y1 = window
    # 40 px box, 2× → 80 px, centered on the box (center 120, 100).
    assert (x0, y0, x1, y1) == (80, 60, 160, 140)


def test_roi_window_clips_to_the_image():
    window = roi_window((_marker(0, 0, size=30),), (100, 100))
    assert window is not None
    x0, y0, x1, y1 = window
    assert x0 == 0 and y0 == 0
    assert x1 <= 100 and y1 <= 100
    assert x1 - x0 >= 8 and y1 - y0 >= 8


def test_plan_roi_is_full_frame_at_start_every_nth_and_when_lost():
    markers = (_marker(100, 80),)
    shape = (544, 728)
    assert plan_roi(None, shape, frame_index=1) is None
    assert plan_roi(markers, shape, frame_index=0) is None
    assert plan_roi(markers, shape, frame_index=10) is None
    assert plan_roi(markers, shape, frame_index=1) == roi_window(markers, shape)
    assert plan_roi((), shape, frame_index=3) is None


def test_shift_detection_puts_corners_back_in_full_image_pixels():
    det = DetectionResult(markers=(_marker(10, 20),))
    shifted = shift_detection(det, 80, 60)
    np.testing.assert_allclose(shifted.markers[0].corners[0], (90, 80))
    assert shift_detection(det, 0, 0) is det
