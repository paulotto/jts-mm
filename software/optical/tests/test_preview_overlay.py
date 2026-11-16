# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Preview ArUco overlay helpers."""

from __future__ import annotations

import numpy as np
import pytest

from optical.tracking.aruco import DetectionResult, MarkerDetection
from optical.tracking.preview_overlay import (
    draw_marker_boxes_bgr,
    markers_from_detection,
    overlay_frames_bgr,
)


def test_markers_from_detection_empty():
    assert markers_from_detection(None) == []
    assert markers_from_detection(DetectionResult(markers=())) == []


def test_draw_marker_boxes_bgr():
    pytest.importorskip("cv2")
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    corners = np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.float64)
    out = draw_marker_boxes_bgr(img, [(7, corners)])
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_overlay_frames_bgr_gray():
    pytest.importorskip("cv2")
    gray = np.zeros((60, 80), dtype=np.uint8)
    corners = np.array([[5, 5], [25, 5], [25, 25], [5, 25]], dtype=np.float64)
    det = DetectionResult(
        markers=(MarkerDetection(marker_id=3, corners=corners),)
    )

    def as_bgr(img):
        import cv2

        a = np.asarray(img)
        if a.ndim == 2:
            return cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        return a

    out = overlay_frames_bgr({"cam0": gray}, {"cam0": det}, as_bgr=as_bgr)
    assert out["cam0"].ndim == 3
    assert out["cam0"].shape[2] == 3
