# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer ArUco overlay: sub-pixel refinement and gray frames (needs cv2)."""

from __future__ import annotations

import numpy as np
import pytest


def _gray_with_tag() -> np.ndarray:
    cv2 = pytest.importorskip("cv2")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    tag = cv2.aruco.generateImageMarker(dictionary, 7, 120)
    canvas = np.full((240, 320), 255, dtype=np.uint8)
    canvas[60:180, 100:220] = tag
    return canvas


def test_overlay_detector_uses_subpix():
    cv2 = pytest.importorskip("cv2")
    from jtsmm.viewer.aruco_overlay import _ArucoDetector

    _ArucoDetector._detector = None
    det = _ArucoDetector._get()
    params = det.getDetectorParameters()
    assert params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX


def test_overlay_accepts_gray_frames():
    pytest.importorskip("cv2")
    from jtsmm.viewer.aruco_overlay import _ArucoDetector, draw_aruco_overlay_rgb

    gray = _gray_with_tag()
    found = _ArucoDetector.detect_markers(gray)
    assert found is not None and [int(i) for i in found[1].ravel()] == [7]
    out = draw_aruco_overlay_rgb(gray)
    assert out.shape == (240, 320, 3)
    assert np.any(out[:, :, 1] > out[:, :, 0])
