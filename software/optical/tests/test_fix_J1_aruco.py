# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""ArUco detector: sub-pixel corner refinement and gray input (needs cv2)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from optical.markers.config import get_aruco_dictionary  # noqa: E402
from optical.tracking.aruco import (  # noqa: E402
    ArucoDetector,
    make_detector_parameters,
    roi_window,
    shift_detection,
)


def _rendered_tag(*, marker_id: int = 3, scale: int = 8, seed: int = 0):
    """Tag warped at 8x with a sub-pixel homography, then area-downsampled.

    Returns (gray, true_corners) where the corners are exact in the output
    pixel grid (pixel centres at integer coordinates, OpenCV convention).
    """
    rng = np.random.default_rng(seed)
    side = 40  # px in the output image: close to a 30 mm tag at 600 mm on lores
    w, h = 200, 160
    d = get_aruco_dictionary("DICT_4X4_50")
    tag_hi = cv2.aruco.generateImageMarker(d, marker_id, side * scale)
    src = np.array(
        [[0, 0], [side * scale, 0], [side * scale, side * scale], [0, side * scale]],
        dtype=np.float64,
    ) - 0.5
    ang = np.radians(rng.uniform(-20, 20))
    c, s = np.cos(ang), np.sin(ang)
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64) * side / 2
    base = base @ np.array([[c, -s], [s, c]]).T
    base *= np.array([1.0, rng.uniform(0.8, 1.0)])  # mild foreshortening
    dst = base + np.array([w / 2 + rng.uniform(-0.5, 0.5), h / 2 + rng.uniform(-0.5, 0.5)])
    dst_hi = (dst + 0.5) * scale - 0.5
    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst_hi.astype(np.float32))
    canvas = cv2.warpPerspective(
        tag_hi, H, (w * scale, h * scale), flags=cv2.INTER_LINEAR, borderValue=255
    )
    gray = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return gray, dst


def _corner_error(det, true_corners) -> float:
    assert len(det.markers) == 1
    return float(np.mean(np.linalg.norm(det.markers[0].corners - true_corners, axis=1)))


def test_detector_defaults_to_subpix():
    params = make_detector_parameters()
    assert params.cornerRefinementMethod == cv2.aruco.CORNER_REFINE_SUBPIX
    assert params.cornerRefinementWinSize == 3
    params = make_detector_parameters(corner_refinement="subpix", corner_refine_win_px=5)
    assert params.cornerRefinementWinSize == 5
    assert (
        make_detector_parameters(corner_refinement="none").cornerRefinementMethod
        == cv2.aruco.CORNER_REFINE_NONE
    )
    with pytest.raises(ValueError):
        make_detector_parameters(corner_refinement="bogus")


def test_subpix_corners_are_more_accurate_than_unrefined():
    refined = ArucoDetector("DICT_4X4_50")
    raw = ArucoDetector("DICT_4X4_50", corner_refinement="none")
    err_ref, err_raw = [], []
    for seed in range(12):
        gray, truth = _rendered_tag(seed=seed)
        err_ref.append(_corner_error(refined.detect(gray), truth))
        err_raw.append(_corner_error(raw.detect(gray), truth))
    assert np.mean(err_ref) < 0.35
    assert np.mean(err_ref) < 0.7 * np.mean(err_raw)


def test_detect_rgb_accepts_gray_and_single_channel():
    det = ArucoDetector("DICT_4X4_50")
    gray, _ = _rendered_tag(marker_id=7)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for img in (gray, gray[:, :, None], rgb):
        out = det.detect_rgb(img)
        assert [m.marker_id for m in out.markers] == [7]
    assert det.detect_rgb(np.zeros((0, 0), np.uint8)).markers == ()


def test_crop_detect_matches_full_frame_after_shift():
    det = ArucoDetector("DICT_4X4_50", corner_refinement="none")
    gray, _ = _rendered_tag(marker_id=3)
    canvas = np.full((400, 500), 255, np.uint8)
    y, x = 100, 150
    canvas[y : y + gray.shape[0], x : x + gray.shape[1]] = gray
    full = det.detect(canvas)
    assert len(full.markers) == 1
    window = roi_window(full.markers, canvas.shape)
    assert window is not None
    x0, y0, x1, y1 = window
    crop = det.detect(np.ascontiguousarray(canvas[y0:y1, x0:x1]))
    shifted = shift_detection(crop, x0, y0)
    assert len(shifted.markers) == 1
    assert shifted.markers[0].marker_id == full.markers[0].marker_id
    err = np.mean(np.linalg.norm(shifted.markers[0].corners - full.markers[0].corners, axis=1))
    assert err < 1.5
