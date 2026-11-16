# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ArUco marker overlay for viewer camera previews (optical camera_test pattern)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "draw_aruco_overlay_bgr",
    "draw_aruco_overlay_rgb",
    "draw_marker_boxes_bgr",
    "draw_marker_boxes_rgb",
]

MarkerBoxes = Iterable[tuple[int, np.ndarray]]


class _ArucoDetector:
    """Lazy OpenCV ArUco detector — DICT_4X4_50, same as optical camera_test."""

    _detector = None

    # Same sub-pixel corner refinement as the optical tracker, so the viewer
    # boxes sit where the tracker measures.
    _REFINE_WIN_PX = 3

    @classmethod
    def _get(cls):
        if cls._detector is None:
            from cv2 import aruco

            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            params = aruco.DetectorParameters()
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = cls._REFINE_WIN_PX
            cls._detector = aruco.ArucoDetector(dictionary, params)
        return cls._detector

    @classmethod
    def detect_markers(cls, bgr_image: np.ndarray):
        if bgr_image is None or bgr_image.size == 0:
            return None
        import cv2

        arr = np.asarray(bgr_image)
        if arr.ndim == 2:
            gray = arr
        elif arr.ndim == 3 and arr.shape[2] == 1:
            gray = arr[:, :, 0]
        else:
            gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cls._get().detectMarkers(gray)
        if ids is None:
            return None
        return corners, ids, rejected


def _scale_corners(
    corners: np.ndarray,
    *,
    dst_wh: tuple[int, int],
    src_wh: tuple[int, int] | None,
) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if src_wh is None:
        return pts
    sw, sh = src_wh
    dw, dh = dst_wh
    if sw <= 0 or sh <= 0 or (sw == dw and sh == dh):
        return pts
    return pts * np.array([dw / float(sw), dh / float(sh)], dtype=np.float64)


def _paint_boxes(
    image: np.ndarray,
    markers: MarkerBoxes,
    *,
    src_wh: tuple[int, int] | None,
    color: Sequence[int],
) -> np.ndarray:
    import cv2

    result = image.copy()
    h, w = result.shape[:2]
    bgr_color = (int(color[0]), int(color[1]), int(color[2]))
    for marker_id, corners in markers:
        pts = _scale_corners(corners, dst_wh=(w, h), src_wh=src_wh).astype(int)
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), bgr_color, 2)
        cv2.putText(
            result,
            str(int(marker_id)),
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bgr_color,
            1,
            cv2.LINE_AA,
        )
    return result


def draw_marker_boxes_bgr(
    bgr_image: np.ndarray,
    markers: MarkerBoxes,
    *,
    src_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw green boxes from already-detected corners (no ArUco pass)."""
    return _paint_boxes(bgr_image, markers, src_wh=src_wh, color=(0, 255, 0))


def draw_marker_boxes_rgb(
    rgb_image: np.ndarray,
    markers: MarkerBoxes,
    *,
    src_wh: tuple[int, int] | None = None,
) -> np.ndarray:
    """Gray or RGB in → RGB out using tracker corners."""
    import cv2

    arr = np.asarray(rgb_image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    else:
        arr = arr[:, :, :3]
    return _paint_boxes(arr, markers, src_wh=src_wh, color=(0, 255, 0))


def draw_aruco_overlay_bgr(bgr_image: np.ndarray) -> np.ndarray:
    """Return a copy with green bounding boxes + IDs (optical camera_test.draw_overlay)."""
    detection = _ArucoDetector.detect_markers(bgr_image)
    if detection is None:
        return bgr_image.copy()

    corners, ids, _ = detection
    markers = [
        (int(marker_id), np.asarray(corner[0]))
        for corner, marker_id in zip(corners, ids.flatten())
    ]
    return draw_marker_boxes_bgr(bgr_image, markers)


def draw_aruco_overlay_rgb(rgb_image: np.ndarray) -> np.ndarray:
    """RGB (or gray) in → RGB out with ArUco boxes drawn (detects; viewer-only path)."""
    import cv2

    arr = np.asarray(rgb_image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return arr
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    over = draw_aruco_overlay_bgr(bgr)
    return cv2.cvtColor(over, cv2.COLOR_BGR2RGB)
