# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
ArUco marker detection (IDs + corners) for the jaw-tracking pipeline.

See ``docs/optical/03_pose_estimation_and_output.md`` §1.1.
"""

from __future__ import annotations

import atexit
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import numpy as np

from optical.markers.config import (
    DEFAULT_CORNER_REFINEMENT,
    DEFAULT_CORNER_REFINE_WIN_PX,
    get_aruco_dictionary,
)

# Shared pool for stereo detect. OpenCV's detectMarkers typically releases the
# GIL, so two cameras can run on two threads instead of back-to-back.
_DETECT_POOL: ThreadPoolExecutor | None = None
_DETECT_POOL_LOCK = threading.Lock()


def _detect_pool() -> ThreadPoolExecutor:
    global _DETECT_POOL
    with _DETECT_POOL_LOCK:
        if _DETECT_POOL is None:
            _DETECT_POOL = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="aruco-detect"
            )
            atexit.register(_DETECT_POOL.shutdown, wait=False)
        return _DETECT_POOL


@dataclass(frozen=True)
class MarkerDetection:
    """One detected tag in a single camera frame."""

    marker_id: int
    corners: np.ndarray  # (4, 2) float64 image pixels


@dataclass(frozen=True)
class DetectionResult:
    """All markers detected in one grayscale frame."""

    markers: Tuple[MarkerDetection, ...]

    @property
    def ids(self) -> np.ndarray:
        return np.array([m.marker_id for m in self.markers], dtype=np.int32)

    @property
    def corner_arrays(self) -> List[np.ndarray]:
        """OpenCV-style corner list for drawing / legacy APIs."""
        return [m.corners.reshape(1, 4, 2).astype(np.float32) for m in self.markers]


def make_detector_parameters(
    *,
    corner_refinement: str = DEFAULT_CORNER_REFINEMENT,
    corner_refine_win_px: int = DEFAULT_CORNER_REFINE_WIN_PX,
):
    """``cv2.aruco.DetectorParameters`` with the tracker's corner refinement.

    OpenCV's default is no refinement: corners stay on the integer-ish contour
    fit, which costs ~0.3–0.5 px of corner noise on half-res lores frames and
    feeds straight into PnP / triangulation. ``subpix`` runs ``cornerSubPix`` in
    a ``(2·win+1)²`` window; keep ``win`` below one tag cell (a 30 px tag has
    ~5 px cells) so the search cannot jump to an inner bit corner.
    """
    import cv2

    params = cv2.aruco.DetectorParameters()
    mode = str(corner_refinement or "none").strip().lower()
    methods = {
        "none": cv2.aruco.CORNER_REFINE_NONE,
        "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
        "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    }
    if mode not in methods:
        raise ValueError(
            f"corner_refinement must be one of {sorted(methods)}, got {corner_refinement!r}"
        )
    params.cornerRefinementMethod = methods[mode]
    if mode == "subpix":
        win = int(corner_refine_win_px)
        if win < 1:
            raise ValueError("corner_refine_win_px must be >= 1")
        params.cornerRefinementWinSize = win
    return params


class ArucoDetector:
    """
    Thin wrapper around ``cv2.aruco.ArucoDetector``.

    Args:
        dictionary_name: OpenCV dictionary constant (e.g. ``DICT_4X4_50``).
        corner_refinement: ``subpix`` (default), ``contour`` or ``none``.
        corner_refine_win_px: Half-size of the sub-pixel search window.
    """

    def __init__(
        self,
        dictionary_name: str,
        *,
        corner_refinement: str = DEFAULT_CORNER_REFINEMENT,
        corner_refine_win_px: int = DEFAULT_CORNER_REFINE_WIN_PX,
    ) -> None:
        import cv2

        dictionary = get_aruco_dictionary(dictionary_name)
        self._detector = cv2.aruco.ArucoDetector(
            dictionary,
            make_detector_parameters(
                corner_refinement=corner_refinement,
                corner_refine_win_px=corner_refine_win_px,
            ),
        )
        self.dictionary_name = dictionary_name
        self.corner_refinement = str(corner_refinement)
        self.corner_refine_win_px = int(corner_refine_win_px)

    def detect(self, gray: np.ndarray) -> DetectionResult:
        """
        Detect ArUco markers in a grayscale image.

        Args:
            gray: Single-channel uint8 image.

        Returns:
            :class:`DetectionResult`; empty if no markers found.
        """
        if gray is None or gray.size == 0:
            return DetectionResult(markers=())

        corners, ids, _rejected = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return DetectionResult(markers=())

        markers: List[MarkerDetection] = []
        for corner, mid in zip(corners, ids.flatten()):
            markers.append(
                MarkerDetection(
                    marker_id=int(mid),
                    corners=np.asarray(corner[0], dtype=np.float64),
                )
            )
        return DetectionResult(markers=tuple(markers))

    def detect_rgb(self, rgb: np.ndarray) -> DetectionResult:
        """Detect markers in an RGB frame; gray (HxW / HxWx1) passes straight through.

        Lores tracking frames are the Y plane already, so callers such as
        ``optical eval`` may hand in 2-D arrays here.
        """
        import cv2

        arr = np.asarray(rgb) if rgb is not None else None
        if arr is None or arr.size == 0:
            return DetectionResult(markers=())
        if arr.ndim == 2:
            return self.detect(arr)
        if arr.ndim == 3 and arr.shape[2] == 1:
            return self.detect(arr[:, :, 0])
        if arr.ndim == 3 and arr.shape[2] == 4:
            return self.detect(cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY))
        return self.detect(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY))


# Once a camera has markers, later frames search this multiple of their box
# instead of the whole lores frame. A full frame still runs on this cadence
# and whenever the window comes back empty.
ROI_SCALE = 2.0
ROI_FULL_EVERY = 10


def roi_window(
    markers: Tuple[MarkerDetection, ...],
    image_shape: Tuple[int, ...],
    *,
    scale: float = ROI_SCALE,
) -> Tuple[int, int, int, int] | None:
    """Axis-aligned search window ``(x0, y0, x1, y1)`` around ``markers``.

    The window is ``scale`` times the union of the marker boxes, centered on
    that union, and clipped to the image. ``None`` when the box is unusable.
    """
    if not markers or len(image_shape) < 2:
        return None
    height, width = int(image_shape[0]), int(image_shape[1])
    if height < 2 or width < 2:
        return None
    pts = np.concatenate([np.asarray(m.corners, dtype=np.float64) for m in markers], axis=0)
    if pts.size == 0:
        return None
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx = (float(x0) + float(x1)) / 2.0
    cy = (float(y0) + float(y1)) / 2.0
    half_w = max(float(x1) - float(x0), 1.0) * float(scale) / 2.0
    half_h = max(float(y1) - float(y0), 1.0) * float(scale) / 2.0
    left = max(0, int(np.floor(cx - half_w)))
    top = max(0, int(np.floor(cy - half_h)))
    right = min(width, int(np.ceil(cx + half_w)))
    bottom = min(height, int(np.ceil(cy + half_h)))
    if right - left < 8 or bottom - top < 8:
        return None
    return left, top, right, bottom


def plan_roi(
    markers: Tuple[MarkerDetection, ...] | None,
    image_shape: Tuple[int, ...],
    *,
    frame_index: int,
    full_every: int = ROI_FULL_EVERY,
    scale: float = ROI_SCALE,
) -> Tuple[int, int, int, int] | None:
    """Search window for this frame, or ``None`` to scan the whole image.

    Frame 0 and every ``full_every``-th frame are full scans, as is any frame
    that has no previous markers.
    """
    if not markers or int(full_every) <= 1 or int(frame_index) % int(full_every) == 0:
        return None
    return roi_window(markers, image_shape, scale=scale)


def shift_detection(det: DetectionResult, dx: float, dy: float) -> DetectionResult:
    """Move corners from a crop back into full-image pixels."""
    if not det.markers or (dx == 0 and dy == 0):
        return det
    offset = np.array([dx, dy], dtype=np.float64)
    return DetectionResult(
        markers=tuple(
            MarkerDetection(marker_id=m.marker_id, corners=np.asarray(m.corners, dtype=np.float64) + offset)
            for m in det.markers
        )
    )


def _detect_image(detector: ArucoDetector, img: np.ndarray) -> DetectionResult:
    """Detect on gray in-place; RGB still goes through ``detect_rgb``."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return detector.detect(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return detector.detect(arr[:, :, 0])
    return detector.detect_rgb(arr)


def detect_frames(
    detector: ArucoDetector,
    frames: Mapping[str, np.ndarray],
    *,
    parallel: bool = True,
) -> Dict[str, DetectionResult]:
    """Detect markers in each camera image (gray or RGB).

    With two or more frames and ``parallel=True``, cameras are submitted to a
    small thread pool. OpenCV often releases the GIL inside ``detectMarkers``.
    """
    items = list(frames.items())
    if not items:
        return {}
    if not parallel or len(items) == 1:
        return {cam_id: _detect_image(detector, img) for cam_id, img in items}
    pool = _detect_pool()
    futures = {cam_id: pool.submit(_detect_image, detector, img) for cam_id, img in items}
    return {cam_id: fut.result() for cam_id, fut in futures.items()}
