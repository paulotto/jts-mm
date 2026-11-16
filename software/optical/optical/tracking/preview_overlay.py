# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ArUco boxes for live MJPEG preview (optical track / eval)."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from optical.tracking.aruco import DetectionResult

MarkerBoxes = Iterable[Tuple[int, np.ndarray]]


def markers_from_detection(det: DetectionResult | None) -> list[tuple[int, np.ndarray]]:
    if det is None:
        return []
    return [(int(m.marker_id), np.asarray(m.corners, dtype=np.float64)) for m in det.markers]


def markers_by_cam(
    detections: Mapping[str, DetectionResult] | None,
) -> Dict[str, list[tuple[int, np.ndarray]]]:
    if not detections:
        return {}
    return {cid: markers_from_detection(det) for cid, det in detections.items()}


def draw_marker_boxes_bgr(
    bgr: np.ndarray,
    markers: MarkerBoxes,
    *,
    color: Sequence[int] = (0, 255, 0),
) -> np.ndarray:
    """Draw green bounding boxes + IDs on a BGR image (copy)."""
    import cv2

    out = np.asarray(bgr).copy()
    bgr_color = (int(color[0]), int(color[1]), int(color[2]))
    for marker_id, corners in markers:
        pts = np.asarray(corners, dtype=np.float64).reshape(-1, 2).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(out, (x, y), (x + bw, y + bh), bgr_color, 2)
        cv2.putText(
            out,
            str(int(marker_id)),
            (x, max(y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            bgr_color,
            1,
            cv2.LINE_AA,
        )
    return out


def overlay_frames_bgr(
    frames: Mapping[str, np.ndarray],
    detections: Mapping[str, DetectionResult] | None,
    *,
    as_bgr,
) -> Dict[str, np.ndarray]:
    """Convert each frame to BGR and paint ArUco boxes from ``detections``."""
    by_cam = markers_by_cam(detections)
    out: Dict[str, np.ndarray] = {}
    for cam_id, img in frames.items():
        bgr = as_bgr(img)
        markers = by_cam.get(cam_id) or []
        out[cam_id] = draw_marker_boxes_bgr(bgr, markers) if markers else bgr
    return out
