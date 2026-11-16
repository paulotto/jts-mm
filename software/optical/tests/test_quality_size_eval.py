# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for size × image-quality evaluation helpers."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from optical.evaluation.image_quality import (
    QualityProfile,
    degrade_rgb,
    resolve_profile_for_frame,
    scale_intrinsics,
)
from optical.evaluation.run_quality_size_eval import _evaluate_quality
from optical.markers.config import get_aruco_dictionary
from optical.tracking.aruco import ArucoDetector
from optical.calibration.io import CameraExtrinsics, CameraIntrinsics


def _synthetic_marker_rgb(
    *,
    marker_id: int = 2,
    dictionary: str = "DICT_4X4_50",
    canvas_wh: tuple[int, int] = (640, 480),
    side_px: int = 160,
) -> np.ndarray:
    """Render one ArUco tag centered on an RGB canvas."""
    w, h = canvas_wh
    d = get_aruco_dictionary(dictionary)
    tag = np.zeros((side_px, side_px), dtype=np.uint8)
    cv2.aruco.generateImageMarker(d, marker_id, side_px, tag, 1)
    gray = np.full((h, w), 220, dtype=np.uint8)
    x0 = (w - side_px) // 2
    y0 = (h - side_px) // 2
    gray[y0 : y0 + side_px, x0 : x0 + side_px] = tag
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def test_scale_intrinsics_linear():
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]])
    K2 = scale_intrinsics(K, src_wh=(1000, 800), dst_wh=(500, 400))
    assert K2[0, 0] == pytest.approx(500.0)
    assert K2[1, 1] == pytest.approx(400.0)
    assert K2[0, 2] == pytest.approx(249.75)  # (c + 0.5) * s - 0.5
    assert K2[1, 2] == pytest.approx(199.75)
    # Original unchanged
    assert K[0, 0] == 1000.0


def test_degrade_rgb_downscale_and_jpeg_shape():
    rgb = np.zeros((1088, 1456, 3), dtype=np.uint8)
    rgb[100:200, 100:200] = 255
    profile = QualityProfile(
        name="half_jpeg",
        width=728,
        height=544,
        jpeg_quality=50,
    )
    out = degrade_rgb(rgb, profile)
    assert out.shape == (544, 728, 3)
    assert out.dtype == np.uint8


def test_resolve_profile_scale():
    p = resolve_profile_for_frame({"name": "half", "scale": 0.5}, src_wh=(1456, 1088))
    assert p.width == 728
    assert p.height == 544
    assert p.name == "half"


def test_reprojection_rms_mm_scales_with_side_px():
    from optical.evaluation.metrics import reprojection_rms_mm

    # Same px error on a half-size image → twice the mm error.
    full = reprojection_rms_mm(reprojection_rms_px=0.3, size_mm=40.0, side_px=100.0)
    half = reprojection_rms_mm(reprojection_rms_px=0.15, size_mm=40.0, side_px=50.0)
    assert full == pytest.approx(0.12)
    assert half == pytest.approx(0.12)


def test_downscale_reduces_marker_side_px():
    rgb = _synthetic_marker_rgb(side_px=200, canvas_wh=(800, 600))
    detector = ArucoDetector("DICT_4X4_50")
    full = detector.detect_rgb(rgb)
    assert full.markers, "synthetic marker should be detected at full res"
    side_full = float(
        np.mean(
            [
                np.linalg.norm(full.markers[0].corners[(i + 1) % 4] - full.markers[0].corners[i])
                for i in range(4)
            ]
        )
    )

    half = degrade_rgb(rgb, QualityProfile(name="half", width=400, height=300))
    det_half = detector.detect_rgb(half)
    assert det_half.markers, "synthetic marker should still be detected at half res"
    side_half = float(
        np.mean(
            [
                np.linalg.norm(
                    det_half.markers[0].corners[(i + 1) % 4] - det_half.markers[0].corners[i]
                )
                for i in range(4)
            ]
        )
    )
    assert side_half < side_full * 0.7
    assert side_half > side_full * 0.3


def test_evaluate_quality_matrix_prefers_higher_res():
    """Higher-quality profile should yield larger side_px and pass more readily."""
    from optical.evaluation.run_quality_size_eval import MarkerSpec

    rgb = _synthetic_marker_rgb(marker_id=2, side_px=180, canvas_wh=(640, 480))
    burst = [{"cam0": rgb}, {"cam0": rgb.copy()}, {"cam0": rgb.copy()}]

    # Fake intrinsics roughly matching the canvas.
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5, dtype=np.float64)
    intrinsics = {
        "cam0": CameraIntrinsics(K=K, dist=dist, image_size=(640, 480), rms_px=0.1)
    }
    extrinsics = {
        "cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.1)
    }
    detector = ArucoDetector("DICT_4X4_50")
    markers = [MarkerSpec(marker_id=2, size_mm=30.0, label="30mm")]
    thresholds = {
        "min_detection_rate_pct": 50,
        "max_translation_jitter_mm": 50.0,
        "min_marker_side_px": 10,
    }

    full_profile = QualityProfile(name="full", width=640, height=480)
    tiny_profile = QualityProfile(name="tiny", width=80, height=60)

    full_res, full_t = _evaluate_quality(
        burst=burst,
        profile=full_profile,
        markers=markers,
        cam_ids=["cam0"],
        primary_cam_id="cam0",
        detector=detector,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        min_views=2,
        thresholds=thresholds,
        do_triangulation=False,
    )
    tiny_res, tiny_t = _evaluate_quality(
        burst=burst,
        profile=tiny_profile,
        markers=markers,
        cam_ids=["cam0"],
        primary_cam_id="cam0",
        detector=detector,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        min_views=2,
        thresholds=thresholds,
        do_triangulation=False,
    )

    assert full_res[0].detection_rate_pct == 100.0
    assert full_res[0].min_side_px > tiny_res[0].min_side_px
    # Tiny may still detect, but side_px must be much smaller.
    assert full_res[0].min_side_px >= tiny_res[0].min_side_px * 4
    assert full_t["detect_ms_mean"] >= 0.0
    assert tiny_t["detect_ms_mean"] >= 0.0
