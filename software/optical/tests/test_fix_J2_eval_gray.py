# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`optical eval` on the default lores stream hands 2-D Y planes to detection."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from optical.calibration.io import CameraIntrinsics  # noqa: E402
from optical.evaluation import run_marker_size_eval as rmse  # noqa: E402
from optical.tracking.aruco import DetectionResult, MarkerDetection  # noqa: E402


class _GrayOnlyDetector:
    """Stands in for ArucoDetector: detect_rgb() on a 2-D frame is the old crash."""

    def __init__(self, corners: np.ndarray) -> None:
        self.gray_shapes: list[tuple[int, ...]] = []
        self._corners = corners

    def detect(self, gray: np.ndarray) -> DetectionResult:
        assert gray.ndim == 2
        self.gray_shapes.append(gray.shape)
        return DetectionResult(markers=(MarkerDetection(marker_id=3, corners=self._corners),))

    def detect_rgb(self, rgb: np.ndarray) -> DetectionResult:
        # Same failure mode as cv2.cvtColor(gray, COLOR_RGB2GRAY).
        raise cv2.error("detect_rgb called with a 2-D frame")


class _Feed:
    def __init__(self, frames: dict[str, np.ndarray], n: int) -> None:
        self._frames = frames
        self._seq = 0
        self._n = n

    def set_status(self, _text: str) -> None:
        pass

    def latest_frames(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._frames.items()}

    def latest_frames_seq(self, after_seq: int = -1, timeout_s: float = 1.0):
        self._seq += 1
        return self._seq, self.latest_frames()


def test_record_frames_accepts_2d_lores_frames() -> None:
    gray = np.full((544, 728), 200, dtype=np.uint8)
    corners = np.array([[300, 200], [360, 200], [360, 260], [300, 260]], dtype=np.float64)
    det = _GrayOnlyDetector(corners)
    K = np.array([[870.0, 0, 364], [0, 870.0, 272], [0, 0, 1]])
    intr = {"cam0": CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(728, 544), rms_px=0.3)}
    flags, samples, _tri_flags, _tri = rmse._record_frames(
        phase_name="static",
        n_frames=5,
        warmup_frames=0,
        feed=_Feed({"cam0": gray}, 5),
        detector=det,
        markers=[rmse.MarkerSpec(marker_id=3, size_mm=30.0, label="30mm")],
        cam_ids=["cam0"],
        primary_cam_id="cam0",
        intrinsics=intr,
        extrinsics={},
        min_views=1,
        enable_triangulation=False,
    )
    assert flags[3] == [True] * 5
    assert len(samples[3]) == 5
    assert det.gray_shapes and all(s == (544, 728) for s in det.gray_shapes)


def test_draw_markers_on_gray_returns_bgr() -> None:
    gray = np.zeros((40, 60), dtype=np.uint8)
    out = rmse._draw_markers(gray, {3: np.array([[5, 5], [20, 5], [20, 20], [5, 20]])}, {3: "x"})
    assert out.shape == (40, 60, 3)
    assert out[..., 1].max() == 255  # green overlay visible


@pytest.mark.skipif(
    not hasattr(getattr(cv2, "aruco", None), "ArucoDetector"),
    reason="needs cv2.aruco.ArucoDetector (OpenCV >= 4.7)",
)
def test_real_detector_on_2d_frame() -> None:
    from optical.markers.config import get_aruco_dictionary
    from optical.tracking.aruco import ArucoDetector, _detect_image

    d = get_aruco_dictionary("DICT_4X4_50")
    tag = np.zeros((120, 120), dtype=np.uint8)
    cv2.aruco.generateImageMarker(d, 3, 120, tag, 1)
    gray = np.full((544, 728), 230, dtype=np.uint8)
    gray[200:320, 300:420] = tag
    res = _detect_image(ArucoDetector("DICT_4X4_50"), gray)
    assert [m.marker_id for m in res.markers] == [3]
