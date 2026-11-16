# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""K scaling: pixel-centre aligned principal point, aspect guard, calibration-size source."""

from __future__ import annotations

import numpy as np
import pytest

from _fix_J2_imports import import_optical

geometry, io = import_optical("optical.calibration.geometry", "optical.calibration.io")


def _centroid(img: np.ndarray) -> tuple[float, float]:
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    return float((img * xx).sum() / img.sum()), float((img * yy).sum() / img.sum())


@pytest.mark.parametrize("factor", [2, 4])
def test_principal_point_follows_pixel_centres(factor: int) -> None:
    """A blob at (x0, y0) lands at (x0 + 0.5)/f - 0.5 after an f×f area downscale."""
    W, H = 1456, 1088
    x0, y0 = 700.0, 500.0
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * 6.0**2))
    small = img.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))
    cx_meas, cy_meas = _centroid(small)

    K = np.array([[1700.0, 0.0, x0], [0.0, 1700.0, y0], [0.0, 0.0, 1.0]])
    K2 = geometry.scale_K(K, (W, H), (W // factor, H // factor))
    assert K2[0, 2] == pytest.approx(cx_meas, abs=1e-3)
    assert K2[1, 2] == pytest.approx(cy_meas, abs=1e-3)
    assert K2[0, 0] == pytest.approx(1700.0 / factor)


def test_scale_K_rejects_aspect_change() -> None:
    K = np.array([[1000.0, 0, 728.0], [0, 1000.0, 544.0], [0, 0, 1]])
    with pytest.raises(ValueError, match="aspect ratio"):
        geometry.scale_K(K, (1456, 1088), (1456, 816))  # 16:9 crop
    # Rounding of odd sizes (<0.5 %) is fine.
    geometry.scale_K(K, (1456, 1088), (486, 363))


def test_scale_camera_intrinsics_uses_half_pixel_formula() -> None:
    K = np.array([[1000.0, 0, 728.0], [0, 1000.0, 544.0], [0, 0, 1]])
    intr = io.CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.1)
    out = io.scale_camera_intrinsics(intr, (728, 544))
    assert out.K[0, 2] == pytest.approx(363.75)
    assert out.K[1, 2] == pytest.approx(271.75)
    with pytest.raises(ValueError):
        io.scale_camera_intrinsics(intr, (728, 400))


def test_quality_eval_scales_from_calibration_size() -> None:
    pytest.importorskip("cv2")
    from optical.evaluation.image_quality import QualityProfile
    from optical.evaluation.run_quality_size_eval import _scaled_intrinsics_map

    K = np.array([[1000.0, 0, 728.0], [0, 1000.0, 544.0], [0, 0, 1]])
    intr = {"cam0": io.CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.1)}
    out = _scaled_intrinsics_map(intr, profile=QualityProfile(name="h", width=728, height=544))
    assert out["cam0"].K[0, 0] == pytest.approx(500.0)
    assert out["cam0"].image_size == (728, 544)
