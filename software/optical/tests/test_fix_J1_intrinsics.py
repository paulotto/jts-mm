# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Half-pixel-correct intrinsics rescaling (pure numpy; no cv2)."""

from __future__ import annotations

import numpy as np
import pytest

from optical.calibration.io import CameraIntrinsics
from optical.tracking.tracker import scale_intrinsics_for_capture

FULL = CameraIntrinsics(
    K=np.array([[1740.0, 0.0, 728.3], [0.0, 1741.0, 543.9], [0.0, 0.0, 1.0]]),
    dist=np.array([-0.1, 0.05, 0.0, 0.0, 0.0]),
    image_size=(1456, 1088),
    rms_px=0.25,
)


def _project(K, X):
    uv = X @ K.T
    return uv[:, :2] / uv[:, 2:]


def test_half_res_uses_pixel_centre_convention():
    half = scale_intrinsics_for_capture(FULL, (728, 544))
    assert half.image_size == (728, 544)
    assert half.K[0, 0] == pytest.approx(870.0)
    assert half.K[1, 1] == pytest.approx(870.5)
    assert half.K[0, 2] == pytest.approx((728.3 + 0.5) / 2 - 0.5)
    assert half.K[1, 2] == pytest.approx((543.9 + 0.5) / 2 - 0.5)
    assert np.array_equal(half.dist, FULL.dist)
    # A 2x2 box-downsample maps full-res pixel u to (u + 0.5)/2 - 0.5.
    X = np.array([[30.0, -20.0, 600.0], [-80.0, 55.0, 500.0]])
    uv_full = _project(FULL.K, X)
    assert np.allclose(_project(half.K, X), (uv_full + 0.5) / 2 - 0.5)


def test_same_size_is_identity_and_idempotent():
    assert scale_intrinsics_for_capture(FULL, (1456, 1088)) is FULL
    half = scale_intrinsics_for_capture(FULL, (728, 544))
    assert scale_intrinsics_for_capture(half, (728, 544)) is half


def test_small_aspect_change_ok_large_rejected():
    scale_intrinsics_for_capture(FULL, (640, 480))  # 0.36 % aspect change
    with pytest.raises(ValueError, match="aspect"):
        scale_intrinsics_for_capture(FULL, (728, 400))
    with pytest.raises(ValueError):
        scale_intrinsics_for_capture(FULL, (0, 544))
