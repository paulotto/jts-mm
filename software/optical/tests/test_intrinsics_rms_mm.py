# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Board-plane mm conversion for intrinsics RMS."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("cv2", reason="needs OpenCV")

from optical.calibration.multicam import (
    _board_square_size_mm,
    _build_checkerboard_obj_points,
    _intrinsics_rms_mm,
    _median_board_square_px,
)


def test_board_square_size_mm_from_grid():
    objp = _build_checkerboard_obj_points((5, 4), 37.0)
    assert _board_square_size_mm(objp) == 37.0


def test_median_board_square_px_synthetic():
    square_mm = 37.0
    objp = _build_checkerboard_obj_points((5, 4), square_mm)
    # One square = 50 px in image; add tiny noise.
    imgp = objp[:, :2].copy() * (50.0 / square_mm)
    imgp += np.random.default_rng(0).normal(0, 0.01, imgp.shape)
    pitch = _median_board_square_px([objp], [imgp.reshape(-1, 1, 2)], square_size_mm=square_mm)
    assert pitch == pytest.approx(50.0, abs=0.01)


def test_intrinsics_rms_mm_scales_with_pitch():
    square_mm = 37.0
    objp = _build_checkerboard_obj_points((5, 4), square_mm)
    imgp = objp[:, :2].copy() * (40.0 / square_mm)
    rms_mm = _intrinsics_rms_mm(0.8, [objp], [imgp.reshape(-1, 1, 2)])
    # 0.8 px * 37 mm / 40 px ≈ 0.74 mm
    assert rms_mm == pytest.approx(0.74, abs=0.01)
