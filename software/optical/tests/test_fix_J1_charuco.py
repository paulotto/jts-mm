# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Printable ChArUco board keeps the requested square pitch (needs cv2 >= 4.7)."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2.aruco, "ArucoDetector"):
    pytest.skip("cv2.aruco.ArucoDetector needs OpenCV >= 4.7", allow_module_level=True)

from optical.markers.config import generate_charuco_board_png  # noqa: E402
from optical.tracking.aruco import ArucoDetector  # noqa: E402


@pytest.mark.parametrize("margin_mm", [0.0, 5.0, 12.0])
def test_charuco_png_square_pitch_matches_request(tmp_path, margin_mm):
    sx, sy, square_mm, marker_mm, dpi = 7, 5, 30.0, 22.0, 150
    out = generate_charuco_board_png(
        squares_x=sx,
        squares_y=sy,
        square_length_mm=square_mm,
        marker_length_mm=marker_mm,
        dictionary_name="DICT_4X4_50",
        out_path=tmp_path / "board.png",
        dpi=dpi,
        margin_mm=margin_mm,
    )
    img = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    ppmm = dpi / 25.4
    margin_px = int(round(margin_mm * ppmm))
    assert img.shape[1] == int(round(sx * square_mm * ppmm)) + 2 * margin_px
    assert img.shape[0] == int(round(sy * square_mm * ppmm)) + 2 * margin_px

    det = ArucoDetector("DICT_4X4_50", corner_refinement="none").detect(img)
    assert len(det.markers) == (sx * sy) // 2
    centres = np.array([m.corners.mean(axis=0) for m in det.markers])
    sides = np.array(
        [np.linalg.norm(np.diff(np.vstack([m.corners, m.corners[:1]]), axis=0), axis=1).mean()
         for m in det.markers]
    )
    # Markers on the same row sit two squares apart.
    pitches = []
    for c in centres:
        same_row = centres[(np.abs(centres[:, 1] - c[1]) < 2) & (centres[:, 0] > c[0] + 1)]
        if len(same_row):
            pitches.append(same_row[:, 0].min() - c[0])
    pitch_px = float(np.median(pitches)) / 2.0
    assert pitch_px == pytest.approx(square_mm * ppmm, rel=0.01)
    assert float(np.median(sides)) == pytest.approx(marker_mm * ppmm, rel=0.02)
