# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Lores track path helpers (match jtsmm record --optical-track)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pytest

pytest.importorskip("cv2", reason="needs OpenCV")

from optical.calibration.io import CameraIntrinsics, scale_camera_intrinsics
from optical.calibration.multicam import CamSpec, enable_track_lores, half_lores_size
from optical.tracking.frame_store import load_frame, write_frame, write_session_meta


def test_half_lores_size():
    assert half_lores_size((1456, 1088)) == (728, 544)
    assert half_lores_size((640, 480)) == (320, 240)


def test_enable_track_lores_sets_half():
    cams = [CamSpec(cam_id="cam0", camera_num=0, resolution=(1456, 1088))]
    enable_track_lores(cams)
    assert cams[0].lores_size == (728, 544)


def test_scale_intrinsics_to_lores():
    K = np.array([[1000.0, 0, 728.0], [0, 1000.0, 544.0], [0, 0, 1]], dtype=np.float64)
    intr = CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.1)
    scaled = scale_camera_intrinsics(intr, (728, 544))
    assert scaled.image_size == (728, 544)
    assert scaled.K[0, 0] == 500.0
    assert scaled.K[0, 2] == 363.75  # (728 + 0.5) * 0.5 - 0.5 (pixel-centre aligned)


def test_frame_store_gray_roundtrip(tmp_path: Path):
    session = tmp_path / "rec"
    write_session_meta(
        session,
        cam_ids=["cam0"],
        image_size=(728, 544),
        stream="lores_y",
    )
    gray = np.arange(728 * 544, dtype=np.uint8).reshape(544, 728)
    write_frame(session, 0, {"cam0": gray})
    loaded = load_frame(session, 0, ["cam0"])
    assert loaded["cam0"].ndim == 2
    assert loaded["cam0"].shape == (544, 728)
    assert np.array_equal(loaded["cam0"], gray)
