# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MJPEG preview is downscaled and rate-limited; capture may be full-res or lores."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("cv2", reason="needs OpenCV")

from optical.calibration.multicam import (
    _PREVIEW_HZ,
    _PREVIEW_MAX_WH,
    _PREVIEW_PERIOD_S,
    _STATUS_BAR_H,
    _draw_status_bar,
    _fit_preview,
)


def test_preview_runs_at_20hz():
    assert _PREVIEW_HZ == 20.0
    assert _PREVIEW_PERIOD_S == pytest.approx(0.05)


def test_fit_preview_downscales_stacked_stereo():
    max_w, max_h = _PREVIEW_MAX_WH
    stacked = np.zeros((1088, 2912, 3), dtype=np.uint8)
    out = _fit_preview(stacked)
    assert out.shape[1] <= max_w
    assert out.shape[0] <= max_h
    assert out.shape[2] == 3


def test_fit_preview_leaves_small_frames_unchanged():
    small = np.zeros((240, 320, 3), dtype=np.uint8)
    out = _fit_preview(small)
    assert out.shape == small.shape
    assert out is small


def test_status_bar_is_compact_on_preview_sized_frames():
    preview = np.zeros((_PREVIEW_MAX_WH[1], _PREVIEW_MAX_WH[0], 3), dtype=np.uint8)
    out = _draw_status_bar(preview, "cam0 captures 3/25 | Board detected", (0, 220, 0))
    assert out.shape == preview.shape
    assert _STATUS_BAR_H == 20  # cdf7a68 shrank the preview font
    assert tuple(int(v) for v in out[_STATUS_BAR_H // 2, 2]) == (0, 0, 0)
    # Must not cover a large fraction of the preview.
    assert _STATUS_BAR_H / preview.shape[0] < 0.15
