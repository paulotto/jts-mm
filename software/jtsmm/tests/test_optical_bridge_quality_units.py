# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The /optical/quality column carries the unit its rms_kind label names."""

from __future__ import annotations

import math
from types import SimpleNamespace

from jtsmm.capture.optical_bridge import _quality_from_record


def _rec(**q):
    return SimpleNamespace(quality={"MP": q})


def test_triangulation_stores_kabsch_mm_not_pixels():
    out = _quality_from_record(_rec(reprojection_rms_px=0.7, fit_rms_mm=1.9, n_tags=2, n_cameras=2), "triangulation")
    assert next(v for v in out.values() if v["n_tags"] == 2)["reprojection_rms"] == 1.9


def test_triangulation_without_fit_is_nan_not_the_pixel_value():
    out = _quality_from_record(_rec(reprojection_rms_px=0.7, n_tags=2, n_cameras=2), "triangulation")
    assert math.isnan(next(v for v in out.values() if v["n_tags"] == 2)["reprojection_rms"])


def test_pnp_stores_pixels():
    out = _quality_from_record(_rec(reprojection_rms_px=0.7, fit_rms_mm=1.9, n_tags=2, n_cameras=1), "pnp")
    assert next(v for v in out.values() if v["n_tags"] == 2)["reprojection_rms"] == 0.7
