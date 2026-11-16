# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""OpticalTrackBridge scales K to the lores stream with the half-pixel formula."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("optical", reason="needs the in-repo optical package")

from jtsmm.capture.optical_bridge import OpticalTrackBridge  # noqa: E402
from optical.calibration.io import CameraIntrinsics  # noqa: E402


def _bridge(intr) -> OpticalTrackBridge:
    b = object.__new__(OpticalTrackBridge)
    b.tracker = SimpleNamespace(intrinsics={"cam0": intr})
    return b


def _cam(lores=(728, 544)):
    return SimpleNamespace(cam_id="cam0", resolution=(1456, 1088), lores_size=lores)


def test_principal_point_uses_pixel_centres():
    K = np.array([[1000.0, 0, 727.5], [0, 1000.0, 543.5], [0, 0, 1]])
    b = _bridge(CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.3))
    b._scale_intrinsics([_cam()], CameraIntrinsics)
    K2 = b.tracker.intrinsics["cam0"].K
    assert K2[0, 0] == pytest.approx(500.0) and K2[1, 1] == pytest.approx(500.0)
    # (c + 0.5) * 0.5 - 0.5, not c * 0.5
    assert K2[0, 2] == pytest.approx(363.5) and K2[1, 2] == pytest.approx(271.5)
    assert tuple(b.tracker.intrinsics["cam0"].image_size) == (728, 544)


def test_aspect_change_is_refused():
    K = np.array([[1000.0, 0, 700.0], [0, 1000.0, 500.0], [0, 0, 1]])
    b = _bridge(CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.3))
    with pytest.raises(ValueError):
        b._scale_intrinsics([_cam(lores=(728, 400))], CameraIntrinsics)
