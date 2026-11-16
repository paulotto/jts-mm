# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Small J2 fixes: triangulation 'rms' is an RMS; unimplemented capture keys are announced."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def test_tri_reprojection_is_rms_not_mean() -> None:
    from optical.evaluation.triangulation_eval import _rms, reprojection_residuals_px

    P = np.hstack([np.eye(3), np.zeros((3, 1))])  # normalised camera, point at z=1
    X = np.array([0.0, 0.0, 1.0])
    obs = [(P, np.array([3.0, 0.0])), (P, np.array([0.0, 4.0]))]
    r = reprojection_residuals_px(X, obs)
    assert np.allclose(r, [3.0, 4.0])
    assert _rms(r) == pytest.approx(np.sqrt(12.5))  # the old code reported the mean, 3.5


def test_lock_keys_are_reported_as_ignored(tmp_path: Path, capsys) -> None:
    from optical.calibration import multicam
    from optical.calibration.extrinsics import calibration_cache_paths

    bundle = tmp_path / "calibration_bundle.h5"
    np.savez(
        calibration_cache_paths(bundle)[0],
        cam0_K=np.eye(3), cam0_dist=np.zeros(5),
        cam0_image_size=np.array([640, 480], np.int32), cam0_rms_px=np.asarray(0.2),
    )
    cfg = {
        "cameras": [{"id": "cam0", "camera_num": 0, "resolution": [640, 480]}],
        "board": {"type": "checkerboard", "checkerboard_inner_corners": [9, 6], "square_size_mm": 25.0},
        "capture": {"lock_exposure": True, "lock_focus": True},
        "output": {"bundle_path": str(bundle)},
    }
    with (
        patch("optical.calibration.multicam._load_json", return_value=cfg),
        patch("optical.calibration.multicam._build_board_detector", return_value=object()),
    ):
        multicam.run_calibration(config_path=tmp_path / "x.yaml", mode="bundle", backend="v4l2")
    assert "lock_exposure/lock_focus are not implemented" in capsys.readouterr().out
    assert bundle.is_file()
