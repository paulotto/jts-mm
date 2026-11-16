# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CT pivot: worst-first rejection, conditioning warning, bounded sampling, trust gate."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest


# ct_pivot imports the optical preview stack, which needs OpenCV.
pytest.importorskip("cv2", reason="ct_pivot needs OpenCV (optical preview)")
pytest.importorskip("optical", reason="needs the in-repo optical package")

from jtsmm.capture import ct_pivot
from jtsmm.capture.ct_pivot import (
    _sample_ct_pose,
    ct_tip_trustworthy,
    reject_high_err_poses,
)
from optical.tracking.pivot import solve_pivot_ols


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _poses(n: int = 10, *, two_axes: bool = True) -> list[np.ndarray]:
    tip = np.array([0.0, -220.0, -5.0])
    pivot = np.array([10.0, 20.0, 300.0])
    out = []
    for i in range(n):
        R = _rot("y", 12.0 * i)
        if two_axes:
            R = _rot("x", 25.0 * ((i % 3) - 1)) @ R
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pivot - R @ tip
        out.append(T)
    return out


def test_one_gross_outlier_costs_exactly_one_pose():
    poses = _poses(10)
    poses[4] = poses[4].copy()
    poses[4][:3, 3] += np.array([40.0, -25.0, 10.0])
    first = solve_pivot_ols(poses)
    # the outlier drags the solve: several GOOD poses read above threshold too
    assert sum(e > ct_pivot.MAX_PER_POSE_ERR_MM for e in first.per_pose_err_mm) > 1
    logs: list[str] = []
    kept, result, _f, n_rej = reject_high_err_poses(poses, first, log=logs.append)
    assert n_rej == 1 and len(kept) == 9
    assert result.max_err_mm < 1e-6
    assert any("rejecting pose 5/10" in m for m in logs)


def test_rejection_stops_at_min_poses():
    poses = _poses(5)
    for i in (1, 3):
        poses[i] = poses[i].copy()
        poses[i][:3, 3] += np.array([30.0 * (i - 2), 20.0, 0.0])
    logs: list[str] = []
    kept, _r, _f, n_rej = reject_high_err_poses(
        poses, solve_pivot_ols(poses), min_poses=4, log=logs.append
    )
    assert len(kept) >= 4 and n_rej <= 1
    assert any("would leave <4 poses" in m for m in logs)


def test_single_axis_poses_warn_about_conditioning():
    poses = _poses(6, two_axes=False)
    logs: list[str] = []
    reject_high_err_poses(poses, solve_pivot_ols(poses), log=logs.append)
    assert any("cond(A)" in m for m in logs)
    logs.clear()
    good = _poses(9)
    reject_high_err_poses(good, solve_pivot_ols(good), log=logs.append)
    assert not any("cond(A)" in m for m in logs)


class _Optical:
    """Tracker that sees the CT body ``n_visible`` times, then never again."""

    cam_ids = ["cam0"]

    def __init__(self, n_visible: int) -> None:
        self.left = n_visible
        self.tracker = SimpleNamespace(process_frame=self._process, detect_all=lambda f: {},
                                       last_detections=None)

    def _process(self, frames, **_kw):
        self.left -= 1
        pose = np.eye(4) if self.left >= 0 else np.full((4, 4), np.nan)
        return SimpleNamespace(pose_CT=pose)


class _Cam:
    cam_id = "cam0"

    def drain_frames(self, **_kw):
        return [SimpleNamespace(rgb=np.zeros((4, 4), dtype=np.uint8))]


def test_sampling_ends_with_partial_samples_when_body_leaves_view():
    t0 = time.monotonic()
    pose, diag = _sample_ct_pose(_Optical(3), [_Cam()], settle_s=0.05, min_samples=8, max_s=0.3)
    assert time.monotonic() - t0 < 2.0
    assert pose is not None and diag.valid_ct == 3 and diag.timed_out


def test_sampling_polls_keys_and_aborts_on_q():
    with pytest.raises(RuntimeError, match="aborted"):
        _sample_ct_pose(_Optical(0), [_Cam()], settle_s=5.0, max_s=10.0, poll_key=lambda: ord("q"))


def test_trust_gate():
    assert ct_tip_trustworthy(SimpleNamespace(rms_mm=1.2))
    assert not ct_tip_trustworthy(SimpleNamespace(rms_mm=7.0))
    assert not ct_tip_trustworthy(SimpleNamespace(rms_mm=float("nan")))
