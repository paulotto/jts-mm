# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""SE(3) pose helpers."""

from __future__ import annotations

import numpy as np

from optical.tracking.pose import is_rigid_se3, is_valid_pose, nan_pose


def test_identity_is_rigid():
    assert is_rigid_se3(np.eye(4))
    assert is_valid_pose(np.eye(4))


def test_nan_pose_not_rigid():
    p = nan_pose()
    assert not is_valid_pose(p)
    assert not is_rigid_se3(p)


def test_scaled_rotation_not_rigid():
    t = np.eye(4)
    t[:3, :3] *= 2.0
    assert is_valid_pose(t)
    assert not is_rigid_se3(t)


def test_bad_last_row_not_rigid():
    t = np.eye(4)
    t[3, 3] = 2.0
    assert not is_rigid_se3(t)
