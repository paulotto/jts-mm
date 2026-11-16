# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Pose estimation, tracking, and JTS output contract (doc 03).

See ``docs/optical/03_pose_estimation_and_output.md``.
"""

from optical.tracking.io import FrameRecord, read_tracking_session, write_tracking_session
from optical.tracking.tracker import ArucoJawTracker, TrackerConfig

__all__ = [
    "ArucoJawTracker",
    "FrameRecord",
    "TrackerConfig",
    "read_tracking_session",
    "write_tracking_session",
]
