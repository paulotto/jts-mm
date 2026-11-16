# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU lever arm from the hand-eye mounting sessions (validation plan VD15).

The method moved to :mod:`biomech.imu.lever` (2026-10-29): the dataset stage
``camerafree`` (X5) estimates its E5/E6 lever arms with it from the hand-eye session, and the
dataset study's code hash must not contain validation code. This module re-exports it
unchanged for the validation stages and tests.
"""

from biomech.imu.lever import *  # noqa: F401,F403
from biomech.imu.lever import DEFAULTS, G_MM_S2, design, fit, motion_band_hz, read_slot  # noqa: F401
from biomech.imu.lever import run_ref_imu_lever  # noqa: F401
