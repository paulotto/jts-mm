# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-camera capture metadata (picamera2 SensorTimestamp, optional trigger id)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CamFrameMeta:
    cam_id: str
    sensor_timestamp_ns: int
    t_host_monotonic_ns: int
    trigger_id: int | None = None
    frame_idx: int = 0
