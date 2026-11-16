# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""capture_cam: no wall-clock fallback for a missing SensorTimestamp; buffer default."""

from __future__ import annotations

from typing import Any

import jtsmm.capture.capture_cam as cam_mod
from jtsmm.capture.capture_cam import CameraCapture, cameras_from_config


class _Req:
    def __init__(self, ts: int | None, released: list) -> None:
        self.ts = ts
        self.released = released

    def get_metadata(self) -> dict[str, Any]:
        meta: dict[str, Any] = {"ExposureTime": 1000, "AnalogueGain": 1.0}
        if self.ts is not None:
            meta["SensorTimestamp"] = self.ts
        return meta

    def release(self) -> None:
        self.released.append(self.ts)


def _cam(monkeypatch) -> CameraCapture:
    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)
    cam = CameraCapture(cam_id="cam0", device="csi0", resolution=(8, 8), exposure_us=1000,
                        gain=1.0, expect_rate_hz=30.0)
    cam._mock = False
    cam._picam = object()
    return cam


def test_frame_without_sensor_timestamp_is_dropped_and_counted(monkeypatch):
    cam = _cam(monkeypatch)
    released: list = []
    pending = [_Req(1_000, released), _Req(None, released), _Req(3_000, released)]
    monkeypatch.setattr(cam, "_acquire_request", lambda *, blocking: pending.pop(0) if pending else None)
    frames = cam.drain_frames(max_frames=8)
    assert [f.sensor_timestamp_ns for f in frames] == [1_000, 3_000]
    assert [f.frame_idx for f in frames] == [1, 3]  # the gap marks the drop
    assert cam.missing_sensor_ts == 1
    assert released == [1_000, None, 3_000]  # every request still returned to libcamera


def test_buffer_count_default_is_six_when_yaml_omits_it(monkeypatch):
    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)
    cams = cameras_from_config({"list": [{"id": "cam0", "device": "csi0"}]})
    assert cams[0].buffer_count == 6
    cams = cameras_from_config({"list": [{"id": "cam0", "device": "csi0", "buffer_count": 8}]})
    assert cams[0].buffer_count == 8
