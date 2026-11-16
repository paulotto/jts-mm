# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Camera YAML → CameraCapture → libcamera controls."""

from __future__ import annotations

import copy
import importlib.util
from concurrent.futures import Future

import pytest

from jtsmm.capture.capture_cam import (
    CameraCapture,
    apply_mount_rotation,
    cameras_from_config,
    libcamera_controls_from_cam,
    resolve_camera_num,
    y_plane,
)
from jtsmm.config import validate_config


def test_resolve_camera_num_aliases():
    assert resolve_camera_num("csi0") == 0
    assert resolve_camera_num("CSI1") == 1
    assert resolve_camera_num("2") == 2
    assert resolve_camera_num(3) == 3
    with pytest.raises(ValueError):
        resolve_camera_num("usb0")


def test_cameras_from_config_template(resolved_config, monkeypatch):
    import jtsmm.capture.capture_cam as cam_mod

    # Keep this unit test off the real CSI bus (Pi hosts open picamera2 otherwise).
    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)

    cams = cameras_from_config(resolved_config["host"]["cameras"])
    assert len(cams) == 2
    cam0 = cams[0]
    assert cam0.cam_id == "cam0"
    assert cam0.device == "csi0"
    assert cam0.resolution == (1456, 1088)
    assert cam0.exposure_us == 12000
    assert cam0.gain == 1.0
    assert cam0.ae_enable is False
    assert cam0.awb_enable is False
    assert cam0.awb_gains == (2.3, 2.3)
    assert cam0.format == "YUV420"
    assert cam0.buffer_count == 6
    assert cam0.noise_reduction_mode == "high_quality"
    assert cam0.expect_rate_hz == 40.0
    assert cam0.lores_size is None
    assert cam0._mock is True


def test_cameras_from_config_track_lores_half_res(resolved_config, monkeypatch):
    import jtsmm.capture.capture_cam as cam_mod

    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)
    cams = cameras_from_config(
        resolved_config["host"]["cameras"],
        track_lores=True,
    )
    cam0 = cams[0]
    assert cam0.format == "YUV420"
    assert cam0.lores_size == (cam0.resolution[0] // 2, cam0.resolution[1] // 2)
    assert cam0.lores_size == (728, 544)
    assert cam0.lores_format == "YUV420"


def test_apply_mount_rotation_180(monkeypatch):
    import numpy as np
    import jtsmm.capture.capture_cam as cam_mod

    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    rgb[1, 2] = (4, 5, 6)
    monkeypatch.setattr(cam_mod, "MOUNT_ROTATE_180", True)
    out = apply_mount_rotation(gray)
    assert out[0, 0] == gray[-1, -1]
    assert out[-1, -1] == gray[0, 0]
    assert out.flags["C_CONTIGUOUS"]
    flipped = apply_mount_rotation(rgb)
    np.testing.assert_array_equal(flipped[0, 0], (4, 5, 6))
    monkeypatch.setattr(cam_mod, "MOUNT_ROTATE_180", False)
    np.testing.assert_array_equal(apply_mount_rotation(gray), gray)
    assert apply_mount_rotation(None) is None


def test_y_plane_slices_yuv420_luma():
    import numpy as np

    w, h = 8, 6
    yuv = np.arange(h * 3 // 2 * w, dtype=np.uint8).reshape(h * 3 // 2, w)
    y = y_plane(yuv, (w, h))
    assert y.shape == (h, w)
    assert y.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(y, yuv[:h, :w])
    yuv[0, 0] = 255
    assert y[0, 0] != 255  # copy, not a view of UV-backing buffer


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs OpenCV")
def test_y_plane_accepts_gray_and_rgb():
    import numpy as np

    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = y_plane(gray, (4, 3))
    np.testing.assert_array_equal(out, gray)
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[:, :] = (10, 20, 30)
    y = y_plane(rgb, (4, 3))
    assert y.shape == (3, 4)
    assert y.dtype == np.uint8


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs OpenCV")
def test_scale_intrinsics_half_res_matches_lores():
    """Lores ArUco K scaling (OpticalTrackBridge._scale_intrinsics must use the same formula)."""
    import numpy as np

    from optical.evaluation.image_quality import scale_intrinsics

    K = np.array([[1000.0, 0.0, 728.0], [0.0, 1000.0, 544.0], [0.0, 0.0, 1.0]])
    K2 = scale_intrinsics(K, src_wh=(1456, 1088), dst_wh=(728, 544))
    assert K2[0, 0] == pytest.approx(500.0)
    assert K2[0, 2] == pytest.approx(363.75)  # pixel-centre aligned: (c + 0.5) * s - 0.5


def test_libcamera_controls_manual_exposure_gain():
    cams = cameras_from_config(
        {
            "list": [
                {
                    "id": "cam0",
                    "device": "csi0",
                    "resolution": [640, 480],
                    "exposure_us": 3000,
                    "gain": 2.0,
                    "ae_enable": False,
                    "awb_enable": False,
                    "awb_gains": [1.2, 1.8],
                    "sharpness": 0.5,
                    "brightness": 0.1,
                    "contrast": 1.2,
                    "saturation": 0.9,
                    "noise_reduction_mode": "off",
                    "trigger": {"source": "mcu", "expect_rate_hz": 60},
                }
            ]
        }
    )
    controls = libcamera_controls_from_cam(cams[0])
    assert controls["AeEnable"] is False
    assert controls["AwbEnable"] is False
    assert controls["ExposureTime"] == 3000
    assert controls["AnalogueGain"] == 2.0
    assert controls["ColourGains"] == (1.2, 1.8)
    assert controls["Sharpness"] == 0.5
    assert controls["NoiseReductionMode"] == 0
    assert controls["FrameDurationLimits"] == (16667, 16667)


def test_libcamera_controls_ae_on_omits_manual_exposure():
    cams = cameras_from_config(
        {
            "list": [
                {
                    "id": "cam0",
                    "device": "0",
                    "resolution": [640, 480],
                    "exposure_us": 4000,
                    "gain": 1.0,
                    "ae_enable": True,
                    "awb_enable": True,
                    "trigger": {"source": "mcu", "expect_rate_hz": 30},
                }
            ]
        }
    )
    controls = libcamera_controls_from_cam(cams[0])
    assert controls["AeEnable"] is True
    assert "ExposureTime" not in controls
    assert "AnalogueGain" not in controls
    assert "ColourGains" not in controls
    assert controls["FrameDurationLimits"] == (33333, 33333)


def test_invalid_camera_gain_rejected(resolved_config):
    doc = copy.deepcopy(resolved_config)
    doc["host"]["cameras"]["list"][0]["gain"] = 99.0
    errors = validate_config(doc)
    assert any("gain" in e for e in errors)


def test_track_b_ae_enable_warns(resolved_config):
    doc = copy.deepcopy(resolved_config)
    doc["host"]["cameras"]["list"][0]["ae_enable"] = True
    errors = validate_config(doc)
    assert any(e.startswith("warn:") and "ae_enable" in e for e in errors)


class _FakeRequest:
    def __init__(self, ts: int = 1_000, exposure: int = 3000, gain: float = 2.0) -> None:
        self._meta = {
            "SensorTimestamp": ts,
            "ExposureTime": exposure,
            "AnalogueGain": gain,
        }
        self.released = False

    def get_metadata(self) -> dict:
        return dict(self._meta)

    def release(self) -> None:
        self.released = True


class _FakeJob:
    """Mirrors picamera2.Job: get_result, no release."""

    def __init__(self) -> None:
        self._future: Future = Future()

    def complete(self, request: _FakeRequest) -> None:
        self._future.set_result(request)

    def get_result(self, timeout: float | None = None):
        return self._future.result(timeout=timeout)


class _FakePicam:
    def __init__(self) -> None:
        self.capture_waits: list[object] = []
        self.last_job: _FakeJob | None = None

    def capture_request(self, wait=None):
        self.capture_waits.append(wait)
        if wait is False:
            self.last_job = _FakeJob()
            return self.last_job
        raise AssertionError("blocking capture_request() should not be used while polling")


def _wired_cam(monkeypatch, picam: _FakePicam) -> CameraCapture:
    monkeypatch.setattr(
        "jtsmm.capture.capture_cam._try_open_picamera2", lambda cam: False
    )
    cam = CameraCapture(
        cam_id="cam0",
        device="csi0",
        resolution=(64, 48),
        exposure_us=4000,
        gain=1.0,
        expect_rate_hz=60,
    )
    cam._mock = False
    cam._picam = picam
    return cam


def test_poll_frame_does_not_release_picamera2_job(monkeypatch):
    """capture_request(wait=False) returns a Job; treating it as a request
    used to crash with ``'Job' object has no attribute 'release'``."""
    picam = _FakePicam()
    cam = _wired_cam(monkeypatch, picam)

    assert cam.poll_frame() is None
    assert picam.capture_waits == [False]
    job = picam.last_job
    assert job is not None
    assert cam._pending_job is job

    # A second poll must reuse the outstanding job, not enqueue another.
    assert cam.poll_frame() is None
    assert picam.capture_waits == [False]

    request = _FakeRequest(ts=42_000, exposure=2500, gain=1.5)
    job.complete(request)
    frame = cam.poll_frame()
    assert frame is not None
    assert frame.sensor_timestamp_ns == 42_000
    assert frame.exposure_us == 2500
    assert frame.gain == 1.5
    assert frame.mock is False
    assert request.released is True
    assert cam._pending_job is None

    # Next poll starts a fresh job.
    assert cam.poll_frame() is None
    assert picam.capture_waits == [False, False]
