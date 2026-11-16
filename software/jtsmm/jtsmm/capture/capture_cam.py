# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""picamera2 dual IMX296 capture — mock when hardware unavailable."""

from __future__ import annotations

import re
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Iterator

# Map YAML noise_reduction_mode → libcamera NoiseReductionMode enum values.
_NOISE_REDUCTION = {
    "off": 0,
    "fast": 1,
    "high_quality": 2,
}

# CSI modules are mounted 180° (ribbon out the "top"). Applied in software on
# every pixel buffer (record lores, live viewer, optical calibration) because
# libcamera Transform is often a no-op on IMX296. Flip this to disable.
MOUNT_ROTATE_180 = True


@dataclass
class CamFrameMeta:
    cam_id: str
    frame_idx: int
    sensor_timestamp_ns: int
    t_host_monotonic_ns: int
    exposure_us: int
    gain: float
    trigger_id: int | None = None
    mock: bool = False
    # Pixel buffer for ArUco / viewer. Tracking lores is the Y plane (HxW uint8),
    # not RGB — the field name is historical.
    rgb: Any | None = None


@dataclass
class CameraCapture:
    """Per-camera capture process helper (mock or picamera2)."""

    cam_id: str
    device: str
    resolution: tuple[int, int]
    exposure_us: int
    gain: float
    expect_rate_hz: float
    format: str = "YUV420"
    # 6, not 4: with 4 the ISP ran out of free buffers whenever the capture loop
    # stalled for a couple of frames (bursty SensorTimestamp gaps, p99 = 8
    # periods on take1). One 1456x1088 YUV420 main (2.4 MB) + half-res lores
    # (0.6 MB) is ~3 MB per buffer: 2 cameras x 6 = ~36 MB of CMA.
    buffer_count: int = 6
    ae_enable: bool = False
    awb_enable: bool = False
    awb_gains: tuple[float, float] = (1.5, 1.5)
    sharpness: float = 1.0
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    noise_reduction_mode: str = "fast"
    # Optional second stream for cheap luma (ArUco / viewer). When set, ``with_rgb``
    # pulls ``lores`` instead of copying full-res ``main``, then keeps only Y.
    lores_size: tuple[int, int] | None = None
    lores_format: str = "YUV420"
    _frame_idx: int = 0
    # Frames dropped because libcamera sent no SensorTimestamp.
    missing_sensor_ts: int = 0
    _last_poll_dropped: bool = field(default=False, repr=False)
    _mock: bool = True
    _picam: Any = None
    _pending_job: Any = field(default=None, repr=False)
    _next_frame_at: float = field(default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._mock = not _try_open_picamera2(self)

    @property
    def is_mock(self) -> bool:
        """True when picamera2 is unavailable and this camera synthesizes metadata."""
        return self._mock

    def start(self) -> None:
        self._next_frame_at = 0.0
        self._pending_job = None
        if not self._mock and self._picam is not None:
            _apply_camera_controls(self._picam, self)
            self._picam.start()

    def stop(self) -> None:
        self._pending_job = None
        if not self._mock and self._picam is not None:
            self._picam.stop()
            self._picam.close()
            self._picam = None

    def iter_frames(self, duration_s: float) -> Iterator[CamFrameMeta]:
        period = 1.0 / max(self.expect_rate_hz, 1.0)
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            frame = self._capture_one(blocking=True)
            if frame is not None:
                yield frame
            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def poll_frame(self, *, with_rgb: bool = False) -> CamFrameMeta | None:
        """Poll at most one frame for the capture hot loop (no sleep).

        Mock cameras are host-rate-limited to ``expect_rate_hz``. Real picamera2
        cameras use a non-blocking request so the supervisor can drain the
        completed-request queue without stalling MCU USB pumping.
        """
        if self._mock:
            period = 1.0 / max(self.expect_rate_hz, 1.0)
            now = time.monotonic()
            if now < self._next_frame_at:
                return None
            self._next_frame_at = now + period
            return self._capture_one(blocking=False, with_rgb=with_rgb)
        return self._capture_one(blocking=False, with_rgb=with_rgb)

    def drain_frames(
        self,
        *,
        max_frames: int = 8,
        with_rgb: bool = False,
        rgb_limit: int | None = None,
    ) -> list[CamFrameMeta]:
        """Drain up to ``max_frames`` completed camera requests.

        ``rgb_limit`` caps how many of the drained frames carry an RGB array; the
        rest are metadata-only. A caller that needs occasional pixels must ask for
        them here rather than polling afterwards, because this drain runs until the
        completed-request queue is empty.
        """
        out: list[CamFrameMeta] = []
        rgb_left = max_frames if rgb_limit is None else max(0, int(rgb_limit))
        for _ in range(max(1, max_frames)):
            take_rgb = with_rgb and rgb_left > 0
            self._last_poll_dropped = False
            frame = self.poll_frame(with_rgb=take_rgb)
            if frame is None:
                if self._last_poll_dropped:
                    continue  # a dropped frame, not an empty queue
                break
            if take_rgb and frame.rgb is not None:
                rgb_left -= 1
            out.append(frame)
            if self._mock:
                break
        return out

    def _capture_one(self, *, blocking: bool = True, with_rgb: bool = False) -> CamFrameMeta | None:
        if self._mock:
            self._frame_idx += 1
            rgb = _mock_preview_rgb(self.cam_id, self._frame_idx) if with_rgb else None
            return CamFrameMeta(
                cam_id=self.cam_id,
                frame_idx=self._frame_idx,
                sensor_timestamp_ns=time.time_ns(),
                t_host_monotonic_ns=time.monotonic_ns(),
                exposure_us=self.exposure_us,
                gain=self.gain,
                trigger_id=self._frame_idx,
                mock=True,
                rgb=rgb,
            )
        request = self._acquire_request(blocking=blocking)
        if request is None:
            return None
        self._frame_idx += 1
        try:
            meta = request.get_metadata()
            raw_ts = meta.get("SensorTimestamp")
            if raw_ts is None:
                # The old fallback, time.time_ns(), put a WALL-clock value into the
                # camera (CLOCK_BOOTTIME) timestamp stream: epochs apart, and it
                # poisons the trigger fit. Drop the frame and count it; the
                # frame_idx gap shows where.
                self.missing_sensor_ts += 1
                self._last_poll_dropped = True
                return None
            ts = int(raw_ts)
            exposure = int(meta.get("ExposureTime", self.exposure_us) or self.exposure_us)
            gain = float(meta.get("AnalogueGain", self.gain) or self.gain)
            rgb = None
            if with_rgb:
                if self.lores_size is not None:
                    raw = request.make_array("lores")
                    rgb = apply_mount_rotation(y_plane(raw, self.lores_size))
                else:
                    raw = request.make_array("main")
                    rgb = apply_mount_rotation(y_plane(raw, self.resolution))
        finally:
            # CompletedRequest.release(); never call this on a picamera2 Job.
            if hasattr(request, "release"):
                request.release()
        return CamFrameMeta(
            cam_id=self.cam_id,
            frame_idx=self._frame_idx,
            sensor_timestamp_ns=ts,
            t_host_monotonic_ns=time.monotonic_ns(),
            exposure_us=exposure,
            gain=gain,
            mock=False,
            rgb=rgb,
        )

    def _acquire_request(self, *, blocking: bool) -> Any | None:
        """Return one CompletedRequest, or None if none is ready (non-blocking).

        picamera2 ``capture_request(wait=False)`` yields a ``Job``, not a
        ``CompletedRequest``. Keep a single outstanding job and poll it with
        ``get_result(timeout=0)`` so the capture hot loop never stalls the MCU.
        """
        picam = self._picam
        if picam is None:
            return None
        if self._pending_job is not None:
            return self._finish_job(self._pending_job, blocking=blocking)
        if blocking:
            return picam.capture_request()
        try:
            job = picam.capture_request(wait=False)
        except TypeError:
            # Very old picamera2 without wait= — grab only if a request is queued.
            if not getattr(picam, "completed_requests", None):
                return None
            return picam.capture_request()
        if hasattr(job, "release") and not hasattr(job, "get_result"):
            return job
        self._pending_job = job
        return self._finish_job(job, blocking=False)

    def _finish_job(self, job: Any, *, blocking: bool) -> Any | None:
        try:
            request = job.get_result(timeout=None if blocking else 0)
        except FuturesTimeout:
            return None
        except Exception:
            self._pending_job = None
            raise
        self._pending_job = None
        return request


def _mock_preview_rgb(cam_id: str, frame_idx: int) -> Any:
    """Tiny gray stand-in so mock sessions can exercise the preview JPEG dump."""
    import numpy as np

    img = np.full((16, 16), int(frame_idx) & 0xFF, dtype=np.uint8)
    img[0, 0] = abs(hash(cam_id)) & 0xFF
    return img


def apply_mount_rotation(arr: Any) -> Any:
    """Rotate an image array 180° in-plane (same H×W, channels last).

    Used instead of ``libcamera.Transform(hflip, vflip)`` so live preview,
    recording lores, and calibration all share one mount correction. No-op when
    ``MOUNT_ROTATE_180`` is false or ``arr`` is None.
    """
    if not MOUNT_ROTATE_180 or arr is None:
        return arr
    import numpy as np

    a = np.asarray(arr)
    if a.ndim < 2:
        return a
    return np.ascontiguousarray(a[::-1, ::-1])


def y_plane(arr: Any, size: tuple[int, int]) -> Any:
    """Copy the luma plane from a picamera2 array.

    YUV420 is packed as ``(h * 3 // 2, w)`` with Y in the first ``h`` rows.
    Already-gray ``HxW`` is sliced to ``size``. RGB/BGR ``HxWx3`` is converted
    once (fallback if a caller still requests a colour lores).
    """
    import numpy as np

    w, h = int(size[0]), int(size[1])
    a = np.asarray(arr)
    if a.ndim == 2:
        return np.array(a[:h, :w], dtype=np.uint8, copy=True)
    if a.ndim == 3 and a.shape[2] == 1:
        return np.array(a[:h, :w, 0], dtype=np.uint8, copy=True)
    if a.ndim == 3 and a.shape[2] >= 3:
        import cv2

        patch = a[:h, :w, :3] if a.shape[0] >= h and a.shape[1] >= w else a[:, :, :3]
        return cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return np.asarray(a, dtype=np.uint8)


def resolve_camera_num(device: str | int) -> int:
    """Map YAML ``device`` (``csi0``, ``0``, …) to a picamera2 camera index."""
    if isinstance(device, int):
        return device
    text = str(device).strip().lower()
    if text.isdigit():
        return int(text)
    match = re.fullmatch(r"csi(\d+)", text)
    if match:
        return int(match.group(1))
    raise ValueError(f"unsupported camera device id: {device!r}")


def libcamera_controls_from_cam(cam: CameraCapture) -> dict[str, Any]:
    """Build the libcamera control dict applied via picamera2 ``set_controls``."""
    controls: dict[str, Any] = {
        "AeEnable": bool(cam.ae_enable),
        "AwbEnable": bool(cam.awb_enable),
        "Sharpness": float(cam.sharpness),
        "Brightness": float(cam.brightness),
        "Contrast": float(cam.contrast),
        "Saturation": float(cam.saturation),
        "NoiseReductionMode": _NOISE_REDUCTION.get(cam.noise_reduction_mode, 1),
    }
    if not cam.ae_enable:
        controls["ExposureTime"] = int(cam.exposure_us)
        controls["AnalogueGain"] = float(cam.gain)
    if not cam.awb_enable:
        controls["ColourGains"] = (float(cam.awb_gains[0]), float(cam.awb_gains[1]))
    # Free-run framerate hint; ignored under IMX296 external trigger_mode=1.
    if cam.expect_rate_hz > 0:
        period_us = int(round(1_000_000.0 / float(cam.expect_rate_hz)))
        controls["FrameDurationLimits"] = (period_us, period_us)
    return controls


def _apply_camera_controls(picam: Any, cam: CameraCapture) -> None:
    controls = libcamera_controls_from_cam(cam)
    try:
        picam.set_controls(controls)
    except Exception:
        # Some controls are sensor-/pipeline-dependent; apply what remains.
        for key, value in controls.items():
            try:
                picam.set_controls({key: value})
            except Exception:
                continue


def _try_open_picamera2(cam: CameraCapture) -> bool:
    try:
        from picamera2 import Picamera2  # type: ignore[import-untyped]
    except ImportError:
        return False
    try:
        cam_num = resolve_camera_num(cam.device)
        picam = Picamera2(cam_num)
        w, h = cam.resolution
        main = {"size": (w, h), "format": cam.format}
        lores = None
        if cam.lores_size is not None:
            lw, lh = cam.lores_size
            lores = {"size": (int(lw), int(lh)), "format": str(cam.lores_format)}
        config = picam.create_video_configuration(
            main=main,
            lores=lores,
            buffer_count=int(cam.buffer_count),
        )
        picam.configure(config)
        _apply_camera_controls(picam, cam)
        cam._picam = picam
        return True
    except Exception:
        return False


def cameras_from_config(
    host_cameras: dict[str, Any],
    *,
    track_lores: bool = False,
) -> list[CameraCapture]:
    """Build ``CameraCapture`` list from host YAML.

    ``track_lores``: add a half-resolution YUV420 ``lores`` stream so ArUco and
    the viewer use the Y plane instead of copying full-res ``main``.
    """
    out: list[CameraCapture] = []
    for entry in host_cameras.get("list") or []:
        res = entry.get("resolution", [1456, 1088])
        trig = entry.get("trigger") or {}
        awb = entry.get("awb_gains", [1.5, 1.5])
        width, height = int(res[0]), int(res[1])
        lores_size = (max(1, width // 2), max(1, height // 2)) if track_lores else None
        out.append(
            CameraCapture(
                cam_id=str(entry.get("id", f"cam{len(out)}")),
                device=str(entry.get("device", "csi0")),
                resolution=(width, height),
                format=str(entry.get("format", "YUV420")),
                buffer_count=int(entry.get("buffer_count", CameraCapture.buffer_count)),
                exposure_us=int(entry.get("exposure_us", 4000)),
                gain=float(entry.get("gain", 1.0)),
                ae_enable=bool(entry.get("ae_enable", False)),
                awb_enable=bool(entry.get("awb_enable", False)),
                awb_gains=(float(awb[0]), float(awb[1])),
                sharpness=float(entry.get("sharpness", 1.0)),
                brightness=float(entry.get("brightness", 0.0)),
                contrast=float(entry.get("contrast", 1.0)),
                saturation=float(entry.get("saturation", 1.0)),
                noise_reduction_mode=str(entry.get("noise_reduction_mode", "fast")),
                expect_rate_hz=float(trig.get("expect_rate_hz", 60)),
                lores_size=lores_size,
                lores_format="YUV420",
            )
        )
    return out
