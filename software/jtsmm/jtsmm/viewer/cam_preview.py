# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CSI / mock preview feed → Hub JPEG slots (optical-style MJPEG source).

Opens cameras in a lightweight preview configuration (not the full-rate
recording stream) and pushes JPEG frames into ``Hub.update_camera_jpeg`` at
~preview Hz. Live frames get the same ArUco bounding-box overlay as
``optical.camera_test``. The FastAPI server exposes them as
``multipart/x-mixed-replace`` streams.
"""

from __future__ import annotations

import threading
import time
from typing import Any

# No module-level cv2: it was imported here unused, which made `jtsmm view
# --mock` (and every viewer test) need OpenCV although jpeg_util encodes via PIL
# when cv2 is absent. The live path's ArUco overlay imports cv2 lazily.

from jtsmm.capture.capture_cam import apply_mount_rotation
from jtsmm.viewer.hub import Hub
from jtsmm.viewer.jpeg_util import PREVIEW_JPEG_QUALITY, encode_jpeg_rgb, synthetic_preview_rgb

__all__ = ["CamPreviewFeed", "PREVIEW_SIZE", "PREVIEW_HZ", "PREVIEW_JPEG_QUALITY"]

PREVIEW_SIZE = (320, 240)
PREVIEW_HZ = 30.0

class CamPreviewFeed:
    """Background thread that publishes camera JPEGs into a Hub."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._picams: list[tuple[str, Any]] = []  # (cam_id, Picamera2)
        self._mock_ids: list[str] = []

    @property
    def camera_ids(self) -> list[str]:
        if self._mock_ids:
            return list(self._mock_ids)
        return [cid for cid, _ in self._picams]

    def start_live(self, hub: Hub, host_cameras: dict[str, Any]) -> list[str]:
        """Open configured CSI cams that are present; skip missing ones."""
        self.stop()
        ids: list[str] = []
        for entry in host_cameras.get("list") or []:
            cam_id = str(entry.get("id", f"cam{len(ids)}"))
            device = entry.get("device", f"csi{len(ids)}")
            picam = _try_open_preview(device, entry)
            if picam is None:
                continue
            self._picams.append((cam_id, picam))
            ids.append(cam_id)
        hub.set_camera_ids(ids)
        if ids:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_live, args=(hub,), name="cam-preview", daemon=True
            )
            self._thread.start()
        return ids

    def start_mock(self, hub: Hub, n_cameras: int = 2) -> list[str]:
        """Synthetic animated previews (no hardware)."""
        self.stop()
        n = max(0, int(n_cameras))
        ids = [f"cam{i}" for i in range(n)]
        self._mock_ids = ids
        hub.set_camera_ids(ids)
        if ids:
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_mock, args=(hub,), name="cam-preview-mock", daemon=True
            )
            self._thread.start()
        return ids

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for _, picam in self._picams:
            try:
                picam.stop()
                picam.close()
            except Exception:
                pass
        self._picams.clear()
        self._mock_ids = []

    def _run_live(self, hub: Hub) -> None:
        from jtsmm.viewer.aruco_overlay import draw_aruco_overlay_rgb

        period = 1.0 / PREVIEW_HZ
        while not self._stop.is_set():
            t0 = time.monotonic()
            for cam_id, picam in self._picams:
                try:
                    frame = picam.capture_array()
                    if frame.ndim != 3 or frame.shape[2] < 3:
                        continue
                    frame = apply_mount_rotation(frame)
                    # picamera2 RGB888 → ArUco overlay (optical camera_test) → JPEG
                    rgb = draw_aruco_overlay_rgb(frame[:, :, :3])
                    hub.update_camera_jpeg(
                        cam_id, encode_jpeg_rgb(rgb, quality=PREVIEW_JPEG_QUALITY)
                    )
                except Exception:
                    continue
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)

    def _run_mock(self, hub: Hub) -> None:
        period = 1.0 / PREVIEW_HZ
        t = 0.0
        while not self._stop.is_set():
            for cam_id in self._mock_ids:
                rgb = synthetic_preview_rgb(cam_id, size=PREVIEW_SIZE, t=t)
                hub.update_camera_jpeg(
                    cam_id, encode_jpeg_rgb(rgb, quality=PREVIEW_JPEG_QUALITY)
                )
            t += period
            time.sleep(period)


def _try_open_preview(device: Any, cam_entry: dict[str, Any] | None = None) -> Any | None:
    try:
        from types import SimpleNamespace

        from picamera2 import Picamera2  # type: ignore[import-untyped]
        from jtsmm.capture.capture_cam import libcamera_controls_from_cam, resolve_camera_num
    except Exception:
        return None
    try:
        cam_num = resolve_camera_num(device)
        picam = Picamera2(cam_num)
        w, h = PREVIEW_SIZE
        cfg = picam.create_preview_configuration(main={"size": (w, h), "format": "BGR888"})
        picam.configure(cfg)
        if cam_entry is not None:
            # Build controls from the YAML entry without constructing CameraCapture —
            # CameraCapture.__post_init__ opens a second Picamera2 on the same
            # sensor and leaves the first instance half-initialized (allocator missing).
            trig = cam_entry.get("trigger") or {}
            awb = cam_entry.get("awb_gains", [1.5, 1.5])
            controls = libcamera_controls_from_cam(
                SimpleNamespace(
                    ae_enable=bool(cam_entry.get("ae_enable", False)),
                    awb_enable=bool(cam_entry.get("awb_enable", False)),
                    awb_gains=(float(awb[0]), float(awb[1])),
                    exposure_us=int(cam_entry.get("exposure_us", 4000)),
                    gain=float(cam_entry.get("gain", 1.0)),
                    sharpness=float(cam_entry.get("sharpness", 1.0)),
                    brightness=float(cam_entry.get("brightness", 0.0)),
                    contrast=float(cam_entry.get("contrast", 1.0)),
                    saturation=float(cam_entry.get("saturation", 1.0)),
                    noise_reduction_mode=str(cam_entry.get("noise_reduction_mode", "fast")),
                    expect_rate_hz=float(trig.get("expect_rate_hz", 60)),
                )
            )
            try:
                picam.set_controls(controls)
            except Exception:
                for key, value in controls.items():
                    try:
                        picam.set_controls({key: value})
                    except Exception:
                        continue
        picam.start()
        return picam
    except Exception:
        return None
