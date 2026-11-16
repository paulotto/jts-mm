# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Bridge RP2350 TRIGGER_EVENT stream into optical frame timestamps."""

from __future__ import annotations

import threading
from typing import Sequence

from optical.sync.frame_meta import CamFrameMeta


class HatSyncBridge:
    """Feed MCU triggers into :class:`ClockAligner` and map camera frames to ``t_master_us``."""

    def __init__(self, port: str | None = None) -> None:
        self._port_arg = port
        self._port: str | None = None
        self._aligner = None
        self._mcu = None
        self._lock = threading.Lock()
        self.bundle_trigger_id_mismatch = 0

    def start(self) -> str:
        """Find the HAT and begin USB bulk ingest of TRIGGER_EVENT records.

        Starts no MCU stream. This used to call ``start_mcu_streams`` — EMG,
        both IMUs *and* the trigger — so an optical-only take also streamed
        ads/imu data nobody recorded, and the trigger ran before picamera2 was
        up. The pulse is owned by :mod:`optical.sync.hat_trigger` and is armed
        before picamera2 starts; call :meth:`reset_trigger_backlog` right
        before arming.
        """
        from jtsmm.capture.capture_mcu import McuCapture
        from jtsmm.capture.serial_util import find_hat_port
        from jtsmm.sync.clock_align import ClockAligner

        self._port = find_hat_port(self._port_arg)
        self._aligner = ClockAligner()
        self._mcu = McuCapture(self._port)
        self._mcu.on_record = self._on_mcu_record
        self._mcu.start()
        return self._port

    def reset_trigger_backlog(self) -> int:
        """Drop buffered triggers so no pre-camera pulse becomes the first anchor."""
        if self._aligner is None:
            return 0
        with self._lock:
            return int(self._aligner.reset_trigger_backlog())

    def _on_mcu_record(self, rec) -> None:
        if self._aligner is None:
            return
        with self._lock:
            self._aligner.ingest_mcu(rec)

    def align_frame_bundle(
        self,
        meta_by_cam: dict[str, CamFrameMeta],
        *,
        frame_idx: int,
        cam_order: Sequence[str],
    ) -> tuple[dict[str, int], int | None]:
        """Return ``t_cam_ns`` per camera and a shared ``t_master_us`` for the bundle."""
        if self._aligner is None:
            raise RuntimeError("HatSyncBridge.start() was not called")

        t_cam_ns: dict[str, int] = {}
        t_master_us: int | None = None
        trigger_ids: set[int] = set()
        with self._lock:
            for cam_id in cam_order:
                meta = meta_by_cam.get(cam_id)
                if meta is None:
                    continue
                t_cam_ns[cam_id] = meta.sensor_timestamp_ns
                aligned = self._aligner.ingest_cam(
                    cam_id,
                    meta.sensor_timestamp_ns,
                    meta.trigger_id,
                    frame_idx=frame_idx,
                )
                if aligned is None:
                    continue
                if t_master_us is None:
                    t_master_us = aligned.t_master_us
                tid = (getattr(aligned, "payload", None) or {}).get("trigger_id")
                if tid is not None:
                    trigger_ids.add(int(tid))
            # Frames paired by SensorTimestamp should map to one pulse; if the
            # aligner disagrees, one of the two stamps is off by >= 1 period.
            if len(trigger_ids) > 1:
                self.bundle_trigger_id_mismatch += 1
        return t_cam_ns, t_master_us

    def sync_report(self) -> dict:
        if self._aligner is None:
            return {}
        with self._lock:
            report = self._aligner.sync_report()
        report["bundle_trigger_id_mismatch"] = int(self.bundle_trigger_id_mismatch)
        return report

    def stop(self) -> None:
        """Stop USB ingest (the trigger itself is stopped by ``hat_trigger``)."""
        if self._mcu is not None:
            self._mcu.stop()
            self._mcu = None
        self._aligner = None
