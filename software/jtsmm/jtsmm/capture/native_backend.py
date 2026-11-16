# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Adapters for the native C++/libusb capture + libhdf5 writer path."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from jtsmm import native
from jtsmm._version import __version__
from jtsmm.config.catalog import DEFAULT_EMG_SAMPLE_RATE_SPS, FULL_CHANNEL_NAMES, channel_lsb_uv
from jtsmm.sync.clock_align import AlignedRecord


class NativeCapture:
    """Wraps ``NativeMcuCapture`` to mirror ``McuCapture``'s start/iter/stop API."""

    def __init__(self, port: str | None = None, *, queue_max: int = 65536) -> None:
        self.port = port
        self._cap = native.NativeMcuCapture(queue_max)

    def start(self) -> None:
        self._cap.start()

    def stop(self) -> None:
        self._cap.stop()

    def settle(self, *, quiet_ms: int = 250) -> None:
        """Drain stale USB data after streams start (cold-start EMG gap mitigation)."""
        deadline = time.monotonic() + quiet_ms / 1000.0
        while time.monotonic() < deadline:
            self.discard_queued()
            time.sleep(0.02)

    def iter_records(self, timeout_s: float = 0.2) -> Iterator[Any]:
        deadline = time.monotonic() + timeout_s if timeout_s > 0 else None
        while True:
            wait = 0.05
            if deadline is not None:
                wait = max(0.0, min(wait, deadline - time.monotonic()))
                if wait == 0.0:
                    return
            rec = self._cap.next_record(wait)
            if rec is None:
                if deadline is not None and time.monotonic() >= deadline:
                    return
                continue
            yield rec

    def drain_records(self, limit: int | None = None) -> list[Any]:
        """Pop up to ``limit`` parsed records (all of them when ``limit`` is None).

        The limit matters: a caller that drains everything and then processes only
        the first N *discards* the remainder, because they have already left the
        queue. Bounding the pop instead leaves the rest for the next slice.
        """
        out: list[Any] = []
        while limit is None or len(out) < limit:
            rec = self._cap.next_record(0.0)
            if rec is None:
                break
            out.append(rec)
        return out

    def discard_queued(self) -> None:
        self._cap.discard_queued_records()

    def pump_to_writer(self, writer: NativeWriter, *, max_records: int = 8192) -> dict[str, Any]:
        return dict(self._cap.pump_to_writer(writer._writer, max_records))

    def set_viewer_tap_enabled(self, enabled: bool, *, bucket_samples: int = 4) -> None:
        """Enable/disable the native decimated viewer tap (in-recording viewing)."""
        self._cap.set_viewer_tap_enabled(enabled, bucket_samples)

    def get_viewer_frame(self) -> Any:
        """Snapshot + reset the decimated viewer accumulators (see ViewerTap)."""
        return self._cap.get_viewer_frame()

    @property
    def crc_errors(self) -> int:
        return int(self._cap.crc_errors)

    @property
    def queue_drops(self) -> dict[str, int]:
        """Host-side losses: records/chunks discarded because a queue overflowed."""
        return {
            "raw_chunks": int(getattr(self._cap, "raw_chunks_dropped", 0) or 0),
            "records": int(getattr(self._cap, "records_dropped", 0) or 0),
        }

    def _link_counters(self) -> dict[str, Any]:
        fn = getattr(self._cap, "link_counters", None)
        return dict(fn()) if fn is not None else {}

    @property
    def link_stats(self) -> dict[str, int]:
        """Cumulative link counters; still valid after ``stop()`` (sync report)."""
        usb = self._link_counters()
        out = {
            "bytes_rx": int(getattr(self._cap, "bytes_rx", 0) or 0),
            "records_rx": int(getattr(self._cap, "records_rx", 0) or 0),
            "crc_errors": int(getattr(self._cap, "crc_errors", 0) or 0),
            "resyncs": int(getattr(self._cap, "resyncs", 0) or 0),
        }
        for key in (
            "usb_errors",
            "usb_stalls",
            "usb_overflows",
            "usb_no_device",
            "usb_partial_bytes",
            "usb_lost_bytes",
            "usb_resubmit_failures",
        ):
            out[key] = int(usb.get(key, 0) or 0)
        out["device_lost"] = int(bool(usb.get("device_lost", False)))
        return out

    def link_health(self) -> dict[str, Any]:
        """Liveness snapshot for a capture watchdog.

        ``device_lost``: no USB transfer is armed any more while capturing (unplug,
        endpoint stall, error storm) -- no further data can arrive.
        ``last_record_age_s``: seconds since the parser last produced a record
        (since ``start()`` if none yet).
        """
        usb = self._link_counters()
        return {
            "device_lost": bool(usb.get("device_lost", False)),
            "usb_errors": int(usb.get("usb_errors", 0) or 0),
            "usb_stalls": int(usb.get("usb_stalls", 0) or 0),
            "bytes_rx": int(usb.get("bytes_rx", getattr(self._cap, "bytes_rx", 0)) or 0),
            "last_record_age_s": float(usb.get("last_record_age_s", 0.0) or 0.0),
        }


def _imu_report_rates(slot_cfg: dict[str, Any]) -> dict[str, int]:
    """Enabled report -> rate_hz, exactly as hdf5_writer.py derives its rate attrs."""
    return {
        str(name): int(entry.get("rate_hz", 0) or 0)
        for name, entry in (slot_cfg.get("reports") or {}).items()
        if isinstance(entry, dict) and entry.get("enabled", True)
    }


class NativeWriter:
    """Wraps ``NativeHdf5Writer`` for supervisor-driven multimodal sessions."""

    def __init__(
        self,
        *,
        path: Path,
        config_hash: str,
        mcu_config: dict[str, Any],
        host_config: dict[str, Any],
        firmware_version: str | None = None,
        session_meta: dict[str, Any] | None = None,
    ) -> None:
        emg = mcu_config.get("emg", {})
        # "mode" has its own dedicated constructor arg (kept for backward
        # compatibility); every other session_meta key -- including custom
        # `--meta key=value` entries from `jtsmm record` -- is written verbatim
        # as its own /meta string attr, matching Hdf5Writer's flattening.
        extra_meta = {
            str(k): str(v) for k, v in (session_meta or {}).items() if k != "mode"
        }
        gain = int(emg.get("default_gain", 6))
        self._writer = native.NativeHdf5Writer(
            path=str(path),
            config_hash=config_hash,
            mcu_config_json=json.dumps(mcu_config, separators=(",", ":")),
            host_config_json=json.dumps(host_config, separators=(",", ":")),
            software_version=__version__,
            schema_version=int(mcu_config.get("schema_version", 1)),
            firmware_version=firmware_version or "",
            mode=str((session_meta or {}).get("mode", "acquisition")),
            emg_sample_rate_sps=int(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS)),
            emg_gain=gain,
            channel_names=list(FULL_CHANNEL_NAMES),
            # CH7/CH8 run at a fixed gain; one scalar LSB mis-scaled them (K5).
            lsb_uv_per_channel=channel_lsb_uv(gain, len(FULL_CHANNEL_NAMES)),
            imu_rates_mta=_imu_report_rates(mcu_config.get("imu_mta", {})),
            imu_rates_cra=_imu_report_rates(mcu_config.get("imu_cra", {})),
            fusion_mta=mcu_config.get("imu_mta", {}).get("fusion", {}).get(
                "primary", "game_rotation_vector"
            ),
            fusion_cra=mcu_config.get("imu_cra", {}).get("fusion", {}).get(
                "primary", "game_rotation_vector"
            ),
            trigger_rate_hz=mcu_config.get("trigger", {}).get("rate_hz"),
            extra_meta=extra_meta,
        )
        self._open = False

    def open(self) -> None:
        self._writer.open()
        self._open = True

    @property
    def has_open_file(self) -> bool:
        return self._open

    def append_camera(self, rec: AlignedRecord) -> None:
        payload = rec.payload
        raw_tid = payload.get("trigger_id", 0)
        # Defense in depth: ClockAligner always writes a concrete int on success,
        # but coerce None / missing so int(None) cannot crash the recorder.
        trigger_id = 0 if raw_tid is None else int(raw_tid)
        self._writer.append_camera(
            str(payload.get("cam_id", "cam0")),
            int(rec.t_master_us),
            int(payload.get("sensor_timestamp_ns", 0)),
            trigger_id,
            int(payload.get("frame_idx", 0)),
            str(payload.get("video_file", "") or ""),
        )

    def append_event(
        self,
        *,
        t_master_us: int,
        kind: str,
        text: str,
        phase_kind: str = "",
        t_host_ns: int = 0,
        map_quality: str = "",
        viewer_t_s: float | None = None,
    ) -> int:
        """Same contract as ``Hdf5Writer.append_event`` (returns the /events row)."""
        return int(self._writer.append_event(
            t_master_us=int(t_master_us), kind=str(kind), text=str(text),
            phase_kind=str(phase_kind), t_host_ns=int(t_host_ns),
            map_quality=str(map_quality),
            viewer_t_s=None if viewer_t_s is None else float(viewer_t_s),
        ))

    def set_event_time(self, row: int, *, t_master_us: int, map_quality: str) -> None:
        self._writer.set_event_time(int(row), int(t_master_us), str(map_quality))

    def flush(self) -> None:
        self._writer.flush()

    def write_sync_report(self, report: dict[str, Any]) -> None:
        self._writer.set_sync_report(json.dumps(report, separators=(",", ":")))

    def write_sync_drops(self, drops: dict[str, int]) -> None:
        self._writer.write_sync_drops_json(json.dumps(drops, separators=(",", ":")))

    def close(self) -> None:
        if self._open:
            # The native close() releases the file even when it raises (disk
            # full); mark it closed first so the error is not retried/re-raised.
            self._open = False
            self._writer.close()
