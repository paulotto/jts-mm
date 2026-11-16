# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Multimodal HDF5 session writer (s02 §8)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm._version import __version__
from jtsmm.config.catalog import FULL_CHANNEL_NAMES, channel_lsb_uv, lsb_uv as _lsb_uv
from jtsmm.imu import units as imu_units
from jtsmm.io.file_hashes import sha256_file
from jtsmm.protocol.records import FUSION_REPORT_IDS
from jtsmm.sync.clock_align import AlignedRecord

# Muscle channels CH1-6 only (bit0=CH1); mirrors jtsmm.viewer.leadoff's bit
# convention. Kept as a self-contained helper (not imported from `viewer`) so
# `jtsmm.io` -- a lower layer that `viewer` itself reads sessions through --
# does not depend on it.
_MUSCLE_LOFF_MASK = 0x3F


def _muscle_leadoff_mask(status: bytes | bytearray | None) -> tuple[int, int]:
    """Muscle-only (statp_mask, statn_mask) from the 3-byte ADS1299 status word.

    DRL (CH8/IN8N, bit 7) is intentionally dropped -- its comparator needs the
    six muscle channels muted to be meaningful (see
    ``jtsmm.emg.check.run_emg_check``), so a raw per-sample bit here would be
    misleading rather than informative.
    """
    if not status or len(status) < 3:
        return 0, 0
    b0, b1, b2 = status[0], status[1], status[2]
    statp = ((b0 & 0x0F) << 4) | (b1 >> 4)
    statn = ((b1 & 0x0F) << 4) | (b2 >> 4)
    return statp & _MUSCLE_LOFF_MASK, statn & _MUSCLE_LOFF_MASK


# Mirrors the native writer's attribute text (hdf5_session.cpp::flush_emg) --
# keep the two in sync if this wording changes.
_LEADOFF_STATUS_NOTE = (
    "columns=[statp_mask,statn_mask] from the ADS1299 status word; bit i = "
    "CH(i+1), muscle CH1-6 only (bits 6-7 always 0). 1 = lead-off comparator "
    "tripped (that input/leg reads as disconnected). All-zero for the whole "
    "session unless mcu.emg.leadoff.during_recording was true (firmware "
    "clears the LOFF registers otherwise, so the bits are stale, not "
    "'connected'). DRL (CH8/IN8N) is intentionally excluded: its comparator "
    "shares a current-return path with the six muscle channels' own "
    "source+sink pairs, which masks a genuine DRL disconnect while they are "
    "live -- use `jtsmm emg check` (isolates DRL by muting CH1-6 first) for "
    "DRL contact instead."
)

# Raw motion-vector streams persisted per IMU slot: (payload key, dataset name,
# RAW sensor_id for SI conversion, SI unit label, config report name for rate).
# BNO085 per-report calibration status (SH-2 status bits 1:0): 0 unreliable,
# 1 low, 2 medium, 3 high. Stored next to every IMU stream as <name>_accuracy,
# row-aligned; ACCURACY_ABSENT marks a record that carried no status byte.
ACCURACY_ABSENT = 255
ACCURACY_NOTE = (
    "BNO085 calibration status per sample: 0 unreliable, 1 low, 2 medium, 3 high; "
    "255 = not sent (older firmware)"
)

_IMU_MOTION_STREAMS = (
    ("accel", "accel", 0x01, "m/s^2", "accelerometer"),
    ("gyro", "gyro", 0x02, "rad/s", "gyroscope"),
    ("mag", "mag", 0x03, "uT", "magnetometer"),
)


@dataclass
class Hdf5Writer:
    path: Path
    config_hash: str
    mcu_config: dict[str, Any]
    host_config: dict[str, Any]
    session_meta: dict[str, Any] = field(default_factory=dict)
    firmware_version: str | None = None
    sidecar_paths: dict[str, Path] = field(default_factory=dict)
    _file: h5py.File | None = None
    _opened_at: float = 0.0
    _emg_samples: list[np.ndarray] = field(default_factory=list)
    _emg_t: list[np.ndarray] = field(default_factory=list)
    _emg_loff: list[tuple[int, int]] = field(default_factory=list)
    # K1 acquisition sequence per EMG row (None when the firmware sends none).
    _emg_seq: list[int | None] = field(default_factory=list)
    # h5py.File.flush() (H5Fflush) at most this often, and at close. Without it
    # the on-disk metadata only describes the datasets as they were created, so
    # a recorder killed mid-take (SSH drop, OOM, power) left an unreadable file.
    file_flush_interval_s: float = 1.0
    _last_file_flush: float = 0.0
    _trig_id: list[int] = field(default_factory=list)
    _trig_t: list[int] = field(default_factory=list)
    # Per-slot IMU buffers. `_imu[slot]` maps a stream name (quat/accel/gyro/mag/
    # linear_accel) to a list of (t_master_us, values...) tuples; quat rows also
    # carry the fusion report_id so multiple fusion types can coexist.
    _imu: dict[str, dict[str, list]] = field(
        default_factory=lambda: {"imu_mta": {}, "imu_cra": {}}
    )
    _cam_meta: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _optical_rows: list[dict[str, Any]] = field(default_factory=list)
    # /optical/corners camera index -> cam_id, in order of first appearance.
    _corner_cams: list[str] = field(default_factory=list)
    _events: list[dict[str, Any]] = field(default_factory=list)
    _events_flushed: int = 0  # /events rows already on disk

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "w", libver="latest")
        self._opened_at = time.time()

        meta = self._file.create_group("meta")
        meta.attrs["software_version"] = __version__
        meta.attrs["schema_version"] = int(self.mcu_config.get("schema_version", 1))
        meta.attrs["jts_schema_version"] = "1.1"
        meta.attrs["created_unix_utc"] = self._opened_at
        # Human-readable recording date/time alongside created_unix_utc.
        meta.attrs["recording_date"] = (
            datetime.fromtimestamp(self._opened_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        meta.attrs["config_hash"] = self.config_hash
        meta.attrs["mcu_config_json"] = json.dumps(self.mcu_config, separators=(",", ":"))
        meta.attrs["host_config_json"] = json.dumps(self.host_config, separators=(",", ":"))
        if self.firmware_version:
            meta.attrs["firmware_version"] = self.firmware_version
        for k, v in self.session_meta.items():
            meta.attrs[k] = v

        self._file.create_group("emg")
        self._file.create_group("imu_mta")
        self._file.create_group("imu_cra")
        self._file.create_group("cameras")
        self._file.create_group("optical")
        self._file.create_group("triggers")
        self._file.create_group("events")
        self._file.create_group("sync")

    def append(self, rec: AlignedRecord) -> None:
        if rec.modality == "emg":
            ch = rec.payload.get("channels")
            if ch:
                self._emg_samples.append(np.asarray(ch, dtype=np.int32))
                self._emg_t.append(np.int64(rec.t_master_us))
                self._emg_loff.append(_muscle_leadoff_mask(rec.payload.get("status")))
                seq = rec.payload.get("seq")
                self._emg_seq.append(int(seq) & 0xFFFFFFFF if seq is not None else None)
        elif rec.modality == "trigger":
            self._trig_id.append(int(rec.payload.get("trigger_id", 0)))
            self._trig_t.append(int(rec.t_master_us))
        elif rec.modality in ("imu_mta", "imu_cra"):
            self._append_imu(rec.modality, rec.t_master_us, rec.payload)
        elif rec.modality == "camera":
            cid = str(rec.payload.get("cam_id", "cam0"))
            row = dict(rec.payload)
            row["t_master_us"] = rec.t_master_us
            self._cam_meta.setdefault(cid, []).append(row)

    def _append_imu(self, slot: str, t_master_us: int, payload: dict[str, Any]) -> None:
        buf = self._imu[slot]
        def acc(name: str) -> int:
            v = payload.get(f"{name}_accuracy")
            return ACCURACY_ABSENT if v is None else int(v) & 0x03

        if "quat" in payload:
            report_id = payload.get("quat_report_id")
            buf.setdefault("quat", []).append(
                (int(t_master_us), payload["quat"], report_id, acc("quat")))
        if "linear_accel" in payload:
            buf.setdefault("linear_accel", []).append(
                (int(t_master_us), payload["linear_accel"], acc("linear_accel")))
        for key, _dsname, _sid, _unit, _rname in _IMU_MOTION_STREAMS:
            if key in payload:
                buf.setdefault(key, []).append((int(t_master_us), payload[key], acc(key)))

    def flush(self) -> None:
        if self._file is None:
            return
        emg_g = self._file["emg"]
        if self._emg_samples:
            samples = np.vstack(self._emg_samples)
            t_us = np.asarray(self._emg_t, dtype=np.uint64)
            loff = np.asarray(self._emg_loff, dtype=np.uint8).reshape(-1, 2)
            gain = int(self.mcu_config.get("emg", {}).get("default_gain", 6))
            fs = int(self.mcu_config.get("emg", {}).get("sample_rate_sps", 1000))
            lsb = _lsb_uv(gain)
            # Per column: CH7/CH8 run at a fixed gain of 24 (K5).
            lsb_cols = np.asarray(channel_lsb_uv(gain, samples.shape[1]), dtype=np.float64)
            # float32 x float32, exactly like the native writer (flush_emg), so
            # both writers produce bit-identical samples_uv by construction.
            # (float64 math then a cast only matched because every catalog LSB
            # happens to be exact in float32.)
            uv = samples.astype(np.float32) * lsb_cols.astype(np.float32)[None, :]
            if "samples" not in emg_g:
                emg_g.create_dataset("samples", data=samples, maxshape=(None, 8), chunks=True)
                emg_g.create_dataset("samples_uv", data=uv, maxshape=(None, 8))
                emg_g.create_dataset("t_master_us", data=t_us, maxshape=(None,), chunks=True)
                emg_g.create_dataset("leadoff_status", data=loff, maxshape=(None, 2), chunks=True)
                emg_g.attrs["sample_rate_sps"] = fs
                emg_g.attrs["gain"] = gain
                emg_g.attrs["lsb_uv"] = lsb
                emg_g.attrs["lsb_uv_per_channel"] = lsb_cols
                emg_g.attrs["channel_names"] = np.array(FULL_CHANNEL_NAMES, dtype="S")
                emg_g.attrs["leadoff_status_note"] = _LEADOFF_STATUS_NOTE
                if all(q is not None for q in self._emg_seq):
                    # /emg/seq only when the firmware sends it (K1).
                    emg_g.create_dataset(
                        "seq", data=np.asarray(self._emg_seq, dtype=np.uint32),
                        maxshape=(None,), chunks=True,
                    )
            else:
                self._resize_append(emg_g["samples"], samples)
                self._resize_append(emg_g["samples_uv"], uv)
                self._resize_append(emg_g["t_master_us"], t_us)
                self._resize_append(emg_g["leadoff_status"], loff)
                if "seq" in emg_g:
                    # Rows must stay aligned; a row without seq (cannot happen
                    # with one firmware) is stored as 0 and shows up as a gap.
                    seq = np.asarray([q if q is not None else 0 for q in self._emg_seq],
                                     dtype=np.uint32)
                    self._resize_append(emg_g["seq"], seq)
            self._emg_samples.clear()
            self._emg_t.clear()
            self._emg_loff.clear()
            self._emg_seq.clear()

        trig_g = self._file["triggers"]
        if self._trig_id:
            tid = np.asarray(self._trig_id, dtype=np.uint64)
            tt = np.asarray(self._trig_t, dtype=np.uint64)
            if "trigger_id" not in trig_g:
                trig_g.create_dataset("trigger_id", data=tid, maxshape=(None,), chunks=True)
                trig_g.create_dataset("t_master_us", data=tt, maxshape=(None,), chunks=True)
                rate = self.mcu_config.get("trigger", {}).get("rate_hz")
                if rate is not None:
                    trig_g.attrs["rate_hz"] = rate
                trig_g.attrs["source"] = "mcu_pio"
            else:
                self._resize_append(trig_g["trigger_id"], tid)
                self._resize_append(trig_g["t_master_us"], tt)
            self._trig_id.clear()
            self._trig_t.clear()

        for slot in ("imu_mta", "imu_cra"):
            self._flush_imu_slot(slot)

        cam_root = self._file["cameras"]
        for cid, rows in self._cam_meta.items():
            if not rows:
                continue
            grp = cam_root.require_group(cid)
            t_arr = np.asarray([r.get("t_master_us", 0) for r in rows], dtype=np.uint64)
            s_arr = np.asarray([r.get("sensor_timestamp_ns", 0) for r in rows], dtype=np.uint64)
            tr_arr = np.asarray([r.get("trigger_id", 0) for r in rows], dtype=np.uint64)
            fidx = np.asarray([r.get("frame_idx", 0) for r in rows], dtype=np.uint64)
            if "t_master_us" not in grp:
                grp.create_dataset("t_master_us", data=t_arr, maxshape=(None,), chunks=True)
                grp.create_dataset("sensor_timestamp_ns", data=s_arr, maxshape=(None,), chunks=True)
                grp.create_dataset("trigger_id", data=tr_arr, maxshape=(None,), chunks=True)
                # Always: a first block whose indices were all 0 used to skip the
                # dataset for the whole take.
                grp.create_dataset("frame_idx", data=fidx, maxshape=(None,), chunks=True)
                if rows[0].get("video_file"):
                    grp.attrs["video_file"] = rows[0]["video_file"]
            else:
                self._resize_append(grp["t_master_us"], t_arr)
                self._resize_append(grp["sensor_timestamp_ns"], s_arr)
                self._resize_append(grp["trigger_id"], tr_arr)
                if "frame_idx" in grp:
                    self._resize_append(grp["frame_idx"], fidx)
            self._cam_meta[cid] = []

        optical_g = self._file["optical"]
        if self._optical_rows:
            # Corners point at their /optical row: count the rows before appending.
            base_row = int(optical_g["t_master_us"].shape[0]) if "t_master_us" in optical_g else 0
            self._flush_optical_corners(optical_g, self._optical_rows, base_row)
            t_arr = np.asarray([r["t_master_us"] for r in self._optical_rows], dtype=np.uint64)
            pose_mta = np.asarray([r["pose_mta"] for r in self._optical_rows], dtype=np.float32)
            pose_cra = np.asarray([r["pose_cra"] for r in self._optical_rows], dtype=np.float32)
            pose_ct = np.asarray([r["pose_ct"] for r in self._optical_rows], dtype=np.float32)
            if "t_master_us" not in optical_g:
                optical_g.create_dataset("t_master_us", data=t_arr, maxshape=(None,), chunks=True)
                optical_g.create_dataset("pose_mta", data=pose_mta, maxshape=(None, 4, 4), chunks=True)
                optical_g.create_dataset("pose_cra", data=pose_cra, maxshape=(None, 4, 4), chunks=True)
                optical_g.create_dataset("pose_ct", data=pose_ct, maxshape=(None, 4, 4), chunks=True)
            else:
                self._resize_append(optical_g["t_master_us"], t_arr)
                self._resize_append(optical_g["pose_mta"], pose_mta)
                self._resize_append(optical_g["pose_cra"], pose_cra)
                self._resize_append(optical_g["pose_ct"], pose_ct)
            self._flush_optical_quality(optical_g, self._optical_rows)
            self._optical_rows.clear()

        self._flush_events()
        now = time.monotonic()
        if now - self._last_file_flush >= float(self.file_flush_interval_s):
            self._file.flush()
            self._last_file_flush = now

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
        """Append a session event (phase start/end, viewer note, ...) on the master clock.

        ``viewer_t_s`` is the viewer's own display time of a note (NaN for other
        events). Returns the event's row in ``/events`` (for `set_event_time`).
        Mirrored by the native writer (hdf5_session.cpp::append_event).
        """
        self._events.append(
            {
                "t_master_us": int(t_master_us),
                "kind": str(kind),
                "text": str(text),
                "phase_kind": str(phase_kind),
                "t_host_ns": int(t_host_ns),
                "map_quality": str(map_quality),
                "viewer_t_s": float("nan") if viewer_t_s is None else float(viewer_t_s),
            }
        )
        return self._events_flushed + len(self._events) - 1

    def set_event_time(self, row: int, *, t_master_us: int, map_quality: str) -> None:
        """Re-stamp event ``row`` (already on disk or still buffered).

        The supervisor writes viewer notes as they arrive (so a killed recorder
        keeps them) and re-maps their host times with the final clock fit at close.
        """
        row = int(row)
        if row < 0 or row >= self._events_flushed + len(self._events):
            raise IndexError(f"no /events row {row}")
        if row >= self._events_flushed:
            ev = self._events[row - self._events_flushed]
            ev["t_master_us"], ev["map_quality"] = int(t_master_us), str(map_quality)
            return
        assert self._file is not None
        ev_g = self._file["events"]
        ev_g["t_master_us"][row] = np.uint64(t_master_us)
        ev_g["map_quality"][row] = str(map_quality)

    def write_phases_table(self, phases: list[dict[str, Any]]) -> None:
        """Write completed phase intervals under ``/phases`` (attrs + datasets)."""
        if self._file is None or not phases:
            return
        if "phases" in self._file:
            del self._file["phases"]
        grp = self._file.create_group("phases")
        names = [str(p.get("name", "")) for p in phases]
        kinds = [str(p.get("kind", "")) for p in phases]
        t0 = np.asarray([int(p["t_start_us"]) for p in phases], dtype=np.uint64)
        t1 = np.asarray([int(p["t_end_us"]) for p in phases], dtype=np.uint64)
        grp.create_dataset("name", data=np.asarray(names, dtype=h5py.string_dtype()))
        grp.create_dataset("kind", data=np.asarray(kinds, dtype=h5py.string_dtype()))
        grp.create_dataset("t_start_us", data=t0)
        grp.create_dataset("t_end_us", data=t1)
        grp.attrs["n_phases"] = len(phases)
        grp.attrs["phases_json"] = json.dumps(phases, separators=(",", ":"))

    def _flush_events(self) -> None:
        if self._file is None or not self._events:
            return
        ev_g = self._file["events"]
        t_arr = np.asarray([e["t_master_us"] for e in self._events], dtype=np.uint64)
        kinds = [e["kind"] for e in self._events]
        texts = [e["text"] for e in self._events]
        phase_kinds = [e["phase_kind"] for e in self._events]
        host = np.asarray([e["t_host_ns"] for e in self._events], dtype=np.uint64)
        quals = [e["map_quality"] for e in self._events]
        view_t = np.asarray([e["viewer_t_s"] for e in self._events], dtype=np.float64)
        str_dt = h5py.string_dtype()
        if "t_master_us" not in ev_g:
            ev_g.create_dataset("t_master_us", data=t_arr, maxshape=(None,), chunks=True)
            ev_g.create_dataset("kind", data=np.asarray(kinds, dtype=str_dt), maxshape=(None,), chunks=True)
            ev_g.create_dataset("text", data=np.asarray(texts, dtype=str_dt), maxshape=(None,), chunks=True)
            ev_g.create_dataset(
                "phase_kind", data=np.asarray(phase_kinds, dtype=str_dt), maxshape=(None,), chunks=True
            )
            ev_g.create_dataset("t_host_ns", data=host, maxshape=(None,), chunks=True)
            ev_g.create_dataset(
                "map_quality", data=np.asarray(quals, dtype=str_dt), maxshape=(None,), chunks=True
            )
            ev_g.create_dataset("viewer_t_s", data=view_t, maxshape=(None,), chunks=True)
        else:
            self._resize_append(ev_g["t_master_us"], t_arr)
            self._resize_append_str(ev_g["kind"], kinds)
            self._resize_append_str(ev_g["text"], texts)
            self._resize_append_str(ev_g["phase_kind"], phase_kinds)
            self._resize_append(ev_g["t_host_ns"], host)
            self._resize_append_str(ev_g["map_quality"], quals)
            self._resize_append(ev_g["viewer_t_s"], view_t)
        self._events_flushed += len(self._events)
        self._events.clear()

    def append_optical(
        self,
        *,
        t_master_us: int,
        pose_mta: np.ndarray,
        pose_cra: np.ndarray,
        pose_ct: np.ndarray,
        quality: dict[str, Any] | None = None,
        corners: list[tuple[str, int, Any]] | None = None,
    ) -> None:
        """One tracked frame. ``corners``: the 2D detections the poses were
        estimated from, as ``(cam_id, tag_id, (4, 2) pixels)`` (see /optical/corners)."""
        self._optical_rows.append(
            {
                "t_master_us": int(t_master_us),
                "pose_mta": np.asarray(pose_mta, dtype=np.float32),
                "pose_cra": np.asarray(pose_cra, dtype=np.float32),
                "pose_ct": np.asarray(pose_ct, dtype=np.float32),
                "quality": dict(quality or {}),
                "corners": list(corners or ()),
            }
        )

    def set_optical_calibration(self, cameras: dict[str, dict[str, Any]]) -> None:
        """Store the camera model the live tracker used, so /optical/corners can be
        re-triangulated offline without the calibration bundle.

        ``cameras[cam_id]``: ``K`` (3, 3), ``dist`` (N,), ``image_size`` (w, h) of the
        image the corners are in, ``T_world_cam`` (4, 4), optional ``rms_px`` /
        ``extrinsics_rms_px``. Writing again replaces the stored set.
        """
        assert self._file is not None, "open() first"
        cg = self._file["optical"].require_group("corners")
        if "calibration" in cg:
            del cg["calibration"]
        cal = cg.create_group("calibration")
        cal.attrs["note"] = (
            "camera model of the live tracker: K/dist describe the image the corners "
            "are in (image_size, w x h); T_world_cam maps camera to world (= cam0) "
            "coordinates in mm"
        )
        for cam_id, c in cameras.items():
            g = cal.create_group(str(cam_id))
            g.create_dataset("K", data=np.asarray(c["K"], dtype=np.float64).reshape(3, 3))
            g.create_dataset("dist", data=np.asarray(c["dist"], dtype=np.float64).reshape(-1))
            g.create_dataset(
                "T_world_cam", data=np.asarray(c["T_world_cam"], dtype=np.float64).reshape(4, 4))
            g.attrs["image_size"] = np.asarray(c["image_size"], dtype=np.int64).reshape(2)
            for key in ("rms_px", "extrinsics_rms_px"):
                if c.get(key) is not None:
                    g.attrs[key] = float(c[key])
            self._corner_cam_index(str(cam_id))
        cg.attrs["cam_ids"] = np.array(self._corner_cams, dtype="S")

    def _corner_cam_index(self, cam_id: str) -> int:
        if cam_id not in self._corner_cams:
            self._corner_cams.append(cam_id)
        return self._corner_cams.index(cam_id)

    def _flush_optical_corners(
        self, optical_g: h5py.Group, rows: list[dict[str, Any]], base_row: int
    ) -> None:
        """Append /optical/corners: one row per (frame, camera, tag)."""
        opt_row: list[int] = []
        t_us: list[int] = []
        cam: list[int] = []
        tag: list[int] = []
        pts: list[np.ndarray] = []
        for i, row in enumerate(rows):
            for cam_id, tag_id, corners in row.get("corners") or ():
                opt_row.append(base_row + i)
                t_us.append(int(row["t_master_us"]))
                cam.append(self._corner_cam_index(str(cam_id)))
                tag.append(int(tag_id))
                pts.append(np.asarray(corners, dtype=np.float32).reshape(4, 2))
        if not pts:
            return
        cg = optical_g.require_group("corners")
        cols = {
            "optical_row": np.asarray(opt_row, dtype=np.uint32),
            "t_master_us": np.asarray(t_us, dtype=np.uint64),
            "cam": np.asarray(cam, dtype=np.uint8),
            "tag_id": np.asarray(tag, dtype=np.uint16),
            "corners_px": np.stack(pts),
        }
        if "corners_px" not in cg:
            for name, arr in cols.items():
                cg.create_dataset(name, data=arr, maxshape=(None, *arr.shape[1:]), chunks=True)
            cg.attrs["note"] = (
                "2D ArUco corners the live tracker estimated /optical poses from, one row "
                "per (frame, camera, tag). optical_row indexes /optical/t_master_us etc.; "
                "cam indexes the cam_ids attribute; corners_px (x, y) in the image the "
                "tracker saw (see calibration/<cam>/K and image_size), in OpenCV ArUco "
                "order: top-left, top-right, bottom-right, bottom-left of the tag. All "
                "detected tags, including ones the pose fit then rejected."
            )
        else:
            for name, arr in cols.items():
                self._resize_append(cg[name], arr)
        cg.attrs["cam_ids"] = np.array(self._corner_cams, dtype="S")

    def _flush_optical_quality(self, optical_g: h5py.Group, rows: list[dict[str, Any]]) -> None:
        qg = optical_g.require_group("quality")
        method = str(self.session_meta.get("optical_pose_method") or "")
        qg.attrs["rms_kind"] = "kabsch_mm" if method == "triangulation" else "reprojection_px"
        qg.attrs["note"] = (
            "per-body n_tags, n_cameras, reprojection_rms; "
            "rms is PnP corner reprojection (px) or triangulation Kabsch residual (mm)"
        )
        for slot in ("mta", "cra", "ct"):
            n_tags = np.zeros(len(rows), dtype=np.uint8)
            n_cams = np.zeros(len(rows), dtype=np.uint8)
            rms = np.full(len(rows), np.nan, dtype=np.float32)
            for i, row in enumerate(rows):
                q = (row.get("quality") or {}).get(slot) or {}
                n_tags[i] = int(q.get("n_tags") or 0)
                n_cams[i] = int(q.get("n_cameras") or 0)
                raw = q.get("reprojection_rms", q.get("reprojection_rms_px", np.nan))
                try:
                    rms[i] = float(raw)
                except (TypeError, ValueError):
                    rms[i] = np.nan
            sg = qg.require_group(slot)
            if "n_tags" not in sg:
                sg.create_dataset("n_tags", data=n_tags, maxshape=(None,), chunks=True)
                sg.create_dataset("n_cameras", data=n_cams, maxshape=(None,), chunks=True)
                sg.create_dataset("reprojection_rms", data=rms, maxshape=(None,), chunks=True)
            else:
                self._resize_append(sg["n_tags"], n_tags)
                self._resize_append(sg["n_cameras"], n_cams)
                self._resize_append(sg["reprojection_rms"], rms)

    def _flush_imu_slot(self, slot: str) -> None:
        assert self._file is not None
        buf = self._imu[slot]
        grp = self._file[slot]
        cfg = self.mcu_config.get(slot, {})
        rates = {
            name: int(entry.get("rate_hz", 0))
            for name, entry in (cfg.get("reports") or {}).items()
            if entry.get("enabled", True)
        }

        quat_rows = buf.get("quat")
        if quat_rows:
            t = np.asarray([r[0] for r in quat_rows], dtype=np.uint64)
            q = np.asarray([r[1] for r in quat_rows], dtype=np.float32)
            rid = np.asarray([(r[2] if r[2] is not None else 0) for r in quat_rows],
                             dtype=np.uint8)
            qacc = np.asarray([r[3] for r in quat_rows], dtype=np.uint8)
            if "quat" not in grp:
                grp.create_dataset("quat", data=q, maxshape=(None, 4), chunks=True)
                grp.create_dataset("t_master_us_quat", data=t, maxshape=(None,), chunks=True)
                grp.create_dataset("quat_report_id", data=rid, maxshape=(None,), chunks=True)
                grp.create_dataset("quat_accuracy", data=qacc, maxshape=(None,), chunks=True)
                grp["quat_accuracy"].attrs["note"] = ACCURACY_NOTE
                fusion = cfg.get("fusion", {})
                grp.attrs["fusion_mode"] = fusion.get("primary", "game_rotation_vector")
                report_ids = {int(x) for x in rid if x}
                if len(report_ids) == 1:
                    grp.attrs["quat_fusion"] = FUSION_REPORT_IDS.get(
                        report_ids.pop(), grp.attrs["fusion_mode"]
                    )
            else:
                self._resize_append(grp["quat"], q)
                self._resize_append(grp["t_master_us_quat"], t)
                self._resize_append(grp["quat_report_id"], rid)
                self._resize_append(grp["quat_accuracy"], qacc)
            buf["quat"] = []

        # int16 raw motion vectors -> SI (m/s^2, rad/s, uT) via the units module.
        for key, dsname, sensor_id, unit, rname in _IMU_MOTION_STREAMS:
            rows = buf.get(key)
            if not rows:
                continue
            t = np.asarray([r[0] for r in rows], dtype=np.uint64)
            counts = np.asarray([r[1] for r in rows], dtype=np.float64)
            si = imu_units.raw_array_to_si(sensor_id, counts)
            acc = np.asarray([r[2] for r in rows], dtype=np.uint8)
            self._append_vec3(grp, dsname, t, si, acc, unit=unit, rate_hz=rates.get(rname))
            buf[key] = []

        lin_rows = buf.get("linear_accel")
        if lin_rows:
            t = np.asarray([r[0] for r in lin_rows], dtype=np.uint64)
            v = np.asarray([r[1] for r in lin_rows], dtype=np.float32)
            acc = np.asarray([r[2] for r in lin_rows], dtype=np.uint8)
            self._append_vec3(grp, "linear_accel", t, v, acc, unit="m/s^2",
                              rate_hz=rates.get("linear_acceleration"))
            buf["linear_accel"] = []

    def _append_vec3(
        self,
        grp: h5py.Group,
        name: str,
        t: np.ndarray,
        v: np.ndarray,
        acc: np.ndarray,
        *,
        unit: str,
        rate_hz: int | None,
    ) -> None:
        t_name = f"t_master_us_{name}"
        a_name = f"{name}_accuracy"
        if name not in grp:
            ds = grp.create_dataset(name, data=v, maxshape=(None, 3), chunks=True)
            grp.create_dataset(t_name, data=t, maxshape=(None,), chunks=True)
            grp.create_dataset(a_name, data=acc, maxshape=(None,), chunks=True)
            grp[a_name].attrs["note"] = ACCURACY_NOTE
            ds.attrs["units"] = unit
            ds.attrs["device"] = "BNO085"
            if rate_hz:
                ds.attrs["rate_hz"] = int(rate_hz)
        else:
            self._resize_append(grp[name], v)
            self._resize_append(grp[t_name], t)
            self._resize_append(grp[a_name], acc)

    def write_sync_report(self, report: dict[str, Any]) -> None:
        if self._file is None:
            return
        self._file.require_group("sync").attrs["report_json"] = json.dumps(
            report, separators=(",", ":")
        )

    def write_sync_drops(self, drops: dict[str, int]) -> None:
        if self._file is None:
            return
        sync = self._file.require_group("sync")
        for key, count in drops.items():
            dname = f"drops/{key}"
            if dname in sync:
                continue
            sync.create_dataset(dname, data=np.asarray([[0, count]], dtype=np.uint64))

    def close(self) -> None:
        self.flush()
        if self._file is not None:
            meta = self._file["meta"]
            meta.attrs["session_duration_s"] = max(time.time() - self._opened_at, 0.0)
            sidecar_table = {
                name: sha256_file(p) for name, p in self.sidecar_paths.items() if p.is_file()
            }
            meta.attrs["file_hashes_json"] = json.dumps(sidecar_table, separators=(",", ":"), sort_keys=True)
            self._file.flush()
            self._file.close()
            self._file = None

    @staticmethod
    def _resize_append(dset: h5py.Dataset, block: np.ndarray) -> None:
        old = dset.shape[0]
        new = old + block.shape[0]
        dset.resize((new,) + dset.shape[1:])
        dset[old:new] = block

    @staticmethod
    def _resize_append_str(dset: h5py.Dataset, values: list[str]) -> None:
        old = dset.shape[0]
        new = old + len(values)
        dset.resize((new,))
        dset[old:new] = np.asarray(values, dtype=h5py.string_dtype())
