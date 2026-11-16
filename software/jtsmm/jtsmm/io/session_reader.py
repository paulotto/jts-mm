# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Read-only accessor for recorded JTS-MM sessions (s07 §3.2).

`Session` opens a ``session.h5`` and exposes the EMG, IMU (MTA/CRA), trigger,
sync, and meta groups with chunked, windowed reads suited to both the replay
viewer and programmatic analysis. Timestamp vectors are cached in RAM (cheap:
16 B/sample) so seeking is a `searchsorted`, while the wide sample matrices are
read lazily by slice so long 2 kHz sessions never have to fit in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm.config.catalog import (FULL_CHANNEL_NAMES, LEGACY_CHANNEL_NAMES_DIGASTRIC_SWAPPED,
                                  channel_lsb_uv, lsb_uv)

__all__ = ["Session", "EmgStream", "QuatStream", "ImuKinematicsStream", "TriggerStream",
           "montage_corrected"]


def _attr(obj: Any, key: str, default: Any = None) -> Any:
    val = obj.attrs.get(key, default)
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, np.generic):
        return val.item()
    return val


def _opt_u8(g: Any, name: str) -> np.ndarray | None:
    return np.asarray(g[name], dtype=np.uint8) if name in g else None


def _decode_names(raw: Any) -> list[str]:
    if raw is None:
        return list(FULL_CHANNEL_NAMES)
    out: list[str] = []
    for item in np.asarray(raw).ravel():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8", "replace"))
        else:
            out.append(str(item))
    return out


def montage_corrected(names: list[str]) -> tuple[list[str], str | None]:
    """The channel names of a session as the electrodes were cabled, and a note when they were
    corrected. Recorders before 2026-11-03 wrote CH3/CH4 as digastric_l/digastric_r, the
    reverse of the cable labels (an03 §1): exactly that list is read as FULL_CHANNEL_NAMES."""
    if tuple(names) == LEGACY_CHANNEL_NAMES_DIGASTRIC_SWAPPED:
        return list(FULL_CHANNEL_NAMES), (
            "channel_names corrected on read: the file names CH3/CH4 digastric_l/digastric_r, "
            "the cables are CH3 digastric_r, CH4 digastric_l (hardware check 2026-11-03)")
    return names, None


@dataclass
class EmgStream:
    """EMG samples; ``samples_uv`` is a lazy HDF5 dataset, ``t_us`` is cached."""

    t_us: np.ndarray
    _samples_uv: h5py.Dataset | np.ndarray
    fs_hz: float
    gain: int
    lsb_uv: float
    channel_names: list[str] = field(default_factory=lambda: list(FULL_CHANNEL_NAMES))
    # (N, 2) [statp_mask, statn_mask], muscle CH1-6 only (DRL excluded, see
    # h03 §6); None for sessions written before this dataset existed.
    _leadoff_status: h5py.Dataset | np.ndarray | None = None
    # Per-column LSB (µV): CH7/CH8 run at a fixed gain of 24 (K5).
    lsb_uv_per_channel: list[float] | None = None
    # Files written before lsb_uv_per_channel stored CH7/CH8 with the muscle
    # LSB; this per-column factor corrects them on read (None = no correction).
    _uv_scale: np.ndarray | None = None
    uv_note: str | None = None
    # Set when the file's channel_names were corrected on read (montage_corrected).
    montage_note: str | None = None

    @property
    def n(self) -> int:
        return int(self.t_us.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self._samples_uv.shape[1]) if self._samples_uv.ndim == 2 else 0

    def index_at(self, t_us: int) -> int:
        """First sample index at or after ``t_us``."""
        return int(np.searchsorted(self.t_us, np.uint64(t_us), side="left"))

    def window(self, i0: int, i1: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(t_us[i0:i1], samples_uv[i0:i1])`` (samples read lazily)."""
        i0 = max(0, int(i0))
        i1 = min(self.n, int(i1))
        if i1 <= i0:
            empty_uv = np.empty((0, self.n_channels), dtype=np.float32)
            return self.t_us[i0:i0], empty_uv
        return self.t_us[i0:i1], self._scaled(self._samples_uv[i0:i1])

    @property
    def samples_uv(self) -> np.ndarray:
        """Materialize the whole matrix (use only for short sessions/tests)."""
        return self._scaled(self._samples_uv[()])

    def _scaled(self, block: Any) -> np.ndarray:
        arr = np.asarray(block, dtype=np.float32)
        if self._uv_scale is not None and arr.ndim == 2 and arr.shape[1] == self._uv_scale.size:
            arr = arr * self._uv_scale[None, :]
        return arr

    @property
    def has_leadoff_status(self) -> bool:
        return self._leadoff_status is not None

    def leadoff_status_window(self, i0: int, i1: int) -> np.ndarray:
        """``[statp_mask, statn_mask]`` for ``[i0:i1)``; all-zero if not recorded."""
        i0 = max(0, int(i0))
        i1 = min(self.n, int(i1))
        if self._leadoff_status is None or i1 <= i0:
            return np.zeros((max(0, i1 - i0), 2), dtype=np.uint8)
        return np.asarray(self._leadoff_status[i0:i1], dtype=np.uint8)


@dataclass
class Vec3Stream:
    """A generic (N,3) IMU motion stream with its own timestamp vector."""

    t_us: np.ndarray
    data: np.ndarray  # (N, 3) float32 in SI units
    units: str = ""
    rate_hz: int = 0
    # BNO085 calibration status per row (0..3, 255 = not sent); None in older files.
    accuracy: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.t_us.shape[0])

    def index_at(self, t_us: int) -> int:
        return int(np.searchsorted(self.t_us, np.uint64(t_us), side="left"))


@dataclass
class ImuKinematicsStream:
    """One IMU slot: fusion quaternion plus any enabled motion vectors.

    ``t_us``/``quat`` mirror the legacy :class:`QuatStream` surface so existing
    replay/analysis code keeps working; ``accel``/``gyro``/``mag``/
    ``linear_accel`` are present only for streams recorded in the session.
    """

    t_us: np.ndarray
    quat: np.ndarray  # (N, 4) float32, (w, x, y, z)
    fusion_mode: str = "game_rotation_vector"
    quat_report_id: np.ndarray | None = None
    quat_accuracy: np.ndarray | None = None  # as Vec3Stream.accuracy
    accel: Vec3Stream | None = None
    gyro: Vec3Stream | None = None
    mag: Vec3Stream | None = None
    linear_accel: Vec3Stream | None = None

    @property
    def n(self) -> int:
        return int(self.t_us.shape[0])

    def index_at(self, t_us: int) -> int:
        return int(np.searchsorted(self.t_us, np.uint64(t_us), side="left"))

    def motion_streams(self) -> dict[str, Vec3Stream]:
        """Present (name -> stream) motion vectors, for viewers/analysis."""
        out: dict[str, Vec3Stream] = {}
        for name in ("accel", "gyro", "mag", "linear_accel"):
            s = getattr(self, name)
            if s is not None and s.n > 0:
                out[name] = s
        return out


# Backwards-compatible alias: older code imports QuatStream.
QuatStream = ImuKinematicsStream


@dataclass
class TriggerStream:
    t_us: np.ndarray
    trigger_id: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t_us.shape[0])


class Session:
    """Read-only view of a recorded ``session.h5``.

    Use as a context manager::

        with Session.open("session.h5") as s:
            t, uv = s.emg.window(0, 1000)
            q = s.imu_mta.quat
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._f: h5py.File | None = h5py.File(str(self.path), "r")
        self.meta: dict[str, Any] = self._read_meta()
        self.emg: EmgStream | None = self._read_emg()
        self.imu_mta: ImuKinematicsStream | None = self._read_imu("imu_mta")
        self.imu_cra: ImuKinematicsStream | None = self._read_imu("imu_cra")
        self.triggers: TriggerStream | None = self._read_triggers()
        self.sync_report: dict[str, Any] = self._read_sync_report()

    @classmethod
    def open(cls, path: str | Path) -> "Session":
        return cls(path)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def t0_us(self) -> int:
        starts = [s.t_us[0] for s in self._streams() if s.n > 0]
        return int(min(starts)) if starts else 0

    @property
    def t1_us(self) -> int:
        ends = [s.t_us[-1] for s in self._streams() if s.n > 0]
        return int(max(ends)) if ends else 0

    @property
    def duration_s(self) -> float:
        meta_dur = self.meta.get("session_duration_s")
        if isinstance(meta_dur, (int, float)) and meta_dur > 0:
            return float(meta_dur)
        span = self.t1_us - self.t0_us
        return span / 1e6 if span > 0 else 0.0

    def event_times_us(self) -> list[int]:
        """Marker timestamps for 'jump to event' — /events if present, else triggers."""
        f = self._require()
        if "events" in f and "t_master_us" in f["events"]:
            return [int(x) for x in np.asarray(f["events"]["t_master_us"])]
        if self.triggers is not None and self.triggers.n > 0:
            return [int(x) for x in self.triggers.t_us]
        return []

    def events(self) -> list[dict[str, Any]]:
        """All ``/events`` rows (phase edges, viewer notes, ...) sorted by ``t_master_us``.

        Rows are stored in write order (notes as they arrive, phase edges at
        close), not time order. Keys: t_master_us, kind, text, phase_kind,
        t_host_ns, map_quality, viewer_t_s (None when absent/NaN).
        """
        f = self._require()
        g = f.get("events")
        if g is None or "t_master_us" not in g:
            return []
        n = int(g["t_master_us"].shape[0])

        def col(name: str, default: Any) -> list[Any]:
            if name not in g:
                return [default] * n
            ds = g[name]
            if h5py.check_string_dtype(ds.dtype) is not None:
                return [str(x) for x in ds.asstr()[()]]
            return ds[()].tolist()

        rows = [
            {
                "t_master_us": int(t), "kind": k, "text": txt, "phase_kind": pk,
                "t_host_ns": int(h), "map_quality": q,
                "viewer_t_s": float(v) if v is not None and np.isfinite(v) else None,
            }
            for t, k, txt, pk, h, q, v in zip(
                col("t_master_us", 0), col("kind", ""), col("text", ""), col("phase_kind", ""),
                col("t_host_ns", 0), col("map_quality", ""), col("viewer_t_s", None),
            )
        ]
        return sorted(rows, key=lambda r: r["t_master_us"])

    def notes(self) -> list[dict[str, Any]]:
        """Operator notes typed into the live viewer (``/events`` kind ``note``)."""
        return [e for e in self.events() if e["kind"] == "note"]

    def mcu_logs(self) -> list[dict[str, Any]]:
        """Firmware log lines (IMU recovery / watchdog, apply warnings; kind ``mcu_log``)."""
        return [e for e in self.events() if e["kind"] == "mcu_log"]

    def _streams(self) -> list[Any]:
        return [s for s in (self.emg, self.imu_mta, self.imu_cra) if s is not None]

    def _require(self) -> h5py.File:
        if self._f is None:
            raise RuntimeError("Session is closed")
        return self._f

    def _read_meta(self) -> dict[str, Any]:
        f = self._require()
        out: dict[str, Any] = {}
        if "meta" in f:
            for k in f["meta"].attrs:
                out[k] = _attr(f["meta"], k)
        return out

    def _read_emg(self) -> EmgStream | None:
        f = self._require()
        if "emg" not in f or "samples_uv" not in f["emg"]:
            return None
        g = f["emg"]
        t_us = np.asarray(g["t_master_us"], dtype=np.uint64)
        gain = int(_attr(g, "gain", 6) or 6)
        fs = float(_attr(g, "sample_rate_sps", 2000) or 2000)  # files without the attr predate 4 kSPS
        lsb = float(_attr(g, "lsb_uv", lsb_uv(gain)))
        names, montage_note = montage_corrected(_decode_names(g.attrs.get("channel_names")))
        n_cols = int(g["samples_uv"].shape[1]) if g["samples_uv"].ndim == 2 else 0
        per = g.attrs.get("lsb_uv_per_channel")
        scale = None
        note = None
        if per is not None:
            per_list = [float(v) for v in np.asarray(per, dtype=np.float64).ravel()]
        else:
            # Legacy file: samples_uv = counts * muscle LSB in every column.
            per_list = channel_lsb_uv(gain, n_cols)
            factors = np.asarray([p / lsb for p in per_list], dtype=np.float32) if lsb > 0 else None
            if factors is not None and np.any(np.abs(factors - 1.0) > 1e-6):
                scale = factors
                note = (
                    "file predates lsb_uv_per_channel: CH7/CH8 samples_uv corrected on read "
                    "from the muscle gain to their fixed gain 24"
                )
        return EmgStream(t_us=t_us, _samples_uv=g["samples_uv"], fs_hz=fs, gain=gain,
                         lsb_uv=lsb, channel_names=names,
                         _leadoff_status=g.get("leadoff_status"),
                         lsb_uv_per_channel=per_list, _uv_scale=scale, uv_note=note,
                         montage_note=montage_note)

    def _read_vec3(self, g: h5py.Group, name: str) -> Vec3Stream | None:
        if name not in g:
            return None
        t_name = f"t_master_us_{name}"
        if t_name not in g:
            return None
        ds = g[name]
        return Vec3Stream(
            t_us=np.asarray(g[t_name], dtype=np.uint64),
            data=np.asarray(ds, dtype=np.float32),
            units=str(_attr(ds, "units", "")),
            rate_hz=int(_attr(ds, "rate_hz", 0) or 0),
            accuracy=_opt_u8(g, f"{name}_accuracy"),
        )

    def _read_imu(self, group: str) -> ImuKinematicsStream | None:
        f = self._require()
        if group not in f:
            return None
        g = f[group]
        # A slot may carry motion vectors without a fusion quaternion
        # (e.g. profile: datasheet_motion), so don't require "quat".
        motion = {name: self._read_vec3(g, name) for name in ("accel", "gyro", "mag", "linear_accel")}
        has_quat = "quat" in g
        if not has_quat and not any(v is not None for v in motion.values()):
            return None
        if has_quat:
            t = np.asarray(g["t_master_us_quat"], dtype=np.uint64)
            q = np.asarray(g["quat"], dtype=np.float32)
            rid = _opt_u8(g, "quat_report_id")
        else:
            t = np.empty(0, dtype=np.uint64)
            q = np.empty((0, 4), dtype=np.float32)
            rid = None
        return ImuKinematicsStream(
            t_us=t,
            quat=q,
            fusion_mode=str(_attr(g, "fusion_mode", "game_rotation_vector")),
            quat_report_id=rid,
            quat_accuracy=_opt_u8(g, "quat_accuracy") if has_quat else None,
            accel=motion["accel"],
            gyro=motion["gyro"],
            mag=motion["mag"],
            linear_accel=motion["linear_accel"],
        )

    def _read_triggers(self) -> TriggerStream | None:
        f = self._require()
        if "triggers" not in f or "trigger_id" not in f["triggers"]:
            return None
        g = f["triggers"]
        return TriggerStream(
            t_us=np.asarray(g["t_master_us"], dtype=np.uint64),
            trigger_id=np.asarray(g["trigger_id"], dtype=np.uint64),
        )

    def _read_sync_report(self) -> dict[str, Any]:
        f = self._require()
        if "sync" not in f:
            return {}
        raw = f["sync"].attrs.get("report_json")
        if raw is None:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            obj = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            return {}
        return obj if isinstance(obj, dict) else {}
