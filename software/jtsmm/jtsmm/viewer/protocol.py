# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer wire protocol: a compact, binary-friendly snapshot over WebSocket.

Snapshots are serialized with MessagePack. Wide EMG arrays are carried as raw
little-endian Float32 blobs (msgpack ``bin``) so the browser can wrap them in a
``Float32Array`` with zero parsing cost; everything else is plain scalars/maps.

MessagePack is an optional dependency (``jtsmm[viewer]``); ``encode``/``decode``
raise a clear error if it is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Snapshot", "ImuSample", "encode", "decode", "have_msgpack"]


def _msgpack():
    try:
        import msgpack  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "msgpack is required for the viewer wire protocol; install jtsmm[viewer]."
        ) from exc
    return msgpack


def have_msgpack() -> bool:
    try:
        import msgpack  # type: ignore  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


def _f32_bytes(arr: np.ndarray | None) -> bytes:
    if arr is None:
        return b""
    return np.ascontiguousarray(arr, dtype="<f4").tobytes()


# Motion stream keys carried in an ImuSample window (SI units on the wire).
IMU_MOTION_STREAMS = ("accel", "gyro", "mag", "linear_accel")


@dataclass
class ImuSample:
    """Per-slot IMU state for one frame.

    ``quat`` is the latest fusion orientation (w,x,y,z) used to drive the 3D
    body; ``quat_report_id`` is the SH-2 report that produced it (0x05 rotation
    vector, 0x08 game, 0x09 geomagnetic) so the UI can label the fusion source.
    ``motion`` holds short rolling windows for the enabled raw/derived streams
    (accelerometer, gyro, magnetometer, linear acceleration) as ``(t, v)`` where
    ``t`` is (n,) seconds relative to session t0 and ``v`` is (n, 3) SI values.
    """

    quat: tuple[float, float, float, float] | None = None
    t_s: float = 0.0
    quat_report_id: int | None = None
    # stream name -> (t (n,), v (n, 3)); either may be None/empty when disabled.
    motion: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = field(
        default_factory=dict
    )
    # Measured (true) per-stream rate in Hz, computed by the hub over a rolling
    # window from sample counts (not WebSocket poll rate). Keys are "quat" + the
    # motion stream names.
    rates: dict[str, float] = field(default_factory=dict)
    # Latest BNO085 calibration/accuracy per stream (0=unreliable..3=high).
    cal: dict[str, int] = field(default_factory=dict)

    def to_map(self) -> dict[str, Any] | None:
        motion_map: dict[str, Any] = {}
        for name, (t, v) in self.motion.items():
            if t is None or v is None or len(t) == 0:
                continue
            motion_map[name] = {"t": _f32_bytes(t), "v": _f32_bytes(v)}
        if self.quat is None and not motion_map and not self.rates and not self.cal:
            return None
        out: dict[str, Any] = {"t": float(self.t_s)}
        if self.quat is not None:
            out["q"] = [float(x) for x in self.quat]
        if self.quat_report_id is not None:
            out["qid"] = int(self.quat_report_id)
        if motion_map:
            out["motion"] = motion_map
        if self.rates:
            out["rates"] = {k: float(v) for k, v in self.rates.items()}
        if self.cal:
            out["cal"] = {k: int(v) for k, v in self.cal.items()}
        return out


@dataclass
class Snapshot:
    """One coalesced viewer frame (incremental append semantics on the client)."""

    seq: int = 0
    t_ref_us: int = 0
    t0_us: int = 0
    dropped: int = 0
    # EMG (all µV); t is seconds relative to session t0.
    emg_n: int = 0
    emg_channels: int = 8
    emg_fs: float = 0.0        # configured ADC sample rate (target)
    emg_hz: float = 0.0        # measured (true) ADC rate over a rolling window
    camera_hz: float = 0.0     # measured MCU trigger rate (camera sync proxy)
    emg_lsb_uv: float = 0.0
    emg_t: np.ndarray | None = None          # (n,)
    emg_lo: np.ndarray | None = None          # (n, c)
    emg_hi: np.ndarray | None = None          # (n, c)
    emg_filt: np.ndarray | None = None        # (n, c) or None
    emg_env: np.ndarray | None = None         # (n, c) or None
    imu_mta: ImuSample = field(default_factory=ImuSample)
    imu_cra: ImuSample = field(default_factory=ImuSample)
    # Derived jaw kinematics (viewer estimate): relative orientation q_rel and a
    # ZUPT-bounded relative translation estimate. None when both quats absent.
    imu_q_rel: tuple[float, float, float, float] | None = None
    imu_pos_rel: tuple[float, float, float] | None = None
    imu_pos_valid: bool = False
    leadoff_p: list[bool] = field(default_factory=lambda: [False] * 8)
    leadoff_n: list[bool] = field(default_factory=lambda: [False] * 8)
    leadoff_active: bool = False
    # Lead-off comparators: "on" (bits trusted), "off" or "unknown".
    leadoff_cmp: str = "unknown"
    drl_leadoff: bool = False
    saturation: list[bool] = field(default_factory=lambda: [False] * 8)
    impedance: list[dict[str, Any]] | None = None
    stats: dict[str, Any] | None = None
    health: dict[str, Any] = field(default_factory=dict)

    def to_map(self) -> dict[str, Any]:
        emg: dict[str, Any] = {
            "n": int(self.emg_n),
            "c": int(self.emg_channels),
            "fs": float(self.emg_fs),
            "hz": float(self.emg_hz),
            "lsb_uv": float(self.emg_lsb_uv),
            "t": _f32_bytes(self.emg_t),
            "lo": _f32_bytes(self.emg_lo),
            "hi": _f32_bytes(self.emg_hi),
            "filt": _f32_bytes(self.emg_filt),
            "env": _f32_bytes(self.emg_env),
        }
        return {
            "type": "frame",
            "seq": int(self.seq),
            "t_ref_us": int(self.t_ref_us),
            "t0_us": int(self.t0_us),
            "dropped": int(self.dropped),
            "emg": emg,
            "camera_hz": float(self.camera_hz),
            "imu": {
                "mta": self.imu_mta.to_map(),
                "cra": self.imu_cra.to_map(),
                "derived": self._derived_map(),
            },
            "leadoff": {
                "active": bool(self.leadoff_active),
                "cmp": str(self.leadoff_cmp),
                "p": list(self.leadoff_p),
                "n": list(self.leadoff_n),
                "drl": bool(self.drl_leadoff),
            },
            "impedance": self.impedance,
            "sat": list(self.saturation),
            "stats": self.stats,
            "health": self.health,
        }

    def _derived_map(self) -> dict[str, Any] | None:
        if self.imu_q_rel is None and self.imu_pos_rel is None:
            return None
        out: dict[str, Any] = {}
        if self.imu_q_rel is not None:
            out["q_rel"] = [float(x) for x in self.imu_q_rel]
        if self.imu_pos_rel is not None:
            out["pos"] = [float(x) for x in self.imu_pos_rel]
            out["pos_valid"] = bool(self.imu_pos_valid)
        return out


def encode(snapshot: Snapshot) -> bytes:
    msgpack = _msgpack()
    return msgpack.packb(snapshot.to_map(), use_bin_type=True)


def _to_f32(blob: bytes, cols: int) -> np.ndarray:
    arr = np.frombuffer(blob, dtype="<f4")
    if cols > 1 and arr.size:
        return arr.reshape(-1, cols)
    return arr


def decode(raw: bytes) -> dict[str, Any]:
    """Decode a frame back to a dict with NumPy arrays (used by tests/tools)."""
    msgpack = _msgpack()
    obj = msgpack.unpackb(raw, raw=False)
    emg = obj.get("emg", {})
    cols = int(emg.get("c", 8))
    for key in ("lo", "hi", "filt", "env"):
        emg[key] = _to_f32(emg.get(key, b""), cols)
    emg["t"] = _to_f32(emg.get("t", b""), 1)
    obj["emg"] = emg
    imu = obj.get("imu")
    if isinstance(imu, dict):
        for slot in ("mta", "cra"):
            sample = imu.get(slot)
            if not isinstance(sample, dict):
                continue
            motion = sample.get("motion")
            if isinstance(motion, dict):
                for name, stream in motion.items():
                    if not isinstance(stream, dict):
                        continue
                    stream["t"] = _to_f32(stream.get("t", b""), 1)
                    stream["v"] = _to_f32(stream.get("v", b""), 3)
    return obj
