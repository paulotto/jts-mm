# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU streams of one take, sanitized and resampled (analysis_plan §3.3, an04 §2).

The pilot recorded per IMU: game-rotation-vector quaternions (~390 Hz,
world<-body, no magnetometer: each IMU has its own heading reference) and the
raw accelerometer *including gravity* (~255 Hz). No raw gyroscope.
"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field

import numpy as np

from jtsmm.imu.batch import quat_prepare, quat_resample, sanitize_stream
from jtsmm.imu.gravity_lin_accel import G

from biomech.config import package_default
from biomech.io.session import SessionData

# Longest sample gap bridged by resampling (E33.7): the qc coverage parameter, one value for
# every IMU consumer.
GAP_S: float = float(package_default("stages", "qc", "imu", "gap_s"))
GAP_US: int = int(round(GAP_S * 1e6))


@dataclass
class ImuStream:
    slot: str
    t_q: np.ndarray        # (N,) int64
    q: np.ndarray          # (N, 4) w,x,y,z
    t_a: np.ndarray        # (M,) int64
    a: np.ndarray          # (M, 3) m/s^2, gravity included
    n_dropped_q: int
    n_dropped_a: int
    _qp: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def quat_at(self, t_us: np.ndarray, max_gap_us: int = GAP_US) -> np.ndarray:
        # The normalized, sign-continuous stream is computed once per stream (identical to
        # what quat_resample would recompute on every call).
        if self._qp is None:
            self._qp = quat_prepare(self.q)
        return quat_resample(self.t_q, self._qp, t_us, max_gap_us, prepared=True)

    def accel_at(self, t_us: np.ndarray, max_gap_us: int = GAP_US) -> np.ndarray:
        out = np.column_stack([np.interp(t_us, self.t_a, self.a[:, k]) for k in range(3)])
        j = np.clip(np.searchsorted(self.t_a, t_us), 1, len(self.t_a) - 1)
        gap = self.t_a[j] - self.t_a[j - 1]
        out[(gap > max_gap_us) | (t_us < self.t_a[0]) | (t_us > self.t_a[-1])] = np.nan
        return out


# Streams per open session: the imu and ml_windows stages of one participant chain share the
# chain's SessionData (pipeline.UnitCache), so the streams are read and sanitized once. The
# arrays are read-only, so no consumer can change what another one sees.
_LOADED: "weakref.WeakKeyDictionary[SessionData, dict[str, ImuStream | None]]" = \
    weakref.WeakKeyDictionary()
_LOCK = threading.Lock()


def load(sd: SessionData, slot: str) -> ImuStream | None:
    """Sanitized IMU stream ``slot`` of the take (None when missing); cached per session
    object (a pure function of the file, the arrays are read-only)."""
    with _LOCK:
        per = _LOADED.get(sd)
        if per is not None and slot in per:
            return per[slot]
    out = _load(sd, slot)
    with _LOCK:
        _LOADED.setdefault(sd, {})[slot] = out
    return out


def _load(sd: SessionData, slot: str) -> ImuStream | None:
    imu = sd.imu(slot)
    if imu is None or imu.n == 0 or imu.accel is None:
        return None
    q_ok = (imu.quat_accuracy != 0) if imu.quat_accuracy is not None else np.ones(imu.n, bool)
    t_q, q = sanitize_stream(imu.t_us[q_ok], imu.quat[q_ok].astype(np.float64))
    acc = imu.accel
    a_ok = (acc.accuracy != 0) if acc.accuracy is not None else np.ones(acc.n, bool)
    t_a, a = sanitize_stream(acc.t_us[a_ok], acc.data[a_ok].astype(np.float64))
    for arr in (t_q, q, t_a, a):
        arr.flags.writeable = False
    return ImuStream(slot, t_q, q, t_a, a, int(imu.n - len(t_q)), int(acc.n - len(t_a)))


__all__ = ["G", "ImuStream", "load"]
