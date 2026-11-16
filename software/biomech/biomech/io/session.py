# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Phase-windowed access to one recorded take.

A thin layer over :class:`jtsmm.io.session_reader.Session`: it adds the phase
table, the optical group (which the capture reader does not expose), the
integrity sidecar, and muscle-channel selection by name. Wide matrices are read
only for the windows asked for, so analysing a 15-min take reads seconds of EMG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm.config.catalog import FULL_CHANNEL_NAMES
from jtsmm.io.session_reader import ImuKinematicsStream, Session
from jtsmm.tools.plot_mp_trajectory import PhaseInterval, load_phases

# CH1-6 of the montage (an03 §1), the recorder's own list. The file's channel_names (corrected
# on read by jtsmm's session reader) win when present.
MUSCLES: tuple[str, ...] = tuple(FULL_CHANNEL_NAMES[:6])
OPENERS: tuple[str, ...] = ("digastric_l", "digastric_r")
CLOSERS: tuple[str, ...] = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l")


@dataclass
class OpticalData:
    t_us: np.ndarray                     # (T,) uint64
    poses: dict[str, np.ndarray]         # body -> (T, 4, 4) float64, mm, NaN when untracked
    quality: dict[str, dict[str, np.ndarray]]

    def lock(self, body: str) -> np.ndarray:
        """Boolean (T,) mask of frames with a finite pose for ``body``."""
        p = self.poses[body]
        return np.isfinite(p.reshape(p.shape[0], -1)).all(axis=1)


class SessionData:
    """Use as a context manager; closes the HDF5 file on exit."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self._s = Session.open(self.folder / "session.h5")
        with h5py.File(self.folder / "session.h5", "r") as f:
            self.phases: list[PhaseInterval] = list(load_phases(f))
        self._optical: OpticalData | None = None

    def __enter__(self) -> "SessionData":
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._s.close()

    # -- metadata -----------------------------------------------------------------------
    @property
    def meta(self) -> dict[str, Any]:
        return self._s.meta

    @property
    def sync_report(self) -> dict[str, Any]:
        return self._s.sync_report

    @property
    def duration_s(self) -> float:
        return self._s.duration_s

    def events(self) -> list[dict[str, Any]]:
        return self._s.events()

    def integrity(self) -> dict[str, Any]:
        p = self.folder / "integrity.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

    # -- phases -------------------------------------------------------------------------
    def phase(self, name: str) -> PhaseInterval | None:
        for ph in self.phases:
            if ph.name == name:
                return ph
        return None

    def phase_names(self) -> list[str]:
        return [ph.name for ph in self.phases]

    # -- EMG ----------------------------------------------------------------------------
    @property
    def emg_fs(self) -> float:
        return float(self._s.emg.fs_hz) if self._s.emg is not None else float("nan")

    def muscle_names(self) -> list[str]:
        emg = self._s.emg
        if emg is None:
            return []
        names = [n for n in emg.channel_names if n in MUSCLES]
        return names or list(MUSCLES[: min(6, emg.n_channels)])

    def emg_full_scale_uv(self, channels: list[str] | None = None) -> np.ndarray:
        """ADC full scale per channel in uV (24-bit: LSB x (2^23 - 1))."""
        emg = self._s.emg
        names = channels or self.muscle_names()
        lsb = emg.lsb_uv_per_channel or [emg.lsb_uv] * emg.n_channels
        return np.array([lsb[self._emg_col(n)] * 8388607.0 for n in names])

    def emg_window(self, t0_us: int, t1_us: int, channels: list[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """``(t_us, uV (N, C), names)`` for ``[t0, t1]``; muscle channels by default."""
        emg = self._s.emg
        if emg is None:
            raise LookupError("session has no EMG")
        names = channels or self.muscle_names()
        cols = [self._emg_col(n) for n in names]
        i0, i1 = emg.index_at(t0_us), emg.index_at(t1_us + 1)
        t, uv = emg.window(i0, i1)
        return t, uv[:, cols].astype(np.float64), names

    def leadoff_window(self, t0_us: int, t1_us: int, channels: list[str] | None = None
                       ) -> np.ndarray:
        """Boolean (N, C): the channel's P or N lead-off bit is set (CH1-6 only)."""
        emg = self._s.emg
        names = channels or self.muscle_names()
        i0, i1 = emg.index_at(t0_us), emg.index_at(t1_us + 1)
        st = emg.leadoff_status_window(i0, i1)
        bits = np.array([1 << self._emg_col(n) for n in names], dtype=np.uint8)
        return ((st[:, 0:1] | st[:, 1:2]) & bits[None, :]) != 0

    def emg_phase(self, phase: str, channels: list[str] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ph = self.phase(phase)
        if ph is None:
            raise LookupError(f"no phase {phase!r} in {self.folder.name}")
        return self.emg_window(ph.t_start_us, ph.t_end_us, channels)

    def _emg_col(self, name: str) -> int:
        names = self._s.emg.channel_names
        if name in names:
            return names.index(name)
        if name in MUSCLES:
            return MUSCLES.index(name)
        raise KeyError(name)

    # -- IMU ----------------------------------------------------------------------------
    def imu(self, slot: str) -> ImuKinematicsStream | None:
        return {"mta": self._s.imu_mta, "cra": self._s.imu_cra}[slot]

    # -- optical ------------------------------------------------------------------------
    def optical(self) -> OpticalData | None:
        if self._optical is not None:
            return self._optical
        with h5py.File(self.folder / "session.h5", "r") as f:
            g = f.get("optical")
            if g is None or "t_master_us" not in g:
                return None
            poses = {k[len("pose_"):]: np.asarray(g[k], dtype=np.float64)
                     for k in g if k.startswith("pose_")}
            quality: dict[str, dict[str, np.ndarray]] = {}
            if "quality" in g:
                for body, qg in g["quality"].items():
                    quality[body] = {k: np.asarray(v) for k, v in qg.items()}
            self._optical = OpticalData(t_us=np.asarray(g["t_master_us"], dtype=np.uint64),
                                        poses=poses, quality=quality)
        return self._optical
