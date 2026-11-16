# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Modality hub: coalesces source samples into ~30 Hz viewer snapshots.

A `Source` pushes decimated EMG rows (per-channel min/max bands in µV), the
latest IMU quaternions, the latest lead-off status word, and the latest STATS
map. The hub runs the causal display filter, keeps a bounded pending buffer of
*new* display rows (incremental-append semantics on the client), and hands them
out via `snapshot()`. If a consumer lags, the oldest pending rows are dropped —
the hub never blocks the source (single-USB-owner / real-time capture safety).
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections import deque
from typing import Any

import numpy as np

from jtsmm.config.catalog import (
    AUX_CHANNEL_GAIN,
    VIEWER_EMG_CHANNELS,
    VIEWER_EMG_PLOT_COUNT,
)
from jtsmm.config.catalog import lsb_uv as catalog_lsb_uv
from jtsmm.imu.kinematics import (
    ZuptIntegrator,
    apply_mounting_quat,
    quat_at_time,
    relative_linear_accel,
    relative_quaternion,
)
from jtsmm.viewer.leadoff import COMPARATOR_CHANNELS, decode_status_word, saturation_flags
from jtsmm.viewer.processing import CausalEmgFilter, EMG_BAND_HZ
from jtsmm.viewer.protocol import IMU_MOTION_STREAMS, ImuSample, Snapshot

__all__ = ["Hub"]

TARGET_DISPLAY_FS = 500.0  # per-channel display rate after decimation


def display_bucket(emg_fs: float) -> int:
    """Raw samples per display row, so every EMG rate is shown at ~500 rows/s.

    The in-recording viewer buckets on the native tap; a fixed 4 gave 1000
    rows/s at 4 kSPS (twice the WebSocket traffic and filter work).
    """
    if emg_fs <= 0:
        return 1
    return max(1, int(round(float(emg_fs) / TARGET_DISPLAY_FS)))
IMU_SLOTS = ("mta", "cra")
# Debounce ADS1299 comparator flicker on floating inputs (~30 Hz snapshots).
LEADOFF_BAD_FRAMES = 4
LEADOFF_GOOD_FRAMES = 12
# Streams for which the hub reports a measured (true) rate: the coalesced fusion
# quaternion plus every raw/derived motion stream.
IMU_RATE_STREAMS = ("quat",) + IMU_MOTION_STREAMS
# BNO085 SH-2 accuracy keys surfaced to the viewer (0=unreliable..3=high).
IMU_CAL_STREAMS = ("quat",) + IMU_MOTION_STREAMS
IMU_RATE_WINDOW_S = 2.0  # rolling window for the measured-rate estimate
# Pending motion samples per slot/stream are bounded to the newest
# IMU_MOTION_KEEP_S (and IMU_MOTION_MAX samples): with no client draining the
# hub they grew without limit (~356 MB/h inside the recording process).
IMU_MOTION_KEEP_S = 2.0
IMU_MOTION_MAX = 2048
# Linear-accel samples kept per slot for timestamp pairing (~2.5 s at 400 Hz);
# a slot that stops reporting cannot make the other one's buffer grow.
LIN_PAIR_MAX = 1024
# Do not interpolate the other slot's accel across a hole wider than this.
LIN_PAIR_MAX_GAP_US = 50_000
EMG_RATE_WINDOW_S = 2.0  # rolling window for the measured EMG rate


class Hub:
    def __init__(
        self,
        *,
        emg_fs: float,
        emg_lsb_uv: float,
        n_channels: int = 8,
        channel_names: list[str] | None = None,
        display_fs: float = TARGET_DISPLAY_FS,
        filter_enabled: bool = True,
        max_pending_rows: int = 4096,
        leadoff_during_recording: bool = False,
        leadoff_current_na: float = 6.0,
        line_hz: float = 50.0,
        mains_mode: str = "track",
        col_lsb_uv: list[float] | None = None,
        comparators_enabled: bool | None = None,
        leadoff_ac_tone: bool = False,
    ) -> None:
        self.emg_fs = float(emg_fs)
        self.emg_lsb_uv = float(emg_lsb_uv)
        self.n_channels = int(n_channels)
        # Per-display-column LSB (saturation needs counts). The standard viewer
        # layout is CH1-6 + CH8, and CH8 runs at the fixed gain 24.
        if col_lsb_uv is not None and len(col_lsb_uv) == self.n_channels:
            self.col_lsb_uv = [float(v) for v in col_lsb_uv]
        elif self.n_channels == VIEWER_EMG_PLOT_COUNT:
            self.col_lsb_uv = [self.emg_lsb_uv] * VIEWER_EMG_CHANNELS + [
                catalog_lsb_uv(AUX_CHANNEL_GAIN)
            ]
        else:
            self.col_lsb_uv = [self.emg_lsb_uv] * self.n_channels
        self.channel_names = list(channel_names) if channel_names else []
        self.display_fs = float(display_fs) if display_fs > 0 else self.emg_fs
        self.max_pending_rows = int(max_pending_rows)
        self.leadoff_during_recording = bool(leadoff_during_recording)
        self.leadoff_current_na = float(leadoff_current_na)
        self.line_hz = float(line_hz) if line_hz else 50.0
        # "track" fits and subtracts the line tone (and locks to the real grid
        # frequency); "notch" is the older 5 Hz-per-harmonic biquad bank.
        self.mains_mode = str(mains_mode or "track")
        self._impedance_enabled = False
        # Live |Z| needs the AC lead-off tone; without it there is nothing to
        # measure and the panel shows n/a (it used to show "good 0.0 kOhm").
        self.leadoff_ac_tone = bool(leadoff_ac_tone)
        self._impedance: list[dict[str, Any]] | None = None
        self._impedance_buf: list[tuple[np.ndarray, np.ndarray]] = []
        self._impedance_buf_n = 0
        self._impedance_max_samples = 4096
        self._impedance_new = 0

        self._lock = threading.Lock()
        # Serialises the (stateful) display filters. They run OUTSIDE _lock so
        # the WebSocket snapshot never waits for a filter pass.
        self._filter_lock = threading.Lock()
        self._t: list[np.ndarray] = []
        self._lo: list[np.ndarray] = []
        self._hi: list[np.ndarray] = []
        self._filt: list[np.ndarray] = []
        self._env: list[np.ndarray] = []
        self._pending_rows = 0
        self._dropped = 0

        self._t0_us = 0
        self._t0_set = False
        self._t_ref_us = 0
        self._seq = 0

        # Rolling (t_last_us, raw_sample_count) for the measured true EMG rate.
        self._emg_rate: deque[tuple[int, int]] = deque()

        # Latest fusion orientation per slot (drives the 3D bodies).
        self._imu_quat: dict[str, tuple[float, float, float, float] | None] = {
            "mta": None, "cra": None
        }
        self._R_mount: dict[str, np.ndarray | None] = {"mta": None, "cra": None}
        self._imu_quat_t: dict[str, float] = {"mta": 0.0, "cra": 0.0}
        self._imu_quat_id: dict[str, int | None] = {"mta": None, "cra": None}
        # Latest BNO085 calibration/accuracy per slot/stream (0..3); omitted when
        # unknown (stream not yet received).
        self._imu_cal: dict[str, dict[str, int]] = {
            slot: {} for slot in IMU_SLOTS
        }
        # Pending (incremental) motion samples per slot/stream as (t_rel_s, xyz),
        # cleared each snapshot and bounded in between (IMU_MOTION_KEEP_S).
        self._imu_motion: dict[str, dict[str, deque[tuple[float, list[float]]]]] = {
            slot: {s: deque(maxlen=IMU_MOTION_MAX) for s in IMU_MOTION_STREAMS}
            for slot in IMU_SLOTS
        }
        # Rolling (t_last_us, sample_count) per slot/stream for measured IMU rates.
        self._imu_rate: dict[str, dict[str, deque[tuple[int, int]]]] = {
            slot: {s: deque() for s in IMU_RATE_STREAMS} for slot in IMU_SLOTS
        }
        # Timestamped quaternion history per slot (for lin-accel alignment).
        self._imu_quat_hist: dict[str, deque[tuple[int, tuple[float, float, float, float]]]] = {
            slot: deque(maxlen=256) for slot in IMU_SLOTS
        }
        # Rolling (t_last_us, trigger_event_count) for camera-sync rate proxy.
        self._trigger_rate: deque[tuple[int, int]] = deque()
        # Latest JPEG preview per camera (MJPEG source for /stream/{cam_id}).
        self._camera_order: list[str] = []
        self._camera_jpeg: dict[str, bytes] = {}
        # Timestamped chip-frame linear acceleration per slot, paired BY TIME for
        # the relative estimate. MTA samples are the integration grid and wait
        # in _lin_pending until CRA data covers their timestamp.
        self._lin_pending: list[tuple[int, tuple[float, float, float]]] = []
        self._lin_cra: list[tuple[int, tuple[float, float, float]]] = []
        self._lin_last_t: int | None = None
        # Which motion stream feeds the estimate: linear_accel when recorded,
        # else the gravity-included accel (kinematics_accel). Gravity is +g along
        # world Z for both game rotation vectors, whatever their heading, so it
        # cancels in the MTA-CRA difference just like the BNO085's own removal.
        self._lin_source: str | None = None
        self._zupt = ZuptIntegrator()
        self._q_rel: tuple[float, float, float, float] | None = None
        self._pos_rel: tuple[float, float, float] | None = None
        self._pos_valid = False
        self._leadoff_p = [False] * self.n_channels
        self._leadoff_n = [False] * self.n_channels
        self._leadoff_bad_streak = [0] * self.n_channels
        self._leadoff_good_streak = [0] * self.n_channels
        self._leadoff_latched = [False] * self.n_channels
        # Lead-off comparators: True/False from the MCU's CONFIG4 readback,
        # None when unknown (e.g. in-recording tap). See comparator_state().
        self._cmp_enabled = comparators_enabled
        self._cmp_seen = False
        self._drl_off = False
        self._drl_bad_streak = 0
        self._drl_good_streak = 0
        self._drl_latched = False
        self._saturation = [False] * self.n_channels
        self._sat_hold = [0] * self.n_channels
        self._stats: dict | None = None

        display_lpf = (
            min(EMG_BAND_HZ, (self.display_fs / 2.0) * 0.9)
            if self.display_fs > 0 else EMG_BAND_HZ
        )
        self._filter = (
            CausalEmgFilter(
                self.display_fs, self.n_channels,
                lpf_hz=display_lpf, notch_hz=self.line_hz,
                mains_mode=self.mains_mode,
            )
            if filter_enabled else None
        )
        # Full-rate twin used by append_emg_raw so 250 Hz (and other harmonics
        # above the 500 Hz display Nyquist) are notched *before* decimation.
        same_rate = abs(self.emg_fs - self.display_fs) < 1e-6
        if not filter_enabled:
            self._filter_native = None
        elif same_rate:
            self._filter_native = self._filter
        else:
            self._filter_native = CausalEmgFilter(
                self.emg_fs, self.n_channels, lpf_hz=display_lpf, notch_hz=self.line_hz,
                mains_mode=self.mains_mode,
            )

    def set_mounting_rotation(self, slot: str, R: np.ndarray | None) -> None:
        """Optional ``body←chip`` rotation applied before ``q_rel`` (identity if None)."""
        key = "mta" if slot in ("mta", "imu_mta") else "cra"
        with self._lock:
            if R is None:
                self._R_mount[key] = None
            else:
                self._R_mount[key] = np.asarray(R, dtype=np.float64).reshape(3, 3)

    def _body_quat(
        self, slot: str, q: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        R = self._R_mount.get(slot)
        if R is None:
            return q
        return apply_mounting_quat(q, R)

    @property
    def decim(self) -> int:
        """Raw-sample decimation factor to reach the display rate."""
        if self.emg_fs <= 0 or self.display_fs <= 0:
            return 1
        return max(1, int(round(self.emg_fs / self.display_fs)))

    def _rel_s(self, t_us: np.ndarray) -> np.ndarray:
        return (t_us.astype(np.float64) - float(self._t0_us)) / 1e6

    def append_emg_rows(
        self,
        t_us: np.ndarray,
        lo_uv: np.ndarray,
        hi_uv: np.ndarray,
        *,
        raw_count: int | None = None,
        mean_uv: np.ndarray | None = None,
    ) -> None:
        """Append already-decimated display rows (tap path).

        ``mean_uv`` is the per-bucket mean the filtered/envelope traces are
        computed from; without it the hub falls back to the min/max midpoint.
        """
        t_us = np.asarray(t_us, dtype=np.uint64)
        lo_uv = np.asarray(lo_uv, dtype=np.float32).reshape(t_us.shape[0], -1)
        hi_uv = np.asarray(hi_uv, dtype=np.float32).reshape(t_us.shape[0], -1)
        if t_us.shape[0] == 0:
            return
        mean_arr = None
        if mean_uv is not None:
            mean_arr = np.asarray(mean_uv, dtype=np.float32).reshape(t_us.shape[0], -1)
        n_raw = int(raw_count) if raw_count is not None else int(t_us.shape[0]) * self.decim
        self._append(t_us, lo_uv, hi_uv, raw_count=n_raw, mean_uv=mean_arr)

    def append_emg_raw(self, t_us: np.ndarray, samples_uv: np.ndarray) -> None:
        """Decimate full-rate raw samples to display rows, then append."""
        t_us = np.asarray(t_us, dtype=np.uint64)
        samples_uv = np.asarray(samples_uv, dtype=np.float32)
        if t_us.shape[0] == 0:
            return
        if samples_uv.ndim == 1:
            samples_uv = samples_uv[:, None]
        filt_full = env_full = None
        if self._filter_native is not None:
            # Timestamp-aware: a dropped-sample gap must advance the mains
            # reference or the tracker cancels nothing (see process_timed).
            with self._filter_lock:
                filt_full, env_full = self._filter_native.process_timed(t_us, samples_uv)
        d = self.decim
        n = t_us.shape[0]
        if d <= 1:
            self._append(
                t_us, samples_uv, samples_uv, raw_count=n,
                filt=filt_full, env=env_full,
            )
            self._feed_impedance(t_us, samples_uv)
            self._update_rail_leadoff_locked(samples_uv)
            return
        starts = np.arange(0, n, d)
        counts = np.minimum(d, n - starts).astype(np.float32)[:, None]
        lo = np.minimum.reduceat(samples_uv, starts, axis=0)
        hi = np.maximum.reduceat(samples_uv, starts, axis=0)
        mean = (np.add.reduceat(samples_uv, starts, axis=0) / counts).astype(np.float32)
        tt = t_us[starts]
        filt = filt_full[::d] if filt_full is not None else None
        env = env_full[::d] if env_full is not None else None
        self._append(tt, lo, hi, raw_count=n, filt=filt, env=env, mean_uv=mean)
        self._feed_impedance(t_us, samples_uv)
        self._update_rail_leadoff_locked(samples_uv)

    def _update_rail_leadoff_locked(self, samples_uv: np.ndarray) -> None:
        if not self.leadoff_during_recording:
            return
        from jtsmm.config.catalog import EMG_GAINS, lsb_uv as catalog_lsb
        from jtsmm.emg.capture import flat_rail_contact_off

        gain = 6
        for g in EMG_GAINS:
            if abs(catalog_lsb(g) - self.emg_lsb_uv) < 1e-6:
                gain = g
                break
        rail = flat_rail_contact_off(
            samples_uv[:, :VIEWER_EMG_CHANNELS],
            gain=gain,
            n_channels=VIEWER_EMG_CHANNELS,
        )
        with self._lock:
            for i, off in enumerate(rail):
                if not off or i >= self.n_channels:
                    continue
                self._leadoff_bad_streak[i] = LEADOFF_BAD_FRAMES
                # Also restart the "good" count: otherwise all-zero comparator
                # status (comparators off/unknown) unlatches a still-railed input
                # after LEADOFF_GOOD_FRAMES status frames.
                self._leadoff_good_streak[i] = 0
                self._leadoff_latched[i] = True
                self._leadoff_p[i] = True
                self._leadoff_n[i] = True

    def _feed_impedance(self, t_us: np.ndarray, samples_uv: np.ndarray) -> None:
        """Buffer raw rows and refresh live |Z| at most every ~0.5 s of data.

        The estimate (least squares at the real timestamps, see
        jtsmm.emg.impedance) runs outside the hub lock.
        """
        n_ch = min(VIEWER_EMG_CHANNELS, self.n_channels)
        with self._lock:
            if not self._impedance_enabled or not self.leadoff_during_recording:
                return
            if not self.leadoff_ac_tone:
                self._impedance = [
                    {"ch": ch + 1, "z_kohm": None, "quality": "n/a",
                     "note": "AC lead-off tone off (leadoff.ac_tone=false)"}
                    for ch in range(n_ch)
                ]
                return
            self._impedance_buf.append(
                (np.asarray(t_us, dtype=np.int64).copy(),
                 np.asarray(samples_uv, dtype=np.float32).copy())
            )
            self._impedance_buf_n += int(samples_uv.shape[0])
            self._impedance_new += int(samples_uv.shape[0])
            while self._impedance_buf_n > self._impedance_max_samples and self._impedance_buf:
                dropped = self._impedance_buf.pop(0)
                self._impedance_buf_n -= int(dropped[1].shape[0])
            min_n = max(512, int(self.emg_fs * 0.5))
            if self._impedance_buf_n < min_n or self._impedance_new < int(self.emg_fs * 0.5):
                return
            self._impedance_new = 0
            t_block = np.concatenate([b[0] for b in self._impedance_buf])
            block = np.vstack([b[1] for b in self._impedance_buf])
        from jtsmm.emg.impedance import estimate_impedance

        results = estimate_impedance(
            block,
            self.emg_fs,
            current_na=self.leadoff_current_na,
            n_channels=n_ch,
            line_hz=self.line_hz,
            t_us=t_block,
        )
        rows = [
            {"ch": r.channel, "z_kohm": r.z_kohm, "quality": r.quality, "note": r.note}
            for r in results
        ]
        with self._lock:
            if self._impedance_enabled:
                self._impedance = rows

    def _append(
        self,
        t_us: np.ndarray,
        lo_uv: np.ndarray,
        hi_uv: np.ndarray,
        raw_count: int | None = None,
        filt: np.ndarray | None = None,
        env: np.ndarray | None = None,
        mean_uv: np.ndarray | None = None,
    ) -> None:
        if filt is not None and env is not None:
            filt = np.asarray(filt, dtype=np.float32)
            env = np.asarray(env, dtype=np.float32)
        else:
            if mean_uv is not None:
                x = mean_uv.astype(np.float64)
            else:
                x = 0.5 * (lo_uv + hi_uv)
            if self._filter is not None:
                with self._filter_lock:
                    filt, env = self._filter.process_timed(t_us, x)
            else:
                filt = x.astype(np.float32)
                env = np.abs(x).astype(np.float32)
        with self._lock:
            if not self._t0_set:
                self._t0_us = int(t_us[0])
                self._t0_set = True
            self._t_ref_us = int(t_us[-1])
            self._record_emg_rate_locked(
                int(t_us[-1]),
                int(raw_count) if raw_count is not None else int(t_us.shape[0]),
            )
            self._t.append(self._rel_s(t_us).astype(np.float32))
            self._lo.append(lo_uv)
            self._hi.append(hi_uv)
            self._filt.append(filt)
            self._env.append(env)
            self._pending_rows += t_us.shape[0]
            sat_now = saturation_flags(
                np.maximum(np.abs(hi_uv), np.abs(lo_uv)), self.col_lsb_uv
            )
            for i, s in enumerate(sat_now):
                if s:
                    self._sat_hold[i] = 8
                elif self._sat_hold[i] > 0:
                    self._sat_hold[i] -= 1
            self._saturation = [h > 0 for h in self._sat_hold]
            self._trim_locked()

    def _trim_locked(self) -> None:
        while self._pending_rows > self.max_pending_rows and len(self._t) > 1:
            dropped = self._t[0].shape[0]
            self._t.pop(0)
            self._lo.pop(0)
            self._hi.pop(0)
            self._filt.pop(0)
            self._env.pop(0)
            self._pending_rows -= dropped
            self._dropped += dropped

    def set_imu_mta(
        self,
        quat: tuple[float, float, float, float] | None,
        t_us: int,
        report_id: int | None = None,
        accuracy: int | None = None,
    ) -> None:
        self._set_imu_quat("mta", quat, t_us, report_id, accuracy)

    def set_imu_cra(
        self,
        quat: tuple[float, float, float, float] | None,
        t_us: int,
        report_id: int | None = None,
        accuracy: int | None = None,
    ) -> None:
        self._set_imu_quat("cra", quat, t_us, report_id, accuracy)

    def _set_imu_quat(
        self,
        slot: str,
        quat: tuple[float, float, float, float] | None,
        t_us: int,
        report_id: int | None,
        accuracy: int | None = None,
    ) -> None:
        if quat is None:
            return
        with self._lock:
            self._imu_quat[slot] = tuple(float(x) for x in quat)
            self._imu_quat_t[slot] = self._t_rel_scalar(t_us)
            if report_id is not None:
                self._imu_quat_id[slot] = int(report_id)
            if accuracy is not None:
                self._imu_cal[slot]["quat"] = int(accuracy) & 0x03
            self._imu_quat_hist[slot].append((int(t_us), self._imu_quat[slot]))
            self._record_imu_rate_locked(slot, "quat", t_us, 1)
            self._update_derived_locked()

    def push_imu_quat_series(
        self,
        slot: str,
        t_us: np.ndarray,
        quats: np.ndarray,
        report_id: int | None = None,
        accuracies: np.ndarray | None = None,
    ) -> None:
        """Append a batch of fusion quaternions (tap path)."""
        if slot not in IMU_SLOTS:
            return
        t_us = np.asarray(t_us, dtype=np.uint64).reshape(-1)
        quats = np.asarray(quats, dtype=np.float32).reshape(t_us.shape[0], 4)
        if t_us.shape[0] == 0:
            return
        with self._lock:
            for i in range(t_us.shape[0]):
                q = tuple(float(x) for x in quats[i])
                self._imu_quat[slot] = q
                self._imu_quat_t[slot] = self._t_rel_scalar(int(t_us[i]))
                if report_id is not None:
                    self._imu_quat_id[slot] = int(report_id)
                if accuracies is not None and i < len(accuracies):
                    self._imu_cal[slot]["quat"] = int(accuracies[i]) & 0x03
                self._imu_quat_hist[slot].append((int(t_us[i]), q))
            self._record_imu_rate_locked(slot, "quat", int(t_us[-1]), int(t_us.shape[0]))
            self._update_derived_locked()

    def push_imu_motion(
        self,
        slot: str,
        stream: str,
        t_us: int,
        vec: tuple[float, float, float],
        accuracy: int | None = None,
    ) -> None:
        """Append one SI motion sample (accel/gyro/mag/linear_accel) for a slot."""
        if slot not in IMU_SLOTS or stream not in IMU_MOTION_STREAMS or vec is None:
            return
        v = [float(vec[0]), float(vec[1]), float(vec[2])]
        with self._lock:
            self._push_motion_locked(slot, stream, self._t_rel_scalar(t_us), v)
            self._record_imu_rate_locked(slot, stream, t_us, 1)
            if accuracy is not None:
                self._imu_cal[slot][stream] = int(accuracy) & 0x03
            if self._feeds_translation_locked(stream):
                self._add_lin_locked(slot, int(t_us), (v[0], v[1], v[2]))
                self._integrate_translation_locked()

    def push_imu_motion_series(
        self,
        slot: str,
        stream: str,
        t_us: np.ndarray,
        vecs: np.ndarray,
        accuracies: np.ndarray | None = None,
    ) -> None:
        """Append a batch of motion samples (tap path)."""
        if slot not in IMU_SLOTS or stream not in IMU_MOTION_STREAMS:
            return
        t_us = np.asarray(t_us, dtype=np.uint64).reshape(-1)
        vecs = np.asarray(vecs, dtype=np.float32).reshape(t_us.shape[0], 3)
        if t_us.shape[0] == 0:
            return
        with self._lock:
            feeds = self._feeds_translation_locked(stream)
            for i in range(t_us.shape[0]):
                v = [float(vecs[i, 0]), float(vecs[i, 1]), float(vecs[i, 2])]
                self._push_motion_locked(slot, stream, self._t_rel_scalar(int(t_us[i])), v)
                if accuracies is not None and i < len(accuracies):
                    self._imu_cal[slot][stream] = int(accuracies[i]) & 0x03
                if feeds:
                    self._add_lin_locked(slot, int(t_us[i]), (v[0], v[1], v[2]))
            if feeds:
                self._integrate_translation_locked()
            self._record_imu_rate_locked(slot, stream, int(t_us[-1]), int(t_us.shape[0]))

    def _push_motion_locked(self, slot: str, stream: str, t_rel: float, v: list[float]) -> None:
        dq = self._imu_motion[slot][stream]
        dq.append((t_rel, v))
        cutoff = t_rel - IMU_MOTION_KEEP_S
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    def _record_emg_rate_locked(self, t_us: int, raw_count: int) -> None:
        """Record (arrival ts, raw sample count) and prune the rolling window.

        ``raw_count`` is the number of *undecimated* EMG samples represented by
        the batch, so the measured rate reflects the true ADC output rate even
        though the hub stores decimated display rows.
        """
        dq = self._emg_rate
        dq.append((int(t_us), max(0, int(raw_count))))
        cutoff = int(t_us) - int(EMG_RATE_WINDOW_S * 1e6)
        while len(dq) > 1 and dq[0][0] < cutoff:
            dq.popleft()

    def _emg_measured_hz_locked(self) -> float:
        """Measured EMG rate over the window: samples-after-first / Δt."""
        dq = self._emg_rate
        if len(dq) < 2:
            return 0.0
        span_s = (dq[-1][0] - dq[0][0]) / 1e6
        if span_s <= 0:
            return 0.0
        # The first entry only anchors the window start; count everything after.
        n = sum(c for _, c in list(dq)[1:])
        return n / span_s

    def _record_imu_rate_locked(self, slot: str, stream: str, t_us: int, count: int) -> None:
        """Record (arrival ts, sample count) and prune outside the rolling window."""
        dq = self._imu_rate[slot][stream]
        dq.append((int(t_us), max(0, int(count))))
        cutoff = int(t_us) - int(IMU_RATE_WINDOW_S * 1e6)
        while len(dq) > 1 and dq[0][0] < cutoff:
            dq.popleft()

    def _measured_hz_locked(self, slot: str, stream: str) -> float:
        """Measured rate over the rolling window: samples-after-first / Δt."""
        dq = self._imu_rate[slot][stream]
        if len(dq) < 2:
            return 0.0
        span_s = (dq[-1][0] - dq[0][0]) / 1e6
        if span_s <= 0:
            return 0.0
        n = sum(c for _, c in list(dq)[1:])
        return n / span_s

    def note_trigger_events(self, t_us: int, count: int) -> None:
        """Record MCU trigger events (camera sync rate proxy)."""
        if count <= 0:
            return
        with self._lock:
            dq = self._trigger_rate
            dq.append((int(t_us), int(count)))
            cutoff = int(t_us) - int(IMU_RATE_WINDOW_S * 1e6)
            while len(dq) > 1 and dq[0][0] < cutoff:
                dq.popleft()

    def set_camera_ids(self, cam_ids: list[str]) -> None:
        """Declare which cameras are connected (drives the cam dock tile set)."""
        ids = [str(c) for c in cam_ids]
        with self._lock:
            self._camera_order = ids
            # Drop previews for cams that disappeared; keep existing JPEGs for survivors.
            self._camera_jpeg = {k: v for k, v in self._camera_jpeg.items() if k in ids}

    def camera_ids(self) -> list[str]:
        with self._lock:
            return list(self._camera_order)

    def update_camera_jpeg(self, cam_id: str, jpeg: bytes) -> None:
        """Publish one JPEG frame for the MJPEG stream (optical PreviewServer.update)."""
        if not jpeg:
            return
        cid = str(cam_id)
        with self._lock:
            if cid not in self._camera_order:
                self._camera_order.append(cid)
            self._camera_jpeg[cid] = bytes(jpeg)

    def get_camera_jpeg(self, cam_id: str) -> bytes | None:
        with self._lock:
            return self._camera_jpeg.get(str(cam_id))

    def _trigger_measured_hz_locked(self) -> float:
        dq = self._trigger_rate
        if len(dq) < 2:
            return 0.0
        span_s = (dq[-1][0] - dq[0][0]) / 1e6
        if span_s <= 0:
            return 0.0
        n = sum(c for _, c in list(dq)[1:])
        return n / span_s

    def _feeds_translation_locked(self, stream: str) -> bool:
        """Whether ``stream`` drives the translation estimate (see _lin_source)."""
        if stream == "linear_accel" and self._lin_source != "linear_accel":
            if self._lin_source == "accel":  # both streams on: drop the accel pairs
                self._lin_pending.clear()
                self._lin_cra.clear()
                self._lin_last_t = None
                self._zupt.reset()
            self._lin_source = "linear_accel"
        elif stream == "accel" and self._lin_source is None:
            self._lin_source = "accel"
        return stream == self._lin_source

    def _add_lin_locked(self, slot: str, t_us: int, a: tuple[float, float, float]) -> None:
        buf = self._lin_pending if slot == "mta" else self._lin_cra
        if buf and t_us <= buf[-1][0]:
            return                      # duplicate / out of order: drop
        if slot == "mta" and self._lin_last_t is not None and t_us <= self._lin_last_t:
            return
        buf.append((t_us, a))
        if len(buf) > LIN_PAIR_MAX:
            del buf[: len(buf) - LIN_PAIR_MAX]

    def _cra_lin_at(self, t_us: int) -> tuple[float, float, float] | None:
        """CRA linear accel linearly interpolated at ``t_us`` (None across a hole)."""
        buf = self._lin_cra
        j = bisect_left(buf, t_us, key=lambda e: e[0])
        if j < len(buf) and buf[j][0] == t_us:
            return buf[j][1]
        if j == 0 or j >= len(buf):
            return None
        (t0, a0), (t1, a1) = buf[j - 1], buf[j]
        if t1 - t0 > LIN_PAIR_MAX_GAP_US:
            return None
        u = (t_us - t0) / float(t1 - t0)
        return tuple(a0[k] + u * (a1[k] - a0[k]) for k in range(3))  # type: ignore[return-value]

    def _integrate_translation_locked(self) -> None:
        """Advance the ZUPT estimate over every MTA sample CRA data now covers.

        Pairs samples by timestamp. The tap path delivers a whole MTA batch and
        then a whole CRA batch; pairing each new sample with the other slot's
        *latest* value (as this did) matched stale data and fed the integrator
        interleaved, partly repeated intervals -- 1 m/s² for 1 s came out as
        1.79 m instead of 0.5 m.
        """
        pend, cra = self._lin_pending, self._lin_cra
        if not pend or not cra:
            return
        horizon = cra[-1][0]
        hist_mta = list(self._imu_quat_hist["mta"])
        hist_cra = list(self._imu_quat_hist["cra"])
        n_done = 0
        for t_us, a_mta in pend:
            if t_us > horizon:
                break
            n_done += 1
            self._lin_last_t = t_us
            a_cra = self._cra_lin_at(t_us)
            q_mta = quat_at_time(hist_mta, t_us)
            q_cra = quat_at_time(hist_cra, t_us)
            if a_cra is None or q_mta is None or q_cra is None:
                continue
            # Linear accel is in the CHIP frame: rotate it with the raw
            # world<-chip quaternion (a_world = R_wc a_chip), never the
            # mounting-corrected one.
            a_rel = relative_linear_accel(a_mta, q_mta, a_cra, q_cra)
            pos, valid = self._zupt.update(float(t_us) / 1e6, a_rel)
            self._pos_rel = pos
            self._pos_valid = valid
        if n_done:
            del pend[:n_done]
            # Keep one CRA sample at/before the last processed time for interpolation.
            j = bisect_left(cra, self._lin_last_t, key=lambda e: e[0])
            if j > 1:
                del cra[: j - 1]

    def _update_derived_locked(self) -> None:
        q_mta = self._imu_quat["mta"]
        q_cra = self._imu_quat["cra"]
        if q_mta is not None and q_cra is not None:
            self._q_rel = relative_quaternion(
                self._body_quat("mta", q_mta),
                self._body_quat("cra", q_cra),
            )

    def _imu_sample_locked(self, slot: str) -> ImuSample:
        motion: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
        for stream, dq in self._imu_motion[slot].items():
            if dq:
                motion[stream] = (
                    np.asarray([e[0] for e in dq], dtype=np.float32),
                    np.asarray([e[1] for e in dq], dtype=np.float32).reshape(-1, 3),
                )
            dq.clear()
        rates = {
            stream: hz
            for stream in IMU_RATE_STREAMS
            if (hz := self._measured_hz_locked(slot, stream)) > 0
        }
        cal = dict(self._imu_cal[slot])
        return ImuSample(
            quat=self._imu_quat[slot],
            t_s=self._imu_quat_t[slot],
            quat_report_id=self._imu_quat_id[slot],
            motion=motion,
            rates=rates,
            cal=cal,
        )

    def update_imu_cal(self, slot: str, cal: dict[str, int] | None) -> None:
        """Merge latest BNO085 calibration/accuracy for a slot (viewer tap path)."""
        if slot not in IMU_SLOTS or not cal:
            return
        with self._lock:
            for stream, acc in cal.items():
                if stream in IMU_CAL_STREAMS:
                    self._imu_cal[slot][stream] = int(acc) & 0x03

    def _t_rel_scalar(self, t_us: int) -> float:
        if not self._t0_set:
            return 0.0
        return (float(t_us) - float(self._t0_us)) / 1e6

    def _update_leadoff_locked(self, statp: list[bool], statn: list[bool]) -> None:
        """Latch comparator lead-off with hysteresis to avoid LED flicker."""
        for i in range(min(VIEWER_EMG_CHANNELS, self.n_channels)):
            raw_off = bool(statp[i] or statn[i])
            if raw_off:
                self._leadoff_bad_streak[i] += 1
                self._leadoff_good_streak[i] = 0
                if self._leadoff_bad_streak[i] >= LEADOFF_BAD_FRAMES:
                    self._leadoff_latched[i] = True
            else:
                self._leadoff_good_streak[i] += 1
                self._leadoff_bad_streak[i] = 0
                if self._leadoff_good_streak[i] >= LEADOFF_GOOD_FRAMES:
                    self._leadoff_latched[i] = False
            off = self._leadoff_latched[i]
            self._leadoff_p[i] = off
            self._leadoff_n[i] = off

    def _update_drl_leadoff_locked(self, raw_off: bool) -> None:
        if raw_off:
            self._drl_bad_streak += 1
            self._drl_good_streak = 0
            if self._drl_bad_streak >= LEADOFF_BAD_FRAMES:
                self._drl_latched = True
        else:
            self._drl_good_streak += 1
            self._drl_bad_streak = 0
            if self._drl_good_streak >= LEADOFF_GOOD_FRAMES:
                self._drl_latched = False
        self._drl_off = self._drl_latched

    def set_comparators_enabled(self, enabled: bool | None) -> None:
        """Record whether the MCU reports the lead-off comparators powered."""
        with self._lock:
            self._cmp_enabled = enabled

    def _cmp_state_locked(self) -> str:
        """on | off | unknown -- same rule as jtsmm.emg.capture.comparator_state."""
        if self._cmp_enabled is True or (self._cmp_enabled is None and self._cmp_seen):
            return "on"
        if self._cmp_enabled is False:
            return "off"
        return "unknown"

    def set_leadoff(self, status: bytes | None) -> None:
        if not status:
            return
        if not self.leadoff_during_recording:
            with self._lock:
                self._leadoff_p = [False] * self.n_channels
                self._leadoff_n = [False] * self.n_channels
                self._leadoff_bad_streak = [0] * self.n_channels
                self._leadoff_good_streak = [0] * self.n_channels
                self._leadoff_latched = [False] * self.n_channels
                self._drl_off = False
                self._drl_bad_streak = 0
                self._drl_good_streak = 0
                self._drl_latched = False
            return
        st = decode_status_word(status)
        # CH7/CH8 comparator bits are invalid on this board and masked (K8).
        n_cmp = min(VIEWER_EMG_CHANNELS, COMPARATOR_CHANNELS)
        statp = list(st.statp[:n_cmp])
        statn = list(st.statn[:n_cmp])
        with self._lock:
            if any(statp) or any(statn):
                self._cmp_seen = True
            if self._cmp_state_locked() != "on":
                # All-zero bits from powered-down comparators are not "connected":
                # do not latch anything from them (the snapshot says "unknown").
                statp = [False] * n_cmp
                statn = [False] * n_cmp
            self._update_leadoff_locked(statp, statn)
            self._update_drl_leadoff_locked(False)

    def reset_stream(self) -> None:
        """Forget everything tied to the stream position (replay seek).

        Filter state, pending display rows, rate windows, IMU histories and the
        ZUPT estimate all describe the old position; carrying them across a
        seek rings the filters and pairs samples from different times. Call it
        from the producing thread (the filters run outside the lock).
        """
        with self._filter_lock:
            for f in {id(f): f for f in (self._filter, self._filter_native) if f is not None}.values():
                f.reset()
        with self._lock:
            self._t.clear()
            self._lo.clear()
            self._hi.clear()
            self._filt.clear()
            self._env.clear()
            self._pending_rows = 0
            self._emg_rate.clear()
            self._trigger_rate.clear()
            for slot in IMU_SLOTS:
                self._imu_quat_hist[slot].clear()
                for dq in self._imu_motion[slot].values():
                    dq.clear()
                for dq in self._imu_rate[slot].values():
                    dq.clear()
            self._lin_pending.clear()
            self._lin_cra.clear()
            self._lin_last_t = None
            self._lin_source = None
            self._zupt.reset()
            self._pos_rel = None
            self._impedance_buf.clear()
            self._impedance_buf_n = 0
            self._impedance_new = 0
            self._sat_hold = [0] * self.n_channels
            self._saturation = [False] * self.n_channels

    def set_impedance_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._impedance_enabled = bool(enabled)
            self._impedance_new = 0
            if not enabled:
                self._impedance = None
                self._impedance_buf.clear()
                self._impedance_buf_n = 0

    @property
    def leadoff_active(self) -> bool:
        return self.leadoff_during_recording

    def set_stats(self, stats: dict | None) -> None:
        if stats is None:
            return
        with self._lock:
            self._stats = {k: int(v) if isinstance(v, (int, np.integer)) else v
                           for k, v in stats.items()}

    def _health(self) -> dict:
        s = self._stats or {}
        missed = int(s.get("missed_drdy", 0))
        drdy = int(s.get("drdy_count", 0))
        miss_rate = (missed / (drdy + missed)) if (drdy + missed) > 0 else 0.0
        # ~0.2% missed_drdy at 2 kHz is documented as normal (SPI DMA contention).
        emg_ok = int(s.get("emg_drops", 0)) == 0 and miss_rate <= 0.005
        return {
            "emg_ok": bool(emg_ok),
            "tx_ok": int(s.get("tx_drops", 0)) == 0,
            "imu_ok": int(s.get("imu_mta_drops", 0)) == 0
            and int(s.get("imu_cra_drops", 0)) == 0,
            "max_drdy_gap_us": int(s.get("max_drdy_gap_us", 0)),
            "missed_drdy_rate": float(miss_rate),
        }

    def snapshot(self) -> Snapshot:
        with self._lock:
            self._seq += 1
            if self._t:
                t = np.concatenate(self._t)
                lo = np.vstack(self._lo)
                hi = np.vstack(self._hi)
                filt = np.vstack(self._filt)
                env = np.vstack(self._env)
            else:
                t = np.empty(0, dtype=np.float32)
                lo = np.empty((0, self.n_channels), dtype=np.float32)
                hi = np.empty((0, self.n_channels), dtype=np.float32)
                filt = lo
                env = lo
            self._t.clear()
            self._lo.clear()
            self._hi.clear()
            self._filt.clear()
            self._env.clear()
            self._pending_rows = 0
            dropped = self._dropped
            self._dropped = 0
            mta = self._imu_sample_locked("mta")
            cra = self._imu_sample_locked("cra")
            snap = Snapshot(
                seq=self._seq,
                t_ref_us=self._t_ref_us,
                t0_us=self._t0_us,
                dropped=dropped,
                emg_n=int(t.shape[0]),
                emg_channels=self.n_channels,
                emg_fs=self.emg_fs,
                emg_hz=self._emg_measured_hz_locked(),
                emg_lsb_uv=self.emg_lsb_uv,
                emg_t=t,
                emg_lo=lo,
                emg_hi=hi,
                emg_filt=filt,
                emg_env=env,
                camera_hz=self._trigger_measured_hz_locked(),
                imu_mta=mta,
                imu_cra=cra,
                imu_q_rel=self._q_rel,
                imu_pos_rel=self._pos_rel,
                imu_pos_valid=self._pos_valid,
                leadoff_p=list(self._leadoff_p),
                leadoff_n=list(self._leadoff_n),
                leadoff_active=self.leadoff_during_recording,
                leadoff_cmp=self._cmp_state_locked(),
                # Always False: the IN8N comparator bit is masked (K8); the
                # front-end renders DRL's LED as a static n/a. Kept for the
                # wire format.
                drl_leadoff=self._drl_off,
                impedance=list(self._impedance) if (self._impedance_enabled and self._impedance) else None,
                saturation=list(self._saturation),
                stats=self._stats,
                health=self._health(),
            )
        return snap
