# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Affine clock alignment on t_master_us (s06).

Track B camera association: when the camera does not supply a concrete
``trigger_id`` (real picamera2 path), frames are matched by ordered sequence
consume against buffered MCU ``TRIGGER_EVENT`` ids — not by "latest trigger".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jtsmm.protocol import McuRecord, RecordType

# Hard cap so a stopped camera cannot pin the trigger buffer forever.
_TRIGGERS_MAX = 4096


@dataclass
class AffineFit:
    a: float = 1.0
    b: float = 0.0
    n_samples: int = 0
    residual_rms_us: float = 0.0

    def map(self, t_local: float) -> float:
        return self.a * t_local + self.b


@dataclass
class AlignedRecord:
    modality: str
    t_master_us: int
    payload: dict[str, Any]
    residual_us: float = 0.0


@dataclass
class ClockAligner:
    """Route MCU records and camera metadata; maintain per-stream affine fits."""

    window: int = 30
    triggers_max: int = _TRIGGERS_MAX
    predict_min_samples: int = 4
    predict_tolerance_us: float = 4_000.0
    predict_max_jump: int = 8
    _triggers: list[tuple[int, int]] = field(default_factory=list)  # trigger_id, t_master
    _trigger_index: dict[int, int] = field(default_factory=dict)  # id -> t_master_us
    _cam_cursor: dict[str, int | None] = field(default_factory=dict)
    # Shared sequence origin so every camera's first None-id frame maps to the
    # same TRIGGER_EVENT even if an earlier camera already advanced its cursor.
    _seq_origin: int | None = None
    _seq_origin_t: int | None = None
    _cam_fits: dict[str, AffineFit] = field(default_factory=dict)
    _pi_fit: AffineFit = field(default_factory=AffineFit)
    _cam_last_good_t_master: dict[str, int] = field(default_factory=dict)
    _trigger_period_us: float | None = None
    _last_ingest_waiting: bool = False
    drops: dict[str, int] = field(default_factory=dict)
    slip_count: dict[str, int] = field(default_factory=dict)
    camera_drop_recovered: dict[str, int] = field(default_factory=dict)
    validation_fail_count: dict[str, int] = field(default_factory=dict)
    validation_relock_count: dict[str, int] = field(default_factory=dict)
    anomaly_count: dict[str, int] = field(default_factory=dict)
    waiting_for_trigger: dict[str, int] = field(default_factory=dict)
    discarded_trigger_backlog: int = 0
    trigger_stamp_counts: dict[str, int] = field(default_factory=dict)
    trigger_fallback_count: int = 0
    pps_samples: int = 0
    pps_anchor: bool = True
    anchor_info: dict[str, dict[str, Any]] = field(default_factory=dict)
    _assign_pairs: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    assign_resid_max: int = 20000
    _last_trigger_t: int | None = None
    _last_trigger_id: int | None = None
    # Edges lost with no id gap (time step k periods, id step < k).
    trigger_lost_edges: int = 0
    _trigger_dt_us: list[float] = field(default_factory=list)
    # Consecutive SensorTimestamp↔trigger mismatches. After this many, the affine
    # fit is assumed stale (libcamera restart / timestamp jump) and sequence
    # consume re-locks instead of dropping the rest of the take.
    validation_relock_after: int = 8
    _validation_fail_streak: dict[str, int] = field(default_factory=dict)
    _last_host_ns: int | None = None
    _last_master_us: int | None = None
    # --- camera cadence tracking -------------------------------------------
    # Advance a camera's cursor by the number of trigger periods that elapsed in
    # its OWN SensorTimestamps, not by one per delivered frame. The camera drops
    # frames in bursts (measured: SensorTimestamp gaps to 14 periods at p99=8),
    # and advance-by-one then mislabels everything after each burst. This is
    # purely differential -- camera clock deltas against the trigger period -- so
    # unlike the affine-fit prediction it cannot be corrupted by the very
    # assignments it is meant to correct. On the rig take it collapsed the
    # frame<->trigger spread from 59 frames to 0.
    delta_tracking: bool = True
    # Largest gap, in trigger periods, that cadence tracking will believe. Real
    # camera drop bursts measured p99 = 8 and max 14 periods. Anything much
    # larger is a SensorTimestamp discontinuity (libcamera restart), not a drop
    # burst -- believing it would jump the cursor thousands of ids ahead and then
    # wait forever, so those fall through to the fit-based prediction and the
    # validation re-lock instead.
    cadence_max_advance: int = 30
    _cam_last_sensor_us: dict[str, float] = field(default_factory=dict)
    # The cadence REFERENCE id, which is whatever the camera's own clock says the
    # frame was -- not necessarily the id whose timestamp we managed to use. Those
    # differ when a TRIGGER_EVENT was lost over USB (368 of 14037 on the rig take).
    # Substituting a neighbour and then referencing *it* would shift the mapping
    # permanently, once per lost event; keeping the reference on the predicted id
    # confines the damage to that one frame.
    _cam_cadence_ref: dict[str, int] = field(default_factory=dict)
    cadence_advance: dict[str, int] = field(default_factory=dict)
    cadence_missing_trigger: dict[str, int] = field(default_factory=dict)
    # One-shot anchor refinement: with the cadence fix the residual is constant,
    # so its median over a warm-up window measures the anchor error precisely.
    # Only corrects gross errors -- a fraction of a frame is genuine pipeline
    # latency, not a mistake.
    anchor_refine_after: int = 90
    anchor_refine_min_frames: float = 3.0
    anchor_refine_max_spread: float = 2.0
    _anchor_refined: dict[str, bool] = field(default_factory=dict)
    # --- host <-> MCU clock (``_pi_fit``) -----------------------------------
    # Built from real ``pps`` handshake samples only, and every one is kept: the
    # start pass must not be evicted by the end pass, because the slope comes
    # from the distance between the two. A start pass alone spans ~0.2 s, where a
    # free slope is pure noise (rig: 757 ppm -> hundreds of ms by the end of a
    # take), so the fit is offset-only (a = 1) until the samples span at least
    # ``pps_slope_min_span_s`` and a free slope afterwards.
    pps_slope_min_span_s: float = 30.0
    pps_keep_max: int = 1024
    _pps_xs: list[float] = field(default_factory=list)
    _pps_ys: list[float] = field(default_factory=list)
    pi_fit_mode: str | None = None
    # Fallback when no handshake answered: (host processing time, MCU stamp)
    # pairs. Host time here is when the record was *processed*, not when it was
    # stamped, so each pair carries a positive, variable delay -- a slope fitted
    # through them is meaningless (slope ~20 in simulation). Offset-only, taking
    # the least-delayed pair of a recent window, and never labelled ``pi_fit``.
    soft_window: int = 256
    _soft_offsets: list[float] = field(default_factory=list)
    # --- first-frame anchor ---------------------------------------------------
    # Set from the session MCU config via configure_from_mcu(); see
    # expected_sensor_latency_us().
    trigger_rate_hz: float | None = None
    trigger_pulse_width_us: float | None = None
    sensor_latency_extra_us: float = 500.0
    sensor_latency_us: float | None = None
    # Cameras the supervisor started. Buffered triggers are not trimmed below
    # the slowest cursor until each of them has one, so a camera whose first
    # frame lands after another camera's cannot find its trigger already gone.
    expected_cameras: tuple[str, ...] = ()
    trim_wait_max_triggers: int = 600

    def note_host_mcu_pair(self, t_host_ns: int, t_master_us: int) -> None:
        """Soft PPS from a (host processing time, MCU stamp) pair.

        Feeds only the offset-only fallback used while no real ``pps`` sample
        exists; it never touches ``_pi_fit``, so a later handshake is not mixed
        with noisy USB-arrival times.
        """
        t_ns = int(t_host_ns)
        t_us = int(t_master_us)
        self._last_host_ns = t_ns
        self._last_master_us = t_us
        self._soft_offsets.append(float(t_us) - t_ns / 1000.0)
        if len(self._soft_offsets) > int(self.soft_window):
            del self._soft_offsets[0]

    @staticmethod
    def _host_now_ns() -> int:
        """Host CLOCK_MONOTONIC now (the clock the PPS fit maps from)."""
        return time.monotonic_ns()

    def map_pi_ns(self, t_host_ns: int | None = None) -> tuple[int | None, str]:
        """Map host monotonic ns → ``t_master_us``.

        Returns ``(t_master_us, quality)`` where quality is one of:
        ``pi_fit`` (real PPS handshake fit), ``mcu_arrival`` (offset from record
        arrival times, a few ms late at best), ``last_trigger``, ``unavailable``.

        Uses the fit as it stands *now*. Stamps taken live during a take only
        have the start pass (offset-only); calling this again at close, after the
        end ``pps`` pass, re-maps stored ``t_host_ns`` with the final free-slope
        fit -- which is what the supervisor does for ``/events`` and ``/phases``.
        """
        now_ns = int(t_host_ns if t_host_ns is not None else time.monotonic_ns())
        fit = self._pi_fit
        if fit.n_samples >= 1:
            return int(round(fit.map(now_ns / 1000.0))), "pi_fit"
        if self._soft_offsets:
            return int(round(now_ns / 1000.0 + max(self._soft_offsets))), "mcu_arrival"
        if self._last_host_ns is not None and self._last_master_us is not None:
            dt_us = (now_ns - self._last_host_ns) / 1000.0
            return int(round(self._last_master_us + dt_us)), "mcu_arrival"
        if self._triggers:
            return int(self._triggers[-1][1]), "last_trigger"
        return None, "unavailable"

    def ingest_mcu(self, rec: McuRecord) -> list[AlignedRecord]:
        # Soft PPS from every MCU arrival until a true PPS protocol is wired.
        self.note_host_mcu_pair(time.monotonic_ns(), int(rec.t_master_us))
        out: list[AlignedRecord] = []
        if rec.record_type == int(RecordType.TRIGGER_EVENT):
            trig_id = int(rec.decoded.get("trigger_id", 0))
            self.note_trigger(
                trig_id,
                rec.t_master_us,
                stamp_source=rec.decoded.get("stamp_source"),
                fallback=bool(rec.decoded.get("stamp_fallback", False)),
            )
            out.append(
                AlignedRecord("trigger", rec.t_master_us, rec.decoded),
            )
        elif rec.record_type == int(RecordType.LED_FLASH_EVENT):
            out.append(AlignedRecord("led_flash", rec.t_master_us, rec.decoded))
        elif rec.record_type == int(RecordType.EMG_BLOCK):
            out.append(AlignedRecord("emg", rec.t_master_us, rec.decoded))
        elif rec.record_type in (
            int(RecordType.IMU_MTA_QUAT),
            int(RecordType.IMU_MTA_RAW),
            int(RecordType.IMU_MTA_LIN_ACCEL),
        ):
            out.append(AlignedRecord("imu_mta", rec.t_master_us, rec.decoded))
        elif rec.record_type in (
            int(RecordType.IMU_CRA_QUAT),
            int(RecordType.IMU_CRA_RAW),
            int(RecordType.IMU_CRA_LIN_ACCEL),
        ):
            out.append(AlignedRecord("imu_cra", rec.t_master_us, rec.decoded))
        return out

    def reset_trigger_backlog(self) -> int:
        """Drop buffered TRIGGER_EVENTs and the sequence origin.

        Call after cameras are live so a pre-frame pulse cannot become the
        first-frame anchor. Returns how many events were discarded.
        """
        n = len(self._triggers)
        self._triggers.clear()
        self._trigger_index.clear()
        self._cam_cursor.clear()
        self._seq_origin = None
        self._seq_origin_t = None
        # The trigger may be re-armed after a pause; that step is not a period.
        self._last_trigger_t = None
        self._last_trigger_id = None
        self.discarded_trigger_backlog += n
        return n

    def note_trigger(
        self,
        trigger_id: int,
        t_master_us: int,
        *,
        stamp_source: str | int | None = None,
        fallback: bool = False,
    ) -> None:
        tid = int(trigger_id)
        t_us = int(t_master_us)
        # Per-period step: an id gap (TRIGGER_EVENT lost over USB, or an edge
        # the firmware counted but could not stamp, K4) spans several periods
        # and must not inflate the period estimate that cadence tracking
        # divides by.
        step: float | None = None
        d_id = tid - self._last_trigger_id if self._last_trigger_id is not None else 0
        if self._last_trigger_t is not None and t_us > self._last_trigger_t and d_id >= 1:
            step = float(t_us - self._last_trigger_t) / float(d_id)
            ref = self._median_trigger_period_us()
            if ref and len(self._trigger_dt_us) >= 8:
                k = int(round(float(t_us - self._last_trigger_t) / ref))
                if k > d_id:
                    # Time says k periods, ids say fewer: an edge was lost with
                    # no id gap (firmware before K4). Normalise by k instead.
                    self.trigger_lost_edges += k - d_id
                    step = float(t_us - self._last_trigger_t) / float(k)
        if step is not None:
            if self._trigger_period_us is None:
                self._trigger_period_us = step
            else:
                # Slow EMA: trigger period is hardware-driven and should be stable.
                self._trigger_period_us = 0.9 * self._trigger_period_us + 0.1 * step
            self._trigger_dt_us.append(step)
            if len(self._trigger_dt_us) > 64:
                del self._trigger_dt_us[0]
        self._triggers.append((tid, t_us))
        self._trigger_index[tid] = t_us
        self._trim_triggers()
        self._last_trigger_t = t_us
        self._last_trigger_id = tid
        if stamp_source is not None:
            if isinstance(stamp_source, int):
                from jtsmm.protocol.records import TRIGGER_STAMP_NAMES

                key = TRIGGER_STAMP_NAMES.get(stamp_source, f"0x{stamp_source:02X}")
            else:
                key = str(stamp_source)
            self.trigger_stamp_counts[key] = self.trigger_stamp_counts.get(key, 0) + 1
        if fallback:
            self.trigger_fallback_count += 1

    def ingest_cam(
        self,
        cam_id: str,
        sensor_ts_ns: int,
        trigger_id: int | None,
        *,
        frame_idx: int = 0,
        t_host_ns: int | None = None,
    ) -> AlignedRecord | None:
        self._last_ingest_waiting = False
        if trigger_id is None:
            matched_id, t_master = self._assign_next_trigger(
                cam_id, sensor_ts_ns=sensor_ts_ns, t_host_ns=t_host_ns
            )
        else:
            matched_id = int(trigger_id)
            t_master = self._lookup_trigger_exact(matched_id)
            if t_master is not None:
                # Keep exact-match path consistent with sequence consume: advance
                # past this id so shared-buffer trim can proceed if cursors exist.
                self._cam_cursor[cam_id] = matched_id + 1
                self._trim_triggers()

        if t_master is None:
            if not self._last_ingest_waiting:
                self.drops[cam_id] = self.drops.get(cam_id, 0) + 1
            return None

        fit = self._cam_fits.setdefault(cam_id, AffineFit())
        t_local = sensor_ts_ns / 1000.0  # ns -> µs scale for fit
        self._update_fit(fit, t_local, float(t_master))
        mapped = int(fit.map(t_local))
        residual = float(t_master - mapped)
        self._cam_last_good_t_master[cam_id] = int(t_master)
        # Cadence reference: only successful assignments advance it, so a wait or
        # a drop cannot corrupt the next interval.
        self._cam_last_sensor_us[cam_id] = t_local
        if matched_id is not None:
            self._cam_cadence_ref[cam_id] = int(matched_id)
        self._note_assign_pair(cam_id, float(t_master), t_local)
        self._maybe_refine_anchor(cam_id)
        return AlignedRecord(
            "camera",
            t_master,
            {
                "cam_id": cam_id,
                "sensor_timestamp_ns": sensor_ts_ns,
                "trigger_id": matched_id,
                "frame_idx": frame_idx,
            },
            residual_us=residual,
        )

    @property
    def last_ingest_waiting(self) -> bool:
        """True when the last ``ingest_cam`` returned None because the trigger is not buffered yet."""
        return self._last_ingest_waiting

    def trigger_backlog(self, cam_id: str) -> int:
        """Buffered TRIGGER_EVENTs at/after this camera's consume cursor."""
        if not self._triggers:
            return 0
        cursor = self._cam_cursor.get(cam_id)
        if cursor is None:
            return len(self._triggers)
        return sum(1 for tid, _ in self._triggers if tid >= cursor)

    def ingest_pps(self, t_local_ns: int, t_master_us: int) -> None:
        """One host<->MCU clock sample: host CLOCK_MONOTONIC ns vs t_master_us."""
        self._pps_xs.append(t_local_ns / 1000.0)
        self._pps_ys.append(float(t_master_us))
        if len(self._pps_xs) > int(self.pps_keep_max):
            # Thin the interior, never the ends: the span is what fixes the slope.
            del self._pps_xs[1:-1:2]
            del self._pps_ys[1:-1:2]
        self.pps_samples += 1
        self._refit_pps()

    def pps_span_s(self) -> float:
        """Host-time span covered by the real PPS samples, in seconds."""
        if len(self._pps_xs) < 2:
            return 0.0
        return (max(self._pps_xs) - min(self._pps_xs)) / 1e6

    def _refit_pps(self) -> None:
        fit = self._pi_fit
        x = np.asarray(self._pps_xs, dtype=np.float64)
        y = np.asarray(self._pps_ys, dtype=np.float64)
        fit.n_samples = int(x.size)
        if x.size == 0:
            return
        if x.size >= 2 and self.pps_span_s() >= float(self.pps_slope_min_span_s):
            # Centre before lstsq: host/camera clocks are epoch-scale.
            x0, y0 = float(x[0]), float(y[0])
            A = np.vstack([x - x0, np.ones_like(x)]).T
            coef, _, _, _ = np.linalg.lstsq(A, y - y0, rcond=None)
            fit.a = float(coef[0])
            fit.b = float(y0 - fit.a * x0 + coef[1])
            self.pi_fit_mode = "free_slope"
        else:
            fit.a = 1.0
            fit.b = float(np.median(y - x))
            self.pi_fit_mode = "offset_only"
        fit.residual_rms_us = float(np.sqrt(np.mean((y - (fit.a * x + fit.b)) ** 2)))

    def _note_assign_pair(self, cam_id: str, t_master_us: float, t_local_us: float) -> None:
        lst = self._assign_pairs.setdefault(cam_id, [])
        lst.append((t_master_us, t_local_us))
        if len(lst) > self.assign_resid_max:
            del lst[::2]

    def expected_sensor_latency_us(self) -> float:
        """Expected SensorTimestamp minus trigger edge, in µs.

        The IMX296 in trigger mode stamps a frame about one pulse width plus
        ~0.5 ms after the edge (measured 12.5 ms at 30 Hz with a 12 ms pulse).
        Ignoring it left only a 4.2 ms margin before the first-frame anchor
        picks the *next* trigger. ``sensor_latency_us`` overrides the estimate.
        """
        if self.sensor_latency_us is not None:
            return float(self.sensor_latency_us)
        if self.trigger_pulse_width_us:
            return float(self.trigger_pulse_width_us) + float(self.sensor_latency_extra_us)
        return 0.0

    def configure_trigger(
        self, *, rate_hz: float | None = None, pulse_width_us: float | None = None
    ) -> None:
        """Nominal trigger rate and pulse width (from the session MCU config)."""
        if rate_hz is not None and float(rate_hz) > 0:
            self.trigger_rate_hz = float(rate_hz)
        if pulse_width_us is not None and float(pulse_width_us) > 0:
            self.trigger_pulse_width_us = float(pulse_width_us)

    def configure_from_mcu(self, mcu_config: dict[str, Any] | None) -> None:
        """``configure_trigger`` from an MCU apply payload / ``mcu`` block."""
        trig = (mcu_config or {}).get("trigger") or {}
        pulse = trig.get("pulse") or {}
        try:
            self.configure_trigger(
                rate_hz=trig.get("rate_hz"), pulse_width_us=pulse.get("width_us")
            )
        except (TypeError, ValueError):
            pass

    def _anchor_period_us(self) -> float | None:
        """Trigger period for the anchor: measured median, else EMA, else nominal.

        The median needs >= 3 triggers; a camera whose first frame arrives right
        after trigger start must not lose the PPS anchor for want of history.
        """
        period = self._median_trigger_period_us() or self._trigger_period_us
        if (not period or period <= 0) and self.trigger_rate_hz:
            period = 1e6 / float(self.trigger_rate_hz)
        return float(period) if period and period > 0 else None

    def _anchor_trigger_by_time(
        self, t_local_us: float
    ) -> tuple[int, int, float] | str | None:
        """Buffered trigger nearest the PPS-mapped SensorTimestamp.

        The expected sensor latency is subtracted first, so the prediction is
        the trigger edge rather than a point most of a period after it.

        Returns (trigger_id, t_master_us, offset_us); ``"wait"`` when the frame's
        trigger lies beyond the newest buffered one but has already fired by the
        host clock (delivered over USB, not read yet); or None when there is no
        usable fit / no trigger within half a period / the prediction lies in
        the future. Only ever used for a camera's first frame.

        The wait used to be capped at 4 periods past the newest buffered
        trigger. A frame whose trigger was further ahead of the buffer (camera
        start-up with only the first pulses read) then fell through to
        ``oldest_buffered`` and was labelled with the first trigger. What bounds
        the wait is whether the trigger can have fired yet, i.e. host time.
        """
        if not self.pps_anchor or self.pps_samples < 2 or not self._triggers:
            return None
        period = self._anchor_period_us()
        if not period:
            return None
        predicted = self._pi_fit.map(t_local_us) - self.expected_sensor_latency_us()
        newest_t = float(self._triggers[-1][1])
        if predicted > newest_t + period / 2.0:
            now_master, _ = self.map_pi_ns(self._host_now_ns())
            if now_master is not None and predicted <= float(now_master) + period:
                return "wait"
            return None
        best_id, best_t, best_err = None, None, None
        for tid, t_us in self._triggers:
            err = abs(float(t_us) - predicted)
            if best_err is None or err < best_err:
                best_id, best_t, best_err = tid, t_us, err
        if best_id is None or best_err is None or best_err > period / 2.0:
            return None
        return int(best_id), int(best_t), float(best_t) - predicted

    def assignment_residual_stats(self, cam_id: str) -> dict[str, Any] | None:
        """Assigned-trigger time minus the frame's PPS-mapped SensorTimestamp."""
        lst = self._assign_pairs.get(cam_id)
        if not lst or len(lst) < 2 or self._pi_fit.n_samples < 2:
            return None
        arr = np.asarray(lst, dtype=np.float64)
        t_master = arr[:, 0]
        t_local = arr[:, 1]
        r = t_master - (self._pi_fit.a * t_local + self._pi_fit.b)
        t = t_master / 1e6
        p10 = float(np.percentile(r, 10))
        p90 = float(np.percentile(r, 90))
        out: dict[str, Any] = {
            "n": int(r.size),
            "median_us": float(np.median(r)),
            "p10_us": p10,
            "p90_us": p90,
            "pi_fit_n": int(self._pi_fit.n_samples),
        }
        span = float(t[-1] - t[0])
        if span > 1.0:
            slope, _ = np.polyfit(t - t[0], r, 1)
            out["drift_us_per_s"] = float(slope)
        period = self._anchor_period_us()
        if period:
            out["median_frames"] = float(out["median_us"] / period)
            out["spread_frames"] = float((p90 - p10) / period)
        return out

    def _median_trigger_period_us(self) -> float | None:
        if len(self._trigger_dt_us) < 2:
            return None
        return float(np.median(np.asarray(self._trigger_dt_us, dtype=np.float64)))

    def _first_frame_trigger(
        self, *, t_host_ns: int | None
    ) -> tuple[int, int] | None:
        """First completed request: oldest trigger, unless the buffer is pipeline fill.

        IMX296 pulses as soon as ``trig_start`` runs, but libcamera may only
        return a frame after several periods. Those fill pulses have no frame.
        When ``t_host_ns`` is set and the buffer is long (≥6 events, >4 periods),
        lock to the trigger nearest host arrival (usually the newest). Unit tests
        omit ``t_host_ns`` and keep oldest-first.
        """
        if not self._triggers:
            return None
        oldest_id, oldest_t = self._triggers[0]
        newest_id, newest_t = self._triggers[-1]
        period = self._trigger_period_us
        if period is None or period <= 0:
            if len(self._triggers) >= 2:
                period = float(newest_t - oldest_t) / float(len(self._triggers) - 1)
            else:
                period = 33_333.0
        span = float(newest_t - oldest_t)
        pipeline_fill = (
            t_host_ns is not None
            and len(self._triggers) >= 6
            and span > 4.0 * period
        )
        if not pipeline_fill:
            return oldest_id, oldest_t

        target = float(newest_t)
        mapped, quality = self.map_pi_ns(int(t_host_ns))
        if mapped is not None and quality in ("pi_fit", "mcu_arrival"):
            target = float(mapped)

        window = 2.5 * period
        best: tuple[int, int] | None = None
        best_err = float("inf")
        for tid, t_us in self._triggers:
            if float(newest_t - t_us) > window:
                continue
            err = abs(float(t_us) - target)
            if err < best_err:
                best = (int(tid), int(t_us))
                best_err = err
        if best is None:
            return newest_id, newest_t
        skipped = sum(1 for tid, _ in self._triggers if tid < best[0])
        if skipped:
            self.discarded_trigger_backlog += skipped
        return best

    def _assign_next_trigger(
        self,
        cam_id: str,
        *,
        sensor_ts_ns: int,
        t_host_ns: int | None = None,
    ) -> tuple[int | None, int | None]:
        """Ordered sequence consume for frames with trigger_id=None.

        Returns (matched_trigger_id, t_master_us), or (None, None) on drop/wait.
        When the next trigger has not arrived over USB yet, sets
        ``_last_ingest_waiting`` and leaves the cursor unchanged.
        """
        cursor = self._cam_cursor.get(cam_id)

        if cursor is None:
            t_local = sensor_ts_ns / 1000.0
            anchored = self._anchor_trigger_by_time(t_local)
            if anchored == "wait":
                self._last_ingest_waiting = True
                self.waiting_for_trigger[cam_id] = self.waiting_for_trigger.get(cam_id, 0) + 1
                return None, None
            if isinstance(anchored, tuple):
                tid, t_us, offset_us = anchored
                self._cam_cursor[cam_id] = tid + 1
                self.anchor_info[cam_id] = {
                    "method": "pps_time",
                    "trigger_id": int(tid),
                    "offset_us": round(float(offset_us), 1),
                    "latency_us": round(self.expected_sensor_latency_us(), 1),
                    "skipped_triggers": int(tid - self._triggers[0][0]),
                }
                self._trim_triggers()
                return tid, t_us

        if cursor is None:
            if self._seq_origin is None:
                picked = self._first_frame_trigger(t_host_ns=t_host_ns)
                if picked is None:
                    self._last_ingest_waiting = True
                    self.waiting_for_trigger[cam_id] = self.waiting_for_trigger.get(cam_id, 0) + 1
                    return None, None
                origin_id, origin_t = picked
                self._seq_origin = origin_id
                self._seq_origin_t = origin_t
            origin = self._seq_origin
            origin_t = self._trigger_index.get(origin, self._seq_origin_t)
            if origin_t is None:
                self._last_ingest_waiting = True
                self.waiting_for_trigger[cam_id] = self.waiting_for_trigger.get(cam_id, 0) + 1
                return None, None
            self._cam_cursor[cam_id] = origin + 1
            self.anchor_info.setdefault(
                cam_id,
                {
                    "method": "oldest_buffered",
                    "trigger_id": int(origin),
                    "note": "no usable PPS fit / no trigger within half a period",
                },
            )
            self._trim_triggers()
            return origin, origin_t

        # Cadence first: how many trigger periods elapsed in the camera's own
        # clock since its last *assigned* frame? n>1 means the camera dropped
        # n-1 frames and the cursor must skip them.
        cadence = self._cadence_target(cam_id, sensor_ts_ns=sensor_ts_ns, cursor=cursor)
        if cadence is not None:
            t_cad = self._trigger_index.get(cadence)
            if t_cad is None:
                if not self._triggers or cadence > self._triggers[-1][0]:
                    # Not delivered yet -- wait, do not guess. The cadence
                    # reference must NOT move here: the supervisor retries a held
                    # frame on every flush, and each retry would otherwise add
                    # another period to it, stalling the camera for the rest of
                    # the take after a single late TRIGGER_EVENT.
                    self._last_ingest_waiting = True
                    self.waiting_for_trigger[cam_id] = (
                        self.waiting_for_trigger.get(cam_id, 0) + 1
                    )
                    return None, None
                # The event for this frame was lost over USB. Drop this frame
                # rather than label it with a neighbour's time. The reference
                # still advances to the predicted id, so a lost TRIGGER_EVENT
                # cannot shift the mapping.
                self._cam_cadence_ref[cam_id] = cadence
                self.cadence_missing_trigger[cam_id] = (
                    self.cadence_missing_trigger.get(cam_id, 0) + 1
                )
                self._cam_cursor[cam_id] = cadence + 1
                self._cam_last_sensor_us[cam_id] = sensor_ts_ns / 1000.0
                self._trim_triggers()
                return cadence, None
            # Cadence decides *which* id; the existing validation decides whether
            # to trust it, so the fail-streak / re-lock safety net still applies
            # to a frame whose SensorTimestamp disagrees with the affine fit.
            if not self._accept_trigger_match(
                cam_id, sensor_ts_ns=sensor_ts_ns, t_master_us=t_cad
            ):
                self._cam_cadence_ref[cam_id] = cadence
                self._cam_cursor[cam_id] = cadence + 1
                self._cam_last_sensor_us[cam_id] = sensor_ts_ns / 1000.0
                self._trim_triggers()
                return cadence, None
            if cadence > cursor:
                skipped = cadence - cursor
                self.cadence_advance[cam_id] = self.cadence_advance.get(cam_id, 0) + skipped
                # These are frames the camera never delivered: the same thing the
                # fit-based prediction counts, reached by a drift-free route.
                self.camera_drop_recovered[cam_id] = (
                    self.camera_drop_recovered.get(cam_id, 0) + skipped
                )
            self._cam_cursor[cam_id] = cadence + 1
            self._trim_triggers()
            return cadence, t_cad

        predicted = self._predict_trigger_id(cam_id, sensor_ts_ns=sensor_ts_ns, cursor=cursor)
        if predicted is not None and predicted > cursor:
            t_pred = self._trigger_index.get(predicted)
            if t_pred is not None and self._validate_prediction(
                cam_id, sensor_ts_ns=sensor_ts_ns, t_master_us=t_pred
            ):
                self._validation_fail_streak[cam_id] = 0
                self.camera_drop_recovered[cam_id] = self.camera_drop_recovered.get(cam_id, 0) + (
                    predicted - cursor
                )
                self._cam_cursor[cam_id] = predicted + 1
                self._trim_triggers()
                return predicted, t_pred

        t_us = self._trigger_index.get(cursor)
        if t_us is not None:
            if not self._accept_trigger_match(
                cam_id, sensor_ts_ns=sensor_ts_ns, t_master_us=t_us
            ):
                self._cam_cursor[cam_id] = cursor + 1
                self._trim_triggers()
                return cursor, None
            self._cam_cursor[cam_id] = cursor + 1
            self._trim_triggers()
            return cursor, t_us

        # Expected id missing: slip forward if a greater id is already buffered,
        # otherwise the trigger has not arrived over USB yet (wait, don't drop).
        greater = self._first_trigger_ge(cursor)
        if greater is None:
            self._last_ingest_waiting = True
            self.waiting_for_trigger[cam_id] = self.waiting_for_trigger.get(cam_id, 0) + 1
            return None, None

        slip_id, slip_t = greater
        self.slip_count[cam_id] = self.slip_count.get(cam_id, 0) + (slip_id - cursor)
        if not self._accept_trigger_match(
            cam_id, sensor_ts_ns=sensor_ts_ns, t_master_us=slip_t
        ):
            self._cam_cursor[cam_id] = slip_id + 1
            self._trim_triggers()
            return slip_id, None
        self._cam_cursor[cam_id] = slip_id + 1
        self._trim_triggers()
        return slip_id, slip_t

    def _cadence_target(
        self, cam_id: str, *, sensor_ts_ns: int | None, cursor: int
    ) -> int | None:
        """Trigger id implied by the camera's own inter-frame interval.

        ``None`` when disabled, when the trigger period is not yet known, or on
        a camera's first assigned frame (no previous timestamp to difference).
        """
        if not self.delta_tracking or sensor_ts_ns is None:
            return None
        period = self._trigger_period_us or self._median_trigger_period_us()
        if not period or period <= 0:
            return None
        prev_us = self._cam_last_sensor_us.get(cam_id)
        prev_id = self._cam_cadence_ref.get(cam_id)
        if prev_us is None or prev_id is None:
            return None
        n = int(round((sensor_ts_ns / 1000.0 - prev_us) / period))
        if n < 1 or n > int(self.cadence_max_advance):
            # Not advancing by a whole period, or a jump too large to be a drop
            # burst: fall back to the ordinary path rather than guessing.
            return None
        return prev_id + n

    def _maybe_refine_anchor(self, cam_id: str) -> None:
        """Shift the cursor once if the warm-up residual shows a gross anchor error."""
        if self._anchor_refined.get(cam_id):
            return
        pairs = self._assign_pairs.get(cam_id)
        if not pairs or len(pairs) < int(self.anchor_refine_after):
            return
        stats = self.assignment_residual_stats(cam_id)
        if stats is None:
            return
        self._anchor_refined[cam_id] = True
        med = stats.get("median_frames")
        if med is None or abs(float(med)) < float(self.anchor_refine_min_frames):
            return
        # Only trust the median once the association is locked. A large spread
        # means the frame<->trigger tracking is still wrong, so the median is not
        # measuring an anchor offset and "correcting" by it makes things worse
        # (observed: a -13760 frame shift on an unlocked camera).
        spread = stats.get("spread_frames")
        if spread is None or float(spread) > float(self.anchor_refine_max_spread):
            self._anchor_refined[cam_id] = False  # retry once it locks
            return
        shift = int(round(float(med)))
        cursor = self._cam_cursor.get(cam_id)
        last = self._cam_cadence_ref.get(cam_id)
        if cursor is None or last is None:
            return
        self._cam_cursor[cam_id] = cursor - shift
        self._cam_cadence_ref[cam_id] = last - shift
        # The camera fit was learnt from the wrong pairing and now sits `shift`
        # periods off: left alone, validation would reject the corrected frames
        # until a re-lock. Start it over from the corrected association.
        self._reset_cam_fit(cam_id)
        self._validation_fail_streak[cam_id] = 0
        info = self.anchor_info.setdefault(cam_id, {})
        info["refined_by_frames"] = -shift
        # Frames already written with the old labels (they are not rewritten);
        # the integrity check reports them, since the residual history that
        # showed the offset is cleared below.
        info["refined_after_frames"] = len(pairs)
        info["pre_refine_median_frames"] = round(float(med), 2)
        # Start the residual history over so the reported median describes the
        # corrected association rather than a mix of before and after.
        self._assign_pairs[cam_id] = []

    def _predict_trigger_id(
        self,
        cam_id: str,
        *,
        sensor_ts_ns: int | None,
        cursor: int,
    ) -> int | None:
        if sensor_ts_ns is None:
            return None
        fit = self._cam_fits.get(cam_id)
        period = self._trigger_period_us
        last_good = self._cam_last_good_t_master.get(cam_id)
        if fit is None or fit.n_samples < self.predict_min_samples or period is None or period <= 0 or last_good is None:
            return None
        # Refuse wild jumps while the affine fit is still noisy.
        if fit.residual_rms_us > max(self.predict_tolerance_us, 1_000.0):
            return None

        predicted_t_master = fit.map(sensor_ts_ns / 1000.0)
        expected_ticks = int(round((predicted_t_master - float(last_good)) / period))
        if expected_ticks <= 0:
            self.anomaly_count[cam_id] = self.anomaly_count.get(cam_id, 0) + 1
            return cursor
        jump = max(0, expected_ticks - 1)
        if jump > self.predict_max_jump:
            jump = self.predict_max_jump
        return cursor + jump

    def _validate_prediction(self, cam_id: str, *, sensor_ts_ns: int, t_master_us: int) -> bool:
        fit = self._cam_fits.get(cam_id)
        if fit is None or fit.n_samples < self.predict_min_samples:
            return True
        t_local = sensor_ts_ns / 1000.0
        predicted_t_master = fit.map(t_local)
        tol = max(self.predict_tolerance_us, 6.0 * max(fit.residual_rms_us, 0.0))
        return abs(float(t_master_us) - predicted_t_master) <= tol

    def _reset_cam_fit(self, cam_id: str) -> None:
        self._cam_fits[cam_id] = AffineFit()
        self._cam_last_good_t_master.pop(cam_id, None)

    def _accept_trigger_match(
        self, cam_id: str, *, sensor_ts_ns: int, t_master_us: int
    ) -> bool:
        """True → pair this frame with ``t_master_us``.

        A short streak of SensorTimestamp mismatches drops the frame (one-off
        bad trigger). A long streak means the affine fit is stale — typical of
        a libcamera restart that jumps SensorTimestamp — so we accept the
        sequence pairing and re-lock the fit instead of dropping the rest of
        the take.
        """
        if self._validate_prediction(
            cam_id, sensor_ts_ns=sensor_ts_ns, t_master_us=t_master_us
        ):
            self._validation_fail_streak[cam_id] = 0
            return True
        self.validation_fail_count[cam_id] = self.validation_fail_count.get(cam_id, 0) + 1
        streak = self._validation_fail_streak.get(cam_id, 0) + 1
        self._validation_fail_streak[cam_id] = streak
        if streak < int(self.validation_relock_after):
            return False
        self._reset_cam_fit(cam_id)
        self._validation_fail_streak[cam_id] = 0
        self.validation_relock_count[cam_id] = self.validation_relock_count.get(cam_id, 0) + 1
        return True

    def _first_trigger_ge(self, min_id: int) -> tuple[int, int] | None:
        for tid, t_us in self._triggers:
            if tid >= min_id:
                return tid, t_us
        return None

    def _lookup_trigger_exact(self, trigger_id: int) -> int | None:
        return self._trigger_index.get(int(trigger_id))

    def _trim_triggers(self) -> None:
        """Drop leading triggers already consumed by every known camera cursor.

        Also enforces ``triggers_max`` so a stalled camera cannot grow the list
        without bound. Late-locking cameras still resolve the shared sequence
        origin via ``_seq_origin`` / ``_seq_origin_t`` even after list trim.
        """
        cursors = [c for c in self._cam_cursor.values() if c is not None]
        # A camera that has not delivered its first frame yet needs the triggers
        # around it for the anchor. Hold the watermark until every expected
        # camera has a cursor -- or the buffer shows it is not coming.
        pending = [
            c for c in self.expected_cameras
            if self._cam_cursor.get(c) is None
        ]
        if pending and len(self._triggers) <= int(self.trim_wait_max_triggers):
            cursors = []
        if cursors:
            watermark = min(cursors)
            while self._triggers and self._triggers[0][0] < watermark:
                old_id, _ = self._triggers.pop(0)
                self._trigger_index.pop(old_id, None)

        while len(self._triggers) > self.triggers_max:
            old_id, _ = self._triggers.pop(0)
            self._trigger_index.pop(old_id, None)

    def _update_fit(self, fit: AffineFit, t_local: float, t_master: float) -> None:
        # Rolling WLS for t_master ≈ a * t_local + b. Camera SensorTimestamp
        # values are epoch-scale (ns→µs ≈ 1e15); fitting those uncentered in
        # float64 makes the design matrix [x, 1] rank-deficient and produces
        # nonsense a/b plus a huge residual_rms_us. Center before lstsq.
        if not hasattr(fit, "_xs"):
            fit._xs = []  # type: ignore[attr-defined]
            fit._ys = []  # type: ignore[attr-defined]
        xs: list[float] = fit._xs  # type: ignore[attr-defined]
        ys: list[float] = fit._ys  # type: ignore[attr-defined]
        xs.append(t_local)
        ys.append(t_master)
        if len(xs) > self.window:
            xs.pop(0)
            ys.pop(0)
        if len(xs) < 2:
            fit.b = t_master - t_local
            fit.a = 1.0
            fit.n_samples = len(xs)
            fit.residual_rms_us = 0.0
            return
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        x0 = float(x[0])
        y0 = float(y[0])
        x_c = x - x0
        y_c = y - y0
        # Degenerate / near-constant window: keep a=1 identity slope.
        if float(np.max(np.abs(x_c))) < 1e-9:
            fit.a = 1.0
            fit.b = float(np.mean(y - x))
        else:
            A = np.vstack([x_c, np.ones_like(x_c)]).T
            coef, _, _, _ = np.linalg.lstsq(A, y_c, rcond=None)
            fit.a = float(coef[0])
            fit.b = float(y0 - fit.a * x0 + coef[1])
        fit.n_samples = len(xs)
        pred = fit.a * x + fit.b
        fit.residual_rms_us = float(np.sqrt(np.mean((y - pred) ** 2)))

    def sync_report(self) -> dict[str, Any]:
        cameras: dict[str, Any] = {}
        for cid, f in self._cam_fits.items():
            entry: dict[str, Any] = {"a": f.a, "b": f.b, "residual_rms_us": f.residual_rms_us}
            stats = self.assignment_residual_stats(cid)
            if stats is not None:
                entry["assign_residual"] = stats
            cameras[cid] = entry
        return {
            "pi_to_master": {
                "a": self._pi_fit.a,
                "b": self._pi_fit.b,
                "residual_rms_us": self._pi_fit.residual_rms_us,
                "n_samples": int(self._pi_fit.n_samples),
                "pps_samples": int(self.pps_samples),
                "mode": self.pi_fit_mode,
                "span_s": round(self.pps_span_s(), 3),
            },
            "cameras": cameras,
            "drops": dict(self.drops),
            "slip_count": dict(self.slip_count),
            "camera_drop_recovered": dict(self.camera_drop_recovered),
            "validation_fail_count": dict(self.validation_fail_count),
            "validation_relock_count": dict(self.validation_relock_count),
            "anomaly_count": dict(self.anomaly_count),
            "waiting_for_trigger": dict(self.waiting_for_trigger),
            "discarded_trigger_backlog": int(self.discarded_trigger_backlog),
            "pi_fit_n_samples": int(self._pi_fit.n_samples),
            "trigger_stamp": {
                "counts": dict(self.trigger_stamp_counts),
                "fallback": int(self.trigger_fallback_count),
            },
            "camera_anchor": {cid: dict(v) for cid, v in self.anchor_info.items()},
            "cadence_advance": dict(self.cadence_advance),
            "cadence_missing_trigger": dict(self.cadence_missing_trigger),
            "trigger_lost_edges": int(self.trigger_lost_edges),
        }
