# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Sources feed the `Hub`: live (owns USB), tap (in-recording), replay, mock.

All three modes converge on the same `Hub`/server/frontend; they differ only in
where samples come from. Each source runs its own daemon thread and pushes into
the hub, which coalesces to ~30 Hz snapshots. Sources never block on the hub
(the hub drops rows when a consumer lags), preserving the single-USB-owner and
real-time capture guarantees.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from jtsmm.config.catalog import (
    DEFAULT_EMG_SAMPLE_RATE_SPS,
    DRL_STREAM_INDEX,
    FULL_CHANNEL_NAMES,
    MUSCLE_CHANNEL_NAMES,
    VIEWER_EMG_CHANNELS,
    VIEWER_EMG_PLOT_COUNT,
    VIEWER_EMG_PLOT_NAMES,
    VIEWER_EMG_STREAM_COLS,
    channel_lsb_uv,
    lsb_uv,
)
from jtsmm.viewer.hub import Hub

__all__ = ["Source", "LiveSource", "TapSource", "ReplaySource", "MockSource"]


def _slice_emg_for_viewer(samples_uv: np.ndarray) -> np.ndarray:
    """Muscle columns plus CH8 DRL from an 8-wide stream (drops CH7)."""
    arr = np.asarray(samples_uv, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, VIEWER_EMG_PLOT_COUNT), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] <= DRL_STREAM_INDEX:
        return arr[:, : min(VIEWER_EMG_PLOT_COUNT, arr.shape[1])]
    return arr[:, VIEWER_EMG_STREAM_COLS].astype(np.float32, copy=False)


def _counts_to_uv(counts: Any, gain: int) -> np.ndarray:
    """ADC counts (N, C) -> µV with per-channel LSB (CH7/CH8 at fixed gain 24)."""
    arr = np.asarray(counts, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.size == 0:
        return arr
    return arr * np.asarray(channel_lsb_uv(gain, arr.shape[1]), dtype=np.float32)


class Source(ABC):
    """Produces samples into a `Hub`. Replay controls are no-ops by default."""

    kind = "base"

    @abstractmethod
    def meta(self) -> dict[str, Any]:
        """Session metadata for GET /api/session."""

    @abstractmethod
    def make_hub(self) -> Hub:
        """Create a hub configured for this source's stream geometry."""

    @abstractmethod
    def start(self, hub: Hub) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    # Replay controls (overridden by ReplaySource).
    def seek(self, t_s: float) -> None:  # noqa: B027 - intentional no-op
        pass

    def set_speed(self, speed: float) -> None:  # noqa: B027
        pass

    def set_paused(self, paused: bool) -> None:  # noqa: B027
        pass

    def position(self) -> dict[str, Any]:
        return {}


def _emg_meta(
    fs: float,
    gain: int,
    n_channels: int,
    names: list[str],
    emg: dict | None = None,
    line_hz: float = 50.0,
    mains_mode: str = "track",
) -> dict[str, Any]:
    leadoff = (emg or {}).get("leadoff") or {}
    return {
        "line_hz": float(line_hz),
        "emg_fs": float(fs),
        "emg_gain": int(gain),
        "emg_lsb_uv": lsb_uv(gain),
        "n_channels": int(n_channels),
        "muscle_channels": VIEWER_EMG_CHANNELS,
        "channel_names": list(names),
        "leadoff_during_recording": bool(leadoff.get("during_recording")),
        "leadoff_current_na": int(leadoff.get("current_na", 6) or 6),
    }


def _session_line_hz(meta: dict[str, Any], default: float = 50.0) -> float:
    """host.emg_check.line_hz recorded in a session's /meta, or ``default``."""
    raw = meta.get("host_config_json")
    if raw is None:
        return default
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        doc = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return default
    host = doc.get("host") if isinstance(doc, dict) else None
    check = (host or {}).get("emg_check") or {}
    try:
        return float(check.get("line_hz", default)) or default
    except (TypeError, ValueError):
        return default


def _session_mains_mode(meta: dict[str, Any]) -> str:
    """host.emg_filter.mains_mode recorded with the session, else the default config's."""
    raw = meta.get("host_config_json")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            doc = json.loads(str(raw))
        except (json.JSONDecodeError, TypeError):
            doc = None
        if isinstance(doc, dict):
            try:
                from jtsmm.config.registry import emg_filter_mains_mode

                if ((doc.get("host") or {}).get("emg_filter") or {}).get("mains_mode"):
                    return emg_filter_mains_mode(doc)
            except Exception:  # noqa: BLE001 - fall back to the default config
                pass
    return _mains_mode_default()


def _mains_mode_default() -> str:
    """Viewer mains mode from config, falling back to the tracking canceller."""
    try:
        from jtsmm.config import load
        from jtsmm.config.registry import emg_filter_mains_mode

        return emg_filter_mains_mode(load())
    except Exception:  # noqa: BLE001 - replay must work without a config
        return "track"


def _hub_from_emg(
    emg: dict,
    *,
    n_channels: int = VIEWER_EMG_PLOT_COUNT,
    names: list[str] | None = None,
    line_hz: float = 50.0,
    mains_mode: str = "track",
) -> Hub:
    fs = int(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS) or DEFAULT_EMG_SAMPLE_RATE_SPS)
    gain = int(emg.get("default_gain", 6) or 6)
    leadoff = emg.get("leadoff") or {}
    return Hub(
        emg_fs=fs,
        emg_lsb_uv=lsb_uv(gain),
        n_channels=n_channels,
        channel_names=list(names) if names else list(VIEWER_EMG_PLOT_NAMES),
        leadoff_during_recording=bool(leadoff.get("during_recording")),
        leadoff_current_na=float(leadoff.get("current_na", 6) or 6),
        leadoff_ac_tone=bool(leadoff.get("ac_tone")),
        line_hz=line_hz,
        mains_mode=mains_mode,
    )


class _ThreadedSource(Source):
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._hub: Hub | None = None

    def _spawn(self, hub: Hub, target: Any) -> None:
        self._hub = hub
        self._stop.clear()
        self._thread = threading.Thread(target=target, name=f"viewer-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class LiveSource(_ThreadedSource):
    """Standalone live monitor: owns the USB endpoint (not for use while recording)."""

    kind = "live"

    def __init__(
        self,
        config_path: Path | str | None = None,
        port: str | None = None,
        *,
        bias_drive: bool | None = None,
        bias_channels: str | None = None,
        active_channels: str | None = None,
        leadoff: bool | None = None,
        default_gain: int | None = None,
        monitoring: bool = False,
    ) -> None:
        super().__init__()
        self.config_path = config_path
        self.port = port
        self._cap: Any = None
        self._serial_port: str | None = None
        self._skip_apply = False
        from jtsmm.config import apply_emg_overrides, load

        self._resolved = load(config_path)
        apply_emg_overrides(
            self._resolved,
            bias_drive=bias_drive,
            bias_channels=bias_channels,
            active_channels=active_channels,
            leadoff=leadoff,
            default_gain=default_gain,
            monitoring=monitoring,
        )
        emg = (self._resolved.get("mcu") or {}).get("emg") or {}
        self._fs = int(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS) or DEFAULT_EMG_SAMPLE_RATE_SPS)
        self._gain = int(emg.get("default_gain", 6) or 6)
        self._emg = emg
        self._cam_feed: Any = None
        from jtsmm.config.registry import emg_check_line_hz, emg_filter_mains_mode
        self._line_hz = emg_check_line_hz(self._resolved)
        self._mains_mode = emg_filter_mains_mode(self._resolved)
        self._mcu_logs = 0

    def _imu_enabled_from_config(self) -> dict[str, bool]:
        """Which IMU slots the config enables (the viewer starts only those)."""
        from jtsmm.config.mcu_apply import mcu_stream_enables

        en = mcu_stream_enables(self._resolved)
        return {"mta": en["imu_mta"], "cra": en["imu_cra"]}

    def meta(self) -> dict[str, Any]:
        m = _emg_meta(
            self._fs, self._gain, VIEWER_EMG_PLOT_COUNT, list(VIEWER_EMG_PLOT_NAMES),
            self._emg, line_hz=self._line_hz, mains_mode=self._mains_mode,
        )
        cams = self._cam_feed.camera_ids if self._cam_feed is not None else []
        m.update({"kind": self.kind, "mode": "live", "duration_s": 0.0, "camera_ids": list(cams)})
        m["imu_enabled"] = self._imu_enabled_from_config()
        return m

    def make_hub(self) -> Hub:
        return _hub_from_emg(self._emg, line_hz=self._line_hz, mains_mode=self._mains_mode)

    def start(self, hub: Hub) -> None:
        from jtsmm import native
        from jtsmm.capture.native_backend import NativeCapture
        from jtsmm.capture.serial_util import find_hat_port
        from jtsmm.config import apply_mcu, open_hat_serial, start_mcu_streams, stop_mcu_streams
        from jtsmm.config.mcu_apply import mcu_stream_enables

        if not native.use_native():
            raise RuntimeError("LiveSource requires the native capture extension.")
        self._serial_port = find_hat_port(self.port or (self._resolved.get("hat") or {}).get("serial_port"))
        ser = open_hat_serial(str(self._serial_port))
        try:
            # A prior `jtsmm view`/`record` that didn't shut down cleanly (killed,
            # crashed, or an earlier stop that failed silently) can leave the ADS1299
            # mid-stream. Sending `config apply` into that state races the DRDY-driven reads
            # against register writes on the same SPI bus and can wedge the MCU
            # until power-cycled. Force streams off first; harmless if already idle.
            stop_mcu_streams(ser)
            time.sleep(0.2)
            # A failed apply leaves the MCU on another gain/rate than the one
            # the viewer scales by: refuse to show mis-scaled data.
            if not apply_mcu(ser, self._resolved, persist=False):
                raise RuntimeError("MCU config apply failed; not starting the live viewer")
            if hub.leadoff_active:
                from jtsmm.emg.capture import query_comparators_enabled

                hub.set_comparators_enabled(query_comparators_enabled(ser))
            # Open the USB reader before the streams start: libusb takes ~1.5 s to
            # come up, and EMG produced meanwhile overflowed the MCU ring (~1000
            # emg_drops at 2 kHz on every start, measured on the HAT).
            self._cap = NativeCapture(port=str(self._serial_port))
            self._cap.start()
            # Only the streams the config enables (as `jtsmm record` does): a
            # disabled IMU slot is refused by the firmware with a warning.
            en = mcu_stream_enables(self._resolved)
            for slot, on in self._imu_enabled_from_config().items():
                if not on:
                    print(f"viewer: IMU {slot.upper()} disabled in the config "
                          f"(mcu.imu_{slot}.enabled: false) -- not started", file=sys.stderr)
            start_mcu_streams(ser, emg=en["emg"], trigger=en["trigger"],
                              imu_mta=en["imu_mta"], imu_cra=en["imu_cra"])
        except BaseException:
            if self._cap is not None:
                self._cap.stop()
                self._cap = None
            raise
        finally:
            ser.close()
        self._cap.settle(quiet_ms=400)
        self._lsb = lsb_uv(self._gain)
        from jtsmm.viewer.cam_preview import CamPreviewFeed

        host_cams = (self._resolved.get("host") or {}).get("cameras") or {}
        self._cam_feed = CamPreviewFeed()
        self._cam_feed.start_live(hub, host_cams)
        self._spawn(hub, self._run)

    def _run(self) -> None:
        from jtsmm.imu.units import accel_to_si, gyro_to_si, mag_to_si
        from jtsmm.protocol.types import RecordType

        hub = self._hub
        assert hub is not None and self._cap is not None
        emg_t: list[int] = []
        emg_ch: list[list[int]] = []
        last_status: bytes | None = None
        last_push = time.monotonic()

        raw_to_si = {"accel": accel_to_si, "gyro": gyro_to_si, "mag": mag_to_si}

        def push_raw(slot: str, dec: dict, t_us: int) -> None:
            # Tagged RAW carries exactly one of accel/gyro/mag as int16 counts.
            for stream, conv in raw_to_si.items():
                counts = dec.get(stream)
                if counts is not None:
                    acc = dec.get(f"{stream}_accuracy")
                    hub.push_imu_motion(slot, stream, t_us, conv(counts), accuracy=acc)

        while not self._stop.is_set():
            recs = self._cap.drain_records()
            if not recs:
                time.sleep(0.005)
            for rec in recs:
                rt = rec.record_type
                dec = rec.decoded
                t_us = int(rec.t_master_us)
                if rt == int(RecordType.EMG_BLOCK):
                    ch = dec.get("channels")
                    if ch:
                        emg_t.append(t_us)
                        emg_ch.append(ch)
                        last_status = dec.get("status")
                elif rt == int(RecordType.IMU_MTA_QUAT):
                    hub.set_imu_mta(
                        dec.get("quat"), t_us, dec.get("quat_report_id"),
                        dec.get("quat_accuracy"),
                    )
                elif rt == int(RecordType.IMU_CRA_QUAT):
                    hub.set_imu_cra(
                        dec.get("quat"), t_us, dec.get("quat_report_id"),
                        dec.get("quat_accuracy"),
                    )
                elif rt == int(RecordType.IMU_MTA_RAW):
                    push_raw("mta", dec, t_us)
                elif rt == int(RecordType.IMU_CRA_RAW):
                    push_raw("cra", dec, t_us)
                elif rt == int(RecordType.IMU_MTA_LIN_ACCEL):
                    hub.push_imu_motion(
                        "mta", "linear_accel", t_us, dec.get("linear_accel"),
                        accuracy=dec.get("linear_accel_accuracy"),
                    )
                elif rt == int(RecordType.IMU_CRA_LIN_ACCEL):
                    hub.push_imu_motion(
                        "cra", "linear_accel", t_us, dec.get("linear_accel"),
                        accuracy=dec.get("linear_accel_accuracy"),
                    )
                elif rt == int(RecordType.STATS):
                    hub.set_stats(dict(dec))
                elif rt == int(RecordType.TRIGGER_EVENT):
                    hub.note_trigger_events(t_us, 1)
                elif rt == int(RecordType.LOG):
                    # Firmware diagnosis (IMU NOT FOUND / WATCHDOG / recover), as
                    # `jtsmm record` prints it; capped so a looping message cannot flood.
                    self._mcu_logs += 1
                    if self._mcu_logs <= 200:
                        print(f"mcu: {str(dec.get('text', '')).rstrip()}", file=sys.stderr)
            now = time.monotonic()
            if emg_t and (now - last_push) >= 0.02:
                t = np.asarray(emg_t, dtype=np.uint64)
                uv = _slice_emg_for_viewer(_counts_to_uv(emg_ch, self._gain))
                hub.append_emg_raw(t, uv)
                if last_status is not None:
                    hub.set_leadoff(last_status)
                emg_t = []
                emg_ch = []
                last_push = now

    def stop(self) -> None:
        super().stop()
        if self._cam_feed is not None:
            try:
                self._cam_feed.stop()
            finally:
                self._cam_feed = None
        port = self._serial_port
        if self._cap is not None:
            try:
                self._cap.stop()
            finally:
                self._cap = None
        if port:
            try:
                from jtsmm.config import stop_mcu_streams_robust

                stop_mcu_streams_robust(str(port))
            except Exception as exc:
                # Don't hide this: an MCU that didn't confirm stopping may still be
                # streaming, and the *next* `apply_mcu` racing that stream is what
                # wedges the board. Surface it so the user knows to check/power-cycle
                # instead of silently limping into the next session.
                import sys

                print(f"warning: failed to stop MCU streams cleanly on {port}: {exc}", file=sys.stderr)
            self._serial_port = None


class TapSource(_ThreadedSource):
    """In-recording monitor: reads the recorder's native tap (no USB of its own).

    By default the tap hands over every sample (``bucket_samples=1``) and the
    hub filters and decimates them exactly as `jtsmm view` does (append_emg_raw).
    The old pre-bucketed rows (min/max/mean of N samples, filtered at the
    500 Hz display rate) let 250-500 Hz content fold into the filtered trace,
    skipped the saturated-input lead-off check and the live |Z|, so the
    recording viewer looked worse than `jtsmm view` on the same signal
    (bench, 2026-09-28). ``bucket_samples > 1`` keeps that lighter path.
    """

    kind = "tap"

    def __init__(
        self,
        native_cap: Any,
        *,
        emg_fs: float,
        emg_gain: int,
        bucket_samples: int | None = None,
        poll_hz: float = 30.0,
        channel_names: list[str] | None = None,
        leadoff_during_recording: bool = False,
        leadoff_current_na: float = 6.0,
        line_hz: float = 50.0,
        mains_mode: str = "track",
        imu_enabled: dict[str, bool] | None = None,
        comparators_enabled: bool | None = None,
        leadoff_ac_tone: bool = False,
    ) -> None:
        super().__init__()
        self._cap = native_cap
        self._cmp_enabled = comparators_enabled
        self._ac_tone = bool(leadoff_ac_tone)
        self._imu_enabled = dict(imu_enabled) if imu_enabled is not None else None
        self._line_hz = float(line_hz) or 50.0
        self._mains_mode = str(mains_mode or "track")
        self._fs = float(emg_fs)
        self._gain = int(emg_gain)
        self._lsb = lsb_uv(self._gain)
        # Default 1: full-rate samples, processed like `jtsmm view` (see above).
        self._bucket = int(bucket_samples) if bucket_samples else 1
        self._poll_dt = 1.0 / poll_hz if poll_hz > 0 else 0.033
        self._names = list(channel_names) if channel_names else list(FULL_CHANNEL_NAMES)
        self._leadoff_during = bool(leadoff_during_recording)
        self._leadoff_na = float(leadoff_current_na)
        self._last_emg_seen: int | None = None

    def meta(self) -> dict[str, Any]:
        m = _emg_meta(
            self._fs, self._gain, VIEWER_EMG_PLOT_COUNT, list(VIEWER_EMG_PLOT_NAMES),
            {"leadoff": {"during_recording": self._leadoff_during, "current_na": self._leadoff_na}},
            line_hz=self._line_hz,
            mains_mode=self._mains_mode,
        )
        m.update({"kind": self.kind, "mode": "recording", "duration_s": 0.0})
        if self._imu_enabled is not None:
            m["imu_enabled"] = dict(self._imu_enabled)
        return m

    def make_hub(self) -> Hub:
        # Full-rate rows: the hub decimates to its default display rate itself.
        kw: dict[str, Any] = {} if self._bucket <= 1 else {
            "display_fs": self._fs / self._bucket}
        return Hub(
            emg_fs=self._fs,
            emg_lsb_uv=self._lsb,
            n_channels=VIEWER_EMG_PLOT_COUNT,
            channel_names=list(VIEWER_EMG_PLOT_NAMES),
            leadoff_during_recording=self._leadoff_during,
            leadoff_current_na=self._leadoff_na,
            leadoff_ac_tone=self._ac_tone,
            comparators_enabled=self._cmp_enabled,
            line_hz=self._line_hz,
            mains_mode=self._mains_mode,
            **kw,
        )

    def start(self, hub: Hub) -> None:
        self._cap.set_viewer_tap_enabled(True, bucket_samples=self._bucket)
        self._spawn(hub, self._run)

    def poll_once(self, hub: Hub) -> None:
        """Read one native tap frame into the hub (also used directly in tests)."""
        frame = self._cap.get_viewer_frame()
        n = int(np.asarray(frame.emg_t).shape[0])
        emg_seen = int(getattr(frame, "emg_seen", 0) or 0)
        raw_n: int | None = None
        if self._last_emg_seen is not None and emg_seen > self._last_emg_seen:
            raw_n = emg_seen - self._last_emg_seen
        self._last_emg_seen = emg_seen
        if n > 0 and self._bucket <= 1:
            # One sample per row: min == max == the sample (int counts).
            uv = _slice_emg_for_viewer(_counts_to_uv(frame.emg_min, self._gain))
            hub.append_emg_raw(np.asarray(frame.emg_t, dtype=np.uint64), uv)
        elif n > 0:
            lo = _slice_emg_for_viewer(_counts_to_uv(frame.emg_min, self._gain))
            hi = _slice_emg_for_viewer(_counts_to_uv(frame.emg_max, self._gain))
            mean = None
            mean_raw = getattr(frame, "emg_mean", None)
            if mean_raw is not None and np.asarray(mean_raw).shape[0] == n:
                mean = _slice_emg_for_viewer(_counts_to_uv(mean_raw, self._gain))
            hub.append_emg_rows(frame.emg_t, lo, hi, raw_count=raw_n, mean_uv=mean)
        if frame.status is not None:
            hub.set_leadoff(frame.status)
        self._push_tap_quat(hub, frame, "mta")
        self._push_tap_quat(hub, frame, "cra")
        self._push_tap_motion_batches(hub, frame)
        trig = int(getattr(frame, "trig_seen", 0) or 0)
        if trig > 0:
            t_ref = int(getattr(frame, "mta_t", 0) or getattr(frame, "cra_t", 0) or 0)
            hub.note_trigger_events(t_ref, trig)
        if frame.stats is not None:
            hub.set_stats(dict(frame.stats))

    def _push_tap_quat(self, hub: Hub, frame: Any, slot: str) -> None:
        times = getattr(frame, f"{slot}_quat_times", None)
        vals = getattr(frame, f"{slot}_quat_vals", None)
        cal = getattr(frame, f"{slot}_cal", None) or {}
        setter = hub.set_imu_mta if slot == "mta" else hub.set_imu_cra
        if times is not None and vals is not None and len(times) > 0:
            rid = getattr(frame, f"{slot}_quat_report_id", None)
            hub.push_imu_quat_series(slot, np.asarray(times), np.asarray(vals), report_id=rid)
            hub.update_imu_cal(slot, cal)
            return
        quat = getattr(frame, f"{slot}_quat", None)
        if quat is not None:
            t_us = int(getattr(frame, f"{slot}_t", 0))
            setter(
                quat, t_us, getattr(frame, f"{slot}_quat_report_id", None),
                cal.get("quat"),
            )
            hub.update_imu_cal(slot, cal)

    _TAP_MOTION_BATCH = {
        "accel": "accel", "gyro": "gyro", "mag": "mag", "linear_accel": "lin",
    }

    def _push_tap_motion_batches(self, hub: Hub, frame: Any) -> None:
        for slot in ("mta", "cra"):
            cal = getattr(frame, f"{slot}_cal", None) or {}
            for stream, key in self._TAP_MOTION_BATCH.items():
                t_arr = getattr(frame, f"{slot}_{key}_t", None)
                v_arr = getattr(frame, f"{slot}_{key}_v", None)
                if t_arr is not None and v_arr is not None and len(t_arr) > 0:
                    acc = cal.get(stream)
                    hub.push_imu_motion_series(
                        slot, stream, np.asarray(t_arr), np.asarray(v_arr),
                        accuracies=np.full(len(t_arr), acc) if acc is not None else None,
                    )
                    continue
                vec = getattr(frame, f"{slot}_{key}", None)
                if vec is not None:
                    t_us = int(getattr(frame, f"{slot}_t", 0))
                    hub.push_imu_motion(slot, stream, t_us, vec, accuracy=cal.get(stream))

    def _run(self) -> None:
        hub = self._hub
        assert hub is not None
        while not self._stop.is_set():
            self.poll_once(hub)
            time.sleep(self._poll_dt)

    def stop(self) -> None:
        super().stop()
        try:
            self._cap.set_viewer_tap_enabled(False)
        except Exception:
            pass


REPLAY_MIN_SPEED = 0.01
REPLAY_MAX_SPEED = 64.0


def validate_replay_speed(speed: Any) -> float:
    """Finite, positive playback speed clamped to [0.01, 64]; ValueError otherwise.

    An ``inf``/``NaN``/negative speed from /api/speed used to reach the replay
    thread's cursor arithmetic and kill it (OverflowError), freezing replay.
    """
    v = float(speed)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"speed must be a finite number > 0, got {speed!r}")
    return min(max(v, REPLAY_MIN_SPEED), REPLAY_MAX_SPEED)


class ReplaySource(_ThreadedSource):
    """Replay a recorded session at wall-clock * speed with pause/seek.

    All read indices belong to the replay thread. ``seek()`` only records the
    target cursor and bumps a generation counter under the lock; the thread
    adopts it at the top of its next iteration (recomputing every index, and
    resetting the hub's filters and pending data). Seek used to write the
    indices from the HTTP thread while the replay thread was mid-window, which
    then overwrote them with the pre-seek values.
    """

    kind = "replay"

    def __init__(self, path: Path | str, *, speed: float = 1.0, start_s: float = 0.0) -> None:
        super().__init__()
        from jtsmm.io.session_reader import Session

        self.path = Path(path)
        self._session = Session.open(self.path)
        self._speed = validate_replay_speed(speed)
        self._paused = False
        self._ctl = threading.Lock()
        emg = self._session.emg
        self._fs = emg.fs_hz if emg else 2000.0
        self._gain = emg.gain if emg else 6
        self._lsb = emg.lsb_uv if emg else lsb_uv(self._gain)
        self._names = emg.channel_names if emg else list(FULL_CHANNEL_NAMES)
        self._line_hz = _session_line_hz(self._session.meta)
        self._mains_mode = _session_mains_mode(self._session.meta)
        self._t0 = self._session.t0_us
        self._duration = self._session.duration_s
        self._cursor_us = self._t0 + int(max(0.0, float(start_s)) * 1e6)
        self._seek_gen = 0            # bumped by seek(); guarded by _ctl
        self._emg_idx = 0
        self._mta_idx = 0
        self._cra_idx = 0
        self._motion_idx: dict[str, dict[str, int]] = {"mta": {}, "cra": {}}
        self._set_indices(self._cursor_us)

    def meta(self) -> dict[str, Any]:
        m = _emg_meta(
            self._fs, self._gain, VIEWER_EMG_PLOT_COUNT, list(VIEWER_EMG_PLOT_NAMES),
            line_hz=self._line_hz,
            mains_mode=self._mains_mode,
        )
        m.update({
            "kind": self.kind,
            "mode": "replay",
            "duration_s": float(self._duration),
            "path": str(self.path),
            "events_s": [max(0.0, (t - self._t0) / 1e6) for t in self._session.event_times_us()],
        })
        return m

    def make_hub(self) -> Hub:
        from jtsmm.imu.mounting import load_mounting_rotation

        hub = Hub(
            emg_fs=self._fs,
            emg_lsb_uv=self._lsb,
            n_channels=VIEWER_EMG_PLOT_COUNT,
            channel_names=list(VIEWER_EMG_PLOT_NAMES),
            line_hz=self._line_hz,
            mains_mode=self._mains_mode,
        )
        for slot in ("imu_mta", "imu_cra"):
            R = load_mounting_rotation(self.path, slot)
            if R is not None:
                hub.set_mounting_rotation(slot, R)
        return hub

    def start(self, hub: Hub) -> None:
        self._spawn(hub, self._run)

    def set_speed(self, speed: float) -> None:
        v = validate_replay_speed(speed)
        with self._ctl:
            self._speed = v

    def set_paused(self, paused: bool) -> None:
        with self._ctl:
            self._paused = bool(paused)

    def seek(self, t_s: float) -> None:
        t = float(t_s)
        if not math.isfinite(t):
            raise ValueError(f"seek time must be finite, got {t_s!r}")
        t = min(max(0.0, t), float(self._duration) if self._duration > 0 else t)
        with self._ctl:
            self._cursor_us = self._t0 + int(t * 1e6)
            self._seek_gen += 1

    def _set_indices(self, cursor: int) -> None:
        """Every read index at ``cursor`` (replay thread / constructor only)."""
        emg = self._session.emg
        mta = self._session.imu_mta
        cra = self._session.imu_cra
        self._emg_idx = emg.index_at(cursor) if emg else 0
        self._mta_idx = mta.index_at(cursor) if mta else 0
        self._cra_idx = cra.index_at(cursor) if cra else 0
        for slot, stream in (("mta", mta), ("cra", cra)):
            self._motion_idx[slot] = {}
            if stream is not None:
                for name, vs in stream.motion_streams().items():
                    self._motion_idx[slot][name] = vs.index_at(cursor)

    def position(self) -> dict[str, Any]:
        with self._ctl:
            return {
                "t_s": max(0.0, (self._cursor_us - self._t0) / 1e6),
                "duration_s": float(self._duration),
                "speed": self._speed,
                "paused": self._paused,
            }

    def _run(self) -> None:
        hub = self._hub
        assert hub is not None
        emg = self._session.emg
        mta = self._session.imu_mta
        cra = self._session.imu_cra
        last_wall = time.monotonic()
        gen = 0                       # generation the constructor's indices match
        while not self._stop.is_set():
            now = time.monotonic()
            dt = now - last_wall
            last_wall = now
            with self._ctl:
                paused = self._paused
                speed = self._speed
                seek_gen = self._seek_gen
                seek_cursor = self._cursor_us
            if seek_gen != gen:
                # Adopt the seek here, in the only thread that owns the indices,
                # and drop filter state / pending rows from the old position.
                gen = seek_gen
                self._set_indices(seek_cursor)
                hub.reset_stream()
            if paused:
                time.sleep(0.03)
                continue
            with self._ctl:
                if self._seek_gen != gen:
                    continue          # a seek landed meanwhile: adopt it first
                self._cursor_us += int(dt * speed * 1e6)
                cursor = self._cursor_us
            if emg is not None:
                new_idx = emg.index_at(cursor)
                if new_idx > self._emg_idx:
                    t, uv = emg.window(self._emg_idx, new_idx)
                    uv = _slice_emg_for_viewer(uv)
                    hub.append_emg_raw(t, uv)
                    self._emg_idx = new_idx
            if mta is not None and self._mta_idx < mta.n:
                ni = mta.index_at(cursor)
                if ni > self._mta_idx:
                    i = min(ni, mta.n) - 1
                    hub.set_imu_mta(
                        tuple(float(x) for x in mta.quat[i]),
                        int(mta.t_us[i]),
                        self._quat_report_id(mta, i),
                    )
                    self._mta_idx = ni
            if cra is not None and self._cra_idx < cra.n:
                ni = cra.index_at(cursor)
                if ni > self._cra_idx:
                    i = min(ni, cra.n) - 1
                    hub.set_imu_cra(
                        tuple(float(x) for x in cra.quat[i]),
                        int(cra.t_us[i]),
                        self._quat_report_id(cra, i),
                    )
                    self._cra_idx = ni
            self._replay_motion(hub, "mta", mta, cursor)
            self._replay_motion(hub, "cra", cra, cursor)
            if emg is not None and self._emg_idx >= emg.n:
                # Reached the end; hold until seeked or stopped.
                time.sleep(0.05)
            else:
                time.sleep(0.02)

    @staticmethod
    def _quat_report_id(stream: Any, i: int) -> int | None:
        rid = getattr(stream, "quat_report_id", None)
        if rid is not None and i < len(rid):
            return int(rid[i])
        return None

    def _replay_motion(self, hub: Hub, slot: str, stream: Any, cursor: int) -> None:
        if stream is None:
            return
        for name, vs in stream.motion_streams().items():
            idx = self._motion_idx[slot].get(name, 0)
            if idx >= vs.n:
                continue
            ni = vs.index_at(cursor)
            if ni <= idx:
                continue
            for i in range(idx, min(ni, vs.n)):
                hub.push_imu_motion(
                    slot, name, int(vs.t_us[i]), tuple(float(x) for x in vs.data[i])
                )
            self._motion_idx[slot][name] = ni

    def stop(self) -> None:
        super().stop()
        self._session.close()


class MockSource(_ThreadedSource):
    """Synthetic EMG + rotating IMU for tests/demos (no hardware)."""

    kind = "mock"

    def __init__(
        self,
        *,
        fs: float = 2000.0,
        gain: int = 6,
        n_channels: int = VIEWER_EMG_PLOT_COUNT,
        n_cameras: int = 2,
    ) -> None:
        super().__init__()
        self._fs = float(fs)
        self._gain = int(gain)
        self._n = int(n_channels)
        self._lsb = lsb_uv(self._gain)
        self._phase = 0.0
        self._n_cameras = int(n_cameras)
        self._cam_feed: Any = None

    def meta(self) -> dict[str, Any]:
        m = _emg_meta(self._fs, self._gain, self._n, list(VIEWER_EMG_PLOT_NAMES)[: self._n])
        cams = self._cam_feed.camera_ids if self._cam_feed is not None else [
            f"cam{i}" for i in range(self._n_cameras)
        ]
        m.update({
            "kind": self.kind, "mode": "mock", "duration_s": 0.0, "camera_ids": list(cams),
        })
        return m

    def make_hub(self) -> Hub:
        return Hub(emg_fs=self._fs, emg_lsb_uv=self._lsb, n_channels=self._n,
                   channel_names=list(VIEWER_EMG_PLOT_NAMES)[: self._n])

    def start(self, hub: Hub) -> None:
        from jtsmm.viewer.cam_preview import CamPreviewFeed

        self._cam_feed = CamPreviewFeed()
        self._cam_feed.start_mock(hub, self._n_cameras)
        self._t_us = 0
        self._spawn(hub, self._run)

    def stop(self) -> None:
        super().stop()
        if self._cam_feed is not None:
            try:
                self._cam_feed.stop()
            finally:
                self._cam_feed = None

    # SH-2 game rotation vector: the synthetic fusion source the mock reports.
    _MOCK_QUAT_REPORT_ID = 0x08
    _MOCK_MOTION_FS = 100.0  # synthetic accel/gyro/mag/lin sample rate

    def _run(self) -> None:
        hub = self._hub
        assert hub is not None
        rng = np.random.default_rng(1)
        chunk_dt = 0.033
        dt_us = int(1e6 / self._fs)
        m_dt_us = int(1e6 / self._MOCK_MOTION_FS)
        while not self._stop.is_set():
            n = max(1, int(self._fs * chunk_dt))
            k = np.arange(n)
            t = self._t_us + k * dt_us
            base = 40.0 * np.sin(2 * np.pi * 80.0 * (t / 1e6))[:, None]
            uv = (base + rng.normal(0, 8, size=(n, self._n))).astype(np.float32)
            hub.append_emg_raw(t.astype(np.uint64), uv)
            self._phase += 0.05
            q = (float(np.cos(self._phase)), float(np.sin(self._phase)), 0.0, 0.0)
            hub.set_imu_mta(q, int(t[-1]), self._MOCK_QUAT_REPORT_ID)
            hub.set_imu_cra((1.0, 0.0, 0.0, 0.0), int(t[-1]), self._MOCK_QUAT_REPORT_ID)
            self._push_mock_motion(hub, rng, m_dt_us, chunk_dt)
            self._t_us = int(t[-1]) + dt_us
            time.sleep(chunk_dt)

    def _push_mock_motion(self, hub: Hub, rng: Any, m_dt_us: int, chunk_dt: float) -> None:
        """Synthesize accel/gyro/mag/linear_accel so the UI has live strips."""
        nm = max(1, int(self._MOCK_MOTION_FS * chunk_dt))
        for j in range(nm):
            tm = self._t_us + j * m_dt_us
            w = 2 * np.pi * 1.5 * (tm / 1e6)  # ~1.5 Hz jaw-like oscillation
            sw, cw = float(np.sin(w)), float(np.cos(w))
            # MTA: gravity + oscillation on accel, matching gyro, small lin accel.
            hub.push_imu_motion("mta", "accel", tm,
                                (0.5 * sw + rng.normal(0, 0.05), rng.normal(0, 0.05), 9.81 + 0.5 * cw))
            hub.push_imu_motion("mta", "gyro", tm, (0.4 * cw, 0.1 * sw, rng.normal(0, 0.02)))
            hub.push_imu_motion("mta", "mag", tm, (25.0 + sw, -8.0 + 0.5 * cw, 40.0))
            hub.push_imu_motion("mta", "linear_accel", tm, (0.5 * sw, 0.0, 0.5 * cw))
            # CRA: near-still head reference.
            hub.push_imu_motion("cra", "accel", tm, (rng.normal(0, 0.03), rng.normal(0, 0.03), 9.81))
            hub.push_imu_motion("cra", "gyro", tm, (rng.normal(0, 0.01), rng.normal(0, 0.01), 0.0))
            hub.push_imu_motion("cra", "mag", tm, (25.0, -8.0, 40.0))
            hub.push_imu_motion("cra", "linear_accel", tm,
                                (rng.normal(0, 0.02), rng.normal(0, 0.02), rng.normal(0, 0.02)))
