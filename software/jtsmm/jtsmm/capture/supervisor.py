# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Process supervisor for multi-process capture (s02 §1.1)."""

from __future__ import annotations

import json
import signal
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jtsmm import native
from jtsmm.config.catalog import DEFAULT_EMG_SAMPLE_RATE_SPS
from jtsmm.config import (
    open_hat_serial,
    split_host_mcu,
    start_mcu_sensor_streams,  # noqa: F401 - tests monkeypatch this name
    start_mcu_streams,
    start_mcu_trigger,
    stop_mcu_streams,
    stop_mcu_streams_robust,
)
from jtsmm.config.mcu_apply import mcu_stream_enables
from jtsmm.config.registry import (
    capture_camera_delta_tracking,
    capture_camera_pps_anchor,
    capture_native_prefer,
    capture_pps_samples,
    capture_settle_ms,
    capture_stream_restart,
    capture_trigger_after_cameras,
    emg_check_line_hz,
    emg_filter_mains_mode,
)
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.io.session_layout import (
    ct_calibration_recording_dir,
    experiment_name_from_config,
    session_output_from_config,
)
from jtsmm.io.session_reproduce import RecordReproduceSpec, write_session_reproduce
from jtsmm.protocol.types import RECORD_NAMES, RecordType
from jtsmm.sync.clock_align import AlignedRecord, ClockAligner

from .capture_cam import cameras_from_config
from .native_backend import NativeCapture, NativeWriter
from .optical_bridge import OpticalTrackBridge, nth
from .phase_walker import PhaseWalker, optical_frame_intervals

_MCU_PUMP_BATCH = 128
_MCU_LOG_PRINT_MAX = 200  # firmware log lines echoed per take; all go to /events
_CAM_HOLD_MAX = 120  # ~2 s at 60 Hz
_VIEWER_PERIOD_S = 0.1
_VIEWER_ONLY_RGB_PERIOD_S = 0.1


def _require_native() -> None:
    if not native.use_native() or native.NativeHdf5Writer is None:
        raise RuntimeError(
            "Native capture extension required for real (non-mock) recording. "
            "Build it with software/jtsmm/native/build.sh."
        )


def _arm_mcu_streams(
    port: str,
    native_cap: NativeCapture | None,
    resolved: dict[str, Any],
    *,
    emg: bool = True,
    trigger: bool = True,
    imu_mta: bool = True,
    imu_cra: bool = True,
) -> None:
    """Start MCU streams, drain USB, optionally restart for a clean session baseline.

    ``trigger=False`` leaves the camera trigger disarmed; ``run`` arms it with
    ``_arm_mcu_trigger`` once libcamera is delivering frames. Disabled IMU slots
    are not started (``imu_stream on`` would force-enable them in firmware).
    """
    settle_ms = capture_settle_ms(resolved)
    with open_hat_serial(port) as ser:
        start_mcu_streams(ser, emg=emg, trigger=trigger, imu_mta=imu_mta, imu_cra=imu_cra)
    if native_cap is None:
        if settle_ms > 0:
            time.sleep(settle_ms / 1000.0)
        return
    native_cap.settle(quiet_ms=settle_ms)
    if not capture_stream_restart(resolved):
        return
    with open_hat_serial(port) as ser:
        stop_mcu_streams(ser)
    time.sleep(0.25)
    with open_hat_serial(port) as ser:
        start_mcu_streams(ser, emg=emg, trigger=trigger, imu_mta=imu_mta, imu_cra=imu_cra)
    native_cap.settle(quiet_ms=settle_ms)


def _arm_mcu_trigger(port: str) -> None:
    """Arm the camera trigger after the cameras are running."""
    with open_hat_serial(port) as ser:
        start_mcu_trigger(ser)


_PPS_MAX_RTT_NS = 30_000_000
_PPS_REPLY_TIMEOUT_S = 0.3
_FINAL_STATS_TIMEOUT_S = 1.0
_START_STATS_TIMEOUT_S = 1.0
_FINAL_DRAIN_MAX_S = 2.0
_CAMERA_STOP_TIMEOUT_S = 5.0
# After stop() has been entered, a few pulses let the in-flight IMX296 frame
# finish. The trigger then goes off without waiting for picamera2's teardown,
# so the recorded tail is this grace and not both sequential stop() calls
# (~0.67 s on the bench).
_CAMERA_STOP_GRACE_S = 0.12


def _bounded_call(fn: Any, timeout_s: float, what: str) -> Any:
    """Run ``fn`` in a daemon thread; give up (and say so) after ``timeout_s``.

    For teardown calls into libraries that can block indefinitely (picamera2's
    stop()). The thread keeps running if it hangs, but the teardown continues
    and the session file is closed. Exceptions from ``fn`` are re-raised.
    """
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["error"] = exc

    th = threading.Thread(target=_run, name=f"teardown-{what}", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(f"{what} did not return within {timeout_s:g} s (left running)")
    if "error" in box:
        raise box["error"]
    return box.get("value")
# Dead-link watchdog: EMG arrives at >= 250 records/s whenever the streams are
# armed, so a full second without one means the USB link or the HAT is gone.
_WATCHDOG_SILENCE_S = 1.0


# Free-disk preflight: fail below max(floor, margin x estimate), warn below the
# larger pair. A plan without a timeout is estimated as one hour.
_DISK_FAIL_FLOOR_GB = 0.5
_DISK_WARN_FLOOR_GB = 2.0
_DISK_FAIL_MARGIN = 1.5
_DISK_WARN_MARGIN = 3.0
_DISK_OPEN_ENDED_S = 3600.0


def estimate_session_bytes_per_s(
    mcu_cfg: dict[str, Any], n_cameras: int, *, preview_jpegs: bool, optical: bool
) -> float:
    """Rough HDF5 (+ preview JPEG) growth rate for the disk-space preflight."""
    emg = mcu_cfg.get("emg") or {}
    rate = float(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS) or 0) * 74.0  # 8x(i32 + f32) + t + lead-off
    for slot in ("imu_mta", "imu_cra"):
        reports = (mcu_cfg.get(slot) or {}).get("reports") or {}
        for rep in reports.values():
            if isinstance(rep, dict) and rep.get("enabled", True):
                rate += float(rep.get("rate_hz", 0) or 0) * 28.0
    trig_hz = float((mcu_cfg.get("trigger") or {}).get("rate_hz", 30) or 30)
    rate += trig_hz * 16.0 + n_cameras * trig_hz * 40.0
    if optical:
        rate += trig_hz * 3 * 70.0
    if preview_jpegs:
        rate += n_cameras * 10.0 * 25_000.0  # ~10 Hz x ~25 kB per camera
    return rate


def disk_space_preflight(path: Path, est_bytes: float) -> dict[str, Any]:
    """Raise when the session disk is too full for the take; warn when tight."""
    import shutil

    free = float(shutil.disk_usage(path).free)
    gb = 1e9
    fail_below = max(_DISK_FAIL_FLOOR_GB * gb, _DISK_FAIL_MARGIN * est_bytes)
    warn_below = max(_DISK_WARN_FLOOR_GB * gb, _DISK_WARN_MARGIN * est_bytes)
    info = {"free_gb": round(free / gb, 2), "estimate_gb": round(est_bytes / gb, 3)}
    if free < fail_below:
        raise RuntimeError(
            f"not enough free disk space in {path}: {free / gb:.2f} GB free, the take needs "
            f"~{est_bytes / gb:.2f} GB (refusing below {fail_below / gb:.2f} GB) -- free space "
            "or shorten --duration"
        )
    if free < warn_below:
        print(
            f"WARNING: low disk space: {free / gb:.2f} GB free, this take needs "
            f"~{est_bytes / gb:.2f} GB",
            file=sys.stderr,
        )
        info["warning"] = True
    return info


def _span_s(timing: dict[str, int | None], a: str, b: str) -> float | None:
    ta, tb = timing.get(a), timing.get(b)
    if ta is None or tb is None:
        return None
    return round((tb - ta) / 1e9, 3)


def _session_timing(timing: dict[str, int | None]) -> dict[str, Any]:
    """Durations on the host clock; /meta/session_duration_s is wall time incl. setup."""
    t_now = time.monotonic_ns()
    return {
        "wall_s": round((t_now - timing["open"]) / 1e9, 3) if timing.get("open") else None,
        "setup_s": _span_s(timing, "open", "loop_start"),
        "capture_s": _span_s(timing, "loop_start", "loop_end"),
        # first to last MCU record / camera frame arriving in the main loop
        "data_span_s": _span_s(timing, "first_data", "last_data"),
        "note": "wall_s = file open to teardown (what /meta/session_duration_s measures); "
        "data_span_s = first to last data arrival in the capture loop",
    }


def _startup_timing(timing: dict[str, int | None]) -> dict[str, Any]:
    return {
        "camera_start_s": _span_s(timing, "cams_start", "cams_started"),
        # EMG/IMU recorded before the first camera trigger (kept, not a hole)
        "sensor_preroll_s": _span_s(timing, "sensors_armed", "trigger_armed"),
    }


def _timing_meta(st: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "session_duration_kind": "wall clock, file open to close, includes setup "
        "(see capture_duration_s)"
    }
    span = st.get("data_span_s")
    if span is None:
        span = st.get("capture_s")
    if span is not None:
        out["capture_duration_s"] = float(span)
    return out


def _set_meta_attrs(writer: Any, attrs: dict[str, Any]) -> None:
    """Late /meta attrs; only the Python writer exposes its open file."""
    f = getattr(writer, "_file", None)
    if f is None:
        return
    for k, v in attrs.items():
        f["meta"].attrs[k] = v


def _watched_record_count(counts: Counter[int], *, emg: bool) -> int:
    if emg:
        return int(counts.get(int(RecordType.EMG_BLOCK), 0))
    return int(sum(counts.values()))


def _device_lost(native_cap: Any) -> bool:
    """``NativeCapture.link_health()["device_lost"]`` where the backend has it."""
    health = getattr(native_cap, "link_health", None)
    if health is None:
        return False
    try:
        info = health() if callable(health) else health
    except Exception:  # noqa: BLE001 - a health probe must never stop a take by itself
        return False
    return bool(isinstance(info, dict) and info.get("device_lost"))


def _stats_dict(rec: Any) -> dict[str, Any]:
    """REC_STATS counters plus the record's own t_master_us (lets the checker age them)."""
    out = dict(getattr(rec, "decoded", None) or {})
    t = getattr(rec, "t_master_us", None)
    if t is not None:
        out["t_master_us"] = int(t)
    return out


def _send_stats_request(ser: Any) -> None:
    """`stats` makes the firmware emit one REC_STATS immediately (K2)."""
    ser.write(b"stats\r\n")
    ser.flush()


def _stats_newer(stats: dict[str, Any] | None, seq0: Any) -> bool:
    """Is ``stats`` newer than the REC_STATS with ``stats_seq == seq0``?

    Firmware before K2 has no ``stats_seq``: then any REC_STATS that arrived
    after the request counts (the caller checks that one arrived).
    """
    if not stats:
        return False
    seq = stats.get("stats_seq")
    if seq is None or seq0 is None:
        return True
    return 0 < ((int(seq) - int(seq0)) & 0xFFFFFFFF) < 0x80000000


def _pps_handshake(
    port: str,
    native_cap: Any,
    aligner: ClockAligner,
    n_samples: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Fit host CLOCK_MONOTONIC -> t_master_us from `pps <id>` round trips."""
    result: dict[str, Any] = {
        "pass": label,
        "attempted": int(n_samples),
        "accepted": 0,
        "timeouts": 0,
        "rejected_rtt": 0,
        "rtt_ms_median": None,
    }
    if n_samples <= 0 or native_cap is None:
        return result
    rtts_ms: list[float] = []
    try:
        with open_hat_serial(port) as ser:
            base = int(time.monotonic_ns() // 1_000_000) & 0x7FFFFFFF
            for i in range(int(n_samples)):
                pid = base * 100 + i
                native_cap.discard_queued()
                t0 = time.monotonic_ns()
                ser.write(f"pps {pid}\r\n".encode("ascii"))
                ser.flush()
                reply = None
                t1 = 0
                deadline = time.monotonic() + _PPS_REPLY_TIMEOUT_S
                while reply is None and time.monotonic() < deadline:
                    for rec in native_cap.drain_records():
                        if int(rec.record_type) != int(RecordType.PPS_REPLY):
                            continue
                        dec = getattr(rec, "decoded", None) or {}
                        if int(dec.get("pps_id", -1)) == pid:
                            reply = rec
                            t1 = time.monotonic_ns()
                            break
                    if reply is None:
                        time.sleep(0.001)
                if reply is None:
                    result["timeouts"] += 1
                    continue
                rtt_ns = t1 - t0
                if rtt_ns > _PPS_MAX_RTT_NS:
                    result["rejected_rtt"] += 1
                    continue
                aligner.ingest_pps((t0 + t1) // 2, int(reply.t_master_us))
                rtts_ms.append(rtt_ns / 1e6)
                result["accepted"] += 1
                time.sleep(0.02)
    except Exception as exc:  # noqa: BLE001 - a sync aid must never break a take
        result["error"] = str(exc)
    if rtts_ms:
        result["rtt_ms_median"] = float(np.median(rtts_ms))
    return result


def _pi_clock_note() -> dict[str, Any]:
    """Which host clock the PPS fit is in, and its offset to CLOCK_BOOTTIME."""
    note: dict[str, Any] = {"domain": "CLOCK_MONOTONIC_ns"}
    try:
        boot = time.clock_gettime_ns(time.CLOCK_BOOTTIME)  # type: ignore[attr-defined]
        mono = time.monotonic_ns()
        note["boottime_minus_monotonic_ns"] = int(boot - mono)
    except Exception:  # noqa: BLE001
        note["boottime_minus_monotonic_ns"] = None
    return note


def remap_phase_stamps(walker: Any, aligner: ClockAligner, *, enabled: bool = True) -> dict[str, Any]:
    """Re-stamp phase edges and intervals from their host times with the final fit.

    SPACE edges are stamped live with whatever host->master fit exists at that
    moment -- after the 8-sample start PPS pass that is a free-slope fit over
    ~170 ms, never updated during the take. On take1 its slope came out at
    1.000757 (757 ppm, true drift < 1 ppm), so /events, /phases and the JTS
    frame intervals were +86 ms ... +367 ms late. Every edge keeps its
    ``t_host_ns``, so after the end PPS pass the final fit (start + end
    samples spanning the whole take) re-maps them exactly.

    Mutates ``walker.events`` and ``walker.recorded`` in place. Only edges the
    final fit can map (quality ``pi_fit``) are changed; a fallback mapping from
    the last MCU arrival would be worse than the live stamp. Returns counts, the
    largest correction and ``live_intervals`` (the original (start, end) per
    recorded phase) for the sync report.
    """
    events = list(getattr(walker, "events", []) or [])
    out: dict[str, Any] = {
        "n_events": len(events),
        "n_remapped": 0,
        "max_abs_correction_us": 0,
        "fit_n_samples": int(getattr(aligner._pi_fit, "n_samples", 0)),  # noqa: SLF001
        "live_intervals": [],
    }
    if not enabled or not events:
        return out
    for ev in events:
        if ev.map_quality == "mock":
            continue
        t_new, quality = aligner.map_pi_ns(int(ev.t_host_ns))
        if t_new is None or quality != "pi_fit":
            continue
        corr = int(t_new) - int(ev.t_master_us)
        ev.t_master_us = int(t_new)
        ev.map_quality = quality
        out["n_remapped"] += 1
        out["max_abs_correction_us"] = max(out["max_abs_correction_us"], abs(corr))
    # Each recorded phase is one (start, end) edge pair, in order.
    recorded = list(getattr(walker, "recorded", []) or [])
    k = 0
    start_ev = None
    for ev in events:
        if ev.edge == "start":
            start_ev = ev
        elif ev.edge == "end" and start_ev is not None:
            if k < len(recorded):
                ph = recorded[k]
                out["live_intervals"].append((int(ph.t_start_us), int(ph.t_end_us)))
                ph.t_start_us = int(start_ev.t_master_us)
                # Same non-empty guarantee as PhaseWalker._end.
                ph.t_end_us = max(int(ev.t_master_us), ph.t_start_us + 1)
                ev.t_master_us = ph.t_end_us
                ph.map_quality_start = start_ev.map_quality
                ph.map_quality_end = ev.map_quality
            k += 1
            start_ev = None
    return out


def remap_note_stamps(
    notes: list[dict[str, Any]], aligner: ClockAligner, writer: Any, *, enabled: bool = True
) -> dict[str, Any]:
    """Re-stamp viewer notes already written to ``/events`` with the final fit.

    Notes are written as they arrive (so a killed recorder keeps them), mapped
    with the live fit; like phase edges (`remap_phase_stamps`) they are
    re-mapped from their stored ``t_host_ns`` after the end PPS pass, in place
    via ``writer.set_event_time``. Only ``pi_fit`` mappings replace a stamp.
    Mutates the note dicts (``row``, ``t_host_ns``, ``t_master_us``, ``map_quality``).
    """
    out: dict[str, Any] = {"n_notes": len(notes), "n_remapped": 0, "max_abs_correction_us": 0}
    if not enabled:
        return out
    for note in notes:
        if note.get("map_quality") == "mock":
            continue
        t_new, quality = aligner.map_pi_ns(int(note["t_host_ns"]))
        if t_new is None or quality != "pi_fit":
            continue
        writer.set_event_time(int(note["row"]), t_master_us=int(t_new), map_quality=quality)
        corr = int(t_new) - int(note["t_master_us"])
        note["t_master_us"], note["map_quality"] = int(t_new), quality
        out["n_remapped"] += 1
        out["max_abs_correction_us"] = max(out["max_abs_correction_us"], abs(corr))
    return out


class _Teardown:
    """Run shutdown steps in order; a failing step is logged and never skips the next.

    Before, one raise (a camera that would not stop, a full disk in flush)
    skipped every later step including ``writer.close()`` -- leaving an
    unreadable file and native threads running.
    """

    def __init__(self) -> None:
        self.failed: list[str] = []

    def step(self, name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - incl. a stray KeyboardInterrupt
            msg = f"{name}: {type(exc).__name__}: {exc}"
            self.failed.append(msg)
            print(f"WARNING: teardown step failed -- {msg}", file=sys.stderr)
            return None


class SessionStopSignal(BaseException):
    """SIGTERM/SIGHUP arrived during setup (before the take started).

    A BaseException so the many ``except Exception`` guards in the setup code
    (optional viewer, reproduce bundle, ...) cannot swallow it.
    """


class _StopRequest:
    """Turn SIGTERM / SIGHUP (whole run) and SIGINT (main loop) into a graceful stop.

    Without handlers an SSH drop (SIGHUP) or `kill` (SIGTERM) killed the
    recorder mid-write and left an unreadable HDF5. Phases:
    - ``setup``: SIGTERM/SIGHUP raise :class:`SessionStopSignal` (no take yet;
      the teardown still closes everything). Ctrl-C keeps its default
      KeyboardInterrupt here.
    - ``loop`` / ``finish``: the first signal only sets ``reason``; the loop ends
      through the normal finish (tail drain, end PPS, sync report, close). A
      second Ctrl-C in the loop is a hard abort, which still closes the file.
    - ``teardown``: further signals are ignored.
    Handlers are only installed on the main thread (``signal`` requires it).
    """

    def __init__(self) -> None:
        self.reason: str | None = None
        self.signame: str | None = None
        self.phase = "setup"
        self._prev: dict[int, Any] = {}

    def _set(self, signum: int) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            prev = signal.signal(signum, self._handler)
        except (ValueError, OSError):
            return
        self._prev.setdefault(signum, prev)

    def install(self) -> None:
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None:
                self._set(sig)

    def enter_loop(self) -> None:
        self.phase = "loop"
        self._set(signal.SIGINT)

    def leave_loop(self) -> None:
        # The SIGINT handler stays through the finish (tail drain, end PPS): a
        # Ctrl-C there must not abandon the file either.
        self.phase = "finish"

    def begin_teardown(self) -> None:
        self.phase = "teardown"

    def restore(self) -> None:
        for signum, prev in self._prev.items():
            try:
                signal.signal(signum, prev)
            except (ValueError, OSError, TypeError):
                pass
        self._prev.clear()

    def _handler(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        if self.phase == "teardown":
            print(f"\n{name}: already closing the session file, please wait", file=sys.stderr)
            return
        if self.reason is not None:
            if signum == signal.SIGINT and self.phase == "loop":
                raise KeyboardInterrupt  # second Ctrl-C: hard abort (file still closed)
            return
        interactive = signum == signal.SIGINT
        self.reason = "operator_stop" if interactive else "signal"
        if not interactive:
            self.signame = name
        print(
            f"\n{name}: stopping the take and closing the session file"
            + (" (Ctrl-C again to abort hard)" if interactive else ""),
            file=sys.stderr,
        )
        if self.phase == "setup" and not interactive:
            raise SessionStopSignal(name)


class _NullSink:
    """Stand-in writer for the optical worker when the HDF5 file never opened."""

    def append_optical(self, **_kw: Any) -> None:
        return None


def _ignore_sigint() -> Any:
    """Ignore Ctrl-C during teardown (a second press must not abandon the file)."""
    if threading.current_thread() is not threading.main_thread():
        return None
    try:
        return signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        return None


def _restore_sigint(previous: Any) -> None:
    if previous is None:
        return
    try:
        signal.signal(signal.SIGINT, previous)
    except (ValueError, OSError, TypeError):
        pass


class SessionSupervisor:
    """MCU capture in child process; align + HDF5 writer in parent."""

    def __init__(
        self,
        resolved: dict[str, Any],
        port: str,
        *,
        viewer: bool = False,
        viewer_host: str = "0.0.0.0",
        viewer_port: int = 8770,
        extra_meta: dict[str, Any] | None = None,
        reproduce: RecordReproduceSpec | None = None,
        optical_track: bool = False,
        optical_config: Path | None = None,
        optical_track_hz: float | None = None,
        experiment_plan: Any | None = None,
        write_jts_config: bool = True,
        jts_config_path: Path | None = None,
        ct_calibration: bool = False,
        ct_calibration_n_poses: int = 20,
        ct_tip_calibration: Path | None = None,
        persist_ct_calibration: bool = False,
        ct_calibration_recordings_dir: Path | str | None = None,
        ct_calibration_only: bool = False,
        enable_emg: bool = True,
    ) -> None:
        self.resolved = resolved
        self.port = port
        self.viewer = viewer
        self.viewer_host = viewer_host
        self.viewer_port = viewer_port
        self.extra_meta = dict(extra_meta or {})
        self.reproduce = reproduce
        self.optical_track = optical_track
        self.optical_config = optical_config
        self.optical_track_hz = optical_track_hz
        self.experiment_plan = experiment_plan
        self.write_jts_config = write_jts_config
        self.jts_config_path = jts_config_path
        self.ct_calibration = ct_calibration
        self.ct_calibration_n_poses = int(ct_calibration_n_poses)
        self.ct_tip_calibration = ct_tip_calibration
        self.persist_ct_calibration = bool(persist_ct_calibration)
        self.ct_calibration_recordings_dir = ct_calibration_recordings_dir
        self.ct_calibration_only = bool(ct_calibration_only)
        self.enable_emg = bool(enable_emg)

    def run(
        self, duration_s: float | None, *, mock_only: bool = False, streams_running: bool = False
    ) -> Path:
        mcu_cfg, host_bundle = split_host_mcu(self.resolved)
        from jtsmm.config import applied_config_hash

        cfg_hash = applied_config_hash(self.resolved)
        plan_name = (
            str(self.experiment_plan.name) if self.experiment_plan is not None else None
        )
        if not plan_name:
            plan_name = experiment_name_from_config(self.resolved)
        session_dir, h5_path = session_output_from_config(
            self.resolved, experiment_name=plan_name
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        if self.reproduce is not None:
            try:
                self.reproduce.duration_s = float(duration_s) if duration_s is not None else None
                written = write_session_reproduce(session_dir, self.resolved, self.reproduce)
                script = written.get("script")
                if script is not None:
                    print(f"reproduce: {script}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — never abort a recording for this
                print(f"warning: failed to write session reproduce bundle ({exc})", file=sys.stderr)

        firmware_version = None
        try:
            from jtsmm.doctor.mcu_info import query_mcu_info

            firmware_version = query_mcu_info(self.port).firmware_version
        except Exception:
            pass

        session_meta = {"mode": (self.resolved.get("session") or {}).get("mode", "acquisition")}
        session_meta["emg_enabled"] = "true" if self.enable_emg else "false"
        session_meta.update({str(k): str(v) for k, v in self.extra_meta.items()})

        plan_needs_optical = (
            self.experiment_plan is None or bool(self.experiment_plan.needs_optical)
        )
        mvc_only = (
            self.experiment_plan is not None
            and not plan_needs_optical
            and not self.ct_calibration
        )
        need_optical = bool(
            (self.optical_track or self.ct_calibration) and not mvc_only
        )
        want_cameras = bool(self.viewer or not mvc_only)

        # Everything that fails on bad input or a missing TTY/module is checked
        # BEFORE any hardware is touched, so a typo cannot leave cameras, the USB
        # capture threads or MCU streams running behind a half-written file.
        if need_optical and self.optical_config is None:
            raise RuntimeError(
                "optical tracking / ct_calibration requires --optical-config "
                "or session.optical_config"
            )
        tip_path: Path | None = None
        if not self.ct_calibration and self.ct_tip_calibration is not None and need_optical:
            tip_path = Path(self.ct_tip_calibration)
            if not tip_path.is_file():
                raise RuntimeError(
                    f"session.ct_tip_calibration not found: {tip_path} "
                    "(run with session.ct_calibration: true once to create it)"
                )
        keys_ctx = None
        if self.experiment_plan is not None:
            try:
                from jtsmm.io.terminal_keys import TerminalKeys

                keys_ctx = TerminalKeys()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"--experiment-plan requires an interactive TTY for SPACE keys ({exc})"
                ) from exc
        if self.ct_calibration and not mock_only:
            try:
                from jtsmm.io.terminal_keys import TerminalKeys

                TerminalKeys()
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"CT tip calibration requires an interactive TTY for SPACE keys ({exc})"
                ) from exc
            from .ct_pivot import run_ct_pivot_calibration  # noqa: F401 - fail before hardware
        draw_boxes = encode_jpeg = None
        preview_size = (320, 240)
        if self.experiment_plan is not None or self.viewer:
            try:
                from jtsmm.viewer.aruco_overlay import draw_marker_boxes_rgb
                from jtsmm.viewer.cam_preview import PREVIEW_SIZE
                from jtsmm.viewer.jpeg_util import encode_jpeg as _encode_jpeg

                draw_boxes, encode_jpeg, preview_size = (
                    draw_marker_boxes_rgb,
                    _encode_jpeg,
                    PREVIEW_SIZE,
                )
            except Exception as exc:  # noqa: BLE001
                if self.experiment_plan is not None:
                    raise RuntimeError(
                        f"experiment-plan preview JPEGs need the viewer modules ({exc})"
                    ) from exc
                print(f"warning: viewer camera previews disabled ({exc})", file=sys.stderr)
        n_cams_cfg = (
            len((host_bundle.get("host", {}).get("cameras", {}) or {}).get("list") or [])
            if want_cameras
            else 0
        )
        est_rate = estimate_session_bytes_per_s(
            mcu_cfg,
            n_cams_cfg,
            preview_jpegs=self.experiment_plan is not None and want_cameras,
            optical=need_optical,
        )
        est_s = float(duration_s) if duration_s is not None else _DISK_OPEN_ENDED_S
        disk_info = disk_space_preflight(session_dir, est_rate * est_s)

        # --- state shared by the setup, the main loop and the teardown ----------
        cameras: list[Any] = []
        optical: OpticalTrackBridge | None = None
        writer: Any = None
        writer_opened = False
        native_cap: NativeCapture | None = None
        use_native_writer = False
        viewer_server: Any = None
        viewer_hub: Any = None
        # Viewer notes written to /events: {row, t_host_ns, t_master_us, map_quality}.
        note_rows: list[dict[str, Any]] = []
        notes_failed = 0
        keys_entered = False
        preview_dump: Any = None
        walker: PhaseWalker | None = None
        loop_started = False
        # Where a failure happened ("setup" / "take" / "finish"): the CLI tells
        # the operator (and integrity.json) which it was.
        run_phase = "setup"
        failure: BaseException | None = None
        end_reason = "setup"
        stop = _StopRequest()
        watchdog: dict[str, Any] | None = None
        timing: dict[str, int | None] = dict.fromkeys(
            ("open", "sensors_armed", "cams_start", "cams_started", "trigger_armed",
             "loop_start", "loop_end", "first_data", "last_data")
        )
        aligner = ClockAligner(
            pps_anchor=capture_camera_pps_anchor(self.resolved),
            delta_tracking=capture_camera_delta_tracking(self.resolved),
        )
        # Nominal trigger rate + pulse width: first-frame latency correction and
        # the fallback period before enough triggers have been measured.
        aligner.configure_from_mcu(mcu_cfg)
        trigger_after_cams = capture_trigger_after_cameras(self.resolved)
        # mcu.<block>.enabled: the firmware parses these but does not act on
        # them, so a disabled stream is simply never started from here.
        enables = mcu_stream_enables(self.resolved)
        emg_on = self.enable_emg and enables["emg"]
        trigger_on = enables["trigger"]
        if not enables["emg"] and self.enable_emg:
            print("note: mcu.emg.enabled is false -- EMG is not started", file=sys.stderr)
        for slot in ("imu_mta", "imu_cra"):
            if not enables[slot]:
                print(f"note: mcu.{slot}.enabled is false -- that IMU slot is not started",
                      file=sys.stderr)
        if not trigger_on and want_cameras and not mock_only:
            print(
                "WARNING: mcu.trigger.enabled is false but cameras are configured -- the "
                "cameras wait for an MCU trigger that never comes (no camera frames)",
                file=sys.stderr,
            )
        pps_n = capture_pps_samples(self.resolved)
        pps_start: dict[str, Any] | None = None
        pps_end: dict[str, Any] | None = None
        if streams_running:
            capture_order = "external"
        elif trigger_after_cams:
            capture_order = "sensors,cameras,trigger"
        else:
            capture_order = "sensors+trigger,cameras"
        # Optical: lores RGB every drained frame when recording poses. Viewer-only: ~10 Hz.
        # MVC phases pause ArUco even if session.optical_track is on.
        write_optical = bool(need_optical)
        viewer_rgb_due: dict[str, float] = {}
        # Per camera. One shared slot let cam0 consume every 10 Hz tick, so
        # cam1's preview JPEG was skipped for the whole take.
        viewer_jpeg_due: dict[str, float] = {}
        dump_jpeg_due: dict[str, float] = {}
        camera_ingest = {"on": True}
        record_counts: Counter[int] = Counter()
        last_stats: dict[str, Any] | None = None
        stats_seen = 0  # REC_STATS records observed (freshness check at the stop)
        stats_start: dict[str, Any] | None = None  # baseline for this take's counter deltas
        cam_hold: deque[tuple[Any, Any]] = deque()
        cam_hold_overflow = 0
        sync_report: dict[str, Any] = {}

        mcu_logs = 0
        mcu_logs_failed = 0

        def _note_mcu_log(t_master_us: int, text: str) -> None:
            """A firmware log line (IMU recovery / watchdog, apply warnings): print it
            and keep it in /events (kind ``mcu_log``, stamped on the MCU clock)."""
            nonlocal mcu_logs, mcu_logs_failed
            text = str(text).rstrip("\r\n\x00")
            mcu_logs += 1
            if mcu_logs <= _MCU_LOG_PRINT_MAX:
                print(f"mcu: {text}", file=sys.stderr)
            elif mcu_logs == _MCU_LOG_PRINT_MAX + 1:
                print(f"mcu: over {_MCU_LOG_PRINT_MAX} firmware log lines; the rest go to "
                      "/events only", file=sys.stderr)
            if not writer_opened:
                return
            try:
                writer.append_event(t_master_us=int(t_master_us), kind="mcu_log", text=text,
                                    map_quality="mcu_clock")
            except Exception:  # noqa: BLE001 - a log line must never stop the take
                mcu_logs_failed += 1

        def _handle_pump(pumped: dict[str, Any]) -> bool:
            nonlocal last_stats, stats_seen
            got = False
            by_type = pumped.get("by_type")
            if isinstance(by_type, dict):
                # Rows per record type (RAW / LIN_ACCEL / STATS included).
                for rtype, n in by_type.items():
                    if int(n or 0):
                        record_counts[int(rtype)] += int(n)
                        got = True
            else:  # native module without by_type
                for key, rtype in (
                    ("emg", RecordType.EMG_BLOCK),
                    ("mta", RecordType.IMU_MTA_QUAT),
                    ("cra", RecordType.IMU_CRA_QUAT),
                    ("trig", RecordType.TRIGGER_EVENT),
                ):
                    n = int(pumped.get(key) or 0)
                    if n:
                        record_counts[int(rtype)] += n
                        got = True
            for tr in pumped.get("triggers", ()):
                t_us = int(tr["t_master_us"])
                src = tr.get("stamp_source")
                aligner.note_trigger(
                    int(tr["trigger_id"]),
                    t_us,
                    stamp_source=int(src) if src is not None else None,
                    fallback=bool(tr.get("stamp_fallback", False)),
                )
                aligner.note_host_mcu_pair(time.monotonic_ns(), t_us)
            for lg in pumped.get("logs", ()):
                _note_mcu_log(int(lg["t_master_us"]), str(lg["text"]))
                got = True
            stats_rec = pumped.get("last_stats")
            if stats_rec is not None:
                last_stats = _stats_dict(stats_rec)
                stats_seen += 1
            return got

        def _pump_native_python(*, max_records: int) -> bool:
            nonlocal last_stats, stats_seen
            if native_cap is None:
                return False
            got = False
            # When the recorder viewer is enabled but we are on the Python writer
            # path (i.e. `use_native_writer == False`), we don't feed the native
            # viewer tap. In that case, push EMG/IMU into the *same* viewer hub
            # directly from drained MCU records so the web UI keeps working.
            viewer_push = viewer_hub is not None
            if viewer_push:
                # Local imports to keep the non-viewer path lightweight.
                from jtsmm.imu.units import accel_to_si, gyro_to_si, mag_to_si
                from jtsmm.viewer.sources import _slice_emg_for_viewer
                import numpy as np

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
                            viewer_hub.push_imu_motion(
                                slot,
                                stream,
                                t_us,
                                conv(counts),
                                accuracy=acc,
                            )

            # Bound the POP, not the loop: draining the whole queue and then
            # breaking at max_records threw the remainder away (it had already
            # left the queue). At ~3200 records/s a 128-record cap is ~40 ms of
            # data, so every loop hiccup past that silently lost EMG -- measured
            # 742 gaps and 2.95% of samples missing on a 488 s take, with the
            # native queue counters reading zero because the loss was here.
            for count, rec in enumerate(
                native_cap.drain_records(limit=max_records), start=1
            ):
                got = True
                # REC_STATS carries every MCU counter (emg_drops, tx_drops,
                # missed_drdy, frame_misalign, ...). `aligner.ingest_mcu` does not
                # route it and this path never captured it, so `mcu_stats_end` was
                # absent from the sync report for every session that used the
                # Python writer (optical / CT / experiment-plan runs) -- which
                # silently disabled the MCU-side integrity checks, since a missing
                # field reads as zero. Capture it before the viewer branch so it
                # is recorded whether or not the viewer is running.
                if rec.record_type == int(RecordType.STATS):
                    last_stats = _stats_dict(rec)
                    stats_seen += 1
                elif rec.record_type == int(RecordType.LOG):
                    _note_mcu_log(int(rec.t_master_us), str(rec.decoded.get("text", "")))
                    continue
                if viewer_push:
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
                        viewer_hub.set_imu_mta(
                            dec.get("quat"),
                            t_us,
                            dec.get("quat_report_id"),
                            dec.get("quat_accuracy"),
                        )
                    elif rt == int(RecordType.IMU_CRA_QUAT):
                        viewer_hub.set_imu_cra(
                            dec.get("quat"),
                            t_us,
                            dec.get("quat_report_id"),
                            dec.get("quat_accuracy"),
                        )
                    elif rt == int(RecordType.IMU_MTA_RAW):
                        push_raw("mta", dec, t_us)
                    elif rt == int(RecordType.IMU_CRA_RAW):
                        push_raw("cra", dec, t_us)
                    elif rt == int(RecordType.IMU_MTA_LIN_ACCEL):
                        viewer_hub.push_imu_motion(
                            "mta",
                            "linear_accel",
                            t_us,
                            dec.get("linear_accel"),
                            accuracy=dec.get("linear_accel_accuracy"),
                        )
                    elif rt == int(RecordType.IMU_CRA_LIN_ACCEL):
                        viewer_hub.push_imu_motion(
                            "cra",
                            "linear_accel",
                            t_us,
                            dec.get("linear_accel"),
                            accuracy=dec.get("linear_accel_accuracy"),
                        )
                    elif rt == int(RecordType.STATS):
                        viewer_hub.set_stats(dict(dec))
                    elif rt == int(RecordType.TRIGGER_EVENT):
                        viewer_hub.note_trigger_events(t_us, 1)

                    # Push EMG at a fixed UI cadence so the hub has time
                    # to coalesce and the browser doesn't get starved.
                    now = time.monotonic()
                    if emg_t and (now - last_push) >= 0.02:
                        t = np.asarray(emg_t, dtype=np.uint64)
                        uv = np.asarray(emg_ch, dtype=np.float32) * float(viewer_hub.emg_lsb_uv)
                        uv = _slice_emg_for_viewer(uv)
                        viewer_hub.append_emg_raw(t, uv)
                        if last_status is not None:
                            viewer_hub.set_leadoff(last_status)
                        emg_t = []
                        emg_ch = []
                        last_push = now

                for aligned in aligner.ingest_mcu(rec):
                    writer.append(aligned)
                # Count by wire type (same as the native pump's by_type), so RAW
                # and LIN_ACCEL rows are no longer tallied as QUAT.
                record_counts[int(rec.record_type)] += 1
                if count >= max_records:
                    break
            if viewer_push and emg_t:
                # Final flush when the pump slice ends.
                t = np.asarray(emg_t, dtype=np.uint64)
                uv = np.asarray(emg_ch, dtype=np.float32) * float(viewer_hub.emg_lsb_uv)
                uv = _slice_emg_for_viewer(uv)
                viewer_hub.append_emg_raw(t, uv)
                if last_status is not None:
                    viewer_hub.set_leadoff(last_status)
            return got

        def _align_and_write_cam(cam: Any, frame: Any) -> Any:
            aligned = aligner.ingest_cam(
                cam.cam_id,
                frame.sensor_timestamp_ns,
                frame.trigger_id,
                frame_idx=frame.frame_idx,
                t_host_ns=getattr(frame, "t_host_monotonic_ns", None),
            )
            if aligned is None and mock_only:
                aligned = AlignedRecord(
                    "camera",
                    int(frame.sensor_timestamp_ns // 1000),
                    {
                        "cam_id": cam.cam_id,
                        "frame_idx": frame.frame_idx,
                        "sensor_timestamp_ns": frame.sensor_timestamp_ns,
                        "trigger_id": frame.trigger_id,
                        "mock": True,
                    },
                )
            if aligned is None:
                return None
            if use_native_writer:
                writer.append_camera(aligned)
            else:
                writer.append(aligned)
            return aligned

        def _encode_preview_jpeg(cam_id: str, rgb: Any) -> bytes | None:
            if encode_jpeg is None:
                return None
            try:
                import cv2
                import numpy as np

                arr = np.asarray(rgb)
                if arr.ndim == 3:
                    arr = arr[:, :, :3]
                w, h = preview_size
                if arr.shape[1] != w or arr.shape[0] != h:
                    arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
                # Never re-detect on the capture thread. While optical tracking
                # is running, paint the worker's corners; otherwise skip overlay.
                if write_optical and optical is not None and draw_boxes is not None:
                    packed = optical.overlay_markers(cam_id)
                    if packed is not None:
                        markers, src_wh = packed
                        if markers:
                            arr = draw_boxes(arr, markers, src_wh=src_wh)
                return encode_jpeg(arr)
            except Exception as exc:  # noqa: BLE001
                if optical is not None:
                    optical.stats["preview_failed"] += 1
                    if optical.stats["preview_failed"] <= 3:
                        print(f"warning: viewer preview encode failed: {exc}", file=sys.stderr)
                return None

        def _publish_viewer_jpeg(cam_id: str, rgb: Any) -> None:
            if viewer_hub is None:
                return
            now = time.monotonic()
            if now < viewer_jpeg_due.get(cam_id, 0.0):
                return
            viewer_jpeg_due[cam_id] = now + _VIEWER_PERIOD_S
            jpeg = _encode_preview_jpeg(cam_id, rgb)
            if jpeg is not None:
                viewer_hub.update_camera_jpeg(cam_id, jpeg)

        def _maybe_dump_preview(cam_id: str, rgb: Any, t_master_us: int) -> bool:
            """Write a viewer JPEG during an active experiment phase. Returns True if encoded."""
            if preview_dump is None or walker is None or not walker.active:
                return False
            now = time.monotonic()
            due = dump_jpeg_due.get(cam_id, 0.0)
            if now < due:
                return False
            dump_jpeg_due[cam_id] = now + _VIEWER_PERIOD_S
            jpeg = _encode_preview_jpeg(cam_id, rgb)
            if not jpeg:
                return False
            cur = walker.current
            preview_dump.offer(
                cam_id=cam_id,
                jpeg=jpeg,
                t_master_us=int(t_master_us),
                phase_index=int(walker.phase_index),
                phase_name=str(getattr(cur, "name", "phase")),
                phase_kind=str(getattr(cur, "kind", "experiment")),
            )
            if viewer_hub is not None:
                viewer_hub.update_camera_jpeg(cam_id, jpeg)
            return True

        def _flush_cam_hold() -> int:
            aligned_n = 0
            while cam_hold:
                cam, frame = cam_hold[0]
                aligned = _align_and_write_cam(cam, frame)
                if aligned is None and aligner.last_ingest_waiting:
                    break
                cam_hold.popleft()
                if aligned is None:
                    continue
                aligned_n += 1
                rgb = getattr(frame, "rgb", None)
                if rgb is None:
                    continue
                if write_optical and optical is not None:
                    optical.stats["rgb_frames"] += 1
                    optical.note_rgb(
                        cam_id=cam.cam_id,
                        t_master_us=int(aligned.t_master_us),
                        rgb=rgb,
                        sensor_timestamp_ns=int(frame.sensor_timestamp_ns),
                    )
                dumped = _maybe_dump_preview(cam.cam_id, rgb, int(aligned.t_master_us))
                if not dumped:
                    _publish_viewer_jpeg(cam.cam_id, rgb)
            return aligned_n

        def _hold_cam_frame(cam: Any, frame: Any) -> None:
            nonlocal cam_hold_overflow
            if len(cam_hold) >= _CAM_HOLD_MAX:
                cam_hold.popleft()
                cam_hold_overflow += 1
            cam_hold.append((cam, frame))

        def _ingest_camera_frames(*, max_frames: int) -> bool:
            if not camera_ingest["on"]:
                return False
            got = False
            now = time.monotonic()
            # MVC still pairs every frame to TRIGGER_EVENT so the camera cursor
            # never goes stale (that lag would then infect later optical poses).
            # ArUco / preview RGB stay off; /cameras timing rows keep flowing.
            emg_only = walker is not None and walker.is_emg_only_slot()
            optical_on = write_optical and not emg_only
            for cam in cameras:
                optical_rgb = (
                    optical_on and optical is not None and optical.wants_cam(cam.cam_id)
                )
                phase_dump = (
                    preview_dump is not None
                    and walker is not None
                    and walker.active
                    and not walker.is_emg_only_slot()
                )
                viewer_rgb = (
                    (viewer_hub is not None or phase_dump)
                    and now >= viewer_rgb_due.get(cam.cam_id, 0.0)
                )
                take_rgb = optical_rgb or viewer_rgb
                for frame in cam.drain_frames(
                    max_frames=max_frames,
                    with_rgb=take_rgb,
                    rgb_limit=None if optical_rgb else (1 if viewer_rgb else 0),
                ):
                    got = True
                    if getattr(frame, "rgb", None) is not None and not optical_rgb:
                        viewer_rgb_due[cam.cam_id] = now + _VIEWER_ONLY_RGB_PERIOD_S
                    _hold_cam_frame(cam, frame)

            if _flush_cam_hold():
                got = True
            if write_optical and optical is not None:
                optical.flush()
                optical.drain_to_writer(writer)
            return got

        def _pump_mcu_slice() -> bool:
            if native_cap is None:
                return False
            if use_native_writer:
                return _handle_pump(
                    native_cap.pump_to_writer(writer, max_records=_MCU_PUMP_BATCH)
                )
            return _pump_native_python(max_records=_MCU_PUMP_BATCH)

        def _stamp_phase() -> tuple[int, int, str]:
            return _map_host(time.monotonic_ns())

        def _map_host(t_host: int) -> tuple[int, int, str]:
            if mock_only:
                return t_host, int(t_host // 1000), "mock"
            t_master, quality = aligner.map_pi_ns(t_host)
            if t_master is None:
                # Last resort: host µs — still write the event for audit.
                return t_host, int(t_host // 1000), "unavailable"
            return t_host, int(t_master), quality

        def _final_drain() -> None:
            # Streams are still on here: bound it in time, or a host slower than
            # the data rate would never see an empty queue. The stop below drains
            # the (then finite) rest.
            _ingest_camera_frames(max_frames=16)
            deadline = time.monotonic() + _FINAL_DRAIN_MAX_S
            if native_cap is not None:
                while _pump_mcu_slice() and time.monotonic() < deadline:
                    _flush_cam_hold()
            _flush_cam_hold()

        def _scan_for_stats() -> bool:
            """No take to write into (setup failed / CT only): just look for REC_STATS."""
            nonlocal last_stats, stats_seen
            recs = native_cap.drain_records(limit=_MCU_PUMP_BATCH) if native_cap else []
            for rec in recs:
                if int(rec.record_type) == int(RecordType.STATS):
                    last_stats = _stats_dict(rec)
                    stats_seen += 1
            return bool(recs)

        def _await_fresh_stats(
            drain_once: Callable[[], bool], *, seen0: int, seq0: Any, timeout_s: float
        ) -> bool:
            """Drain until a REC_STATS newer than (seen0, seq0) arrived, or timeout."""
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                got = drain_once()
                if stats_seen > seen0 and _stats_newer(last_stats, seq0):
                    return True
                if not got:
                    time.sleep(0.005)
            return False

        def _fetch_start_stats() -> dict[str, Any] | None:
            """Baseline REC_STATS right after the streams are armed.

            The firmware resets its counters at different moments (ring pushes/
            drops only at boot, IMU seq gaps/resets never, drdy/missed_drdy at
            stream start, tx drops at ads_start), so `mcu_stats_end` alone mixes
            this take with every earlier one since boot. The checker subtracts
            this snapshot. Records are pumped into the file one at a time, so the
            EMG row count at the snapshot is exact (kept as
            ``host_emg_rows_before``); nothing is discarded. The start PPS pass /
            settle already discarded everything queued before the stream start.
            """
            seen0, seq0 = stats_seen, (last_stats or {}).get("stats_seq")
            try:
                with open_hat_serial(self.port) as ser:
                    _send_stats_request(ser)
            except Exception as exc:  # noqa: BLE001 - a baseline must never break a take
                print(f"warning: could not request start MCU stats ({exc})", file=sys.stderr)
                return None

            def _one() -> bool:
                if use_native_writer:
                    return _handle_pump(native_cap.pump_to_writer(writer, max_records=1))
                return _pump_native_python(max_records=1)

            if not _await_fresh_stats(
                _one, seen0=seen0, seq0=seq0, timeout_s=_START_STATS_TIMEOUT_S
            ):
                print(
                    "warning: no REC_STATS after `stats` at the start -- MCU counters in "
                    "the integrity check cover everything since the HAT booted",
                    file=sys.stderr,
                )
                return None
            out = dict(last_stats or {})
            out["host_emg_rows_before"] = int(record_counts[int(RecordType.EMG_BLOCK)])
            return out

        def _stop_mcu_and_fetch_stats() -> dict[str, Any]:
            info: dict[str, Any] = {"cdc_stop": False, "fresh": False}
            seen0 = stats_seen
            seq0 = (last_stats or {}).get("stats_seq")
            with open_hat_serial(self.port) as ser:
                stop_mcu_streams(ser)
                info["cdc_stop"] = True
                try:
                    _send_stats_request(ser)
                except Exception as exc:  # noqa: BLE001 - streams are already stopped
                    print(f"warning: could not request final MCU stats ({exc})", file=sys.stderr)
                    return info
            into_file = loop_started and writer_opened

            def _drain_once() -> bool:
                if into_file:
                    got = _pump_mcu_slice()
                    _ingest_camera_frames(max_frames=16)
                    return got
                return _scan_for_stats()

            deadline = time.monotonic() + _FINAL_STATS_TIMEOUT_S
            if _await_fresh_stats(
                _drain_once, seen0=seen0, seq0=seq0, timeout_s=_FINAL_STATS_TIMEOUT_S
            ):
                info["fresh"] = True
                while _drain_once() and time.monotonic() < deadline:
                    pass  # streams are off: the tail is finite
            if into_file:
                _flush_cam_hold()
            if not info["fresh"]:
                print(
                    "warning: no REC_STATS after the stop -- mcu_stats_end may predate "
                    "the end of the take (firmware without `stats`?)",
                    file=sys.stderr,
                )
            return info

        def _collect_viewer_notes() -> int:
            """Write notes the viewer received since the last call to /events."""
            nonlocal notes_failed
            log = getattr(viewer_server, "notes", None)
            if log is None or not writer_opened:
                return 0
            n = 0
            for note in log.drain():
                t_host = int(note["t_host_ns"])
                _, t_master, quality = _map_host(t_host)
                try:
                    row = writer.append_event(
                        t_master_us=t_master, kind="note", text=str(note["text"]),
                        t_host_ns=t_host, map_quality=quality, viewer_t_s=note.get("t_s"),
                    )
                except Exception as exc:  # noqa: BLE001 - a note must never stop the take
                    notes_failed += 1
                    print(f"warning: viewer note not written ({exc}): {note['text']!r}",
                          file=sys.stderr)
                    continue
                note_rows.append({"row": int(row), "t_host_ns": t_host,
                                  "t_master_us": int(t_master), "map_quality": quality})
                n += 1
            return n

        def _write_phase_events(events: list[Any]) -> None:
            for ev in events:
                writer.append_event(
                    t_master_us=ev.t_master_us,
                    kind=f"phase_{ev.edge}",
                    text=ev.name,
                    phase_kind=ev.kind,
                    t_host_ns=ev.t_host_ns,
                    map_quality=ev.map_quality,
                )

        try:
            stop.install()
            cameras = (
                cameras_from_config(
                    host_bundle.get("host", {}).get("cameras", {}),
                    track_lores=want_cameras,
                )
                if want_cameras
                else []
            )
            if not mock_only:
                # Drop YAML cams that failed to open — mock frames would pollute sync.
                opened = [c for c in cameras if not c.is_mock]
                skipped = [c.cam_id for c in cameras if c.is_mock]
                if skipped:
                    print(
                        f"warning: skipping cameras without picamera2 device: {', '.join(skipped)}",
                        file=sys.stderr,
                    )
                    session_meta["skipped_cameras"] = ",".join(skipped)
                cameras = opened
            # The aligner keeps early triggers until every recording camera has
            # anchored, instead of trimming them from under a slow starter.
            aligner.expected_cameras = tuple(c.cam_id for c in cameras)
            viewer_rgb_due.update({c.cam_id: 0.0 for c in cameras})
            dump_jpeg_due.update({c.cam_id: 0.0 for c in cameras})

            if need_optical and not mock_only and not cameras:
                raise RuntimeError(
                    "optical tracking is on but no cameras opened — check host.cameras "
                    "and picamera2 (IMU mounting cal cannot run without ArUco RGB)"
                )

            if need_optical:
                try:
                    optical = OpticalTrackBridge(
                        self.optical_config, cameras, track_hz=self.optical_track_hz
                    )
                    stride = int(getattr(optical, "_track_stride", 1) or 1)
                    if stride > 1:
                        print(
                            f"optical: live tracking every {nth(stride)} stereo pair "
                            f"({optical._camera_hz / stride:g} Hz); "
                            f"cameras stay at {optical._camera_hz:g} Hz",
                            file=sys.stderr,
                        )
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"failed to start optical tracker: {exc}") from exc
                session_meta.update(optical.session_meta(cameras, self.optical_config))

            if self.experiment_plan is not None:
                session_meta["experiment_name"] = str(self.experiment_plan.name)
                session_meta["experiment_plan_json"] = json.dumps(
                    self.experiment_plan.to_dict(), separators=(",", ":")
                )

            use_native_cfg = capture_native_prefer(self.resolved)
            use_native_writer = (
                use_native_cfg
                and native.use_native()
                and native.NativeHdf5Writer is not None
                and not mock_only
                and not need_optical
                and not self.ct_calibration
                and self.experiment_plan is None
            )
            if (use_native_writer and self.viewer
                    and not hasattr(native.NativeHdf5Writer, "append_event")):
                # A native module built before /events support cannot store
                # viewer notes; the Python writer can (same file layout).
                print("note: native HDF5 writer predates /events (rebuild with "
                      "native/build.sh); using the Python writer so viewer notes are saved",
                      file=sys.stderr)
                use_native_writer = False
            if not mock_only and not (native.use_native() and native.NativeHdf5Writer is not None):
                _require_native()

            writer_kw = dict(
                path=h5_path,
                config_hash=cfg_hash,
                mcu_config=mcu_cfg,
                host_config=host_bundle,
                firmware_version=firmware_version,
                session_meta=session_meta,
            )
            # Say which backend runs every time: USB capture is always native on
            # hardware; only the HDF5 writer varies (Python when the take needs
            # optical tracking, CT calibration, a phase plan or native_prefer off).
            capture_kind = "mock" if mock_only else "native (C++/libusb)"
            if use_native_writer:
                writer_kind = "native (libhdf5)"
            else:
                why = [w for w, on in (
                    ("optical tracking", need_optical),
                    ("CT calibration", self.ct_calibration),
                    ("experiment plan", self.experiment_plan is not None),
                    ("native_prefer: false", not use_native_cfg),
                    ("mock", mock_only),
                ) if on]
                writer_kind = "Python" + (f" ({', '.join(why)})" if why else "")
            print(f"capture backend: {capture_kind}; HDF5 writer: {writer_kind}", file=sys.stderr)
            if use_native_writer:
                writer = NativeWriter(**writer_kw)
            else:
                writer = Hdf5Writer(**writer_kw)
            writer.open()
            writer_opened = True
            timing["open"] = time.monotonic_ns()

            if not mock_only:
                native_cap = NativeCapture(port=self.port)
                native_cap.start()
                if streams_running:
                    native_cap.discard_queued()

            if not mock_only and not streams_running:
                _arm_mcu_streams(
                    self.port,
                    native_cap,
                    self.resolved,
                    emg=emg_on,
                    trigger=trigger_on and not trigger_after_cams,
                    imu_mta=enables["imu_mta"],
                    imu_cra=enables["imu_cra"],
                )
                timing["sensors_armed"] = time.monotonic_ns()
                if native_cap is not None and pps_n > 0:
                    pps_start = _pps_handshake(self.port, native_cap, aligner, pps_n, label="start")
                    _print_pps(pps_start, aligner)
                if native_cap is not None:
                    stats_start = _fetch_start_stats()

            # EMG/IMU records keep queueing in the native capture (C++ threads,
            # no GIL) while picamera2 starts (~0.5-1 s); the loop's bounded pops
            # catch up afterwards, so this is sensor pre-roll, not a hole.
            timing["cams_start"] = time.monotonic_ns()
            for cam in cameras:
                cam.start()
            timing["cams_started"] = time.monotonic_ns()

            if not mock_only:
                if streams_running and native_cap is not None:
                    native_cap.discard_queued()
                if trigger_after_cams:
                    aligner.reset_trigger_backlog()
                    if not streams_running and trigger_on:
                        _arm_mcu_trigger(self.port)
                        timing["trigger_armed"] = time.monotonic_ns()

            tip_applied = False
            if self.ct_calibration:
                if optical is None:
                    raise RuntimeError("ct_calibration requires optical tracking")
                if mock_only:
                    print(
                        "warning: skipping CT pivot calibration in --mock "
                        "(no camera RGB); tip not updated",
                        file=sys.stderr,
                    )
                else:
                    from .ct_pivot import run_ct_pivot_calibration, write_ct_tip_calibration

                    discard = None
                    if native_cap is not None:
                        discard = native_cap.discard_queued
                    persist_dir = None
                    if self.persist_ct_calibration:
                        persist_dir = ct_calibration_recording_dir(
                            self.ct_calibration_recordings_dir,
                            experiment_name=plan_name,
                        )
                    pivot = run_ct_pivot_calibration(
                        optical,
                        cameras,
                        n_poses=self.ct_calibration_n_poses,
                        session_dir=session_dir,
                        persist_dir=persist_dir,
                        experiment_name=plan_name,
                        discard_mcu=discard,
                    )
                    from .ct_pivot import ct_tip_trustworthy

                    trusted = ct_tip_trustworthy(pivot)
                    if self.ct_tip_calibration is not None and trusted:
                        shared = write_ct_tip_calibration(self.ct_tip_calibration, pivot)
                        print(
                            f"Wrote reusable CT tip offset -> {shared}",
                            file=sys.stderr,
                        )
                    elif self.ct_tip_calibration is not None:
                        # An untrustworthy tip must not overwrite the good one
                        # the next sessions load; it stays with this session.
                        print(
                            f"NOT updating the shared CT tip file {self.ct_tip_calibration} "
                            f"(rms {pivot.rms_mm:.1f} mm is too large); this session keeps "
                            "the new tip in its own ct_tip_calibration.json",
                            file=sys.stderr,
                        )
                    optical.set_ct_tip_body_mm(pivot.tip_body_mm)
                    session_meta["ct_tip_body_mm"] = ",".join(
                        f"{x:.6f}" for x in pivot.tip_body_mm.tolist()
                    )
                    session_meta["ct_tip_rms_mm"] = f"{pivot.rms_mm:.6f}"
                    session_meta["ct_tip_source"] = "calibrated"
                    session_meta["ct_tip_trusted"] = "true" if trusted else "false"
                    tip_applied = True
            elif tip_path is not None and optical is not None:
                from .ct_pivot import load_ct_tip_calibration

                pivot = load_ct_tip_calibration(tip_path)
                optical.set_ct_tip_body_mm(pivot.tip_body_mm)
                session_meta["ct_tip_body_mm"] = ",".join(
                    f"{x:.6f}" for x in pivot.tip_body_mm.tolist()
                )
                session_meta["ct_tip_rms_mm"] = f"{pivot.rms_mm:.6f}"
                session_meta["ct_tip_source"] = str(tip_path)
                print(
                    f"Loaded CT tip offset from {tip_path}  "
                    f"tip_body_mm={pivot.tip_body_mm.tolist()}  rms={pivot.rms_mm:.3f} mm",
                    file=sys.stderr,
                )
                tip_applied = True

            if tip_applied and hasattr(writer, "_file") and writer._file is not None:
                for k in ("ct_tip_body_mm", "ct_tip_rms_mm", "ct_tip_source", "ct_tip_trusted"):
                    if k in session_meta:
                        writer._file["meta"].attrs[k] = session_meta[k]

            if self.ct_calibration_only:
                print("CT tip calibration done (no take).", file=sys.stderr)
                end_reason = "ct_calibration_only"
            else:
                viewer_server = self._maybe_start_viewer(native_cap, mcu_cfg)
                viewer_hub = viewer_server.hub if viewer_server is not None else None
                if getattr(viewer_server, "notes", None) is not None:
                    # Tells the viewer UI that notes go into this session file.
                    viewer_server.notes.saved_to = h5_path.name
                if viewer_hub is not None:
                    viewer_hub.set_camera_ids([c.cam_id for c in cameras])

                if self.experiment_plan is not None:
                    walker = PhaseWalker.from_plan(self.experiment_plan, _stamp_phase)
                    self._print_plan_banner(walker, duration_s, str(self.experiment_plan.name))
                    from .preview_dump import PreviewJpegDump

                    if want_cameras:
                        preview_dump = PreviewJpegDump(session_dir)
                        if hasattr(writer, "sidecar_paths"):
                            writer.sidecar_paths[preview_dump.INDEX_NAME] = preview_dump.index_path
                        print(
                            f"preview JPEGs: during active phases -> {preview_dump.root}",
                            file=sys.stderr,
                        )

                if duration_s is None:
                    t_end = float("inf")
                else:
                    t_end = time.monotonic() + float(duration_s)
                next_flush = time.monotonic()
                if keys_ctx is not None:
                    keys_ctx.__enter__()
                    keys_entered = True
                loop_started = True
                run_phase = "take"
                watchdog_on = native_cap is not None and not mock_only and any(enables.values())
                wd_count = -1
                wd_progress_t = time.monotonic()
                wd_health_due = 0.0
                timing["loop_start"] = time.monotonic_ns()
                stop.enter_loop()
                try:
                    while time.monotonic() < t_end:
                        if stop.reason is not None:
                            break
                        mcu_got = False
                        for _ in range(4):
                            if not _pump_mcu_slice():
                                break
                            mcu_got = True
                        if watchdog_on:
                            now_wd = time.monotonic()
                            cnt = _watched_record_count(record_counts, emg=emg_on)
                            if cnt != wd_count:
                                wd_count, wd_progress_t = cnt, now_wd
                            elif now_wd - wd_progress_t > _WATCHDOG_SILENCE_S:
                                watchdog = {
                                    "reason": "no EMG records" if emg_on else "no MCU records",
                                    "silent_s": round(now_wd - wd_progress_t, 3),
                                }
                                break
                            if now_wd >= wd_health_due:
                                wd_health_due = now_wd + 0.25
                                if _device_lost(native_cap):
                                    watchdog = {"reason": "device_lost"}
                                    break
                        cam_got = _ingest_camera_frames(max_frames=8)
                        if mcu_got:
                            _flush_cam_hold()
                        if mcu_got or cam_got:
                            now_ns = time.monotonic_ns()
                            if timing["first_data"] is None:
                                timing["first_data"] = now_ns
                            timing["last_data"] = now_ns
                        if walker is not None and keys_ctx is not None:
                            # Edges stay in walker.events; they are re-stamped
                            # with the final PPS fit and written at close.
                            evs = walker.poll(keys_ctx.poll())
                            for _ev in evs:
                                if optical is not None and walker.is_emg_only_slot():
                                    optical.flush(force=True)
                            if walker.finished:
                                print("Stopping capture (phases finished).", file=sys.stderr)
                                break
                        if not mcu_got and not cam_got:
                            time.sleep(0.0001)
                        if time.monotonic() >= next_flush:
                            _collect_viewer_notes()
                            writer.flush()
                            next_flush = time.monotonic() + 0.25
                except KeyboardInterrupt:
                    # Fallback when the SIGINT handler could not be installed (or
                    # a raw ^C byte reached TerminalKeys): still a stop request.
                    # A Ctrl-C *after* a stop request is a hard abort.
                    if stop.reason is not None:
                        raise
                    stop.reason = "operator_stop"
                finally:
                    stop.leave_loop()
                    timing["loop_end"] = time.monotonic_ns()
                run_phase = "finish"

                if watchdog is not None:
                    end_reason = "watchdog"
                    print(
                        f"ERROR: {watchdog['reason']}"
                        + (f" for {watchdog['silent_s']:.1f} s" if "silent_s" in watchdog else "")
                        + " while the MCU streams are armed -- USB link dead or HAT reset? "
                        "Stopping the take and closing the file.",
                        file=sys.stderr,
                    )
                elif stop.reason is not None:
                    end_reason = stop.reason
                elif walker is not None and walker.finished:
                    end_reason = "operator_stop" if walker.aborted else "phases_complete"
                elif walker is not None:
                    # --duration is the plan's safety timeout: it ran out before
                    # the last phase, so the take is incomplete, not "duration".
                    end_reason = "plan_timeout"
                else:
                    end_reason = "duration"
                if walker is not None and not walker.finished:
                    print(f"Take ended ({end_reason}) — closing open phase(s).", file=sys.stderr)
                    walker.poll(ord("q"))
        except BaseException as exc:  # noqa: BLE001 - re-raised after the teardown
            failure = exc
            if isinstance(exc, SessionStopSignal):
                end_reason = "signal"
            elif isinstance(exc, KeyboardInterrupt):
                end_reason = "operator_abort"
            else:
                end_reason = f"exception: {type(exc).__name__}: {exc}"

        # --- teardown: every step isolated; the writer is always closed last ------
        stop.begin_teardown()
        td = _Teardown()
        prev_sigint = _ignore_sigint()
        try:
            if keys_entered and keys_ctx is not None:
                td.step("restore terminal", lambda: keys_ctx.__exit__(None, None, None))
            if loop_started:
                td.step("final drain", _final_drain)
            # Cameras stop while the trigger still pulses: an externally
            # triggered IMX296 can wait in picamera2's stop() for a frame that no
            # trigger will start (bench 2026-09-25). Both stops run together.
            # After a short grace the trigger goes off, and only then do we wait
            # for stop() to return. Waiting for both sequential stops before
            # trig_stop left ~0.67 s of pulses with no camera rows. Ingest is
            # off once stop() has started so the stats drain cannot race it.
            # Each join is bounded, so a hang cannot keep the file open.
            camera_ingest["on"] = False
            stop_jobs: list[tuple[Any, threading.Thread, dict[str, Any]]] = []
            for cam in cameras:
                box: dict[str, Any] = {}

                def _run_stop(cam: Any = cam, box: dict[str, Any] = box) -> None:
                    try:
                        box["value"] = cam.stop()
                    except BaseException as exc:  # noqa: BLE001 - re-raised on join
                        box["error"] = exc

                th = threading.Thread(
                    target=_run_stop, name=f"teardown-camera {cam.cam_id} stop", daemon=True
                )
                th.start()
                stop_jobs.append((cam, th, box))
            if stop_jobs and any(getattr(c, "_picam", None) is not None for c, _, _ in stop_jobs):
                time.sleep(_CAMERA_STOP_GRACE_S)
            elif stop_jobs:
                # Test doubles return from stop() immediately and log as they
                # do. Wait that out so the stop is ordered before trig_stop.
                # A stop that never returns (the hang case) falls through and
                # the bounded join below reports it.
                deadline = time.monotonic() + 0.05
                while time.monotonic() < deadline and any(
                    "value" not in box and "error" not in box for _, _, box in stop_jobs
                ):
                    time.sleep(0.001)
            # Stop order: trigger + streams off over CDC while the native capture
            # still reads the vendor link (CDC works alongside it -- the PPS
            # handshake relies on that too), ask for a fresh REC_STATS and drain
            # the tail into the file, THEN the end PPS pass (which discards
            # queued records) and only then stop the native capture. Stopping
            # the capture first left `mcu_stats_end` as whatever periodic stats
            # record happened to arrive last, before the stop.
            final_stats: dict[str, Any] = {}
            if native_cap is not None and not mock_only:
                got = td.step("MCU stop + final stats", _stop_mcu_and_fetch_stats)
                if isinstance(got, dict):
                    final_stats = got
            for cam, th, box in stop_jobs:
                def _join_stop(
                    th: threading.Thread = th,
                    box: dict[str, Any] = box,
                    cam: Any = cam,
                ) -> None:
                    th.join(_CAMERA_STOP_TIMEOUT_S)
                    if th.is_alive():
                        raise TimeoutError(
                            f"camera {cam.cam_id} stop did not return within "
                            f"{_CAMERA_STOP_TIMEOUT_S:g} s (left running)"
                        )
                    if "error" in box:
                        raise box["error"]

                td.step(f"camera {cam.cam_id} stop", _join_stop)
            if loop_started and failure is None and native_cap is not None and not mock_only and pps_n > 0:
                pps_end = td.step(
                    "end PPS",
                    lambda: _pps_handshake(self.port, native_cap, aligner, pps_n, label="end"),
                )
                td.step("print PPS", lambda: _print_pps(pps_end, aligner))
            if native_cap is not None:
                td.step("native capture stop", native_cap.stop)
            if not mock_only and not final_stats.get("cdc_stop"):
                td.step("MCU stream stop (fallback)", lambda: stop_mcu_streams_robust(self.port))
            if optical is not None:
                td.step(
                    "optical shutdown",
                    lambda: optical.shutdown(writer if writer_opened else _NullSink()),
                )
            if viewer_server is not None:
                td.step("viewer shutdown", viewer_server.shutdown)
                # Notes typed after the last in-loop flush (until the server stopped).
                td.step("viewer notes", _collect_viewer_notes)
            dump_stats = None
            if preview_dump is not None:
                dump_stats = td.step("preview JPEG dump close", preview_dump.close)
                if dump_stats is None:
                    dump_stats = {"written": 0, "dropped": 0, "write_errors": 1}
                n_written = int(dump_stats.get("written") or 0)
                n_drop = int(dump_stats.get("dropped") or 0)
                print(
                    f"preview JPEGs: wrote {n_written} frames"
                    + (f" (dropped {n_drop})" if n_drop else ""),
                    file=sys.stderr,
                )
            if writer_opened:
                built = td.step(
                    "sync report",
                    lambda: self._build_sync_report(
                        aligner,
                        dump_stats=dump_stats,
                        last_stats=last_stats,
                        cam_hold_overflow=cam_hold_overflow,
                        cam_hold_leftover=len(cam_hold),
                        optical=optical,
                        capture_order=capture_order,
                        pps_start=pps_start,
                        pps_end=pps_end,
                        native_cap=native_cap,
                    ),
                )
                sync_report = built if built is not None else {}
                sync_report["end_reason"] = end_reason
                if stop.signame is not None:
                    sync_report["end_signal"] = stop.signame
                if watchdog is not None:
                    sync_report["watchdog"] = dict(watchdog)
                sync_report["disk_preflight"] = disk_info
                if cameras:
                    sync_report["camera_missing_sensor_ts"] = {
                        c.cam_id: int(getattr(c, "missing_sensor_ts", 0) or 0) for c in cameras
                    }
                sync_report["session_timing"] = _session_timing(timing)
                sync_report["startup"] = _startup_timing(timing)
                td.step(
                    "timing meta",
                    lambda: _set_meta_attrs(writer, _timing_meta(sync_report["session_timing"])),
                )
                if stats_start is not None:
                    sync_report["mcu_stats_start"] = stats_start
                if final_stats:
                    # False: no REC_STATS arrived after the stop, so mcu_stats_end
                    # (if present) predates the end of the take.
                    sync_report["mcu_stats_end_fresh"] = bool(final_stats.get("fresh"))
                if walker is not None:
                    # After the end PPS pass: re-stamp every edge from its stored
                    # host time with the final fit, BEFORE /events, /phases and
                    # the JTS frame intervals are written.
                    remap = td.step(
                        "re-map phase stamps",
                        lambda: remap_phase_stamps(walker, aligner, enabled=not mock_only),
                    )
                    phases = [p.to_dict() for p in walker.recorded]
                    if isinstance(remap, dict):
                        for ph, live in zip(phases, remap.pop("live_intervals", [])):
                            ph["t_start_us_live"], ph["t_end_us_live"] = live
                        sync_report["phase_stamp_remap"] = remap
                        if remap["n_remapped"]:
                            print(
                                f"phase stamps: re-mapped {remap['n_remapped']}/"
                                f"{remap['n_events']} edges with the final PPS fit "
                                f"(max correction {remap['max_abs_correction_us'] / 1000.0:.1f} ms)",
                                file=sys.stderr,
                            )
                    sync_report["phases"] = phases
                    sync_report["phases_aborted"] = bool(walker.aborted)
                    td.step("phase events", lambda: _write_phase_events(walker.events))
                    td.step("phases table", lambda: writer.write_phases_table(phases))
                if viewer_server is not None or note_rows:
                    # After the end PPS pass, like the phase edges above.
                    notes_info = td.step(
                        "re-map note stamps",
                        lambda: remap_note_stamps(note_rows, aligner, writer,
                                                  enabled=not mock_only),
                    )
                    if not isinstance(notes_info, dict):
                        notes_info = {"n_notes": len(note_rows)}
                    notes_info["n_failed"] = notes_failed
                    notes_info["stored_in"] = "/events (kind=note)"
                    sync_report["viewer_notes"] = notes_info
                sync_report["mcu_logs"] = {"n": mcu_logs, "n_failed": mcu_logs_failed,
                                           "stored_in": "/events (kind=mcu_log)"}
                if td.failed:
                    sync_report["teardown_errors"] = list(td.failed)
                td.step("write sync report", lambda: writer.write_sync_report(sync_report))
                td.step("write sync drops", lambda: writer.write_sync_drops(aligner.drops))
            if writer is not None:
                td.step("close HDF5", writer.close)
        finally:
            _restore_sigint(prev_sigint)
            stop.restore()

        if end_reason not in ("duration", "phases_complete", "ct_calibration_only"):
            print(f"take ended: {end_reason}", file=sys.stderr)
        if failure is not None:
            try:
                failure.jtsmm_phase = run_phase  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - some builtins refuse attributes
                pass
            if writer_opened:
                try:
                    failure.jtsmm_session_h5 = h5_path  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 - some builtins refuse attributes
                    pass
                print(f"session file closed after failure -> {h5_path}", file=sys.stderr)
            raise failure

        if self.ct_calibration_only:
            return h5_path

        self._embed_imu_mounting(h5_path)

        if walker is not None and walker.recorded and self.write_jts_config:
            _maybe_write_jts_config(
                h5_path,
                plan=self.experiment_plan,
                timed=walker.recorded,
                jts_config_path=self.jts_config_path,
            )

        _print_record_summary(record_counts)
        _print_mcu_stats(last_stats)
        _print_sync_summary(sync_report)
        return h5_path

    def _embed_imu_mounting(self, h5_path: Path) -> None:
        """Copy the accepted IMU mountings of ``session.imu_mounting_calibration`` into the
        closed session (``/calibration/imu_*``), as the CT tip goes into ``/meta``. Not in
        an IMU calibration take: ``jtsmm imu calibrate`` writes its fresh fit there."""
        from jtsmm.config.registry import session_imu_mounting_calibration

        if str(self.extra_meta.get("protocol", "")) == "imu_mounting":
            return
        src = session_imu_mounting_calibration(self.resolved)
        if not src or not Path(src).exists():
            print("IMU mounting: no stable mounting file -- not embedded "
                  "(run `jtsmm imu calibrate` after each remount)", file=sys.stderr)
            return
        try:
            from jtsmm.imu.mounting import embed_mounting

            slots = embed_mounting(h5_path, src)
        except Exception as exc:  # noqa: BLE001 - never lose a take over this
            print(f"warning: IMU mounting not embedded: {exc}", file=sys.stderr)
            return
        if slots:
            print(f"IMU mounting embedded ({', '.join(slots)}) from {src}", file=sys.stderr)

    @staticmethod
    def _print_plan_banner(walker: PhaseWalker, duration_s: float | None, plan_name: str) -> None:
        n_mvc = sum(1 for p in walker.phases if getattr(p, "kind", "") == "mvc")
        n_cal = sum(1 for p in walker.phases if "calibration" in getattr(p, "kind", ""))
        n_mot = sum(1 for p in walker.phases if getattr(p, "kind", "") == "experiment")
        bits = []
        if n_mvc:
            bits.append(f"{n_mvc} MVC EMG-only")
        if n_cal:
            bits.append(f"{n_cal} landmark")
        if n_mot:
            bits.append(f"{n_mot} motion")
        print("=" * 72, file=sys.stderr)
        if n_mvc and not n_mot and not n_cal:
            print(
                f"MVC recording: {len(walker.phases)} EMG-only phases "
                "(mvc_opening, mvc_closing). Independent of the motion plan.",
                file=sys.stderr,
            )
        else:
            print(
                f"Experiment '{plan_name}': "
                f"{len(walker.phases)} phases ({', '.join(bits) or 'none'}). "
                "Recording continues between SPACE presses.",
                file=sys.stderr,
            )
        print("  SPACE = start/end phase | Q/ESC = finish early", file=sys.stderr)
        if n_mvc:
            print(
                "  MVC: EMG + IMU; cameras/ArUco off unless the live viewer is on.",
                file=sys.stderr,
            )
        if duration_s is None:
            print(
                "  No duration timeout — session ends when the last phase ends.",
                file=sys.stderr,
            )
        else:
            print(
                f"  Safety timeout: {float(duration_s):g}s "
                "(SPACE still walks the plan).",
                file=sys.stderr,
            )
        print("=" * 72, file=sys.stderr)
        walker.print_prompt()

    @staticmethod
    def _build_sync_report(
        aligner: ClockAligner,
        *,
        dump_stats: dict[str, Any] | None,
        last_stats: dict[str, Any] | None,
        cam_hold_overflow: int,
        cam_hold_leftover: int,
        optical: Any,
        capture_order: str,
        pps_start: dict[str, Any] | None,
        pps_end: dict[str, Any] | None,
        native_cap: Any,
    ) -> dict[str, Any]:
        sync_report = aligner.sync_report()
        if dump_stats is not None:
            sync_report["preview_jpeg"] = dump_stats
        if last_stats is not None:
            sync_report["mcu_stats_end"] = last_stats
        # Always present (0 included) so the integrity check can grade them:
        # overflow = camera frames dropped because their trigger never arrived
        # within the hold window; leftover = frames still unpaired at the end.
        sync_report["cam_hold_overflow"] = int(cam_hold_overflow)
        sync_report["cam_hold_leftover"] = int(cam_hold_leftover)
        relocks = sync_report.get("validation_relock_count") or {}
        fails = sync_report.get("validation_fail_count") or {}
        if any(int(v) > 0 for v in relocks.values()):
            print(
                f"warning: camera trigger fit re-locked {dict(relocks)} "
                f"after validation_fail {dict(fails)}",
                file=sys.stderr,
            )
        elif any(int(v) > 50 for v in fails.values()):
            print(
                f"warning: camera trigger validation failed {dict(fails)} "
                "(frames dropped; camera timestamps disagreed with MCU triggers)",
                file=sys.stderr,
            )
        if optical is not None and optical.stats:
            sync_report["optical"] = dict(optical.stats)
            st = optical.stats
            stride = int(st.get("track_stride") or 1)
            rate = ""
            if stride > 1:
                rate = f" (every {nth(stride)} pair)"
            print(
                "optical: "
                f"paired={int(st.get('paired', 0))} "
                f"enqueued={int(st.get('enqueued', 0))} "
                f"track_skipped={int(st.get('track_skipped', 0))} "
                f"queue_dropped={int(st.get('queue_dropped', 0))} "
                f"pair_dropped={int(st.get('pair_dropped', 0))} "
                f"queue_depth_max={int(st.get('queue_depth_max', 0))}"
                f"{rate}",
                file=sys.stderr,
            )
        sync_report["capture_order"] = capture_order
        sync_report["pps"] = {"start": pps_start, "end": pps_end}
        sync_report["pi_clock"] = _pi_clock_note()
        if native_cap is not None:
            sync_report["host_capture"] = {
                "queue_drops": native_cap.queue_drops,
                **native_cap.link_stats,
            }
        return sync_report

    def _maybe_start_viewer(self, native_cap: NativeCapture | None, mcu_cfg: dict[str, Any]) -> Any:
        """Launch the in-recording viewer (native tap + FastAPI thread), if enabled."""
        if not self.viewer or native_cap is None:
            return None
        try:
            from jtsmm.viewer.server import ViewerServer
            from jtsmm.viewer.sources import TapSource

            emg = mcu_cfg.get("emg", {})
            leadoff = emg.get("leadoff") or {}
            source = TapSource(
                native_cap,
                emg_fs=int(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS)),
                emg_gain=int(emg.get("default_gain", 6)),
                leadoff_during_recording=bool(leadoff.get("during_recording")),
                leadoff_current_na=float(leadoff.get("current_na", 6) or 6),
                line_hz=emg_check_line_hz(self.resolved),
                mains_mode=emg_filter_mains_mode(self.resolved),
                imu_enabled={s: mcu_stream_enables(self.resolved)[f"imu_{s}"] for s in ("mta", "cra")},
                comparators_enabled={"true": True, "false": False}.get(
                    str(self.extra_meta.get("emg_comparators_enabled", ""))),
                leadoff_ac_tone=bool(leadoff.get("ac_tone")),
            )
            server = ViewerServer(source, host=self.viewer_host, port=self.viewer_port)
            server.start_in_thread()
            server.print_startup(extra="in-recording tap")
            return server
        except Exception as exc:  # noqa: BLE001 - viewer must never break recording
            print(f"warning: viewer disabled ({exc})", file=sys.stderr)
            return None


def _maybe_write_jts_config(
    h5_path: Path,
    *,
    plan: Any,
    timed: list[Any],
    jts_config_path: Path | None,
) -> None:
    """Build optical-style JTS config from master-time phases when optical rows exist."""
    try:
        import h5py
        from optical.tracking.experiment import RecordedPhase, build_jts_config, write_jts_config
    except Exception as exc:  # noqa: BLE001
        print(f"warning: skipped JTS config ({exc})", file=sys.stderr)
        return

    try:
        with h5py.File(h5_path, "r") as f:
            if "optical" not in f or "t_master_us" not in f["optical"]:
                print(
                    "warning: no /optical/t_master_us — wrote phase times only; "
                    "skipping JTS frame_interval config",
                    file=sys.stderr,
                )
                return
            t_opt = f["optical"]["t_master_us"][()]
            try:
                from jtsmm.tools.timegrid import session_frame_period_us

                # Frame indices live on the trigger grid; take its period from
                # the triggers, not re-estimated from gappy optical rows.
                period_us = session_frame_period_us(f)
            except Exception:  # noqa: BLE001 - fall back to the optical-row estimate
                period_us = None
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not read optical timestamps for JTS config ({exc})", file=sys.stderr)
        return

    # Inclusive [start, end] frame intervals (JTS convention).
    intervals = optical_frame_intervals(timed, t_opt, period_us=period_us)
    if not intervals:
        print("warning: no optical frames overlapped phase intervals — skipped JTS config", file=sys.stderr)
        return
    recorded = [
        RecordedPhase(name=n, kind=k, start_frame=s, end_frame=e)
        for n, k, s, e in intervals
    ]
    out = jts_config_path or (h5_path.parent / f"{plan.slug}_jts_config.json")
    cfg = build_jts_config(recorded, session_path=h5_path, description=plan.description)
    # Multimodal session path — keep data_source pointing at this HDF5.
    write_jts_config(out, cfg)
    print(f"Wrote JTS config -> {out}", file=sys.stderr)


def _print_mcu_stats(stats: dict[str, Any] | None) -> None:
    if not stats:
        return
    order = [
        "drdy_count",
        "missed_drdy",
        "emg_push",
        "emg_drops",
        "imu_mta_push",
        "imu_mta_drops",
        "imu_cra_push",
        "imu_cra_drops",
        "events_push",
        "events_drops",
        "usb_tx_bytes",
        "sensor_loop_hz",
        "comms_loop_hz",
        "tx_drops",
        "max_drdy_gap_us",
        "imu_svc_max_us",
        "leadoff_max_us",
        "leadoff_count",
        "poll_max_us",
        "frame_misalign",
    ]
    parts = [f"{k}={stats[k]}" for k in order if k in stats]
    print(f"mcu_stats: {' '.join(parts)}", file=sys.stderr)


def _print_pps(result: dict[str, Any] | None, aligner: ClockAligner) -> None:
    if not result:
        return
    rtt = result.get("rtt_ms_median")
    rtt_s = f"{rtt:.1f} ms" if rtt is not None else "n/a"
    fit = aligner._pi_fit  # noqa: SLF001 - report-only
    print(
        f"pps[{result.get('pass')}]: accepted {result.get('accepted')}/{result.get('attempted')} "
        f"(timeouts {result.get('timeouts')}, slow {result.get('rejected_rtt')}), "
        f"rtt median {rtt_s}, pi->master a={fit.a:.9f} b={fit.b:.0f} us n={fit.n_samples}",
        file=sys.stderr,
    )
    if result.get("accepted", 0) == 0:
        print(
            "warning: no `pps` replies -- firmware older than 2026-09 or CDC/bulk link "
            "problem; camera<->trigger pairing cannot be cross-checked this session",
            file=sys.stderr,
        )


def _print_sync_summary(sync: dict[str, Any]) -> None:
    """Say in words what the sync report says in numbers (mirrors `jtsmm integrity`)."""
    notes = sync.get("viewer_notes")
    if isinstance(notes, dict):
        line = f"viewer notes: {int(notes.get('n_notes') or 0)} saved to /events (kind=note)"
        if notes.get("n_remapped"):
            line += (f", {notes['n_remapped']} re-mapped with the final PPS fit (max "
                     f"{int(notes.get('max_abs_correction_us') or 0) / 1000.0:.1f} ms)")
        print(line, file=sys.stderr)
        if int(notes.get("n_failed") or 0):
            print(f"WARNING: {notes['n_failed']} viewer note(s) could not be written",
                  file=sys.stderr)
    stamp = sync.get("trigger_stamp") or {}
    counts = stamp.get("counts") or {}
    if counts:
        print("trigger stamps: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())),
              file=sys.stderr)
    if int(stamp.get("fallback", 0) or 0) > 0:
        print(
            f"WARNING: {stamp['fallback']} TRIGGER_EVENTs were stamped by the firmware's "
            "timer fallback (PIO IRQ went silent) -- camera times are predicted, not "
            "measured; check `trig_status` on the HAT.",
            file=sys.stderr,
        )
    elif counts and counts.get("pio_irq", 0) == 0 and (counts.get("timer", 0) or counts.get("unknown", 0)):
        print(
            "note: TRIGGER_EVENTs are timer-stamped (legacy firmware or `trig_stamp timer`); "
            "expect ~4 us/frame drift against the real pulses on firmware before 2026-09.",
            file=sys.stderr,
        )
    overflow = int(sync.get("cam_hold_overflow") or 0)
    leftover = int(sync.get("cam_hold_leftover") or 0)
    if overflow or leftover:
        print(
            f"WARNING: camera frames never paired with a trigger: {overflow} dropped from the "
            f"hold buffer, {leftover} left unpaired at the end",
            file=sys.stderr,
        )
    prev = sync.get("preview_jpeg") or {}
    if int(prev.get("dropped") or 0) or int(prev.get("write_errors") or 0):
        print(
            f"WARNING: preview JPEGs dropped={prev.get('dropped', 0)} "
            f"write_errors={prev.get('write_errors', 0)}",
            file=sys.stderr,
        )
    host = sync.get("host_capture") or {}
    drops = host.get("queue_drops") or {}
    if any(int(v or 0) > 0 for v in drops.values()):
        print(
            f"WARNING: host capture queues overflowed (raw_chunks={drops.get('raw_chunks', 0)}, "
            f"records={drops.get('records', 0)}) -- data was lost on the Pi, not the HAT.",
            file=sys.stderr,
        )
    for cid, anc in (sync.get("camera_anchor") or {}).items():
        if not isinstance(anc, dict):
            continue
        if anc.get("method") == "pps_time":
            print(
                f"camera {cid}: first frame anchored on trigger {anc.get('trigger_id')} by "
                f"PPS time (offset {anc.get('offset_us')} us, skipped "
                f"{anc.get('skipped_triggers')} start-up pulses)",
                file=sys.stderr,
            )
        else:
            print(
                f"camera {cid}: first frame anchored on the oldest buffered trigger "
                f"({anc.get('note', 'no PPS fit')}) -- frames may be labelled early",
                file=sys.stderr,
            )
    for cid, cam in (sync.get("cameras") or {}).items():
        ar = cam.get("assign_residual") if isinstance(cam, dict) else None
        if not ar:
            continue
        frames = ar.get("median_frames")
        frames_s = f" ({frames:+.2f} frames)" if frames is not None else ""
        spread = ar.get("spread_frames")
        spread_s = f", spread {spread:.2f} frames" if spread is not None else ""
        drift = ar.get("drift_us_per_s")
        drift_s = f", drift {drift:+.0f} us/s" if drift is not None else ""
        print(
            f"camera {cid}: assigned-trigger minus PPS-mapped SensorTimestamp: "
            f"median {ar['median_us'] / 1000.0:+.1f} ms{frames_s}{spread_s}{drift_s} "
            f"(n={ar['n']}, pps n={ar.get('pi_fit_n')})",
            file=sys.stderr,
        )


def _print_record_summary(counts: Counter[int]) -> None:
    if not counts:
        return
    parts: list[str] = []
    for rtype, n in sorted(counts.items()):
        name = RECORD_NAMES.get(rtype, f"0x{rtype:02X}")
        parts.append(f"{name}={n}")
    print(f"record_stats: {', '.join(parts)}", file=sys.stderr)
    emg = counts.get(int(RecordType.EMG_BLOCK), 0)
    mta = counts.get(int(RecordType.IMU_MTA_QUAT), 0)
    cra = counts.get(int(RecordType.IMU_CRA_QUAT), 0)
    trig = counts.get(int(RecordType.TRIGGER_EVENT), 0)
    print(
        f"record_summary: EMG_BLOCK={emg} IMU_MTA_QUAT={mta} IMU_CRA_QUAT={cra} TRIGGER_EVENT={trig}",
        file=sys.stderr,
    )
