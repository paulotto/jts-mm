# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU sample-rate validation over J_USB1 (`jtsmm doctor imu-rates`).

Applies a report configuration to the HAT, captures for a fixed window, and
compares the **measured** per-stream rate (from ``t_master_us``) against the
**configured** rate, together with the firmware's IMU drop counters. The pure
functions (:func:`measured_hz`, :func:`analyze`, :func:`expected_rates`) carry
the logic and are unit-tested with synthetic records; :func:`run_case` drives
real hardware and is only exercised on a connected bench.

Bench setup (per the plan): HAT on **J_USB1** (USB bulk + CDC ``config apply``);
SWD used only for flashing. No debug-UART required.
"""

from __future__ import annotations

import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from jtsmm.protocol.types import RecordType

__all__ = [
    "StreamRate",
    "CaseResult",
    "measured_hz",
    "expected_rates",
    "analyze",
    "run_case",
    "SweepRow",
    "default_min_fusion_hz",
    "sweep_configs",
    "sweep_rates_for",
    "sweep_report_for",
    "summarize_sweep",
]

# USB record type -> (slot, kind). kind selects how the report name is derived.
SLOT_BY_RECORD: dict[int, tuple[str, str]] = {
    int(RecordType.IMU_MTA_QUAT): ("mta", "quat"),
    int(RecordType.IMU_MTA_RAW): ("mta", "raw"),
    int(RecordType.IMU_MTA_LIN_ACCEL): ("mta", "lin"),
    int(RecordType.IMU_CRA_QUAT): ("cra", "quat"),
    int(RecordType.IMU_CRA_RAW): ("cra", "raw"),
    int(RecordType.IMU_CRA_LIN_ACCEL): ("cra", "lin"),
}

# Decoded RAW key -> catalog report name.
RAW_KEY_TO_REPORT = {"accel": "accelerometer", "gyro": "gyroscope", "mag": "magnetometer"}

# STATS field names for per-slot IMU push/drop counters.
_DROP_FIELDS = {"mta": ("imu_mta_push", "imu_mta_drops"), "cra": ("imu_cra_push", "imu_cra_drops")}

RATE_TOL = 0.02        # ±2% of configured rate
MAX_DROP_RATE = 0.005  # < 0.5% dropped

# SH-2 report ids (imu_status prints granted intervals by id).
SH2_REPORT_IDS = {
    "accelerometer": 0x01, "gyroscope": 0x02, "magnetometer": 0x03,
    "linear_acceleration": 0x04, "rotation_vector": 0x05, "gravity": 0x06,
    "game_rotation_vector": 0x08, "geomagnetic_rotation_vector": 0x09,
    "raw_accelerometer": 0x14, "arvr_stabilized_rotation_vector": 0x28,
    "arvr_stabilized_game_rotation_vector": 0x29, "gyro_integrated_rotation_vector": 0x2A,
}
# imu_status counters whose growth means reports were lost before the ring.
LINK_COUNTERS = ("seq_gaps", "trunc_pkts", "cont_pkts", "i2c_tx_aborts", "i2c_timeouts")


@dataclass
class StreamRate:
    slot: str
    report: str
    n: int
    measured_hz: float
    expected_hz: float
    drop_rate: float
    ok: bool
    note: str = ""
    # 1 / median sample interval: the rate the BNO085 samples at. Well above
    # measured_hz means samples went missing (intervals at 2x, 3x the median).
    median_hz: float = 0.0
    # Rate the BNO085 granted in its Get Feature Response (None: not reported).
    granted_hz: float | None = None


@dataclass
class CaseResult:
    label: str
    duration_s: float
    streams: list[StreamRate] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    # slot -> growth of LINK_COUNTERS over the case (imu_status before/after).
    link: dict[str, dict[str, int]] = field(default_factory=dict)
    status_text: str = ""  # raw imu_status after the case (read times, I2C clock, ts)
    # slot -> SH-2 sequence gaps between the first and last STATS record of the
    # measurement window (excludes the stream start, where reports queued in
    # the BNO085 while nobody read them leave a one-off jump); {} on firmware
    # without the counter.
    window_gaps: dict[str, int] = field(default_factory=dict)
    window_span_s: float = 0.0
    note: str = ""

    @property
    def passed(self) -> bool:
        return not self.note and bool(self.streams) and all(s.ok for s in self.streams)


def measured_hz(timestamps_us: list[int]) -> float:
    """Rate from a monotonically-increasing ``t_master_us`` list: (N-1)/Δt."""
    if len(timestamps_us) < 2:
        return 0.0
    span_s = (int(timestamps_us[-1]) - int(timestamps_us[0])) / 1e6
    if span_s <= 0:
        return 0.0
    return (len(timestamps_us) - 1) / span_s


def median_interval_hz(timestamps_us: list[int]) -> float:
    """1 / median spacing: the sampling rate, robust to missing samples."""
    if len(timestamps_us) < 3:
        return 0.0
    dts = [b - a for a, b in zip(timestamps_us, timestamps_us[1:]) if b > a]
    return 1e6 / statistics.median(dts) if dts else 0.0


_STATUS_SLOT = re.compile(r"^imu_(mta|cra):")
_STATUS_COUNTER = re.compile(r"\b(" + "|".join(LINK_COUNTERS) + r")=(\d+)")
_STATUS_GRANT = re.compile(r"report 0x([0-9A-Fa-f]{2}) req=\d+Hz granted=(?:(\d+)us|none)")


def parse_imu_status(text: str) -> dict[str, dict[str, Any]]:
    """``imu_status`` console text -> per slot {counter: int, "granted": {id: µs|None}}."""
    out: dict[str, dict[str, Any]] = {}
    cur: dict[str, Any] | None = None
    for line in text.splitlines():
        m = _STATUS_SLOT.match(line.strip())
        if m:
            cur = out.setdefault(m.group(1), {"granted": {}})
            continue
        if cur is None:
            continue
        for name, val in _STATUS_COUNTER.findall(line):
            cur[name] = int(val)
        g = _STATUS_GRANT.search(line)
        if g:
            cur["granted"][int(g.group(1), 16)] = int(g.group(2)) if g.group(2) else None
    return out


def query_imu_status(ser: Any, *, timeout_s: float = 2.0) -> dict[str, dict[str, Any]]:
    """Send ``imu_status`` (streams stopped) and parse it; {} when there is no answer."""
    return parse_imu_status(query_imu_status_text(ser, timeout_s=timeout_s))


def query_imu_status_text(ser: Any, *, timeout_s: float = 2.0) -> str:
    """Raw ``imu_status`` console text ("" when there is no answer)."""
    try:
        ser.reset_input_buffer()
        ser.write(b"imu_status\r\n")
        ser.flush()
        buf = bytearray()
        deadline = time.monotonic() + timeout_s
        quiet_since = None
        while time.monotonic() < deadline:
            chunk = ser.read(1024)
            if chunk:
                buf.extend(chunk)
                quiet_since = None
            elif b"imu_cra:" in buf:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since > 0.3:
                    break
            else:
                time.sleep(0.01)
        return buf.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - a diagnostic must never fail the case
        return ""


def link_growth(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
                ) -> dict[str, dict[str, int]]:
    """Per slot: how much each LINK_COUNTER grew between two imu_status reads."""
    out: dict[str, dict[str, int]] = {}
    for slot, a in after.items():
        b = before.get(slot, {})
        out[slot] = {k: int(a[k]) - int(b.get(k, 0)) for k in LINK_COUNTERS if k in a}
    return out


def apply_granted(streams: list[StreamRate], status: dict[str, dict[str, Any]]) -> None:
    """Fill StreamRate.granted_hz from an imu_status read."""
    for st in streams:
        rid = SH2_REPORT_IDS.get(st.report)
        us = (status.get(st.slot) or {}).get("granted", {}).get(rid)
        if us:
            st.granted_hz = 1e6 / us


def report_name(kind: str, decoded: dict[str, Any], fallback_fusion: str | None) -> str | None:
    """Map a decoded record to its catalog report name (or None if unknown)."""
    if kind == "quat":
        return decoded.get("fusion") or fallback_fusion
    if kind == "lin":
        return "linear_acceleration"
    for key, name in RAW_KEY_TO_REPORT.items():
        if key in decoded:
            return name
    return None


def expected_rates(resolved: dict[str, Any]) -> dict[tuple[str, str], float]:
    """(slot, report) -> configured rate_hz for every enabled report in the config."""
    mcu = resolved.get("mcu") or {}
    out: dict[tuple[str, str], float] = {}
    for slot in ("mta", "cra"):
        reports = ((mcu.get(f"imu_{slot}") or {}).get("reports")) or {}
        for name, entry in reports.items():
            if isinstance(entry, dict) and entry.get("enabled"):
                out[(slot, name)] = float(entry.get("rate_hz", 0) or 0)
    return out


def _drop_rate(stats: dict[str, Any], slot: str) -> float:
    push_f, drop_f = _DROP_FIELDS[slot]
    push = int(stats.get(push_f, 0) or 0)
    drops = int(stats.get(drop_f, 0) or 0)
    total = push + drops
    return (drops / total) if total > 0 else 0.0


def analyze(
    timestamps: dict[tuple[str, str], list[int]],
    expected: dict[tuple[str, str], float],
    stats: dict[str, Any] | None = None,
    *,
    rate_tol: float = RATE_TOL,
    max_drop_rate: float = MAX_DROP_RATE,
) -> list[StreamRate]:
    """Compare measured vs configured rates + drop budget for every stream."""
    stats = stats or {}
    results: list[StreamRate] = []
    for key in sorted(set(expected) | set(timestamps)):
        slot, report = key
        ts = timestamps.get(key, [])
        n = len(ts)
        exp = expected.get(key, 0.0)
        meas = measured_hz(ts)
        drop = _drop_rate(stats, slot)
        note = ""
        if exp <= 0:
            ok = False
            note = "unexpected stream (not in config)"
        elif n < 2:
            ok = False
            note = "no data captured"
        else:
            within = abs(meas - exp) <= rate_tol * exp
            drops_ok = drop < max_drop_rate
            ok = within and drops_ok
            if not within:
                note = f"rate off by {100 * (meas - exp) / exp:+.1f}%"
            elif not drops_ok:
                note = f"drops {100 * drop:.2f}% ≥ {100 * max_drop_rate:.2f}%"
        results.append(StreamRate(slot, report, n, meas, exp, drop, ok, note,
                                  median_hz=median_interval_hz(ts)))
    return results


def measure_capture(
    cap: Any, duration_s: float, stop_event: Any = None, window: dict[str, Any] | None = None
) -> tuple[dict[tuple[str, str], list[int]], dict[str, Any]]:
    """Drain native records for ``duration_s`` collecting per-stream timestamps.

    ``window`` (optional) receives the first STATS record and the master times
    of the first and last one, for window-only counter deltas.
    """
    timestamps: dict[tuple[str, str], list[int]] = defaultdict(list)
    last_stats: dict[str, Any] = {}
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        if stop_event is not None and stop_event.is_set():
            break
        recs = cap.drain_records()
        if not recs:
            time.sleep(0.002)
            continue
        for rec in recs:
            rt = int(rec.record_type)
            if rt in SLOT_BY_RECORD:
                slot, kind = SLOT_BY_RECORD[rt]
                name = report_name(kind, rec.decoded, None)
                # Retain untagged (legacy) quats under (slot, None) so
                # _reconcile_fusion can attribute them to the sole fusion; RAW/LIN
                # are always identifiable, so drop only unknown RAW sensor ids.
                if name is not None or kind == "quat":
                    timestamps[(slot, name)].append(int(rec.t_master_us))
            elif rt == int(RecordType.STATS):
                last_stats = dict(rec.decoded)
                if window is not None:
                    window.setdefault("first", last_stats)
                    window.setdefault("t_first_us", int(rec.t_master_us))
                    window["t_last_us"] = int(rec.t_master_us)
    return dict(timestamps), last_stats


def window_seq_gaps(window: dict[str, Any], last_stats: dict[str, Any]
                    ) -> tuple[dict[str, int], float]:
    """(slot -> SH-2 seq gaps, span s) between the window's first and last STATS."""
    first = window.get("first") or {}
    span_s = (int(window.get("t_last_us", 0)) - int(window.get("t_first_us", 0))) / 1e6
    out: dict[str, int] = {}
    for slot in ("mta", "cra"):
        key = f"imu_{slot}_seq_gaps"
        if key in first and key in last_stats:
            out[slot] = int(last_stats[key]) - int(first[key])
    return (out, span_s) if span_s > 0 else ({}, 0.0)


def run_case(
    *,
    port: str,
    resolved: dict[str, Any],
    duration_s: float = 10.0,
    settle_ms: int = 400,
    label: str = "",
    emg: bool = True,
) -> CaseResult:
    """Apply ``resolved`` over J_USB1, capture, and analyze (hardware path)."""
    from jtsmm import native
    from jtsmm.capture.native_backend import NativeCapture
    from jtsmm.capture.serial_util import find_hat_port
    from jtsmm.config import apply_mcu, open_hat_serial, start_mcu_streams, stop_mcu_streams

    if not native.use_native():
        raise RuntimeError("imu-rates requires the native capture extension.")

    serial_port = find_hat_port(port or (resolved.get("hat") or {}).get("serial_port"))
    ser = open_hat_serial(str(serial_port))
    try:
        # A fresh apply immediately after the previous case's stream teardown can
        # miss its completion line; retry once after a brief settle before failing.
        applied = apply_mcu(ser, resolved, persist=False)
        if not applied:
            time.sleep(0.5)
            applied = apply_mcu(ser, resolved, persist=False)
        if not applied:
            return CaseResult(
                label=label or "case",
                duration_s=duration_s,
                note="config apply FAILED (firmware kept its previous config)",
            )
        status_before = query_imu_status(ser)
        # Open the USB reader before the streams start: libusb takes ~1.5 s to
        # come up, and EMG produced meanwhile overflowed the MCU ring (~1000
        # emg_drops at 2 kHz on every start, measured on the HAT).
        cap = NativeCapture(port=str(serial_port))
        cap.start()
        try:
            start_mcu_streams(ser, emg=emg)
        except BaseException:
            cap.stop()
            raise
    finally:
        ser.close()

    try:
        cap.settle(quiet_ms=settle_ms)
        window: dict[str, Any] = {}
        timestamps, stats = measure_capture(cap, duration_s, window=window)
    finally:
        cap.stop()
        status_text = ""
        try:
            with open_hat_serial(str(serial_port)) as ser2:
                stop_mcu_streams(ser2)
                status_text = query_imu_status_text(ser2)
        except Exception:
            pass
    status_after = parse_imu_status(status_text)

    expected = _reconcile_fusion(timestamps, expected_rates(resolved))
    streams = analyze(timestamps, expected, stats)
    window_gaps, window_span_s = window_seq_gaps(window, stats)
    apply_granted(streams, status_after)  # after: also holds unsolicited updates
    return CaseResult(
        label=label or "case",
        duration_s=duration_s,
        streams=streams,
        stats=stats,
        # Includes the settle and stop time around the window (~1 s).
        link=link_growth(status_before, status_after),
        status_text=status_text,
        window_gaps=window_gaps,
        window_span_s=window_span_s,
    )


def _reconcile_fusion(
    timestamps: dict[tuple[str, str], list[int]], expected: dict[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    """Attribute legacy untagged quats to the single configured fusion per slot."""
    for slot in ("mta", "cra"):
        fusions = [r for (s, r) in expected if s == slot and _is_fusion(r)]
        legacy = timestamps.get((slot, None))  # report_name returned None
        if legacy and len(fusions) == 1:
            timestamps[(slot, fusions[0])] = timestamps.pop((slot, None))
    return expected


def _is_fusion(report: str) -> bool:
    from jtsmm.config.catalog import FUSION_REPORTS

    return report in FUSION_REPORTS


# --------------------------------------------------------------------------- #
# Rate sweep (`jtsmm doctor imu-rates --sweep`)
# --------------------------------------------------------------------------- #
# The BNO085 fits each request into its own schedule around the fusion report,
# so the delivered rate is not monotonic in the request (linear accel @200 gave
# ~120 Hz, @250 ~220 Hz). A sweep requests one report at several rates, keeps
# the rest of the config, and measures what each slot delivers.

SWEEP_DEFAULT_RATES = (100, 125, 200, 250, 333, 400, 500)
# Default floor for the fusion rate a request may leave, as a fraction of the
# configured fusion rate (390 -> 331.5 Hz: ~350 measured next to a 250 Hz vector
# passes, the ~331 Hz of the throttled three-report case does not).
SWEEP_MIN_FUSION_FRACTION = 0.85


def _sweep_slots(slot: str) -> tuple[str, ...]:
    return ("mta", "cra") if slot == "both" else (slot,)


def _enabled_reports(resolved: dict[str, Any], slot: str) -> dict[str, dict[str, Any]]:
    block = (resolved.get("mcu") or {}).get(f"imu_{slot}") or {}
    if block.get("enabled") is False:
        return {}
    return {n: c for n, c in (block.get("reports") or {}).items()
            if isinstance(c, dict) and c.get("enabled", True)}


def sweep_report_for(resolved: dict[str, Any], slot: str = "both") -> str:
    """The one non-fusion report enabled in the swept slot(s), e.g. accelerometer."""
    names = {n for s in _sweep_slots(slot) for n in _enabled_reports(resolved, s)
             if not _is_fusion(n)}
    if len(names) != 1:
        raise ValueError(
            f"--sweep needs exactly one enabled non-fusion report to sweep, found "
            f"{sorted(names) or 'none'}; name it with --sweep-report")
    return names.pop()


def sweep_rates_for(report: str, rates: list[float] | None = None) -> list[float]:
    """Requested rates, limited to the report's catalog maximum."""
    from jtsmm.config.catalog import BNO085_REPORT_CATALOG

    spec = BNO085_REPORT_CATALOG.get(report)
    if spec is None:
        raise ValueError(f"unknown report {report!r}")
    wanted = rates if rates else SWEEP_DEFAULT_RATES
    out = sorted({float(r) for r in wanted if 0 < float(r) <= spec.max_rate_hz})
    if not out:
        raise ValueError(f"no sweep rate within 1..{spec.max_rate_hz} Hz for {report}")
    return out


def default_min_fusion_hz(resolved: dict[str, Any], slot: str = "both") -> float | None:
    """SWEEP_MIN_FUSION_FRACTION of the lowest configured fusion rate (None: no fusion)."""
    rates = [float(c.get("rate_hz", 0) or 0) for s in _sweep_slots(slot)
             for n, c in _enabled_reports(resolved, s).items() if _is_fusion(n)]
    return SWEEP_MIN_FUSION_FRACTION * min(rates) if rates else None


def sweep_configs(
    resolved: dict[str, Any], report: str, rates: list[float], slot: str = "both"
) -> list[tuple[str, dict[str, Any]]]:
    """(label, config) per rate: ``report`` enabled at that rate in the swept slots."""
    import copy

    out = []
    for rate in rates:
        doc = copy.deepcopy(resolved)
        for s in _sweep_slots(slot):
            block = doc.setdefault("mcu", {}).setdefault(f"imu_{s}", {})
            entry = block.setdefault("reports", {}).setdefault(report, {})
            entry["enabled"] = True
            entry["rate_hz"] = int(round(rate))
            fusion = block.get("fusion") or {}
            if _is_fusion(report) and fusion:
                fusion["rate_hz"] = int(round(rate))  # validate: must agree with the report
        out.append((f"{report}@{int(round(rate))}", doc))
    return out


@dataclass
class SweepRow:
    requested_hz: float
    fusion_hz: dict[str, float]   # slot -> measured fusion rate (0 = none)
    swept_hz: dict[str, float]    # slot -> measured rate of the swept report
    max_drop_rate: float          # USB ring drops (firmware -> host)
    complete: bool                # applied, and every expected stream delivered data
    # slot -> 1 / median interval of the swept report (its sampling rate)
    swept_median_hz: dict[str, float] = field(default_factory=dict)
    # slot -> rate the BNO085 granted for the swept report (None: not reported)
    swept_granted_hz: dict[str, float | None] = field(default_factory=dict)
    # Worst slot: reports lost between BNO085 and ring (SH-2 seq gaps / all reports).
    loss_frac: float = 0.0
    best: bool = False


# Requests whose delivered rate is within this of the fastest count as a tie;
# the tie goes to the request closest to what is delivered, so the configured
# rate describes the data (the integrity check compares the two).
SWEEP_TIE_FRACTION = 0.02


def summarize_sweep(
    results: list[CaseResult],
    report: str,
    rates: list[float],
    slot: str = "both",
    min_fusion_hz: float | None = None,
) -> list[SweepRow]:
    """One row per request; ``best`` marks the highest swept rate on the slower
    slot among complete requests without drops or losses whose fusion rate
    (slower slot) stays at or above ``min_fusion_hz`` (None: no fusion
    condition). Near-ties (SWEEP_TIE_FRACTION) go to the request closest to
    the delivered rate."""
    slots = _sweep_slots(slot)
    rows: list[SweepRow] = []
    for rate, res in zip(rates, results):
        fusion = {s: 0.0 for s in slots}
        swept = {s: 0.0 for s in slots}
        med: dict[str, float] = {s: 0.0 for s in slots}
        granted: dict[str, float | None] = {s: None for s in slots}
        n_slot = {s: 0 for s in slots}
        drop = 0.0
        for st in res.streams:
            if st.slot not in slots:
                continue
            drop = max(drop, st.drop_rate)
            n_slot[st.slot] += st.n
            if st.report == report:
                swept[st.slot] = st.measured_hz
                med[st.slot] = st.median_hz
                granted[st.slot] = st.granted_hz
            elif _is_fusion(st.report):
                fusion[st.slot] = st.measured_hz
        loss = 0.0
        for s in slots:
            if s in res.window_gaps and res.window_span_s > 0:
                # Reports the slot delivered within the STATS span.
                gaps = res.window_gaps[s]
                n_span = n_slot[s] * min(1.0, res.window_span_s / res.duration_s)
            else:  # older firmware: imu_status growth, stream start included
                gaps = int((res.link.get(s) or {}).get("seq_gaps", 0))
                n_span = n_slot[s]
            if gaps > 0:
                loss = max(loss, gaps / (n_span + gaps))
        complete = (not res.note and bool(res.streams)
                    and all(st.n >= 2 for st in res.streams if st.slot in slots))
        rows.append(SweepRow(rate, fusion, swept, drop, complete, med, granted, loss))

    keep = [r for r in rows if r.complete and r.max_drop_rate < MAX_DROP_RATE
            and r.loss_frac < MAX_DROP_RATE
            and (min_fusion_hz is None or min(r.fusion_hz.values()) >= min_fusion_hz)]
    if keep:
        top = max(min(r.swept_hz.values()) for r in keep)
        ties = [r for r in keep if min(r.swept_hz.values()) >= (1 - SWEEP_TIE_FRACTION) * top]

        def mismatch(r: SweepRow) -> float:
            return abs(min(r.swept_hz.values()) - r.requested_hz) / r.requested_hz

        min(ties, key=lambda r: (mismatch(r), r.requested_hz)).best = True
    return rows
