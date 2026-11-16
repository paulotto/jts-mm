# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Offline IMU stream-rate report from a recorded session.

Manual fallback for ``jtsmm doctor imu-rates`` (Phase 4): record a fixed window
per preset (``jtsmm record --duration 10``), then measure per-stream Hz from the
stored ``t_master_us_*`` vectors::

    python -m jtsmm.tools.imu_rate_report session.h5

Reports, per IMU slot and stream: the overall rate (samples / span), the rate
with silent holes excluded (a hole is a step longer than ``GAP_FACTOR`` median
periods -- e.g. take1's 26.9 s MTA dropout), the number and total length of
those holes, and the configured rate (fusion ``rate_hz`` for the quaternion,
the dataset's ``rate_hz`` attribute for motion streams).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jtsmm.io.session_reader import Session

GAP_FACTOR = 10.0


@dataclass
class StreamReport:
    slot: str
    stream: str
    n: int
    measured_hz: float
    configured_hz: float
    active_hz: float = 0.0  # rate with holes excluded
    n_gaps: int = 0
    gap_s: float = 0.0


def _hz(t_us: np.ndarray) -> float:
    if t_us is None or t_us.shape[0] < 2:
        return 0.0
    span_s = (float(t_us[-1]) - float(t_us[0])) / 1e6
    return (t_us.shape[0] - 1) / span_s if span_s > 0 else 0.0


def gap_stats(t_us: np.ndarray, gap_factor: float = GAP_FACTOR) -> tuple[float, int, float]:
    """``(rate_hz_without_gaps, n_gaps, total_gap_s)`` for one timestamp vector."""
    if t_us is None or t_us.shape[0] < 2:
        return 0.0, 0, 0.0
    dt = np.diff(np.asarray(t_us, dtype=np.float64))
    pos = dt[dt > 0]
    if pos.size == 0:
        return 0.0, 0, 0.0
    gap = dt > gap_factor * float(np.median(pos))
    active_s = float(dt[~gap].sum()) / 1e6
    rate = float((~gap).sum()) / active_s if active_s > 0 else 0.0
    return rate, int(gap.sum()), float(dt[gap].sum()) / 1e6


def _configured_fusion_hz(meta: dict[str, Any], slot: str) -> float:
    raw = meta.get("mcu_config_json")
    if not raw:
        return 0.0
    try:
        cfg = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
    except (ValueError, AttributeError):
        return 0.0
    blk = cfg.get(f"imu_{slot}") or {}
    fusion = blk.get("fusion") or {}
    rate = fusion.get("rate_hz")
    if not rate:
        rate = ((blk.get("reports") or {}).get(str(fusion.get("primary", ""))) or {}).get("rate_hz")
    return float(rate or 0.0)


def _report(slot: str, stream: str, t_us: np.ndarray, configured: float) -> StreamReport:
    active, n_gaps, gap_s = gap_stats(t_us)
    return StreamReport(slot, stream, int(t_us.shape[0]), _hz(t_us), configured,
                        active_hz=active, n_gaps=n_gaps, gap_s=gap_s)


def report_session(path: Path | str) -> list[StreamReport]:
    """Measure per-stream Hz for both IMU slots in a recorded session."""
    session = Session.open(Path(path))
    out: list[StreamReport] = []
    try:
        for slot, stream in (("mta", session.imu_mta), ("cra", session.imu_cra)):
            if stream is None:
                continue
            if getattr(stream, "quat", None) is not None and stream.n >= 2:
                out.append(_report(slot, "quat", stream.t_us, _configured_fusion_hz(session.meta, slot)))
            for name, vs in stream.motion_streams().items():
                out.append(_report(slot, name, vs.t_us, float(getattr(vs, "rate_hz", 0) or 0)))
    finally:
        session.close()
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m jtsmm.tools.imu_rate_report <session.h5>", file=sys.stderr)
        return 2
    reports = report_session(argv[0])
    if not reports:
        print("no IMU streams found in session")
        return 0
    print(
        f"{'slot':4}  {'stream':<16} {'n':>8} {'overall Hz':>11} {'w/o gaps Hz':>12} "
        f"{'gaps':>5} {'gap s':>7} {'configured Hz':>14}"
    )
    for r in reports:
        cfg = f"{r.configured_hz:.0f}" if r.configured_hz else "-"
        print(
            f"{r.slot:4}  {r.stream:<16} {r.n:>8} {r.measured_hz:>11.1f} {r.active_hz:>12.1f} "
            f"{r.n_gaps:>5} {r.gap_s:>7.2f} {cfg:>14}"
        )
    print(f"(gap = step > {GAP_FACTOR:g}x the median period; 'w/o gaps' excludes them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
