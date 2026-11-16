# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""First-frame PPS anchor waits for a trigger that has fired but is not read yet.

The wait used to be capped at 4 periods past the newest buffered trigger: with
triggers 1-2 buffered and a first frame exposed at trigger 8, the frame fell
through to ``oldest_buffered`` and was labelled trigger 1.
"""

from __future__ import annotations

from jtsmm.sync.clock_align import ClockAligner

P = 33_333
T0 = 2_000_000
OFF_NS = 5_000_000_000  # host ns = master us * 1000 + OFF_NS
LATENCY_US = 12_500


def _aligner() -> ClockAligner:
    al = ClockAligner(pps_anchor=True, delta_tracking=True, anchor_refine_after=10**9)
    for k in range(6):
        tm = T0 - 500_000 + k * 50_000
        al.ingest_pps(tm * 1000 + OFF_NS, tm)
    al.configure_trigger(rate_hz=30, pulse_width_us=12_000)
    return al


def _host_ns(master_us: int) -> int:
    return master_us * 1000 + OFF_NS


def _sensor_ns(pulse: int) -> int:
    return _host_ns(T0 + pulse * P + LATENCY_US)


def test_frame_far_ahead_of_the_buffer_waits_for_its_trigger():
    al = _aligner()
    for i in (1, 2):
        al.note_trigger(i, T0 + i * P)
    # the frame arrived ~20 ms after its exposure: trigger 8 has fired by now
    al._host_now_ns = lambda: _host_ns(T0 + 8 * P + 20_000)  # noqa: SLF001
    assert al.ingest_cam("cam0", _sensor_ns(8), None) is None
    assert al.last_ingest_waiting
    assert "cam0" not in al.anchor_info  # not labelled trigger 1 (oldest_buffered)
    for i in range(3, 9):
        al.note_trigger(i, T0 + i * P)
    r = al.ingest_cam("cam0", _sensor_ns(8), None)
    assert r is not None and r.payload["trigger_id"] == 8
    assert al.anchor_info["cam0"]["method"] == "pps_time"


def test_prediction_in_the_future_does_not_wait():
    """A frame 'from' a trigger that cannot have fired yet means a bad fit: fall back."""
    al = _aligner()
    for i in (1, 2):
        al.note_trigger(i, T0 + i * P)
    al._host_now_ns = lambda: _host_ns(T0 + 2 * P + 5_000)  # noqa: SLF001
    r = al.ingest_cam("cam0", _sensor_ns(8), None)
    assert r is not None and not al.last_ingest_waiting
    assert al.anchor_info["cam0"]["method"] == "oldest_buffered"
