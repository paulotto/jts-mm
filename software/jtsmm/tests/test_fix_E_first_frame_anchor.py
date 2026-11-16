# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""First-frame PPS anchor: sensor latency, short trigger history, trim race."""

from __future__ import annotations

from jtsmm.sync.clock_align import ClockAligner

P = 33_333
T0 = 2_000_000
OFF_NS = 5_000_000_000
LATENCY_US = 12_500  # SensorTimestamp - edge at 30 Hz with a 12 ms pulse


def _aligner(**kw) -> ClockAligner:
    al = ClockAligner(pps_anchor=True, delta_tracking=True, anchor_refine_after=10**9, **kw)
    for k in range(6):
        tm = T0 - 500_000 + k * 50_000
        al.ingest_pps(tm * 1000 + OFF_NS, tm)
    return al


def _sensor_ns(pulse: int, *, extra_us: int = 0) -> int:
    return (T0 + pulse * P + LATENCY_US + extra_us) * 1000 + OFF_NS


def test_latency_is_subtracted_before_picking_the_nearest_trigger():
    """+5 ms of PPS/timestamp error on top of 12.5 ms latency crosses P/2."""
    for configured, expect in ((False, 11), (True, 10)):
        al = _aligner()
        if configured:
            al.configure_from_mcu({"trigger": {"rate_hz": 30, "pulse": {"width_us": 12000}}})
        for i in range(1, 30):
            al.note_trigger(i, T0 + i * P)
        r = al.ingest_cam("cam0", _sensor_ns(10, extra_us=5_000), None)
        assert r is not None
        assert r.payload["trigger_id"] == expect
    assert al.anchor_info["cam0"]["latency_us"] == 12_500.0


def test_anchor_works_with_only_two_buffered_triggers():
    """The median period needs >= 3 triggers; fall back to the nominal rate."""
    al = _aligner()
    al.configure_trigger(rate_hz=30, pulse_width_us=12_000)
    al.note_trigger(1, T0 + P)
    r = al.ingest_cam("cam0", _sensor_ns(1), None)
    assert r is not None and r.payload["trigger_id"] == 1
    assert al.anchor_info["cam0"]["method"] == "pps_time"


def test_frame_ahead_of_its_trigger_waits_instead_of_falling_back():
    al = _aligner()
    al.configure_trigger(rate_hz=30, pulse_width_us=12_000)
    for i in range(1, 6):
        al.note_trigger(i, T0 + i * P)
    assert al.ingest_cam("cam0", _sensor_ns(6), None) is None
    assert al.last_ingest_waiting
    assert "cam0" not in al.anchor_info
    al.note_trigger(6, T0 + 6 * P)
    r = al.ingest_cam("cam0", _sensor_ns(6), None)
    assert r is not None and r.payload["trigger_id"] == 6
    assert al.anchor_info["cam0"]["method"] == "pps_time"


def test_late_camera_still_finds_its_trigger_when_expected():
    """cam0 consumes 40 frames before cam1's first frame (pulse 5) arrives."""
    for expected, ok in (((), False), (("cam0", "cam1"), True)):
        al = _aligner(expected_cameras=expected)
        al.configure_trigger(rate_hz=30, pulse_width_us=12_000)
        for i in range(1, 60):
            al.note_trigger(i, T0 + i * P)
        for i in range(1, 45):
            assert al.ingest_cam("cam0", _sensor_ns(i), None) is not None
        r = al.ingest_cam("cam1", _sensor_ns(5), None)
        good = r is not None and r.payload["trigger_id"] == 5
        assert good is ok, (expected, r, al.anchor_info.get("cam1"))
    # once both have cursors trimming resumes
    al.ingest_cam("cam1", _sensor_ns(6), None)
    assert al._triggers[0][0] >= 6  # noqa: SLF001


def test_absent_expected_camera_does_not_pin_the_buffer_forever():
    al = _aligner(expected_cameras=("cam0", "cam1"), trim_wait_max_triggers=100)
    for i in range(1, 400):
        al.note_trigger(i, T0 + i * P)
        al.ingest_cam("cam0", _sensor_ns(i), None)
    assert len(al._triggers) <= 101  # noqa: SLF001
