# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cadence tracking vs. frames that arrive before their TRIGGER_EVENT.

The supervisor keeps such frames in a shared FIFO (``cam_hold``) and retries the
head on every flush until ``ingest_cam`` stops reporting ``last_ingest_waiting``.
The cadence reference used to advance on every one of those retries, so a single
trigger event arriving 20 ms after its frame stalled the camera for the rest of
the take (251 of 300 frames stuck) and, through the shared FIFO, the other
camera too. This drives the aligner exactly the way the supervisor does.
"""

from __future__ import annotations

from collections import deque

from jtsmm.sync.clock_align import ClockAligner

P_US = 33_333
T0_US = 2_000_000
OFF_NS = 5_000_000_000  # SensorTimestamp domain offset (camera clock != MCU clock)


def _simulate(
    late: dict[int, int],
    *,
    cams: tuple[str, ...] = ("cam0",),
    flush_ms: float = 1.0,
    n: int = 300,
    frame_latency_us: int = 12_500,
) -> dict[str, dict[str, int]]:
    """Return per-camera {labelled, wrong, dropped, stuck}.

    ``late`` maps trigger id -> delivery delay (µs after the edge); every other
    TRIGGER_EVENT arrives 1 ms after its edge. Frames are delivered
    ``frame_latency_us`` after their trigger, so a late event arrives after its
    frame and the frame must wait in the hold queue.
    """
    al = ClockAligner(pps_anchor=True, delta_tracking=True, triggers_max=10_000)
    for k in range(6):
        tm = T0_US + k * 50_000
        al.ingest_pps(tm * 1000 + OFF_NS, tm)
    events: list[tuple[int, int, int, str]] = []  # (host_us, kind, id, cam)
    for i in range(1, n + 1):
        tt = T0_US + i * P_US
        events.append((tt + late.get(i, 1_000), 0, i, ""))
        for j, cam in enumerate(cams):
            events.append((tt + frame_latency_us + 100 * j, 1, i, cam))
    events.sort()
    hold: deque[tuple[str, int]] = deque()
    labels: dict[str, dict[int, int | None]] = {c: {} for c in cams}
    t = events[0][0]
    j = 0
    end = events[-1][0] + 200_000
    while t < end:
        while j < len(events) and events[j][0] <= t:
            _, kind, i, cam = events[j]
            j += 1
            if kind == 0:
                al.note_trigger(i, T0_US + i * P_US, stamp_source="pio_irq")
            else:
                hold.append((cam, i))
        while hold:  # mirrors supervisor._flush_cam_hold
            cam, fi = hold[0]
            r = al.ingest_cam(cam, (T0_US + fi * P_US) * 1000 + OFF_NS, None, frame_idx=fi)
            if r is None and al.last_ingest_waiting:
                break
            hold.popleft()
            labels[cam][fi] = None if r is None else int(r.payload["trigger_id"])
        t += int(flush_ms * 1000)
    out: dict[str, dict[str, int]] = {}
    for cam in cams:
        lab = labels[cam]
        out[cam] = {
            "labelled": len(lab),
            "wrong": sum(1 for f, v in lab.items() if v is not None and v != f),
            "dropped": sum(1 for v in lab.values() if v is None),
            "stuck": sum(1 for c, _ in hold if c == cam),
        }
    return out


def _all_good(res: dict[str, dict[str, int]], n: int = 300) -> bool:
    return all(r == {"labelled": n, "wrong": 0, "dropped": 0, "stuck": 0} for r in res.values())


def test_no_late_trigger_baseline():
    assert _all_good(_simulate({}))


def test_single_trigger_20ms_late_retried_every_1ms_flush():
    res = _simulate({50: 20_000 + 12_500}, flush_ms=1.0)
    assert _all_good(res), res


def test_single_late_trigger_with_slower_flush():
    assert _all_good(_simulate({50: 32_500}, flush_ms=5.0))


def test_several_late_triggers():
    late = {i: 12_500 + 15_000 for i in (40, 90, 140, 190, 240)}
    res = _simulate(late)
    assert _all_good(res), res


def test_two_cameras_share_the_hold_queue():
    """A stalled camera would block the other one through the shared FIFO."""
    late = {i: 12_500 + 20_000 for i in (30, 31, 150)}
    res = _simulate(late, cams=("cam0", "cam1"))
    assert _all_good(res), res


def test_waiting_does_not_move_the_cadence_reference():
    al = ClockAligner(pps_anchor=False, delta_tracking=True)
    for i in range(1, 6):
        al.note_trigger(i, T0_US + i * P_US)
    for i in range(1, 6):
        r = al.ingest_cam("cam0", (T0_US + i * P_US) * 1000, None)
        assert r is not None
    ref = al._cam_cadence_ref["cam0"]  # noqa: SLF001
    for _ in range(50):  # frame 6 is ahead of its trigger; retried many times
        assert al.ingest_cam("cam0", (T0_US + 6 * P_US) * 1000, None) is None
        assert al.last_ingest_waiting
    assert al._cam_cadence_ref["cam0"] == ref  # noqa: SLF001
    al.note_trigger(6, T0_US + 6 * P_US)
    r = al.ingest_cam("cam0", (T0_US + 6 * P_US) * 1000, None)
    assert r is not None and r.payload["trigger_id"] == 6
