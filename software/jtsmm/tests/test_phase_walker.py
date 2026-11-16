# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for experiment phase walker + master-clock stamping."""

from __future__ import annotations

from types import SimpleNamespace

from jtsmm.capture.phase_walker import PhaseWalker, TimedPhase, optical_frame_intervals
from jtsmm.sync.clock_align import ClockAligner


def _phases(*names: str):
    return [SimpleNamespace(name=n, kind="experiment") for n in names]


def test_phase_walker_space_start_end():
    stamps = iter(
        [
            (1_000_000_000, 1000, "pi_fit"),
            (1_100_000_000, 1100, "pi_fit"),
            (1_200_000_000, 1200, "pi_fit"),
            (1_300_000_000, 1300, "pi_fit"),
        ]
    )

    def stamp_fn():
        return next(stamps)

    walker = PhaseWalker(phases=_phases("a", "b"), stamp_fn=stamp_fn)
    assert walker.poll(None) == []
    ev = walker.poll(ord(" "))
    assert len(ev) == 1 and ev[0].edge == "start" and ev[0].name == "a"
    assert walker.active
    ev = walker.poll(ord(" "))
    assert len(ev) == 1 and ev[0].edge == "end"
    assert walker.recorded[0].t_start_us == 1000
    assert walker.recorded[0].t_end_us == 1100
    assert not walker.active
    walker.poll(ord(" "))
    walker.poll(ord(" "))
    assert walker.finished
    assert len(walker.recorded) == 2


def test_phase_walker_q_finishes_early():
    n = {"i": 0}

    def stamp_fn():
        n["i"] += 1
        return (n["i"] * 1_000_000_000, n["i"] * 1000, "mock")

    walker = PhaseWalker(phases=_phases("a", "b", "c"), stamp_fn=stamp_fn)
    walker.poll(ord(" "))
    ev = walker.poll(ord("q"))
    assert walker.finished and walker.aborted
    assert len(ev) == 1 and ev[0].edge == "end"
    assert len(walker.recorded) == 1


def test_map_pi_ns_fallbacks():
    aligner = ClockAligner()
    t, q = aligner.map_pi_ns(1_000_000_000)
    assert q == "unavailable" and t is None

    # Record-arrival pairs are a fallback, never labelled as a PPS fit.
    aligner.note_host_mcu_pair(1_000_000_000, 50_000)
    t, q = aligner.map_pi_ns(1_000_000_000)
    assert q == "mcu_arrival"
    assert t == 50_000

    aligner.note_host_mcu_pair(1_001_000_000, 51_000)
    t2, q2 = aligner.map_pi_ns(1_002_000_000)
    assert q2 == "mcu_arrival"
    assert abs(t2 - 52_000) < 50  # ~1 ms host → ~1 ms master


def test_optical_frame_intervals():
    timed = [
        TimedPhase("p", "experiment", 100, 300, "pi_fit", "pi_fit"),
        TimedPhase("q", "experiment", 350, 650, "pi_fit", "pi_fit"),
    ]
    # 100 us trigger period; the frame at 400 was dropped (absent row).
    t_opt = [0, 100, 200, 300, 500, 600]
    out = optical_frame_intervals(timed, t_opt)
    assert out[0] == ("p", "experiment", 1, 2)  # inclusive: frames 1..2
    # Trigger-grid frames 4..6 (400, 500, 600), not rows 4..5.
    assert out[1] == ("q", "experiment", 4, 6)


def test_optical_frame_intervals_skips_mvc():
    timed = [
        TimedPhase("mvc_opening", "mvc", 0, 80, "pi_fit", "pi_fit"),
        TimedPhase("p", "experiment", 100, 300, "pi_fit", "pi_fit"),
    ]
    t_opt = [0, 100, 200, 300]
    out = optical_frame_intervals(timed, t_opt)
    assert out == [("p", "experiment", 1, 2)]


def test_phase_walker_mvc_status_emg_only():
    def stamp_fn():
        return (1_000_000_000, 1000, "mock")

    walker = PhaseWalker(
        phases=[SimpleNamespace(name="mvc_opening", kind="mvc")],
        stamp_fn=stamp_fn,
    )
    assert "EMG only" in walker.status_line()
    assert walker.is_emg_only_slot()
    assert not walker.is_emg_only_active()
    walker.poll(ord(" "))
    assert walker.is_emg_only_active()
    assert walker.is_emg_only_slot()


def test_phase_walker_status_names_landmark_tooth():
    walker = PhaseWalker(
        phases=[
            SimpleNamespace(
                name="mand_point_1",
                kind="calibration_mandibular",
                hint="lower central incisor (incisal edge, midline)",
            ),
            SimpleNamespace(name="open_close", kind="experiment"),
        ],
        stamp_fn=lambda: (1_000_000_000, 1000, "mock"),
    )
    assert "→ lower central incisor" in walker.status_line()
    walker.poll(ord(" "))
    walker.poll(ord(" "))
    assert "→" not in walker.status_line()
