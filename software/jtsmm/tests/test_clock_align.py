# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Affine clock alignment and Track B sequence association."""

from __future__ import annotations

import struct

import pytest

from jtsmm.protocol import McuRecord, RecordType
from jtsmm.sync.clock_align import ClockAligner

from tests.helpers import build_frame


def _trigger_rec(trig_id: int, t_us: int) -> McuRecord:
    payload = b"\x00\x00\x00\x00" + struct.pack("<Q", trig_id)
    frame = build_frame(int(RecordType.TRIGGER_EVENT), t_us, payload)
    from jtsmm.protocol import FrameParser

    return FrameParser().feed(frame)[0]


def _feed_triggers(aligner: ClockAligner, n: int, *, start: int = 1, period_us: int = 16_666) -> None:
    for i in range(start, start + n):
        for rec in aligner.ingest_mcu(_trigger_rec(i, i * period_us)):
            assert rec.modality == "trigger"


def test_trigger_ingest_and_camera_align():
    aligner = ClockAligner()
    for i in range(1, 6):
        for rec in aligner.ingest_mcu(_trigger_rec(i, i * 16_666)):
            assert rec.modality == "trigger"

    cam = aligner.ingest_cam("cam0", sensor_ts_ns=3 * 16_666 * 1000, trigger_id=3)
    assert cam is not None
    assert cam.t_master_us == 3 * 16_666
    assert cam.payload["trigger_id"] == 3


def test_affine_fit_improves_with_samples():
    aligner = ClockAligner(window=10)
    for i in range(20):
        t_local = float(i * 1000)
        t_master = t_local + 5.0
        aligner.ingest_pps(int(t_local * 1000), int(t_master))
    report = aligner.sync_report()
    assert abs(report["pi_to_master"]["a"] - 1.0) < 0.01
    assert abs(report["pi_to_master"]["b"] - 5.0) < 0.5


def test_affine_fit_stable_with_epoch_scale_timestamps():
    """SensorTimestamp is epoch-ns; uncentered lstsq used to report ~1e5 µs RMS."""
    aligner = ClockAligner(window=30)
    t0_local_us = 1.784532021847e15  # sensor_ns / 1000
    t0_master_us = 689_859_247.0
    for i in range(30):
        t_local_us = t0_local_us + i * 16_666.0
        t_master_us = t0_master_us + i * 16_666.0
        aligner.ingest_pps(int(t_local_us * 1000.0), int(t_master_us))
    fit = aligner.sync_report()["pi_to_master"]
    assert abs(fit["a"] - 1.0) < 1e-6
    assert fit["residual_rms_us"] < 1.0


def test_sequence_consume_ignores_newer_buffered_triggers():
    """Regression: do not attach the latest trigger when MCU is ahead of frames."""
    aligner = ClockAligner()
    _feed_triggers(aligner, 5)  # ids 1..5 buffered before any frame

    cam = aligner.ingest_cam("cam0", sensor_ts_ns=1_000_000, trigger_id=None, frame_idx=0)
    assert cam is not None
    assert cam.payload["trigger_id"] == 1
    assert cam.t_master_us == 1 * 16_666

    cam2 = aligner.ingest_cam("cam0", sensor_ts_ns=2_000_000, trigger_id=None, frame_idx=1)
    assert cam2 is not None
    assert cam2.payload["trigger_id"] == 2
    assert cam2.t_master_us == 2 * 16_666


def test_sequence_payload_trigger_id_never_none():
    aligner = ClockAligner()
    _feed_triggers(aligner, 3)
    for i in range(3):
        cam = aligner.ingest_cam("cam0", sensor_ts_ns=(i + 1) * 1_000_000, trigger_id=None, frame_idx=i)
        assert cam is not None
        assert cam.payload["trigger_id"] is not None
        assert isinstance(cam.payload["trigger_id"], int)


def test_two_cameras_independent_cursors_share_buffer():
    aligner = ClockAligner()
    _feed_triggers(aligner, 4)

    a0 = aligner.ingest_cam("cam0", sensor_ts_ns=100, trigger_id=None, frame_idx=0)
    b0 = aligner.ingest_cam("cam1", sensor_ts_ns=100, trigger_id=None, frame_idx=0)
    assert a0 is not None and b0 is not None
    assert a0.payload["trigger_id"] == 1
    assert b0.payload["trigger_id"] == 1
    assert a0.t_master_us == b0.t_master_us == 16_666

    a1 = aligner.ingest_cam("cam0", sensor_ts_ns=200, trigger_id=None, frame_idx=1)
    assert a1 is not None and a1.payload["trigger_id"] == 2

    b1 = aligner.ingest_cam("cam1", sensor_ts_ns=200, trigger_id=None, frame_idx=1)
    assert b1 is not None and b1.payload["trigger_id"] == 2

    # After both cameras advance, leading entries below min cursor are trimmed
    a2 = aligner.ingest_cam("cam0", sensor_ts_ns=300, trigger_id=None, frame_idx=2)
    b2 = aligner.ingest_cam("cam1", sensor_ts_ns=300, trigger_id=None, frame_idx=2)
    assert a2 is not None and b2 is not None
    assert a2.payload["trigger_id"] == b2.payload["trigger_id"] == 3
    assert all(tid >= 3 for tid, _ in aligner._triggers)


def test_slip_jumps_forward_when_expected_id_missing():
    aligner = ClockAligner()
    # Buffer ids 1,2,4 (skip 3) — simulate MCU ahead / missed USB event for id 3
    for i in (1, 2, 4):
        aligner.ingest_mcu(_trigger_rec(i, i * 16_666))

    assert aligner.ingest_cam("cam0", sensor_ts_ns=1, trigger_id=None).payload["trigger_id"] == 1
    assert aligner.ingest_cam("cam0", sensor_ts_ns=2, trigger_id=None).payload["trigger_id"] == 2

    slipped = aligner.ingest_cam("cam0", sensor_ts_ns=3, trigger_id=None, frame_idx=2)
    assert slipped is not None
    assert slipped.payload["trigger_id"] == 4
    assert slipped.t_master_us == 4 * 16_666
    # cursor was 3; first id >= 3 is 4; slip_count += (4 - 3) = 1
    assert aligner.slip_count["cam0"] == 1
    report = aligner.sync_report()
    assert report["slip_count"]["cam0"] == 1


def test_trigger_not_yet_arrived_waits_without_advancing():
    aligner = ClockAligner()
    aligner.ingest_mcu(_trigger_rec(1, 16_666))
    assert aligner.ingest_cam("cam0", sensor_ts_ns=1, trigger_id=None).payload["trigger_id"] == 1

    # Next expected is 2, but only id 1 was ever buffered — wait (not a drop).
    waiting = aligner.ingest_cam("cam0", sensor_ts_ns=2, trigger_id=None, frame_idx=1)
    assert waiting is None
    assert aligner.last_ingest_waiting is True
    assert aligner.drops.get("cam0", 0) == 0
    assert aligner.waiting_for_trigger["cam0"] == 1
    assert aligner._cam_cursor["cam0"] == 2

    aligner.ingest_mcu(_trigger_rec(2, 2 * 16_666))
    ok = aligner.ingest_cam("cam0", sensor_ts_ns=2, trigger_id=None, frame_idx=1)
    assert ok is not None
    assert ok.payload["trigger_id"] == 2
    assert ok.t_master_us == 2 * 16_666


def test_no_triggers_yet_waits_first_frame():
    aligner = ClockAligner()
    assert aligner.ingest_cam("cam0", sensor_ts_ns=1, trigger_id=None) is None
    assert aligner.last_ingest_waiting is True
    assert aligner.drops.get("cam0", 0) == 0
    assert aligner.waiting_for_trigger["cam0"] == 1
    assert "cam0" not in aligner._cam_cursor or aligner._cam_cursor.get("cam0") is None


def test_buffer_bound_with_consuming_cameras():
    """Interleave triggers and frames — buffer must not grow without bound."""
    aligner = ClockAligner(triggers_max=64)
    for i in range(1, 201):
        aligner.ingest_mcu(_trigger_rec(i, i * 16_666))
        cam = aligner.ingest_cam("cam0", sensor_ts_ns=i * 1000, trigger_id=None, frame_idx=i - 1)
        assert cam is not None
        assert cam.payload["trigger_id"] == i
        assert len(aligner._triggers) <= aligner.triggers_max
    # After steady consume, watermark keeps the list small
    assert len(aligner._triggers) < 8


def test_hard_cap_without_camera_consume():
    aligner = ClockAligner(triggers_max=32)
    _feed_triggers(aligner, 100)
    assert len(aligner._triggers) == 32
    # Oldest retained should be near the end of the 100
    assert aligner._triggers[0][0] >= 100 - 32 + 1


def test_exact_id_still_works_when_sequence_also_used():
    aligner = ClockAligner()
    _feed_triggers(aligner, 5)
    # Exact lookup mid-stream
    cam = aligner.ingest_cam("cam0", sensor_ts_ns=3_000_000, trigger_id=3)
    assert cam is not None
    assert cam.payload["trigger_id"] == 3
    # Subsequent None frames continue from cursor past 3
    nxt = aligner.ingest_cam("cam0", sensor_ts_ns=4_000_000, trigger_id=None)
    assert nxt is not None
    assert nxt.payload["trigger_id"] == 4


def test_camera_drop_recovered_from_sensor_timestamp_gap():
    aligner = ClockAligner()
    _feed_triggers(aligner, 8)

    # Warm up the affine fit with correctly locked frames.
    for i in range(4):
        cam = aligner.ingest_cam(
            "cam0",
            sensor_ts_ns=(i + 1) * 16_666 * 1000,
            trigger_id=None,
            frame_idx=i,
        )
        assert cam is not None
        assert cam.payload["trigger_id"] == i + 1

    # Camera dropped the frame for trigger 5; next delivered frame lands one period later.
    recovered = aligner.ingest_cam(
        "cam0",
        sensor_ts_ns=6 * 16_666 * 1000,
        trigger_id=None,
        frame_idx=4,
    )
    assert recovered is not None
    assert recovered.payload["trigger_id"] == 6
    assert recovered.t_master_us == 6 * 16_666
    assert aligner.camera_drop_recovered["cam0"] == 1
    assert aligner._cam_cursor["cam0"] == 7


def test_prediction_validation_rejects_bad_match():
    aligner = ClockAligner()
    for i in range(1, 7):
        t_us = i * 16_666
        if i == 5:
            t_us += 10_000
        aligner.ingest_mcu(_trigger_rec(i, t_us))

    for i in range(4):
        cam = aligner.ingest_cam(
            "cam0",
            sensor_ts_ns=(i + 1) * 16_666 * 1000,
            trigger_id=None,
            frame_idx=i,
        )
        assert cam is not None

    # Trigger 5 is badly delayed vs the sensor-time prediction; reject it.
    bad = aligner.ingest_cam(
        "cam0",
        sensor_ts_ns=5 * 16_666 * 1000,
        trigger_id=None,
        frame_idx=4,
    )
    assert bad is None
    assert aligner.validation_fail_count["cam0"] == 1
    assert aligner._cam_cursor["cam0"] == 6
    assert aligner.validation_relock_count.get("cam0", 0) == 0


def test_validation_fail_streak_relocks_instead_of_dropping_the_take():
    """A SensorTimestamp jump must not discard every subsequent frame.

    This is the 33 s cutoff seen in session_20260817T114849Z_d1937307:
    validation_fail_count == drops for the rest of the session.
    """
    aligner = ClockAligner(validation_relock_after=8)
    period = 16_666
    _feed_triggers(aligner, 40, period_us=period)

    for i in range(8):
        cam = aligner.ingest_cam(
            "cam0",
            sensor_ts_ns=(i + 1) * period * 1000,
            trigger_id=None,
            frame_idx=i,
        )
        assert cam is not None

    # Jump SensorTimestamp as if libcamera restarted, while MCU triggers continue.
    ts_jump_ns = 10_000_000_000
    dropped = 0
    kept = 0
    for i in range(8, 24):
        cam = aligner.ingest_cam(
            "cam0",
            sensor_ts_ns=(i + 1) * period * 1000 + ts_jump_ns,
            trigger_id=None,
            frame_idx=i,
        )
        if cam is None:
            dropped += 1
        else:
            kept += 1
            assert cam.payload["trigger_id"] == i + 1

    assert dropped == 7  # streak below relock_after
    assert kept >= 8  # re-locked; remaining frames written
    assert aligner.validation_relock_count["cam0"] >= 1
    assert aligner.drops.get("cam0", 0) == dropped


def test_reset_trigger_backlog_reanchors_first_frame():
    aligner = ClockAligner()
    period = 33_333
    for i in range(15):
        aligner.note_trigger(i, 1_000_000 + i * period)
    assert aligner.reset_trigger_backlog() == 15
    aligner.note_trigger(15, 1_000_000 + 15 * period)
    rec = aligner.ingest_cam("cam0", sensor_ts_ns=9_000_000_000, trigger_id=None)
    assert rec is not None
    assert rec.payload["trigger_id"] == 15
    assert rec.t_master_us == 1_000_000 + 15 * period


def test_pipeline_fill_first_frame_locks_to_recent_trigger():
    """Long pre-frame trigger burst: first completed request is a recent pulse."""
    aligner = ClockAligner()
    period = 33_333
    for i in range(6):
        aligner.note_trigger(i, 1_000_000 + i * period)
    rec = aligner.ingest_cam(
        "cam0",
        sensor_ts_ns=9_000_000_000,
        trigger_id=None,
        t_host_ns=2_000_000_000,
    )
    assert rec is not None
    assert rec.payload["trigger_id"] == 5
    assert rec.t_master_us == 1_000_000 + 5 * period


def _ingest_seq(aligner: ClockAligner, cam: str, i: int, *, period_us: int) -> object:
    return aligner.ingest_cam(
        cam,
        sensor_ts_ns=i * period_us * 1000,
        trigger_id=None,
        frame_idx=i - 1,
    )


def test_unconsumed_mvc_gap_relocks_onto_stale_trigger():
    """Dropping frames while triggers continue (old MVC drain) stamps later poses early."""
    period = 16_666
    aligner = ClockAligner()
    _feed_triggers(aligner, 60, period_us=period)
    for i in range(1, 9):
        rec = _ingest_seq(aligner, "cam0", i, period_us=period)
        assert rec is not None
        assert rec.payload["trigger_id"] == i

    # Resume at exposure 50 without consuming 9..49 (MVC discarded those frames).
    kept = None
    for i in range(50, 60):
        rec = _ingest_seq(aligner, "cam0", i, period_us=period)
        if rec is not None:
            kept = rec
            break
    assert kept is not None
    lag_ticks = 50 - int(kept.payload["trigger_id"])
    assert lag_ticks > 8, kept.payload["trigger_id"]


def test_consuming_through_gap_keeps_trigger_lock():
    """Keep pairing every frame (new MVC path): resume stays on the live pulse."""
    period = 16_666
    aligner = ClockAligner()
    _feed_triggers(aligner, 60, period_us=period)
    last = None
    for i in range(1, 51):
        last = _ingest_seq(aligner, "cam0", i, period_us=period)
        assert last is not None
        assert last.payload["trigger_id"] == i
    assert last is not None
    assert last.t_master_us == 50 * period


def _rig(al, *, n_pps=5, off_ns=7_000_000_000, period=16_666, n_trig=120, first_frame=3):
    # NOTE: callers that exercise the *anchor* pass delta_tracking=False, since
    # cadence tracking corrects a start-up offset on its own.
    """Trigger stream + a camera whose pipeline dropped the first frames.

    ``first_frame`` is the pulse the camera's first *delivered* frame actually
    exposed at, i.e. ``first_frame - 1`` start-up pulses produced no frame.
    """
    for k in range(n_pps):
        t_master = 1_000_000 + k * 100_000
        al.ingest_pps(t_master * 1000 + off_ns, t_master)
    for i in range(1, n_trig + 1):
        al.note_trigger(i, 2_000_000 + i * period, stamp_source="pio_irq")
    for i in range(first_frame, n_trig + 1):
        true_master = 2_000_000 + i * period
        al.ingest_cam("cam0", true_master * 1000 + off_ns, None, frame_idx=i - first_frame + 1)
    return al


def test_pps_anchor_pairs_the_first_delivered_frame_correctly():
    """The camera dropped 2 start-up frames; anchoring must still label frame 1 as pulse 3."""
    from jtsmm.sync.clock_align import ClockAligner

    al = _rig(ClockAligner(pps_anchor=True, delta_tracking=False, anchor_refine_after=10**9))
    anchor = al.sync_report()["camera_anchor"]["cam0"]
    assert anchor["method"] == "pps_time"
    assert anchor["trigger_id"] == 3
    assert anchor["skipped_triggers"] == 2
    assert abs(anchor["offset_us"]) < 100.0

    st = al.assignment_residual_stats("cam0")
    assert st is not None
    assert abs(st["median_frames"]) < 0.05, st          # correctly paired
    assert abs(st["spread_frames"]) < 0.05, st          # no slip after the anchor
    assert abs(st["drift_us_per_s"]) < 5.0, st


def test_without_anchor_the_startup_loss_shows_as_off_by_n():
    """pps_anchor=False keeps the old rule, so the same rig reads -2 frames."""
    from jtsmm.sync.clock_align import ClockAligner

    al = _rig(ClockAligner(pps_anchor=False, delta_tracking=False, anchor_refine_after=10**9))
    assert al.sync_report()["camera_anchor"]["cam0"]["method"] == "oldest_buffered"
    st = al.assignment_residual_stats("cam0")
    assert st is not None and st["n"] == 118
    assert abs(st["median_frames"] + 2.0) < 0.05, st
    assert abs(st["drift_us_per_s"]) < 5.0


def test_anchor_reproduces_the_measured_31_frame_case():
    """The rig session: ~31 start-up pulses lost at 60 Hz -> -517 ms.

    Without the PPS anchor the first frame locks to the oldest buffered trigger,
    a 31-frame error. The anchor refinement now measures and removes it, so the
    assertion is on the correction rather than on the raw offset.
    """
    from jtsmm.sync.clock_align import ClockAligner

    off = ClockAligner(pps_anchor=False, delta_tracking=False)
    _rig(off, n_trig=600, first_frame=32)
    assert off.anchor_info["cam0"]["refined_by_frames"] == 31
    st_off = off.assignment_residual_stats("cam0")
    assert abs(st_off["median_frames"]) < 0.1, st_off

    # With refinement disabled the raw -31 frame offset is still visible.
    raw = ClockAligner(pps_anchor=False, delta_tracking=False, anchor_refine_after=10**9)
    _rig(raw, n_trig=600, first_frame=32)
    st_raw = raw.assignment_residual_stats("cam0")
    assert abs(st_raw["median_frames"] + 31.0) < 0.1, st_raw
    assert abs(st_raw["median_us"] + 31 * 16_666) < 2000.0

    on = ClockAligner(pps_anchor=True, delta_tracking=False, anchor_refine_after=10**9)
    _rig(on, n_trig=600, first_frame=32)
    st_on = on.assignment_residual_stats("cam0")
    assert abs(st_on["median_frames"]) < 0.05, st_on
    assert on.sync_report()["camera_anchor"]["cam0"]["skipped_triggers"] == 31


def test_residual_uses_the_final_pps_fit_not_the_live_one():
    """Regression: a slope-poor start fit must not appear as ~700 us/s of drift.

    Reproduces the rig numbers -- 8 PPS samples over ~0.2 s give a ~700 ppm
    slope error; computing residuals per frame during the take turned that into
    a bogus +700 us/s drift. Stats are now computed from the fit as it stands
    when the report is built, so adding the end pass fixes the number.
    """
    from jtsmm.sync.clock_align import ClockAligner

    al = ClockAligner(pps_anchor=True)
    off_ns = 3_000_000_000
    # Start pass: 8 samples inside 200 ms, each with a little jitter -> the
    # offset is well determined but the slope is not.
    jitter = [0, 300, -250, 180, -120, 260, -300, 90]
    for k in range(8):
        t_master = 1_000_000 + k * 25_000
        al.ingest_pps(t_master * 1000 + off_ns + jitter[k] * 1000, t_master)
    # A free slope over 0.2 s would be hundreds of ppm off; the start pass is
    # therefore fitted offset-only until the samples span pps_slope_min_span_s.
    assert al._pi_fit.a == 1.0 and al.pi_fit_mode == "offset_only"

    period = 16_666
    for i in range(1, 3601):  # 60 s at 60 Hz
        al.note_trigger(i, 2_000_000 + i * period, stamp_source="pio_irq")
        true_master = 2_000_000 + i * period
        al.ingest_cam("cam0", true_master * 1000 + off_ns, None, frame_idx=i)

    # End pass: same clock, 60 s later -> the slope is now constrained.
    for k in range(8):
        t_master = 62_000_000 + k * 25_000
        al.ingest_pps(t_master * 1000 + off_ns + jitter[k] * 1000, t_master)
    assert abs(1.0 - al._pi_fit.a) * 1e6 < 20.0
    assert al.pi_fit_mode == "free_slope"

    st = al.assignment_residual_stats("cam0")
    assert st is not None and st["pi_fit_n"] == 16
    # With the final fit the residual is flat; with the live start fit this
    # same data reported hundreds of us/s.
    assert abs(st["drift_us_per_s"]) < 25.0, st
    assert abs(st["median_frames"]) < 0.1, st


def test_assignment_residual_detects_stamp_drift():
    """Trigger stamps running 4 us/frame ahead of the pulses (the pre-fix PIO bug) show as drift."""
    from jtsmm.sync.clock_align import ClockAligner

    al = ClockAligner()
    off_ns = 3_000_000_000
    for k in range(4):
        t_master = 500_000 + k * 250_000
        al.ingest_pps(t_master * 1000 + off_ns, t_master)
    period = 16_666
    for i in range(1, 181):  # 3 s at 60 Hz
        stamped = 2_000_000 + i * period          # what the firmware timer said
        al.note_trigger(i, stamped, stamp_source="timer")
        true_pulse = 2_000_000 + i * (period + 4)  # what the pin actually did
        al.ingest_cam("cam0", true_pulse * 1000 + off_ns, None, frame_idx=i)
    st = al.assignment_residual_stats("cam0")
    assert st is not None
    # 4 us/frame at 60 Hz = -240 us/s (stamps fall behind the pulses).
    assert st["drift_us_per_s"] == pytest.approx(-240.0, abs=10.0)


def test_note_trigger_counts_stamp_sources_and_fallback():
    from jtsmm.sync.clock_align import ClockAligner

    al = ClockAligner()
    al.note_trigger(1, 1000, stamp_source=1)
    al.note_trigger(2, 2000, stamp_source=2, fallback=True)
    al.note_trigger(3, 3000, stamp_source=0)
    al.note_trigger(4, 4000)  # legacy path without stamp info
    rep = al.sync_report()["trigger_stamp"]
    assert rep["counts"] == {"pio_irq": 1, "timer": 1, "unknown": 1}
    assert rep["fallback"] == 1


# --- camera cadence tracking (2026-09-18) ------------------------------------


def _cadence_rig(*, delta, period=33_333, n_trig=600, drop_bursts=(), missing_trigger_ids=()):
    """Triggers at `period`, one camera whose pipeline drops frames in bursts.

    `drop_bursts` are (first_pulse, count) pairs the camera never delivers;
    `missing_trigger_ids` are TRIGGER_EVENTs lost over USB (absent from /triggers).
    """
    from jtsmm.sync.clock_align import ClockAligner

    al = ClockAligner(pps_anchor=True, delta_tracking=delta, triggers_max=10_000)
    off_ns = 5_000_000_000
    t0 = 2_000_000
    for k in range(6):
        tm = t0 + k * 50_000
        al.ingest_pps(tm * 1000 + off_ns, tm)

    dropped = set()
    for first, count in drop_bursts:
        dropped.update(range(first, first + count))
    missing = set(missing_trigger_ids)

    for i in range(1, n_trig + 1):
        if i in missing:
            continue
        al.note_trigger(i, t0 + i * period, stamp_source="pio_irq")

    for i in range(1, n_trig + 1):
        if i in dropped:
            continue                      # camera never delivered this frame
        true_master = t0 + i * period     # its SensorTimestamp is the real pulse
        al.ingest_cam("cam0", true_master * 1000 + off_ns, None, frame_idx=i)
    return al


def test_cadence_tracking_survives_camera_frame_drops():
    """The rig failure: SensorTimestamp gaps to 14 periods shifted everything after."""
    al = _cadence_rig(delta=True, drop_bursts=[(100, 8), (250, 14), (400, 5)])
    st = al.assignment_residual_stats("cam0")
    assert st is not None
    assert abs(st["median_frames"]) < 0.1, st
    assert st["spread_frames"] < 0.1, st          # measured 0.01 on the real take
    # One advance of n+1 per burst of n dropped frames -> n extra ticks each.
    assert al.cadence_advance.get("cam0", 0) == 8 + 14 + 5


def test_without_cadence_tracking_drops_shift_the_association():
    """delta_tracking=False keeps the old advance-by-one behaviour."""
    al = _cadence_rig(delta=False, drop_bursts=[(100, 8), (250, 14), (400, 5)])
    st = al.assignment_residual_stats("cam0")
    assert st is not None
    assert st["spread_frames"] > 5.0, st
    assert al.cadence_advance.get("cam0", 0) == 0


def test_a_lost_trigger_event_costs_one_frame_not_the_mapping():
    """368 of 14037 TRIGGER_EVENTs were lost over USB on the rig take.

    Substituting a neighbour and referencing it would shift the mapping once per
    lost event; the reference must stay on the cadence-predicted id.
    """
    lost = (150, 151, 300, 455)
    al = _cadence_rig(delta=True, missing_trigger_ids=lost)
    st = al.assignment_residual_stats("cam0")
    assert st is not None
    assert st["spread_frames"] < 0.1, st           # mapping unharmed
    assert abs(st["median_frames"]) < 0.1, st
    assert al.cadence_missing_trigger.get("cam0", 0) == len(lost)
    assert al.drops.get("cam0", 0) == len(lost)    # exactly those frames lost


def test_cadence_and_lost_triggers_together():
    al = _cadence_rig(
        delta=True, drop_bursts=[(120, 10), (300, 6)], missing_trigger_ids=(200, 201, 450)
    )
    st = al.assignment_residual_stats("cam0")
    assert st["spread_frames"] < 0.1, st
    assert abs(st["median_frames"]) < 0.1, st


def test_anchor_refinement_corrects_a_gross_offset_once():
    from jtsmm.sync.clock_align import ClockAligner

    al = ClockAligner(pps_anchor=False, delta_tracking=True, triggers_max=10_000)
    period, t0, off_ns = 33_333, 2_000_000, 5_000_000_000
    for k in range(6):
        tm = t0 + k * 50_000
        al.ingest_pps(tm * 1000 + off_ns, tm)
    for i in range(1, 400):
        al.note_trigger(i, t0 + i * period, stamp_source="pio_irq")
    # Camera's first delivered frame is really pulse 20, but with pps_anchor off
    # the aligner locks it to the oldest buffered trigger -> a 19-frame offset.
    for i in range(20, 400):
        al.ingest_cam("cam0", (t0 + i * period) * 1000 + off_ns, None, frame_idx=i)
    info = al.anchor_info.get("cam0", {})
    assert info.get("refined_by_frames") == 19, info
    st = al.assignment_residual_stats("cam0")
    assert abs(st["median_frames"]) < 0.1, st      # corrected


def test_anchor_refinement_refuses_to_act_on_an_unlocked_camera():
    """Refining on a large spread shifted one camera by -13760 frames in testing."""
    al = _cadence_rig(delta=False, drop_bursts=[(50, 20), (120, 30), (200, 25)])
    st = al.assignment_residual_stats("cam0")
    assert st["spread_frames"] > 2.0, st
    info = al.anchor_info.get("cam0", {})
    assert "refined_by_frames" not in info, info
