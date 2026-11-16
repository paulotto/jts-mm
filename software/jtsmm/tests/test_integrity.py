# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Integrity checker on synthetic HDF5 fixtures.

These fixtures encode the real regressions found on recorded sessions:
camera trailing starvation, broken sync residuals, samples_uv drift, etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm.integrity.checker import check_session, check_session_issues

from tests.helpers import make_session_h5


def test_valid_session_passes(tmp_path: Path):
    path = tmp_path / "good.h5"
    make_session_h5(path, duration_s=1.0)
    report = check_session(path)
    assert report.ok, report.issues
    assert not any("trailing gap" in w for w in report.warnings)


def test_emg_disabled_skips_emg_rate_checks(tmp_path: Path):
    path = tmp_path / "no_ads.h5"
    make_session_h5(path, duration_s=1.0, gap_at=500)
    with h5py.File(path, "a") as f:
        f["meta"].attrs["emg_enabled"] = "false"
    report = check_session(path)
    assert report.ok, report.issues
    assert report.summary.get("emg", {}).get("skipped") is True


def test_emg_gap_fails(tmp_path: Path):
    path = tmp_path / "gap.h5"
    make_session_h5(path, duration_s=1.0, gap_at=500)
    report = check_session(path)
    assert not report.ok
    assert any("EMG continuity" in e for e in report.issues)


def test_emg_small_jitter_warns_not_errors(tmp_path: Path):
    """Single-period slips (~2.5–3.5× dt) warn; only large gaps fail."""
    path = tmp_path / "jitter.h5"
    make_session_h5(path, duration_s=1.0, emg_fs=2000)
    with h5py.File(path, "a") as f:
        t = f["emg"]["t_master_us"]
        arr = np.asarray(t[:], dtype=np.int64)
        # Inject a 2.6-period slip (1300 µs @ 2 kHz) — warn only.
        arr[500:] += 800
        t[...] = arr

    report = check_session(path)
    assert report.ok, report.issues
    assert any("jitter gaps" in w for w in report.warnings)


def _patch_mcu_stats(path: Path, **fields: int) -> None:
    with h5py.File(path, "a") as f:
        sync = f["sync"]
        doc = json.loads(sync.attrs["report_json"])
        doc.setdefault("mcu_stats_end", {}).update(fields)
        sync.attrs["report_json"] = json.dumps(doc)


def test_emg_frame_misalign_fails(tmp_path: Path):
    """A slipped ADS1299 byte stream must fail, not pass as a clean session.

    The samples in the affected span are the wrong bytes, not noisy readings, so
    there is no tolerable rate — any non-zero count is an error.
    """
    path = tmp_path / "misalign.h5"
    make_session_h5(path, duration_s=1.0)
    _patch_mcu_stats(
        path, frame_misalign=137, emg_misalign_drops=137, misalign_max_streak=9, emg_resyncs=2
    )
    report = check_session(path)
    assert not report.ok
    assert any("framing slipped" in e for e in report.issues)
    assert report.summary["emg_framing"]["frame_misalign"] == 137


def test_spi_wedge_warns_only(tmp_path: Path):
    """A wedged DMA transfer costs samples but does not invalidate the data."""
    path = tmp_path / "wedge.h5"
    make_session_h5(path, duration_s=1.0)
    _patch_mcu_stats(path, spi_wedges=3)
    report = check_session(path)
    assert report.ok, report.issues
    assert any("SPI DMA wedged" in w for w in report.warnings)


def test_config_hash_mismatch_fails(tmp_path: Path):
    path = tmp_path / "bad_hash.h5"
    make_session_h5(path, duration_s=1.0, bad_config_hash=True)
    report = check_session(path)
    assert not report.ok
    assert any("config_hash" in e for e in report.issues)


def test_missing_file():
    report = check_session("/nonexistent/session.h5")
    assert not report.ok


def test_missing_emg_warns(tmp_path: Path):
    path = tmp_path / "no_emg.h5"
    make_session_h5(path, duration_s=1.0)
    with h5py.File(path, "a") as f:
        emg = f["emg"]
        for key in list(emg.keys()):
            del emg[key]

    report = check_session(path)
    assert any("no EMG" in w for w in report.warnings)


def test_kinematics_accel_bench_rates_raise_no_rate_warning(tmp_path: Path):
    """Bench sweep 2026-10-06 (kinematics_accel, 0 sequence gaps): game RV 352 Hz
    and accel 232 Hz against 390/250 configured is scheduling, not loss."""
    from jtsmm.integrity.checker import IMU_RATE_TOL, _imu_rate_tolerance

    def rate_warnings(quat_hz: float, acc_hz: float) -> list[str]:
        path = tmp_path / f"accel_{quat_hz:.0f}.h5"
        make_session_h5(path, duration_s=10.0, imu_hz=quat_hz, emg_fs=2000, emg_samples=20000)
        n = int(acc_hz * 10)
        t = np.linspace(0, 10e6, n, endpoint=False, dtype=np.uint64)
        acc = np.tile(np.float32([0.0, 0.0, 9.81]), (n, 1))
        with h5py.File(path, "a") as f:
            for slot in ("imu_mta", "imu_cra"):
                g = f[slot]
                for name in ("linear_accel", "t_master_us_linear_accel"):
                    if name in g:
                        del g[name]
                g.create_dataset("accel", data=acc)
                g.create_dataset("t_master_us_accel", data=t)
        return [w for w in check_session(path).warnings if "vs configured" in w]

    assert rate_warnings(352.0, 232.0) == []
    hits = rate_warnings(300.0, 200.0)  # -23 % / -20 %: still flagged
    assert any("fusion rate" in w for w in hits) and any("accel" in w for w in hits)
    assert _imu_rate_tolerance("magnetometer") == IMU_RATE_TOL["magnetometer"]


def test_kinematics_default_rates_pass_not_ring_loss(tmp_path: Path):
    """~360 Hz quat + ~243 Hz linear vs 390/250 configured is normal BNO085 scheduling."""
    path = tmp_path / "bno_sched.h5"
    make_session_h5(path, duration_s=10.0, imu_hz=359.0, emg_fs=2000, emg_samples=20000)
    import json

    lin_n = 2430
    lin_t = np.linspace(0, 10e6, lin_n, endpoint=False, dtype=np.uint64)
    lin = np.zeros((lin_n, 3), dtype=np.float32)

    with h5py.File(path, "a") as f:
        for slot in ("imu_mta", "imu_cra"):
            g = f[slot]
            if "linear_accel" in g:
                del g["linear_accel"]
            if "t_master_us_linear_accel" in g:
                del g["t_master_us_linear_accel"]
            g.create_dataset("linear_accel", data=lin)
            g.create_dataset("t_master_us_linear_accel", data=lin_t)
        sync = f.require_group("sync")
        sync.attrs["report_json"] = json.dumps(
            {
                "mcu_stats_end": {
                    "imu_mta_push": 3600,
                    "imu_mta_drops": 0,
                    "imu_cra_push": 3600,
                    "imu_cra_drops": 0,
                    "missed_drdy": 0,
                },
                "cameras": {
                    "cam0": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0},
                    "cam1": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0},
                },
                "drops": {},
            }
        )

    report = check_session(path)
    assert report.ok, report.issues
    assert not any("ring drop" in e for e in report.issues)


# ---------------------------------------------------------------------------
# Regressions from the 30 s session.h5 audit
# ---------------------------------------------------------------------------


def test_camera_trailing_starvation_fails(tmp_path: Path):
    """Reproduce the ~3 s cam0 cliff on a 30 s / 60 Hz trigger session."""
    path = tmp_path / "cam_starve.h5"
    make_session_h5(
        path,
        duration_s=30.0,
        emg_fs=2000,
        emg_samples=60_000,
        trig_hz=60,
        cam_ids=("cam0",),
        cam_trailing_gap_s=3.0,
    )
    report = check_session(path)
    assert not report.ok, "trailing camera starvation must fail integrity"
    blob = " ".join(report.issues)
    assert "trailing gap" in blob or "covers" in blob
    assert "cam0" in blob


def test_camera_trailing_gap_just_under_error_warns(tmp_path: Path):
    path = tmp_path / "cam_soft.h5"
    make_session_h5(
        path,
        duration_s=10.0,
        emg_fs=2000,
        emg_samples=20_000,
        cam_ids=("cam0",),
        cam_trailing_gap_s=0.3,  # warn band
    )
    report = check_session(path)
    # 0.3 s on 10 s is 97% coverage (>= 95%) so only trailing-gap warn.
    assert report.ok, report.issues
    assert any("trailing gap" in w for w in report.warnings)


def test_broken_sync_residual_fails(tmp_path: Path):
    """Frame labels that slip against the SensorTimestamps, over the whole take."""
    path = tmp_path / "bad_residual.h5"
    make_session_h5(path, duration_s=4.0, cam_ids=("cam0",))
    with h5py.File(path, "a") as f:
        t = np.asarray(f["cameras"]["cam0"]["t_master_us"], dtype=np.int64)
        t[120:] += 40 * 16_667  # association slipped 40 pulses at frame 120
        f["cameras"]["cam0"]["t_master_us"][...] = t.astype(np.uint64)
    report = check_session(path)
    assert not report.ok
    assert any("residual_rms_us" in e and "broken" in e for e in report.issues)
    assert any("+40 at frame 120" in w for w in report.warnings), report.warnings
    assert report.summary["cameras"]["cam0"]["label_steps"] == 1


def test_recorder_residual_is_used_without_sensor_timestamps(tmp_path: Path):
    """Older files: only the recorder's (last-30-frame) residual exists."""
    path = tmp_path / "legacy.h5"
    make_session_h5(path, duration_s=2.0, cam_ids=("cam0",), bad_sync_residual_us=144_251.0)
    with h5py.File(path, "a") as f:
        del f["cameras"]["cam0"]["sensor_timestamp_ns"]
    report = check_session(path)
    assert any("residual_rms_us=144251" in e for e in report.issues), report.issues


def test_samples_uv_inconsistency_fails(tmp_path: Path):
    path = tmp_path / "bad_uv.h5"
    make_session_h5(path, duration_s=1.0, bad_samples_uv=True)
    report = check_session(path)
    assert not report.ok
    assert any("samples_uv inconsistent" in e for e in report.issues)


def test_emg_row_mismatch_fails(tmp_path: Path):
    path = tmp_path / "emg_mismatch.h5"
    make_session_h5(path, duration_s=1.0, emg_row_mismatch=True)
    report = check_session(path)
    assert not report.ok
    assert any("samples rows" in e for e in report.issues)


def test_noncontiguous_trigger_ids_fail(tmp_path: Path):
    path = tmp_path / "trig_skip.h5"
    make_session_h5(path, duration_s=1.0, noncontiguous_triggers=True)
    report = check_session(path)
    assert not report.ok
    assert any("not contiguous" in e for e in report.issues)


def test_camera_coverage_counts_frames_not_time_span(tmp_path: Path):
    """take1: 99.8 % of the time span covered, 12.7 % of the frames missing."""
    path = tmp_path / "thin_cam.h5"
    make_session_h5(path, duration_s=4.0, cam_ids=("cam0",), cam_frame_frac=0.85)
    report = check_session(path)
    assert any("camera cam0" in e and "trigger pulses in its span" in e for e in report.issues)
    assert abs(report.summary["cameras"]["cam0"]["coverage_frac"] - 0.85) < 0.01


def test_camera_coverage_threshold_is_configurable(tmp_path: Path):
    path = tmp_path / "thin_cam.h5"
    make_session_h5(path, duration_s=4.0, cam_ids=("cam0",), cam_frame_frac=0.85)
    report = check_session(path, resolved_config=_th_cfg(camera_coverage_min_frac=0.8))
    assert not any("trigger pulses in its span" in e for e in report.issues), report.issues


def test_configured_missing_camera_with_drops_warns(tmp_path: Path):
    path = tmp_path / "cam1_absent.h5"
    make_session_h5(
        path,
        duration_s=1.0,
        cam_ids=("cam0",),  # host YAML still lists cam1
        sync_drops={"cam1": 1692},
    )
    report = check_session(path)
    assert report.ok, report.issues
    assert any("cam1" in w and "missing" in w for w in report.warnings)


def test_omit_samples_uv_warns(tmp_path: Path):
    path = tmp_path / "no_uv.h5"
    make_session_h5(path, duration_s=1.0, omit_samples_uv=True)
    report = check_session(path)
    assert report.ok, report.issues
    assert any("samples_uv" in w for w in report.warnings)


def test_corrupt_quaternions_warn(tmp_path: Path):
    path = tmp_path / "bad_quat.h5"
    make_session_h5(path, duration_s=1.0, corrupt_quat=True)
    report = check_session(path)
    assert report.ok, report.issues
    assert any("unit norm" in w for w in report.warnings)


def test_high_sync_drops_warn(tmp_path: Path):
    path = tmp_path / "drops.h5"
    make_session_h5(path, duration_s=1.0, cam_ids=("cam0",), sync_drops={"cam0": 250})
    report = check_session(path)
    assert report.ok, report.issues
    assert any("sync.drops[cam0]" in w for w in report.warnings)


def test_skipped_camera_without_drops_warns_with_reason(tmp_path: Path):
    """cam1 configured but skipped at startup (no picamera2 device, zero
    /sync drops since it was never polled at all) must still surface a clear
    reason instead of being silently treated as fine."""
    path = tmp_path / "skipped_cam.h5"
    make_session_h5(
        path, duration_s=1.0, cam_ids=("cam0",), sync_drops={}, skipped_cameras=("cam1",),
    )
    report = check_session(path)
    assert report.ok, report.issues
    assert any(
        "cam1" in w and "no picamera2 device" in w for w in report.warnings
    ), report.warnings


def test_missing_camera_without_drops_or_skip_reason_is_silent(tmp_path: Path):
    """Baseline: an unconnected camera with no drops and no recorded skip
    reason (older sessions) is still allowed — matches real hardware where
    only a subset of configured cameras is physically present."""
    path = tmp_path / "missing_cam.h5"
    make_session_h5(path, duration_s=1.0, cam_ids=("cam0",), sync_drops={})
    report = check_session(path)
    assert report.ok, report.issues
    assert not any("cam1" in w for w in report.warnings)


def test_check_session_issues_prefixes_warnings(tmp_path: Path):
    path = tmp_path / "issues.h5"
    make_session_h5(path, duration_s=1.0, omit_samples_uv=True)
    issues = check_session_issues(path)
    assert any(i.startswith("warn:") for i in issues)


def test_real_session_fixture_shape_if_present():
    """Optional smoke: repo data/session.h5 (whatever was last recorded) must
    parse cleanly and must not resurrect the fixed cam0 starvation cliff —
    a large trailing gap on any camera stream is still a hard failure."""
    root = Path(__file__).resolve().parents[3]
    session = root / "data" / "session.h5"
    if not session.is_file():
        pytest.skip("data/session.h5 not present")
    report = check_session(session)
    assert not any("poll starvation" in e for e in report.issues), report.issues
    for cam_name, cam in report.summary.get("cameras", {}).items():
        gap = cam.get("trailing_gap_s")
        assert gap is None or gap < 0.5, f"{cam_name}: trailing_gap_s={gap}"


# --- 2026-09 additions: coverage + sync-report cross-checks ---------------------


def test_imu_stream_that_stops_early_fails(tmp_path: Path):
    """An IMU that died mid-take passed before: its rate was measured over its own span."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=4.0, emg_samples=100_000, imu_truncate_s=2.0)
    report = check_session(path)
    assert any("imu_mta" in e and "stopped early" in e for e in report.issues), report.issues
    assert any("imu_cra" in e and "stopped early" in e for e in report.issues)
    assert report.summary["coverage"]["imu_mta"]["end_gap_s"] > 1.5


def test_emg_stream_that_stops_early_fails(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=4.0, emg_samples=100_000, emg_truncate_s=1.0)
    report = check_session(path)
    assert any(e.startswith("emg:") and "stopped early" in e for e in report.issues), report.issues


def test_full_coverage_passes_without_coverage_issues(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=4.0, emg_samples=100_000)
    report = check_session(path)
    assert not any("stopped early" in e for e in report.issues), report.issues
    assert "coverage" in report.summary


def test_trigger_stamp_fallback_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={"trigger_stamp": {"counts": {"pio_irq": 50, "timer": 10}, "fallback": 10}},
    )
    report = check_session(path)
    assert report.ok, report.issues
    assert any("timer fallback" in w for w in report.warnings), report.warnings


def test_trigger_stamp_timer_only_warns_legacy(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"trigger_stamp": {"counts": {"unknown": 60}, "fallback": 0}})
    report = check_session(path)
    assert any("timer-based" in w for w in report.warnings), report.warnings


def test_trigger_stamp_pio_irq_is_quiet(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"trigger_stamp": {"counts": {"pio_irq": 60}, "fallback": 0}})
    report = check_session(path)
    assert not any("trigger stamps" in w for w in report.warnings), report.warnings
    assert report.summary["sync"]["trigger_stamp"]["counts"] == {"pio_irq": 60}


def test_host_queue_drops_fail(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={"host_capture": {"queue_drops": {"raw_chunks": 0, "records": 12}, "crc_errors": 0}},
    )
    report = check_session(path)
    assert any("host capture queues overflowed" in e for e in report.issues), report.issues


def test_no_pps_replies_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={"pps": {"start": {"attempted": 8, "accepted": 0}, "end": None}},
    )
    report = check_session(path)
    assert any("no `pps` replies" in w for w in report.warnings), report.warnings


def test_camera_assignment_offset_and_drift_warn(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={
            "cameras": {
                "cam0": {
                    "a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                    "assign_residual": {
                        "n": 500, "median_us": -33332.0, "p10_us": -33400.0, "p90_us": -33300.0,
                        "median_frames": -2.0, "drift_us_per_s": -240.0,
                    },
                }
            }
        },
    )
    report = check_session(path)
    assert any("camera cam0" in w and "off by" in w for w in report.warnings), report.warnings
    assert any("camera cam0" in w and "drift" in w for w in report.warnings), report.warnings
    assert report.summary["cameras"]["cam0"]["assign_residual_frames"] == -2.0


def test_camera_assignment_small_residual_is_quiet(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={
            "cameras": {
                "cam0": {
                    "a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                    "assign_residual": {
                        "n": 500, "median_us": 4100.0, "p10_us": 3900.0, "p90_us": 4300.0,
                        "median_frames": 0.25, "drift_us_per_s": 2.0,
                    },
                }
            }
        },
    )
    report = check_session(path)
    assert not any("camera cam0" in w and ("off by" in w or "drift" in w) for w in report.warnings)



def _offset_session(path: Path, median_us: float) -> None:
    """40 fps (25 ms), 12.5 ms expected trigger-to-sensor latency, PPS anchor."""
    make_session_h5(
        path,
        sync_extra={
            "cameras": {
                "cam0": {
                    "a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                    "assign_residual": {
                        "n": 500, "median_us": median_us, "p10_us": median_us - 100.0,
                        "p90_us": median_us + 100.0, "median_frames": median_us / 25000.0,
                        "spread_frames": 0.01, "drift_us_per_s": 0.5,
                    },
                }
            },
            "camera_anchor": {"cam0": {"method": "pps_time", "latency_us": 12500.0}},
        },
    )


def test_camera_assignment_one_frame_off_the_expected_latency_fails(tmp_path: Path):
    # 2026-11-05: a one-frame error puts the median at -37.5 ms = -1.5 periods,
    # which the |median| gate (warn > 1.5) lets through; the deviation from the
    # expected latency (-1.0 period) does not.
    path = tmp_path / "s.h5"
    _offset_session(path, -37500.0)
    report = check_session(path)
    assert any("camera cam0" in e and "trigger-to-sensor" in e for e in report.issues), \
        report.issues
    assert report.summary["cameras"]["cam0"]["assign_offset_dev_frames"] == -1.0


def test_camera_assignment_at_the_expected_latency_is_quiet(tmp_path: Path):
    path = tmp_path / "s.h5"
    _offset_session(path, -12480.0)       # the validation sessions: -12.48 ms at 40 fps
    report = check_session(path)
    assert not any("trigger-to-sensor" in m for m in report.issues + report.warnings)
    assert abs(report.summary["cameras"]["cam0"]["assign_offset_dev_frames"]) < 0.01

# --- 2026-09-18: dead IMU slot, anchor reporting, gap re-attribution ---------


def test_enabled_imu_slot_with_zero_pushes_is_an_error(tmp_path: Path):
    """The rig's CRA port: enabled, streamed, produced nothing. Was a 'legacy session?' warn."""
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={
            "mcu_stats_end": {
                "imu_cra_push": 0,
                "imu_cra_i2c_aborts": 10775,
                "imu_i2c_aborts": 10775,
            }
        },
    )
    report = check_session(path)
    errs = [e for e in report.issues if e.startswith("imu_cra:")]
    assert errs, report.issues
    assert "produced no samples" in errs[0]
    assert "10775 I2C aborts" in errs[0]
    assert "PCA9546" in errs[0] and "mcu.imu_cra.enabled: false" in errs[0]
    # ...and it must not also be described as a legacy file.
    assert not any("legacy session" in w and "imu_cra" in w for w in report.warnings)


def test_healthy_imu_slot_push_count_is_not_flagged(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"mcu_stats_end": {"imu_cra_push": 38186}})
    report = check_session(path)
    assert not any("produced no samples" in e for e in report.issues), report.issues


def test_trailing_gap_matching_the_offset_is_not_called_starvation(tmp_path: Path):
    """0.517 s gap with a -31 frame offset is the labelling error, not poll starvation."""
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        duration_s=30.0,
        emg_samples=100_000,
        cam_trailing_gap_s=0.6,
        sync_extra={
            "cameras": {
                cid: {
                    "a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                    "assign_residual": {
                        "n": 200, "median_us": -600_000.0, "p10_us": -601_000.0,
                        "p90_us": -599_000.0, "median_frames": -36.0,
                        "spread_frames": 0.06, "drift_us_per_s": 1.0,
                    },
                }
                for cid in ("cam0", "cam1")
            }
        },
    )
    report = check_session(path)
    gap_errs = [e for e in report.issues if "trailing gap" in e]
    assert gap_errs, report.issues
    assert all("labelling offset" in e for e in gap_errs), gap_errs
    # The old, misleading phrasing must be gone (the new text says "not poll starvation").
    assert not any("likely poll starvation" in e for e in gap_errs), gap_errs


def test_trailing_gap_without_an_offset_still_reads_as_starvation(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=30.0, emg_samples=100_000, cam_trailing_gap_s=0.6)
    report = check_session(path)
    assert any("poll starvation" in e for e in report.issues), report.issues


def test_oldest_buffered_anchor_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={
            "camera_anchor": {
                "cam0": {"method": "oldest_buffered", "trigger_id": 1, "note": "no usable PPS fit"}
            }
        },
    )
    report = check_session(path)
    assert any("anchored on the oldest buffered trigger" in w for w in report.warnings)
    assert report.summary["sync"]["camera_anchor"]["cam0"] == "oldest_buffered"


def test_pps_time_anchor_is_quiet(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(
        path,
        sync_extra={
            "camera_anchor": {
                "cam0": {"method": "pps_time", "trigger_id": 32, "offset_us": 120.0,
                         "skipped_triggers": 31}
            }
        },
    )
    report = check_session(path)
    assert not any("anchored on the oldest" in w for w in report.warnings), report.warnings
    assert report.summary["sync"]["camera_anchor"]["cam0"] == "pps_time"


def _cam_sync(**resid):
    base = {"n": 500, "median_us": 0.0, "p10_us": 0.0, "p90_us": 0.0}
    base.update(resid)
    return {"cameras": {"cam0": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                                 "assign_residual": base}}}


def test_mid_take_association_slip_warns(tmp_path: Path):
    """A slip under 3 frames is a warning: recoverable, worth looking at."""
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=4000.0, p10_us=3000.0, p90_us=60_000.0,
        median_frames=0.24, spread_frames=1.8, drift_us_per_s=3.0))
    report = check_session(path)
    assert any("offset is not constant" in w for w in report.warnings), report.warnings
    assert report.ok, report.issues


# --- the anchor failure this gate exists for ----------------------------------
#
# Trigger armed before the camera pipeline was streaming -> the first frame
# anchors on a stale pulse -> every row inherits the same offset. Nothing
# downstream looks wrong: the poses are internally consistent, just attributed
# to the wrong instant. optical_imu_sync catches it physically, but only with
# optical tracking AND enough head motion; these gates are what is left on an
# EMG/IMU take or a static one.


def test_constant_frame_offset_is_an_error_not_a_warning(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, p10_us=-333_400.0, p90_us=-333_300.0,
        median_frames=-10.0, spread_frames=0.01))
    report = check_session(path)
    assert not report.ok, "a 10-frame anchor slip must fail the session"
    assert any("inherited the offset" in e for e in report.issues), report.issues
    assert report.summary["cameras"]["cam0"]["assign_residual_frames"] == -10.0


def test_two_frame_offset_still_only_warns(tmp_path: Path):
    """Below the error grade, so a sub-frame calibration issue cannot fail a take."""
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-66_664.0, median_frames=-2.0))
    report = check_session(path)
    assert report.ok, report.issues
    assert any("off by" in w for w in report.warnings), report.warnings


def test_large_mid_take_slip_is_an_error(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=4000.0, p10_us=3000.0, p90_us=90_000.0,
        median_frames=0.24, spread_frames=5.2))
    report = check_session(path)
    assert not report.ok, report.issues
    assert any("different camera-time errors" in e for e in report.issues), report.issues


def test_runaway_drift_is_an_error(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(median_frames=0.1, drift_us_per_s=900.0))
    report = check_session(path)
    assert not report.ok, report.issues
    assert any("no constant offset correction" in e for e in report.issues), report.issues


def test_moderate_drift_still_only_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(median_frames=0.1, drift_us_per_s=-240.0))
    report = check_session(path)
    assert report.ok, report.issues
    assert any("drift" in w for w in report.warnings), report.warnings


# --- EMG signal quality (mains / saturated DRL) -------------------------------


def test_mains_dominated_emg_fails_integrity(tmp_path: Path):
    """Reproduces session_20260918T081708Z: ~7.6 mV of 50 Hz passed as OK before."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=10.0, emg_samples=100_000, emg_fs=2000, emg_mains_uv=7600.0)
    report = check_session(path)
    errs = [e for e in report.issues if "mains interference dominates" in e]
    assert errs, report.issues
    assert "50 Hz" in errs[0] and "not muscle activity" in errs[0]
    sig = report.summary["emg_signal"]
    assert sig["line_hz"] == 50.0
    assert sig["channels"]["masseter_r"]["bad_window_frac"] == 1.0
    assert sig["channels"]["masseter_r"]["mains_share"] > 0.9


def test_swinging_drl_warns_but_does_not_fail_integrity(tmp_path: Path):
    """A swinging bias node is reported, never failed, until the driven loop is
    bench-verified (the readback is informational)."""
    path = tmp_path / "s.h5"
    # 60 mV RMS: well above "high" yet unclipped at CH8's +-187.5 mV full scale
    # (gain 24); a bigger swing rails the column and reads "n/a" by design.
    make_session_h5(path, duration_s=10.0, emg_samples=100_000, emg_fs=2000, drl_ac_mv=60.0)
    report = check_session(path)
    assert not any("DRL" in e for e in report.issues), report.issues
    assert any("DRL readback" in w and "high" in w for w in report.warnings), report.warnings
    assert report.summary["emg_signal"]["drl_ac_mv"] == pytest.approx(60.0, rel=0.1)
    assert report.summary["emg_signal"]["drl_status"] == "high"


def test_small_drl_swing_only_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=10.0, emg_samples=100_000, emg_fs=2000, drl_ac_mv=20.0)
    report = check_session(path)
    assert not any("DRL" in e for e in report.issues), report.issues
    assert any("DRL readback" in w for w in report.warnings), report.warnings


def test_pinned_drl_does_not_fail_integrity(tmp_path: Path):
    """The MVC case: CH8 at the rail with no AC is the healthy state here."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=10.0, emg_samples=100_000, emg_fs=2000, drl_pinned=True)
    report = check_session(path)
    sig = report.summary["emg_signal"]
    assert sig["drl_rail_frac"] > 0.9 and sig["drl_ac_mv"] < 1.0, sig
    assert sig["drl_status"] == "n/a"  # railed: no verdict either way
    assert not any("DRL" in m for m in report.issues + report.warnings)


def test_clean_long_session_has_no_emg_signal_findings(tmp_path: Path):
    """The quiet direction end-to-end: a healthy record must stay silent."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=10.0, emg_samples=100_000, emg_fs=2000)
    report = check_session(path)
    assert "emg_signal" in report.summary
    assert not any("mains interference" in m for m in report.issues + report.warnings)
    assert not any("DRL/bias amplifier" in m for m in report.issues + report.warnings)


def test_optical_session_passes(tmp_path: Path):
    path = tmp_path / "opt_ok.h5"
    make_session_h5(path, duration_s=1.0, optical=True)
    report = check_session(path)
    assert report.ok, report.issues
    assert report.summary["optical"]["rows"] > 0
    assert report.summary["optical"]["bodies"]["HP"]["lock_frac"] == 1.0


def test_optical_tracking_without_rows_fails(tmp_path: Path):
    path = tmp_path / "opt_missing.h5"
    make_session_h5(path, duration_s=1.0, optical_missing=True)
    report = check_session(path)
    assert not report.ok
    assert any("optical tracking enabled" in e for e in report.issues)


def test_optical_bad_se3_fails(tmp_path: Path):
    path = tmp_path / "opt_se3.h5"
    make_session_h5(path, duration_s=1.0, optical=True, optical_bad_se3=True)
    report = check_session(path)
    assert not report.ok
    assert any("SE(3)" in e for e in report.issues)


def test_optical_jump_fails(tmp_path: Path):
    path = tmp_path / "opt_jump.h5"
    make_session_h5(path, duration_s=1.0, optical=True, optical_jump_mm=300.0)
    report = check_session(path)
    assert not report.ok
    assert any("jumps" in e for e in report.issues)


def test_decimated_live_tracking_is_not_frame_loss(tmp_path: Path):
    """Every 6th pose at a recorded stride of 6 covers the live rate, not a drop."""
    path = tmp_path / "decim.h5"
    make_session_h5(path, duration_s=2.0, optical=True, optical_track_stride=6)
    report = check_session(path)
    assert not any("optical rows cover" in m for m in report.issues + report.warnings), (
        report.issues + report.warnings
    )
    assert not any("stereo lock" in m for m in report.issues + report.warnings)
    assert report.summary["optical"]["track_stride"] == 6


def test_thin_optical_without_stride_still_fails_coverage(tmp_path: Path):
    path = tmp_path / "thin.h5"
    make_session_h5(path, duration_s=2.0, optical=True, optical_track_stride=6)
    with h5py.File(path, "a") as f:
        del f["meta"].attrs["optical_track_stride"]
    report = check_session(path)
    assert any("optical rows cover" in e for e in report.issues)


def test_optical_low_lock_fails(tmp_path: Path):
    path = tmp_path / "opt_lock.h5"
    make_session_h5(
        path,
        duration_s=1.0,
        optical=True,
        optical_nan_frac={"HP": 0.98, "MP": 0.98},
    )
    report = check_session(path)
    assert not report.ok
    assert any("stereo lock" in e for e in report.issues)


# --- configurable gates -------------------------------------------------------
#
# These decide whether a finished recording counts as usable, so they are policy
# rather than physics and belong in the operator YAML. They are read from the
# closed HDF5, so they cannot affect sampling -- only the verdict and exit code.


def _th_cfg(**kw):
    return {"host": {"integrity": {"thresholds": kw}}}


def test_thresholds_default_when_no_config_block():
    from jtsmm.integrity.checker import IntegrityThresholds

    assert IntegrityThresholds.from_config({}) is None
    assert IntegrityThresholds.from_config({"host": {"integrity": {}}}) is None
    assert IntegrityThresholds.from_config(None) is None
    assert IntegrityThresholds().assign_frames_err == 3.0


def test_thresholds_partial_override_keeps_other_defaults():
    from jtsmm.integrity.checker import IntegrityThresholds

    th = IntegrityThresholds.from_config(_th_cfg(assign_frames_err=8.0))
    assert th.assign_frames_err == 8.0
    assert th.assign_frames_warn == 1.5           # untouched
    assert th.stream_start_lag_err_s == 10.0


def test_thresholds_ignore_unknown_and_unparseable_keys():
    from jtsmm.integrity.checker import IntegrityThresholds

    th = IntegrityThresholds.from_config(
        _th_cfg(assign_frames_err=5.0, nonsense=1.0, sync_residual_err_us="abc")
    )
    assert th.assign_frames_err == 5.0
    assert th.sync_residual_err_us == 50_000.0


def test_inverted_warn_error_pair_falls_back_and_says_so():
    """err below warn makes the warning unreachable -- a silent footgun."""
    from jtsmm.integrity.checker import IntegrityThresholds

    th, notes = IntegrityThresholds.from_config(
        _th_cfg(assign_frames_warn=5.0, assign_frames_err=1.0)
    ).validated()
    assert th.assign_frames_warn == 1.5 and th.assign_frames_err == 3.0
    assert any("assign_frames_err" in n and "unreachable" in n for n in notes)


def _strip_session_integrity(path: Path) -> None:
    """Remove host.integrity from the config embedded in the session."""
    with h5py.File(path, "a") as f:
        cfg = json.loads(f["meta"].attrs["host_config_json"])
        cfg.get("host", {}).pop("integrity", None)
        f["meta"].attrs["host_config_json"] = json.dumps(cfg)


def _set_session_integrity(path: Path, block: dict) -> None:
    with h5py.File(path, "a") as f:
        cfg = json.loads(f["meta"].attrs["host_config_json"])
        cfg.setdefault("host", {})["integrity"] = block
        f["meta"].attrs["host_config_json"] = json.dumps(cfg)


def test_loosened_gate_lets_a_session_pass(tmp_path: Path):
    """The 10-frame anchor slip errors by default; a study that accepts it says so."""
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, median_frames=-10.0, spread_frames=0.01))

    assert not ck.check_session(path).ok

    report = ck.check_session(
        path, resolved_config=_th_cfg(assign_frames_err=20.0), config_source="/x/op.yaml"
    )
    assert report.ok, report.issues
    assert any("off by" in w for w in report.warnings), report.warnings
    assert report.summary["thresholds"]["source"] == "explicit config"
    assert report.summary["thresholds"]["source_path"] == "/x/op.yaml"
    assert report.summary["thresholds"]["assign_frames_err"] == 20.0


def test_tightened_gate_fails_a_session_that_passed(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(median_us=-66_664.0, median_frames=-2.0))
    assert ck.check_session(path).ok
    assert not ck.check_session(path, resolved_config=_th_cfg(assign_frames_err=1.6)).ok


def test_session_embedded_config_is_the_fallback(tmp_path: Path):
    """A file stays checkable on a machine with no config of its own: the
    thresholds recorded inside the session are used, and named as the source."""
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path)
    _set_session_integrity(path, {"thresholds": {"emg_gap_err_periods": 7.0}})
    th = ck.check_session(path).summary["thresholds"]
    assert th["source"] == "session config"
    assert th["source_path"].endswith("s.h5:/meta/host_config_json")
    assert th["emg_gap_err_periods"] == 7.0


def test_explicit_config_without_the_block_falls_through_to_the_session(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path)
    _set_session_integrity(path, {"thresholds": {"emg_gap_err_periods": 7.0}})
    th = ck.check_session(path, resolved_config={"host": {}}).summary["thresholds"]
    assert th["source"] == "session config" and th["emg_gap_err_periods"] == 7.0


def test_defaults_are_recorded_when_neither_config_has_the_block(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path)
    _strip_session_integrity(path)
    th = ck.check_session(path).summary["thresholds"]
    assert th["source"] == "defaults" and th["source_path"] is None
    assert th["emg_gap_err_periods"] == 4.0


def test_no_config_is_picked_up_implicitly_from_this_machine(tmp_path: Path, monkeypatch):
    """The packaged template (or a stray ~/jtsmm.yaml) must not steer a verdict."""
    import jtsmm.config as cfg_mod
    from jtsmm.integrity import checker as ck

    def _boom(*_a, **_k):
        raise AssertionError("check_session must not load a config on its own")

    monkeypatch.setattr(cfg_mod, "load", _boom)
    path = tmp_path / "s.h5"
    make_session_h5(path)
    _strip_session_integrity(path)
    assert ck.check_session(path).summary["thresholds"]["source"] == "defaults"


def test_cli_config_option_supplies_the_gates(tmp_path: Path):
    import yaml

    from jtsmm.cli.integrity import run_integrity_check

    from tests.helpers import load_operator_template

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, median_frames=-10.0, spread_frames=0.01))
    doc = load_operator_template()
    doc["host"]["integrity"]["thresholds"]["assign_frames_err"] = 20.0
    cfg = tmp_path / "op.yaml"
    cfg.write_text(yaml.safe_dump(doc), encoding="utf-8")
    report = run_integrity_check(path, config_path=cfg, plain=True)
    assert report.ok, report.issues
    assert report.summary["thresholds"]["source_path"] == str(cfg.resolve())
    with pytest.raises(SystemExit):
        run_integrity_check(path, plain=True)  # session config: the 3-frame gate fails


# --- errors_as_warnings -------------------------------------------------------
#
# For a run that must not be stopped by a verdict. The hazard is laundering: a
# session passed with the gates off must never look like one that passed on its
# merits, or it ends up in a results table indistinguishable from a good take.


def _eaw(flag: bool, **th):
    block = {"errors_as_warnings": flag}
    if th:
        block["thresholds"] = th
    return {"host": {"integrity": block}}


def test_errors_as_warnings_clears_the_failure(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, median_frames=-10.0, spread_frames=0.01))

    strict = ck.check_session(path, resolved_config=_eaw(False))
    assert not strict.ok and len(strict.issues) >= 1

    soft = ck.check_session(path, resolved_config=_eaw(True))
    assert soft.ok, soft.issues
    assert soft.issues == []
    assert soft.summary["errors_as_warnings"]["source"] == "explicit config"


def test_demotion_is_recorded_and_not_laundered(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, median_frames=-10.0, spread_frames=0.01))
    report = ck.check_session(path, resolved_config=_eaw(True))

    eaw = report.summary["errors_as_warnings"]
    assert eaw["enabled"] is True
    assert any("inherited the offset" in m for m in eaw["demoted"]), eaw
    # banner first, full text preserved behind a marker
    assert "errors_as_warnings is ON" in report.warnings[0]
    assert any(w.startswith("would-fail: ") for w in report.warnings)
    # and the verdict line says so, so a pasted report cannot mislead
    assert "demoted by errors_as_warnings" in ck.format_report(report)


def test_errors_as_warnings_leaves_a_clean_session_alone(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path)
    report = ck.check_session(path, resolved_config=_eaw(True))
    assert report.summary["errors_as_warnings"]["demoted"] == []
    assert not any("would-fail" in w for w in report.warnings)
    assert "demoted by" not in ck.format_report(report)


def test_errors_as_warnings_falls_back_to_the_session_config(tmp_path: Path):
    from jtsmm.integrity import checker as ck

    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_cam_sync(
        median_us=-333_320.0, median_frames=-10.0, spread_frames=0.01))
    _set_session_integrity(path, {"errors_as_warnings": True})
    report = ck.check_session(path)
    assert report.ok
    assert report.summary["errors_as_warnings"]["source"] == "session config"
    # an explicit config that says false wins over the session's true
    assert not ck.check_session(path, resolved_config=_eaw(False)).ok


def test_abort_on_error_accessor_defaults_to_true():
    from jtsmm.config.registry import record_abort_on_error

    assert record_abort_on_error({}) is True
    assert record_abort_on_error({"host": {"record": {"abort_on_error": False}}}) is False
