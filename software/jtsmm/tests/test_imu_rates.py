# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU rate-validation analysis (pure logic; no hardware)."""

from __future__ import annotations

import pytest

from jtsmm.doctor.imu_rates import (
    analyze,
    expected_rates,
    measure_capture,
    measured_hz,
    report_name,
)
from jtsmm.protocol.types import RecordType


def _ts(rate_hz: float, n: int, start_us: int = 0) -> list[int]:
    dt = 1e6 / rate_hz
    return [int(start_us + i * dt) for i in range(n)]


def test_measured_hz_basic():
    assert measured_hz(_ts(400, 401)) == 400.0
    assert measured_hz([]) == 0.0
    assert measured_hz([5]) == 0.0


def test_report_name_mapping():
    assert report_name("quat", {"fusion": "game_rotation_vector"}, None) == "game_rotation_vector"
    assert report_name("quat", {"fusion": None}, "rotation_vector") == "rotation_vector"
    assert report_name("lin", {}, None) == "linear_acceleration"
    assert report_name("raw", {"gyro": (1, 2, 3)}, None) == "gyroscope"
    assert report_name("raw", {"accel": (1, 2, 3)}, None) == "accelerometer"
    assert report_name("raw", {"mag": (1, 2, 3)}, None) == "magnetometer"


def test_expected_rates_reads_enabled_only():
    resolved = {
        "mcu": {
            "imu_mta": {"reports": {
                "game_rotation_vector": {"enabled": True, "rate_hz": 390},
                "gyroscope": {"enabled": True, "rate_hz": 400},
                "magnetometer": {"enabled": False, "rate_hz": 100},
            }},
            "imu_cra": {"reports": {"game_rotation_vector": {"enabled": True, "rate_hz": 390}}},
        }
    }
    exp = expected_rates(resolved)
    assert exp[("mta", "game_rotation_vector")] == 390
    assert exp[("mta", "gyroscope")] == 400
    assert ("mta", "magnetometer") not in exp
    assert exp[("cra", "game_rotation_vector")] == 390


def test_analyze_pass_within_tolerance():
    expected = {("mta", "gyroscope"): 400.0}
    timestamps = {("mta", "gyroscope"): _ts(398, 400)}  # ~0.5% low, within 2%
    stats = {"imu_mta_push": 400, "imu_mta_drops": 0}
    [r] = analyze(timestamps, expected, stats)
    assert r.ok and r.note == ""


def test_analyze_fails_on_rate_and_drops_and_missing():
    expected = {
        ("mta", "gyroscope"): 400.0,
        ("mta", "linear_acceleration"): 200.0,
        ("cra", "game_rotation_vector"): 390.0,
    }
    timestamps = {
        ("mta", "gyroscope"): _ts(300, 300),          # 25% low -> FAIL
        ("cra", "game_rotation_vector"): _ts(390, 390),  # ok rate but drops high
        # linear_acceleration missing entirely -> no data FAIL
    }
    stats = {"imu_cra_push": 100, "imu_cra_drops": 10}  # 9% drops
    results = {(_.slot, _.report): _ for _ in analyze(timestamps, expected, stats)}
    assert not results[("mta", "gyroscope")].ok
    assert "rate off" in results[("mta", "gyroscope")].note
    assert not results[("mta", "linear_acceleration")].ok
    assert results[("mta", "linear_acceleration")].note == "no data captured"
    assert not results[("cra", "game_rotation_vector")].ok
    assert "drops" in results[("cra", "game_rotation_vector")].note


def test_analyze_flags_unexpected_stream():
    [r] = analyze({("mta", "magnetometer"): _ts(100, 100)}, {})
    assert not r.ok and "unexpected" in r.note


def test_offline_report_measures_session_rates(tmp_path):
    """python -m jtsmm.tools.imu_rate_report: measured Hz from stored timestamps."""
    pytest.importorskip("h5py")
    from jtsmm.io.hdf5_writer import Hdf5Writer
    from jtsmm.sync.clock_align import AlignedRecord
    from jtsmm.tools.imu_rate_report import report_session

    mcu = {
        "schema_version": 1,
        "imu_mta": {"reports": {
            "game_rotation_vector": {"enabled": True, "rate_hz": 390},
            "gyroscope": {"enabled": True, "rate_hz": 400},
        }},
    }
    path = tmp_path / "session.h5"
    w = Hdf5Writer(path=path, config_hash="beef", mcu_config=mcu, host_config={})
    w.open()
    # 391 quat samples @ 390 Hz and 401 gyro samples @ 400 Hz over ~1 s.
    for i in range(391):
        w.append(AlignedRecord("imu_mta", int(i * 1e6 / 390),
                               {"quat": (1.0, 0.0, 0.0, 0.0), "quat_report_id": 0x08}))
    for i in range(401):
        w.append(AlignedRecord("imu_mta", int(i * 1e6 / 400),
                               {"gyro": (10, 20, 30), "sensor_id": 0x02}))
    w.close()

    by = {(r.slot, r.stream): r for r in report_session(path)}
    assert by[("mta", "quat")].measured_hz == pytest.approx(390, rel=0.01)
    assert by[("mta", "gyro")].measured_hz == pytest.approx(400, rel=0.01)
    assert by[("mta", "gyro")].configured_hz == 400


def _sweep_doc():
    from tests.helpers import load_operator_template

    return load_operator_template()  # kinematics_accel: game_rv@390 + accelerometer@250


def test_sweep_picks_the_lone_motion_report_and_caps_rates():
    from jtsmm.doctor.imu_rates import sweep_rates_for, sweep_report_for

    doc = _sweep_doc()
    assert sweep_report_for(doc) == "accelerometer"
    assert sweep_rates_for("accelerometer") == [100, 125, 200, 250, 333, 400, 500]
    assert sweep_rates_for("gyroscope") == [100, 125, 200, 250, 333, 400]  # max 400
    assert sweep_rates_for("accelerometer", [250, 600, 250]) == [250]
    with pytest.raises(ValueError, match="no sweep rate"):
        sweep_rates_for("magnetometer", [200])
    doc["mcu"]["imu_mta"]["reports"]["gyroscope"]["enabled"] = True
    with pytest.raises(ValueError, match="exactly one"):
        sweep_report_for(doc)
    assert sweep_report_for(doc, "cra") == "accelerometer"


def test_sweep_configs_change_only_the_swept_report():
    from jtsmm.config import build_mcu_apply_payload
    from jtsmm.doctor.imu_rates import sweep_configs

    doc = _sweep_doc()
    cases = sweep_configs(doc, "accelerometer", [125, 333], slot="mta")
    assert [label for label, _ in cases] == ["accelerometer@125", "accelerometer@333"]
    wire = build_mcu_apply_payload(cases[1][1])
    assert wire["imu_mta"]["reports"] == {
        "game_rotation_vector": {"enabled": True, "rate_hz": 390},
        "accelerometer": {"enabled": True, "rate_hz": 333},
    }
    assert wire["imu_cra"]["reports"]["accelerometer"]["rate_hz"] == 250  # untouched
    assert doc["mcu"]["imu_mta"]["reports"]["accelerometer"]["rate_hz"] == 250  # no aliasing
    # Sweeping the fusion report keeps fusion.rate_hz in step (the validator demands it).
    _, cfg = sweep_configs(doc, "game_rotation_vector", [200])[0]
    assert cfg["mcu"]["imu_cra"]["fusion"]["rate_hz"] == 200


def _case(label, rates, drop=0.0, note=""):
    from jtsmm.doctor.imu_rates import CaseResult, StreamRate

    streams = [StreamRate(slot, rep, 100, hz, 0.0, drop, False)
               for (slot, rep), hz in rates.items()]
    return CaseResult(label=label, duration_s=10.0, streams=streams, note=note)


def test_sweep_best_is_the_fastest_request_that_keeps_the_fusion_rate():
    from jtsmm.doctor.imu_rates import summarize_sweep

    def both(grv, acc):
        return {(slot, rep): hz for slot in ("mta", "cra")
                for rep, hz in (("game_rotation_vector", grv), ("accelerometer", acc))}

    results = [
        _case("200", both(398.0, 121.0)),   # bad interleave: collapses
        _case("250", both(352.0, 221.0)),
        _case("400", both(331.0, 330.0)),   # faster accel, but costs the fusion report
        _case("500", both(352.0, 240.0), drop=0.01),  # drops
        _case("333", {}, note="config apply FAILED"),
    ]
    rates = [200, 250, 400, 500, 333]
    rows = summarize_sweep(results, "accelerometer", rates, min_fusion_hz=0.85 * 390)
    assert [r.best for r in rows] == [False, True, False, False, False]
    assert rows[1].swept_hz == {"mta": 221.0, "cra": 221.0}
    assert not rows[4].complete
    # A lower floor accepts the throttled fusion for the faster accel.
    rows = summarize_sweep(results, "accelerometer", rates, min_fusion_hz=300)
    assert [r.best for r in rows] == [False, False, True, False, False]
    # Nothing usable -> no best.
    assert not any(r.best for r in summarize_sweep(results[3:], "accelerometer", [500, 333]))


def test_default_fusion_floor_follows_the_config():
    from jtsmm.doctor.imu_rates import default_min_fusion_hz

    doc = _sweep_doc()
    assert default_min_fusion_hz(doc) == pytest.approx(331.5)
    for slot in ("imu_mta", "imu_cra"):
        doc["mcu"][slot]["reports"]["game_rotation_vector"]["enabled"] = False
    assert default_min_fusion_hz(doc) is None


_STATUS = """imu_status
imu_mta: enabled=1 state=ok streaming=0 wanted=0 recovering=0 int=1 reports=2
  seq_gaps=246 resets=0 cont_pkts=3 trunc_pkts=0 i2c_timeouts=1
  i2c_tx_aborts=2 (NACK/arb-lost seen by the DMA reader; each costs a packet)
  report 0x08 req=390Hz granted=2500us (400.0Hz)
  report 0x01 req=400Hz granted=none
imu_cra: enabled=1 state=ok streaming=0 wanted=0 recovering=0 int=1 reports=2
  seq_gaps=440 resets=0 cont_pkts=0 trunc_pkts=0 i2c_timeouts=0
  i2c_tx_aborts=0 (NACK/arb-lost seen by the DMA reader; each costs a packet)
  report 0x08 req=390Hz granted=2500us (400.0Hz)
  report 0x01 req=400Hz granted=5882us (170.0Hz)
"""


def test_imu_status_parse_granted_and_link_growth():
    from jtsmm.doctor.imu_rates import StreamRate, apply_granted, link_growth, parse_imu_status

    st = parse_imu_status(_STATUS)
    assert st["mta"]["seq_gaps"] == 246 and st["mta"]["i2c_tx_aborts"] == 2
    assert st["mta"]["granted"] == {0x08: 2500, 0x01: None}
    assert st["cra"]["granted"][0x01] == 5882
    later = parse_imu_status(_STATUS.replace("seq_gaps=440", "seq_gaps=1440"))
    assert link_growth(st, later)["cra"]["seq_gaps"] == 1000
    assert link_growth(st, later)["mta"]["seq_gaps"] == 0
    streams = [StreamRate("cra", "accelerometer", 10, 170.0, 400.0, 0.0, False),
               StreamRate("mta", "accelerometer", 10, 170.0, 400.0, 0.0, False)]
    apply_granted(streams, later)
    assert streams[0].granted_hz == pytest.approx(170.0, abs=0.1)
    assert streams[1].granted_hz is None


def test_median_interval_rate_sees_through_missing_samples():
    from jtsmm.doctor.imu_rates import median_interval_hz

    ts = _ts(400, 400)
    lossy = [t for i, t in enumerate(ts) if i % 3]  # every third sample lost
    assert measured_hz(lossy) == pytest.approx(267, rel=0.01)
    assert median_interval_hz(lossy) == pytest.approx(400, rel=0.01)


def test_sweep_from_the_bench_prefers_the_request_that_matches_delivery():
    """Bench sweep 2026-10-06: 200 and 250 both deliver ~239 Hz; 250 wins the
    near-tie because it matches (200 would warn at +20 % in every take)."""
    from jtsmm.doctor.imu_rates import CaseResult, StreamRate, summarize_sweep

    bench = {100: (377, 125), 125: (379, 125), 200: (349, 239), 250: (351, 238),
             333: (393, 169), 400: (393, 170), 500: (395, 176)}
    results = []
    for req, (grv, acc) in bench.items():
        streams = [StreamRate(s, r, 1000, hz, 0.0, 0.0, False)
                   for s in ("mta", "cra")
                   for r, hz in (("game_rotation_vector", grv), ("accelerometer", acc))]
        results.append(CaseResult(label=str(req), duration_s=10.0, streams=streams))
    rows = summarize_sweep(results, "accelerometer", list(bench), min_fusion_hz=331.5)
    assert [r.requested_hz for r in rows if r.best] == [250]
    # Losses disqualify: the same 250 case with 1 % SH-2 gaps on CRA.
    results[3].link = {"cra": {"seq_gaps": 40}}
    rows = summarize_sweep(results, "accelerometer", list(bench), min_fusion_hz=331.5)
    assert rows[3].loss_frac == pytest.approx(40 / 2040)
    assert [r.requested_hz for r in rows if r.best] == [200]


def test_window_gaps_exclude_the_stream_start():
    """Bench 2026-10-06: game RV alone delivered 398.9 Hz of 400 granted, yet
    imu_status counted ~100 gaps since apply (reports queued before the stream
    started). Only gaps between the window's STATS records count."""
    from jtsmm.doctor.imu_rates import (
        CaseResult, StreamRate, summarize_sweep, window_seq_gaps)

    class _Rec:
        def __init__(self, t, gaps):
            self.record_type = int(RecordType.STATS)
            self.t_master_us = t
            self.decoded = {"imu_mta_seq_gaps": gaps, "imu_cra_seq_gaps": 7}

    class _Cap:
        def __init__(self):
            self.batches = [[_Rec(1_000_000, 120)], [_Rec(6_000_000, 122)]]

        def drain_records(self):
            return self.batches.pop(0) if self.batches else []

    window: dict = {}
    _, last = measure_capture(_Cap(), 0.05, window=window)
    gaps, span = window_seq_gaps(window, last)
    assert gaps == {"mta": 2, "cra": 0} and span == 5.0
    assert window_seq_gaps({}, last) == ({}, 0.0)

    st = [StreamRate("mta", "game_rotation_vector", 3990, 399.0, 390.0, 0.0, False)]
    res = CaseResult("grv", 10.0, st, link={"mta": {"seq_gaps": 124}},
                     window_gaps={"mta": 2}, window_span_s=5.0)
    row = summarize_sweep([res], "game_rotation_vector", [390], slot="mta")[0]
    assert row.loss_frac == pytest.approx(2 / (1995 + 2))
