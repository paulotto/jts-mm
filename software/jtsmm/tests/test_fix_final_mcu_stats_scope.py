# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MCU counters are graded per take: ``mcu_stats_end - mcu_stats_start``.

The firmware resets its counters at different moments (ring counters only at
boot, IMU seq gaps / resets never), so the end values alone blamed every take
for everything since the HAT booted.
"""

from __future__ import annotations

from pathlib import Path

from jtsmm.integrity.checker import check_session, take_scoped_mcu_stats

from tests.helpers import make_session_h5

# Counters as the HAT reports them after earlier takes since boot.
_BOOT = {
    "drdy_count": 3, "missed_drdy": 0, "emg_push": 250_000, "emg_drops": 40,
    "imu_mta_push": 90_000, "imu_mta_drops": 0, "imu_cra_push": 90_000,
    "imu_cra_drops": 0, "imu_cra_resets": 2, "imu_mta_seq_gaps": 300,
    "tx_drops": 0, "tx_drops_emg": 0, "tx_drops_imu": 0, "tx_drops_other": 0,
    "sensor_loop_hz": 9000, "misalign_max_streak": 4, "stats_seq": 10,
}


def _plus(base: dict, **inc: int) -> dict:
    out = dict(base)
    for k, v in inc.items():
        out[k] = (int(out.get(k, 0)) + v) & 0xFFFFFFFF
    return out


def test_deltas_keep_gauges_and_handle_u32_wrap():
    start = dict(_BOOT, usb_tx_bytes=0xFFFF_FF00, t_master_us=1_000)
    end = dict(_BOOT, emg_push=251_000, usb_tx_bytes=0x100, sensor_loop_hz=8000,
               misalign_max_streak=9, stats_seq=12, t_master_us=2_000_000)
    got, info = take_scoped_mcu_stats(end, start)
    assert info == {"scope": "take"}
    assert got["emg_push"] == 1000 and got["emg_drops"] == 0
    assert got["imu_cra_resets"] == 0 and got["imu_mta_seq_gaps"] == 0
    assert got["usb_tx_bytes"] == 0x200  # wrapped past 2**32
    # gauges and the record stamp are end values, not differences
    assert got["sensor_loop_hz"] == 8000 and got["misalign_max_streak"] == 9
    assert got["stats_seq"] == 12 and got["t_master_us"] == 2_000_000


def test_counter_reset_during_the_take_uses_its_end_value():
    got, info = take_scoped_mcu_stats({"missed_drdy": 3, "emg_push": 20}, {"missed_drdy": 500, "emg_push": 10})
    assert got["missed_drdy"] == 3 and got["emg_push"] == 10
    assert info["reset_during_take"] == ["missed_drdy"]


def test_without_start_snapshot_end_values_are_used_and_labelled():
    end = {"emg_push": 5}
    got, info = take_scoped_mcu_stats(end, None)
    assert got is end and info["scope"] == "since_boot"
    assert take_scoped_mcu_stats(None, {"emg_push": 1}) == (None, {"scope": None})


def test_earlier_takes_no_longer_warn(tmp_path: Path):
    path = tmp_path / "s.h5"
    start = dict(_BOOT, host_emg_rows_before=5)
    end = _plus(_BOOT, emg_push=995, drdy_count=995, imu_mta_push=400, imu_cra_push=400,
                stats_seq=1)
    make_session_h5(path, sync_extra={"mcu_stats_start": start, "mcu_stats_end": end})
    r = check_session(path)  # 1000 EMG rows, 5 of them before the snapshot
    ms = r.summary["mcu_stats"]
    assert ms["scope"] == "take" and ms["emg_push_unaccounted"] == 0
    assert not any("unaccounted" in w for w in r.warnings), r.warnings
    assert not any("BNO085 reset" in w for w in r.warnings), r.warnings
    assert not any("ring drop rate" in e for e in r.issues), r.issues

    # Same end values without the start snapshot: the old since-boot view.
    make_session_h5(path, sync_extra={"mcu_stats_end": end})
    r = check_session(path)
    assert r.summary["mcu_stats"]["scope"] == "since_boot"
    assert any("unaccounted" in w and "since the HAT booted" in w for w in r.warnings)
    assert any("imu_cra: BNO085 reset 2x" in w for w in r.warnings)


def test_imu_that_worked_before_but_not_this_take_is_dead(tmp_path: Path):
    path = tmp_path / "s.h5"
    end = _plus(_BOOT, emg_push=1000, imu_cra_push=400, stats_seq=1)  # MTA pushed nothing
    make_session_h5(path, sync_extra={"mcu_stats_start": dict(_BOOT), "mcu_stats_end": end})
    r = check_session(path)
    assert any(e.startswith("imu_mta: enabled but produced no samples") for e in r.issues), r.issues


def test_tx_drops_imu_fraction_uses_this_takes_pushes(tmp_path: Path):
    path = tmp_path / "s.h5"
    # tx_drops_* reset at ads_start, the IMU ring counters did not.
    end = _plus(_BOOT, emg_push=1000, imu_mta_push=200, imu_cra_push=200, tx_drops_imu=40,
                tx_drops=40, stats_seq=1)
    make_session_h5(path, sync_extra={"mcu_stats_start": dict(_BOOT), "mcu_stats_end": end})
    r = check_session(path)
    assert any("40 IMU records" in e and "budget" in e for e in r.issues), r.issues
