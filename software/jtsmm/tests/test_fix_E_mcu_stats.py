# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MCU loss counters from ``sync.mcu_stats_end`` (incl. K2 per-type drops)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py

from jtsmm.integrity.checker import check_session

from tests.helpers import make_session_h5


def _stats(path: Path, stats: dict | None) -> None:
    with h5py.File(path, "a") as f:
        doc = json.loads(f["sync"].attrs["report_json"])
        if stats is None:
            doc.pop("mcu_stats_end", None)
        else:
            doc["mcu_stats_end"] = stats
        f["sync"].attrs["report_json"] = json.dumps(doc)


def test_absent_stats_warn_and_summary_says_unknown(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path)
    _stats(path, None)
    report = check_session(path)
    assert any("no mcu_stats_end" in w for w in report.warnings), report.warnings
    assert report.summary["mcu_stats"] == {"present": False}
    assert all(v is None for v in report.summary["emg_framing"].values())


def test_per_type_tx_drops_are_gated(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path)  # 1000 EMG rows
    base = {"missed_drdy": 0, "drdy_count": 1000, "emg_drops": 0, "imu_mta_push": 400,
            "imu_cra_push": 400, "imu_mta_drops": 0, "imu_cra_drops": 0,
            "tx_drops": 0, "tx_drops_emg": 0, "tx_drops_imu": 0, "tx_drops_other": 0,
            "stats_seq": 17}
    _stats(path, dict(base, tx_drops_emg=2, tx_drops=2))
    r = check_session(path)
    assert any("2 EMG records at the USB FIFO" in w for w in r.warnings), r.warnings
    assert r.ok, r.issues
    assert r.summary["mcu_stats"]["stats_seq"] == 17

    _stats(path, dict(base, tx_drops_emg=50, tx_drops_imu=40, tx_drops=90))
    r = check_session(path)
    assert any("50 EMG records" in e and "budget" in e for e in r.issues), r.issues
    assert any("40 IMU records" in e and "budget" in e for e in r.issues), r.issues


def test_legacy_tx_drops_without_types_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"mcu_stats_end": {"tx_drops": 12}})
    r = check_session(path)
    assert any("tx_drops; firmware without per-type" in w for w in r.warnings), r.warnings


def test_imu_seq_gaps_count_as_true_loss_and_resets_warn(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"mcu_stats_end": {
        "imu_mta_push": 1000, "imu_mta_seq_gaps": 20, "imu_cra_push": 1000,
        "imu_cra_resets": 1}})
    r = check_session(path)
    assert any(e.startswith("imu_mta ring drop rate") for e in r.issues), r.issues
    assert any("imu_cra: BNO085 reset 1x" in w for w in r.warnings)


def test_emg_push_cross_check(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra={"mcu_stats_end": {"emg_push": 1300, "emg_drops": 0}})
    r = check_session(path)
    assert r.summary["mcu_stats"]["emg_push_unaccounted"] == 300
    assert any("+300 unaccounted" in w for w in r.warnings), r.warnings
