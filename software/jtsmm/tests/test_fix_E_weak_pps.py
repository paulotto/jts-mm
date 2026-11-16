# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Assignment-residual gates downgrade when the PPS fit cannot back them."""

from __future__ import annotations

from pathlib import Path

from jtsmm.integrity.checker import check_session

from tests.helpers import make_session_h5

_BAD = {"n": 500, "median_us": -333_320.0, "p10_us": 0.0, "p90_us": 0.0,
        "median_frames": -10.0, "spread_frames": 4.0, "drift_us_per_s": 900.0}
_OK_PASS = {"attempted": 8, "accepted": 8}


def _session(tmp_path: Path, **extra) -> Path:
    path = tmp_path / "s.h5"
    sync = {"cameras": {"cam0": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0,
                                 "assign_residual": dict(_BAD)}}}
    sync.update(extra)
    make_session_h5(path, sync_extra=sync)
    return path


def _assign_errors(report) -> list[str]:
    return [e for e in report.issues if "camera cam0" in e and (
        "inherited the offset" in e or "drift" in e or "not constant" in e)]


def test_full_pps_keeps_errors(tmp_path: Path):
    r = check_session(_session(tmp_path, pps={"start": _OK_PASS, "end": _OK_PASS},
                               pi_to_master={"pps_samples": 16, "mode": "free_slope"}))
    assert len(_assign_errors(r)) == 3, r.issues


def test_missing_end_pass_downgrades(tmp_path: Path):
    r = check_session(_session(tmp_path, pps={"start": _OK_PASS},
                               pi_to_master={"pps_samples": 8}))
    assert not _assign_errors(r), r.issues
    assert any("downgraded to warnings: no end-of-take PPS pass" in w for w in r.warnings)
    assert any("inherited the offset" in w for w in r.warnings)


def test_no_pps_samples_downgrades(tmp_path: Path):
    r = check_session(_session(tmp_path, pi_to_master={"pps_samples": 0}))
    assert not _assign_errors(r), r.issues
    assert r.summary["sync"]["assign_gates_downgraded"].startswith("no PPS samples")


def test_end_pass_without_replies_downgrades(tmp_path: Path):
    r = check_session(_session(tmp_path, pps={"start": _OK_PASS,
                                              "end": {"attempted": 8, "accepted": 0}}))
    assert not _assign_errors(r), r.issues


def test_legacy_file_without_pps_info_keeps_errors(tmp_path: Path):
    r = check_session(_session(tmp_path))
    assert len(_assign_errors(r)) == 3
