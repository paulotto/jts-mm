# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Session integrity summary + `jtsmm integrity check` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

from jtsmm.cli.integrity import integrity_cmd
from jtsmm.cli.main import main
from jtsmm.cli.record import record_cmd
from jtsmm.integrity import check_session
from jtsmm.integrity.summary import (
    INTEGRITY_JSON_NAME,
    INTEGRITY_TXT_NAME,
    SessionIntegritySummary,
    format_session_summary,
    load_emg_from_session_h5,
    resolve_session_h5,
    set_emg_check,
    set_hdf5,
    set_preflight,
    write_session_integrity,
)
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL, make_session_h5
from tests.test_session_reproduce import _force_tmp_storage


def test_resolve_session_h5_accepts_file_and_dir(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    assert resolve_session_h5(h5) == h5
    assert resolve_session_h5(tmp_path) == h5


def test_resolve_session_h5_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_session_h5(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_session_h5(tmp_path / "nope.h5")


def test_write_session_integrity_includes_all_checks(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    report = check_session(h5)

    summary = SessionIntegritySummary(session_dir=tmp_path)
    set_preflight(summary, warnings=["imx296 trigger_mode sysfs not present"], infos=["viewer ok"])
    set_emg_check(
        summary,
        result={"passed": True, "note": "", "channels": [], "drl": None},
        passed=True,
    )
    set_hdf5(summary, report)

    written = write_session_integrity(tmp_path, summary)
    assert written[INTEGRITY_JSON_NAME].is_file()
    assert written[INTEGRITY_TXT_NAME].is_file()

    doc = json.loads(written[INTEGRITY_JSON_NAME].read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["ok"] is True
    assert doc["overall"] in ("ok", "ok_with_warnings")
    assert set(doc["checks"]) == {"preflight", "emg", "hdf5"}
    assert doc["checks"]["preflight"]["warnings"]
    assert doc["checks"]["emg"]["ok"] is True
    assert doc["checks"]["hdf5"]["ran"] is True

    text = written[INTEGRITY_TXT_NAME].read_text(encoding="utf-8")
    assert "jtsmm session integrity" in text
    assert "preflight" in text
    assert "emg" in text
    assert "hdf5" in text


def test_summary_overall_fail_when_any_stage_fails(tmp_path: Path):
    summary = SessionIntegritySummary(session_dir=tmp_path)
    set_preflight(summary, errors=[], warnings=[])
    set_emg_check(
        summary,
        result={
            "passed": False,
            "note": "channel fail",
            "channels": [
                {
                    "channel": 1,
                    "name": "VL",
                    "ok": False,
                    "notes": ["noise high"],
                    "leadoff_ok": True,
                }
            ],
            "drl": None,
        },
        passed=False,
    )
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    set_hdf5(summary, check_session(h5))
    assert summary.ok is False
    assert summary.overall == "fail"
    assert summary.emg.errors
    written = write_session_integrity(tmp_path, summary)
    doc = json.loads(written[INTEGRITY_JSON_NAME].read_text(encoding="utf-8"))
    assert doc["overall"] == "fail"
    assert doc["checks"]["emg"]["ok"] is False


def test_summary_overall_fail_when_hdf5_fails(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=1.0, cam_trailing_gap_s=3.0, cam_ids=("cam0",))
    report = check_session(h5)
    assert not report.ok, "fixture must fail HDF5 checks"
    summary = SessionIntegritySummary(session_dir=tmp_path)
    set_preflight(summary, skipped=True, skip_reason="n/a")
    set_emg_check(summary, skipped=True, skip_reason="n/a")
    set_hdf5(summary, report)
    assert summary.overall == "fail"
    doc = json.loads(
        write_session_integrity(tmp_path, summary)[INTEGRITY_JSON_NAME].read_text(
            encoding="utf-8"
        )
    )
    assert doc["checks"]["hdf5"]["errors"]


def test_format_session_summary_marks_skipped(tmp_path: Path):
    summary = SessionIntegritySummary(session_dir=tmp_path)
    set_preflight(summary, skipped=True, skip_reason="--no-doctor")
    set_emg_check(summary, skipped=True, skip_reason="--mock")
    summary.hdf5.ran = True
    summary.hdf5.skipped = True
    summary.hdf5.skip_reason = "--no-integrity"
    text = format_session_summary(summary)
    assert "skipped" in text
    assert "--mock" in text


def test_load_emg_from_session_h5_roundtrip(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    payload = {
        "passed": False,
        "note": "bench fail",
        "channels": [{"channel": 2, "name": "RF", "ok": False, "notes": ["loff"]}],
        "drl": None,
    }
    with h5py.File(h5, "a") as f:
        f["meta"].attrs["emg_check_json"] = json.dumps(payload)
        f["meta"].attrs["emg_check_passed"] = "false"

    detail, passed = load_emg_from_session_h5(h5)
    assert detail is not None
    assert detail["passed"] is False
    assert passed is False

    runner = CliRunner()
    result = runner.invoke(integrity_cmd, ["check", "--write-summary", str(tmp_path)])
    # HDF5 itself is healthy; EMG section comes from /meta.
    assert result.exit_code == 0, result.output
    doc = json.loads((tmp_path / INTEGRITY_JSON_NAME).read_text(encoding="utf-8"))
    assert doc["checks"]["emg"]["ran"] is True
    assert doc["checks"]["emg"]["skipped"] is False
    assert doc["checks"]["emg"]["ok"] is False
    assert doc["overall"] == "fail"


def test_integrity_check_cli_accepts_directory(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    runner = CliRunner()
    result = runner.invoke(integrity_cmd, ["check", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "jtsmm integrity report" in result.output


def test_integrity_check_cli_fails_on_broken_session(tmp_path: Path):
    h5 = tmp_path / "bad.h5"
    make_session_h5(h5, duration_s=1.0, noncontiguous_triggers=True)
    runner = CliRunner()
    result = runner.invoke(main, ["integrity", "check", str(h5)])
    assert result.exit_code == 1, result.output
    assert "error" in result.output.lower() or "FAIL" in result.output


def test_integrity_check_cli_strict_fails_on_warnings(tmp_path: Path):
    h5 = tmp_path / "warn.h5"
    make_session_h5(h5, duration_s=1.0, omit_samples_uv=True)
    runner = CliRunner()
    ok = runner.invoke(integrity_cmd, ["check", str(h5)])
    assert ok.exit_code == 0, ok.output
    strict = runner.invoke(integrity_cmd, ["check", "--strict", str(h5)])
    assert strict.exit_code == 1, strict.output


def test_integrity_check_cli_write_summary(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    runner = CliRunner()
    result = runner.invoke(integrity_cmd, ["check", "--write-summary", str(h5)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / INTEGRITY_JSON_NAME).is_file()
    assert (tmp_path / INTEGRITY_TXT_NAME).is_file()
    assert "Wrote integrity summary" in result.output
    doc = json.loads((tmp_path / INTEGRITY_JSON_NAME).read_text(encoding="utf-8"))
    assert doc["checks"]["hdf5"]["ran"] is True
    assert doc["checks"]["preflight"]["skipped"] is True


def test_integrity_check_via_main_group(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)
    runner = CliRunner()
    result = runner.invoke(main, ["integrity", "check", str(h5)])
    assert result.exit_code == 0, result.output


def test_integrity_group_help_without_subcommand():
    runner = CliRunner()
    result = runner.invoke(main, ["integrity"])
    assert result.exit_code == 0
    assert "check" in result.output


def test_record_writes_integrity_summary_even_with_no_integrity(
    tmp_config_path: Path, tmp_path: Path
):
    """`--no-integrity` skips the HDF5 pass but must still leave a summary
    documenting that preflight/EMG/HDF5 were skipped or ran."""
    data_root = tmp_path / "data"
    _force_tmp_storage(tmp_config_path, data_root)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--duration",
            "0.1",
            "--no-doctor",
            "--no-emg-check",
            "--no-integrity",
        ],
        env={"JTSMM_DATA_DIR": str(data_root)},
    )
    assert result.exit_code == 0, result.output
    out_line = next(
        line for line in result.output.splitlines() if line.startswith("Wrote session")
    )
    session_dir = Path(out_line.split("-> ", 1)[1].strip()).parent
    doc = json.loads((session_dir / INTEGRITY_JSON_NAME).read_text(encoding="utf-8"))
    assert doc["checks"]["hdf5"]["skipped"] is True
    assert "--no-integrity" in (doc["checks"]["hdf5"].get("skip_reason") or "")
    assert doc["checks"]["preflight"]["skipped"] is True
    assert doc["checks"]["emg"]["skipped"] is True


def test_record_integrity_summary_captures_preflight_warnings(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    data_root = tmp_path / "data"
    _force_tmp_storage(tmp_config_path, data_root)

    def _fake_preflight(**_kwargs):
        return [
            "warn: imx296 trigger_mode sysfs not present",
            "info: viewer local URL http://127.0.0.1:8770/",
        ]

    monkeypatch.setattr("jtsmm.cli.record.run_preflight", _fake_preflight)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--duration",
            "0.1",
            "--no-emg-check",
        ],
        env={"JTSMM_DATA_DIR": str(data_root)},
    )
    assert result.exit_code == 0, result.output
    out_line = next(
        line for line in result.output.splitlines() if line.startswith("Wrote session")
    )
    session_dir = Path(out_line.split("-> ", 1)[1].strip()).parent
    doc = json.loads((session_dir / INTEGRITY_JSON_NAME).read_text(encoding="utf-8"))
    assert doc["checks"]["preflight"]["ran"] is True
    assert doc["checks"]["preflight"]["skipped"] is False
    assert any("imx296" in w for w in doc["checks"]["preflight"]["warnings"])
    assert doc["checks"]["hdf5"]["ran"] is True
    assert doc["checks"]["emg"]["skipped"] is True


def test_guards_fail_when_summary_writer_broken(tmp_path: Path, monkeypatch):
    """Regression proof: the happy-path ``is_file()`` assertions fail if the
    writer returns paths without creating the files."""
    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5)

    def _broken_write(session_directory, summary):
        return {
            INTEGRITY_JSON_NAME: Path(session_directory) / INTEGRITY_JSON_NAME,
            INTEGRITY_TXT_NAME: Path(session_directory) / INTEGRITY_TXT_NAME,
        }

    monkeypatch.setattr(
        "jtsmm.integrity.summary.write_session_integrity", _broken_write
    )
    from jtsmm.integrity import summary as summary_mod

    summary = SessionIntegritySummary(session_dir=tmp_path)
    set_hdf5(summary, check_session(h5))
    written = summary_mod.write_session_integrity(tmp_path, summary)
    with pytest.raises(AssertionError):
        assert written[INTEGRITY_JSON_NAME].is_file()
    with pytest.raises(AssertionError):
        assert written[INTEGRITY_TXT_NAME].is_file()
