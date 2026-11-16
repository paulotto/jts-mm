# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""`jtsmm record` CLI: --meta/--subject/--protocol parsing and --gain wiring."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
from click.testing import CliRunner

from jtsmm.cli.record import _parse_meta, record_cmd
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


def test_parse_meta_basic():
    assert _parse_meta(("operator=jd", "note=first session")) == {
        "operator": "jd",
        "note": "first session",
    }


def test_parse_meta_subject_and_protocol():
    meta = _parse_meta((), subject="P07", protocol="motion_battery_v1")
    assert meta == {"subject_id": "P07", "protocol": "motion_battery_v1"}


def test_parse_meta_rejects_missing_equals():
    with pytest.raises(Exception):
        _parse_meta(("no_equals_sign",))


def test_parse_meta_rejects_reserved_key():
    with pytest.raises(Exception):
        _parse_meta(("config_hash=deadbeef",))
    with pytest.raises(Exception):
        _parse_meta(("subject_id=P07",))  # must use --subject, not --meta


def test_record_cmd_mock_writes_gain_and_meta(tmp_config_path: Path, tmp_path: Path):
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
            "--no-integrity",
            "--gain",
            "12",
            "--subject",
            "P07",
            "--protocol",
            "motion_battery_v1",
            "--meta",
            "operator=jd",
        ],
    )
    assert result.exit_code == 0, result.output

    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    out_path = Path(out_line.split("-> ", 1)[1].strip())
    with h5py.File(out_path, "r") as f:
        mcu_cfg = json.loads(f["meta"].attrs["mcu_config_json"])
        assert int(mcu_cfg["emg"]["default_gain"]) == 12
        assert f["meta"].attrs["subject_id"] == "P07"
        assert f["meta"].attrs["protocol"] == "motion_battery_v1"
        assert f["meta"].attrs["operator"] == "jd"
        assert "recording_date" in f["meta"].attrs


def test_record_cmd_prints_formatted_integrity_report(tmp_config_path: Path):
    """With --no-integrity omitted, `jtsmm record` runs the checker and prints
    the nicely formatted summary table, not the old flat warning:/error: lines."""
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert "jtsmm session integrity" in result.output
    assert "Wrote integrity summary ->" in result.output
    assert "RESULT: OK" in result.output or "overall     OK" in result.output

    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    session_dir = Path(out_line.split("-> ", 1)[1].strip()).parent
    assert (session_dir / "integrity.json").is_file()
    assert (session_dir / "integrity.txt").is_file()
    doc = json.loads((session_dir / "integrity.json").read_text(encoding="utf-8"))
    assert doc["overall"] in ("ok", "ok_with_warnings")
    assert doc["checks"]["hdf5"]["ran"] is True
    assert doc["checks"]["preflight"]["skipped"] is True
    assert doc["checks"]["emg"]["skipped"] is True


def test_record_cmd_aborts_with_exit_2_on_integrity_error(tmp_config_path: Path, monkeypatch):
    """A genuinely broken session must fail loudly (distinct exit code 2)
    with the formatted report still shown, not just a silent pass."""
    from jtsmm.integrity import IntegrityReport

    def _fake_check_session(path, **_kwargs):
        report = IntegrityReport(path=path)
        report.add_error("EMG t_master_us not strictly monotonic")
        return report

    monkeypatch.setattr("jtsmm.integrity.check_session", _fake_check_session)
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
        ],
    )
    assert result.exit_code == 2, result.output
    assert "jtsmm session integrity" in result.output
    assert "ERRORS (1)" in result.output or "error: EMG t_master_us" in result.output
    assert "FAIL" in result.output
    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    session_dir = Path(out_line.split("-> ", 1)[1].strip()).parent
    doc = json.loads((session_dir / "integrity.json").read_text(encoding="utf-8"))
    assert doc["overall"] == "fail"
    assert doc["checks"]["hdf5"]["ok"] is False


def test_record_cmd_rejects_bad_meta(tmp_config_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", "--meta", "config_hash=x"],
    )
    assert result.exit_code != 0


def test_record_cmd_no_longer_has_single_process_flag(tmp_config_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", "--single-process"],
    )
    assert result.exit_code != 0
    assert "no such option" in result.output.lower()


def test_record_cmd_mock_skips_emg_check_even_when_enabled(tmp_config_path: Path):
    """`--emg-check` is on by default, but it needs real hardware (native
    capture + serial) -- `--mock` must always skip it, not hang/fail trying
    to open a port, and must not write an emg_check_json meta attr."""
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
            "--no-integrity",
            "--emg-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "EMG check" not in result.output

    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    out_path = Path(out_line.split("-> ", 1)[1].strip())
    with h5py.File(out_path, "r") as f:
        assert "emg_check_json" not in f["meta"].attrs
        assert "emg_check_passed" not in f["meta"].attrs


def test_record_cmd_accepts_no_emg_check_flag(tmp_config_path: Path):
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
            "--no-integrity",
            "--no-emg-check",
        ],
    )
    assert result.exit_code == 0, result.output


def test_record_preflight_info_and_warn_do_not_abort(tmp_config_path: Path, monkeypatch):
    """Viewer doctor info lines must not block record (regression)."""

    def _fake_preflight(**_kwargs):
        return [
            "warn: imx296 trigger_mode sysfs not present (camera module not loaded?)",
            "info: viewer Wi-Fi radio enabled",
            "info: viewer local URL http://127.0.0.1:8770/",
            "info: viewer LAN URL http://192.168.1.47:8770/",
            "info: viewer LAN URL http://jtsmm.local:8770/",
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
            "--no-integrity",
            "--no-emg-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "error: info:" not in result.output
    assert "warning: imx296" in result.output
    assert "viewer local URL" in result.output


def test_record_preflight_hard_error_aborts(tmp_config_path: Path, monkeypatch):
    monkeypatch.setattr(
        "jtsmm.cli.record.run_preflight",
        lambda **_kw: ["firmware version mismatch"],
    )
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
            "--no-integrity",
            "--no-emg-check",
        ],
    )
    assert result.exit_code == 1
    assert "error: firmware version mismatch" in result.output


def test_no_abort_keeps_exit_zero_on_an_integrity_error(tmp_config_path: Path, monkeypatch):
    """--no-abort: report the failure, write everything, do not stop the run.

    The file and integrity.json are written either way -- the integrity check
    runs on the closed HDF5 -- so this only changes the exit code and lets a
    batch of takes continue.
    """
    from jtsmm.integrity import IntegrityReport

    def _fake_check_session(path, **_kwargs):
        report = IntegrityReport(path=path)
        report.add_error("EMG t_master_us not strictly monotonic")
        return report

    monkeypatch.setattr("jtsmm.integrity.check_session", _fake_check_session)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config", str(tmp_config_path), "--mock",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--duration", "0.1", "--no-doctor", "--no-emg-check", "--no-abort",
        ],
    )
    assert result.exit_code == 0, result.output
    # still visible, and still on disk
    assert "quality gates will report but not stop" in result.output
    assert "not" in result.output and "aborting" in result.output
    assert "EMG t_master_us" in result.output
    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    session_dir = Path(out_line.split("-> ", 1)[1].strip()).parent
    assert (session_dir / "integrity.json").is_file()


def test_abort_on_error_false_in_config_matches_the_flag(tmp_config_path: Path, monkeypatch):
    import yaml as _yaml
    from jtsmm.integrity import IntegrityReport

    doc = _yaml.safe_load(tmp_config_path.read_text())
    doc["host"]["record"]["abort_on_error"] = False
    tmp_config_path.write_text(_yaml.safe_dump(doc))

    def _fake_check_session(path, **_kwargs):
        report = IntegrityReport(path=path)
        report.add_error("EMG t_master_us not strictly monotonic")
        return report

    monkeypatch.setattr("jtsmm.integrity.check_session", _fake_check_session)
    result = CliRunner().invoke(
        record_cmd,
        [
            "--config", str(tmp_config_path), "--mock",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--duration", "0.1", "--no-doctor", "--no-emg-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "quality gates will report but not stop" in result.output
