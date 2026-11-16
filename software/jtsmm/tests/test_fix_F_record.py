# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm record`: EMG-check restore, resolved_config to integrity, failed-take summary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from click.testing import CliRunner

from jtsmm.cli.record import record_cmd
from jtsmm.config import build_mcu_apply_payload
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


class _Ser:
    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def _patch_hardware_check(monkeypatch, calls: list) -> None:
    monkeypatch.setattr(
        "jtsmm.emg.check.run_emg_check",
        lambda resolved, **_kw: calls.append(("check", build_mcu_apply_payload(resolved)))
        or SimpleNamespace(passed=True),
    )
    monkeypatch.setattr("jtsmm.emg.check.format_check_table", lambda _r: "check table")
    monkeypatch.setattr("jtsmm.emg.check.hard_contact_failures", lambda _r: [])
    monkeypatch.setattr("jtsmm.emg.check.check_result_to_dict", lambda _r: {"passed": True})
    monkeypatch.setattr("jtsmm.capture.serial_util.find_hat_port", lambda _p: "/dev/fake_hat")
    monkeypatch.setattr("jtsmm.config.open_hat_serial", lambda _p: _Ser())
    monkeypatch.setattr("jtsmm.config.stop_mcu_streams", lambda _ser: calls.append(("stop",)))
    monkeypatch.setattr(
        "jtsmm.config.apply_mcu",
        lambda _ser, resolved, **_kw: calls.append(("restore", build_mcu_apply_payload(resolved))) or True,
    )


def _fake_run_session(tmp_path: Path, seen: dict):
    def run(config_path, **kw):
        seen.update(kw)
        out = tmp_path / "session_dir" / "session.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    return run


_OVERRIDES = ["--gain", "12", "--bias-channels", "1,2", "--active-channels", "1-3", "--no-leadoff"]


def test_emg_check_restore_runs_with_apply_disabled_and_keeps_overrides(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    doc = yaml.safe_load(tmp_config_path.read_text())
    doc["host"]["record"]["mcu_apply"]["enabled"] = False
    tmp_config_path.write_text(yaml.safe_dump(doc))

    calls: list[Any] = []
    seen: dict = {}
    _patch_hardware_check(monkeypatch, calls)
    monkeypatch.setattr("jtsmm.cli.record.run_session", _fake_run_session(tmp_path, seen))

    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), *MOCK_RECORD_DISABLE_OPTICAL, "--duration", "1",
         "--no-doctor", "--no-integrity", *_OVERRIDES],
    )
    assert result.exit_code == 0, result.output
    kinds = [c[0] for c in calls]
    assert kinds == ["check", "stop", "restore"], kinds
    restored = calls[2][1]["emg"]
    assert restored["default_gain"] == 12
    assert restored["bias"]["channels"] == 0b000011
    assert restored["pd_channels"] == 0b111000
    assert restored["leadoff"]["during_recording"] is False
    assert seen["meta"]["mcu_config_restored_after_emg_check"] == "true"
    # ... and the take itself is asked for the same overrides.
    assert seen["default_gain"] == 12 and seen["bias_channels"] == "1,2"
    assert seen["active_channels"] == "1-3" and seen["leadoff"] is False


def test_check_session_gets_the_resolved_config_and_its_source(
    tmp_config_path: Path, tmp_path: Path, monkeypatch
):
    from jtsmm.integrity import IntegrityReport

    got: dict = {}

    def check(path, *, resolved_config=None, config_source=None, **_kw):
        got["cfg"], got["src"] = resolved_config, config_source
        return IntegrityReport(path=path)

    monkeypatch.setattr("jtsmm.integrity.check_session", check)
    monkeypatch.setattr("jtsmm.cli.record.run_session", _fake_run_session(tmp_path, {}))
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", *MOCK_RECORD_DISABLE_OPTICAL,
         "--duration", "0.1", "--no-doctor", "--gain", "12"],
    )
    assert result.exit_code == 0, result.output
    assert got["cfg"]["mcu"]["emg"]["default_gain"] == 12
    assert got["src"] == str(tmp_config_path)


def test_failed_take_still_gets_an_integrity_summary(tmp_config_path: Path, tmp_path: Path, monkeypatch):
    from jtsmm.integrity import IntegrityReport

    partial = tmp_path / "aborted" / "session.h5"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"")

    def exploding_run_session(*_a, **_kw):
        exc = RuntimeError("camera failed to start")
        exc.jtsmm_session_h5 = partial  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr("jtsmm.cli.record.run_session", exploding_run_session)
    monkeypatch.setattr(
        "jtsmm.integrity.check_session", lambda path, **_kw: IntegrityReport(path=path)
    )
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", *MOCK_RECORD_DISABLE_OPTICAL,
         "--duration", "0.1", "--no-doctor"],
    )
    assert result.exit_code == 1, result.output
    assert "record failed: camera failed to start" in result.output
    doc = json.loads((partial.parent / "integrity.json").read_text(encoding="utf-8"))
    assert doc["overall"] == "fail"
    assert any("camera failed to start" in e for e in doc["checks"]["hdf5"]["errors"])
