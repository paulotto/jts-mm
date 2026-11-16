# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Tests for ``jtsmm trigger`` — CDC payloads only; no config apply / streams."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from jtsmm.cli.main import main
from jtsmm.cli.trigger import build_trig_start_command, send_trigger_command, trigger_cmd


def test_build_trig_start_bare():
    assert build_trig_start_command() == b"trig_start\r\n"


def test_build_trig_start_rate():
    assert build_trig_start_command(rate=60) == b"trig_start 60\r\n"


def test_build_trig_start_rate_width():
    assert build_trig_start_command(rate=60, width_us=4000) == b"trig_start 60 4000\r\n"


def test_build_trig_start_width_without_rate_errors():
    with pytest.raises(ValueError, match="--width-us requires --rate"):
        build_trig_start_command(width_us=4000)


def test_cli_default_start_writes_trig_start():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value="") as send:
        result = runner.invoke(trigger_cmd, [])
    assert result.exit_code == 0, result.output
    send.assert_called_once()
    assert send.call_args.args[0] == b"trig_start\r\n"
    assert send.call_args.kwargs.get("read_reply") is True


def test_cli_rate_60():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value="") as send:
        result = runner.invoke(trigger_cmd, ["--rate", "60"])
    assert result.exit_code == 0, result.output
    send.assert_called_once()
    assert send.call_args.args[0] == b"trig_start 60\r\n"
    assert send.call_args.kwargs.get("read_reply") is True


def test_cli_rate_60_mcu_unknown_fails():
    runner = CliRunner()
    with patch(
        "jtsmm.cli.trigger.send_trigger_command",
        return_value="unknown: trig_start 60\r\n",
    ):
        result = runner.invoke(trigger_cmd, ["--rate", "60"])
    assert result.exit_code != 0
    assert "MCU rejected" in result.output
    assert "unknown: trig_start 60" in result.output


def test_cli_start_while_config_apply_busy_fails():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value="trig_start BUSY\r\n"):
        result = runner.invoke(trigger_cmd, ["--rate", "60"])
    assert result.exit_code != 0
    assert "MCU rejected" in result.output
    assert "config apply is still running" in result.output
    assert "may not support" not in result.output


def test_cli_rate_and_width():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value="") as send:
        result = runner.invoke(trigger_cmd, ["--rate", "60", "--width-us", "4000"])
    assert result.exit_code == 0, result.output
    assert send.call_args.args[0] == b"trig_start 60 4000\r\n"


def test_cli_stop():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command") as send:
        result = runner.invoke(trigger_cmd, ["--stop"])
    assert result.exit_code == 0, result.output
    assert send.call_args.args[0] == b"trig_stop\r\n"


def test_cli_status():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value="trig running=0 id=0\r\n") as send:
        result = runner.invoke(trigger_cmd, ["--status"])
    assert result.exit_code == 0, result.output
    assert send.call_args.args[0] == b"trig_status\r\n"
    assert "trig running=0" in result.output


def test_cli_stop_with_rate_rejected():
    runner = CliRunner()
    result = runner.invoke(trigger_cmd, ["--stop", "--rate", "60"])
    assert result.exit_code != 0
    assert "--rate" in result.output or "error" in result.output.lower()


def test_send_trigger_never_calls_apply_or_streams(monkeypatch):
    """Regression: trigger path must not touch apply_mcu / start_mcu_streams."""
    calls: list[str] = []

    def _boom(name: str):
        def _inner(*_a, **_k):
            calls.append(name)
            raise AssertionError(f"{name} must not be called")

        return _inner

    monkeypatch.setattr("jtsmm.config.apply_mcu", _boom("apply_mcu"), raising=False)
    monkeypatch.setattr("jtsmm.config.start_mcu_streams", _boom("start_mcu_streams"), raising=False)
    monkeypatch.setattr("jtsmm.config.stop_mcu_streams", _boom("stop_mcu_streams"), raising=False)

    fake_ser = MagicMock()
    fake_ser.__enter__ = MagicMock(return_value=fake_ser)
    fake_ser.__exit__ = MagicMock(return_value=False)
    fake_ser.in_waiting = 0

    with (
        patch("jtsmm.cli.trigger.find_hat_port", return_value="/dev/fake_hat"),
        patch("jtsmm.cli.trigger.serial.Serial", return_value=fake_ser),
        patch("jtsmm.cli.trigger.time.sleep", return_value=None),
    ):
        send_trigger_command(b"trig_start\r\n")

    fake_ser.write.assert_called_once_with(b"trig_start\r\n")
    assert calls == []


def test_main_registers_trigger():
    runner = CliRunner()
    result = runner.invoke(main, ["trigger", "--help"])
    assert result.exit_code == 0
    assert "trig_start" in result.output or "camera trigger" in result.output.lower()
