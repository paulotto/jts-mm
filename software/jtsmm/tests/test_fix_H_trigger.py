# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm trigger`: firmware bounds, and report what the MCU actually runs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from jtsmm.cli.trigger import (
    build_trig_start_command,
    parse_trig_status,
    send_trigger_command,
    trigger_cmd,
)

STATUS_30HZ = (
    "trig running=1 id=12\r\n"
    "  stamp: configured=pio active=pio fallback=no\r\n"
    "  comp=on period_us=33333 pin_period_us=33333 edge=exposure_start(low)\r\n"
    "  pio_pulses=12 timer_events=0 spurious_irq=0\r\n"
)


@pytest.mark.parametrize("rate", [0, -1, 201, 65536 + 30])
def test_rate_outside_firmware_range_rejected(rate):
    with pytest.raises(ValueError, match="1-200 Hz"):
        build_trig_start_command(rate=rate)


@pytest.mark.parametrize("rate,width", [(30, 33333), (30, 40000), (200, 5000), (60, 0)])
def test_width_must_be_below_period(rate, width):
    with pytest.raises(ValueError, match="below the"):
        build_trig_start_command(rate=rate, width_us=width)


def test_bounds_inclusive():
    assert build_trig_start_command(rate=200, width_us=4999) == b"trig_start 200 4999\r\n"
    assert build_trig_start_command(rate=1) == b"trig_start 1\r\n"


def test_parse_trig_status_ignores_pin_period():
    assert parse_trig_status(STATUS_30HZ) == {"running": 1, "period_us": 33333}
    assert parse_trig_status("") is None


def test_cli_prints_read_back_rate():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value=STATUS_30HZ) as send:
        result = runner.invoke(trigger_cmd, [])
    assert result.exit_code == 0, result.output
    assert send.call_args.kwargs.get("then_status") is True
    assert "30.000 Hz (period 33333 us, read back)" in result.output
    assert "MCU config rate" not in result.output


def test_cli_fails_when_trigger_not_running():
    runner = CliRunner()
    reply = STATUS_30HZ.replace("running=1", "running=0")
    with patch("jtsmm.cli.trigger.send_trigger_command", return_value=reply):
        result = runner.invoke(trigger_cmd, ["--rate", "30"])
    assert result.exit_code != 0
    assert "NOT running" in result.output


def test_cli_rejects_out_of_range_before_touching_serial():
    runner = CliRunner()
    with patch("jtsmm.cli.trigger.send_trigger_command") as send:
        result = runner.invoke(trigger_cmd, ["--rate", "500"])
    assert result.exit_code != 0
    send.assert_not_called()


def test_send_then_status_reads_status_on_same_connection():
    fake = MagicMock()
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    replies = [b"", STATUS_30HZ.encode()]
    type(fake).in_waiting = property(lambda self: len(replies[0]) if replies else 0)

    def _read(n):
        return replies.pop(0)

    fake.read.side_effect = _read

    def _write(data):
        if data == b"trig_status\r\n":
            replies.pop(0)  # drop the empty start reply

    fake.write.side_effect = _write
    with (
        patch("jtsmm.cli.trigger.find_hat_port", return_value="/dev/fake"),
        patch("jtsmm.cli.trigger.serial.Serial", return_value=fake),
        patch("jtsmm.cli.trigger.time.sleep", return_value=None),
    ):
        text = send_trigger_command(
            b"trig_start\r\n", read_reply=True, then_status=True, timeout_s=0.05
        )
    assert [c.args[0] for c in fake.write.call_args_list] == [b"trig_start\r\n", b"trig_status\r\n"]
    assert parse_trig_status(text) == {"running": 1, "period_us": 33333}


def _serial_answering(dump: bytes):
    """Fake CDC port that answers `config dump` with ``dump``."""
    fake = MagicMock()
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    pending = bytearray()
    type(fake).in_waiting = property(lambda self: len(pending))

    def _read(n):
        out = bytes(pending[:n])
        del pending[:n]
        return out

    def _write(data):
        if data == b"config dump\r\n":
            pending.extend(dump)

    fake.read.side_effect = _read
    fake.write.side_effect = _write
    return fake


def _patched(fake):
    return (
        patch("jtsmm.cli.trigger.find_hat_port", return_value="/dev/fake"),
        patch("jtsmm.cli.trigger.serial.Serial", return_value=fake),
        patch("jtsmm.cli.trigger.time.sleep", return_value=None),
    )


def test_rate_without_width_rejected_when_configured_width_too_long():
    fake = _serial_answering(b'{"trigger":{"rate_hz":30,"pulse":{"width_us":12000}}}\r\n')
    p1, p2, p3 = _patched(fake)
    with p1, p2, p3:
        result = CliRunner().invoke(trigger_cmd, ["--rate", "100"])
    assert result.exit_code != 0
    assert "12000 us is not below the 10000 us period" in result.output
    assert b"trig_start 100\r\n" not in [c.args[0] for c in fake.write.call_args_list]


def test_rate_without_width_allowed_when_configured_width_fits():
    fake = _serial_answering(b'{"trigger":{"rate_hz":30,"pulse":{"width_us":4000}}}\r\n')
    p1, p2, p3 = _patched(fake)
    with p1, p2, p3:
        send_trigger_command(b"trig_start 100\r\n", width_check_rate=100)
    assert fake.write.call_args_list[-1].args[0] == b"trig_start 100\r\n"


def test_cli_status_prints_the_whole_block():
    """--status shows the counters the bench check needs, not just the first line."""
    block = STATUS_30HZ.replace("spurious_irq=0", "spurious_irq=0 lost_edges=0 irq_level=1")
    fake = MagicMock()
    fake.__enter__ = MagicMock(return_value=fake)
    fake.__exit__ = MagicMock(return_value=False)
    pending = bytearray()
    type(fake).in_waiting = property(lambda self: min(len(pending), 20))  # arrives in chunks

    def _read(n):
        out = bytes(pending[:n])
        del pending[:n]
        return out

    def _write(data):
        if data == b"trig_status\r\n":
            pending.extend(block.encode())

    fake.read.side_effect = _read
    fake.write.side_effect = _write
    p1, p2, p3 = _patched(fake)
    with p1, p2, p3:
        result = CliRunner().invoke(trigger_cmd, ["--status"])
    assert result.exit_code == 0, result.output
    for field in ("trig running=1", "fallback=no", "pio_pulses=12", "lost_edges=0"):
        assert field in result.output
