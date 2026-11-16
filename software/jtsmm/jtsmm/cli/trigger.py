# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""jtsmm trigger — start/stop/status the MCU camera trigger only."""

from __future__ import annotations

import re
import time

import click
import serial

from jtsmm.capture.serial_util import find_hat_port
from jtsmm.config.catalog import TRIGGER_RATE_HZ_MAX, TRIGGER_RATE_HZ_MIN, trigger_period_us


def build_trig_start_command(*, rate: int | None = None, width_us: int | None = None) -> bytes:
    """Build the CDC line for ``trig_start`` (optional rate / pulse width).

    Bounds mirror the firmware: rate_hz is a uint16 capped at 200 by
    trigger_config_validate (a larger value would wrap), and a width at or
    beyond the period is silently replaced by period/2 in trigger_timing.c.
    """
    if width_us is not None and rate is None:
        raise ValueError("--width-us requires --rate")
    if rate is not None:
        if not TRIGGER_RATE_HZ_MIN <= rate <= TRIGGER_RATE_HZ_MAX:
            raise ValueError(
                f"--rate must be {TRIGGER_RATE_HZ_MIN}-{TRIGGER_RATE_HZ_MAX} Hz (got {rate})"
            )
        if width_us is not None:
            period = trigger_period_us(rate)
            if not 0 < width_us < period:
                raise ValueError(
                    f"--width-us must be > 0 and below the {period} us period at {rate} Hz "
                    f"(got {width_us})"
                )
            return f"trig_start {rate} {width_us}\r\n".encode("ascii")
        return f"trig_start {rate}\r\n".encode("ascii")
    return b"trig_start\r\n"


def parse_trig_status(text: str) -> dict[str, int] | None:
    """Pull ``running`` / ``period_us`` out of the firmware's ``trig_status`` lines."""
    run = re.search(r"trig running=(\d+)", text)
    period = re.search(r"\bperiod_us=(\d+)", text)
    if run is None or period is None:
        return None
    return {"running": int(run.group(1)), "period_us": int(period.group(1))}


def send_trigger_command(
    cmd: bytes,
    port: str | None = None,
    *,
    read_reply: bool = False,
    timeout_s: float = 1.0,
    then_status: bool = False,
    width_check_rate: int | None = None,
) -> str:
    """Open HAT CDC, send one command, optionally read a short reply. Trigger-only.

    ``then_status`` also sends ``trig_status`` on the same connection and appends
    its reply, so a start can be confirmed from what the MCU actually runs.
    ``width_check_rate``: before sending, read the MCU's configured pulse width
    (`config dump`) and raise ``ValueError`` if it is not below the period at that
    rate -- the firmware would silently halve it. No answer: send anyway.
    """
    serial_port = find_hat_port(port)
    with serial.Serial(serial_port, 115200, timeout=0.05) as ser:
        ser.dtr = False
        time.sleep(0.05)
        ser.dtr = True
        time.sleep(0.2)
        ser.reset_input_buffer()
        if width_check_rate is not None:
            _check_configured_width(ser, width_check_rate)
        ser.write(cmd)
        ser.flush()
        if not read_reply:
            return ""
        if cmd == b"trig_status\r\n":
            return _read_trig_status(ser, timeout_s)
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            n = ser.in_waiting
            if n:
                buf.extend(ser.read(n))
                text = bytes(buf).decode("utf-8", errors="replace")
                if (
                    "trig running=" in text
                    or "trig_start FAIL" in text
                    or "trig_start BUSY" in text
                    or text.lstrip().startswith("unknown:")
                ):
                    break
            else:
                time.sleep(0.02)
        text = bytes(buf).decode("utf-8", errors="replace")
        if (
            not then_status
            or "FAIL" in text
            or "BUSY" in text
            or text.lstrip().startswith("unknown:")
        ):
            return text
        ser.write(b"trig_status\r\n")
        ser.flush()
        return text + _read_trig_status(ser, timeout_s)


def _read_trig_status(ser: serial.Serial, timeout_s: float) -> str:
    """Read the whole ``trig_status`` block (trigger_pio_print_status).

    It ends with the ``pio_pulses=`` line, optionally followed by a WARNING line,
    so after that line keep reading briefly.
    """
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))
            if re.search(rb"pio_pulses=[^\n]*\n", bytes(buf)):
                deadline = min(deadline, time.monotonic() + 0.05)
        else:
            time.sleep(0.02)
    return buf.decode("utf-8", errors="replace")


def _check_configured_width(ser: serial.Serial, rate: int) -> None:
    from jtsmm.config.mcu_apply import read_config_dump

    dump = read_config_dump(ser, timeout_s=1.0) or {}
    width = ((dump.get("trigger") or {}).get("pulse") or {}).get("width_us")
    period = trigger_period_us(rate)
    if isinstance(width, (int, float)) and width >= period:
        raise ValueError(
            f"the MCU's configured pulse width {int(width)} us is not below the {period} us "
            f"period at {rate} Hz (the firmware would silently use {period // 2} us); "
            "pass --width-us"
        )


@click.command("trigger")
@click.option("--port", default=None, help="J_USB1 serial port (auto-detect if omitted)")
@click.option(
    "--rate", type=int, default=None, help="Trigger rate override, 1-200 Hz (default: MCU config)"
)
@click.option(
    "--width-us",
    type=int,
    default=None,
    help="Pulse width in microseconds, below the period 1e6/rate (requires --rate)",
)
@click.option("--stop", "do_stop", is_flag=True, help="Stop the trigger (trig_stop only)")
@click.option("--status", "do_status", is_flag=True, help="Query trig_status")
def trigger_cmd(
    port: str | None,
    rate: int | None,
    width_us: int | None,
    do_stop: bool,
    do_status: bool,
) -> None:
    """Arm/disarm the MCU camera trigger only (no EMG/IMU, no config apply)."""
    if do_stop and do_status:
        raise SystemExit("error: use only one of --stop or --status")
    if (do_stop or do_status) and (rate is not None or width_us is not None):
        raise SystemExit("error: --rate/--width-us cannot be combined with --stop/--status")

    try:
        if do_stop:
            send_trigger_command(b"trig_stop\r\n", port)
            click.echo("trigger stopped")
            return
        if do_status:
            reply = send_trigger_command(b"trig_status\r\n", port, read_reply=True)
            text = reply.strip() or "(no reply)"
            start = text.find("trig running=")
            click.echo(text[start:] if start >= 0 else text)
            return

        cmd = build_trig_start_command(rate=rate, width_us=width_us)
        reply = send_trigger_command(
            cmd,
            port,
            read_reply=True,
            then_status=True,
            width_check_rate=rate if rate is not None and width_us is None else None,
        )
        text = reply.strip()
        if text.startswith("unknown:") or "trig_start FAIL" in text or "trig_start BUSY" in text:
            hint = ""
            if "trig_start BUSY" in text:
                hint = " (a config apply is still running; retry when it has finished)"
            elif rate is not None or width_us is not None:
                hint = (
                    " (this firmware may not support trig_start rate/width overrides; "
                    "run `jtsmm trigger` without --rate after `jtsmm config apply`, "
                    "or reflash MCU firmware)"
                )
            raise SystemExit(f"error: MCU rejected trigger start: {text}{hint}")
        _report_started(text, rate=rate, width_us=width_us)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc


def _report_started(reply: str, *, rate: int | None, width_us: int | None) -> None:
    """Print what the MCU reports it is running, not what was asked for."""
    status = parse_trig_status(reply)
    if status is None:
        click.echo(
            "warning: trigger start sent but trig_status could not be read back; "
            "check with `jtsmm trigger --status`",
            err=True,
        )
        return
    if not status["running"]:
        raise SystemExit("error: MCU reports the trigger is NOT running after trig_start")
    period = status["period_us"]
    actual_hz = 1e6 / period if period else float("nan")
    if width_us is not None:
        width = f"{width_us} us"
    else:
        width = "MCU config value (trig_status does not report the width)"
    click.echo(
        f"trigger running: {actual_hz:.3f} Hz (period {period} us, read back); pulse {width}"
    )
    if rate is not None and period != trigger_period_us(rate):
        click.echo(f"warning: requested {rate} Hz but the MCU runs a {period} us period", err=True)
    if rate is not None and width_us is None:
        click.echo(
            "note: the pulse width is the MCU's configured one (checked against the new "
            f"{trigger_period_us(rate)} us period when `config dump` answered)",
            err=True,
        )


if __name__ == "__main__":
    trigger_cmd.main(standalone_mode=True)
