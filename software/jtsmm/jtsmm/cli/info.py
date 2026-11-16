# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm info — MCU firmware version and port."""

from __future__ import annotations

import json

import click

from jtsmm.doctor.mcu_info import query_mcu_info


@click.command("info")
@click.option("--port", default=None, help="J_USB1 serial port (auto-detect if omitted)")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON")
def info_cmd(port: str | None, as_json: bool) -> None:
    """Query firmware version on the HAT MCU."""
    try:
        info = query_mcu_info(port)
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from exc

    if as_json:
        click.echo(
            json.dumps(
                {
                    "port": info.port,
                    "firmware_version": info.firmware_version,
                    "raw_banner": info.raw_banner.strip(),
                },
                indent=2,
            )
        )
    else:
        click.echo(f"port: {info.port}")
        click.echo(f"firmware_version: {info.firmware_version}")


if __name__ == "__main__":
    info_cmd.main(standalone_mode=True)
