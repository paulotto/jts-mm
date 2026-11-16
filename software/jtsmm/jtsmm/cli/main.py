# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""CLI entrypoints: record, doctor, config apply."""

from __future__ import annotations

import click

from .config_cmd import config_group
from .ct_cmd import ct_calibrate_cmd
from .doctor import doctor_cmd
from .emg_cmd import emg_group
from .imu_cmd import imu_group
from .info import info_cmd
from .integrity import integrity_cmd
from .record import record_cmd
from .trigger import trigger_cmd
from .view import replay_cmd, view_cmd


@click.group()
@click.version_option(package_name="jtsmm")
def main() -> None:
    """JTS-MM production host capture stack."""


main.add_command(record_cmd)
main.add_command(ct_calibrate_cmd)
main.add_command(doctor_cmd)
main.add_command(info_cmd)
main.add_command(integrity_cmd)
main.add_command(config_group)
main.add_command(emg_group)
main.add_command(imu_group)
main.add_command(view_cmd)
main.add_command(replay_cmd)
main.add_command(trigger_cmd)


if __name__ == "__main__":
    main()
