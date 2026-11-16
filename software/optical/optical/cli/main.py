# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""CLI entrypoints: calibrate, track, replay, markers, eval."""

from __future__ import annotations

import functools

import click

from optical.sync.hat_trigger import CALIBRATE_TRIGGER_RATE_HZ

from .calibrate import calibrate_cmd
from .eval_cmd import eval_cmd, eval_quality_cmd
from .markers import markers_cmd
from .replay import replay_cmd
from .track import track_cmd


def _with_hat_trigger(cmd: click.Command, *, rate_hz: int | None = None) -> click.Command:
    """Run ``jtsmm trigger`` around a live optical command.

    ``rate_hz`` overrides ``mcu.trigger.rate_hz``. ``None`` uses the configured rate.
    """
    inner = cmd.callback
    if inner is None:
        return cmd

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):
        # Offline / non-HAT paths: bundle export, eval-quality --frames-dir, v4l2.
        if (
            kwargs.get("mode") == "bundle"
            or kwargs.get("frames_dir") is not None
            or str(kwargs.get("backend") or "").lower() == "v4l2"
        ):
            return inner(*args, **kwargs)
        from optical.sync.hat_trigger import hat_trigger

        with hat_trigger(rate_hz=rate_hz):
            return inner(*args, **kwargs)

    cmd.callback = wrapper
    return cmd


@click.group()
@click.version_option(package_name="optical")
def main() -> None:
    """ArUco multi-camera jaw tracking (Qualisys replacement)."""


main.add_command(_with_hat_trigger(calibrate_cmd, rate_hz=CALIBRATE_TRIGGER_RATE_HZ))
# track arms the trigger itself, after picamera2 is streaming (tracking/run.py).
main.add_command(track_cmd)
main.add_command(replay_cmd)
main.add_command(markers_cmd)
main.add_command(_with_hat_trigger(eval_cmd))
main.add_command(_with_hat_trigger(eval_quality_cmd))


if __name__ == "__main__":
    main()
