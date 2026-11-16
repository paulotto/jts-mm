# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Non-blocking single-key reads from an interactive TTY (SPACE / Q / ESC)."""

from __future__ import annotations

import select
import sys
import termios
import tty


class TerminalKeys:
    """cbreak-mode stdin poller used by experiment phases and CT pivot calib."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY")
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)

    def __enter__(self) -> TerminalKeys:
        # cbreak (not raw) keeps ISIG enabled so Ctrl+C still raises SIGINT.
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def poll(self) -> int | None:
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        ch = sys.stdin.read(1)
        if ch == " ":
            return ord(" ")
        if ch in ("q", "Q"):
            return ord("q")
        if ch == "\x1b":
            return 27
        if ch == "\x03":
            raise KeyboardInterrupt
        return None
