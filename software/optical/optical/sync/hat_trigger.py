# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Arm the MCU camera trigger for live optical capture.

IMX296 cameras stay in external-trigger mode, so every live optical command
needs the HAT pulsing. This module runs the same commands as:

    jtsmm trigger --rate <mcu.trigger.rate_hz from jtsmm.yaml, default 30>
    jtsmm trigger --stop

The context manager is reentrant: nested live paths (CLI wrapper + PicamRig
or ``HatSyncBridge``) start/stop the pulse once.

``hat_trigger(defer=True)`` holds the pulse back so a caller can start the
cameras first and then :func:`arm_hat_trigger` — the ``trigger_after_cameras``
order ``jtsmm record`` uses. With the trigger running first, the camera that
starts later misses pulses the other one exposed, and frames pair up wrongly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Fallback when jtsmm.yaml cannot be read; matches mcu.trigger.rate_hz there
# and the cameras' trigger.expect_rate_hz.
DEFAULT_RATE_HZ = 30
# optical calibrate always pulses at this rate. It copies full-res frames one
# camera after another; a 60 Hz period (16.7 ms) is shorter than that copy, so
# the two SensorTimestamps never land on the same trigger. A still board does
# not need the operator rate from jtsmm.yaml.
CALIBRATE_TRIGGER_RATE_HZ = 30
SKIP_ENV = "OPTICAL_SKIP_HAT_TRIGGER"

_lock = threading.Lock()
_depth = 0
_armed = False
_rate_hz: int | None = None
_port: str | None = None


def configured_trigger_rate_hz(config_path: str | Path | None = None) -> int:
    """``mcu.trigger.rate_hz`` from the jtsmm config (``DEFAULT_RATE_HZ`` if unreadable)."""
    try:
        from jtsmm.config import load

        doc = load(config_path, resolve=True, validate=False, strict=False)
        rate = int((doc.get("mcu") or {}).get("trigger", {}).get("rate_hz") or 0)
        if rate > 0:
            return rate
    except Exception:
        pass
    return DEFAULT_RATE_HZ


def _jtsmm_executable() -> str:
    found = shutil.which("jtsmm")
    if found:
        return found
    sibling = Path(sys.executable).resolve().parent / "jtsmm"
    if os.access(sibling, os.X_OK):
        return str(sibling)
    raise FileNotFoundError(
        "jtsmm not found on PATH; install software/jtsmm and activate the venv "
        "so live optical commands can run `jtsmm trigger`"
    )


def invoke_jtsmm_trigger(*args: str) -> None:
    """Run ``jtsmm trigger …`` and raise if it fails."""
    cmd = [_jtsmm_executable(), "trigger", *args]
    proc = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.stdout:
        sys.stdout.write(proc.stdout)
        if not proc.stdout.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    if proc.returncode != 0:
        detail = (proc.stdout or "").strip()
        extra = f": {detail}" if detail else ""
        raise RuntimeError(f"{' '.join(cmd)} failed (exit {proc.returncode}){extra}")


def _skip() -> bool:
    return bool(os.environ.get(SKIP_ENV))


def _port_args() -> tuple[str, ...]:
    return ("--port", _port) if _port else ()


def start_hat_trigger(
    *,
    rate_hz: int | None = None,
    defer: bool = False,
    port: str | None = None,
) -> None:
    """Start ``jtsmm trigger --rate <hz>`` unless already armed in this process.

    ``rate_hz=None`` uses :func:`configured_trigger_rate_hz`. ``defer=True``
    takes the reference without pulsing; call :func:`arm_hat_trigger` later.
    ``port`` is passed as ``jtsmm trigger --port`` (default: auto-detect).
    """
    global _depth, _armed, _rate_hz, _port
    if _skip():
        return
    if rate_hz is not None and int(rate_hz) <= 0:
        raise ValueError("rate_hz must be > 0")
    with _lock:
        if _depth == 0:
            rate = int(rate_hz) if rate_hz is not None else configured_trigger_rate_hz()
            _rate_hz = rate
            _port = str(port) if port else None
            _armed = False
            if not defer:
                invoke_jtsmm_trigger("--rate", str(rate), *_port_args())
                _armed = True
        _depth += 1


def arm_hat_trigger() -> bool:
    """Start the pulse held back by ``defer=True``; returns True if it was started now."""
    global _armed
    if _skip():
        return False
    with _lock:
        if _depth <= 0:
            raise RuntimeError("arm_hat_trigger() outside hat_trigger()")
        if _armed:
            return False
        invoke_jtsmm_trigger("--rate", str(int(_rate_hz or DEFAULT_RATE_HZ)), *_port_args())
        _armed = True
        return True


def active_trigger_rate_hz() -> int | None:
    """Rate of the trigger this process holds (``None`` outside ``hat_trigger``)."""
    with _lock:
        return _rate_hz if _depth > 0 else None


def stop_hat_trigger() -> None:
    """Stop the trigger when the last nested live path finishes."""
    global _depth, _armed, _rate_hz, _port
    if _skip():
        return
    with _lock:
        if _depth <= 0:
            return
        _depth -= 1
        if _depth == 0:
            port_args = _port_args()
            _armed = False
            _rate_hz = None
            _port = None
            # Also sent when a deferred trigger was never armed: harmless, and
            # it stops a pulse train left running by an earlier crash.
            invoke_jtsmm_trigger("--stop", *port_args)


def reset_hat_trigger_state() -> None:
    """Test helper: drop nested depth without sending CDC commands."""
    global _depth, _armed, _rate_hz, _port
    with _lock:
        _depth = 0
        _armed = False
        _rate_hz = None
        _port = None


@contextmanager
def hat_trigger(
    *,
    rate_hz: int | None = None,
    defer: bool = False,
    port: str | None = None,
) -> Iterator[None]:
    """Run ``jtsmm trigger --rate <hz>`` for the block, then ``jtsmm trigger --stop``."""
    start_hat_trigger(rate_hz=rate_hz, defer=defer, port=port)
    try:
        yield
    finally:
        try:
            stop_hat_trigger()
        except Exception as exc:
            print(f"warning: jtsmm trigger --stop failed: {exc}", file=sys.stderr)
