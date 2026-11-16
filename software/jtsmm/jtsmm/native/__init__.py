# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optional native (C++/libusb/libhdf5) capture + storage backend.

The compiled extension ``_jtsmm_native`` lives beside this file once built (see
``software/jtsmm/native/build.sh``). USB bulk capture + frame parsing always use
native code when the extension is built (``use_native()``). The libhdf5 writer is
used for sensor-only sessions; multimodal sessions with cameras use native USB
capture plus the Python HDF5 writer for camera metadata rows.

``JTSMM_NATIVE=0`` disables the extension; it does NOT switch to a pure-Python
USB path. Real (non-mock) recording, ``jtsmm emg check``, the live viewer source
and ``jtsmm doctor`` imu-rates then refuse with a RuntimeError; only mock
sessions and offline tools run without it. (``capture.capture_mcu`` is a
test/diagnostic reader, too slow for a full-rate take.)
"""

from __future__ import annotations

import os

try:  # pragma: no cover - import guard depends on build state
    from . import _jtsmm_native as _ext  # type: ignore[attr-defined]

    FrameParser = _ext.FrameParser
    NativeMcuCapture = _ext.NativeMcuCapture
    NativeHdf5Writer = getattr(_ext, "NativeHdf5Writer", None)
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
    _ext = None
    FrameParser = None  # type: ignore[assignment]
    NativeMcuCapture = None  # type: ignore[assignment]
    NativeHdf5Writer = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


def is_available() -> bool:
    """True if the compiled native extension imported successfully."""
    return _ext is not None


def import_error() -> Exception | None:
    """The exception raised while importing the extension, if any (for diagnostics)."""
    return _IMPORT_ERROR


def use_native() -> bool:
    """Whether the native path should be used: available and not disabled.

    ``JTSMM_NATIVE`` overrides autodetection — ``0`` disables native (callers
    that need live USB capture then refuse, see the module docstring), ``1``
    requires it.
    """
    flag = os.environ.get("JTSMM_NATIVE")
    if flag is not None:
        return flag not in ("0", "false", "False", "")
    return is_available()


__all__ = [
    "FrameParser",
    "NativeMcuCapture",
    "NativeHdf5Writer",
    "is_available",
    "import_error",
    "use_native",
]
