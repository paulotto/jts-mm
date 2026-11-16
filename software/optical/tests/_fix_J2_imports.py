# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Import pure-numpy optical modules on hosts without OpenCV (test helper).

``optical/__init__.py`` imports ``optical.tracking`` (→ cv2), so on a venv
without OpenCV even numpy-only modules such as ``optical.calibration.geometry``
fail to import. When that happens, import the requested submodules through a
temporary bare ``optical`` package and remove it again afterwards, so other
tests still see the real import behaviour.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

_OPTICAL_DIR = Path(__file__).resolve().parents[1] / "optical"


def import_optical(*names: str) -> list[types.ModuleType]:
    try:
        import optical  # noqa: F401
    except ImportError:
        pass
    else:
        return [importlib.import_module(n) for n in names]

    before = set(sys.modules)
    pkg = types.ModuleType("optical")
    pkg.__path__ = [str(_OPTICAL_DIR)]  # type: ignore[attr-defined]
    sys.modules["optical"] = pkg
    try:
        return [importlib.import_module(n) for n in names]
    finally:
        for key in set(sys.modules) - before:
            if key == "optical" or key.startswith("optical."):
                del sys.modules[key]
        sys.modules.pop("optical", None)
