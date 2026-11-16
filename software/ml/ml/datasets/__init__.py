# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""File-based dataset building (``ml dataset``) on top of :mod:`ml.core`."""

from .builder import build_dataset, build_pool, mvc_reference_uv, phase_labels
from .discover import SessionRef, resolve_session_inputs, sidecars_in

__all__ = ["SessionRef", "build_dataset", "build_pool", "mvc_reference_uv", "phase_labels",
           "resolve_session_inputs", "sidecars_in"]
