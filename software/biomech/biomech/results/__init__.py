# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result store (lazy: importing ``biomech.results.provenance`` must not pull pandas)."""

from __future__ import annotations

from typing import Any

__all__ = ["COHORT_UNIT", "ResultStore", "RunLock", "Status", "UnitMeta"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from biomech.results import store

        return getattr(store, name)
    raise AttributeError(f"module 'biomech.results' has no attribute {name!r}")
