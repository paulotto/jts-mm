# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stable SHA-256 hash of resolved configuration."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(resolved: dict[str, Any]) -> str:
    """Return hex SHA-256 of canonical resolved config (MCU + host + session)."""
    payload = _canonical_json(resolved)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
