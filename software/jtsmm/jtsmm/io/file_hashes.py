# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Session file SHA-256 hashes for integrity checking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_file_hashes(paths: dict[str, Path]) -> str:
    """Return JSON dict mapping basename -> sha256 hex."""
    table = {name: sha256_file(p) for name, p in paths.items() if p.is_file()}
    return json.dumps(table, separators=(",", ":"), sort_keys=True)
