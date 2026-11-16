# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sidecar checksums in /meta/file_hashes_json are actually verified."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py

from jtsmm.integrity.checker import check_session

from tests.helpers import make_session_h5


def _session_with_sidecar(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "session.h5"
    make_session_h5(path)
    side = tmp_path / "preview_index.jsonl"
    side.write_text('{"frame": 1}\n', encoding="utf-8")
    table = {side.name: hashlib.sha256(side.read_bytes()).hexdigest()}
    with h5py.File(path, "a") as f:
        f["meta"].attrs["file_hashes_json"] = json.dumps(table)
    return path, side


def test_intact_sidecar_verifies(tmp_path: Path):
    path, _ = _session_with_sidecar(tmp_path)
    report = check_session(path)
    assert report.summary["sidecars"] == {"preview_index.jsonl": "ok"}
    assert report.ok, report.issues


def test_modified_sidecar_fails(tmp_path: Path):
    path, side = _session_with_sidecar(tmp_path)
    side.write_text('{"frame": 2}\n', encoding="utf-8")
    report = check_session(path)
    assert any("sidecar checksum mismatch for preview_index.jsonl" in e for e in report.issues)


def test_missing_sidecar_warns(tmp_path: Path):
    path, side = _session_with_sidecar(tmp_path)
    side.unlink()
    report = check_session(path)
    assert report.summary["sidecars"] == {"preview_index.jsonl": "missing"}
    assert any("preview_index.jsonl listed in file_hashes_json is missing" in w
               for w in report.warnings)
