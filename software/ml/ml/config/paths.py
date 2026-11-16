# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Resolve ML data directories (``~/ml_data``, complementary to ``~/jtsmm_data``)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Default tree under data_dir:
#   ~/ml_data/
#     processed/     windowed datasets
#     ml_runs/       training runs
#     ml_evals/      standalone eval artifacts
#     sessions/      optional mock / copied sessions for ML work
DEFAULT_ML_DATA_DIR = Path.home() / "ml_data"
DEFAULT_JTSMM_DATA_DIR = Path.home() / "jtsmm_data"
_ENV_VAR = "ML_DATA_DIR"
_JTSMM_ENV_VAR = "JTSMM_DATA_DIR"


def expand_path(value: str | Path, *, base: Path | None = None) -> Path:
    """Expand ``~``; resolve relative paths against ``base`` (or cwd if base is None).

    Legacy repo-relative prefixes (``data/``, ``software/``) resolve from the
    repository root so old configs keep working.
    """
    path = Path(os.path.expanduser(str(value)))
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in ("data", "software", "docs"):
        repo = Path(__file__).resolve().parents[4]
        return (repo / path).resolve()
    if base is not None:
        return (base / path).resolve()
    return path.resolve()


def session_data_dir(paths: dict[str, Any] | None = None) -> Path:
    """Live recording root for multi-session builds.

    Precedence: ``JTSMM_DATA_DIR`` > ``paths.session_data_dir`` > ``~/jtsmm_data``.
    """
    if env := os.environ.get(_JTSMM_ENV_VAR):
        return expand_path(env)
    if paths and paths.get("session_data_dir"):
        return expand_path(paths["session_data_dir"])
    return DEFAULT_JTSMM_DATA_DIR.resolve()


def ml_data_dir(paths: dict[str, Any] | None = None) -> Path:
    """Root ML data directory: ``ML_DATA_DIR`` > ``paths.data_dir`` > ``~/ml_data``."""
    if env := os.environ.get(_ENV_VAR):
        return expand_path(env)
    if paths and paths.get("data_dir"):
        return expand_path(paths["data_dir"])
    return DEFAULT_ML_DATA_DIR.resolve()


def processed_dir(paths: dict[str, Any] | None = None) -> Path:
    root = ml_data_dir(paths)
    rel = (paths or {}).get("processed_dir") or "processed"
    return expand_path(rel, base=root)


def runs_dir(paths: dict[str, Any] | None = None) -> Path:
    root = ml_data_dir(paths)
    rel = (paths or {}).get("runs_dir") or "ml_runs"
    return expand_path(rel, base=root)


def evals_dir(paths: dict[str, Any] | None = None) -> Path:
    root = ml_data_dir(paths)
    rel = (paths or {}).get("evals_dir") or "ml_evals"
    return expand_path(rel, base=root)


def sessions_dir(paths: dict[str, Any] | None = None) -> Path:
    """Optional ML-side session copies / mock sessions."""
    root = ml_data_dir(paths)
    rel = (paths or {}).get("sessions_dir") or "sessions"
    return expand_path(rel, base=root)


def ensure_ml_data_dirs(paths: dict[str, Any] | None = None) -> dict[str, Path]:
    """Create the standard ML data subdirs; return resolved paths."""
    out = {
        "data_dir": ml_data_dir(paths),
        "processed_dir": processed_dir(paths),
        "runs_dir": runs_dir(paths),
        "evals_dir": evals_dir(paths),
        "sessions_dir": sessions_dir(paths),
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def resolve_paths_section(doc: dict[str, Any] | None) -> dict[str, str]:
    """Normalize a monolithic config ``paths:`` block to absolute string paths."""
    raw = doc.get("paths") if isinstance(doc, dict) and isinstance(doc.get("paths"), dict) else {}
    raw = dict(raw or {})
    root = ml_data_dir(raw)
    return {
        "data_dir": str(root),
        "processed_dir": str(processed_dir(raw)),
        "runs_dir": str(runs_dir(raw)),
        "evals_dir": str(evals_dir(raw)),
        "sessions_dir": str(sessions_dir(raw)),
        "session_data_dir": str(session_data_dir(raw)),
    }
