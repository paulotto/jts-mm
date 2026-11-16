# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Session directory layout (s07)."""

from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jtsmm.config.registry import paths_data_dir

_FALLBACK_WARNED = False


def _dev_data_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "jtsmm" / "data"
    return Path.home() / ".local" / "share" / "jtsmm" / "data"


def _storage_base_usable(path: Path) -> bool:
    """True when path exists and is writable, or every missing parent can be created."""
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK)
    parent = path
    while not parent.exists():
        parent = parent.parent
        if parent == parent.parent:
            return False
    return os.access(parent, os.W_OK)


def resolve_storage_base(resolved: dict[str, Any]) -> Path:
    """Pick session output root: JTSMM_DATA_DIR > host.paths.data_dir > storage.base_dir."""
    global _FALLBACK_WARNED
    if env := os.environ.get("JTSMM_DATA_DIR"):
        base = Path(env)
        if not _storage_base_usable(base):
            raise PermissionError(
                f"JTSMM_DATA_DIR is not writable: {base} "
                f"(pick another directory or fix permissions)"
            )
        return base
    configured = Path(paths_data_dir(resolved)).expanduser()
    if _storage_base_usable(configured):
        return configured

    fallback = _dev_data_dir()
    if not _FALLBACK_WARNED:
        print(
            f"warning: storage base {configured} is not writable; "
            f"using {fallback} (set JTSMM_DATA_DIR to override)",
            file=sys.stderr,
        )
        _FALLBACK_WARNED = True
    return fallback


def experiment_slug(name: str | None) -> str | None:
    """Filesystem-safe fragment from ``session.experiment.name`` (or a loaded plan)."""
    if name is None:
        return None
    s = re.sub(r"[^\w\-]+", "_", str(name).strip().lower())
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s or s in {".", ".."}:
        return None
    return s[:80]


def experiment_name_from_config(resolved: dict[str, Any]) -> str | None:
    session = resolved.get("session")
    if not isinstance(session, dict):
        return None
    exp = session.get("experiment")
    if not isinstance(exp, dict):
        return None
    name = exp.get("name")
    if name is None:
        return None
    text = str(name).strip()
    return text or None


def session_dir(
    base_dir: Path,
    *,
    stamp: str | None = None,
    sid: str | None = None,
    experiment_name: str | None = None,
) -> Path:
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = sid or str(uuid.uuid4())[:8]
    slug = experiment_slug(experiment_name)
    if slug:
        return base_dir / f"session_{slug}_{stamp}_{sid}"
    return base_dir / f"session_{stamp}_{sid}"


def session_h5_path(session_directory: Path) -> Path:
    return session_directory / "session.h5"


def cam_mkv_path(session_directory: Path, cam_id: str) -> Path:
    return session_directory / f"{cam_id}.mkv"


PREVIEW_SUBDIR = "preview"
PREVIEW_INDEX_NAME = "preview_index.jsonl"


def preview_dir(session_directory: Path) -> Path:
    return session_directory / PREVIEW_SUBDIR


def preview_index_path(session_directory: Path) -> Path:
    return session_directory / PREVIEW_INDEX_NAME


def session_output_from_config(
    resolved: dict[str, Any], *, experiment_name: str | None = None
) -> tuple[Path, Path]:
    base = resolve_storage_base(resolved)
    name = experiment_name if experiment_name is not None else experiment_name_from_config(resolved)
    sdir = session_dir(base, experiment_name=name)
    return sdir, session_h5_path(sdir)


CT_CALIB_RECORDINGS_DEFAULT = (
    "~/experiments/5_2_1_pivot_calibration_repeatability/recordings"
)
IMU_CALIB_RECORDINGS_DEFAULT = "~/experiments/5_2_2_imu_optical_handeye/recordings"


def _calib_recording_dir(
    recordings_dir: Path | str | None,
    default: str,
    *,
    experiment_name: str | None,
    stamp: str | None = None,
) -> Path:
    """Run folder ``{slug}_{stamp}`` under a recordings base (optional ``{experiment_name}``)."""
    slug = experiment_slug(experiment_name) or "unnamed"
    template = str(recordings_dir) if recordings_dir else default
    expanded = template.replace("{experiment_name}", slug)
    base = Path(expanded).expanduser()
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{slug}_{stamp}"


def ct_calibration_recording_dir(
    recordings_dir: Path | str | None,
    *,
    experiment_name: str | None,
    stamp: str | None = None,
) -> Path:
    """Run folder for persisted CT tip calib poses/frames.

    ``recordings_dir`` may include ``{experiment_name}``. The leaf directory is
    always ``{slug}_{stamp}`` under the expanded base.
    """
    return _calib_recording_dir(
        recordings_dir,
        CT_CALIB_RECORDINGS_DEFAULT,
        experiment_name=experiment_name,
        stamp=stamp,
    )


def imu_calibration_recording_dir(
    recordings_dir: Path | str | None,
    *,
    experiment_name: str | None,
    stamp: str | None = None,
) -> Path:
    """Run folder for persisted IMU–optical hand-eye calibration takes.

    Same leaf layout as CT: ``{slug}_{stamp}`` under the recordings base.
    """
    return _calib_recording_dir(
        recordings_dir,
        IMU_CALIB_RECORDINGS_DEFAULT,
        experiment_name=experiment_name,
        stamp=stamp,
    )
