# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Repository layout and config path resolution for the optical package.

Operator config lives under ``software/optical/etc/`` (same idea as ``jtsmm.yaml``):

- ``software/optical/etc/optical.yaml`` — auto 1- or 2-camera profile at runtime
- ``software/optical/etc/ct_tip_calibration.json`` — reusable CT tip offset
- ``software/optical/etc/jts_config.json`` — JTS analysis template
- ``~/optical.yaml`` — optional operator override of the default
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optical.config.loader import default_config_path, load_config_file, package_etc_dir

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent.parent

DOCS_DIR = REPO_ROOT / "docs" / "optical"
ETC_DIR = package_etc_dir()
DATA_DIR = REPO_ROOT / "data" / "optical"
CALIBRATIONS_DIR = DATA_DIR / "calibrations"
TRACKING_DIR = DATA_DIR / "tracking"
REPLAY_DIR = DATA_DIR / "replay"

DEFAULTS_FILE = ETC_DIR / "optical.yaml"

RECORDINGS_DIR_DEFAULT = "~/experiments/{experiment_folder}/recordings"
SESSION_H5_NAME = "session.h5"
# Snapshotted into each --persist-frames leaf for offline reproducibility.
CALIBRATION_BUNDLE_SNAPSHOT_NAME = "calibration_bundle.h5"
OPTICAL_CONFIG_SNAPSHOT_NAME = "optical.yaml"


def resolve_repo_path(relative: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_config_path(relative: str | Path) -> Path:
    """Resolve a path relative to ``software/optical/etc/`` (or repo root).

    Bare filenames (e.g. ``optical.yaml``) resolve under ``etc/``.
    Paths that start with ``software/`` or ``data/`` resolve from the repo root.
    """
    path = Path(relative)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in ("software", "data"):
        return REPO_ROOT / path
    return ETC_DIR / path


def load_defaults() -> dict:
    """Load the default optical.yaml document."""
    return load_config_file(default_config_path())


def default_path(key: str, *, fallback_key: str | None = None) -> Path:
    """
    Resolve a logical config key to the default optical.yaml.

    Historical keys (``marker_config``, ``tracker_config``, …) all map to the
    same etc file; sections are selected by the loaders.
    """
    _ = key, fallback_key
    return default_config_path()


def calibration_bundle_path(rig_id: str) -> Path:
    """Default HDF5 calibration bundle path for a rig."""
    return CALIBRATIONS_DIR / rig_id / "calibration_bundle.h5"


def tracking_session_path(experiment: str) -> Path:
    """Default HDF5 tracking session output, named by experiment."""
    return TRACKING_DIR / f"{experiment}.h5"


def path_slug(text: str | None, *, fallback: str = "unnamed") -> str:
    """Filesystem-safe fragment for experiment recording leaves."""
    s = re.sub(r"[^\w\-]+", "_", str(text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s or s in {".", ".."}:
        return fallback
    return s[:80]


def experiment_recording_dir(
    recordings_dir: Path | str | None,
    *,
    experiment_folder: str | None,
    experiment_name: str | None,
    condition: str | None,
    stamp: str | None = None,
) -> Path:
    """Run folder for persisted optical track frames + ``session.h5``.

    ``recordings_dir`` may include ``{experiment_folder}`` / ``{experiment_name}``.
    The leaf is always ``{name}_{condition}_{stamp}`` under the expanded base.
    """
    folder = path_slug(experiment_folder, fallback="optical_experiment")
    name = path_slug(experiment_name, fallback="unnamed")
    cond = path_slug(condition, fallback="run")
    template = str(recordings_dir) if recordings_dir else RECORDINGS_DIR_DEFAULT
    expanded = (
        template.replace("{experiment_folder}", folder).replace("{experiment_name}", name)
    )
    base = Path(expanded).expanduser()
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base / f"{name}_{cond}_{stamp}"


def resolve_persist_recording(
    doc: dict[str, Any],
    *,
    experiment_name: str | None,
    condition: str,
    stamp: str | None = None,
) -> Path:
    """Resolve the persist leaf from a loaded optical.yaml document."""
    paths = doc.get("paths") if isinstance(doc.get("paths"), dict) else {}
    exp = doc.get("experiment") if isinstance(doc.get("experiment"), dict) else {}
    recordings_dir = paths.get("recordings_dir") if isinstance(paths, dict) else None
    folder = exp.get("folder") if isinstance(exp, dict) else None
    name = experiment_name if experiment_name is not None else (
        exp.get("name") if isinstance(exp, dict) else None
    )
    return experiment_recording_dir(
        recordings_dir,
        experiment_folder=str(folder) if folder is not None else None,
        experiment_name=str(name) if name is not None else None,
        condition=condition,
        stamp=stamp,
    )


def recording_calibration_bundle(session_dir: Path) -> Path | None:
    """Return leaf-local ``calibration_bundle.h5`` if present."""
    path = Path(session_dir) / CALIBRATION_BUNDLE_SNAPSHOT_NAME
    return path if path.is_file() else None


def recording_optical_config(session_dir: Path) -> Path | None:
    """Return leaf-local ``optical.yaml`` snapshot if present."""
    path = Path(session_dir) / OPTICAL_CONFIG_SNAPSHOT_NAME
    return path if path.is_file() else None


def snapshot_persist_calibration(
    session_dir: Path,
    *,
    bundle_path: Path,
    config_path: Path | None = None,
) -> dict[str, Path]:
    """Copy the active calibration bundle (and optional config) into a persist leaf.

    Offline eval/replay then prefer these snapshots over whatever is currently
    configured in ``optical.yaml``, so takes stay reproducible after recalibration.
    """
    import shutil

    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)

    src_bundle = Path(bundle_path)
    if not src_bundle.is_file():
        raise FileNotFoundError(f"calibration bundle not found: {src_bundle}")

    dest_bundle = session_dir / CALIBRATION_BUNDLE_SNAPSHOT_NAME
    shutil.copy2(src_bundle, dest_bundle)
    out: dict[str, Path] = {"calibration_bundle": dest_bundle}

    if config_path is not None:
        src_cfg = Path(config_path)
        if src_cfg.is_file():
            dest_cfg = session_dir / OPTICAL_CONFIG_SNAPSHOT_NAME
            shutil.copy2(src_cfg, dest_cfg)
            out["optical_config"] = dest_cfg

    return out
