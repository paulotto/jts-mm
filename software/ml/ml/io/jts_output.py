# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Read JTS analysis output HDF5 (jaw_motion.h5 or embedded /jts/ in session.h5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

DEFAULT_JTS_GROUP = "T_model_origin_mand_landmark_t"
PREFERRED_JTS_GROUPS = (
    "T_model_origin_mand_landmark_t_smooth",
    "T_model_origin_mand_landmark_t",
)


@dataclass(frozen=True)
class JtsSource:
    """Resolved location of JTS trajectory data."""

    path: Path
    parent_group: str | None = None
    trajectory_group: str = DEFAULT_JTS_GROUP

    @property
    def display_path(self) -> str:
        if self.parent_group:
            return f"{self.path}:/{self.parent_group}/{self.trajectory_group}"
        return str(self.path)


@dataclass
class JtsTrajectory:
    """Mandibular landmark trajectory from JTS analysis."""

    path: Path
    group_name: str
    translations: np.ndarray  # (N, 3)
    translations_smooth: np.ndarray  # (N, 3)
    sample_rate: float
    unit: str
    derivatives: dict[str, np.ndarray]
    metadata: dict[str, Any]
    parent_group: str | None = None


def find_embedded_jts_group(
    session_path: str | Path,
    *,
    parent_group: str = "jts",
) -> str | None:
    """Return trajectory group name under ``parent_group`` if embedded JTS exists."""
    session_path = Path(session_path)
    if not session_path.is_file():
        return None
    with h5py.File(session_path, "r") as f:
        if parent_group not in f:
            return None
        root = f[parent_group]
        for name in PREFERRED_JTS_GROUPS:
            if name in root and "translations" in root[name]:
                return name
    return None


def load_jts_trajectory(
    path: str | Path,
    *,
    group_name: str = DEFAULT_JTS_GROUP,
    parent_group: str | None = None,
    prefer_smooth: bool = True,
) -> JtsTrajectory:
    """Load landmark translations from a JTS HDF5 file or embedded session group."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    if parent_group:
        group = _read_hdf5_trajectory_group(path, parent_group, group_name)
    else:
        try:
            from jts import helper as hlp
        except ImportError as exc:
            raise ImportError(
                "install jaw_tracking to read JTS output: pip install -e software/jaw_tracking"
            ) from exc

        data = hlp.load_hdf5_transformations(str(path), group_name=group_name)
        if group_name not in data:
            available = ", ".join(sorted(data))
            raise ValueError(f"group {group_name!r} not in {path}; available: {available}")
        group = data[group_name]

    translations = np.asarray(group["translations"], dtype=np.float64)
    smooth_key = "translations_smooth"
    if smooth_key in group:
        smooth = np.asarray(group[smooth_key], dtype=np.float64)
    else:
        smooth = translations.copy()

    derivatives = dict(group.get("derivatives") or {})

    return JtsTrajectory(
        path=path,
        group_name=group_name,
        translations=translations,
        translations_smooth=smooth if prefer_smooth else translations,
        sample_rate=float(group.get("sample_rate", 0.0)),
        unit=str(group.get("unit", "m")),
        derivatives=derivatives,
        metadata={"raw_keys": list(group.keys())},
        parent_group=parent_group,
    )


def load_jts_trajectory_from_source(source: JtsSource, *, prefer_smooth: bool = True) -> JtsTrajectory:
    """Load a trajectory described by :class:`JtsSource`."""
    return load_jts_trajectory(
        source.path,
        group_name=source.trajectory_group,
        parent_group=source.parent_group,
        prefer_smooth=prefer_smooth,
    )


def _read_hdf5_trajectory_group(
    path: Path,
    parent_group: str,
    group_name: str,
) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        if parent_group not in f:
            raise ValueError(f"parent group {parent_group!r} not in {path}")
        root = f[parent_group]
        if group_name not in root:
            available = ", ".join(sorted(root.keys()))
            raise ValueError(
                f"group {group_name!r} not under /{parent_group} in {path}; available: {available}"
            )
        grp = root[group_name]
        if "translations" not in grp:
            raise ValueError(f"group /{parent_group}/{group_name} has no translations dataset")

        data: dict[str, Any] = {
            "translations": np.asarray(grp["translations"][:], dtype=np.float64),
            "sample_rate": float(grp.attrs.get("sample_rate", 0.0)),
            "unit": str(grp.attrs.get("unit", "m")),
        }
        if "translations_smooth" in grp:
            data["translations_smooth"] = np.asarray(grp["translations_smooth"][:], dtype=np.float64)

        derivatives: dict[str, np.ndarray] = {}
        if "derivatives" in grp:
            dgrp = grp["derivatives"]
            for key in dgrp.keys():
                derivatives[str(key)] = np.asarray(dgrp[key][:], dtype=np.float64)
        data["derivatives"] = derivatives
        return data
