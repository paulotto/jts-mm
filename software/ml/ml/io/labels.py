# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Label sidecar read/write (``<session_stem>_labels.h5``).

``/labels/{t_master_us, optical_frame, y, bout_id}`` on the optical frame grid.
``y`` is uint8: 0..6 are the classes, **255 = unlabelled** (ignored downstream,
never REST). Schema 1.0 files (written before 2026-10-15) had no unlabelled
value, so every unpainted frame was REST there; they are still readable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ml.config.classes import (CLASS_COUNT, CLASS_NAMES, IGNORE, LABEL_SCHEMA_VERSION,
                               READABLE_LABEL_SCHEMAS, UNLABELLED)


@dataclass
class LabelSidecar:
    t_master_us: np.ndarray      # (F,) int64
    optical_frame: np.ndarray    # (F,) int64
    y: np.ndarray                # (F,) uint8, 255 = unlabelled
    bout_id: np.ndarray          # (F,) int32, contiguous same-value runs from 1
    attrs: dict[str, Any]

    def window_labels(self) -> np.ndarray:
        """Labels as used by the window pipeline: int64 with -1 = ignore."""
        y = self.y.astype(np.int64)
        return np.where(y == UNLABELLED, IGNORE, y)


def default_label_path(session_path: str | Path) -> Path:
    session_path = Path(session_path)
    return session_path.with_name(f"{session_path.stem}_labels.h5")


def write_labels(path: str | Path, *, t_master_us: np.ndarray, optical_frame: np.ndarray,
                 y: np.ndarray, bout_id: np.ndarray | None = None, session_id: str = "",
                 subject_id: str = "", jts_output: str | Path = "", jts_config: str | Path = "",
                 jts_group: str = "", labeler: str = "manual", source: str = "manual",
                 trajectory_source: str = "jts", extra_attrs: dict[str, Any] | None = None
                 ) -> Path:
    """Write a label sidecar; ``y`` may contain 255 (unlabelled)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t_master_us = np.asarray(t_master_us, dtype=np.int64)
    optical_frame = np.asarray(optical_frame, dtype=np.int64)
    y = np.asarray(y, dtype=np.uint8)
    bout_id = compute_bout_ids(y) if bout_id is None else np.asarray(bout_id, dtype=np.int32)
    _validate(t_master_us, optical_frame, y, bout_id)
    counts = {CLASS_NAMES[i]: int(np.sum(y == i)) for i in range(CLASS_COUNT)}
    with h5py.File(path, "w") as f:
        g = f.create_group("labels")
        g.create_dataset("t_master_us", data=t_master_us.astype(np.uint64))
        g.create_dataset("optical_frame", data=optical_frame)
        g.create_dataset("y", data=y)
        g.create_dataset("bout_id", data=bout_id)
        attrs = {
            "class_names": json.dumps(CLASS_NAMES), "unlabelled_value": UNLABELLED,
            "source": source, "trajectory_source": trajectory_source,
            "jts_output": str(jts_output), "jts_config": str(jts_config), "jts_group": jts_group,
            "session_id": session_id, "subject_id": subject_id, "labeler": labeler,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "schema_version": LABEL_SCHEMA_VERSION,
            "n_labeled": int(np.sum(y != UNLABELLED)),
            "n_unlabelled": int(np.sum(y == UNLABELLED)),
            "n_rest": int(np.sum(y == 0)), "class_counts": json.dumps(counts),
        }
        attrs.update({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                      for k, v in (extra_attrs or {}).items()})
        for k, v in attrs.items():
            g.attrs[k] = v
    return path


def read_labels(path: str | Path) -> LabelSidecar:
    """Load and validate a sidecar. Bout ids are recomputed whenever any is negative
    (older writers gave every REST sample the id -1, so REST collapsed into one group
    that collided across pooled sessions)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if "labels" not in f:
            raise ValueError(f"missing /labels group: {path}")
        g = f["labels"]
        t = np.asarray(g["t_master_us"][()], dtype=np.int64)
        frame = np.asarray(g["optical_frame"][()], dtype=np.int64)
        y = np.asarray(g["y"][()], dtype=np.uint8)
        bout = np.asarray(g["bout_id"][()], dtype=np.int32) if "bout_id" in g else None
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}
    for key in ("class_names", "class_counts"):
        if isinstance(attrs.get(key), str):
            attrs[key] = json.loads(attrs[key])
    schema = str(attrs.get("schema_version", ""))
    if schema and schema not in READABLE_LABEL_SCHEMAS:
        raise ValueError(f"unsupported label schema {schema!r} in {path}")
    if bout is None or bout.size != y.size or np.any(bout < 0):
        bout = compute_bout_ids(y)
    _validate(t, frame, y, bout)
    return LabelSidecar(t, frame, y, bout, attrs)


def _validate(t: np.ndarray, frame: np.ndarray, y: np.ndarray, bout: np.ndarray) -> None:
    n = len(t)
    if not (len(frame) == len(y) == len(bout) == n):
        raise ValueError("label arrays must have equal length")
    if n > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("t_master_us must be strictly increasing")
    bad = (y >= CLASS_COUNT) & (y != UNLABELLED)
    if np.any(bad):
        raise ValueError(f"label values must be 0..{CLASS_COUNT - 1} or {UNLABELLED} (unlabelled)")


def compute_bout_ids(y: np.ndarray) -> np.ndarray:
    """One id per contiguous run of the same value (REST and unlabelled included), from 1."""
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(0, dtype=np.int32)
    return np.cumsum(np.concatenate(([True], y[1:] != y[:-1]))).astype(np.int32)
