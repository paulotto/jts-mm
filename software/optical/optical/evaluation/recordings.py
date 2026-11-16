# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Bench recordings of the optical studies (§5.1.x): leaves, conditions, frames.

A recording leaf is one ``optical track --persist-frames`` run, a folder
``{name}_{condition}_{UTC stamp}`` holding ``meta.json`` and ``frames/``
(``optical.paths.experiment_recording_dir``). Replay a leaf with the tracker
from :func:`optical.tracking.tracker.load_tracker_for_recording`, which picks
up the leaf's own calibration/config snapshot.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

_STAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)$")


def iter_recording_leaves(recordings_dir: Path | str) -> list[Path]:
    """Direct children of ``recordings_dir`` that contain ``meta.json``, sorted by name."""
    root = Path(recordings_dir).expanduser()
    if not root.is_dir():
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "meta.json").is_file()]


def parse_condition_from_leaf(leaf: Path | str, *, known: Sequence[str] | None = None) -> str:
    """Condition label of a ``{name}_{condition}_{stamp}`` leaf.

    With ``known`` (e.g. ``["static_near", "near"]``) the longest known
    condition ending the prefix wins, so labels with underscores work; without
    it the last ``_`` token before the stamp is the condition.
    """
    name = leaf.name if isinstance(leaf, Path) else str(leaf)
    m = _STAMP_RE.search(name)
    if m:
        prefix = name[: m.start()]
        if known:
            for cond in sorted({str(c) for c in known}, key=len, reverse=True):
                if prefix.endswith(f"_{cond}") or prefix == cond:
                    return cond
        return prefix.split("_")[-1]
    return name


def latest_leaf_per_condition(leaves: Sequence[Path], conditions: Sequence[str]
                              ) -> dict[str, Path]:
    """Newest leaf (by name, i.e. UTC stamp) for each of ``conditions``."""
    best: dict[str, Path] = {}
    for leaf in leaves:
        cond = parse_condition_from_leaf(leaf, known=conditions)
        if cond in conditions and (cond not in best or leaf.name > best[cond].name):
            best[cond] = leaf
    return best


def iter_session_frames(frames_dir: Path | str, max_frames: Optional[int] = None
                        ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    """``(frame_idx, {cam_id: image})`` of a leaf, stopping after ``max_frames``."""
    from optical.tracking.frame_store import iter_frames

    for n, item in enumerate(iter_frames(Path(frames_dir)), start=1):
        yield item
        if max_frames is not None and n >= max_frames:
            break


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: Path | str, payload: Any) -> Path:
    """Indented strict JSON (NaN -> null, numpy -> Python); creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


__all__ = [
    "iter_recording_leaves", "iter_session_frames", "latest_leaf_per_condition",
    "parse_condition_from_leaf", "write_json",
]
