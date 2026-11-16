# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Experiment phase slicing for sequential labeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml.config.classes import UNLABELLED
from ml.io.session import PhaseInterval
from ml.labeling.trajectory import LabelingGrid


@dataclass(frozen=True)
class PhaseView:
    """One experiment phase mapped onto the full labeling grid."""

    index: int
    total: int
    phase: PhaseInterval
    global_indices: np.ndarray  # indices into the full session grid
    t_s: np.ndarray  # seconds, zero-based within the phase
    t_s_abs: np.ndarray  # seconds from session start (for status messages)


def list_experiment_phases(phases: list[PhaseInterval]) -> list[PhaseInterval]:
    """Return experiment phases in recording order."""
    exp = [p for p in phases if p.kind == "experiment"]
    return sorted(exp, key=lambda p: (p.t_start_us, p.t_end_us, p.name))


def build_phase_views(grid: LabelingGrid) -> list[PhaseView]:
    """Build phase views; falls back to one view spanning the full grid."""
    phases = list_experiment_phases(grid.phases)
    if not phases:
        phases = [
            PhaseInterval(
                name="full_session",
                kind="experiment",
                t_start_us=int(grid.t_master_us[0]),
                t_end_us=int(grid.t_master_us[-1]),
            )
        ]

    views: list[PhaseView] = []
    for idx, phase in enumerate(phases):
        global_indices = _indices_for_phase(grid, phase)
        if global_indices.size == 0:
            continue
        t_abs = grid.t_s[global_indices]
        views.append(
            PhaseView(
                index=len(views),
                total=0,  # filled below
                phase=phase,
                global_indices=global_indices,
                t_s=t_abs - float(t_abs[0]),
                t_s_abs=t_abs,
            )
        )

    if not views:
        n = len(grid.t_s)
        indices = np.arange(n, dtype=np.int64)
        views = [
            PhaseView(
                index=0,
                total=1,
                phase=PhaseInterval("full_session", "experiment", int(grid.t_master_us[0]), int(grid.t_master_us[-1])),
                global_indices=indices,
                t_s=grid.t_s - float(grid.t_s[0]),
                t_s_abs=grid.t_s.copy(),
            )
        ]

    total = len(views)
    return [
        PhaseView(
            index=v.index,
            total=total,
            phase=v.phase,
            global_indices=v.global_indices,
            t_s=v.t_s,
            t_s_abs=v.t_s_abs,
        )
        for v in views
    ]


def _indices_for_phase(grid: LabelingGrid, phase: PhaseInterval) -> np.ndarray:
    t = grid.t_master_us.astype(np.int64)
    mask = (t >= phase.t_start_us) & (t <= phase.t_end_us)
    return np.where(mask)[0]


def apply_label_to_phase(
    labels: list[int],
    view: PhaseView,
    *,
    t_start: float,
    t_end: float,
    class_id: int,
) -> tuple[list[int], int]:
    """Paint ``class_id`` onto global labels for the selected span within ``view``."""
    lo, hi = sorted([t_start, t_end])
    local_mask = (view.t_s >= lo) & (view.t_s <= hi)
    global_idx = view.global_indices[local_mask]
    updated = list(labels)
    arr = np.asarray(updated, dtype=int)
    arr[global_idx] = class_id
    return arr.tolist(), int(global_idx.size)


def reset_phase_labels(labels: list[int], view: PhaseView) -> list[int]:
    """Clear a phase back to unlabelled (255), not to REST."""
    updated = list(labels)
    arr = np.asarray(updated, dtype=int)
    arr[view.global_indices] = UNLABELLED
    return arr.tolist()
