# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Resolve tag-frame orientation ambiguity in ``mounts_T_body_tag`` from data.

A printed square tag has eight indistinguishable mountings: four rotations about
its own normal, each with the normal pointing either way. CAD gives the seat
pose but not which of those eight the sticker ended up in, and OpenCV's tag
frame (x right, y up, z out of the marker, corner 0 at ``(-s/2, +s/2)``) is not
the convention most exporters use. A wrong choice leaves translations and tag
size correct while the Kabsch residual jumps to roughly the tag edge length.

Given triangulated corners of a body, :func:`fit_body_mount_corrections`
searches those eight candidates per tag and reports the combination that
minimises the residual, plus ready-to-paste YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from optical.markers.config import MarkerConfig
from optical.tracking.pose import tag_object_points
from optical.tracking.triangulation import TriangulatedPoint, kabsch


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _candidates() -> List[Tuple[str, np.ndarray]]:
    """The eight symmetries of a square tag, as rotations of the tag frame."""
    out: List[Tuple[str, np.ndarray]] = []
    for flip, flip_name in ((False, ""), (True, "+flip")):
        base = _rot_x(180.0) if flip else np.eye(3)
        for k in range(4):
            deg = 90 * k
            name = f"rotz{deg}{flip_name}" if (k or flip) else "as-configured"
            out.append((name, base @ _rot_z(deg)))
    return out


MOUNT_CANDIDATES: List[Tuple[str, np.ndarray]] = _candidates()


@dataclass
class MountFitResult:
    """Best per-tag orientation correction for one rigid body."""

    body: str
    tag_ids: List[int]
    corrections: Dict[int, str]
    corrected_T_body_tag: Dict[int, np.ndarray]
    rms_mm: float
    baseline_rms_mm: float
    n_points: int

    @property
    def changed(self) -> bool:
        return any(name != "as-configured" for name in self.corrections.values())

    def summary(self) -> str:
        picks = ", ".join(f"{tid}:{self.corrections[tid]}" for tid in self.tag_ids)
        return (
            f"{self.body} mount fit: rms {self.baseline_rms_mm:.2f}mm -> "
            f"{self.rms_mm:.2f}mm  [{picks}]"
        )

    def yaml_snippet(self) -> str:
        """``mounts_T_body_tag`` entries for the corrected tags."""
        lines = ["  mounts_T_body_tag:"]
        for tid in self.tag_ids:
            T = self.corrected_T_body_tag[tid]
            lines.append(f'    "{tid}":')
            lines.append("      [")
            for r in range(4):
                row = ", ".join(f"{v:.7g}" for v in T[r])
                comma = "," if r < 3 else ""
                lines.append(f"        [{row}]{comma}")
            lines.append("      ]")
        return "\n".join(lines)


def _matched_points(
    triangulated: Sequence[TriangulatedPoint],
    *,
    tag_ids: Sequence[int],
    marker_config: MarkerConfig,
    corrections: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    obj = tag_object_points(marker_config.tag_size_mm)
    tmap = {(p.marker_id, p.corner_index): p for p in triangulated}
    model: List[np.ndarray] = []
    observed: List[np.ndarray] = []
    for tid in tag_ids:
        T = marker_config.mounts_T_body_tag.get(tid)
        if T is None:
            continue
        T_fixed = T.copy()
        T_fixed[:3, :3] = T[:3, :3] @ corrections[tid]
        for idx, pt_tag in enumerate(obj):
            pt = tmap.get((tid, idx))
            if pt is None:
                continue
            model.append((T_fixed @ np.append(pt_tag, 1.0))[:3])
            observed.append(pt.position_mm)
    if not model:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.stack(model), np.stack(observed)


def fit_body_mount_corrections(
    triangulated: Sequence[TriangulatedPoint],
    *,
    body: str,
    marker_config: MarkerConfig,
) -> Optional[MountFitResult]:
    """Search tag-frame symmetries for the lowest Kabsch residual.

    Returns ``None`` when fewer than two tags of *body* were triangulated, since
    a single tag fits any orientation exactly and carries no information.
    """
    lo, hi = marker_config.id_ranges[body]
    seen = {
        p.marker_id
        for p in triangulated
        if lo <= p.marker_id <= hi and p.marker_id in marker_config.mounts_T_body_tag
    }
    tag_ids = sorted(seen)
    if len(tag_ids) < 2:
        return None

    identity = {tid: np.eye(3) for tid in tag_ids}
    model, observed = _matched_points(
        triangulated, tag_ids=tag_ids, marker_config=marker_config, corrections=identity
    )
    if model.shape[0] < 3:
        return None
    _, baseline_rms = kabsch(model, observed)
    n_points = int(model.shape[0])  # corrections rotate tags, never change matching

    best: Optional[Tuple[float, Dict[int, str], Dict[int, np.ndarray]]] = None
    for combo in product(MOUNT_CANDIDATES, repeat=len(tag_ids)):
        corrections = {tid: combo[i][1] for i, tid in enumerate(tag_ids)}
        names = {tid: combo[i][0] for i, tid in enumerate(tag_ids)}
        model, observed = _matched_points(
            triangulated, tag_ids=tag_ids, marker_config=marker_config, corrections=corrections
        )
        if model.shape[0] < 3:
            continue
        _, rms = kabsch(model, observed)
        if best is None or rms < best[0]:
            best = (rms, names, corrections)

    if best is None:
        return None

    rms, names, corrections = best
    corrected = {}
    for tid in tag_ids:
        T = marker_config.mounts_T_body_tag[tid].copy()
        T[:3, :3] = T[:3, :3] @ corrections[tid]
        corrected[tid] = T
    return MountFitResult(
        body=body,
        tag_ids=tag_ids,
        corrections=names,
        corrected_T_body_tag=corrected,
        rms_mm=float(rms),
        baseline_rms_mm=float(baseline_rms),
        n_points=n_points,
    )


def measure_body_geometry(
    triangulated: Sequence[TriangulatedPoint],
    *,
    body: str,
    marker_config: MarkerConfig,
) -> str:
    """Model-free readout: triangulated tag edge lengths and tag separations.

    Compares against what the config expects, which separates a tag-size or
    world-scale error (numbers disagree) from an orientation error (they agree).
    """
    lo, hi = marker_config.id_ranges[body]
    tmap = {(p.marker_id, p.corner_index): p for p in triangulated}
    tags = sorted({p.marker_id for p in triangulated if lo <= p.marker_id <= hi})

    parts: List[str] = []
    centres: Dict[int, np.ndarray] = {}
    for tid in tags:
        pts = [tmap[(tid, i)].position_mm for i in range(4) if (tid, i) in tmap]
        if len(pts) < 4:
            parts.append(f"tag{tid}: {len(pts)}/4 corners")
            continue
        c = np.stack(pts)
        edges = [float(np.linalg.norm(c[i] - c[(i + 1) % 4])) for i in range(4)]
        centres[tid] = c.mean(axis=0)
        parts.append(f"tag{tid} edge {np.mean(edges):.1f}mm (exp {marker_config.tag_size_mm:.1f})")

    for a, b in ((a, b) for i, a in enumerate(tags) for b in tags[i + 1:]):
        if a in centres and b in centres:
            got = float(np.linalg.norm(centres[a] - centres[b]))
            Ta = marker_config.mounts_T_body_tag.get(a)
            Tb = marker_config.mounts_T_body_tag.get(b)
            if Ta is not None and Tb is not None:
                exp = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
                parts.append(f"{a}-{b} sep {got:.1f}mm (exp {exp:.1f})")
            else:
                parts.append(f"{a}-{b} sep {got:.1f}mm")

    return "  ".join(parts) if parts else "no triangulated corners"
