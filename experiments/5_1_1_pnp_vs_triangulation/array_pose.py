# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Shared pose collection and ladder-slot parsing for 5_1_1_pnp_vs_triangulation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

AXES = ("x", "y", "z")
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_CALIB_PREFIXES = ("mand_point", "max_point", "tmj_")
_STAMP_RE = re.compile(r"_(\d{8}T\d{6}Z)$")


def is_calibration_section(name: str) -> bool:
    """Landmark-digitizing phases inserted ahead of an experiment-plan section."""
    key = (name or "").strip().lower()
    return key.startswith(_CALIB_PREFIXES)


def load_experiment_sections(leaf: Path) -> Dict[str, tuple[int, int]]:
    """Frame intervals from the leaf's JTS config (``sub_experiments``).

    Intervals are inclusive ``[start, end]`` and index the stored frames.
    Calibration phases are kept; callers drop them when they want the
    measurement section only.
    """
    configs = sorted(Path(leaf).glob("*_jts_config.json"))
    if not configs:
        return {}
    cfg = json.loads(configs[-1].read_text(encoding="utf-8"))
    experiment = (cfg.get("analysis") or {}).get("experiment") or {}
    subs = experiment.get("sub_experiments") or {}
    out: Dict[str, tuple[int, int]] = {}
    if not isinstance(subs, dict):
        return out
    for name, interval in subs.items():
        if (
            isinstance(interval, (list, tuple))
            and len(interval) == 2
        ):
            out[str(name)] = (int(interval[0]), int(interval[1]))
    return out


def slot_aliases(slot: dict) -> set[str]:
    """Names that refer to one hand-measured ladder slot."""
    section = str(slot["section"]).strip().lower()
    mm = float(slot["mm"])
    aliases = {section, f"{section}mm"}
    if abs(mm - round(mm)) < 1e-6:
        n = str(int(round(mm)))
        aliases.update({n, f"d{n}", f"{n}mm", f"d{n}mm", f"slot_{n}", f"slot{n}"})
    text = f"{mm:g}"
    aliases.update({text, f"d{text}", f"{text}mm", f"d{text}mm"})
    return aliases


def match_slot(token: str, slots: Sequence[dict]) -> dict | None:
    """Return the ladder slot named by ``token`` (section name or a distance)."""
    key = (token or "").strip().lower().replace("-", "_")
    if not key:
        return None
    for slot in slots:
        if key in slot_aliases(slot):
            return slot
    return None


def orientation_and_token(leaf: Path, orientations: Iterable[str]) -> tuple[str | None, str]:
    """Split ``…_{orientation}_{token}_{stamp}`` out of a recording leaf name.

    ``token`` is empty when the condition is only the orientation
    (``…_lateral_{stamp}``). The distance then comes from the experiment-plan
    section inside the leaf.
    """
    name = leaf.name
    stamp = _STAMP_RE.search(name)
    body = name[: stamp.start()] if stamp else name
    for orient in sorted({str(o) for o in orientations}, key=len, reverse=True):
        marker = f"_{orient}_"
        idx = body.find(marker)
        if idx >= 0:
            return orient, body[idx + len(marker) :]
        if body.endswith(f"_{orient}") or body == orient:
            return orient, ""
    return None, ""


def parse_ladder_leaf(
    leaf: Path,
    orientations: Iterable[str],
    ranges: Iterable[str],
) -> tuple[str | None, str | None, str]:
    """Return ``(range, orientation, slot_token)`` from a recording leaf name.

    The condition is ``{near|mid|far}_{lateral|frontal}_{section}``, in either
    order of range and orientation. ``slot_token`` is whatever follows both
    (``d40`` in ``…_near_lateral_d40_{stamp}``), or empty when the slot is
    only named by the experiment-plan section.
    """
    name = leaf.name
    stamp = _STAMP_RE.search(name)
    body = name[: stamp.start()] if stamp else name
    parts = body.split("_")
    range_set = {str(r) for r in ranges}
    orient_set = {str(o) for o in orientations}
    range_i: int | None = None
    orient_i: int | None = None
    for i, part in enumerate(parts):
        if part in range_set:
            range_i = i
        if part in orient_set:
            orient_i = i
    found_range = None if range_i is None else parts[range_i]
    found_orient = None if orient_i is None else parts[orient_i]
    if range_i is None or orient_i is None:
        return found_range, found_orient, ""
    token = "_".join(parts[max(range_i, orient_i) + 1 :])
    return found_range, found_orient, token


def section_interval_for_slot(
    sections: Dict[str, tuple[int, int]],
    slot: dict,
    slots: Sequence[dict],
) -> tuple[str | None, tuple[int, int] | None]:
    """Pick the measurement section for ``slot``.

    Prefers a section whose name matches the slot. A take with exactly one
    non-calibration section uses that section, unless its name is a different
    ladder slot.
    """
    measurement = {
        name: interval
        for name, interval in sections.items()
        if not is_calibration_section(name)
    }
    aliases = slot_aliases(slot)
    for name, interval in measurement.items():
        if name.strip().lower() in aliases:
            return name, interval
    if len(measurement) == 1:
        name, interval = next(iter(measurement.items()))
        other = match_slot(name, slots)
        if other is None or other["section"] == slot["section"]:
            return name, interval
    return None, None


def axis_sigma_mm(ts: np.ndarray) -> dict:
    """Per-axis σ of translation about its temporal mean (mm)."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[0] < 2:
        return {axis: float("nan") for axis in AXES}
    std = np.std(ts, axis=0)
    return {axis: float(std[i]) for axis, i in AXIS_INDEX.items()}


def collect_body_poses(
    leaf: Path,
    *,
    tracker,
    tcfg,
    bodies: List[str],
    pnp_camera_id: str,
    frame_interval: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> tuple[int, Dict[str, Dict[str, np.ndarray]]]:
    """PnP (one camera) and triangulation poses for ``bodies``.

    ``frame_interval`` is an inclusive stored-frame range (one experiment-plan
    section). ``max_frames`` caps how many frames inside that range are used.
    """
    from optical.tracking.pose_estimation import estimate_world_body_poses_pnp
    from optical.tracking.triangulation import estimate_world_body_poses_triangulation

    from optical.evaluation.recordings import iter_session_frames

    series: Dict[str, Dict[str, List]] = {
        body: {"pnp_t": [], "pnp_R": [], "tri_t": [], "tri_R": []} for body in bodies
    }
    n_frames = 0
    for frame_idx, frames in iter_session_frames(leaf):
        if frame_interval is not None:
            lo, hi = frame_interval
            if int(frame_idx) < lo:
                continue
            if int(frame_idx) > hi:
                break
        n_frames += 1
        detections = tracker.detect_all(frames)
        pnp_detections = (
            {pnp_camera_id: detections[pnp_camera_id]}
            if pnp_camera_id in detections
            else {}
        )
        pnp = estimate_world_body_poses_pnp(
            pnp_detections,
            intrinsics=tracker.intrinsics,
            extrinsics=tracker.extrinsics,
            marker_config=tracker.marker_config,
            use_reprojection_weights=tcfg.use_reprojection_weights,
            max_reprojection_rms_px=tcfg.pnp_max_reprojection_rms_px,
            bodies=bodies,
        )
        tri, _ = estimate_world_body_poses_triangulation(
            detections,
            intrinsics=tracker.intrinsics,
            extrinsics=tracker.extrinsics,
            marker_config=tracker.marker_config,
            min_views=tcfg.triangulation_min_views,
            max_residual_mm=tcfg.triangulation_max_residual_mm,
            bodies=bodies,
        )
        for body in bodies:
            if pnp.get(body) is not None and pnp[body].valid:
                T = pnp[body].T_world_body
                series[body]["pnp_t"].append(T[:3, 3].copy())
                series[body]["pnp_R"].append(T[:3, :3].copy())
            if tri.get(body) is not None and tri[body].valid:
                T = tri[body].T_world_body
                series[body]["tri_t"].append(T[:3, 3].copy())
                series[body]["tri_R"].append(T[:3, :3].copy())
        if max_frames is not None and n_frames >= max_frames:
            break

    packed: Dict[str, Dict[str, np.ndarray]] = {}
    for body in bodies:
        packed[body] = {
            "pnp_t": _as_translations(series[body]["pnp_t"]),
            "pnp_R": _as_rotations(series[body]["pnp_R"]),
            "tri_t": _as_translations(series[body]["tri_t"]),
            "tri_R": _as_rotations(series[body]["tri_R"]),
        }
    return n_frames, packed


def _as_translations(samples: List[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def _as_rotations(samples: List[np.ndarray]) -> np.ndarray:
    if not samples:
        return np.zeros((0, 3, 3), dtype=np.float64)
    return np.asarray(samples, dtype=np.float64)


def warn_missing_body(tracker, body: str) -> None:
    """Print a hint when ``body`` is not a rigid body in the recording config."""
    ranges = getattr(tracker.marker_config, "id_ranges", {}) or {}
    known = sorted(str(k) for k in ranges)
    if body not in ranges:
        print(
            f"Warning: rigid body {body!r} is not in this recording's marker "
            f"id_ranges ({', '.join(known) or 'none'}). "
            f"Define the two-marker array in optical.yaml — not the CRA (HP)."
        )
