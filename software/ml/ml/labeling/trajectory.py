# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Load trajectories for the labeling UI (JTS output or optical fallback)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ml.io.jts_output import (
    DEFAULT_JTS_GROUP,
    JtsSource,
    find_embedded_jts_group,
    load_jts_trajectory_from_source,
)
from ml.io.session import PhaseInterval, SessionData, load_session


@dataclass
class LabelingGrid:
    """Trajectory samples aligned to the optical / label timebase."""

    session: SessionData
    t_master_us: np.ndarray  # (N,) uint64
    optical_frame: np.ndarray  # (N,) int64 absolute optical frame index
    t_s: np.ndarray  # (N,) seconds from first sample
    translations: np.ndarray  # (N, 3) raw for display
    translations_smooth: np.ndarray  # (N, 3) primary trace
    velocity: np.ndarray | None  # (N, 3) optional mm/s
    phases: list[PhaseInterval]
    trajectory_source: str  # "jts" | "optical_fallback"
    jts_output: Path | None = None
    jts_config: Path | None = None
    jts_group: str = DEFAULT_JTS_GROUP
    unit: str = "mm"
    meta: dict[str, Any] = field(default_factory=dict)


def discover_jts_config(session_path: Path) -> Path | None:
    """Find ``*_jts_config.json`` next to the session."""
    session_path = Path(session_path)
    candidates = [session_path.with_name(f"{session_path.stem}_jts_config.json")]
    for path in sorted(session_path.parent.glob("*_jts_config.json")):
        candidates.append(path)
    seen: set[Path] = set()
    for cand in candidates:
        cand = cand.resolve()
        if cand in seen or not cand.is_file():
            continue
        seen.add(cand)
        return cand
    return None


def resolve_jts_source(
    session_path: Path,
    jts_config: Path | None = None,
    jts_output: Path | None = None,
) -> JtsSource | None:
    """Locate JTS trajectories in a sidecar file or embedded under ``/jts/`` in the session."""
    session_path = Path(session_path)
    explicit = Path(jts_output) if jts_output else None

    sidecar: JtsSource | None = None
    if jts_config and jts_config.is_file():
        cfg = json.loads(jts_config.read_text(encoding="utf-8"))
        out_dir = Path(cfg.get("output", {}).get("directory", "./analysis_results"))
        if not out_dir.is_absolute():
            out_dir = (jts_config.parent / out_dir).resolve()
        h5_name = cfg.get("output", {}).get("hdf5_filename", "jaw_motion.h5")
        candidate = out_dir / h5_name
        if candidate.is_file():
            sidecar = JtsSource(path=candidate.resolve())

    if sidecar is None:
        for candidate in (
            session_path.parent / "analysis_results" / "jaw_motion.h5",
            session_path.with_name("jaw_motion.h5"),
            Path("analysis_results/jaw_motion.h5"),
        ):
            if candidate.is_file():
                sidecar = JtsSource(path=candidate.resolve())
                break

    embedded = _discover_embedded_jts_source(session_path, jts_config)

    if explicit is not None:
        explicit = explicit.resolve()
        if embedded is not None and explicit == session_path.resolve():
            return embedded
        return JtsSource(path=explicit)

    return sidecar or embedded


def discover_jts_output(session_path: Path, jts_config: Path | None = None) -> Path | None:
    """Guess JTS ``jaw_motion.h5`` from config output directory or common locations."""
    source = resolve_jts_source(session_path, jts_config)
    return source.path if source else None


def _discover_embedded_jts_source(
    session_path: Path,
    jts_config: Path | None,
) -> JtsSource | None:
    """Find JTS trajectories appended into the session HDF5 (``output.hdf5_append_group``)."""
    session_path = Path(session_path)
    if not session_path.is_file():
        return None

    append_group = "jts"
    if jts_config and jts_config.is_file():
        cfg = json.loads(jts_config.read_text(encoding="utf-8"))
        configured = cfg.get("output", {}).get("hdf5_append_group")
        if configured:
            append_group = str(configured).strip("/")

    trajectory_group = find_embedded_jts_group(session_path, parent_group=append_group)
    if trajectory_group is None:
        return None

    return JtsSource(
        path=session_path.resolve(),
        parent_group=append_group,
        trajectory_group=trajectory_group,
    )


def load_labeling_grid(
    session_path: str | Path,
    *,
    jts_output: str | Path | None = None,
    jts_config: str | Path | None = None,
    invert_axis: tuple[bool, bool, bool] = (False, False, False),
) -> LabelingGrid:
    """Build the labeling timebase and trajectory arrays."""
    session = load_session(session_path)
    if session.optical_t_us is None:
        raise ValueError(f"session has no /optical/t_master_us: {session.path}")

    cfg_path = Path(jts_config) if jts_config else discover_jts_config(session.path)
    jts_source = resolve_jts_source(
        session.path,
        cfg_path,
        Path(jts_output) if jts_output else None,
    )

    optical_t = np.asarray(session.optical_t_us, dtype=np.uint64)
    optical_frame = np.arange(len(optical_t), dtype=np.int64)

    if jts_source is not None and jts_source.path.is_file():
        try:
            return _grid_from_jts(
                session,
                optical_t,
                optical_frame,
                jts_source,
                cfg_path,
                invert_axis=invert_axis,
            )
        except ImportError:
            import warnings

            warnings.warn(
                f"JTS data found at {jts_source.display_path} but jaw_tracking is not installed; "
                "using /optical CRA-relative fallback. "
                "Install with: pip install -e software/jaw_tracking",
                stacklevel=2,
            )

    return _grid_from_optical(session, optical_t, optical_frame, invert_axis=invert_axis)


def _grid_from_jts(
    session: SessionData,
    optical_t: np.ndarray,
    optical_frame: np.ndarray,
    jts_source: JtsSource,
    jts_config: Path | None,
    *,
    invert_axis: tuple[bool, bool, bool],
) -> LabelingGrid:
    traj = load_jts_trajectory_from_source(jts_source)
    cfg = _load_jts_config(jts_config) if jts_config else {}

    raw = _to_mm(traj.translations, traj.unit)
    smooth = _to_mm(traj.translations_smooth, traj.unit)
    raw, smooth, t_us, frames = _align_trajectory_to_optical(
        raw,
        smooth,
        optical_t,
        optical_frame,
        cfg,
        sample_rate=traj.sample_rate,
    )

    raw, smooth = _apply_inversions(raw, smooth, invert_axis)
    t_s = (t_us.astype(np.float64) - float(t_us[0])) * 1e-6
    velocity = _velocity_mm_s(smooth, t_s, traj.derivatives, traj.unit)

    return LabelingGrid(
        session=session,
        t_master_us=t_us,
        optical_frame=frames,
        t_s=t_s,
        translations=raw,
        translations_smooth=smooth,
        velocity=velocity,
        phases=session.phases,
        trajectory_source="jts",
        jts_output=jts_source.path.resolve(),
        jts_config=jts_config.resolve() if jts_config else None,
        jts_group=(
            f"{jts_source.parent_group}/{traj.group_name}"
            if jts_source.parent_group
            else traj.group_name
        ),
        unit="mm",
        meta={"jts_sample_rate_hz": traj.sample_rate},
    )


def _grid_from_optical(
    session: SessionData,
    optical_t: np.ndarray,
    optical_frame: np.ndarray,
    *,
    invert_axis: tuple[bool, bool, bool],
) -> LabelingGrid:
    with h5py.File(session.path, "r") as f:
        if "optical" not in f:
            raise ValueError("session missing /optical for fallback trajectory")
        pose_mta = np.asarray(f["optical"]["pose_mta"][()], dtype=np.float64)
        pose_cra = np.asarray(f["optical"]["pose_cra"][()], dtype=np.float64)

    if len(pose_mta) != len(optical_t):
        raise ValueError(
            f"optical pose count ({len(pose_mta)}) != t_master_us count ({len(optical_t)})"
        )

    xyz = _relative_translation_mm(pose_mta, pose_cra)
    raw = xyz.copy()
    smooth = _light_smooth(xyz)
    raw, smooth = _apply_inversions(raw, smooth, invert_axis)
    t_s = (optical_t.astype(np.float64) - float(optical_t[0])) * 1e-6
    velocity = _velocity_mm_s(smooth, t_s, {}, "mm")

    return LabelingGrid(
        session=session,
        t_master_us=optical_t,
        optical_frame=optical_frame,
        t_s=t_s,
        translations=raw,
        translations_smooth=smooth,
        velocity=velocity,
        phases=session.phases,
        trajectory_source="optical_fallback",
        unit="mm",
        meta={"note": "CRA-relative MP translation from /optical poses"},
    )


def _load_jts_config(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _align_trajectory_to_optical(
    raw: np.ndarray,
    smooth: np.ndarray,
    optical_t: np.ndarray,
    optical_frame: np.ndarray,
    cfg: dict[str, Any],
    *,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map JTS frames onto optical frames, or raise.

    Accepted: the trajectory covers every optical frame, or exactly the JTS config's
    ``analysis.experiment.frame_interval`` (inclusive). Anything else used to be
    truncated or anchored at frame 0, which silently shifted every label in time.
    """
    del sample_rate
    n_opt, n_jts = len(optical_t), len(smooth)
    if n_jts == n_opt:
        return raw, smooth, optical_t, optical_frame
    interval = cfg.get("analysis", {}).get("experiment", {}).get("frame_interval")
    if interval and len(interval) == 2:
        start, end = int(interval[0]), int(interval[1])
        if not (0 <= start <= end < n_opt):
            raise ValueError(f"JTS frame_interval {interval} lies outside the {n_opt} optical "
                             "frames of this session")
        if n_jts == end - start + 1:
            return raw, smooth, optical_t[start:end + 1], optical_frame[start:end + 1]
        raise ValueError(f"JTS trajectory has {n_jts} frames but frame_interval {interval} spans "
                         f"{end - start + 1}; labels would be shifted in time. Re-run JTS for this "
                         "session or pass the matching --jts-config")
    raise ValueError(f"JTS trajectory length ({n_jts}) does not match the optical grid ({n_opt}) "
                     "and the JTS config has no analysis.experiment.frame_interval to align it; "
                     "labels would be shifted in time. Pass the matching --jts-config")


def _relative_translation_mm(pose_mta: np.ndarray, pose_cra: np.ndarray) -> np.ndarray:
    n = pose_mta.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for i in range(n):
        if np.any(np.isnan(pose_mta[i])) or np.any(np.isnan(pose_cra[i])):
            continue
        t_rel = np.linalg.inv(pose_cra[i]) @ pose_mta[i]
        out[i] = t_rel[:3, 3]
    return out


def _to_mm(xyz: np.ndarray, unit: str) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if unit.lower() in {"m", "meter", "meters"}:
        return xyz * 1000.0
    return xyz


def _light_smooth(xyz: np.ndarray, window: int = 11) -> np.ndarray:
    if len(xyz) < window:
        return xyz.copy()
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(xyz, window_length=window | 1, polyorder=3, axis=0)
    except Exception:
        kernel = np.ones(window) / window
        return np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, xyz)


def _apply_inversions(
    raw: np.ndarray,
    smooth: np.ndarray,
    invert_axis: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    raw = raw.copy()
    smooth = smooth.copy()
    for axis, invert in enumerate(invert_axis):
        if invert:
            raw[:, axis] *= -1.0
            smooth[:, axis] *= -1.0
    return raw, smooth


def _velocity_mm_s(
    smooth: np.ndarray,
    t_s: np.ndarray,
    derivatives: dict[str, np.ndarray],
    unit: str,
) -> np.ndarray | None:
    if "translational_velocity" in derivatives:
        vel = np.asarray(derivatives["translational_velocity"], dtype=np.float64)
        if vel.shape[0] == smooth.shape[0]:
            if unit.lower() in {"m", "meter", "meters"}:
                vel = vel * 1000.0
            return vel
    if len(t_s) < 2:
        return None
    vel = np.gradient(smooth, t_s, axis=0)
    return vel
