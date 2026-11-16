#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
aruco_camera.py: ArUco camera tracking data source for offline jaw analysis.

This module is the JTS-side replacement for Qualisys (Strategy B in
``docs/optical/04_migration_from_qualisys.md``). It reads either:

1. Standalone optical tracking HDF5 from ``write_tracking_session``
   (``bodies/{HP,MP,CT}/`` layout — ``docs/optical/03_*.md``), or
2. Multimodal ``session.h5`` from ``jtsmm record --optical-track``
   (``/optical/{pose_cra,pose_mta,pose_ct}`` — ``docs/software/s02_data_pipeline.md``)

and exposes it through the same :class:`~jts.core.MotionCaptureData` interface
used by the Qualisys handler.

Body name mapping (multimodal → JTS):

- ``pose_cra`` → ``HP`` (headpiece / CRA, reference body)
- ``pose_mta`` → ``MP`` (mouthpiece / MTA, moving body)
- ``pose_ct``  → ``CT`` (calibration tool / digitizing pointer)

Positions are in millimeters (JTS internal unit). Untracked frames are stored
as ``NaN`` poses, matching the Qualisys convention.

The loaded bodies are represented as :class:`jts.qualisys.RigidBody` instances so
that the existing calibration / relative-motion machinery
(``QualysisData.compute_average_transformations``,
``QualysisData.compute_relative_transformations``, ...) works unchanged.
"""

__author__ = "Paul-Otto Müller"
__copyright__ = "Copyright 2026, Paul-Otto Müller"
__credits__ = ["Paul-Otto Müller"]
__license__ = "CC BY-NC-SA 4.0"
__version__ = "1.1.1"
__maintainer__ = "Paul-Otto Müller"
__status__ = "Development"
__date__ = '04.02.2026'
__url__ = "https://github.com/paulotto/jaw_tracking_system"

import json

import h5py
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import helper as hlp
from . import qualisys as qtm
from .core import FrameInterval, MotionCaptureData

logger = hlp.setup_logger(__name__)

# Bodies JTS consumes, in the order written by the optical.
DEFAULT_BODIES = ("HP", "MP", "CT")
# Standalone files write "1.0"; multimodal session.h5 writes the integer 1
# (``meta.schema_version`` of the recorder). Both are major version 1.
SUPPORTED_SCHEMA_VERSIONS = ("1.0",)
_SUPPORTED_SCHEMA_MAJOR = (1,)

# Multimodal session.h5 /optical/ datasets → JTS rigid-body names.
OPTICAL_POSE_TO_BODY = {
    "pose_cra": "HP",
    "pose_mta": "MP",
    "pose_ct": "CT",
}


class ArucoCameraData:
    """
    Loads an ArUco-camera tracking session HDF5 into Qualisys-compatible bodies.

    Supports both the standalone ``bodies/`` layout and multimodal ``/optical/``.
    Each tracked body is converted into a :class:`jts.qualisys.RigidBody`
    holding ``(N, 3)`` positions (mm) and ``(N, 3, 3)`` rotation matrices.
    """

    def __init__(self, filename: str) -> None:
        """
        Load and parse a tracking session HDF5 file.

        Args:
            filename: Path to standalone ``<experiment>.h5`` or multimodal ``session.h5``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is missing required groups or has no bodies.
        """
        if not Path(filename).exists():
            raise FileNotFoundError(f"File not found: {filename}")

        logger.info(f"Loading ArUco camera tracking session from: {filename}")

        self.filename = filename
        self.layout: str = ""  # "bodies" | "optical"
        self.rig_id: str = ""
        self.experiment_name: str = ""
        self.schema_version: str = ""
        self.frame_rate: float = 0.0
        self.frames: int = 0
        self.units_position: str = "mm"
        self.timestamps: Dict[str, Any] = {}
        self.grid_period_us: Optional[float] = None  # set when rows sit on the trigger grid
        self.quality: Dict[str, Dict[str, np.ndarray]] = {}
        self.rigid_bodies: Dict[str, qtm.RigidBody] = {}

        with h5py.File(filename, "r") as f:
            self.layout = self._detect_layout(f)
            self._parse_meta(f)
            if self.layout == "optical":
                self._parse_optical(f)
            else:
                self._parse_timestamps(f)
                self._parse_bodies(f)
                self._parse_quality(f)

        if not self.rigid_bodies:
            raise ValueError(f"No rigid bodies found in tracking session: {filename}")

        logger.info(
            f"Loaded {len(self.rigid_bodies)} bodies "
            f"({', '.join(self.rigid_bodies)}), {self.frames} frames at {self.frame_rate} Hz "
            f"[layout={self.layout}]"
        )

    @staticmethod
    def _detect_layout(f: h5py.File) -> str:
        """Return ``'optical'`` for multimodal session.h5, else ``'bodies'``."""
        if "optical" in f:
            opt = f["optical"]
            if any(name in opt for name in OPTICAL_POSE_TO_BODY):
                return "optical"
        if "bodies" in f:
            return "bodies"
        raise ValueError(
            "Tracking session has neither 'bodies/' nor '/optical/pose_*' groups"
        )

    # ----------------------------------------------------------------- parsing
    def _parse_meta(self, f: h5py.File) -> None:
        """Read the ``meta/`` group attributes (present in both layouts)."""
        if "meta" not in f:
            if self.layout == "optical":
                # Multimodal sessions always have meta, but stay lenient.
                logger.warning("Multimodal session is missing the 'meta' group")
                return
            raise ValueError("Tracking session is missing the 'meta' group")

        meta = dict(f["meta"].attrs.items())

        self.schema_version = _as_str(meta.get("schema_version", ""))
        if self.schema_version and _schema_major(self.schema_version) not in _SUPPORTED_SCHEMA_MAJOR:
            logger.warning(
                f"Tracking session schema version {self.schema_version!r} is not in "
                f"supported set {SUPPORTED_SCHEMA_VERSIONS}; attempting to load anyway."
            )

        self.rig_id = _as_str(meta.get("rig_id", ""))
        self.experiment_name = _as_str(
            meta.get("experiment_name", meta.get("session_id", ""))
        )
        self.frame_rate = float(meta.get("frame_rate_hz", 0.0) or 0.0)
        self.units_position = _as_str(meta.get("units_position", "mm")) or "mm"

        n_frames = meta.get("n_frames", None)
        self.frames = int(n_frames) if n_frames is not None else 0

        if self.units_position != "mm":
            logger.warning(
                f"Tracking session position unit is {self.units_position!r}; "
                f"JTS expects millimeters. Values are used as-is."
            )

    def _parse_timestamps(self, f: h5py.File) -> None:
        """Read the optional ``timestamps/`` group (standalone layout)."""
        if "timestamps" not in f:
            return

        ts = f["timestamps"]
        if "frame_idx" in ts:
            self.timestamps["frame_idx"] = ts["frame_idx"][()]
        if "t_master_us" in ts:
            self.timestamps["t_master_us"] = ts["t_master_us"][()]
        if "t_cam_ns" in ts:
            self.timestamps["t_cam_ns"] = {
                cam_id: ts["t_cam_ns"][cam_id][()] for cam_id in ts["t_cam_ns"].keys()
            }

    def _parse_bodies(self, f: h5py.File) -> None:
        """Convert each ``bodies/<name>/`` group into a RigidBody."""
        if "bodies" not in f:
            raise ValueError("Tracking session is missing the 'bodies' group")

        bodies_grp = f["bodies"]
        # Preserve the documented body order, then append any extras.
        ordered = [b for b in DEFAULT_BODIES if b in bodies_grp]
        ordered += [b for b in bodies_grp.keys() if b not in ordered]

        n_frames_seen = []
        for body in ordered:
            bg = bodies_grp[body]
            positions, rotations = self._extract_pose_arrays(body, bg)
            n_frames_seen.append(positions.shape[0])

            rb = qtm.RigidBody(name=body, positions=positions, rotations=rotations)
            self.rigid_bodies[body] = rb
            logger.debug(f"Loaded body '{body}' with {positions.shape[0]} frames")

        if n_frames_seen:
            self._reconcile_frame_count(max(n_frames_seen))

    def _parse_optical(self, f: h5py.File) -> None:
        """Convert multimodal ``/optical/pose_*`` datasets into RigidBodies.

        The recorder stores one row per *processed* frame: dropped camera
        frames are absent rows, not NaN rows, so row index is not time. Rows
        are therefore placed on the trigger grid (frame ``k`` = trigger at
        ``t0 + k * period``, ``t0`` = first optical row, period from
        ``/triggers``, else ``/triggers@rate_hz``, else ``meta.mcu_config_json``,
        else estimated from the rows); holes become NaN poses, exactly like a
        Qualisys frame without a lock. Frame indices in the JTS config are on
        this grid (``jtsmm.capture.phase_walker.optical_frame_intervals``).
        """
        opt = f["optical"]
        n_frames_seen = []
        raw: Dict[str, np.ndarray] = {}

        for ds_name, body in OPTICAL_POSE_TO_BODY.items():
            if ds_name not in opt:
                logger.warning(f"Multimodal /optical/ is missing '{ds_name}' ({body})")
                continue
            pose = np.asarray(opt[ds_name][()], dtype=np.float64)
            if pose.ndim != 3 or pose.shape[1:] != (4, 4):
                raise ValueError(
                    f"/optical/{ds_name} must have shape (N, 4, 4), got {pose.shape}"
                )
            raw[body] = pose

        t_us = None
        if "t_master_us" in opt:
            t_us = np.asarray(opt["t_master_us"][()], dtype=np.float64)
        n_rows = min([p.shape[0] for p in raw.values()] + ([t_us.shape[0]] if t_us is not None else []),
                     default=0)

        grid = None
        if t_us is not None and n_rows >= 2:
            grid = _trigger_grid(f, t_us[:n_rows])

        if grid is not None:
            row_frame, period_us = grid
            keep = row_frame >= 0
            n_grid = int(row_frame[keep][-1]) + 1
            t_grid = float(t_us[0]) + np.arange(n_grid, dtype=np.float64) * period_us
            # Measured stamp where a frame exists, nominal trigger time in holes.
            t_grid[row_frame[keep]] = t_us[:n_rows][keep]
            present = np.zeros(n_grid, dtype=bool)
            present[row_frame[keep]] = True
            self.grid_period_us = period_us
            self.frame_rate = 1e6 / period_us
            self.timestamps["t_master_us"] = t_grid
            self.timestamps["frame_idx"] = np.arange(n_grid, dtype=np.int64)
            self.timestamps["row_present"] = present
            self.timestamps["row_frame"] = row_frame
            if n_grid != n_rows:
                logger.info(
                    f"Placed {n_rows} optical rows on the {self.frame_rate:.4g} Hz "
                    f"trigger grid: {n_grid} frames, {n_grid - n_rows} missing (NaN)"
                )
        elif t_us is not None:
            self.timestamps["t_master_us"] = t_us
            self.timestamps["frame_idx"] = np.arange(t_us.shape[0], dtype=np.int64)
            if self.frame_rate <= 0.0:
                self.frame_rate = _frame_rate_from_t_master_us(t_us)

        for body, pose in raw.items():
            if grid is not None:
                row_frame, _ = grid
                keep = row_frame >= 0
                laid = np.full((int(row_frame[keep][-1]) + 1, 4, 4), np.nan)
                laid[row_frame[keep]] = pose[:n_rows][keep]
                pose = laid
            positions = pose[:, :3, 3].copy()
            rotations = pose[:, :3, :3].copy()
            n_frames_seen.append(positions.shape[0])
            self.rigid_bodies[body] = qtm.RigidBody(
                name=body, positions=positions, rotations=rotations
            )
            logger.debug(f"Loaded /optical/ body '{body}' with {positions.shape[0]} frames")

        if n_frames_seen:
            if grid is not None:
                self.frames = max(n_frames_seen)  # meta.n_frames counts rows
            else:
                self._reconcile_frame_count(max(n_frames_seen))

        if self.frame_rate <= 0.0:
            logger.warning(
                "Could not determine frame_rate_hz for multimodal /optical/; "
                "downstream HDF5 export may fail until sample rate is known."
            )

    def _reconcile_frame_count(self, max_frames: int) -> None:
        if self.frames == 0:
            self.frames = max_frames
        elif self.frames != max_frames:
            logger.warning(
                f"meta/n_frames ({self.frames}) disagrees with body frame count "
                f"({max_frames}); using {max_frames}."
            )
            self.frames = max_frames

    @staticmethod
    def _extract_pose_arrays(body: str, bg: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build ``(N, 3)`` positions and ``(N, 3, 3)`` rotations for one body.

        Prefers the dedicated ``position_mm`` / ``rotation_matrix`` datasets and
        falls back to decomposing ``pose_4x4`` if those are absent. Untracked
        frames are returned as ``NaN`` (Qualisys-compatible).
        """
        if "position_mm" in bg and "rotation_matrix" in bg:
            positions = np.asarray(bg["position_mm"][()], dtype=np.float64)
            rotations = np.asarray(bg["rotation_matrix"][()], dtype=np.float64)
        elif "pose_4x4" in bg:
            pose = np.asarray(bg["pose_4x4"][()], dtype=np.float64)
            positions = pose[:, :3, 3].copy()
            rotations = pose[:, :3, :3].copy()
        else:
            raise ValueError(
                f"Body '{body}' has neither 'position_mm'/'rotation_matrix' nor 'pose_4x4'"
            )

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"Body '{body}' positions must have shape (N, 3), got {positions.shape}")
        if rotations.shape[1:] != (3, 3):
            raise ValueError(f"Body '{body}' rotations must have shape (N, 3, 3), got {rotations.shape}")

        return positions, rotations

    def _parse_quality(self, f: h5py.File) -> None:
        """Read the optional ``quality/`` group (kept for diagnostics)."""
        if "quality" not in f:
            return
        for body in f["quality"].keys():
            qg = f["quality"][body]
            self.quality[body] = {
                key: qg[key][()] for key in qg.keys()
            }


class ArucoCameraMotionCaptureData(MotionCaptureData):
    """
    Concrete :class:`MotionCaptureData` for the recorder's ArUco camera sessions.

    Mirrors :class:`jts.core.QualysisMotionCaptureData` but loads HDF5 pose
    trajectories instead of Qualisys ``.mat`` exports.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ArUco camera data handler.

        Args:
            config: ``data_source`` configuration block.
        """
        super().__init__(config)

        self.aruco_data: Optional[ArucoCameraData] = None
        self.plot_utils = None

    def load_data(self, filename: str) -> None:
        """
        Load an ArUco camera tracking session HDF5 file.

        Args:
            filename: Path to standalone tracking HDF5 or multimodal ``session.h5``.
        """
        logger.info(f"Loading ArUco camera data from {filename}")

        self.aruco_data = ArucoCameraData(filename)
        # QualysisDataPlotUtils only needs a `.rigid_bodies` attribute, so it can
        # visualize our bodies too (the `visualization.raw_data` config option).
        self.plot_utils = qtm.QualysisDataPlotUtils(self.aruco_data)
        self.rigid_bodies = self.aruco_data.rigid_bodies
        self.frame_rate = self.aruco_data.frame_rate
        self.total_frames = self.aruco_data.frames

        logger.info(
            f"✓ Loaded {len(self.rigid_bodies)} rigid bodies, "
            f"{self.total_frames} frames at {self.frame_rate} Hz"
        )

    def get_rigid_body_transform(self, body_name: str,
                                 frame_interval: Optional[FrameInterval] = None) -> np.ndarray:
        """
        Get ``T_world<-body`` transforms for a rigid body.

        Args:
            body_name: Name of the rigid body (``HP``, ``MP`` or ``CT``).
            frame_interval: Optional frame interval to extract.

        Returns:
            Transformation matrices of shape ``(N, 4, 4)``.
        """
        if body_name not in self.rigid_bodies:
            raise ValueError(
                f"Rigid body '{body_name}' not found. Available: {list(self.rigid_bodies)}"
            )

        frame_tuple = frame_interval.to_tuple() if frame_interval else None

        return qtm.T_from_rigid_body(self.rigid_bodies[body_name], frame_tuple)

    def frame_times_us(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """MCU master time (µs) of each frame index; NaN for ``-1`` (synthetic)."""
        if self.aruco_data is None:
            return None
        t = self.aruco_data.timestamps.get("t_master_us")
        if t is None:
            return None
        t = np.asarray(t, dtype=np.float64)
        frames = np.asarray(frames, dtype=np.int64)
        out = np.full(frames.shape, np.nan)
        ok = (frames >= 0) & (frames < t.shape[0])
        out[ok] = t[frames[ok]]
        return out

    def compute_relative_transform(self, body1: str, body2: str,
                                   frame_interval: Optional[FrameInterval] = None) -> np.ndarray:
        """
        Compute the relative transform ``T_body1<-body2`` over time.

        Args:
            body1: Reference rigid body name.
            body2: Moving rigid body name.
            frame_interval: Optional frame interval.

        Returns:
            Relative transforms of shape ``(N, 4, 4)`` (``inv(T_world_body1) @ T_world_body2``).
        """
        for body in (body1, body2):
            if body not in self.rigid_bodies:
                raise ValueError(
                    f"Rigid body '{body}' not found. Available: {list(self.rigid_bodies)}"
                )

        rb_list = [self.rigid_bodies[body1], self.rigid_bodies[body2]]
        frame_intervals = [frame_interval.to_tuple()] if frame_interval else [(0, self.total_frames - 1)]

        result = qtm.QualysisData.compute_relative_transformations(rb_list, frame_intervals)
        key = list(result.keys())[0]

        return result[key][f'{body1}-{body2}']


def _estimate_period_us(t_us: np.ndarray) -> Optional[float]:
    """Frame period from row stamps with missing rows (median, then span/steps).

    Same rule as ``jtsmm.tools.timegrid.estimate_period_us``.
    """
    t = np.asarray(t_us, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return None
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        return None
    p0 = float(np.median(d))
    k = np.round(d / p0)
    ok = k >= 1
    if not np.any(ok):
        return p0
    return float(d[ok].sum() / k[ok].sum())


def _session_period_us(f: h5py.File, t_rows: np.ndarray) -> Optional[float]:
    """Trigger period: /triggers stamps+ids, /triggers@rate_hz, mcu config, rows."""
    if "triggers" in f:
        g = f["triggers"]
        if "t_master_us" in g and "trigger_id" in g:
            t = np.asarray(g["t_master_us"][()], dtype=np.float64)
            ids = np.asarray(g["trigger_id"][()], dtype=np.float64)
            if t.size >= 2 and ids.shape == t.shape and ids[-1] > ids[0]:
                return float((t[-1] - t[0]) / (ids[-1] - ids[0]))
        rate = g.attrs.get("rate_hz")
        if rate:
            return 1e6 / float(rate)
    if "meta" in f and "mcu_config_json" in f["meta"].attrs:
        try:
            cfg = json.loads(_as_str(f["meta"].attrs["mcu_config_json"]))
            rate = (cfg.get("trigger") or {}).get("rate_hz")
        except (ValueError, AttributeError):
            rate = None
        if rate:
            return 1e6 / float(rate)
    return _estimate_period_us(t_rows)


def _trigger_grid(f: h5py.File, t_rows: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    """``(row_frame, period_us)``: grid index per optical row (-1 = dropped).

    ``row_frame = round((t - t_rows[0]) / period)`` -- the rule shared with
    ``jtsmm.tools.timegrid.frame_grid_index``. Returns ``None`` (keep the row
    layout) when no sane period is available.
    """
    t = np.asarray(t_rows, dtype=np.float64)
    if t.size < 2 or not np.isfinite(t).all() or np.any(np.diff(t) < 0):
        logger.warning("optical t_master_us is not finite and sorted; frames stay row-indexed")
        return None
    period = _session_period_us(f, t)
    if not period or period <= 0.0:
        return None
    row_frame = np.round((t - t[0]) / period).astype(np.int64)
    if row_frame[-1] + 1 > 20 * t.size + 1000:
        logger.warning(
            f"trigger grid would have {row_frame[-1] + 1} frames for {t.size} optical "
            f"rows (period {period:.1f} us); frames stay row-indexed"
        )
        return None
    dup = np.flatnonzero(np.diff(row_frame) == 0) + 1
    if dup.size:
        logger.warning(f"{dup.size} optical rows share a trigger slot; keeping the first")
        row_frame[dup] = -1
    return row_frame, float(period)


def _frame_rate_from_t_master_us(t_us: np.ndarray) -> float:
    """Estimate Hz from MCU master timestamps (median reciprocal of Δt)."""
    t = np.asarray(t_us, dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size < 2:
        return 0.0
    dts = np.diff(finite)
    dts = dts[dts > 0]
    if dts.size == 0:
        return 0.0
    median_dt_s = float(np.median(dts)) * 1e-6
    if median_dt_s <= 0.0:
        return 0.0
    return 1.0 / median_dt_s


def _schema_major(version: str) -> Optional[int]:
    """Major number of ``"1"``, ``"1.0"``, ``"1.2"`` …; ``None`` if unparsable."""
    try:
        return int(str(version).strip().split(".")[0])
    except ValueError:
        return None


def _as_str(value: Any) -> str:
    """Decode an HDF5 attribute (which may be ``bytes``) to ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value) if value is not None else ""
