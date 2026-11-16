# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Read/write jaw-tracker pose trajectories (JTS offline contract).

Output schema follows ``docs/optical/03_pose_estimation_and_output.md`` §3 and
``docs/optical/04_migration_from_qualisys.md`` §3.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import h5py
import numpy as np

from optical.tracking.pose import is_valid_pose, nan_pose, pose_to_position_rotation


BODY_NAMES = ("HP", "MP", "CT")
MARKERS_3D_LAYOUT = "flat_v1"


@dataclass
class FrameRecord:
    """Per-frame tracker output consumed by JTS (Strategy B)."""

    frame_idx: int
    t_cam_ns: Dict[str, int] = field(default_factory=dict)
    t_master_us: Optional[int] = None
    pose_HP: np.ndarray = field(default_factory=nan_pose)
    pose_MP: np.ndarray = field(default_factory=nan_pose)
    pose_CT: np.ndarray = field(default_factory=nan_pose)
    quality: Dict[str, Any] = field(default_factory=dict)
    markers_3d: Optional[List[Dict[str, Any]]] = None

    def pose_for(self, body: str) -> np.ndarray:
        return {"HP": self.pose_HP, "MP": self.pose_MP, "CT": self.pose_CT}[body]

    def set_pose(self, body: str, T: np.ndarray) -> None:
        if body == "HP":
            self.pose_HP = T
        elif body == "MP":
            self.pose_MP = T
        elif body == "CT":
            self.pose_CT = T
        else:
            raise KeyError(body)


def _rate_from_arrays(t_master_us: np.ndarray, t_cam_ns: Mapping[str, np.ndarray]) -> float:
    masters = np.asarray(t_master_us, dtype=np.float64)
    masters = masters[np.isfinite(masters)]
    if masters.size >= 2:
        dt = np.diff(masters)
        dt = dt[dt > 0]
        if len(dt):
            return float(1e6 / np.median(dt))
    # Prefer a camera that appears on every frame.
    for cam_id in sorted(t_cam_ns):
        stamps = np.asarray(t_cam_ns[cam_id], dtype=np.float64)
        if stamps.size < 2 or not np.all(np.isfinite(stamps)):
            continue
        dt = np.diff(stamps)
        dt = dt[dt > 0]
        if len(dt):
            return float(1e9 / np.median(dt))
    return 0.0


def estimate_frame_rate_hz(frames: Sequence[FrameRecord]) -> float:
    """Nominal Hz from recorded timestamps (``t_master_us``, else ``t_cam_ns``)."""
    if len(frames) < 2:
        return 0.0
    masters = np.array(
        [np.nan if f.t_master_us is None else float(f.t_master_us) for f in frames]
    )
    cam_ids = sorted({cid for fr in frames for cid in fr.t_cam_ns})
    cams = {
        cid: np.array([float(fr.t_cam_ns.get(cid, np.nan)) for fr in frames]) for cid in cam_ids
    }
    return _rate_from_arrays(masters, cams)


def _attr_value(value: Any) -> Any:
    """HDF5-safe attribute: dicts / lists become JSON strings."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    if value is None:
        return ""
    return value


class TrackingSessionWriter:
    """
    Incremental writer for the tracking HDF5 (same layout as a one-shot write).

    Rows are buffered and appended to resizable datasets every ``flush_every``
    frames, and the file is flushed each time, so a crash or a killed process
    loses at most one buffer instead of the whole take. ``close()`` writes the
    final ``n_frames`` / ``frame_rate_hz`` / notes.

    Layout::

        meta/          attrs: schema_version, frame_rate_hz, n_frames, ...
        timestamps/    frame_idx, t_master_us, t_cam_ns/<cam_id>
        bodies/HP/     pose_4x4, position_mm, rotation_matrix, tracking_valid
        bodies/MP/     ...
        bodies/CT/     ...
        quality/       per-body tags_used, reprojection_rms_px, fit_rms_mm, n_cameras
        markers_3d/    optional flat table: frame_row, marker_id, corner_index,
                       position_mm, n_views, reprojection_rms_px
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        frame_rate_hz: float | None = None,
        rig_id: str = "",
        experiment_name: str = "",
        method: str = "pnp",
        schema_version: str = "1.0",
        notes: str = "",
        flush_every: int = 30,
    ) -> None:
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_rate_hz = frame_rate_hz
        self._notes = notes
        # 30 frames (1 s at 30 fps) keeps each synchronous flush to a few ms on a
        # Pi 5; 300 took 50-75 ms and cost both cameras a trigger every 10 s.
        self._flush_every = max(1, int(flush_every))
        self._buf: List[FrameRecord] = []
        self._journal: list | None = None
        self._n = 0
        self._extra_meta: Dict[str, Any] = {}
        self._closed = False

        self._f = h5py.File(self.path, "w")
        meta = self._f.create_group("meta")
        meta.attrs["schema_version"] = schema_version
        meta.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        meta.attrs["rig_id"] = rig_id
        meta.attrs["experiment_name"] = experiment_name
        meta.attrs["frame_rate_hz"] = float(frame_rate_hz or 0.0)
        meta.attrs["n_frames"] = 0
        meta.attrs["method"] = method
        meta.attrs["notes"] = notes
        meta.attrs["units_position"] = "mm"
        meta.attrs["bodies"] = np.array(BODY_NAMES, dtype="S8")

        ts = self._f.create_group("timestamps")
        self._ds(ts, "frame_idx", (), np.int64)
        self._ds(ts, "t_master_us", (), np.float64)
        ts.create_group("t_cam_ns")
        bodies = self._f.create_group("bodies")
        quality = self._f.create_group("quality")
        for body in BODY_NAMES:
            bg = bodies.create_group(body)
            self._ds(bg, "pose_4x4", (4, 4), np.float64)
            self._ds(bg, "position_mm", (3,), np.float64)
            self._ds(bg, "rotation_matrix", (3, 3), np.float64)
            self._ds(bg, "tracking_valid", (), bool)
            qg = quality.create_group(body)
            self._ds(qg, "tags_used", (), np.int32)
            self._ds(qg, "reprojection_rms_px", (), np.float64)
            self._ds(qg, "fit_rms_mm", (), np.float64)
            self._ds(qg, "n_cameras", (), np.int32)

    @staticmethod
    def _ds(group: h5py.Group, name: str, tail: tuple, dtype) -> h5py.Dataset:
        return group.create_dataset(
            name,
            shape=(0,) + tuple(tail),
            maxshape=(None,) + tuple(tail),
            dtype=dtype,
            chunks=(max(1, 1024 // max(1, int(np.prod(tail) if tail else 1))),) + tuple(tail),
        )

    def _append(self, ds: h5py.Dataset, data: np.ndarray) -> None:
        n0 = ds.shape[0]
        if self._journal is not None:
            self._journal.append((ds, n0))
        ds.resize(n0 + data.shape[0], axis=0)
        ds[n0:] = data

    @property
    def n_frames(self) -> int:
        return self._n + len(self._buf)

    def set_meta(self, **attrs: Any) -> None:
        """Extra ``meta/`` attributes written at close (dicts stored as JSON)."""
        self._extra_meta.update(attrs)

    def append(self, frame: FrameRecord) -> None:
        if self._closed:
            raise RuntimeError("TrackingSessionWriter is closed")
        self._buf.append(frame)
        if len(self._buf) >= self._flush_every:
            self.flush()

    def extend(self, frames: Sequence[FrameRecord]) -> None:
        for fr in frames:
            self.append(fr)

    def flush(self) -> None:
        """Append the buffered frames to every dataset, all or nothing.

        A failed write (disk full, Ctrl+C mid-flush) shrinks every touched
        dataset back to its previous length and keeps the frames buffered, so
        ``timestamps/*``, ``bodies/*`` and ``meta/n_frames`` never disagree.
        """
        if not self._buf:
            return
        frames = self._buf
        self._journal = []
        try:
            self._write_frames(frames)
        except BaseException:
            for ds, n_before in reversed(self._journal):
                try:
                    ds.resize(n_before, axis=0)
                except Exception:  # noqa: BLE001 - best effort, the original error wins
                    pass
            raise
        finally:
            self._journal = None
        self._buf = []

    def _write_frames(self, frames: List[FrameRecord]) -> None:
        n0, k = self._n, len(frames)
        f = self._f

        ts = f["timestamps"]
        self._append(ts["frame_idx"], np.array([fr.frame_idx for fr in frames], dtype=np.int64))
        self._append(
            ts["t_master_us"],
            np.array([np.nan if fr.t_master_us is None else float(fr.t_master_us) for fr in frames]),
        )
        tcam = ts["t_cam_ns"]
        cam_ids = set(tcam.keys()) | {cid for fr in frames for cid in fr.t_cam_ns}
        for cam_id in sorted(cam_ids):
            if cam_id not in tcam:
                ds = self._ds(tcam, cam_id, (), np.float64)
                if n0:
                    self._append(ds, np.full(n0, np.nan))  # camera appeared mid-take
            self._append(
                tcam[cam_id],
                np.array([float(fr.t_cam_ns.get(cam_id, np.nan)) for fr in frames]),
            )

        for body in BODY_NAMES:
            poses = np.stack([np.asarray(fr.pose_for(body), dtype=np.float64) for fr in frames])
            pos = np.full((k, 3), np.nan)
            rot = np.full((k, 3, 3), np.nan)
            valid = np.zeros(k, dtype=bool)
            for i in range(k):
                if is_valid_pose(poses[i]):
                    pos[i], rot[i] = pose_to_position_rotation(poses[i])
                    valid[i] = True
            bg = f["bodies"][body]
            self._append(bg["pose_4x4"], poses)
            self._append(bg["position_mm"], pos)
            self._append(bg["rotation_matrix"], rot)
            self._append(bg["tracking_valid"], valid)

            qs = [fr.quality.get(body, {}) for fr in frames]
            qg = f["quality"][body]
            self._append(qg["tags_used"], np.array([int(q.get("n_tags", 0)) for q in qs], np.int32))
            self._append(
                qg["reprojection_rms_px"],
                np.array([float(q.get("reprojection_rms_px", np.nan)) for q in qs]),
            )
            self._append(
                qg["fit_rms_mm"], np.array([float(q.get("fit_rms_mm", np.nan)) for q in qs])
            )
            self._append(qg["n_cameras"], np.array([int(q.get("n_cameras", 0)) for q in qs], np.int32))

        rows, mids, corners, xyz, views, rms = [], [], [], [], [], []
        for i, fr in enumerate(frames):
            for pt in fr.markers_3d or ():
                rows.append(n0 + i)
                mids.append(int(pt.get("marker_id", -1)))
                corners.append(int(pt.get("corner_index", -1)))
                xyz.append(np.asarray(pt["position_mm"], dtype=np.float64).reshape(3))
                views.append(int(pt.get("n_views", 0)))
                rms.append(float(pt.get("reprojection_rms_px", np.nan)))
        if rows:
            if "markers_3d" not in f:
                g = f.create_group("markers_3d")
                g.attrs["layout"] = MARKERS_3D_LAYOUT
                self._ds(g, "frame_row", (), np.int32)
                self._ds(g, "marker_id", (), np.int16)
                self._ds(g, "corner_index", (), np.int8)
                self._ds(g, "position_mm", (3,), np.float64)
                self._ds(g, "n_views", (), np.int8)
                self._ds(g, "reprojection_rms_px", (), np.float32)
            g = f["markers_3d"]
            self._append(g["frame_row"], np.array(rows, np.int32))
            self._append(g["marker_id"], np.array(mids, np.int16))
            self._append(g["corner_index"], np.array(corners, np.int8))
            self._append(g["position_mm"], np.stack(xyz))
            self._append(g["n_views"], np.array(views, np.int8))
            self._append(g["reprojection_rms_px"], np.array(rms, np.float32))

        self._n += k
        f["meta"].attrs["n_frames"] = self._n
        f.flush()

    def close(self, *, notes: str | None = None) -> Path:
        """Flush, write final meta and close (idempotent)."""
        if self._closed:
            return self.path
        try:
            self.flush()
            meta = self._f["meta"]
            if notes is not None:
                meta.attrs["notes"] = notes
            if self._frame_rate_hz is None:
                ts = self._f["timestamps"]
                cams = {cid: ts["t_cam_ns"][cid][()] for cid in ts["t_cam_ns"].keys()}
                meta.attrs["frame_rate_hz"] = _rate_from_arrays(ts["t_master_us"][()], cams)
            meta.attrs["n_frames"] = self._n
            for key, value in self._extra_meta.items():
                meta.attrs[key] = _attr_value(value)
            if not len(self._f["timestamps"]["t_cam_ns"].keys()):
                del self._f["timestamps"]["t_cam_ns"]  # one-shot writer never had it empty
        finally:
            self._closed = True
            self._f.close()
        return self.path

    def __enter__(self) -> "TrackingSessionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_tracking_session(
    output_path: str | Path,
    *,
    frames: Sequence[FrameRecord],
    frame_rate_hz: float | None = None,
    rig_id: str = "",
    experiment_name: str = "",
    method: str = "pnp",
    schema_version: str = "1.0",
    notes: str = "",
    extra_meta: Mapping[str, Any] | None = None,
) -> Path:
    """
    Write an HDF5 tracking session for JTS ``aruco_camera`` ingestion.

    ``frame_rate_hz`` defaults to the median rate inferred from frame timestamps
    (MCU ``t_master_us`` when present, else camera ``t_cam_ns``). Zero frames
    give a valid, empty session. Layout: :class:`TrackingSessionWriter`.
    """
    writer = TrackingSessionWriter(
        output_path,
        frame_rate_hz=frame_rate_hz,
        rig_id=rig_id,
        experiment_name=experiment_name,
        method=method,
        schema_version=schema_version,
        notes=notes,
        flush_every=max(1, len(frames)),
    )
    with writer:
        writer.extend(frames)
        if extra_meta:
            writer.set_meta(**dict(extra_meta))
    return writer.path


def read_markers_3d(f: h5py.File | str | Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Triangulated corners as flat arrays, or ``None`` when absent.

    Keys: ``frame_row`` (index into ``timestamps/``), ``marker_id``,
    ``corner_index``, ``position_mm`` (M, 3), plus ``n_views`` /
    ``reprojection_rms_px`` when stored. Reads the flat layout and the old
    one-group-per-point layout (``frame_<row>/point_<j>``).
    """
    if not isinstance(f, h5py.File):
        with h5py.File(f, "r") as fh:
            return read_markers_3d(fh)
    if "markers_3d" not in f:
        return None
    g = f["markers_3d"]
    if "frame_row" in g:
        return {k: g[k][()] for k in g.keys() if isinstance(g[k], h5py.Dataset)}

    rows, mids, corners, xyz = [], [], [], []
    for fname in sorted(g.keys()):
        if not fname.startswith("frame_"):
            continue
        row = int(fname.split("_", 1)[1])
        fg = g[fname]
        for pname in sorted(fg.keys()):
            pg = fg[pname]
            rows.append(row)
            mids.append(int(pg.attrs.get("marker_id", -1)))
            corners.append(int(pg.attrs.get("corner_index", -1)))
            xyz.append(np.asarray(pg["position_mm"][()], dtype=np.float64).reshape(3))
    return {
        "frame_row": np.array(rows, np.int32),
        "marker_id": np.array(mids, np.int16),
        "corner_index": np.array(corners, np.int8),
        "position_mm": np.stack(xyz) if xyz else np.zeros((0, 3)),
    }


def read_tracking_session(path: str | Path) -> Dict[str, Any]:
    """
    Load a tracking session written by :func:`write_tracking_session`.

    Returns:
        Dict with ``meta``, ``timestamps``, ``bodies``, ``quality`` and, when
        present, ``markers_3d`` (see :func:`read_markers_3d`).
    """
    path = Path(path)
    out: Dict[str, Any] = {"meta": {}, "timestamps": {}, "bodies": {}, "quality": {}}

    with h5py.File(path, "r") as f:
        if "meta" in f:
            out["meta"] = dict(f["meta"].attrs.items())

        if "timestamps" in f:
            ts = f["timestamps"]
            out["timestamps"] = {
                "frame_idx": ts["frame_idx"][()] if "frame_idx" in ts else None,
                "t_master_us": ts["t_master_us"][()] if "t_master_us" in ts else None,
            }
            if "t_cam_ns" in ts:
                out["timestamps"]["t_cam_ns"] = {
                    cam_id: ts["t_cam_ns"][cam_id][()] for cam_id in ts["t_cam_ns"].keys()
                }

        if "bodies" in f:
            for body in f["bodies"].keys():
                bg = f["bodies"][body]
                out["bodies"][body] = {
                    "pose_4x4": bg["pose_4x4"][()],
                    "position_mm": bg["position_mm"][()],
                    "rotation_matrix": bg["rotation_matrix"][()],
                    "tracking_valid": bg["tracking_valid"][()],
                }

        if "quality" in f:
            for body in f["quality"].keys():
                qg = f["quality"][body]
                out["quality"][body] = {
                    "tags_used": qg["tags_used"][()],
                    "reprojection_rms_px": qg["reprojection_rms_px"][()],
                    "n_cameras": qg["n_cameras"][()],
                }
                if "fit_rms_mm" in qg:
                    out["quality"][body]["fit_rms_mm"] = qg["fit_rms_mm"][()]

        m3d = read_markers_3d(f)
        if m3d is not None:
            out["markers_3d"] = m3d

    return out
