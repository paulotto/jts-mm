# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Read ``/optical/corners``: the 2D tag corners behind each ``/optical`` pose.

Recorded since 2026-10 (sessions before that hold only the fitted poses). With
the stored camera model the corners can be re-triangulated or re-fitted offline,
e.g. with an improved calibration or marker geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class CameraModel:
    """Camera model the live tracker used for these corners."""

    K: np.ndarray  # (3, 3), for the image the corners are in
    dist: np.ndarray  # (N,) OpenCV distortion coefficients
    image_size: tuple[int, int]  # (w, h) of that image
    T_world_cam: np.ndarray  # (4, 4) camera -> world (= cam0), mm
    rms_px: float | None = None
    extrinsics_rms_px: float | None = None


@dataclass
class OpticalCorners:
    """One row per (frame, camera, tag), in recording order."""

    cam_ids: list[str]
    optical_row: np.ndarray  # (N,) uint32, index into /optical/*
    t_master_us: np.ndarray  # (N,) uint64, = /optical/t_master_us[optical_row]
    cam: np.ndarray  # (N,) uint8, index into cam_ids
    tag_id: np.ndarray  # (N,) uint16
    corners_px: np.ndarray  # (N, 4, 2) float32; TL, TR, BR, BL
    calibration: dict[str, CameraModel] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.optical_row.shape[0])

    def for_row(self, row: int) -> dict[str, dict[int, np.ndarray]]:
        """``{cam_id: {tag_id: (4, 2) corners}}`` of one /optical row."""
        lo = int(np.searchsorted(self.optical_row, row, side="left"))
        hi = int(np.searchsorted(self.optical_row, row, side="right"))
        out: dict[str, dict[int, np.ndarray]] = {}
        for i in range(lo, hi):
            cam_id = self.cam_ids[int(self.cam[i])]
            out.setdefault(cam_id, {})[int(self.tag_id[i])] = self.corners_px[i]
        return out


def _str_list(values: Any) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(values).reshape(-1)]


def _read_calibration(cg: h5py.Group) -> dict[str, CameraModel]:
    out: dict[str, CameraModel] = {}
    if "calibration" not in cg:
        return out
    for cam_id, g in cg["calibration"].items():
        size = np.asarray(g.attrs["image_size"]).reshape(2)
        out[str(cam_id)] = CameraModel(
            K=np.asarray(g["K"], dtype=np.float64),
            dist=np.asarray(g["dist"], dtype=np.float64),
            image_size=(int(size[0]), int(size[1])),
            T_world_cam=np.asarray(g["T_world_cam"], dtype=np.float64),
            rms_px=float(g.attrs["rms_px"]) if "rms_px" in g.attrs else None,
            extrinsics_rms_px=(
                float(g.attrs["extrinsics_rms_px"]) if "extrinsics_rms_px" in g.attrs else None
            ),
        )
    return out


def read_optical_corners(source: str | Path | h5py.File) -> OpticalCorners | None:
    """Corners of a session file, or None when it has none (older recordings)."""
    if isinstance(source, h5py.File):
        return _read(source)
    with h5py.File(str(source), "r") as f:
        return _read(f)


def _read(f: h5py.File) -> OpticalCorners | None:
    if "optical" not in f or "corners" not in f["optical"]:
        return None
    cg = f["optical"]["corners"]
    calibration = _read_calibration(cg)
    cam_ids = _str_list(cg.attrs.get("cam_ids", []))
    if "corners_px" not in cg:
        return OpticalCorners(
            cam_ids=cam_ids,
            optical_row=np.empty(0, dtype=np.uint32),
            t_master_us=np.empty(0, dtype=np.uint64),
            cam=np.empty(0, dtype=np.uint8),
            tag_id=np.empty(0, dtype=np.uint16),
            corners_px=np.empty((0, 4, 2), dtype=np.float32),
            calibration=calibration,
        )
    return OpticalCorners(
        cam_ids=cam_ids,
        optical_row=np.asarray(cg["optical_row"], dtype=np.uint32),
        t_master_us=np.asarray(cg["t_master_us"], dtype=np.uint64),
        cam=np.asarray(cg["cam"], dtype=np.uint8),
        tag_id=np.asarray(cg["tag_id"], dtype=np.uint16),
        corners_px=np.asarray(cg["corners_px"], dtype=np.float32),
        calibration=calibration,
    )
