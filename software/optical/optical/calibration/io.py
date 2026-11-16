# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Read and write multi-camera calibration bundles (HDF5).

Schema is defined in ``docs/optical/02_calibration_and_frames.md``.
Each camera stores intrinsics (K, distortion) and extrinsics
(``T_world_cam``: homogeneous transform from camera frame to world frame).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from optical.calibration.geometry import scale_K


@dataclass(frozen=True)
class CameraIntrinsics:
    """Per-camera intrinsic calibration from a checkerboard solve."""

    K: np.ndarray  # (3, 3) pinhole camera matrix
    dist: np.ndarray  # (N,) OpenCV distortion coefficients
    image_size: Tuple[int, int]  # (width, height) in pixels
    rms_px: float  # mean reprojection error over calibration captures


@dataclass(frozen=True)
class CameraExtrinsics:
    """Rigid transform from camera coordinates to the shared world frame.

    ``world`` is the first calibrated camera (``cam0``): ``T_world_cam0 = I``.
    """

    T_world_cam: np.ndarray  # (4, 4) world(=cam0) <- cam
    # Mean per-camera board PnP residual during extrinsics capture. Says how well
    # each single board pose fit, NOT how good the camera-to-camera transform is.
    rms_px: float
    # Extrinsics quality: RMS (px) of board corners seen by cam_i, predicted from
    # cam0's board pose through T_world_cam (see calibration.extrinsics). NaN for a
    # single-camera rig or bundles written before this field existed.
    cross_view_rms_px: float = float("nan")


_RMS_PX_DESCRIPTION = (
    "mean per-camera board PnP residual (px) during extrinsics capture; "
    "not an extrinsic-quality metric (see cross_view_rms_px)"
)
_CROSS_VIEW_DESCRIPTION = (
    "RMS (px) of board corners detected by this camera vs. their prediction from "
    "cam0's board pose through T_world_cam, over accepted snapshots"
)


def _as_np(a: Any, shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Coerce to ndarray and optionally validate shape."""
    arr = np.asarray(a)
    if shape is not None and tuple(arr.shape) != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def ensure_T(T: np.ndarray) -> np.ndarray:
    """
    Normalize a 4×4 transform: float64 dtype and bottom row ``[0, 0, 0, 1]``.

    Args:
        T: Candidate homogeneous transform.

    Returns:
        Valid 4×4 rigid transform.
    """
    T = _as_np(T, (4, 4)).astype(np.float64)
    T[3, :] = np.array([0.0, 0.0, 0.0, 1.0])
    return T


def write_calibration_bundle(
    bundle_path: str | Path,
    *,
    rig_id: str,
    schema_version: str,
    cameras_intrinsics: Dict[str, CameraIntrinsics],
    cameras_extrinsics: Dict[str, CameraExtrinsics],
    fiducials: Dict[str, Any],
    notes: str = "",
    extrinsics_quality: Optional[Dict[str, Dict[str, Any]]] = None,
    meta_attrs: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write an HDF5 calibration bundle for a multi-camera rig.

    Layout: ``meta/``, ``cameras/<cam_id>/intrinsics|extrinsics``, ``fiducials/`` (attrs).
    The file is written to a temporary name in the same directory and renamed
    into place, so a crash or full disk never leaves a truncated bundle behind
    (the tracker would otherwise load a half-written file).

    Args:
        bundle_path: Output ``.h5`` path; parent directories are created.
        rig_id: Identifier for this physical camera rig.
        schema_version: Bundle format version (e.g. ``"1.0"``).
        cameras_intrinsics: Map ``cam_id`` → intrinsics per camera.
        cameras_extrinsics: Map ``cam_id`` → ``T_world_cam`` per camera.
        fiducials: ArUco metadata (dictionary, tag size, marker config path).
        notes: Optional free-text stored in ``meta``.
        extrinsics_quality: Optional per-camera arrays/scalars written under
            ``cameras/<cam_id>/extrinsics/quality/`` (e.g. per-snapshot
            cross-view error, inlier mask).
        meta_attrs: Optional extra ``meta`` attributes (e.g. gate results).

    Returns:
        Resolved path to the written file.
    """
    bundle_path = Path(bundle_path)
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = bundle_path.with_name(f".{bundle_path.name}.tmp-{os.getpid()}")
    try:
        _write_bundle_file(
            tmp_path,
            rig_id=rig_id,
            schema_version=schema_version,
            cameras_intrinsics=cameras_intrinsics,
            cameras_extrinsics=cameras_extrinsics,
            fiducials=fiducials,
            notes=notes,
            extrinsics_quality=extrinsics_quality or {},
            meta_attrs=meta_attrs or {},
        )
        os.replace(tmp_path, bundle_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return bundle_path


def _write_bundle_file(
    path: Path,
    *,
    rig_id: str,
    schema_version: str,
    cameras_intrinsics: Dict[str, CameraIntrinsics],
    cameras_extrinsics: Dict[str, CameraExtrinsics],
    fiducials: Dict[str, Any],
    notes: str,
    extrinsics_quality: Dict[str, Dict[str, Any]],
    meta_attrs: Dict[str, Any],
) -> None:
    created_utc = datetime.now(timezone.utc).isoformat()

    with h5py.File(path, "w") as f:
        meta = f.require_group("meta")
        meta.attrs["schema_version"] = schema_version
        meta.attrs["created_utc"] = created_utc
        meta.attrs["rig_id"] = rig_id
        meta.attrs["notes"] = notes
        for k, v in meta_attrs.items():
            meta.attrs[str(k)] = v

        cams_grp = f.require_group("cameras")
        for cam_id, intr in cameras_intrinsics.items():
            cg = cams_grp.require_group(cam_id)
            ig = cg.require_group("intrinsics")
            ig.create_dataset("K", data=_as_np(intr.K, (3, 3)).astype(np.float64))
            ig.create_dataset("dist", data=np.asarray(intr.dist, dtype=np.float64).ravel())
            ig.create_dataset("image_size", data=np.asarray(intr.image_size, dtype=np.int32))
            ig.create_dataset("rms_px", data=np.asarray(intr.rms_px, dtype=np.float64))

        for cam_id, ext in cameras_extrinsics.items():
            cg = cams_grp.require_group(cam_id)
            eg = cg.require_group("extrinsics")
            eg.create_dataset("T_world_cam", data=ensure_T(ext.T_world_cam))
            ds = eg.create_dataset("rms_px", data=np.asarray(ext.rms_px, dtype=np.float64))
            ds.attrs["description"] = _RMS_PX_DESCRIPTION
            ds = eg.create_dataset(
                "cross_view_rms_px", data=np.asarray(ext.cross_view_rms_px, dtype=np.float64)
            )
            ds.attrs["description"] = _CROSS_VIEW_DESCRIPTION
            quality = extrinsics_quality.get(cam_id)
            if quality:
                qg = eg.require_group("quality")
                for k, v in quality.items():
                    qg.create_dataset(str(k), data=np.asarray(v))

        fid = f.require_group("fiducials")
        for k, v in fiducials.items():
            fid.attrs[str(k)] = v


def read_calibration_bundle(bundle_path: str | Path) -> Dict[str, Any]:
    """
    Load a calibration bundle written by :func:`write_calibration_bundle`.

    Args:
        bundle_path: Path to ``calibration_bundle.h5``.

    Returns:
        Dict with ``meta``, ``cameras`` (per ``cam_id``: ``intrinsics``, ``extrinsics``),
        and ``fiducials``.
    """
    bundle_path = Path(bundle_path)
    out: Dict[str, Any] = {"meta": {}, "cameras": {}, "fiducials": {}}

    with h5py.File(bundle_path, "r") as f:
        out["meta"] = dict(f["meta"].attrs.items()) if "meta" in f else {}
        if "fiducials" in f:
            out["fiducials"] = dict(f["fiducials"].attrs.items())

        cams = out["cameras"]
        if "cameras" in f:
            for cam_id in f["cameras"].keys():
                cg = f["cameras"][cam_id]
                cams[cam_id] = {}
                if "intrinsics" in cg:
                    ig = cg["intrinsics"]
                    cams[cam_id]["intrinsics"] = {
                        "K": ig["K"][()],
                        "dist": ig["dist"][()],
                        "image_size": tuple(int(x) for x in ig["image_size"][()]),
                        "rms_px": float(np.asarray(ig["rms_px"][()])),
                    }
                if "extrinsics" in cg:
                    eg = cg["extrinsics"]
                    cams[cam_id]["extrinsics"] = {
                        "T_world_cam": eg["T_world_cam"][()],
                        "rms_px": float(np.asarray(eg["rms_px"][()])),
                        "cross_view_rms_px": (
                            float(np.asarray(eg["cross_view_rms_px"][()]))
                            if "cross_view_rms_px" in eg
                            else float("nan")
                        ),
                    }
                    if "quality" in eg:
                        cams[cam_id]["extrinsics"]["quality"] = {
                            k: eg["quality"][k][()] for k in eg["quality"].keys()
                        }

    return out


def load_cameras_from_bundle(
    bundle: Dict[str, Any],
) -> Tuple[Dict[str, CameraIntrinsics], Dict[str, CameraExtrinsics]]:
    """
    Parse typed intrinsics/extrinsics dicts from :func:`read_calibration_bundle`.

    Returns:
        ``(intrinsics, extrinsics)`` maps keyed by ``cam_id``.
    """
    intrinsics: Dict[str, CameraIntrinsics] = {}
    extrinsics: Dict[str, CameraExtrinsics] = {}
    for cam_id, cam_data in bundle.get("cameras", {}).items():
        if "intrinsics" in cam_data:
            i = cam_data["intrinsics"]
            intrinsics[cam_id] = CameraIntrinsics(
                K=i["K"],
                dist=i["dist"],
                image_size=tuple(i["image_size"]),
                rms_px=float(i["rms_px"]),
            )
        if "extrinsics" in cam_data:
            e = cam_data["extrinsics"]
            extrinsics[cam_id] = CameraExtrinsics(
                T_world_cam=ensure_T(e["T_world_cam"]),
                rms_px=float(e["rms_px"]),
                cross_view_rms_px=float(e.get("cross_view_rms_px", float("nan"))),
            )
    return intrinsics, extrinsics


def scale_camera_intrinsics(
    intr: CameraIntrinsics,
    image_size: Tuple[int, int],
) -> CameraIntrinsics:
    """
    Scale pinhole ``K`` from ``intr.image_size`` to a new capture resolution.

    Uses :func:`optical.calibration.geometry.scale_K` (pixel-centre aligned
    principal point; raises when the aspect ratio changes by > 0.5 %).
    Distortion coefficients are unchanged (they act on normalised
    coordinates). ``rms_px`` is kept as metadata from the original solve.
    """
    dst_w, dst_h = int(image_size[0]), int(image_size[1])
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"Invalid image_size {image_size!r}")
    src_w, src_h = int(intr.image_size[0]), int(intr.image_size[1])
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid calibration image_size {intr.image_size!r}")
    if (src_w, src_h) == (dst_w, dst_h):
        return intr
    return CameraIntrinsics(
        K=scale_K(intr.K, (src_w, src_h), (dst_w, dst_h)),
        dist=np.asarray(intr.dist, dtype=np.float64).copy(),
        image_size=(dst_w, dst_h),
        rms_px=float(intr.rms_px),
    )
