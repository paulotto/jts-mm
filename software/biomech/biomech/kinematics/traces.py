# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mandible position traces for the evaluation scripts (experiments/5_4_2 and 5_4_3).

Two sources, in order of preference:

* the JTS model-frame landmark (``jaw_motion.h5`` written by
  ``jtsmm.tools.run_jts_analysis``), mm;
* the optical relative pose: mandibular marker origin in the cranial marker
  frame, ``R_craᵀ (t_mta − t_cra)``, mm.

``opening_1d`` reduces either to the scalar opening ``‖p − p_rest‖`` the
peak-to-peak cycle detection runs on. These are the earlier definitions; the ICORR pipeline uses the incisor point in the maxillary frame
(``biomech.kinematics.analysis``) instead.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from biomech.io.session import SessionData
from biomech.kinematics.se3 import relative

JTS_GROUPS = ("T_model_origin_mand_landmark_t_smooth", "T_model_origin_mand_landmark_t")


def _folder(path: str | Path) -> Path:
    p = Path(path)
    return p.parent if p.name == "session.h5" else p


def load_optical_trace(session: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """``(t_master_us int64 (T,), xyz mm (T, 3))`` of the mandible in the cranial frame.

    None when the take has no optical group or misses one of the two markers.
    Untracked frames are NaN.
    """
    with SessionData(_folder(session)) as sd:
        opt = sd.optical()
    if opt is None or "mta" not in opt.poses or "cra" not in opt.poses:
        return None
    mta, cra = opt.poses["mta"], opt.poses["cra"]
    n = min(len(opt.t_us), len(mta), len(cra))
    rel = relative(cra[:n], mta[:n])
    return opt.t_us[:n].astype(np.int64), rel[:, :3, 3]


def load_jts_trace(path: str | Path) -> tuple[np.ndarray, np.ndarray] | None:
    """``(t_us from 0 int64, xyz mm)`` of the JTS landmark (smoothed when present)."""
    path = Path(path)
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        for name in JTS_GROUPS:
            g = f.get(name)
            if g is None or "translations" not in g:
                continue
            xyz = np.asarray(g["translations"], dtype=np.float64)
            unit = str(g.attrs.get("unit") or g.attrs.get("units") or "mm")
            if unit.lower() in ("m", "meter", "metres", "meters"):
                xyz = xyz * 1000.0
            fs = float(g.attrs.get("sample_rate") or 60.0)
            t = (np.arange(xyz.shape[0]) / fs * 1e6).astype(np.int64)
            return t, xyz
    return None


def load_kinematic_trace(session: str | Path, jts_path: str | Path | None = None
                         ) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Preferred trace of one take: ``(t_master_us, xyz mm, source)``.

    ``source`` is ``"jts"`` when ``jts_path`` (default ``<take>/jaw_motion.h5``)
    exists, else ``"optical_relative"``. JTS frames carry no master time, so they
    are placed on the optical clock: same length -> 1:1; otherwise the JTS trace
    is linearly resampled by index onto the optical frames (kept from the
    original §5.4.2 script for comparability).
    """
    folder = _folder(session)
    jp = Path(jts_path) if jts_path is not None else folder / "jaw_motion.h5"
    jts = load_jts_trace(jp) if jp.is_file() else None
    opt = load_optical_trace(folder)
    if jts is None:
        return None if opt is None else (opt[0], opt[1], "optical_relative")
    t, xyz = jts
    if opt is not None:
        t_opt = opt[0]
        if len(xyz) != len(t_opt):
            idx = np.linspace(0, len(xyz) - 1, len(t_opt))
            xyz = np.column_stack(
                [np.interp(idx, np.arange(len(xyz)), xyz[:, k]) for k in range(3)])
        t = t_opt
    return t, xyz, "jts"


def opening_1d(xyz: np.ndarray, rest: np.ndarray | None = None) -> np.ndarray:
    """Scalar opening ``‖p − origin‖`` (mm); NaN rows stay NaN.

    ``origin`` is the mean of the finite ``rest`` samples, else the mean of the
    finite samples of ``xyz`` itself.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    origin = None
    if rest is not None:
        r = np.asarray(rest, dtype=np.float64)
        r = r[np.isfinite(r).all(axis=1)] if r.size else r.reshape(0, 3)
        if r.size:
            origin = r.mean(axis=0)
    if origin is None:
        origin = xyz[finite].mean(axis=0) if finite.any() else np.zeros(3)
    out = np.linalg.norm(xyz - origin, axis=1)
    out[~finite] = np.nan
    return out


def sample_rate_hz(t_us: np.ndarray, default: float = 60.0) -> float:
    """Median rate of a stamp series; ``default`` when it has no positive step."""
    dt = np.diff(np.asarray(t_us, dtype=np.float64))
    dt = dt[dt > 0]
    return 1e6 / float(np.median(dt)) if dt.size else float(default)


__all__ = ["load_jts_trace", "load_kinematic_trace", "load_optical_trace", "opening_1d",
           "sample_rate_hz"]
