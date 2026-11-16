#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Visualize a recorded JTS-MM session with the vendored JTS analysis stack.

Plots, on a shared master-clock time axis:

- optical world-frame X/Y/Z (HP / MP / CT via ``jts.aruco_camera``)
- IMU world-frame X/Y/Z (ZUPT double-integral of linear accel, or ``/derived/pose``)
- EMG muscles + DRL (CH7 ``bias_debug`` skipped), high-passed and with mains
  removed at the recorded timestamps (same chain as ``plot_emg --mains offline``)

Lines break wherever a stream has a time gap (dropped frames / samples).

Trajectories are also written as a JTS HDF5 sidecar (``translations`` / ``rotations``)
so ``jts.plotly_visualization.JawMotionVisualizer`` and ``jts.helper`` can reopen them.

Examples::

    python -m jtsmm.tools.visualize_session path/to/session.h5
    python -m jtsmm.tools.visualize_session path/to/session_dir --no-show
    python -m jtsmm.tools.visualize_session session.h5 --out viz/ --png
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm.io.session_layout import session_h5_path
from jtsmm.io.session_reader import Session
from jtsmm.imu.gravity_lin_accel import G
from jtsmm.imu.kinematics import ZuptIntegrator, apply_mounting_quat
from jtsmm.imu.mounting import load_mounting_rotation
from jtsmm.protocol.records import FUSION_REPORT_IDS
from jtsmm.tools.plot_mp_trajectory import (
    PhaseInterval,
    _PHASE_COLORS,
    load_phases,
)
from jtsmm.tools.jts_path import ensure_jts_on_path
from jtsmm.tools.timegrid import break_gaps


# IMU HDF5 group → JTS rigid-body name used to seed world origin.
_IMU_SEED_BODY = {"imu_mta": "MP", "imu_cra": "HP"}
_OPTICAL_LABEL = {"HP": "optical_HP", "MP": "optical_MP", "CT": "optical_CT"}
_IMU_LABEL = {"imu_mta": "imu_MTA_zupt", "imu_cra": "imu_CRA_zupt"}
_AXIS = ("X", "Y", "Z")
_OPT_COLORS = {"HP": "#8e44ad", "MP": "#27ae60", "CT": "#c0392b"}
_IMU_COLORS = {"imu_mta": "#1e8449", "imu_cra": "#6c3483"}
_EMG_COLORS = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#7f8c8d",
    "#2c3e50",
)


@dataclass
class NamedTrajectory:
    """One SE(3) stream in millimetres, timestamped on the MCU master clock."""

    name: str
    T: np.ndarray  # (N, 4, 4)
    t_us: np.ndarray
    sample_rate_hz: float
    unit: str = "mm"
    source: str = ""  # "optical" | "imu_zupt" | "derived"


@dataclass
class EmgPlot:
    t_us: np.ndarray
    samples_uv: np.ndarray  # (N, C)
    channel_names: list[str]
    fs_hz: float


@dataclass
class SessionViz:
    path: Path
    t0_us: float
    optical: list[NamedTrajectory] = field(default_factory=list)
    imu: list[NamedTrajectory] = field(default_factory=list)
    emg: EmgPlot | None = None
    phases: list[PhaseInterval] = field(default_factory=list)
    # Viewer operator notes: (t_master_us, text), from /events kind "note".
    notes: list[tuple[int, str]] = field(default_factory=list)


def resolve_session_h5(path: Path) -> Path:
    """Accept a ``session.h5`` file or a session directory containing one."""
    path = path.expanduser().resolve()
    if path.is_dir():
        candidate = session_h5_path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no session.h5 in directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path


def _sample_rate_hz(t_us: np.ndarray, fallback: float = 1.0) -> float:
    t = np.asarray(t_us, dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size < 2:
        return float(fallback)
    dts = np.diff(finite)
    dts = dts[dts > 0]
    if dts.size == 0:
        return float(fallback)
    median_dt_s = float(np.median(dts)) * 1e-6
    if median_dt_s <= 0.0:
        return float(fallback)
    return 1.0 / median_dt_s


def _t_s(t_us: np.ndarray, t0_us: float) -> np.ndarray:
    return (np.asarray(t_us, dtype=np.float64) - t0_us) * 1e-6


def _phase_spans_s(
    phases: list[PhaseInterval], t0_us: float
) -> list[tuple[str, float, float]]:
    return [
        (p.name, (p.t_start_us - t0_us) * 1e-6, (p.t_end_us - t0_us) * 1e-6)
        for p in phases
    ]


def _xyz_mm(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=np.float64)[:, :3, 3].copy()


def _finite_pose_mask(T: np.ndarray) -> np.ndarray:
    p = np.asarray(T)
    if p.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(p).reshape(len(p), -1).all(axis=1)


def _first_finite_xyz(T: np.ndarray) -> np.ndarray:
    mask = _finite_pose_mask(T)
    if not np.any(mask):
        return np.zeros(3, dtype=np.float64)
    return T[int(np.flatnonzero(mask)[0]), :3, 3].astype(np.float64)


def _quat_wxyz_to_matrices(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"expected (N, 4) wxyz quaternions, got {q.shape}")
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    qn = q / n
    w, x, y, z = qn[:, 0], qn[:, 1], qn[:, 2], qn[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 0, 1] = 2.0 * (x * y - z * w)
    out[:, 0, 2] = 2.0 * (x * z + y * w)
    out[:, 1, 0] = 2.0 * (x * y + z * w)
    out[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 1, 2] = 2.0 * (y * z - x * w)
    out[:, 2, 0] = 2.0 * (x * z - y * w)
    out[:, 2, 1] = 2.0 * (y * z + x * w)
    out[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    bad = ~np.isfinite(q).all(axis=1)
    out[bad] = np.nan
    return out


def _slerp_quat_wxyz(t_q_us: np.ndarray, quat: np.ndarray, t_out_us: np.ndarray) -> np.ndarray:
    """Interpolate wxyz quaternions onto ``t_out_us`` (MCU master µs)."""
    from jtsmm.imu.kinematics import quat_slerp

    t_q = np.asarray(t_q_us, dtype=np.float64)
    q = np.asarray(quat, dtype=np.float64)
    t_out = np.asarray(t_out_us, dtype=np.float64)
    order = np.argsort(t_q)
    t_q, q = t_q[order], q[order]
    finite = np.isfinite(t_q) & np.isfinite(q).all(axis=1)
    t_q, q = t_q[finite], q[finite]
    out = np.full((t_out.shape[0], 4), np.nan, dtype=np.float64)
    if t_q.size == 0:
        return out
    if t_q.size == 1:
        out[:] = q[0]
        return out
    keep = np.ones(t_q.size, dtype=bool)
    keep[1:] = np.diff(t_q) > 0
    t_q, q = t_q[keep], q[keep]
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    q = q / n
    idx = np.searchsorted(t_q, t_out, side="right") - 1
    idx = np.clip(idx, 0, t_q.size - 2)
    t0 = t_q[idx]
    t1 = t_q[idx + 1]
    dt = t1 - t0
    u = np.where(dt > 0, (t_out - t0) / dt, 0.0)
    u = np.clip(u, 0.0, 1.0)
    for i in range(t_out.size):
        q0 = tuple(float(c) for c in q[idx[i]])
        q1 = tuple(float(c) for c in q[idx[i] + 1])
        out[i] = quat_slerp(q0, q1, float(u[i]))  # type: ignore[assignment]
    return out


def _primary_quat_rows(imu: Any) -> np.ndarray:
    """Rows of the slot's primary fusion report (boolean mask over ``imu.quat``).

    A slot can record more than one fusion report (e.g. game + geomagnetic
    rotation vector); they are different estimates of different frames, so
    they must not be interpolated into each other. Keep the report named by
    ``fusion_mode``, else the most frequent report id.
    """
    rid = getattr(imu, "quat_report_id", None)
    n = int(imu.n)
    if rid is None or len(rid) != n or n == 0:
        return np.ones(n, dtype=bool)
    rid = np.asarray(rid, dtype=np.int64)
    ids, counts = np.unique(rid[rid > 0], return_counts=True)
    if ids.size <= 1:
        return np.ones(n, dtype=bool)
    by_name = {v: k for k, v in FUSION_REPORT_IDS.items()}
    want = by_name.get(str(getattr(imu, "fusion_mode", "")))
    if want not in ids:
        want = int(ids[np.argmax(counts)])
    return rid == want


def world_linear_accel(
    t_q_us: np.ndarray, q_chip: np.ndarray, t_lin_us: np.ndarray, a_chip: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(a_world, R_wc)`` at the linear-accel timestamps.

    The BNO085 reports linear acceleration in the chip frame and the fusion
    quaternion as world <- chip, so ``a_world = R_wc a_chip``. The IMU
    mounting rotation (body <- chip) plays no part in that: it only changes
    which frame a plotted *orientation* refers to. Rows without a valid
    orientation are NaN.
    """
    R_wc = _quat_wxyz_to_matrices(_slerp_quat_wxyz(t_q_us, q_chip, t_lin_us))
    a = np.asarray(a_chip, dtype=np.float64)
    a_world = np.einsum("nij,nj->ni", np.nan_to_num(R_wc, nan=0.0), a)
    bad = ~np.isfinite(R_wc).reshape(R_wc.shape[0], -1).all(axis=1)
    a_world[bad] = np.nan
    return a_world, R_wc


def _apply_mounting(quat_wxyz: np.ndarray, R_mount: np.ndarray | None) -> np.ndarray:
    if R_mount is None:
        return quat_wxyz
    out = np.empty_like(quat_wxyz, dtype=np.float64)
    for i, q in enumerate(quat_wxyz):
        if not np.isfinite(q).all():
            out[i] = np.nan
            continue
        out[i] = apply_mounting_quat(
            (float(q[0]), float(q[1]), float(q[2]), float(q[3])), R_mount
        )
    return out


def _integrate_linear_accel_mm(
    t_us: np.ndarray,
    accel_mps2: np.ndarray,
    *,
    zupt_threshold: float = 0.3,
    zupt_hold_s: float = 0.1,
    auto_reset_s: float = 1e9,
) -> np.ndarray:
    """Double-integrate world-frame linear accel → millimetres (ZUPT estimate)."""
    t_s = np.asarray(t_us, dtype=np.float64) * 1e-6
    a = np.asarray(accel_mps2, dtype=np.float64)
    if a.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(t_s)
    a = a.copy()
    if np.any(finite):
        a[finite] -= np.median(a[finite], axis=0)
    zupt = ZuptIntegrator(
        zupt_threshold=zupt_threshold,
        zupt_hold_s=zupt_hold_s,
        auto_reset_s=auto_reset_s,
    )
    pos_m = np.zeros((t_s.size, 3), dtype=np.float64)
    for i in range(t_s.size):
        if not finite[i]:
            pos_m[i] = np.nan
            continue
        p, _ = zupt.update(float(t_s[i]), (float(a[i, 0]), float(a[i, 1]), float(a[i, 2])))
        pos_m[i] = p
    return pos_m * 1000.0


def _poses_from_R_t(R: np.ndarray, t_mm: np.ndarray) -> np.ndarray:
    n = t_mm.shape[0]
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = t_mm
    return T


def load_optical_with_jts(session_h5: Path) -> list[NamedTrajectory]:
    """Load HP/MP/CT world poses through the JTS ArUco camera adapter."""
    ensure_jts_on_path()
    from jts.aruco_camera import ArucoCameraMotionCaptureData

    handler = ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(session_h5))
    t_us = handler.aruco_data.timestamps.get("t_master_us")
    n = int(handler.total_frames)
    if t_us is None:
        t_us = np.arange(n, dtype=np.float64)
    else:
        t_us = np.asarray(t_us, dtype=np.float64)
    fs = float(handler.frame_rate) or _sample_rate_hz(t_us, fallback=60.0)
    out: list[NamedTrajectory] = []
    for body in ("HP", "MP", "CT"):
        if body not in handler.rigid_bodies:
            continue
        T = np.asarray(handler.get_rigid_body_transform(body), dtype=np.float64)
        if T.shape[0] != t_us.shape[0]:
            n_use = min(T.shape[0], t_us.shape[0])
            T, t_body = T[:n_use], t_us[:n_use]
        else:
            t_body = t_us
        out.append(
            NamedTrajectory(
                name=_OPTICAL_LABEL[body],
                T=T,
                t_us=t_body,
                sample_rate_hz=fs,
                source="optical",
            )
        )
    return out


def _load_derived_imu_poses(session_h5: Path) -> list[NamedTrajectory]:
    """Prefer ``/derived/pose`` when the offline IMU pipeline has run."""
    out: list[NamedTrajectory] = []
    with h5py.File(session_h5, "r") as f:
        if "derived" not in f or "pose" not in f["derived"]:
            return out
        g = f["derived"]["pose"]
        t_us = (
            np.asarray(g["t_master_us"][()], dtype=np.float64)
            if "t_master_us" in g
            else None
        )
        mapping = (
            ("pose_mta", "imu_MTA_derived"),
            ("pose_cra", "imu_CRA_derived"),
            ("pose_4x4", "imu_fused_derived"),
        )
        for ds, name in mapping:
            if ds not in g:
                continue
            T = np.asarray(g[ds][()], dtype=np.float64)
            if T.ndim != 3 or T.shape[1:] != (4, 4):
                continue
            t = t_us if t_us is not None else np.arange(T.shape[0], dtype=np.float64)
            if t.shape[0] != T.shape[0]:
                n = min(t.shape[0], T.shape[0])
                T, t = T[:n], t[:n]
            out.append(
                NamedTrajectory(
                    name=name,
                    T=T,
                    t_us=t,
                    sample_rate_hz=_sample_rate_hz(t),
                    source="derived",
                )
            )
    return out


def load_imu_trajectories(
    session_h5: Path,
    optical: list[NamedTrajectory],
    *,
    auto_reset_s: float = 1e9,
) -> list[NamedTrajectory]:
    """Build millimetre IMU poses from ``/derived/pose`` or ZUPT of linear accel.

    Without a linear_accel stream (kinematics_accel) the gravity-included accel
    is used: in the world frame it is linear accel + g along Z, removed here.
    """
    derived = _load_derived_imu_poses(session_h5)
    if derived:
        return derived

    optical_by_short = {tr.name.replace("optical_", ""): tr for tr in optical}
    out: list[NamedTrajectory] = []
    with Session.open(session_h5) as session:
        for slot, seed_body in _IMU_SEED_BODY.items():
            imu = getattr(session, slot, None)
            if imu is None:
                continue
            lin = imu.linear_accel
            with_gravity = lin is None or lin.n == 0
            if with_gravity:
                lin = imu.accel
            if lin is None or lin.n == 0 or imu.n == 0:
                continue
            rows = _primary_quat_rows(imu)
            q_chip = np.asarray(imu.quat, dtype=np.float64)[rows]
            t_q = np.asarray(imu.t_us)[rows]
            a_world, R_wc = world_linear_accel(t_q, q_chip, lin.t_us, lin.data)
            if with_gravity:
                a_world[:, 2] -= G
            R_mount = load_mounting_rotation(session_h5, slot)
            if R_mount is None:
                R_world = R_wc
            else:
                q_body = _apply_mounting(q_chip, R_mount)
                R_world = _quat_wxyz_to_matrices(_slerp_quat_wxyz(t_q, q_body, lin.t_us))
            pos_mm = _integrate_linear_accel_mm(
                lin.t_us, a_world, auto_reset_s=auto_reset_s
            )
            seed = optical_by_short.get(seed_body)
            origin = _first_finite_xyz(seed.T) if seed is not None else np.zeros(3)
            pos_mm = pos_mm + origin
            T = _poses_from_R_t(R_world, pos_mm)
            out.append(
                NamedTrajectory(
                    name=_IMU_LABEL[slot],
                    T=T,
                    t_us=np.asarray(lin.t_us, dtype=np.float64),
                    sample_rate_hz=_sample_rate_hz(lin.t_us, fallback=250.0),
                    source="imu_zupt",
                )
            )
    return out


def load_emg(session_h5: Path, *, filtered: bool = True) -> EmgPlot | None:
    """Muscle channels + DRL in µV, filtered the way ``plot_emg`` does.

    ``samples_uv`` is raw ADC output: electrode DC offsets of tens of mV and
    mains dwarf the EMG. With ``filtered`` (default) the 20 Hz high-pass and
    the block-wise mains fit at the recorded timestamps are applied
    (:func:`jtsmm.tools.plot_emg.filter_emg`, ``mains_mode="offline"``). CH7
    ``bias_debug`` is not an electrode signal and is skipped.
    """
    from jtsmm.tools.plot_emg import filter_emg, viewer_channel_names
    from jtsmm.viewer.sources import _slice_emg_for_viewer

    with Session.open(session_h5) as session:
        if session.emg is None or session.emg.n == 0:
            return None
        stored = list(session.emg.channel_names)
        uv = np.asarray(session.emg.samples_uv, dtype=np.float32)
        t_us = np.asarray(session.emg.t_us, dtype=np.float64)
        fs = float(session.emg.fs_hz)
    if uv.ndim != 2:
        return None
    if len(stored) < uv.shape[1]:
        stored = stored + [f"ch{i + 1}" for i in range(len(stored), uv.shape[1])]
    uv = _slice_emg_for_viewer(uv)
    names = viewer_channel_names(stored, uv.shape[1])
    if filtered and uv.shape[0] > 0:
        uv, _env = filter_emg(uv, fs, mains_mode="offline", t_us=t_us)
    return EmgPlot(
        t_us=t_us,
        samples_uv=np.asarray(uv, dtype=np.float32),
        channel_names=names[: uv.shape[1]],
        fs_hz=fs,
    )


def load_session_viz(session_h5: Path, *, imu_auto_reset_s: float = 1e9) -> SessionViz:
    session_h5 = resolve_session_h5(session_h5)
    with h5py.File(session_h5, "r") as f:
        phases = load_phases(f)

    optical: list[NamedTrajectory] = []
    try:
        optical = load_optical_with_jts(session_h5)
    except (ImportError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"warning: optical via JTS not loaded ({exc})", file=sys.stderr)

    imu = load_imu_trajectories(session_h5, optical, auto_reset_s=imu_auto_reset_s)
    emg = load_emg(session_h5)

    starts: list[float] = []
    for tr in (*optical, *imu):
        finite = np.isfinite(tr.t_us)
        if np.any(finite):
            starts.append(float(tr.t_us[finite][0]))
    if emg is not None and emg.t_us.size:
        starts.append(float(emg.t_us[0]))
    t0 = min(starts) if starts else 0.0
    with Session.open(session_h5) as session:
        notes = [(int(n["t_master_us"]), str(n["text"])) for n in session.notes()]
    return SessionViz(
        path=session_h5, t0_us=t0, optical=optical, imu=imu, emg=emg, phases=phases,
        notes=notes,
    )


def write_jts_sidecar(viz: SessionViz, out_h5: Path) -> Path:
    """Store optical + IMU poses in the JTS translations/rotations layout."""
    ensure_jts_on_path()
    from jts.helper import store_transformations

    trajs = [tr for tr in (*viz.optical, *viz.imu) if tr.T.shape[0] > 0]
    if not trajs:
        raise ValueError("no optical or IMU trajectories to write")
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    store_transformations(
        [tr.T for tr in trajs],
        [tr.sample_rate_hz for tr in trajs],
        out_h5,
        metadata=[f"{tr.source} from {viz.path.name}" for tr in trajs],
        store_as_quaternion=False,
        group_names=[tr.name for tr in trajs],
        unit="mm",
    )
    return out_h5


def _minmax_downsample(
    t: np.ndarray, y: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep EMG spikes visible by storing min and max of each bin."""
    n = int(t.shape[0])
    if n <= max_points or max_points < 4:
        return t, y
    bins = max(2, max_points // 2)
    edges = np.linspace(0, n, bins + 1, dtype=int)
    t_out: list[float] = []
    y_out: list[float] = []
    for i0, i1 in zip(edges[:-1], edges[1:]):
        if i1 <= i0:
            continue
        sl = y[i0:i1]
        ts = t[i0:i1]
        imin = int(np.argmin(sl))
        imax = int(np.argmax(sl))
        if ts[imin] <= ts[imax]:
            t_out.extend((float(ts[imin]), float(ts[imax])))
            y_out.extend((float(sl[imin]), float(sl[imax])))
        else:
            t_out.extend((float(ts[imax]), float(ts[imin])))
            y_out.extend((float(sl[imax]), float(sl[imin])))
    return np.asarray(t_out), np.asarray(y_out)


def _gapped_xyz(tr: NamedTrajectory, t0_us: float) -> tuple[np.ndarray, np.ndarray]:
    """``(t_s, xyz)`` with NaN rows kept and a NaN inserted at every time gap."""
    t = _t_s(tr.t_us, t0_us)
    xyz = _xyz_mm(tr.T)
    ok = np.isfinite(t)
    period = 1.0 / tr.sample_rate_hz if tr.sample_rate_hz > 0 else None
    t_g, xyz_g = break_gaps(t[ok], xyz[ok], period=period)
    return t_g, xyz_g


def _emg_downsampled(
    t: np.ndarray, y: np.ndarray, fs_hz: float, max_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Min/max-downsample each gap-free run separately; NaN between runs."""
    if t.size < 2:
        return t, y
    period = 1.0 / fs_hz if fs_hz > 0 else float(np.median(np.diff(t)))
    cuts = np.flatnonzero(np.diff(t) > 1.5 * period) + 1
    if cuts.size == 0:
        return _minmax_downsample(t, y, max_points)
    t_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, t.size]):
        budget = max(4, int(max_points * (b - a) / t.size))
        td, yd = _minmax_downsample(t[a:b], y[a:b], budget)
        t_parts += [np.asarray(td, dtype=np.float64), np.array([np.nan])]
        y_parts += [np.asarray(yd, dtype=np.float64), np.array([np.nan])]
    return np.concatenate(t_parts[:-1]), np.concatenate(y_parts[:-1])


def _shade_phases(ax: Any, spans: list[tuple[str, float, float]], *, label: bool) -> None:
    for i, (name, t0, t1) in enumerate(spans):
        ax.axvspan(t0, t1, color=_PHASE_COLORS[i % len(_PHASE_COLORS)], alpha=0.28, lw=0, zorder=0)
        if label:
            mid = 0.5 * (t0 + t1)
            ax.text(
                mid,
                0.98,
                name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                rotation=90 if (t1 - t0) < 0.4 else 0,
                clip_on=True,
                zorder=3,
            )


def _mark_notes(ax: Any, notes_s: list[tuple[float, str]], *, label: bool) -> None:
    """Dotted vertical line per viewer note; text on the top axis only."""
    for t, text in notes_s:
        ax.axvline(t, color="#555", ls=":", lw=0.9, zorder=4)
        if label:
            ax.text(t, 0.02, text[:40], transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="bottom", fontsize=7, color="#333", clip_on=True, zorder=5)


def plot_matplotlib(
    viz: SessionViz,
    *,
    save: Path | None,
    show: bool,
    show_phases: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from jtsmm.tools.plot_emg import _FILTER_SETTLE_S, _settled_ylim

    written: list[Path] = []
    spans = _phase_spans_s(viz.phases, viz.t0_us) if show_phases else []
    notes_s = [((t - viz.t0_us) * 1e-6, text) for t, text in viz.notes]
    legend_phases = [
        Patch(facecolor=_PHASE_COLORS[i % len(_PHASE_COLORS)], alpha=0.45, label=name)
        for i, (name, *_rest) in enumerate(spans)
    ]

    def _xyz_figure(trajs: list[NamedTrajectory], title: str, colors: dict[str, str]) -> Any:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 7.5), constrained_layout=True)
        for ax, axis in zip(axes, range(3)):
            _shade_phases(ax, spans, label=(ax is axes[0]))
            _mark_notes(ax, notes_s, label=(ax is axes[0]))
            for tr in trajs:
                t, xyz = _gapped_xyz(tr, viz.t0_us)
                color = colors.get(tr.name, None)
                # Match by suffix for IMU labels.
                if color is None:
                    for key, col in colors.items():
                        if key in tr.name or tr.name.endswith(key):
                            color = col
                            break
                ax.plot(t, xyz[:, axis], lw=1.0, label=tr.name, color=color, zorder=2)
            ax.set_ylabel(f"{_AXIS[axis]} [mm]")
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("t − t₀ (s), MCU master")
        handles, labels = axes[0].get_legend_handles_labels()
        if legend_phases:
            handles = handles + legend_phases
            labels = labels + [p.get_label() for p in legend_phases]
        if handles:
            axes[0].legend(handles, labels, loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
        fig.suptitle(title)
        return fig

    opt_colors = {tr.name: _OPT_COLORS.get(tr.name.replace("optical_", ""), "#333") for tr in viz.optical}
    imu_colors = {}
    for tr in viz.imu:
        if "MTA" in tr.name:
            imu_colors[tr.name] = _IMU_COLORS["imu_mta"]
        elif "CRA" in tr.name:
            imu_colors[tr.name] = _IMU_COLORS["imu_cra"]
        else:
            imu_colors[tr.name] = "#1f618d"

    figs: list[tuple[str, Any]] = []
    if viz.optical:
        figs.append(
            (
                "xyz_optical",
                _xyz_figure(viz.optical, f"Optical XYZ — {viz.path.name}", opt_colors),
            )
        )
    if viz.imu:
        imu_note = "IMU XYZ (ZUPT estimate, not millimetre truth)"
        if viz.imu[0].source == "derived":
            imu_note = "IMU XYZ from /derived/pose"
        figs.append(
            ("xyz_imu", _xyz_figure(viz.imu, f"{imu_note} — {viz.path.name}", imu_colors))
        )

    if viz.optical or viz.imu:
        ensure_jts_on_path()
        from jts.helper import plot_trajectories

        trajs = [tr for tr in (*viz.optical, *viz.imu) if _finite_pose_mask(tr.T).any()]
        if trajs:
            fig3d, _ax = plot_trajectories(
                [tr.T[_finite_pose_mask(tr.T)] for tr in trajs],
                title=f"Optical + IMU trajectories — {viz.path.name}",
                labels=[tr.name for tr in trajs],
                plot_3d=True,
                plot_rot=False,
                figure_size=[10.0, 8.0],
                axes_label_fontsize=12.0,
                axes_tick_fontsize=10.0,
                title_fontsize=14.0,
            )
            figs.append(("trajectories_3d", fig3d))

    if viz.emg is not None:
        n_ch = viz.emg.samples_uv.shape[1]
        fig_e, axes_e = plt.subplots(
            n_ch, 1, sharex=True, figsize=(12, max(8.0, 1.15 * n_ch)), constrained_layout=True
        )
        if n_ch == 1:
            axes_e = [axes_e]
        t = _t_s(viz.emg.t_us, viz.t0_us)
        for i, ax in enumerate(axes_e):
            _shade_phases(ax, spans, label=(i == 0))
            _mark_notes(ax, notes_s, label=(i == 0))
            td, yd = _emg_downsampled(t, viz.emg.samples_uv[:, i], viz.emg.fs_hz, 80_000)
            ax.plot(td, yd, color=_EMG_COLORS[i % len(_EMG_COLORS)], lw=0.6, zorder=2)
            ylim = _settled_ylim(td, (yd,), _FILTER_SETTLE_S)  # skip filter start-up
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_ylabel(viz.emg.channel_names[i], fontsize=8)
            ax.grid(True, alpha=0.3)
        axes_e[-1].set_xlabel("t − t₀ (s), MCU master")
        fig_e.suptitle(f"EMG (HPF 20 Hz, mains removed) [µV] — {viz.path.name}")
        if legend_phases:
            axes_e[0].legend(handles=legend_phases, loc="upper right", fontsize=7, ncol=4)
        figs.append(("emg", fig_e))

    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        stem = viz.path.stem
        for tag, fig in figs:
            dest = save / f"{stem}_{tag}.png"
            fig.savefig(dest, dpi=140)
            written.append(dest)
            print(f"Wrote {dest}")

    if show:
        plt.show()
    else:
        for _tag, fig in figs:
            plt.close(fig)
    return written


def _stride(n: int, max_points: int) -> int:
    if n <= max_points or max_points < 2:
        return 1
    return max(1, int(np.ceil(n / max_points)))


def _add_phase_vrects(
    fig: Any,
    *,
    row: int,
    spans: list[tuple[str, float, float]],
) -> None:
    """Shade experiment phases on a 2D subplot.

    ``exclude_empty_subplots=False`` is required: Plotly's default empty-subplot
    probe reads ``trace['xaxis']``, which Scatter3d traces do not have.
    """
    for i, (_name, t0, t1) in enumerate(spans):
        fig.add_vrect(
            x0=t0,
            x1=t1,
            fillcolor=_PHASE_COLORS[i % len(_PHASE_COLORS)],
            opacity=0.25,
            line_width=0,
            row=row,
            col=1,
            exclude_empty_subplots=False,
        )


def _add_note_vlines(fig: Any, *, row: int, notes_s: list[tuple[float, str]], label: bool) -> None:
    """Dotted vertical line per viewer note (text on the first row only)."""
    for t, text in notes_s:
        ann = (dict(annotation_text=text[:40], annotation_textangle=-90,
                    annotation_font_size=9) if label else {})
        fig.add_vline(x=t, line_dash="dot", line_width=1, line_color="#555", row=row, col=1,
                      exclude_empty_subplots=False, **ann)


def _write_plotly_html(path: Path, figures: list[Any], *, title: str) -> None:
    chunks: list[str] = []
    for i, fig in enumerate(figures):
        chunks.append(
            fig.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{title}</title></head><body>\n"
        + "\n".join(chunks)
        + "\n</body></html>\n",
        encoding="utf-8",
    )


def plot_plotly_html(
    viz: SessionViz,
    sidecar_h5: Path | None,
    html_path: Path,
    *,
    show: bool,
    show_phases: bool,
) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ensure_jts_on_path()
    from jts.plotly_visualization import JawMotionVisualizer

    n_emg = viz.emg.samples_uv.shape[1] if viz.emg is not None else 0
    has_traj = bool(viz.optical or viz.imu)
    figures: list[Any] = []

    # Keep the 3D scene in its own figure. Mixing Scatter3d into a 2D subplot
    # grid makes Plotly's add_vrect inspect 3D traces for a 2D ``xaxis``.
    if has_traj and sidecar_h5 is not None and sidecar_h5.is_file():
        jts_viz = JawMotionVisualizer(sidecar_h5)
        fig3d = jts_viz.plot_trajectory_3d(
            show_frames=False,
            title=f"Optical + IMU trajectories — {viz.path.name}",
        )
        for trace in fig3d.data:
            xs = getattr(trace, "x", None)
            if xs is None:
                continue
            step = _stride(len(xs), 8_000)
            if step > 1 and hasattr(trace, "x"):
                trace.x = xs[::step]
                if getattr(trace, "y", None) is not None:
                    trace.y = trace.y[::step]
                if getattr(trace, "z", None) is not None:
                    trace.z = trace.z[::step]
        figures.append(fig3d)

    specs: list[list[dict[str, Any]]] = []
    titles: list[str] = []
    if has_traj:
        specs.extend([[{"type": "xy"}]] * 3)
        titles.extend(["Translation X [mm]", "Translation Y [mm]", "Translation Z [mm]"])
    if n_emg:
        specs.extend([[{"type": "xy"}]] * n_emg)
        titles.extend(viz.emg.channel_names if viz.emg is not None else [])
    if not specs:
        if not figures:
            raise ValueError("nothing to plot")
        _write_plotly_html(html_path, figures, title=viz.path.name)
        print(f"Wrote {html_path}")
        return html_path

    fig = make_subplots(
        rows=len(specs),
        cols=1,
        specs=specs,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=titles,
    )

    spans = _phase_spans_s(viz.phases, viz.t0_us) if show_phases else []
    notes_s = [((t - viz.t0_us) * 1e-6, text) for t, text in viz.notes]
    xyz_row0 = 1 if has_traj else None

    def _add_xyz(trajs: list[NamedTrajectory], dash: str) -> None:
        if xyz_row0 is None:
            return
        for tr in trajs:
            step = _stride(tr.t_us.size, 80_000)
            sub = NamedTrajectory(tr.name, tr.T[::step], tr.t_us[::step],
                                  tr.sample_rate_hz / step, tr.unit, tr.source)
            t_m, xyz_m = _gapped_xyz(sub, viz.t0_us)
            for axis in range(3):
                fig.add_trace(
                    go.Scattergl(
                        x=t_m,
                        y=xyz_m[:, axis],
                        mode="lines",
                        name=tr.name if axis == 0 else None,
                        legendgroup=tr.name,
                        showlegend=(axis == 0),
                        line=dict(width=1.4, dash=dash),
                        connectgaps=False,
                    ),
                    row=xyz_row0 + axis,
                    col=1,
                )

    if has_traj:
        _add_xyz(viz.optical, "solid")
        _add_xyz(viz.imu, "dash")
        for axis in range(3):
            row = xyz_row0 + axis  # type: ignore[operator]
            _add_phase_vrects(fig, row=row, spans=spans)
            _add_note_vlines(fig, row=row, notes_s=notes_s, label=(axis == 0))
            fig.update_yaxes(title_text=f"{_AXIS[axis]} [mm]", row=row, col=1)

    if viz.emg is not None:
        t = _t_s(viz.emg.t_us, viz.t0_us)
        emg_row0 = (4 if has_traj else 1)
        for i in range(n_emg):
            td, yd = _emg_downsampled(t, viz.emg.samples_uv[:, i], viz.emg.fs_hz, 80_000)
            fig.add_trace(
                go.Scattergl(
                    x=td,
                    y=yd,
                    mode="lines",
                    name=viz.emg.channel_names[i],
                    line=dict(width=0.8, color=_EMG_COLORS[i % len(_EMG_COLORS)]),
                    showlegend=True,
                ),
                row=emg_row0 + i,
                col=1,
            )
            fig.update_yaxes(title_text="µV", row=emg_row0 + i, col=1)
            _add_phase_vrects(fig, row=emg_row0 + i, spans=spans)
            _add_note_vlines(fig, row=emg_row0 + i, notes_s=notes_s,
                             label=(i == 0 and not has_traj))

    fig.update_xaxes(title_text="t − t₀ (s), MCU master", row=len(specs), col=1)
    fig.update_layout(
        title=f"XYZ + EMG — {viz.path.name}",
        height=max(700, 180 * len(specs)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    figures.append(fig)

    _write_plotly_html(html_path, figures, title=f"Session visualization — {viz.path.name}")
    print(f"Wrote {html_path}")
    if show:
        try:
            import webbrowser

            webbrowser.open(html_path.resolve().as_uri())
        except Exception as exc:  # noqa: BLE001
            print(f"warning: could not open browser ({exc}); open {html_path}", file=sys.stderr)
    return html_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="session.h5 or a session directory")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: next to the session file)",
    )
    ap.add_argument("--html", type=Path, default=None, help="HTML path (implies Plotly output)")
    ap.add_argument("--png", action="store_true", help="Also write matplotlib PNGs into --out")
    ap.add_argument("--no-html", action="store_true", help="Skip Plotly HTML (matplotlib only)")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive windows")
    ap.add_argument("--no-phases", action="store_true", help="Do not shade experiment phases")
    ap.add_argument(
        "--keep-sidecar",
        action="store_true",
        help="Keep the JTS translations/rotations HDF5 next to the HTML",
    )
    ap.add_argument(
        "--imu-reset-s",
        type=float,
        default=1e9,
        help="ZUPT hard-reset interval in seconds (default: disabled)",
    )
    args = ap.parse_args(argv)

    try:
        session_h5 = resolve_session_h5(args.session)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        ensure_jts_on_path()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    viz = load_session_viz(session_h5, imu_auto_reset_s=args.imu_reset_s)
    if not viz.optical and not viz.imu and viz.emg is None:
        print(f"error: no optical, IMU, or EMG data in {session_h5}", file=sys.stderr)
        return 1

    print(f"{session_h5}: optical={len(viz.optical)}  imu={len(viz.imu)}  "
          f"emg_ch={0 if viz.emg is None else viz.emg.samples_uv.shape[1]}")
    if viz.phases:
        print(f"  phases: {len(viz.phases)} ({', '.join(p.name for p in viz.phases)})")
    if viz.notes:
        print(f"  viewer notes: {len(viz.notes)} (dotted lines)")
    if viz.imu and viz.imu[0].source == "imu_zupt":
        print("  note: IMU XYZ is a drift-prone ZUPT estimate, not optical ground truth")

    out_dir = args.out if args.out is not None else session_h5.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    sidecar = out_dir / f"{session_h5.stem}_jts_trajectories.h5"
    sidecar_written: Path | None = None
    tmp_ctx = None
    if viz.optical or viz.imu:
        need_sidecar = (not args.no_html) or args.keep_sidecar
        if need_sidecar and args.keep_sidecar:
            sidecar_written = write_jts_sidecar(viz, sidecar)
            print(f"Wrote JTS sidecar {sidecar_written}")
        elif need_sidecar:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="jts_viz_")
            sidecar_written = write_jts_sidecar(viz, Path(tmp_ctx.name) / sidecar.name)

    html_path = args.html or (out_dir / f"{session_h5.stem}_viz.html")
    show = not args.no_show

    try:
        if not args.no_html:
            try:
                plot_plotly_html(
                    viz,
                    sidecar_written,
                    html_path,
                    show=show,
                    show_phases=not args.no_phases,
                )
                show = False  # matplotlib should not also pop windows
            except ImportError:
                print("warning: plotly not installed; falling back to matplotlib", file=sys.stderr)
                args.png = True
        if args.png or args.no_html:
            plot_matplotlib(
                viz,
                save=out_dir,
                show=show,
                show_phases=not args.no_phases,
            )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
