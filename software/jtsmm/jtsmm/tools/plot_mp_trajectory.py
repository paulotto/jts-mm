#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Plot the smoothed mandibular landmark in the dental model frame.

Falls back to the MP origin in the HP frame when the session has no JTS config.

Supports:

- multimodal ``session.h5`` from ``jtsmm record --optical-track``
  (``/optical/pose_mta``, ``/optical/t_master_us``)
- standalone optical tracking HDF5 from ``optical track``
  (``bodies/MP/position_mm`` or ``pose_4x4``)

When the session has ``/phases`` (or ``/events`` phase_start/end) from
``jtsmm record --experiment-plan``, the 2D plot and the 3D figures start
at the first ``rest`` phase (landmark calibration / MVC before rest are
omitted). Peak-to-peak cycles on the opening magnitude (same rule as
§5.4.2) are marked as dotted vertical lines on the 2D XYZ plot. Extra
When a ``*_jts_config.json`` sits next to the session, every figure uses
``T_model_origin_mand_landmark_t_smooth`` (metres, Z up on the 3D views),
the same series as ``jaw_motion_3d.png``. The 2D stack is that landmark
versus master time, with phase bands. The combined 3D matches
``jaw_motion_3d.png``. A second 3D drops the last phase. One 3D axes per
phase shows that phase alone. The landmark figure sweeps ``mand_point_1``
and both TMJ points with the smoothed model-frame pose.

Examples::

    python -m jtsmm.tools.plot_mp_trajectory path/to/session.h5
    python -m jtsmm.tools.plot_mp_trajectory tracking.h5 --save mp_xyz.png
    python -m jtsmm.tools.plot_mp_trajectory session.h5 --out viz/ --no-show
    python -m jtsmm.tools.plot_mp_trajectory session.h5 --body HP
    python -m jtsmm.tools.plot_mp_trajectory session.h5 --no-phases
    python -m jtsmm.tools.plot_mp_trajectory session.h5 --no-cycles
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from jtsmm.tools.timegrid import break_gaps, session_frame_period_us


# Logical body → multimodal session dataset under /optical/
_SESSION_BODY = {
    "MP": "pose_mta",
    "MTA": "pose_mta",
    "HP": "pose_cra",
    "CRA": "pose_cra",
    "CT": "pose_ct",
}

# Distinct soft fills for phase spans (cycled).
_PHASE_COLORS = (
    "#f5cba7",
    "#a9cce3",
    "#abebc6",
    "#f5b7b1",
    "#d7bde2",
    "#f9e79f",
    "#a3e4d7",
    "#d5d8dc",
)


@dataclass(frozen=True)
class PhaseInterval:
    name: str
    kind: str
    t_start_us: int
    t_end_us: int


@dataclass
class Trajectory:
    t_s: np.ndarray
    xyz_mm: np.ndarray
    time_label: str
    t0_us: float | None  # absolute master µs of t_s==0, if known
    phases: list[PhaseInterval]
    period_s: float | None = None  # frame period; lines break where dt > 1.5 periods
    frame: str = "world"  # "hp" HP marker, "model" dental model CS
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)
    unit: str = "mm"


def _translation_mm(poses: np.ndarray) -> np.ndarray:
    """Extract (N, 3) translation from (N, 4, 4) SE(3); units are mm in this pipeline."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4) poses, got {poses.shape}")
    return poses[:, :3, 3].copy()


def _decode_str(v: object) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def load_phases(f: h5py.File) -> list[PhaseInterval]:
    """Load phase intervals from ``/phases``, ``/events``, or sync report JSON."""
    out: list[PhaseInterval] = []

    if "phases" in f and "t_start_us" in f["phases"]:
        g = f["phases"]
        names = [_decode_str(x) for x in g["name"][()]]
        kinds = (
            [_decode_str(x) for x in g["kind"][()]]
            if "kind" in g
            else ["experiment"] * len(names)
        )
        t0 = np.asarray(g["t_start_us"][()], dtype=np.int64)
        t1 = np.asarray(g["t_end_us"][()], dtype=np.int64)
        for name, kind, a, b in zip(names, kinds, t0, t1):
            out.append(PhaseInterval(name=name, kind=kind, t_start_us=int(a), t_end_us=int(b)))
        return out

    if "events" in f and "t_master_us" in f["events"] and "kind" in f["events"]:
        ev = f["events"]
        kinds = [_decode_str(x) for x in ev["kind"][()]]
        texts = [_decode_str(x) for x in ev["text"][()]] if "text" in ev else [""] * len(kinds)
        phase_kinds = (
            [_decode_str(x) for x in ev["phase_kind"][()]]
            if "phase_kind" in ev
            else ["experiment"] * len(kinds)
        )
        times = np.asarray(ev["t_master_us"][()], dtype=np.int64)
        open_start: dict[str, tuple[int, str]] = {}
        for kind, text, pk, t_us in zip(kinds, texts, phase_kinds, times):
            if kind in ("phase_start", "start_block"):
                open_start[text] = (int(t_us), pk)
            elif kind in ("phase_end", "end_block") and text in open_start:
                t0, pk0 = open_start.pop(text)
                out.append(
                    PhaseInterval(
                        name=text,
                        kind=pk0 or pk,
                        t_start_us=t0,
                        t_end_us=int(t_us),
                    )
                )
        if out:
            return out

    if "sync" in f and "report_json" in f["sync"].attrs:
        raw = f["sync"].attrs["report_json"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            report = json.loads(raw)
        except json.JSONDecodeError:
            return out
        for p in report.get("phases") or []:
            out.append(
                PhaseInterval(
                    name=str(p.get("name", "")),
                    kind=str(p.get("kind", "experiment")),
                    t_start_us=int(p["t_start_us"]),
                    t_end_us=int(p["t_end_us"]),
                )
            )
    return out


def relative_origin_mm(pose_body: np.ndarray, pose_ref: np.ndarray) -> np.ndarray:
    """Origin of ``pose_body`` expressed in the ``pose_ref`` frame, millimetres."""
    n = min(len(pose_body), len(pose_ref))
    body = np.asarray(pose_body[:n], dtype=np.float64)
    ref = np.asarray(pose_ref[:n], dtype=np.float64)
    delta = body[:, :3, 3] - ref[:, :3, 3]
    out = np.einsum("nji,nj->ni", ref[:, :3, :3], delta)
    ok = (
        np.isfinite(body[:, :3]).all(axis=(1, 2))
        & np.isfinite(ref[:, :3]).all(axis=(1, 2))
        & np.isfinite(delta).all(axis=1)
    )
    out[~ok] = np.nan
    return out


def sweep_body_point(
    pose_body: np.ndarray,
    pose_ref: np.ndarray,
    point_body_mm: np.ndarray,
) -> np.ndarray:
    """Rigid point of ``pose_body``, expressed in the ``pose_ref`` frame."""
    n = min(len(pose_body), len(pose_ref))
    body = np.asarray(pose_body[:n], dtype=np.float64)
    ref = np.asarray(pose_ref[:n], dtype=np.float64)
    point = np.asarray(point_body_mm, dtype=np.float64).reshape(3)
    world = np.einsum("nij,j->ni", body[:, :3, :3], point) + body[:, :3, 3]
    delta = world - ref[:, :3, 3]
    out = np.einsum("nji,nj->ni", ref[:, :3, :3], delta)
    ok = np.isfinite(body[:, :3]).all(axis=(1, 2)) & np.isfinite(ref[:, :3]).all(axis=(1, 2))
    out[~ok] = np.nan
    return out


def _point_in_body(
    pose_body: np.ndarray,
    pose_tip: np.ndarray,
    mask: np.ndarray,
    *,
    min_samples: int = 5,
) -> np.ndarray | None:
    """Median tip position in the body frame over ``mask``."""
    n = min(len(pose_body), len(pose_tip), len(mask))
    keep = np.asarray(mask[:n], dtype=bool)
    if int(keep.sum()) < min_samples:
        return None
    body = np.asarray(pose_body[:n][keep], dtype=np.float64)
    tip = np.asarray(pose_tip[:n][keep, :3, 3], dtype=np.float64)
    ok = np.isfinite(body[:, :3]).all(axis=(1, 2)) & np.isfinite(tip).all(axis=1)
    if int(ok.sum()) < min_samples:
        return None
    local = np.einsum("nji,nj->ni", body[ok, :3, :3], tip[ok] - body[ok, :3, 3])
    return np.median(local, axis=0)


# Lower incisal point in red; both condyle digitizations in green.
_LANDMARK_NAMES = ("mand_point_1", "tmj_left", "tmj_right")
_LANDMARK_COLORS = {
    "mand_point_1": "#c0392b",
    "tmj_left": "#196f3d",
    "tmj_right": "#1e8449",
}


def digitize_landmarks(
    pose_mp: np.ndarray,
    pose_hp: np.ndarray,
    pose_ct: np.ndarray,
    t_us: np.ndarray | None,
    phases: list[PhaseInterval],
) -> dict[str, np.ndarray]:
    """Incisal point and both TMJ points swept by the mandible, in the HP frame.

    Each name is the median CT tip during its calibration phase, stored in the
    MP frame, then moved with every MP pose.
    """
    n = min(len(pose_mp), len(pose_hp), len(pose_ct))
    if t_us is None or n == 0:
        return {}
    t = np.asarray(t_us[:n], dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    for phase in phases:
        key = phase.name.strip().lower()
        if key not in _LANDMARK_NAMES or key in out:
            continue
        mask = (t >= float(phase.t_start_us)) & (t <= float(phase.t_end_us))
        point = _point_in_body(pose_mp[:n], pose_ct[:n], mask)
        if point is None:
            continue
        out[key] = sweep_body_point(pose_mp[:n], pose_hp[:n], point)
    return out


def _pose(group: h5py.Group, name: str) -> np.ndarray | None:
    if name not in group:
        return None
    return np.asarray(group[name][()], dtype=np.float64)


def load_trajectory(path: Path, *, body: str = "MP") -> Trajectory:
    """Load translation, relative time axis, and phase intervals.

    MP/MTA is the mandible origin in the HP (cranial marker) frame when both
    poses are stored. Other bodies stay in the camera world frame.
    """
    body_key = body.strip().upper()
    with h5py.File(path, "r") as f:
        phases = load_phases(f)

        # Multimodal jtsmm session
        if "optical" in f and any(k.startswith("pose_") for k in f["optical"]):
            ds_name = _SESSION_BODY.get(body_key)
            if ds_name is None or ds_name not in f["optical"]:
                available = sorted(k for k in f["optical"] if k.startswith("pose_"))
                raise KeyError(
                    f"body {body!r} not in session /optical/ "
                    f"(need one of {list(_SESSION_BODY)}; available={available})"
                )
            opt = f["optical"]
            pose_mp = _pose(opt, "pose_mta")
            pose_hp = _pose(opt, "pose_cra")
            pose_ct = _pose(opt, "pose_ct")
            frame = "world"
            if body_key in ("MP", "MTA") and pose_mp is not None and pose_hp is not None:
                xyz = relative_origin_mm(pose_mp, pose_hp)
                frame = "hp"
            else:
                xyz = _translation_mm(opt[ds_name][()])
            t_us = (
                np.asarray(opt["t_master_us"][()], dtype=np.float64)
                if "t_master_us" in opt
                else None
            )
            landmarks = {}
            if pose_mp is not None and pose_hp is not None and pose_ct is not None:
                landmarks = digitize_landmarks(pose_mp, pose_hp, pose_ct, t_us, phases)
            n = int(xyz.shape[0])
            landmarks = {k: v for k, v in landmarks.items() if len(v) == n}
            if t_us is not None:
                t_us = t_us[:n]
                t0_us = float(t_us[0]) if t_us.size else None
                t_s = (t_us - t_us[0]) * 1e-6 if t_us.size else np.asarray([], dtype=np.float64)
                p_us = session_frame_period_us(f)
                return Trajectory(
                    t_s, xyz, "t − t₀ (s), MCU master", t0_us, phases,
                    period_s=p_us * 1e-6 if p_us else None,
                    frame=frame, landmarks=landmarks,
                )
            t_s = np.arange(n, dtype=np.float64)
            return Trajectory(t_s, xyz, "frame index", None, phases, frame=frame, landmarks=landmarks)

        # Standalone optical tracking session
        body_grp = f.get(f"bodies/{body_key}")
        if body_grp is None and body_key in ("MTA", "CRA"):
            alt = "MP" if body_key == "MTA" else "HP"
            body_grp = f.get(f"bodies/{alt}")
            body_key = alt
        if body_grp is None:
            raise KeyError(
                f"neither /optical/pose_* nor bodies/{body_key}/ found in {path}"
            )
        if "position_mm" in body_grp:
            xyz = np.asarray(body_grp["position_mm"][()], dtype=np.float64)
        else:
            xyz = _translation_mm(body_grp["pose_4x4"][()])
        if "timestamps" in f and "t_master_us" in f["timestamps"]:
            t_us = np.asarray(f["timestamps"]["t_master_us"][()], dtype=np.float64)
            if np.isfinite(t_us).any():
                t0 = float(t_us[np.isfinite(t_us)][0])
                t_s = np.where(np.isfinite(t_us), (t_us - t0) * 1e-6, np.nan)
                return Trajectory(t_s, xyz, "t − t₀ (s), MCU master", t0, phases)
        if "timestamps" in f and "frame_idx" in f["timestamps"]:
            t_s = np.asarray(f["timestamps"]["frame_idx"][()], dtype=np.float64)
            return Trajectory(t_s, xyz, "frame index", None, phases)
        t_s = np.arange(xyz.shape[0], dtype=np.float64)
        return Trajectory(t_s, xyz, "frame index", None, phases)


_CYCLE_MOTIONS = ("open_close", "protrusion_retrusion", "left_right", "chewing")
_CYCLE_LINE = dict(color="#1a1a1a", ls=":", lw=0.7, alpha=0.7, zorder=3)


def find_cycles(
    y: np.ndarray,
    fs_hz: float,
    min_period_s: float = 0.4,
    *,
    t_s: np.ndarray | None = None,
    prominence_frac: float = 0.25,
    fill_frac: float = 0.12,
) -> list[tuple[int, int]]:
    """Peak-to-peak cycles on a 1-D amplitude trace (opening maxima).

    A second pass inserts peaks into unusually long inter-peak gaps (typical
    skipped, lower-amplitude repetition) using a lower prominence.
    """
    from scipy.signal import find_peaks

    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 8 or fs_hz <= 0:
        return []
    if t_s is None:
        tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    else:
        tt = np.asarray(t_s, dtype=np.float64).ravel()
        if tt.size != y.size:
            tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    distance = max(int(min_period_s * fs_hz), 2)
    y_std = float(np.nanstd(y))
    prominence = max(y_std * float(prominence_frac), 1e-6)
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    peaks = [int(p) for p in peaks]
    if len(peaks) >= 3:
        med = float(np.median(np.diff(tt[peaks])))
        max_gap = max(2.2 * med, 2.0 * min_period_s)
    else:
        max_gap = max(2.5, 2.0 * min_period_s)
    extra: list[int] = []
    fill_prom = max(y_std * float(fill_frac), 1e-6)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if tt[b] - tt[a] <= max_gap:
            continue
        sl = slice(a + 1, b)
        if int(np.isfinite(y[sl]).sum()) < 8:
            continue
        loc, props = find_peaks(y[sl], distance=distance, prominence=fill_prom)
        if loc.size == 0:
            continue
        cand: list[tuple[float, int]] = []
        prominences = np.asarray(props.get("prominences", []), dtype=float)
        mid = 0.5 * (tt[a] + tt[b])
        for j, rel in enumerate(loc):
            k = int(a + 1 + rel)
            if tt[k] - tt[a] < min_period_s or tt[b] - tt[k] < min_period_s:
                continue
            pr = float(prominences[j]) if j < prominences.size else 0.0
            cand.append((abs(tt[k] - mid), -pr, k))
        if cand:
            extra.append(min(cand)[2])
    if extra:
        peaks = sorted(set(peaks).union(extra))
    if len(peaks) < 2:
        return []
    out: list[tuple[int, int]] = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if b - a >= distance:
            out.append((int(a), int(b)))
    return out


def _sample_rate_hz(traj: Trajectory, idx: np.ndarray) -> float:
    if traj.period_s and traj.period_s > 0:
        return 1.0 / float(traj.period_s)
    if idx.size < 2:
        return 60.0
    dt = np.diff(traj.t_s[idx])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 60.0
    return 1.0 / float(np.median(dt))


def cycle_boundary_times_s(traj: Trajectory) -> list[float]:
    """Relative seconds of opening peaks that bound peak-to-peak cycles."""
    rest = next((p for p in traj.phases if p.name.strip().lower() == "rest"), None)
    origin = None
    if rest is not None:
        rxyz = phase_xyz(traj, rest)
        if rxyz.size:
            origin = np.nanmean(rxyz, axis=0)
    if origin is None:
        finite = traj.xyz_mm[np.isfinite(traj.xyz_mm).all(axis=1)]
        if finite.size == 0:
            return []
        origin = np.nanmean(finite, axis=0)
    amp = np.linalg.norm(traj.xyz_mm - origin, axis=1)
    amp = np.where(np.isfinite(traj.xyz_mm).all(axis=1), amp, np.nan)
    times: list[float] = []
    for phase in traj.phases:
        if phase.name not in _CYCLE_MOTIONS:
            continue
        mask = phase_mask(traj, phase)
        idx = np.flatnonzero(mask)
        if idx.size < 8:
            continue
        segs = find_cycles(amp[idx], _sample_rate_hz(traj, idx), t_s=traj.t_s[idx])
        if not segs:
            continue
        peaks = [segs[0][0], *[b for _a, b in segs]]
        for k in peaks:
            times.append(float(traj.t_s[idx[int(k)]]))
    return times


_MAX_3D_POINTS = 25_000
_COMBINED_3D_COLOR = "#1a5276"


def is_calibration_phase(phase: PhaseInterval) -> bool:
    """Landmark digitizing / other calibration blocks — not jaw-motion takes."""
    blob = f"{phase.kind} {phase.name}".lower()
    if "calibration" in blob:
        return True
    name = phase.name.lower()
    return name.startswith(("mand_point", "max_point"))


def motion_phases(phases: list[PhaseInterval]) -> list[PhaseInterval]:
    """Experiment / MVC / rest phases, excluding calibration."""
    return [p for p in phases if not is_calibration_phase(p)]


def from_rest_phases(phases: list[PhaseInterval]) -> list[PhaseInterval]:
    """Motion phases from the first ``rest`` onward (inclusive).

    If the session has no rest phase, all motion phases are kept.
    """
    motion = motion_phases(phases)
    for i, p in enumerate(motion):
        if p.name.strip().lower() == "rest":
            return motion[i:]
    return motion


def from_rest_phases_drop_last(phases: list[PhaseInterval]) -> list[PhaseInterval]:
    """``from_rest_phases`` without the last interval (e.g. drop ``free_motion``)."""
    rest_onward = from_rest_phases(phases)
    if len(rest_onward) <= 1:
        return []
    return rest_onward[:-1]


def phase_mask(
    traj: Trajectory,
    phase: PhaseInterval,
    xyz: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask of finite samples whose master time falls inside ``phase``."""
    pts = traj.xyz_mm if xyz is None else xyz
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(traj.t_s)
    if traj.t0_us is None:
        return np.zeros(traj.t_s.shape[0], dtype=bool)
    t_us = traj.t0_us + traj.t_s * 1e6
    return finite & (t_us >= float(phase.t_start_us)) & (t_us <= float(phase.t_end_us))


def phase_xyz(traj: Trajectory, phase: PhaseInterval) -> np.ndarray:
    """``(N, 3)`` plotted-frame millimetres inside ``phase`` (possibly empty)."""
    return traj.xyz_mm[phase_mask(traj, phase)]


def rest_offset_mm(traj: Trajectory) -> np.ndarray:
    """Median pose of the rest phase, or of all finite samples when rest is absent."""
    rest = next((p for p in traj.phases if p.name.strip().lower() == "rest"), None)
    pts = phase_xyz(traj, rest) if rest is not None else np.empty((0, 3))
    if pts.size == 0:
        finite = np.isfinite(traj.xyz_mm).all(axis=1)
        pts = traj.xyz_mm[finite]
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.nanmedian(pts, axis=0)


def _gapped(traj: Trajectory, mask: np.ndarray, xyz: np.ndarray | None = None) -> np.ndarray:
    """Finite samples under ``mask`` with a NaN row at every time gap.

    Lets a line plot break across dropped frames / lost tracking instead of
    drawing a straight segment through them.
    """
    pts = traj.xyz_mm if xyz is None else xyz
    _t, out = break_gaps(traj.t_s[mask], pts[mask], period=traj.period_s)
    return out


def _thin_xyz(xyz: np.ndarray, max_points: int = _MAX_3D_POINTS) -> np.ndarray:
    n = int(xyz.shape[0])
    if n <= max_points:
        return xyz
    idx = np.linspace(0, n - 1, max_points).astype(int)
    idx[0] = 0
    idx[-1] = n - 1
    return xyz[idx]


def _set_equal_3d(ax, xyz: np.ndarray) -> None:
    finite = xyz[np.isfinite(xyz).reshape(len(xyz), -1).all(axis=1)]
    if finite.size == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    span = float(np.max(maxs - mins))
    if not np.isfinite(span) or span < 1e-6:
        span = 1.0
    span = span * 1.1
    mid = 0.5 * (mins + maxs)
    ax.set_xlim(mid[0] - span / 2.0, mid[0] + span / 2.0)
    ax.set_ylim(mid[1] - span / 2.0, mid[1] + span / 2.0)
    ax.set_zlim(mid[2] - span / 2.0, mid[2] + span / 2.0)
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except AttributeError:
        pass


def _phase_color(index: int) -> str:
    return _PHASE_COLORS[index % len(_PHASE_COLORS)]


def _line_3d(ax, xyz: np.ndarray, *, color: str, label: str | None, lw: float = 1.1) -> None:
    pts = _thin_xyz(xyz)
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw, label=label)


# Y up; yaw so +Z is front-left and +X is a bit more front-right (not a head-on view).
_3D_ELEV = 25
_3D_AZIM = 35


def _axis_labels(frame: str, unit: str = "mm") -> tuple[str, str, str]:
    if frame == "hp_rest":
        return ("X (mm from rest)", "Y (mm from rest)", "Z (mm from rest)")
    if frame == "hp":
        return ("X (mm, HP)", "Y (mm, HP)", "Z (mm, HP)")
    if frame == "model":
        return (f"X [{unit}]", f"Y [{unit}]", f"Z [{unit}]")
    return ("X (mm)", "Y (mm)", "Z (mm)")


def _style_3d_axes(ax, *, xlabel_size: float | None = None, frame: str = "world") -> None:
    """Millimetres in the plotted frame. Y is drawn upward."""
    kw = {} if xlabel_size is None else {"fontsize": xlabel_size}
    labels = _axis_labels(frame)
    ax.set_xlabel(labels[0], **kw)
    ax.set_ylabel(labels[1], **kw)
    ax.set_zlabel(labels[2], **kw)
    ax.view_init(elev=_3D_ELEV, azim=_3D_AZIM, roll=0, vertical_axis="y")


@dataclass
class ModelTrajectory:
    """Smoothed mandibular landmark in the dental model frame."""

    xyz: np.ndarray
    unit: str
    scale: float
    t_us: np.ndarray
    hz: float
    T_smooth: np.ndarray
    points: dict[str, np.ndarray]


def jts_model_trajectory(session: Path) -> ModelTrajectory | None:
    """Smoothed landmark pose, as exported for ``jaw_motion_3d``.

    ``xyz`` and ``points`` are in the config output unit (metres when
    ``scale_factor`` is 0.001). ``points`` holds ``mand_point_1`` and any
    digitized TMJ points, moved by the smoothed pose. ``None`` when the
    session has no JTS config.
    """
    from jtsmm.tools.jts_path import ensure_jts_on_path
    from jtsmm.tools.run_jts_analysis import (
        DEFAULT_MAX_GAP_S,
        find_jts_config,
        interpolate_se3_gaps,
        prepare_config,
        resolve_session_h5,
    )

    try:
        session_h5 = resolve_session_h5(session)
        config_path = find_jts_config(session_h5, None)
    except FileNotFoundError:
        return None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = prepare_config(
        config,
        session_h5=session_h5,
        out_dir=session_h5.parent,
        append_session=False,
    )
    out = cfg.setdefault("output", {})
    out["save_hdf5"] = False
    out["save_csv"] = False
    out["save_plots"] = False
    ensure_jts_on_path()
    from jts.core import JawMotionAnalysis

    analysis = JawMotionAnalysis(cfg)
    analysis.load_data()
    analysis.perform_calibration()
    analysis.compute_relative_motion()
    analysis.transform_to_ref_markers_coordinates()
    analysis.register_to_model_coordinates()
    raw = analysis.trajectories.get("T_model_origin_mand_landmark_t")
    if raw is None:
        return None
    smooth_cfg = (cfg.get("analysis") or {}).get("smoothing") or {}
    max_gap_s = float(smooth_cfg.get("max_gap_s", DEFAULT_MAX_GAP_S))
    hz = float(getattr(analysis.motion_data, "frame_rate", 0.0) or 0.0)
    cap = int(round(max_gap_s * hz)) if hz > 0 else 0
    filled, _n_fill = interpolate_se3_gaps(raw, max_gap_frames=cap)
    analysis.trajectories["T_model_origin_mand_landmark_t"] = filled
    analysis.smooth_trajectory()
    smooth = analysis.trajectories.get("T_model_origin_mand_landmark_t_smooth")
    if smooth is None:
        return None
    smooth = np.asarray(smooth, dtype=np.float64)
    scale = float(out.get("scale_factor", 0.001))
    unit = str(out.get("unit", "m"))
    t_us = analysis.frame_times_us()
    if t_us is None or len(t_us) != len(smooth):
        t_us = np.full(len(smooth), np.nan)
    else:
        t_us = np.asarray(t_us, dtype=np.float64)
    mand = ((cfg.get("analysis") or {}).get("calibration") or {}).get("mandibular") or {}
    bodies = list(mand.get("rigid_bodies") or ["MP"])
    T_marker = analysis.calibration_transforms.get(f"T_{bodies[0]}_landmark")
    points = _model_points(session_h5, smooth, T_marker, scale)
    return ModelTrajectory(
        xyz=np.asarray(smooth[:, :3, 3], dtype=np.float64) * scale,
        unit=unit,
        scale=scale,
        t_us=t_us,
        hz=hz,
        T_smooth=smooth,
        points=points,
    )


def _model_points(
    session_h5: Path,
    T_smooth: np.ndarray,
    T_marker_landmark: np.ndarray | None,
    scale: float,
) -> dict[str, np.ndarray]:
    """Landmark-frame points swept by the smoothed model pose, in output units."""
    local: dict[str, np.ndarray] = {"mand_point_1": np.zeros(3, dtype=np.float64)}
    if T_marker_landmark is not None:
        try:
            T_inv = np.linalg.inv(np.asarray(T_marker_landmark, dtype=np.float64))
        except np.linalg.LinAlgError:
            T_inv = None
        if T_inv is not None:
            with h5py.File(session_h5, "r") as f:
                phases = load_phases(f)
                opt = f["optical"] if "optical" in f else None
                pose_mp = _pose(opt, "pose_mta") if opt is not None else None
                pose_ct = _pose(opt, "pose_ct") if opt is not None else None
                t_us = (
                    np.asarray(opt["t_master_us"][()], dtype=np.float64)
                    if opt is not None and "t_master_us" in opt
                    else None
                )
            if pose_mp is not None and pose_ct is not None and t_us is not None:
                n = min(len(pose_mp), len(pose_ct), len(t_us))
                clock = t_us[:n]
                for phase in phases:
                    key = phase.name.strip().lower()
                    if key not in ("tmj_left", "tmj_right") or key in local:
                        continue
                    mask = (clock >= float(phase.t_start_us)) & (clock <= float(phase.t_end_us))
                    point = _point_in_body(pose_mp[:n], pose_ct[:n], mask)
                    if point is None:
                        continue
                    hom = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
                    local[key] = (T_inv @ hom)[:3]
    R = T_smooth[:, :3, :3]
    origin = T_smooth[:, :3, 3]
    finite = np.isfinite(origin).all(axis=1) & np.isfinite(R).all(axis=(1, 2))
    out: dict[str, np.ndarray] = {}
    for name, point in local.items():
        xyz = np.einsum("nij,j->ni", R, point) + origin
        xyz[~finite] = np.nan
        out[name] = xyz * scale
    return out


def trajectory_from_model(model: ModelTrajectory, phases: list[PhaseInterval]) -> Trajectory:
    """Time-aligned landmark translation for the 2D stack and cycle marks."""
    finite = np.isfinite(model.t_us)
    t0 = float(np.min(model.t_us[finite])) if np.any(finite) else 0.0
    t_s = np.full(len(model.t_us), np.nan, dtype=np.float64)
    t_s[finite] = (model.t_us[finite] - t0) * 1e-6
    period = 1.0 / model.hz if model.hz > 0 else None
    return Trajectory(
        t_s,
        model.xyz,
        "t (s) from first exported frame, MCU master",
        t0,
        phases,
        period_s=period,
        frame="model",
        unit=model.unit,
    )


def phase_row_mask(
    t_us: np.ndarray,
    phases: list[PhaseInterval],
    *,
    include_bridges: bool = False,
) -> np.ndarray:
    """Rows whose master time falls in ``phases``.

    With ``include_bridges``, synthetic rows that sit between two kept samples
    stay in, so the cubic joins used by the combined JTS path are kept and the
    join into a dropped phase is not.
    """
    t_us = np.asarray(t_us, dtype=np.float64)
    inside = np.zeros(t_us.shape[0], dtype=bool)
    finite = np.isfinite(t_us)
    for phase in phases:
        inside |= finite & (t_us >= float(phase.t_start_us)) & (t_us <= float(phase.t_end_us))
    if not include_bridges:
        return inside
    out = inside.copy()
    idx = np.flatnonzero(finite)
    for a, b in zip(idx[:-1], idx[1:]):
        if inside[a] and inside[b]:
            out[a : b + 1] = True
    return out


def _set_model_3d_limits(ax, xyz: np.ndarray) -> None:
    """Same cube as the JTS matplotlib fallback: span about the mean, Z up."""
    pts = xyz[np.isfinite(xyz).all(axis=1)]
    if pts.size == 0:
        return
    span = float(np.ptp(pts, axis=0).max()) or 1.0
    mid = pts.mean(axis=0)
    half = 0.55 * span
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)


def plot_model_landmark_3d(
    xyz: np.ndarray,
    *,
    unit: str,
    title: str,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """Same static view as ``jaw_motion_3d.png``: Z up, data aspect, one line."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], lw=1.2)
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    ax.set_zlabel(f"Z [{unit}]")
    ax.set_title(title)
    _set_model_3d_limits(ax, xyz)
    ax.grid(True, alpha=0.3)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_model_phases_3d(
    model: ModelTrajectory,
    phases: list[PhaseInterval],
    *,
    title: str,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """One Z-up axes per phase, same landmark series as the combined 3D."""
    import matplotlib.pyplot as plt

    panels: list[tuple[str, str, np.ndarray]] = []
    for i, phase in enumerate(from_rest_phases(phases)):
        mask = phase_row_mask(model.t_us, [phase])
        if int(mask.sum()) < 2:
            continue
        panels.append((phase.name, _phase_color(i), model.xyz[mask]))
    if not panels:
        print("  3D by phase: no finite model-frame samples")
        return

    n = len(panels)
    ncols = 3 if n > 4 else (2 if n > 1 else 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (name, color, xyz) in zip(axes_flat, panels):
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel(f"X [{model.unit}]", fontsize=8)
        ax.set_ylabel(f"Y [{model.unit}]", fontsize=8)
        ax.set_zlabel(f"Z [{model.unit}]", fontsize=8)
        _set_model_3d_limits(ax, xyz)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_model_points_3d(
    points: dict[str, np.ndarray],
    *,
    unit: str,
    title: str,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """``mand_point_1`` and TMJ tracks in the model frame, Z up."""
    import matplotlib.pyplot as plt

    series = [(name, points[name]) for name in _LANDMARK_NAMES if name in points]
    if not series:
        print("  landmarks: none in the model frame")
        return
    fig = plt.figure(figsize=(8.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    drawn: list[np.ndarray] = []
    for name, xyz in series:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=_LANDMARK_COLORS[name], lw=1.2, label=name)
        drawn.append(xyz)
    ax.set_xlabel(f"X [{unit}]")
    ax.set_ylabel(f"Y [{unit}]")
    ax.set_zlabel(f"Z [{unit}]")
    ax.set_title(title)
    _set_model_3d_limits(ax, np.concatenate(drawn, axis=0))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_xyz_3d_combined(
    traj: Trajectory,
    *,
    title: str,
    save: Path | None = None,
    show: bool = True,
    phases: list[PhaseInterval] | None = None,
) -> None:
    """One 3D axes, single color. Motion phases only; gaps are not connected."""
    import matplotlib.pyplot as plt

    if phases is None:
        phases = from_rest_phases(traj.phases)
    segments: list[np.ndarray] = []
    if phases:
        for phase in phases:
            mask = phase_mask(traj, phase)
            if int(mask.sum()) >= 2:
                segments.append(_gapped(traj, mask))
    else:
        valid = np.isfinite(traj.xyz_mm).all(axis=1) & np.isfinite(traj.t_s)
        if int(valid.sum()) >= 2:
            segments.append(_gapped(traj, valid))

    if not segments:
        print("  3D combined: no finite motion-phase samples")
        return

    fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    all_xyz = np.concatenate(segments, axis=0)
    for xyz in segments:
        _line_3d(ax, xyz, color=_COMBINED_3D_COLOR, label=None)
    _set_equal_3d(ax, all_xyz)
    _style_3d_axes(ax, frame=traj.frame)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_xyz_3d_by_phase(
    traj: Trajectory,
    *,
    title: str,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """One 3D axes per phase: MP–HP movement from the rest pose, own scale."""
    import matplotlib.pyplot as plt

    offset = rest_offset_mm(traj) if traj.frame == "hp" else np.zeros(3)
    panels: list[tuple[str, str, np.ndarray]] = []
    plot_phases = from_rest_phases(traj.phases)
    for i, phase in enumerate(traj.phases):
        if phase not in plot_phases:
            continue
        mask = phase_mask(traj, phase)
        if int(mask.sum()) < 2:
            continue
        moved = traj.xyz_mm - offset
        panels.append((phase.name, _phase_color(i), _gapped(traj, mask, xyz=moved)))
    if not panels:
        print("  3D by phase: no finite motion-phase samples")
        return

    n = len(panels)
    ncols = 3 if n > 4 else (2 if n > 1 else 1)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 4.6 * nrows),
        subplot_kw={"projection": "3d"},
        constrained_layout=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    frame = traj.frame
    for ax, (name, color, xyz) in zip(axes_flat, panels):
        _line_3d(ax, xyz, color=color, label=None, lw=1.2)
        _set_equal_3d(ax, xyz)
        ax.set_title(name, fontsize=10)
        _style_3d_axes(ax, xlabel_size=8, frame="hp_rest" if frame == "hp" else frame)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title)

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _phase_spans_s(
    phases: list[PhaseInterval],
    t0_us: float | None,
) -> list[tuple[str, float, float, str]]:
    """Map absolute phase µs → relative seconds (same axis as the plot)."""
    if t0_us is None:
        return []
    out: list[tuple[str, float, float, str]] = []
    for p in phases:
        out.append(
            (
                p.name,
                (float(p.t_start_us) - t0_us) * 1e-6,
                (float(p.t_end_us) - t0_us) * 1e-6,
                p.kind,
            )
        )
    return out


def plot_xyz(
    traj: Trajectory,
    *,
    title: str,
    save: Path | None = None,
    show: bool = True,
    show_phases: bool = True,
    show_cycles: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    valid = np.isfinite(traj.xyz_mm).all(axis=1) & np.isfinite(traj.t_s)
    t = traj.t_s[valid]
    xyz = traj.xyz_mm[valid]
    if xyz.shape[0] == 0:
        raise SystemExit("no finite MP samples to plot")

    plot_phases = from_rest_phases(traj.phases) if show_phases else []
    if show_phases and not plot_phases:
        plot_phases = list(traj.phases)
    spans = _phase_spans_s(plot_phases, traj.t0_us) if show_phases else []

    if show_phases and traj.t0_us is not None and plot_phases:
        t_abs = traj.t0_us + t * 1e6
        t_start = float(plot_phases[0].t_start_us)
        keep = t_abs >= t_start
        t = t[keep]
        xyz = xyz[keep]
        if xyz.shape[0] == 0:
            raise SystemExit("no finite MP samples from rest onward")
        t_abs = traj.t0_us + t * 1e6
        cal_mask = np.zeros(t.shape[0], dtype=bool)
        for p in traj.phases:
            if is_calibration_phase(p):
                cal_mask |= (t_abs >= float(p.t_start_us)) & (t_abs <= float(p.t_end_us))
        if np.any(cal_mask) and not np.all(cal_mask):
            xyz = xyz.copy()
            xyz[cal_mask] = np.nan

    # Break the line wherever frames are missing (dt > 1.5 frame periods).
    t, xyz = break_gaps(t, xyz, period=traj.period_s)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5), constrained_layout=True)
    labels = _axis_labels(traj.frame, unit=traj.unit)
    colors = ("#c0392b", "#27ae60", "#2980b9")
    for ax, col, lab, c in zip(axes, range(3), labels, colors):
        for i, (name, t0, t1, _kind) in enumerate(spans):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            ax.axvspan(t0, t1, color=color, alpha=0.35, lw=0, zorder=0)
            if ax is axes[0]:
                ax.axvline(t0, color="#7f8c8d", lw=0.6, alpha=0.7, zorder=1)
                ax.axvline(t1, color="#7f8c8d", lw=0.6, alpha=0.7, zorder=1)
                mid = 0.5 * (t0 + t1)
                ax.text(
                    mid,
                    0.98,
                    name,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    rotation=90 if (t1 - t0) < max(0.05 * (np.nanmax(t) - np.nanmin(t) if t.size > 1 else 1.0), 0.2) else 0,
                    clip_on=True,
                    zorder=3,
                )
        ax.plot(t, xyz[:, col], color=c, lw=1.0, zorder=2)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3, zorder=1)
    cycle_ts = cycle_boundary_times_s(traj) if show_cycles else []
    for ax in axes:
        for tc in cycle_ts:
            ax.axvline(tc, **_CYCLE_LINE)
    axes[-1].set_xlabel(traj.time_label)
    fig.suptitle(title)

    handles: list = []
    if spans:
        handles.extend(
            Patch(facecolor=_PHASE_COLORS[i % len(_PHASE_COLORS)], alpha=0.45, label=name)
            for i, (name, *_rest) in enumerate(spans)
        )
    if cycle_ts:
        handles.append(Line2D([0], [0], color=_CYCLE_LINE["color"], ls=":", lw=1.2, label="cycles"))
    if handles:
        axes[0].legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            ncol=min(4, max(1, len(handles))),
            framealpha=0.9,
        )

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_landmarks_3d(
    traj: Trajectory,
    *,
    title: str,
    save: Path | None = None,
    show: bool = True,
) -> None:
    """Incisal point in red and both TMJ points in green, in the HP frame."""
    import matplotlib.pyplot as plt

    phases = from_rest_phases(traj.phases)
    drawn: list[np.ndarray] = []
    labeled: set[str] = set()
    fig = None
    ax = None
    for name in _LANDMARK_NAMES:
        xyz = traj.landmarks.get(name)
        if xyz is None:
            continue
        pieces: list[np.ndarray] = []
        if phases:
            for phase in phases:
                mask = phase_mask(traj, phase, xyz)
                if int(mask.sum()) >= 2:
                    pieces.append(_gapped(traj, mask, xyz))
        else:
            valid = np.isfinite(xyz).all(axis=1) & np.isfinite(traj.t_s)
            if int(valid.sum()) >= 2:
                pieces.append(_gapped(traj, valid, xyz))
        if not pieces:
            continue
        if fig is None:
            fig = plt.figure(figsize=(8.5, 7.5), constrained_layout=True)
            ax = fig.add_subplot(111, projection="3d")
        color = _LANDMARK_COLORS[name]
        for piece in pieces:
            label = name if name not in labeled else None
            labeled.add(name)
            _line_3d(ax, piece, color=color, label=label, lw=1.4)
            drawn.append(piece)

    if fig is None or ax is None or not drawn:
        print("  landmarks: none (need mand_point_1 / tmj_left / tmj_right with CT + MP)")
        return

    _set_equal_3d(ax, np.concatenate(drawn, axis=0))
    _style_3d_axes(ax, frame="hp")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    missing = [name for name in _LANDMARK_NAMES if name not in traj.landmarks]
    if missing:
        print(f"  landmarks missing: {', '.join(missing)}")

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150)
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _output_paths(
    path: Path,
    body: str,
    save: Path | None,
    out: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
    """Return (2D, 3D combined, 3D without last phase, 3D by-phase, landmarks) paths."""
    if save is not None:
        suffix = save.suffix or ".png"
        return (
            save,
            save.with_name(f"{save.stem}_3d{suffix}"),
            save.with_name(f"{save.stem}_3d_no_last{suffix}"),
            save.with_name(f"{save.stem}_3d_phases{suffix}"),
            save.with_name(f"{save.stem}_landmarks{suffix}"),
        )
    if out is not None:
        tag = f"{path.stem}_{body.strip().lower()}_xyz"
        return (
            out / f"{tag}.png",
            out / f"{tag}_3d.png",
            out / f"{tag}_3d_no_last.png",
            out / f"{tag}_3d_phases.png",
            out / f"{tag}_landmarks.png",
        )
    return None, None, None, None, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, help="session.h5 or optical tracking HDF5")
    ap.add_argument(
        "--body",
        default="MP",
        help="Body to plot: MP/MTA (default), HP/CRA, or CT",
    )
    ap.add_argument("--save", type=Path, default=None, help="2D XYZ PNG/PDF (3D files use the same stem)")
    ap.add_argument("--out", type=Path, default=None, help="Directory for 2D + 3D PNGs (alternative to --save)")
    ap.add_argument("--no-show", action="store_true", help="Do not open an interactive window")
    ap.add_argument("--no-phases", action="store_true", help="Do not shade experiment phases on the 2D plot")
    ap.add_argument("--no-cycles", action="store_true", help="Do not mark peak-to-peak cycles on the 2D XYZ plot")
    args = ap.parse_args(argv)

    path = args.session
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    save_2d, save_3d, save_3d_no_last, save_3d_phases, save_landmarks = _output_paths(
        path, args.body, args.save, args.out
    )
    if args.no_show and save_2d is None:
        print("error: --no-show needs --save or --out", file=sys.stderr)
        return 1

    try:
        traj = load_trajectory(path, body=args.body)
    except (KeyError, ValueError, OSError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {msg}", file=sys.stderr)
        return 1
    n_finite = int(np.isfinite(traj.xyz_mm).all(axis=1).sum())
    frame_note = " relative to HP" if traj.frame == "hp" else ""
    print(f"{path}: {args.body.upper()}{frame_note}  N={traj.xyz_mm.shape[0]}  finite={n_finite}")
    rest_onward = from_rest_phases(traj.phases) if traj.phases else []
    no_last = from_rest_phases_drop_last(traj.phases) if traj.phases else []
    if traj.phases:
        print(f"  phases: {len(traj.phases)} ({', '.join(p.name for p in traj.phases)})")
        print(
            f"  plot from rest: {len(rest_onward)} "
            f"({', '.join(p.name for p in rest_onward) or 'none'})"
        )
        dropped = rest_onward[-1].name if rest_onward and no_last else None
        if dropped:
            print(
                f"  3D without last phase ({dropped}): {len(no_last)} "
                f"({', '.join(p.name for p in no_last)})"
            )
    elif not args.no_phases:
        print("  phases: none found (/phases, /events, or sync.report_json)")

    show = not args.no_show
    body = args.body.upper()
    where = "relative to HP" if traj.frame == "hp" else "world"
    model = None
    if body in ("MP", "MTA"):
        try:
            model = jts_model_trajectory(path)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"  JTS landmark failed ({exc}); using MP origin in HP")
    if model is not None:
        print(
            f"  figures: smoothed mandibular landmark in model CS "
            f"({model.unit}, same series as jaw_motion)"
        )
        plotted = trajectory_from_model(model, traj.phases)
        plot_xyz(
            plotted,
            title=f"mandibular landmark, model CS — {path.name}",
            save=save_2d,
            show=show,
            show_phases=not args.no_phases,
            show_cycles=not args.no_cycles,
        )
        plot_model_landmark_3d(
            model.xyz,
            unit=model.unit,
            title="Jaw Motion Trajectories - 3D View",
            save=save_3d,
            show=show,
        )
        if no_last:
            dropped_name = rest_onward[-1].name
            kept = model.xyz.copy()
            kept[~phase_row_mask(model.t_us, no_last, include_bridges=True)] = np.nan
            plot_model_landmark_3d(
                kept,
                unit=model.unit,
                title=f"Jaw Motion Trajectories - 3D View (without {dropped_name})",
                save=save_3d_no_last,
                show=show,
            )
        elif save_3d_no_last is not None:
            print("  3D without last phase: skipped (need at least two phases from rest)")
        plot_model_phases_3d(
            model,
            traj.phases,
            title=f"mandibular landmark by phase, model CS — {path.name}",
            save=save_3d_phases,
            show=show,
        )
        plot_model_points_3d(
            model.points,
            unit=model.unit,
            title=f"mand_point_1 and TMJ, model CS — {path.name}",
            save=save_landmarks,
            show=show,
        )
        return 0

    plot_xyz(
        traj,
        title=f"{body} {where} — {path.name}",
        save=save_2d,
        show=show,
        show_phases=not args.no_phases,
        show_cycles=not args.no_cycles,
    )
    plot_xyz_3d_combined(
        traj,
        title=f"{body} {where} — {path.name}",
        save=save_3d,
        show=show,
    )
    if no_last:
        dropped_name = rest_onward[-1].name
        plot_xyz_3d_combined(
            traj,
            title=f"{body} {where} — without {dropped_name} — {path.name}",
            save=save_3d_no_last,
            show=show,
            phases=no_last,
        )
    elif save_3d_no_last is not None:
        print("  3D without last phase: skipped (need at least two phases from rest)")
    phase_where = "movement from rest, MP in HP" if traj.frame == "hp" else where
    plot_xyz_3d_by_phase(
        traj,
        title=f"{body} {phase_where} — {path.name}",
        save=save_3d_phases,
        show=show,
    )
    if traj.frame == "hp" or traj.landmarks:
        plot_landmarks_3d(
            traj,
            title=f"mand_point_1 and TMJ relative to HP — {path.name}",
            save=save_landmarks,
            show=show,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
