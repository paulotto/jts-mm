#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Trajectory quality for every motion session.

Writes one JSON file and one Markdown file with four figures of merit:

1. Rest scatter of the raw mandibular landmark in the dental model frame,
   ``r ± σ_r`` with ``r_i = ||p_i - mean(p)||`` (population σ).
2. Per phase, the share of optical frames where both markers have two tags
   and a finite pose, plus the gaps the Savitzky–Golay input fill bridged.
3. Pose-fit RMS of the cranial and mandibular markers at rest and in motion.
   These sessions store a triangulation Kabsch residual in millimetres
   (``optical/quality`` attribute ``rms_kind``), which is the fit column
   available in the file.
4. Cycle repeatability of the smoothed landmark. One cycle is one full
   repetition: open and close, left and right, forward and back, or one
   chew. Peaks are taken on that task axis (Y, X, Z, Y in the model frame),
   and each peak is one cycle from the trough before it to the trough after.

The same rest scatter and cycle metrics are repeated after replacing the
fast part of the cranial-marker rotation with the cranial IMU quaternion.
Translations and the slow orientation stay optical. The constant
camera-from-IMU map is fit on the rest phase of that recording.

One take per participant: the latest motion take (``--take first`` for the
pre-specified first take of an01 §2).

Usage
-----
  python eval_trajectory_quality.py
  python eval_trajectory_quality.py --data-dir ~/jtsmm_data --subject s03 --out-dir /tmp/tq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from biomech.cohort import discover_sessions as _discover
from biomech.io.inventory import select_takes, subject_pids
from biomech.kinematics.cycles import time_normalize
from biomech.paths import resolve_data_dir, results_root

CYCLE_N = 101
CYCLE_MOTIONS = ("open_close", "protrusion_retrusion", "left_right", "chewing")
# Model frame: X left–right, Y opening, Z forward–back.
_CYCLE_AXIS = {
    "open_close": 1,
    "protrusion_retrusion": 2,
    "left_right": 0,
    "chewing": 1,
}
_CYCLE_PROMINENCE = 0.25
_CYCLE_MIN_PERIOD_S = 0.5
PHASE_ORDER = ("rest",) + CYCLE_MOTIONS + ("free_motion",)


def discover_sessions(root: Path, subject: str | None = None, take: str = "latest"
                      ) -> list[tuple[str, Path]]:
    """``(subject_id, session.h5)`` of one motion take per participant (exact pid match)."""
    pids = subject_pids(subject)
    motion = [s for s in _discover(root)
              if s.kind == "motion" and (pids is None or s.pid in pids)]
    return [(f"{s.pid}_{s.sex}_{s.age}", s.h5) for s in select_takes(motion, take)]


def default_out_dir() -> Path:
    return results_root() / "data" / "trajectory_quality"


def _num(value: float, digits: int = 4) -> float | None:
    x = float(value)
    if not np.isfinite(x):
        return None
    return round(x, digits)


def _fmt(value: float | None, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _pm(mean: float | None, sigma: float | None, digits: int) -> str:
    if mean is None or sigma is None:
        return "—"
    return f"{mean:.{digits}f} ± {sigma:.{digits}f}"


def _r_sigma(points: np.ndarray) -> dict[str, Any]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return {"r_mm": None, "sigma_r_mm": None, "n": int(pts.shape[0] if pts.ndim == 2 else 0)}
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if pts.shape[0] < 2:
        return {"r_mm": None, "sigma_r_mm": None, "n": int(pts.shape[0])}
    radius = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    return {
        "r_mm": _num(float(np.mean(radius))),
        "sigma_r_mm": _num(float(np.std(radius))),
        "n": int(pts.shape[0]),
    }


def _finite_pose(poses: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    return np.isfinite(poses[:, :3]).all(axis=(1, 2))


def _runs(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]])
    return int(np.count_nonzero(np.diff(padded.astype(np.int8)) == 1))


def _cmc(stack: np.ndarray) -> float | None:
    """Coefficient of multiple correlation of waveforms shaped (cycles, time)."""
    y = np.asarray(stack, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 2:
        return None
    keep = np.isfinite(y).all(axis=0)
    y = y[:, keep]
    if y.shape[0] < 2 or y.shape[1] < 2:
        return None
    mean_t = y.mean(axis=0)
    grand = float(y.mean())
    within = float(np.sum((y - mean_t) ** 2))
    total = float(np.sum((y - grand) ** 2))
    if total <= 0:
        return None
    ratio = 1.0 - within / total
    if ratio < 0:
        return None
    return _num(float(np.sqrt(ratio)), 4)


def _fill_1d(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).copy()
    good = np.isfinite(y)
    if int(good.sum()) < 2 or good.all():
        return y
    idx = np.arange(y.size, dtype=np.float64)
    y[~good] = np.interp(idx[~good], idx[good], y[good])
    return y


def _apply_fused_cranial_rotation(analysis: Any, session_h5: Path) -> None:
    """Replace HP rotation on the trigger grid with the preview fusion.

    Optical translation is left as loaded. The fast band of ``R_HP`` (above
    ``SLOW_S``) follows the cranial IMU quaternion, aligned on the rest phase.
    """
    import h5py

    from preview_imu_rotation import SLOW_S, _align, _finite_pose, _fuse_rotation, _quat_wxyz_to_R, _slerp_R

    with h5py.File(session_h5, "r") as f:
        t = np.asarray(f["optical/t_master_us"][()], dtype=np.float64)
        cra = np.asarray(f["optical/pose_cra"][()], dtype=np.float64)
        n_hp = np.asarray(f["optical/quality/cra/n_tags"][()], dtype=np.int16)
        names = [_decode_phase(x) for x in f["phases/name"][()]]
        t0 = np.asarray(f["phases/t_start_us"][()], dtype=np.float64)
        t1 = np.asarray(f["phases/t_end_us"][()], dtype=np.float64)
        t_imu = np.asarray(f["imu_cra/t_master_us_quat"][()], dtype=np.float64)
        quat = np.asarray(f["imu_cra/quat"][()], dtype=np.float64)
    rest = np.zeros(len(t), dtype=bool)
    for name, a, b in zip(names, t0, t1):
        if name.strip().lower() == "rest":
            rest |= (t >= a) & (t <= b)
    finite = _finite_pose(cra)
    rest &= finite & (n_hp[: len(t)] == 2)
    if int(rest.sum()) < 50:
        raise RuntimeError("rest phase too short to align the cranial IMU")
    R_imu = _slerp_R(t_imu, _quat_wxyz_to_R(quat), t)
    R_align = _align(cra[rest, :3, :3], R_imu[rest])
    R_imu_cam = np.einsum("ij,njk->nik", R_align, R_imu)
    both = finite & np.isfinite(R_imu_cam).reshape(len(t), -1).all(axis=1)
    R_opt = np.asarray(cra[:, :3, :3], dtype=np.float64)
    R_fused = np.array(R_opt, copy=True)
    idx = np.flatnonzero(both)
    R_fused[idx] = _fuse_rotation(R_opt[idx], R_imu_cam[idx], SLOW_S)

    aruco = analysis.motion_data.aruco_data
    row_frame = np.asarray(aruco.timestamps["row_frame"], dtype=np.int64)
    hp = analysis.motion_data.rigid_bodies["HP"]
    n = min(len(row_frame), len(R_fused))
    keep = row_frame[:n] >= 0
    hp.rotations[row_frame[:n][keep]] = R_fused[:n][keep]


def _decode_phase(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_landmark(session_h5: Path, *, fuse_hp: bool = False) -> dict[str, Any]:
    """Raw and smoothed model-frame landmark, millimetres, plus the gap mask."""
    from jtsmm.tools.jts_path import ensure_jts_on_path
    from jtsmm.tools.run_jts_analysis import (
        DEFAULT_MAX_GAP_S,
        _valid_pose_mask,
        find_jts_config,
        interpolate_se3_gaps,
        prepare_config,
        resolve_session_h5,
    )

    session_h5 = resolve_session_h5(session_h5)
    config_path = find_jts_config(session_h5, None)
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
    if fuse_hp:
        _apply_fused_cranial_rotation(analysis, session_h5)
    analysis.perform_calibration()
    analysis.compute_relative_motion()
    analysis.transform_to_ref_markers_coordinates()
    analysis.register_to_model_coordinates()
    key = "T_model_origin_mand_landmark_t"
    raw = analysis.trajectories.get(key)
    if raw is None:
        raise RuntimeError("model landmark trajectory missing")
    raw = np.asarray(raw, dtype=np.float64)
    smooth_cfg = (cfg.get("analysis") or {}).get("smoothing") or {}
    max_gap_s = float(smooth_cfg.get("max_gap_s", DEFAULT_MAX_GAP_S))
    hz = float(getattr(analysis.motion_data, "frame_rate", 0.0) or 0.0)
    cap = int(round(max_gap_s * hz)) if hz > 0 else 0
    filled, _n_fill = interpolate_se3_gaps(raw, max_gap_frames=cap)
    valid_raw = _valid_pose_mask(raw)
    valid_filled = _valid_pose_mask(filled)
    gap_filled = ~valid_raw & valid_filled
    left_nan = ~valid_filled
    analysis.trajectories[key] = filled
    analysis.smooth_trajectory()
    smooth = analysis.trajectories.get(key + "_smooth")
    if smooth is None:
        raise RuntimeError("smoothed landmark missing")
    smooth = np.asarray(smooth, dtype=np.float64)
    t_us = analysis.frame_times_us()
    if t_us is None or len(t_us) != len(raw):
        t_us = np.full(len(raw), np.nan)
    else:
        t_us = np.asarray(t_us, dtype=np.float64)
    return {
        "hz": hz,
        "max_gap_s": max_gap_s,
        "t_us": t_us,
        "xyz_raw_mm": raw[:, :3, 3],
        "xyz_smooth_mm": smooth[:, :3, 3],
        "gap_filled": gap_filled,
        "left_nan": left_nan,
    }


def _optical(session_h5: Path) -> dict[str, Any]:
    import h5py

    from jtsmm.tools.plot_mp_trajectory import load_phases

    with h5py.File(session_h5, "r") as f:
        phases = load_phases(f)
        opt = f["optical"]
        t_us = np.asarray(opt["t_master_us"][()], dtype=np.float64)
        pose_hp = np.asarray(opt["pose_cra"][()], dtype=np.float64)
        pose_mp = np.asarray(opt["pose_mta"][()], dtype=np.float64)
        quality = opt["quality"]
        rms_kind = str(quality.attrs.get("rms_kind") or "unknown")
        n_hp = np.asarray(quality["cra"]["n_tags"][()], dtype=np.int16)
        n_mp = np.asarray(quality["mta"]["n_tags"][()], dtype=np.int16)
        rms_hp = np.asarray(quality["cra"]["reprojection_rms"][()], dtype=np.float64)
        rms_mp = np.asarray(quality["mta"]["reprojection_rms"][()], dtype=np.float64)
    n = min(len(t_us), len(pose_hp), len(pose_mp), len(n_hp), len(n_mp), len(rms_hp), len(rms_mp))
    return {
        "phases": phases,
        "t_us": t_us[:n],
        "both_pose": _finite_pose(pose_hp[:n]) & _finite_pose(pose_mp[:n]),
        "both_tags": (n_hp[:n] == 2) & (n_mp[:n] == 2),
        "rms_hp": rms_hp[:n],
        "rms_mp": rms_mp[:n],
        "rms_kind": rms_kind,
    }


def _in_phase(t_us: np.ndarray, phase: Any) -> np.ndarray:
    t = np.asarray(t_us, dtype=np.float64)
    return np.isfinite(t) & (t >= float(phase.t_start_us)) & (t <= float(phase.t_end_us))


def _phase_rows(optical: dict[str, Any], landmark: dict[str, Any]) -> list[dict[str, Any]]:
    hz = float(landmark["hz"]) if landmark["hz"] else float("nan")
    rows = []
    by_name = {p.name.strip().lower(): p for p in optical["phases"]}
    for name in PHASE_ORDER:
        phase = by_name.get(name)
        if phase is None:
            continue
        opt_m = _in_phase(optical["t_us"], phase)
        n = int(opt_m.sum())
        n_valid = int((opt_m & optical["both_tags"] & optical["both_pose"]).sum())
        jts_m = _in_phase(landmark["t_us"], phase)
        filled = jts_m & landmark["gap_filled"]
        left = jts_m & landmark["left_nan"]
        n_filled = int(filled.sum())
        rows.append(
            {
                "phase": name,
                "n_frames": n,
                "n_valid": n_valid,
                "pct_valid": _num(100.0 * n_valid / n, 2) if n else None,
                "n_gap_frames": n_filled,
                "n_gaps": _runs(filled),
                "gap_duration_s": _num(n_filled / hz, 3) if hz and np.isfinite(hz) else None,
                "n_unfilled": int(left.sum()),
            }
        )
    return rows


def _rms(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64)[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"rms": None, "n": 0}
    return {"rms": _num(float(np.sqrt(np.mean(v ** 2)))), "n": int(v.size)}


def _fit(optical: dict[str, Any]) -> dict[str, Any]:
    by_name = {p.name.strip().lower(): p for p in optical["phases"]}
    rest = by_name.get("rest")
    rest_m = _in_phase(optical["t_us"], rest) if rest is not None else np.zeros(len(optical["t_us"]), dtype=bool)
    motion_m = np.zeros(len(optical["t_us"]), dtype=bool)
    for name in CYCLE_MOTIONS + ("free_motion",):
        phase = by_name.get(name)
        if phase is not None:
            motion_m |= _in_phase(optical["t_us"], phase)
    unit = "mm" if optical["rms_kind"] == "kabsch_mm" else ("px" if optical["rms_kind"] == "reprojection_px" else optical["rms_kind"])
    out: dict[str, Any] = {"rms_kind": optical["rms_kind"], "unit": unit}
    for label, mask in (("rest", rest_m), ("motion", motion_m)):
        hp = _rms(optical["rms_hp"], mask)
        mp = _rms(optical["rms_mp"], mask)
        out[label] = {"hp": hp["rms"], "mp": mp["rms"], "n_hp": hp["n"], "n_mp": mp["n"]}
    return out


def _task_signal(column: np.ndarray) -> np.ndarray:
    """Demeaned axis, oriented so the larger excursion is positive."""
    y = np.asarray(column, dtype=np.float64)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y.copy()
    dev = y - float(np.median(finite))
    if float(np.max(finite - np.median(finite))) < float(-(np.min(finite - np.median(finite)))):
        dev = -dev
    return dev


def _repetitions(y: np.ndarray, hz: float) -> list[tuple[int, int]]:
    """One cycle per peak: trough before the peak to trough after it."""
    from scipy.signal import find_peaks

    signal = _fill_1d(y)
    span = float(np.nanmax(signal) - np.nanmin(signal))
    if not np.isfinite(span) or span < 0.5 or hz <= 0:
        return []
    distance = max(int(_CYCLE_MIN_PERIOD_S * hz), 2)
    peaks, _props = find_peaks(signal, distance=distance, prominence=_CYCLE_PROMINENCE * span)
    peaks = [int(p) for p in peaks]
    n = len(signal)
    out: list[tuple[int, int]] = []
    min_len = max(int(0.3 * hz), 4)
    for i, peak in enumerate(peaks):
        left0 = 0 if i == 0 else peaks[i - 1]
        right1 = n - 1 if i + 1 == len(peaks) else peaks[i + 1]
        left = left0 + int(np.argmin(signal[left0 : peak + 1]))
        right = peak + int(np.argmin(signal[peak : right1 + 1]))
        if right - left >= min_len:
            out.append((left, right))
    return out


def _cycles(optical: dict[str, Any], landmark: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {p.name.strip().lower(): p for p in optical["phases"]}
    rest = by_name.get("rest")
    raw_rest = landmark["xyz_smooth_mm"][_in_phase(landmark["t_us"], rest)] if rest is not None else np.empty((0, 3))
    raw_rest = raw_rest[np.isfinite(raw_rest).all(axis=1)]
    origin = np.mean(raw_rest, axis=0) if raw_rest.shape[0] else np.zeros(3)
    hz = float(landmark["hz"]) if landmark["hz"] else 40.0
    rows = []
    for name in CYCLE_MOTIONS:
        phase = by_name.get(name)
        if phase is None:
            continue
        inside = _in_phase(landmark["t_us"], phase)
        idx = np.flatnonzero(inside)
        empty = {
            "motion": name,
            "n_cycles": 0,
            "rms_mean_mm": None,
            "rms_sd_mm": None,
            "rom_mean_mm": None,
            "rom_sd_mm": None,
            "cmc": None,
        }
        if idx.size < 8:
            rows.append(empty)
            continue
        sl = slice(int(idx[0]), int(idx[-1]) + 1)
        xyz = landmark["xyz_smooth_mm"][sl]
        task = _task_signal(xyz[:, _CYCLE_AXIS[name]])
        amp = np.linalg.norm(xyz - origin, axis=1)
        amp = np.where(np.isfinite(xyz).all(axis=1), amp, np.nan)
        segs = _repetitions(task, hz)
        curves: list[np.ndarray] = []
        waves: list[np.ndarray] = []
        roms: list[float] = []
        for a, b in segs:
            block = xyz[a : b + 1]
            if int(np.isfinite(block).all(axis=1).sum()) < 4:
                continue
            curve = time_normalize(block, CYCLE_N)
            if not np.isfinite(curve).all():
                continue
            opening = amp[a : b + 1]
            opening = opening[np.isfinite(opening)]
            wave = task[a : b + 1]
            if opening.size < 4 or int(np.isfinite(wave).sum()) < 4:
                continue
            curves.append(curve)
            waves.append(time_normalize(wave, CYCLE_N))
            roms.append(float(np.max(opening) - np.min(opening)))
        if len(curves) < 1:
            rows.append(empty)
            continue
        stack = np.stack(curves, axis=0)
        mean = np.mean(stack, axis=0)
        per = np.sqrt(np.mean(np.sum((stack - mean) ** 2, axis=2), axis=1))
        rom = np.asarray(roms, dtype=np.float64)
        rows.append(
            {
                "motion": name,
                "n_cycles": int(len(curves)),
                "rms_mean_mm": _num(float(np.mean(per))),
                "rms_sd_mm": _num(float(np.std(per))) if per.size >= 2 else None,
                "rom_mean_mm": _num(float(np.mean(rom))),
                "rom_sd_mm": _num(float(np.std(rom))) if rom.size >= 2 else None,
                "cmc": _cmc(np.stack(waves, axis=0)),
            }
        )
    return rows


def _landmark_metrics(optical: dict[str, Any], landmark: dict[str, Any]) -> dict[str, Any]:
    rest = next((p for p in optical["phases"] if p.name.strip().lower() == "rest"), None)
    rest_pts = (
        landmark["xyz_raw_mm"][_in_phase(landmark["t_us"], rest)]
        if rest is not None
        else np.empty((0, 3))
    )
    return {"rest_scatter": _r_sigma(rest_pts), "cycles": _cycles(optical, landmark)}


def evaluate(subject: str, session_h5: Path) -> dict[str, Any]:
    optical = _optical(session_h5)
    landmark = load_landmark(session_h5, fuse_hp=False)
    fused = load_landmark(session_h5, fuse_hp=True)
    metrics = _landmark_metrics(optical, landmark)
    imu = _landmark_metrics(optical, fused)
    imu["method"] = (
        "fast cranial rotation from imu_cra quaternion; "
        "constant camera-from-IMU map fit on rest; slow orientation and both translations stay optical"
    )
    return {
        "subject": subject,
        "session": str(session_h5),
        "sample_rate_hz": _num(landmark["hz"], 4),
        "rest_scatter": metrics["rest_scatter"],
        "phases": _phase_rows(optical, landmark),
        "pose_fit": _fit(optical),
        "cycles": metrics["cycles"],
        "imu": imu,
    }


def _markdown(payload: dict[str, Any]) -> str:
    subjects = payload["subjects"]
    fit_unit = "mm"
    for row in subjects:
        if "pose_fit" in row:
            fit_unit = str(row["pose_fit"].get("unit") or fit_unit)
            break
    lines = [
        "# Quality of the jaw trajectories",
        "",
        "Landmark in the dental model coordinate system. Rest scatter on the unsmoothed trajectory, cycles on the smoothed trajectory (Savitzky–Golay, window 11, order 3). σ is the population standard deviation.",
        "",
        "## Rest scatter",
        "",
        "`r_i = ||p_i − p̄||` over the rest phase.",
        "",
        "| Subject | r ± σ_r (mm) | n |",
        "|---|---:|---:|",
    ]
    for row in subjects:
        if "error" in row:
            lines.append(f"| {row['subject']} | {row['error']} | — |")
            continue
        rest = row["rest_scatter"]
        lines.append(
            f"| {row['subject']} | {_pm(rest.get('r_mm'), rest.get('sigma_r_mm'), 2)} | {rest.get('n')} |"
        )
    lines += [
        "",
        "## Valid frames and filled gaps",
        "",
        "Valid: both markers with two tags and a finite pose. Gaps: frames that were interpolated over at most 0.5 s before smoothing.",
        "",
        "| Subject | Phase | valid (%) | n | filled frames | gaps | duration (s) | open |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in subjects:
        if "error" in row:
            continue
        for phase in row["phases"]:
            lines.append(
                "| {subject} | {phase} | {pct} | {n} | {filled} | {gaps} | {dur} | {open} |".format(
                    subject=row["subject"],
                    phase=phase["phase"],
                    pct=_fmt(phase["pct_valid"], 1),
                    n=phase["n_frames"],
                    filled=phase["n_gap_frames"],
                    gaps=phase["n_gaps"],
                    dur=_fmt(phase["gap_duration_s"], 2),
                    open=phase["n_unfilled"],
                )
            )
    lines += [
        "",
        f"## Pose fit ({fit_unit})",
        "",
        "RMS of the stored fit residual. With triangulation this is the Kabsch residual of the two tags, in millimetres.",
        "",
        f"| Subject | Condition | HP ({fit_unit}) | n HP | MP ({fit_unit}) | n MP |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in subjects:
        if "error" in row:
            continue
        fit = row["pose_fit"]
        for label, title in (("rest", "rest"), ("motion", "motion")):
            block = fit[label]
            lines.append(
                f"| {row['subject']} | {title} | {_fmt(block['hp'], 3)} | {block['n_hp']} | {_fmt(block['mp'], 3)} | {block['n_mp']} |"
            )
    lines += [
        "",
        "## Cycles",
        "",
        "One cycle is one full repetition: opening and closing on Y, left and right on X, forward and back on Z, one chewing cycle on Y. One peak is counted per repetition, from the trough before it to the trough after it. RMS is the mean 3D distance to the mean curve. ROM is max − min of `||p − p_rest||`. CMC is the agreement of this task axis (0 to 1).",
        "",
        "| Subject | Task | Cycles | RMS to the mean curve (mm) | ROM (mm) | CMC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in subjects:
        if "error" in row:
            continue
        for cycle in row["cycles"]:
            lines.append(
                "| {subject} | {motion} | {n} | {rms} | {rom} | {cmc} |".format(
                    subject=row["subject"],
                    motion=cycle["motion"],
                    n=cycle["n_cycles"],
                    rms=_pm(cycle["rms_mean_mm"], cycle["rms_sd_mm"], 2),
                    rom=_pm(cycle["rom_mean_mm"], cycle["rom_sd_mm"], 2),
                    cmc=_fmt(cycle["cmc"], 3),
                )
            )
    lines += [
        "",
        "## IMU-corrected trajectory",
        "",
        "The same landmark after the fast part of the cranial rotation (above 1 s) was replaced by the quaternion of the cranial IMU. Translation and slow orientation stay optical. The constant map camera←IMU is estimated on the rest phase of the same recording, because the sessions contain no accepted mounting rotation. Pose fit and valid frames do not change by this and are given above.",
        "",
        "### Rest scatter",
        "",
        "| Subject | optical r ± σ_r (mm) | IMU r ± σ_r (mm) | n |",
        "|---|---:|---:|---:|",
    ]
    for row in subjects:
        if "error" in row or "imu" not in row:
            continue
        optical = row["rest_scatter"]
        imu = row["imu"]["rest_scatter"]
        lines.append(
            "| {subject} | {opt} | {imu} | {n} |".format(
                subject=row["subject"],
                opt=_pm(optical.get("r_mm"), optical.get("sigma_r_mm"), 2),
                imu=_pm(imu.get("r_mm"), imu.get("sigma_r_mm"), 2),
                n=imu.get("n"),
            )
        )
    lines += [
        "",
        "### Cycles",
        "",
        "| Subject | Task | n optical | n IMU | RMS optical (mm) | RMS IMU (mm) | ROM optical (mm) | ROM IMU (mm) | CMC optical | CMC IMU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in subjects:
        if "error" in row or "imu" not in row:
            continue
        by_motion = {c["motion"]: c for c in row["imu"]["cycles"]}
        for cycle in row["cycles"]:
            imu = by_motion.get(cycle["motion"], {})
            lines.append(
                "| {subject} | {motion} | {n_o} | {n_i} | {rms_o} | {rms_i} | {rom_o} | {rom_i} | {cmc_o} | {cmc_i} |".format(
                    subject=row["subject"],
                    motion=cycle["motion"],
                    n_o=cycle["n_cycles"],
                    n_i=imu.get("n_cycles"),
                    rms_o=_pm(cycle.get("rms_mean_mm"), cycle.get("rms_sd_mm"), 2),
                    rms_i=_pm(imu.get("rms_mean_mm"), imu.get("rms_sd_mm"), 2),
                    rom_o=_pm(cycle.get("rom_mean_mm"), cycle.get("rom_sd_mm"), 2),
                    rom_i=_pm(imu.get("rom_mean_mm"), imu.get("rom_sd_mm"), 2),
                    cmc_o=_fmt(cycle.get("cmc"), 3),
                    cmc_i=_fmt(imu.get("cmc"), 3),
                )
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", "--data", dest="data_dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--take", choices=("latest", "first"), default="latest",
                    help="which motion take per participant (default: latest)")
    ap.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=None,
                    help="directory for the JSON and Markdown files "
                    "(default: $BIOMECH_RESULTS_DIR or ~/experiments, then /data/trajectory_quality)")
    args = ap.parse_args(argv)
    data_dir = resolve_data_dir(args.data_dir)
    sessions = discover_sessions(data_dir, args.subject, args.take)
    if not sessions:
        print(f"error: no motion sessions in {data_dir}", file=sys.stderr)
        return 1
    out_dir = (args.out_dir or default_out_dir()).expanduser()
    rows: list[dict[str, Any]] = []
    for subject, h5 in sessions:
        print(f"{subject} …", flush=True)
        try:
            rows.append(evaluate(subject, h5))
        except (OSError, ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            rows.append({"subject": subject, "session": str(h5), "error": str(exc)})
    payload = {
        "definitions": {
            "rest_scatter": "r_i = ||p_i - mean(p)|| on the raw model-frame landmark during rest; r_mm ± sigma_r_mm, population std, millimetres",
            "pct_valid": "optical frames in the phase with both markers at 2 tags and a finite pose",
            "gaps": "frames interpolated before Savitzky-Golay, holes up to analysis.smoothing.max_gap_s (0.5 s)",
            "pose_fit": "RMS of optical/quality/*/reprojection_rms; unit is the file attribute rms_kind",
            "cycles": "one repetition per peak on the task axis (open/chew Y, left-right X, forward-back Z), trough to trough, 101 samples; CMC of that axis",
            "imu": (
                "same rest scatter and cycles after replacing the fast part of the cranial "
                "marker rotation with the imu_cra quaternion; cutoff 1 s; constant "
                "camera-from-IMU map fit by SVD on rest frames with two cranial tags"
            ),
        },
        "subjects": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trajectory_quality.json"
    md_path = out_dir / "trajectory_quality.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
