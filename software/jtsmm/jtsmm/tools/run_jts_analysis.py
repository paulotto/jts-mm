#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Run a JTS jaw-motion analysis and plot the smoothed model-frame trajectory.

Loads the session sidecar ``*_jts_config.json`` (or ``--config``), runs
``JawMotionAnalysis.run_analysis()``, writes the JTS HDF5/CSV export, and
plots raw vs Savitzky–Golay-smoothed XYZ (3×1 matplotlib).

JTS never receives tooth IDs. Landmarks are six named digitizing phases
(``mand_point_*`` / ``max_point_*``). The CT tip is held on each tooth while
those frames are recorded; ``build_jts_config`` writes the frame intervals
into the sidecar. See :func:`format_landmark_report`.

Examples::

    python -m jtsmm.tools.run_jts_analysis path/to/session.h5
    python -m jtsmm.tools.run_jts_analysis path/to/session_dir --out viz/ --no-show
    python -m jtsmm.tools.run_jts_analysis session.h5 --config s02_f_44_jts_config.json
    python -m jtsmm.tools.run_jts_analysis session.h5 --landmarks-only
    python -m jtsmm.tools.run_jts_analysis session.h5 --append-session
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from jtsmm.tools.plot_optical_phases import resolve_session_h5
from jtsmm.tools.jts_path import ensure_jts_on_path
from jtsmm.tools.timegrid import break_gaps


# Copied from optical.tracking.experiment.LANDMARK_TEETH (optical is an
# optional extra). JTS itself only sees the names + intervals.
LANDMARK_TEETH = {
    "mand_point_1": "lower central incisor (incisal edge, midline)",
    "mand_point_2": "lower left first molar",
    "mand_point_3": "lower right first molar",
    "max_point_1": "upper central incisor (incisal edge, midline)",
    "max_point_2": "upper left first molar",
    "max_point_3": "upper right first molar",
}

# Copied from optical.tracking.experiment._default_jts_config — a template
# subject's maxillary model landmarks, not this session's scan.
_TEMPLATE_MODEL_POINTS = (
    (-0.244678, -109.416, 101.356),
    (23.6189, -106.201, 73.9031),
    (-24.0445, -106.078, 74.1795),
)

_AXIS = ("X (mm)", "Y (mm)", "Z (mm)")
_RAW_COLOR = "#7f8c8d"
_SMOOTH_COLOR = "#1a5276"
_SUB_COLOR = "#c0392b"


def find_jts_config(session_h5: Path, explicit: Path | None = None) -> Path:
    """Resolve the JTS JSON sidecar next to a session, or ``explicit``."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"JTS config not found: {path}")
        return path
    parent = session_h5.parent
    matches = sorted(parent.glob("*_jts_config.json"))
    if not matches:
        raise FileNotFoundError(
            f"no *_jts_config.json next to {session_h5} "
            f"(pass --config, or re-record with an experiment plan)"
        )
    if len(matches) == 1:
        return matches[0]
    preferred = parent / f"{session_h5.stem}_jts_config.json"
    if preferred in matches:
        return preferred
    raise FileNotFoundError(
        f"multiple JTS configs in {parent}: "
        + ", ".join(p.name for p in matches)
        + " — pass --config"
    )


def _bodies_label(rigid_bodies: list[str] | None) -> str:
    if not rigid_bodies or len(rigid_bodies) < 2:
        return "marker←CT"
    return f"{rigid_bodies[0]}←{rigid_bodies[1]}"


def _model_points_are_template(model_points: Any) -> bool:
    try:
        arr = np.asarray(model_points, dtype=np.float64)
        tmpl = np.asarray(_TEMPLATE_MODEL_POINTS, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if arr.shape != tmpl.shape:
        return False
    return bool(np.allclose(arr, tmpl, rtol=1e-5, atol=1e-4))


def format_landmark_report(config: dict[str, Any]) -> str:
    """Human-readable mapping: named points → teeth, frames, Kabsch model_points."""
    analysis = config.get("analysis") or {}
    calib = analysis.get("calibration") or {}
    origin_idx = int(analysis.get("coordinate_origin_index", 0))
    xform = analysis.get("coordinate_transform") or {}
    lines: list[str] = [
        "How JTS gets landmarks (no tooth IDs are passed in)",
        "  Operator digitizes 6 named points with the CT tip during SPACE phases.",
        "  Frame intervals in this config tell JTS which samples to average.",
        "  For each point, JTS takes the CT origin in the marker CS",
        "  (mandible: MP←CT, maxilla: HP←CT), then PCA-fits a landmark frame.",
        f"  coordinate_origin_index={origin_idx} → origin is the "
        f"{origin_idx + 1}. listed point of each triplet.",
        "",
    ]
    for jaw in ("mandibular", "maxillary"):
        block = calib.get(jaw) or {}
        bodies = list(block.get("rigid_bodies") or [])
        points = list(block.get("points") or [])
        lines.append(f"  {jaw}  {_bodies_label(bodies)}")
        if not points:
            lines.append("    (no digitized points in this config)")
            continue
        for i, pt in enumerate(points):
            name = str(pt.get("name") or f"point_{i+1}")
            interval = pt.get("frame_interval")
            tooth = LANDMARK_TEETH.get(name, "not a standard JTS-MM tooth name")
            origin = "  ← landmark CS origin" if i == origin_idx else ""
            lines.append(f"    {name:14s}  frames {interval}  {tooth}{origin}")
        lines.append("")

    model_points = xform.get("model_points")
    calib_type = str(xform.get("calibration_type") or "maxillary")
    enabled = bool(xform.get("enabled", True))
    lines.append(
        f"  Kabsch model_points (calibration_type={calib_type}, enabled={enabled})"
    )
    lines.append(
        "  These are 3D points in the dental-model / scan CS, paired in list "
        "order with the digitized points of that jaw — not tooth labels."
    )
    jaw_pts = list((calib.get(calib_type) or {}).get("points") or [])
    if model_points:
        for i, xyz in enumerate(model_points):
            partner = ""
            if i < len(jaw_pts):
                partner = f"  ↔ {jaw_pts[i].get('name')}"
            lines.append(f"    [{i}] {xyz}{partner}")
    else:
        lines.append("    (none — registration will fail if enabled)")
    if enabled and _model_points_are_template(model_points):
        lines.append(
            "  WARNING: model_points match the default template from "
            "build_jts_config (another subject's maxillary scan), not this session."
        )
        lines.append(
            "  Replace them with this subject's scan landmarks in the same order "
            "as the maxillary (or calibration_type) point list."
        )
    lines.append("")
    smooth = analysis.get("smoothing") or {}
    lines.append(
        "  Smoothing: Savitzky–Golay on T_model_origin_mand_landmark_t "
        f"(enabled={smooth.get('enabled', True)}, "
        f"window={smooth.get('window_length')}, poly={smooth.get('poly_order')})"
    )
    return "\n".join(lines)


_BODY_DS = {"HP": "pose_cra", "MP": "pose_mta", "CT": "pose_ct"}


def _finite_pose_mask(T: np.ndarray) -> np.ndarray:
    p = np.asarray(T)
    if p.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(p).reshape(len(p), -1).all(axis=1)


def _optical_row_frames(f: Any) -> np.ndarray | None:
    """Trigger-grid frame of each /optical row (see ``jtsmm.tools.timegrid``)."""
    from jtsmm.tools.timegrid import frame_grid_index, session_frame_period_us, estimate_period_us

    opt = f["optical"]
    if "t_master_us" not in opt:
        return None
    t = np.asarray(opt["t_master_us"][()], dtype=np.float64)
    n = min([t.shape[0]] + [opt[ds].shape[0] for ds in _BODY_DS.values() if ds in opt])
    t = t[:n]
    if n < 2 or np.any(np.diff(t) <= 0):
        return None
    period = session_frame_period_us(f) or estimate_period_us(t)
    if not period:
        return None
    return frame_grid_index(t, period)


def format_calib_lock_report(session_h5: Path, config: dict[str, Any]) -> str:
    """Per-landmark optical lock (both marker and CT finite) in the config intervals."""
    import h5py

    calib = (config.get("analysis") or {}).get("calibration") or {}
    lines = ["  Optical lock in digitizing windows (both bodies finite):"]
    try:
        with h5py.File(session_h5, "r") as f:
            if "optical" not in f:
                return "  Optical lock: no /optical/ group"
            opt = f["optical"]
            poses: dict[str, np.ndarray] = {}
            for body, ds in _BODY_DS.items():
                if ds in opt:
                    poses[body] = np.asarray(opt[ds][()], dtype=np.float64)
            row_frame = _optical_row_frames(f)
    except OSError as exc:
        return f"  Optical lock: could not read {session_h5} ({exc})"
    if row_frame is not None:
        # Config frames are trigger-grid frames (dropped frames count, as NaN),
        # the same layout jts.aruco_camera builds.
        n_grid = int(row_frame[-1]) + 1
        for body, P in poses.items():
            laid = np.full((n_grid, 4, 4), np.nan)
            laid[row_frame] = P[: row_frame.shape[0]]
            poses[body] = laid

    any_row = False
    for jaw in ("mandibular", "maxillary"):
        block = calib.get(jaw) or {}
        bodies = list(block.get("rigid_bodies") or [])
        if len(bodies) < 2:
            continue
        a, b = bodies[0], bodies[1]
        if a not in poses or b not in poses:
            lines.append(f"    {jaw}: missing {a} or {b} in /optical/")
            continue
        n = min(poses[a].shape[0], poses[b].shape[0])
        both = _finite_pose_mask(poses[a][:n]) & _finite_pose_mask(poses[b][:n])
        for pt in block.get("points") or []:
            name = str(pt.get("name") or "?")
            interval = pt.get("frame_interval") or [0, -1]
            lo, hi = int(interval[0]), int(interval[1])
            hi = min(hi, n - 1)
            lo = max(lo, 0)
            if hi < lo:
                lines.append(f"    {name:14s}  empty interval {interval}")
                continue
            window = both[lo : hi + 1]
            n_win = int(window.size)
            n_ok = int(np.count_nonzero(window))
            pct = 100.0 * n_ok / n_win if n_win else 0.0
            lines.append(
                f"    {name:14s}  {n_ok}/{n_win}  {pct:5.1f}%  {_BODY_DS[a]}+{_BODY_DS[b]}"
            )
            any_row = True
    if not any_row:
        lines.append("    (no calibration intervals)")
    return "\n".join(lines)


def prepare_config(
    config: dict[str, Any],
    *,
    session_h5: Path,
    out_dir: Path,
    append_session: bool,
) -> dict[str, Any]:
    """Point the config at this session, skip JTS plt.show(), choose HDF5 dest."""
    cfg = copy.deepcopy(config)
    src = cfg.setdefault("data_source", {})
    src["type"] = src.get("type") or "aruco_camera"
    src["filename"] = str(session_h5)
    src.setdefault("mode", "offline")

    viz = cfg.setdefault("visualization", {})
    # JTS visualize_results() calls plt.show() even when show_plots is false.
    viz["show_plots"] = False
    viz["raw_data"] = False
    viz["calibration_transforms"] = False
    viz["relative_marker_motion"] = False
    viz["landmark_motion"] = False
    viz["final_trajectory"] = False
    viz["plot_2d_components"] = False
    viz["plot_export_only"] = False

    out = cfg.setdefault("output", {})
    out["directory"] = str(out_dir)
    out["save_hdf5"] = True
    out.setdefault("save_csv", True)
    out["save_plots"] = False
    if append_session:
        out["hdf5_append_group"] = out.get("hdf5_append_group") or "jts"
    else:
        out["hdf5_append_group"] = None
        out.setdefault("hdf5_filename", "jaw_motion.h5")
    return cfg


def _xyz(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=np.float64)[:, :3, 3]


def _t_s(n: int, sample_rate_hz: float) -> np.ndarray:
    hz = float(sample_rate_hz) if sample_rate_hz and sample_rate_hz > 0 else 1.0
    return np.arange(n, dtype=np.float64) / hz


def plot_xyz_trajectories(
    series: list[tuple[str, np.ndarray, str]],
    *,
    sample_rate_hz: float,
    title: str,
    save: Path | None,
    show: bool,
    t_us: np.ndarray | None = None,
) -> Path | None:
    """Plot one or more (N,4,4) trajectories as a 3×1 XYZ stack (millimetres).

    With ``t_us`` (per-row master time, NaN for JTS's synthetic transition
    frames) the x axis is real time: phases sit where they were recorded and
    the synthetic bridges between them are not drawn. Without it the axis is
    ``row / sample_rate_hz``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5), constrained_layout=True)
    timed = t_us is not None and np.isfinite(np.asarray(t_us, dtype=np.float64)).any()
    for name, T, color in series:
        xyz = _xyz(T)
        if timed and len(t_us) == xyz.shape[0]:
            t = np.asarray(t_us, dtype=np.float64)
            t_s = (t - np.nanmin(t)) * 1e-6
            ok = np.isfinite(t_s)
            t_s, xyz_p = break_gaps(t_s[ok], xyz[ok], period=1.0 / sample_rate_hz if sample_rate_hz else None)
        else:
            t_s, xyz_p = _t_s(xyz.shape[0], sample_rate_hz), xyz
        for ax, axis in zip(axes, range(3)):
            ax.plot(t_s, xyz_p[:, axis], color=color, lw=1.2, label=name if ax is axes[0] else None)
    for ax, ylab in zip(axes, _AXIS):
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(
        "t (s) from first exported frame, MCU master" if timed
        else "t (s) from combined experiment start (row / rate)"
    )
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle(title)

    written: Path | None = None
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        written = save
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return written


def _sample_rate(analysis: Any) -> float:
    return float(getattr(analysis.motion_data, "frame_rate", 0.0) or 0.0)


def _frame_times(analysis: Any, sub_name: str | None) -> np.ndarray | None:
    """Per-row master µs of the main trajectory (or a sub-experiment), if known."""
    getter = getattr(analysis, "frame_times_us", None)
    if getter is None:
        return None
    if sub_name is None:
        return getter()
    frames = (getattr(analysis, "sub_experiment_source_frames", None) or {}).get(sub_name)
    return None if frames is None else getter(frames)


def plot_analysis(
    analysis: Any,
    *,
    out_dir: Path,
    stem: str,
    show: bool,
) -> list[Path]:
    """Write PNG(s) for the combined smoothed trajectory and any sub-experiments."""
    written: list[Path] = []
    hz = _sample_rate(analysis)
    traj = analysis.trajectories
    series: list[tuple[str, np.ndarray, str]] = []
    raw = traj.get("T_model_origin_mand_landmark_t")
    smooth = traj.get("T_model_origin_mand_landmark_t_smooth")
    if raw is not None:
        series.append(("raw (model CS)", raw, _RAW_COLOR))
    if smooth is not None:
        series.append(("smoothed (model CS)", smooth, _SMOOTH_COLOR))
    if series:
        path = plot_xyz_trajectories(
            series,
            sample_rate_hz=hz,
            title=f"JTS mandibular landmark in model CS — {stem}",
            save=out_dir / f"{stem}_jts_smoothed.png",
            show=show,
            t_us=_frame_times(analysis, None),
        )
        if path is not None:
            written.append(path)

    subs = getattr(analysis, "sub_experiment_trajectories", None) or {}
    for name, T in subs.items():
        if T is None:
            continue
        path = plot_xyz_trajectories(
            [(f"{name} (HP-marker CS, unsmoothed)", T, _SUB_COLOR)],
            sample_rate_hz=hz,
            title=f"JTS sub-experiment {name} — {stem}",
            save=out_dir / f"{stem}_jts_sub_{name}.png",
            show=show,
            t_us=_frame_times(analysis, name),
        )
        if path is not None:
            written.append(path)
    return written


DEFAULT_MAX_GAP_S = 0.5


def interpolate_se3_gaps(
    T: np.ndarray, *, max_gap_frames: int | None = None
) -> tuple[np.ndarray, int]:
    """Fill NaN/singular holes of at most ``max_gap_frames`` so smoothing can run.

    Translation is interpolated linearly, rotation by Slerp between the poses
    on either side. Longer holes and holes at either end (no pose on one side)
    stay NaN -- a straight line across seconds of lost tracking is invented
    data. ``max_gap_frames=None`` fills every interior hole.
    Returns ``(filled_copy, n_filled_frames)``; ``T`` is not modified.
    """
    from scipy.spatial.transform import Rotation as R
    from scipy.spatial.transform import Slerp

    T = np.asarray(T, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4):
        raise ValueError(f"expected (N,4,4), got {T.shape}")
    good = _valid_pose_mask(T)
    out = T.copy()
    out[~good] = np.nan
    if good.all():
        return out, 0
    if not np.any(good):
        raise ValueError("trajectory has no finite SE(3) samples to interpolate")

    fill = np.zeros(len(T), dtype=bool)
    idx_good = np.flatnonzero(good)
    for a, b in zip(idx_good[:-1], idx_good[1:]):
        n_missing = b - a - 1
        if n_missing and (max_gap_frames is None or n_missing <= max_gap_frames):
            fill[a + 1 : b] = True
    if not np.any(fill):
        return out, 0

    idx = np.arange(len(T), dtype=np.float64)
    where = idx[fill]
    for k in range(3):
        out[fill, k, 3] = np.interp(where, idx[good], T[good, k, 3])
    slerp = Slerp(idx[good], R.from_matrix(T[good, :3, :3]))
    out[fill, :3, :3] = slerp(where).as_matrix()
    out[fill, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return out, int(fill.sum())


def _valid_pose_mask(T: np.ndarray) -> np.ndarray:
    """Finite pose with a non-singular rotation block."""
    finite = _finite_pose_mask(T)
    if np.any(finite):
        dets = np.linalg.det(T[finite, :3, :3])
        ok = np.zeros(len(T), dtype=bool)
        ok[np.flatnonzero(finite)] = np.isfinite(dets) & (np.abs(dets) > 1e-6)
        finite = ok
    return finite


def run_analysis(config: dict[str, Any], *, max_gap_s: float | None = None) -> Any:
    """Run JTS offline; bridge short tracking dropouts for the smoother only.

    Holes up to ``max_gap_s`` (default ``analysis.smoothing.max_gap_s`` or
    0.5 s) are interpolated in a *copy* that feeds Savitzky-Golay. The raw
    trajectory stays as measured (NaN where tracking was lost) and every
    exported group gets a ``gap_filled`` mask next to ``valid``.
    """
    ensure_jts_on_path()
    from jts.core import JawMotionAnalysis

    analysis = JawMotionAnalysis(config)
    analysis.load_data()
    analysis.perform_calibration()
    analysis.compute_relative_motion()
    analysis.transform_to_ref_markers_coordinates()
    analysis.register_to_model_coordinates()
    key = "T_model_origin_mand_landmark_t"
    raw = analysis.trajectories.get(key)
    if raw is not None:
        if max_gap_s is None:
            smooth_cfg = (config.get("analysis") or {}).get("smoothing") or {}
            max_gap_s = float(smooth_cfg.get("max_gap_s", DEFAULT_MAX_GAP_S))
        hz = _sample_rate(analysis)
        cap = int(round(max_gap_s * hz)) if hz > 0 else 0
        filled, n_fill = interpolate_se3_gaps(raw, max_gap_frames=cap)
        gap_filled = ~_valid_pose_mask(raw) & _valid_pose_mask(filled)
        n_left = int(np.count_nonzero(~_valid_pose_mask(filled)))
        if n_fill or n_left:
            print(
                f"smoothing input: filled {n_fill} frames in gaps <= {max_gap_s:g} s "
                f"({cap} frames); {n_left} frames in longer gaps stay NaN"
            )
        analysis.trajectories[key] = filled
        analysis.smooth_trajectory()
        analysis.trajectories[key] = raw
        if analysis.active_trajectory is filled:
            analysis.active_trajectory = raw
        analysis.frame_datasets["gap_filled"] = gap_filled
    else:
        analysis.smooth_trajectory()
    analysis.visualize_results()
    analysis.save_results()
    return analysis


def _result_paths(config: dict[str, Any], session_h5: Path) -> tuple[Path, str]:
    out = config.get("output") or {}
    append = out.get("hdf5_append_group") or None
    if append:
        return session_h5, f"{session_h5}:/{str(append).strip('/')}"
    directory = Path(out.get("directory") or ".")
    return directory / out.get("hdf5_filename", "jaw_motion.h5"), str(
        directory / out.get("hdf5_filename", "jaw_motion.h5")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="session.h5 or a session directory")
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JTS JSON (default: unique *_jts_config.json next to the session)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Directory for HDF5/CSV/PNG")
    ap.add_argument(
        "--append-session",
        action="store_true",
        help="Write trajectories into session.h5:/jts instead of a sidecar jaw_motion.h5",
    )
    ap.add_argument(
        "--landmarks-only",
        action="store_true",
        help="Print how landmarks/teeth are passed to JTS and exit",
    )
    ap.add_argument("--no-show", action="store_true", help="Do not open plot windows")
    ap.add_argument(
        "--max-gap-s",
        type=float,
        default=None,
        help=(
            "Longest tracking hole bridged for smoothing (default: "
            f"analysis.smoothing.max_gap_s or {DEFAULT_MAX_GAP_S}); longer holes stay NaN"
        ),
    )
    args = ap.parse_args(argv)

    try:
        session_h5 = resolve_session_h5(args.session)
        config_path = find_jts_config(session_h5, args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with config_path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    print(f"session: {session_h5}")
    print(f"config:  {config_path}")
    print(format_landmark_report(config))
    print(format_calib_lock_report(session_h5, config))
    if args.landmarks_only:
        return 0

    out_dir = args.out
    if out_dir is None:
        out_dir = session_h5.parent
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = prepare_config(
        config,
        session_h5=session_h5,
        out_dir=out_dir,
        append_session=args.append_session,
    )

    try:
        analysis = run_analysis(cfg, max_gap_s=args.max_gap_s)
    except Exception as exc:  # noqa: BLE001
        print(f"error: JTS analysis failed: {exc}", file=sys.stderr)
        return 1

    _h5, dest = _result_paths(cfg, session_h5)
    traj = analysis.trajectories
    smooth = traj.get("T_model_origin_mand_landmark_t_smooth")
    raw = traj.get("T_model_origin_mand_landmark_t")
    active = getattr(analysis, "active_trajectory_label", None)
    n_smooth = 0 if smooth is None else int(smooth.shape[0])
    n_raw = 0 if raw is None else int(raw.shape[0])
    hz = _sample_rate(analysis)
    print(
        f"trajectories: raw={n_raw}  smoothed={n_smooth}  "
        f"active={active}  {hz:.3g} Hz"
    )
    print(f"HDF5: {dest}")
    csv_name = (cfg.get("output") or {}).get("csv_filename", "jaw_motion")
    if (cfg.get("output") or {}).get("save_csv", True) and active:
        print(f"CSV:  {out_dir / f'{csv_name}_{active}.csv'}")

    plot_analysis(
        analysis,
        out_dir=out_dir,
        stem=session_h5.stem,
        show=not args.no_show,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
