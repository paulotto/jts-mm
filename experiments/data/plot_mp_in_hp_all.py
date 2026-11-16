#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Mandibular-marker origin for every subject, in the dental model orientation.

The point is the MP origin in the cranial marker. Its rest median is removed,
then the same registration that JTS uses (maxillary points onto the model)
rotates the path into the model axes: X left–right, Y opening, Z forward–back.
The camera is the JTS matplotlib view, Z up. A Savitzky–Golay filter
(window 11, order 3) runs on contiguous stretches with two tags on both markers.

One panel per participant (latest motion take), four per row.

Usage
-----
  python plot_mp_in_hp_all.py
  python plot_mp_in_hp_all.py --data-dir ~/jtsmm_data --subject s01,s02,s04 --out /tmp/mp.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from biomech.cohort import discover_sessions  # noqa: E402
from biomech.io.inventory import select_takes, subject_pids  # noqa: E402
from biomech.paths import resolve_data_dir, results_root  # noqa: E402

_PHASES = (
    ("open_close", "Opening–closing", "#1f77b4"),
    ("protrusion_retrusion", "Forward–back", "#ff7f0e"),
    ("left_right", "Left–right", "#2ca02c"),
    ("chewing", "Chewing", "#d62728"),
    ("free_motion", "Free motion", "#7f7f7f"),
)


def default_out() -> Path:
    return results_root() / "data" / "trajectory_quality" / "mp_in_hp_all_subjects.png"


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _smooth(xyz: np.ndarray) -> np.ndarray:
    out = np.array(xyz, copy=True)
    finite = np.isfinite(out).all(axis=1)
    edges = np.diff(np.concatenate(([0], finite.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    for a, b in zip(starts, stops):
        if b - a < 11:
            continue
        for axis in range(3):
            out[a:b, axis] = savgol_filter(out[a:b, axis], 11, 3)
    out[~finite] = np.nan
    return out


def _model_rotation(session_h5: Path) -> np.ndarray:
    """Rotation of ``T_model_origin_max_marker``: cranial-marker vectors → model axes."""
    import json

    from jts.core import JawMotionAnalysis
    from jts.helper import kabsch_algorithm
    from jtsmm.tools.jts_path import ensure_jts_on_path
    from jtsmm.tools.run_jts_analysis import find_jts_config, prepare_config, resolve_session_h5

    session_h5 = resolve_session_h5(session_h5)
    config = json.loads(find_jts_config(session_h5, None).read_text(encoding="utf-8"))
    cfg = prepare_config(config, session_h5=session_h5, out_dir=session_h5.parent, append_session=False)
    out = cfg.setdefault("output", {})
    out["save_hdf5"] = False
    out["save_csv"] = False
    out["save_plots"] = False
    ensure_jts_on_path()
    analysis = JawMotionAnalysis(cfg)
    analysis.load_data()
    analysis.perform_calibration()
    transform = cfg["analysis"]["coordinate_transform"]
    mocap = np.asarray(analysis.calibration_points[transform["calibration_type"]], dtype=np.float64)
    model = np.asarray(transform["model_points"], dtype=np.float64)
    return kabsch_algorithm(mocap, model)[:3, :3]


def _load(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as f:
        opt = f["optical"]
        t = np.asarray(opt["t_master_us"][()], dtype=np.float64)
        cra = np.asarray(opt["pose_cra"][()], dtype=np.float64)
        mta = np.asarray(opt["pose_mta"][()], dtype=np.float64)
        n_hp = np.asarray(opt["quality/cra/n_tags"][()], dtype=np.int16)
        n_mp = np.asarray(opt["quality/mta/n_tags"][()], dtype=np.int16)
        names = [_decode(x) for x in f["phases/name"][()]]
        t0 = np.asarray(f["phases/t_start_us"][()], dtype=np.float64)
        t1 = np.asarray(f["phases/t_end_us"][()], dtype=np.float64)
    n = min(len(t), len(cra), len(mta), len(n_hp), len(n_mp))
    t, cra, mta, n_hp, n_mp = t[:n], cra[:n], mta[:n], n_hp[:n], n_mp[:n]
    delta = mta[:, :3, 3] - cra[:, :3, 3]
    xyz = np.einsum("nji,nj->ni", cra[:, :3, :3], delta)
    ok = (
        np.isfinite(cra[:, :3]).all(axis=(1, 2))
        & np.isfinite(mta[:, :3]).all(axis=(1, 2))
        & (n_hp == 2)
        & (n_mp == 2)
    )
    xyz[~ok] = np.nan
    rest = np.zeros(n, dtype=bool)
    masks = {key: np.zeros(n, dtype=bool) for key, _label, _color in _PHASES}
    for name, a, b in zip(names, t0, t1):
        label = name.strip().lower()
        inside = (t >= a) & (t <= b)
        if label == "rest":
            rest |= inside
        if label in masks:
            masks[label] |= inside
    origin = np.nanmedian(xyz[rest], axis=0)
    if not np.isfinite(origin).all():
        raise RuntimeError(f"no rest samples in {path}")
    xyz = _smooth(np.einsum("ij,nj->ni", _model_rotation(path), xyz - origin))
    out: dict[str, np.ndarray] = {}
    for key, _label, _color in _PHASES:
        piece = np.array(xyz, copy=True)
        piece[~masks[key]] = np.nan
        out[key] = piece
    return out


def _limits(paths: list[dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    chunks = []
    for phases in paths:
        for xyz in phases.values():
            finite = xyz[np.isfinite(xyz).all(axis=1)]
            if finite.size:
                chunks.append(finite)
    cloud = np.concatenate(chunks, axis=0)
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    span = float(np.max(maxs - mins)) * 1.08
    mid = 0.5 * (mins + maxs)
    return mid[0], mid[1], mid[2], span


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--take", choices=("latest", "first"), default="latest")
    ap.add_argument("--out", type=Path, default=None, help="PNG path")
    args = ap.parse_args(argv)
    pids = subject_pids(args.subject)
    motion = [s for s in discover_sessions(resolve_data_dir(args.data_dir))
              if s.kind == "motion" and (pids is None or s.pid in pids)]
    sessions = [(f"{s.pid}_{s.sex}_{s.age}", s.h5) for s in select_takes(motion, args.take)]
    if not sessions:
        print("error: no motion sessions")
        return 1
    out_path = (args.out or default_out()).expanduser()
    loaded = [(subject, _load(path)) for subject, path in sessions]
    cx, cy, cz, span = _limits([phases for _subject, phases in loaded])
    n_rows = max(2, -(-len(loaded) // 4))
    fig = plt.figure(figsize=(16.0, 4.4 * n_rows))
    axes = [fig.add_subplot(n_rows, 4, i + 1, projection="3d") for i in range(len(loaded))]
    for ax, (subject, phases) in zip(axes, loaded):
        for key, _label, color in _PHASES:
            xyz = phases[key]
            if int(np.isfinite(xyz).all(axis=1).sum()) < 2:
                continue
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.05)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_zlim(cz - span / 2, cz + span / 2)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=30, azim=-60)
        ax.set_title(subject, fontsize=11, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        n = sum(int(np.isfinite(xyz).all(axis=1).sum()) for xyz in phases.values())
        print(f"{subject}: n={n}")
    for ax in axes:
        ax.set_xlabel("X [mm]", fontsize=8, labelpad=1)
        ax.set_ylabel("Y [mm]", fontsize=8, labelpad=1)
        ax.set_zlabel("Z [mm]", fontsize=8, labelpad=1)
    handles = [
        plt.Line2D([0], [0], color=color, lw=2.0, label=label) for _key, label, color in _PHASES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        fontsize=10,
    )
    fig.suptitle("MP origin, model axes, rest at the origin", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.86, wspace=0.02, hspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
