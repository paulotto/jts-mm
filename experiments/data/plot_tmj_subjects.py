#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""One figure per subject: tmj_left on the left, tmj_right on the right.

Each condyle is the digitized CT point, moved by the smoothed mandibular pose
in the dental model frame. The rest median of that condyle is the origin.
Z is up, camera elev 30°, azim −60°.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from biomech.cohort import discover_sessions, parse_session_name  # noqa: E402
from biomech.io.inventory import select_takes, subject_pids  # noqa: E402
from biomech.paths import resolve_data_dir, results_root  # noqa: E402
from jtsmm.tools.plot_mp_trajectory import jts_model_trajectory, load_phases  # noqa: E402

_PHASES = (
    ("open_close", "Opening–closing", "#1f77b4"),
    ("protrusion_retrusion", "Forward–back", "#ff7f0e"),
    ("left_right", "Left–right", "#2ca02c"),
    ("chewing", "Chewing", "#d62728"),
    ("free_motion", "Free motion", "#7f7f7f"),
)
_LANDMARKS = ("tmj_left", "tmj_right")


def default_out_root() -> Path:
    return results_root() / "data"


def _mm(model, name: str) -> np.ndarray:
    xyz = np.asarray(model.points[name], dtype=np.float64)
    if model.unit == "m":
        xyz = xyz * 1000.0
    return xyz


def _load(path: Path) -> dict[str, dict[str, np.ndarray]]:
    import h5py

    model = jts_model_trajectory(path)
    if model is None:
        raise RuntimeError(f"no model trajectory for {path}")
    with h5py.File(path, "r") as f:
        phases = load_phases(f)
    t = np.asarray(model.t_us, dtype=np.float64)
    rest = np.zeros(len(t), dtype=bool)
    masks = {key: np.zeros(len(t), dtype=bool) for key, _label, _color in _PHASES}
    for phase in phases:
        label = phase.name.strip().lower()
        inside = np.isfinite(t) & (t >= float(phase.t_start_us)) & (t <= float(phase.t_end_us))
        if label == "rest":
            rest |= inside
        if label in masks:
            masks[label] |= inside
    out: dict[str, dict[str, np.ndarray]] = {}
    for name in _LANDMARKS:
        if name not in model.points:
            out[name] = {}
            continue
        xyz = _mm(model, name)
        origin = np.nanmedian(xyz[rest], axis=0)
        if not np.isfinite(origin).all():
            origin = np.nanmedian(xyz, axis=0)
        xyz = xyz - origin
        pieces: dict[str, np.ndarray] = {}
        for key, _label, _color in _PHASES:
            piece = np.array(xyz, copy=True)
            piece[~masks[key]] = np.nan
            pieces[key] = piece
        out[name] = pieces
    return out


def _limits(pieces_by_side: dict[str, dict[str, np.ndarray]]) -> tuple[float, float, float, float]:
    chunks = []
    for pieces in pieces_by_side.values():
        for xyz in pieces.values():
            finite = xyz[np.isfinite(xyz).all(axis=1)]
            if finite.size:
                chunks.append(finite)
    cloud = np.concatenate(chunks, axis=0)
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    span = float(np.max(maxs - mins)) * 1.08
    mid = 0.5 * (mins + maxs)
    return mid[0], mid[1], mid[2], span


def _draw(subject: str, sides: dict[str, dict[str, np.ndarray]], dest: Path) -> None:
    cx, cy, cz, span = _limits(sides)
    fig = plt.figure(figsize=(12.5, 6.2))
    for i, landmark in enumerate(_LANDMARKS):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for key, _label, color in _PHASES:
            xyz = sides[landmark].get(key)
            if xyz is None or int(np.isfinite(xyz).all(axis=1).sum()) < 2:
                continue
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, lw=1.15)
        ax.set_xlim(cx - span / 2, cx + span / 2)
        ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_zlim(cz - span / 2, cz + span / 2)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        ax.view_init(elev=30, azim=-60)
        ax.set_title(landmark, fontsize=12, pad=2)
        ax.set_xlabel("X [mm]", fontsize=9, labelpad=2)
        ax.set_ylabel("Y [mm]", fontsize=9, labelpad=2)
        ax.set_zlabel("Z [mm]", fontsize=9, labelpad=2)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color=color, lw=2.0, label=label) for _key, label, color in _PHASES
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"{subject}  ·  model coordinate system, rest at the origin", fontsize=13, y=1.02)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.82, wspace=0.08)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "sessions",
        nargs="*",
        type=Path,
        help="session.h5 or session directory; default is every motion subject, latest take",
    )
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--take", choices=("latest", "first"), default="latest")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="figures go to <out-dir>/<subject>/tmj/ "
                    "(default: $BIOMECH_RESULTS_DIR or ~/experiments, then /data)")
    args = ap.parse_args(argv)
    out_root = (args.out_dir or default_out_root()).expanduser()
    if args.sessions:
        sessions = []
        for raw in args.sessions:
            path = raw.expanduser().resolve()
            h5 = path if path.is_file() else path / "session.h5"
            info = parse_session_name(h5.parent.name)
            if info is None or not h5.is_file():
                print(f"error: no session for {raw}", file=sys.stderr)
                return 1
            sessions.append((f"{info['pid']}_{info['sex']}_{info['age']}", h5))
    else:
        pids = subject_pids(args.subject)
        motion = [s for s in discover_sessions(resolve_data_dir(args.data_dir))
                  if s.kind == "motion" and (pids is None or s.pid in pids)]
        sessions = [(f"{s.pid}_{s.sex}_{s.age}", s.h5) for s in select_takes(motion, args.take)]
    for subject, path in sessions:
        print(f"{subject} …", flush=True)
        sides = _load(path)
        dest = out_root / subject / "tmj" / "tmj_left_right.png"
        _draw(subject, sides, dest)
        for name in _LANDMARKS:
            n = sum(int(np.isfinite(xyz).all(axis=1).sum()) for xyz in sides[name].values())
            print(f"  {name}: n={n}")
        print(f"  Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
