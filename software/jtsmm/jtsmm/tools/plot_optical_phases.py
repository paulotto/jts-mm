#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Plot optical trajectories separately for each experiment phase.

Default: mouthpiece relative to headpiece (``T_HP←MP``, same as JTS
``inv(T_world_HP) @ T_world_MP``). One matplotlib figure per phase, 3×1 stack
of X/Y/Z vs time (seconds from phase start). By default the first finite sample
of the phase is subtracted so the curve is jaw *motion*, not the ~20 cm array
offset.

``--world`` overlays HP / MP / CT in the camera world frame instead.

Examples::

    python -m jtsmm.tools.plot_optical_phases path/to/session.h5
    python -m jtsmm.tools.plot_optical_phases session.h5 --out viz/ --no-show
    python -m jtsmm.tools.plot_optical_phases session.h5 --world
    python -m jtsmm.tools.plot_optical_phases session.h5 --absolute --phases open_close
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jtsmm.io.session_layout import session_h5_path
from jtsmm.tools.timegrid import break_gaps
from jtsmm.tools.plot_mp_trajectory import (
    PhaseInterval,
    _SESSION_BODY,
    load_phases,
)

_BODY_ORDER = ("HP", "MP", "CT")
_BODY_COLORS = {
    "HP": "#8e44ad",
    "MP": "#27ae60",
    "CT": "#c0392b",
    "MP in HP": "#1a5276",
}
_AXIS = ("X (mm)", "Y (mm)", "Z (mm)")
_DELTA_AXIS = ("ΔX (mm)", "ΔY (mm)", "ΔZ (mm)")


@dataclass
class OpticalBody:
    name: str
    t_us: np.ndarray
    T: np.ndarray  # (N, 4, 4) T_world←body, millimetres

    @property
    def xyz_mm(self) -> np.ndarray:
        return np.asarray(self.T[:, :3, 3], dtype=np.float64)


def resolve_session_h5(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        candidate = session_h5_path(path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no session.h5 in directory: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", s).strip("._-") or "phase"


def _finite_pose(T: np.ndarray) -> np.ndarray:
    p = np.asarray(T)
    if p.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(p).reshape(len(p), -1).all(axis=1)


def load_optical_bodies(path: Path, *, bodies: list[str] | None = None) -> list[OpticalBody]:
    """Load world-frame SE(3) poses for HP/MP/CT from a multimodal session."""
    wanted = [b.strip().upper() for b in (bodies or list(_BODY_ORDER))]
    out: list[OpticalBody] = []
    with h5py.File(path, "r") as f:
        if "optical" not in f:
            raise KeyError(f"no /optical/ group in {path}")
        og = f["optical"]
        if "t_master_us" not in og:
            raise KeyError(f"no /optical/t_master_us in {path}")
        t_us = np.asarray(og["t_master_us"][()], dtype=np.float64)
        available: list[str] = []
        for body in wanted:
            ds = _SESSION_BODY.get(body)
            if ds is None:
                raise KeyError(f"unknown body {body!r}; use HP/MP/CT (or CRA/MTA)")
            label = {"MTA": "MP", "CRA": "HP"}.get(body, body)
            if ds not in og:
                continue
            T = np.asarray(og[ds][()], dtype=np.float64)
            if T.ndim != 3 or T.shape[1:] != (4, 4):
                raise ValueError(f"/optical/{ds} must be (N,4,4), got {T.shape}")
            n = min(t_us.shape[0], T.shape[0])
            out.append(OpticalBody(name=label, t_us=t_us[:n], T=T[:n]))
            available.append(ds)
    if not out:
        raise KeyError(f"no requested optical poses in {path} (available={available})")
    return out


def relative_mp_in_hp(hp: OpticalBody, mp: OpticalBody) -> OpticalBody:
    """``T_HP←MP = inv(T_world←HP) @ T_world←MP`` (JTS mandible-vs-cranium pose)."""
    n = min(hp.T.shape[0], mp.T.shape[0], hp.t_us.shape[0], mp.t_us.shape[0])
    T_hp = hp.T[:n]
    T_mp = mp.T[:n]
    finite = _finite_pose(T_hp) & _finite_pose(T_mp)
    T_rel = np.full((n, 4, 4), np.nan, dtype=np.float64)
    if np.any(finite):
        R_hp = T_hp[finite, :3, :3]
        t_hp = T_hp[finite, :3, 3]
        R_mp = T_mp[finite, :3, :3]
        t_mp = T_mp[finite, :3, 3]
        # R_rel = R_hp^T R_mp ; t_rel = R_hp^T (t_mp - t_hp)
        T_rel[finite, :3, :3] = np.einsum("nji,njk->nik", R_hp, R_mp)
        T_rel[finite, :3, 3] = np.einsum("nji,nj->ni", R_hp, t_mp - t_hp)
        T_rel[finite, 3, :] = (0.0, 0.0, 0.0, 1.0)
    return OpticalBody(name="MP in HP", t_us=hp.t_us[:n].copy(), T=T_rel)


def slice_phase(
    bodies: list[OpticalBody],
    phase: PhaseInterval,
    *,
    delta: bool = False,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return ``(name, t_s_from_phase_start, xyz_mm)`` inside the phase."""
    t0 = float(phase.t_start_us)
    t1 = float(phase.t_end_us)
    sliced: list[tuple[str, np.ndarray, np.ndarray]] = []
    for body in bodies:
        mask = (body.t_us >= t0) & (body.t_us <= t1) & np.isfinite(body.t_us) & _finite_pose(body.T)
        if not np.any(mask):
            continue
        t_s = (body.t_us[mask] - t0) * 1e-6
        xyz = body.xyz_mm[mask].copy()
        if delta:
            xyz -= xyz[0]
        sliced.append((body.name, t_s, xyz))
    return sliced


def plot_phase(
    sliced: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    title: str,
    ylabel: tuple[str, str, str],
    save: Path | None,
    show: bool,
    period_s: float | None = None,
) -> Path | None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5), constrained_layout=True)
    # Break lines across dropped frames / lost lock (dt > 1.5 frame periods).
    gapped = [(name, *break_gaps(t_s, xyz, period=period_s)) for name, t_s, xyz in sliced]
    for ax, axis, ylab in zip(axes, range(3), ylabel):
        for name, t_s, xyz in gapped:
            ax.plot(
                t_s,
                xyz[:, axis],
                color=_BODY_COLORS.get(name, "#333333"),
                lw=1.1,
                label=name if ax is axes[0] else None,
            )
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("t − phase start (s)")
    axes[0].legend(loc="upper right", fontsize=9, ncol=3, framealpha=0.9)
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


def _by_name(bodies: list[OpticalBody], name: str) -> OpticalBody | None:
    for body in bodies:
        if body.name == name:
            return body
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="session.h5 or a session directory")
    ap.add_argument(
        "--world",
        action="store_true",
        help="Plot HP/MP/CT in world frame instead of MP relative to HP",
    )
    ap.add_argument(
        "--absolute",
        action="store_true",
        help="Do not subtract the first sample (keep the HP-frame array offset)",
    )
    ap.add_argument(
        "--body",
        nargs="+",
        default=None,
        help="With --world: bodies to plot (HP/MP/CT). Ignored for relative plots.",
    )
    ap.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Only these phase names (default: all)",
    )
    ap.add_argument("--out", type=Path, default=None, help="Directory for PNG files")
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive windows")
    args = ap.parse_args(argv)

    try:
        path = resolve_session_h5(args.session)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with h5py.File(path, "r") as f:
        phases = load_phases(f)
        from jtsmm.tools.timegrid import session_frame_period_us

        p_us = session_frame_period_us(f)
    if not phases:
        print(f"error: no phases in {path} (/phases, /events, or sync.report_json)", file=sys.stderr)
        return 1

    if args.phases:
        want = {n.strip().lower() for n in args.phases}
        phases = [p for p in phases if p.name.lower() in want]
        if not phases:
            print(f"error: none of {sorted(want)} found in session phases", file=sys.stderr)
            return 1

    try:
        bodies = load_optical_bodies(path, bodies=args.body if args.world else None)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    delta = not args.absolute
    if args.world:
        plot_bodies = bodies
        tag = "optical"
        kind = "world XYZ"
        ylabel = _AXIS
    else:
        hp = _by_name(bodies, "HP")
        mp = _by_name(bodies, "MP")
        if hp is None or mp is None:
            print("error: relative MP-in-HP needs /optical/pose_cra and pose_mta", file=sys.stderr)
            return 1
        plot_bodies = [relative_mp_in_hp(hp, mp)]
        tag = "mp_in_hp"
        kind = "MP in HP" + (" Δ from phase start" if delta else " absolute")
        ylabel = _DELTA_AXIS if delta else _AXIS

    print(f"{path}: {kind}  bodies={', '.join(b.name for b in plot_bodies)}  phases={len(phases)}")
    out_dir = args.out
    if out_dir is None and args.no_show:
        out_dir = path.parent
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_plotted = 0
    for phase in phases:
        sliced = slice_phase(plot_bodies, phase, delta=delta and not args.world)
        n_fin = sum(xyz.shape[0] for _n, _t, xyz in sliced)
        dur_s = (phase.t_end_us - phase.t_start_us) / 1e6
        print(
            f"  {phase.name}: {dur_s:.2f}s  finite={n_fin}  "
            f"bodies={', '.join(n for n, *_ in sliced) or 'none'}"
        )
        if not sliced:
            continue
        save = None
        if out_dir is not None:
            save = out_dir / f"{path.stem}_{tag}_{_slug(phase.name)}.png"
        plot_phase(
            sliced,
            title=f"{phase.name} ({phase.kind}) — {kind} — {path.name}",
            ylabel=ylabel,
            save=save,
            show=not args.no_show,
            period_s=p_us * 1e-6 if p_us else None,
        )
        n_plotted += 1

    if n_plotted == 0:
        print("error: no finite optical samples in the selected phases", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
