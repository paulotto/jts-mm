#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Evaluate the optical part of a ``jtsmm record`` session.

Prints a text report (lock rates, stereo overlap, working volume) and writes
a PNG with:

- lock timeline (HP / MP / CT vs experiment phases)
- per-phase lock-rate bars
- XY / XZ / YZ scatter of finite body origins (where tracking actually
  worked), with camera positions if the calibration bundle is available

This uses ``/optical``, ``/cameras``, ``/phases``, ``/sync`` and ``/triggers``.
Lock rates are shown against optical rows *and* against the triggers fired in
the same window: dropped frames have no row, so only the latter is the true
rate. Fail *reasons*
(one-view vs RMS) are stored under ``/optical/quality`` when recorded with a
current writer. ``--strict`` applies the same lock-rate / SE(3) / drop budgets
as ``jtsmm integrity check``.

Examples::

    python -m jtsmm.tools.eval_optical_session session.h5
    python -m jtsmm.tools.eval_optical_session session.h5 --save optical_eval.png --no-show
    python -m jtsmm.tools.eval_optical_session session.h5 --no-show --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from jtsmm.tools.plot_mp_trajectory import PhaseInterval, _PHASE_COLORS, load_phases
from jtsmm.tools.timegrid import session_frame_period_us

_BODIES = (
    ("HP", "pose_cra", "#8e44ad"),
    ("MP", "pose_mta", "#27ae60"),
    ("CT", "pose_ct", "#c0392b"),
)
_MOTION = frozenset({"open_close", "left_right", "protrusion_retrusion", "chewing"})
_MAND = frozenset({"mand_point_1", "mand_point_2", "mand_point_3"})
_MAX = frozenset({"max_point_1", "max_point_2", "max_point_3"})


def _dec(v: object) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def _finite_mask(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses)
    if p.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(p).reshape(len(p), -1).all(axis=1)


def _pct(n: int, d: int) -> str:
    if d <= 0:
        return "  n/a"
    return f"{100.0 * n / d:5.1f}%"


def _nan_pose() -> np.ndarray:
    return np.full((4, 4), np.nan, dtype=np.float64)


@dataclass
class OpticalEval:
    path: Path
    duration_s: float
    t_us: np.ndarray
    t_s: np.ndarray
    masks: dict[str, np.ndarray]
    xyz: dict[str, np.ndarray]
    phases: list[PhaseInterval]
    cam_t: dict[str, np.ndarray]
    stereo_t: np.ndarray
    unique_optical: int
    sync: dict
    meta: dict
    cam_xyz: dict[str, np.ndarray] = field(default_factory=dict)
    trig_t: np.ndarray | None = None  # /triggers/t_master_us
    trig_id: np.ndarray | None = None
    period_us: float | None = None  # trigger period (fallback when /triggers is absent)

    @property
    def n_stereo(self) -> int:
        return int(len(self.stereo_t))

    def n_triggers(self, t_lo_us: float, t_hi_us: float) -> int | None:
        """Camera triggers fired in ``[t_lo, t_hi)`` -- the true frame count.

        Optical rows only exist for frames that were captured *and* processed,
        so rows understate the denominator whenever frames are dropped. Lost
        trigger events inside the window are counted from the id gap.
        """
        if self.trig_t is not None and self.trig_t.size:
            sel = (self.trig_t >= t_lo_us) & (self.trig_t < t_hi_us)
            if not np.any(sel):
                return 0
            if self.trig_id is not None and self.trig_id.shape == self.trig_t.shape:
                ids = self.trig_id[sel]
                return int(ids.max() - ids.min() + 1)
            return int(sel.sum())
        if self.period_us:
            return int(np.ceil((t_hi_us - t_lo_us) / self.period_us))
        return None

    def session_window_us(self) -> tuple[float, float]:
        """``[first optical row, last optical row]`` as a half-open window."""
        if not self.t_us.size:
            return 0.0, 0.0
        return float(self.t_us[0]), float(self.t_us[-1]) + 1.0


def _load_cam_xyz(meta: dict) -> dict[str, np.ndarray]:
    """Camera origins in world millimetres, from the bundle named in the session."""
    cfg = meta.get("optical_config")
    if not cfg:
        return {}
    try:
        from jtsmm.config.loader import repo_root
    except Exception:
        repo = Path(__file__).resolve().parents[4]
    else:
        repo = repo_root()
    cfg_path = Path(str(cfg))
    candidates = [cfg_path]
    if not cfg_path.is_absolute():
        candidates.append(repo / cfg_path)
    try:
        from optical.calibration.io import load_cameras_from_bundle, read_calibration_bundle
        from optical.config.loader import load_config_file
        from optical.config.resolve import resolve_optical_doc
    except Exception:
        return {}
    for c in candidates:
        if not c.is_file():
            continue
        try:
            doc = resolve_optical_doc(load_config_file(c), n_cameras=2, quiet=True)
            bundle = Path(str(doc["paths"]["calibration_bundle"]))
            if not bundle.is_absolute():
                bundle = repo / bundle
            if not bundle.is_file():
                continue
            _intr, extr = load_cameras_from_bundle(read_calibration_bundle(str(bundle)))
            return {cid: np.asarray(e.T_world_cam[:3, 3], float) for cid, e in extr.items()}
        except Exception:
            continue
    return {}


def load_eval(path: Path) -> OpticalEval:
    with h5py.File(path, "r") as f:
        if "optical" not in f or "t_master_us" not in f["optical"]:
            raise SystemExit(f"{path}: no /optical/t_master_us")
        og = f["optical"]
        t_us = np.asarray(og["t_master_us"], dtype=np.int64)
        t0 = int(t_us[0]) if t_us.size else 0
        t_s = (t_us - t0) / 1e6
        masks: dict[str, np.ndarray] = {}
        xyz: dict[str, np.ndarray] = {}
        for name, ds, _c in _BODIES:
            if ds not in og:
                masks[name] = np.zeros(len(t_us), dtype=bool)
                xyz[name] = np.full((len(t_us), 3), np.nan)
                continue
            P = np.asarray(og[ds], dtype=np.float64)
            m = _finite_mask(P)
            masks[name] = m
            xyz[name] = np.full((len(t_us), 3), np.nan)
            if m.any():
                xyz[name][m] = P[m, :3, 3]

        phases = load_phases(f)

        cam_t: dict[str, np.ndarray] = {}
        if "cameras" in f:
            for cid in f["cameras"]:
                g = f["cameras"][cid]
                if "t_master_us" in g:
                    cam_t[str(cid)] = np.asarray(g["t_master_us"], dtype=np.int64)

        ids = sorted(cam_t)
        if len(ids) >= 2:
            stereo_t = np.intersect1d(cam_t[ids[0]], cam_t[ids[1]])
        else:
            stereo_t = np.asarray([], dtype=np.int64)

        sync: dict = {}
        if "sync" in f and "report_json" in f["sync"].attrs:
            raw = f["sync"].attrs["report_json"]
            if isinstance(raw, bytes):
                raw = raw.decode()
            try:
                obj = json.loads(str(raw))
                if isinstance(obj, dict):
                    sync = obj
            except json.JSONDecodeError:
                pass

        meta = {str(k): _dec(v) for k, v in f["meta"].attrs.items()} if "meta" in f else {}
        duration_s = float(meta.get("session_duration_s") or 0.0)
        if duration_s <= 0 and t_us.size:
            duration_s = float(t_us[-1] - t_us[0]) / 1e6

        trig_t = trig_id = None
        if "triggers" in f and "t_master_us" in f["triggers"]:
            trig_t = np.asarray(f["triggers"]["t_master_us"], dtype=np.int64)
            if "trigger_id" in f["triggers"]:
                trig_id = np.asarray(f["triggers"]["trigger_id"], dtype=np.int64)
        period_us = session_frame_period_us(f)

    ev = OpticalEval(
        path=path,
        duration_s=duration_s,
        t_us=t_us,
        t_s=t_s,
        masks=masks,
        xyz=xyz,
        phases=phases,
        cam_t=cam_t,
        stereo_t=stereo_t,
        unique_optical=int(len(np.unique(t_us))) if t_us.size else 0,
        sync=sync,
        meta=meta,
        cam_xyz=_load_cam_xyz(meta),
        trig_t=trig_t,
        trig_id=trig_id,
        period_us=period_us,
    )
    return ev


def _rate(ev: OpticalEval, body: str, sel: np.ndarray | None = None) -> tuple[int, int]:
    m = ev.masks[body]
    if sel is not None:
        m = m & sel
        d = int(sel.sum())
    else:
        d = len(m)
    return int(m.sum()), d


def _trig_rate(ev: OpticalEval, body: str, t_lo: float, t_hi: float) -> tuple[int, int | None]:
    """Finite frames of ``body`` / triggers fired in ``[t_lo, t_hi)``."""
    sel = (ev.t_us >= t_lo) & (ev.t_us < t_hi)
    return int((ev.masks[body] & sel).sum()), ev.n_triggers(t_lo, t_hi)


def _sel_names(ev: OpticalEval, names: frozenset[str]) -> np.ndarray:
    sel = np.zeros(len(ev.t_us), dtype=bool)
    for ph in ev.phases:
        if ph.name in names:
            sel |= (ev.t_us >= ph.t_start_us) & (ev.t_us < ph.t_end_us)
    return sel


def _pts(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pts = xyz[mask]
    if pts.size == 0:
        return pts.reshape(0, 3)
    return pts[np.isfinite(pts).all(axis=1)]


def _volume(xyz: np.ndarray, mask: np.ndarray) -> str:
    pts = _pts(xyz, mask)
    if len(pts) < 2:
        if len(pts) == 1:
            p = pts[0]
            return f"n=1  [{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}] mm"
        return "n=0"
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = hi - lo
    mid = pts.mean(axis=0)
    return (
        f"n={len(pts)}  "
        f"x[{lo[0]:.0f},{hi[0]:.0f}]  y[{lo[1]:.0f},{hi[1]:.0f}]  "
        f"z[{lo[2]:.0f},{hi[2]:.0f}] mm  "
        f"span {span[0]:.0f}×{span[1]:.0f}×{span[2]:.0f} mm  "
        f"centroid [{mid[0]:.0f}, {mid[1]:.0f}, {mid[2]:.0f}]"
    )


def format_report(ev: OpticalEval) -> str:
    lines: list[str] = []
    w = 78
    lines.append("=" * w)
    lines.append(f" optical evaluation  {ev.path}")
    lines.append("=" * w)
    lines.append(
        f" duration {ev.duration_s:.1f}s   pose_method={ev.meta.get('optical_pose_method')}  "
        f"cams={ev.meta.get('optical_cam_ids')}"
    )
    tip = ev.meta.get("ct_tip_body_mm")
    if tip:
        lines.append(
            f" CT tip {tip}  rms={ev.meta.get('ct_tip_rms_mm')}  src={ev.meta.get('ct_tip_source')}"
        )
    opt = ev.sync.get("optical") if isinstance(ev.sync.get("optical"), dict) else {}
    cam_bits = "  ".join(f"{k}={len(v)}" for k, v in sorted(ev.cam_t.items()))
    lines.append(f" cameras: {cam_bits}  stereo_ts={ev.n_stereo}  drops={ev.sync.get('drops')}")
    lines.append(
        f" optical rows={len(ev.t_us)}  unique_t={ev.unique_optical}  "
        f"enqueued={opt.get('enqueued')}  processed={opt.get('processed')}  "
        f"queue_dropped={opt.get('queue_dropped')}  pair_dropped={opt.get('pair_dropped')}"
    )
    relock = ev.sync.get("validation_relock_count")
    vfail = ev.sync.get("validation_fail_count")
    if relock or vfail:
        lines.append(f" aligner relock={relock}  val_fail={vfail}")
    lines.append("")
    lo, hi = ev.session_window_us()
    n_trig = ev.n_triggers(lo, hi)
    try:
        track_stride = int(ev.meta.get("optical_track_stride") or 1)
    except (TypeError, ValueError):
        track_stride = 1
    if track_stride > 1:
        hz = ev.meta.get("optical_track_hz_effective") or ev.meta.get("optical_track_hz") or "?"
        lines.append(
            f" live tracking stride {track_stride} ({hz} Hz); "
            "camera timestamps stay at the trigger rate, and /optical only has detected poses"
        )
    if n_trig is not None:
        lines.append(
            f"Lock rate  (finite 4×4 / optical rows | / {n_trig} triggers from first to last row)"
        )
        if track_stride > 1:
            lines.append(
                "  rows exist only for pairs that were detected; "
                "the trigger ratio counts the pairs live tracking skipped"
            )
        else:
            lines.append(
                "  rows exist only for frames that were captured and processed; "
                "the trigger ratio is the true lock rate"
            )
    else:
        lines.append("Lock rate  (finite 4×4 / optical rows; no /triggers -- true rate unknown)")
    for body, _ds, _c in _BODIES:
        k, d = _rate(ev, body)
        cell = f"  {body:3s}  {k:5d}/{d:<5d}  {_pct(k, d)}"
        if n_trig is not None:
            cell += f"   |  {k:5d}/{n_trig:<5d}  {_pct(k, n_trig)}"
        lines.append(cell)
    if ev.unique_optical and ev.unique_optical != len(ev.t_us):
        lines.append("  (duplicate timestamps inflate the denominator; unique times below)")
        for body, _ds, _c in _BODIES:
            t_ok = ev.t_us[ev.masks[body]]
            n_u = int(len(np.unique(t_ok))) if t_ok.size else 0
            lines.append(
                f"  {body:3s} unique finite times {n_u}/{ev.unique_optical}  "
                f"{_pct(n_u, ev.unique_optical)}"
            )
    if ev.n_stereo:
        lines.append("  among stereo-timestamp frames (true triangulation opportunities)")
        for body, _ds, _c in _BODIES:
            t_ok = np.unique(ev.t_us[ev.masks[body]]) if ev.masks[body].any() else np.array([], np.int64)
            n_in = int(len(np.intersect1d(t_ok, ev.stereo_t)))
            lines.append(f"  {body:3s}  {n_in:5d}/{ev.n_stereo:<5d}  {_pct(n_in, ev.n_stereo)}")

    if ev.phases:
        lines.append("")
        lines.append("per phase: lock % of optical rows (/ of triggers)")
        lines.append(
            f"{'phase':28s} {'dur':>6} {'rows':>6} {'trig':>6} "
            f"{'HP':>15} {'MP':>15} {'CT':>15}"
        )
        for ph in ev.phases:
            sel = (ev.t_us >= ph.t_start_us) & (ev.t_us < ph.t_end_us)
            dur = (ph.t_end_us - ph.t_start_us) / 1e6
            n_tr = ev.n_triggers(ph.t_start_us, ph.t_end_us)
            cells = []
            for body, _ds, _c in _BODIES:
                k, d = _rate(ev, body, sel)
                cell = _pct(k, d) if d else "   n/a"
                if n_tr:
                    cell += f" ({_pct(k, n_tr).strip()})"
                cells.append(cell)
            tr = f"{n_tr:6d}" if n_tr is not None else "     -"
            lines.append(
                f"{ph.name:28s} {dur:5.1f}s {int(sel.sum()):6d} {tr} "
                + "".join(f"{c:>16}" for c in cells)
            )
        for label, names in (
            ("landmarks mand_*", _MAND),
            ("landmarks max_*", _MAX),
            ("motion", _MOTION),
        ):
            sel = _sel_names(ev, names)
            if not sel.any():
                continue
            trig = [ev.n_triggers(p.t_start_us, p.t_end_us) for p in ev.phases if p.name in names]
            n_tr = sum(trig) if trig and all(x is not None for x in trig) else None
            bits = []
            for body, _ds, _c in _BODIES:
                k, d = _rate(ev, body, sel)
                bit = f"{body} {_pct(k, d)}"
                if n_tr:
                    bit += f" ({_pct(k, n_tr).strip()})"
                bits.append(bit)
            lines.append(f"  {label:22s} " + "  ".join(bits))

    lines.append("")
    lines.append("Where finite poses sit (body origin, world mm)")
    for body, _ds, _c in _BODIES:
        lines.append(f"  {body}: {_volume(ev.xyz[body], ev.masks[body])}")
    if ev.cam_xyz:
        for cid, p in sorted(ev.cam_xyz.items()):
            lines.append(f"  {cid} origin: [{p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}] mm")
    else:
        lines.append("  cameras: (calibration bundle not found — origins omitted)")

    lines.append("=" * w)
    return "\n".join(lines)


def plot_eval(ev: OpticalEval, *, save: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig = plt.figure(figsize=(12.5, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.15])
    ax_tl = fig.add_subplot(gs[0, :])
    ax_bar = fig.add_subplot(gs[1, 0])
    ax_xy = fig.add_subplot(gs[1, 1])
    ax_xz = fig.add_subplot(gs[2, 0])
    ax_yz = fig.add_subplot(gs[2, 1])

    t = ev.t_s
    tmax = float(t[-1]) if t.size else ev.duration_s
    t0_us = int(ev.t_us[0]) if ev.t_us.size else 0
    phase_handles: list[Patch] = []
    for i, ph in enumerate(ev.phases):
        t0 = (ph.t_start_us - t0_us) / 1e6
        t1 = (ph.t_end_us - t0_us) / 1e6
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
        ax_tl.axvspan(t0, t1, color=color, alpha=0.45, lw=0, zorder=0)
        phase_handles.append(Patch(facecolor=color, alpha=0.45, label=ph.name))

    for i, (body, _ds, color) in enumerate(_BODIES):
        m = ev.masks[body]
        ax_tl.scatter(t[m], np.full(int(m.sum()), 2 - i), s=4, c=color, linewidths=0, zorder=3, label=f"{body} lock")
    ax_tl.set_yticks([0, 1, 2])
    ax_tl.set_yticklabels(["CT", "MP", "HP"])
    ax_tl.set_xlim(0, max(tmax, 0.1))
    ax_tl.set_ylim(-0.45, 2.55)
    ax_tl.set_xlabel("time since first optical row (s)")
    ax_tl.set_title("When each body actually locked (dot = finite pose)")
    ax_tl.grid(True, axis="x", alpha=0.25)
    body_leg = ax_tl.legend(loc="upper right", fontsize=8, markerscale=3)
    ax_tl.add_artist(body_leg)
    if phase_handles:
        ax_tl.legend(
            handles=phase_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=min(len(phase_handles), 5),
            fontsize=7,
            frameon=False,
        )

    names = [ph.name for ph in ev.phases] or ["whole session"]
    windows = [(ph.t_start_us, ph.t_end_us) for ph in ev.phases] or [ev.session_window_us()]
    by_trigger = all(ev.n_triggers(lo, hi) for lo, hi in windows)
    x = np.arange(len(names))
    width = 0.25
    for j, (body, _ds, color) in enumerate(_BODIES):
        vals = []
        for lo, hi in windows:
            if by_trigger:
                k, d = _trig_rate(ev, body, lo, hi)
            else:
                sel = (ev.t_us >= lo) & (ev.t_us < hi)
                k, d = _rate(ev, body, sel)
            vals.append(100.0 * k / d if d else 0.0)
        ax_bar.bar(x + (j - 1) * width, vals, width, color=color, label=body)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("lock % of triggers" if by_trigger else "lock % of optical rows")
    ax_bar.set_ylim(0, 100)
    ax_bar.set_title("Lock rate by phase")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.3)

    def _scatter(ax, ix, iy, xlabel, ylabel, title):
        for body, _ds, color in _BODIES:
            pts = _pts(ev.xyz[body], ev.masks[body])
            if len(pts) == 0:
                continue
            ax.scatter(pts[:, ix], pts[:, iy], s=8, c=color, alpha=0.55, linewidths=0, label=body)
        for cid, p in sorted(ev.cam_xyz.items()):
            ax.scatter([p[ix]], [p[iy]], marker="^", s=70, c="k", zorder=4)
            ax.annotate(cid, (p[ix], p[iy]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, markerscale=1.4)

    _scatter(ax_xy, 0, 1, "X (mm)", "Y (mm)", "Finite body origins — XY (world)")
    _scatter(ax_xz, 0, 2, "X (mm)", "Z (mm)", "Finite body origins — XZ")
    _scatter(ax_yz, 1, 2, "Y (mm)", "Z (mm)", "Finite body origins — YZ")

    fig.suptitle(f"Optical session  {ev.path.name}   {ev.duration_s:.0f}s", fontsize=13)
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        print(f"Wrote {save}", file=sys.stderr)
    if show:
        plt.show()
    else:
        plt.close(fig)


def write_synthetic_session(
    path: Path,
    *,
    n: int = 200,
    dt_us: int = 16_667,
    drop: tuple[int, ...] | list[int] = (),
) -> Path:
    """Minimal session.h5 for tests / demos (not a full multimodal recording).

    ``n`` trigger periods of ``dt_us`` are written to ``/triggers``; frames
    listed in ``drop`` are missing from ``/optical`` and ``/cameras`` (absent
    rows, like a real dropped camera frame).
    """
    path = Path(path)
    t_trig = (np.arange(n, dtype=np.int64) + 1) * dt_us
    t = t_trig
    nan_pose = np.tile(_nan_pose(), (n, 1, 1))
    mp = nan_pose.copy()
    hp = nan_pose.copy()
    ct = nan_pose.copy()
    eye = np.eye(4, dtype=np.float64)

    def _lock(poses: np.ndarray, sl: slice, xyz: np.ndarray) -> None:
        poses[sl] = eye
        poses[sl, :3, 3] = xyz

    # Phase windows in sample indices.
    # mand_point_1: MP+CT lock in a tight blob (working volume).
    _lock(mp, slice(0, 40), np.array([80.0, 20.0, 450.0]))
    _lock(ct, slice(0, 40), np.array([90.0, 10.0, 440.0]))
    # max_point_1: HP almost never (aiming problem).
    _lock(hp, slice(50, 52), np.array([40.0, 80.0, 400.0]))
    # open_close: MP rare.
    _lock(mp, slice(90, 95), np.array([82.0, 22.0, 452.0]))

    phases = [
        ("mand_point_1", 0, 40),
        ("mand_point_2", 40, 45),
        ("mand_point_3", 45, 50),
        ("max_point_1", 50, 70),
        ("max_point_2", 70, 75),
        ("max_point_3", 75, 80),
        ("open_close", 80, 140),
        ("left_right", 140, 160),
        ("protrusion_retrusion", 160, 180),
        ("chewing", 180, 200),
    ]

    keep = np.setdiff1d(np.arange(n), np.asarray(drop, dtype=np.int64))
    t, mp, hp, ct = t[keep], mp[keep], hp[keep], ct[keep]
    n_rows = int(keep.size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["session_duration_s"] = float(n * dt_us) / 1e6
        meta.attrs["optical_tracking"] = "true"
        meta.attrs["optical_pose_method"] = "triangulation"
        meta.attrs["optical_cam_ids"] = "cam0,cam1"
        og = f.create_group("optical")
        og.create_dataset("t_master_us", data=t)
        og.create_dataset("pose_mta", data=mp)
        og.create_dataset("pose_cra", data=hp)
        og.create_dataset("pose_ct", data=ct)
        qg = og.create_group("quality")
        qg.attrs["rms_kind"] = "kabsch_mm"
        for slot, poses in (("mta", mp), ("cra", hp), ("ct", ct)):
            locked = np.isfinite(poses).reshape(n_rows, -1).all(axis=1)
            sg = qg.create_group(slot)
            sg.create_dataset("n_tags", data=np.where(locked, 2, 0).astype(np.uint8))
            sg.create_dataset("n_cameras", data=np.where(locked, 2, 0).astype(np.uint8))
            sg.create_dataset(
                "reprojection_rms",
                data=np.where(locked, 0.4, np.nan).astype(np.float32),
            )
        cams = f.create_group("cameras")
        for cid in ("cam0", "cam1"):
            g = cams.create_group(cid)
            g.create_dataset("t_master_us", data=t)
        pg = f.create_group("phases")
        pg.create_dataset("name", data=np.array([p[0] for p in phases], dtype="S"))
        pg.create_dataset("kind", data=np.array(["experiment"] * len(phases), dtype="S"))
        pg.create_dataset(
            "t_start_us", data=np.array([t_trig[a] for _, a, _ in phases], dtype=np.int64)
        )
        pg.create_dataset(
            "t_end_us",
            data=np.array([int(t_trig[b - 1] + dt_us) for _, _, b in phases], dtype=np.int64),
        )
        tg = f.create_group("triggers")
        tg.create_dataset("t_master_us", data=t_trig.astype(np.uint64))
        tg.create_dataset("trigger_id", data=np.arange(1, n + 1, dtype=np.uint64))
        tg.attrs["rate_hz"] = int(round(1e6 / dt_us))
        sync = f.create_group("sync")
        sync.attrs["report_json"] = json.dumps(
            {
                "drops": {"cam0": 3, "cam1": 2},
                "optical": {
                    "enqueued": n_rows,
                    "processed": n_rows,
                    "written": n_rows,
                    "queue_dropped": 0,
                },
            }
        )
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="session.h5")
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="PNG path (default: <session_dir>/optical_eval.png)",
    )
    ap.add_argument("--no-plot", action="store_true", help="Text report only")
    ap.add_argument("--no-show", action="store_true", help="Do not open a window")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail (exit 2) on optical integrity errors; also fail on warnings",
    )
    args = ap.parse_args(argv)

    path = args.session
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    ev = load_eval(path)
    report = format_report(ev)
    print(report)
    save = args.save
    if save is None:
        save = path.parent / "optical_eval.png"
    txt = save.with_suffix(".txt") if save.suffix.lower() in {".png", ".pdf", ".svg"} else Path(str(save) + ".txt")
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {txt}", file=sys.stderr)

    strict_rc = 0
    if args.strict:
        import h5py
        from jtsmm.integrity.checker import IntegrityReport
        from jtsmm.integrity.optical import check_optical

        irep = IntegrityReport(path=path)
        with h5py.File(path, "r") as f:
            check_optical(f, irep, tracked=True)
        if irep.issues:
            print(f"OPTICAL INTEGRITY ERRORS ({len(irep.issues)})", file=sys.stderr)
            for err in irep.issues:
                print(f"  - {err}", file=sys.stderr)
        if irep.warnings:
            print(f"OPTICAL INTEGRITY WARNINGS ({len(irep.warnings)})", file=sys.stderr)
            for w in irep.warnings:
                print(f"  - {w}", file=sys.stderr)
        if not irep.ok or irep.warnings:
            strict_rc = 2

    if not args.no_plot:
        try:
            plot_eval(ev, save=save, show=not args.no_show)
        except ImportError as exc:
            print(f"skipping plot ({exc})", file=sys.stderr)
    return strict_rc


if __name__ == "__main__":
    raise SystemExit(main())
