# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Optical ``/optical`` integrity: lock rates, SE(3), jumps, drop budgets."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

_BODY_DS = (("HP", "pose_cra"), ("MP", "pose_mta"), ("CT", "pose_ct"))
_MOTION = frozenset({"open_close", "left_right", "protrusion_retrusion", "chewing"})
_PHASE_LOCK_MIN = {
    "mand": {"MP": 0.50, "CT": 0.50},
    "max": {"HP": 0.50, "CT": 0.50},
    "motion": {"HP": 0.40, "MP": 0.40},
}
_PHASE_MIN_ROWS = 10
# Among stereo-timestamp frames: both HP and MP below this → error.
_LOCK_STEREO_ERR = 0.05
_LOCK_STEREO_WARN = 0.30
_PAIR_DROP_WARN = 0.25
_PAIR_DROP_ERR = 0.50
_QUEUE_DROP_WARN = 0.10
_QUEUE_DROP_ERR = 0.25
_FAILED_ERR = 10
_JUMP_WARN_MM = 100.0
_JUMP_ERR_MM = 250.0
_HP_MP_DIST_WARN_LO = 20.0
_HP_MP_DIST_WARN_HI = 500.0
_HP_MP_DIST_ERR_HI = 800.0
_ORIGIN_WARN_MM = 2000.0
_SE3_ATOL = 1e-2
_OPTICAL_VS_STEREO_WARN = 0.50
_OPTICAL_VS_STEREO_ERR = 0.20


def _truthy(raw: Any) -> bool:
    if raw is None:
        return False
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return bool(raw)
    return str(raw).strip().lower() in ("1", "true", "yes")


def _finite_mask(poses: np.ndarray) -> np.ndarray:
    p = np.asarray(poses)
    if p.size == 0:
        return np.zeros((0,), dtype=bool)
    return np.isfinite(p).reshape(len(p), -1).all(axis=1)


def _is_rigid_se3(t: np.ndarray, *, atol: float = _SE3_ATOL) -> bool:
    if t.shape != (4, 4) or not np.all(np.isfinite(t)):
        return False
    t = np.asarray(t, dtype=np.float64)
    if not np.allclose(t[3, :], (0.0, 0.0, 0.0, 1.0), atol=atol):
        return False
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=atol):
        return False
    return abs(float(np.linalg.det(r)) - 1.0) <= max(atol * 10.0, 1e-2)


def _phase_kind(name: str) -> str | None:
    if name.startswith("mand_"):
        return "mand"
    if name.startswith("max_"):
        return "max"
    if name in _MOTION:
        return "motion"
    return None


def _load_phases(f: Any) -> list[tuple[str, int, int]]:
    if "phases" not in f:
        return []
    g = f["phases"]
    if "name" not in g or "t_start_us" not in g or "t_end_us" not in g:
        return []
    names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in g["name"][()]]
    t0 = np.asarray(g["t_start_us"], dtype=np.int64)
    t1 = np.asarray(g["t_end_us"], dtype=np.int64)
    out = []
    for i, name in enumerate(names):
        if i < t0.size and i < t1.size:
            out.append((name, int(t0[i]), int(t1[i])))
    return out


def _meta_int(meta: Any, key: str, default: int = 1) -> int:
    if meta is None:
        return default
    raw = meta.attrs.get(key, default)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return default
    return n if n >= 1 else default


def _scale_to_tracked(frac: float, stride: int) -> float:
    """Map a fraction of every stereo frame onto the live-tracked subset.

    A stride of 3 keeps one pair in three, so 33 % of stereo timestamps is a
    full live track, not a drop. Stride 1 leaves the fraction unchanged.
    """
    if stride <= 1:
        return float(frac)
    return min(1.0, float(frac) * int(stride))


def _lock_frac(mask: np.ndarray, sel: np.ndarray | None = None) -> tuple[int, int, float]:
    m = np.asarray(mask, dtype=bool)
    if sel is not None:
        m = m & sel
        d = int(np.asarray(sel).sum())
    else:
        d = int(m.size)
    k = int(m.sum())
    return k, d, (k / d if d else 0.0)


def check_optical(f: Any, report: Any, *, tracked: bool | None = None) -> None:
    """Validate ``/optical`` when tracking was on or pose datasets are present."""
    meta = f.get("meta")
    if tracked is None:
        tracked = bool(meta is not None and _truthy(meta.attrs.get("optical_tracking")))

    og = f.get("optical")
    has_rows = og is not None and "t_master_us" in og and int(og["t_master_us"].shape[0]) > 0
    if tracked and not has_rows:
        report.add_error("optical tracking enabled but /optical/t_master_us is missing or empty")
        return
    if not has_rows:
        return

    t_us = np.asarray(og["t_master_us"], dtype=np.int64)
    n = int(t_us.size)
    if n >= 2:
        dt = np.diff(t_us)
        if np.any(dt < 0):
            report.add_error("optical t_master_us not non-decreasing")
        n_dup = int(n - len(np.unique(t_us)))
        if n_dup > 0:
            report.add_warn(f"optical t_master_us has {n_dup} duplicate timestamps")

    bodies: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray] = {}
    xyz: dict[str, np.ndarray] = {}
    for body, ds in _BODY_DS:
        if ds not in og:
            report.add_error(f"missing /optical/{ds}")
            masks[body] = np.zeros(n, dtype=bool)
            xyz[body] = np.full((n, 3), np.nan)
            continue
        poses = np.asarray(og[ds], dtype=np.float64)
        if poses.shape != (n, 4, 4):
            report.add_error(f"/optical/{ds} shape {poses.shape} expected ({n}, 4, 4)")
            masks[body] = np.zeros(n, dtype=bool)
            xyz[body] = np.full((n, 3), np.nan)
            continue
        mask = _finite_mask(poses)
        masks[body] = mask
        xyz[body] = np.full((n, 3), np.nan)
        if mask.any():
            xyz[body][mask] = poses[mask, :3, 3]
        bad_se3 = 0
        for i in np.flatnonzero(mask):
            if not _is_rigid_se3(poses[int(i)]):
                bad_se3 += 1
        if bad_se3:
            report.add_error(f"optical {body}: {bad_se3} finite poses are not rigid SE(3)")
        k, d, frac = _lock_frac(mask)
        bodies[body] = {"finite": k, "rows": d, "lock_frac": round(frac, 4)}

        if mask.any():
            origins = xyz[body][mask]
            far = origins[np.linalg.norm(origins, axis=1) > _ORIGIN_WARN_MM]
            if len(far):
                report.add_warn(
                    f"optical {body}: {len(far)} origins farther than {_ORIGIN_WARN_MM:.0f} mm from world origin"
                )

        idx = np.flatnonzero(mask)
        if idx.size >= 2:
            jumps = np.linalg.norm(xyz[body][idx[1:]] - xyz[body][idx[:-1]], axis=1)
            max_jump = float(np.max(jumps)) if jumps.size else 0.0
            bodies[body]["max_jump_mm"] = round(max_jump, 2)
            n_err = int(np.sum(jumps >= _JUMP_ERR_MM))
            n_warn = int(np.sum((jumps >= _JUMP_WARN_MM) & (jumps < _JUMP_ERR_MM)))
            if n_err:
                report.add_error(
                    f"optical {body}: {n_err} inter-frame jumps ≥ {_JUMP_ERR_MM:.0f} mm "
                    f"(max {max_jump:.1f} mm)"
                )
            elif n_warn:
                report.add_warn(
                    f"optical {body}: {n_warn} inter-frame jumps ≥ {_JUMP_WARN_MM:.0f} mm "
                    f"(max {max_jump:.1f} mm)"
                )

    both = masks.get("HP", np.zeros(n, dtype=bool)) & masks.get("MP", np.zeros(n, dtype=bool))
    if both.any():
        dist = np.linalg.norm(xyz["HP"][both] - xyz["MP"][both], axis=1)
        lo, hi = float(np.min(dist)), float(np.max(dist))
        bodies.setdefault("HP_MP", {})["dist_mm"] = {"min": round(lo, 1), "max": round(hi, 1)}
        n_err = int(np.sum(dist > _HP_MP_DIST_ERR_HI))
        n_warn = int(np.sum((dist < _HP_MP_DIST_WARN_LO) | (dist > _HP_MP_DIST_WARN_HI)))
        if n_err:
            report.add_error(
                f"optical HP–MP distance: {n_err} frames > {_HP_MP_DIST_ERR_HI:.0f} mm "
                f"(range {lo:.0f}–{hi:.0f} mm)"
            )
        elif n_warn:
            report.add_warn(
                f"optical HP–MP distance outside {_HP_MP_DIST_WARN_LO:.0f}–{_HP_MP_DIST_WARN_HI:.0f} mm "
                f"({n_warn} frames; range {lo:.0f}–{hi:.0f} mm)"
            )

    qg = og.get("quality") if og is not None else None
    if qg is None:
        report.add_warn("optical quality group missing (legacy session; fail reasons not stored)")
    else:
        for slot in ("mta", "cra", "ct"):
            if slot not in qg:
                report.add_warn(f"optical quality/{slot} missing")
                continue
            sg = qg[slot]
            for name in ("n_tags", "n_cameras", "reprojection_rms"):
                if name not in sg:
                    report.add_warn(f"optical quality/{slot}/{name} missing")
                    continue
                if int(sg[name].shape[0]) != n:
                    report.add_error(
                        f"optical quality/{slot}/{name} length {sg[name].shape[0]} != {n}"
                    )

    cam_t: dict[str, np.ndarray] = {}
    if "cameras" in f:
        for cid in f["cameras"]:
            g = f["cameras"][cid]
            if "t_master_us" in g:
                cam_t[str(cid)] = np.asarray(g["t_master_us"], dtype=np.int64)
    ids = sorted(cam_t)
    stereo_t = np.intersect1d(cam_t[ids[0]], cam_t[ids[1]]) if len(ids) >= 2 else np.asarray([], dtype=np.int64)
    n_stereo = int(stereo_t.size)
    unique_opt = int(len(np.unique(t_us))) if n else 0
    stride = _meta_int(meta, "optical_track_stride", 1)
    if n_stereo >= 20:
        overlap = int(len(np.intersect1d(np.unique(t_us), stereo_t)))
        cov = overlap / n_stereo
        tracked = _scale_to_tracked(cov, stride)
        stride_note = (
            f" (live stride {stride}: {tracked * 100:.1f}% of that rate)"
            if stride > 1 else ""
        )
        if tracked < _OPTICAL_VS_STEREO_ERR:
            report.add_error(
                f"optical rows cover {overlap}/{n_stereo} stereo camera timestamps "
                f"({cov * 100:.1f}% < {_OPTICAL_VS_STEREO_ERR * 100:.0f}%)"
                f"{stride_note}"
            )
        elif tracked < _OPTICAL_VS_STEREO_WARN:
            report.add_warn(
                f"optical rows cover {overlap}/{n_stereo} stereo camera timestamps "
                f"({cov * 100:.1f}%){stride_note}"
            )
        if n_stereo:
            for body, _ds in _BODY_DS[:2]:
                t_ok = np.unique(t_us[masks[body]]) if masks[body].any() else np.array([], dtype=np.int64)
                k = int(len(np.intersect1d(t_ok, stereo_t)))
                frac = _scale_to_tracked(k / n_stereo, stride)
                # setdefault: a body whose pose dataset is missing has no entry.
                bodies.setdefault(body, {})["stereo_lock_frac"] = round(frac, 4)
            hp = bodies.get("HP", {}).get("stereo_lock_frac", 0.0)
            mp = bodies.get("MP", {}).get("stereo_lock_frac", 0.0)
            if hp < _LOCK_STEREO_ERR and mp < _LOCK_STEREO_ERR:
                report.add_error(
                    f"optical stereo lock HP {hp * 100:.1f}% and MP {mp * 100:.1f}% "
                    f"(both < {_LOCK_STEREO_ERR * 100:.0f}%)"
                )
            else:
                if hp < _LOCK_STEREO_WARN:
                    report.add_warn(f"optical stereo lock HP {hp * 100:.1f}% (< {_LOCK_STEREO_WARN * 100:.0f}%)")
                if mp < _LOCK_STEREO_WARN:
                    report.add_warn(f"optical stereo lock MP {mp * 100:.1f}% (< {_LOCK_STEREO_WARN * 100:.0f}%)")

    for name, t0, t1 in _load_phases(f):
        kind = _phase_kind(name)
        if kind is None:
            continue
        sel = (t_us >= t0) & (t_us < t1)
        if int(sel.sum()) < _PHASE_MIN_ROWS:
            continue
        mins = _PHASE_LOCK_MIN[kind]
        for body, min_frac in mins.items():
            _k, d, frac = _lock_frac(masks.get(body, np.zeros(n, dtype=bool)), sel)
            if d < _PHASE_MIN_ROWS:
                continue
            if frac < min_frac:
                report.add_error(
                    f"optical phase {name}: {body} lock {frac * 100:.1f}% "
                    f"({_k}/{d}) < {min_frac * 100:.0f}%"
                )

    sync_raw = {}
    if "sync" in f and "report_json" in f["sync"].attrs:
        raw = f["sync"].attrs["report_json"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            obj = json.loads(str(raw))
            if isinstance(obj, dict):
                sync_raw = obj
        except json.JSONDecodeError:
            sync_raw = {}
    opt_stats = sync_raw.get("optical") if isinstance(sync_raw.get("optical"), dict) else {}
    paired = int(opt_stats.get("paired") or 0)
    pair_dropped = int(opt_stats.get("pair_dropped") or 0)
    enqueued = int(opt_stats.get("enqueued") or 0)
    queue_dropped = int(opt_stats.get("queue_dropped") or 0)
    failed = int(opt_stats.get("failed") or 0)
    pair_den = paired + pair_dropped
    if pair_den > 0:
        rate = pair_dropped / pair_den
        if rate >= _PAIR_DROP_ERR:
            report.add_error(f"optical pair_dropped {pair_dropped}/{pair_den} ({rate * 100:.1f}%)")
        elif rate >= _PAIR_DROP_WARN:
            report.add_warn(f"optical pair_dropped {pair_dropped}/{pair_den} ({rate * 100:.1f}%)")
    if enqueued > 0 or queue_dropped > 0:
        qden = enqueued + queue_dropped
        rate = queue_dropped / qden if qden else 0.0
        if rate >= _QUEUE_DROP_ERR:
            report.add_error(f"optical queue_dropped {queue_dropped}/{qden} ({rate * 100:.1f}%)")
        elif rate >= _QUEUE_DROP_WARN:
            report.add_warn(f"optical queue_dropped {queue_dropped}/{qden} ({rate * 100:.1f}%)")
    if failed >= _FAILED_ERR:
        report.add_error(f"optical process_frame failed {failed} times")
    elif failed > 0:
        report.add_warn(f"optical process_frame failed {failed} times")

    report.summary["optical"] = {
        "rows": n,
        "unique_t": unique_opt,
        "stereo_ts": n_stereo,
        "track_stride": stride,
        "bodies": bodies,
        "pair_dropped": pair_dropped,
        "queue_dropped": queue_dropped,
        "failed": failed,
    }
