# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stream-level quality products stored by the ``qc`` stage (G2).

The data views ``data_imu_quality``, ``data_timeline`` and ``data_tracking_scene`` used to
read the recorded session on every draw. These functions compute what they show once, in
the ``qc`` stage, so the views (and a frozen snapshot) work from the result store alone:

* :func:`imu_streams` -- per IMU stream the sample-interval summary (1st/99th percentile,
  median, longest, share of long intervals) and the coverage per experiment phase;
* :func:`timeline` -- the covered intervals of every stream on the master clock (a gap
  longer than three sample intervals splits a bar), the optical lock runs, the protocol
  phases, preview-frame times, and the optical fit residual RMS (stored by the capture as
  ``reprojection_rms``, ``rms_kind = kabsch_mm``: the RMS of the rigid tag fit in mm, E31)
  thinned to 4000 points;
* :func:`scene` -- world-frame (reference camera) origins of the headpiece, mandibular
  array and CT pointer tip per phase, thinned to <= 1500 frames, with three key-frame poses.

Times in the tables are seconds from the first phase start (``t0_us`` stored with them).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

SCENE_MAX_FRAMES = 1500
RMS_BUDGET = 4000
BODIES = ("cra", "mta", "ct")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) index runs of True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _segments(t_s: np.ndarray, max_gap: float) -> list[tuple[float, float]]:
    """Covered [t0, t1] intervals of a sample-time vector (gaps > max_gap split them)."""
    t = np.asarray(t_s, float)
    if t.size == 0:
        return []
    br = np.flatnonzero(np.diff(t) > max_gap)
    return [(float(t[a]), float(t[b])) for a, b in zip(np.r_[0, br + 1], np.r_[br, t.size - 1])]


def _minmax_index(y: np.ndarray, budget: int) -> np.ndarray:
    """Indices of the min and max per bin (peaks kept; NaN samples next to valid ones kept,
    so gaps stay gaps). Analysis-side twin of ``viz.downsample.thin_index`` for one series."""
    n = y.size
    if n <= budget:
        return np.arange(n)
    fin = np.isfinite(y)
    edge = np.flatnonzero(np.diff(fin.astype(np.int8)) != 0)
    keep = [np.r_[0, n - 1], edge, edge + 1]
    for b in np.array_split(np.arange(n), max(budget // 2, 1)):
        f = b[fin[b]]
        if f.size:
            keep.append(f[[np.argmin(y[f]), np.argmax(y[f])]])
    return np.unique(np.concatenate(keep))


def _imu_list(sd: Any) -> list[tuple[str, np.ndarray]]:
    out = []
    for slot in ("mta", "cra"):
        st = sd.imu(slot)
        if st is None:
            continue
        out.append((f"{slot.upper()} quat", np.asarray(st.t_us, np.int64)))
        if st.accel is not None:
            out.append((f"{slot.upper()} accel", np.asarray(st.accel.t_us, np.int64)))
    return out


def imu_streams(sd: Any, pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(intervals, coverage) tables of the IMU streams of a take."""
    rows, cov = [], []
    phases = [ph for ph in sd.phases if ph.kind == "experiment"] or list(sd.phases)
    for k, (lab, t) in enumerate(_imu_list(sd)):
        dt = np.diff(t) * 1e-3
        # Repeated or backward stamps are counted, not hidden (E36; IMU preprocessing
        # sanitizes them, an04); the interval statistics use the forward steps.
        nonmono_n = int(np.sum(dt <= 0))
        dt = dt[dt > 0]
        row: dict[str, Any] = {"pid": pid, "stream": lab, "order": k, "n_samples": int(t.size),
                               "nonmono_n": nonmono_n}
        if dt.size >= 2:
            med = float(np.median(dt))
            q1, q99 = np.percentile(dt, [1, 99])
            row.update(rate_hz=1000.0 / med, interval_median_ms=med, interval_p01_ms=float(q1),
                       interval_p99_ms=float(q99), interval_max_ms=float(dt.max()),
                       long_interval_pct=100.0 * float(np.mean(dt > 1.5 * med)))
        rows.append(row)
        step = np.median(np.diff(t)) if t.size > 1 else np.nan
        for j, ph in enumerate(phases):
            n = int(((t >= ph.t_start_us) & (t <= ph.t_end_us)).sum())
            exp = (ph.t_end_us - ph.t_start_us) / step + 1 if np.isfinite(step) and step > 0 \
                else np.nan
            cov.append({"pid": pid, "stream": lab, "phase": ph.name, "phase_order": j,
                        "coverage_pct": min(100.0, 100.0 * n / exp) if exp and np.isfinite(exp)
                        else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(cov)


def _preview_times(sd: Any) -> dict[str, np.ndarray]:
    path = getattr(sd, "folder", None)
    f = path / "preview_index.jsonl" if path is not None else None
    if f is None or not f.is_file():
        return {}
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x]
    out: dict[str, list[float]] = {}
    for r in rows:
        out.setdefault(str(r["cam_id"]), []).append(float(r["t_master_us"]))
    return {k: np.sort(np.asarray(v)) for k, v in out.items()}


def timeline(sd: Any, pid: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """(lanes, segments, arrays): lane order/label/rate, covered intervals per lane, and the
    thinned reprojection RMS per optical body (``timeline__{body}__t_s`` / ``__fit_rms_mm``)."""
    s = sd._s
    starts = [ph.t_start_us for ph in sd.phases]
    t0 = float(min(starts)) if starts else float(getattr(s, "t0_us", 0))

    def sec(t: Any) -> np.ndarray:
        return (np.asarray(t, float) - t0) * 1e-6

    lanes: list[dict[str, Any]] = []
    segs: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}

    def lane(name: str, kind: str, rate: float = np.nan, lock: float = np.nan) -> int:
        lanes.append({"pid": pid, "lane": name, "order": len(lanes), "kind": kind,
                      "rate_hz": rate, "lock_pct": lock, "t0_us": int(t0)})
        return len(lanes) - 1

    def add(i: int, intervals: list[tuple[float, float]], phase: str = "") -> None:
        segs.extend({"pid": pid, "lane": lanes[i]["lane"], "t0_s": a, "t1_s": b,
                     "phase": phase} for a, b in intervals)

    def cover(name: str, t: np.ndarray | None, kind: str = "stream") -> None:
        if t is None or len(t) < 2:
            lane(name, kind)
            return
        ts = sec(t)
        dt = float(np.median(np.diff(ts)))
        add(lane(name, kind, 1.0 / dt if dt > 0 else np.nan), _segments(ts, 3 * dt))

    i = lane("phases", "phases")
    for ph in sd.phases:
        add(i, [(float(sec(ph.t_start_us)), float(sec(ph.t_end_us)))], ph.name)
    cover("EMG", s.emg.t_us if s.emg is not None else None)
    for slot in ("mta", "cra"):
        st = sd.imu(slot)
        cover(f"IMU {slot.upper()} quat", None if st is None else st.t_us)
        cover(f"IMU {slot.upper()} accel",
              None if st is None or st.accel is None else st.accel.t_us)
    opt = sd.optical()
    for body in BODIES:
        if opt is None or body not in opt.poses:
            lane(f"optical {body.upper()}", "optical")
            continue
        ts = sec(opt.t_us)
        lock = opt.lock(body)
        dt = float(np.median(np.diff(ts)))
        i = lane(f"optical {body.upper()}", "optical", 1.0 / dt, 100.0 * float(lock.mean()))
        add(i, [(float(ts[a]), float(ts[b - 1])) for a, b in _runs(lock)])
        r = np.asarray(opt.quality.get(body, {}).get("reprojection_rms", []), float)
        if r.size == ts.size and np.isfinite(r).any():
            keep = _minmax_index(r, RMS_BUDGET)
            arrays[f"timeline__{body}__t_s"] = ts[keep].astype(np.float32)
            arrays[f"timeline__{body}__fit_rms_mm"] = r[keep].astype(np.float32)
    for cam, tt in sorted(_preview_times(sd).items()):
        cover(f"{cam} previews", tt, "previews")
    trig = getattr(s, "triggers", None)
    cover("triggers", None if trig is None else trig.t_us, "triggers")
    return pd.DataFrame(lanes), pd.DataFrame(segs, columns=["pid", "lane", "t0_s", "t1_s",
                                                            "phase"]), arrays


def scene(sd: Any) -> dict[str, np.ndarray]:
    """World-frame origins per phase and body (``scene__{phase}__{body}__xyz``, mm, float32)
    and three key-frame poses (``scene__{phase}__{body}__key``, (3, 4, 4)); ``all`` = the
    whole take."""
    opt = sd.optical()
    if opt is None:
        return {}
    t = opt.t_us.astype(np.int64)
    out: dict[str, np.ndarray] = {}
    sels = [("all", np.ones(t.size, bool))] + [
        (ph.name, (t >= ph.t_start_us) & (t <= ph.t_end_us)) for ph in sd.phases]
    for name, sel in sels:
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            continue
        idx = idx[:: max(1, idx.size // SCENE_MAX_FRAMES)]
        for body in BODIES:
            P = opt.poses.get(body)
            if P is None:
                continue
            Q = P[idx]
            out[f"scene__{name}__{body}__xyz"] = Q[:, :3, 3].astype(np.float32)
            ok = np.flatnonzero(np.isfinite(Q[:, :3, 3]).all(axis=1))
            if body != "ct" and ok.size:
                key = ok[np.linspace(0, ok.size - 1, 3).astype(int)]
                out[f"scene__{name}__{body}__key"] = Q[key].astype(np.float32)
    return out
