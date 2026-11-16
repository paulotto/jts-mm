# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-system anatomical frames (validation plan V5; master plan 4.2, R5c).

Each system digitizes the same dental landmarks with the calibration tool (CT) in its own
head (HP) and jaw (MP) bodies, builds its own maxillary and mandibular frames from them,
and reports the jaw-in-head motion

    ``M(t) = F_max⁻¹ · T_head(t)⁻¹ · T_jaw(t) · F_mand``.

No transform between the two systems' head or jaw bodies is fitted, and the world
transform cancels in ``T_head⁻¹ T_jaw``. Constant offsets of reconstructed marker points
(flat disks, sphere centres) are rigid in the body frame and cancel too, so the comparison
does not depend on marker geometry (R5d.1).

Conventions (``frames/landmarks``, user 2026-10-22): ``*_point_1`` is the incisor,
``*_point_2`` the left first molar, ``*_point_3`` the right first molar. Both frames use
the maxillary construction (``frames.maxillary.maxillary_frame``): origin at the incisor,
x anterior, y left, z up. The lower-incisor position in the maxillary frame is therefore
the translation of ``M``.

Every function works on plain arrays so that JTS-MM and Qualisys data go through the
same code: times on the master clock (s), CT tip positions in the world of the system
(mm), body poses (N, 4, 4) in mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from biomech import timebase
from biomech.frames.landmarks import LANDMARKS
from biomech.frames.maxillary import maxillary_frame
from biomech.kinematics import se3

HOLDS: tuple[str, ...] = tuple(LANDMARKS)            # the 8 landmark hold phases
BODY_OF = {n: s.body for n, s in LANDMARKS.items()}  # "mp" (jaw) | "hp" (head)


@dataclass
class Hold:
    name: str
    body: str                  # "mp" | "hp"
    point: np.ndarray          # (3,) mm in the body frame
    mad_mm: float
    n: int
    t0_s: float                # still run used (master clock)
    t1_s: float
    idx: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, int))


def local_tip(T_body: np.ndarray, p_world: np.ndarray) -> np.ndarray:
    """CT tip in the body frame, per frame: ``T_body⁻¹ · p``; NaN where either is missing."""
    out = np.full(p_world.shape, np.nan)
    ok = se3.valid(T_body) & np.isfinite(p_world).all(axis=1)
    R = T_body[ok, :3, :3]
    out[ok] = np.einsum("nji,nj->ni", R, p_world[ok] - T_body[ok, :3, 3])
    return out


def still_run(t_s: np.ndarray, local: np.ndarray, *, win_s: float = 1.0, tol_mm: float = 1.5,
              tail_mm: float = 6.0) -> np.ndarray:
    """Indices of the still part of a hold: the longest run (≥ ``win_s``) whose rolling
    ``win_s`` SD stays below ``tol_mm`` on every axis; without one, the frames within
    ``tail_mm`` of the median of the phase's last fifth (the hold ends the phase). The same
    rule for both systems (time-based window, so frame rates may differ)."""
    ok = np.isfinite(local).all(axis=1)
    if ok.sum() < 5:
        return np.zeros(0, int)
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else 0.025
    w = max(int(round(win_s / dt)), 5)
    sd = pd.DataFrame(local).rolling(w, center=True, min_periods=w // 2).std().max(axis=1)
    still = (sd < tol_mm).to_numpy() & ok
    runs = [r for r in timebase.segments(still) if r[1] - r[0] >= w]
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        return np.arange(a, b)[ok[a:b]]
    good = np.flatnonzero(ok)
    tail = np.median(local[good[int(0.8 * len(good)):]], axis=0)
    return good[np.linalg.norm(local[good] - tail, axis=1) < tail_mm]


def digitize(t_s: np.ndarray, tip_world: np.ndarray, T_head: np.ndarray, T_jaw: np.ndarray,
             phases: Sequence, names: Sequence[str] = HOLDS, min_n: int = 5,
             windows: dict[str, tuple[float, float]] | None = None) -> dict[str, Hold]:
    """The landmarks of one system: per hold phase (``phases``: objects with ``name``,
    ``t_start_us``, ``t_end_us`` on the master clock), the median tip position in the
    hold's body frame over the still run. Condyle holds are also returned in the head
    frame (``<name>_hp``). ``windows`` (master s, per hold): use exactly these frames
    instead of the system's own still run -- the window both systems were still in
    (:func:`common_windows`, erratum EV2), so both digitize the same physical event."""
    t_s = np.asarray(t_s, float)
    by = {p.name: p for p in phases}
    out: dict[str, Hold] = {}
    for name in names:
        ph = by.get(name)
        if ph is None:
            continue
        sel = np.flatnonzero((t_s >= ph.t_start_us * 1e-6) & (t_s <= ph.t_end_us * 1e-6))
        if len(sel) < min_n:
            continue
        bodies = [(BODY_OF[name], T_jaw if BODY_OF[name] == "mp" else T_head, name)]
        if LANDMARKS[name].jaw == "condyle":
            bodies.append(("hp", T_head, f"{name}_hp"))
        loc_main = local_tip(bodies[0][1][sel], tip_world[sel])
        if windows is not None and name in windows:
            w0, w1 = windows[name]
            keep = np.flatnonzero((t_s[sel] >= w0) & (t_s[sel] <= w1)
                                  & np.isfinite(loc_main).all(axis=1))
        else:
            keep = still_run(t_s[sel], loc_main)
        if len(keep) < min_n:
            continue
        for body, T, key in bodies:
            loc = loc_main if key == name else local_tip(T[sel], tip_world[sel])
            pts = loc[keep][np.isfinite(loc[keep]).all(axis=1)]
            if len(pts) < min_n:
                continue
            med = np.median(pts, axis=0)
            out[key] = Hold(key, body, med, float(np.median(np.linalg.norm(pts - med, axis=1))),
                            int(len(pts)), float(t_s[sel][keep[0]]), float(t_s[sel][keep[-1]]),
                            sel[keep])
    return out


def common_windows(a: dict[str, Hold], b: dict[str, Hold], min_s: float = 0.5
                   ) -> dict[str, tuple[float, float]]:
    """Per hold the time window (master s) both systems were still in: the intersection
    of their still runs, when it lasts ≥ ``min_s`` (otherwise the hold is left out and
    each system keeps its own still run)."""
    out = {}
    for n in set(a) & set(b):
        if n.endswith("_hp"):
            continue
        t0, t1 = max(a[n].t0_s, b[n].t0_s), min(a[n].t1_s, b[n].t1_s)
        if t1 - t0 >= min_s:
            out[n] = (t0, t1)
    return out


@dataclass
class Frames:
    F_max: np.ndarray          # T_head<-maxillary (4, 4)
    F_mand: np.ndarray         # T_jaw<-mandibular (4, 4)
    tmj_mand: dict[str, np.ndarray]   # condyle points in the mandibular frame


def anatomical_frames(holds: dict[str, Hold]) -> Frames:
    need = ("max_point_1", "max_point_2", "max_point_3", "mand_point_1", "mand_point_2",
            "mand_point_3")
    missing = [n for n in need if n not in holds]
    if missing:
        raise LookupError(f"landmarks missing: {missing}")
    F_max = maxillary_frame(holds["max_point_2"].point, holds["max_point_3"].point,
                            holds["max_point_1"].point)
    F_mand = maxillary_frame(holds["mand_point_2"].point, holds["mand_point_3"].point,
                             holds["mand_point_1"].point)
    inv = se3.inv(F_mand)
    tmj = {n: inv[:3, :3] @ holds[n].point + inv[:3, 3] for n in ("tmj_left", "tmj_right")
           if n in holds}
    return Frames(F_max, F_mand, tmj)


def jaw_in_head(T_head: np.ndarray, T_jaw: np.ndarray, fr: Frames) -> np.ndarray:
    """``M(t) = F_max⁻¹ · T_head⁻¹ · T_jaw · F_mand`` (N, 4, 4); NaN where a body is missing."""
    rel = se3.relative(T_head, T_jaw)
    M = np.einsum("ij,njk,kl->nil", se3.inv(fr.F_max), rel, fr.F_mand)
    M[~se3.valid(rel)] = np.nan
    return M


def points_of(M: np.ndarray, fr: Frames) -> dict[str, np.ndarray]:
    """Lower incisor and condyle points in the maxillary frame, (N, 3) each."""
    out = {"incisor": M[:, :3, 3].copy()}
    for n, p in fr.tmj_mand.items():
        out[n] = se3.apply(M, p)
    return out


def rotation_error_deg(M_a: np.ndarray, M_b: np.ndarray) -> np.ndarray:
    """Angle of ``R_a R_bᵀ`` per frame (deg), NaN where either is missing."""
    D = np.einsum("nij,nkj->nik", M_a[:, :3, :3], M_b[:, :3, :3])
    tr = np.clip((np.trace(D, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def landmark_table(holds: dict[str, Hold], system: str, pid: str) -> pd.DataFrame:
    return pd.DataFrame([{"pid": pid, "system": system, "landmark": h.name, "body": h.body,
                          "x_mm": h.point[0], "y_mm": h.point[1], "z_mm": h.point[2],
                          "mad_mm": h.mad_mm, "n": h.n, "t0_s": h.t0_s, "t1_s": h.t1_s}
                         for h in holds.values()])


def within_body_distances(holds: dict[str, Hold]) -> dict[tuple[str, str], float]:
    """Frame-free geometry: pairwise distances between the landmarks of the same body."""
    out = {}
    names = sorted(holds)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if holds[a].body == holds[b].body:
                out[(a, b)] = float(np.linalg.norm(holds[a].point - holds[b].point))
    return out
