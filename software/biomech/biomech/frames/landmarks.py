# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The eight landmarks digitized with the CT pointer during the calibration holds.

Convention (operator checklist op01, ``optical.tracking.experiment.LANDMARK_HINTS``;
left/right are the participant's):

==============  ============================================  =========  ===========
phase           landmark                                      body       role
==============  ============================================  =========  ===========
mand_point_1    lower central incisor (incisal edge, midline) MP         incisor
mand_point_2    lower left first molar                        MP         left
mand_point_3    lower right first molar                       MP         right
max_point_1     upper central incisor (incisal edge, midline) HP         incisor
max_point_2     upper left first molar                        HP         left
max_point_3     upper right first molar                       HP         right
tmj_left        left condyle, skin-palpated                   MP and HP  condyle
tmj_right       right condyle, skin-palpated                  MP and HP  condyle
==============  ============================================  =========  ===========

Each landmark is the median CT-tip position (``/optical/pose_ct`` is the tip)
in its body's frame over the central part of the hold; the tip's spread in
the hold (3-D MAD) is kept as digitization noise for the uncertainty stage.
The condyles are stored in the mandibular frame (they move with the mandible:
condylar paths, hinge models) *and* in the headpiece frame (``tmj_*_hp``: a
fixed skull reference for the maxillary-frame check).

The tooth roles follow the convention and are *verified* from geometry
(:func:`dental_roles`): the incisor must be the point most midline between the
condyles and farthest from them, and the molars must sit on the side of their
condyle. The incisor is never relabelled (``*_point_1`` is always the
incisor); only an unambiguous swap of the two molars overrides the convention,
and it is flagged. The condyles are checked for
anatomical plausibility (:func:`condyle_check`): skin palpation can land far
from the joint, so condyle-based outcomes are secondary (analysis_plan A4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from biomech import timebase
from biomech.io.session import OpticalData
from biomech.kinematics import se3


@dataclass(frozen=True)
class LandmarkSpec:
    jaw: str          # "mandible" | "maxilla" | "condyle"
    role: str         # "incisor" | "left" | "right"
    body: str         # frame the landmark belongs to: "mp" | "hp"
    description: str


LANDMARKS: dict[str, LandmarkSpec] = {
    "mand_point_1": LandmarkSpec("mandible", "incisor", "mp",
                                 "lower central incisor (incisal edge, midline)"),
    "mand_point_2": LandmarkSpec("mandible", "left", "mp", "lower left first molar"),
    "mand_point_3": LandmarkSpec("mandible", "right", "mp", "lower right first molar"),
    "max_point_1": LandmarkSpec("maxilla", "incisor", "hp",
                                "upper central incisor (incisal edge, midline)"),
    "max_point_2": LandmarkSpec("maxilla", "left", "hp", "upper left first molar"),
    "max_point_3": LandmarkSpec("maxilla", "right", "hp", "upper right first molar"),
    "tmj_left": LandmarkSpec("condyle", "left", "mp",
                             "left condyle (skin-palpated, participant's left)"),
    "tmj_right": LandmarkSpec("condyle", "right", "mp",
                              "right condyle (skin-palpated, participant's right)"),
}
MAND = tuple(n for n, s in LANDMARKS.items() if s.jaw == "mandible")
MAX = tuple(n for n, s in LANDMARKS.items() if s.jaw == "maxilla")
TMJ = tuple(n for n, s in LANDMARKS.items() if s.jaw == "condyle")


def by_role(jaw: str) -> dict[str, str]:
    """role -> landmark name for one jaw, per the convention."""
    return {s.role: n for n, s in LANDMARKS.items() if s.jaw == jaw}


@dataclass
class Landmark:
    name: str
    body: str                 # "mp" | "hp"
    point: np.ndarray         # (3,) mm in that body's frame
    mad_mm: float             # 3D median absolute deviation of the tip in the hold
    n: int                    # samples used


@dataclass
class Landmarks:
    items: dict[str, Landmark] = field(default_factory=dict)

    def get(self, name: str) -> np.ndarray:
        return self.items[name].point

    def has(self, *names: str) -> bool:
        return all(n in self.items for n in names)


def _hold(opt: OpticalData, body_pose: np.ndarray, t0: int, t1: int, trim: float
          ) -> tuple[np.ndarray, float, int] | None:
    t = opt.t_us.astype(np.int64)
    span = t1 - t0
    sel = (t >= t0 + trim * span) & (t <= t1 - trim * span)
    tip = opt.poses["ct"][sel]
    body = body_pose[sel]
    ok = se3.valid(tip) & se3.valid(body)
    if ok.sum() < 5:
        return None
    local = np.einsum("nji,nj->ni", body[ok, :3, :3], tip[ok, :3, 3] - body[ok, :3, 3])
    med = np.median(local, axis=0)
    mad = float(np.median(np.linalg.norm(local - med, axis=1)))
    return med, mad, int(ok.sum())


def _still_hold(opt: OpticalData, body_pose: np.ndarray, t0: int, t1: int,
                win_s: float = 1.0, tol_mm: float = 1.5, tail_mm: float = 6.0
                ) -> tuple[np.ndarray, float, int] | None:
    """The tip held still in the body frame, for phases that also contain other motion
    (the comparison takes run the CT sync motion inside ``mand_point_1``, then hold).

    The longest run (at least ``win_s`` long) whose rolling ``win_s`` SD stays below
    ``tol_mm`` (per axis); without one, the frames within ``tail_mm`` of the median of the phase's last fifth (the hold
    ends the phase)."""
    t = opt.t_us.astype(np.int64)
    sel = np.flatnonzero((t >= t0) & (t <= t1))
    tip, body = opt.poses["ct"][sel], body_pose[sel]
    ok = se3.valid(tip) & se3.valid(body)
    if ok.sum() < 5:
        return None
    local = np.full((len(sel), 3), np.nan)
    local[ok] = np.einsum("nji,nj->ni", body[ok, :3, :3], tip[ok, :3, 3] - body[ok, :3, 3])
    dt = float(np.median(np.diff(t[sel]))) * 1e-6 if len(sel) > 1 else 0.025
    w = max(int(round(win_s / dt)), 5)
    sd = pd.DataFrame(local).rolling(w, center=True, min_periods=w // 2).std().max(axis=1)
    still = (sd < tol_mm).to_numpy() & ok
    runs = [r for r in timebase.segments(still) if r[1] - r[0] >= w]
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        pts = local[a:b][ok[a:b]]
    else:
        good = local[ok]
        tail = np.median(good[int(0.8 * len(good)):], axis=0)
        pts = good[np.linalg.norm(good - tail, axis=1) < tail_mm]
    if len(pts) < 5:
        return None
    med = np.median(pts, axis=0)
    return med, float(np.median(np.linalg.norm(pts - med, axis=1))), int(len(pts))


def digitize(opt: OpticalData, phases: list, trim: float = 0.2, hold: str = "phase",
             still_phases: tuple[str, ...] = ("mand_point_1",)) -> Landmarks:
    """All eight landmarks (+ condyles in HP); a missing or untracked hold is skipped.

    ``hold="phase"``: median over the central part of each phase (dataset takes).
    ``hold="still"``: for the phases in ``still_phases``, the still part only
    (:func:`_still_hold`; the validation comparison takes)."""
    if hold not in ("phase", "still"):
        raise ValueError(f"hold must be 'phase' or 'still', got {hold!r}")
    by_name = {ph.name: ph for ph in phases}
    pose_key = {"mp": "mta", "hp": "cra"}
    out = Landmarks()
    if "ct" not in opt.poses:
        return out
    for name, spec in LANDMARKS.items():
        ph = by_name.get(name)
        if ph is None:
            continue
        if hold == "still" and name in still_phases:
            res = _still_hold(opt, opt.poses[pose_key[spec.body]], ph.t_start_us, ph.t_end_us)
        else:
            res = _hold(opt, opt.poses[pose_key[spec.body]], ph.t_start_us, ph.t_end_us, trim)
        if res is not None:
            out.items[name] = Landmark(name, spec.body, *res)
        if spec.jaw == "condyle":
            res = _hold(opt, opt.poses["cra"], ph.t_start_us, ph.t_end_us, trim)
            if res is not None:
                out.items[name + "_hp"] = Landmark(name + "_hp", "hp", *res)
    return out


@dataclass
class DentalRoles:
    """Tooth role -> landmark name for one jaw, with the geometric verification."""
    incisor: str
    left: str
    right: str
    incisor_asym_mm: float        # |d(left condyle) - d(right condyle)| of the incisor
    side_margin_mm: float         # smallest |asym| among the two molars
    geometry_ok: bool             # the geometric assignment is unambiguous
    agrees: bool                  # ... and equals the convention
    relabelled: bool              # convention overridden by an unambiguous contradiction


def _geometric(points: dict[str, np.ndarray], tmj_left: np.ndarray, tmj_right: np.ndarray
               ) -> tuple[str, str, str, dict[str, float], bool]:
    names = sorted(points)
    mid = 0.5 * (tmj_left + tmj_right)
    asym = {n: float(np.linalg.norm(points[n] - tmj_left) - np.linalg.norm(points[n] - tmj_right))
            for n in names}
    dmid = {n: float(np.linalg.norm(points[n] - mid)) for n in names}
    inc = min(names, key=lambda n: abs(asym[n]) - dmid[n])
    left, right = sorted((n for n in names if n != inc), key=lambda n: asym[n])
    margin = min(abs(asym[left]), abs(asym[right]))
    ok = asym[left] < 0 < asym[right] and abs(asym[inc]) < 0.5 * margin
    return inc, left, right, asym, ok


def dental_roles(points: dict[str, np.ndarray], tmj_left: np.ndarray, tmj_right: np.ndarray,
                 convention: dict[str, str] | None = None) -> DentalRoles:
    """Roles from ``convention`` (role -> name), verified against the condyle geometry.

    Geometry: the incisor is the point most midline between the two condyle
    points and farthest from their midpoint; each molar is closer to its own
    condyle. Without a convention (or when every convention name is missing)
    the geometric assignment is used.
    """
    inc, left, right, asym, ok = _geometric(points, tmj_left, tmj_right)
    margin = min(abs(asym[left]), abs(asym[right]))
    conv = convention or {}
    have = all(conv.get(r) in points for r in ("incisor", "left", "right"))
    if not have:
        return DentalRoles(inc, left, right, abs(asym[inc]), margin, ok, False, False)
    c = (conv["incisor"], conv["left"], conv["right"])
    agrees = (inc, left, right) == c
    # The incisor is always the convention's (mand_point_1 / max_point_1: the operator
    # digitizes it first, user 2026-10-22); only the two molars may be swapped, and only
    # when their sides are unambiguous.
    a_l, a_r = asym[conv["left"]], asym[conv["right"]]
    if a_l > 0 > a_r and min(abs(a_l), abs(a_r)) > 0:
        return DentalRoles(c[0], c[2], c[1], abs(asym[c[0]]), margin, ok, False, True)
    return DentalRoles(*c, abs(asym[c[0]]), margin, ok, agrees, False)


@dataclass(frozen=True)
class CondyleCheck:
    incisor_left_mm: float        # lower incisor <-> left condyle (MP frame)
    incisor_right_mm: float
    intercondylar_mm: float
    plausible: bool
    reason: str = ""              # why implausible: "too_close" | "below_incisal_level" |
    #                               "too_shallow" (D18) | ""


def condyle_check(incisor: np.ndarray, tmj_left: np.ndarray, tmj_right: np.ndarray,
                  min_incisor_mm: float = 70.0) -> CondyleCheck:
    """Skin-palpated condyles vs. anatomy: condyle–incisor distance below
    ``min_incisor_mm`` means the palpation landed far from the joint (adult
    condylion–incisor distances are around 100 mm; the skin point is lateral,
    never closer). The result is a flag; condyle-based outcomes are reported
    with and without implausible participants."""
    dl = float(np.linalg.norm(incisor - tmj_left))
    dr = float(np.linalg.norm(incisor - tmj_right))
    ok = min(dl, dr) >= min_incisor_mm
    return CondyleCheck(dl, dr, float(np.linalg.norm(tmj_left - tmj_right)), ok,
                        "" if ok else "too_close")
