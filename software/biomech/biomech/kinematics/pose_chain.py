# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mandible motion in the subject's maxillary frame (an02 §1-2).

``T_max<-mp(t) = T_max<-hp · inv(T_w<-hp(t)) · T_w<-mp(t)``

1. relative pose headpiece -> mandibular array from the two optical bodies;
2. interior tracking gaps <= ``max_gap_s`` bridged by SE(3) interpolation
   (flagged), longer gaps stay NaN;
3. zero-phase low-pass of translation and rotation (``lowpass_hz``);
4. the eight landmarks from the calibration holds (6 teeth + 2 skin-palpated
   condyles); tooth roles from the convention, verified against the condyles;
5. expressed in the maxillary frame of :mod:`biomech.frames.maxillary`.

Any mandibular point given in MP coordinates is then tracked as
``apply(T, p_mp)`` in maxillary coordinates (x anterior, y left, z up; mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from biomech.frames.landmarks import (MAND, MAX, TMJ, CondyleCheck, DentalRoles, Landmarks,
                                      by_role, condyle_check, dental_roles, digitize)
from biomech.frames.maxillary import check, maxillary_frame
from biomech.io.session import SessionData
from biomech.kinematics import se3


@dataclass(frozen=True)
class KinParams:
    max_gap_s: float = 0.5
    lowpass_hz: float = 6.0
    lowpass_order: int = 2
    hold_trim: float = 0.2
    min_condyle_incisor_mm: float = 70.0
    min_condyle_superior_mm: float = 5.0
    min_condyle_posterior_mm: float = 40.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "KinParams":
        return cls(**(d or {}))


@dataclass
class JawMotion:
    t_us: np.ndarray                 # (N,) int64, optical frames
    fs: float
    T: np.ndarray                    # (N, 4, 4) T_max<-mp, smoothed; NaN = untracked
    T_raw: np.ndarray                # same, before smoothing (after gap bridging)
    bridged: np.ndarray              # (N,) frames filled by interpolation
    T_max_hp: np.ndarray             # (4, 4)
    landmarks: Landmarks
    upper: DentalRoles
    lower: DentalRoles
    condyles: CondyleCheck
    points_mp: dict[str, np.ndarray] = field(default_factory=dict)  # role -> MP coords
    upper_max: dict[str, np.ndarray] = field(default_factory=dict)  # role -> maxillary coords
    frame_check: dict[str, float] = field(default_factory=dict)

    def track(self, role: str, T: np.ndarray | None = None) -> np.ndarray:
        """(N, 3) maxillary coordinates of a mandibular point by role."""
        return se3.apply(self.T if T is None else T, self.points_mp[role])

    def window(self, t0_us: int, t1_us: int) -> np.ndarray:
        return (self.t_us >= t0_us) & (self.t_us <= t1_us)


def condyle_level_check(cc: CondyleCheck, fc: dict[str, float], min_superior_mm: float
                        ) -> CondyleCheck:
    """E13: both condyle points must lie ``min_superior_mm`` above the incisal level.

    Both must be checkable (E24): ``min(nan, x)`` is NaN but ``min(x, nan)`` is x, so the
    former ``min(left, right)`` depended on the argument order; a missing level now marks
    the condyles implausible ("condyle_level_unknown").
    """
    zl, zr = float(fc.get("tmj_left_z", np.nan)), float(fc.get("tmj_right_z", np.nan))
    if not cc.plausible:
        return cc
    if not (np.isfinite(zl) and np.isfinite(zr)):
        return replace(cc, plausible=False, reason="condyle_level_unknown")
    if not min(zl, zr) >= min_superior_mm:
        return replace(cc, plausible=False, reason="below_incisal_level")
    return cc


def condyle_depth_check(cc: CondyleCheck, fc: dict[str, float], min_posterior_mm: float
                        ) -> CondyleCheck:
    """D18 (revised): the mean sagittal depth of the two condyle points behind the upper
    incisor (maxillary frame, x anterior) must be at least ``min_posterior_mm``.

    The side mean cancels a yaw of the dental frame (three tooth-surface points; a few
    degrees of yaw move the two condyle points ~60 mm off the midline by opposite amounts
    in x). Tolerance budget: condyles lie ~75-90 mm behind the incisal point; a skin point
    marked by palpation may sit up to ~15 mm anterior to the joint (condyle translated onto
    the eminence) with ~10 mm scatter, so 75 - 15 - 2*10 = 40 mm, which is also the depth
    of the first molars: a point not behind the molars did not land on the joint. The 3-D
    distance check (``min_condyle_incisor_mm``) misses shallow points because the lateral
    offset adds to it."""
    xl, xr = float(fc.get("tmj_left_x", np.nan)), float(fc.get("tmj_right_x", np.nan))
    if not cc.plausible:
        return cc
    if not (np.isfinite(xl) and np.isfinite(xr)):
        return replace(cc, plausible=False, reason="condyle_depth_unknown")
    if not -0.5 * (xl + xr) >= min_posterior_mm:
        return replace(cc, plausible=False, reason="too_shallow")
    return cc


def build(sd: SessionData, p: KinParams = KinParams()) -> JawMotion:
    opt = sd.optical()
    if opt is None:
        raise LookupError(f"{sd.folder.name}: no optical data")
    t = opt.t_us.astype(np.int64)
    dt = np.median(np.diff(t)) * 1e-6
    fs = 1.0 / dt
    rel = se3.relative(opt.poses["cra"], opt.poses["mta"])
    rel, bridged = se3.bridge_gaps(rel, int(round(p.max_gap_s * fs)))

    lm = digitize(opt, sd.phases, p.hold_trim)
    need = MAX + MAND + TMJ + tuple(f"{n}_hp" for n in TMJ)
    missing = [n for n in need if not lm.has(n)]
    if missing:
        raise LookupError(f"{sd.folder.name}: landmarks missing: {missing}")
    upper = dental_roles({n: lm.get(n) for n in MAX}, lm.get("tmj_left_hp"),
                         lm.get("tmj_right_hp"), by_role("maxilla"))
    lower = dental_roles({n: lm.get(n) for n in MAND}, lm.get("tmj_left"), lm.get("tmj_right"),
                         by_role("mandible"))
    T_hp_max = maxillary_frame(lm.get(upper.left), lm.get(upper.right), lm.get(upper.incisor))
    T_max_hp = se3.inv(T_hp_max)
    T_raw = se3.compose(T_max_hp[None], rel)
    T_raw[~se3.valid(rel)] = np.nan
    T = se3.lowpass(T_raw, fs, p.lowpass_hz, p.lowpass_order)
    points = {
        "incisor": lm.get(lower.incisor), "left": lm.get(lower.left),
        "right": lm.get(lower.right), "tmj_left": lm.get("tmj_left"),
        "tmj_right": lm.get("tmj_right"),
    }
    points["tmj_mid"] = 0.5 * (points["tmj_left"] + points["tmj_right"])
    upper_max = {r: se3.apply(T_max_hp, lm.get(getattr(upper, r)))
                 for r in ("incisor", "left", "right")}
    fc = check(T_max_hp, {"tmj_left": lm.get("tmj_left_hp"), "tmj_right": lm.get("tmj_right_hp")})
    cc = condyle_check(points["incisor"], points["tmj_left"], points["tmj_right"],
                       p.min_condyle_incisor_mm)
    # E13: the condyle points must also lie above the incisal level (maxillary frame).
    cc = condyle_level_check(cc, fc, p.min_condyle_superior_mm)
    cc = condyle_depth_check(cc, fc, p.min_condyle_posterior_mm)
    return JawMotion(t, fs, T, T_raw, bridged, T_max_hp, lm, upper, lower, cc, points, upper_max,
                     fc)
