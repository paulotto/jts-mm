# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-participant jaw geometry from the digitized landmarks and the recorded motion (D21).

Static geometry (rigid landmark sets, no motion needed):

* Bonwill triangle: lower incisor and the two condyle points; ``bonwill_ratio`` = mean
  condyle–incisor side / intercondylar width (1 for Bonwill's equilateral triangle).
* Balkwill angle: between the Bonwill plane and the lower occlusal plane (incisor and the
  two first molars), both through the incisor.
* Arch depth: incisor to the line through the two first molars (upper and lower).
* Condyle height: condyle midpoint above the lower occlusal plane (at closure, signed).

Functional geometry (the kinematic stand-ins for the anatomy, maxillary frame at closure):

* Hinge radius: lower incisor to the best-fit hinge axis of the openings.
* Axis offset: where the best-fit axis pierces the sagittal plane minus the palpated
  condyle midpoint (x anterior, z up), and the angle between the axis and the line through
  the palpated condyles.

Not estimated: guidance and Bennett angles from the protrusion and laterotrusion paths. The
protocol did not ask for tooth contact and the closed pose is the rest position (with the
freeway space), so the incisor path in protrusion rose in 9 of 14 participants and the
balancing condyle points moved up to 144 deg from the sagittal plane (D21): protocol and
skin-palpation effects, not occlusal or condylar guidance.

The condyle points are skin-palpated (lateral to the joint); every condyle-based value
carries ``condyles_plausible`` from ``kinematics:anatomy`` (flag, don't exclude). The
best-fit axis is the equivalent fixed axis of opening (rotation plus condylar
translation), not the anatomical condylar axis: it lies below and behind the condyles,
perpendicular to their path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biomech.kinematics import se3
from biomech.kinematics.hinge_fit import HingeResult
from biomech.kinematics.pose_chain import JawMotion


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _normal(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return _unit(np.cross(b - a, c - a))


def _line_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    u = _unit(b - a)
    d = p - a
    return float(np.linalg.norm(d - (d @ u) * u))


def jaw_geometry(jm: JawMotion, closed_T: np.ndarray | None, hinge: HingeResult | None
                 ) -> dict[str, Any]:
    pm, up = jm.points_mp, jm.upper_max
    inc, ml, mr = pm["incisor"], pm["left"], pm["right"]
    tl, tr = pm["tmj_left"], pm["tmj_right"]
    side = 0.5 * (np.linalg.norm(tl - inc) + np.linalg.norm(tr - inc))
    cos_b = abs(float(_normal(inc, tl, tr) @ _normal(inc, ml, mr)))
    out: dict[str, Any] = {
        "bonwill_side_mm": float(side),
        "bonwill_ratio": float(side / np.linalg.norm(tl - tr)),
        "balkwill_deg": float(np.degrees(np.arccos(min(cos_b, 1.0)))),
        "lower_arch_depth_mm": _line_dist(inc, ml, mr),
        "upper_arch_depth_mm": _line_dist(up["incisor"], up["left"], up["right"]),
    }
    if closed_T is None:
        return out
    c = {r: se3.apply(closed_T, pm[r]) for r in ("incisor", "left", "right", "tmj_left",
                                                 "tmj_right", "tmj_mid")}
    n_occ = _normal(c["incisor"], c["left"], c["right"])
    n_occ = n_occ if n_occ[2] >= 0 else -n_occ
    out["condyle_height_mm"] = float((c["tmj_mid"] - c["incisor"]) @ n_occ)
    if hinge is not None:
        n = _unit(hinge.n)
        d = c["incisor"] - hinge.s
        out["hinge_radius_mm"] = float(np.linalg.norm(d - (d @ n) * n))
        if abs(n[1]) > 0.5:
            pierce = hinge.s - (hinge.s[1] / n[1]) * n
            out["hinge_axis_dx_mm"] = float(pierce[0] - c["tmj_mid"][0])
            out["hinge_axis_dz_mm"] = float(pierce[2] - c["tmj_mid"][2])
            out["hinge_axis_offset_mm"] = float(np.hypot(out["hinge_axis_dx_mm"],
                                                         out["hinge_axis_dz_mm"]))
        cos_a = abs(float(n @ _unit(c["tmj_left"] - c["tmj_right"])))
        out["hinge_axis_angle_deg"] = float(np.degrees(np.arccos(min(cos_a, 1.0))))
    return out
