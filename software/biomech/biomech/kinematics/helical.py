# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Finite helical axes (screw axes) between pose pairs (an02 §5).

For two poses A, B of the mandible in the maxillary frame, the displacement
``D = B · inv(A)`` maps every mandibular point from its position at A to its
position at B. Its screw axis has unit direction n, rotation phi about it, a
translation t along it, and passes through the point s closest to the origin
(Spoor & Veldpaus 1980; Woltring et al. 1985).

Axis position error grows as (pose noise / phi), so pairs are taken
``step_deg`` apart (default 5 deg) rather than frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3


@dataclass(frozen=True)
class Screw:
    n: np.ndarray          # unit axis direction
    s: np.ndarray          # point on the axis closest to the origin (mm)
    phi_deg: float         # rotation about n
    t_mm: float            # translation along n

    def pierce(self, axis: int = 1, value: float = 0.0) -> np.ndarray | None:
        """Intersection with the plane ``x[axis] = value`` (default: mid-sagittal y = 0)."""
        if abs(self.n[axis]) <= 0.5:          # as coupling.hinge_summary / geometry
            return None
        lam = (value - self.s[axis]) / self.n[axis]
        return self.s + lam * self.n


def screw(A: np.ndarray, B: np.ndarray) -> Screw | None:
    """Screw axis of the displacement from pose A to pose B (both T_parent<-body)."""
    D = se3.compose(B, se3.inv(A))
    if not np.isfinite(D).all():
        return None
    rv = Rotation.from_matrix(D[:3, :3]).as_rotvec()
    phi = float(np.linalg.norm(rv))
    if phi < 1e-6:
        return None
    n = rv / phi
    d = D[:3, 3]
    t = float(n @ d)
    # (I - R) s = d - t n has a one-parameter family of solutions (along n);
    # the minimum-norm one is the point closest to the origin.
    s = np.linalg.lstsq(np.eye(3) - D[:3, :3], d - t * n, rcond=None)[0]
    s = s - (s @ n) * n
    return Screw(n, s, float(np.degrees(phi)), t)


def stepped_screws(T: np.ndarray, angle_deg: np.ndarray, step_deg: float = 5.0
                   ) -> list[tuple[int, int, Screw]]:
    """Successive screws along a monotonic movement, each spanning >= ``step_deg``.

    ``angle_deg`` is the movement's progress (e.g. opening angle) per frame.
    """
    out: list[tuple[int, int, Screw]] = []
    i = 0
    n = len(T)
    while i < n - 1:
        if not np.isfinite(angle_deg[i]):
            i += 1
            continue
        j = i + 1
        while j < n and (not np.isfinite(angle_deg[j]) or abs(angle_deg[j] - angle_deg[i])
                         < step_deg):
            j += 1
        if j >= n:
            break
        sc = screw(T[i], T[j])
        if sc is not None:
            out.append((i, j, sc))
        i = j
    return out
