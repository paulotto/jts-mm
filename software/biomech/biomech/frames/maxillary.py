# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Subject-specific maxillary frame (analysis_plan §4.1, an02 §1).

Built from the three digitized upper teeth, in the headpiece (HP) frame:

* origin  O = upper central incisor (``max_point_1``);
* y (left)     = unit(upper left first molar - upper right first molar)
                 (``max_point_2`` - ``max_point_3``);
* x (anterior) = component of (O - molar midpoint) orthogonal to y;
* z (superior) = x × y  (right-handed: x anterior, y left, z up).

The two skin-palpated condyles (``tmj_left/right``, HP frame) must then lie
above the occlusal plane (z > 0), behind the incisors (x < 0) and on their own
side; ``check`` reports all three, which also catches swapped left/right
labels (they flip z).
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def maxillary_frame(upper_left: np.ndarray, upper_right: np.ndarray, upper_incisor: np.ndarray
                    ) -> np.ndarray:
    """``T_hp<-max`` (4x4): the maxillary frame expressed in HP coordinates."""
    o = np.asarray(upper_incisor, dtype=np.float64)
    y = _unit(np.asarray(upper_left, float) - np.asarray(upper_right, float))
    x0 = o - 0.5 * (np.asarray(upper_left, float) + np.asarray(upper_right, float))
    x = _unit(x0 - (x0 @ y) * y)
    z = np.cross(x, y)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, o
    return T


def check(T_max_hp: np.ndarray, tmj_hp: dict[str, np.ndarray]) -> dict[str, float]:
    """Plausibility of the frame from the condyle positions (maxillary coordinates)."""
    out: dict[str, float] = {}
    for name, p in tmj_hp.items():
        q = T_max_hp[:3, :3] @ p + T_max_hp[:3, 3]
        out[f"{name}_x"], out[f"{name}_y"], out[f"{name}_z"] = map(float, q)
    if {"tmj_left", "tmj_right"} <= set(tmj_hp):
        out["lr_ok"] = float(out["tmj_left_y"] > out["tmj_right_y"])
        out["posterior_ok"] = float(max(out["tmj_left_x"], out["tmj_right_x"]) < 0)
        out["superior_ok"] = float(min(out["tmj_left_z"], out["tmj_right_z"]) > 0)
        # mean sagittal depth behind the incisor: insensitive to a yaw of the frame (D18)
        out["tmj_depth_mm"] = -0.5 * (out["tmj_left_x"] + out["tmj_right_x"])
    return out
