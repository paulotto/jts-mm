# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Template Kabsch ``model_points`` agree with the landmark convention.

JTS pairs ``coordinate_transform.model_points`` with the digitized maxillary
points in list order (``max_point_1..3``), so the template rows must be the
incisor, the participant's left first molar and the right first molar
(``LANDMARK_TEETH``). The model frame is right-handed with +x the
participant's left, +y superior and +z anterior (the same layout as the
headpiece frame): the incisor is the most anterior point on the midline, the
left molar has x > 0 and the occlusal triangle's normal
``(p2 - p1) x (p3 - p1)`` points up. With the rows of 2 and 3 swapped, a
proper Kabsch rotation flips the triangle over and puts the condyles below
the teeth. Checked on the Oct 2026 pilot (headpiece frame, right-handed):
the digitized triangle in convention order has its normal towards the
skin-palpated condyles, 7-44 mm above the occlusal plane in 11 of 12
participants (s11 within 4 mm of it); after Kabsch onto these rows the
condyles land at the same heights, with rows 2/3 swapped at minus them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optical.tracking.experiment import LANDMARK_HINTS, LANDMARK_TEETH, _default_jts_config

REPO = Path(__file__).resolve().parents[3]


def _template_sources() -> dict[str, list]:
    out = {
        "experiment._default_jts_config": _default_jts_config()["analysis"]["coordinate_transform"][
            "model_points"
        ]
    }
    for rel in ("software/optical/etc/jts_config.json", "software/jaw_tracking/config/config.json"):
        path = REPO / rel
        if path.is_file():
            out[rel] = json.loads(path.read_text())["analysis"]["coordinate_transform"]["model_points"]
    return out


def test_maxillary_points_follow_landmark_teeth_order():
    names = [n for n in LANDMARK_TEETH if n.startswith("max_point_")]
    assert names == ["max_point_1", "max_point_2", "max_point_3"]
    assert "incisor" in LANDMARK_TEETH["max_point_1"]
    assert "left" in LANDMARK_TEETH["max_point_2"] and "molar" in LANDMARK_TEETH["max_point_2"]
    assert "right" in LANDMARK_TEETH["max_point_3"] and "molar" in LANDMARK_TEETH["max_point_3"]
    assert "participant's left" in LANDMARK_HINTS["tmj_left"]


@pytest.mark.parametrize("source", sorted(_template_sources()))
def test_template_model_points_match_convention(source):
    pts = np.asarray(_template_sources()[source], dtype=float)
    assert pts.shape == (3, 3)
    incisor, left, right = pts
    # incisor on the midline, the molars symmetric about it
    assert abs(incisor[0]) < 0.1 * abs(left[0] - right[0])
    # +x is the participant's left
    assert left[0] > incisor[0] > right[0]
    # +z anterior: the incisor is in front of both molars
    assert incisor[2] > left[2] + 15 and incisor[2] > right[2] + 15
    # right-handed (+x left, +y up, +z anterior): the occlusal triangle's normal points up
    normal = np.cross(left - incisor, right - incisor)
    normal /= np.linalg.norm(normal)
    assert normal[1] > 0.9


def test_template_copies_agree():
    srcs = _template_sources()
    ref = np.asarray(srcs.pop("experiment._default_jts_config"), dtype=float)
    for name, pts in srcs.items():
        np.testing.assert_allclose(np.asarray(pts, dtype=float), ref, err_msg=name)
