# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5 review fixes (errata E29+)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomech.stats.cohort_kinematics import hinge_noise_counts


def test_untested_participants_never_count_as_above_noise():
    """E30: a participant without a null (too few cycles: no above/ratio columns; or a null
    without draws: above False, ratio NaN) is neither above nor in the denominator."""
    hn = pd.DataFrame({"pid": ["s01", "s02", "s03", "s04"],
                       "above_invivo": [True, False, np.nan, False],
                       "ratio_invivo": [1.4, 0.8, np.nan, np.nan]})
    assert hinge_noise_counts(hn, "invivo") == (1, 2)
    assert hinge_noise_counts(pd.DataFrame(), "invivo") == (0, 0)
    assert hinge_noise_counts(hn.drop(columns="above_invivo"), "invivo") == (0, 0)


def test_sensitivity_compare_matches_rows_by_identity_not_position():
    """E33: a variant without one participant must not shift every row by one."""
    from biomech.sensitivity import compare_tables

    a = pd.DataFrame({"pid": ["s01", "s02", "s03"], "v": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"pid": ["s01", "s03"], "v": [1.0, 3.5]})
    c = compare_tables(a, b).set_index("key")
    assert c.loc["pid=s03", "diff"] == 0.5 and c.loc["pid=s01", "diff"] == 0.0
    assert c.loc["pid=s02", "match"] == "primary_only" and np.isnan(c.loc["pid=s02", "variant"])


def test_spm_bars_from_stored_clusters():
    """E0: clusters as stored (text or list; a count is not a range) become one bar row per
    test, labelled in the legend."""
    from biomech.viz.spec import Panel
    from biomech.viz.views import blocks as B

    assert B.parse_clusters("[(0, 10, 0.0036), (59, 100, 0.0)]") == [(0.0, 10.0, 0.0036),
                                                                     (59.0, 100.0, 0.0)]
    assert B.parse_clusters("3") == [] and B.parse_clusters(None) == []
    assert B.parse_clusters([(1, 2)])[0][:2] == (1.0, 2.0)
    p = Panel("x")
    n = B.add_spm_bars(p, [("A vs B", "[(0, 10, 0.01)]", "context"), ("none", "[]", "cat:1"),
                           ("C vs D", [(20, 30, 0.02), (50, 60, 0.03)], "cat:2")])
    bars = [s for s in p.series if s.kind == "sigbar"]
    assert n == 2 and [s.style["row"] for s in bars] == [0, 2]
    assert bars[0].label == "SPM: A vs B" and list(bars[1].x) == [20.0, 50.0]


def test_condyle_depth_uses_the_side_mean():
    """D18 (revised): a yaw of the dental frame shifts the two condyle points by opposite
    amounts in x; the side mean must pass, a point pair in front of the molars must not."""
    from biomech.frames.landmarks import CondyleCheck
    from biomech.kinematics.pose_chain import condyle_depth_check

    ok = CondyleCheck(95.0, 95.0, 125.0, True)
    yawed = {"tmj_left_x": -91.0, "tmj_right_x": -40.0}       # mean 65.5 mm
    assert condyle_depth_check(ok, yawed, 40.0).plausible
    shallow = {"tmj_left_x": -15.0, "tmj_right_x": -29.5}     # mean 22 mm
    assert condyle_depth_check(ok, shallow, 40.0).reason == "too_shallow"
    assert condyle_depth_check(ok, {"tmj_left_x": -70.0}, 40.0).reason == "condyle_depth_unknown"
