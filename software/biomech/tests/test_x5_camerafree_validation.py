# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD20 core (``validation.stages_camerafree.run_ref_camerafree``) on a noise-free camera-free
synthetic take (``testing.synth_camerafree``): with the optics and "Qualisys" equal to the
truth, every rotation rung is non-inferior to the optics and its opening error is tiny; the
accelerometer bridge with the true lever arms stays within a fraction of a millimetre."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.testing.synth_camerafree import CFSpec, generate
from biomech.validation import anatomy as AN
from biomech.validation import selection as SEL
from biomech.validation.stages_camerafree import run_ref_camerafree

pytestmark = pytest.mark.slow


def _interp(t, x, tq):
    return np.column_stack([np.interp(tq, t, x[:, j]) for j in range(x.shape[1])])


@pytest.fixture(scope="module")
def vd20():
    syn = generate(CFSpec(head_pitch_deg=6.0, head_roll_deg=3.0, head_yaw_deg=4.0,
                          head_sway_mm=3.0, free_translation_mm=0.0))
    tr = syn.truth
    t40 = np.arange(tr.t_s[0] + 0.0125, tr.t_s[-1], 0.025)
    rv = _interp(tr.t_s, tr.rotvec_deg, t40)
    inc = _interp(tr.t_s, tr.incisor_mm, t40)
    M = np.zeros((len(t40), 4, 4))
    M[:, :3, :3] = Rotation.from_rotvec(np.radians(rv)).as_matrix()
    M[:, :3, 3] = inc
    M[:, 3, 3] = 1.0
    F_max = se3.inv(tr.T_max_hp)
    shift = np.eye(4)
    shift[:3, 3] = tr.incisor0_mm
    F_mand = se3.inv(tr.T_max_mp_closed) @ shift          # mandibular frame: origin incisor
    d = tr.condyle0_mm - tr.incisor0_mm
    fr = AN.Frames(F_max, F_mand, {"tmj_left": d, "tmj_right": d})
    held = [SEL.Interval(n, a + 0.5, b - 0.5) for n, a, b in tr.tasks]
    tables, _ = run_ref_camerafree(
        t40, M, M.copy(), fr, held, np.zeros(len(t40), bool), syn.cra, syn.mta,
        {"cra": tr.R_mount_cra, "mta": tr.R_mount_mta},
        {"cra": tr.lever_cra_mm, "mta": tr.lever_mta_mm}, {"n_boot": 200}, "s99")
    return tables


def test_rotation_rungs_tiny_and_noninferior(vd20):
    seg = vd20["cf_val_segments"]
    pooled = seg[seg["segment"] == "movement_pooled"].set_index("estimator")
    assert pooled.loc["F0", "rmse_open_deg"] < 1e-6
    for e in ("E2a", "E2b", "E3a", "E3b"):
        assert pooled.loc[e, "rmse_open_deg"] < 0.05, (e, pooled.loc[e])
        assert pooled.loc[e, "rmse_geo_deg"] < 0.1, (e, pooled.loc[e])
    ni = vd20["cf_val_noninferiority"]
    rot = ni[ni["outcome"] == "opening_abs_err_deg"].set_index("estimator")
    assert rot.loc[["E2a", "E3b"], "noninferior"].all()


def test_translation_and_drift(vd20):
    seg = vd20["cf_val_segments"]
    pooled = seg[seg["segment"] == "movement_pooled"].set_index("estimator")
    assert pooled.loc["E5", "rmse_inc_mm"] < 0.5, pooled.loc["E5"]
    oc = seg[(seg["segment"] == "open_close")].set_index("estimator")
    assert oc.loc["E4own", "rmse_inc_mm"] < 0.2        # the model family holds the truth here
    dr = vd20["cf_val_drift"]
    assert set(dr["clock"]) == {"calibration", "anchor"}
    assert (dr["slope_dpm"].abs() < 0.1).all()          # no drift in a noise-free take
    s = vd20["cf_val_summary"].iloc[0]
    assert s["psi_b_identifiable"] and s["n_anchors"] >= 5 and "E4gen" in s["note"]


def test_validation_views_build(vd20):
    """The VD20 views on the core's tables (two participants: the second a copy)."""
    import pandas as pd

    from biomech.viz.views.camerafree import (val_camerafree, val_table_camerafree,
                                              val_table_camerafree_drift)
    from biomech.viz.views.registry import ViewRequest

    tabs = {k: pd.concat([v, v.assign(pid="s98")], ignore_index=True) for k, v in vd20.items()}

    class Ctx:
        def table(self, stage, name, pids=None):
            if stage != "ref_camerafree" or name not in tabs:
                raise KeyError(name)
            return tabs[name]

        def pids(self):
            return ["s98", "s99"]

        def pseudonym(self, pid):
            return {"s98": 1, "s99": 2}[pid]

    fig = val_camerafree(ViewRequest(Ctx()))
    assert fig.panels and all(p.series for p in fig.panels) and "n = 2" in fig.caption
    tab = val_table_camerafree(ViewRequest(Ctx()))
    assert any(str(r["pid"]).startswith("mean of 2") for r in tab.rows)
    assert {r["noninferior"] for r in tab.rows} >= {"yes"}
    assert not any("drift" in str(r["outcome"]) for r in tab.rows)   # drift has its own table
    drift = val_table_camerafree_drift(ViewRequest(Ctx()))
    assert drift.rows and any(str(r["pid"]).startswith("mean of 2") for r in drift.rows)
