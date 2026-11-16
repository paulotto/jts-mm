# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""X5b oracle decomposition and X5c task-conditioned translation model (dataset plan X5b, X5c;
) on synthetic takes with a known truth (:mod:`biomech.testing.synth_camerafree`).

* X5b separates the two error sources: with a perfect rotation the optical-rotation
  combination carries exactly the translation-model error (and equals the full E4own), with
  a perfect translation model the optical-translation combination carries exactly the
  rotation-induced error ``(R_est − R_opt)(p0 − c0)``; each combination is blind to the other
  source.
* X5c with one class (every calibration frame) is E4own; per-class fits use only their
  labelled frames and report why a class has no model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.imu import camerafree as cf
from biomech.testing.synth_camerafree import CFSpec, generate

ROT = ("open_close", "chewing")
TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing")


def _model(tr, t):
    cal = tr.mask("free_motion")
    thd = cf.theta_rate_dps(t, tr.rotvec_deg[:, 1])
    return cf.TranslationModel.fit(tr.rotvec_deg[cal], thd[cal], tr.tau_mm[cal]), thd


def _rms3(d: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(np.sum(d ** 2, axis=1))))


def _yaw(R: np.ndarray, deg: float) -> np.ndarray:
    """A constant off-axis rotation error of ``deg`` about the maxillary z axis."""
    return Rotation.from_rotvec([0.0, 0.0, np.radians(deg)]).as_matrix() @ R


@pytest.fixture(scope="module")
def exact():
    """Translation inside the model family in the rotational tasks (E4own is exact there)."""
    return generate(CFSpec(free_translation_mm=0.0)).truth


@pytest.fixture(scope="module")
def misfit():
    """Free-motion translation no rotation explains: the E4own model carries an error."""
    return generate(CFSpec(free_translation_mm=1.5, seed=3)).truth


def _decompose(tr, R_est, model, thd):
    t = tr.t_s
    tau_opt = tr.condyle_mm - tr.condyle0_mm
    o = cf.oracle_incisors(R_est, tr.R_rel, model.predict(tr.rotvec_deg, thd), tau_opt,
                           tr.incisor0_mm, tr.condyle0_mm)
    rv = np.degrees(Rotation.from_matrix(R_est).as_rotvec())
    full = cf.incisor_position(R_est, model.predict(rv, cf.theta_rate_dps(t, rv[:, 1])),
                               tr.incisor0_mm, tr.condyle0_mm)
    return o, full


def test_x5b_optical_translation_reproduces_the_optics(exact):
    """Optical rotation + optical translation is the optical incisor (the chain is closed)."""
    tr = exact
    o = cf.oracle_incisors(tr.R_rel, tr.R_rel, np.zeros_like(tr.tau_mm),
                           tr.condyle_mm - tr.condyle0_mm, tr.incisor0_mm, tr.condyle0_mm)
    assert np.abs(o["X5b_rotation"] - tr.incisor_mm).max() < 1e-9


def test_x5b_perfect_rotation_isolates_the_model_error(misfit):
    """R_est = R_opt: (i) equals the full E4own and its error is τ̂(r_opt) − τ_opt alone;
    (ii) is the optics."""
    tr = misfit
    model, thd = _model(tr, tr.t_s)
    o, full = _decompose(tr, tr.R_rel, model, thd)
    m = tr.mask(*TASKS)
    e_i = _rms3((o["X5b_model"] - tr.incisor_mm)[m])
    model_err = _rms3((model.predict(tr.rotvec_deg, thd) - (tr.condyle_mm - tr.condyle0_mm))[m])
    assert e_i > 0.3                                       # the misfit is visible
    assert e_i == pytest.approx(model_err, rel=1e-9)
    assert np.allclose(o["X5b_model"], full, atol=1e-9)
    assert _rms3((o["X5b_rotation"] - tr.incisor_mm)[m]) < 1e-9


def test_x5b_perfect_model_isolates_the_rotation_error(exact):
    """Translation model exact (rotational tasks) and a 3° off-axis rotation error: (ii)
    equals the analytic rotation-induced error (R_est − R_opt)(p0 − c0); (i) stays at the
    (zero) model error whatever the rotation error."""
    tr = exact
    model, thd = _model(tr, tr.t_s)
    m = tr.mask(*ROT)
    lever = tr.incisor0_mm - tr.condyle0_mm
    errs = {}
    for deg in (0.0, 3.0):
        R_est = _yaw(tr.R_rel, deg)
        o, full = _decompose(tr, R_est, model, thd)
        e_ii = _rms3((o["X5b_rotation"] - tr.incisor_mm)[m])
        analytic = _rms3(np.einsum("nij,j->ni", R_est - tr.R_rel, lever)[m])
        assert e_ii == pytest.approx(analytic, rel=1e-9, abs=1e-12)
        errs[deg] = (_rms3((o["X5b_model"] - tr.incisor_mm)[m]), e_ii,
                     _rms3((full - tr.incisor_mm)[m]))
    assert errs[0.0][0] < 0.05 and errs[3.0][0] == errs[0.0][0]      # (i) blind to rotation
    assert errs[3.0][1] > 3.0                                     # ~90 mm lever x 3°
    assert errs[3.0][2] > 0.5 * errs[3.0][1]          # the full E4own carries the rotation


def test_x5c_one_class_equals_e4own(misfit):
    tr = misfit
    t = tr.t_s
    cal = tr.mask("free_motion")
    thd = cf.theta_rate_dps(t, tr.rotvec_deg[:, 1])
    e4 = cf.TranslationModel.fit(tr.rotvec_deg[cal], thd[cal], tr.tau_mm[cal])
    fits = cf.fit_task_models(tr.rotvec_deg[cal], thd[cal], tr.tau_mm[cal], None,
                              {k: ("all", None) for k in TASKS}, min_n=10)
    assert len({id(f) for f in fits.values()}) == 1           # one shared fit
    f = fits["open_close"]
    assert f.model is not None and f.n_cal == int(cal.sum()) and f.note == ""
    assert f.model.lam == e4.lam and np.array_equal(f.model.coef, e4.coef)
    masks = {k: tr.mask(k) for k in TASKS}
    tau = cf.predict_by_task(fits, masks, tr.rotvec_deg, thd)
    m = tr.mask(*TASKS)
    assert np.array_equal(tau[m], e4.predict(tr.rotvec_deg, thd)[m])
    assert np.isnan(tau[~m]).all()                            # outside the tasks: no model


def test_x5c_classes_use_their_frames_and_give_reasons(misfit):
    tr = misfit
    cal = tr.mask("free_motion")
    thd = cf.theta_rate_dps(tr.t_s, tr.rotvec_deg[:, 1])
    rv, th, tau = tr.rotvec_deg[cal], thd[cal], tr.tau_mm[cal]
    codes = np.where(rv[:, 1] > np.median(rv[:, 1]), 1, 3)   # "OPEN" half, "PROTRUSION" half
    codes[:5] = 5                                             # a class with 5 frames
    spec = {"open_close": ("OPEN", (1,)), "chewing": ("OPEN", (1,)),
            "protrusion_retrusion": ("PROTRUSION", (3,)), "left_right": ("LEFT", (5,))}
    fits = cf.fit_task_models(rv, th, tau, codes, spec, min_n=50)
    assert fits["open_close"] is fits["chewing"]
    assert fits["open_close"].n_cal == int((codes == 1).sum())
    assert fits["protrusion_retrusion"].model is not None
    lr = fits["left_right"]
    assert lr.model is None and lr.n_cal == 5 and "< 50" in lr.note
    direct = cf.TranslationModel.fit(rv[codes == 3], th[codes == 3], tau[codes == 3])
    assert np.array_equal(fits["protrusion_retrusion"].model.coef, direct.coef)
    none = cf.fit_task_models(rv, th, tau, None, spec, min_n=50)
    assert all(f.model is None and "no optical labels" in f.note for f in none.values())
    tau_p = cf.predict_by_task(fits, {"left_right": tr.mask("left_right")}, tr.rotvec_deg, thd)
    assert np.isnan(tau_p).all()                              # no model: NaN, not E4own


def test_stage_label_helpers():
    from biomech.imu.stage_camerafree import labels_on_grid, x5c_task_labels
    from biomech.labels.auto_labels import CODE

    lab = {"free_motion__t_us": np.array([0, 25_000, 50_000]),
           "free_motion__label": np.array([1, -1, 3])}
    got = labels_on_grid(lab, "free_motion", np.array([-5, 0, 10_000, 25_000, 40_000, 90_000]))
    assert got.tolist() == [1, 1, -1, -1, 3, 3]
    assert labels_on_grid(lab, "rest", np.arange(3)) is None
    assert labels_on_grid(None, "free_motion", np.arange(3)) is None
    spec = x5c_task_labels({"open_close": ["OPEN", "CLOSE"], "chewing": None})
    assert spec == {"open_close": ("OPEN+CLOSE", (CODE["OPEN"], CODE["CLOSE"])),
                    "chewing": ("all", None)}
    with pytest.raises(ValueError, match="unknown label"):
        x5c_task_labels({"open_close": ["OPENING"]})


def _rows(est: str, base: float, pids: list[str]) -> list[dict]:
    group = {"open_close": "rotational", "chewing": "rotational",
             "protrusion_retrusion": "translational", "left_right": "translational"}
    out = []
    for i, pid in enumerate(pids):
        for j, t in enumerate(TASKS):
            out.append({"pid": pid, "estimator": est, "task": t, "scored": True,
                        "task_group": group[t], "rmse_inc_mm": base + 0.1 * i + 0.5 * j})
        out.append({"pid": pid, "estimator": est, "task": "scored", "scored": True,
                    "task_group": "all", "rmse_inc_mm": base + 1.0})
    return out


def test_cohort_oracle_tables():
    from biomech.stats.cohort_camerafree import oracle

    pids = [f"s{i:02d}" for i in range(1, 8)]
    dec = pd.DataFrame(_rows("X5b_model", 1.0, pids) + _rows("X5b_rotation", 2.0, pids)
                       + _rows("E4own", 3.0, pids))           # the copied E4own is ignored
    x5c = pd.DataFrame(_rows("X5c", 2.5, pids)
                       + [{"pid": "s01", "estimator": "X5c", "task": "left_right", "n": 0,
                           "scored": True, "task_group": "translational",
                           "note": "no model"}])
    tr = pd.DataFrame(_rows("E4own", 3.0, pids) + _rows("E4hinge", 9.0, pids))
    out = oracle(dec, x5c, tr, "E3b", seed=1, n_boot=200)
    part, coh, k = out["cf_oracle_participant"], out["cf_oracle_cohort"], \
        out["cf_oracle_contrasts"]
    assert set(part["estimator"]) == {"X5b_model", "X5b_rotation", "X5c", "E4own"}
    assert {"rmse_inc_mm__open_close", "rmse_inc_mm__left_right", "rmse_inc_mm_rot"} \
        <= set(part["outcome"])
    r = coh[(coh["estimator"] == "X5b_model") & (coh["outcome"] == "rmse_inc_mm__chewing")]
    assert r.iloc[0]["n"] == 7 and r.iloc[0]["median"] == pytest.approx(1.0 + 0.3 + 1.5)
    assert coh["verdict"].eq("").all() and coh["bound"].isna().all()   # descriptive
    assert set(coh["analysis"]) == {"X5b", "X5c", "E4own (reference)"}
    assert set(k["family"]) == {"X5b", "X5c"}
    x = k[(k["family"] == "X5c") & (k["outcome"] == "rmse_inc_mm_rot")].iloc[0]
    assert x["hl"] == pytest.approx(-0.5) and x["n_a_better"] == 7
    assert k.groupby("family")["p_holm"].apply(lambda p: (p >= 0).all()).all()
    empty = oracle(pd.DataFrame(), pd.DataFrame(), tr, "E3b", seed=1, n_boot=200)
    assert set(empty) == {"cf_oracle_cohort"} and "note" in empty["cf_oracle_cohort"]


def test_oracle_views_build():
    """The decomposition figure and table read the cohort tables; empty store -> a card."""
    from biomech.stats.cohort_camerafree import oracle
    from biomech.viz.views.camerafree import cf_decomposition, cf_table_oracle
    from biomech.viz.views.registry import ViewRequest

    pids = [f"s{i:02d}" for i in range(1, 8)]
    dec = pd.DataFrame(_rows("X5b_model", 1.0, pids) + _rows("X5b_rotation", 2.0, pids))
    out = oracle(dec, pd.DataFrame(_rows("X5c", 2.5, pids)),
                 pd.DataFrame(_rows("E4own", 3.0, pids)), "E3b", seed=1, n_boot=200)

    class Ctx:
        def __init__(self, tables):
            self.tables = tables

        def table(self, stage, name, pids=None):
            if (stage, name) not in self.tables:
                raise KeyError(name)
            return self.tables[(stage, name)]

    ctx = Ctx({("camerafree_cohort", k): v for k, v in out.items()})
    fig = cf_decomposition(ViewRequest(ctx=ctx))
    # tasks as rows (2026-11-15): one labelled median marker per combination, a median in
    # each of the five rows
    meds = [s for s in fig.panels[0].series if s.kind == "scatter" and s.role != "faint"]
    assert len([s for s in meds if s.label]) == 4 and len(meds) == 20
    assert all(np.isfinite(s.x).all() for s in meds)
    assert "n = 7" in fig.caption
    tab = cf_table_oracle(ViewRequest(ctx=ctx))
    rot = next(r for r in tab.rows if r["task"] == "open-close and chewing")
    assert rot["c_rot"].startswith("1.00") and rot["c_task"].startswith("\u22120.50")
    assert "X5b" not in tab.caption and tab.caption
    empty = cf_decomposition(ViewRequest(ctx=Ctx({})))
    # the card: only the message (a text series), no data series
    assert empty.caption and all(s.kind == "text" for p in empty.panels for s in p.series)
