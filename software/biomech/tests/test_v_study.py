# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The validation study's plumbing: tagged comparison takes in the cohort (opt-in only),
stage groups, and the ref_sync stage on a synthetic take written in the real file formats."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from biomech.cohort import discover_sessions, draft_manifest, load_cohort, parse_session_name
from biomech.pipeline import get_stage, resolve_order

CMP = "session_s10_m_38_qualisys_comparison_20250314T100000Z_00000009"
MOTION = "session_s10_m_38_20250307T101000Z_0000000a"
MVC = "session_s10_m_38_mvc_20250307T100000Z_0000000b"


def _tree(root: Path, names: list[str]) -> Path:
    for n in names:
        (root / n).mkdir(parents=True)
        (root / n / "session.h5").write_bytes(b"x")
    return root


# -- cohort: comparison takes are opt-in --------------------------------------------------
def test_tagged_take_invisible_by_default():
    assert parse_session_name(CMP) is None
    info = parse_session_name(CMP, ["qualisys_comparison"])
    assert info["pid"] == "s10" and info["kind"] == "motion"
    assert info["tag"] == "qualisys_comparison"
    plain = parse_session_name(MOTION, ["qualisys_comparison"])
    assert "tag" not in plain                                # untagged parse unchanged


def test_dataset_discovery_and_draft_unchanged(tmp_path: Path):
    """icorr2027 behaviour: a comparison take next to the dataset takes is ignored."""
    with_cmp = _tree(tmp_path / "a", [MOTION, MVC, CMP])
    without = _tree(tmp_path / "b", [MOTION, MVC])
    assert [s.name for s in discover_sessions(with_cmp)] == [s.name for s in
                                                             discover_sessions(without)]
    assert draft_manifest(with_cmp, "x") == draft_manifest(without, "x")
    found = discover_sessions(with_cmp, tags=["qualisys_comparison"])
    assert {s.name for s in found} == {MOTION, MVC, CMP}
    assert next(s for s in found if s.name == CMP).tag == "qualisys_comparison"


def _manifest(tmp_path: Path, **top) -> Path:
    doc = {"study": "v", **top, "participants": [
        {"pid": "s10", "sex": "m", "age": 38, "motion": CMP, "mvc": None, "include": True}]}
    p = tmp_path / "cohort.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return p


def test_validate_requires_opt_in(tmp_path: Path):
    data = _tree(tmp_path / "d", [CMP])
    problems = load_cohort(_manifest(tmp_path), data).validate()
    assert any("qualisys_comparison take" in x and "session_tags" in x for x in problems)
    assert any("no mvc session" in x for x in problems)       # require_mvc defaults to true
    ok = load_cohort(_manifest(tmp_path, session_tags=["qualisys_comparison"],
                               require_mvc=False), data)
    assert ok.validate() == []
    with pytest.raises(ValueError, match="unknown session_tags"):
        load_cohort(_manifest(tmp_path, session_tags=["typo"]), data)


def test_repo_validation_cohort_parses():
    root = Path(__file__).resolve().parents[3] / "experiments" / "jtsmm_validation"
    if not (root / "cohort.yaml").is_file():
        pytest.skip("validation study not in this checkout")
    c = load_cohort(root / "cohort.yaml", root)                 # data dir irrelevant here
    assert [p.pid for p in c.included()] == ["s10", "s14"]
    assert c.session_tags == ("qualisys_comparison",) and c.require_mvc is False
    problems = [x for x in c.validate() if "missing" not in x]  # sessions are not in git
    assert problems == []
    ref = yaml.safe_load((root / "reference.yaml").read_text(encoding="utf-8"))
    assert set(ref["participants"]) == {"s10", "s14"}
    assert ref["bodies"] == {"CT": "ct", "MP": "mta", "HP": "cra"}


# -- stage groups -------------------------------------------------------------------------
def test_ref_sync_only_in_validation_group():
    st = get_stage("ref_sync")
    assert st.group == "validation" and st.scope == "participant"
    assert "ref_sync" not in resolve_order(None, ("dataset",))
    order = resolve_order(None, ("validation",))
    core = ["ref_sync", "ref_tip", "ref_quality", "ref_kinematics", "ref_agreement"]
    assert [s for s in order if s in core] == core          # present, in dependency order
    assert not set(order) & set(resolve_order(None, ("dataset",)))
    with pytest.raises(ValueError, match="pipeline.groups"):
        resolve_order(["ref_sync"], ("dataset",))


# -- the stage's work on synthetic files in the real formats -------------------------------
def test_run_ref_sync_on_synthetic_files(tmp_path: Path):
    from biomech.validation.stages import participant_reference, run_ref_sync
    from biomech.validation.synthetic import make_take, write_qtm_mat

    tk = make_take(seed=2, duration=240.0)
    take_dir = tmp_path / CMP
    (take_dir / "qualisys_emg").mkdir(parents=True)
    n = len(tk.t_q_s)
    start_frame = int(round(tk.t_q_s[0] * 200.0)) + 1           # QTM time = (frame-1)/fs
    write_qtm_mat(take_dir / "qualisys_emg" / "take.mat", "take_01", 200.0,
                  {"CT": tk.Q["CT"], "MP": tk.Q["MP"], "EvalArraz": np.zeros((n, 4, 4)),
                   "HP": tk.Q["HP"]}, start_frame=start_frame)
    doc = {"participants": {"s10": {"qtm_mat": "qualisys_emg/take.mat",
                                    "lag_hint_s": tk.model.offset_s + 1.0}}}
    ref = participant_reference(doc, "s10", take_dir)
    optical = SimpleNamespace(t_us=np.round(tk.t_trig_s * 1e6).astype(np.uint64),
                              poses={"ct": tk.S["CT"], "mta": tk.S["MP"], "cra": tk.S["HP"]})
    session = SimpleNamespace(optical=lambda: optical, phases=tk.phases)
    tables, arrays = run_ref_sync(session, ref, {"CT": "ct", "MP": "mta", "HP": "cra"}, {},
                                  {"sigma_pos_mm": 0.2, "sigma_rot_deg": 0.1,
                                   "sliding_variant": "both"}, "s10")
    sync = tables["sync"].iloc[0]
    true = tk.model.recentred(sync["t_ref_s"])
    assert abs(sync["offset_s"] - true.offset_s) < 0.5e-3
    assert abs(sync["alpha_ppm"] - 30.0) < 2.0
    assert sync["heldout_mp_pos_rms_mm"] < 0.3 and sync["missing_bodies"] == ""
    assert set(tables) == {"sync", "sync_methods", "registration", "heldout", "sliding_lag",
                           "windows", "clock_variants", "clock_criteria"}
    assert {"ct", "joint", "two_stage", "hybrid", "tukey"} <= set(tables["clock_variants"].variant)
    assert sync["clock_primary"] in ("ct", "two_stage")
    # VD19: HAC clock SEs next to the pre-VD19 AR(1) ones; the sliding-slope time models are
    # variants (re-scored in ref_kinematics:robustness) that never choose the clock
    assert np.isfinite(sync["se_alpha_ppm_hac"]) and sync["n_windows_hac"] == 2
    assert 0.2 < sync["se_alpha_ppm_hac"] / sync["se_alpha_ppm"] < 5.0
    assert {"se_alpha_ppm_ar1", "se_offset_ms_ar1", "se_inflation_ar1"} <= set(sync.index)
    cv = tables["clock_variants"].set_index("variant")
    assert {"sliding_omega", "sliding_pose"} <= set(cv.index)
    assert abs(cv.loc["sliding_pose", "alpha_ppm"] - 30.0) < 3.0
    assert np.isfinite(cv.loc["ct", "se_alpha_ppm_hac"])
    cc = tables["clock_criteria"].set_index("variant")
    assert {"pass", "pass_hac", "window_diff_se_ms_ar1", "window_diff_se_ms_hac"} <= set(cc.columns)
    reg = tables["registration"].set_index("transform")
    assert list(reg.index) == ["X", "Y_CT", "Y_MP", "Y_HP"]
    # R5b: the clock and X come from the CT alone; Y_MP / Y_HP are closed-form afterwards
    # (no SE here; their uncertainty is the leave-one-task-out spread in ref_kinematics)
    assert (reg.loc[["X", "Y_CT"], "se_trans_mm"] < 0.1).all()
    assert reg.loc[["Y_MP", "Y_HP"], "se_trans_mm"].isna().all()
    assert abs(sync["joint_d_offset_ms"]) < 0.5 and not sync["joint_disagrees"]
    # sync-view arrays (R5b)
    for key in ("ct_visible_s", "ct_visible_q", "omega_s", "omega_q", "xcorr_sync_start_rho"):
        assert key in arrays, key
    assert len(arrays["xcorr_sync_start_rho"]) == len(arrays["xcorr_sync_start_lags_s"])
    assert {"edges", "fine", "sliding_pose", "sliding_omega"} <= set(
        tables["sync_methods"]["method"])
    assert arrays["edge_pairs"].shape[1] == 3 and len(arrays["sliding_t_s"]) > 10
    for t in tables.values():
        assert (t["pid"] == "s10").all()
