# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sensitivity runs (biomech.sensitivity) and dotted config overrides (D14)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomech import sensitivity as S
from biomech.config import ConfigError, dotted_overrides, load_config
from biomech.results import COHORT_UNIT, ResultStore

STUDY = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027" / "study.yaml"
# the study's cohort file (age, sex, session names) is released only with the data
needs_cohort = pytest.mark.skipif(not (STUDY.parent / "cohort.yaml").exists(),
                                  reason="the study's cohort.yaml is not in this checkout")


def test_dotted_overrides_and_load_config(tmp_path):
    assert dotted_overrides({"a.b": 1, "a.c.d": 2}) == {"a": {"b": 1, "c": {"d": 2}}}
    cfg = load_config(STUDY, out_dir=tmp_path,
                      overrides={"signals.emg.mask.analysis_rules": ["rail", "lowfreq"]})
    assert cfg.signals["emg"]["mask"]["analysis_rules"] == ["rail", "lowfreq"]
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config(STUDY, out_dir=tmp_path, overrides={"signals.emg.nope": 1})
    with pytest.raises(ConfigError, match="expected"):
        load_config(STUDY, out_dir=tmp_path, overrides={"signals.emg.mask.dilate_s": "x"})


@needs_cohort
def test_variant_config_runs_into_its_own_dir(tmp_path):
    v = S.variant_config(STUDY, "lowfreq_mask", out_dir=tmp_path)
    assert v.out_dir == tmp_path / "sensitivity" / "lowfreq_mask"
    assert "lowfreq" in v.signals["emg"]["mask"]["analysis_rules"]
    base = load_config(STUDY, out_dir=tmp_path)
    assert "lowfreq" not in base.signals["emg"]["mask"]["analysis_rules"]     # D14 default
    assert v.params_hash("emg") != base.params_hash("emg")                     # own cache keys
    with pytest.raises(ConfigError, match="unknown sensitivity variant"):
        S.variant_config(STUDY, "nope", out_dir=tmp_path)


def test_compare_tables_keys_values_and_repeats():
    a = pd.DataFrame({"task": ["oc", "oc", "ch"], "muscle": ["m", "m", "m"],
                      "n": [10, 11, 5], "p": [0.01, 0.2, 0.5], "sig": [True, False, False]})
    b = a.copy()
    b.loc[0, "n"] = 12
    b.loc[2, "sig"] = True
    c = S.compare_tables(a, b)
    assert set(c.column) == {"n", "p", "sig"}
    row = c[(c.column == "n") & (c.key == "task=oc|muscle=m")].iloc[0]
    assert row.primary == 10 and row.variant == 12 and row["diff"] == 2
    assert (c.key == "task=oc|muscle=m|#1").any()                   # repeated key -> position
    assert c[(c.column == "sig") & (c.key == "task=ch|muscle=m")]["diff"].iloc[0] == 1
    # a row only in one version shows up with NaN on the other side
    d = S.compare_tables(a, b.iloc[:2])
    assert d[d.key == "task=ch|muscle=m"]["variant"].isna().all()


def test_compare_stores(tmp_path):
    p, v = ResultStore(tmp_path / "p"), ResultStore(tmp_path / "v")
    t = pd.DataFrame({"test": ["a", "b"], "n_clusters": [1, 0]})
    p.write("stats_emg", COHORT_UNIT, "k1", tables={"spm": t})
    v.write("stats_emg", COHORT_UNIT, "k2", tables={"spm": t.assign(n_clusters=[2, 0])})
    p.write("emg", "s01", "k", tables={"activation": t})            # participant tables skipped
    df = S.compare(p, v)
    assert set(df.stage) == {"stats_emg"} and set(df.table) == {"spm"}
    assert np.isclose(df[df.key == "test=a"]["diff"].iloc[0], 1.0)


def test_parameter_columns_are_keys_and_changed_text_is_not():
    a = pd.DataFrame({"task": ["oc"] * 2, "drop_db": [2.0, 3.0], "detected": [0.5, 0.9],
                      "clusters": ["[(0, 1)]", "[(5, 9)]"]})
    b = a.assign(detected=[0.6, 0.9], clusters=["[(0, 2)]", "[(5, 9)]"])
    c = S.compare_tables(a, b)
    assert set(c.column) == {"detected"}
    r = c.set_index("key")
    assert r.loc["task=oc|drop_db=2.0", "diff"] == pytest.approx(0.1)
    assert r.loc["task=oc|drop_db=3.0", "diff"] == 0


@needs_cohort
def test_skip_stages_closure_and_comparison(tmp_path):
    """D22: a variant's skip_stages drops those stages and everything downstream from its run
    and from its comparison (even older variant units), and the comparison records them."""
    import yaml

    from biomech.pipeline import group_stages

    base = load_config(STUDY, out_dir=tmp_path)
    skip = ["camerafree", "camerafree_cohort", "imu_dyncal", "imu_dyncal_cohort", "ml_windows",
            "ml_eval", "ml_online"]
    assert S.skipped_stages(base, "lowpass_8hz") == skip
    assert S.skipped_stages(base, "lowfreq_mask") == skip
    assert S.skipped_stages(base, "labels_p95") == ["camerafree", "camerafree_cohort",
                                                     "imu_dyncal", "imu_dyncal_cohort"]
    assert S.skipped_stages(base, "reporting_population") == [] and S.variant_stages(
        base, "reporting_population") is None
    run = S.variant_stages(base, "lowpass_8hz")
    assert "labels" in run and "stats_timing" in run and not {"ml_eval", "ml_online"} & set(run)
    assert set(run) | set(skip) == set(group_stages(base.groups))
    # unknown stage names fail when the variant is loaded
    doc = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    doc["sensitivity"]["lowpass_8hz"]["skip_stages"] = ["ml_evl"]
    doc["paths"] = {k: str((STUDY.parent / v).resolve()) for k, v in doc["paths"].items()}
    bad = tmp_path / "study.yaml"
    bad.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown skip_stages"):
        S.variant_config(bad, "lowpass_8hz", out_dir=tmp_path)
    # the comparison leaves a stale ml_eval unit of the variant out and records the skip;
    # lowpass_8hz builds on reporting_population (D36), so it compares with that store
    t = pd.DataFrame({"test": ["a"], "median": [1.0]})
    p = ResultStore(S.base_config(base, "lowpass_8hz").out_dir)
    v = ResultStore(S.variant_dir(base, "lowpass_8hz"))
    for st in ("stats_timing", "ml_eval"):
        p.write(st, COHORT_UNIT, "k1", tables={"t": t})
        v.write(st, COHORT_UNIT, "k0", tables={"t": t.assign(median=[2.0])})
    S.write_comparison(base, "lowpass_8hz")
    c = S.read_comparison(tmp_path, "lowpass_8hz")
    assert set(c.stage) == {"stats_timing"}
    assert S.not_run(c) == skip
