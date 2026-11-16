# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Small dataset-infrastructure items (Part 3 step 8)."""

from __future__ import annotations

import yaml
from click.testing import CliRunner

from biomech.cohort import draft_manifest, load_cohort


# -- A16: cohort check lists session folders the manifest does not name ------------------------
def test_cohort_check_lists_unlisted_sessions(tmp_path, data_dir):
    from biomech.cli import main

    doc = draft_manifest(data_dir, "t")
    for row in doc["participants"]:                  # s02's second take: never assigned
        row["secondary"] = []
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    cohort = load_cohort(tmp_path / "cohort.yaml", data_dir)
    unlisted = [s.name for s in cohort.unlisted_sessions()]
    assert unlisted == ["session_s02_f_44_20250306T101000Z_00000005"]
    # a tagged comparison take stays invisible unless the manifest opts in
    tagged = data_dir / "session_s01_f_31_qualisys_comparison_20250320T100000Z_abcdef12"
    tagged.mkdir()
    (tagged / "session.h5").write_bytes(b"x")
    assert [s.name for s in cohort.unlisted_sessions()] == unlisted

    cfg = {"study": "t", "paths": {"data_dir": str(data_dir), "out_dir": "out",
                                   "cohort": "cohort.yaml"}}
    (tmp_path / "study.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    r = CliRunner().invoke(main, ["cohort", "check", "--config", str(tmp_path / "study.yaml")])
    assert r.exit_code == 0, r.output                  # informational, not a problem
    assert "UNLISTED s02: session_s02_f_44_20250306T101000Z_00000005 (motion)" in r.output
    assert "qualisys" not in r.output
    # listed as a secondary take: no longer unlisted
    for row in doc["participants"]:
        if row["pid"] == "s02":
            row["secondary"] = ["session_s02_f_44_20250306T101000Z_00000005"]
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    assert load_cohort(tmp_path / "cohort.yaml", data_dir).unlisted_sessions() == []


# -- reporting_population: a sensitivity variant that filters the cohort by manifest flag ---------------
def test_reporting_population_variant_excludes_flagged_participants(tmp_path, data_dir):
    from biomech import sensitivity as S
    from biomech.config import load_config

    doc = draft_manifest(data_dir, "t")
    for row in doc["participants"]:
        if row["pid"] == "s03":
            row["flags"] = ["outside_protocol", "reference_subset"]
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    cfg = {"study": "t", "paths": {"data_dir": str(data_dir), "out_dir": "out",
                                   "cohort": "cohort.yaml"}}
    (tmp_path / "study.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    base = load_config(tmp_path / "study.yaml")
    assert "reporting_population" in S.variants(base)
    v = S.variant_config(tmp_path / "study.yaml", "reporting_population")
    assert v.cohort_path == S.variant_dir(base, "reporting_population") / "cohort.yaml"
    assert v.out_dir == S.variant_dir(base, "reporting_population")
    primary = load_cohort(base.cohort_path, base.data_dir)
    variant = load_cohort(v.cohort_path, v.data_dir)
    assert [p.pid for p in primary.included()] == ["s01", "s02", "s03"]
    assert [p.pid for p in variant.included()] == ["s01", "s02"]
    s03 = variant.get("s03")
    assert s03.exclusion_reason == "sensitivity reporting_population: flag outside_protocol"
    assert s03.flags == ["outside_protocol", "reference_subset"] and variant.validate() == []
    # every other row (and session) unchanged, so participant units link from the primary
    assert [p for p in variant.participants if p.pid != "s03"] == \
        [p for p in primary.participants if p.pid != "s03"]
    # stage overrides are unchanged; the primary manifest is never touched
    assert v.stages == base.stages
    assert yaml.safe_load((tmp_path / "cohort.yaml").read_text()) == doc


# -- per-group code hash ------------------------------------------------------------------------
def test_group_code_hash_separates_dataset_and_validation_code():
    from pathlib import Path

    from biomech import pipeline as P

    rel = lambda fs: {f.as_posix().split("/software/")[-1] for f in fs}      # noqa: E731
    ds, val = rel(P.group_files("dataset")), rel(P.group_files("validation"))
    # validation-only code is not in the dataset hash ...
    assert not {f for f in ds if f.startswith("biomech/biomech/validation/")}
    assert "biomech/biomech/io/reference.py" not in ds
    assert "biomech/biomech/validation/stages.py" in val
    assert "biomech/biomech/io/reference.py" in val
    # ... dataset-only code (the ML plugin, the EMG/timing analysis) not in the validation one
    assert not {f for f in val if f.startswith("ml/")}
    assert "biomech/biomech/stages.py" in ds and "biomech/biomech/stages.py" not in val
    assert "biomech/biomech/timing/lead.py" in ds
    # shared code and data files are in both
    assert "biomech/biomech/kinematics/se3.py" in ds & val
    assert "biomech/biomech/norms/norms.yaml" in ds & val
    assert P.code_hash("dataset") != P.code_hash("validation") != P.code_hash()
    every = {f for _, f in P.hashed_files()}
    assert set(P.group_files("dataset")) <= every and set(P.group_files("validation")) <= every
    assert all(isinstance(f, Path) for f in P.group_files("dataset"))


def test_stage_keys_use_their_group_hash(monkeypatch, study):
    from biomech import pipeline as P
    from biomech.cohort import load_cohort

    cfg = study()
    cohort = load_cohort(cfg.cohort_path, cfg.data_dir)
    ds = next(n for n, s in P.registry().items()
              if s.group == "dataset" and s.scope == "participant" and not s.depends)

    def keys(h):
        monkeypatch.setattr(P, "_CODE_HASH", h)
        return P.KeyResolver(cfg, cohort).key(ds, "s01")

    base = keys({"dataset": "d0", "validation": "v0", None: "a0"})
    assert keys({"dataset": "d0", "validation": "v1", None: "a1"}) == base    # validation edit
    assert keys({"dataset": "d1", "validation": "v0", None: "a1"}) != base    # dataset edit
