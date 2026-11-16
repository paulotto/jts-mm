# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""plan-code families, value vocabularies, reader-text checks, catalog
misses, paper sets (no store needed)."""

from __future__ import annotations

import pytest

from biomech import catalog, papers, pipeline
from biomech.check.text import dropped_code_issues, text_issues
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V


def test_code_families_cover_validation_codes_and_spare_names():
    assert V.internal_codes("VD3 offsets (EV1, V8) after D6 and X2; A4b, R6") == [
        "VD3", "EV1", "V8", "D6", "X2", "A4b", "R6"]
    # pseudonyms, percentiles and estimator names are not codes
    assert V.internal_codes("P3 at P95, F2 and F2P_causal") == []
    assert V.reader_text("deviation VD3") == "post hoc"


def test_vocabulary_terms_and_misses():
    V.unknown_terms(reset=True)
    assert V.term("task", "protrusion_retrusion") == "protrusion"
    assert V.term("task", "open_close", "title") == "Open–close"
    assert V.term("rule", "drive_taskbase") == "driving burst, in-task threshold"   # comma kept
    assert V.term("estimator", "F2P") == "Kalman smoother with position fusion"
    assert V.term("task", "no_such_task") == "no such task"
    assert ("task", "no_such_task") in V.unknown_terms(reset=True)
    assert V.unknown_terms() == set()
    # every domain entry has its three forms
    for dom, entries in V.vocabulary()[0].items():
        for k, e in entries.items():
            assert set(e) == {"label", "short", "title"}, (dom, k)


def test_reader_cells():
    assert V.cell("segment", "movement_pooled") == "all movement tasks"
    assert V.cell("transform", "A_maxillary") == "maxillary frame offset"
    assert V.cell("feature", "amp_mm") == "Incisal amplitude"          # a column name -> label
    assert V.cell("pid", "P1") == "P1"                                 # no domain: unchanged
    assert V.cell("segment", 3.0) == 3.0
    assert B.reader_cell("spm_h0reject", True) == "yes"
    assert B.reader_cell("adequate", False) == "no"


def test_text_issues_find_codes_identifiers_and_raw_values():
    p = Panel("(a) False releases", Axis("M_Q error σ_ref", "mm"), Axis("y", ""),
              [Series("line", [0, 1], [0, 1], label="f95_coh_hz"),
               Series("line", [0, 1], [1, 0], label="nan")])
    f = FigureSpec("x", "t", [p], caption="as in VD3; CMC n/a")
    iss = " | ".join(text_issues(f))
    assert "VD3" in iss and "f95_coh_hz" in iss and "n/a" in iss and "nan" in iss
    assert "False" not in iss and "M_Q" not in iss
    t = TableSpec("t", "clean caption", [Column("a", "A")], [{"a": "True"}, {"a": "fine"}])
    assert any("True" in i for i in text_issues(t))
    ok = FigureSpec("y", "t", [Panel("(a) Opening", Axis("Time", "s"), Axis("Angle", "deg"), [])],
                    caption="Median and 95 % CI per participant.")
    assert text_issues(ok) == []


def test_dropped_codes_that_empty_a_label_or_gap_a_sentence_are_reported():
    p = Panel("(a)", Axis("RMSE", "deg"), Axis(""), [])
    p.y.ticks, p.y.ticklabels = [0.0, 1.0], ["E3b", "own model"]
    raw = FigureSpec("x", "t", [p], caption="the slope of E2 against time (E27)")
    iss = " | ".join(dropped_code_issues(raw, V.reader_spec(raw)))
    assert "emptied" in iss and "E3b" in iss and "mid-sentence" in iss and "E2" in iss
    assert "E27" not in iss                   # a parenthetical code drops cleanly
    t = TableSpec("t", "rungs", [Column("r", "Rung")], [{"r": "E3b"}, {"r": "own model"}])
    assert any("E3b" in i for i in dropped_code_issues(t, V.reader_spec(t)))
    ok = FigureSpec("y", "t", [Panel("(a)", Axis("x"), Axis("y"), [])],
                    caption="closed-pose resets against the optics (E27)")
    assert dropped_code_issues(ok, V.reader_spec(ok)) == []


def test_catalog_records_misses():
    catalog.misses(reset=True)
    assert catalog.get("incisor_rms_mm").label == "Incisor RMS difference"
    assert catalog.misses() == set()
    catalog.get("no_such_column_mm")
    assert catalog.misses(reset=True) == {"no_such_column_mm"}


def test_reader_wording_files_do_not_change_results():
    for f in ("codes.yaml", "vocab.yaml", "glossary.yaml"):
        assert pipeline._is_presentation(("etc", f))
    assert pipeline._is_presentation(("papers.py",))


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_paper_set_targets_labels_and_legacy(tmp_path):
    _write(tmp_path, "study.yaml", "study: x\n")
    p = _write(tmp_path, "paper_a.yaml", """
snapshot: r7
targets:
  journal: {venue: springer, out_dir: out/journal}
  conference: {venue: ieee, out_dir: out/conf}
default_target: journal
items:
  - {id: icorr_hinge, label: hinge, in: [journal, conference]}
  - {id: jnl_c2, part: supporting, in: [journal]}
  - {id: val_kin_waveforms, pid: s10, phase: open_close}
  - {id: val_kin_waveforms, pid: s14, phase: open_close}
""")
    _write(tmp_path, "paper_old.yaml", "status: legacy\nvenue: ieee\nitems: [icorr_hinge]\n")
    ps = papers.load(p)
    assert ps.snapshot == "r7" and ps.target().venue == "springer"
    assert ps.target("conference").out_dir == (tmp_path / "out/conf").resolve()
    # items without `in:` belong to every target
    assert [i.label for i in ps.items_for("conference")] == [
        "hinge", "valKinWaveformsOpenClose", "valKinWaveformsOpenCloseB"]
    assert len(ps.items_for("journal")) == 4
    labels = [i.label for i in ps.items]
    assert labels[1] == "jnlCTwo" and ps.items[1].part == "supporting"
    # the participant never enters a label; repeats are told apart by order
    assert labels[2:] == ["valKinWaveformsOpenClose", "valKinWaveformsOpenCloseB"]
    assert all("s10" not in lb and "One" not in lb for lb in labels)
    assert ps.items[2].entry() == {"id": "val_kin_waveforms", "pid": "s10", "phase": "open_close",
                                   "_label": "valKinWaveformsOpenClose", "_part": "core",
                                   "_section": ""}
    assert [s.name for s in papers.study_paper_sets(tmp_path / "study.yaml")] == ["paper_a"]


def test_paper_set_rejects_bad_labels_and_targets(tmp_path):
    bad = _write(tmp_path, "paper_b.yaml", "items: [{id: a, label: fig1}]\n")
    with pytest.raises(ValueError, match="letters only"):
        papers.load(bad)
    bad = _write(tmp_path, "paper_c.yaml", "items: [{id: a, in: [nowhere]}]\n")
    with pytest.raises(ValueError, match="unknown target"):
        papers.load(bad)
    old = papers.load(_write(tmp_path, "paper_d.yaml", "venue: ieee\nout_dir: x\nitems: [a]\n"))
    assert list(old.targets) == ["default"] and old.target().venue == "ieee"


def test_text_gaps_collect_the_full_lists(tmp_path):
    import json

    from biomech.check.outputs import GAPS, Report, write_text_gaps

    rep = Report("s")
    rep.add("text", "v1: reader text", "warn", "labels guessed: a, b")
    rep.add(GAPS, "v1", "pass", json.dumps({"labels": [f"c{i}" for i in range(20)],
                                             "values": ["task:x"]}))
    rep.add(GAPS, "v2", "pass", json.dumps({"labels": ["c0"], "values": []}))
    doc = write_text_gaps(rep, tmp_path / "g.json")
    assert len(doc["labels"]) == 20 and doc["labels"]["c0"] == ["v1", "v2"]
    assert doc["values"] == {"task:x": ["v1"]} and doc["identifiers"] == {}
    assert [i.area for i in rep.items] == ["text"]          # internal rows removed
    assert json.loads((tmp_path / "g.json").read_text()) == doc


def test_columns_with_several_domains_and_composite_values():
    from biomech.viz.views import vocab as V

    V.unknown_terms(reset=True)
    assert V.cell("variant", "ct") == "CT only (primary)"                     # clock variant
    assert V.cell("variant", "loto").startswith("leave one task out")       # alignment
    assert V.cell("channel", "incisor_x").startswith("incisor x")
    assert V.cell("channel", "masseter_r") == "masseter R"
    assert V.cell("variant", "cycle_start|3d|lowpass") == \
        "cycle-start reference, 3-D displacement, low-passed"
    assert V.cell("variant", "norm=max1s|onset=drive") == \
        "normalization: largest 1-s mean; onset rule: driving burst"
    assert V.cell("flags", "edge,no_return") == "cut by a phase edge, no return to rest"
    assert V.unknown_terms(reset=True) == set()
    assert V.cell("variant", "nonsense_x") == "nonsense x"
    assert ("sensitivity_variant", "nonsense_x") in V.unknown_terms(reset=True)


def test_raw_identifiers_are_located_by_column():
    from biomech.check.text import raw_id_cells
    from biomech.viz.spec import Column, TableSpec

    t = TableSpec("t", "", [Column("a", "A"), Column("b", "B")],
                  [{"a": "open_close", "b": "fine"}, {"a": "x", "b": "two words"}])
    assert raw_id_cells(t) == {"a": ["open_close"]}


def test_pivoted_feature_names_are_composed_and_reader_names_pass():
    from biomech import catalog
    from biomech.viz.views import vocab as V

    catalog.misses(reset=True)
    assert catalog.get("masseter_r|log_rms").label == "Masseter R, log RMS"
    assert catalog.get("a_cra_x|mean").label == "Head acceleration x, mean"
    assert catalog.get("digastric_l|mnf").unit == "\\hertz"
    assert catalog.misses(reset=True) == set()
    catalog.get("nonsense|x")
    assert catalog.misses(reset=True) == {"nonsense|x"}
    V.unknown_terms(reset=True)
    assert V.cell("muscle", "masseter R") == "masseter R"       # translated by a view
    lab = catalog.get("rest_pct").label
    assert V.cell("metric", lab) == lab                           # a catalog label as value
    assert V.cell("estimator", "rms_deg") == catalog.get("rms_deg").label
    assert V.unknown_terms(reset=True) == set() and catalog.misses(reset=True) == set()


def test_identifiers_inside_text_cells_are_rewritten():
    from biomech.viz.views import vocab as V

    assert V.cell("outcome", "median_pct|open_close|masseter_r") == \
        "Median activation, open-close, masseter R"
    assert V.cell("test", "emg_imu_vs_imu") == "EMG + IMU vs IMU"
    assert V.cell("p3", "too_close") == V.term("condyle_reason", "too_close")
    assert V.cell("test", "anova1_x_y") == "anova1_x_y"          # unknown: left for the check


def test_redacted_booleans_read_yes_no_and_mixed_column_domains():
    import pandas as pd

    d = B.redact(pd.DataFrame({"flag": [True, None, False], "note": ["a", "b", None]}))
    assert list(d["flag"]) == [True, None, False]       # not the strings "True" / "False"
    assert [B.reader_cell("flag", v) for v in d["flag"]][::2] == ["yes", "no"]
    # a column whose values are column names, else another domain (key: column, c2e rows)
    assert V.cell("key", "rmse_inc_mm") == catalog.get("rmse_inc_mm").label
    assert V.cell("key", "h1") == V.term("c2e_hypothesis", "h1")
    catalog.misses(reset=True)
    V.cell("key", "not_a_column")
    assert ("c2e_hypothesis", "not_a_column") in V.unknown_terms(reset=True)
