# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""infrastructure: per-target item parameters, the requirements-table scope and
evidence labels, cited / mean numbers and their n, and the n checks of check-outputs (stub
contexts; no store needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from biomech import papers
from biomech.check import integrity as I
from biomech.check import outputs as C
from biomech.viz import numbers as N
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec


# -- paper sets: target_params ------------------------------------------------------------------
PAPER = """\
snapshot: r9
targets:
  journal: {venue: springer, out_dir: j}
  conference: {venue: ieee, out_dir: c}
items:
  - {section: "2 Requirements"}
  - {id: table_requirements, label: tabRequirements, in: [journal, conference], target_params: {conference: {scope: conference}}}   # note
  - {id: icorr_hinge, label: hinge}
"""


def test_target_params_apply_per_target(tmp_path):
    p = tmp_path / "paper_x.yaml"
    p.write_text(PAPER)
    ps = papers.load(p)
    j = {i.view_id: i.params for i in ps.items_for("journal")}
    c = {i.view_id: i.params for i in ps.items_for("conference")}
    assert j["table_requirements"] == {} and c["table_requirements"] == {"scope": "conference"}
    assert [i.entry()["_label"] for i in ps.items_for("conference")] == ["tabRequirements",
                                                                          "hinge"]
    # every distinct parameter set once (the checks build each)
    assert [(i.view_id, i.params) for i in ps.variants()] == [
        ("table_requirements", {}), ("table_requirements", {"scope": "conference"}),
        ("icorr_hinge", {})]
    # the gallery edit keeps the per-target parameters and the comment
    new = papers.set_item_param(p, 0, "form", "bar")
    assert new.items[0].target_params == {"conference": {"scope": "conference"}}
    assert "# note" in p.read_text()


@pytest.mark.parametrize("bad, msg", [
    ("target_params: {poster: {scope: x}}", "unknown target"),
    ("in: [journal], target_params: {conference: {scope: x}}", "not in"),
    ("target_params: {conference: {label: x}}", "only set view parameters"),
])
def test_target_params_errors(tmp_path, bad, msg):
    p = tmp_path / "paper_x.yaml"
    p.write_text(PAPER.replace("in: [journal, conference], target_params: "
                               "{conference: {scope: conference}}", bad))
    with pytest.raises(ValueError, match=msg):
        papers.load(p)


# -- table_requirements: scope and evidence labels ----------------------------------------------
class ReqCtx:
    def __init__(self, p_first=0.023, p_drive=0.23):
        rows = [("opening_mm", "workspace", "descriptive", 14),
                ("misalign_hinge_fit_mm", "kinematics", "confirmatory", 14),
                ("peak_vel_dps", "dynamics", "descriptive", 14),
                ("lead_first_ms", "intent", "confirmatory", 9),
                ("lead_drive_ms", "intent", "confirmatory_posthoc_rule", 8),
                ("closer_release_ms", "intent", "exploratory", 10),
                ("imu_opening_loa_subject_mm", "sensing", "descriptive", 13),
                ("imu_opening_loa_loso_mm", "sensing", "descriptive", 13)]
        self.req = pd.DataFrame([{"key": k, "group": g, "requirement": k, "value": 1.0,
                                  "ci_lo": 0.5, "ci_hi": 1.5, "median": 1.0, "unit": "mm",
                                  "statistic": "median", "n": n, "derivation": "d",
                                  "evidence": e, "implication": "", "note": ""}
                                 for k, g, e, n in rows])
        self.timing = pd.DataFrame({"task": "open_close", "direction": "opening",
                                    "detector": "tkeo", "metric": "lead_ms",
                                    "rule": ["first", "drive"], "p_holm": [p_first, p_drive]})
        self.store = SimpleNamespace(stages=lambda: ["requirements", "stats_timing"])

    def table(self, stage, name):
        return {"requirements": self.req, "stats_timing": self.timing}[stage].copy()

    def pids(self):
        return [f"s{i:02d}" for i in range(1, 15)]


def _req_rows(ctx, **opt):
    from biomech.viz.views import registry as R

    t = R.build("table_requirements", R.request(ctx, "table_requirements", **opt))
    return {r["req"]: r["evidence"] for r in t.rows if not r.get("_rule") and r["evidence"]}, t


def test_requirements_scope_and_evidence():
    full, _ = _req_rows(ReqCtx())
    conf, t = _req_rows(ReqCtx(), scope="conference")
    # 2026-11-10 (JNER revision 4.1, 4.4): evidence in the agreed words, release rows named
    # by method
    assert "lead_first_ms" in full
    assert full["Closer release before opening, average envelope"] == "exploratory (post hoc)"
    assert set(conf) == {"opening_mm", "misalign_hinge_fit_mm", "peak_vel_dps",
                         "imu_opening_loa_subject_mm", "imu_opening_loa_loso_mm"}
    assert full["lead_first_ms"] == "not confirmed"
    assert full["lead_drive_ms"] == "not confirmed (post hoc onset rule)"
    assert full["imu_opening_loa_loso_mm"] == "confirmatory"
    assert full["misalign_hinge_fit_mm"] == "confirmatory"
    assert "pre-specified" not in " ".join(full.values())
    assert t.meta["caption"]["n"] == "n = 14 participants"


def test_c2_not_confirmed_refused_when_both_rules_significant():
    with pytest.raises(ValueError, match="both onset rules"):
        _req_rows(ReqCtx(p_first=0.01, p_drive=0.02))


# -- numbers: n, mean_of, cite ------------------------------------------------------------------
class Ctx:
    def __init__(self, tables, pids=()):
        self.tables = tables
        self._pids = list(pids)

    def table(self, stage, name):
        return self.tables.get(f"{stage}:{name}", pd.DataFrame()).copy()

    def pids(self):
        return self._pids


SEG = pd.DataFrame({"pid": ["s10", "s10", "s14", "s14"], "task": ["a", "b", "a", "b"],
                    "rms_mm": [1.0, 3.0, 2.0, 4.0], "n": [50, 60, 70, 80]})
SYS = pd.DataFrame({"pid": ["s10", "s14"], "jitter_mm": [0.78, 0.70]})


def test_n_forms_and_aggregate_counts_participants():
    ctx = Ctx({"ref:seg": SEG})
    decl = {"aggMed": {"source": "ref:seg", "column": "rms_mm", "agg": "median"},
            "oneInt": {"source": "ref:seg", "column": "rms_mm",
                       "filter": {"pid": "s10", "task": "a"}, "n": 1},
            "distinct": {"source": "ref:seg", "column": "rms_mm", "filter": {"task": "a"},
                         "agg": "mean", "n": {"distinct": "pid"}},
            "cycles": {"source": "ref:seg", "column": "rms_mm",
                       "filter": {"pid": "s14", "task": "b"}, "n": "n", "n_of": "cycles"}}
    got = {n.key: n for n in N.resolve(ctx, decl)}
    assert got["aggMed"].n == 2 and got["aggMed"].n_rows == 4      # participants, not rows
    assert got["oneInt"].n == 1 and got["distinct"].n == 2
    assert got["cycles"].n == 80 and got["cycles"].n_of == "cycles"


def test_mean_of_is_the_descriptive_mean_of_two_participants():
    ctx = Ctx({"ref:seg": SEG})
    decl = {"rmsPOne": {"source": "ref:seg", "column": "rms_mm",
                        "filter": {"pid": "s10", "task": "a"}, "fmt": ".2f",
                        "unit": "\\milli\\metre"},
            "rmsPTwo": {"source": "ref:seg", "column": "rms_mm",
                        "filter": {"pid": "s14", "task": "a"}, "fmt": ".2f",
                        "unit": "\\milli\\metre"},
            "rmsMean": {"mean_of": ["rmsPOne", "rmsPTwo"], "text": "mean of P1, P2"}}
    got = {n.key: n for n in N.resolve(ctx, decl)}
    m = got["rmsMean"]
    assert m.value == pytest.approx(1.5) and m.text == "1.50" and m.n == 2
    assert m.unit == "\\milli\\metre" and m.source == "mean_of:rmsPOne,rmsPTwo"
    with pytest.raises(ValueError, match="not declared"):
        N.resolve(ctx, {**decl, "rmsMean": {"mean_of": ["rmsPOne", "nope"]}})
    with pytest.raises(ValueError, match="different units"):
        N.resolve(ctx, {**decl, "x": {"source": "ref:seg", "column": "rms_mm",
                                      "filter": {"pid": "s10", "task": "b"}, "unit": ""},
                        "rmsMean": {"mean_of": ["rmsPOne", "x"]}})


def _cite_files(tmp_path, snapshot="v7"):
    sysdir = tmp_path / "jtsmm_validation"
    sysdir.mkdir()
    (sysdir / "study.yaml").write_text("study: v\n")
    (sysdir / "numbers_systems.yaml").write_text(yaml.safe_dump({
        "jitterOptical": {"source": "ref_fusion:smoothness", "column": "jitter_mm",
                          "agg": "median", "fmt": ".2f", "unit": "\\milli\\metre"}}))
    (sysdir / "paper_systems.yaml").write_text("snapshot: v7\nnumbers: numbers_systems.yaml\n"
                                               "items: []\n")
    dsdir = tmp_path / "icorr2027"
    dsdir.mkdir()
    decl = dsdir / "numbers_dataset.yaml"
    decl.write_text(yaml.safe_dump({
        "reqOpening": {"source": "requirements:requirements", "column": "value",
                       "filter": {"key": "opening_mm"}, "n": "n", "fmt": ".1f"},
        "cite": {"systems": {"study": "../jtsmm_validation/study.yaml", "snapshot": snapshot,
                             "label": "systems paper",
                             "from": "../jtsmm_validation/numbers_systems.yaml",
                             "numbers": {"jitterOptical": "same",
                                         "jitterOne": {"source": "ref_fusion:smoothness",
                                                       "column": "jitter_mm",
                                                       "filter": {"pid": "s10"},
                                                       "fmt": ".2f"}}}}}))
    return decl, sysdir


REQ = pd.DataFrame({"key": ["opening_mm"], "value": [39.69], "n": [14]})


def test_cited_numbers_resolve_write_and_verify(tmp_path):
    decl, sysdir = _cite_files(tmp_path)
    loaded = N.load(decl)
    assert loaded["jitterOptical"]["_cite"]["study"] == str(sysdir / "study.yaml")
    assert loaded["jitterOptical"]["column"] == "jitter_mm"          # "same": reused
    other = Ctx({"ref_fusion:smoothness": SYS}, pids=["s10", "s14"])
    opened = []

    def opener(study, snap):
        opened.append((study, snap))
        return other

    ctx = Ctx({"requirements:requirements": REQ})
    nums = N.resolve(ctx, loaded, opener)
    assert opened == [(str(sysdir / "study.yaml"), "v7")]            # one context per source
    got = {n.key: n for n in nums}
    assert got["jitterOptical"].text == "0.74" and got["jitterOptical"].n == 2
    assert got["jitterOptical"].cite["label"] == "systems paper"
    jp, tp = N.write(tmp_path / "out", nums, {"snapshot": "r9"}, decl)
    doc = json.loads(jp.read_text())
    assert set(doc["numbers"]) == {"reqOpening"}
    assert set(doc["cited"]) == {"jitterOptical", "jitterOne"}
    assert doc["cited"]["jitterOne"]["cited"] is True
    assert doc["cited"]["jitterOne"]["snapshot"] == "v7"
    tex = tp.read_text()
    assert "\\newcommand{\\numJitterOptical}" in tex and "Cited from the systems paper" in tex
    assert N.verify(tmp_path / "out", ctx, opener) == []
    # the cited store changed: the export is stale
    other.tables["ref_fusion:smoothness"] = SYS.assign(jitter_mm=[0.9, 0.8])
    probs = N.verify(tmp_path / "out", ctx, opener)
    assert any("jitterOptical" in p for p in probs)


def test_cited_snapshot_must_be_the_pinned_one(tmp_path):
    decl, sysdir = _cite_files(tmp_path, snapshot="v6")
    other = Ctx({"ref_fusion:smoothness": SYS})
    nums = N.resolve(Ctx({"requirements:requirements": REQ}), N.load(decl),
                     lambda s, n: other)
    N.write(tmp_path / "out", nums, {"snapshot": "r9"}, decl)
    probs = I.verify_cited(tmp_path / "out")
    assert probs and "pins v7" in probs[0]


def test_cite_declaration_errors(tmp_path):
    decl, _ = _cite_files(tmp_path)
    doc = yaml.safe_load(decl.read_text())
    doc["cite"]["systems"]["numbers"]["missing"] = "same"
    decl.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="'same' needs the key"):
        N.load(decl)
    del doc["cite"]["systems"]["numbers"]["missing"]
    del doc["cite"]["systems"]["snapshot"]
    decl.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match="needs study, snapshot and numbers"):
        N.load(decl)


def test_manuscript_cited_number_needs_a_citation(tmp_path):
    decl, _ = _cite_files(tmp_path)
    nums = N.resolve(Ctx({"requirements:requirements": REQ}), N.load(decl),
                     lambda s, n: Ctx({"ref_fusion:smoothness": SYS}))
    exp = tmp_path / "export"
    N.write(exp, nums, {"snapshot": "r9"}, decl)
    tex = tmp_path / "paper"
    tex.mkdir()
    (tex / "main.tex").write_text(
        "Opening was \\numReqOpening. Raw optics jitter by \\numJitterOptical "
        "\\cite{sys}. The single value \\numJitterOneVal is quoted bare.\n")
    r = N.check_manuscript(tex, [exp])
    assert r["undefined"] == [] and r["uncited"] == ["numJitterOne"]


# -- check-outputs: n ---------------------------------------------------------------------------
def _fig(n_text, ticks):
    p = Panel("", Axis("x", ticklabels=ticks), Axis("y"),
              [Series("scatter", np.arange(len(ticks), dtype=float), np.ones(len(ticks)))])
    return attach(FigureSpec("f", "t", [p]), CaptionSpec("x", n=n_text))


def test_caption_participants_parsing():
    assert C.caption_participants("n = 14 participants", 14) == [14]
    assert C.caption_participants("n = 84 channels of 14 participants", 14) == [14]
    assert C.caption_participants("n = up to 14 shape-usable participants per muscle", 14) \
        == [14]
    assert C.caption_participants("P1, comparison take", 2) == [1]
    assert C.caption_participants("8 conditions, 10–11 slots each", 2) == []


def test_caption_n_issues():
    labels = [f"P{i}" for i in range(1, 14)]
    probs, known = C.caption_n_issues(_fig("n = 13 participants", labels), 14, 13)
    assert probs == [] and known
    probs, known = C.caption_n_issues(_fig("n = 14 participants", labels), 14, 13)
    assert known and "labels 13 participants" in probs[0]
    probs, _ = C.caption_n_issues(_fig("n = 15 participants", []), 14, 13)
    assert "more than 14" in probs[0]
    probs, _ = C.caption_n_issues(_fig("n = 14 adults", []), 14, 13)   # no "participants"
    assert probs == []
    probs, _ = C.caption_n_issues(_fig("n = 14 participants (adults)", []), 14, 13)
    assert "more than 13" in probs[0]
    # P95 is a percentile, not a participant; a table labels its participants in cells
    t = attach(TableSpec("t", "", [Column("p", "P")],
                         [{"p": "P1"}, {"p": "P2"}, {"p": "P95"}]),
               CaptionSpec("x", n="n = 2 participants"))
    assert C.spec_pseudonyms(t, 14) == {1, 2}
    assert C.caption_n_issues(t, 2, 2) == ([], True)
    # a status after the pseudonym ("P3 (flagged)", kin_condyle3d panel titles) still labels P3
    flagged = [f"P{i}" for i in range(1, 11)] + ["P11 (flagged)", "P12 (unknown)", "P13 (flagged)"]
    assert C.caption_n_issues(_fig("n = 13 participants", flagged), 14, 13) == ([], True)
    # no labels: checked against the participants in the source tables (fewer is fine)
    probs, known = C.caption_n_issues(_fig("n = 13 participants", []), 14, 13, data_n=11)
    assert known and "source tables hold 11" in probs[0]
    assert C.caption_n_issues(_fig("n = 11 participants", []), 14, 13, data_n=13) == ([], True)
    assert C.caption_n_issues(_fig("n = 11 participants", []), 14, 13)[1] is False
    assert C.caption_n_issues(_fig("8 conditions", []), 14, 13) == ([], True)   # no count


def test_source_participants():
    class Ctx:
        def table(self, stage, name):
            if name == "arrays":
                raise KeyError(name)
            return {"a": pd.DataFrame({"pid": ["s01", "s02", "s02"]}),
                    "b": pd.DataFrame({"pid": ["s01", "s02", "s03", ""]}),
                    "c": pd.DataFrame({"x": [1]})}[name]

    f = FigureSpec("f", "t", [], sources=["st:a", "st:b", "st:c", "st:arrays"])
    assert C.source_participants(Ctx(), f) == 3
    assert C.source_participants(Ctx(), FigureSpec("g", "t", [], sources=["st:c"])) is None
    # a cohort table: its participant-count column; pid_* columns of a pairwise table
    Ctx.table = lambda self, st, name: {
        "d": pd.DataFrame({"n": [13, 12, 7]}),
        "e": pd.DataFrame({"pid_smaller": ["s10"], "pid_larger": ["s14"]})}[name]
    assert C.source_participants(Ctx(), FigureSpec("h", "t", [], sources=["st:d"])) == 13
    assert C.source_participants(Ctx(), FigureSpec("i", "t", [], sources=["st:e"])) == 2
    # arrays of one named participant (meta pid): one participant, not "not checkable"
    Ctx.table = lambda self, st, name: (_ for _ in ()).throw(KeyError(name))
    one = FigureSpec("j", "t", [], sources=["emg:arrays"], meta={"pid": "s04"})
    assert C.source_participants(Ctx(), one) == 1
    assert C.source_participants(Ctx(), FigureSpec("k", "t", [], sources=["emg:arrays"])) is None
    # a single-participant item names its participant
    assert C.caption_n_issues(_fig("P1, comparison take", []), 2, 2) == ([], True)
    assert "the cohort has 2" in C.caption_n_issues(_fig("P3, comparison take", []), 2, 2)[0][0]


def test_number_n_issues():
    def num(key, n, **kw):
        return N.Number(key, 1.0, "1.0", n=n, source=kw.pop("source", "a:b"), **kw)

    nums = [num("reqOpening", 14), num("reqOpeningAdults", 14), num("imp", 84, n_of="channels"),
            num("cited", 2, cite={"study": "/v/study.yaml", "snapshot": "v7", "label": "s"}),
            num("citedBad", 3, cite={"study": "/v/study.yaml", "snapshot": "v7", "label": "s"}),
            num("x", 13, source="sensitivity:reporting_population")]
    probs = C.number_n_issues(nums, 14, 13, {"/v/study.yaml": 2})
    assert probs == ["reqOpeningAdults: n = 14 > 13 adults",
                     "citedBad: n = 3 > 2 the cited study's participants"]
    assert C.number_n_issues([num("a", 15)], 14, 13) == ["a: n = 15 > 14 participants"]


def test_stale_n_lines(tmp_path):
    p = tmp_path / "an15_x.md"
    p.write_text("cohort n = 14\nold: n = 12 participants\nn ≈ 12\nn = 12 cycles\n")
    assert C.stale_n_lines([p]) == ["an15_x.md:2", "an15_x.md:3"]
