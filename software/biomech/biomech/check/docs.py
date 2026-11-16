# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Docs against the results and the code (J8 / J11): ``biomech check-docs``.

The papers are written from the docs (the method documents, the paper notes and the storyboards) and the result files, so
every number a doc states must be a macro of a numbers file, quoted with its current value, and
every code reference must exist. Checks, per doc:

* **macros** -- every ``\\numKey`` (with the ``Val`` / ``CI`` / ``N`` forms) is declared in a
  numbers file (the exports' ``numbers.json``, own and cited numbers);
* **values** -- a group of macros followed by values in parentheses, e.g.
  ``\\numA`` / ``\\numB`` (7 / 6), quotes the macros' current texts, in order;
* **hand-typed** (paper-facing docs only, :data:`STRICT`) -- a number with a unit that no
  macro covers, outside code spans, history sections (:data:`HISTORY_RE`) and lines marked
  ``<!-- num-ok: reason -->`` (a method constant, a literature value, a definition);
* **code refs** -- ``path.py:123`` and ``path.py`` references point to existing files and
  lines;
* **snapshots** -- snapshot names (``r13a``, ``v11``) other than the pinned ones outside
  history sections;
* **population** -- "n = 14", "all 14" or "outside the protocol" in a paper-facing doc, outside history
  sections and lines saying it is excluded (D36: the papers report the 13 adults);
* **index** -- every doc of ``docs/analysis`` is listed in its README.

Hand-typed numbers and population hits are reported per line, so the docs can be converted
(J8); the other checks are failures.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DOCS = REPO / "docs" / "analysis"
# Paper-facing docs: every result number is a macro.
STRICT = ("an14_system_validation.md", "an15_paper_handoff.md", "an16_paper_storyboards.md",
          "an17_paper_arguments.md")
EXPORTS = (REPO / "data/analysis/icorr2027/paper_dataset/journal",
           REPO / "data/analysis/jtsmm_validation/paper_systems")
# Code references are written relative to the repository, a package directory or a package.
CODE_ROOTS = (REPO, REPO / "software",
              *sorted(p for p in (REPO / "software").glob("*") if p.is_dir()),
              *sorted(p for p in (REPO / "software").glob("*/*") if p.is_dir()))

MACRO_RE = re.compile(r"\\num([A-Z][A-Za-z]*)")
PLACEHOLDERS = {"numKey", "numKeyVal", "numKeyCI", "numKeyN", "numX", "numXVal", "numXCI",
                "numXN"}                                 # examples in the docs, not numbers
HISTORY_DOCS = re.compile(r"^plan_R[0-7]\.md$")        # earlier plans: a record, not checked
# A run of backticked macros joined by separators, then "(values ...)".
GROUP_RE = re.compile(r"((?:`\\num[A-Z][A-Za-z…]*`(?:\s*(?:/|±|–|-|,|and|vs\.?|to)\s*)?)+)"
                      r"\s*\(([^()]*)\)")
NUM_RE = re.compile(r"(?<![A-Za-z\d.])[−-]?\d+(?:\.\d+)?")
UNITS = r"(?:°/s²|°/s|mm/s²|mm/s|mm|cm|°|deg|ms|s\b|Hz|%|µV|uV|kΩ|dB|ppm|N·m|g\b|px)"
HAND_RE = re.compile(rf"(?<![\w.`\\])[−-]?\d+(?:\.\d+)?\s?{UNITS}")
CODE_SPAN_RE = re.compile(r"`[^`]*`")
OK_MARK = "<!-- num-ok"
# A section is a record of its time (not checked for stale snapshot names) when its heading
# says so or carries a date: a dated entry names the runs of that day by design.
HISTORY_RE = re.compile(r"^#+ .*(\b(history|status|errat|changelog|old values|superseded|"
                        r"log|deviations?)\b|\b20\d\d-\d\d-\d\d\b)", re.I)
REF_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w-]+\.(?:py|yaml|md|tex|sh))(?::(\d+))?")
SNAP_RE = re.compile(r"(?<![\w/.-])((?:r\d+[ab]?)|(?:v\d+))(?![\w-]|\.\d)")   # not v1.0
SNAP_TIME_RE = re.compile(r"\b(before|since|until|earlier|first|interim|previous(ly)?|"
                          r"predated|then|formerly|old|were|was|done)\b|\b20\d\d-\d\d-\d\d\b",
                          re.I)
# A snapshot name counts on a line about snapshots or result stores; the hardware, software and
# operations docs are not checked (there "v1" is a board or schema version).
SNAP_CONTEXT_RE = re.compile(r"\b(snapshots?|pinned|frozen|stores?)\b", re.I)
NO_SNAP_DIRS = ("hardware", "software", "firmware", "optical", "ml", "operations",
                "calibration_validation", "bringup")
# Plans name the files they will create; a dated plan is a record of its time.
PLAN_DOC_RE = re.compile(r"^(plan_R\d+\w*|research_plan|\d+_implementation_roadmap)\.md$")
# Relative shell invocations (./flash.sh, .../eval.py) and placeholder folders (EEPTOOLS/x.sh)
# name a file in the reader's working directory, not in the repository.
LOCAL_REF_RE = re.compile(r"^(\.{1,3}/|[A-Z][A-Z0-9_]+/)")
POP_RE = re.compile(r"\bn\s*=\s*14\b|\ball 14\b|\boutside the protocol\b", re.I)


@dataclass
class DocReport:
    doc: str
    undefined: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    hand_typed: list[str] = field(default_factory=list)
    bad_refs: list[str] = field(default_factory=list)
    stale_snapshots: list[str] = field(default_factory=list)
    population: list[str] = field(default_factory=list)

    def failures(self) -> int:
        return len(self.undefined) + len(self.mismatched) + len(self.bad_refs)


def macro_texts(exports: tuple[Path, ...] = EXPORTS) -> dict[str, str]:
    """Macro name (``numKey``, ``numKeyVal``, ``numKeyN``, ``numKeyCI``) -> its text, from the
    exports' ``numbers.json`` (own and cited numbers)."""
    out: dict[str, str] = {}
    for e in exports:
        p = Path(e) / "numbers.json"
        if not p.is_file():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        for group in ("numbers", "cited"):
            for key, n in (doc.get(group) or {}).items():
                base = "num" + key[:1].upper() + key[1:]
                out[base] = out[base + "Val"] = str(n.get("text", ""))
                if n.get("n") is not None:
                    out[base + "N"] = str(n["n"])
                if n.get("ci_text"):
                    out[base + "CI"] = "[{}, {}]".format(*n["ci_text"])
    return out


def pinned_snapshots() -> set[str]:
    from biomech.results.snapshot import pinned

    return set(pinned(REPO / "experiments" / "icorr2027" / "study.yaml"))


def _run_in(a: list[float], b: list[float]) -> bool:
    """``a`` occurs as one contiguous run in ``b``."""
    return any(all(abs(x - y) <= 1e-9 for x, y in zip(b[i:i + len(a)], a))
               for i in range(len(b) - len(a) + 1)) if a and len(a) <= len(b) else False


def _norm(s: str) -> str:
    return s.replace("−", "-").replace(",", ".")


def check_doc(path: Path, macros: dict[str, str], pins: set[str],
              strict: bool | None = None) -> DocReport:
    rep = DocReport(path.name)
    strict = path.name in STRICT if strict is None else strict
    history = False
    try:
        rel = path.resolve().relative_to(REPO).parts
    except ValueError:
        rel = ()
    snap_exempt = bool(set(rel[:2]) & set(NO_SNAP_DIRS))
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            history = bool(HISTORY_RE.match(line))
        where = f"{path.name}:{no}"
        for m in MACRO_RE.finditer(line):
            name = "num" + m.group(1)
            # no exports at all (a checkout without data/): names cannot be resolved, not checked
            if macros and name not in macros and name not in PLACEHOLDERS \
                    and line[m.end():m.end() + 1] not in ("…", "<"):   # "…" / "<A|B>" patterns
                rep.undefined.append(f"{where}: \\{name}")
        for g in GROUP_RE.finditer(line):
            names = ["num" + x for x in MACRO_RE.findall(g.group(1))]
            if not re.match(r"\s*[−~≈-]?\d", g.group(2)):
                continue                               # a remark, not the values
            vals = NUM_RE.findall(g.group(2))
            want = [macros.get(n) for n in names]
            if not vals or any(w is None for w in want):
                continue
            # The values quote the macros in order, possibly with more around them (a group
            # starting on the line before, a mean, a range): the macros' values must appear
            # as one contiguous run.
            got = [float(_norm(v)) for v in vals]
            exp = [float(_norm(NUM_RE.findall(w)[0])) for w in want if NUM_RE.findall(w)]
            if len(exp) == len(want) and not (_run_in(exp, got) or _run_in(got, exp)):
                rep.mismatched.append(f"{where}: {', '.join(names)} = {exp} but the text "
                                      f"says {got}")
        for m in REF_RE.finditer(line):
            ref, ln = m.group(1), m.group(2)
            if "/" not in ref and not ln:
                continue                               # a bare file name: too ambiguous
            if ref.startswith(("data/", "../", "figures/", "tables/")) \
                    or LOCAL_REF_RE.match(ref) or PLAN_DOC_RE.match(path.name):
                continue                     # generated outputs; plans name files to come
            hit = next((r / ref for r in CODE_ROOTS if (r / ref).is_file()), None) \
                or _by_suffix(ref)
            if hit is None and "/" in ref:
                rep.bad_refs.append(f"{where}: {ref} not found")
            elif hit is not None and ln and int(ln) > sum(1 for _ in hit.open(errors="ignore")):
                rep.bad_refs.append(f"{where}: {ref}:{ln} beyond the end of the file")
        if history:
            continue
        # A non-pinned snapshot name is stale unless the line places it in time (it also names
        # a current pin, or says before / since / until / first / interim ...); plans describe
        # the runs as planned on their date.
        if not PLAN_DOC_RE.match(path.name) and not snap_exempt and not (
                any(re.search(rf"(?<![\w/.-]){re.escape(pn)}(?![\w-])", line) for pn in pins)
                or SNAP_TIME_RE.search(line)):
            for m in SNAP_RE.finditer(line):
                if m.group(1) not in pins and SNAP_CONTEXT_RE.search(line):
                    rep.stale_snapshots.append(f"{where}: {m.group(1)}")
        if not strict or OK_MARK in line:
            continue
        bare = CODE_SPAN_RE.sub(" ", GROUP_RE.sub(" ", line))
        for m in HAND_RE.finditer(bare):
            rep.hand_typed.append(f"{where}: {m.group(0).strip()}")
        if POP_RE.search(line) and not re.search(r"exclud|record|not in the paper", line, re.I):
            rep.population.append(f"{where}: {line.strip()[:120]}")
    return rep


_FILES: list[str] = []


def _by_suffix(ref: str) -> Path | None:
    """The one tracked file whose path ends with ``ref`` (a path relative to some package
    directory, e.g. ``views/blocks.py``); None when none or several match."""
    import subprocess

    if not _FILES:
        r = subprocess.run(["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True)
        _FILES.extend(r.stdout.splitlines())
    hits = [f for f in _FILES if f == ref or f.endswith("/" + ref)]
    return REPO / hits[0] if len(hits) == 1 else None


def index_missing(docs_dir: Path = DOCS) -> list[str]:
    readme = (docs_dir / "README.md").read_text(encoding="utf-8")
    return sorted(p.name for p in docs_dir.glob("*.md")
                  if p.name != "README.md" and p.name not in readme)


ITEM_RE = re.compile(r"`([a-z][a-z0-9_]+)(?:--[^`]*)?`")
STORYBOARD = DOCS / "an16_paper_storyboards.md"
PAPER_SETS = {"1": REPO / "experiments/icorr2027/paper_dataset.yaml",
              "2": REPO / "experiments/jtsmm_validation/paper_systems.yaml"}


def claim_coverage(path: Path = STORYBOARD) -> tuple[list[str], list[str]]:
    """Every claim of the storyboard carried by an item of its paper set (J2.5):
    the section tables of the storyboard, §1 (dataset) and §2 (systems), with a "Claim" and an "Items"
    column. Failures: a claim row without an item, an item id that its paper set does not
    list. Warnings: core / supporting items of a paper set that no claim row names."""
    from biomech import papers

    fails, warns = [], []
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## (\d)\. ", text, flags=re.M)
    for num, body in zip(parts[1::2], parts[2::2]):
        if num not in PAPER_SETS:
            continue
        ps = papers.load(PAPER_SETS[num])
        listed = {(e if isinstance(e := i.entry(), str) else e["id"]) for i in ps.items}
        main = {(e if isinstance(e := i.entry(), str) else e["id"]) for i in ps.items
                if getattr(i, "part", "") in ("core", "supporting")}
        named: set[str] = set()
        cols = None
        for line in body.splitlines():
            if not line.startswith("|"):
                cols = None
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "Claim" in cells and any(c.startswith("Items") for c in cells):
                cols = (cells.index("Claim"),
                        next(i for i, c in enumerate(cells) if c.startswith("Items")))
                continue
            if cols is None or set(line) <= set("|-: "):
                continue
            items = cells[cols[1]] if cols[1] < len(cells) else ""
            ids = [m for m in ITEM_RE.findall(items)]
            sect = re.sub(r"\*", "", cells[0])[:40]
            if not ids:
                fails.append(f"storyboard §{num} {sect}: claim without an item")
            for i in ids:
                if i not in listed:
                    fails.append(f"storyboard §{num} {sect}: {i} is not in {PAPER_SETS[num].name}")
            named |= set(ids)
        for i in sorted(main - named):
            warns.append(f"{PAPER_SETS[num].name}: {i} carries no claim row in storyboard §{num}")
    return fails, warns


def check_docs(paths: list[Path] | None = None) -> tuple[list[DocReport], list[str]]:
    """Reports for ``paths`` (default: every doc of docs/analysis and the two analysis
    plans) and the docs missing from the index."""
    macros, pins = macro_texts(), pinned_snapshots()
    paths = paths or (sorted(p for p in DOCS.glob("*.md") if not HISTORY_DOCS.match(p.name))
                      + [REPO / "experiments/icorr2027/analysis_plan.md",
                         REPO / "experiments/jtsmm_validation/analysis_plan.md"])
    return [check_doc(p, macros, pins) for p in paths], index_missing()

def fill_doc(path: Path, macros: dict[str, str]) -> tuple[int, list[str]]:
    """Rewrite the values quoted after macro groups with the macros' current texts, where the
    parentheses hold exactly one number per macro (separators and units kept): after a
    re-run the docs follow the exports. Returns (values changed, groups left for a human:
    a different count of numbers than macros)."""
    text = path.read_text(encoding="utf-8")
    changed, left = 0, []

    def repl(g: re.Match) -> str:
        nonlocal changed
        names = ["num" + x for x in MACRO_RE.findall(g.group(1))]
        body = g.group(2)
        if not re.match(r"\s*[−~≈-]?\d", body) or any(n not in macros for n in names):
            return g.group(0)
        nums = list(NUM_RE.finditer(body))
        if len(nums) != len(names):
            want = [float(_norm(NUM_RE.findall(macros[n])[0])) for n in names
                    if NUM_RE.findall(macros[n])]
            if not _run_in(want, [float(_norm(m.group(0))) for m in nums]):
                left.append(f"{path.name}: {g.group(0)[:100]}")
            return g.group(0)
        out, last = [], 0
        for m, n in zip(nums, names):
            new = NUM_RE.findall(macros[n])
            new = new[0].replace("-", "−") if new else m.group(0)
            if _norm(new) != _norm(m.group(0)):
                changed += 1
            out.append(body[last:m.start()] + new)
            last = m.end()
        out.append(body[last:])
        whole, head = g.group(0), g.group(1)
        gap = whole[len(head):whole.index("(", len(head))]       # the text before "("
        return f"{head}{gap}({''.join(out)})"

    new_text = "\n".join(GROUP_RE.sub(repl, line) for line in text.split("\n"))
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return changed, left
