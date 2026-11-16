# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The repository's markdown documentation as one typeset PDF book, with checks.

    software/jtsmm/.venv/bin/python scripts/docs_pdf/build_docs_pdf.py [--volumes]
        [--part KEY ...] [--checks-only] [--jobs N] [--out build/docs_pdf] [--strict]

Every tracked markdown file (``docs/`` and the READMEs and reports around the code; not
``data/``) goes into one part of the book (:data:`PARTS`: overview, hardware, firmware,
software, optical tracking, calibration and operations, analysis and papers, machine learning,
outlook, and an appendix of records); each part opens with a short introduction and each file
becomes a chapter (``--volumes``: one PDF per part). pandoc (GitHub markdown) -> LaTeX through
``docs.lua`` (cross-document links become internal links, placeholders such as ``<session>``
stay text, code paths wrap, wide tables get widths from their content, Mermaid diagrams are
rendered with mermaid-cli and cached) -> LuaLaTeX with ``style.tex`` (Libertinus, the papers'
Okabe-Ito blue).

Checks, written to ``<out>/report.md`` (and summarised on the console):

* sources -- links to other docs, their anchors (GitHub rules) and to repository files;
  pipe tables whose rows have more cells than the header (pandoc drops the extra cells);
  unbalanced code fences; heading levels that skip; missing images; files in no part;
* consistency -- ``biomech.check.docs`` on every doc: undefined or mis-quoted ``\\num``
  macros, file:line references that do not resolve, stale snapshot names; the docs indexes
  (``docs/README.md`` and each folder's README) list every doc of their folder;
* up-to-dateness -- a doc whose referenced code changed long after the doc's last commit (or
  its ``<!-- docs-reviewed: YYYY-MM-DD -->`` marker; records and dated reports are exempt),
  uncommitted doc changes (the PDF is built from the working tree), TODO markers;
* format (after the build, per document) -- LaTeX overfull boxes (text or tables wider than
  the type area), missing glyphs, unresolved internal links, text outside the type area in
  the PDF, near-empty pages, fonts not embedded.

Exit status 1 with ``--strict`` when an error-level issue remains.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXCLUDE = ("data/", "build/")

@dataclass(frozen=True)
class Part:
    """One part of the book (and one volume with ``--volumes``). ``members`` are repository
    paths (a file) or prefixes (ending in "/"); a file belongs to the part with the longest
    matching member, and within a part the members give the reading order."""
    key: str
    title: str
    intro: str
    members: tuple[str, ...]
    appendix: bool = False

# The story: what the system is, how it is built (hardware, firmware, software), how it tracks
# and is calibrated and run, what the studies made of the data, the ML on top, what is left;
# records of their time (plans, dated reports, generated audits) go to the appendix.
PARTS = [
    Part("overview", "Overview and architecture",
         "What JTS-MM is and why it exists: an open-source, low-cost multimodal jaw tracking "
         "platform (optical tracking, two IMUs and six-channel surface EMG on one master "
         "clock) that replaces the Qualisys-based jaw tracking system and adds the signals "
         "needed to detect jaw-motion intent for an exoskeleton. This part gives the idea, the "
         "architecture and the timing rule everything else builds on, and maps the "
         "repository and the documentation.",
         ("README.md", "docs/00_overview.md", "docs/01_architecture.md", "docs/README.md",
          )),
    Part("hardware", "Hardware",
         "The physical system, from the parts list to the boards: the Raspberry Pi 5 with the "
         "custom HAT (RP2350 microcontroller, ADS1299 EMG front end, IMU and trigger "
         "interfaces, power and grounding), the cameras, the IMUs and the mechanical parts "
         "worn on the teeth. The schematic build guides and the layout guide record how the "
         "HAT was designed; the component catalog (§6.1) lists what the recordings used.",
         ("docs/02_components_bom.md", "docs/04_component_catalog.md", "docs/hardware/",
          "hardware/mechanical/",
          "hardware/")),
    Part("firmware", "Firmware",
         "The RP2350 firmware that owns the master clock: it stamps every EMG and IMU sample, "
         "drives the camera trigger and streams the records to the Pi."
         "",
         ("firmware/jtsmm/README.md", "firmware/",
          )),
    Part("software", "Software and data",
         "The software on the Pi: the capture processes, the HDF5 session format (a superset "
         "of the original JTS format), clock alignment, signal processing, the runtime "
         "configuration and the live viewer; then the package READMEs and the vendored JTS "
         "analysis code.",
         ("docs/software/", "software/jtsmm/", "schema/", "software/jaw_tracking/",
          "scripts/")),
    Part("optical", "Optical tracking",
         "Marker-based tracking with two hardware-triggered global-shutter cameras: marker "
         "setup, calibration and coordinate frames, pose estimation and output, and the "
         "migration from the Qualisys system.",
         ("docs/optical/", "software/optical/")),
    Part("calibration", "Calibration, validation and operations",
         "How a session is prepared and run: camera, IMU and EMG calibration, the reference "
         "validation against Qualisys and Delsys, the per-session checklist, participant "
         "preparation, the EMG safety SOP, remote viewing and troubleshooting.",
         ("docs/calibration_validation/", "docs/operations/")),
    Part("analysis", "Analysis and papers",
         "From recorded sessions to results: the offline analysis package biomech (cohort and "
         "quality, frames and kinematics, EMG, IMU, timing and onsets, uncertainty, "
         "statistics, export, viewer), the system validation, and the two studies behind the "
         "dataset paper and the systems paper; then the evaluations of the recording system.",
         ("docs/analysis/README.md", "docs/analysis/", "software/biomech/",
          "experiments/README.md", "experiments/icorr2027/README.md", "experiments/icorr2027/",
          "experiments/jtsmm_validation/README.md", "experiments/jtsmm_validation/",
          "experiments/data/", "experiments/")),
    Part("ml", "Machine learning",
         "Jaw-motion intent decoding on top of the analysis: ground-truth labels, causal EMG "
         "and IMU windows and features, models and training, evaluation protocols, sensor "
         "fusion and online use.",
         ("docs/ml/", "software/ml/")),
]
PART_BY_KEY = {pt.key: pt for pt in PARTS}
# chapter titles in the book where a file's own title would read like another one's
CHAPTER_TITLES = {"docs/README.md": "Documentation guide and conventions"}
PART_ORDER = [pt.key for pt in PARTS]

# -- collection -----------------------------------------------------------------------------------
def md_files() -> list[str]:
    r = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co", "--exclude-standard",
                        "*.md"], capture_output=True, text=True, check=True)
    return sorted(p for p in r.stdout.splitlines()
                  if not p.startswith(EXCLUDE) and (ROOT / p).is_file())

def _member_match(path: str) -> tuple[str, int] | None:
    """(part key, member index) of the longest member matching ``path``."""
    best: tuple[int, str, int] | None = None
    for pt in PARTS:
        for i, m in enumerate(pt.members):
            if path == m or (m.endswith("/") and path.startswith(m)):
                if best is None or len(m) > best[0]:
                    best = (len(m), pt.key, i)
    return (best[1], best[2]) if best else None

def part_of(path: str) -> str:
    hit = _member_match(path)
    return hit[0] if hit else "overview"

LINK_RE = re.compile(r"(?<!!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

def index_order(files: list[str]) -> dict[str, int]:
    """Reading order: the order in which docs/README.md and the folder READMEs link the docs
    (a breadth-first walk from the root README), then the rest alphabetically."""
    order: dict[str, int] = {}
    queue = ["README.md", "docs/README.md"]
    known = set(files)
    while queue:
        cur = queue.pop(0)
        if cur in order or cur not in known:
            continue
        order[cur] = len(order)
        text = (ROOT / cur).read_text(encoding="utf-8", errors="replace")
        for m in LINK_RE.finditer(_strip_code(text)):
            tgt = m.group(1).split("#")[0]
            if tgt.endswith(".md") and not re.match(r"^[a-z]+:", tgt):
                full = os.path.normpath(os.path.join(os.path.dirname(cur), tgt))
                if full in known and full not in order:
                    queue.append(full)
    return order

def _strip_code(text: str) -> str:
    """The text without fenced code blocks and inline code spans (links inside are not
    links)."""
    out, fence = [], False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            fence = not fence
            out.append("")
            continue
        out.append("" if fence else re.sub(r"`[^`]*`", "``", line))
    return "\n".join(out)

# -- GitHub anchors -------------------------------------------------------------------------------
def slug(text: str) -> str:
    """GitHub's heading anchor: lower case, markup and punctuation removed (letters, digits,
    marks, '-', '_' and spaces kept), spaces -> '-'."""
    t = re.sub(r"`([^`]*)`", r"\1", text)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = t.replace("*", "").strip().lower()
    keep = []
    for ch in t:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in ("L", "N", "M"):
            keep.append(ch)
    return "".join(keep).replace(" ", "-")

def anchors_of(text: str) -> set[str]:
    out: set[str] = set()
    seen: dict[str, int] = defaultdict(int)
    fence = False
    for line in text.splitlines():                  # headings keep their inline code
        if FENCE_RE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            s = slug(m.group(2))
            n = seen[s]
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] += 1
        for a in re.finditer(r"<a\s+[^>]*(?:id|name)\s*=\s*\"([^\"]+)\"", line):
            out.add(a.group(1))
    return out

# -- source checks ----------------------------------------------------------------------------------
@dataclass
class Issues:
    items: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))

    def add(self, category: str, level: str, msg: str) -> None:
        self.items[category].append((level, msg))

    def count(self, level: str) -> int:
        return sum(1 for v in self.items.values() for lv, _ in v if lv == level)

def _cells(row: str) -> int:
    s = re.sub(r"<!--.*?-->", "", row).strip()          # a trailing check marker is no cell
    s = re.sub(r"`[^`]*`", "``", s)
    s = s.replace("\\|", "")
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return s.count("|") + 1

def check_sources(files: list[str], iss: Issues) -> None:
    texts = {f: (ROOT / f).read_text(encoding="utf-8", errors="replace") for f in files}
    anchors = {f: anchors_of(t) for f, t in texts.items()}
    tracked = set(subprocess.run(["git", "-C", str(ROOT), "ls-files", "-co",
                                  "--exclude-standard"], capture_output=True,
                                 text=True).stdout.splitlines())
    for f, text in texts.items():
        lines = text.splitlines()
        # fences
        n_fence = sum(1 for ln in lines if FENCE_RE.match(ln))
        if n_fence % 2:
            iss.add("sources", "error", f"{f}: unbalanced code fence ({n_fence} fence lines)")
        plain = _strip_code(text).splitlines()
        # tables: every row as many cells as the separator row
        i = 0
        while i < len(plain):
            if plain[i].lstrip().startswith("|") and i + 1 < len(plain) and re.match(
                    r"^\s*\|?\s*:?-{2,}", plain[i + 1]):
                ncol = _cells(plain[i + 1])
                j = i
                while j < len(plain) and plain[j].lstrip().startswith("|"):
                    if j != i + 1 and _cells(plain[j]) > ncol:    # fewer: padded, fine
                        iss.add("sources", "error",
                                f"{f}:{j + 1}: table row has {_cells(plain[j])} cells, the "
                                f"header {ncol} (the extra cells are dropped)")
                    j += 1
                i = j
            else:
                i += 1
        # heading levels
        prev = 0
        for no, ln in enumerate(plain, 1):
            m = re.match(r"^(#{1,6})\s", ln)
            if m:
                lv = len(m.group(1))
                if prev and lv > prev + 1:
                    iss.add("sources", "warn", f"{f}:{no}: heading jumps from level {prev} "
                                               f"to {lv}")
                prev = lv
        # links and images
        for no, ln in enumerate(plain, 1):
            for m in list(LINK_RE.finditer(ln)) + list(IMG_RE.finditer(ln)):
                tgt = m.group(1).strip("<>")
                if re.match(r"^[a-z][a-z0-9+.-]*:", tgt):
                    continue                                    # http:, mailto: ...
                path, _, frag = tgt.partition("#")
                full = os.path.normpath(os.path.join(os.path.dirname(f), path)) if path else f
                if path and full not in tracked and not (ROOT / full).exists():
                    iss.add("links", "error", f"{f}:{no}: link to a missing file: {tgt}")
                    continue
                if frag and full.endswith(".md") and full in anchors \
                        and frag not in anchors[full] and frag.lower() not in anchors[full]:
                    iss.add("links", "error", f"{f}:{no}: anchor not found: {tgt}")
        # math delimiters GitHub does not render (\[ \] / \( \)): use $$ / $
        for no, ln in enumerate(plain, 1):
            if re.search(r"(^|[^\\])\\[\[(]", ln) and re.search(r"\\[\])]", ln) \
                    or ln.strip() in ("\\[", "\\]"):
                iss.add("sources", "error", f"{f}:{no}: math in \\[ \\] or \\( \\) (not "
                                            "rendered by GitHub or pandoc): use $$ / $")
        # TODO markers (info)
        n_todo = len(re.findall(r"\bTODO\b|\bFIXME\b", _strip_code(text)))
        if n_todo:
            iss.add("up-to-date", "info", f"{f}: {n_todo} TODO / FIXME marker(s)")

def check_consistency(files: list[str], iss: Issues) -> None:
    """biomech.check.docs on every doc (macros, references, stale snapshot names) and the
    docs indexes."""
    sys.path.insert(0, str(ROOT / "software" / "biomech"))
    try:
        from biomech.check import docs as D
    except Exception as exc:                                  # outside the analysis venv
        iss.add("consistency", "warn", f"biomech.check.docs not importable ({exc}); run with "
                                       "software/jtsmm/.venv/bin/python")
        return
    macros, pins = D.macro_texts(), D.pinned_snapshots()
    if not macros:
        iss.add("consistency", "info", "no paper exports (numbers.json under data/): the \\num "
                                       "macros of the docs are not checked")
    for f in files:
        if not f.startswith(("docs/", "experiments/")):
            continue
        rep = D.check_doc(ROOT / f, macros, pins, strict=f.startswith("docs/analysis/")
                          and (ROOT / f).name in D.STRICT)
        for kind, level in (("undefined", "error"), ("mismatched", "error"),
                            ("bad_refs", "warn"), ("stale_snapshots", "warn")):
            for hit in getattr(rep, kind, []):
                iss.add("consistency", level, f"{f}: {kind.replace('_', ' ')}: "
                                              f"{hit.split(': ', 1)[-1]}")
    # every doc of a docs folder is linked from docs/README.md or its folder README
    root_readme = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    for f in files:
        if not f.startswith("docs/") or f.endswith("README.md"):
            continue
        folder = Path(f).parent
        folder_readme = ROOT / folder / "README.md"
        hay = root_readme + (folder_readme.read_text(encoding="utf-8")
                             if folder_readme.is_file() else "")
        if Path(f).name not in hay:
            iss.add("consistency", "warn", f"{f}: not listed in docs/README.md or "
                                           f"{folder}/README.md")

def check_up_to_date(files: list[str], iss: Issues, lag_days: int = 60) -> None:
    """Docs whose referenced code changed long after the doc; uncommitted doc changes."""
    log = subprocess.run(["git", "-C", str(ROOT), "log", "--format=@%ct", "--name-only"],
                         capture_output=True, text=True).stdout
    last: dict[str, int] = {}
    t = 0
    for line in log.splitlines():
        if line.startswith("@"):
            t = int(line[1:])
        elif line and line not in last:
            last[line] = t
    ref_re = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w-]+\.(?:py|c|h|yaml|lua|sh|toml))")
    reviewed_re = re.compile(r"<!--\s*docs-reviewed:\s*(\d{4}-\d{2}-\d{2})\s*-->")
    for f in files:
        if f not in last or part_of(f) == "records" or re.search(r"\d{4}-\d{2}-\d{2}", f):
            continue                    # a record of its time describes the code of its time
        text = (ROOT / f).read_text(errors="replace")
        rev = reviewed_re.search(text)  # a review after the last commit (no edit needed)
        if rev:
            last[f] = max(last[f], int(dt.datetime.fromisoformat(rev.group(1)).timestamp()))
        refs = {m.group(1) for m in ref_re.finditer(text)}
        newer = []
        for r in refs:
            full = os.path.normpath(os.path.join(os.path.dirname(f), r))
            for cand in (r, full):
                if cand in last:
                    if last[cand] - last[f] > lag_days * 86400:
                        newer.append((last[cand], cand))
                    break
        if newer:
            newest = max(newer)
            days = (newest[0] - last[f]) // 86400
            iss.add("up-to-date", "warn",
                    f"{f}: {len(newer)} referenced file(s) changed more than {lag_days} days "
                    f"after the doc's last commit (e.g. {newest[1]}, {days} days later); review")
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--"] + files,
                           capture_output=True, text=True).stdout.splitlines()
    for d in dirty:
        iss.add("up-to-date", "info", f"uncommitted: {d.strip()} (the PDF shows the working "
                                      "tree)")

# -- build ----------------------------------------------------------------------------------------
def _latex_esc(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
            .replace("%", "\\%").replace("#", "\\#").replace("$", "\\$"))

def highlight_macros() -> str:
    """pandoc's syntax-highlighting token macros (from a standalone run, so they always match
    the installed pandoc)."""
    src = "```python\nx = 1\n```\n"
    r = subprocess.run(["pandoc", "-f", "gfm", "-t", "latex", "-s", "--highlight-style=tango"],
                       input=src, capture_output=True, text=True, check=True)
    return "\n".join(ln for ln in r.stdout.splitlines() if re.match(r"\\newcommand\{\\\w+Tok\}", ln))

def pandoc_fragment(path: str, key: str, vol: str, docmap: Path, frag: Path,
                    issues_file: Path, langs: str, mermaid_dir: Path) -> str:
    env = {**os.environ, "DOCS_PDF_ISSUES": str(issues_file), "DOCS_PDF_LANGS": langs,
           "DOCS_PDF_ROOT": str(ROOT), "DOCS_PDF_MERMAID": str(mermaid_dir)}
    cmd = ["pandoc", str(ROOT / path), "-f", "gfm", "-t", "latex", "--wrap=preserve",
           "--top-level-division=chapter", "--listings",
           "--lua-filter", str(HERE / "docs.lua"),
           "-M", f"docpath={path}", "-M", f"dockey={key}", "-M", f"docmap={docmap}",
           "-M", f"volume={vol}", "--resource-path", str((ROOT / path).parent),
           *(["-M", f"doctitle={CHAPTER_TITLES[path]}"] if path in CHAPTER_TITLES else []),
           "-o", str(frag)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    return r.stderr.strip()

def _part_pre(pt: Part) -> str:
    return (f"\\setpartpreamble[u][.8\\linewidth]{{\\vspace{{10mm}}\\noindent"
            f"{{\\color{{accent}}\\rule{{\\linewidth}}{{.8pt}}}}\\par\\vspace{{4mm}}"
            f"\\noindent{{\\normalsize\\color{{ink}}{_latex_esc(pt.intro)}\\par}}}}")

def book_tex(title: str, number: int | None, parts: list[tuple[Part, list[tuple[str, str]]]],
             stamp: str, macros: str) -> str:
    """The book (all parts, each with a part page and its introduction) or one volume (one
    part; its introduction on the title page)."""
    single = len(parts) == 1
    vol_label = (f"Volume {number} \\textperiodcentered{{}} {title}" if number else title)
    ndocs = sum(len(d) for _, d in parts)
    body = []
    for pt, docs in parts:
        foot = ("JTS-MM documentation \\textperiodcentered{} "
                + ("Appendix: " if pt.appendix else "") + _latex_esc(pt.title))
        # the page break first, so the last page of the previous part keeps its footer
        body.append(f"\\clearpage\\renewcommand{{\\volumetitle}}{{{foot}}}")
        if not single:
            body.append(_part_pre(pt))
            body.append(("\\addpart" if pt.appendix else "\\part")
                        + f"{{{('Appendix: ' if pt.appendix else '') + _latex_esc(pt.title)}}}")
        body += [f"\\typeout{{DOCSPDF BEGIN {p}}}\n\\input{{frags/{k}.tex}}\n"
                 f"\\typeout{{DOCSPDF END {p}}}" for p, k in docs]
    if single:
        about = _latex_esc(parts[0][0].intro)
    else:
        about = ("The parts follow the system from the idea to the papers: overview and "
                 "architecture, hardware, firmware, software, optical tracking, calibration "
                 "and operations, the analysis behind the two papers, machine learning and the "
                 "outlook. Each part opens with a short introduction. Records of their time "
                 "(earlier plans, dated reports, generated audits) are collected in the "
                 "appendix.")
    return f"""\\documentclass[10pt, a4paper, oneside, headings=standardclasses,
  numbers=noendperiod]{{scrreprt}}
\\newcommand{{\\volumetitle}}{{JTS-MM documentation}}
\\newcommand{{\\docsbuildstamp}}{{{stamp}}}
\\input{{style.tex}}
{macros}
\\hypersetup{{pdftitle={{JTS-MM documentation: {title}}}, pdfauthor={{JTS-MM}}}}
\\begin{{document}}
\\begin{{titlepage}}
  \\vspace*{{30mm}}
  {{\\sffamily\\color{{muted}}\\large JTS-MM \\textperiodcentered{{}} open-source multimodal jaw tracking\\par}}
  \\vspace{{6mm}}
  {{\\sffamily\\bfseries\\fontsize{{30}}{{34}}\\selectfont\\color{{accentdark}} Documentation\\par}}
  \\vspace{{3mm}}
  {{\\sffamily\\fontsize{{20}}{{24}}\\selectfont\\color{{accent}} {vol_label}\\par}}
  \\vspace{{4mm}}{{\\color{{accent}}\\rule{{\\linewidth}}{{1.6pt}}}}\\par
  \\vspace{{8mm}}
  {{\\color{{ink}} {about}\\par}}
  \\vspace{{5mm}}
  {{\\small\\color{{muted}} {ndocs} documents from the repository's markdown sources, typeset
  for reading and printing. Links between documents are live{'' if not single else '; a document in another volume is marked [volume]'}.\\par}}
  \\vfill
  {{\\small\\sffamily\\color{{muted}} {stamp}\\par}}
\\end{{titlepage}}
\\pdfbookmark[0]{{Contents}}{{contents}}
\\tableofcontents
{chr(10).join(body)}
\\end{{document}}
"""

# -- mermaid ----------------------------------------------------------------------------------------
NODE_BIN = Path.home() / ".local/opt/node/bin"
MMDC_DEFAULT = Path.home() / ".local/opt/mermaid-cli/node_modules/.bin/mmdc"
# The docs style: Okabe-Ito blue for borders, light tints for fills, the text ink; Libertinus
# Sans (the headings' face) from the TeX tree, DejaVu Sans as fallback.
MERMAID_CONFIG = {
    "theme": "base",
    "themeVariables": {
        "fontFamily": "Libertinus Sans, DejaVu Sans, sans-serif", "fontSize": "18px",
        "primaryColor": "#EAF3FA", "primaryBorderColor": "#0072B2",
        "primaryTextColor": "#1F2328", "lineColor": "#57606A", "textColor": "#1F2328",
        "secondaryColor": "#FBF3E4", "secondaryBorderColor": "#E69F00",
        "tertiaryColor": "#F6F8FA", "tertiaryBorderColor": "#D0D7DE",
        "clusterBkg": "#F6F8FA", "clusterBorder": "#D0D7DE", "edgeLabelBackground": "#FFFFFF",
        "noteBkgColor": "#FBF3E4", "noteBorderColor": "#E69F00",
        "actorBkg": "#EAF3FA", "actorBorder": "#0072B2", "signalColor": "#57606A",
        "sectionBkgColor": "#EAF3FA", "altSectionBkgColor": "#FFFFFF",
        "taskBkgColor": "#0072B2", "taskBorderColor": "#084F7E", "taskTextColor": "#FFFFFF",
        "taskTextOutsideColor": "#1F2328", "gridColor": "#D0D7DE", "todayLineColor": "#D55E00",
    },
    # large labels, tight spacing: the diagrams are scaled down to the line width, so the
    # label share of the drawing decides the printed text size
    "flowchart": {"htmlLabels": True, "curve": "basis", "padding": 6, "nodeSpacing": 20,
                  "rankSpacing": 32, "useMaxWidth": False, "wrappingWidth": 200},
    "gantt": {"fontSize": 18, "sectionFontSize": 18, "useMaxWidth": False,
              "axisFormat": "%d %b", "tickInterval": "1week", "barHeight": 28,
              "barGap": 6, "leftPadding": 90},
}
# the frame a diagram is scaled into (style.tex \docsmermaid: \linewidth, .85\textheight), in pt
FIG_W, FIG_H = 449.0, 580.0
LABEL_PT = 18 * 0.75

def _mermaid_css() -> str:
    faces = []
    for style, weight, name in (("normal", 400, "Regular"), ("normal", 700, "Bold"),
                                ("italic", 400, "Italic")):
        f = subprocess.run(["kpsewhich", f"LibertinusSans-{name}.otf"], capture_output=True,
                           text=True).stdout.strip()
        if f:
            faces.append(f"@font-face {{ font-family: 'Libertinus Sans'; font-style: {style}; "
                         f"font-weight: {weight}; src: url('file://{f}'); }}")
    return "\n".join(faces) + "\n"

def render_mermaid(mdir: Path, owners: dict[str, str], jobs: int, iss: Issues) -> None:
    """Render every ``<hash>.mmd`` the filter wrote that has no PDF yet (cached by the hash
    of the source and the configuration); failures keep the source box in the PDF."""
    import json

    cfg_text = json.dumps(MERMAID_CONFIG, indent=1, sort_keys=True)
    css_text = _mermaid_css()
    stamp = mdir / "config.stamp"
    if not stamp.exists() or stamp.read_text() != cfg_text + css_text:
        for old in mdir.glob("*.pdf"):
            old.unlink()                       # the look changed: render everything again
        stamp.write_text(cfg_text + css_text)
    (mdir / "config.json").write_text(cfg_text)
    (mdir / "fonts.css").write_text(css_text)
    todo = sorted(m for m in mdir.glob("*.mmd") if not m.with_suffix(".pdf").exists()
                  and m.stem in owners)
    mmdc = os.environ.get("DOCS_PDF_MMDC") or shutil.which("mmdc") or (
        str(MMDC_DEFAULT) if MMDC_DEFAULT.exists() else None)
    if todo and not mmdc:
        iss.add("build", "warn", f"mermaid-cli (mmdc) not found: {len(todo)} diagram(s) shown "
                                 "as source (install: docs/README.md, Reading the docs as PDF)")
        return
    env = {**os.environ, "PATH": f"{NODE_BIN}:{os.environ.get('PATH', '')}"}

    def run(src: str, pdf: str) -> str:
        r = subprocess.run([mmdc, "-i", src, "-o", pdf, "--pdfFit", "-w", "1000",
                            "-c", "config.json", "-C", "fonts.css", "-b", "white", "-q"],
                           cwd=mdir, env=env, capture_output=True, text=True, timeout=300)
        ok = r.returncode == 0 and (mdir / pdf).exists()
        return "" if ok else (r.stderr.strip().splitlines() or ["failed"])[-1][:200]

    def scale(pdf: Path) -> float:
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
        w, h = (float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
        return min(FIG_W / w, FIG_H / h, 1.0)

    def one(m: Path) -> tuple[Path, str]:
        pdf = m.with_suffix(".pdf")
        err = run(m.name, pdf.name)
        if err:
            return m, err
        # A long left-to-right flowchart prints its text tiny once scaled to the line width;
        # top to bottom it often fits larger. Keep whichever prints larger.
        text = m.read_text()
        head = re.match(r"\s*(flowchart|graph)\s+(LR|RL)\b", text)
        if head and scale(pdf) < 0.62:
            alt = m.with_name(m.stem + ".tb.mmd")
            alt.write_text(text[:head.start(2)] + "TB" + text[head.end(2):])
            if not run(alt.name, alt.with_suffix(".pdf").name) \
                    and scale(alt.with_suffix(".pdf")) > scale(pdf) * 1.15:
                alt.with_suffix(".pdf").replace(pdf)
            for f in (alt, alt.with_suffix(".pdf")):
                f.unlink(missing_ok=True)
        return m, ""

    with cf.ThreadPoolExecutor(max_workers=max(1, min(jobs, 4))) as ex:
        for m, err in ex.map(one, todo):
            if err:
                iss.add("build", "warn", f"{owners[m.stem]}: mermaid diagram not rendered "
                                         f"(source shown): {err}")
    for h, doc in sorted(owners.items(), key=lambda x: x[1]):
        if (mdir / f"{h}.pdf").exists():
            pt = LABEL_PT * scale(mdir / f"{h}.pdf")         # labels, scaled to fit
            if pt < 6.0:
                iss.add("format", "warn", f"{doc}: a mermaid diagram's labels print at "
                                          f"{pt:.1f} pt (too wide or tall for the page; split "
                                          "it or shorten the labels)")
    n = sum(1 for h in owners if (mdir / f"{h}.pdf").exists())
    iss.add("build", "info", f"mermaid: {n} of {len(owners)} diagram(s) rendered "
                             f"({len(todo)} new)")

def compile_volume(tex: Path, passes: int = 3) -> tuple[bool, str]:
    log = ""
    for i in range(passes):
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error",
                            tex.name], cwd=tex.parent, capture_output=True, text=True,
                           timeout=7200)
        log = (tex.with_suffix(".log")).read_text(errors="replace")
        if r.returncode != 0:
            return False, log
        if i >= 1 and "Rerun" not in log and "undefined references" not in log.lower():
            break
    return True, log

# -- format checks (after the build) -------------------------------------------------------------
OVERFULL_RE = re.compile(r"Overfull \\hbox \(([\d.]+)pt too wide\) (?:in paragraph|in alignment|"
                         r"detected) at lines? (\d+)(?:--(\d+))?")

def check_log(log: str, vol: str, iss: Issues, tol_pt: float = 1.0) -> None:
    cur = "(volume front matter)"
    missing: dict[str, set[str]] = defaultdict(set)
    for line in log.splitlines():
        m = re.match(r"DOCSPDF BEGIN (.+)", line)
        if m:
            cur = m.group(1)
            continue
        m = OVERFULL_RE.search(line)
        if m and float(m.group(1)) > tol_pt:
            iss.add("format", "error" if float(m.group(1)) > 6 else "warn",
                    f"[{vol}] {cur}: overfull box {float(m.group(1)):.1f} pt "
                    f"(fragment lines {m.group(2)}{'–' + m.group(3) if m.group(3) else ''})")
        m = re.search(r"Missing character: There is no (.) \((U\+[0-9A-F]+)\)", line)
        if m:
            missing[cur].add(f"{m.group(1)} {m.group(2)}")
        m = re.search(r"(?:Hyper reference|Reference) `([^']+)' on page \d+ undefined", line)
        if m:
            iss.add("format", "error", f"[{vol}] {cur}: unresolved internal link {m.group(1)}")
        m = re.search(r"name\{([^}]+)\} has been referenced but does not exist", line)
        if m:
            iss.add("format", "error", f"[{vol}]: link target missing in the PDF: {m.group(1)}")
    for doc, chars in missing.items():
        iss.add("format", "error", f"[{vol}] {doc}: glyphs missing in every font: "
                                   f"{', '.join(sorted(chars))}")

def check_pdf(pdf: Path, vol: str, iss: Issues) -> dict[str, float]:
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    pages = int(re.search(r"Pages:\s+(\d+)", info).group(1))
    w, h = (float(x) for x in re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info).groups())
    # the type area (style.tex geometry): 24 mm sides, 26 / 30 mm top / bottom, plus the header
    mm = 72 / 25.4
    # microtype hangs punctuation and hyphens up to ~4 pt into the margin (intended; real
    # overflow is caught exactly by the overfull-box check of the log)
    left, right = 24 * mm - 5.0, w - 24 * mm + 5.0
    bbox = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True,
                          text=True).stdout
    page, words, outside, sparse = 0, 0, defaultdict(int), []
    for line in bbox.splitlines():
        if line.lstrip().startswith("<page"):
            if page and words < 25 and page > 2:
                sparse.append(page)
            page, words = page + 1, 0
        m = re.match(r'\s*<word xMin="([\d.]+)" yMin="[\d.]+" xMax="([\d.]+)"', line)
        if m:
            words += 1
            x0, x1 = float(m.group(1)), float(m.group(2))
            if x0 < left or x1 > right:
                outside[page] += 1
    for p, n in sorted(outside.items()):
        iss.add("format", "error", f"[{vol}] page {p}: {n} word(s) outside the type area")
    if sparse:
        iss.add("format", "info", f"[{vol}] near-empty pages (< 25 words; chapter ends are "
                                  f"fine): {', '.join(map(str, sparse[:30]))}"
                                  f"{' …' if len(sparse) > 30 else ''}")
    fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
    for ln in fonts.splitlines()[2:]:
        cols = ln.split()
        if len(cols) >= 5 and "no" in cols[-5:-3]:
            iss.add("format", "error", f"[{vol}] font not embedded: {cols[0]}")
    return {"pages": pages}

# -- report ---------------------------------------------------------------------------------------
def write_report(out: Path, iss: Issues, built: dict[str, dict], stamp: str) -> Path:
    lines = ["# Docs PDF: build and check report", "", f"{stamp}", ""]
    if built:
        lines += ["| Volume | Documents | Pages | PDF |", "|---|---|---|---|"]
        for vol, b in built.items():
            lines.append(f"| {b['title']} | {b['docs']} | {b.get('pages', '–')} | "
                         f"`{b.get('pdf', 'failed')}` |")
        lines.append("")
    lines += [f"Errors: {iss.count('error')}, warnings: {iss.count('warn')}, "
              f"notes: {iss.count('info')}.", ""]
    for cat in ("sources", "links", "consistency", "up-to-date", "build", "format"):
        items = iss.items.get(cat, [])
        lines += [f"## {cat} ({len(items)})", ""]
        for level in ("error", "warn", "info"):
            for lv, msg in items:
                if lv == level:
                    lines.append(f"- **{lv}** {msg}")
        lines.append("")
    path = out / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

# -- main -----------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--volumes", action="store_true",
                    help="one PDF per part instead of the book")
    ap.add_argument("--part", action="append", choices=PART_ORDER,
                    help="only these parts (repeatable; implies --volumes)")
    ap.add_argument("--checks-only", action="store_true", help="source checks, no PDF")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "docs_pdf")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any error-level issue")
    a = ap.parse_args()
    volumes = a.volumes or bool(a.part)

    files = md_files()
    order = index_order(files)
    part = {f: part_of(f) for f in files}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--"]
                                + files, capture_output=True, text=True).stdout.strip())
    stamp = (f"Built {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from commit {commit}"
             + (" with uncommitted doc changes" if dirty else ""))

    iss = Issues()
    for f in files:
        if _member_match(f) is None:
            iss.add("sources", "warn", f"{f}: in no part of the book (PARTS in "
                                       "build_docs_pdf.py); placed in the overview")
    print(f"{len(files)} markdown files; checking sources …", flush=True)
    check_sources(files, iss)
    check_consistency(files, iss)
    check_up_to_date(files, iss)
    built: dict[str, dict] = {}
    a.out.mkdir(parents=True, exist_ok=True)
    if not a.checks_only:
        frags = a.out / "frags"
        if frags.exists():
            shutil.rmtree(frags)
        frags.mkdir()
        mdir = a.out / "mermaid"
        mdir.mkdir(exist_ok=True)
        shutil.copy(HERE / "style.tex", a.out / "style.tex")
        keys = {f: f"d{i:03d}" for i, f in enumerate(files)}
        vol_title = {f: (PART_BY_KEY[part[f]].title if volumes else "all") for f in files}
        docmap = a.out / "docmap.tsv"
        docmap.write_text("".join(f"{f}\t{keys[f]}\t{vol_title[f]}\n" for f in files),
                          encoding="utf-8")
        langs = subprocess.run(["pandoc", "--list-highlight-languages"], capture_output=True,
                               text=True).stdout.split()
        issues_file = a.out / "filter_issues.tsv"
        issues_file.write_text("")
        wanted = [k for k in PART_ORDER if not a.part or k in a.part]
        todo = [f for f in files if part[f] in wanted]
        print(f"pandoc: {len(todo)} files …", flush=True)
        with cf.ThreadPoolExecutor(max_workers=max(a.jobs, 2)) as ex:
            futs = {ex.submit(pandoc_fragment, f, keys[f], vol_title[f], docmap,
                              frags / f"{keys[f]}.tex", issues_file, ",".join(langs), mdir): f
                    for f in todo}
            for fut in cf.as_completed(futs):
                err = fut.result()
                for ln in err.splitlines():
                    if ln.strip():
                        iss.add("build", "warn", f"{futs[fut]}: pandoc: {ln.strip()[:200]}")
        owners: dict[str, str] = {}
        for ln in issues_file.read_text().splitlines():
            kind, _, msg = ln.partition("\t")
            if kind == "mermaid":
                doc, _, h = msg.partition("\t")
                owners[h] = doc
                continue
            iss.add("links" if kind in ("broken-link", "missing-image") else "build",
                    "error" if kind in ("broken-link", "missing-image") else "warn",
                    f"{kind}: {msg}")
        if owners:
            print(f"mermaid: {len(owners)} diagram(s) …", flush=True)
            render_mermaid(mdir, owners, a.jobs, iss)
        macros = highlight_macros()

        def member_rank(f: str) -> tuple:
            hit = _member_match(f)
            m = PART_BY_KEY[hit[0]].members[hit[1]] if hit else ""
            readme_first = not (f.endswith("README.md") and os.path.dirname(f) + "/" == m)
            # numbered docs (an07_…, h04_…) in number order; one number's docs (a subsystem
            # and its build guides) in index order; unnumbered ones after them
            num = re.match(r"([a-z]{0,3})(\d+)_", os.path.basename(f))
            group = (0, num.group(1), int(num.group(2))) if num else (1, "", 0)
            return (hit[1] if hit else 99, readme_first, group, order.get(f, 10_000), f)

        def docs_of(k: str) -> list[tuple[str, str]]:
            return [(f, keys[f]) for f in sorted((f for f in todo if part[f] == k),
                                                 key=member_rank)]

        def build(unit: str) -> tuple[str, dict]:
            if unit == "book":
                parts = [(PART_BY_KEY[k], docs_of(k)) for k in wanted]
                title, num, name = "Complete documentation", None, "jtsmm_docs"
            else:
                pt = PART_BY_KEY[unit]
                parts = [(pt, docs_of(unit))]
                num = PART_ORDER.index(unit) + 1
                title, name = pt.title, f"{num:02d}_{unit}"
            tex = a.out / f"{name}.tex"
            tex.write_text(book_tex(title, num, parts, stamp, macros), encoding="utf-8")
            ok, log = compile_volume(tex)
            info = {"title": title, "docs": sum(len(d) for _, d in parts), "log": log,
                    "ok": ok, "name": name}
            if ok:
                info["pdf"] = f"{name}.pdf"
            return unit, info

        units = wanted if volumes else ["book"]
        print(f"lualatex: {len(units)} PDF(s) …", flush=True)
        with cf.ThreadPoolExecutor(max_workers=min(a.jobs, len(units))) as ex:
            for unit, info in ex.map(build, units):
                built[unit] = info
                if not info["ok"]:
                    err = re.search(r"^! .*(?:\n.*){0,6}", info["log"], re.M)
                    iss.add("build", "error", f"[{unit}] LaTeX failed: "
                                              f"{err.group(0) if err else info['log'][-600:]}")
                    continue
                check_log(info["log"], unit, iss)
                built[unit].update(check_pdf(a.out / info["pdf"], unit, iss))
                print(f"  {info['pdf']}: {built[unit]['pages']} pages", flush=True)
        for b in built.values():
            b.pop("log", None)
    rep = write_report(a.out, iss, built, stamp)
    print(f"errors {iss.count('error')}, warnings {iss.count('warn')}, notes "
          f"{iss.count('info')} -> {rep}")
    return 1 if (a.strict and iss.count("error")) else 0

if __name__ == "__main__":
    sys.exit(main())
