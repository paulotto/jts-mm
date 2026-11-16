# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Repository docs: relative links, anchors and stale facts.

* Every relative markdown link (inline, image, reference definition, or
  ``<a href>``) in the documentation set must point at an existing file or
  directory, and a ``#fragment`` into a markdown file must match one of its
  GitHub-style heading slugs (or an explicit ``<a id/name>`` anchor).
* No doc may describe 30 Hz as *the* default trigger/camera rate: the operator
  template triggers at 40 Hz since Oct 2026. The deliberate 30 Hz values (the
  ``imx296_dual_csi`` profile / firmware boot preset, raw ``trig_start``,
  ``track_a_led``, ``optical calibrate``) are allowed.
* ``linear_accel`` (the BNO085 gravity-free report) was never recorded in the
  study profile (``kinematics_accel`` records ``accel`` with gravity). Only
  the docs that describe the optional report itself may mention it.

Run from anywhere: ``pytest -q tests`` at the repo root.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO = Path(__file__).resolve().parents[1]

def _doc_files() -> list[Path]:
    files: set[Path] = set()
    files.update((REPO / "docs").rglob("*.md"))
    files.update((REPO / "experiments").rglob("README.md"))
    files.update((REPO / "software").glob("*/README.md"))
    files.update((REPO / "hardware").glob("*/README.md"))
    files.update((REPO / "hardware" / "mechanical").rglob("README.md"))
    names = ["README.md"]
    for name in names:
        if (REPO / name).is_file():
            files.add(REPO / name)
    skip = ("/.venv/", "/node_modules/", "/site-packages/", "/libs/")
    # relative to the checkout (so a checkout below a hidden folder is still scanned); files
    # below a hidden folder of the checkout (tool settings, worktrees) are skipped
    return sorted(p for p in files
                  if not any(s in "/" + p.relative_to(REPO).as_posix() for s in skip)
                  and not any(part.startswith(".") for part in p.relative_to(REPO).parts[:-1]))

DOCS = _doc_files()

def _rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()

# --------------------------------------------------------------------------- #
# Markdown parsing (just enough of CommonMark / GFM)
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

def _strip_code(text: str) -> list[tuple[int, str]]:
    """(line number, line) outside fenced code, with inline code blanked."""
    text = _HTML_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    out: list[tuple[int, str]] = []
    fence: str | None = None
    for i, line in enumerate(text.splitlines(), 1):
        m = _FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)[0] * len(m.group(1))
            continue
        if fence is not None:
            if line.strip().startswith(fence) and set(line.strip()) <= {fence[0]}:
                fence = None
            continue
        out.append((i, _INLINE_CODE.sub(lambda mm: " " * len(mm.group(0)), line)))
    return out

# [text](target "title") and ![alt](target); one level of nested brackets in the text.
_INLINE_LINK = re.compile(
    r"!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]\(\s*(<[^>]*>|[^)\s]+(?:\([^)\s]*\)[^)\s]*)*)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
_HREF = re.compile(r"""<(?:a|img)\b[^>]*?\b(?:href|src)\s*=\s*["']([^"']+)["']""", re.I)

def _links(path: Path) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for n, line in _strip_code(path.read_text(encoding="utf-8")):
        for m in _INLINE_LINK.finditer(line):
            found.append((n, m.group(1)))
        m = _REF_DEF.match(line)
        if m:
            found.append((n, m.group(1)))
        for m in _HREF.finditer(line):
            found.append((n, m.group(1)))
    return [(n, t[1:-1] if t.startswith("<") and t.endswith(">") else t) for n, t in found]

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_ANCHOR_TAG = re.compile(r"""<a\b[^>]*\b(?:id|name)\s*=\s*["']([^"']+)["']""", re.I)
_MD_LINK_TEXT = re.compile(r"!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")

def github_slug(heading: str) -> str:
    """GitHub's heading anchor: rendered text, lower-cased, punctuation dropped."""
    text = _MD_LINK_TEXT.sub(r"\1", heading)
    text = _HTML_TAG.sub("", text)
    text = text.replace("`", "")
    text = re.sub(r"(\*\*|__|\*|~~)", "", text)
    # '_' emphasis around words renders away, but a literal '_' inside a word stays.
    text = re.sub(r"(?<![\w])_(?=\S)|(?<=\S)_(?![\w])", "", text)
    text = text.strip().lower()
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "-_ " or cat[0] in ("L", "N") or cat == "Mn":
            out.append("-" if ch == " " else ch)
    return "".join(out)

@lru_cache(maxsize=None)
def anchors(path: Path) -> frozenset[str]:
    seen: dict[str, int] = {}
    result: set[str] = set()
    lines = _strip_code(path.read_text(encoding="utf-8"))
    raw = path.read_text(encoding="utf-8").splitlines()
    for idx, (n, line) in enumerate(lines):
        result.update(m.group(1) for m in _ANCHOR_TAG.finditer(raw[n - 1]))
        m = _HEADING.match(raw[n - 1])
        title = m.group(2) if m else None
        if title is None and idx + 1 < len(lines):
            nxt_n, _ = lines[idx + 1]
            if nxt_n == n + 1 and _SETEXT.match(raw[nxt_n - 1]) and line.strip() \
                    and not line.lstrip().startswith(("|", "-", "*", ">")):
                title = raw[n - 1].strip()
        if title is None:
            continue
        slug = github_slug(title)
        k = seen.get(slug, 0)
        seen[slug] = k + 1
        result.add(slug if k == 0 else f"{slug}-{k}")
    return frozenset(result)

_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

def _check(path: Path) -> list[str]:
    errors: list[str] = []
    for n, target in _links(path):
        if not target or _SCHEME.match(target) or target.startswith("//"):
            continue
        file_part, _, frag = target.partition("#")
        file_part = unquote(file_part.split("?", 1)[0])
        if file_part.startswith("/"):
            dest = REPO / file_part.lstrip("/")
        elif file_part:
            dest = (path.parent / file_part).resolve()
        else:
            dest = path
        where = f"{_rel(path)}:{n}: {target}"
        if not dest.exists():
            errors.append(f"{where} -> missing {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
            continue
        if frag and dest.is_file() and dest.suffix.lower() == ".md":
            if unquote(frag).lower() not in anchors(dest) and frag not in anchors(dest):
                errors.append(f"{where} -> no anchor #{frag} in {_rel(dest)}")
    return errors

def test_doc_set_is_not_empty():
    assert len(DOCS) > 20

@pytest.mark.parametrize("doc", DOCS, ids=_rel)
def test_relative_links_and_anchors_resolve(doc: Path):
    errors = _check(doc)
    assert not errors, "broken links:\n  " + "\n  ".join(errors)

def test_github_slug_examples():
    assert github_slug("3.3 Trigger rate (40 Hz)") == "33-trigger-rate-40-hz"
    assert github_slug("`jtsmm record` — options") == "jtsmm-record--options"
    assert github_slug("9. Fiducials & scene lighting") == "9-fiducials--scene-lighting"
    assert github_slug("mand_point_1 and **bold**") == "mand_point_1-and-bold"
    assert github_slug("[Link](x.md) text") == "link-text"

# --------------------------------------------------------------------------- #
# Stale facts
# --------------------------------------------------------------------------- #

_DEFAULT_30HZ = re.compile(
    r"default[^.;|\n]{0,60}?\b30\s*(?:Hz|fps)\b|\b30\s*(?:Hz|fps)\b[^.;|\n]{0,25}?\bdefault",
    re.I,
)
# A 30 Hz default is right when it names one of these on the same line.
_DELIBERATE_30HZ = re.compile(
    r"preset|imx296|firmware|trig_start|track_a_led|calibrat|boot|profile|webcam|C920|Track A",
    re.I,
)

def test_no_doc_calls_30hz_the_default_rate():
    hits = []
    for doc in DOCS:
        raw = doc.read_text(encoding="utf-8").splitlines()
        for n, line in _strip_code(doc.read_text(encoding="utf-8")):
            for m in _DEFAULT_30HZ.finditer(line):
                # the allow-list also looks inside inline code (`trig_start`, paths)
                if not _DELIBERATE_30HZ.search(raw[n - 1]):
                    hits.append(f"{_rel(doc)}:{n}: …{m.group(0)}…")
    assert not hits, "stale '30 Hz default' (the template triggers at 40 Hz):\n  " + "\n  ".join(hits)

# Docs that describe the optional BNO085 linear_acceleration report itself.
_LINEAR_ACCEL_OK = {
    "docs/hardware/h04_imu_subsystem.md",
    "docs/software/s02_data_pipeline.md",
    "docs/software/s05_imu_processing.md",
    "docs/software/s08_runtime_configuration.md",
}
_LINEAR_ACCEL = re.compile(r"\blinear_accel\b(?!_from_raw)")

def test_linear_accel_only_where_the_optional_report_is_described():
    hits = []
    for doc in DOCS:
        rel = _rel(doc)
        if rel.startswith("docs/ml/") or rel in _LINEAR_ACCEL_OK:
            continue
        for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            if _LINEAR_ACCEL.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:120]}")
    assert not hits, (
        "linear_accel was never recorded (kinematics_accel records accel with gravity; "
        "use jtsmm.imu.gravity_lin_accel.linear_accel_from_raw):\n  " + "\n  ".join(hits)
    )
