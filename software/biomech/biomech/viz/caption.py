# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Structured, self-explanatory captions.

A view describes its figure or table once, in parts, and the caption is composed from them,
so every caption states the same things in the same order:

* ``shows``: what the item shows (one sentence);
* ``marks``: what each mark means -- ("dots", "participants"), ("diamond", "median"),
  ("line", "95 % CI"), ("bar above the curves", "SPM cluster") ...;
* ``n``: the sample ("14 participants, 8 cycles each");
* ``statistic``: the summary and interval ("median and BCa 95 % CI");
* ``test``: the test and its correction ("one-sided Wilcoxon signed-rank test, Holm");
* ``evidence``: pre-specified / post hoc / exploratory;
* ``notes``: footnotes ("* CMC undefined: ...").

``codes`` keep the analysis-plan codes for the analyst (never rendered). :func:`attach`
stores the parts in ``spec.meta["caption"]`` and the rendered text in ``spec.caption``, so
the export, the viewer and the integrity check keep working on ``caption``.
:func:`expand_abbreviations` spells out each glossary abbreviation at its first use; the
reader wording (:func:`vocab.reader_spec`) applies it to every caption.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

GLOSSARY_PATH = Path(__file__).resolve().parents[1] / "etc" / "glossary.yaml"


@dataclass
class CaptionSpec:
    shows: str
    marks: list[tuple[str, str]] = field(default_factory=list)
    n: str = ""
    statistic: str = ""
    test: str = ""
    evidence: str = ""
    notes: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)


def _sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    if not re.match(r"^[a-z]\s*[=<>≤≥]", s):        # "n = 14" keeps its symbol case
        s = s[:1].upper() + s[1:]
    return s if s.endswith((".", "!", "?")) else s + "."


def render(cap: CaptionSpec) -> str:
    """The caption text: shows. One sentence per mark. n. Statistic. Test. Evidence. Notes.
    (Plain sentences since 2026-11-10: no semicolons; :func:`plain_prose` is applied to the
    reader text.)"""
    parts = [_sentence(cap.shows)]
    parts += [_sentence(f"{g}: {m}") for g, m in cap.marks]
    for x in (cap.n, cap.statistic, cap.test, cap.evidence):
        if x:
            parts.append(_sentence(x))
    parts += [_sentence(x) for x in cap.notes if x]
    return " ".join(p for p in parts if p)


_RANGE_NUM = re.compile(r"(?<=[\d%°])\s*–\s*(?=[−+\d]|[A-Z]\d)")
_RANGE_LETTER = re.compile(r"(?<=[(,\s])([a-h])–([a-h])(?=[),\s])")


def plain_prose(text: str) -> str:
    """Text rules for captions and table notes (2026-11-10): no semicolons, no dashes,
    plain sentences. A semicolon outside parentheses ends the sentence, inside them it
    becomes a comma; a number or panel range "0–12°", "(b–f)" reads "0 to 12°", "(b to f)";
    a spaced em or en dash becomes a comma. Name compounds (Bland–Altman, Teager–Kaiser)
    keep their dash."""
    if not text:
        return text
    t = _RANGE_LETTER.sub(r"\1 to \2", _RANGE_NUM.sub(" to ", text))
    t = re.sub(r"\s+[—–]\s+", ", ", t)
    out, depth, i = [], 0, 0
    while i < len(t):
        ch = t[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(depth - 1, 0)
        if ch == ";":
            if depth:
                out.append(",")
                i += 1
                continue
            else:
                out.append(".")
                j = i + 1
                while j < len(t) and t[j] == " ":
                    j += 1
                out.append(" ")
                # capitalise the next word unless it is a symbol ("p = ", "n = ") or a unit
                if j < len(t) and t[j].islower() and not re.match(r"[a-z]\s*[=<>≤≥]", t[j:]):
                    out.append(t[j].upper())
                    j += 1
                i = j
                continue
        out.append(ch)
        i += 1
    t = re.sub(r"\.\.", ".", "".join(out))
    # a sentence that opens with panel letters starts with a capital after them: "(b, c) one
    # value" -> "(b, c) One value" (not a symbol such as "n = " or "p = ")
    return _PANEL_START.sub(lambda m: m.group(1) + m.group(2).upper(), t)


_PANEL_START = re.compile(r"((?:^|[.!?]\s+)\([a-z](?:(?:, | to | and )[a-z])*\)\s+)"
                          r"(?!(?:and|or|to)\b)([a-z])(?![\w-]*\s*[=<>≤≥])")


def attach(spec: Any, cap: CaptionSpec) -> Any:
    """Store ``cap`` in ``spec.meta["caption"]`` and its text in ``spec.caption``."""
    spec.caption = render(cap)
    meta = getattr(spec, "meta", None)
    if isinstance(meta, dict):
        meta["caption"] = asdict(cap)
    return spec


@lru_cache(maxsize=1)
def glossary() -> dict[str, str]:
    import yaml

    doc = yaml.safe_load(GLOSSARY_PATH.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in (doc.get("abbreviations") or {}).items()}


def expand_abbreviations(text: str) -> str:
    """Each glossary abbreviation spelled out at its first use ("long form (ABBR)"), unless
    the text already gives the long form; later uses stay short."""
    if not text:
        return text
    out = text
    for abbr, long in glossary().items():
        rx = re.compile(rf"(?<![\w(]){re.escape(abbr)}(?![\w])")
        m = rx.search(out)
        if m is None or long.lower() in out.lower():
            continue
        # a sentence start, also after a panel letter ("(c) RMS ..." -> "(c) Root mean ...")
        start = m.start() == 0 or bool(re.search(r"(^|[.!?]\s+)(\([a-z](, [a-z])*\)\s+)?$",
                                                 out[:m.start()]))
        text = (long[:1].upper() + long[1:]) if start else long
        out = out[:m.start()] + f"{text} ({abbr})" + out[m.end():]
    return out
