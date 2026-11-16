# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reader-facing text of figures and tables.

A paper item must read on its own: no internal plan codes (``D6``, ``VD3``), no raw
identifiers (``movement_pooled``, ``f95_coh_hz``), no raw values (``True``, ``nan``,
``n/a``), no column label guessed from a column name, every categorical value known to the
vocabulary, and no label or sentence left empty or gapped by a dropped code
(:func:`dropped_code_issues`). :func:`item_issues` builds one item and lists what it finds;
``check-outputs`` runs it over the study's paper sets (paper items fail, dashboards are reported).
"""

from __future__ import annotations

import re
from typing import Any

# A lower-case word with an underscore (a Python/stage identifier). LaTeX math such as
# "M_Q" or "σ_ref" starts with a capital or a non-ASCII letter and is not matched.
RAW_ID_RE = re.compile(r"(?<![\w\\{$])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w}$])")
RAW_VALUE_RE = re.compile(r"(?<![\w/])(nan|NaN|n/a)(?![\w/])")
RAW_WHOLE = {"True", "False", "None", "nan", "NaN", "inf", "-inf"}


def _all_texts(spec: Any) -> list[str]:
    """Every reader-facing slot of a FigureSpec / TableSpec, empty ones included, in a fixed
    order (a spec and its :func:`~biomech.viz.views.vocab.reader_spec` copy line up)."""
    from biomech.viz.spec import FigureSpec

    out = [getattr(spec, "title", ""), getattr(spec, "caption", ""), getattr(spec, "notes", "")]
    if isinstance(spec, FigureSpec):
        for p in spec.panels:
            out += [p.title, p.note]
            for a in (p.x, p.y, p.z):
                if a is not None:
                    out += [a.label, *(a.ticklabels or [])]
            if p.colorbar is not None:
                out.append(p.colorbar.label)
            for s in p.series:
                out += [s.label or "", *(s.text or [])]
    else:
        out += [c.header for c in spec.columns] + [c.group or "" for c in spec.columns]
        out += [v for r in spec.rows for v in r.values() if isinstance(v, str)]
    return [str(t) if t else "" for t in out]


def reader_texts(spec: Any) -> list[str]:
    """Every reader-facing string of a FigureSpec / TableSpec (title, caption, notes, panel
    titles, axis labels and tick labels, legend entries, in-panel text, table headers and
    text cells)."""
    return [t for t in _all_texts(spec) if t]


def dropped_code_issues(raw: Any, shown: Any) -> list[str]:
    """Where the reader wording loses content: a label, tick label or table cell that only
    held a dropped code (errata / plan-item codes such as ``E3b`` have no reader word) comes
    out empty, and a dropped code between two words leaves a gap in the sentence ("the slope
    of E2 against time" -> "the slope of against time"). ``raw`` is the built spec, ``shown``
    its reader copy."""
    from biomech.viz.views import vocab as V

    before, after = _all_texts(raw), _all_texts(shown)
    emptied = sorted({b.strip() for b, a in zip(before, after) if b.strip() and not a.strip()})
    gaps = sorted({m.group(0) for t in before for m in V.CODE_RE.finditer(t)
                   if V.CODE_WORD[m.group(1)] is None
                   and re.search(r"[A-Za-z]\s+$", t[:m.start()])
                   and re.match(r"\s+[a-z]", t[m.end():])})
    issues = []
    if emptied:
        issues.append(f"labels emptied by dropping plan codes: {', '.join(emptied[:12])}")
    if gaps:
        issues.append(f"plan codes dropped mid-sentence (name what they are): "
                      f"{', '.join(gaps[:12])}")
    return issues


def raw_id_cells(spec: Any) -> dict[str, list[str]]:
    """Where the raw identifiers of ``spec`` sit: table column key -> identifiers (for a
    figure: its texts under "figure"), so the vocabulary gap can be closed at the column."""
    from biomech.viz.spec import TableSpec

    out: dict[str, set[str]] = {}
    if isinstance(spec, TableSpec):
        for r in spec.rows:
            for k, v in r.items():
                if isinstance(v, str):
                    out.setdefault(str(k), set()).update(m.group(1) for m in
                                                         RAW_ID_RE.finditer(v))
        texts = [spec.caption or "", spec.notes or ""] + [c.header for c in spec.columns]
    else:
        texts = reader_texts(spec)
    out.setdefault("figure" if not isinstance(spec, TableSpec) else "text", set()).update(
        m.group(1) for t in texts for m in RAW_ID_RE.finditer(t))
    return {k: sorted(v) for k, v in out.items() if v}


def text_issues(spec: Any) -> list[str]:
    """Plan codes, raw identifiers and raw values in the reader-facing text of ``spec``
    (labels guessed from column names are caught while building: :func:`item_issues`)."""
    from biomech.viz.views import vocab as V

    texts = reader_texts(spec)
    issues: list[str] = []
    codes = sorted({c for t in texts for c in V.internal_codes(t)})
    if codes:
        issues.append(f"plan codes: {', '.join(codes)}")
    ids = sorted({m.group(1) for t in texts for m in RAW_ID_RE.finditer(t)})
    if ids:
        issues.append(f"raw identifiers: {', '.join(ids[:12])}")
    vals = sorted({m.group(1) for t in texts for m in RAW_VALUE_RE.finditer(t)}
                  | {t.strip() for t in texts if t.strip() in RAW_WHOLE})
    if vals:
        issues.append(f"raw values: {', '.join(vals)}")
    return issues


# -- one meaning per colour within a figure (colour harmonisation 2026-11-05) ---------------------
# A hue may carry one meaning per figure; the meaning of a series is its role (dashed / dotted
# variants ``:alt`` / ``:dot`` count as the same). Neutral roles (ink, greys, references) and
# the labelled phase lanes (``hspan``) are exempt. Roles that are one concept by design:
COLOUR_SAME = {"entity:masseter_r": "group:closers", "entity:digastric_r": "group:openers"}
COLOUR_NEUTRAL = {"accent", "context", "muted", "faint", "flagged", "reference", "threshold",
                  "band", "noise"}


def colour_meaning_issues(spec: Any) -> list[str]:
    """Colours that carry two meanings (roles) in one figure."""
    from collections import defaultdict

    from biomech.viz import style as ST

    def base(role: str) -> str:
        r = role or ""
        for suf in (":alt", ":dot", ":fill"):   # a line and its fill are one meaning
            r = r[: -len(suf)] if r.endswith(suf) else r
        return COLOUR_SAME.get(r, r)

    use: dict[str, set[str]] = defaultdict(set)
    for p in getattr(spec, "panels", None) or []:
        for s in p.series:
            r = base(s.role)
            # colormap-coloured kinds carry no role colour
            if s.kind in ("hspan", "text", "image", "heatmap", "surface3d") or r in COLOUR_NEUTRAL:
                continue
            explicit = (s.style or {}).get("color")
            # an explicit colour is the meaning itself (e.g. a muscle's colour on both systems,
            # the role then only sets the line style)
            c = explicit or (ST.ROLES.get(s.role or "") or (None,))[0]
            if c and c not in ("ink", "ink2", "grey", "lightgrey", "black"):
                use[c].add(f"colour:{explicit}" if explicit else r)
    return [f"colour {c} has {len(r)} meanings in one figure: {', '.join(sorted(r))}"
            for c, r in sorted(use.items()) if len(r) > 1]


def item_issues(ctx: Any, view_id: str, params: dict[str, Any],
                spec_out: list[Any] | None = None) -> list[str]:
    """Build one item (as the export does: reader wording applied) and list its text issues,
    the catalog labels it guessed and the categorical values the vocabulary lacks. The built
    spec is appended to ``spec_out`` when given (further checks without a second build)."""
    from biomech import catalog
    from biomech.viz.views import registry as R
    from biomech.viz.views import vocab as V

    catalog.misses(reset=True)
    V.unknown_terms(reset=True)
    raw = R.build(view_id, R.request(ctx, view_id, **params))
    spec = V.reader_spec(raw)
    if spec_out is not None:
        spec_out.append(spec)
    issues = text_issues(spec) + dropped_code_issues(raw, spec) + colour_meaning_issues(spec)
    miss = sorted(catalog.misses(reset=True))
    if miss:
        issues.append(f"labels guessed (no catalog entry): {', '.join(miss[:12])}")
    unk = sorted(f"{d}:{v}" for d, v in V.unknown_terms(reset=True))
    if unk:
        issues.append(f"values without a vocabulary entry: {', '.join(unk[:12])}")
    if not (getattr(spec, "meta", None) or {}).get("caption"):
        issues.append(UNSTRUCTURED)
    return issues


# Reported separately (a warning while the paper items migrate to CaptionSpec).
UNSTRUCTURED = "caption not structured (viz.caption.CaptionSpec)"


# -- caption rubric (J1.2) ----------------------------------------------------------------
CAPTION_MIN_WORDS = 25        # below: the caption cannot say what, how, n and the statistic
CAPTION_MAX_WORDS = 250       # above: split the figure or move details into the notes ...
CAPTION_WORDS_PER_PANEL = 50  # ... plus this per panel beyond two (multi-panel journal figures)
CAPTION_MAX_CAP = 400
# Abbreviations in column headers must be spelled out in the caption or the table note
# (the glossary does it at first use in the caption); symbols ($...$) need the note.
# Hyphenated names (the system JTS-MM) are names, not abbreviations.
ABBREV_RE = re.compile(r"(?<![\w$\\-])([A-Z]{2,}[a-z]?)(?![\w-])")


def caption_rubric_issues(spec: Any) -> list[str]:
    """What keeps a built (reader-worded) caption from being self-explanatory without the
    text: too short or too long, a figure without its encodings (``marks``), no statistic,
    no n, a table whose symbols / abbreviations have no note. Purely descriptive items
    (protocol, schematic) state "n" and the encodings only."""
    from biomech.viz.spec import TableSpec

    cap = (getattr(spec, "meta", None) or {}).get("caption") or {}
    text = getattr(spec, "caption", "") or ""
    words = len(text.split())
    out: list[str] = []
    if words < CAPTION_MIN_WORDS:
        out.append(f"caption short ({words} words < {CAPTION_MIN_WORDS})")
    panels = len(getattr(spec, "panels", []) or [])
    limit = min(CAPTION_MAX_CAP, CAPTION_MAX_WORDS + CAPTION_WORDS_PER_PANEL * max(0, panels - 2))
    if words > limit:
        out.append(f"caption long ({words} words > {limit}): split or use notes")
    if not cap:
        return out
    if not cap.get("n"):
        out.append("no n")
    if isinstance(spec, TableSpec):
        heads = " ".join(f"{c.header} {c.group or ''}" for c in spec.columns)
        told = f"{text} {spec.notes or ''}"
        unexplained = sorted({a for a in ABBREV_RE.findall(heads) if f"({a})" not in told
                              and f"{a}:" not in told and f"{a} =" not in told})
        if unexplained:
            out.append(f"table abbreviations not spelled out: {', '.join(unexplained)}")
        if "$" in heads and not spec.notes:
            out.append("table symbols without a note")
        if not cap.get("statistic"):
            out.append("no statistic")
    else:
        if not cap.get("marks"):
            out.append("encodings not stated (marks)")
        if not cap.get("statistic") and not cap.get("evidence") == "descriptive":
            out.append("no statistic")
    return out
