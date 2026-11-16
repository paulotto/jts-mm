# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One vocabulary for every view: entity order and labels, units, catalog-driven axes.

Views import names from here instead of keeping their own copies, so a muscle, task or
unit reads the same in every panel, renderer and export. Column labels and units come
from ``etc/catalog.yaml`` via :func:`axis` / :func:`label`; a column without a catalog
entry falls back to its name (``check-outputs`` reports those).
"""

from __future__ import annotations

import copy
import re
from functools import lru_cache
from typing import Any

from biomech import catalog
from biomech.viz.spec import Axis, Colorbar

VOCAB_PATH = catalog.CATALOG_PATH.parent / "vocab.yaml"


@lru_cache(maxsize=1)
def vocabulary() -> tuple[dict[str, dict[str, dict[str, str]]], dict[str, tuple[str, ...]]]:
    """``(domains, columns)`` of ``etc/vocab.yaml``; every entry normalized to
    ``{label, short, title}``; a column maps to one domain or to several, tried in order
    (``variant`` holds clock, tip, alignment and decoder variants)."""
    import yaml

    doc = yaml.safe_load(VOCAB_PATH.read_text(encoding="utf-8")) or {}
    domains: dict[str, dict[str, dict[str, str]]] = {}
    for dom, entries in (doc.get("domains") or {}).items():
        out = {}
        for k, v in (entries or {}).items():
            e = {"label": str(v)} if not isinstance(v, dict) else {a: str(b) for a, b in v.items()}
            e.setdefault("short", e["label"])
            e.setdefault("title", e["label"][:1].upper() + e["label"][1:])
            out[str(k)] = e
        domains[str(dom)] = out
    return domains, {str(k): tuple(str(d) for d in (v if isinstance(v, list) else [v]))
                     for k, v in (doc.get("columns") or {}).items()}


def term(domain: str, value: Any, form: str = "label") -> str:
    """Reader name of a stored categorical ``value`` in ``domain`` (``form``: label, short,
    title). Unknown values read as the value with underscores spaced (``check-outputs``
    reports them for paper items through :func:`unknown_terms`)."""
    return term_in((domain,), value, form)


def term_in(domains: tuple[str, ...], value: Any, form: str = "label") -> str:
    """:func:`term` over several domains (the first that knows the value). A composite
    value without an entry is read part by part: ``a|b`` -> "a, b" and ``key=value`` ->
    "key: value" (parts joined by "; ") with the key's label and the value's reader name (sensitivity
    variants such as ``detector=envelope|onset=drive``; keys named by the domain
    ``variant_key``); ``a,b`` (flag lists) like ``a|b``."""
    v = str(value)
    vocab = vocabulary()[0]
    for d in domains:
        e = vocab.get(d, {}).get(v)
        if e is not None:
            return e[form]
    if v in catalog.load() and "column" not in domains:   # a metric name used as a value
        return catalog.load()[v].label
    for d in domains:               # already a reader name (a view translated it): as is
        if any(v in (e["label"], e["short"], e["title"]) for e in vocab.get(d, {}).values()):
            return v
    if any(c in v for c in "|=,"):
        keys = vocab.get("variant_key", {})
        parts = []
        for p in re.split(r"[|,]", v):
            k, eq, x = p.partition("=")
            if eq:
                key = keys[k]["label"] if k in keys else \
                    (catalog.load()[k].label if k in catalog.load() else k)
                parts.append(f"{key}: {cell(k, x) if k in vocabulary()[1] else term_in(domains, x, form)}")
            else:
                parts.append(term_in(domains, p, form))
        return ("; " if "=" in v else ", ").join(parts)
    _MISSES.add((domains[0] if domains else "", v))
    return v.replace("_", " ")


def labels(domain: str, form: str = "label") -> dict[str, str]:
    return {k: e[form] for k, e in vocabulary()[0].get(domain, {}).items()}


_MISSES: set[tuple[str, str]] = set()


def unknown_terms(reset: bool = False) -> set[tuple[str, str]]:
    """(domain, value) pairs looked up without a vocabulary entry since the last reset."""
    out = set(_MISSES)
    if reset:
        _MISSES.clear()
    return out


# Identifiers inside free text (composite keys, notes): the same pattern the reader-text
# check flags (check/text.py RAW_ID_RE).
_ID_RE = re.compile(r"(?<![\w\\{$])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\w}$])")
# Domains searched for an identifier inside text, in this order (a value of an earlier
# domain wins); catalog columns come first.
TEXT_DOMAINS = ("task", "muscle", "landmark", "hinge_model", "feature_set", "rule",
                "sensitivity_variant", "emg_norm", "ref_chain", "body", "estimator",
                "method", "condyle_reason", "usability_flag", "text_term")


@lru_cache(maxsize=1)
def _token_names() -> dict[str, str]:
    vocab = vocabulary()[0]
    out: dict[str, str] = {}
    for d in reversed(TEXT_DOMAINS):
        out.update({k: e["label"] for k, e in vocab.get(d, {}).items()})
    out.update({k: e.label for k, e in catalog.load().items()})
    return out


def rewrite_ids(text: str) -> str:
    """Known identifiers inside a text cell replaced by their reader names (catalog label,
    else the first vocabulary domain that knows them): ``median_pct|open_close`` ->
    "Median, open-close"-style cells. Unknown identifiers stay (the check reports them)."""
    names = _token_names()

    def one(tok: str) -> str:
        if tok in names:
            return names[tok]
        a, vs, b = tok.partition("_vs_")             # "emg_imu_vs_imu": a comparison
        if vs and a in names and b in names:
            return f"{names[a]} vs {names[b]}"
        return tok

    out = _ID_RE.sub(lambda m: one(m.group(1)), text)
    if "|" in text and re.fullmatch(r"[\w|.]+", text):   # a pure composite key
        out = out.replace("|", ", ")
    return out


@lru_cache(maxsize=1)
def _catalog_labels() -> frozenset[str]:
    return frozenset(e.label for e in catalog.load().values())


def cell(column: str, value: Any) -> Any:
    """A table cell in reader form: categorical values of a column with a vocabulary domain
    become their reader names, values of ``column``-domain columns their catalog labels;
    anything else is returned unchanged."""
    if not isinstance(value, str) or not value:
        return value
    doms = vocabulary()[1].get(column)
    if doms is None:
        return rewrite_ids(value) if "_" in value else value
    if "column" in doms:                # column names first, then the column's other domains
        if value not in catalog.load() and value in _catalog_labels():
            return value                    # already a reader name
        rest = tuple(d for d in doms if d != "column")
        if value in catalog.load() or not rest:
            return catalog.get(value).label
        return term_in(rest, value)
    return term_in(doms, value)


_CHOICE_DOMAIN = {"form": "form", "frames": "frames_choice", "scope": "requirements_scope",
                  "lead": "lead_source"}


def picker_label(name: str) -> str:
    """Reader name of a view picker (the viewer control): "form" -> "Depiction"."""
    e = vocabulary()[0].get("picker", {}).get(name)
    return e["title"] if e else name.replace("_", " ").capitalize()


def choice_label(name: str, value: Any) -> str:
    """Reader name of a picker value (a form, a frame choice, a task, a column ...)."""
    doms = (_CHOICE_DOMAIN[name],) if name in _CHOICE_DOMAIN else \
        vocabulary()[1].get(name, ())
    if "column" in doms and isinstance(value, str) and (value in catalog.load()
                                                        or doms == ("column",)):
        return catalog.get(value).label
    for d in doms:
        e = vocabulary()[0].get(d, {}).get(str(value))
        if e:
            return e["label"]
    return str(value).replace("_", " ")


# Display order (right before left, closers before openers) -- not the recording order.
MUSCLES = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l", "digastric_r",
           "digastric_l")
MUSCLE_LABEL = {m: term("muscle", m) for m in MUSCLES}
OPENERS = ("digastric_r", "digastric_l")
CLOSERS = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l")

TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
TASK_LABEL = labels("task")
# Recorded phases on a timeline (R5b): the movement tasks keep their fixed categorical
# hues, rest takes the sixth; calibration / landmark holds and sync windows are neutral
# and carry a text label instead of a hue (the palette has six hues).
PHASE_ROLE = {**{t: f"cat:{i}" for i, t in enumerate(TASKS)}, "rest": "cat:5"}
PHASE_SHORT = labels("task", "short")


def phase_role(name: str) -> str:
    return PHASE_ROLE.get(name, "faint")


def phase_label(name: str, short: bool = False) -> str:
    return term("task", name, "short" if short else "label")


SPECTRAL_TASKS = ("open_close", "chewing", "free_motion", "mvc_opening", "mvc_closing")
STAT_LABEL = labels("stat")
PERSP_LABEL = labels("perspective")
MODEL_LABEL = labels("ml_model")
FSET_LABEL = labels("feature_set")

# siunitx units
MM, MMS = "\\milli\\metre", "\\milli\\metre\\per\\second"
MS2 = "\\metre\\per\\second\\squared"
DEG, DPS, DPS2 = "\\degree", "\\degree\\per\\second", "\\degree\\per\\second\\squared"
PCT, SEC, MSEC, HZ, DB, UV = ("\\percent", "\\second", "\\milli\\second", "\\hertz",
                              "\\decibel", "\\micro\\volt")
PER_MIN = "\\per\\minute"
# The normalized EMG envelope: % of the largest activity recorded (analysis_plan D11).
EMG_NORM = "Activation (% of max recorded)"
EMG_NORM_SHORT = "% of max recorded"


def task_label(t: str) -> str:
    return term("task", t)


def muscle_label(m: str) -> str:
    return term("muscle", m)


def label(column: str) -> str:
    return catalog.get(column).label


def unit(column: str) -> str:
    return catalog.get(column).unit


def axis(column: str, label_: str | None = None, **kw) -> Axis:
    """Axis titled and unit-tagged from the catalog entry of ``column``."""
    e = catalog.get(column)
    return Axis(label_ if label_ is not None else e.label, e.unit, **kw)


def colorbar(column: str, label_: str | None = None, **kw) -> Colorbar:
    e = catalog.get(column)
    return Colorbar(label_ if label_ is not None else e.label, e.unit, **kw)


def fmt(column: str, value: float) -> str:
    """The catalog's rounding of one value (``--`` for NaN)."""
    return catalog.get(column).fmt(value)


def known(column: str) -> bool:
    return column in catalog.load()


# -- reader-facing wording (D2) ---------------------------------------------------------------
# Internal plan codes (analysis_plan.md, research plan) mean nothing to a reader: claims and
# secondary / amended analyses (C1, S2, A4) were pre-specified, deviations (D6) are post hoc,
# X-analyses exploratory; errata (E17) and plan items (G3, W2, R4) have no reader wording and
# are dropped. Pseudonyms (P3) and percentiles (P95) are not codes.
CODES_PATH = catalog.CATALOG_PATH.parent / "codes.yaml"


def _load_codes() -> dict[str, str | None]:
    import yaml

    doc = yaml.safe_load(CODES_PATH.read_text(encoding="utf-8")) or {}
    return {str(k): v for k, v in (doc.get("families") or {}).items()}


CODE_WORD = _load_codes()
# Longest family prefix first, so "VD3" is a VD code and "EV1" an EV code.
_FAMILIES = "|".join(sorted(CODE_WORD, key=len, reverse=True))
CODE_RE = re.compile(rf"(?<![A-Za-z0-9_])({_FAMILIES})(\d{{1,2}}[a-z]?)(?![A-Za-z0-9_])")


def internal_codes(text: str) -> list[str]:
    """Internal plan codes in ``text`` (``["D6", "X2"]``), in order, unique."""
    out: list[str] = []
    for m in CODE_RE.finditer(text or ""):
        if m.group(0) not in out:
            out.append(m.group(0))
    return out


def reader_text(text: str) -> str:
    """``text`` with internal plan codes replaced by reader wording.

    "deviation D6" -> "post hoc"; "exploratory X2" -> "exploratory"; a code whose word the
    text already contains is dropped ("X2 envelopes, X3 detector (exploratory)" ->
    "envelopes, detector (exploratory)"); otherwise the code becomes its word ("C1" ->
    "pre-specified"); errata and plan-item codes are dropped. Punctuation left empty by a
    dropped code is tidied.
    """
    if not text or not CODE_RE.search(text):
        return text
    s = re.sub(r"\bdeviations?\s+(?:VD|D)\d{1,2}[a-z]?\b", "post hoc", text)
    s = re.sub(r"\bexploratory\s+X\d{1,2}[a-z]?\b", "exploratory", s)
    s = re.sub(r"\b(?:pre-specified|confirmatory)\s+[CS]\d{1,2}[a-z]?\b", "pre-specified", s)

    def sub(m: re.Match) -> str:
        word = CODE_WORD[m.group(1)]
        if word is None or word in s.lower():
            return ""
        return word

    s = CODE_RE.sub(sub, s)
    # Tidy what dropped codes leave behind: "( )", "(, x)", " ,", "a/ ", doubled spaces.
    for a, b in ((r"\(\s*[,;/]?\s*\)", ""), (r"\(\s*[,;/]\s*", "("), (r"\s*[,;/]\s*\)", ")"),
                 (r"(?<=\w)/(?=[\s,;)]|$)", ""), (r"(^|\s)/(?=\w)", r"\1"),
                 (r"\s+([,;.)])", r"\1"), (r"([(])\s+", r"\1"), (r",\s*,", ","), (r"\s{2,}", " "),
                 (r"^\s*[,;:]\s*", ""), (r"(pre-specified|post hoc|exploratory)(\W+\1)+", r"\1")):
        s = re.sub(a, b, s)
    return s.strip()


def reader_spec(spec: Any) -> Any:
    """A copy of a FigureSpec / TableSpec whose reader-facing text (caption, notes, titles,
    labels, legend entries, in-panel text, table cells and headers) uses reader wording
    (:func:`reader_text`). The original (possibly cached by the viewer) is not changed."""
    from biomech.viz.spec import FigureSpec

    from biomech.viz.caption import expand_abbreviations, plain_prose

    out = copy.deepcopy(spec)
    R = reader_text
    if isinstance(out, FigureSpec):
        out.caption = plain_prose(expand_abbreviations(R(out.caption)))
        for p in out.panels:
            p.title, p.note = R(p.title), R(p.note)
            for a in (p.x, p.y, p.z):
                a.label = R(a.label)
                if a.ticklabels:
                    a.ticklabels = [R(t) for t in a.ticklabels]
            if p.colorbar is not None:
                p.colorbar.label = R(p.colorbar.label)
            for s in p.series:
                s.label = R(s.label) if s.label else s.label
                if s.text:
                    s.text = [R(t) for t in s.text]
    else:
        out.caption = plain_prose(expand_abbreviations(R(out.caption)))
        out.notes = plain_prose(R(out.notes))
        for c in out.columns:
            c.header = R(c.header)
            c.group = R(c.group) if c.group else c.group
        out.rows = [{k: (R(v) if isinstance(v, str) else v) for k, v in r.items()}
                    for r in out.rows]
    return out
