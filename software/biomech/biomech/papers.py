# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Paper sets: which items a paper shows, for which output target.

A paper set is a YAML file next to the study file (``paper_*.yaml``)::

    snapshot: r7                       # the frozen store every number comes from
    variant: reporting_population               # optional: the reporting population
    numbers: numbers_dataset.yaml      # in-text numbers (optional)
    targets:                           # one or more outputs (venue profile + folder)
      journal: {venue: springer, out_dir: ../../data/analysis/icorr2027/paper_dataset/journal}
      conference: {venue: ieee, out_dir: ../../data/analysis/icorr2027/paper_dataset/conference}
    default_target: journal
    items:
      - {id: icorr_hinge, label: hinge, part: core, in: [journal, conference]}
      - {id: val_kin_waveforms, pid: s10, phase: open_close, label: waveformsPOne}

``label`` names the item for LaTeX (``\\figcaption{hinge}``; letters only), ``part`` is
``core``, ``supporting`` or ``supplement``, ``in`` lists the targets (default: all);
``target_params`` overrides view parameters for one target (the conference
requirements table shows fewer rows than the journal one)::

      - {id: table_requirements, label: tabRequirements, in: [journal, conference],
         target_params: {conference: {scope: conference}}}

Every other key is a view parameter. An entry ``- {section: "3 Not a hinge"}`` starts a
section of the storyboard: the items after it belong to it (viewer paper board, manifest).

:func:`set_item_param` writes one view parameter of one item back into the file (the
viewer's gallery: "use this depiction"), line by line so the comments stay, and re-reads
the file to verify that exactly that parameter changed. The single-target form of earlier rounds (``venue``,
``out_dir`` at the top) is still read, as the target ``default``.

A paper set with ``status: legacy`` is kept as history and left out of the study's boards.

``supplementary_videos`` (J2.7) lists the videos the paper ships, as tracked items::

    supplementary_videos:
      - {layout: workspace3d, label: videoWorkspace}
      - {layout: onset_rules, pid: s12, speed: 0.25, theme: light, label: videoOnsetRules}
      - {layout: v3_release, pid: s05, phase: [open_close, chewing], label: videoRelease}

``layout`` is an animation layout (``biomech animate --layout``); ``pid`` (not for the cohort
layouts), ``phase`` (one or a list), ``speed`` and ``theme`` select the render (defaults: the
layout's); ``label`` names it (letters only, unique in the set). :func:`find_video` matches an
entry with the rendered files by the stem ``biomech animate`` gives them (any size; the
manifest's layout, theme and speed must agree), and ``biomech check-outputs`` reports an entry
that is not rendered or was rendered from another store than the set pins (manifest
``snapshot`` and ``variant``); the gallery lists them first.

``variant`` names the sensitivity variant whose store the paper reports (the
dataset paper reads ``reporting_population``, n = 13; the primary store, n = 14, stays the analysis of
record). ``snapshot`` is then a snapshot *of that store* (``biomech snapshot --variant
reporting_population NAME``). :meth:`PaperSet.store_config` resolves both through
:func:`biomech.sensitivity.store_config`, the one place; ``export``, ``numbers``,
``verify-export`` (from the manifest's provenance), ``check-outputs`` (:func:`paper_context`)
and the gallery read the paper set's store through it.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ITEM_KEYS = ("id", "label", "part", "in", "target_params", "width")
WIDTHS = ("column", "text")
VIDEO_KEYS = ("layout", "pid", "phase", "speed", "theme", "label")
PARTS = ("core", "supporting", "supplement")
LABEL_RE = re.compile(r"^[A-Za-z]+$")


@dataclass(frozen=True)
class Target:
    name: str
    venue: str
    out_dir: Path


@dataclass(frozen=True)
class PaperItem:
    view_id: str
    params: dict[str, Any]
    label: str
    part: str
    targets: tuple[str, ...] | None          # None: every target
    section: str = ""                        # storyboard section (``{section: ...}`` entry)
    index: int = 0                           # position among the file's items
    target_params: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-target overrides
    width: str | None = None                 # "column" | "text": overrides the view's width

    def for_target(self, target: str) -> "PaperItem":
        """The item as exported for ``target``: its view parameters with that target's
        overrides (``target_params``) applied."""
        over = self.target_params.get(target)
        if not over:
            return self
        import dataclasses

        return dataclasses.replace(self, params={**self.params, **over})

    def entry(self) -> dict[str, Any]:
        """The export entry of :func:`biomech.viz.export.export`: ``{id, **params}`` plus
        ``_label`` / ``_part`` / ``_section`` (item metadata for the manifest and
        ``captions.tex``)."""
        e = {"id": self.view_id, **self.params, "_label": self.label, "_part": self.part,
             "_section": self.section}
        if self.width:
            e["_width"] = self.width
        return e


@dataclass(frozen=True)
class SupplementaryVideo:
    """One entry of ``supplementary_videos``: which render of which layout the paper ships."""

    layout: str
    label: str
    pid: str | None = None
    phases: tuple[str, ...] = ()
    speed: float | None = None
    theme: str | None = None
    index: int = 0

    def describe(self) -> str:
        """Reader name without the study id (``layout`` and the settings given)."""
        bits = [self.layout] + list(self.phases)
        bits += [f"{self.speed:g}x"] if self.speed is not None else []
        bits += [self.theme] if self.theme else []
        return f"{self.label} ({', '.join(bits)})"


@dataclass(frozen=True)
class InteractiveItem:
    """One entry of ``supplementary_interactive`` (2026-11-05): a view written as a
    self-contained interactive HTML file (rotatable 3-D) for the additional files."""
    view_id: str
    params: dict[str, Any]
    label: str


@dataclass
class PaperSet:
    path: Path
    name: str
    snapshot: str | None
    numbers: Path | None
    targets: dict[str, Target]
    default_target: str
    items: list[PaperItem] = field(default_factory=list)
    status: str = "active"
    variant: str | None = None         # reporting population (sensitivity variant; None: primary)
    videos: list[SupplementaryVideo] = field(default_factory=list)  # supplementary_videos
    interactive: list[InteractiveItem] = field(default_factory=list)  # supplementary_interactive
    data: list[InteractiveItem] = field(default_factory=list)  # supplementary_data (CSV)
    captions: dict[str, Path] = field(default_factory=dict)    # {target: caption file}

    def caption_overrides(self, target: str | None = None) -> dict[str, dict[str, Any]]:
        """The written captions of ``target`` (``captions: {target: file}``, 2026-11-10, the
        JNER compression round): ``{label: {"text": LaTeX, "panels": n | None}}`` plus the
        word limits under ``"_limits"`` (``{part: {figure: n, table: n}}``). The text replaces
        the view's generated caption in ``captions.tex`` verbatim (LaTeX, may use the set's
        number macros); ``panels`` pins the figure's panel count, so a figure that changes
        its panels fails the export until its caption is rewritten."""
        name = self.target(target).name
        f = self.captions.get(name)
        if f is None:
            return {}
        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        labels = {i.label for i in self.items_for(name)}
        out: dict[str, dict[str, Any]] = {"_limits": dict(doc.pop("_limits", {}) or {})}
        for lab, e in doc.items():
            if lab not in labels:
                raise ValueError(f"{f.name}: {lab!r} is not an item of target {name!r}")
            e = {"text": e} if isinstance(e, str) else dict(e)
            if not str(e.get("text", "")).strip():
                raise ValueError(f"{f.name}: {lab}: empty text")
            out[str(lab)] = {"text": " ".join(str(e["text"]).split()),
                             "panels": int(e["panels"]) if e.get("panels") else None}
        return out

    def store_config(self, cfg: Any, snapshot: str | None = None,
                     variant: str | None = None) -> Any:
        """The configuration this paper set reads: the store of its ``variant`` at its pinned
        ``snapshot`` (:func:`biomech.sensitivity.store_config`). ``snapshot`` / ``variant``
        override the set's (``live``: the live store; ``primary``: the primary analysis). An
        unpinned set of the population ``cfg`` reads uses ``cfg`` itself (whatever store the
        caller reads). A missing snapshot names the paper set and how to take one."""
        from biomech.sensitivity import PRIMARY
        from biomech.sensitivity import store_config as resolve

        var = self.variant if variant is None else variant
        var = None if var in (None, "", PRIMARY) else var
        snap = snapshot or self.snapshot
        if not snap and var == getattr(cfg, "variant", None):
            return cfg
        try:
            return resolve(cfg, var, snap)
        except FileNotFoundError as exc:
            where, hint = ("the primary store", "") if var is None else (
                f"the {var} store", f"; freeze one with `biomech snapshot --variant {var} "
                "NAME` and pin it here")
            raise FileNotFoundError(f"{self.path.name}: snapshot {snap!r} of {where}: "
                                    f"{exc}{hint}") from exc

    def target(self, name: str | None = None) -> Target:
        name = name or self.default_target
        if name not in self.targets:
            raise ValueError(f"{self.path.name}: no target {name!r} (known: "
                             f"{', '.join(self.targets)})")
        return self.targets[name]

    def items_for(self, target: str | None = None) -> list[PaperItem]:
        t = self.target(target).name
        return [i.for_target(t) for i in self.items if i.targets is None or t in i.targets]

    def variants(self) -> list[PaperItem]:
        """Every distinct (item, parameters) over the targets: each item once, plus once per
        target whose ``target_params`` change its parameters (checks build every one)."""
        out: list[PaperItem] = []
        for i in self.items:
            out.append(i)
            seen = [i.params]
            for t in i.targets or tuple(self.targets):
                v = i.for_target(t)
                if v.params not in seen:
                    seen.append(v.params)
                    out.append(v)
        return out

    def sections(self, target: str | None = None) -> list[str]:
        """Section titles in storyboard order (those with items for ``target``)."""
        out: list[str] = []
        for i in self.items_for(target):
            if i.section not in out:
                out.append(i.section)
        return out


def _label(view_id: str, params: dict[str, Any]) -> str:
    """Default item label: the view id and its parameter values (never the participant: a
    study id must not reach LaTeX) as CamelCase letters, digits spelled out
    (``val_kin_waveforms`` + ``phase: open_close`` -> ``valKinWaveformsOpenClose``)."""
    digits = "Zero One Two Three Four Five Six Seven Eight Nine".split()
    vals = [str(v) for k, v in params.items() if k != "pid"]
    words = re.split(r"[^A-Za-z0-9]+", " ".join([view_id, *vals]))
    out = []
    for w in words:
        w = re.sub(r"\d", lambda m: digits[int(m.group(0))], w)
        if w:
            out.append(w[:1].upper() + w[1:])
    s = "".join(out)
    return s[:1].lower() + s[1:]


def load(path: str | Path) -> PaperSet:
    """Read a paper set; relative output folders resolve against the file."""
    path = Path(path).resolve()
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def folder(p: Any) -> Path:
        q = Path(str(p))
        return q if q.is_absolute() else (path.parent / q).resolve()

    if doc.get("targets"):
        targets = {str(k): Target(str(k), str(v.get("venue", "ieee")),
                                  folder(v.get("out_dir", f"paper_{k}")))
                   for k, v in doc["targets"].items()}
        default = str(doc.get("default_target") or next(iter(targets)))
    else:
        targets = {"default": Target("default", str(doc.get("venue") or "ieee"),
                                     folder(doc.get("out_dir") or "paper"))}
        default = "default"
    items: list[PaperItem] = []
    section = ""
    for e in doc.get("items") or []:
        d = {"id": e} if isinstance(e, str) else dict(e)
        if "section" in d:
            if set(d) != {"section"}:
                raise ValueError(f"{path.name}: a section entry holds only 'section': {d}")
            section = str(d["section"])
            continue
        vid = str(d["id"])
        params = {k: v for k, v in d.items() if k not in ITEM_KEYS}
        part = str(d.get("part", "core"))
        if part not in PARTS:
            raise ValueError(f"{path.name}: item {vid}: part {part!r} not in {PARTS}")
        label = str(d.get("label") or "")
        if not label:                     # default: from the view; repeats get B, C, ...
            base = _label(vid, params)
            n = sum(1 for i in items if i.label == base or i.label.rstrip("BCDEFGHIJ") == base)
            label = base + ("" if n == 0 else "BCDEFGHIJ"[n - 1])
        if not LABEL_RE.match(label):
            raise ValueError(f"{path.name}: item {vid}: label {label!r} must be letters only "
                             "(it becomes a LaTeX name)")
        tg = d.get("in")
        tg = tuple(str(t) for t in tg) if tg else None
        for t in tg or ():
            if t not in targets:
                raise ValueError(f"{path.name}: item {vid}: unknown target {t!r}")
        tp = d.get("target_params") or {}
        if not isinstance(tp, dict) or not all(isinstance(v, dict) for v in tp.values()):
            raise ValueError(f"{path.name}: item {vid}: target_params must map a target to "
                             "{parameter: value}")
        for t, over in tp.items():
            if str(t) not in targets:
                raise ValueError(f"{path.name}: item {vid}: target_params: unknown target {t!r}")
            if tg and str(t) not in tg:
                raise ValueError(f"{path.name}: item {vid}: target_params for {t!r}, which "
                                 "the item is not in")
            bad = [k for k in over if k in ITEM_KEYS or k == "section"]
            if bad:
                raise ValueError(f"{path.name}: item {vid}: target_params may only set view "
                                 f"parameters, not {bad}")
        width = d.get("width")
        if width is not None and width not in WIDTHS:
            raise ValueError(f"{path.name}: item {vid}: width {width!r} not in {WIDTHS}")
        items.append(PaperItem(vid, params, label, part, tg, section, len(items),
                               {str(t): dict(v) for t, v in tp.items()}, width))
    labels = [i.label for i in items]
    dup = sorted({x for x in labels if labels.count(x) > 1})
    if dup:
        raise ValueError(f"{path.name}: duplicate item labels {dup}")
    videos = _videos(path, doc.get("supplementary_videos"), set(labels))
    interactive = _interactive(path, doc.get("supplementary_interactive"),
                               set(labels) | {v.label for v in videos})
    # supplementary_data (2026-11-10, JNER revision 5.7): a table view written as a CSV file
    # for the additional files (e.g. the workspace envelope coordinates for designers)
    data = _interactive(path, doc.get("supplementary_data"),
                        set(labels) | {v.label for v in videos} | {i.label for i in interactive},
                        "supplementary_data")
    numbers = doc.get("numbers")
    caps = doc.get("captions") or {}
    if not isinstance(caps, dict) or any(str(k) not in targets for k in caps):
        raise ValueError(f"{path.name}: captions must map a target to a caption file")
    variant = doc.get("variant")
    if variant is not None and (not isinstance(variant, str) or not variant.strip()):
        raise ValueError(f"{path.name}: variant must name a sensitivity variant, not {variant!r}")
    return PaperSet(path, path.stem, doc.get("snapshot"),
                    (path.parent / numbers).resolve() if numbers else None,
                    targets, default, items, str(doc.get("status", "active")),
                    str(variant).strip() if variant else None, videos, interactive, data,
                    {str(k): (path.parent / str(v)).resolve() for k, v in caps.items()})


def _interactive(path: Path, raw: Any, taken: set[str],
                 key: str = "supplementary_interactive") -> list[InteractiveItem]:
    """The ``supplementary_interactive`` (or ``supplementary_data``) entries ({id, label, view
    parameters}), validated."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: {key} must be a list of {{id, label, ...}}")
    out: list[InteractiveItem] = []
    for k, e in enumerate(raw):
        if not isinstance(e, dict) or "id" not in e or "label" not in e:
            raise ValueError(f"{path.name}: {key}[{k}] needs id and label")
        label = str(e["label"])
        if not LABEL_RE.match(label) or label in taken:
            raise ValueError(f"{path.name}: {key}[{k}]: label {label!r} "
                             "must be letters only and unique in the set")
        taken.add(label)
        out.append(InteractiveItem(str(e["id"]), {kk: v for kk, v in e.items()
                                                  if kk not in ("id", "label")}, label))
    return out


def _videos(path: Path, raw: Any, item_labels: set[str]) -> list[SupplementaryVideo]:
    """The ``supplementary_videos`` entries, validated (keys, label, types)."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: supplementary_videos must be a list of "
                         "{layout, label, ...} entries")
    out: list[SupplementaryVideo] = []
    for k, e in enumerate(raw):
        if not isinstance(e, dict) or "layout" not in e:
            raise ValueError(f"{path.name}: supplementary_videos[{k}] needs a layout: {e!r}")
        bad = sorted(set(e) - set(VIDEO_KEYS))
        if bad:
            raise ValueError(f"{path.name}: supplementary_videos[{k}]: unknown keys {bad} "
                             f"(allowed: {', '.join(VIDEO_KEYS)})")
        label = str(e.get("label") or "")
        if not LABEL_RE.match(label):
            raise ValueError(f"{path.name}: supplementary_videos[{k}]: label {label!r} must "
                             "be letters only (it becomes a LaTeX name)")
        if label in item_labels or any(v.label == label for v in out):
            raise ValueError(f"{path.name}: supplementary_videos[{k}]: duplicate label "
                             f"{label!r}")
        ph = e.get("phase")
        phases = tuple(str(x) for x in ph) if isinstance(ph, (list, tuple)) else \
            ((str(ph),) if ph else ())
        theme = e.get("theme")
        if theme is not None and theme not in ("dark", "light"):
            raise ValueError(f"{path.name}: supplementary_videos[{k}]: theme {theme!r} "
                             "(dark or light)")
        try:
            speed = float(e["speed"]) if e.get("speed") is not None else None
        except (TypeError, ValueError):
            raise ValueError(f"{path.name}: supplementary_videos[{k}]: speed must be a "
                             f"number, not {e['speed']!r}") from None
        out.append(SupplementaryVideo(str(e["layout"]), label,
                                      str(e["pid"]) if e.get("pid") else None, phases, speed,
                                      str(theme) if theme else None, k))
    return out


def video_dirs(*configs: Any) -> list[Path]:
    """The ``videos`` folders of the given stores (each once, in order): where ``biomech
    animate`` writes by default (``<out_dir>/videos``)."""
    out: list[Path] = []
    for c in configs:
        d = getattr(c, "out_dir", None)
        if d is not None and Path(d) / "videos" not in out:
            out.append(Path(d) / "videos")
    return out


def video_stem_for(v: SupplementaryVideo, pseudonym: Any = None) -> str:
    """The file stem ``biomech animate`` gives the entry at the default size (1080p)."""
    from biomech.anim.layouts import COHORT_LAYOUTS, default_phases, default_speed, video_stem

    speed = v.speed if v.speed is not None else default_speed(v.layout)
    if v.layout in COHORT_LAYOUTS:
        return video_stem(v.layout, None, list(v.phases) or ["open_close"], speed=speed,
                          theme=v.theme)
    if not v.pid:
        raise ValueError(f"supplementary video {v.label}: layout {v.layout} needs a pid")
    n = pseudonym(v.pid) if pseudonym is not None else None
    if n is None:
        raise ValueError(f"supplementary video {v.label}: no pseudonym for its participant")
    return video_stem(v.layout, v.pid, list(v.phases) or default_phases(v.layout),
                      pseudonym=n, speed=speed, theme=v.theme)


def find_video(v: SupplementaryVideo, dirs: list[Path], pseudonym: Any = None
               ) -> tuple[Path | None, dict[str, Any] | None]:
    """The rendered file of entry ``v`` and its manifest: ``<stem>.mp4`` at the default size,
    else the same stem with only a size suffix (``_720p``, ``_4k``, ``_print``, a width or WxH);
    the manifest's layout, theme and speed must agree. (None, None) when not rendered."""
    import json

    stem = video_stem_for(v, pseudonym)
    size_rx = re.compile(re.escape(stem) + r"(_(720p|4k|print|\d+|\d+x\d+))?")
    speed = v.speed
    for d in dirs:
        cands = sorted(Path(d).glob(f"{stem}*.mp4"), key=lambda q: (q.stem != stem, q.name))
        for mp4 in cands:
            if not size_rx.fullmatch(mp4.stem):
                continue
            js = mp4.with_suffix(".json")
            try:
                meta = json.loads(js.read_text(encoding="utf-8")) if js.is_file() else {}
            except ValueError:
                meta = {}
            if meta.get("layout", v.layout) != v.layout:
                continue
            if v.theme and meta.get("theme", v.theme) != v.theme:
                continue
            if speed is not None and abs(float(meta.get("speed", speed)) - speed) > 1e-9:
                continue
            return mp4, meta
    return None, None


def video_store_problems(meta: dict[str, Any] | None, pset: "PaperSet") -> list[str]:
    """Why a rendered video does not come from the store the paper set pins (empty: it
    does): the manifest's ``snapshot`` (``live`` when unpinned) and, when recorded,
    ``variant`` (``primary`` for the primary analysis)."""
    if meta is None:
        return ["no manifest next to the video"]
    out = []
    want = pset.snapshot or "live"
    got = meta.get("snapshot")
    if got is None:
        out.append(f"the manifest records no snapshot (paper set pins {want})")
    elif str(got) != str(want):
        out.append(f"rendered from snapshot {got}, the paper set pins {want}")
    if "variant" in meta:
        wv = pset.variant or "primary"
        if str(meta["variant"]) != wv:
            out.append(f"rendered from the {meta['variant']} store, the paper set reads {wv}")
    return out


@contextlib.contextmanager
def paper_context(ctx: Any, pset: PaperSet, snapshot: str | None = None) -> Any:
    """A :class:`~biomech.io.context.DataContext` on the store ``pset`` reads
    (:meth:`PaperSet.store_config`): ``ctx`` itself when it reads that store already, else a
    new context (with that store's own cohort manifest), closed on exit."""
    cfg = pset.store_config(ctx.config, snapshot)
    if cfg is ctx.config:
        yield ctx
        return
    from biomech.io.context import DataContext

    c = DataContext(cfg)
    try:
        yield c
    finally:
        c.close()


def study_paper_sets(study_yaml: str | Path | None) -> list[PaperSet]:
    """The active paper sets of a study: every ``paper_*.yaml`` next to its study file,
    without the ones marked ``status: legacy``."""
    if study_yaml is None:
        return []
    here = Path(study_yaml).resolve().parent
    out = []
    for p in sorted(here.glob("paper_*.yaml")):
        ps = load(p)
        if ps.status != "legacy":
            out.append(ps)
    return out


_ITEM_LINE = re.compile(r"^(\s*-\s+)(.*?)(\s*(#.*)?)$")


def _item_lines(lines: list[str]) -> list[int]:
    """Line numbers of the item entries (one line each) under ``items:``, sections left out."""
    out, inside = [], False
    for n, line in enumerate(lines):
        if re.match(r"^items:\s*(#.*)?$", line):
            inside = True
            continue
        if inside and re.match(r"^\S", line) and not line.startswith("#"):
            break                                  # the next top-level key
        if inside and re.match(r"^\s*-\s", line) and "section:" not in line.split("#")[0]:
            out.append(n)
    return out


def set_item_param(path: str | Path, index: int, key: str, value: Any) -> PaperSet:
    """Set view parameter ``key`` of item ``index`` (``PaperItem.index``) to ``value`` in
    the file and return the re-read set. A bare entry (``- view_id``) becomes a flow map.
    An item without an explicit ``label`` gets its current one written out (default labels
    derive from the parameters). Fails, leaving the file unchanged, unless the re-read set
    differs from the old one in exactly that parameter."""
    if key in ITEM_KEYS or key == "section" or not re.fullmatch(r"[a-z_]+", key):
        raise ValueError(f"not a view parameter: {key!r}")
    path = Path(path)
    old = load(path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    rows = _item_lines(lines)
    if not 0 <= index < len(rows) or len(rows) != len(old.items):
        raise ValueError(f"{path.name}: item {index} not found as a single-line entry")
    n = rows[index]
    eol = "\n" if lines[n].endswith("\n") else ""
    m = _ITEM_LINE.match(lines[n].rstrip("\n"))
    lead, body, tail = m.group(1), m.group(2).strip(), m.group(3) or ""
    val = yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("...").strip()
    if not body.startswith("{"):
        body = "{id: " + body + "}"
    inner = body[1:-1]
    if re.search(rf"(^|,)\s*{key}\s*:", inner):
        inner = re.sub(rf"((^|,)\s*{key}\s*:\s*)([^,]*)", rf"\g<1>{val}", inner, count=1)
    else:
        inner = re.sub(r"^(\s*id\s*:\s*[^,]*)", rf"\g<1>, {key}: {val}", inner, count=1)
    if not re.search(r"(^|,)\s*label\s*:", inner):
        # A default label derives from the parameters: pin the current one, so the LaTeX
        # name (\figcaption{...}, \input paths) survives the change of depiction.
        inner = re.sub(r"^(\s*id\s*:\s*[^,]*)", rf"\g<1>, label: {old.items[index].label}",
                       inner, count=1)
    lines[n] = f"{lead}{{{inner}}}{tail}{eol}"
    path.write_text("".join(lines), encoding="utf-8")
    try:
        new = load(path)
        a, b = old.items[index], new.items[index]
        same_rest = [i.entry() for j, i in enumerate(old.items) if j != index] == \
            [i.entry() for j, i in enumerate(new.items) if j != index]
        want = {**a.params, key: value}
        if not (same_rest and b.params == want and b.view_id == a.view_id and
                b.label == a.label and b.part == a.part and b.targets == a.targets and
                b.target_params == a.target_params):
            raise ValueError(f"{path.name}: the edit changed more than {key} of item {index}")
    except Exception:
        path.write_text(text, encoding="utf-8")
        raise
    return new
