# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The 3D-printed parts of the setup: which part, which version, used in which study.

One manifest, ``hardware/mechanical/parts.yaml``, lists every part (teeth attachment, markers,
calibration tool, ...), its versions (``variants``: an STL, optionally a FreeCAD body), the
design analyses, and ``usage``: the variant each study (or participant) used. ``null`` usage means
"not yet specified"; every consumer then falls back to what it showed before the manifest existed
(the original JTS Blender renders), so filling ``usage`` later changes figures, not code.

Pure (PyYAML + pathlib): read by the paper views, the gallery and by the tools in
``hardware/mechanical/tools`` (which put ``software/biomech`` on ``sys.path``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

MECH_REL = Path("hardware") / "mechanical"
STATUSES = ("used", "variant", "printed", "superseded", "concept")


def mech_dir() -> Path:
    """``hardware/mechanical`` of this checkout (this file is software/biomech/biomech/)."""
    return Path(__file__).resolve().parents[3] / MECH_REL


@dataclass(frozen=True)
class Variant:
    """One version of a part. ``stl`` is the authoritative geometry; ``fcstd_body`` names its
    FreeCAD body when the CAD exists (``cad_pending``: STL only, model to be added)."""

    part: str
    id: str
    title: str
    status: str
    stl: Path | None = None
    fcstd_body: str | None = None
    params: dict[str, float] = field(default_factory=dict)
    print: Path | None = None
    cad_pending: bool = False
    date: str = ""
    note: str = ""
    # own FreeCAD project when the body lives in another model than the part's ``fcstd`` (the
    # Qualisys + ArUco markers: the reflective-marker model of 2026-11-04)
    fcstd: Path | None = None


@dataclass(frozen=True)
class Part:
    id: str
    title: str
    role: str                       # what the part is for (one line)
    material: dict[str, Any]
    fcstd: Path | None
    legacy_render: Path | None      # what the setup figures showed before (original JTS render)
    variants: dict[str, Variant]
    inspired_by: tuple[str, ...] = ()
    studies: tuple[str, ...] = ()   # studies the part takes part in (keys of `usage`)
    fcstd_extra: tuple[str, ...] = ()   # reference objects kept in the part FCStd (teeth scan)
    candidate: str | None = None        # variant shown (marked candidate) while usage is null

    @property
    def bodies(self) -> list[str]:
        """The FreeCAD bodies of the CAD-backed variants in the part's own ``fcstd``, in variant
        order, once each (variants with their own ``fcstd`` are not in it)."""
        return list(dict.fromkeys(v.fcstd_body for v in self.variants.values()
                                  if v.fcstd_body and v.fcstd is None))

    def source(self, v: "Variant") -> Path | None:
        """The FreeCAD project a variant is exported from."""
        return v.fcstd or self.fcstd


@dataclass(frozen=True)
class Assembly:
    """A setup body as the figures show it (head piece, mouth piece, calibration tool)."""

    id: str
    title: str
    parts: tuple[str, ...]
    legacy_render: Path | None


class Manifest:
    def __init__(self, doc: dict[str, Any], root: Path):
        self.root = root
        self.doc = doc
        self.parts: dict[str, Part] = {}
        for pid, p in (doc.get("parts") or {}).items():
            fc = root / p["fcstd"] if p.get("fcstd") else None
            vs = {}
            for vid, v in (p.get("variants") or {}).items():
                vs[vid] = Variant(
                    part=pid, id=vid, title=str(v.get("title", vid)),
                    status=str(v.get("status", "variant")),
                    stl=root / v["stl"] if v.get("stl") else None,
                    fcstd_body=v.get("fcstd_body"), params=dict(v.get("params") or {}),
                    print=root / v["print"] if v.get("print") else None,
                    cad_pending=bool(v.get("cad_pending", False)),
                    date=str(v.get("date", "")), note=str(v.get("note", "")).strip(),
                    fcstd=root / v["fcstd"] if v.get("fcstd") else None)
            self.parts[pid] = Part(
                id=pid, title=str(p.get("title", pid)), role=str(p.get("role", "")).strip(),
                material=dict(p.get("material") or {}), fcstd=fc,
                legacy_render=root / p["legacy_render"] if p.get("legacy_render") else None,
                variants=vs, inspired_by=tuple(p.get("inspired_by") or ()),
                studies=tuple(s for s, u in (doc.get("usage") or {}).items()
                              if s != "overrides" and pid in (u or {})),
                fcstd_extra=tuple(p.get("fcstd_extra") or ()), candidate=p.get("candidate"))
        self.assemblies: dict[str, Assembly] = {
            aid: Assembly(aid, str(a.get("title", aid)), tuple(a.get("parts") or ()),
                          root / a["legacy_render"] if a.get("legacy_render") else None)
            for aid, a in (doc.get("assemblies") or {}).items()}
        self.analyses: dict[str, dict[str, Any]] = dict(doc.get("analyses") or {})
        self.usage: dict[str, dict[str, str | None]] = {
            s: dict(u or {}) for s, u in (doc.get("usage") or {}).items() if s != "overrides"}
        self.overrides: dict[str, dict[str, dict[str, str]]] = dict(
            (doc.get("usage") or {}).get("overrides") or {})

    # -- usage -----------------------------------------------------------------------------
    def used(self, study: str, part: str, pid: str | None = None) -> Variant | None:
        """The variant of ``part`` that ``study`` (participant ``pid``) used; None = not yet
        specified (or the part is not part of that study)."""
        vid = None
        if pid is not None:
            vid = ((self.overrides.get(study) or {}).get(pid) or {}).get(part)
        if vid is None:
            vid = (self.usage.get(study) or {}).get(part)
        return self.parts[part].variants[vid] if vid else None

    def shown(self, study: str, part: str) -> tuple[Variant | None, bool]:
        """(variant, specified): the used variant, else the part's ``candidate``."""
        v = self.used(study, part)
        if v is not None:
            return v, True
        p = self.parts[part]
        return (p.variants.get(p.candidate) if p.candidate else None), False

    def specified(self, study: str) -> bool:
        """Every part of ``study`` has its variant stated."""
        u = self.usage.get(study) or {}
        return bool(u) and all(v for v in u.values())

    def study_parts(self, study: str) -> list[tuple[Part, Variant, bool]]:
        """(part, shown variant, specified) for the parts of ``study``; a study without a
        ``usage`` entry (e.g. a synthetic one) gets every part that has a candidate."""
        ids = list(self.usage.get(study) or [p.id for p in self.parts.values() if p.candidate])
        out = []
        for pid in ids:
            v, specified = self.shown(study, pid)
            if v is not None:
                out.append((self.parts[pid], v, specified))
        return out

    # -- renders ---------------------------------------------------------------------------
    def render(self, part: str, variant: str, view: str = "iso") -> Path:
        return self.root / "renders" / f"{part}__{variant}__{view}.png"

    def assembly_render(self, study: str | None, assembly: str) -> Path | None:
        """The image of a setup body for ``study``: the render of the used variant of its
        first part with a render (head marker, mouth marker, tool); while usage is not
        specified, the original render (``legacy_render``)."""
        a = self.assemblies[assembly]
        if study is not None:
            for p in a.parts:
                v = self.used(study, p)
                if v is not None and self.render(p, v.id).is_file():
                    return self.render(p, v.id)
        return a.legacy_render if a.legacy_render and a.legacy_render.is_file() else None

    def images(self) -> list[tuple[str, Path]]:
        """Every render (caption, path) for the gallery: parts and variants, then concepts."""
        out = []
        for p in self.parts.values():
            for v in p.variants.values():
                r = self.render(p.id, v.id)
                if r.is_file():
                    out.append((f"{p.title}, {v.title} ({v.status}; no participant)", r))
        for aid, a in self.analyses.items():
            for cap, rel in (a.get("figures") or {}).items():
                if (self.root / rel).is_file():
                    out.append((f"{a.get('title', aid)}: {cap}", self.root / rel))
        return out

    # -- checks ----------------------------------------------------------------------------
    def problems(self, studies: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
        """(errors, notes). Errors: a missing file, an unknown status, usage naming an unknown
        part/variant/study. Notes: CAD pending, usage not yet specified."""
        err, note = [], []
        for p in self.parts.values():
            if p.fcstd is not None and not p.fcstd.is_file():
                err.append(f"{p.id}: fcstd {p.fcstd} missing")
            for v in p.variants.values():
                if v.status not in STATUSES:
                    err.append(f"{p.id}.{v.id}: status {v.status!r} not in {STATUSES}")
                for what, f in (("stl", v.stl), ("print", v.print)):
                    if f is not None and not f.is_file():
                        err.append(f"{p.id}.{v.id}: {what} {f} missing")
                if v.stl is None and v.fcstd_body is None:
                    err.append(f"{p.id}.{v.id}: neither stl nor fcstd_body")
                if v.fcstd_body and p.source(v) is None:
                    err.append(f"{p.id}.{v.id}: fcstd_body without a part or variant fcstd")
                if v.fcstd is not None and not v.fcstd.is_file():
                    err.append(f"{p.id}.{v.id}: fcstd {v.fcstd} missing")
                if v.cad_pending:
                    note.append(f"{p.id}.{v.id}: CAD pending (STL only)")
            if p.candidate and p.candidate not in p.variants:
                err.append(f"{p.id}: candidate {p.candidate!r} is not a variant")
        for a in self.assemblies.values():
            for p in a.parts:
                if p not in self.parts:
                    err.append(f"assembly {a.id}: unknown part {p}")
        for s, u in self.usage.items():
            if studies and s not in studies:
                err.append(f"usage: unknown study {s!r}")
            for p, vid in u.items():
                if p not in self.parts:
                    err.append(f"usage.{s}: unknown part {p!r}")
                elif vid is None:
                    note.append(f"usage.{s}.{p}: not yet specified")
                elif vid not in self.parts[p].variants:
                    err.append(f"usage.{s}.{p}: unknown variant {vid!r}")
        for s, by_pid in self.overrides.items():
            for pid, m in (by_pid or {}).items():
                for p, vid in (m or {}).items():
                    if p not in self.parts or vid not in self.parts[p].variants:
                        err.append(f"usage.overrides.{s}.{pid}: unknown {p}={vid}")
        for aid, a in self.analyses.items():
            if not (self.root / str(a.get("path", ""))).is_dir():
                err.append(f"analysis {aid}: path {a.get('path')} missing")
        return err, note


def load(path: str | Path | None = None) -> Manifest:
    """The manifest (cached per path); ``path`` defaults to this checkout's parts.yaml."""
    p = Path(path) if path else mech_dir() / "parts.yaml"
    return _load(str(p.resolve()), p.stat().st_mtime if p.is_file() else 0.0)


@lru_cache(maxsize=4)
def _load(path: str, _mtime: float) -> Manifest:
    p = Path(path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return Manifest(doc or {}, p.parent)
