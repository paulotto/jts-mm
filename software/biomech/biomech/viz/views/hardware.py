# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The 3D-printed parts of the setup (hardware/mechanical, read through biomech.hwparts).

One builder per item, registered for both study groups (the dataset paper: ``hw_parts`` /
``hw_table``; the systems paper: ``val_hw_parts`` / ``val_hw_table``), so both papers show the
same parts and the same versions. Which version a study used comes from ``usage`` in
hardware/mechanical/parts.yaml; while it is not specified, the figures show the current candidate
version and the caption says so (check-outputs warns). The images are the committed renders and
figures of hardware/mechanical (``make renders analyses`` there), not participant data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from biomech import hwparts
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views.registry import ViewRequest, register

FIG_FORMS = ("used", "lineage", "variants", "concepts", "incisor")
TABLE_FORMS = ("parts", "crosscheck")
SOURCES = ["hardware:parts.yaml"]
UNSPECIFIED = ("which version of each part was used in this study is still to be confirmed; "
               "the current versions are shown")
# Reader words of the parts manifest's display text: no tag ids, body codes,
# CLI or doc codes, British spelling of the manifest kept out of the IEEE paper.
PART_TITLE = {"CT calibrater": "Tip calibrator"}
_PURPOSE_DROP = (r"\s*\((?:MP|HP|CT),\s*(?:ids|tags)\s*[\d/]+\)",   # "(MP, ids 13/17)"
                 r"\s*\(ids\s*[\d/]+\)",                        # "(ids 35/41)"
                 r"\s*\(jtsmm [a-z-]+\)",                        # "(jtsmm ct-calibrate)"
                 r"\bcv\d+,\s*")                                 # "(cv05, 20 mm steps)"


def _purpose(text: str) -> str:
    import re

    t = text
    for pat in _PURPOSE_DROP:
        t = re.sub(pat, "", t)
    t = re.sub(r"\s*\(cv\d+\)", "", t)
    return t.replace("digitisation", "digitization").replace("  ", " ").strip()


def _version(title: str) -> str:
    """A version title for print: no calendar date ("Version 5, 5 mm (print 2026-10-03)" ->
    "Version 5, 5 mm")."""
    import re

    return re.sub(r"\s*\(print \d{4}-\d{2}-\d{2}\)", "", str(title)).strip()


def _quantity(q: str) -> tuple[str, str]:
    """"tag spacing / mm" -> ("tag spacing", "mm"); deg -> °."""
    name, _, unit = str(q).rpartition(" / ")
    if not name:
        return str(q), ""
    return name, "°" if unit.strip() == "deg" else unit.strip()


def image_panel(path: Path | None, title: str, max_px: int = 900) -> Panel:
    """An image file as a panel: transparent margins cropped, on white, at most ``max_px``."""
    p = Panel(title, Axis(""), Axis(""), [], legend="none")
    try:
        from PIL import Image

        im = Image.open(path).convert("RGBA")
        im = im.crop(im.getbbox() or (0, 0, *im.size))
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        im.thumbnail((max_px, max_px))
        img = np.asarray(im, np.uint8)
    except Exception:
        p.series.append(Series("text", x=np.array([0.5]), y=np.array([0.5]),
                               text=["image not found"]))
        return p
    p.series.append(Series("image", img=img, extent=(0, img.shape[1], img.shape[0], 0)))
    p.x.ticks, p.x.ticklabels, p.x.lim = [], [], (0, img.shape[1])
    p.y.ticks, p.y.ticklabels, p.y.lim = [], [], (0, img.shape[0])
    p.y.reverse = True
    p.equal = True
    return p


def assembly_image(study: str | None, assembly: str) -> Path | None:
    """The image of a setup body (head piece, mouth piece, calibration tool) for ``study``:
    the render of the used version, the original JTS render while usage is not specified."""
    try:
        return hwparts.load().assembly_render(study, assembly)
    except Exception:                               # manifest missing: no image
        return None


def usage_note(study: str | None) -> list[str]:
    """Caption note for a figure that shows the worn hardware."""
    try:
        man = hwparts.load()
    except Exception:
        return []
    if study is None or study not in man.usage:
        return []
    if not man.specified(study):
        return ["Hardware renders: " + UNSPECIFIED + " (original JTS renders where no "
                "version is specified)."]
    used = [f"{man.parts[p].title.lower()} {v.title}" for p, v in
            ((p, man.used(study, p)) for p in man.usage[study]) if v is not None]
    return ["Part versions: " + "; ".join(used) + "."]


def _study(req: ViewRequest) -> str:
    return str(getattr(req.ctx.config, "study", "") or "")


def _grid(panels: list[Panel], cols: int) -> tuple[int, int]:
    cols = max(1, min(cols, len(panels)))
    return int(np.ceil(len(panels) / cols)), cols


# -- figure --------------------------------------------------------------------------------------
def parts_figure(req: ViewRequest, fid: str) -> FigureSpec:
    man = hwparts.load()
    study = _study(req)
    form = str(req.options.get("form", "used"))
    title = "3D-printed parts"
    panels: list[Panel] = []
    notes: list[str] = []
    if form == "used":
        spec_all = True
        for part, v, specified in man.study_parts(study):
            spec_all &= specified
            t = f"{PART_TITLE.get(part.title, part.title)}: {_version(v.title)}"
            panels.append(image_panel(man.render(part.id, v.id), t))
        shows = ("the 3D-printed parts of the study, rendered from the print geometry "
                 "(STL) at a common view per part")
        if not spec_all:
            notes.append(UNSPECIFIED[0].upper() + UNSPECIFIED[1:] + ".")
        cols = 4
    elif form == "lineage":
        panels = [image_panel(man.root / "figures" / "lineage_teeth_attachment.png",
                              "Teeth attachment", 2100),
                  image_panel(man.root / "figures" / "marker_versions.png", "Markers", 2100)]
        shows = ("the design history of the parts: the teeth attachment from the original JTS "
                 "part to version 5, and the head marker, mouth marker and calibration tool "
                 "versions; "
                 "renders at one scale per row (scale bars)")
        cols = 1
    elif form == "variants":
        panels = [image_panel(man.root / "figures" / "v5_variants.png",
                              "Version-5 variants", 2100)]
        shows = ("the version-5 teeth-attachment variants, coloured by the distance of each "
                 "surface point to the saved CAD state (5.5 mm); the variants differ in the "
                 "inner-arc separation only, the 2026-10-03 print is an earlier version-5 state")
        cols = 1
    elif form == "concepts":
        a = man.analyses.get("molar_clamp_concepts", {})
        for cap, rel in (a.get("figures") or {}).items():
            panels.append(image_panel(man.root / rel, cap[0].upper() + cap[1:]))
        shows = ("the molar-clamp concept study on a lower-arch scan: the lingual undercut "
                 "per tooth and five clamp concepts (none was built; the study informed "
                 "version 5)")
        cols = 3
    else:                                                       # incisor
        a = man.analyses.get("incisor_attachment", {})
        for cap, rel in (a.get("figures") or {}).items():
            panels.append(image_panel(man.root / rel, cap[0].upper() + cap[1:], 1400))
        shows = ("why flat jaws do not hold a lower incisor (wedge-shaped crown, friction "
                 "needed above tan 33° = 0.65) and why straight wings do not follow the arch "
                 "past the canine: the analyses behind version 5")
        cols = 2
    if not panels:
        panels = [Panel("no images", Axis(""), Axis(""),
                        [Series("text", x=np.array([0.5]), y=np.array([0.5]),
                                text=["run make renders analyses in hardware/mechanical"])],
                        legend="none")]
    for i, p in enumerate(panels):
        p.title = f"({chr(97 + i)}) {p.title}" if len(panels) > 1 else p.title
    rows, cols = _grid(panels, cols)
    spec = FigureSpec(fid, title, panels, rows=rows, cols=cols, width="text",
                      height_in=1.6 if cols > 1 else 2.4, sources=SOURCES,
                      meta={"study": study, "form": form})
    return attach(spec, CaptionSpec(
        shows, [("renders", "the part versions as printed (CAD or STL), not to a common "
                            "scale across panels")],
        "no participants (hardware of the setup)", "print geometry, no statistic",
        evidence="descriptive", notes=notes))


# -- tables --------------------------------------------------------------------------------------
def _index(man: hwparts.Manifest) -> dict[str, Any]:
    import json

    p = man.root / "renders" / "index.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def parts_table(req: ViewRequest, tid: str) -> TableSpec:
    man = hwparts.load()
    study = _study(req)
    form = str(req.options.get("form", "parts"))
    if form == "crosscheck":
        import json

        p = man.root / "tables" / "crosscheck.json"
        rows_x = json.loads(p.read_text())["rows"] if p.is_file() else []
        names = {"HP": "head marker", "MP": "mouth marker", "CT": "calibration tool",
                 "eval_array": "trueness array"}
        rows = []
        for r in rows_x:
            qn, unit = _quantity(r["quantity"])
            d = 1 if unit == "°" else 2                # one rule: mm to 0.01, angles to 0.1°

            def val(x: Any) -> str:
                return "–" if x is None else f"{float(x):.{d}f}"

            note = str(r.get("note", "") or "")
            if note.startswith("config: pivot tip z in the body frame"):
                note = "configuration: depth of the pivot-calibrated tip in the tool frame"
            rows.append({"marker": names.get(r["body"], r["body"]),
                         "check": f"{qn} ({unit})" if unit else qn,
                         "config_value": val(r.get("config")), "cad_value": val(r.get("cad")),
                         "remark": note.replace("vs.", "vs")})
        cols = [Column("marker", "Part", align="l"), Column("check", "Quantity", align="X"),
                Column("config_value", "Configuration", align="r"),
                Column("cad_value", "CAD", align="r"), Column("remark", "Note", align="X")]
        t = TableSpec(tid, "", cols, rows, width="text", sources=SOURCES)
        return attach(t, CaptionSpec(
            "the print geometry (CAD) against the optical configuration and the pivot "
            "calibration of the calibration-tool tip",
            [("Configuration", "tag poses of the optical configuration; the tip offset of the "
                               "pivot calibration"),
             ("CAD", "measured on the print geometry (tag seats, rod axis, IMU seat)")],
            "no participants (hardware of the setup)", "single values, no statistic",
            evidence="descriptive",
            notes=["A consistency check of the hardware; no analysis input is changed."]))
    index = _index(man)
    rows = []
    spec_all = True
    for pr, v, specified in man.study_parts(study):
        spec_all &= specified
        g = index.get(f"{pr.id}/{v.id}", {})
        rows.append({
            "part_title": PART_TITLE.get(pr.title, pr.title),
            "version": _version(v.title),                       # candidates: said in the caption
            "purpose": _purpose(pr.role.split(";")[0].rstrip(".")),
            "material_name": pr.material.get("name", "–"),
            "size": " × ".join(f"{x:.0f}" for x in g.get("size_mm", [])) or "–",
            "mass_printed": f"{g['mass_printed_g']:.1f}" if g.get("mass_printed_g") else "–",
            "mass_solid": f"{g['mass_solid_g']:.1f}" if g.get("mass_solid_g") else "–",
            "cad_source": "pending" if v.cad_pending else ("FreeCAD" if v.fcstd_body else "STL")})
    cols = [Column("part_title", "Part", align="l"), Column("version", "Version", align="X"),
            Column("purpose", "Purpose", align="X"), Column("material_name", "Material", align="l"),
            Column("size", "Size (mm)", align="l"),
            Column("mass_printed", "Printed", align="r", group="Mass (g)"),
            Column("mass_solid", "Solid", align="r", group="Mass (g)"),
            Column("cad_source", "CAD", align="l")]
    t = TableSpec(tid, "", cols, rows, width="text", sources=SOURCES)
    return attach(t, CaptionSpec(
        "the 3D-printed parts of the study: version, purpose, material, bounding box, and mass",
        # the printed mass is an estimate: the parts were not weighed (the IMU and its cable
        # are fixed to the mouth piece); about 10 % infill per the user (2026-11-04)
        [("Mass", "from the print geometry, not weighed: printed = the two wall lines solid and "
                  "the inside at about 10 % infill (15 % where the slicer record says so), an "
                  "estimate; solid = 100 % infill"),
         ("CAD", "FreeCAD model in the repository, or STL only (model pending)")],
        "no participants (hardware of the setup)", "print geometry, no statistic",
        evidence="descriptive",
        notes=[] if spec_all else [UNSPECIFIED[0].upper() + UNSPECIFIED[1:] + "."]))


for _fid, _tid, _grp in (("hw_parts", "hw_table", "dataset"),
                         ("val_hw_parts", "val_hw_table", "validation")):
    register(_fid, "paper", "3D-printed parts", choices={"form": FIG_FORMS}, form="used",
             group=_grp)(lambda req, _f=_fid: parts_figure(req, _f))
    register(_tid, "paper", "3D-printed parts: versions and masses",
             choices={"form": TABLE_FORMS}, form="parts",
             group=_grp)(lambda req, _t=_tid: parts_table(req, _t))
