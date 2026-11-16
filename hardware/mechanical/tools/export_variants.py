# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Export every CAD-backed variant of parts.yaml to build/regen/<part>/<variant>.stl (FreeCAD 1.1).

    freecad.cmd tools/export_variants.py [--pass part ...]      (make parts)

A variant with ``fcstd_body`` is exported from its part's FCStd after setting its ``params``
(``<sheet label>.<alias>: value``) on a freshly opened document; the part file itself is never
saved. Features in error after the recompute are reported (the body keeps its last valid
shape). tools/check.py then compares the exports with the committed STLs (Hausdorff distance)
or installs a missing one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FreeCAD as App  # noqa: E402
import Mesh  # noqa: E402
from common import BUILD, finish, manifest, say  # noqa: E402

OUT = BUILD / "regen"


def set_params(doc, params):
    for key, value in params.items():
        label, alias = key.rsplit(".", 1)
        sheets = [o for o in doc.getObjectsByLabel(label) if o.TypeId == "Spreadsheet::Sheet"]
        if len(sheets) != 1:
            raise KeyError(f"sheet {label!r} not found")
        cell = sheets[0].getCellFromAlias(alias)
        if not cell:
            raise KeyError(f"alias {alias!r} not in sheet {label!r}")
        sheets[0].set(cell, str(value))


def export(part, v):
    """Export one variant; returns 1 on failure."""
    doc = App.openDocument(str(part.source(v)))
    try:
        set_params(doc, v.params)
        doc.recompute()
        obj = doc.getObject(v.fcstd_body)
        if obj.Shape.isNull() or not obj.Shape.isValid():
            say(f"{part.id}/{v.id}: {v.fcstd_body} has no valid shape after recompute")
            return 1
        # datum objects (LCS) may lose their attachment; only solid features matter, and the
        # exported geometry is checked against the committed STL by tools/check.py
        broken = [o.Name for o in obj.Group if o.isDerivedFrom("PartDesign::Feature")
                  and ("Invalid" in o.State or "Error" in o.State)]
        if broken:
            say(f"{part.id}/{v.id}: warning, features in error (body keeps its last valid "
                f"shape): {broken}")
        out = OUT / part.id / f"{v.id}.stl"
        out.parent.mkdir(parents=True, exist_ok=True)
        Mesh.export([obj], str(out))
        say(f"{part.id}/{v.id}: exported {v.fcstd_body} {v.params or ''}")
        return 0
    except Exception as e:  # noqa: BLE001 (report per variant, keep going)
        say(f"{part.id}/{v.id}: FAILED {e}")
        return 1
    finally:
        App.closeDocument(doc.Name)


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".py")]
    bad = 0
    for part in manifest().parts.values():
        if only and part.id not in only:
            continue
        for v in part.variants.values():
            if not v.fcstd_body:
                continue
            src = part.source(v)
            if src is None or not src.is_file():
                say(f"{part.id}/{v.id}: {src.name if src else 'no fcstd'} missing (make split)")
                bad += 1
                continue
            bad += export(part, v)
    finish(bad, "export")


main()
