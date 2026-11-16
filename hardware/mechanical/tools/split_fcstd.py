# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Split source/JTS_Models_simple_aruco.FCStd into one FreeCAD project per part (FreeCAD 1.1).

    freecad.cmd tools/split_fcstd.py --pass [--force]     (make split; after the source changes)

For every part of parts.yaml with an ``fcstd``: open the source, keep the bodies its variants
name (``fcstd_body``) plus ``fcstd_extra`` objects with everything they depend on (parameter
sheets, sketches of other bodies they reference), delete all other objects, and save it as the
part's FCStd. Deleting (instead of copying into a new document) keeps FreeCAD's topological
element names, so the parametric variants (e.g. ``ta_separation``) still recompute correctly.
Parameter sheets are cleaned on the copy: cells without content and alias (whole-column unit
formatting, ~32 000 per sheet in the source) are cleared, which takes the 17 MB Document.xml
down to a few hundred kB. Every copied body must recompute to the same volume and bounding box as
in the source (exit 1 otherwise). An existing part FCStd is overwritten only with ``--force`` (it
may hold edits made after the split).
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import FreeCAD as App  # noqa: E402
from common import BUILD, MECH, finish, manifest, say  # noqa: E402

SRC = MECH / "source" / "JTS_Models_simple_aruco.FCStd"


def clean_sheet(sheet):
    n = 0
    for c in sheet.getUsedCells():
        if not sheet.getContents(c) and not sheet.getAlias(c):
            sheet.clear(c)
            n += 1
    return n


def signature(shape):
    b = shape.BoundBox
    return (round(shape.Volume, 4), tuple(round(v, 4) for v in
                                          (b.XMin, b.YMin, b.ZMin, b.XMax, b.YMax, b.ZMax)))


def main():
    force = "--force" in sys.argv
    src = App.openDocument(str(SRC))
    src.recompute()
    bad = 0
    for part in manifest().parts.values():
        if part.fcstd is None:
            continue
        rel = part.fcstd.relative_to(MECH)
        if part.fcstd.exists() and not force:
            say(f"{part.id}: {rel} exists (use --force to overwrite)")
            continue
        names = part.bodies + list(part.fcstd_extra)
        if any(src.getObject(n) is None for n in names):
            say(f"{part.id}: unknown object in {names}")
            bad += 1
            continue
        work = BUILD / f"split_{part.id}.FCStd"         # src stays open unchanged
        work.parent.mkdir(exist_ok=True)
        shutil.copyfile(SRC, work)
        doc = App.openDocument(str(work))
        keep = set()
        for o in (doc.getObject(n) for n in names):
            keep |= {o.Name} | {d.Name for d in o.OutListRecursive}
        for o in reversed(doc.TopologicalSortedObjects):     # dependents before dependencies
            if o.Name not in keep:
                doc.removeObject(o.Name)
        cleared = sum(clean_sheet(o) for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet")
        doc.recompute()
        for n in part.bodies:
            a, b = src.getObject(n), doc.getObject(n)
            ok = b is not None and b.Shape.isValid() and signature(a.Shape) == signature(b.Shape)
            say(f"{part.id}: {n} ({a.Label}) {'ok' if ok else 'MISMATCH'} "
                f"vol {a.Shape.Volume:.1f} mm3")
            bad += not ok
        for n in part.fcstd_extra:  # reference objects (teeth scan): hidden
            o = doc.getObject(n)
            if hasattr(o, "Visibility"):
                o.Visibility = False
        part.fcstd.parent.mkdir(parents=True, exist_ok=True)
        doc.saveAs(str(part.fcstd))
        say(f"{part.id}: {len(doc.Objects)} objects, {cleared} empty sheet cells cleared -> "
            f"{rel} ({part.fcstd.stat().st_size / 1e6:.2f} MB)")
        App.closeDocument(doc.Name)
    finish(bad, "split")


main()
