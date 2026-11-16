# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Check parts.yaml and the part files; install CAD exports (make check / make parts).

    ../.venv/bin/python tools/check.py            manifest, files, renders, CAD exports
    ../.venv/bin/python tools/check.py --update   also replace committed STLs by the CAD exports
                                                  (after an intentional CAD edit)

* manifest: every file exists, statuses known, usage names known parts/variants/studies
  (errors); CAD pending and unspecified usage are listed as notes;
* CAD exports (build/regen from tools/export_variants.py): an export whose variant has no STL yet
  is installed; otherwise it must match the committed STL within TOL (symmetric Hausdorff
  distance), so a stale or broken model never silently replaces printed geometry;
* renders: every variant with an STL has a render and an index entry (after make renders).
Exit 1 on any error.
"""

from __future__ import annotations

import argparse
import shutil
import sys

from common import BUILD, INDEX, REPO, hausdorff, load_mesh, manifest, read_json

TOL = 0.01      # mm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--no-renders", action="store_true")
    a = ap.parse_args(argv)
    man = manifest()
    # a usage key must name a study of experiments/
    studies = tuple(p.name for p in (REPO / "experiments").iterdir() if p.is_dir())
    errors: list[str] = []
    regen = BUILD / "regen"
    for part in man.parts.values():
        for v in part.variants.values():
            new = regen / part.id / f"{v.id}.stl"
            if not v.fcstd_body or not new.is_file() or v.stl is None:
                continue
            if not v.stl.is_file():
                v.stl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(new, v.stl)
                print(f"installed {v.stl.relative_to(man.root)} (from {part.id}.FCStd {v.fcstd_body})")
                continue
            d = hausdorff(load_mesh(new), load_mesh(v.stl))
            if d <= TOL:
                print(f"cad ok   {part.id}/{v.id}: export = committed STL ({d:.4f} mm)")
            elif a.update:
                shutil.copyfile(new, v.stl)
                print(f"updated  {part.id}/{v.id}: committed STL replaced ({d:.3f} mm)")
            else:
                errors.append(f"{part.id}/{v.id}: CAD export differs from the committed STL by "
                              f"{d:.3f} mm (run `make parts UPDATE=1` if the CAD edit is intended)")
    perr, notes = man.problems(studies)        # after installing exports (they may add STLs)
    errors = perr + errors
    if not a.no_renders and INDEX.is_file():
        index = read_json(INDEX, {})
        for part in man.parts.values():
            for v in part.variants.values():
                if v.stl is not None and v.stl.is_file() and (
                        f"{part.id}/{v.id}" not in index or not man.render(part.id, v.id).is_file()):
                    errors.append(f"{part.id}/{v.id}: no render (make renders)")
    for n in notes:
        print("note    ", n)
    for e in errors:
        print("ERROR   ", e)
    print(f"check: {len(errors)} errors, {len(notes)} notes")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
