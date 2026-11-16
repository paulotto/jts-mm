# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Export the data of the 3D viewer (make viewer): viewer/manifest.json + decimated meshes.

The viewer (viewer/index.html, served from the repo root: `make serve`) loads the part STLs
directly from parts/ (they are small); only the teeth scan and the voxel concept geometry
(analyses/molar_clamp_concepts/stl, git-ignored) are decimated into viewer/models/.
"""

from __future__ import annotations

import sys

import numpy as np
import trimesh
from common import INDEX, MECH, REPO, manifest, read_json, sha, write_json

MODELS = MECH / "viewer" / "models"
CONCEPTS = MECH / "analyses" / "molar_clamp_concepts" / "stl"
SCAN_FACES = 40000
CONCEPT_FACES = 5000
CONCEPT_ITEMS = [  # file stem, title, kind (colour)
    ("c1_printed", "C1 snap shell", "printed"),
    ("c2_rigid", "C2 clothespin jaw: frame", "rigid"), ("c2_spring", "C2: flexure jaw", "spring"),
    ("c3_rigid", "C3 wire clasps: plate", "rigid"), ("c3_wire_0p7mm", "C3: 0.7 mm clasps", "wire"),
    ("c4_rigid", "C4 spring arch: pads", "rigid"), ("c4_printed", "C4: printed parts", "printed"),
    ("c4_arch_wire_0p9mm", "C4: 0.9 mm arch wire", "wire"),
    ("c5_rigid", "C5 rigid clamp", "rigid"), ("c5_tpu", "C5: TPU liner", "tpu")]


def decimate(src, dst, faces, cache):
    """Decimated binary STL (fast-simplification); redone only when the source changed."""
    key = f"{sha(src)}/{faces}"
    if cache.get(dst.name) == key and dst.is_file():
        return
    import fast_simplification

    m = trimesh.load(src, force="mesh")
    if len(m.faces) > faces:
        v, f = fast_simplification.simplify(np.asarray(m.vertices, np.float32),
                                            np.asarray(m.faces), 1 - faces / len(m.faces))
        m = trimesh.Trimesh(v, f, process=True)
    trimesh.repair.fix_normals(m)
    MODELS.mkdir(parents=True, exist_ok=True)
    m.export(dst)
    cache[dst.name] = key
    print(f"decimated {src.name} -> {dst.relative_to(MECH)} ({len(m.faces)} faces)")


def url(path) -> str:
    """URL of a repo file as served from the repo root (make serve)."""
    return "/" + str(path.resolve().relative_to(REPO.resolve()))


def main() -> int:
    man = manifest()
    index = read_json(INDEX, {})
    cache = read_json(MODELS / "sources.json", {})
    decimate(MECH / "scan" / "teeth_02.stl", MODELS / "teeth_02.stl", SCAN_FACES, cache)
    concepts = []
    for stem, title, kind in CONCEPT_ITEMS:
        src = CONCEPTS / f"{stem}.stl"
        dst = MODELS / f"concept_{stem}.stl"
        if src.is_file():
            decimate(src, dst, CONCEPT_FACES, cache)
        if dst.is_file():
            concepts.append({"id": stem, "title": title, "kind": kind, "url": url(dst),
                             "concept": stem.split("_")[0]})
    write_json(MODELS / "sources.json", cache)
    xc = read_json(MECH / "tables" / "crosscheck.json", {}).get("geometry", {})
    overlays = {g["part"] + "/" + g["variant"]: g for g in xc.values()}
    parts = []
    for p in man.parts.values():
        vs = []
        for v in p.variants.values():
            g = index.get(f"{p.id}/{v.id}")
            if v.stl is None or not v.stl.is_file() or g is None:
                continue
            vs.append({"id": v.id, "title": v.title, "status": v.status, "url": url(v.stl),
                       "cad": "pending" if v.cad_pending else ("yes" if v.fcstd_body else "no"),
                       "date": v.date, "note": v.note, "size_mm": g["size_mm"],
                       "volume_mm3": g["volume_mm3"], "triangles": g["triangles"],
                       "mass_solid_g": g.get("mass_solid_g"),
                       "mass_printed_g": g.get("mass_printed_g"),
                       "overlay": overlays.get(f"{p.id}/{v.id}")})
        used = {s: (u or {}).get(p.id) for s, u in man.usage.items()}
        parts.append({"id": p.id, "title": p.title, "role": p.role,
                      "material": p.material.get("name", ""), "variants": vs, "used": used,
                      "seat_on_scan": (man.doc["parts"][p.id] or {}).get("seat_on_scan")})
    write_json(MECH / "viewer" / "manifest.json", {
        "parts": parts, "concepts": concepts,
        "scan": {"title": "Teeth scan (Teeth_02, lower arch)", "url": url(MODELS / "teeth_02.stl")},
        "studies": list(man.usage)})
    print(f"viewer/manifest.json: {len(parts)} parts, {len(concepts)} concept meshes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
