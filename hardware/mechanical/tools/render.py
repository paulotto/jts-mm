# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Render every part variant of parts.yaml and record its geometry (renders/index.json).

One PNG per variant (``renders/<part>__<variant>__iso.png``, 2000 x 1600 px, transparent
background, parallel projection, 256-colour palette). All variants of one part share the camera
and the scale, so they compare 1:1; the scale (mm per pixel) is stored in the index for scale
bars, with size, volume, area and the masses (common.masses). A render is redone only when its
STL, the part's camera or this script's style changed (the index keeps the hashes), so
`make renders` is fast and a second run changes nothing.

    ../.venv/bin/python tools/render.py [--force] [part ...]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import trimesh
from common import (INDEX, OLD, PRINTED, RENDERS, manifest, masses, print_settings, read_json,
                    say, sha, shrink_png, write_json)

STYLE = "v4"            # bump to re-render everything after a style change
SIZE = (2000, 1600)     # 300 dpi at ~6.7 in: sharp in a text-width figure
DEFAULT_VIEW = {"elevation": 35.0, "azimuth": -55.0, "roll": 0.0, "up": "z"}


def geometry(stl) -> dict:
    m = trimesh.load(stl, force="mesh")
    m.merge_vertices()
    ext = m.bounds[1] - m.bounds[0]
    # volume by the divergence theorem: exact for closed meshes and within 0.1 % of the CAD
    # volume for the FreeCAD exports with T-junction cracks (watertight: false)
    return {"triangles": int(len(m.faces)), "watertight": bool(m.is_watertight),
            "volume_mm3": round(float(m.volume), 1),
            "area_mm2": round(float(m.area), 1),
            "size_mm": [round(float(v), 2) for v in ext]}


def camera(view: dict, center: np.ndarray, dist: float):
    """Position and up vector for an elevation/azimuth view about the part's ``up`` axis."""
    up = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]),
          "z": np.array([0, 0, 1.0]), "-y": np.array([0, -1.0, 0]),
          "-z": np.array([0, 0, -1.0])}[view.get("up", "z")]
    el, az = np.radians(view["elevation"]), np.radians(view["azimuth"])
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    d = np.cos(el) * (np.cos(az) * e1 + np.sin(az) * e2) + np.sin(el) * up
    return center + dist * d, up


def render(stl, out, colour, view, center, scale):
    import pyvista as pv

    pl = pv.Plotter(off_screen=True, window_size=SIZE, lighting="light_kit")
    pl.add_mesh(pv.read(str(stl)), color=colour, smooth_shading=True, split_sharp_edges=True,
                specular=0.25, specular_power=20, ambient=0.15)
    pl.enable_parallel_projection()
    pos, up = camera(view, center, 1000.0)
    pl.camera_position = [tuple(pos), tuple(center), tuple(up)]
    pl.camera.roll += view.get("roll", 0.0)
    pl.camera.parallel_scale = scale
    pl.screenshot(str(out), transparent_background=True)
    pl.close()
    shrink_png(out, max(SIZE), quantize=True)
    return 2.0 * scale / SIZE[1]           # mm per pixel (vertical)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parts", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    man = manifest()
    index = read_json(INDEX, {})
    RENDERS.mkdir(exist_ok=True)
    views = (man.doc.get("render") or {}).get("views") or {}
    done = 0
    for part in man.parts.values():
        if a.parts and part.id not in a.parts:
            continue
        stls = {v.id: v.stl for v in part.variants.values() if v.stl and v.stl.is_file()}
        if not stls:
            continue
        # one camera and one scale per part: the union of its variants (variants overlap in
        # their design frame; a variant placed elsewhere is centred on its own box)
        boxes = {vid: trimesh.load(s, force="mesh").bounds for vid, s in stls.items()}
        half = max(float(np.linalg.norm(b[1] - b[0])) for b in boxes.values()) / 2 * 1.02
        view = {**DEFAULT_VIEW, **(views.get(part.id) or {})}
        for v in part.variants.values():
            if v.id not in stls:
                continue
            out = man.render(part.id, v.id)
            key = "/".join([sha(stls[v.id]), STYLE, repr(sorted(view.items())), f"{half:.3f}"])
            rel = f"{part.id}/{v.id}"
            if not a.force and index.get(rel, {}).get("key") == key and out.is_file():
                continue
            centre = boxes[v.id].mean(axis=0)
            colour = OLD if v.status == "superseded" else PRINTED
            mmpx = render(stls[v.id], out, colour, view, centre, half)
            index[rel] = {"key": key, "mm_per_px": round(mmpx, 5), "png": out.name,
                          "stl": str(stls[v.id].relative_to(man.root)), **geometry(stls[v.id])}
            done += 1
            say(f"rendered {rel}")
    valid = {f"{p.id}/{v.id}" for p in man.parts.values() for v in p.variants.values()}
    index = {k: val for k, val in index.items() if k in valid}
    # masses here, once, for every reader (analyses, paper table, viewer): they follow the
    # material and print record in parts.yaml without a re-render
    for p in man.parts.values():
        for v in p.variants.values():
            g = index.get(f"{p.id}/{v.id}")
            if g is not None:
                g["material"] = p.material.get("name", "")
                g.update(masses(g["volume_mm3"], g["area_mm2"],
                                float(p.material.get("density_g_cm3") or 0), print_settings(v)))
    write_json(INDEX, index)
    say(f"{done} rendered, {len(index)} in {INDEX.relative_to(man.root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
