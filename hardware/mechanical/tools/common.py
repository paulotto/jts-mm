# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared paths and helpers of the hardware/mechanical tools.

Imported by the venv tools (../.venv, see Makefile) and by the FreeCAD scripts (freecad.cmd):
everything here needs only the standard library and PyYAML at import time; mesh and plotting
libraries are imported inside the functions that use them.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from pathlib import Path

MECH = Path(__file__).resolve().parents[1]
REPO = MECH.parents[1]
sys.path.insert(0, str(REPO / "software" / "biomech"))

from biomech import hwparts  # noqa: E402  (pure module: PyYAML + pathlib)

RENDERS = MECH / "renders"
FIGURES = MECH / "figures"
TABLES = MECH / "tables"
BUILD = MECH / "build"
INDEX = RENDERS / "index.json"

# colours: printed parts, superseded parts, teeth (the concept study's tooth colour)
PRINTED = "#8fb3d9"
OLD = "#b8bcc2"
TOOTH = "#f3e6d2"

# printed-mass estimate: wall lines of 0.42 mm (0.4 mm nozzle). Defaults for variants without a
# print record: 2 walls and about 10 % infill, as the parts were printed (user, 2026-11-04; the two
# print records say 15 % and keep it)
WALL_MM = 0.42
DEFAULT_PRINT = {"walls": 2, "infill": 0.10}

say = functools.partial(print, flush=True)      # freecad.cmd drops unflushed output


def finish(bad: int, what: str) -> int:
    """Print the summary line; exit 1 on failures (os._exit: FreeCAD swallows SystemExit)."""
    say(f"{what}:", "FAILED" if bad else "ok")
    if bad:
        sys.stdout.flush()
        os._exit(1)
    return 0


def manifest() -> hwparts.Manifest:
    return hwparts.load(MECH / "parts.yaml")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def write_text(path: Path, text: str) -> None:
    """Write only on change (keeps `make all` twice diff-free)."""
    if not path.is_file() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, sort_keys=True) + "\n")


# -- geometry ------------------------------------------------------------------------------------
def load_mesh(path):
    import trimesh

    return trimesh.load(path, force="mesh")


def distances(points, mesh):
    """Unsigned distance of each point to the surface of ``mesh``."""
    import numpy as np
    from trimesh.proximity import ProximityQuery

    return np.abs(ProximityQuery(mesh).signed_distance(points))


def hausdorff(a, b) -> float:
    """Symmetric Hausdorff distance of two meshes (over their vertices)."""
    return float(max(distances(a.vertices, b).max(), distances(b.vertices, a).max()))


def print_settings(variant) -> dict:
    """Walls and infill of the variant's print record (print/*.json), else the defaults."""
    if variant.print is None:
        return dict(DEFAULT_PRINT)
    s = read_json(variant.print.with_suffix(".json"), {})
    return {"walls": int(s.get("wall_loops") or DEFAULT_PRINT["walls"]),
            "infill": float(str(s.get("sparse_infill_density", "15%")).rstrip("%")) / 100}


def masses(volume_mm3: float, area_mm2: float, density_g_cm3: float, pr: dict) -> dict:
    """Solid mass (100 % infill) and the printed estimate: walls (count x WALL_MM over the
    surface) solid, the rest at the infill fraction."""
    rho = density_g_cm3 / 1000.0                        # g/mm3
    shell = min(volume_mm3, area_mm2 * pr["walls"] * WALL_MM)
    return {"mass_solid_g": round(volume_mm3 * rho, 1),
            "mass_printed_g": round((shell + pr["infill"] * (volume_mm3 - shell)) * rho, 1)}


# -- figures -------------------------------------------------------------------------------------
def mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
                         "font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "hardware-mechanical",
                         "path.simplify": False})
    return plt


def save_fig(fig, name: str, dpi: int = 300) -> None:
    """figures/<name>.{pdf,svg,png}; metadata stripped so a re-run gives identical files.
    ``dpi`` applies to all three: the vector files embed their renders (and rasterized layers)
    at it, so without it matplotlib would resample them to its default 100 dpi."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "dpi": dpi}
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}, **kw)
    fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None}, **kw)
    fig.savefig(FIGURES / f"{name}.png", metadata={"Software": None}, **kw)


def shrink_png(path: Path, max_px: int, quantize: bool) -> bool:
    """Downscale a PNG in place to at most ``max_px`` on the long side and (``quantize``) to a
    256-colour palette with alpha (about 6x smaller, no visible loss on shaded renders).
    Idempotent: an image already small enough / already a palette image is left alone.
    Returns whether the file changed."""
    from PIL import Image

    im = Image.open(path)
    changed = False
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px), Image.LANCZOS)
        changed = True
    if quantize and im.mode != "P":
        im = im.convert("RGBA").quantize(colors=256, method=Image.Quantize.FASTOCTREE,
                                         dither=Image.Dither.NONE)
        changed = True
    if changed:
        im.save(path, optimize=True)
    return changed
