# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Read the slicer records (print/*.3mf, Bambu Studio): settings and the printed geometry.

    ../.venv/bin/python tools/print_records.py        (part of make parts)

For every variant with a ``print`` record: writes ``print/<record>.json`` (printer, filament,
layer height, infill, walls, supports, slicer, date) and, when the variant's ``stl`` does not
exist yet, extracts the printed object (first object of the plate, moved back by the slicer's
``source_offset`` into the frame of the STL it was imported from) to that path.
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import trimesh
from common import manifest, write_json

NS = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
KEYS = ("printer_model", "print_settings_id", "layer_height", "sparse_infill_density",
        "sparse_infill_pattern", "wall_loops", "enable_support", "support_type",
        "nozzle_diameter")


def settings(z: zipfile.ZipFile) -> dict:
    cfg = json.loads(z.read("Metadata/project_settings.config"))
    model = z.read("3D/3dmodel.model").decode()
    meta = dict(re.findall(r'<metadata name="([^"]+)">([^<]*)</metadata>', model))
    ms = ET.fromstring(z.read("Metadata/model_settings.config"))
    obj = ms.find("object")
    ext = int(obj.find("metadata[@key='extruder']").get("value")) if obj is not None else 1
    out = {k: cfg.get(k) for k in KEYS}
    out["nozzle_diameter"] = (out["nozzle_diameter"] or [None])[0]
    out["filament"] = cfg.get("filament_settings_id", [None])[ext - 1]
    out["filament_type"] = cfg.get("filament_type", [None])[ext - 1]
    out["enable_support"] = out["enable_support"] in ("1", 1, True)
    out["slicer"] = meta.get("Application", "")
    out["date"] = meta.get("CreationDate", "")
    out["copies_on_plate"] = len(ms.findall("object"))
    return out


def printed_mesh(z: zipfile.ZipFile) -> trimesh.Trimesh:
    ms = ET.fromstring(z.read("Metadata/model_settings.config"))
    part = ms.find("object/part")
    off = np.array([float(part.find(f"metadata[@key='source_offset_{a}']").get("value"))
                    for a in "xyz"])
    name = sorted((n for n in z.namelist() if re.match(r"3D/Objects/object_\d+\.model$", n)),
                  key=lambda n: int(re.findall(r"\d+", n)[-1]))[0]
    mesh = ET.fromstring(z.read(name)).find(".//m:mesh", NS)
    v = np.array([[float(x.get(a)) for a in "xyz"] for x in mesh.find("m:vertices", NS)])
    f = np.array([[int(t.get(a)) for a in ("v1", "v2", "v3")] for t in mesh.find("m:triangles", NS)])
    return trimesh.Trimesh(v + off, f, process=True)


def main() -> int:
    man = manifest()
    for part in man.parts.values():
        for v in part.variants.values():
            if v.print is None:
                continue
            with zipfile.ZipFile(v.print) as z:
                write_json(v.print.with_suffix(".json"), settings(z))
                if v.stl is not None and not v.stl.is_file():
                    v.stl.parent.mkdir(parents=True, exist_ok=True)
                    printed_mesh(z).export(v.stl)
                    print(f"{part.id}/{v.id}: extracted {v.stl.relative_to(man.root)}")
            print(f"{part.id}/{v.id}: settings -> {v.print.with_suffix('.json').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
