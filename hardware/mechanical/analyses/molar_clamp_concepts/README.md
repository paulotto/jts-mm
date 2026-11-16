<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Molar clamp concepts for the JTS-MM teeth attachment

Concept study (2026-09-23) for a teeth attachment that clamps the lower molars from the lingual
and buccal side instead of being fixed to the incisors. **None of the five concepts was built as
drawn**; the study (undercut survey, molar arc, stop-and-pad reasoning) informed the V5 teeth
attachment in [`../../parts.yaml`](../../parts.yaml) (`inspired_by: molar_clamp_concepts`).
Overview of all parts and analyses: [`../../README.md`](../../README.md).

* `latex/molar_clamp_concepts.pdf` – the report (rationale, five concepts, spring sizing,
  build instructions, sources). Built from `latex/` (`cd latex && latexmk -pdf`; needs
  `cleveref`, `siunitx`, `tcolorbox`, `tabularx`).
* `latex/` – sections in `content/`, generated tables in `generated/`, figures in
  `figures/{svg,pdf,png}`: 2D figures as SVG (editable), PDF (used by LaTeX) and PNG (200 dpi
  preview); 3D renders as PNG at their native 1600 × 1230 px, stored as 256-colour palette images
  (`make concepts` runs `tools/downscale.py … --quantize 'render_*'`, no visible loss, ~6× smaller).
* `scripts/` – regenerate every figure, table and STL from the shared teeth scan
  [`../../scan/teeth_02.stl`](../../scan/teeth_02.stl), ~3 min (3D renders use VTK off-screen):
  `make concepts` in `hardware/mechanical` (scripts, renders shrunk, report built), or
  `bash scripts/run_all.sh` with `../../.venv/bin` first on `PATH` (`SKIP_LATEX=1` skips the
  report).
* `scripts/params.py` – every dimension of every concept in one place. Change a value, re-run, and
  the STLs, renders, drawings, parameter tables and CAD reference geometry follow.
* `cad/` – reference geometry for rebuilding the parts in FreeCAD (written by `s10_export_cad.py`):
  sweep paths (`path_*.csv/.dxf`), tooth section outlines at 13 stations (`section_*.csv`),
  `stations.csv`, `all_reference_geometry.dxf` and `import_reference_geometry.FCMacro` (a copy of
  `scripts/cad_macro_template.FCMacro`, written by `s10`). Run the macro in FreeCAD
  (Macro → Macros…) with the teeth model open (`../../parts/teeth_attachment/teeth_attachment.FCStd`
  contains `Teeth_02`). It creates `REF_curves` (B-spline paths), `REF_sections` (one sketch per
  station; sketch x = n toward the tongue, y = h occlusal height) and `REF_path_starts`. Long parts
  (bars, jaws): Part → Sweep a profile along a path, Frenet off. Short parts (loop, pads): Pad a
  sketch drawn on `SEC_M1_mid_R`. See Section 6 of the PDF.
* `stl/` – concept geometry in the frame of `Teeth_02` (voxel-based, not print files). Regenerated
  by `s04` and git-ignored (45 MB); decimated copies (5000 faces each) for the 3D viewer are in
  [`../../viewer/models/`](../../viewer/models/) (`make viewer`).
* `build/` – caches written by the scripts (git-ignored, regenerable).

## Scripts
`s00` export the teeth from a FreeCAD file (optional:
`freecad.cmd scripts/s00_export_teeth_freecad.py -- ../../parts/teeth_attachment/teeth_attachment.FCStd`,
writes `../../scan/teeth_02.stl`) · `s01` arch frame ·
`s02` voxel model and distance field · `s03` undercut survey · `s04` concept geometry (voxel masks
→ STL) · `s05` 3D renders · `s06` mechanics figures · `s07` section figures · `s08` spring sizing ·
`s09` dimensioned drawings and parameter tables · `s10` CAD reference geometry and macro.

## Editing figures
2D figures are saved as SVG with text kept as text (edit labels in Inkscape), plus PDF (used by
LaTeX) and PNG. If you edit an SVG by hand, save a PDF copy under the same name in
`latex/figures/pdf/`; re-running the scripts overwrites hand edits. Dimensioned drawings (`drw_*`)
are drawn from `params.py`; change dimensions there, not in the SVG. 3D renders are PNG; change
their content in `s04_concepts.py` (design masks) or the camera in `s05_render3d.py`.
