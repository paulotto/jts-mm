<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Mechanical parts (3D-printed hardware)

The 3D-printed parts of the JTS-MM setup — teeth attachment, mouth and head markers, calibration
tool, CT calibrater, teeth protector, trueness ladder and array — with every version, the FreeCAD
models, the print records, and the design analyses behind the teeth attachment. One manifest,
[`parts.yaml`](parts.yaml), lists all of it and states **which version each study used**; the
paper figures, the gallery and the 3D viewer read it (through
[`biomech/hwparts.py`](../../software/biomech/biomech/hwparts.py)). Design rationale, results and
open items: [`docs/hardware/h11_mechanical_parts.md`](../../docs/hardware/h11_mechanical_parts.md).

## Quickstart

```bash
make venv          # once: tools environment (.venv) from tools/requirements.txt
make all           # parts, renders, analyses, viewer data, README tables, check
make serve         # then open http://127.0.0.1:8072/hardware/mechanical/viewer/
```

`make all` needs FreeCAD 1.1 (`freecad.cmd`, the snap) for its first step (`make parts`); without
FreeCAD run `make renders analyses viewer docs check`, which work from the committed STLs.
`make all` twice in a row changes no file.

## Which version a study used

Fill `usage` at the end of [`parts.yaml`](parts.yaml) with variant ids from the parts table below
(`null` = not yet specified), per participant under `usage.overrides` if a part changed during a
study:

```yaml
usage:
  icorr2027:
    teeth_attachment: v5_5mm
    mouth_marker: qualisys_aruco
  overrides:
    jtsmm_validation: {s03: {teeth_attachment: v5_4mm}}
```

Then run `make docs` (updates the tables here); nothing else is needed:
- the setup figures of both papers (`icorr_setup`, `val_setup`, form `schematic`) show the render
  of the used head marker, mouth marker and calibration tool and name the versions in the caption;
- the supplement items `hw_parts` / `hw_table` (dataset) and `val_hw_parts` / `val_hw_table`
  (systems) show and list the used versions;
- `biomech check-outputs` stops warning "part versions of the study specified".

While a study's usage is `null`, the setup figures keep the original JTS renders and the
supplement items show each part's `candidate` version, marked "(candidate)". A study without a
`usage` entry (e.g. the synthetic test study) lists every part that has a candidate.

<!-- usage:begin -->
| Study | Teeth attachment | Mouth marker | Head marker | Calibration tool | CT calibrater | Trueness ladder | Trueness array |
|---|---|---|---|---|---|---|---|
| icorr2027 | `v5_print_2026-10-03` | `aruco` | `aruco_2026-08-18` | `aruco` | – | – | – |
| jtsmm_validation | `v5_print_2026-10-03` | `qualisys_aruco` | `qualisys_aruco` | `qualisys_aruco` | `v1` | `v1` | `v1` |
<!-- usage:end -->

## Parts and versions

Generated from `parts.yaml` by `make docs`. *CAD pending*: the STL is the authoritative geometry
and the FreeCAD model is still to be added (see [Workflows](#workflows)).

<!-- parts:begin -->
| Part | Variant | Status | CAD | Size (mm) | Volume (cm³) | Date | Render |
|---|---|---|---|---|---|---|---|
| Teeth attachment | [JTS original](../../software/jaw_tracking/models/JTS_Teeth_Attachment.stl) (`jts`) | superseded | STL only | 60 × 24 × 49 | 1.54 |  | [iso](renders/teeth_attachment__jts__iso.png) |
| Teeth attachment | [Version 1](parts/teeth_attachment/v1.stl) (`v1`) | superseded | `teeth_attachment.FCStd` Body001 | 60 × 29 × 48 | 1.84 |  | [iso](renders/teeth_attachment__v1__iso.png) |
| Teeth attachment | [Version 2](parts/teeth_attachment/v2.stl) (`v2`) | superseded | `teeth_attachment.FCStd` Body007 | 47 × 24 × 39 | 1.96 |  | [iso](renders/teeth_attachment__v2__iso.png) |
| Teeth attachment | [Version 4](parts/teeth_attachment/v4.stl) (`v4`) | superseded | `teeth_attachment.FCStd` Body008 | 51 × 22 × 23 | 1.48 |  | [iso](renders/teeth_attachment__v4__iso.png) |
| Teeth attachment | [Version 5 draft](parts/teeth_attachment/v5_draft.stl) (`v5_draft`) | superseded | `teeth_attachment.FCStd` Body010 | 73 × 21 × 58 | 2.43 |  | [iso](renders/teeth_attachment__v5_draft__iso.png) |
| Teeth attachment | [Version 5, 5 mm (print 2026-10-03)](parts/teeth_attachment/v5_print_2026-10-03.stl) (`v5_print_2026-10-03`) | printed | STL only | 64 × 21 × 55 | 2.28 | 2026-10-03 | [iso](renders/teeth_attachment__v5_print_2026-10-03__iso.png) |
| Teeth attachment | [Version 5, 4 mm](parts/teeth_attachment/v5_4mm.stl) (`v5_4mm`) | variant | pending | 70 × 21 × 59 | 2.48 | 2026-10-08 | [iso](renders/teeth_attachment__v5_4mm__iso.png) |
| Teeth attachment | [Version 5, 4.5 mm](parts/teeth_attachment/v5_4_5mm.stl) (`v5_4_5mm`) | variant | `teeth_attachment.FCStd` Body011 | 70 × 21 × 58 | 2.51 | 2026-10-09 | [iso](renders/teeth_attachment__v5_4_5mm__iso.png) |
| Teeth attachment | [Version 5, 5 mm](parts/teeth_attachment/v5_5mm.stl) (`v5_5mm`) | variant | `teeth_attachment.FCStd` Body011 | 70 × 21 × 58 | 2.45 | 2026-10-08 | [iso](renders/teeth_attachment__v5_5mm__iso.png) |
| Teeth attachment | [Version 5, 5.5 mm](parts/teeth_attachment/v5_5_5mm.stl) (`v5_5_5mm`) | variant | `teeth_attachment.FCStd` Body011 | 70 × 21 × 58 | 2.39 | 2026-10-09 | [iso](renders/teeth_attachment__v5_5_5mm__iso.png) |
| Teeth attachment | [Version 5, 6.5 mm](parts/teeth_attachment/v5_6_5mm.stl) (`v5_6_5mm`) | variant | pending | 70 × 21 × 58 | 2.45 | 2026-10-09 | [iso](renders/teeth_attachment__v5_6_5mm__iso.png) |
| Mouth marker | [JTS original](../../software/jaw_tracking/models/JTS_Mouth_Marker.stl) (`jts`) | superseded | STL only | 87 × 52 × 9 | 12.04 |  | [iso](renders/mouth_marker__jts__iso.png) |
| Mouth marker | [Version 1](parts/mouth_marker/v1.stl) (`v1`) | superseded | `mouth_marker.FCStd` Body | 145 × 104 × 9 | 47.50 |  | [iso](renders/mouth_marker__v1__iso.png) |
| Mouth marker | [ArUco + IMU (CAD state 2026-10-09)](parts/mouth_marker/aruco.stl) (`aruco`) | variant | `mouth_marker.FCStd` Body005 | 80 × 65 × 12 | 9.05 |  | [iso](renders/mouth_marker__aruco__iso.png) |
| Mouth marker | [Qualisys + ArUco](parts/mouth_marker/qualisys_aruco.stl) (`qualisys_aruco`) | variant | `qualisys_aruco.FCStd` Body005 | 105 × 65 × 12 | 12.85 | 2026-10-13 | [iso](renders/mouth_marker__qualisys_aruco__iso.png) |
| Head marker | [JTS original](../../software/jaw_tracking/models/JTS_Head_Marker.stl) (`jts`) | superseded | STL only | 179 × 93 × 25 | 23.66 |  | [iso](renders/head_marker__jts__iso.png) |
| Head marker | [ArUco + IMU (print 2026-08-18)](parts/head_marker/aruco_2026-08-18.stl) (`aruco_2026-08-18`) | printed | `head_marker.FCStd` Body002 | 179 × 104 × 31 | 44.45 | 2026-08-18 | [iso](renders/head_marker__aruco_2026-08-18__iso.png) |
| Head marker | [Qualisys + ArUco](parts/head_marker/qualisys_aruco.stl) (`qualisys_aruco`) | variant | `qualisys_aruco.FCStd` Body002 | 179 × 134 × 31 | 50.15 | 2026-10-13 | [iso](renders/head_marker__qualisys_aruco__iso.png) |
| Calibration tool | [JTS original](../../software/jaw_tracking/models/JTS_Calibration_Tool.stl) (`jts`) | superseded | STL only | 79 × 232 × 10 | 32.75 |  | [iso](renders/calibration_tool__jts__iso.png) |
| Calibration tool | [ArUco (CAD state 2026-10-09)](parts/calibration_tool/aruco.stl) (`aruco`) | superseded | `calibration_tool.FCStd` Body003 | 137 × 232 × 10 | 65.36 |  | [iso](renders/calibration_tool__aruco__iso.png) |
| Calibration tool | [Qualisys + ArUco](parts/calibration_tool/qualisys_aruco.stl) (`qualisys_aruco`) | variant | `qualisys_aruco.FCStd` Body003 | 137 × 232 × 10 | 64.77 | 2026-10-13 | [iso](renders/calibration_tool__qualisys_aruco__iso.png) |
| CT calibrater | [CAD state 2026-10-09](parts/ct_calibrater/cad_2026-10-09.stl) (`cad_2026-10-09`) | superseded | `ct_calibrater.FCStd` Body006 | 20 × 10 × 20 | 3.98 |  | [iso](renders/ct_calibrater__cad_2026-10-09__iso.png) |
| CT calibrater | [Version 1](parts/ct_calibrater/v1.stl) (`v1`) | variant | pending | 20 × 10 × 20 | 3.92 | 2026-10-19 | [iso](renders/ct_calibrater__v1__iso.png) |
| Teeth protector | [Version 1](parts/teeth_protector/v1.stl) (`v1`) | variant | `teeth_protector.FCStd` Body009 | 16 × 10 × 2 | 0.12 | 2026-09-23 | [iso](renders/teeth_protector__v1__iso.png) |
| Trueness ladder | [Version 1](parts/trueness_ladder/v1.stl) (`v1`) | variant | pending | 230 × 15 × 50 | 161.50 | 2026-10-01 | [iso](renders/trueness_ladder__v1__iso.png) |
| Trueness array | [Version 1](parts/trueness_array/v1.stl) (`v1`) | variant | pending | 100 × 65 × 10 | 43.09 | 2026-10-02 | [iso](renders/trueness_array__v1__iso.png) |
<!-- parts:end -->

## Layout

| Path | Contents |
|---|---|
| [`parts.yaml`](parts.yaml) | the manifest: parts, variants, candidates, analyses, render cameras, usage ([fields](#partsyaml-fields)) |
| `parts/<part>/` | `<part>.FCStd` (one FreeCAD project per part, all its versions as bodies) and one STL per variant |
| [`scan/teeth_02.stl`](scan/teeth_02.stl) | lower-arch scan used by the concept study and the viewer (also hidden in `teeth_attachment.FCStd`) |
| [`print/`](print/) | slicer records (Bambu Studio 3MF) and their settings (`*.json`, written by `make parts`) |
| `source/` | local only (git-ignored): the original combined FreeCAD file, for `make split`; master copy outside the repository |
| [`renders/`](renders/) | one render per variant (2000 × 1600 px, palette PNG) + `index.json` (size, volume, area, mm per pixel, material, solid and printed mass) |
| [`figures/`](figures/), [`tables/`](tables/) | analysis figures (PDF and SVG with images at 300 dpi, PNG 300 dpi) and tables (CSV, LaTeX; `crosscheck.json` also has the measured geometry) |
| [`viewer/`](viewer/) | interactive 3D viewer (three.js r160, offline), `manifest.json`, decimated meshes in `models/` (scan 40 000, concepts 5000 faces) |
| [`analyses/incisor_attachment/`](analyses/incisor_attachment/README.md) | why flat jaws slip off a lower incisor; arch vs. straight wings |
| [`analyses/molar_clamp_concepts/`](analyses/molar_clamp_concepts/README.md) | molar-clamp concept study (scripts, report, CAD reference geometry) |
| [`tools/`](tools/) | the scripts behind `make` (shared helpers in `tools/common.py`) |

## `parts.yaml` fields

| Level | Field | Meaning |
|---|---|---|
| part | `title`, `role` | reader-facing name and one-line purpose (tables, viewer) |
| part | `fcstd` | the part's FreeCAD file (`parts/<part>/<part>.FCStd`) |
| part | `fcstd_extra` | objects kept hidden in the part file besides the bodies (`Teeth_02`) |
| part | `material` | `{name, density_g_cm3}` for the masses |
| part | `legacy_render` | the original JTS render (setup figures while usage is `null`) |
| part | `candidate` | variant shown, marked "(candidate)", while usage is `null` |
| part | `inspired_by` | analyses behind the part |
| part | `seat_on_scan` | placement of the part on `scan/teeth_02.stl` (`null`; for a future fit map) |
| variant | `title` | reader-facing; must not look like a plan code ("V5", "C1": the paper text would rewrite it), hence "Version 5" |
| variant | `status` | `variant` (candidate version), `printed` (slicer record), `superseded`, `used`, `concept` |
| variant | `stl` | the geometry that was or would be printed (required) |
| variant | `fcstd_body` | the FreeCAD body it is exported from (`make parts` checks the export against `stl`) |
| variant | `fcstd` | its own FreeCAD project when the body is not in the part's `fcstd` (the Qualisys + ArUco markers: split from the reflective-marker model of 2026-11-04) |
| variant | `params` | `{<sheet label>.<alias>: value}` set before the export (version-5 variants: `Params.ta_separation`) |
| variant | `cad_pending` | `true`: STL only, model to be added |
| variant | `print` | slicer record (`print/*.3mf`); its settings feed the printed-mass estimate |
| variant | `date`, `note` | export or print date; provenance remarks |
| top | `assemblies` | setup bodies of the figures (head piece, mouth piece, calibration tool) → parts, legacy render |
| top | `analyses` | design analyses with their figures (gallery, supplement forms `concepts` / `incisor`) |
| top | `render.views` | per-part render camera (`elevation`, `azimuth`, `roll`, `up`) |
| top | `usage` | variant per study and part; `overrides.<study>.<pid>.<part>` per participant |

## Commands

From `hardware/mechanical/` unless noted.

### Make targets

| Target | What it does |
|---|---|
| `make venv` | creates `.venv` with `tools/requirements.txt` (once) |
| `make all` | `parts renders analyses viewer docs check` |
| `make parts` | reads the print records (`tools/print_records.py`), exports every CAD-backed variant with FreeCAD (`tools/export_variants.py` → `build/regen/`, log `build/export.log`) and compares it with the committed STL (Hausdorff ≤ 0.01 mm; `tools/check.py`); installs STLs of new CAD variants. `UPDATE=1` replaces committed STLs after an intentional CAD edit |
| `make renders` | renders new or changed variants (`tools/render.py`; cameras in `render.views`) and refreshes `renders/index.json` (geometry and masses) |
| `make analyses` | lineage and marker figures, version-5 variant distances, parts and cross-check tables (`tools/analyses.py`) |
| `make viewer` | `viewer/manifest.json` and the decimated scan and concept meshes (`tools/viewer_export.py`) |
| `make docs` | the generated tables of this README (`tools/docs.py`) |
| `make check` | manifest, files, renders, CAD exports (`tools/check.py`; exit 1 on errors) |
| `make concepts` | re-runs the molar-clamp concept study (~3 min), shrinks its renders, builds its report |
| `make split` | re-creates the part FCStds from `source/JTS_Models_simple_aruco.FCStd` (copy it there first; only if the combined file changes again). `FORCE=1` overwrites part files, which may hold later edits |
| `make serve` | serves the repository on port 8072 for the viewer |
| `make clean` | removes `build/` |

### Tool scripts (what `make` calls; run with `.venv/bin/python` or `freecad.cmd`)

| Command | Options |
|---|---|
| `.venv/bin/python tools/render.py [--force] [part …]` | `--force`: re-render everything; parts: only these |
| `.venv/bin/python tools/check.py [--update] [--no-renders]` | `--update`: replace committed STLs by the CAD exports; `--no-renders`: skip the render check |
| `.venv/bin/python tools/print_records.py` | settings of every print record → `print/*.json`; extracts the printed geometry when a variant's STL is missing |
| `.venv/bin/python tools/analyses.py` | needs `renders/index.json` |
| `.venv/bin/python tools/viewer_export.py` | concept meshes only if `analyses/molar_clamp_concepts/stl/` exists (`make concepts`) |
| `.venv/bin/python tools/docs.py` | rewrites the marked sections of this README |
| `.venv/bin/python tools/downscale.py <dir> [MAX] [--quantize GLOB]` | shrink PNGs in place (default MAX 2000 px); `--quantize 'render_*'`: 256-colour palette |
| `freecad.cmd tools/export_variants.py [--pass part …]` | export CAD-backed variants to `build/regen/` |
| `freecad.cmd tools/split_fcstd.py --pass [--force]` | split `source/` into the part files (FreeCAD scripts take their own arguments after `--pass`) |

### Concept study

| Command | What it does |
|---|---|
| `make concepts` | everything below plus shrinking the renders and building the report |
| `PATH=$PWD/.venv/bin:$PATH bash analyses/molar_clamp_concepts/scripts/run_all.sh` | scripts `s01`–`s10`, then the report (latexmk) |
| `SKIP_LATEX=1 PATH=$PWD/.venv/bin:$PATH bash analyses/molar_clamp_concepts/scripts/run_all.sh` | scripts only |
| `freecad.cmd analyses/molar_clamp_concepts/scripts/s00_export_teeth_freecad.py` | re-export the teeth scan from `source/` (optional) |
| `cd analyses/molar_clamp_concepts/latex && latexmk -pdf` | the report only |

### Papers, checks and tests (from the repository root)

| Command | What it does |
|---|---|
| `biomech export --config experiments/icorr2027/study.yaml --paper experiments/icorr2027/paper_dataset.yaml [--target journal] [--gallery]` | dataset paper incl. `hw_parts`, `hw_table` (journal target); `--gallery`: every form, plus the part renders and figures in the gallery's image row |
| `biomech export --config experiments/jtsmm_validation/study.yaml --paper experiments/jtsmm_validation/paper_systems.yaml [--gallery]` | systems paper incl. `val_hw_parts`, `val_hw_table` (`parts`, `crosscheck`) |
| `biomech check-outputs --config experiments/icorr2027/study.yaml` | includes the hardware checks (manifest valid; part versions specified) |
| `pytest -q software/biomech/tests/test_hwparts.py` | manifest, renders, CAD bodies, usage and candidates, titles vs the reader text, every form builds and renders |
| `pytest -q tests/test_docs_links.py` | links and anchors of the docs, this README included |

## Resolution and size

Part renders are 2000 × 1600 px, figures embed their images at 300 dpi (PDF, SVG) and are written
as 300-dpi PNGs, the concept renders keep their native 1600 × 1230 px; all 3D renders are stored as
256-colour palette PNGs (no visible loss on the shaded renders, about six times smaller). In the
papers, image panels are capped at 2100 px (text width at 300 dpi). The folder adds about 28 MB to
the repository (compressed); the 45 MB voxel concept STLs, `build/`, `.venv/` and `source/` are
git-ignored.

## Workflows

**Edit a part.** Open `parts/<part>/<part>.FCStd` in FreeCAD, change it (dimensions live in the
`Params*` spreadsheets), save. Then `make parts UPDATE=1 renders analyses viewer docs`. A new
version is better a new body (copy the body) plus a new variant in `parts.yaml`, so the printed
version stays reproducible.

**Add a variant.** Add it under the part's `variants` with `stl:` (where the STL goes),
`fcstd_body:` and, for a parametric variant, `params: {<sheet label>.<alias>: value}` (as the
version-5 variants: `Params.ta_separation`). `make parts` exports and installs it, `make renders`
renders it. An STL-only variant needs only `stl:` and `cad_pending: true`. Give it a `title`
like "Version 6", not "V6".

**Add a CAD model for an STL-only (*CAD pending*) variant.**
1. Save the model as `parts/<part>/<part>.FCStd` (or add a body to the existing one) and set
   `fcstd_body:` of the variant.
2. `make parts`: the export must match the committed STL within 0.01 mm, so a stale model never
   replaces the printed geometry silently; on a mismatch, either fix the model or (if the model
   is the newer truth) re-run with `UPDATE=1`.
3. Remove `cad_pending: true`.

**Add a print record.** Save the slicer project as `print/<part>_<variant>_<date>.3mf` and set
`print:` of the variant; `make parts` writes its settings (printer, filament, layer height,
infill, walls, supports) to `print/*.json`, which the printed-mass estimate uses.

**Seat the teeth attachment on the scan.** The scan and the attachment are not in one frame.
Place the attachment on `Teeth_02` in `teeth_attachment.FCStd` and copy its placement to
`parts.teeth_attachment.seat_on_scan` (`[x, y, z, yaw, pitch, roll]`, mm and degrees); a fit
map (contact and gap per variant) can then be added to `tools/analyses.py` (not implemented yet).

## Provenance

Imported on 2026-10-29 from a CAD working folder outside the repository (unchanged there):
`JTS_Models_simple_aruco.FCStd` (FreeCAD 1.1, last saved 2026-10-09; not committed, it is
fully contained in the part files) split into the part files by `tools/split_fcstd.py` (every
body recomputes to the source volume and box; the parameter sheets were cleared of ~32 000
formatted empty cells each); the exported STLs, the 3MF records, the incisor analysis figures and
the molar-clamp concept study (superseded first-draft images, duplicates and the 45 MB voxel STLs
left out). The original JTS parts stay in
[`software/jaw_tracking/models/`](../../software/jaw_tracking/models/) and are listed here as the
`jts` variants.
