#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Regenerate every figure, table and STL, then build the PDF.
# Usage: ./scripts/run_all.sh            (from the project root or anywhere)
set -euo pipefail
cd "$(dirname "$0")"
python3 s01_arch_frame.py
python3 s02_voxelize.py
python3 s03_survey.py
python3 s04_concepts.py
python3 s05_render3d.py
python3 s06_fig_mechanics.py
python3 s07_fig_sections.py
python3 s08_springs.py
python3 s09_drawings.py
python3 s10_export_cad.py
[ -n "${SKIP_LATEX:-}" ] && { echo "done (LaTeX skipped)"; exit 0; }
cd ../latex && latexmk -pdf -interaction=nonstopmode molar_clamp_concepts.tex
echo "done -> latex/molar_clamp_concepts.pdf"
