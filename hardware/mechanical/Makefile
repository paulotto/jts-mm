# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# hardware/mechanical: regenerate everything derived from parts.yaml and the part files.
#   make venv      once: tools venv (.venv) with tools/requirements.txt
#   make all       parts + renders + analyses + viewer + docs + check (a second run changes nothing)
#   make parts     CAD -> STL for every CAD-backed variant, checked against the committed STLs
#                  (UPDATE=1: replace committed STLs after an intentional CAD edit; needs FreeCAD)
#   make renders | analyses | viewer | docs | check    the single steps of `make all`
#   make concepts  re-run the molar-clamp concept study (~3 min) and rebuild its report
#   make split     re-create parts/*/*.FCStd from source/ (FORCE=1 overwrites existing part files)
#   make serve     serve the repo on http://127.0.0.1:8072 (3D viewer: /hardware/mechanical/viewer/)
#   make clean     remove build/
# Every target and tool script: README.md "Commands".

PY      ?= .venv/bin/python
FREECAD ?= freecad.cmd
UPDATE  ?=
FORCE   ?=

.PHONY: all venv split parts renders analyses viewer docs check concepts serve clean

all: parts renders analyses viewer docs check

venv: .venv/.ok
.venv/.ok: tools/requirements.txt
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r tools/requirements.txt
	touch $@

split:
	$(FREECAD) tools/split_fcstd.py --pass $(if $(FORCE),--force)

parts:
	$(PY) tools/print_records.py
	rm -rf build/regen
	$(FREECAD) tools/export_variants.py > build/export.log 2>&1 || (grep -a "FAILED\|export:" build/export.log; exit 1)
	@grep -aoE '[a-z_]+/[a-z0-9_.-]+: (warning|FAILED).*|export: .*' build/export.log || true
	$(PY) tools/check.py --no-renders $(if $(UPDATE),--update)

renders:
	$(PY) tools/render.py

analyses:
	$(PY) tools/analyses.py

viewer:
	$(PY) tools/viewer_export.py

docs:
	$(PY) tools/docs.py

check:
	$(PY) tools/check.py

concepts:
	cd analyses/molar_clamp_concepts/scripts && PATH=$(CURDIR)/.venv/bin:$$PATH SKIP_LATEX=1 bash run_all.sh
	$(PY) tools/downscale.py analyses/molar_clamp_concepts/latex/figures/png 2000 --quantize 'render_*'
	cd analyses/molar_clamp_concepts/latex && latexmk -g -pdf -interaction=nonstopmode molar_clamp_concepts.tex > /dev/null

serve:
	cd ../.. && python3 -m http.server 8072 --bind 127.0.0.1

clean:
	rm -rf build
