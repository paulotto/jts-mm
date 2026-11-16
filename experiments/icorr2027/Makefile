# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# ICORR 2027 study targets. Override the interpreter with `make BIOMECH=...`, the
# session root with `make DATA=...`, parallelism with `make JOBS=...`; `make video PID=s03
# LAYOUT=kinematics PHASE=open_close SPEED=0.25` renders another clip (docs/analysis/an11).
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO    := $(abspath $(HERE)../..)
# `biomech` on PATH (activated venv) or the repo venv, so `make` works without activating.
BIOMECH ?= $(or $(shell command -v biomech 2>/dev/null),$(wildcard $(REPO)/software/jtsmm/.venv/bin/biomech),$(error biomech not found: activate the venv or pass BIOMECH=/path/to/biomech))
JOBS    ?= 6
CFG     := --config $(HERE)study.yaml $(if $(DATA),--data-dir $(DATA))
PID     ?= s01
LAYOUT  ?= multimodal
PHASE   ?= chewing
SPEED   ?= 0.5

.PHONY: all check qc paper paper-journal paper-conference paper-camerafree gallery verify sync-paper sync-paper-icorr sync-paper-additional export-all view status check-all video

all: check
	$(BIOMECH) run $(CFG) -j $(JOBS)

check:
	$(BIOMECH) cohort check $(CFG)

qc: check
	$(BIOMECH) run $(CFG) --stage qc_cohort -j $(JOBS)

paper: paper-journal paper-conference

# The dataset paper set (one paper; the conference cut is the `conference` target);
# --standalone writes one upload PDF per figure (figures/<id>/<id>.pdf).
paper-journal:
	$(BIOMECH) export $(CFG) --paper $(HERE)paper_dataset.yaml --target journal --compile --check-size --standalone

paper-conference:
	$(BIOMECH) export $(CFG) --paper $(HERE)paper_dataset.yaml --target conference --compile --check-size --standalone

# The ICORR camera-free paper (2026-11-12): X5 in IEEE two-column format
paper-camerafree:
	$(BIOMECH) export $(CFG) --paper $(HERE)paper_camerafree.yaml --target conference --compile --check-size --standalone

# Every depiction of every item of the journal set, to choose the forms from (gallery.html).
gallery:
	$(BIOMECH) export $(CFG) --paper $(HERE)paper_dataset.yaml --target journal --gallery --standalone

verify:
	$(BIOMECH) verify-export $(CFG) $(REPO)/data/analysis/icorr2027/paper_dataset/journal
	$(BIOMECH) verify-export $(CFG) $(REPO)/data/analysis/icorr2027/paper_dataset/conference

# Copy the journal export into the JNER manuscript (MANUSCRIPT/results: numbers, captions,
# figures and tables by paper label; README there) and check the manuscript against it.
MANUSCRIPT ?= $(error set MANUSCRIPT=<folder of the JNER manuscript>)
sync-paper:
	$(BIOMECH) paper-sync $(REPO)/data/analysis/icorr2027/paper_dataset/journal --to $(MANUSCRIPT)/results --manuscript $(MANUSCRIPT)

# the same plus the additional files (interactive HTML, supplementary videos; tens of MB)
# The ICORR camera-free paper (2026-11-12)
ICORR_MANUSCRIPT ?= $(error set ICORR_MANUSCRIPT=<folder of the ICORR manuscript>)
sync-paper-icorr:
	$(BIOMECH) paper-sync $(REPO)/data/analysis/icorr2027/paper_camerafree/conference --to $(ICORR_MANUSCRIPT)/results --manuscript $(ICORR_MANUSCRIPT)

sync-paper-additional:
	$(BIOMECH) paper-sync $(REPO)/data/analysis/icorr2027/paper_dataset/journal --to $(MANUSCRIPT)/results --manuscript $(MANUSCRIPT) --additional

# Every registered view (paper, dashboards, data atlas; all perspectives and participants)
# as TikZ + PNG/PDF previews -> data/analysis/icorr2027/export_all/ (git-ignored).
export-all:
	$(BIOMECH) export $(CFG) --all

view:
	$(BIOMECH) view $(CFG)

status:
	$(BIOMECH) status $(CFG)

check-all:
	$(BIOMECH) check-outputs $(CFG)

# Synchronized video (+ filmstrip) -> data/analysis/icorr2027/videos/ (git-ignored). Camera
# frames show the participant and are never included here (biomech animate --with-camera).
video:
	$(BIOMECH) animate $(CFG) --pid $(PID) --layout $(LAYOUT) --phase $(PHASE) \
		--speed $(SPEED) --fps 30 --size 1080p --filmstrip
