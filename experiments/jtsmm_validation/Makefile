# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# System validation study (systems paper) targets. Override the interpreter with
# `make BIOMECH=...`, parallelism with `make JOBS=...` (docs/analysis/an14).
HERE    := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
REPO    := $(abspath $(HERE)../..)
BIOMECH ?= $(or $(shell command -v biomech 2>/dev/null),$(wildcard $(REPO)/software/jtsmm/.venv/bin/biomech),$(error biomech not found: activate the venv or pass BIOMECH=/path/to/biomech))
JOBS    ?= 6
CFG     := --config $(HERE)study.yaml $(if $(DATA),--data-dir $(DATA))
PAPER   := --paper $(HERE)paper_systems.yaml

.PHONY: all paper gallery verify sync-paper sync-paper-additional check-all status integrity-status

all:
	$(BIOMECH) run $(CFG) -j $(JOBS)

# The systems paper set from its pinned snapshot: preview.pdf with captions, the journal
# upload PDF per figure (figures/<id>/<id>.pdf) and the captioned PDF per item (captioned/).
paper:
	$(BIOMECH) export $(CFG) $(PAPER) --compile --check-size --standalone

# Every depiction of every item, to choose the forms from (gallery.html).
gallery:
	$(BIOMECH) export $(CFG) $(PAPER) --gallery --standalone

verify:
	$(BIOMECH) verify-export $(CFG) $(REPO)/data/analysis/jtsmm_validation/paper_systems

# Copy the export into the IEEE Sensors manuscript (MANUSCRIPT/results; README there) and check
# the manuscript against it.
MANUSCRIPT ?= $(error set MANUSCRIPT=<folder of the systems-paper manuscript>)
sync-paper:
	$(BIOMECH) paper-sync $(REPO)/data/analysis/jtsmm_validation/paper_systems --to $(MANUSCRIPT)/results --manuscript $(MANUSCRIPT)

# the same plus the additional files (interactive HTML, supplementary videos; tens of MB)
sync-paper-additional:
	$(BIOMECH) paper-sync $(REPO)/data/analysis/jtsmm_validation/paper_systems --to $(MANUSCRIPT)/results --manuscript $(MANUSCRIPT) --additional

check-all:
	$(BIOMECH) check-outputs $(CFG)

status:
	$(BIOMECH) status $(CFG)
