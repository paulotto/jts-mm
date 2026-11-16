# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The snapshots the papers pin, as defaults of the audit scripts (J1.4): the dataset
analysis of record (``record_snapshot`` of paper_dataset.yaml, primary store), the dataset
paper store (its ``snapshot``, reporting_population) and the systems paper store (paper_systems.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"


def _paper(rel: str) -> dict:
    return yaml.safe_load((EXP / rel).read_text(encoding="utf-8")) or {}


def dataset_record() -> str:
    """The dataset analysis of record (primary store, n = 14)."""
    return str(_paper("icorr2027/paper_dataset.yaml")["record_snapshot"])


def dataset_paper() -> str:
    """The dataset paper store (reporting_population variant)."""
    return str(_paper("icorr2027/paper_dataset.yaml")["snapshot"])


def systems() -> str:
    """The systems paper store."""
    return str(_paper("jtsmm_validation/paper_systems.yaml")["snapshot"])
