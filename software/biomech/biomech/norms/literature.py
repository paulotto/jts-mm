# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Comparison of participant values with literature reference values (``norms.yaml``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

NORMS_PATH = Path(__file__).with_name("norms.yaml")


@lru_cache(maxsize=2)
def load(path: str | Path = NORMS_PATH) -> dict[str, list[dict[str, Any]]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _ref(entries: list[dict[str, Any]], sex: str) -> dict[str, Any] | None:
    for want in (sex, "all"):
        for e in entries:
            if e.get("sex") == want:
                return e
    return None


def compare(subj: pd.DataFrame, norms: dict[str, list[dict[str, Any]]] | None = None
            ) -> pd.DataFrame:
    """Per participant and metric: difference to the matching reference, z if an SD exists."""
    norms = load() if norms is None else norms
    rows = []
    for metric, entries in norms.items():
        if metric not in subj:
            continue
        for _, r in subj.iterrows():
            ref = _ref(entries, r.get("sex", "all"))
            if ref is None or not np.isfinite(r[metric]):
                continue
            diff = float(r[metric] - ref["mean"])
            sd = ref.get("sd")
            rows.append({"pid": r["pid"], "metric": metric, "value": float(r[metric]),
                         "ref_mean": ref["mean"], "ref_sd": sd, "ref_sex": ref["sex"],
                         "diff": diff, "z": diff / sd if sd else np.nan,
                         "source": ref["source"]})
    return pd.DataFrame(rows)
