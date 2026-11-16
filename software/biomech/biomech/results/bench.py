# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stage timing of a run from its event log (``biomech bench``)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def run_timing(events: list[dict[str, Any]]) -> tuple[pd.DataFrame, float]:
    """Per stage: units run, summed and maximal unit time, and the stage's wall span
    (first ``stage_start`` to last ``unit_done``); plus the run's wall time."""
    done = [e for e in events if e.get("kind") == "unit_done"]
    wall = (max(e["t"] for e in events) - min(e["t"] for e in events)) if events else 0.0
    if not done:
        return pd.DataFrame(), wall
    d = pd.DataFrame([{"stage": e["stage"], "unit": e["unit"], "status": e["status"],
                       "elapsed_s": float(e["elapsed_s"] or 0.0), "t": e["t"]} for e in done])
    start = {e["stage"]: e["t"] for e in reversed(events) if e.get("kind") == "stage_start"}
    g = d.groupby("stage", sort=False)
    out = pd.DataFrame({"n_units": g.size(), "sum_s": g.elapsed_s.sum(),
                        "max_s": g.elapsed_s.max(),
                        "n_failed": g.status.apply(lambda s: int((s == "failed").sum()))})
    out["span_s"] = [g.t.max()[k] - start.get(k, g.t.min()[k] - out.loc[k, "max_s"])
                     for k in out.index]
    out["share"] = out.sum_s / out.sum_s.sum()
    return out.reset_index().sort_values("sum_s", ascending=False, ignore_index=True), wall
