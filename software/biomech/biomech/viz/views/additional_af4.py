# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""JNER Additional file 4 tables: data completeness and exclusions per
participant, recording and analysis, with the per-recording IMU sample integrity.

``jnl_table_completeness`` (label ``suppTabCompleteness``; the export also writes it as CSV for
the repository): one column per participant (pseudonym order), rows in two blocks:

* **Recording:** what was recorded (optical tracking, jaw and head IMU, six EMG channels, MVC
  take), EMG channels flagged by the quality checks (``qc_cohort:emg_flags``), and the IMU
  sample integrity of the take (``qc:imu_intervals``): samples received out of order (the
  quaternion streams of the jaw and the head IMU; sorted by their hardware time stamp before
  use) and the longest sample interval of any IMU stream;
* **Analysis:** whether the participant enters each outcome (the rows of the outcome-n figure,
  ``kin_r8._outcome_rows``): used, used with a flag shown, or no usable value.

Flag, don't exclude: a participant leaves an outcome only without a usable value. The one
participant recorded outside the protocol is excluded from every analysis and not listed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Column, TableSpec
from biomech.viz.views.registry import ViewRequest, register

MARK = {"used": "+", "flagged": "(+)", "none": "–"}


def _t(ctx: Any, stage: str, table: str) -> pd.DataFrame:
    try:
        return ctx.table(stage, table)
    except Exception:  # noqa: BLE001 -- a store without that stage
        return pd.DataFrame()


def completeness_rows(ctx: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """(participant columns, rows) of the completeness table; a row holds ``block``, ``item``
    and one cell per participant key ``p<k>``."""
    from biomech.viz.views.kin_r8 import _outcome_rows

    pids = list(ctx.pids())
    keys = [f"p{k}" for k in range(len(pids))]
    rows: list[dict[str, Any]] = []

    def row(block: str, item: str, cells: dict[str, Any]) -> None:
        rows.append({"block": block, "item": item,
                     **{k: cells.get(p, "") for k, p in zip(keys, pids)}})

    qc_emg = _t(ctx, "qc", "emg")
    have_emg = set(qc_emg.pid) if not qc_emg.empty else set()
    ivl = _t(ctx, "qc", "imu_intervals")
    have_imu = set(ivl.pid) if not ivl.empty else set()
    kin = _t(ctx, "kinematics", "cycles")
    have_opt = set(kin.pid) if not kin.empty else set()
    row("Recording", "Optical tracking", {p: "+" if p in have_opt else "–" for p in pids})
    row("Recording", "Jaw and head IMU", {p: "+" if p in have_imu else "–" for p in pids})
    row("Recording", "EMG, six channels", {p: "+" if p in have_emg else "–" for p in pids})
    mvc = {}
    for p in pids:
        try:
            mvc[p] = "+" if ctx.cohort.get(p).mvc else "–"
        except Exception:  # noqa: BLE001
            mvc[p] = ""
    row("Recording", "MVC take", mvc)
    fl = _t(ctx, "qc_cohort", "emg_flags")
    nfl = fl.groupby("pid").size() if not fl.empty else pd.Series(dtype=int)
    row("Recording", "EMG channels flagged (of 6)",
        {p: f"{int(nfl.get(p, 0))}" if p in have_emg else "" for p in pids})
    if not ivl.empty:
        q = ivl[ivl.stream.astype(str).str.endswith("quat")]
        for prefix, lab in (("MTA", "jaw"), ("CRA", "head")):
            g = q[q.stream.astype(str).str.startswith(prefix)].set_index("pid")
            row("Recording", f"IMU orientation samples, {lab} (thousands)",
                {p: f"{g.loc[p, 'n_samples'] / 1e3:.0f}" if p in g.index else "" for p in pids})
            row("Recording", f"of these out of order, {lab}",
                {p: f"{int(g.loc[p, 'nonmono_n'])}" if p in g.index else "" for p in pids})
        mx = ivl.groupby("pid").interval_max_ms.max()
        row("Recording", "Longest IMU sample interval (ms)",
            {p: f"{mx[p]:.1f}" if p in mx.index else "" for p in pids})
    for grp, lab, st in _outcome_rows(ctx):
        row("Analysis", lab[:1].upper() + lab[1:], {p: MARK.get(st.get(p, "none"), "–")
                                                   for p in pids})
    return pids, rows


@register("jnl_table_completeness", "paper",
          "Data completeness and exclusions per participant", group="dataset")
def jnl_table_completeness(req: ViewRequest) -> TableSpec:
    ctx = req.ctx
    pids, rows = completeness_rows(ctx)
    keys = [f"p{k}" for k in range(len(pids))]
    cols = [Column("item", "", align="l")] + [
        Column(k, f"P{ctx.pseudonym(p)}", align="c", group="Participant")
        for k, p in zip(keys, pids)]
    layout: list[dict[str, Any]] = []
    block = None
    for r in rows:
        if r["block"] != block:
            if block is not None:
                layout.append({"_rule": True})
            layout.append({"_head": True, "item": r["block"]})
            block = r["block"]
        layout.append({k: v for k, v in r.items() if k != "block"})
    # "–:" with the colon: a dash between spaces is read as punctuation and replaced by a
    # comma in the reader text, which dropped the symbol from the note (2026-11-13)
    note = ("+: used. (+): used, with a flag shown (implausible condyle points, or a "
            "noise-limited band). --: no usable value. IMU samples out of order were sorted by "
            "their hardware time stamp before use. One participant recorded outside the protocol "
            "was excluded from every analysis and is not listed.")
    t = TableSpec("jnl_table_completeness", "", cols, layout, notes=note, width="text",
                  sources=["qc:emg", "qc:imu_intervals", "qc_cohort:emg_flags",
                           "kinematics:cycles", "stats_kinematics:subjects",
                           "uncertainty:hinge_noise", "stats_timing:timing_spec",
                           "stats_imu:imu_opening_cycles", "ml_eval:loso"])
    n_used = {r["item"]: sum(1 for k in keys if r.get(k) in ("+", "(+)"))
              for r in rows if r["block"] == "Analysis"}
    full = sum(1 for v in n_used.values() if v == len(pids))
    return attach(t, CaptionSpec(
        "what was recorded per participant, the quality flags, and IMU sample integrity of "
        "each take, and which participant enters each outcome. Every recorded take is analysed; quality is flagged, not excluded, and a "
        "participant leaves an outcome only without a usable value",
        n=f"{len(pids)} participants; {full} of {len(n_used)} outcomes use every participant",
        statistic="per participant", evidence="descriptive"))
