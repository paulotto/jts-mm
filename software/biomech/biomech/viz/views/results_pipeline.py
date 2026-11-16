# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Which participants each cohort statistic is based on (pipeline fail policy, an08 §1).

Every cohort stage stores a ``coverage`` table: the included participants it used and the
ones it left out with the reason (failed or skipped upstream, missing input). A participant
missing from a statistic must be visible next to the statistic.
"""

from __future__ import annotations

import numpy as np

from biomech.viz.spec import Column, FigureSpec, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register


def _coverage(ctx):
    import pandas as pd

    from biomech.pipeline import registry

    rows = []
    for name, st in registry().items():
        if st.scope != "cohort" or st.group not in ctx.config.groups:
            continue
        df = ctx.table(name, "coverage")
        if df.empty:
            continue
        rows.append(df.assign(stage=name))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["stage", "pid", "used", "reason"])


@register("pipeline_coverage", "dashboard", "Participants per cohort statistic", domain="qc")
def pipeline_coverage(req: ViewRequest) -> FigureSpec | TableSpec:
    """Heatmap stage × participant: used (1) or left out (0), with the reasons as a table."""
    ctx = req.ctx
    cov = _coverage(ctx)
    pids = ctx.pids()
    if cov.empty:
        return TableSpec("pipeline_coverage", "No cohort stage has stored coverage yet.",
                         [Column("note", "Note", align="l")], [{"note": "run the pipeline"}])
    cov = cov.assign(used=cov.used.astype(float))
    stages = sorted(cov.stage.unique())
    p = B.matrix_heat(cov, "stage", "pid", "used", "Used in the statistic (1) / left out (0)",
                      rows=stages, cols=pids, row_labels=[V.term("stage", s) for s in stages],
                      col_labels=B.pseudo_rows(ctx, pids), fmt="{:.0f}")
    out = cov[cov.used < 1]
    lab = {pid: f"P{ctx.pseudonym(pid)}" for pid in pids}
    note = "; ".join(f"{V.term('stage', r.stage)}: {lab.get(r.pid, 'excluded')} ({r.reason})"
                     for r in out.itertuples()) or "every included participant used everywhere"
    return FigureSpec("pipeline_coverage", "Participants per cohort statistic", [p],
                      width="text", height_in=max(1.4, 0.18 * len(stages)),
                      caption="Which included participants each cohort stage used. A "
                              "participant is left out only when an upstream unit failed or "
                              "its input is missing (never because it was not run). "
                              f"{int(np.sum(cov.used < 1))} left out: " + note + ".",
                      sources=["*:coverage"])


@register("labels_rule_agreement", "dashboard", "Intent labels: axis-scale floor vs previous rule",
          domain="ml")
def labels_rule_agreement(req: ViewRequest) -> FigureSpec:
    """Share of label samples that the axis-scale floor (D16) changed, per phase."""
    from biomech.viz.spec import Axis

    a = req.ctx.table("labels", "label_agreement")
    tasks = [t for t in V.TASKS if not a.empty and (a.phase == t).any()]
    p1 = B.strip([V.task_label(t) for t in tasks],
                 [100 * a[a.phase == t].changed_frac.to_numpy(float) for t in tasks],
                 "Label samples changed", Axis("Changed", V.PCT))
    p2 = B.strip([V.task_label(t) for t in tasks],
                 [a[a.phase == t].kappa.to_numpy(float) for t in tasks],
                 "Agreement with the previous rule", Axis("Cohen's κ", lim=(0.0, 1.0)))
    return FigureSpec("labels_rule_agreement", "Intent labels: axis-scale floor vs previous rule",
                      [p1, p2], rows=1, cols=2, width="text",
                      height_in=max(1.3, 0.2 * len(tasks)),
                      caption="Per participant and phase: share of labelled samples whose "
                              "class changed when each free-motion axis scale is floored at "
                              "0.25 × the largest axis scale (post hoc rule), and Cohen's κ between the "
                              "two labellings. Single-axis phases are unchanged by "
                              "construction.", sources=["labels:label_agreement"])
