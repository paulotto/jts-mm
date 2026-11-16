# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Power curves of a future confirmatory test of the closer release (J27 / J27a;
planning simulation, 2026-11-09).

(a) open-close (H2 at alpha 0.05), (b) chewing (H3 under Holm over the two tasks): simulated
power against the number of usable participants, resampled from ours, for the combined test
at the observed effect and with the location effect shifted toward the bias margin by 25 % and
50 % of the pool median, and the locking condition alone at the observed effect; reference
lines at 0.8 and 0.9. Stored results only (the J27 audit file through ``viz.derived_dataset``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views.registry import ViewRequest, register

TASKS = (("open_close", "(a) Open-close (H2)"), ("chewing", "(b) Chewing (H3, Holm)"))
CURVES = (("scale", 0.0, "combined", "combined, observed effect", "solid", "group:closers"),
          ("shift_loc", 0.25, "combined", "combined, location shrunk 25 %", "dashed",
           "group:closers"),
          ("shift_loc", 0.5, "combined", "combined, location shrunk 50 %", "dotted",
           "group:closers"),
          ("scale", 0.0, "locking", "locking alone, observed effect", "solid", "context"))


@register("jnl_h2_power", "paper", "Sample size for a confirmatory release test",
          group="dataset")
def jnl_h2_power(req: ViewRequest) -> Any:
    from biomech.viz.derived_dataset import _j27_file

    ctx = req.ctx
    try:
        d, meta = _j27_file(ctx)
    except FileNotFoundError:
        d, meta = None, {}
    panels = []
    for k, (task, title) in enumerate(TASKS):
        p = Panel(title, Axis("Usable participants", "", lim=(5.0, 31.0)),
                  Axis("Power", "", lim=(0.0, 1.02)), [], legend="below" if k == 0 else None)
        if d is not None:
            for var, s, test, lab, dash, role in CURVES:
                g = d[(d.task == task) & (d.variant == var) & np.isclose(d.shrink, s)
                      & (d.test == test)].sort_values("n")
                if len(g):
                    p.series.append(Series("line", g["n"].to_numpy(float),
                                           g["power"].to_numpy(float), role=role,
                                           style={"width": 1.0, "dash": dash},
                                           label=lab if k == 0 else None))
        for lv in (0.8, 0.9):
            p.series.append(Series("hline", y=np.array([lv]), role="reference"))
        if k:
            p.y.ticklabels = None
        p.sharex = "n"
        panels.append(p)
    spec = FigureSpec("jnl_h2_power", "Sample size for a confirmatory release test", panels,
                      rows=1, cols=2, width="text", height_in=2.2,
                      sources=["derived:h2_power", "timing:c2e_null"])
    pools = meta.get("pools", {})
    n_txt = (f"{meta.get('n_rep', 0):,} simulated studies per n and task, resampled from "
             f"{pools.get('open_close', {}).get('n', 0)} (open-close) and "
             f"{pools.get('chewing', {}).get('n', 0)} (chewing) usable participants "
             f"(at least {meta.get('min_usable', 0)} usable channel-cycles each)").replace(",", " ")
    return attach(spec, CaptionSpec(
        "simulated power of a future confirmatory test of the closer release before the "
        "opening, as the release test is implemented (locking of the release depth against "
        "200 circular-shift surrogates and release lead beyond the bias margin, one-sided "
        "Wilcoxon signed-rank tests over participants, combined p the larger of the two), "
        "against the number of usable participants; participants drawn with replacement from "
        "ours. The shrunk curves move every participant's release lead toward the margin by "
        "25 % or 50 % of the median effect",
        [("solid blue", "combined test, observed effect"),
         ("dashed / dotted blue", "location effect shrunk by 25 % / 50 %"),
         ("grey", "locking condition alone"),
         ("horizontal lines", "power 0.8 and 0.9")],
        n_txt, "power = share of simulated studies with a rejection",
        "open-close at alpha 0.05; chewing under Holm over the two tasks",
        "planning simulation (post hoc); the release hypothesis stays exploratory in this "
        "study", codes=["J27"]))
