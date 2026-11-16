# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG activity relevance per task and muscle (analysis plan D19, descriptive).

In the unloaded tasks some muscles stay near rest (the closers during open-close). A cell
(participant x task x muscle) is **active** when the 95th percentile of its normalized,
masked envelope over the task (``emg:activation.p95_pct``, % of the largest activity
recorded) reaches the rest mean + ``k_sd`` SD of the same envelope in the rest phase and at
least ``min_pct``; otherwise it is **near rest** -- marked in figures and tables, never
dropped. The rule was fixed before any value was computed (D19).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def relevance(prep: Any, activation: pd.DataFrame, pid: str, *, k_sd: float = 3.0,
              min_pct: float = 5.0) -> pd.DataFrame:
    """One row per task (not rest) and muscle: rest mean and SD of the normalized envelope
    (unmasked samples), the threshold, the task p95, ``active`` and the reason if not."""
    rest = prep.phase("rest")
    names = list(prep.names)
    rows = []
    for i, m in enumerate(names):
        x = np.asarray(rest.norm[:, i], float)
        keep = np.isfinite(x) & ~np.asarray(rest.mask[:, i], bool)
        mu = float(np.mean(x[keep])) if keep.any() else np.nan
        sd = float(np.std(x[keep], ddof=1)) if keep.sum() > 1 else np.nan
        thr = max(mu + k_sd * sd, min_pct) if np.isfinite(mu) and np.isfinite(sd) else np.nan
        sub = activation[(activation.muscle == m) & (activation.task != "rest")]
        for r in sub.itertuples():
            p95 = float(r.p95_pct)
            if not (np.isfinite(p95) and np.isfinite(thr)):
                active, why = False, "no data"
            elif p95 >= thr:
                active, why = True, ""
            else:
                active = False
                why = "below rest + k SD" if p95 < mu + k_sd * sd else f"below {min_pct:g} %"
            rows.append({"pid": pid, "task": r.task, "muscle": m, "rest_mean_pct": mu,
                         "rest_sd_pct": sd, "threshold_pct": thr, "p95_pct": p95,
                         "active": active, "reason": why})
    return pd.DataFrame(rows)
