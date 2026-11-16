# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Exploratory fusion diagnostics (biomech stage ``ml_eval``, analysis_plan D13).

* :func:`variance_ratio` -- per feature column, the variance *between*
  participants (of the per-participant means) and the mean variance *within*
  participants, and their ratio. A large ratio means a feature mostly encodes who
  the participant is, which LOSO cannot learn away.
* :func:`norm_views` -- the EMG block before and after the label-free
  rest-relative normalization (D13) for every window of a pooled dataset, whatever
  ``emg_norm`` the windows were built with: stored rest statistics are inverted for
  ``rest_relative`` windows; for ``mvc`` windows the statistics are computed from
  the participant's ``rest``-phase windows in the pool.
* :func:`fusion_diagnostics` -- the two combined into one table.

"Before" is on the scale D13 centres (log for linear amplitude features such as
``mav``; the default features are already log or counts, so it equals the stored
``mvc`` features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config.classes import REST_PHASE
from ml.core import features as F
from ml.core.dataset import WindowDataset


def variance_ratio(X: np.ndarray, pid: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Between- vs within-participant variance per column (participants weighted equally)."""
    X = np.asarray(X, dtype=np.float64)
    pids = np.unique(pid)
    means = np.array([np.nanmean(X[pid == p], axis=0) for p in pids])
    within = np.nanmean(np.array([np.nanvar(X[pid == p], axis=0) for p in pids]), axis=0)
    between = np.nanvar(means, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = between / within
    return pd.DataFrame({"feature": names, "var_between": between, "var_within": within,
                         "ratio": ratio, "icc": between / (between + within)})


def _sources(ds: WindowDataset) -> dict[str, dict]:
    """Build metadata per take (session id), as stored by build_windows / concat."""
    metas = ds.meta.get("sources", [ds.meta])
    return {str(m.get("session")): m for m in metas if isinstance(m, dict) and m.get("session")}


def norm_views(ds: WindowDataset, block: str = "emg") -> tuple[np.ndarray, np.ndarray]:
    """(before, after) D13 views of ``block`` for every row of ``ds`` (per take)."""
    names = ds.names[block]
    X = ds.blocks[block]
    before, after = np.full(X.shape, np.nan), np.full(X.shape, np.nan)
    src = _sources(ds)
    for sid in np.unique(ds.session):
        rows = ds.session == sid
        m = src.get(str(sid), {})
        st = (m.get("norm") or {}).get(block)
        if m.get("emg_norm") == "rest_relative" and st:
            c, s = np.asarray(st["center"]), np.asarray(st["scale"])
            after[rows] = X[rows]
            before[rows] = X[rows] * s + c
        else:
            rest = rows & (ds.phase == REST_PHASE)
            before[rows] = F.to_norm_scale(X[rows], names)
            if rest.sum() >= 2:
                c, s = F.rest_relative_stats(X[rest], names)
                after[rows] = F.apply_rest_relative(X[rows], names, c, s)
    return before, after


def fusion_diagnostics(ds: WindowDataset, block: str = "emg") -> pd.DataFrame:
    """Per feature: between/within-participant variance before and after D13 (all windows)."""
    if block not in ds.blocks or ds.blocks[block].ndim != 2:
        return pd.DataFrame()
    before, after = norm_views(ds, block)
    names = ds.names[block]
    b = variance_ratio(before, ds.pid, names).set_index("feature").add_suffix("_before")
    a = variance_ratio(after, ds.pid, names).set_index("feature").add_suffix("_after")
    out = b.join(a).reset_index()
    out.insert(1, "kind", [F.feature_kind(n) for n in names])
    out.insert(1, "channel", [n.split("|")[0] for n in names])
    out["n_windows"] = len(ds)
    out["n_participants"] = int(np.unique(ds.pid).size)
    return out


__all__ = ["fusion_diagnostics", "norm_views", "variance_ratio"]
