# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD23 (descriptive): EMG side check of the JTS-MM montage against Delsys.

The JTS-MM channel names (the montage in the code, ``io.session.MUSCLES``) and the Delsys
sensors (labelled by side) record the same muscles of the same participant in the same tasks,
on different days (``ref_emg``, cross-session). If the code's sides are right, each JTS-MM
side's mean cycle profile has the shape of the *same* Delsys side; if the montage mirrored
the sides, it has the shape of the *opposite* side. Per participant, profile (task, and side
of the movement in ``left_right``) and muscle pair the four shape correlations

    r(a, b) = profile_r(JTS-MM a, Delsys b),   a, b in {R, L}

of the ensemble-mean profiles (``ref_emg`` arrays, % of each system's own maximum, cycles
with a masked point dropped as in the D5 comparison) give the side index

    SI = ½ [r(R,R) + r(L,L) − r(R,L) − r(L,R)]   in [−2, 2]:

+2 the sides as in the code, −2 swapped, ≈ 0 not identifiable (both sides behave alike, as
in the symmetric tasks and for the co-activating digastrics). Only tasks in which the two
sides act differently are informative (``side_tasks``, by default the laterotrusion
``left_right``). Read from the stored ``ref_emg`` arrays only; nothing is recomputed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from biomech.validation.emg import (PROFILE_PREFIX, PROFILE_SUFFIX, finite_cycles,
                                    profile_array_key, profile_r, split_profile_key)

DEFAULTS: dict[str, Any] = {
    "pairs": ["masseter", "temporalis", "digastric"],   # muscle pairs (<pair>_r, <pair>_l)
    "min_cycles": 2,               # cycles per system and side for an ensemble mean (as D5)
    "side_tasks": ["left_right"],  # tasks in which the two sides act differently
}
SIDES = ("r", "l")
COLUMNS = ["pid", "profile", "task", "side", "muscle", "side_informative", "n_jtsmm_r",
           "n_jtsmm_l", "n_delsys_r", "n_delsys_l", "r_rr", "r_ll", "r_rl", "r_lr", "side_index",
           "reason"]


def side_index(r_rr: float, r_ll: float, r_rl: float, r_lr: float) -> float:
    """½[r(R,R) + r(L,L) − r(R,L) − r(L,R)] with r(a, b) = JTS-MM side a vs Delsys side b:
    +2 sides as in the code, −2 swapped, 0 not identifiable; NaN if any r is NaN."""
    return 0.5 * (r_rr + r_ll - r_rl - r_lr)


def profile_keys(arrays: Mapping[str, Any]) -> list[str]:
    """Profile keys (``task`` or ``task__side``) stored for both systems, sorted."""
    def keys(system: str) -> set[str]:
        pre = f"{PROFILE_PREFIX}{system}__"
        return {k[len(pre):-len(PROFILE_SUFFIX)] for k in arrays
                if k.startswith(pre) and k.endswith(PROFILE_SUFFIX)}
    return sorted(keys("jtsmm") & keys("delsys"))


def _means(Y: np.ndarray, idx: Mapping[str, int], pair: str, min_cycles: int
           ) -> tuple[dict[str, np.ndarray | None], dict[str, int]]:
    """Ensemble-mean profile and cycle count per side of one system (None: too few)."""
    mean: dict[str, np.ndarray | None] = {}
    n: dict[str, int] = {}
    for s in SIDES:
        y = finite_cycles(np.asarray(Y, float)[:, :, idx[f"{pair}_{s}"]])
        n[s] = int(len(y))
        mean[s] = y.mean(axis=0) if len(y) >= min_cycles else None
    return mean, n


def side_check(arrays: Mapping[str, Any], pid: str, params: Mapping[str, Any] | None = None
               ) -> pd.DataFrame:
    """Table ``side_check``: one row per profile key and muscle pair (columns
    :data:`COLUMNS`); ``reason`` names why the index is NaN (no channel, too few cycles,
    constant profile), empty otherwise."""
    p = {**DEFAULTS, **(params or {})}
    min_cycles = int(p["min_cycles"])
    side_tasks = set(p["side_tasks"])
    names = [str(n) for n in arrays.get("names", [])]
    idx = {n: i for i, n in enumerate(names)}
    rows: list[dict[str, Any]] = []
    for key in profile_keys(arrays) if names else []:
        task, side = split_profile_key(key)
        A, B = arrays[profile_array_key("jtsmm", key)], arrays[profile_array_key("delsys", key)]
        for pair in p["pairs"]:
            row: dict[str, Any] = {"pid": pid, "profile": key, "task": task, "side": side,
                                   "muscle": pair, "side_informative": task in side_tasks,
                                   "reason": ""}
            if any(f"{pair}_{s}" not in idx for s in SIDES):
                rows.append({**row, "side_index": np.nan, "reason": "no channel"})
                continue
            ma, na = _means(A, idx, pair, min_cycles)
            mb, nb = _means(B, idx, pair, min_cycles)
            row.update({f"n_jtsmm_{s}": na[s] for s in SIDES})
            row.update({f"n_delsys_{s}": nb[s] for s in SIDES})
            if any(m is None for m in (*ma.values(), *mb.values())):
                rows.append({**row, "side_index": np.nan, "reason": "too few cycles"})
                continue
            r = {f"r_{a}{b}": profile_r(ma[a], mb[b]) for a in SIDES for b in SIDES}
            si = side_index(r["r_rr"], r["r_ll"], r["r_rl"], r["r_lr"])
            rows.append({**row, **r, "side_index": si,
                         "reason": "" if np.isfinite(si) else "constant profile"})
    return pd.DataFrame(rows, columns=COLUMNS)


def run_ref_emg_side(arrays: Mapping[str, Any], pid: str,
                     params: Mapping[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """The tables of ``ref_emg_side`` from the participant's ``ref_emg`` arrays."""
    return {"side_check": side_check(arrays, pid, params)}

