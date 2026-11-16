# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD21 sensitivity: the hinge fits (``ref_hinge``) with and without the reference-quality
gate (V8 reports results with and without it; read-only on a stored validation snapshot).

``ref_agreement`` keeps a cycle only when no frame of it is gated (V8: a head- or jaw-plate
marker missing or gap-filled, an inter-marker distance off its median by more than
0.5 mm, or a marker residual above 1 mm). Here the cycles are found again with the same
rule (``agreement.find_cycles``) on the same held-out movement intervals, once with and
once without the gate, and both sets go through the pipeline's hinge fit
(``hinge.run_ref_hinge``). The gated rows reproduce the stored ``ref_hinge`` table.

    python hinge_ungated.py [--snapshot v10] [--tasks open_close chewing]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

from biomech.config import load_config
from biomech.io.context import DataContext
from biomech.validation import selection as SEL
from biomech.validation.agreement import find_cycles
from biomech.validation.hinge import run_ref_hinge
from biomech.validation.stages import DEFAULTS

STUDY = "experiments/jtsmm_validation/study.yaml"


def hinge_rows(ctx: DataContext, pid: str, tasks: list[str], gated: bool) -> pd.DataFrame:
    arr = ctx.arrays("ref_kinematics", pid)
    t = np.asarray(arr["t_master_s"], float)
    fs = 1.0 / float(np.median(np.diff(t)))
    held = SEL.movement_intervals(ctx.phases(pid), t, np.asarray(arr["ct_visible"], bool),
                                  trim_s=DEFAULTS["trim_s"])
    use = arr if gated else {k: v for k, v in arr.items() if k != "gate"}
    rows = [{"task": h.name, "start": a, "peak": pk, "end": b}
            for h in held if h.name in tasks
            for _, a, pk, b in find_cycles(use, h.name, (t >= h.t0) & (t <= h.t1), fs)]
    cyc = pd.DataFrame(rows, columns=["task", "start", "peak", "end"])
    h = run_ref_hinge(arr, cyc, pid)["hinge"]
    return h[h.task.isin(tasks)].assign(gate="with gate" if gated else "without gate")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    ap.add_argument("--tasks", nargs="+", default=["open_close", "chewing"])
    a = ap.parse_args()
    ctx = DataContext(load_config(STUDY).with_snapshot(a.snapshot))
    out = pd.concat([hinge_rows(ctx, pid, a.tasks, g) for pid in ctx.pids()
                     for g in (True, False)], ignore_index=True)
    out["pid"] = out["pid"].map(lambda p: f"P{ctx.pseudonym(p)}")
    cols = ["pid", "task", "gate", "system", "n_openings", "hinge_fit_mm", "hinge_tmj_mm",
            "axis_to_condyles_mm"]
    print(out[cols].sort_values(["pid", "task", "gate", "system"]).to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
