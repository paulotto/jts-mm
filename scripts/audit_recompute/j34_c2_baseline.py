# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J34.2: how far the in-task baseline of the openers sits above the rest threshold
(descriptive, post hoc; definitions fixed in J34 before this ran).

Per participant of the paper store (pin r16a, reporting_population) the timing stage's own
``lead.analyse`` runs on the participant's EMG prep, IMU arrays and kinematic cycles (the D23
null, X2 and X3 switched off); ``WindowOnsets.__call__`` is wrapped so that for every opening
window and opener channel the TKEO detector signal of the window is read with the stage's own
threshold:

* ``active``: median over the first 50 ms above the rest threshold (the start gate);
* ``taskbase_db``: 10·log10 of the median over the first ``TASKBASE_S`` (0.3 s) over the rest
  threshold (TKEO is an energy operator, so this is the 20·log10 amplitude scale of
  ``burst_db``).

Reproduction first: ``active`` must equal the stored ``active_at_start`` (TKEO, rule first) of
every opener channel-cycle, and the recomputed onsets must equal the stored ones (exactly;
``burst_db`` to 1e-9); the script stops otherwise.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j34_c2_baseline.py [-j 8]

Output: ``<adults store>/audit/j34_c2_baseline_<snap>.parquet`` (one row per opener
channel-cycle) and ``.json`` (settings, reproduction, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
GATE_S = 0.05                 # the start gate's window (lead.WindowOnsets, hard-coded there)
KEY = ["task", "cycle", "muscle"]


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def participant(args: tuple[str, str]) -> dict[str, Any]:
    snapshot, pid = args
    from biomech.io.context import DataContext
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.timing import lead
    from jtsmm.emg.onset import detector_signal

    cfg = _config(snapshot)
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    rec: list[dict[str, Any]] = []
    orig_call = lead.WindowOnsets.__call__
    saved = (lead.analyse_ensemble, lead.analyse_release)

    def call(self, t_mech, tw, sel, x, xc_all, cleaned, mask, group):
        for out in orig_call(self, t_mech, tw, sel, x, xc_all, cleaned, mask, group):
            m, d, rule = out[0], out[1], out[2]
            if d == "tkeo" and rule == "first" and m.startswith("digastric"):
                i = self.names.index(m)
                sig = detector_signal(np.nan_to_num(x[sel, i]), self.fs, self.params[d])
                thr = float(self.thr[(d, i)])
                n0 = max(int(GATE_S * self.fs), 1)
                n_tb = max(int(lead.TASKBASE_S * self.fs), 1)
                med_tb = float(np.median(sig[:n_tb]))
                rec.append({"t_mech_us": float(t_mech), "muscle": m,
                            "active": bool(np.median(sig[:n0]) > thr), "thr": thr,
                            "median_taskbase": med_tb,
                            "taskbase_db": float(10 * np.log10(med_tb / thr))
                            if thr > 0 and med_tb > 0 else np.nan})
            yield out

    try:
        lead.WindowOnsets.__call__ = call
        lead.analyse_ensemble = lambda *a, **k: ([], {})     # X2: not needed here
        lead.analyse_release = lambda *a, **k: ([], [])      # X3: not needed here
        pc = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("timing"),
                                store=dc.store, participant=part,
                                cache=UnitCache(dc.cohort, part))
        tables, _ = lead.analyse(pc.emg_prep(), pid, pc.upstream_arrays("imu"),
                                 pc.upstream("kinematics", "cycles"), pc.params, pc.seed,
                                 n_null=0)
        stored = dc.table("timing", "onsets", [pid])
    finally:
        lead.WindowOnsets.__call__ = orig_call
        lead.analyse_ensemble, lead.analyse_release = saved
        dc.close()
    sel = lambda o: o[(o.detector == "tkeo") & (o.rule == "first")      # noqa: E731
                      & (o.direction == "opening") & o.muscle.str.startswith("digastric")]
    new, old = sel(tables["onsets"]), sel(stored)
    cols = KEY + ["t_mech_us", "t_onset_us", "active_at_start", "usable", "burst_db"]
    a = new[cols].sort_values(KEY).reset_index(drop=True)
    b = old[cols].sort_values(KEY).reset_index(drop=True)
    # Exact for every column but burst_db (a log of an RMS peak: equal to ~1e-11 dB, the
    # summation order of the recomputed envelope).
    exact = [c for c in cols if c != "burst_db"]
    onsets_equal = bool(len(a) == len(b) and a[exact].equals(b[exact])
                        and np.allclose(a.burst_db, b.burst_db, rtol=1e-9, atol=1e-9,
                                        equal_nan=True))
    r = pd.DataFrame(rec).merge(a[KEY + ["t_mech_us", "active_at_start", "usable"]],
                                on=["t_mech_us", "muscle"], how="right", validate="1:1")
    r.insert(0, "pid", pid)
    return {"rows": r, "onsets_equal": onsets_equal,
            "gate_equal": bool((r.active == r.active_at_start.astype(bool)).all()),
            "n": int(len(r))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("-j", "--jobs", type=int, default=8)
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        pids = list(dc.pids())
    finally:
        dc.close()
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        res = dict(zip(pids, ex.map(participant, [(a.snapshot, q) for q in pids])))
    repro = {q: {k: v[k] for k in ("onsets_equal", "gate_equal", "n")} for q, v in res.items()}
    if not all(v["onsets_equal"] and v["gate_equal"] for v in repro.values()):
        raise SystemExit(f"reproduction failed: {json.dumps(repro, indent=1)}")
    out = pd.concat([v["rows"] for v in res.values()], ignore_index=True)
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"j34_c2_baseline_{a.snapshot}.parquet"
    out.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    from biomech.timing.lead import TASKBASE_S

    meta = {"snapshot": a.snapshot, "git_sha": sha, "variant": VARIANT, "gate_s": GATE_S,
            "taskbase_s": TASKBASE_S, "n_channel_cycles": int(len(out)),
            "reproduction": repro, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    per = out.groupby("pid").taskbase_db.median()
    print(per.round(1).to_string())
    print(f"channel-cycles {len(out)}, active {int(out.active.sum())}; per-participant median "
          f"{per.median():.1f} dB (range {per.min():.1f} to {per.max():.1f})")
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
