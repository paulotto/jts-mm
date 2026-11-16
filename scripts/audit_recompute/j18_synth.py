# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J18.6: calibration of the C2 onset and mechanical-onset detectors on synthetic
cycles with a known EMG-to-motion lead, in the participants' real rest EMG (exploratory;
fixed in J18 before this ran).

Per donor participant (the 13 adults of the paper store, pin r16a), synthetic seed
(3 per donor), opener lead L in {0, 20, 50, 100} ms (jitter 20 ms) and level:

* ``testing.synth.timing_arrays`` unchanged (its motion, cycles, bursts and IMU noise);
* EMG = g x the synthetic EMG + the donor's cleaned real rest EMG (per channel the unmasked
  samples, looped from a random offset; the simulator runs at the donor's sampling rate);
* g puts the open-close opener burst (the simulator's 60 uV RMS; chewing openers 40 uV, i.e.
  3.5 dB lower) at the target level re the donor's opener rest envelope median: the cohort
  P25 / P50 / P75 of the opener ``burst_db`` (stored onsets: TKEO, rule first, opening,
  openers, all channel-cycles, both tasks);
* ``timing.lead.analyse`` unchanged (TKEO and the other detectors, rules first and drive, the
  C2 window, its own mechanical-onset detector; ``n_null`` 0, X2 / X3 skipped).

Error per opener channel-cycle = measured lead - true lead (true lead = the cycle start minus
the opener burst start of the simulator's truth; the burst ramps over 20 ms, its 50 % point
sits 10 ms inside).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j18_synth.py [-j 4]

Output: ``<adults store>/audit/j18_synth_<snap>.parquet`` (one row per opener channel-cycle)
and ``j18_synth_<snap>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
LAGS_MS = (0.0, 20.0, 50.0, 100.0)
JITTER_MS = 20.0
N_REP = 3
OPENER_AMP_UV = 60.0                     # simulator's open-close opener burst RMS


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def levels_db(snapshot: str) -> dict[str, float]:
    from biomech.io.context import DataContext
    from biomech.io.session import OPENERS

    dc = DataContext(_config(snapshot))
    try:
        o = dc.table("timing", "onsets")
    finally:
        dc.close()
    v = o[(o.detector == "tkeo") & (o.rule == "first") & (o.direction == "opening")
          & o.muscle.isin(OPENERS)].burst_db.to_numpy(float)
    v = v[np.isfinite(v)]
    q = np.percentile(v, [25, 50, 75])
    return {"P25": float(q[0]), "P50": float(q[1]), "P75": float(q[2])}


def donor(args: tuple[str, str, int, dict[str, float]]) -> pd.DataFrame:
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _donor(args)


def _donor(args: tuple[str, str, int, dict[str, float]]) -> pd.DataFrame:
    snapshot, pid, k_donor, levels = args
    from biomech.emg.artifacts import masked_envelope
    from biomech.io.context import DataContext
    from biomech.io.session import OPENERS
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.testing import synth as S
    from biomech.timing import lead

    saved = (lead.analyse_ensemble, lead.analyse_release)
    cfg = _config(snapshot)
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    rows: list[dict[str, Any]] = []
    try:
        pc = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("timing"),
                                store=dc.store, participant=part, cache=cache)
        p = pc.params
        prep = pc.emg_prep()
        rest = prep.phase("rest")
        fs = float(rest.fs)
        names = list(prep.names)
        pools = []
        for m in S.MUSCLES:
            i = names.index(m)
            v = np.asarray(rest.cleaned[:, i], float)[~np.asarray(rest.mask[:, i], bool)]
            pools.append(v[np.isfinite(v)])
        rest_x = np.where(rest.mask, np.nan, rest.cleaned)
        rl = lead._rest_level(rest_x, rest.mask, fs, 0.05)
        r_open = float(np.mean([rl[names.index(m)] for m in OPENERS]))
        lead.analyse_ensemble = lambda *a, **k: ([], {})
        lead.analyse_release = lambda *a, **k: ([], [])
        for rep in range(N_REP):
            for lag in LAGS_MS:
                spec = S.Spec(pid=f"syn{k_donor:02d}{rep}", seed=10_000 + 100 * k_donor + rep,
                              opener_lead_ms=lag, lead_jitter_ms=JITTER_MS,
                              emg_fs=int(round(fs)))
                ta = S.timing_arrays(spec)
                truth = {t: {k: (b["cycle"][0] - b["opener"][0]) * 1e3
                             for k, b in enumerate(bs)} for t, bs in ta.truth.items()}
                for lname, ldb in levels.items():
                    g = r_open * 10 ** (ldb / 20) / OPENER_AMP_UV
                    rng = np.random.default_rng([k_donor, rep, int(lag), int(round(ldb * 100))])
                    phases = {}
                    for ph in ["rest", "open_close", "chewing"]:
                        a = ta.prep.phase(ph)
                        n = len(a.t_us)
                        noise = np.column_stack([
                            pool[(rng.integers(0, pool.size) + np.arange(n)) % pool.size]
                            if pool.size else np.zeros(n) for pool in pools])
                        x = g * a.cleaned + noise
                        phases[ph] = S._PhaseArrays(a.t_us, a.fs, x, a.mask,
                                                    masked_envelope(x, a.mask, a.fs, 0.15))
                    sp = S.SynthPrep(list(S.MUSCLES), phases)
                    tables, _ = lead.analyse(sp, spec.pid, ta.imu_arrays, ta.kin_cycles, p,
                                             spec.seed, n_null=0)
                    on = tables["onsets"]
                    on = on[(on.direction == "opening") & on.muscle.isin(OPENERS)]
                    for _, r in on.iterrows():
                        tl = truth.get(r.task, {}).get(int(r.cycle), np.nan)
                        rows.append({"donor": pid, "rep": rep, "lag_ms": lag, "level": lname,
                                     "level_db": ldb, "g": g, "task": r.task,
                                     "detector": r.detector, "rule": r.rule,
                                     "cycle": int(r.cycle), "muscle": r.muscle,
                                     "true_lead_ms": tl, "lead_ms": r.lead_ms,
                                     "error_ms": r.lead_ms - tl, "usable": bool(r.usable),
                                     "burst_db": r.burst_db})
        return pd.DataFrame(rows)
    finally:
        lead.analyse_ensemble, lead.analyse_release = saved
        cache.close()
        dc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--pids", nargs="*")
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    pids = a.pids or sorted(dc.table("timing", "summary").pid.unique())
    dc.close()
    lv = levels_db(a.snapshot)
    jobs = [(a.snapshot, p, k, lv) for k, p in enumerate(pids)]
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        df = pd.concat(list(ex.map(donor, jobs)), ignore_index=True)
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = a.snapshot + ("" if not a.pids else "_partial")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"j18_synth_{suffix}.parquet"
    df.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "donors": pids, "levels_db": lv,
            "lags_ms": LAGS_MS, "jitter_ms": JITTER_MS, "n_rep": N_REP,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (out_dir / f"j18_synth_{suffix}.json").write_text(json.dumps(meta, indent=1))
    u = df[(df.detector == "tkeo") & df.rule.isin(["first", "drive"])]
    s = (u.groupby(["level", "lag_ms", "task", "rule"])
         .agg(usable=("usable", "mean"),
              bias=("error_ms", lambda e: float(np.nanmedian(e[u.loc[e.index, "usable"]])))))
    print(s.round(2).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
