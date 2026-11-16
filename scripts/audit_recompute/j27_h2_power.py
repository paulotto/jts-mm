# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J27: power of a future confirmatory test of the closer release (H2), by
resampling the current usable participants (planning simulation; definitions fixed in J27 before this ran).

Per task (open-close: H2, chewing: H3) the usable participants of the paper store (pin r16a,
``timing:c2e_null``, direction opening, source release, ``n_usable`` >= ``min_usable``) with
their (``log_depth_ratio``, ``median_ms``) pair. Per replicate n of them drawn with
replacement; locking = one-sided signed-rank of the log depth ratio > 0, location = one-sided
signed-rank of median_ms - delta > 0 (``stats.families.wilcoxon_signed``, as the stored test),
combined p = max. Location shrink s: median_ms' = delta + (1 - s)(median_ms - delta).
Open-close at alpha; chewing under Holm over the two tasks (independent draws per task).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j27_h2_power.py [-j 8]

Output: ``<adults store>/audit/j27_h2_power_<snap>.parquet`` (power per task, n, shrink,
test) and ``.json`` (pools, reproduction, settings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import _pins  # noqa: E402

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
TASKS = ("open_close", "chewing")
N_GRID = tuple(range(6, 31, 2))
SHRINK = (0.0, 0.25, 0.5)
# J27a: (variant, s). "scale" = the pre-specified shrink (invariant for the signed-rank test);
# "shift_loc" / "shift_both" move values toward the null by s x the pool's median effect.
VARIANTS = ([("scale", s) for s in SHRINK]
            + [(v, s) for v in ("shift_loc", "shift_both") for s in (0.25, 0.5)])
N_REP = 10_000
ALPHA = 0.05


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def pools(snapshot: str) -> tuple[dict[str, pd.DataFrame], dict, pd.DataFrame]:
    from biomech.io.context import DataContext

    cfg = _config(snapshot)
    dc = DataContext(cfg)
    try:
        nul = dc.table("timing", "c2e_null")
        coh = dc.table("stats_timing", "c2e_cohort")
        c2e = dict(cfg.params("timing").get("c2e") or {})
        n_part = int(dc.table("acquisition_cohort", "per_session").pid.nunique())
    finally:
        dc.close()
    out = {}
    for task in TASKS:
        g = nul[(nul.scope == task) & (nul.direction == "opening") & (nul.source == "release")]
        out[task] = g[g.n_usable.fillna(0) >= int(c2e["min_usable"])][
            ["pid", "n_usable", "log_depth_ratio", "median_ms"]].reset_index(drop=True)
    c2e["n_participants"] = n_part
    return out, c2e, coh


def pvals(lr: np.ndarray, med: np.ndarray, delta: float, min_n: int) -> tuple[float, float]:
    from biomech.stats.families import wilcoxon_signed

    pl = wilcoxon_signed(lr, "greater", min_n)[1]
    pc = wilcoxon_signed(med - delta, "greater", min_n)[1]
    return pl, pc


def run_task(args: tuple) -> pd.DataFrame:
    from threadpoolctl import threadpool_limits

    task, n, lr, med, delta, min_n, seed = args
    rng = np.random.default_rng([seed, TASKS.index(task), n])
    k = lr.size
    m_lead = float(np.nanmedian(med)) - delta       # pool median location effect
    m_lr = float(np.nanmedian(lr))                  # pool median locking effect
    rows = []
    with threadpool_limits(limits=1):
        for rep in range(N_REP):
            idx = rng.integers(0, k, n)
            for var, s in VARIANTS:
                x_lr, x_med = lr[idx], med[idx]
                if var == "scale":
                    x_med = delta + (1.0 - s) * (x_med - delta)
                else:
                    x_med = x_med - s * m_lead
                    if var == "shift_both":
                        x_lr = x_lr - s * m_lr
                pl, pc = pvals(x_lr, x_med, delta, min_n)
                rows.append((rep, var, s, pl, pc))
    d = pd.DataFrame(rows, columns=["rep", "variant", "shrink", "p_lock", "p_loc"])
    return d.assign(task=task, n=n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("-j", "--jobs", type=int, default=8)
    a = ap.parse_args()
    pool, c2e, coh = pools(a.snapshot)
    delta, min_n = float(c2e["delta_ms"]), int(c2e["min_usable"])
    # reproduction of the stored H2 / H3 p values
    repro = {}
    for task, key in (("open_close", "h2"), ("chewing", "h3")):
        g = pool[task]
        pl, pc = pvals(g.log_depth_ratio.to_numpy(float), g.median_ms.to_numpy(float), delta,
                       min_n)
        r = coh[coh.key == key].iloc[0]
        repro[task] = {"p_lock": pl, "p_loc": pc, "stored_p_lock": float(r.p_lock),
                       "stored_p_loc": float(r.p_loc),
                       "equal": bool(np.isclose(pl, r.p_lock, rtol=0, atol=1e-12)
                                     and np.isclose(pc, r.p_loc, rtol=0, atol=1e-12))}
    if not all(v["equal"] for v in repro.values()):
        raise SystemExit(f"reproduction failed: {repro}")
    seed = int(_config(a.snapshot).seed or 0) + 27
    jobs = [(t, n, pool[t].log_depth_ratio.to_numpy(float), pool[t].median_ms.to_numpy(float),
             delta, min_n, seed) for t in TASKS for n in N_GRID]
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        d = pd.concat(list(ex.map(run_task, jobs)), ignore_index=True)
    d["p"] = d[["p_lock", "p_loc"]].max(axis=1, skipna=False)
    # chewing under Holm over the two tasks: pair replicate r of chewing with replicate r of
    # open-close at the same n and shrink (independent draws)
    w = d.pivot_table(index=["n", "variant", "shrink", "rep"], columns="task",
                      values=["p", "p_lock", "p_loc"])
    rows = []
    for (n, var, s), g in w.groupby(level=["n", "variant", "shrink"]):
        for test in ("p", "p_lock", "p_loc"):
            po = g[(test, "open_close")].to_numpy(float)
            pch = g[(test, "chewing")].to_numpy(float)
            ok_o = po <= ALPHA
            holm_ch = (pch <= ALPHA / 2) | ((po <= ALPHA / 2) & (pch <= ALPHA))
            name = {"p": "combined", "p_lock": "locking", "p_loc": "location"}[test]
            rows.append({"task": "open_close", "n": n, "variant": var, "shrink": s, "test": name,
                         "power": float(np.mean(ok_o)), "n_rep": int(po.size)})
            rows.append({"task": "chewing", "n": n, "variant": var, "shrink": s, "test": name,
                         "power": float(np.mean(holm_ch)), "n_rep": int(pch.size),
                         "power_unadjusted": float(np.mean(pch <= ALPHA))})
    out = pd.DataFrame(rows)
    cfg = _config(a.snapshot)
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"j27_h2_power_{a.snapshot}.parquet"
    out.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "delta_ms": delta, "min_usable": min_n,
            "alpha": ALPHA, "n_rep": N_REP, "n_grid": list(N_GRID),
            "variants": [list(v) for v in VARIANTS],
            "seed": seed, "n_participants": c2e["n_participants"], "reproduction": repro,
            "pools": {t: {"n": int(len(g)), "pids": list(g.pid),
                          "n_usable_min": int(g.n_usable.min()),
                          "n_usable_median": float(g.n_usable.median()),
                          "n_usable_max": int(g.n_usable.max()),
                          "median_finite": int(np.isfinite(g.median_ms).sum())}
                      for t, g in pool.items()},
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (out_dir / f"j27_h2_power_{a.snapshot}.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 200)
    sel = out[out.n.isin([6, 8, 10, 12, 16, 20, 30])]
    print(sel.pivot_table(index=["task", "variant", "shrink", "n"], columns="test",
                          values="power").round(3).to_string())
    print(json.dumps({k: meta[k] for k in ("reproduction", "pools")}, indent=1))


if __name__ == "__main__":
    main()
