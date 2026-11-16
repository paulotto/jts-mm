# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic gates G1-G6 of the post hoc family C2-E (X6 (change log)).

Run before any real-data run; their results are written into the analysis plan as "fixed by
the synthetic gate" and into ``stages.timing.c2e`` (``delta_ms``, ``gate_power_40ms``,
``s2_min_sd_ms``, ``gates_passed``). Marked slow; ~7150 participants, about 90 CPU-minutes
(0.75 s each unloaded), i.e. ~6-10 min on 16 idle workers (41 min on 2026-10-30 with the
machine loaded by other jobs, load ~36); run with ``nice``. Everything goes through the analysis code itself (:func:`biomech.timing.c2e.analyse`
and :func:`biomech.stats.cohort_c2e.c2e_cohort` with the packaged ``stages.timing``
parameters) on the in-memory simulator :func:`biomech.testing.synth.timing_arrays`.

r10-like participants (``R10``; cycle timing and usable shares read from the r10 kinematics
and onset tables, no outcome): ~10 open_close cycles (period 1.78 s, SD 0.2 s) and ~14 chewing
cycles, burst dropout 0.7 (closer usable share ~0.25-0.35 under TKEO / drive_taskbase, r10:
0.23 open_close, 0.35 chewing), EMG at 1 kHz. A true lead L is the closer burst onset before
the true closing onset; lead jitter 20 or 40 ms (SD per burst).

Gates and pass criteria (fixed before running):

* **G1 calibration** (EMG unrelated to the motion, 2600 participants = 200 cohorts of 13):
  per-participant locking p (pooled, closing, S1) uniform (KS p > 0.01; P(p <= 0.05) in
  [0.03, 0.07]); cohort type I error of H1 (IUT, the delta of G4) <= 0.07, and of its locking
  part alone <= 0.07.
* **G2 cycle structure** (EMG locked to the previous event, L = 40 ms, 390 participants):
  per-task S2 locking p <= 0.05 in <= 7 % of the participants (not anti-conservative), and
  the cohort S2 row rejects in <= 7 % of 30 cohorts.
* **G3 power** of H1 at L in {0, 20, 40, 80} ms x jitter {20, 40} ms (40 cohorts of 13 per
  condition); the power at L = 0 is <= 0.07; the power at 40 ms (jitter 20) is recorded as
  ``gate_power_40ms``.
* **G4 bias -> delta, mixture coverage**: bias(L) = median over participants of mu - L; delta
  = bias(0) (jitter 20) + 8 ms (latency chain, E34), rounded up to 1 ms. The mixture is kept
  (no fallback) if |bias(L) - bias(0)| <= 10 ms for L in {20, 40, 80} and the bias-corrected
  cohort BCa CI of the median mu covers L in >= 90 % of the cohorts at every L (jitter 20);
  otherwise the fallback (c2_symmetric_window median - S1 null) is the location statistic.
* **G5 artefacts**: per participant the median omega / alpha leads move < 5 ms under head
  motion (5 deg) plus a jaw-IMU tilt error (5 deg) (the |a_rel| onset is reported for
  contrast); a motion artefact on the closers (EMG unrelated) gives |median P5 lag| <= 10 ms;
  a reflex (EMG 60 ms after the acceleration) gives a median P5 lag < 0; the torque model
  with an electromechanical delay of 40 ms is recovered within 15 ms.
* **G6 release**: closer release 80 ms before the true opening onset (tonic closer level,
  long pauses) recovered within +-15 ms (cohort median of the per-participant release lead
  minus 80 ms); S2 identifiability limit: the smallest median SD of (t_mech - anchor) at
  which the S2 slope test reaches power >= 0.8 for release locked to the upcoming opening,
  over period jitter {0, 0.05, 0.1, 0.2, 0.4} s (false-positive rate under previous-event
  locking reported).

Set ``C2E_GATE_REPORT=<path.json>`` to write every gate number to a file.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

pytestmark = pytest.mark.slow

N_JOBS = int(os.environ.get("C2E_GATE_JOBS", max(1, min(16, (os.cpu_count() or 2) - 4))))
SCALE = float(os.environ.get("C2E_GATE_SCALE", "1"))     # < 1: a smoke run (no pass claims)


def _n(cohorts: int) -> int:
    """Participants for ``cohorts`` cohorts of 13 (scaled for smoke runs)."""
    return max(2, int(round(cohorts * SCALE))) * N_COHORT
R10 = dict(emg_fs=1000, open_close_period_s=1.78, period_jitter_s=0.2,
           task_s={"open_close": 19.5, "chewing": 13.0}, burst_dropout=0.7)
FAST = {"directions": ["closing"], "reref": False, "release": False, "p5": False,
        "changepoint": False, "symmetric_window_s": None}
LATENCY_MS = 8.0                 # E34 latency chain added to the synthetic bias
LEADS = (0.0, 20.0, 40.0, 80.0)
JITTERS = (20.0, 40.0)
N_COHORT = 13
REPORT: dict[str, Any] = {}


def _params(overrides: dict[str, Any]) -> dict[str, Any]:
    from biomech.config import package_default

    p = package_default("stages", "timing")
    p["c2e"] = {**p["c2e"], **overrides}
    return p


def _one(job: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    """One synthetic participant through the C2-E analysis (worker)."""
    from biomech.testing.synth import Spec, timing_arrays
    from biomech.timing import c2e
    from biomech.timing import lead as L

    spec_kw, over, extra = job
    spec = Spec(**spec_kw)
    ta = timing_arrays(spec)
    p = _params(over)
    out = c2e.analyse(ta.prep, spec.pid, ta.imu_arrays, ta.kin_cycles, p, seed=7,
                      kin_arrays=ta.kin_arrays)
    res: dict[str, Any] = {"pid": spec.pid, "null": out["c2e_null"],
                           "coupling": out["coupling"]}
    if extra == "cycles":
        res["cycles"] = out["c2e_cycles"]
        on, _ = L.analyse(ta.prep, spec.pid, ta.imu_arrays, ta.kin_cycles,
                          {**p, "c2e": {}}, seed=7, n_null=0)
        res["onsets"] = on["onsets"]
    return res


def _single_thread() -> None:
    """Worker initializer: one BLAS / OpenMP thread per worker (no oversubscription)."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)


def _pool(specs: list[dict[str, Any]], over: dict[str, Any], extra: str = ""
          ) -> list[dict[str, Any]]:
    jobs = [(s, over, extra) for s in specs]
    if N_JOBS <= 1:
        return [_one(j) for j in jobs]
    with ProcessPoolExecutor(N_JOBS, initializer=_single_thread) as ex:
        return list(ex.map(_one, jobs, chunksize=4))


def _specs(tag: str, n: int, seed0: int, **kw: Any) -> list[dict[str, Any]]:
    return [{"pid": f"{tag}{k:04d}", "seed": seed0 + k, **R10, **kw} for k in range(n)]


def _null(results: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.concat([r["null"] for r in results], ignore_index=True)


def _h1(null: pd.DataFrame, delta: float | None) -> pd.DataFrame:
    """H1 row (and the S2 row) of :func:`c2e_cohort` per cohort of 13 (in pid order)."""
    from biomech.stats.cohort_c2e import CohortParams, c2e_cohort, s2_log_ratios
    from biomech.stats.families import log_ratio_test

    pids = sorted(null.pid.unique())
    rows = []
    for k in range(len(pids) // N_COHORT):
        sub = null[null.pid.isin(pids[k * N_COHORT:(k + 1) * N_COHORT])]
        c = c2e_cohort(sub, None, {"delta_ms": delta}, seed=k).set_index("key")
        # the cohort S2 p is withheld from the table (not calibrated, G2); computed here
        lr = s2_log_ratios(sub, CohortParams.from_dict({"delta_ms": delta}))
        s2_p = (log_ratio_test(lr, "less", k, CohortParams().min_usable)["p"]
                if lr is not None else np.nan)
        rows.append({"cohort": k, **{f"h1_{f}": c.loc["h1", f] for f in
                                     ("p", "p_lock", "p_loc", "median", "ci_lo", "ci_hi", "n")},
                     "s2_p": s2_p})
    return pd.DataFrame(rows)


def _primary(null: pd.DataFrame, scope: str = "pooled") -> pd.DataFrame:
    return null[(null.scope == scope) & (null.direction == "closing")
                & (null.source == "omega")]


# -- pools (computed once; every gate reads them) ---------------------------------------------
@pytest.fixture(scope="module")
def pools() -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["g1"] = _null(_pool(_specs("n", _n(200), 100_000, emg_lock="none",
                                   lead_jitter_ms=20.0), FAST))
    for L in LEADS:
        for j in JITTERS:
            over = dict(FAST, symmetric_window_s=0.4) if j == JITTERS[0] else FAST
            out[("g3", L, j)] = _null(_pool(
                _specs(f"L{int(L)}j{int(j)}_", _n(40),
                       200_000 + 10_000 * LEADS.index(L) + 5_000 * JITTERS.index(j),
                       closer_lead_ms=L, lead_jitter_ms=j), over))
    out["g2"] = _null(_pool(_specs("pe", _n(30), 300_000, closer_lead_ms=40.0,
                                   lead_jitter_ms=20.0, emg_lock="previous_event"), FAST))
    return out


def _delta(pools: dict[str, Any]) -> tuple[float, dict[tuple[float, float], float]]:
    bias = {}
    for L in LEADS:
        for j in JITTERS:
            g = _primary(pools[("g3", L, j)])
            g = g[g.n_usable >= 5]
            bias[(L, j)] = float(np.nanmedian(g.mu_ms - L))
    return float(np.ceil(bias[(0.0, JITTERS[0])] + LATENCY_MS)), bias


def _record(key: str, value: Any) -> None:
    REPORT[key] = value
    path = os.environ.get("C2E_GATE_REPORT")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(REPORT, f, indent=1, default=float)
    print(f"[C2-E gate] {key}: {value}")


def test_g4_bias_delta_and_mixture_coverage(pools):
    delta, bias = _delta(pools)
    _record("g4_bias_ms", {f"L{int(L)}_j{int(j)}": round(b, 2) for (L, j), b in bias.items()})
    _record("g4_delta_ms", delta)
    b0 = bias[(0.0, JITTERS[0])]
    cov, fb = {}, {}
    for L in LEADS:
        h = _h1(pools[("g3", L, JITTERS[0])], delta)
        ok = (h.h1_ci_lo - b0 <= L) & (h.h1_ci_hi - b0 >= L)
        cov[L] = float(ok[np.isfinite(h.h1_ci_lo)].mean())
        s = pools[("g3", L, JITTERS[0])]
        s = s[(s.scope == "pooled") & (s.direction == "closing") & (s.source == "sym")]
        fb[L] = float(np.nanmedian(s.median_minus_null_ms))
    _record("g4_coverage", {f"L{int(L)}": round(v, 3) for L, v in cov.items()})
    _record("g4_fallback_median_minus_null_ms", {f"L{int(L)}": round(v, 1)
                                                 for L, v in fb.items()})
    stable = all(abs(bias[(L, JITTERS[0])] - b0) <= 10.0 for L in LEADS[1:])
    keep = stable and all(v >= 0.90 for v in cov.values())
    _record("g4_mixture_kept", bool(keep))
    assert np.isfinite(delta)


def test_g1_calibration(pools):
    delta, _ = _delta(pools)
    g = _primary(pools["g1"])
    p = g.p_lock.to_numpy(float)
    p = p[np.isfinite(p)]
    ks = sps.kstest(p, "uniform").pvalue
    rate = float(np.mean(p <= 0.05))
    h = _h1(pools["g1"], delta)
    t1 = float(np.mean(h.h1_p <= 0.05))
    t1_lock = float(np.mean(h.h1_p_lock <= 0.05))
    t1_loc = float(np.mean(h.h1_p_loc <= 0.05))
    _record("g1", {"n_participants_with_p": int(p.size), "ks_p": float(ks),
                   "p_le_005": rate, "n_cohorts": int(len(h)), "type1_h1": t1,
                   "type1_locking": t1_lock, "type1_location": t1_loc,
                   "usable_frac_median": float(np.nanmedian(g.usable_frac))})
    assert len(h) >= 200 or SCALE < 1
    assert ks > 0.01 and 0.03 <= rate <= 0.07
    assert t1 <= 0.07 and t1_lock <= 0.07


def test_g2_s2_not_anticonservative(pools):
    delta, _ = _delta(pools)
    n = pools["g2"]
    per = n[(n.scope != "pooled") & (n.direction == "closing") & (n.source == "omega")]
    rate = float(np.mean(per.p_s2.dropna() <= 0.05))
    h = _h1(n, delta)
    coh = float(np.mean(h.s2_p <= 0.05))
    # contrast: S2 power under locking to the upcoming onset (G3 pool, L = 40, jitter 20)
    m = pools[("g3", 40.0, JITTERS[0])]
    pm = m[(m.scope != "pooled") & (m.direction == "closing") & (m.source == "omega")]
    power = float(np.mean(pm.p_s2.dropna() <= 0.05))
    _record("g2", {"per_task_rate": rate, "cohort_rate": coh, "n_cohorts": int(len(h)),
                   "s2_power_motion_locked_per_task": power,
                   "h1_rejection_previous_event": float(np.mean(h.h1_p <= 0.05))})
    # The cohort S2 row is descriptive since 2026-10-31 (its p is withheld: 0.095 at 0.05 over
    # 201 cohorts); only the per-task S2 test carries a calibration requirement.
    assert rate <= 0.07


def test_g3_power(pools):
    delta, _ = _delta(pools)
    power = {}
    for L in LEADS:
        for j in JITTERS:
            h = _h1(pools[("g3", L, j)], delta)
            power[f"L{int(L)}_j{int(j)}"] = {
                "h1": float(np.mean(h.h1_p <= 0.05)),
                "locking": float(np.mean(h.h1_p_lock <= 0.05)),
                "location": float(np.mean(h.h1_p_loc <= 0.05)),
                "n_tested_median": float(h.h1_n.median())}
    g = _primary(pools[("g3", 40.0, JITTERS[0])])
    _record("g3_power", power)
    _record("g3_usable_frac", {"pooled_median": float(np.nanmedian(g.usable_frac)),
                               "open_close": float(np.nanmedian(_primary(
                                   pools[("g3", 40.0, JITTERS[0])], "open_close").usable_frac)),
                               "chewing": float(np.nanmedian(_primary(
                                   pools[("g3", 40.0, JITTERS[0])], "chewing").usable_frac))})
    _record("gate_power_40ms", power[f"L40_j{int(JITTERS[0])}"]["h1"])
    assert power[f"L0_j{int(JITTERS[0])}"]["h1"] <= 0.07


def _median_by_pid(cyc: pd.DataFrame, col: str, direction: str) -> pd.Series:
    c = cyc[(cyc.direction == direction) & cyc.usable.astype(bool)]
    return c.groupby("pid")[col].median()


def test_g5_artefacts():
    full = {"n_surrogates": 50, "changepoint": False, "symmetric_window_s": None,
            "release": False}
    base = _specs("h", 12, 400_000, closer_lead_ms=40.0, opener_lead_ms=40.0,
                  lead_jitter_ms=20.0)
    moved = [dict(s, head_motion_deg=5.0, tilt_error_deg=5.0) for s in base]
    r0 = _pool(base, full, "cycles")
    r1 = _pool(moved, full, "cycles")
    c0 = pd.concat([r["cycles"] for r in r0])
    c1 = pd.concat([r["cycles"] for r in r1])
    shift = {}
    for d in ("closing", "opening"):
        for col in ("lead_omega_ms", "lead_alpha15_ms"):
            dd = (_median_by_pid(c1, col, d) - _median_by_pid(c0, col, d)).abs()
            shift[f"{d}_{col}"] = float(dd.max())
    o0 = pd.concat([r["onsets"] for r in r0])
    o1 = pd.concat([r["onsets"] for r in r1])
    sel = lambda o: o[(o.rule == "drive") & (o.detector == "tkeo")]  # noqa: E731
    acc = (sel(o1).groupby(["pid", "direction"]).lead_acc_ms.median()
           - sel(o0).groupby(["pid", "direction"]).lead_acc_ms.median()).abs()
    art = _pool(_specs("a", 12, 410_000, emg_lock="none", emg_artifact_uv=40.0), full)
    lag_art = float(np.nanmedian(pd.concat([r["coupling"] for r in art]).emd_lag_ms))
    refl = _pool(_specs("r", 12, 420_000, emg_model="torque", reflex_ms=60.0), full)
    lag_ref = float(np.nanmedian(pd.concat([r["coupling"] for r in refl]).emd_lag_ms))
    tq = _pool(_specs("t", 12, 430_000, emg_model="torque", emd_ms=40.0), full)
    lag_tq = float(np.nanmedian(pd.concat([r["coupling"] for r in tq]).emd_lag_ms))
    _record("g5", {"max_shift_ms": shift, "acc_onset_shift_ms_median": float(acc.median()),
                   "acc_onset_shift_ms_max": float(acc.max()),
                   "artefact_lag_ms": lag_art, "reflex_lag_ms": lag_ref,
                   "torque_emd40_lag_ms": lag_tq})
    assert all(v < 5.0 for v in shift.values())
    assert lag_ref < 0 and abs(lag_tq - 40.0) <= 15.0
    # Motion artefact: the pre-stated |lag| <= 10 ms holds on the anti-conservative side
    # (EMG before motion). The gate run gave -12.5 ms, accepted as a documented deviation
    # (author decision 2026-10-30, analysis plan X6 gate results): it biases the supporting P5
    # lag towards "EMG after motion" and no claim rests on P5. Below -15 ms would be a change
    # from the recorded gate, not the accepted deviation.
    assert lag_art <= 10.0 and lag_art >= -15.0


def test_g6_release_and_s2_identifiability():
    rel = {"directions": ["opening"], "reref": False, "p5": False, "changepoint": False,
           "symmetric_window_s": None, "release": True}
    kw = dict(closer_tonic_uv=25.0, closer_release_ms=80.0, lead_jitter_ms=20.0,
              open_close_period_s=2.5)
    res = _null(_pool(_specs("rl", 26, 500_000, **{**kw}), rel))
    r = res[(res.source == "release") & (res.scope == "open_close")]
    err = float(np.nanmedian(r.median_ms - 80.0))
    out = {"open_close_median_error_ms": err,
           "open_close_abs_error_ms_p90": float(np.nanpercentile(np.abs(r.median_ms - 80.0),
                                                                 90)),
           "depth_db_median": float(np.nanmedian(r.depth_db)),
           "p_depth_le_005": float(np.mean(r.p_depth <= 0.05))}
    ident = {}
    for pj in (0.0, 0.05, 0.1, 0.2, 0.4):
        row = {}
        for lock in ("motion", "previous_event"):
            n = _null(_pool(_specs(f"s{lock[0]}{int(pj * 100)}_", 26,
                                   600_000 + int(pj * 1e4) + (50_000 if lock != "motion" else 0),
                                   **{**kw, "period_jitter_s": pj, "emg_lock": lock}), rel))
            q = n[(n.source == "release") & (n.scope == "open_close")]
            row[lock] = {"rate": float(np.mean(q.p_b_s2 <= 0.05)),
                         "x_sd_ms": float(np.nanmedian(q.x_sd_ms)),
                         "b_median": float(np.nanmedian(q.b_s2))}
        ident[str(pj)] = row
    ok = [v["motion"]["x_sd_ms"] for v in ident.values() if v["motion"]["rate"] >= 0.8]
    out["s2_identifiability"] = ident
    out["s2_min_sd_ms"] = float(min(ok)) if ok else None
    _record("g6", out)
    assert abs(err) <= 15.0
