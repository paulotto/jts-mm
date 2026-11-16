# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU dynamic-calibration audit (descriptive, read-only on stored data).

The JTS-MM firmware enables the BNO085 motion-engine dynamic calibration (accelerometer,
gyroscope, magnetometer) on every configuration apply (``firmware/jtsmm/src/domain/
imu_config.c:436``), so it was active in every recording. For the game rotation vector the
gyroscope zero-rate offset (and accelerometer) estimates are updated while the device
detects stillness; quaternion accuracy stays 3, so an update is silent. The question: do the
recordings show steps or drift-rate changes of the jaw-in-head IMU orientation aligned with
still periods (rest phase, landmark holds, idle gaps)?

**Still periods** (both IMUs): the angular speed of each IMU's game rotation vector
(per-sample geodesic increment on a 100 Hz grid, centred ``SMOOTH_S`` mean) below
``STILL_DPS`` for at least ``STILL_MIN_S``.

**Error traces** (motion removed by an optical reference):

* dataset (``camerafree`` arrays of the store, 200 Hz over the experiment phases): the
  opening error of E2a (hand-eye mountings, one closed-pose reference at rest, no resets)
  against the optics, ``open_deg__E2a − open_deg__opt``, at the IMU-detected closed samples
  (``closed``; pose-independent: the pure drift of the relative orientation);
* validation (``ref_fusion`` arrays, 40 Hz): the IMU-alone (VD10 ``I``) geodesic rotation
  error against Qualisys, ``err_rot_I_deg``, where finite.

**Per trace:** 0.25-s block medians; a *step* is a change between consecutive blocks (no gap
> ``MAX_GAP_S``) larger than ``STEP_MIN_DEG`` and ``STEP_K`` robust SDs of all block
changes. *Aligned*: the step lies inside a still period or within ``AFTER_S`` after its end;
the expected aligned share is the share of block changes that lie there (no alignment ->
equal shares). Drift: the median over the still periods of the trace's OLS slope inside each
(deg/min) against the slope over the samples outside them, per take. **Transitions:** the
jump of the trace across each phase boundary (first against last ``JUMP_S`` of samples)
minus the take's drift over the gap, for gaps with and without a still period. **Anchors**
(dataset): the relative heading drift (deg/min) from the E3b anchors at rest and in the idle
gaps between tasks. **Per still period** additionally the pure-IMU relative
orientation (``R_craᵀ Rz(ψ) R_mta``, ψ = the E2a heading of ``cf_summary``; 0 for the
validation takes) change over the period (deg, deg/min; includes the participant's own
micro-motion).

Usage (paths relative to the repository root, all optional)::

    python scripts/audit_recompute/imu_dyncal.py [--store data/analysis/icorr2027/snapshots/<record>]
        [--vstore data/analysis/jtsmm_validation/snapshots/<systems pin>] [--out data/analysis/icorr2027/audit/imu_dyncal.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as Rot

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

from biomech.imu import camerafree as cf
from biomech.imu.preprocess import load
from biomech.io.session import SessionData

ROOT = Path(__file__).resolve().parents[2]
GRID_HZ = 100.0          # IMU analysis grid (Hz)
SMOOTH_S = 0.2           # centred mean of the angular speed (s): quaternion noise ~0.7 deg/s
STILL_DPS = 2.0          # still: both IMUs below this angular speed (deg/s) ...
STILL_MIN_S = 2.0        # ... for at least this long (s)
BLOCK_S = 0.25           # block-median length of the error traces (s)
MAX_GAP_S = 1.0          # consecutive blocks at most this far apart (s): no step across motion
STEP_MIN_DEG = 0.25      # step: block change above this (deg) ...
STEP_K = 5.0             # ... and above this many robust SDs of all block changes
AFTER_S = 2.0            # aligned: inside a still period or up to this long after its end (s)
JUMP_S = 1.0             # transition jump: last / first 1 s of error samples of two phases


def validation_takes() -> dict[str, str]:
    """pid -> motion take of the validation study, from its cohort file (session names are
    not written into code; the cohort file belongs to the data, not to the public release)."""
    f = ROOT / "experiments" / "jtsmm_validation" / "cohort.yaml"
    if not f.is_file():
        return {}
    doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return {str(p["pid"]): str(p["motion"]) for p in doc.get("participants", [])
            if p.get("motion") and p.get("include", True)}


# -- IMU stillness and relative orientation ---------------------------------------------------
def imu_grid(sd: SessionData) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(grid µs, head quaternions, jaw quaternions) on the common span, or None."""
    c, m = load(sd, "cra"), load(sd, "mta")
    if c is None or m is None:
        return None
    t0, t1 = max(c.t_q[0], m.t_q[0]), min(c.t_q[-1], m.t_q[-1])
    tg = np.arange(t0, t1, int(round(1e6 / GRID_HZ)), dtype=np.int64)
    return tg, c.quat_at(tg), m.quat_at(tg)


def speed_dps(q: np.ndarray) -> np.ndarray:
    """Angular speed (deg/s) of a wxyz quaternion series on the grid, centred mean."""
    R = Rot.from_quat(q[:, [1, 2, 3, 0]])
    w = np.degrees((R[:-1].inv() * R[1:]).magnitude()) * GRID_HZ
    n = max(1, int(round(SMOOTH_S * GRID_HZ)))
    w = np.convolve(np.r_[w, w[-1]], np.ones(n) / n, "same")
    return w


def runs(mask: np.ndarray, min_n: int) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(i), int(j)) for i, j in zip(a, b) if j - i >= min_n]


def still_periods(tg: np.ndarray, qc: np.ndarray, qm: np.ndarray) -> list[tuple[int, int]]:
    still = (speed_dps(qc) < STILL_DPS) & (speed_dps(qm) < STILL_DPS)
    return runs(still, int(round(STILL_MIN_S * GRID_HZ)))


def where(sd: SessionData, t0_us: int, t1_us: int) -> str:
    """The recorded phase a span overlaps most (``gap`` outside every phase)."""
    best, name = 0, "gap"
    for n in sd.phase_names():
        ph = sd.phase(n)
        ov = min(t1_us, ph.t_end_us) - max(t0_us, ph.t_start_us)
        if ov > best:
            best, name = ov, n
    return name


def period_rows(sd: SessionData, tg: np.ndarray, qc: np.ndarray, qm: np.ndarray,
                psi_rad: float, per: list[tuple[int, int]]) -> list[dict]:
    """Per still period: phase, duration and the relative IMU orientation change."""
    M = cf.imu_product(qc, qm, psi_rad)
    out = []
    n_edge = max(1, int(round(0.5 * GRID_HZ)))
    for a, b in per:
        rv = np.degrees(cf._rotvec(M[a:b] @ cf._T(M[a])))
        dur = (tg[b - 1] - tg[a]) * 1e-6
        net = float(np.linalg.norm(rv[-n_edge:].mean(0) - rv[:n_edge].mean(0)))
        out.append({"t0_s": tg[a] * 1e-6, "t1_s": tg[b - 1] * 1e-6, "duration_s": dur,
                    "phase": where(sd, int(tg[a]), int(tg[b - 1])),
                    "rel_net_deg": net, "rel_rate_dpm": net / dur * 60.0})
    return out


# -- error traces: steps and drift ------------------------------------------------------------
def blocks(t_s: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """0.25-s block medians (time = block centre) of the finite samples."""
    ok = np.isfinite(e)
    t, e = t_s[ok], e[ok]
    if t.size == 0:
        return np.array([]), np.array([])
    k = np.floor((t - t[0]) / BLOCK_S).astype(int)
    u, idx = np.unique(k, return_index=True)
    tb = np.array([np.median(t[i:j]) for i, j in zip(idx, np.r_[idx[1:], t.size])])
    eb = np.array([np.median(e[i:j]) for i, j in zip(idx, np.r_[idx[1:], t.size])])
    return tb, eb


def in_window(t: np.ndarray, per_s: list[tuple[float, float]], after: float) -> np.ndarray:
    m = np.zeros(t.size, bool)
    for a, b in per_s:
        m |= (t >= a) & (t <= b + after)
    return m


def slope_dpm(t: np.ndarray, e: np.ndarray) -> float:
    ok = np.isfinite(e)
    if ok.sum() < 20 or np.ptp(t[ok]) < 1.0:
        return np.nan
    return float(np.polyfit(t[ok] / 60.0, e[ok], 1)[0])


def _median_or_nan(x: list[float]) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def trace_summary(t_s: np.ndarray, e: np.ndarray, per_s: list[tuple[float, float]]) -> dict:
    """Steps of the block-median trace and their alignment with the still periods; drift
    inside and outside the still periods."""
    tb, eb = blocks(t_s, e)
    out = {"n_samples": int(np.isfinite(e).sum())}
    if tb.size < 3:
        return {**out, "n_changes": 0}
    dt, de = np.diff(tb), np.diff(eb)
    tm = 0.5 * (tb[1:] + tb[:-1])
    ok = dt <= MAX_GAP_S
    de, tm = de[ok], tm[ok]
    sd = 1.4826 * float(np.median(np.abs(de - np.median(de)))) if de.size else np.nan
    thr = max(STEP_MIN_DEG, STEP_K * sd) if np.isfinite(sd) else STEP_MIN_DEG
    step = np.abs(de) > thr
    near = in_window(tm, per_s, AFTER_S)
    inside = in_window(t_s, per_s, 0.0)
    return {**out, "n_changes": int(de.size), "change_sd_deg": sd, "step_threshold_deg": thr,
            "n_steps": int(step.sum()),
            "n_steps_aligned": int((step & near).sum()),
            "share_aligned_expected": float(near.mean()) if near.size else np.nan,
            "max_step_deg": float(np.max(np.abs(de))) if de.size else np.nan,
            "max_step_aligned_deg": float(np.max(np.abs(de[near]))) if near.any() else np.nan,
            "drift_in_still_dpm": _median_or_nan([slope_dpm(t_s[m], e[m]) for m in (
                (t_s >= a) & (t_s <= b) for a, b in per_s)]),
            "drift_outside_dpm": slope_dpm(t_s[~inside], e[~inside]),
            "drift_all_dpm": slope_dpm(t_s, e),
            "still_share_of_trace": float(inside[np.isfinite(e)].mean())}


def transitions(t_s: np.ndarray, e: np.ndarray, spans: list[tuple[str, float, float]],
                per_s: list[tuple[float, float]], drift_dpm: float) -> list[dict]:
    """Per pair of consecutive phases with error samples: the jump of the trace across the
    boundary (median of the first ``JUMP_S`` of samples of the next phase minus the last
    ``JUMP_S`` of the previous one), the jump expected from the take's drift slope, and the
    still time in between (the transition out of a still period)."""
    out = []
    seg = []
    for name, a, b in spans:
        m = (t_s >= a) & (t_s <= b) & np.isfinite(e)
        if m.sum() >= 5:
            seg.append((name, t_s[m], e[m]))
    for (n0, t0, e0), (n1, t1, e1) in zip(seg[:-1], seg[1:]):
        w0, w1 = e0[t0 >= t0[-1] - JUMP_S], e1[t1 <= t1[0] + JUMP_S]
        gap = float(t1[0] - t0[-1])
        still = sum(max(0.0, min(b, t1[0]) - max(a, t0[-1])) for a, b in per_s)
        jump = float(np.median(w1) - np.median(w0))
        exp = drift_dpm * gap / 60.0 if np.isfinite(drift_dpm) else np.nan
        out.append({"from": n0, "to": n1, "gap_s": gap, "still_in_gap_s": still,
                    "jump_deg": jump, "expected_deg": exp, "residual_deg": jump - exp,
                    "window_sd_deg": float(np.std(np.r_[w0 - np.median(w0),
                                                        w1 - np.median(w1)]))})
    return out


def anchor_drift(store: Path, pid: str, t_first_s: float) -> dict:
    """Relative heading drift from the E3b anchors in the rest phase and the idle gaps
    between tasks (closed, still: the cleanest closed poses; in protrusion and
    laterotrusion the anchor heading absorbs the jaw's own rotation): OLS slope (deg/min)
    of the cumulative heading part, residual SD and largest residual (a step candidate)."""
    f = store / "cache" / "camerafree" / pid / "cf_anchors.parquet"
    if not f.is_file():
        return {}
    a = pd.read_parquet(f)
    a = a[(a["estimator"] == "E3b") & a["task"].isin(["rest", "between"])]
    if len(a) < 3:
        return {"anchor_n": len(a)}
    t = (t_first_s + a["t0_s"].to_numpy(float)) / 60.0
    h = a["heading_deg"].to_numpy(float)
    k, c = np.polyfit(t, h, 1)
    r = h - (k * t + c)
    return {"anchor_n": len(a), "anchor_heading_dpm": float(k),
            "anchor_heading_net_deg": float(h[-1] - h[0]),
            "anchor_resid_sd_deg": float(np.std(r, ddof=2)) if len(a) > 2 else np.nan,
            "anchor_resid_max_deg": float(np.max(np.abs(r)))}


# -- the two studies ---------------------------------------------------------------------------
def dataset(store: Path, data_dir: Path, cohort: Path, tr_rows: list[dict]
            ) -> tuple[list[dict], list[dict]]:
    takes = {p["pid"]: p["motion"] for p in yaml.safe_load(cohort.read_text())["participants"]
             if p.get("include", True) and p.get("motion")}
    summ, per_rows = [], []
    for pid, take in sorted(takes.items()):
        base = {"study": "dataset", "pid": pid}
        sf = store / "cache" / "camerafree" / pid / "cf_summary.parquet"
        if not sf.is_file():
            summ.append({**base, "note": "no camerafree result (no jaw IMU)"})
            continue
        s = pd.read_parquet(sf).iloc[0]
        a = np.load(store / "cache" / "camerafree" / pid / "arrays.npz")
        with SessionData(data_dir / take) as sd:
            g = imu_grid(sd)
            if g is None:
                summ.append({**base, "note": "IMU streams missing"})
                continue
            tg, qc, qm = g
            per = still_periods(tg, qc, qm)
            prow = period_rows(sd, tg, qc, qm, float(np.radians(s["psi_a_deg"])), per)
        per_s = [(r["t0_s"], r["t1_s"]) for r in prow]
        t = np.asarray(a["t_s"], float)
        e = np.asarray(a["open_deg__E2a"], float) - np.asarray(a["open_deg__opt"], float)
        e[~np.asarray(a["closed"], bool)] = np.nan
        tr = trace_summary(t, e, per_s)
        rates = np.array([r["rel_rate_dpm"] for r in prow])
        names = [str(x) for x in a["phase_names"]]
        ph = np.asarray(a["phase"], int)
        spans = [(n, float(t[ph == k][0]), float(t[ph == k][-1]))
                 for k, n in enumerate(names) if (ph == k).any()]
        jumps = transitions(t, e, spans, per_s, tr.get("drift_all_dpm", np.nan))
        tr_rows.extend({**base, **j} for j in jumps)
        summ.append({**base, "trace": "E2a opening error at closed samples", **tr,
                     "n_still": len(prow), "still_s": float(sum(r["duration_s"] for r in prow)),
                     "rel_rate_median_dpm": float(np.median(rates)) if rates.size else np.nan,
                     **anchor_drift(store, pid, float(t[0])),
                     "max_abs_residual_jump_deg": float(np.max(np.abs(
                         [j["residual_deg"] for j in jumps]))) if jumps else np.nan,
                     "note": ""})
        for r in prow:
            sel = (t >= r["t0_s"]) & (t <= r["t1_s"])
            r["err_change_deg"] = _edge_change(t[sel], e[sel])
            per_rows.append({**base, **r})
    return summ, per_rows


def _edge_change(t: np.ndarray, e: np.ndarray) -> float:
    """Error change over a still period: last 0.5 s minus first 0.5 s (NaN without data)."""
    ok = np.isfinite(e)
    if ok.sum() < 10:
        return np.nan
    t, e = t[ok], e[ok]
    a, b = e[t <= t[0] + 0.5], e[t >= t[-1] - 0.5]
    return float(np.median(b) - np.median(a)) if a.size and b.size else np.nan


def validation(vstore: Path, vdata: Path, tr_rows: list[dict]
               ) -> tuple[list[dict], list[dict]]:
    summ, per_rows = [], []
    for pid, take in validation_takes().items():
        base = {"study": "validation", "pid": pid}
        f = vstore / "cache" / "ref_fusion" / pid / "arrays.npz"
        if not f.is_file() or not (vdata / take).is_dir():
            summ.append({**base, "note": "no ref_fusion arrays or session"})
            continue
        a = np.load(f)
        with SessionData(vdata / take) as sd:
            g = imu_grid(sd)
            if g is None:
                summ.append({**base, "note": "IMU streams missing"})
                continue
            tg, qc, qm = g
            prow = period_rows(sd, tg, qc, qm, 0.0, still_periods(tg, qc, qm))
            phases = {n: (sd.phase(n).t_start_us * 1e-6, sd.phase(n).t_end_us * 1e-6)
                      for n in sd.phase_names()}
        per_s = [(r["t0_s"], r["t1_s"]) for r in prow]
        t = np.asarray(a["t_master_s"], float)
        e = np.asarray(a["err_rot_I_deg"], float)
        tr = trace_summary(t, e, per_s)
        spans = [(n, sd_ph[0], sd_ph[1]) for n, sd_ph in phases.items()]
        jumps = transitions(t, e, spans, per_s, tr.get("drift_all_dpm", np.nan))
        tr_rows.extend({**base, **j} for j in jumps)
        summ.append({**base, "trace": "IMU-alone rotation error vs Qualisys", **tr,
                     "max_abs_residual_jump_deg": float(np.max(np.abs(
                         [j["residual_deg"] for j in jumps]))) if jumps else np.nan,
                     "n_still": len(prow), "still_s": float(sum(r["duration_s"] for r in prow)),
                     "rel_rate_median_dpm": float(np.median([r["rel_rate_dpm"] for r in prow]))
                     if prow else np.nan, "note": ""})
        for r in prow:
            sel = (t >= r["t0_s"]) & (t <= r["t1_s"])
            per_rows.append({**base, **r, "err_change_deg": _edge_change(t[sel], e[sel])})
    return summ, per_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", type=Path,
                    default=ROOT / f"data/analysis/icorr2027/snapshots/{_pins.dataset_record()}")
    ap.add_argument("--vstore", type=Path,
                    default=ROOT / f"data/analysis/jtsmm_validation/snapshots/{_pins.systems()}")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/pilot_study/jtsmm_data")
    ap.add_argument("--vdata-dir", type=Path, default=ROOT / "data/jtsmm_qualisys_delsys_eval")
    ap.add_argument("--cohort", type=Path, default=ROOT / "experiments/icorr2027/cohort.yaml")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data/analysis/icorr2027/audit/imu_dyncal.csv")
    args = ap.parse_args()
    jumps: list[dict] = []
    s1, p1 = dataset(args.store, args.data_dir, args.cohort, jumps)
    s2, p2 = validation(args.vstore, args.vdata_dir, jumps)
    summ, per, jumps = pd.DataFrame(s1 + s2), pd.DataFrame(p1 + p2), pd.DataFrame(jumps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out, index=False, float_format="%.4g")
    per.to_csv(args.out.with_name(args.out.stem + "_periods.csv"), index=False,
               float_format="%.4g")
    jumps.to_csv(args.out.with_name(args.out.stem + "_transitions.csv"), index=False,
                 float_format="%.4g")
    pd.set_option("display.width", 200)
    cols = ["study", "pid", "n_still", "still_s", "n_changes", "step_threshold_deg", "n_steps",
            "n_steps_aligned", "share_aligned_expected", "max_step_deg",
            "drift_in_still_dpm", "drift_outside_dpm", "rel_rate_median_dpm", "note"]
    print(summ[[c for c in cols if c in summ]].round(3).to_string(index=False))
    for study, g in summ.groupby("study"):
        g = g[g["n_changes"].fillna(0) > 0]
        if g.empty:
            continue
        n, k = int(g["n_steps"].sum()), int(g["n_steps_aligned"].sum())
        exp = float((g["share_aligned_expected"] * g["n_steps"]).sum())
        print(f"\n{study}: {len(g)} takes, {n} steps, {k} aligned with a still period "
              f"(expected without alignment {exp:.1f}); median drift in still "
              f"{g['drift_in_still_dpm'].median():.2f} deg/min, outside "
              f"{g['drift_outside_dpm'].median():.2f} deg/min")
    if not per.empty:
        for (study, ph), g in per.assign(kind=per["phase"].map(
                lambda x: "rest" if x == "rest" else "hold" if "point" in x or "tmj" in x
                else "gap" if x == "gap" else "task")).groupby(["study", "kind"]):
            print(f"{study:10s} {ph:5s} n={len(g):3d}  median {g['duration_s'].median():5.1f} s  "
                  f"relative IMU change median {g['rel_net_deg'].median():.2f} deg "
                  f"({g['rel_rate_dpm'].median():.1f} deg/min); error change over the period "
                  f"median |.| {g['err_change_deg'].abs().median():.3f} deg "
                  f"(n={int(g['err_change_deg'].notna().sum())})")
    if not jumps.empty:
        for study, g in jumps.groupby("study"):
            st = g["still_in_gap_s"] >= STILL_MIN_S
            print(f"{study:10s} transitions n={len(g)}: |residual jump| median "
                  f"{g['residual_deg'].abs().median():.2f} deg (after >= {STILL_MIN_S:g} s "
                  f"still: {g.loc[st, 'residual_deg'].abs().median():.2f}, n={int(st.sum())}; "
                  f"without: {g.loc[~st, 'residual_deg'].abs().median():.2f}, "
                  f"n={int((~st).sum())}); window SD median {g['window_sd_deg'].median():.2f}")
    d = summ[summ["study"] == "dataset"]
    if "anchor_heading_dpm" in d:
        print(f"dataset anchor heading drift (rest + idle-gap anchors): median "
              f"{d['anchor_heading_dpm'].median():.2f} deg/min (range "
              f"{d['anchor_heading_dpm'].min():.2f} to {d['anchor_heading_dpm'].max():.2f}), "
              f"residual SD median {d['anchor_resid_sd_deg'].median():.2f} deg, largest "
              f"residual {d['anchor_resid_max_deg'].max():.2f} deg")
    print(f"\nwritten: {args.out}, {args.out.stem}_periods.csv, {args.out.stem}_transitions.csv")


if __name__ == "__main__":
    main()
