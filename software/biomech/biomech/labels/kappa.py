# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Validity check of the automatic intent labels (stage ``labels_check``,
the planned kappa check of the C3b labels).

The ``labels`` stage labels every optical frame from the incisal-point velocity in the
maxillary frame (Butterworth 4 Hz + gradient, hysteresis at 20 % / 10 % of the phase's P95
speed, task-restricted classes, D16 axis scaling in free motion). No manual labels exist, so
the stored labels are compared with a **second, independent rule** (no ``auto_labels`` code)
on random windows per participant and phase:

* open_close / chewing: the optical **opening angle** (rotation, not incisor translation),
  differentiated with a Savitzky-Golay filter (order 2, window 0.25 s rounded up to an odd
  frame count: 11 frames at 40 Hz); OPEN when the opening rate
  exceeds +10 deg/s, CLOSE below -10 deg/s, else REST (fixed threshold, no hysteresis);
* protrusion_retrusion / left_right: the incisor velocity along the task axis,
  PROTRUSION / RETRUSION or LEFT / RIGHT beyond +-5 mm/s;
* free_motion: moving when the 3-D incisor speed exceeds 8 mm/s, the class from the
  component with the largest absolute speed (no per-axis normalisation);
* rest: REST unless the 3-D speed exceeds 8 mm/s (then the dominant-axis class);
* runs shorter than 0.10 s merge into the preceding run (as in the stage).

The thresholds were fixed before the first comparison (audit on r9). Variant ``rest_floor``
(declared after that run showed the fixed 3-D speed threshold inside the rest-phase
tracking noise of some participants): every threshold raised to the P99 of the same signal
in the participant's own rest phase, never lowered. Both variants are reported, with and
without the rest phase (the stage labels the whole rest phase REST by design).

Cohen's kappa over the frames labelled by both rules,

    kappa = (p_o - p_e) / (1 - p_e),   p_o = sum_k m_kk,   p_e = sum_k m_k. m_.k,

with ``m`` the normalised confusion matrix of the seven classes. Levels: per participant and
phase, per participant (phases pooled), per phase and overall (participants pooled), the
median / range of the per-participant kappas, kappa over all frames (no sampling), and a
boundary-tolerant kappa (frames within +-2 frames of a transition in either labelling left
out: onset timing vs class identity). Planned target (research plan): kappa >= 0.8.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from biomech.labels.auto_labels import CLASSES   # the stored label encoding only

C = {c: i for i, c in enumerate(CLASSES)}
FIXED = {"theta": 10.0, "x": 5.0, "y": 5.0, "speed": 8.0}   # deg/s, mm/s, mm/s, mm/s
MIN_RUN_S = 0.10
SG_S = 0.25
EDGE_FRAMES = 2          # +-2 frames (about +-50 ms at 40 Hz) around a label transition
VARIANTS = ("fixed", "rest_floor")
DEFAULTS: dict[str, Any] = {"windows": 10, "window_s": 2.0, "seed": 20261029}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa from the confusion matrix (frames labelled in both, label >= 0)."""
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return np.nan
    k = len(CLASSES)
    m = np.zeros((k, k))
    np.add.at(m, (a, b), 1)
    m /= m.sum()
    po = np.trace(m)
    pe = float(m.sum(1) @ m.sum(0))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def _fill(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x))
    cols = x if x.ndim > 1 else x[:, None]
    for j in range(cols.shape[1]):
        ok = np.isfinite(cols[:, j])
        if ok.sum() >= 2:
            cols[~ok, j] = np.interp(idx[~ok], idx[ok], cols[ok, j])
    return cols if x.ndim > 1 else cols[:, 0]


def _deriv(x: np.ndarray, fs: float) -> np.ndarray:
    w = max(5, int(round(SG_S * fs)) | 1)
    if len(x) <= w:
        return np.gradient(x, 1.0 / fs, axis=0)
    return savgol_filter(x, w, 2, deriv=1, delta=1.0 / fs, axis=0)


def near_transition(a: np.ndarray, b: np.ndarray, tol: int) -> np.ndarray:
    """Frames within ``tol`` frames of a label change in ``a`` or ``b``."""
    edge = np.zeros(a.size, bool)
    for s_ in (a, b):
        for i in np.flatnonzero(np.diff(s_) != 0):
            edge[max(i - tol, 0):min(i + 1 + tol + 1, a.size)] = True
    return edge


def _merge_runs(lab: np.ndarray, n_min: int) -> np.ndarray:
    out = lab.copy()
    i = 0
    while i < len(out):
        j = i
        while j < len(out) and out[j] == out[i]:
            j += 1
        if j - i < n_min and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def rest_floor(t_us: np.ndarray, inc: np.ndarray, theta: np.ndarray) -> dict[str, float]:
    """P99 of each rule signal in the rest phase (the participant's noise floor)."""
    fs = 1e6 / float(np.median(np.diff(t_us)))
    v = _deriv(_fill(inc), fs)
    w = _deriv(_fill(theta), fs)
    return {"theta": float(np.nanpercentile(np.abs(w), 99)),
            "x": float(np.nanpercentile(np.abs(v[:, 0]), 99)),
            "y": float(np.nanpercentile(np.abs(v[:, 1]), 99)),
            "speed": float(np.nanpercentile(np.linalg.norm(v, axis=1), 99))}


def second_rule(phase: str, t_us: np.ndarray, inc: np.ndarray, theta: np.ndarray,
                thr: dict[str, float] | None = None) -> np.ndarray:
    """Class index per frame by the second rule (-1: incisor not tracked)."""
    thr = thr or FIXED
    fs = 1e6 / float(np.median(np.diff(t_us)))
    tracked = np.isfinite(inc).all(1)
    lab = np.full(len(t_us), C["REST"], np.int8)
    v = _deriv(_fill(inc), fs)                       # mm/s, maxillary frame (x ant, y left, z up)
    if phase in ("open_close", "chewing"):
        w = _deriv(_fill(theta), fs)                 # deg/s, + opening
        lab[w > thr["theta"]] = C["OPEN"]
        lab[w < -thr["theta"]] = C["CLOSE"]
    elif phase == "protrusion_retrusion":
        lab[v[:, 0] > thr["x"]] = C["PROTRUSION"]
        lab[v[:, 0] < -thr["x"]] = C["RETRUSION"]
    elif phase == "left_right":
        lab[v[:, 1] > thr["y"]] = C["LEFT"]
        lab[v[:, 1] < -thr["y"]] = C["RIGHT"]
    else:                                            # free_motion, rest
        moving = np.linalg.norm(v, axis=1) > thr["speed"]
        dom = np.argmax(np.abs(v), axis=1)
        pos = {2: C["CLOSE"], 0: C["PROTRUSION"], 1: C["LEFT"]}   # +z up = closing
        neg = {2: C["OPEN"], 0: C["RETRUSION"], 1: C["RIGHT"]}
        for ax in range(3):
            sel = moving & (dom == ax)
            lab[sel & (v[:, ax] > 0)] = pos[ax]
            lab[sel & (v[:, ax] < 0)] = neg[ax]
    lab = _merge_runs(lab, max(1, int(round(MIN_RUN_S * fs))))
    lab[~tracked] = -1
    return lab


def _frames(a: np.ndarray, b: np.ndarray) -> int:
    return int(((a >= 0) & (b >= 0)).sum())


def run_variant(data: dict[str, tuple[dict, dict]], variant: str,
                params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Rows of one threshold variant; ``data``: pid -> (labels arrays, kinematics arrays)."""
    p = {**DEFAULTS, **(params or {})}
    rng = np.random.default_rng(int(p["seed"]))      # same windows for both variants
    rows: list[dict[str, Any]] = []
    smp: dict[str, list] = {"a": [], "b": [], "pid": [], "phase": []}
    alls: dict[str, list] = {"a": [], "b": [], "phase": [], "edge": []}
    for pid, (lab, kin) in data.items():
        thr = dict(FIXED)
        if variant == "rest_floor" and "rest__incisor" in kin:
            fl = rest_floor(kin["rest__t_us"].astype(np.int64),
                            kin["rest__incisor"].astype(float),
                            kin["rest__theta_raw_deg"].astype(float))
            thr = {k: max(FIXED[k], fl[k]) for k in FIXED}
        for ph in sorted(k.split("__")[0] for k in lab if k.endswith("__label")):
            t = lab[f"{ph}__t_us"].astype(np.int64)
            a = lab[f"{ph}__label"].astype(int)
            if f"{ph}__incisor" not in kin or not np.array_equal(kin[f"{ph}__t_us"], t):
                continue
            b = second_rule(ph, t, kin[f"{ph}__incisor"].astype(float),
                            kin[f"{ph}__theta_raw_deg"].astype(float), thr).astype(int)
            alls["a"].append(a)
            alls["b"].append(b)
            alls["phase"].append(np.full(a.size, ph))
            alls["edge"].append(near_transition(a, b, EDGE_FRAMES))
            fs = 1e6 / float(np.median(np.diff(t)))
            n_w = int(round(float(p["window_s"]) * fs))
            starts = (np.array([0]) if len(t) <= n_w
                      else rng.integers(0, len(t) - n_w, int(p["windows"])))
            idx = np.unique(np.concatenate([np.arange(s0, min(s0 + n_w, len(t)))
                                            for s0 in starts]))
            ai, bi = a[idx], b[idx]
            ok = (ai >= 0) & (bi >= 0)
            smp["a"].append(ai)
            smp["b"].append(bi)
            smp["pid"].append(np.full(idx.size, pid))
            smp["phase"].append(np.full(idx.size, ph))
            rows.append({"variant": variant, "scope": "phase", "level": "participant_phase",
                         "pid": pid, "phase": ph, "n_frames": int(ok.sum()),
                         "kappa": kappa(ai, bi),
                         "agreement": float(np.mean(ai[ok] == bi[ok])) if ok.any() else np.nan,
                         "kappa_all_frames": kappa(a, b),
                         "thr_theta_dps": thr["theta"], "thr_x_mm_s": thr["x"],
                         "thr_y_mm_s": thr["y"], "thr_speed_mm_s": thr["speed"]})
    if not smp["a"]:
        return rows
    A, B = np.concatenate(smp["a"]), np.concatenate(smp["b"])
    P, PH = np.concatenate(smp["pid"]), np.concatenate(smp["phase"])
    AA, BB, PP = (np.concatenate(alls[k]) for k in ("a", "b", "phase"))
    for scope, keep, keep_all in (("all_phases", np.ones(A.size, bool), np.ones(AA.size, bool)),
                                  ("without_rest", PH != "rest", PP != "rest")):
        per_pid = []
        for pid in data:
            k = (P == pid) & keep
            if not k.any():
                continue
            kp = kappa(A[k], B[k])
            per_pid.append(kp)
            rows.append({"variant": variant, "scope": scope, "level": "participant",
                         "pid": pid, "phase": "pooled", "n_frames": _frames(A[k], B[k]),
                         "kappa": kp})
        rows.append({"variant": variant, "scope": scope, "level": "overall_sampled",
                     "pid": "", "phase": "pooled", "n_frames": _frames(A[keep], B[keep]),
                     "kappa": kappa(A[keep], B[keep]),
                     "kappa_median_participants": float(np.median(per_pid)),
                     "kappa_min_participants": float(np.min(per_pid)),
                     "kappa_max_participants": float(np.max(per_pid)),
                     "n_participants": len(per_pid)})
        rows.append({"variant": variant, "scope": scope, "level": "overall_all_frames",
                     "pid": "", "phase": "pooled",
                     "n_frames": _frames(AA[keep_all], BB[keep_all]),
                     "kappa": kappa(AA[keep_all], BB[keep_all])})
    # boundary-tolerant kappa: frames within +-EDGE_FRAMES of a transition in either labelling
    # left out, so onset/offset timing is separated from class identity (all frames, no rest)
    E = np.concatenate(alls["edge"])
    for ph in ["pooled", *sorted(set(PP) - {"rest"})]:
        k = ~E & ((PP != "rest") if ph == "pooled" else (PP == ph))
        rows.append({"variant": variant, "scope": "boundary_tolerant", "level":
                     "boundary_tolerant", "pid": "", "phase": ph,
                     "n_frames": _frames(AA[k], BB[k]), "kappa": kappa(AA[k], BB[k])})
    for ph in sorted(set(PH)):
        k = PH == ph
        rows.append({"variant": variant, "scope": "phase", "level": "phase", "pid": "",
                     "phase": ph, "n_frames": _frames(A[k], B[k]), "kappa": kappa(A[k], B[k])})
    return rows


def run_labels_check(data: dict[str, tuple[dict, dict]],
                     params: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Table ``kappa``: every level of both threshold variants (see the module docstring)."""
    rows = [r for v in VARIANTS for r in run_variant(data, v, params)]
    return {"kappa": pd.DataFrame(rows)}
