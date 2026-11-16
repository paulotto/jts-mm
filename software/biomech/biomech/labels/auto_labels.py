# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Intent labels from optical kinematics (C3b ground truth, an05 §4 / docs/ml/ml01).

Labels describe the *direction of jaw motion* at each instant, derived from the
incisal-point velocity in the maxillary frame (x anterior, y left, z up):

=====================  ===================================================
phase                  allowed classes (axis)
=====================  ===================================================
rest                   REST
open_close, chewing    REST, OPEN (-z), CLOSE (+z)
protrusion_retrusion   REST, PROTRUSION (+x), RETRUSION (-x)
left_right             REST, LEFT (+y), RIGHT (-y)
free_motion            all; the axis with the largest normalized speed wins
=====================  ===================================================

Motion vs REST uses hysteresis on the speed along the task axis: enter above
``hi_frac`` and leave below ``lo_frac`` of the phase's 95th-percentile speed;
segments shorter than ``min_seg_s`` are merged into their neighbours.

In ``free_motion`` every axis' speed is divided by its scale before the axes compete
(D16): the axis' own P95 speed, floored at ``axis_scale_floor`` (default 0.25) x the
largest axis scale of the phase. Without the floor (``axis_scale_floor: 0``, the
pre-D16 rule and the ``labels_p95`` sensitivity) a hardly used axis is divided by its
own small P95, its noise is amplified to full scale and it produces spurious
LEFT / RIGHT / PROTRUSION labels. :func:`label_agreement` compares two labellings. Labels are
causal-agnostic ground truth (computed offline with zero-phase smoothing), the
classifiers that use them are not.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

CLASSES = ("REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT")
CODE = {c: i for i, c in enumerate(CLASSES)}
AXES = {  # phase -> list of (axis index, sign, positive class, negative class)
    "open_close": [(2, -1.0, "OPEN", "CLOSE")],
    "chewing": [(2, -1.0, "OPEN", "CLOSE")],
    "protrusion_retrusion": [(0, 1.0, "PROTRUSION", "RETRUSION")],
    "left_right": [(1, 1.0, "LEFT", "RIGHT")],
    "free_motion": [(2, -1.0, "OPEN", "CLOSE"), (0, 1.0, "PROTRUSION", "RETRUSION"),
                    (1, 1.0, "LEFT", "RIGHT")],
}


def _velocity(t_us: np.ndarray, p: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    x = p.copy()
    idx = np.arange(len(x))
    for k in range(3):
        ok = np.isfinite(x[:, k])
        if ok.sum() >= 2:
            x[~ok, k] = np.interp(idx[~ok], idx[ok], x[ok, k])
    if len(x) > 15:
        sos = butter(2, cutoff / (0.5 * fs), output="sos")
        x = sosfiltfilt(sos, x, axis=0)
    v = np.gradient(x, t_us * 1e-6, axis=0)
    v[~np.isfinite(p).all(1)] = np.nan
    return v


def _hysteresis(speed: np.ndarray, hi: float, lo: float) -> np.ndarray:
    moving = np.zeros(speed.shape, bool)
    state = False
    for i, s in enumerate(speed):
        if not np.isfinite(s):
            state = False
        elif state and s < lo:
            state = False
        elif not state and s > hi:
            state = True
        moving[i] = state
    return moving


def _merge_short(lab: np.ndarray, n_min: int) -> np.ndarray:
    out = lab.copy()
    changed = True
    while changed:
        changed = False
        b = np.flatnonzero(np.diff(out) != 0) + 1
        starts = np.concatenate(([0], b))
        ends = np.concatenate((b, [len(out)]))
        for s, e in zip(starts, ends):
            if e - s < n_min and len(starts) > 1:
                left = out[s - 1] if s > 0 else None
                right = out[e] if e < len(out) else None
                fill = left if left is not None else right
                if fill is not None and fill != out[s]:
                    out[s:e] = fill
                    changed = True
                    break
    return out


AXIS_SCALE_FLOOR = 0.25      # D16 default of ``stages.labels.axis_scale_floor``


def axis_scales(refs: list[float] | np.ndarray, floor: float = AXIS_SCALE_FLOOR) -> np.ndarray:
    """Per-axis speed scales: own P95, floored at ``floor`` x the largest finite one
    (``floor`` = 0: own P95 unchanged). A NaN scale (axis never tracked) stays NaN."""
    r = np.asarray(refs, dtype=np.float64).copy()
    fin = np.isfinite(r)
    if floor > 0 and fin.any():
        r[fin] = np.maximum(r[fin], float(floor) * float(np.max(r[fin])))
    return r


def label_phase(phase: str, t_us: np.ndarray, incisor: np.ndarray, fs: float,
                p: dict[str, Any]) -> np.ndarray:
    """int8 class codes per sample; -1 where the incisal point is untracked.

    ``p``: ``vel_cutoff_hz``, ``hi_frac``, ``lo_frac``, ``min_seg_s`` and optionally
    ``axis_scale_floor`` (free_motion axis scales, :func:`axis_scales`; default
    :data:`AXIS_SCALE_FLOOR`)."""
    n = len(t_us)
    lab = np.full(n, CODE["REST"], dtype=np.int8)
    if phase not in AXES:
        lab[~np.isfinite(incisor).all(1)] = -1
        return lab
    v = _velocity(t_us, incisor, fs, p["vel_cutoff_hz"])
    comps = []
    for axis, sgn, pos, neg in AXES[phase]:
        s = sgn * v[:, axis]
        ref = np.nanpercentile(np.abs(s), 95) if np.isfinite(s).any() else np.nan
        comps.append((s, ref, pos, neg))
    if len(comps) == 1:
        s, ref, pos, neg = comps[0]
        moving = _hysteresis(np.abs(s), p["hi_frac"] * ref, p["lo_frac"] * ref)
        lab[moving & (s > 0)] = CODE[pos]
        lab[moving & (s < 0)] = CODE[neg]
    else:
        scale = axis_scales([r for _, r, _, _ in comps],
                            float(p.get("axis_scale_floor", AXIS_SCALE_FLOOR)))
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.stack([np.abs(s) / r for (s, _, _, _), r in zip(comps, scale)], axis=1)
        best = np.nanargmax(np.nan_to_num(norm, nan=-1.0), axis=1)
        speed = np.nanmax(norm, axis=1)
        moving = _hysteresis(speed, p["hi_frac"], p["lo_frac"])
        for k, (s, _, pos, neg) in enumerate(comps):
            sel = moving & (best == k)
            lab[sel & (s > 0)] = CODE[pos]
            lab[sel & (s < 0)] = CODE[neg]
    lab = _merge_short(lab, max(int(round(p["min_seg_s"] * fs)), 1))
    lab[~np.isfinite(incisor).all(1)] = -1
    return lab


def label_agreement(old: np.ndarray, new: np.ndarray) -> dict[str, Any]:
    """Agreement of two labellings of the same samples (e.g. pre-D16 vs D16).

    Over the samples labelled in both (codes >= 0): ``n``, ``changed_frac`` (share of
    samples whose class differs), ``kappa`` and per class (keyed by class name) the
    sample counts ``n_old`` / ``n_new`` and ``changed_frac`` = share of the class's
    *old* samples that now carry another class (NaN for a class absent before)."""
    a, b = np.asarray(old), np.asarray(new)
    if a.shape != b.shape:
        raise ValueError(f"labellings differ in length: {a.shape} vs {b.shape}")
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    per = {}
    for code, name in enumerate(CLASSES):
        sel = a == code
        per[name] = {"n_old": int(sel.sum()), "n_new": int(np.sum(b == code)),
                     "changed_frac": float(np.mean(b[sel] != code)) if sel.any() else np.nan}
    return {"n": int(a.size), "changed_frac": float(np.mean(a != b)) if a.size else np.nan,
            "kappa": cohen_kappa(a, b), "per_class": per}


def compare_axis_scale(phase: str, t_us: np.ndarray, incisor: np.ndarray, fs: float,
                       p: dict[str, Any], old_floor: float = 0.0
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Labels with ``old_floor`` (default 0: pre-D16) and with ``p``'s
    ``axis_scale_floor`` (D16), and their :func:`label_agreement`."""
    old = label_phase(phase, t_us, incisor, fs, {**p, "axis_scale_floor": old_floor})
    new = label_phase(phase, t_us, incisor, fs, p)
    return old, new, label_agreement(old, new)


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Agreement of two label sequences (e.g. auto vs manual); -1 entries ignored."""
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return np.nan
    cats = np.union1d(a, b)
    po = float(np.mean(a == b))
    pe = float(sum(np.mean(a == c) * np.mean(b == c) for c in cats))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
