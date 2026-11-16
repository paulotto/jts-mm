# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical + IMU orientation fusion for the validation study (V4 amended).

Adapter between a recorded take and :mod:`jtsmm.imu.fusion`. For the head plate (HP,
optical body ``cra``, IMU ``imu_cra``) and the mandible plate (MP, ``mta``, ``imu_mta``) it
returns, at the caller's query times (optical frame times and/or the 200-Hz reference
times between frames):

* ``F0`` optical only (SLERP between valid frames; ``F0_hold`` = last valid frame),
* ``F1`` SO(3) complementary filter (causal),
* ``F2`` error-state Kalman filter + RTS smoother (``F2_causal`` = the forward filter),
* ``F2P`` F2 orientation with the position fused from the accelerometer
  (:mod:`jtsmm.imu.position`: gravity removed with the chip orientation, Kalman filter +
  RTS smoother on position / velocity / bias; ``F2P_causal`` = both forward only),

plus the calibration (Z, X, lag, drift, residuals), the settings derived from the sensors,
and the gating / NIS diagnostics. F0–F2 fuse **rotations only**; their position stays
optical (linear interpolation between valid frames, with the bracketing gap reported so
that the caller applies its own no-bridging rule). F2P also fuses the position
(VD6; secondary result, VD8).

Rules (R5c.6): the calibration and lag are fitted on the caller's training mask only (the
CT sync windows plus the training tasks of a leave-one-task-out fold), which must be
disjoint from the evaluated interval (asserted). The filter settings come from sensor
characteristics (:func:`jtsmm.imu.fusion.settings_from_sensors`): the rest phase of the
JTS-MM optical and IMU streams, the calibration's drift and increment error — no
reference data, no tuning.

Typical use (one leave-one-task-out fold)::

    ctx = prepare(session, params, train_mask_fn=lambda t: mask_of(t, sync + train_tasks),
                  eval_mask_fn=lambda t: mask_of(t, [held]))
    out = fuse(ctx, t_query_s)                      # or run_fusion(...) for both at once
    out["HP"]["R"]["F2"], out["MP"]["T"]["F1"], out["HP"]["calibration"], ...
    drops = dropout_masks(ctx.t_frames_s, [(held.t0, held.t1)])
    out_1s = fuse(ctx, t_query_s, drop_mask=drops[1.0])

Times are master-clock seconds at the optical exposure centre
(:func:`biomech.validation.sync.frame_times`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jtsmm.imu import fusion as FU
from jtsmm.imu import position as PO

from biomech.imu.preprocess import GAP_S as IMU_GAP_S
from biomech.validation import selection as SEL
from biomech.validation.sync import frame_times

DROPOUT_LENGTHS_S: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
ESTIMATORS: tuple[str, ...] = ("F0", "F0_hold", "F1", "F2", "F2_causal", "F2P", "F2P_causal")

DEFAULTS: dict[str, Any] = {
    "bodies": {"HP": "cra", "MP": "mta"},       # validation name -> JTS-MM optical body
    "imu_slots": {"HP": "cra", "MP": "mta"},    # validation name -> SessionData.imu slot
    "exposure_s": 0.012,
    "imu_max_gap_s": IMU_GAP_S,      # qc.imu.gap_s (E33.7)
    "rest_phase": "rest",
    "rest_trim_s": 0.5,
    "gate_prob": 0.999,
    "reset_s": 1.0,
    "lag_range_s": (-0.05, 0.08),
    "fit_drift": None,                          # None: when the training span >= 60 s
    "lag_mode": "pooled",                       # "pooled" | "per_body" | a fixed lag (s)
    "noise_source": "train_structure",          # "train_structure" | "rest"
    "seed": 0,
}

MaskFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class BodyContext:
    name: str
    body: str
    slot: str
    R_opt: np.ndarray                 # (N, 3, 3) projected, NaN untracked
    p_opt: np.ndarray                 # (N, 3) mm, NaN untracked
    quality: dict[str, np.ndarray]
    imu: FU.ImuTrack
    calibration: FU.Calibration
    settings: FU.FusionSettings
    imu_info: dict[str, Any] = field(default_factory=dict)
    # position fusion (F2P): gravity-free accelerometer in the optical world and settings
    acc_t_s: np.ndarray | None = None
    acc_mm_s2: np.ndarray | None = None
    pos_settings: Any = None


@dataclass
class FusionContext:
    """What :func:`prepare` fitted: frame times, the training/evaluation masks and, per
    body, the optical data, the IMU track, the calibration and the settings."""

    t_frames_s: np.ndarray
    train_mask: np.ndarray
    eval_mask: np.ndarray | None
    rest_intervals: list[tuple[float, float]]
    bodies: dict[str, BodyContext]
    params: dict[str, Any]


def _imu_track(session: Any, slot: str, max_gap_s: float) -> FU.ImuTrack | None:
    imu = session.imu(slot)
    if imu is None or getattr(imu, "n", len(imu.t_us)) == 0:
        return None
    return FU.ImuTrack.from_stream(np.asarray(imu.t_us), np.asarray(imu.quat, np.float64),
                                   getattr(imu, "quat_accuracy", None), max_gap_s)


def imu_coverage(track: FU.ImuTrack, intervals: Sequence[SEL.Interval],
                 max_gap_s: float) -> dict[str, Any]:
    """Rate and coverage of the sanitized stream overall and per interval: the share of
    the interval inside gap-free IMU segments and the longest hole."""
    t = track.t_s
    out: dict[str, Any] = {"rate_hz": track.rate_hz(), "n": int(len(t)),
                           "n_dropped": int(track.n_dropped),
                           "n_gaps": int(np.sum(np.diff(track.t_us) > max_gap_s * 1e6)),
                           "per_segment": {}}
    for iv in intervals:
        grid = np.arange(iv.t0, iv.t1, 0.005)
        k = (t >= iv.t0) & (t <= iv.t1)
        tk = t[k]
        out["per_segment"][iv.name] = {
            "rate_hz": float((len(tk) - 1) / max(tk[-1] - tk[0], 1e-9)) if len(tk) > 1
            else 0.0,
            "coverage": float(np.mean(track.segment(grid) >= 0)) if len(grid) else np.nan,
            "max_hole_ms": float(np.max(np.diff(tk)) * 1e3) if len(tk) > 1 else np.nan}
    return out


def prepare(session: Any, params: Mapping[str, Any] | None, train_mask_fn: MaskFn,
            eval_mask_fn: MaskFn | None = None) -> FusionContext:
    """Calibrate and derive the settings for HP and MP.

    ``train_mask_fn(t_frames_s) -> bool (N,)`` selects the training frames (CT sync
    windows + training tasks of the fold); ``eval_mask_fn`` the evaluated frames. Raises
    ``ValueError`` when they overlap or a body has no IMU.
    """
    p = {**DEFAULTS, **dict(params or {})}
    opt = session.optical()
    if opt is None:
        raise LookupError("session has no optical data")
    t = frame_times(opt.t_us, p["exposure_s"])
    train = np.asarray(train_mask_fn(t), bool)
    ev = None if eval_mask_fn is None else np.asarray(eval_mask_fn(t), bool)
    if train.shape != t.shape:
        raise ValueError("train_mask_fn must return one flag per optical frame")
    if ev is not None and np.any(train & ev):
        raise ValueError(f"training and evaluation frames overlap ({int(np.sum(train & ev))}"
                         " frames): the calibration must not see the evaluated interval")
    ct = opt.poses.get("ct")
    ct_ok = (np.isfinite(ct.reshape(len(ct), -1)).all(axis=1) if ct is not None
             else np.zeros(len(t), bool))
    rest = SEL.movement_intervals(session.phases, t, ct_ok, tasks=(p["rest_phase"],),
                                  trim_s=p["rest_trim_s"])
    rest_iv = [(r.t0, r.t1) for r in rest]
    if not rest_iv:
        raise LookupError(f"no {p['rest_phase']!r} phase for the noise settings")
    segs = SEL.movement_intervals(session.phases, t, ct_ok) + SEL.sync_windows(
        session.phases, t, ct_ok)
    data: dict[str, tuple] = {}
    for name, body in p["bodies"].items():
        T = np.asarray(opt.poses[body], dtype=np.float64)
        R, ok = FU.valid_rotations(T[:, :3, :3])
        pos = np.where(ok[:, None], T[:, :3, 3], np.nan)
        track = _imu_track(session, p["imu_slots"][name], p["imu_max_gap_s"])
        if track is None:
            raise LookupError(f"no IMU stream for {name}")
        q = {k: np.asarray(v) for k, v in opt.quality.get(body, {}).items()}
        data[name] = (body, R, pos, track, q)

    mounting = dict(p.get("imu_mounting") or {})          # name -> R_mount (body<-chip)

    def cal_of(name: str, lag: float | None) -> FU.Calibration:
        _, R, _, track, _ = data[name]
        Xf = np.asarray(mounting[name], float).T if name in mounting else None   # chip<-body
        return FU.calibrate(t, R, track, train, lag_range_s=tuple(p["lag_range_s"]),
                            fit_drift=p["fit_drift"], lag_s=lag, seed=p["seed"], X_fixed=Xf)

    mode = p["lag_mode"]
    fixed = None if isinstance(mode, str) else float(mode)
    cals = {name: cal_of(name, fixed) for name in data}
    pooled = {"mode": mode if isinstance(mode, str) else "fixed",
              "per_body_ms": {n: c.lag_s * 1e3 for n, c in cals.items()},
              "per_body_se_ms": {n: c.lag_se_s * 1e3 for n, c in cals.items()}}
    if mode == "pooled" and len(cals) > 1:
        # one chip type and stamping path: inverse-variance weighted mean of the body lags
        # (a body that barely rotates, e.g. the head, has a large SE and little weight)
        lag = np.array([c.lag_s for c in cals.values()])
        se = np.array([c.lag_se_s for c in cals.values()])
        w = np.where(np.isfinite(se) & (se > 0), 1.0 / np.maximum(se, 1e-4) ** 2, 0.0)
        if w.sum() > 0:
            lag_p = float(np.sum(w * lag) / w.sum())
            pooled.update(lag_ms=lag_p * 1e3, se_ms=float(1e3 / np.sqrt(w.sum())))
            cals = {name: cal_of(name, lag_p) for name in data}
            for c in cals.values():
                c.lag_se_s = pooled["se_ms"] * 1e-3
    bodies: dict[str, BodyContext] = {}
    for name, (body, R, pos, track, q) in data.items():
        cal = cals[name]
        st = FU.settings_from_sensors(
            t, R, track, cal, rest_iv, q, gate_prob=p["gate_prob"], reset_s=p["reset_s"],
            train_mask=train if p["noise_source"] == "train_structure" else None)
        info = imu_coverage(track, segs, p["imu_max_gap_s"])
        info["lag"] = pooled
        acc_t = acc_a = pst = None
        acc = getattr(session.imu(p["imu_slots"][name]), "accel", None)
        if acc is not None and getattr(acc, "n", len(acc.t_us)) > 100:
            acc_t, acc_a = PO.world_acceleration(cal, track, np.asarray(acc.t_us, float) * 1e-6,
                                                 np.asarray(acc.data, np.float64))
            pst = PO.settings_from_data(t, pos, train, rest_iv, acc_t, acc_a)
        bodies[name] = BodyContext(name, body, p["imu_slots"][name], R, pos, q, track, cal, st,
                                   info, acc_t, acc_a, pst)
    return FusionContext(t, train, ev, rest_iv, bodies, p)


def load_mounting(path: str | Any, slot: str) -> np.ndarray | None:
    """``R_mount`` (body←chip, 3x3) of IMU ``slot`` (``mta`` / ``cra``) from a persisted
    hand-eye calibration (``jtsmm imu calibrate --persist-imu-calibration``:
    ``imu_mounting.json``, or a ``session.h5`` with ``/calibration/imu_<slot>``). Only an
    accepted, proper rotation is returned (``jtsmm.imu.mounting.load_mounting_rotation``)."""
    import json
    from pathlib import Path

    from jtsmm.imu.mounting import load_mounting_rotation

    path = Path(path)
    if path.suffix == ".h5":
        return load_mounting_rotation(path, slot)
    doc = json.loads(path.read_text(encoding="utf-8"))
    key = slot if slot.startswith("imu_") else f"imu_{slot}"
    for k in (key, slot, key[4:]):
        for container in (doc, doc.get("slots") or {}, doc.get("results") or {}):
            e = container.get(k) if isinstance(container, dict) else None
            if not isinstance(e, dict) or e.get("accepted", True) is False:
                continue
            if "R_mount" in e:
                R = np.asarray(e["R_mount"], float)
            elif "T" in e:
                R = np.asarray(e["T"], float)[:3, :3]
            else:
                continue
            if R.shape == (3, 3) and np.isfinite(R).all() and \
                    np.allclose(R.T @ R, np.eye(3), atol=1e-4) and np.linalg.det(R) > 0.5:
                return R
    return None


def _interp_position(t: np.ndarray, p: np.ndarray, tq: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(p).all(axis=1)
    tv, pv = t[ok], p[ok]
    out = np.full((len(tq), 3), np.nan)
    gap = np.full(len(tq), np.nan)
    if len(tv) < 2:
        return out, gap
    j = np.searchsorted(tv, tq, side="right") - 1
    m = (j >= 0) & (j < len(tv) - 1) | (tq == tv[-1])
    jc = np.clip(j, 0, len(tv) - 2)
    f = np.clip((tq - tv[jc]) / (tv[jc + 1] - tv[jc]), 0.0, 1.0)
    out[m] = pv[jc[m]] + f[m, None] * (pv[jc[m] + 1] - pv[jc[m]])
    gap[m] = tv[jc[m] + 1] - tv[jc[m]]
    return out, gap


def _pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.full((len(R), 4, 4), np.nan)
    T[:, :3, :3] = R
    T[:, :3, 3] = p
    T[:, 3, :] = (0.0, 0.0, 0.0, 1.0)
    bad = ~(np.isfinite(R.reshape(len(R), -1)).all(1) & np.isfinite(p).all(1))
    T[bad] = np.nan
    return T


def fuse(ctx: FusionContext, t_query_s: np.ndarray | None = None,
         drop_mask: np.ndarray | None = None,
         estimators: Sequence[str] = ESTIMATORS,
         pos_window: tuple[float, float] | None = None) -> dict[str, Any]:
    """Run the estimators with the fitted calibration and settings.

    ``drop_mask`` (N,) removes optical frames (simulated dropouts, from
    :func:`dropout_masks`) from every estimator's input, including F0; the calibration is
    not refitted (it never saw the evaluated interval). ``pos_window`` (s) limits the
    position filter of F2P to that span (its pose is NaN outside). Returns::

        {"t_query_s", "t_frames_s", "drop_mask",
         "HP"/"MP": {"body", "slot",
                     "R": {estimator: (M, 3, 3)}, "T": {estimator: (M, 4, 4)},
                     "p": (M, 3) optical position (linear), "p_gap_s": (M,) bracketing gap,
                     "calibration": dict, "settings": dict, "imu": coverage dict,
                     "diagnostics": {"F1": {...}, "F2": {..., "nis": (N,), "gated": (N,)}}}}
    """
    t = ctx.t_frames_s
    tq = t if t_query_s is None else np.asarray(t_query_s, dtype=np.float64)
    drop = None if drop_mask is None else np.asarray(drop_mask, bool)
    out: dict[str, Any] = {"t_query_s": tq, "t_frames_s": t, "drop_mask": drop}
    for name, b in ctx.bodies.items():
        R_in = b.R_opt if drop is None else np.where(drop[:, None, None], np.nan, b.R_opt)
        p_in = b.p_opt if drop is None else np.where(drop[:, None], np.nan, b.p_opt)
        Rs: dict[str, np.ndarray] = {}
        diag: dict[str, Any] = {}
        if "F0" in estimators:
            Rs["F0"] = FU.f0_optical(t, R_in, tq, "slerp")
        if "F0_hold" in estimators:
            Rs["F0_hold"] = FU.f0_optical(t, R_in, tq, "hold")
        if "F1" in estimators:
            r1 = FU.f1_complementary(t, R_in, b.imu, b.calibration, b.settings, tq,
                                     quality=b.quality)
            Rs["F1"] = r1.R_query
            diag["F1"] = {**r1.diagnostics, "gated": r1.gated, "innov_deg": r1.innov_deg}
        want_p = {"F2P", "F2P_causal"} & set(estimators) and b.pos_settings is not None
        if "F2" in estimators or "F2_causal" in estimators or want_p:
            r2 = FU.f2_kalman(t, R_in, b.imu, b.calibration, b.settings, tq,
                              quality=b.quality,
                              smooth=bool({"F2", "F2P"} & set(estimators)))
            if "F2" in estimators:
                Rs["F2"] = r2.R_query_smooth
            if "F2_causal" in estimators:
                Rs["F2_causal"] = r2.R_query
            if want_p:
                Rs_p = {"F2P": r2.R_query_smooth, "F2P_causal": r2.R_query}
            d2 = {k: v for k, v in r2.diagnostics.items() if not k.startswith("R_frames")}
            diag["F2"] = {**d2, "gated": r2.gated, "nis": r2.nis, "innov_deg": r2.innov_deg,
                          "resets": r2.resets}
        p, gap = _interp_position(t, p_in, tq)
        T = {k: _pose(v, p) for k, v in Rs.items()}
        if want_p:
            ps, pf = PO.position_kf(t, p_in, np.isfinite(p_in).all(axis=1), b.acc_t_s,
                                    b.acc_mm_s2, b.pos_settings, tq, window=pos_window)
            for k, pp in (("F2P", ps), ("F2P_causal", pf)):
                if k in estimators:
                    Rs[k] = Rs_p[k]
                    T[k] = _pose(Rs_p[k], pp)
        out[name] = {"body": b.body, "slot": b.slot, "R": Rs, "T": T,
                     "p": p, "p_gap_s": gap,
                     "calibration": b.calibration.to_dict(), "settings": b.settings.to_dict(),
                     "pos_settings": (b.pos_settings.to_dict() if b.pos_settings is not None
                                      else None),
                     "imu": b.imu_info, "diagnostics": diag}
    return out


def run_fusion(session: Any, params: Mapping[str, Any] | None, train_mask_fn: MaskFn, *,
               t_query_s: np.ndarray | None = None, eval_mask_fn: MaskFn | None = None,
               drop_mask: np.ndarray | None = None) -> dict[str, Any]:
    """:func:`prepare` + :func:`fuse` in one call (default query times: the optical frame
    times). The result also carries ``"train_mask"`` and ``"eval_mask"``."""
    ctx = prepare(session, params, train_mask_fn, eval_mask_fn)
    out = fuse(ctx, t_query_s, drop_mask)
    out["train_mask"], out["eval_mask"] = ctx.train_mask, ctx.eval_mask
    return out


def dropout_masks(t_s: np.ndarray, intervals: Sequence[tuple[float, float]] |
                  Sequence[SEL.Interval], lengths: Sequence[float] = DROPOUT_LENGTHS_S, *,
                  spacing_s: float = 1.0, margin_s: float = 0.25) -> dict[float, np.ndarray]:
    """Simulated optical dropouts: for each length, gaps of that length placed back to
    back inside each interval (``margin_s`` from its ends, ``spacing_s`` of untouched frames
    between gaps so the filter re-converges). Returns ``{length: bool (N,)}`` over ``t_s``;
    deterministic (no randomness)."""
    t = np.asarray(t_s, dtype=np.float64)
    ivs = [(iv.t0, iv.t1) if isinstance(iv, SEL.Interval) else (float(iv[0]), float(iv[1]))
           for iv in intervals]
    out: dict[float, np.ndarray] = {}
    for L in lengths:
        m = np.zeros(len(t), bool)
        for a, b in ivs:
            s = a + margin_s
            while s + L <= b - margin_s:
                m |= (t >= s) & (t < s + L)
                s += L + spacing_s
        out[float(L)] = m
    return out


def frame_errors_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between two rotation stacks (NaN where either is missing)."""
    return FU.rot_error_deg(Ra, Rb)


__all__ = ["DEFAULTS", "DROPOUT_LENGTHS_S", "ESTIMATORS", "BodyContext", "FusionContext",
           "dropout_masks", "frame_errors_deg", "fuse", "imu_coverage", "prepare",
           "run_fusion"]
