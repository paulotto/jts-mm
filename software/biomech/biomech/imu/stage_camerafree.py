# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stages of the camera-free study X5 in the dataset store (dataset plan X5, D28).

``camerafree`` (per participant; depends on qc, kinematics, imu, labels) runs the estimator
ladder of :mod:`biomech.imu.camerafree` on one 200 Hz grid over the whole take (first to
last experiment phase) and scores it against the optics:

* **E0** the optics (reference): the kinematics jaw pose ``{phase}__T`` (40 Hz, already
  through the dataset pose chain, 6 Hz zero-phase low-pass), relative to the optical closed
  pose of the rest phase (:func:`closed_pose`; every camera-free rung refers to the rest
  closed pose, so a reference at the open_close cycle start, ``T_closed``, would add a
  constant offset that is no IMU error), put on the grid by linear interpolation of rotation
  vectors and points (no interpolation over more than 2.5 optical frame intervals);
* **E1** the in-take 7-parameter calibration of the ``imu`` stage (``imu:calibration``,
  fitted on free_motion), read back -- the same model as C3a (``e1_vs_c3a_max_deg`` in
  ``cf_summary`` checks the opening against ``imu`` ``{phase}__theta_quat``);
* **E2a / E2b / E3a / E3b** (hold and slerp): :func:`biomech.imu.camerafree.rotation_ladder`
  with the hand-eye mountings and the landmark frame ``T_max_hp``; E2a's ψ from the rest
  phase against the optical ``T_closed`` (as H4.2 states);
* **E4hinge / E4own / E5 / E6** on the primary E3 (``primary_e3``, default E3b: the
  camera-free ψ, which falls back to E2a's when ψ is not identifiable, H4.8); E4own's ridge
  model on the optical calibration block (free_motion), E5/E6 accelerometer bridging with the
  closed-pose anchors (smoother; ``_causal``: the forward filter);
* E4gen (leave-one-subject-out over the optical couplings of every participant with optics)
  and E7 live in ``camerafree_cohort``;
* **X5b** (``cf_decomposition``, oracle decomposition of E4own's incisor error): optical
  rotation + the E4own model driven by it (translation-model error alone), the primary E3
  + the optical condyle-midpoint translation (rotation-induced error alone), next to the full
  E4own (the ``cf_translation`` rows, copied);
* **X5c** (``cf_task_model``, task-conditioned translation): the E4own family fitted per
  movement class on the calibration frames whose optical labels (``labels`` stage) fall in
  that class (``x5c_task_labels``; at least ``x5c_min_cal_s`` of them), applied to each
  scored task with its true task label (oracle class: the upper bound of the deferred E7).
  Both are reported next to E4own, never as camera-free results.

Every estimate goes through the ground truth's low-pass (``signals.kinematics``) before it is
scored; the optics are not filtered again. Scored tasks: every experiment phase except
``exclude_scored`` (rest; free_motion, the calibration block of E1 and E4own, is reported
with ``scored`` False). The landmark holds are not experiment phases.

**Calibration inputs** (D28; parameters in the study file's ``stages.camerafree`` block,
paths relative to the study file):

* ``mounting``: the hand-eye calibration (a ``session_imu_mounting_*/session.h5`` with
  ``/calibration/imu_<slot>``, or the stable ``imu_mounting.json`` of ``jtsmm imu
  calibrate``); ``mounting_by_pid`` overrides it per participant. Read with
  :mod:`jtsmm.imu.mounting` (accepted, proper rotations only) -- the functions
  ``validation.fusion.load_mounting`` wraps; importing that module would put the validation
  sync code into the dataset study's code hash;
* lever arms (E5/E6 only): ``lever: session`` estimates them from the same hand-eye session
  (the VD15 method, :mod:`biomech.imu.lever`, primary cut-off, no bootstrap) -- the
  calibration data themselves, not the validation study's results (studies never share
  results); ``lever: param`` takes ``lever_mm`` with its ``lever_source``. Without a lever
  arm E5/E6 are not computed (``cf_summary.note``).

Without a ``mounting`` the stage stores only ``cf_summary`` with the reason (the synthetic
test study and any study that does not run X5); nothing camera-free is computed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from biomech.imu import camerafree as cf
from biomech.kinematics import se3
from biomech.pipeline import CohortContext, ParticipantContext, StageResult, stage

SLOTS = ("cra", "mta")
ROT_EST = ("E1", "E2a", "E2b", "E3a", "E3a_slerp", "E3b", "E3b_slerp")
TRANS_EST = ("E4hinge", "E4own", "E5", "E5_causal", "E6", "E6_causal")
TASK_GROUP = {"open_close": "rotational", "chewing": "rotational",
              "protrusion_retrusion": "translational", "left_right": "translational"}
DEFAULTS: dict[str, Any] = {
    "mounting": None,                  # hand-eye calibration (session.h5 or imu_mounting.json)
    "mounting_by_pid": {},             # pid -> file (overrides ``mounting``)
    "lever": "session",                # session | param
    "lever_mm": None,                  # {cra: [x, y, z], mta: [x, y, z]} plate frame, mm
    "lever_source": "",
    "rest_mask": "labels",             # labels: rest phase ∩ intent label REST | phase
    "calibration_phase": "free_motion",
    "exclude_scored": ["rest", "free_motion"],
    "primary_e3": "E3b",
    "grid_hz": 200.0,
    "closed": dict(cf.CLOSED),         # detector (H4.2, fixed)
    "bridge": dict(cf.BRIDGE),         # E5 noise floors (gate)
    "pseudo_hz": 5.0,                  # E6 pseudo-measurement rate (gate)
    "ridge_log10_lambda": [-8.0, 2.0, 21],   # E4 ridge grid (inner blocked CV)
    "ridge_folds": 5,
    "detect_tol_s": 0.3,               # detection sensitivity/precision tolerance
    "drift_hz": 1.0,                   # cf_drift samples per second
    "max_opt_gap_frames": 2.5,
    # X5c: scored task -> optical label classes of its movement (labels stage) whose
    # calibration frames train that task's model; null = every calibration frame (E4own).
    # Free-motion labels cannot tell chewing from open-close: both use OPEN/CLOSE.
    "x5c_task_labels": {"open_close": ["OPEN", "CLOSE"], "chewing": ["OPEN", "CLOSE"],
                        "protrusion_retrusion": ["PROTRUSION", "RETRUSION"],
                        "left_right": ["LEFT", "RIGHT"]},
    "x5c_min_cal_s": 2.0,              # least calibration time (s) of a class to fit it
}


def params(raw: dict[str, Any] | None) -> dict[str, Any]:
    p = {**DEFAULTS, **(raw or {})}
    for k in ("closed", "bridge"):
        p[k] = {**DEFAULTS[k], **(raw or {}).get(k, {})}
    return p


def study_path(config: Any, rel: str | Path) -> Path:
    """A path of the study block: absolute, or relative to the study file."""
    p = Path(str(rel)).expanduser()
    if p.is_absolute():
        return p
    base = config.source.parent if getattr(config, "source", None) is not None else Path.cwd()
    return (base / p).resolve()


def load_handeye(path: Path, slot: str) -> np.ndarray | None:
    """``R_mount`` (body←chip) of ``slot`` from a hand-eye session (``.h5``) or the stable
    mounting file (``.json``); None unless accepted and a proper rotation."""
    from jtsmm.imu.mounting import load_mounting_file, load_mounting_rotation

    if path.suffix == ".h5":
        return load_mounting_rotation(path, slot)
    e = load_mounting_file(path).get(f"imu_{slot}") or load_mounting_file(path).get(slot)
    return None if e is None else np.asarray(e["R_mount"], float)


def handeye_levers(path: Path) -> dict[str, np.ndarray]:
    """Lever arms (plate frame, mm) from the hand-eye session itself (VD15 method)."""
    from biomech.imu import lever as LV

    out: dict[str, np.ndarray] = {}
    if path.suffix != ".h5":
        return out
    for slot in SLOTS:
        d = LV.read_slot(path, slot)
        if d is None:
            continue
        M, y, t = LV.design(d, float(LV.DEFAULTS["fc_hz"]), LV.DEFAULTS)
        lv = LV.fit(M, y, t, {**LV.DEFAULTS, "n_boot": 0}, 0)
        if "r_x_mm" in lv:
            out[slot] = np.array([lv[f"r_{a}_mm"] for a in "xyz"], float)
    return out


# -- optics on the grid ----------------------------------------------------------------------
def kin_phases(kin: dict[str, Any]) -> list[str]:
    return sorted({k.split("__")[0] for k in kin if k.endswith("__t_us")
                   and f"{k.split('__')[0]}__T" in kin},
                  key=lambda ph: int(np.asarray(kin[f"{ph}__t_us"])[0])
                  if len(kin[f"{ph}__t_us"]) else 0)


def mp_points(kin: dict[str, Any], phases: list[str]) -> dict[str, np.ndarray]:
    """Mandibular points (jaw-plate frame) of the incisor and condyles, back from the
    kinematics tracks (``track = T · p_mp``)."""
    out = {}
    for role in ("incisor", "tmj_left", "tmj_right"):
        pts = []
        for ph in phases:
            T = np.asarray(kin[f"{ph}__T"], float)
            x = np.asarray(kin.get(f"{ph}__{role}", np.full((len(T), 3), np.nan)), float)
            ok = se3.valid(T) & np.isfinite(x).all(axis=1)
            if ok.any():
                pts.append(np.einsum("nji,nj->ni", T[ok, :3, :3], x[ok] - T[ok, :3, 3]))
        out[role] = np.median(np.vstack(pts), axis=0) if pts else np.full(3, np.nan)
    return out


def closed_pose(kin: dict[str, Any], labels: dict[str, Any] | None = None) -> np.ndarray:
    """The optical closed pose of X5: the chordal-mean rotation and mean translation of the
    rest phase's jaw poses (with ``labels``: only frames labelled REST or untracked); the
    kinematics ``T_closed`` (an open_close cycle start) without a rest phase."""
    if "rest__T" in kin:
        T = np.asarray(kin["rest__T"], float)
        ok = se3.valid(T)
        if labels is not None and "rest__label" in labels and \
                len(labels["rest__label"]) == len(T):
            ok &= np.asarray(labels["rest__label"]) <= 0
        if ok.sum() >= 5:
            out = np.eye(4)
            out[:3, :3] = cf._chordal_mean(T[ok, :3, :3])
            out[:3, 3] = T[ok, :3, 3].mean(axis=0)
            return out
    return np.asarray(kin["T_closed"], float)


def landmarks0(kin: dict[str, Any], phases: list[str], T_ref: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """(lower incisor, condyle midpoint) at the closed pose ``T_ref``, maxillary frame."""
    Tc = np.asarray(T_ref, float)
    pm = mp_points(kin, phases)
    inc0 = Tc[:3, :3] @ pm["incisor"] + Tc[:3, 3]
    c0 = Tc[:3, :3] @ (0.5 * (pm["tmj_left"] + pm["tmj_right"])) + Tc[:3, 3]
    return inc0, c0


def labels_on_grid(lab: dict[str, Any] | None, phase: str, tg: np.ndarray) -> np.ndarray | None:
    """The optical intent label codes (labels stage, ``auto_labels.CLASSES``; < 0 untracked)
    of ``phase`` at the grid times ``tg`` (µs): the label of the first optical frame at or
    after each time (clipped to the phase's frames). None without labels of the phase."""
    if lab is None or f"{phase}__label" not in lab or f"{phase}__t_us" not in lab:
        return None
    tl = np.asarray(lab[f"{phase}__t_us"], np.int64)
    if not len(tl):
        return None
    k = np.clip(np.searchsorted(tl, tg), 0, len(tl) - 1)
    return np.asarray(lab[f"{phase}__label"])[k]


def x5c_task_labels(spec: dict[str, Any]) -> dict[str, tuple[str, tuple[int, ...] | None]]:
    """``x5c_task_labels`` with label names turned into codes: task -> (class name, codes);
    None (every calibration frame, E4own's training set) is the class ``all``."""
    from biomech.labels.auto_labels import CODE

    out: dict[str, tuple[str, tuple[int, ...] | None]] = {}
    for task, names in (spec or {}).items():
        if names is None:
            out[str(task)] = ("all", None)
            continue
        bad = [n for n in names if n not in CODE]
        if bad:
            raise ValueError(f"x5c_task_labels.{task}: unknown label(s) {bad}")
        out[str(task)] = ("+".join(names), tuple(CODE[n] for n in names))
    return out


def _interp(t_src: np.ndarray, x: np.ndarray, tg: np.ndarray, max_gap_us: float) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(t_src), -1)
    ok = np.isfinite(x).all(axis=1)
    out = np.full((len(tg), x.shape[1]), np.nan)
    if ok.sum() < 2:
        return out
    ts = np.asarray(t_src, float)[ok]
    for j in range(x.shape[1]):
        out[:, j] = np.interp(tg, ts, x[ok, j], left=np.nan, right=np.nan)
    k = np.clip(np.searchsorted(ts, tg), 1, len(ts) - 1)
    out[(ts[k] - ts[k - 1]) > max_gap_us] = np.nan
    return out


def optical_grid(kin: dict[str, Any], phases: list[str], tg: np.ndarray, T_ref: np.ndarray,
                 gap_frames: float = 2.5) -> dict[str, np.ndarray]:
    """Optical jaw motion on the grid ``tg`` (µs): rotation vector since the closed pose
    ``T_ref`` (deg, maxillary frame), lower incisor and condyle midpoint (mm). Per phase,
    never across."""
    Tc = np.asarray(T_ref, float)
    rv = np.full((len(tg), 3), np.nan)
    inc = np.full((len(tg), 3), np.nan)
    cmid = np.full((len(tg), 3), np.nan)
    for ph in phases:
        t = np.asarray(kin[f"{ph}__t_us"], np.int64)
        if len(t) < 2:
            continue
        sel = (tg >= t[0]) & (tg <= t[-1])
        if not sel.any():
            continue
        gap = gap_frames * float(np.median(np.diff(t)))
        T = np.asarray(kin[f"{ph}__T"], float)
        rv[sel] = _interp(t, np.degrees(se3.rotvec_rel(T, Tc)), tg[sel], gap)
        inc[sel] = _interp(t, kin[f"{ph}__incisor"], tg[sel], gap)
        cm = 0.5 * (np.asarray(kin[f"{ph}__tmj_left"], float)
                    + np.asarray(kin[f"{ph}__tmj_right"], float))
        cmid[sel] = _interp(t, cm, tg[sel], gap)
    return {"rotvec_deg": rv, "incisor_mm": inc, "condyle_mm": cmid}


def rot_of(rotvec_deg: np.ndarray) -> np.ndarray:
    out = np.full((len(rotvec_deg), 3, 3), np.nan)
    ok = np.isfinite(rotvec_deg).all(axis=1)
    if ok.any():
        out[ok] = Rot.from_rotvec(np.radians(rotvec_deg[ok])).as_matrix()
    return out


def optical_block(kin: dict[str, Any], phase: str, grid_hz: float = 200.0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """One participant's optical coupling on ``phase`` (the calibration block) for the
    translation models: rotation vector (deg), θ̇ (deg/s), condyle-midpoint translation (mm)
    on a ``grid_hz`` grid, and the incisor–condyle distance (mm). None without data."""
    if f"{phase}__t_us" not in kin or "T_closed" not in kin:
        return None
    t = np.asarray(kin[f"{phase}__t_us"], np.int64)
    if len(t) < 10:
        return None
    phases = kin_phases(kin)
    T_ref = closed_pose(kin)
    inc0, c0 = landmarks0(kin, phases, T_ref)
    if not (np.isfinite(inc0).all() and np.isfinite(c0).all()):
        return None
    tg = np.arange(t[0], t[-1], int(round(1e6 / grid_hz)), dtype=np.int64)
    o = optical_grid(kin, [phase], tg, T_ref)
    rv = o["rotvec_deg"]
    tau = o["condyle_mm"] - c0
    thd = cf.theta_rate_dps(tg * 1e-6, rv[:, 1])
    ok = np.isfinite(rv).all(1) & np.isfinite(tau).all(1) & np.isfinite(thd)
    if ok.sum() < 50:
        return None
    return rv[ok], thd[ok], tau[ok], float(np.linalg.norm(inc0 - c0))


# -- scoring ---------------------------------------------------------------------------------
def lowpass(R: np.ndarray, p: np.ndarray | None, fs: float, fc: float, order: int
            ) -> tuple[np.ndarray, np.ndarray | None]:
    """The ground truth's zero-phase pose low-pass (``se3.lowpass``) of an estimate."""
    T = np.zeros((len(R), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = 0.0 if p is None else p
    T[:, 3, 3] = 1.0
    T[~np.isfinite(T.reshape(len(T), -1)).all(1)] = np.nan
    T = se3.lowpass(T, fs, fc, order)
    return T[:, :3, :3], (None if p is None else T[:, :3, 3])


def score_rows(base: dict[str, Any], t_s: np.ndarray, R: np.ndarray, R_opt: np.ndarray,
               masks: dict[str, np.ndarray], p: np.ndarray | None = None,
               p_opt: np.ndarray | None = None, min_n: int = 20) -> list[dict[str, Any]]:
    rows = []
    for task, m in masks.items():
        ok = m & np.isfinite(R.reshape(len(R), -1)).all(1) & \
            np.isfinite(R_opt.reshape(len(R_opt), -1)).all(1)
        if p is not None:
            ok &= np.isfinite(p).all(1) & np.isfinite(p_opt).all(1)
        if ok.sum() < min_n:
            continue
        s = cf.score(t_s, R, R_opt, p, p_opt, ok, fc_hz=None)
        rows.append({**base, "task": task, **s})
    return rows


def _amp(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.max(x)) if x.size else np.nan


def cycle_rows(pid: str, cyc: pd.DataFrame, t_us: np.ndarray,
               opening: dict[str, np.ndarray], incisor: dict[str, np.ndarray]
               ) -> list[dict[str, Any]]:
    """Per kinematics cycle and estimator: the cycle amplitude (opening: peak minus the value
    at the cycle start, as C3a; incisor: largest 3-D excursion from the cycle start), with the
    optical one (``opt`` key of the dicts)."""
    rows = []
    if cyc is None or cyc.empty:
        return rows
    for c in cyc[cyc["ok"].astype(bool)].itertuples():
        sel = np.flatnonzero((t_us >= c.t_start_us) & (t_us <= c.t_end_us))
        if sel.size < 10:
            continue
        base = {"pid": pid, "task": c.task, "side": c.side, "cycle": int(c.cycle)}
        for quantity, sig, amp in (("opening_deg", opening, lambda x: _amp(x - x[0])),
                                   ("incisor_mm", incisor,
                                    lambda x: _amp(np.linalg.norm(x - x[0], axis=1)))):
            if "opt" not in sig:
                continue
            ref = amp(sig["opt"][sel])
            for est, x in sig.items():
                if est == "opt":
                    continue
                v = amp(x[sel])
                rows.append({**base, "estimator": est, "quantity": quantity, "amp_est": v,
                             "amp_opt": ref, "amp_diff": v - ref})
    return rows


def closed_runs(t_s: np.ndarray, theta: np.ndarray, rest: np.ndarray, p: dict[str, Any]
                ) -> list[tuple[int, int]]:
    """Optical closed runs: the detector's rules on the optical opening and its rotation
    speed (|d rotvec / dt|, causal 0.1 s mean) -- the reference of the detection metrics."""
    return cf.detect_closed(t_s, theta[:, 1], _opt_speed(t_s, theta, p), rest, params=p).runs


def _opt_speed(t_s: np.ndarray, rv: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    from biomech import timebase

    w = np.column_stack([timebase.gradient(t_s * 1e6, rv[:, j]) for j in range(3)])
    fs = 1.0 / float(np.median(np.diff(t_s)))
    w = cf._causal_mean(w, max(1, int(round(p["speed_smooth_s"] * fs))))
    return np.linalg.norm(w, axis=1)


def _fmean(x: np.ndarray) -> np.ndarray:
    """Mean of the finite rows (NaN without one: an anchor between phases has no optics)."""
    ok = np.isfinite(x).all(axis=1)
    return x[ok].mean(axis=0) if ok.any() else np.full(x.shape[1], np.nan)


def _overlap(a: tuple[float, float], runs: list[tuple[float, float]], tol: float) -> bool:
    return any(min(a[1], b1) + tol >= max(a[0], b0) for b0, b1 in runs)


# -- X5b / X5c (oracle analyses next to E4own) -----------------------------------------------
# estimator -> what it takes from where (``component``: the error it isolates)
DECOMPOSITION = {
    "X5b_model": lambda prim: {"component": "translation_model", "rotation": "optics",
                               "translation": "E4own model"},
    "X5b_rotation": lambda prim: {"component": "rotation", "rotation": prim,
                                  "translation": "optics"},
    "E4own": lambda prim: {"component": "full", "rotation": prim, "translation": "E4own model"},
}


def decomposition_rows(pid: str, t_s: np.ndarray, lp: tuple[float, float, int],
                       R3: np.ndarray, R_opt: np.ndarray,
                       opt: dict[str, np.ndarray], tau_opt: np.ndarray,
                       tau_model_opt: np.ndarray, inc0: np.ndarray, c0: np.ndarray,
                       masks: dict[str, np.ndarray], prim: str) -> list[dict[str, Any]]:
    """X5b rows per task (:func:`biomech.imu.camerafree.oracle_incisors`): each oracle incisor
    through the ground truth's low-pass ``lp`` = (fs Hz, cut-off Hz, order) with its own
    rotation (optics: ``R_opt``; camera-free: the primary E3 ``R3``) and scored like E4own."""
    fs, fc, order = lp
    inc = cf.oracle_incisors(R3, R_opt, tau_model_opt, tau_opt, inc0, c0)
    rows: list[dict[str, Any]] = []
    for est, R in (("X5b_model", R_opt), ("X5b_rotation", R3)):
        R_l, x = lowpass(R, inc[est], fs, fc, order)
        rows += score_rows({"pid": pid, "estimator": est, **DECOMPOSITION[est](prim)}, t_s,
                           R_l, R_opt, masks, x, opt["incisor_mm"])
    return rows


def task_model_rows(pid: str, t_s: np.ndarray, lp: tuple[float, float, int], R3: np.ndarray,
                    R3_lp: np.ndarray, R_opt: np.ndarray, inc_opt: np.ndarray,
                    rv3: np.ndarray, thd3: np.ndarray, fits: dict[str, cf.TaskModel],
                    masks: dict[str, np.ndarray], inc0: np.ndarray, c0: np.ndarray,
                    tasks: list[str]) -> list[dict[str, Any]]:
    """X5c rows: the incisor from E3 plus each scored task's class model (oracle class),
    low-passed like E4own (per task: NaN between the tasks) and scored per task and pooled;
    a task without a model or without enough scored samples gets a row with the reason."""
    fs, fc, order = lp
    sub = {t: masks[t] for t in tasks if t in masks}
    tau = cf.predict_by_task(fits, sub, rv3, thd3)
    _, x = lowpass(R3, cf.incisor_position(R3, tau, inc0, c0), fs, fc, order)
    if sub:
        sub["scored"] = masks["scored"]
    rows = score_rows({"pid": pid, "estimator": "X5c"}, t_s, R3_lp, R_opt, sub, x, inc_opt)
    done = {r["task"] for r in rows}
    for r in rows:
        tm = fits.get(r["task"])
        r.update(_fit_info(tm) if tm is not None else {"label_class": "pooled", "note": ""})
    for t in tasks:
        if t not in done:
            info = _fit_info(fits[t])
            rows.append({"pid": pid, "estimator": "X5c", "task": t, "n": 0, **info,
                         "note": info["note"] or "too few scored samples"})
    return rows


def _fit_info(tm: cf.TaskModel) -> dict[str, Any]:
    m = tm.model
    return {"label_class": tm.label_class, "n_cal_samples": tm.n_cal,
            "lambda": float(m.lam) if m is not None else np.nan,
            "cv_rmse_mm": float(m.cv_rmse) if m is not None else np.nan, "note": tm.note}


# -- the participant stage -------------------------------------------------------------------
def _calibration_e1(cal: pd.DataFrame):
    from biomech.imu.calibration import ImuCalibration

    r = cal.iloc[0]
    A = Rot.from_rotvec(np.radians([r[f"A_rv{a}_deg"] for a in "xyz"])).as_matrix()
    X = Rot.from_rotvec(np.radians([r[f"X_rv{a}_deg"] for a in "xyz"])).as_matrix()
    return ImuCalibration(A, float(np.radians(r["psi_deg"])), X, float(r["rms_deg"]),
                          int(r["n"]), str(r["init"]))


def _note(pid: str, note: str, **kw: Any) -> StageResult:
    return StageResult(tables={"cf_summary": pd.DataFrame([{"pid": pid, "configured": False,
                                                            **kw, "note": note}])})


def run_camerafree(ctx: ParticipantContext) -> StageResult:
    from biomech.imu.preprocess import load
    from biomech.kinematics.pose_chain import KinParams

    p = params(ctx.params)
    pid = ctx.pid
    mfile = (p.get("mounting_by_pid") or {}).get(pid) or p.get("mounting")
    if not mfile:
        return _note(pid, "no hand-eye mounting configured (stages.camerafree.mounting): "
                          "camera-free estimators not computed")
    mpath = study_path(ctx.config, mfile)
    R_mount = {s: load_handeye(mpath, s) for s in SLOTS} if mpath.is_file() else {}
    if any(R_mount.get(s) is None for s in SLOTS):
        raise LookupError(f"hand-eye mounting {mpath.name}: no accepted mounting for "
                          f"{[s for s in SLOTS if R_mount.get(s) is None]}")
    sd = ctx.session("motion")
    mta, cra = load(sd, "mta"), load(sd, "cra")
    if mta is None or cra is None:
        raise LookupError("IMU streams missing")
    kin = ctx.upstream_arrays("kinematics")
    if "T_closed" not in kin:
        raise LookupError("no optical closed pose (kinematics T_closed)")
    kp = KinParams.from_dict(ctx.signals.get("kinematics"))
    phases = kin_phases(kin)
    spans = [(ph, sd.phase(ph)) for ph in phases if sd.phase(ph) is not None]
    spans = [(n, int(ph.t_start_us), int(ph.t_end_us)) for n, ph in spans]
    if not any(n == "rest" for n, *_ in spans):
        raise LookupError("no rest phase: the closed reference and detector threshold need it")
    step = int(round(1e6 / float(p["grid_hz"])))
    tg = np.arange(min(a for _, a, _ in spans), max(b for *_, b in spans), step, dtype=np.int64)
    t_s = tg * 1e-6
    fs = float(p["grid_hz"])
    names = [n for n, *_ in spans]
    phase = np.full(len(tg), -1, np.int16)
    for k, (n, a, b) in enumerate(spans):
        phase[(tg >= a) & (tg <= b)] = k
    mask_of = {n: phase == k for k, n in enumerate(names)}

    # rest mask: rest phase, optionally only where the optical intent label says REST
    rest = mask_of["rest"].copy()
    try:
        lab_all: dict[str, Any] | None = ctx.upstream_arrays("labels")
    except (KeyError, FileNotFoundError, LookupError):
        lab_all = None                                # X5c then notes "no optical labels"
    if p["rest_mask"] == "labels" and lab_all is None:
        raise LookupError("rest_mask: labels, but no labels-stage arrays")
    lab = lab_all if p["rest_mask"] == "labels" else None
    ll = labels_on_grid(lab, "rest", tg[rest])
    if ll is not None:
        rest[np.flatnonzero(rest)] = ll <= 0              # REST (0) or untracked (< 0)
    T_closed = np.asarray(kin["T_closed"], float)
    T_max_hp = np.asarray(kin["T_max_hp"], float)
    T_ref = closed_pose(kin, lab)                     # optical closed pose (rest)
    opt = optical_grid(kin, phases, tg, T_ref, float(p["max_opt_gap_frames"]))
    R_opt = rot_of(opt["rotvec_deg"])
    inc0, c0 = landmarks0(kin, phases, T_ref)

    # rotation ladder
    mp = cf.mounting_prior(R_mount["cra"], R_mount["mta"], T_max_hp)
    lad = cf.rotation_ladder(tg, cra, mta, mp.A, rest, X=mp.X, R_closed=T_closed[:3, :3],
                             params=p["closed"])
    qc, qm = cra.quat_at(tg), mta.quat_at(tg)
    cal = _calibration_e1(ctx.upstream("imu", "calibration"))
    est_R = {"E1": cal.predict(qc, qm) @ T_ref[:3, :3].T, **lad.estimates}
    prim = str(p["primary_e3"])
    an = lad.anchored[prim]
    R3 = est_R[prim]

    # E1 = C3a: the in-take model at the imu stage's own grid
    e1_c3a = np.nan
    imu_arr = ctx.upstream_arrays("imu")
    for ph in names:
        if f"{ph}__theta_quat" in imu_arr:
            ti = np.asarray(imu_arr[f"{ph}__t_us"], np.int64)
            th = cf.opening_deg(cal.predict(cra.quat_at(ti), mta.quat_at(ti))
                                @ T_closed[:3, :3].T)
            d = th - np.asarray(imu_arr[f"{ph}__theta_quat"], float)
            e1_c3a = float(np.nanmax(np.abs(d))) if np.isfinite(d).any() else np.nan
            break

    # translation on the primary E3
    rv3 = np.degrees(cf._rotvec(R3))
    thd3 = cf.theta_rate_dps(t_s, rv3[:, 1])
    est_p: dict[str, np.ndarray] = {"E4hinge": cf.incisor_position(R3, None, inc0, c0)}
    calm = mask_of.get(str(p["calibration_phase"]), np.zeros(len(tg), bool))
    tau_opt = opt["condyle_mm"] - c0
    thd_opt = cf.theta_rate_dps(t_s, opt["rotvec_deg"][:, 1])
    lo, hi, k_l = p["ridge_log10_lambda"]
    lams = np.logspace(float(lo), float(hi), int(k_l))
    okc = calm & np.isfinite(opt["rotvec_deg"]).all(1) & np.isfinite(tau_opt).all(1) & \
        np.isfinite(thd_opt)
    model, notes = None, []
    try:
        model = cf.TranslationModel.fit(opt["rotvec_deg"][okc], thd_opt[okc], tau_opt[okc],
                                        lambdas=lams, n_folds=int(p["ridge_folds"]))
        tau_own = model.predict(rv3, thd3)
        est_p["E4own"] = cf.incisor_position(R3, tau_own, inc0, c0)
    except ValueError as exc:
        notes.append(f"E4own: {exc}")
    levers, lever_src = {}, ""
    if p["lever"] == "session":
        levers, lever_src = handeye_levers(mpath), "hand-eye session (lever-arm method of the validation study)"
    elif p.get("lever_mm"):
        levers = {s: np.asarray(p["lever_mm"][s], float) for s in SLOTS if s in p["lever_mm"]}
        lever_src = str(p.get("lever_source") or "study parameter")
    if all(s in levers for s in SLOTS):
        geom = cf.jaw_geometry(T_max_hp, T_ref, levers["cra"], levers["mta"], inc0, c0)
        amask = an.anchor_mask()
        br = cf.accel_bridge(t_s, R3, an.psi_rad, cra, mta, mp.A, geom, amask, rest,
                             params=p["bridge"])
        est_p["E5"], est_p["E5_causal"] = br.incisor_mm, br.incisor_fwd_mm
        if model is not None:
            b6 = cf.fuse_translation(t_s, R3, an.psi_rad, cra, mta, mp.A, geom, amask, rest,
                                     tau_own, model.cv_rmse, pseudo_hz=float(p["pseudo_hz"]),
                                     params=p["bridge"])
            est_p["E6"], est_p["E6_causal"] = b6.incisor_mm, b6.incisor_fwd_mm
    else:
        notes.append("E5/E6 not computed: no lever arm for "
                     f"{[s for s in SLOTS if s not in levers]}")

    # low-pass every estimate like the ground truth, then score per task
    R_lp = {e: lowpass(R, None, fs, kp.lowpass_hz, kp.lowpass_order)[0] for e, R in est_R.items()}
    R3_lp = R_lp[prim]
    p_lp = {e: lowpass(R3, x, fs, kp.lowpass_hz, kp.lowpass_order)[1] for e, x in est_p.items()}
    excl = set(p["exclude_scored"])
    scored_names = [n for n in names if n not in excl]
    masks = {n: mask_of[n] for n in names if n != "rest"}
    masks["scored"] = np.isin(phase, [names.index(n) for n in scored_names])
    rot_rows, tr_rows = [], []
    for e, R in R_lp.items():
        rot_rows += score_rows({"pid": pid, "estimator": e}, t_s, R, R_opt, masks)
    for e, x in p_lp.items():
        tr_rows += score_rows({"pid": pid, "estimator": e}, t_s, R3_lp, R_opt, masks, x,
                              opt["incisor_mm"])

    # X5b oracle decomposition of E4own and X5c task-conditioned model (never camera-free)
    lp = (fs, kp.lowpass_hz, kp.lowpass_order)
    dec_rows: list[dict[str, Any]] = []
    if model is not None:
        dec_rows = decomposition_rows(pid, t_s, lp, R3, R_opt, opt, tau_opt,
                                      model.predict(opt["rotvec_deg"], thd_opt), inc0, c0,
                                      masks, prim)
        dec_rows += [{**r, **DECOMPOSITION["E4own"](prim)} for r in tr_rows
                     if r["estimator"] == "E4own"]
    else:
        dec_rows = [{"pid": pid, "estimator": "", "task": "scored", "n": 0,
                     "note": "X5b not computed: no E4own model"}]
    codes_cal = labels_on_grid(lab_all, str(p["calibration_phase"]), tg)
    fits = cf.fit_task_models(
        opt["rotvec_deg"][okc], thd_opt[okc], tau_opt[okc],
        None if codes_cal is None else codes_cal[okc], x5c_task_labels(p["x5c_task_labels"]),
        min_n=int(round(float(p["x5c_min_cal_s"]) * fs)), lambdas=lams,
        n_folds=int(p["ridge_folds"]))
    x5c = task_model_rows(pid, t_s, lp, R3, R3_lp, R_opt, opt["incisor_mm"], rv3, thd3,
                          fits, masks, inc0, c0, [n for n in scored_names if n in fits])
    for rows in (rot_rows, tr_rows, dec_rows, x5c):
        for r in rows:
            r["scored"] = r["task"] == "scored" or r["task"] in scored_names
            r["task_group"] = TASK_GROUP.get(r["task"], "all" if r["task"] == "scored"
                                             else "other")

    # per-cycle amplitudes
    open_sig = {"opt": opt["rotvec_deg"][:, 1], **{e: cf.opening_deg(R) for e, R in R_lp.items()}}
    inc_sig = {"opt": opt["incisor_mm"], **p_lp}
    cyc = cycle_rows(pid, ctx.upstream("kinematics", "cycles"), tg, open_sig, inc_sig)

    # anchors: residuals, optical scatter, detection against the optical closed runs
    pc = {**cf.CLOSED, **p["closed"]}
    opt_runs = [(t_s[a], t_s[b - 1]) for a, b in closed_runs(t_s, opt["rotvec_deg"], rest, pc)]
    in_scored = masks["scored"]
    opt_sc = [r for r in opt_runs if in_scored[np.searchsorted(t_s, 0.5 * (r[0] + r[1]))
                                               .clip(0, len(t_s) - 1)]]
    tol = float(p["detect_tol_s"])
    anc_rows = []
    for name in ("E3a", "E3b"):
        a = lad.anchored[name].anchors
        for j in range(len(a["t0_s"])):
            w = slice(int(a["w0"][j]), int(a["w1"][j]))
            rvw, iw = _fmean(opt["rotvec_deg"][w]), _fmean(opt["incisor_mm"][w])
            k = int(a["i0"][j])
            anc_rows.append({
                "pid": pid, "estimator": name, "anchor": j,
                "task": names[phase[k]] if phase[k] >= 0 else "between",
                "t0_s": float(a["t0_s"][j] - t_s[0]), "t_confirm_s": float(a["t_confirm_s"][j] - t_s[0]),
                "t1_s": float(a["t1_s"][j] - t_s[0]),
                "duration_s": float(a["t1_s"][j] - a["t0_s"][j]),
                "psi_deg": float(np.degrees(a["psi_rad"][j])),
                "heading_deg": float(a["heading_deg"][j]),
                "tilt_deg": float(np.linalg.norm(a["tilt_rotvec_deg"][j])),
                "resid_deg": float(np.linalg.norm(a["resid_rotvec_deg"][j])),
                "resid_open_deg": float(a["resid_rotvec_deg"][j][1]),
                "opt_rot_deg": float(np.linalg.norm(rvw)), "opt_open_deg": float(rvw[1]),
                "opt_inc_mm": float(np.linalg.norm(iw - inc0)),
                "matched": _overlap((a["t0_s"][j], a["t1_s"][j]), opt_runs, tol)})
    det = [(t_s[a], t_s[b - 1]) for a, b in an.closed.runs]
    det_sc = [r for r in det if in_scored[np.searchsorted(t_s, 0.5 * (r[0] + r[1]))
                                          .clip(0, len(t_s) - 1)]]
    sens = float(np.mean([_overlap(r, det, tol) for r in opt_sc])) if opt_sc else np.nan
    prec = float(np.mean([_overlap(r, opt_runs, tol) for r in det_sc])) if det_sc else np.nan

    # drift: signed opening error vs time since the reference and since the last anchor
    k_rest = np.flatnonzero(mask_of["rest"])
    t_ref = 0.5 * (t_s[k_rest[0]] + t_s[k_rest[-1]])
    w1 = np.asarray(an.anchors["w1"], int)
    last = np.full(len(tg), np.nan)
    if w1.size:
        ends = t_s[w1 - 1]
        j = np.searchsorted(ends, t_s, side="right") - 1
        last = np.where(j >= 0, t_s - ends[np.clip(j, 0, None)], np.nan)
    stride = max(1, int(round(fs / float(p["drift_hz"]))))
    pick = np.flatnonzero(masks["scored"])[::stride]
    drift_rows = []
    th_opt = opt["rotvec_deg"][:, 1]
    for e in ("E1", "E2a", "E2b", "E3a", "E3b"):
        th = cf.opening_deg(R_lp[e])
        for i in pick:
            d = th[i] - th_opt[i]
            if np.isfinite(d):
                drift_rows.append({"pid": pid, "estimator": e, "task": names[phase[i]],
                                   "t_s": float(t_s[i] - t_s[0]),
                                   "t_since_ref_s": float(t_s[i] - t_ref),
                                   "t_since_anchor_s": float(last[i]),
                                   "err_open_deg": float(d)})
    sig = cf.drift_signature(an, t_s, cra, mta)
    pb = lad.psi_b
    summary = {
        "pid": pid, "configured": True, "mounting_kind": mpath.suffix.lstrip("."),
        "lever_source": lever_src,
        **{f"lever_{s}_{a}_mm": float(levers[s][i]) if s in levers else np.nan
           for s in SLOTS for i, a in enumerate("xyz")},
        "rest_mask": p["rest_mask"], "rest_s": float(rest.sum() / fs),
        "closed_vs_tclosed_deg": float(np.degrees(np.linalg.norm(
            cf._rotvec(T_ref[:3, :3] @ T_closed[:3, :3].T)))),
        "psi_a_deg": float(np.degrees(lad.psi0.get("a", np.nan))),
        "psi_b_deg": float(np.degrees(pb.psi_best_rad)),
        "psi_b_identifiable": bool(pb.identifiable),
        "psi_b_fallback": bool(lad.flags.get("psi_b_fallback", False)),
        "tilt_range_deg": float(pb.tilt_range_deg), "psi_b_spread_deg": float(pb.spread_deg),
        "primary_e3": prim, "rest_p90_deg": float(an.closed.rest_p90_deg),
        "threshold_deg": float(an.closed.threshold_deg),
        "n_anchors": int(len(an.anchors["t0_s"])), "n_closed_opt": len(opt_sc),
        "det_sensitivity": sens, "det_precision": prec,
        "drift_label": str(sig.get("label", "none")),
        **{f"drift_{k}": float(sig.get(k, np.nan)) for k in
           ("heading_deg", "tilt_deg", "fusion_tilt_deg", "step_frac", "net_deg")},
        "e4own_lambda": float(model.lam) if model is not None else np.nan,
        "e4own_cv_rmse_mm": float(model.cv_rmse) if model is not None else np.nan,
        "e1_calibration_rms_deg": float(cal.rms_deg), "e1_vs_c3a_max_deg": e1_c3a,
        **{f"incisor0_{a}_mm": float(v) for a, v in zip("xyz", inc0)},
        **{f"condyle0_{a}_mm": float(v) for a, v in zip("xyz", c0)},
        "scale_mm": float(np.linalg.norm(inc0 - c0)),
        "duration_s": float(t_s[-1] - t_s[0]), "note": "; ".join(notes)}
    f32 = np.float32
    arrays: dict[str, np.ndarray] = {
        "t_s": t_s.astype(np.float64), "phase": phase.astype(np.int16),
        "phase_names": np.asarray(names), "anchor": an.anchor_mask().astype(np.uint8),
        "closed": an.closed.mask.astype(np.uint8),
        "rotvec_deg__opt": opt["rotvec_deg"].astype(f32),
        "rotvec_deg__E3": rv3.astype(f32),          # before the low-pass (E4gen input)
        "incisor_mm__opt": opt["incisor_mm"].astype(f32),
        "open_deg__opt": th_opt.astype(f32)}
    for e, R in R_lp.items():
        arrays[f"open_deg__{e}"] = cf.opening_deg(R).astype(f32)
    for e, x in p_lp.items():
        arrays[f"incisor_mm__{e}"] = x.astype(f32)
    return StageResult(tables={
        "cf_summary": pd.DataFrame([summary]), "cf_rotation": pd.DataFrame(rot_rows),
        "cf_translation": pd.DataFrame(tr_rows), "cf_cycles": pd.DataFrame(cyc),
        "cf_anchors": pd.DataFrame(anc_rows), "cf_drift": pd.DataFrame(drift_rows),
        "cf_decomposition": pd.DataFrame(dec_rows),
        "cf_task_model": pd.DataFrame(x5c)}, arrays=arrays)


@stage("camerafree", depends=("qc", "kinematics", "imu", "labels"), version="2",
       description="X5 (exploratory): camera-free jaw pose -- rotation ladder E1-E3 and "
                   "translation E4hinge/E4own/E5/E6 against the optics, anchors, drift; "
                   "X5b oracle decomposition and X5c task-conditioned model of E4own")
def camerafree_stage(ctx: ParticipantContext) -> StageResult:
    return run_camerafree(ctx)


# -- the cohort stage ------------------------------------------------------------------------
from biomech.stats import cohort_camerafree as _cohort  # noqa: E402


@stage("camerafree_cohort", scope="cohort", depends=("kinematics", "camerafree"),
       partial=("camerafree",), version="2",
       description="X5: E4gen (leave-one-subject-out over every participant's optics), "
                   "cohort statistics (BCa medians, paired ladder contrasts, drift, "
                   "repeated-measures Bland-Altman of the cycle amplitudes); X5b/X5c "
                   "cohort summaries and contrasts")
def camerafree_cohort_stage(ctx: CohortContext) -> StageResult:
    return StageResult(tables=_cohort.run(ctx, params(ctx.config.params("camerafree")),
                                          ctx.params))
