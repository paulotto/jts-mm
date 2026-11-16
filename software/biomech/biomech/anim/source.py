# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The data one animation plays: a :class:`Clip` (one participant, one or more
phase windows) or a :class:`CohortClip` (all participants' cycle-normalized paths).

Everything is read through :class:`biomech.io.context.DataContext` (stored stage
arrays, the session's optical group, preview frames), cut to the requested
windows and kept as plain numpy arrays, so a clip pickles cheaply into the
parallel video workers. A modality that is missing (stage not run, no IMU, no
previews, ...) is ``None``; layouts drop its panels and re-flow.

Privacy: only ``label`` (the pseudonym, "Participant 3") is ever drawn. Camera
previews show the participant and are loaded only with ``with_camera=True``.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from biomech.labels.auto_labels import CLASSES
from biomech.viz.views import vocab as V

LANDMARKS = ("incisor", "left", "right", "tmj_left", "tmj_right")
MUSCLES = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l", "digastric_r",
           "digastric_l")
MUSCLE_LABEL = V.labels("muscle", "title")
PHASE_LABEL = V.labels("task", "title")
BODY_LABEL = V.labels("body", "title")
PAD_S = 6.0                        # data kept around a window (scrolling panels look back/ahead)


log = logging.getLogger(__name__)


def _snapshot(ctx) -> str:
    """The store a clip is read from ("live" or the frozen snapshot), for the manifest."""
    return str(getattr(getattr(ctx, "config", None), "snapshot", None) or "live")


def _variant(ctx) -> str:
    """The reporting population of that store (a sensitivity variant, or "primary"), for the
    manifest: a paper set pins both (``biomech check-outputs``, J2.7)."""
    return str(getattr(getattr(ctx, "config", None), "variant", None) or "primary")


def _skip(what: str, pid: str, exc: Exception) -> None:
    """A modality that cannot be loaded is left out (the layout re-flows) -- never silently."""
    log.warning("animation %s: %s left out (%s: %s)", pid, what, type(exc).__name__, exc)

@dataclass
class Clip:
    pid: str
    label: str                                   # pseudonym shown on screen
    segments: list[tuple[str, float, float]]     # (phase, t0_us, t1_us), play order
    kin: dict[str, np.ndarray] | None = None
    maxilla: dict[str, np.ndarray] | None = None     # static landmarks, maxillary frame
    emg: dict[str, Any] | None = None
    imu: dict[str, np.ndarray] | None = None
    labels: dict[str, np.ndarray] | None = None
    posterior: dict[str, np.ndarray] | None = None
    optical: dict[str, Any] | None = None
    previews: dict[str, dict[str, Any]] | None = None   # cam -> {t_us, paths}
    with_camera: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    # Layout-specific derived modalities (D5): "hinge" (fixed-hinge ghost, residual),
    # "release" (X3 closer envelope, threshold, decisions), "online" (stream decoder).
    extra: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def t0_us(self) -> float:
        return min(a for _, a, _ in self.segments)

    @property
    def t1_us(self) -> float:
        return max(b for _, _, b in self.segments)

    def modalities(self) -> list[str]:
        return [k for k in ("kin", "emg", "imu", "labels", "posterior", "optical", "previews")
                if getattr(self, k) is not None] + sorted(self.extra)


@dataclass
class CohortClip:
    task: str
    labels: list[str]                            # pseudonyms, panel order
    trials: list[np.ndarray]                     # per participant (K, 101, 3) incisor paths
    means: np.ndarray                            # (P, 101, 3)
    speed: np.ndarray                            # (P, 101) speed along the mean path (mm/%)
    meta: dict[str, Any] = field(default_factory=dict)
    # Cohort layouts other than ``cohort`` keep their data here (J2.7: the
    # ``workspace3d`` point clouds, hull and design box); travels with the clip files.
    extra: dict[str, dict[str, Any]] = field(default_factory=dict)


# -- helpers -------------------------------------------------------------------------------
def _concat(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    parts = [p for p in parts if p and len(p.get("t_us", ())) > 0]
    if not parts:
        return None
    keys = set.intersection(*(set(p) for p in parts))
    out = {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}
    order = np.argsort(out["t_us"], kind="stable")
    return {k: v[order] for k, v in out.items()}


def _cut(a: dict[str, np.ndarray], phase: str, keys: tuple[str, ...], t0: float, t1: float
         ) -> dict[str, np.ndarray]:
    t = a.get(f"{phase}__t_us")
    if t is None:
        return {}
    t = np.asarray(t, dtype=np.float64)
    sel = (t >= t0 - PAD_S * 1e6) & (t <= t1 + PAD_S * 1e6)
    out = {"t_us": t[sel]}
    for k in keys:
        v = a.get(f"{phase}__{k}")
        if v is not None and len(v) == len(t):
            out[k] = np.asarray(v)[sel]
    return out


def _kinematics(ctx, pid: str, segs) -> tuple[dict | None, dict | None]:
    from biomech.kinematics.derivatives import angular_velocity, derivatives

    try:
        a = ctx.arrays("kinematics", pid)
    except (OSError, ValueError, KeyError) as exc:
        _skip("kinematics", pid, exc)
        return None, None
    parts = []
    for ph, t0, t1 in segs:
        p = _cut(a, ph, ("T", *LANDMARKS, "incisor_vel", "omega", "theta_deg"), t0, t1)
        if len(p.get("t_us", ())) < 2 or not all(k in p for k in ("T", *LANDMARKS)):
            continue
        p = {k: (v.astype(np.float64) if k != "t_us" else v) for k, v in p.items()}
        # Older stores have no derivative arrays: compute them with the shared module.
        if "incisor_vel" not in p:
            p["incisor_vel"] = derivatives(p["t_us"], p["incisor"])[0]
        if "omega" not in p:
            p["omega"] = angular_velocity(p["t_us"], p["T"][:, :3, :3])
        if "theta_deg" not in p:
            p["theta_deg"] = np.full(len(p["t_us"]), np.nan)
        p["phase"] = np.full(len(p["t_us"]), segs.index((ph, t0, t1)), dtype=np.int16)
        parts.append(p)
    kin = _concat(parts)
    if kin is not None:
        kin["R"] = kin.pop("T")[:, :3, :3]
        kin["speed"] = np.linalg.norm(kin["incisor_vel"], axis=1)
    maxilla = None
    try:
        lm = ctx.table("kinematics", "landmarks", [pid])
        T = a.get("T_max_hp")
        if T is not None and not lm.empty:
            hp = lm[lm.body == "hp"].set_index("name")
            pts = {}
            for name, key in (("max_point_1", "incisor"), ("max_point_2", "left"),
                              ("max_point_3", "right"), ("tmj_left_hp", "tmj_left"),
                              ("tmj_right_hp", "tmj_right")):
                if name in hp.index:
                    role = hp.loc[name, "role_used"] if "role_used" in hp else key
                    if name.startswith("max_point") and isinstance(role, str) and role in (
                            "incisor", "left", "right"):
                        key = role
                    p = np.r_[hp.loc[name, ["x", "y", "z"]].to_numpy(float), 1.0]
                    pts[key] = (np.asarray(T, float) @ p)[:3]
            maxilla = pts or None
    except (OSError, ValueError, KeyError, IndexError) as exc:
        _skip("maxillary landmarks", pid, exc)
        maxilla = None
    return kin, maxilla


def _stage(ctx, stage: str, pid: str, segs, keys: tuple[str, ...]) -> dict | None:
    try:
        a = ctx.arrays(stage, pid)
    except (OSError, ValueError, KeyError) as exc:
        _skip(stage, pid, exc)
        return None
    return _concat([_cut(a, ph, keys, t0, t1) for ph, t0, t1 in segs])


def _posterior(ctx, pid: str, segs) -> dict | None:
    """The decoder's window-level decisions (LOSO, primary model, EMG + IMU) stored by
    ml_eval (``pred__<pid>__emg_imu__t_us`` / ``__yhat``), as a one-hot (N, 7) "posterior"
    -- decisions, not calibrated probabilities; ``None`` when ml_eval stored none."""
    from biomech.results import COHORT_UNIT

    a = ctx.arrays("ml_eval", COHORT_UNIT)
    t, yhat = a.get(f"pred__{pid}__emg_imu__t_us"), a.get(f"pred__{pid}__emg_imu__yhat")
    if t is None or yhat is None or len(t) != len(yhat):
        return None
    t = np.asarray(t, np.float64)
    sel = np.zeros(len(t), bool)
    for _, t0, t1 in segs:
        sel |= (t >= t0 - PAD_S * 1e6) & (t <= t1 + PAD_S * 1e6)
    if not sel.any():
        return None
    y = np.asarray(yhat, int)[sel]
    p = np.zeros((y.size, 7))
    ok = (y >= 0) & (y < 7)
    p[np.flatnonzero(ok), y[ok]] = 1.0
    return {"t_us": t[sel], "p": p}


def _session(ctx, pid: str, take: str, segs, with_camera: bool):
    from biomech.io.session import SessionData

    path = ctx.session_path(pid, take)
    names: list[str] = []
    optical = None
    if path is None:
        return names, optical, None
    try:
        with SessionData(path) as sd:
            names = sd.muscle_names()
            o = sd.optical()
            if o is not None:
                t = o.t_us.astype(np.float64)
                sel = np.zeros(len(t), bool)
                for _, t0, t1 in segs:
                    sel |= (t >= t0 - PAD_S * 1e6) & (t <= t1 + PAD_S * 1e6)
                if sel.any():
                    optical = {"t_us": t[sel],
                               "poses": {b: p[sel] for b, p in o.poses.items()
                                         if np.isfinite(p[sel].reshape(sel.sum(), -1)).any()},
                               "n_cameras": {b: np.asarray(q["n_cameras"], float)[sel]
                                             for b, q in o.quality.items() if "n_cameras" in q},
                               "n_tags": {b: np.asarray(q["n_tags"], float)[sel]
                                          for b, q in o.quality.items() if "n_tags" in q},
                               "rms": {b: np.asarray(q["reprojection_rms"], float)[sel]
                                       for b, q in o.quality.items()
                                       if "reprojection_rms" in q}}
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        _skip("optical tracking quality", pid, exc)
    previews = None
    if with_camera:
        rows = ctx.previews(pid, take)
        cams: dict[str, dict[str, list]] = {}
        for r in rows:
            t = float(r["t_master_us"])
            if any(t0 - PAD_S * 1e6 <= t <= t1 for _, t0, t1 in segs):
                c = cams.setdefault(r["cam_id"], {"t_us": [], "paths": []})
                c["t_us"].append(t)
                c["paths"].append(r["abs_path"])
        previews = {}
        for cam, c in sorted(cams.items()):
            o = np.argsort(c["t_us"], kind="stable")
            previews[cam] = {"t_us": np.asarray(c["t_us"])[o],
                             "paths": [c["paths"][i] for i in o]}
        previews = previews or None
    return names, optical, previews


# -- D5 extras ---------------------------------------------------------------------------------
def _near_segments(t: np.ndarray, segs, pad_s: float = PAD_S) -> np.ndarray:
    t = np.asarray(t, np.float64)
    sel = np.zeros(len(t), bool)
    for _, t0, t1 in segs:
        sel |= (t >= t0 - pad_s * 1e6) & (t <= t1 + pad_s * 1e6)
    return sel


def _hinge(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """The participant's best-fit fixed hinge (C1: stage kinematics, table ``hinge``, model
    ``hinge_fit``: axis direction n and its point at y = 0) applied to the clip.

    The ghost mandible starts from the closed reference pose ``T_closed`` and turns about the
    fixed axis by the angle that best maps the five landmarks onto the measured ones at each
    sample (:func:`biomech.kinematics.hinge_fit.best_angles`, the fit's own rule); the
    residual is measured minus ghost at the incisor. (The fit itself references every
    opening to its own closed start pose; the video uses one closed pose throughout.)"""
    from biomech.kinematics.hinge_fit import best_angles, rotate

    k = clip.kin
    if k is None:
        return None
    tab = ctx.table("kinematics", "hinge", [pid])
    row = tab[tab["model"] == "hinge_fit"] if "model" in tab else tab.iloc[:0]
    if row.empty:
        raise ValueError("no hinge fit stored (too few open-close cycles?)")
    r = row.iloc[0]
    n = np.array([r["axis_x"], r["axis_y"], r["axis_z"]], float)
    s = np.array([r["axis_pierce_x_mm"], 0.0, r["axis_pierce_z_mm"]], float)
    if not (np.isfinite(n).all() and np.isfinite(s).all()):
        raise ValueError("the stored hinge axis does not cross the midsagittal plane")
    n = n / np.linalg.norm(n)
    a = ctx.arrays("kinematics", pid)
    Tc = a.get("T_closed")
    if Tc is None:
        raise ValueError("no closed reference pose (T_closed) stored")
    Tc = np.asarray(Tc, float)
    # Landmarks in the mandibular frame, from the stored poses and tracks (constant).
    pmp = []
    for role in LANDMARKS:
        vals = []
        for ph, _, _ in clip.segments:
            T, b = a.get(f"{ph}__T"), a.get(f"{ph}__{role}")
            if T is None or b is None:
                continue
            T, b = np.asarray(T, float), np.asarray(b, float)
            ok = np.isfinite(T.reshape(len(T), -1)).all(axis=1) & np.isfinite(b).all(axis=1)
            vals.append(np.einsum("nji,nj->ni", T[ok, :3, :3], b[ok] - T[ok, :3, 3]))
        v = np.concatenate(vals) if vals else np.empty((0, 3))
        if not len(v):
            raise ValueError(f"no valid pose to place the {role} landmark")
        pmp.append(np.median(v, axis=0))
    A = (Tc[:3, :3] @ np.array(pmp).T).T + Tc[:3, 3]                       # (5, 3) closed
    B = np.stack([np.asarray(k[r_], float) for r_ in LANDMARKS], axis=1)   # (N, 5, 3)
    ok = np.isfinite(B).all(axis=(1, 2))
    th = np.full(len(B), np.nan)
    if ok.any():
        th[ok] = best_angles(np.broadcast_to(A - s, B[ok].shape), B[ok] - s, n)
    G = np.full(B.shape, np.nan)
    if ok.any():
        G[ok] = rotate(np.broadcast_to(A - s, B[ok].shape).copy(), n, th[ok]) + s
    res = B[:, 0] - G[:, 0]
    return {"t_us": np.asarray(k["t_us"], np.float64), "ghost": G, "residual": res,
            "residual_mm": np.linalg.norm(res, axis=1), "theta_h_deg": np.degrees(th),
            "disp_mm": np.linalg.norm(B[:, 0] - A[0], axis=1),
            "disp_model_mm": np.linalg.norm(G[:, 0] - A[0], axis=1), "closed": A,
            "axis_n": n, "axis_s": s,
            "rms_incisor_mm": np.asarray(float(r["rms_incisor_mm"])),
            "max_incisor_mm": np.asarray(float(r["max_incisor_mm"]))}


def release_threshold(t_us: np.ndarray, e_db: np.ndarray, p) -> np.ndarray:
    """The release threshold B(t) − ``drop_db`` of :func:`biomech.timing.release.detect`
    per sample (lagged median of the 10-ms decimated envelope over [t − base_hi_s,
    t − base_lo_s], NaN held at the last valid value; NaN where there is no baseline yet).
    ``detect`` decides where E(t) <= this threshold has held for ``hold_s`` (test_r4_anim)."""
    from scipy.ndimage import median_filter

    from biomech.timing.release import _causal_fill

    t = np.asarray(t_us, dtype=np.int64)
    e_db = np.asarray(e_db, dtype=np.float64)
    out = np.full(e_db.size, np.nan)
    if t.size < 4:
        return out
    fs = 1e6 / np.median(np.diff(t))
    lo, hi = int(round(p.base_lo_s * fs)), int(round(p.base_hi_s * fs))
    step = max(int(round(0.01 * fs)), 1)
    ed = e_db[::step]
    w = max((hi - lo) // step, 1)
    filled, first = _causal_fill(ed)
    med = median_filter(filled, size=w, mode="nearest", origin=(w - 1) // 2)
    med[: min(first, med.size)] = np.nan
    base_d = np.full(ed.size, np.nan)
    shift = lo // step
    base_d[shift:] = med[: ed.size - shift] if shift else med
    return np.repeat(base_d, step)[: e_db.size] - p.drop_db


def _release(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """The causal closer-release detector (X3, :mod:`biomech.timing.release`) on the clip's
    phases, recomputed from the session exactly as the timing stage does (causally cleaned
    EMG, rest level of the same chain): group closer envelope E (dB re rest), the threshold
    and the decision times; envelope and threshold kept at 200 Hz for display."""
    from biomech.timing.ensemble import EnsembleParams
    from biomech.timing.lead import _rest_level
    from biomech.timing.release import ReleaseParams, causal_db, detect

    p = ctx.config.params("timing")
    rp = ReleaseParams.from_dict(p.get("release"))
    ep = EnsembleParams.from_dict(p.get("ensemble"))
    prep = ctx.emg(pid)
    rest = prep.phase("rest")
    if rest is None:
        raise ValueError("no rest phase (no rest level for the release envelope)")
    level = _rest_level(np.asarray(prep.causal("rest")), rest.mask, rest.fs, ep.envelope_s)
    parts, decs = [], []
    for ph in dict.fromkeys(ph for ph, _, _ in clip.segments):
        pe = prep.phase(ph)
        t = np.asarray(pe.t_us, np.int64)
        e = causal_db(t, np.asarray(prep.causal(ph)), pe.mask, level, list(prep.names),
                      rp.window_s)
        thr = release_threshold(t, e, rp)
        dec = detect(t, e, rp)
        step = max(int(round(pe.fs / 200.0)), 1)
        sel = _near_segments(t, clip.segments)
        idx = np.flatnonzero(sel)[::step]
        parts.append({"t_us": t[idx].astype(np.float64), "e_db": e[idx], "thr_db": thr[idx]})
        decs.append(dec[_near_segments(dec, clip.segments)])
    out = _concat(parts)
    if out is None:
        return None
    out["dec_us"] = np.sort(np.concatenate(decs)).astype(np.float64)
    out.update(drop_db=np.asarray(rp.drop_db), hold_s=np.asarray(rp.hold_s),
               window_s=np.asarray(rp.window_s), base_lo_s=np.asarray(rp.base_lo_s),
               base_hi_s=np.asarray(rp.base_hi_s))
    return out


EVENT_KINDS = ("hit", "false", "other")


def _online(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """The pseudo-online stream decoder (stage ``ml_online``, primary feature set, no
    calibration): decisions at their emission time (``online__<pid>__t_us``), true class,
    smoothed and raw stream decisions, causal release decisions, and the classified onset
    events of table ``decisions`` (hit / false / other, decision latency)."""
    from biomech.results import COHORT_UNIT

    a = ctx.arrays("ml_online", COHORT_UNIT)
    t = a.get(f"online__{pid}__t_us")
    if t is None:
        return None
    t = np.asarray(t, np.float64)
    sel = _near_segments(t, clip.segments)
    o = np.argsort(t[sel], kind="stable")
    out: dict[str, Any] = {"t_us": t[sel][o]}
    for k in ("t_end_us", "y", "yhat", "yhat_raw"):
        v = a.get(f"online__{pid}__{k}")
        if v is not None and len(v) == len(t):
            out[k] = np.asarray(v)[sel][o].astype(np.float64)
    rel = np.asarray(a.get(f"online__{pid}__release_us", np.empty(0)), np.float64)
    out["release_us"] = np.sort(rel[_near_segments(rel, clip.segments)])
    prim = ctx.config.params("ml_online").get("primary_feature_set", "emg_imu")
    try:
        d = ctx.table("ml_online", "decisions")
    except (OSError, ValueError, KeyError) as exc:
        _skip("online decision events", pid, exc)
        d = None
    if d is not None and not d.empty and {"pid", "feature_set", "mode"} <= set(d.columns):
        d = d[(d["pid"] == pid) & (d["feature_set"] == prim) & (d["mode"] == "none")]
        te = d["t_us"].to_numpy(np.float64)
        keep = _near_segments(te, clip.segments)
        d = d[keep].sort_values("t_us", kind="stable")
        out["ev_t_us"] = d["t_us"].to_numpy(np.float64)
        out["ev_kind"] = np.array([EVENT_KINDS.index(k) if k in EVENT_KINDS else 2
                                   for k in d["kind"]], np.float64)
        out["ev_latency_ms"] = d["latency_ms"].to_numpy(np.float64)
        out["ev_cls"] = d["cls"].to_numpy(np.float64)
    return out


EXTRAS = {"hinge": _hinge, "release": _release, "online": _online}


def load_extras(ctx, pid: str, clip: Clip, names: tuple[str, ...] | list[str]) -> None:
    """Add the named derived modalities to ``clip.extra`` (one that cannot be loaded is left
    out with a logged warning; the layout decides whether it can do without)."""
    for name in names:
        try:
            v = EXTRAS[name](ctx, pid, clip)
        except (OSError, ValueError, KeyError, IndexError, AttributeError) as exc:
            _skip(name, pid, exc)
            continue
        if v is not None:
            clip.extra[name] = v


# -- public --------------------------------------------------------------------------------
def load_clip(ctx, pid: str, phases: str | list[str], t0_s: float | None = None,
              t1_s: float | None = None, *, with_camera: bool = False,
              extras: tuple[str, ...] | list[str] = ()) -> Clip:
    """The modalities of ``pid`` during ``phases`` (play order).

    ``t0_s``/``t1_s`` cut a window in seconds from the start of *each* selected phase (with
    several phases every phase is cut the same way, then played in order); ``t1_s`` is
    clipped to the phase end. A window that leaves a phase empty is an error naming the
    phase (never silently ignored)."""
    phases = [phases] if isinstance(phases, str) else list(phases)
    if pid not in ctx.pids():
        raise ValueError(f"{pid!r} is not an included participant")
    segs: list[tuple[str, float, float]] = []
    take = None
    for ph in phases:
        tk, iv = ctx.phase(pid, ph)
        if iv is None:
            raise ValueError(f"{pid} has no phase {ph!r}")
        take = take or tk
        a, b = float(iv.t_start_us), float(iv.t_end_us)
        if t0_s is not None or t1_s is not None:
            a0 = a + (t0_s or 0.0) * 1e6
            b = min(b, a + t1_s * 1e6) if t1_s is not None else b
            a = a0
        if b <= a:
            dur = (float(iv.t_end_us) - float(iv.t_start_us)) * 1e-6
            raise ValueError(f"empty window in {ph!r}: t0/t1 ({t0_s}, {t1_s}) s from the phase "
                             f"start, the phase lasts {dur:.1f} s")
        segs.append((ph, a, b))
    kin, maxilla = _kinematics(ctx, pid, segs)
    emg = _stage(ctx, "emg", pid, segs, ("norm_pct", "masked"))
    names, optical, previews = _session(ctx, pid, take or "motion", segs, with_camera)
    if emg is not None and "norm_pct" in emg:
        pct = emg["norm_pct"].astype(np.float64)
        if "masked" in emg:
            pct[emg["masked"].astype(bool)] = np.nan       # masked samples are NaN, never 0
        stored = ctx.arrays("emg", pid).get("profile__names")
        nm = [str(s) for s in stored] if stored is not None else names
        if len(nm) != pct.shape[1]:
            nm = list(MUSCLES[: pct.shape[1]])
        emg = {"t_us": emg["t_us"], "pct": pct, "names": nm}
    else:
        emg = None
    imu = _stage(ctx, "imu", pid, segs, ("theta_quat", "theta_comp", "theta_opt"))
    if imu is not None and "theta_quat" not in imu:
        imu = None
    labels = _stage(ctx, "labels", pid, segs, ("label",))
    if labels is not None and "label" not in labels:
        labels = None
    clip = Clip(pid=pid, label=f"Participant {ctx.pseudonym(pid)}", segments=segs, kin=kin,
                maxilla=maxilla, emg=emg, imu=imu, labels=labels,
                posterior=_posterior(ctx, pid, segs), optical=optical, previews=previews,
                with_camera=with_camera, meta={"take": take, "classes": list(CLASSES),
                                               "window_s": [t0_s, t1_s],
                                               "snapshot": _snapshot(ctx),
                                               "variant": _variant(ctx)})
    load_extras(ctx, pid, clip, extras)
    return clip


def load_cohort_clip(ctx, task: str = "open_close") -> CohortClip:
    """Every included participant's incisor paths of ``task``, cycle-normalized."""
    from biomech.ensembles import aggregate, kinematic

    ens = kinematic(ctx, task, "incisor", align="path")
    if ens.n_trials:
        ens = ens.ok() if ens.ok().n_trials else ens
    labels, trials, means, dur = [], [], [], []
    if ens.n_trials:
        summ = aggregate(ens, "participant", "mean_sd")
        for g, pid in enumerate(summ.groups):
            sel = (ens.trials.pid == pid).to_numpy()
            labels.append(f"P{ctx.pseudonym(pid)}")
            trials.append(ens.data[sel])
            means.append(summ.center[g])
            d = ens.trials.duration_s[sel].to_numpy(float) if "duration_s" in ens.trials \
                else np.array([np.nan])
            dur.append(np.nanmean(d) if np.isfinite(d).any() else 1.0)
    M = np.stack(means) if means else np.zeros((0, 101, 3))
    # Speed along the mean path in mm/s: |dP/d%| x 100 % / mean cycle duration.
    sp = (np.linalg.norm(np.gradient(M, axis=1), axis=2) * 100.0 / np.asarray(dur)[:, None]
          if len(M) else np.zeros((0, 101)))
    order = np.argsort([int(s[1:]) for s in labels]) if labels else []
    return CohortClip(task=task, labels=[labels[i] for i in order],
                      trials=[trials[i] for i in order], means=M[list(order)] if len(M) else M,
                      speed=sp[list(order)] if len(sp) else sp,
                      meta={"n_trials": int(ens.n_trials), "snapshot": _snapshot(ctx),
                            "variant": _variant(ctx)})
