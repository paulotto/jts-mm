# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trueness, precision and accuracy of an optical tracker on the hand-measured ladder
(plan D6, ISO 5725 terms; validation plan §2.5).

**Core (system-agnostic).** Input: per ladder slot the samples of the tracked array's
position (:class:`SlotSamples`; mm, any world frame) and the 30-s static takes
(:class:`PrecisionSamples`). Per orientation (frontal/lateral) and camera distance
(near/far):

* static window: the central ``static_central`` (80 %) of each slot take (D0b);
* the slot mean; the metric ladder line ``p ≈ c + s·u`` with the hand-measured distance
  ``s`` (scale fixed to the ruler, direction free: a rigid tilt of the ladder is absorbed;
  port of ``experiments/5_1_1_pnp_vs_triangulation/eval_trueness._metric_line``);
* **trueness** = mean deviation of the measured step from the nominal step,
  ``‖p_{k+1} − p_k‖ − (s_{k+1} − s_k)`` (mm), with a t-based 95 % CI over the steps
  (consecutive steps share a slot, so they are not independent: the CI is approximate);
  also the residuals to the metric line, the spacing error from the reference slot (the
  5.1.1 ``distance_error_mm``) and the scale error of a free-scale fit (%);
* **precision** = SD of the static samples (repeatability): within each slot window
  (pooled) and in the 30-s takes after removing a linear drift trend per axis (the drift
  is reported separately, mm/min and over the take); the SD of the slot means about the
  line is reported too;
* **accuracy** = RMS of the total error of every static sample against its nominal
  position on the metric line (trueness and precision together).

**Adapters.** :func:`qualisys_ladder` reads the QTM ``.mat`` exports
(``trueness-{orientation}-{distance}-d{mm}.mat``, ``precision-{distance}.mat``; rigid body
``EvalArraz``; any other or ``.qtm``-only file is logged, e.g. the stray
``trueness-lateral-far-d2000015``). :func:`jtsmm_ladder` reads the 5.1.1 recording leaves
(``…_{near|far}_{lateral|frontal}_d{mm}_{stamp}``, ``…precision…_{near|far}…``) through a
pose loader; the default loader is the 5.1.1 ``array_pose.collect_body_poses`` (PnP or
triangulation; frames at the nominal 40 Hz). ``reference.yaml: jtsmm_ladder_dir`` is
null until the recordings are copied from the Pi: the JTS-MM side is then skipped with a
note and the outputs hold Qualisys only.

The comparison error budget (D4) uses both systems' precision as the expected
disagreement floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",
    "static_central": 0.8,          # D0b: central 80 % of each slot take
    "precision_central": 1.0,       # 30-s takes: whole take, linear drift removed
    "ci": 0.95,
    "reference_mm": 0.0,            # spacing errors relative to this slot
    "qualisys_body": "EvalArraz",
    "ignore": ["trueness-lateral-far-d2000015"],
    "jtsmm_body": "eval_array",     # 5.1.1 study.yaml array_body
    "jtsmm_method": "triangulation",  # or "pnp"
    "jtsmm_fs_hz": 40.0,
    "jtsmm_max_frames": 400,        # 5.1.1: first 10 s of the section
    # VD28 (J14): the same capture-session recordings with single-camera PnP
    # (cam0 images, the tracker's PnP) as a third system ``jtsmm_pnp``
    "pnp": True,
    "pnp_body": "eval_array",
    "pnp_camera": "cam0",
}

ORIENTATIONS = ("frontal", "lateral")
DISTANCES = ("near", "mid", "far")


@dataclass
class SlotSamples:
    system: str
    orientation: str
    distance: str
    slot: str                 # e.g. "d040"
    nominal_mm: float
    t_s: np.ndarray           # (N,)
    p_mm: np.ndarray          # (N, 3); NaN rows = untracked
    source: str = ""


@dataclass
class PrecisionSamples:
    system: str
    distance: str
    t_s: np.ndarray
    p_mm: np.ndarray
    orientation: str = ""
    source: str = ""


# -- core --------------------------------------------------------------------------------------
def central(n: int, frac: float) -> slice:
    """The central ``frac`` of ``n`` samples."""
    cut = int(round(0.5 * (1.0 - float(frac)) * n))
    return slice(cut, n - cut)


def static_samples(t: np.ndarray, p: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    s = central(len(p), frac)
    t, p = np.asarray(t, float)[s], np.asarray(p, float)[s]
    ok = np.isfinite(p).all(axis=1)
    return t[ok], p[ok]


def metric_line(nominal_mm: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``p ≈ c + s·u`` with ``‖u‖ = 1`` and ``s`` the hand-measured distance (scale fixed,
    direction free; the least-squares solution is ``u ∝ Σ (s − s̄) p``)."""
    s = np.asarray(nominal_mm, dtype=np.float64).reshape(-1)
    p = np.asarray(points, dtype=np.float64)
    if s.shape[0] < 2 or p.shape != (s.shape[0], 3):
        raise ValueError("need at least two 3D slot positions")
    ds = s - s.mean()
    if float(ds @ ds) < 1e-12:
        raise ValueError("hand-measured distances are identical")
    v = p.T @ ds
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("ladder direction is undefined")
    u = v / n
    return p.mean(axis=0) - s.mean() * u, u


def t_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and two-sided t confidence interval."""
    from scipy import stats

    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(x.mean()) if x.size else np.nan, np.nan, np.nan)
    m = float(x.mean())
    h = float(stats.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))
    return m, m - h, m + h


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x * x))) if x.size else np.nan


def ladder_metrics(slots: Sequence[SlotSamples], static_central: float = 0.8,
                   ci: float = 0.95, reference_mm: float = 0.0
                   ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Per-slot rows and per (system, orientation, distance) trueness/precision/accuracy."""
    slot_rows, rows, arrays = [], [], {}
    groups: dict[tuple[str, str, str], list[SlotSamples]] = {}
    for s in slots:
        groups.setdefault((s.system, s.orientation, s.distance), []).append(s)
    for (sysm, ori, dist), ss in sorted(groups.items()):
        ss = sorted(ss, key=lambda s: s.nominal_mm)
        stat = [static_samples(s.t_s, s.p_mm, static_central) for s in ss]
        keep = [i for i, (_, p) in enumerate(stat) if len(p) >= 2]
        ss, stat = [ss[i] for i in keep], [stat[i] for i in keep]
        base = {"system": sysm, "orientation": ori, "distance": dist}
        if len(ss) < 2:
            rows.append({**base, "n_slots": len(ss), "note": "fewer than 2 usable slots"})
            continue
        nom = np.array([s.nominal_mm for s in ss])
        means = np.stack([p.mean(axis=0) for _, p in stat])
        c, u = metric_line(nom, means)
        fit = c[None, :] + nom[:, None] * u[None, :]
        resid = means - fit
        k_ref = int(np.argmin(np.abs(nom - reference_mm)))
        dist_err = np.linalg.norm(means - means[k_ref], axis=1) - np.abs(nom - nom[k_ref])
        step = np.linalg.norm(np.diff(means, axis=0), axis=1) - np.diff(nom)
        proj = (means - means.mean(axis=0)) @ u
        from scipy import stats as _st

        lr = _st.linregress(nom, proj)
        slope = float(lr.slope)
        tq = float(_st.t.ppf(0.5 + ci / 2, max(len(nom) - 2, 1)))
        # the scale part of the residual to the metric line: (slope - 1)(s - mean s) along u
        sc_res = (slope - 1.0) * (nom - nom.mean())
        var = np.stack([p.var(axis=0, ddof=1) for _, p in stat])        # (S, 3)
        n_s = np.array([len(p) for _, p in stat])
        pooled = np.sqrt(((n_s - 1)[:, None] * var).sum(axis=0) / (n_s - 1).sum())
        tot = np.concatenate([np.linalg.norm(p - fit[i], axis=1) for i, (_, p) in
                              enumerate(stat)])
        m, lo, hi = t_ci(step, ci)
        rn = np.linalg.norm(resid, axis=1)
        rows.append({**base, "n_slots": len(ss), "n_steps": int(step.size),
                     "nominal_span_mm": float(nom.max() - nom.min()),
                     "trueness_step_mean_mm": m, "trueness_ci_lo_mm": lo,
                     "trueness_ci_hi_mm": hi,
                     "step_dev_sd_mm": float(step.std(ddof=1)) if step.size > 1 else np.nan,
                     "step_dev_rms_mm": _rms(step),
                     "step_dev_max_abs_mm": float(np.max(np.abs(step))),
                     "distance_error_rms_mm": _rms(np.delete(dist_err, k_ref)),
                     "distance_error_end_mm": float(dist_err[np.argmax(np.abs(nom - nom[k_ref]))]),
                     "line_resid_rms_mm": _rms(rn),
                     "line_resid_max_mm": float(rn.max()),
                     "slot_mean_resid_sd_mm": float(np.sqrt((rn ** 2).sum() / max(len(rn) - 2, 1))),
                     "scale_error_pct": 100.0 * (slope - 1.0),
                     "scale_error_ci_pct": 100.0 * tq * float(lr.stderr),
                     "line_resid_scale_rms_mm": _rms(sc_res),
                     "line_resid_scale_share_pct": 100.0 * _rms(sc_res) ** 2 / _rms(rn) ** 2
                     if _rms(rn) > 0 else np.nan,
                     "line_resid_noscale_rms_mm": float(np.sqrt(max(_rms(rn) ** 2
                                                                    - _rms(sc_res) ** 2, 0.0))),
                     "precision_static_sd_x_mm": float(pooled[0]),
                     "precision_static_sd_y_mm": float(pooled[1]),
                     "precision_static_sd_z_mm": float(pooled[2]),
                     "precision_static_sd_3d_mm": float(np.sqrt((pooled ** 2).sum())),
                     "accuracy_rms_mm": _rms(tot),
                     "n_samples": int(n_s.sum()),
                     "u_x": float(u[0]), "u_y": float(u[1]), "u_z": float(u[2])})
        key = f"{sysm}__{ori}__{dist}"
        arrays[f"ladder__{key}__nominal_mm"] = nom
        arrays[f"ladder__{key}__resid_mm"] = resid
        arrays[f"ladder__{key}__step_dev_mm"] = step
        for i, s in enumerate(ss):
            sd = np.sqrt(var[i])
            slot_rows.append({**base, "slot": s.slot, "nominal_mm": float(nom[i]),
                              "n_total": int(len(s.p_mm)), "n_static": int(n_s[i]),
                              "mean_x_mm": float(means[i, 0]), "mean_y_mm": float(means[i, 1]),
                              "mean_z_mm": float(means[i, 2]),
                              "sd_x_mm": float(sd[0]), "sd_y_mm": float(sd[1]),
                              "sd_z_mm": float(sd[2]), "sd_3d_mm": float(np.linalg.norm(sd)),
                              "resid_x_mm": float(resid[i, 0]), "resid_y_mm": float(resid[i, 1]),
                              "resid_z_mm": float(resid[i, 2]), "resid_mm": float(rn[i]),
                              "resid_along_mm": float(resid[i] @ u),
                              "distance_error_mm": float(dist_err[i]),
                              "step_dev_mm": float(step[i - 1]) if i > 0 else np.nan,
                              "source": s.source})
    return pd.DataFrame(slot_rows), pd.DataFrame(rows), arrays


def precision_metrics(takes: Sequence[PrecisionSamples], central_frac: float = 1.0
                      ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """SD of the static samples after removing a per-axis linear trend; the trend is the
    drift (mm/min, and its size over the take). The raw SD (no detrending) is reported."""
    rows, arrays = [], {}
    for tk in takes:
        t, p = static_samples(tk.t_s, tk.p_mm, central_frac)
        base = {"system": tk.system, "orientation": tk.orientation, "distance": tk.distance,
                "source": tk.source}
        if len(p) < 10:
            rows.append({**base, "n": int(len(p)), "note": "fewer than 10 tracked samples"})
            continue
        X = np.column_stack([np.ones_like(t), t - t.mean()])
        coef, *_ = np.linalg.lstsq(X, p, rcond=None)
        res = p - X @ coef
        sd = res.std(axis=0, ddof=2)
        raw = p.std(axis=0, ddof=1)
        drift = coef[1] * 60.0                                       # mm/min per axis
        dur = float(t[-1] - t[0])
        rows.append({**base, "n": int(len(p)), "duration_s": dur,
                     "sd_x_mm": float(sd[0]), "sd_y_mm": float(sd[1]), "sd_z_mm": float(sd[2]),
                     "sd_3d_mm": float(np.sqrt((sd ** 2).sum())),
                     "sd_3d_raw_mm": float(np.sqrt((raw ** 2).sum())),
                     "drift_x_mm_per_min": float(drift[0]), "drift_y_mm_per_min": float(drift[1]),
                     "drift_z_mm_per_min": float(drift[2]),
                     "drift_mm_per_min": float(np.linalg.norm(drift)),
                     "drift_total_mm": float(np.linalg.norm(coef[1]) * dur)})
        key = f"{tk.system}__{tk.orientation or 'any'}__{tk.distance}"
        arrays[f"precision__{key}__t_s"] = (t - t[0]).astype(np.float32)
        arrays[f"precision__{key}__detrended_mm"] = res.astype(np.float32)
    return pd.DataFrame(rows), arrays


def summary_table(trueness: pd.DataFrame, precision: pd.DataFrame) -> pd.DataFrame:
    """Trueness, both precisions and accuracy side by side per system, orientation and
    distance (the 30-s precision of the same system and distance)."""
    if trueness.empty:
        return pd.DataFrame()
    cols = ["system", "orientation", "distance", "n_slots", "trueness_step_mean_mm",
            "trueness_ci_lo_mm", "trueness_ci_hi_mm", "line_resid_rms_mm", "scale_error_pct",
            "precision_static_sd_3d_mm", "accuracy_rms_mm"]
    t = trueness[[c for c in cols if c in trueness]].copy()
    if precision.empty or "sd_3d_mm" not in precision:
        return t
    pr = precision[["system", "distance", "sd_3d_mm", "drift_mm_per_min", "drift_total_mm"]]
    pr = pr.rename(columns={"sd_3d_mm": "precision_30s_sd_3d_mm"})
    return t.merge(pr, on=["system", "distance"], how="left")


def scale_table(trueness: pd.DataFrame, reference: str = "qualisys") -> pd.DataFrame:
    """Scale error per system and condition, and relative to the reference system recorded
    simultaneously (the same ladder: an error of the hand-measured spacing cancels); CIs
    combined in quadrature."""
    if trueness.empty or "scale_error_pct" not in trueness:
        return pd.DataFrame()
    t = trueness.dropna(subset=["scale_error_pct"])
    ref = t[t["system"] == reference].set_index(["orientation", "distance"])
    rows = []
    for _, r in t.iterrows():
        row = {k: r[k] for k in ("system", "orientation", "distance", "scale_error_pct",
                                 "scale_error_ci_pct", "line_resid_rms_mm",
                                 "line_resid_scale_share_pct", "line_resid_noscale_rms_mm")
               if k in r}
        key = (r["orientation"], r["distance"])
        if r["system"] != reference and key in ref.index:
            q = ref.loc[key]
            row["scale_vs_ref_pct"] = float(r["scale_error_pct"] - q["scale_error_pct"])
            row["scale_vs_ref_ci_pct"] = float(np.hypot(r["scale_error_ci_pct"],
                                                        q["scale_error_ci_pct"]))
        rows.append(row)
    return pd.DataFrame(rows)


# -- adapters ----------------------------------------------------------------------------------
_Q_TRUE = re.compile(r"^trueness-(?P<o>frontal|lateral)-(?P<d>near|mid|far)-d(?P<mm>\d{3})$")
_Q_PREC = re.compile(r"^precision-(?P<d>near|mid|far)$")


def qualisys_ladder(folder: str | Path, body: str = "EvalArraz",
                    ignore: Sequence[str] = ()
                    ) -> tuple[list[SlotSamples], list[PrecisionSamples], pd.DataFrame]:
    """The Qualisys ladder exports. Slot files ``trueness-{o}-{d}-d{mm}.mat`` and precision
    files ``precision-{d}.mat``; every other stem (``.qtm``-only, unknown name, ``ignore``
    list) is logged with the reason. Positions = the ``body`` origin (mm, QTM frame)."""
    from biomech.io.reference import read_qtm_mat

    folder = Path(folder)
    stems = sorted({p.stem for p in folder.iterdir() if p.suffix.lower() in (".mat", ".qtm")})
    slots, prec, log = [], [], []
    for stem in stems:
        mat = folder / f"{stem}.mat"
        mt, mp = _Q_TRUE.match(stem), _Q_PREC.match(stem)
        reason = ("in the ignore list" if stem in set(ignore) else
                  "no .mat export" if not mat.is_file() else
                  "name is not a ladder slot or precision take" if not (mt or mp) else "")
        if reason:
            log.append({"system": "qualisys", "file": stem, "status": "ignored",
                        "reason": reason})
            continue
        q = read_qtm_mat(mat)
        if body not in q.bodies:
            log.append({"system": "qualisys", "file": stem, "status": "ignored",
                        "reason": f"rigid body {body!r} missing"})
            continue
        p = q.bodies[body].T[:, :3, 3]
        n_ok = int(np.isfinite(p).all(axis=1).sum())
        log.append({"system": "qualisys", "file": stem, "status": "used", "reason": "",
                    "n_frames": int(len(p)), "n_tracked": n_ok, "fs_hz": q.fs_hz})
        if mt:
            slots.append(SlotSamples("qualisys", mt["o"], mt["d"], f"d{mt['mm']}",
                                     float(mt["mm"]), q.t_s, p, stem))
        else:
            prec.append(PrecisionSamples("qualisys", mp["d"], q.t_s, p, source=stem))
    return slots, prec, pd.DataFrame(log)


_STAMP = re.compile(r"_(\d{8}T\d{6}Z)$")
_SLOT = re.compile(r"^d?(?P<mm>\d+)(mm)?$")


def parse_leaf(name: str) -> dict[str, Any] | None:
    """``…_{near|mid|far}_{lateral|frontal}_d{mm}_{stamp}`` (either order of distance and
    orientation) -> kind/distance/orientation/mm; precision leaves carry ``precision`` and a
    distance. None for anything else."""
    body = _STAMP.sub("", name)
    parts = body.split("_")
    low = [x.lower() for x in parts]
    dist = next((x for x in low if x in DISTANCES), None)
    ori = next((x for x in low if x in ORIENTATIONS), None)
    if any("precision" in x for x in low):
        return {"kind": "precision", "distance": dist, "orientation": ori or ""} if dist \
            else None
    if not any("trueness" in x for x in low) or dist is None or ori is None:
        return None
    idx = max(low.index(dist), low.index(ori))
    m = next((mm for mm in (_SLOT.match(x) for x in low[idx + 1:]) if mm), None)
    if m is None:
        return None
    return {"kind": "trueness", "distance": dist, "orientation": ori, "mm": float(m["mm"])}


def _array_pose_module() -> Any:
    """The 5.1.1 ``array_pose`` module, loaded from its file (no sys.path change)."""
    import importlib.util

    from biomech.config import REPO_ROOT

    path = Path(REPO_ROOT) / "experiments" / "5_1_1_pnp_vs_triangulation" / "array_pose.py"
    spec = importlib.util.spec_from_file_location("_eval511_array_pose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                                  # type: ignore[union-attr]
    return mod


def default_jtsmm_loader(body: str, method: str = "triangulation", fs_hz: float = 40.0,
                         max_frames: int | None = 400, pnp_camera_id: str = "cam0"
                         ) -> Callable[[Path, str], tuple[np.ndarray, np.ndarray]]:
    """Pose loader for the 5.1.1 recording leaves: the array's translation per frame
    (``collect_body_poses``; ``method`` = triangulation or pnp) inside the leaf's measurement
    section (slot takes; first ``max_frames``) or the whole leaf (precision takes). Frame
    times are nominal (``fs_hz``); the loader does not see dropped frames."""
    def load(leaf: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
        from optical.config.loader import default_config_path
        from optical.tracking.tracker import load_tracker_for_recording

        ap = _array_pose_module()
        tracker, tcfg = load_tracker_for_recording(default_config_path(), leaf)
        interval = None
        if kind == "trueness":
            sections = {k: v for k, v in ap.load_experiment_sections(leaf).items()
                        if not ap.is_calibration_section(k)}
            if len(sections) == 1:
                interval = next(iter(sections.values()))
        _, packed = ap.collect_body_poses(
            leaf, tracker=tracker, tcfg=tcfg, bodies=[body], pnp_camera_id=pnp_camera_id,
            frame_interval=interval, max_frames=max_frames if kind == "trueness" else None)
        p = packed[body]["tri_t" if method == "triangulation" else "pnp_t"]
        return np.arange(len(p)) / fs_hz, np.asarray(p, float).reshape(-1, 3)
    return load


def jtsmm_ladder(folder: str | Path,
                 loader: Callable[[Path, str], tuple[np.ndarray, np.ndarray]]
                 ) -> tuple[list[SlotSamples], list[PrecisionSamples], pd.DataFrame]:
    """The JTS-MM ladder recordings (5.1.1 format) under ``folder``: one leaf per slot and
    precision condition (the newest by name when repeated); ``loader(leaf, kind)`` returns
    ``(t_s, p_mm)``. Leaves that do not parse are logged."""
    folder = Path(folder)
    leaves = sorted({p.parent for p in folder.rglob("*") if p.is_file()}
                    | {p for p in folder.iterdir() if p.is_dir()})
    chosen: dict[tuple, tuple[Path, dict[str, Any]]] = {}
    log = []
    for leaf in leaves:
        if leaf == folder:
            continue
        info = parse_leaf(leaf.name)
        if info is None:
            if "trueness" in leaf.name.lower() or "precision" in leaf.name.lower():
                log.append({"system": "jtsmm", "file": leaf.name, "status": "ignored",
                            "reason": "name does not parse"})
            continue
        key = (info["kind"], info["distance"], info.get("orientation", ""), info.get("mm"))
        prev = chosen.get(key)
        if prev is not None:
            older = min(prev[0], leaf, key=lambda q: q.name)
            log.append({"system": "jtsmm", "file": older.name, "status": "ignored",
                        "reason": "a newer leaf of the same condition exists"})
            if older == leaf:
                continue
        chosen[key] = (leaf, info)
    slots, prec = [], []
    for (kind, dist, ori, mm), (leaf, info) in sorted(chosen.items(), key=lambda kv: str(kv[0])):
        try:
            t, p = loader(leaf, kind)
        except Exception as exc:                       # an unreadable leaf is logged, not fatal
            log.append({"system": "jtsmm", "file": leaf.name, "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}"})
            continue
        log.append({"system": "jtsmm", "file": leaf.name, "status": "used", "reason": "",
                    "n_frames": int(len(p)),
                    "n_tracked": int(np.isfinite(p).all(axis=1).sum())})
        if kind == "trueness":
            slots.append(SlotSamples("jtsmm", ori, dist, f"d{int(mm):03d}", float(mm), t, p,
                                     leaf.name))
        else:
            prec.append(PrecisionSamples("jtsmm", dist, t, p, ori, leaf.name))
    return slots, prec, pd.DataFrame(log)


def array_centroid(session_h5: str | Path, tag_ids: Sequence[int] = (35, 41)
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The evaluation array's position per frame from a JTS-MM session's triangulated tag
    corners (``markers_3d``): the centroid of all corners of ``tag_ids``, only in frames
    where every corner is present (a partial set would shift the centroid). Returns
    ``(t_s, p_mm)`` on the master clock (NaN rows where incomplete)."""
    import h5py

    with h5py.File(session_h5, "r") as f:
        t = np.asarray(f["timestamps/t_master_us"][:], float) * 1e-6
        m = f["markers_3d"]
        fr = np.asarray(m["frame_row"][:], int)
        ids = np.asarray(m["marker_id"][:], int)
        pos = np.asarray(m["position_mm"][:], float)
    keep = np.isin(ids, list(tag_ids))
    fr, pos = fr[keep], pos[keep]
    n_need = 4 * len(tag_ids)
    cnt = np.bincount(fr, minlength=len(t))
    s = np.zeros((len(t), 3))
    np.add.at(s, fr, pos)
    p = np.full((len(t), 3), np.nan)
    ok = cnt == n_need
    p[ok] = s[ok] / n_need
    return t, p


def plateaus(t: np.ndarray, p: np.ndarray, *, win_s: float = 1.0, sd_mm: float = 0.25,
             min_s: float = 3.0, step_mm: float = 20.0) -> list[tuple[float, np.ndarray]]:
    """Still plateaus of a stepped ladder recording: runs of at least ``min_s`` whose
    rolling ``win_s`` SD stays below ``sd_mm`` on every axis; neighbouring runs closer than
    ``step_mm / 4`` are merged. The plateaus are projected on the slide direction (first
    principal axis of their means) and mapped to the nominal grid ``step_mm · k`` from the
    first slot. Returns ``[(nominal_mm, frame_indices)]``."""
    from biomech import timebase

    dt = float(np.nanmedian(np.diff(t)))
    w = max(int(round(win_s / dt)), 5)
    sd = pd.DataFrame(p).rolling(w, center=True, min_periods=w // 2).std().max(axis=1)
    still = (sd < sd_mm).to_numpy() & np.isfinite(p).all(axis=1)
    runs = [(a, b) for a, b in timebase.segments(still) if (t[b - 1] - t[a]) >= min_s]
    groups: list[list[int]] = []
    for a, b in runs:
        idx = np.arange(a, b)
        c = np.nanmean(p[idx], axis=0)
        if groups and np.linalg.norm(np.nanmean(p[groups[-1]], axis=0) - c) < step_mm / 4:
            groups[-1] = list(groups[-1]) + idx.tolist()
        else:
            groups.append(idx.tolist())
    if len(groups) < 2:
        return []
    means = np.array([np.nanmean(p[g], axis=0) for g in groups])
    u = np.linalg.svd(means - means.mean(axis=0))[2][0]
    proj = (means - means[0]) @ u
    if np.nanmean(proj) < 0:
        proj = -proj
    return [(float(step_mm * round(x / step_mm)), np.asarray(g)) for x, g in zip(proj, groups)]


def jtsmm_session_ladder(folder: str | Path, tag_ids: Sequence[int] = (35, 41),
                         loader: Callable[[Path, str], tuple[np.ndarray, np.ndarray]]
                         | None = None, system: str = "jtsmm"
                         ) -> tuple[list[SlotSamples], list[PrecisionSamples], pd.DataFrame]:
    """The JTS-MM ladder in the capture-session format: one recording per condition
    (``trueness_{orientation}_{rep}_{distance}_{stamp}``: every slot in sequence;
    ``precision_{rep}_{distance}_{stamp}``: one static 30-s take), each a folder with a
    ``session.h5``. Slots are the still plateaus (:func:`plateaus`)."""
    folder = Path(folder)
    slots, prec, log = [], [], []
    for leaf in sorted(q for q in folder.iterdir() if (q / "session.h5").is_file()):
        low = [x.lower() for x in _STAMP.sub("", leaf.name).split("_")]
        dist = next((x for x in low if x in DISTANCES), None)
        ori = next((x for x in low if x in ORIENTATIONS), "")
        kind = "precision" if "precision" in low else "trueness"
        try:
            t, p = (array_centroid(leaf / "session.h5", tag_ids) if loader is None
                    else loader(leaf, kind))
        except (KeyError, OSError, ValueError, ImportError) as exc:
            log.append({"system": system, "file": leaf.name, "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if "precision" in low and dist:
            prec.append(PrecisionSamples(system, dist, t, p, ori, leaf.name))
            log.append({"system": system, "file": leaf.name, "status": "used", "reason": "",
                        "n_frames": int(len(p)), "n_tracked": int(np.isfinite(p).all(1).sum())})
        elif "trueness" in low and dist and ori:
            if loader is None:
                pl = plateaus(t, p)
            else:
                # the slots come from the triangulated positions of the same frames: a
                # single-camera series is too noisy at far range for the plateau detector
                # (PnP depth SD ~1.3 mm against the 0.25 mm still threshold)
                tc, pc = array_centroid(leaf / "session.h5", tag_ids)
                pl = plateaus(tc, pc) if len(pc) == len(p) else []
                if len(pc) != len(p):
                    log.append({"system": system, "file": leaf.name, "status": "failed",
                                "reason": f"{len(p)} frames vs {len(pc)} session rows"})
                    continue
            for mm, idx in pl:
                slots.append(SlotSamples(system, ori, dist, f"d{int(mm):03d}", float(mm),
                                         t[idx], p[idx], leaf.name))
            log.append({"system": system, "file": leaf.name, "status": "used",
                        "reason": f"{len(pl)} plateaus", "n_frames": int(len(p)),
                        "n_tracked": int(np.isfinite(p).all(1).sum())})
        else:
            log.append({"system": system, "file": leaf.name, "status": "ignored",
                        "reason": "name does not parse"})
    return slots, prec, pd.DataFrame(log)


# -- orchestration -----------------------------------------------------------------------------
def run_ref_trueness(doc: dict[str, Any], data_dir: Path, p: dict[str, Any],
                     jtsmm_loader: Callable[[Path, str], tuple[np.ndarray, np.ndarray]]
                     | None = None
                     ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """The work of ``ref_trueness``: both adapters (where configured) and the core metrics.
    ``doc`` = the reference manifest; ladder dirs are relative to ``data_dir``."""
    p = {**DEFAULTS, **(p or {})}
    slots: list[SlotSamples] = []
    prec: list[PrecisionSamples] = []
    pnp_slots: list[SlotSamples] = []          # VD28: kept apart from the paper tables
    pnp_prec: list[PrecisionSamples] = []
    logs, notes = [], []
    for system, key in (("qualisys", "qualisys_ladder_dir"), ("jtsmm", "jtsmm_ladder_dir")):
        rel = doc.get(key)
        if not rel:
            notes.append({"system": system, "note": f"{key} not set: {system} skipped"
                          + (" (recordings on the Pi; the views show Qualisys only)"
                             if system == "jtsmm" else "")})
            continue
        folder = (Path(data_dir) / str(rel)).resolve()
        if not folder.is_dir():
            notes.append({"system": system, "note": f"{key} = {folder} not found: skipped"})
            continue
        if system == "qualisys":
            s, pr, lg = qualisys_ladder(folder, p["qualisys_body"], p["ignore"])
        elif any((q / "session.h5").is_file() for q in folder.iterdir() if q.is_dir()):
            s, pr, lg = jtsmm_session_ladder(folder)          # capture-session format
            if p.get("pnp"):                                   # VD28: single camera
                ld = default_jtsmm_loader(p["pnp_body"], "pnp", float(p["jtsmm_fs_hz"]), None,
                                          p["pnp_camera"])
                s2, pr2, lg2 = jtsmm_session_ladder(folder, loader=ld, system="jtsmm_pnp")
                pnp_slots, pnp_prec = pnp_slots + s2, pnp_prec + pr2
                lg = pd.concat([lg, lg2], ignore_index=True)
        else:
            ld = jtsmm_loader or default_jtsmm_loader(
                p["jtsmm_body"], p["jtsmm_method"], float(p["jtsmm_fs_hz"]),
                p["jtsmm_max_frames"])
            s, pr, lg = jtsmm_ladder(folder, ld)
        slots += s
        prec += pr
        logs.append(lg)
        notes.append({"system": system, "note": f"{len(s)} slot takes, {len(pr)} precision "
                                                f"takes from {folder.name}"})
    slot_tab, true_tab, arrays = ladder_metrics(slots, float(p["static_central"]),
                                                float(p["ci"]), float(p["reference_mm"]))
    prec_tab, parr = precision_metrics(prec, float(p["precision_central"]))
    arrays.update(parr)
    logs = [lg for lg in logs if not lg.empty]
    out_pnp: dict[str, pd.DataFrame] = {}
    scale_in = true_tab
    if pnp_slots or pnp_prec:
        _, pt, _ = ladder_metrics(pnp_slots, float(p["static_central"]), float(p["ci"]),
                                  float(p["reference_mm"]))
        pp, _ = precision_metrics(pnp_prec, float(p["precision_central"]))
        out_pnp = {"pnp_trueness": pt, "pnp_precision": pp}
        scale_in = pd.concat([true_tab, pt], ignore_index=True)
    return ({"slots": slot_tab, "trueness": true_tab, "precision": prec_tab,
             "summary": summary_table(true_tab, prec_tab), "scale": scale_table(scale_in),
             **out_pnp,
             "files": pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(),
             "notes": pd.DataFrame(notes)}, arrays)
