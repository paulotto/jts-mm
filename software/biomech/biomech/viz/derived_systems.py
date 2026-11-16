# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Derived tables for the systems paper's in-text numbers (``source: derived:<name>``, see
:mod:`biomech.viz.derived`; 2026-11-05, the Sensors draft's TODOs).

Each table summarises what the pinned validation store (or, for ``recorder``, the two
comparison takes' own ``session.h5`` metadata, which the recorder writes once and the store
fingerprints) already holds; nothing here changes a result:

* ``imu_lag`` -- per participant and held-out task: the lag of the JTS-MM IMU orientation
  against the JTS-MM optical orientation fitted by the fusion calibration (``ref_fusion``,
  pooled over both bodies), the lag of the IMU accelerometer against Qualisys
  (``ref_imu_accel``, sensor variant, median over the bodies; 5-ms grid) and the heading drift
  of the magnetometer-free fusion per body. A wrong-by-one frame association would shift the
  optical lag by a whole frame period (25 ms at 40 Hz); the IMU has no association step.
* ``lp_agreement`` -- per participant and axis (incisor x, y, z; rotation about x, y, z from
  rest), frames aligned, movement-task frames: least-products regression of JTS-MM on
  Qualisys (slope = sign(r) sd(JTS-MM) / sd(Qualisys), intercept), r, RMSE and NRMSE (RMSE /
  range of Qualisys, %), the form of Mostashiri et al. (2018). Descriptive (frames are
  autocorrelated: no CIs).
* ``recorder`` -- per participant: trigger rate and pulse width (``/meta`` MCU config), the
  tracking stream size, the calibrated camera baseline and field of view (the live tracker's
  camera model in ``/optical/corners/calibration``), the working distance (median distance of
  the mandibular array from each camera over the take), the sEMG poll and DRDY statistics of
  the MCU (``/sync`` report; ``poll_max_us`` is a high-water mark since boot, the firmware
  keeps no distribution), and the achieved IMU quaternion / accelerometer rates.
* ``phase_durations`` -- per participant and phase: the duration of the operator-marked phase.
* ``imu_mounting`` -- per participant: the hand-eye mounting session the take uses (VD14 rule in
  ``reference.yaml``: a participant's own ``imu_mounting``, else the study-level one) and, per
  IMU (mta / cra), the duration of its tumble phase (``imu_mount_mta`` / ``imu_mount_cra``),
  the fit RMS and the selected / paired sample counts of ``/calibration/imu_<body>``.
* ``chewing_opening`` -- for the DATA-SET store (cited by the systems paper): per participant
  the median incisal opening of the usable chewing cycles (``kinematics:cycles``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.derived import _table

AXES = (("incisor_x", "incisor_s_aligned_mm", "incisor_q_mm", 0),
        ("incisor_y", "incisor_s_aligned_mm", "incisor_q_mm", 1),
        ("incisor_z", "incisor_s_aligned_mm", "incisor_q_mm", 2),
        ("rot_x", "rot_s_aligned_deg", "rot_q_aligned_deg", 0),
        ("rot_y", "rot_s_aligned_deg", "rot_q_aligned_deg", 1),
        ("rot_z", "rot_s_aligned_deg", "rot_q_aligned_deg", 2))


@_table("imu_lag")
def imu_lag(ctx: Any) -> pd.DataFrame:
    cal = ctx.table("ref_fusion", "calibration")
    acc = ctx.table("ref_imu_accel", "calibration")
    if cal.empty:
        return pd.DataFrame()
    cal = cal[cal["variant"].isna()] if "variant" in cal else cal
    rows = []
    for (pid, fold), g in cal.groupby(["pid", "fold"], sort=False):
        a = acc[(acc["pid"] == pid) & (acc["fold"] == fold) & acc["primary"].astype(bool)
                & (acc["variant"] == "sensor")] if not acc.empty else acc
        drift = {str(b): float(v) for b, v in zip(g["body"], g["drift_heading_deg_min"])}
        rows.append({"pid": pid, "fold": fold,
                     "lag_opt_ms": float(g["lag_ms"].median()),       # pooled: same per body
                     "lag_opt_se_ms": float(g["lag_se_ms"].median()),
                     "lag_ref_ms": float(a["lag_ms"].median()) if len(a) else np.nan,
                     "drift_hp_deg_min": drift.get("HP", np.nan),
                     "drift_mp_deg_min": drift.get("MP", np.nan)})
    return pd.DataFrame(rows)


def _lp(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    r = float(np.corrcoef(x, y)[0, 1])
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    slope = float(np.sign(r) * sy / sx) if sx > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    rng = float(np.max(x) - np.min(x))
    return {"lp_slope": slope, "lp_intercept": float(np.mean(y) - slope * np.mean(x)),
            "lp_r": r, "lp_rmse": rmse, "range_ref": rng,
            "nrmse_pct": 100.0 * rmse / rng if rng > 0 else np.nan}


@_table("lp_agreement")
def lp_agreement(ctx: Any) -> pd.DataFrame:
    from biomech.validation.selection import MOVEMENT_TASKS

    rows = []
    for pid in ctx.pids():
        a = ctx.arrays("ref_kinematics", pid)
        if "phase" not in a:
            continue
        k = np.isin(np.asarray(a["phase"]).astype(str), MOVEMENT_TASKS)
        if "evaluated" in a:
            k &= np.asarray(a["evaluated"], bool)
        for axis, s_key, q_key, j in AXES:
            if s_key not in a or q_key not in a:
                continue
            y, x = np.asarray(a[s_key], float)[:, j], np.asarray(a[q_key], float)[:, j]
            ok = k & np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            rows.append({"pid": pid, "axis": axis, "n": int(ok.sum()), **_lp(x[ok], y[ok])})
    return pd.DataFrame(rows)


def _rate(t_us: np.ndarray) -> float:
    t = np.asarray(t_us, float)
    t = t[np.isfinite(t)]
    return float(1e6 / np.median(np.diff(t))) if t.size > 2 else np.nan


@_table("recorder")
def recorder(ctx: Any) -> pd.DataFrame:
    import json

    import h5py

    rows = []
    for pid in ctx.pids():
        d = ctx.session_path(pid)
        f = (d / "session.h5") if d is not None else None
        if f is None or not f.is_file():
            continue
        with h5py.File(f, "r") as h:
            m = h["meta"].attrs

            def attr(k: str) -> Any:
                v = m.get(k)
                return v.decode() if isinstance(v, bytes) else v

            trig = (json.loads(attr("mcu_config_json") or "{}").get("trigger") or {})
            row: dict[str, Any] = {"pid": pid, "trigger_rate_hz": float(trig.get("rate_hz",
                                                                                np.nan)),
                                   "trigger_pulse_us": float((trig.get("pulse") or {})
                                                             .get("width_us", np.nan))}
            size = str(attr("optical_rgb_size") or "x").split("x")
            row["stream_w_px"], row["stream_h_px"] = (float(size[0]), float(size[1])) \
                if len(size) == 2 and size[0] else (np.nan, np.nan)
            if "optical/corners/calibration" in h:
                cal = h["optical/corners/calibration"]
                K0, T1 = cal["cam0"]["K"][:], cal["cam1"]["T_world_cam"][:]
                w, hh = (float(v) for v in cal["cam0"].attrs["image_size"])
                row["baseline_mm"] = float(np.linalg.norm(T1[:3, 3]))
                row["hfov_deg"] = float(np.degrees(2 * np.arctan(0.5 * w / K0[0, 0])))
                row["vfov_deg"] = float(np.degrees(2 * np.arctan(0.5 * hh / K0[1, 1])))
                P = h["optical/pose_mta"][:]
                ok = np.isfinite(P[:, :3, 3]).all(axis=1)
                p = P[ok, :3, 3].astype(float)
                if len(p):
                    row["dist_cam0_mm"] = float(np.median(np.linalg.norm(p, axis=1)))
                    row["dist_cam1_mm"] = float(np.median(np.linalg.norm(p - T1[:3, 3],
                                                                         axis=1)))
            if "sync" in h and "report_json" in h["sync"].attrs:
                s = h["sync"].attrs["report_json"]
                rep = json.loads(s.decode() if isinstance(s, bytes) else s)
                e, b = rep.get("mcu_stats_end") or {}, rep.get("mcu_stats_start") or {}
                row["poll_max_us"] = float(e.get("poll_max_us", np.nan))
                row["drdy_gap_max_us"] = float(e.get("max_drdy_gap_us", np.nan))
                row["missed_drdy_take"] = float(e.get("missed_drdy", np.nan)
                                                - b.get("missed_drdy", 0))
                row["drdy_take"] = float(e.get("drdy_count", np.nan) - b.get("drdy_count", 0))
            for body in ("mta", "cra"):
                g = h.get(f"imu_{body}")
                if g is not None:
                    row[f"quat_rate_{body}_hz"] = _rate(g["t_master_us_quat"][:])
                    row[f"accel_rate_{body}_hz"] = _rate(g["t_master_us_accel"][:])
        rows.append(row)
    return pd.DataFrame(rows)


@_table("phase_durations")
def phase_durations(ctx: Any) -> pd.DataFrame:
    rows = []
    for pid in ctx.pids():
        for ph in ctx.phases(pid):
            rows.append({"pid": pid, "phase": ph.name,
                         "duration_s": (ph.t_end_us - ph.t_start_us) * 1e-6})
    return pd.DataFrame(rows)


@_table("chewing_opening")
def chewing_opening(ctx: Any) -> pd.DataFrame:
    c = ctx.table("kinematics", "cycles")
    if c.empty:
        return pd.DataFrame()
    c = c[(c["task"] == "chewing") & c["ok"].astype(bool)]
    return (c.groupby("pid")["opening_mm"].median().rename("chew_opening_mm").reset_index())


@_table("imu_mounting")
def imu_mounting(ctx: Any) -> pd.DataFrame:
    import h5py

    from biomech.validation.stages import load_reference, reference_path

    rel = (ctx.config.params("ref_camerafree") or {}).get("reference", "reference.yaml")
    doc = load_reference(reference_path(ctx.config, rel))
    rows = []
    for pid in ctx.pids():
        entry = (doc.get("participants") or {}).get(pid) or {}
        f = entry.get("imu_mounting") if "imu_mounting" in entry else doc.get("imu_mounting")
        if not f or isinstance(f, dict):
            continue
        path = ctx.config.data_dir / str(f)
        if not path.is_file():
            continue
        row: dict[str, Any] = {"pid": pid, "session": Path(str(f)).parent.name}
        with h5py.File(path, "r") as h:
            ph = h["phases"]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in ph["name"][()]]
            dur = {n: (b - a) * 1e-6 for n, a, b in zip(names, ph["t_start_us"][()],
                                                         ph["t_end_us"][()])}
            for body in ("mta", "cra"):
                row[f"mount_{body}_s"] = float(dur.get(f"imu_mount_{body}", np.nan))
                g = h.get(f"calibration/imu_{body}")
                row[f"mount_{body}_rms_deg"] = float(g.attrs["rms_deg"]) if g is not None \
                    else np.nan
                row[f"mount_{body}_n_selected"] = int(g.attrs["n_selected"]) if g is not None \
                    else 0
                row[f"mount_{body}_n_pairs"] = int(g.attrs["n_pairs"]) if g is not None else 0
        rows.append(row)
    return pd.DataFrame(rows)



# -- J30: audit files of VD29-VD32 on the pinned snapshot -----------------------
def _audit_file(ctx: Any, name: str) -> pd.DataFrame:
    """``<store>/audit/<name>_<snapshot>.parquet`` (``scripts/audit_recompute/<script>.py``),
    with its sha256 so that the export provenance changes when the file does."""
    import hashlib

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"{name}_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/{name}.py")
    return pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


@_table("head_cluster")
def head_cluster(ctx: Any) -> pd.DataFrame:
    """VD31 (post hoc, descriptive): the Qualisys head-plate cluster's principal axes u1 (long),
    u2, u3. Rows pid x part x component x task: ``mc`` the section-5.4 Monte Carlo rotation of
    the head body alone split into the u1 and (u2, u3) parts (median over the movement tasks,
    with the analytic sigma_m / sqrt(s2^2 + s3^2)); ``heldout`` / ``heldout_const`` /
    ``heldout_varying`` the held-out head-plate rotation difference (pooled, its per-interval
    constant and the remainder) projected on u1, (u2, u3) and the JTS-MM tag baseline
    (``jts_x``), shares of the mean square; ``marking_frame`` the maxillary frame error of the
    marking in the same axes; ``aligned_rms`` offset and scatter of the pooled aligned incisor
    RMS."""
    return _audit_file(ctx, "vd31_head_cluster")


@_table("hinge_floor")
def hinge_floor(ctx: Any) -> pd.DataFrame:
    """VD32 (post hoc, descriptive): per participant and hinge model (``fit`` best-fit fixed
    axis, ``tmj`` axis through the condyle points) the observed Qualisys incisor residual on
    the open-close openings and its noise floor (P50 / P95 / P99 of the same fit on the noise-
    free model reconstruction plus the section-5.4 Qualisys marker noise, independent per pose,
    1000 draws), ratio observed / P99 and above."""
    return _audit_file(ctx, "vd32_hinge_floor")


@_table("array_noise")
def array_noise(ctx: Any) -> pd.DataFrame:
    """VD29 (post hoc, descriptive; with the disclosed addendum 2): lever arms of the two JTS-MM
    arrays (``source`` rest), their pose noise per body axis (``mc`` Monte Carlo from the
    corner reprojection residual, ``bench_calibrated`` scaled by the bench ratio k,
    ``rest_jitter`` / ``movement_jitter`` the in-vivo upper bound) and the first-order sigma_p at
    the incisor and the condyle mean with the CRA share (``equation_exact``), the what-ifs
    (50-mm tags, head lever halved) and the bench k. Rows pid x array x quantity x axis x point x
    source x variant."""
    return _audit_file(ctx, "vd29_array_noise")


@_table("tag_viewing")
def tag_viewing(ctx: Any) -> pd.DataFrame:
    """VD30 (post hoc, descriptive): per take, array (``paper_array`` HP / MP), camera and tag
    (``all`` = pooled) over the movement frames: incidence angle of the tag normal to the line of
    sight, tag side in pixels, reprojection residual (median, quartiles) and detection rate."""
    return _audit_file(ctx, "vd30_tag_viewing")


@_table("array_noise_checks")
def array_noise_checks(ctx: Any) -> pd.DataFrame:
    """VD29 checks of the propagation: per participant and point (incisor, condyle) on the
    computed scale, the first-order exact sigma_p against the Monte Carlo of the refitted
    jaw-in-head point (``exact_vs_mc_pct`` = 100 (exact / MC - 1)) and the isotropic form's
    shortfall against the exact one (``iso_gap_pct`` = 100 (1 - isotropic / exact)); and, from
    VD30, the smallest per-tag detection rate over arrays, cameras and tags (``point`` =
    ``detection``, column ``detection_min``)."""
    d = _audit_file(ctx, "vd29_array_noise")
    v = _audit_file(ctx, "vd30_tag_viewing")
    rows = []
    for pid in sorted(set(d.pid) - {"bench"}):
        g = d[(d.pid == pid) & (d.quantity == "sigma_p") & (d.variant == "computed")]
        for pt in ("incisor", "condyle"):
            val = {s: float(g[(g.point == pt) & (g.source == s)].value.iloc[0])
                   for s in ("equation_exact", "equation_isotropic", "mc_direct")}
            rows.append({"pid": pid, "point": pt,
                         "exact_vs_mc_pct": 100 * (val["equation_exact"] / val["mc_direct"] - 1),
                         "iso_gap_pct": 100 * (1 - val["equation_isotropic"]
                                               / val["equation_exact"]),
                         "detection_min": np.nan})
        t = v[(v.pid == pid) & (v.quantity == "detection_rate") & v.camera.isin(["cam0", "cam1"])
              & (v.tag != "all")]
        rows.append({"pid": pid, "point": "detection", "exact_vs_mc_pct": np.nan,
                     "iso_gap_pct": np.nan, "detection_min": float(t["median"].min())})
    return pd.DataFrame(rows)


@_table("qualisys_noise")
def qualisys_noise(ctx: Any) -> pd.DataFrame:
    """VD34 (post hoc, descriptive): ``rho`` the observed Qualisys orientation scatter above 6 Hz
    over the white-noise prediction of the section-5.4 Monte Carlo per body and axis group;
    ``sigma_q`` and ``budget`` the stored values and the sensitivity with each body's marker
    noise scaled by its 3-D rho (a lower bound of the Qualisys noise). Rows with ``task`` =
    ``median`` are the medians over the movement tasks."""
    return _audit_file(ctx, "vd34_qualisys_noise")


@_table("tag_model")
def tag_model(ctx: Any) -> pd.DataFrame:
    """VD35 (post hoc, descriptive): the comparison chain rerun with the JTS-MM poses refitted on
    the nominal tag model and on the tag model estimated from each take's own corners
    (``model`` stored / nominal / estimated): sync registration and held-out values, the
    held-out head-plate split (VD31 b), the aligned offset / scatter / RMS (VD31 d) and the
    own-frame RMS; ``corner_offset`` the estimated corners' distance from the nominal model."""
    return _audit_file(ctx, "vd35_tag_model")


@_table("hinge_floor_scaled")
def hinge_floor_scaled(ctx: Any) -> pd.DataFrame:
    """VD36 (post hoc, descriptive): the VD32 noise floor of the Qualisys hinge residuals with
    each body's marker noise scaled by its VD34 ratio, the lower bound of the floor (VD32 is the
    upper bound); same columns as ``hinge_floor`` plus the two scale factors."""
    return _audit_file(ctx, "vd36_hinge_floor_scaled")
