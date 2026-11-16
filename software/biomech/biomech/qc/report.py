# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-take QC tables and the channel exclusion rule (an01 §3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.emg.artifacts import RULES
from biomech.emg.prepare import EmgPrep
from biomech.io.session import SessionData
from biomech.qc import metrics as m

MOTION_PHASES = ("rest", "open_close", "protrusion_retrusion", "left_right", "chewing",
                 "free_motion")
CALIBRATION_PHASES = ("mand_point_1", "mand_point_2", "mand_point_3", "max_point_1",
                      "max_point_2", "max_point_3", "tmj_left", "tmj_right")
MVC_PHASES = ("mvc_opening", "mvc_closing")


def _integrity_counts(doc: dict[str, Any]) -> dict[str, Any]:
    checks = doc.get("checks") or {}
    return {
        "integrity_overall": doc.get("overall"),
        "integrity_errors": sum(len(c.get("errors") or []) for c in checks.values()),
        "integrity_warnings": sum(len(c.get("warnings") or []) for c in checks.values()),
        "emg_precheck": (checks.get("emg") or {}).get("status"),
        "hdf5_check": (checks.get("hdf5") or {}).get("status"),
    }


def session_table(sd: SessionData, pid: str, take: str) -> pd.DataFrame:
    expected = MOTION_PHASES + CALIBRATION_PHASES if take == "motion" else MVC_PHASES
    present = sd.phase_names()
    row = {
        "pid": pid, "take": take, "session": sd.folder.name,
        "duration_s": sd.duration_s,
        "n_phases": len(present),
        "missing_phases": ",".join(p for p in expected if p not in present),
        "n_mcu_logs": sum(e["kind"] == "mcu_log" for e in sd.events()),
        **_integrity_counts(sd.integrity()),
    }
    return pd.DataFrame([row])


def phase_table(sd: SessionData, pid: str, take: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"pid": pid, "take": take, "phase": ph.name, "kind": ph.kind,
         "t_start_us": ph.t_start_us, "t_end_us": ph.t_end_us,
         "phase_s": (ph.t_end_us - ph.t_start_us) / 1e6}
        for ph in sd.phases
    ])


def emg_table(prep: EmgPrep, pid: str, u: dict[str, Any]) -> pd.DataFrame:
    """One row per muscle: noise, references, masks and per-analysis usability.

    Nothing is excluded here. The flags say which analysis families may use a
    channel (analysis_plan §3.1); timing usability is decided per cycle later.

    SNR = reference / rest envelope median, with the MVC/RVC p95 as the reference. Where
    that is missing (no MVC take, or the channel's reference phase absent) the D11
    reference (``prep.ref_uv()``: the largest activity recorded, i.e. the task maximum)
    sets the SNR instead and ``ref_from_task`` is True; formerly the SNR was NaN and every
    channel unusable. The amplitude family still needs a valid MVC/RVC reference.
    """
    motion = prep.motion
    fs = motion.emg_fs
    rest = prep.rest
    f0 = prep.line_hz(motion)
    share_raw = m.channel_mains_share(rest["raw"] - rest["raw"].mean(axis=0), fs, f0)
    share_clean = m.channel_mains_share(rest["cleaned"], fs, f0)
    refs = prep.refs
    names = prep.names

    task_mask = []
    lead = []
    reasons: dict[str, np.ndarray] = {}
    for ph in motion.phases:
        if ph.name not in MOTION_PHASES:
            continue
        pe = prep.phase(ph.name)
        task_mask.append(pe.mask)
        lead.append(motion.leadoff_window(ph.t_start_us, ph.t_end_us, names))
        for r, cnt in pe.masks.counts.items():
            reasons[r] = reasons.get(r, 0) + cnt
    tm = np.concatenate(task_mask) if task_mask else np.zeros((0, len(names)), bool)
    lo = np.concatenate(lead) if lead else np.zeros((0, len(names)), bool)
    n_task = max(tm.shape[0], 1)

    rows = []
    d11 = prep.ref_uv()
    for i, n in enumerate(names):
        ref = float(refs["p95_uv"][i])
        from_task = bool(not np.isfinite(ref) and np.isfinite(d11[i]))
        snr = m.snr_db(float(d11[i]) if from_task else ref, float(rest["env_median_uv"][i]))
        ref_ok = bool(np.isfinite(ref) and refs["mask_frac_effort"][i] <= u["ref_mask_max"])
        row = {
            "pid": pid, "muscle": n, "line_hz": f0,
            "rest_rms_uv": float(rest["rms_uv"][i]),
            "rest_env_median_uv": float(rest["env_median_uv"][i]),
            "rest_lf_uv": float(rest["lf_uv"][i]),
            "ref_phase": str(refs["phase"][i]),
            "ref_p95_uv": ref,
            "ref_max1s_uv": float(refs["max1s_uv"][i]),
            "ref_mask_frac": float(refs["mask_frac"][i]),
            "ref_mask_frac_effort": float(refs["mask_frac_effort"][i]),
            # D11: normalization reference = largest activity recorded; "task" marks a
            # reference contraction that a task exceeded (failed or submaximal).
            "ref_uv": float(d11[i]),
            "ref_from_task": from_task,
            "ref_source": str(prep.reference["source"][i]),
            "mvc_over_task": float(prep.reference["mvc_over_task"][i]),
            "snr_db": snr,
            "mains_share_raw": float(share_raw[i]),
            "mains_share_clean": float(share_clean[i]),
            "leadoff_frac": m.fraction(lo[:, i]) if lo.size else float("nan"),
            "mask_frac_task": float(tm[:, i].mean()) if tm.size else float("nan"),
        }
        for r in RULES:
            row[f"mask_{r}_frac"] = float(reasons[r][i] / n_task) if r in reasons else 0.0
        snr_ok = bool(np.isfinite(snr) and snr >= u["snr_min_db"])
        row.update({
            "usable_shape": snr_ok,
            "usable_amplitude": snr_ok and ref_ok,
            "rest_noise_limited": bool(row["rest_rms_uv"] > u["rest_noise_uv"]),
            # D14: the low-frequency rule only flags (it no longer excludes samples).
            "lowfreq_flag": bool(row["mask_lowfreq_frac"] > u.get("lowfreq_flag_frac", 0.10)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def imu_table(sd: SessionData, pid: str, p: dict[str, Any]) -> pd.DataFrame:
    """One row per IMU slot and stream (quat/accel) over the motion phases."""
    task = [ph for ph in sd.phases if ph.name in MOTION_PHASES]
    rows = []
    for slot in ("mta", "cra"):
        imu = sd.imu(slot)
        streams = []
        if imu is not None and imu.n:
            streams.append(("quat", imu.t_us, imu.quat_accuracy))
        if imu is not None and imu.accel is not None:
            streams.append(("accel", imu.accel.t_us, imu.accel.accuracy))
        if not streams:
            rows.append({"pid": pid, "slot": slot, "stream": "quat", "n": 0, "coverage": 0.0})
            continue
        for stream, t, acc in streams:
            cov = [m.coverage(t, ph.t_start_us, ph.t_end_us, p["gap_s"]) for ph in task]
            dur = np.array([ph.t_end_us - ph.t_start_us for ph in task], dtype=float)
            in_phase = np.zeros(t.shape[0], dtype=bool)
            for ph in task:
                in_phase |= (t >= ph.t_start_us) & (t <= ph.t_end_us)
            acc0 = (m.fraction(np.asarray(acc)[in_phase] == 0)
                    if acc is not None and in_phase.any() else float("nan"))
            rows.append({
                "pid": pid, "slot": slot, "stream": stream, "n": int(t.shape[0]),
                "rate_hz": m.stream_rate_hz(t),
                "nonmono_n": m.nonmonotonic_count(t),
                "coverage": float(np.average(cov, weights=dur)) if task else float("nan"),
                "coverage_min_phase": float(np.min(cov)) if task else float("nan"),
                "acc0_frac": acc0,
            })
    return pd.DataFrame(rows)


def optical_table(sd: SessionData, pid: str, p: dict[str, Any]) -> pd.DataFrame:
    """Lock share per phase and body (all phases: calibration holds included)."""
    opt = sd.optical()
    rows = []
    if opt is None:
        return pd.DataFrame(rows)
    t = opt.t_us.astype(np.int64)
    for body in p["bodies"]:
        if body not in opt.poses:
            continue
        lock = opt.lock(body)
        for ph in sd.phases:
            sel = (t >= ph.t_start_us) & (t <= ph.t_end_us)
            rows.append({"pid": pid, "body": body, "phase": ph.name, "kind": ph.kind,
                         "n_frames": int(sel.sum()), "lock_frac": m.fraction(lock[sel])})
    return pd.DataFrame(rows)
