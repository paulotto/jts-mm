# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""What every stored table column and array means: the data dictionary of the result store.

Tables are described by the metric catalog (``etc/catalog.yaml``; columns without an entry
get a label and unit from their name, :func:`biomech.catalog.guess`). Arrays are described
here, per stage, by family patterns (task, muscle, participant and model names replaced by
``{task}``, ``{muscle}``, ``{pid}``, ``{model}``; see :func:`biomech.results.access.family`)
with ``fnmatch`` wildcards. ``biomech describe`` prints the dictionary, snapshots store it
as ``schema.json``, and tests/test_r4_schema.py fails on an array family without an entry.

Conventions: time stamps ``*t_us`` are master-clock microseconds (int64); ``{task}`` arrays
on a per-task time grid share that task's ``{task}__t_us``; positions are in the maxillary
frame (x anterior, y left, z up; mm); EMG activation is % of the largest activity recorded
(D11); classes are REST, OPEN, CLOSE, PROTRUSION, RETRUSION, LEFT, RIGHT (0-6, -1 = none).
"""

from __future__ import annotations

import fnmatch
from typing import Any

# stage -> [(family pattern, unit, shape, description)]
ARRAYS: dict[str, list[tuple[str, str, str, str]]] = {
    "kinematics": [
        ("{task}__t_us", "µs", "(N,)", "optical frame time stamps of the task (40 Hz)"),
        ("{task}__T", "mm, rotation", "(N, 4, 4)",
         "mandible pose in the maxillary frame (homogeneous transform), low-passed"),
        ("{task}__incisor", "mm", "(N, 3)", "lower incisor point position"),
        ("{task}__incisor_vel", "mm/s", "(N, 3)", "incisor velocity (Savitzky–Golay)"),
        ("{task}__incisor_acc", "mm/s²", "(N, 3)", "incisor acceleration (Savitzky–Golay)"),
        ("{task}__left", "mm", "(N, 3)", "lower left first molar point"),
        ("{task}__right", "mm", "(N, 3)", "lower right first molar point"),
        ("{task}__tmj_left", "mm", "(N, 3)", "left (skin-palpated) condyle point"),
        ("{task}__tmj_right", "mm", "(N, 3)", "right (skin-palpated) condyle point"),
        ("{task}__omega", "°/s", "(N, 3)", "mandible angular velocity (maxillary frame)"),
        ("{task}__theta_deg", "°", "(N,)", "opening angle (rotation about the transverse axis)"),
        ("{task}__theta_raw_deg", "°", "(N,)", "opening angle without the pose low-pass"),
        ("{task}__theta_vel", "°/s", "(N,)", "opening angular velocity"),
        ("{task}__theta_acc", "°/s²", "(N,)", "opening angular acceleration"),
        ("T_max_hp", "mm, rotation", "(4, 4)",
         "maxilla pose at the landmark hold (frame of the static landmarks)"),
        ("T_closed", "mm, rotation", "(4, 4)", "closed-mouth reference pose"),
        ("coupling__grid_deg", "°", "(G,)", "opening-angle grid of the coupling curves"),
        ("coupling__*", "mm", "(cycles, G)",
         "per opening cycle: incisal opening / condyle displacement vs opening angle"),
        ("hinge__theta_deg", "°", "(F,)", "opening angle of the frames used by the hinge fits"),
        ("hinge__residual_incisor", "mm", "(F,)", "incisor residual of the best fixed hinge"),
        ("hinge__tau_mm", "mm", "(F,)", "translation of the rotation + translation model"),
    ],
    "qc": [
        ("timeline__*__t_s", "s", "(K,)",
         "time of the thinned fit residual RMS (s from the first phase start)"),
        ("timeline__*__fit_rms_mm", "mm", "(K,)",
         "optical rigid-fit residual RMS (kabsch_mm), min/max per bin (≤ 4000 points)"),
        ("scene__*__xyz", "mm", "(≤1500, 3)",
         "body origin in the reference camera frame, thinned (phase or 'all')"),
        ("scene__*__key", "mm, rotation", "(3, 4, 4)",
         "body pose at the first, middle and last tracked thinned frame"),
    ],
    "emg": [
        ("{task}__t_us", "µs", "(N,)", "time grid of the stored envelopes (decimated)"),
        ("{task}__norm_pct", "%", "(N, C)", "masked RMS envelope, % of max recorded (D11)"),
        ("{task}__env_uv", "µV", "(N, C)", "masked RMS envelope"),
        ("{task}__masked", "0/1", "(N, C)", "analysis mask (rail, lead-off, envelope; D14)"),
        ("profile__names", "", "(C,)", "channel order of the profiles"),
        ("profile__{task}__pct", "%", "(cycles, 101, C)",
         "time-normalized activation per cycle (0-100 % of the cycle)"),
        ("profile__{task}__cycle", "", "(cycles,)", "cycle index of each profile"),
        ("profile__{task}__theta_deg", "°", "(cycles, 101)", "opening angle per cycle"),
        ("spectral__freq", "Hz", "(F,)", "frequency axis of the stored PSDs"),
        ("spectral__{task}__psd", "µV²/Hz", "(C, F)", "Welch PSD of the cleaned, masked EMG"),
        ("spectral__{task}__win", "log10 share", "(C, windows, bins)",
         "per-window spectral shape on 10-Hz bins (input of the spectral SPM)"),
        ("mains__freq", "Hz", "(F,)", "frequency axis of the raw / mains-removed PSDs (≤ 500 Hz)"),
        ("mains__{task}__raw_psd", "µV²/Hz", "(C, F)",
         "Welch PSD (1-s segments) of the raw EMG (median removed), whole phase"),
        ("mains__{task}__demained_psd", "µV²/Hz", "(C, F)", "Welch PSD after mains removal"),
        ("mains__{task}__line_hz", "Hz", "(1,)", "grid frequency removed in the phase"),
        ("chain__{task}__t_us", "µs", "(M,)", "time stamps of the stored chain example (2 s)"),
        ("chain__{task}__t0_us", "µs", "(1,)", "phase start (time origin of the chain view)"),
        ("chain__{task}__raw", "µV", "(M, C)", "chain example: raw EMG, median removed"),
        ("chain__{task}__demained", "µV", "(M, C)", "chain example: after mains removal"),
        ("chain__{task}__cleaned", "µV", "(M, C)", "chain example: band-passed (zero phase)"),
        ("chain__{task}__norm", "%", "(M, C)", "chain example: masked envelope, % of max recorded"),
        ("chain__{task}__masked", "0/1", "(M, C)", "chain example: analysis mask (D14)"),
        ("*", "", "", "see emg analysis (an03)"),
    ],
    "imu": [
        ("{task}__t_us", "µs", "(N,)", "IMU analysis grid (200 Hz)"),
        ("{task}__theta_opt", "°", "(N,)", "optical opening angle on the IMU grid"),
        ("{task}__theta_quat", "°", "(N,)", "IMU opening angle, quaternion estimator"),
        ("{task}__theta_comp", "°", "(N,)", "IMU opening angle, complementary estimator"),
        ("{task}__theta_tilt", "°", "(N,)", "IMU opening angle, tilt estimator"),
        ("{task}__omega_dps", "°/s", "(N,)", "IMU opening angular velocity (smoothed)"),
        ("{task}__acc_rel_ms2", "m/s²", "(N,)", "relative acceleration magnitude (mandible − cranium)"),
        ("{task}__quasi_static", "0/1", "(N,)", "quasi-static samples (tilt estimator valid)"),
        ("bandwidth__{task}__freq", "Hz", "(F,)", "frequency axis of the IMU angular-velocity PSD"),
        ("bandwidth__{task}__psd", "(°/s)²/Hz", "(F,)", "IMU angular-velocity PSD (≤ 50 Hz)"),
        ("bandwidth__{task}__coh_freq", "Hz", "(F,)", "frequency axis of the IMU–optical coherence"),
        ("bandwidth__{task}__coh", "", "(F,)", "bias-corrected IMU–optical coherence C′"),
        ("bandwidth__{task}__psd_imu40", "(°/s)²/Hz", "(F,)",
         "IMU PSD on the common 40 Hz grid (C′·P = optical-confirmed power)"),
    ],
    "timing": [
        ("ensemble__grid_ms", "ms", "(G,)", "time from mechanical onset"),
        ("ensemble__{task}__*", "dB", "(G,)",
         "median group envelope around the onset (agonist/antagonist, opening/closing; X2)"),
    ],
    "labels": [
        ("{task}__t_us", "µs", "(N,)", "label time stamps"),
        ("{task}__label", "class", "(N,)", "intent label from optical incisal velocity"),
    ],
    "uncertainty": [
        ("null__bench", "mm", "(sims,)", "H0 (perfect hinge + bench noise) incisor RMS"),
        ("null__invivo", "mm", "(sims,)", "H0 (perfect hinge + in-vivo noise) incisor RMS"),
        ("null__invivo_slow", "mm", "(sims,)",
         "H0 with the slow rest noise (3-s running median, 3-s blocks; D24)"),
        ("null__invivo_motion", "mm", "(sims,)",
         "H0 with the rest noise scaled to the motion error (D24; when configured)"),
        ("null__invivo_teeth", "mm", "(sims,)",
         "H0 in-vivo draws refitted with the teeth-only hinge (D25)"),
        ("null__bench_tmj", "mm", "(sims,)",
         "H0 (perfect hinge about the condyle axis + bench noise) incisor RMS of the hinge "
         "through the condyle points (D33)"),
        ("null__invivo_tmj", "mm", "(sims,)",
         "H0 (perfect hinge about the condyle axis + in-vivo noise) incisor RMS of the hinge "
         "through the condyle points (D33)"),
        ("ci__*", "mm", "(boot,)", "bootstrap draws of the observed incisor RMS per model"),
    ],
    "stats_kinematics": [
        ("grid_deg", "°", "(G,)", "opening-angle grid of the cohort coupling band"),
        ("*__subjects", "mm", "(participants, G)", "participant median curves"),
        ("*__median", "mm", "(G,)", "cohort median curve"),
        ("*__q1", "mm", "(G,)", "cohort first quartile"),
        ("*__q3", "mm", "(G,)", "cohort third quartile"),
    ],
    "stats_emg": [
        ("spm1__z", "t", "(101,)", "SPM{t}: masseter L vs R over the chewing cycle"),
        ("spm1__left", "%", "(participants, 101)", "participant mean profiles (left)"),
        ("spm1__right", "%", "(participants, 101)", "participant mean profiles (right)"),
        ("spm2__z", "t", "(101,)", "SPM{t}: openers vs closers over the open-close cycle"),
        ("spm2__openers", "peak = 1", "(participants, 101)", "peak-normalized opener profiles"),
        ("spm2__closers", "peak = 1", "(participants, 101)", "peak-normalized closer profiles"),
        ("varratio__*__between", "unit²", "(101,)", "between-participant variance"),
        ("varratio__*__within", "unit²", "(101,)", "mean within-participant variance"),
        ("varratio__*__ratio", "", "(101,)", "between / within variance"),
        ("spectral__grid_hz", "Hz", "(bins,)", "10-Hz bin centres of the spectral shape"),
        ("spectral__{task}__{muscle}__subjects", "log10 share", "(participants, bins)",
         "participant mean spectral shape"),
        ("spectral__{task}__{muscle}__mean", "log10 share", "(bins,)", "cohort mean"),
        ("spectral__{task}__{muscle}__sd", "log10 share", "(bins,)", "cohort SD"),
        ("spectral_spm1__z", "t", "(bins,)", "SPM{t} along frequency: masseter L vs R, chewing"),
    ],
    "ml_windows": [
        ("X_*", "standardized", "(windows, features[, steps])", "window features per block"),
        ("y", "class", "(windows,)", "label at the window end"),
        ("t_end_us", "µs", "(windows,)", "window end time"),
        ("phase", "index", "(windows,)", "phase index (ml.config.classes.PHASES)"),
        ("norm_*", "", "(features,)", "feature standardization (center/scale)"),
    ],
    "ml_eval": [
        ("confusion__*", "windows", "(7, 7)", "LOSO confusion (true × predicted), summed"),
        ("permutation_null", "BA", "(n_perm,)", "null of the cohort mean LOSO balanced accuracy"),
        ("pred__{pid}__*__t_us", "µs", "(windows,)", "window end times of the LOSO test fold"),
        ("pred__{pid}__*__y", "class", "(windows,)", "true class"),
        ("pred__{pid}__*__yhat", "class", "(windows,)", "LOSO prediction (primary model)"),
    ],
    "ml_online": [
        ("online__{pid}__t_us", "µs", "(windows,)", "decision (emission) time in the replay"),
        ("online__{pid}__t_end_us", "µs", "(windows,)", "end of the decided window"),
        ("online__{pid}__y", "class", "(windows,)", "true class"),
        ("online__{pid}__yhat", "class", "(windows,)", "smoothed stream decision"),
        ("online__{pid}__yhat_raw", "class", "(windows,)", "raw stream decision"),
        ("online__{pid}__phase", "index", "(windows,)", "phase index"),
        ("online__{pid}__release_us", "µs", "(decisions,)", "causal release decisions (X3)"),
    ],
}
# A family not matched by a specific pattern of its stage; "*" entries are catch-alls that
# the completeness test does not accept.
CATCH_ALL = "*"


def array_entry(stage: str, family: str) -> dict[str, str] | None:
    for pat, unit, shape, desc in ARRAYS.get(stage, []):
        if pat != CATCH_ALL and fnmatch.fnmatchcase(family, pat):
            return {"pattern": pat, "unit": unit, "shape": shape, "description": desc}
    return None


def column_entry(column: str) -> dict[str, Any]:
    from biomech import catalog

    e = catalog.get(column)
    return {"label": e.label, "unit": e.unit, "decimals": e.decimals,
            "description": e.desc, "catalogued": column in catalog.load()}


def describe(store: Any, stage: str | None = None) -> dict[str, Any]:
    """Data dictionary of the stored tables and array families (one unit per stage read)."""
    from biomech.results.access import family

    out: dict[str, Any] = {}
    for st in ([stage] if stage else store.stages()):
        tables: dict[str, Any] = {}
        arrays: dict[str, Any] = {}
        for unit in store.units(st):
            m = store.meta(st, unit)
            if m is None or m.status != "ok":
                continue
            for t in m.tables:
                if t not in tables:
                    df = store.read_table(st, t, [unit])
                    tables[t] = {c: {**column_entry(c), "dtype": str(df[c].dtype)}
                                 for c in df.columns if c != "unit"}
            if m.has_arrays:
                for k, v in store.read_arrays(st, unit).items():
                    f = family(k)
                    if f not in arrays:
                        arrays[f] = {**(array_entry(st, f) or {"description": ""}),
                                     "dtype": str(v.dtype), "example": k}
            if unit == "_cohort" or (tables and arrays):
                break
        out[st] = {"tables": tables, "arrays": arrays}
    return out
