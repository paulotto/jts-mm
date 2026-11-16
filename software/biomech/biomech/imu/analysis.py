# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Wearable proxies from the two IMUs (C3a, an04 §4-5) and mechanical-onset signals.

Per participant:

1. calibration (:mod:`biomech.imu.calibration`) on ``calibration_phase``;
2. on a 200 Hz grid per motion phase, three opening-angle estimators, all
   about the maxillary y axis relative to the optical closed pose:
   * ``quat``  calibrated relative orientation from the game rotation vectors;
   * ``tilt``  gravity seen by both IMUs (raw accel incl. gravity): angle that
     turns the head's gravity into the mandible's, valid only while both are
     quasi-static -- drift-free and heading-free;
   * ``comp``  complementary fusion of the two (``tau_s``), causal;
3. agreement with optical per task (continuous and cycle-referenced), lag and
   drift;
4. mechanical-onset signals for the timing stage: opening angular velocity
   (deg/s, from ``quat``) and the magnitude of the mandible-minus-head linear
   acceleration (gravity removed per IMU, heading offset from the calibration).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from jtsmm.imu.batch import quat_to_matrix
from jtsmm.imu.tilt import G, complementary, quasi_static, signed_angle_about, unit

from biomech import timebase
from biomech.imu.calibration import ImuCalibration, fit
from biomech.imu.preprocess import ImuStream, load
from biomech.io.session import SessionData
from biomech.timing.coupling import lagged_corr, peak_lag

EY = np.array([0.0, 1.0, 0.0])
ESTIMATORS = ("quat", "tilt", "comp")


def _theta_y(R: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    D = R @ R_ref.T
    out = np.full(len(R), np.nan)
    ok = np.isfinite(D.reshape(len(D), -1)).all(1)
    if ok.any():
        out[ok] = np.degrees(Rot.from_matrix(D[ok]).as_rotvec()[:, 1])
    return out


def phase_signals(cal: ImuCalibration, mta: ImuStream, cra: ImuStream, t_us: np.ndarray,
                  R_closed: np.ndarray, p: dict[str, Any]) -> dict[str, np.ndarray]:
    qc, qm = cra.quat_at(t_us), mta.quat_at(t_us)
    R_hat = cal.predict(qc, qm)
    out: dict[str, np.ndarray] = {"t_us": t_us}
    out["theta_quat"] = _theta_y(R_hat, R_closed)
    ac, am = cra.accel_at(t_us), mta.accel_at(t_us)
    g_head = unit(ac) @ cal.A.T                         # in maxillary coordinates
    g_mand = unit(am) @ (R_closed @ cal.X.T).T          # mandible frame, at-closure axes
    out["theta_tilt"] = -np.degrees(signed_angle_about(g_head, g_mand, EY))
    valid = quasi_static(ac, p["quasi_static_tol"]) & quasi_static(am, p["quasi_static_tol"])
    out["quasi_static"] = valid.astype(np.uint8)
    t_s = t_us * 1e-6
    out["theta_comp"] = complementary(t_s, out["theta_quat"], out["theta_tilt"], valid,
                                      p["tau_s"])
    w = timebase.odd_window(p["omega_smooth_s"], p["grid_hz"])   # odd: centred (E17)
    # Per finite segment: a gap never turns into a derivative spike (audit P2-8).
    out["omega_dps"] = timebase.gradient(t_us, out["theta_quat"], smooth_n=w)
    # Linear acceleration: gravity removed per IMU, mandible minus head in the head's world.
    Rc, Rm = quat_to_matrix(qc), quat_to_matrix(qm)
    gz = np.array([0.0, 0.0, G])
    lin_c = np.einsum("nij,nj->ni", Rc, ac) - gz
    lin_m = np.einsum("nij,nj->ni", Rm, am) - gz
    Rz = Rot.from_rotvec([0, 0, cal.psi_rad]).as_matrix()
    out["acc_rel_ms2"] = np.linalg.norm(lin_m @ Rz.T - lin_c, axis=1)
    return out


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan


def _lag_ms(t_us: np.ndarray, ref: np.ndarray, x: np.ndarray, max_ms: float = 100.0,
            min_pairs: int = 50) -> float:
    """Lag of ``x`` behind ``ref`` (ms, > 0 = x late) by cross-correlating derivatives.

    NaN-aware (E23): the derivatives are taken per gap-free segment (``timebase.gradient``,
    NaN stays NaN) and each lag's normalized correlation uses only the sample pairs where
    both are valid; formerly gaps were zero-filled before the gradient, which put a step
    (a derivative spike) at every gap edge into the correlation.
    """
    ok = np.isfinite(ref) & np.isfinite(x)
    if ok.sum() < min_pairs:
        return np.nan
    t = np.asarray(t_us, dtype=np.float64)
    dt = np.median(np.diff(t)) * 1e-6
    a = timebase.gradient(t, np.where(ok, ref, np.nan))
    b = timebase.gradient(t, np.where(ok, x, np.nan))
    k = int(round(max_ms * 1e-3 / dt))
    lags, c = lagged_corr(a, b, k, min_pairs)          # one helper (timing.coupling, X6)
    lag, _ = peak_lag(lags, c)
    return float(lag * dt * 1e3) if np.isfinite(lag) else np.nan


def agreement(sig: dict[str, np.ndarray]) -> dict[str, float]:
    """Continuous agreement of each estimator with the optical angle: RMSE and bias (deg).

    ``tilt`` is defined only while both IMUs are quasi-static, so its agreement uses those
    samples only (E23); elsewhere the gravity angle is contaminated by linear acceleration.
    """
    out: dict[str, float] = {}
    qs = np.asarray(sig["quasi_static"]).astype(bool)
    for e in ESTIMATORS:
        th = np.asarray(sig[f"theta_{e}"], dtype=np.float64)
        if e == "tilt":
            th = np.where(qs, th, np.nan)
        d = th - sig["theta_opt"]
        out[f"rmse_{e}_deg"] = _rmse(th, sig["theta_opt"])
        out[f"bias_{e}_deg"] = float(np.nanmean(d)) if np.isfinite(d).any() else np.nan
    return out


def _optical_gap_us(t_opt_us: np.ndarray, factor: float = 2.5) -> float:
    """Longest interpolation gap of the optical angle: ``factor`` x its median frame
    interval (E23; formerly a fixed 60 ms, i.e. 2.4 frames at 40 Hz only)."""
    t = np.asarray(t_opt_us, dtype=np.float64)
    return float(factor * np.median(np.diff(t))) if t.size > 1 else np.inf


def own_model_openings(cyc: pd.DataFrame, opening_model: pd.DataFrame,
                       opening_loco: pd.DataFrame | None = None, estimators=ESTIMATORS
                       ) -> pd.DataFrame:
    """The participant's own opening model (opening = a1·θ + a2·θ², kinematics
    ``opening_model``) applied to the IMU cycle peak angles, in place:

    * ``opening_subject_{e}_mm`` (D26): out of sample. An open_close cycle whose opening
      entered the fit is predicted by the model fitted without it (kinematics
      ``opening_model_loco``, leave-one-cycle-out; NaN when that fit was not possible); every
      other cycle (chewing, open_close cycles not in the fit) never entered the fit and gets
      the full model;
    * ``opening_subject_insample_{e}_mm``: the full model for every cycle (in-sample for the
      fitted open_close cycles; the r9 value, kept as a sensitivity).
    """
    if cyc.empty or opening_model.empty:
        return cyc
    a1 = float(opening_model.a1_mm_per_deg.iloc[0])
    a2 = float(opening_model.a2_mm_per_deg2.iloc[0])
    b1 = pd.Series(a1, index=cyc.index, dtype=float)
    b2 = pd.Series(a2, index=cyc.index, dtype=float)
    if opening_loco is not None and not opening_loco.empty:
        lo = opening_loco.drop_duplicates("cycle").set_index("cycle")
        fitted = (cyc.task == "open_close") & cyc.cycle.isin(lo.index)
        b1[fitted] = cyc.cycle[fitted].map(lo.a1_mm_per_deg).astype(float)
        b2[fitted] = cyc.cycle[fitted].map(lo.a2_mm_per_deg2).astype(float)
    for e in estimators:
        th = cyc[f"theta_peak_{e}_deg"]
        cyc[f"opening_subject_{e}_mm"] = b1 * th + b2 * th ** 2
        cyc[f"opening_subject_insample_{e}_mm"] = a1 * th + a2 * th ** 2
    return cyc


def analyse(sd: SessionData, pid: str, kin_arrays: dict[str, Any], kin_cycles: pd.DataFrame,
            opening_model: pd.DataFrame, p: dict[str, Any],
            opening_loco: pd.DataFrame | None = None
            ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    mta, cra = load(sd, "mta"), load(sd, "cra")
    if mta is None or cra is None:
        raise LookupError("IMU streams missing")
    shift = int(p.get("latency_shift_us") or 0)
    if shift:                       # X11 `imu_latency` variant: IMU stamps moved earlier
        from dataclasses import replace

        mta, cra = (replace(s, t_q=s.t_q - shift, t_a=s.t_a - shift) for s in (mta, cra))
    cal_ph = p["calibration_phase"]
    tc = kin_arrays[f"{cal_ph}__t_us"]
    R_opt = kin_arrays[f"{cal_ph}__T"][:, :3, :3].astype(np.float64)
    cal = fit(cra.quat_at(tc), mta.quat_at(tc), R_opt)
    R_closed = kin_arrays["T_closed"][:3, :3]
    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {}
    tables["calibration"] = pd.DataFrame([{
        "pid": pid, "phase": cal_ph, "rms_deg": cal.rms_deg, "n": cal.n, "init": cal.init,
        "psi_deg": float(np.degrees(cal.psi_rad)),
        # The fitted mountings, as rotation vectors (deg), for provenance and the
        # synthetic ground-truth test.
        **{f"X_rv{a}_deg": float(v) for a, v in zip("xyz", np.degrees(
            Rot.from_matrix(cal.X).as_rotvec()))},
        **{f"A_rv{a}_deg": float(v) for a, v in zip("xyz", np.degrees(
            Rot.from_matrix(cal.A).as_rotvec()))},
        "dropped_q_mta": mta.n_dropped_q, "dropped_q_cra": cra.n_dropped_q,
        "dropped_a_mta": mta.n_dropped_a, "dropped_a_cra": cra.n_dropped_a}])

    step = int(1e6 / p["grid_hz"])
    agree, cyc_rows, diffs = [], [], []
    for ph in sd.phases:
        if ph.kind != "experiment" or f"{ph.name}__t_us" not in kin_arrays:
            continue
        tg = np.arange(ph.t_start_us, ph.t_end_us, step, dtype=np.int64)
        sig = phase_signals(cal, mta, cra, tg, R_closed, p)
        ko = kin_arrays[f"{ph.name}__t_us"].astype(np.int64)
        th_o = kin_arrays[f"{ph.name}__theta_deg"].astype(np.float64)
        ok = np.isfinite(th_o)
        sig["theta_opt"] = np.interp(tg, ko[ok], th_o[ok], left=np.nan, right=np.nan) \
            if ok.sum() > 2 else np.full(tg.shape, np.nan)
        # NaN out optical gaps longer than 2.5 optical frame intervals
        j = np.clip(np.searchsorted(ko[ok], tg), 1, max(ok.sum() - 1, 1))
        if ok.sum() > 2:
            gap = ko[ok][j] - ko[ok][j - 1]
            sig["theta_opt"][gap > _optical_gap_us(ko)] = np.nan
        for k, v in sig.items():
            arrays[f"{ph.name}__{k}"] = v.astype(np.float32) if v.dtype.kind == "f" else v
        row = {"pid": pid, "task": ph.name, "is_calibration": ph.name == cal_ph,
               "quasi_static_frac": float(sig["quasi_static"].mean()),
               "lag_ms": _lag_ms(tg, sig["theta_opt"], sig["theta_quat"])}
        row.update(agreement(sig))
        agree.append(row)
        d = sig["theta_quat"] - sig["theta_opt"]
        good = np.isfinite(d)
        diffs.append((tg[good] * 1e-6, d[good]))

        cyc = kin_cycles[(kin_cycles.task == ph.name) & kin_cycles.ok] if not kin_cycles.empty \
            else kin_cycles
        for _, c in cyc.iterrows():
            sel = (tg >= c.t_start_us) & (tg <= c.t_end_us)
            if sel.sum() < 10:
                continue
            r = {"pid": pid, "task": ph.name, "side": c.side, "cycle": int(c.cycle),
                 "opening_opt_mm": float(c.opening_mm)}
            o = sig["theta_opt"][sel] - sig["theta_opt"][sel][0]
            r["theta_peak_opt_deg"] = float(np.nanmax(o)) if np.isfinite(o).any() else np.nan
            qs = sig["quasi_static"][sel].astype(bool)
            for e in ESTIMATORS:
                x = sig[f"theta_{e}"][sel].astype(np.float64)
                x = x - x[0]
                if e == "tilt":            # only defined while both IMUs are quasi-static
                    x = np.where(qs, x, np.nan)
                r[f"theta_peak_{e}_deg"] = float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
                r[f"rmse_cycle_{e}_deg"] = _rmse(x, o)
            cyc_rows.append(r)

    tables["agreement"] = pd.DataFrame(agree)
    tables["cycles"] = own_model_openings(pd.DataFrame(cyc_rows), opening_model, opening_loco)
    if diffs:
        t = np.concatenate([d[0] for d in diffs])
        d = np.concatenate([d[1] for d in diffs])
        slope = np.polyfit(t, d, 1)[0] * 60.0 if t.size > 10 else np.nan
        tables["calibration"]["drift_quat_deg_per_min"] = float(slope)
    tables["bandwidth"] = movement_bandwidth(pid, arrays, p["grid_hz"], kin_arrays,
                                             spectra=arrays)
    return tables, arrays


COHERENCE_MIN = 0.5           # IMU-optical coherence that counts as shared (real) movement
MIN_COHERENT_BAND_HZ = 2.0    # below: the opening angle does not carry this task's motion


def movement_bandwidth(pid: str, arrays: dict[str, np.ndarray], grid_hz: float,
                       kin_arrays: dict[str, np.ndarray] | None = None,
                       spectra: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Frequencies holding 50/95/99 % of the jaw angular-velocity power, per task (X1, D12).

    ``f50/f95/f99_hz``: from the IMU opening angular velocity (200 Hz grid, 20 ms
    smoothing), 4-s Welch windows over gap-free stretches. f_q is the smallest f with
    ∫₀^f P = q·∫P.

    Noise check (D12): the IMU spectrum holds sensor-specific noise that appears only
    during motion (tooth-contact transients, sensor fusion), so a rest-phase noise floor
    does not remove it. An independent sensor does: the unsmoothed optical opening angle
    (40 Hz, kinematics ``__theta_raw_deg``) and the IMU are compared on a common 40 Hz grid.
    Only the part of the IMU power that the optical signal shares counts as movement:
    P_coh(f) = C(f)·P_imu(f), C = magnitude-squared coherence (Welch, 4-s segments),
    0 < f < 20 Hz, with C bias-corrected for K averaged segments,
    C' = max(0, (C − 1/K)/(1 − 1/K)). ``f95_coh_hz`` / ``f99_coh_hz`` are the quantiles of
    C'·P_imu; ``coh_band_hz`` is the upper edge of the band, from the lowest frequencies
    up, where C' ≥ ``COHERENCE_MIN``; ``noise_limited`` when that band ends below
    ``MIN_COHERENT_BAND_HZ`` (the opening angle does not carry the task's motion, e.g.
    protrusion), so the task does not enter the bandwidth requirement.

    D27 sensitivity, next to the coherence-weighted primary: ``f95_hz`` is the IMU alone;
    ``f95_opt_hz`` the optics alone, the f95 of the unsmoothed optical angular-velocity
    power on the same 40 Hz grid and band (0 < f < 20 Hz; its 40 Hz noise floor biases it
    upward, as the IMU's motion noise biases ``f95_hz``).

    ``spectra`` (a dict, filled in place) receives the curves behind the numbers for the
    views: ``bandwidth__{task}__freq`` / ``__psd`` (IMU angular-velocity PSD, <= 50 Hz) and
    ``bandwidth__{task}__coh_freq`` / ``__coh`` (bias-corrected coherence C') /
    ``__psd_imu40`` (IMU PSD on the common 40 Hz grid, 0-20 Hz).
    """
    from scipy.signal import coherence, welch

    from biomech.spectral import power_quantiles, welch_masked

    rows = []
    for key in [k for k in arrays if k.endswith("__omega_dps")]:
        task = key[: -len("__omega_dps")]
        w = np.asarray(arrays[key], float)
        if np.isfinite(w).sum() < 4 * grid_hz:
            continue
        f, psd, per = welch_masked(w, grid_hz, None, seg_s=4.0)
        q = power_quantiles(f, psd[0], (0.5, 0.95, 0.99), f_min=0.05)
        if spectra is not None:
            keep = f <= 50.0
            spectra[f"bandwidth__{task}__freq"] = f[keep].astype(np.float32)
            spectra[f"bandwidth__{task}__psd"] = psd[0][keep].astype(np.float32)
        row = {"pid": pid, "task": task, "f50_hz": q[0], "f95_hz": q[1], "f99_hz": q[2],
               "n_windows": int(np.isfinite(per[0, :, 0]).sum()),
               "f95_coh_hz": np.nan, "f99_coh_hz": np.nan, "coh_band_hz": np.nan,
               "f95_opt_hz": np.nan, "noise_limited": True}
        k_t, k_th = f"{task}__t_us", f"{task}__theta_raw_deg"
        if kin_arrays is not None and k_th in kin_arrays and f"{task}__t_us" in arrays:
            to = np.asarray(kin_arrays[k_t], float)
            tho = np.asarray(kin_arrays[k_th], float)
            ti = np.asarray(arrays[f"{task}__t_us"], float)
            ok = np.isfinite(tho)
            if ok.sum() > 8 * 40:
                g = np.arange(max(to[ok][0], ti[0]), min(to[ok][-1], ti[-1]), 1e6 / 40.0)
                wo = np.gradient(np.interp(g, to[ok], tho[ok]), 1 / 40.0)
                fin = np.isfinite(w)
                wi = np.interp(g, ti[fin], _antialias(ti[fin], w[fin], 16.0))
                if g.size >= 2 * 160:
                    fc, C = coherence(wo, wi, 40.0, nperseg=160)
                    _, Pi = welch(wi, 40.0, nperseg=160)
                    _, Po = welch(wo, 40.0, nperseg=160)
                    # Unrelated signals have E[C] ≈ 1/K for K averaged segments (50 %
                    # overlap): remove that bias before weighting the power.
                    K = max(1, 2 * (g.size // 160) - 1)
                    Cu = np.clip((C - 1.0 / K) / (1.0 - 1.0 / K), 0.0, 1.0) if K > 1 else C
                    band = (fc > 0) & (fc < 20.0)
                    hi = np.flatnonzero(band)
                    run = hi[np.cumprod(Cu[hi] >= COHERENCE_MIN).astype(bool)]
                    edge = float(fc[run[-1]]) if run.size else 0.0
                    row["coh_band_hz"] = edge
                    if spectra is not None:
                        spectra[f"bandwidth__{task}__coh_freq"] = fc[band].astype(np.float32)
                        spectra[f"bandwidth__{task}__coh"] = Cu[band].astype(np.float32)
                        spectra[f"bandwidth__{task}__psd_imu40"] = Pi[band].astype(np.float32)
                    row["noise_limited"] = bool(edge < MIN_COHERENT_BAND_HZ)
                    if Po[band].sum() > 0:          # D27: the optics alone
                        row["f95_opt_hz"] = power_quantiles(fc[band], Po[band], (0.95,),
                                                            f_min=0.05)[0]
                    if (Cu * Pi)[band].sum() > 0:
                        qc = power_quantiles(fc[band], (Cu * Pi)[band], (0.95, 0.99),
                                             f_min=0.05)
                        row.update(f95_coh_hz=qc[0], f99_coh_hz=qc[1])
        rows.append(row)
    return pd.DataFrame(rows)


def _antialias(t_us: np.ndarray, x: np.ndarray, fc_hz: float) -> np.ndarray:
    """``x`` low-passed (zero-phase 4th-order Butterworth at ``fc_hz``) before it is put on a
    coarser grid (E29: the 40 Hz coherence grid must not alias IMU content above 20 Hz)."""
    from scipy.signal import butter, filtfilt

    if x.size < 32:
        return x
    fs = 1e6 / float(np.median(np.diff(t_us)))
    if not np.isfinite(fs) or fc_hz >= 0.5 * fs:
        return x
    b, a = butter(4, fc_hz / (0.5 * fs))
    return filtfilt(b, a, x)
