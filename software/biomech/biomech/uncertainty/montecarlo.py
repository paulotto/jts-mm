# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Is the hinge residual above what measurement noise alone produces? (G′1, an06 §2)

H0 (a perfect hinge): for each participant, an ideal hinge motion is rebuilt
from that participant's own openings (fitted axis, measured opening angles),
pose noise is added frame by frame, the same low-pass and hinge fit run, and the
residual at the incisor recorded. Two noise levels:

* ``bench``   white noise of the optical precision from the bench (IEEE SENSORS 2025:
              0.182 mm, 0.126 deg, 3-D), split evenly over the axes;
* ``invivo``  the participant's own fast pose fluctuation at rest, before smoothing
              (``clinical.rest_residuals``: perturbation minus its 1-s running median),
              resampled in 1-s blocks (:class:`ResidualNoise`, E28) so the null has the
              measured size *and* temporal correlation; it also contains fast
              physiological micro-motion, so this null is conservative.

The observed residual is called "above noise" when it exceeds the 99th
percentile of the H0 distribution (primary, pre-specified). Its own uncertainty comes from
landmark perturbation (teeth: max(MAD, ``tooth_sd_mm``); condyles: ``tmj_sd_mm``) plus a
cycle bootstrap.

D24 (robustness of the null, reported next to the primary; :func:`hinge_nulls`):

* the Monte-Carlo p-value ``p_mc_<tag>`` = (k + 1)/(N + 1), k the simulations at or above
  the observed residual, N the finite simulations, stored next to the binary ``above_<tag>``;
* ``invivo_slow``   the rest residuals high-passed by a 3-s running median and resampled
                    in 3-s blocks (:data:`SLOW_HP_S`): noise down to about 0.3 Hz kept;
* ``invivo_motion`` the 1-s rest residuals scaled by ``motion_noise_scale`` (a cited
                    constant: the ratio of the optical tracking error during movement to
                    that at rest, from the systems paper); off while the parameter is null;
* ``invivo_teeth``  (D25) the teeth-only hinge fit (``hinge_teeth``) on the same noise
                    draws as ``invivo``, compared with the observed teeth-only residual.

D33 (post hoc, the anatomically specific form of C1; :func:`hinge_nulls`, ``tmj_null``):

* ``bench_tmj``, ``invivo_tmj``  H0 = a perfect hinge about the axis through the two condyle
                    points (the closed-pose condyle points of the openings, averaged, as
                    ``hinge_through`` uses them), rebuilt with the measured opening angles;
                    the same bench / primary in-vivo noise and filter; the angles refitted
                    about that fixed axis (``hinge_through``) and the incisor residual
                    compared with the observed ``hinge_tmj`` residual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from biomech.kinematics import se3
from biomech.kinematics.coupling import HINGE_POINTS, Opening, hinge_data, hinge_fits
from biomech.kinematics.hinge_fit import TEETH, fit_hinge, hinge_through
from biomech.kinematics.pose_chain import JawMotion

BENCH_T_MM = 0.182
BENCH_R_DEG = 0.126


def _hinge_pose(n: np.ndarray, s: np.ndarray, theta_rad: np.ndarray) -> np.ndarray:
    R = Rot.from_rotvec(np.outer(theta_rad, n)).as_matrix()
    T = np.tile(np.eye(4), (len(theta_rad), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = s - np.einsum("nij,j->ni", R, s)
    return T


class ResidualNoise:
    """Block bootstrap of measured pose residuals (``clinical.rest_residuals``, E28).

    ``draw(n, rng)`` concatenates blocks of ``block`` consecutive frames, each from a random
    run and start, into an (n, 6) sequence [rot deg xyz, trans mm xyz]: the noise keeps the
    measured size and its correlation within a block. Runs shorter than a block are used
    whole. ``scale`` multiplies every drawn residual (D24 motion-noise variant; 1 = as
    measured); the random numbers drawn do not depend on it."""

    def __init__(self, runs: list[np.ndarray], block: int, scale: float = 1.0) -> None:
        self.runs = [np.asarray(r, float) for r in runs if len(r)]
        self.block = max(int(block), 1)
        self.scale = float(scale)
        if not self.runs:
            raise ValueError("no rest residuals")

    def draw(self, n: int, rng: np.random.Generator) -> np.ndarray:
        parts, got = [], 0
        lens = np.array([len(r) for r in self.runs], float)
        p = lens / lens.sum()                     # runs drawn by their length
        while got < n:
            r = self.runs[int(rng.choice(len(self.runs), p=p))]
            L = min(self.block, len(r))
            a = int(rng.integers(0, len(r) - L + 1))
            parts.append(r[a:a + L])
            got += L
        out = np.concatenate(parts)[:n]
        return out * self.scale if self.scale != 1.0 else out


def _noisy(T: np.ndarray, sd_t: float | np.ndarray, sd_r_deg: float | np.ndarray,
           rng: np.random.Generator, noise: ResidualNoise | None = None) -> np.ndarray:
    """Pose noise ``T' = dT · T`` with ``dT = [dR, dt]`` (left multiplication, the
    parametrization ``clinical.pose_noise`` estimates): white with ``sd_*`` (scalars, same
    on every axis, or per-axis (3,)), or resampled residuals when ``noise`` is given."""
    n = len(T)
    if noise is not None:
        e = noise.draw(n, rng)
        rv, dt = np.radians(e[:, :3]), e[:, 3:]
    else:
        sd_r = np.broadcast_to(np.radians(np.asarray(sd_r_deg, float)), (3,))
        sd_tt = np.broadcast_to(np.asarray(sd_t, float), (3,))
        rv = rng.normal(0.0, 1.0, (n, 3)) * sd_r
        dt = rng.normal(0.0, 1.0, (n, 3)) * sd_tt
    dR = Rot.from_rotvec(rv).as_matrix()
    out = T.copy()
    out[:, :3, :3] = dR @ T[:, :3, :3]
    out[:, :3, 3] = np.einsum("nij,nj->ni", dR, T[:, :3, 3]) + dt
    out[~se3.valid(T)] = np.nan
    return out


def _stack(P: np.ndarray, Ts: list[np.ndarray], max_frames: int) -> tuple[np.ndarray, np.ndarray]:
    A, B = [], []
    if not Ts:
        return np.empty((0, len(P), 3)), np.empty((0, len(P), 3))
    for T in Ts:
        ok = np.flatnonzero(se3.valid(T))
        if ok.size < 3:
            continue
        pick = ok[np.unique(np.linspace(1, ok.size - 1, min(max_frames, ok.size - 1)).astype(int))]
        a = se3.apply(T[0], P)
        b = np.einsum("fij,kj->fki", T[pick, :3, :3], P) + T[pick, None, :3, 3]
        A.append(np.broadcast_to(a, b.shape))
        B.append(b)
    if not A:
        return np.empty((0, len(P), 3)), np.empty((0, len(P), 3))
    return np.ascontiguousarray(np.concatenate(A)), np.concatenate(B)


PAD_S = 1.0          # ideal motion continued on both sides of each opening before filtering
MIN_FRAMES = 10      # fewer valid frames in total: no simulation (NaN result)


def _padded_theta(theta_deg: np.ndarray, n_pad: int) -> np.ndarray:
    """Opening angle continued by ``n_pad`` frames on both sides by even reflection about
    the first and last frame (repeated if the opening is shorter than the pad).

    The opening starts at the closed rest pose and ends at the peak, both turning points
    (velocity ~ 0), and in ``open_close`` the motion before the start is the previous
    closing and after the peak the closing: the reflection is the hinge motion continued
    smoothly (position and velocity continuous), not a jump to a held pose.
    """
    th = np.asarray(theta_deg, float)
    if n_pad <= 0 or th.size < 2:
        return th.copy()
    return np.pad(th, n_pad, mode="symmetric")


def null_distribution(jm: JawMotion, ops: list[Opening], n_axis: np.ndarray, s_axis: np.ndarray,
                      sd_t: float | np.ndarray, sd_r_deg: float | np.ndarray, n_sims: int,
                      lowpass_hz: float, max_frames: int, seed: int,
                      lowpass_order: int = 2, pad_s: float = PAD_S,
                      noise: ResidualNoise | None = None) -> np.ndarray:
    """H0 distribution of the hinge-fit incisor RMS (``n_sims`` draws); see
    :func:`null_distributions`."""
    return null_distributions(jm, ops, n_axis, s_axis, sd_t, sd_r_deg, n_sims, lowpass_hz,
                              max_frames, seed, lowpass_order, pad_s, noise)["hinge_fit"]


def null_distributions(jm: JawMotion, ops: list[Opening], n_axis: np.ndarray,
                       s_axis: np.ndarray, sd_t: float | np.ndarray,
                       sd_r_deg: float | np.ndarray, n_sims: int, lowpass_hz: float,
                       max_frames: int, seed: int, lowpass_order: int = 2,
                       pad_s: float = PAD_S, noise: ResidualNoise | None = None,
                       teeth: bool = False, through_condyles: bool = False
                       ) -> dict[str, np.ndarray]:
    """H0 distribution of the hinge-fit incisor RMS (``n_sims`` draws), as
    ``{"hinge_fit": draws}``; with ``teeth`` also ``"hinge_teeth"``: the D25 teeth-only fit
    on the same noisy draws (the random numbers, and so ``hinge_fit``, do not change).
    ``through_condyles`` (D33): the draws are refitted about the fixed axis through the
    condyle points (``hinge_through``) instead, returned as ``{"hinge_tmj": draws}``; the
    caller passes that axis as ``n_axis``, ``s_axis``.

    Filtering matches the data (E19): the measured poses were low-passed as long continuous
    segments of the whole phase (``pose_chain``), so no opening sits at a filter edge. Each
    ideal opening is therefore built with ``pad_s`` of hinge motion continued on both sides
    (:func:`_padded_theta`), white pose noise is added to every frame (padding included),
    the padded sequence is low-passed once (same filter as the data), and only then is the
    opening cut out (``noise``: resampled residuals instead of white noise with ``sd_*``).
    Filtering each bare opening instead left openings shorter than 15
    frames unfiltered (``se3.lowpass`` ``min_len``) and put filter edge transients into all.
    Untracked frames (NaN angle) stay untracked.

    With no opening or fewer than ``MIN_FRAMES`` valid frames in total, returns
    ``n_sims`` NaN (A3).
    """
    rng = np.random.default_rng(seed)
    out = np.full(max(int(n_sims), 0), np.nan)
    res = ({"hinge_tmj": out} if through_condyles else
           {"hinge_fit": out} | ({"hinge_teeth": out.copy()} if teeth else {}))
    P = np.stack([jm.points_mp[r] for r in HINGE_POINTS])
    n_pad = max(int(round(pad_s * jm.fs)), 0)
    ideal, cuts = [], []
    for op in ops:
        # Untracked frames stay untracked (NaN pose) instead of jumping to 0 deg, which
        # the low-pass would smear into a fake movement.
        th = np.radians(_padded_theta(op.theta, n_pad))
        pad = (th.size - len(op.theta)) // 2
        T = np.full((len(th), 4, 4), np.nan)
        ok = np.isfinite(th)
        if ok.any() and se3.valid(op.T[:1])[0]:
            T[ok] = se3.compose(_hinge_pose(n_axis, s_axis, th[ok]), op.T[0][None])
        ideal.append(T)
        cuts.append(slice(pad, pad + len(op.theta)))
    n_valid = sum(int(se3.valid(T[c]).sum()) for T, c in zip(ideal, cuts))
    if not ideal or n_valid < MIN_FRAMES:
        return res
    # All noise draws first, in the order of the former per-simulation loop (simulation,
    # then opening), so the generator yields the same numbers; then every opening is
    # low-passed for all simulations at once (se3.lowpass_batch == se3.lowpass per draw).
    noisy = [[_noisy(T, sd_t, sd_r_deg, rng, noise) for T in ideal] for _ in range(out.size)]
    # The same filter as the measured poses (pose_chain), so null and data match.
    filt = [se3.lowpass_batch(np.stack([sim[j] for sim in noisy]), jm.fs, lowpass_hz,
                              lowpass_order)[:, c] if out.size else None
            for j, c in enumerate(cuts)]
    del noisy
    for k in range(out.size):
        Ts = [f[k] for f in filt]
        A, B = _stack(P, Ts, max_frames)
        if A.shape[0] < MIN_FRAMES:
            continue
        if through_condyles:
            out[k] = hinge_through(A, B, A[:, 3].mean(axis=0), A[:, 4].mean(axis=0)).rms(0)
            continue
        out[k] = fit_hinge(A, B, s0=s_axis, n0=n_axis).rms(0)
        if teeth:
            res["hinge_teeth"][k] = fit_hinge(A, B, s0=s_axis, n0=n_axis, points=TEETH,
                                              model="hinge_teeth").rms(0)
    return res


def observed_ci(jm: JawMotion, ops: list[Opening], n_boot: int, tooth_sd: float, tmj_sd: float,
                max_frames: int, seed: int) -> dict[str, np.ndarray]:
    """Bootstrap over cycles x landmark perturbation; returns per-model incisor RMS draws.

    With no opening or fewer than ``MIN_FRAMES`` stacked frames: empty draws (A3).
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"hinge_fit": [], "hinge_tmj": [], "rot_trans": []}
    base = np.stack([jm.points_mp[r] for r in HINGE_POINTS])
    def mad(name: str) -> float:
        it = jm.landmarks.items.get(name)
        return float(it.mad_mm) if it is not None and np.isfinite(it.mad_mm) else 0.0

    # Each landmark: its own digitization MAD, at least the nominal SD (an06 §2).
    sd = np.array([max(tooth_sd, mad(jm.lower.incisor)), max(tooth_sd, mad(jm.lower.left)),
                   max(tooth_sd, mad(jm.lower.right)), max(tmj_sd, mad("tmj_left")),
                   max(tmj_sd, mad("tmj_right"))])
    # Resamples start from the full-data fits (warm start): same optimum, fewer steps.
    A0, B0 = _stack(base, [op.T for op in ops], max_frames)
    if not ops or A0.shape[0] < MIN_FRAMES:
        return {k: np.array(v) for k, v in draws.items()}
    init = hinge_fits(A0, B0, teeth=False)
    for _ in range(n_boot):
        P = base + rng.normal(0, 1, base.shape) * sd[:, None]
        pick = rng.integers(0, len(ops), len(ops))
        A, B = _stack(P, [ops[i].T for i in pick], max_frames)
        if A.shape[0] < MIN_FRAMES:
            continue
        for name, r in hinge_fits(A, B, init, teeth=False).items():
            draws[name].append(r.rms(0))
    return {k: np.array(v) for k, v in draws.items()}


def mc_pvalue(observed: float, draws: np.ndarray) -> float:
    """Monte-Carlo p-value (k + 1)/(N + 1) (D24): k the finite draws at or above
    ``observed``, N the finite draws; NaN without draws or observation. Never 0: with N
    simulations the smallest attainable value is 1/(N + 1)."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(observed):
        return np.nan
    return float((np.count_nonzero(d >= observed) + 1) / (d.size + 1))


def model_of(tag: str) -> str:
    """The observed model a null ``tag`` is compared with: ``*_teeth`` -> the teeth-only
    fit, ``*_tmj`` -> the hinge through the condyle points (D33), every other null -> the
    best-fit hinge."""
    if tag.endswith("_teeth"):
        return "hinge_teeth"
    return "hinge_tmj" if tag.endswith("_tmj") else "hinge_fit"


def condyle_axis(jm: JawMotion, ops: list[Opening],
                 max_frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(n, s) of the axis through the condyle points exactly as ``hinge_through`` gets it on
    the measured data (``coupling.hinge_fits``): the closed-pose condyle points averaged over
    the frames of ``hinge_data`` (each opening weighted by its subsampled frames), direction
    left minus right, so a positive measured opening angle rotates the ideal jaw open as in
    the data; None without usable openings."""
    A, _, _ = hinge_data(jm, ops, max_frames)
    if not len(A):
        return None
    left, right = A[:, 3].mean(axis=0), A[:, 4].mean(axis=0)
    n = left - right
    return n / np.linalg.norm(n), 0.5 * (left + right)


def summarize(observed: dict[str, float], null: dict[str, np.ndarray],
              ci: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in observed.items():
        out[f"obs_{k}_mm"] = v
        if k in ci and ci[k].size:
            out[f"obs_{k}_ci_lo"], out[f"obs_{k}_ci_hi"] = np.percentile(ci[k], [2.5, 97.5])
    for tag, d in null.items():
        d = np.asarray(d, float)
        d = d[np.isfinite(d)]
        obs = float(observed.get(model_of(tag), np.nan))
        out[f"n_sims_{tag}"] = int(d.size)
        if d.size == 0:                       # no simulation possible (A3)
            out[f"null_{tag}_p50_mm"] = out[f"null_{tag}_p99_mm"] = np.nan
            out[f"above_{tag}"] = False
            out[f"ratio_{tag}"] = np.nan
            out[f"p_mc_{tag}"] = np.nan
            continue
        p99 = float(np.percentile(d, 99))
        out[f"null_{tag}_p50_mm"] = float(np.median(d))
        out[f"null_{tag}_p99_mm"] = p99
        out[f"above_{tag}"] = bool(obs > p99)
        out[f"ratio_{tag}"] = float(obs / p99) if p99 > 0 else np.nan
        out[f"p_mc_{tag}"] = mc_pvalue(obs, d)
    return out


SLOW_HP_S = 3.0      # D24 slow-noise variant: 3-s running median and 3-s blocks


def rest_noise(T_rest: np.ndarray | None, fs: float, hp_s: float, scale: float = 1.0
               ) -> ResidualNoise | None:
    """The participant's rest residuals (``clinical.rest_residuals`` with a ``hp_s`` running
    median) as block-resampled noise with blocks of the same length; None without a usable
    rest phase."""
    from biomech.kinematics.clinical import rest_residuals
    from biomech.timebase import odd_window

    if T_rest is None or not len(T_rest):
        return None
    runs = rest_residuals(T_rest, fs, hp_s)
    return ResidualNoise(runs, odd_window(hp_s, fs), scale) if runs else None


def hinge_nulls(jm: JawMotion, ops: list[Opening], n_axis: np.ndarray, s_axis: np.ndarray,
                T_rest: np.ndarray | None, floor_sd_t: np.ndarray, floor_sd_r_deg: np.ndarray,
                p: dict[str, Any], lowpass_hz: float, lowpass_order: int, max_frames: int,
                seed: int) -> dict[str, np.ndarray]:
    """Every H0 distribution of the uncertainty stage, by noise tag.

    ``p``: ``stages.uncertainty`` (``n_sims``; ``slow_noise``: compute ``invivo_slow``;
    ``motion_noise_scale``: null = no ``invivo_motion``; ``teeth_null``: compute
    ``invivo_teeth``; ``tmj_null``: compute ``bench_tmj`` and ``invivo_tmj``, D33). Seeds:
    bench ``seed``, invivo ``seed + 1`` (the CI uses ``seed + 2``), slow ``seed + 3``, motion
    ``seed + 4``, bench_tmj ``seed + 5``, invivo_tmj ``seed + 6``, so adding a variant never
    changes another null.
    The in-vivo nulls resample the 1-s rest residuals (block resampling, E28). Without usable
    1-s rest residuals there is no in-vivo null: the floor SDs are NaN, every draw is NaN and
    the participant counts as untested. Only ``invivo_slow`` falls back to white noise with
    the 1-s residual SDs, when no rest run reaches 3 s.
    """
    from biomech.kinematics.clinical import REST_HP_S

    n = int(p["n_sims"])
    args = (lowpass_hz, max_frames)

    def invivo(noise: ResidualNoise | None, s: int, teeth: bool = False):
        return null_distributions(jm, ops, n_axis, s_axis, floor_sd_t, floor_sd_r_deg, n,
                                  *args, s, lowpass_order, noise=noise, teeth=teeth)

    out = {"bench": null_distribution(jm, ops, n_axis, s_axis, BENCH_T_MM / np.sqrt(3),
                                      BENCH_R_DEG / np.sqrt(3), n, *args, seed,
                                      lowpass_order)}
    teeth = bool(p.get("teeth_null", True))
    prim = invivo(rest_noise(T_rest, jm.fs, REST_HP_S), seed + 1, teeth)
    out["invivo"] = prim["hinge_fit"]
    if teeth:
        out["invivo_teeth"] = prim["hinge_teeth"]
    if p.get("slow_noise", True):
        out["invivo_slow"] = invivo(rest_noise(T_rest, jm.fs, SLOW_HP_S), seed + 3)["hinge_fit"]
    scale = p.get("motion_noise_scale")
    if scale is not None:
        nz = rest_noise(T_rest, jm.fs, REST_HP_S, float(scale))
        # without rest residuals: the stored floor, scaled the same way
        out["invivo_motion"] = (invivo(nz, seed + 4) if nz is not None else
                                null_distributions(jm, ops, n_axis, s_axis,
                                                   floor_sd_t * float(scale),
                                                   floor_sd_r_deg * float(scale), n, *args,
                                                   seed + 4, lowpass_order))["hinge_fit"]
    ax = condyle_axis(jm, ops, max_frames) if p.get("tmj_null", False) else None
    if ax is not None:
        n_c, s_c = ax
        out["bench_tmj"] = null_distributions(
            jm, ops, n_c, s_c, BENCH_T_MM / np.sqrt(3), BENCH_R_DEG / np.sqrt(3), n, *args,
            seed + 5, lowpass_order, through_condyles=True)["hinge_tmj"]
        out["invivo_tmj"] = null_distributions(
            jm, ops, n_c, s_c, floor_sd_t, floor_sd_r_deg, n, *args, seed + 6, lowpass_order,
            noise=rest_noise(T_rest, jm.fs, REST_HP_S), through_condyles=True)["hinge_tmj"]
    return out
