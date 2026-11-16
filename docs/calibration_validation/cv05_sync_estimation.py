#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reference implementation of the offline time registration in cv05 §3.

Registers a foreign clock (Qualisys, Delsys) to the JTS-MM master clock from
short synchronization sequences recorded at the START and END of a take:

    t_X = (1 + eps) * t_M + theta          (affine clock model, per take)

Pipeline per take (cv05 §3.4 / §8.1):
  1. frame-invariant signal per system: geodesic angle phi(t) of the relative
     rotation R_rel = R_CRA^T R_MTA (jaw sync) or of the wand rotation, w.r.t.
     the mean rotation of the REST segment that precedes the motion;
  2. detect the sync windows (rest -> motion -> rest) in each stream;
  3. coarse lag: normalized cross-correlation on a uniform grid (unambiguous
     because the sync motion is aperiodic and bounded by rest);
  4. fine lag: least squares between JTS-MM phi at the exposure centroids and
     the blur-matched (box filter over tau_exp), interpolated reference phi;
     Gauss-Newton standard error (optimistic by ~2x in simulation: use the
     spread of the hold-out residuals as the reported uncertainty);
  5. clock model from the start and end lags (weighted line fit);
  6. optional hold-out windows (inside the take) -> residuals = registration
     uncertainty. They are diagnostics only and are never fed back
     (circularity rule, cv05 §3.4).
  sEMG: envelope cross-correlation of the clench bursts (same clock model).

numpy only.  Run `python3 cv05_sync_estimation.py` for a synthetic self-test.
"""
from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# rotations
# ----------------------------------------------------------------------------
def project_SO3(M):
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def chordal_mean(R):
    """Mean rotation of a stack (N,3,3), e.g. over a rest segment."""
    return project_SO3(np.nanmean(R, axis=0))

def geodesic_angle(R, R_ref):
    """phi = angle(R R_ref^T) in degrees; invariant to world frame and body offset."""
    tr = np.einsum('nij,ij->n', R, R_ref)          # trace(R R_ref^T)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))

def relative_rotation(R_cra, R_mta):
    return np.einsum('nji,njk->nik', R_cra, R_mta)  # R_cra^T R_mta

def angular_speed(t, R):
    """|omega| (deg/s) from consecutive rotations; also invariant."""
    dR = np.einsum('nij,nkj->nik', R[1:], R[:-1])
    ang = geodesic_angle(dR, np.eye(3))
    return 0.5 * (t[1:] + t[:-1]), ang / np.diff(t)

# ----------------------------------------------------------------------------
# sync-window detection: rest (>= min_rest) -> motion -> rest (>= min_rest)
# ----------------------------------------------------------------------------
def find_sync_windows(t, speed, thr=8.0, smooth=0.3, min_rest=1.5, min_motion=2.0, max_motion=15.0,
                      max_gap=0.7, min_blip=0.3):
    """Return a list of (t_rest_start, t_motion_start, t_motion_end, t_rest_end).
    The angular speed is smoothed first (frame-to-frame noise of the 40 Hz pose
    alone gives several deg/s)."""
    n = max(1, int(round(smooth / np.median(np.diff(t)))))
    speed = np.convolve(speed, np.ones(n) / n, mode='same')
    moving = speed > thr
    dt = np.median(np.diff(t))
    def _runs(b):
        e = np.flatnonzero(np.diff(b.astype(int))) + 1
        return list(zip(np.r_[0, e], np.r_[e, b.size]))
    for a, b in _runs(moving):                     # fill short pauses inside a motion
        if not moving[a] and a > 0 and b < moving.size and (b - a) * dt < max_gap:
            moving[a:b] = True
    for a, b in _runs(moving):                     # drop short blips during rest
        if moving[a] and (b - a) * dt < min_blip:
            moving[a:b] = False
    edges = np.flatnonzero(np.diff(moving.astype(int))) + 1
    bounds = np.r_[0, edges, moving.size]
    segs = [(t[a], t[min(b, t.size - 1)], moving[a]) for a, b in zip(bounds[:-1], bounds[1:])]
    out = []
    for i in range(1, len(segs) - 1):
        (r0a, r0b, m0), (ma, mb, m1), (r1a, r1b, m2) = segs[i - 1], segs[i], segs[i + 1]
        if (not m0) and m1 and (not m2) and (r0b - r0a) >= min_rest and (r1b - r1a) >= min_rest \
                and min_motion <= (mb - ma) <= max_motion:
            out.append((r0a, ma, mb, r1b))
    return out

# ----------------------------------------------------------------------------
# blur-matched resampling of the reference
# ----------------------------------------------------------------------------
def box_average(t, y, tc, width):
    """Mean of the piecewise-linear interpolant of y over [tc-w/2, tc+w/2] (exact)."""
    F = np.r_[0.0, np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(t))]   # integral
    a, b = tc - width / 2, tc + width / 2
    if width <= 0:
        return np.interp(tc, t, y)
    return (np.interp(b, t, F) - np.interp(a, t, F)) / width

# ----------------------------------------------------------------------------
# lag estimation (lag = t_X - t_M, i.e. add lag to master time to get X time)
# ----------------------------------------------------------------------------
def coarse_lag(tM, yM, tX, yX, lag_range, fs=1000.0):
    """Normalized cross-correlation on a uniform grid; returns (lag, peak ratio)."""
    g = np.arange(tM[0], tM[-1], 1 / fs)
    a = np.interp(g, tM, yM); a = (a - a.mean()) / (a.std() + 1e-12)
    lags = np.arange(lag_range[0], lag_range[1], 1 / fs)
    c = np.empty(lags.size)
    for i, L in enumerate(lags):
        m = (g + L >= tX[0]) & (g + L <= tX[-1])
        if m.sum() < 0.5 * g.size:
            c[i] = -np.inf; continue
        b = np.interp(g[m] + L, tX, yX); b = (b - b.mean()) / (b.std() + 1e-12)
        c[i] = np.mean(a[m] * b)
    k = int(np.argmax(c))
    # ambiguity: main peak vs the best secondary peak outside the main lobe
    # (main lobe = contiguous region around k with c > c[k]/2). ratio >> 1: unique.
    lo, hi = k, k
    while lo > 0 and c[lo - 1] > 0.5 * c[k]: lo -= 1
    while hi < c.size - 1 and c[hi + 1] > 0.5 * c[k]: hi += 1
    outside = np.r_[c[:lo], c[hi + 1:]]
    outside = outside[np.isfinite(outside)]
    ratio = c[k] / max(outside.max(), 1e-6) if outside.size else np.inf
    return lags[k], ratio

def fine_lag(tM_c, yM, tX, yX, lag0, tau_exp, half_range=0.01, step=2e-5):
    """LS lag between master samples (at exposure centroids) and the blur-matched
    reference; returns (lag, standard error, residual RMS)."""
    def resid(L):
        tq = tM_c + L
        m = (tq - tau_exp / 2 > tX[0]) & (tq + tau_exp / 2 < tX[-1])
        return yM[m] - box_average(tX, yX, tq[m], tau_exp), m
    lags = lag0 + np.arange(-half_range, half_range + step, step)
    sse = np.array([np.mean(resid(L)[0] ** 2) for L in lags])
    k = int(np.clip(np.argmin(sse), 1, lags.size - 2))
    a, b, c = sse[k - 1], sse[k], sse[k + 1]
    L = lags[k] + 0.5 * step * (a - c) / (a - 2 * b + c)
    r, m = resid(L)
    s2 = np.sum(r ** 2) / max(r.size - 1, 1)
    h = 1e-4                                       # d(model)/d(lag), numerically
    J = (box_average(tX, yX, tM_c[m] + L + h, tau_exp) - box_average(tX, yX, tM_c[m] + L - h, tau_exp)) / (2 * h)
    se = np.sqrt(s2 / np.sum(J ** 2))
    return L, se, np.sqrt(s2)

def envelope(t, x, win=0.05):
    """Moving-RMS envelope (for the sEMG clench bursts), on the input grid."""
    fs = 1.0 / np.median(np.diff(t)); n = max(1, int(round(win * fs)))
    return np.sqrt(np.convolve(x ** 2, np.ones(n) / n, mode='same'))

def envelope_lag(tM, eM, tX, eX, lag_range, fs=2000.0):
    """Cross-correlation of two envelopes with parabolic peak interpolation."""
    L, _ = coarse_lag(tM, eM, tX, eX, lag_range, fs=fs)
    g = np.arange(tM[0], tM[-1], 1 / fs)
    a = np.interp(g, tM, eM); a -= a.mean()
    def cc(l):
        b = np.interp(g + l, tX, eX); b -= b.mean(); return np.dot(a, b)
    d = 1 / fs; c0, cm, cp = cc(L), cc(L - d), cc(L + d)
    return L + 0.5 * d * (cm - cp) / (cm - 2 * c0 + cp)

# ----------------------------------------------------------------------------
# clock model
# ----------------------------------------------------------------------------
def clock_model(t_win, lags, se):
    """Weighted LS fit lag(t_M) = eps * t_M + theta.  Returns eps, theta, cov."""
    t_win, lags, se = map(np.asarray, (t_win, lags, se))
    t0 = t_win.mean()                               # centre for conditioning
    A = np.c_[t_win - t0, np.ones_like(t_win)]
    W = np.diag(1 / se ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    eps, c = cov @ A.T @ W @ lags
    theta = c - eps * t0
    return eps, theta, cov

def master_to_X(t_M, eps, theta):
    return (1 + eps) * t_M + theta

def X_to_master(t_X, eps, theta):
    return (t_X - theta) / (1 + eps)

def register_take(tJ_c, RJ, tQ, RQ, tau_exp, lag_range, holdout=()):
    """Full kinematic pipeline for one take. RJ/RQ: relative (or wand) rotations.
    Returns dict with per-window lags, clock model and hold-out residuals."""
    tJs, sJ = angular_speed(tJ_c, RJ)
    tQs, sQ = angular_speed(tQ, RQ)
    win = find_sync_windows(tJs, sJ)
    if len(win) < 2:
        raise RuntimeError('need a sync sequence at the start and at the end of the take')
    use = [win[0], win[-1]]
    res = {'windows': [], 'holdout': []}
    for (r0, m0, m1, r1) in use:
        mJ = (tJ_c >= r0) & (tJ_c <= r1)
        restJ = (tJ_c >= r0) & (tJ_c < m0)
        phiJ = geodesic_angle(RJ[mJ], chordal_mean(RJ[restJ]))
        # coarse lag on the angular speed (needs no reference pose)
        mJs = (tJs >= r0) & (tJs <= r1)
        L0, ratio = coarse_lag(tJs[mJs], sJ[mJs], tQs, sQ, lag_range)
        # reference rest pose in Q: same physical rest interval, mapped with L0
        restQ = (tQ >= r0 + L0 + 0.1) & (tQ < m0 + L0 - 0.1)
        mQ = (tQ >= r0 + L0 - 0.5) & (tQ <= r1 + L0 + 0.5)
        phiQ = geodesic_angle(RQ[mQ], chordal_mean(RQ[restQ]))
        L, se, rms = fine_lag(tJ_c[mJ], phiJ, tQ[mQ], phiQ, L0, tau_exp)
        res['windows'].append(dict(t=0.5 * (m0 + m1), coarse=L0, peak_ratio=ratio, lag=L, se=se, rms=rms))
    eps, theta, cov = clock_model([w['t'] for w in res['windows']],
                                  [w['lag'] for w in res['windows']],
                                  [w['se'] for w in res['windows']])
    res.update(eps=eps, theta=theta, cov=cov)
    for (a, b) in holdout:                          # diagnostics only, never refit
        m = (tJ_c >= a) & (tJ_c <= b)
        Lp = master_to_X(0.5 * (a + b), eps, theta) - 0.5 * (a + b)
        mQ = (tQ >= a + Lp - 0.5) & (tQ <= b + Lp + 0.5)
        # reference pose = a turning point of the motion (pose farthest from the
        # window mean): velocity ~ 0 there, so the residual lag uncertainty does
        # not bias the reference; phi then stays away from its fold at 0.
        tJm, RJm = tJ_c[m], RJ[m]
        k = int(np.argmax(geodesic_angle(RJm, chordal_mean(RJm))))
        k = int(np.clip(k, 2, RJm.shape[0] - 3))
        Rref = chordal_mean(RJm[k - 2:k + 3])
        mQi = np.abs(tQ - (tJm[k] + Lp)) <= 2.5 / 40.0 + 1e-9
        RrefQ = chordal_mean(RQ[mQi])
        L, se, _ = fine_lag(tJ_c[m], geodesic_angle(RJ[m], Rref), tQ[mQ],
                            geodesic_angle(RQ[mQ], RrefQ), Lp, tau_exp, half_range=0.02)
        res['holdout'].append(dict(t=0.5 * (a + b), residual=L - Lp, se=se))
    return res

# ----------------------------------------------------------------------------
# synthetic self-test
# ----------------------------------------------------------------------------
def _expm(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K

def _selftest(seed=3):
    rng = np.random.default_rng(seed)
    T, fine = 300.0, 1e-3
    t = np.arange(0, T, fine)
    # jaw opening angle (deg) about a mostly-transverse axis with small lateral part
    ang = np.zeros_like(t); lat = np.zeros_like(t)
    def burst(a, b):                                 # irregular jaw-sync motion
        m = (t >= a) & (t < b); n = m.sum(); f = np.fft.rfftfreq(n, fine)
        X = (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) * ((f > 0.4) & (f < 2.5))
        x = np.fft.irfft(X, n); x = (x - x.min()); x = 25 * x / x.max()
        w = np.sin(np.pi * np.linspace(0, 1, n)) ** 0.5   # smooth on/off
        ang[m] = x * w
        lat[m] = 4 * np.fft.irfft(X * 1j, n) / np.abs(np.fft.irfft(X, n)).max() * w
    burst(5, 12); burst(288, 295)
    chew = (t > 20) & (t < 280)                       # quasi-periodic chewing in between
    ang[chew] = 8 * (1 - np.cos(2 * np.pi * 1.3 * t[chew])) + 0.5 * rng.normal(size=chew.sum()).cumsum() * 0.001
    def Rtrue_at(tq):                                # exact (interpolated) truth at any time
        a = np.radians(np.interp(tq, t, ang)); l = np.radians(np.interp(tq, t, lat))
        ax = np.c_[np.ones_like(tq), 0.15 * np.sin(0.3 * tq), 0.1 * np.ones_like(tq)]
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        return np.stack([_expm(ai * axi + li * np.array([0, 0, 1.0])) for ai, axi, li in zip(a, ax, l)])
    B_C, B_M = _expm(rng.normal(size=3)), _expm(rng.normal(size=3))   # tag<->sphere offsets
    # JTS-MM: 40 Hz, 12 ms exposure, stamps at exposure start (master clock)
    tau_exp, fJ = 0.012, 40.0
    tJ = np.arange(0.013, T - 0.1, 1 / fJ)
    sub = np.linspace(0, tau_exp, 13)
    Rsub = Rtrue_at((tJ[:, None] + sub[None, :]).ravel()).reshape(tJ.size, sub.size, 3, 3)
    noiseJ = np.radians(0.1)
    RJ = np.stack([project_SO3(Rs.mean(0)) @ _expm(rng.normal(size=3) * noiseJ / np.sqrt(3)) for Rs in Rsub])
    # Qualisys: 200 Hz in its own clock: t_Q = (1+eps) t_M + theta (short exposure, stamped at centre)
    eps_true, theta_true = 40e-6, 3.2171
    tQ_clock = np.arange(theta_true + 0.002, master_to_X(T - 0.1, eps_true, theta_true), 1 / 200.0)
    tQ_master = X_to_master(tQ_clock, eps_true, theta_true)
    RQ = np.stack([B_C.T @ R @ B_M @ _expm(rng.normal(size=3) * np.radians(0.02) / np.sqrt(3))
                   for R in Rtrue_at(tQ_master)])
    res = register_take(tJ + tau_exp / 2, RJ, tQ_clock, RQ, tau_exp,
                        lag_range=(theta_true - 1.0, theta_true + 1.0),
                        holdout=[(100, 110), (200, 210)])
    print("per-window lags (start, end):")
    for w in res['windows']:
        true = master_to_X(w['t'], eps_true, theta_true) - w['t']
        print(f"  t={w['t']:7.2f} s  coarse {w['coarse']*1e3:9.2f} ms  fine {w['lag']*1e3:9.3f} ms "
              f"(true {true*1e3:9.3f})  se {w['se']*1e6:6.1f} us  peak ratio {w['peak_ratio']:.2f}")
    print(f"eps   = {res['eps']*1e6:8.3f} ppm   (true {eps_true*1e6:.3f})")
    print(f"theta = {res['theta']*1e3:8.3f} ms    (true {theta_true*1e3:.3f})")
    for h in res['holdout']:
        print(f"hold-out t={h['t']:.0f} s: residual {h['residual']*1e6:7.1f} us (se {h['se']*1e6:.1f} us)  [diagnostic only]")
    tt = np.linspace(0, T, 7)
    err = (master_to_X(tt, res['eps'], res['theta']) - master_to_X(tt, eps_true, theta_true)) * 1e6
    print("mapping error over the take (us):", np.round(err, 1))
    # --- sEMG: clench bursts, JTS-MM 4 kHz vs Delsys ~2148 Hz on its own clock
    eps_D, theta_D, lag_on = -25e-6, 0.8123, 0.0
    def emg(tq):
        env = sum(np.exp(-0.5 * ((tq - c) / 0.25) ** 4) for c in (14.0, 16.5, 19.0))  # 3 clenches
        return env * rng.normal(size=tq.size) * 100 + rng.normal(size=tq.size) * 3
    tE = np.arange(12.0, 21.0, 1 / 4000.0)
    tD_clock = master_to_X(np.arange(12.0, 21.0, 1 / 2148.1481), eps_D, theta_D)
    # same muscle, adjacent site: independent noise realisations, common envelope
    LD = envelope_lag(tE, envelope(tE, emg(tE)), tD_clock,
                      envelope(tD_clock, emg(X_to_master(tD_clock, eps_D, theta_D))),
                      lag_range=(theta_D - 0.5, theta_D + 0.5))
    true = master_to_X(16.5, eps_D, theta_D) - 16.5
    print(f"sEMG clench lag: {LD*1e3:.3f} ms (true {true*1e3:.3f} ms)")

if __name__ == '__main__':
    _selftest()
