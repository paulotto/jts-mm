#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Companion to cv05_reference_validation_protocol.md §3: Monte Carlo of offline sync estimators.
# Run: python3 cv05_sync_montecarlo.py  (numpy only; ~5-10 min)
"""Monte Carlo: time-offset estimation between JTS-MM (40 Hz, 12 ms exposure)
and Qualisys (200 Hz, ~0.5 ms exposure) for three sync events:
  (1) paper-cover reveal, (2) motion cross-correlation of a frame-invariant
  rotation angle, (3) impact ("hinge") event fitted per system."""
import numpy as np
rng = np.random.default_rng(1)
FJ, TEXP, FQ, TQEXP = 40.0, 12e-3, 200.0, 0.5e-3
SIG_J, SIG_Q = 0.10, 0.02          # deg, angle noise (JTS precision ~0.13 deg)
SIGP_J, SIGP_Q = 0.20, 0.05        # mm, position noise
FINE = 1e-4                         # 0.1 ms simulation grid

def bandlimited(T, fmin=0.3, fmax=2.5, amp=30.0):
    t = np.arange(0, T, FINE); f = np.fft.rfftfreq(t.size, FINE)
    X = (rng.normal(size=f.size) + 1j*rng.normal(size=f.size)) * ((f > fmin) & (f < fmax))
    x = np.fft.irfft(X, t.size); return t, amp * x / x.std()

def sample(t, x, f, texp, phase, sig, blur=True):
    tk = np.arange(phase, t[-1] - texp - 0.05, 1/f)
    if not blur: return tk, np.interp(tk, t, x) + rng.normal(0, sig, tk.size)
    n = max(1, int(round(texp/FINE))); c = np.convolve(x, np.ones(n)/n, 'valid')
    return tk, np.interp(tk, t[:c.size], c) + rng.normal(0, sig, tk.size)  # tk = exposure start

def est_offset_motion(tJ, yJ, tQ, yQ, guess, model_blur=True):
    # JTS sample k is the mean over [tJ, tJ+TEXP]; emulate on Qualisys by box filter
    if model_blur:
        nb = int(round(TEXP*FQ)) + 1
        yQf = np.convolve(yQ, np.ones(nb)/nb, 'valid'); tQf = tQ[:yQf.size] + (nb-1)/(2*FQ)
    else:
        yQf, tQf = yQ, tQ
    def sse(tau):
        q = np.interp(tJ + TEXP/2 + tau, tQf, yQf); m = (tJ+TEXP/2+tau > tQf[0]) & (tJ+TEXP/2+tau < tQf[-1])
        return np.mean((yJ[m]-q[m])**2)
    taus = guess + np.arange(-0.05, 0.05, 5e-4); s = np.array([sse(x) for x in taus])
    i = np.clip(s.argmin(), 1, len(s)-2)
    for _ in range(3):  # local refinement
        taus = taus[i] + np.linspace(-taus[1]+taus[0], taus[1]-taus[0], 21); s = np.array([sse(x) for x in taus]); i = np.clip(s.argmin(),1,19)
    a, b, c = s[i-1], s[i], s[i+1]; d = taus[1]-taus[0]
    return taus[i] + 0.5*d*(a-c)/(a-2*b+c)

def run_motion(N=300, T=15.0, model_blur=True):
    err = []
    for _ in range(N):
        t, x = bandlimited(T); theta = rng.uniform(0, 1/FQ)   # true offset (Q clock - J clock), sub-frame
        tJ, yJ = sample(t, x, FJ, TEXP, rng.uniform(0, 1/FJ), SIG_J, True)
        tQ, yQ = sample(t, x, FQ, TQEXP, rng.uniform(0, 1/FQ), SIG_Q, True)
        # express Qualisys times on its own clock: tQ_clock = tQ + theta ; estimate theta
        th = est_offset_motion(tJ, yJ, tQ + TQEXP/2 + theta, yQ, 0.0, model_blur)
        err.append(th - theta)
    return np.array(err)*1e3

def run_paper(N=20000, reps=1):
    out = []
    for _ in range(N):
        e = []
        for _ in range(reps):
            t0 = rng.uniform(0, 1); dr = rng.uniform(0.01, 0.04)   # reveal takes 10-40 ms
            vis_J = t0 + dr                  # tag fully uncovered (ArUco/AprilTag needs whole tag)
            vis_Q = t0 + 0.6*dr              # 3 of 4 spheres visible -> 6DOF body solved earlier
            pJ, pQ = rng.uniform(0, 1/FJ), rng.uniform(0, 1/FQ)
            kJ = np.ceil((vis_J - pJ)*FJ); kQ = np.ceil((vis_Q - pQ)*FQ)   # first exposure starting after visibility
            detJ = pJ + kJ/FJ + TEXP/2; detQ = pQ + kQ/FQ          # compare at mid exposure
            e.append(detJ - detQ)            # estimated offset; truth is 0
        out.append(np.mean(e))
    return np.array(out)*1e3

def hinge_fit(tk, y, texp, f_fine=FINE):
    # model: y = v*(ti - t) for t<ti, 0 after; JTS sample = mean over exposure
    best = None
    for ti in np.arange(tk[0]+0.05, tk[-1]-0.05, 2e-4):
        if texp > 1e-3:
            s = np.linspace(0, texp, 25)[None, :] + tk[:, None]
            g = np.clip(ti - s, 0, None).mean(1)
        else:
            g = np.clip(ti - tk, 0, None)
        v = (g @ y) / (g @ g); r = np.sum((y - v*g)**2)
        if best is None or r < best[0]: best = (r, ti)
    return best[1]

def run_impact(N=300, v=300.0):
    eJ, eQ = [], []
    for _ in range(N):
        ti = 0.5 + rng.uniform(0, 0.05)
        tk = np.arange(rng.uniform(0, 1/FJ), 1.0, 1/FJ); tq = np.arange(rng.uniform(0, 1/FQ), 1.0, 1/FQ)
        m = lambda tt, te: np.clip(ti - (np.linspace(0, te, 25)[None, :] + tt[:, None]), 0, None).mean(1)*v
        yJ = m(tk, TEXP) + rng.normal(0, SIGP_J, tk.size); yQ = v*np.clip(ti - tq, 0, None) + rng.normal(0, SIGP_Q, tq.size)
        sl = (tk > ti-0.25) & (tk < ti+0.25); sq = (tq > ti-0.25) & (tq < ti+0.25)
        eJ.append(hinge_fit(tk[sl], yJ[sl], TEXP) - ti); eQ.append(hinge_fit(tq[sq], yQ[sq], 0) - ti)
    return np.array(eJ)*1e3, np.array(eQ)*1e3

def s(name, e): print(f"{name:52s} bias {np.mean(e):+7.3f} ms  SD {np.std(e):6.3f} ms  95% |e| {np.percentile(np.abs(e),95):6.3f} ms")
s("paper reveal, 1 event", run_paper(reps=1))
s("paper reveal, mean of 5 events", run_paper(N=4000, reps=5))
s("motion xcorr 15 s, blur modelled", run_motion(150, 15, True))
s("motion xcorr 15 s, blur NOT modelled", run_motion(150, 15, False))
s("motion xcorr 5 s, blur modelled", run_motion(150, 5, True))
eJ, eQ = run_impact(150); s("impact, JTS-MM (40 Hz, 12 ms)", eJ); s("impact, Qualisys (200 Hz)", eQ); s("impact, offset J-Q (one event)", eJ-eQ)
