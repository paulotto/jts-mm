# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Offline mains-interference removal for recorded EMG (pure NumPy, no SciPy).

Why not just notch. A biquad notch bank removes a *fixed band* around each
harmonic: the viewer's default is 5 Hz wide × 8 harmonics = 40 Hz gouged out of
the 20-450 Hz EMG band, and it still misses energy when the grid sits at
50.016 Hz rather than 50.000. This module instead fits and subtracts the
interference itself:

    for each short block, least-squares fit sin/cos at f0, 2*f0, ... k*f0
    and subtract the fit

Only the *coherent* sinusoid leaves; broadband EMG at those same frequencies
stays. Refitting per block tracks slow drift in grid amplitude and phase.

Measured on s01_f_30 take1 (467 s, 6 muscles, ~90% of in-band power at the line
frequency), resting RMS against a 20-450 Hz bandpass alone:

    masseter_r    403 uV -> 11 uV       temporalis_l  126 uV -> 7 uV
    digastric_l   132 uV -> 36 uV       digastric_r   434 uV -> 26 uV
    masseter_l    139 uV -> 5 uV        temporalis_r 3232 uV -> 168 uV

chewing-vs-rest separability from 0.99-1.06 (indistinguishable) to 3.3-7.3, at
a cost of ~2% of broadband EMG energy (measured by injecting a known signal).

It does not rescue everything: temporalis_r carried 3.2 mV of interference on a
61 kOhm contact and is 14x better but still unusable. Fix the coupling; this is
salvage, not a substitute. See docs/software/s09_mains_removal.md.

The live viewer uses a *causal* variant of the same idea; see
``jtsmm.viewer.processing.HarmonicCanceller``.
"""

from __future__ import annotations

import numpy as np

from .onset import odd_window

__all__ = [
    "estimate_line_hz",
    "fit_tone_amplitude",
    "harmonic_numbers",
    "measure_line_hz",
    "uniform_grid",
    "remove_mains",
    "mains_share",
    "DEFAULT_HARMONICS",
    "DEFAULT_BLOCK_S",
]

DEFAULT_HARMONICS = 9          # 50..450 Hz at a 50 Hz line
DEFAULT_BLOCK_S = 1.0          # refit window; long enough to resolve, short enough to track
EMG_BAND = (20.0, 450.0)
_SHARE_HALF_HZ = 1.5           # attribution half-width, wide enough for grid wander
# A harmonic just past the band edge still leaks into it; see harmonic_numbers().
HARMONIC_MARGIN_HZ = 2.0


def estimate_line_hz(
    x: np.ndarray,
    fs_hz: float,
    *,
    candidates: tuple[float, ...] = (50.0, 60.0),
    max_seconds: float = 120.0,
) -> float:
    """Grid frequency: ~0.002 Hz from 5 s, ~0.001 Hz from 20 s of data.

    Picks 50 vs 60 by total power, then refines the Hann-windowed periodogram
    peak by fitting a parabola to the *log* power of the three top bins. The
    refinement matters: the recorded grid sat at 50.016 Hz and 49.986 Hz in two
    sessions an hour apart, and a fit or notch at exactly 50.000 leaves a
    visible residual.

    Log, not linear power: the Hann main lobe is close to a Gaussian, whose log
    is exactly a parabola, so the vertex lands on the true frequency. A parabola
    through the linear power (what this did until 2026-09) is biased towards
    the bin centre by up to about half the sub-bin offset -- 0.014 Hz at 5 s,
    which leaves ~10x more residual mains after a fit than the 0.002 Hz here.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 16 or fs_hz <= 0:
        return float(candidates[0])
    n = min(x.size, int(max_seconds * fs_hz))
    seg = x[:n]
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs_hz)
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 0.0

    best, best_p = float(candidates[0]), -1.0
    for f0 in candidates:
        m = (freqs >= f0 - 2.0) & (freqs <= f0 + 2.0)
        if not np.any(m):
            continue
        p = float(spec[m].sum())
        if p > best_p:
            best_p, best = p, f0
    m = (freqs >= best - 2.0) & (freqs <= best + 2.0)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return best
    i = int(idx[np.argmax(spec[idx])])
    if 0 < i < spec.size - 1 and min(spec[i - 1], spec[i], spec[i + 1]) > 0.0:
        a, b, c = np.log(spec[i - 1 : i + 2])
        den = a - 2.0 * b + c
        delta = 0.5 * (a - c) / den if den < 0.0 else 0.0
        return float(freqs[i] + float(np.clip(delta, -0.5, 0.5)) * df)
    return float(freqs[i])


def measure_line_hz(
    x: np.ndarray,
    fs_hz: float,
    nominal_hz: float,
    *,
    t_us: np.ndarray | None = None,
    max_dev_hz: float = 0.3,
    min_snr: float = 5.0,
) -> float | None:
    """The grid frequency in this capture, or None when it cannot be trusted.

    For short checks that must fit mains at the *real* frequency: a fit at the
    nominal 50.000 Hz while the grid ran at 50.016 Hz left 72 uV of a 1 mV hum
    behind (3 uV at the measured frequency). None when the tone is not clearly
    above its neighbourhood (a clean room: then any "peak" is noise and the
    nominal frequency is the better guess) or lands more than ``max_dev_hz``
    from nominal (grids stay within +-0.2 Hz outside emergencies).
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < 64 or fs_hz <= 0 or nominal_hz <= 0 or nominal_hz >= fs_hz * 0.5:
        return None
    if t_us is not None and np.asarray(t_us).ravel().size == v.size:
        _, grid = uniform_grid(np.asarray(t_us), v, fs_hz)
        grid = np.asarray(grid, dtype=np.float64).ravel()
    else:
        grid = v
    if grid.size < 64:
        return None
    f = estimate_line_hz(grid, fs_hz, candidates=(float(nominal_hz),))
    if not np.isfinite(f) or abs(f - nominal_hz) > max_dev_hz:
        return None
    # Tone vs its spectral neighbourhood, both by least squares at the real
    # sample times. Neighbours sit 3..8 resolution bins away on either side.
    dur = grid.size / fs_hz
    amp = fit_tone_amplitude(v, fs_hz, f, t_us=t_us)
    floor = [
        fit_tone_amplitude(v, fs_hz, f + sgn * k / dur, t_us=t_us)
        for k in range(3, 9)
        for sgn in (-1.0, 1.0)
    ]
    noise = float(np.median(floor)) if floor else 0.0
    if amp <= 0.0 or amp < min_snr * noise:
        return None
    return float(f)


def uniform_grid(
    t_us: np.ndarray, x: np.ndarray, fs_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``x`` onto an evenly spaced time base derived from ``t_us``.

    This is not optional housekeeping -- it decides whether mains removal works
    at all. A recorded stream with dropped samples is packed contiguously, so
    row index stops being proportional to time, and every filter here (and every
    filter anywhere) assumes it is. On s01_f_30 take1, 2.92% of samples were
    missing and the compressed time base put the apparent line frequency at
    50.224 Hz instead of the true 50.020 -- a 0.2 Hz error, past the point where
    any notch or tracker helps. Worse, the gaps are not evenly spread, so the
    apparent frequency *wanders* and no single correction fixes it.

    @return ``(t_uniform_us, x_uniform)``.
    """
    t = np.asarray(t_us, dtype=np.float64).ravel()
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if t.size < 2 or t.size != a.shape[0] or fs_hz <= 0:
        return np.asarray(t_us), np.asarray(x)
    step = 1e6 / fs_hz
    tu = np.arange(t[0], t[-1], step, dtype=np.float64)
    xu = np.column_stack([np.interp(tu, t, a[:, c]) for c in range(a.shape[1])])
    return tu, (xu[:, 0] if np.asarray(x).ndim == 1 else xu)


def harmonic_numbers(
    f0: float,
    fs_hz: float,
    *,
    n_harmonics: int = DEFAULT_HARMONICS,
    max_hz: float = EMG_BAND[1],
) -> list[int]:
    """Harmonic numbers ``k`` whose tone ``k*f0`` matters for the ``max_hz`` band.

    ``HARMONIC_MARGIN_HZ`` past the band edge still counts: at a 50.02 Hz grid
    the 9th harmonic sits at 450.18 Hz, and its leakage lands in the 450 Hz bin
    that the band RMS includes. A strict ``k*f0 < max_hz`` dropped it exactly
    when the grid ran fast. Always stays below Nyquist.
    """
    if f0 <= 0 or fs_hz <= 0:
        return []
    top = min(float(max_hz) + HARMONIC_MARGIN_HZ, fs_hz * 0.5)
    return [k for k in range(1, int(n_harmonics) + 1) if k * f0 < top]


def fit_tone_amplitude(
    x: np.ndarray,
    fs_hz: float,
    tone_hz: float,
    *,
    t_us: np.ndarray | None = None,
    nuisance_hz: tuple[float, ...] | list[float] = (),
) -> float:
    """Peak amplitude of a sinusoid at exactly ``tone_hz``, by least squares.

    Unlike an FFT bin or an integer-bin Goertzel this evaluates the exact
    frequency -- no scalloping, whatever the capture length -- and with
    ``t_us`` it fits at the real sample times, so dropped samples cost a little
    noise instead of smearing the tone. The design matrix carries an intercept
    and a linear trend (electrode DC offset is not orthogonal to a sinusoid at
    gappy timestamps; see remove_mains) and, optionally, sin/cos pairs at
    ``nuisance_hz`` fitted jointly so a strong neighbour (mains next to the
    31.25 Hz lead-off tone) cannot leak into the estimate.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < 8 or fs_hz <= 0 or tone_hz <= 0:
        return 0.0
    if t_us is not None and np.asarray(t_us).ravel().size == v.size:
        t = np.asarray(t_us, dtype=np.float64).ravel()
        t = (t - t[0]) / 1e6
    else:
        t = np.arange(v.size, dtype=np.float64) / fs_hz
    freqs = [float(tone_hz)] + [
        float(f) for f in nuisance_hz if f > 0 and abs(f - tone_hz) > 1e-6
    ]
    cols: list[np.ndarray] = []
    for f in freqs:
        w = 2.0 * np.pi * f * t
        cols.extend([np.sin(w), np.cos(w)])
    tc = t - t.mean()
    cols.extend([np.ones_like(t), tc])
    d = np.column_stack(cols)
    if d.shape[0] <= d.shape[1]:
        return 0.0
    coef, *_ = np.linalg.lstsq(d, v, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


def remove_mains(
    x: np.ndarray,
    fs_hz: float,
    *,
    t_us: np.ndarray | None = None,
    line_hz: float | None = None,
    n_harmonics: int = DEFAULT_HARMONICS,
    block_s: float = DEFAULT_BLOCK_S,
    max_hz: float = EMG_BAND[1],
    robust: bool = False,
    robust_env_s: float = 0.05,
    amp_order: int | str = 0,
) -> tuple[np.ndarray, float]:
    """Subtract the line frequency and its harmonics from ``(N,)`` or ``(N, C)``.

    @param robust   Down-weight EMG bursts in each block's fit. An ordinary
                    least-squares fit also "explains" the burst's own broadband
                    energy at the harmonic frequencies; the resulting amplitude
                    error is subtracted as a steady tone over the whole block,
                    so quiet stretches next to a burst gain a residual hum. On
                    synthetic EMG (3 uV baseline, 60 uV bursts in half of each
                    1-s block, 9 harmonics) the baseline next to bursts rose to
                    ~5 uV RMS and onset detection moved early by 30-130 ms. With
                    ``robust`` the fit is repeated with weights 1 / env^2, env
                    being the first-pass residual's local RMS (floored at its
                    10th percentile in the block), which takes the bursts out of the estimate.
    @param amp_order  Let each harmonic's amplitude and phase vary within a block as a
                    polynomial of this order (sin/cos times Legendre P0..Pn of block
                    time). 0 = constant per block. Hum whose amplitude is modulated by
                    movement (cable/electrode motion) otherwise leaves a residual locked
                    to the movement: at 1 Hz / 30 % modulation, 2.5 uV with order 0 vs
                    1.6 uV with order 2 (1-s blocks, synthetic) -- but each order also
                    absorbs a little EMG at the harmonics when the hum is steady.
                    ``"auto"`` picks the order (0..3) per block and channel by BIC, so the
                    model is only as flexible as the data demand.

    @param t_us     Per-sample timestamps (``emg/t_master_us``). PASS THESE for
                    recorded data: the fit happens on a uniform grid and the
                    result is mapped back onto the original rows, so dropped
                    samples cannot smear the line frequency. See uniform_grid().
    @param line_hz  None estimates it from the data (recommended).
    @return ``(cleaned, line_hz_used)``; ``cleaned`` matches the input shape.
    """
    a = np.asarray(x, dtype=np.float64)
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
    if a.shape[0] == 0 or fs_hz <= 0 or n_harmonics <= 0:
        return (a[:, 0].copy() if one_d else a.copy()), float(line_hz or 0.0)

    # Sample times in seconds. Least squares does NOT need uniform sampling, so
    # with timestamps we fit and subtract at the REAL sample times and never put
    # an interpolator near the data. Resampling to a uniform grid, fitting there
    # and mapping back (which this did first) subtracts a sinusoid that
    # interpolation has shifted off the mains actually in the samples: on the
    # s01_f_30 MVC that turned a correct 247 -> 147 uV into 247 -> 274, i.e. it
    # ADDED energy. The grid survives only for the frequency estimate, which is
    # an FFT and does need uniform spacing.
    has_t = t_us is not None and np.asarray(t_us).ravel().size == a.shape[0]
    if has_t:
        t_rel = np.asarray(t_us, dtype=np.float64).ravel()
        t_rel = (t_rel - t_rel[0]) / 1e6
    else:
        t_rel = np.arange(a.shape[0], dtype=np.float64) / fs_hz

    if line_hz is None:
        # Pick the loudest column to estimate from: the interference is common,
        # so the strongest channel gives the cleanest peak.
        loudest = int(np.argmax(a.std(axis=0)))
        src = a[:, loudest]
        if has_t:
            _, src = uniform_grid(np.asarray(t_us), src, fs_hz)
        line_hz = estimate_line_hz(np.asarray(src).ravel(), fs_hz)
    f0 = float(line_hz)

    ks = harmonic_numbers(f0, fs_hz, n_harmonics=n_harmonics, max_hz=max_hz)
    if not ks:
        return (a[:, 0].copy() if one_d else a.copy()), f0

    # Abutting blocks, each fitted and subtracted in place. The design matrix
    # carries a constant and a linear-trend column alongside the sinusoids, and
    # only the sinusoids are subtracted. Without them this was badly wrong on
    # real data and looked like physics: an Ag/AgCl pair sits at 10-25 mV of DC
    # offset, and on a uniform grid whole cycles of a sinusoid are orthogonal to
    # a constant -- but at the real timestamps, with dropped samples punching
    # holes in the cycles, they are not. About 2% of a 25 mV offset leaked into
    # the fitted 50 Hz amplitude, i.e. ~500 uV of tone SUBTRACTED that was never
    # there. On s01_f_30 take1 that raised in-band RMS in 10 of 11 rest blocks
    # (132 -> 288 uV) and read as unremovable modulated interference. With the
    # intercept: 29 uV, in-band RMS down in every block, same EMG cost.
    #
    # Half-overlapping Hann overlap-add of the model was also tried and is
    # worse: neighbouring blocks fit slightly different phases, so averaging
    # attenuates the thing being subtracted (6.5 -> 158 uV vs a 6.55 uV truth).
    #
    # A remainder shorter than one block is merged into the block before it.
    # Fitted on its own, a few ms of data make the sinusoids nearly collinear
    # with the intercept/trend columns; the fit then "explains" the electrode
    # offset with huge sinusoids, subtracts them, and leaves a step that the
    # band-pass turns into a transient (445 uV RMS in the last second of s09's
    # mvc_closing digastric, vs 8 uV elsewhere; 10 of 576 pilot windows hit).
    y = a.copy()
    auto = amp_order == "auto"
    max_order = AUTO_MAX_ORDER if auto else max(int(amp_order), 0)
    n_mod = max_order + 1
    n_sin = 2 * len(ks) * n_mod
    block = max(int(round(block_s * fs_hz)), 2 * n_sin + 4)
    starts = list(range(0, y.shape[0], block))
    if len(starts) > 1 and y.shape[0] - starts[-1] < block:
        starts.pop()
    win = odd_window(robust_env_s, fs_hz)       # odd: the centred burst envelope is symmetric
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else y.shape[0]
        n = e - s
        if n < n_sin + 4:            # too short to constrain the fit; leave it
            continue
        t = t_rel[s:e] - t_rel[s]
        d, order_of = _design(t, ks, f0, n_mod)
        yb = y[s:e]
        if not auto:
            coef, *_ = np.linalg.lstsq(d, yb, rcond=None)
            if robust:
                coef = _reweighted_fit(d, yb, coef, win)
            y[s:e] -= d[:, :n_sin] @ coef[:n_sin]
            continue
        # Order chosen per channel by BIC on the (robustly) weighted fit; the weights
        # come from the order-0 robust fit so every order is judged on the same samples.
        # Columns are nested (nuisance, then order 0, 1, ...), so one Cholesky factor of the
        # small weighted Gram matrix G = L L^T gives the residual of every order:
        # RSS_q = |y_w|^2 - |z[:q]|^2 with L z = D_w^T y_w (z is Q^T y_w of the QR).
        perm = np.argsort(order_of, kind="stable")          # -1 (nuisance) first, then 0..M
        dp, op = d[:, perm], order_of[perm]
        q_of = [int(np.sum(op <= m)) for m in range(n_mod)]
        base = np.flatnonzero(op <= 0)
        c0, *_ = np.linalg.lstsq(dp[:, base], yb, rcond=None)
        sw = _robust_weights(dp[:, base], yb, c0, win) if robust else np.ones_like(yb)
        logn = np.log(n)
        for c in range(yb.shape[1]):
            dw = dp * sw[:, c, None]
            yw = yb[:, c] * sw[:, c]
            G, b = dw.T @ dw, dw.T @ yw
            try:
                L = np.linalg.cholesky(G)
            except np.linalg.LinAlgError:                 # rank-deficient block: tiny ridge
                L = np.linalg.cholesky(G + 1e-10 * np.trace(G) / len(G) * np.eye(len(G)))
            z = np.linalg.solve(L, b)
            tot = float(yw @ yw)
            bics = [n * np.log(max(tot - float(z[:q] @ z[:q]), 1e-300) / n) + q * logn
                    for q in q_of]
            q = q_of[int(np.argmin(bics))]
            sol = np.linalg.solve(L[:q, :q].T, z[:q])
            hum = op[:q] >= 0                         # sinusoid columns only, not DC/trend
            y[s:e, c] -= dp[:, :q][:, hum] @ sol[hum]
    return (y[:, 0] if one_d else y), f0



AUTO_MAX_ORDER = 3


def _design(t: np.ndarray, ks: list[int], f0: float, n_mod: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """[sin/cos(k f0 t) x Legendre P_m(tau) for k, m] + [1, t]; and each column's order m
    (-1 for the nuisance columns)."""
    n = t.size
    n_sin = 2 * len(ks) * n_mod
    d = np.empty((n, n_sin + 2))
    order_of = np.empty(n_sin + 2, dtype=int)
    tau = 2.0 * (t - t[0]) / max(t[-1] - t[0], 1e-12) - 1.0      # block time in [-1, 1]
    mods = [np.polynomial.legendre.Legendre.basis(m)(tau) for m in range(n_mod)]
    col = 0
    for k in ks:
        w = 2.0 * np.pi * k * f0 * t
        sw, cw = np.sin(w), np.cos(w)
        for m, mod in enumerate(mods):
            d[:, col], d[:, col + 1] = sw * mod, cw * mod
            order_of[col] = order_of[col + 1] = m
            col += 2
    d[:, n_sin] = 1.0
    d[:, n_sin + 1] = t - t.mean()
    order_of[n_sin:] = -1
    return d, order_of


def _robust_weights(d: np.ndarray, yb: np.ndarray, coef: np.ndarray, win: int,
                    n_iter: int = 2, floor_q: float = 10.0) -> np.ndarray:
    """sqrt-weights (n, C) of the reweighted fit (see ``_reweighted_fit``)."""
    out = coef.copy()
    win = win if win % 2 else win + 1              # odd: mode="same" is centred only then
    kern = np.ones(win) / win
    sw = np.ones_like(yb)
    for _ in range(n_iter):
        r = yb - d @ out
        for c in range(yb.shape[1]):
            env = np.sqrt(np.maximum(np.convolve(r[:, c] ** 2, kern, mode="same"), 0.0))
            floor = float(np.percentile(env, floor_q))
            if not np.isfinite(floor) or floor <= 0.0:
                continue
            sw[:, c] = 1.0 / np.maximum(env, floor)
            sol, *_ = np.linalg.lstsq(d * sw[:, c, None], yb[:, c] * sw[:, c], rcond=None)
            out[:, c] = sol
    return sw


def _reweighted_fit(d: np.ndarray, yb: np.ndarray, coef: np.ndarray, win: int,
                    n_iter: int = 2, floor_q: float = 10.0) -> np.ndarray:
    """Reweighted least squares per column (see ``remove_mains(robust=True)``).

    Weights are 1 / env^2 with env the local RMS of the current residual,
    floored at its ``floor_q``-th percentile (a low quantile, so the floor stays
    at the quiet level even when bursts fill most of the block).
    """
    out = coef.copy()
    win = win if win % 2 else win + 1              # odd: mode="same" is centred only then
    kern = np.ones(win) / win
    for _ in range(n_iter):
        r = yb - d @ out
        for c in range(yb.shape[1]):
            env = np.sqrt(np.maximum(np.convolve(r[:, c] ** 2, kern, mode="same"), 0.0))
            floor = float(np.percentile(env, floor_q))
            if not np.isfinite(floor) or floor <= 0.0:
                continue
            sw = 1.0 / np.maximum(env, floor)
            sol, *_ = np.linalg.lstsq(d * sw[:, None], yb[:, c] * sw, rcond=None)
            out[:, c] = sol
    return out


def mains_share(
    x: np.ndarray, fs_hz: float, line_hz: float, *, n_harmonics: int = DEFAULT_HARMONICS
) -> float:
    """Fraction of in-band (20-450 Hz) power sitting at the line harmonics.

    ~0.9 means the channel is interference with EMG buried in it; <0.1 is clean.
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size < 16 or fs_hz <= 0:
        return 0.0
    v = v - v.mean()
    spec = np.abs(np.fft.rfft(v * np.hanning(v.size))) ** 2
    freqs = np.fft.rfftfreq(v.size, 1.0 / fs_hz)
    in_band = (freqs >= EMG_BAND[0]) & (freqs <= EMG_BAND[1])
    total = float(spec[in_band].sum())
    if total <= 0:
        return 0.0
    hit = np.zeros_like(in_band)
    for k in harmonic_numbers(line_hz, fs_hz, n_harmonics=n_harmonics):
        fk = k * line_hz
        hit |= (freqs >= fk - _SHARE_HALF_HZ) & (freqs <= fk + _SHARE_HALF_HZ)
    return float(spec[in_band & hit].sum() / total)
