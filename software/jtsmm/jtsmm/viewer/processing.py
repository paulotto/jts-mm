# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Streaming, causal EMG display filters (pure NumPy, no SciPy dependency).

The live/replay viewer needs a *causal* filter (no ``filtfilt`` edge cones) that
can be fed block-by-block while keeping state across blocks. We implement the
standard RBJ biquad sections (high-pass, low-pass, notch) and cascade them, so
the whole thing is a few dozen lines of arithmetic with zero heavy deps — the
production venv on the Pi does not carry SciPy.

Coefficients follow the Audio-EQ Cookbook (Robert Bristow-Johnson). Each section
runs Direct-Form-II transposed with per-channel state. The recursions are linear
and time-invariant, so a block is evaluated in chunks of ``_CHUNK`` samples with
precomputed state-space matrices (``y = T x + O s``, ``s' = A^L s + F x``) instead
of a Python loop per sample -- identical results to ~1e-12, about 10x less CPU
(the per-sample loops cost ~62 ms of CPU per second of 2 kHz data on a PC,
several times that on the Pi, inside the recording process).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "Biquad",
    "CausalEmgFilter",
    "HarmonicCanceller",
    "line_notch_freqs",
]

# Surface-EMG band used to decide which line harmonics are worth notching.
# 50 Hz × 8 = 400 Hz, the last harmonic below the 450 Hz LPF.
EMG_BAND_HZ = 450.0
DEFAULT_NOTCH_HARMONICS = 8
_CHUNK = 64


class _StateSpace2:
    """Chunked evaluation of a 2-state LTI recursion, many channels at once.

    ``s[n+1] = A s[n] + B x[n]``, ``y[n] = C s[n] + D x[n]`` with the state for
    all M channels held as a (2, M) array. Exact linear algebra on the same
    recursion as the per-sample loop; only the summation order differs.
    """

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: float, L: int = _CHUNK):
        A = np.asarray(A, dtype=np.float64).reshape(2, 2)
        B = np.asarray(B, dtype=np.float64).reshape(2)
        C = np.asarray(C, dtype=np.float64).reshape(2)
        self.L = int(L)
        pw = [np.eye(2)]
        for _ in range(self.L):
            pw.append(A @ pw[-1])
        self._apow = np.stack(pw)                         # A^0 .. A^L
        h = np.array([C @ pw[i] @ B for i in range(self.L)])   # impulse response (n >= 1)
        T = np.zeros((self.L, self.L))
        for i in range(self.L):
            T[i, i] = D
            if i:
                T[i, :i] = h[i - 1 :: -1][:i]
        self._T = T
        self._O = np.stack([C @ pw[i] for i in range(self.L)])        # (L, 2)
        self._F = np.stack([pw[self.L - 1 - j] @ B for j in range(self.L)], axis=1)  # (2, L)

    def run(self, x: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``x`` (N, M), ``state`` (2, M) -> (y (N, M), new state)."""
        n = x.shape[0]
        y = np.empty_like(x, dtype=np.float64)
        s = state
        L = self.L
        for a in range(0, n, L):
            r = min(L, n - a)
            xc = x[a : a + r]
            y[a : a + r] = self._T[:r, :r] @ xc + self._O[:r] @ s
            s = self._apow[r] @ s + self._F[:, L - r :] @ xc
        return y, s


@dataclass
class Biquad:
    """A normalized biquad section with per-channel DF-II-T state."""

    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    n_channels: int

    def __post_init__(self) -> None:
        self._z1 = np.zeros(self.n_channels, dtype=np.float64)
        self._z2 = np.zeros(self.n_channels, dtype=np.float64)
        # DF-II-T as state space: s = (z1, z2), y = b0 x + z1.
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        self._ss = _StateSpace2(
            [[-a1, 1.0], [-a2, 0.0]], [b1 - a1 * b0, b2 - a2 * b0], [1.0, 0.0], b0
        )

    def reset(self) -> None:
        self._z1[:] = 0.0
        self._z2[:] = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a ``(N, C)`` block; returns ``(N, C)`` float64, state carried over."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape[0] == 0:
            return np.empty_like(x)
        y, s = self._ss.run(x, np.vstack([self._z1, self._z2]))
        self._z1, self._z2 = s[0].copy(), s[1].copy()
        return y


def _lowpass(fs: float, fc: float, q: float, n: int) -> Biquad:
    w0 = 2.0 * math.pi * fc / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha
    return Biquad((1 - cw) / 2 / a0, (1 - cw) / a0, (1 - cw) / 2 / a0,
                  (-2 * cw) / a0, (1 - alpha) / a0, n)


def _highpass(fs: float, fc: float, q: float, n: int) -> Biquad:
    w0 = 2.0 * math.pi * fc / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha
    return Biquad((1 + cw) / 2 / a0, -(1 + cw) / a0, (1 + cw) / 2 / a0,
                  (-2 * cw) / a0, (1 - alpha) / a0, n)


def _dcblock(fs: float, fc: float, n: int) -> Biquad:
    """First-order DC blocker, y = a*(y1 + x - x1), as a degenerate biquad.

    Used ahead of the harmonic canceller (see CausalEmgFilter). A second-order
    20 Hz high-pass cannot go there: it rings at ~20 Hz for ~50 ms whenever the
    mains tone jumps phase across a dropped-sample gap -- inside the EMG band,
    and not a sinusoid the canceller can model (20 uV of residual on a 300 uV
    tone with 3% dropouts, 7 uV with the canceller ahead of it). But the
    canceller cannot see raw DC either: a 25 mV electrode offset times a
    reference whose phase skips at every gap does not average to zero, and it
    leaked ~600 uV of phantom tone. A first-order block far below the line
    frequency removes the offset and, at a phase jump, produces only a slow
    settling tail that the canceller's own low-pass rejects.
    """
    a = math.exp(-2.0 * math.pi * fc / fs) if fs > 0 else 0.0
    return Biquad(a, -a, 0.0, -a, 0.0, n)


def _notch(fs: float, f0: float, q: float, n: int) -> Biquad:
    w0 = 2.0 * math.pi * f0 / fs
    cw, sw = math.cos(w0), math.sin(w0)
    alpha = sw / (2.0 * q)
    a0 = 1.0 + alpha
    return Biquad(1 / a0, (-2 * cw) / a0, 1 / a0, (-2 * cw) / a0, (1 - alpha) / a0, n)


def line_notch_freqs(
    fs_hz: float,
    notch_hz: float = 50.0,
    *,
    max_hz: float = EMG_BAND_HZ,
    max_harmonics: int = DEFAULT_NOTCH_HARMONICS,
) -> list[float]:
    """Integer harmonics of the line tone that fit in the EMG band below Nyquist.

    For 50 Hz this is 50, 100, 150, 200, 250, 300, 350, 400 Hz whenever ``fs``
    is high enough. Tones at or near Nyquist are omitted — an RBJ notch at
    ``fs/2`` is degenerate.
    """
    if fs_hz <= 0 or notch_hz <= 0 or max_harmonics <= 0:
        return []
    # Stay a bit below Nyquist so the biquad does not collapse to a no-op.
    cap = min(float(max_hz), fs_hz * 0.5 * 0.95)
    freqs: list[float] = []
    for h in range(1, max_harmonics + 1):
        f0 = notch_hz * h
        if f0 >= cap:
            break
        freqs.append(f0)
    return freqs


class HarmonicCanceller:
    """Causal, stateful removal of a line tone and its harmonics.

    The biquad notch bank removes a fixed *band* -- 5 Hz wide by default, times
    eight harmonics, is 40 Hz of the 20-450 Hz EMG band thrown away, and it is
    still centred on a nominal 50.000 Hz while the real grid sits at 50.016.

    This instead tracks the interference and subtracts it. Each harmonic is
    demodulated to DC against its own quadrature pair, smoothed by a one-pole
    at ``track_hz``, and the reconstruction is subtracted from the input:

        I <- I + a*(x*sin - I)      Q <- Q + a*(x*cos - Q)
        x <- x - 2*(I*sin + Q*cos)

    The removed band is ~2*``track_hz`` wide (1 Hz by default, against the
    notch's 5 Hz), so far less EMG goes with it, and because I/Q are free to
    rotate the canceller follows grid drift of up to ``track_hz`` without being
    retuned. It is feed-forward -- the estimate is driven by the input, not by
    its own output -- so it cannot ring or go unstable, and neighbouring
    harmonics are 50 Hz apart, far outside each tracker's passband.

    State persists across blocks, so successive viewer frames join seamlessly.
    Convergence takes roughly ``1/(2*pi*track_hz)`` seconds (~0.16 s default);
    until then the tone is only partly removed.
    """

    def __init__(
        self,
        fs_hz: float,
        n_channels: int,
        freqs: list[float],
        *,
        track_hz: float = 0.5,
        lock_hz: float = 0.5,
    ) -> None:
        self.fs_hz = float(fs_hz)
        self.n_channels = int(n_channels)
        self.freqs = [float(f) for f in freqs]
        self.track_hz = float(track_hz)
        # The tracker is narrow, so it only cancels if its oscillator sits ON the
        # grid: at a 0.05 Hz offset -- inside ENTSO-E's normal band -- a fixed
        # 50.000 Hz oscillator leaves 20% of the tone, worse than a plain notch.
        # So estimate the offset from the fundamental's phasor rotation and pull
        # the oscillators onto it. Clamped to +-lock_hz so it can never walk off
        # onto EMG, and frozen while the phasor is too small to trust.
        self.lock_hz = float(lock_hz)
        self._df = 0.0
        self._prev_angle: float | None = None
        # Re-lock on a fixed amount of DATA, not once per call: viewer blocks
        # vary with load, and a per-block step would make the lock rate (and so
        # the output) depend on how the stream happened to be chunked.
        self._lock_every = max(1, int(round(0.25 * self.fs_hz)))
        self._since_lock = 0
        k = len(self.freqs)
        self._theta = np.zeros(k, dtype=np.float64)
        self._dtheta = np.array(
            [2.0 * math.pi * f / self.fs_hz for f in self.freqs], dtype=np.float64
        )
        # Two cascaded one-poles, not one: demodulating x*sin puts the wanted
        # term at DC but also an image at 2*f. A single pole leaves ~0.5/2f of
        # that image in I/Q, which re-modulates into a few percent of residual
        # tone. Cascading squares the rejection (~0.1%) for one extra multiply.
        self._i = np.zeros((k, self.n_channels), dtype=np.float64)
        self._q = np.zeros((k, self.n_channels), dtype=np.float64)
        self._i2 = np.zeros((k, self.n_channels), dtype=np.float64)
        self._q2 = np.zeros((k, self.n_channels), dtype=np.float64)
        # One-pole coefficient for the chosen tracking bandwidth.
        self._alpha = (
            1.0 - math.exp(-2.0 * math.pi * self.track_hz / self.fs_hz)
            if self.fs_hz > 0
            else 0.0
        )
        # The two cascaded one-poles as one 2-state system per (harmonic,
        # channel), driven by u = x*sin (or x*cos); the output is the second
        # pole's value *before* this sample's update, which is what the
        # reconstruction subtracts.
        a = self._alpha
        r = 1.0 - a
        self._ss = _StateSpace2([[r, 0.0], [a * r, r]], [a, a * a], [0.0, 1.0], 0.0)

    def reset(self) -> None:
        self._theta[:] = 0.0
        self._i[:] = 0.0
        self._q[:] = 0.0
        self._i2[:] = 0.0
        self._q2[:] = 0.0
        self._df = 0.0
        self._prev_angle = None
        self._since_lock = 0

    def advance(self, n_samples: int) -> None:
        """Move the oscillators forward over ``n_samples`` that never arrived.

        The tracker assumes consecutive samples are consecutive in time. A
        dropped-sample gap breaks that: real time (and the mains) moves on while
        the oscillator does not, so after an 11 ms gap the reference is 198
        degrees out and the I/Q estimate -- 0.3 s of memory at 0.5 Hz -- is
        wrong until it re-converges. On s01_f_30 take1 gaps arrived 2.2x a
        second, the tracker was never converged, and it cancelled nothing while
        the old notch bank still managed 2.5x. Advancing the phase over the gap
        keeps the reference on the grid; the I/Q estimates are still valid.
        """
        if n_samples > 0 and self.freqs:
            self._theta = np.mod(self._theta + self._dtheta * float(n_samples), 2.0 * math.pi)

    def process(self, x: np.ndarray) -> np.ndarray:
        """Cancel the tracked harmonics in a ``(N, C)`` block.

        Vectorised form of the per-sample recursion

            out = x - 2*sum_k(I2*sin + Q2*cos)
            I <- I + a*(x*sin - I),  I2 <- I2 + a*(I - I2)   (same for Q, cos)
            theta <- theta + dtheta
        """
        x = np.asarray(x, dtype=np.float64)
        if not self.freqs or x.shape[0] == 0:
            return x
        n, c = x.shape
        k = len(self.freqs)
        th = self._theta[None, :] + np.arange(n, dtype=np.float64)[:, None] * self._dtheta[None, :]
        sn, cs = np.sin(th), np.cos(th)                               # (N, K)
        u = (x[:, None, :] * sn[:, :, None]).reshape(n, k * c)
        v = (x[:, None, :] * cs[:, :, None]).reshape(n, k * c)
        i2_seq, s_i = self._ss.run(u, np.vstack([self._i.reshape(-1), self._i2.reshape(-1)]))
        q2_seq, s_q = self._ss.run(v, np.vstack([self._q.reshape(-1), self._q2.reshape(-1)]))
        i2_seq = i2_seq.reshape(n, k, c)
        q2_seq = q2_seq.reshape(n, k, c)
        out = x - 2.0 * (i2_seq * sn[:, :, None] + q2_seq * cs[:, :, None]).sum(axis=1)
        self._i, self._i2 = s_i[0].reshape(k, c).copy(), s_i[1].reshape(k, c).copy()
        self._q, self._q2 = s_q[0].reshape(k, c).copy(), s_q[1].reshape(k, c).copy()
        # Keep the accumulator bounded without disturbing the phase.
        self._theta = np.mod(self._theta + n * self._dtheta, 2.0 * math.pi)
        self._relock(n)
        return out

    def _relock(self, n_samples: int) -> None:
        """Pull the oscillators onto the real grid frequency, once per block.

        With the oscillator at f and the grid at f+d, the fundamental's phasor
        (I + jQ) rotates at exactly d, so the angle it advances over a block
        measures the error directly. Harmonic k is locked to the same grid, so
        its error is k*d -- one estimator serves the whole bank.
        """
        if self.lock_hz <= 0 or not self.freqs or self.fs_hz <= 0:
            return
        self._since_lock += n_samples
        if self._since_lock < self._lock_every:
            return
        n_samples, self._since_lock = self._since_lock, 0
        i0 = float(self._i2[0].mean())
        q0 = float(self._q2[0].mean())
        mag = math.hypot(i0, q0)
        angle = math.atan2(q0, i0)
        prev = self._prev_angle
        self._prev_angle = angle
        if prev is None or mag <= 1e-9:
            return
        dt = n_samples / self.fs_hz
        # Unwrap into (-pi, pi]; a block must be short enough that the true
        # rotation cannot alias, which holds for any |d| < 1/(2*dt).
        d_ang = (angle - prev + math.pi) % (2.0 * math.pi) - math.pi
        err = d_ang / (2.0 * math.pi * dt)
        if not math.isfinite(err):
            return
        # Slew-limit, then clamp the total correction.
        step = max(-self.lock_hz, min(self.lock_hz, err)) * 0.25
        self._df = max(-self.lock_hz, min(self.lock_hz, self._df + step))
        self._dtheta = np.array(
            [2.0 * math.pi * (f + k * self._df) / self.fs_hz
             for k, f in enumerate(self.freqs, start=1)],
            dtype=np.float64,
        )

    @property
    def locked_offset_hz(self) -> float:
        """Current estimate of grid minus nominal, in Hz (diagnostic)."""
        return self._df


class CausalEmgFilter:
    """Causal HPF -> line-notch harmonics -> LPF chain plus a rectified envelope.

    ``process`` returns ``(filtered, envelope)`` for an input ``(N, C)`` µV block,
    keeping filter state so successive blocks join seamlessly.

    Line interference is handled at every harmonic of ``notch_hz`` that sits in
    the EMG band below Nyquist (50, 100, …, 400 Hz at 2 kHz / 50 Hz line);
    harmonics at or near Nyquist are skipped. ``mains_mode`` picks how:

    ``"track"`` (default)
        :class:`HarmonicCanceller` -- fits and subtracts the tone. ~1 Hz of band
        lost per harmonic, and it follows a grid that is not exactly 50.000 Hz.
        Chain is DC-block -> canceller -> HPF -> LPF, and it must be fed through
        ``process_timed`` when the stream can have dropped samples: without the
        timestamps the reference drifts off the grid at every gap and the
        canceller removes nothing (measured: 402 uV vs 403 raw on a 2.9% dropout
        recording, while the notch bank still halved it).
    ``"notch"``
        the original RBJ biquad bank. ``notch_q`` is multiplied by the harmonic
        number so each notch keeps the same absolute width (~5 Hz), but that is
        still 40 Hz of the EMG band gone across eight harmonics, always centred
        on the nominal frequency.
    ``"off"``
        band-pass only.

    Filtering is salvage, not a fix: modulated interference -- the signature of a
    bias loop swinging against its limit -- forms a broad shoulder around each
    harmonic that no harmonic method removes, and nothing recovers a signal a
    clipping DRL never resolved.
    """

    def __init__(
        self,
        fs_hz: float,
        n_channels: int = 8,
        *,
        hpf_hz: float = 20.0,
        lpf_hz: float = EMG_BAND_HZ,
        notch_hz: float = 50.0,
        notch_harmonics: int = DEFAULT_NOTCH_HARMONICS,
        notch_q: float = 10.0,
        env_lpf_hz: float = 5.0,
        mains_mode: str = "track",
        track_hz: float = 0.5,
    ) -> None:
        self.fs_hz = float(fs_hz)
        self.n_channels = int(n_channels)
        nyq = self.fs_hz / 2.0
        hpf = min(hpf_hz, nyq * 0.5)
        lpf = min(lpf_hz, nyq * 0.9)
        self.notch_freqs = line_notch_freqs(
            self.fs_hz, notch_hz, max_harmonics=notch_harmonics
        )
        if mains_mode not in ("track", "notch", "off"):
            raise ValueError(f"mains_mode must be track/notch/off, got {mains_mode!r}")
        self.mains_mode = mains_mode
        self._sections: list[Any] = []
        if hpf < lpf and self.fs_hz > 0:
            if mains_mode == "track" and self.notch_freqs:
                # DC block -> canceller -> HPF -> LPF. The canceller goes ahead of
                # the 20 Hz HPF so a phase jump at a dropped-sample gap never
                # reaches the HPF as a raw step (see _dcblock for the numbers).
                self._sections.append(_dcblock(self.fs_hz, 1.0, self.n_channels))
                self._sections.append(
                    HarmonicCanceller(
                        self.fs_hz, self.n_channels, self.notch_freqs, track_hz=track_hz
                    )
                )
            self._sections.append(_highpass(self.fs_hz, hpf, math.sqrt(0.5), self.n_channels))
            if mains_mode == "notch":
                for h, f0 in enumerate(self.notch_freqs, start=1):
                    # Q scales with the harmonic number so every notch keeps the SAME
                    # absolute width (BW = f0/Q = notch_hz/notch_q, ~5 Hz by default).
                    self._sections.append(_notch(self.fs_hz, f0, notch_q * h, self.n_channels))
            self._sections.append(_lowpass(self.fs_hz, lpf, math.sqrt(0.5), self.n_channels))
        env_fc = min(env_lpf_hz, nyq * 0.9)
        self._env = _lowpass(self.fs_hz, env_fc, math.sqrt(0.5), self.n_channels) if env_fc > 0 else None
        self._last_t_us: float | None = None

    def reset(self) -> None:
        for s in self._sections:
            s.reset()
        if self._env is not None:
            self._env.reset()
        self._last_t_us = None

    def advance(self, n_samples: int) -> None:
        """Account for ``n_samples`` that were dropped before the next block.

        Only the harmonic canceller carries a phase that must follow real time;
        the biquads just keep their state, which is the right thing for a short
        hole.
        """
        for sec in self._sections:
            adv = getattr(sec, "advance", None)
            if adv is not None:
                adv(int(n_samples))

    def process_timed(
        self, t_us: np.ndarray, uv: np.ndarray, *, gap_factor: float = 1.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """``process`` for a stream that may have dropped samples.

        Splits the block wherever consecutive timestamps are more than
        ``gap_factor`` sample periods apart -- and at the join with the previous
        call -- and advances the filter over the missing samples, so the mains
        reference stays on the grid across dropouts. Always prefer this when
        timestamps exist; ``process`` alone treats the rows as contiguous.
        """
        t = np.asarray(t_us, dtype=np.float64).reshape(-1)
        x = np.asarray(uv, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        n = x.shape[0]
        if n == 0 or t.shape[0] != n or self.fs_hz <= 0:
            return self.process(x)
        dt = 1e6 / self.fs_hz
        last = getattr(self, "_last_t_us", None)
        if last is not None and t[0] - last > gap_factor * dt:
            self.advance(int(round((t[0] - last) / dt)) - 1)
        cuts = np.flatnonzero(np.diff(t) > gap_factor * dt) + 1
        filt = np.empty((n, x.shape[1]), dtype=np.float32)
        env = np.empty_like(filt)
        start = 0
        for c in list(cuts) + [n]:
            f, e = self.process(x[start:c])
            filt[start:c], env[start:c] = f, e
            if c < n:
                self.advance(int(round((t[c] - t[c - 1]) / dt)) - 1)
            start = c
        self._last_t_us = float(t[-1])
        return filt, env

    def process(self, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(uv, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[0] == 0:
            zero = np.zeros_like(x, dtype=np.float32)
            return zero, zero
        y = x
        for s in self._sections:
            y = s.process(y)
        if self._env is not None:
            env = self._env.process(np.abs(y))
        else:
            env = np.abs(y)
        return y.astype(np.float32), env.astype(np.float32)


def rectified_envelope(uv: np.ndarray, fs_hz: float, *, env_lpf_hz: float = 5.0) -> np.ndarray:
    """5 Hz low-pass of |signal|, the same envelope stage as ``CausalEmgFilter``."""
    x = np.asarray(uv, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[:, None]
    if x.shape[0] == 0 or fs_hz <= 0:
        out = np.abs(x).astype(np.float32)
        return out[:, 0] if single else out
    env_fc = min(float(env_lpf_hz), (float(fs_hz) / 2.0) * 0.9)
    if env_fc <= 0:
        out = np.abs(x).astype(np.float32)
        return out[:, 0] if single else out
    env = _lowpass(float(fs_hz), env_fc, math.sqrt(0.5), x.shape[1]).process(np.abs(x))
    out = np.asarray(env, dtype=np.float32)
    return out[:, 0] if single else out
