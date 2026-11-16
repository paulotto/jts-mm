# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The X3 closer-release detector, streaming and on the causal chain (ml08 §4).

:mod:`biomech.timing.release` runs the detector offline on a whole phase of the causally
cleaned EMG (E27). :class:`StreamRelease` is the same detector fed sample blocks of the
causal chain (:class:`ml.online.stream.CausalChain`, masked samples NaN):

* group envelope E(t): per closer the trailing RMS over ``window_s`` of the valid samples
  (NaN when fewer than half are valid), in dB re the closer's rest level, mean over closers;
* baseline B(t): the trailing ``median_filter`` of the 10-ms decimated E over
  [t - ``base_hi_s``, t - ``base_lo_s``] (rank ``w // 2``, as ``scipy.ndimage``; the first
  values padded with the first one, as ``mode="nearest"``);
* decision when E <= B - ``drop_db`` for ``hold_s``; ``refractory_s`` between decisions.

On a NaN-free stretch fed from its first sample the decisions equal
``detect(t, causal_db(...))`` (tested, also with masked stretches): a NaN E value (all
closers masked) is held at the last valid value in the baseline, values before the first
valid one take that value, and a baseline window ending before it gives no baseline -- as
offline (E21). Difference: the detector runs continuously from the end of the rest phase
(offline it restarts in every task), so its baseline at a task start is not padded.
The rest level comes from the rest phase of the same causal chain.
"""

from __future__ import annotations

import warnings

import numpy as np

from biomech.timing.release import ReleaseParams


class StreamRelease:
    def __init__(self, fs: float, rest_level: np.ndarray, p: ReleaseParams) -> None:
        self.fs, self.p = float(fs), p
        self.rest = np.asarray(rest_level, dtype=np.float64)
        k = self.rest.size
        self.n = max(int(round(p.window_s * fs)), 1)
        self._x2 = np.zeros((self.n - 1, k))
        self._v = np.zeros((self.n - 1, k))
        self.step = max(int(round(0.01 * fs)), 1)
        lo, hi = int(round(p.base_lo_s * fs)), int(round(p.base_hi_s * fs))
        self.w = max((hi - lo) // self.step, 1)
        self.shift = lo // self.step
        self.hold = max(int(round(p.hold_s * fs)), 1)
        self.refr = int(round(p.refractory_s * fs))
        self._filled = np.zeros(1024)          # decimated E, NaN replaced causally
        self._nf = 0
        self._first = -1                        # index of the first valid decimated E
        self._last_v = np.nan
        self._base_j = -1
        self._base_v = np.nan
        self.i = 0                              # samples processed
        self._run = 0
        self._last = -self.refr - 1
        self.decisions: list[int] = []

    def envelope_db(self, x: np.ndarray) -> np.ndarray:
        """E for a block (carries the last ``window`` samples)."""
        x = np.asarray(x, dtype=np.float64)
        v = np.isfinite(x)
        x0 = np.where(v, x, 0.0)
        X2 = np.vstack([self._x2, x0 * x0])
        V = np.vstack([self._v, v.astype(np.float64)])
        n, m = self.n, x.shape[0]
        ker = np.ones(n)
        num = np.stack([np.convolve(X2[:, j], ker)[n - 1:n - 1 + m] for j in range(X2.shape[1])],
                       axis=1)
        den = np.stack([np.convolve(V[:, j], ker)[n - 1:n - 1 + m] for j in range(V.shape[1])],
                       axis=1)
        if n > 1:
            self._x2, self._v = X2[-(n - 1):], V[-(n - 1):]
        with np.errstate(invalid="ignore", divide="ignore"):
            env = np.sqrt(num / den)
            env[den < 0.5 * n] = np.nan
            db = 20.0 * np.log10(np.maximum(env, 1e-9) / self.rest[None, :])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean(db, axis=1)

    def _base(self, j: int) -> float:
        if j != self._base_j:
            jj = j - self.shift
            if jj < 0 or self._first < 0 or jj < self._first:
                self._base_v = np.nan
            else:
                idx = np.maximum(np.arange(jj - self.w + 1, jj + 1), 0)
                vals = np.sort(self._filled[idx])
                self._base_v = float(vals[self.w // 2])      # scipy.ndimage rank (upper median)
            self._base_j = j
        return self._base_v

    def feed(self, t_us: np.ndarray, x: np.ndarray) -> list[int]:
        """Process a block of closer samples (NaN = masked); returns new decision times."""
        e = self.envelope_db(x)
        m = e.size
        if m == 0:
            return []
        i = self.i + np.arange(m)
        for r in np.flatnonzero(i % self.step == 0):       # decimated E (10 ms)
            v = float(e[r])
            if self._nf == self._filled.size:
                self._filled = np.concatenate([self._filled, np.zeros(self._filled.size)])
            if np.isfinite(v):
                if self._first < 0:                     # leading NaN take the first valid value
                    self._first = self._nf
                    self._filled[:self._nf] = v
                self._last_v = v
            else:
                v = self._last_v if self._first >= 0 else 0.0   # causal hold (E21)
            self._filled[self._nf] = v
            self._nf += 1
        j = i // self.step
        uj, inv = np.unique(j, return_inverse=True)
        b = np.array([self._base(int(q)) for q in uj])[inv]
        with np.errstate(invalid="ignore"):
            below = np.isfinite(e) & np.isfinite(b) & (e <= b - self.p.drop_db)
        # Run length of ``below`` carried across blocks.
        last_false = np.maximum.accumulate(np.where(~below, np.arange(m), -1))
        run = np.where(last_false >= 0, np.arange(m) - last_false, np.arange(m) + 1 + self._run)
        new = []
        for r in np.flatnonzero(run == self.hold):
            if i[r] - self._last > self.refr:
                new.append(int(t_us[r]))
                self._last = int(i[r])
        self._run = int(run[-1])
        self.i += m
        self.decisions += new
        return new


__all__ = ["ReleaseParams", "StreamRelease"]
