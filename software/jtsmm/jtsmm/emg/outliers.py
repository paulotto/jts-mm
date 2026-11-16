# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Drop short EMG envelope spikes that are not a sustained contraction.

The detector uses a 150 ms RMS envelope. A crossing above 150 µV is removed
when it lasts at most 0.4 s and the surrounding second is quiet. MVC closers
stay above that level for more than a second, so their baseline keeps them.
"""

from __future__ import annotations

import numpy as np

from jtsmm.emg.onset import odd_window

_OUTLIER_UV = 150.0
_OUTLIER_MAX_RUN_S = 0.40
_OUTLIER_CONTEXT_S = 1.0
_OUTLIER_CONTEXT_PAD_S = 0.30
_OUTLIER_MERGE_GAP_S = 0.30
_OUTLIER_BASELINE_FRAC = 0.75


def _rms_envelope(uv: np.ndarray, fs_hz: float, window_s: float) -> np.ndarray:
    w = odd_window(window_s, fs_hz)            # centred: odd length, no half-sample shift
    kernel = np.ones(w, dtype=np.float64) / w
    if uv.ndim == 1:
        return np.sqrt(np.convolve(uv * uv, kernel, mode="same"))
    return np.column_stack(
        [np.sqrt(np.convolve(uv[:, i] * uv[:, i], kernel, mode="same")) for i in range(uv.shape[1])]
    )


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    if m.size == 0 or not np.any(m):
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts.insert(0, 0)
    if m[-1]:
        ends.append(int(m.size))
    return list(zip(starts, ends))


def _merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged: list[list[int]] = [[runs[0][0], runs[0][1]]]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged]


def suppress_envelope_outliers(
    uv: np.ndarray,
    fs_hz: float,
    *,
    level_uv: float = _OUTLIER_UV,
    max_run_s: float = _OUTLIER_MAX_RUN_S,
    envelope_s: float = 0.15,
) -> np.ndarray:
    """Zero short RMS spikes above ``level_uv``, including one window of skirt.

    Returns a float64 array with the same shape as ``uv``. Channels whose
    surrounding level is already high are left unchanged.
    """
    a = np.asarray(uv, dtype=np.float64)
    single = a.ndim == 1
    if single:
        a = a[:, None]
    env = _rms_envelope(a, fs_hz, envelope_s)
    out = a.copy()
    pad = int(round(envelope_s * fs_hz))
    gap = int(round(_OUTLIER_MERGE_GAP_S * fs_hz))
    ctx_n = int(round(_OUTLIER_CONTEXT_S * fs_hz))
    ctx_pad = int(round(_OUTLIER_CONTEXT_PAD_S * fs_hz))
    n = out.shape[0]
    for i in range(out.shape[1]):
        col = env[:, i]
        for start, end in _merge_runs(_true_runs(col > level_uv), gap):
            if (end - start) / fs_hz > max_run_s:
                continue
            left = col[max(0, start - ctx_n) : max(0, start - ctx_pad)]
            right = col[min(n, end + ctx_pad) : min(n, end + ctx_n)]
            base = np.concatenate([left, right])
            base = base[np.isfinite(base)]
            if base.size >= 8 and float(np.percentile(base, 95.0)) >= level_uv * _OUTLIER_BASELINE_FRAC:
                continue
            out[max(0, start - pad) : min(n, end + pad), i] = 0.0
    if single:
        return out[:, 0]
    return out
