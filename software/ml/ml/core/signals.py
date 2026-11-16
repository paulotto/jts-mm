# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Signal conditioning shared by every ML entry point (array arguments only).

* EMG: jtsmm's real-time chain (:class:`jtsmm.viewer.processing.CausalEmgFilter`,
  tracking mains canceller over all in-band harmonics, 20-450 Hz) run through
  ``process_timed`` so dropped samples do not detune the canceller. The chain is
  causal; nothing here looks at future samples or at labels.
* Masks: ADC rail and lead-off (bit ``i`` of the status word is CH(i+1) of the
  *recorded* channel order, so bits are looked up by original column index),
  optional flat-line detection. Masked raw samples are held at the last valid
  value before filtering (causal fill) and set to NaN afterwards.
* Normalization references, all label-free and all computed on the *same causal
  chain* as the windows (a reference from biomech's zero-phase chain has a
  different gain): ``p95`` of the causal RMS envelope in the channel's MVC phase,
  ``max_recorded`` (largest envelope over the MVC phase and every task, biomech
  D11), ``max1s`` (largest 1-s mean of it), rest (RMS in the unlabelled ``rest``
  phase) or none. With the default rest-relative feature normalization (D13) the
  features do not depend on a per-participant gain, so the reference matters for
  ``emg_norm: mvc`` only.
* IMU: body-frame angular velocity by *backward* differences of the game rotation
  vector and gravity-free acceleration from the raw ``accel`` (which includes
  gravity; ``linear_accel`` was never recorded).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d

from ml.config.classes import DIGASTRIC

G = 9.80665


# -- timestamps ------------------------------------------------------------------------
def sanitize_timestamps(t_us: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Sort by time and drop repeated stamps (first kept); arrays follow the same rows."""
    from jtsmm.imu.batch import sanitize_stream

    return sanitize_stream(t_us, *arrays)


# -- EMG -------------------------------------------------------------------------------
def line_hz(uv: np.ndarray, fs: float, t_us: np.ndarray | None, nominal_hz: float = 50.0) -> float:
    """Grid frequency of a take, measured on its loudest channel (nominal if unclear)."""
    from jtsmm.emg.mains import measure_line_hz

    a = np.asarray(uv, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.shape[0] < int(fs):
        return float(nominal_hz)
    loud = int(np.argmax(a.std(axis=0)))
    # The nominal grid first, then the other one (50 <-> 60 Hz): a 60-Hz recording must not
    # be "cleaned" at 50 Hz (as biomech.emg.pipeline.take_line_hz).
    for nom in (nominal_hz, 60.0 if round(nominal_hz) == 50 else 50.0):
        f = measure_line_hz(a[:, loud], fs, nom, t_us=t_us)
        if f is not None:
            return float(f)
    return float(nominal_hz)


def causal_emg(t_us: np.ndarray, uv: np.ndarray, fs: float, line: float, *,
               hpf_hz: float = 20.0, lpf_hz: float = 450.0) -> np.ndarray:
    """Causally filtered (N, C) EMG in uV (tracking mains canceller + band-pass)."""
    from jtsmm.viewer.processing import CausalEmgFilter

    f = CausalEmgFilter(fs, uv.shape[1], hpf_hz=hpf_hz, lpf_hz=lpf_hz, notch_hz=line,
                        mains_mode="track")
    out, _ = f.process_timed(np.asarray(t_us), np.asarray(uv, dtype=np.float64))
    return np.asarray(out, dtype=np.float64)


def hold_fill(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Replace masked samples by the last valid sample of the same channel (0 before any).

    Causal: unlike interpolation it never reads a later sample.
    """
    if mask is None or not np.any(mask):
        return x
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx = np.where(mask, -1, np.arange(n)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    cols = np.broadcast_to(np.arange(x.shape[1]), idx.shape)
    return np.where(idx >= 0, x[np.maximum(idx, 0), cols], 0.0)


def rail_mask(raw_uv: np.ndarray, full_scale_uv: np.ndarray | float | None,
              frac: float = 0.99) -> np.ndarray:
    """Non-finite samples and samples within ``1 - frac`` of the ADC full scale."""
    x = np.asarray(raw_uv)
    m = ~np.isfinite(x)
    if full_scale_uv is not None:
        m |= np.abs(x) >= frac * np.broadcast_to(np.asarray(full_scale_uv, float), x.shape[1:])
    return m


def leadoff_mask(status: np.ndarray | None, channel_index: list[int], n: int) -> np.ndarray:
    """(N, C) lead-off from the ``[statp, statn]`` status word.

    ``channel_index[c]`` is column c's index in the recorded order; bit i = CH(i+1)
    exists for CH1-6 only.
    """
    out = np.zeros((n, len(channel_index)), dtype=bool)
    if status is None:
        return out
    st = np.asarray(status)
    if st.ndim != 2 or st.shape[0] != n or st.shape[1] < 2:
        return out
    either = st[:, 0].astype(np.int64) | st[:, 1].astype(np.int64)
    for c, src in enumerate(channel_index):
        if 0 <= src < 6:
            out[:, c] = (either & (1 << src)) != 0
    return out


def flatline_mask(x: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Samples inside a ``window_s`` stretch with zero variance (disconnected electrode)."""
    x = np.asarray(x, dtype=np.float64)
    if window_s <= 0 or x.shape[0] < 2:
        return np.zeros(x.shape, dtype=bool)
    w = max(2, int(round(window_s * fs)))
    if x.shape[0] < w:
        return np.zeros(x.shape, dtype=bool)
    mean = uniform_filter1d(x, w, axis=0, mode="nearest")
    var = uniform_filter1d(x * x, w, axis=0, mode="nearest") - mean ** 2
    flat = var <= 1e-12 * np.maximum(1.0, mean ** 2)
    return binary_dilation(flat, structure=np.ones((w, 1), bool))


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """Extend a (N, C) mask by ``k`` samples on both sides."""
    if k <= 0 or not np.any(mask):
        return mask
    return binary_dilation(mask, structure=np.ones((2 * k + 1, 1), bool))


def rms_envelope(x: np.ndarray, fs: float, window_s: float = 0.15) -> np.ndarray:
    """Centred moving RMS (offline references only, never a model input)."""
    w = max(int(round(window_s * fs)), 1)
    return np.sqrt(np.maximum(uniform_filter1d(np.nan_to_num(x) ** 2, w, axis=0,
                                               mode="nearest"), 0.0))


def causal_rms_envelope(x: np.ndarray, fs: float, window_s: float = 0.15,
                        min_valid: float = 0.5) -> np.ndarray:
    """Trailing moving RMS over the valid (finite) samples of the last ``window_s``.

    NaN where fewer than ``min_valid`` of the window is valid or the sample itself is
    masked. Its maximum and percentiles equal those of a centred envelope up to the
    edges (it is the same curve delayed by half a window).
    """
    x = np.asarray(x, dtype=np.float64)
    w = max(int(round(window_s * fs)), 1)
    v = np.isfinite(x)
    cs = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), np.where(v, x * x, 0.0)]), axis=0)
    cn = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), v]), axis=0)
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(hi - w, 0)
    s, n = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.sqrt(np.maximum(s / n, 0.0))
    return np.where(v & (n >= min_valid * w), env, np.nan)


def _trailing_mean_max(env: np.ndarray, n: int, min_valid: float = 0.5) -> float:
    if env.size < n:
        return float("nan")
    v = np.isfinite(env)
    cs = np.concatenate([[0.0], np.cumsum(np.where(v, env, 0.0))])
    cn = np.concatenate([[0], np.cumsum(v)])
    s, k = cs[n:] - cs[:-n], cn[n:] - cn[:-n]
    ok = k >= min_valid * n
    return float(np.max(s[ok] / k[ok])) if ok.any() else float("nan")


REFERENCE_MODES = ("p95", "max_recorded", "max1s")


def emg_reference(mvc: dict[str, np.ndarray], tasks: dict[str, np.ndarray] | None,
                  channels: list[str], fs: float, *, mode: str = "p95",
                  percentile: float = 95.0, envelope_s: float = 0.15
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel reference (uV) from causally filtered EMG and its source per channel.

    ``mvc`` maps ``mvc_opening`` / ``mvc_closing`` to filtered (N, C) EMG with masked
    samples NaN; digastrics take the opening phase, closers the closing phase (as
    biomech's ``EmgPrep.refs``). ``tasks`` maps motion-take task phases to the same.
    Modes mirror biomech's ``signals.emg.reference``: ``p95`` of the causal envelope in
    the MVC phase; ``max_recorded`` its largest value over the MVC phase and every task
    (D11); ``max1s`` the largest 1-s mean over the same. ``source`` is ``mvc``, ``task``
    or ``none`` per channel.
    """
    if mode not in REFERENCE_MODES:
        raise ValueError(f"reference mode must be one of {REFERENCE_MODES}, got {mode!r}")
    ref = np.full(len(channels), np.nan)
    src = np.array(["none"] * len(channels), dtype=object)

    def stat(env: np.ndarray, kind: str) -> float:
        if not np.isfinite(env).any():
            return float("nan")
        if kind == "p95":
            return float(np.nanpercentile(env, percentile))
        if kind == "max_recorded":
            return float(np.nanmax(env))
        return _trailing_mean_max(env, int(round(fs)))

    for c, name in enumerate(channels):
        x = mvc.get("mvc_opening" if name in DIGASTRIC else "mvc_closing")
        if x is not None:
            ref[c] = stat(causal_rms_envelope(x[:, c:c + 1], fs, envelope_s)[:, 0], mode)
            if np.isfinite(ref[c]):
                src[c] = "mvc"
        if mode == "p95":
            continue
        for x in (tasks or {}).values():
            v = stat(causal_rms_envelope(x[:, c:c + 1], fs, envelope_s)[:, 0], mode)
            if np.isfinite(v) and not v <= ref[c]:
                ref[c], src[c] = v, "task"
    return ref, src


def mvc_reference(segments: dict[str, np.ndarray], channels: list[str], fs: float, *,
                  percentile: float = 95.0, envelope_s: float = 0.15) -> np.ndarray:
    """Per-channel MVC reference (uV): p95 of the causal RMS envelope of the filtered MVC."""
    return emg_reference(segments, None, channels, fs, mode="p95", percentile=percentile,
                         envelope_s=envelope_s)[0]


def rest_reference(x_rest: np.ndarray) -> np.ndarray:
    """Per-channel RMS (uV) of the filtered, unlabelled rest phase."""
    return np.sqrt(np.nanmean(np.asarray(x_rest, float) ** 2, axis=0))


# -- IMU -------------------------------------------------------------------------------
def imu_signals(t_grid: np.ndarray, q_mta: tuple, q_cra: tuple, a_mta: tuple, a_cra: tuple
                ) -> np.ndarray:
    """(N, 12): body-frame angular velocity (deg/s) of MTA, CRA; gravity-free accel (m/s^2).

    Each argument is ``(t_us, values)`` sanitized (strictly increasing time).
    Angular velocity uses the backward difference q[k-1]^-1 q[k] (causal).
    """
    from jtsmm.imu.batch import quat_conj, quat_mul, quat_resample, quat_to_matrix, quat_to_rotvec

    out = []
    for (tq, q), (ta, a) in ((q_mta, a_mta), (q_cra, a_cra)):
        qg = quat_resample(tq, q, t_grid)
        dq = quat_mul(quat_conj(np.vstack([qg[:1], qg[:-1]])), qg)
        dt = np.diff(t_grid, prepend=t_grid[0] - (t_grid[1] - t_grid[0])) * 1e-6
        w = np.degrees(quat_to_rotvec(dq)) / dt[:, None]
        ag = np.column_stack([np.interp(t_grid, ta, a[:, k]) for k in range(3)])
        R = quat_to_matrix(qg)
        g_body = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, G]))
        out.append((w, ag - g_body))
    (w_m, a_m), (w_c, a_c) = out
    return np.column_stack([w_m, w_c, a_m, a_c])
