# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Signal-quality checks on recorded EMG (mains domination, saturated DRL).

Why this exists: every EMG check the pipeline had judged *rates and continuity*.
A session could therefore be 98% mains interference with the bias amplifier
clipping against its rail and still report ``RESULT: OK`` -- which is exactly
what happened on session_20260918T081708Z (7.8 mV RMS on CH1, of which 98% was
49.98 Hz and its harmonics, with ``BIAS_DRN`` railed 45% of the time).

Two independent, physically grounded measures, both computed from the stored
counts so no extra instrumentation is needed:

**Mains domination.** Per active muscle channel, the share of 20-450 Hz power
that sits at the line frequency and its harmonics. Resting skin EMG is 5-20 uV
RMS and broadband; mains pickup is a narrow, grid-tracking sine. Flagging needs
*both* a high mains share and a high absolute amplitude, so a genuinely quiet
channel with a little hum, and a strong contraction with none, both stay quiet.

**DRL saturation.** CH8 reads back the ADS1299 bias amplifier's own output
(MUX8 = BIAS_DRN). A regulating loop sits mid-range; one pinned to its rail has
run out of headroom and is injecting a clipped copy of the common-mode into the
subject instead of cancelling it. That turns the reference electrode from the
fix into the cause, and it is invisible in the muscle channels alone.

Analysis is windowed and bounded, so a long session costs no more than a short
one, and the *fraction of windows* affected answers the question that a single
whole-file number cannot: was it bad throughout, or only part of the time.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from jtsmm.emg.capture import (
    DRL_AC_ERR_MV,
    DRL_AC_WARN_MV,
    drl_ac_mv,
    drl_rail_fraction,
    drl_readback_status,
)

__all__ = [
    "analyse_emg_signal",
    "EMG_BAND_LO_HZ",
    "EMG_BAND_HI_HZ",
    "MAINS_SHARE_BAD",
    "MAINS_RMS_MIN_UV",
    "MAINS_BAD_WINDOW_FRAC_ERR",
    "MAINS_BAD_WINDOW_FRAC_WARN",
    "DRL_AC_WARN_MV",
    "DRL_AC_ERR_MV",
]

# Band the verdict is formed in: below 20 Hz is motion/DC, above 450 Hz is not EMG.
EMG_BAND_LO_HZ = 20.0
EMG_BAND_HI_HZ = 450.0
# Half-width of the notch used to attribute power to a mains harmonic. Wide
# enough for real grid wander (measured 49.98 Hz, not 50.000) and for FFT
# leakage at a 2 s window; narrow enough not to swallow the EMG band.
MAINS_BIN_HALF_HZ = 1.5
# A window counts as mains-dominated when *both* hold: more than half the
# in-band power is at the line frequency, and the channel is far louder than
# resting EMG. Either alone is normal (quiet channel with faint hum; strong
# contraction with clean mains).
MAINS_SHARE_BAD = 0.5
MAINS_RMS_MIN_UV = 200.0
# Some bad windows are worth a warning; a session that is mostly bad is an error.
MAINS_BAD_WINDOW_FRAC_WARN = 0.05
MAINS_BAD_WINDOW_FRAC_ERR = 0.20
# DRL_AC_* and drl_ac_mv() live in jtsmm.emg.capture and are re-exported here so
# `jtsmm emg check` and `jtsmm integrity` share one definition. The verdict is on
# the bias node's AC content, not its absolute level: with IN8P unwired a healthy
# quiescent loop reads pinned at full scale (see capture.py).

_FULL_SCALE = float(1 << 23)


def _window_starts(n_samples: int, window: int, max_windows: int) -> list[int]:
    """Up to ``max_windows`` window starts spread evenly over the record."""
    if n_samples < window:
        return []
    n_full = n_samples // window
    if n_full <= max_windows:
        return [i * window for i in range(n_full)]
    # Spread rather than take the head: intermittent faults are what we are after.
    return [int(round(k)) * window for k in np.linspace(0, n_full - 1, max_windows)]


def _detect_line_hz(
    blocks: Sequence[np.ndarray], channels: Sequence[int], lsb_uv: float, fs_hz: float,
    candidates: Sequence[float] = (50.0, 60.0),
) -> float:
    """Pick the line frequency by TOTAL power across every window and channel.

    Summed, not voted per window: interference is often intermittent, so a
    per-window argmax lets the clean majority outvote the few bad windows and
    the share is then measured at the wrong frequency (which silently hid a
    real fault in testing). Summing lets the bad windows, where the energy
    actually is, decide. With no hum anywhere the pick is arbitrary, but then
    every share is low and nothing fires either way.
    """
    totals = {f0: 0.0 for f0 in candidates}
    for b in blocks:
        for ch in channels:
            if ch >= b.shape[1]:
                continue
            x = b[:, ch] * lsb_uv
            x = x - x.mean()
            spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
            freqs = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
            for f0 in candidates:
                m = (freqs >= f0 - MAINS_BIN_HALF_HZ) & (freqs <= f0 + MAINS_BIN_HALF_HZ)
                if np.any(m):
                    totals[f0] += float(np.sum(spec[m]))
    return float(max(totals, key=lambda k: totals[k]))


def _band_and_mains_power(
    block_uv: np.ndarray, fs_hz: float, line_hz: float
) -> tuple[float, float]:
    """(in-band power, power at line harmonics) for one window, in uV^2 units."""
    x = block_uv - block_uv.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    in_band = (freqs >= EMG_BAND_LO_HZ) & (freqs <= EMG_BAND_HI_HZ)
    band_p = float(np.sum(spec[in_band]))
    mains = np.zeros_like(in_band)
    k = 1
    while k * line_hz <= EMG_BAND_HI_HZ:
        f0 = k * line_hz
        mains |= (freqs >= f0 - MAINS_BIN_HALF_HZ) & (freqs <= f0 + MAINS_BIN_HALF_HZ)
        k += 1
    mains_p = float(np.sum(spec[mains & in_band]))
    return band_p, mains_p


def _band_rms_uv(block_uv: np.ndarray, fs_hz: float) -> float:
    """RMS of the 20-450 Hz content of one window."""
    x = block_uv - block_uv.mean()
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs_hz)
    spec[(freqs < EMG_BAND_LO_HZ) | (freqs > EMG_BAND_HI_HZ)] = 0.0
    return float(np.std(np.fft.irfft(spec, x.size)))


def analyse_emg_signal(
    samples: Any,
    *,
    fs_hz: float,
    lsb_uv: float,
    active_channels: Sequence[int],
    drl_column: int | None = 7,
    drl_meaningful: bool = True,
    line_hz: float | None = None,
    window_s: float = 2.0,
    max_windows: int = 24,
    lsb_uv_per_channel: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Judge recorded EMG for mains domination and a saturated DRL.

    @param samples          ``(N, 8)`` raw counts; an h5py dataset is read by slice.
    @param active_channels  0-based muscle columns to judge (powered-down ones excluded).
    @param drl_column       Column holding the ``BIAS_DRN`` readback, or None to skip.
    @param drl_meaningful   False when CH8 is not the bias readback (e.g. the
                            isolated ``probe_drl`` session), which skips the DRL test.
    @param line_hz          Mains frequency; None auto-picks 50 or 60 from the data.
    @param lsb_uv_per_channel  Per-column LSB (µV). CH7/CH8 run at a fixed gain
                            of 24 whatever the muscle gain, so the DRL column
                            must not be scaled with ``lsb_uv``; falls back to
                            ``lsb_uv`` for every column when None.
    @return dict with ``channels``, ``drl``, ``line_hz``, ``windows`` and a
            ``verdict`` block of booleans the caller turns into errors/warnings.
    """
    out: dict[str, Any] = {
        "channels": {},
        "drl": None,
        "line_hz": None,
        "windows": 0,
        "verdict": {"mains_error": False, "mains_warn": False,
                    "drl_error": False, "drl_warn": False},
    }
    n = int(getattr(samples, "shape", (0, 0))[0])
    if n == 0 or fs_hz <= 0 or lsb_uv <= 0:
        return out
    window = max(64, int(round(window_s * fs_hz)))
    starts = _window_starts(n, window, max_windows)
    if not starts:
        return out
    out["windows"] = len(starts)

    # Read each window once and reuse it for every measure.
    blocks = [np.asarray(samples[i : i + window, :], dtype=np.float64) for i in starts]

    if line_hz is None and active_channels:
        line_hz = _detect_line_hz(blocks, active_channels, lsb_uv, fs_hz)
    out["line_hz"] = float(line_hz) if line_hz else None

    for ch in active_channels:
        if ch >= blocks[0].shape[1]:
            continue
        ch_lsb = (
            float(lsb_uv_per_channel[ch])
            if lsb_uv_per_channel is not None and ch < len(lsb_uv_per_channel)
            else lsb_uv
        )
        shares: list[float] = []
        rms: list[float] = []
        bad = 0
        for b in blocks:
            col = b[:, ch] * ch_lsb
            band_p, mains_p = _band_and_mains_power(col, fs_hz, float(line_hz or 50.0))
            share = (mains_p / band_p) if band_p > 0 else 0.0
            r = _band_rms_uv(col, fs_hz)
            shares.append(share)
            rms.append(r)
            if share >= MAINS_SHARE_BAD and r >= MAINS_RMS_MIN_UV:
                bad += 1
        frac_bad = bad / len(blocks)
        shares_a = np.asarray(shares)
        rms_a = np.asarray(rms)
        is_bad = (shares_a >= MAINS_SHARE_BAD) & (rms_a >= MAINS_RMS_MIN_UV)
        # When only part of the session is bad, the session medians describe the
        # GOOD part; quote the affected windows instead so the message matches
        # what is actually wrong.
        sel = is_bad if is_bad.any() else np.ones_like(is_bad, dtype=bool)
        out["channels"][ch] = {
            "median_mains_share": float(np.median(shares_a)),
            "median_band_rms_uv": float(np.median(rms_a)),
            "max_band_rms_uv": float(np.max(rms_a)),
            "bad_median_mains_share": float(np.median(shares_a[sel])),
            "bad_median_band_rms_uv": float(np.median(rms_a[sel])),
            "bad_window_frac": float(frac_bad),
        }
        if frac_bad >= MAINS_BAD_WINDOW_FRAC_ERR:
            out["verdict"]["mains_error"] = True
        elif frac_bad > MAINS_BAD_WINDOW_FRAC_WARN:
            out["verdict"]["mains_warn"] = True

    if drl_column is not None and drl_meaningful and drl_column < blocks[0].shape[1]:
        drl_lsb = (
            float(lsb_uv_per_channel[drl_column])
            if lsb_uv_per_channel is not None and drl_column < len(lsb_uv_per_channel)
            else lsb_uv
        )
        col = np.concatenate([b[:, drl_column] for b in blocks])
        # AC per window, pooled as an RMS: the windows are spread over the whole
        # session, and taking one std over their concatenation counted the DC
        # drift BETWEEN windows as "AC" swing.
        per_win = [drl_ac_mv(b[:, drl_column], drl_lsb) for b in blocks]
        ac_mv = float(np.sqrt(np.mean(np.square(per_win)))) if per_win else 0.0
        rail_frac = drl_rail_fraction(col)
        status = drl_readback_status(ac_mv, rail_frac)
        out["drl"] = {
            "ac_mv": ac_mv,
            "rail_frac": rail_frac,
            # Same grading as `jtsmm emg check`; "n/a" when the column is railed
            # (a clipped ADC has no AC content, which is not a healthy loop).
            "status": status,
            "median_abs_frac_fs": float(np.median(np.abs(col)) / _FULL_SCALE),
        }
        # Informational until the driven loop is bench-verified: warn, never fail.
        if status in ("warn", "high"):
            out["verdict"]["drl_warn"] = True

    return out
