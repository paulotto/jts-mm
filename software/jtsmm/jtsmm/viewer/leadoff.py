# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Decode the ADS1299 status word carried in every EMG_BLOCK record.

The first 3 payload bytes of an EMG block are the 24-bit ADS1299 status word::

    bits[23:20] = 0b1100 (fixed)
    bits[19:12] = LOFF_STATP[7:0]   (positive-input lead-off, one bit per channel)
    bits[11:4]  = LOFF_STATN[7:0]   (negative-input lead-off, one bit per channel)
    bits[3:0]   = GPIO[3:0]

A set bit means that electrode input is detected as lead-off (disconnected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["COMPARATOR_CHANNELS", "LeadoffStatus", "decode_status_word", "saturation_flags"]

# Only CH1-6 comparator bits are valid: IN8P is unwired and CH7 is not an
# electrode, so the CH7/CH8 bits are masked by firmware and hosts (K8).
COMPARATOR_CHANNELS = 6

# ADS1299 full-scale is +/- Vref/gain; treat >99% of raw 24-bit span as railed.
_RAW_FULL_SCALE = 1 << 23
_SATURATION_FRAC = 0.99


@dataclass
class LeadoffStatus:
    statp: list[bool]  # per-channel positive-input lead-off
    statn: list[bool]  # per-channel negative-input lead-off
    gpio: int

    @property
    def any_off(self) -> list[bool]:
        return [p or n for p, n in zip(self.statp, self.statn)]


def decode_status_word(status: bytes | bytearray | None) -> LeadoffStatus:
    """Decode 3 status bytes (big-endian) into per-channel lead-off flags."""
    if not status or len(status) < 3:
        return LeadoffStatus([False] * 8, [False] * 8, 0)
    b0, b1, b2 = status[0], status[1], status[2]
    statp = ((b0 & 0x0F) << 4) | (b1 >> 4)
    statn = ((b1 & 0x0F) << 4) | (b2 >> 4)
    gpio = b2 & 0x0F
    return LeadoffStatus(
        statp=[bool((statp >> i) & 1) for i in range(8)],
        statn=[bool((statn >> i) & 1) for i in range(8)],
        gpio=gpio,
    )


def saturation_flags(samples_uv_or_counts: np.ndarray, lsb_uv: Any) -> list[bool]:
    """Per-channel saturation heuristic: any sample within 1% of full scale.

    Accepts a ``(N, C)`` array of microvolts (converted back to counts via
    ``lsb_uv``: a scalar, or one LSB per column since CH7/CH8 run at a fixed
    gain 24). Returns one bool per channel.
    """
    arr = np.asarray(samples_uv_or_counts, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    lsb = np.asarray(lsb_uv, dtype=np.float64).reshape(-1)
    if lsb.size not in (1, arr.shape[1]):
        lsb = lsb[:1]
    if arr.size == 0 or lsb.size == 0 or np.any(lsb <= 0):
        return [False] * (arr.shape[1] if arr.ndim == 2 else 8)
    counts = np.abs(arr) / lsb
    limit = _RAW_FULL_SCALE * _SATURATION_FRAC
    return [bool(np.any(counts[:, c] >= limit)) for c in range(arr.shape[1])]
