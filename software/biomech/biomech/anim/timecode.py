# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Hidden, machine-readable timecode in every video frame (an11 §5).

A strip of 56 square blocks in the frame's bottom-right corner, ``b`` pixels each
(``b`` = 4 px up to 1080 rows, 8 px at 2160), encodes

* 8 sync bits ``10110010``,
* the frame's master-clock time in microseconds (40 bits, MSB first),
* an 8-bit checksum (sum of the five time bytes mod 256).

Bit 0 is the background colour; bit 1 differs from it by 80 grey levels (lighter
on a dark background, darker on a light one), so the strip is a faint line of
dots. Blocks are even-sized and on even coordinates, so H.264 4:2:0 at CRF 18
preserves them; :func:`read_timecode` takes the mean of each block's inner
pixels and thresholds between the sync bits' two levels, so it is independent
of the theme and of the polarity. No OCR is involved.
"""

from __future__ import annotations

import numpy as np

SYNC = (1, 0, 1, 1, 0, 0, 1, 0)
N_TIME = 40
N_BITS = len(SYNC) + N_TIME + 8
DELTA = 80


def block(height: int) -> int:
    """Block size (px) for a frame ``height`` pixels tall."""
    return max(4, int(round(4 * height / 1080 / 2)) * 2)


def _bits(t_us: int) -> list[int]:
    t = int(t_us) & ((1 << N_TIME) - 1)
    tb = [(t >> (N_TIME - 1 - k)) & 1 for k in range(N_TIME)]
    chk = sum((t >> (8 * j)) & 0xFF for j in range(5)) & 0xFF
    return list(SYNC) + tb + [(chk >> (7 - k)) & 1 for k in range(8)]


def stamp(frame: np.ndarray, t_us: float, bg: np.ndarray | None = None) -> np.ndarray:
    """Write the timecode of ``t_us`` into ``frame`` (H, W, 3|4 uint8) in place; ``bg``
    (RGB 0..255) is bit 0 (default: the pixel left of the strip)."""
    H, W = frame.shape[:2]
    b = block(H)
    x0, y0 = W - N_BITS * b, H - b
    if x0 < 0:
        raise ValueError(f"frame too narrow for the timecode ({W} px < {N_BITS * b})")
    base = (np.asarray(bg, float) if bg is not None
            else frame[y0:H, max(x0 - 1, 0), :3].astype(float).mean(axis=0))
    lum = base.mean()
    one = np.clip(base + (DELTA if lum < 128 else -DELTA), 0, 255)
    for k, bit in enumerate(_bits(int(round(t_us)))):
        frame[y0:H, x0 + k * b:x0 + (k + 1) * b, :3] = one if bit else base
    return frame


def read_timecode(frame_rgb: np.ndarray) -> int | None:
    """The master-clock time (us) stamped into a decoded frame, or ``None`` if the strip
    is missing or corrupt (sync or checksum mismatch)."""
    f = np.asarray(frame_rgb)
    H, W = f.shape[:2]
    b = block(H)
    x0, y0 = W - N_BITS * b, H - b
    if x0 < 0:
        return None
    m = max(1, b // 4)
    strip = f[y0 + m:H - m, x0:W, :3].astype(float).mean(axis=(0, 2))       # (N_BITS * b,)
    vals = np.array([strip[k * b + m:(k + 1) * b - m].mean() for k in range(N_BITS)])
    s = np.array(SYNC, bool)
    hi, lo = vals[:8][s].mean(), vals[:8][~s].mean()
    if abs(hi - lo) < DELTA / 3:
        return None
    bits = ((vals - lo) / (hi - lo) > 0.5).astype(int)
    if tuple(bits[:8]) != SYNC:
        return None
    t = 0
    for v in bits[8:8 + N_TIME]:
        t = (t << 1) | int(v)
    chk = 0
    for v in bits[8 + N_TIME:]:
        chk = (chk << 1) | int(v)
    if chk != sum((t >> (8 * j)) & 0xFF for j in range(5)) & 0xFF:
        return None
    return t
