# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frame-period and gap helpers shared by the offline tools.

Dropped camera frames are *absent rows* in ``/optical`` and ``/cameras`` (no
NaN placeholder), so row index is not time. Everything that needs time from
row data goes through here:

- :func:`session_frame_period_us` -- trigger period of a session
  (``/triggers`` → ``/triggers@rate_hz`` → ``meta.mcu_config_json``).
- :func:`frame_grid_index` -- place rows on the trigger grid
  ``k = round((t - t0) / period)`` with ``t0`` the first row.
  ``jts.aruco_camera`` implements the same rule for JTS (it cannot import
  jtsmm); ``tests/test_fix_I_timegrid.py`` keeps the two in step.
- :func:`break_gaps` -- NaN separators so plot lines do not bridge gaps.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def estimate_period_us(t_us: Any) -> float | None:
    """Frame period from row timestamps that may have missing rows.

    Median step first (robust to drops), then refined to span / number of
    periods so the long-run period is exact enough to index an hour of frames.
    """
    t = np.asarray(t_us, dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return None
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        return None
    p0 = float(np.median(d))
    k = np.round(d / p0)
    ok = k >= 1
    if not np.any(ok):
        return p0
    return float(d[ok].sum() / k[ok].sum())


def trigger_period_us(
    t_us: Any, trigger_id: Any = None, rate_hz: float | None = None
) -> float | None:
    """Period of the MCU trigger train (µs), tolerant of lost trigger events."""
    t = np.asarray(t_us, dtype=np.float64) if t_us is not None else np.empty(0)
    if trigger_id is not None and t.size >= 2:
        ids = np.asarray(trigger_id, dtype=np.float64)
        if ids.shape == t.shape and ids[-1] > ids[0]:
            return float((t[-1] - t[0]) / (ids[-1] - ids[0]))
    if rate_hz:
        return 1e6 / float(rate_hz)
    return estimate_period_us(t)


def session_frame_period_us(f: Any) -> float | None:
    """Trigger period of an open ``session.h5``; ``None`` if nothing says."""
    if "triggers" in f:
        g = f["triggers"]
        rate = g.attrs.get("rate_hz")
        t = g["t_master_us"][()] if "t_master_us" in g else None
        ids = g["trigger_id"][()] if "trigger_id" in g else None
        if t is not None and len(t) >= 2:
            p = trigger_period_us(t, ids, None)
            if p:
                return p
        if rate:
            return 1e6 / float(rate)
    if "meta" in f and "mcu_config_json" in f["meta"].attrs:
        raw = f["meta"].attrs["mcu_config_json"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            rate = (json.loads(str(raw)).get("trigger") or {}).get("rate_hz")
        except (ValueError, AttributeError):
            rate = None
        if rate:
            return 1e6 / float(rate)
    return None


def frame_grid_index(t_us: Any, period_us: float, t0_us: float | None = None) -> np.ndarray:
    """Trigger-grid index of each row: ``round((t - t0) / period)``."""
    t = np.asarray(t_us, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    t0 = float(t[0]) if t0_us is None else float(t0_us)
    return np.round((t - t0) / float(period_us)).astype(np.int64)


def break_gaps(
    t: Any, *ys: Any, period: float | None = None, factor: float = 1.5
) -> tuple[np.ndarray, ...]:
    """Insert a NaN row wherever ``diff(t) > factor * period``.

    ``matplotlib`` and Plotly (``connectgaps=False``) then draw a break instead
    of a straight line across the hole. ``period`` defaults to the median step.
    Returns ``(t, *ys)`` with the separators inserted.
    """
    t = np.asarray(t, dtype=np.float64)
    arrs = [np.asarray(y, dtype=np.float64) for y in ys]
    if t.size < 2:
        return (t, *arrs)
    d = np.diff(t)
    if period is None:
        pos = d[d > 0]
        if pos.size == 0:
            return (t, *arrs)
        period = float(np.median(pos))
    cut = np.flatnonzero(d > factor * float(period)) + 1
    if cut.size == 0:
        return (t, *arrs)
    t_out = np.insert(t, cut, np.nan)
    out = []
    for y in arrs:
        fill = np.full((cut.size,) + y.shape[1:], np.nan)
        out.append(np.insert(y, cut, fill, axis=0))
    return (t_out, *out)
