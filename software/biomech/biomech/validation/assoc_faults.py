# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Robustness of the frame-trigger association under injected faults (VD26, J14;
ported from and checked against the earlier evaluations 5.3.2 of the recording
system, an14 §11e).

The recordings never exercise most association paths (every drop was a camera drop that
the cadence step resolved; an20). To show where the association breaks, a recording's raw
streams are replayed through the recorder's own ``jtsmm.sync.clock_align.ClockAligner``
(unchanged since the recordings) with controlled faults.

**Replay.** Events in host arrival order: a TRIGGER_EVENT ``trigger_usb_ms`` after its edge
(plus an injected delivery delay), a frame ``frame_latency_ms`` after its trigger. The first
frame of each camera is anchored to its recorded trigger (the start-up anchoring is not under
test; its trigger is never lost); every other frame goes through the sequence association
(``trigger_id=None``). A frame the aligner holds (its trigger not delivered yet) is retried
after every new TRIGGER_EVENT, in order, as the supervisor does. Ground truth is the trigger
id recorded for each frame (the replay without faults reproduces it for every frame, which is
checked and reported). Each delivered frame ends *correct*, *wrong* (another id) or
*unassigned* (discarded by the aligner, or still held at the end).

**Faults** (each at several levels, ``seeds`` runs per level and recording): extra single
camera drops (% of frames), ten bursts of L consecutive dropped frames per camera, lost
TRIGGER_EVENTs (%), Gaussian jitter on the trigger master time and on the SensorTimestamp
(sigma, ms), and late TRIGGER_EVENTs (fixed delay plus an exponential share of mean
delay/2, order kept). Plus the ablation without the cadence step (``delta_tracking`` off).

**Tolerance limit** per fault: the largest level with no wrong assignment in any run (and
the first level with one); unassigned frames are reported separately (a lost
TRIGGER_EVENT leaves its frame unassigned by design, never with a neighbour's time).
"""

from __future__ import annotations

import hashlib
import heapq
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "trigger_usb_ms": 2.0,
    "frame_latency_ms": 40.0,
    "seeds": 10,
    "faults": {
        "frame_drop": [0, 5, 10, 20, 35, 50],
        "burst": [1, 5, 10, 20, 29, 30, 40, 60],
        "trigger_loss": [0, 1, 2, 5, 10],
        "trigger_jitter": [0, 0.5, 1, 2, 4],
        "sensor_jitter": [0, 1, 2, 4, 6],
        "delivery_delay": [0, 50, 100, 250, 500],
    },
    "n_bursts": 10,
}

UNITS = {"frame_drop": "% of frames", "burst": "frames per burst",
         "trigger_loss": "% of TRIGGER_EVENTs", "trigger_jitter": "sigma (ms)",
         "sensor_jitter": "sigma (ms)", "delivery_delay": "delay (ms)"}


def _seed(*parts: Any) -> int:
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def build_streams(cams: dict[str, Any], trig_id: np.ndarray, trig_t: np.ndarray, fault: str,
                  level: float, rng: np.random.Generator, p: dict[str, Any]
                  ) -> tuple[list[tuple[float, int, int]], dict[str, tuple[np.ndarray, ...]]]:
    """Trigger arrivals [(t_arrive, id, t_master)] and per camera (sensor_ns, true_id,
    t_arrive) with the fault applied."""
    tid, tt = trig_id.copy(), trig_t.astype(np.float64).copy()
    keep_t = np.ones(tid.size, bool)
    anchors = {int(c.trigger_id[c.trigger_id > 0][0]) for c in cams.values()}
    if fault == "trigger_loss" and level > 0:
        keep_t = rng.random(tid.size) >= level / 100.0
        keep_t |= np.isin(tid, list(anchors))
    if fault == "trigger_jitter" and level > 0:
        tt = tt + rng.normal(0.0, level * 1e3, tt.size)
    arrive = trig_t.astype(np.float64) + p["trigger_usb_ms"] * 1e3
    if fault == "delivery_delay" and level > 0:
        arrive = arrive + level * 1e3 + rng.exponential(level * 1e3 / 2.0, arrive.size)
        arrive = np.maximum.accumulate(arrive)                      # USB keeps the order
    trigs = [(float(a), int(i), int(round(t))) for a, i, t, k in zip(arrive, tid, tt, keep_t)
             if k]
    out = {}
    for c, st in cams.items():
        ok = st.trigger_id > 0
        sens = st.sensor_ns[ok].astype(np.float64)
        true = st.trigger_id[ok].copy()
        tm = st.t_master_us[ok].astype(np.float64)
        n = sens.size
        keep = np.ones(n, bool)
        if fault == "frame_drop" and level > 0:
            keep[1:] = rng.random(n - 1) >= level / 100.0
        if fault == "burst":
            L = int(level)
            starts = (np.linspace(n * 0.05, n * 0.95, int(p["n_bursts"])).astype(int)
                      + rng.integers(0, 50, int(p["n_bursts"])))
            for a in starts:
                keep[a:a + L] = False
            keep[0] = True
        if fault == "sensor_jitter" and level > 0:
            sens = sens + rng.normal(0.0, level * 1e6, n)
        out[c] = (np.round(sens[keep]).astype(np.int64), true[keep],
                  tm[keep] + p["frame_latency_ms"] * 1e3)
    return trigs, out


def replay(cams: dict[str, Any], trig_id: np.ndarray, trig_t: np.ndarray, fault: str,
           level: float, seed: int, p: dict[str, Any], *, delta_tracking: bool = True
           ) -> dict[str, Any]:
    """One replay; counts of correct / wrong / unassigned frames per camera."""
    from jtsmm.sync.clock_align import ClockAligner

    rng = np.random.default_rng(_seed(fault, level, seed))
    trigs, streams = build_streams(cams, trig_id, trig_t, fault, level, rng, p)
    al = ClockAligner()
    al.delta_tracking = delta_tracking
    al.expected_cameras = tuple(streams)
    ev = [(a, 0, "t", (i, t)) for a, i, t in trigs]
    for ci, (c, (sens, _, arr)) in enumerate(streams.items()):
        ev += [(float(arr[k]), 1 + ci, c, k) for k in range(sens.size)]
    heapq.heapify(ev)
    res = {c: np.full(streams[c][0].size, -2, np.int64) for c in streams}
    pending: dict[str, list[int]] = {c: [] for c in streams}
    first = {c: True for c in streams}

    def attempt(c: str, k: int) -> bool:
        sens, true, _ = streams[c]
        if first[c]:
            if al._lookup_trigger_exact(int(true[k])) is None:
                return False
            r = al.ingest_cam(c, int(sens[k]), int(true[k]), frame_idx=k)
            first[c] = False
        else:
            r = al.ingest_cam(c, int(sens[k]), None, frame_idx=k)
        if r is None:
            return not al.last_ingest_waiting
        res[c][k] = int(r.payload["trigger_id"])
        return True

    def flush(c: str) -> None:
        while pending[c]:
            if not attempt(c, pending[c][0]):
                return
            pending[c].pop(0)

    while ev:
        _, _, kind, pl = heapq.heappop(ev)
        if kind == "t":
            al.note_trigger(pl[0], pl[1])
            for c in streams:
                flush(c)
        elif pending[kind] or not attempt(kind, pl):
            pending[kind].append(pl)
    out = {"fault": fault, "level": float(level), "seed": int(seed),
           "delta_tracking": bool(delta_tracking)}
    for c, (sens, true, _) in streams.items():
        r = res[c]
        n = int(r.size)
        correct = int(np.sum(r == true))
        un = int(np.sum(r == -2))
        out[f"{c}_n"] = n
        out[f"{c}_correct"] = correct
        out[f"{c}_wrong"] = n - correct - un
        out[f"{c}_unassigned"] = un
    return out


def run_recording(cams: dict[str, Any], trig_id: np.ndarray, trig_t: np.ndarray,
                  p: dict[str, Any] | None = None) -> pd.DataFrame:
    """Every fault, level and seed on one recording, plus the clean replay and the ablation."""
    p = {**DEFAULTS, **(p or {})}
    rows = [replay(cams, trig_id, trig_t, "clean", 0, 0, p),
            replay(cams, trig_id, trig_t, "cadence_off", 0, 0, p, delta_tracking=False)]
    for fault, levels in p["faults"].items():
        for lv in levels:
            for s in range(int(p["seeds"])):
                rows.append(replay(cams, trig_id, trig_t, fault, lv, s, p))
    df = pd.DataFrame(rows)
    cs = sorted({c.split("_")[0] for c in df.columns if c.endswith("_n")})
    df["n"] = df[[f"{c}_n" for c in cs]].sum(axis=1)
    for k in ("correct", "wrong", "unassigned"):
        df[k] = df[[f"{c}_{k}" for c in cs]].sum(axis=1)
        df[f"{k}_pct"] = 100.0 * df[k] / df["n"].clip(lower=1)
    return df


def summarise(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per fault and level (mean, min, max over recordings x seeds) and the tolerance limit
    per fault (largest level with no wrong frame in any run)."""
    f = runs[~runs["fault"].isin(["clean", "cadence_off"])]
    lev = (f.groupby(["fault", "level"])
           .agg(n_runs=("wrong", "size"), wrong_pct_mean=("wrong_pct", "mean"),
                wrong_pct_max=("wrong_pct", "max"), unassigned_pct_mean=("unassigned_pct", "mean"),
                unassigned_pct_max=("unassigned_pct", "max"),
                runs_with_wrong=("wrong", lambda s: int((s > 0).sum())))
           .reset_index())
    lim = []
    for fault, g in lev.groupby("fault"):
        g = g.sort_values("level")
        bad = g[g["runs_with_wrong"] > 0]
        first_bad = float(bad["level"].min()) if len(bad) else np.nan
        ok = g[g["level"] < first_bad] if len(bad) else g
        lim.append({"fault": fault, "unit": UNITS.get(fault, ""),
                    "tolerated_level": float(ok["level"].max()) if len(ok) else np.nan,
                    "first_failing_level": first_bad,
                    "max_level_tested": float(g["level"].max()),
                    "wrong_pct_mean_at_first_failing": float(bad.iloc[0]["wrong_pct_mean"])
                    if len(bad) else np.nan,
                    "wrong_pct_max_at_first_failing": float(bad.iloc[0]["wrong_pct_max"])
                    if len(bad) else np.nan})
    return lev, pd.DataFrame(lim)


def load_recording(h5: str | Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    from biomech.qc.acquisition import read_streams

    cams, tid, tt, _ = read_streams(h5)
    return cams, tid, tt
