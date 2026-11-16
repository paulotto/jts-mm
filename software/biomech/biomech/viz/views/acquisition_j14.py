# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Views of the acquisition-quality integration (J14): the acquisition quality per participant
(dataset, X11) and the frame-trigger association under injected faults (systems paper,
VD26). The side notes of VD27 / VD28 and the calibration sensitivity are in-text numbers
(their complete tables are the registered table views).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views.registry import ViewRequest, register

GD = {"group": "dataset"}
GV = {"group": "validation", "domain": "validation"}


def _fmt(v: Any, d: int) -> str:
    return "–" if v is None or not np.isfinite(float(v)) else f"{float(v):.{d}f}"


@register("acq_table_sessions", "paper", "Acquisition quality per participant", **GD)
def acq_table_sessions(req: ViewRequest) -> TableSpec:
    """Per participant in recording order: recording day, camera frame drops, the camera
    time base, the IMU-to-camera lag and the stereo calibration in use (X11, an20)."""
    ctx = req.ctx
    d = ctx.table("acquisition_cohort", "per_session")
    s = ctx.table("acquisition_cohort", "summary")
    tid = "acq_table_sessions"
    if d.empty:
        return TableSpec(tid, "no acquisition results", [], [])
    d = d.sort_values("recording_order")
    rows = []
    for r in d.itertuples():
        rows.append({
            "who": f"P{ctx.pseudonym(r.pid)}", "day": int(r.day_index),
            "drop0": _fmt(r.drop_rate_pct_cam0, 2), "drop1": _fmt(r.drop_rate_pct_cam1, 2),
            "burst": f"{int(max(r.burst_max_cam0, r.burst_max_cam1))}",
            "res0": _fmt(r.resid_rms_us_cam0, 1), "res1": _fmt(r.resid_rms_us_cam1, 1),
            "lag": _fmt(getattr(r, "lag_ms", np.nan), 1),
            "epi": _fmt(getattr(r, "epipolar_median_px", np.nan), 2)})
    cols = [Column("who", "Participant", align="l"), Column("day", "Day", align="r"),
            Column("drop0", "cam0", align="r", group="Drop rate (%)"),
            Column("drop1", "cam1", align="r", group="Drop rate (%)"),
            Column("burst", "Longest burst (frames)", align="r"),
            Column("res0", "cam0", align="r", group="Time base RMS (µs)"),
            Column("res1", "cam1", align="r", group="Time base RMS (µs)"),
            Column("lag", "IMU lag (ms)", align="r"),
            Column("epi", "Epipolar error (px)", align="r")]
    note = ""
    if not s.empty:
        x = s.iloc[0]
        note = (f"Recorded on {int(x.n_days)} days within {int(x.span_days)} days; pooled drop "
                f"rate {x.drop_rate_pooled_pct:.2f} % of the triggers; every drop was "
                f"resolved by the cadence step. Time base: residual of the live clock fit "
                f"(trailing 30 frames), pooled RMS {x.resid_rms_pooled_us:.1f} µs, one frame "
                f"period = 25000 µs. IMU lag: jaw IMU behind the optics (exposure centre).")
    t = TableSpec(tid, "", cols, rows, notes=note, width="text",
                  sources=["acquisition_cohort:per_session", "acquisition_cohort:summary"])
    return attach(t, CaptionSpec(
        "acquisition quality of every take in recording order: camera frame drops during "
        "regular operation (the start-up of the cameras not counted) and the longest run of "
        "dropped frames, the root mean square (RMS) residual of the camera time base, the lag "
        "of the jaw IMU's "
        "angular speed behind the optics, and the median epipolar error of the stereo "
        "calibration in use where the take stored it (from the fifth recording day on; "
        "– not stored)",
        n=f"{len(d)} participants, one take each", statistic="per take",
        evidence="descriptive"))


FAULT_TITLES = {"frame_drop": ("Extra camera drops", "Dropped frames (%)"),
                "burst": ("Drop bursts (10 per camera)", "Burst length (frames)"),
                "trigger_loss": ("Lost trigger events", "Lost events (%)"),
                "trigger_jitter": ("Trigger time jitter", "SD (ms)"),
                "sensor_jitter": ("Sensor time-stamp jitter", "SD (ms)"),
                "delivery_delay": ("Late trigger events", "Delay (ms)")}


@register("val_assoc_faults", "paper", "Frame-trigger association under injected faults", **GV)
def val_assoc_faults(req: ViewRequest) -> FigureSpec:
    """Per fault: the share of wrongly assigned and of unassigned frames over the fault
    level (mean and range over the takes and seeds; VD26)."""
    fid, title = "val_assoc_faults", "Frame-trigger association under injected faults"
    ctx = req.ctx
    lev = ctx.table("ref_assoc_cohort", "levels")
    summ = ctx.table("ref_assoc_cohort", "summary")
    if lev.empty:
        return FigureSpec(fid, title, [], caption="no fault-injection results")
    panels = []
    for fault, (ttl, xlab) in FAULT_TITLES.items():
        g = lev[lev["fault"] == fault].sort_values("level")
        if g.empty:
            continue
        x = g["level"].to_numpy(float)
        top = float(np.nanmax(np.r_[g["wrong_pct_max"], g["unassigned_pct_mean"], 1.0]))
        p = Panel(ttl, Axis(xlab, ""), Axis("Frames (%)", "", lim=(0.0, 1.08 * top)), [],
                  legend="below")
        p.series.append(Series("band", x, g["wrong_pct_mean"].to_numpy(float) * 0,
                               y2=g["wrong_pct_max"].to_numpy(float), role="quaternary",
                               style={"alpha": 0.15}))
        p.series.append(Series("line", x, g["wrong_pct_mean"].to_numpy(float),
                               role="quaternary", label="wrong trigger (mean; band: largest)"))
        p.series.append(Series("scatter", x, g["wrong_pct_mean"].to_numpy(float),
                               role="quaternary", style={"ms": "o"}))
        p.series.append(Series("line", x, g["unassigned_pct_mean"].to_numpy(float),
                               role="muted", label="unassigned (mean)"))
        if fault == "burst":
            p.series.append(Series("vline", x=np.array([30.0]), role="threshold"))
        panels.append(p)
    n = ""
    if not summ.empty:
        r = summ.iloc[0]
        n = (f"{int(r.n_recordings)} takes (comparison and dataset take of both participants), "
             f"{int(r.seeds)} seeds per level")
    from biomech.viz.views.validation import shared_legend

    shared_legend(panels)                  # one legend for the six panels
    spec = FigureSpec(fid, title, panels, rows=2, cols=3, width="text", height_in=1.5,
                      sources=["ref_assoc_cohort:levels", "ref_assoc_cohort:summary"])
    return attach(spec, CaptionSpec(
        "the recorded camera and trigger streams replayed through the recorder's frame-trigger "
        "association with injected faults; one panel per fault: share of the delivered frames "
        "assigned to a wrong trigger and left unassigned, over the fault level",
        [("vermillion line and band", "wrong trigger: mean and largest share over the runs"),
         ("grey line", "unassigned frames (mean); a lost trigger event leaves its frame "
                       "unassigned by design"),
         ("dashed line", "30 trigger periods, the largest gap the cadence step bridges")],
        n=n, statistic="mean and maximum over takes x seeds", evidence="descriptive"))
