# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tables derived from stored results for the in-text numbers (2026-11-05).

A numbers entry with ``source: derived:<name>`` reads one of these tables instead of a stage
table. Each is computed from what the (frozen) store already holds, with the same helpers the
figures use, so a figure and its macro cannot disagree and no stage has to be re-run for a
value the analysis already determines:

* ``coupling_curve`` -- per opening angle of the cohort grid: the median anterior translation of
  the condyle-point midpoint over the participants (``stats_kinematics`` cohort arrays), its
  pointwise 95 % percentile-bootstrap CI (the band of the hinge figure, the study seed) and n;
* ``workspace_box`` -- per participant: the share of the incisal positions of all tasks outside
  the P95 design box, and the share outside only because behind it (``suppWorkspace`` (b),
  ``requirements`` (a, b)).
* ``condyle_path`` -- per participant: the open-close displacement of the condyle-point midpoint
  from the closed pose (opening < 1 deg) to the widest openings (top 10 % of the opening angle),
  maxillary frame; its inclination below the horizontal; the direction of the best-fit axis
  from the condyle points (``kinematics:geometry``), and the direction kinematics predicts for
  it: a rotation plus a translation of the condyles along a path inclined alpha is one rotation
  about a point behind-below the condyles at 90 deg - alpha below the horizontal.

The functions live under ``viz/`` (presentation code, not hashed with the analysis): the store
fixes the inputs, the code here only summarises them.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

TABLES: dict[str, Callable[[Any], pd.DataFrame]] = {}


def _table(name: str) -> Callable:
    def deco(fn: Callable[[Any], pd.DataFrame]) -> Callable[[Any], pd.DataFrame]:
        TABLES[name] = fn
        return fn
    return deco


def table(ctx: Any, name: str) -> pd.DataFrame:
    """The derived table ``name`` for the store ``ctx`` reads (tables registered here or in
    the per-paper modules ``derived_dataset`` / ``derived_systems``)."""
    import importlib

    for mod in ("derived_dataset", "derived_systems"):
        try:
            importlib.import_module(f"biomech.viz.{mod}")
        except ModuleNotFoundError as exc:
            if exc.name != f"biomech.viz.{mod}":
                raise
    if name not in TABLES:
        raise ValueError(f"derived:{name}: unknown (known: {sorted(TABLES)})")
    return TABLES[name](ctx)


@_table("coupling_curve")
def coupling_curve(ctx: Any) -> pd.DataFrame:
    from biomech.viz.views import blocks as B
    from biomech.viz.views.kin_r8 import _coupling_curves

    g, S, _, rows = _coupling_curves(ctx)
    if g is None or S is None or not len(rows):
        return pd.DataFrame()
    S = np.atleast_2d(np.asarray(S, float))
    lo, hi = B.boot_curve_ci(S, centre="median", seed=B.seed_of(ctx))   # as the figure band
    n = np.isfinite(S).sum(axis=0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(np.where(np.isfinite(S), S, np.nan), axis=0)
    return pd.DataFrame({"angle_deg": np.asarray(g, float), "cond_trans_mm": med,
                         "ci_lo": lo, "ci_hi": hi, "n": n.astype(int)})


@_table("workspace_box")
def workspace_box(ctx: Any) -> pd.DataFrame:
    from biomech.viz.views.kin_r8 import (_arr, _behind, _outside, _requirements, design_box,
                                          incisor_cloud)

    box = design_box(_requirements(ctx))
    out = []
    for pid in ctx.pids():
        P, _ = incisor_cloud(_arr(ctx, "kinematics", pid))
        if len(P) < 5:                          # as kin_workspace3d: too few positions
            continue
        # J31.1: positions above the rest position (the box's upper face)
        out.append({"pid": pid, "ws_outside_pct": _outside(P, box),
                    "ws_behind_pct": _behind(P, box),
                    "ws_above_pct": float(100.0 * np.mean(P[:, 2] > 1e-9)),
                    "above_p99_mm": float(max(np.percentile(P[:, 2], 99), 0.0)),
                    "n_points": int(len(P))})
    return pd.DataFrame(out)


@_table("coupling_sign")
def coupling_sign(ctx: Any) -> pd.DataFrame:
    """J31.3: participants with a coupling slope (X9), those with a negative one (the
    palpated condyle midpoint moves backward with opening) and those of them whose condyle
    points pass the plausibility checks."""
    cs = ctx.table("stats_kinematics", "coupling_summary")
    s = pd.to_numeric(cs["slope_mm_per_deg"], errors="coerce")
    neg = s < 0
    return pd.DataFrame([{"n": int(s.notna().sum()), "n_negative": int(neg.sum()),
                          "n_negative_plausible": int((neg & cs["condyles_plausible"]
                                                       .astype(bool)).sum())}])


@_table("workspace_fill")
def workspace_fill(ctx: Any) -> pd.DataFrame:
    """J31.2: per plane the share (%) of the P95 design box that the cohort envelope
    (union of the participants' convex outlines, requirements a, b) covers, on the 0.2-mm
    raster of the envelope outlines."""
    from matplotlib.path import Path as MPath

    from biomech.viz.views.paper_journal import PLANES, envelope_outlines

    env = envelope_outlines(ctx)
    box, res, rows = env["box"], 0.2, []
    if box is None:
        return pd.DataFrame(columns=["plane", "fill_pct", "box_area_mm2", "n"])
    lo, hi = box
    for plane, (i, j), _, _ in PLANES:
        xs = np.arange(lo[i] + res / 2, hi[i], res)
        ys = np.arange(lo[j] + res / 2, hi[j], res)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        inside = np.zeros(len(pts), bool)
        for d in env["per"].values():
            if plane in d:
                inside |= MPath(d[plane]).contains_points(pts, radius=1e-9)
        rows.append({"plane": plane, "fill_pct": float(100.0 * inside.mean()),
                     "box_area_mm2": float((hi[i] - lo[i]) * (hi[j] - lo[j])),
                     "n": int(env["n"])})
    return pd.DataFrame(rows)


@_table("condyle_path")
def condyle_path(ctx: Any) -> pd.DataFrame:
    geo = ctx.table("kinematics", "geometry")
    cs = ctx.table("stats_kinematics", "coupling_summary")
    plaus = dict(zip(cs["pid"], cs["condyles_plausible"])) \
        if not cs.empty and "condyles_plausible" in cs else {}
    out = []
    for pid in ctx.pids():
        a = ctx.arrays("kinematics", pid)
        if "open_close__theta_deg" not in a or "open_close__tmj_left" not in a:
            continue
        th = np.asarray(a["open_close__theta_deg"], float)
        C = 0.5 * (np.asarray(a["open_close__tmj_left"], float)
                   + np.asarray(a["open_close__tmj_right"], float))
        ok = np.isfinite(th) & np.isfinite(C).all(axis=1)
        if ok.sum() < 10:
            continue
        closed, wide = ok & (th < 1.0), ok & (th > np.percentile(th[ok], 90))
        if not closed.any() or not wide.any():
            continue
        d = np.median(C[wide], axis=0) - np.median(C[closed], axis=0)
        alpha = float(np.degrees(np.arctan2(-d[2], d[0])))       # below the horizontal
        g = geo[geo["pid"] == pid] if not geo.empty else geo
        dx = float(g["hinge_axis_dx_mm"].iloc[0]) if len(g) else np.nan
        dz = float(g["hinge_axis_dz_mm"].iloc[0]) if len(g) else np.nan
        out.append({"pid": pid, "condyles_plausible": bool(plaus.get(pid, False)),
                    "cond_path_dx_mm": float(d[0]), "cond_path_dz_mm": float(d[2]),
                    "cond_path_incl_deg": alpha,
                    "axis_dir_deg": float(np.degrees(np.arctan2(-dz, -dx))),
                    "axis_dir_pred_deg": 90.0 - alpha})
    return pd.DataFrame(out)
