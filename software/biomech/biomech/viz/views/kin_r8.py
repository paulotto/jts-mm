# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.4 (3-D and kinematics views) and the dataset part of J2.5 (claims coverage).

J2.4, 3-D:

* ``kin_workspace3d``: the Posselt-type incisal workspace in 3-D -- the incisal positions of
  every task as displacement from the rest position, the convex hull (a cage of sections) and the
  design workspace box from the stored P95 requirements; forms ``overlay`` (every
  participant's hull, the chosen one highlighted), ``multiples`` (one panel per participant)
  and ``3d`` (one participant with the point cloud coloured by task);
* ``kin_condyle3d``: condyle-point paths of opening and the finite helical axes as cohort small
  multiples (``multiples``: 3-D; ``2d``: sagittal plane); participants whose palpated condyles
  fail the plausibility check are drawn muted (flagged, not excluded);
* ``kin_coupling_surface``: anterior condyle-midpoint translation over opening angle x
  participant (``3d``: surface) or as curves with the cohort (generic) median curve and the
  stored per-participant slope and generic-curve miss (``curves``).

J2.5, dataset gaps (one figure per claim or argument that had none):

* ``cov_outcome_n``: which participant enters which outcome (flag, not exclude);
* ``req_lowpass_bound``: peak opening velocity under the low-pass sensitivity runs (lower bound);
* ``req_envelope``: closing / free-motion speed, peak angular acceleration with their P95 design
  values, and the coverage of the sample maximum against n;
* ``kin_null_variants``: the hinge residual against every noise null, and the condyle
  plausibility sensitivity of the condyle-based outcomes;
* ``kin_geometry_lit``: jaw geometry per participant against literature ranges;
* ``timing_window_null``: the observed EMG lead against the window-only chance null;
* ``timing_single_cycle``: single-cycle usability of the onsets and the minimum-cycle gate;
* ``emg_spm_sigbar``: the confirmatory SPM comparisons as mean ± SD bands with significance bars;
* ``var_cmc_cycles``: waveform similarity within vs between participants and cycles needed.

Tables that only a later re-run stores (the closing / free-motion / acceleration subject columns,
``requirements:max_coverage``, ``stats_kinematics:coupling_summary``) are read when present; a
result set without them gets an annotated empty panel, never an exception.

Literature ranges (``kin_geometry_lit``) come from ``etc/literature.yaml`` when it exists
(unhashed, so no re-run is needed); without it only the cohort distribution is shown. Format::

    geometry:
      bonwill_side_mm:
        - {value: 101.6, label: "Bonwill equilateral triangle", ref: "R1"}
      balkwill_deg:
        - {low: 18.0, high: 26.0, label: "condylar centre to top", ref: "R3-R5"}

(``value`` draws a reference line, ``low`` / ``high`` a shaded range; ``ref`` is the reference
tag shown in the panel note.) A flat file without the ``geometry:`` key is read the same way.
"""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd

from biomech import catalog
from biomech.results import COHORT_UNIT
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series, TableSpec
from biomech.viz import style
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.citations import cite_label, tidy
from biomech.viz.views.registry import ViewRequest, register

MM, DEG, DPS, PCT, MSEC, HZ = V.MM, V.DEG, V.DPS, V.PCT, V.MSEC, V.HZ
TASKS = V.TASKS
LITERATURE_PATH = catalog.CATALOG_PATH.parent / "literature.yaml"
RULES = (("first", "pre-specified"), ("drive", "post hoc"))
RULE = V.labels("rule")
TIMING_ROWS = (("open_close", "opening"), ("chewing", "opening"))
DETECTOR = "tkeo"                                  # the confirmatory onset detector
HULL_RINGS = 5                                     # drawing: horizontal hull sections
MAX_POINTS_3D = 1500                               # drawing: thinned point clouds
NOT_STORED = "not in this result set"
NEXT_RUN = "not stored in this result set; the next re-run stores it"


# -- helpers ------------------------------------------------------------------------------------
def _t(ctx: Any, stage: str, name: str, pids: list[str] | None = None) -> pd.DataFrame:
    """A stored table; empty when the stage or table is absent (older result sets)."""
    try:
        if stage not in ctx.store.stages():
            return pd.DataFrame()
        return ctx.table(stage, name, pids) if pids is not None else ctx.table(stage, name)
    except (KeyError, FileNotFoundError, ValueError):
        return pd.DataFrame()


def _arr(ctx: Any, stage: str, unit: str) -> dict:
    try:
        return ctx.arrays(stage, unit) if stage in ctx.store.stages() else {}
    except (KeyError, FileNotFoundError, ValueError):
        return {}


def _f(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _empty(p: Panel, msg: str = "no data") -> Panel:
    p.note = msg
    return p


def _fin(v: Any) -> np.ndarray:
    a = np.asarray(v, float).ravel()
    return a[np.isfinite(a)]


def _who(ctx: Any, pid: str) -> str:
    return f"P{ctx.pseudonym(pid)}"


def _num(v: float, d: int = 0) -> str:
    return "--" if not np.isfinite(v) else f"{v:.{d}f}"


def _pid(req: ViewRequest) -> str | None:
    pids = req.ctx.pids()
    if req.pid in pids:
        return req.pid
    try:
        from biomech.viz.views.paper_icorr import representative

        return representative(req.ctx)
    except Exception:
        return pids[0] if pids else None


def _subjects(ctx: Any) -> pd.DataFrame:
    s = _t(ctx, "stats_kinematics", "subjects")
    if s.empty or "pid" not in s:
        return pd.DataFrame(index=pd.Index([], name="pid"))
    return s.drop_duplicates("pid").set_index("pid")


def _col(s: pd.DataFrame, col: str, pids: Sequence[str]) -> np.ndarray:
    if s.empty or col not in s:
        return np.full(len(pids), np.nan)
    return pd.to_numeric(s[col].reindex(list(pids)), errors="coerce").to_numpy(float)


def plausible(ctx: Any) -> dict[str, bool | None]:
    """Per included participant: palpated condyles plausible (True), implausible (False),
    unknown (None) -- the stored flag (``stats_kinematics:subjects.condyles_plausible``)."""
    s = _subjects(ctx)
    out: dict[str, bool | None] = {}
    for pid in ctx.pids():
        v = _f(s.at[pid, "condyles_plausible"]) if pid in s.index and \
            "condyles_plausible" in s else np.nan
        out[pid] = None if not np.isfinite(v) else bool(v >= 1)
    return out


def _requirements(ctx: Any) -> pd.DataFrame:
    rq = _t(ctx, "requirements", "requirements")
    return rq.drop_duplicates("key").set_index("key") if not rq.empty and "key" in rq \
        else pd.DataFrame()


def _req(rq: pd.DataFrame, key: str, col: str = "value") -> float:
    return _f(rq.at[key, col]) if not rq.empty and key in rq.index and col in rq else np.nan


def _p_design() -> float:
    from biomech.stats.requirements import P_DESIGN

    return float(P_DESIGN)


def _nan_join(segments: list[np.ndarray]) -> np.ndarray:
    """(k, 3) pieces joined by NaN rows into one polyline (one series, line breaks)."""
    out = []
    for s in segments:
        out += [np.asarray(s, float), np.full((1, 3), np.nan)]
    return np.vstack(out) if out else np.empty((0, 3))


def _hull(P: np.ndarray):
    from scipy.spatial import ConvexHull, QhullError

    P = np.asarray(P, float)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 5:
        return P, None
    try:
        return P, ConvexHull(P)
    except (QhullError, ValueError):
        return P, None


def _section(P: np.ndarray, simplices: np.ndarray, axis: int, level: float
             ) -> np.ndarray | None:
    """Closed outline (k, 3) of the convex hull cut by the plane ``x[axis] = level``."""
    from scipy.spatial import ConvexHull, QhullError

    pts = []
    for tri in simplices:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = P[tri[a]], P[tri[b]]
            da, db = pa[axis] - level, pb[axis] - level
            if da * db < 0:
                pts.append(pa + da / (da - db) * (pb - pa))
    if len(pts) < 3:
        return None
    Q = np.array(pts)
    other = [i for i in range(3) if i != axis]
    try:
        h = ConvexHull(Q[:, other])
    except (QhullError, ValueError):
        return None
    ring = Q[h.vertices]
    return np.vstack([ring, ring[:1]])


def hull_cage(P: np.ndarray, rings: int = HULL_RINGS) -> np.ndarray | None:
    """The 3-D convex hull of the points ``P`` (N, 3) drawn as a cage: the sagittal and the
    frontal section through its centroid and ``rings`` horizontal sections between its lowest
    and highest point, as one NaN-separated polyline; None for fewer than 5 points or a flat
    cloud."""
    P, h = _hull(P)
    if h is None:
        return None
    c = P[h.vertices].mean(axis=0)
    cuts = [(1, c[1]), (0, c[0])]
    z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
    cuts += [(2, float(z)) for z in np.linspace(z0, z1, rings + 2)[1:-1]]
    segs = [r for r in (_section(P, h.simplices, ax, lv) for ax, lv in cuts) if r is not None]
    return _nan_join(segs) if segs else None


def box_edges(lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """The 12 edges of the axis-aligned box lo..hi as one NaN-separated polyline."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                  for z in (lo[2], hi[2])])
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if int(np.sum(np.abs(c[i] - c[j]) > 0)) == 1]
    return _nan_join([c[[i, j]] for i, j in edges])


def _thin(n: int, m: int) -> np.ndarray:
    return np.arange(n) if n <= m else np.unique(np.linspace(0, n - 1, m).astype(int))


def _l3(P: np.ndarray, role: str, label: str | None = None, **style: Any) -> Series:
    return Series("line3d", P[:, 0], P[:, 1], z=P[:, 2], role=role, label=label, style=style)


def _caxes() -> tuple[Axis, Axis, Axis]:
    return Axis("Anterior", MM), Axis("Left", MM), Axis("Superior", MM)


def _set_lims(panels: list[Panel], pts: list[np.ndarray], pad: float = 0.06) -> None:
    """One common box for 3-D small multiples (the same mm scale in every panel)."""
    P = np.vstack([p for p in pts if p is not None and len(p)]) if pts else np.empty((0, 3))
    P = P[np.isfinite(P).all(axis=1)] if len(P) else P
    if not len(P):
        return
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = np.maximum(hi - lo, 1.0)
    for p in panels:
        for ax, a, b, s in zip((p.x, p.y, p.z), lo, hi, span):
            ax.lim = (float(a - pad * s), float(b + pad * s))


def _multiples_axes(panels: list[Panel]) -> None:
    """3-D / 2-D small multiples: axis labels only on the first panel (the viewer restores
    them when it reflows the panels), the legend below the last panel that has entries."""
    for p in panels[1:]:
        for k, ax in (("x", p.x), ("y", p.y), ("z", p.z)):
            if ax.label:
                p.hidden_labels[k] = (ax.label, ax.unit)
            ax.label, ax.unit = "", ""
    for p in panels:
        p.legend = "none"
    last = next((p for p in reversed(panels) if p.legend_entries() and not p.note.startswith(
        "no ")), None)
    if last is not None:
        last.legend = "below"


def _grid(n: int, cols: int) -> tuple[int, int]:
    cols = max(1, min(cols, n))
    return int(np.ceil(n / cols)), cols


# == J2.4 (a): 3-D workspace envelope ================================================================
def incisor_cloud(a: dict) -> tuple[np.ndarray, np.ndarray]:
    """Incisal positions of every task (maxillary frame) as displacement from the rest
    position (median of the rest phase; without one the highest position) and the task of each
    point."""
    parts, tags = [], []
    for t in TASKS:
        P = a.get(f"{t}__incisor")
        if P is None:
            continue
        P = np.asarray(P, float)
        P = P[np.isfinite(P).all(axis=1)] if P.ndim == 2 else np.empty((0, 3))
        if len(P):
            parts.append(P)
            tags.append(np.full(len(P), t, dtype=object))
    if not parts:
        return np.empty((0, 3)), np.empty(0, dtype=object)
    P = np.vstack(parts)
    R = np.asarray(a.get("rest__incisor", np.empty((0, 3))), float)
    R = R[np.isfinite(R).all(axis=1)] if R.ndim == 2 else np.empty((0, 3))
    ref = np.median(R, axis=0) if len(R) else P[int(np.argmax(P[:, 2]))]
    return P - ref, np.concatenate(tags)


def design_box(rq: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """The workspace box of the stored P95 design values from the rest position: forward by the
    protrusion, backward by the posterior incisal excursion (X7; 0 before it is stored), to both
    sides by the laterotrusion (larger side), down by the opening."""
    pro, lat, op = (_req(rq, k) for k in ("protrusion_mm", "laterotrusion_mm", "opening_mm"))
    if not all(np.isfinite(v) for v in (pro, lat, op)):
        return None
    back = _req(rq, "posterior_excursion_mm")
    back = back if np.isfinite(back) else 0.0
    return np.array([-back, -lat, -op]), np.array([pro, lat, 0.0])


def _outside(P: np.ndarray, box: tuple[np.ndarray, np.ndarray] | None) -> float:
    if box is None or not len(P):
        return np.nan
    lo, hi = box
    return float(100.0 * np.mean(~((P >= lo - 1e-9) & (P <= hi + 1e-9)).all(axis=1)))


def _behind(P: np.ndarray, box: tuple[np.ndarray, np.ndarray] | None) -> float:
    """Share (%) of the positions outside the box only because they lie behind it (inside the
    lateral and vertical ranges)."""
    if box is None or not len(P):
        return np.nan
    lo, hi = box
    side = ((P[:, 1:] >= lo[1:] - 1e-9) & (P[:, 1:] <= hi[1:] + 1e-9)).all(axis=1)
    return float(100.0 * np.mean(side & (P[:, 0] < lo[0] - 1e-9)))


WORKSPACE_FORMS = ("overlay", "multiples", "3d")


@register("kin_workspace3d", "paper", "Incisal workspace in 3-D", params=("pid",),
          choices={"form": WORKSPACE_FORMS}, form="overlay")
def kin_workspace3d(req: ViewRequest) -> FigureSpec:
    """Posselt-type 3-D workspace: incisal positions of all tasks, their convex hull and the
    design workspace box of the P95 requirements."""
    ctx = req.ctx
    form = req.options.get("form", "overlay")
    pids = ctx.pids()
    sel = _pid(req)
    rq = _requirements(ctx)
    box = design_box(rq)
    p95 = _p_design()
    clouds = {pid: incisor_cloud(_arr(ctx, "kinematics", pid)) for pid in pids}
    have = [pid for pid in pids if len(clouds[pid][0]) >= 5]
    ax, ay, az = _caxes()
    view = (-58.0, 18.0)
    box_series = (_l3(box_edges(*box), "threshold", f"design box (P{p95:.0f})")
                  if box is not None else None)
    outside = {pid: _outside(clouds[pid][0], box) for pid in have}
    behind = {pid: _behind(clouds[pid][0], box) for pid in have}
    v_out, v_back = _fin(list(outside.values())), _fin(list(behind.values()))
    mostly_back = bool(v_out.size and np.median(v_back) > 0.5 * np.median(v_out))
    has_back = box is not None and np.isfinite(_req(rq, "posterior_excursion_mm"))
    span = ("forward by the protrusion, "
            + ("backward by the posterior incisal excursion, " if has_back else "")
            + "to both sides by the laterotrusion, down by the opening")
    back_words = ("; most of the outside positions lie behind it, beyond the design value of "
                  "the posterior excursion" if has_back else
                  "; most of the outside positions lie behind it: the incisal point moves "
                  "backward while opening, a direction the box does not bound")
    sources = ["kinematics:arrays", "requirements:requirements"]
    if form == "multiples":
        panels = []
        for pid in have:
            P, tags = clouds[pid]
            p = Panel(_who(ctx, pid), ax, ay, [], projection="3d", z=az, view=view,
                      legend="none")
            idx = _thin(len(P), 300)
            p.series.append(Series("scatter3d", P[idx, 0], P[idx, 1], z=P[idx, 2],
                                   role="faint", style={"size": 0.25}))
            e = hull_cage(P)
            if e is not None:
                p.series.append(_l3(e, "accent", "hull", width=0.5))
            if box_series is not None:
                p.series.append(_l3(box_edges(*box), "threshold", box_series.label,
                                    width=0.6))
            if np.isfinite(outside[pid]):
                p.note = f"{outside[pid]:.0f} % outside"
            panels.append(p)
        if not panels:
            panels = [_empty(Panel("Workspace", ax, ay, [], projection="3d", z=az),
                             "no incisal kinematics")]
        _set_lims(panels, [clouds[q][0] for q in have] + ([np.vstack(box)] if box else []))
        _multiples_axes(panels)
        rows, cols = _grid(len(panels), 5)
        spec = FigureSpec("kin_workspace3d", "Incisal workspace in 3-D", panels, rows=rows,
                          cols=cols, width="text", height_in=1.35, sources=sources)
        shows = ("Incisal workspace of every participant in 3-D: the incisal positions of all "
                 "five tasks as displacement from the rest position (thinned), their convex "
                 "hull (the Posselt-type envelope in 3-D) drawn as a cage of sections and, in "
                 "every panel, the "
                 f"design workspace box spanned by the P{p95:.0f} requirements ({span}); one "
                 "common millimetre scale. Note: share of the participant's positions outside "
                 "the box; axes as in the first panel")
        marks = [("grey dots", "incisal positions"),
                 ("black lines", "hull sections (sagittal, frontal, horizontal)"),
                 ("dashed box", f"P{p95:.0f} design workspace")]
    else:
        single = form == "3d"
        p = Panel("" if single else "(a) Hulls of all participants", ax, ay, [],
                  projection="3d", z=az, view=view, legend="below")
        if single and sel in have:
            P, tags = clouds[sel]
            for t in TASKS:
                m = tags == t
                if not m.any():
                    continue
                Q = P[m][_thin(int(m.sum()), MAX_POINTS_3D // len(TASKS))]
                p.series.append(Series("scatter3d", Q[:, 0], Q[:, 1], z=Q[:, 2],
                                       role=V.phase_role(t), label=V.task_label(t),
                                       style={"size": 0.55}))
            e = hull_cage(P)
            if e is not None:
                p.series.append(_l3(e, "context", "hull sections", width=0.6))
        else:
            first = True
            for pid in have:
                if pid == sel:
                    continue
                e = hull_cage(clouds[pid][0], rings=0)
                if e is not None:
                    p.series.append(_l3(e, "faint", "other participants" if first else None,
                                        width=0.5))
                    first = False
            if sel in have:
                e = hull_cage(clouds[sel][0])
                if e is not None:
                    p.series.append(_l3(e, "accent", f"{_who(ctx, sel)} (highlighted)",
                                        width=0.9))
        if box_series is not None:
            p.series.append(box_series)
        if not p.series:
            _empty(p, "no incisal kinematics")
        panels = [p]
        if not single:
            # (b) per participant: share of the positions outside the box, and the part of it
            # that lies only behind the box (backward travel, within the other two ranges)
            pb = Panel("(b) Outside the box", Axis("Incisal positions", PCT), Axis(""), [],
                       legend="below")
            order = sorted(have, key=lambda q: outside[q] if np.isfinite(outside[q]) else -1)
            y = np.arange(len(order), dtype=float)
            vals = np.array([outside[q] for q in order], float)
            back = np.array([behind[q] for q in order], float)
            if np.isfinite(vals).any():
                pb.series.append(Series("barh", vals, y, role="context",
                                        style={"bar_width": 0.62}, label="outside"))
                pb.series.append(Series("barh", back, y, role="quaternary",
                                        style={"bar_width": 0.62},
                                        label="only behind it"))
                pb.y.ticks, pb.y.ticklabels = list(y), B.pseudo_rows(ctx, order)
                pb.y.reverse = True
                pb.x.lim = (0.0, max(5.0, float(np.nanmax(vals)) * 1.08))
            else:
                _empty(pb, "no design box" if box is None else "no data")
            pb.height = 0.8
            panels.append(pb)
        spec = FigureSpec("kin_workspace3d", "Incisal workspace in 3-D", panels,
                          rows=1, cols=len(panels), width="text" if not single else "column",
                          height_in=2.6 if not single else 2.9, sources=sources,
                          meta={"pid": sel})
        who = _who(ctx, sel) if sel else "no participant"
        if single:
            shows = (f"Incisal workspace of {who} in 3-D: every incisal position of the five "
                     "tasks as displacement from the rest position, coloured by task, their "
                     "convex hull (the Posselt-type envelope in 3-D) as a cage of sections, and "
                     f"the design workspace box spanned by the P{p95:.0f} requirements ({span})"
                     + (f"; {outside.get(sel, np.nan):.0f} % of the positions lie "
                                   "outside the box" if np.isfinite(outside.get(sel, np.nan))
                                   else ""))
            marks = [("dots", "incisal positions, colour by task"),
                     ("grey lines", "hull sections (sagittal, frontal, horizontal)"),
                     ("dashed box", f"P{p95:.0f} design workspace")]
        else:
            shows = ("(a) Convex hull of every participant's incisal positions over the five "
                     "tasks, as displacement from the rest position (Posselt-type envelope in "
                     "3-D): its sagittal and frontal section through the centre, and for "
                     f"{who} also horizontal sections; and the design "
                     f"workspace box spanned by the P{p95:.0f} requirements ({span}). (b) Per "
                     "participant the share of the incisal positions outside the box and the "
                     "part of it that lies only behind the box"
                     + (back_words if mostly_back else ""))
            marks = [("grey lines in (a)", "other participants' hull sections"),
                     ("black lines in (a)", f"{who}'s hull sections"),
                     ("dashed box in (a)", f"P{p95:.0f} design workspace"),
                     ("grey bars in (b)", "outside the box"),
                     ("vermilion bars in (b)", "only behind the box"
                      + (" (beyond its posterior bound)" if has_back else ""))]
    n = len(have)
    return attach(spec, CaptionSpec(
        shows, marks, f"n = {n} participants",
        f"design values: P{p95:.0f} across participants (Harrell-Davis) of the largest "
        "excursion per participant; hull over all positions",
        evidence="descriptive",
        notes=["Maxillary frame: x anterior, y left, z up; the axes have equal millimetre "
               "scales."] + ([] if box is not None else
                             ["The requirements are not in this result set; no box is drawn."])
        + (["The posterior incisal excursion is not in this result set; the box starts at the "
            "rest position."] if box is not None and not has_back else []),
        codes=["S1", "J2.4"]))


# == J2.4 (b): condyle paths and helical-axis migration ==============================================
def _fha_segments(fha: pd.DataFrame, half: float) -> dict[bool, list[np.ndarray]]:
    """Finite helical axes as segments ±``half`` around their sagittal piercing point, split
    into early steps (ending at or below 10°) and later ones."""
    out: dict[bool, list[np.ndarray]] = {True: [], False: []}
    if fha.empty:
        return out
    for r in fha.itertuples():
        n = np.array([r.n_x, r.n_y, r.n_z], float)
        c = np.array([r.pierce_x_mm, 0.0, r.pierce_z_mm], float)
        if not (np.isfinite(n).all() and np.isfinite(c).all()):
            continue
        out[bool(r.theta_to_deg <= FHA_EARLY_DEG)].append(np.vstack([c - half * n, c + half * n]))
    return out


FHA_EARLY_DEG = 10.0       # the early / later split of icorr_hinge (b) and (a, path3d)
CONDYLE_FORMS = ("2d", "multiples", "representatives")


@register("kin_condyle3d", "paper", "Condyle paths and helical axes",
          choices={"form": CONDYLE_FORMS}, form="2d")
def kin_condyle3d(req: ViewRequest) -> FigureSpec:
    """Per participant: condyle-point paths of opening and the finite helical axes (early vs
    later opening), small multiples; implausible condyles muted. Form ``representatives``
    (13 3-D panels do not read at print size): three participants with plausible
    condyles -- the smallest, median and largest condyle excursion -- at a readable size."""
    ctx = req.ctx
    form = req.options.get("form", "2d")     # 2-D sagittal: the 3-D multiples do not read at
    pids = ctx.pids()                         # print size; 3-D stays a gallery form
    pl = plausible(ctx)
    fha = _t(ctx, "kinematics", "fha")
    subj = _subjects(ctx)
    if form == "representatives":
        ok = [p for p in pids if pl.get(p) is True]
        exc = np.fmax(_col(subj, "condyle_left_disp_mm", ok),
                      _col(subj, "condyle_right_disp_mm", ok))
        fin = [(e, p) for e, p in zip(exc, ok) if np.isfinite(e)]
        fin.sort()
        if len(fin) >= 3:
            pids = [fin[0][1], fin[len(fin) // 2][1], fin[-1][1]]
    ic = _col(subj, "intercondylar_mm", pids)
    half = 0.5 * float(np.nanmedian(ic)) if np.isfinite(ic).any() else 50.0
    three = form != "2d"
    panels, pts = [], []
    n_cond = 0
    for pid in pids:
        a = _arr(ctx, "kinematics", pid)
        L, R = (np.asarray(a.get(f"open_close__tmj_{s}", np.empty((0, 3))), float)
                for s in ("left", "right"))
        ok = pl.get(pid)
        muted = ok is not True
        title = _who(ctx, pid) + ("" if ok else " (flagged)" if ok is False else " (unknown)")
        if three:
            p = Panel(title, *_caxes()[:2], [], projection="3d", z=_caxes()[2],
                      view=(-62.0, 14.0), legend="none")
        else:
            p = Panel(title, Axis("Anterior", MM), Axis("Superior", MM), [], equal=True,
                      legend="none")
        seg = _fha_segments(fha[fha.pid == pid] if not fha.empty else fha, half)
        for early, role in ((True, "cat:2"), (False, "cat:4")):
            if not seg[early]:
                continue
            E = _nan_join(seg[early])
            lab = None if muted else f"helical axis {'≤' if early else '>'} {FHA_EARLY_DEG:.0f}°"
            if three:
                p.series.append(_l3(E, role if not muted else "faint", lab, width=0.45))
            else:
                C = np.array([[s_[:, 0].mean(), s_[:, 2].mean()] for s_ in seg[early]])
                # readable at print size; flagged participants mid-grey, not near-white
                p.series.append(Series("scatter", C[:, 0], C[:, 1], role=role if not muted
                                       else "context", label=lab, style={"size": 1.1}))
            pts.append(E)
        for side, P in (("left", L), ("right", R)):
            if P.ndim != 2 or not len(P):
                continue
            pts.append(P)
            role = f"side:{side}" if not muted else "context"
            lab = None if muted else f"{side} condyle point"
            if three:
                p.series.append(_l3(P, role, lab, width=0.9))
            else:
                p.series.append(Series("line", P[:, 0], P[:, 2], role=role, label=lab,
                                       style={"width": 0.9}))
        if any(P.ndim == 2 and len(P) for P in (L, R)):
            n_cond += 1
        if not p.series:
            _empty(p, "no kinematics")
        panels.append(p)
    if three:
        _set_lims(panels, pts)
    else:
        from biomech.viz.views.data import fit_equal

        fit_equal(panels, same=True)
        for p in panels:                 # narrow panels: a tick every 100 mm
            if p.x.lim is not None:
                lo, hi = p.x.lim
                p.x.ticks = [float(t) for t in np.arange(np.ceil(lo / 50) * 50, hi, 100)]
    _multiples_axes(panels)
    for p in panels:                                 # the legend below an unflagged panel
        p.legend = "none"
    last = next((p for p in reversed(panels)
                 if any(s.role.startswith("side:") for s in p.series)), None)
    if last is not None:
        last.legend = "below"
    rows, cols = _grid(len(panels), 5)
    n_flag = sum(1 for p in pids if pl.get(p) is False)
    if form == "representatives":
        rows, cols = 1, len(panels)
    spec = FigureSpec("kin_condyle3d", "Condyle paths and helical axes", panels, rows=rows,
                      cols=cols, width="text",
                      height_in=2.4 if form == "representatives" else 1.35 if three else 1.25,
                      sources=["kinematics:arrays", "kinematics:fha",
                               "stats_kinematics:subjects"])
    plane = ("in 3-D" if three else "in the sagittal plane (the helical axes as the midpoints "
             "of their segments)")
    if form == "representatives":
        plane = ("in 3-D, for three participants with plausible condyle points (smallest, "
                 "median and largest condyle excursion; all participants: the other forms)")
    return attach(spec, CaptionSpec(
        f"Per participant, {plane}: paths of the two skin-palpated condyle points during the "
        "open-close task and the finite helical axes of successive opening steps, drawn over "
        "the median intercondylar width around their sagittal piercing point. Participants whose "
        "palpated condyles fail the plausibility check (depth, distance, or level) are drawn in "
        "grey and marked flagged: shown, not excluded",
        [("blue and orange lines", "left and right condyle point"),
         ("green lines" if three else "green diamonds", f"helical axes of steps ending at or "
                                                         f"below {FHA_EARLY_DEG:.0f}°"),
         ("pink lines" if three else "pink triangles", "helical axes of later steps"),
         ("grey", "flagged condyles"), ("axes", "as in the first panel")],
        f"n = {len(pids)} participants, {n_flag} with flagged condyles, {n_cond} with condyle "
        "paths",
        "every opening cycle; finite helical axes in steps of the opening angle",
        evidence="descriptive",
        notes=["Maxillary frame: x anterior, y left, z up; one common millimetre scale. Condyle "
               "points are skin landmarks, not the anatomical condyles."],
        codes=["C1", "D25", "J2.4"]))


# == J2.4 (c): coupling surface ======================================================================
def coupling_summary(ctx: Any) -> pd.DataFrame:
    """The stored per-participant coupling summary (slope, generic-curve miss), indexed by
    pid; empty when this result set has none."""
    c = _t(ctx, "stats_kinematics", "coupling_summary")
    return c.drop_duplicates("pid").set_index("pid") if not c.empty and "pid" in c \
        else pd.DataFrame()


def _coupling_curves(ctx: Any) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None,
                                         list[str]]:
    """Grid, participant median curves (rows = participants in study order) and the cohort
    median curve of the anterior condyle-midpoint translation."""
    a = _arr(ctx, "stats_kinematics", COHORT_UNIT)
    g = a.get("grid_deg")
    S = a.get("tmj_mid_ant_mm__subjects")
    med = a.get("tmj_mid_ant_mm__median")
    pids = ctx.pids()
    if g is None or S is None:
        return None, None, None, []
    S = np.atleast_2d(np.asarray(S, float))
    rows = list(pids[:len(S)]) if len(S) <= len(pids) else []
    g = np.asarray(g, float)
    med = np.asarray(med, float) if med is not None else None
    keep = np.isfinite(S[:len(rows)]).any(axis=0) if rows else np.ones(g.size, bool)
    if keep.any():                         # only the angles some participant reaches
        last = int(np.flatnonzero(keep).max()) + 1
        g, S = g[:last], S[:, :last]
        med = med[:last] if med is not None else None
    return g, S[:len(rows)], med, rows


def _slope_window() -> tuple[float, float]:
    from biomech.stats.cohort_kinematics import COUPLING_SLOPE_DEG

    return tuple(float(v) for v in COUPLING_SLOPE_DEG)       # type: ignore[return-value]


COUPLING_FORMS = ("curves", "3d")


@register("kin_coupling_surface", "paper", "Condyle translation over opening angle",
          choices={"form": COUPLING_FORMS}, form="curves")
def kin_coupling_surface(req: ViewRequest) -> FigureSpec:
    """Anterior condyle-midpoint translation over the opening angle per participant: a surface
    (angle x participant) or curves with the generic median, slope and generic-curve miss."""
    ctx = req.ctx
    form = req.options.get("form", "curves")
    g, S, med, rows = _coupling_curves(ctx)
    cs = coupling_summary(ctx)
    pl = plausible(ctx)
    w0, w1 = _slope_window()
    have_cs = not cs.empty and "slope_mm_per_deg" in cs
    sources = ["stats_kinematics:coupling", "stats_kinematics:coupling_summary",
               "stats_kinematics:subjects"]
    if g is None or S is None or not rows:
        p = _empty(Panel("Coupling", Axis("Opening angle", DEG), Axis("Translation", MM)),
                   "no coupling curves")
        spec = FigureSpec("kin_coupling_surface", "Condyle translation over opening angle",
                          [p], width="column", height_in=1.6, sources=sources)
        return attach(spec, CaptionSpec(
            "Anterior translation of the condyle-point midpoint over the opening angle per "
            "participant; this result set holds no coupling curves",
            [("panel", "empty")], "n = 0 participants", "none", evidence="descriptive"))
    slope = {pid: _f(cs.at[pid, "slope_mm_per_deg"]) if have_cs and pid in cs.index else np.nan
             for pid in rows}
    miss = {pid: _f(cs.at[pid, "generic_rms_mm"]) if have_cs and pid in cs.index
            and "generic_rms_mm" in cs else np.nan for pid in rows}
    win = (g >= w0) & (g <= w1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mid = {pid: float(np.nanmean(S[i, win])) for i, pid in enumerate(rows)}
    key = slope if have_cs else mid
    order = sorted(rows, key=lambda q: (not np.isfinite(key[q]), key[q]))
    neg = [q for q in rows if np.isfinite(slope[q]) and slope[q] < 0]
    if form == "3d":
        Z = np.array([S[rows.index(q)] for q in order])
        y = np.arange(1, len(order) + 1, dtype=float)
        p = Panel("", Axis("Opening angle", DEG), Axis("Participant rank"), [],
                  projection="3d", z=Axis("Anterior translation", MM), view=(-128.0, 26.0),
                  legend="below")
        p.series.append(Series("surface3d", x=g, y=y, z=Z, style={"ramp": "sequential",
                                                                    "alpha": 0.9}))
        if med is not None and np.isfinite(med).any():
            p.series.append(Series("line3d", g, np.zeros_like(g), z=med, role="accent",
                                   label="cohort median curve (front edge)",
                                   style={"width": 1.2}))
        for i, q in enumerate(order):
            if q in neg:
                p.series.append(Series("line3d", g, np.full_like(g, y[i]), z=Z[i],
                                       role="quaternary", style={"width": 0.9},
                                       label="negative slope" if q == next(
                                           x for x in order if x in neg) else None))
        spec = FigureSpec("kin_coupling_surface", "Condyle translation over opening angle",
                          [p], width="column", height_in=2.7, sources=sources)
        how = ("by the stored slope" if have_cs else
               f"by the mean translation over {w0:.0f}–{w1:.0f}°")
        return attach(spec, CaptionSpec(
            "Anterior translation of the condyle-point midpoint over the opening angle (rows: "
            f"participants ranked {how}; surface height and colour: translation, from the "
            "start of each opening), with the cohort median curve along the front edge. Each "
            "participant's curve is the median over the open-close openings where at least "
            "the minimum number of cycles reach that angle",
            [("surface", "translation, colour by height"),
             ("black line", "cohort median curve")]
            + ([("vermilion lines", "participants with a negative slope")] if neg else []),
            f"n = {len(rows)} participants",
            "median over openings per participant; median over participants",
            evidence="post hoc description",
            notes=["Condyle points are skin-palpated; a negative slope means the palpated "
                   "midpoint moves backward, which a skin landmark can do."],
            codes=["X9", "J2.4"]))
    # -- curves form: (a) curves, (b) slope, (c) generic-curve miss
    pa = Panel("(a) Translation over opening angle", Axis("Opening angle", DEG),
               Axis("Anterior translation", MM), [], legend="below")
    pa.series.append(Series("hspan", x=np.array([w0]), y2=np.array([w1]), role="faint"))
    first_n = True
    for q in rows:
        y_ = S[rows.index(q)]
        if not np.isfinite(y_).any():           # no curve: nothing to draw (pgfplots warns)
            continue
        if q in neg:
            pa.series.append(Series("line", g, y_, role="quaternary", style={"width": 0.8},
                                    label="negative slope" if first_n else None))
            first_n = False
        else:
            pa.series.append(Series("line", g, y_, role="context", style={"width": 0.6}))
    if med is not None and np.isfinite(med).any():
        B.add_ci_band(pa, g, S, centre="median", seed=B.seed_of(ctx), where=np.isfinite(med))
        pa.series.append(Series("line", g, med, role="accent", style={"width": 1.4},
                                label="cohort median (generic curve)"))
    panels = [pa]
    labels = B.pseudo_rows(ctx, order)
    yy = np.arange(len(order), dtype=float)
    for tag, vals, title, axis in (
            ("b", slope, "(b) Slope", Axis(f"Slope {w0:.0f}–{w1:.0f}°", "\\milli\\metre\\per"
                                           "\\degree")),
            ("c", miss, "(c) Generic-curve miss", Axis("RMS difference", MM))):
        p = Panel(title, axis, Axis(""), [], legend="none")
        v = np.array([vals[q] for q in order], float)
        if np.isfinite(v).any():
            okm = np.array([pl.get(q) is True for q in order])
            lo, hi = B.median_ci(v, B.seed_of(ctx))
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("hspan", x=np.array([lo]), y2=np.array([hi]),
                                       role="faint"))
            p.series.append(Series("vline", x=np.array([float(np.nanmedian(v))]),
                                   role="context"))
            if tag == "b":
                p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
            for m, role, st in ((okm, "accent", None), (~okm, "flagged", None)):
                if np.isfinite(v[m]).any():     # an empty group draws nothing
                    p.series.append(Series("scatter", v[m], yy[m], role=role,
                                           **({"style": st} if st else {})))
            p.y.ticks, p.y.ticklabels = list(yy), labels
            p.y.reverse = True
        else:
            _empty(p, NOT_STORED)
        panels.append(p)
    spec = FigureSpec("kin_coupling_surface", "Condyle translation over opening angle", panels,
                      rows=1, cols=3, width="text", height_in=max(1.7, B.rows_height(
                          len(order), per=0.13, extra=0.3)), sources=sources)
    marks = [("grey lines in (a)", "participant median curves"),
             ("black line and band in (a)", "cohort median with its 95 % bootstrap CI"),
             ("shaded in (a)", f"slope window {w0:.0f}–{w1:.0f}°")]
    if neg:
        marks.append(("vermilion lines in (a)", "participants with a negative slope"))
    marks += [("dots in (b, c)", "participants (black: plausible condyles; grey triangles: "
                                 "flagged)"),
              ("line and band in (b, c)", "median and its 95 % BCa CI")]
    return attach(spec, CaptionSpec(
        "(a) Anterior translation of the condyle-point midpoint from the start of each opening "
        "over the opening angle: each participant's median curve over the open-close openings "
        "and the cohort median curve, the generic coupling model. (b) Per participant the "
        f"Theil-Sen slope of the translation over {w0:.0f}–{w1:.0f}°. (c) Per participant the "
        "RMS difference between the own median curve and the cohort median of the other "
        "participants (how much a generic coupling model misses)",
        marks, f"n = {len(rows)} participants"
        + (f", {sum(np.isfinite(list(slope.values())))} with a slope" if have_cs else ""),
        "medians over openings and participants; Theil-Sen slope; RMS over the angles both "
        "curves cover", evidence="post hoc description",
        notes=["Condyle points are skin-palpated; negative slopes are consistent with that "
               "caveat."] + ([] if have_cs else
                             ["(b, c): the per-participant coupling summary is " + NEXT_RUN
                              + "."]),
        codes=["X9", "J4.4", "J2.4"]))


# == J2.5 A1: which participant enters which outcome =================================================
def _outcome_rows(ctx: Any) -> list[tuple[str, str, dict[str, str]]]:
    """(group, outcome, pid -> state) with state "used", "flagged" (used, flag shown) or "none"
    (no usable value), from the stored per-participant values each outcome is computed from."""
    pids = ctx.pids()
    s = _subjects(ctx)
    pl = plausible(ctx)

    def from_values(v: np.ndarray, flag: dict[str, bool] | None = None) -> dict[str, str]:
        return {pid: ("none" if not np.isfinite(x) else
                      "flagged" if flag is not None and flag.get(pid) else "used")
                for pid, x in zip(pids, v)}

    cond_flag = {pid: pl.get(pid) is not True for pid in pids}
    lat = np.fmax(_col(s, "laterotrusion_left_mm", pids), _col(s, "laterotrusion_right_mm", pids))
    cond = np.fmax(_col(s, "condyle_left_disp_mm", pids), _col(s, "condyle_right_disp_mm", pids))
    rows = [("workspace", "largest incisal opening",
             from_values(_col(s, "max_observed_opening_mm", pids))),
            ("workspace", "protrusion", from_values(_col(s, "protrusion_mm", pids))),
            ("workspace", "laterotrusion", from_values(lat)),
            ("workspace", "opening rotation", from_values(_col(s, "opening_angle_deg", pids))),
            ("dynamics", "peak opening velocity",
             from_values(_col(s, "peak_open_vel_max_dps", pids)))]
    bw = _t(ctx, "imu", "bandwidth")
    if not bw.empty and "f95_coh_hz" in bw:
        from biomech.viz.views.paper_journal import BANDWIDTH_TASKS

        b = bw[bw.task.isin(BANDWIDTH_TASKS)]
        nl = b.noise_limited.astype(bool) if "noise_limited" in b else pd.Series(False,
                                                                                 index=b.index)
        st = {}
        for pid in pids:
            q = b[b.pid == pid]
            fin = q.f95_coh_hz.notna()
            st[pid] = ("used" if (fin & ~nl[q.index]).any() else
                       "flagged" if fin.any() else "none")
        rows.append(("dynamics", "movement bandwidth", st))
    rows.append(("joint", "condyle-point translation", from_values(cond, cond_flag)))
    hn = _t(ctx, "uncertainty", "hinge_noise")
    hn = hn.drop_duplicates("pid").set_index("pid") if not hn.empty else hn
    for col, lab, flag in (("obs_hinge_fit_mm", "misalignment, compromise axis", None),
                           ("obs_hinge_tmj_mm", "misalignment, hinge at the joint", cond_flag)):
        rows.append(("joint", lab, from_values(_col(hn, col, pids), flag)))
    ts = _t(ctx, "stats_timing", "timing_spec")
    if not ts.empty:
        from biomech.viz.views.paper_journal import LEAD_SPEC

        oc = ts[ts.outcome == "open_close|opening|lead"]
        for rule, _ in RULES:
            v = oc[oc.variant == LEAD_SPEC.format(rule)].drop_duplicates("pid").set_index(
                "pid").value
            rows.append(("intent", f"EMG lead, {RULE.get(rule, rule)}",
                         from_values(pd.to_numeric(v.reindex(pids), errors="coerce")
                                     .to_numpy(float))))
    ic = _t(ctx, "stats_imu", "imu_opening_cycles")
    if not ic.empty and "pid" in ic:
        val = next((c for c in ("opening_imu_loso_mm", "opening_imu_mm") if c in ic), None)
        got = (ic.dropna(subset=[val]) if val else ic).pid.unique()
        rows.append(("sensing", "IMU incisal opening",
                     {pid: "used" if pid in set(got) else "none" for pid in pids}))
    lo = _t(ctx, "ml_eval", "loso")
    if not lo.empty and {"pid", "feature_set", "balanced_accuracy"} <= set(lo.columns):
        v = lo[lo.feature_set == "emg_imu"].groupby("pid").balanced_accuracy.max()
        rows.append(("intent", "intent decoding",
                     from_values(pd.to_numeric(v.reindex(pids), errors="coerce")
                                 .to_numpy(float))))
    return rows


GROUP_ROLE = {"workspace": "cat:0", "dynamics": "cat:1", "joint": "cat:2", "sensing": "cat:4",
              "intent": "cat:3"}


@register("cov_outcome_n", "paper", "Participants per outcome")
def cov_outcome_n(req: ViewRequest) -> FigureSpec:
    """Which participant enters which outcome: used, used with a flag shown, or without a usable
    value -- the n behind every outcome under flag-not-exclude."""
    ctx = req.ctx
    pids = ctx.pids()
    rows = _outcome_rows(ctx)
    p = Panel("Participants per outcome", Axis(""), Axis(""), [], legend="below")
    first = {"used": True, "flagged": True, "none": True}
    for i, (grp, lab, st) in enumerate(rows):
        for state, role, ms, size, text in (
                ("used", GROUP_ROLE.get(grp, "accent"), "o", 1.0, "used"),
                ("flagged", "secondary", "v", 1.25, "used, flag shown (condyles or "
                                                    "noise-limited band)"),
                ("none", "faint", "x", 0.8, "no usable value")):
            xs = [j for j, pid in enumerate(pids) if st.get(pid) == state]
            if not xs:
                continue
            p.series.append(Series("scatter", np.array(xs, float), np.full(len(xs), float(i)),
                                   role=role if state != "used" else "accent",
                                   style={"ms": ms, "size": size},
                                   label=text if first[state] else None))
            first[state] = False
        n_used = sum(1 for v in st.values() if v in ("used", "flagged"))
        p.series.append(Series("text", np.array([len(pids) - 0.3]), np.array([float(i)]),
                               text=[f"{n_used}"], style={"anchor": "right"}))
    p.y.ticks = [float(i) for i in range(len(rows))]
    p.y.ticklabels = [lab for _, lab, _ in rows]
    p.y.lim = (-0.6, len(rows) - 0.4)
    p.y.reverse = True
    p.x.ticks = [float(j) for j in range(len(pids))]
    p.x.ticklabels = B.pseudo_rows(ctx, pids)
    p.x.lim = (-0.6, len(pids) + 0.6)
    p.x.label = "Participant (right: n entering)"
    if not any(v != "none" for _, _, st in rows for v in st.values()):
        _empty(p, "no outcome values")
    spec = FigureSpec("cov_outcome_n", "Participants per outcome", [p], width="text",
                      height_in=B.rows_height(len(rows), per=0.17, extra=0.5),
                      sources=["stats_kinematics:subjects", "uncertainty:hinge_noise",
                               "imu:bandwidth", "stats_timing:timing_spec",
                               "stats_imu:imu_opening_cycles", "ml_eval:loso"])
    n_any = sum(1 for _, _, st in rows if all(v != "none" for v in st.values()))
    return attach(spec, CaptionSpec(
        "Which participant enters which outcome (columns: participants; rows: the outcomes of "
        "the requirements, the joint model, sensing, and intent). Every recorded take is "
        "analysed and quality is flagged, not excluded: a participant leaves an outcome only "
        "when it has no usable value (for the EMG lead: fewer than the minimum usable "
        "channel-cycles), and a flagged value stays in and is marked. The number at the right "
        "is the n of that outcome",
        [("black dots", "used"), ("orange triangles", "used, with a flag shown"),
         ("light crosses", "no usable value")],
        f"n = {len(pids)} participants recorded; {n_any} of {len(rows)} outcomes use every "
        "participant",
        "counts of participants with a usable value", evidence="descriptive",
        codes=["A1", "J2.5"]))


# == J2.5 A2: peak velocity under the low-pass sensitivity runs =======================================
def lowpass_runs(ctx: Any) -> list[tuple[str, float]]:
    """(variant, low-pass cut-off in Hz) of the sensitivity runs that change only the pose
    low-pass, by cut-off."""
    from biomech import sensitivity as S

    try:
        specs = S.variants(ctx.config)
    except AttributeError:
        return []
    out = []
    for name, spec in specs.items():
        sets = (spec or {}).get("set") or {}
        if list(sets) == ["signals.kinematics.lowpass_hz"]:
            out.append((name, float(sets["signals.kinematics.lowpass_hz"])))
    return sorted(out, key=lambda r: r[1])


def _signal(ctx: Any, block: str, key: str) -> float:
    """A signal-processing parameter of the study (``signals.<block>.<key>``), NaN without."""
    try:
        return _f(((getattr(ctx.config, "signals", None) or {}).get(block) or {}).get(key))
    except AttributeError:
        return float("nan")


def primary_lowpass(ctx: Any) -> float:
    """The primary pose low-pass cut-off (``signals.kinematics.lowpass_hz``)."""
    return _signal(ctx, "kinematics", "lowpass_hz")


def _variant_subject_values(ctx: Any, name: str, column: str) -> pd.Series:
    from biomech.viz.numbers import sensitivity_table

    try:
        t = sensitivity_table(ctx, name)
    except Exception:
        return pd.Series(dtype=float)
    if t.empty or "@pid" not in t:
        return pd.Series(dtype=float)
    r = t[(t.stage == "stats_kinematics") & (t.table == "subjects") & (t.column == column)]
    r = r[r["@pid"].isin(ctx.pids())]
    # The subjects key also holds per-participant task labels (e.g. peak_acc_task); when a run
    # changes one, the participant is a primary-only plus a variant-only row: keep its one value.
    v = pd.to_numeric(r.set_index("@pid")["variant"], errors="coerce").dropna()
    return v.groupby(level=0).first()


@register("req_lowpass_bound", "paper", "Peak velocity is a lower bound")
def req_lowpass_bound(req: ViewRequest) -> FigureSpec:
    """Per participant the peak opening angular velocity with the primary pose low-pass and with
    each low-pass sensitivity run, the P95 per cut-off, and the ratio to the primary value."""
    from biomech.stats.requirements import harrell_davis

    ctx = req.ctx
    pids = ctx.pids()
    col = "peak_open_vel_max_dps"
    f0 = primary_lowpass(ctx)
    runs = lowpass_runs(ctx)
    base = pd.Series(_col(_subjects(ctx), col, pids), index=pids)
    cols = {f0: base}
    for name, fc in runs:
        v = _variant_subject_values(ctx, name, col)
        if v.notna().any():
            cols[fc] = v.reindex(pids)
    M = pd.DataFrame(cols).sort_index(axis=1)
    cut = [float(c) for c in M.columns]
    p95 = _p_design()
    pa = Panel("(a) Peak opening velocity", Axis("Pose low-pass cut-off", HZ),
               Axis("Peak angular velocity", DPS), [], legend="below")
    pb = Panel("(b) Relative to the primary", Axis("Ratio to the primary value"), Axis(""), [],
               legend="none")
    n_runs = len(cut) - 1
    if n_runs >= 1:
        x = np.array(cut)
        for pid in pids:
            y = M.loc[pid].to_numpy(float)
            if np.isfinite(y).sum() >= 2:
                pa.series.append(Series("line", x, y, role="context", style={"width": 0.6}))
        hd = np.array([harrell_davis(M[c].to_numpy(float), p95 / 100) for c in M.columns])
        pa.series.append(Series("line", x, hd, role="accent", style={"width": 1.4},
                                label=f"P{p95:.0f} across participants"))
        pa.series.append(Series("scatter", x, hd, role="accent", style={"ms": "d",
                                                                        "size": 1.3}))
        pa.series.append(Series("vline", x=np.array([f0]), role="reference",
                                label="primary cut-off"))
        pa.x.ticks = [float(c) for c in cut]
        pa.x.ticklabels = [f"{c:g}" for c in cut]
        ratios = [(M[c] / M[f0]).to_numpy(float) for c in M.columns if c != f0]
        labels = [f"{c:g} Hz" for c in M.columns if c != f0]
        pb = B.strip_ci(labels, ratios, "(b) Relative to the primary",
                        Axis(f"Ratio to the {f0:g} Hz value"), ref=1.0, seed=B.seed_of(ctx))
        pb.legend = "none"
    else:
        _empty(pa, "low-pass sensitivity runs " + NOT_STORED)
        _empty(pb, "no sensitivity runs")
    spec = FigureSpec("req_lowpass_bound", "Peak velocity is a lower bound", [pa, pb], rows=1,
                      cols=2, width="text", height_in=1.7,
                      sources=["stats_kinematics:subjects"]
                      + [f"sensitivity:{n}" for n, _ in runs])
    others = ", ".join(f"{c:g}" for c in cut if c != f0) or "none"
    return attach(spec, CaptionSpec(
        "Peak opening angular velocity per participant (largest open-close cycle) with the "
        f"primary pose low-pass ({f0:g} Hz) and with the low-pass sensitivity runs ({others} "
        "Hz), each re-run on the same participants. (a) Values per cut-off with the design "
        f"value (P{p95:.0f}); (b) per cut-off the ratio to the primary value. A higher cut-off "
        "keeps more of the peak, so the primary requirement is a lower bound; the comparison with "
        "Qualisys at 200 Hz in the companion validation study bounds the remaining loss and "
        "is cited in the text",
        [("grey lines in (a)", "participants"), ("black diamonds in (a)",
                                                  f"P{p95:.0f} (Harrell-Davis)"),
         ("dashed line in (a)", "primary cut-off"), ("dots in (b)", "participants"),
         ("diamond and line in (b)", "median with its 95 % BCa CI")],
        f"n = {int(M[f0].notna().sum()) if f0 in M else 0} participants",
        f"P{p95:.0f} across participants (Harrell-Davis); median ratio with BCa CI",
        evidence="post hoc sensitivity",
        codes=["D20", "VD9", "A2", "J2.5"]))


# == J2.5 A2: requirement envelope and coverage of the sample maximum ==================================
ENVELOPE_ROWS = (("peak_open_vel_max_dps", "peak_vel_dps", "opening, open-close"),
                 ("peak_close_vel_max_dps", "close_vel_dps", "closing, open-close"),
                 ("free_peak_open_vel_dps", "free_peak_open_vel_dps", "opening, free motion"),
                 ("free_peak_close_vel_dps", None, "closing, free motion"))


@register("req_envelope", "paper", "Dynamic requirement envelope")
def req_envelope(req: ViewRequest) -> FigureSpec:
    """Peak opening and closing angular velocity (open-close and free motion) and peak angular
    acceleration per participant with their P95 design values, and the chance that the sample
    maximum covers a population quantile against n."""
    ctx = req.ctx
    pids = ctx.pids()
    s = _subjects(ctx)
    rq = _requirements(ctx)
    p95 = _p_design()
    # (a) velocities
    labels, vals, marks, lows, missing = [], [], [], [], []
    for col, key, lab in ENVELOPE_ROWS:
        v = _col(s, col, pids)
        labels.append(lab)
        vals.append(v)
        if not np.isfinite(v).any():
            missing.append(lab)
        val = _req(rq, key) if key else np.nan
        marks.append((val, np.nan, np.nan))
        lows.append(_req(rq, key, "ci_lo") if key else np.nan)
    pa = B.strip_mark(labels, vals, "(a) Peak angular velocity",
                      Axis("Peak angular velocity", DPS), marks=marks,
                      mark_label=f"P{p95:.0f}")
    lo_lab = "from the one-sided lower 95 % bound"
    for i, (lo, m) in enumerate(zip(lows, marks)):
        if np.isfinite(lo) and np.isfinite(m[0]):
            pa.series.append(Series("line", np.array([lo, m[0]]), np.array([float(i)] * 2),
                                    role="accent", style={"width": 1.3}, label=lo_lab))
            lo_lab = None
    pa.legend = "below"
    if missing:
        pa.note = f"{len(missing)} rows {NOT_STORED}"
    # (b) peak angular acceleration (largest, P99 robustness) per participant
    pb = Panel("(b) Peak angular acceleration", Axis("Angular acceleration", V.DPS2),
               Axis(""),
               [], legend="below")
    acc, acc99 = _col(s, "peak_acc_dps2", pids), _col(s, "peak_acc_p99_dps2", pids)
    task = (s["peak_acc_task"].reindex(pids).astype(object).to_numpy()
            if not s.empty and "peak_acc_task" in s else np.full(len(pids), "", dtype=object))
    if np.isfinite(acc).any():
        order = [i for i in np.argsort(np.where(np.isfinite(acc), acc, np.inf))]
        y = np.arange(len(order), dtype=float)
        for k, i in enumerate(order):
            if np.isfinite(acc99[i]) and np.isfinite(acc[i]):
                pb.series.append(Series("line", np.array([acc99[i], acc[i]]),
                                        np.array([k, k], float), role="faint",
                                        style={"width": 1.0}))
        pb.series.append(Series("scatter", acc99[order], y, role="context",
                                style={"ms": "s", "size": 1.15}, label="99th percentile"))
        for t in TASKS:
            m = np.array([task[i] == t for i in order])
            if m.any():
                pb.series.append(Series("scatter", acc[order][m], y[m], role=V.phase_role(t),
                                        style={"size": 1.25},
                                        label=f"largest, in {V.task_label(t)}"))
        val, lo = _req(rq, "peak_acc_dps2"), _req(rq, "peak_acc_dps2", "ci_lo")
        if np.isfinite(lo) and np.isfinite(val):
            pb.series.insert(0, Series("hspan", x=np.array([lo]), y2=np.array([val]),
                                       role="faint"))
        if np.isfinite(val):
            pb.series.append(Series("vline", x=np.array([val]), role="threshold"))
        pb.y.ticks, pb.y.ticklabels = list(y), B.pseudo_rows(ctx, [pids[i] for i in order])
        pb.y.reverse = True
    else:
        _empty(pb, NOT_STORED)
    # (c) coverage of the sample maximum
    cov = _t(ctx, "requirements", "max_coverage")
    pc = Panel("(c) Sample maximum vs n", Axis("Participants"),
               Axis("Coverage", PCT, lim=(0.0, 104.0)), [], legend="below")
    conf = np.nan
    if not cov.empty and {"q", "n", "p_max_exceeds"} <= set(cov.columns):
        conf = _f(cov.conf.iloc[0]) if "conf" in cov else np.nan
        sm = cov.smallest_n.astype(bool) if "smallest_n" in cov else pd.Series(False,
                                                                                index=cov.index)
        for i, (q, g) in enumerate(cov[~sm].groupby("q")):
            g = g.sort_values("n")
            role = f"cat:{i}"
            pc.series.append(Series("line", g.n.to_numpy(float), 100 * g.p_max_exceeds
                                    .to_numpy(float), role=role,
                                    label=f"P{100 * q:.0f} quantile"))
            need = cov[sm & (cov.q == q)]
            if not need.empty:
                nn = float(need.n.iloc[0])
                yn = 100 * _f(need.p_max_exceeds.iloc[0])
                pc.series.append(Series("scatter", np.array([nn]), np.array([yn]),
                                        role=role, style={"ms": "d", "size": 1.4}))
                # the n right under its own diamond (it sat detached below the line)
                pc.series.append(Series("text", np.array([nn]), np.array([yn - 4.0]),
                                        text=[f"{nn:.0f}"], role=role,
                                        style={"anchor": "below"}))
        if np.isfinite(conf):
            # reference lines unlabelled in the plot (the labels sat on the curves); the
            # caption names them
            pc.series.append(Series("hline", y=np.array([100 * conf]), role="reference"))
        pc.series.append(Series("vline", x=np.array([float(len(pids))]), role="threshold"))
    else:
        _empty(pc, NOT_STORED)
    n_rows = max(len(pids), 4)
    pa.height = 0.62
    spec = FigureSpec("req_envelope", "Dynamic requirement envelope", [pa, pb, pc], rows=1,
                      cols=3, width="text", height_in=B.rows_height(n_rows, per=0.13,
                                                                     extra=0.4),
                      sources=["stats_kinematics:subjects", "requirements:requirements",
                               "requirements:max_coverage"])
    return attach(spec, CaptionSpec(
        "Dynamic requirements beyond the self-paced opening speed. (a) Per participant the "
        "largest opening and closing angular velocity of the open-close task and of the free "
        "motion, with the design value where it is a requirement. (b) Per participant the "
        "largest absolute angular acceleration of the opening angle over open-close, chewing, "
        "and free motion (coloured by the task where it occurred) and its 99th percentile, the "
        "robustness value a single sample cannot decide. (c) Coverage: the chance that the largest of n "
        "participants exceeds the population quantile, 1 − qⁿ, for the 80th and 90th "
        "percentile: what more participants would buy",
        [("grey dots in (a)", "participants"), ("black tick in (a)", f"P{p95:.0f}"),
         ("black line in (a)", "from the one-sided lower 95 % bound to it"),
         ("dots in (b)", "largest value, colour by task"),
         ("grey squares in (b)", "99th percentile"),
         ("dashed line and shading in (b)", f"P{p95:.0f} and its lower bound"),
         ("diamonds in (c)", "smallest n reaching the confidence line (n printed below)"),
         ("dashed horizontal line in (c)", f"{B.num(100 * conf, 0)} % confidence"),
         ("dashed vertical line in (c)", "this cohort's n")],
        f"n = {len(pids)} participants",
        f"P{p95:.0f} across participants (Harrell-Davis) with the one-sided lower order "
        "statistic; (c) closed form, no data",
        evidence="post hoc description; lower bounds of the 40 Hz optics with the pose "
                 "low-pass",
        notes=([] if not missing else
               ["Rows without values (" + ", ".join(missing) + "): " + NEXT_RUN + "."])
        + ([] if pb.series else ["(b): the angular acceleration is " + NEXT_RUN + "."])
        + ([] if pc.series else ["(c): the coverage table is " + NEXT_RUN + "."]),
        codes=["X7", "X8", "J4.2", "J4.3", "J2.5"]))


# == J2.5 A3: the noise nulls and the condyle plausibility sensitivity =================================
NULLS = (("bench", "best-fit", "bench noise"),
         ("invivo", "best-fit", "in-vivo noise"),
         ("invivo_slow", "best-fit", "slow rest noise"),
         ("invivo_motion", "best-fit", "motion-scaled noise"),
         ("invivo_teeth", "best-fit", "teeth-only fit"),
         ("bench_tmj", "condyle", "bench noise"),
         ("invivo_tmj", "condyle", "in-vivo noise"))
SENS_METRICS = {"condyle_left_disp_mm": "excursion, left condyle",
                "condyle_right_disp_mm": "excursion, right condyle",
                "hinge_rms_incisor_hinge_tmj": "condyle-hinge residual",
                "hinge_rms_incisor_hinge_fit": "best-fit hinge residual",
                "rms_condyles_mm": "condyle residual, hinge + transl.",
                "dist_tmj": "helical axis to condyles"}
SENS_GROUPS = (("all", "cat:0", "all participants"),
               ("plausible_condyles", "cat:1", "plausible condyles"),
               ("plausible_condyles_30mm", "cat:2", "plausible, depth limit lower"),
               ("plausible_condyles_50mm", "cat:3", "plausible, depth limit higher"))


@register("kin_null_variants", "paper", "Hinge residual against every noise null")
def kin_null_variants(req: ViewRequest) -> FigureSpec:
    """(a) Per participant the hinge residual over the 99th percentile of each noise-only null;
    (b) condyle-based outcomes for all participants and for the plausible condyles."""
    ctx = req.ctx
    pids = ctx.pids()
    pl = plausible(ctx)
    hn = _t(ctx, "uncertainty", "hinge_noise")
    hn = hn.drop_duplicates("pid").set_index("pid").reindex(pids) if not hn.empty else hn
    tests = _t(ctx, "stats_kinematics", "hinge_noise_tests")
    nulls = [n for n in NULLS if not hn.empty and f"ratio_{n[0]}" in hn]
    pa = Panel("(a) Residual over noise", Axis("Observed / noise-only P99", log=True),
               Axis(""), [], legend="below")
    okm = np.array([pl.get(p) is True for p in pids])
    first = True
    for i, (k, model, lab) in enumerate(nulls):
        v = hn[f"ratio_{k}"].to_numpy(float)
        jit = np.linspace(-0.16, 0.16, len(v)) if len(v) > 1 else np.zeros(len(v))
        for mask, role, ms, name in ((okm, "context", "o", "participants, plausible condyles"),
                                     (~okm, "flagged", "v", "participants, flagged condyles")):
            m = mask & np.isfinite(v)
            if m.any():
                pa.series.append(Series("scatter", v[m], i + jit[m] + 0.12, role=role,
                                        style={"ms": ms, "size": 0.8},
                                        label=name if first else None))
        first = False
        for g, role, off, glab in (("all", "cat:0", -0.22, "median, all [95 % CI]"),
                                   ("plausible_condyles", "cat:1", -0.34,
                                    "median, plausible condyles [95 % CI]")):
            r = tests[(tests.noise == k) & (tests.group == g)] if not tests.empty else tests
            if r.empty:
                continue
            r = r.iloc[0]
            if np.isfinite(_f(r.noise_ratio_ci_lo)) and np.isfinite(_f(r.noise_ratio_ci_hi)):
                pa.series.append(Series("line", np.array([r.noise_ratio_ci_lo,
                                                          r.noise_ratio_ci_hi]),
                                        np.array([i + off] * 2), role=role,
                                        style={"width": 1.3}))
            pa.series.append(Series("scatter", np.array([_f(r.noise_ratio_median)]),
                                    np.array([i + off]), role=role,
                                    style={"ms": "d", "size": 1.3},
                                    label=glab if i == 0 else None))
    pa.series.append(Series("vline", x=np.array([1.0]), role="reference"))
    counts = []
    for k, _, _ in nulls:
        r = tests[(tests.noise == k) & (tests.group == "all")] if not tests.empty else tests
        counts.append(f" ({int(r.above_n.iloc[0])}/{int(r.n_tested.iloc[0])})"
                      if not r.empty else "")
    pa.y.ticks = [float(i) for i in range(len(nulls))]
    pa.y.ticklabels = [f"{model}: {lab}{c}" for (_, model, lab), c in zip(nulls, counts)]
    pa.y.lim = (-0.6, len(nulls) - 0.4)
    pa.y.reverse = True
    if not nulls:
        _empty(pa, "no noise nulls")
    # (b) condyle sensitivity
    cs = _t(ctx, "stats_kinematics", "condyle_sensitivity")
    metrics = [m for m in SENS_METRICS if not cs.empty and (cs.metric == m).any()]
    pb = Panel("(b) Condyle plausibility", Axis("Median", MM, log=True), Axis(""), [],
               legend="below")
    for gi, (g, role, lab) in enumerate(SENS_GROUPS):
        d = cs[cs.group == g].drop_duplicates("metric").set_index("metric").reindex(metrics) \
            if metrics else cs
        if d.empty or not d["median"].notna().any():
            continue
        y = np.arange(len(metrics)) + (gi - 1.5) * 0.17
        for yi, lo, hi in zip(y, d.ci_lo, d.ci_hi):
            if np.isfinite(lo) and np.isfinite(hi):
                pb.series.append(Series("line", np.array([lo, hi]), np.array([yi, yi]),
                                        role=role))
        pb.series.append(Series("scatter", d["median"].to_numpy(float), y, role=role,
                                label=f"{lab} (n = {int(np.nanmax(d.n))})"))
    pb.y.ticks = [float(i) for i in range(len(metrics))]
    pb.y.ticklabels = [SENS_METRICS[m] for m in metrics]
    pb.y.lim = (-0.6, len(metrics) - 0.4)
    pb.y.reverse = True
    if not metrics:
        _empty(pb, "no condyle sensitivity")
    from biomech.stats.cohort_kinematics import CONDYLE_DEPTH_NEIGHBOURS_MM

    depth = _signal(ctx, "kinematics", "min_condyle_posterior_mm")
    lo_d, hi_d = CONDYLE_DEPTH_NEIGHBOURS_MM
    spec = FigureSpec("kin_null_variants", "Hinge residual against every noise null", [pa, pb],
                      rows=1, cols=2, width="text",
                      height_in=B.rows_height(max(len(nulls), len(metrics)), per=0.26,
                                              extra=0.5),
                      sources=["uncertainty:hinge_noise", "stats_kinematics:hinge_noise_tests",
                               "stats_kinematics:condyle_sensitivity",
                               "stats_kinematics:subjects"])
    n_pl = sum(1 for v in pl.values() if v is True)
    return attach(spec, CaptionSpec(
        "(a) Per participant the RMS incisal residual of a fixed hinge divided by the 99th "
        "percentile of the residual a perfect hinge produces with noise only (above 1: more "
        "than noise), for the best-fit hinge under the bench and in-vivo rest noise and the "
        "three sensitivity nulls (slow rest noise, rest noise scaled to the motion, teeth-only "
        "fit), and for the hinge through the palpated condyles; in brackets: participants above "
        "their own noise of those tested. (b) Condyle-based outcomes for "
        "all participants, for those whose palpated condyles pass the plausibility check, and "
        f"with the depth limit of that check moved to {lo_d:g} and {hi_d:g} mm"
        + (f" (primary {depth:g} mm)" if np.isfinite(depth) else ""),
        [("grey dots in (a)", "participants with plausible condyles"),
         ("grey triangles in (a)", "participants with flagged condyles"),
         ("dashed line in (a)", "residual equal to the noise P99"),
         ("diamonds and lines", "median with its 95 % BCa CI, colour by participant group")],
        f"n = {len(pids)} participants, {n_pl} with plausible condyles",
        "per participant a Monte-Carlo null (99th percentile); medians with BCa CIs; exact "
        "Wilcoxon signed-rank test of the log ratio against 0",
        evidence="(a) in-vivo best-fit hinge pre-specified, the other nulls post hoc "
                 "sensitivities, the condyle hinge secondary (post hoc test); (b) post hoc "
                 "sensitivity",
        codes=["C1", "D24", "D25", "D29", "D33", "A3", "J2.5"]))


# == J2.5 A4: geometry against literature =============================================================
GEOMETRY = (("bonwill_side_mm", "Bonwill side", MM), ("bonwill_ratio", "Side / intercondylar",
                                                       ""),
            ("balkwill_deg", "Balkwill angle", DEG), ("intercondylar_mm", "Intercondylar", MM),
            ("condyle_height_mm", "Condyle height", MM),
            ("lower_arch_depth_mm", "Lower arch depth", MM),
            ("upper_arch_depth_mm", "Upper arch depth", MM),
            ("hinge_radius_mm", "Hinge radius", MM))


def literature(path: Any = None) -> dict[str, list[dict[str, Any]]]:
    """Literature reference values per metric from ``etc/literature.yaml`` (section
    ``geometry`` or the file itself); {} without the file."""
    import yaml
    from pathlib import Path

    p = Path(path) if path is not None else LITERATURE_PATH
    if not p.is_file():
        return {}
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    sec = doc.get("geometry", doc) if isinstance(doc, dict) else {}
    out = {}
    for k, v in (sec or {}).items():
        entries = v if isinstance(v, list) else [v]
        out[str(k)] = [e for e in entries if isinstance(e, dict)]
    return out


@register("kin_geometry_lit", "paper", "Jaw geometry against the literature")
def kin_geometry_lit(req: ViewRequest) -> FigureSpec:
    """Per geometry measure: every participant, the cohort median with its CI and, when
    ``etc/literature.yaml`` gives them, the literature value or range."""
    from biomech.stats.cohort_kinematics import CONDYLE_METRICS

    ctx = req.ctx
    pids = ctx.pids()
    s = _subjects(ctx)
    pl = plausible(ctx)
    lit = literature()
    okm = np.array([pl.get(p) is True for p in pids])
    panels, n_lit, refs = [], 0, []
    for col, lab, unit in GEOMETRY:
        v = _col(s, col, pids)
        # x label with the unit on every panel; the literature sources go to the caption, a
        # panel note would run over the data
        xlab = "Angle" if unit == DEG else ("Length" if unit == MM else "Ratio")
        p = Panel(lab, Axis(xlab, unit), Axis(""), [], legend="none")
        if not np.isfinite(v).any():
            panels.append(_empty(p))
            continue
        cond = col in CONDYLE_METRICS
        use = v[okm] if cond else v
        for e in lit.get(col, []):
            lo, hi, val = _f(e.get("low")), _f(e.get("high")), _f(e.get("value"))
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("hspan", x=np.array([lo]), y2=np.array([hi]),
                                       role="cat:4", style={"alpha": 0.25}))
            if np.isfinite(val):
                p.series.append(Series("vline", x=np.array([val]), role="cat:4"))
            tag = str(e.get("ref") or e.get("label") or "")
            if tag:
                refs.append((lab, tag))
            n_lit += 1
        jit = np.linspace(-0.25, 0.25, len(v))
        m_ok = (okm | (not cond)) & np.isfinite(v)
        p.series.append(Series("scatter", v[m_ok], jit[m_ok], role="context",
                               style={"size": 0.8}))
        m_fl = cond & ~okm & np.isfinite(v)
        if np.any(m_fl):
            p.series.append(Series("scatter", v[m_fl], jit[m_fl], role="flagged",
                                   style={"size": 1.05}))   # a triangle reads smaller than a dot
        lo, hi = B.median_ci(use, B.seed_of(ctx))
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("line", np.array([lo, hi]), np.array([0.0, 0.0]),
                                   role="accent", style={"width": 1.6}))
        if _fin(use).size:
            p.series.append(Series("scatter", np.array([float(np.median(_fin(use)))]),
                                   np.array([0.0]), role="accent",
                                   style={"ms": "d", "size": 1.5}))
        p.y.ticks, p.y.ticklabels, p.y.lim = [], [], (-0.6, 0.6)
        p.height = 0.45
        panels.append(p)
    rows, cols = _grid(len(panels), 2)
    spec = FigureSpec("kin_geometry_lit", "Jaw geometry against the literature", panels,
                      rows=rows, cols=cols, width="text", height_in=0.85,
                      sources=["stats_kinematics:subjects", "kinematics:geometry"])
    n_pl = int(okm.sum())
    marks = [("grey dots", "participants"),
             ("grey triangles", "flagged condyles (shown, not in the median of condyle-based "
                                 "measures)"),
             ("black diamond and line", "median with its 95 % BCa CI")]
    if n_lit:
        marks.append(("pink line or band", "literature value or range"))
    return attach(spec, CaptionSpec(
        "Jaw geometry per participant: the Bonwill triangle side (incisor to condyle point) and "
        "its ratio to the intercondylar width, the Balkwill angle, the intercondylar width, the "
        "condyle height above the occlusal plane, both arch depths, and the radius of the "
        "best-fit hinge"
        + (", with the literature values" if n_lit else
           "; literature values are added from the reference file when it exists"),
        marks, f"n = {len(pids)} participants, {n_pl} with plausible condyles (condyle-based "
               "medians)",
        "median with 95 % BCa bootstrap CI", evidence="descriptive",
        notes=["Occlusal plane and arch depths through the incisor and the second molars "
               "(the molar of Balkwill's occlusal plane)."] + (["Literature: " + "; ".join(
                   f"{lab}, {', '.join(t for m, t in refs if m == lab)}"
                   for lab in dict.fromkeys(m for m, _ in refs))] if refs else [])
        + ["Literature values depend on the condylar reference point; the condyle points "
           "here are skin landmarks over the lateral joint."],
        codes=["S1", "D21", "A4", "J2.5"]))


# == J2.5 C1: window-only null vs the observed lead ===================================================
def _timing_rows(ctx: Any) -> list[tuple[str, str, str]]:
    return [(t, d, r) for t, d in TIMING_ROWS for r, _ in RULES]


@register("timing_window_null", "paper", "EMG lead against the window-only null")
def timing_window_null(req: ViewRequest) -> FigureSpec:
    """Per participant the observed EMG lead and the lead the same detector finds at random
    burst positions in the same search window (the chance null), and their difference."""
    ctx = req.ctx
    tn = _t(ctx, "stats_timing", "timing_null")
    tc = _t(ctx, "stats_timing", "timing_cohort")
    rows = _timing_rows(ctx)
    labels = [f"{V.task_label(t)}, {RULE.get(r, r)}" for t, d, r in rows]
    obs, null, diff, pvals = [], [], [], []
    for t, d, r in rows:
        g = tn[(tn.task == t) & (tn.direction == d) & (tn.detector == DETECTOR)
               & (tn.rule == r)] if not tn.empty else tn
        obs.append(g.lead_ms.to_numpy(float) if not g.empty else np.empty(0))
        null.append(g.lead_null_ms.to_numpy(float) if not g.empty else np.empty(0))
        diff.append(g.lead_minus_null_ms.to_numpy(float) if not g.empty else np.empty(0))
        c = tc[(tc.task == t) & (tc.direction == d) & (tc.detector == DETECTOR)
               & (tc.rule == r) & (tc.metric == "lead_ms")] if not tc.empty else tc
        pvals.append(_f(c.p_null_holm.iloc[0]) if not c.empty and "p_null_holm" in c
                     else np.nan)
    pa = B.paired_strip(labels, obs, null, "(a) Observed lead and chance null",
                        Axis("Lead before motion onset", MSEC), label_a="observed",
                        label_b="window-only null", roles=("accent", "context"),
                        seed=B.seed_of(ctx), ref=0.0)
    pb = B.strip_ci(labels, diff, "(b) Observed minus null", Axis("Difference", MSEC),
                    ref=0.0, seed=B.seed_of(ctx))
    for i, p_ in enumerate(pvals):
        if np.isfinite(p_):
            pb.series.append(Series("text", np.array([np.nan]), np.array([float(i)]),
                                    text=[f"p {'< 0.001' if p_ < 0.001 else f'= {p_:.3f}'}"],
                                    style={"anchor": "left"}))
    xs = _fin(np.concatenate(diff)) if diff else np.empty(0)
    if xs.size:
        span = float(np.ptp(np.r_[xs, 0.0]) or 1.0)
        xr = float(max(xs.max(), 0.0)) + 0.55 * span
        for s_ in pb.series:
            if s_.kind == "text":
                s_.x = np.array([xr])
        pb.x.lim = (float(min(xs.min(), 0.0)) - 0.05 * span, xr + 0.02 * span)
    pb.y.ticklabels = [""] * len(labels)
    if not any(len(o) for o in obs):
        _empty(pa, "no chance-null leads")
        _empty(pb)
    pre = _f(B.cfg(ctx, "timing", "emg_pre_s"))
    post = _f(B.cfg(ctx, "timing", "emg_post_s"))
    win = (f" (−{1000 * pre:.0f} to +{1000 * post:.0f} ms around the onset)"
           if np.isfinite(pre) and np.isfinite(post) else "")
    n = max((len(_fin(o)) for o in obs), default=0)
    spec = FigureSpec("timing_window_null", "EMG lead against the window-only null", [pa, pb],
                      rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(rows), per=0.32, extra=0.5),
                      sources=["stats_timing:timing_null", "stats_timing:timing_cohort"])
    return attach(spec, CaptionSpec(
        "Lead of the digastric EMG onset (TKEO detector) over the mechanical opening onset per "
        "participant, in open-close and chewing under both onset rules, next to the chance "
        "null (shift null): the same detector applied to EMG circularly shifted at random "
        f"against the motion, inside the same asymmetric search window{win}, which reaches "
        "further before the onset than after it. (a) Observed and null lead per participant; "
        "(b) their "
        "difference. Random bursts already produce a large positive lead, so the null measures "
        "the window, not physiology, and has no power for a physiological lead shorter than the "
        "window bias",
        [("black dots and diamond", "observed lead per participant, median [95 % BCa CI]"),
         ("grey dots and diamond", "window-only null"),
         ("dashed line", "zero"),
         ("numbers in (b)", "Holm-adjusted p that the observed lead exceeds the null")],
        f"up to n = {n} participants per row (those with at least the minimum usable "
        "channel-cycles)",
        "per participant the median over cycles (observed) and over the null draws; median "
        "with BCa CI; one-sided Wilcoxon signed-rank test that the observed lead exceeds the "
        "null, Holm over the confirmatory rows",
        evidence="first-onset rule pre-specified, driving-burst rule post hoc; the "
                 "window-bias reading post hoc",
        codes=["C2", "D23", "D34", "C1", "J2.5"]))


# == J2.5 C1: single-cycle usability ==================================================================
@register("timing_single_cycle", "paper", "Single-cycle onset usability")
def timing_single_cycle(req: ViewRequest) -> FigureSpec:
    """Per participant: share of usable single-cycle onsets and of channels already active at
    the window start, and the usable count against the minimum-cycle gate."""
    ctx = req.ctx
    pids = ctx.pids()
    u = _t(ctx, "stats_timing", "timing_usability")
    kf = _f(B.cfg(ctx, "stats_timing", "min_cycles"))
    k = int(kf) if np.isfinite(kf) else 0
    rows = _timing_rows(ctx)
    labels = [f"{V.task_label(t)}, {RULE.get(r, r)}" for t, d, r in rows]
    usable, active, counts = [], [], []
    for t, d, r in rows:
        g = u[(u.task == t) & (u.direction == d) & (u.detector == DETECTOR) & (u.rule == r)] \
            if not u.empty else u
        g = g.drop_duplicates("pid").set_index("pid").reindex(pids) if not g.empty else g
        if g.empty:
            usable.append(np.empty(0))
            active.append(np.empty(0))
            counts.append(np.full(len(pids), np.nan))
            continue
        usable.append(100 * g.usable_frac.to_numpy(float))
        active.append(100 * g.active_at_start.to_numpy(float))
        counts.append(np.round(g.usable_frac.to_numpy(float) * g.n.to_numpy(float)))
    pa = B.paired_strip(labels, usable, active, "(a) Share of channel-cycles",
                        Axis("Channel-cycles", PCT), label_a="usable onset",
                        label_b="active at window start", roles=("accent", "cat:3"),
                        seed=B.seed_of(ctx))
    pb = Panel("(b) Usable channel-cycles", Axis("Usable channel-cycles"), Axis(""), [],
               legend="below")
    first_in, first_out = True, True
    for i, c in enumerate(counts):
        c = np.asarray(c, float)
        jit = np.linspace(-0.2, 0.2, len(c)) if len(c) > 1 else np.zeros(len(c))
        inn = np.isfinite(c) & (c >= k)
        out = np.isfinite(c) & (c < k)
        if inn.any():
            pb.series.append(Series("scatter", c[inn], i + jit[inn], role="accent",
                                    style={"size": 0.9},
                                    label="enters the cohort test" if first_in else None))
            first_in = False
        if out.any():
            pb.series.append(Series("scatter", c[out], i + jit[out], role="flagged",
                                    label="below the minimum" if first_out else None))
            first_out = False
        pb.series.append(Series("text", np.array([np.nan]), np.array([float(i)]),
                                text=[f"{int(inn.sum())}/{int(np.isfinite(c).sum())}"],
                                style={"anchor": "left"}))
    cmax = max((float(np.nanmax(c)) for c in counts if np.isfinite(c).any()), default=1.0)
    for s_ in pb.series:
        if s_.kind == "text":
            s_.x = np.array([cmax * 1.35])
    pb.x.lim = (-0.5, cmax * 1.37)
    if k:
        pb.series.append(Series("vline", x=np.array([k - 0.5]), role="threshold",
                                label=f"minimum {k} channel-cycles"))
    pb.y.ticks = [float(i) for i in range(len(rows))]
    pb.y.ticklabels = [""] * len(rows)
    pb.y.lim = (-0.6, len(rows) - 0.4)
    pb.y.reverse = True
    if u.empty:
        _empty(pa, "no usability table")
        _empty(pb)
    spec = FigureSpec("timing_single_cycle", "Single-cycle onset usability", [pa, pb], rows=1,
                      cols=2, width="text",
                      height_in=B.rows_height(len(rows), per=0.32, extra=0.5),
                      sources=["stats_timing:timing_usability"])
    return attach(spec, CaptionSpec(
        "Single-cycle usability of the digastric EMG onsets (TKEO detector) in open-close and "
        "chewing openings under both onset rules. (a) Per participant the share of "
        "channel-cycles with a usable onset and the share whose detector signal is already "
        "above threshold at the start of the search window (no onset identifiable). (b) Per "
        "participant the number of usable channel-cycles against the minimum a participant "
        "needs to enter the cohort test; numbers at the right: participants entering of those "
        "recorded",
        [("black dots and diamond in (a)", "usable share, median [95 % BCa CI]"),
         ("vermilion dots and diamond in (a)", "share active at the window start"),
         ("black dots in (b)", "participants entering"),
         ("grey triangles in (b)", "participants below the minimum"),
         ("dashed line in (b)", "the minimum")],
        f"n = {len(pids)} participants",
        "shares over a participant's channel-cycles; median with BCa CI",
        evidence="descriptive", codes=["C2", "D15", "C1", "J2.5"]))


# == J4.1: SPM in significance-bar style ==============================================================
SPM_FORMS = ("band", "difference")
SPM_ROLES = {"openers_vs_closers_open_close": ("group:openers", "group:closers"),
             "masseter_L_vs_R_chewing": ("entity:masseter_l", "entity:masseter_r")}


@register("emg_spm_sigbar", "paper", "SPM comparisons with significance bars",
          choices={"form": SPM_FORMS}, form="band")
def emg_spm_sigbar(req: ViewRequest) -> FigureSpec:
    """The confirmatory SPM comparisons (openers vs closers in open-close, masseter left vs right
    in chewing) as mean ± SD bands, with thin bars over the supra-threshold clusters."""
    from biomech.viz.views.results_emg import SPM_TESTS

    ctx = req.ctx
    form = req.options.get("form", "band")
    arr = _arr(ctx, "stats_emg", COHORT_UNIT)
    spm = _t(ctx, "stats_emg", "spm")
    conf = spm[spm.family == "confirmatory"] if not spm.empty and "family" in spm else spm
    n_fam = len(conf)
    panels, ns = [], []
    for i, test in enumerate(("openers_vs_closers_open_close", "masseter_L_vs_R_chewing")):
        key, title, (ka, kb), (la, lb), yax = SPM_TESTS[test]
        ra, rb = SPM_ROLES[test]
        row = spm[spm.test == test] if not spm.empty else spm
        cl = row.clusters.iloc[0] if not row.empty else None
        A = np.asarray(arr.get(f"{key}__{ka}", np.empty((0, 0))), float)
        Bm = np.asarray(arr.get(f"{key}__{kb}", np.empty((0, 0))), float)
        tag = "ab"[i]
        if form == "difference":
            p = Panel(f"({tag}) {title}", Axis("Cycle", PCT), Axis(f"{la} − {lb}", yax.unit),
                      [], legend="below")
        else:
            p = Panel(f"({tag}) {title}", Axis("Cycle", PCT), yax, [], legend="below")
        if A.ndim == 2 and Bm.ndim == 2 and A.size and A.shape == Bm.shape:
            x = np.linspace(0, 100, A.shape[1])
            ns.append(A.shape[0])
            if form == "difference":
                B.add_curves(p, x, A - Bm, centre="mean", label="mean difference",
                             role="accent", show_subjects=False)
                p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
                bar_role = "context"
            else:
                B.add_curves(p, x, A, centre="mean", label=la, role=ra, show_subjects=False)
                B.add_curves(p, x, Bm, centre="mean", label=lb, role=rb, show_subjects=False)
                bar_role = "accent"
            ph = _f(row.p_holm.iloc[0]) if not row.empty and "p_holm" in row else np.nan
            pt = ("" if not np.isfinite(ph) else
                  f", Holm p {'< 0.001' if ph < 0.001 else f'= {ph:.3f}'}")
            got = B.add_spm_bars(p, [("clusters" + pt, cl, bar_role)])
            if not got:
                p.note = "no supra-threshold cluster" + pt
        else:
            _empty(p, "test inputs not stored")
        panels.append(p)
    spec = FigureSpec("emg_spm_sigbar", "SPM comparisons with significance bars", panels,
                      rows=1, cols=2, width="text", height_in=1.6,
                      sources=["stats_emg:spm", "stats_emg:arrays"])
    marks = ([("lines and bands", "mean ± SD across participants, colour by muscle group "
                                  "(openers vermilion, closers blue) or side (left orange "
                                  "dashed, right blue)")]
             if form != "difference" else
             [("line and band", "mean ± SD of the per-participant difference"),
              ("dashed line", "no difference")])
    marks.append(("thin bar above the curves", "supra-threshold cluster"))
    return attach(spec, CaptionSpec(
        "The two confirmatory waveform comparisons over the time-normalised cycle: (a) opener "
        "(digastric) vs closer (masseter, temporalis) activation in open-close, each "
        "participant's mean cycle with the peak normalised to 1; (b) left vs right masseter "
        "activation in chewing"
        + (", shown as the paired difference" if form == "difference" else ""),
        marks, f"n = {max(ns) if ns else 0} participants (participant means)",
        "mean ± SD; " + B.spm_caption("paired SPM{t}, two-tailed",
                                      f"Holm over the {n_fam} confirmatory tests"
                                      if n_fam else ""),
        evidence="pre-specified", codes=["S2", "J4.1", "J2.5"]))


# == J4.1: waveform similarity and cycles needed =======================================================
@register("var_cmc_cycles", "paper", "Within- vs between-participant similarity")
def var_cmc_cycles(req: ViewRequest) -> FigureSpec:
    """(a) Coefficient of multiple correlation of the opening-angle waveform within each
    participant and between participant means; (b) share of participants whose cycle mean is
    within half the minimal detectable change after n cycles."""
    from biomech.viz.views.results_emg import cmc_ci

    ctx = req.ctx
    seed = B.seed_of(ctx)
    try:
        cm = cmc_ci(ctx, seed=seed)
    except Exception:
        cm = None
    spm = _t(ctx, "stats_emg", "spm")
    row = spm[spm.test == "cmc_opening_angle"] if not spm.empty and "test" in spm else spm
    pa = Panel("(a) Opening-angle waveform", Axis("CMC"), Axis(""), [], legend="below")
    summary: list[tuple[str, float, float, float, str]] = []
    pids = ctx.pids()
    nb = 0
    if cm is not None:
        pids = [p for p in ctx.pids() if p in cm["within"]]
        nb = int(cm["n_boot"])
        summary = [("median within", *cm["median_within"], "accent"),
                   ("between means", *cm["between"], "cat:4")]   # not (b)'s orange (2026-11-05)
    elif not row.empty:
        summary = [("median within", _f(row.cmc_within_median.iloc[0]), np.nan, np.nan,
                    "accent"),
                   ("between means", _f(row.cmc_between.iloc[0]), np.nan, np.nan, "cat:4")]
        pa.note = "per-participant values not reproducible from the stored profiles"
    rows_pa = (pids if cm is not None else []) + [lab for lab, *_ in summary]
    if cm is not None:
        for i, p in enumerate(pids):
            est, lo, hi = cm["within"][p]
            if np.isfinite(lo) and np.isfinite(hi):
                pa.series.append(Series("line", np.array([lo, hi]), np.array([i, i], float),
                                        role="context", style={"width": 1.0}))
        pa.series.append(Series("scatter", np.array([cm["within"][p][0] for p in pids], float),
                                np.arange(len(pids), dtype=float), role="context",
                                label="within a participant [95 % CI]"))
    for j, (lab, est, lo, hi, role) in enumerate(summary):
        yv = float(len(rows_pa) - len(summary) + j)
        if np.isfinite(lo) and np.isfinite(hi):
            pa.series.append(Series("line", np.array([lo, hi]), np.array([yv, yv]), role=role,
                                    style={"width": 1.6}))
        pa.series.append(Series("scatter", np.array([est]), np.array([yv]), role=role,
                                style={"ms": "d", "size": 1.5}))
    if summary:
        pa.series.insert(0, Series("hline", y=np.array([len(rows_pa) - len(summary) - 0.5]),
                                   role="faint"))
        pa.y.ticks = [float(i) for i in range(len(rows_pa))]
        pa.y.ticklabels = (B.pseudo_rows(ctx, pids) if cm is not None else []) + \
            [lab for lab, *_ in summary]
        pa.y.lim = (-0.6, len(rows_pa) - 0.4)
        pa.y.reverse = True
    else:
        _empty(pa, "no waveform similarity")
    cn = _t(ctx, "variability", "cycles_needed")
    rel = _t(ctx, "stats_kinematics", "reliability")
    pb = Panel("(b) Cycles needed", Axis("Cycles"),
               Axis("Participants within target", PCT, lim=(0.0, 105.0)), [], legend="below")
    keys = list(dict.fromkeys(zip(cn.task, cn.metric, cn.side.fillna("")))) if not cn.empty \
        else []
    n_max, ci = 1, -1
    from biomech.viz.views.results_kinematics import _label

    for task, metric, side in keys:
        g = cn[(cn.task == task) & (cn.metric == metric) & (cn.side.fillna("") == side)]
        nm = int(np.nanmax(g.n_available)) if g.n_available.notna().any() else 1
        n_max = max(n_max, nm)
        x = np.arange(1, nm + 1, dtype=float)
        n = g.n_needed.to_numpy(float)
        yv = 100 * np.array([(n <= q).sum() for q in x]) / max(len(n), 1)
        r = rel[(rel.task == task) & (rel.metric == metric) & (rel.side.fillna("") == side)] \
            if not rel.empty else rel
        u = style.plain_unit(V.unit(metric)) if V.unit(metric) else ""
        mdc = (f", MDC95 {_f(r.mdc95.iloc[0]):.1f}" + ("" if not u else u if u == "°"
                                                       else f" {u}")
               if not r.empty else "")                    # with its unit
        ml = _label(metric)
        lab = f"{V.task_label(task)}{' ' + side if side else ''}, {ml[:1].lower()}{ml[1:]}{mdc}"
        if side != "right":
            ci += 1
        pb.series.append(Series("line", x, yv, role=f"cat:{ci % 6}" + (
            ":alt" if side == "right" else ""), label=lab))
    if not keys:
        _empty(pb, "no cycles-needed table")
    pb.x.lim = (0.5, n_max + 0.5)
    step = 1 if n_max <= 8 else 2                         # whole cycles only
    pb.x.ticks = [float(v) for v in range(step, n_max + 1, step)]
    spec = FigureSpec("var_cmc_cycles", "Within- vs between-participant similarity", [pa, pb],
                      rows=1, cols=2, width="text",
                      height_in=B.rows_height(len(rows_pa), per=0.13, extra=0.4),
                      sources=["stats_emg:spm", "emg:arrays", "variability:cycles_needed",
                               "stats_kinematics:reliability"])
    return attach(spec, CaptionSpec(
        "(a) Similarity of the open-close opening-angle waveform: the coefficient of multiple "
        "correlation (CMC) over each participant's cycles and the CMC of the participant mean "
        "curves; similar values within and between mean the same shape in every participant, "
        "so differences lie in amplitude and timing, not shape. (b) Share of participants "
        "whose mean over n cycles has a 95 % CI half-width of at most half the minimal "
        "detectable change (MDC95), per kinematic measure",
        [("grey dots and lines in (a)", "within-participant CMC with its 95 % CI"),
         ("black diamond in (a)", "median of the within-participant CMC with its CI"),
         ("orange diamond in (a)", "CMC between the participant means with its CI"),
         ("lines in (b)", "measures (laterotrusion to the right dashed)")],
        f"n = {len(pids)} participants",
        (f"CMC with 95 % bootstrap CIs ({nb} resamples of cycles within participants); "
         if nb else "CMC; ") + "share of participants reaching the target",
        evidence="pre-specified (variability); the cycles-needed reading descriptive",
        notes=["The target is half the MDC95, which is itself set by the cycle-to-cycle "
               "variation: a noisy measure has a wide target and reaches it after as few cycles "
               "as a precise one, so (b) shows when a mean is stable relative to its own noise "
               "floor, not its absolute precision."],
        codes=["S2", "J4.1", "J2.5"]))


# == J4.5: reference values against the literature =================================================
def literature_entries(path: Any = None) -> pd.DataFrame:
    """The normative values of ``etc/literature.yaml`` (section ``entries``; J7) as a
    table; empty without the file."""
    import yaml
    from pathlib import Path

    p = Path(path) if path is not None else LITERATURE_PATH
    if not p.is_file():
        return pd.DataFrame()
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return pd.DataFrame([e for e in (doc.get("entries") or []) if isinstance(e, dict)])


def _ms(mean: float, sd: float, lo: float, hi: float, d: int = 1) -> str:
    out = f"{mean:.{d}f}" if np.isfinite(mean) else ""
    if np.isfinite(sd):
        out += f" ± {sd:.{d}f}"
    if np.isfinite(lo) and np.isfinite(hi):
        out += (" " if out else "") + f"({lo:.{d}f}–{hi:.{d}f})"
    return out.replace("-", "−") or "–"


@register("table_reference_literature", "paper", "Reference values against the literature")
def table_reference_literature(req: ViewRequest) -> TableSpec:
    """Each dataset reference value that the literature reports with a matching name: the
    cohort mean ± SD next to every verified literature value (mean ± SD, range, n, population,
    method, source), definitions compared as the ``method`` column states (J4.5). The
    BibTeX key of each source row stays in the table's JSON (``_key``) for the manuscript."""
    from biomech.viz.spec import Column

    ctx = req.ctx
    lit = literature_entries()
    rv = ctx.table("stats_kinematics", "reference_values")
    rows: list[dict[str, Any]] = []
    n = 0
    for q in (lit["quantity"].drop_duplicates() if not lit.empty else []):
        r = rv[(rv.metric == q) & (rv.group == "all")] if not rv.empty else rv
        if r.empty:
            continue
        r = r.iloc[0]
        n = max(n, int(_f(r.n)) if np.isfinite(_f(r.n)) else 0)
        e_q = catalog.get(q)
        u = style.plain_unit(e_q.unit) if e_q.unit else ""
        rows.append({"_head": True, "what": e_q.label + (f" ({u})" if u else "")})
        rows.append({"what": "this dataset (task-based)", "value": _ms(_f(r["mean"]), _f(r.sd),
                                                                     np.nan, np.nan),
                     "n": f"{int(_f(r.n))}", "method": "self-paced tasks, optical tracking",
                     "source": "this study"})
        for _, e in lit[lit.quantity == q].iterrows():
            if not any(np.isfinite(_f(e.get(k))) for k in ("mean", "lo", "hi")):
                continue                                 # a source with no value for it
            sex = str(e.get("sex") or "all")
            rows.append({"what": f"{e.get('population', '')}" + (f", {sex}" if sex != "all"
                                                                 else ""),
                         "value": _ms(_f(e.get("mean")), _f(e.get("sd")), _f(e.get("lo")),
                                      _f(e.get("hi"))),
                         "n": "" if pd.isna(e.get("n")) else f"{int(e['n'])}",
                         "method": tidy(str(e.get("method") or "")),
                         "source": cite_label(str(e.get("key") or "")),
                         "_key": str(e.get("key") or "")})
    if not rows:
        rows = [{"what": "no literature value matches a stored reference value"}]
    cols = [Column("what", "Population / quantity", align="X"),
            Column("value", "Mean ± SD (range)", align="l"), Column("n", "n", align="l"),
            Column("method", "Method", align="X"), Column("source", "Source", align="l")]
    t = TableSpec("table_reference_literature", "", cols, rows, width="text",
                  notes="Definitions differ between sources (maximal vs self-paced, straight-line "
                        "vs vertical component); compare a value only with a matching method. "
                        "The dataset's values are task-based lower bounds, not maximal-effort "
                        "range of motion. Sources as first author and year.",
                  sources=["stats_kinematics:reference_values", "etc:literature"])
    return attach(t, CaptionSpec(
        "reference values of this dataset next to the published values of the same quantity: "
        "workspace, opening angle, and chewing, with population, method, and source of each "
        "published value",
        n=f"n = {n} participants in this dataset; published n per row",
        statistic="mean ± standard deviation (SD); range where the source gives one",
        evidence="descriptive"))


# == C1 framed as kinematic compatibility (JNER draft Table 3, 2026-11-05) ========================
def _cited_systems_hinge(task: str = "chewing") -> tuple[float, str]:
    """The best-fit hinge residual of P1 on the Qualisys reference for ``task`` from the pinned
    systems snapshot (the dataset has no chewing hinge fit): (value mm, snapshot) or NaN."""
    import yaml

    from biomech.paths import repo_root
    from biomech.viz.numbers import open_cited

    root = repo_root()
    try:
        snap = str(yaml.safe_load((root / "experiments/jtsmm_validation/paper_systems.yaml")
                                  .read_text())["snapshot"])
        h = open_cited(str(root / "experiments/jtsmm_validation/study.yaml"), snap) \
            .table("ref_hinge", "hinge")
        r = h[(h.pid == "s10") & (h.task == task) & (h.system == "qualisys")]
        return (float(r.hinge_fit_mm.iloc[0]) if not r.empty else np.nan), snap
    except Exception:                          # no systems store on this machine
        return np.nan, ""


COMPAT_FORMS = ("own", "with_chewing")


@register("table_compatibility", "paper", "Kinematic compatibility of exoskeleton structures",
          choices={"form": COMPAT_FORMS}, form="own")
def table_compatibility(req: ViewRequest) -> TableSpec:
    """How far each candidate exoskeleton structure would misalign the device at the incisor:
    a hinge at the joint (through the palpated condyle points, D33), one compromise axis (the
    best-fit hinge, C1), the generic rotation-translation coupling (X9) and a passive
    translation (D10), with the count of participants beyond their own in-vivo noise where it
    is tested. Form ``own`` (default) uses this data set only (a self-contained paper table);
    ``with_chewing`` adds the chewing row cited from the validation study's Qualisys
    reference (one participant)."""
    from biomech.viz.spec import Column

    ctx = req.ctx
    chewing = req.options.get("form", "own") == "with_chewing"
    rq = _t(ctx, "requirements", "requirements")
    tests = _t(ctx, "stats_kinematics", "hinge_noise_tests")
    hs = _t(ctx, "stats_kinematics", "hinge_noise_summary")
    cs = _t(ctx, "stats_kinematics", "coupling_summary")

    def row(key: str) -> dict[str, Any]:
        r = rq[rq.key == key] if not rq.empty else rq
        return r.iloc[0].to_dict() if not r.empty else {}

    from biomech.viz.views.paper import req_decimals

    def est(r: dict[str, Any]) -> str:
        # the rounding of the requirements table and the text (one decimal below 20)
        v = _f(r.get("value"))
        return B.est_ci(v, _f(r.get("ci_lo")), _f(r.get("ci_hi")), req_decimals(v),
                        adaptive=False) if np.isfinite(v) else "--"

    def ratio(med: Any, lo: Any, hi: Any) -> str:
        m = _f(med)
        return B.est_ci(m, _f(lo), _f(hi), 2, adaptive=False) if np.isfinite(m) else "--"

    tmj, fit, tr = row("misalign_hinge_tmj_mm"), row("misalign_hinge_fit_mm"), \
        row("translation_mm")
    t = tests[(tests.noise == "invivo_tmj") & (tests.group == "all")] \
        if not tests.empty and "noise" in tests else tests
    tmj_n = (f"{int(t.above_n.iloc[0])} of {int(t.n_tested.iloc[0])}" if not t.empty else "--")
    fit_n = (f"{int(hs.above_invivo_n.iloc[0])} of "
             f"{int(hs.n_tested_invivo.iloc[0] if 'n_tested_invivo' in hs else hs.n.iloc[0])}"
             if not hs.empty and "above_invivo_n" in hs else "--")
    tmj_r = (ratio(t.noise_ratio_median.iloc[0], t.noise_ratio_ci_lo.iloc[0],
                   t.noise_ratio_ci_hi.iloc[0]) if not t.empty and "noise_ratio_median" in t
             else "--")
    fit_r = (ratio(hs.ratio_invivo_median.iloc[0], hs.ratio_invivo_ci_lo.iloc[0],
                   hs.ratio_invivo_ci_hi.iloc[0]) if not hs.empty
             and "ratio_invivo_median" in hs else "--")
    chew, snap = _cited_systems_hinge("chewing") if chewing else (np.nan, "")
    gen = _f(pd.to_numeric(cs.generic_rms_mm, errors="coerce").median()) \
        if not cs.empty and "generic_rms_mm" in cs else np.nan
    tr_v = _f(tr.get("value")) if tr else np.nan
    rows = [
        {"s": "Hinge at the joint (through the condyle points)", "m": est(tmj),
         "r": tmj_r, "n": tmj_n, "e": "exploratory (post hoc)"},
        {"s": "One compromise axis (best-fit hinge), open-close", "m": est(fit),
         "r": fit_r, "n": fit_n, "e": "confirmatory"},
        *([{"s": "One compromise axis, chewing (Qualisys)",
            "m": B.num(chew, 2) if np.isfinite(chew) else "--", "r": "--", "n": "--",
            "e": "companion validation study, one participant (cited)"}] if chewing else []),
        {"s": "Generic rotation–translation coupling", "m": B.num(gen, 1) + " (RMS)"
         if np.isfinite(gen) else "--", "r": "--", "n": "--", "e": "exploratory (planned)"},
        {"s": "Passive translation", "m": "none by design, must permit "
         + (B.num(tr_v, req_decimals(tr_v)) if np.isfinite(tr_v) else "--") + " mm",
         "r": "--", "n": "--", "e": "descriptive (design value, P95)"}]
    cols = [Column("s", "Structure", align="X"),     # all wrap: fixed l columns starve (a)
            Column("m", "Misalignment at the incisor (mm)", align="X"),
            Column("r", "Ratio to the noise (median)", align="X"),
            Column("n", "Beyond the participant's noise", align="X"),
            Column("e", "Evidence", align="X")]
    n = int(_f(tmj.get("n"))) if tmj and np.isfinite(_f(tmj.get("n"))) else len(ctx.pids())
    spec = TableSpec("table_compatibility", "", cols, rows, width="text",
                     # the columns are defined in the caption (2026-11-13: the note repeated
                     # them); the note keeps what the caption does not say
                     notes=("Chewing: best-fit hinge on the Qualisys data of the companion "
                            "validation study, one participant. " if chewing else "")
                           + "--: not tested.",
                     sources=["requirements:requirements", "stats_kinematics:hinge_noise_tests",
                              "stats_kinematics:hinge_noise_summary",
                              "stats_kinematics:coupling_summary"]
                     + ([f"ref_hinge:hinge (jtsmm_validation, snapshot {snap})"] if snap else []))
    return attach(spec, CaptionSpec(
        "misalignment at the incisor that each candidate exoskeleton structure would impose "
        "on the device during opening, and whether it exceeds the participant's own "
        "measurement noise",
        n=f"n = {n} participants" + (" (chewing row: n = 1)" if chewing else ""),
        statistic="per participant the RMS residual of the structure's best fit to the "
                  "incisal path; median with 95 % BCa CI; Monte-Carlo in-vivo noise null "
                  "(99th percentile) per participant",
        # the evidence column carries it per row; "Mixed (per row)." added nothing (paper
        # agent 2026-11-10)
        codes=["C1", "D33", "X9", "D10"]))


# == J2.5 A1: label validity, compact =============================================================
KAPPA_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")


@register("table_label_kappa", "paper", "Intent label validity")
def table_label_kappa(req: ViewRequest) -> TableSpec:
    """Cohen's kappa of the automatic intent labels against a second, independent rule:
    overall (sampled windows), without the rest phase, without the frames near a transition,
    and per task, for the fixed thresholds and the thresholds raised to the rest-noise floor
    (``labels_check:kappa``; the complete table is ``tab_labels_check__kappa``)."""
    from biomech.viz.spec import Column

    k = req.ctx.table("labels_check", "kappa")
    variants = [v for v in ("fixed", "rest_floor") if not k.empty and v in set(k.variant)]

    def get(v: str, level: str, scope: str | None = None, phase: str | None = None
            ) -> pd.Series | None:
        q = k[(k.variant == v) & (k.level == level)]
        if scope is not None:
            q = q[q.scope == scope]
        if phase is not None:
            q = q[q.phase == phase]
        return q.iloc[0] if len(q) else None

    def cell(r: pd.Series | None, spread: bool = False) -> str:
        """Participant level first (J22.4): the median (range) of the participant
        kappas, the pooled kappa in brackets; rows without participant values: pooled only."""
        if r is None or not np.isfinite(_f(r.kappa)):
            return "–"
        if spread and np.isfinite(_f(r.get("kappa_min_participants"))):
            return (f"{_f(r.kappa_median_participants):.2f} "
                    f"({_f(r.kappa_min_participants):.2f}–{_f(r.kappa_max_participants):.2f}; "
                    f"pooled {_f(r.kappa):.2f})")
        return f"{_f(r.kappa):.2f} (pooled)"

    def task_cell(v: str, t: str) -> str:
        q = k[(k.variant == v) & (k.level == "participant_phase") & (k.phase == t)]["kappa"]
        q = pd.to_numeric(q, errors="coerce").dropna()
        if q.empty:
            return "–"
        r = get(v, "phase", "phase", t)
        pooled = f"; pooled {_f(r.kappa):.2f}" if r is not None else ""
        return f"{q.median():.2f} ({q.min():.2f}–{q.max():.2f}{pooled})"

    rows: list[dict[str, Any]] = []
    for lab, args, spread in (("All phases, sampled windows", ("overall_sampled", "all_phases"),
                               True),
                              ("Without the rest phase", ("overall_sampled", "without_rest"), True),
                              ("Without frames near a transition", ("boundary_tolerant", None,
                                                                    "pooled"), False)):
        rows.append({"what": lab, **{v: cell(get(v, *args), spread) for v in variants}})
    rows.append({"_rule": True})
    for t in KAPPA_TASKS:
        rows.append({"what": V.task_label(t), **{v: task_cell(v, t) for v in variants}})
    r0 = get("fixed", "overall_sampled", "all_phases") if "fixed" in variants else None
    n = int(_f(r0.n_participants)) if r0 is not None and np.isfinite(_f(r0.n_participants)) \
        else 0
    cols = [Column("what", "Frames compared", align="X")] + [
        Column(v, {"fixed": "Fixed thresholds",
                   "rest_floor": "Thresholds at the rest-noise floor"}[v], align="l")
        for v in variants]
    t = TableSpec("table_label_kappa", "", cols, rows, width="text",
                  notes="Kappa: Cohen's kappa of the stored labels against the second rule on "
                        "random 2-s windows, per participant; cells: median over the "
                        "participants (range; the kappa pooled over all participants' frames). "
                        "The kappa without the frames near a transition exists only pooled "
                        "over frames; compare it with the pooled kappa without the rest phase. "
                        "The rest phase is labelled as a whole, so its kappa is 0 by "
                        "construction and the rows without it are the informative ones.",
                  sources=["labels_check:kappa"])
    return attach(t, CaptionSpec(
        "validity of the automatic intent labels (the ground truth of the decoding): agreement "
        "with a second, independent labelling rule, overall, without the rest phase, without "
        "the frames near a class transition, and per movement task",
        n=f"n = {n} participants", statistic="Cohen's kappa per participant; median and range "
                                             "over the participants, the kappa pooled over "
                                             "frames in brackets (only pooled near transitions)",
        evidence="descriptive (label validity)"))
