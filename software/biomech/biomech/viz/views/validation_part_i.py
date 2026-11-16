# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper views of the validation additions (metrology order, I4).

* ``val_budget``, ``val_table_budget``, ``val_table_marking_gap`` (VD22): the error budget
  per participant -- the observed marking and motion errors split in quadrature into the
  reference part and the JTS-MM part -- and the marking frame-error gap between the
  participants with its candidate causes (stage ``ref_budget``);
* ``val_hinge``, ``val_table_hinge`` (VD21): the dataset's hinge models fitted to the same
  openings on JTS-MM and on Qualisys (stage ``ref_hinge``; fitness for purpose);
* ``val_emg_side``, ``val_table_emg_side`` (VD23): the EMG side check of the JTS-MM montage
  against Delsys (stage ``ref_emg_side``).

n = 2: every value per participant; the two-participant mean in the tables is descriptive
(no interval, no test). Stored results only; pseudonyms only.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from biomech import catalog
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import _empty, _table, _who, n_text
from biomech.viz.views.validation import pshape_text, shape_by_rows

G = {"group": "validation", "domain": "validation"}
DEG, MM = "\\degree", "\\milli\\metre"
UNIT = {"mm": MM, "deg": DEG}
UNIT_TEXT = {"mm": "mm", "deg": "°"}
EVIDENCE = "post hoc (descriptive)"
MEAN_NOTE = "the mean of the participants is descriptive (no interval, no test)"


def _n2(ctx: Any, pids: Any) -> str:
    """n of the shown participants, with the note on their descriptive mean."""
    return f"{n_text(ctx, pids)}; {MEAN_NOTE}"
# (label, role, marker) of the three budget terms
TERMS = {"sigma_obs": ("observed (σ_obs)", "muted", None),
         "sigma_jts_corrected": ("JTS-MM part (σ_JTS)", "system:jtsmm", None),
         "sigma_ref": ("Qualisys part (σ_Q)", "system:qualisys", "d")}


def _pids(d: pd.DataFrame) -> list[str]:
    return sorted(d["pid"].unique()) if "pid" in d and len(d) else []


def _mean_row(rows: list[dict[str, Any]], first: str, keys: list[str], label: str
              ) -> dict[str, Any]:
    """The descriptive mean of the numeric ``keys`` over ``rows`` (``first`` = row label)."""
    out: dict[str, Any] = {first: label}
    for k in keys:
        v = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float, np.floating))]
        out[k] = float(np.mean(v)) if len(v) == len(rows) and v else np.nan
    return out


# == VD22: error budget ============================================================================
def budget_lines(b: pd.DataFrame) -> pd.DataFrame:
    """One line per participant, part and outcome of the budget figure and table: the marking
    rows, rest, and the median over the movement tasks of each term (σ_obs, σ_Q, σ_JTS and
    the reference share, each its own median; σ_JTS over the tasks where the JTS-MM part is
    resolvable, ``separable``). ``n_rows`` / ``n_separable``: rows behind the line and those
    with a resolvable JTS-MM part (σ_JTS NaN when there is none)."""
    cols = ["sigma_obs", "sigma_ref", "sigma_jts_corrected", "ref_var_share"]
    b = b.assign(n_rows=1, n_separable=b["separable"].astype(int) if "separable" in b
                 else np.isfinite(b["sigma_jts_corrected"].to_numpy(float)).astype(int))
    keep = b[b["kind"].isin(["marking", "static"])].copy()
    keep["line"] = np.where(keep["kind"] == "marking", "marking", "rest")
    mov = b[b["kind"] == "movement"]
    med = (mov.groupby(["pid", "part", "outcome", "unit"], sort=False)
           .agg(**{c: (c, "median") for c in cols}, n_rows=("n_rows", "sum"),
                n_separable=("n_separable", "sum"))
           .reset_index().assign(line="movement")) if not mov.empty else pd.DataFrame()
    out = pd.concat([keep, med], ignore_index=True)
    order = {"marking": 0, "rest": 1, "movement": 2}
    return out.sort_values(["pid", "line"], key=lambda c: c.map(order) if c.name == "line"
                           else c, kind="stable").reset_index(drop=True)


LINE_LABEL = {"marking": "marking", "rest": "rest", "movement": "movement tasks (median)"}
LINE_SHORT = {"marking": "", "rest": "rest", "movement": "tasks"}
NOT_RESOLVED = "†"           # tick mark: the JTS-MM part is not resolvable (σ_Q ≥ σ_obs)
OUTCOME_LABEL = {"teeth": "teeth points", "condyle": "condyle points", "incisor": "incisor",
                 "rotation": "rotation"}
OUTCOME_SHORT = {"teeth": "teeth", "condyle": "condyles", "incisor": "incisor",
                 "rotation": ""}
# budget figure panels: (title, part, outcomes, unit)
BUDGET_PANELS = (("(a) Point marking", "marking", ("teeth", "condyle"), MM),
                 ("(b) Motion: incisor and condyles", "motion", ("incisor", "condyle"), MM),
                 ("(c) Motion: rotation", "motion", ("rotation",), DEG))


def _budget_panel(ctx: Any, d: pd.DataFrame, title: str, unit: str) -> Panel:
    labels = [f"{_who(ctx, r.pid)} " + ", ".join(x for x in (OUTCOME_SHORT[r.outcome],
                                                             LINE_SHORT[r.line]) if x)
              + (f" {NOT_RESOLVED}" if not np.isfinite(r.sigma_jts_corrected) else "")
              for r in d.itertuples()]
    y = np.arange(len(d), dtype=float)
    p = Panel(title, Axis("σ", unit), Axis(""), [], legend="none")
    p.series.append(Series("barh", d["sigma_obs"].to_numpy(float), y, role="muted",
                           label=TERMS["sigma_obs"][0], style={"bar_width": 0.7}))
    p.series.append(Series("barh", d["sigma_jts_corrected"].to_numpy(float), y,
                           role=TERMS["sigma_jts_corrected"][1],
                           label=TERMS["sigma_jts_corrected"][0], style={"bar_width": 0.3}))
    p.series.append(Series("scatter", d["sigma_ref"].to_numpy(float), y,
                           role=TERMS["sigma_ref"][1], label=TERMS["sigma_ref"][0],
                           style={"ms": TERMS["sigma_ref"][2], "size": 1.2}))
    p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
    # X1: the reference-part marker in the participant's shape
    shape_by_rows(p, list(d["pid"]), sorted(ctx.pids()) if hasattr(ctx, "pids") else
                  sorted(d["pid"].unique()))
    return p


@register("val_budget", "dashboard", "Error budget per participant",
          choices={"form": ("bars", "stacked", "dot")}, form="bars", **G)
def val_budget(req: ViewRequest) -> FigureSpec:
    """VD22: per participant the observed error (point marking; rest and movement tasks of
    the motion), its reference part and the JTS-MM part, σ_obs² = σ_JTS² + σ_Q², in one
    row. The per-hold marking error against the CT-predicted tip error is in
    ``val_marking_causes`` (b) and ``val_tip_holds`` (b) (one message per figure).
    Forms ``stacked`` / ``dot``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "bars"))
    if form != "bars":
        from biomech.viz.views.validation_r8 import budget_form

        return budget_form(req, form)
    fid, title = "val_budget", "Error budget per participant"
    ctx = req.ctx
    b = _table(ctx, "ref_budget", "budget")
    if b.empty:
        return _empty(fid, title, "no ref_budget results")
    lines = budget_lines(b)
    panels = []
    for lab, part, outs, unit in BUDGET_PANELS:
        d = lines[(lines["part"] == part) & lines["outcome"].isin(outs)]
        if not d.empty:
            d = d.sort_values(["pid", "outcome"], key=lambda c: c.map(
                {o: i for i, o in enumerate(outs)}) if c.name == "outcome" else c,
                kind="stable")
            panels.append(_budget_panel(ctx, d, lab, unit))
    if panels:
        panels[0].legend = "below"
    n_rows = max(len(lines.drop_duplicates(["pid", "part", "outcome"])) // max(len(panels), 1), 4)
    spec = FigureSpec(fid, title, panels, rows=1, cols=max(len(panels), 1), width="text",
                      height_in=B.rows_height(n_rows, per=0.18),
                      sources=["ref_budget:budget"])
    return attach(spec, CaptionSpec(
        "error budget per participant: the observed JTS-MM − Qualisys difference split in "
        "quadrature into the Qualisys part and the JTS-MM part (σ_obs² = σ_JTS² + σ_Q²) "
        "for (a) the point marking (registered point difference of the teeth and condyle points) "
        "and (b, c) the motion at rest and in the movement tasks (bias-removed jaw-in-head "
        "disagreement; tasks: median over the tasks of each term, the JTS-MM part over the "
        "tasks where it is resolvable)",
        [("wide bar", "observed difference"), ("narrow bar", "JTS-MM part"),
         ("black marker", "Qualisys part"), pshape_text(ctx, b["pid"]),
         # the mark is explained only where a row carries it
         *([(NOT_RESOLVED, "JTS-MM part not resolvable (no narrow bar)")]
           if (~np.isfinite(lines["sigma_jts_corrected"].astype(float))).any() else [])],
        n_text(ctx, b["pid"]),
        statistic="per participant, no pooling: marking = RMS point difference over the teeth or "
                  "condyle points; rest = bias-removed RMS over the gated rest frames; tasks = "
                  "median over the movement tasks of each term's bias-removed RMS",
        evidence=EVIDENCE,
        notes=["Where the Qualisys part reaches the observed difference (σ_Q ≥ σ_obs) the "
               "JTS-MM part cannot be resolved; it is not drawn.",
               "The marking difference per hold against the tip difference the calibration tool's "
               "orientation predicts is shown in the marking-cause figures."],
        codes=["VD22", "V9", "VD4", "EV1"]))


def _resolvable(r: Mapping[str, Any]) -> str:
    """``yes`` / ``no`` for one row, ``k of n`` for a median over the tasks."""
    k, n = int(r["n_separable"]), int(r["n_rows"])
    return ("yes" if k else "no") if n == 1 else f"{k} of {n}"


def _budget_rows(ctx: Any, lines: pd.DataFrame) -> list[dict[str, Any]]:
    keys = ["sigma_obs", "sigma_ref", "sigma_jts_corrected", "ref_var_share"]
    rows: list[dict[str, Any]] = []
    for (line, outcome, unit), g in lines.groupby(["line", "outcome", "unit"], sort=False):
        part = [{"pid": _who(ctx, r["pid"]), "what": f"{LINE_LABEL[line]}, "
                 f"{OUTCOME_LABEL.get(outcome, outcome)}", "unit_text": UNIT_TEXT[unit],
                 "resolvable": _resolvable(r), **{k: float(r[k]) for k in keys}}
                for r in g.to_dict("records")]
        rows += part
        if len(part) > 1:
            rows.append({**_mean_row(part, "pid", keys, f"mean of {len(part)}"),
                         "what": part[0]["what"], "unit_text": part[0]["unit_text"]})
    return rows


@register("val_table_budget", "dashboard", "Table: error budget per participant", **G)
def val_table_budget(req: ViewRequest) -> TableSpec:
    """VD22: per participant, line (marking, rest, movement tasks) and outcome the observed
    error, the reference and the JTS-MM part and the reference share of the variance, with
    the descriptive two-participant mean."""
    b = _table(req.ctx, "ref_budget", "budget")
    rows = _budget_rows(req.ctx, budget_lines(b)) if not b.empty else []
    cols = [Column("what", "Difference", align="l"), Column("pid", "Participant", align="l"),
            Column("unit_text", "Unit", align="l"),
            Column("sigma_obs", "Observed", "", 2), Column("sigma_ref", "Qualisys part", "", 2),
            Column("sigma_jts_corrected", "JTS-MM part", "", 2),
            Column("resolvable", "Resolvable", align="r"),
            Column("ref_var_share", "Qualisys share", "", 2)]
    t = TableSpec("val_table_budget", "", cols, rows, width="text",
                  sources=["ref_budget:budget"])
    return attach(t, CaptionSpec(
        "error budget per participant: observed difference, its Qualisys part (propagated "
        "Qualisys uncertainty σ_Q) and the JTS-MM part σ_JTS = √(σ_obs² − σ_Q²); "
        "Qualisys share = σ_Q²/σ_obs², the share of the observed variance the Qualisys part "
        "alone accounts for (capped at 1); resolvable: σ_Q < σ_obs, otherwise the JTS-MM "
        "part cannot be separated from the Qualisys noise and is left empty",
        n=_n2(req.ctx, b["pid"]) if not b.empty else "",
        statistic="per participant: marking = RMS point difference over the teeth or condyle "
                  "points; rest = bias-removed RMS over the gated rest frames; movement tasks "
                  "= median over the tasks of each term (the JTS-MM part over the resolvable "
                  "tasks); mean row = mean of the participants",
        notes=["Units: mm for the points and the incisor, ° for the rotation."],
        evidence=EVIDENCE, codes=["VD22", "V9"]))


@register("val_table_marking_gap", "dashboard", "Table: marking gap between the participants",
          **G)
def val_table_marking_gap(req: ViewRequest) -> TableSpec:
    """VD22: the frame errors of the marking and their candidate causes for the participant
    with the smallest and the one with the largest maxillary frame error."""
    ctx = req.ctx
    g = _table(ctx, "ref_budget", "marking_gap")
    rows = []
    lo = hi = ""
    for r in g.to_dict("records"):
        q = str(r["quantity"])
        e = catalog.get(q)
        lo, hi = _who(ctx, r["pid_smaller"]), _who(ctx, r["pid_larger"])
        # one digit rule per quantity: counts as integers, the 200-mm lever
        # to 0.1 mm, everything else to 0.01
        count = q.startswith("n_") or "swap" in q
        d = 0 if count else 1 if q == "ct_lever_mm" else 2
        ql = e.label.replace("CT ", "calibration-tool ")
        rows.append({"quantity": ql[0].upper() + ql[1:],
                     "unit": _plain(e.unit),
                     **{k: B.num(float(r[k]), d) for k in ("value_smaller", "value_larger",
                                                           "larger_minus_smaller")},
                     "larger_over_smaller": "–" if count else B.num(float(r["larger_over_smaller"]),
                                                                   2)})
    cols = [Column("quantity", "Quantity", align="X"), Column("unit", "Unit", align="l"),
            Column("value_smaller", lo or "First", align="r"),
            Column("value_larger", hi or "Second", align="r"),
            Column("larger_minus_smaller", "Difference", align="r"),
            Column("larger_over_smaller", "Ratio", align="r")]
    t = TableSpec("val_table_marking_gap", "", cols, rows, width="text",
                  sources=["ref_budget:marking_gap"])
    return attach(t, CaptionSpec(
        "the frame differences of the point marking and their candidate causes for the two "
        "participants: the calibration-tool orientation disagreement of the two systems during "
        "the holds and the tip difference it predicts over the calibration-tool lever (median over "
        "the holds), the geometry of the Qualisys head cluster (smallest triangle height; "
        "second over first singular value, near 0 = nearly collinear), the holds with swapped "
        "calibration-tool marker labels (counts) and the Qualisys uncertainty of the teeth "
        "landmarks",
        n="n = 2 participants",
        statistic=f"difference = {hi or 'second'} − {lo or 'first'}; ratio = {hi or 'second'} / "
                  f"{lo or 'first'} (not for counts)",
        evidence=EVIDENCE, codes=["VD22", "VD4", "VD7", "EV1"]))


def _plain(unit: str) -> str:
    from biomech.viz.style import plain_unit

    return plain_unit(unit)


# == VD21: hinge models on both systems ==========================================================
HINGE_MODELS = (("hinge_fit_mm", "best-fit fixed axis"), ("hinge_tmj_mm", "axis through the "
                                                                          "condyle points"))
SYSTEM_STYLE = {"jtsmm": ("JTS-MM", "system:jtsmm", "o"),
                "qualisys": ("Qualisys", "system:qualisys", "s")}


PLATE = {"HP": "head plate", "MP": "jaw plate"}


def gate_reason(ctx: Any, pid: str, task: str, q: pd.DataFrame | None = None) -> str:
    """" (x % of its frames gated: <plate> <why>)" for one participant-task, from
    ``ref_quality`` (largest share over the head and jaw plates); "" without a record."""
    from biomech.validation import quality as Q

    if q is None:
        q = _table(ctx, "ref_quality", "consistency")
    if not {"pid", "segment", "body", "frac_gated"} <= set(q.columns):
        return ""
    g = q[(q.pid == pid) & (q.segment == task) & q.body.isin(list(PLATE))]
    if not len(g):
        return ""
    w = g.loc[g.frac_gated.idxmax()]
    why = ("markers missing" if w.frac_missing >= 0.5 * w.frac_gated else
           "markers gap-filled" if w.frac_gapfilled >= 0.5 * w.frac_gated else
           f"inter-marker distances off their median by more than {Q.GATE_DIST_MM:g} mm"
           if w.dev_max_mm > Q.GATE_DIST_MM else
           f"marker residuals above {Q.GATE_RESID_MM:g} mm")
    return f" ({100 * float(w.frac_gated):.0f} % of its frames gated: {PLATE[w.body]} {why})"


def _hinge_gate_notes(ctx: Any, h: pd.DataFrame) -> list[str]:
    """Why a participant-task has no hinge fit: every opening touched frames that the
    Qualisys quality gate (V8: Qualisys's own marker quality) excludes; the share of gated
    frames of that task from ``ref_quality`` (largest over the head and jaw plates)."""
    if "n_openings" not in h:
        return []
    q = _table(ctx, "ref_quality", "consistency")
    return [f"{_who(ctx, r.pid)}, {V.phase_label(r.task).lower()}: no opening clear of the "
            f"Qualisys quality gate{gate_reason(ctx, r.pid, r.task, q)}"
            for r in h[h.n_openings == 0].drop_duplicates(["pid", "task"]).itertuples()]


def missing_cycle_notes(ctx: Any, have: pd.DataFrame, pids: list[str],
                        tasks: tuple[str, ...] = ("open_close", "protrusion_retrusion",
                                                  "left_right", "chewing")) -> list[str]:
    """One note per participant-task without a cycle in a cycle-based table (``have``: its
    pid / task rows): ``ref_agreement`` keeps a cycle only when none of its frames is gated
   , and the note gives the gated share and its cause from ``ref_quality``."""
    seen = set(zip(have["pid"], have["task"])) if {"pid", "task"} <= set(have.columns) else set()
    q = _table(ctx, "ref_quality", "consistency")
    miss = [(p_, t_) for p_ in pids for t_ in tasks if (p_, t_) not in seen]
    if not miss:
        return []
    parts = [f"{_who(ctx, p_)} {V.phase_label(t_).lower()}{gate_reason(ctx, p_, t_, q)}"
             for p_, t_ in miss]
    return ["Rows are missing where no cycle passes the Qualisys quality gate (a cycle "
            "counts only if none of its frames is gated): " + "; ".join(parts) + "."]


@register("val_hinge", "dashboard", "Hinge models on both systems",
          choices={"form": ("offset_dots", "dumbbell", "difference")}, form="offset_dots", **G)
def val_hinge(req: ViewRequest) -> FigureSpec:
    """VD21: per participant, task and hinge model the incisor RMS residual of the fit to the
    JTS-MM and to the Qualisys trajectories of the same openings. Forms ``dumbbell`` /
    ``difference``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "offset_dots"))
    if form != "offset_dots":
        from biomech.viz.views.validation_r8 import hinge_form

        return hinge_form(req, form)
    fid, title = "val_hinge", "Hinge models on both systems"
    ctx = req.ctx
    h = _table(ctx, "ref_hinge", "hinge")
    if h.empty or "hinge_fit_mm" not in h:
        return _empty(fid, title, "no ref_hinge results")
    tasks = [t for t in V.TASKS if (h["task"] == t).any()]
    rows = [(pid, col) for pid in _pids(h) for col, _ in HINGE_MODELS]
    y = np.arange(len(rows), dtype=float)
    labels = [f"{_who(ctx, pid)} {dict(HINGE_MODELS)[col]}" for pid, col in rows]
    top = float(np.nanmax(h[[c for c, _ in HINGE_MODELS]].to_numpy(float))) * 1.08
    panels = []
    for k, task in enumerate(tasks):
        p = Panel(f"({'abcd'[k]}) {V.term('task', task, 'title')}",
                  Axis("Incisor residual", MM, lim=(0.0, top)), Axis(""), [],
                  legend="below" if k == 0 else "none")
        for j, (sys_, (lab, role, ms)) in enumerate(SYSTEM_STYLE.items()):
            x = [float(h[(h.pid == pid) & (h.task == task) & (h.system == sys_)][col].iloc[0])
                 if ((h.pid == pid) & (h.task == task) & (h.system == sys_)).any() else np.nan
                 for pid, col in rows]
            p.series.append(Series("scatter", np.asarray(x), y + (j - 0.5) * 0.2, role=role,
                                   label=lab, style={"ms": ms}))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
        shape_by_rows(p, [pid for pid, _ in rows], _pids(h))    # shape = participant (X1)
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=1, cols=max(1, len(panels)), width="text",
                      height_in=B.rows_height(len(rows), per=0.25),
                      sources=["ref_hinge:hinge"])
    return attach(spec, CaptionSpec(
        "the dataset's two fixed-hinge models fitted to the same openings (start to peak) "
        "on the JTS-MM and on the Qualisys trajectories, each system in its own head frame: "
        "incisor RMS residual of the best-fit fixed axis and of the axis through the condyle "
        "points. Qualisys tracks with sub-millimetre precision, so its residual bounds the "
        "anatomical non-hinge motion; the JTS-MM minus Qualisys difference is what tracking "
        "with JTS-MM adds",
        [("blue", "JTS-MM"), ("black", "Qualisys"), pshape_text(ctx, _pids(h)),
         ("rows", "participant and hinge model")],
        n_text(ctx, h["pid"]),
        statistic="per participant, task and model the RMS residual over frames sampled from "
                  "each opening (start to peak), all openings of the task together; no pooling "
                  "over participants",
        evidence="exploratory", notes=_hinge_gate_notes(ctx, h),
        codes=["VD21", "C1", "D33"]))


@register("val_table_hinge", "dashboard", "Table: hinge models on both systems", **G)
def val_table_hinge(req: ViewRequest) -> TableSpec:
    """VD21: per participant and task the hinge residuals on both systems, their difference
    and the best-fit axis's distance from the condyle midpoint, with the descriptive mean."""
    ctx = req.ctx
    h = _table(ctx, "ref_hinge", "hinge")
    keys = ["fit_s", "fit_q", "d_fit", "tmj_s", "tmj_q", "d_tmj", "axis_s", "axis_q",
            "n_openings"]
    rows: list[dict[str, Any]] = []
    if not h.empty and "hinge_fit_mm" in h:
        for task in [t for t in V.TASKS if (h["task"] == t).any()]:
            part = []
            for pid in _pids(h):
                g = h[(h.pid == pid) & (h.task == task)].set_index("system")
                if not {"jtsmm", "qualisys"} <= set(g.index):
                    continue
                s, q = g.loc["jtsmm"], g.loc["qualisys"]
                part.append({"pid": _who(ctx, pid), "task": V.phase_label(task),
                             "fit_s": s.hinge_fit_mm, "fit_q": q.hinge_fit_mm,
                             "d_fit": s.hinge_fit_mm - q.hinge_fit_mm,
                             "tmj_s": s.hinge_tmj_mm, "tmj_q": q.hinge_tmj_mm,
                             "d_tmj": s.hinge_tmj_mm - q.hinge_tmj_mm,
                             "axis_s": s.axis_to_condyles_mm, "axis_q": q.axis_to_condyles_mm,
                             "n_openings": int(q.n_openings)})
            rows += part
            ok = [r for r in part if np.isfinite(r["fit_s"]) and np.isfinite(r["fit_q"])]
            if len(ok) > 1:
                rows.append({**_mean_row(ok, "pid", keys[:-1], f"mean of {len(ok)}"),
                             "task": V.phase_label(task), "n_openings": np.nan})
    cols = [Column("pid", "Participant", align="l"), Column("task", "Task", align="l"),
            Column("n_openings", "Openings", "", 0),
            Column("fit_s", "JTS-MM", MM, 2, group="Best-fit axis"),
            Column("fit_q", "Qualisys", MM, 2, group="Best-fit axis"),
            Column("d_fit", "Difference", MM, 2, group="Best-fit axis"),
            Column("tmj_s", "JTS-MM", MM, 2, group="Condyle axis"),
            Column("tmj_q", "Qualisys", MM, 2, group="Condyle axis"),
            Column("d_tmj", "Difference", MM, 2, group="Condyle axis"),
            Column("axis_s", "JTS-MM", MM, 1, group="Best-fit axis to condyle midpoint"),
            Column("axis_q", "Qualisys", MM, 1, group="Best-fit axis to condyle midpoint")]
    t = TableSpec("val_table_hinge", "", cols, rows, width="text",
                  sources=["ref_hinge:hinge", "ref_hinge:hinge_diff"])
    return attach(t, CaptionSpec(
        "incisor RMS residual of the dataset's fixed-hinge models on the same openings, "
        "JTS-MM and Qualisys each in their own head frame, the difference JTS-MM minus "
        "Qualisys (what tracking with JTS-MM adds), and the distance of the best-fit axis from the "
        "midpoint of the condyle points",
        n=_n2(ctx, h["pid"]) if not h.empty else "",
        statistic="per participant and task the RMS residual over frames sampled from each "
                  "opening (start to peak), all openings together (Openings = their number); "
                  "mean row = mean of the participants",
        evidence="exploratory", notes=_hinge_gate_notes(ctx, h), codes=["VD21"]))


# == VD23: EMG side check ========================================================================
def _profile_label(task: str, side: str) -> str:
    lab = V.phase_label(task)
    return f"{lab}, to the {side}" if side else lab


@register("val_emg_side", "dashboard", "EMG: side check of the montage against Delsys",
          choices={"form": ("offset_dots", "heatmap", "correlations")}, form="offset_dots",
          **G)
def val_emg_side(req: ViewRequest) -> FigureSpec:
    """VD23: per muscle pair, task and participant the side index ½[r(R,R) + r(L,L) −
    r(R,L) − r(L,R)] of the JTS-MM and Delsys mean cycle profiles. Forms ``heatmap`` /
    ``correlations``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "offset_dots"))
    if form != "offset_dots":
        from biomech.viz.views.validation_r8 import emg_side_form

        return emg_side_form(req, form)
    fid, title = "val_emg_side", "EMG: side check of the montage against Delsys"
    ctx = req.ctx
    d = _table(ctx, "ref_emg_side", "side_check")
    if d.empty:
        return _empty(fid, title, "no ref_emg_side results")
    d = d.assign(side=d["side"].fillna(""))
    keys = list(dict.fromkeys(zip(d["task"], d["side"])))
    order = {t: i for i, t in enumerate(V.TASKS)}
    keys.sort(key=lambda k: (order.get(k[0], len(order)), k[1]))
    informative = {(r.task, r.side) for r in d[d["side_informative"].astype(bool)].itertuples()}
    labels = [_profile_label(t, s) + (" *" if (t, s) in informative else "") for t, s in keys]
    y = np.arange(len(keys), dtype=float)
    panels = []
    for k, pair in enumerate(dict.fromkeys(d["muscle"])):
        p = Panel(f"({'abcdef'[k]}) {V.term('muscle', pair, 'title')}",
                  Axis("Side index", "", lim=(-2.2, 2.2)), Axis(""), [],
                  legend="below" if k == 0 else "none")
        p.series.append(Series("vline", x=np.array([-2.0, 0.0, 2.0]), role="reference"))
        for i, pid in enumerate(_pids(d)):
            g = d[(d.pid == pid) & (d.muscle == pair)].set_index(["task", "side"])
            x = [float(g.loc[key, "side_index"]) if key in g.index else np.nan for key in keys]
            p.series.append(Series("scatter", np.asarray(x), y + (i - 0.5) * 0.2,
                                   role="accent", label=_who(ctx, pid),
                                   style={"ms": "os"[i % 2]}))
        p.y.ticks, p.y.ticklabels, p.y.reverse = list(y), labels, True
        panels.append(p)
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=B.rows_height(len(keys), per=0.25),
                      sources=["ref_emg_side:side_check"])
    return attach(spec, CaptionSpec(
        "side check of the JTS-MM electrode montage against the side-labeled Delsys sensors: "
        "shape correlation r of the ensemble-mean cycle profiles (each system's own "
        "recording day) between JTS-MM side a and Delsys side b, combined into the side index "
        "½[r(R,R) + r(L,L) − r(R,L) − r(L,R)]: +2 the sides as in the montage, −2 swapped, "
        "near 0 not identifiable (both sides act alike)",
        [("markers", "participants"), ("*", "side-informative task (laterotrusion)"),
         ("lines", "−2, 0, +2")],
        n_text(ctx, d["pid"]),
        statistic="per participant, task and muscle pair one side index from the four "
                  "correlations, no summary",
        evidence=EVIDENCE,
        notes=["Pearson r over the 101 time-normalized points (equivalently, the mean product "
               "of the z-scored profiles)."],
        codes=["VD23", "I3b"]))


@register("val_table_emg_side", "dashboard", "Table: EMG side check against Delsys", **G)
def val_table_emg_side(req: ViewRequest) -> TableSpec:
    """VD23: the four shape correlations and the side index per participant, profile and
    muscle pair."""
    d = _table(req.ctx, "ref_emg_side", "side_check")
    if not d.empty:
        order = {c: {v: i for i, v in enumerate(vals)} for c, vals in
                 (("task", V.TASKS), ("muscle", list(dict.fromkeys(d["muscle"]))))}
        d = d.sort_values(["pid", "task", "side", "muscle"],
                          key=lambda c: c.map(order[c.name]) if c.name in order else c,
                          kind="stable")
    t = B.frame_table(d, "val_table_emg_side", "",
                      columns=["pid", "task", "side", "muscle", "side_informative", "r_rr",
                               "r_ll", "r_rl", "r_lr", "side_index"], ctx=req.ctx)
    t.sources = ["ref_emg_side:side_check"]
    return attach(t, CaptionSpec(
        "side check of the JTS-MM montage against Delsys per participant, task (laterotrusion "
        "to the left and to the right separately) and muscle pair: shape correlations r(a, b) "
        "of the JTS-MM side a and the Delsys side b mean cycle profiles and the side index "
        "½[r(R,R) + r(L,L) − r(R,L) − r(L,R)] (+2 sides as in the montage, −2 swapped)",
        n=n_text(req.ctx, d["pid"]) if "pid" in d and len(d) else "",
        statistic="Pearson r over the 101 time-normalized points of the two ensemble-mean "
                  "profiles (mean over each system's cycles); one row per participant, task "
                  "and muscle pair, no summary",
        evidence=EVIDENCE, codes=["VD23"]))
