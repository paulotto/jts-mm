# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ICORR 2027 figure set (IEEE, 6 pages; publication strategy D).

* ``icorr_setup`` (Fig. 1): the recording protocol of one participant and a synchronized
  example of every modality during open-close -- opening angle (optical and IMU), opener and
  closer activation, and the intent label derived from the optical motion.
* ``icorr_hinge`` (Fig. 2, C1): incisal path, helical-axis migration, and per participant the
  fixed-hinge misalignment with its bootstrap CI against that participant's own noise-only
  99th percentiles (in vivo, bench); the translation the joint must allow (condyle-point
  excursion) with the design value of the requirements table.
* ``icorr_imu`` (Fig. 3, C3a): Bland-Altman agreement of IMU-estimated and optical incisal
  opening for the generic (leave-one-subject-out) and the participant's own model, with the
  repeated-measures bias and limits of agreement, their participant-cluster bootstrap CIs and
  the proportional-bias line.

Table I is ``table_requirements`` (paper.py), built from the ``requirements`` stage.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from biomech.kinematics import se3
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import blocks as B
from biomech.viz.views.paper import _empty, fig_kinematics
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.vocab import (DEG, MM, OPENERS, CLOSERS, PCT, SEC,
                                     TASK_LABEL)

LABEL_NAMES = ("rest", "open", "close", "protrude", "retrude", "left", "right")
EXAMPLE_S = 8.0          # length of the synchronized example
LEAD_IN_S = 1.0          # ... starting this long before the first open-close cycle


def representative(ctx) -> str | None:
    """The participant whose self-paced opening is closest to the cohort median among those
    with every EMG channel usable for shape (a typical, complete example)."""
    subj = ctx.table("stats_kinematics", "subjects")
    qc = ctx.table("qc", "emg")
    pids = ctx.pids()
    if subj.empty or not pids:
        return pids[0] if pids else None
    full = set(pids)
    if not qc.empty and "usable_shape" in qc:
        ok = qc.groupby("pid").usable_shape.all()
        full = {p for p in pids if bool(ok.get(p, False))} or set(pids)
    s = subj[subj.pid.isin(full)].dropna(subset=["opening_selfpaced_mm"])
    if s.empty:
        return pids[0]
    med = s.opening_selfpaced_mm.median()
    return str(s.loc[(s.opening_selfpaced_mm - med).abs().idxmin(), "pid"])


def _rel(t_us: np.ndarray, t0: float) -> np.ndarray:
    return (np.asarray(t_us, float) - t0) * 1e-6


# Hardware renders (no participant) for the schematic depiction of the setup: the used part
# versions of hardware/mechanical/parts.yaml (biomech.hwparts), the original JTS renders while
# the study's usage is not specified there.
RENDER_ASSEMBLIES = {"headpiece": "head_piece", "mouthpiece": "mouth_piece",
                     "calibration tool": "calibration_tool"}


def render_panel(name: str, title: str, max_px: int = 500, study: str | None = None) -> Panel:
    """A hardware render as an image panel: transparent margins cropped, composited on
    white, at most ``max_px`` wide (the figure stays small)."""
    from biomech.viz.views.hardware import assembly_image, image_panel

    path = assembly_image(study, RENDER_ASSEMBLIES[name])
    if path is None:
        return _empty(Panel(title, Axis(""), Axis(""), [], legend="none"), "render not found")
    return image_panel(path, title, max_px)


def _rates(ctx) -> tuple[dict[str, float], int]:
    """Achieved rates (median over the participants) of the streams shown in the setup and
    the number of participants they come from."""
    lanes = ctx.table("qc", "timeline_lanes")
    if lanes.empty or "lane" not in lanes:
        return {}, 0
    med = lanes.groupby("lane").rate_hz.median()
    n = int(lanes.pid.nunique()) if "pid" in lanes else 0
    return {k: float(med[k]) for k in ("EMG", "IMU MTA quat", "optical MTA") if k in med}, n


SETUP_FORMS = ("story", "signals", "annotated", "schematic")


RAW_ONE_TAKE = "raw signals of one take, no summary statistic"


def _shift_letters(s: str) -> str:
    """'(c)' -> '(d)' etc.: the excerpt's panel letters one on (the lane is (a))."""
    import re

    return re.sub(r"\(([a-f])\)", lambda m: f"({chr(ord(m.group(1)) + 1)})", s)


def _setup_story(ctx, pid: str | None) -> FigureSpec:
    """Form ``story`` of :func:`icorr_setup` (2026-11-05): (a) the protocol of the motion take
    as a thin phase lane over the session, the excerpt marked; (b-f) the multimodal excerpt of
    ``emg_story`` (raw EMG, its short-time spectrum, group envelopes, optical and IMU opening
    angle, intent label with the decoder and release decisions) on one time axis."""
    from biomech.viz.views import registry as R
    from biomech.viz.views import vocab as _vocab

    fid, title = "icorr_setup", "Protocol and synchronized modalities"
    if pid is None:
        return attach(FigureSpec(fid, title, [_empty(Panel("(a) Protocol", Axis(""), Axis("")))]),
                      CaptionSpec("no participant"))
    who = f"P{ctx.pseudonym(pid)}"
    st = R.build("emg_story", R.request(ctx, "emg_story", pid=pid))
    ph = ctx.table("qc", "phases", [pid])
    ph = ph[ph["take"] == "motion"].sort_values("t_start_us") if not ph.empty else ph
    win = (st.meta or {}).get("window_us")
    if not ph.empty:
        t0 = float(ph.t_start_us.min())
        spans = [(str(r.phase), (r.t_start_us - t0) / 60e6, (r.t_end_us - t0) / 60e6)
                 for r in ph.itertuples()]
        # the landmark holds as one labelled block (JNER revision 3.1: L1, L2, U1, ... crowded
        # the lane)
        merged: list[tuple[str, float, float]] = []
        for name, a, b in spans:
            hold = _vocab.phase_role(name) == "faint"
            if hold and merged and merged[-1][0] == "landmark holds":
                merged[-1] = ("landmark holds", merged[-1][1], b)
            else:
                merged.append(("landmark holds" if hold else name, a, b))
        lane = B.phase_lane(merged, "(a) Protocol")
        lane.sharex = None
        lane.x = Axis("Session time", "\\minute")
        lane.height = 0.55
        if win:
            c = (0.5 * (win[0] + win[1]) - t0) / 60e6
            lane.series.append(Series("vline", x=np.array([c]), role="threshold",
                                      style={"width": 1.2}))
    else:
        lane = _empty(Panel("(a) Protocol", Axis("Session time", "\\minute"), Axis("")))
    panels = [lane]
    for i, p in enumerate(st.panels):
        p.title = f"({'bcdefg'[i]}) " + p.heading()[1]
        p.tag = ""
        # JNER revision 3.1: the signals only. The decoder and the release decisions are
        # results (decodingOnline, timing figures), so the intent panel shows the label alone
        p.series = [sr for sr in p.series
                    if not (sr.kind == "vline" and sr.role == "group:closers")
                    and not (sr.label or "").startswith(("decoder", "release decision"))]
        if p.title.startswith(f"({'bcdefg'[i]}) Intent"):
            p.title = f"({'bcdefg'[i]}) Intent label"
            p.legend = "none"
            for sr in p.series:
                sr.label = None
        if p.y.label == "Frequency":
            p.y.label = "f"                  # the long label ran into the panel letter (3.0)
        if p.legend_mode() == "below":       # legends inside, no white band below (3.0)
            p.legend = "inside:north east"
            ys = [float(v) for sr in p.series if sr.kind == "line" and sr.y is not None
                  for v in np.asarray(sr.y, float) if np.isfinite(v)]
            if ys:                           # headroom for the legend above the traces
                lo_, hi_ = min(ys), max(ys)
                p.y.lim = (lo_ - 0.05 * (hi_ - lo_), hi_ + 0.55 * (hi_ - lo_))
        panels.append(p)
    # 0.55 in per row: the six rows of a main figure in about half a page
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text",
                      height_in=0.55, sources=["qc:phases"] + list(st.sources or []),
                      perspective=f"{who}, protocol and multimodal excerpt",
                      meta={"pid": pid, "combines": ["emg_story"]})
    cs = dict((st.meta or {}).get("caption") or {})
    shows = str(cs.get("shows", "")).replace("excerpt on one time axis:",
                                             "excerpt on one time axis (b–f):", 1)
    # the excerpt without the decoder and release results (3.1)
    shows = re.sub(r",? the simulated real-time decoder's smoothed decisions and the causal "
                   r"closer-release decisions", "", shows)
    shows = re.sub(r",? ?(with )?the (smoothed )?decoder[^.;]*", "", shows)
    cs["n"] = re.sub(r", \d+ release decisions? in the excerpt", "", str(cs.get("n") or ""))
    shows += (". The raw signal and its spectrum are shown for the right masseter only, as an "
              "example channel; the envelopes in (d) average all six channels by muscle group")
    cs["marks"] = [(g, m) for g, m in (cs.get("marks") or [])
                   if not any(w in (g + m).lower() for w in ("decoder", "release", "vertical"))]
    return attach(spec, CaptionSpec(
        f"(a) Recording protocol of {who}: landmark holds, rest, and the five movement tasks "
        "in session order as a phase lane, the vertical line marks the excerpt below. "
        + shows,
        [("lane (a)", "recorded phases in their task colours, holds grey")]
        + [(_shift_letters(g), m) for g, m in (cs.get("marks") or [])],
        str(cs.get("n") or ""), str(cs.get("statistic") or RAW_ONE_TAKE),
        evidence="descriptive"))


@register("icorr_setup", "paper", "Protocol and synchronized modalities", params=("pid",),
          choices={"form": SETUP_FORMS}, form="story")
def icorr_setup(req: ViewRequest) -> FigureSpec:
    """Form ``story`` (default since 2026-11-05): protocol lane and the multimodal excerpt
    (:func:`_setup_story`); ``signals`` / ``annotated`` / ``schematic``: the protocol bars and an
    8-s open-close example (the old figure; ``schematic`` with the hardware renders)."""
    ctx = req.ctx
    form = req.options.get("form", "story")
    pid = req.pid if req.pid in ctx.pids() else representative(ctx)
    if form == "story":
        return _setup_story(ctx, pid)
    fid, title = "icorr_setup", "Protocol and synchronized modalities"
    who = f"P{ctx.pseudonym(pid)}" if pid else "no participant"
    # (a) protocol timeline (motion take, session order)
    ph = ctx.table("qc", "phases", [pid]) if pid else pd.DataFrame()
    ph = ph[ph["take"] == "motion"].sort_values("t_start_us") if not ph.empty else ph
    pa = Panel("(a) Protocol", Axis("Session time", "\\minute"), Axis(""), legend="none")
    if not ph.empty:
        t0 = float(ph.t_start_us.min())
        exp = ph[ph.kind == "experiment"]
        cal = ph[ph.kind != "experiment"]
        rows = [("landmark holds", cal)] + [(TASK_LABEL.get(r.phase, r.phase), exp[exp.phase == r.phase])
                                           for r in exp.itertuples()]
        for i, (_lab, g) in enumerate(rows):
            for r in g.itertuples():
                a, b = (r.t_start_us - t0) / 60e6, (r.t_end_us - t0) / 60e6
                pa.series.append(Series("line", np.array([a, b]), np.array([i, i], float),
                                        role="accent" if r.kind == "experiment" else "context",
                                        style={"width": 4.0}))
        pa.y.ticks = [float(i) for i in range(len(rows))]
        pa.y.ticklabels = [lab for lab, _ in rows]
        pa.y.reverse = True
    else:
        _empty(pa)
    # (b-d) synchronized example in open-close
    kin = ctx.arrays("kinematics", pid) if pid else {}
    imu = ctx.arrays("imu", pid) if pid else {}
    emg = ctx.arrays("emg", pid) if pid else {}
    lab = ctx.arrays("labels", pid) if pid else {}
    cyc = ctx.table("kinematics", "cycles", [pid]) if pid else pd.DataFrame()
    cyc = cyc[cyc.task == "open_close"] if not cyc.empty else cyc
    t_ph0 = float(kin["open_close__t_us"][0]) if "open_close__t_us" in kin else None
    start = (float(cyc.t_start_us.min()) - LEAD_IN_S * 1e6) if not cyc.empty else t_ph0
    pb = Panel("(b) Opening angle", Axis("Time", SEC), Axis("Angle", DEG), legend="below",
               sharex="t")
    pc = Panel("(c) Activation", Axis("Time", SEC), Axis("Activation", PCT),
               legend="below", sharex="t")
    pd_ = Panel("(d) Intent label", Axis("Time", SEC), Axis(""), legend="none", sharex="t")
    if start is not None:
        def win(t: np.ndarray) -> np.ndarray:
            x = _rel(t, start)
            return (x >= 0) & (x <= EXAMPLE_S)

        if "open_close__theta_deg" in kin:
            t = kin["open_close__t_us"]
            m = win(t)
            pb.series.append(Series("line", _rel(t, start)[m], kin["open_close__theta_deg"][m],
                                    role="accent", label="optical"))
        if "open_close__theta_quat" in imu:
            t = imu["open_close__t_us"]
            m = win(t)
            pb.series.append(Series("line", _rel(t, start)[m], imu["open_close__theta_quat"][m],
                                    role="cat:1", style={"width": 0.8}, label="IMU"))
        if "open_close__norm_pct" in emg and "profile__names" in emg:
            names = [str(n) for n in emg["profile__names"]]
            t = emg["open_close__t_us"]
            m = win(t)
            # thin grey per muscle, bold group means in the group colours (six coloured
            # channels read as spaghetti and hid the openers / closers contrast)
            for grp, role, glab in ((OPENERS, "group:openers", "openers (digastric)"),
                                    (CLOSERS, "group:closers", "closers (masseter, temporalis)")):
                cols = [names.index(mu) for mu in grp if mu in names]
                if not cols:
                    continue
                y = emg["open_close__norm_pct"][m][:, cols]
                for j in range(y.shape[1]):
                    pc.series.append(Series("line", _rel(t, start)[m], y[:, j], role="context",
                                            style={"width": 0.5}))
                with np.errstate(all="ignore"):
                    mean = np.nanmean(y, axis=1)
                pc.series.append(Series("line", _rel(t, start)[m], mean, role=role,
                                        style={"width": 1.5}, label=glab))
        if "open_close__label" in lab:
            t = lab["open_close__t_us"]
            m = win(t)
            y = lab["open_close__label"][m].astype(float)
            y[y < 0] = np.nan
            pd_.series.append(Series("line", _rel(t, start)[m], y, role="accent",
                                     style={"width": 1.2}))
            # Every class that occurs in the window gets a tick (open-close is mostly rest,
            # open and close; free motion or a drift into protrusion shows up too).
            present = sorted({int(v) for v in y[np.isfinite(y)]} | {0, 1, 2})
            pd_.y.ticks = [float(v) for v in present]
            pd_.y.ticklabels = [LABEL_NAMES[v] for v in present]
            pd_.y.lim = (min(present) - 0.4, max(present) + 0.4)
    for p in (pb, pc, pd_):
        if not p.series:
            _empty(p)
        p.x.lim = (0.0, EXAMPLE_S)
    rate_mark: list[tuple[str, str]] = []
    n_txt = f"n = 1 participant ({who})" if pid else "no participant"
    if form in ("annotated", "schematic"):
        r, n_rates = _rates(ctx)
        hz = {k: f"{v:.0f} Hz" for k, v in r.items()}
        pb.title = ("(b) Opening angle · optical " + hz.get("optical MTA", "–") + ", IMU "
                    + hz.get("IMU MTA quat", "–"))
        pc.title = "(c) Activation · EMG " + hz.get("EMG", "–") + ", 6 channels"
        pd_.title = "(d) Intent label · from the optical incisal velocity"
        rate_mark = [("panel titles", "the achieved sampling rate of each stream (median over "
                                      "the participants; IMU: orientation stream of the "
                                      "mouth-piece IMU)")]
        n_txt += f"; sampling rates from {n_rates} participants"
    srcs = ["qc:phases", "kinematics:arrays", "imu:arrays", "emg:arrays", "labels:arrays"]
    example = (f"{EXAMPLE_S:.0f} s of open-close from {LEAD_IN_S:.0f} s before its first cycle, "
               "on the common clock: opening angle from optical tracking and from the relative "
               "orientation of the two IMUs (quaternion estimator, one trace); masked RMS activation of the openers (digastrics) and "
               "closers (masseters, temporalis) in % of the largest activity recorded per "
               "channel; the intent label derived from the optical incisal velocity")
    protocol = ("Recording protocol of one participant: landmark digitization holds, rest and "
                "the five movement tasks in session order")
    marks = [("bars in the protocol", "the recorded phases (movement tasks black, landmark "
                                      "holds grey)"),
             ("opening angle", "optical (black) and IMU (thin orange line)"),
             ("activation", "thin grey: each muscle; bold: mean of the openers (vermilion) "
                            "and of the closers (blue)"),
             ("intent label", "the class over time")]
    if form == "schematic":
        from biomech.viz.views.hardware import usage_note

        study = str(getattr(ctx.config, "study", "") or "")
        ph_ = render_panel("headpiece", "(a) Head piece", study=study)
        pm_ = render_panel("mouthpiece", "(b) Mouth piece", study=study)
        for q, t in ((pa, "(c) Protocol"), (pb, "(d)" + pb.title[3:]),
                     (pc, "(e)" + pc.title[3:]), (pd_, "(f)" + pd_.title[3:])):
            q.title = t
        spec = FigureSpec(fid, title, [ph_, pm_, pa, pb, pc, pd_], rows=3, cols=2,
                          width="text", height_in=1.25, sources=srcs, perspective=who,
                          meta={"pid": pid})
        return attach(spec, CaptionSpec(
            "(a, b) The worn hardware (renders): head piece and mouth piece with their ArUco "
            f"tag bodies. (c) {protocol}. (d–f) {example}", marks + rate_mark, n_txt,
            RAW_ONE_TAKE, evidence="descriptive", notes=usage_note(study)))
    spec = FigureSpec(fid, title, [pa, pb, pc, pd_], rows=4, cols=1, width="column",
                      height_in=0.95, sources=srcs, perspective=who, meta={"pid": pid})
    return attach(spec, CaptionSpec(f"(a) {protocol}. (b–d) {example}", marks + rate_mark,
                                    n_txt, RAW_ONE_TAKE, evidence="descriptive"))


HINGE_FORMS = ("panels", "panels_fit", "panels_helical", "path3d", "multiples")
# The opening range a two-phase (rotation, then translation) model treats as pure rotation:
# no project or literature value is fixed for it (translation is present from the first
# millimetres), so the band marks the conventional ~12° of the textbook model (2026-11-05).
PURE_ROTATION_DEG = 12.0
HINGE_ARCS = (("hinge_fit", "best-fit fixed hinge", "accent", "dashed"),
              ("hinge_tmj", "hinge through the palpated condyles", "cat:1", "dotted"))


def _hinge_arc(row: pd.Series, inc0: np.ndarray, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Sagittal (x, z) arc a perfect hinge of the stored model (axis point at y = 0, the
    model's largest opening angle) gives the incisor from its closed position."""
    ctr = np.array([row.axis_pierce_x_mm, row.axis_pierce_z_mm], float)
    v0 = inc0 - ctr
    phi = np.radians(np.linspace(0.0, float(row.theta_max_deg), n))
    out = []
    for sign in (1.0, -1.0):                   # the sense that moves the incisor down
        c, s_ = np.cos(sign * phi), np.sin(sign * phi)
        out.append(ctr + np.column_stack([c * v0[0] - s_ * v0[1], s_ * v0[0] + c * v0[1]]))
    arc = min(out, key=lambda a: a[-1, 1])
    return arc[:, 0], arc[:, 1]


def _median_arcs(ctx) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """Per hinge model the arc a perfect hinge draws for the cohort median: the median axis
    location relative to each participant's closed incisor (sagittal, head frame) and the
    median largest opening angle of that model, from the origin (closed incisor). Returns
    (model, label, role, x, z)."""
    h = ctx.table("kinematics", "hinge")
    if h.empty:
        return []
    rel: dict[str, list[tuple[float, float, float]]] = {}
    for pid in ctx.pids():
        sg = _sagittal(ctx, pid)
        if sg is None:
            continue
        inc0 = sg[2]
        for model in ("hinge_tmj", "hinge_fit"):
            r = h[(h.pid == pid) & (h.model == model)]
            if r.empty or not np.isfinite(float(r.axis_pierce_x_mm.iloc[0])):
                continue
            r = r.iloc[0]
            rel.setdefault(model, []).append((float(r.axis_pierce_x_mm) - inc0[0],
                                              float(r.axis_pierce_z_mm) - inc0[1],
                                              float(r.theta_max_deg)))
    out = []
    for model, lab in (("hinge_tmj", "hinge at the joint"),
                       ("hinge_fit", "one compromise axis (best fit)")):
        v = np.array(rel.get(model, []), float)
        if not len(v):
            continue
        cx, cz, th = np.nanmedian(v, axis=0)
        row = pd.Series({"axis_pierce_x_mm": cx, "axis_pierce_z_mm": cz, "theta_max_deg": th})
        x, z = _hinge_arc(row, np.zeros(2))
        out.append((model, lab, f"model:{model}", x, z))
    return out


def _coupling_panel(ctx) -> Panel:
    """(b) Anterior translation of the condyle-point midpoint over the opening angle:
    participant median curves, cohort median with its 95 % bootstrap CI, and the range a
    two-phase model treats as pure rotation (shaded)."""
    from biomech.viz.views.kin_r8 import _coupling_curves

    p = Panel("(b) Condyle translation with opening", Axis("Opening angle", DEG),
              Axis("Anterior translation", MM), [], legend="inside:north west")
    g, S, med, rows = _coupling_curves(ctx)
    if g is None or S is None or not np.isfinite(S).any():
        return _empty(p)
    p.series.append(Series("hspan", x=np.array([0.0]), y2=np.array([PURE_ROTATION_DEG]),
                           role="faint"))
    # the band labelled in the panel (JNER revision 3.3), not in a legend below
    ylo = float(np.nanmin(S))
    p.series.append(Series("text", np.array([0.3]), np.array([ylo]),
                           text=["pure rotation in the two-phase model, "
                                 f"up to {PURE_ROTATION_DEG:g}°"], style={"anchor": "right"}))
    for y_ in S:
        if np.isfinite(y_).any():
            p.series.append(Series("line", g, y_, role="context", style={"width": 0.6}))
    if med is not None and np.isfinite(med).any():
        B.add_ci_band(p, g, S, centre="median", seed=B.seed_of(ctx), where=np.isfinite(med))
        p.series.append(Series("line", g, med, role="accent", style={"width": 1.4},
                               label="cohort median [95 % CI]"))
    p.series.append(Series("hline", y=np.array([0.0]), role="reference"))
    return p


def _sagittal(ctx, pid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The participant's incisor over the open-close phase (head frame, x and z) and its
    closed position (median below 1° opening)."""
    a = ctx.arrays("kinematics", pid)
    if "open_close__incisor" not in a or "open_close__theta_deg" not in a:
        return None
    inc, th = np.asarray(a["open_close__incisor"], float), np.asarray(a["open_close__theta_deg"])
    closed = np.isfinite(th) & (th < 1.0) & np.isfinite(inc).all(axis=1)
    if not closed.any():
        return None
    return inc[:, 0], inc[:, 2], np.nanmedian(inc[closed][:, [0, 2]], axis=0)


def _hinge_multiples(ctx) -> FigureSpec:
    h = ctx.table("kinematics", "hinge")
    pids = [q for q in ctx.pids() if not h.empty and (h.pid == q).any()]
    order = sorted(pids, key=ctx.pseudonym)
    cols = 5
    rows = int(np.ceil(len(order) / cols)) if order else 1
    panels, n_drawn = [], 0
    for k, pid in enumerate(order):
        r = h[(h.pid == pid)].set_index("model")
        p = Panel(f"P{ctx.pseudonym(pid)}", Axis("Anterior", MM), Axis("Superior", MM),
                  legend="none", equal=True, sharex="hinge")
        sg = _sagittal(ctx, pid)
        if sg is None:
            panels.append(_empty(p))
            continue
        x, z, inc0 = sg
        n_drawn += 1
        p.series.append(Series("line", x, z, role="context", style={"width": 0.6}))
        for model, lab, role, dash in HINGE_ARCS:
            if model in r.index and np.isfinite(r.loc[model, "axis_pierce_x_mm"]):
                ax, az = _hinge_arc(r.loc[model], inc0)
                p.series.append(Series("line", ax, az, role=role,
                                       style={"dash": dash, "width": 1.1},
                                       label=lab if k == 0 else None))
        if "hinge_fit" in r.index:
            p.note = f"{r.loc['hinge_fit', 'rms_incisor_mm']:.1f} mm"
        if k % cols:
            p.y.label, p.y.unit = "", ""
        if k < len(order) - cols:
            p.x.label, p.x.unit = "", ""
        panels.append(p)
    if panels:
        panels[0].legend = "below"
    spec = FigureSpec("icorr_hinge", "Jaw opening is not a hinge", panels, rows=rows,
                      cols=cols, width="text", height_in=1.25,
                      sources=["kinematics:arrays", "kinematics:hinge"])
    return attach(spec, CaptionSpec(
        "sagittal incisal path over the open-close task of every participant (grey; head "
        "frame, anterior to the right, origin at the closed incisor) with the arc a perfect "
        "hinge would draw: the best-fit fixed hinge (dashed) and the hinge through the "
        "palpated condyles (dotted), each from the closed position to that model's largest "
        "opening angle",
        [("number", "RMS misalignment of the best-fit hinge at the incisor")],
        f"n = {n_drawn} participants",
        "RMS distance of the incisal path from the arc, per participant",
        evidence="confirmatory (C1, best-fit hinge); condyle hinge secondary, post hoc test",
        notes=["The hinge through the condyles misses the path in every participant; the "
               "best-fit hinge follows it only because its axis lies far below and behind the "
               "condyles (the helical axis migrates during opening)."], codes=["C1"]))


def _design_marks(letter: str, design: str) -> list[tuple[str, str]]:
    """The design value of the translation (dashed) and its interval (band) in panel
    ``letter``: since E31 the one-sided lower 95 % bound, shaded from it to the value."""
    # no value in the caption (2026-11-10: values and verdicts are in the text)
    return [(f"dashed line in ({letter})", "the 95th percentile across participants, the "
             "design value"),
            (f"band in ({letter})", "from its one-sided lower 95 % bound to the value")]


def _design_span(r: "pd.Series") -> tuple[float, float] | None:
    """The shaded range of a design row: CI low to CI high, or with a one-sided lower bound
    (no upper end, E31) from the bound to the value; None without a bound."""
    lo, hi = float(r.ci_lo), float(r.ci_hi)
    hi = hi if np.isfinite(hi) else float(r.value)
    return (lo, hi) if np.isfinite(lo) and np.isfinite(hi) else None


def _design_text(r: "pd.Series") -> str:
    """'23 mm (lower bound 17)' / '23 mm [17, 29]' for the caption."""
    v, lo, hi = float(r.value), float(r.ci_lo), float(r.ci_hi)
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{v:.0f} mm [{lo:.0f}, {hi:.0f}]"
    return f"{v:.0f} mm" + (f", lower bound {lo:.0f} mm" if np.isfinite(lo) else "")


def _hinge_path3d(ctx, pc: Panel, pd_: Panel, n_above: int, n_all: int,
                  design: str) -> FigureSpec:
    pid = representative(ctx)
    p3 = Panel("(a) Path and helical axes", Axis("Anterior", MM), Axis("Left", MM),
               projection="3d", z=Axis("Superior", MM), view=(-82.0, 16.0), legend="below")
    a = ctx.arrays("kinematics", pid) if pid else {}
    fha = ctx.table("kinematics", "fha", [pid]) if pid else pd.DataFrame()
    h = ctx.table("kinematics", "hinge", [pid]) if pid else pd.DataFrame()
    lm = ctx.table("kinematics", "landmarks", [pid]) if pid else pd.DataFrame()
    if "open_close__incisor" in a:
        P = np.asarray(a["open_close__incisor"], float)
        p3.series.append(Series("line3d", P[:, 0], P[:, 1], z=P[:, 2], role="context",
                                style={"width": 0.5}, label="incisal path"))
    half = 30.0
    for early, role, lab in ((True, "cat:0", "helical axis ≤ 10°"),
                             (False, "cat:1", "helical axis > 10°")):
        g = fha[(fha.theta_to_deg <= 10) == early] if not fha.empty else fha
        for k, r in enumerate(g.itertuples()):
            n = np.array([r.n_x, r.n_y, r.n_z], float)
            c = np.array([r.pierce_x_mm, 0.0, r.pierce_z_mm], float)
            if not (np.isfinite(n).all() and np.isfinite(c).all()):
                continue
            e = np.vstack([c - half * n, c + half * n])
            p3.series.append(Series("line3d", e[:, 0], e[:, 1], z=e[:, 2], role=role,
                                    style={"width": 0.5}, label=lab if k == 0 else None))
    r = h[h.model == "hinge_fit"] if not h.empty else h
    if not r.empty:
        r = r.iloc[0]
        n = np.array([r.axis_x, r.axis_y, r.axis_z], float)
        c = np.array([r.axis_pierce_x_mm, 0.0, r.axis_pierce_z_mm], float)
        e = np.vstack([c - 1.4 * half * n, c + 1.4 * half * n])
        p3.series.append(Series("line3d", e[:, 0], e[:, 1], z=e[:, 2], role="accent",
                                style={"width": 1.4, "dash": "dashed"},
                                label="best-fit hinge"))
    if not lm.empty and "T_closed" in a:
        L = lm[lm.name.isin(["tmj_left", "tmj_right"])][["x", "y", "z"]].to_numpy(float)
        if L.size:
            Q = se3.apply(a["T_closed"], L)
            p3.series.append(Series("scatter3d", Q[:, 0], Q[:, 1], z=Q[:, 2], role="reference",
                                    style={"ms": "d", "size": 1.3}, label="condyles"))
    if not p3.series:
        _empty(p3)
    # The projected box overhangs its cell to the left with tick labels: the grid (25 mm)
    # carries the scale instead (caption).
    for ax in (p3.x, p3.y, p3.z):
        ax.ticks, ax.ticklabels = [-50.0, -25.0, 0.0, 25.0, 50.0], [""] * 5
    # (b) dumbbell: in-vivo noise-only P99 -> observed misalignment, per participant
    hn = ctx.table("uncertainty", "hinge_noise")
    pb = Panel("(b) Misalignment vs noise", Axis("RMS residual at the incisor", MM, log=True),
               Axis(""), legend="below")
    if not hn.empty and "obs_hinge_fit_mm" in hn:
        d = hn.dropna(subset=["obs_hinge_fit_mm"]).sort_values("obs_hinge_fit_mm")
        y = np.arange(len(d), dtype=float)
        for i, q in enumerate(d.itertuples()):
            if np.isfinite(q.null_invivo_p99_mm):
                pb.series.append(Series("line", np.array([q.null_invivo_p99_mm,
                                                          q.obs_hinge_fit_mm]),
                                        np.array([i, i], float), role="context",
                                        style={"width": 1.2}))
        pb.series += [
            Series("scatter", d.null_invivo_p99_mm.to_numpy(float), y, role="context",
                   style={"ms": "s", "size": 1.1}, label="noise only, in vivo (P99)"),
            Series("scatter", d.obs_hinge_fit_mm.to_numpy(float), y, role="accent",
                   style={"ms": "o", "size": 1.2}, label="observed")]
        pb.y.ticks, pb.y.ticklabels = list(y), B.pseudo_rows(ctx, list(d.pid))
        pb.y.reverse = True
        v = np.r_[d.null_invivo_p99_mm.to_numpy(float), d.obs_hinge_fit_mm.to_numpy(float)]
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:                            # markers at the ends stay inside the axis
            pb.x.lim = (float(v.min()) / 1.15, float(v.max()) * 1.15)
    else:
        _empty(pb)
    pd_.title = "(c) Translation to allow"
    who = f"P{ctx.pseudonym(pid)}" if pid else "no participant"
    hb = 2.1
    n_rows = max(len(pb.y.ticks or []), len(pd_.y.ticks or []))
    pb.height = pd_.height = B.rows_height(n_rows, per=0.125, extra=0.1, lo=hb) / hb
    spec = FigureSpec("icorr_hinge", "Jaw opening is not a hinge", [p3, pb, pd_], rows=2,
                      cols=2, width="text", height_in=hb,
                      meta={"above_invivo": n_above, "n": n_all, "pid": pid},
                      sources=["kinematics:arrays", "kinematics:fha", "kinematics:hinge",
                               "kinematics:landmarks", "uncertainty:hinge_noise",
                               "stats_kinematics:subjects", "requirements:requirements"])
    return attach(spec, CaptionSpec(
        "(a) Incisal path over the open-close task of one participant (" + who + ", head "
        "frame) with the finite helical axes of the opening steps up to 10° and beyond, the "
        "best-fit fixed hinge (dashed) and the palpated condyles (diamonds; grid 25 mm): the "
        "axis migrates during opening, so no single hinge reproduces the path. (b) Per "
        "participant, the RMS misalignment of the best-fit fixed hinge at the incisor against "
        "the 99th percentile a perfect hinge produces with that participant's in-vivo rest "
        "fluctuation. (c) Anterior excursion of the condyle points during opening (larger "
        "side) per participant",
        [("dots in (b)", "observed misalignment"), ("squares in (b)", "noise-only 99th "
                                                                     "percentile")]
        + _design_marks("c", design),
        f"n = {n_rows} participants",
        f"the residual exceeds its noise in {n_above}/{n_all} participants", codes=["C1"]))


@register("icorr_hinge", "paper", "Jaw opening is not a hinge",
          choices={"form": HINGE_FORMS}, form="panels")
def icorr_hinge(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    form = req.options.get("form", "panels")
    if form == "multiples":
        return _hinge_multiples(ctx)
    base = fig_kinematics(ViewRequest(ctx, perspective="cohort", stat="median_iqr"))
    pa, pb = base.panels[0], base.panels[1]
    helical = form == "panels_helical"           # the earlier layout: paths alone, FHA arrows
    arcs = [] if helical else _median_arcs(ctx)
    if arcs:
        # (a) the arcs each structure predicts for the cohort median, over the measured paths
        for _model, lab, role, x, z in arcs:
            pa.series.append(Series("line", x, z, role=role,
                                    style={"width": 1.2, "dash": "dashed"}, label=lab))
        for s in pa.series:
            if s.kind == "line" and s.role == "accent" and not s.label:
                s.label = "measured, cohort median"
        # the legend in the free cell width right of the narrow equal-aspect axis: an x axis
        # stretched to hold it inside compressed the paths
        pa.legend = "inside:outer north east"
        pa.title = "(a) Incisal path and hinge predictions"
    if not helical:
        pb = _coupling_panel(ctx)
    # The condyle origin as a legend entry instead of a text next to the marker.
    for sr in list(pb.series):
        if sr.kind == "text" and sr.text == ["condyles"]:
            pb.series.remove(sr)
        elif sr.kind == "scatter" and sr.role == "reference":
            sr.label = "palpated condyles (origin)"
    # (c) per participant: observed hinge misalignment [bootstrap CI] vs that participant's
    # noise-only 99th percentiles. Default form: both structures, the hinge at the joint
    # (through the condyle points, D33) leading and the one compromise axis (best fit) below
    # it, each against its own in-vivo null (kinematic-compatibility framing);
    # form "panels_fit": the best-fit hinge alone with its bench and in-vivo nulls.
    hn = ctx.table("uncertainty", "hinge_noise")
    pc = Panel("(c) Misalignment vs noise", Axis("RMS residual at the incisor", MM, log=True),
               Axis(""), legend="inside:south east")
    both = form in ("panels", "panels_helical") and not hn.empty and "obs_hinge_tmj_mm" in hn \
        and hn.obs_hinge_tmj_mm.notna().any()
    if both:
        # one participant order for (c) and (d): by the main outcome, the misalignment of the
        # hinge at the joint, largest at the top (JNER revision 3.0)
        d = hn.dropna(subset=["obs_hinge_tmj_mm"]).sort_values("obs_hinge_tmj_mm",
                                                               ascending=False)
        for col, ci, null, role, off, lab in (
                ("obs_hinge_tmj_mm", "obs_hinge_tmj", "null_invivo_tmj_p99_mm",
                 "model:hinge_tmj", -0.18, "hinge at the joint"),
                ("obs_hinge_fit_mm", "obs_hinge_fit", "null_invivo_p99_mm",
                 "model:hinge_fit", 0.18, "compromise axis")):
            y = np.arange(len(d), dtype=float) + off
            for i, r in enumerate(d.itertuples()):
                lo, hi = getattr(r, f"{ci}_ci_lo"), getattr(r, f"{ci}_ci_hi")
                if np.isfinite(lo) and np.isfinite(hi):
                    pc.series.append(Series("line", np.array([lo, hi]), np.array([y[i]] * 2),
                                            role=role, style={"width": 1.0}))
            # the noise tick on the row of its structure, in its colour (JNER revision 3.3)
            pc.series += [
                Series("scatter", d[null].to_numpy(float), y, role=role,
                       style={"ms": "|", "size": 1.5}),
                Series("scatter", d[col].to_numpy(float), y, role=role,
                       style={"size": 1.15}, label=lab)]
        nan = np.array([np.nan])                 # legend-only key of the noise ticks
        pc.series.append(Series("scatter", nan, nan, role="context",
                                style={"ms": "|", "size": 1.5}, label="noise only (P99)"))
        pc.x.lim = (0.2, 40.0)                   # room for the legend at the lower right
        pc.y.ticks = [float(i) for i in range(len(d))]
        pc.y.ticklabels = B.pseudo_rows(ctx, list(d.pid))
        pc.y.reverse = True
        from biomech.stats.cohort_kinematics import hinge_noise_counts

        n_above, n_all = hinge_noise_counts(d, "invivo")
        tmj_above, tmj_all = hinge_noise_counts(d, "invivo_tmj")
    elif not hn.empty and "obs_hinge_fit_mm" in hn:
        d = hn.dropna(subset=["obs_hinge_fit_mm"]).sort_values("obs_hinge_fit_mm")
        y = np.arange(len(d), dtype=float)
        for i, r in enumerate(d.itertuples()):
            if np.isfinite(r.obs_hinge_fit_ci_lo) and np.isfinite(r.obs_hinge_fit_ci_hi):
                pc.series.append(Series("line", np.array([r.obs_hinge_fit_ci_lo,
                                                          r.obs_hinge_fit_ci_hi]),
                                        np.array([i, i], float), role="accent",
                                        style={"width": 1.0}))
        pc.series += [
            Series("scatter", d.null_bench_p99_mm.to_numpy(float), y, role="faint",
                   style={"ms": "s", "size": 0.9}, label="noise only, bench (P99)"),
            Series("scatter", d.null_invivo_p99_mm.to_numpy(float), y, role="context",
                   style={"ms": "s", "size": 1.1}, label="noise only, in vivo (P99)"),
            Series("scatter", d.obs_hinge_fit_mm.to_numpy(float), y, role="accent",
                   style={"ms": "o", "size": 1.2}, label="observed [95 % CI]")]
        pc.y.ticks, pc.y.ticklabels = list(y), B.pseudo_rows(ctx, list(d.pid))
        pc.y.reverse = True
        from biomech.stats.cohort_kinematics import hinge_noise_counts

        n_above, n_all = hinge_noise_counts(d, "invivo")
    else:
        n_above, n_all = 0, 0
    hs = ctx.table("stats_kinematics", "hinge_noise_summary")     # the stored cohort count
    if not hs.empty and "above_invivo_n" in hs:
        n_above = int(hs.above_invivo_n.iloc[0])
        n_all = int(hs.n_tested_invivo.iloc[0] if "n_tested_invivo" in hs else hs.n.iloc[0])
    if not pc.series:
        _empty(pc)
    # (d) translation the joint must allow: condyle-point excursion per participant
    subj = ctx.table("stats_kinematics", "subjects")
    rq = ctx.table("requirements", "requirements") if "requirements" in ctx.store.stages() \
        else pd.DataFrame()
    pd_ = Panel("(d) Translation to allow", Axis("Condyle-point excursion", MM), Axis(""),
                legend="none")
    design = ""
    if not subj.empty:
        cond = pd.Series(subj[["condyle_left_disp_mm", "condyle_right_disp_mm"]].max(axis=1)
                         .to_numpy(float), index=subj.pid).dropna().sort_values()
        if both:                                 # the order of (c) (JNER revision 3.0)
            order = [p for p in d.pid if p in cond.index]
            cond = cond.reindex(order + [p for p in cond.index if p not in order])
        y = np.arange(len(cond), dtype=float)
        r = rq[rq.key == "translation_mm"] if not rq.empty else rq
        if not r.empty:
            r = r.iloc[0]
            span = _design_span(r)
            if span is not None:
                pd_.series.append(Series("hspan", x=np.array([span[0]]),
                                         y2=np.array([span[1]]), role="muted"))
            pd_.series.append(Series("vline", x=np.array([r.value]), role="threshold"))
            design = _design_text(r)
        pd_.series.append(Series("scatter", cond.to_numpy(), y, role="context",
                                 style={"size": 1.2}))
        if not r.empty and np.isfinite(r.value):
            pd_.series.append(Series("text", np.array([r.value]), np.array([-0.6]),
                                     text=["P95"], style={"anchor": "left"}))
        pd_.y.ticks, pd_.y.ticklabels = list(y), B.pseudo_rows(ctx, list(cond.index))
        pd_.y.reverse = True
    else:
        _empty(pd_)
    if form == "path3d":
        return _hinge_path3d(ctx, pc, pd_, n_above, n_all, design)
    if both:
        tests = ctx.table("stats_kinematics", "hinge_noise_tests")
        r = tests[(tests.noise == "invivo_tmj") & (tests.group == "all")] \
            if not tests.empty and "noise" in tests else tests
        if not r.empty:                          # the stored cohort count (D33)
            tmj_above, tmj_all = int(r.above_n.iloc[0]), int(r.n_tested.iloc[0])
    # n-aware: the per-participant rows of (c, d) keep >= 0.125 in per participant.
    h = 1.8
    n_rows = max(len(pc.y.ticks or []), len(pd_.y.ticks or []))
    pc.height = pd_.height = B.rows_height(n_rows, per=0.125, extra=0.1, lo=h) / h
    spec = FigureSpec("icorr_hinge", "Jaw opening is not a hinge", [pa, pb, pc, pd_], rows=2,
                      cols=2, width="text", height_in=h,
                      meta={"above_invivo": n_above, "n": n_all},
                      sources=base.sources + ["uncertainty:hinge_noise",
                                              "stats_kinematics:subjects",
                                              "requirements:requirements"]
                      + ([] if helical else ["stats_kinematics:coupling"]))
    # JNER revision 2026-11-10: under 300 words, "in grey" instead of "; ..." lists
    a_txt = ("(a) Sagittal incisal path of opening (participant medians in grey, cohort median "
             "in black) with the arcs a perfect hinge would draw for the cohort median, for the "
             "hinge at the joint (through the condyle points) and the compromise axis (best-fit "
             "fixed hinge), each from its median axis location to its median largest opening "
             "angle"
             if arcs else
             "(a) Sagittal incisal path of opening (participant medians in grey, cohort median "
             "in black)")
    b_txt = ("(b) Shift of the finite helical axis from early (≤ 10°) to further opening, "
             "relative to the palpated condyles, one arrow per participant" if helical else
             "(b) Anterior translation of the condyle-point midpoint over the opening angle "
             "(participant median curves in grey, cohort median with its 95 % bootstrap CI), "
             f"with the range up to {PURE_ROTATION_DEG:g}° that a two-phase model treats as "
             "pure rotation shaded")
    b_marks = [] if helical else [
        ("vermilion and blue dashed lines in (a)", "hinge at the joint and compromise axis"),
        ("grey dots in (d)", "participants")]
    b_notes = [] if helical else [
        "The condyle points are palpated on the skin, so (b) is a supporting description; the "
        "incisor-based misalignment in (c) is the primary evidence."]
    return attach(spec, CaptionSpec(
        f"{a_txt}. {b_txt}. (c) "
        + ("RMS misalignment at the incisal point of the two candidate structures per "
           "participant, each against the 99th percentile of the residual that structure "
           "produces with the participant's in-vivo rest fluctuation alone. Participants in "
           "(c) and (d) are ordered by the misalignment of the hinge at the joint" if both else
           "RMS misalignment at the incisal point of the best-fit fixed hinge per participant "
           "and the 99th percentile of the residual a perfect hinge produces with that "
           "participant's in-vivo rest fluctuation or with the bench noise")
        + ". (d) Anterior excursion of the condyle points during opening (larger side) per "
        "participant",
        ([("vermilion squares and blue dots in (c)", "hinge at the joint and compromise "
           "axis, observed, with the 95 % CI from a cycle and landmark bootstrap"),
          ("ticks in (c)", "noise-only 99th percentile, in vivo, of each structure, "
                           "in its colour on its row")]
         if both else
         [("dots in (c)", "observed misalignment (95 % CI from a cycle and landmark bootstrap)"),
          ("squares in (c)", "noise-only 99th percentiles, in vivo and bench")])
        + b_marks + _design_marks("d", design),
        f"n = {n_rows} participants",
        ("a participant's misalignment is beyond the noise when it exceeds that 99th "
         "percentile. The bench and the stricter noise models are in the supplement" if both
         else "the residual is beyond the noise when it exceeds its in-vivo 99th percentile"),
        notes=b_notes, codes=["C1", "D33"] + ([] if helical else ["X9"])))


def _ba_panel(ctx, c: pd.DataFrame, ba: pd.Series | None, col: str, title: str) -> Panel:
    p = Panel(title, Axis("Mean of IMU and optical opening", MM),
              Axis("IMU − optical", MM), legend="none", sharex="ba")
    if c.empty or col not in c:
        return _empty(p)
    mean = ((c[col] + c.opening_opt_mm) / 2).to_numpy(float)
    diff = (c[col] - c.opening_opt_mm).to_numpy(float)
    p.series.append(Series("scatter", mean, diff, role="context", style={"size": 0.7}))
    g = pd.DataFrame({"pid": c.pid.to_numpy(), "m": mean, "d": diff}).groupby("pid").mean()
    p.series.append(Series("scatter", g.m.to_numpy(float), g.d.to_numpy(float), role="accent",
                           style={"size": 1.1}))
    if ba is None:
        return p
    xr = np.array([np.nanmin(mean), np.nanmax(mean)])
    for k, lab in (("bias", "bias"), ("loa_lo", "LoA"), ("loa_hi", "LoA")):
        lo, hi = ba.get(f"{k}_ci_lo", np.nan), ba.get(f"{k}_ci_hi", np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            p.series.append(Series("band", xr, np.array([lo, lo]), y2=np.array([hi, hi]),
                                   role="band", style={"alpha": 0.15}))
        v = float(ba[k])
        p.series.append(Series("hline", y=np.array([v]),
                               role="reference" if k == "bias" else "threshold",
                               text=[f"{lab} {B.snum(v, 1)}"]))
    slope = ba.get("prop_slope", np.nan)
    if np.isfinite(slope):
        m0 = float(np.nanmean(mean))
        # neutral dark line: orange is the EMG colour (JNER revision 3.0, 2026-11-10)
        p.series.append(Series("line", xr, ba["bias"] + slope * (xr - m0), role="accent",
                               style={"width": 0.8, "color": "ink2"}))
        lo, hi = ba.get("prop_slope_ci_lo", np.nan), ba.get("prop_slope_ci_hi", np.nan)
        p.note = (f"slope {slope:+.3f}" + (f" [{lo:+.3f}, {hi:+.3f}]"
                                           if np.isfinite(lo) and np.isfinite(hi) else ""))
    return p


def _scatter_panel(c: pd.DataFrame, col: str, title: str, lim: tuple[float, float]) -> Panel:
    """IMU-estimated against optical opening per cycle (grey) and participant mean (black),
    with the identity line."""
    p = Panel(title, Axis("Optical opening", MM, lim=lim), Axis("IMU opening", MM, lim=lim),
              legend="none")
    if c.empty or col not in c:
        return _empty(p)
    p.series.append(Series("line", np.array(lim), np.array(lim), role="reference"))
    p.series.append(Series("scatter", c.opening_opt_mm.to_numpy(float), c[col].to_numpy(float),
                           role="context", style={"size": 0.7}))
    g = c.groupby("pid")[["opening_opt_mm", col]].mean()
    p.series.append(Series("scatter", g.opening_opt_mm.to_numpy(float), g[col].to_numpy(float),
                           role="accent", style={"size": 1.1}))
    p.equal = True
    return p


def _forest_panel(ctx, c: pd.DataFrame, ba: pd.Series | None, col: str, title: str,
                  order: list[str]) -> Panel:
    """Per participant: mean IMU − optical difference over the cycles with mean ± 1.96 SD of
    the cycle differences (that participant's limits); cohort bias and limits as lines with
    their CI bands."""
    p = Panel(title, Axis("IMU − optical", MM), Axis(""), legend="none")
    if c.empty or col not in c:
        return _empty(p)
    d = (c[col] - c.opening_opt_mm).groupby(c.pid)
    m, s = d.mean(), d.std(ddof=1)
    y = np.arange(len(order), dtype=float)
    if ba is not None:
        for k in ("bias", "loa_lo", "loa_hi"):
            lo, hi = ba.get(f"{k}_ci_lo", np.nan), ba.get(f"{k}_ci_hi", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                p.series.append(Series("hspan", x=np.array([lo]), y2=np.array([hi]),
                                       role="muted"))
            p.series.append(Series("vline", x=np.array([float(ba[k])]),
                                   role="reference" if k == "bias" else "threshold"))
    for i, pid in enumerate(order):
        if pid in m and np.isfinite(s.get(pid, np.nan)):
            p.series.append(Series("line", np.array([m[pid] - 1.96 * s[pid],
                                                     m[pid] + 1.96 * s[pid]]),
                                   np.array([i, i], float), role="context",
                                   style={"width": 1.2}))
    p.series.append(Series("scatter", np.array([m.get(q, np.nan) for q in order], float), y,
                           role="accent", style={"ms": "d", "size": 1.2}))
    p.y.ticks, p.y.ticklabels = list(y), B.pseudo_rows(ctx, order)
    p.y.reverse = True
    return p


IMU_FORMS = ("bland_altman", "scatter", "forest")


def _bias_label_side(p: Panel) -> None:
    """Bland–Altman labels sit at the right end (a left bias label collided with the y tick
    "0"), unless a limit of agreement lies within ~12 % of the drawn y range of the
    bias: then the bias label moves to the left end so the two labels do not overlap."""
    lines = [s for s in p.series if s.kind == "hline" and s.text]
    bias = [s for s in lines if s.text[0].startswith("bias")]
    loas = [float(s.y[0]) for s in lines if s.text[0].startswith("LoA")]
    if not bias or not loas or p.y.lim is None:
        return
    span = p.y.lim[1] - p.y.lim[0]
    if min(abs(v - float(bias[0].y[0])) for v in loas) < 0.12 * span:
        bias[0].style = {**bias[0].style, "anchor": "left"}


@register("icorr_imu", "paper", "IMU recovers incisal opening",
          choices={"form": IMU_FORMS}, form="bland_altman")
def icorr_imu(req: ViewRequest) -> FigureSpec:
    ctx = req.ctx
    form = req.options.get("form", "bland_altman")
    c = ctx.table("stats_imu", "imu_opening_cycles")
    c = c[c.task == "open_close"] if not c.empty else c
    ba = ctx.table("stats_imu", "imu_opening_ba")
    ag = ctx.table("stats_imu", "imu_agreement_cohort")
    q = ag[(ag.task == "open_close") & (ag.estimator == "quat")] if not ag.empty else ag
    angle = (f"the opening angle itself agrees within {q['median'].iloc[0]:.1f}° RMSE "
             f"(median; 95 % CI {B.ci(q.ci_lo.iloc[0], q.ci_hi.iloc[0], 1)}°)"
             if not q.empty else "")
    n_txt = (f"n = {c.pid.nunique()} participants, {len(c)} open-close cycles" if not c.empty
             else "")

    def row(model: str) -> pd.Series | None:
        r = ba[(ba.task == "open_close") & (ba.estimator == "quat") & (ba.model == model)] \
            if not ba.empty else ba
        return r.iloc[0] if not r.empty else None

    cols = (("loso", "opening_loso_quat_mm", "(a) Generic model"),
            ("subject", "opening_subject_quat_mm", "(b) Own model"))
    srcs = ["stats_imu:imu_opening_cycles", "stats_imu:imu_opening_ba",
            "stats_imu:imu_agreement_cohort"]
    who = ("per open-close cycle and participant mean. (a) Generic model, fitted on the other "
           "participants (an unseen user); (b) the participant's own model")
    dots = [("grey dots", "open-close cycles"), ("black dots", "participant means")]
    if form == "scatter":
        v = np.concatenate([c[k].to_numpy(float) for k in
                            ("opening_opt_mm", cols[0][1], cols[1][1]) if k in c]) \
            if not c.empty else np.array([0.0, 1.0])
        v = v[np.isfinite(v)]
        lim = (float(np.floor(v.min() / 5) * 5), float(np.ceil(v.max() / 5) * 5)) \
            if v.size else (0.0, 1.0)
        panels = [_scatter_panel(c, col, t, lim) for _m, col, t in cols]
        spec = FigureSpec("icorr_imu", "IMU recovers incisal opening", panels, rows=1, cols=2,
                          width="column", height_in=1.75, sources=srcs)
        return attach(spec, CaptionSpec(
            "incisal opening estimated from the two IMUs (quaternion opening angle and an "
            "angle-to-opening model) against the optical measurement, " + who,
            dots + [("grey line", "identity (perfect agreement)")], n_txt, angle,
            codes=["C3a"]))
    if form == "forest":
        order = list((c.opening_opt_mm.groupby(c.pid).mean().sort_values().index)
                     if not c.empty else [])
        panels = [_forest_panel(ctx, c, row(m), col, t, order) for m, col, t in cols]
        panels[1].y.ticklabels = [""] * len(order)
        xs = np.concatenate([np.asarray(v, float).ravel() for q in panels for sr in q.series
                             for v in (sr.x, sr.y2 if sr.kind == "hspan" else None)
                             if v is not None])
        if xs.size and np.isfinite(xs).any():
            lim = float(np.nanmax(np.abs(xs))) * 1.08
            for q in panels:
                q.x.lim = (-lim, lim)
        h = B.rows_height(len(order), per=0.13, extra=0.55, lo=1.4)
        spec = FigureSpec("icorr_imu", "IMU recovers incisal opening", panels, rows=1, cols=2,
                          width="column", height_in=h, sources=srcs)
        return attach(spec, CaptionSpec(
            "difference between the IMU-estimated and the optical incisal opening per "
            "participant (rows ordered by their mean optical opening). (a) Generic model, "
            "fitted on the other participants; (b) the participant's own model",
            [("diamond", "mean over the open-close cycles"),
             ("line", "that participant's mean ± 1.96 SD of the cycle differences"),
             ("vertical lines", "cohort bias (solid) and repeated-measures 95 % limits of "
                                "agreement (dashed)"),
             ("shaded", "their 95 % CIs (participant-cluster bootstrap)")], n_txt, angle,
            codes=["C3a"]))
    pa = _ba_panel(ctx, c, row("loso"), cols[0][1], cols[0][2])
    pb = _ba_panel(ctx, c, row("subject"), cols[1][1], cols[1][2])
    # one y range for both, so the two models compare at a glance
    # The y range covers the points and the CI bands of bias and limits of agreement.
    ys = [np.asarray(v, float) for p in (pa, pb) for s in p.series
          if s.kind in ("scatter", "band") for v in (s.y, s.y2) if v is not None]
    if ys:
        lim = float(np.nanmax(np.abs(np.concatenate(ys)))) * 1.15
        pa.y.lim = pb.y.lim = (-lim, lim)
    for p in (pa, pb):
        _bias_label_side(p)
    # A free strip at the right for the line labels ("bias", "LoA"): at the data's right end
    # they were printed over the cycles and participant means (2026-11-13). Both panels get
    # the same x range; the lines run into the strip, the CI bands end with the data.
    xs = [np.asarray(s.x, float) for p in (pa, pb) for s in p.series if s.kind == "scatter"]
    xs = np.concatenate(xs) if xs else np.empty(0)
    xs = xs[np.isfinite(xs)]
    if xs.size:
        lo_x, hi_x = float(xs.min()), float(xs.max())
        span = hi_x - lo_x
        pa.x.lim = pb.x.lim = (lo_x - 0.04 * span, hi_x + 0.24 * span)
    pa.x.label, pa.x.unit = "", ""        # stacked: the shared x label sits below (b)
    spec = FigureSpec("icorr_imu", "IMU recovers incisal opening", [pa, pb], rows=2, cols=1,
                      width="column", height_in=1.35, sources=srcs)
    return attach(spec, CaptionSpec(
        "Bland–Altman agreement of the incisal opening estimated from the two IMUs "
        "(quaternion opening angle and an angle-to-opening model) with the optical "
        "measurement, " + who,
        dots + [("dashed", "bias and repeated-measures 95 % limits of agreement"),
                ("bands", "their 95 % CIs (participant-cluster bootstrap)"),
                ("orange", "proportional bias (difference regressed on the mean)")],
        n_txt, angle, codes=["C3a"]))
