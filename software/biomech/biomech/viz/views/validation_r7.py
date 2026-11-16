# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Systems-paper views of the validation additions.

* ``val_imu_alone`` (VD10): the IMU alone against Qualisys, next to JTS-MM optics, per task;
* ``val_sampling`` (VD9): what the dataset chain (40 Hz, 6 Hz low-pass) does to the peak
  velocities, and the movement bandwidth of the 200 Hz reference;
* ``val_table_filtered`` (VD12): the agreement after the dataset pose chain beside the
  unfiltered agreement;
* ``val_fusion_smoothing`` (VD16): why raw 40-Hz optical trajectories look ragged, and what
  a 6-Hz low-pass (the dataset's cut-off) and the IMU fusion do to them, against Qualisys at
  200 Hz;
* ``val_table_imu_lever`` (VD15): the IMU lever arms from the hand-eye sessions;
* ``val_imu_accel``, ``val_table_imu_accel`` (VD17): the accelerometer against the Qualisys
  acceleration of the IMU site.

Stored results only (``ref_fusion:imu_opening|smoothness`` and arrays,
``ref_sampling:bias|spectrum``, ``ref_kinematics:segments*``, ``ref_imu_lever``,
``ref_imu_accel``); nothing is recomputed.
"""

from __future__ import annotations

import numpy as np

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import _arrays, _empty, _pid, _spans, _table, _who, n_text
from biomech.viz.views.validation import pshape_text, shape_by_pid, shape_by_rows

G = {"group": "validation", "domain": "validation"}
DEG, HZ, MM = "\\degree", "\\hertz", "\\milli\\metre"
COMPARISONS = (("imu_vs_qualisys", "IMU alone vs Qualisys", "cat:2", "^"),
               ("optical_vs_qualisys", "JTS-MM optics vs Qualisys", "system:jtsmm", "o"),
               ("imu_vs_optical", "IMU alone vs JTS-MM optics", "muted", "s"))
METRICS = {"rmse_deg": ("RMSE", "RMS difference of the opening angle"),
           "bias_deg": ("Bias", "mean difference of the opening angle (first minus second)"),
           "sd_deg": ("SD", "SD of the difference (bias removed)")}
QUANTITY = {"peak_open_vel_dps": "opening velocity", "peak_close_vel_dps": "closing velocity",
            "peak_incisal_speed_mm_s": "incisal speed"}
# colour = chain (system colours); the marker is the participant's
PIPE = {"ref40": ("dataset chain on Qualisys at 40 Hz", "system:qualisys", "o"),
        "jtsmm": ("dataset chain on JTS-MM", "system:jtsmm", "o")}


def _upper(values) -> float:
    """Axis end a little beyond the largest finite value (shared by the participant panels)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(1.08 * v.max()) if v.size and v.max() > 0 else 1.0


def _tasks(values) -> list[str]:
    order = list(V.TASKS)
    vals = list(dict.fromkeys(values))
    return [t for t in order if t in vals] + [t for t in vals if t not in order]


@register("val_imu_alone", "dashboard", "IMU alone vs Qualisys: opening angle", **G,
          choices={"metric": tuple(METRICS), "form": ("offset_dots", "dumbbell", "limits")},
          metric="rmse_deg", form="offset_dots")
def val_imu_alone(req: ViewRequest) -> FigureSpec:
    """VD10: per participant and task the opening-angle agreement of the IMU alone with
    Qualisys, of JTS-MM optics with Qualisys and of the IMU with the optics. Forms
    ``dumbbell`` / ``limits``: ``validation_r8`` (J2.6)."""
    form = str(req.options.get("form", "offset_dots"))
    if form != "offset_dots":
        from biomech.viz.views.validation_r8 import imu_alone_form

        return imu_alone_form(req, form)
    fid, title = "val_imu_alone", "IMU alone vs Qualisys: opening angle"
    d = _table(req.ctx, "ref_fusion", "imu_opening")
    if d.empty:
        return _empty(fid, title, "no ref_fusion imu_opening table")
    metric = req.options.get("metric", "rmse_deg")
    mlab, mdesc = METRICS[metric]
    panels = []
    for i_pid, pid in enumerate(sorted(d["pid"].unique())):
        g = d[d["pid"] == pid]
        tasks = _tasks(g["segment"])
        p = Panel(_who(req.ctx, pid), Axis(mlab, DEG), Axis(""), [], legend="below")
        for k, (comp, lab, role, ms) in enumerate(COMPARISONS):
            x = [float(g[(g.segment == t) & (g.comparison == comp)][metric].iloc[0])
                 if ((g.segment == t) & (g.comparison == comp)).any() else np.nan
                 for t in tasks]
            y = np.arange(len(tasks), dtype=float) + (k - 1) * 0.18
            p.series.append(Series("scatter", np.asarray(x, float), y, role=role,
                                   style={"ms": ms, "size": 1.3}, label=lab))
        if metric == "bias_deg":
            p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        else:
            p.x.lim = (0.0, _upper(d[metric]))
        p.y.ticks = [float(i) for i in range(len(tasks))]
        p.y.ticklabels = [V.phase_label(t) for t in tasks]
        p.y.reverse = True
        shape_by_pid([p], i_pid)              # shape = participant, colour = comparison (X1)
        panels.append(p)
    for p in panels[:-1]:                     # one legend, under the last panel
        p.legend = "none"
    spec = FigureSpec(fid, title, panels, rows=1, cols=len(panels), width="text",
                      height_in=2.1, sources=["ref_fusion:imu_opening"])
    return attach(spec, CaptionSpec(
        f"opening-angle agreement per movement task ({mdesc}): the IMU alone (the "
        "calibrated head and mouth IMUs, no optical correction) against Qualisys, JTS-MM "
        "optics against Qualisys, and the IMU against the optics; tasks in recording order",
        [("colors", "IMU alone vs Qualisys (green), JTS-MM optics vs Qualisys (blue), IMU alone vs "
                    "JTS-MM optics (grey) (legend)"),
         ("circle / square", "panel of " + " / ".join(_who(req.ctx, x) for x in
                                                    sorted(d["pid"].unique())))],
        n_text(req.ctx, d["pid"], "all frames of each task"),
        statistic=f"per participant and task the {mdesc} over the task's frames, no pooling "
                  "over participants",
        evidence="post hoc (descriptive)"))


@register("val_sampling", "dashboard", "Sampling: peak velocities and bandwidth",
          choices={"form": ("panels", "quantities", "bandwidth")}, form="panels", **G)
def val_sampling(req: ViewRequest) -> FigureSpec:
    """VD9: (a) per participant, task and quantity the median ratio of the per-cycle peak to
    the 200 Hz reference with its cycle-bootstrap CI, for the dataset chain on Qualisys and
    on JTS-MM; (b) f95 of the reference's opening-rate power per task and participant with
    the 20 Hz Nyquist frequency of 40 Hz optics. Forms ``quantities`` / ``bandwidth``:
    ``validation_r8`` (J2.6; the acceleration ratios when stored)."""
    form = str(req.options.get("form", "panels"))
    if form != "panels":
        from biomech.viz.views.validation_r8 import sampling_form

        return sampling_form(req, form)
    fid, title = "val_sampling", "Sampling: peak velocities and bandwidth"
    b = _table(req.ctx, "ref_sampling", "bias")
    sp = _table(req.ctx, "ref_sampling", "spectrum")
    if b.empty and sp.empty:
        return _empty(fid, title, "no ref_sampling tables")
    # the velocity quantities of VD9; the acceleration ratio (VD24, noisy reference) is the
    # ``quantities`` form
    b = b[b.quantity.isin(list(QUANTITY))] if not b.empty else b
    rows = sorted({(r.pid, r.task, r.quantity) for r in b.itertuples()},
                  key=lambda x: (x[0], _tasks(b.task).index(x[1]), list(QUANTITY).index(x[2])))
    labels = [f"{_who(req.ctx, p)} {V.phase_label(t)}, {QUANTITY.get(q, q)}"
              for p, t, q in rows]
    pa = Panel("(a) Peak relative to Qualisys at 200 Hz", Axis("Ratio", ""), Axis(""), [],
               legend="below")
    for k, (pipe, (lab, role, ms)) in enumerate(PIPE.items()):
        off = -0.17 if k == 0 else 0.17
        for i, (p, t, q) in enumerate(rows):
            r = b[(b.pid == p) & (b.task == t) & (b.quantity == q) & (b.pipeline == pipe)]
            if r.empty:
                continue
            r = r.iloc[0]
            pa.series.append(Series("line", np.array([r.ratio_ci_lo, r.ratio_ci_hi], float),
                                    np.array([i + off] * 2), role=role, style={"width": 1.1}))
        est = [float(b[(b.pid == p) & (b.task == t) & (b.quantity == q) &
                       (b.pipeline == pipe)].ratio_median.iloc[0])
               if ((b.pid == p) & (b.task == t) & (b.quantity == q) &
                   (b.pipeline == pipe)).any() else np.nan for p, t, q in rows]
        pa.series.append(Series("scatter", np.asarray(est, float),
                                np.arange(len(rows), dtype=float) + off, role=role,
                                style={"ms": ms, "size": 1.3}, label=lab))
    # unlabelled (the label sat on the CIs); named in the caption
    pa.series.append(Series("vline", x=np.array([1.0]), role="reference"))
    pa.y.ticks, pa.y.ticklabels = [float(i) for i in range(len(rows))], labels
    pa.y.reverse = True
    shape_by_rows(pa, [p_ for p_, _, _ in rows], sorted({p_ for p_, _, _ in rows}))
    tasks = _tasks(sp.task) if not sp.empty else []
    # The unfiltered f95 includes reference noise (EV7 addendum): an upper bound of the
    # opening-rate spectrum, never quoted as the movement bandwidth.
    pb = Panel("(b) Qualisys spectrum, f95 (unfiltered)", Axis("f95", HZ), Axis(""), [],
               legend="below")
    for k, pid in enumerate(sorted(sp["pid"].unique()) if not sp.empty else []):
        g = sp[sp.pid == pid].set_index("task")
        x = [float(g.loc[t, "f95_hz"]) if t in g.index else np.nan for t in tasks]
        pb.series.append(Series("scatter", np.asarray(x, float),
                                np.arange(len(tasks), dtype=float) + (k - 0.5) * 0.2,
                                # participants by shape, in ink: blue / orange mean the
                                # systems in (a)
                                role="accent", style={"ms": "o" if k == 0 else "s",
                                                      "size": 1.3},
                                label=_who(req.ctx, pid)))
    pb.series.append(Series("vline", x=np.array([20.0]), role="threshold",
                            label="Nyquist 20 Hz"))
    pb.y.ticks = [float(i) for i in range(len(tasks))]
    pb.y.ticklabels = [V.phase_label(t) for t in tasks]
    pb.y.reverse = True
    # room right of the 20 Hz line for its label
    pb.x.lim = (0.0, 1.45 * _upper(list(sp.f95_hz) + [20.0]) if not sp.empty else 35.0)
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=max(2.4, 0.16 * len(rows) + 0.9),
                      sources=["ref_sampling:bias", "ref_sampling:spectrum"])
    return attach(spec, CaptionSpec(
        "what 40-Hz optics and the dataset processing do to peak velocities, and how much "
        "opening-rate bandwidth Qualisys shows: "
        "(a) per cycle the peak opening and closing angular velocity and the peak incisal "
        "speed through the dataset processing (40 Hz frames, 6 Hz zero-phase low-pass, "
        "0.15 s Savitzky–Golay derivative) relative to Qualisys at 200 Hz (20 Hz low-pass, "
        "35 ms window), for the chain applied to Qualisys resampled at 40 Hz and to JTS-MM; "
        "(b) the frequency "
        "below which 95 % of the opening-rate power of Qualisys at 200 Hz lies, per task, "
        "with the Nyquist frequency of 40 Hz optics (20 Hz); unfiltered, so it includes "
        "Qualisys noise and is an upper bound of the opening-rate spectrum, not the "
        "movement bandwidth. "
        "In protrusion and laterotrusion the opening angle hardly moves, so their spectrum "
        "reflects Qualisys noise",
        [("markers (a)", "median ratio, black = Qualisys at 40 Hz, blue = JTS-MM"),
         pshape_text(req.ctx, [p_ for p_, _, _ in rows]),
         ("lines (a)", "95 % bootstrap CI over cycles"),
         ("vertical line (a)", "ratio 1, equal to Qualisys at 200 Hz"),
         ("black markers (b)", "f95 per participant"),
         ("dashed line (b)", "Nyquist frequency of 40 Hz optics")],
        n_text(req.ctx, sorted(set(b.get("pid", [])) | set(sp.get("pid", [])))),
        statistic="(a) per participant, task and quantity the median ratio over the cycles "
                  "with its 95 % cycle-bootstrap CI; (b) one f95 per participant and task",
        evidence="post hoc (descriptive)",
        notes=["The ratios are a caveat for the peak velocities of 40 Hz optics, not a "
               "correction."]))


@register("val_table_filtered", "dashboard", "Agreement after the dataset pose chain", **G)
def val_table_filtered(req: ViewRequest) -> TableSpec:
    """VD12: pooled movement-task agreement without filtering and after the dataset pose
    chain (both systems filtered; JTS-MM only), own frames and frames aligned."""
    ctx = req.ctx
    f = _table(ctx, "ref_kinematics", "segments_filtered")
    s = _table(ctx, "ref_kinematics", "segments")
    a = _table(ctx, "ref_kinematics", "segments_aligned")
    rows = []
    for pid in sorted(set(f.get("pid", [])) | set(s.get("pid", []))):
        for frames in ("own", "aligned"):
            base = s[(s.pid == pid) & (s.segment == "movement_pooled") & s.gated] \
                if frames == "own" and not s.empty else \
                a[(a.pid == pid) & (a.segment == "movement_pooled")] if not a.empty else a
            entries = [("none", base)] + [
                (c, f[(f.pid == pid) & (f.frames == frames) & (f.chain == c) &
                      (f.segment == "movement_pooled")] if not f.empty else f)
                for c in ("both", "jtsmm_only")]
            for chain, r in entries:
                if r is None or r.empty:
                    continue
                r = r.iloc[0]
                rows.append({"pid": _who(ctx, pid), "frames": V.choice_label("frames", frames),
                             "chain": "not filtered" if chain == "none"
                             else V.term("filter_chain", chain),
                             "n": int(r["n"]), "incisor_rms_mm": float(r["incisor_rms_mm"]),
                             "incisor_dyn_rms_mm": float(r["incisor_dyn_rms_mm"]),
                             "rot_rms_deg": float(r["rot_rms_deg"])})
    cols = [Column("pid", "Participant", align="l"), Column("frames", "Frames", align="l"),
            Column("chain", "Filtered", align="l"), Column("n", "n", "", 0, "S"),
            Column("incisor_rms_mm", "Incisor RMS", MM, 2, "S"),
            Column("incisor_dyn_rms_mm", "Incisor RMS, bias removed", MM, 2, "S"),
            Column("rot_rms_deg", "Rotation RMS", DEG, 2, "S")]
    t = TableSpec("val_table_filtered", "", cols, rows, width="text",
                  sources=["ref_kinematics:segments_filtered", "ref_kinematics:segments",
                           "ref_kinematics:segments_aligned"])
    return attach(t, CaptionSpec(
        "jaw-in-head agreement over the pooled movement tasks (Qualisys quality gate "
        "applied) without filtering and after the dataset study's pose processing (gaps up "
        "to 0.5 s bridged, 6 Hz zero-phase low-pass), applied to both systems or to JTS-MM "
        "only; own frames and frames aligned (leave-one-task-out transforms of the primary "
        "analysis): RMS incisor and rotation difference", [],
        n_text(ctx, sorted(set(f.get("pid", [])) | set(s.get("pid", []))),
               "frames per row in column n"),
        statistic="per participant the RMS over all movement-task frames that pass the "
                  "Qualisys quality gate (tasks pooled within the participant)",
        evidence="sensitivity (post hoc)"))


# == VD16: smoothing of the optical trajectories ==================================================
SMOOTH = (("qualisys", "Qualisys, 200 Hz", "system:qualisys"),
          ("F0", "JTS-MM optical, 40 Hz", "system:jtsmm"),
          ("F0_lp6", "optical + 6 Hz low-pass", "cat:1"),
          ("F2", "optical + IMU, rotation", "cat:2"),
          ("F2P", "optical + IMU, rotation and position", "cat:4"))


RATES = {"mixed": "optics at its 40-Hz frames, Qualisys and fusion at 200 Hz",
         "frames": "all at the 40-Hz optical frames",
         "reference": "all at the 200-Hz Qualisys frames (optics linear between its frames)"}


@register("val_fusion_smoothing", "dashboard", "Fusion: smoothing the optical trajectory",
          params=("pid", "phase"), **G, window_s=4.0, rate="mixed",
          choices={"rate": tuple(RATES)})
def val_fusion_smoothing(req: ViewRequest) -> FigureSpec:
    """VD16: (a) the anterior incisor coordinate and (b) the sagittal incisor path over a few
    cycles of one held-out task, frame-aligned: raw 40-Hz optics, a 6-Hz zero-phase low-pass
    (the dataset's cut-off; 4th order per finite run, not the dataset chain), the position
    fusion at 200 Hz and Qualisys at 200 Hz; (c) the error jitter above 6 Hz (VD18) per
    estimator, participant and task."""
    from biomech.viz.views.validation import EST_SHORT

    fid, title = "val_fusion_smoothing", "Fusion: smoothing the optical trajectory"
    ctx = req.ctx
    pid = _pid(req)
    a = _arrays(ctx, "ref_fusion", pid) if pid else {}
    k = _arrays(ctx, "ref_kinematics", pid) if pid else {}
    sm = _table(ctx, "ref_fusion", "smoothness")
    if "t_q200_s" not in a or not k or sm.empty:
        return _empty(fid, title, "no smoothing arrays (ref_fusion, VD16)")
    phase = req.phase or "open_close"
    off = float(np.asarray(a["t_master_s"], float)[0] - np.asarray(k["t_s"], float)[0])
    span = next(((t0, t1) for n, t0, t1 in _spans(k) if n == phase), None)
    if span is None:
        return _empty(fid, title, f"no {phase} phase")
    w = float(req.options.get("window_s", 4.0))
    t0 = span[0] + 1.0
    t1 = min(span[1], t0 + w)
    tq = np.asarray(a["t_q200_s"], float) - off
    to = np.asarray(a["t_master_s"], float) - off
    kq, ko = (tq >= t0) & (tq <= t1), (to >= t0) & (to <= t1)
    rate = req.options.get("rate", "mixed")
    if rate == "reference":                  # every estimator at the 200-Hz reference frames
        series = {e: (tq[kq], a[f"incisor_q200_{e}_mm"][kq])
                  for e, *_ in SMOOTH if f"incisor_q200_{e}_mm" in a}
    elif rate == "frames":                   # every estimator at the 40-Hz optical frames
        series = {"qualisys": (to[ko], a["incisor_q_mm"][ko]),
                  **{e: (to[ko], a[f"incisor_aligned_{e}_mm"][ko])
                     for e in ("F0", "F0_lp6", "F2P")}}
    else:
        series = {"qualisys": (tq[kq], a["incisor_q200_qualisys_mm"][kq]),
                  "F0": (to[ko], a["incisor_aligned_F0_mm"][ko]),
                  "F0_lp6": (to[ko], a["incisor_aligned_F0_lp6_mm"][ko]),
                  "F2P": (tq[kq], a["incisor_q200_F2P_mm"][kq])}
    pa = Panel("(a) Anterior incisor position", Axis("Time in take", "\\second"),
               Axis("x (anterior)", MM), [], legend="below")
    pb = Panel("(b) Sagittal incisor path", Axis("x (anterior)", MM), Axis("z (superior)", MM),
               [], equal=True)
    for key, lab, role in SMOOTH:
        if key not in series:
            continue
        t, x = series[key]
        if not np.isfinite(x).any():
            continue
        st = {"width": 0.9 if key == "qualisys" else 0.6}
        pa.series.append(Series("line", t, x[:, 0], role=role, style=st, label=lab))
        pb.series.append(Series("line", x[:, 0], x[:, 2], role=role, style=st))
        if key == "F0" and rate != "reference":   # the 40-Hz samples themselves
            pa.series.append(Series("scatter", t, x[:, 0], role=role,
                                    style={"ms": "o", "size": 0.5}))
    # VD18: the reported jitter is that of the error against Qualisys (jitter_err_rms_mm);
    # Qualisys' own content above 6 Hz is the floor under it (dashed, per participant).
    order = [e for e, *_ in SMOOTH if e in set(sm.estimator) and e != "qualisys"]
    xs = np.arange(len(order), dtype=float)
    pc = Panel("(c) Difference jitter above 6 Hz", Axis("", ticks=list(xs),
               ticklabels=[EST_SHORT.get(e, e) for e in order]),
               Axis("RMS", MM), [], legend="below")
    pids = sorted(sm.pid.unique())
    for i, p_ in enumerate(pids):
        # tasks grey (one line each; the medians carry the result); participant = line style
        # (P1 solid, P2 dashed)
        # and marker. Not the task colours: open-close blue would read as JTS-MM in
        # this figure (colour harmonisation 2026-11-05)
        for seg, g in sm[sm.pid == p_].groupby("segment"):
            g = g.set_index("estimator").reindex(order)
            yv = g["jitter_err_rms_mm"].to_numpy(float)
            pc.series.append(Series("line", xs, yv, role="context",
                                    style={"width": 0.6, **({"dash": "dashed"} if i else {})},
                                    label="tasks (P1 solid, P2 dashed)" if (i == 0 and seg ==
                                                                            sorted(set(sm.segment))[0])
                                    else None))

        med = sm[(sm.pid == p_) & sm.estimator.isin(order)].groupby("estimator") \
            .jitter_err_rms_mm.median().reindex(order).to_numpy(float)
        pc.series.append(Series("scatter", xs, med, role="accent",
                                style={"ms": "o" if i == 0 else "s", "size": 1.2},
                                label=f"{_who(ctx, p_)} median over tasks"))
        floor = sm[(sm.pid == p_) & (sm.estimator == "qualisys")].jitter_rms_mm.median()
        if np.isfinite(floor):
            pc.series.append(Series("hline", y=np.array([floor]), role="reference",
                                    style={"dash": "dotted"}))
            pc.series.append(Series("text", x=np.array([xs[-1]]), y=np.array([floor]),
                                    text=[f"Qualisys, {_who(ctx, p_)}"],
                                    style={"anchor": "above"}))
    per = sm[sm.estimator.isin(order)].groupby(["estimator", "pid"]).jitter_err_rms_mm.median()
    mean = per.groupby("estimator").mean().reindex(order).to_numpy(float)
    pc.series.append(Series("scatter", xs, mean, role="reference", style={"size": 1.6},
                            label="mean of the participants' medians"))
    spec = FigureSpec(fid, title, [pa, pb, pc], rows=1, cols=3, width="text", height_in=2.2,
                      sources=["ref_fusion:arrays", "ref_fusion:smoothness",
                               "ref_kinematics:arrays"],
                      meta={"pid": pid, "phase": phase, "rate": rate})
    return attach(spec, CaptionSpec(
        f"frame-aligned incisor trajectory of {_who(ctx, pid)} over {t1 - t0:.0f} s of "
        f"{V.phase_label(phase).lower()} (held-out task) with (a) the anterior coordinate and "
        "(b) the sagittal path: the raw 40-Hz optical poses carry frame-to-frame noise; a "
        "6-Hz zero-phase low-pass (the dataset's cut-off) removes it, and the IMU fusion "
        "(Kalman smoother with position, 200 Hz) follows Qualisys between the optical frames. "
        "(c) RMS of the difference to Qualisys above 6 Hz at the 200-Hz Qualisys frames. "
        f"Time base of (a, b): {RATES[rate]}",
        [("colors (a, b)", "Qualisys, raw optics, 6-Hz low-pass and fusion (legend)"),
         ("dots (a)", "raw 40-Hz optical samples"),
         ("thin grey lines (c)", "one participant and movement task each, solid / dashed = "
                                 "first / second participant"),
         ("black circle / square (c)", "the participant's median over the tasks"),
         ("grey dots (c)", "mean of the two medians"),
         ("dotted lines (c)", "Qualisys' own content above 6 Hz, per participant (labeled)")],
        n_text(ctx, sm["pid"]),
        statistic="(a, b) raw signals of one task window, no summary; (c) RMS above 6 Hz per "
                  "participant and task, the median per participant and the descriptive mean "
                  "of the two medians (no interval)",
        evidence="post hoc (descriptive)"))


@register("val_table_imu_lever", "dashboard", "Table: IMU lever arm", **G)
def val_table_imu_lever(req: ViewRequest) -> TableSpec:
    """VD15: the primary lever-arm estimate per hand-eye session and IMU with its bootstrap
    CIs, the gravity check, the cut-off spread and the between-session difference."""
    ctx = req.ctx
    lv = _table(ctx, "ref_imu_lever", "lever")
    rp = _table(ctx, "ref_imu_lever", "repeatability")
    rows = []
    body = {"MP": "jaw (mandibular plate)", "HP": "head (maxillary plate)"}
    for r in (lv[lv.primary].sort_values(["body", "date"]).itertuples() if not lv.empty
              else []):
        ci = "; ".join(f"[{getattr(r, f'r_{ax}_ci_lo_mm'):.0f}, "
                       f"{getattr(r, f'r_{ax}_ci_hi_mm'):.0f}]" for ax in "xyz")
        rows.append({"imu": body.get(r.body, r.body), "date": str(r.date)[:10],
                     "fc_hz": float(r.fc_hz), "n_frames": int(r.n_frames),
                     "r": f"({r.r_x_mm:.0f}, {r.r_y_mm:.0f}, {r.r_z_mm:.0f})", "ci": ci,
                     "g_ratio": float(r.g_ratio), "fc_spread_mm": float(r.fc_spread_mm)})
    cols = [Column("imu", "IMU", align="l"), Column("date", "Session", align="l"),
            Column("fc_hz", "Cut-off", "\\hertz", 1, "S"), Column("n_frames", "Frames", "", 0, "S"),
            Column("r", "Lever arm (x, y, z)", MM, align="l"),
            Column("ci", "95 % CI per axis", MM, align="l"),
            Column("g_ratio", "|g| / 9.81", "", 3, "S"),
            Column("fc_spread_mm", "Cut-off spread", MM, 1, "S")]
    rep = "; ".join(f"{body.get(x.body, x.body)}: {x.diff_norm_mm:.0f} mm apart, mounting "
                    f"changed by {x.mount_change_deg:.0f}°" for x in rp.itertuples()) \
        if not rp.empty else ""
    t = TableSpec("val_table_imu_lever", "", cols, rows, width="text",
                  sources=["ref_imu_lever:lever", "ref_imu_lever:repeatability"])
    return attach(t, CaptionSpec(
        "IMU position on its plate (optical body frame, from the body origin) from the "
        "hand-eye sessions: accelerometer against the optical rigid-body acceleration "
        "(gravity, accelerometer bias and lever arm fitted together), cut-off 6 Hz, 1-s "
        "block bootstrap; cut-off spread = largest per-axis range over 4, 6 and 8 Hz"
        + (f". Between the sessions: {rep}" if rep else ""),
        [], f"{len(rows)} fits (one per hand-eye session and IMU; no participant), optical "
            "frames per fit in column Frames",
        statistic="least-squares lever arm per fit with a 95 % CI per axis from a 1-s block "
                  "bootstrap; |g| / 9.81: fitted gravity magnitude over the standard value",
        evidence="post hoc (descriptive)"))


# == VD17: accelerometer vs Qualisys ==============================================================
ACC_VARIANT = (("sensor", "accelerometer, Qualisys orientation", "system:jtsmm", "o"),
               ("no_lever", "without the lever arm", "muted", "s"),
               ("own_orientation", "gravity removed with the IMU's own orientation", "cat:4",
                "^"))
BODY = {"MP": "jaw IMU", "HP": "head IMU"}
MS2 = "\\milli\\metre\\per\\second\\squared"


@register("val_imu_accel", "dashboard", "Accelerometer vs Qualisys at the IMU site",
          params=("pid", "phase"), **G, body="MP", choices={"body": tuple(BODY)})
def val_imu_accel(req: ViewRequest) -> FigureSpec:
    """VD17: (a) a few seconds of one held-out task: the dynamic acceleration at the IMU site
    from Qualisys, from the accelerometer (reference orientation) and with the IMU's own
    orientation; (b) per participant, IMU and task the RMS error of each variant against the
    reference acceleration."""
    fid, title = "val_imu_accel", "Accelerometer vs Qualisys at the IMU site"
    ctx = req.ctx
    d = _table(ctx, "ref_imu_accel", "accel")
    if d.empty:
        return _empty(fid, title, "no ref_imu_accel table (VD17)")
    d = d[d.primary]
    pid = _pid(req)
    body = req.options.get("body", "MP")
    phase = req.phase or "open_close"
    a = _arrays(ctx, "ref_imu_accel", pid) if pid else {}
    k = _arrays(ctx, "ref_kinematics", pid) if pid else {}
    fu = _arrays(ctx, "ref_fusion", pid) if pid else {}
    pa = Panel(f"(a) {BODY.get(body, body).capitalize()}, {V.phase_label(phase).lower()}",
               Axis("Time in take", "\\second"), Axis("Dynamic acceleration (chip z)", MS2), [],
               legend="below")
    key = f"{body}__dyn_ref_mm_s2"
    if key in a and k and "t_master_s" in fu:
        off = float(np.asarray(fu["t_master_s"], float)[0] - np.asarray(k["t_s"], float)[0])
        span = next(((t0, t1) for n, t0, t1 in _spans(k) if n == phase), None)
        if span is not None:
            t = np.asarray(a["t_q_master_s"], float) - off
            kk = (t >= span[0] + 1.0) & (t <= min(span[1], span[0] + 5.0))
            for arr, lab, role, w in ((f"{body}__dyn_ref_mm_s2", "Qualisys at the IMU site",
                                       "system:qualisys", 0.9),
                                      (f"{body}__dyn_meas_mm_s2", ACC_VARIANT[0][1],
                                       "system:jtsmm", 0.6),
                                      (f"{body}__dyn_own_mm_s2", ACC_VARIANT[2][1], "cat:4", 0.6)):
                if arr in a:
                    pa.series.append(Series("line", t[kk], np.asarray(a[arr])[kk, 2],
                                            role=role, style={"width": w}, label=lab))
    if not pa.series:
        pa.note = "no trace for this participant, IMU and task"
    rows = sorted({(r.pid, r.body) for r in d.itertuples()})
    tasks = _tasks(d.task)
    pb = Panel("(b) RMS difference per held-out task", Axis("RMS", MS2), Axis(""), [],
               legend="below")
    labels, y0 = [], 0.0
    for p_, b_ in rows:
        for t_ in tasks:
            labels.append(f"{_who(ctx, p_)} {BODY.get(b_, b_)}, {V.phase_label(t_)}")
        g = d[(d.pid == p_) & (d.body == b_)]
        ys = y0 + np.arange(len(tasks), dtype=float)
        ref = [float(g[(g.task == t_) & (g.variant == "sensor")].rms_ref_mm_s2.iloc[0])
               if ((g.task == t_) & (g.variant == "sensor")).any() else np.nan for t_ in tasks]
        pb.series.append(Series("barh", np.asarray(ref, float), ys, role="faint",
                                label="Qualisys acceleration (RMS)" if y0 == 0 else None))
        for j, (var, lab, role, ms) in enumerate(ACC_VARIANT):
            x = [float(g[(g.task == t_) & (g.variant == var)].rms_err_mm_s2.iloc[0])
                 if ((g.task == t_) & (g.variant == var)).any() else np.nan for t_ in tasks]
            # the three variants side by side within the row (they overlapped)
            pb.series.append(Series("scatter", np.asarray(x, float), ys + (j - 1) * 0.22,
                                    role=role,
                                    style={"ms": ms, "size": 1.2},
                                    label=lab if y0 == 0 else None))
        y0 += len(tasks)
    pb.y.ticks, pb.y.ticklabels, pb.y.reverse = [float(i) for i in range(len(labels))], labels, True
    shape_by_rows(pb, [p_ for p_, _ in rows for _ in tasks],     # X1: shape = participant
                  sorted({p_ for p_, _ in rows}))
    spec = FigureSpec(fid, title, [pa, pb], rows=1, cols=2, width="text",
                      height_in=max(2.4, 0.13 * len(labels) + 0.9),
                      sources=["ref_imu_accel:accel", "ref_imu_accel:arrays",
                               "ref_kinematics:arrays", "ref_fusion:arrays"],
                      meta={"pid": pid, "phase": phase, "body": body})
    return attach(spec, CaptionSpec(
        "accelerometer of the JTS-MM IMUs against the Qualisys acceleration of the IMU site "
        "(200-Hz body pose with the hand-eye lever arm; both low-passed at 6 Hz), in the "
        "chip frame with gravity removed; chip rotation, gravity direction (|g| fixed), "
        "accelerometer bias and lag fitted on the other tasks, so each task is held out. "
        "(a) a few seconds of one task; (b) RMS difference per task for the accelerometer with the "
        "Qualisys orientation, the same without the lever arm, and with gravity removed by "
        "the IMU's own orientation (what a wearable alone obtains), against the RMS of the "
        "Qualisys acceleration (bars). At the P2 head IMU the Qualisys acceleration is limited "
        "by the nearly collinear head cluster",
        [("lines (a)", "Qualisys at the IMU site, accelerometer with the Qualisys "
                       "orientation, and with the IMU's own orientation (legend)"),
         ("bars (b)", "Qualisys acceleration (RMS)"),
         ("markers (b)", "RMS difference per variant, color = variant (legend), circle / square "
                         "= first / second participant")],
        n_text(ctx, d["pid"], f"{d['task'].nunique()} held-out tasks each"),
        statistic="(a) raw signals, no summary; (b) per participant, IMU and held-out task the "
                  "RMS over the task's frames, no pooling",
        evidence="post hoc (descriptive)"))


@register("val_table_imu_accel", "dashboard", "Table: accelerometer vs Qualisys", **G)
def val_table_imu_accel(req: ViewRequest) -> TableSpec:
    """VD17: per participant and IMU the median over the held-out tasks of each variant's
    error, the reference acceleration, variance explained, coherence and rest levels, with
    the calibration (lag, prior vs fitted rotation)."""
    ctx = req.ctx
    d = _table(ctx, "ref_imu_accel", "accel")
    c = _table(ctx, "ref_imu_accel", "calibration")
    rows = []
    if not d.empty:
        d = d[d.primary]
        c = c[c.primary & (c.variant == "sensor")] if not c.empty else c
        for (p_, b_), g in d.groupby(["pid", "body"]):
            s_ = g[g.variant == "sensor"]
            def med(v, col, g=g):
                return float(g[g.variant == v][col].median()) if (g.variant == v).any() \
                    else np.nan
            cc = c[(c.pid == p_) & (c.body == b_)] if not c.empty else c
            rows.append({"pid": _who(ctx, p_), "imu": BODY.get(b_, b_),
                         "rms_ref_mm_s2": float(s_.rms_ref_mm_s2.median()),
                         "rms_err_mm_s2": med("sensor", "rms_err_mm_s2"),
                         "no_lever_mm_s2": med("no_lever", "rms_err_mm_s2"),
                         "own_mm_s2": med("own_orientation", "rms_err_mm_s2"),
                         "r2_dyn": float(s_.r2_dyn.median()),
                         "coherence": float(s_.coherence.median()),
                         "rest_rms_meas_mm_s2": float(s_.rest_rms_meas_mm_s2.median()),
                         "rest_rms_ref_mm_s2": float(s_.rest_rms_ref_mm_s2.median()),
                         "lag_ms": float(cc.lag_ms.median()) if not cc.empty else np.nan,
                         "prior_vs_fit_deg": float(cc.prior_vs_fit_deg.median())
                         if not cc.empty else np.nan})
    cols = [Column("pid", "Participant", align="l"), Column("imu", "IMU", align="l"),
            Column("rms_ref_mm_s2", "Qualisys RMS", MS2, 0, "S"),
            Column("rms_err_mm_s2", "Difference", MS2, 0, "S"),
            Column("no_lever_mm_s2", "Difference, no lever arm", MS2, 0, "S"),
            Column("own_mm_s2", "Difference, own orientation", MS2, 0, "S"),
            Column("r2_dyn", "R²", "", 2, "S"), Column("coherence", "Coherence", "", 2, "S"),
            Column("rest_rms_meas_mm_s2", "Rest, IMU", MS2, 0, "S"),
            Column("rest_rms_ref_mm_s2", "Rest, Qualisys", MS2, 0, "S"),
            Column("lag_ms", "Lag", "\\milli\\second", 0, "S"),
            Column("prior_vs_fit_deg", "Prior vs fit", DEG, 1, "S")]
    t = TableSpec("val_table_imu_accel", "", cols, rows, width="text",
                  sources=["ref_imu_accel:accel", "ref_imu_accel:calibration"])
    return attach(t, CaptionSpec(
        "accelerometer against the Qualisys acceleration of the IMU site, median over the "
        "held-out tasks (6 Hz): RMS of the Qualisys dynamic acceleration, RMS difference with the "
        "Qualisys orientation, without the lever arm and with the IMU's own orientation, "
        "variance explained, coherence, rest levels of both systems, the fitted lag and the "
        "angle between the prior chip rotation (hand-eye mounting and registration) and the "
        "fitted one", [],
        n_text(ctx, d["pid"], f"{d['task'].nunique()} held-out tasks per row")
        if not d.empty else "",
        statistic="per participant and IMU the median over the held-out tasks of each task's "
                  "value (RMS over the task's frames; R²: variance explained)",
        evidence="post hoc (descriptive)",
        notes=["R² below 0: the variance of the difference exceeds the variance of the Qualisys "
               "acceleration. At the head the dynamic acceleration is small and the Qualisys "
               "acceleration depends on the head-cluster geometry, which is nearly collinear for "
               "one participant (marking-gap table)."]
        if any(np.isfinite(r["r2_dyn"]) and r["r2_dyn"] < 0 for r in rows) else []))
