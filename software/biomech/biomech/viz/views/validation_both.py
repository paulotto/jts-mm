# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Both comparison takes in one figure (IEEE Sensors, 2026-11-05).

P1 and P2 are two separate reference measurements and are shown with equal weight: a main
figure that shows one take alone reads as selective next to tables that report both. These
views put the per-participant figures of the same result side by side on shared y-axes per
row (``sharey``), so an offset of one take is visible against the other:

* ``val_kin_waveforms_both`` -- cycle waveforms of both takes: form ``overlay`` (default) shows
  each take's cycles in its own column and the two differences JTS-MM − Qualisys overlaid in
  a third (P1 / P2 in vermillion / sky blue and solid / dashed); form ``columns`` repeats the single-take
  layout (cycles, difference) once per take;
* ``val_emg_profiles_both`` -- the EMG activation profiles of both takes: one row per muscle
  group, columns P1 right, P1 left, P2 right, P2 left.

The single-take views stay paper items (supporting / supplement).
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import numpy as np

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import FigureSpec, Panel
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.validation import _empty, _who

G = {"group": "validation", "domain": "validation"}
WAVE_FORMS = ("overlay", "columns", "overlay_all")
_STATS = re.compile(r"^(.*?)\s*\((r = .*)\)$")


def _subs(req: ViewRequest, view: str, **kw: Any) -> list[tuple[str, FigureSpec]]:
    """``view`` built once per participant (sorted pids), with its caption parts."""
    return [(pid, R.build(view, R.request(req.ctx, view, pid=pid, **kw)))
            for pid in sorted(req.ctx.pids())]


def _cap(spec: FigureSpec) -> dict[str, Any]:
    return dict((spec.meta or {}).get("caption") or {})


@register("val_kin_waveforms_both", "dashboard", "Cycle waveforms of both takes",
          params=("phase",), phases=("open_close", "chewing", "protrusion_retrusion",
                                     "left_right"),
          choices={"frames": ["aligned", "own"], "form": WAVE_FORMS}, form="overlay", **G)
def kin_waveforms_both(req: ViewRequest) -> FigureSpec:
    """``val_kin_waveforms`` of P1 and P2 in one figure, y-axes shared per channel row."""
    fid, title = "val_kin_waveforms_both", "Cycle waveforms of both takes"
    ctx = req.ctx
    task = req.phase if isinstance(req.phase, str) else "open_close"
    frames = str(req.options.get("frames", "aligned"))
    form = str(req.options.get("form", "overlay"))
    subs = _subs(req, "val_kin_waveforms", phase=task, frames=frames)
    if len(subs) < 2 or any(len(s.panels) < 2 for _, s in subs):
        return _empty(fid, title, f"no {V.phase_label(task)} cycles for both takes")
    who = [_who(ctx, pid) for pid, _ in subs]
    by = [{p.heading()[1]: p for p in s.panels} for _, s in subs]
    chans = [t for t in by[0] if not t.startswith("Δ ") and all(t in b and f"Δ {t}" in b
                                                                 for b in by)]
    panels: list[Panel] = []
    for i, ch in enumerate(chans):
        first = i == 0
        cyc = [b[ch] for b in by]
        dif = [b[f"Δ {ch}"] for b in by]
        for w, p in zip(who, cyc):
            p.title, p.sharey = f"{w}: {ch}", f"cycles{i}"
            p.legend = "below" if first and w == who[0] else "none"
        if form == "columns":
            for w, p in zip(who, dif):
                p.title, p.sharey, p.legend = f"{w}: Δ", f"diff{i}", "none"
            panels += [cyc[0], dif[0], cyc[1], dif[1]]
            continue
        d = Panel(f"Δ {ch}", dif[0].x, dataclasses.replace(dif[0].y),
                  [], legend="below" if first else "none")
        for j, (w, p) in enumerate(zip(who, dif)):
            role = f"participant:{j}"
            for s in p.series:
                if s.kind in ("band", "line"):
                    d.series.append(dataclasses.replace(
                        s, role=role, label=w if (s.kind == "line" and first) else None))
                elif j == 0:                      # the bounds once
                    d.series.append(s)
        lims = [p.y.lim for p in dif if p.y.lim]
        if lims:
            d.y.lim = (min(lo for lo, _ in lims), max(hi for _, hi in lims))
        if form == "overlay_all":
            panels += [_both_cycles(cyc, who, ch, first), d]
            continue
        panels += [cyc[0], cyc[1], d]
    if form == "overlay_all":                     # channels across: cycles row, Δ row
        panels = panels[0::2] + panels[1::2]
        rows, cols, h = 2, len(chans), 1.25
    else:
        cols = 4 if form == "columns" else 3
        rows, h = len(panels) // cols, 0.95
    spec = FigureSpec(fid, title, panels, rows=rows, cols=cols, width="text",
                      height_in=h, sources=list(dict.fromkeys(
                          src for _, s in subs for src in (s.sources or []))),
                      perspective=f"{' and '.join(who)}, {V.phase_label(task)} cycles",
                      meta={"combines": ["val_kin_waveforms"], "legend_span": "figure"})
    c0 = _cap(subs[0][1])
    marks = [m for m in (c0.get("marks") or []) if not str(m[0]).endswith("(right)")]
    layout = {"overlay": "first two columns: each take's time-normalized cycles per channel; "
                         "third column: the difference JTS-MM − Qualisys of both takes",
              "columns": "per take: time-normalized cycles per channel and their difference "
                         "JTS-MM − Qualisys",
              "overlay_all": f"top: the time-normalized cycles of both takes per channel "
                             f"({who[0]} vermillion, {who[1]} sky blue; JTS-MM solid, Qualisys "
                             "dashed); bottom: the difference JTS-MM − Qualisys of both "
                             "takes"}[form]
    if form == "overlay_all":
        marks = [m for m in marks if not str(m[0]).startswith(("solid", "dashed line and"))]
    marks += ([("line and band (third column)", f"mean difference and its 95 % band, {who[0]} "
                                                f"solid, {who[1]} dashed (clipped at the axis "
                                                "where few cycles reach a node)")]
              if form == "overlay" else
              [("line and band (bottom)", f"mean difference and its 95 % band, {who[0]} solid, "
                                          f"{who[1]} dashed")]
              if form == "overlay_all" else
              [("line and band (Δ)", "mean difference and its 95 % band (clipped at the axis "
                                     "where few cycles reach a node)")])
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of JTS-MM and Qualisys in both comparison takes, "
        + ("frames aligned" if frames == "aligned" else "each system in its own anatomical "
           "frames") + f"; {layout}; y-axes shared per channel. "
        + ("Frames aligned: constant maxillary and mandibular offsets fitted on the other "
           "tasks. " if frames == "aligned" else "")
        + "Displacement from the cycle start (the constant offset of the two systems' "
          "anatomical frames removed; it is reported separately)",
        marks, "; ".join(str(_cap(s).get("n") or "") for _, s in subs),
        statistic=c0.get("statistic") or "", test=c0.get("test") or "",
        notes=c0.get("notes") or []))


_SYS_DASH = {"system:jtsmm": ("solid", "JTS-MM"), "system:qualisys": ("dashed", "Qualisys"),
             "system:delsys": ("dotted", "Delsys")}


def _both_cycles(cyc: list[Panel], who: list[str], ch: str, first: bool) -> Panel:
    """Form ``overlay_all``: the cycles of both takes on one axis -- colour = take
    (``participant:i``), line style = system (JTS-MM solid, Qualisys dashed), one SPM bar row
    per take in the take's colour."""
    p = Panel(ch, cyc[0].x, dataclasses.replace(cyc[0].y, lim=None), [],
              legend="below" if first else "none")
    for j, (w, c) in enumerate(zip(who, cyc)):
        role = f"participant:{j}"
        for s in c.series:
            if s.kind == "sigbar":
                p.series.append(dataclasses.replace(
                    s, role=role, label=f"SPM: {w}" if first else None,
                    style={**(s.style or {}), "row": j}))
                continue
            dash, name = _SYS_DASH.get(s.role or "", ("solid", "JTS-MM"))
            n = re.search(r"\(n = [^)]*\)", s.label or "")      # keep the cycle count
            lab = (f"{w} {name}" + (f" {n.group(0)}" if n else "")
                   if (first and s.kind == "line" and s.label) else None)
            p.series.append(dataclasses.replace(
                s, role=role, label=lab, style={**(s.style or {}), "dash": dash}))
    return p


@register("val_emg_profiles_both", "dashboard", "EMG activation profiles of both takes",
          params=("phase",), phases=("open_close", "chewing"),
          choices={"form": ("columns", "overlay_all")}, form="columns", **G)
def emg_profiles_both(req: ViewRequest) -> FigureSpec:
    """``val_emg_profiles`` of P1 and P2 in one figure: a row per muscle group, columns P1
    right / left, P2 right / left, y-axes shared per row; r and CMC in the panel titles."""
    fid, title = "val_emg_profiles_both", "EMG activation profiles of both takes"
    ctx = req.ctx
    task = req.phase if isinstance(req.phase, str) else "open_close"
    subs = _subs(req, "val_emg_profiles", phase=task)
    if len(subs) < 2 or any(not s.panels or not s.id.startswith("val_emg_profiles")
                            for _, s in subs):
        return _empty(fid, title, f"no {V.phase_label(task)} profiles for both takes")
    who = [_who(ctx, pid) for pid, _ in subs]
    n = min(len(s.panels) for _, s in subs)
    pairs = n // 2                                # (right, left) per muscle group
    if req.options.get("form", "columns") == "overlay_all":
        return _emg_overlay(fid, title, subs, who, task, n)
    panels: list[Panel] = []
    for g in range(pairs):
        for k, (w, (_, s)) in enumerate(zip(who, subs)):
            for p in s.panels[2 * g:2 * g + 2]:
                m = _STATS.match(p.title)
                lab, stats = (m.group(1), m.group(2)) if m else (p.title, "")
                # "r = 0.59, CMC undefined*" -> "r 0.59, CMC –*": one line at a quarter width
                short = stats.replace(" = ", " ").replace("CMC undefined*", "CMC –*")
                p.title = f"{w}: {lab}" + (f"\n{short}" if short else "")
                p.sharey = f"group{g}"
                p.legend = "none"
                panels.append(p)
    if not panels:
        return _empty(fid, title, f"no {V.phase_label(task)} muscle pairs for both takes")
    for k in (0, 2):                              # one legend per take (cycle counts differ)
        if k < len(panels):
            panels[k].legend = "below"
    spec = FigureSpec(fid, title, panels, rows=pairs, cols=4, width="text", height_in=1.3,
                      sources=list(dict.fromkeys(src for _, s in subs
                                                 for src in (s.sources or []))),
                      perspective=f"{' and '.join(who)}, {V.phase_label(task)}",
                      meta={"combines": ["val_emg_profiles"]})
    c0 = _cap(subs[0][1])
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of each muscle in both comparison takes, both systems; "
        f"columns {who[0]} right, {who[0]} left, {who[1]} right, {who[1]} left; y-axes "
        "shared per muscle row",
        [(g, m) if g != "*" else ("–*", m) for g, m in (c0.get("marks") or [])],
        f"{' and '.join(who)}, cycle count n per system in the legend of each take",
        statistic=c0.get("statistic") or "", test=c0.get("test") or "",
        notes=c0.get("notes") or []))


def _emg_overlay(fid: str, title: str, subs: list, who: list[str], task: str,
                 n: int) -> FigureSpec:
    """Form ``overlay_all`` of ``val_emg_profiles_both``: one panel per muscle with both takes
    (colour = take, JTS-MM solid, Delsys dotted); r and CMC of each take in the title."""
    panels = []
    for k in range(n):
        src = [s.panels[k] for _, s in subs]
        parts = [_STATS.match(p.title) for p in src]
        lab = parts[0].group(1) if parts[0] else src[0].title
        stats = "\n".join(f"{w} " + m.group(2).replace(" = ", " ").replace("CMC undefined*",
                                                                         "CMC –*")
                          for w, m in zip(who, parts) if m)
        p = _both_cycles(src, who, lab, k == 0)
        p.title = lab + (f"\n{stats}" if stats else "")
        panels.append(p)
    # 1.05 in per row: the figure within 0.45 of the IEEE text height (drafts' request)
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(n / 3)), cols=3, width="text",
                      height_in=1.05, sources=list(dict.fromkeys(
                          src for _, s in subs for src in (s.sources or []))),
                      perspective=f"{' and '.join(who)}, {V.phase_label(task)}",
                      meta={"combines": ["val_emg_profiles"], "legend_span": "figure"})
    c0 = _cap(subs[0][1])
    return attach(spec, CaptionSpec(
        f"{V.phase_label(task)} cycles of each muscle in both comparison takes on one axis "
        f"({who[0]} vermillion, {who[1]} sky blue; JTS-MM solid, Delsys dotted)",
        [(g, m) if g != "*" else ("–*", m) for g, m in (c0.get("marks") or [])],
        f"{' and '.join(who)}, cycle count n per system in the legend",
        statistic=c0.get("statistic") or "", test=c0.get("test") or "",
        notes=c0.get("notes") or []))
