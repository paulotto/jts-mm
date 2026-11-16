# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Signals of the validation study: both systems over one task of the take.

The validation study has no dataset stages (EMG envelopes, IMU opening, labels), so the
dataset signals rows stay empty there. This figure shows what the validation stages stored
per participant, on the JTS-MM time base (``ref_kinematics`` and ``ref_fusion`` arrays):

* lower incisor x, y, z and the opening rotation of both systems between their anatomical
  frames -- JTS-MM blue solid, Qualisys dashed (the system roles of every figure);
  ``frames="aligned"`` shows JTS-MM after the constant frame alignment (LOTO fit, VD3);
* the disagreement: incisor distance, and the rotation of JTS-MM optics and of the IMU
  alone (estimator I, VD10) against Qualisys -- in full on the frames the stored numbers
  use (evaluated tasks, reference not gated), faint elsewhere;
* frames with the calibration tool (CT) in view shaded (not evaluated).

Stored results only, nothing recomputed: the same arrays the validation views draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biomech.viz import style
from biomech.viz.render_plotly import template

XYZ = ("x (anterior)", "y (left)", "z (superior)")
ROWS = (("Incisor x (mm)", 0.15), ("Incisor y (mm)", 0.15), ("Incisor z (mm)", 0.15),
        ("Opening rotation (°)", 0.19), ("Incisor disagreement (mm)", 0.17),
        ("Rotation disagreement (°)", 0.19))
SYSTEMS = {"s": ("JTS-MM", "system:jtsmm"), "q": ("Qualisys", "system:qualisys")}
FRAMES = {"own": "each system's own frames", "aligned": "frames aligned (LOTO)"}
DASH = {"solid": "solid", "dashed": "dash", "dotted": "dot", "dashdot": "dashdot"}
MAX_SHADES = 400                   # CT-in-view runs drawn as shaded bands


def _arrays(ctx: Any, stage: str, pid: str) -> dict[str, np.ndarray]:
    try:
        return ctx.arrays(stage, pid)
    except (KeyError, FileNotFoundError, LookupError, OSError):
        return {}


def available(ctx: Any, pid: str | None) -> bool:
    return bool(pid) and "incisor_s_mm" in _arrays(ctx, "ref_kinematics", pid)


def _line(role: str, theme: str, width: float = 1.4) -> dict[str, Any]:
    col, dash, _ = style.role(role)
    return {"color": style.hexcolor(col, theme), "width": width, "dash": DASH.get(dash, "solid")}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    from biomech import timebase

    return [(int(a), int(b)) for a, b in timebase.segments(np.asarray(mask, bool))]


def validation_figure(ctx: Any, pid: str, phase: str | None, frames: str = "aligned",
                      theme: str = "light") -> go.Figure:
    """Both systems over task ``phase`` (``None`` or "all": the whole take), seconds from
    the task start. ``frames``: "own" or "aligned" (JTS-MM after the frame alignment)."""
    a = _arrays(ctx, "ref_kinematics", pid)
    f = _arrays(ctx, "ref_fusion", pid)
    fig = make_subplots(rows=len(ROWS), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                        row_heights=[h for _, h in ROWS])
    fig.update_layout(template=template(theme), margin={"l": 64, "r": 16, "t": 8, "b": 40},
                      hovermode="x unified", showlegend=True,
                      legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 0,
                              "bgcolor": "rgba(0,0,0,0)"})
    ink2 = style.hexcolor("ink2", theme)
    if "incisor_s_mm" not in a:
        fig.add_annotation(text="no stored validation kinematics for this participant",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font={"color": ink2})
        return fig
    t = np.asarray(a["t_s"], float)
    ph = np.asarray(a.get("phase", np.array([""] * len(t)))).astype(str)
    sel = np.ones(len(t), bool) if phase in (None, "", "all") else ph == phase
    if not sel.any():
        fig.add_annotation(text=f"no validation samples in '{phase}'", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False, font={"color": ink2})
        return fig
    t0 = float(t[sel][0])
    ts = t[sel] - t0
    sfx = "_aligned" if frames == "aligned" and "incisor_s_aligned_mm" in a else ""
    for k in range(3):
        for s, (name, role) in SYSTEMS.items():
            key = f"incisor_{s}{sfx if s == 's' else ''}_mm"
            fig.add_trace(go.Scattergl(x=ts, y=a[key][sel, k], mode="lines", name=name,
                                       legendgroup=name, showlegend=k == 0,
                                       line=_line(role, theme),
                                       hovertemplate=f"{name} {XYZ[k]}: %{{y:.2f}} mm"
                                                     "<extra></extra>"),
                          row=k + 1, col=1)
    for s, (name, role) in SYSTEMS.items():
        key = f"rot_{s}{sfx}_deg" if f"rot_{s}{sfx}_deg" in a else f"rot_{s}_deg"
        fig.add_trace(go.Scattergl(x=ts, y=a[key][sel, 1], mode="lines", name=name,
                                   legendgroup=name, showlegend=False, line=_line(role, theme),
                                   hovertemplate=f"{name} opening: %{{y:.2f}}°<extra></extra>"),
                      row=4, col=1)
    # The stored agreement numbers use the frames inside the evaluated tasks that the
    # reference quality gate keeps; the others are drawn faint.
    gate = np.asarray(a.get("gate", np.zeros(len(t), bool)), bool)
    used = (np.asarray(a.get("evaluated", np.ones(len(t), bool)), bool) & ~gate)[sel]
    for row, key, name, unit in ((5, f"err_incisor{sfx}_mm", "|Δ incisor|", "mm"),
                                 (6, f"err_rot{sfx}_deg", "|Δ rotation|", "°")):
        y = np.asarray(a.get(key, a[key.replace(sfx, "")]), float)[sel]
        for part, role, lab, show in ((used, "accent", f"{name}, JTS-MM optics", True),
                                      (~used, "faint", "frames not evaluated", row == 5)):
            fig.add_trace(go.Scattergl(x=ts, y=np.where(part, y, np.nan), mode="lines",
                                       name=lab, legendgroup=lab, showlegend=show,
                                       line=_line(role, theme, 1.0),
                                       hovertemplate=f"{lab}: %{{y:.2f}} {unit}"
                                                     "<extra></extra>"),
                          row=row, col=1)
    key_i = f"err_rot{sfx}_I_deg" if f"err_rot{sfx}_I_deg" in f else "err_rot_I_deg"
    if key_i in f and len(f[key_i]) == len(t):
        fig.add_trace(go.Scattergl(x=ts, y=np.asarray(f[key_i])[sel], mode="lines",
                                   name="|Δ rotation|, IMU alone",
                                   line=_line("cat:2", theme, 1.0),
                                   hovertemplate="|Δ rotation|, IMU alone: %{y:.2f}°"
                                                 "<extra></extra>"),
                      row=6, col=1)
    ct = np.asarray(a.get("ct_visible", np.zeros(len(t), bool)), bool)[sel]
    runs = _runs(ct)
    if 0 < len(runs) <= MAX_SHADES:
        shade = style.hexcolor("faint", theme)
        # One band per run across all rows (the rows share x), not one per row.
        fig.update_layout(shapes=[{"type": "rect", "xref": "x", "yref": "paper",
                                   "x0": float(ts[i]), "x1": float(ts[max(j - 1, i)]),
                                   "y0": 0, "y1": 1, "fillcolor": shade, "opacity": 0.45,
                                   "line": {"width": 0}, "layer": "below"}
                                  for i, j in runs])
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="CT in view "
                                 "(not evaluated)", marker={"symbol": "square", "size": 10,
                                                            "color": shade}), row=1, col=1)
    for r, (title, _) in enumerate(ROWS, start=1):
        fig.update_yaxes(title_text=title, title_font={"size": 11}, row=r, col=1)
    what = "whole take" if phase in (None, "", "all") else str(phase).replace("_", "-")
    fig.update_xaxes(title_text=f"Time from the start of the {what} (s); {FRAMES[frames]}",
                     row=len(ROWS), col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor=ink2, spikedash="dot")
    return fig
