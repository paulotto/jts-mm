# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The systems-paper video: layout ``validation``, registered on import.

JTS-MM against the Qualisys reference on the same take (validation study), from the stored
validation results only (``ref_kinematics`` and ``ref_fusion`` arrays, on the JTS-MM time
base), nothing recomputed:

* left: the mandible of both systems as the triangle lower incisor -- condyles in the
  maxillary frame (3-D, and the sagittal and frontal planes below), JTS-MM blue solid, Qualisys dashed,
  with the incisor trails. JTS-MM is shown after the constant frame alignment (leave-one-
  task-out fit, VD3), so the remaining difference is the motion error, not the marking;
* right, scrolling: the opening rotation of both systems, the incisor disagreement, the rotation disagreement of JTS-MM optics and of the IMU alone (estimator
  I, VD10) -- drawn in full on the frames the stored numbers use (evaluated tasks, reference
  not gated; the same n as the tables), faint elsewhere -- and rows marking why a frame is
  not evaluated (reference quality gate, calibration tool in view);
* burned-in captions per task (default cues, or ``Animation.captions``) and a key.

It plays a participant of the validation study (``biomech animate --config
experiments/jtsmm_validation/study.yaml --layout validation --pid ...``); dataset studies
have no ``ref_*`` arrays and the layout says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from biomech.anim import layouts as Lo
from biomech.anim import source as S
from biomech.anim import supplementary as Sup
from biomech.anim.layers import Outline, Ribbon, RibbonRow, Scope, Trail, alpha_rgba, rgb
from biomech.anim.layouts import (HEADER_H, KEY_H, MM, Composition, Region, _header, _key,
                                  _lim, _now_line, _spec, _time_axis)
from biomech.anim.source import Clip
from biomech.anim.timeline import Timeline
from biomech.viz.spec import Axis, Panel, Series

SYS = (("s", "JTS-MM", "system:jtsmm"), ("q", "Qualisys", "system:qualisys"))
IMU_ROLE = "cat:2"                          # the IMU-alone estimator, as in the viewer
ERR_ROLE = "accent"
TRIANGLE = ("tmj_left", "incisor", "tmj_right")
KEY = [("line", "system:jtsmm", "JTS-MM (frames aligned)"),
       ("dash", "system:qualisys", "Qualisys (reference)"),
       ("line", IMU_ROLE, "IMU alone"), ("line", "faint", "Frame not evaluated"),
       ("square", "grey", "Why: reference gated or tool in view")]
CUES = [["JTS-MM (blue) and Qualisys (dashed): lower incisor and both condyles in the "
         "maxillary frame; JTS-MM after the constant frame alignment",
         "Right: the opening rotation of both systems, then their disagreement "
         "(JTS-MM optics; the IMU alone in green)",
         "Faint and grey: frames not evaluated (reference quality gate, calibration tool "
         "in view); every reported number uses only the others"]]


def _validation(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """The stored validation arrays of ``pid`` near the clip's segments (master time)."""
    a = ctx.arrays("ref_kinematics", pid)
    if "incisor_q_mm" not in a or "t_master_s" not in a:
        raise ValueError("no ref_kinematics arrays (validation study only)")
    t = np.asarray(a["t_master_s"], np.float64) * 1e6
    sel = S._near_segments(t, clip.segments)
    if not sel.any():
        return None
    out: dict[str, Any] = {"t_us": t[sel]}
    for s, _, _ in SYS:
        sfx = "_aligned" if s == "s" else ""
        for lm in TRIANGLE:
            out[f"{lm}_{s}"] = np.asarray(a[f"{lm}_{s}{sfx}_mm"], float)[sel]
        rot = f"rot_{s}_aligned_deg" if f"rot_{s}_aligned_deg" in a else f"rot_{s}_deg"
        out[f"rot_{s}"] = np.asarray(a[rot], float)[sel]
    out["err_incisor"] = np.asarray(a.get("err_incisor_aligned_mm", a["err_incisor_mm"]),
                                    float)[sel]
    out["err_rot"] = np.asarray(a.get("err_rot_aligned_deg", a["err_rot_deg"]), float)[sel]
    out["ct"] = np.asarray(a.get("ct_visible", np.zeros(len(t), bool)), float)[sel]
    # The frames every stored agreement number uses: inside the evaluated tasks and not
    # gated by the reference quality (ref_kinematics tables, e.g. segments_aligned n).
    gate = np.asarray(a.get("gate", np.zeros(len(t), bool)), bool)
    ev = np.asarray(a.get("evaluated", np.ones(len(t), bool)), bool)
    out["gate"] = gate[sel].astype(float)
    out["used"] = (ev & ~gate)[sel]
    try:
        f = ctx.arrays("ref_fusion", pid)
    except (OSError, ValueError, KeyError) as exc:
        S._skip("IMU-alone disagreement", pid, exc)
        f = {}
    key = "err_rot_aligned_I_deg" if "err_rot_aligned_I_deg" in f else "err_rot_I_deg"
    if key in f and len(f[key]) == len(t):
        out["err_rot_imu"] = np.asarray(f[key], float)[sel]
    return out


def _split(y: np.ndarray, used: np.ndarray) -> list[np.ndarray]:
    """(values on the evaluated frames, values elsewhere): the stored numbers use only the
    first; the second is drawn faint."""
    y = np.asarray(y, float)
    return [np.where(used, y, np.nan), np.where(used, np.nan, y)]


def _in(t: np.ndarray, clip: Clip) -> np.ndarray:
    return Lo._in_segments(t, clip)


def _triangle(v: dict[str, Any], s: str) -> np.ndarray:
    return np.stack([v[f"{lm}_{s}"] for lm in TRIANGLE], axis=1)          # (N, 3, 3)


def _limits(v: dict[str, Any], clip: Clip) -> list[tuple[float, float]]:
    sel = _in(v["t_us"], clip)
    P = np.concatenate([_triangle(v, s)[sel].reshape(-1, 3) for s, _, _ in SYS])
    return [_lim([P[:, d]], 6.0, 20.0) for d in range(3)]


def _jaw_panel(v: dict[str, Any], clip: Clip, key: str, dims: tuple[int, ...], tau: float,
               title: str) -> tuple[Panel, list[Any]]:
    lims = _limits(v, clip)
    sel = _in(v["t_us"], clip)
    if len(dims) == 3:
        p = Panel(title, Axis("x", MM, lim=lims[0]), Axis("y", MM, lim=lims[1]),
                  projection="3d", z=Axis("z", MM, lim=lims[2]), view=(-58.0, 18.0),
                  legend="none")
    else:
        a, b = dims
        p = Panel(title, Axis(*Lo._AX[a], lim=lims[a]), Axis(*Lo._AX[b], lim=lims[b]),
                  equal=True, legend="none")
    layers: list[Any] = []
    t = v["t_us"]
    for s, _, role in SYS:
        inc = v[f"incisor_{s}"]
        _, P = Lo._breaks(t[sel], inc[sel])
        if len(dims) == 3:
            p.series.append(Series("line3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], role="faint"))
        else:
            p.series.append(Series("line", x=P[:, dims[0]], y=P[:, dims[1]], role="faint"))
        layers.append(Outline(key, t, _triangle(v, s), dims, role, dashed=s == "q",
                              width=2.2))
        layers.append(Trail(key, t, inc, dims, tau, role, width=2.0, size=0.9))
    return p, layers


def _scope_panel(title: str, unit_axis: Axis, key: str, t: np.ndarray, cols: list[np.ndarray],
                 roles: list[str], labels: list[str], window, clip: Clip,
                 lo0: float | None = None, min_span: float = 4.0
                 ) -> tuple[Panel, list[Any]]:
    sel = _in(t, clip)
    lo, hi = _lim([c[sel] for c in cols], 0.5, min_span)
    if lo0 is not None:
        lo = lo0
    unit_axis.lim = (lo, hi)
    p = Panel(title, _time_axis(window, False), unit_axis, legend="below", sharex="t")
    for r, lab in zip(roles, labels):
        p.series.append(Series("line", x=np.array([]), y=np.array([]), role=r, label=lab))
    p.series.append(_now_line())
    return p, [Scope(key, t, np.column_stack(cols), roles, window)]


def validation(a: Lo.Animation, tl: Timeline) -> Composition:
    clip: Clip = a.clip
    v = S_require(clip)
    head, layers = _header(a, tl)
    regions = [head]
    win, tau = a.window_s, a.trail_s
    y0, bh = Sup._body(HEADER_H + Sup.GAP)
    lw = 0.42
    p3, l3 = _jaw_panel(v, clip, "jaw:0", (0, 1, 2), tau, "Mandible, both systems")
    regions.append(Region("jaw", (0.0, y0, lw, bh * 0.58), _spec("jaw", [p3], 1), zoom=1.2))
    layers += l3
    # Sagittal and frontal planes side by side below the 3-D view (equal scales).
    planes = []
    for k, (dims, title) in enumerate((((0, 2), "Sagittal plane"), ((1, 2), "Frontal plane"))):
        pp, lp = _jaw_panel(v, clip, f"plane:{k}", dims, tau, title)
        planes.append(pp)
        layers += lp
    regions.append(Region("plane", (0.0, y0 + bh * 0.59, lw, bh * 0.41),
                          _spec("plane", planes, rows=1, cols=2, equal_scale="shared")))
    t = v["t_us"]
    cand = [
        _scope_panel("Opening rotation", Axis("Rotation", "\\degree"), "right:0", t,
                     [v["rot_s"][:, 1], v["rot_q"][:, 1]], [r for _, _, r in SYS],
                     [n for _, n, _ in SYS], win, clip),
        _scope_panel("Incisor disagreement", Axis("|Δ|", MM), "right:2", t,
                     _split(v["err_incisor"], v["used"]), [ERR_ROLE, "faint"],
                     ["JTS-MM optics", "frames not evaluated"], win, clip, lo0=0.0,
                     min_span=2.0)]
    rot_cols, rot_roles, rot_labels = (_split(v["err_rot"], v["used"]), [ERR_ROLE, "faint"],
                                       ["JTS-MM optics", "frames not evaluated"])
    if "err_rot_imu" in v:
        rot_cols.append(v["err_rot_imu"])
        rot_roles.append(IMU_ROLE)
        rot_labels.append("IMU alone")
    cand.append(_scope_panel("Rotation disagreement", Axis("|Δ|", "\\degree"), "right:3", t,
                             rot_cols, rot_roles, rot_labels, win, clip, lo0=0.0,
                             min_span=2.0))
    rows = [RibbonRow("Reference gated", t, v["gate"], alpha_rgba("grey", a.theme)),
            RibbonRow("Tool in view", t, v["ct"], alpha_rgba("grey", a.theme))]
    rib = Panel("Frames not evaluated", _time_axis(win, True),
                Lo._rows_axis([r.name for r in rows]), legend="none", sharex="t")
    bg = (rgb("panel", a.theme) * 255).round().astype(np.uint8)
    rib.series.append(Series("image", img=np.tile(bg, (len(rows), 2, 1)),
                             extent=(win[0], win[1], -0.5, len(rows) - 0.5)))
    rib.series.append(_now_line())
    cand.append((rib, [Ribbon("right:4", rows, win, theme=a.theme)]))
    # One region per panel (weights = heights), axes aligned across the column.
    names = ["rot", "einc", "erot", "tool"]
    boxes = Lo._flow(list(zip(names, (1.3, 0.95, 1.1, 1.05))), lw + 0.01, y0,
                     1 - lw - 0.02, bh, gap=0.008)
    for n, (p, ls) in zip(names, cand):
        for ly in ls:
            ly.panel = f"{n}:0"
        regions.append(Region(n, boxes[n], _spec(n, [p], 1), align="right"))
        layers += ls
    cap, cap_layer = Sup._caption_band(a, tl, CUES)
    regions.append(cap)
    layers.append(cap_layer)
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(KEY, None)))
    return Composition("validation", regions, layers, tl)


def S_require(clip: Clip) -> dict[str, Any]:
    return Sup._require(clip, "validation", "validation")


def _register() -> None:
    S.EXTRAS["validation"] = _validation
    Lo.LAYOUTS["validation"] = validation
    Lo.EXTRAS["validation"] = ("validation",)
    Lo.PREVIEWS["validation"] = Sup._preview_from(["jaw", "plane", "rot", "erot"], film=1)
    Lo.DEFAULT_PHASES["validation"] = ("open_close",)
    Lo.STILLS.add("validation")


_register()

__all__ = ["validation"]
