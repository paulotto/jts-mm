# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The D5 supplementary videos: three layouts registered in :data:`layouts.LAYOUTS` (an11).

``v1_hinge``    "Not a hinge": the measured mandible and a ghost mandible moved by the
                participant's best-fit fixed hinge (C1, stage kinematics, table ``hinge``),
                the live residual vector at the incisor, the instantaneous helical axis,
                and a sagittal inset (opening angle vs incisal displacement).
``v2_intent``   "Intent cues in simulated real time": closer envelope with the causal
                release threshold and decisions (X3), opener envelope, opening angle
                (optical and IMU), label ribbon with the stream decoder's decisions
                (stage ``ml_online``), live counters (latency, false activations). Every
                scrolling panel shows only the past (window [-before, 0] s) and the frame
                carries the banner "simulated real time, no device".
``v3_release``  Open-close vs chewing release behaviour: two phases side by side, played in
                step, on the same axes (closer envelope, threshold, decisions, opening
                angle, running decision counts).

Each has per-segment burned-in captions (:class:`~biomech.anim.layers.Caption`; default
cues per layout, or ``Animation.captions``) and gets a PNG filmstrip next to the video
(:data:`layouts.STILLS`, :func:`biomech.anim.render_video.filmstrip_png`). The layouts reuse
the panels and layers of :mod:`biomech.anim.layouts`; the derived data (hinge ghost, release
envelope, stream decisions) are loaded once by :mod:`biomech.anim.source` (``EXTRAS``) and
travel with the clip (also into the ``.clip.npz`` files, :mod:`biomech.anim.clipfile`).

No exoskeleton or closed-loop device exists: the decoder decisions are an offline replay of
recorded data (pseudo-online, ml08), never a device's actions.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np

from biomech.anim import layouts as Lo
from biomech.anim.layers import (LABEL_COLORS, OUTLINE, Caption, Chip, Events, Layer, Marker,
                                 Outline, Pose, RibbonRow, Scope, Shifted, Tally, Trail,
                                 categorical_rgba, clip_segments)
from biomech.anim.layouts import (CLASS_NAMES, HEADER_H, KEY_H, KIN_KEY, MM, OPENERS,
                                  Composition, Region, _header, _in_segments, _key, _lim, _nice,
                                  _now_line, _spec, _speed_vmax, _time_axis, angle_panel,
                                  emg_panel, jaw3d_panel, plane_panel, ribbon_panel)
from biomech.anim.source import LANDMARKS, PHASE_LABEL, Clip
from biomech.anim.timeline import Timeline
from biomech.viz import style
from biomech.viz.spec import Axis, Panel, Series

BANNER = "simulated real time, no device"
HINGE_ROLE = "anim:fixed_hinge"        # yellow: distinct from the helical axis (vermillion)
RESID_ROLE = "tertiary"                # green (V1 hides the frame triad, which uses green)
X3_ROLE = "anim:release_x3"            # vermillion ▼: causal release decisions (X3)
STREAM_ROLE = "anim:release_stream"    # purple |: the stream replay's release decisions
CAPTION_H = 0.072
BANNER_H = 0.04
GAP = 0.012


def _require(clip: Clip, name: str, layout: str) -> dict[str, Any]:
    d = clip.extra.get(name)
    if d is None:
        raise ValueError(f"{clip.label}: layout {layout!r} needs the {name!r} data, which "
                         "could not be loaded (see the log; has the stage run?)")
    return d


# -- captions -------------------------------------------------------------------------------------
def spread_cues(tl: Timeline, per_segment: list[list[str]]) -> tuple[tuple[float, float, str], ...]:
    """Cues (v0_s, v1_s, text): segment ``j``'s texts share its video span equally (the last
    list is reused for further segments)."""
    out = []
    starts = tl.video_starts_s
    durs = tl.master_durations_s / tl.speed
    for j in range(len(tl.segments)):
        texts = per_segment[min(j, len(per_segment) - 1)] if per_segment else []
        if not texts:
            continue
        step = durs[j] / len(texts)
        for k, txt in enumerate(texts):
            out.append((float(starts[j] + k * step), float(starts[j] + (k + 1) * step), txt))
    return tuple(out)


def _caption_band(a: Lo.Animation, tl: Timeline, default: list[list[str]]
                  ) -> tuple[Region, Layer]:
    cues = a.captions if a.captions is not None else spread_cues(tl, default)
    box = (0.0, 1 - KEY_H - 0.006 - CAPTION_H - 0.004, 1.0, CAPTION_H)

    def draw(fig, theme, sc):
        ax = fig.add_axes([0.006, 0.06, 0.988, 0.88])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhspan(0, 1, color=style.hexcolor("panel", theme), lw=0)
        fig._anim_axes = {}

    layer = Caption("caption", [(0.5, 0.5, "center", lambda s: s["caption"], "label", "ink")],
                    cues=tuple(cues), starts_s=tuple(float(v) for v in tl.video_starts_s),
                    seg_t0_us=tuple(float(s0) for s0, _ in tl.segments), speed=tl.speed,
                    wrap=150)
    return Region("caption", box, draw=draw), layer


def _banner(a: Lo.Animation) -> Region:
    text = BANNER if abs(a.speed - 1.0) < 1e-9 else f"{BANNER} (played at {a.speed:g}×)"

    def draw(fig, theme, sc):
        fig.text(0.5, 0.5, text, ha="center", va="center", fontweight="bold",
                 fontsize=style.FONT_PT["label"] * sc * 1.2,
                 color=style.hexcolor("surface", theme),
                 bbox={"boxstyle": "round,pad=0.3,rounding_size=0.4",
                       "fc": style.hexcolor("vermillion", theme), "ec": "none"})

    return Region("banner", (0.0, HEADER_H + 0.004, 1.0, BANNER_H), draw=draw)


def _body(top: float) -> tuple[float, float]:
    """(y0, height) of the body between ``top`` and the caption band."""
    bottom = 1 - KEY_H - 0.006 - CAPTION_H - 0.004 - GAP
    return top, bottom - top


# -- V1: not a hinge -------------------------------------------------------------------------------
def _residual_gain(res_mm: np.ndarray, span_mm: float) -> float:
    """Display gain of the residual arrow: about 8 % of the panel span at the 97th
    percentile (1, 2, 5, 10, ... x); 1 when the residual is visible as it is."""
    r = res_mm[np.isfinite(res_mm)]
    q = float(np.percentile(r, 97)) if r.size else 0.0
    if q <= 0 or q >= 0.08 * span_mm:
        return 1.0
    want = 0.08 * span_mm / q
    for g in (1, 2, 5, 10, 20, 50, 100):
        if g >= want:
            return float(g)
    return 100.0


def _v1_layers(clip: Clip, h: dict[str, Any], key: str, dims: tuple[int, ...], gain: float
               ) -> list[Layer]:
    t = h["t_us"]
    inc = np.asarray(clip.kin["incisor"], float)
    base = inc - gain * h["residual"]                    # arrow: (scaled) ghost -> measured
    ghost = h["ghost"][:, [LANDMARKS.index(n) for n in OUTLINE]]
    return [Outline(key, t, ghost, dims, HINGE_ROLE, dashed=True, width=2.2),
            Outline(key, t, np.stack([base, inc], axis=1), dims, RESID_ROLE, dashed=False,
                    width=3.0, head=True, nodes=False)]


def _hinge_axis_series(h: dict[str, Any], lims) -> list[Series]:
    n, s = h["axis_n"], h["axis_s"]
    seg = np.array([[s - 200 * n, s + 200 * n]])
    seg, inside = clip_segments(seg, lims)
    if not inside[0]:
        return []
    q = seg[0]
    return [Series("line3d", x=q[:, 0], y=q[:, 1], z=q[:, 2], role=HINGE_ROLE,
                   style={"width": 1.6})]


AXIS_VIEW_MM = 80.0      # V1 widens the jaw views to show the fixed axis up to this far out


def _with_axis_in_view(clip: Clip, s: np.ndarray) -> tuple[Clip, bool]:
    """The clip with the fixed axis point added to the landmarks the jaw views are scaled
    to (an extra maxillary key, not drawn) when it lies within ``AXIS_VIEW_MM`` of the box;
    returns (clip, whether it is in view)."""
    lims = Lo._jaw_limits(clip)
    out = max(max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(s, lims))
    if out == 0.0:
        return clip, True
    if out > AXIS_VIEW_MM:
        return clip, False
    return dataclasses.replace(clip, maxilla={**(clip.maxilla or {}),
                                              "_fixed_axis": np.asarray(s, float)}), True


def v1_hinge(a: Lo.Animation, tl: Timeline) -> Composition:
    clip: Clip = a.clip
    if clip.kin is None:
        raise ValueError(f"{clip.label}: no kinematics for {clip.segments}")
    h = _require(clip, "hinge", "v1_hinge")
    head, layers = _header(a, tl)
    regions = [head]
    tau, win = a.trail_s, a.window_s
    clip, axis_in_view = _with_axis_in_view(clip, h["axis_s"])
    vmax = _speed_vmax(clip)
    y0, bh = _body(HEADER_H + GAP)
    lims = Lo._jaw_limits(clip)
    gain = _residual_gain(h["residual_mm"], max(b - a_ for a_, b in lims))
    # 3-D: measured mandible (pose, trails, helical axis), hinge ghost, fixed axis, residual.
    p3, l3 = jaw3d_panel(clip, "jaw:0", tau, vmax, title="Mandible and best-fit fixed hinge")
    p3.series += _hinge_axis_series(h, lims)
    for ly in l3:
        if isinstance(ly, Pose):
            ly.triad_mm = 0.0               # no frame triad: its green would read as residual
    regions.append(Region("jaw", (0.0, y0, 0.46, bh), _spec("jaw", [p3], 1), zoom=1.1))
    layers += l3 + _v1_layers(clip, h, "jaw:0", (0, 1, 2), gain)
    # Sagittal plane, same overlays; the fixed axis pierces it at one point.
    ps, lsag = plane_panel(clip, "planes:0", (0, 2), tau, vmax)
    s = h["axis_s"]
    ps.series.append(Series("scatter", x=np.array([s[0]]), y=np.array([s[2]]),
                            role=HINGE_ROLE, style={"ms": "d", "size": 1.4}))
    regions.append(Region("planes", (0.47, y0, 0.52, bh * 0.56), _spec("planes", [ps], 1)))
    layers += lsag + _v1_layers(clip, h, "planes:0", (0, 2), gain)
    # Residual over time and the inset: opening angle vs incisal displacement.
    t = h["t_us"]
    sel = _in_segments(t, clip)
    rmax = _nice(max(float(np.nanmax(h["residual_mm"][sel])) * 1.1, 0.5)) \
        if np.isfinite(h["residual_mm"][sel]).any() else 1.0
    pr = Panel("Hinge residual at the incisor", _time_axis(win, True),
               Axis("Residual", MM, lim=(0.0, rmax)), legend="none")
    pr.series.append(_now_line())
    th = np.asarray(clip.kin["theta_deg"], float)
    th_label = "Opening angle"
    if not np.isfinite(th[sel]).any():
        th, th_label = h["theta_h_deg"], "Hinge angle"
    xl = _lim([th[sel]], 1.0, 5.0)
    yl = _lim([h["disp_mm"][sel], h["disp_model_mm"][sel]], 1.0, 5.0)
    pi = Panel(f"{th_label} vs incisal displacement", Axis(th_label, "\\degree", lim=xl),
               Axis("Incisal displacement", MM, lim=yl), legend="none")
    tt, xy = Lo._breaks(t[sel], np.column_stack([th, h["disp_mm"], h["disp_model_mm"]])[sel])
    pi.series.append(Series("line", x=xy[:, 0], y=xy[:, 1], role="faint"))
    pi.series.append(Series("line", x=xy[:, 0], y=xy[:, 2], role=HINGE_ROLE,
                            style={"width": 0.6}))
    regions.append(Region("lower", (0.47, y0 + bh * 0.58, 0.52, bh * 0.42),
                          _spec("lower", [pr, pi], rows=1, cols=2)))
    layers.append(Scope("lower:0", t, h["residual_mm"], [RESID_ROLE], win, width=1.8))
    P = np.column_stack([th, np.zeros_like(th), h["disp_mm"]])
    Pm = np.column_stack([th, np.zeros_like(th), h["disp_model_mm"]])
    layers.append(Trail("lower:1", t, P, (0, 2), tau, "entity:incisor", width=2.4))
    layers.append(Marker("lower:1", t, Pm, (0, 2), HINGE_ROLE, size=0.9))
    rms, mx = float(h["rms_incisor_mm"]), float(h["max_incisor_mm"])
    arrow = "Residual" + (f" ×{gain:g}" if gain != 1 else "")
    cap, cap_layer = _caption_band(a, tl, [[
        "Measured mandible (grey) and the best-fit fixed hinge (dashed yellow): the ghost "
        "turns about one fixed axis by the angle that best matches the measured landmarks.",
        f"Green arrow: the residual at the incisor, what a fixed hinge misplaces "
        f"(hinge fit over the open–close cycles: RMS {rms:.1f} mm, max {mx:.1f} mm)"
        + (f"; drawn ×{gain:g}." if gain != 1 else "."),
        "Red dashed: the instantaneous helical axis. For a hinge it would stay on the fixed "
        "axis (yellow) and the arrow would stay at zero."
        + ("" if axis_in_view else f" (The fixed axis lies outside the view: it crosses "
                                   f"the midsagittal plane at x = {s[0]:.0f} mm, "
                                   f"z = {s[2]:.0f} mm.)"),
        f"Bottom right: {th_label.lower()} vs incisal displacement, measured (dot) and "
        "fixed-hinge prediction (yellow)."]])
    regions.append(cap)
    layers.append(cap_layer)
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [e for e in KIN_KEY if e[2] not in ("Maxilla", "Mandible")]
        + [("dash", "yellow", "Fixed hinge"), ("line", "green", arrow)],
        vmax, gap_in=0.2, bar_in=0.9)))
    return Composition("v1_hinge", regions, layers, tl, meta={"residual_gain": gain})


# -- shared: the release panel ---------------------------------------------------------------------
def release_panel(clip: Clip, key: str, window, xlabel: bool, title: str,
                  stream: bool = False, offset_us: float = 0.0, legend: str = "below"
                  ) -> tuple[Panel, list[Layer]] | None:
    """Closer envelope E (dB re rest), the causal release threshold and the decisions (X3);
    with ``stream`` also the stream replay's release decisions (ml_online)."""
    r = clip.extra.get("release")
    if r is None:
        return None
    t = r["t_us"]
    sel = _in_segments(t, clip)
    lo, hi = _lim([r["e_db"][sel], r["thr_db"][sel]], 1.0, 6.0)
    hi += 0.12 * (hi - lo)                                  # room for the decision markers
    p = Panel(title, _time_axis(window, xlabel), Axis("Envelope", "\\decibel", lim=(lo, hi)),
              legend=legend, sharex="t")
    p.series += [Series("line", x=np.array([]), y=np.array([]), role="group:closers",
                        label="Closer envelope"),
                 Series("line", x=np.array([]), y=np.array([]), role="threshold",
                        label=f"Threshold (baseline − {float(r['drop_db']):g} dB)")]
    p.series.append(_now_line())
    ys = hi - 0.05 * (hi - lo)
    layers: list[Layer] = [Scope(key, t, r["e_db"], ["group:closers"], window, width=1.6),
                           Scope(key, t, r["thr_db"], ["threshold"], window, width=1.5),
                           Events(key, r["dec_us"], ys, window, X3_ROLE, "v")]
    on = clip.extra.get("online")
    if stream and on is not None and len(on.get("release_us", ())):
        layers.append(Events(key, on["release_us"], hi - 0.11 * (hi - lo), window,
                             STREAM_ROLE, "|", size=1.1))
    if offset_us:
        layers = [Shifted(ly, offset_us) for ly in layers]
    return p, layers


# -- V2: intent cues in simulated real time --------------------------------------------------------
def _stream_rows(clip: Clip, theme: str) -> list[RibbonRow]:
    rows = []
    lut = categorical_rgba(LABEL_COLORS, theme)            # -1 (no decision): transparent
    if clip.labels is not None:
        rows.append(RibbonRow("Label", clip.labels["t_us"], clip.labels["label"], lut))
    on = clip.extra.get("online")
    if on is not None:
        for k, name in (("yhat", "Stream"), ("yhat_raw", "Stream raw")):
            if k in on:                                    # held from the emission time on
                rows.append(RibbonRow(name, on["t_us"], on[k], lut, "hold"))
    return rows


def _class_legend(rows: list[RibbonRow], clip: Clip) -> list[Series]:
    """Swatches of the intent classes that occur in the label and decision rows."""
    present: set[int] = set()
    for r in rows:
        v = np.asarray(r.values, float)[_in_segments(np.asarray(r.t_src, float), clip)]
        present |= {int(x) for x in np.unique(v[np.isfinite(v)]) if int(x) in LABEL_COLORS}
    return [Series("bar", x=np.array([]), y=np.array([]), role=Lo._CAT_ROLE[LABEL_COLORS[k]],
                   label=CLASS_NAMES[k].title()) for k in sorted(present)]


def _fmt_ms(v: float) -> str:
    return f"{v:.0f} ms" if np.isfinite(v) else "–"


def _counters_region(clip: Clip, box, tl: Timeline, layers: list[Layer]) -> Region:
    on = clip.extra.get("online")
    rel = clip.extra.get("release")
    rows: list[tuple[str, Any]] = []
    ev: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if on is not None and "ev_t_us" in on:
        k = on["ev_kind"]
        ev["hit"] = (on["ev_t_us"][k == 0], on["ev_latency_ms"][k == 0])
        ev["false"] = (on["ev_t_us"][k == 1], np.zeros(int((k == 1).sum())))
        ev["other"] = (on["ev_t_us"][k == 2], np.zeros(int((k == 2).sum())))
        rows += [("Onsets detected", lambda s: f"{s['hit_n']}"),
                 ("Latency, last", lambda s: _fmt_ms(s["hit_last"])),
                 ("Latency, median so far", lambda s: _fmt_ms(s["hit_median"])),
                 ("False activations", lambda s: f"{s['false_n']}"),
                 ("Other decision changes", lambda s: f"{s['other_n']}")]
    if rel is not None:
        ev["x3"] = (rel["dec_us"], np.zeros(len(rel["dec_us"])))
        rows.append(("Release decisions", lambda s: f"{s['x3_n']}"))
    if on is not None and len(on.get("release_us", ())):
        ev["stream"] = (on["release_us"], np.zeros(len(on["release_us"])))
        rows.append(("Release decisions (stream)", lambda s: f"{s['stream_n']}"))
    chips = []
    if on is not None and "yhat" in on:
        chips.append(("Stream decision now", on["t_us"], on["yhat"], "hold"))
    if clip.labels is not None:
        chips.append(("Label now", clip.labels["t_us"], clip.labels["label"], "hold"))
    n = len(chips) + len(rows)
    ys = list(np.linspace(0.84, 0.08, max(n, 2)))[:n] if n else []

    def draw(fig, theme, sc):
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.axhspan(0, 1, color=style.hexcolor("panel", theme), lw=0)
        fig.text(0.05, 0.94, "Stream decoder, pseudo-online replay", ha="left", va="center",
                 fontsize=style.FONT_PT["label"] * sc * 1.15, fontweight="bold",
                 color=style.hexcolor("ink", theme))
        for (lab, *_), y in zip(chips + rows, ys):
            fig.text(0.05, y, lab, ha="left", va="center", fontsize=style.FONT_PT["label"] * sc,
                     color=style.hexcolor("ink2", theme))
        fig._anim_axes = {}

    for (lab, t, v, method), y in zip(chips, ys):
        layers.append(Chip("counters", t, v, list(CLASS_NAMES), x=0.62, y=y, method=method,
                           size=1.0))
    items = [(0.95, y, "right", f, "label", "ink") for (_, f), y in zip(rows, ys[len(chips):])]
    if items:
        layers.append(Tally("counters", items, weight=tuple(str(i) for i in range(len(items))),
                            t_ref=lambda t, seg: tl.clip_seconds(t, seg), events=ev,
                            segments=tl.segments))
    return Region("counters", box, draw=draw)


def v2_intent(a: Lo.Animation, tl: Timeline) -> Composition:
    clip: Clip = a.clip
    if "release" not in clip.extra and "online" not in clip.extra:
        raise ValueError(f"{clip.label}: layout 'v2_intent' needs the release envelope (X3) "
                         "or the stream decoder (ml_online); neither could be loaded")
    th = a.theme
    win = (-abs(a.window_s[0]), 0.0)                       # only the past is shown
    head, layers = _header(a, tl)
    layers = [ly for ly in layers if not isinstance(ly, Chip)]   # the counters show it
    regions = [head, _banner(a)]
    y0, bh = _body(HEADER_H + 0.004 + BANNER_H + GAP)
    cand = []
    rel = release_panel(clip, "rel:0", win, False, "Closer envelope and release detector",
                        stream=True, legend="none")
    if rel:
        cand.append(("rel", 2.0, rel))
    op = emg_panel(clip, "op:0", OPENERS, win, clip.t0_us, False,
                   "Opener activation (% of max recorded)")
    if op:
        op[0].legend = "none"
        cand.append(("op", 1.4, op))
    ang = angle_panel(clip, "angle:0", win, clip.t0_us, False)
    if ang:
        ang[0].legend = "none"
        cand.append(("angle", 1.4, ang))
    rows = _stream_rows(clip, th)
    rib = ribbon_panel(rows, "ribbon:0", "Label and stream decisions", win, clip.t0_us, False,
                       th, clip)
    if rib:
        rib[0].series = [x for x in rib[0].series if x.kind != "bar"] + _class_legend(rows, clip)
        cand.append(("ribbon", 0.9 + 0.35 * len(rows), rib))
    if cand:
        cand[-1][2][0].x = _time_axis(win, True)
    boxes = Lo._flow([(n, w) for n, w, _ in cand], 0.0, y0, 0.655, bh, gap=0.008)
    for n, _, (pnl, ls) in cand:
        regions.append(Region(n, boxes[n], _spec(n, [pnl], 1), align="left"))
        layers += ls
    regions.append(_counters_region(clip, (0.665, y0, 0.33, bh), tl, layers))
    cap, cap_layer = _caption_band(a, tl, [[
        f"Offline replay of recorded data: {BANNER}. Every panel shows only the past "
        f"{abs(win[0]):g} s.",
        "Top: closer envelope (dB re rest) and the causal release threshold (dashed); "
        "▼ release decisions, | the stream replay's release decisions.",
        "Ribbon: the label, then the stream decoder's smoothed and raw decisions, shown from "
        "the moment they were emitted.",
        "Right: onsets detected, decision latency and false activations so far "
        "(pseudo-online evaluation)."]])
    regions += [cap, Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [("line", "group:closers", "Closer envelope"), ("dash", "threshold", "Threshold"),
         ("v", X3_ROLE, "Release"), ("|", STREAM_ROLE, "Release (stream)"),
         ("line", "entity:digastric_r", "Digastric R"),
         ("dash", "entity:digastric_l", "Digastric L"), ("line", "primary", "Optical angle"),
         ("dash", Lo.IMU_ROLE, "IMU angle")], None, gap_in=0.22))]
    layers.append(cap_layer)
    return Composition("v2_intent", regions, layers, tl)


# -- V3: open-close vs chewing ---------------------------------------------------------------------
def _v3_segments(clip: Clip) -> tuple[tuple[str, float, float], tuple[str, float, float], float]:
    if len(clip.segments) < 2:
        raise ValueError("layout 'v3_release' plays two phases side by side (default: "
                         "open_close and chewing); give two --phase")
    (pa, a0, a1), (pb, b0, b1) = clip.segments[:2]
    d = min(a1 - a0, b1 - b0)
    return (pa, a0, a0 + d), (pb, b0, b0 + d), b0 - a0


def v3_timeline(a: Lo.Animation) -> Timeline:
    """One segment: the first phase; the second plays in step (offset ``b0 - a0``)."""
    (pa, a0, a1), (pb, _, _), _ = _v3_segments(a.clip)
    lab = f"{PHASE_LABEL.get(pa, pa)} | {PHASE_LABEL.get(pb, pb)}"
    return Timeline(((a0, a1),), a.fps, a.speed, (lab,))


def v3_release(a: Lo.Animation, tl: Timeline) -> Composition:
    clip: Clip = a.clip
    _require(clip, "release", "v3_release")
    (pa, a0, _), (pb, b0, _), off = _v3_segments(clip)
    win = a.window_s
    head, layers = _header(a, tl)
    layers = [ly for ly in layers if not isinstance(ly, Chip)]
    regions = [head]
    y0, bh = _body(HEADER_H + GAP)
    tally_h = 0.05
    for side, (ph, t0, shift) in enumerate(((pa, a0, 0.0), (pb, b0, off))):
        x0 = 0.0 if side == 0 else 0.505
        panels = []
        res = release_panel(clip, f"col{side}:{len(panels)}", win, False,
                            f"{PHASE_LABEL.get(ph, ph)}: closer envelope", offset_us=shift)
        if res:
            panels.append(res)
        ang = angle_panel(clip, f"col{side}:{len(panels)}", win, clip.t0_us, True)
        if ang:
            pnl, ls = ang
            pnl.title = f"{PHASE_LABEL.get(ph, ph)}: opening angle"
            panels.append((pnl, [Shifted(ly, shift) if shift else ly for ly in ls]))
        regions.append(Region(f"col{side}", (x0, y0, 0.495, bh - tally_h - 0.006),
                              _spec(f"col{side}", [p for p, _ in panels], rows=len(panels)),
                              align="cols"))
        for _, ls in panels:
            layers += ls
        r = clip.extra["release"]
        seg = ((t0, t0 + tl.master_durations_s[0] * 1e6),)

        def draw(fig, theme, sc):
            fig._anim_axes = {}

        regions.append(Region(f"tally{side}", (x0, y0 + bh - tally_h, 0.495, tally_h),
                              draw=draw))
        tal = Tally(f"tally{side}", [(0.06, 0.5, "left", lambda s: (
            f"{s['rel_n']} release decision{'' if s['rel_n'] == 1 else 's'} in "
            f"{max(s['t_s'], 0):.1f} s"
            + (f"  ({60 * s['rel_n'] / s['t_s']:.0f} per min)" if s["t_s"] > 5 else "")),
            "label", "ink")], t_ref=lambda t, seg_, t0=t0: (np.asarray(t, float) - t0) * 1e-6,
            events={"rel": (r["dec_us"], np.zeros(len(r["dec_us"])))}, segments=seg)
        layers.append(Shifted(tal, shift) if shift else tal)
    a_lab, b_lab = PHASE_LABEL.get(pa, pa).lower(), PHASE_LABEL.get(pb, pb).lower()
    cap, cap_layer = _caption_band(a, tl, [[
        f"Release behaviour of the closers: {a_lab} (left) and {b_lab} (right), played in "
        "step from each phase's start, on the same axes.",
        "Closer envelope (dB re rest) and the causal release threshold, a lagged baseline "
        f"minus {float(clip.extra['release']['drop_db']):g} dB (dashed); ▼ release decisions.",
        "Below: the opening angle; compare how the decisions line up with the openings in "
        "each task."]])
    regions += [cap, Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [("line", "group:closers", "Closer envelope"), ("dash", "threshold", "Threshold"),
         ("v", X3_ROLE, "Release decision"), ("line", "primary", "Optical angle"),
         ("dash", Lo.IMU_ROLE, "IMU angle")], None))]
    layers.append(cap_layer)
    return Composition("v3_release", regions, layers, tl, meta={"offset_us": off})


# -- previews (Plotly scrubber, LaTeX filmstrip) ---------------------------------------------------
def _preview_from(names: list[str], film: int = 0):
    """Compact preview: the named regions' panels in one spec, their layers re-addressed."""

    def build(a: Lo.Animation):
        comp = a.compose()
        panels, layers = [], []
        for name in names:
            r = next((r for r in comp.regions if r.name == name and r.spec is not None), None)
            if r is None:
                continue
            for k, p in enumerate(r.spec.panels):
                j = len(panels)
                panels.append(copy.deepcopy(p))
                for ly in comp.layers:
                    if ly.panel == f"{name}:{k}":
                        ly.panel = f"preview:{j}"
                        layers.append(ly)
        if not panels:
            raise ValueError(f"{a.clip.label}: nothing to preview for layout {a.layout!r}")
        cols = 2 if len(panels) > 1 else 1
        rows = int(np.ceil(len(panels) / cols))
        return _spec("preview", panels, rows=rows, cols=cols), layers, min(film,
                                                                           len(panels) - 1)

    return build


def _register() -> None:
    style.ROLES.setdefault(HINGE_ROLE, ("yellow", "dashed", "d"))
    style.ROLES.setdefault(X3_ROLE, ("vermillion", "solid", "v"))
    style.ROLES.setdefault(STREAM_ROLE, ("purple", "solid", "|"))
    Lo.LAYOUTS.update(v1_hinge=v1_hinge, v2_intent=v2_intent, v3_release=v3_release)
    Lo.EXTRAS.update(v1_hinge=("hinge",), v2_intent=("release", "online"),
                     v3_release=("release",))
    Lo.TIMELINES["v3_release"] = v3_timeline
    Lo.PREVIEWS.update(v1_hinge=_preview_from(["jaw", "planes", "lower"], film=1),
                       v2_intent=_preview_from(["rel", "op", "angle", "ribbon"]),
                       v3_release=_preview_from(["col0", "col1"]))
    Lo.DEFAULT_PHASES.update(v1_hinge=("open_close",), v2_intent=("open_close",),
                             v3_release=("open_close", "chewing"))
    Lo.DEFAULT_SPEED["v2_intent"] = 1.0
    Lo.BANNERS["v2_intent"] = BANNER
    Lo.STILLS.update({"v1_hinge", "v2_intent", "v3_release"})


_register()

__all__ = ["BANNER", "release_panel", "spread_cues", "v1_hinge", "v2_intent", "v3_release",
           "v3_timeline"]
