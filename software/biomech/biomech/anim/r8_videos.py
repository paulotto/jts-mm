# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.7: three more video layouts, registered in :data:`layouts.LAYOUTS` on import.

``workspace3d``      a slow 3-D rotation of the cohort's incisal workspace: every incisal
                     position of the five tasks as displacement from the rest position
                     (thinned, colour and marker per participant), the convex hull of all
                     positions (a cage of sections) and the design workspace box of the stored
                     P95 requirements (backward by ``posterior_excursion_mm`` when the
                     requirements table holds it, else the box starts at the rest position).
                     All participants; no ``--pid`` (a cohort layout like ``cohort``).
``emg_spectrogram``  a scrolling EMG view of one participant and task: the band-passed EMG of
                     the six channels, the closer and opener envelopes, the opening angle, a
                     live short-time spectrum strip per muscle group and the group spectra at
                     the current instant.
``onset_rules``      an explainer of the C2 onset rules on real open-close cycles of one
                     participant: the digastric band-passed EMG and its TKEO detector signal
                     against the rest threshold, the mechanical onset, and the onsets of the
                     first-onset rule and of the driving-burst rule appearing in turn, cycle
                     after cycle, with a subtitle line per cycle from the stored onsets.

Every number on screen and in the subtitles comes from the stored results, the configuration
the stages ran with, or a named display constant of this module or of the views it reuses
(:mod:`biomech.viz.views.kin_r8`, :mod:`biomech.viz.views.emg_r8`); nothing is hand-typed.
The derived data are loaded once (``source.EXTRAS`` / ``layouts.COHORT_LOADERS``) and travel
with the clip files. Each layout gets burned-in subtitles (and the ``.vtt``) and a PNG
filmstrip (``layouts.STILLS``) at ``STILL_WIDTH`` px per frame, usable as a paper figure when
rendered with ``--theme light``.

Presentation code (``pipeline.PRESENTATION``): it reads stored results and the recorded
session, recomputes only display signals with the stages' own routines, and never changes a
result.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from biomech.anim import layouts as Lo
from biomech.anim import source as S
from biomech.anim import supplementary as Sup
from biomech.anim.layers import (Profile, ScrollImage, SegmentEvents, SegmentTrace,
                                 Spin, Text, magnitude_ramp, ramp_rgb)
from biomech.anim.layouts import (CLOSERS, HEADER_H, KEY_H, MM, OPENERS,
                                  Composition, Region, _header, _key, _spec, _time_axis)
from biomech.anim.source import MUSCLE_LABEL, MUSCLES, Clip, CohortClip
from biomech.anim.timeline import Timeline
from biomech.viz import style
from biomech.viz.spec import Axis, Colorbar, Panel, Series

# -- display constants (quoted on screen from these names) -----------------------------------------
ROT_S = 24.0                 # workspace3d: one full turn lasts this long at speed 1
ROT_ELEV = 18.0              # its elevation (deg) and the gentle wobble about it
ROT_ELEV_AMP = 8.0
WS_POINTS = 500              # incisal positions shown per participant (evenly thinned)
EMG_DISPLAY_HZ = 1000.0      # emg_spectrogram: min/max pairs per bin of 1 / this (s)
EMG_ROW_SCALE = 99.5         # each EMG row scaled to this percentile of |EMG| in the clip
ONSET_CYCLES = 8             # onset_rules: cycles played (in order, within --t0/--t1)
ONSET_MARGIN_S = 0.15        # shown before and after the onset search window
STILL_WIDTH = 1280           # px per filmstrip frame (paper figure: 3 x 2 frames)
MARKERS = "osd^v"            # participant k: colour CATEGORICAL[k % 6], marker by k // 6
RULE_GLYPH = SimpleNamespace(first=("v", "secondary"), drive=("d", "accent"))  # emg_r8


def _fin(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _ms(v: float) -> str:
    """'120 ms before' / '35 ms after' (lead > 0: EMG before the motion)."""
    if not _fin(v):
        return "no onset"
    return f"{abs(v):.0f} ms {'before' if v >= 0 else 'after'}"


# == workspace3d =======================================================================================
def _participant_glyph(k: int) -> tuple[str, str]:
    """(role, marker) of the k-th participant: categorical colour, marker by colour cycle."""
    return f"cat:{k % len(style.CATEGORICAL)}", MARKERS[(k // len(style.CATEGORICAL))
                                                       % len(MARKERS)]


def load_workspace_clip(ctx, task: str = "open_close") -> CohortClip:
    """The cohort's incisal workspace (``task`` is ignored: every task enters): per included
    participant the incisal positions of all tasks as displacement from rest
    (:func:`kin_r8.incisor_cloud`), thinned to ``WS_POINTS``; the convex hull of all positions
    (:func:`kin_r8.hull_cage`, its volume); the design box (:func:`kin_r8.design_box`) and the
    share of each participant's positions outside it."""
    from biomech.viz.views import kin_r8 as K

    pids = list(ctx.pids())
    rq = K._requirements(ctx)
    box = K.design_box(rq)
    labels, pts, n_pts, outside, allp = [], [], [], [], []
    for pid in sorted(pids, key=lambda p: int(ctx.pseudonym(p))):
        P, _ = K.incisor_cloud(K._arr(ctx, "kinematics", pid))
        if len(P) < 5:
            continue
        labels.append(f"P{ctx.pseudonym(pid)}")
        pts.append(P[K._thin(len(P), WS_POINTS)].astype(np.float64))
        n_pts.append(len(P))
        outside.append(K._outside(P, box))
        allp.append(P)
    if not allp:
        raise ValueError("no incisal kinematics in this store (stage kinematics)")
    A = np.vstack(allp)
    cage = K.hull_cage(A)
    _, hull = K._hull(A)
    nan3 = np.full(3, np.nan)
    ws = {"points": pts, "n_points": np.asarray(n_pts, float),
          "cage": cage if cage is not None else np.empty((0, 3)),
          "hull_volume_mm3": np.asarray(float(hull.volume) if hull is not None else np.nan),
          "box_lo": box[0] if box is not None else nan3,
          "box_hi": box[1] if box is not None else nan3,
          "posterior_stored": np.asarray(_fin(K._req(rq, "posterior_excursion_mm"))),
          "outside_pct": np.asarray(outside, float),
          "p_design": np.asarray(K._p_design()), "hull_rings": np.asarray(K.HULL_RINGS)}
    return CohortClip(task="workspace", labels=labels, trials=[],
                      means=np.zeros((0, 101, 3)), speed=np.zeros((0, 101)),
                      meta={"n_trials": 0, "snapshot": S._snapshot(ctx),
                            "variant": S._variant(ctx)},
                      extra={"workspace": ws})


def _ws(a: Lo.Animation) -> dict[str, Any]:
    ws = a.clip.extra.get("workspace") if isinstance(a.clip, CohortClip) else None
    if ws is None:
        raise ValueError("layout 'workspace3d' needs the cohort workspace data")
    return ws


def workspace_timeline(a: Lo.Animation) -> Timeline:
    """One segment: one full turn of ``ROT_S`` s of (pseudo) master time."""
    return Timeline(((0.0, ROT_S * 1e6),), a.fps, a.speed, ("workspace",))


def _cube(points: list[np.ndarray], pad: float = 0.04) -> list[tuple[float, float]]:
    """Equal limits on all three axes (mm for mm at any view angle of the rotation)."""
    P = np.vstack([np.asarray(p, float).reshape(-1, 3) for p in points if np.size(p)])
    P = P[np.isfinite(P).all(axis=1)]
    lo, hi = P.min(axis=0), P.max(axis=0)
    c, half = 0.5 * (lo + hi), 0.5 * float(np.max(hi - lo)) * (1 + 2 * pad)
    return [(float(v - half), float(v + half)) for v in c]


def _ws_box_text(ws: dict[str, Any]) -> list[str]:
    lo, hi = np.asarray(ws["box_lo"], float), np.asarray(ws["box_hi"], float)
    p = float(ws["p_design"])
    if not np.isfinite(lo).all():
        return ["The requirements are not in this result set; no design box is drawn."]
    back = (f"{-lo[0]:.1f} mm backward (posterior incisal excursion)"
            if bool(ws["posterior_stored"]) else
            "no backward bound stored: the box starts at the rest position")
    return [f"Design box (P{p:.0f} requirements): {hi[0]:.1f} mm forward (protrusion), "
            f"{hi[1]:.1f} mm to each side (laterotrusion), {-lo[2]:.1f} mm down (opening); "
            f"{back}."]


def workspace3d(a: Lo.Animation, tl: Timeline) -> Composition:
    ws = _ws(a)
    c: CohortClip = a.clip
    head, layers = _header(a, tl)
    spin = Spin("jaw:0", ROT_S * 1e6, azim0=-58.0, elev=ROT_ELEV, elev_amp=ROT_ELEV_AMP)
    text = next(ly for ly in layers if isinstance(ly, Text))
    turn_s = ROT_S / a.speed
    text.items = [text.items[0],
                  (text.items[1][0], 0.58, "left", lambda s: "Incisal workspace", "title",
                   "ink2"),
                  (0.80, 0.58, "right", lambda s: f"azimuth {s['t_s']:.0f}°", "title", "ink"),
                  (0.988, 0.58, "right", lambda s: f"one turn in {turn_s:g} s", "label",
                   "ink2")]
    text.t_ref = lambda t, seg: spin.angles(t)[0] % 360.0
    regions = [head]
    y0, bh = Sup._body(HEADER_H + Sup.GAP)
    lo, hi = np.asarray(ws["box_lo"], float), np.asarray(ws["box_hi"], float)
    from biomech.viz.views import kin_r8 as K

    corners = [np.vstack([lo, hi])] if np.isfinite(lo).all() else []
    lims = _cube(list(ws["points"]) + corners + [np.zeros((1, 3))])
    p = Panel("", Axis("Anterior", MM, lim=lims[0]), Axis("Left", MM, lim=lims[1]),
              projection="3d", z=Axis("Superior", MM, lim=lims[2]), view=(-58.0, ROT_ELEV),
              legend="none")
    for k, P in enumerate(ws["points"]):
        role, mk = _participant_glyph(k)
        P = np.asarray(P, float)
        p.series.append(Series("scatter3d", x=P[:, 0], y=P[:, 1], z=P[:, 2], role=role,
                               style={"ms": mk, "size": 0.42}))
    cage = np.asarray(ws["cage"], float)
    if len(cage):
        p.series.append(Series("line3d", x=cage[:, 0], y=cage[:, 1], z=cage[:, 2],
                               role="accent", style={"width": 0.9}))
    if corners:
        E = K.box_edges(lo, hi)
        p.series.append(Series("line3d", x=E[:, 0], y=E[:, 1], z=E[:, 2], role="threshold",
                               style={"width": 1.3}))
    p.series.append(Series("scatter3d", x=np.zeros(1), y=np.zeros(1), z=np.zeros(1),
                           role="accent", style={"ms": "d", "size": 1.6}))
    regions.append(Region("jaw", (0.0, y0, 0.70, bh), _spec("jaw", [p], 1), zoom=1.3))
    layers.append(spin)
    regions.append(_ws_board(c, ws, (0.71, y0, 0.285, bh)))
    p95 = float(ws["p_design"])
    n = len(c.labels)
    vol = float(ws["hull_volume_mm3"]) / 1e3
    out = np.asarray(ws["outside_pct"], float)
    med = float(np.nanmedian(out)) if np.isfinite(out).any() else np.nan
    cues = [f"Incisal positions of all five tasks of {n} participants, as displacement from the "
            f"rest position (diamond), thinned to {WS_POINTS} per participant; colour and "
            "marker: participant.",
            f"{'White' if a.theme == 'dark' else 'Black'} cage: the convex hull of all "
            "positions of the cohort (sagittal and frontal "
            f"sections, {int(ws['hull_rings'])} horizontal sections)"
            + (f", {vol:.0f} cm³." if np.isfinite(vol) else "."),
            "Dashed: d" + _ws_box_text(ws)[0][1:]]
    if np.isfinite(med):
        cues.append(f"A participant's positions lie outside the P{p95:.0f} box in "
                    f"{med:.0f} % of the samples (median over participants); the hull shows "
                    "where they go.")
    cap, cap_layer = Sup._caption_band(a, tl, [cues])
    regions.append(cap)
    layers.append(cap_layer)
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [("line", "accent", "Convex hull (cohort)"),
         ("dash", "threshold", f"Design box (P{p95:.0f})"),
         ("d", "accent", "Rest position")], None)))
    return Composition("workspace3d", regions, layers, tl, meta={"lims": lims})


def _ws_board(c: CohortClip, ws: dict[str, Any], box) -> Region:
    """Right column: the participants (glyph, pseudonym, share outside the box) and the
    stored design values."""
    out = np.asarray(ws["outside_pct"], float)

    def draw(fig, theme, sc):
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhspan(0, 1, color=style.hexcolor("panel", theme), lw=0)
        ink, ink2 = style.hexcolor("ink", theme), style.hexcolor("ink2", theme)
        fs = style.FONT_PT["label"] * sc
        W, H = fig.get_size_inches()
        fig.text(0.05, 0.965, "Participants", ha="left", va="center", fontweight="bold",
                 fontsize=fs * 1.1, color=ink)
        fig.text(0.95, 0.965, "outside the box", ha="right", va="center", fontsize=fs * 0.9,
                 color=ink2)
        n = len(c.labels)
        rows = max(n, 1)
        top, bot = 0.92, 0.36
        step = min((top - bot) / rows, 0.06)
        for k, lab in enumerate(c.labels):
            y = top - (k + 0.5) * step
            role, mk = _participant_glyph(k)
            col = style.hexcolor(style.role(role)[0], theme)
            ax.plot([0.09], [y], ls="none", marker=mk, ms=style.MARKER_PT * sc * 3.0,
                    color=col, markeredgecolor=style.hexcolor("surface", theme),
                    markeredgewidth=0.4 * sc)
            fig.text(0.17, y, lab, ha="left", va="center", fontsize=fs, color=ink)
            v = out[k] if k < len(out) else np.nan
            fig.text(0.95, y, f"{v:.0f} %" if np.isfinite(v) else "–", ha="right",
                     va="center", fontsize=fs, color=ink2)
        import textwrap

        wrap = max(20, int(W * 72 / (fs * 0.55)))
        lines = textwrap.wrap(_ws_box_text(ws)[0], wrap)
        fig.text(0.05, 0.30, "\n".join(lines), ha="left", va="top", fontsize=fs * 0.95,
                 color=ink2, linespacing=1.35)
        fig._anim_axes = {}

    return Region("board", box, draw=draw)


# == shared: time ticks and stacked columns =============================================================
def _taxis(window: tuple[float, float], label: bool) -> Axis:
    """A scrolling time axis with whole-second ticks (half seconds for short windows); tick
    labels only on the column's bottom panel (``label``)."""
    lo, hi = float(window[0]), float(window[1])
    step = 1.0 if hi - lo >= 3 else 0.5
    ticks = list(np.arange(np.ceil(lo / step) * step, hi + 1e-9, step))
    ax = _time_axis(window, label)
    ax.ticks = [float(v) for v in ticks]
    ax.ticklabels = [f"{v:g}" for v in ticks] if label else [""] * len(ticks)
    return ax


def _column(name: str, items: list[tuple[Panel, float, list]], box, layers: list,
            post=None) -> Region:
    """One region holding the stacked panels (relative heights), so their axes share the
    column's edges; each panel's layers are re-addressed to ``name:k``."""
    panels = []
    for k, (p, h, ls) in enumerate(items):
        p.height = h
        panels.append(p)
        for ly in ls:
            ly.panel = f"{name}:{k}"
        layers += ls
    return Region(name, box, _spec(name, panels, rows=len(panels)), post=post)


# == emg_spectrogram ====================================================================================
def _minmax(t: np.ndarray, x: np.ndarray, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Display decimation keeping the peaks: per bin of ``step`` samples the minimum and the
    maximum (at the bin's first and middle sample time); a bin with a masked (NaN) sample is
    NaN."""
    n = (len(t) // step) * step
    if n == 0:
        return t[:0].astype(float), x[:0]
    X = x[:n].reshape(n // step, step, -1)
    bad = ~np.isfinite(X).all(axis=1)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanmin(np.where(np.isfinite(X), X, np.inf), axis=1), \
            np.nanmax(np.where(np.isfinite(X), X, -np.inf), axis=1)
    lo[bad], hi[bad] = np.nan, np.nan
    tb = t[:n].reshape(n // step, step)
    tt = np.column_stack([tb[:, 0], tb[:, step // 2]]).ravel().astype(np.float64)
    yy = np.stack([lo, hi], axis=1).reshape(-1, X.shape[2])
    return tt, yy


def _group_spectrogram(x: np.ndarray, mask: np.ndarray, fs: float, cols: list[int],
                       band: tuple[float, float], nper: int, nover: int
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, column centre s, PSD dB (F, M)) of the group: the channels' short-time PSD
    averaged over the channels whose window holds no masked sample (as emg_r8 ``chain``)."""
    from scipy.signal import spectrogram

    acc, cnt, f_keep, tt = None, None, None, None
    for c in cols:
        f, tt, Sx = spectrogram(np.nan_to_num(x[:, c]), fs, nperseg=nper, noverlap=nover,
                                scaling="density")
        keep = (f >= band[0]) & (f <= band[1])
        f_keep, Sx = f[keep], Sx[keep]
        cm = np.r_[0, np.cumsum(mask[:, c].astype(int))]
        ctr = np.round(tt * fs).astype(int)
        a = np.clip(ctr - nper // 2, 0, len(x))
        b = np.clip(ctr + nper // 2, 0, len(x))
        ok = (cm[b] - cm[a]) == 0
        Sx = np.where(ok[None, :], Sx, np.nan)
        if acc is None:
            acc, cnt = np.zeros_like(Sx), np.zeros(Sx.shape)
        fin = np.isfinite(Sx)
        acc += np.where(fin, Sx, 0.0)
        cnt += fin
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        Z = 10 * np.log10(np.where(P > 0, P, np.nan))
    return f_keep, tt, Z


def _emg_spectro(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """Band-passed (mains-removed, masked = NaN) EMG of every channel near the clip's
    segments, decimated for display (min/max per 1 / ``EMG_DISPLAY_HZ`` s) and scaled per
    channel; the group short-time spectra (window ``emg_r8.STFT_S``, overlap
    ``emg_r8.STFT_OVERLAP``, band of the signal chain)."""
    from biomech.viz.views import emg_r8 as E

    prep = ctx.emg(pid)
    names = list(prep.names)
    order = [m for m in MUSCLES if m in names]
    band = E._band(ctx)
    parts, specs = [], {"closers": [], "openers": []}
    for ph in dict.fromkeys(ph for ph, _, _ in clip.segments):
        pe = prep.phase(ph)
        if pe is None:
            continue
        t = np.asarray(pe.t_us, np.int64)
        sel = np.flatnonzero(S._near_segments(t, clip.segments))
        if not sel.size:
            continue
        sl = slice(int(sel[0]), int(sel[-1]) + 1)
        fs = float(pe.fs)
        X = np.asarray(pe.cleaned, float)[sl][:, [names.index(m) for m in order]]
        M = np.asarray(pe.mask, bool)[sl][:, [names.index(m) for m in order]]
        X = np.where(M, np.nan, X)
        step = max(int(round(fs / EMG_DISPLAY_HZ)), 1)
        td, yd = _minmax(t[sl], X, step)
        parts.append({"t_us": td, "x": yd.astype(np.float32)})
        nper = max(int(round(E.STFT_S * fs)), 16)
        nover = int(round(nper * E.STFT_OVERLAP))
        for g, grp in (("closers", CLOSERS), ("openers", OPENERS)):
            cols = [order.index(m) for m in grp if m in order]
            if not cols:
                continue
            f, tt, Z = _group_spectrogram(X, M, fs, cols, band, nper, nover)
            specs[g].append((t[sl][0] + tt * 1e6, f, Z))
    out = S._concat(parts)
    if out is None:
        return None
    seg = Lo._in_segments(out["t_us"], clip)
    amp = np.array([np.nanpercentile(np.abs(out["x"][seg, c]), EMG_ROW_SCALE)
                    if np.isfinite(out["x"][seg, c]).any() else 1.0
                    for c in range(out["x"].shape[1])])
    out["amp_uv"] = np.where(amp > 0, amp, 1.0)
    out["muscles"] = np.array([MUSCLES.index(m) for m in order], np.int16)
    for g, lst in specs.items():
        if not lst:
            continue
        tc = np.concatenate([a for a, _, _ in lst]).astype(np.float64)
        Z = np.concatenate([z for _, _, z in lst], axis=1).astype(np.float32)
        o = np.argsort(tc, kind="stable")
        out[f"{g}_t_us"], out[f"{g}_db"], out["freq_hz"] = tc[o], Z[:, o], lst[0][1]
    out.update(band_hz=np.asarray(band, float), env_ms=np.asarray(E._env_ms(ctx)),
               stft_s=np.asarray(E.STFT_S), stft_overlap=np.asarray(E.STFT_OVERLAP),
               display_hz=np.asarray(EMG_DISPLAY_HZ), row_pct=np.asarray(EMG_ROW_SCALE))
    return out


def _db_lim(d: dict[str, Any], clip: Clip, groups: list[str]) -> tuple[float, float]:
    vals = []
    for g in groups:
        sel = Lo._in_segments(d[f"{g}_t_us"], clip)
        z = np.asarray(d[f"{g}_db"], float)[:, sel]
        vals.append(z[np.isfinite(z)])
    v = np.concatenate(vals) if vals else np.empty(0)
    if not v.size:
        return (0.0, 1.0)
    return float(np.floor(np.percentile(v, 5))), float(np.ceil(np.percentile(v, 99.5)))


def emg_spectrogram(a: Lo.Animation, tl: Timeline) -> Composition:
    clip: Clip = a.clip
    d = Sup._require(clip, "emg_spectro", "emg_spectrogram")
    th, win = a.theme, a.window_s
    head, layers = _header(a, tl)
    regions = [head]
    y0, bh = Sup._body(HEADER_H + Sup.GAP)
    order = [MUSCLES[int(k)] for k in d["muscles"]]
    R = len(order)
    amp = np.asarray(d["amp_uv"], float)
    # (a) six channels, one row each (first on top), scaled to their own amplitude
    Y = np.asarray(d["x"], float) / (2.2 * amp[None, :]) + (R - 1 - np.arange(R))[None, :]
    names = [f"\u2007\u2007{MUSCLE_LABEL.get(m, m)} ±{amp[k]:.0f} µV" for k, m in enumerate(order)]
    pa = Panel("Band-passed EMG", _taxis(win, False), Lo._rows_axis(names), legend="none",
               sharex="t")
    pa.series.append(Lo._now_line())
    left = [(pa, 2.3, [Lo.Scope("", d["t_us"], Y, [f"entity:{m}" for m in order], win,
                                width=0.8)])]
    for grp, title in ((CLOSERS, "Closer envelopes (% of max recorded)"),
                       (OPENERS, "Opener envelopes (% of max recorded)")):
        r = Lo.emg_panel(clip, "", grp, win, clip.t0_us, False, title)
        if r:
            r[0].legend = "none"
            r[0].x = _taxis(win, False)
            left.append((r[0], 0.95, r[1]))
    ang = Lo.angle_panel(clip, "", win, clip.t0_us, False)
    if ang:
        ang[0].legend = "none"
        ang[0].x = _taxis(win, False)
        left.append((ang[0], 0.8, ang[1]))
    left[-1][0].x = _taxis(win, True)
    regions.append(_column("left", left, (0.03, y0, 0.57, bh), layers))
    # right: spectrum strips per group and the spectra now
    groups = [g for g in ("closers", "openers") if f"{g}_db" in d]
    if groups:
        zl = _db_lim(d, clip, groups)
        ramp = magnitude_ramp(th)
        f = np.asarray(d["freq_hz"], float)
        df = float(f[1] - f[0]) if f.size > 1 else 1.0
        ylim = (float(f[0] - df / 2), float(f[-1] + df / 2))
        right = []
        for g in groups:
            title = f"{'Closers' if g == 'closers' else 'Openers'}: short-time spectrum"
            ps = Panel(title, _taxis(win, False), Axis("Frequency", "\\hertz", lim=ylim),
                       legend="none", sharex="t",
                       colorbar=Colorbar("PSD", "\\decibel", lim=zl))
            ps.series.append(Lo._now_line())
            Z = np.asarray(d[f"{g}_db"], float)
            rgbz = ramp_rgb(((Z - zl[0]) / (zl[1] - zl[0])).ravel(), ramp).reshape(Z.shape + (3,))
            right.append((ps, 1.0, [ScrollImage("", d[f"{g}_t_us"], rgbz, ylim, win)]))
        right[-1][0].x = _taxis(win, True)
        top_h = bh * 0.60
        regions.append(_column("strips", right, (0.62, y0, 0.375, top_h), layers,
                               post=Lo._recolor_colorbar(ramp)))
        pn = Panel("Spectrum now", Axis("Frequency", "\\hertz", lim=ylim),
                   Axis("PSD", "\\decibel", lim=zl), legend="below")
        tc = np.asarray(d[f"{groups[0]}_t_us"], float)
        Yn = []
        for g in groups:
            Z = np.asarray(d[f"{g}_db"], float)
            tg = np.asarray(d[f"{g}_t_us"], float)
            Yn.append(Z.T if np.array_equal(tg, tc) else
                      np.array([np.interp(tc, tg, z, left=np.nan, right=np.nan) for z in Z]).T)
            pn.series.append(Series("line", x=np.array([]), y=np.array([]),
                                    role=f"group:{g}", label=g.title()))
        hold = 2.0 * float(np.median(np.diff(tc))) * 1e-6 if tc.size > 1 else 0.25
        layers.append(Profile("now:0", tc, f, np.stack(Yn, axis=1),
                              [f"group:{g}" for g in groups], hold_s=hold))
        regions.append(Region("now", (0.62, y0 + top_h + 0.01, 0.375, bh - top_h - 0.01),
                              _spec("now", [pn], 1)))
    lo_hz, hi_hz = (float(v) for v in d["band_hz"])
    cues = [f"Band-passed EMG ({lo_hz:g}–{hi_hz:g} Hz, mains removed) of the six channels, each "
            f"row scaled to its own {float(d['row_pct']):g}th percentile of |EMG| (row label); "
            "gaps: samples excluded by the analysis mask.",
            f"Envelopes ({float(d['env_ms']):g} ms RMS, % of max recorded) of the closers and "
            "openers, and the opening angle (optical solid, IMU dashed).",
            f"Right: short-time spectrum of each muscle group ({1e3 * float(d['stft_s']):g} ms "
            f"windows, {100 * float(d['stft_overlap']):g} % overlap; the channels' spectra "
            "averaged in the group, dB), grey where a window holds masked samples.",
            "Bottom right: each group's short-time spectrum at the current instant (the strip "
            "column under the now line)."]
    cap, cap_layer = Sup._caption_band(a, tl, [cues])
    regions.append(cap)
    layers.append(cap_layer)
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [("line", f"entity:{m}", MUSCLE_LABEL.get(m, m)) for m in order]
        + [("line", "primary", "Optical angle"), ("dash", Lo.IMU_ROLE, "IMU angle")],
        None, gap_in=0.2)))
    return Composition("emg_spectrogram", regions, layers, tl)


# == onset_rules ==========================================================================================
def _onset_explainer(ctx, pid: str, clip: Clip) -> dict[str, Any] | None:
    """The stored open-close opening onsets of one digastric (the one with more usable
    channel-cycles under either rule; ties: right), up to ``ONSET_CYCLES`` cycles whose onset
    search window lies inside the clip; the band-passed EMG and the detector signal re the rest
    threshold around them, recomputed by the routines of emg_r8 ``chain`` (the timing stage's
    detector and threshold)."""
    from biomech.viz.views import emg_r8 as E

    task = "open_close"
    seg = next(((a, b) for ph, a, b in clip.segments if ph == task), None)
    if seg is None:
        raise ValueError("layout 'onset_rules' plays the open_close phase")
    rows = E._onset_rows(ctx, pid, task)
    if rows.empty:
        raise ValueError("no stored timing onsets (stage timing)")
    rows = rows[(rows.direction == "opening") & rows.muscle.isin(OPENERS)]
    use = {m: int(rows[rows.muscle == m][[c for c in ("usable_first", "usable_drive")
                                          if c in rows]].any(axis=1).sum()) for m in OPENERS}
    muscle = max(OPENERS, key=lambda m: (use.get(m, 0), m == "digastric_r"))
    pre = float(E._tp(ctx, "emg_pre_s", default=0.6))
    post = float(E._tp(ctx, "emg_post_s", default=0.2))
    r = rows[(rows.muscle == muscle) & (rows.t_mech_us - pre * 1e6 >= seg[0])
             & (rows.t_mech_us + post * 1e6 <= seg[1])]
    on = [c for c in ("t_on_first", "t_on_drive") if c in r]
    r = r[r[on].notna().any(axis=1)].sort_values("t_mech_us").reset_index(drop=True)
    # ONSET_CYCLES consecutive cycles: the run with the most cycles usable under either rule
    # (the first such run), as the signal-chain view picks its excerpt (emg_r8 ``chain``).
    ok = r[[c for c in ("usable_first", "usable_drive") if c in r]].any(axis=1).to_numpy(int)
    n = min(ONSET_CYCLES, len(r))
    k0 = int(np.argmax([ok[k:k + n].sum() for k in range(max(len(r) - n + 1, 1))]))
    r = r.iloc[k0:k0 + n]
    if r.empty:
        raise ValueError(f"no {muscle} opening onsets inside the window")
    sg = E._signal(ctx, pid, task, muscle)
    if sg is None or not sg.get("session"):
        raise ValueError("the recorded session is needed for the detector signal")
    op = E._onset_params(ctx)
    thr = E._threshold(sg["rest"], sg["fs"], op)
    t = np.asarray(sg["t_us"], np.int64)
    x = np.where(sg["mask"], np.nan, sg["cleaned"])
    det = E._detector_signal(x, sg["fs"], op)
    with np.errstate(divide="ignore", invalid="ignore"):
        ddb = (10 * np.log10(np.maximum(det / thr, 1e-2)) if _fin(thr) and thr > 0
               else np.full(det.shape, np.nan))
    ddb = np.where(sg["mask"], np.nan, ddb)
    tm = r.t_mech_us.to_numpy(float)
    keep = np.zeros(len(t), bool)
    for v in tm:
        keep |= (t >= v - (pre + ONSET_MARGIN_S) * 1e6) & (t <= v + (post + ONSET_MARGIN_S) * 1e6)

    def col(name: str, fill: float = np.nan) -> np.ndarray:
        return r[name].to_numpy(float) if name in r else np.full(len(r), fill)

    return {"t_us": t[keep].astype(np.float64), "emg_uv": x[keep], "tkeo_db": ddb[keep],
            "muscle": np.asarray(MUSCLES.index(muscle)), "threshold": np.asarray(thr),
            "k_sd": np.asarray(float(E._tp(ctx, "k_sd", default=3.0))),
            "burst_min_db": np.asarray(float(E._tp(ctx, "burst_min_db", default=6.0))),
            "pre_s": np.asarray(pre), "post_s": np.asarray(post),
            "margin_s": np.asarray(ONSET_MARGIN_S),
            "cycle": col("cycle"), "t_mech_us": tm, "t_omega_us": col("t_omega_us"),
            "t_on_first": col("t_on_first"), "t_on_drive": col("t_on_drive"),
            "usable_first": col("usable_first", 0.0), "usable_drive": col("usable_drive", 0.0),
            "lead_first_ms": col("lead_first"), "lead_drive_ms": col("lead_drive"),
            "burst_db": col("burst_db"), "active_at_start": col("active_at_start", 0.0),
            "n_cycles_task": np.asarray(int(len(rows[rows.muscle == muscle])))}


def _win(d: dict[str, Any]) -> tuple[float, float]:
    m = float(d["margin_s"])
    return -(float(d["pre_s"]) + m), float(d["post_s"]) + m


def onset_timeline(a: Lo.Animation) -> Timeline:
    """One segment per cycle: the onset search window around the mechanical onset plus
    ``ONSET_MARGIN_S`` on each side, in play order (labels: Cycle k/N)."""
    d = Sup._require(a.clip, "onset_rules", "onset_rules")
    lo, hi = _win(d)
    tm = np.asarray(d["t_mech_us"], float)
    n = len(tm)
    return Timeline(tuple((float(v + lo * 1e6), float(v + hi * 1e6)) for v in tm), a.fps,
                    a.speed, tuple(f"Cycle {j + 1} of {n}" for j in range(n)))


def _reason(d: dict[str, Any], j: int, rule: str) -> str:
    from biomech.viz.views import emg_r8 as E

    row = SimpleNamespace(**{f"usable_{rule}": bool(d[f"usable_{rule}"][j]),
                             "active_at_start": bool(d["active_at_start"][j]),
                             "burst_db": float(d["burst_db"][j]),
                             f"t_on_{rule}": float(d[f"t_on_{rule}"][j])})
    return E._reason(row, rule, float(d["burst_min_db"]))


def _cycle_line(d: dict[str, Any], j: int, n: int) -> str:
    parts = []
    for rule, name in (("first", "first onset"), ("drive", "driving burst")):
        lead = float(d[f"lead_{rule}_ms"][j])
        what = _ms(lead) + (" the mechanical onset" if _fin(lead) else "")
        parts.append(f"{name} {what} ({_reason(d, j, rule)})")
    return f"Cycle {j + 1} of {n}: " + "; ".join(parts) + "."


def _running(d: dict[str, Any], rule: str, j: int, shown: bool) -> tuple[int, int, float]:
    """(usable, played, median lead of the usable ones) over cycles 0..j (j itself once its
    onset has been passed)."""
    k = j + 1 if shown else j
    u = np.asarray(d[f"usable_{rule}"][:k], float) > 0
    lead = np.asarray(d[f"lead_{rule}_ms"][:k], float)
    v = lead[u & np.isfinite(lead)]
    return int(u.sum()), k, float(np.median(v)) if v.size else np.nan


class _Board(Text):
    """Right column of ``onset_rules``: this cycle's onsets per rule as the cursor passes them,
    and the running counts and median leads over the cycles played."""

    def __init__(self, panel: str, d: dict[str, Any], rule_names: dict[str, str]) -> None:
        self.d = d
        ys = np.linspace(0.86, 0.51, 6)
        items = []
        for k, (rule, _) in enumerate((("first", 0), ("drive", 0))):
            items.append((0.06, ys[3 * k], "left", lambda s, r=rule: rule_names[r], "label",
                          "ink"))
            items.append((0.10, ys[3 * k + 1], "left", lambda s, r=rule: s[f"{r}_now"], "label",
                          "ink2"))
            items.append((0.10, ys[3 * k + 2], "left", lambda s, r=rule: s[f"{r}_sofar"],
                          "label", "ink2"))
        super().__init__(panel, items, weight=("0", "3"))

    def prepare(self, t_us, seg):
        d = self.d
        t = np.asarray(t_us, float)
        seg = np.asarray(seg, int)
        ex: dict[str, np.ndarray] = {}
        for rule in ("first", "drive"):
            now, sofar = [], []
            for ti, j in zip(t, seg):
                te = float(d[f"t_on_{rule}"][j])
                passed = _fin(te) and ti >= te
                end = ti >= float(d["t_mech_us"][j]) + float(d["post_s"]) * 1e6
                if passed:
                    lead = float(d[f"lead_{rule}_ms"][j])
                    now.append(f"this cycle: {_ms(lead)} the motion, {_reason(d, j, rule)}")
                elif not _fin(te) and end:
                    now.append(f"this cycle: no onset, {_reason(d, j, rule)}")
                else:
                    now.append("this cycle: …")
                nu, k, med = _running(d, rule, int(j), passed or end)
                sofar.append(f"so far: {nu} of {k} cycles usable"
                             + (f", median lead {med:.0f} ms" if np.isfinite(med) else ""))
            ex[f"{rule}_now"] = np.array(now, dtype=object)
            ex[f"{rule}_sofar"] = np.array(sofar, dtype=object)
        self.extra = ex
        super().prepare(t_us, seg)


def onset_rules(a: Lo.Animation, tl: Timeline) -> Composition:
    from biomech.viz.views import emg_r8 as E
    from biomech.viz.views import kin_r8 as K

    clip: Clip = a.clip
    d = Sup._require(clip, "onset_rules", "onset_rules")
    muscle = MUSCLES[int(d["muscle"])]
    ml = MUSCLE_LABEL.get(muscle, muscle)
    lo, hi = _win(d)
    tm = np.asarray(d["t_mech_us"], float)
    n = len(tm)
    xlim = (lo * 1e3, hi * 1e3)
    head, layers = _header(a, tl)
    layers = [ly for ly in layers if not isinstance(ly, Lo.Chip)]
    text = next(ly for ly in layers if isinstance(ly, Text))
    text.items[2] = (0.80, 0.58, "right",
                     lambda s: f"{s['t_s'] * 1e3:+.0f} ms re motion onset", "title", "ink")
    text.t_ref = lambda t, seg: (np.asarray(t, float) - tm[np.asarray(seg, int)]) * 1e-6
    regions = [head]
    y0, bh = Sup._body(HEADER_H + Sup.GAP)
    role = f"entity:{muscle}"
    t = d["t_us"]
    pre_ms, post_ms = 1e3 * float(d["pre_s"]), 1e3 * float(d["post_s"])

    step = 100.0 if xlim[1] - xlim[0] <= 800 else 200.0
    xt = [float(v) for v in np.arange(np.ceil(xlim[0] / step) * step, xlim[1] + 1e-9, step)]

    def xaxis(xlab: bool) -> Axis:
        return Axis("Time re mechanical onset" if xlab else "",
                    "\\milli\\second" if xlab else "", lim=xlim, ticks=xt,
                    ticklabels=[f"{v:g}" for v in xt] if xlab else [""] * len(xt))

    def base(title: str, yax: Axis, xlab: bool) -> Panel:
        p = Panel(title, xaxis(xlab), yax, legend="none", sharex="t")
        p.series.append(Series("hspan", x=np.array([-pre_ms]), y2=np.array([post_ms]),
                               role="faint", style={"alpha": 0.35}))
        p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        return p

    emg = np.asarray(d["emg_uv"], float)
    amp = Lo._nice(float(np.nanpercentile(np.abs(emg), 99.9)) * 1.05) \
        if np.isfinite(emg).any() else 1.0
    pa = base(f"Band-passed EMG, {ml} (µV)", Axis("", "", lim=(-amp, amp)), False)
    db = np.asarray(d["tkeo_db"], float)
    top = max(20.0, Lo._nice(float(np.nanpercentile(db, 99.9)))) if np.isfinite(db).any() \
        else 20.0
    det = E._det_label()
    pb = base(f"{det} detector signal re rest threshold (dB)", Axis("", "", lim=(-20.0, top)),
              False)
    pb.series.append(Series("hline", y=np.array([0.0]), role="threshold"))
    panels = [("emg", 1.0, pa, [SegmentTrace("emg:0", t, emg, tm, lo, hi, role, width=0.9,
                                             cursor=True)]),
              ("det", 1.5, pb, [SegmentTrace("det:0", t, db, tm, lo, hi, role, width=1.3,
                                             cursor=True)])]
    for rule in ("first", "drive"):
        te = np.asarray(d[f"t_on_{rule}"], float)
        yv = np.zeros(len(te))              # on the threshold line, as emg_r8 ``chain`` (c)
        mk, rl = getattr(RULE_GLYPH, rule)
        panels[1][3].append(SegmentEvents("det:0", te, yv, tm, rl, mk, size=1.15))
    k = clip.kin
    th = None
    if k is not None and np.isfinite(k["theta_deg"]).any():
        th, th_t = np.asarray(k["theta_deg"], float), np.asarray(k["t_us"], float)
    elif clip.imu is not None:
        th, th_t = np.asarray(clip.imu["theta_quat"], float), np.asarray(clip.imu["t_us"], float)
    if th is not None:
        sel = np.zeros(len(th_t), bool)
        for v in tm:
            sel |= (th_t >= v + lo * 1e6) & (th_t <= v + hi * 1e6)
        yl = Lo._lim([th[sel]], 1.0, 5.0)
        pc = base("Opening angle (°)", Axis("", "", lim=yl), True)
        panels.append(("angle", 1.0, pc, [SegmentTrace("angle:0", th_t, th, tm, lo, hi,
                                                       "primary", width=1.6, cursor=True)]))
    else:
        panels[-1][2].x = xaxis(True)
    regions.append(_column("left", [(pnl, w, ls) for _, w, pnl, ls in panels],
                           (0.0, y0, 0.64, bh), layers))
    rule_names = {r: f"{E.RULE_LABEL[r][:1].upper()}{E.RULE_LABEL[r][1:]} rule ({tag})"
                  for r, tag in K.RULES}

    def draw(fig, theme, sc):
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhspan(0, 1, color=style.hexcolor("panel", theme), lw=0)
        fig.text(0.06, 0.94, f"{det} onsets, {ml}", ha="left", va="center", fontweight="bold",
                 fontsize=style.FONT_PT["label"] * sc * 1.15,
                 color=style.hexcolor("ink", theme))
        for (rule, _), y in zip(K.RULES, (0.86, 0.86 - 3 * 0.07)):
            mk, rl = getattr(RULE_GLYPH, rule)
            ax.plot([0.035], [y], ls="none", marker=mk, ms=style.MARKER_PT * sc * 3.6,
                    color=style.hexcolor(style.role(rl)[0], theme),
                    markeredgecolor=style.hexcolor("ink", theme), markeredgewidth=0.5 * sc)
        fig._anim_axes = {}

    k_sd = float(d["k_sd"])
    intro = (f"{det} detector signal of the {ml} in dB re its rest threshold (rest mean + "
             f"{k_sd:g} SD). Shaded: the onset search window, {pre_ms:.0f} ms before to "
             f"{post_ms:.0f} ms after the mechanical onset (dashed: the earliest of the IMU "
             "velocity and acceleration onsets).")
    rules = ("First-onset rule: the first threshold crossing in the window. Driving-burst "
             "rule: the last crossing at or before the mechanical onset whose burst is still "
             f"on. A channel-cycle is usable only with a burst ≥ {float(d['burst_min_db']):g} "
             "dB above rest and no activity at the window start.")

    def draw_all(fig, theme, sc):
        import textwrap

        draw(fig, theme, sc)
        W, _ = fig.get_size_inches()
        fs = style.FONT_PT["label"] * sc * 0.8
        wrap = max(24, int(W * 72 / (fs * 0.56)))
        txt = "\n".join(textwrap.wrap(intro, wrap) + [""] + textwrap.wrap(rules, wrap))
        fig.text(0.06, 0.45, txt, ha="left", va="top", fontsize=fs,
                 color=style.hexcolor("ink2", theme), linespacing=1.3)

    regions.append(Region("board", (0.65, y0, 0.345, bh), draw=draw_all))
    layers.append(_Board("board", d, rule_names))
    per = [[_cycle_line(d, j, n)] for j in range(n)]
    cap, cap_layer = Sup._caption_band(a, tl, per)
    regions.append(cap)
    layers.append(cap_layer)
    regions.append(Region("key", (0.0, 1 - KEY_H - 0.006, 1.0, KEY_H), draw=_key(
        [("line", role, ml), ("dash", "threshold", "Rest threshold"),
         ("dash", "reference", "Mechanical onset"), ("square", "faint", "Search window"),
         ("v", RULE_GLYPH.first[1], "First onset"), ("d", RULE_GLYPH.drive[1], "Driving burst"),
         ("line", "primary", "Opening angle")], None, gap_in=0.24)))
    return Composition("onset_rules", regions, layers, tl, meta={"muscle": muscle})


# -- registration ---------------------------------------------------------------------------------------
def _register() -> None:
    S.EXTRAS.update(emg_spectro=_emg_spectro, onset_rules=_onset_explainer)
    Lo.LAYOUTS.update(workspace3d=workspace3d, emg_spectrogram=emg_spectrogram,
                      onset_rules=onset_rules)
    Lo.COHORT_LAYOUTS.add("workspace3d")
    Lo.COHORT_LOADERS["workspace3d"] = load_workspace_clip
    Lo.EXTRAS.update(emg_spectrogram=("emg_spectro",), onset_rules=("onset_rules",))
    Lo.TIMELINES.update(workspace3d=workspace_timeline, onset_rules=onset_timeline)
    Lo.PREVIEWS.update(workspace3d=Sup._preview_from(["jaw"]),
                       emg_spectrogram=Sup._preview_from(["left", "strips"]),
                       onset_rules=Sup._preview_from(["left"], film=1))
    Lo.DEFAULT_PHASES.update(emg_spectrogram=("chewing",), onset_rules=("open_close",))
    Lo.DEFAULT_SPEED.update(workspace3d=1.0, emg_spectrogram=0.5, onset_rules=0.25)
    Lo.STILLS.update({"workspace3d", "emg_spectrogram", "onset_rules"})
    Lo.STILL_WIDTH.update(workspace3d=STILL_WIDTH, emg_spectrogram=STILL_WIDTH,
                          onset_rules=STILL_WIDTH)


_register()

__all__ = ["emg_spectrogram", "load_workspace_clip", "onset_rules", "onset_timeline",
           "workspace3d", "workspace_timeline"]
