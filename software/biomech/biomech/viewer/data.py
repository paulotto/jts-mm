# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Data-explorer helpers: raw and processed signals of one phase, camera frames, 3D jaw.

Pure functions returning Plotly figures / data; the Dash layer only wires them.
Everything goes through :class:`biomech.io.context.DataContext` (cached EMG
preparations whose files are closed on eviction, the take that holds a phase,
included participants only). Signals are shown in seconds from the phase start.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from biomech.io.context import DataContext
from biomech.viz import style
from biomech.viz.downsample import minmax
from biomech.viz.render_plotly import template

LABEL_NAMES = ("REST", "OPEN", "CLOSE", "PROTRUSION", "RETRUSION", "LEFT", "RIGHT")
ROWS = ("EMG", "Opening angle (°)", "Incisal point re median (mm)", "IMU", "Intent label")
# Unit of the normalized EMG envelope (the reference is the largest activity recorded,
# analysis_plan D11). One constant: the axis title and the mode selector read it.
EMG_NORM_LABEL = "% of max recorded"
# Mask marks: one row per channel below the envelope, coloured by the rule that hit
# (biomech.emg.artifacts.RULES); "dilation" = masked only by the +-dilate_s margin.
MASK_REASONS = {"leadoff": ("lead-off", "ink2", "square"),
                "rail": ("rail (saturation)", "vermillion", "x"),
                "envelope": ("envelope > factor x reference", "purple", "triangle-up"),
                "lowfreq": ("low-frequency artifact", "orange", "diamond"),
                "dilation": ("dilation margin", "grey", "circle")}
ALL_MOTIONS = "all"           # explorer phase selector: every motion task


def _c(entity: str, theme: str) -> str:
    return style.hexcolor(style.ENTITY.get(entity, entity), theme)


def _dec(t: np.ndarray, y: np.ndarray, bins: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = minmax(y, bins)
    idx = np.unique(np.concatenate([lo, hi]))
    return t[idx], y[idx]


def phase_names(ctx: DataContext, pid: str) -> list[str]:
    return [ph.name for ph in ctx.phases(pid, "motion")] + [
        ph.name for ph in ctx.phases(pid, "mvc")]


def _tasks() -> tuple[str, ...]:
    from biomech.viz.views.data import TASKS

    return TASKS


def phase_options(names: list[str]) -> list[dict[str, str]]:
    """Explorer phase selector: "all motions" (when a motion task exists) + every phase."""
    from biomech.viz.views.vocab import phase_label

    opts = [{"label": phase_label(n), "value": n} for n in names]
    if any(t in names for t in _tasks()):
        opts.insert(0, {"label": "all motions", "value": ALL_MOTIONS})
    return opts


def as_selection(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else [v for v in value if v]


def selected_phases(value: str | list[str] | None, names: list[str]) -> list[str]:
    """Concrete phases of a selection: "all" expands to the motion tasks present (task
    order), the rest keep their order; unknown names are dropped."""
    out: list[str] = []
    for v in as_selection(value):
        for p in ([t for t in _tasks() if t in names] if v == ALL_MOTIONS else [v]):
            if p in names and p not in out:
                out.append(p)
    return out


def primary_phase(value: str | list[str] | None, names: list[str]) -> str | None:
    """The one phase of single-phase views (time series, camera, most atlas views): the
    concrete motion picked *last* (a multi-select appends, so picking chewing after
    open-close switches the time series to chewing while the 3-D view overlays both);
    "all motions" alone means open-close (else the first task)."""
    for v in reversed(as_selection(value)):
        if v != ALL_MOTIONS and v in names:
            return v
    sel = selected_phases(value, names)
    if "open_close" in sel:
        return "open_close"
    return sel[0] if sel else (names[0] if names else None)


def view_phase(value: str | list[str] | None, names: list[str]) -> str | list[str] | None:
    """Phase argument of a multi-phase view (registry ``multi_phase``): "all", one phase or a
    list of motion tasks."""
    sel = as_selection(value)
    if ALL_MOTIONS in sel:
        return ALL_MOTIONS
    chosen = selected_phases(sel, names)
    tasks = [t for t in _tasks() if t in chosen]          # task order: a stable id
    if not tasks:
        return primary_phase(value, names)
    return tasks[0] if len(tasks) == 1 else tasks


def pid_token(ctx: DataContext, pid: str) -> str:
    """The participant in a URL: its pseudonym ("P3"), never the study id."""
    return f"P{ctx.pseudonym(pid)}"


def pid_from_token(ctx: DataContext, token: str | None) -> str | None:
    """Participant of a URL token ("P3"; a legacy study id still resolves), else None."""
    if not token:
        return None
    pids = ctx.pids()
    if token in pids:
        return token
    t = token[1:] if token[:1] in ("P", "p") else token
    return pids[int(t) - 1] if t.isdigit() and 1 <= int(t) <= len(pids) else None


def pid_label(ctx: DataContext, pid: str) -> str:
    """Participant label everywhere in the viewer: the pseudonym only ("P10"), as in the
    figures. Study ids appear only on the Overview page with "show study ids"
    ticked (:func:`unit_label`)."""
    return f"P{ctx.pseudonym(pid)}"


def unit_label(ctx: DataContext, unit: str, operator: bool = False) -> str:
    """Overview label of a result unit: the pseudonym, "excluded" for participants outside
    the cohort, the unit itself for study-level units; ``operator`` adds the study id."""
    from biomech.results import COHORT_UNIT

    if unit == COHORT_UNIT:
        return "cohort"
    names = ctx.pseudonyms()
    base = f"P{names[unit]}" if unit in names else "excluded"
    return f"{base} · {unit}" if operator else base


def _mask_parts(pe) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """(dilated full mask, rule -> undilated hits) of a PhaseEmg (MaskResult or bare array)."""
    mr = getattr(pe, "masks", None)
    if mr is None and hasattr(getattr(pe, "mask", None), "hits"):
        mr = pe.mask
    full = np.asarray(mr.mask if mr is not None else pe.mask, dtype=bool)
    hits = {k: np.asarray(v, dtype=bool) for k, v in (getattr(mr, "hits", None) or {}).items()
            if np.shape(v) == full.shape}
    return full, hits


def mask_excludes(pe, reason: str) -> bool:
    """Does ``reason`` exclude samples (analysis rule, D14) or only flag them?"""
    mr = getattr(pe, "masks", None)
    rules = getattr(mr, "analysis_rules", None)
    return reason == "dilation" or rules is None or reason in rules


def mask_reasons(pe, i: int) -> dict[str, np.ndarray]:
    """Per reason, the masked samples of channel ``i``; "dilation" = masked by no rule."""
    full, hits = _mask_parts(pe)
    out: dict[str, np.ndarray] = {}
    anyhit = np.zeros(full.shape[0], dtype=bool)
    for r in MASK_REASONS:
        if r in hits and hits[r][:, i].any():
            out[r] = hits[r][:, i]
            anyhit |= hits[r][:, i]
    rest = full[:, i] & ~anyhit
    if rest.any():
        out["dilation"] = rest
    return out


def unmasked_norm(prep, pe) -> np.ndarray:
    """RMS envelope of the cleaned signal without any mask, on the scale of ``pe.norm``.

    The scale is read off the phase itself (norm / masked envelope per channel), so it is
    always the reference ``pe.norm`` was normalized with, whichever that is.
    """
    from biomech.emg.pipeline import rms_envelope

    win = float(getattr(getattr(prep, "p", None), "envelope_s", 0.15) or 0.15)
    env = rms_envelope(np.asarray(pe.cleaned, float), float(pe.fs), win)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(pe.norm, float) / np.asarray(pe.env, float)
    ratio[~np.isfinite(ratio)] = np.nan
    scale = np.full(env.shape[1], np.nan)
    for c in range(env.shape[1]):
        if np.isfinite(ratio[:, c]).any():
            scale[c] = np.nanmedian(ratio[:, c])
    return env * scale[None, :]


OVERLAYS = {"onsets": "mechanical onsets", "release": "causal release decisions",
            "decoder": "decoder output (LOSO, primary model)"}
# Rows of the signals figure (title, relative height); the mask row only in envelope mode.
SIGNAL_ROWS = (("EMG", 0.28), ("Masks", 0.10), ("Angle (°)", 0.14),
               ("Incisal (mm)", 0.14), ("ω (°/s)", 0.10), ("|a| (m/s²)", 0.10),
               ("Label", 0.14))


def channel_names(ctx: DataContext, pid: str | None) -> list[str]:
    """EMG channel names of the participant's session (explorer channel selector)."""
    if not pid:
        return []
    try:
        return list(ctx.emg(pid).names)
    except Exception:                    # unreadable session: no channels offered
        return []


MVC_NOTE = ("MVC take: no optical tracking, IMU processing or intent labels "
            "(EMG only)")
# A study without the dataset stages (the validation study runs only its own ``ref_*``
# stages): the angle, incisal, IMU and label rows have nothing stored to show.
NO_STAGE_NOTE = "no '{stage}' stage in this study: see Results → Validation"
NO_RESULT_NOTE = "no stored '{stage}' results for this participant"
EMG_MASKED_NOTE = "all samples masked in this take: see raw or cleaned"
EMG_FLAT_NOTE = "no EMG signal in this take (constant on every selected channel)"


def _row_note(fig: go.Figure, row: int, text: str, theme: str) -> None:
    """A note centred in row ``row`` of the signals figure (an intentionally empty row).
    Plotly draws only subplots that hold a trace, so an empty one gets an invisible trace;
    the note is anchored to the row's own axes."""
    n = "" if row == 1 else row
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False,
                             hoverinfo="skip"), row=row, col=1)
    fig.add_annotation(text=text, xref=f"x{n} domain", yref=f"y{n} domain", x=0.5, y=0.5,
                       showarrow=False, font={"size": 11, "color": style.hexcolor("ink2", theme)})


def explorer_figure(ctx: DataContext, pid: str, phase: str, channels: list[str] | None,
                    emg_mode: str, theme: str = "light", unmasked: bool = False,
                    overlays: list[str] | None = None, height_px: int | None = None
                    ) -> go.Figure:
    """Synchronized signals of one phase, one row per quantity (each with its own unit):
    EMG, the artifact marks by reason (envelope mode), opening angle, incisal point,
    angular velocity, relative acceleration and the intent label.

    ``unmasked``: in envelope mode also draw the envelope without the artifact mask (faint).
    ``overlays``: keys of :data:`OVERLAYS` -- vertical marks on every row (mechanical
    onsets, causal release decisions) and the decoder's predicted class in the label row.
    Spike lines on all rows follow the cursor. ``channels``: the EMG channels to draw
    (``None``: all; an empty list draws none, with a note). ``height_px``: the layout height
    (the viewer's graph is responsive and fills its box, which carries the height).
    MVC phases have EMG only: the other rows say so instead of staying silently empty.
    """
    overlays = list(overlays or [])
    heights = [h for _, h in SIGNAL_ROWS]
    fig = make_subplots(rows=len(SIGNAL_ROWS), cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=heights)
    take, ph = ctx.phase(pid, phase)
    if ph is None:
        return fig
    t0 = ph.t_start_us
    prep = ctx.emg(pid)
    pe = prep.phase(phase, take=take)
    ts = (pe.t_us - t0) * 1e-6
    faint = unmasked_norm(prep, pe) if emg_mode == "envelope" and unmasked else None
    shown: set[str] = set()
    sel = [i for i, ch in enumerate(pe.names) if channels is None or ch in channels]
    if not sel:
        _row_note(fig, 1, "no EMG channel selected", theme)
    for row_i, i in enumerate(sel):
        ch = pe.names[i]
        label = ch.replace("_", " ")
        if faint is not None and np.isfinite(faint[:, i]).any():
            tx, yx = _dec(ts, faint[:, i])
            fig.add_trace(go.Scattergl(x=tx, y=yx, mode="lines", name=f"{label}, unmasked",
                                       legend="legend", showlegend=False, opacity=0.35,
                                       hoverinfo="skip",
                                       line={"width": 1, "color": _c(ch, theme)}), row=1, col=1)
        y = {"raw": pe.raw[:, i] - np.median(pe.raw[:, i]), "clean": pe.cleaned[:, i],
             "envelope": pe.norm[:, i]}[emg_mode]
        unit = "µV" if emg_mode != "envelope" else EMG_NORM_LABEL
        tx, yx = _dec(ts, y.astype(float))
        fig.add_trace(go.Scattergl(x=tx, y=yx, mode="lines", name=label, legend="legend",
                                   line={"width": 1, "color": _c(ch, theme)},
                                   hovertemplate=f"{label}: %{{y:.3~g}} {unit}<extra></extra>"),
                      row=1, col=1)
        if emg_mode != "envelope":
            continue
        # Mask marks in their own row: y = channel slot, so the EMG autorange is untouched.
        for reason, m in mask_reasons(pe, i).items():
            name, col, sym = MASK_REASONS[reason]
            name = f"{'masked' if mask_excludes(pe, reason) else 'flagged (kept)'}: {name}"
            tm, _ = _dec(ts[m], np.zeros(int(m.sum())))
            fig.add_trace(go.Scattergl(x=tm, y=np.full(tm.size, float(row_i)), mode="markers",
                                       marker={"size": 4, "color": style.hexcolor(col, theme),
                                               "symbol": sym},
                                       name=name, legend="legend",
                                       legendgroup=f"mask-{reason}",
                                       showlegend=reason not in shown,
                                       hovertemplate=f"{label}: {name}<extra></extra>"),
                          row=2, col=1)
            shown.add(reason)
    if sel and not (np.ptp(pe.raw[:, sel], axis=0) > 0).any():
        _row_note(fig, 1, EMG_FLAT_NOTE, theme)        # e.g. electrodes not applied
    elif emg_mode == "envelope" and sel and not np.isfinite(pe.norm[:, sel]).any():
        _row_note(fig, 1, EMG_MASKED_NOTE, theme)
    fig.update_yaxes(title_text={"raw": "µV", "clean": "µV", "envelope": EMG_NORM_LABEL}[emg_mode],
                     row=1, col=1)
    fig.update_yaxes(tickvals=list(range(len(sel))), showgrid=False,
                     ticktext=[pe.names[i].replace("_", " ") for i in sel],
                     range=[len(sel) - 0.5, -0.5], tickfont={"size": 8}, row=2, col=1)
    if take == "motion":
        missing = NO_STAGE_NOTE if "dataset" not in ctx.config.groups else NO_RESULT_NOTE
        imu = ctx.arrays("imu", pid)
        if not imu:
            for r in (3, 5, 6):
                _row_note(fig, r, missing.format(stage="imu"), theme)
        if f"{phase}__t_us" in imu:
            ti = (imu[f"{phase}__t_us"] - t0) * 1e-6
            for name, lab, col in (("theta_opt", "optical", "ink"),
                                   ("theta_quat", "IMU quaternion", "blue"),
                                   ("theta_comp", "IMU complementary", "green"),
                                   ("theta_tilt", "IMU tilt", "orange")):
                fig.add_trace(go.Scattergl(x=ti, y=imu[f"{phase}__{name}"], mode="lines",
                                           name=lab, legend="legend3",
                                           hovertemplate=f"{lab}: %{{y:.3~g}}°<extra></extra>",
                                           line={"width": 1.3, "color": style.hexcolor(col, theme)}),
                              row=3, col=1)
            # the axis follows the optical and quaternion angles: the accelerometer tilt
            # spikes with linear acceleration (chewing) and would squash the others
            ref = np.concatenate([np.asarray(imu[f"{phase}__{k}"], float)
                                  for k in ("theta_opt", "theta_quat")])
            ref = ref[np.isfinite(ref)]
            if ref.size:
                lo, hi = np.percentile(ref, [0.5, 99.5])
                pad = 0.15 * max(hi - lo, 1.0)
                fig.update_yaxes(range=[lo - pad, hi + pad], row=3, col=1)
            fig.add_trace(go.Scattergl(x=ti, y=imu[f"{phase}__omega_dps"], mode="lines",
                                       name="ω", showlegend=False,
                                       hovertemplate="ω: %{y:.3~g} °/s<extra></extra>",
                                       line={"width": 1, "color": style.hexcolor("purple", theme)}),
                          row=5, col=1)
            fig.add_trace(go.Scattergl(x=ti, y=imu[f"{phase}__acc_rel_ms2"], mode="lines",
                                       name="|a rel|", showlegend=False,
                                       hovertemplate="|a rel|: %{y:.3~g} m/s²<extra></extra>",
                                       line={"width": 1, "color": style.hexcolor("sky", theme)}),
                          row=6, col=1)
        kin = ctx.arrays("kinematics", pid)
        if not kin:
            _row_note(fig, 4, missing.format(stage="kinematics"), theme)
        if f"{phase}__t_us" in kin:
            tk = (kin[f"{phase}__t_us"] - t0) * 1e-6
            inc = kin[f"{phase}__incisor"]
            inc = inc - np.nanmedian(inc, axis=0)
            for k, (ax, role) in enumerate((("x anterior", "axis:x"), ("y left", "axis:y"),
                                            ("z up", "axis:z"))):
                fig.add_trace(go.Scattergl(x=tk, y=inc[:, k], mode="lines", name=ax,
                                           legend="legend4",
                                           hovertemplate=f"{ax}: %{{y:.3~g}} mm<extra></extra>",
                                           line={"width": 1.2, "color": style.hexcolor(
                                               style.role(role)[0], theme)}), row=4, col=1)
        lab = ctx.arrays("labels", pid)
        if not lab:
            _row_note(fig, 7, missing.format(stage="labels"), theme)
        if f"{phase}__t_us" in lab:
            tl = (lab[f"{phase}__t_us"] - t0) * 1e-6
            y = lab[f"{phase}__label"].astype(float)
            y[y < 0] = np.nan
            fig.add_trace(go.Scattergl(x=tl, y=y, mode="lines", name="label (optical)",
                                       legend="legend7", showlegend="decoder" in overlays,
                                       line={"shape": "hv", "width": 1.5,
                                             "color": style.hexcolor("ink2", theme)}),
                          row=7, col=1)
            fig.update_yaxes(tickvals=list(range(7)), ticktext=list(LABEL_NAMES), row=7, col=1)
        if "decoder" in overlays:
            _decoder_trace(fig, ctx, pid, ph, t0, theme)
    else:
        for r in (3, 4, 5, 6, 7):
            _row_note(fig, r, MVC_NOTE if r == 3 else "MVC take: not available", theme)
    _event_overlays(fig, ctx, pid, phase, t0, overlays, theme)
    fig.update_xaxes(title_text="time from phase start (s)", row=len(SIGNAL_ROWS), col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                     spikedash="solid", spikecolor=style.hexcolor("ink2", theme))
    # One horizontal legend above each row that has one.
    legends = {}
    for r, name in ((1, "legend"), (3, "legend3"), (4, "legend4"), (7, "legend7")):
        dom = fig.layout[f"yaxis{'' if r == 1 else r}"].domain
        legends[name] = {"x": 0.0, "y": dom[1] + 0.004, "xanchor": "left", "yanchor": "bottom",
                         "orientation": "h", "bgcolor": "rgba(0,0,0,0)",
                         "font": {"size": 11}}
    # The EMG legend (channels + mask reasons) wraps to two or three lines on a narrow card:
    # reserve the room and keep the margins fixed; Plotly's margin auto-expansion would
    # shrink the plot area under the paper-positioned legends of the other rows.
    fig.update_layout(height=int(height_px or 980), template=template(theme),
                      margin={"l": 64, "r": 16, "t": 80 if shown else 28, "b": 44,
                              "autoexpand": False},
                      uirevision=f"{pid}-{phase}",
                      hovermode="x", hoverdistance=-1, clickmode="event", **legends)
    for r, (title, _) in enumerate(SIGNAL_ROWS, start=1):
        if r == 1:
            continue
        fig.update_yaxes(title_text=title if r != 2 else None, title_font={"size": 11},
                         row=r, col=1)
    if emg_mode != "envelope":
        fig.update_yaxes(visible=False, row=2, col=1)
    return fig


def _event_overlays(fig: go.Figure, ctx: DataContext, pid: str, phase: str, t0: int,
                    overlays: list[str], theme: str) -> None:
    """Vertical marks across all rows: mechanical onsets / causal release decisions."""
    marks: list[tuple[float, str, str]] = []
    if "onsets" in overlays:
        on = ctx.table("timing", "onsets", [pid])
        if not on.empty and "t_mech_us" in on:
            g = on[on.task == phase].drop_duplicates(["cycle", "direction"])
            for _, r in g.iterrows():
                marks.append(((r.t_mech_us - t0) * 1e-6,
                              "orange" if r.direction == "opening" else "blue",
                              f"{r.direction} onset"))
    if "release" in overlays:
        rel = ctx.table("timing", "release", [pid])
        if not rel.empty and "t_release_us" in rel:
            for v in rel[rel.task == phase].t_release_us.dropna():
                marks.append(((v - t0) * 1e-6, "vermillion", "causal release decision"))
    # One full-height shape per mark (the x axes are shared), set in one layout update:
    # add_vline per row was ~0.1 s per mark.
    shapes = [{"type": "line", "x0": float(x), "x1": float(x), "xref": "x", "yref": "paper",
               "y0": 0, "y1": 1, "opacity": 0.8, "layer": "below",
               "line": {"color": style.hexcolor(col, theme), "width": 1, "dash": "dot"}}
              for x, col, _ in marks]
    if shapes:
        fig.update_layout(shapes=list(fig.layout.shapes) + shapes)


def _decoder_trace(fig: go.Figure, ctx: DataContext, pid: str, ph: Any, t0: int,
                   theme: str) -> None:
    from biomech.results import COHORT_UNIT

    try:
        a = ctx.arrays("ml_eval", COHORT_UNIT)
    except Exception:
        return
    for fset, col in (("emg_imu", "vermillion"), ("imu", "green")):
        t, yhat = a.get(f"pred__{pid}__{fset}__t_us"), a.get(f"pred__{pid}__{fset}__yhat")
        if t is None or yhat is None:
            continue
        sel = (t >= ph.t_start_us) & (t <= ph.t_end_us)
        if not sel.any():
            continue
        fig.add_trace(go.Scattergl(x=(t[sel] - t0) * 1e-6, y=yhat[sel].astype(float),
                                   mode="markers", name=f"decoded ({fset.replace('_', '+')})",
                                   legend="legend7", marker={"size": 3, "color":
                                                             style.hexcolor(col, theme)}),
                      row=7, col=1)


def preview_frames(ctx: DataContext, pid: str, phase: str, t_us: int) -> list[tuple[str, str]]:
    """(camera, data-URI) of the stored preview JPEGs nearest to ``t_us`` in the phase's take."""
    take = ctx.take_of(pid, phase) or "motion"
    best: dict[str, tuple[int, str]] = {}
    for r in ctx.previews(pid, take):
        d = abs(int(r["t_master_us"]) - t_us)
        if r["cam_id"] not in best or d < best[r["cam_id"]][0]:
            best[r["cam_id"]] = (d, r["abs_path"])
    out = []
    for cam, (d, path) in sorted(best.items()):
        if d > 2_000_000:
            continue
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        out.append((f"{cam} (Δt {d / 1e3:.0f} ms)", f"data:image/jpeg;base64,{data}"))
    return out


def landmark_rows(ctx: DataContext, pid: str) -> list[dict[str, Any]]:
    """The eight landmarks of a participant: what was touched, frame, hold quality."""
    lm = ctx.table("kinematics", "landmarks", [pid])
    an = ctx.table("kinematics", "anatomy", [pid])
    if lm.empty:
        return []
    rows = []
    for _, r in lm.iterrows():
        rows.append({"landmark": r["name"], "touched": r.get("description", ""),
                     "frame": r["body"].upper(), "role": r.get("role_used", ""),
                     "hold MAD (mm)": f"{float(r['mad_mm']):.2f}", "samples": int(r["n"])})
    if not an.empty:
        a = an.iloc[0]
        rows.append({"landmark": "check", "touched": "tooth roles vs. condyle geometry",
                     "frame": "", "role": "agrees" if a.get("roles_upper_geometry_agrees", 0)
                     and a.get("roles_lower_geometry_agrees", 0) else
                     ("relabelled" if a.get("roles_relabelled", 0) else "ambiguous"),
                     "hold MAD (mm)": "", "samples": ""})
        rows.append({"landmark": "check", "touched": "condyle-incisor distance L / R (mm)",
                     "frame": "MP", "role": "plausible" if a.get("condyles_plausible", 0)
                     else "implausible",
                     "hold MAD (mm)": f"{a.get('condyle_incisor_left_mm', float('nan')):.0f} / "
                                      f"{a.get('condyle_incisor_right_mm', float('nan')):.0f}",
                     "samples": ""})
    return rows


def phase_hint(phase: str) -> str:
    from biomech.frames.landmarks import LANDMARKS

    spec = LANDMARKS.get(phase)
    return f"Landmark hold: CT tip on the {spec.description}" if spec else ""


def jaw3d_figure(ctx: DataContext, pid: str, phase: str | list[str],
                 theme: str = "light", view: str = "oblique") -> go.Figure:
    """Mandibular landmark paths in the phase(s) plus the static maxillary landmarks.

    One phase: each landmark in its own colour. Several phases (or "all"): each task in its
    task colour (``viz.views.data.TASK_ROLE``), the incisal path bold with one legend entry
    per task, molar and condyle paths thin in the same colour (one legend group per task).
    """
    from biomech.viz.views.data import TASK_LABEL, TASK_ROLE

    kin = ctx.arrays("kinematics", pid)
    names = sorted({k.split("__", 1)[0] for k in kin if "__" in k})
    phases = (selected_phases(phase, names) if not isinstance(phase, str) or phase == ALL_MOTIONS
              else [phase])
    fig = go.Figure()
    moving = (("incisor", "lower incisor"), ("left", "lower left molar"),
              ("right", "lower right molar"), ("tmj_left", "left condyle point"),
              ("tmj_right", "right condyle point"))
    multi = len(phases) > 1
    for ph in phases:
        tcol = style.hexcolor(style.role(TASK_ROLE.get(ph, "primary"))[0], theme)
        tlab = TASK_LABEL.get(ph, ph)
        for role, label in moving:
            key = f"{ph}__{role}"
            if key not in kin:
                continue
            p = kin[key]
            if not multi:
                fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                                           name=label,
                                           line={"color": _c(role, theme), "width": 4}))
                continue
            main = role == "incisor"
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                                       name=tlab if main else f"{tlab}: {label}",
                                       legendgroup=ph, showlegend=main,
                                       opacity=1.0 if main else 0.5,
                                       line={"color": tcol, "width": 4 if main else 2}))
    lm = ctx.table("kinematics", "landmarks", [pid])
    T = kin.get("T_max_hp")
    if T is not None and not lm.empty:
        hp = lm[lm.body == "hp"]
        P = hp[["x", "y", "z"]].to_numpy(float) @ T[:3, :3].T + T[:3, 3]
        fig.add_trace(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
                                   name="static landmarks", hovertext=list(hp.name),
                                   marker={"size": 4, "color": style.hexcolor("ink2", theme),
                                           "symbol": "diamond"}))
    # Last trace: the time cursor of the signals plot (filled by a Patch on click).
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers", name="cursor",
                               showlegend=False, hoverinfo="skip",
                               marker={"size": 6, "color": style.hexcolor("ink", theme),
                                       "line": {"width": 2,
                                                "color": style.hexcolor("panel", theme)}}))
    ax = {"tickfont": {"size": 9}, "title": {"font": {"size": 10}}}
    fig.update_layout(height=480, template=template(theme),
                      margin={"l": 0, "r": 0, "t": 4, "b": 0},
                      uirevision=f"{pid}-3d-{view}",
                      legend={"orientation": "h", "y": -0.02, "yanchor": "top", "x": 0,
                              "font": {"size": 10}},
                      scene={"aspectmode": "data", "domain": {"x": [0, 1], "y": [0.14, 1]},
                             "camera": CAMERAS.get(view, CAMERAS["oblique"]),
                             "xaxis": {**ax, "title": {"text": "x anterior (mm)",
                                                       "font": {"size": 10}}},
                             "yaxis": {**ax, "title": {"text": "y left (mm)",
                                                       "font": {"size": 10}}},
                             "zaxis": {**ax, "title": {"text": "z up (mm)",
                                                       "font": {"size": 10}}}})
    return fig


def cursor_points(ctx: DataContext, pid: str, phase: str, t_s: float
                  ) -> tuple[list[float], list[float], list[float]]:
    """Moving landmarks (incisor, molars, condyle points) at the optical frame nearest to
    ``t_s`` seconds from the phase start; empty lists when that phase has no poses."""
    kin = ctx.arrays("kinematics", pid)
    t = kin.get(f"{phase}__t_us")
    if t is None or len(t) == 0:
        return [], [], []
    i = int(np.argmin(np.abs((t - t[0]) * 1e-6 - float(t_s))))
    pts = [kin[k][i] for k in (f"{phase}__{r}" for r in ("incisor", "left", "right",
                                                          "tmj_left", "tmj_right")) if k in kin]
    pts = [p for p in pts if np.isfinite(p).all()]
    return ([float(p[0]) for p in pts], [float(p[1]) for p in pts], [float(p[2]) for p in pts])


# Orthographic camera presets of the 3-D jaw view (maxillary frame: x anterior, y left, z up).
CAMERAS = {
    "oblique": {"eye": {"x": 1.3, "y": -1.3, "z": 0.7}, "projection": {"type": "orthographic"}},
    "sagittal": {"eye": {"x": 0.0, "y": -2.2, "z": 0.0}, "up": {"x": 0, "y": 0, "z": 1},
                 "projection": {"type": "orthographic"}},
    "frontal": {"eye": {"x": 2.2, "y": 0.0, "z": 0.0}, "up": {"x": 0, "y": 0, "z": 1},
                "projection": {"type": "orthographic"}},
    "horizontal": {"eye": {"x": 0.0, "y": 0.0, "z": 2.2}, "up": {"x": 1, "y": 0, "z": 0},
                   "projection": {"type": "orthographic"}},
}
