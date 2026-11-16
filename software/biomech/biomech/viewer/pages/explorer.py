# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Signals: synchronized signals of one phase (+ camera frames, 3D jaw, landmarks), the
data atlas (every registered ``data`` view for the chosen participant/phase) and, in the
validation study, both systems over one task (``viewer/validation_signals.py``, the
default tab there).

The phase selector takes several phases and "all motions": the 3-D jaw view and the
multi-phase atlas views (registry ``multi_phase``: orthographic, 3-D) overlay them, one
colour per task; the time series and the single-phase views show the first selected phase.

The channel selector lists the participant's own EMG channels (session names); an empty
selection shows no channel (with a note). The Atlas tab builds its view only while visible.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Callable
from urllib.parse import parse_qs, urlencode

from dash import Input, Output, Patch, State, dash_table, dcc, html, no_update

from biomech.viewer import data
from biomech.viewer import validation_signals as VS
from biomech.viewer.components import env_id, env_of, env_store, graph_box
from biomech.viewer.context import ViewerContext
from biomech.viewer.pages.analysis import (PERSP_LABEL, STAT_LABEL, _store_token,
                                           build_cached, render)
from biomech.viz.views import registry as R


def channel_options(names: list[str]) -> list[dict[str, str]]:
    return [{"label": f" {m.replace('_', ' ')} ", "value": m} for m in names]


MODES = ("raw", "clean", "envelope")


def layout(ctx: ViewerContext, search: str | None = None) -> html.Div:
    """``search`` (deep link): ``p`` = pseudonym (P3), ``phase`` = comma list, ``mode``."""
    pids = ctx.data.pids()
    q = {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}
    pid0 = data.pid_from_token(ctx.data, q.get("p")) or (pids[0] if pids else None)
    mode0 = q.get("mode") if q.get("mode") in MODES else "envelope"
    phase0 = [p for p in (q.get("phase") or "").split(",") if p] or None
    atlas = [{"label": v.title, "value": v.id} for v in R.views("data", ctx.config.groups)]
    chans = data.channel_names(ctx.data, pid0) if pid0 else []
    validation = is_validation(ctx)
    both = [
        html.Div(className="card controls", children=[
            dcc.RadioItems(id="ex-val-frames", value="aligned", inline=True, className="seg",
                           options=[{"label": f" {v} ", "value": k}
                                    for k, v in VS.FRAMES.items()]),
            html.Span("JTS-MM blue solid, Qualisys dashed; the stored validation results "
                      "(ref_kinematics, ref_fusion)", className="muted")]),
        html.Div(className="card", children=dcc.Loading(graph_box(
            "ex-val-graph", height_px=900, min_px=420)))]
    # Its components exist in every study (hidden outside the validation study), so the
    # callback never sees a missing input next to ex-pid.
    tabs = [dcc.Tab(label="Both systems", value="validation", children=both)] \
        if validation else []
    hidden = [] if validation else [html.Div(both, style={"display": "none"})]
    return html.Div(className="page", children=[
        env_store("signals"), *hidden,
        html.Div(className="card controls", children=[
            dcc.Dropdown(id="ex-pid", options=[{"label": data.pid_label(ctx.data, p),
                                                "value": p} for p in pids],
                         value=pid0, clearable=False, className="dd-mid"),
            dcc.Dropdown(id="ex-phase", multi=True, clearable=False, className="dd-wide",
                         value=phase0,
                         placeholder="phase(s); several or all motions overlay in 3-D"),
            html.Span(id="ex-phase-note", className="muted"),
        ]),
        dcc.Tabs(id="ex-tab", value="validation" if validation else "signals",
                 className="tabs", children=[*tabs,
            dcc.Tab(label="Signals", value="signals", children=[
                html.Div(className="card controls", children=[
                    dcc.RadioItems(id="ex-mode", value=mode0, inline=True, className="seg",
                                   options=[{"label": " raw ", "value": "raw"},
                                            {"label": " cleaned ", "value": "clean"},
                                            {"label": f" envelope ({data.EMG_NORM_LABEL}) ",
                                             "value": "envelope"}]),
                    dcc.Checklist(id="ex-ch", options=channel_options(chans), value=chans,
                                  inline=True),
                    dcc.Checklist(id="ex-unmasked", value=[], inline=True,
                                  options=[{"label": " show unmasked envelope (faint)",
                                            "value": "on"}]),
                    dcc.Checklist(id="ex-overlays", value=[], inline=True,
                                  options=[{"label": f" {lab} ", "value": k}
                                           for k, lab in data.OVERLAYS.items()]),
                ]),
                html.Div(className="split", id="ex-split", children=[
                    html.Div(className="card", children=dcc.Loading(graph_box(
                        "ex-graph", height_px=980, min_px=420, box_id="ex-graph-box"))),
                    html.Div(className="split-handle", title="drag to resize the columns; "
                             "double-click resets"),
                    html.Div(className="side", children=[
                        html.H4("Camera at cursor"),
                        html.Div("Click a trace to show the nearest preview frames.",
                                 className="muted", id="ex-cam-note"),
                        html.Div(id="ex-cam"),
                        html.H4("Jaw paths (maxillary frame)"),
                        dcc.RadioItems(id="ex-cam3d", value="oblique", inline=True,
                                       className="seg",
                                       options=[{"label": f" {k} ", "value": k}
                                                for k in data.CAMERAS]),
                        html.Div(id="ex-hint", className="muted"),
                        graph_box("ex-3d", aspect=1.0, min_px=260),
                        dcc.Store(id="ex-3d-cursor"),     # index of the cursor trace
                    ]),
                ]),
                html.Div(className="card", children=[
                    html.H4("Landmarks (6 teeth + 2 skin-palpated condyles)"),
                    dash_table.DataTable(id="ex-landmarks", style_table={"overflowX": "auto"},
                                         style_cell={"fontSize": "12px", "whiteSpace": "normal",
                                                     "textAlign": "left", "padding": "4px 8px",
                                                     "backgroundColor": "var(--card)",
                                                     "color": "var(--fg)", "border": "none",
                                                     "borderBottom": "1px solid var(--line)"},
                                         style_header={"fontWeight": "600",
                                                       "backgroundColor": "var(--card)",
                                                       "color": "var(--fg)"}),
                ]),
            ]),
            dcc.Tab(label="Atlas", value="atlas", children=[
                html.Div(className="card controls", children=[
                    dcc.Dropdown(id="ex-atlas", options=atlas,
                                 value=atlas[0]["value"] if atlas else None, clearable=False,
                                 className="dd-wide", placeholder="no data views registered"),
                    dcc.RadioItems(id="ex-persp", inline=True, className="seg"),
                    dcc.Dropdown(id="ex-stat", options=[{"label": v, "value": k}
                                                        for k, v in STAT_LABEL.items()],
                                 clearable=False, className="dd-mid"),
                ]),
                html.Div(className="card", children=dcc.Loading(html.Div(id="ex-atlas-body"))),
            ]),
        ]),
    ])


_FIGS: OrderedDict[str, Any] = OrderedDict()
_FIGS_LOCK = threading.Lock()
FIG_CACHE = 12


def cached(ctx: ViewerContext, key: tuple, build: Callable[[], Any]) -> Any:
    """A Signals figure, built once per (inputs, store state): switching back to a
    participant, task or mode re-sends the figure without rebuilding it."""
    k = json.dumps([key, _store_token(ctx)], default=str)
    with _FIGS_LOCK:
        if k in _FIGS:
            _FIGS.move_to_end(k)
            return _FIGS[k]
    fig = build()
    with _FIGS_LOCK:
        _FIGS[k] = fig
        while len(_FIGS) > FIG_CACHE:
            _FIGS.popitem(last=False)
    return fig


def is_validation(ctx: ViewerContext) -> bool:
    """The validation study (its own ``ref_*`` stages, no dataset stages)."""
    return "validation" in tuple(getattr(ctx.config, "groups", None) or ())


def signal_height(viewport: int | None) -> int:
    """Height of the signals figure: taller on large screens, never below ~110 px per row."""
    return int(min(max(0.9 * (viewport or 1280), 800), 1300))


def _names(ctx: ViewerContext, pid: str | None) -> list[str]:
    return data.phase_names(ctx.data, pid) if pid else []


def register(app, ctx: ViewerContext) -> None:
    @app.callback(Output("ex-phase", "options"), Output("ex-phase", "value"),
                  Input("ex-pid", "value"), State("ex-phase", "value"))
    def _phases(pid, current):
        if not pid:
            return [], []
        names = _names(ctx, pid)
        keep = [v for v in data.as_selection(current)
                if v in names or (v == data.ALL_MOTIONS and data.selected_phases(v, names))]
        value = keep or (["open_close"] if "open_close" in names else names[:1])
        return data.phase_options(names), value

    @app.callback(Output("ex-ch", "options"), Output("ex-ch", "value"),
                  Input("ex-pid", "value"), State("ex-ch", "value"), State("ex-ch", "options"))
    def _channels(pid, current, old_opts):
        return channels_for(ctx, pid, current, old_opts)

    @app.callback(Output("ex-graph", "figure"), Output("ex-hint", "children"),
                  Output("ex-phase-note", "children"),
                  Input("ex-pid", "value"), Input("ex-phase", "value"),
                  Input("ex-mode", "value"), Input("ex-ch", "value"),
                  Input("ex-unmasked", "value"), Input("ex-overlays", "value"),
                  Input(env_id("signals"), "data"))
    def _plot(pid, sel, mode, chans, unmasked, overlays, env):
        # No viewport input: the graph is responsive and its box carries the height
        # (_graph_height), so a resize does not rebuild the 7-row figure.
        names = _names(ctx, pid)
        phase = data.primary_phase(sel, names)
        if not pid or not phase or env is None:
            return no_update, no_update, no_update
        fig = cached(ctx, ("signals", pid, phase, sorted(chans or []), mode, env_of(env)[0],
                           "on" in (unmasked or []), sorted(overlays or [])),
                     lambda: data.explorer_figure(ctx.data, pid, phase, list(chans or []),
                                                  mode, env_of(env)[0],
                                                  unmasked="on" in (unmasked or []),
                                                  overlays=list(overlays or [])))
        phases3d = data.selected_phases(sel, names) or [phase]
        note = (f"time series: {phase} (the motion picked last); 3-D overlays "
                f"{len(phases3d)} phases" if len(phases3d) > 1 else "")
        return fig, data.phase_hint(phase), note

    @app.callback(Output("ex-3d", "figure"), Output("ex-3d-cursor", "data"),
                  Input("ex-pid", "value"), Input("ex-phase", "value"),
                  Input(env_id("signals"), "data"), Input("ex-cam3d", "value"))
    def _plot3d(pid, sel, env, cam3d):
        """The 3-D view on its own: a camera change does not rebuild the time series."""
        names = _names(ctx, pid)
        phase = data.primary_phase(sel, names)
        if not pid or not phase or env is None:
            return no_update, no_update
        phases3d = data.selected_phases(sel, names) or [phase]
        fig = data.jaw3d_figure(ctx.data, pid, phases3d if len(phases3d) > 1 else phase,
                                env_of(env)[0], view=cam3d or "oblique")
        return fig, len(fig.data) - 1

    @app.callback(Output("ex-3d", "figure", allow_duplicate=True),
                  Input("ex-graph", "hoverData"), Input("ex-graph", "clickData"),
                  State("ex-pid", "value"), State("ex-phase", "value"),
                  State("ex-3d-cursor", "data"), prevent_initial_call=True)
    def _cursor3d(hover, click, pid, sel, idx):
        """Cursor -> 3-D: the landmarks at the hovered (or clicked) time, patched into the
        last trace (no rebuild of the 3-D figure), so the 3-D path follows the spike line
        across the signal rows."""
        from dash import ctx as dctx

        trig = [t["prop_id"] for t in (dctx.triggered or [])]
        ev = hover if any(t.endswith("hoverData") for t in trig) else click
        phase = data.primary_phase(sel, _names(ctx, pid))
        if not ev or not ev.get("points") or not pid or not phase or idx is None:
            return no_update
        try:
            t_s = float(ev["points"][0]["x"])
        except (TypeError, ValueError, KeyError):
            return no_update
        x, y, z = data.cursor_points(ctx.data, pid, phase, t_s)
        p = Patch()
        p["data"][idx]["x"], p["data"][idx]["y"], p["data"][idx]["z"] = x, y, z
        return p

    @app.callback(Output("url", "search", allow_duplicate=True), Input("ex-pid", "value"),
                  Input("ex-phase", "value"), Input("ex-mode", "value"),
                  State("url", "pathname"), prevent_initial_call=True)
    def _url(pid, sel, mode, path):
        """Deep link of the explorer state (pseudonym only), no page reload."""
        if path not in ("/signals", "/explorer") or not pid:
            return no_update
        q = {"p": data.pid_token(ctx.data, pid), "mode": mode or "envelope"}
        phases = data.as_selection(sel)
        if phases:
            q["phase"] = ",".join(phases)
        return "?" + urlencode(q)

    @app.callback(Output("ex-graph-box", "style"), Input(env_id("signals"), "data"))
    def _graph_height(env):
        """The graph is responsive (it fills its box), so the box carries the height:
        0.9 x the window width, 800-1300 px; the user can still drag it."""
        return {"height": f"{signal_height(env_of(env)[1])}px", "minHeight": "420px"}

    @app.callback(Output("ex-unmasked", "options"), Input("ex-mode", "value"))
    def _unmasked_applies(mode):
        """The unmasked envelope only exists in envelope mode: disabled (with a reason)."""
        off = mode != "envelope"
        return [{"label": " show unmasked envelope (faint)" + (
            " — envelope mode only" if off else ""), "value": "on", "disabled": off}]

    @app.callback(Output("ex-landmarks", "data"), Output("ex-landmarks", "columns"),
                  Input("ex-pid", "value"))
    def _landmarks(pid):
        rows = data.landmark_rows(ctx.data, pid) if pid else []
        cols = [{"name": k, "id": k} for k in (rows[0] if rows else {})]
        return rows, cols

    @app.callback(Output("ex-cam", "children"), Output("ex-cam-note", "children"),
                  Input("ex-graph", "clickData"), State("ex-pid", "value"),
                  State("ex-phase", "value"))
    def _camera(click, pid, sel):
        phase = data.primary_phase(sel, _names(ctx, pid))
        if not click or not pid or not phase:
            return no_update, no_update
        x_s = float(click["points"][0]["x"])
        _, ph = ctx.data.phase(pid, phase)
        if ph is None:
            return no_update, no_update
        t_us = int(ph.t_start_us + x_s * 1e6)
        frames = data.preview_frames(ctx.data, pid, phase, t_us)
        if not frames:
            return [], f"no preview frame within 2 s of t = {x_s:.2f} s"
        return ([html.Figure([html.Img(src=src, style={"width": "100%"}),
                              html.Figcaption(cap)]) for cap, src in frames],
                f"t = {x_s:.2f} s from phase start")

    @app.callback(Output("ex-val-graph", "figure"), Input("ex-pid", "value"),
                  Input("ex-phase", "value"), Input("ex-val-frames", "value"),
                  Input(env_id("signals"), "data"), Input("ex-tab", "value"))
    def _validation(pid, sel, frames, env, tab):
        """Both systems over the first selected task (built only while the tab shows)."""
        if tab != "validation" or not pid or env is None:
            return no_update
        names = _names(ctx, pid)
        phase = data.primary_phase(sel, names)
        return cached(ctx, ("both", pid, phase, frames, env_of(env)[0]),
                      lambda: VS.validation_figure(ctx.data, pid, phase, frames or "aligned",
                                                   env_of(env)[0]))

    @app.callback(Output("ex-persp", "options"), Output("ex-persp", "value"),
                  Output("ex-stat", "value"), Output("ex-persp", "style"),
                  Output("ex-stat", "style"), Input("ex-atlas", "value"))
    def _atlas_controls(view):
        if not view:
            return [], None, None, {"display": "none"}, {"display": "none"}
        v = R.get(view)
        hide = {"display": "none"}
        return ([{"label": PERSP_LABEL.get(p, p), "value": p} for p in v.perspectives],
                v.defaults.get("perspective"), v.defaults.get("stat", "mean_sd"),
                {} if v.perspectives else hide, {} if "stat" in v.params else hide)

    @app.callback(Output("ex-atlas-body", "children"), Input("ex-atlas", "value"),
                  Input("ex-persp", "value"), Input("ex-stat", "value"),
                  Input("ex-pid", "value"), Input("ex-phase", "value"),
                  Input(env_id("signals"), "data"), Input("ex-tab", "value"))
    def _atlas(view, persp, stat, pid, sel, env, tab):
        return atlas_body(ctx, view, persp, stat, pid, sel, *env_of(env), tab)


def channels_for(ctx: ViewerContext, pid, current, old_opts):
    """Channel options of ``pid`` (its session's EMG names); the selection is kept where
    the channels exist, and a full selection stays full."""
    names = data.channel_names(ctx.data, pid)
    old = [o["value"] for o in (old_opts or [])]
    cur = list(current or [])
    if not old or set(old) <= set(cur):
        value = list(names)
    else:
        value = [n for n in names if n in cur]
    return channel_options(names), value


def atlas_body(ctx: ViewerContext, view, persp, stat, pid, sel, theme, viewport,
               tab: str | None = "atlas"):
    """The Atlas tab body; nothing is built while the tab is hidden."""
    if tab != "atlas":
        return no_update
    if not view:
        return html.P("No data views registered.", className="muted")
    try:
        v = R.get(view)
        names = _names(ctx, pid)
        phase = None
        if "phase" in v.params:
            phase = (data.view_phase(sel, names) if v.multi_phase
                     else data.primary_phase(sel, names))
        kw = {k: val for k, val in (("perspective", persp), ("stat", stat), ("pid", pid),
                                    ("phase", phase)) if val is not None}
        spec, err = build_cached(ctx, view, kw)
        if spec is None:
            return html.Pre(err, className="log")
        return render(spec, "ex-atlas-graph", theme or "light", viewport)
    except Exception as exc:
        return html.Pre(f"{type(exc).__name__}: {exc}", className="log")
