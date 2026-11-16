# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Overview: the study, stage x participant status, run / cancel, live progress.

Participants appear by pseudonym; "show study ids" adds the session ids for the operator
(local use only)."""

from __future__ import annotations

import re

from dash import Input, Output, State, dash_table, dcc, html, no_update

from biomech.pipeline import registry, stage_status
from biomech.viewer.components import short_path
from biomech.viewer.context import ViewerContext
from biomech.viewer.data import pid_label, unit_label
from biomech.viewer.runs import describe_event

STATUS_COLORS = {"up-to-date": "var(--ok)", "stale": "var(--warn)", "missing": "var(--muted)",
                 "failed": "var(--bad)"}


def status_table(ctx: ViewerContext, operator: bool = False) -> tuple[list[dict], list[dict]]:
    """Stage x unit status; units by pseudonym (study ids only with ``operator``), in
    natural participant order with the cohort unit last."""
    df = stage_status(ctx.config)
    wide = df.pivot(index="unit", columns="stage", values="status").fillna("")
    order = [s for s in registry() if s in wide.columns]
    wide = wide[order].reset_index()
    wide["unit"] = [unit_label(ctx.data, u, operator) for u in wide["unit"]]
    wide = wide.iloc[sorted(range(len(wide)), key=lambda i: _unit_key(wide["unit"][i]))]
    cols = [{"name": "participant", "id": "unit"}] + [{"name": s, "id": s} for s in order]
    return wide.to_dict("records"), cols


def _unit_key(label: str) -> tuple[int, str]:
    m = re.match(r"P(\d+)", label)
    return (int(m.group(1)), label) if m else (10 ** 9 + (label == "cohort"), label)


def layout(ctx: ViewerContext) -> html.Div:
    stages = list(registry())
    pids = [p.pid for p in ctx.cohort.included()]
    data, cols = status_table(ctx)
    cond = [{"if": {"filter_query": f'{{{c["id"]}}} = "{st}"', "column_id": c["id"]},
             "backgroundColor": col, "color": "var(--on-status)"}
            for c in cols[1:] for st, col in STATUS_COLORS.items()]
    return html.Div(className="page", children=[
        html.Div(className="card", children=[
            html.H3("Study"),
            html.Div(className="kv", children=[
                html.Span("study"), html.B(ctx.config.study),
                html.Span("config"), html.Code(short_path(ctx.config.source), className="path"),
                html.Span("config hash"), html.Code(ctx.config.hash),
                html.Span("analysis plan"), html.Code(ctx.plan_tag()),
                html.Span("participants"), html.B(
                    f"{len(pids)} included / {len(ctx.cohort.participants)} in manifest"),
                html.Span("results"), html.Code(short_path(ctx.config.out_dir), className="path"),
            ]),
        ]),
        html.Div(className="card", children=[
            html.H3("Run"),
            html.Div(className="row", children=[
                dcc.Dropdown(id="cc-stages", options=stages, multi=True,
                             placeholder="stages (default: all)", style={"minWidth": "320px"}),
                dcc.Dropdown(id="cc-pids", options=[{"label": pid_label(ctx.data, p), "value": p}
                                                for p in pids], multi=True,
                             placeholder="participants (default: all)",
                             style={"minWidth": "260px"}),
                dcc.Checklist(id="cc-force", options=[{"label": " force re-run",
                                                       "value": "force"}], value=[]),
                dcc.Input(id="cc-jobs", type="number", min=1, max=16, value=4,
                          style={"width": "60px"}),
                html.Button("Run", id="cc-run", className="btn primary"),
                html.Button("Cancel", id="cc-cancel", className="btn"),
            ]),
            html.Progress(id="cc-progress", value="0", max="1", style={"width": "100%"}),
            html.Div(id="cc-state", className="muted"),
            html.Pre(id="cc-log", className="log"),
            # Polls only while a run is in progress (Run enables it, the last tick stops it).
            dcc.Interval(id="cc-tick", interval=1000,
                         disabled=not ctx.runs.snapshot().running),
        ]),
        html.Div(className="card", children=[
            html.Div(className="row", children=[
                html.H3("Status (against the current configuration)"),
                html.Span(dcc.Checklist(id="cc-operator", value=[], inline=True,
                                        options=[{"label": " show study ids (operator)",
                                                  "value": "on"}]),
                          title="Study ids identify the recording sessions: local use only, "
                                "never in figures or exports")]),
            dash_table.DataTable(id="cc-status", data=data, columns=cols, style_data_conditional=cond,
                                 style_cell={"fontSize": "12px", "padding": "4px",
                                             "fontFamily": "var(--mono)",
                                             "backgroundColor": "var(--card)",
                                             "color": "var(--fg)", "border": "1px solid var(--line)"},
                                 style_header={"fontWeight": "600", "backgroundColor": "var(--bg)"},
                                 style_table={"overflowX": "auto"}),
        ]),
    ])


def register(app, ctx: ViewerContext) -> None:
    @app.callback(Output("cc-state", "children", allow_duplicate=True),
                  Output("cc-tick", "disabled", allow_duplicate=True),
                  Input("cc-run", "n_clicks"), State("cc-stages", "value"),
                  State("cc-pids", "value"), State("cc-force", "value"), State("cc-jobs", "value"),
                  prevent_initial_call=True)
    def _run(_, stages, pids, force, jobs):
        ok = ctx.runs.start(ctx.config, stages or None, pids or None, force="force" in (force or []),
                            n_jobs=int(jobs or 4))
        return ("started" if ok else "a run is already in progress"), False

    @app.callback(Output("cc-state", "children", allow_duplicate=True),
                  Input("cc-cancel", "n_clicks"), prevent_initial_call=True)
    def _cancel(_):
        ctx.runs.cancel()
        return "cancel requested (stops after the current unit)"

    @app.callback(Output("cc-progress", "value"), Output("cc-progress", "max"),
                  Output("cc-log", "children"), Output("cc-status", "data"),
                  Output("cc-tick", "disabled"),
                  Input("cc-tick", "n_intervals"), State("cc-status", "data"),
                  State("cc-operator", "value"))
    def _tick(_, current, operator):
        op = "on" in (operator or [])
        s = ctx.runs.snapshot()
        done, total = ctx.runs.progress()
        log = "\n".join(describe_event(e, lambda u: unit_label(ctx.data, u, op))
                        for e in s.events[-40:])
        if s.error:
            log += "\n" + s.error
        refresh = (s.finished is not None and not s.running
                   and getattr(_tick, "_last", None) != s.finished)
        if refresh:
            _tick._last = s.finished
            data, _ = status_table(ctx, op)
            return str(done), str(max(total, 1)), log, data, not s.running
        return (str(done), str(max(total, 1)), log,
                no_update if current else status_table(ctx, op)[0], not s.running)

    @app.callback(Output("cc-status", "data", allow_duplicate=True),
                  Input("cc-operator", "value"), prevent_initial_call=True)
    def _operator(value):
        return status_table(ctx, "on" in (value or []))[0]
