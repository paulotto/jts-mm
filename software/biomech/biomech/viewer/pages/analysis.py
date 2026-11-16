# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Results: every registered result view, grouped by domain, and every stored table (the
paper board is ``pages/paper.py``).

*Views* tab: pick a domain and a view (or search); one row of controls scopes it --
perspective, statistic, participant, phase, alignment and the view's own pickers, each
shown only when the view understands it. The selection lives in the URL
(``/results?view=emg_psd&pid=s03&phase=chewing``), so a link reopens the same figure; the
URL is written with ``dcc.Location(refresh="callback-nav")`` (history entry, no reload).
Built figures are cached per (view, parameters, store state): the newest run and whether
it finished (a run in progress invalidates as units complete), the store or snapshot read.
Export writes the view as TikZ (current zoom -> axis limits), PDF or PNG; exported ids
carry non-default parameters, so an explored perspective never overwrites a paper item.

*Tables* tab (computed only while visible): any stored table (stage, table,
participants), sortable and filterable, CSV download, and a quick strip plot of one
numeric column by participant/task/muscle. Participants appear as pseudonyms only (P<n>,
natural order; "excluded" for units outside the cohort), also in the CSV.
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from collections import OrderedDict
from typing import Any
from urllib.parse import parse_qs, urlencode

from dash import ALL, Input, Output, State, dash_table, dcc, html, no_update

from biomech.viewer.components import env_id, env_of, env_store, graph_box
from biomech.viewer.context import ViewerContext
from biomech.viewer.data import pid_label
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import apply_zoom, aspect_of, merge_relayout, to_plotly
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R
from biomech.viz.views.vocab import PERSP_LABEL, STAT_LABEL, TASK_LABEL, TASKS, reader_spec
from biomech.viz.views import vocab as V

# Results shows results (paper items, dashboards); the data views live in the explorer's
# Atlas tab, so no view appears on both pages.
KIND_LABEL = {"paper": "Paper", "dashboard": "Dashboard"}
RESULT_KINDS = tuple(KIND_LABEL)
ALIGN_LABEL = {"cycle": "cycle (0-100 %)", "angle": "opening angle", "time": "time",
               "event": "motion onset", "path": "path"}
HIDE = {"display": "none"}
_CACHE: OrderedDict[str, Any] = OrderedDict()
_CACHE_LOCK = threading.Lock()
CACHE_SIZE = 48
EXCLUDED = "excluded"


def _table(spec: TableSpec) -> html.Div:
    """A table spec in plain text: headers with their units, LaTeX markup stripped."""
    from biomech.viz import layout as L
    from biomech.viz.style import plain_unit

    def head(c) -> str:
        h = L.plain_text(c.header)
        u = plain_unit(c.unit) if c.unit else ""
        return f"{h} ({u})" if u else h

    def cell(v: Any, c) -> Any:
        if isinstance(v, float):
            return f"{v:.{c.decimals}f}" if v == v else "–"
        if isinstance(v, str):
            return L.plain_text(v)
        return v

    rows = [{c.key: cell(r.get(c.key), c) for c in spec.columns} for r in spec.rows
            if not r.get("_rule")]
    cellstyle = {"fontSize": "12.5px", "whiteSpace": "normal", "textAlign": "left",
                 "backgroundColor": "var(--card)", "color": "var(--fg)", "padding": "4px 8px",
                 "border": "none", "borderBottom": "1px solid var(--line)",
                 "fontVariantNumeric": "tabular-nums"}
    return html.Div([
        html.P(L.plain_text(spec.caption), className="caption-top"),
        dash_table.DataTable(data=rows, columns=[{"name": head(c), "id": c.key}
                                                 for c in spec.columns],
                             style_cell=cellstyle, style_table={"overflowX": "auto"},
                             style_header={**cellstyle, "fontWeight": "600",
                                           "borderBottom": "1px solid var(--axis)"}),
        html.P(L.plain_text(spec.notes), className="caption")])


def sources_drawer(spec: FigureSpec | TableSpec, store: str | None = None,
                   papers: list[str] | None = None) -> html.Details:
    """What a figure or table is drawn from: its stored tables (``stage:table``), the store
    read (snapshot or live) and the paper items it appears as."""
    rows = [html.Li(html.Code(s)) for s in (spec.sources or [])] or [
        html.Li("no stored source declared", className="muted")]
    extra = []
    if store:
        extra.append(html.P(["read from: ", html.Code(store)], className="caption"))
    if papers:
        extra.append(html.P("in the paper as: " + ", ".join(papers), className="caption"))
    return html.Details(className="sources", children=[
        html.Summary("Sources"), html.Ul(rows), *extra])


def render(spec: FigureSpec | TableSpec, gid: Any, theme: str, viewport: int | None = None,
           revision: str | None = None, drawer: html.Details | None = None) -> html.Div:
    """A table, or the figure laid out for the card width at the paper's aspect ratio
    (``to_plotly(fit=True)``) in a box that keeps that ratio and can be resized.

    ``revision`` (the view and its parameters): Plotly keeps the user's zoom across
    re-renders with the same revision (theme, resize) and resets it when it changes."""
    if isinstance(spec, TableSpec):
        return html.Div([_table(spec), *([drawer] if drawer else [])])
    fig = to_plotly(spec, theme=theme, viewport_px=viewport or 1280, fit=True)
    if revision is not None:
        fig.update_layout(uirevision=revision)
    return html.Div([graph_box(gid, fig, aspect=aspect_of(fig)),
                     html.P(spec.caption, className="caption"),
                     *([drawer] if drawer else [])])


def paper_uses(ctx: ViewerContext) -> dict[str, list[str]]:
    """view id -> the labels of the paper items drawing it (every active paper set)."""
    out: dict[str, list[str]] = {}
    for ps in getattr(ctx, "paper_sets", None) or []:
        for i in ps.items:
            out.setdefault(i.view_id, []).append(i.label)
    return out


def _store_token(ctx: ViewerContext) -> str:
    """Changes whenever the stored results can have changed: part of every cache key.

    The store read (live directory or a frozen snapshot), the newest run and its state --
    finished (``finished_utc`` of its manifest) or, while it runs, the size of its event log,
    which grows with every unit that completes (a figure built mid-run is rebuilt)."""
    cfg = ctx.config
    parts = [str(getattr(cfg, "read_dir", "")), str(getattr(cfg, "snapshot", None) or "live")]
    try:
        runs = ctx.store.runs()
    except OSError:
        runs = []
    if runs:
        rid = runs[0]
        man = ctx.store.run_manifest(rid)
        if man.get("finished_utc"):
            parts += [rid, str(man["finished_utc"])]
        else:
            ev = ctx.store.run_dir(rid) / "events.jsonl"
            try:
                size = ev.stat().st_size
            except OSError:
                size = 0
            parts += [rid, f"running:{size}"]
    return "|".join(parts)


def _params(view: str, persp=None, stat=None, pid=None, phase=None, align=None,
            choices: dict[str, Any] | None = None) -> dict[str, Any]:
    v = R.get(view)
    out: dict[str, Any] = {}
    if v.perspectives and persp:
        out["perspective"] = persp
    if "stat" in v.params and stat:
        out["stat"] = stat
    if "pid" in v.params and pid:
        out["pid"] = pid
    if "phase" in v.params and phase:
        out["phase"] = phase
    if "align" in v.params and align:
        out["align"] = align
    for k, val in (choices or {}).items():
        if k in v.choices and val is not None:
            out[k] = val
    return out


def build_cached(ctx: ViewerContext, view: str, kw: dict[str, Any]):
    """(spec, None) or (None, error text) of ``view`` with request parameters ``kw``;
    cached per (view, parameters, store state) -- the Results page and the paper board."""
    key = json.dumps([view, kw, _store_token(ctx)], sort_keys=True, default=str)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key], None
    try:
        # Reader wording, as in the export: no internal plan codes on screen either.
        spec = reader_spec(R.build(view, R.request(ctx.data, view, **kw)))
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
    with _CACHE_LOCK:
        _CACHE[key] = spec
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return spec, None


def _build(ctx: ViewerContext, view: str, persp=None, stat=None, pid=None, phase=None,
           align=None, choices: dict[str, Any] | None = None):
    """(spec, None) or (None, error text); cached per parameters and store state."""
    return build_cached(ctx, view, _params(view, persp, stat, pid, phase, align, choices))


def _error_card(err: str) -> html.Div:
    first, _, rest = err.partition("\n")
    return html.Div(className="error-card", children=[
        html.B("This view could not be built"), html.P(first),
        html.Details([html.Summary("details"), html.Pre(rest, className="log")])])


def view_options(domain: str | None = None, groups: Any = None,
                 in_paper: dict[str, list[str]] | None = None) -> list[dict[str, str]]:
    """Result views (``groups``: the study's view groups, :func:`registry.views`); views
    drawn by a paper item are marked "· paper" (``in_paper``: :func:`paper_uses`)."""
    from biomech.viz.views.registry import DOMAINS

    out = []
    for dom, vs in R.by_domain(RESULT_KINDS, groups).items():
        if domain and domain != "all" and dom != domain:
            continue
        for v in vs:
            mark = " · paper" if (in_paper or {}).get(v.id) else ""
            out.append({"label": f"{DOMAINS.get(dom, dom)} · {v.title}{mark}", "value": v.id})
    return out


def layout(ctx: ViewerContext, search: str | None = None) -> html.Div:
    from biomech.viz.views.registry import DOMAINS

    opts = view_options(groups=ctx.config.groups, in_paper=paper_uses(ctx))
    first = next((o["value"] for o in opts if o["value"].startswith("fig_")), opts[0]["value"]
                 if opts else None)
    pids = ctx.data.pids()
    q = {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}
    if q.get("view") in {o["value"] for o in opts}:
        first = q["view"]                       # deep link
    from biomech.viewer.data import pid_from_token

    pid0 = pid_from_token(ctx.data, q.get("p") or q.get("pid")) or (pids[0] if pids else None)
    doms = [{"label": "all", "value": "all"}] + [
        {"label": DOMAINS.get(d, d), "value": d}
        for d in R.by_domain(RESULT_KINDS, ctx.config.groups)]
    stages = sorted(ctx.store.stages())
    views_tab = html.Div([
        html.Div(className="card controls", children=[
            dcc.Dropdown(id="an-domain", options=doms, value="all", clearable=False,
                         className="dd-mid", placeholder="domain"),
            dcc.Dropdown(id="an-view", options=opts, value=first, clearable=False,
                         className="dd-wide", placeholder="search views"),
            dcc.RadioItems(id="an-persp", inline=True, className="seg"),
            dcc.Dropdown(id="an-stat", options=[{"label": v, "value": k}
                                                for k, v in STAT_LABEL.items()],
                         clearable=False, className="dd-mid"),
            dcc.Dropdown(id="an-pid", options=[{"label": pid_label(ctx.data, p), "value": p}
                                               for p in pids],
                         value=pid0, clearable=False, className="dd-mid"),
            dcc.Dropdown(id="an-phase", clearable=False, className="dd-mid",
                         placeholder="phase"),
            dcc.Dropdown(id="an-align", clearable=False, className="dd-mid",
                         placeholder="align"),
            html.Div(id="an-choices", className="controls"),
            html.Button("Export → LaTeX", id="an-export", className="btn primary"),
            html.Button("PDF", id="an-export-pdf", className="btn"),
            html.Button("PNG", id="an-export-png", className="btn"),
            dcc.Loading(html.Span(id="an-msg", className="muted"), type="dot"),
        ]),
        html.P(id="an-desc", className="caption"),
        dcc.Store(id="an-zoom"),
        html.Div(className="card", children=dcc.Loading(html.Div(id="an-body"))),
    ])
    tables_tab = html.Div([
        html.Div(className="card controls", children=[
            dcc.Dropdown(id="tb-stage", options=[{"label": s_, "value": s_} for s_ in stages],
                         value=stages[0] if stages else None, clearable=False,
                         className="dd-mid", placeholder="stage"),
            dcc.Dropdown(id="tb-table", clearable=False, className="dd-mid",
                         placeholder="table"),
            dcc.Dropdown(id="tb-pids", options=[{"label": pid_label(ctx.data, p), "value": p}
                                                for p in pids], multi=True,
                         placeholder="all participants", className="dd-wide"),
            html.Span(id="tb-info", className="muted"),
        ]),
        html.Div(className="card", children=dcc.Loading(html.Div(id="tb-body"))),
        html.Div(className="card controls", children=[
            html.Span("Quick plot", className="muted"),
            dcc.Dropdown(id="tb-col", clearable=False, className="dd-mid",
                         placeholder="numeric column"),
            dcc.Dropdown(id="tb-by", clearable=False, className="dd-mid",
                         placeholder="rows by"),
        ]),
        html.Div(className="card", children=html.Div(id="tb-plot")),
    ])
    return html.Div(className="page", children=[
        env_store("results"),
        dcc.Tabs(id="an-tabs", value="views", className="tabs", children=[
            dcc.Tab(label="Views", value="views", children=views_tab),
            dcc.Tab(label="Tables", value="tables", children=tables_tab)]),
    ])


def _choice_values(ids: list[dict], vals: list[Any]) -> dict[str, Any]:
    return {i["name"]: v for i, v in zip(ids or [], vals or [])}


def _typed(raw: str | None, vals: list[Any]) -> Any:
    """The picker value whose text is ``raw`` (URLs carry strings; options keep their type)."""
    if raw is None:
        return None
    return next((x for x in vals if str(x) == raw), None)


def _valid(value: Any, allowed: list[Any], default: Any = None) -> Any:
    """``value`` if allowed, else ``default`` if allowed, else the first allowed value."""
    if value in allowed:
        return value
    if default in allowed:
        return default
    return allowed[0] if allowed else None


def controls_for(ctx: ViewerContext, view: str, search: str | None) -> dict[str, Any]:
    """Control options and values of ``view``; a deep link (``search`` naming this view)
    restores perspective, statistic, phase, alignment and pickers -- each validated against
    the view; defaults that are not offered fall back to the first value."""
    v = R.get(view)
    q = {k: val[0] for k, val in parse_qs((search or "").lstrip("?")).items()} \
        if search and q_view(search) == view else {}
    persps = list(v.perspectives)
    phases = list(v.phases) or ([v.defaults["phase"]] if v.defaults.get("phase") else
                                list(TASKS))
    aligns = list(v.aligns)
    pickers = []
    for name in v.choices:
        vals = v.choice_values(name, ctx.data)
        if not vals:
            continue
        val = _valid(_typed(q.get(name), vals), vals, v.defaults.get(name, vals[0]))
        pickers.append((name, vals, val))
    stats = list(STAT_LABEL)
    return {
        "persp_opts": [{"label": PERSP_LABEL.get(p, p), "value": p} for p in persps],
        "persp": _valid(q.get("perspective"), persps, v.defaults.get("perspective")),
        "stat": _valid(q.get("stat") if "stat" in v.params else None, stats,
                       v.defaults.get("stat", "mean_sd")),
        "phase_opts": [{"label": TASK_LABEL.get(p, p), "value": p} for p in phases],
        "phase": _valid(q.get("phase"), phases, v.defaults.get("phase")),
        "align_opts": [{"label": ALIGN_LABEL.get(a, a), "value": a} for a in aligns],
        "align": _valid(q.get("align"), aligns, v.defaults.get("align")),
        "pickers": pickers, "view": v}


def register(app, ctx: ViewerContext) -> None:
    @app.callback(Output("an-view", "options"), Input("an-domain", "value"))
    def _domain(domain):
        return view_options(domain, ctx.config.groups, paper_uses(ctx))

    @app.callback(Output("an-persp", "options"), Output("an-persp", "value"),
                  Output("an-stat", "value"), Output("an-persp", "style"),
                  Output("an-stat", "style"), Output("an-pid", "style"),
                  Output("an-phase", "options"), Output("an-phase", "value"),
                  Output("an-phase", "style"), Output("an-align", "options"),
                  Output("an-align", "value"), Output("an-align", "style"),
                  Output("an-choices", "children"), Output("an-desc", "children"),
                  Input("an-view", "value"), State("url", "search"))
    def _controls(view, search):
        c = controls_for(ctx, view, search)
        v = c["view"]
        pickers = [html.Div(className="picker", children=[
            html.Span(V.picker_label(name), className="muted"),
            dcc.Dropdown(id={"type": "an-choice", "name": name}, clearable=False,
                         options=[{"label": V.choice_label(name, x), "value": x}
                                  for x in vals],
                         value=val, className="dd-mid")]) for name, vals, val in c["pickers"]]
        return (c["persp_opts"], c["persp"], c["stat"],
                {} if v.perspectives else HIDE, {} if "stat" in v.params else HIDE,
                {} if "pid" in v.params else HIDE,
                c["phase_opts"], c["phase"],
                {} if "phase" in v.params and c["phase_opts"] else HIDE,
                c["align_opts"], c["align"],
                {} if "align" in v.params and c["align_opts"] else HIDE, pickers,
                v.description)

    @app.callback(Output("an-body", "children"), Output("url", "search"),
                  Input("an-view", "value"), Input("an-persp", "value"),
                  Input("an-stat", "value"), Input("an-pid", "value"),
                  Input("an-phase", "value"), Input("an-align", "value"),
                  Input({"type": "an-choice", "name": ALL}, "value"),
                  Input(env_id("results"), "data"),
                  State({"type": "an-choice", "name": ALL}, "id"), State("url", "pathname"))
    def _show(view, persp, stat, pid, phase, align, cvals, env, cids, path):
        from dash import ctx as dctx

        if env is None:                 # the page's environment arrives right after it
            return no_update, no_update
        theme, viewport = env_of(env)
        return show(ctx, view, persp, stat, pid, phase, align, cvals, theme, viewport, cids,
                    path, initial=dctx.triggered_id is None)

    @app.callback(Output("an-zoom", "data"), Input("an-graph", "relayoutData"),
                  State("an-zoom", "data"), prevent_initial_call=True)
    def _zoom(relayout, current):
        return merge_relayout(current, relayout)

    @app.callback(Output("an-zoom", "data", allow_duplicate=True), Input("an-view", "value"),
                  Input("an-persp", "value"), Input("an-pid", "value"),
                  Input("an-phase", "value"), Input("an-align", "value"),
                  Input("an-stat", "value"),
                  Input({"type": "an-choice", "name": ALL}, "value"),
                  prevent_initial_call=True)
    def _reset_zoom(*_):
        return {}

    @app.callback(Output("an-msg", "children"), Input("an-export", "n_clicks"),
                  Input("an-export-pdf", "n_clicks"), Input("an-export-png", "n_clicks"),
                  State("an-view", "value"), State("an-persp", "value"),
                  State("an-stat", "value"), State("an-pid", "value"),
                  State("an-phase", "value"), State("an-align", "value"),
                  State({"type": "an-choice", "name": ALL}, "value"),
                  State({"type": "an-choice", "name": ALL}, "id"),
                  State("an-zoom", "data"), prevent_initial_call=True)
    def _export(n_tex, n_pdf, n_png, view, persp, stat, pid, phase, align, cvals, cids, zoom):
        from dash import ctx as dctx

        spec, err = _build(ctx, view, persp, stat, pid, phase, align,
                           _choice_values(cids, cvals))
        if spec is None:
            return f"failed: {err.splitlines()[0]}"
        if isinstance(spec, FigureSpec):
            spec = apply_zoom(spec, zoom)
        # Single views go to their own folder: a paper folder holds exactly its paper set
        # (manifest-checked).
        dest = ctx.config.out_dir / "viewer_exports"
        if dctx.triggered_id in ("an-export-pdf", "an-export-png") and \
                isinstance(spec, FigureSpec):
            ext = "pdf" if dctx.triggered_id == "an-export-pdf" else "png"
            p = render_mpl.render_figure(spec, dest / "previews" / f"{spec.id}.{ext}")
            return f"written {p}"
        render_pgf.write_style(dest)
        p = (render_pgf.render_figure(spec, dest) if isinstance(spec, FigureSpec)
             else render_pgf.render_table(spec, dest))
        return f"written {p}"

    _register_tables(app, ctx)


def show(ctx: ViewerContext, view, persp, stat, pid, phase, align, cvals, theme, viewport,
         cids, path, *, initial: bool = False):
    """Body and URL query of the Results view (the ``_show`` callback, testable)."""
    choices = _choice_values(cids, cvals)
    # The first render must not overwrite a deep link before _controls applied it.
    if initial:
        path = None
    spec, err = _build(ctx, view, persp, stat, pid, phase, align, choices)
    kw = _params(view, persp, stat, pid, phase, align, choices)
    from biomech.viewer.data import pid_token

    q = {k: v for k, v in kw.items() if isinstance(v, (str, int, float))}
    if "pid" in q:                              # the URL carries the pseudonym only
        q["p"] = pid_token(ctx.data, q.pop("pid"))
    search = "?" + urlencode({"view": view, **q})
    url = search if path in ("/results", "/analysis") else no_update
    if spec is None:
        return _error_card(err), url
    cfg = ctx.config
    store = f"snapshot {cfg.snapshot}" if getattr(cfg, "snapshot", None) else "live store"
    drawer = sources_drawer(spec, store, paper_uses(ctx).get(view))
    return (render(spec, "an-graph", theme or "light", viewport,
                   revision=json.dumps([view, kw], sort_keys=True, default=str),
                   drawer=drawer), url)


def q_view(search: str | None) -> str | None:
    q = parse_qs((search or "").lstrip("?"))
    return q.get("view", [None])[0]


# -- table browser -------------------------------------------------------------------------
def _pseudonym_key(label: str) -> tuple[int, str]:
    """Natural order of participant labels: P2 < P10; 'excluded' last."""
    m = re.fullmatch(r"P(\d+)", str(label))
    return (int(m.group(1)), "") if m else (10 ** 9, str(label))


def _browse(ctx: ViewerContext, stage: str, table: str, pids: list[str] | None):
    """A stored table for the browser: pseudonyms only (``P<n>``; units outside the cohort
    become "excluded"), rows in natural participant order."""
    from biomech.pipeline import get_stage

    try:
        cohort = get_stage(stage).scope == "cohort"
    except KeyError:
        cohort = False
    df = ctx.data.table(stage, table, None if cohort else (pids or None))
    df = df.drop(columns=[c for c in ("unit",) if c in df.columns])
    if "pid" in df.columns:
        names = ctx.data.pseudonyms()
        lab = [f"P{names[p]}" if p in names else EXCLUDED for p in df["pid"]]
        order = sorted(range(len(lab)), key=lambda i: (_pseudonym_key(lab[i]), i))
        df = df.assign(pid=lab).iloc[order].reset_index(drop=True)
    return df


def _register_tables(app, ctx: ViewerContext) -> None:
    @app.callback(Output("tb-table", "options"), Output("tb-table", "value"),
                  Input("tb-stage", "value"))
    def _tables(stage):
        from biomech.results import COHORT_UNIT

        if not stage:
            return [], None
        names: set[str] = set()
        for unit in ctx.store.units(stage):
            m = ctx.store.meta(stage, unit)
            if m is not None and m.status == "ok":
                names.update(m.tables)
            if unit == COHORT_UNIT:
                break
        opts = [{"label": n, "value": n} for n in sorted(names)]
        return opts, (opts[0]["value"] if opts else None)

    @app.callback(Output("tb-body", "children"), Output("tb-info", "children"),
                  Output("tb-col", "options"), Output("tb-col", "value"),
                  Output("tb-by", "options"), Output("tb-by", "value"),
                  Input("tb-stage", "value"), Input("tb-table", "value"),
                  Input("tb-pids", "value"), Input("an-tabs", "value"))
    def _grid(stage, table, pids, tab):
        return grid(ctx, stage, table, pids, tab)

    @app.callback(Output("tb-plot", "children"), Input("tb-col", "value"),
                  Input("tb-by", "value"), State("tb-stage", "value"),
                  State("tb-table", "value"), State("tb-pids", "value"),
                  Input(env_id("results"), "data"), Input("an-tabs", "value"))
    def _plot(col, by, stage, table, pids, env, tab):
        return quick_plot(ctx, col, by, stage, table, pids, *env_of(env), tab)


def grid(ctx: ViewerContext, stage, table, pids, tab: str | None = "tables"):
    """The table browser body (the ``_grid`` callback); nothing is read while the Tables
    tab is hidden."""
    import pandas as pd

    if tab != "tables":
        return (no_update,) * 6
    if not stage or not table:
        return html.P("no table", className="muted"), "", [], None, [], None
    try:
        df = _browse(ctx, stage, table, pids)
        num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
        cat = [c for c in df.columns if c not in num and df[c].nunique() <= 60]
        shown = df.head(5000)
        cols = [{"name": c, "id": c, "type": "numeric" if c in num else "text",
                 "format": {"specifier": ".4~g"} if c in num else None} for c in shown.columns]
        cols = [{k: v for k, v in c.items() if v is not None} for c in cols]
        cell = {"fontSize": "12px", "whiteSpace": "normal", "textAlign": "left",
                "backgroundColor": "var(--card)", "color": "var(--fg)", "padding": "3px 6px",
                "border": "none", "borderBottom": "1px solid var(--line)",
                "fontVariantNumeric": "tabular-nums", "maxWidth": "260px"}
        body = dash_table.DataTable(
            data=shown.to_dict("records"), columns=cols, sort_action="native",
            filter_action="native", page_size=25, export_format="csv",
            style_cell=cell, style_table={"overflowX": "auto"},
            style_header={**cell, "fontWeight": "600", "borderBottom": "1px solid var(--axis)"},
            style_filter={**cell, "backgroundColor": "var(--bg)"})
        info = f"{len(df)} rows x {len(df.columns)} columns" + (
            " (first 5000 shown)" if len(df) > 5000 else "")
        pref = ("pid", "task", "muscle", "model", "feature_set", "phase", "test")
        by = [c for c in pref if c in cat] + [c for c in cat if c not in pref]
    except Exception as exc:
        return _error_card(f"{type(exc).__name__}: {exc}"), "", [], None, [], None
    return (body, info, [{"label": c, "value": c} for c in num], num[0] if num else None,
            [{"label": c, "value": c} for c in by], by[0] if by else None)


def quick_plot(ctx: ViewerContext, col, by, stage, table, pids, theme, viewport,
               tab: str | None = "tables"):
    """Strip plot of one numeric column by a grouping (pseudonyms, natural order)."""
    from biomech.viz.views import blocks as B
    from biomech.viz.views import vocab as V

    if tab != "tables":
        return no_update
    if not (col and by and stage and table):
        return html.P("choose a numeric column and a grouping", className="muted")
    try:
        df = _browse(ctx, stage, table, pids)
        order = (sorted(df[by].dropna().unique(), key=_pseudonym_key) if by == "pid"
                 else None)
        p = B.strip_by(df, col, by, f"{V.label(col)} by {by}", xaxis=V.axis(col), order=order)
        spec = FigureSpec(f"browse--{stage}--{table}--{col}", f"{stage}: {table}", [p],
                          width="column",
                          height_in=max(1.4, 0.2 * df[by].nunique()),
                          caption=f"{col} of {stage}:{table}, one dot per row, "
                                  "diamond = median.")
        return render(spec, "tb-graph", theme or "light", viewport)
    except Exception as exc:
        return _error_card(f"{type(exc).__name__}: {exc}")
