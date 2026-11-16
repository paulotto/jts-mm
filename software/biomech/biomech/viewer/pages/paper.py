# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Paper: the study's paper set (``paper_*.yaml``) as a storyboard.

*Board*: the items of one target in storyboard order under their section headers
(``{section: ...}`` entries of the paper set), each card titled with its parameters
(participant pseudonym, task, frames, depiction) and badged with its part (core /
supporting / supplement) and LaTeX label. Only the chosen section is built (lazy; "all
sections" builds everything), through the Results page's figure cache.

*Gallery*: every depiction an item offers (its ``form`` picker) side by side. "Use this"
writes the choice into the paper set (``papers.set_item_param``: one line, comments kept,
re-read and verified) after a confirmation; the export then draws that form.

The long exports (the paper set with manifest and numbers; every view) run as background
jobs with progress and cancel, one at a time; the timer polls only while one runs.
"""

from __future__ import annotations

from typing import Any

from dash import ALL, Input, Output, State, dcc, html, no_update

from biomech.viewer.components import env_id, env_of, env_store, short_path
from biomech.viewer.context import ViewerContext
from biomech.viewer.pages.analysis import _error_card, build_cached, render, sources_drawer
from biomech.viz.export import export
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

ALL_SECTIONS = "__all__"
PART_LABEL = {"core": "core", "supporting": "supporting", "supplement": "supplement"}
MODES = {"board": "Board", "gallery": "Gallery (choose depictions)"}


def _store(ctx: ViewerContext) -> str:
    snap = getattr(ctx.config, "snapshot", None)
    return f"snapshot {snap}" if snap else "live store"


def _pset(ctx: ViewerContext, name: str | None = None) -> Any:
    sets = getattr(ctx, "paper_sets", None) or []
    return next((p for p in sets if p.name == name), sets[0] if sets else None)


def item_title(ctx: ViewerContext, view_id: str, params: dict[str, Any]) -> str:
    """The view's title and its parameters in reader words ("P1", "open-close", ...)."""
    try:
        title = R.get(view_id).title
    except KeyError:
        return view_id
    bits = []
    for k, v in params.items():
        if k == "pid":
            bits.append(f"P{ctx.data.pseudonym(v)}" if ctx.data is not None else "P?")
        elif k == "phase":
            bits.append(V.term("task", v))
        else:
            bits.append(V.choice_label(k, v))
    return title + (f" · {', '.join(bits)}" if bits else "")


def layout(ctx: ViewerContext) -> html.Div:
    ps = _pset(ctx)
    running = ctx.exports.snapshot().running
    sets = getattr(ctx, "paper_sets", None) or []
    tgt = ps.target() if ps else None
    secs = ps.sections(tgt.name) if ps else []
    choose = html.Div(className="card controls", children=[
        dcc.Dropdown(id="pb-set", options=[{"label": p.path.name, "value": p.name}
                                           for p in sets],
                     value=ps.name if ps else None, clearable=False, className="dd-mid",
                     style={} if len(sets) > 1 else {"display": "none"}),
        dcc.Dropdown(id="pb-target", options=[{"label": f"target: {t}", "value": t}
                                              for t in (ps.targets if ps else [])],
                     value=tgt.name if tgt else None, clearable=False, className="dd-mid",
                     style={} if ps and len(ps.targets) > 1 else {"display": "none"}),
        dcc.Dropdown(id="pb-section", options=_section_options(secs),
                     value=secs[0] if secs else ALL_SECTIONS, clearable=False,
                     className="dd-wide"),
        dcc.Checklist(id="pb-parts", value=list(PART_LABEL), inline=True,
                      options=[{"label": f" {v} ", "value": k} for k, v in PART_LABEL.items()]),
        dcc.RadioItems(id="pb-mode", value="board", inline=True, className="seg",
                       options=[{"label": f" {v} ", "value": k} for k, v in MODES.items()]),
        html.Span(id="pb-set-msg", className="muted"),
    ]) if ps else html.Div(className="card", children=html.P(
        "This study has no paper set (paper_*.yaml next to its study file): the board shows "
        "every paper view.", className="muted"))
    return html.Div(className="page", children=[
        env_store("paper"),
        choose,
        html.Div(className="card controls", children=[
            html.Button("Export the paper set", id="pb-export", className="btn primary",
                        disabled=running,
                        title="Items, manifest, captions.tex and numbers.tex into the "
                              "target folder"),
            html.Button("Export every view", id="pb-export-every", className="btn",
                        disabled=running,
                        title="All paper, dashboard and data views, every perspective and "
                              "participant: TikZ + PNG/PDF previews in <results>/export_all"),
            html.Button("Cancel export", id="pb-cancel", className="btn"),
            dcc.Checklist(id="pb-compile", options=[{"label": " compile preview PDF",
                                                     "value": "c"},
                                                    {"label": " check sizes", "value": "s"}],
                          value=["c"], inline=True),
            html.Code(short_path(tgt.out_dir if tgt else ctx.export_dir), id="pb-dir",
                      className="path"),
            html.Progress(id="pb-progress", value="0", max="1", style={"width": "200px"}),
            html.Span(id="pb-msg", className="muted"),
            dcc.Interval(id="pb-tick", interval=1000, disabled=not running),
        ]),
        html.Pre(id="pb-every-msg", className="log", style={"maxHeight": "160px"}),
        dcc.Loading(html.Div(id="pb-body")),
    ])


def _section_options(secs: list[str]) -> list[dict[str, str]]:
    return ([{"label": s or "(no section)", "value": s} for s in secs] +
            [{"label": "all sections", "value": ALL_SECTIONS}])


def _items(ctx: ViewerContext, set_name: str | None, target: str | None,
           section: str | None, parts: list[str] | None) -> list[Any]:
    ps = _pset(ctx, set_name)
    if ps is None:
        return []
    items = ps.items_for(target)
    keep = set(parts or [])
    return [i for i in items if i.part in keep and
            (section in (None, ALL_SECTIONS) or i.section == section)]


def _badges(item: Any) -> html.Span:
    return html.Span(className="badges", children=[
        html.Span(PART_LABEL.get(item.part, item.part), className=f"badge part-{item.part}"),
        html.Code(f"\\figcaption{{{item.label}}}", className="badge",
                  title="LaTeX name of the item (captions.tex, the export folder)")])


def board_blocks(ctx: ViewerContext, items: list[Any], theme: str,
                 viewport: int | None = None) -> list[Any]:
    """Section headers and one card per item (built through the figure cache); graph ids
    carry the item's index in the paper set (the same view may appear twice)."""
    out: list[Any] = []
    section = None
    for item in items:
        if item.section != section:
            section = item.section
            out.append(html.H2(section or "(no section)", className="section-head"))
        try:
            spec, err = build_cached(ctx, item.view_id, item.params)
            body = (render(spec, {"type": "pb-graph", "index": item.index}, theme, viewport,
                           drawer=sources_drawer(spec, _store(ctx)))
                    if spec is not None else _error_card(err))
        except Exception as exc:
            body = html.Pre(f"{type(exc).__name__}: {exc}", className="log")
        out.append(html.Div(className="card", children=[
            html.Div(className="card-head", children=[
                html.H3(item_title(ctx, item.view_id, item.params)), _badges(item)]),
            body]))
    return out


def legacy_blocks(ctx: ViewerContext, theme: str, viewport: int | None = None) -> list[Any]:
    """Without a paper set: every paper view of the study, one card each."""
    from biomech.viz.export import _item

    out = []
    for i, entry in enumerate(ctx.paper_items or
                              [v.id for v in R.views("paper", ctx.config.groups)]):
        vid, params = _item(entry)
        spec, err = build_cached(ctx, vid, params)
        body = (render(spec, {"type": "pb-graph", "index": i}, theme, viewport)
                if spec is not None else _error_card(err))
        out.append(html.Div(className="card", children=[
            html.H3(item_title(ctx, vid, params)), body]))
    return out


def gallery_blocks(ctx: ViewerContext, items: list[Any], theme: str,
                   viewport: int | None = None) -> list[Any]:
    """Every depiction of each item side by side; the one in the paper set is marked, the
    others carry "Use this" (with a confirmation)."""
    out: list[Any] = []
    section = None
    narrow = int((viewport or 1280) / 2)          # two depictions per row
    for item in items:
        if item.section != section:
            section = item.section
            out.append(html.H2(section or "(no section)", className="section-head"))
        v = R.get(item.view_id)
        forms = v.choice_values("form", ctx.data) if "form" in v.choices else []
        head = html.Div(className="card-head", children=[
            html.H3(item_title(ctx, item.view_id,
                               {k: x for k, x in item.params.items() if k != "form"})),
            _badges(item)])
        if not forms:
            out.append(html.Div(className="card", children=[
                head, html.P("One depiction only (no depiction choice for this item).",
                             className="muted")]))
            continue
        current = item.params.get("form", v.defaults.get("form", forms[0]))
        cells = []
        for f in forms:
            spec, err = build_cached(ctx, item.view_id, {**item.params, "form": f})
            body = (render(spec, {"type": "pb-gal", "index": item.index, "form": str(f)},
                           theme, narrow) if spec is not None else _error_card(err))
            chosen = f == current
            action = (html.Span("in the paper set", className="badge chosen") if chosen else
                      dcc.ConfirmDialogProvider(
                          html.Button("Use this", className="btn"),
                          id={"type": "pb-use", "index": item.index, "form": str(f)},
                          message=f"Write 'form: {f}' for {item.label} into "
                                  f"{_pset(ctx).path.name}? The next export draws it."))
            cells.append(html.Div(className="gallery-cell" + (" chosen" if chosen else ""),
                                  children=[html.Div(className="card-head", children=[
                                      html.B(V.choice_label("form", f)), action]), body]))
        out.append(html.Div(className="card", children=[
            head, html.Div(className="gallery", children=cells)]))
    return out


def choose_form(ctx: ViewerContext, set_name: str | None, index: int, form: str) -> str:
    """Write ``form`` for item ``index`` into the paper set and reload it (the board and
    exports read the new choice); returns the message shown."""
    from biomech import papers

    ps = _pset(ctx, set_name)
    item = next(i for i in ps.items if i.index == index)
    v = R.get(item.view_id)
    allowed = v.choice_values("form", ctx.data) if "form" in v.choices else []
    value = next((x for x in allowed if str(x) == str(form)), None)
    if value is None:
        raise ValueError(f"{item.view_id} has no depiction {form!r}")
    new = papers.set_item_param(ps.path, index, "form", value)
    ctx.paper_sets = [new if p.name == ps.name else p for p in ctx.paper_sets]
    return f"{item.label}: form {value} written to {ps.path.name}"


# -- exports ---------------------------------------------------------------------------------
def start_export(ctx: ViewerContext, which: str, opts: list[str] | None = None,
                 set_name: str | None = None, target: str | None = None) -> bool:
    """Start the paper-set export (``which="paper"``) or the every-view export
    (``"every"``) as a background job; False when an export is already running."""
    opts = list(opts or [])
    groups = getattr(ctx.config, "groups", None)        # the study's view groups

    def work(progress, cancelled) -> str:
        if which == "every":
            from biomech.viz.export import all_items

            dest = ctx.config.out_dir / "export_all"
            items = all_items(ctx.data, groups=groups)
            res = export(ctx.data, items, dest, on_progress=progress, cancel=cancelled,
                         groups=groups)
            msg = (f"{len(items)} items: {len(res['figures'])} figures, {len(res['tables'])} "
                   f"tables -> {dest} (previews/*.png, *.pdf)")
            if res["errors"]:
                msg += f"\n{len(res['errors'])} error(s):\n" + "\n".join(res["errors"][:10])
            return msg
        ps = _pset(ctx, set_name)
        if ps is not None:                 # the study's paper set: items, manifest, numbers
            from biomech.viz.export import export_paper_set

            dest = ps.target(target).out_dir
            res = export_paper_set(ctx.data, ps, target, compile_pdf="c" in opts,
                                   check_size="s" in opts, on_progress=progress,
                                   cancel=cancelled, groups=groups)
        else:
            dest = ctx.export_dir
            res = export(ctx.data, ctx.paper_items or None, dest,
                         compile_pdf="c" in opts, check_size="s" in opts,
                         on_progress=progress, cancel=cancelled,
                         venue=getattr(ctx, "venue", None) or "ieee", groups=groups)
        msg = f"{len(res['figures'])} figures, {len(res['tables'])} tables -> {dest}"
        if res["errors"]:
            msg += f"; {len(res['errors'])} error(s): {res['errors'][0][:200]}"
        elif "c" in opts and not res["cancelled"]:
            msg += " (preview.pdf compiled)"
        if res["problems"]:
            msg += f"; {len(res['problems'])} layout problem(s): {res['problems'][0][:160]}"
        return msg

    name = "export of every view" if which == "every" else "paper export"
    return ctx.exports.start(name, work)


def register(app, ctx: ViewerContext) -> None:
    @app.callback(Output("pb-section", "options"), Output("pb-section", "value"),
                  Output("pb-dir", "children"), Input("pb-set", "value"),
                  Input("pb-target", "value"), State("pb-section", "value"))
    def _sections(set_name, target, current):
        ps = _pset(ctx, set_name)
        if ps is None:
            return [], ALL_SECTIONS, short_path(ctx.export_dir)
        secs = ps.sections(target)
        keep = current if current in secs or current == ALL_SECTIONS else \
            (secs[0] if secs else ALL_SECTIONS)
        return _section_options(secs), keep, short_path(ps.target(target).out_dir)

    @app.callback(Output("pb-body", "children"), Input(env_id("paper"), "data"),
                  Input("pb-set", "value"), Input("pb-target", "value"),
                  Input("pb-section", "value"), Input("pb-parts", "value"),
                  Input("pb-mode", "value"))
    def _body(env, set_name, target, section, parts, mode):
        if env is None:
            return no_update
        theme, viewport = env_of(env)
        if _pset(ctx, set_name) is None:
            return legacy_blocks(ctx, theme, viewport)
        items = _items(ctx, set_name, target, section, parts)
        if not items:
            return html.P("No item in this selection.", className="muted")
        fn = gallery_blocks if mode == "gallery" else board_blocks
        return fn(ctx, items, theme, viewport)

    @app.callback(Output("pb-set-msg", "children"),
                  Output("pb-mode", "value", allow_duplicate=True),
                  Input({"type": "pb-use", "index": ALL, "form": ALL}, "submit_n_clicks"),
                  State("pb-set", "value"), prevent_initial_call=True)
    def _use(clicks, set_name):
        from dash import ctx as dctx

        t = dctx.triggered_id
        if not t or not any(clicks or []):
            return no_update, no_update
        try:
            msg = choose_form(ctx, set_name, int(t["index"]), t["form"])
        except Exception as exc:
            return f"not written: {type(exc).__name__}: {exc}", no_update
        return msg, "gallery"               # re-renders the gallery with the new choice

    @app.callback(Output("pb-msg", "children", allow_duplicate=True),
                  Output("pb-tick", "disabled", allow_duplicate=True),
                  Input("pb-export", "n_clicks"), Input("pb-export-every", "n_clicks"),
                  State("pb-compile", "value"), State("pb-set", "value"),
                  State("pb-target", "value"), prevent_initial_call=True)
    def _start(n_paper, n_every, opts, set_name, target):
        from dash import ctx as dctx

        which = "every" if dctx.triggered_id == "pb-export-every" else "paper"
        if not start_export(ctx, which, opts, set_name, target):
            return "an export is already running (cancel it first)", False
        return "export started", False

    @app.callback(Output("pb-msg", "children", allow_duplicate=True),
                  Input("pb-cancel", "n_clicks"), prevent_initial_call=True)
    def _cancel(_):
        if not ctx.exports.snapshot().running:
            return "no export running"
        ctx.exports.cancel()
        return "cancel requested (stops after the current item)"

    @app.callback(Output("pb-progress", "value"), Output("pb-progress", "max"),
                  Output("pb-msg", "children"), Output("pb-every-msg", "children"),
                  Output("pb-export", "disabled"), Output("pb-export-every", "disabled"),
                  Output("pb-tick", "disabled"), Input("pb-tick", "n_intervals"))
    def _tick(_):
        d = ctx.exports.describe()
        if not d["text"]:
            return "0", "1", no_update, no_update, False, False, True
        head, _, rest = d["text"].partition("\n")
        return (str(d["value"]), str(d["max"]), head, rest or no_update, d["running"],
                d["running"], not d["running"])
