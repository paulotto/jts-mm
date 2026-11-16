# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""biomech analysis viewer: ``biomech view --config experiments/<study>/study.yaml``.

Pages: Overview (runs and stage status), Signals (synchronized recordings + the data
atlas), Results (every result view by domain, with perspectives and pickers, plus a browser
for every stored table), Paper (the study's paper set), Videos (preview and render, the same
path as ``biomech animate``). All analysis runs
through ``biomech.pipeline.run``; all figures are registered ``viz`` views, so
"Export -> LaTeX" writes exactly what the paper build writes.

The theme follows the system (``prefers-color-scheme``, also when the OS switches while the
page is open: a ``matchMedia`` listener in ``assets/resize.js``) unless overridden in the
navigation bar; figures are re-rendered with the matching Plotly template.

URL: ``dcc.Location(refresh="callback-nav")`` -- pages are routed on the pathname; the
Results page writes its selection into the query string from a callback, which adds a
history entry without reloading the page.

Sizing: ``assets/resize.js`` re-lays out every Plotly graph whose container changes size
(ResizeObserver, also on graphs added later), drives the explorer's column divider and
pushes the window width into the ``viewport`` store (debounced) so the figures are laid
out again for the new width (``render_plotly.to_plotly(fit=True)``, stacked on phones).

The werkzeug access log (one line per request) is off unless ``verbose``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from dash import ALL, ClientsideFunction, Dash, Input, Output, State, dcc, html  # noqa: F401

from biomech.viewer.context import ViewerContext
from biomech.viewer.pages import analysis, animation, control, explorer, paper

# Navigation: Overview (runs and status) · Signals (synchronized recordings and
# the data atlas) · Results · Paper (the study's paper set) · Videos.
PAGES = {"/": ("Overview", control.layout), "/signals": ("Signals", explorer.layout),
         "/results": ("Results", analysis.layout), "/paper": ("Paper", paper.layout),
         "/videos": ("Videos", animation.layout)}
ALIASES = {"/analysis": "/results", "/explorer": "/signals",     # old links keep working
           "/animation": "/videos"}
_THEME_JS = """
function(choice) {
    var dark = choice === 'dark' ||
        (choice !== 'light' && window.matchMedia &&
         window.matchMedia('(prefers-color-scheme: dark)').matches);
    var t = dark ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
    window.biomechViewer = window.biomechViewer || {};
    window.biomechViewer.themeChoice = choice;     // assets/resize.js: follow the OS on 'auto'
    return t;
}
"""
_VIEWPORT_JS = """
function(path) {
    var w = window.innerWidth;
    if (window.biomechViewer) { window.biomechViewer.lastWidth = w; }
    return w;
}
"""

_NAV_JS = """
function(path, ids) {
    var alias = %s;
    var p = alias[path] || path || '/';
    return ids.map(function(i) { return i.href === p ? 'navlink active' : 'navlink'; });
}
""" % json.dumps(ALIASES)
_ENV_JS = """
function(theme, viewport, ids) {
    return (ids || []).map(function() { return {theme: theme, viewport: viewport}; });
}
"""


def create_app(ctx: ViewerContext) -> Dash:
    app = Dash(__name__, title=f"biomech · {ctx.config.study}",
               assets_folder=str(Path(__file__).parent / "assets"),
               suppress_callback_exceptions=True)
    app.layout = html.Div(className="shell", children=[
        # callback-nav: writing url.search (Results deep links) must not reload the page.
        dcc.Location(id="url", refresh="callback-nav"),
        dcc.Store(id="theme", data="light"),
        dcc.Store(id="viewport", data=1280),
        html.Nav(className="nav", children=[
            html.B(f"biomech · {ctx.config.study}" + (
                f" · snapshot {ctx.config.snapshot} (read-only)" if ctx.config.snapshot
                else ""), className="brand"),
            html.Div(className="navlinks", children=[
                dcc.Link(name, href=href, className="navlink",
                         id={"type": "navlink", "href": href})
                for href, (name, _) in PAGES.items()]),
            dcc.RadioItems(id="theme-choice", value="auto", inline=True, className="themes",
                           options=[{"label": " auto ", "value": "auto"},
                                    {"label": " light ", "value": "light"},
                                    {"label": " dark ", "value": "dark"}]),
        ]),
        html.Main(id="page"),
    ])
    app.clientside_callback(_THEME_JS, Output("theme", "data"), Input("theme-choice", "value"))
    # On navigation; window resizes reach the store from assets/resize.js (debounced).
    app.clientside_callback(_VIEWPORT_JS, Output("viewport", "data"), Input("url", "pathname"))
    app.clientside_callback(_NAV_JS, Output({"type": "navlink", "href": ALL}, "className"),
                            Input("url", "pathname"),
                            State({"type": "navlink", "href": ALL}, "id"))
    # Every page's environment copy (components.env_store): pages never take the shell's
    # stores as inputs, so no callback mixes shell and page components.
    app.clientside_callback(_ENV_JS, Output({"type": "page-env", "page": ALL}, "data"),
                            Input("theme", "data"), Input("viewport", "data"),
                            State({"type": "page-env", "page": ALL}, "id"))

    @app.callback(Output("page", "children"), Input("url", "pathname"), State("url", "search"))
    def _route(path, search):
        import inspect

        path = ALIASES.get(path, path)
        _, builder = PAGES.get(path or "/", PAGES["/"])
        # Pages that take a deep link (?view=...&pid=...) get the query at build time.
        if "search" in inspect.signature(builder).parameters:
            return builder(ctx, search=search)
        return builder(ctx)

    control.register(app, ctx)
    explorer.register(app, ctx)
    analysis.register(app, ctx)
    paper.register(app, ctx)
    animation.register(app, ctx)

    @app.server.route("/exports/<path:rel>")
    def _files(rel: str):            # serve exported files (preview.pdf etc.) only
        from flask import abort, send_from_directory

        root = ctx.export_dir.resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            abort(404)
        return send_from_directory(root, rel)

    @app.server.route("/video-files/<path:rel>")
    def _videos(rel: str):           # rendered videos of this study only (local viewer)
        from flask import abort, send_from_directory

        root = (ctx.config.out_dir / "videos").resolve()
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            abort(404)
        return send_from_directory(root, rel)

    return app


def quiet_logging(verbose: bool = False) -> None:
    """Silence the werkzeug access log (one line per request) unless ``verbose``."""
    logging.getLogger("werkzeug").setLevel(logging.INFO if verbose else logging.WARNING)


def main(study: str, port: int = 8060, host: str = "127.0.0.1", debug: bool = False,
         data_dir: str | None = None, verbose: bool = False, snapshot: str | None = None
         ) -> None:
    quiet_logging(verbose)
    ctx = ViewerContext.from_study(study, data_dir, snapshot=snapshot)
    create_app(ctx).run(host=host, port=port, debug=debug)
