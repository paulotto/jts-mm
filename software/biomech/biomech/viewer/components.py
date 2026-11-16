# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared Dash components: a graph that follows its card (aspect ratio, user-resizable).

``graph_box`` wraps a ``dcc.Graph`` in a ``.graph-box`` (assets/style.css): the box keeps
the figure's aspect ratio as the page width changes (CSS ``aspect-ratio``) or a fixed
height, can be dragged taller or shorter (``resize: vertical``), and the graph fills it
(``responsive=True``, 100 % height). ``assets/resize.js`` re-lays out every Plotly graph
whose container changes size.
"""

from __future__ import annotations

from typing import Any

from dash import dcc, html

GRAPH_CONFIG = {"displaylogo": False, "responsive": True}


def graph_box(gid: str, figure: Any = None, *, aspect: float | None = None,
              height_px: int | None = None, min_px: int | None = None,
              class_name: str = "", box_id: str | None = None) -> html.Div:
    """A resizable box holding graph ``gid``: ``aspect`` = width / height, or ``height_px``."""
    style: dict[str, str] = {}
    if aspect:
        style["aspectRatio"] = f"{aspect:.4f}"
    if height_px:
        style["height"] = f"{int(height_px)}px"
    if min_px:
        style["minHeight"] = f"{int(min_px)}px"
    kw: dict[str, Any] = {} if figure is None else {"figure": figure}
    graph = dcc.Graph(id=gid, config=GRAPH_CONFIG, responsive=True,
                      style={"height": "100%", "width": "100%"}, **kw)
    kw2: dict[str, Any] = {"id": box_id} if box_id else {}
    return html.Div(graph, className=f"graph-box {class_name}".strip(), style=style, **kw2)


# -- page environment ------------------------------------------------------------------------
# Theme and window width live in the shell ("theme", "viewport"). A page callback that took
# them as Inputs next to its own controls fired on every theme or resize event also while
# another page was shown, and Dash reported the page's controls as nonexistent.
# Each page therefore holds its own copy (``env_store``), which the shell fills through a
# pattern-matching output (app.py): page callbacks take only inputs of their own page.
def env_id(page: str) -> dict[str, str]:
    return {"type": "page-env", "page": page}


def env_store(page: str) -> Any:
    return dcc.Store(id=env_id(page))


def env_of(env: dict[str, Any] | None) -> tuple[str, int]:
    """(theme, viewport width) of a page environment (defaults before the first copy)."""
    env = env or {}
    return env.get("theme") or "light", int(env.get("viewport") or 1280)


def short_path(path: Any) -> str:
    """A path relative to the repository where possible (pages show where things live
    without a horizontal scroll)."""
    from pathlib import Path

    from biomech.config import REPO_ROOT

    p = Path(str(path))
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)
