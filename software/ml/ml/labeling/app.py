# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Dash labeling application for JTS / optical trajectories."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path

import numpy as np
from werkzeug.serving import make_server

from ml.config.classes import UNLABELLED
from ml.io.labels import default_label_path, read_labels, write_labels
from ml.labeling.constants import LABEL_COLORS, PAINT_IDS, display_class_name
from ml.labeling.figures import (
    build_label_bar,
    build_main_figure,
    parse_selection_range,
    selection_from_slider,
)
from ml.labeling.phases import PhaseView, apply_label_to_phase, build_phase_views, reset_phase_labels
from ml.labeling.trajectory import LabelingGrid, load_labeling_grid


def run_label_app(
    session_path: Path,
    *,
    jts_output: Path | None = None,
    jts_config: Path | None = None,
    label_path: Path | None = None,
    port: int = 8050,
    host: str = "0.0.0.0",
    show_velocity: bool = False,
    invert_axis: tuple[bool, bool, bool] = (False, False, False),
) -> Path:
    """Launch the interactive labeling GUI and block until the user saves."""
    try:
        import dash
        from dash import Input, Output, State, callback_context, dcc, html
        from dash.exceptions import PreventUpdate
    except ImportError as exc:
        raise ImportError(
            "ml label requires the label extra. From the repo root run:\n"
            "  pip install -e './software/ml[label]'\n"
            "Or from software/ml:\n"
            "  pip install -e '.[label]'"
        ) from exc

    session_path = Path(session_path)
    grid = load_labeling_grid(
        session_path,
        jts_output=jts_output,
        jts_config=jts_config,
        invert_axis=invert_axis,
    )
    phase_views = build_phase_views(grid)
    label_path = Path(label_path) if label_path else default_label_path(session_path)
    labels = _initial_labels(grid, label_path)

    undo_stack: deque[list[int]] = deque(maxlen=50)
    done_evt = threading.Event()
    saved_path: list[Path | None] = [None]
    n = len(grid.t_s)

    app = dash.Dash(__name__, title=f"ml label: {grid.session.session_id}")

    def _view(phase_idx: int) -> PhaseView:
        return phase_views[int(phase_idx)]

    def _slider_defaults(phase_idx: int) -> tuple[float, float, float, float]:
        view = _view(phase_idx)
        t_max = float(view.t_s[-1]) if len(view.t_s) else 1.0
        return 0.0, max(t_max, 0.001), max(t_max / 500.0, 0.001), min(1.0, t_max)

    def _phase_heading(phase_idx: int) -> str:
        return _view(phase_idx).phase.name.replace("_", " ")

    def _phase_step(phase_idx: int) -> str:
        view = _view(phase_idx)
        return f"Experiment phase {view.index + 1} of {view.total}"

    s_min, s_max, s_step, s_hi = _slider_defaults(0)

    app.layout = html.Div(
        style={
            "fontFamily": "system-ui, sans-serif",
            "padding": "16px",
            "maxWidth": "1400px",
            "margin": "0 auto",
        },
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "space-between",
                    "marginBottom": "12px",
                    "gap": "12px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        children=[
                            html.H2(
                                f"Manual labeling — {grid.session.session_id}",
                                style={"margin": 0, "fontSize": "18px", "fontWeight": "600"},
                            ),
                            html.P(
                                f"Trajectory: {grid.trajectory_source} · "
                                f"{n} frames total · save → {label_path.name}",
                                style={"margin": "4px 0 0", "fontSize": "12px", "color": "#666"},
                            ),
                        ]
                    ),
                    html.Div(
                        style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
                        children=[
                            html.Button("← Previous", id="btn-prev-phase", n_clicks=0, style=_button_style("#546E7A")),
                            html.Button("Next →", id="btn-next-phase", n_clicks=0, style=_button_style("#546E7A")),
                            html.Button("Undo", id="btn-undo", n_clicks=0, style=_button_style("#616161")),
                            html.Button("Reset phase", id="btn-reset", n_clicks=0, style=_button_style("#757575")),
                            html.Button("Save", id="btn-save", n_clicks=0, style=_button_style("#1976D2")),
                            html.Button("Save & close", id="btn-save-close", n_clicks=0, style=_button_style("#0D47A1")),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="phase-banner",
                style={
                    "background": "#E3F2FD",
                    "border": "1px solid #90CAF9",
                    "borderRadius": "10px",
                    "padding": "16px 20px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        id="phase-step",
                        children=_phase_step(0),
                        style={
                            "fontSize": "13px",
                            "fontWeight": "600",
                            "letterSpacing": "0.04em",
                            "textTransform": "uppercase",
                            "color": "#1565C0",
                            "marginBottom": "4px",
                        },
                    ),
                    html.H1(
                        id="phase-heading",
                        children=_phase_heading(0),
                        style={
                            "margin": 0,
                            "fontSize": "32px",
                            "fontWeight": "700",
                            "color": "#0D47A1",
                            "lineHeight": 1.15,
                        },
                    ),
                ],
            ),
            html.P(
                "Label one experiment phase at a time. Select a span (box/lasso or slider), "
                "pick a class, then use Next to continue. Unpainted frames are ignored (not REST): "
                "paint REST where the jaw rests, IGNORE clears a span. Save writes all phases.",
                style={"fontSize": "13px", "color": "#555", "marginBottom": "8px"},
            ),
            html.Div(
                id="label-buttons",
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap", "marginBottom": "12px"},
                children=[
                    html.Button(
                        f"{cid}  {display_class_name(cid)}",
                        id={"type": "label-btn", "index": cid},
                        n_clicks=0,
                        style={
                            "padding": "6px 14px",
                            "fontSize": "13px",
                            "background": LABEL_COLORS[cid],
                            "color": "white",
                            "border": "none",
                            "borderRadius": "5px",
                            "cursor": "pointer",
                        },
                    )
                    for cid in PAINT_IDS
                ],
            ),
            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Label("Selection range within phase (s)", style={"fontSize": "12px", "color": "#555"}),
                    dcc.RangeSlider(
                        id="range-slider",
                        min=s_min,
                        max=s_max,
                        step=s_step,
                        value=[s_min, s_hi],
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
            ),
            dcc.Graph(
                id="main-plot",
                config={
                    "modeBarButtonsToAdd": ["select2d", "lasso2d"],
                    "scrollZoom": True,
                    "displayModeBar": True,
                },
                style={"height": "520px"},
            ),
            dcc.Graph(id="label-bar", style={"height": "120px", "marginTop": "4px"}),
            html.Div(
                id="status",
                style={"fontSize": "12px", "color": "#666", "marginTop": "6px", "minHeight": "18px"},
            ),
            dcc.Store(id="store-labels", data=labels.tolist()),
            dcc.Store(id="store-phase-idx", data=0),
            dcc.Store(id="store-selection", data={"x_start": None, "x_end": None, "source": "plot"}),
            dcc.Store(id="store-done", data=False),
            dcc.Store(id="store-velocity", data=show_velocity),
        ],
    )

    def _snapshot(current: list[int]) -> None:
        undo_stack.append(list(current))

    def _phase_slice(view: PhaseView, current_labels: list[int]) -> np.ndarray:
        return np.asarray(current_labels, dtype=int)[view.global_indices]

    def _apply_span(
        current: list[int],
        phase_idx: int,
        x0: float | None,
        x1: float | None,
        class_id: int,
    ) -> tuple[list[int], str]:
        if x0 is None or x1 is None:
            return current, "No range selected — box/lasso on the plot or adjust the range slider."
        view = _view(phase_idx)
        _snapshot(current)
        updated, count = apply_label_to_phase(
            current,
            view,
            t_start=x0,
            t_end=x1,
            class_id=class_id,
        )
        if count == 0:
            if undo_stack:
                undo_stack.pop()
            lo, hi = sorted([x0, x1])
            return current, f"Selected range {lo:.3f}–{hi:.3f} s matches no samples in this phase."
        lo, hi = sorted([x0, x1])
        return updated, (
            f"[{view.phase.name}] Applied '{display_class_name(class_id)}' to "
            f"{lo:.3f}–{hi:.3f} s ({count} frames)."
        )

    @app.callback(
        Output("store-selection", "data"),
        Input("main-plot", "selectedData"),
        prevent_initial_call=True,
    )
    def capture_plot_selection(selected_data):
        x0, x1 = parse_selection_range(selected_data)
        return {"x_start": x0, "x_end": x1, "source": "plot"}

    @app.callback(
        Output("store-selection", "data", allow_duplicate=True),
        Input("range-slider", "value"),
        prevent_initial_call=True,
    )
    def capture_slider_selection(value):
        x0, x1 = selection_from_slider(value)
        return {"x_start": x0, "x_end": x1, "source": "slider"}

    label_button_inputs = [Input({"type": "label-btn", "index": cid}, "n_clicks") for cid in PAINT_IDS]

    @app.callback(
        Output("store-labels", "data"),
        Output("status", "children"),
        label_button_inputs,
        State("store-selection", "data"),
        State("store-labels", "data"),
        State("store-phase-idx", "data"),
        prevent_initial_call=True,
    )
    def apply_label(*args):
        *_clicks, selection, current_labels, phase_idx = args
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        prop = ctx.triggered[0]["prop_id"]
        if not prop.startswith('{"index"'):
            raise PreventUpdate
        class_id = int(json.loads(prop.split(".")[0])["index"])
        updated, status = _apply_span(
            current_labels,
            int(phase_idx or 0),
            selection.get("x_start"),
            selection.get("x_end"),
            class_id,
        )
        return updated, status

    @app.callback(
        Output("store-labels", "data", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Input("btn-undo", "n_clicks"),
        prevent_initial_call=True,
    )
    def undo(_n):
        if not undo_stack:
            raise PreventUpdate
        restored = undo_stack.pop()
        return restored, "Undid last label change."

    @app.callback(
        Output("store-labels", "data", allow_duplicate=True),
        Output("status", "children", allow_duplicate=True),
        Input("btn-reset", "n_clicks"),
        State("store-labels", "data"),
        State("store-phase-idx", "data"),
        prevent_initial_call=True,
    )
    def reset_labels(_n, current_labels, phase_idx):
        if _n:
            _snapshot(current_labels)
        view = _view(int(phase_idx or 0))
        restored = reset_phase_labels(current_labels, view)
        return restored, f"Reset phase '{view.phase.name}' to unlabelled (ignored)."

    @app.callback(
        Output("store-phase-idx", "data"),
        Output("range-slider", "min"),
        Output("range-slider", "max"),
        Output("range-slider", "step"),
        Output("range-slider", "value"),
        Output("phase-heading", "children"),
        Output("phase-step", "children"),
        Output("status", "children", allow_duplicate=True),
        Input("btn-next-phase", "n_clicks"),
        Input("btn-prev-phase", "n_clicks"),
        State("store-phase-idx", "data"),
        prevent_initial_call=True,
    )
    def change_phase(next_clicks, prev_clicks, phase_idx):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        idx = int(phase_idx or 0)
        if ctx.triggered_id == "btn-next-phase":
            if idx >= len(phase_views) - 1:
                return (
                    idx,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    _phase_heading(idx),
                    _phase_step(idx),
                    "Already on the last experiment phase.",
                )
            idx += 1
        elif ctx.triggered_id == "btn-prev-phase":
            if idx <= 0:
                return (
                    idx,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    dash.no_update,
                    _phase_heading(idx),
                    _phase_step(idx),
                    "Already on the first experiment phase.",
                )
            idx -= 1
        else:
            raise PreventUpdate

        s_min, s_max, s_step, s_hi = _slider_defaults(idx)
        return (
            idx,
            s_min,
            s_max,
            s_step,
            [s_min, s_hi],
            _phase_heading(idx),
            _phase_step(idx),
            f"Switched to phase '{_view(idx).phase.name}'.",
        )

    def _persist(current_labels: list[int]) -> Path:
        y = np.asarray(current_labels, dtype=np.uint8)
        return write_labels(
            label_path,
            t_master_us=grid.t_master_us,
            optical_frame=grid.optical_frame,
            y=y,
            session_id=grid.session.session_id,
            subject_id=grid.session.subject_id,
            jts_output=grid.jts_output or "",
            jts_config=grid.jts_config or "",
            jts_group=grid.jts_group,
            trajectory_source=grid.trajectory_source,
        )

    @app.callback(
        Output("status", "children", allow_duplicate=True),
        Input("btn-save", "n_clicks"),
        State("store-labels", "data"),
        prevent_initial_call=True,
    )
    def save_only(_n, current_labels):
        if not _n:
            raise PreventUpdate
        path = _persist(current_labels)
        saved_path[0] = path
        return f"Saved labels → {path}"

    @app.callback(
        Output("store-done", "data"),
        Output("status", "children", allow_duplicate=True),
        Input("btn-save-close", "n_clicks"),
        State("store-labels", "data"),
        prevent_initial_call=True,
    )
    def save_and_close(_n, current_labels):
        if not _n:
            raise PreventUpdate
        path = _persist(current_labels)
        saved_path[0] = path
        done_evt.set()
        return True, f"Saved labels → {path}. Closing…"

    @app.callback(
        Output("main-plot", "figure"),
        Output("label-bar", "figure"),
        Input("store-labels", "data"),
        Input("store-phase-idx", "data"),
        State("store-velocity", "data"),
    )
    def refresh_figures(current_labels, phase_idx, show_vel):
        view = _view(int(phase_idx or 0))
        lbl_phase = _phase_slice(view, current_labels)
        gi = view.global_indices
        fig = build_main_figure(
            view.t_s,
            grid.translations[gi],
            grid.translations_smooth[gi],
            lbl_phase,
            phases=None,
            velocity=grid.velocity[gi] if grid.velocity is not None else None,
            show_velocity=bool(show_vel),
            uirevision=f"phase-{view.index}",
        )
        return fig, build_label_bar(lbl_phase)

    server = make_server(host, port, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    bind = "all interfaces" if host in {"0.0.0.0", "::"} else host
    print(f"\n[ml label] GUI at http://{host}:{port}/  ({bind})")
    print(f"           Session: {session_path}")
    print(f"           Phases: {len(phase_views)} experiment phase(s)")
    print(f"           Trajectory source: {grid.trajectory_source}")
    print(f"           Save target: {label_path}\n")

    done_evt.wait()
    server.shutdown()
    thread.join(timeout=3)

    result = saved_path[0] or label_path
    print(f"[ml label] Done. Labels at {result}")
    return Path(result)


def _initial_labels(grid: LabelingGrid, label_path: Path) -> np.ndarray:
    n = len(grid.t_master_us)
    if label_path.is_file():
        try:
            sidecar = read_labels(label_path)
            if len(sidecar.y) == n and np.array_equal(sidecar.t_master_us, grid.t_master_us):
                print(f"[ml label] Loaded existing labels from {label_path}")
                return np.asarray(sidecar.y, dtype=np.uint8)
            print(
                f"[ml label] Warning: existing label file length mismatch "
                f"({len(sidecar.y)} vs {n}); starting fresh."
            )
        except Exception as exc:
            print(f"[ml label] Warning: could not load {label_path}: {exc}")
    # Unpainted frames are unlabelled (255) and ignored downstream; REST only where painted.
    return np.full(n, UNLABELLED, dtype=np.uint8)


def _button_style(bg: str) -> dict:
    return {
        "padding": "8px 16px",
        "fontSize": "13px",
        "background": bg,
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "cursor": "pointer",
    }
