# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Plotly figures for the labeling UI."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml.config.classes import CLASS_NAMES, UNLABELLED
from ml.io.session import PhaseInterval
from ml.labeling.constants import AXIS_TITLES, LABEL_COLORS, PHASE_COLORS, display_class_name


def build_main_figure(
    t_s: np.ndarray,
    translations: np.ndarray,
    translations_smooth: np.ndarray,
    labels: np.ndarray,
    *,
    phases: list[PhaseInterval] | None = None,
    t0_us: int = 0,
    velocity: np.ndarray | None = None,
    show_velocity: bool = False,
    uirevision: str = "keep",
) -> go.Figure:
    lbl = np.asarray(labels, dtype=int)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=AXIS_TITLES,
        vertical_spacing=0.06,
    )

    for axis in range(3):
        row = axis + 1
        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=translations[:, axis],
                mode="lines",
                line={"width": 1, "color": "#BBBBBB"},
                name=f"raw {AXIS_TITLES[axis]}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_s,
                y=translations_smooth[:, axis],
                mode="lines",
                line={"width": 1.5, "color": "#333333"},
                name=AXIS_TITLES[axis],
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        if show_velocity and velocity is not None:
            fig.add_trace(
                go.Scatter(
                    x=t_s,
                    y=velocity[:, axis],
                    mode="lines",
                    line={"width": 1, "color": "#1565C0", "dash": "dot"},
                    name=f"vel {axis}",
                    showlegend=False,
                    yaxis="y2" if row == 1 else None,
                ),
                row=row,
                col=1,
            )

        _add_label_bands(fig, t_s, lbl, row=row)
        if phases:
            _add_phase_bands(fig, phases, t0_us, row=row)

    fig.update_layout(
        margin={"l": 55, "r": 20, "t": 36, "b": 28},
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
        dragmode="select",
        selectdirection="h",
        hovermode="x unified",
        uirevision=uirevision,
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def build_label_bar(labels: np.ndarray) -> go.Figure:
    n = max(len(labels), 1)
    lbl = np.asarray(labels, dtype=int)
    class_ids = [*sorted(CLASS_NAMES), UNLABELLED]
    vals = [float(np.sum(lbl == cid)) / n * 100.0 for cid in class_ids]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=[display_class_name(cid) for cid in class_ids],
            orientation="h",
            marker_color=[LABEL_COLORS[cid] for cid in class_ids],
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
        )
    )
    fig.update_layout(
        margin={"l": 110, "r": 20, "t": 4, "b": 20},
        paper_bgcolor="white",
        plot_bgcolor="#F8F9FA",
        xaxis={"range": [0, max(100.0, max(vals) * 1.1 if vals else 100.0)], "title": "coverage (%)"},
        yaxis={"title": None},
        showlegend=False,
        height=120,
    )
    return fig


def _add_label_bands(fig: go.Figure, t_s: np.ndarray, labels: np.ndarray, *, row: int) -> None:
    for class_id in CLASS_NAMES:        # unlabelled (255) spans stay unshaded
        mask = labels == class_id
        if not mask.any():
            continue
        for start, end in _contiguous_segments(mask):
            x0 = float(t_s[start])
            x1 = float(t_s[min(end, len(t_s) - 1)])
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=LABEL_COLORS[class_id],
                opacity=0.18,
                layer="below",
                line_width=0,
                row=row,
                col=1,
            )


def _add_phase_bands(fig: go.Figure, phases: list[PhaseInterval], t0_us: int, *, row: int) -> None:
    for idx, phase in enumerate(phases):
        if phase.kind != "experiment":
            continue
        x0 = (phase.t_start_us - t0_us) * 1e-6
        x1 = (phase.t_end_us - t0_us) * 1e-6
        color = PHASE_COLORS[idx % len(PHASE_COLORS)]
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text=phase.name,
            annotation_position="top left",
            row=row,
            col=1,
        )


def _contiguous_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], mask, [False]))
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def parse_selection_range(selected_data: dict | None) -> tuple[float | None, float | None]:
    if not selected_data:
        return None, None
    rng = selected_data.get("range") or {}
    for key in ("x", "x2", "x3"):
        if key in rng and len(rng[key]) == 2:
            return float(min(rng[key])), float(max(rng[key]))
    points = selected_data.get("points") or []
    if points:
        xs = [p.get("x") for p in points if p.get("x") is not None]
        if xs:
            return float(min(xs)), float(max(xs))
    return None, None


def selection_from_slider(slider_value: list[float] | None) -> tuple[float | None, float | None]:
    if not slider_value or len(slider_value) != 2:
        return None, None
    return float(min(slider_value)), float(max(slider_value))
