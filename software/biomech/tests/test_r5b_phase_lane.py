# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5b: labelled phase bands (one colour per movement task, rest; neutral holds)."""

from __future__ import annotations

import numpy as np

from biomech.viz import render_mpl, render_plotly
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import blocks as B


def _labels():
    t = np.arange(0, 60, 0.025)
    lab = np.full(len(t), "", dtype=object)
    for name, a, b in [("mand_point_1", 0, 8), ("max_point_1", 9, 11), ("rest", 12, 20),
                       ("open_close", 21, 35), ("chewing", 36, 50), ("free_motion", 51, 59)]:
        lab[(t >= a) & (t < b)] = name
    return t, lab


def test_phase_spans_and_lane():
    t, lab = _labels()
    spans = B.phase_spans(t, lab)
    assert [s[0] for s in spans] == ["mand_point_1", "max_point_1", "rest", "open_close",
                                     "chewing", "free_motion"]
    lane = B.phase_lane(spans, extra=[("sync_start", 1.0, 6.0)])
    roles = {s.role for s in lane.series if s.kind == "hspan"}
    assert {"cat:0", "cat:3", "cat:4", "cat:5", "faint", "context"} <= roles
    text = [s for s in lane.series if s.kind == "text"][0].text
    assert "open" in text and "L1" in text and "sync" in text


def test_lane_renders_above_a_data_panel(tmp_path):
    t, lab = _labels()
    spans = B.phase_spans(t, lab)
    lane = B.phase_lane(spans)
    data = Panel(title="incisor", x=Axis("time", "\\second"), y=Axis("x", "\\milli\\metre"))
    data.series.append(Series("line", x=t, y=np.sin(t), label="JTS-MM"))
    B.add_phase_bands(data, spans)
    spec = FigureSpec("lane", "", [lane, data], rows=2, cols=1, width="text")
    render_mpl.render_figure(spec, tmp_path / "lane.png", dpi=60)
    assert (tmp_path / "lane.png").stat().st_size > 1000
    assert render_plotly.figure(spec) is not None if hasattr(render_plotly, "figure") else True
