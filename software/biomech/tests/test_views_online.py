# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pseudo-online (X4) views build, render and show stored numbers only (W7)."""

from __future__ import annotations

import re

import numpy as np
import pytest

from biomech.io.context import DataContext
from biomech.viz import render_mpl, render_pgf, render_plotly
from biomech.viz.views import registry as R

pytestmark = pytest.mark.slow
VIEWS = ("online_accuracy", "online_timeline", "online_compute", "online_release",
         "online_feature_deviation")


@pytest.fixture(scope="module")
def ctx(synth_study):
    return DataContext(synth_study[0])


@pytest.mark.parametrize("vid", VIEWS)
def test_online_view_renders_everywhere(ctx, vid, tmp_path):
    spec = R.build(vid, R.request(ctx, vid, pid=ctx.pids()[0]))
    render_plotly.to_plotly(spec)
    render_mpl.render_figure(spec, tmp_path / f"{vid}.png")
    tex = render_pgf.render_figure(spec, tmp_path)
    text = spec.caption + " ".join(p.title for p in spec.panels) + " ".join(
        " ".join(p.y.ticklabels or []) for p in spec.panels)
    assert not re.search(r"\bs\d{2}\b", text) and not re.search(r"\bs\d{2}\b",
                                                                tex.read_text())


def test_online_accuracy_shows_the_stored_stream_accuracy(ctx):
    s = ctx.table("ml_online", "summary")
    spec = R.build("online_accuracy", R.request(ctx, "online_accuracy"))
    pts = np.concatenate([np.asarray(q.x, float) for q in spec.panels[0].series
                          if q.kind == "scatter" and q.role == "context"])
    want = s[s.applicable.astype(bool)].ba_raw.to_numpy(float)
    np.testing.assert_allclose(np.sort(pts), np.sort(want[np.isfinite(want)]))
