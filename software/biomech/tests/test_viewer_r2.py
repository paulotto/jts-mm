# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer R2 (plan W2): mask reasons, responsive sizing, multi-phase 3-D, id labels, logging."""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest

pytest.importorskip("dash")

from biomech.emg.artifacts import MaskResult  # noqa: E402
from biomech.io.context import DataContext  # noqa: E402
from biomech.viewer import data as VD  # noqa: E402
from biomech.viz.views import registry as R  # noqa: E402
from biomech.viz.views.data import TASK_LABEL, TASKS  # noqa: E402


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


@pytest.fixture(scope="module")
def vctx(synth_study):
    from biomech.viewer.context import ViewerContext

    v = ViewerContext.from_study(synth_study[0].source)
    yield v
    v.data.close()


def _find(node, cid):
    if getattr(node, "id", None) == cid:
        return node
    kids = getattr(node, "children", None)
    for k in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if k is not None and not isinstance(k, (str, int, float)):
            hit = _find(k, cid)
            if hit is not None:
                return hit
    return None


def _motions(ctx, pid):
    return [t for t in TASKS if t in VD.phase_names(ctx, pid)]


# -- multi-phase selection ------------------------------------------------------------------
def test_explorer_layout_has_multi_phase_selector(vctx):
    from biomech.viewer.pages import explorer

    page = explorer.layout(vctx)
    dd = _find(page, "ex-phase")
    assert dd is not None and dd.multi is True
    assert _find(page, "ex-unmasked") is not None
    split = _find(page, "ex-split")
    assert any("split-handle" in (getattr(c, "className", "") or "") for c in split.children)
    names = VD.phase_names(vctx.data, vctx.data.pids()[0])
    opts = VD.phase_options(names)
    assert opts[0] == {"label": "all motions", "value": VD.ALL_MOTIONS}
    assert VD.selected_phases([VD.ALL_MOTIONS], names) == [t for t in TASKS if t in names]
    assert VD.selected_phases(["chewing", "open_close", "chewing"], names) == ["chewing",
                                                                              "open_close"]
    assert VD.primary_phase([VD.ALL_MOTIONS], names) == "open_close"
    assert VD.primary_phase([VD.ALL_MOTIONS, "chewing"], names) == "chewing"
    # picking a second motion switches the single-phase views to it (user report)
    assert VD.primary_phase(["open_close", "chewing"], names) == "chewing"
    assert VD.primary_phase(["chewing", "open_close"], names) == "open_close"
    assert VD.view_phase([VD.ALL_MOTIONS], names) == VD.ALL_MOTIONS
    assert VD.view_phase(["chewing"], names) == "chewing"
    assert VD.view_phase(["chewing", "open_close"], names) == ["open_close", "chewing"]


def test_jaw3d_figure_draws_each_task_in_its_colour(ctx):
    from biomech.viz import style
    from biomech.viz.views.data import TASK_ROLE

    pid = ctx.pids()[0]
    tasks = _motions(ctx, pid)
    assert len(tasks) >= 2
    fig = VD.jaw3d_figure(ctx, pid, tasks[:2], "light")
    for t in tasks[:2]:
        group = [tr for tr in fig.data if tr.legendgroup == t]
        assert group, t
        lead = [tr for tr in group if tr.showlegend is not False]
        assert len(lead) == 1 and lead[0].name == TASK_LABEL[t]
        want = style.hexcolor(style.role(TASK_ROLE[t])[0], "light")
        assert {tr.line.color for tr in group} == {want}
    every = VD.jaw3d_figure(ctx, pid, VD.ALL_MOTIONS, "light")
    assert {tr.legendgroup for tr in every.data if tr.legendgroup} == set(tasks)
    one = VD.jaw3d_figure(ctx, pid, tasks[0], "light")     # one phase: colour per landmark
    assert not any(tr.legendgroup for tr in one.data)


def test_atlas_views_overlay_a_list_of_phases(ctx):
    pid = ctx.pids()[0]
    tasks = _motions(ctx, pid)[:2]
    for vid in ("data_jaw3d", "data_orthographic"):
        assert R.get(vid).multi_phase
        spec = R.build(vid, R.request(ctx, vid, pid=pid, phase=tasks))
        labels = {s.label for p in spec.panels for s in p.series if s.label}
        assert {TASK_LABEL[t] for t in tasks} <= labels
        others = {TASK_LABEL[t] for t in TASKS if t not in tasks}
        assert not (others & labels)


def test_registry_ids_for_list_phases(ctx):
    pid = ctx.pids()[0]

    def sid(vid, phase):
        return R.build(vid, R.request(ctx, vid, pid=pid, phase=phase)).id

    a = sid("data_jaw3d", ["chewing", "open_close"])
    assert a.endswith("--phase-open_close+chewing")
    assert sid("data_jaw3d", ["open_close", "chewing", "open_close"]) == a      # stable
    assert sid("data_jaw3d", "all").endswith("--phase-all")
    assert sid("data_jaw3d", list(TASKS)).endswith("--phase-all")
    assert sid("data_jaw3d", ["chewing"]).endswith("--phase-chewing")
    assert R.phase_token(["left_right", "open_close"]) == "open_close+left_right"
    # A single-phase view takes the first phase of a list.
    req = R.request(ctx, "data_speed_path", pid=pid, phase=["chewing", "open_close"])
    assert req.phase == "chewing"
    assert not R.get("data_speed_path").multi_phase


# -- mask reasons ---------------------------------------------------------------------------
def test_mask_marks_by_reason_with_one_legend_entry(ctx, monkeypatch):
    pid = ctx.pids()[0]
    take, ph = ctx.phase(pid, "open_close")
    prep = ctx.emg(pid)
    pe = prep.phase("open_close", take=take)
    n, c = pe.raw.shape
    hits = {r: np.zeros((n, c), bool) for r in ("rail", "leadoff", "lowfreq")}
    hits["rail"][100:300, :] = True                  # every channel: one legend entry
    hits["leadoff"][500:520, 0] = True
    hits["lowfreq"][900:950, 1] = True
    full = hits["rail"] | hits["leadoff"] | hits["lowfreq"]
    full[1000:1010, 2] = True                        # dilation only
    rules = ("rail", "leadoff", "envelope", "lowfreq")          # sensitivity config
    fake = dataclasses.replace(pe, masks=MaskResult(full, hits=hits, analysis_rules=rules))
    monkeypatch.setattr(prep, "phase", lambda *a, **k: fake)

    fig = VD.explorer_figure(ctx, pid, "open_close", None, "envelope", "light")
    marks = [t for t in fig.data if (t.name or "").startswith("masked: ")]
    by = {}
    for t in marks:
        by.setdefault(t.name, []).append(t)
    assert set(by) == {"masked: " + VD.MASK_REASONS[r][0]
                       for r in ("rail", "leadoff", "lowfreq", "dilation")}
    rail = by["masked: " + VD.MASK_REASONS["rail"][0]]
    assert len(rail) == c and sum(t.showlegend is not False for t in rail) == 1
    assert len({t.marker.color for t in marks}) == 4             # one colour per reason
    assert fig.layout.yaxis.title.text == VD.EMG_NORM_LABEL

    # D14 default: lowfreq hits are shown as a QC flag, not as excluded samples.
    full14 = hits["rail"] | hits["leadoff"]
    fake14 = dataclasses.replace(pe, masks=MaskResult(full14, hits=hits))
    monkeypatch.setattr(prep, "phase", lambda *a, **k: fake14)
    names = {t.name for t in VD.explorer_figure(ctx, pid, "open_close", None, "envelope",
                                                 "light").data if t.name}
    assert "flagged (kept): " + VD.MASK_REASONS["lowfreq"][0] in names
    assert "masked: " + VD.MASK_REASONS["lowfreq"][0] not in names
    monkeypatch.setattr(prep, "phase", lambda *a, **k: fake)

    # Unmasked envelope: faint, on the same scale as pe.norm where nothing is masked.
    fig2 = VD.explorer_figure(ctx, pid, "open_close", None, "envelope", "light", unmasked=True)
    faint = [t for t in fig2.data if (t.name or "").endswith(", unmasked")]
    assert len(faint) == c and all(t.opacity < 0.5 for t in faint)
    from scipy.ndimage import binary_dilation

    un = VD.unmasked_norm(prep, pe)
    # Next to masked samples the masked envelope averages fewer samples: compare away from them.
    w = int(round(prep.p.envelope_s * pe.fs)) + 2
    near = binary_dilation(pe.mask, structure=np.ones((2 * w + 1, 1), bool)) \
        if pe.mask.any() else pe.mask
    ok = ~near & np.isfinite(pe.norm)
    ok[:w] = ok[-w:] = False
    assert ok.sum() > 1000
    assert np.allclose(un[ok], pe.norm[ok], rtol=1e-6, atol=1e-6)
    # No marks outside envelope mode.
    raw = VD.explorer_figure(ctx, pid, "open_close", None, "raw", "light")
    assert not any((t.name or "").startswith("masked: ") for t in raw.data)


# -- responsive sizing ----------------------------------------------------------------------
def test_resize_asset_present_and_served(vctx):
    from biomech.viewer.app import create_app

    app = create_app(vctx)
    client = app.server.test_client()
    r = client.get("/assets/resize.js")
    assert r.status_code == 200
    js = r.get_data(as_text=True)
    assert "ResizeObserver" in js and "Plotly.Plots.resize" in js
    assert "MutationObserver" in js and "set_props('viewport'" in js
    assert "resize.js" in client.get("/").get_data(as_text=True)


def test_render_fits_the_paper_aspect(ctx):
    from biomech.viewer.pages.analysis import render
    from biomech.viz.render_plotly import aspect_of, paper_aspect, to_plotly

    pid = ctx.pids()[0]
    spec = R.build("data_orthographic", R.request(ctx, "data_orthographic", pid=pid))
    fig = to_plotly(spec, viewport_px=1500, fit=True)
    a = aspect_of(fig)
    assert 0.6 / paper_aspect(spec) < a < 1.6 / paper_aspect(spec)
    flat = to_plotly(spec, viewport_px=1500)              # legacy fixed-height layout
    assert fig.layout.height > flat.layout.height        # equal-aspect panels not squeezed
    box = render(spec, "g", "light", 1500).children[0]
    assert "graph-box" in box.className and box.style["aspectRatio"]
    graph = box.children
    assert graph.responsive is True and graph.style["height"] == "100%"


# -- labels and logging ---------------------------------------------------------------------
def test_dropdown_label_shows_the_pseudonym_only(vctx):
    from biomech.check.outputs import PID_RE
    from biomech.viewer.pages import analysis, explorer

    pids = vctx.data.pids()
    assert VD.pid_label(vctx.data, pids[0]) == "P1"
    assert VD.unit_label(vctx.data, pids[0], operator=True) == f"P1 · {pids[0]}"
    dd = _find(explorer.layout(vctx), "ex-pid")
    assert [o["label"] for o in dd.options] == [f"P{i + 1}" for i in range(len(pids))]
    assert _find(analysis.layout(vctx), "an-pid").options[0]["label"] == "P1"
    # Figures keep pseudonyms only.
    spec = R.build("data_jaw3d", R.request(vctx.data, "data_jaw3d", pid=pids[0], phase="all"))
    text = [spec.title, spec.caption, spec.perspective] + [
        s.label or "" for p in spec.panels for s in p.series] + [p.title for p in spec.panels]
    assert not any(PID_RE.search(t or "") for t in text)


def test_main_silences_the_access_log(monkeypatch):
    from click.testing import CliRunner

    from biomech import cli
    from biomech.viewer import app as A

    class _App:
        def run(self, **kw):
            pass

    monkeypatch.setattr(A.ViewerContext, "from_study", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(A, "create_app", lambda ctx: _App())
    log = logging.getLogger("werkzeug")
    old = log.level
    try:
        A.main("study.yaml")
        assert log.level == logging.WARNING
        A.main("study.yaml", verbose=True)
        assert log.level == logging.INFO
    finally:
        log.setLevel(old)
    assert "--verbose" in CliRunner().invoke(cli.main, ["view", "--help"]).output
