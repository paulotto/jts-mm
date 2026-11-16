# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 viewer fixes on the synthetic study (explorer rows and channels, paper board, maps,
substitution notes, the app over HTTP)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dash")

from biomech.viz.views import registry as R  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def vctx(synth_study):
    from biomech.viewer.context import ViewerContext

    v = ViewerContext.from_study(synth_study[0].source)
    yield v
    v.data.close()


def _ids(node, out=None):
    out = [] if out is None else out
    cid = getattr(node, "id", None)
    if cid is not None:
        out.append(cid)
    kids = getattr(node, "children", None)
    for k in (kids if isinstance(kids, (list, tuple)) else [kids]):
        if k is not None and not isinstance(k, (str, int, float)):
            _ids(k, out)
    return out


def test_explorer_channels_from_session_and_empty_selection(vctx):
    from biomech.viewer import data as VD
    from biomech.viewer.pages import explorer as ex

    pid = vctx.data.pids()[0]
    names = list(vctx.data.emg(pid).names)
    opts, value = ex.channels_for(vctx, pid, None, None)
    assert [o["value"] for o in opts] == names and value == names
    _, keep = ex.channels_for(vctx, pid, names[:2], opts)          # partial selection kept
    assert keep == names[:2]
    _, none = ex.channels_for(vctx, pid, [], opts)                   # empty stays empty
    assert none == []
    fig = VD.explorer_figure(vctx.data, pid, "open_close", [], "envelope", "light")
    assert not [t for t in fig.data if t.legend == "legend" and t.mode == "lines"]
    assert any(a.text == "no EMG channel selected" for a in fig.layout.annotations)


def test_explorer_mvc_phase_explains_empty_rows(vctx):
    from biomech.viewer import data as VD

    pid = vctx.data.pids()[0]
    mvc = [ph.name for ph in vctx.data.phases(pid, "mvc")]
    assert mvc
    fig = VD.explorer_figure(vctx.data, pid, mvc[0], None, "envelope", "light")
    texts = [a.text for a in fig.layout.annotations]
    assert VD.MVC_NOTE in texts and texts.count("MVC take: not available") == 4
    motion = VD.explorer_figure(vctx.data, pid, "open_close", None, "envelope", "light")
    assert VD.MVC_NOTE not in [a.text for a in motion.layout.annotations]


def test_explorer_explains_missing_stage_results(vctx, monkeypatch):
    from biomech.viewer import data as VD

    pid = vctx.data.pids()[0]
    full = VD.explorer_figure(vctx.data, pid, "open_close", None, "envelope", "light")
    assert not [a for a in full.layout.annotations if "results" in a.text or "masked" in a.text]
    # no stored imu / kinematics / labels arrays (a failed stage in a dataset study)
    monkeypatch.setattr(vctx.data, "arrays", lambda stage, p: {})
    fig = VD.explorer_figure(vctx.data, pid, "open_close", None, "envelope", "light")
    texts = [a.text for a in fig.layout.annotations]
    assert texts.count(VD.NO_RESULT_NOTE.format(stage="imu")) == 3
    assert VD.NO_RESULT_NOTE.format(stage="kinematics") in texts
    assert VD.NO_RESULT_NOTE.format(stage="labels") in texts


def test_paper_board_ids_unique_and_cached(vctx, monkeypatch):
    from biomech.viewer.pages import analysis as an
    from biomech.viewer.pages import paper as pb

    vid = R.views("paper")[0].id
    monkeypatch.setattr(vctx, "paper_items", [vid, vid])
    monkeypatch.setattr(vctx, "paper_sets", [])
    an._CACHE.clear()
    blocks = pb.legacy_blocks(vctx, "light", 1200)
    graphs = [i for b in blocks for i in _ids(b) if isinstance(i, dict)]
    assert len(graphs) == len({str(g) for g in graphs}) >= 1
    assert all(g["type"] == "pb-graph" for g in graphs)
    assert len(an._CACHE) == 1                       # both items through one cache entry


def test_demand_map_and_raster_are_heatmaps_with_colourbars(vctx):
    from biomech.viz.render_plotly import to_plotly

    pid = vctx.data.pids()[0]
    for vid in ("data_demand_map", "data_activation_raster"):
        spec = R.build(vid, R.request(vctx.data, vid, pid=pid))
        imgs = [s for p in spec.panels for s in p.series if s.kind == "image"]
        assert imgs and all(np.ndim(s.img) == 2 and s.style.get("cmap") for s in imgs)
        fig = to_plotly(spec, theme="dark")
        assert not any(t.type == "image" for t in fig.data)
        heat = [t for t in fig.data if t.type == "heatmap"]
        assert heat and max(np.asarray(t.z).size for t in heat) < 50_000
        bars = [t for t in fig.data if getattr(t, "marker", None) is not None
                and t.marker.showscale]
        assert len(bars) == sum(p.colorbar is not None for p in spec.panels) >= 1


def test_emg_spectra_without_mains_frequency(vctx, monkeypatch):
    # The stored spectra (G2) carry the grid frequency; NaN = none detected.
    ctx = vctx.data
    orig = ctx.arrays

    def arrays(stage, unit=None, *a, **kw):
        out = orig(stage, unit, *a, **kw)
        if stage == "emg":
            out = {k: (np.array([np.nan]) if k.endswith("__line_hz") else v)
                   for k, v in out.items()}
        return out

    monkeypatch.setattr(ctx, "arrays", arrays)
    pid = ctx.pids()[0]
    spec = R.build("data_emg_spectra", R.request(vctx.data, "data_emg_spectra", pid=pid))
    assert not any(s.kind == "vline" for p in spec.panels for s in p.series)
    assert "no mains frequency detected" in spec.caption


def test_online_timeline_says_whose_stream_is_shown(vctx, monkeypatch):
    from biomech.viz.views import results_online as RO

    pids = vctx.data.pids()
    monkeypatch.setattr(RO, "_online_pids", lambda ctx: pids[1:])
    spec = R.build("online_timeline", R.request(vctx.data, "online_timeline", pid=pids[0]))
    assert spec.panels[0].note == "P1 has no data; showing P2"
    assert spec.caption.startswith("P1 has no data; showing P2.")
    assert "--pid-" not in spec.id


def test_app_serves_results_page_and_layout(vctx):
    from biomech.viewer.app import create_app

    app = create_app(vctx)
    client = app.server.test_client()
    lay = client.get("/_dash-layout")
    assert lay.status_code == 200
    assert '"id":"url","refresh":"callback-nav"' in lay.get_data(as_text=True).replace(" ", "")
    deps = client.get("/_dash-dependencies").get_json()
    outs = [d["output"] for d in deps]
    assert any("url.search" in o for o in outs)
    assert not any(d["output"].startswith("ex-graph.figure") and
                   any(i["id"] == "viewport" for i in d["inputs"]) for d in deps)
    assert client.get("/results?view=fig_x").status_code == 200
