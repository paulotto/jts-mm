# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D5 supplementary videos (an11): v1_hinge, v2_intent, v3_release."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from biomech import timebase

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
LAYOUTS = ["v1_hinge", "v2_intent", "v3_release"]


@pytest.fixture(scope="module")
def ctx(synth_study):
    from biomech.io.context import DataContext

    c = DataContext(synth_study[0])
    yield c
    c.close()


@pytest.fixture(scope="module")
def anims(ctx):
    from biomech.anim import make

    return {lay: make(ctx, "s01", lay, None, 0.0, 4.0, speed=None, fps=10) for lay in LAYOUTS}


def _texts(scene) -> list[str]:
    out = []
    for d in scene.regions:
        out += [t.get_text() for t in d["fig"].findobj(lambda o: hasattr(o, "get_text"))]
    return out


@pytest.mark.parametrize("layout", LAYOUTS)
def test_layout_renders_with_captions_and_pseudonym_only(anims, layout):
    from biomech.anim import timecode
    from biomech.anim.layers import Caption

    a = anims[layout]
    assert a.clip.label == "Participant 1"
    sc = a.scene(640)
    try:
        sc.prepare([0, a.timeline.n_frames // 2, a.timeline.n_frames - 1])
        f = sc.frame(1)
        texts = _texts(sc)
        cap = next(ly for ly in sc.comp.layers if isinstance(ly, Caption))
    finally:
        sc.close()
    assert f.shape == (360, 640, 3) and f.std() > 5
    assert timecode.read_timecode(f) == int(round(a.timeline.frame_times(
        [a.timeline.n_frames // 2])[0]))
    assert not any("s01" in t for t in texts)
    # Per-segment captions: every frame shows a cue, and the cues change over the video.
    assert all(cap.strings[i][0] for i in range(3)) and len({s[0] for s in cap.strings}) > 1
    if layout == "v2_intent":
        assert any("simulated real time, no device" in t for t in texts)
        assert a.speed == 1.0 and a.manifest()["banner"] == "simulated real time, no device"


def test_v2_shows_only_the_past(anims):
    from biomech.anim.layers import Events, Ribbon, Scope

    a = anims["v2_intent"]
    comp = a.compose()
    scrolling = [ly for ly in comp.layers if isinstance(ly, (Scope, Ribbon, Events))]
    assert scrolling and all(ly.window[1] == 0.0 for ly in scrolling)
    tl = a.timeline
    for ly in scrolling:
        ly.prepare(tl.frame_times(), tl.frame_segments())
    for ly in scrolling:
        if isinstance(ly, Events):
            for i in range(tl.n_frames):
                x, _ = ly.window_data(i)
                assert np.all(x <= 0)
        if isinstance(ly, Scope):
            for i in range(0, tl.n_frames, 7):
                x, _ = ly.window_data(i)
                assert np.nanmax(x, initial=-1) <= 0
    rib = next(ly for ly in scrolling if isinstance(ly, Ribbon))
    stream = [r for r in rib.rows if r.name.startswith("Stream")]
    assert stream and all(r.method == "hold" for r in stream)   # shown from emission on


def test_release_threshold_reproduces_detector(ctx):
    """detect() decides exactly where E <= the displayed threshold has held for hold_s."""
    from biomech.anim.source import release_threshold
    from biomech.timing.ensemble import EnsembleParams
    from biomech.timing.lead import _rest_level
    from biomech.timing.release import ReleaseParams, _decisions, causal_db, detect

    p = ctx.config.params("timing")
    rp, ep = ReleaseParams.from_dict(p.get("release")), EnsembleParams.from_dict(p.get("ensemble"))
    prep = ctx.emg("s01")
    rest = prep.phase("rest")
    level = _rest_level(np.asarray(prep.causal("rest")), rest.mask, rest.fs, ep.envelope_s)
    for ph in ("open_close", "chewing"):
        pe = prep.phase(ph)
        t = np.asarray(pe.t_us, np.int64)
        e = causal_db(t, np.asarray(prep.causal(ph)), pe.mask, level, list(prep.names),
                      rp.window_s)
        thr = release_threshold(t, e, rp)
        fs = 1e6 / np.median(np.diff(t))
        below = np.isfinite(e) & np.isfinite(thr) & (e <= thr)
        mine = t[_decisions(below, max(int(round(rp.hold_s * fs)), 1),
                            int(round(rp.refractory_s * fs)))]
        np.testing.assert_array_equal(mine, detect(t, e, rp))
        assert mine.size > 0


def test_hinge_ghost_turns_about_the_fixed_axis(anims, ctx):
    a = anims["v1_hinge"]
    h = a.clip.extra["hinge"]
    n, s = h["axis_n"], h["axis_s"]
    tab = ctx.table("kinematics", "hinge", ["s01"])
    row = tab[tab.model == "hinge_fit"].iloc[0]
    assert float(h["rms_incisor_mm"]) == pytest.approx(row.rms_incisor_mm)
    np.testing.assert_allclose(n, row[["axis_x", "axis_y", "axis_z"]].to_numpy(float),
                               atol=1e-9)
    ok = np.isfinite(h["ghost"]).all(axis=(1, 2))
    assert ok.mean() > 0.9
    # A rotation about the axis keeps every landmark's distance to it.
    G = h["ghost"][ok]
    r_ghost = np.linalg.norm(np.cross(G - s, n), axis=2)
    r_closed = np.linalg.norm(np.cross(h["closed"] - s, n), axis=1)
    np.testing.assert_allclose(r_ghost, np.broadcast_to(r_closed, r_ghost.shape), atol=1e-6)
    # Residual = measured minus ghost at the incisor; the synthetic jaw is near a hinge.
    np.testing.assert_allclose(h["residual"][ok], a.clip.kin["incisor"][ok] - G[:, 0])
    assert np.nanmedian(h["residual_mm"]) < 5.0


def test_v3_plays_both_phases_in_step(anims):
    from biomech.anim.layers import Scope, Shifted

    a = anims["v3_release"]
    tl = a.timeline
    (pa, a0, _), (pb, b0, _) = a.clip.segments[:2]
    assert len(tl.segments) == 1 and tl.segments[0][0] == a0
    assert tl.duration_s == pytest.approx(4.0 / a.speed)
    comp = a.compose()
    right = [ly for ly in comp.layers if isinstance(ly, Shifted)]
    assert right and all(ly.offset_us == pytest.approx(b0 - a0) for ly in right)
    sc = next(ly for ly in right if isinstance(ly.inner, Scope))
    t = tl.frame_times([5])
    sc.prepare(t, np.zeros(1, int))
    want = timebase.resample(sc.inner.t_src, sc.inner.y[:, 0], t + (b0 - a0), max_gap_s=0.25)
    np.testing.assert_allclose(sc.inner.now[:, 0], want)


def test_tally_counts_events_in_play_order():
    from biomech.anim.layers import Tally

    segs = ((100.0, 200.0), (0.0, 50.0))                     # played out of time order
    ev = {"e": (np.array([10.0, 40.0, 120.0, 150.0, 300.0]), np.array([1, 2, 3, 4, 5.0]))}
    t = np.array([110.0, 160.0, 200.0, 5.0, 45.0])
    seg = np.array([0, 0, 0, 1, 1])
    ta = Tally("x", [(0, 0, "left", lambda s: f"{s['e_n']}", "label", "ink")],
               t_ref=lambda tt, sg: tt * 0, events=ev, segments=segs)
    ta.prepare(t, seg)
    # Segment 0 holds 120 and 150 (300 is outside every segment); segment 1 adds 10 and 40.
    assert [s[0] for s in ta.strings] == ["0", "2", "2", "2", "4"]
    np.testing.assert_array_equal(ta.extra["e_last"][1:], [4, 4, 4, 2])
    assert ta.extra["e_median"][4] == pytest.approx(2.5)


def test_custom_captions_and_manifest(ctx):
    from biomech.anim import make
    from biomech.anim.layers import Caption

    a = make(ctx, "s01", "v3_release", None, 0.0, 2.0, fps=10,
             captions=((0.0, 1.0, "first cue"), (1.0, 99.0, "second cue")))
    cap = next(ly for ly in a.compose().layers if isinstance(ly, Caption))
    tl = a.timeline
    cap.prepare(tl.frame_times([0, 15]), tl.frame_segments([0, 15]))
    assert [s[0] for s in cap.strings] == ["first cue", "second cue"]
    assert a.manifest()["captions"][0] == [0.0, 1.0, "first cue"]


def test_filmstrip_png_tiles_frames(anims, tmp_path):
    from PIL import Image

    from biomech.anim.render_video import filmstrip_png

    p = filmstrip_png(anims["v1_hinge"], tmp_path / "f.png", n=6, cols=3, width=320, gap=8)
    with Image.open(p) as im:
        assert im.size == (3 * 320 + 4 * 8, 2 * 180 + 3 * 8)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_preview_frames(anims, layout):
    from biomech.anim.render_plotly import preview_figure

    fig = preview_figure(anims[layout], fps=5, max_frames=4)
    assert len(fig.frames) == 4
    n = len(fig.frames[0].data)
    assert n > 0 and all(len(f.data) == n for f in fig.frames)


def test_missing_data_is_named(ctx):
    import dataclasses

    from biomech.anim import make

    an = make(ctx, "s01", "v1_hinge", None, 0.0, 1.0, fps=10)
    bare = an.with_(clip=dataclasses.replace(an.clip, extra={}))
    with pytest.raises(ValueError, match="hinge"):
        bare.compose()


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not installed")
def test_cli_renders_v2_with_banner_clip_and_filmstrip(ctx, tmp_path):
    from click.testing import CliRunner

    from biomech.cli import main

    out = tmp_path / "videos" / "v2.mp4"
    r = CliRunner().invoke(main, ["animate", "--config", str(ctx.config.source), "--pid", "s01",
                                  "--layout", "v2_intent", "--t0", "0", "--t1", "1.5", "--fps",
                                  "10", "--size", "320", "-j", "1", "--out", str(out)])
    assert r.exit_code == 0, r.output
    m = json.loads(out.with_suffix(".json").read_text())
    assert m["banner"] == "simulated real time, no device" and m["speed"] == 1.0
    assert m["labels"] == ["open_close"]
    names = sorted(p.name for p in out.parent.iterdir())
    assert names == ["v2.clip.json", "v2.clip.npz", "v2.json", "v2.mp4", "v2.vtt",
                     "v2_filmstrip.png"]                  # .vtt: the captions
    assert m["filmstrip"] == "v2_filmstrip.png"
    assert "s01" not in (out.parent / "v2.clip.json").read_text()
