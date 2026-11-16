# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.7: the three new video layouts, ``supplementary_videos`` in paper sets and the
check-outputs video check (synthetic data, fast)."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

NEW = ("workspace3d", "emg_spectrogram", "onset_rules")


# -- registration -----------------------------------------------------------------------------
def test_layouts_are_registered_with_hooks_and_names():
    from biomech.anim import LAYOUTS, default_phases, default_speed
    from biomech.anim import layouts as Lo
    from biomech.viz.views import vocab as V

    for k in NEW:
        assert k in LAYOUTS and k in Lo.STILLS and k in Lo.PREVIEWS
        assert Lo.STILL_WIDTH[k] >= 1280                     # paper-figure filmstrip
        assert V.term("video_layout", k) != k
    assert "workspace3d" in Lo.COHORT_LAYOUTS and "workspace3d" in Lo.COHORT_LOADERS
    assert Lo.EXTRAS["emg_spectrogram"] == ("emg_spectro",)
    assert Lo.EXTRAS["onset_rules"] == ("onset_rules",)
    assert default_phases("onset_rules") == ["open_close"]
    assert default_speed("workspace3d") == 1.0
    assert Lo.video_stem("workspace3d", None, ["open_close"], speed=4.0) == "workspace3d_4x"


def test_cli_layout_choices_include_the_new_layouts():
    from biomech.anim import LAYOUTS
    from biomech.cli import main

    opt = next(p for p in main.commands["animate"].params if p.name == "layout")
    assert list(opt.type.choices) == list(LAYOUTS)
    assert any(p.name == "variant" for p in main.commands["animate"].params)


# -- layers -----------------------------------------------------------------------------------
def test_spin_turns_once_per_period():
    from biomech.anim.layers import Spin

    s = Spin("x:0", 10e6, azim0=-58.0, elev=18.0, elev_amp=5.0)
    s.prepare(np.array([0.0, 2.5e6, 5e6]), np.zeros(3, int))
    assert np.allclose(s.azim, [-58.0, 32.0, 122.0])
    assert np.allclose(s.elev_, [18.0, 23.0, 18.0])


def test_scroll_image_shows_the_window_columns():
    from biomech.anim.layers import ScrollImage

    t = np.arange(100) * 1e5                                  # one column per 0.1 s
    img = np.random.default_rng(0).random((4, 100, 3))
    s = ScrollImage("x:0", t, img, (0.0, 4.0), (-1.0, 0.5))
    s.prepare(np.array([5e6]), np.zeros(1, int))
    data, ext = s.window_data(0)
    assert data.shape[1] == 16                                # 4.0 s .. 5.5 s inclusive
    assert np.allclose(data, img[:, 40:56])
    assert ext[0] == pytest.approx(-1.05) and ext[1] == pytest.approx(0.55)


def test_segment_events_appear_when_the_cursor_passes_them():
    from biomech.anim.layers import SegmentEvents

    ev = SegmentEvents("x:0", np.array([1e6, np.nan]), np.array([0.0, 0.0]),
                       np.array([1.2e6, 5e6]))
    ev.prepare(np.array([0.9e6, 1.1e6, 5.5e6]), np.array([0, 0, 1]))
    assert ev.point(0) == ([], [])
    assert ev.point(1) == ([pytest.approx(-200.0)], [0.0])   # ms re the reference
    assert ev.point(2) == ([], [])                            # segment without an event


def test_progress_follows_overlapping_segments():
    from biomech.anim.layouts import _Progress
    from biomech.anim.timeline import Timeline

    tl = Timeline(((0.0, 2e6), (1e6, 3e6)), fps=10, speed=1.0)
    p = _Progress("h", tl)
    p.prepare(np.array([1.5e6, 1.5e6]), np.array([0, 1]))
    assert np.allclose(p.f, [0.375, 0.625])                  # 1.5 s and 2.5 s of 4 s


# -- synthetic animations ---------------------------------------------------------------------
def _workspace_anim(theme="dark"):
    from biomech.anim.layouts import Animation
    from biomech.anim.source import CohortClip
    from biomech.viz.views import kin_r8 as K

    rng = np.random.default_rng(1)
    pts = [rng.normal([-3, 0, -15], [4, 5, 10], (60, 3)) for _ in range(3)]
    A = np.vstack(pts)
    ws = {"points": pts, "n_points": np.array([60.0] * 3), "cage": K.hull_cage(A),
          "hull_volume_mm3": np.asarray(5000.0), "box_lo": np.array([0.0, -10.0, -35.0]),
          "box_hi": np.array([12.0, 10.0, 0.0]), "posterior_stored": np.asarray(False),
          "outside_pct": np.array([40.0, 55.0, 60.0]), "p_design": np.asarray(95.0),
          "hull_rings": np.asarray(5)}
    clip = CohortClip("workspace", ["P1", "P2", "P3"], [], np.zeros((0, 101, 3)),
                      np.zeros((0, 101)), meta={"snapshot": "r0", "variant": "primary"},
                      extra={"workspace": ws})
    return Animation(clip, "workspace3d", fps=4, speed=4.0, theme=theme)


def _onset_anim():
    from biomech.anim.layouts import Animation
    from biomech.anim.source import Clip

    fs = 2000.0
    t = np.arange(0, 20, 1 / fs) * 1e6
    rng = np.random.default_rng(2)
    tm = np.array([4e6, 9e6, 14e6])
    d = {"t_us": t, "emg_uv": rng.normal(0, 20, t.size), "tkeo_db": rng.normal(-5, 3, t.size),
         "muscle": np.asarray(4), "threshold": np.asarray(1.0), "k_sd": np.asarray(3.0),
         "burst_min_db": np.asarray(6.0), "pre_s": np.asarray(0.6), "post_s": np.asarray(0.2),
         "margin_s": np.asarray(0.15), "cycle": np.arange(3.0), "t_mech_us": tm,
         "t_omega_us": tm, "t_on_first": tm - np.array([400e3, 80e3, np.nan]),
         "t_on_drive": tm - np.array([50e3, 80e3, 30e3]),
         "usable_first": np.array([1.0, 1.0, 0.0]), "usable_drive": np.array([1.0, 1.0, 1.0]),
         "lead_first_ms": np.array([400.0, 80.0, np.nan]),
         "lead_drive_ms": np.array([50.0, 80.0, 30.0]), "burst_db": np.array([12.0, 9.0, 8.0]),
         "active_at_start": np.zeros(3), "n_cycles_task": np.asarray(3)}
    tk = np.arange(0, 20, 0.01) * 1e6
    kin = {"t_us": tk, "theta_deg": 10 * np.clip(np.sin(tk * 1e-6), 0, None)}
    clip = Clip("p", "Participant 1", [("open_close", 0.0, 20e6)], kin=kin,
                extra={"onset_rules": d})
    return Animation(clip, "onset_rules", fps=4, speed=1.0, theme="light")


def _emg_anim():
    from biomech.anim.layouts import Animation
    from biomech.anim.source import Clip

    t = np.arange(0, 10, 1e-3) * 1e6
    rng = np.random.default_rng(3)
    tc = np.arange(0, 10, 0.016) * 1e6
    f = np.arange(20, 450, 15.6)
    d = {"t_us": t, "x": rng.normal(0, 50, (t.size, 6)), "amp_uv": np.full(6, 100.0),
         "muscles": np.arange(6, dtype=np.int16), "closers_t_us": tc, "openers_t_us": tc,
         "closers_db": rng.normal(-10, 5, (f.size, tc.size)),
         "openers_db": rng.normal(-12, 5, (f.size, tc.size)), "freq_hz": f,
         "band_hz": np.array([20.0, 450.0]), "env_ms": np.asarray(150.0),
         "stft_s": np.asarray(0.064), "stft_overlap": np.asarray(0.75),
         "display_hz": np.asarray(1000.0), "row_pct": np.asarray(99.5)}
    clip = Clip("p", "Participant 1", [("chewing", 2e6, 6e6)], extra={"emg_spectro": d})
    return Animation(clip, "emg_spectrogram", fps=4, speed=1.0, theme="dark")


@pytest.mark.parametrize("make_anim", [_workspace_anim, _onset_anim, _emg_anim])
def test_new_layouts_render_frames_and_captions(make_anim):
    from biomech.anim.render_video import caption_cues

    a = make_anim()
    sc = a.scene((320, 180), stamp=False)
    try:
        sc.prepare(np.array([0, a.timeline.n_frames - 1]))
        f0, f1 = sc.frame(0), sc.frame(1)
    finally:
        sc.close()
    assert f0.shape == (180, 320, 3) and not np.array_equal(f0, f1)
    cues = caption_cues(a)
    assert cues and all(txt for _, _, txt in cues)


def test_onset_rules_plays_one_segment_per_cycle_with_stored_leads():
    from biomech.anim.r8_videos import _cycle_line

    a = _onset_anim()
    tl = a.timeline
    assert len(tl.segments) == 3 and tl.labels[0] == "Cycle 1 of 3"
    lo, hi = tl.segments[1]
    assert lo == pytest.approx(9e6 - 0.75e6) and hi == pytest.approx(9e6 + 0.35e6)
    d = a.clip.extra["onset_rules"]
    line = _cycle_line(d, 0, 3)
    assert "400 ms before" in line and "50 ms before" in line
    assert "no onset" in _cycle_line(d, 2, 3)


def test_workspace_rotates_and_clip_files_round_trip(tmp_path):
    from biomech.anim.clipfile import load_clip_file, save_clip

    a = _workspace_anim()
    comp = a.compose()
    lims = comp.meta["lims"]
    assert len({round(h - lo, 6) for lo, h in lims}) == 1     # a cube: mm for mm when turning
    npz, js = save_clip(a, tmp_path / "workspace3d.mp4")
    b = load_clip_file(js)
    assert b.layout == "workspace3d"
    for x, y in zip(a.clip.extra["workspace"]["points"], b.clip.extra["workspace"]["points"]):
        np.testing.assert_array_equal(x, y)
    assert "variant" in a.manifest() and a.manifest()["snapshot"] == "r0"


# -- paper sets ---------------------------------------------------------------------------------
def _pset(tmp_path, videos: str, snapshot="r9", variant=None):
    text = (f"snapshot: {snapshot}\n" + (f"variant: {variant}\n" if variant else "")
            + "out_dir: out\nitems:\n  - {id: table_requirements, label: tabReq}\n" + videos)
    p = tmp_path / "paper_test.yaml"
    p.write_text(text, encoding="utf-8")
    from biomech import papers

    return papers.load(p)


def test_paper_set_reads_supplementary_videos(tmp_path):
    ps = _pset(tmp_path, "supplementary_videos:\n"
               "  - {layout: workspace3d, label: videoWorkspace}\n"
               "  - {layout: v3_release, pid: s05, phase: [open_close, chewing], speed: 0.5,"
               " theme: light, label: videoRelease}\n")
    assert [v.label for v in ps.videos] == ["videoWorkspace", "videoRelease"]
    v = ps.videos[1]
    assert v.phases == ("open_close", "chewing") and v.speed == 0.5 and v.theme == "light"
    assert "s05" not in v.describe()
    assert _pset(tmp_path, "").videos == []


@pytest.mark.parametrize("entry, msg", [
    ("  - {layout: workspace3d}\n", "label"),
    ("  - {layout: workspace3d, label: tabReq}\n", "duplicate"),
    ("  - {layout: workspace3d, label: v1, size: 720p}\n", "label"),
    ("  - {layout: workspace3d, label: vid, size: 720p}\n", "unknown keys"),
    ("  - {layout: onset_rules, label: vid, theme: blue}\n", "theme"),
    ("  - {label: vid}\n", "layout"),
])
def test_paper_set_rejects_bad_video_entries(tmp_path, entry, msg):
    with pytest.raises(ValueError, match=msg):
        _pset(tmp_path, "supplementary_videos:\n" + entry)


def _render_stub(d, stem, **meta):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.mp4").write_bytes(b"")
    (d / f"{stem}.json").write_text(json.dumps(meta), encoding="utf-8")


def test_find_video_and_store_problems(tmp_path):
    from biomech import papers

    ps = _pset(tmp_path, "supplementary_videos:\n"
               "  - {layout: onset_rules, pid: s12, label: videoOnset}\n"
               "  - {layout: workspace3d, label: videoWorkspace}\n", variant="reporting_population")
    vids = tmp_path / "videos"
    pseud = {"s12": 11}.get
    v_on, v_ws = ps.videos
    assert papers.video_stem_for(v_on, pseud) == "onset_rules_P11_open_close_0.25x"
    assert papers.find_video(v_on, [vids], pseud) == (None, None)
    _render_stub(vids, "onset_rules_P11_open_close_0.25x_720p", layout="onset_rules",
                 speed=0.25, theme="dark", snapshot="r9", variant="reporting_population")
    mp4, meta = papers.find_video(v_on, [vids], pseud)
    assert mp4.name.endswith("_720p.mp4") and papers.video_store_problems(meta, ps) == []
    _render_stub(vids, "workspace3d_1x", layout="workspace3d", speed=1.0, snapshot="r8",
                 variant="primary")
    _, meta = papers.find_video(v_ws, [vids], pseud)
    probs = papers.video_store_problems(meta, ps)
    assert any("snapshot r8" in p for p in probs) and any("primary" in p for p in probs)


def test_check_outputs_reports_missing_and_foreign_videos(tmp_path, monkeypatch):
    from biomech import papers
    from biomech.check import outputs as O

    ps = _pset(tmp_path, "supplementary_videos:\n"
               "  - {layout: onset_rules, pid: s12, label: videoOnset}\n"
               "  - {layout: workspace3d, label: videoWorkspace}\n"
               "  - {layout: emg_spectrogram, pid: s12, label: videoEmg}\n"
               "  - {layout: no_such_layout, label: videoBad}\n")
    _render_stub(tmp_path / "videos", "workspace3d_1x", layout="workspace3d", speed=1.0,
                 snapshot="r9", variant="primary")
    _render_stub(tmp_path / "videos", "emg_spectrogram_P11_chewing", layout="emg_spectrogram",
                 speed=0.5, snapshot="live", variant="primary")
    cfg = SimpleNamespace(source=tmp_path / "study.yaml", out_dir=tmp_path)
    ctx = SimpleNamespace(config=cfg, pids=lambda: ["s12"], pseudonym=lambda p: 11)

    @contextlib.contextmanager
    def same(c, p, snapshot=None):
        yield c

    monkeypatch.setattr(papers, "study_paper_sets", lambda src: [ps])
    monkeypatch.setattr(papers, "paper_context", same)
    rep = O.Report("t")
    O.check_paper_videos(rep, ctx)
    st = {i.name.split("supplementary video ")[1].split(" ")[0]: (i.status, i.detail)
          for i in rep.items}
    assert st["videoOnset"][0] == "warn" and "not rendered" in st["videoOnset"][1]
    assert st["videoWorkspace"][0] == "pass"
    assert st["videoEmg"][0] == "fail" and "snapshot live" in st["videoEmg"][1]
    assert st["videoBad"][0] == "fail" and "unknown layout" in st["videoBad"][1]


def test_gallery_lists_the_paper_set_videos(tmp_path):
    from biomech.viz import gallery

    ps = _pset(tmp_path, "supplementary_videos:\n"
               "  - {layout: workspace3d, label: videoWorkspace}\n"
               "  - {layout: onset_rules, pid: s12, label: videoOnset}\n")
    _render_stub(tmp_path / "videos", "workspace3d_1x", layout="workspace3d", speed=1.0,
                 snapshot="r9")
    listed = gallery.listed_videos(ps, [tmp_path / "videos"], {"s12": 11}.get)
    assert [e["status"] for e in listed] == ["rendered", "not rendered"]
    out = tmp_path / "gal"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    html = gallery.write_html(out, title="g", listed=listed).read_text(encoding="utf-8")
    assert "videoWorkspace" in html and "not rendered" in html and "s12" not in html


def test_paper_set_reads_supplementary_interactive(tmp_path):
    ps = _pset(tmp_path, "supplementary_videos:\n  - {layout: workspace3d, label: videoWs}\n"
               "supplementary_interactive:\n"
               "  - {id: kin_workspace3d, label: interactiveWs}\n"
               "  - {id: kin_condyle3d, label: interactiveCp, form: arrows}\n")
    assert [(i.view_id, i.label) for i in ps.interactive] == [
        ("kin_workspace3d", "interactiveWs"), ("kin_condyle3d", "interactiveCp")]
    assert ps.interactive[1].params == {"form": "arrows"}
    for bad in ("  - {id: kin_workspace3d, label: videoWs}\n",       # taken by a video
                "  - {id: kin_workspace3d, label: tabReq}\n",        # taken by an item
                "  - {label: interactiveX}\n"):
        with pytest.raises(ValueError, match="supplementary_interactive"):
            _pset(tmp_path, "supplementary_videos:\n  - {layout: workspace3d, label: videoWs}\n"
                  "supplementary_interactive:\n" + bad)
