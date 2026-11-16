# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Animated visualizer (an11): timeline, layer state, video, timecode, preview, LaTeX."""

from __future__ import annotations

import dataclasses
import shutil

import numpy as np
import pytest

from biomech import timebase
from biomech.anim import timecode
from biomech.anim.layers import (OUTLINE, Pose, Ribbon, RibbonRow, Scope, Trail,
                                 categorical_rgba)
from biomech.anim.timeline import Timeline

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")


# -- timeline ----------------------------------------------------------------------------------
def test_timeline_maps_are_inverse():
    tl = Timeline(((10e6, 13e6), (20e6, 21.5e6)), fps=30, speed=0.5, labels=("a", "b"))
    assert tl.duration_s == pytest.approx(9.0)
    assert tl.n_frames == 270
    v = np.arange(tl.n_frames) / tl.fps
    t = tl.to_master(v)
    np.testing.assert_allclose(tl.to_video(t), v, atol=1e-9)
    # Master -> video -> master inside the (half-open) segments; NaN in the cut.
    tm = np.r_[np.linspace(10e6, 13e6, 50, endpoint=False), np.linspace(20e6, 21.5e6, 50)]
    np.testing.assert_allclose(tl.to_master(tl.to_video(tm)), tm, rtol=0, atol=1e-3)
    assert np.isnan(tl.to_video(np.array([15e6])))[0]
    assert tl.frame_of(np.array([15e6]))[0] == -1
    np.testing.assert_array_equal(tl.frame_of(tl.frame_times()), np.arange(tl.n_frames))
    # The cut: frame 180 (video 6 s) is the first of segment b.
    seg = tl.frame_segments()
    assert seg[179] == 0 and seg[180] == 1 and tl.frame_times([180])[0] == 20e6
    chunks = tl.chunks(7)
    assert sum(len(c) for c in chunks) == tl.n_frames and chunks[0].start == 0


def test_timeline_validates():
    with pytest.raises(ValueError):
        Timeline(((5.0, 5.0),))
    with pytest.raises(ValueError):
        Timeline(((0.0, 1e6),), speed=0)


# -- timecode ----------------------------------------------------------------------------------
@pytest.mark.parametrize("bg", [(13, 17, 23), (255, 255, 255)])
def test_timecode_roundtrip(bg):
    rng = np.random.default_rng(0)
    f = np.zeros((180, 320, 3), np.uint8) + np.array(bg, np.uint8)
    for t in (0, 1_926_643_134, 2 ** 40 - 1, 81_583_333):
        timecode.stamp(f, t, bg=np.array(bg))
        assert timecode.read_timecode(f) == t
        noisy = np.clip(f.astype(int) + rng.integers(-12, 13, f.shape), 0, 255).astype(np.uint8)
        assert timecode.read_timecode(noisy) == t
    assert timecode.read_timecode(np.zeros((180, 320, 3), np.uint8)) is None
    g = f.copy()
    timecode.stamp(g, 12345, bg=np.array(bg))
    b = timecode.block(180)
    x0 = 320 - timecode.N_BITS * b
    g[-b:, x0 + 30 * b:x0 + 31 * b] = 255 - g[-b:, x0 + 30 * b:x0 + 31 * b]   # flip a bit
    assert timecode.read_timecode(g) is None                                  # checksum


# -- layer state equals directly interpolated data ----------------------------------------------
def _path(n=200, fs=40.0, seed=0):
    rng = np.random.default_rng(seed)
    t = 1e9 + np.arange(n) * 1e6 / fs
    P = np.cumsum(rng.normal(0, 0.5, (n, 3)), axis=0)
    return t, P


def test_trail_state_and_window():
    t, P = _path()
    sp = np.linalg.norm(np.gradient(P, axis=0), axis=1)
    tr = Trail("p:0", t, P, (0, 2), tau_s=0.5, role="entity:incisor", speed=sp, vmax=2.0)
    tq = np.linspace(t[30], t[-5], 37)
    tr.prepare(tq, np.zeros(len(tq), int))
    np.testing.assert_allclose(tr.head_p, timebase.resample(t, P, tq))
    for i in range(len(tq)):
        pts, tt, s = tr.trail(i)
        # The trail covers exactly [t - tau, t]: interpolated ends, samples strictly inside.
        assert tt[0] == pytest.approx(tq[i] - 0.5e6) and tt[-1] == pytest.approx(tq[i])
        assert np.all((tt[1:-1] > tq[i] - 0.5e6) & (tt[1:-1] < tq[i]))
        assert np.all(np.diff(tt) >= 0)
        np.testing.assert_allclose(pts[0], timebase.resample(t, P, [tq[i] - 0.5e6])[0])
        np.testing.assert_allclose(pts[1:-1], P[(t > tq[i] - 0.5e6) & (t < tq[i])])
        assert len(s) == len(pts)
    ser = tr.series(3)
    assert [x.kind for x in ser] == ["colorline", "scatter"]
    np.testing.assert_allclose(ser[1].x, tr.head_p[3, [0]])


def test_trail_fades_with_age():
    t, P = _path()
    tr = Trail("p:0", t, P, (0, 1, 2), tau_s=1.0)
    tr.prepare(np.array([t[100]]), np.zeros(1, int))
    tr._base, tr._ramp = np.ones(3), ("000000", "FFFFFF")
    pts, tt, sp = tr.trail(0)
    a = tr.colors(tt, sp, t[100])[:, 3]
    assert np.all(np.diff(a) >= 0) and a[-1] > 0.9 and a[0] < 0.2


def test_pose_state_equals_interpolated_landmarks():
    t, P = _path(120)
    th = np.radians(np.linspace(0, 20, len(t)))
    R = np.stack([[[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
                  for a in th])
    kin = {"t_us": t, "R": R, "omega": np.tile([0, 20.0, 0], (len(t), 1)),
           "incisor_vel": np.tile([0, 0, -10.0], (len(t), 1))}
    for k, n in enumerate(OUTLINE):
        kin[n] = P + k * 10.0
    pose = Pose("p:0", kin)
    tq = np.linspace(t[5], t[-5], 11)
    pose.prepare(tq, np.zeros(len(tq), int))
    for k, n in enumerate(OUTLINE):
        np.testing.assert_allclose(pose.outline[:, k], timebase.resample(t, kin[n], tq))
    np.testing.assert_allclose(pose.R, timebase.resample_rotations(t, R, tq), atol=1e-12)
    # Pure rotation about y through the incisor-offset point: the helical axis is along y.
    n = np.diff(pose.axis_seg, axis=1)[:, 0]
    n /= np.linalg.norm(n, axis=1)[:, None]
    np.testing.assert_allclose(np.abs(n[:, 1]), 1.0, atol=1e-9)


def test_scope_and_ribbon_windows():
    t, P = _path(400)
    sc = Scope("p:0", t, P[:, :2], ["primary", "secondary"], window=(-2.0, 0.5))
    tq = np.array([t[150], t[300]])
    sc.prepare(tq, np.zeros(2, int))
    for i in range(2):
        x, y = sc.window_data(i)
        assert x.min() >= -2.0 and x.max() <= 0.5
        np.testing.assert_allclose(y, P[(t >= tq[i] - 2e6) & (t <= tq[i] + 0.5e6), :2])
    np.testing.assert_allclose(sc.now, timebase.resample(t, P[:, :2], tq))
    lab = (np.arange(len(t)) // 40 % 3).astype(float)
    rb = Ribbon("p:0", [RibbonRow("Label", t, lab, categorical_rgba({0: "grey", 1: "orange",
                                                                     2: "blue"}, "dark"))],
                window=(-1.0, 1.0), cols=50)
    rb.prepare(tq, np.zeros(2, int))
    want = timebase.resample(t, lab, tq[0] + np.linspace(-1, 1, 50) * 1e6, method="nearest")
    np.testing.assert_array_equal(rb.vals[0, 0], want)


# -- store-backed: one synthetic participant ----------------------------------------------------
@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    from biomech.config import load_config
    from biomech.io.context import DataContext
    from biomech.pipeline import run
    from biomech.testing.synth import Spec, write_study

    root = tmp_path_factory.mktemp("anim")
    study, _ = write_study(root, [Spec(pid="s01", emg_fs=2000, seed=1)])
    cfg = load_config(study)
    s = run(cfg, ["kinematics", "emg", "imu", "labels"], None, n_jobs=1)
    assert s.n_failed == 0
    c = DataContext(cfg)
    yield c
    c.close()


def _anim(ctx, layout="multimodal", **kw):
    from biomech.anim import make

    return make(ctx, "s01", layout, "chewing", 0.0, kw.pop("t1", 1.0), speed=0.5, fps=10, **kw)


def test_clip_has_every_modality_and_only_the_pseudonym(ctx):
    a = _anim(ctx)
    c = a.clip
    assert set(c.modalities()) >= {"kin", "emg", "imu", "labels", "optical"}
    assert c.label == "Participant 1" and c.previews is None
    assert a.timeline.n_frames == 20 and a.timeline.duration_s == pytest.approx(2.0)
    texts = []
    sc = a.scene(640)
    try:
        sc.prepare([0, 10])
        for d in sc.regions:
            texts += [t.get_text() for t in d["fig"].findobj(lambda o: hasattr(o, "get_text"))]
        img = sc.frame(1)
    finally:
        sc.close()
    assert img.shape == (360, 640, 3) and img.dtype == np.uint8
    assert not any("s01" in t for t in texts)
    assert timecode.read_timecode(img) == int(round(a.timeline.frame_times([10])[0]))


def test_marker_pixel_position_matches_data(ctx):
    """The incisor head marker of the sagittal plane sits where the data say."""
    a = _anim(ctx)
    sc = a.scene(960, stamp=False)
    try:
        sc.prepare([5])
        trail = next(ly for ly in sc.comp.layers if isinstance(ly, Trail)
                     and ly.panel == "planes:0" and ly.role == "entity:incisor")
        want = timebase.resample(a.clip.kin["t_us"], a.clip.kin["incisor"], sc.times)[0]
        np.testing.assert_allclose(trail.head_p[0], want)
        sc.frame(0)
        x, y = trail._dot.get_data()
        assert (x[0], y[0]) == pytest.approx((want[0], want[2]))
    finally:
        sc.close()


def test_missing_modalities_reflow(ctx):
    from biomech.anim.layouts import HEADER_H, KEY_H, Animation

    a = _anim(ctx)
    full = {r.name: r.box for r in a.compose().regions}
    assert {"emg", "angle", "ribbon", "jaw", "planes"} <= set(full)
    c = dataclasses.replace(a.clip, imu=None, labels=None, optical=None)
    comp = Animation(c, "multimodal", a.fps, a.speed).compose()
    names = [r.name for r in comp.regions]
    assert "ribbon" not in names and "emg" in names and "angle" in names
    angle = next(r for r in comp.regions if r.name == "angle").spec.panels[0]
    assert [s.label for s in angle.series if s.label] == ["Optical (reference)"]
    c2 = dataclasses.replace(c, emg=None)
    comp2 = Animation(c2, "multimodal", a.fps, a.speed).compose()
    right = [r for r in comp2.regions if r.align == "right"]
    assert [r.name for r in right] == ["angle"]
    body_top, body_h = HEADER_H + 0.012, 1 - HEADER_H - 0.024 - KEY_H
    assert right[0].box[1] == pytest.approx(body_top)
    assert right[0].box[3] == pytest.approx(body_h)               # it takes the whole column
    # Every layout still renders a frame without the dropped modalities.
    sc = Animation(c2, "multimodal", a.fps, a.speed).scene(480)
    try:
        sc.prepare([0])
        assert sc.frame(0).shape == (270, 480, 3)
    finally:
        sc.close()
    with pytest.raises(ValueError):
        Animation(dataclasses.replace(c2, kin=None, previews=None), "kinematics").compose()


@pytest.mark.parametrize("layout", ["kinematics", "tracking", "emg"])
def test_other_layouts_render(ctx, layout):
    a = _anim(ctx, layout)
    sc = a.scene(480)
    try:
        sc.prepare([0, 19])
        f = sc.frame(1)
    finally:
        sc.close()
    assert f.shape == (270, 480, 3) and f.std() > 5


def test_cohort_layout(ctx):
    from biomech.anim import make

    a = make(ctx, None, "cohort", task="open_close", speed=0.5, fps=10)
    assert a.clip.labels == ["P1"] and a.clip.means.shape[1:] == (101, 3)
    sc = a.scene(480)
    try:
        sc.prepare([0, 5])
        assert sc.frame(1).shape == (270, 480, 3)
    finally:
        sc.close()


@needs_ffmpeg
def test_short_render_probe_and_sync(ctx, tmp_path):
    from biomech.anim.render_video import probe, render_video, verify_sync

    a = _anim(ctx)
    seen = []
    res = render_video(a, tmp_path / "v.mp4", size=320, n_jobs=1,
                       on_progress=lambda d, n: seen.append((d, n)))
    p = probe(res.path)
    assert p["codec"] == "h264" and p["pix_fmt"] == "yuv420p"
    assert p["n_frames"] == 20 and p["duration_s"] == pytest.approx(2.0, abs=0.05)
    assert p["width"] % 2 == 0 and p["height"] % 2 == 0 and (p["width"], p["height"]) == (320,
                                                                                         180)
    assert seen[-1] == (20, 20)
    s = verify_sync(res.path, a, n_samples=6)
    assert s["unreadable"] == 0 and s["max_err_us"] == 0
    import json

    m = json.loads(res.manifest.read_text())
    assert m["with_camera"] is False and m["pseudonym"] == "Participant 1"
    assert m["expected_duration_s"] == pytest.approx(2.0) and len(m["frames"]) > 2


@needs_ffmpeg
def test_chunked_equals_single_process(ctx, tmp_path):
    from biomech.anim.render_video import frame_md5, render_video

    a = _anim(ctx)
    one = render_video(a, tmp_path / "one.mp4", size=320, n_jobs=1, crf=0, frame_hashes=True)
    par = render_video(a, tmp_path / "par.mp4", size=320, n_jobs=2, chunk_frames=7, crf=0,
                       frame_hashes=True)
    assert par.n_chunks >= 3
    assert one.frame_hashes == par.frame_hashes and len(one.frame_hashes) == 20
    assert frame_md5(one.path) == frame_md5(par.path)


@needs_ffmpeg
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cancel_leaves_no_file(ctx, tmp_path, n_jobs):
    from biomech.anim.render_video import Cancelled, render_video

    a = _anim(ctx, t1=3.0)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    out = tmp_path / "c.mp4"
    with pytest.raises(Cancelled):
        render_video(a, out, size=320, n_jobs=n_jobs, chunk_frames=10, cancel=cancel)
    assert not out.exists() and not out.with_suffix(".json").exists()
    assert list(tmp_path.iterdir()) == []                     # no temporary leftovers


def test_plotly_preview_frames(ctx):
    from biomech.anim.render_plotly import preview_figure

    a = _anim(ctx)
    fig = preview_figure(a, fps=5, max_frames=8)
    assert len(fig.frames) == 8
    n = len(fig.frames[0].data)
    assert n > 0 and all(len(f.data) == n for f in fig.frames)
    assert fig.layout.sliders and fig.layout.updatemenus
    assert "Participant 1" in fig.frames[3].layout.title.text


@pytest.mark.skipif(shutil.which("lualatex") is None or shutil.which("pdfinfo") is None,
                    reason="no LuaLaTeX/pdfinfo")
def test_filmstrip_compiles_within_width(ctx, tmp_path):
    from biomech.anim.render_latex import filmstrip, filmstrip_spec
    from biomech.viz.export import check_figure_size

    a = _anim(ctx, "kinematics", t1=2.0)
    spec = filmstrip_spec(a, n=4)
    assert len(spec.panels) == 4 and spec.panels[0].title.startswith("t = ")
    tex = filmstrip(a, tmp_path, n=4, fig_id=spec.id)
    assert tex.is_file() and "\\begin{axis}" in tex.read_text()
    probs = check_figure_size(tmp_path, spec)
    if any("pgfplots.sty' not found" in p for p in probs):
        pytest.skip("pgfplots not installed")
    assert not probs, probs


def test_animate_frames_snippet(ctx, tmp_path):
    from biomech.anim.render_latex import animate_frames

    a = _anim(ctx, "kinematics")
    tex = animate_frames(a, tmp_path, fps=5, size=320)
    pngs = sorted((tmp_path / "frames").glob("frame-*.png"))
    assert len(pngs) == 10
    s = tex.read_text()
    assert "\\animategraphics" in s and "{frames/frame-}{0}{9}" in s


def test_cli_animate(ctx, tmp_path):
    pytest.importorskip("click")
    if not FFMPEG:
        pytest.skip("ffmpeg not installed")
    from click.testing import CliRunner

    from biomech.cli import main

    cfg = ctx.config
    out = tmp_path / "cli.mp4"
    r = CliRunner().invoke(main, ["animate", "--config", str(cfg.source), "--pid", "s01",
                                  "--layout", "kinematics", "--phase", "chewing", "--t0", "0",
                                  "--t1", "0.6", "--fps", "10", "--size", "320", "-j", "1",
                                  "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert out.is_file() and out.with_suffix(".json").is_file()


def test_viewer_video_job_renders_and_cancels(ctx, tmp_path):
    if not FFMPEG:
        pytest.skip("ffmpeg not installed")
    from biomech.anim.render_video import probe
    from biomech.viewer.pages.animation import _make_kw
    from biomech.viewer.videos import VideoJob

    kw = _make_kw("kinematics", "s01", ["chewing"], 0.0, 0.6, 1.0, 10, False, "light")
    job = VideoJob()
    out = tmp_path / "job.mp4"
    assert job.start(ctx, out, kw, size=320, n_jobs=1)
    assert not job.start(ctx, out, kw, size=320)            # one render at a time
    job.join(120)
    s = job.snapshot()
    assert not s.running and s.error is None and not s.cancelled
    assert s.done == s.total == 6 and probe(out)["n_frames"] == 6

    kw = _make_kw("kinematics", "s01", ["chewing"], 0.0, 3.0, 1.0, 10, False, "light")
    out2 = tmp_path / "cancelled.mp4"
    job.start(ctx, out2, kw, size=320, n_jobs=1)
    job.cancel()
    job.join(120)
    s = job.snapshot()
    assert s.cancelled and not out2.exists()


def test_viewer_video_path_matches_cli(ctx):
    from biomech.viewer.pages.animation import _make_kw, video_path
    from biomech.viewer.context import ViewerContext

    vc = ViewerContext.from_study(ctx.config.source)
    kw = _make_kw("multimodal", "s01", ["open_close", "chewing"], None, None, 0.5, 30, True,
                  "dark")
    # Pseudonym, never the study id; defaults (0.5x, dark, 1080p) leave the stem short.
    assert video_path(vc, kw).name == "multimodal_P1_open_close+chewing_camera.mp4"
    assert video_path(vc, _make_kw("cohort", None, ["open_close"], None, None, 0.5, 30, False,
                                   "dark")).name == "cohort_open_close.mp4"
    # Settings that change the frames change the name (no silent overwrite).
    k2 = _make_kw("multimodal", "s01", ["open_close"], 1.0, 4.0, 1.0, 30, False, "light")
    assert video_path(vc, {**k2, "size": "720p"}).name == \
        "multimodal_P1_open_close_t1-4_1x_light_720p.mp4"
