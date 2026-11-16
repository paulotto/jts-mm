# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Clip files (G2, an11): render, store, re-render from the clip alone."""

from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pytest

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
needs_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg/ffprobe not installed")
STUDY_IDS = ("s01", "s02", "s03", "s04")


@pytest.fixture(scope="module")
def ctx(synth_study):
    from biomech.io.context import DataContext

    c = DataContext(synth_study[0])
    yield c
    c.close()


def _leaks(texts):
    from biomech.check.outputs import NAME_RE, pid_leaks

    return pid_leaks(texts) + pid_leaks(texts, NAME_RE)


def _same_timeline(a, b):
    ta, tb = a.timeline, b.timeline
    assert ta.n_frames == tb.n_frames and ta.duration_s == pytest.approx(tb.duration_s)
    assert ta.segments == tb.segments and ta.labels == tb.labels
    np.testing.assert_array_equal(ta.frame_times(), tb.frame_times())


@needs_ffmpeg
def test_render_writes_clip_and_rerenders_from_it_alone(ctx, tmp_path):
    """Render a short clip, re-render it from the clip files in a folder far from the study
    (no --config, no store, no session) with another theme: same frames count and timeline,
    readable timecode; no study id in the clip json or any file name."""
    from click.testing import CliRunner

    from biomech.anim import make, video_stem
    from biomech.anim.clipfile import clip_paths, load_clip_file
    from biomech.anim.render_video import probe, render_video, verify_sync
    from biomech.cli import main

    anim = make(ctx, "s01", "multimodal", ["open_close", "chewing"], 0.0, 0.8, speed=0.5,
                fps=10)
    stem = video_stem("multimodal", "s01", ["open_close", "chewing"],
                      pseudonym=ctx.pseudonym("s01"), t0_s=0.0, t1_s=0.8, size="320", fps=10)
    out = tmp_path / "videos" / f"{stem}.mp4"
    res = render_video(anim, out, size="320", n_jobs=1, frame_hashes=True)
    npz, js = clip_paths(out)
    assert res.clip_files == (npz, js) and npz.is_file() and js.is_file()
    man = json.loads(out.with_suffix(".json").read_text())
    assert man["clip_files"] == [npz.name, js.name]
    doc = json.loads(js.read_text())
    assert doc["clip"]["pseudonym"] == "P1" and doc["clip"]["camera_frames"] == "none"
    assert doc["render"]["size"] == "320"
    # Pseudonym only: no study id in the clip json or in any file name (nor npz keys).
    import zipfile

    keys = zipfile.ZipFile(npz).namelist()
    assert not _leaks([js.read_text()] + [p.name for p in out.parent.iterdir()] + keys)
    # Same settings from the clip: identical frames (the clip holds everything).
    again = load_clip_file(js)
    _same_timeline(anim, again)
    res2 = render_video(again, tmp_path / "same" / "x.mp4", size="320", n_jobs=1,
                        frame_hashes=True, clip_files=False)
    assert res2.frame_hashes == res.frame_hashes

    # Far from the study: only the two clip files.
    far = tmp_path / "far" / "videos"
    far.mkdir(parents=True)
    for p in (npz, js):
        shutil.copy(p, far / p.name)
    r = CliRunner().invoke(main, ["animate", "--from-clip", str(far / js.name), "--theme",
                                  "light", "-j", "1"])
    assert r.exit_code == 0, r.output
    want = far / (video_stem("multimodal", "s01", ["open_close", "chewing"], pseudonym=1,
                             t0_s=0.0, t1_s=0.8, size="320", fps=10, theme="light") + ".mp4")
    assert want.is_file(), sorted(p.name for p in far.iterdir())
    light = load_clip_file(far / js.name).with_(theme="light")
    _same_timeline(anim, light)
    pr = probe(want)
    assert pr["n_frames"] == anim.timeline.n_frames and (pr["width"], pr["height"]) == (320, 180)
    s = verify_sync(want, light, n_samples=5)
    assert s["unreadable"] == 0 and s["max_err_us"] == 0
    m2 = json.loads(want.with_suffix(".json").read_text())
    assert m2["theme"] == "light" and m2["n_frames"] == man["n_frames"]
    assert m2["frames"] == man["frames"]                     # same master-clock timeline
    assert not _leaks([p.name for p in far.iterdir()]
                      + [p.read_text() for p in far.glob("*.json")])


def test_clip_files_never_hold_camera_frames(ctx, tmp_path):
    from biomech.anim import make
    from biomech.anim.clipfile import load_clip_file, save_clip

    anim = make(ctx, "s01", "kinematics", "chewing", 0.0, 0.5, fps=10)
    fake = {"cam0": {"t_us": np.array([anim.clip.t0_us]), "paths": ["/nowhere/frame_000.jpg"]}}
    cam = anim.with_(clip=dataclasses.replace(anim.clip, previews=fake, with_camera=True))
    npz, js = save_clip(cam, tmp_path / "v.mp4")
    text = js.read_text()
    assert ".jpg" not in text and "previews" not in text and "s01" not in text
    assert json.loads(text)["clip"]["camera_frames"] == "omitted"
    back = load_clip_file(npz)
    assert back.clip.previews is None and back.clip.with_camera is False
    assert back.manifest()["with_camera"] is False
    _same_timeline(anim, back)


def test_cohort_clip_roundtrip(ctx, tmp_path):
    from biomech.anim import make
    from biomech.anim.clipfile import clip_stem, load_clip_file, read_clip_doc, save_clip

    anim = make(ctx, None, "cohort", task="open_close", speed=0.5, fps=10)
    npz, js = save_clip(anim, tmp_path / "cohort_open_close.mp4")
    back = load_clip_file(js)
    assert back.clip.labels == anim.clip.labels
    np.testing.assert_array_equal(back.clip.means, anim.clip.means)
    assert len(back.clip.trials) == len(anim.clip.trials)
    _same_timeline(anim, back)
    assert clip_stem(back.with_(theme="light"), read_clip_doc(js)) == "cohort_open_close_10fps_light"


@pytest.mark.parametrize("layout", ["v1_hinge", "v2_intent", "v3_release"])
def test_supplementary_clip_reproduces_frames(ctx, tmp_path, layout):
    """The derived D5 data (hinge ghost, release envelope, stream decisions) travel in the
    clip: a frame rendered from the clip equals the frame rendered after loading."""
    from biomech.anim import make
    from biomech.anim.clipfile import load_clip_file, save_clip

    anim = make(ctx, "s01", layout, None, 0.0, 3.0, speed=None, fps=10)
    npz, js = save_clip(anim, tmp_path / f"{layout}.mp4", size="320")
    back = load_clip_file(js)
    assert set(back.clip.extra) == set(anim.clip.extra) and back.clip.extra
    _same_timeline(anim, back)
    frames = []
    for a in (anim, back):
        sc = a.scene(320, stamp=True)
        try:
            sc.prepare([0, a.timeline.n_frames // 2])
            frames.append(sc.frame(1))
        finally:
            sc.close()
    np.testing.assert_array_equal(frames[0], frames[1])
    assert not _leaks([js.read_text()])


def test_cli_layout_choices_match_registry():
    from biomech.anim import LAYOUTS
    from biomech.cli import main

    opt = next(p for p in main.commands["animate"].params if p.name == "layout")
    assert list(opt.type.choices) == list(LAYOUTS)


def test_from_clip_rejects_missing_partner(tmp_path):
    from click.testing import CliRunner

    from biomech.cli import main

    js = tmp_path / "x.clip.json"
    js.write_text("{}")
    r = CliRunner().invoke(main, ["animate", "--from-clip", str(js)])
    assert r.exit_code != 0 and "clip files missing" in r.output
