# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""the systems-paper video layout (``validation``) on a synthetic clip, and
reader names for every layout."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _arrays(n: int = 600):
    t = np.arange(n) * 0.025
    ph = np.array(["open_close"] * n)
    th = 20 * np.clip(np.sin(2 * np.pi * 0.5 * t), 0, None)
    inc = np.c_[-0.3 * th, np.zeros(n), -0.9 * th]
    tl, tr = np.tile([-70.0, 50.0, 30.0], (n, 1)), np.tile([-70.0, -50.0, 30.0], (n, 1))
    gate = np.zeros(n, bool)
    gate[200:260] = True
    kin = {"t_s": t, "t_master_s": 100.0 + t, "phase": ph,
           "incisor_q_mm": inc, "incisor_s_mm": inc + 2.0, "incisor_s_aligned_mm": inc + 0.5,
           "tmj_left_q_mm": tl, "tmj_left_s_mm": tl + 2, "tmj_left_s_aligned_mm": tl + 0.5,
           "tmj_right_q_mm": tr, "tmj_right_s_mm": tr + 2, "tmj_right_s_aligned_mm": tr + 0.5,
           "rot_q_deg": np.c_[0 * th, th, 0 * th], "rot_s_deg": np.c_[0 * th, th + 3, 0 * th],
           "rot_q_aligned_deg": np.c_[0 * th, th, 0 * th],
           "rot_s_aligned_deg": np.c_[0 * th, th + 1, 0 * th],
           "err_incisor_mm": np.full(n, 2.0), "err_incisor_aligned_mm": np.full(n, 0.5),
           "err_rot_deg": np.full(n, 3.0), "err_rot_aligned_deg": np.full(n, 1.0),
           "ct_visible": np.zeros(n, bool), "gate": gate, "evaluated": np.ones(n, bool)}
    fus = {"t_master_s": kin["t_master_s"], "err_rot_aligned_I_deg": np.full(n, 2.0)}
    return {"ref_kinematics": kin, "ref_fusion": fus}


@pytest.fixture
def anim():
    from biomech.anim import layouts as Lo
    from biomech.anim import validation_video as VV
    from biomech.anim.source import Clip

    arr = _arrays()
    ctx = SimpleNamespace(arrays=lambda st, pid: arr[st])
    clip = Clip("p", "Participant 1", [("open_close", 102e6, 112e6)])
    clip.extra["validation"] = VV._validation(ctx, "p", clip)
    return Lo.Animation(clip, "validation", fps=10, speed=1.0, theme="light")


def test_validation_layout_is_registered_with_defaults():
    from biomech.anim import LAYOUTS, default_phases
    from biomech.anim.layouts import EXTRAS, PREVIEWS, STILLS

    assert "validation" in LAYOUTS and EXTRAS["validation"] == ("validation",)
    assert "validation" in PREVIEWS and "validation" in STILLS
    assert default_phases("validation") == ["open_close"]


def test_every_layout_has_a_reader_name():
    from biomech.anim import LAYOUTS
    from biomech.viz.views import vocab as V

    for k in LAYOUTS:
        assert V.term("video_layout", k) != k, k


def test_loader_keeps_the_stored_values_and_the_evaluated_frames(anim):
    v = anim.clip.extra["validation"]
    arr = _arrays()["ref_kinematics"]
    t = arr["t_master_s"] * 1e6
    sel = (t >= 102e6 - 6e6) & (t <= 112e6 + 6e6)
    assert np.allclose(v["incisor_s"], arr["incisor_s_aligned_mm"][sel])   # frames aligned
    assert np.allclose(v["rot_s"], arr["rot_s_aligned_deg"][sel])
    assert np.allclose(v["err_incisor"], 0.5) and np.allclose(v["err_rot_imu"], 2.0)
    assert v["used"].sum() == (~arr["gate"][sel]).sum()          # the tables' frames


def test_validation_layout_composes_and_renders(anim):
    comp = anim.compose()
    names = {r.name for r in comp.regions}
    assert {"header", "jaw", "plane", "rot", "einc", "erot", "tool", "caption", "key"} <= names
    from biomech.anim.layers import Outline

    outlines = [ly for ly in comp.layers if isinstance(ly, Outline)]
    assert {o.role for o in outlines} == {"system:jtsmm", "system:qualisys"}
    assert all(o.dashed == (o.role == "system:qualisys") for o in outlines)
    sc = anim.scene("640")
    sc.prepare([5])
    img = sc.frame(0)
    assert img.ndim == 3 and img.shape[2] == 3 and img.std() > 0
    assert anim.manifest()["layout"] == "validation"


def test_split_draws_unused_frames_faint():
    from biomech.anim.validation_video import _split

    used = np.array([True, False, True])
    a, b = _split(np.array([1.0, 2.0, 3.0]), used)
    assert np.isnan(a[1]) and a[0] == 1 and np.isnan(b[0]) and b[1] == 2


def test_captions_become_a_subtitle_track(anim, tmp_path):
    from biomech.anim.render_video import caption_cues, write_vtt

    cues = caption_cues(anim)
    assert cues and all(b > a for a, b, _ in cues)
    p = write_vtt(cues, tmp_path / "v.vtt")
    text = p.read_text()
    assert text.startswith("WEBVTT") and "-->" in text and cues[0][2] in text
    assert "00:00:00.000 --> " in text
