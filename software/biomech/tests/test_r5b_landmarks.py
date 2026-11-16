# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5b: the incisor landmark of the comparison takes (the CT sync motion lies inside
``mand_point_1``) and incisor roles fixed by convention."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from biomech.frames.landmarks import _still_hold, dental_roles, digitize
from biomech.io.session import OpticalData


def _take(rng: np.random.Generator) -> tuple[OpticalData, list]:
    """50 s of mand_point_1 at 40 Hz: 30 s of arbitrary CT motion, then a 20 s hold of
    the tip on the incisor at (0, 19, -30) mm in the jaw (mta) frame; the jaw is static."""
    n, fs = 2000, 40.0
    t_us = (np.arange(n) / fs * 1e6).astype(np.uint64)
    mta = np.repeat(np.eye(4)[None], n, axis=0)
    mta[:, :3, 3] = [10.0, 40.0, 650.0]
    ct = np.repeat(np.eye(4)[None], n, axis=0)
    inc = np.array([0.0, 19.0, -30.0])
    ct[:, :3, 3] = mta[:, :3, 3] + inc + rng.normal(0, 0.3, (n, 3))
    m = n * 3 // 5
    # the sync motion happens away from the face (real takes: ~300 mm off in the jaw frame)
    ct[:m, :3, 3] += [0.0, -150.0, 300.0] + 60.0 * np.sin(np.linspace(0, 40, m))[:, None]
    ct[: m // 4, :3, 3] = np.nan                      # CT on/off bursts
    poses = {"ct": ct, "mta": mta, "cra": mta.copy()}
    ph = SimpleNamespace(name="mand_point_1", kind="hold", t_start_us=int(t_us[0]),
                         t_end_us=int(t_us[-1]))
    return OpticalData(t_us, poses, {}), [ph]


def test_still_hold_recovers_the_incisor_after_a_sync_motion():
    opt, phases = _take(np.random.default_rng(3))
    p, mad, n = _still_hold(opt, opt.poses["mta"], phases[0].t_start_us, phases[0].t_end_us)
    assert np.linalg.norm(p - [0.0, 19.0, -30.0]) < 0.5 and mad < 1.0 and n > 600
    still = digitize(opt, phases, hold="still").get("mand_point_1")
    phase = digitize(opt, phases).get("mand_point_1")     # dataset default
    assert np.linalg.norm(still - [0.0, 19.0, -30.0]) < 0.5
    assert np.linalg.norm(phase - [0.0, 19.0, -30.0]) > 5.0     # the old rule is corrupted


def test_incisor_is_never_relabelled():
    tl, tr = np.array([-90.0, 55, 20]), np.array([-90.0, -55, 20])
    conv = {"incisor": "m1", "left": "m2", "right": "m3"}
    # geometry would pick m2 as the incisor (m1 digitized off the midline, as with a
    # corrupted hold); the convention still holds
    pts = {"m1": np.array([-60.0, 40, 0]), "m2": np.array([0.0, 0, 0]),
           "m3": np.array([-40.0, -25, 0])}
    r = dental_roles(pts, tl, tr, conv)
    assert r.incisor == "m1" and not r.agrees
