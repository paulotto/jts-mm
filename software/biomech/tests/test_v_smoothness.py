# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD16/VD18: jitter above 6 Hz -- the estimator's own content (smoothness) and the content
of its error against the reference (the reported jitter since VD18)."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.validation import evaluate as EV

FS = 200.0


def _xyz(x: np.ndarray) -> np.ndarray:
    return np.c_[x, np.zeros_like(x), np.zeros_like(x)]


def test_jitter_rms_is_the_content_above_6hz():
    """2 Hz (5 mm) + 10 Hz (A) on one axis: ≈ A/√2 times the high-pass gain at 10 Hz
    (1 − 1/(1 + (10/6)^8) = 0.984 for the 4th-order Butterworth run forward-backward), plus
    in quadrature the edge transient of the slow part (sosfiltfilt's default odd padding of
    only 15 samples): about 0.3 % of the slow amplitude as an RMS over a 20 s run, all of
    it within ~0.5 s of the run ends."""
    t = np.arange(0, 20, 1 / FS)
    slow = _xyz(5.0 * np.sin(2 * np.pi * 2 * t))
    j_slow = EV._jitter_rms(slow, FS)
    assert 0.005 < j_slow < 0.02                                   # 0.0156 mm
    hp = slow - EV._lowpass_runs(slow, FS, EV.SMOOTH_HZ)
    inner = (t > 0.5) & (t < 19.5)
    assert np.abs(hp[inner]).max() < 1e-3 and np.abs(hp[~inner]).max() > 0.05
    g = 1 - 1 / (1 + (10 / 6) ** 8)
    for A in (0.1, 0.5):
        x = slow + _xyz(A * np.sin(2 * np.pi * 10 * t))
        assert EV._jitter_rms(x, FS) == pytest.approx(np.hypot(g * A / np.sqrt(2), j_slow),
                                                      rel=0.01)
        assert EV._jitter_rms(x, FS) == pytest.approx(A / np.sqrt(2), rel=0.05)


def test_jitter_runs_shorter_than_1s_are_skipped():
    t = np.arange(0, 10, 1 / FS)
    x = _xyz(0.3 * np.sin(2 * np.pi * 15 * t))
    x[100] = np.nan                                     # leaves a 0.5 s run at the start
    hp = x - EV._lowpass_runs(x, FS, EV.SMOOTH_HZ)
    assert np.isnan(hp[:100]).all() and np.isfinite(hp[101:]).all()
    assert EV._jitter_rms(x, FS) == pytest.approx(0.3 / np.sqrt(2), rel=0.05)


def test_error_jitter_is_against_the_reference():
    """Real motion above 6 Hz (8 Hz, B) is in the reference. An estimator that tracks it with
    a 12 Hz error (A) has own jitter ≈ √(B² + A²)/√2 but error jitter ≈ A/√2; a low-pass
    estimator that drops the 8 Hz motion is smooth but its error jitter is ≈ B/√2."""
    t = np.arange(0, 20, 1 / FS)
    A, B = 0.1, 0.4
    slow = 5.0 * np.sin(2 * np.pi * 1.5 * t)
    ref = _xyz(slow + B * np.sin(2 * np.pi * 8 * t))
    est = ref + _xyz(A * np.sin(2 * np.pi * 12 * t))
    g8, g12 = 1 - 1 / (1 + (8 / 6) ** 8), 1 - 1 / (1 + (12 / 6) ** 8)
    r = EV.smoothness_row(est, ref, FS)
    assert r["jitter_err_rms_mm"] == pytest.approx(g12 * A / np.sqrt(2), rel=0.02)
    assert r["jitter_rms_mm"] == pytest.approx(np.hypot(g8 * B, g12 * A) / np.sqrt(2), rel=0.03)
    assert r["err_rms_mm"] == pytest.approx(A / np.sqrt(2), rel=0.02)
    lp = EV.smoothness_row(_xyz(slow), ref, FS)
    assert lp["jitter_rms_mm"] < 0.02                  # "smoothest" (edge transient only)
    assert lp["jitter_err_rms_mm"] == pytest.approx(g8 * B / np.sqrt(2), rel=0.03)  # not faithful
    q = EV.smoothness_row(ref, ref, FS, is_ref=True)
    assert q["jitter_err_rms_mm"] == 0.0 and q["err_rms_mm"] == 0.0
    assert q["n"] == len(t)


def test_aligned_pooled_removes_constant_frame_offsets():
    """VD19 re-scoring in frames aligned (ref_kinematics:robustness): constant maxillary and
    mandibular offsets between the systems are removed leave-one-task-out, the remaining
    frame noise stays."""
    from types import SimpleNamespace

    from scipy.spatial.transform import Rotation as Ro

    from biomech.validation import registration as REG
    from biomech.validation.selection import Interval

    rng = np.random.default_rng(0)
    t = np.arange(0, 60, 1 / 40)
    ang = np.radians(15) * (1 - np.cos(2 * np.pi * 0.7 * t)) / 2
    lat = np.radians(5) * np.sin(2 * np.pi * 0.3 * t)
    M_Q = REG.make_pose(Ro.from_rotvec(np.c_[0.3 * lat, ang, lat]).as_matrix(),
                        np.c_[60 + 5 * np.sin(2 * np.pi * 0.2 * t), 3 * np.sin(2 * np.pi * 0.5 * t),
                              -60 + 10 * ang])
    A = REG.make_pose(Ro.from_rotvec([0.02, -0.03, 0.01]).as_matrix(), np.array([1.0, -2.0, 0.5]))
    B = REG.make_pose(Ro.from_rotvec([-0.01, 0.02, 0.03]).as_matrix(), np.array([0.5, 1.0, -1.5]))
    M_S = np.einsum("ij,njk,kl->nil", np.linalg.inv(A), M_Q, np.linalg.inv(B))
    M_S[:, :3, 3] += rng.normal(0, 0.1, (len(t), 3))
    o = {"M_S": M_S, "M_Q": M_Q, "inc_q": M_Q[:, :3, 3]}
    mov = [Interval(n, a, b) for n, a, b in (("a", 0, 15), ("b", 15, 30), ("c", 30, 45),
                                             ("d", 45, 60))]
    r = EV.aligned_pooled(SimpleNamespace(tS=t), o, mov, np.zeros(len(t), bool))
    raw = np.sqrt(np.mean(np.sum((M_S[:, :3, 3] - M_Q[:, :3, 3]) ** 2, axis=1)))
    assert raw > 1.0
    assert r["incisor_aligned_rms_mm"] == pytest.approx(0.1 * np.sqrt(3), rel=0.2)
    assert r["rot_aligned_rms_deg"] < 0.1
