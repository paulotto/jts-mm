# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD9: rate and filter effects on peak velocities, on a known motion (no store needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from biomech.validation import sampling as S

F_HZ, AMP_DEG = 3.0, 12.0


def _poses(t: np.ndarray) -> np.ndarray:
    th = AMP_DEG * (1 - np.cos(2 * np.pi * F_HZ * t)) / 2          # 0 .. 12 deg
    M = np.tile(np.eye(4), (len(t), 1, 1))
    M[:, :3, :3] = Rotation.from_rotvec(np.column_stack([0 * th, np.radians(th), 0 * th])
                                        ).as_matrix()
    M[:, :3, 3] = np.column_stack([90 * np.sin(np.radians(th)), 0 * th, -90 * (1 - np.cos(
        np.radians(th)))])
    return M


def test_dataset_chain_underestimates_fast_peaks_and_reference_does_not():
    t200 = np.arange(0, 6, 1 / 200)
    t40 = np.arange(0, 6, 1 / 40)
    rest = t200 < 0.01
    true_peak = AMP_DEG * np.pi * F_HZ                                # max dθ/dt, deg/s
    d200, _ = S._chain(t200, _poses(t200), *S.PIPELINES["ref200"], rest)
    d40, _ = S._chain(t40, _poses(t40), *S.PIPELINES["ref40"], t40 < 0.01)
    p200 = np.nanmax(d200[(t200 > 1) & (t200 < 5)])
    p40 = np.nanmax(d40[(t40 > 1) & (t40 < 5)])
    assert abs(p200 / true_peak - 1) < 0.03                          # the reference is right
    assert p40 / true_peak < 0.95                                    # 6 Hz + 0.15 s at 3 Hz


def test_peaks_bias_and_spectrum():
    t = np.arange(0, 6, 1 / 200)
    rate = AMP_DEG * np.pi * F_HZ * np.sin(2 * np.pi * F_HZ * t)
    sig = {"ref200": (t, rate, np.abs(rate)), "ref40": (t, 0.8 * rate, 0.8 * np.abs(rate))}
    cyc = pd.DataFrame({"task": ["open_close"] * 4, "side": [""] * 4, "cycle": range(4),
                        "start": [200, 400, 600, 800], "end": [266, 466, 666, 866]})
    peaks = S.cycle_peaks("s01", sig, cyc, t)
    assert len(peaks) == 8
    b = S.bias(peaks)
    r = b[b.quantity == "peak_open_vel_dps"].iloc[0]
    assert r.pipeline == "ref40" and abs(r.ratio_median - 0.8) < 1e-9 and r.n == 4
    sp = S.spectrum("s01", t, rate, [("open_close", 0.0, 6.0)]).iloc[0]
    assert abs(sp.f50_hz - F_HZ) < 0.5 and sp.f95_hz < 4.0 and sp.share_above_20hz < 1e-3


def test_dataset_chain_bridges_gaps_and_removes_noise_like_the_pose_chain():
    """VD12: the validation's dataset chain equals the dataset pose chain (bridge, then the
    zero-phase low-pass) and lowers the noise of a slow jaw motion."""
    from biomech.kinematics import se3
    from biomech.validation.evaluate import dataset_chain

    rng = np.random.default_rng(0)
    t = np.arange(800) / 40.0
    th = np.radians(10 * np.sin(2 * np.pi * 0.5 * t))
    M = np.tile(np.eye(4), (len(t), 1, 1))
    M[:, 0, 0] = M[:, 2, 2] = np.cos(th)
    M[:, 0, 2], M[:, 2, 0] = np.sin(th), -np.sin(th)
    M[:, :3, 3] = np.c_[np.zeros_like(t), np.zeros_like(t), -20 * np.sin(2 * np.pi * 0.5 * t)]
    noisy = M.copy()
    noisy[:, :3, 3] += rng.normal(0, 0.5, (len(t), 3))
    noisy[300:310] = np.nan                                     # a 0.25 s gap
    chain = {"max_gap_s": 0.5, "lowpass_hz": 6.0, "lowpass_order": 2}
    out = dataset_chain(noisy, t, chain)
    assert se3.valid(out).all()                                 # gap bridged
    ref, _ = se3.bridge_gaps(noisy, 20)
    assert np.allclose(out, se3.lowpass(ref, 40.0, 6.0, 2), equal_nan=True)
    err_raw = np.linalg.norm(noisy[:, :3, 3] - M[:, :3, 3], axis=1)
    err_f = np.linalg.norm(out[:, :3, 3] - M[:, :3, 3], axis=1)
    keep = np.ones(len(t), bool)
    keep[295:315] = False
    assert np.nanmean(err_f[keep]) < 0.6 * np.nanmean(err_raw[keep])
