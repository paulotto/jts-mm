# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""labels_check (biomech.labels.kappa): Cohen's kappa and the second labelling rule."""

from __future__ import annotations

import numpy as np

from biomech.labels import kappa as K


def test_kappa_values():
    a = np.array([0, 1, 2, 1, 0, 2, 1, 1])
    assert K.kappa(a, a) == 1.0
    b = a.copy()
    b[:4] = 0
    m = np.zeros((7, 7))
    np.add.at(m, (a, b), 1)
    m /= m.sum()
    po, pe = np.trace(m), m.sum(1) @ m.sum(0)
    assert np.isclose(K.kappa(a, b), (po - pe) / (1 - pe))
    # untracked frames (-1) are left out; a constant rater gives kappa 0
    assert K.kappa(np.array([1, -1, 2]), np.array([1, 0, 2])) == 1.0
    assert np.isclose(K.kappa(np.array([0, 0, 0, 0]), np.array([0, 1, 0, 2])), 0.0)


def test_second_rule_opening_angle():
    fs = 40.0
    t = np.arange(0, 6, 1 / fs)
    t_us = (t * 1e6).astype(np.int64)
    theta = 20 * (1 - np.cos(2 * np.pi * t / 3)) / 2            # open-close, 3-s period
    inc = np.zeros((t.size, 3))
    lab = K.second_rule("open_close", t_us, inc, theta)
    rate = np.gradient(theta, 1 / fs)
    core = (np.abs(rate) > 15) & (t > 0.5) & (t < 5.5)
    want = np.where(rate > 0, K.C["OPEN"], K.C["CLOSE"])
    assert np.mean(lab[core] == want[core]) > 0.95
    assert (lab[np.abs(rate) < 2][5:-5] == K.C["REST"]).mean() > 0.6


def test_run_labels_check_levels():
    fs = 40.0
    t_us = (np.arange(0, 20, 1 / fs) * 1e6).astype(np.int64)
    theta = 20 * (1 - np.cos(2 * np.pi * t_us * 1e-6 / 3)) / 2
    inc = np.zeros((t_us.size, 3))
    lab = K.second_rule("open_close", t_us, inc, theta)
    kin = {"open_close__t_us": t_us, "open_close__incisor": inc,
           "open_close__theta_raw_deg": theta}
    out = K.run_labels_check({"s01": ({"open_close__t_us": t_us, "open_close__label": lab},
                                      kin)})["kappa"]
    assert set(out.variant) == {"fixed", "rest_floor"}
    pp = out[(out.variant == "fixed") & (out.level == "participant_phase")]
    assert np.isclose(pp.kappa.iloc[0], 1.0) and np.isclose(pp.kappa_all_frames.iloc[0], 1.0)
