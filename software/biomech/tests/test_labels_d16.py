# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D16: free-motion label axes scaled by max(own P95, 0.25 x largest axis P95).

Synthetic incisal trajectory: open/close cycles along z with a nearly unused lateral
axis (sub-millimetre jitter). With per-axis P95 scaling (pre-D16, ``axis_scale_floor:
0``) the jitter is amplified to full scale and wins against the true axis between and
during movements; with the floor the labels follow the real movement.
"""

from __future__ import annotations

import numpy as np
import pytest

from biomech.labels.auto_labels import (AXIS_SCALE_FLOOR, CLASSES, CODE, axis_scales,
                                        compare_axis_scale, label_agreement, label_phase)

P = {"vel_cutoff_hz": 4.0, "hi_frac": 0.2, "lo_frac": 0.1, "min_seg_s": 0.1}
FS = 40.0


def _trajectory(seed: int = 0):
    rng = np.random.default_rng(seed)
    t = (np.arange(0, 20, 1 / FS) * 1e6).astype(np.int64)
    ts = t * 1e-6
    z = -15 * (1 - np.cos(2 * np.pi * 0.4 * ts))                 # opening to 30 mm
    z[ts > 16] = z[np.searchsorted(ts, 16)]                       # still at the end
    y = 0.15 * rng.standard_normal(t.size).cumsum() / np.sqrt(FS)  # lateral jitter, < 1 mm
    x = 0.05 * rng.standard_normal(t.size)
    inc = np.column_stack([x, y, z])
    return t, inc


def test_axis_scales_floor():
    np.testing.assert_allclose(axis_scales([10.0, 1.0, 4.0]), [10.0, 2.5, 4.0])
    np.testing.assert_allclose(axis_scales([10.0, 1.0, 4.0], 0.0), [10.0, 1.0, 4.0])
    s = axis_scales([np.nan, 2.0, 8.0])
    assert np.isnan(s[0]) and s[1] == 2.0 and AXIS_SCALE_FLOOR == 0.25


def test_nearly_unused_axis_no_longer_creates_lateral_labels():
    t, inc = _trajectory()
    old, new, agr = compare_axis_scale("free_motion", t, inc, FS,
                                       {**P, "axis_scale_floor": 0.25})
    lateral = (CODE["LEFT"], CODE["RIGHT"], CODE["PROTRUSION"], CODE["RETRUSION"])
    assert np.isin(old, lateral).mean() > 0.05                    # spurious pre-D16 labels
    assert np.isin(new, lateral).mean() < 0.01
    # The open/close movement is labelled as such with the floor ...
    opening = np.gradient(inc[:, 2]) < -0.3
    assert np.mean(new[opening] == CODE["OPEN"]) > 0.9
    # ... and as in the single-axis open_close phase.
    oc = label_phase("open_close", t, inc, FS, P)
    assert np.mean(new == oc) > 0.95
    # The agreement report.
    assert agr["n"] == t.size and 0.05 < agr["changed_frac"] < 1.0
    assert set(agr["per_class"]) == set(CLASSES)
    assert agr["per_class"]["LEFT"]["n_new"] + agr["per_class"]["RIGHT"]["n_new"] \
        < agr["per_class"]["LEFT"]["n_old"] + agr["per_class"]["RIGHT"]["n_old"]
    assert agr["kappa"] < 1.0


def test_floor_zero_is_the_pre_d16_rule_and_default_is_d16():
    t, inc = _trajectory(1)
    old = label_phase("free_motion", t, inc, FS, {**P, "axis_scale_floor": 0.0})
    # pre-D16 reference implementation: each axis divided by its own P95
    from biomech.labels import auto_labels as al

    v = al._velocity(t, inc, FS, P["vel_cutoff_hz"])
    comps = [(sg * v[:, ax], np.nanpercentile(np.abs(sg * v[:, ax]), 95), pos, neg)
             for ax, sg, pos, neg in al.AXES["free_motion"]]
    norm = np.stack([np.abs(s) / r for s, r, _, _ in comps], axis=1)
    best = np.nanargmax(np.nan_to_num(norm, nan=-1.0), axis=1)
    moving = al._hysteresis(np.nanmax(norm, axis=1), P["hi_frac"], P["lo_frac"])
    ref = np.zeros(t.size, np.int8)
    for k, (s, _, pos, neg) in enumerate(comps):
        sel = moving & (best == k)
        ref[sel & (s > 0)] = CODE[pos]
        ref[sel & (s < 0)] = CODE[neg]
    ref = al._merge_short(ref, max(int(round(P["min_seg_s"] * FS)), 1))
    np.testing.assert_array_equal(old, ref)
    np.testing.assert_array_equal(label_phase("free_motion", t, inc, FS, P),
                                  label_phase("free_motion", t, inc, FS,
                                              {**P, "axis_scale_floor": 0.25}))


def test_single_axis_phases_unaffected():
    t, inc = _trajectory(2)
    for ph in ("open_close", "left_right", "protrusion_retrusion", "rest"):
        np.testing.assert_array_equal(
            label_phase(ph, t, inc, FS, {**P, "axis_scale_floor": 0.0}),
            label_phase(ph, t, inc, FS, P))


def test_label_agreement_ignores_untracked_and_checks_length():
    a = np.array([0, 1, 1, 5, -1, 2], np.int8)
    b = np.array([0, 1, 0, 1, 3, -1], np.int8)
    r = label_agreement(a, b)
    assert r["n"] == 4 and r["changed_frac"] == pytest.approx(0.5)
    assert r["per_class"]["OPEN"] == {"n_old": 2, "n_new": 2, "changed_frac": 0.5}
    assert r["per_class"]["LEFT"]["changed_frac"] == 1.0
    assert np.isnan(r["per_class"]["RIGHT"]["changed_frac"])
    with pytest.raises(ValueError):
        label_agreement(a, b[:3])
