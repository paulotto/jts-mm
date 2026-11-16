# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""the depiction drawers (no store needed)."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.viz.spec import Axis
from biomech.viz.views import forms
from biomech.viz.views import vocab as V

rng = np.random.default_rng(3)
LABELS = ["a", "b", "c"]
VALUES = [rng.gamma(4, 0.3, 14) for _ in LABELS]


@pytest.mark.parametrize("form", forms.SUMMARY)
def test_every_form_draws_rows_and_says_what_it_shows(form):
    p, marks = forms.summary(form, LABELS, VALUES, "t", Axis("x", "mm"))
    assert p.series and marks and all(len(m) == 2 and m[1] for m in marks)
    if form != "ecdf":
        assert p.y.ticklabels == LABELS
    p2, _ = forms.summary(form, LABELS, VALUES, "t", Axis("x", "mm"))
    for s, t in zip(p.series, p2.series):              # deterministic (seeded, no jitter rng)
        assert np.array_equal(np.asarray(s.x), np.asarray(t.x), equal_nan=True)


def test_density_on_a_log_axis_stays_positive_and_bar_needs_a_linear_axis():
    p, _ = forms.summary("violin", LABELS, VALUES, "t", Axis("x", "mm", log=True))
    xs = np.concatenate([np.asarray(s.x, float) for s in p.series if s.kind == "fill"])
    assert xs.size and (xs > 0).all()
    with pytest.raises(ValueError, match="linear axis"):
        forms.summary("bar", LABELS, VALUES, "t", Axis("x", "mm", log=True))


def test_empty_rows_and_unknown_form():
    p, _ = forms.summary("box", LABELS, [np.array([]), np.array([np.nan]), []], "t", Axis("x"))
    assert p.note == "no data"
    with pytest.raises(ValueError, match="unknown form"):
        forms.summary("pie", LABELS, VALUES, "t", Axis("x"))


def test_form_names_for_readers():
    assert V.picker_label("form") == "Depiction"
    assert V.choice_label("form", "raincloud") == "raincloud (density, box, dots)"
    for f in forms.SUMMARY:
        assert V.choice_label("form", f) != f.replace("_", " ") or f == "ecdf"
