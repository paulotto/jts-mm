# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every depiction (form) of every dataset-paper item builds on the synthetic
study and renders for the IEEE and Springer venues, the default form keeps the item's id (others
add ``--form-<x>``),
and every form has a reader name."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomech import papers
from biomech.io.context import DataContext
from biomech.viz import render_pgf, style
from biomech.viz.spec import FigureSpec
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

pytestmark = pytest.mark.slow

PAPER = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027" / "paper_dataset.yaml"


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


def _form_items(ctx) -> list[tuple[str, str]]:
    ids = dict.fromkeys(i.view_id for i in papers.load(PAPER).items)
    return [(v, f) for v in ids for f in R.get(v).choice_values("form", ctx)]


def test_every_form_of_every_dataset_item_builds_and_renders(ctx, tmp_path):
    items = _form_items(ctx)
    assert len(items) >= 25                               # the C1 gallery, not a stub
    for vid, form in items:
        spec = R.build(vid, R.request(ctx, vid, form=form))
        default = R.get(vid).defaults.get("form")
        assert spec.id == (vid if form == default else f"{vid}--form-{form}"), (vid, form)
        if isinstance(spec, FigureSpec):
            assert spec.caption, (vid, form)
            for venue in ("ieee", "springer"):
                with style.use_venue(venue):
                    render_pgf.render_figure(spec, tmp_path / f"{vid}_{form}_{venue}")


def test_every_form_has_a_reader_name(ctx):
    for _vid, form in _form_items(ctx):
        assert V.choice_label("form", form) != form.replace("_", " "), form
