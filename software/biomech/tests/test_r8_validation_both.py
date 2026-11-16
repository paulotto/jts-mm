# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Both comparison takes with equal weight (``viz/views/validation_both``, 2026-11-05): panels
with the same ``sharey`` key get one y range (data, set limits and the significance-bar
head-room), an explicit line break in a title starts a new line, and the two-take views
arrange the single-take panels per row with shared y-axes."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.viz import layout as L
from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import registry as R
from biomech.viz.views import validation_both as VB

X = np.linspace(0, 100, 101)


def _panel(title: str, scale: float, key: str | None = None, bars: bool = False) -> Panel:
    p = Panel(title, Axis("Cycle", "\\percent"), Axis("y", "mm"),
              [Series("line", X, scale * np.sin(X / 30.0), role="system:jtsmm")], sharey=key)
    if bars:
        p.series.append(Series("sigbar", np.array([10.0]), y2=np.array([20.0]), role="accent",
                               style={"row": 0}))
    return p


def test_sharey_gives_one_range_with_bar_headroom():
    a, b = _panel("a", 1.0, "k", bars=True), _panel("b", 5.0, "k")
    c = _panel("c", 1.0)
    lay = L.compute(FigureSpec("t", "t", [a, b, c], rows=1, cols=3, width="text"))
    ya, yb, yc = (pl.y.lim for pl in lay.panels)
    assert ya == yb
    lo5 = 5.0 * float(np.sin(X / 30.0).min())
    assert ya[1] > 5.0 and ya[0] < lo5            # the larger take's data fit both
    assert yc[1] < 2.0                            # an unshared panel keeps its own range
    # head-room for the bar row above the data of both
    assert ya[1] - 5.0 > 0.04 * (ya[1] - ya[0])


def test_wrap_keeps_explicit_line_breaks():
    assert L.wrap("P1: masseter R\nr 0.59, CMC 0.51", 5.0, 8.0) == \
        ["P1: masseter R", "r 0.59, CMC 0.51"]


class Ctx:
    def pids(self):
        return ["s10", "s14"]

    def pseudonym(self, pid):
        return {"s10": 1, "s14": 2}[pid]

    def pseudonyms(self):
        return {"s10": 1, "s14": 2}


def _wave(pid: str) -> FigureSpec:
    k = 1.0 if pid == "s10" else 2.0
    panels = []
    for ch in ("Incisor z", "Opening rotation"):
        panels.append(_panel(ch, k))
        d = Panel(f"Δ {ch}", Axis("Cycle", "\\percent"), Axis("Δ", "mm"),
                  [Series("band", X, -k + 0 * X, y2=k + 0 * X, role="accent"),
                   Series("line", X, 0 * X, role="accent"),
                   Series("hline", y=np.array([2.0]), role="reference")])
        d.y.lim = (-2.0 * k, 2.0 * k)
        panels.append(d)
    return attach(FigureSpec(f"val_kin_waveforms--pid-{pid}", "w", panels, rows=2, cols=2),
                  CaptionSpec("x", [("line and band (right)", "diff"), ("solid", "JTS-MM")],
                              f"P{1 if pid == 's10' else 2}, 5 open-close cycles"))


def _emg(pid: str) -> FigureSpec:
    titles = ["masseter R (r = 0.59, CMC undefined*)", "masseter L (r = 0.84, CMC = 0.51)",
              "digastric R (r = 0.98, CMC = 0.97)", "digastric L (r = 0.95, CMC = 0.94)"]
    return attach(FigureSpec(f"val_emg_profiles--pid-{pid}", "e",
                             [_panel(t, 1.0) for t in titles], rows=2, cols=2),
                  CaptionSpec("x", [("*", "CMC undefined")], "n"))


@pytest.fixture()
def fake_build(monkeypatch):
    def build(view, req):
        pid = req.options.get("pid") or getattr(req, "pid", None)
        return {"val_kin_waveforms": _wave, "val_emg_profiles": _emg}[view](pid)

    def request(ctx, view, **kw):
        return type("Q", (), {"options": kw, "pid": kw.get("pid"), "ctx": ctx})()

    monkeypatch.setattr(R, "build", build)
    monkeypatch.setattr(R, "request", request)


def _req(**opts):
    return type("Q", (), {"ctx": Ctx(), "options": opts, "phase": "open_close"})()


def test_waveforms_overlay_puts_both_differences_on_one_axis(fake_build):
    spec = VB.kin_waveforms_both(_req())
    assert (spec.rows, spec.cols) == (2, 3)
    p1, p2, d = spec.panels[:3]
    assert (p1.title, p2.title) == ("P1: Incisor z", "P2: Incisor z")
    assert p1.sharey == p2.sharey
    roles = {s.role for s in d.series if s.kind in ("band", "line")}
    assert roles == {"participant:0", "participant:1"}
    assert sum(s.kind == "hline" for s in d.series) == 1     # the bounds once
    assert d.y.lim == (-4.0, 4.0)                              # union of both takes
    assert "P1" in spec.caption and "P2" in spec.caption


def test_waveforms_columns_form(fake_build):
    spec = VB.kin_waveforms_both(_req(form="columns"))
    assert (spec.rows, spec.cols) == (2, 4)
    t = [p.title for p in spec.panels[:4]]
    assert t == ["P1: Incisor z", "P1: Δ", "P2: Incisor z", "P2: Δ"]
    assert spec.panels[1].sharey == spec.panels[3].sharey


def test_emg_both_rows_per_muscle_group_and_short_stats(fake_build):
    spec = VB.emg_profiles_both(_req())
    assert (spec.rows, spec.cols) == (2, 4)
    t = [p.title for p in spec.panels[:4]]
    assert t[0] == "P1: masseter R\nr 0.59, CMC –*"
    assert t[2].startswith("P2: masseter R")
    assert len({p.sharey for p in spec.panels[:4]}) == 1
    assert spec.panels[4].sharey != spec.panels[0].sharey


def test_waveforms_overlay_all_one_axis_per_channel(fake_build):
    spec = VB.kin_waveforms_both(_req(form="overlay_all"))
    assert (spec.rows, spec.cols) == (2, 2)                    # cycles row, Δ row
    top = spec.panels[0]
    roles = {s.role for s in top.series if s.kind == "line"}
    assert roles == {"participant:0", "participant:1"}
    dashes = {(s.role, (s.style or {}).get("dash")) for s in top.series if s.kind == "line"}
    assert ("participant:1", "solid") in dashes                # JTS-MM solid in both takes
    assert spec.panels[2].title.startswith("Δ ")


def test_take_colours_are_distinguishable_and_not_the_system_blue():
    from biomech.viz import style as S
    a, b = S.ROLES["participant:0"][0], S.ROLES["participant:1"][0]
    assert a != b and "blue" not in (a, b)


def test_colour_meaning_check():
    """check/text.colour_meaning_issues (2026-11-05): one meaning per hue within a figure; the
    closers take the lead muscle's colour by design; explicit colours and dashed variants are
    one meaning; colormap kinds are exempt."""
    from biomech.check.text import colour_meaning_issues

    def fig(*series):
        return FigureSpec("t", "t", [Panel("a", Axis(), Axis(), list(series))])

    ln = lambda role, **st: Series("line", X, X, role=role, style=st)   # noqa: E731
    assert colour_meaning_issues(fig(ln("system:jtsmm"), ln("cat:0"))) != []        # both blue
    assert colour_meaning_issues(fig(ln("entity:masseter_r"), ln("group:closers"))) == []
    assert colour_meaning_issues(fig(ln("cat:2"), ln("cat:2:dot"), ln("cat:2:alt"))) == []
    assert colour_meaning_issues(fig(ln("system:jtsmm", color="orange"),
                                     ln("system:delsys", color="orange"))) == []
    assert colour_meaning_issues(fig(Series("surface3d", x=X, y=X, z=np.ones((101, 101))),
                                     ln("group:closers"))) == []
