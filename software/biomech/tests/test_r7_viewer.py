# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""(viewer revision), fast: no callback mixes shell and page inputs, the
stylesheet mirrors the figure themes, navigation names, polling only while a job runs,
pseudonym-only participant labels."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("dash")

from dash import Dash  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "biomech"
SHELL = {"url", "theme", "viewport", "theme-choice", "page"}


def _registered_app() -> Dash:
    from biomech.viewer.pages import analysis, animation, control, explorer, paper

    app = Dash(__name__, suppress_callback_exceptions=True)
    ctx = SimpleNamespace(config=SimpleNamespace(groups=None), data=None)
    for page in (control, explorer, analysis, paper, animation):
        page.register(app, ctx)
    return app


def _ids(deps: list[dict]) -> list[object]:
    out = []
    for d in deps:
        i = d["id"]
        if isinstance(i, str) and i.startswith("{"):
            i = json.loads(i)
        out.append(i)
    return out


def _is_wildcard(i: object) -> bool:
    return isinstance(i, dict) and any(not isinstance(v, (str, int, float)) or
                                       str(v) in ("ALL", "MATCH", "ALLSMALLER", '["ALL"]')
                                       for v in i.values())


def test_no_callback_mixes_shell_and_page_inputs():
    """A page callback with a shell input fired while another page was shown and Dash
    reported the page's components as nonexistent (the an-view console error)."""
    app = _registered_app()
    bad = []
    for key, cb in app.callback_map.items():
        ins = [i for i in _ids(cb.get("inputs", [])) if not _is_wildcard(i)]
        shell = [i for i in ins if isinstance(i, str) and i in SHELL]
        page = [i for i in ins if not (isinstance(i, str) and i in SHELL)]
        if shell and page:
            bad.append(f"{key}: shell {shell} + page {page}")
    assert not bad, "\n".join(bad)


def test_pages_hold_their_environment_store():
    for page, name in (("analysis", "results"), ("paper", "paper"),
                       ("explorer", "signals"), ("animation", "videos")):
        src = (SRC / "viewer" / "pages" / f"{page}.py").read_text()
        assert f'env_store("{name}")' in src and f'env_id("{name}")' in src


def test_stylesheet_mirrors_the_figure_themes():
    from biomech.viz.style import THEMES

    css = (SRC / "viewer" / "assets" / "style.css").read_text()
    blocks = {"light": re.search(r":root \{(.*?)\}", css, re.S).group(1),
              "dark": re.search(r':root\[data-theme="dark"\] \{(.*?)\}', css, re.S).group(1),
              "dark-os": re.search(r'prefers-color-scheme: dark\) \{\s*:root:not\(\[data-theme='
                                   r'"light"\]\) \{(.*?)\}', css, re.S).group(1)}
    pairs = {"card": "panel", "fg": "ink", "fg2": "ink2", "muted": "muted", "line": "grid",
             "axis": "axis"}
    for name, body in blocks.items():
        theme = THEMES["light" if name == "light" else "dark"]
        tok = dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", body))
        for css_name, key in pairs.items():
            assert tok[css_name].lower() == "#" + theme[key].lower(), (name, css_name)


def test_figure_background_is_the_card_colour():
    from biomech.viz import render_plotly as RP
    from biomech.viz.style import THEMES

    for theme in ("light", "dark"):
        t = RP.template(theme)
        assert t.layout.paper_bgcolor.lstrip("#").lower() == THEMES[theme]["panel"].lower()


def test_navigation_names_and_old_links():
    from biomech.viewer import app as A

    assert [n for n, _ in A.PAGES.values()] == ["Overview", "Signals", "Results", "Paper",
                                                "Videos"]
    for old, new in (("/explorer", "/signals"), ("/animation", "/videos"),
                     ("/analysis", "/results")):
        assert A.ALIASES[old] == new


def test_timers_poll_only_while_a_job_runs():
    """Every interval starts disabled unless its job runs, and its tick can stop it."""
    for page in ("control", "paper", "animation"):
        src = (SRC / "viewer" / "pages" / f"{page}.py").read_text()
        for m in re.finditer(r'dcc\.Interval\(id="([a-z-]+)"', src):
            iid = m.group(1)
            seg = src[m.start():m.start() + 200]
            assert "disabled=" in seg, iid
            assert f'Output("{iid}", "disabled")' in src, iid


def test_video_theme_is_explicit():
    src = (SRC / "viewer" / "pages" / "animation.py").read_text()
    assert 'State("theme", "data")' not in src and 'id="av-theme", value="dark"' in src


def test_participant_labels_are_pseudonyms_only():
    from biomech.viewer.data import pid_label, unit_label

    data = SimpleNamespace(pseudonym=lambda p: {"s09": 3}[p],
                           pseudonyms=lambda: {"s09": 3})
    assert pid_label(data, "s09") == "P3"
    assert unit_label(data, "s09") == "P3"
    assert unit_label(data, "s99") == "excluded"
    assert unit_label(data, "s09", operator=True) == "P3 · s09"
    assert unit_label(data, "_cohort") == "cohort"


# -- paper sets: sections and the gallery's item editor ----------------------------------------
PAPER = """# a paper set
targets: {t: {venue: ieee, out_dir: out}}
items:
  - {section: "1 First"}
  - imu_agreement                 # bare entry, default label
  - {id: imu_agreement, form: box, label: imuBox, part: supplement}   # keeps this comment
  - {section: "Supplement"}
  - {id: fig_kinematics, label: kin}
"""


def test_sections_follow_their_header(tmp_path):
    from biomech import papers

    (tmp_path / "paper_x.yaml").write_text(PAPER)
    ps = papers.load(tmp_path / "paper_x.yaml")
    assert [i.section for i in ps.items] == ["1 First", "1 First", "Supplement"]
    assert [i.index for i in ps.items] == [0, 1, 2]
    assert ps.sections() == ["1 First", "Supplement"]
    assert ps.items[2].entry()["_section"] == "Supplement"


def test_set_item_param_edits_one_line_and_pins_the_label(tmp_path):
    from biomech import papers

    p = tmp_path / "paper_x.yaml"
    p.write_text(PAPER)
    label0 = papers.load(p).items[0].label
    ps = papers.set_item_param(p, 0, "form", "ecdf")
    assert ps.items[0].params == {"form": "ecdf"} and ps.items[0].label == label0
    ps = papers.set_item_param(p, 1, "form", "violin")
    text = p.read_text()
    assert "form: violin" in text and "# keeps this comment" in text and "# a paper set" in text
    assert ps.items[1].label == "imuBox" and ps.items[2].params == {}
    before = p.read_text()
    for bad in (("label", "x"), ("section", "y"), ("in", "t")):
        with pytest.raises(ValueError):
            papers.set_item_param(p, 2, *bad)
    with pytest.raises(ValueError):
        papers.set_item_param(p, 7, "form", "box")
    assert p.read_text() == before                 # refused edits leave the file as it was


def test_gallery_choice_is_validated_and_reloaded(tmp_path):
    from biomech import papers
    from biomech.viewer.pages import paper as pb

    p = tmp_path / "paper_x.yaml"
    p.write_text(PAPER)
    ctx = SimpleNamespace(paper_sets=[papers.load(p)], data=None)
    with pytest.raises(ValueError):                       # not a depiction of this view
        pb.choose_form(ctx, None, 1, "spaghetti")
    msg = pb.choose_form(ctx, None, 1, "raincloud")
    assert "imuBox" in msg and ctx.paper_sets[0].items[1].params["form"] == "raincloud"
    assert papers.load(p).items[1].params["form"] == "raincloud"


def test_item_titles_use_reader_words():
    from biomech.viewer.pages import paper as pb

    data = SimpleNamespace(pseudonym=lambda p: 2)
    t = pb.item_title(SimpleNamespace(data=data), "imu_agreement",
                      {"pid": "s99", "phase": "open_close", "form": "box"})
    assert "P2" in t and "s99" not in t and "open_close" not in t and "box plot" in t


# -- Signals: both systems (validation study) --------------------------------------------------
def _val_ctx(n: int = 400):
    import numpy as np

    t = np.arange(n) * 0.025
    ph = np.array(["rest"] * 100 + ["open_close"] * 200 + ["chewing"] * (n - 300))
    inc = np.c_[np.sin(t), np.cos(t), t]
    ct = np.zeros(n, bool)
    ct[120:130] = True
    kin = {"t_s": t, "phase": ph, "incisor_s_mm": inc, "incisor_q_mm": inc + 0.1,
           "incisor_s_aligned_mm": inc + 0.05, "rot_s_deg": inc, "rot_q_deg": inc,
           "rot_s_aligned_deg": inc, "rot_q_aligned_deg": inc,
           "err_incisor_mm": np.full(n, 0.1), "err_incisor_aligned_mm": np.full(n, 0.05),
           "err_rot_deg": np.full(n, 1.0), "err_rot_aligned_deg": np.full(n, 0.5),
           "ct_visible": ct}
    fus = {"t_master_s": t, "err_rot_I_deg": np.full(n, 2.0),
           "err_rot_aligned_I_deg": np.full(n, 1.5)}
    arrays = {"ref_kinematics": kin, "ref_fusion": fus}
    return SimpleNamespace(arrays=lambda st, pid: arrays[st])


def test_validation_signals_show_both_systems_and_the_imu():
    import numpy as np

    from biomech.viewer import validation_signals as VS
    from biomech.viz import style

    ctx = _val_ctx()
    fig = VS.validation_figure(ctx, "p", "open_close", "aligned", "light")
    names = {t.name for t in fig.data}
    assert {"JTS-MM", "Qualisys", "|Δ rotation|, IMU alone"} <= names
    q = next(t for t in fig.data if t.name == "Qualisys")
    s = next(t for t in fig.data if t.name == "JTS-MM")
    assert q.line.dash == "dash" and s.line.dash == "solid"
    assert s.line.color == style.hexcolor(style.role("system:jtsmm")[0], "light")
    assert len(s.x) == 200 and float(s.x[0]) == 0.0           # the task, from its start
    assert np.allclose(s.y, np.sin(np.arange(100, 300) * 0.025) + 0.05)   # aligned JTS-MM
    imu = next(t for t in fig.data if t.name == "|Δ rotation|, IMU alone")
    assert np.allclose(imu.y, 1.5)                            # aligned IMU-alone error
    assert len(fig.layout.shapes) == 1                        # one CT-in-view run
    own = VS.validation_figure(ctx, "p", "open_close", "own", "light")
    s = next(t for t in own.data if t.name == "JTS-MM")
    assert np.allclose(s.y, np.sin(np.arange(100, 300) * 0.025))
    empty = VS.validation_figure(ctx, "p", "free_motion", "own", "light")
    assert not empty.data and "no validation samples" in empty.layout.annotations[0].text


def test_signals_page_offers_both_systems_only_in_the_validation_study():
    src = (SRC / "viewer" / "pages" / "explorer.py").read_text()
    assert 'value="validation" if validation else "signals"' in src
    assert 'html.Div(both, style={"display": "none"})' in src    # inputs exist everywhere
