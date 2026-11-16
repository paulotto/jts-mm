# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frequency-domain views (views/frequency.py and movement_bandwidth) on the synthetic study.

Every new view is built for every declared phase, rendered with Plotly, matplotlib and
pgf, and checked for content: panel and series counts, finite values that match the
store, pseudonym labels and no study ids in any text. Outputs the synthetic study does
not produce (low-frequency mask hits, the L-vs-R spectral SPM{t}, which needs five
participants) are injected into a patched context to check those code paths too.
"""

from __future__ import annotations

import ast
import re
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

from biomech.io.context import DataContext
from biomech.results.access import track
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import to_plotly
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

matplotlib.use("Agg")
pytestmark = pytest.mark.slow

NEW = ("emg_psd", "emg_spectral_descriptors", "emg_spectral_spm", "emg_spectral_cohort",
       "emg_spectral_cohort_table", "emg_mask_validation", "emg_mains", "imu_spectra",
       "chewing_rate", "movement_bandwidth")
STUDY_ID = re.compile(r"(?<![A-Za-z0-9])s\d{2}(?![0-9])")


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


def _cases():
    out = []
    for vid in NEW:
        v = R.get(vid)
        for ph in R.expand_phases(v):
            out.append((vid, ph))
    return out


def _build(ctx, vid, phase=None, c=None):
    v = R.get(vid)
    kw = {"phase": phase} if phase else {}
    if "pid" in v.params:
        kw["pid"] = ctx.pids()[1]
    return R.build(vid, R.request(c or ctx, vid, **kw))


def _texts(spec) -> list[str]:
    if isinstance(spec, TableSpec):
        return [spec.caption, spec.notes, *(c.header for c in spec.columns),
                *(str(v) for r in spec.rows for v in r.values())]
    out = [spec.id, spec.title, spec.caption, spec.perspective]
    for p in spec.panels:
        out += [p.title, p.note, p.x.label, p.y.label, *(p.x.ticklabels or []),
                *(p.y.ticklabels or [])]
        for s in p.series:
            out += [s.label or "", *(s.text or [])]
    return out


def _render(spec, tmp_path):
    if isinstance(spec, TableSpec):
        tex = render_pgf.render_table(spec, tmp_path).read_text()
    else:
        to_plotly(spec)
        render_mpl.render_figure(spec, tmp_path / f"{spec.id}.png", dpi=80)
        tex = render_pgf.render_figure(spec, tmp_path).read_text()
    assert not STUDY_ID.search(tex), STUDY_ID.search(tex)


def _finite(spec) -> bool:
    for p in spec.panels:
        for s in p.series:
            for a in (s.x, s.y, s.z):
                if a is not None and np.isfinite(np.asarray(a, float)).any():
                    return True
    return False


def _series(p, kind, role=None):
    return [s for s in p.series if s.kind == kind and (role is None or s.role == role)]


def _medians(p):
    """The labelled median diamonds of a paired strip (one per condition; the unit dots and
    the CI lines carry no label)."""
    return [s for s in _series(p, "scatter") if s.label]


def test_registered_with_domains():
    want = {"emg_psd": "emg_freq", "emg_spectral_descriptors": "emg_freq",
            "emg_spectral_spm": "emg_freq", "emg_spectral_cohort": "emg_freq",
            "emg_spectral_cohort_table": "emg_freq", "emg_mask_validation": "emg_freq",
            "emg_mains": "emg_freq", "imu_spectra": "imu", "movement_bandwidth": "imu",
            "chewing_rate": "kinematics"}
    for vid, dom in want.items():
        v = R.get(vid)
        assert v.domain == dom and v.kind == "dashboard", vid
    assert set(R.get("emg_psd").phases) >= {"chewing", "rest", "mvc_closing"}
    assert set(R.get("imu_spectra").phases) == {"rest", *V.TASKS}


@pytest.mark.parametrize("vid,phase", _cases())
def test_builds_renders_and_hides_study_ids(ctx, tmp_path, vid, phase):
    spec = _build(ctx, vid, phase)
    texts = _texts(spec)
    assert not any(STUDY_ID.search(t or "") for t in texts), [t for t in texts
                                                              if STUDY_ID.search(t or "")]
    if isinstance(spec, FigureSpec):
        assert spec.caption and spec.sources
        if vid != "emg_mask_validation":                 # no low-frequency hits in synth
            assert _finite(spec), vid
        for p in spec.panels:                            # legends only with >= 2 entries
            assert p.legend_mode() == "none" or len(p.legend_entries()) >= 2
    else:
        assert spec.rows
    _render(spec, tmp_path)


def test_emg_psd_matches_store(ctx):
    pid = ctx.pids()[1]
    spec = _build(ctx, "emg_psd", "chewing")
    a = ctx.arrays("emg", pid)
    sp = ctx.table("emg", "spectral", [pid])
    names = list(sp[sp.task == "chewing"].muscle)
    f0 = float(ctx.table("qc", "emg", [pid]).line_hz.median())
    assert len(spec.panels) == 6 and spec.rows == 2 and spec.cols == 3
    for m, p in zip(V.MUSCLES, spec.panels):
        (line,) = _series(p, "line")
        assert line.role == f"entity:{m}"
        ref = 10 * np.log10(a["spectral__chewing__psd"][names.index(m)])
        np.testing.assert_allclose(line.y, ref, rtol=1e-5)
        (harm,) = _series(p, "vline")
        np.testing.assert_allclose(harm.x / f0, np.arange(1, len(harm.x) + 1))
        assert "MNF" in p.note
    assert spec.perspective.startswith(f"P{ctx.pseudonym(pid)}")


def test_spectral_descriptors_content(ctx):
    spec = _build(ctx, "emg_spectral_descriptors", "chewing")
    assert len(spec.panels) == 3
    sp = ctx.table("emg", "spectral")
    use = sp[(sp.task == "chewing") & sp.usable_shape.astype(bool)]
    mnf = spec.panels[0]
    pts = np.concatenate([s.x for s in _series(mnf, "scatter", "context")])
    assert np.allclose(np.sort(pts), np.sort(use.mnf_hz.to_numpy()))
    (hm,) = _series(spec.panels[2], "heatmap")
    assert hm.z.shape == (use.muscle.nunique(), 4)
    assert np.all((hm.z >= 0) & (hm.z <= 100))
    assert np.allclose(hm.z.sum(axis=1), 100, atol=5)      # medians of shares: ~100 %


def test_spectral_cohort_and_table(ctx):
    sc = ctx.table("stats_emg", "spectral_cohort")
    spec = _build(ctx, "emg_spectral_cohort")
    assert len(spec.panels) == 3
    for p in spec.panels:
        (hm,) = _series(p, "heatmap")
        assert hm.z.shape == (sc.task.nunique(), sc.muscle.nunique())
    t = _build(ctx, "emg_spectral_cohort_table")
    assert len(t.rows) == len(sc)
    keys = {c.key for c in t.columns}
    assert {"mnf_mean_hz", "mdf_mean_hz", "mnf_sd_between_hz", "mnf_sd_within_hz",
            "between_over_within"} <= keys


def test_spectral_spm_anova_and_patched_t(ctx):
    spm = ctx.table("stats_emg", "spectral_spm")
    spec = _build(ctx, "emg_spectral_spm")
    assert len(spec.panels) == 2
    an = spm[spm.test == "anova1_masseter_r_chewing_spectrum"].iloc[0]
    cl = ast.literal_eval(an.clusters_hz)
    bars = _series(spec.panels[1], "sigbar")
    if cl:
        assert len(bars) == 1 and len(bars[0].x) == len(cl)
        assert [tuple(c[:2]) for c in cl] == [tuple(c) for c in spec.meta["anova_clusters_hz"]]
    # The paired SPM{t} needs >= 5 participants; inject one to check that path.
    grid = ctx.store.read_arrays("stats_emg", "_cohort")["spectral__grid_hz"]
    z = 3 * np.sin(np.linspace(0, 3, len(grid)))
    spm2 = spm.copy()
    i = spm2.index[spm2.test == "masseter_L_vs_R_chewing_spectrum"][0]
    spm2.loc[i, ["n", "zstar", "supra_frac", "clusters_hz"]] = [6, 2.5, 0.2,
                                                                "[(100, 160, 0.01)]"]
    c = _Patched(ctx, {("stats_emg", "spectral_spm"): spm2},
                 {"stats_emg": {"spectral_spm1__z": z}})
    spec2 = R.build("emg_spectral_spm", R.request(c, "emg_spectral_spm"))
    p1 = spec2.panels[0]
    (line,) = _series(p1, "line")
    np.testing.assert_allclose(line.y, z)
    assert sorted(float(s.y[0]) for s in _series(p1, "hline")) == [-2.5, 2.5]
    (span,) = _series(p1, "hspan")
    assert float(span.x[0]) == 100 and float(span.y2[0]) == 160


def test_mask_validation_patched(ctx, tmp_path):
    from biomech.stats.cohort_emg import mask_validation_table

    rng = np.random.default_rng(0)
    rows = []
    for pid in ctx.pids():
        for m in V.MUSCLES:
            rows.append({"pid": pid, "task": "chewing", "muscle": m, "n_masked": 5,
                         "n_clean": 40, "share_20_40_masked": 0.05 + 0.01 * rng.random(),
                         "share_20_40_clean": 0.04, "share_q1": 0.03, "share_q3": 0.06,
                         "mnf_masked_hz": 120 + rng.random(), "mnf_clean_hz": 125.0,
                         "mnf_q1": 110, "mnf_q3": 140, "conservative": bool(rng.random() > .3)})
    mv = pd.DataFrame(rows)
    c = _Patched(ctx, {("emg", "mask_validation"): mv,
                       ("stats_emg", "mask_validation"): mask_validation_table(mv)})
    spec = R.build("emg_mask_validation", R.request(c, "emg_mask_validation"))
    assert len(spec.panels) == 3
    a, b = _medians(spec.panels[0])           # median diamonds (paired_strip)
    assert np.allclose(b.x, 4.0) and np.all(a.x > 4.0)
    est = np.concatenate([s.x for s in _series(spec.panels[2], "scatter")])
    assert est[0] == pytest.approx(100 * mv.conservative.mean())
    assert len(est) == 1 + len(V.MUSCLES)
    _render(spec, tmp_path)
    # Synthetic study: no hits -> an annotated empty figure, never an error.
    empty = _build(ctx, "emg_mask_validation")
    assert len(empty.panels) == 1 and "no channel-task" in empty.caption


def test_mains_content(ctx):
    spec = _build(ctx, "emg_mains")
    q = ctx.table("qc", "emg")
    p1, p2, p3 = spec.panels
    assert [t.split()[0] for t in p1.y.ticklabels] == [f"P{i + 1}" for i in
                                                      range(len(ctx.pids()))]
    d = np.concatenate([s.x for s in _series(p1, "scatter")])
    nominal = np.array([float(t.split("(")[1].split()[0]) for t in p1.y.ticklabels])
    lhz = q.groupby("pid").line_hz.median().reindex(ctx.pids()).to_numpy()
    np.testing.assert_allclose(d, 1e3 * (lhz - nominal))
    raw, clean = _medians(p2)
    assert np.all(raw.x > clean.x)
    assert _series(p3, "vline")                             # rest-noise limit


def test_imu_spectra_content(ctx):
    pid = ctx.pids()[1]
    a = ctx.arrays("imu", pid)
    if "bandwidth__chewing__psd" not in a:
        pytest.fail("imu arrays bandwidth__{task}__psd missing from the store")
    spec = _build(ctx, "imu_spectra", "chewing")
    bw = ctx.table("imu", "bandwidth", [pid]).set_index("task").loc["chewing"]
    p1, p2 = spec.panels
    xs = sorted(float(s.x[0]) for s in _series(p1, "vline"))
    want = sorted(v for v in (bw.f50_hz, bw.f95_hz, bw.f99_hz, bw.f95_coh_hz)
                  if np.isfinite(v))
    np.testing.assert_allclose(xs, want)
    assert len(p1.legend_entries()) == 2                    # raw and C'P
    (th,) = _series(p2, "hline")
    assert float(th.y[0]) == 0.5
    if bw.coh_band_hz > 0:
        (span,) = _series(p2, "hspan")
        assert float(span.y2[0]) == pytest.approx(bw.coh_band_hz)


def test_movement_bandwidth_content(ctx):
    spec = _build(ctx, "movement_bandwidth")
    bw = ctx.table("imu", "bandwidth")
    bw = bw[bw.task != "rest"]
    p1, p2 = spec.panels
    assert [s.label for s in p1.legend_entries()] == ["f50", "f95", "f99"]
    n95 = sum(len(s.x) for s in p1.series if s.label == "f95")
    assert n95 == int(np.isfinite(bw.f95_hz).sum())
    nl = bw[bw.noise_limited.astype(bool)]
    # noise-limited values: the shared "flagged" marker
    marked = sum(len(s.x) for s in _series(p2, "scatter", "flagged"))
    assert marked == int(np.isfinite(nl.f95_coh_hz).sum() + np.isfinite(nl.coh_band_hz).sum())


def test_chewing_rate_content(ctx):
    spec = _build(ctx, "chewing_rate")
    cyc = ctx.table("kinematics", "cycles")
    ok = cyc[(cyc.task == "chewing") & cyc.ok.astype(bool)]
    p1, p2 = spec.panels
    assert p1.y.ticklabels == [f"P{ctx.pseudonym(p)}" for p in ctx.pids()]
    (bars,) = _series(p2, "bar")
    assert int(bars.y.sum()) == len(ok)
    subj = ctx.table("stats_kinematics", "subjects").set_index("pid")
    sq = [s for s in _series(p1, "scatter") if s.label == "1 / median cycle duration"][0]
    np.testing.assert_allclose(sq.x, subj.chewing_freq_hz.reindex(ctx.pids()).to_numpy())


def test_frequency_outputs_are_read(ctx):
    """The outputs these views exist for are read (coverage guard, per family)."""
    from biomech.results.access import family

    with track() as log:
        for vid, ph in _cases():
            _build(ctx, vid, ph)
            if "pid" in R.get(vid).params:
                R.build(vid, R.request(ctx, vid, phase=ph, pid=ctx.pids()[0]))
    tables = log.tables
    for t in [("emg", "spectral"), ("emg", "mask_validation"), ("stats_emg", "mask_validation"),
              ("stats_emg", "spectral_spm"), ("stats_emg", "spectral_cohort"),
              ("imu", "bandwidth"), ("qc", "emg"), ("stats_kinematics", "subjects"),
              ("kinematics", "cycles")]:
        assert t in tables, t
    fams = log.families()
    pid = ctx.pids()[0]
    want = {("stats_emg", "spectral__grid_hz")}
    for stage, prefix in (("emg", "spectral__"), ("imu", "bandwidth__")):
        want |= {(stage, family(k)) for k in ctx.arrays(stage, pid).keys()
                 if k.startswith(prefix) and not k.endswith("__win")}   # __win: INTERNAL
    if "spectral_spm1__z" in ctx.store.read_arrays("stats_emg", "_cohort").keys():
        want.add(("stats_emg", "spectral_spm1__z"))
    assert want <= fams, want - fams


class _Patched:
    """A DataContext with some tables / cohort arrays replaced (paths synth lacks)."""

    def __init__(self, ctx, tables=None, cohort_arrays=None):
        self._ctx, self._t, self._a = ctx, tables or {}, cohort_arrays or {}
        self.store = SimpleNamespace(read_arrays=self._read)

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def table(self, stage, name, pids=None):
        if (stage, name) in self._t:
            return self._t[(stage, name)].copy()
        return self._ctx.table(stage, name, pids)

    def _read(self, stage, unit):
        d = dict(self._ctx.store.read_arrays(stage, unit))
        d.update(self._a.get(stage, {}))
        return d

    def arrays(self, stage, unit):              # views read cohort arrays via ctx.arrays
        return self._read(stage, unit)
