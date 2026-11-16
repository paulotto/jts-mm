# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Data atlas (W6): every data view x perspective on the synthetic ground truth.

Checks rendering (matplotlib overlaps, layout, Plotly, a pgf sample compiled
with LuaLaTeX at column/text width), the numbers the views draw against the
store and the truth, and graceful degradation when a modality is missing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomech import ensembles as E
from biomech.io.context import DataContext
from biomech.viz import layout as L
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import to_plotly
from biomech.viz.views import registry
from biomech.viz.views import data as D

PID = "s01"


@pytest.fixture(scope="module")
def ctx(synth_study):
    cfg = synth_study[0]
    c = DataContext(cfg)
    yield c
    c.close()


def _cases():
    out = []
    for v in registry.views("data"):
        for p in v.perspectives or (None,):
            out.append((v.id, p))
    return out


def _build(ctx, view_id, **kw):
    return registry.build(view_id, registry.request(ctx, view_id, pid=PID, **kw))


def _series(spec, kind=None, panel=None):
    panels = spec.panels if panel is None else [spec.panels[panel]]
    return [s for p in panels for s in p.series if kind is None or s.kind == kind]


def test_registry_has_the_atlas():
    ids = {v.id for v in registry.views("data")}
    assert {"data_orthographic", "data_jaw3d", "data_speed_path", "data_condylar_paths",
            "data_cohort_atlas", "data_activation_path", "data_kinematic_stack",
            "data_cycle_profiles", "data_phase_portrait", "data_demand_map", "data_imu_accel",
            "data_emg_chain", "data_emg_spectra", "data_activation_raster", "data_emg_cycle",
            "data_imu_estimators", "data_imu_quality", "data_timeline", "data_tracking_scene",
            "data_landmarks"} <= ids
    for v in registry.views("data"):
        assert v.perspectives, v.id
        assert v.defaults.get("perspective") in v.perspectives, v.id


@pytest.mark.parametrize("view_id,perspective", _cases())
def test_view_renders_clean(ctx, view_id, perspective):
    spec = _build(ctx, view_id, perspective=perspective)
    assert spec.panels
    b = render_mpl.build(spec)
    try:
        assert render_mpl.text_overlaps(b) == []
        assert L.check(b.layout) == []
    finally:
        import matplotlib.pyplot as plt

        plt.close(b.fig)
    fig = to_plotly(spec)
    assert len(fig.data) > 0 or fig.layout.annotations
    # Pseudonyms only: no pid strings in anything a reader sees.
    text = " ".join([spec.title, spec.caption, spec.perspective] +
                    [p.title + " " + p.note for p in spec.panels])
    assert not re.search(r"\bs0\d\b", text), text


PGF_SAMPLE = ("data_orthographic", "data_jaw3d", "data_speed_path", "data_activation_raster",
              "data_kinematic_stack")


@pytest.mark.skipif(shutil.which("lualatex") is None or shutil.which("pdfinfo") is None,
                    reason="no LuaLaTeX/pdfinfo")
def test_pgf_sample_compiles_within_width(ctx, tmp_path: Path):
    render_pgf.write_style(tmp_path)
    for vid in PGF_SAMPLE:
        spec = _build(ctx, vid)
        render_pgf.render_figure(spec, tmp_path)
        tex = render_pgf.standalone_figure(tmp_path, spec.id)
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                           cwd=tmp_path, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 and "pgfplots.sty' not found" in r.stdout:
            pytest.skip("pgfplots not installed")
        assert r.returncode == 0, r.stdout[-2000:]
        info = subprocess.run(["pdfinfo", tex.with_suffix(".pdf").name], cwd=tmp_path,
                              capture_output=True, text=True).stdout
        w = float(re.search(r"Page size:\s+([\d.]+) x", info).group(1))
        limit = (3.5 if spec.width == "column" else 7.16) * 72 + 1
        assert w <= limit, (vid, w, limit)


# -- numbers against the store and the truth -------------------------------------------------
def test_projections_equal_3d_coordinates(ctx):
    a = ctx.arrays("kinematics", PID)
    P = a["open_close__incisor"].astype(float)
    planes = _build(ctx, "data_orthographic")
    lab = D.TASK_LABEL["open_close"]
    for k, (ix, iy) in enumerate((D.PLANES["sagittal"], D.PLANES["frontal"],
                                  D.PLANES["horizontal"])):
        s = next(s for s in _series(planes, "line", k) if s.role == D.TASK_ROLE["open_close"])
        np.testing.assert_allclose(s.x, P[:, ix])
        np.testing.assert_allclose(s.y, P[:, iy])
    s3 = next(s for s in _series(_build(ctx, "data_jaw3d"), "line3d") if s.label == lab)
    np.testing.assert_allclose(np.column_stack([s3.x, s3.y, s3.z]), P)


def test_envelope_contains_every_path_point(ctx):
    from matplotlib.path import Path as MPath

    spec = _build(ctx, "data_orthographic")
    hull = next(s for s in _series(spec, "fill", 0))
    poly = MPath(np.column_stack([hull.x, hull.y]))
    a = ctx.arrays("kinematics", PID)
    pts = np.vstack([a[f"{t}__incisor"][:, [0, 2]] for t in D.TASKS if f"{t}__incisor" in a])
    pts = pts[np.isfinite(pts).all(axis=1)]
    inside = poly.contains_points(pts, radius=1e-6) | poly.contains_points(pts, radius=-1e-6)
    assert inside.all(), int((~inside).sum())


def test_envelope_spans_the_true_opening(ctx, synth_study):
    truth = synth_study[2][PID]
    spec = _build(ctx, "data_orthographic")
    hull = next(s for s in _series(spec, "fill", 0))
    lower_incisor_z = truth.landmarks_max["lower_incisor"][2]
    # The deepest point of the envelope is the largest opening of any task.
    assert lower_incisor_z - np.min(hull.y) >= truth.opening_mm - 0.5


def test_speed_colours_equal_velocity_norm(ctx):
    spec = _build(ctx, "data_speed_path")
    a = ctx.arrays("kinematics", PID)
    v = np.linalg.norm(a["open_close__incisor_vel"].astype(float), axis=1)
    s = _series(spec, "colorline", 0)[0]
    np.testing.assert_allclose(s.c, v)
    lim = spec.panels[1].colorbar.lim
    assert lim[0] == 0 and lim[1] >= np.nanpercentile(v, 99)


def test_min_jerk_reference_integrates_to_amplitude(ctx):
    tau = np.linspace(0, 1, 20001)
    v = D.min_jerk_velocity(tau, 25.0, 0.6)
    assert np.trapezoid(v, tau * 0.6) == pytest.approx(25.0, rel=1e-6)
    assert v.max() == pytest.approx(1.875 * 25.0 / 0.6, rel=1e-6)
    for persp in ("trials", "cohort"):
        spec = _build(ctx, "data_cycle_profiles", perspective=persp)
        mj = spec.meta["min_jerk"]
        s = next(s for s in _series(spec, "line", 1) if s.label == "minimum jerk")
        t = np.asarray(s.x) / 100.0 * mj["duration_s"]
        y = np.asarray(s.y) * mj["peak_dps"]
        op = np.asarray(s.x) <= mj["p_pct"]
        # 101-point grid: the trapezoid sum is close to, not exactly, the amplitude.
        assert np.trapezoid(y[op], t[op]) == pytest.approx(mj["amplitude_deg"], rel=0.03)
        assert np.trapezoid(y[~op], t[~op]) < 0
        assert np.nanmax(s.y) == pytest.approx(1.0, abs=0.02)


def test_demand_map_p95_equals_numpy_percentile(ctx):
    spec = _build(ctx, "data_demand_map")
    a = ctx.arrays("kinematics", PID)
    th = a["open_close__theta_vel"].astype(float)
    acc = a["open_close__theta_acc"].astype(float)
    ok = np.isfinite(th) & np.isfinite(acc)
    p95 = spec.meta["p95"][("angular", "open_close")]
    assert p95[0] == pytest.approx(np.percentile(np.abs(th[ok]), 95))
    assert p95[1] == pytest.approx(np.percentile(np.abs(acc[ok]), 95))
    vlines = [s for s in _series(spec, "vline", 0)]
    assert np.max(vlines[0].x) == pytest.approx(p95[0])
    # Cohort: median of the participants' percentiles (hierarchy kept).
    spec = _build(ctx, "data_demand_map", perspective="cohort")
    per = []
    for pid in ctx.pids():
        b = ctx.arrays("kinematics", pid)
        x, y = b["open_close__theta_vel"].astype(float), b["open_close__theta_acc"].astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        per.append(np.percentile(np.abs(x[m]), 95))
    assert spec.meta["p95"][("angular", "open_close")][0] == pytest.approx(np.median(per))


def test_cohort_perspective_equals_aggregate(ctx):
    for stat in ("median_iqr", "mean_sd"):
        spec = _build(ctx, "data_cycle_profiles", perspective="cohort", stat=stat)
        agg = E.aggregate(E.kinematic(ctx, "open_close", "theta", "cycle"), "cohort", stat)
        centre = [s for s in _series(spec, "line", 0)
                  if s.role == "accent" and s.style.get("width") == 1.3][0]
        np.testing.assert_allclose(centre.y, agg.center[0, :, 0], atol=1e-9)
        band = _series(spec, "band", 0)[0]
        np.testing.assert_allclose(band.y, agg.lo[0, :, 0], atol=1e-9)
        np.testing.assert_allclose(band.y2, agg.hi[0, :, 0], atol=1e-9)
        members = [s for s in _series(spec, "line", 0) if s.role == "context"]
        assert len(members) == int(agg.n[0]) == len(ctx.pids())


def test_trials_perspective_excludes_nothing_and_orders_cycles(ctx):
    spec = _build(ctx, "data_kinematic_stack", perspective="trials", align="cycle")
    ens = E.kinematic(ctx, "open_close", "theta", "cycle")
    n = int((ens.trials.pid == PID).sum())
    lines = [s for s in _series(spec, None, 0) if s.kind in ("colorline", "line")
             and s.role in ("primary", "noise")]
    assert len(lines) == n
    orders = sorted({float(s.c[0]) for s in lines if s.kind == "colorline"})
    assert orders == [float(k) for k in range(1, len(orders) + 1)]


def test_landmark_mad_matches_the_kinematics_stage(ctx):
    spec = _build(ctx, "data_landmarks")
    lm = ctx.table("kinematics", "landmarks", [PID]).set_index("name")
    for name, mad in spec.meta["mad_mm"].items():
        assert mad == pytest.approx(float(lm.loc[name, "mad_mm"]), rel=1e-6, abs=1e-9)


# -- graceful degradation ----------------------------------------------------------------------
class _Missing:
    """A DataContext view with one modality removed."""

    def __init__(self, ctx, drop: str):
        self._ctx, self._drop = ctx, drop

    def __getattr__(self, name):
        return getattr(self._ctx, name)

    def arrays(self, stage, pid):
        if stage == self._drop or (self._drop == "derivatives" and stage == "kinematics"):
            if self._drop == "derivatives":
                a = self._ctx.arrays(stage, pid)
                return {k: v for k, v in a.items()
                        if not k.endswith(("_vel", "_acc", "__omega"))}
            return {}
        return self._ctx.arrays(stage, pid)

    def emg(self, pid):
        if self._drop == "session":
            raise FileNotFoundError("no session files")
        return self._ctx.emg(pid)

    def previews(self, pid, take="motion"):
        return [] if self._drop == "session" else self._ctx.previews(pid, take)


@pytest.mark.parametrize("drop", ["kinematics", "emg", "imu", "derivatives", "session"])
def test_missing_modality_gives_annotated_panels(ctx, drop):
    import matplotlib.pyplot as plt

    m = _Missing(ctx, drop)
    annotated = 0
    for v in registry.views("data"):
        spec = registry.build(v.id, registry.request(m, v.id, pid=PID))
        b = render_mpl.build(spec)
        plt.close(b.fig)
        annotated += any(s.kind == "text" and p.x.ticks == [] for p in spec.panels
                         for s in p.series)
    assert annotated >= 2, drop


def test_layout_terminates_on_short_axes_and_keeps_ticks():
    """Regression: fit_y looped forever when the sparsest nice ticks were still 3, and a
    nice step could leave a single tick (e.g. step 100 on -85..82)."""
    from biomech.viz.spec import Axis, FigureSpec, Panel, Series

    ticks, _ = L.linear_ticks(-85.3, 82.1, 3)
    assert len(ticks) >= 3
    x = np.linspace(0, 1, 50)
    spec = FigureSpec("t", "t", [Panel("", Axis("x"), Axis("y"),
                                       [Series("line", x, np.sin(9 * x) * 1.0)])],
                      height_in=0.25)
    lay = L.compute(spec)
    assert 2 <= len(lay.panels[0].y.ticks) <= 3


def test_fig2_angle_and_emg_share_the_cycle_cut(ctx):
    """Fig. 2a and 2b: the same cycles cut at start..end (the SPM cycles)."""
    from biomech import ensembles as E

    emg, theta = E.profile_ensembles(ctx, "open_close")
    assert theta.n_trials == emg.n_trials > 0
    assert list(theta.trials.cycle) == list(emg.trials.cycle)
    np.testing.assert_allclose(theta.grid, emg.grid)
