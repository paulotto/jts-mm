# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D21: per-participant jaw geometry (known geometry) and the electrode impedance of the
pre-take EMG check (table, summary, associations), and their views on the synthetic study."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _jaw():
    """Incisor at the origin, molars 40 mm back and 25 mm to each side, condyle points
    90 mm back, 50 mm up and 55 mm to each side; closed pose = identity."""
    pm = {"incisor": np.zeros(3), "left": np.array([-40.0, 25.0, 0.0]),
          "right": np.array([-40.0, -25.0, 0.0]), "tmj_left": np.array([-90.0, 55.0, 50.0]),
          "tmj_right": np.array([-90.0, -55.0, 50.0])}
    pm["tmj_mid"] = 0.5 * (pm["tmj_left"] + pm["tmj_right"])
    up = {"incisor": np.array([2.0, 0, 1.0]), "left": np.array([-36.0, 27.0, 1.0]),
          "right": np.array([-36.0, -27.0, 1.0])}
    return SimpleNamespace(points_mp=pm, upper_max=up)


def test_geometry_of_a_known_jaw():
    from biomech.kinematics.geometry import jaw_geometry
    from biomech.kinematics.hinge_fit import HingeResult

    # axis along y through (-100, 0, 30): 10 mm behind and 20 mm below the condyle midpoint
    h = HingeResult("hinge_fit", np.array([0.0, 1.0, 0.0]), np.array([-100.0, 7.0, 30.0]),
                    None, np.zeros(1), None, np.zeros((1, 5)))
    g = jaw_geometry(_jaw(), np.eye(4), h)
    side = np.linalg.norm([90.0, 55.0, 50.0])
    assert g["bonwill_side_mm"] == pytest.approx(side)
    assert g["bonwill_ratio"] == pytest.approx(side / 110.0)
    # occlusal plane z = 0; Bonwill plane through the incisor and (-90, ±55, 50)
    assert g["balkwill_deg"] == pytest.approx(np.degrees(np.arctan2(50.0, 90.0)))
    assert g["lower_arch_depth_mm"] == pytest.approx(40.0)
    assert g["upper_arch_depth_mm"] == pytest.approx(38.0)
    assert g["condyle_height_mm"] == pytest.approx(50.0)
    assert g["hinge_radius_mm"] == pytest.approx(np.hypot(100.0, 30.0))
    assert g["hinge_axis_dx_mm"] == pytest.approx(-10.0)
    assert g["hinge_axis_dz_mm"] == pytest.approx(-20.0)
    assert g["hinge_axis_offset_mm"] == pytest.approx(np.hypot(10.0, 20.0))
    assert g["hinge_axis_angle_deg"] == pytest.approx(0.0, abs=1e-9)
    assert set(jaw_geometry(_jaw(), None, None)) == {
        "bonwill_side_mm", "bonwill_ratio", "balkwill_deg", "lower_arch_depth_mm",
        "upper_arch_depth_mm"}                              # no closed pose: landmarks only


def _sd(t: float, z: list[float | None], passed: bool = False):
    ch = [{"channel": k + 1, "name": m, "z_kohm": v, "z_quality": "good",
           "noise_rms_uv": 5.0, "noise_ok": True, "line_peak_uv": 1.0, "line_ok": True}
          for k, (m, v) in enumerate(zip(("masseter_r", "digastric_r"), z))]
    return SimpleNamespace(meta={"created_unix_utc": str(t), "emg_check_json": json.dumps(
        {"passed": passed, "channels": ch, "line_hz_measured": 50.0})})


def test_check_table_orders_takes_and_flags_contact():
    from biomech.qc.impedance import check_table, summary

    t = check_table([("motion", _sd(1000.0 + 180, [20.0, 60.0])),
                     ("mvc", _sd(1000.0, [25.0, 70.0])), ("x", None)], "p1")
    m = t[t["take"] == "motion"].set_index("muscle")
    assert (t[t["take"] == "mvc"]["order"] == 1).all() and (m["order"] == 2).all()
    assert m.loc["masseter_r", "minutes"] == pytest.approx(3.0)
    assert m.loc["masseter_r", "contact_ok"] and not m.loc["digastric_r", "contact_ok"]
    s = summary(t).set_index(["take", "muscle"])
    assert s.loc[("motion", "all"), "above_limit_n"] == 1
    assert s.loc[("second_minus_first", "all"), "z_median_kohm"] == pytest.approx(-7.5)
    assert check_table([("motion", SimpleNamespace(meta={}))], "p1").empty


def test_check_table_names_channels_as_cabled():
    """The check JSON of files before 2026-11-03 names CH3 digastric_l; the muscle comes from
    the channel number and the session's corrected names (an03 §1)."""
    from biomech.io.session import MUSCLES
    from biomech.qc.impedance import check_table

    sd = _sd(1000.0, [20.0, 30.0])
    d = json.loads(sd.meta["emg_check_json"])
    d["channels"] = [dict(d["channels"][0], channel=3, name="digastric_l")]
    sd.meta["emg_check_json"] = json.dumps(d)
    sd.muscle_names = lambda: list(MUSCLES)
    t = check_table([("motion", sd)], "p1")
    assert list(t["muscle"]) == ["digastric_r"]


def test_associations_hold_participant_and_muscle_fixed():
    from biomech.qc.impedance import associations

    rng = np.random.default_rng(0)
    rows, emg = [], []
    for i in range(12):
        off = rng.normal(0, 20)                     # participant level: shared by both
        for k, mu in enumerate(("masseter_r", "temporalis_r", "digastric_r")):
            z = 20 + 15 * k + off + rng.normal(0, 5)
            rows.append({"pid": f"p{i}", "take": "motion", "muscle": mu, "z_kohm": z,
                         "noise_rms_uv": 5.0, "line_peak_uv": 1.0})
            emg.append({"pid": f"p{i}", "muscle": mu, "snr_db": 30 - 5 * k - 0.2 * off,
                        "rest_lf_uv": 1 + 0.1 * z, "rest_rms_uv": rng.normal(5, 1),
                        "mains_share_raw": rng.uniform()})
    a = associations(pd.DataFrame(rows), pd.DataFrame(emg), seed=1).set_index(
        "recorded_quantity")
    # SNR follows muscle and participant, not |Z| itself: pooled strong, adjusted ~ 0
    snr = a[a.check_quantity == "z_kohm"].loc["snr_db"]
    assert snr.rho_pooled < -0.7 and abs(snr.rho_adj) < 0.3
    lf = a[a.check_quantity == "z_kohm"].loc["rest_lf_uv"]
    assert lf.rho_adj > 0.7 and lf.p_holm < 0.01
    assert lf.rho_pooled_lo <= lf.rho_pooled <= lf.rho_pooled_hi


# -- on the synthetic study ---------------------------------------------------------------------
STUDY_ID = re.compile(r"\bs\d{2}\b")


@pytest.fixture(scope="module")
def ctx(synth_study):
    from biomech.io.context import DataContext

    c = DataContext(synth_study[0])
    yield c
    c.close()


def test_stages_store_the_d21_tables(ctx):
    chk = ctx.table("qc", "emg_check")
    assert len(chk) == 6 * 2 * len(ctx.pids())
    assert set(ctx.table("qc_cohort", "impedance")["take"]) >= {"motion", "mvc",
                                                                "second_minus_first"}
    assert not ctx.table("qc_cohort", "impedance_assoc").empty
    geo = ctx.table("kinematics", "geometry")
    assert set(geo.pid) == set(ctx.pids()) and geo.hinge_radius_mm.notna().all()
    rv = ctx.table("stats_kinematics", "reference_values")
    assert {"balkwill_deg", "hinge_radius_mm"} <= set(rv.metric)
    cs = ctx.table("stats_kinematics", "condyle_sensitivity")
    assert "balkwill_deg" in set(cs.metric) and "hinge_radius_mm" not in set(cs.metric)


@pytest.mark.parametrize("view_id", ["emg_impedance", "table_impedance", "table_geometry",
                                     "kin_hinge_axis_location", "qc_heatmap_emg_check"])
def test_d21_views_build_and_render(ctx, view_id, tmp_path):
    from biomech.viz import render_mpl, render_pgf
    from biomech.viz.spec import TableSpec
    from biomech.viz.views import registry as R

    spec = R.build(view_id, R.request(ctx, view_id))
    assert not STUDY_ID.search(repr(spec)) and spec.sources
    if isinstance(spec, TableSpec):
        assert spec.rows
        assert render_pgf.render_table(spec, tmp_path).exists()
        return
    assert any(s.kind == "scatter" or s.kind == "heatmap" for p in spec.panels
               for s in p.series)
    assert render_mpl.render_figure(spec, tmp_path / f"{view_id}.png", dpi=60).exists()
    assert render_pgf.render_figure(spec, tmp_path).exists()


def test_geometry_table_numbers_are_the_stored_ones(ctx):
    from biomech.viz.views import registry as R

    spec = R.build("table_geometry", R.request(ctx, "table_geometry"))
    rv = ctx.table("stats_kinematics", "reference_values")
    r = rv[(rv.metric == "hinge_radius_mm") & (rv.group == "all")].iloc[0]
    row = [x for x in spec.rows if x["q"].lower().startswith("hinge radius")][0]
    assert row["all"].startswith(f"{r['median']:.1f} [".replace("-", "−"))


def test_math_headers_are_never_split():
    from biomech.viz.render_pgf import _wrap_header

    assert _wrap_header("$p_{\\mathrm{Holm}}$", "", 8) == "$p_{\\mathrm{Holm}}$"
    assert _wrap_header("Check vs recorded", "", 8).startswith("\\shortstack{")


def test_rescale_divides_by_the_channel_scale_and_keeps_the_recorded_value():
    """B1: |Z| / k of the channel (CH number -> muscle as cabled); unchanged without a scale."""
    import pandas as pd

    from biomech.io.session import MUSCLES
    from biomech.qc.impedance import rescale

    chk = pd.DataFrame({"muscle": [MUSCLES[2], MUSCLES[0]], "z_kohm": [55.0, 20.0],
                        "contact_ok": [False, True]})
    assert rescale(chk, None) is chk and rescale(chk, {}) is chk
    out = rescale(chk, {3: 1.25})
    assert MUSCLES[2] == "digastric_r"
    assert list(out["z_kohm"]) == [44.0, 20.0] and list(out["z_kohm_recorded"]) == [55.0, 20.0]
    assert list(out["contact_ok"]) == [True, True]
    import pytest
    with pytest.raises(ValueError, match="> 0"):
        rescale(chk, {1: 0.0})
