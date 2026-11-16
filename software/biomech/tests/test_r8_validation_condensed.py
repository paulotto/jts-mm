# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Condensed systems-paper items (``viz/views/validation_condensed``, 2026-11-05): the combined
figures are registered next to their sources, the new tables read the stored values (missing
sources give dashes, not errors), and the methods table reports the held-out residuals per
body next to the jaw-in-head pose."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.check.text import caption_rubric_issues, text_issues
from biomech.viz.spec import TableSpec
from biomech.viz.views import registry as R
from biomech.viz.views import validation_condensed as VC
from biomech.viz.views import validation_gallery as VG
from biomech.viz.views import vocab as V

PIDS = ("s01", "s02")


class Ctx:
    def __init__(self, tables: dict):
        self._t = tables
        self.config = type("C", (), {"seed": 0})()

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1

    def pseudonyms(self):
        return {p: i + 1 for i, p in enumerate(PIDS)}

    def table(self, stage, name, pids=None):
        d = self._t.get((stage, name), pd.DataFrame())
        return d[d["pid"].isin(pids)] if pids is not None and "pid" in d else d

    def arrays(self, stage, pid):
        return {}


@pytest.fixture()
def ctx():
    ft = pd.DataFrame([{"pid": p, "n_intervals": 1000, "long_max_ms": 75.0} for p in PIDS])
    gaps = pd.DataFrame([{"pid": p, "missing_frames": m, "n": n} for p in PIDS
                         for m, n in ((1, 30), (2, 5))])
    sync = pd.DataFrame([{"pid": p, "cam_assign_drift_ppm": d, "heldout_mp_pos_rms_mm": 0.5,
                          "heldout_hp_pos_rms_mm": 0.7, "heldout_rel_pos_rms_mm": 10.0,
                          "heldout_rel_rot_rms_deg": 2.0, "qtm_fs_hz": 200.0,
                          "alpha_ppm": 12.0, "se_alpha_ppm_hac": 4.0, "se_offset_ms_hac": 0.5}
                         for p, d in zip(PIDS, (0.4, -5.4))])
    av = pd.DataFrame([{"pid": p, "system": s, "segment": t, "n": 100, "valid_frac": 1.0}
                       for p in PIDS for s in ("jtsmm", "qualisys")
                       for t in ("rest", "open_close", "chewing")])
    held = pd.DataFrame([{"pid": p, "interval": t, "body": "HP", "n": n, "rot_rms_deg": r}
                         for p in PIDS for t, n, r in (("rest", 100, 9.0),
                                                       ("open_close", 300, 3.0),
                                                       ("chewing", 100, 5.0))])
    hinge = pd.DataFrame([{"pid": p, "task": "open_close", "system": s, "hinge_fit_mm": 2.0,
                           "hinge_tmj_mm": 4.0} for p in PIDS for s in ("jtsmm", "qualisys")])
    return Ctx({("ref_diagnostics", "frame_timing"): ft, ("ref_diagnostics", "frame_gaps"): gaps,
                ("ref_sync", "sync"): sync, ("ref_quality", "availability"): av,
                ("ref_sync", "heldout"): held, ("ref_hinge", "hinge"): hinge})


def _build(ctx, vid, **params):
    return V.reader_spec(R.build(vid, R.request(ctx, vid, **params)))


def test_condensed_items_are_registered_validation_views():
    for vid in ("val_marking_combined", "val_motion_agreement", "val_sync_combined",
                "val_table_integrity", "val_table_emg_bench", "val_table_fitness"):
        assert R.get(vid).group == "validation", vid
    assert "pid" in R.get("val_sync_combined").params


def test_integrity_counts_dropped_frames_over_the_expected_frames(ctx):
    t = _build(ctx, "val_table_integrity")
    assert isinstance(t, TableSpec)
    p1 = t.rows[0]
    assert p1["pid"] == "P1"
    assert p1["dropped"] == f"{100 * 40 / 1040:.1f}"              # 30 x 1 + 5 x 2 missing
    assert p1["valid_jtsmm"] == "100.0" and p1["drift"] == "0.4"
    assert t.rows[-1]["pid"] == "Mean" and t.rows[-1]["drift"] == "-2.5"
    assert "not stored" in t.notes.lower()
    assert not text_issues(t) and not caption_rubric_issues(t)


def test_fitness_reads_what_is_stored_and_dashes_the_rest(ctx):
    t = _build(ctx, "val_table_fitness")
    vals = [r for r in t.rows if not r.get("_head")]
    assert len(vals) == len(VC.FITNESS)
    assert vals[0]["p0"] == "2.00" and vals[0]["mean"] == "2.00"   # hinge, stored
    assert vals[4]["p0"] == "–" and vals[4]["mean"] == "–"         # camera-free, absent
    assert not text_issues(t) and not caption_rubric_issues(t)


def test_emg_bench_uses_the_dataset_current(ctx, monkeypatch):
    imp = pd.DataFrame([{"current_na": c, "channel": ch, "k": k, "k_min": 0.9, "k_max": 1.0,
                         "max_abs_err_pct": 5.0, "repeat_max_pct": 0.1}
                        for c, k in ((6, 0.95), (24, 0.97)) for ch in ("1", "all")])
    noi = pd.DataFrame([{"channel": ch, "noise_median_uv": 0.5, "noise_min_uv": 0.5,
                         "noise_max_uv": 0.7, "limit_uv": 10.0, "vivo_noise_median_uv": 12.0,
                         "ratio_bench_vivo": 0.04, "n_vivo": 13} for ch in ("1", "all")])
    monkeypatch.setattr(VC, "_bench", lambda n: imp if "impedance" in n else noi)
    t = _build(ctx, "val_table_emg_bench")
    rows = [r for r in t.rows if not r.get("_rule")]
    assert [r["channel"] for r in rows] == ["CH1", "All channels"]
    assert rows[0]["k"] == "0.95" and rows[1]["k"] == "0.95 (0.90–1.00)"
    assert "24 nA" in t.notes and "0.97" in t.notes
    assert not text_issues(t) and not caption_rubric_issues(t)


def test_methods_reports_held_out_residuals_per_body(ctx):
    assert VG._heldout_rot(ctx.table("ref_sync", "heldout"), "s01", "HP") == pytest.approx(
        np.sqrt((300 * 9 + 100 * 25) / 400))                    # movement tasks, rest left out
    t = _build(ctx, "val_table_methods")
    assert t.rows[0]["heldout_body"] == "0.50 / 0.70"
    assert t.rows[0]["heldout"] == "10.00 / 2.00"
    assert "frame diagnostic" in t.notes
