# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Participant-major and stage-major scheduling give the same store (W3.2).

The synthetic study of the session fixture ran participant-major (the default: shared
sessions/EMG preparation per participant, detached ``uncertainty`` tasks, cohort stages
released per dependency); here it is run again stage-major into a second store and every
table and array is compared exactly (wall-clock columns aside). The plugin (ML) cohort
units are linked from the first store instead of re-run: they are cached in the second
run only if every participant unit they read got the same key there.
"""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from biomech.config import load_config
from biomech.pipeline import registry, run
from biomech.results import COHORT_UNIT, ResultStore

pytestmark = pytest.mark.slow


TIMING_COLS = ("fit_time_s", "elapsed_s", "time_s")     # wall-clock measurements (suffixes)
# Wall-clock columns of the pseudo-online stage (ml_online, X4).
TIMING_NAMES = {"chunk_ms_median", "chunk_ms_p99", "chunk_ms_max", "realtime_factor",
                "calibration_ms", "replay_wall_s", "fit0_s", "cal_fit_s"}


def _assert_same(a, b, where: str) -> None:
    if isinstance(a, pd.DataFrame):
        drop = [c for c in a.columns if str(c).endswith(TIMING_COLS) or c in TIMING_NAMES]
        a, b = a.drop(columns=drop), b.drop(columns=[c for c in drop if c in b.columns])
        # long tables with a timing metric row (e.g. ml_eval variants_summary)
        for c in [c for c in a.columns if a[c].dtype == object]:
            keep = ~a[c].astype(str).str.endswith(TIMING_COLS)
            a, b = a[keep.values], b[keep.values]
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_exact=True, obj=where)
        return
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape and a.dtype == b.dtype, where
    np.testing.assert_array_equal(a, b, err_msg=where)          # exact; NaN == NaN


def _failures(summary, store) -> str:
    """The failed units of a run with their stored errors (assertion message)."""
    out = []
    for stage, units in summary.statuses.items():
        for unit, status in units.items():
            if status == "failed":
                m = store.meta(stage, unit)
                out.append(f"{stage}/{unit}: {getattr(m, 'error', None)}")
    return "; ".join(out)


def test_stage_major_store_equals_participant_major(synth_study, tmp_path):
    cfg, store, _, summary = synth_study
    assert summary.n_failed == 0, _failures(summary, store)
    cfg2 = load_config(cfg.source, out_dir=tmp_path / "stage_major")
    linked = [n for n, st in registry().items() if st.scope == "cohort"
              and st.func.__module__.split(".")[0] != "biomech" and n in store.stages()]
    for n in linked:
        shutil.copytree(store.root / "cache" / n, cfg2.out_dir / "cache" / n)
    s2 = run(cfg2, None, n_jobs=4, scheduler="stage")
    assert s2.n_failed == 0, _failures(s2, ResultStore(cfg2.out_dir))
    for n in linked:                       # same upstream keys, so nothing to re-run
        assert s2.statuses[n] == {COHORT_UNIT: "cached"}, n
    other = ResultStore(cfg2.out_dir)
    assert other.stages() == store.stages()
    for stage in store.stages():
        assert other.units(stage) == store.units(stage), stage
        for unit in store.units(stage):
            m1, m2 = store.meta(stage, unit), other.meta(stage, unit)
            assert m1.key == m2.key and set(m1.tables) == set(m2.tables), (stage, unit)
            for t in m1.tables:
                _assert_same(store.read_table(stage, t, [unit]),
                             other.read_table(stage, t, [unit]), f"{stage}/{unit}/{t}")
            a1, a2 = store.read_arrays(stage, unit), other.read_arrays(stage, unit)
            assert set(a1) == set(a2), (stage, unit)
            for k in a1:
                _assert_same(a1[k], a2[k], f"{stage}/{unit}/{k}")
