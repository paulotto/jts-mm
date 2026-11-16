# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml_online with the replays on a reserved core (overlapping the calibration refits) gives
the same decisions and tables as the serial schedule; its fold models come from the model
cache ml_eval filled (synthetic biomech study, slow)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.slow

TIMING = {"chunk_ms_median", "chunk_ms_p99", "chunk_ms_max", "realtime_factor",
          "calibration_ms", "replay_wall_s", "fit0_s", "cal_fit_s"}


def test_replay_overlap_gives_the_serial_tables(tmp_path, monkeypatch):
    pytest.importorskip("biomech")
    from biomech.config import load_config
    from biomech.pipeline import run
    from biomech.results import ResultStore
    from biomech.testing.synth import DEFAULT_SPECS, write_study

    from ml.protocols import pool as pl

    if pl.reserve_core() is None:
        pytest.skip("no CPU affinity / too few CPUs for a reserved replay core")
    fast = {"ml_eval": {"configs": {"lda": [], "svm": ["emg_imu", "imu"], "mlp": []},
                        "blocked": [], "n_perm": 0, "n_jobs": 2,
                        "variants": {"late_fusion": {"w_step": 0.5},
                                     "few_shot": {"feature_sets": ["imu"], "k": [0, 20]}},
                        "diagnostics": {"fusion_variance": False}},
            "ml_online": {"n_jobs": 3}}
    path, _ = write_study(tmp_path, [replace(s, emg_fs=2000) for s in DEFAULT_SPECS[:3]],
                          stages=fast)
    cfg = load_config(path)
    monkeypatch.setenv("ML_ONLINE_REPLAY_OVERLAP", "1")
    assert run(cfg, ["ml_online"], n_jobs=4).n_failed == 0
    store = ResultStore(cfg.out_dir)
    assert list((cfg.out_dir / "model_cache").glob("*.joblib"))
    first = {t: store.read_table("ml_online", t, ["_cohort"])
             for t in store.meta("ml_online", "_cohort").tables}
    arr1 = store.read_arrays("ml_online", "_cohort")
    s1 = first["summary"]
    assert (s1.hyper == "main_loso").all()

    monkeypatch.setenv("ML_ONLINE_REPLAY_OVERLAP", "0")
    assert run(cfg, ["ml_online"], force=["ml_online"], n_jobs=4).n_failed == 0
    for t, a in first.items():
        b = store.read_table("ml_online", t, ["_cohort"])
        drop = [c for c in a.columns if c in TIMING]
        pd.testing.assert_frame_equal(a.drop(columns=drop).reset_index(drop=True),
                                      b.drop(columns=drop).reset_index(drop=True),
                                      check_exact=True, obj=t)
    arr2 = store.read_arrays("ml_online", "_cohort")
    assert set(arr1) == set(arr2)
    for k in arr1:
        np.testing.assert_array_equal(arr1[k], arr2[k], err_msg=k)
