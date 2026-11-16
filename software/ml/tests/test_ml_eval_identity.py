# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``ml_eval`` gives the tables of the code before the 2026-10-31 speed-up, bit for bit.

The speed-up (post-run batch) fits each inner-search fit that several outer
LOSO folds share once (same rows, grid point and seed) and runs every blocked-CV fold fit of
all configurations in one pool, a fit that two configurations share once. Neither may change
a value: :func:`test_ml_eval_tables_equal_the_reference` runs ``ml_eval_stage`` on a small
synthetic study (five participants of unequal size, two of them equal, so shared inner fits
occur; every protocol, the variants, importance and the permutation test on) and compares
every table cell and array with ``data/ml_eval_identity_reference.json``, written by the
pre-change code (the frozen ``ml`` package first on ``PYTHONPATH``)::

    PYTHONPATH=<pre-change ml>:. python -m tests.test_ml_eval_identity tests/data/ml_eval_identity_reference.json

Excluded: the wall-clock fit times (``fit_time_s`` columns, ``variants_summary`` rows of the
metric ``fit_time_s``). Floats are compared exactly (JSON keeps the shortest repr, which
round-trips), NaN equal to NaN.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.core.dataset import concat
from tests.test_review_ml_fixes import _stage_ctx
from tests.test_variants import _participant

REFERENCE = Path(__file__).parent / "data" / "ml_eval_identity_reference.json"
TIMING = "fit_time_s"
PARAMS: dict[str, Any] = {
    "configs": {"lda": ["emg_imu"], "svm": ["imu", "emg_imu", "emg_nodig"], "mlp": ["emg"]},
    "blocked": [["lda", "emg_imu"], ["svm", "imu"]],
    "primary_model": "svm", "inner_folds": 2, "max_train": 400, "n_perm": 3, "n_jobs": 2,
    "diagnostics": {"fusion_variance": True,
                    "importance": {"model": "svm", "feature_set": "emg_imu", "n_repeats": 2}},
    "variants": {"hyper": "main", "late_fusion": {"w_step": 0.5},
                 "reduced_emg": {"feature_set": "emg_imu", "features": ["log_rms", "mnf"]},
                 "few_shot": {"feature_sets": ["imu", "emg_imu"], "k": [0, 5, 10],
                              "guard_s": 0.35, "weight": "balanced", "mode": "per_phase"}},
}


def _run() -> Any:
    from ml.biomech_stages import ml_eval_stage

    ds = concat([_participant(f"s0{i}", i, n_run=n)
                 for i, n in zip(range(1, 6), (6, 8, 8, 10, 12))])
    ctx = _stage_ctx(ds, PARAMS["configs"])
    ctx.params = dict(PARAMS)
    return ml_eval_stage(ctx)


def _value(v: Any) -> Any:
    if isinstance(v, (np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else float(v)
    return v if isinstance(v, (int, bool, str)) or v is None else str(v)


def canonical(res: Any) -> dict[str, Any]:
    """JSON-ready tables (columns, dtypes, cells; timing excluded) and arrays."""
    tables = {}
    for name, df in sorted(res.tables.items()):
        df = pd.DataFrame(df)
        if name == "variants_summary" and "metric" in df:
            df = df[df["metric"] != TIMING]
        df = df.drop(columns=[c for c in df.columns if c == TIMING]).reset_index(drop=True)
        tables[name] = {"columns": [str(c) for c in df.columns],
                        "dtypes": [str(t) for t in df.dtypes],
                        "rows": [[_value(v) for v in r] for r in df.itertuples(index=False)]}
    arrays = {k: {"dtype": str(np.asarray(v).dtype), "shape": list(np.shape(v)),
                  "values": [_value(x) for x in np.asarray(v).ravel()]}
              for k, v in sorted(res.arrays.items())}
    return {"tables": tables, "arrays": arrays}


@pytest.mark.slow
def test_ml_eval_tables_equal_the_reference():
    pytest.importorskip("biomech")
    ref = json.loads(REFERENCE.read_text())
    got = json.loads(json.dumps(canonical(_run())))
    assert sorted(got["tables"]) == sorted(ref["tables"])
    for name in ref["tables"]:
        assert got["tables"][name] == ref["tables"][name], name
    assert sorted(got["arrays"]) == sorted(ref["arrays"])
    for name in ref["arrays"]:
        assert got["arrays"][name] == ref["arrays"][name], name


if __name__ == "__main__":                       # write the reference (pre-change code)
    out = Path(sys.argv[1])
    out.write_text(json.dumps(canonical(_run()), separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)")
