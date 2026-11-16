# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every stored analysis output is shown by some view (W6).

Builds every registered view (every declared phase and perspective) on the synthetic
study under access tracking; a stored table or array family that no view reads -- and
that is not an INTERNAL input of another stage -- fails the test. Adding a stage output
therefore means adding (or extending) a view.
"""

from __future__ import annotations

import pytest

from biomech.check import coverage as C
from biomech.io.context import DataContext
from biomech.results.access import family

pytestmark = pytest.mark.slow


def test_family_tokens():
    assert family("open_close__norm_pct") == "{task}__norm_pct"
    assert family("spectral__chewing__masseter_r__mean") == "spectral__{task}__{muscle}__mean"
    assert family("pred__s03__emg_imu__t_us") == "pred__{pid}__emg_imu__t_us"
    assert family("confusion__emg_imu|svm") == "confusion__emg_imu|{model}"
    assert family("mvc_closing__t_us") == "{task}__t_us"


def test_every_output_has_a_view(synth_study):
    cfg, store, _, _ = synth_study
    ctx = DataContext(cfg)
    rep = C.report(ctx)
    assert not rep["errors"], rep["errors"][:10]
    missing = rep["tables"] + rep["arrays"]
    assert not missing, "outputs without a view (add a view or an INTERNAL reason):\n" + \
        "\n".join(f"  {s}:{n}" for s, n in missing)
