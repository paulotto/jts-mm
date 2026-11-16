# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""X10 view ``ml_ablation``: paired differences from the stored variants and the main run."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from biomech.viz.views import results_ml as R


def _ctx(variants: pd.DataFrame, loso: pd.DataFrame):
    tabs = {"variants": variants, "loso": loso}
    return SimpleNamespace(store=SimpleNamespace(stages=lambda: ["ml_eval"]),
                           table=lambda st, t: tabs.get(t, pd.DataFrame()),
                           pids=lambda: ["s01", "s02"])


def test_ablation_deltas_and_view():
    v = pd.DataFrame({"variant": ["muscle_ablation"] * 4 + ["reduced_emg"],
                      "feature_set": ["emg-masseter+imu", "emg-masseter+imu",
                                      "emg-digastric_r+imu", "emg-digastric_r+imu",
                                      "emg[log_rms,mnf]+imu"],
                      "pid": ["s01", "s02", "s01", "s02", "s01"],
                      "balanced_accuracy": [0.70, 0.62, 0.80, 0.66, 0.5]})
    lo = pd.DataFrame({"pid": ["s01", "s02", "s01"], "feature_set": ["emg_imu", "emg_imu", "imu"],
                       "model": ["svm", "svm", "svm"], "balanced_accuracy": [0.75, 0.65, 0.1]})
    d = R.ablation_deltas(_ctx(v, lo)).set_index(["group", "pid"])
    assert d.loc[("masseter", "s01"), "delta"] == pytest.approx(-0.05)
    assert d.loc[("digastric_r", "s02"), "delta"] == pytest.approx(0.01)
    spec = R.ml_ablation(SimpleNamespace(ctx=_ctx(v, lo)))
    ticks = spec.panels[0].y.ticklabels
    assert ticks == ["masseter, 2/2 lower", "digastric R, 0/2 lower"]
    empty = R.ml_ablation(SimpleNamespace(ctx=_ctx(pd.DataFrame(), lo)))
    assert "no muscle-ablation" in empty.panels[0].note
