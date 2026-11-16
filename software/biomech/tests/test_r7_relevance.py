# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D19: EMG activity relevance (no store needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from biomech.emg.relevance import relevance


def _prep():
    rng = np.random.default_rng(0)
    norm = np.column_stack([rng.normal(2.0, 0.5, 2000), rng.normal(1.0, 0.2, 2000)])
    mask = np.zeros_like(norm, bool)
    mask[:100, 0] = True                                # masked samples are ignored
    norm[:100, 0] = 90.0
    rest = SimpleNamespace(norm=norm, mask=mask)
    return SimpleNamespace(names=["masseter_r", "digastric_r"],
                           phase=lambda name: rest)


def test_active_needs_rest_plus_k_sd_and_the_floor():
    act = pd.DataFrame({"task": ["rest", "open_close", "open_close", "chewing", "chewing"],
                        "muscle": ["masseter_r", "masseter_r", "digastric_r", "masseter_r",
                                   "digastric_r"],
                        "p95_pct": [3.0, 3.2, 30.0, 40.0, 4.0]})
    r = relevance(_prep(), act, "s01").set_index(["task", "muscle"])
    assert "rest" not in r.index.get_level_values(0)
    m = r.loc[("open_close", "masseter_r")]
    assert abs(m.rest_mean_pct - 2.0) < 0.05 and abs(m.rest_sd_pct - 0.5) < 0.05
    assert not m.active and m.reason == "below rest + k SD"     # 3.2 < 2 + 3 x 0.5
    assert r.loc[("open_close", "digastric_r")].active
    assert r.loc[("chewing", "masseter_r")].active
    d = r.loc[("chewing", "digastric_r")]                      # above rest + 3 SD (1.6) ...
    assert not d.active and d.reason == "below 5 %"             # ... but under the floor
    assert d.threshold_pct == 5.0
