# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The JTS frame intervals use the session's trigger period explicitly."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import h5py
import numpy as np


def test_jts_export_passes_the_trigger_period(tmp_path, monkeypatch):
    from jtsmm.capture import supervisor as sup_mod

    h5 = tmp_path / "session.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("triggers")
        t = 1_000_000 + np.arange(100, dtype=np.uint64) * 33_333
        g["t_master_us"] = t
        g["trigger_id"] = np.arange(1, 101, dtype=np.uint64)
        g.attrs["rate_hz"] = 30
        f.create_group("optical")["t_master_us"] = t[::3]  # gappy optical rows

    seen: dict = {}

    def fake_intervals(timed, t_opt, *, period_us=None):
        seen["period_us"] = period_us
        return [("p", "experiment", 0, 10)]

    fake = types.ModuleType("optical.tracking.experiment")
    fake.RecordedPhase = lambda **kw: SimpleNamespace(**kw)
    fake.build_jts_config = lambda recorded, **kw: {"recorded": recorded}
    fake.write_jts_config = lambda out, cfg: seen.update(out=out, cfg=cfg)
    monkeypatch.setitem(sys.modules, "optical.tracking.experiment", fake)
    monkeypatch.setattr(sup_mod, "optical_frame_intervals", fake_intervals)

    plan = SimpleNamespace(slug="p", description="")
    sup_mod._maybe_write_jts_config(h5, plan=plan, timed=[object()], jts_config_path=None)
    assert abs(seen["period_us"] - 33_333) < 1.0
    assert seen["out"] == tmp_path / "p_jts_config.json"
