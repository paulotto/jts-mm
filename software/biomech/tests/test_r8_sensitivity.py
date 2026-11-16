# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""D36 / S5.4: sensitivity variants built on another variant (``base:``) and the
between-day swap of a participant's sessions (``cohort_swap``)."""

from __future__ import annotations

import pytest
import yaml

from biomech import sensitivity as S
from biomech.config import ConfigError

SPECS = {"reporting_population": {"exclude_flags": ["outside_protocol"]},
         "burst": {"base": "reporting_population", "set": {"stages.timing.burst_min_db": 3.0}},
         "nested": {"base": "burst", "set": {"stages.timing.burst_min_db": 10.0,
                                             "stages.timing.k_sd": 5.0}},
         "day2": {"base": "reporting_population", "cohort_swap": {"s02": 1}},
         "loop_a": {"base": "loop_b"}, "loop_b": {"base": "loop_a"}}


def test_changes_follow_the_base_chain():
    sets, flags, swap = S._changes(SPECS, "nested")
    assert sets == {"stages.timing.burst_min_db": 10.0, "stages.timing.k_sd": 5.0}
    assert flags == ["outside_protocol"] and swap == {}
    assert S._changes(SPECS, "day2") == ({}, ["outside_protocol"], {"s02": 1})
    with pytest.raises(ConfigError, match="circular"):
        S._changes(SPECS, "loop_a")


def test_cohort_swap_takes_the_second_day(tmp_path):
    rows = {"participants": [
        {"pid": "s02", "motion": "session_s02_f_44_20250303T101000Z_a",
         "mvc": "session_s02_f_44_mvc_20250303T100000Z_b",
         "secondary": ["session_s02_f_44_20250306T101000Z_c",
                       "session_s02_f_44_mvc_20250306T100000Z_d"]},
        {"pid": "s09", "motion": "m7", "flags": ["outside_protocol"]}]}
    src = tmp_path / "cohort.yaml"
    src.write_text(yaml.safe_dump(rows))
    out = S.filtered_cohort(src, tmp_path / "v" / "cohort.yaml", ["outside_protocol"], "day2", {"s02": 1})
    doc = yaml.safe_load(out.read_text())
    s02, s09 = doc["participants"]
    assert s02["motion"].endswith("20250306T101000Z_c")
    assert s02["mvc"].endswith("20250306T100000Z_d")
    assert sorted(s02["secondary"]) == ["session_s02_f_44_20250303T101000Z_a",
                                        "session_s02_f_44_mvc_20250303T100000Z_b"]
    assert s09["include"] is False and "flag outside_protocol" in s09["exclusion_reason"]
    with pytest.raises(ConfigError, match="no secondary motion take for day 2"):
        S.filtered_cohort(src, tmp_path / "w.yaml", [], "x", {"s02": 2})
