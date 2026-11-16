# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""B2 shorted-input noise bench check: ``scripts/bench/emg_noise_bench.py`` on saved check
tables in the format of ``jtsmm.emg.check.format_check_table``."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture()
def bench():
    sys.path.insert(0, str(ROOT / "scripts" / "bench"))
    try:
        import emg_noise_bench
        yield emg_noise_bench
    finally:
        sys.path.remove(str(ROOT / "scripts" / "bench"))


def _table(noise: dict[int, float]) -> str:
    """A saved check table (header, rule, six channels, DRL), as the check prints it."""
    lines = [f"{'CH':>3}  {'name':<14} {'leadoff':>7}  {'|Z| kΩ':>7}  {'noise µV':>9}  "
             f"{'50Hz µV':>8}  status", "-" * 68]
    for ch in range(1, 7):
        lines.append(f"{ch:3d}  {'m' + str(ch):<14} {'PASS':>7}  {'    n/a'}  "
                     f"{noise[ch]:9.1f}  {1.5:8.1f}  PASS")
    lines.append(f"{'DRL':>3}  {'drl':<14} {'PASS':>7}  {'n/a':>7}  {'n/a':>9}  {'n/a':>8}  PASS")
    return "\n".join(lines) + "\n"


def test_noise_per_channel_and_in_vivo_comparison(bench, tmp_path: Path, monkeypatch):
    f = tmp_path / "b"
    f.mkdir()
    for rep, add in (("1", 0.0), ("2", 0.2), ("3", 0.4)):
        (f / f"check_{rep}.txt").write_text(
            "EMG check …\n" + _table({c: 2.0 * c + add for c in range(1, 7)}), encoding="utf-8")
    (f / "notes.md").write_text("bias default\n", encoding="utf-8")
    cap = bench.read_captures(f)
    assert len(cap) == 18 and set(cap["channel"]) == set(range(1, 7))

    vivo = pd.DataFrame({"pid": ["p1", "p2"] * 6, "channel": np.repeat(range(1, 7), 2),
                         "noise_uv": [10.0, 20.0] * 6, "line_peak_uv": 3.0})
    tab = bench.channel_table(cap, vivo).set_index("channel")
    assert tab.loc["1", "noise_median_uv"] == pytest.approx(2.2)
    assert tab.loc["1", "noise_min_uv"] == pytest.approx(2.0)
    # bench 12.2 µV on CH6 is below its 15 µV limit; 2.2 µV on CH1 below half of 10 µV
    assert not tab.loc["6", "above_limit"] and tab.loc["1", "below_half_limit"]
    assert tab.loc["1", "ratio_bench_vivo"] == pytest.approx(np.median([2.2 / 10, 2.2 / 20]))
    assert tab.loc["1", "vivo_excess_median_uv"] == pytest.approx(
        np.median([np.sqrt(100 - 2.2 ** 2), np.sqrt(400 - 2.2 ** 2)]))
    assert tab.loc["all", "n_channels"] == 6 and tab.loc["all", "vivo_above_limit_n"] == 6
    # 20 µV exceeds every limit, 10 µV none (CH1-4: not above 10; CH5-6: below 15)


def test_incomplete_table_is_refused(bench, tmp_path: Path):
    f = tmp_path / "b"
    f.mkdir()
    (f / "check_1.txt").write_text("\n".join(_table({c: 1.0 for c in range(1, 7)})
                                             .splitlines()[:5]), encoding="utf-8")
    with pytest.raises(ValueError, match="channel rows"):
        bench.read_captures(f)
