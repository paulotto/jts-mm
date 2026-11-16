# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""B1 electrode-impedance scale bench check (dataset analysis plan): the script
``scripts/bench/emg_impedance_bench.py`` on synthetic exports, and the ``bench:`` numbers
source that reads its tables."""

from __future__ import annotations

import json
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
        import emg_impedance_bench
        yield emg_impedance_bench
    finally:
        sys.path.remove(str(ROOT / "scripts" / "bench"))


def _write(folder: Path, k: dict[int, float], *, missing: tuple = (), i_na: int = 6,
           true: dict[float, float] | None = None) -> None:
    """Exports R<r>_CH<c>_<i>nA_<a|b>.json with |Z| = k[c] * R_true (b 0.2 kΩ higher) and a
    resistors.csv with one resistor per value moved over every channel."""
    true = true or {10.0: 9.95, 22.0: 22.1, 47.0: 46.8, 100.0: 100.4}
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"nominal_kohm": list(true), "channel": "all",
                  "multimeter_kohm": list(true.values())}).to_csv(folder / "resistors.csv",
                                                                   index=False)
    for r, rt in true.items():
        for c in range(1, 7):
            for rep, dz in (("a", 0.0), ("b", 0.2)):
                if (r, c, rep) in missing:
                    continue
                chans = [{"channel": j, "name": f"m{j}", "z_kohm": (k[c] * rt + dz) if j == c
                          else 900.0, "v_tone_uv": 1.0, "quality": "good", "note": ""}
                         for j in range(1, 7)]
                (folder / f"R{r:g}_CH{c}_{i_na}nA_{rep}.json").write_text(
                    json.dumps({"meta": {"current_na": i_na}, "channels": chans}))


def test_scale_and_decision_within_tolerance(bench, tmp_path: Path):
    k = {1: 1.02, 2: 0.98, 3: 1.00, 4: 1.05, 5: 0.97, 6: 1.01}
    _write(tmp_path / "b", k)
    assert bench.main([str(tmp_path / "b"), "--tables", str(tmp_path / "t")]) == 0
    tab = pd.read_csv(tmp_path / "t" / "emg_impedance_channels.csv", dtype={"channel": str})
    per = tab[tab["channel"] != "all"].set_index("channel")
    # the a captures are exactly k * R; the b captures add 0.2 kΩ, so k is a little higher
    for c, kc in k.items():
        assert per.loc[str(c), "k"] == pytest.approx(kc, abs=0.01)
        assert per.loc[str(c), "repeat_max_kohm"] == pytest.approx(0.2)
    a = tab[tab["channel"] == "all"].iloc[0]
    assert a["decision"] == "stands" and not a["rank_affected"]
    assert a["k_min"] == pytest.approx(0.97, abs=0.01) and a["k_max"] == pytest.approx(1.05,
                                                                                     abs=0.01)
    cap = pd.read_csv(tmp_path / "t" / "emg_impedance_captures.csv")
    assert len(cap) == 48 and set(cap["r_source"]) == {"multimeter"}
    # the error is against the multimeter value, not the nominal one
    row = cap[(cap["channel"] == 3) & (cap["nominal_kohm"] == 22) & (cap["rep"] == "a")].iloc[0]
    assert row["r_true_kohm"] == 22.1 and row["rel_err"] == pytest.approx(0.0, abs=1e-12)


def test_erratum_when_a_channel_is_off_and_missing_captures_reported(bench, tmp_path: Path,
                                                                        capsys):
    k = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.15}
    _write(tmp_path / "b", k, missing=((47.0, 2, "b"),))
    assert bench.main([str(tmp_path / "b"), "--tables", str(tmp_path / "t")]) == 0
    out = capsys.readouterr().out
    assert "R47_CH2_6nA_b" in out
    tab = pd.read_csv(tmp_path / "t" / "emg_impedance_channels.csv", dtype={"channel": str})
    a = tab[tab["channel"] == "all"].iloc[0]
    assert a["decision"].startswith("erratum") and a["rank_affected"]
    assert not tab.set_index("channel").loc["6", "within_tol"]


def test_missing_multimeter_value_fails_unless_allowed(bench, tmp_path: Path):
    _write(tmp_path / "b", {c: 1.0 for c in range(1, 7)})
    res = pd.read_csv(tmp_path / "b" / "resistors.csv")
    res[res["nominal_kohm"] != 100].to_csv(tmp_path / "b" / "resistors.csv", index=False)
    with pytest.raises(ValueError, match="no multimeter value"):
        bench.main([str(tmp_path / "b"), "--tables", str(tmp_path / "t")])
    assert bench.main([str(tmp_path / "b"), "--tables", str(tmp_path / "t"),
                       "--allow-nominal"]) == 0
    cap = pd.read_csv(tmp_path / "t" / "emg_impedance_captures.csv")
    assert set(cap.loc[cap["nominal_kohm"] == 100, "r_source"]) == {"nominal"}


def test_per_channel_resistor_row_wins(bench):
    res = pd.DataFrame({"nominal_kohm": [22.0, 22.0], "channel": [0, 3],
                        "multimeter_kohm": [22.1, 21.9]})
    assert bench.true_value(res, 22.0, 3) == 21.9
    assert bench.true_value(res, 22.0, 1) == 22.1
    assert np.isnan(bench.true_value(res, 47.0, 1))


def test_bench_numbers_source(monkeypatch, tmp_path: Path):
    from biomech.viz import numbers as N

    pd.DataFrame({"current_na": [6, 6], "channel": ["1", "all"], "k": [1.02, 1.01],
                  "max_abs_err_pct": [3.0, 4.5]}).to_csv(tmp_path / "emg_impedance_channels.csv",
                                                         index=False)
    monkeypatch.setattr(N, "BENCH_TABLES", tmp_path)
    out = N._resolve_rows(None, {"benchImpK": {
        "source": "bench:emg_impedance_channels", "filter": {"channel": "all"},
        "column": "k", "fmt": ".2f", "unit": ""}})
    assert out[0].value == pytest.approx(1.01) and out[0].text == "1.01"
    assert N.bench_table("absent").empty


def test_blank_template_cells_are_missing_values(bench, tmp_path: Path):
    """The template ships with empty multimeter cells: they count as not measured."""
    (tmp_path / "resistors.csv").write_text(
        "nominal_kohm,channel,multimeter_kohm\n10,all,9.97\n22,all,\n", encoding="utf-8")
    res = bench.read_resistors(tmp_path / "resistors.csv")
    assert bench.true_value(res, 10.0, 4) == 9.97 and np.isnan(bench.true_value(res, 22.0, 4))


def test_store_sensitivity_counts_with_scale_and_curve(bench, tmp_path: Path, monkeypatch):
    """Stored |Z| moved by the bench error: per channel scale k and calibration curve."""
    import biomech.results as R
    from biomech.io.session import MUSCLES

    _write(tmp_path / "b", {c: (0.9 if c == 1 else 1.0) for c in range(1, 7)})
    assert bench.main([str(tmp_path / "b"), "--tables", str(tmp_path / "t")]) == 0
    cap = pd.read_csv(tmp_path / "t" / "emg_impedance_captures.csv")
    tab = pd.read_csv(tmp_path / "t" / "emg_impedance_channels.csv", dtype={"channel": str})
    chk = pd.DataFrame({"take": ["motion"] * 3, "muscle": [MUSCLES[0], MUSCLES[0], MUSCLES[1]],
                        "z_kohm": [46.0, 20.0, 49.0]})

    class Store:
        def __init__(self, root):
            pass

        def read_table(self, stage, table):
            return chk

    monkeypatch.setattr(R, "ResultStore", Store)
    s = bench.store_sensitivity(tmp_path, cap, tab).iloc[0]
    # CH1 reads 10 % low (46 -> ~51 true); CH2 is exact (49 stays below the limit)
    assert s["above_recorded_n"] == 0 and s["above_scale_n"] == 1 and s["above_curve_n"] == 1
    assert s["n"] == 3 and s["median_recorded_kohm"] == 46.0
