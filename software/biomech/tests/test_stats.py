# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Statistics helpers on synthetic data with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomech.stats.core import bland_altman, describe, hedges_g, holm, icc_sem_mdc
from biomech.stats.waveforms import anova1_participants, cmc, cycles_needed, paired_t, \
    variance_ratio


def test_describe_and_ci_cover_mean():
    rng = np.random.default_rng(0)
    d = describe(rng.normal(10, 2, 40))
    assert d["n"] == 40 and d["ci_lo"] < 10 < d["ci_hi"] and abs(d["sd"] - 2) < 0.6


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_hedges_g_sign():
    assert hedges_g(np.arange(10) + 5.0, np.arange(10) * 1.0)["g"] > 1


def test_icc_high_for_stable_subjects():
    rng = np.random.default_rng(1)
    rows = [{"pid": f"s{i}", "rep": r, "v": 10 * i + rng.normal(0, 1)}
            for i in range(8) for r in range(6)]
    out = icc_sem_mdc(pd.DataFrame(rows), "v")
    assert out["icc31"] > 0.95 and out["mdc95"] > 0


def test_bland_altman():
    out = bland_altman(np.array([1.0, 2, 3, 4]) + 0.5, np.array([1.0, 2, 3, 4]))
    assert abs(out["bias"] - 0.5) < 1e-12 and out["loa_lo"] == out["loa_hi"]


def test_cmc_and_variance_ratio():
    x = np.linspace(0, 2 * np.pi, 101)
    rng = np.random.default_rng(2)
    same = np.sin(x) + rng.normal(0, 0.01, (10, 101))
    assert cmc(same) > 0.99
    prof = {f"s{i}": i * np.sin(x) + rng.normal(0, 0.05, (8, 101)) for i in range(1, 6)}
    vr = variance_ratio(prof)
    assert vr["ratio"][25] > 10


def test_spm_detects_offset_and_anova():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 101)
    A = np.sin(2 * np.pi * x) + rng.normal(0, 0.2, (12, 101))
    B = A + 0.6 * np.exp(-((x - 0.5) / 0.1) ** 2) + rng.normal(0, 0.1, (12, 101))
    r = paired_t(A, B)
    assert r["h0reject"] and any(c["start_pct"] < 55 < c["end_pct"] for c in r["clusters"])
    prof = {f"s{i}": A[:6] + i * 0.5 for i in range(4)}
    assert anova1_participants(prof)["h0reject"]


def test_cycles_needed_monotone():
    ns, hw, n = cycles_needed(np.random.default_rng(4).normal(0, 1, 20), 0.6)
    assert hw[2] > hw[-1] and n is not None


def test_norms_compare_prefers_sex_specific_and_z_only_with_sd():
    from biomech.norms.literature import compare

    norms = {"m1": [{"sex": "all", "mean": 10.0, "sd": None, "source": "a"},
                    {"sex": "f", "mean": 8.0, "sd": 2.0, "source": "b"}]}
    subj = pd.DataFrame({"pid": ["s1", "s2"], "sex": ["f", "m"], "m1": [10.0, 12.0]})
    out = compare(subj, norms).set_index("pid")
    assert out.loc["s1", "z"] == 1.0 and out.loc["s1", "source"] == "b"
    assert np.isnan(out.loc["s2", "z"]) and out.loc["s2", "diff"] == 2.0
