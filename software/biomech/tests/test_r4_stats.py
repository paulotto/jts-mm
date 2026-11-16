# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 statistics: E25 BCa fallback and non-converging mixed model, D15 C2 aggregation and
Holm families, confirmatory SPM decision on the Holm-adjusted p."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.stats import cohort_emg as ce
from biomech.stats import cohort_misc as cm
from biomech.stats.core import describe, holm


def test_e25_bca_nan_falls_back_to_percentile():
    v = np.array([1, 1, 1, 1, 1, 1, 1, 1, 5.0])    # jackknife medians all equal: BCa undefined
    d = describe(v, stat="median")
    assert np.isfinite(d["ci_lo"]) and np.isfinite(d["ci_hi"]) and d["ci_percentile"]
    d2 = describe(np.random.default_rng(0).normal(size=20), stat="mean")
    assert np.isfinite(d2["ci_lo"]) and not d2["ci_percentile"]
    assert describe(np.array([1.0, 2.0]))["ci_percentile"] is False


def _activation(n_pid=6):
    rows, refs = [], []
    for k in range(n_pid):
        for m in ("masseter_l", "digastric_l"):
            refs.append({"pid": f"p{k}", "muscle": m, "usable_amplitude": True})
            for task, lvl in (("rest", 2.0), ("open_close", 20.0), ("chewing", 40.0)):
                rows.append({"pid": f"p{k}", "task": task, "muscle": m,
                             "median_pct": lvl * (1 + 0.1 * k), "rest_rms_uv": 3.0})
    return pd.DataFrame(rows), pd.DataFrame(refs)


def test_e25_non_converging_mixed_model_keeps_other_tables(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("mixed model did not converge with any optimizer: test")

    monkeypatch.setattr(ce, "mixed_model", boom)
    act, refs = _activation()
    out = ce.activation_tables(act, refs, seed=0)
    assert not out["activation_cohort"].empty and not out["rest_activity"].empty
    lmm = out["activation_lmm"]
    assert len(lmm) == 1 and lmm.status.iloc[0] == "not converged"
    assert "term" not in lmm                       # views read "no model", not a broken table


def _onsets(seed=0):
    """Six participants; p5 has only 2 usable opener channel-cycles."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(6):
        for task in ("open_close", "chewing"):
            for rule in ("drive", "first"):
                for det in ("tkeo", "envelope"):
                    for m, n_cyc, base in (("digastric_l", 8, 40.0), ("digastric_r", 2, 120.0)):
                        for c in range(n_cyc):
                            usable = (k != 5) or (m == "digastric_l" and c < 2)
                            if k == 5 and m == "digastric_r":
                                usable = False
                            lead = base + 10 * k + rng.normal(0, 5)
                            rows.append({"pid": f"p{k}", "task": task, "direction": "opening",
                                         "muscle": m, "detector": det, "rule": rule,
                                         "cycle": c, "lead_ms": lead, "lead_omega_ms": lead,
                                         "lead_acc_ms": lead, "lead_opt_ms": lead,
                                         "causal_delay_ms": 20.0, "anticipation_ms": lead - 20,
                                         "usable": usable, "active_at_start": False})
    on = pd.DataFrame(rows)
    keys = ["pid", "task", "direction", "muscle", "detector", "rule"]
    u = on[on.usable]
    summ = (u.groupby(keys).agg(n=("lead_ms", "size"), lead_ms=("lead_ms", "median"),
                                lead_omega_ms=("lead_omega_ms", "median"),
                                lead_acc_ms=("lead_acc_ms", "median"),
                                lead_opt_ms=("lead_opt_ms", "median"),
                                causal_delay_ms=("causal_delay_ms", "median"),
                                anticipation_ms=("anticipation_ms", "median")).reset_index())
    return summ, on


def test_d15_pooled_vs_muscle_mean_and_min_cycles():
    summ, on = _onsets()
    per, n_ex = cm.per_participant_timing(summ, on, "open_close", "opening", "tkeo", "drive")
    assert n_ex == 1 and "p5" not in per.index and len(per) == 5
    u = on[on.usable & (on.task == "open_close") & (on.detector == "tkeo") & (on.rule == "drive")]
    assert per.loc["p0", "lead_ms"] == pytest.approx(u[u.pid == "p0"].lead_ms.median())
    old, n_ex_old = cm.per_participant_timing(summ, on, "open_close", "opening", "tkeo", "drive",
                                              aggregation="muscle_mean", min_cycles=1)
    assert n_ex_old == 0 and "p5" in old.index
    s0 = summ[(summ.pid == "p0") & (summ.task == "open_close") & (summ.detector == "tkeo")
              & (summ.rule == "drive")]
    assert old.loc["p0", "lead_ms"] == pytest.approx(s0.lead_ms.mean())
    # The pooled value weighs channels by their cycles (8 vs 2): closer to digastric_l.
    assert per.loc["p0", "lead_ms"] < old.loc["p0", "lead_ms"]
    with pytest.raises(ValueError):
        cm.per_participant_timing(summ, on, "open_close", "opening", "tkeo", "drive",
                                  aggregation="bogus")


def test_d15_timing_tables_families_and_holm():
    summ, on = _onsets()
    out = cm.timing_tables(summ, on, seed=0, params={"aggregation": "pooled", "min_cycles": 3},
                           onset_rule="drive")
    tc = out["timing_cohort"]
    conf = tc[tc.family == "confirmatory"]
    assert set(zip(conf.task, conf.metric, conf.detector, conf.rule)) == {
        (t, "lead_ms", "tkeo", r) for t in ("open_close", "chewing") for r in ("first", "drive")}
    np.testing.assert_allclose(conf.p_holm, holm(conf.p.tolist()))
    ex = tc[(tc.family == "exploratory") & (tc.metric == "lead_ms")]
    assert len(ex) == 4                            # 2 tasks x 2 detectors x 2 rules - 4
    np.testing.assert_allclose(ex.p_holm, holm(ex.p.tolist()))
    assert (tc[tc.metric == "lead_omega_ms"].family == "descriptive").all()
    assert tc.p_holm[tc.family == "descriptive"].isna().all()
    assert (tc.n_excluded_min_cycles == 1).all() and (tc.n == 5).all()
    old = cm.timing_tables(summ, on, seed=0, params={"aggregation": "muscle_mean",
                                                     "min_cycles": 1})
    assert (old["timing_cohort"].n == 6).all()


def test_d15_config_keys_and_sensitivity_variant():
    import yaml

    from biomech.config import DEFAULTS_PATH

    doc = yaml.safe_load(open(DEFAULTS_PATH, encoding="utf-8"))
    assert doc["stages"]["stats_timing"] == {"aggregation": "pooled", "min_cycles": 3,
                                             "confirmatory_rules": ["first", "drive"]}
    assert doc["sensitivity"]["c2_muscle_mean"]["set"] == {
        "stages.stats_timing.aggregation": "muscle_mean", "stages.stats_timing.min_cycles": 1}


def test_d15_confirmatory_spm_uses_holm_adjusted_p():
    rng = np.random.default_rng(0)
    profiles, ok = {}, {}
    for k in range(8):
        P = rng.normal(10, 1, (6, 101, 6))
        il, ir = ce.MUSCLES.index("masseter_l"), ce.MUSCLES.index("masseter_r")
        x = np.linspace(0, 1, 101)
        P[:, :, il] += 3.0 * np.exp(-((x - 0.5) / 0.1) ** 2)       # L > R mid-cycle
        P[:, :, ir] += 0.0
        profiles[f"p{k}"] = {"chewing__pct": P, "open_close__pct": P.copy(),
                             "names": np.array(ce.MUSCLES)}
        ok[f"p{k}"] = set(ce.MUSCLES)
    t, _ = ce.spm_tests(profiles, ok)
    conf = t[t.family == "confirmatory"]
    assert {"h0reject", "h0reject_raw", "p_holm"} <= set(conf.columns)
    for r in conf.itertuples():
        if r.h0reject_raw is None or (isinstance(r.h0reject_raw, float)
                                      and np.isnan(r.h0reject_raw)):
            continue
        assert bool(r.h0reject) == bool(r.p_holm < 0.05)
        assert not (r.h0reject and not r.h0reject_raw)              # Holm never adds rejections
    assert bool(conf[conf.test == "masseter_L_vs_R_chewing"].h0reject.iloc[0])
