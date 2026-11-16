# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.3: the twelve EMG gallery views of ``viz/views/emg_r8.py``.

Every view and form is built on a small synthetic store held in memory (a stand-in for the
DataContext surface the views read: tables, arrays, config, pseudonyms; no recorded session,
so the signal views fall back to the stored chain example), and on an empty store. Checks:
registration (group, kind, forms), a FigureSpec with panels, the structured caption passing
the caption rubric, no plan codes / raw identifiers / raw values in the reader text, and a
clean matplotlib layout. Fast: a few seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.check.text import caption_rubric_issues, text_issues
from biomech.viz import render_mpl
from biomech.viz.spec import FigureSpec
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V
from biomech.viz.views.emg_r8 import CYCLE_TASKS, PHASE_BINS

VIEWS = ("emg_signal_spectrum", "emg_cycle_spectrogram", "emg_onset_raster",
         "emg_motion_coupling", "emg_activation_clock", "emg_surface", "emg_on_path",
         "emg_story", "emg_feature_embedding", "emg_ridgelines", "emg_usability_flow",
         "kin_phase_portrait")
PIDS = ["s01", "s02", "s03", "s04"]
NAMES = np.array(V.MUSCLES)                                     # channel order of the arrays
TASK_S = {"rest": 6.0, "open_close": 16.0, "chewing": 10.0, "protrusion_retrusion": 10.0,
          "left_right": 10.0, "free_motion": 10.0}
PERIOD = {"open_close": 2.0, "chewing": 0.8}
T0 = {t: 1_000_000_000 + i * 60_000_000 for i, t in enumerate(TASK_S)}


class _Config:
    seed = 7
    signals: dict = {}

    def params(self, stage):
        return {}


class _Store:
    def stages(self):
        return []


class Ctx:
    """Tables by (stage, name) and arrays by (stage, unit), read like DataContext."""

    def __init__(self, tables=None, arrays=None):
        self.tables = tables or {}
        self._arrays = arrays or {}
        self.config = _Config()
        self.store = _Store()

    def table(self, stage, name, pids=None):
        if (stage, name) not in self.tables:
            raise KeyError(name)
        d = self.tables[(stage, name)].copy()
        if pids and "pid" in d:
            d = d[d["pid"].isin(pids)]
        return d

    def arrays(self, stage, unit):
        return self._arrays.get((stage, unit), {})

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1

    def onset_rule(self):
        return "drive"


def _cycle_starts(task: str) -> np.ndarray:
    per = PERIOD.get(task, 2.0)
    return T0[task] + (np.arange(0.5, TASK_S[task] - per, per) * 1e6).astype(np.int64)


def _phase(t_us: np.ndarray, task: str) -> np.ndarray:
    per = PERIOD.get(task, 2.0)
    return ((t_us - T0[task]) / 1e6 - 0.5) / per % 1.0


def synthetic(seed: int = 3) -> Ctx:
    rng = np.random.default_rng(seed)
    tables: dict[tuple[str, str], list[dict]] = {k: [] for k in (
        ("kinematics", "cycles"), ("timing", "onsets"), ("timing", "coupling"),
        ("emg", "spectral"), ("emg", "activation"))}
    arrays: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    grid_hz = np.arange(25.0, 460.0, 10.0)
    freq = np.arange(10.0, 501.0, 2.0)
    for k, pid in enumerate(PIDS):
        e: dict[str, np.ndarray] = {"profile__names": NAMES, "spectral__freq": freq}
        kin: dict[str, np.ndarray] = {}
        imu: dict[str, np.ndarray] = {}
        lab: dict[str, np.ndarray] = {}
        for task, dur in TASK_S.items():
            t = T0[task] + np.arange(0, int(dur * 100)) * 10_000          # 100 Hz envelope grid
            ph = _phase(t, task) if task != "rest" else np.zeros(t.size)
            act_open = 5 + 30 * np.exp(-((ph - 0.3) / 0.12) ** 2) * (task != "rest")
            act_close = 5 + 20 * np.exp(-((ph - 0.75) / 0.12) ** 2) * (task != "rest")
            norm = np.column_stack([act_close if n in V.CLOSERS else act_open for n in NAMES])
            norm = norm * (1 + 0.1 * k) + rng.normal(0, 0.5, norm.shape)
            e[f"{task}__t_us"] = t
            e[f"{task}__norm_pct"] = norm.astype(np.float32)
            e[f"{task}__env_uv"] = (3 * norm).astype(np.float32)
            e[f"{task}__masked"] = np.zeros(norm.shape, np.uint8)
            P = (1.0 / (1 + (freq / 80.0) ** 2))[None, :] * (1 + 0.2 * k) * np.ones((6, 1))
            e[f"spectral__{task}__psd"] = P.astype(np.float32)
            for n in NAMES:
                tables[("emg", "spectral")].append({"pid": pid, "task": task, "muscle": n,
                                                    "mnf_hz": 110.0, "mdf_hz": 90.0,
                                                    "usable_shape": True})
                if task == "rest":
                    tables[("emg", "activation")].append({"pid": pid, "task": "rest",
                                                          "muscle": n, "p95_pct": 8.0,
                                                          "median_pct": 5.0})
            # kinematics (40 Hz), IMU (200 Hz), labels (40 Hz)
            tk = T0[task] + np.arange(0, int(dur * 40)) * 25_000
            phk = _phase(tk, task) if task != "rest" else np.zeros(tk.size)
            th = 12 * np.sin(np.pi * phk) ** 2
            kin[f"{task}__t_us"] = tk
            kin[f"{task}__theta_deg"] = th
            kin[f"{task}__theta_vel"] = np.gradient(th, tk / 1e6)
            kin[f"{task}__theta_acc"] = np.gradient(kin[f"{task}__theta_vel"], tk / 1e6)
            kin[f"{task}__incisor"] = np.column_stack([-0.3 * th, 2 * np.sin(2 * np.pi * phk),
                                                       -th - 10])
            ti = T0[task] + np.arange(0, int(dur * 200)) * 5_000
            thi = 12 * np.sin(np.pi * (_phase(ti, task) if task != "rest" else 0 * ti)) ** 2
            imu[f"{task}__t_us"] = ti
            imu[f"{task}__theta_quat"] = thi + rng.normal(0, 0.05, ti.size)
            imu[f"{task}__theta_opt"] = thi
            lab[f"{task}__t_us"] = tk
            lab[f"{task}__label"] = np.where(phk < 0.5, 1, 2).astype(np.int8)
            # stored chain example: the whole phase at 1 kHz
            if task in ("rest",) + CYCLE_TASKS:
                tc = T0[task] + np.arange(0, int(dur * 1000)) * 1000
                phc = _phase(tc, task) if task != "rest" else np.zeros(tc.size)
                amp = np.column_stack([
                    5 + 40 * np.exp(-((phc - (0.75 if n in V.CLOSERS else 0.3)) / 0.1) ** 2)
                    * (task != "rest") for n in NAMES])
                x = amp * rng.normal(0, 1, amp.shape)
                e[f"chain__{task}__t_us"] = tc
                e[f"chain__{task}__raw"] = (x + 20).astype(np.float32)
                e[f"chain__{task}__cleaned"] = x.astype(np.float32)
                e[f"chain__{task}__norm"] = amp.astype(np.float32)
                e[f"chain__{task}__masked"] = np.zeros(amp.shape, np.uint8)
            if task in CYCLE_TASKS:
                st = _cycle_starts(task)
                per = PERIOD[task]
                u = np.linspace(0, 1, 101)
                prof = np.column_stack([(20 if n in V.CLOSERS else 30)
                                        * np.exp(-((u - (0.75 if n in V.CLOSERS else 0.3))
                                                   / 0.12) ** 2) + 5 for n in NAMES])
                e[f"profile__{task}__pct"] = np.stack([prof * (1 + 0.05 * j)
                                                       for j in range(len(st))]
                                                      ).astype(np.float32)
                e[f"profile__{task}__theta_deg"] = np.tile(
                    12 * np.sin(np.pi * np.linspace(0, 1, 101)) ** 2, (len(st), 1))
                S = int((dur - 0.5) / 0.25)
                e[f"spectral__{task}__win"] = (np.log10(
                    np.tile((1.0 / (1 + (grid_hz / 80.0) ** 2)) / 3.0, (6, S, 1)))
                    + rng.normal(0, 0.02, (6, S, grid_hz.size))).astype(np.float32)
                for c, s0 in enumerate(st):
                    t_mech = float(s0 + 0.05e6)
                    tables[("kinematics", "cycles")].append({
                        "pid": pid, "task": task, "cycle": c, "ok": True,
                        "t_start_us": int(s0), "t_end_us": int(s0 + per * 1e6)})
                    for direction, grp in (("opening", V.OPENERS), ("closing", V.CLOSERS)):
                        tm = t_mech if direction == "opening" else t_mech + 0.5 * per * 1e6
                        for m in grp:
                            for rule in ("first", "drive", "drive_taskbase"):
                                lead = {"first": 150.0, "drive": 40.0}.get(rule, 30.0) \
                                    + rng.normal(0, 10)
                                use = bool((c + k) % 3)
                                tables[("timing", "onsets")].append({
                                    "pid": pid, "task": task, "cycle": c, "side": "",
                                    "direction": direction, "muscle": m, "detector": "tkeo",
                                    "rule": rule, "t_mech_us": tm, "t_omega_us": tm + 5e3,
                                    "t_onset_us": tm - lead * 1e3 if use or c % 2 else np.nan,
                                    "usable": use, "burst_db": 10.0 if use else 3.0,
                                    "active_at_start": c % 5 == 4})
                tables[("timing", "coupling")].append({"pid": pid, "task": task,
                                                       "emd_lag_ms": 12.0 + k,
                                                       "emd_r": 0.15, "emd_r_null_median": 0.05,
                                                       "p_coupling": 0.01})
        arrays[("emg", pid)] = e
        arrays[("kinematics", pid)] = kin
        arrays[("imu", pid)] = imu
        arrays[("labels", pid)] = lab
        n_w = 400
        y = rng.integers(0, 7, n_w)
        X = rng.normal(0, 1, (n_w, 24)) + y[:, None] * 0.5 + k
        arrays[("ml_windows", pid)] = {"X_emg": X.astype(np.float32), "y": y.astype(np.int8)}
    on = {}
    for pid in PIDS:
        t = T0["chewing"] + np.arange(0, 10_000_000, 50_000)
        on[f"online__{pid}__t_us"] = t
        on[f"online__{pid}__yhat"] = np.where(_phase(t, "chewing") < 0.5, 1, 2).astype(np.int8)
        on[f"online__{pid}__release_us"] = _cycle_starts("chewing")[1:4]
    arrays[("ml_online", "_cohort")] = on
    arrays[("stats_emg", "_cohort")] = {"spectral__grid_hz": grid_hz}
    t = {key: pd.DataFrame(v) for key, v in tables.items()}
    t[("stats_emg", "spm")] = pd.DataFrame([{"family": "confirmatory",
                                             "test": "openers_vs_closers_open_close",
                                             "clusters": "[(0, 11, 0.003), (59, 100, 0.0)]"}])
    return Ctx(t, arrays)


@pytest.fixture(scope="module")
def ctx() -> Ctx:
    return synthetic()


def _cases() -> list[tuple[str, dict]]:
    out = []
    for vid in VIEWS:
        v = R.get(vid)
        for form in v.choice_values("form") or [None]:
            kw = {"form": form} if form else {}
            out.append((vid, kw))
        for ph in v.phases[1:]:
            out.append((vid, {"phase": ph}))
    return out


def test_registered_in_the_dataset_group():
    for vid in VIEWS:
        v = R.get(vid)
        assert v.group == "dataset" and v.kind == "dashboard", vid
        assert v.domain in R.DOMAINS, vid
        forms = v.choice_values("form")
        if forms:
            assert v.defaults["form"] in forms, vid
    assert len([v for v in R.views() if v.builder.__module__.endswith(".emg_r8")]) == len(VIEWS)


@pytest.mark.parametrize("vid,kw", _cases())
def test_view_builds_with_a_clean_caption(ctx, vid, kw):
    params = dict(kw)
    if "pid" in R.get(vid).params:
        params["pid"] = "s02"
    spec = R.build(vid, R.request(ctx, vid, **params))
    assert isinstance(spec, FigureSpec) and spec.panels, vid
    assert "nothing to show" not in spec.caption, (vid, kw)     # the data path, not the card
    shown = V.reader_spec(spec)
    assert caption_rubric_issues(shown) == [], (vid, kw, shown.caption)
    assert text_issues(shown) == [], (vid, kw)
    assert spec.meta.get("caption"), vid
    b = render_mpl.build(spec)
    try:
        assert render_mpl.text_overlaps(b) == [], (vid, kw)
    finally:
        import matplotlib.pyplot as plt

        plt.close(b.fig)


def test_views_build_on_an_empty_store():
    empty = Ctx()
    for vid in VIEWS:
        spec = R.build(vid, R.request(empty, vid))
        assert spec.caption and spec.meta.get("caption"), vid


def test_cycle_spectrogram_uses_the_configured_bins(ctx):
    spec = R.build("emg_cycle_spectrogram", R.request(ctx, "emg_cycle_spectrogram"))
    img = [s for s in spec.panels[0].series if s.kind == "image"][0].img
    assert img.shape[1] == PHASE_BINS
    assert np.isfinite(img).any()


def test_usability_flow_counts_add_up(ctx):
    spec = R.build("emg_usability_flow", R.request(ctx, "emg_usability_flow"))
    on = ctx.tables[("timing", "onsets")]
    n_open = len(on[(on.direction == "opening") & (on.rule == "drive")])
    assert sum(spec.meta["counts"]["opening"].values()) == n_open
    use = int(on[(on.direction == "opening") & (on.rule == "drive")].usable.sum())
    assert spec.meta["counts"]["opening"]["usable"] == use


def test_onset_qc_sample_is_seeded(ctx):
    a = R.build("emg_signal_spectrum", R.request(ctx, "emg_signal_spectrum", pid="s01",
                                                 form="onset_qc"))
    b = R.build("emg_signal_spectrum", R.request(ctx, "emg_signal_spectrum", pid="s01",
                                                 form="onset_qc"))
    assert a.meta["sample"] == b.meta["sample"] and a.meta["sample"]
