# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pins the helpers that replaced ``experiments/_lib`` (W3) on synthetic data."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.signal import find_peaks

from biomech import paths
from biomech.io.inventory import inventory, scan, select_takes, subject_pids
from biomech.kinematics.peak_cycles import nan_fraction, peak_to_peak_cycles
from biomech.kinematics.traces import (
    load_jts_trace,
    load_kinematic_trace,
    load_optical_trace,
    opening_1d,
    sample_rate_hz,
)
from biomech.results.provenance import write_csv, write_json, write_provenance
from test_w3_synth import make_take

MOTION = "session_s01_f_31_20250310T091000Z_cd0000"
MVC = "session_s01_f_31_mvc_20250310T090000Z_ab0000"


# -- paths ------------------------------------------------------------------------------


def test_resolve_data_dir_precedence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pilot = repo / "data" / "pilot_study" / "jtsmm_data"
    pilot.mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(paths, "repo_root", lambda: repo)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("JTSMM_DATA_DIR", str(tmp_path / "env"))

    assert paths.resolve_data_dir(tmp_path / "cli") == tmp_path / "cli"
    assert paths.resolve_data_dir("~/x") == home / "x"
    assert paths.resolve_data_dir(None) == tmp_path / "env"
    monkeypatch.delenv("JTSMM_DATA_DIR")
    assert paths.resolve_data_dir() == pilot
    shutil.rmtree(pilot)
    assert paths.resolve_data_dir() == home / "jtsmm_data"
    monkeypatch.setattr(paths, "repo_root", lambda: None)
    assert paths.resolve_data_dir("") == home / "jtsmm_data"


def test_results_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("BIOMECH_RESULTS_DIR", raising=False)
    assert paths.results_root() == tmp_path / "home" / "experiments"
    assert paths.results_dir("5_4_1") == tmp_path / "home" / "experiments" / "5_4_1" / "results"
    monkeypatch.setenv("BIOMECH_RESULTS_DIR", str(tmp_path / "res"))
    assert paths.results_dir("5_4_1") == tmp_path / "res" / "5_4_1" / "results"
    assert paths.results_dir("5_4_1", tmp_path / "o") == tmp_path / "o"


def test_repo_root_from_checkout_and_from_site_packages(tmp_path):
    here = Path(__file__).resolve()
    assert paths.repo_root() == here.parents[3]
    # The same module installed outside a checkout knows no repo.
    site = tmp_path / "site-packages" / "biomech"
    site.mkdir(parents=True)
    shutil.copy(paths.__file__, site / "paths.py")
    spec = importlib.util.spec_from_file_location("w3_paths_copy", site / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.repo_root() is None


# -- inventory --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def takes(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("w3_inv")
    make_take(root, MOTION, kind="motion", seed=1)
    make_take(root, MVC, kind="mvc", seed=2)
    return root


def test_inventory_flags_and_counts(takes):
    inv = inventory(takes / MOTION)
    assert (inv.pid, inv.subject_id, inv.sex, inv.age) == ("s01", "s01_f_31", "f", 31)
    assert inv.take_kind == "motion" and inv.phase_kind == "motion" and inv.complete
    assert inv.has_emg and inv.has_optical and inv.has_cameras and inv.has_sync
    assert inv.has_imu_mta and inv.has_imu_cra and not inv.has_jts_config
    assert inv.n_emg_samples > 0 and inv.n_optical_frames > 0
    assert set(inv.n_cam_frames) and all(n > 0 for n in inv.n_cam_frames.values())
    assert inv.findings == []
    assert inventory(takes / MOTION / "session.h5").name == MOTION
    mvc = inventory(takes / MVC)
    assert (mvc.take_kind, mvc.phase_kind) == ("mvc", "mvc")
    row = inv.to_row()
    assert row["h5"].endswith("session.h5") and json.dumps(row)


def test_inventory_reports_kind_mismatch_and_missing_phases(tmp_path):
    wrong = make_take(tmp_path, "session_s02_m_30_mvc_20250311T090000Z_ee0000", kind="motion")
    inv = inventory(wrong)
    assert inv.take_kind == "mvc"                         # the folder name decides ...
    assert inv.phase_kind == "motion"                     # ... the phases only check it
    assert any(f.startswith("kind_mismatch") for f in inv.findings)
    with h5py.File(wrong / "session.h5", "r+") as f:
        del f["phases"]
    inv = inventory(wrong)
    assert "no_phases" in inv.findings and not inv.complete


def test_scan_matches_pids_exactly(tmp_path):
    for name in ("session_s1_f_20_20250310T090000Z_aa", "session_s10_m_40_20250310T090000Z_bb"):
        make_take(tmp_path, name, kind="motion")
    assert [t.pid for t in scan(tmp_path, pids=["s1"])] == ["s1"]
    assert [t.pid for t in scan(tmp_path, pids=subject_pids("s10_m_40"))] == ["s10"]
    assert scan(tmp_path, kind="mvc") == []
    assert subject_pids("s03_m_33, S1") == ["s03", "s1"]
    assert subject_pids(None) is None


def test_select_takes_first_and_latest():
    from biomech.cohort import SessionInfo, parse_session_name

    names = ["session_s01_f_31_20250310T090000Z_aa", "session_s01_f_31_20250312T090000Z_bb",
             "session_s01_f_31_mvc_20250311T090000Z_cc"]
    infos = [SessionInfo(name=n, path=Path(n), **parse_session_name(n)) for n in names]
    assert [t.name for t in select_takes(infos, "first")] == [names[0], names[2]]
    assert [t.name for t in select_takes(infos, "latest")] == [names[1], names[2]]
    assert len(select_takes(infos, "all")) == 3


# -- provenance -------------------------------------------------------------------------


def test_write_provenance_records_git_and_versions(tmp_path, takes):
    inv = inventory(takes / MOTION)
    dest = write_provenance(tmp_path, "5_x_test", "script.py", [inv, takes / MVC],
                            data_dir=takes, extra={"nan": float("nan"), "arr": np.arange(2)})
    doc = json.loads(dest.read_text())
    assert dest.name == "provenance.json"
    assert doc["study"] == "5_x_test" and doc["script"] == "script.py"
    assert doc["n_sessions"] == 2 and doc["sessions"][0]["pid"] == "s01"
    assert doc["sessions"][1] == {"name": MVC, "path": str(takes / MVC)}
    assert {"sha", "dirty"} <= set(doc["git"])          # + dirty_paths, other_changes (J14)
    sha = doc["git"]["sha"]
    assert sha is None or (len(sha) == 40 and int(sha, 16) >= 0)
    assert doc["versions"]["biomech"] and "numpy" in doc["versions"]
    assert doc["extra"] == {"nan": None, "arr": [0, 1]}
    assert doc["data_dir"] == str(takes)


def test_write_json_and_csv(tmp_path):
    p = write_json(tmp_path / "a" / "x.json", {"v": np.float64("nan"), "i": np.int64(3),
                                               "p": Path("/x")})
    assert json.loads(p.read_text()) == {"v": None, "i": 3, "p": "/x"}
    c = write_csv(tmp_path / "x.csv", [{"a": 1}, {"a": 2, "b": 3}])
    assert c.read_text().splitlines() == ["a,b", "1,", "2,3"]
    assert write_csv(tmp_path / "e.csv", []).read_text() == ""


# -- peak-to-peak cycles ----------------------------------------------------------------


def _legacy_find_cycles(y, fs_hz, min_period_s=0.4, *, t_s=None, prominence_frac=0.25,
                        fill_frac=0.12):
    """Verbatim copy of the deleted ``experiments/_lib/session_study.find_cycles``."""
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < 8 or fs_hz <= 0:
        return []
    if t_s is None:
        tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    else:
        tt = np.asarray(t_s, dtype=np.float64).ravel()
        if tt.size != y.size:
            tt = np.arange(y.size, dtype=np.float64) / float(fs_hz)
    distance = max(int(min_period_s * fs_hz), 2)
    y_std = float(np.nanstd(y))
    prominence = max(y_std * float(prominence_frac), 1e-6)
    peaks, _ = find_peaks(y, distance=distance, prominence=prominence)
    peaks = [int(p) for p in peaks]
    if len(peaks) >= 3:
        med = float(np.median(np.diff(tt[peaks])))
        max_gap = max(2.2 * med, 2.0 * min_period_s)
    else:
        max_gap = max(2.5, 2.0 * min_period_s)
    extra = []
    fill_prom = max(y_std * float(fill_frac), 1e-6)
    for a, b in zip(peaks[:-1], peaks[1:]):
        if tt[b] - tt[a] <= max_gap:
            continue
        sl = slice(a + 1, b)
        if int(np.isfinite(y[sl]).sum()) < 8:
            continue
        loc, props = find_peaks(y[sl], distance=distance, prominence=fill_prom)
        if loc.size == 0:
            continue
        cand = []
        prominences = np.asarray(props.get("prominences", []), dtype=float)
        mid = 0.5 * (tt[a] + tt[b])
        for j, rel in enumerate(loc):
            k = int(a + 1 + rel)
            if tt[k] - tt[a] < min_period_s or tt[b] - tt[k] < min_period_s:
                continue
            pr = float(prominences[j]) if j < prominences.size else 0.0
            cand.append((abs(tt[k] - mid), -pr, k))
        if cand:
            extra.append(min(cand)[2])
    if extra:
        peaks = sorted(set(peaks).union(extra))
    if len(peaks) < 2:
        return []
    return [(int(a), int(b)) for a, b in zip(peaks[:-1], peaks[1:]) if b - a >= distance]


def _reps(fs=40.0, n_rep=10, quiet=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_rep * fs)) / fs
    amp = np.ones_like(t) * 10.0
    if quiet is not None:                       # two weak repetitions between strong ones
        amp[(t >= quiet) & (t < quiet + 2.0)] = 0.6
    return t, amp * 0.5 * (1 - np.cos(2 * np.pi * t)) + rng.normal(0, 0.05, t.size)


@pytest.mark.parametrize("quiet", [None, 4.0])
def test_peak_to_peak_matches_legacy_on_finite_traces(quiet):
    t, y = _reps(quiet=quiet)
    got = peak_to_peak_cycles(y, 40.0, t_s=t)
    assert got == _legacy_find_cycles(y, 40.0, t_s=t)
    # 10 peaks -> 9 cycles; with the weak pair, 8 strong peaks + 1 filled -> 8 cycles
    assert len(got) == (9 if quiet is None else 8)
    if quiet is None:
        assert all(0.9 < t[b] - t[a] < 1.1 for a, b in got)


def test_peak_to_peak_fills_long_gaps_with_the_quiet_repetition():
    t, y = _reps(quiet=4.0)
    peaks = {a for a, _ in peak_to_peak_cycles(y, 40.0, t_s=t)}
    first_pass, _ = find_peaks(y, distance=16, prominence=0.25 * np.std(y))
    assert not any(4.0 <= t[p] < 6.0 for p in first_pass)
    assert sum(4.0 <= t[p] < 6.0 for p in peaks) == 1   # the second pass adds one


def test_peak_to_peak_is_nan_safe():
    t, y = _reps()
    clean = peak_to_peak_cycles(y, 40.0, t_s=t)
    y_gap = y.copy()
    gap = (t > 5.3) & (t < 5.7)                  # covers the peak at t = 5.5 s
    y_gap[gap] = np.nan
    got = peak_to_peak_cycles(y_gap, 40.0, t_s=t)
    assert got and all(np.isfinite(y_gap[a]) and np.isfinite(y_gap[b]) for a, b in got)
    assert not any(gap[a] or gap[b] for a, b in got)
    assert [c for c in clean if t[c[1]] < 5.0] == [c for c in got if t[c[1]] < 5.0]
    assert any(nan_fraction(y_gap, a, b) > 0 for a, b in got)
    assert peak_to_peak_cycles(np.full(50, np.nan), 40.0) == []
    assert peak_to_peak_cycles(y[:5], 40.0) == []


# -- traces -----------------------------------------------------------------------------


def test_opening_1d():
    xyz = np.array([[0, 0, 0], [3, 4, 0], [np.nan, 0, 0], [1, 1, 1]], dtype=float)
    rest = np.array([[0, 0, 0], [0, 0, 0], [np.nan, 1, 1]])
    out = opening_1d(xyz, rest)
    assert out[0] == 0 and out[1] == 5 and np.isnan(out[2])
    assert np.isclose(out[3], np.sqrt(3))
    # without rest, the origin is the mean of the finite samples
    assert np.allclose(opening_1d(xyz)[[0, 1]],
                       np.linalg.norm(xyz[[0, 1]] - xyz[[0, 1, 3]].mean(0), axis=1))


def test_load_optical_trace_is_mandible_in_cranial_frame(takes):
    t, xyz = load_optical_trace(takes / MOTION)
    with h5py.File(takes / MOTION / "session.h5", "r") as f:
        mta = f["optical/pose_mta"][()].astype(float)
        cra = f["optical/pose_cra"][()].astype(float)
        t_ref = f["optical/t_master_us"][()]
    ref = np.einsum("nji,nj->ni", cra[:, :3, :3], mta[:, :3, 3] - cra[:, :3, 3])
    assert t.dtype == np.int64 and np.array_equal(t, t_ref.astype(np.int64))
    assert np.allclose(xyz, ref, equal_nan=True)
    assert np.isnan(xyz).any()
    assert sample_rate_hz(t) == pytest.approx(40.0, rel=0.05)
    assert sample_rate_hz(np.array([5])) == 60.0


def test_load_jts_trace_and_preference(takes, tmp_path):
    jts = tmp_path / "jaw_motion.h5"
    with h5py.File(jts, "w") as f:
        g = f.create_group("T_model_origin_mand_landmark_t")
        g.create_dataset("translations", data=np.ones((10, 3)) * 0.01)
        g.attrs["unit"] = "m"
        g.attrs["sample_rate"] = 20.0
    t, xyz = load_jts_trace(jts)
    assert np.allclose(xyz, 10.0) and t[1] == 50_000
    assert load_jts_trace(tmp_path / "missing.h5") is None
    t2, xyz2, src = load_kinematic_trace(takes / MOTION, jts)
    t_opt, _ = load_optical_trace(takes / MOTION)
    assert src == "jts" and np.array_equal(t2, t_opt) and xyz2.shape == (t_opt.size, 3)
    assert load_kinematic_trace(takes / MOTION)[2] == "optical_relative"


# -- dependency footprint ---------------------------------------------------------------

_BLOCK = "import sys\nfor m in ('pandas', 'pyarrow'):\n    sys.modules[m] = None\n"


def _import_without_pandas(modules: str) -> subprocess.CompletedProcess:
    code = _BLOCK + f"import {modules}\nprint('ok')\n"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[1], timeout=120)


def test_core_imports_without_pandas_and_pyarrow():
    r = _import_without_pandas(
        "biomech.cohort, biomech.config, biomech.io, biomech.emg.pipeline, "
        "biomech.kinematics.cycles, biomech.paths, biomech.io.inventory, "
        "biomech.kinematics.traces, biomech.kinematics.peak_cycles, biomech.emg.prepare")
    assert r.returncode == 0, r.stderr[-2000:]


def test_provenance_imports_without_pandas():
    r = _import_without_pandas("biomech.results.provenance")
    assert r.returncode == 0, r.stderr[-2000:]
