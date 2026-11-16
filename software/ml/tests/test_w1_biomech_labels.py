# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``ml labels from-biomech`` on a synthetic biomech store and cohort manifest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from ml.cli.main import main
from ml.config.classes import UNLABELLED
from ml.io.labels import read_labels
from tests import synth

pytest.importorskip("biomech.results.store")


def _setup(tmp_path):
    from biomech.results.store import ResultStore

    data = tmp_path / "data"
    s1 = synth.write_session(synth.folder(data, "s01"))
    m1 = synth.write_mvc(synth.folder(data, "s01", mvc=True))
    s1b = synth.write_session(synth.folder(data, "s01", stamp="20250306T102000Z"), seed=5)
    tracks, _ = synth.label_track()
    store = ResultStore(tmp_path / "out")
    arrays = {}
    for ph, (t, y) in tracks.items():
        y = y.astype(np.int8).copy()
        if ph == "left_right":
            y[:10] = -1                                       # untracked
        arrays[f"{ph}__t_us"], arrays[f"{ph}__label"] = t, y
    store.write("labels", "s01", "k1", tables={"counts": pd.DataFrame({"a": [1]})}, arrays=arrays)
    refs = pd.DataFrame({"pid": "s01", "muscle": list(synth.MUSCLE_CHANNEL_NAMES),
                         "p95_uv": np.arange(6) + 100.0})
    store.write("emg", "s01", "k2", tables={"refs": refs})
    cohort = tmp_path / "cohort.yaml"
    cohort.write_text(yaml.safe_dump({"study": "t", "participants": [
        {"pid": "s01", "sex": "f", "age": 31, "motion": s1.parent.name, "mvc": m1.parent.name,
         "include": True, "secondary": [s1b.parent.name]},
        {"pid": "s02", "sex": "f", "age": 36, "motion": "session_s02_f_36_20250302T101000Z_x",
         "mvc": None, "include": True}]}))
    return data, store, cohort, s1, m1


def test_from_biomech_writes_primary_take_sidecar(tmp_path):
    data, store, cohort, s1, m1 = _setup(tmp_path)
    out = tmp_path / "ml_labels"
    r = CliRunner().invoke(main, ["labels", "from-biomech", "--store", str(store.root),
                                  "--cohort", str(cohort), "--out", str(out), "--data-dir",
                                  str(data)])
    assert r.exit_code == 0, r.output
    assert "s02: skipped" in r.output
    files = sorted(out.glob("*_labels.h5"))
    assert [f.name for f in files] == [f"{s1.parent.name}_labels.h5"]   # primary take only
    lab = read_labels(files[0])
    assert lab.attrs["subject_id"] == "s01" and lab.attrs["session_id"] == s1.parent.name
    assert int(np.sum(lab.y == UNLABELLED)) == 10 and lab.y.max() == UNLABELLED
    tracks, opt = synth.label_track()
    assert np.array_equal(opt[lab.optical_frame], lab.t_master_us)
    assert lab.attrs["mvc_session"].endswith(f"{m1.parent.name}/session.h5")
    from ml.datasets.builder import build_dataset

    # The MVC take wins: its reference is computed on the windows' causal chain.
    d = build_dataset(s1, label_path=files[0])
    assert d.meta["ref_chain"] == "causal" and d.meta["ref_source"].endswith("session.h5")
    assert d.pid[0] == "s01"
    m1.unlink()                                              # no MVC take: the stored ref
    d = build_dataset(s1, label_path=files[0])
    assert d.meta["ref_uv"] == list(np.arange(6) + 100.0)
    assert "biomech emg/refs" in d.meta["ref_source"]
    assert d.meta["ref_chain"] == "zero-phase (biomech)"
