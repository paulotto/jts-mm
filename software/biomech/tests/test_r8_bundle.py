# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J9.4: the per-snapshot bundle (lossless HDF5 with checksums and provenance), its
verification, the verifying reader and the snapshot archive, on a synthetic snapshot."""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest

from biomech.results import bundle as BD
from biomech.results.store import ResultStore


def _snapshot(root, name="r99"):
    """A snapshot directory as `biomech snapshot` writes it, covering every column encoding,
    a Parquet-fallback table, string / 0-d arrays, a failed unit and a comparison file."""
    snap = root / "snapshots" / name
    st = ResultStore(snap)
    t = pd.DataFrame({"x": [1.5, np.nan, -0.0], "n": [1, 2, 3], "ok": [True, False, True],
                      "pid": ["s01", "s02", "s03"], "note": ["a", None, ""],
                      "flag": [True, None, False], "empty": [None, None, None]})
    st.write("kin", "s01", "k1", tables={"t": t, "zero": t.iloc[:0]},
             arrays={"a": np.arange(6.0).reshape(2, 3), "lab": np.array(["open", "close"]),
                     "s": np.array(3.0), "i": np.array([1, 2], dtype=np.uint16)})
    st.write("kin", "s02", "k2", status="failed", error="no data")
    ts = pd.DataFrame({"when": pd.to_datetime(["2026-11-02", "2026-11-03"]), "v": [1, 2]})
    st.write("stats", "_cohort", "k3", tables={"times": ts})
    (snap / "cohort.yaml").write_text("participants: [s01, s02]\n", encoding="utf-8")
    (snap / "sensitivity" / "reporting_population").mkdir(parents=True)
    pd.DataFrame({"key": ["a"], "delta": [0.1]}).to_parquet(
        snap / "sensitivity" / "reporting_population" / "comparison.parquet", index=False)
    info = {"name": name, "study": "demo", "source_out_dir": str(root), "runs": ["run1"],
            "git": {"sha": "abc", "dirty": False}, "code_hash": "h", "config": {"a": 1},
            "status": [{"stage": "kin", "unit": "s01", "status": "up-to-date"},
                       {"stage": "kin", "unit": "s02", "status": "failed"},
                       {"stage": "stats", "unit": "_cohort", "status": "stale"}],
            "stale_units": 2, "created_utc": "2026-11-02T00:00:00Z", "note": ""}
    (snap / "snapshot.json").write_text(json.dumps(info), encoding="utf-8")
    return snap


def test_round_trip_equals_the_store(tmp_path):
    snap = _snapshot(tmp_path)
    p = BD.write(snap, pinned_by=["experiments/demo/paper_demo.yaml"])
    assert p == snap / "bundle_r99.h5" and BD.sidecar(p).is_file()
    assert not p.with_name(p.name + ".tmp").exists()
    assert BD.verify(p, against=snap) == []
    st = ResultStore(snap)
    with BD.open_bundle(p) as b:
        for stage, name in (("kin", "t"), ("kin", "zero"), ("stats", "times")):
            pd.testing.assert_frame_equal(b.read_table(stage, name),
                                          st.read_table(stage, name), check_exact=True)
        got, want = b.read_arrays("kin", "s01"), st.read_arrays("kin", "s01")
        assert set(got) == set(want)
        for k in want:
            assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k])
        assert b.meta("kin", "s02")["error"] == "no data"
        assert b.comparison("reporting_population").delta.tolist() == [0.1]
        m = b.manifest
        units = m[m.kind == "unit"].set_index("unit")
        assert units.loc["s02", "status"] == "failed" and bool(units.loc["s02", "stale"])
        assert not bool(units.loc["s01", "stale"])
        enc = m[m.kind == "table"].set_index("table").encoding
        assert enc["t"] == "columns" and enc["times"] == "parquet"
        cols = m[(m.kind == "column") & (m.table == "t")].set_index("column").encoding
        assert cols.to_dict() == {"x": "native", "n": "native", "ok": "native", "pid": "str",
                                  "note": "str", "flag": "bool_none", "empty": "none"}
        assert b.provenance["snapshot"] == "r99" and b.provenance["runs"] == ["run1"]


def test_flipped_byte_and_sidecar_fail(tmp_path):
    snap = _snapshot(tmp_path)
    p = BD.write(snap, pinned_by=["x"])
    with h5py.File(p, "r+") as h5:
        ds = h5["/cache/kin/s01/tables/t/n"]
        v = ds[()]
        v[0] += 1
        ds[...] = v
    probs = BD.verify(p)
    assert any("sidecar" in x for x in probs)
    assert any("content hash differs: /cache/kin/s01/tables/t/n" in x for x in probs)
    with pytest.raises(BD.BundleError):
        BD.open_bundle(p)
    assert any("value differs" in x for x in BD.verify(p, against=snap, check_sidecar=False))


def test_missing_or_extra_unit_fails(tmp_path):
    snap = _snapshot(tmp_path)
    p = BD.write(snap, pinned_by=["x"])
    q = tmp_path / "copy.h5"
    shutil.copy(p, q)
    with h5py.File(q, "r+") as h5:
        del h5["/cache/kin/s01"]
    assert any("dataset missing" in x for x in BD.verify(q, check_sidecar=False))
    ResultStore(snap).write("kin", "s03", "k9", tables={"t": pd.DataFrame({"x": [1.0]})})
    probs = BD.verify(p, against=snap)
    assert any("unit missing: kin/s03" in x for x in probs)
    assert any("not in the bundle: cache/kin/s03" in x for x in probs)


def test_wrong_snapshot_fails(tmp_path):
    p = BD.write(_snapshot(tmp_path, "r99"), pinned_by=["x"])
    other = _snapshot(tmp_path, "r98")
    probs = BD.verify(p, against=other)
    assert any("compared with 'r98'" in x for x in probs)


def test_interrupted_write_leaves_no_bundle(tmp_path, monkeypatch):
    snap = _snapshot(tmp_path)

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(BD, "_provenance", boom)
    with pytest.raises(KeyboardInterrupt):
        BD.write(snap)
    assert not list(snap.glob("bundle_*"))
    monkeypatch.undo()
    monkeypatch.setattr(BD, "verify", lambda *a, **k: ["forced"])
    with pytest.raises(BD.BundleError):
        BD.write(snap)
    assert not list(snap.glob("bundle_*"))


def test_reader_refuses_an_unpinned_bundle(tmp_path):
    snap = _snapshot(tmp_path)
    p = BD.write(snap)
    with pytest.raises(BD.BundleError, match="not pinned"):
        BD.open_bundle(p)
    BD.open_bundle(p, allow_unpinned=True).close()
    exp = tmp_path / "experiments" / "demo"
    exp.mkdir(parents=True)
    (exp / "study.yaml").write_text("study: demo\n", encoding="utf-8")
    (exp / "paper_demo.yaml").write_text("snapshot: r99\n", encoding="utf-8")
    BD.open_bundle(p, study=exp / "study.yaml").close()
    (exp / "paper_demo.yaml").write_text("snapshot: r100\n", encoding="utf-8")
    with pytest.raises(BD.BundleError):
        BD.open_bundle(p, study=exp / "study.yaml")


def test_archive_round_trip_and_tamper(tmp_path):
    snap = _snapshot(tmp_path)
    a = BD.write_archive(snap)
    assert a.name == "r99.tar.zst" and BD.verify_archive(a, against=snap) == []
    (snap / "cohort.yaml").write_text("participants: [s01]\n", encoding="utf-8")
    assert any("differs" in x for x in BD.verify_archive(a, against=snap))
    with open(a, "ab") as fh:
        fh.write(b"x")
    assert any("sidecar" in x for x in BD.verify_archive(a))


def test_info_for_export_provenance(tmp_path):
    snap = _snapshot(tmp_path)
    assert BD.info(BD.bundle_path(snap)) is None
    p = BD.write(snap, pinned_by=["x"])
    i = BD.info(p)
    assert i["file"] == "bundle_r99.h5" and len(i["sha256"]) == 64
    assert i["sha256"] == BD.file_sha256(p)
