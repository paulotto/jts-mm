# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Lossless per-snapshot bundle: every result of a frozen snapshot in one HDF5 file, with its
integrity check and a verifying reader (J9.4).

The working store stays the per-unit Parquet / NPZ cache (:mod:`biomech.results.store`): its
content keys make re-runs incremental, ``compare_units`` diffs per unit, cloud sync is
fine-grained and a damaged file loses one unit. A bundle is the read-only consumption format
of a *pinned* snapshot -- one file to hand on, open in MATLAB / h5py / pandas, or archive.

``biomech bundle NAME [--variant V]`` writes ``bundle_<NAME>.h5`` into the snapshot directory
(next to ``snapshot.json``; deleted with the snapshot, never touched by the variant rsync)::

    /cache/<stage>/<unit>/              attrs: status, key, run_id, store_status (snapshot.json)
        meta.json                       the unit's meta.json (raw bytes)
        tables/<table>/<column>         one dataset per column; group attrs: columns (order),
                                        encoding, source, source_sha256; column attrs: label,
                                        unit (catalog.yaml), dtype
        arrays/<key>                    the arrays.npz entries (attrs: numpy dtype)
    /files/<path>                       every other snapshot file (snapshot.json, schema.json,
                                        catalog / study / cohort yaml, sensitivity comparisons)
                                        as raw bytes
    /_bundle/manifest, /_bundle/provenance   canonical JSON (below)

Column encodings (lossless; ``verify --against`` proves it value by value): numeric and bool
columns as they are; ``str`` (text with None: a ``<column>%00null`` mask beside it);
``bool_none`` (True / False / None as int8 1 / 0 / -1); ``none`` (all None). A table that fits
none of them (or has a non-default index / non-text column names) is stored as its Parquet
bytes (encoding ``parquet``). Names are escaped for HDF5 (``%`` -> ``%25``, ``/`` -> ``%2F``, a
leading ``.`` -> ``%2E``); the original names are in the attributes.

Integrity and consistency:

* **manifest** -- one record per unit (status, store status, stale flag, its source files),
  per table (rows, columns, dtypes, encoding), per dataset (path, sha256 of its content:
  dtype, shape and bytes; strings length-prefixed) and per source file (sha256). Units flagged
  failed or stale are present with their flag; nothing is skipped.
* **provenance** -- snapshot name, study, variant, runs, git, code hash, hashes of the
  configuration, study / cohort files and ``snapshot.json``, the paper files that pin the
  snapshot, writer version, creation time.
* **bundle hash** -- sha256 over the canonical manifest and provenance JSON, in the root
  attribute ``bundle_sha256``; the file's own sha256 in the sidecar
  ``bundle_<NAME>.h5.sha256`` (``sha256sum -c`` format).
* **atomic write** -- written to ``*.tmp``, verified against the snapshot, then renamed: a
  bundle under its final name is always complete and verified.
* :func:`verify` -- sidecar, bundle hash, every dataset re-hashed, no missing or extra dataset;
  ``against`` a snapshot directory: the same source files (no unit missing or extra), equal
  source hashes, every table / array / file equal value for value (exact dtypes, NaN-aware),
  provenance equal to ``snapshot.json``.
* :func:`open_bundle` -- checks sidecar and bundle hash on open (``verify=False`` skips it)
  and refuses a bundle whose snapshot is not pinned (``allow_unpinned=True`` to read anyway).

``biomech bundle NAME --archive`` also writes ``<NAME>.tar.zst`` of the snapshot directory
(``SHA256SUMS`` as the first member, sidecar ``.sha256``), checked by :func:`verify_archive`.
``biomech export-data`` stays the pseudonymised public release.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FORMAT = "biomech-bundle"
WRITER_VERSION = "1"
PREFIX = "bundle_"
NULL_SUFFIX = "%00null"
ARCHIVE_SUFFIX = ".tar.zst"


class BundleError(RuntimeError):
    """A bundle that is incomplete, corrupt, unpinned or not the snapshot it claims to be."""


# -- helpers ---------------------------------------------------------------------------------
def bundle_path(snap_dir: str | Path) -> Path:
    snap_dir = Path(snap_dir)
    return snap_dir / f"{PREFIX}{snap_dir.name}.h5"


def archive_path(snap_dir: str | Path) -> Path:
    snap_dir = Path(snap_dir)
    return snap_dir.parent / f"{snap_dir.name}{ARCHIVE_SUFFIX}"


def sidecar(path: str | Path) -> Path:
    return Path(f"{path}.sha256")


def h5name(name: str) -> str:
    s = str(name).replace("%", "%25").replace("/", "%2F")
    return "%2E" + s[1:] if s.startswith(".") else (s or "%00")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dataset_sha256(ds: Any) -> str:
    """Content hash of one HDF5 dataset: dtype, shape and bytes (C order); variable-length
    strings element by element, each prefixed by its length."""
    import h5py

    h = hashlib.sha256()
    v = ds[()]
    if h5py.check_string_dtype(ds.dtype) is not None:
        h.update(f"vstr|{ds.shape}|".encode())
        for x in np.asarray(v, dtype=object).ravel():
            b = x if isinstance(x, bytes) else str(x).encode()
            h.update(len(b).to_bytes(8, "little"))
            h.update(b)
    else:
        a = np.ascontiguousarray(v)
        h.update(f"{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _source_files(snap_dir: Path) -> list[str]:
    """Every file of a snapshot directory (relative POSIX paths), without bundles, archives,
    sidecars, temp files and dot-prefixed entries (as :func:`snapshot._copy_tree`)."""
    out = []
    for dirpath, dirnames, filenames in os.walk(snap_dir):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        rel = Path(dirpath).relative_to(snap_dir)
        for f in sorted(filenames):
            if f.startswith(".") or (rel == Path(".") and f.startswith(PREFIX)):
                continue
            out.append((rel / f).as_posix())
    return out


# -- table encodings -------------------------------------------------------------------------
def _encode_table(df: Any) -> tuple[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] | None:
    """(encoding, {column: (data, attrs)}) or None when the table needs the Parquet fallback."""
    import pandas as pd

    if not isinstance(df.index, pd.RangeIndex) or df.index.start != 0 or df.index.step != 1:
        return None
    if not all(isinstance(c, str) for c in df.columns) or df.columns.duplicated().any():
        return None
    if len(df.columns) == 0:        # a table without columns (e.g. an empty notes table): the
        return None                 # column encoding cannot restore its column index type
    cols: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for c in df.columns:
        s = df[c]
        dt = str(s.dtype)
        if s.dtype.kind in "biuf" and isinstance(s.dtype, np.dtype):
            cols[c] = (s.to_numpy(), {"encoding": "native", "dtype": dt})
            continue
        if s.dtype != object:
            return None
        types = {type(v) for v in s}
        if types <= {str, type(None)} and str in types:
            mask = s.isna().to_numpy() if type(None) in types else None
            data = np.array(["" if v is None else v for v in s], dtype=object)
            cols[c] = (data, {"encoding": "str", "dtype": dt, "null": mask})
        elif types <= {bool, type(None)} and bool in types:
            data = np.array([-1 if v is None else int(v) for v in s], dtype=np.int8)
            cols[c] = (data, {"encoding": "bool_none", "dtype": dt})
        elif types <= {type(None)}:
            cols[c] = (np.zeros(len(s), dtype=np.uint8), {"encoding": "none", "dtype": dt})
        else:
            return None
    return "columns", cols


def _decode_column(g: Any, name: str) -> Any:
    import pandas as pd

    ds = g[h5name(name)]
    enc = ds.attrs["encoding"]
    if enc == "native":
        return pd.Series(ds[()], name=name)
    if enc == "str":
        v = np.array(ds.asstr()[()], dtype=object)
        nk = h5name(name) + NULL_SUFFIX
        if nk in g:
            v[g[nk][()].astype(bool)] = None
        return pd.Series(v, name=name, dtype=object)
    if enc == "bool_none":
        v = ds[()]
        return pd.Series([None if x < 0 else bool(x) for x in v], name=name, dtype=object)
    if enc == "none":
        return pd.Series([None] * ds.shape[0], name=name, dtype=object)
    raise BundleError(f"{ds.name}: unknown column encoding {enc!r}")


def _decode_table(g: Any) -> Any:
    import pandas as pd

    if g.attrs["encoding"] == "parquet":
        return pd.read_parquet(io.BytesIO(g["__parquet__"][()].tobytes()))
    cols = json.loads(g.attrs["columns"])
    n = int(g.attrs["rows"])
    if not cols:
        return pd.DataFrame(index=pd.RangeIndex(n))
    return pd.concat([_decode_column(g, c) for c in cols], axis=1)


# -- write -----------------------------------------------------------------------------------
class _Writer:
    def __init__(self, h5: Any) -> None:
        self.h5 = h5
        self.records: list[dict[str, Any]] = []

    def dataset(self, path: str, data: np.ndarray, rec: dict[str, Any],
                attrs: dict[str, Any] | None = None) -> None:
        import h5py

        kw: dict[str, Any] = {}
        if data.dtype == object or data.dtype.kind == "U":
            data = np.asarray(data, dtype=object)
            kw["dtype"] = h5py.string_dtype("utf-8")
        if data.ndim and data.size:
            kw.update(compression="gzip", compression_opts=4, shuffle=data.dtype != object)
        ds = self.h5.create_dataset(path, data=data, **kw)
        for k, v in (attrs or {}).items():
            ds.attrs[k] = v
        self.records.append({**rec, "path": path, "sha256": dataset_sha256(ds),
                             "shape": list(ds.shape)})

    def raw(self, path: str, src: Path, rel: str, rec: dict[str, Any]) -> None:
        b = src.read_bytes()
        self.dataset(path, np.frombuffer(b, dtype=np.uint8), {
            **rec, "kind": rec.get("kind", "file"), "source": rel, "source_sha256": _sha(b),
            "bytes": len(b)})


def _write_table(w: _Writer, gpath: str, src: Path, rel: str, rec: dict[str, Any]) -> None:
    import pandas as pd

    from biomech.results.schema import column_entry

    raw = src.read_bytes()
    df = pd.read_parquet(io.BytesIO(raw))
    enc = _encode_table(df)
    g = w.h5.create_group(gpath)
    g.attrs.update(source=rel, source_sha256=_sha(raw), rows=len(df),
                   columns=json.dumps([str(c) for c in df.columns]))
    trec = {**rec, "kind": "table", "path": gpath, "rows": len(df),
            "columns": [str(c) for c in df.columns], "dtypes": [str(t) for t in df.dtypes],
            "source": rel, "source_sha256": _sha(raw)}
    if enc is None:
        g.attrs["encoding"] = "parquet"
        w.dataset(f"{gpath}/__parquet__", np.frombuffer(raw, dtype=np.uint8),
                  {**rec, "kind": "table_bytes"})
        w.records.append({**trec, "encoding": "parquet"})
        return
    g.attrs["encoding"] = "columns"
    for c, (data, a) in enc[1].items():
        null = a.pop("null", None)
        e = column_entry(c)
        w.dataset(f"{gpath}/{h5name(c)}", data,
                  {**rec, "kind": "column", "column": c, "encoding": a["encoding"],
                   "dtype": a["dtype"], "label": e["label"], "measure_unit": e["unit"]},
                  {**a, "name": c, "label": e["label"] or "", "unit": e["unit"] or ""})
        if null is not None:
            w.dataset(f"{gpath}/{h5name(c)}{NULL_SUFFIX}", null.astype(np.uint8),
                      {**rec, "kind": "null_mask", "column": c})
    w.records.append({**trec, "encoding": "columns"})


def _write_unit(w: _Writer, snap_dir: Path, stage: str, unit: str,
                store_status: dict[tuple[str, str], str]) -> None:
    udir = snap_dir / "cache" / stage / unit
    gpath = f"/cache/{h5name(stage)}/{h5name(unit)}"
    g = w.h5.require_group(gpath)
    meta: dict[str, Any] = {}
    if (udir / "meta.json").is_file():
        meta = json.loads((udir / "meta.json").read_text(encoding="utf-8"))
    st = store_status.get((stage, unit), "unknown")
    g.attrs.update(status=str(meta.get("status", "missing-meta")), key=str(meta.get("key", "")),
                   run_id=str(meta.get("run_id") or ""), store_status=st)
    base = {"stage": stage, "unit": unit}
    files = []
    for f in sorted(p for p in udir.iterdir() if p.is_file() and not p.name.startswith(".")):
        rel = f.relative_to(snap_dir).as_posix()
        files.append(rel)
        if f.suffix == ".parquet":
            _write_table(w, f"{gpath}/tables/{h5name(f.stem)}", f, rel,
                         {**base, "table": f.stem})
        elif f.name == "arrays.npz":
            raw = f.read_bytes()
            with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                for k in z.files:
                    a = z[k]
                    w.dataset(f"{gpath}/arrays/{h5name(k)}", a,
                              {**base, "kind": "array", "key": k, "dtype": a.dtype.str,
                               "source": rel, "source_sha256": _sha(raw)},
                              {"name": k, "dtype": a.dtype.str})
        else:
            w.raw(f"{gpath}/{h5name(f.name)}", f, rel, {**base, "kind": "file"})
    w.records.append({**base, "kind": "unit", "path": gpath,
                      "status": meta.get("status", "missing-meta"), "store_status": st,
                      "stale": st != "up-to-date", "tables": meta.get("tables") or [],
                      "has_arrays": bool(meta.get("has_arrays")), "files": files})


def _provenance(snap_dir: Path, info: dict[str, Any],
                pinned_by: Iterable[str]) -> dict[str, Any]:
    from biomech.results.provenance import git_state

    def opt_sha(name: str) -> str | None:
        p = snap_dir / name
        return file_sha256(p) if p.is_file() else None

    src = Path(str(info.get("source_out_dir", "")))
    variant = src.name if src.parent.name == "sensitivity" else None
    return {"format": FORMAT, "writer_version": WRITER_VERSION,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "snapshot": info.get("name"), "study": info.get("study"), "variant": variant,
            "snapshot_created_utc": info.get("created_utc"), "note": info.get("note"),
            "runs": info.get("runs") or [], "git": info.get("git"),
            "code_hash": info.get("code_hash"), "stale_units": info.get("stale_units"),
            "config_sha256": _sha(_canon(info.get("config"))),
            "snapshot_json_sha256": opt_sha("snapshot.json"),
            "study_yaml_sha256": opt_sha("study.yaml"),
            "cohort_yaml_sha256": opt_sha("cohort.yaml"),
            "pinned_by": sorted(set(pinned_by)), "writer_git": git_state()}


def write(snap_dir: str | Path, out: str | Path | None = None, *, overwrite: bool = False,
          pinned_by: Iterable[str] = ()) -> Path:
    """Bundle snapshot directory ``snap_dir`` (default target :func:`bundle_path`); verified
    against the snapshot before it gets its final name. Returns the bundle path."""
    import h5py

    snap_dir = Path(snap_dir)
    info = json.loads((snap_dir / "snapshot.json").read_text(encoding="utf-8"))
    final = Path(out) if out else bundle_path(snap_dir)
    if final.exists() and not overwrite:
        raise FileExistsError(f"{final} exists (use --overwrite)")
    tmp = final.with_name(final.name + ".tmp")
    tmp.unlink(missing_ok=True)
    store_status = {(r["stage"], r["unit"]): r["status"] for r in info.get("status") or []}
    try:
        with h5py.File(tmp, "w") as h5:
            w = _Writer(h5)
            cache = snap_dir / "cache"
            for sdir in sorted(p for p in cache.iterdir() if p.is_dir()
                               and not p.name.startswith(".")) if cache.is_dir() else []:
                for udir in sorted(p for p in sdir.iterdir() if p.is_dir()
                                   and not p.name.startswith(".")):
                    _write_unit(w, snap_dir, sdir.name, udir.name, store_status)
            for rel in _source_files(snap_dir):
                if not rel.startswith("cache/"):
                    w.raw(f"/files/{'/'.join(h5name(p) for p in rel.split('/'))}",
                          snap_dir / rel, rel, {"kind": "file"})
            manifest = sorted(w.records, key=lambda r: (r["path"], r["kind"]))
            prov = _provenance(snap_dir, info, pinned_by)
            h5.create_dataset("/_bundle/manifest", data=_canon(manifest).decode(),
                              dtype=h5py.string_dtype("utf-8"))
            h5.create_dataset("/_bundle/provenance", data=_canon(prov).decode(),
                              dtype=h5py.string_dtype("utf-8"))
            h5.attrs.update(format=FORMAT, writer_version=WRITER_VERSION,
                            snapshot=str(info.get("name")),
                            bundle_sha256=_sha(_canon(manifest) + b"\n" + _canon(prov)))
        probs = verify(tmp, against=snap_dir, check_sidecar=False)
        if probs:
            raise BundleError(f"{final.name}: the fresh bundle fails its verification: "
                              + " | ".join(probs[:10]))
        digest = file_sha256(tmp)
        os.replace(tmp, final)
        sc = sidecar(final)
        sc_tmp = sc.with_name(sc.name + ".tmp")
        sc_tmp.write_text(f"{digest}  {final.name}\n", encoding="utf-8")
        os.replace(sc_tmp, sc)
    finally:
        tmp.unlink(missing_ok=True)
    return final


# -- verify ----------------------------------------------------------------------------------
def _read_json(h5: Any, path: str) -> Any:
    v = h5[path][()]
    return json.loads(v.decode() if isinstance(v, bytes) else v)


def sidecar_problems(path: str | Path) -> list[str]:
    """The file's sha256 against its sidecar ``<file>.sha256``."""
    path = Path(path)
    sc = sidecar(path)
    if not sc.is_file():
        return [f"{path.name}: no sidecar {sc.name}"]
    want = (sc.read_text(encoding="utf-8").split() or [""])[0]
    got = file_sha256(path)
    return [] if want == got else [f"{path.name}: file sha256 {got[:12]} != sidecar {want[:12]}"]


def _bundle_hash(h5: Any) -> list[str]:
    m = h5["/_bundle/manifest"][()]
    p = h5["/_bundle/provenance"][()]
    m, p = (x if isinstance(x, bytes) else x.encode() for x in (m, p))
    got = _sha(m + b"\n" + p)
    want = str(h5.attrs.get("bundle_sha256", ""))
    return [] if got == want else [f"bundle hash {got[:12]} != recorded {want[:12]}"]


def verify(path: str | Path, against: str | Path | None = None, *,
           check_sidecar: bool = True) -> list[str]:
    """Problems of bundle ``path`` (empty list: intact and, with ``against`` a snapshot
    directory, equal to that snapshot). See the module docstring for what is checked."""
    import h5py
    import pandas as pd

    path = Path(path)
    probs: list[str] = []
    if check_sidecar:
        probs += sidecar_problems(path)
    try:
        h5 = h5py.File(path, "r")
    except OSError as exc:
        return probs + [f"{path.name}: not readable as HDF5 ({exc})"]
    with h5:
        if h5.attrs.get("format") != FORMAT:
            return probs + [f"{path.name}: not a {FORMAT} file"]
        probs += _bundle_hash(h5)
        manifest = _read_json(h5, "/_bundle/manifest")
        prov = _read_json(h5, "/_bundle/provenance")
        recs = {r["path"]: r for r in manifest if "sha256" in r}
        present: list[str] = []
        h5.visititems(lambda n, o: present.append("/" + n)
                      if isinstance(o, h5py.Dataset) and not n.startswith("_bundle/") else None)
        for p in sorted(set(recs) - set(present)):
            probs.append(f"dataset missing: {p}")
        for p in sorted(set(present) - set(recs)):
            probs.append(f"dataset not in the manifest: {p}")
        for p in sorted(set(recs) & set(present)):
            if dataset_sha256(h5[p]) != recs[p]["sha256"]:
                probs.append(f"content hash differs: {p}")
        for r in manifest:
            if r["kind"] == "table" and r["path"] in h5:
                if json.loads(h5[r["path"]].attrs["columns"]) != r["columns"]:
                    probs.append(f"column list differs: {r['path']}")
        if against is None:
            return probs
        snap = Path(against)
        info = json.loads((snap / "snapshot.json").read_text(encoding="utf-8"))
        if prov.get("snapshot") != info.get("name"):
            probs.append(f"bundle of snapshot {prov.get('snapshot')!r}, compared with "
                         f"{info.get('name')!r}")
        if prov.get("snapshot_json_sha256") != file_sha256(snap / "snapshot.json"):
            probs.append("snapshot.json differs from the one bundled (provenance)")
        for k in ("runs", "code_hash", "git"):
            if _canon(prov.get(k)) != _canon(info.get(k)):
                probs.append(f"provenance {k} differs from snapshot.json")
        sources: dict[str, str] = {}
        for r in manifest:
            if "source" in r:
                sources[r["source"]] = r["source_sha256"]
        on_disk = set(_source_files(snap))
        for rel in sorted(on_disk - set(sources)):
            probs.append(f"not in the bundle: {rel}")
        for rel in sorted(set(sources) - on_disk):
            probs.append(f"in the bundle but not in the snapshot: {rel}")
        for rel in sorted(on_disk & set(sources)):
            if file_sha256(snap / rel) != sources[rel]:
                probs.append(f"source hash differs: {rel}")
        units = {(r["stage"], r["unit"]) for r in manifest if r["kind"] == "unit"}
        units_disk = {tuple(rel.split("/")[1:3]) for rel in on_disk if rel.startswith("cache/")}
        for su in sorted(units_disk - units):
            probs.append(f"unit missing: {'/'.join(su)}")
        for su in sorted(units - units_disk):
            probs.append(f"extra unit: {'/'.join(su)}")
        # value by value: what a reader gets equals what the store reader gets
        for r in manifest:
            if r["path"] not in h5:
                continue
            src = snap / r.get("source", "")
            if not src.is_file():
                continue
            try:
                if r["kind"] == "table":
                    pd.testing.assert_frame_equal(_decode_table(h5[r["path"]]),
                                                  pd.read_parquet(src), check_exact=True)
                elif r["kind"] == "array":
                    with np.load(src, allow_pickle=False) as z:
                        a, b = _array(h5[r["path"]]), z[r["key"]]
                    if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(
                            a, b, equal_nan=a.dtype.kind in "fc"):
                        raise AssertionError(f"{a.dtype}{a.shape} vs {b.dtype}{b.shape}")
                elif r["kind"] == "file":
                    if h5[r["path"]][()].tobytes() != src.read_bytes():
                        raise AssertionError("bytes differ")
            except AssertionError as exc:
                probs.append(f"value differs: {r['path']} ({str(exc).splitlines()[0][:120]})")
    return probs


def _array(ds: Any) -> np.ndarray:
    import h5py

    if h5py.check_string_dtype(ds.dtype) is not None:
        return np.array(ds.asstr()[()], dtype=ds.attrs["dtype"])
    return np.asarray(ds[()])


# -- read ------------------------------------------------------------------------------------
class Bundle:
    """Read access to a bundle with the :class:`~biomech.results.store.ResultStore` reading
    interface (``stages``, ``units``, ``meta``, ``read_table``, ``read_arrays``) plus
    ``file``, ``comparison``, ``manifest`` and ``provenance``."""

    def __init__(self, path: Path, h5: Any) -> None:
        self.path, self.h5 = path, h5
        self.provenance: dict[str, Any] = _read_json(h5, "/_bundle/provenance")
        self._manifest: list[dict[str, Any]] = _read_json(h5, "/_bundle/manifest")
        self.snapshot: str = str(self.provenance.get("snapshot"))
        self._units = {(r["stage"], r["unit"]): r for r in self._manifest if r["kind"] == "unit"}

    @property
    def manifest(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self._manifest)

    def stages(self) -> list[str]:
        return sorted({s for s, _ in self._units})

    def units(self, stage: str) -> list[str]:
        return sorted(u for s, u in self._units if s == stage)

    def meta(self, stage: str, unit: str) -> dict[str, Any] | None:
        p = f"/cache/{h5name(stage)}/{h5name(unit)}/meta.json"
        return json.loads(self.h5[p][()].tobytes()) if p in self.h5 else None

    def table(self, stage: str, unit: str, name: str) -> Any:
        return _decode_table(self.h5[f"/cache/{h5name(stage)}/{h5name(unit)}/tables/"
                                     f"{h5name(name)}"])

    def read_table(self, stage: str, name: str, units: Iterable[str] | None = None, *,
                   ok_only: bool = True) -> Any:
        """One table concatenated over units, with a ``unit`` column (as the store)."""
        import pandas as pd

        frames = []
        for u in units if units is not None else self.units(stage):
            r = self._units.get((stage, u))
            if r is None or (ok_only and r["status"] != "ok") or name not in r["tables"]:
                continue
            df = self.table(stage, u, name)
            if "unit" not in df.columns:
                df.insert(0, "unit", u)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def read_arrays(self, stage: str, unit: str) -> dict[str, np.ndarray]:
        p = f"/cache/{h5name(stage)}/{h5name(unit)}/arrays"
        if p not in self.h5:
            return {}
        return {ds.attrs["name"]: _array(ds) for ds in self.h5[p].values()}

    def file(self, rel: str) -> bytes:
        return self.h5["/files/" + "/".join(h5name(p) for p in rel.split("/"))][()].tobytes()

    def comparison(self, variant: str | None = None) -> Any:
        """A sensitivity comparison frozen with the snapshot (``variant`` None: the variant
        store's own ``comparison.parquet``)."""
        import pandas as pd

        rel = f"sensitivity/{variant}/comparison.parquet" if variant else "comparison.parquet"
        return pd.read_parquet(io.BytesIO(self.file(rel)))

    def close(self) -> None:
        self.h5.close()

    def __enter__(self) -> "Bundle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def open_bundle(path: str | Path, *, verify: bool = True, study: str | Path | None = None,
                allow_unpinned: bool = False) -> Bundle:
    """Open a bundle for reading. ``verify``: sidecar and bundle hash first (the full
    re-hash is :func:`verify`). The snapshot must be pinned -- by the paper files of
    ``study`` (a study YAML: the current pins) or, without it, by the pins recorded at
    writing -- unless ``allow_unpinned``."""
    import h5py

    path = Path(path)
    if verify:
        probs = sidecar_problems(path)
        if probs:
            raise BundleError(" | ".join(probs))
    h5 = h5py.File(path, "r")
    try:
        if h5.attrs.get("format") != FORMAT:
            raise BundleError(f"{path.name}: not a {FORMAT} file")
        if verify:
            probs = _bundle_hash(h5)
            if probs:
                raise BundleError(f"{path.name}: " + " | ".join(probs))
        b = Bundle(path, h5)
        if not allow_unpinned:
            if study is not None:
                from biomech.results.snapshot import pinned

                pins = pinned(study).get(b.snapshot)
            else:
                pins = b.provenance.get("pinned_by")
            if not pins:
                raise BundleError(f"{path.name}: snapshot {b.snapshot} is not pinned by a "
                                  "paper set (allow_unpinned=True to read it anyway)")
        return b
    except BaseException:
        h5.close()
        raise


def info(path: str | Path) -> dict[str, Any] | None:
    """File name, sidecar sha256 and bundle hash of a bundle (None if absent), cheap: for the
    export provenance (``verify-export`` ties an export to its bundle)."""
    import h5py

    path = Path(path)
    if not path.is_file():
        return None
    sc = sidecar(path)
    with h5py.File(path, "r") as h5:
        bh = str(h5.attrs.get("bundle_sha256", ""))
    return {"file": path.name, "bundle_sha256": bh,
            "sha256": sc.read_text(encoding="utf-8").split()[0] if sc.is_file() else None}


# -- archive ---------------------------------------------------------------------------------
def write_archive(snap_dir: str | Path, out: str | Path | None = None, *,
                  overwrite: bool = False) -> Path:
    """``<NAME>.tar.zst`` of the snapshot directory (without bundles), ``SHA256SUMS`` first,
    written atomically, verified against the snapshot, with a sidecar ``.sha256``."""
    import tarfile

    import zstandard

    snap_dir = Path(snap_dir)
    final = Path(out) if out else archive_path(snap_dir)
    if final.exists() and not overwrite:
        raise FileExistsError(f"{final} exists (use --overwrite)")
    tmp = final.with_name(final.name + ".tmp")
    files = _source_files(snap_dir)
    sums = "".join(f"{file_sha256(snap_dir / f)}  {snap_dir.name}/{f}\n" for f in files).encode()
    try:
        with open(tmp, "wb") as fh, zstandard.ZstdCompressor(level=10).stream_writer(fh) as zw, \
                tarfile.open(fileobj=zw, mode="w|") as tar:
            ti = tarfile.TarInfo(f"{snap_dir.name}/SHA256SUMS")
            ti.size, ti.mtime = len(sums), int(time.time())
            tar.addfile(ti, io.BytesIO(sums))
            for f in files:
                tar.add(snap_dir / f, arcname=f"{snap_dir.name}/{f}", recursive=False)
        probs = verify_archive(tmp, against=snap_dir, check_sidecar=False)
        if probs:
            raise BundleError(f"{final.name}: " + " | ".join(probs[:10]))
        digest = file_sha256(tmp)
        os.replace(tmp, final)
        sidecar(final).write_text(f"{digest}  {final.name}\n", encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)
    return final


def verify_archive(path: str | Path, against: str | Path | None = None, *,
                   check_sidecar: bool = True) -> list[str]:
    """Problems of a ``.tar.zst`` snapshot archive: sidecar, every member against
    ``SHA256SUMS``, no member missing or extra; ``against``: the same files as the snapshot."""
    import tarfile

    import zstandard

    path = Path(path)
    probs = sidecar_problems(path) if check_sidecar else []
    want: dict[str, str] = {}
    got: dict[str, str] = {}
    with open(path, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as zr, \
            tarfile.open(fileobj=zr, mode="r|") as tar:
        for m in tar:
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            if m.name.endswith("/SHA256SUMS") and not want:
                for line in data.decode().splitlines():
                    h, name = line.split("  ", 1)
                    want[name] = h
            else:
                got[m.name] = _sha(data)
    if not want:
        return probs + [f"{path.name}: no SHA256SUMS"]
    for n in sorted(set(want) - set(got)):
        probs.append(f"member missing: {n}")
    for n in sorted(set(got) - set(want)):
        probs.append(f"member not in SHA256SUMS: {n}")
    for n in sorted(set(want) & set(got)):
        if want[n] != got[n]:
            probs.append(f"member hash differs: {n}")
    if against is not None:
        snap = Path(against)
        disk = {f"{snap.name}/{f}": file_sha256(snap / f) for f in _source_files(snap)}
        if disk != want:
            diff = sorted(set(disk.items()) ^ set(want.items()))
            probs.append(f"archive differs from {snap}: {len(diff)} entries "
                         f"(e.g. {diff[0][0] if diff else ''})")
    return probs
