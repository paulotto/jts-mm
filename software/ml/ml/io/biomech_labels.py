# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Label sidecars from the biomech ``labels`` stage (``ml labels from-biomech``).

The biomech stage stores per phase ``<phase>__t_us`` and ``<phase>__label``
(REST=0 .. RIGHT=6, -1 = untracked) on the optical frame clock. The exporter
writes one sidecar per included participant for the participant's *primary*
motion take from the cohort manifest (secondary takes are never pooled):

* ``-1`` (untracked) becomes 255 (unlabelled);
* ``optical_frame`` is looked up by ``searchsorted`` on ``/optical/t_master_us``
  (the label times are optical frame times; a mismatch above half a frame is an
  error, the largest offset is stored in ``max_frame_offset_us``);
* ``subject_id`` is set explicitly to the participant id, ``session_id`` to the
  take's canonical id, and ``session_path`` / ``mvc_session`` point at the files
  so ``ml dataset --labels-dir`` can find the take and its MVC take;
* ``mvc_ref_uv`` copies biomech's per-muscle MVC reference (``emg`` stage, table
  ``refs``, ``p95_uv``) when present, so ``ml dataset`` normalizes exactly like the
  ``ml_windows`` stage.

Sidecars are subject-derived data: write them under a git-ignored directory
(default ``<store>/ml_labels``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from ml.config.classes import UNLABELLED
from ml.io.labels import write_labels
from ml.io.session import _load_meta, canonical_session_id


def resolve_data_dir(cohort_path: Path, data_dir: str | Path | None) -> Path:
    """``--data-dir`` > ``$JTSMM_DATA_DIR`` > ``paths.data_dir`` of a sibling ``study.yaml``."""
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    if os.environ.get("JTSMM_DATA_DIR"):
        return Path(os.environ["JTSMM_DATA_DIR"]).expanduser().resolve()
    study = cohort_path.parent / "study.yaml"
    if study.is_file():
        doc = yaml.safe_load(study.read_text(encoding="utf-8")) or {}
        rel = (doc.get("paths") or {}).get("data_dir")
        if rel:
            return (study.parent / rel).resolve()
    raise ValueError("cannot locate the session data: pass --data-dir or set JTSMM_DATA_DIR")


def export_from_biomech(store_dir: str | Path, cohort_path: str | Path, out_dir: str | Path, *,
                        data_dir: str | Path | None = None, pids: list[str] | None = None
                        ) -> list[dict[str, Any]]:
    """Write ``<out_dir>/<motion take>_labels.h5`` per included participant; one report
    row per participant (``status`` ok / skipped with ``reason``)."""
    from biomech.cohort import load_cohort
    from biomech.results.store import ResultStore

    cohort_path = Path(cohort_path)
    cohort = load_cohort(cohort_path, resolve_data_dir(cohort_path, data_dir))
    store = ResultStore(store_dir)
    out = Path(out_dir)
    rows: list[dict[str, Any]] = []
    for p in cohort.included(pids):
        row: dict[str, Any] = {"pid": p.pid, "session": p.motion}
        meta = store.meta("labels", p.pid)
        h5 = cohort.session_path(p.motion) / "session.h5"
        if meta is None or meta.status != "ok":
            rows.append({**row, "status": "skipped", "reason": "no ok labels result in store"})
            continue
        if not h5.is_file():
            rows.append({**row, "status": "skipped", "reason": f"missing {h5}"})
            continue
        a = store.read_arrays("labels", p.pid)
        phases = sorted({k[:-len("__t_us")] for k in a if k.endswith("__t_us")},
                        key=lambda ph: int(a[f"{ph}__t_us"][0]))
        t = np.concatenate([a[f"{ph}__t_us"] for ph in phases]).astype(np.int64)
        y = np.concatenate([a[f"{ph}__label"] for ph in phases]).astype(np.int64)
        if t.size > 1 and not np.all(np.diff(t) > 0):
            raise ValueError(f"{p.pid}: label times of the phases overlap or are unsorted")
        with h5py.File(h5, "r") as f:
            ot = np.asarray(f["optical/t_master_us"][()], dtype=np.int64)
            sid = canonical_session_id(h5, _load_meta(f))
        j = np.clip(np.searchsorted(ot, t), 1, len(ot) - 1)
        frame = np.where(np.abs(ot[j - 1] - t) <= np.abs(ot[j] - t), j - 1, j)
        off = np.abs(ot[frame] - t)
        half = 0.5 * float(np.median(np.diff(ot)))
        if off.max(initial=0) > half:
            raise ValueError(f"{p.pid}: label times are up to {off.max()} us off the optical "
                             f"frame clock (more than half a frame, {half:.0f} us)")
        ylab = np.where(y < 0, UNLABELLED, y).astype(np.uint8)
        mvc = cohort.session_path(p.mvc) / "session.h5" if p.mvc else None
        refs = {}
        em = store.meta("emg", p.pid)
        if em is not None and em.status == "ok" and "refs" in em.tables:
            rt = store.read_table("emg", "refs", [p.pid])
            refs = {str(m): float(v) for m, v in zip(rt.muscle, rt.p95_uv)}
        path = write_labels(
            out / f"{p.motion}_labels.h5", t_master_us=t, optical_frame=frame, y=ylab,
            session_id=sid, subject_id=p.pid, labeler="biomech.labels", source="biomech:labels",
            trajectory_source="biomech_kinematics",
            extra_attrs={"session_path": str(h5.resolve()),
                         "mvc_session": str(mvc.resolve()) if mvc and mvc.is_file() else "",
                         "biomech_store": str(Path(store_dir).resolve()),
                         "labels_key": meta.key, "labels_run_id": meta.run_id or "",
                         "phases": phases, "max_frame_offset_us": int(off.max(initial=0)),
                         "mvc_ref_uv": refs,
                         "mvc_ref_source": (f"biomech emg/refs p95_uv (key {em.key})"
                                            if refs else "")})
        rows.append({**row, "status": "ok", "path": str(path), "n": int(t.size),
                     "n_unlabelled": int(np.sum(ylab == UNLABELLED))})
    return rows
