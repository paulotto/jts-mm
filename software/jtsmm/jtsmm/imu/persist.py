# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Persist IMU–optical hand-eye calibration takes for §5.2.2 eval."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from jtsmm.imu.mounting import MAX_RMS_DEG, MIN_PAIRS, MIN_SAMPLES
from jtsmm.io.session_layout import (
    IMU_CALIB_RECORDINGS_DEFAULT,
    imu_calibration_recording_dir,
)

IMU_MOUNTING_JSON = "imu_mounting.json"
SESSION_H5_NAME = "session.h5"
KIND = "imu_optical_handeye"
SCHEMA_VERSION = "1.0"

__all__ = [
    "IMU_MOUNTING_JSON",
    "KIND",
    "SESSION_H5_NAME",
    "discover_runs",
    "load_run",
    "next_run",
    "read_session_subject_id",
    "slot_record_from_error",
    "write_imu_mounting_recording",
]


def slot_record_from_error(
    slot: str,
    error: str,
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "accepted": False,
        "rms_deg": None,
        "n_pairs": None,
        "n_samples": None,
        "n_selected": None,
        "phase": phase,
        "error": str(error),
        "axes_3d": None,
        "axis_ratio_opt": None,
        "axis_ratio_imu": None,
        "R_mount": None,
        "T": None,
    }


def read_session_subject_id(path: Path | str) -> str | None:
    """``/meta.subject_id`` from a session HDF5, if present."""
    import h5py

    with h5py.File(path, "r") as f:
        if "meta" not in f:
            return None
        raw = f["meta"].attrs.get("subject_id")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", "replace").strip()
    else:
        text = str(raw).strip()
    return text or None


def next_run(recordings_dir: Path | str | None) -> int:
    """Next independent-repetition index (1-based) under the recordings tree.

    Mounting cal is per rigid IMU–array pair, not per subject — every persisted
    leaf counts, same idea as CT ``ct_tip_calibration_N``.
    """
    base = Path(recordings_dir).expanduser() if recordings_dir else Path(
        IMU_CALIB_RECORDINGS_DEFAULT
    ).expanduser()
    highest = 0
    for path in discover_runs(base):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw = data.get("run", data.get("attempt", 0))
        try:
            highest = max(highest, int(raw or 0))
        except (TypeError, ValueError):
            continue
    return highest + 1


def discover_runs(recordings_dir: Path | str) -> list[Path]:
    """Find ``imu_mounting.json`` files under ``recordings_dir``."""
    root = Path(recordings_dir).expanduser()
    if not root.is_dir():
        return []
    found = sorted(root.rglob(IMU_MOUNTING_JSON))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def load_run(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _snapshot_optical(
    leaf: Path,
    *,
    optical_config_path: Path | str | None,
) -> dict[str, str]:
    """Copy optical.yaml + calibration bundle into the leaf when available."""
    out: dict[str, str] = {}
    if optical_config_path is None:
        return out
    src_cfg = Path(optical_config_path).expanduser()
    if not src_cfg.is_file():
        return out
    try:
        from optical.paths import (
            CALIBRATION_BUNDLE_SNAPSHOT_NAME,
            OPTICAL_CONFIG_SNAPSHOT_NAME,
            snapshot_persist_calibration,
        )
    except ImportError:
        dest_cfg = leaf / "optical.yaml"
        shutil.copy2(src_cfg, dest_cfg)
        out["optical_config"] = dest_cfg.name
        return out

    bundle: Path | None = None
    try:
        from optical.config.loader import calibration_bundle_from_doc, load_config_file

        doc = load_config_file(src_cfg)
        raw = calibration_bundle_from_doc(doc)
        if raw is not None:
            candidate = Path(raw).expanduser()
            if not candidate.is_file():
                candidate = (src_cfg.parent / raw).resolve()
            if candidate.is_file():
                bundle = candidate
    except Exception:  # noqa: BLE001 — snapshot is best-effort
        bundle = None

    if bundle is None:
        dest_cfg = leaf / OPTICAL_CONFIG_SNAPSHOT_NAME
        shutil.copy2(src_cfg, dest_cfg)
        out["optical_config"] = dest_cfg.name
        return out

    arts = snapshot_persist_calibration(
        leaf, bundle_path=bundle, config_path=src_cfg
    )
    out["calibration_bundle"] = arts["calibration_bundle"].name
    out["calibration_bundle_source"] = str(bundle.resolve())
    if "optical_config" in arts:
        out["optical_config"] = arts["optical_config"].name
    return out


def write_imu_mounting_recording(
    out_dir: Path | str,
    slots: dict[str, dict[str, Any]],
    *,
    session_h5: Path | str | None = None,
    subject_id: str | None = None,
    run: int = 1,
    experiment_name: str | None = None,
    source_session: Path | str | None = None,
    optical_config_path: Path | str | None = None,
    max_rms_deg: float = MAX_RMS_DEG,
    min_pairs: int = MIN_PAIRS,
    min_samples: int = MIN_SAMPLES,
) -> Path:
    """Write ``imu_mounting.json`` + session snapshot + optional optical cal.

    ``slots`` maps ``imu_mta`` / ``imu_cra`` to ``to_slot_dict()`` (or error) records.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    snap: dict[str, str] = {}
    if optical_config_path is not None:
        try:
            snap = _snapshot_optical(root, optical_config_path=optical_config_path)
        except Exception as exc:  # noqa: BLE001 — persist must not abort calib
            print(
                f"warning: IMU cal optical snapshot failed ({exc})",
                file=sys.stderr,
                flush=True,
            )

    session_rel: str | None = None
    src_h5 = Path(session_h5).expanduser() if session_h5 is not None else None
    if src_h5 is not None and src_h5.is_file():
        dest_h5 = root / SESSION_H5_NAME
        if src_h5.resolve() != dest_h5.resolve():
            _link_or_copy(src_h5, dest_h5)
        session_rel = SESSION_H5_NAME

    accepted = bool(slots) and all(bool(rec.get("accepted")) for rec in slots.values())
    payload: dict[str, Any] = {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "experiment_name": experiment_name,
        "subject_id": subject_id,
        "run": int(run),
        "accepted": accepted,
        "max_rms_deg": float(max_rms_deg),
        "min_pairs": int(min_pairs),
        "min_samples": int(min_samples),
        "session_h5": session_rel,
        "source_session": str(Path(source_session).expanduser()) if source_session else None,
        "slots": slots,
    }
    if snap.get("calibration_bundle"):
        payload["calibration_bundle"] = snap["calibration_bundle"]
        payload["calibration_bundle_source"] = snap.get("calibration_bundle_source")
    (root / IMU_MOUNTING_JSON).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subject_id": subject_id,
        "run": int(run),
        "accepted": accepted,
        "notes": (
            f"imu optical hand-eye repeatability: {experiment_name or 'unnamed'} run {int(run)}"
        ),
        "session_h5": session_rel,
    }
    if snap.get("optical_config"):
        meta["optical_config"] = snap["optical_config"]
    if snap.get("calibration_bundle"):
        meta["calibration_bundle"] = snap["calibration_bundle"]
        meta["calibration_bundle_source"] = snap.get("calibration_bundle_source")
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return root


def persist_dir_for(
    recordings_dir: Path | str | None,
    *,
    experiment_name: str | None,
    stamp: str | None = None,
) -> Path:
    return imu_calibration_recording_dir(
        recordings_dir,
        experiment_name=experiment_name,
        stamp=stamp,
    )
