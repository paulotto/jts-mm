# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Load monolithic ``optical.yaml`` (and legacy JSON) for the optical package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_HOME = Path.home() / "optical.yaml"
_PACKAGE_ETC = Path(__file__).resolve().parents[2] / "etc"
_DEFAULT_ETC_YAML = _PACKAGE_ETC / "optical.yaml"


def repo_root() -> Path:
    """Repository root."""
    return Path(__file__).resolve().parents[4]


def package_etc_dir() -> Path:
    """``software/optical/etc/`` (operator YAML templates)."""
    return _PACKAGE_ETC


def default_config_path() -> Path:
    """Operator optical.yaml: ``~/optical.yaml``, else ``software/optical/etc/optical.yaml``."""
    if _DEFAULT_HOME.is_file():
        return _DEFAULT_HOME
    return _DEFAULT_ETC_YAML


def load_config_file(path: Path | str) -> dict[str, Any]:
    """Load YAML or JSON; root must be a mapping."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return data


def is_monolithic(doc: dict[str, Any]) -> bool:
    """True if this looks like the grouped optical.yaml (has section keys)."""
    return any(k in doc for k in ("markers", "tracker", "calibration", "experiment"))


def markers_section(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the markers mapping (monolith ``markers:`` or legacy marker JSON root)."""
    if "markers" in doc and isinstance(doc["markers"], dict):
        # Monolith markers must look like marker config (detector / id_ranges),
        # not eval marker lists.
        m = doc["markers"]
        if "detector" in m or "id_ranges" in m or "mounts_T_body_tag" in m:
            return m
    if "detector" in doc and "id_ranges" in doc:
        return doc
    raise KeyError("no markers section (expected markers.detector / id_ranges)")


def tracker_section(doc: dict[str, Any]) -> dict[str, Any]:
    """Tracker settings; legacy tracker JSON is the root itself."""
    if "tracker" in doc and isinstance(doc["tracker"], dict):
        return doc["tracker"]
    if "calibration_bundle_path" in doc or "pose_method" in doc:
        return doc
    raise KeyError("no tracker section")


def calibration_section(doc: dict[str, Any], *, config_path: Path | None = None) -> dict[str, Any]:
    """Calibration settings with fiducials / bundle path filled from monolith."""
    if "calibration" in doc and isinstance(doc["calibration"], dict):
        cfg = dict(doc["calibration"])
        cfg.setdefault("schema_version", doc.get("schema_version", "1.0"))
        if "rig_id" not in cfg and doc.get("rig_id"):
            cfg["rig_id"] = doc["rig_id"]
        paths = doc.get("paths") or {}
        if "output" not in cfg and paths.get("calibration_bundle"):
            cfg["output"] = {"bundle_path": paths["calibration_bundle"]}
        fid = dict(cfg.get("fiducials") or {})
        if config_path is not None:
            fid.setdefault("marker_config_path", str(Path(config_path).resolve()))
        if "markers" in doc and isinstance(doc["markers"], dict):
            det = doc["markers"].get("detector") or {}
            fid.setdefault("detector", det.get("type", "aruco"))
            fid.setdefault("dictionary", det.get("dictionary", "DICT_4X4_50"))
            if "tag_size_mm" in det:
                fid.setdefault("tag_size_mm", det["tag_size_mm"])
        cfg["fiducials"] = fid
        return cfg
    if "cameras" in doc and "board" in doc:
        return doc
    raise KeyError("no calibration section")


def experiment_section(doc: dict[str, Any]) -> dict[str, Any]:
    """Experiment plan mapping."""
    if "experiment" in doc and isinstance(doc["experiment"], dict):
        return doc["experiment"]
    if "name" in doc and "phases" in doc:
        return doc
    raise KeyError("no experiment section")


def eval_section(doc: dict[str, Any], key: str = "eval") -> dict[str, Any]:
    """Eval study mapping (``eval`` or ``eval_quality``), or legacy flat eval JSON."""
    if key in doc and isinstance(doc[key], dict):
        return dict(doc[key])
    if "markers" in doc and isinstance(doc.get("markers"), list):
        return doc
    raise KeyError(f"no {key} section")


def calibration_bundle_from_doc(doc: dict[str, Any]) -> Path | None:
    """Resolve calibration bundle path from monolith paths/tracker/calibration."""
    paths = doc.get("paths") or {}
    if paths.get("calibration_bundle"):
        return Path(str(paths["calibration_bundle"]))
    try:
        t = tracker_section(doc)
        if t.get("calibration_bundle_path"):
            return Path(str(t["calibration_bundle_path"]))
    except KeyError:
        pass
    try:
        c = calibration_section(doc)
        out = (c.get("output") or {}).get("bundle_path")
        if out:
            return Path(str(out))
    except KeyError:
        pass
    return None
