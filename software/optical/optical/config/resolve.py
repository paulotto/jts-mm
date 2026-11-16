# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Resolve monolithic optical.yaml for 1- or 2-camera rigs at runtime."""

from __future__ import annotations

import copy
import sys
from typing import Any

_PROFILE_KEYS = ("1", "2")


def detect_picamera_count() -> int:
    """Return the number of picamera2 devices (0 when unavailable)."""
    try:
        from picamera2 import Picamera2
    except ImportError:
        return 0
    try:
        return len(Picamera2.global_camera_info())
    except Exception:
        return 0


def _profile_key(*, n_cameras: int | None, active_cam_ids: list[str] | None) -> str:
    if active_cam_ids is not None:
        n = len(active_cam_ids)
    elif n_cameras is not None:
        n = int(n_cameras)
    else:
        n = detect_picamera_count()
    if n >= 2:
        return "2"
    if n == 1:
        return "1"
    # Dev / mock hosts with no CSI: default to single-cam profile.
    return "1"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_optical_doc(
    doc: dict[str, Any],
    *,
    active_cam_ids: list[str] | None = None,
    n_cameras: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Apply the 1- or 2-camera profile from ``profiles:`` in *doc*.

    When ``profiles`` is absent the document is returned unchanged (legacy
    flat configs still work). Otherwise the selected profile overrides
    ``rig_id``, ``paths``, ``tracker``, and board geometry in ``calibration``,
    and ``calibration.cameras`` is filtered to the profile ``cam_ids``.
    """
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        return doc

    key = _profile_key(n_cameras=n_cameras, active_cam_ids=active_cam_ids)
    if key not in profiles:
        raise KeyError(f"optical profiles: missing profile {key!r} (have {list(profiles)})")

    profile = profiles[key]
    if not isinstance(profile, dict):
        raise ValueError(f"optical profiles[{key!r}] must be a mapping")

    resolved = copy.deepcopy(doc)
    del resolved["profiles"]

    # Top-level rig id
    if profile.get("rig_id"):
        resolved["rig_id"] = profile["rig_id"]

    # paths.calibration_bundle / tracking_dir
    paths = dict(resolved.get("paths") or {})
    prof_paths = profile.get("paths") or {}
    if isinstance(prof_paths, dict):
        for k, v in prof_paths.items():
            paths[k] = v
    if profile.get("calibration_bundle"):
        paths["calibration_bundle"] = profile["calibration_bundle"]
    if profile.get("tracking_dir"):
        paths["tracking_dir"] = profile["tracking_dir"]
    resolved["paths"] = paths

    # tracker section
    base_tracker = dict(resolved.get("tracker") or {})
    prof_tracker = profile.get("tracker") or {}
    if isinstance(prof_tracker, dict):
        resolved["tracker"] = _deep_merge(base_tracker, prof_tracker)

    # calibration board geometry (ChArUco square/marker sizes differ per rig)
    base_cal = dict(resolved.get("calibration") or {})
    prof_cal = profile.get("calibration") or {}
    if isinstance(prof_cal, dict):
        resolved["calibration"] = _deep_merge(base_cal, prof_cal)

    # Filter cameras to profile cam_ids
    cam_ids = list((resolved.get("tracker") or {}).get("cam_ids") or [])
    all_cams = list((resolved.get("calibration") or {}).get("cameras") or [])
    if cam_ids and all_cams:
        allowed = set(str(c) for c in cam_ids)
        resolved.setdefault("calibration", {})["cameras"] = [
            c for c in all_cams if str(c.get("id")) in allowed
        ]

    if not quiet:
        t = resolved.get("tracker") or {}
        print(
            f"optical: using {key}-camera profile "
            f"(pose_method={t.get('pose_method', '?')}, "
            f"cam_ids={t.get('cam_ids', [])})",
            file=sys.stderr,
        )

    return resolved


def load_resolved_config(
    path: str | Any,
    *,
    active_cam_ids: list[str] | None = None,
    n_cameras: int | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Load YAML/JSON and apply the camera-count profile."""
    from optical.config.loader import load_config_file

    doc = load_config_file(path)
    return resolve_optical_doc(
        doc,
        active_cam_ids=active_cam_ids,
        n_cameras=n_cameras,
        quiet=quiet,
    )
