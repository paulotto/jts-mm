# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Calibration bundle export after extrinsics capture."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("cv2")

from optical.calibration.extrinsics import (  # noqa: E402
    ExtrinsicsFit,
    calibration_cache_paths,
    intrinsics_fingerprint,
)
from optical.calibration.io import CameraExtrinsics, CameraIntrinsics, read_calibration_bundle  # noqa: E402
from optical.calibration.multicam import CamSpec, _export_calibration_bundle, run_calibration  # noqa: E402


def _intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        K=np.eye(3, dtype=np.float64),
        dist=np.zeros(5, dtype=np.float64),
        image_size=(640, 480),
        rms_px=0.4,
    )


def _write_intrinsics_cache(bundle: Path, cam_ids: list[str]) -> None:
    intr = _intrinsics()
    np.savez(
        calibration_cache_paths(bundle)[0],
        **{f"{cid}_K": intr.K for cid in cam_ids},
        **{f"{cid}_dist": intr.dist for cid in cam_ids},
        **{f"{cid}_image_size": np.asarray(intr.image_size, dtype=np.int32) for cid in cam_ids},
        **{f"{cid}_rms_px": np.asarray(intr.rms_px, dtype=np.float64) for cid in cam_ids},
    )


def test_export_calibration_bundle_writes_h5(tmp_path: Path) -> None:
    cams = [CamSpec(cam_id="cam0", camera_num=0), CamSpec(cam_id="cam1", camera_num=1)]
    intr = {cid: _intrinsics() for cid in ("cam0", "cam1")}
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = 120.0
    np.savez(
        tmp_path / "extrinsics_cache.npz",
        cam0_T_world_cam=np.eye(4, dtype=np.float64),
        cam0_rms_px=np.asarray(0.3, dtype=np.float64),
        cam1_T_world_cam=T,
        cam1_rms_px=np.asarray(0.5, dtype=np.float64),
        intrinsics_sha256=np.array(intrinsics_fingerprint(intr)),
    )
    out = tmp_path / "calibration_bundle.h5"
    _export_calibration_bundle(
        cfg={"schema_version": "1.0", "fiducials": {}, "notes": "test"},
        cams=cams,
        intr=intr,
        extr_path=tmp_path / "extrinsics_cache.npz",
        bundle_path=out,
        rig_id="test_rig",
    )
    bundle = read_calibration_bundle(out)
    assert bundle["meta"]["rig_id"] == "test_rig"
    assert "cam0" in bundle["cameras"] and "cam1" in bundle["cameras"]


def test_extrinsics_mode_writes_bundle_after_capture(tmp_path: Path) -> None:
    bundle = tmp_path / "calibration_bundle.h5"
    _write_intrinsics_cache(bundle, ["cam0", "cam1"])
    cfg = {
        "rig_id": "test_rig",
        "schema_version": "1.0",
        "cameras": [
            {"id": "cam0", "camera_num": 0, "resolution": [640, 480]},
            {"id": "cam1", "camera_num": 1, "resolution": [640, 480]},
        ],
        "board": {
            "type": "checkerboard",
            "checkerboard_inner_corners": [9, 6],
            "square_size_mm": 25.0,
        },
        "output": {"bundle_path": str(bundle)},
        "fiducials": {},
        "notes": "",
    }
    T1 = np.eye(4, dtype=np.float64)
    T1[0, 3] = 80.0
    fit = ExtrinsicsFit(
        world_id="cam0",
        extrinsics={
            "cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.2, cross_view_rms_px=0.6),
            "cam1": CameraExtrinsics(T_world_cam=T1, rms_px=0.3, cross_view_rms_px=0.6),
        },
        n_snapshots=0,
    )

    class _Preview:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

    config_path = tmp_path / "cal.yaml"
    config_path.write_text("dummy: true\n", encoding="utf-8")

    with (
        patch("optical.calibration.multicam._load_json", return_value=cfg),
        patch("optical.calibration.multicam._build_board_detector", return_value=object()),
        patch("optical.calibration.multicam.PreviewServer", _Preview),
        patch(
            "optical.calibration.multicam.estimate_extrinsics_snapshots",
            return_value=(fit, {}),
        ),
    ):
        run_calibration(config_path=config_path, mode="extrinsics", backend="v4l2")

    assert calibration_cache_paths(bundle)[1].is_file()
    assert bundle.is_file()
    loaded = read_calibration_bundle(bundle)
    np.testing.assert_allclose(loaded["cameras"]["cam1"]["extrinsics"]["T_world_cam"][0, 3], 80.0)
    assert loaded["cameras"]["cam1"]["extrinsics"]["cross_view_rms_px"] == pytest.approx(0.6)
