# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""multicam run_calibration: gate-controlled bundle write, per-profile caches, intrinsics hash."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from optical.calibration import multicam  # noqa: E402
from optical.calibration.extrinsics import ExtrinsicsFit, calibration_cache_paths  # noqa: E402
from optical.calibration.io import CameraExtrinsics, read_calibration_bundle  # noqa: E402


class _Preview:
    url = "http://test/"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def start(self) -> None:
        pass

    def update(self, *_a, **_k) -> None:
        pass


def _cfg(bundle: Path, cam_ids: list[str]) -> dict:
    return {
        "rig_id": "test_rig",
        "schema_version": "1.0",
        "cameras": [{"id": c, "camera_num": i, "resolution": [640, 480]} for i, c in enumerate(cam_ids)],
        "board": {"type": "checkerboard", "checkerboard_inner_corners": [9, 6], "square_size_mm": 25.0},
        "output": {"bundle_path": str(bundle)},
        "fiducials": {},
    }


def _write_intr(bundle: Path, cam_ids: list[str], fx: float = 500.0) -> None:
    K = np.array([[fx, 0, 320.0], [0, fx, 240.0], [0, 0, 1]])
    np.savez(
        calibration_cache_paths(bundle)[0],
        **{f"{c}_K": K for c in cam_ids},
        **{f"{c}_dist": np.zeros(5) for c in cam_ids},
        **{f"{c}_image_size": np.array([640, 480], dtype=np.int32) for c in cam_ids},
        **{f"{c}_rms_px": np.asarray(0.3) for c in cam_ids},
    )


def _fit(problems: list[str], cross: float = 3.1, n: int = 2) -> ExtrinsicsFit:
    T1 = np.eye(4)
    T1[0, 3] = 450.0
    return ExtrinsicsFit(
        world_id="cam0",
        extrinsics={
            "cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.2, cross_view_rms_px=cross),
            "cam1": CameraExtrinsics(T_world_cam=T1, rms_px=0.3, cross_view_rms_px=cross),
        },
        n_snapshots=n,
        per_snapshot_cross_view_px={"cam1": np.full(n, cross)},
        inliers={"cam1": np.ones(n, dtype=bool)},
        paired={"cam1": np.ones(n, dtype=bool)},
        rot_dev_deg={"cam1": np.zeros(n)},
        trans_dev_mm={"cam1": np.zeros(n)},
        cross_view_rms_px={"cam1": cross},
        cross_view_rms_all_px={"cam1": cross},
        problems=problems,
    )


def _run(tmp_path: Path, cfg: dict, mode: str, fit: ExtrinsicsFit | None = None) -> None:
    cfg_path = tmp_path / "cal.yaml"
    cfg_path.write_text("dummy: true\n")
    with (
        patch("optical.calibration.multicam._load_json", return_value=cfg),
        patch("optical.calibration.multicam._build_board_detector", return_value=object()),
        patch("optical.calibration.multicam.PreviewServer", _Preview),
        patch("optical.calibration.multicam.estimate_extrinsics_snapshots", return_value=(fit, {})),
    ):
        multicam.run_calibration(config_path=cfg_path, mode=mode, backend="v4l2")


def test_failed_gates_keep_cache_and_skip_bundle_then_bundle_mode_overrides(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "calibration_bundle_2cams.h5"
    cfg = _cfg(bundle, ["cam0", "cam1"])
    _write_intr(bundle, ["cam0", "cam1"])
    with pytest.raises(SystemExit, match="Bundle NOT written"):
        _run(tmp_path, cfg, "extrinsics", _fit(
            ["cam0->cam1: cross-view RMS 5.10 px > 2.00 px"], cross=5.1,
        ))
    assert not bundle.exists()
    extr_cache = calibration_cache_paths(bundle)[1]
    assert extr_cache.is_file()
    data = np.load(extr_cache)
    assert not bool(data["gates_passed"])
    assert str(data["world_frame"]) == "cam0"

    _run(tmp_path, cfg, "bundle")
    out = capsys.readouterr().out
    assert "operator override" in out
    assert "5.10 px > 4.00 px" in out
    loaded = read_calibration_bundle(bundle)
    assert not bool(loaded["meta"]["extrinsics_gates_passed"])
    assert "5.10 px > 4.00 px" in str(loaded["meta"]["extrinsics_gate_problems"])
    q = loaded["cameras"]["cam1"]["extrinsics"]["quality"]
    assert np.allclose(q["cross_view_rms_px_per_snapshot"], [5.1, 5.1])


def test_bundle_rejudges_stored_cross_view_against_current_gate(tmp_path: Path, capsys) -> None:
    """A cache that failed at 2 px passes once the gate is 4 px, without recapture."""
    bundle = tmp_path / "calibration_bundle_2cams.h5"
    cfg = _cfg(bundle, ["cam0", "cam1"])
    _write_intr(bundle, ["cam0", "cam1"])
    with pytest.raises(SystemExit, match="Bundle NOT written"):
        _run(tmp_path, cfg, "extrinsics", _fit(
            ["cam0->cam1: cross-view RMS 3.10 px > 2.00 px"], n=5,
        ))
    capsys.readouterr()
    _run(tmp_path, cfg, "bundle")
    out = capsys.readouterr().out
    assert "operator override" not in out
    assert "2.00 px" not in out
    loaded = read_calibration_bundle(bundle)
    assert bool(loaded["meta"]["extrinsics_gates_passed"])
    assert str(loaded["meta"]["extrinsics_gate_problems"]) == ""


def test_bundle_refuses_extrinsics_from_other_intrinsics(tmp_path: Path) -> None:
    bundle = tmp_path / "calibration_bundle_2cams.h5"
    cfg = _cfg(bundle, ["cam0", "cam1"])
    _write_intr(bundle, ["cam0", "cam1"])
    _run(tmp_path, cfg, "extrinsics", _fit([]))
    assert bundle.is_file()
    _write_intr(bundle, ["cam0", "cam1"], fx=510.0)  # intrinsics redone
    with pytest.raises(SystemExit, match="different intrinsics"):
        _run(tmp_path, cfg, "bundle")


def test_one_and_two_camera_profiles_do_not_share_caches(tmp_path: Path) -> None:
    b1 = tmp_path / "calibration_bundle.h5"
    b2 = tmp_path / "calibration_bundle_2cams.h5"
    _write_intr(b2, ["cam0", "cam1"])
    _run(tmp_path, _cfg(b2, ["cam0", "cam1"]), "extrinsics", _fit([]))
    # The 1-cam profile has no caches of its own: it must not pick up the 2-cam ones.
    with pytest.raises(SystemExit, match="Missing intrinsics cache"):
        _run(tmp_path, _cfg(b1, ["cam0"]), "bundle")
    _write_intr(b1, ["cam0"], fx=480.0)
    _run(tmp_path, _cfg(b1, ["cam0"]), "bundle")
    one = read_calibration_bundle(b1)
    assert list(one["cameras"]) == ["cam0"]
    assert one["cameras"]["cam0"]["intrinsics"]["K"][0, 0] == 480.0
    assert read_calibration_bundle(b2)["cameras"]["cam0"]["intrinsics"]["K"][0, 0] == 500.0


def test_legacy_cache_name_gets_a_hint(tmp_path: Path) -> None:
    bundle = tmp_path / "calibration_bundle.h5"
    (tmp_path / "intrinsics_cache.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="legacy intrinsics_cache.npz"):
        _run(tmp_path, _cfg(bundle, ["cam0"]), "bundle")


def test_charuco_detector_rejects_collinear_corners() -> None:
    sq = 38.52
    all_obj = np.array([[sq * (i + 1), sq * (j + 1), 0.0] for j in range(4) for i in range(6)], np.float32)

    class _FakeCharuco:
        def __init__(self, ids):
            self.ids = np.asarray(ids).reshape(-1, 1)

        def detectBoard(self, _gray):
            corners = np.random.default_rng(0).uniform(0, 600, (len(self.ids), 1, 2)).astype(np.float32)
            return corners, self.ids, None, None

    det = object.__new__(multicam.CharucoBoardDetector)
    det._all_obj = all_obj
    det._min_corners = 6
    det._detector = _FakeCharuco(range(6))  # one row of 6 corners
    assert det.detect(np.zeros((10, 10), np.uint8)) is None
    det._detector = _FakeCharuco(range(12))  # two rows
    assert det.detect(np.zeros((10, 10), np.uint8)) is not None


def test_require_calibrated_size() -> None:
    from optical.calibration.io import CameraIntrinsics

    intr = CameraIntrinsics(K=np.eye(3), dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.1)
    multicam._require_calibrated_size(np.zeros((1088, 1456, 3), np.uint8), intr, "cam0")
    with pytest.raises(SystemExit, match="728x544"):
        multicam._require_calibrated_size(np.zeros((544, 728), np.uint8), intr, "cam0")


def test_collinear_view_breaks_real_pnp() -> None:
    """Why the detector gate exists: PnP on one row returns ok=True with a garbage pose."""
    sq = 38.52
    K = np.array([[1740.0, 0, 728], [0, 1740.0, 544], [0, 0, 1]])
    row = np.array([[sq * (i + 1), sq, 0] for i in range(6)], np.float32)
    rv, tv = np.array([0.2, -0.3, 0.1]), np.array([-100.0, -50.0, 600.0])
    img, _ = cv2.projectPoints(row, rv, tv, K, np.zeros(5))
    img = (img + np.random.default_rng(1).normal(0, 0.2, img.shape)).astype(np.float32)
    ok, r, t = cv2.solvePnP(row, img, K, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
    err_mm = float(np.linalg.norm(np.asarray(t).ravel() - tv))
    assert (not ok) or err_mm > 20.0
    assert not multicam.board_points_well_spread(row)


def test_estimate_extrinsics_snapshots_end_to_end(capsys) -> None:
    """Drive the interactive capture loop with a fake rig, keys and detector."""
    from optical.calibration.geometry import invert_T, make_T, rodrigues
    from optical.calibration.io import CameraIntrinsics

    K = np.array([[1740.0, 0, 728.0], [0, 1740.0, 544.0], [0, 0, 1]])
    intr = {
        c: CameraIntrinsics(K=K, dist=np.zeros(5), image_size=(1456, 1088), rms_px=0.2)
        for c in ("cam0", "cam1")
    }
    T_w_c1 = make_T(rodrigues([0.0, 0.35, 0.0]), [450.0, 0.0, 40.0])
    sq = 38.52
    obj = np.array([[sq * (i + 1), sq * (j + 1), 0] for j in range(4) for i in range(6)], np.float32)
    poses = [
        make_T(rodrigues(np.array([np.pi, 0.0, 0.0]) + d), t)
        for d, t in [
            ([0.1, 0.0, 0.0], [150.0, -60.0, 650.0]),
            ([0.0, 0.15, 0.05], [120.0, -40.0, 700.0]),
            ([-0.1, 0.05, 0.0], [180.0, -80.0, 600.0]),
            ([0.05, -0.1, 0.1], [140.0, -50.0, 750.0]),
        ]
    ]
    state = {"k": 0}

    def frames_now():
        # Pixel value encodes the pose index; cam id is the dict key.
        return {c: np.full((1088, 1456, 3), state["k"], np.uint8) for c in ("cam0", "cam1")}

    class Detector:
        """Returns exact projections of the board for the frame's pose index."""

        def __init__(self) -> None:
            self.calls = 0

        def detect(self, gray):
            cam = ("cam0", "cam1")[self.calls % 2]  # loop detects cam0 then cam1
            self.calls += 1
            T0 = poses[int(gray[0, 0]) % len(poses)]
            T = T0 if cam == "cam0" else invert_T(T_w_c1) @ T0
            rv, _ = cv2.Rodrigues(T[:3, :3])
            img, _ = cv2.projectPoints(obj, rv, T[:3, 3], K, np.zeros(5))
            return multicam.BoardDetection(
                obj_points=obj, img_points=img.astype(np.float32), preview_points=img.reshape(-1, 2)
            )

    script = iter([None, ord(" "), ord(" "), ord(" "), ord(" "), ord("q")])

    class Keys:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            pass

        def poll(self):
            key = next(script, ord("q"))
            if key == ord(" "):
                state["k"] += 1  # next SPACE sees the next pose
            return key

    class Rig:
        def close(self):
            pass

    cams = [multicam.CamSpec("cam0", 0), multicam.CamSpec("cam1", 1)]
    with (
        patch.object(multicam, "_open_rig", return_value=Rig()),
        patch.object(multicam, "_TerminalKeys", Keys),
        patch.object(multicam.CameraStream, "start", lambda self: None),
        patch.object(multicam.CameraStream, "latest", lambda self: frames_now()),
    ):
        fit, stats = multicam.estimate_extrinsics_snapshots(
            cams=cams, intrinsics=intr, detector=Detector(), min_snapshots=3,
            backend="v4l2", preview=_Preview(), expected_baseline_mm=452.0,
        )
    out = capsys.readouterr().out
    assert fit.gates_passed, fit.problems
    assert fit.n_snapshots == 4
    np.testing.assert_allclose(fit.extrinsics["cam1"].T_world_cam[:3, 3], T_w_c1[:3, 3], atol=0.5)
    assert fit.extrinsics["cam1"].cross_view_rms_px < 0.1
    assert len(stats["baseline_mm_per_snapshot"]) == 4
    assert "cross-view rms" in out
