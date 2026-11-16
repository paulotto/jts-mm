# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""/optical/corners: the 2D tag corners behind each /optical pose, stored so the
poses can be re-triangulated offline (no cameras needed to test any of this)."""

from __future__ import annotations

import queue
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="needs OpenCV")
pytest.importorskip("optical", reason="needs the in-repo optical package")

from jtsmm.capture.optical_bridge import OpticalTrackBridge  # noqa: E402
from jtsmm.io.hdf5_writer import Hdf5Writer  # noqa: E402
from jtsmm.io.optical_corners import read_optical_corners  # noqa: E402

EYE = np.eye(4, dtype=np.float32)


def _writer(path: Path) -> Hdf5Writer:
    w = Hdf5Writer(path=path, config_hash="x", mcu_config={}, host_config={})
    w.open()
    return w


def _sq(x: float, y: float, s: float = 10.0) -> np.ndarray:
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)


def test_rows_point_at_their_optical_row_across_flushes(tmp_path: Path):
    path = tmp_path / "s.h5"
    w = _writer(path)
    w.set_optical_calibration({
        "cam0": {"K": np.eye(3), "dist": np.zeros(5), "image_size": (728, 544),
                 "T_world_cam": np.eye(4), "rms_px": 0.3, "extrinsics_rms_px": None},
    })
    w.append_optical(t_master_us=100, pose_mta=EYE, pose_cra=EYE, pose_ct=EYE,
                     corners=[("cam0", 13, _sq(1, 2)), ("cam1", 1, _sq(3, 4))])
    w.append_optical(t_master_us=200, pose_mta=EYE, pose_cra=EYE, pose_ct=EYE)  # nothing seen
    w.flush()
    w.append_optical(t_master_us=300, pose_mta=EYE, pose_cra=EYE, pose_ct=EYE,
                     corners=[("cam1", 8, _sq(5, 6)), ("cam0", 17, _sq(7, 8))])
    w.close()

    with h5py.File(path, "r") as f:
        assert f["optical/t_master_us"][()].tolist() == [100, 200, 300]
        cg = f["optical/corners"]
        assert cg["corners_px"].dtype == np.float32 and cg["corners_px"].shape == (4, 4, 2)
        assert "OpenCV ArUco order" in cg.attrs["note"]

    c = read_optical_corners(path)
    assert c is not None and c.n == 4
    assert c.cam_ids == ["cam0", "cam1"]  # calibration first, then first appearance
    assert c.optical_row.tolist() == [0, 0, 2, 2]  # row 1 had no tags; 2 is after a flush
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["optical/t_master_us"][()][c.optical_row], c.t_master_us)
    assert c.tag_id.tolist() == [13, 1, 8, 17]
    assert [c.cam_ids[i] for i in c.cam] == ["cam0", "cam1", "cam1", "cam0"]
    row2 = c.for_row(2)
    assert set(row2) == {"cam0", "cam1"}
    assert np.array_equal(row2["cam0"][17], _sq(7, 8))
    assert c.for_row(1) == {}
    cam0 = c.calibration["cam0"]
    assert cam0.image_size == (728, 544) and cam0.rms_px == pytest.approx(0.3)
    assert cam0.extrinsics_rms_px is None


def test_sessions_without_corners_read_as_none(tmp_path: Path):
    path = tmp_path / "old.h5"
    w = _writer(path)
    w.append_optical(t_master_us=1, pose_mta=EYE, pose_cra=EYE, pose_ct=EYE)
    w.close()
    assert read_optical_corners(path) is None


def _look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """T_world_cam for a camera at ``cam_pos`` looking at ``target`` (OpenCV axes)."""
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = cam_pos
    return T


def test_stored_corners_and_camera_model_re_triangulate_the_tag(tmp_path: Path):
    """End to end: project a known tag through a distorted stereo rig, store the
    pixels and the camera model, read both back and triangulate the 3D corners."""
    K = np.array([[700.0, 0, 364.0], [0, 700.0, 272.0], [0, 0, 1]])
    dist = np.array([-0.21, 0.08, 0.001, -0.0005, 0.0])
    target = np.array([10.0, -5.0, 480.0])
    rig = {"cam0": np.eye(4), "cam1": _look_at(np.array([120.0, 0.0, 15.0]), target)}
    tag = target + np.array([[-15, -15, 0], [15, -15, 0], [15, 15, 0], [-15, 15, 0.0]])

    corners = []
    for cam_id, T in rig.items():
        T_cam_world = np.linalg.inv(T)
        rvec, _ = cv2.Rodrigues(T_cam_world[:3, :3])
        px, _ = cv2.projectPoints(tag, rvec, T_cam_world[:3, 3], K, dist)
        corners.append((cam_id, 13, px.reshape(4, 2)))

    path = tmp_path / "s.h5"
    w = _writer(path)
    w.set_optical_calibration({
        cam_id: {"K": K, "dist": dist, "image_size": (728, 544), "T_world_cam": T}
        for cam_id, T in rig.items()
    })
    w.append_optical(t_master_us=5, pose_mta=EYE, pose_cra=EYE, pose_ct=EYE, corners=corners)
    w.close()

    c = read_optical_corners(path)
    row = c.for_row(0)
    P, pts = [], []
    for cam_id in ("cam0", "cam1"):
        cam = c.calibration[cam_id]
        T_cam_world = np.linalg.inv(cam.T_world_cam)
        P.append(T_cam_world[:3, :])  # normalized image coordinates after undistortion
        und = cv2.undistortPoints(row[cam_id][13].reshape(-1, 1, 2).astype(np.float64),
                                  cam.K, cam.dist)
        pts.append(und.reshape(-1, 2).T)
    X = cv2.triangulatePoints(P[0], P[1], pts[0], pts[1])
    X = (X[:3] / X[3]).T
    np.testing.assert_allclose(X, tag, atol=0.01)  # mm; float32 pixels cost ~1e-4 px


class _FakeTracker:
    """What the bridge worker uses of ArucoJawTracker."""

    def __init__(self, detections_per_frame: list[dict], fail_on: int | None = None):
        from optical.calibration.io import CameraExtrinsics, CameraIntrinsics

        self._dets = detections_per_frame
        self._fail_on = fail_on
        self.last_detections: dict = {}
        self.cam_ids = ["cam0", "cam1"]
        self.intrinsics = {
            cid: CameraIntrinsics(K=np.diag([350.0, 350.0, 1.0]), dist=np.zeros(5),
                                  image_size=(728, 544), rms_px=0.4)
            for cid in self.cam_ids
        }
        self.extrinsics = {
            "cam0": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=float("nan")),
            "cam1": CameraExtrinsics(T_world_cam=np.eye(4), rms_px=0.6),
        }

    def process_frame(self, frames, *, frame_idx, t_cam_ns, t_master_us):
        if frame_idx == self._fail_on:
            raise RuntimeError("detector exploded")
        self.last_detections = self._dets[frame_idx]
        return SimpleNamespace(pose_MP=np.eye(4), pose_HP=np.eye(4) * 2, pose_CT=np.eye(4),
                               quality={})


def _bridge(tracker: _FakeTracker) -> OpticalTrackBridge:
    b = OpticalTrackBridge.__new__(OpticalTrackBridge)
    b.tracker = tracker
    b.stats = Counter()
    b._jobs = queue.Queue()
    b._results = queue.Queue()
    b._calibration_sent = False
    b._ct_tip_body_mm = None
    b._pose_method = "triangulation"
    b._overlay_lock = threading.Lock()
    b._overlay_markers = {}
    b._overlay_src_wh = {}
    return b


def _det(*tags: tuple[int, np.ndarray]):
    from optical.tracking.aruco import DetectionResult, MarkerDetection

    return DetectionResult(markers=tuple(MarkerDetection(i, c.astype(np.float64)) for i, c in tags))


def test_bridge_stores_each_frames_own_corners_and_the_camera_model(tmp_path: Path):
    frames = {"cam0": np.zeros((544, 728), np.uint8), "cam1": np.zeros((544, 728), np.uint8)}
    per_frame = [
        {"cam1": _det((8, _sq(5, 6))), "cam0": _det((13, _sq(1, 2)), (17, _sq(3, 4)))},
        {},  # frame 1 raises: no corners, NaN poses, the row still exists
        {"cam0": _det((13, _sq(9, 9)))},
    ]
    bridge = _bridge(_FakeTracker(per_frame, fail_on=1))
    for i in range(3):
        bridge._jobs.put((i, 1000 + i, frames, {"cam0": 0, "cam1": 0}))
    bridge._jobs.put(None)
    bridge._worker_main()  # synchronous: processes the three jobs, stops at None

    path = tmp_path / "s.h5"
    w = _writer(path)
    bridge.drain_to_writer(w)
    bridge.drain_to_writer(w)  # calibration only once, no rows twice
    w.close()

    c = read_optical_corners(path)
    assert c.optical_row.tolist() == [0, 0, 0, 2]
    assert c.t_master_us.tolist() == [1000, 1000, 1000, 1002]
    assert [(c.cam_ids[k], int(t)) for k, t in zip(c.cam, c.tag_id)] == [
        ("cam0", 13), ("cam0", 17), ("cam1", 8), ("cam0", 13)]  # sorted by camera per frame
    assert np.array_equal(c.for_row(2)["cam0"][13], _sq(9, 9))
    assert set(c.calibration) == {"cam0", "cam1"}
    assert c.calibration["cam1"].extrinsics_rms_px == pytest.approx(0.6)
    assert c.calibration["cam0"].image_size == (728, 544)
    with h5py.File(path, "r") as f:
        assert f["optical/t_master_us"][()].tolist() == [1000, 1001, 1002]
        assert np.isnan(f["optical/pose_mta"][1]).all()
    assert bridge.stats["failed"] == 1


def test_null_sink_and_writers_without_calibration_support_still_work():
    from jtsmm.capture.supervisor import _NullSink

    bridge = _bridge(_FakeTracker([{"cam0": _det((13, _sq(1, 1)))}]))
    bridge._jobs.put((0, 7, {"cam0": np.zeros((4, 4), np.uint8)}, {"cam0": 0}))
    bridge._jobs.put(None)
    bridge._worker_main()
    bridge.drain_to_writer(_NullSink())  # no set_optical_calibration, ignores corners
    assert bridge._results.empty()


def _tag_frame(tag_id: int, x0: int, y0: int, side: int, size=(728, 544)) -> np.ndarray:
    img = np.full((size[1], size[0]), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(d, tag_id, side)
    return img


def test_real_detection_corners_are_full_image_pixels_also_from_the_roi_search():
    """With the real detector: frame 0 scans the whole image, frame 1 only a
    window around the last hit; both must store the same full-image corners."""
    from optical.tracking.aruco import ArucoDetector
    from optical.tracking.tracker import ArucoJawTracker

    tracker = ArucoJawTracker.__new__(ArucoJawTracker)
    tracker.detector = ArucoDetector("DICT_4X4_50")
    tracker.cam_ids = ["cam0"]
    tracker.last_detections = {}
    tracker._roi_frame = {}
    bridge = _bridge(_FakeTracker([]))
    bridge.tracker = tracker

    x0, y0, side = 400, 300, 60
    frame = {"cam0": _tag_frame(13, x0, y0, side)}
    tracker.detect_all(frame)
    full = bridge._detected_corners()
    # Frame 1 really searches a crop (not the whole frame) and finds the tag in it.
    from optical.tracking.aruco import detect_frames, plan_roi

    win = plan_roi(tracker.last_detections["cam0"].markers, frame["cam0"].shape, frame_index=1)
    assert win is not None and (win[0], win[1]) != (0, 0)
    crop = np.ascontiguousarray(frame["cam0"][win[1]:win[3], win[0]:win[2]])
    assert detect_frames(tracker.detector, {"cam0": crop}, parallel=False)["cam0"].markers
    tracker.detect_all(frame)  # frame 1: region-of-interest crop
    roi = bridge._detected_corners()

    assert [(cam, tag) for cam, tag, _ in full] == [("cam0", 13)]
    expected = np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                        dtype=np.float64)
    np.testing.assert_allclose(full[0][2], expected, atol=1.0)  # TL, TR, BR, BL
    np.testing.assert_allclose(roi[0][2], full[0][2], atol=0.05)
