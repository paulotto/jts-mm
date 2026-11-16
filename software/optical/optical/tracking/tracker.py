# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
ArUco jaw tracker: calibration loading, detection, pose estimation, frame records.

See ``docs/optical/03_pose_estimation_and_output.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from optical.calibration.io import (
    CameraExtrinsics,
    CameraIntrinsics,
    load_cameras_from_bundle,
    read_calibration_bundle,
)
from optical.markers.config import MarkerConfig, load_marker_config, validate_marker_config
from optical.tracking.aruco import (
    ArucoDetector,
    DetectionResult,
    detect_frames,
    plan_roi,
    shift_detection,
)
from optical.tracking.io import FrameRecord
from optical.tracking.pose import is_rigid_se3, is_valid_pose, nan_pose
from optical.tracking.pose_estimation import WorldBodyPose, estimate_world_body_poses_pnp
from optical.tracking.triangulation import TriangulatedPoint, estimate_world_body_poses_triangulation


PoseMethod = Literal["pnp", "triangulation"]
POSE_METHODS = ("pnp", "triangulation")


@dataclass
class TrackerConfig:
    """Runtime tracker settings (loaded from optical.yaml ``tracker:`` or legacy JSON)."""

    calibration_bundle_path: Path
    marker_config_path: Path
    pose_method: PoseMethod
    cam_ids: List[str]
    use_reprojection_weights: bool
    triangulation_min_views: int
    triangulation_max_residual_mm: float
    pnp_max_reprojection_rms_px: float
    # PnP flip resolution: trust the previous pose for this many frames after
    # the last unambiguous solve (0 disables the temporal prior).
    pnp_prior_max_frames: int = 15
    # Triangulation: drop a corner view whose reprojection error exceeds this.
    triangulation_max_corner_reproj_px: float = 2.0
    # Reusable CT tip offset (pivot calibration JSON); None = report the CT
    # body origin. Applied by the standalone writers (run / replay), like
    # jtsmm's OpticalTrackBridge does for /optical/pose_ct.
    ct_tip_calibration_path: Optional[Path] = None

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        active_cam_ids: list[str] | None = None,
        n_cameras: int | None = None,
    ) -> TrackerConfig:
        """Load from monolithic optical.yaml or legacy tracker JSON."""
        from optical.config.loader import calibration_bundle_from_doc, tracker_section
        from optical.config.resolve import load_resolved_config
        from optical.paths import resolve_repo_path

        path = Path(path)
        doc = load_resolved_config(
            path,
            active_cam_ids=active_cam_ids,
            n_cameras=n_cameras,
        )
        t = tracker_section(doc)

        bundle = t.get("calibration_bundle_path")
        if not bundle:
            resolved = calibration_bundle_from_doc(doc)
            if resolved is None:
                raise KeyError(f"{path}: missing paths.calibration_bundle / tracker.calibration_bundle_path")
            bundle = resolved
        bundle_path = Path(bundle)
        if not bundle_path.is_absolute():
            bundle_path = resolve_repo_path(bundle_path)

        # Markers live in the same optical.yaml; legacy JSON may still point elsewhere.
        marker_path = Path(t["marker_config_path"]) if t.get("marker_config_path") else path
        if not marker_path.is_absolute() and t.get("marker_config_path"):
            marker_path = resolve_repo_path(marker_path)

        pose_method = str(t.get("pose_method", "pnp")).strip().lower()
        if pose_method not in POSE_METHODS:
            # A typo used to fall through to PnP silently (estimate_poses only
            # tests for "triangulation").
            raise ValueError(
                f"{path}: tracker.pose_method must be one of {POSE_METHODS}, "
                f"got {t.get('pose_method')!r}"
            )
        tri = t.get("triangulation") or {}
        tip_path: Optional[Path] = None
        if t.get("ct_tip_calibration"):
            tip_path = Path(str(t["ct_tip_calibration"])).expanduser()
            if not tip_path.is_absolute():
                tip_path = resolve_repo_path(tip_path)
        cfg = cls(
            calibration_bundle_path=bundle_path,
            marker_config_path=marker_path,
            pose_method=pose_method,  # type: ignore[arg-type]
            cam_ids=list(t.get("cam_ids", [])),
            use_reprojection_weights=bool(t.get("use_reprojection_weights", False)),
            triangulation_min_views=int(tri.get("min_views", 2)),
            triangulation_max_residual_mm=float(tri.get("max_residual_mm", 5.0)),
            pnp_max_reprojection_rms_px=float(
                (t.get("pnp") or {}).get("max_reprojection_rms_px", 8.0)
            ),
            pnp_prior_max_frames=int((t.get("pnp") or {}).get("prior_max_frames", 15)),
            triangulation_max_corner_reproj_px=float(tri.get("max_corner_reprojection_px", 2.0)),
            ct_tip_calibration_path=tip_path,
        )
        errors = cfg.validate()
        if errors:
            raise ValueError(f"{path}: invalid tracker config:\n  " + "\n  ".join(errors))
        return cfg

    def validate(self) -> List[str]:
        """Return config errors (empty when valid)."""
        errors: List[str] = []
        if self.pose_method not in POSE_METHODS:
            errors.append(f"pose_method must be one of {POSE_METHODS}, got {self.pose_method!r}")
        if self.triangulation_min_views < 2:
            errors.append("triangulation.min_views must be >= 2")
        if not self.triangulation_max_residual_mm > 0:
            errors.append("triangulation.max_residual_mm must be > 0")
        if self.pnp_max_reprojection_rms_px < 0:
            errors.append("pnp.max_reprojection_rms_px must be >= 0 (0 disables the gate)")
        if self.triangulation_max_corner_reproj_px < 0:
            errors.append("triangulation.max_corner_reprojection_px must be >= 0 (0 = off)")
        if self.pnp_prior_max_frames < 0:
            errors.append("pnp.prior_max_frames must be >= 0")
        return errors

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        active_cam_ids: list[str] | None = None,
        n_cameras: int | None = None,
    ) -> TrackerConfig:
        """Alias for :meth:`from_path` (legacy name)."""
        return cls.from_path(
            path,
            active_cam_ids=active_cam_ids,
            n_cameras=n_cameras,
        )


def scale_intrinsics_for_capture(
    intr: CameraIntrinsics,
    image_size: tuple[int, int],
) -> CameraIntrinsics:
    """
    Rescale ``K`` from the calibration resolution to a capture resolution.

    Delegates to :func:`optical.calibration.io.scale_camera_intrinsics`:
    pixel-centre aligned principal point ``c' = (c + 0.5)·s − 0.5`` (plain
    ``c·s`` is off by 0.25 px at half-res lores) and ``ValueError`` when the
    aspect ratio changes by more than 0.5 % (a crop, not a scale).
    """
    from optical.calibration.io import scale_camera_intrinsics

    return scale_camera_intrinsics(intr, (int(image_size[0]), int(image_size[1])))


@dataclass
class LastEstimate:
    """Most recent :meth:`ArucoJawTracker.estimate_poses` result (for diagnostics)."""

    method: str
    body_poses: Dict[str, WorldBodyPose]
    triangulated: Optional[List[TriangulatedPoint]]
    fits: Dict[str, Any]


class ArucoJawTracker:
    """
    Convert per-camera ArUco detections into per-frame HP/MP/CT poses.

    Args:
        tracker_config: Paths and algorithm options.
        calibration_bundle: Optional pre-loaded bundle (else loaded from config path).
        marker_config: Optional pre-loaded marker config.
    """

    def __init__(
        self,
        tracker_config: TrackerConfig,
        *,
        calibration_bundle: Optional[Dict[str, Any]] = None,
        marker_config: Optional[MarkerConfig] = None,
    ) -> None:
        cfg_errors = tracker_config.validate()
        if cfg_errors:
            raise ValueError("Invalid tracker config:\n" + "\n".join(cfg_errors))
        self.config = tracker_config

        bundle = calibration_bundle or read_calibration_bundle(tracker_config.calibration_bundle_path)
        self._bundle_meta = bundle.get("meta", {})
        self.intrinsics, self.extrinsics = load_cameras_from_bundle(bundle)

        # Single-camera rig: world = cam (identity T_world_cam). Extrinsics are
        # only meaningful across cameras, so a lone camera needs none.
        # See docs/optical/03_pose_estimation_and_output.md §1.5.
        if len(self.intrinsics) == 1:
            for cam_id in self.intrinsics:
                if cam_id not in self.extrinsics:
                    self.extrinsics[cam_id] = CameraExtrinsics(
                        T_world_cam=np.eye(4, dtype=np.float64),
                        rms_px=float("nan"),
                    )

        self.marker_config = marker_config or load_marker_config(tracker_config.marker_config_path)
        errors = validate_marker_config(self.marker_config)
        if errors:
            raise ValueError("Invalid marker config:\n" + "\n".join(errors))

        self.detector = ArucoDetector(
            self.marker_config.dictionary,
            corner_refinement=self.marker_config.corner_refinement,
            corner_refine_win_px=self.marker_config.corner_refine_win_px,
        )
        self.last_detections: Dict[str, DetectionResult] = {}
        self._roi_frame: Dict[str, int] = {}
        self.last_estimate: Optional[LastEstimate] = None
        # PnP flip prior: last pose per body and frames since it was last
        # confirmed by an unambiguous view (see estimate_world_body_poses_pnp).
        self._prior_T: Dict[str, np.ndarray] = {}
        self._prior_age: Dict[str, int] = {}

        if tracker_config.cam_ids:
            self.cam_ids = list(tracker_config.cam_ids)
        else:
            self.cam_ids = sorted(set(self.intrinsics) & set(self.extrinsics))

        if not self.cam_ids:
            raise ValueError("No cameras with both intrinsics and extrinsics in calibration bundle")

    def scale_intrinsics_to(self, image_size: tuple[int, int]) -> None:
        """Scale all camera ``K`` matrices to a capture resolution (e.g. lores)."""
        dst = (int(image_size[0]), int(image_size[1]))
        self.intrinsics = {
            cam_id: scale_intrinsics_for_capture(intr, dst)
            for cam_id, intr in self.intrinsics.items()
        }

    def scale_intrinsics_to_frame(self, frame: np.ndarray) -> None:
        """Scale intrinsics to match a gray/RGB frame's width×height."""
        arr = np.asarray(frame)
        if arr.ndim < 2:
            raise ValueError("frame must be HxW or HxWxC")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        self.scale_intrinsics_to((w, h))

    @property
    def rig_id(self) -> str:
        return str(self._bundle_meta.get("rig_id", ""))

    def detect_all(
        self,
        frames: Dict[str, np.ndarray],
        *,
        parallel: bool = True,
    ) -> Dict[str, DetectionResult]:
        """Run ArUco detection on gray or RGB frames keyed by ``cam_id``.

        Stereo pairs detect in parallel (thread pool) so OpenCV can use two
        cores while the GIL is released inside ``detectMarkers``. Gray frames
        skip the RGB→gray convert.
        """
        subset = {cam_id: frames[cam_id] for cam_id in self.cam_ids if cam_id in frames}
        prepared: Dict[str, np.ndarray] = {}
        origins: Dict[str, tuple[int, int]] = {}
        for cam_id, img in subset.items():
            index = self._roi_frame.get(cam_id, 0)
            self._roi_frame[cam_id] = index + 1
            prev = self.last_detections.get(cam_id)
            window = plan_roi(
                prev.markers if prev is not None else None,
                np.asarray(img).shape,
                frame_index=index,
            )
            if window is None:
                prepared[cam_id] = img
                origins[cam_id] = (0, 0)
                continue
            x0, y0, x1, y1 = window
            prepared[cam_id] = np.ascontiguousarray(np.asarray(img)[y0:y1, x0:x1])
            origins[cam_id] = (x0, y0)
        out = detect_frames(self.detector, prepared, parallel=parallel)
        final: Dict[str, DetectionResult] = {}
        for cam_id, det in out.items():
            dx, dy = origins.get(cam_id, (0, 0))
            if dx or dy:
                if not det.markers:
                    # The marker left the window. Search the whole frame once.
                    det = detect_frames(
                        self.detector, {cam_id: subset[cam_id]}, parallel=False
                    )[cam_id]
                else:
                    det = shift_detection(det, dx, dy)
            final[cam_id] = det
        self.last_detections = final
        return final

    def reset_pose_prior(self) -> None:
        """Forget the PnP flip prior (call between unrelated takes)."""
        self._prior_T.clear()
        self._prior_age.clear()

    def pose_prior(self) -> Dict[str, np.ndarray]:
        """Recent poses usable as PnP flip prior."""
        max_age = int(self.config.pnp_prior_max_frames)
        return {
            body: T
            for body, T in self._prior_T.items()
            if self._prior_age.get(body, max_age) < max_age
        }

    def _update_pose_prior(self, poses: Dict[str, WorldBodyPose]) -> None:
        # Only fresh evidence (an unambiguous view, or cameras agreeing) resets
        # the age; poses chosen *by* the prior follow the motion but age, so a
        # wrong branch cannot keep itself alive past prior_max_frames.
        for body, pose in poses.items():
            ok = bool(pose.valid) and is_rigid_se3(pose.T_world_body)
            if ok and pose.resolved_by in ("view", "cameras"):
                self._prior_T[body] = np.asarray(pose.T_world_body, dtype=np.float64).copy()
                self._prior_age[body] = 0
                continue
            if ok and pose.resolved_by == "prior":
                self._prior_T[body] = np.asarray(pose.T_world_body, dtype=np.float64).copy()
            if body in self._prior_age:
                self._prior_age[body] += 1

    def estimate_poses(
        self,
        detections: Dict[str, DetectionResult],
        *,
        method: Optional[PoseMethod] = None,
        update_state: bool = True,
    ) -> tuple[Dict[str, WorldBodyPose], Optional[List[TriangulatedPoint]]]:
        """
        Estimate world-frame body poses from detections.

        ``method`` overrides ``pose_method`` (debug / compare). ``update_state``
        false leaves the PnP flip prior untouched (diagnostic re-runs).

        Returns:
            ``(body_poses, triangulated_points)``; triangulated list is non-``None``
            only when the method is ``triangulation``.
        """
        method = method or self.config.pose_method
        if method == "triangulation":
            fits: Dict[str, Any] = {}
            poses, triangulated = estimate_world_body_poses_triangulation(
                detections,
                intrinsics=self.intrinsics,
                extrinsics=self.extrinsics,
                marker_config=self.marker_config,
                min_views=self.config.triangulation_min_views,
                max_residual_mm=self.config.triangulation_max_residual_mm,
                max_corner_reproj_px=self.config.triangulation_max_corner_reproj_px,
                diagnostics=fits,
            )
            if update_state:
                self.last_estimate = LastEstimate(method, poses, triangulated, fits)
            return poses, triangulated

        poses = estimate_world_body_poses_pnp(
            detections,
            intrinsics=self.intrinsics,
            extrinsics=self.extrinsics,
            marker_config=self.marker_config,
            use_reprojection_weights=self.config.use_reprojection_weights,
            max_reprojection_rms_px=self.config.pnp_max_reprojection_rms_px,
            prior_T_world_body=self.pose_prior(),
        )
        if update_state:
            self._update_pose_prior(poses)
            self.last_estimate = LastEstimate(method, poses, None, {})
        return poses, None

    def process_frame(
        self,
        frames: Dict[str, np.ndarray],
        *,
        frame_idx: int,
        t_cam_ns: Optional[Dict[str, int]] = None,
        t_master_us: Optional[int] = None,
    ) -> FrameRecord:
        """
        Full per-frame pipeline: detect → pose → :class:`FrameRecord`.
        """
        if t_cam_ns is None:
            now_ns = time.monotonic_ns()
            t_cam_ns = {cam_id: now_ns for cam_id in frames}

        detections = self.detect_all(frames)
        body_poses, triangulated = self.estimate_poses(detections)

        record = FrameRecord(
            frame_idx=frame_idx,
            t_cam_ns=dict(t_cam_ns),
            t_master_us=t_master_us,
        )

        for body, pose in body_poses.items():
            ok = bool(pose.valid) and is_rigid_se3(pose.T_world_body)
            record.set_pose(body, pose.T_world_body if ok else nan_pose())
            record.quality[body] = {
                "n_tags": pose.n_tags,
                "n_cameras": pose.n_cameras,
                "reprojection_rms_px": pose.reprojection_rms_px,
                "fit_rms_mm": pose.fit_rms_mm,
                "valid": ok,
            }

        if triangulated is not None:
            record.markers_3d = [
                {
                    "marker_id": p.marker_id,
                    "corner_index": p.corner_index,
                    "position_mm": p.position_mm.tolist(),
                    "n_views": p.n_views,
                    "reprojection_rms_px": p.reprojection_rms_px,
                }
                for p in triangulated
            ]

        return record


def load_ct_tip_offset(
    tcfg: TrackerConfig,
) -> Optional[tuple[np.ndarray, Dict[str, str]]]:
    """``(tip_body_mm, meta_attrs)`` from ``tracker.ct_tip_calibration``, or ``None``.

    Raises ``FileNotFoundError`` when the path is configured but missing, so a
    take is not silently recorded at the CT body origin (same rule as jtsmm
    ``session.ct_tip_calibration``).
    """
    import json

    path = getattr(tcfg, "ct_tip_calibration_path", None)
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"tracker.ct_tip_calibration not found: {path} (run a CT pivot calibration "
            "once, e.g. jtsmm record with session.ct_calibration: true, or set it to null)"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    tip = np.asarray(data["tip_body_mm"], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(tip)):
        raise ValueError(f"{path}: tip_body_mm is not finite")
    meta = {
        "ct_tip_body_mm": ",".join(f"{x:.6f}" for x in tip.tolist()),
        "ct_tip_rms_mm": f"{float(data.get('rms_mm', float('nan'))):.6f}",
        "ct_tip_source": str(path),
    }
    return tip, meta


def apply_ct_tip(record: FrameRecord, tip_body_mm: Optional[np.ndarray]) -> None:
    """Move ``record.pose_CT`` to the calibrated tip (orientation unchanged)."""
    if tip_body_mm is None:
        return
    from optical.tracking.pivot import apply_tip_offset

    if is_valid_pose(record.pose_CT):
        record.pose_CT = apply_tip_offset(record.pose_CT, tip_body_mm)


def load_tracker_for_recording(
    config_path: Path | str,
    frames_dir: Path | str | None = None,
) -> tuple["ArucoJawTracker", TrackerConfig]:
    """Build a tracker for offline replay, preferring leaf-local snapshots.

    When ``frames_dir`` contains ``calibration_bundle.h5`` / ``optical.yaml``
    (written by live ``--persist-frames`` / ``--record-frames``), those are used
    instead of the live paths in the operator config.
    """
    from optical.paths import recording_calibration_bundle, recording_optical_config
    from optical.tracking.frame_store import iter_frames, load_session_meta

    frames_path = Path(frames_dir) if frames_dir is not None else None
    cfg_path = Path(config_path)
    meta: Dict[str, Any] = {}
    active_cam_ids: Optional[List[str]] = None
    if frames_path is not None:
        leaf_cfg = recording_optical_config(frames_path)
        if leaf_cfg is not None:
            cfg_path = leaf_cfg
        meta = load_session_meta(frames_path)
        # Profile (1 vs 2 cameras) follows the recording, not how many CSI
        # cameras the replay machine happens to have.
        if meta.get("cam_ids"):
            active_cam_ids = [str(c) for c in meta["cam_ids"]]

    tcfg = TrackerConfig.from_path(cfg_path, active_cam_ids=active_cam_ids)
    bundle = None
    if frames_path is not None:
        leaf_bundle = recording_calibration_bundle(frames_path)
        if leaf_bundle is not None:
            bundle = read_calibration_bundle(leaf_bundle)
            # Keep config path coherent for logging / downstream tooling.
            tcfg.calibration_bundle_path = leaf_bundle

    tracker = ArucoJawTracker(tcfg, calibration_bundle=bundle)

    if frames_path is not None:
        size = meta.get("image_size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            tracker.scale_intrinsics_to((int(size[0]), int(size[1])))
        else:
            for _idx, raw in iter_frames(frames_path):
                if raw:
                    tracker.scale_intrinsics_to_frame(next(iter(raw.values())))
                    break

    return tracker, tcfg
