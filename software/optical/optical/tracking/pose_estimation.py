# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Per-camera PnP pose estimation and rigid-body fusion (Approach A).

One PnP per body and camera over the body-frame corners of *all* visible tags
(not one PnP per tag). A single small tag seen near-frontally has two IPPE
solutions with almost the same reprojection error (the planar "flip"); picking
the lower-error one is a coin toss that shows up as 20–60° rotation spikes.
Stacking every tag of the body into one solve removes most of that ambiguity,
and the rest is resolved against other cameras or the previous frame.

See ``docs/optical/03_pose_estimation_and_output.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from optical.calibration.io import CameraExtrinsics, CameraIntrinsics
from optical.markers.config import MarkerConfig
from optical.tracking.aruco import DetectionResult, MarkerDetection
from optical.tracking.pose import (
    compose_T,
    fuse_transforms,
    invert_T,
    is_valid_pose,
    nan_pose,
    rotation_angle_deg,
    tag_object_points,
    T_from_rvec_tvec,
    weighted_fuse_transforms,
)


BODY_NAMES = ("HP", "MP", "CT")

# Two solutions whose RMS differ by less than this are both plausible; the
# measured flip gap at 0.5 px corner noise is < 0.5 px.
AMBIGUITY_PX = 1.0
# Solutions closer than this in rotation are the same pose (LM converged both).
SAME_POSE_DEG = 5.0
# Two cameras "agree" within this angle. A near-frontal single tag is only good
# to ~10–15° (noise), while the wrong flip is off by twice the viewing angle.
CAMERA_AGREE_DEG = 20.0
# Relative out-of-plane extent below which the object points count as planar.
_PLANAR_REL_TOL = 1e-3


@dataclass
class PoseCandidate:
    """One refined PnP solution of a body in one camera."""

    T_cam_body: np.ndarray
    reprojection_rms_px: float
    per_tag_rms_px: Dict[int, float] = field(default_factory=dict)


@dataclass
class BodyPoseEstimate:
    """Body pose for one camera after candidate selection."""

    body: str
    T_cam_body: np.ndarray
    n_tags: int
    reprojection_rms_px: float
    tag_ids: List[int] = field(default_factory=list)
    candidates: List[PoseCandidate] = field(default_factory=list)
    ambiguous: bool = False


@dataclass
class WorldBodyPose:
    """World-frame body pose after camera fusion."""

    body: str
    T_world_body: np.ndarray
    n_tags: int
    n_cameras: int
    reprojection_rms_px: float
    valid: bool
    # Kabsch residual (triangulation only); NaN for PnP.
    fit_rms_mm: float = float("nan")
    # True when every contributing view had two plausible solutions and no
    # other camera / prior decided between them.
    ambiguous: bool = False
    # How the flip was resolved: view | cameras | prior | none (see below).
    resolved_by: str = ""


def body_for_marker_id(marker_id: int, id_ranges: Dict[str, Tuple[int, int]]) -> Optional[str]:
    """Map a detected tag ID to rigid body name (``HP``, ``MP``, ``CT``)."""
    for body, (lo, hi) in id_ranges.items():
        if lo <= marker_id <= hi:
            return body
    return None


def tag_to_body_pose(
    T_cam_tag: np.ndarray,
    T_body_tag: np.ndarray,
) -> np.ndarray:
    """
    Convert tag pose to body pose: ``T_cam←body = T_cam←tag @ T_tag←body``.

    ``T_body_tag`` is ``T_body←tag`` from marker config; invert for composition.
    """
    T_tag_body = invert_T(T_body_tag)
    return compose_T(T_cam_tag, T_tag_body)


def tag_corners_in_body(marker_id: int, marker_config: MarkerConfig) -> Optional[np.ndarray]:
    """(4, 3) tag corners in the body frame, or ``None`` without a mount."""
    T_body_tag = marker_config.mounts_T_body_tag.get(marker_id)
    if T_body_tag is None:
        return None
    obj = tag_object_points(marker_config.tag_size_mm)
    T = np.asarray(T_body_tag, dtype=np.float64)
    return obj @ T[:3, :3].T + T[:3, 3]


def _is_planar(pts: np.ndarray) -> bool:
    centered = pts - pts.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    return s[0] > 0 and s[-1] <= _PLANAR_REL_TOL * s[0]


def _reprojection(obj: np.ndarray, img: np.ndarray, rvec, tvec, K, dist) -> np.ndarray:
    import cv2

    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return np.linalg.norm(img - proj.reshape(-1, 2), axis=1)


def body_pose_candidates(
    obj_by_tag: Mapping[int, np.ndarray],
    img_by_tag: Mapping[int, np.ndarray],
    *,
    K: np.ndarray,
    dist: np.ndarray,
) -> List[PoseCandidate]:
    """
    All refined PnP solutions for one body in one camera, best RMS first.

    Planar tag sets (every configured body) go through IPPE, which returns
    both planar solutions; each is refined with Levenberg–Marquardt on all
    corners. Non-planar sets use SQPnP (global) + LM.
    """
    import cv2

    tag_ids = list(obj_by_tag)
    if not tag_ids:
        return []
    obj = np.concatenate([np.asarray(obj_by_tag[m], np.float64) for m in tag_ids], axis=0)
    img = np.concatenate([np.asarray(img_by_tag[m], np.float64).reshape(-1, 2) for m in tag_ids])
    if obj.shape[0] < 4:
        return []
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64)
    img_cv = img.reshape(-1, 1, 2)

    flags = cv2.SOLVEPNP_IPPE if _is_planar(obj) else cv2.SOLVEPNP_SQPNP
    try:
        n, rvecs, tvecs, _errs = cv2.solvePnPGeneric(obj, img_cv, K, dist, flags=flags)
    except cv2.error:
        return []
    if not n:
        return []

    out: List[PoseCandidate] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img_cv, K, dist, rvec, tvec)
        except cv2.error:
            pass
        if float(tvec[2, 0]) <= 0.0:
            continue  # body behind the camera
        err = _reprojection(obj, img, rvec, tvec, K, dist)
        per_tag: Dict[int, float] = {}
        i = 0
        for m in tag_ids:
            k = int(np.asarray(obj_by_tag[m]).shape[0])
            per_tag[int(m)] = float(np.sqrt(np.mean(err[i : i + k] ** 2)))
            i += k
        out.append(
            PoseCandidate(
                T_cam_body=T_from_rvec_tvec(rvec, tvec),
                reprojection_rms_px=float(np.sqrt(np.mean(err**2))),
                per_tag_rms_px=per_tag,
            )
        )
    out.sort(key=lambda c: c.reprojection_rms_px)
    return out


def _plausible(cands: Sequence[PoseCandidate], max_rms: float) -> List[PoseCandidate]:
    if not cands:
        return []
    best = cands[0].reprojection_rms_px
    return [
        c
        for c in cands
        if c.reprojection_rms_px <= best + AMBIGUITY_PX
        and (max_rms <= 0.0 or c.reprojection_rms_px <= max_rms)
    ]


def _is_ambiguous(cands: Sequence[PoseCandidate]) -> bool:
    if len(cands) < 2:
        return False
    R0 = cands[0].T_cam_body[:3, :3]
    return any(rotation_angle_deg(R0, c.T_cam_body[:3, :3]) > SAME_POSE_DEG for c in cands[1:])


def estimate_body_pose_in_camera(
    body: str,
    markers: Sequence[MarkerDetection],
    *,
    intrinsics: CameraIntrinsics,
    marker_config: MarkerConfig,
    max_reprojection_rms_px: float | None = None,
) -> Optional[BodyPoseEstimate]:
    """
    One PnP over every mounted tag of ``body`` seen by this camera.

    The gate ``max_reprojection_rms_px`` applies per tag under the joint
    solution: the worst tag above it is dropped and the body re-solved (one bad
    corner set cannot drag the others); a lone tag above it contributes nothing.

    ``T_cam_body`` is the lowest-RMS solution; ``candidates`` keeps every
    plausible alternative for :func:`estimate_world_body_poses_pnp` to resolve.
    """
    max_rms = float(max_reprojection_rms_px) if max_reprojection_rms_px else 0.0
    obj_by_tag: Dict[int, np.ndarray] = {}
    img_by_tag: Dict[int, np.ndarray] = {}
    for m in markers:
        if body_for_marker_id(m.marker_id, marker_config.id_ranges) != body:
            continue
        pts = tag_corners_in_body(m.marker_id, marker_config)
        if pts is None or m.marker_id in obj_by_tag:
            continue  # unmounted, or a duplicate detection of the same ID
        obj_by_tag[m.marker_id] = pts
        img_by_tag[m.marker_id] = np.asarray(m.corners, dtype=np.float64).reshape(4, 2)

    while obj_by_tag:
        cands = body_pose_candidates(obj_by_tag, img_by_tag, K=intrinsics.K, dist=intrinsics.dist)
        if not cands:
            return None
        best = cands[0]
        worst = max(best.per_tag_rms_px, key=best.per_tag_rms_px.get)
        if max_rms > 0.0 and best.per_tag_rms_px[worst] > max_rms:
            if len(obj_by_tag) == 1:
                return None
            del obj_by_tag[worst]
            del img_by_tag[worst]
            continue
        plausible = _plausible(cands, max_rms)
        return BodyPoseEstimate(
            body=body,
            T_cam_body=best.T_cam_body,
            n_tags=len(obj_by_tag),
            reprojection_rms_px=best.reprojection_rms_px,
            tag_ids=sorted(obj_by_tag),
            candidates=plausible,
            ambiguous=_is_ambiguous(plausible),
        )
    return None


def cam_to_world_pose(T_cam_body: np.ndarray, extrinsics: CameraExtrinsics) -> np.ndarray:
    """``T_world←body = T_world←cam @ T_cam←body``."""
    return compose_T(extrinsics.T_world_cam, T_cam_body)


def _closest(cands_world: Sequence[Tuple[np.ndarray, PoseCandidate]], R_ref: np.ndarray):
    return min(cands_world, key=lambda tc: rotation_angle_deg(R_ref, tc[0][:3, :3]))


def _resolve_reference(
    per_cam: Sequence[Tuple[str, BodyPoseEstimate]],
    world_cands: Mapping[str, Sequence[Tuple[np.ndarray, PoseCandidate]]],
    prior: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], str]:
    """Pick the world rotation every camera's candidate is chosen against."""
    anchors = [(cid, est) for cid, est in per_cam if not est.ambiguous]
    if anchors:
        cid, _est = max(anchors, key=lambda ce: (ce[1].n_tags, -ce[1].reprojection_rms_px))
        return world_cands[cid][0][0][:3, :3], "view"
    if len(per_cam) >= 2:
        # Every view is ambiguous: take the candidate the other cameras agree
        # with best (wrong flips of different views point in different directions).
        best_score, best_R = float("inf"), None
        for cid, cands in world_cands.items():
            for T_w, _c in cands:
                score = sum(
                    min(rotation_angle_deg(T_w[:3, :3], T_o[:3, :3]) for T_o, _ in oc)
                    for ocid, oc in world_cands.items()
                    if ocid != cid
                )
                if score < best_score:
                    best_score, best_R = score, T_w[:3, :3]
        if best_R is not None and best_score <= CAMERA_AGREE_DEG * (len(per_cam) - 1):
            return best_R, "cameras"
    if prior is not None and is_valid_pose(prior):
        return np.asarray(prior, dtype=np.float64)[:3, :3], "prior"
    return None, "none"


def estimate_world_body_poses_pnp(
    detections: Dict[str, DetectionResult],
    *,
    intrinsics: Dict[str, CameraIntrinsics],
    extrinsics: Dict[str, CameraExtrinsics],
    marker_config: MarkerConfig,
    use_reprojection_weights: bool = False,
    max_reprojection_rms_px: float | None = None,
    prior_T_world_body: Optional[Mapping[str, np.ndarray]] = None,
    bodies: Optional[Sequence[str]] = None,
) -> Dict[str, WorldBodyPose]:
    """
    Approach A: one PnP per body and camera, resolve flips, then fuse cameras.

    Flip resolution for views with two plausible solutions, in order:

    1. a camera whose own solve is unambiguous (most tags first) — ``view``,
    2. the candidate the other cameras agree with — ``cameras``,
    3. ``prior_T_world_body[body]`` (the caller's recent pose) — ``prior``,
    4. the lowest-RMS solution, flagged ``ambiguous`` — ``none``.

    Fresh evidence (1, 2) outranks the prior so a wrong prior cannot lock in.

    Returns a dict keyed by body name. The default set is ``HP``, ``MP``, ``CT``.
    ``bodies`` solves that list instead (experiment eval); ``jtsmm record``
    leaves it unset.
    """
    priors = dict(prior_T_world_body or {})
    out: Dict[str, WorldBodyPose] = {}
    for body in BODY_NAMES if bodies is None else bodies:
        per_cam: List[Tuple[str, BodyPoseEstimate]] = []
        for cam_id, detection in detections.items():
            if cam_id not in intrinsics or cam_id not in extrinsics:
                continue
            est = estimate_body_pose_in_camera(
                body,
                detection.markers,
                intrinsics=intrinsics[cam_id],
                marker_config=marker_config,
                max_reprojection_rms_px=max_reprojection_rms_px,
            )
            if est is not None:
                per_cam.append((cam_id, est))

        if not per_cam:
            out[body] = WorldBodyPose(
                body=body,
                T_world_body=nan_pose(),
                n_tags=0,
                n_cameras=0,
                reprojection_rms_px=float("nan"),
                valid=False,
            )
            continue

        world_cands = {
            cam_id: [
                (cam_to_world_pose(c.T_cam_body, extrinsics[cam_id]), c)
                for c in (est.candidates or [PoseCandidate(est.T_cam_body, est.reprojection_rms_px)])
            ]
            for cam_id, est in per_cam
        }

        R_ref, resolved_by = _resolve_reference(per_cam, world_cands, priors.get(body))

        Ts: List[np.ndarray] = []
        rms: List[float] = []
        n_tags = 0
        for cam_id, est in per_cam:
            if R_ref is not None:
                T_w, cand = _closest(world_cands[cam_id], R_ref)
            else:
                T_w, cand = world_cands[cam_id][0]
            Ts.append(T_w)
            rms.append(cand.reprojection_rms_px)
            n_tags += est.n_tags

        if use_reprojection_weights:
            T_fused = weighted_fuse_transforms(Ts, [1.0 / max(r, 1e-3) for r in rms])
        else:
            T_fused = fuse_transforms(Ts)

        out[body] = WorldBodyPose(
            body=body,
            T_world_body=T_fused if T_fused is not None else nan_pose(),
            n_tags=n_tags,
            n_cameras=len(per_cam),
            reprojection_rms_px=float(np.median(rms)),
            valid=T_fused is not None and is_valid_pose(T_fused),
            ambiguous=R_ref is None,
            resolved_by=resolved_by,
        )
    return out
