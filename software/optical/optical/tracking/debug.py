# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Per-frame tracker diagnostics for offline replay and calibration debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

import numpy as np

from optical.tracking.aruco import DetectionResult
from optical.tracking.pose_estimation import (
    BODY_NAMES,
    WorldBodyPose,
)
from optical.tracking.tracker import ArucoJawTracker
from optical.tracking.triangulation import (
    KabschFit,
    estimate_world_body_poses_triangulation,
)

PoseMethod = Literal["pnp", "triangulation"]


@dataclass
class BodyDebug:
    """Pose + triangulation diagnostics for one rigid body."""

    body: str
    pose_valid: bool
    n_tags_detected: int
    n_triangulated_corners: int
    n_matched_corners: int
    n_model_corners: int
    kabsch_rms_mm: Optional[float] = None
    fail_reason: Optional[str] = None


@dataclass
class FrameDebugReport:
    """Full diagnostic snapshot for one replay frame."""

    frame_idx: int
    marker_ids_by_cam: Dict[str, List[int]] = field(default_factory=dict)
    stereo_marker_ids: List[int] = field(default_factory=list)
    n_triangulated_corners: int = 0
    bodies: Dict[str, BodyDebug] = field(default_factory=dict)
    pose_method: str = "pnp"

    def _body_token(self, body: str) -> str:
        bd = self.bodies.get(body)
        if bd is None:
            return f"{body}=?"
        token = f"{body}={'OK' if bd.pose_valid else 'FAIL'}"
        if not bd.pose_valid and bd.fail_reason:
            token += f"({bd.fail_reason})"
        elif bd.pose_valid and bd.kabsch_rms_mm is not None:
            if self.pose_method == "pnp":
                token += f"(reproj={bd.kabsch_rms_mm:.1f}px)"
            else:
                token += f"(rms={bd.kabsch_rms_mm:.1f}mm)"
        return token

    def format_line(self, *, verbose: bool = False) -> str:
        parts = [f"frame {self.frame_idx}"]
        for cam_id in sorted(self.marker_ids_by_cam):
            ids = self.marker_ids_by_cam[cam_id]
            parts.append(f"{cam_id}={len(ids)} ids{ids if verbose else ''}")
        if self.pose_method == "triangulation":
            parts.append(f"stereo={len(self.stereo_marker_ids)}")
            parts.append(f"tri_corners={self.n_triangulated_corners}")
        for body in BODY_NAMES:
            parts.append(self._body_token(body))
        return "  ".join(parts)


@dataclass
class MethodCompareReport:
    """PnP vs triangulation diagnostics for the same detections."""

    frame_idx: int
    marker_ids_by_cam: Dict[str, List[int]] = field(default_factory=dict)
    stereo_marker_ids: List[int] = field(default_factory=list)
    n_triangulated_corners: int = 0
    pnp: FrameDebugReport = field(default_factory=lambda: FrameDebugReport(0, pose_method="pnp"))
    triangulation: FrameDebugReport = field(
        default_factory=lambda: FrameDebugReport(0, pose_method="triangulation")
    )

    def format_line(self, *, verbose: bool = False) -> str:
        parts = [f"frame {self.frame_idx}"]
        for cam_id in sorted(self.marker_ids_by_cam):
            ids = self.marker_ids_by_cam[cam_id]
            parts.append(f"{cam_id}={len(ids)} ids{ids if verbose else ''}")
        parts.append(f"stereo={len(self.stereo_marker_ids)}")
        parts.append(f"tri_corners={self.n_triangulated_corners}")
        parts.append("PnP:" + " ".join(self.pnp._body_token(b) for b in BODY_NAMES))
        parts.append("TRI:" + " ".join(self.triangulation._body_token(b) for b in BODY_NAMES))
        return "  ".join(parts)


def _marker_ids_by_cam(detections: Dict[str, DetectionResult]) -> Dict[str, List[int]]:
    return {
        cam_id: sorted(marker.marker_id for marker in detection.markers)
        for cam_id, detection in detections.items()
    }


def _stereo_marker_ids(detections: Dict[str, DetectionResult], min_views: int) -> List[int]:
    counts: Dict[int, int] = {}
    for detection in detections.values():
        seen: Set[int] = set()
        for marker in detection.markers:
            seen.add(marker.marker_id)
        for marker_id in seen:
            counts[marker_id] = counts.get(marker_id, 0) + 1
    return sorted(mid for mid, n in counts.items() if n >= min_views)


def _diagnose_body_triangulation(body: str, fit: Optional[KabschFit], pose: WorldBodyPose) -> BodyDebug:
    """Report the tracker's own trimmed Kabsch fit (no second, untrimmed solve)."""
    if fit is None:
        return BodyDebug(body, False, 0, 0, 0, 0, fail_reason="no_fit")
    diag = BodyDebug(
        body=body,
        pose_valid=bool(pose.valid),
        n_tags_detected=fit.n_tags,
        n_triangulated_corners=fit.n_matched,
        n_matched_corners=fit.n_used,
        n_model_corners=fit.n_model,
        kabsch_rms_mm=None if not np.isfinite(fit.rms_mm) else float(fit.rms_mm),
        fail_reason=None if pose.valid else (fit.fail_reason or "invalid_pose"),
    )
    return diag


def _diagnose_body_pnp(body: str, pose: WorldBodyPose) -> BodyDebug:
    if pose.valid:
        return BodyDebug(
            body=body,
            pose_valid=True,
            n_tags_detected=pose.n_tags,
            n_triangulated_corners=0,
            n_matched_corners=0,
            n_model_corners=0,
            kabsch_rms_mm=pose.reprojection_rms_px,
        )
    reason = "no_pose"
    if pose.n_cameras == 0:
        reason = "no_cameras"
    elif pose.n_tags == 0:
        reason = "no_tags"
    return BodyDebug(
        body=body,
        pose_valid=False,
        n_tags_detected=pose.n_tags,
        n_triangulated_corners=0,
        n_matched_corners=0,
        n_model_corners=0,
        fail_reason=reason,
    )


def _build_report_from_detections(
    tracker: ArucoJawTracker,
    detections: Dict[str, DetectionResult],
    *,
    frame_idx: int,
    pose_method: PoseMethod,
    precomputed=None,
) -> FrameDebugReport:
    """Analyze one pose method from detections.

    ``precomputed`` (a tracker ``LastEstimate`` for the same detections) skips
    the pose solve, so replay does not estimate every frame twice.
    """
    tcfg = tracker.config
    report = FrameDebugReport(
        frame_idx=frame_idx,
        marker_ids_by_cam=_marker_ids_by_cam(detections),
        pose_method=pose_method,
    )
    if precomputed is not None and precomputed.method != pose_method:
        precomputed = None

    if pose_method == "triangulation":
        report.stereo_marker_ids = _stereo_marker_ids(detections, tcfg.triangulation_min_views)
        if precomputed is not None:
            body_poses, triangulated, fits = (
                precomputed.body_poses,
                precomputed.triangulated or [],
                precomputed.fits,
            )
        else:
            fits = {}
            body_poses, triangulated = estimate_world_body_poses_triangulation(
                detections,
                intrinsics=tracker.intrinsics,
                extrinsics=tracker.extrinsics,
                marker_config=tracker.marker_config,
                min_views=tcfg.triangulation_min_views,
                max_residual_mm=tcfg.triangulation_max_residual_mm,
                max_corner_reproj_px=tcfg.triangulation_max_corner_reproj_px,
                diagnostics=fits,
            )
        report.n_triangulated_corners = len(triangulated)
        for body in BODY_NAMES:
            report.bodies[body] = _diagnose_body_triangulation(body, fits.get(body), body_poses[body])
        return report

    if precomputed is not None:
        body_poses = precomputed.body_poses
    else:
        # Same gates and flip prior as the live tracker (read-only), so a
        # replay report matches what process_frame writes.
        body_poses, _ = tracker.estimate_poses(detections, method="pnp", update_state=False)
    for body in BODY_NAMES:
        report.bodies[body] = _diagnose_body_pnp(body, body_poses[body])
    return report


def report_from_last_frame(tracker: ArucoJawTracker, *, frame_idx: int) -> FrameDebugReport:
    """Debug report for the frame :meth:`ArucoJawTracker.process_frame` just ran."""
    last = tracker.last_estimate
    return _build_report_from_detections(
        tracker,
        tracker.last_detections,
        frame_idx=frame_idx,
        pose_method=(last.method if last is not None else tracker.config.pose_method),
        precomputed=last,
    )


def analyze_frame(
    tracker: ArucoJawTracker,
    frames: Dict[str, np.ndarray],
    *,
    frame_idx: int,
    pose_method: Optional[PoseMethod] = None,
) -> FrameDebugReport:
    """
    Run detection + pose estimation and return a structured debug report.

    Args:
        pose_method: Override tracker config method (``pnp`` or ``triangulation``).
    """
    detections = tracker.detect_all(frames)
    method: PoseMethod = pose_method or tracker.config.pose_method  # type: ignore[assignment]
    return _build_report_from_detections(
        tracker, detections, frame_idx=frame_idx, pose_method=method
    )


def compare_methods_frame(
    tracker: ArucoJawTracker,
    frames: Dict[str, np.ndarray],
    *,
    frame_idx: int,
) -> MethodCompareReport:
    """Run PnP and triangulation on the same detections for one frame."""
    detections = tracker.detect_all(frames)
    pnp = _build_report_from_detections(
        tracker, detections, frame_idx=frame_idx, pose_method="pnp"
    )
    tri = _build_report_from_detections(
        tracker, detections, frame_idx=frame_idx, pose_method="triangulation"
    )
    return MethodCompareReport(
        frame_idx=frame_idx,
        marker_ids_by_cam=pnp.marker_ids_by_cam,
        stereo_marker_ids=tri.stereo_marker_ids,
        n_triangulated_corners=tri.n_triangulated_corners,
        pnp=pnp,
        triangulation=tri,
    )


def summarize_reports(reports: List[FrameDebugReport]) -> str:
    """Printable summary over a replay run."""
    if not reports:
        return "No frames processed."

    n = len(reports)
    lines = [f"Processed {n} frame(s), method={reports[0].pose_method}"]

    if reports[0].pose_method == "triangulation":
        avg_tri = sum(r.n_triangulated_corners for r in reports) / n
        avg_stereo = sum(len(r.stereo_marker_ids) for r in reports) / n
        lines.append(f"  avg stereo marker ids: {avg_stereo:.1f}")
        lines.append(f"  avg triangulated corners: {avg_tri:.1f}")

    for body in BODY_NAMES:
        valid_n = sum(1 for r in reports if r.bodies.get(body, BodyDebug(body, False, 0, 0, 0, 0)).pose_valid)
        lines.append(f"  {body}: valid on {valid_n}/{n} frames ({100.0 * valid_n / n:.1f}%)")

    fail_reasons: Dict[str, int] = {}
    for report in reports:
        for body in BODY_NAMES:
            bd = report.bodies.get(body)
            if bd is None or bd.pose_valid or not bd.fail_reason:
                continue
            key = f"{body}:{bd.fail_reason}"
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    if fail_reasons:
        lines.append("  common failures:")
        for key, count in sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"    {key}: {count} frame(s)")

    return "\n".join(lines)


def summarize_compare_reports(reports: List[MethodCompareReport]) -> str:
    """Printable summary comparing PnP and triangulation on the same frames."""
    if not reports:
        return "No frames processed."

    n = len(reports)
    lines = [f"Processed {n} frame(s), PnP vs triangulation on identical detections"]
    avg_stereo = sum(len(r.stereo_marker_ids) for r in reports) / n
    avg_tri = sum(r.n_triangulated_corners for r in reports) / n
    lines.append(f"  avg stereo marker ids: {avg_stereo:.1f}")
    lines.append(f"  avg triangulated corners: {avg_tri:.1f}")

    for label, attr in (("PnP", "pnp"), ("Triangulation", "triangulation")):
        lines.append(f"  [{label}]")
        method_reports = [getattr(r, attr) for r in reports]
        for body in BODY_NAMES:
            valid_n = sum(
                1
                for rep in method_reports
                if rep.bodies.get(body, BodyDebug(body, False, 0, 0, 0, 0)).pose_valid
            )
            lines.append(f"    {body}: valid on {valid_n}/{n} frames ({100.0 * valid_n / n:.1f}%)")

    both_valid = {body: 0 for body in BODY_NAMES}
    pnp_only = {body: 0 for body in BODY_NAMES}
    tri_only = {body: 0 for body in BODY_NAMES}
    neither = {body: 0 for body in BODY_NAMES}
    for report in reports:
        for body in BODY_NAMES:
            pnp_ok = report.pnp.bodies.get(body, BodyDebug(body, False, 0, 0, 0, 0)).pose_valid
            tri_ok = report.triangulation.bodies.get(
                body, BodyDebug(body, False, 0, 0, 0, 0)
            ).pose_valid
            if pnp_ok and tri_ok:
                both_valid[body] += 1
            elif pnp_ok:
                pnp_only[body] += 1
            elif tri_ok:
                tri_only[body] += 1
            else:
                neither[body] += 1

    lines.append("  [Agreement]")
    for body in BODY_NAMES:
        lines.append(
            f"    {body}: both={both_valid[body]}  pnp_only={pnp_only[body]}  "
            f"tri_only={tri_only[body]}  neither={neither[body]}"
        )

    for label, attr in (("PnP", "pnp"), ("Triangulation", "triangulation")):
        fail_reasons: Dict[str, int] = {}
        for report in reports:
            rep = getattr(report, attr)
            for body in BODY_NAMES:
                bd = rep.bodies.get(body)
                if bd is None or bd.pose_valid or not bd.fail_reason:
                    continue
                key = f"{body}:{bd.fail_reason}"
                fail_reasons[key] = fail_reasons.get(key, 0) + 1
        if fail_reasons:
            lines.append(f"  [{label} common failures]")
            for key, count in sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:6]:
                lines.append(f"    {key}: {count} frame(s)")

    return "\n".join(lines)
