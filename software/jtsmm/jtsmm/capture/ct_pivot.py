# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Interactive CT tip pivot calibration (cone + SPACE × N → OLS)."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from optical.calibration.multicam import PreviewServer, _draw_status_bar, _fit_preview, _stack_preview
from optical.tracking.pivot import PivotCalibrationResult, solve_pivot_ols
from optical.tracking.pose import fuse_transforms, is_valid_pose

from jtsmm.io.terminal_keys import TerminalKeys

CT_TIP_CALIBRATION_NAME = "ct_tip_calibration.json"

# Drop SPACE samples whose tip residual exceeds this after OLS (mm).
MAX_PER_POSE_ERR_MM = 4.0

# A tip solve with a larger RMS is "do not trust": kept in the session, never
# written over the shared, reusable tip JSON.
CT_TIP_TRUST_MAX_RMS_MM = 5.0

# Heuristic: cond(A) of the pivot design matrix grows without bound as the
# orientations collapse onto one rotation axis (the tip component along it
# becomes unobservable). Well-spread poses stay in the single digits.
MAX_COND_A = 100.0

# Live CT visibility probe runs the tracker off the hot path; throttle it.
_CT_PROBE_PERIOD_S = 0.3

# One SPACE sample: hold still for settle_s, want min_samples valid poses, but
# never spin longer than this -- once the body leaves view the loop used to wait
# forever with 1-7 samples and no key polling.
_SAMPLE_MAX_S = 3.0


def ct_tip_trustworthy(result: PivotCalibrationResult) -> bool:
    rms = float(result.rms_mm)
    return bool(np.isfinite(rms) and rms <= CT_TIP_TRUST_MAX_RMS_MM)


def reject_high_err_poses(
    poses: list[np.ndarray],
    result: PivotCalibrationResult,
    sample_frames: list[dict[str, np.ndarray]] | None = None,
    *,
    max_err_mm: float = MAX_PER_POSE_ERR_MM,
    min_poses: int = 3,
    log: Callable[[str], None] | None = None,
) -> tuple[list[np.ndarray], PivotCalibrationResult, list[dict[str, np.ndarray]], int]:
    """Drop the WORST pose while it exceeds ``max_err_mm``, re-solving after each.

    One at a time, because a single bad pose drags the OLS tip and inflates the
    residuals of good poses too; dropping every pose above the threshold at once
    threw those good poses away with it. Stops before fewer than ``min_poses``
    would remain (logs a warning) and warns when ``cond_A`` says the
    orientations are too similar. Returns ``(kept_poses, result, kept_frames,
    n_rejected)``.
    """
    emit = log if log is not None else (lambda _msg: None)
    kept_poses = list(poses)
    kept_frames = list(sample_frames or ())
    rejected_total = 0
    while True:
        errs = [float(e) for e in result.per_pose_err_mm]
        if not errs:
            break
        worst = int(np.argmax(errs))
        if errs[worst] <= float(max_err_mm):
            break
        n_bad = sum(1 for e in errs if e > float(max_err_mm))
        if len(kept_poses) - 1 < int(min_poses):
            emit(
                f"warning: {n_bad} pose(s) exceed {max_err_mm:.1f} mm "
                f"but rejecting more would leave <{min_poses} poses — keeping {len(kept_poses)}"
            )
            break
        emit(
            f"  rejecting pose {worst + 1}/{len(kept_poses)}: "
            f"per_pose_err={errs[worst]:.3f} mm (> {max_err_mm:.1f} mm, worst of {n_bad})"
        )
        kept_poses.pop(worst)
        if worst < len(kept_frames):
            kept_frames.pop(worst)
        rejected_total += 1
        result = solve_pivot_ols(kept_poses)
    cond = float(result.cond_A)
    if not np.isfinite(cond) or cond > MAX_COND_A:
        emit(
            f"warning: cond(A)={cond:.1e} -- the poses are too similar (rotations about "
            "one axis?); tilt the tool in two directions for a well-conditioned tip"
        )
    return kept_poses, result, kept_frames, rejected_total


def write_ct_tip_calibration(path: Path | str, result: PivotCalibrationResult) -> Path:
    """Write tip OLS result JSON (tip_body_mm, pivot_world_mm, rms, …)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def _copy_frames(frames: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for cam_id, image in frames.items():
        arr = np.asarray(image)
        out[str(cam_id)] = arr.copy() if arr.size else arr
    return out


def _encode_frame_jpeg(image: np.ndarray) -> bytes | None:
    try:
        from jtsmm.viewer.jpeg_util import encode_jpeg

        return encode_jpeg(np.asarray(image))
    except Exception:  # noqa: BLE001 — persist must not abort calib
        return None


def write_ct_calibration_recording(
    out_dir: Path | str,
    result: PivotCalibrationResult,
    poses: list[np.ndarray],
    sample_frames: list[dict[str, np.ndarray]] | None = None,
    *,
    experiment_name: str | None = None,
    calibration_bundle_path: Path | str | None = None,
    optical_config_path: Path | str | None = None,
) -> Path:
    """Write OLS JSON, 4×4 poses, camera JPEGs, and calibration snapshots.

    When ``calibration_bundle_path`` is set, copies the active optical
    ``calibration_bundle.h5`` (and optional ``optical.yaml``) into ``out_dir``
    so offline recompute stays reproducible after later recalibration.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    write_ct_tip_calibration(root / CT_TIP_CALIBRATION_NAME, result)

    calib_rel: str | None = None
    calib_src: str | None = None
    if calibration_bundle_path is not None:
        from optical.paths import (
            CALIBRATION_BUNDLE_SNAPSHOT_NAME,
            snapshot_persist_calibration,
        )

        arts = snapshot_persist_calibration(
            root,
            bundle_path=Path(calibration_bundle_path),
            config_path=Path(optical_config_path) if optical_config_path else None,
        )
        calib_rel = CALIBRATION_BUNDLE_SNAPSHOT_NAME
        calib_src = str(Path(calibration_bundle_path).resolve())
        print(
            f"  snapshotted calibration → {arts['calibration_bundle'].name}",
            file=sys.stderr,
        )

    cam_ids: list[str] = []
    image_size: tuple[int, int] | None = None
    for frames in sample_frames or ():
        for cam_id, image in frames.items():
            cid = str(cam_id)
            if cid not in cam_ids:
                cam_ids.append(cid)
            if image_size is None:
                arr = np.asarray(image)
                if arr.ndim >= 2:
                    image_size = (int(arr.shape[1]), int(arr.shape[0]))

    payload: dict[str, Any] = {
        "experiment_name": experiment_name,
        "n_poses": int(len(poses)),
        "ols": result.to_dict(),
        "poses_world_body": [np.asarray(T, dtype=np.float64).tolist() for T in poses],
    }
    if calib_rel is not None:
        payload["calibration_bundle"] = calib_rel
        payload["calibration_bundle_source"] = calib_src
    (root / "poses.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Same shape as optical --persist-frames meta so load_tracker_for_recording
    # can scale intrinsics from the leaf without the live optical.yaml.
    meta: dict[str, Any] = {
        "schema_version": "1.0",
        "kind": "ct_tip_calibration",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cam_ids": cam_ids,
        "notes": f"ct tip pivot calibration: {experiment_name or 'unnamed'}",
        "stream": "lores_y",
    }
    if image_size is not None:
        meta["image_size"] = [int(image_size[0]), int(image_size[1])]
    if calib_rel is not None:
        meta["calibration_bundle"] = calib_rel
        meta["calibration_bundle_source"] = calib_src
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    frames_root = root / "frames"
    for idx, frames in enumerate(sample_frames or ()):
        sample_dir = frames_root / f"{idx:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for cam_id, image in frames.items():
            jpeg = _encode_frame_jpeg(image)
            if not jpeg:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(cam_id))
            (sample_dir / f"{safe}.jpg").write_bytes(jpeg)
    return root


def load_ct_tip_calibration(path: Path | str) -> PivotCalibrationResult:
    """Load a previously saved tip calibration JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tip = np.asarray(data["tip_body_mm"], dtype=np.float64).reshape(3)
    pivot = np.asarray(
        data.get("pivot_world_mm", [np.nan, np.nan, np.nan]), dtype=np.float64
    ).reshape(3)
    return PivotCalibrationResult(
        tip_body_mm=tip,
        pivot_world_mm=pivot,
        rms_mm=float(data.get("rms_mm", float("nan"))),
        n_poses=int(data.get("n_poses", 0)),
        residual_norm=float(data.get("residual_norm", float("nan"))),
        tip_length_mm=float(data.get("tip_length_mm", float(np.linalg.norm(tip)))),
        max_err_mm=float(data.get("max_err_mm", float("nan"))),
        median_err_mm=float(data.get("median_err_mm", float("nan"))),
        cond_A=float(data.get("cond_A", float("nan"))),
        per_pose_err_mm=tuple(float(x) for x in (data.get("per_pose_err_mm") or ())),
    )


def _drain_rgb_frames(cameras: list[Any], cam_ids: set[str]) -> dict[str, Any]:
    """Drain latest lores Y from each optical cam (keeps buffers from filling).

    Frames are handed to the tracker exactly as the recording path does in
    ``SessionSupervisor``, so the tip solved here matches the poses written to
    ``/optical`` during the take.
    """
    frames: dict[str, Any] = {}
    for cam in cameras:
        if cam.cam_id not in cam_ids:
            continue
        for frame in cam.drain_frames(max_frames=4, with_rgb=True, rgb_limit=1):
            rgb = getattr(frame, "rgb", None)
            if rgb is not None:
                frames[cam.cam_id] = rgb
    return frames


@dataclass
class _SampleDiag:
    """Why a SPACE press produced (or failed to produce) a CT pose."""

    iterations: int = 0
    drained: Counter = field(default_factory=Counter)
    all_cams: int = 0
    valid_ct: int = 0
    last_frames: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False

    def summary(self, cam_order: list[str]) -> str:
        drained = " ".join(f"{cid}={self.drained.get(cid, 0)}" for cid in cam_order)
        return (
            f"y[{drained}] frames_with_all_cams={self.all_cams} "
            f"valid_CT_poses={self.valid_ct} (loops={self.iterations})"
        )


def _frame_report(optical: Any, frames: dict[str, Any]) -> str | None:
    """Per-body detection / pose diagnostics for one frame bundle."""
    if not frames:
        return None
    try:
        from optical.tracking.debug import analyze_frame

        report = analyze_frame(optical.tracker, frames, frame_idx=0)
        return report.format_line(verbose=True)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never abort calib
        return f"(diagnostics unavailable: {exc})"


def _ct_geometry_and_fix(
    optical: Any,
    frames: dict[str, Any],
) -> tuple[str | None, list[str]]:
    """Model-free CT geometry readout plus a suggested mount correction.

    The readout separates a tag-size / world-scale error (measured edge or
    separation disagrees with the config) from a tag-frame orientation error
    (they agree, yet the Kabsch residual is near the tag edge length). In the
    latter case the mount fitter names which of the eight symmetries each tag is
    actually in and emits the corrected transforms.
    """
    if not frames:
        return None, []
    try:
        from optical.tracking.mount_fit import (
            fit_body_mount_corrections,
            measure_body_geometry,
        )
        from optical.tracking.triangulation import triangulate_tag_corners

        tracker = optical.tracker
        triangulated = triangulate_tag_corners(
            tracker.detect_all(frames),
            intrinsics=tracker.intrinsics,
            extrinsics=tracker.extrinsics,
            marker_config=tracker.marker_config,
            min_views=tracker.config.triangulation_min_views,
        )
        if not triangulated:
            return None, []
        readout = "CT geometry: " + measure_body_geometry(
            triangulated, body="CT", marker_config=tracker.marker_config
        )
        fit = fit_body_mount_corrections(
            triangulated, body="CT", marker_config=tracker.marker_config
        )
        fix: list[str] = []
        if fit is not None and fit.changed and fit.rms_mm < 0.5 * fit.baseline_rms_mm:
            fix.append(fit.summary())
            fix.append("Suggested markers.mounts_T_body_tag (check against CAD before use):")
            fix.extend(fit.yaml_snippet().splitlines())
        return readout, fix
    except Exception as exc:  # noqa: BLE001 - diagnostics must never abort calib
        return f"(mount diagnostics unavailable: {exc})", []


def _ct_token(optical: Any, frames: dict[str, Any]) -> tuple[str, bool]:
    """Short ``CT=…`` token for the live preview status bar."""
    if not frames:
        return "CT=no frames", False
    try:
        from optical.tracking.debug import analyze_frame

        report = analyze_frame(optical.tracker, frames, frame_idx=0)
        body = report.bodies.get("CT")
        if body is None:
            return "CT=?", False
        if body.pose_valid:
            return "CT=OK", True
        return f"CT={body.fail_reason or 'FAIL'}", False
    except Exception:  # noqa: BLE001
        return "", False


def _overlay_from_detections(detections: dict[str, Any] | None, cam_id: str) -> list[tuple[int, Any]]:
    if not detections:
        return []
    det = detections.get(cam_id)
    if det is None:
        return []
    return [(int(m.marker_id), m.corners) for m in getattr(det, "markers", ())]


def _publish_preview(
    preview: PreviewServer | None,
    frames: dict[str, Any],
    cam_order: list[str],
    status: str,
    status_color: tuple[int, int, int] = (0, 220, 0),
    detections: dict[str, Any] | None = None,
) -> None:
    if preview is None or not frames:
        return
    import cv2

    from jtsmm.viewer.aruco_overlay import draw_marker_boxes_bgr

    # ``frames`` are lores Y (or legacy RGB). PreviewServer encodes with
    # cv2.imencode as BGR, so expand gray → BGR here. Overlay uses tracker
    # corners; the tracker keeps the untouched gray frames.
    overlaid = {}
    for cid, rgb in frames.items():
        img = np.asarray(rgb)
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            bgr = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        markers = _overlay_from_detections(detections, cid)
        if markers:
            src_wh = (int(bgr.shape[1]), int(bgr.shape[0]))
            bgr = draw_marker_boxes_bgr(bgr, markers, src_wh=src_wh)
        overlaid[cid] = bgr
    stacked = _stack_preview(overlaid, cam_order)
    if stacked is None:
        return
    preview.update(_draw_status_bar(_fit_preview(stacked), status, status_color))


def _sample_ct_pose(
    optical: Any,
    cameras: list[Any],
    *,
    settle_s: float = 0.6,
    min_samples: int = 8,
    preview: PreviewServer | None = None,
    cam_order: list[str] | None = None,
    status: str = "",
    max_s: float = _SAMPLE_MAX_S,
    poll_key: Callable[[], int | None] | None = None,
) -> tuple[np.ndarray | None, _SampleDiag]:
    """Hold still: collect several valid CT poses and fuse (quat avg + median t).

    Ends when ``min_samples`` are in after ``settle_s``; with none at all after
    ``settle_s + 1 s``; and in any case after ``max_s`` (then whatever was
    collected, >= 1, is fused). Q/ESC via ``poll_key`` aborts the calibration.
    """
    cam_ids = set(optical.cam_ids)
    order = cam_order or list(cam_ids)
    t0 = time.monotonic()
    deadline = t0 + settle_s
    hard_deadline = t0 + max(float(max_s), settle_s)
    samples: list[np.ndarray] = []
    diag = _SampleDiag()
    while True:
        now = time.monotonic()
        if now >= deadline and len(samples) >= min_samples:
            break
        if now > deadline + 1.0 and not samples:
            break
        if now >= hard_deadline:
            diag.timed_out = True
            break
        if poll_key is not None and poll_key() in (27, ord("q")):
            raise RuntimeError("CT pivot calibration aborted by operator")
        diag.iterations += 1
        frames = _drain_rgb_frames(cameras, cam_ids)
        for cid in frames:
            diag.drained[cid] += 1
        if frames:
            diag.last_frames = frames
        dets = None
        if frames and cam_ids and cam_ids.issubset(frames):
            diag.all_cams += 1
            record = optical.tracker.process_frame(
                frames,
                frame_idx=0,
                t_cam_ns={cid: 0 for cid in frames},
                t_master_us=0,
            )
            dets = getattr(optical.tracker, "last_detections", None)
            if is_valid_pose(record.pose_CT):
                diag.valid_ct += 1
                samples.append(np.asarray(record.pose_CT, dtype=np.float64).copy())
        elif frames:
            dets = optical.tracker.detect_all(frames)
        _publish_preview(
            preview, frames, order, status or "sampling…", (0, 200, 255), detections=dets
        )
        time.sleep(0.01)
    if not samples:
        return None, diag
    return fuse_transforms(samples), diag


def run_ct_pivot_calibration(
    optical: Any,
    cameras: list[Any],
    *,
    n_poses: int = 20,
    session_dir: Path | None = None,
    persist_dir: Path | None = None,
    experiment_name: str | None = None,
    discard_mcu: Callable[[], None] | None = None,
    preview_host: str = "0.0.0.0",
    preview_port: int = 5000,
) -> PivotCalibrationResult:
    """Collect ``n_poses`` CT orientations with tip in cone; solve tip OLS.

    Operator: seat the removable tip in the calibration cone, then press SPACE
    for each diverse orientation. Q/ESC aborts. Live MJPEG preview runs for the
    whole phase (same ``PreviewServer`` as ``optical-track`` / calibrate).
    """
    n = max(3, int(n_poses))
    poses: list[np.ndarray] = []
    sample_frames: list[dict[str, np.ndarray]] = []
    cam_ids = set(getattr(optical, "cam_ids", []) or [])
    cam_order = list(cam_ids) or [c.cam_id for c in cameras]
    n_cams = len(cam_ids)
    pose_method = "unknown"
    tracker = getattr(optical, "tracker", None)
    cfg = getattr(tracker, "config", None) if tracker is not None else None
    if cfg is not None:
        pose_method = str(getattr(cfg, "pose_method", "unknown"))

    preview: PreviewServer | None = None
    if preview_port > 0:
        try:
            preview = PreviewServer(preview_host, preview_port)
            preview.start()
            print(
                f"CT tip preview: http://127.0.0.1:{preview_port}/  "
                f"(bind {preview.url})",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001 - preview must not block calib
            print(f"warning: CT tip preview disabled ({exc})", file=sys.stderr)
            preview = None

    print("=" * 72, file=sys.stderr)
    print(
        f"CT tip pivot calibration: seat tip in cone, HOLD STILL, then SPACE × {n} "
        "(vary orientation widely). Q/ESC aborts.",
        file=sys.stderr,
    )
    if n_cams < 2 or pose_method == "pnp":
        print(
            "warning: single-cam / PnP poses have large depth error when you reorient "
            "the tool — tip OLS RMS of cm is common. Connect a second CSI camera "
            "so optical.yaml auto-selects triangulation for tip calib.",
            file=sys.stderr,
        )
    print("=" * 72, file=sys.stderr)

    ct_token = ""
    ct_ok = False
    next_probe = 0.0
    mount_fix_shown = False  # emit the YAML suggestion once, not per retry
    with TerminalKeys() as keys:
        while len(poses) < n:
            if discard_mcu is not None:
                try:
                    discard_mcu()
                except Exception:
                    pass
            frames = _drain_rgb_frames(cameras, cam_ids or {c.cam_id for c in cameras})

            # Live CT visibility so the operator can pose the tool *before* SPACE.
            now = time.monotonic()
            if frames and now >= next_probe:
                ct_token, ct_ok = _ct_token(optical, frames)
                next_probe = now + _CT_PROBE_PERIOD_S

            status = f"CT tip: {len(poses)}/{n} — SPACE when still | Q abort"
            if ct_token:
                status = f"{status}  [{ct_token}]"
            dets = getattr(getattr(optical, "tracker", None), "last_detections", None)
            _publish_preview(
                preview,
                frames,
                cam_order,
                status,
                (0, 220, 0) if ct_ok else (0, 165, 255),
                detections=dets,
            )

            key = keys.poll()
            if key in (27, ord("q")):
                raise RuntimeError("CT pivot calibration aborted by operator")
            if key == ord(" "):
                print(
                    f"  [{len(poses)+1}/{n}] sampling (hold still)…",
                    file=sys.stderr,
                )
                pose, diag = _sample_ct_pose(
                    optical,
                    cameras,
                    preview=preview,
                    cam_order=cam_order,
                    status=f"CT tip: sampling {len(poses)+1}/{n}…",
                    poll_key=keys.poll,
                )
                if pose is not None and diag.timed_out:
                    print(
                        f"        note: fused {diag.valid_ct} pose(s) (CT left view before "
                        "the full hold) — hold still longer if the residual is high",
                        file=sys.stderr,
                    )
                if pose is None:
                    print(
                        f"  [{len(poses)+1}/{n}] CT not visible — retry SPACE",
                        file=sys.stderr,
                    )
                    print(f"        {diag.summary(cam_order)}", file=sys.stderr)
                    report = _frame_report(optical, diag.last_frames)
                    if report:
                        print(f"        {report}", file=sys.stderr)
                    readout, fix = _ct_geometry_and_fix(optical, diag.last_frames)
                    if readout:
                        print(f"        {readout}", file=sys.stderr)
                    if fix and not mount_fix_shown:
                        for line in fix:
                            print(f"        {line}", file=sys.stderr)
                        mount_fix_shown = True
                    continue
                poses.append(pose)
                if persist_dir is not None:
                    sample_frames.append(_copy_frames(diag.last_frames))
                t = pose[:3, 3]
                print(
                    f"  [{len(poses)}/{n}] CT body origin  "
                    f"t=[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm",
                    file=sys.stderr,
                )
            else:
                time.sleep(0.02)

    result = solve_pivot_ols(poses)
    print(
        f"CT tip OLS (initial): tip_body_mm={result.tip_body_mm.tolist()}  "
        f"|tip|={result.tip_length_mm:.1f} mm  "
        f"rms={result.rms_mm:.3f} mm  med={result.median_err_mm:.3f}  "
        f"max={result.max_err_mm:.3f}  cond={result.cond_A:.1e}  (n={result.n_poses})",
        file=sys.stderr,
    )

    poses, result, sample_frames, rejected_total = reject_high_err_poses(
        poses,
        result,
        sample_frames,
        max_err_mm=MAX_PER_POSE_ERR_MM,
        # never discard more than half of what the operator collected
        min_poses=max(3, (len(poses) + 1) // 2),
        log=lambda msg: print(msg, file=sys.stderr),
    )
    if rejected_total:
        print(
            f"CT tip OLS (after rejecting {rejected_total} outlier pose(s)): "
            f"tip_body_mm={result.tip_body_mm.tolist()}  "
            f"|tip|={result.tip_length_mm:.1f} mm  "
            f"rms={result.rms_mm:.3f} mm  med={result.median_err_mm:.3f}  "
            f"max={result.max_err_mm:.3f}  cond={result.cond_A:.1e}  "
            f"(n={result.n_poses})",
            file=sys.stderr,
        )

    if not ct_tip_trustworthy(result):
        print(
            "warning: tip residual is large — tip was not observed as fixed in the "
            "tracked CT body frame (mono PnP depth noise, tip slip, or too-similar "
            "orientations). Do not trust this offset for digitization; it is kept "
            "with this session only, not written to the shared tip file.",
            file=sys.stderr,
        )
    if session_dir is not None:
        out = write_ct_tip_calibration(Path(session_dir) / CT_TIP_CALIBRATION_NAME, result)
        print(f"Wrote {out}", file=sys.stderr)
    if persist_dir is not None:
        try:
            tracker = getattr(optical, "tracker", None)
            tcfg = getattr(tracker, "config", None) if tracker is not None else None
            bundle_path = getattr(tcfg, "calibration_bundle_path", None) if tcfg else None
            opt_cfg = getattr(optical, "config_path", None)
            rec = write_ct_calibration_recording(
                persist_dir,
                result,
                poses,
                sample_frames,
                experiment_name=experiment_name,
                calibration_bundle_path=bundle_path,
                optical_config_path=opt_cfg,
            )
            print(f"Wrote CT calibration recording -> {rec}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — tip JSON still valid
            print(f"warning: failed to persist CT calibration recording ({exc})", file=sys.stderr)
    return result
