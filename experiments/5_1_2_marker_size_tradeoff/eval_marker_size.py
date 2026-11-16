#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Marker size trade-off: triangulation detection + pose jitter vs size/distance.

PnP is intentionally omitted — ``5_1_1_pnp_vs_triangulation`` already showed single-camera PnP is
substantially less precise than stereo triangulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from optical.config.loader import default_config_path, load_config_file
from optical.evaluation.recordings import (
    iter_recording_leaves,
    iter_session_frames,
    parse_condition_from_leaf,
    write_json,
)
from optical.tracking.tracker import load_tracker_for_recording

STUDY_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / STUDY_DIR.name


def _find_marker(detection, marker_id: int):
    for m in detection.markers:
        if int(m.marker_id) == int(marker_id):
            return m
    return None


def _summarize_marker(
    *,
    marker_id: int,
    size_mm: float,
    label: str,
    total_frames: int,
    tri_flags: List[bool],
    tri_samples,
    side_px: List[float],
) -> dict:
    from optical.evaluation.metrics import (
        reprojection_rms_mm,
        rotation_jitter_deg,
        rotation_magnitude_deg,
    )

    detected = int(sum(tri_flags))
    rate = 100.0 * detected / total_frames if total_frames else 0.0
    sides = np.asarray(side_px, dtype=np.float64)

    if tri_samples:
        reproj_px = float(np.mean([s.reprojection_rms_px for s in tri_samples]))
        mean_side = float(np.mean(sides)) if len(sides) else 0.0
        reproj_mm = reprojection_rms_mm(
            reprojection_rms_px=reproj_px,
            size_mm=size_mm,
            side_px=mean_side,
        )
        fit = float(np.mean([s.fit_rms_mm for s in tri_samples]))
        t = np.stack([s.translation_mm for s in tri_samples], axis=0)
        trans_std = float(np.linalg.norm(np.std(t, axis=0)))
        radius = np.linalg.norm(t - np.mean(t, axis=0), axis=1)
        trans_r_mean = float(np.mean(radius))
        trans_sigma_r = float(np.std(radius))
        rvecs = np.stack([s.rotation_rvec for s in tri_samples], axis=0)
        rot_std = float(np.std(rotation_magnitude_deg(rvecs)))
        rot_geo = float(rotation_jitter_deg(rvecs))
    else:
        reproj_px = float("nan")
        reproj_mm = float("nan")
        fit = float("nan")
        trans_std = float("nan")
        trans_r_mean = float("nan")
        trans_sigma_r = float("nan")
        rot_std = float("nan")
        rot_geo = float("nan")

    return {
        "marker_id": marker_id,
        "size_mm": size_mm,
        "label": label,
        "total_frames": total_frames,
        "detected_frames": detected,
        "detection_rate_pct": rate,
        "mean_side_px": float(np.mean(sides)) if len(sides) else 0.0,
        "min_side_px": float(np.min(sides)) if len(sides) else 0.0,
        "translation_std_mm": trans_std,
        "translation_r_mean_mm": trans_r_mean,
        "translation_sigma_r_mm": trans_sigma_r,
        "rotation_jitter_deg": rot_std,
        "rotation_geodesic_rms_deg": rot_geo,
        "reprojection_rms_px_mean": reproj_px,
        "reprojection_rms_mm_mean": reproj_mm,
        "fit_rms_mm_mean": fit,
    }


def evaluate_leaf(
    leaf: Path,
    *,
    tracker,
    tcfg,
    markers: List[dict],
    max_frames: int | None,
) -> dict:
    from optical.evaluation.metrics import MarkerTriSample, marker_side_px
    from optical.evaluation.triangulation_eval import triangulate_marker_pose

    id_to_meta = {int(m["id"]): m for m in markers}
    tri_flags: Dict[int, List[bool]] = {mid: [] for mid in id_to_meta}
    tri_samples: Dict[int, List[MarkerTriSample]] = {mid: [] for mid in id_to_meta}
    side_px: Dict[int, List[float]] = {mid: [] for mid in id_to_meta}

    n_frames = 0
    for frame_idx, frames in iter_session_frames(leaf, max_frames=max_frames):
        n_frames += 1
        detections = tracker.detect_all(frames)

        for mid, meta in id_to_meta.items():
            size_mm = float(meta["size_mm"])

            sides = []
            for det in detections.values():
                m = _find_marker(det, mid)
                if m is not None:
                    sides.append(marker_side_px(m.corners))
            if sides:
                side_px[mid].append(float(min(sides)))

            tri_ok = False
            tri = triangulate_marker_pose(
                detections,
                marker_id=mid,
                size_mm=size_mm,
                intrinsics=tracker.intrinsics,
                extrinsics=tracker.extrinsics,
                min_views=tcfg.triangulation_min_views,
            )
            if tri is not None:
                rvec, _ = cv2.Rodrigues(tri.T_world_tag[:3, :3])
                tri_samples[mid].append(
                    MarkerTriSample(
                        frame_idx=frame_idx,
                        marker_id=mid,
                        size_mm=size_mm,
                        translation_mm=tri.translation_mm,
                        reprojection_rms_px=tri.reprojection_rms_px,
                        fit_rms_mm=tri.fit_rms_mm,
                        rotation_rvec=rvec.reshape(3),
                    )
                )
                tri_ok = True
            tri_flags[mid].append(tri_ok)

    results = []
    for mid, meta in id_to_meta.items():
        results.append(
            _summarize_marker(
                marker_id=mid,
                size_mm=float(meta["size_mm"]),
                label=str(meta.get("label", f"{meta['size_mm']}mm")),
                total_frames=n_frames,
                tri_flags=tri_flags[mid],
                tri_samples=tri_samples[mid],
                side_px=side_px[mid],
            )
        )

    return {
        "recording": str(leaf),
        "condition": parse_condition_from_leaf(leaf),
        "n_frames": n_frames,
        "method": "triangulation",
        "markers": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Marker size × distance (triangulation)")
    ap.add_argument("--study", type=Path, default=STUDY_DIR / "study.yaml")
    ap.add_argument("--recordings", type=Path, default=HOME_STUDY / "recordings")
    ap.add_argument("--conditions", default=None, help="Comma-separated filter")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--output", type=Path, default=HOME_STUDY / "results" / "marker_size.json")
    args = ap.parse_args(argv)

    study = load_config_file(args.study)
    cfg = Path(study["optical_config"]) if study.get("optical_config") else default_config_path()
    markers = list(study.get("markers") or [])
    if args.conditions:
        want = {c.strip() for c in args.conditions.split(",") if c.strip()}
    else:
        want = set(study.get("conditions") or ["near", "mid", "far"])

    reports = []
    for leaf in iter_recording_leaves(args.recordings):
        cond = parse_condition_from_leaf(leaf)
        if want and cond not in want:
            continue
        print(f"Evaluating {leaf.name} …")
        tracker, tcfg = load_tracker_for_recording(cfg, leaf)
        reports.append(
            evaluate_leaf(
                leaf,
                tracker=tracker,
                tcfg=tcfg,
                markers=markers,
                max_frames=args.max_frames,
            )
        )

    payload = {
        "study": "5.1.2_marker_size",
        "method": "triangulation",
        "config": str(cfg),
        "reports": reports,
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output} ({len(reports)} recording(s))")
    for r in reports:
        print(f"\n{r['condition']}  frames={r['n_frames']}  (triangulation)")
        for m in r["markers"]:
            print(
                f"  {m['label']}: det={m['detection_rate_pct']:.1f}% "
                f"side={m['mean_side_px']:.1f}px "
                f"r_t={m['translation_r_mean_mm']:.3f}±{m['translation_sigma_r_mm']:.3f} mm "
                f"R_geo={m['rotation_geodesic_rms_deg']:.3f}° "
                f"reproj={m['reprojection_rms_px_mean']:.3f} px "
                f"fit={m['fit_rms_mm_mean']:.3f} mm"
            )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
