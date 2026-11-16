#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Offline replay of recorded multi-camera frames through the jaw tracker.

Use this to debug detection and pose estimation without live cameras. Pair with
``run.py --record-frames`` (or any directory matching :mod:`frame_store` layout).

Example::

    # Capture during live session
    python software/optical/optical/tracking/run.py \\
        --config software/optical/etc/optical.yaml \\
        --record-frames data/optical/replay/lab_capture_01

    # Replay with per-frame diagnostics
    python software/optical/optical/tracking/replay.py \\
        --config software/optical/etc/optical.yaml \\
        --frames data/optical/replay/lab_capture_01 \\
        --print-every 10

    # Compare PnP vs triangulation on the same capture
    python software/optical/optical/tracking/replay.py \\
        --config software/optical/etc/optical.yaml \\
        --frames data/optical/replay/lab_capture_01 \\
        --compare-methods \\
        --print-every 30

    # Replay and write a tracking HDF5 (same schema as live capture)
    python software/optical/optical/tracking/replay.py \\
        --config software/optical/etc/optical.yaml \\
        --frames data/optical/replay/lab_capture_01 \\
        --output data/optical/tracking/replay_test.h5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    from optical.paths import tracking_session_path
    from optical.tracking.debug import (
        FrameDebugReport,
        MethodCompareReport,
        compare_methods_frame,
        report_from_last_frame,
        summarize_compare_reports,
        summarize_reports,
    )
    from optical.tracking.frame_store import (
        iter_frames,
        load_frame_timestamps,
        load_session_meta,
        session_frame_count,
    )
    from optical.tracking.io import FrameRecord, write_tracking_session
    from optical.tracking.tracker import apply_ct_tip, load_ct_tip_offset, load_tracker_for_recording
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from optical.paths import tracking_session_path
    from optical.tracking.debug import (
        FrameDebugReport,
        MethodCompareReport,
        compare_methods_frame,
        report_from_last_frame,
        summarize_compare_reports,
        summarize_reports,
    )
    from optical.tracking.frame_store import (
        iter_frames,
        load_frame_timestamps,
        load_session_meta,
        session_frame_count,
    )
    from optical.tracking.io import FrameRecord, write_tracking_session
    from optical.tracking.tracker import apply_ct_tip, load_ct_tip_offset, load_tracker_for_recording


def _print_session_header(frames_dir: Path, meta: dict, *, total: int, tcfg, extra: str = "") -> None:
    print(f"  frames: {total}, cams: {meta.get('cam_ids')}")
    if extra:
        print(f"  {extra}")
    print(f"  tracker config: {tcfg.marker_config_path}")
    print(f"  calibration: {tcfg.calibration_bundle_path}")
    size = meta.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        print(f"  intrinsics scaled to {int(size[0])}x{int(size[1])} ({meta.get('stream', 'meta')})")


def _frame_stamps(
    stamps: dict,
    stored_idx: int,
    raw: dict,
    rate: Optional[float],
) -> tuple:
    """``(t_cam_ns, t_master_us)`` for a stored frame.

    Recorded stamps when the dump has them; otherwise times synthesised from
    the stored frame index at the nominal rate (never the replay machine's
    clock, which made the HDF5 rate the replay speed).
    """
    if stored_idx in stamps:
        t_cam, t_master = stamps[stored_idx]
        if t_cam:
            return dict(t_cam), t_master
    if rate:
        t_ns = int(round(stored_idx * 1e9 / rate))
        return {cam_id: t_ns for cam_id in raw}, None
    return {}, None


def run_compare(
    *,
    tracker_config_path: Path,
    frames_dir: Path,
    print_every: int,
    verbose: bool,
    max_frames: Optional[int],
) -> None:
    """Replay the same frames through PnP and triangulation for side-by-side comparison."""
    frames_dir = Path(frames_dir)
    meta = load_session_meta(frames_dir)
    total = session_frame_count(frames_dir)
    if total == 0:
        raise SystemExit(f"No frames found under {frames_dir / 'frames'}")

    tracker, tcfg = load_tracker_for_recording(tracker_config_path, frames_dir)

    print(f"Replay compare: {frames_dir}")
    _print_session_header(
        frames_dir,
        meta,
        total=total,
        tcfg=tcfg,
        extra=f"config pose_method={tcfg.pose_method} (compare runs both PnP and triangulation)",
    )
    print("=" * 70)

    reports: List[MethodCompareReport] = []
    t0 = time.monotonic()

    for i, (_stored_idx, raw) in enumerate(iter_frames(frames_dir)):
        if max_frames is not None and i >= max_frames:
            break

        report = compare_methods_frame(tracker, raw, frame_idx=i)
        reports.append(report)

        if print_every > 0 and (i % print_every == 0):
            print(report.format_line(verbose=verbose))

    elapsed = time.monotonic() - t0
    print("=" * 70)
    print(summarize_compare_reports(reports))
    print(f"Elapsed: {elapsed:.2f}s ({len(reports) / max(elapsed, 1e-6):.1f} fps)")


def run_replay(
    *,
    tracker_config_path: Path,
    frames_dir: Path,
    output_path: Path | None,
    experiment_name: str,
    print_every: int,
    verbose: bool,
    max_frames: Optional[int],
    debug_only: bool,
) -> None:
    """Replay stored frames through the tracker."""
    frames_dir = Path(frames_dir)
    meta = load_session_meta(frames_dir)
    total = session_frame_count(frames_dir)
    if total == 0:
        raise SystemExit(f"No frames found under {frames_dir / 'frames'}")

    tracker, tcfg = load_tracker_for_recording(tracker_config_path, frames_dir)
    tip_meta: dict = {}
    tip_body_mm = None
    try:
        ct_tip = load_ct_tip_offset(tcfg)
    except FileNotFoundError as exc:
        # Replays often run on another machine; warn instead of refusing.
        print(f"warning: {exc}; CT stays at the body origin", file=sys.stderr)
        ct_tip = None
        tip_meta["ct_tip_source"] = f"missing: {tcfg.ct_tip_calibration_path}"
    if ct_tip is not None:
        tip_body_mm, tip_meta = ct_tip

    print(f"Replay: {frames_dir}")
    _print_session_header(
        frames_dir,
        meta,
        total=total,
        tcfg=tcfg,
        extra=f"method: {tcfg.pose_method}",
    )
    if debug_only:
        print("  mode: debug only (no HDF5 output)")
    elif output_path is not None:
        print(f"  output: {output_path}")
    print("=" * 70)

    reports: List[FrameDebugReport] = []
    records: List[FrameRecord] = []
    stamps = load_frame_timestamps(frames_dir)
    rate = meta.get("frame_rate_hz")
    rate = float(rate) if rate else None
    if stamps:
        print(f"  timestamps: {len(stamps)} recorded (timestamps.jsonl)")
    elif rate:
        print(f"  timestamps: none recorded; synthesised at {rate:g} Hz from meta.json")
    else:
        print("  timestamps: none recorded and no frame_rate_hz; HDF5 rate will be 0")
    t0 = time.monotonic()

    # Row == live frame_idx: the live take's JTS intervals index those rows, and
    # the async frame dump may have dropped PNGs (timestamps.jsonl keeps them).
    # A frame the dump lacks becomes a NaN row with its recorded time.
    expected = 0
    missing = 0
    for i, (stored_idx, raw) in enumerate(iter_frames(frames_dir)):
        if max_frames is not None and i >= max_frames:
            break
        while expected < stored_idx:
            t_cam_ns, t_master_us = _frame_stamps(stamps, expected, raw, rate)
            if not debug_only:
                records.append(
                    FrameRecord(frame_idx=expected, t_cam_ns=t_cam_ns, t_master_us=t_master_us)
                )
            missing += 1
            expected += 1
        expected = stored_idx + 1

        t_cam_ns, t_master_us = _frame_stamps(stamps, stored_idx, raw, rate)
        # One detect + pose per frame; the debug report reuses that result.
        record = tracker.process_frame(
            raw, frame_idx=stored_idx, t_cam_ns=t_cam_ns, t_master_us=t_master_us
        )
        report = report_from_last_frame(tracker, frame_idx=stored_idx)
        reports.append(report)

        if print_every > 0 and (i % print_every == 0):
            print(report.format_line(verbose=verbose))

        if not debug_only:
            apply_ct_tip(record, tip_body_mm)
            records.append(record)

    elapsed = time.monotonic() - t0
    print("=" * 70)
    if missing:
        print(f"  {missing} frame(s) absent from the dump written as NaN rows (row == live frame_idx)")
    print(summarize_reports(reports))
    print(f"Elapsed: {elapsed:.2f}s ({len(reports) / max(elapsed, 1e-6):.1f} fps)")

    if debug_only:
        return

    out_path = output_path or tracking_session_path(experiment_name)
    write_tracking_session(
        out_path,
        frames=records,
        frame_rate_hz=rate,
        rig_id=tracker.rig_id,
        experiment_name=experiment_name,
        method=tcfg.pose_method,
        notes=f"offline replay from {frames_dir}",
        extra_meta={
            "replay_timestamps": "recorded" if stamps else ("synthesised" if rate else "none"),
            **tip_meta,
        },
    )
    print(f"Wrote {len(records)} frames to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Replay recorded camera frames through the jaw tracker (debug mode)."
    )
    ap.add_argument(
        "--config",
        required=True,
        help="optical.yaml (default: software/optical/etc/optical.yaml)",
    )
    ap.add_argument(
        "--frames",
        required=True,
        type=Path,
        help="Replay session directory (contains meta.json and frames/)",
    )
    ap.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Write tracking HDF5 to this path (default: data/optical/tracking/<experiment>.h5)",
    )
    ap.add_argument(
        "--experiment-name",
        default="replay",
        help="Experiment name for HDF5 metadata and default output filename",
    )
    ap.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print a debug line every N frames (0 = summary only)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Include detected marker IDs in printed lines",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process at most this many frames",
    )
    ap.add_argument(
        "--compare-methods",
        action="store_true",
        help="Run PnP and triangulation on each frame and compare (implies --debug-only)",
    )
    ap.add_argument(
        "--debug-only",
        action="store_true",
        help="Run diagnostics only; do not write tracking HDF5",
    )
    args = ap.parse_args()

    if args.compare_methods:
        if args.output is not None:
            raise SystemExit("--compare-methods cannot be combined with --output")
        run_compare(
            tracker_config_path=Path(args.config),
            frames_dir=args.frames,
            print_every=args.print_every,
            verbose=args.verbose,
            max_frames=args.max_frames,
        )
        return

    run_replay(
        tracker_config_path=Path(args.config),
        frames_dir=args.frames,
        output_path=Path(args.output) if args.output else None,
        experiment_name=args.experiment_name,
        print_every=args.print_every,
        verbose=args.verbose,
        max_frames=args.max_frames,
        debug_only=args.debug_only,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted (Ctrl+C).")
        sys.exit(130)
