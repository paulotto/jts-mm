#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Trueness: hand-measured ladder slots vs reconstructed array pose.

One recording per slot. The measurement window is the first 400 frames
(10 s) of the experiment-plan section in that take (landmark phases are
ignored). The ladder is placed along depth
(lateral, world z) and left–right (frontal, world x), and each of those
placements is repeated at near, mid, and far.

The ladder may sit at an angle to the camera axes. For each range and
orientation the slot positions are compared with a straight line parameterized
by the hand-measured distance in millimetres (direction free, scale fixed).
A rigid tilt of that line is absorbed. The plotted x/y/z values are the
remaining residual in the world frame. The scalar distance error is
``||p − p_ref||`` minus the hand-measured spacing from the reference slot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from array_pose import (
    AXES,
    AXIS_INDEX,
    axis_sigma_mm,
    collect_body_poses,
    load_experiment_sections,
    match_slot,
    parse_ladder_leaf,
    section_interval_for_slot,
    warn_missing_body,
)
from optical.config.loader import default_config_path, load_config_file
from optical.evaluation.recordings import (
    iter_recording_leaves,
    write_json,
)
from optical.tracking.tracker import load_tracker_for_recording

STUDY_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / STUDY_DIR.name

METHODS = (("pnp", "PnP"), ("triangulation", "Triangulation"))
# First 10 s at 40 Hz. Frames past this in a section are not evaluated.
SECTION_FRAMES = 400


def _named_leaves(recordings_dir: Path, token: str) -> List[Path]:
    token = token.lower()
    return [p for p in iter_recording_leaves(recordings_dir) if token in p.name.lower()]


def _ladder_cfg(study: dict) -> tuple[str, float, List[str], Dict[str, dict], List[dict]]:
    body = str(study.get("array_body") or "").strip()
    if not body:
        raise SystemExit("study.yaml: set array_body to the two-marker array (not the CRA)")
    ladder = study.get("ladder")
    if not isinstance(ladder, dict):
        raise SystemExit("study.yaml: missing ladder:")
    reference_mm = float(ladder.get("reference_mm", 0.0))
    ranges = [str(r) for r in (ladder.get("ranges") or ["near", "mid", "far"])]
    if not ranges:
        raise SystemExit("study.yaml: ladder.ranges is empty")
    orientations = ladder.get("orientations") or {}
    if not isinstance(orientations, dict) or not orientations:
        raise SystemExit("study.yaml: ladder.orientations must define lateral and frontal")
    slots = ladder.get("slots") or []
    if not isinstance(slots, list) or not slots:
        raise SystemExit("study.yaml: ladder.slots is empty")
    parsed_slots = []
    for slot in slots:
        if not isinstance(slot, dict) or "section" not in slot or "mm" not in slot:
            raise SystemExit(f"study.yaml: bad ladder slot {slot!r}")
        parsed_slots.append({"section": str(slot["section"]), "mm": float(slot["mm"])})
    parsed_orient: Dict[str, dict] = {}
    for name, spec in orientations.items():
        if not isinstance(spec, dict) or spec.get("axis") not in AXIS_INDEX:
            raise SystemExit(
                f"study.yaml: orientation {name!r} needs axis x, y, or z"
            )
        parsed_orient[str(name)] = {
            "axis": str(spec["axis"]),
            "label": str(spec.get("label") or spec["axis"]),
        }
    return body, reference_mm, ranges, parsed_orient, parsed_slots


def _reference_slot(slots: List[dict], reference_mm: float) -> dict:
    exact = [s for s in slots if abs(s["mm"] - reference_mm) < 1e-6]
    if exact:
        return exact[0]
    return min(slots, key=lambda s: abs(s["mm"] - reference_mm))


def _summarize(ts: np.ndarray) -> dict | None:
    if ts.ndim != 2 or ts.shape[0] < 1:
        return None
    mean = np.mean(ts, axis=0)
    return {
        "n": int(ts.shape[0]),
        "t_mm": [float(v) for v in mean],
        "axis_sigma_mm": axis_sigma_mm(ts),
    }


def _select_leaves(
    leaves: List[Path],
    orientations: Dict[str, dict],
    ranges: List[str],
    slots: List[dict],
) -> List[dict]:
    """One sample per (range, orientation, slot): the newest matching leaf."""
    chosen: Dict[tuple[str, str, str], dict] = {}
    for leaf in leaves:
        rng, orient, token = parse_ladder_leaf(leaf, orientations, ranges)
        if orient is None or orient not in orientations:
            print(f"Skipping {leaf.name}: no lateral/frontal orientation in the name")
            continue
        if rng is None or rng not in ranges:
            print(f"Skipping {leaf.name}: no near/mid/far range in the name")
            continue
        sections = load_experiment_sections(leaf)
        slot = match_slot(token, slots) if token else None
        if slot is None and token:
            print(f"Skipping {leaf.name}: {token!r} is not a ladder slot")
            continue
        if slot is None:
            # Distance lives in the section name; one recording may hold one section.
            measurement = {
                name: interval
                for name, interval in sections.items()
                if match_slot(name, slots) is not None
            }
            if not measurement:
                print(f"Skipping {leaf.name}: no ladder section in the experiment plan")
                continue
            targets = [(match_slot(name, slots), name, interval) for name, interval in measurement.items()]
        else:
            name, interval = section_interval_for_slot(sections, slot, slots)
            if interval is None and sections:
                print(
                    f"Skipping {leaf.name}: experiment plan has no section for "
                    f"{slot['section']!r}"
                )
                continue
            if interval is None:
                print(
                    f"Warning: {leaf.name} has no experiment-plan section; "
                    f"using the whole take for {slot['section']}"
                )
            targets = [(slot, name, interval)]

        for slot_i, section_name, interval in targets:
            if slot_i is None:
                continue
            key = (rng, orient, str(slot_i["section"]))
            sample = {
                "range": rng,
                "orientation": orient,
                "slot": slot_i,
                "leaf": leaf,
                "section": section_name,
                "interval": interval,
            }
            prev = chosen.get(key)
            if prev is None or leaf.name > prev["leaf"].name:
                chosen[key] = sample
    return list(chosen.values())


def _evaluate_sample(sample: dict, *, tracker, tcfg, body: str, pnp_camera_id: str) -> dict:
    n_frames, packed = collect_body_poses(
        sample["leaf"],
        tracker=tracker,
        tcfg=tcfg,
        bodies=[body],
        pnp_camera_id=pnp_camera_id,
        frame_interval=sample["interval"],
        max_frames=SECTION_FRAMES,
    )
    poses = packed[body]
    methods = {
        "pnp": _summarize(poses["pnp_t"]),
        "triangulation": _summarize(poses["tri_t"]),
    }
    interval = sample["interval"]
    return {
        "recording": str(sample["leaf"]),
        "range": sample["range"],
        "orientation": sample["orientation"],
        "section": sample["section"],
        "slot": sample["slot"]["section"],
        "hand_mm": float(sample["slot"]["mm"]),
        "frame_interval": list(interval) if interval is not None else None,
        "n_frames": n_frames,
        "methods": methods,
    }


def _metric_line(hand_mm: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``p ≈ c + s * u`` with ``||u|| = 1``.

    ``s`` is the hand-measured distance in millimetres, so the scale stays
    fixed to the ruler. Only the direction of the ladder is free, which
    removes a rigid tilt from the residual.
    """
    s = np.asarray(hand_mm, dtype=np.float64).reshape(-1)
    p = np.asarray(points, dtype=np.float64)
    if s.shape[0] < 2 or p.shape != (s.shape[0], 3):
        raise ValueError("need at least two 3D slot positions")
    ds = s - float(np.mean(s))
    if float(np.dot(ds, ds)) < 1e-12:
        raise ValueError("hand-measured distances are identical")
    v = p.T @ ds
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("ladder direction is undefined")
    u = v / n
    c = np.mean(p, axis=0) - float(np.mean(s)) * u
    return c, u


def _errors_for_orientation(rows: List[dict], *, reference: dict) -> List[dict]:
    """World residual to the metric ladder line, plus spacing from the reference slot."""
    ref_row = next((r for r in rows if r["slot"] == reference["section"]), None)
    fits: Dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for key, _label in METHODS:
        usable = [r for r in rows if r["methods"].get(key)]
        if len(usable) < 2:
            fits[key] = None
            continue
        hand = np.array([float(r["hand_mm"]) for r in usable], dtype=np.float64)
        pts = np.stack([np.asarray(r["methods"][key]["t_mm"], dtype=np.float64) for r in usable])
        try:
            fits[key] = _metric_line(hand, pts)
        except ValueError as exc:
            print(f"Warning: {key} line fit skipped ({exc})")
            fits[key] = None

    out = []
    for row in sorted(rows, key=lambda r: r["hand_mm"]):
        methods = {}
        for key, _label in METHODS:
            measured = row["methods"].get(key)
            fit = fits.get(key)
            if not measured or fit is None:
                methods[key] = None
                continue
            c, u = fit
            p = np.asarray(measured["t_mm"], dtype=np.float64)
            s = float(row["hand_mm"])
            residual = p - (c + s * u)
            distance_error = None
            if ref_row is not None and ref_row["methods"].get(key):
                p_ref = np.asarray(ref_row["methods"][key]["t_mm"], dtype=np.float64)
                distance_error = float(
                    np.linalg.norm(p - p_ref) - (s - float(reference["mm"]))
                )
            methods[key] = {
                "error_mm": {
                    axis_name: float(residual[i]) for axis_name, i in AXIS_INDEX.items()
                },
                "distance_error_mm": distance_error,
                "along_mm": float(np.dot(residual, u)),
                "direction_u": [float(v) for v in u],
                "axis_sigma_mm": measured["axis_sigma_mm"],
                "n": measured["n"],
            }
        out.append(
            {
                "slot": row["slot"],
                "section": row["section"],
                "hand_mm": row["hand_mm"],
                "recording": row["recording"],
                "n_frames": row["n_frames"],
                "methods": methods,
            }
        )
    return out


def _plot_method(ax, rows: List[dict], axis: str, key: str, label: str, color: str) -> None:
    xs = [r["hand_mm"] for r in rows]
    ys = []
    yerr = []
    for row in rows:
        block = (row["methods"] or {}).get(key) or {}
        err = (block.get("error_mm") or {}).get(axis, np.nan)
        sigma = (block.get("axis_sigma_mm") or {}).get(axis, np.nan)
        ys.append(err)
        yerr.append(0.0 if sigma != sigma else float(sigma))
    ax.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=3, label=label, color=color)


def plot_trueness(
    orientation: str,
    spec: dict,
    by_range: Dict[str, List[dict]],
    ranges: List[str],
    out: Path,
) -> Path | None:
    """Error vs hand-measured slot distance. Rows are near/mid/far, columns x/y/z."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not available — skipped trueness plot")
        return None

    n = max(1, len(ranges))
    fig, axes = plt.subplots(n, 3, figsize=(11.2, 3.15 * n), sharex=True, squeeze=False)
    for row_i, rng in enumerate(ranges):
        rows = by_range.get(rng) or []
        for col, axis in enumerate(AXES):
            ax = axes[row_i, col]
            if rows:
                _plot_method(ax, rows, axis, "pnp", "PnP", "C0")
                _plot_method(ax, rows, axis, "triangulation", "Triangulation", "C1")
                ax.legend(fontsize=8, loc="best")
            else:
                ax.text(0.5, 0.5, "no recording", ha="center", va="center", transform=ax.transAxes)
            ax.axhline(0.0, color="0.5", lw=0.8)
            ax.set_title(f"{rng} — {axis}")
            if row_i == n - 1:
                ax.set_xlabel("hand-measured distance (mm)")
            if col == 0:
                ax.set_ylabel("residual (mm)")
            ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Trueness — {orientation} ({spec['label']}; residual to metric line)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _fmt_err(block: dict | None) -> str:
    if not block:
        return "n/a"
    err = block["error_mm"]
    dist = block.get("distance_error_mm")
    dist_txt = "n/a" if dist is None else f"{dist:+.3f}"
    return (
        f"dist {dist_txt}  "
        f"x {err['x']:+.3f}  y {err['y']:+.3f}  z {err['z']:+.3f} mm"
        f"  (n={block['n']})"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Trueness: ladder slot distances, lateral and frontal"
    )
    ap.add_argument("--study", type=Path, default=STUDY_DIR / "study.yaml")
    ap.add_argument("--recordings", type=Path, default=HOME_STUDY / "recordings")
    ap.add_argument(
        "--orientations",
        default=None,
        help="Comma-separated subset (default: every ladder.orientations entry)",
    )
    ap.add_argument(
        "--ranges",
        default=None,
        help="Comma-separated subset (default: ladder.ranges, near/mid/far)",
    )
    ap.add_argument("--output", type=Path, default=HOME_STUDY / "results" / "trueness.json")
    ap.add_argument(
        "--plot-dir",
        type=Path,
        default=HOME_STUDY / "results",
        help="Writes trueness_<orientation>.png",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    study = load_config_file(args.study)
    cfg = Path(study["optical_config"]) if study.get("optical_config") else default_config_path()
    body, reference_mm, ranges, orientations, slots = _ladder_cfg(study)
    if args.orientations:
        want = [c.strip() for c in args.orientations.split(",") if c.strip()]
        missing = [c for c in want if c not in orientations]
        if missing:
            raise SystemExit(f"Unknown orientation(s): {', '.join(missing)}")
        orientations = {k: orientations[k] for k in want}
    if args.ranges:
        range_want = [c.strip() for c in args.ranges.split(",") if c.strip()]
        missing = [c for c in range_want if c not in ranges]
        if missing:
            raise SystemExit(f"Unknown range(s): {', '.join(missing)}")
        ranges = range_want
    pnp_cam = str(study.get("pnp_camera_id", "cam0"))
    reference = _reference_slot(slots, reference_mm)
    range_order = {name: i for i, name in enumerate(ranges)}

    leaves = _named_leaves(args.recordings, "trueness")
    if not leaves:
        raise SystemExit(
            f"No recordings with 'trueness' in the name under {args.recordings}"
        )
    selected = _select_leaves(leaves, orientations, ranges, slots)
    if not selected:
        raise SystemExit(f"No trueness recordings matched the ladder slots under {args.recordings}")

    print(f"Array body: {body} (not the CRA)")
    print(f"Ranges: {', '.join(ranges)}")
    print(
        f"Reference slot: {reference['section']} = {reference['mm']:g} mm "
        f"(per range and orientation)"
    )
    reports = []
    warned = False
    for sample in sorted(
        selected,
        key=lambda s: (s["orientation"], range_order.get(s["range"], 0), s["slot"]["mm"]),
    ):
        interval = sample["interval"]
        span = "whole take" if interval is None else f"frames [{interval[0]}, {interval[1]}]"
        print(
            f"Evaluating {sample['leaf'].name}  "
            f"[{sample['range']} {sample['orientation']} "
            f"{sample['slot']['section']} = {sample['slot']['mm']:g} mm]  "
            f"section={sample['section']!r}  {span}, first {SECTION_FRAMES} frames …"
        )
        tracker, tcfg = load_tracker_for_recording(cfg, sample["leaf"])
        if not warned:
            warn_missing_body(tracker, body)
            warned = True
        reports.append(
            _evaluate_sample(
                sample,
                tracker=tracker,
                tcfg=tcfg,
                body=body,
                pnp_camera_id=pnp_cam,
            )
        )

    grouped: Dict[str, Dict[str, List[dict]]] = {name: {} for name in orientations}
    for row in reports:
        grouped.setdefault(row["orientation"], {}).setdefault(row["range"], []).append(row)

    orient_payload = {}
    plot_paths = []
    for name, spec in orientations.items():
        range_payload = {}
        plot_ranges: Dict[str, List[dict]] = {}
        print(f"\n=== {name} ({spec['label']}, intended along {spec['axis']}) ===")
        for rng in ranges:
            rows = grouped.get(name, {}).get(rng) or []
            if not rows:
                print(f"Warning: no trueness recordings for {rng} {name}")
                continue
            if not any(r["slot"] == reference["section"] for r in rows):
                print(
                    f"Warning: {rng} {name} has no reference slot "
                    f"{reference['section']!r}; spacing error will be empty"
                )
            errors = _errors_for_orientation(rows, reference=reference)
            range_payload[rng] = {"slots": errors}
            plot_ranges[rng] = errors
            print(f"\n{rng}")
            print(f"{'slot':<8} {'hand mm':>8}  {'PnP':<52}  Triangulation")
            for row in errors:
                print(
                    f"{row['slot']:<8} {row['hand_mm']:8.2f}  "
                    f"{_fmt_err(row['methods'].get('pnp')):<36}  "
                    f"{_fmt_err(row['methods'].get('triangulation'))}"
                )
        orient_payload[name] = {
            "axis": spec["axis"],
            "label": spec["label"],
            "reference_slot": reference["section"],
            "reference_mm": reference["mm"],
            "ranges": range_payload,
        }
        if plot_ranges and not args.no_plot:
            path = plot_trueness(
                name,
                spec,
                plot_ranges,
                ranges,
                args.plot_dir / f"trueness_{name}.png",
            )
            if path is not None:
                plot_paths.append(str(path))
                print(f"Wrote {path}")

    payload = {
        "study": "5.1.1_trueness",
        "array_body": body,
        "config": str(cfg),
        "ranges": ranges,
        "reference_slot": reference["section"],
        "reference_mm": reference["mm"],
        "section_frames": SECTION_FRAMES,
        "world_frame": "cam0 (x right, y down, z depth)",
        "orientations": orient_payload,
        "plots": plot_paths,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
