#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""EMG, model-frame XYZ and 3D trajectory for one session.

Runs, in order:

1. ``jtsmm.tools.plot_emg`` — one PNG per channel
2. ``jtsmm.tools.run_jts_analysis`` — ``jaw_motion.h5`` and the 2D XYZ stack
   in the dental-model frame (``<session>_jts_smoothed.png``)
3. ``jts.plotly_visualization`` — 3D path of the smoothed landmark trajectory

The session directory must contain ``session.h5`` and a ``*_jts_config.json``.
Every step runs with this interpreter (``sys.executable``), so the jtsmm,
jaw_tracking (``jts``) and biomech packages must be installed in it.

Examples
--------
  ./run_jts_plots.py ~/jtsmm_data/session_s01_f_31_20250302T101000Z_00000002
  ./run_jts_plots.py /path/to/session.h5 --out /tmp/s01_jts
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from biomech.cohort import parse_session_name
from biomech.paths import results_root


def subject_id(session: Path) -> str | None:
    name = session.name if session.is_dir() else session.parent.name
    info = parse_session_name(name)
    return f"{info['pid']}_{info['sex']}_{info['age']}" if info else None


def default_out(session: Path, subject: str | None) -> Path:
    sid = subject or subject_id(session)
    if not sid:
        raise SystemExit(
            "could not read a subject id (s01_f_31) from the session name; pass --out"
        )
    return results_root() / "data" / sid / "jts"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "session",
        type=Path,
        help="session.h5 or the session directory that contains it",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: $BIOMECH_RESULTS_DIR or ~/experiments, "
        "then /data/<subject>/jts)",
    )
    ap.add_argument(
        "--subject",
        default=None,
        help="subject id used in the default output path (default: parsed from the session name)",
    )
    args = ap.parse_args(argv)

    session = args.session.expanduser().resolve()
    if not session.exists():
        print(f"error: not found: {session}", file=sys.stderr)
        return 1

    out = (args.out.expanduser().resolve() if args.out else default_out(session, args.subject))
    out.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    print(f"session: {session}")
    print(f"out:     {out}")

    run([py, "-m", "jtsmm.tools.plot_emg", str(session), "--out", str(out), "--no-show"], env)
    run(
        [py, "-m", "jtsmm.tools.run_jts_analysis", str(session), "--out", str(out), "--no-show"],
        env,
    )
    h5 = out / "jaw_motion.h5"
    if not h5.is_file():
        print(f"error: JTS did not write {h5}", file=sys.stderr)
        return 1
    run(
        [
            py,
            "-m",
            "jts.plotly_visualization",
            str(h5),
            "--no-show",
            "--no-frames",
            "-g",
            "T_model_origin_mand_landmark_t_smooth",
            "-o",
            str(out),
        ],
        env,
    )
    print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
