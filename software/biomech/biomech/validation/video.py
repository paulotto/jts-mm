# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

""""Two systems" video of the validation study (master plan D7, 4.15): the mandible of
JTS-MM and of Qualisys (ghost) in each system's own maxillary frame, the error vector
between the two lower incisors, both incisor traces over time with a now-line, and the
recorded phase.

Reads the stored ``ref_kinematics`` arrays only (incisor and condyle points of both
systems at the JTS-MM frame times). Frames go from the Agg canvas into an ``ffmpeg``
rawvideo pipe (H.264, ``yuv420p``, CRF 18), as in :mod:`biomech.anim.render_video`.
Pseudonyms only; no camera images.

Usage::

    python -m biomech.validation.video --config experiments/jtsmm_validation/study.yaml \\
        --pid s10 --phase open_close --out two_systems_s10_open_close.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from biomech.viz import style
from biomech.viz.views import vocab as V


def _window(a: dict[str, np.ndarray], phase: str | None, t0: float | None, t1: float | None
            ) -> np.ndarray:
    t = np.asarray(a["t_s"], float)
    if phase:
        k = np.asarray(a["phase"]).astype(str) == phase
        if not k.any():
            raise LookupError(f"phase {phase!r} not in the take")
        return k
    lo = t[0] if t0 is None else t0
    hi = t[-1] if t1 is None else t1
    return (t >= lo) & (t <= hi)


def render(a: dict[str, np.ndarray], out: str | Path, *, who: str, phase: str | None = None,
           t0: float | None = None, t1: float | None = None, fps: float = 30.0,
           speed: float = 0.5, trail_s: float = 1.0, theme: str = "light",
           size: tuple[int, int] = (1280, 720)) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found")
    k = _window(a, phase, t0, t1)
    idx = np.flatnonzero(k)
    t = np.asarray(a["t_s"], float)
    ts = t[idx]
    P = {s: np.asarray(a[f"incisor_{s}_mm"], float)[idx] for s in ("s", "q")}
    C = {s: [np.asarray(a[f"{n}_{s}_mm"], float)[idx] for n in ("tmj_left", "tmj_right")
             if f"{n}_{s}_mm" in a] for s in ("s", "q")}
    ph = np.asarray(a["phase"]).astype(str)[idx]
    col_s = style.hexcolor("blue", theme)
    col_q = style.hexcolor("grey", theme)
    col_e = style.hexcolor("vermillion", theme)
    W, H = size
    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax_s = fig.add_axes((0.05, 0.38, 0.42, 0.55))
    ax_f = fig.add_axes((0.55, 0.38, 0.42, 0.55))
    ax_t = fig.add_axes((0.05, 0.07, 0.92, 0.22))
    allp = np.vstack([P["s"], P["q"]] + C["s"] + C["q"])
    for ax, (i, j), title in ((ax_s, (0, 2), "Sagittal (x anterior, z up)"),
                              (ax_f, (1, 2), "Frontal (y left, z up)")):
        lo = np.nanmin(allp[:, [i, j]], axis=0) - 5
        hi = np.nanmax(allp[:, [i, j]], axis=0) + 5
        span = max(hi - lo)
        c = (lo + hi) / 2
        ax.set_xlim(c[0] - span / 2, c[0] + span / 2)
        ax.set_ylim(c[1] - span / 2, c[1] + span / 2)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mm")
    art: dict[str, Any] = {}
    for ax, (i, j), key in ((ax_s, (0, 2), "s"), (ax_f, (1, 2), "f")):
        for s, col, ls in (("s", col_s, "-"), ("q", col_q, "--")):
            art[f"trail_{key}_{s}"], = ax.plot([], [], ls, color=col, lw=1.0, alpha=0.6)
            art[f"mand_{key}_{s}"], = ax.plot([], [], ls, color=col, lw=1.8,
                                              label="JTS-MM" if s == "s" else "Qualisys")
            art[f"inc_{key}_{s}"], = ax.plot([], [], "o", color=col, ms=5)
        art[f"err_{key}"], = ax.plot([], [], "-", color=col_e, lw=2.0, label="Δ incisor")
        art[f"ij_{key}"] = (i, j)
    ax_s.legend(loc="upper right", fontsize=8, frameon=False)
    for s, col, ls in (("s", col_s, "-"), ("q", col_q, "--")):
        ax_t.plot(ts, P[s][:, 2], ls, color=col, lw=0.8)
    ax_t.set_xlabel("time in take (s)")
    ax_t.set_ylabel("incisor z (mm)")
    now = ax_t.axvline(ts[0], color=col_e, lw=1.0)
    txt = fig.text(0.05, 0.965, "", fontsize=11)
    err_txt = fig.text(0.55, 0.965, "", fontsize=11)
    n_frames = int(np.ceil((ts[-1] - ts[0]) / speed * fps)) + 1
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt",
           "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in range(n_frames):
            tv = ts[0] + f * speed / fps
            n = int(np.clip(np.searchsorted(ts, tv), 0, len(ts) - 1))
            tr = (ts >= tv - trail_s) & (ts <= ts[n])
            for key in ("s", "f"):
                i, j = art[f"ij_{key}"]
                for s in ("s", "q"):
                    art[f"trail_{key}_{s}"].set_data(P[s][tr, i], P[s][tr, j])
                    pts = [P[s][n]] + [c[n] for c in C[s]]
                    if len(pts) == 3:          # incisor - left condyle - right condyle - incisor
                        poly = np.array([pts[0], pts[1], pts[2], pts[0]])
                    else:
                        poly = np.array(pts)
                    art[f"mand_{key}_{s}"].set_data(poly[:, i], poly[:, j])
                    art[f"inc_{key}_{s}"].set_data([P[s][n, i]], [P[s][n, j]])
                art[f"err_{key}"].set_data([P["s"][n, i], P["q"][n, i]],
                                           [P["s"][n, j], P["q"][n, j]])
            now.set_xdata([ts[n], ts[n]])
            e = float(np.linalg.norm(P["s"][n] - P["q"][n]))
            txt.set_text(f"{who} · {V.phase_label(ph[n])} · t = {ts[n]:6.2f} s "
                         f"(×{speed:g})")
            err_txt.set_text(f"|Δ incisor| = {e:5.1f} mm (own anatomical frames)")
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        plt.close(fig)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode})")
    return out


def main(argv: list[str] | None = None) -> None:
    from biomech.config import load_config
    from biomech.io.context import DataContext
    from biomech.viz.views.results_kinematics import who

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--pid", required=True)
    ap.add_argument("--phase", default=None)
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ctx = DataContext(load_config(args.config))
    a = ctx.arrays("ref_kinematics", args.pid)
    p = render(a, args.out, who=who(ctx, args.pid), phase=args.phase, t0=args.t0, t1=args.t1,
               fps=args.fps, speed=args.speed)
    print(p)


if __name__ == "__main__":
    main()
