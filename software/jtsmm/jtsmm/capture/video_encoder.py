# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ffmpeg sidecar video encoder (s02 §7)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_frames_to_mkv(
    frames: list[tuple[int, np.ndarray]],
    out_path: Path,
    *,
    fps: float = 60.0,
    codec: str = "ffv1",
) -> bool:
    """Encode a list of (frame_idx, ndarray) to MKV via ffmpeg rawvideo pipe."""
    if not frames or not ffmpeg_available():
        return False
    h, w = frames[0][1].shape[:2]
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        codec,
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for _, arr in frames:
            gray = arr if arr.ndim == 2 else arr[..., 0]
            proc.stdin.write(np.ascontiguousarray(gray, dtype=np.uint8).tobytes())
    finally:
        proc.stdin.close()
        proc.wait(timeout=120)
    return out_path.is_file() and proc.returncode == 0


def worker_video_encoder(
    cam_id: str,
    q_vid: Any,
    out_mkv: Path,
    stop: Any,
    *,
    fps: float = 60.0,
) -> None:
    """Drain video queue and encode at session end (batch mode for mock)."""
    frames: list[tuple[int, np.ndarray]] = []
    while not stop.is_set():
        try:
            item = q_vid.get(timeout=0.2)
        except Exception:
            continue
        if item is None:
            break
        frames.append(item)
    if frames:
        encode_frames_to_mkv(frames, out_mkv, fps=fps)
