# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Save and load synchronized multi-camera frames for offline tracker replay.

Layout::

    <session_dir>/
    ├── meta.json
    ├── timestamps.jsonl      one line per frame: frame_idx, t_master_us, t_cam_ns
    └── frames/
        ├── 000000/
        │   ├── cam0.png
        │   └── cam1.png
        └── 000001/
            └── ...

Frames are stored as PNG (RGB on disk via OpenCV BGR encoding).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

SCHEMA_VERSION = "1.0"
FRAMES_SUBDIR = "frames"
TIMESTAMPS_NAME = "timestamps.jsonl"


def _frame_dir(session_dir: Path, frame_idx: int) -> Path:
    return session_dir / FRAMES_SUBDIR / f"{frame_idx:06d}"


def write_session_meta(
    session_dir: Path,
    *,
    cam_ids: List[str],
    frame_rate_hz: float | None = None,
    notes: str = "",
    image_size: tuple[int, int] | None = None,
    stream: str | None = None,
    calibration_bundle: str | None = None,
    calibration_bundle_source: str | None = None,
) -> Path:
    """Write ``meta.json`` for a replay frame dump.

    ``frame_rate_hz`` is the nominal (trigger) rate; replay passes it to the
    HDF5 writer. When omitted, the rate is inferred from ``timestamps.jsonl``.
    ``cam_ids`` are the cameras that were active, which replay uses to pick
    the 1- / 2-camera profile.

    ``image_size`` / ``stream`` document capture resolution (e.g. lores 728×544).
    ``calibration_bundle`` is the leaf-relative snapshot used for offline replay.
    """
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / FRAMES_SUBDIR).mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cam_ids": list(cam_ids),
        "notes": notes,
    }
    if frame_rate_hz is not None:
        meta["frame_rate_hz"] = float(frame_rate_hz)
    if image_size is not None:
        meta["image_size"] = [int(image_size[0]), int(image_size[1])]
    if stream is not None:
        meta["stream"] = str(stream)
    if calibration_bundle is not None:
        meta["calibration_bundle"] = str(calibration_bundle)
    if calibration_bundle_source is not None:
        meta["calibration_bundle_source"] = str(calibration_bundle_source)
    meta_path = session_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


class FrameTimestampLog:
    """Append-only ``timestamps.jsonl`` next to the frame dump.

    Replay used to stamp frames with the replay machine's clock; the real
    ``SensorTimestamp`` / MCU ``t_master_us`` of each saved frame now travel
    with it.
    """

    def __init__(self, session_dir: Path) -> None:
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        self.path = session_dir / TIMESTAMPS_NAME
        self._fh = self.path.open("a", encoding="utf-8")

    def append(
        self,
        frame_idx: int,
        *,
        t_cam_ns: Dict[str, int] | None,
        t_master_us: int | None,
    ) -> None:
        rec = {
            "frame_idx": int(frame_idx),
            "t_master_us": None if t_master_us is None else int(t_master_us),
            "t_cam_ns": {str(k): int(v) for k, v in (t_cam_ns or {}).items()},
        }
        self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "FrameTimestampLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_frame_timestamps(session_dir: Path) -> Dict[int, Tuple[Dict[str, int], Optional[int]]]:
    """``{frame_idx: (t_cam_ns, t_master_us)}`` from ``timestamps.jsonl`` ({} if absent)."""
    path = Path(session_dir) / TIMESTAMPS_NAME
    out: Dict[int, Tuple[Dict[str, int], Optional[int]]] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn last line after a crash
        t_m = rec.get("t_master_us")
        out[int(rec["frame_idx"])] = (
            {str(k): int(v) for k, v in (rec.get("t_cam_ns") or {}).items()},
            None if t_m is None else int(t_m),
        )
    return out


def load_session_meta(session_dir: Path) -> dict:
    """Load ``meta.json`` from a replay session directory."""
    session_dir = Path(session_dir)
    meta_path = session_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing replay meta file: {meta_path}")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported replay schema {meta.get('schema_version')!r} in {meta_path}"
        )
    return meta


def write_frame(
    session_dir: Path,
    frame_idx: int,
    frames: Dict[str, np.ndarray],
) -> None:
    """Write one synchronized multi-camera frame set (gray or RGB)."""
    import cv2

    out_dir = _frame_dir(session_dir, frame_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam_id, img in frames.items():
        arr = np.asarray(img)
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
            to_write = arr if arr.ndim == 2 else arr[:, :, 0]
        else:
            to_write = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        path = out_dir / f"{cam_id}.png"
        if not cv2.imwrite(str(path), to_write):
            raise RuntimeError(f"Failed to write replay frame: {path}")


def load_frame(session_dir: Path, frame_idx: int, cam_ids: List[str]) -> Dict[str, np.ndarray]:
    """Load one synchronized frame set; missing camera files are skipped.

    Grayscale PNGs (lores Y) load as HxW uint8; colour as RGB HxWx3.
    """
    import cv2

    frame_dir = _frame_dir(session_dir, frame_idx)
    frames: Dict[str, np.ndarray] = {}
    for cam_id in cam_ids:
        path = frame_dir / f"{cam_id}.png"
        if not path.is_file():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read replay frame: {path}")
        if img.ndim == 2:
            frames[cam_id] = img
        else:
            frames[cam_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return frames


def list_frame_indices(session_dir: Path) -> List[int]:
    """Return sorted frame indices present under ``frames/``."""
    frames_root = Path(session_dir) / FRAMES_SUBDIR
    if not frames_root.is_dir():
        return []
    indices: List[int] = []
    for child in frames_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            indices.append(int(child.name))
    return sorted(indices)


def iter_frames(
    session_dir: Path,
    *,
    cam_ids: Optional[List[str]] = None,
) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
    """Yield ``(frame_idx, {cam_id: rgb})`` for each stored frame."""
    session_dir = Path(session_dir)
    meta = load_session_meta(session_dir)
    ids = list(cam_ids or meta.get("cam_ids", []))
    for frame_idx in list_frame_indices(session_dir):
        frames = load_frame(session_dir, frame_idx, ids)
        if frames:
            yield frame_idx, frames


def session_frame_count(session_dir: Path) -> int:
    return len(list_frame_indices(session_dir))


class AsyncFrameWriter:
    """Write replay PNGs on a background thread with a bounded queue.

    PNG-encoding the lores frames inside the capture loop cost a sizeable
    part of the 33 ms frame budget on the Pi and stalled capture whenever the
    SD card hiccupped. The loop now only enqueues; when the queue is full the
    frame set is left out of the dump (never out of tracking) and counted in
    ``dropped``.
    """

    def __init__(self, session_dir: Path, *, max_queue: int = 64) -> None:
        self.session_dir = Path(session_dir)
        self.written = 0
        self.dropped = 0
        self.failed = 0
        self.first_error: str | None = None
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, int(max_queue)))
        self._thread = threading.Thread(target=self._run, name="frame-dump", daemon=True)
        self._thread.start()

    def submit(self, frame_idx: int, frames: Dict[str, np.ndarray]) -> bool:
        try:
            self._q.put_nowait((int(frame_idx), dict(frames)))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            frame_idx, frames = item
            try:
                write_frame(self.session_dir, frame_idx, frames)
                self.written += 1
            except Exception as exc:  # noqa: BLE001 - never kill the writer
                self.failed += 1
                if self.first_error is None:
                    self.first_error = f"frame {frame_idx}: {exc}"

    def close(self, *, timeout_s: float = 30.0) -> None:
        """Finish queued frames (up to ``timeout_s``) and stop the thread."""
        if not self._thread.is_alive():
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                self._q.put(None, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def stats(self) -> dict:
        return {
            "written": int(self.written),
            "dropped": int(self.dropped),
            "failed": int(self.failed),
            "pending": int(self._q.qsize()),
        }
