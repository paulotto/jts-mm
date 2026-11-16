# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Queued viewer-JPEG sidecar for experiment-plan recordings.

Capture stays on the hot path: ``offer()`` is non-blocking and drops frames if
the writer falls behind. Files are written only for frames the supervisor
hands over (active phases, including landmark calibration).
"""

from __future__ import annotations

import json
import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jtsmm.io.session_layout import PREVIEW_INDEX_NAME, preview_dir, preview_index_path

_QUEUE_MAX = 32


def _slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", (text or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "phase"


@dataclass(frozen=True)
class PreviewJpegItem:
    cam_id: str
    jpeg: bytes
    t_master_us: int
    phase_index: int
    phase_name: str
    phase_kind: str


class PreviewJpegDump:
    """Write 320×240-class JPEGs under ``preview/`` plus ``preview_index.jsonl``."""

    INDEX_NAME = PREVIEW_INDEX_NAME

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = Path(session_dir)
        self.root = preview_dir(self.session_dir)
        self.index_path = preview_index_path(self.session_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.written = 0
        self.dropped = 0
        self.write_errors = 0
        self._seq: dict[tuple[int, str], int] = {}
        self._q: queue.Queue[PreviewJpegItem | None] = queue.Queue(maxsize=_QUEUE_MAX)
        self._thread = threading.Thread(
            target=self._run, name="preview-jpeg-dump", daemon=True
        )
        self._thread.start()

    def offer(
        self,
        *,
        cam_id: str,
        jpeg: bytes,
        t_master_us: int,
        phase_index: int,
        phase_name: str,
        phase_kind: str,
    ) -> bool:
        """Queue one already-encoded JPEG. Never blocks the caller."""
        if not jpeg:
            return False
        item = PreviewJpegItem(
            cam_id=str(cam_id),
            jpeg=bytes(jpeg),
            t_master_us=int(t_master_us),
            phase_index=int(phase_index),
            phase_name=str(phase_name),
            phase_kind=str(phase_kind),
        )
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self) -> dict[str, Any]:
        """Flush queued frames and stop the writer thread."""
        try:
            self._q.put(None, timeout=2.0)
        except queue.Full:
            self.dropped += max(0, self._q.qsize())
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=5.0)
        return {
            "written": int(self.written),
            "dropped": int(self.dropped),
            "write_errors": int(self.write_errors),
            "index": str(self.index_path.name),
        }

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self._write_one(item)
            except Exception:
                self.write_errors += 1

    def _write_one(self, item: PreviewJpegItem) -> None:
        key = (item.phase_index, item.cam_id)
        seq = self._seq.get(key, 0)
        self._seq[key] = seq + 1
        folder = self.root / f"{item.phase_index:02d}_{_slug(item.phase_name)}"
        folder.mkdir(parents=True, exist_ok=True)
        rel = folder / f"{item.cam_id}_{seq:06d}.jpg"
        rel.write_bytes(item.jpeg)
        rec = {
            "t_master_us": item.t_master_us,
            "cam_id": item.cam_id,
            "phase": item.phase_name,
            "kind": item.phase_kind,
            "path": str(rel.relative_to(self.session_dir)),
            "bytes": len(item.jpeg),
        }
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.written += 1
