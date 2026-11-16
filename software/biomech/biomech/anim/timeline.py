# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Video time <-> master-clock time (an11 §2).

A :class:`Timeline` plays one or more master-clock segments (phases, or windows
inside a phase) back to back at ``speed`` x real time: 0.25 is slow motion,
4 a time-lapse. Between segments the video *cuts* (the gap between two phases
is never shown). Frame ``k`` shows the instant at video time ``k / fps``.
Segments are half-open in video time: the end of a segment that is followed by
another one maps to the first instant of the next.

Both directions are vectorized and exact inverses inside the segments, so a
frame index, a video time and a master-clock instant can always be converted
into each other (the hidden timecode of every frame is checked against
:meth:`Timeline.frame_times`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Timeline:
    segments: tuple[tuple[float, float], ...]     # master-clock [t0, t1] in us, play order
    fps: float = 30.0
    speed: float = 1.0
    labels: tuple[str, ...] = ()                  # one per segment (phase names)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("a timeline needs at least one segment")
        if self.fps <= 0 or self.speed <= 0:
            raise ValueError("fps and speed must be positive")
        for t0, t1 in self.segments:
            if not t1 > t0:
                raise ValueError(f"empty segment [{t0}, {t1}]")
        if self.labels and len(self.labels) != len(self.segments):
            raise ValueError("one label per segment")

    @classmethod
    def span(cls, t0_us: float, t1_us: float, fps: float = 30.0, speed: float = 1.0,
             label: str = "") -> "Timeline":
        return cls(((float(t0_us), float(t1_us)),), fps, speed, (label,) if label else ())

    # -- sizes ----------------------------------------------------------------------------
    @property
    def master_durations_s(self) -> np.ndarray:
        return np.array([(b - a) * 1e-6 for a, b in self.segments])

    @property
    def video_starts_s(self) -> np.ndarray:
        """Video time at which each segment starts."""
        d = self.master_durations_s / self.speed
        return np.r_[0.0, np.cumsum(d)[:-1]]

    @property
    def duration_s(self) -> float:
        """Video duration (s)."""
        return float(self.master_durations_s.sum() / self.speed)

    @property
    def n_frames(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))

    def label(self, seg: int) -> str:
        return self.labels[seg] if self.labels else ""

    # -- maps -----------------------------------------------------------------------------
    def segment_at(self, v_s: np.ndarray | float) -> np.ndarray:
        """Index of the segment playing at video time(s) ``v_s``."""
        v = np.asarray(v_s, dtype=np.float64)
        j = np.searchsorted(self.video_starts_s, v, side="right") - 1
        return np.clip(j, 0, len(self.segments) - 1)

    def to_master(self, v_s: np.ndarray | float) -> np.ndarray:
        """Master-clock time (us, float) shown at video time ``v_s`` (s)."""
        v = np.asarray(v_s, dtype=np.float64)
        j = self.segment_at(v)
        t0 = np.array([a for a, _ in self.segments])[j]
        t1 = np.array([b for _, b in self.segments])[j]
        t = t0 + (v - self.video_starts_s[j]) * self.speed * 1e6
        return np.minimum(t, t1)

    def to_video(self, t_us: np.ndarray | float) -> np.ndarray:
        """Video time (s) at which master time ``t_us`` is shown; NaN outside every segment."""
        t = np.asarray(t_us, dtype=np.float64)
        out = np.full(t.shape, np.nan)
        for j, (a, b) in enumerate(self.segments):
            sel = (t >= a) & (t <= b) & np.isnan(out)
            out[sel] = self.video_starts_s[j] + (t[sel] - a) / (self.speed * 1e6)
        return out

    def frame_times(self, frames: np.ndarray | range | None = None) -> np.ndarray:
        """Master-clock time (us) of each frame (default: all frames)."""
        k = np.arange(self.n_frames) if frames is None else np.asarray(list(frames)
                                                                       if isinstance(frames, range)
                                                                       else frames)
        return self.to_master(k / self.fps)

    def frame_segments(self, frames: np.ndarray | range | None = None) -> np.ndarray:
        k = np.arange(self.n_frames) if frames is None else np.asarray(list(frames)
                                                                       if isinstance(frames, range)
                                                                       else frames)
        return self.segment_at(k / self.fps)

    def frame_of(self, t_us: np.ndarray | float) -> np.ndarray:
        """Index of the frame showing master time ``t_us`` (nearest; -1 outside)."""
        v = self.to_video(t_us)
        k = np.where(np.isfinite(v), np.round(np.nan_to_num(v) * self.fps), -1)
        return np.minimum(k, self.n_frames - 1).astype(int)

    def clip_seconds(self, t_us: np.ndarray, seg: np.ndarray) -> np.ndarray:
        """Seconds since the start of the frame's segment (the on-screen clock)."""
        t0 = np.array([a for a, _ in self.segments])[np.asarray(seg)]
        return (np.asarray(t_us, dtype=np.float64) - t0) * 1e-6

    def chunks(self, n: int) -> list[range]:
        """Split the frames into about ``n`` contiguous chunks (for parallel rendering)."""
        n = max(1, min(int(n), self.n_frames))
        edges = np.linspace(0, self.n_frames, n + 1).round().astype(int)
        return [range(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
