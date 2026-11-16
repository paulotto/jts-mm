# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Chunked replay of a recorded take through a strictly causal chain (X4, ml08 §2).

:func:`replay` cuts the stored EMG into chunks of ``chunk_s`` worth of samples (default 20 ms)
and hands each chunk, plus the IMU samples stamped at or before its last EMG sample, to an
:class:`OnlineProcessor`. The processor only ever sees the chunks delivered so far; it never
holds the stored arrays, so it cannot read ahead.

Per chunk the processor

1. marks rail (``|x| >= rail_frac`` x ADC full scale, non-finite) and lead-off samples
   (pointwise) and grows the mask *causally* (:func:`causal_dilate`: a masked sample masks the
   next ``k`` samples, never earlier ones; offline the mask grows on both sides);
2. holds masked raw samples at the last valid value (:func:`ml.core.signals.hold_fill` with the
   carried last valid row) and runs the stateful :class:`CausalChain`
   (:class:`jtsmm.viewer.processing.CausalEmgFilter`, tracking mains canceller + band-pass);
   masked samples become NaN; the output goes to a ring buffer;
3. appends the IMU samples to their buffers;
4. after the ``rest`` phase has ended (and only then) calibrates: ZC/SSC/WAMP thresholds
   (:func:`ml.core.features.rest_thresholds`) and the D13 rest-relative statistics
   (:func:`ml.core.features.rest_relative_stats`) from the rest-phase windows, exactly as
   :func:`ml.core.dataset.build_windows` does, and the closer rest level of the release
   detector. The rest windows themselves are produced at that moment (``retro``);
5. schedules trailing windows on the offline grid (window ends ``win - 1 + k stride`` samples
   after a phase's first sample, inside the phase) and emits a window once its EMG has arrived
   and every IMU stream has a sample at or after the window's last IMU grid point (the grid
   value is interpolated between bracketing samples, as offline); after ``imu_max_wait_s`` it
   is emitted anyway (``imu_late``). Features come from
   :func:`ml.core.features.emg_window_features`, :func:`ml.core.signals.imu_signals` and
   :func:`ml.core.features.imu_window_features`, the functions of the offline windows, on the
   same samples;
6. optionally decodes the emitted window (``predict``) and feeds the release detector.

``filter_mode``:

``continuous`` (default)
    one filter from the stream start: what a device would run.
``phase_warmup``
    additionally one filter per phase started ``warmup_s`` before the phase, as the offline
    windows are computed; the windows of a phase read that filter. Only for the equivalence
    test: it needs the phase schedule ``warmup_s`` in advance, so it is not causal with
    respect to the schedule.

The mains canceller re-locks its oscillator only at recording gaps (where the offline
``process_timed`` call splits the block), not after every chunk; the output is then
independent of the chunk size and equals the offline chain on the same samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterator

import numpy as np

from ml.config.classes import REST_PHASE
from ml.core import features as F
from ml.core import signals as S
from ml.core.dataset import Imu, WindowConfig
from ml.core.windows import gap_windows, window_ends, window_labels

FILTER_MODES = ("continuous", "phase_warmup")
IMU_KEYS = ("q_mta", "q_cra", "a_mta", "a_cra")
# The release detector's closers, in biomech's order (biomech.io.session.CLOSERS).
CLOSERS = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l")
_FAR = np.int64(-(10 ** 15))


@dataclass(frozen=True)
class StreamConfig:
    chunk_s: float = 0.02                 # EMG samples per chunk (20 ms)
    dilate_s: float = 0.1                 # causal (hold-only) mask growth
    rail_frac: float = 0.99
    flatline_s: float = 0.0               # causal flat-line rule (0 = off, as ml_windows)
    filter_mode: str = "continuous"       # continuous | phase_warmup (equivalence test only)
    imu_max_wait_s: float = 0.1           # emit a window without its IMU after this
    buffer_s: float = 2.0                 # EMG ring buffer length

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "StreamConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown stream settings {bad}; known: {sorted(known)}")
        out = cls(**d)
        if out.filter_mode not in FILTER_MODES:
            raise ValueError(f"filter_mode must be one of {FILTER_MODES}, got {out.filter_mode!r}")
        if out.chunk_s <= 0:
            raise ValueError("chunk_s must be > 0")
        return out


@dataclass
class StreamInput:
    """One stored take, as the replay source (only :func:`replay` reads these arrays)."""

    t_us: np.ndarray                       # (N,) int64, strictly increasing
    uv: np.ndarray                         # (N, C) raw EMG, uV
    fs: float
    channels: list[str]
    phases: list[tuple[str, int, int]]     # (name, t_start_us, t_end_us), time order
    line_hz: float
    full_scale_uv: np.ndarray | None = None
    leadoff: np.ndarray | None = None      # (N, C) bool lead-off bit per sample
    imu: Imu | None = None                 # sanitized streams


@dataclass
class Chunk:
    index: int
    t: np.ndarray                          # (n,) int64
    uv: np.ndarray                         # (n, C)
    leadoff: np.ndarray | None             # (n, C) bool
    imu: dict[str, tuple[np.ndarray, np.ndarray]]   # samples arrived with this chunk


def iter_chunks(inp: StreamInput, chunk_s: float) -> Iterator[Chunk]:
    """Chunks of ``round(chunk_s * fs)`` EMG samples; each carries the IMU samples stamped at
    or before its last EMG sample and after the previous chunk's (all earlier ones first)."""
    n = max(int(round(chunk_s * inp.fs)), 1)
    t = np.asarray(inp.t_us, dtype=np.int64)
    ptr = dict.fromkeys(IMU_KEYS, 0)
    for c, i0 in enumerate(range(0, t.size, n)):
        i1 = min(i0 + n, t.size)
        T = t[i1 - 1]
        imu: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if inp.imu is not None:
            for k in IMU_KEYS:
                tk, vk = getattr(inp.imu, k)
                j = int(np.searchsorted(tk, T, side="right"))
                imu[k] = (np.asarray(tk[ptr[k]:j], np.int64), np.asarray(vk[ptr[k]:j], float))
                ptr[k] = j
        yield Chunk(c, t[i0:i1], np.asarray(inp.uv[i0:i1], dtype=np.float64),
                    None if inp.leadoff is None else np.asarray(inp.leadoff[i0:i1], bool), imu)


# -- causal masks --------------------------------------------------------------------------
def causal_dilate(hit: np.ndarray, k: int, last_rel: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Hold-only mask growth: a hit masks itself and the next ``k`` samples, never earlier ones.

    ``last_rel`` (C,) is the index of the previous block's last hit relative to this block's
    first row (negative; ``None`` = none yet). Returns the mask and the carry for the next block.
    """
    hit = np.asarray(hit, dtype=bool)
    n, c = hit.shape
    prev = np.full(c, _FAR) if last_rel is None else np.asarray(last_rel, np.int64)
    if n == 0:
        return hit.copy(), prev
    idx = np.where(hit, np.arange(n, dtype=np.int64)[:, None], _FAR)
    idx[0] = np.maximum(idx[0], prev)
    np.maximum.accumulate(idx, axis=0, out=idx)
    m = (np.arange(n, dtype=np.int64)[:, None] - idx) <= max(int(k), 0)
    return m, np.maximum(idx[-1] - n, _FAR)


def causal_flatline(history: np.ndarray, x: np.ndarray, fs: float, window_s: float
                    ) -> np.ndarray:
    """Rows of ``x`` that end a zero-variance stretch of ``window_s`` (trailing window).

    ``history`` holds the preceding raw rows (at least ``window - 1`` once available). The
    offline :func:`ml.core.signals.flatline_mask` uses a centred window and grows the hit
    over the whole stretch; causally a flat stretch is known only once it has lasted
    ``window_s``, so only the samples from then on are marked.
    """
    x = np.asarray(x, dtype=np.float64)
    if window_s <= 0 or x.shape[0] == 0:
        return np.zeros(x.shape, dtype=bool)
    w = max(2, int(round(window_s * fs)))
    full = np.vstack([np.asarray(history, float)[-(w - 1):], x]) if len(history) else x
    if full.shape[0] < w:
        return np.zeros(x.shape, dtype=bool)
    from numpy.lib.stride_tricks import sliding_window_view

    W = sliding_window_view(full, w, axis=0)[-x.shape[0]:]          # (<=n, C, w)
    mean = W.mean(axis=-1)
    flat = W.var(axis=-1) <= 1e-12 * np.maximum(1.0, mean ** 2)
    out = np.zeros(x.shape, dtype=bool)
    out[x.shape[0] - flat.shape[0]:] = flat
    return out


# -- causal EMG chain -----------------------------------------------------------------------
class CausalChain:
    """:class:`~jtsmm.viewer.processing.CausalEmgFilter` fed chunk by chunk.

    Gaps in the timestamps (more than 1.5 sample periods) advance the mains oscillator as
    ``process_timed`` does. The oscillator re-lock (``HarmonicCanceller._relock``) is deferred
    to those gaps: offline, ``process_timed`` processes a gap-free stretch in one call, so the
    re-lock fires only where a gap splits the block; per-chunk re-locking would make the output
    depend on the chunk size. Output float32 values as float64, like
    :func:`ml.core.signals.causal_emg`.
    """

    def __init__(self, fs: float, n_channels: int, line_hz: float, band_hz=(20.0, 450.0),
                 gap_factor: float = 1.5) -> None:
        from jtsmm.viewer.processing import CausalEmgFilter, HarmonicCanceller

        self.f = CausalEmgFilter(fs, n_channels, hpf_hz=band_hz[0], lpf_hz=band_hz[1],
                                 notch_hz=line_hz, mains_mode="track")
        self._relock = HarmonicCanceller._relock
        self._canc = [s for s in getattr(self.f, "_sections", [])
                      if isinstance(s, HarmonicCanceller)]
        for c in self._canc:
            c._relock = self._deferred(c)            # instance attribute shadows the method
        self.dt = 1e6 / float(fs)
        self.gap = float(gap_factor) * self.dt
        self._last: float | None = None

    @staticmethod
    def _deferred(c: Any) -> Callable[[int], None]:
        def acc(n: int) -> None:
            c._since_lock += int(n)
        return acc

    def _at_gap(self, missing: int) -> None:
        for c in self._canc:
            self._relock(c, 0)                       # the re-lock offline does at this point
        self.f.advance(missing)

    def process(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        n = x.shape[0]
        out = np.empty(x.shape, dtype=np.float64)
        if n == 0:
            return out
        if self._last is not None and t[0] - self._last > self.gap:
            self._at_gap(int(round((t[0] - self._last) / self.dt)) - 1)
        cuts = np.flatnonzero(np.diff(t) > self.gap) + 1
        start = 0
        for c in list(cuts) + [n]:
            y, _ = self.f.process(x[start:c])
            out[start:c] = y
            if c < n:
                self._at_gap(int(round((t[c] - t[c - 1]) / self.dt)) - 1)
            start = c
        self._last = float(t[-1])
        return out


class _Ring:
    """The last ``keep`` (or more) rows of a track, addressed by global sample index."""

    def __init__(self, n_channels: int, keep: int) -> None:
        self.keep = max(int(keep), 1)
        self.g0 = 0
        self.t = np.zeros(0, dtype=np.int64)
        self.x = np.zeros((0, n_channels))

    def append(self, g: int, t: np.ndarray, x: np.ndarray) -> None:
        if self.t.size == 0:
            self.g0 = g
        elif g != self.g0 + self.t.size:
            raise ValueError("ring buffer fed out of order")
        self.t = np.concatenate([self.t, t])
        self.x = np.concatenate([self.x, x])
        if self.t.size > 2 * self.keep:
            drop = self.t.size - self.keep
            self.t, self.x, self.g0 = self.t[drop:], self.x[drop:], self.g0 + drop

    def get(self, e: int, win: int) -> tuple[np.ndarray, np.ndarray]:
        i = e - self.g0
        if i - win + 1 < 0 or i >= self.t.size:
            raise IndexError(f"window ending at {e} is not in the ring buffer")
        return self.t[i - win + 1:i + 1], self.x[i - win + 1:i + 1]

    def time(self, e: int) -> int:
        return int(self.t[e - self.g0])


class _Track:
    """Mask -> hold fill -> causal chain -> NaN for masked samples, for samples in
    ``[t_from, t_to]``; the mask is applied from ``mask_from`` on (offline: within the phase)."""

    def __init__(self, fs: float, n_ch: int, line_hz: float, band, k_dil: int, keep: int,
                 t_from: float = -np.inf, t_to: float = np.inf, mask_from: float = -np.inf,
                 flatline_s: float = 0.0) -> None:
        self.chain = CausalChain(fs, n_ch, line_hz, band)
        self.fs, self.k = fs, k_dil
        self.t_from, self.t_to, self.mask_from = t_from, t_to, mask_from
        self.last_valid = np.zeros(n_ch)
        self.last_hit: np.ndarray | None = None
        self.flatline_s = flatline_s
        self._raw_hist = np.zeros((0, n_ch))
        self.ring = _Ring(n_ch, keep)

    def feed(self, g: int, t: np.ndarray, raw: np.ndarray, hit: np.ndarray
             ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray] | None:
        sel = np.flatnonzero((t >= self.t_from) & (t <= self.t_to))
        if sel.size == 0:
            return None
        a, b = int(sel[0]), int(sel[-1]) + 1
        t, raw, hit = t[a:b], raw[a:b], hit[a:b]
        if self.flatline_s > 0:
            hit = hit | causal_flatline(self._raw_hist, raw, self.fs, self.flatline_s)
            w = max(2, int(round(self.flatline_s * self.fs)))
            self._raw_hist = np.vstack([self._raw_hist, raw])[-w:]
        hit = hit & (t >= self.mask_from)[:, None]
        m, self.last_hit = causal_dilate(hit, self.k, self.last_hit)
        filled = S.hold_fill(np.vstack([self.last_valid[None, :], raw]),
                             np.vstack([np.zeros((1, raw.shape[1]), bool), m]))[1:]
        if filled.shape[0]:
            self.last_valid = filled[-1].copy()
        x = np.where(m, np.nan, self.chain.process(t, filled))
        self.ring.append(g + a, t, x)
        return g + a, t, x, m


# -- the processor ---------------------------------------------------------------------------
@dataclass
class _Phase:
    name: str
    t0: int
    t1: int
    p0: int | None = None        # global index of its first sample
    k: int = 0                   # next window number
    done: bool = False


@dataclass
class _Pending:
    phase: _Phase
    e: int                       # global end index
    t_end: int
    t_emg: int                   # chunk time at which its EMG was complete
    chunk_emg: int


@dataclass
class Calibration:
    """What the stream learned from the rest phase (all of it before the tasks)."""

    t_us: int = -1               # chunk time at which it became available
    chunk: int = -1
    ref_uv: np.ndarray | None = None          # rest RMS (features are % of it)
    thr: np.ndarray | None = None
    center: np.ndarray | None = None
    scale: np.ndarray | None = None
    rest_level_uv: np.ndarray | None = None   # release detector (closers)
    n_rest_windows: int = 0
    rest_precedes_tasks: bool = True
    error: str = ""


@dataclass
class StreamResult:
    """Every emitted window (time order of emission) plus the chunk log."""

    windows: dict[str, np.ndarray]
    X_emg: np.ndarray            # D13 rest-relative (NaN when uncalibrated)
    X_emg_raw: np.ndarray        # before D13 normalization (% of the rest RMS)
    X_imu: np.ndarray
    names: dict[str, list[str]]
    calibration: Calibration
    chunk_t_us: np.ndarray       # last EMG sample of each chunk
    chunk_ns: np.ndarray         # processing time per chunk (perf_counter_ns)
    release_us: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.windows["t_end_us"].size)

    def decodable(self) -> np.ndarray:
        w = self.windows
        return w["calibrated"] & ~w["bad_mask"] & ~w["gap"] & w["finite"]

    def attach_labels(self, labels: dict[str, tuple[np.ndarray, np.ndarray]],
                      cfg: WindowConfig) -> np.ndarray:
        """Window labels (``window_labels``, offline rule); evaluation only, never an input."""
        w = self.windows
        y = np.full(len(self), -1, dtype=np.int64)
        for name in np.unique(w["phase"]):
            sel = np.flatnonzero(w["phase"] == name)
            if name not in labels:
                continue
            lt, lab = labels[name]
            y[sel], _ = window_labels(w["t_end_us"][sel], lt, lab, cfg.window_s * 1e6,
                                      mode=cfg.label_at, purity_min=cfg.purity_min,
                                      max_age_us=cfg.label_max_age_s * 1e6)
        w["y"] = y
        return y


class OnlineProcessor:
    """Stateful causal chain; :meth:`feed` takes one :class:`Chunk` at a time."""

    def __init__(self, *, fs: float, channels: list[str], phases: list[tuple[str, int, int]],
                 line_hz: float, wcfg: WindowConfig, scfg: StreamConfig,
                 full_scale_uv: np.ndarray | None = None, has_imu: bool = True,
                 predict: Callable[[np.ndarray, np.ndarray | None], int] | None = None,
                 release_params: Any = None) -> None:
        if wcfg.emg_norm != "rest_relative":
            raise ValueError("the stream implements the D13 rest-relative features only "
                             "(window.emg_norm: rest_relative); 'mvc' needs a reference "
                             "recorded before the take")
        self.fs, self.channels, self.wcfg, self.scfg = float(fs), list(channels), wcfg, scfg
        self.C = len(channels)
        self.line_hz = float(line_hz)
        self.fsc = None if full_scale_uv is None else np.asarray(full_scale_uv, float)
        self.has_imu = has_imu
        self.predict = predict
        self.win = int(round(wcfg.window_s * fs))
        self.stride = int(round(wcfg.stride_s * fs))
        self.w_us = wcfg.window_s * 1e6
        self.iwin = int(round(wcfg.window_s * wcfg.imu_hz))
        self.imu_p = int(1e6 / wcfg.imu_hz)
        self.names = {"emg": F.emg_feature_names(self.channels, wcfg.emg_features),
                      "imu": F.imu_feature_names()}
        keep = int(scfg.buffer_s * fs) + self.win + int((scfg.imu_max_wait_s + scfg.chunk_s) * fs)
        self._keep = keep
        self.k_dil = int(round(scfg.dilate_s * fs))
        band = wcfg.emg_band_hz
        self.main = _Track(fs, self.C, line_hz, band, self.k_dil, keep,
                           flatline_s=scfg.flatline_s)
        self.phases = [_Phase(n, int(a), int(b)) for n, a, b in phases]
        self.rest = next((p for p in self.phases if p.name == REST_PHASE), None)
        self.tracks: dict[str, _Track] = {}
        self._warm = float(wcfg.warmup_s) * 1e6
        self.calib = Calibration()
        if self.rest is not None:
            others = [p.t0 for p in self.phases if p.name != REST_PHASE]
            self.calib.rest_precedes_tasks = not others or self.rest.t1 < min(others)
        self._rest_t: list[np.ndarray] = []
        self._rest_x: list[np.ndarray] = []        # feature track
        self._rest_main: list[np.ndarray] = []     # main track (release level)
        self.imu = {k: _Grow(4 if k[0] == "q" else 3) for k in IMU_KEYS}
        self._imu_cache: dict[str, tuple[int, np.ndarray]] = {}
        self.pending: list[_Pending] = []
        self.out: dict[str, list] = {k: [] for k in (
            "t_end_us", "t_dec_us", "t_dec_emg_us", "chunk", "chunk_emg", "phase", "e",
            "retro", "calibrated", "bad_mask", "gap", "finite", "finite_imu", "imu_late",
            "yhat_inloop")}
        self.X_emg: list[np.ndarray] = []
        self.X_raw: list[np.ndarray] = []
        self.X_imu: list[np.ndarray] = []
        self.n_seen = 0
        self.T = -1
        self.chunk = -1
        self.release_params = release_params
        self.release = None
        idx = [self.channels.index(m) for m in CLOSERS if m in self.channels]
        self.closers = idx
        self._rel_from = np.inf

    # -- helpers ---------------------------------------------------------------------------
    def _track_for(self, ph: _Phase) -> _Track:
        return self.tracks.get(ph.name, self.main)

    def _imu_ready(self, g_time: int) -> bool:
        if not self.has_imu:
            return True
        return all(self.imu[k].n and self.imu[k].last() >= g_time for k in IMU_KEYS)

    def _imu_grid(self, ph: _Phase, t_end: int) -> tuple[int, int]:
        A = ph.t0 - int(self.w_us)                    # offline: arange(t0 - int(W), t1 + 1, P)
        return A, (int(t_end) - A) // self.imu_p

    def _imu_last_grid_t(self, ph: _Phase, t_end: int) -> int:
        """Time of the last IMU grid point of the window ending at ``t_end``."""
        A, ge = self._imu_grid(ph, t_end)
        return A + ge * self.imu_p

    def _imu_rows(self, A: int, lo_k: int, hi_k: int) -> np.ndarray:
        """IMU signals (12) at grid indices ``lo_k..hi_k`` of the grid anchored at ``A``.

        Each grid value depends only on the bracketing IMU samples and the previous grid point
        (backward difference), so rows computed in pieces equal the offline whole-phase grid
        (grid index 0 keeps the offline w = 0 of the first point)."""
        start = max(lo_k - 1, 0)
        grid = A + self.imu_p * np.arange(start, hi_k + 1, dtype=np.int64)
        if grid.size < 2:
            return np.full((hi_k - lo_k + 1, 12), np.nan)
        streams = []
        for k in IMU_KEYS:
            tk, vk = self.imu[k].view()
            lo = max(int(np.searchsorted(tk, grid[0], side="right")) - 2, 0)
            hi = int(np.searchsorted(tk, grid[-1], side="right")) + 2    # bracket after the end
            streams.append((tk[lo:hi], vk[lo:hi]))
        return S.imu_signals(grid, *streams)[lo_k - start:]

    def _imu_features(self, ph: _Phase, t_end: int, cache: bool = True) -> np.ndarray:
        if not self.has_imu:
            return np.full(25, np.nan)
        A, ge = self._imu_grid(ph, t_end)
        k0 = ge - self.iwin + 1
        if k0 < 0:
            return np.full(25, np.nan)                 # as offline (window before the grid)
        c = self._imu_cache.get(ph.name)
        if cache and c is not None and c[0] <= k0 and c[0] + len(c[1]) >= k0:
            first, rows = c
            last = first + len(rows) - 1
            if ge > last:
                rows = np.vstack([rows, self._imu_rows(A, last + 1, ge)])
            rows = rows[k0 - first:]
            first = k0
        else:
            first, rows = k0, self._imu_rows(A, k0, ge)
        if cache:
            self._imu_cache[ph.name] = (first, rows)
        win = rows[:self.iwin]
        return F.imu_window_features(win, np.array([self.iwin - 1]), self.iwin)[0]

    def _emg_features(self, xw: np.ndarray, tw: np.ndarray) -> tuple[np.ndarray, float, bool]:
        xs = 100.0 * xw / self.calib.ref_uv[None, :]
        fe, nanfrac = F.emg_window_features(xs, self.fs, np.array([self.win - 1]), self.win,
                                            self.calib.thr, self.wcfg.emg_features)
        gap = bool((tw[-1] - tw[0]) > self.wcfg.gap_factor * self.w_us)
        return fe[0], float(nanfrac.max()), gap

    def _emit(self, ph: _Phase, t_end: int, e: int, t_emg: int, chunk_emg: int,
              fe: np.ndarray | None, maskfrac: float, gap: bool, fi: np.ndarray,
              retro: bool, imu_late: bool) -> None:
        nf = len(self.names["emg"])
        cal = fe is not None and self.calib.center is not None
        if cal:
            xn = F.apply_rest_relative(fe[None, :], self.names["emg"], self.calib.center,
                                       self.calib.scale)[0]
            raw = fe
        else:
            xn = raw = np.full(nf, np.nan)
        bad = maskfrac > self.wcfg.max_masked
        fin_e = bool(np.isfinite(raw).all())
        fin_i = bool(np.isfinite(fi).all()) if self.has_imu else True
        o = self.out
        o["t_end_us"].append(int(t_end))
        o["t_dec_us"].append(int(self.T))
        o["t_dec_emg_us"].append(int(t_emg))
        o["chunk"].append(self.chunk)
        o["chunk_emg"].append(int(chunk_emg))
        o["phase"].append(ph.name)
        o["e"].append(int(e))
        o["retro"].append(bool(retro))
        o["calibrated"].append(bool(cal))
        o["bad_mask"].append(bool(bad))
        o["gap"].append(bool(gap))
        o["finite"].append(fin_e and fin_i)
        o["finite_imu"].append(fin_i)
        o["imu_late"].append(bool(imu_late))
        yhat = -1
        if self.predict is not None and cal and not bad and not gap and fin_e and fin_i:
            yhat = int(self.predict(xn, fi if self.has_imu else None))
        o["yhat_inloop"].append(yhat)
        self.X_emg.append(xn)
        self.X_raw.append(raw)
        self.X_imu.append(fi)

    # -- calibration from the rest phase ---------------------------------------------------------
    def _calibrate(self) -> None:
        c = self.calib
        c.t_us, c.chunk = int(self.T), self.chunk
        ph = self.rest
        t = np.concatenate(self._rest_t) if self._rest_t else np.zeros(0, np.int64)
        x = np.concatenate(self._rest_x) if self._rest_x else np.zeros((0, self.C))
        xm = np.concatenate(self._rest_main) if self._rest_main else x
        self._rest_t, self._rest_x, self._rest_main = [], [], []
        if t.size < self.win + 2:
            c.error = "rest phase shorter than one window"
            return
        ref = S.rest_reference(x)
        ref = np.where(np.isfinite(ref) & (ref > 0), ref, 1.0)
        c.ref_uv = ref
        xs = 100.0 * x / ref[None, :]
        c.thr = F.rest_thresholds(xs, self.wcfg.thr_sigma)
        ends = window_ends(t.size, self.win, self.stride)
        gap = gap_windows(t, ends, self.win, self.w_us, self.wcfg.gap_factor)
        fe, nanfrac = F.emg_window_features(xs, self.fs, ends, self.win, c.thr,
                                            self.wcfg.emg_features)
        valid = ~(nanfrac.max(axis=1) > self.wcfg.max_masked) & ~gap & np.isfinite(fe).all(1)
        try:
            c.center, c.scale = F.rest_relative_stats(fe[valid], self.names["emg"])
        except ValueError as exc:
            c.error = str(exc)
        c.n_rest_windows = int(valid.sum())
        if self.closers:
            env = S.causal_rms_envelope(xm[:, self.closers], self.fs,
                                        getattr(self.release_params, "window_s", 0.05))
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    lvl = np.nanmedian(env, axis=0)
            c.rest_level_uv = np.where(np.isfinite(lvl) & (lvl > 0), lvl, 1.0)
        # The rest windows, produced now (they could not be normalized earlier).
        for j, e_loc in enumerate(ends):
            te = int(t[e_loc])
            fi = self._imu_features(ph, te)
            self._emit(ph, te, -1, int(self.T), self.chunk,
                       fe[j] if c.center is not None else None, float(nanfrac[j].max()),
                       bool(gap[j]), fi, retro=True, imu_late=False)
        if self.release_params is not None and self.closers and c.rest_level_uv is not None:
            from ml.online.release import StreamRelease

            self.release = StreamRelease(self.fs, c.rest_level_uv, self.release_params)
            self._rel_from = ph.t1

    # -- one chunk -------------------------------------------------------------------------------
    def feed(self, ch: Chunk) -> None:
        t, raw = ch.t, ch.uv
        n = t.size
        if n == 0:
            return
        self.chunk, self.T = ch.index, int(t[-1])
        g = self.n_seen
        hit = S.rail_mask(raw, self.fsc, self.scfg.rail_frac)
        if ch.leadoff is not None:
            hit |= ch.leadoff
        raw = np.where(np.isfinite(raw), raw, 0.0)
        fed = {"__main__": self.main.feed(g, t, raw, hit)}
        if self.scfg.filter_mode == "phase_warmup":
            for ph in self.phases:
                if ph.name not in self.tracks and self.T >= ph.t0 - self._warm:
                    self.tracks[ph.name] = _Track(self.fs, self.C, self.line_hz,
                                                  self.wcfg.emg_band_hz, self.k_dil, self._keep,
                                                  t_from=ph.t0 - self._warm, t_to=ph.t1,
                                                  mask_from=ph.t0,
                                                  flatline_s=self.scfg.flatline_s)
                if ph.name in self.tracks:
                    fed[ph.name] = self.tracks[ph.name].feed(g, t, raw, hit)
        for k, (tk, vk) in ch.imu.items():
            self.imu[k].extend(tk, vk)
        self.n_seen += n

        for ph in self.phases:                         # phase starts (operator markers)
            if ph.p0 is None and self.T >= ph.t0:
                ph.p0 = g + int(np.searchsorted(t, ph.t0, side="left"))

        # Rest-phase samples for the calibration.
        if self.rest is not None and self.calib.t_us < 0:
            src = fed.get(self.rest.name) if self.rest.name in self.tracks else fed["__main__"]
            if src is not None:
                _, ts, xs, _ = src
                sel = (ts >= self.rest.t0) & (ts <= self.rest.t1)
                if sel.any():
                    self._rest_t.append(ts[sel])
                    self._rest_x.append(xs[sel])
                    if self.rest.name in self.tracks and fed["__main__"] is not None:
                        _, tm, xm, _ = fed["__main__"]
                        self._rest_main.append(xm[(tm >= self.rest.t0) & (tm <= self.rest.t1)])
            if self.T >= self.rest.t1:
                last = int(self._rest_t[-1][-1]) if self._rest_t else self.rest.t1
                late = self.T - self.rest.t1 > self.scfg.imu_max_wait_s * 1e6
                if self._imu_ready(self._imu_last_grid_t(self.rest, last)) or late:
                    self._calibrate()

        if self.release is not None and fed["__main__"] is not None:
            _, tm, xm, _ = fed["__main__"]
            sel = tm > self._rel_from
            if sel.any():
                self.release.feed(tm[sel], xm[sel][:, self.closers])

        # Schedule windows (offline grid, inside each phase; rest windows come from _calibrate).
        last_g = self.n_seen - 1
        for ph in self.phases:
            if ph.p0 is None or ph.done or ph is self.rest:
                continue
            while True:
                e = ph.p0 + self.win - 1 + ph.k * self.stride
                if e > last_g:
                    break
                te = self.main.ring.time(e)
                if te > ph.t1:
                    ph.done = True
                    break
                self.pending.append(_Pending(ph, e, te, self.T, self.chunk))
                ph.k += 1

        # Emit what is complete.
        keep = []
        for i, w in enumerate(self.pending):
            calibrated = self.calib.center is not None
            if not calibrated and (self.calib.t_us >= 0 or self.rest is None
                                   or w.t_end < self.rest.t0 or not self.calib.rest_precedes_tasks):
                # No statistics (yet, or ever): emitted undecodable at once.
                self._emit(w.phase, w.t_end, w.e, w.t_emg, w.chunk_emg, None, 1.0, False,
                           np.full(25, np.nan), retro=False, imu_late=False)
                continue
            if not calibrated:
                keep.append(w)
                continue
            ready = self._imu_ready(self._imu_last_grid_t(w.phase, w.t_end))
            late = self.T - w.t_end > self.scfg.imu_max_wait_s * 1e6
            if not ready and not late:
                keep = keep + self.pending[i:]
                break
            tw, xw = self._track_for(w.phase).ring.get(w.e, self.win)
            fe, mf, gap = self._emg_features(xw, tw)
            self._emit(w.phase, w.t_end, w.e, w.t_emg, w.chunk_emg, fe, mf, gap,
                       self._imu_features(w.phase, w.t_end, cache=ready), retro=False,
                       imu_late=not ready)
        self.pending = keep

    def result(self, chunk_t: np.ndarray, chunk_ns: np.ndarray) -> StreamResult:
        o = self.out
        w = {k: np.asarray(v) for k, v in o.items()}
        for k in ("t_end_us", "t_dec_us", "t_dec_emg_us", "e"):
            w[k] = w[k].astype(np.int64)
        w["phase"] = np.asarray(o["phase"], dtype=object).astype(str) if o["phase"] \
            else np.zeros(0, dtype="<U1")
        for k in ("retro", "calibrated", "bad_mask", "gap", "finite", "finite_imu", "imu_late"):
            w[k] = w[k].astype(bool)
        w["yhat_inloop"] = w["yhat_inloop"].astype(np.int64)
        order = np.argsort(w["t_end_us"], kind="stable")         # rest windows came late
        w = {k: v[order] for k, v in w.items()}
        nf = len(self.names["emg"])

        def stack(L: list[np.ndarray], d: int) -> np.ndarray:
            return np.vstack(L)[order] if L else np.zeros((0, d))
        rel = np.asarray(self.release.decisions, np.int64) if self.release is not None \
            else np.zeros(0, np.int64)
        return StreamResult(windows=w, X_emg=stack(self.X_emg, nf), X_emg_raw=stack(self.X_raw, nf),
                            X_imu=stack(self.X_imu, 25), names=self.names, calibration=self.calib,
                            chunk_t_us=np.asarray(chunk_t, np.int64),
                            chunk_ns=np.asarray(chunk_ns, np.int64), release_us=rel,
                            meta={"line_hz": self.line_hz, "filter_mode": self.scfg.filter_mode,
                                  "calibration_chunk": self.calib.chunk,
                                  "chunk_samples": int(round(self.scfg.chunk_s * self.fs))})


class _Grow:
    """Append-only (t, values) buffer with amortized growth."""

    def __init__(self, width: int) -> None:
        self.t = np.zeros(1024, dtype=np.int64)
        self.v = np.zeros((1024, width))
        self.n = 0

    def extend(self, t: np.ndarray, v: np.ndarray) -> None:
        m = len(t)
        if m == 0:
            return
        if self.n + m > self.t.size:
            cap = max(2 * self.t.size, self.n + m)
            self.t = np.concatenate([self.t[:self.n], np.zeros(cap - self.n, np.int64)])
            self.v = np.vstack([self.v[:self.n], np.zeros((cap - self.n, self.v.shape[1]))])
        self.t[self.n:self.n + m] = t
        self.v[self.n:self.n + m] = v
        self.n += m

    def last(self) -> int:
        return int(self.t[self.n - 1])

    def view(self) -> tuple[np.ndarray, np.ndarray]:
        return self.t[:self.n], self.v[:self.n]


def replay(inp: StreamInput, wcfg: WindowConfig = WindowConfig(),
           scfg: StreamConfig = StreamConfig(), *,
           predict: Callable[[np.ndarray, np.ndarray | None], int] | None = None,
           release_params: Any = None, threads: int | None = 1,
           on_chunk: Callable[[OnlineProcessor, Chunk], None] | None = None) -> StreamResult:
    """Replay ``inp`` chunk by chunk; per-chunk processing time with ``threads`` BLAS/OpenMP
    threads (1 = single thread, the reported compute time; None = leave as is)."""
    proc = OnlineProcessor(fs=inp.fs, channels=inp.channels, phases=inp.phases,
                           line_hz=inp.line_hz, wcfg=wcfg, scfg=scfg,
                           full_scale_uv=inp.full_scale_uv, has_imu=inp.imu is not None,
                           predict=predict, release_params=release_params)
    ts, ns = [], []

    def loop() -> None:
        for ch in iter_chunks(inp, scfg.chunk_s):
            t0 = time.perf_counter_ns()
            proc.feed(ch)
            ns.append(time.perf_counter_ns() - t0)
            ts.append(int(ch.t[-1]))
            if on_chunk is not None:
                on_chunk(proc, ch)

    if threads is None:
        loop()
    else:
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=int(threads)):
            loop()
    return proc.result(np.asarray(ts), np.asarray(ns))


__all__ = ["CausalChain", "Calibration", "Chunk", "OnlineProcessor", "StreamConfig",
           "StreamInput", "StreamResult", "causal_dilate", "causal_flatline", "iter_chunks",
           "replay"]
