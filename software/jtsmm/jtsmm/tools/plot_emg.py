#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Plot session EMG the way the live viewer does: filtered, raw, envelope.

Reads only ``/emg`` (``samples_uv``). Processing matches ``jtsmm.viewer.hub.Hub``:
``CausalEmgFilter`` at native Hz with the display LPF
(``min(450, 0.9 * display_fs/2)`` → 225 Hz at 500 Hz), then decimation to
``TARGET_DISPLAY_FS`` (raw as min/max band, filt/env strided). Channels match
the viewer: six muscles + DRL (CH7 bias_debug dropped). Short envelope
spikes above 150 µV are removed from the filtered trace and its envelope.
**One 3×1 figure per channel.**

When the session has phases, only in-phase samples are drawn (gaps between
SPACE intervals are dropped). Landmark calibration phases are omitted unless
you pass them with ``--phases``. ``--full`` keeps the whole take.

Examples::

    python -m jtsmm.tools.plot_emg path/to/session.h5
    python -m jtsmm.tools.plot_emg session.h5 --out viz/ --no-show
    python -m jtsmm.tools.plot_emg session.h5 --channel masseter_l masseter_r
    python -m jtsmm.tools.plot_emg session.h5 --phases open_close chewing
    python -m jtsmm.tools.plot_emg session.h5 --full
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from jtsmm.config.catalog import (
    VIEWER_EMG_PLOT_NAMES,
    VIEWER_EMG_STREAM_COLS,
)
from jtsmm.io.session_reader import EmgStream, Session
from jtsmm.tools.plot_mp_trajectory import (
    PhaseInterval,
    _PHASE_COLORS,
    load_phases,
    motion_phases,
)
from jtsmm.tools.plot_optical_phases import _slug, resolve_session_h5
from jtsmm.viewer.hub import TARGET_DISPLAY_FS
from jtsmm.emg.outliers import suppress_envelope_outliers
from jtsmm.viewer.processing import EMG_BAND_HZ, CausalEmgFilter, rectified_envelope
from jtsmm.viewer.sources import _slice_emg_for_viewer

_CHUNK = 65_536
_MAX_PLOT_POINTS = 40_000
# The causal chain starts from zero state: electrode DC offsets (tens of mV)
# ring through the 20 Hz HPF and the 5 Hz envelope LPF for a few hundred ms.
# Kept out of the y-limits when the take is plotted from its first sample.
_FILTER_SETTLE_S = 1.0
_EMG_COLORS = (
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#2c3e50",
)


@dataclass
class PlotSpan:
    name: str
    t0_s: float
    t1_s: float


@dataclass
class EmgPlotData:
    t_s: np.ndarray
    raw_lo: np.ndarray  # (N, C) µV viewer min-band
    raw_hi: np.ndarray  # (N, C) µV viewer max-band
    filtered: np.ndarray
    envelope: np.ndarray
    names: list[str]
    fs_hz: float
    display_fs: float
    t0_us: float
    spans: list[PlotSpan]
    concatenated: bool = False
    settle_s: float = 0.0  # leading span excluded from the y-limits (filter start-up)

    @property
    def raw(self) -> np.ndarray:
        """Midpoint of the viewer raw min/max band."""
        return 0.5 * (self.raw_lo + self.raw_hi)


def viewer_channel_names(stored: list[str], n_cols: int) -> list[str]:
    """Labels after the viewer column slice (muscles + DRL)."""
    if n_cols <= 0:
        return []
    if len(stored) >= 8:
        return [stored[i] for i in VIEWER_EMG_STREAM_COLS[:n_cols]]
    if n_cols == len(VIEWER_EMG_PLOT_NAMES):
        return list(VIEWER_EMG_PLOT_NAMES)
    return list(stored[:n_cols]) or list(VIEWER_EMG_PLOT_NAMES[:n_cols])


def viewer_display_lpf(display_fs: float) -> float:
    """LPF cutoff used by ``Hub`` for the native-rate ``CausalEmgFilter``."""
    if display_fs <= 0:
        return EMG_BAND_HZ
    return min(EMG_BAND_HZ, (float(display_fs) / 2.0) * 0.9)


def filter_emg(
    uv: np.ndarray,
    fs_hz: float,
    *,
    display_fs: float = TARGET_DISPLAY_FS,
    mains_mode: str = "track",
    line_hz: float = 50.0,
    t_us: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Viewer native-rate chain: HPF → mains removal → display LPF + envelope.

    ``mains_mode="offline"`` swaps the causal canceller for the stronger
    block-wise least-squares fit in :mod:`jtsmm.emg.mains` -- only usable on a
    recorded file, but it needs no convergence time and takes the line frequency
    from the data. Pass ``t_us`` with it: recorded streams are packed around
    dropped samples, so row index is not proportional to time and the line
    frequency smears (0.2 Hz on s01_f_30) unless the gaps are accounted for.
    """
    uv = np.asarray(uv, dtype=np.float32)
    if uv.ndim == 1:
        uv = uv[:, None]
    n, c = uv.shape
    lpf = viewer_display_lpf(display_fs)
    if mains_mode == "offline":
        from jtsmm.emg.mains import remove_mains

        cleaned, _ = remove_mains(uv.astype(np.float64), float(fs_hz), t_us=t_us)
        uv = cleaned.astype(np.float32)
        mains_mode = "off"
    filt = CausalEmgFilter(
        float(fs_hz), c, lpf_hz=lpf, notch_hz=line_hz, mains_mode=mains_mode
    )
    tt = None if t_us is None else np.asarray(t_us, dtype=np.float64).reshape(-1)
    if tt is not None and tt.shape[0] != n:
        tt = None

    def _step(i0: int, i1: int):
        # With timestamps, dropped samples advance the mains reference instead
        # of silently shifting it; this is what the hub does live.
        if tt is None:
            return filt.process(uv[i0:i1])
        return filt.process_timed(tt[i0:i1], uv[i0:i1])

    if n <= _CHUNK:
        filtered, envelope = _step(0, n)
    else:
        filtered = np.empty((n, c), dtype=np.float32)
        envelope = np.empty((n, c), dtype=np.float32)
        for i0 in range(0, n, _CHUNK):
            i1 = min(n, i0 + _CHUNK)
            f, e = _step(i0, i1)
            filtered[i0:i1] = f
            envelope[i0:i1] = e
    suppressed = suppress_envelope_outliers(filtered, float(fs_hz)).astype(np.float32)
    if not np.array_equal(suppressed, filtered):
        filtered = suppressed
        envelope = rectified_envelope(filtered, float(fs_hz))
    return filtered, envelope


def decimate_like_viewer(
    t_us: np.ndarray,
    raw: np.ndarray,
    filtered: np.ndarray,
    envelope: np.ndarray,
    emg_fs: float,
    display_fs: float = TARGET_DISPLAY_FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match ``Hub.append_emg_raw``: min/max raw buckets, stride filt/env."""
    t_us = np.asarray(t_us)
    raw = np.asarray(raw, dtype=np.float32)
    filtered = np.asarray(filtered, dtype=np.float32)
    envelope = np.asarray(envelope, dtype=np.float32)
    if emg_fs <= 0 or display_fs <= 0:
        d = 1
    else:
        d = max(1, int(round(float(emg_fs) / float(display_fs))))
    n = t_us.shape[0]
    if d <= 1 or n == 0:
        return t_us, raw, raw.copy(), filtered, envelope
    nb = (n + d - 1) // d
    c = raw.shape[1]
    lo = np.empty((nb, c), dtype=np.float32)
    hi = np.empty_like(lo)
    tt = np.empty(nb, dtype=t_us.dtype)
    for b in range(nb):
        sl = slice(b * d, min(n, (b + 1) * d))
        s = raw[sl]
        lo[b] = np.nanmin(s, axis=0)
        hi[b] = np.nanmax(s, axis=0)
        tt[b] = t_us[sl][0]
    return tt, lo, hi, filtered[::d][:nb], envelope[::d][:nb]


def _select_channels(
    names: list[str],
    raw: np.ndarray,
    want: list[str] | None,
) -> tuple[list[str], np.ndarray]:
    if not want:
        return names, raw
    want_l = [w.strip().lower() for w in want]
    idx: list[int] = []
    for w in want_l:
        matched = [i for i, n in enumerate(names) if n.lower() == w or w in n.lower()]
        if not matched and w.isdigit():
            k = int(w) - 1
            if 0 <= k < len(names):
                matched = [k]
        if not matched:
            raise KeyError(f"unknown EMG channel {w!r}; have {names}")
        for i in matched:
            if i not in idx:
                idx.append(i)
    return [names[i] for i in idx], raw[:, idx]


def concat_phase_windows(
    t_us: np.ndarray,
    arrays: tuple[np.ndarray, ...],
    phases: list[PhaseInterval],
    *,
    gap_s: float = 0.25,
) -> tuple[np.ndarray, list[np.ndarray], list[PlotSpan]]:
    """Keep only samples inside ``phases``; concatenate with a NaN gap."""
    t_us = np.asarray(t_us, dtype=np.float64)
    parts_t: list[np.ndarray] = []
    parts: list[list[np.ndarray]] = [[] for _ in arrays]
    spans: list[PlotSpan] = []
    cursor = 0.0
    n_ch = arrays[0].shape[1] if arrays and arrays[0].ndim == 2 else 1
    nan_row = np.full((1, n_ch), np.nan, dtype=np.float32)

    for p in phases:
        mask = (t_us >= float(p.t_start_us)) & (t_us <= float(p.t_end_us))
        if not np.any(mask):
            continue
        t_seg = (t_us[mask] - t_us[mask][0]) * 1e-6 + cursor
        spans.append(PlotSpan(p.name, float(t_seg[0]), float(t_seg[-1])))
        parts_t.append(t_seg.astype(np.float64))
        for i, arr in enumerate(arrays):
            parts[i].append(np.asarray(arr[mask], dtype=np.float32))
        cursor = float(t_seg[-1]) + gap_s
        parts_t.append(np.array([cursor - gap_s * 0.5], dtype=np.float64))
        for i in range(len(arrays)):
            parts[i].append(nan_row.copy())

    if not spans:
        raise KeyError("no EMG samples inside the selected phases")
    # Drop the trailing NaN separator.
    parts_t.pop()
    for i in range(len(parts)):
        parts[i].pop()
    t_s = np.concatenate(parts_t)
    out = [np.concatenate(p, axis=0) for p in parts]
    return t_s, out, spans


def _stride_plot(n: int, max_points: int = _MAX_PLOT_POINTS) -> int:
    if n <= max_points:
        return 1
    return int(np.ceil(n / max_points))


def load_emg_plot(
    path: Path,
    *,
    channels: list[str] | None = None,
    phases: list[str] | None = None,
    include_drl: bool = True,
    full: bool = False,
    mains_mode: str = "track",
) -> EmgPlotData:
    with Session.open(path) as ses:
        emg = ses.emg
        if emg is None or emg.n == 0:
            raise KeyError(f"no /emg/samples_uv in {path}")
        t_us = np.asarray(emg.t_us, dtype=np.float64)
        uv = _load_uv(emg)
    uv = _slice_emg_for_viewer(uv)
    names = viewer_channel_names(emg.channel_names, uv.shape[1])
    if not include_drl:
        keep = [i for i, n in enumerate(names) if n.lower() != "drl"]
        if keep:
            names = [names[i] for i in keep]
            uv = uv[:, keep]
    names, uv = _select_channels(names, uv, channels)

    with h5py.File(path, "r") as f:
        all_phases = load_phases(f)
    if phases:
        want = {n.strip().lower() for n in phases}
        all_phases = [p for p in all_phases if p.name.lower() in want]
        if not all_phases:
            raise KeyError(f"none of {sorted(want)} found in session phases")
    else:
        motion = motion_phases(all_phases)
        if motion:
            all_phases = motion

    if uv.shape[0] == 0:
        raise KeyError(f"no EMG samples in {path}")

    display_fs = TARGET_DISPLAY_FS
    filtered, envelope = filter_emg(
        uv, emg.fs_hz, display_fs=display_fs, mains_mode=mains_mode, t_us=t_us
    )
    t_us, raw_lo, raw_hi, filtered, envelope = decimate_like_viewer(
        t_us, uv, filtered, envelope, emg.fs_hz, display_fs
    )
    concatenated = False
    if all_phases and not full:
        t_plot, (raw_lo, raw_hi, filtered, envelope), spans = concat_phase_windows(
            t_us, (raw_lo, raw_hi, filtered, envelope), all_phases
        )
        t0 = float(all_phases[0].t_start_us)
        concatenated = True
    else:
        t0 = float(t_us[0])
        t_plot = (t_us.astype(np.float64) - t0) * 1e-6
        spans = [
            PlotSpan(
                p.name,
                (float(p.t_start_us) - t0) * 1e-6,
                (float(p.t_end_us) - t0) * 1e-6,
            )
            for p in all_phases
        ]

    if raw_lo.shape[0] == 0:
        raise KeyError(f"no EMG samples in the selected window of {path}")

    step = _stride_plot(t_plot.shape[0])
    return EmgPlotData(
        t_s=t_plot[::step],
        raw_lo=raw_lo[::step],
        raw_hi=raw_hi[::step],
        filtered=filtered[::step],
        envelope=envelope[::step],
        names=names,
        fs_hz=float(emg.fs_hz),
        display_fs=float(display_fs),
        t0_us=t0,
        spans=spans,
        concatenated=concatenated,
        settle_s=0.0 if concatenated else _FILTER_SETTLE_S,
    )


def _load_uv(emg: EmgStream) -> np.ndarray:
    """Read ``samples_uv`` in chunks so long takes stay bounded."""
    n, c = emg.n, emg.n_channels
    out = np.empty((n, c), dtype=np.float32)
    for i0 in range(0, n, _CHUNK):
        i1 = min(n, i0 + _CHUNK)
        _, sl = emg.window(i0, i1)
        out[i0:i1] = sl
    return out


def _shade_spans(ax, spans: list[PlotSpan], *, label: bool) -> None:
    from matplotlib.patches import Patch

    if not spans:
        return
    handles = []
    for i, sp in enumerate(spans):
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
        ax.axvspan(sp.t0_s, sp.t1_s, color=color, alpha=0.35, lw=0, zorder=0)
        ax.axvline(sp.t0_s, color="#7f8c8d", lw=0.6, alpha=0.7, zorder=1)
        ax.axvline(sp.t1_s, color="#7f8c8d", lw=0.6, alpha=0.7, zorder=1)
        if label:
            mid = 0.5 * (sp.t0_s + sp.t1_s)
            ax.text(
                mid,
                0.98,
                sp.name,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                rotation=90 if (sp.t1_s - sp.t0_s) < 0.4 else 0,
                clip_on=True,
                zorder=3,
            )
            handles.append(Patch(facecolor=color, alpha=0.45, label=sp.name))
    if label and handles:
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize=7,
            ncol=min(4, max(1, len(handles))),
            framealpha=0.9,
        )


def _settled_ylim(
    t: np.ndarray, ys: tuple[np.ndarray, ...], settle_s: float
) -> tuple[float, float] | None:
    """y-limits over ``t >= t[0] + settle_s``, 5 % padded; None if not needed."""
    if settle_s <= 0 or t.size == 0:
        return None
    keep = np.isfinite(t) & (t >= np.nanmin(t) + settle_s)
    vals = np.concatenate([np.asarray(y, dtype=np.float64)[keep] for y in ys])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
    return lo - pad, hi + pad


def plot_emg_channel(
    data: EmgPlotData,
    channel: int,
    *,
    title: str,
    save: Path | None,
    show: bool,
    show_phases: bool,
) -> Path | None:
    """One 3×1 figure (filtered / raw / envelope) for a single channel."""
    import matplotlib.pyplot as plt

    if channel < 0 or channel >= len(data.names):
        raise IndexError(f"channel index {channel} out of range for {data.names}")
    color = _EMG_COLORS[channel % len(_EMG_COLORS)]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 7.5), constrained_layout=True)
    t = data.t_s
    rows = (
        (data.filtered[:, channel], None, "filtered (µV)"),
        (data.raw_lo[:, channel], data.raw_hi[:, channel], "raw (µV)"),
        (data.envelope[:, channel], None, "envelope (µV)"),
    )
    for ax, (y, y_hi, ylab) in zip(axes, rows):
        if show_phases:
            _shade_spans(ax, data.spans, label=ax is axes[0])
        if y_hi is not None:
            ax.fill_between(t, y, y_hi, color=color, alpha=0.45, lw=0, zorder=2)
            mid = 0.5 * (y + y_hi)
            ax.plot(t, mid, color=color, lw=0.6, zorder=3)
        else:
            ax.plot(t, y, color=color, lw=0.9, zorder=2)
        ylim = _settled_ylim(t, (y,) if y_hi is None else (y, y_hi), data.settle_s)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, zorder=1)
    axes[-1].set_ylim(bottom=0.0)
    xlabel = "t (s), phases only" if data.concatenated else "t − t₀ (s), MCU master"
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title)

    written: Path | None = None
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=140)
        written = save
        print(f"Wrote {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return written


def _png_path(out_dir: Path, stem: str, channel: str, phase_tag: str) -> Path:
    tag = f"{stem}_emg"
    if phase_tag:
        tag += f"_{phase_tag}"
    return out_dir / f"{tag}_{_slug(channel)}.png"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("session", type=Path, help="session.h5 or a session directory")
    ap.add_argument(
        "--channel",
        nargs="+",
        default=None,
        help="Subset of channels (names or 1-based indices). Default: viewer set",
    )
    ap.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Crop to these phase names (default: motion phases, calibration dropped)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Plot the whole take, including time between phases",
    )
    ap.add_argument(
        "--mains",
        choices=("offline", "track", "notch", "off"),
        default="offline",
        help=(
            "Line-interference removal: 'offline' (default) is the block-wise "
            "least-squares fit at the recorded timestamps -- the right tool for a "
            "file; 'track' reproduces the live viewer's causal tracker; 'notch' "
            "the old biquad bank; 'off' none"
        ),
    )
    ap.add_argument("--no-drl", action="store_true", help="Drop the DRL channel")
    ap.add_argument("--no-phases", action="store_true", help="Do not shade experiment phases")
    ap.add_argument("--out", type=Path, default=None, help="Directory for per-channel PNGs")
    ap.add_argument(
        "--save",
        type=Path,
        default=None,
        help="PNG path when plotting a single channel (otherwise treated as output directory)",
    )
    ap.add_argument("--no-show", action="store_true", help="Do not open interactive windows")
    args = ap.parse_args(argv)

    try:
        path = resolve_session_h5(args.session)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        data = load_emg_plot(
            path,
            channels=args.channel,
            phases=args.phases,
            include_drl=not args.no_drl,
            full=args.full,
            mains_mode=args.mains,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{path}: EMG  native {data.fs_hz:.4g} Hz → display {data.display_fs:.4g} Hz  "
        f"LPF {viewer_display_lpf(data.display_fs):.4g} Hz  "
        f"N={data.filtered.shape[0]}  channels={', '.join(data.names)}"
    )
    if data.spans:
        kind = "concatenated" if data.concatenated else "shaded on full take"
        print(f"  phases: {len(data.spans)} ({', '.join(s.name for s in data.spans)}) [{kind}]")

    phase_tag = "_".join(_slug(p) for p in args.phases) if args.phases else ""
    out_dir = args.out
    single_save = args.save
    if single_save is not None and (single_save.is_dir() or len(data.names) > 1):
        out_dir = single_save if single_save.suffix == "" or single_save.is_dir() else single_save.parent
        single_save = None
    if out_dir is None and (args.no_show or len(data.names) > 1):
        out_dir = path.parent
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for i, name in enumerate(data.names):
        if single_save is not None and len(data.names) == 1:
            save = single_save
        elif out_dir is not None:
            save = _png_path(out_dir, path.stem, name, phase_tag)
        else:
            save = None
        plot_emg_channel(
            data,
            i,
            title=f"{name} — filtered / raw / envelope — {path.name}",
            save=save,
            show=not args.no_show,
            show_phases=not args.no_phases,
        )
        n_written += 1
        print(f"  {name}")
    if n_written == 0:
        print("error: no EMG channels to plot", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
