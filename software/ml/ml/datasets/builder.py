# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``ml dataset``: ``session.h5`` + label sidecar -> :class:`~ml.core.dataset.WindowDataset`.

This is the file-based entry point to the shared core. It reads the take with
:mod:`ml.io.session`, takes the labels per experiment phase from the sidecar
(255 -> ignore), measures the mains frequency, builds the rail / lead-off /
flat-line masks, computes the label-free normalization reference and calls
:func:`ml.core.dataset.build_windows` -- the same function the biomech stage
``ml_windows`` uses, so both entry points produce the same windows on equal
settings.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import numpy as np

from ml.config.classes import REST_PHASE
from ml.config.loader import config_hash, load_dataset_config
from ml.config.schema import DatasetConfig
from ml.core import signals as S
from ml.core.dataset import WindowDataset, build_windows, concat
from ml.io.labels import LabelSidecar, default_label_path, read_labels
from ml.io.session import SessionData, load_imu, load_session

LINE_HZ_WINDOW_S = 60.0


def _masks(sd: SessionData, sel: np.ndarray, channels: list[str], cfg: DatasetConfig
           ) -> np.ndarray:
    q = cfg.emg.quality
    raw = sd.emg_samples_uv[sel][:, sd.channel_index(channels)]
    m = S.rail_mask(raw, sd.full_scale_uv(channels), q.rail_frac)
    if q.use_leadoff and sd.leadoff_status is not None:
        m |= S.leadoff_mask(sd.leadoff_status[sel], sd.channel_index(channels), int(sel.sum()))
    if q.flatline_window_s > 0:
        m |= S.flatline_mask(raw, sd.emg_sample_rate_sps, q.flatline_window_s)
    return S.dilate(m, int(round(q.dilate_s * sd.emg_sample_rate_sps)))


def _filtered(sd: SessionData, t0: int, t1: int, channels: list[str], line: float,
              cfg: DatasetConfig) -> np.ndarray:
    """Causally filtered phase (warm-up discarded), masked samples NaN; for references."""
    w = cfg.window
    t = sd.emg_t_us
    sel = (t >= t0 - w.warmup_s * 1e6) & (t <= t1)
    keep = t[sel] >= t0
    m = np.zeros((int(sel.sum()), len(channels)), dtype=bool)
    m[keep] = _masks(sd, sel & (t >= t0), channels, cfg)
    raw = S.hold_fill(sd.emg_samples_uv[sel][:, sd.channel_index(channels)].astype(float), m)
    x = S.causal_emg(t[sel], raw, sd.emg_sample_rate_sps, line, hpf_hz=w.emg_band_hz[0],
                     lpf_hz=w.emg_band_hz[1])
    return np.where(m, np.nan, x)[keep]


def take_line_hz(sd: SessionData, channels: list[str], nominal: float) -> float:
    t0 = min((p.t_start_us for p in sd.phases), default=int(sd.emg_t_us[0]))
    sel = (sd.emg_t_us >= t0) & (sd.emg_t_us <= t0 + LINE_HZ_WINDOW_S * 1e6)
    return S.line_hz(sd.emg_samples_uv[sel][:, sd.channel_index(channels)], sd.emg_sample_rate_sps,
                     sd.emg_t_us[sel], nominal)


def mvc_reference_uv(mvc_path: str | Path, channels: list[str], cfg: DatasetConfig) -> np.ndarray:
    """p95 of the causal RMS envelope in the MVC take's mvc_opening / mvc_closing phases,
    filtered by the same causal chain as the windows (same gain)."""
    sd = load_session(mvc_path)
    line = take_line_hz(sd, channels, cfg.emg.line_hz)
    segs = {}
    for name in ("mvc_opening", "mvc_closing"):
        ph = sd.phase(name)
        if ph is not None:
            segs[name] = _filtered(sd, ph.t_start_us, ph.t_end_us, channels, line, cfg)
    ref = S.mvc_reference(segs, channels, sd.emg_sample_rate_sps,
                          percentile=cfg.emg.mvc_percentile, envelope_s=cfg.emg.envelope_s)
    if not np.all(np.isfinite(ref) & (ref > 0)):
        raise ValueError(f"MVC reference incomplete for {mvc_path} (phases mvc_opening / "
                         f"mvc_closing): {dict(zip(channels, np.round(ref, 2)))}")
    return ref


def _reference(sd: SessionData, labels: LabelSidecar, channels: list[str], line: float,
               cfg: DatasetConfig, mvc_path: str | Path | None) -> tuple[np.ndarray, dict]:
    mode = cfg.emg.normalization
    if mode == "none":
        return np.full(len(channels), 100.0), {"normalization": "none"}
    if mode == "rest":
        ph = sd.phase(REST_PHASE)
        if ph is None:
            raise ValueError(f"normalization 'rest' needs a '{REST_PHASE}' phase in {sd.path}")
        ref = S.rest_reference(_filtered(sd, ph.t_start_us, ph.t_end_us, channels, line, cfg))
        return ref, {"normalization": "rest", "ref_source": "rest phase RMS"}
    # An MVC take is always preferred: the sidecar's mvc_ref_uv comes from biomech's
    # zero-phase chain, whose gain differs from the causal chain of the windows.
    side = str(labels.attrs.get("mvc_session") or "")
    stored = labels.attrs.get("mvc_ref_uv")
    mvc = mvc_path or cfg.emg.mvc_session or (side if side and (Path(side).is_file()
                                                                 or not stored) else None)
    if not mvc and stored:
        refs = json.loads(stored) if isinstance(stored, str) else dict(stored)
        missing = [c for c in channels if c not in refs]
        if missing:
            raise ValueError(f"sidecar mvc_ref_uv lacks {missing}")
        return np.array([refs[c] for c in channels], dtype=np.float64), {
            "normalization": "mvc", "ref_chain": "zero-phase (biomech)",
            "ref_source": str(labels.attrs.get("mvc_ref_source") or "sidecar mvc_ref_uv")}
    if not mvc:
        raise ValueError("normalization 'mvc' needs an MVC take: pass --mvc, set "
                         "dataset.emg.mvc_session, write it into the sidecar (ml labels "
                         "from-biomech does), or use normalization: rest")
    return mvc_reference_uv(mvc, channels, cfg), {"normalization": "mvc", "ref_chain": "causal",
                                                   "ref_source": str(mvc)}


def phase_labels(sd: SessionData, labels: LabelSidecar
                 ) -> tuple[list[tuple[str, int, int]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Experiment phases that contain labelled samples, with their label samples.

    The ``rest`` phase is always included (unlabelled it gives no training windows but
    still sets the ZC/SSC/WAMP thresholds and the rest-relative statistics, D13).
    """
    lt, ly = labels.t_master_us, labels.window_labels()
    exp = sd.experiment_phases()
    if not exp:     # no /phases: one segment over the labelled span
        lab = ly >= 0
        if not lab.any():
            raise ValueError("label sidecar has no labelled samples")
        return ([("full_session", int(lt[lab][0]), int(lt[lab][-1]))],
                {"full_session": (lt, ly)})
    phases, out = [], {}
    for ph in exp:
        m = (lt >= ph.t_start_us) & (lt <= ph.t_end_us)
        if np.any(ly[m] >= 0) or ph.name == REST_PHASE:
            phases.append((ph.name, ph.t_start_us, ph.t_end_us))
            out[ph.name] = (lt[m], ly[m])
    if not any(np.any(y >= 0) for _, y in out.values()):
        raise ValueError(f"no labelled samples inside any experiment phase of {sd.path}")
    return phases, out


def build_dataset(session_path: str | Path, *, label_path: str | Path | None = None,
                  config: DatasetConfig | None = None, mvc_path: str | Path | None = None
                  ) -> WindowDataset:
    """Windows of one take from ``session.h5`` and its label sidecar."""
    cfg = config or load_dataset_config()
    session_path = Path(session_path)
    sd = load_session(session_path)
    lp = Path(label_path) if label_path else default_label_path(session_path)
    labels = read_labels(lp)
    lab_sid = str(labels.attrs.get("session_id") or "").strip()
    if lab_sid and lab_sid not in {"session", sd.session_id}:
        raise ValueError(f"label session_id {lab_sid!r} != session {sd.session_id!r}")
    subject = str(labels.attrs.get("subject_id") or "").strip() or sd.subject_id
    if not subject or subject.lower() == "unknown":
        raise ValueError(f"no subject id for {session_path}: /meta has no subject_id, the folder "
                         "name does not parse as session_<pid>_<sex>_<age>_..., there is no "
                         "experiment_name, and the sidecar has no subject_id attribute")
    if np.any(np.diff(sd.emg_t_us) <= 0):
        raise ValueError(f"EMG timestamps of {session_path} are not strictly increasing")

    channels = list(cfg.emg.channels)
    phases, labs = phase_labels(sd, labels)
    line = take_line_hz(sd, channels, cfg.emg.line_hz)
    ref, ref_info = _reference(sd, labels, channels, line, cfg, mvc_path)
    w = cfg.window
    t = sd.emg_t_us
    sel = (t >= min(p[1] for p in phases) - w.warmup_s * 1e6) & (t <= max(p[2] for p in phases))
    masks = {name: _masks(sd, (t >= t0) & (t <= t1), channels, cfg) for name, t0, t1 in phases}
    imu = load_imu(session_path) if cfg.imu else None
    if cfg.imu and imu is None:
        raise ValueError(f"{session_path} has no usable /imu_mta and /imu_cra; set dataset.imu: "
                         "false for an EMG-only dataset")
    ds = build_windows(pid=subject, session=sd.session_id, phases=phases, emg_t=t[sel],
                       emg_uv=sd.emg_samples_uv[sel][:, sd.channel_index(channels)],
                       emg_fs=sd.emg_sample_rate_sps, line_hz=line, ref_uv=ref, masks=masks,
                       channels=channels, labels=labs, imu=imu, cfg=w)
    if len(ds) == 0:
        raise ValueError(f"no windows admitted for {session_path}: {ds.meta['rejected']}")
    ds.meta.update({**ref_info, "session_path": str(session_path.resolve()),
                    "label_path": str(lp.resolve()),
                    "label_schema_version": str(labels.attrs.get("schema_version", "")),
                    "label_source": str(labels.attrs.get("source", "")),
                    "phases": [p[0] for p in phases], "config_hash": config_hash(cfg),
                    "git_commit": _git_commit(session_path)})
    return ds


def build_pool(refs: list[tuple[Path, Path]], *, config: DatasetConfig | None = None,
               on_error: str = "raise") -> WindowDataset:
    """Build every (session, sidecar) pair and pool them; ``on_error='skip'`` records failures."""
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    parts, failures = [], []
    for sp, lp in refs:
        try:
            parts.append(build_dataset(sp, label_path=lp, config=config))
        except Exception as exc:
            if on_error == "raise":
                raise
            failures.append({"session_path": str(sp), "error": f"{type(exc).__name__}: {exc}"})
    if not parts:
        raise ValueError(f"all {len(refs)} sessions failed: {failures}")
    ds = concat(parts) if len(parts) > 1 else parts[0]
    ds.meta.update({"n_requested": len(refs), "n_built": len(parts), "failures": failures})
    return ds


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path.parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


__all__ = ["build_dataset", "build_pool", "mvc_reference_uv", "phase_labels", "take_line_hz"]
