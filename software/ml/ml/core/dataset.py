# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``WindowDataset``: windowed features of one or more takes, and the one builder.

``build_windows`` is the single place that turns a take into windows. Both entry
points call it: the ``ml dataset`` CLI (``session.h5`` + label sidecar, see
:mod:`ml.datasets.builder`) and the biomech stage ``ml_windows``
(:mod:`ml.biomech_stages`), which supplies biomech's MVC reference and masks.

Per phase it runs the causal EMG chain from ``warmup_s`` before the phase (the
warm-up is discarded), scales to % of the reference, applies the masks, cuts
trailing windows, rejects gap-spanning, over-masked and unlabelled windows, and
computes the feature blocks:

* ``emg``      (n, C*F) EMG window features (:mod:`ml.core.features`)
* ``imu``      (n, 25) IMU window features (when IMU data is given)
* ``seq``      (n, S, C*F_seq) EMG sub-frame features (GRU)
* ``seq_imu``  (n, S, 12) IMU sub-frame means (GRU early fusion)

With ``emg_norm: rest_relative`` (default, analysis_plan D13) the ``emg`` and
``seq`` blocks are then expressed relative to the take's unlabelled ``rest``
phase (:func:`ml.core.features.rest_relative_stats`): every valid rest window
counts, whatever its label, and the statistics are stored in ``meta["norm"]``.
``emg_norm: mvc`` keeps the features in % of the reference (comparison).

Feature sets name one or more blocks joined with ``+`` (``emg+imu``); the
historical names ``emg_imu``, ``emg_nodig`` (EMG without the digastrics) and
``emg_nodig_imu`` are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from ml.config.classes import CLASS_COUNT, CLASS_NAMES, DIGASTRIC, REST_PHASE
from ml.core import features as F
from ml.core import signals as S
from ml.core.windows import gap_windows, label_runs, window_ends, window_labels


@dataclass(frozen=True)
class WindowConfig:
    window_s: float = 0.2
    stride_s: float = 0.05
    imu_hz: float = 200.0
    warmup_s: float = 2.0
    max_masked: float = 0.2              # per-channel masked share above which a window is dropped
    thr_sigma: float = 1.0               # ZC/SSC/WAMP threshold: x rest SD of sample differences
    gap_factor: float = 1.1              # window spanning more than 1.1 W -> dropped
    label_at: str = "end"                # end | majority
    purity_min: float = 0.6              # majority mode only
    label_max_age_s: float = 0.1         # end mode: a label older than this is no label
    seq_subframes: int = 10
    emg_band_hz: tuple[float, float] = (20.0, 450.0)
    emg_features: tuple[str, ...] = F.DEFAULT_EMG_FEATURES
    seq_features: tuple[str, ...] = F.DEFAULT_SEQ_FEATURES
    emg_norm: str = "rest_relative"      # rest_relative (D13) | mvc (% of the reference only)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "WindowConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown window settings {bad}; known: {sorted(known)}")
        if "emg_norm" in d and d["emg_norm"] not in F.EMG_NORMS:
            raise ValueError(f"window.emg_norm must be one of {F.EMG_NORMS}, got {d['emg_norm']!r}")
        if "emg_band_hz" in d:
            d["emg_band_hz"] = tuple(float(v) for v in d["emg_band_hz"])
        for k in ("emg_features", "seq_features"):
            if k in d:
                d[k] = tuple(d[k])
                for n in d[k]:
                    F.get_feature(n)
        return cls(**d)


@dataclass
class WindowDataset:
    """Windows of one or more takes. Rows are in time order within each take."""

    blocks: dict[str, np.ndarray]
    y: np.ndarray                         # int, 0..6
    pid: np.ndarray                       # subject id per row (str)
    session: np.ndarray                   # take id per row (str)
    phase: np.ndarray                     # phase name per row (str)
    t_end_us: np.ndarray                  # int64 window end on the master clock
    groups: np.ndarray                    # label-run id, unique across takes
    names: dict[str, list[str]] = field(default_factory=dict)   # block -> feature names
    purity: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # matrix() per feature set (read-only arrays); not pickled, not copied by replace().
    _matrix_cache: dict[str, tuple[np.ndarray, list[slice]]] = field(
        default_factory=dict, init=False, repr=False, compare=False)

    def __len__(self) -> int:
        return int(self.y.size)

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_matrix_cache"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._matrix_cache = {}

    # -- feature sets --------------------------------------------------------------------
    def block(self, name: str) -> np.ndarray:
        if name in self.blocks:
            return self.blocks[name]
        if name == "emg_nodig" and "emg" in self.blocks:
            keep = [i for i, n in enumerate(self.names.get("emg", []))
                    if n.split("|")[0] not in DIGASTRIC]
            return self.blocks["emg"][:, keep]
        raise KeyError(f"no feature block {name!r}; have {sorted(self.blocks)} (+ emg_nodig)")

    def matrix(self, fset: str) -> tuple[np.ndarray, list[slice]]:
        """Concatenated feature set (float64, read-only) and the column slice of each
        block; computed once per feature set and cached (blocks are not to be modified
        in place afterwards)."""
        hit = self._matrix_cache.get(fset)
        if hit is not None:
            return hit[0], list(hit[1])
        parts = [self.block(b).astype(np.float64) for b in feature_set_blocks(fset)]
        if len({p.ndim for p in parts}) != 1 or (parts[0].ndim == 3 and len(
                {p.shape[1] for p in parts}) != 1):
            raise ValueError(f"feature set {fset!r} mixes window and sequence blocks")
        X = np.concatenate(parts, axis=-1)
        X.flags.writeable = False
        widths = np.cumsum([0] + [p.shape[-1] for p in parts])
        slices = [slice(int(a), int(b)) for a, b in zip(widths[:-1], widths[1:])]
        self._matrix_cache[fset] = (X, slices)
        return X, list(slices)

    def subset(self, idx: np.ndarray) -> "WindowDataset":
        idx = np.asarray(idx)
        return replace(self, blocks={k: v[idx] for k, v in self.blocks.items()}, y=self.y[idx],
                       pid=self.pid[idx], session=self.session[idx], phase=self.phase[idx],
                       t_end_us=self.t_end_us[idx], groups=self.groups[idx],
                       purity=None if self.purity is None else self.purity[idx])

    def summary(self) -> dict[str, Any]:
        return {
            "n_windows": len(self),
            "class_counts": {CLASS_NAMES[i]: int(np.sum(self.y == i)) for i in range(CLASS_COUNT)},
            "n_groups": int(np.unique(self.groups).size),
            "subjects": sorted({str(p) for p in self.pid}),
            "sessions": sorted({str(s) for s in self.session}),
            "blocks": {k: list(v.shape) for k, v in self.blocks.items()},
        }

    # -- I/O -----------------------------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        arrays = {f"X_{k}": v.astype(np.float32) for k, v in self.blocks.items()}
        np.savez_compressed(out / "dataset.npz", y=self.y.astype(np.int8), pid=self.pid.astype(str),
                            session_id=self.session.astype(str), phase=self.phase.astype(str),
                            t_end_us=self.t_end_us.astype(np.int64),
                            groups=self.groups.astype(np.int64),
                            purity=(np.full(len(self), np.nan) if self.purity is None
                                    else self.purity), **arrays)
        meta = {**self.meta, **self.summary(), "feature_names": self.names}
        (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=_json) + "\n",
                                           encoding="utf-8")
        return out / "dataset.npz"

    @classmethod
    def load(cls, path: str | Path) -> "WindowDataset":
        path = Path(path)
        if path.is_dir():
            path = path / "dataset.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        z = np.load(path, allow_pickle=False)
        if "X_feat" in z.files and "X_emg" not in z.files:
            raise ValueError(f"{path} was written by the pre-2026-10-15 builder (X_feat); "
                             "rebuild it with `ml dataset`")
        meta_p = path.with_name("metadata.json")
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.is_file() else {}
        names = meta.pop("feature_names", {}) if isinstance(meta.get("feature_names"), dict) else {}
        blocks = {k[2:]: np.asarray(z[k], dtype=np.float64) for k in z.files if k.startswith("X_")}
        n = len(z["y"])
        purity = np.asarray(z["purity"]) if "purity" in z.files else None
        meta["path"] = str(path.resolve())
        return cls(blocks=blocks, y=np.asarray(z["y"], dtype=np.int64),
                   pid=np.asarray(z["pid"]).astype(str),
                   session=np.asarray(z["session_id"]).astype(str),
                   phase=np.asarray(z["phase"]).astype(str),
                   t_end_us=np.asarray(z["t_end_us"], dtype=np.int64),
                   groups=np.asarray(z["groups"], dtype=np.int64) if "groups" in z.files
                   else np.zeros(n, np.int64), names=names, purity=purity, meta=meta)


def feature_set_blocks(fset: str) -> list[str]:
    alias = {"emg_imu": "emg+imu", "emg_nodig_imu": "emg_nodig+imu", "seq_emg_imu": "seq+seq_imu"}
    return alias.get(fset, fset).split("+")


def concat(parts: list[WindowDataset]) -> WindowDataset:
    """Pool takes. Groups are renumbered so no label run is shared across takes."""
    if not parts:
        raise ValueError("nothing to concatenate")
    keys = list(parts[0].blocks)
    for i, p in enumerate(parts[1:], 1):
        if set(p.blocks) != set(keys):
            raise ValueError(f"take {i} has blocks {sorted(p.blocks)}, expected {sorted(keys)}")
        for k in keys:
            if p.blocks[k].shape[1:] != parts[0].blocks[k].shape[1:] or (
                    p.names.get(k) != parts[0].names.get(k)):
                raise ValueError(f"take {i}: block {k!r} differs in shape or feature names")
    off, groups = 0, []
    for p in parts:
        _, g = np.unique(p.groups, return_inverse=True)
        groups.append(g + off)
        off += int(g.max()) + 1 if g.size else 0
    cat = np.concatenate
    pur = None if any(p.purity is None for p in parts) else cat([p.purity for p in parts])
    meta = {"sources": [p.meta for p in parts]}
    return WindowDataset(blocks={k: cat([p.blocks[k] for p in parts]) for k in keys},
                         y=cat([p.y for p in parts]), pid=cat([p.pid for p in parts]),
                         session=cat([p.session for p in parts]),
                         phase=cat([p.phase for p in parts]),
                         t_end_us=cat([p.t_end_us for p in parts]), groups=cat(groups),
                         names=dict(parts[0].names), purity=pur, meta=meta)


@dataclass
class Imu:
    """Sanitized IMU streams: (t_us, values) for both slots."""

    q_mta: tuple[np.ndarray, np.ndarray]
    q_cra: tuple[np.ndarray, np.ndarray]
    a_mta: tuple[np.ndarray, np.ndarray]
    a_cra: tuple[np.ndarray, np.ndarray]


def filter_phases(phases: list[tuple[str, int, int]], emg_t: np.ndarray, emg_uv: np.ndarray,
                  emg_fs: float, line_hz: float, masks: dict[str, np.ndarray],
                  channels: list[str], cfg: WindowConfig = WindowConfig()
                  ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Step 1 of :func:`build_windows`: every phase through the causal chain (warm-up of
    ``cfg.warmup_s`` discarded), masked samples held before filtering and NaN after;
    phase -> (t_us, filtered uV), *not* yet scaled by the reference. The stage computes
    its normalization reference from these segments too (no second filter pass)."""
    emg_t = np.asarray(emg_t, dtype=np.int64)
    filt: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, t0, t1 in phases:
        sel = (emg_t >= t0 - cfg.warmup_s * 1e6) & (emg_t <= t1)
        t, raw = emg_t[sel], np.asarray(emg_uv[sel], dtype=np.float64)
        keep = t >= t0
        m = masks.get(name)
        if m is not None and m.shape != (int(keep.sum()), len(channels)):
            raise ValueError(f"mask of phase {name!r} has shape {m.shape}, expected "
                             f"{(int(keep.sum()), len(channels))}")
        if m is not None:
            full = np.zeros(raw.shape, dtype=bool)
            full[keep] = m
            raw = S.hold_fill(raw, full)
        x = S.causal_emg(t, raw, emg_fs, line_hz, hpf_hz=cfg.emg_band_hz[0],
                         lpf_hz=cfg.emg_band_hz[1])[keep]
        if m is not None:
            x = np.where(m, np.nan, x)
        filt[name] = (t[keep], x)
    return filt


def build_windows(*, pid: str, session: str, phases: list[tuple[str, int, int]],
                  emg_t: np.ndarray, emg_uv: np.ndarray, emg_fs: float, line_hz: float,
                  ref_uv: np.ndarray, masks: dict[str, np.ndarray], channels: list[str],
                  labels: dict[str, tuple[np.ndarray, np.ndarray]], imu: Imu | None = None,
                  cfg: WindowConfig = WindowConfig(),
                  filtered: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
                  ) -> WindowDataset:
    """Windows of one take.

    @param phases  (name, t_start_us, t_end_us) in time order
    @param emg_t   (N,) master-clock time of ``emg_uv`` covering every phase and its warm-up
    @param ref_uv  (C,) normalization reference; features are in % of it
    @param masks   phase -> (N_phase, C) bool, aligned with the phase's samples
    @param labels  phase -> (t_us, labels) at the label rate, -1 = ignore; a phase
                   without an entry (e.g. an unlabelled ``rest``) gives no training
                   windows but still serves thresholds and the D13 statistics
    @param filtered  :func:`filter_phases` of the same arguments, if already computed
                     (identical result, the filter pass is skipped)
    """
    if not phases:
        raise ValueError(f"{session}: no phases to window")
    ref = np.asarray(ref_uv, dtype=np.float64)
    if ref.shape != (len(channels),) or not np.all(np.isfinite(ref) & (ref > 0)):
        raise ValueError(f"normalization reference must be {len(channels)} positive values, "
                         f"got {ref}")
    emg_t = np.asarray(emg_t, dtype=np.int64)
    win = int(round(cfg.window_s * emg_fs))
    stride = int(round(cfg.stride_s * emg_fs))
    iwin = int(round(cfg.window_s * cfg.imu_hz))
    w_us = cfg.window_s * 1e6

    # 1) Filter every phase (causal, warm-up discarded), mask, scale to % of the reference.
    if filtered is None:
        filtered = filter_phases(phases, emg_t, emg_uv, emg_fs, line_hz, masks, channels, cfg)
    missing = [name for name, _, _ in phases if name not in filtered]
    if missing:
        raise ValueError(f"filtered phases lack {missing}")
    filt = {name: (filtered[name][0], 100.0 * filtered[name][1] / ref[None, :])
            for name, _, _ in phases}

    # 2) ZC/SSC/WAMP thresholds from the unlabelled rest phase (else the first phase),
    #    on the scale of the sample differences these features threshold.
    thr_src = REST_PHASE if REST_PHASE in filt else phases[0][0]
    thr = F.rest_thresholds(filt[thr_src][1], cfg.thr_sigma)
    if cfg.emg_norm == "rest_relative" and REST_PHASE not in filt:
        raise ValueError(f"{session}: emg_norm 'rest_relative' needs the '{REST_PHASE}' phase; "
                         "use window.emg_norm: mvc for takes without one")

    out: dict[str, list] = {k: [] for k in ("emg", "imu", "seq", "seq_imu", "y", "t", "ph",
                                            "pur")}
    rest_emg = np.zeros((0, len(channels) * len(cfg.emg_features)))
    rest_seq = np.zeros((0, cfg.seq_subframes, len(channels) * len(cfg.seq_features)))
    rejected = {"masked": 0, "label": 0, "gap": 0, "nonfinite": 0}
    for name, t0, t1 in phases:
        t, x = filt[name]
        ends = window_ends(len(t), win, stride)
        if ends.size == 0:
            continue
        t_end = t[ends]
        gap = gap_windows(t, ends, win, w_us, cfg.gap_factor)
        fe, nanfrac = F.emg_window_features(x, emg_fs, ends, win, thr, cfg.emg_features)
        fs_seq = F.emg_sequence_features(x, emg_fs, ends, win, thr, cfg.seq_subframes,
                                         cfg.seq_features)
        lt, lab = labels.get(name, (np.zeros(0, np.int64), np.zeros(0, np.int64)))
        y, pur = window_labels(t_end, lt, lab, w_us, mode=cfg.label_at, purity_min=cfg.purity_min,
                               max_age_us=cfg.label_max_age_s * 1e6)
        bad_mask = nanfrac.max(axis=1) > cfg.max_masked
        finite = np.isfinite(fe).all(1)
        if imu is not None:
            tg = np.arange(t0 - int(w_us), t1 + 1, int(1e6 / cfg.imu_hz), dtype=np.int64)
            isig = S.imu_signals(tg, imu.q_mta, imu.q_cra, imu.a_mta, imu.a_cra)
            iends = np.clip(np.searchsorted(tg, t_end, side="right") - 1, 0, len(tg) - 1)
            fi = F.imu_window_features(isig, iends, iwin)
            fsi = F.imu_sequence_features(isig, iends, iwin, cfg.seq_subframes)
            finite &= np.isfinite(fi).all(1)
        if name == REST_PHASE:                   # D13 statistics: labels not used
            valid = ~bad_mask & ~gap & np.isfinite(fe).all(1)
            rest_emg, rest_seq = fe[valid], fs_seq[valid]
        ok = ~bad_mask & (y >= 0) & ~gap & finite
        rejected["masked"] += int(bad_mask.sum())
        rejected["label"] += int((y < 0).sum())
        rejected["gap"] += int(gap.sum())
        rejected["nonfinite"] += int((~finite & ~bad_mask & (y >= 0) & ~gap).sum())
        out["emg"].append(fe[ok])
        out["seq"].append(np.nan_to_num(fs_seq[ok]))
        if imu is not None:
            out["imu"].append(fi[ok])
            out["seq_imu"].append(np.nan_to_num(fsi[ok]))
        out["y"].append(y[ok])
        out["t"].append(t_end[ok])
        out["ph"].append(np.full(int(ok.sum()), name))
        out["pur"].append(pur[ok])

    blocks = {k: np.vstack(out[k]) for k in ("emg", "seq") + (("imu", "seq_imu") if imu else ())
              if out[k]}
    y = np.concatenate(out["y"]).astype(np.int64) if out["y"] else np.zeros(0, np.int64)
    phase = np.concatenate(out["ph"]) if out["ph"] else np.zeros(0, str)
    names = {"emg": F.emg_feature_names(channels, cfg.emg_features),
             "seq": F.seq_feature_names(channels, cfg.seq_features)}
    norm: dict[str, Any] = {}
    if cfg.emg_norm == "rest_relative":
        for b, R in (("emg", rest_emg), ("seq", rest_seq)):
            try:
                c, sc = F.rest_relative_stats(R, names[b])
            except ValueError as exc:
                raise ValueError(f"{session}: {exc} (phase '{REST_PHASE}')") from None
            norm[b] = {"center": c.tolist(), "scale": sc.tolist()}
            if b in blocks:
                blocks[b] = F.apply_rest_relative(blocks[b], names[b], c, sc)
    if imu is not None:
        names["imu"] = F.imu_feature_names()
        names["seq_imu"] = list(F.IMU_SIGNALS)
    n = y.size
    return WindowDataset(
        blocks=blocks, y=y, pid=np.full(n, pid), session=np.full(n, session), phase=phase,
        t_end_us=np.concatenate(out["t"]).astype(np.int64) if out["t"] else np.zeros(0, np.int64),
        groups=label_runs(y, phase), names=names,
        purity=np.concatenate(out["pur"]) if out["pur"] else np.zeros(0),
        meta={"pid": pid, "session": session, "rejected": rejected, "thr_source": thr_src,
              "thr": thr.tolist(), "emg_norm": cfg.emg_norm, "norm": norm,
              "n_rest_windows": int(rest_emg.shape[0]),
              "line_hz": float(line_hz), "ref_uv": ref.tolist(), "channels": list(channels),
              "emg_fs": float(emg_fs), "window": _cfg_dict(cfg)})


def _cfg_dict(cfg: WindowConfig) -> dict[str, Any]:
    return {f.name: (list(v) if isinstance(v := getattr(cfg, f.name), tuple) else v)
            for f in fields(cfg)}


def _json(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))
