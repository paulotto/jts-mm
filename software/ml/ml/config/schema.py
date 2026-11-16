# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Typed configuration (sections ``dataset`` and ``train`` of ``ml.yaml``).

Parsing is strict: an unknown key is an error, and keys whose meaning changed in the
2026-10-15 revision raise with a migration hint instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ml.config.classes import MUSCLE_CHANNEL_NAMES
from ml.core.dataset import WindowConfig

_DEFAULT_ML_DATA = Path.home() / "ml_data"
NORMALIZATIONS = ("mvc", "rest", "none")

_MOVED = {
    "causal": "the EMG chain is always causal (jtsmm CausalEmgFilter); remove 'causal'",
    "windowing": "renamed to 'window' (keys window_s, stride_s, label_at, ...)",
    "features": "feature names moved to window.emg_features / window.seq_features",
    "splits": "the protocol moved to train.protocol (loso_subject | loso_session | blocked)",
    "notch_hz": "use emg.line_hz (nominal mains; measured per take, tracked by the canceller)",
    "notch_harmonics": "all in-band harmonics are cancelled; remove it",
    "notch_q": "the tracking canceller has no Q; remove it",
    "filter_order": "the causal chain has fixed order; remove it",
    "envelope": "use emg.envelope_s (MVC reference only)",
}


def _check(section: str, raw: dict[str, Any], allowed: set[str]) -> None:
    for k in raw:
        if k in _MOVED:
            raise ValueError(f"{section}.{k}: {_MOVED[k]}")
    bad = sorted(set(raw) - allowed)
    if bad:
        raise ValueError(f"unknown keys in {section}: {bad}; allowed: {sorted(allowed)}")


@dataclass
class QualityConfig:
    use_leadoff: bool = True
    rail_frac: float = 0.99            # |raw| >= 99 % of the ADC full scale -> masked
    dilate_s: float = 0.1              # masks grow by this on both sides
    flatline_window_s: float = 0.0     # > 0: zero-variance stretches are masked too


@dataclass
class EmgConfig:
    channels: list[str] = field(default_factory=lambda: list(MUSCLE_CHANNEL_NAMES))
    line_hz: float = 50.0              # nominal mains frequency (measured per take)
    normalization: str = "mvc"         # mvc | rest | none
    mvc_session: str | None = None     # MVC take; default: the sidecar's mvc_session
    mvc_percentile: float = 95.0
    envelope_s: float = 0.15
    quality: QualityConfig = field(default_factory=QualityConfig)


@dataclass
class OutputConfig:
    processed_dir: str = str(_DEFAULT_ML_DATA / "processed")
    runs_dir: str = str(_DEFAULT_ML_DATA / "ml_runs")
    evals_dir: str = str(_DEFAULT_ML_DATA / "ml_evals")


@dataclass
class DatasetConfig:
    seed: int = 42
    emg: EmgConfig = field(default_factory=EmgConfig)
    imu: bool = True
    window: WindowConfig = field(default_factory=WindowConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        _check("dataset", data, {"seed", "emg", "imu", "window", "output"})
        emg_raw = dict(data.get("emg") or {})
        _check("dataset.emg", emg_raw, {f.name for f in fields(EmgConfig)})
        q_raw = dict(emg_raw.pop("quality", None) or {})
        _check("dataset.emg.quality", q_raw, {f.name for f in fields(QualityConfig)})
        if "normalization" in emg_raw:
            norm = str(emg_raw["normalization"]).lower()
            if norm == "robust":
                raise ValueError("dataset.emg.normalization: 'robust' (median/IQR over the whole "
                                 "recording, i.e. including the test data) was removed; use mvc, "
                                 "rest or none")
            if norm not in NORMALIZATIONS:
                raise ValueError(f"dataset.emg.normalization must be one of {NORMALIZATIONS}")
            emg_raw["normalization"] = norm
        imu = data.get("imu", True)
        if isinstance(imu, dict):
            _check("dataset.imu", imu, {"enabled"})
            imu = imu.get("enabled", True)
        out_raw = dict(data.get("output") or {})
        _check("dataset.output", out_raw, {f.name for f in fields(OutputConfig)})
        return cls(seed=int(data.get("seed", 42)),
                   emg=EmgConfig(**{**emg_raw, "quality": QualityConfig(**q_raw)}),
                   imu=bool(imu), window=WindowConfig.from_dict(data.get("window")),
                   output=OutputConfig(**out_raw))


@dataclass
class TrainConfig:
    seed: int = 42
    model: str = "lda"
    feature_set: str = "emg"
    protocol: str = "loso_subject"
    inner_folds: int = 3
    search: bool = True
    max_train: int | None = None
    blocked_k: int = 5
    guard_s: float = 0.35
    n_jobs: int = 1
    probability: bool = True
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        allowed = {f.name for f in fields(cls)}
        _check("train", data, allowed)
        from ml.models import registry
        from ml.protocols.folds import normalize_protocol

        d = dict(data)
        models = {}
        for name, m in dict(d.pop("models", None) or {}).items():
            registry.get(name)
            m = dict(m or {})
            _check(f"train.models.{name}", m, {"params", "grid"})
            models[name] = {"params": dict(m.get("params") or {}), "grid": m.get("grid")}
        out = OutputConfig(**dict(d.pop("output", None) or {}))
        cfg = cls(**d, models=models, output=out)
        cfg.protocol = normalize_protocol(cfg.protocol)
        registry.get(cfg.model)
        return cfg

    def params_for(self, model: str) -> dict[str, Any]:
        return dict((self.models.get(model) or {}).get("params") or {})

    def grid_for(self, model: str) -> dict[str, list[Any]] | None:
        g = (self.models.get(model) or {}).get("grid")
        return None if g is None else dict(g)
