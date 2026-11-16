# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Read the parts of a multimodal ``session.h5`` the ML pipeline needs.

Recorded layout (pilot study): ``/emg/samples_uv`` (N, 8) at ``sample_rate_sps``
(4000 Hz) with ``channel_names`` in recorded order and ``leadoff_status``
``[statp, statn]`` (bit i = CH(i+1)); ``/imu_mta|imu_cra/{quat, accel}`` with
their own ``t_master_us_quat`` / ``t_master_us_accel`` (``accel`` includes
gravity; there is no ``linear_accel`` and no gyro); ``/optical/t_master_us``;
``/phases``; ``/meta`` attributes (no ``subject_id`` in the pilot files).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm.io.session_reader import montage_corrected
from ml.config.classes import MUSCLE_CHANNEL_NAMES


@dataclass(frozen=True)
class PhaseInterval:
    name: str
    kind: str
    t_start_us: int
    t_end_us: int


@dataclass
class SessionData:
    path: Path
    session_id: str
    subject_id: str                      # "" when it cannot be resolved (see resolve_subject_id)
    emg_samples_uv: np.ndarray           # (N, C_all) float32
    emg_t_us: np.ndarray                 # (N,) int64
    emg_sample_rate_sps: float
    emg_channel_names: tuple[str, ...]
    leadoff_status: np.ndarray | None
    optical_t_us: np.ndarray | None
    lsb_uv: np.ndarray | None = None     # per channel
    phases: list[PhaseInterval] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def phase(self, name: str) -> PhaseInterval | None:
        return next((p for p in self.phases if p.name == name), None)

    def experiment_phases(self) -> list[PhaseInterval]:
        exp = [p for p in self.phases if p.kind == "experiment"]
        return sorted(exp, key=lambda p: p.t_start_us)

    def channel_index(self, names: list[str] | tuple[str, ...]) -> list[int]:
        idx = []
        for n in names:
            if n not in self.emg_channel_names:
                raise ValueError(f"missing EMG channel {n!r} in {self.emg_channel_names}")
            idx.append(self.emg_channel_names.index(n))
        return idx

    def full_scale_uv(self, names: list[str] | tuple[str, ...]) -> np.ndarray | None:
        if self.lsb_uv is None:
            return None
        return np.array([self.lsb_uv[i] * 8388607.0 for i in self.channel_index(names)])


def canonical_session_id(path: str | Path, meta: dict[str, Any] | None = None) -> str:
    """``/meta/session_id`` unless generic, else the folder name (``session_*`` or any folder
    holding a ``session.h5``), else the file stem."""
    path = Path(path)
    raw = str((meta or {}).get("session_id") or "").strip()
    if raw and raw != "session":
        return raw
    if path.parent.name.startswith("session_") or path.stem == "session":
        return path.parent.name
    return path.stem


def resolve_subject_id(path: str | Path, meta: dict[str, Any] | None = None) -> str:
    """Subject id: ``/meta/subject_id``, else the participant id parsed from the folder
    name (``session_<pid>_<sex>_<age>_...``, via ``biomech.cohort``), else
    ``/meta/experiment_name``; "" when none exists."""
    meta = meta or {}
    sid = str(meta.get("subject_id") or "").strip()
    if sid and sid.lower() != "unknown":
        return sid
    try:
        from biomech.cohort import parse_session_name
    except ImportError:          # biomech is optional for the legacy CLI
        parse_session_name = None
    if parse_session_name is not None:
        info = parse_session_name(Path(path).parent.name)
        if info:
            return str(info["pid"])
    return str(meta.get("experiment_name") or "").strip()


def load_session(path: str | Path) -> SessionData:
    """EMG, optical timestamps, phases and metadata (IMU via :func:`load_imu`)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as f:
        if "emg" not in f:
            raise ValueError(f"session missing /emg group: {path}")
        emg = f["emg"]
        rate = float(emg.attrs.get("sample_rate_sps", 0))
        if rate <= 0:
            raise ValueError(f"invalid emg sample_rate_sps in {path}")
        # as cabled: files before 2026-11-03 swap the digastric names (an03 §1)
        names = tuple(montage_corrected(list(_decode_names(emg.attrs.get("channel_names"))))[0])
        missing = [n for n in MUSCLE_CHANNEL_NAMES if n not in names]
        if missing:
            raise ValueError(f"session emg missing muscle channels: {missing}")
        lsb = None
        if "lsb_uv_per_channel" in emg.attrs:
            lsb = np.asarray(emg.attrs["lsb_uv_per_channel"], dtype=np.float64)
        elif "lsb_uv" in emg.attrs:
            lsb = np.full(len(names), float(emg.attrs["lsb_uv"]))
        optical_t = None
        if "optical" in f and "t_master_us" in f["optical"]:
            optical_t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        sd = SessionData(
            path=path, session_id="", subject_id="",
            emg_samples_uv=np.asarray(emg["samples_uv"][()], dtype=np.float32),
            emg_t_us=np.asarray(emg["t_master_us"][()], dtype=np.int64),
            emg_sample_rate_sps=rate, emg_channel_names=names,
            leadoff_status=np.asarray(emg["leadoff_status"][()]) if "leadoff_status" in emg
            else None,
            optical_t_us=optical_t, lsb_uv=lsb, phases=_load_phases(f), meta=_load_meta(f))
    sd.session_id = canonical_session_id(path, sd.meta)
    sd.subject_id = resolve_subject_id(path, sd.meta)
    return sd


def load_imu(path: str | Path):
    """Sanitized IMU streams of both slots (:class:`ml.core.dataset.Imu`) or None.

    Quaternion and accel samples with accuracy 0 (fusion not settled) are dropped and
    each stream is sorted with repeated stamps removed.
    """
    from ml.core.dataset import Imu
    from ml.core.signals import sanitize_timestamps

    out = {}
    with h5py.File(path, "r") as f:
        for slot in ("mta", "cra"):
            g = f.get(f"imu_{slot}")
            if g is None or "quat" not in g or "accel" not in g:
                return None
            for kind, key, tkey in (("q", "quat", "t_master_us_quat"),
                                    ("a", "accel", "t_master_us_accel")):
                v = np.asarray(g[key][()], dtype=np.float64)
                t = np.asarray(g[tkey][()], dtype=np.int64)
                acc = g.get(f"{key}_accuracy")
                ok = np.asarray(acc[()]) != 0 if acc is not None else np.ones(len(t), bool)
                if ok.sum() < 2:
                    return None
                out[f"{kind}_{slot}"] = sanitize_timestamps(t[ok], v[ok])
    return Imu(out["q_mta"], out["q_cra"], out["a_mta"], out["a_cra"])


def select_muscle_channels(samples_uv: np.ndarray, channel_names: tuple[str, ...], *,
                           expected: tuple[str, ...] | list[str] = MUSCLE_CHANNEL_NAMES
                           ) -> np.ndarray:
    """(N, C) muscle matrix in ``expected`` order (selection by name, never by position)."""
    idx = []
    for name in expected:
        if name not in channel_names:
            raise ValueError(f"missing muscle channel {name!r} in {channel_names}")
        idx.append(channel_names.index(name))
    return samples_uv[:, idx]


def _decode_names(raw: Any) -> tuple[str, ...]:
    if raw is None:
        raise ValueError("emg channel_names attribute missing")
    if isinstance(raw, (bytes, str)):
        raw = [raw]
    return tuple(x.decode() if isinstance(x, bytes) else str(x) for x in raw)


def _load_phases(f: h5py.File) -> list[PhaseInterval]:
    g = f.get("phases")
    if g is None or "t_start_us" not in g:
        return []
    names = [_s(x) for x in g["name"][()]]
    kinds = [_s(x) for x in g["kind"][()]] if "kind" in g else ["experiment"] * len(names)
    t0 = np.asarray(g["t_start_us"][()], dtype=np.int64)
    t1 = np.asarray(g["t_end_us"][()], dtype=np.int64)
    return [PhaseInterval(n, k, int(a), int(b)) for n, k, a, b in zip(names, kinds, t0, t1)]


def _load_meta(f: h5py.File) -> dict[str, Any]:
    if "meta" not in f:
        return {}
    meta = {k: (_s(v) if isinstance(v, bytes) else v) for k, v in f["meta"].attrs.items()}
    for key in ("experiment_plan_json", "mcu_config_json", "host_config_json"):
        if isinstance(meta.get(key), str):
            try:
                meta[key] = json.loads(meta[key])
            except json.JSONDecodeError:
                pass
    return meta


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
