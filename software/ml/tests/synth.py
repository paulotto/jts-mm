# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic sessions, MVC takes and label sidecars for the tests (never real data).

The layout follows the recorded pilot files: ``/emg/samples_uv`` (N, 8) in the
recorded channel order with ``channel_names``, ``lsb_uv_per_channel`` and
``leadoff_status``; ``/imu_mta|imu_cra/{quat, accel}`` with their own time
vectors (accel includes gravity); ``/optical/t_master_us`` at 40 Hz; ``/phases``;
``/meta`` without ``subject_id`` (the subject comes from the folder name).
EMG amplitude and IMU rotation depend on the class, so the data is learnable.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ml.config.classes import MUSCLE_CHANNEL_NAMES, UNLABELLED
from ml.io.labels import write_labels

FS = 1000.0
OPT_HZ = 40.0
IMU_HZ = 100.0
T0 = 10_000_000                       # first sample on the master clock (us)
# phase, duration (s), classes cycled every 0.5 s
PHASES = (("rest", 3.0, (0,)), ("open_close", 6.0, (0, 1, 0, 2)),
          ("left_right", 6.0, (0, 5, 0, 6)), ("free_motion", 3.0, (0, 1, 0, 5)))
GAP_S = 0.5
# channel gain per class (rows: class 0..6, cols: muscles in file order)
GAIN = np.array([[1, 1, 1, 1, 1, 1],
                 [1, 1, 8, 8, 1, 1],
                 [6, 6, 1, 1, 6, 6],
                 [3, 1, 3, 1, 3, 1],
                 [1, 3, 1, 3, 1, 3],
                 [8, 1, 1, 1, 1, 8],
                 [1, 8, 1, 1, 8, 1]], dtype=float)


def folder(root: Path, pid: str = "s01", mvc: bool = False, stamp: str = "20250302T101000Z"
           ) -> Path:
    d = Path(root) / f"session_{pid}_f_23{'_mvc' if mvc else ''}_{stamp}_ab12cd34"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _timeline(phases=PHASES):
    out, t = [], T0 + 1_000_000
    for name, dur, classes in phases:
        out.append((name, t, t + int(dur * 1e6), classes))
        t += int((dur + GAP_S) * 1e6)
    return out, t + 1_000_000


def label_track(phases=PHASES):
    """Per-phase optical times and labels (what biomech's ``labels`` stage stores)."""
    tl, t_end = _timeline(phases)
    opt = np.arange(T0, t_end, int(1e6 / OPT_HZ), dtype=np.int64)
    out = {}
    for name, a, b, classes in tl:
        t = opt[(opt >= a) & (opt <= b)]
        k = ((t - a) // 500_000) % len(classes)
        out[name] = (t, np.asarray(classes)[k].astype(np.int64))
    return out, opt


def write_session(path_dir: Path, *, seed: int = 0, phases=PHASES, imu: bool = True,
                  meta_subject: str | None = None, emg_gap_s: float | None = None) -> Path:
    """session.h5 in ``path_dir``; ``emg_gap_s`` drops EMG samples (a recording gap)."""
    rng = np.random.default_rng(seed)
    tl, t_end = _timeline(phases)
    tracks, opt = label_track(phases)
    t = np.arange(T0, t_end, int(1e6 / FS), dtype=np.int64)
    cls = np.zeros(t.size, dtype=int)
    for name, (lt, ly) in tracks.items():
        j = np.searchsorted(lt, t, side="right") - 1
        inside = (j >= 0) & (t <= lt[-1] + 25_000) & (t >= lt[0])
        cls[inside] = ly[j[inside]]
    emg = rng.normal(size=(t.size, 6)) * 10.0 * GAIN[cls]
    samples = np.column_stack([emg, rng.normal(size=(t.size, 2))]).astype(np.float32)
    if emg_gap_s:
        a = T0 + int(1.5e6) + int(tl[1][1] - T0)            # inside open_close
        keep = (t < a) | (t >= a + int(emg_gap_s * 1e6))
        t, samples = t[keep], samples[keep]
    h5 = Path(path_dir) / "session.h5"
    with h5py.File(h5, "w") as f:
        m = f.create_group("meta")
        m.attrs["experiment_name"] = Path(path_dir).name
        if meta_subject:
            m.attrs["subject_id"] = meta_subject
        e = f.create_group("emg")
        e.create_dataset("samples_uv", data=samples)
        e.create_dataset("t_master_us", data=t.astype(np.uint64))
        e.create_dataset("leadoff_status", data=np.zeros((t.size, 2), np.uint8))
        e.attrs["sample_rate_sps"] = int(FS)
        e.attrs["channel_names"] = np.array(list(MUSCLE_CHANNEL_NAMES) + ["bias_debug", "drl"],
                                            dtype="S12")
        e.attrs["lsb_uv_per_channel"] = np.array([0.0894] * 6 + [0.02235] * 2)
        o = f.create_group("optical")
        o.create_dataset("t_master_us", data=opt.astype(np.uint64))
        g = f.create_group("phases")
        g.create_dataset("name", data=np.array([p[0] for p in tl], dtype=object),
                         dtype=h5py.string_dtype())
        g.create_dataset("kind", data=np.array(["experiment"] * len(tl), dtype=object),
                         dtype=h5py.string_dtype())
        g.create_dataset("t_start_us", data=np.array([p[1] for p in tl], dtype=np.uint64))
        g.create_dataset("t_end_us", data=np.array([p[2] for p in tl], dtype=np.uint64))
        if imu:
            ti = np.arange(T0, t_end, int(1e6 / IMU_HZ), dtype=np.int64)
            ci = cls[np.clip(np.searchsorted(t, ti), 0, t.size - 1)]
            ang = np.cumsum(np.select([ci == 1, ci == 2, ci == 5, ci == 6],
                                      [0.01, -0.01, 0.02, -0.02], 0.0))
            for slot, k in (("mta", 1.0), ("cra", 0.1)):
                q = np.column_stack([np.cos(k * ang / 2), np.sin(k * ang / 2),
                                     np.zeros_like(ang), np.zeros_like(ang)])
                a = np.tile([0.0, 0.0, 9.80665], (ti.size, 1)) + rng.normal(size=(ti.size, 3)) * .05
                s = f.create_group(f"imu_{slot}")
                s.create_dataset("quat", data=q.astype(np.float32))
                s.create_dataset("t_master_us_quat", data=ti.astype(np.uint64))
                s.create_dataset("quat_accuracy", data=np.full(ti.size, 3, np.uint8))
                s.create_dataset("accel", data=a.astype(np.float32))
                s.create_dataset("t_master_us_accel", data=ti.astype(np.uint64))
                s.create_dataset("accel_accuracy", data=np.full(ti.size, 3, np.uint8))
    return h5


def write_mvc(path_dir: Path, *, seed: int = 1) -> Path:
    phases = (("mvc_opening", 3.0, (0,)), ("mvc_closing", 3.0, (0,)))
    h5 = write_session(path_dir, seed=seed, phases=phases, imu=False)
    with h5py.File(h5, "a") as f:
        x = f["emg/samples_uv"][()]
        x[:, :6] *= 20.0
        f["emg/samples_uv"][...] = x
    return h5


def write_sidecar(h5: Path, out: Path | None = None, *, subject: str = "", mvc: Path | None = None,
                  unlabel: tuple[str, ...] = (), ref_uv: dict | None = None) -> Path:
    """Sidecar on the optical grid with the synthetic labels; phases in ``unlabel`` -> 255."""
    tracks, opt = label_track()
    t = np.concatenate([v[0] for v in tracks.values()])
    y = np.concatenate([np.full(v[0].size, UNLABELLED) if k in unlabel else v[1]
                        for k, v in tracks.items()])
    extra = {"session_path": str(Path(h5).resolve())}
    if mvc is not None:
        extra["mvc_session"] = str(Path(mvc).resolve())
    if ref_uv is not None:
        extra["mvc_ref_uv"] = ref_uv
    return write_labels(out or Path(h5).with_name("session_labels.h5"), t_master_us=t,
                        optical_frame=np.searchsorted(opt, t), y=y.astype(np.uint8),
                        session_id=Path(h5).parent.name, subject_id=subject,
                        extra_attrs=extra)


REST_CONFIG = """
dataset:
  emg: {normalization: rest}
  window: {warmup_s: 0.5}
train: {model: lda, feature_set: emg, protocol: loso_subject, search: false}
paths: {data_dir: %s}
"""


def rest_config(tmp: Path) -> Path:
    Path(tmp).mkdir(parents=True, exist_ok=True)
    p = Path(tmp) / "ml_test.yaml"
    p.write_text(REST_CONFIG % (Path(tmp) / "ml_data"), encoding="utf-8")
    return p
