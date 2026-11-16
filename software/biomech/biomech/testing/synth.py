# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic participants with a known ground truth (W5 end-to-end tests).

``jtsmm.tools.mock_session`` writes the right file format, but its kinematics,
IMU and EMG are arbitrary. This module writes, per participant, a motion take
and an MVC take whose every signal follows from one physical model:

* anatomy in the maxillary frame (x anterior, y left, z up, origin = upper
  incisor): upper/lower incisor and first molars, skin-palpated condyles;
* jaw motion per task: open_close and chewing as a hinge about a fixed axis
  through the condyles plus a *delayed* translation along ``u`` (rotation
  first, translation from 40 % of the opening on; ``translation_mm`` = 0 is a
  pure hinge), protrusion and laterotrusion as translations, free_motion as a
  rich mixture (identifies the IMU calibration);
* optical poses of the headpiece (HP/CRA), mandibular array (MP/MTA) and the
  CT pointer tip in a camera world, with white pose noise; the CT tip rests on
  each landmark during its hold phase;
* two IMUs rigidly mounted on the arrays: game-rotation-vector quaternions
  (independent heading per IMU) and raw accelerometer *including gravity*,
  consistent with the poses (known mounting X, A and heading offset);
* six EMG channels: band-limited noise bursts whose onsets precede the
  mechanical onset by ``lead_ms`` (openers before opening, closers before
  closing), plus mains hum with harmonics, DC offsets and optional lead-off;
* an MVC take with mvc_opening / mvc_closing contractions.

The file skeleton (meta, triggers, cameras, sync report) comes from
``write_mock_session``; the measured groups are replaced. Everything returned
in :class:`Truth` is what the analysis should recover.
"""

from __future__ import annotations

import json
import secrets
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

from jtsmm.config.catalog import FULL_CHANNEL_NAMES, channel_lsb_uv
from jtsmm.imu.gravity_lin_accel import G
from jtsmm.tools.mock_session import write_mock_session

# -- anatomy (maxillary frame, mm) ---------------------------------------------------------
UPPER = {"incisor": (0.0, 0.0, 0.0), "left": (-38.0, 24.0, 0.0), "right": (-38.0, -24.0, 0.0)}
LOWER = {"incisor": (-1.5, 0.0, -2.5), "left": (-37.0, 23.0, -1.0), "right": (-37.0, -23.0, -1.0)}
CONDYLES = {"tmj_left": (-88.0, 52.0, 18.0), "tmj_right": (-88.0, -52.0, 18.0)}
HOLDS = (("mand_point_1", "calibration_mandibular", "lower", "incisor"),
         ("mand_point_2", "calibration_mandibular", "lower", "left"),
         ("mand_point_3", "calibration_mandibular", "lower", "right"),
         ("max_point_1", "calibration_maxillary", "upper", "incisor"),
         ("max_point_2", "calibration_maxillary", "upper", "left"),
         ("max_point_3", "calibration_maxillary", "upper", "right"),
         ("tmj_left", "calibration_tmj", "condyle", "tmj_left"),
         ("tmj_right", "calibration_tmj", "condyle", "tmj_right"))
TASKS = (("rest", 8.0), ("open_close", 12.0), ("protrusion_retrusion", 10.0),
         ("left_right", 12.0), ("chewing", 9.0), ("free_motion", 14.0))
# The generator's muscle order (the recorder's channel order until 2026-11-03). Every random draw
# is made in this order, so each muscle gets the same synthetic signal as before the montage fix
# (an03 §1); the columns go into the recorder's channel order (FULL_CHANNEL_NAMES) when written.
MUSCLES = ("masseter_r", "temporalis_r", "digastric_l", "digastric_r", "masseter_l",
           "temporalis_l")
_FILE_COLS = [MUSCLES.index(m) for m in FULL_CHANNEL_NAMES[:6]]   # file column j <- MUSCLES
OPENERS = ("digastric_l", "digastric_r")
CLOSERS = ("masseter_r", "temporalis_r", "masseter_l", "temporalis_l")


@dataclass
class Spec:
    """What to synthesize for one participant."""

    pid: str = "s01"
    sex: str = "f"
    age: int = 30
    translation_mm: float = 0.0          # delayed translation at full opening (0 = pure hinge)
    theta_max_deg: float = 25.0          # open_close peak opening angle
    lead_ms: float = 80.0                # EMG onset before the true movement start
    protrusion_mm: float = 7.0
    lateral_mm: float = 8.0
    emg_fs: int = 4000
    emg_band_hz: tuple[float, float] = (20.0, 450.0)   # tissue low-pass: lower = lower MNF
    mains_hz: float = 50.0
    leadoff_channel: str | None = None   # a channel with a 1.5-s lead-off in free_motion
    pose_noise_mm: float = 0.04
    pose_noise_deg: float = 0.03
    seed: int = 0
    # Optional timing-truth cases (analysis_plan D23); the defaults reproduce DEFAULT_SPECS
    # exactly (same schedule, same random draws).
    open_close_period_s: float | None = None   # None: 1.4 s (cycles back to back)
    task_s: dict[str, float] | None = None     # task durations (s) overriding TASKS
    emg_decoupled: bool = False                # opener/closer bursts at random times in the
                                               # open_close and chewing phases, unrelated to
                                               # the motion (own generator, seed + 7919)
    # C2-E timing-truth extensions (X6 (change log)). Defaults off; each
    # random part has its own generator (seed + an offset), so DEFAULT_SPECS and the D23
    # fixtures stay byte-identical (test_r11_c2e_units). Times: lead > 0 = EMG first.
    closer_lead_ms: float | None = None        # closer onset before the closing onset (None: lead_ms)
    opener_lead_ms: float | None = None        # opener onset before the opening onset (None: lead_ms)
    lead_jitter_ms: float = 0.0                # per-burst SD of the lead (normal; seed + 15485863)
    period_jitter_s: float = 0.0               # SD of the start-to-start period of open_close and
                                               # chewing; the movement duration scales with the
                                               # period (seed + 104729)
    closer_tonic_uv: float = 0.0               # tonic closer level from the end of a closing to
                                               # the release before the next opening
    closer_release_ms: float | None = None     # tonic level ends (its 50 % point) this long before
                                               # the opening onset (None: 0)
    emg_lock: str = "motion"                   # bursts locked to: motion (the upcoming onset) |
                                               # previous_event (the anchor: end of the previous
                                               # closing for openers, the opening onset for
                                               # closers; mean lead unchanged) | none (random)
    emg_model: str = "burst"                   # burst | torque (envelope ~ the muscle's share of
                                               # the angular acceleration; timing_arrays only)
    emd_ms: float = 0.0                        # torque model: EMG leads the acceleration by this
    quat_noise_deg: float = 0.0                # white rotation noise per IMU orientation (timing_arrays)
    quat_latency_ms: float = 0.0               # IMU orientation stream late against EMG (timing_arrays)
    head_motion_deg: float = 0.0               # head rotation amplitude, 0.25 Hz (timing_arrays)
    tilt_error_deg: float = 0.0                # jaw-IMU gravity tilt error -> |a_rel| leak (timing_arrays)
    emg_artifact_uv: float = 0.0               # motion artefact ~ the closing acceleration, added
                                               # to the closers (timing_arrays)
    reflex_ms: float | None = None             # EMG *after* the movement onset by this (overrides
                                               # the leads; torque model: emd = -reflex_ms)
    burst_dropout: float = 0.0                 # per channel-cycle probability that a burst is
                                               # absent (r10-like usable fractions; seed + 7877)


@dataclass
class Truth:
    spec: dict[str, Any]
    hinge_axis_n: list[float]            # maxillary frame
    hinge_axis_s: list[float]            # a point on the axis (condyle midpoint level)
    translation_dir: list[float]
    landmarks_max: dict[str, list[float]]   # closed-jaw positions, maxillary frame
    open_close_peaks_deg: list[float]
    opening_mm: float                    # vertical incisal displacement at the open_close peak
    X_mta_mp: list[list[float]]          # IMU calibration ground truth
    A_max_cra: list[list[float]]
    psi_rad: float
    motion_dir: str = ""
    mvc_dir: str = ""
    cycles: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    # Per cycle, at its peak: incisor displacement from the closed pose (maxillary
    # frame, mm) and the sagittal angle (deg) -- what the cycle metrics should read.
    peak_incisor_mm: dict[str, list[list[float]]] = field(default_factory=dict)
    peak_theta_deg: dict[str, list[float]] = field(default_factory=dict)


# -- small SE(3) helpers ---------------------------------------------------------------
def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.zeros(np.broadcast_shapes(R.shape[:-2], np.shape(t)[:-1]) + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def _inv(T: np.ndarray) -> np.ndarray:
    R = np.swapaxes(T[..., :3, :3], -1, -2)
    return _T(R, -np.einsum("...ij,...j->...i", R, T[..., :3, 3]))


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,j->...i", T[..., :3, :3], p) + T[..., :3, 3]


def _raised(s: np.ndarray) -> np.ndarray:
    """0 -> 1 smooth step on s in [0, 1] (zero velocity at both ends)."""
    s = np.clip(s, 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * s))


def _cycle_profile(t: np.ndarray, starts: list[float], dur: float) -> np.ndarray:
    """0 -> 1 -> 0 raised-cosine excursion of length ``dur`` from every start."""
    out = np.zeros_like(t)
    for t0 in starts:
        s = (t - t0) / dur
        m = (s >= 0) & (s <= 1)
        out[m] = 0.5 * (1.0 - np.cos(2 * np.pi * s[m]))
    return out


# -- schedule --------------------------------------------------------------------------
def _schedule(gap_s: float = 0.8, hold_s: float = 2.5, task_s: dict[str, float] | None = None
              ) -> list[tuple[str, str, float, float]]:
    out, t = [], 1.0
    for name, kind, *_ in HOLDS:
        out.append((name, kind, t, t + hold_s))
        t += hold_s + gap_s
    for name, dur in TASKS:
        dur = float((task_s or {}).get(name, dur))
        out.append((name, "experiment", t, t + dur))
        t += dur + gap_s
    return out


def _task_cycles(name: str, t0: float, t1: float, open_close_period_s: float | None = None,
                 period_jitter_s: float = 0.0, rng: np.random.Generator | None = None
                 ) -> list[tuple[float, float, float]]:
    """(start, peak, end) seconds of the movement cycles of a task.

    ``period_jitter_s`` > 0 (open_close and chewing, with ``rng``): each cycle's period is
    drawn as period + N(0, period_jitter_s) (clipped to 0.5 ... 1.5 x period) and its movement
    duration scales by the same factor, so the pause and the opening duration vary from cycle
    to cycle (what the S2 cycle shuffle needs to tell rhythm locking from onset locking)."""
    per = {"open_close": (open_close_period_s or 1.4, 1.0), "protrusion_retrusion": (1.6, 1.2),
           "left_right": (1.6, 1.2), "chewing": (0.85, 0.7)}.get(name)
    if per is None:
        return []
    period, dur = per
    jitter = (period_jitter_s > 0 and rng is not None and name in ("open_close", "chewing"))
    out, s = [], t0 + 0.8
    while True:
        f = (float(np.clip(1.0 + rng.normal(0.0, period_jitter_s) / period, 0.5, 1.5))
             if jitter else 1.0)
        d = dur * f
        if not s + d < t1 - 0.4:
            break
        out.append((s, s + d / 2, s + d))
        s += period * f
    return out


# -- the motion model (maxillary frame) --------------------------------------------------
class Motion:
    def __init__(self, spec: Spec, schedule: list[tuple[str, str, float, float]],
                 rng: np.random.Generator) -> None:
        self.spec = spec
        self.sched = schedule
        self.n = np.array([0.0, 1.0, 0.0])                       # opening = +rotation about +y
        cl, cr = np.array(CONDYLES["tmj_left"]), np.array(CONDYLES["tmj_right"])
        self.s = 0.5 * (cl + cr) + np.array([2.0, 0.0, -4.0])    # just below the skin points
        u = np.array([1.0, 0.0, -0.5])
        self.u = u / np.linalg.norm(u)
        jit_rng = (np.random.default_rng(spec.seed + 104729) if spec.period_jitter_s > 0
                   else None)
        self.cycles = {name: _task_cycles(name, a, b, spec.open_close_period_s,
                                          spec.period_jitter_s, jit_rng)
                       for name, _, a, b in schedule}
        self.task = {name: (a, b) for name, _, a, b in schedule}
        self.peaks: dict[str, list[float]] = {}
        # Per-cycle amplitude jitter (+-8 %) so cycles are not identical.
        self.jit = {name: 1.0 + 0.08 * rng.uniform(-1, 1, len(c)) for name, c in self.cycles.items()}
        self.free = self._free_motion_coeffs(rng)

    def _free_motion_coeffs(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        f = rng.uniform(0.15, 0.6, (6, 3))
        ph = rng.uniform(0, 2 * np.pi, (6, 3))
        return {"f": f, "ph": ph}

    def angles(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(theta deg about +y, translation tau mm along u, extra translation (N, 3) mm, extra
        rotation vector (N, 3) deg) of the analytic motion at times ``t`` (s)."""
        sp = self.spec
        theta = np.zeros_like(t)       # deg about +y through s
        tau = np.zeros_like(t)         # mm along u
        trans = np.zeros(t.shape + (3,))
        rot_extra = np.zeros(t.shape + (3,))    # small extra rotation vector (deg)
        for name, cyc in self.cycles.items():
            if not cyc:
                continue
            for i, ((s0, _pk, s1), k) in enumerate(zip(cyc, self.jit[name])):
                prof = _cycle_profile(t, [s0], s1 - s0)
                if name in ("open_close", "chewing"):
                    th_max = sp.theta_max_deg * (0.5 if name == "chewing" else 1.0) * k
                    theta += th_max * prof
                    frac = np.clip((prof - 0.4) / 0.6, 0, 1)
                    tau += sp.translation_mm * k * _raised(frac) * (
                        0.5 if name == "chewing" else 1.0)
                    if name == "chewing":
                        trans[:, 1] += 1.5 * np.sin(2 * np.pi * (t - s0) / (s1 - s0)) * prof
                elif name == "protrusion_retrusion":
                    d = np.array([1.0, 0.0, -0.35])
                    trans += np.outer(prof * sp.protrusion_mm * k, d / np.linalg.norm(d))
                    theta += 3.0 * prof * k
                elif name == "left_right":
                    side = 1.0 if i % 2 == 0 else -1.0          # left, right, left, ...
                    trans[:, 1] += side * sp.lateral_mm * k * prof
                    trans[:, 2] -= 1.0 * prof
                    rot_extra[:, 2] += side * 4.0 * prof        # Bennett-like rotation about z
                    theta += 2.0 * prof
        a, b = self.task["free_motion"]
        m = (t >= a) & (t <= b)
        if m.any():
            w = _raised(np.minimum((t[m] - a) / 1.0, (b - t[m]) / 1.0))
            tt = t[m][:, None]
            f, ph = self.free["f"], self.free["ph"]
            theta[m] += w * 10.0 * (1 + np.sin(2 * np.pi * f[0, 0] * tt[:, 0] + ph[0, 0]))
            rot_extra[m, 0] += w * 5.0 * np.sin(2 * np.pi * f[1, 0] * tt[:, 0] + ph[1, 0])
            rot_extra[m, 2] += w * 6.0 * np.sin(2 * np.pi * f[2, 0] * tt[:, 0] + ph[2, 0])
            trans[m, 0] += w * 3.0 * (1 + np.sin(2 * np.pi * f[3, 0] * tt[:, 0] + ph[3, 0]))
            trans[m, 1] += w * 3.0 * np.sin(2 * np.pi * f[4, 0] * tt[:, 0] + ph[4, 0])
        return theta, tau, trans, rot_extra

    def pose(self, t: np.ndarray) -> np.ndarray:
        """(N, 4, 4) mandible pose relative to its closed pose, maxillary frame."""
        theta, tau, trans, rot_extra = self.angles(t)
        R_h = Rotation.from_rotvec(np.outer(np.radians(theta), self.n)).as_matrix()
        R_x = Rotation.from_rotvec(np.radians(rot_extra)).as_matrix()
        R = R_x @ R_h
        # Hinge about the axis through s, then the delayed translation along u.
        t_vec = self.s - np.einsum("nij,j->ni", R_h, self.s)
        t_vec = np.einsum("nij,nj->ni", R_x, t_vec)
        t_vec += tau[:, None] * self.u + trans
        return _T(R, t_vec)


# -- writer ----------------------------------------------------------------------------
def _rand_rot(rng: np.random.Generator, max_deg: float = 180.0) -> np.ndarray:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return Rotation.from_rotvec(v * np.radians(rng.uniform(0.3, 1.0) * max_deg)).as_matrix()


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()            # x, y, z, w
    q = np.column_stack([q[:, 3], q[:, :3]])
    q[q[:, 0] < 0] *= -1.0
    return q


def _session_name(spec: Spec, mvc: bool, stamp: str) -> str:
    kind = "_mvc" if mvc else ""
    return f"session_{spec.pid}_{spec.sex}_{spec.age}{kind}_{stamp}_{secrets.token_hex(4)}"


def _replace(g: h5py.Group, name: str, data: np.ndarray, **attrs: Any) -> h5py.Dataset:
    if name in g:
        del g[name]
    ds = g.create_dataset(name, data=data)
    for k, v in attrs.items():
        ds.attrs[k] = v
    return ds


def _emg_check(spec: Spec, take_offset: int) -> dict[str, Any]:
    """The recorder's pre-take EMG check (``/meta/emg_check_json``): deterministic |Z| per
    channel (the second take a little lower, as the gel wets the skin)."""
    rng = np.random.default_rng(spec.seed + 77 + take_offset)
    ch = []
    for k, m in enumerate(MUSCLES):
        z = float(15.0 + 8.0 * k + rng.uniform(0, 10) - 3.0 * take_offset)
        noise = float(4.0 + 0.1 * z)
        ch.append({"channel": FULL_CHANNEL_NAMES.index(m) + 1, "name": m,
                   "leadoff_ok": z <= 50, "noise_rms_uv": noise,
                   "noise_ok": noise <= 10, "line_peak_uv": 2.0, "line_ok": True,
                   "z_kohm": z, "z_quality": "good" if z <= 50 else "poor", "notes": [],
                   "contact_hard_fail": z > 50, "noise_total_uv": noise + 0.5,
                   "comparator": "clear"})
    ch.sort(key=lambda c: c["channel"])                 # as the recorder lists them
    return {"passed": all(c["leadoff_ok"] and c["noise_ok"] for c in ch), "note": "",
            "channels": ch, "line_hz_measured": 50.0}


def _write_phases(f: h5py.File, phases: list[tuple[str, str, float, float]]) -> None:
    str_dt = h5py.string_dtype()
    for grp in ("phases", "events"):
        if grp in f:
            del f[grp]
    g = f.create_group("phases")
    g.create_dataset("name", data=np.asarray([p[0] for p in phases], dtype=str_dt))
    g.create_dataset("kind", data=np.asarray([p[1] for p in phases], dtype=str_dt))
    g.create_dataset("t_start_us", data=np.asarray([round(p[2] * 1e6) for p in phases], np.uint64))
    g.create_dataset("t_end_us", data=np.asarray([round(p[3] * 1e6) for p in phases], np.uint64))
    g.attrs["n_phases"] = len(phases)
    ev = f.create_group("events")
    t, kind, text, pk = [], [], [], []
    for name, k, a, b in phases:
        for edge, tt in (("start", a), ("end", b)):
            t.append(round(tt * 1e6))
            kind.append(f"phase_{edge}")
            text.append(name)
            pk.append(k)
    ev.create_dataset("t_master_us", data=np.asarray(t, np.uint64))
    ev.create_dataset("kind", data=np.asarray(kind, dtype=str_dt))
    ev.create_dataset("text", data=np.asarray(text, dtype=str_dt))
    ev.create_dataset("phase_kind", data=np.asarray(pk, dtype=str_dt))
    ev.create_dataset("t_host_ns", data=np.asarray(t, np.uint64) * 1000)
    ev.create_dataset("map_quality", data=np.asarray(["exact"] * len(t), dtype=str_dt))
    ev.create_dataset("viewer_t_s", data=np.full(len(t), np.nan))


def _emg(spec: Spec, t: np.ndarray, env: np.ndarray, rng: np.random.Generator,
         leadoff_span: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(samples int32, samples_uv float32, leadoff (n, 2) uint8) for 8 columns."""
    fs = spec.emg_fs
    n = t.size
    lo, hi = spec.emg_band_hz
    sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    noise = sosfiltfilt(sos, rng.normal(size=(n, 6)), axis=0)
    noise /= noise.std(axis=0)
    uv = noise * env
    hum = np.zeros(n)
    for h, a in ((1, 15.0), (3, 5.0), (5, 3.0), (7, 1.5), (9, 1.0)):
        if h * spec.mains_hz < 0.5 * fs:
            hum += a * np.sin(2 * np.pi * h * spec.mains_hz * t + rng.uniform(0, 2 * np.pi))
    uv = uv + hum[:, None] * rng.uniform(0.5, 1.5, 6)
    uv += rng.uniform(-3000, 3000, 6) + 50.0 * np.sin(2 * np.pi * 0.05 * t)[:, None]
    full = np.zeros((n, 8))
    full[:, :6] = uv[:, _FILE_COLS]                   # generator order -> recorder channel order
    lsb = np.asarray(channel_lsb_uv(6, 8), float)
    counts = np.round(full / lsb).astype(np.int32)
    counts[:, 6] = 0
    leadoff = np.zeros((n, 2), np.uint8)
    if leadoff_span is not None and spec.leadoff_channel:
        i = FULL_CHANNEL_NAMES.index(spec.leadoff_channel)      # bit of the file channel
        m = (t >= leadoff_span[0]) & (t <= leadoff_span[1])
        leadoff[m, 0] |= np.uint8(1 << i)
    return counts, (counts * lsb).astype(np.float32), leadoff


def _decoupled_bursts(spec: Spec, motion: Motion, burst: Any, io: list[int],
                      ic: list[int]) -> None:
    """EMG unrelated to the motion (D23 null case): per task as many opener and closer
    bursts as cycles, of the coupled durations and amplitudes, at times drawn uniformly in
    the phase (an own generator, so the other signals are drawn as in the coupled case)."""
    rng = np.random.default_rng(spec.seed + 7919)
    for name in ("open_close", "chewing"):
        cyc = motion.cycles.get(name, [])
        if not cyc:
            continue
        a, b = motion.task[name]
        for s0, pk, s1 in cyc:
            d_open, d_close = pk - s0, s1 - pk
            u = rng.uniform(a, b - d_open)
            burst(io, u, u + d_open, 60.0 if name == "open_close" else 40.0)
            u = rng.uniform(a, b - d_close)
            burst(ic, u, u + d_close, 50.0 if name == "open_close" else 90.0)


RAMP_S = 0.02                       # burst on/off ramp (the 50 % point sits RAMP_S / 2 inside)
EMG_LOCKS = ("motion", "previous_event", "none")
EMG_MODELS = ("burst", "torque")
# Spec fields only the in-memory simulator (:func:`timing_arrays`) honours.
TIMING_ONLY = ("quat_noise_deg", "quat_latency_ms", "head_motion_deg", "tilt_error_deg",
               "emg_artifact_uv")


def _leads_s(spec: Spec) -> tuple[float, float]:
    """(opener, closer) lead in s (> 0: EMG before the movement onset); ``reflex_ms`` puts
    both after the onset."""
    if spec.reflex_ms is not None:
        return -spec.reflex_ms * 1e-3, -spec.reflex_ms * 1e-3
    lead = spec.lead_ms * 1e-3
    lo = lead if spec.opener_lead_ms is None else spec.opener_lead_ms * 1e-3
    lc = lead if spec.closer_lead_ms is None else spec.closer_lead_ms * 1e-3
    return lo, lc


def burst_schedule(spec: Spec, motion: Motion, name: str) -> list[dict[str, Any]]:
    """Per cycle of task ``name`` the synthetic EMG truth (s): ``cycle`` (start, peak, end),
    the ``opener`` and ``closer`` burst spans, the ``anchor_open`` (end of the previous closing;
    for the first cycle its start minus the mean pause) and the closer
    ``release`` time with its ``tonic`` span (None without ``closer_tonic_uv``).

    ``emg_lock`` motion: bursts start lead (+ jitter) before the upcoming onset;
    previous_event: at the mean lead from the anchor (openers: previous closing end + mean
    pause - lead; closers: opening onset + mean opening duration - lead), so the mean lead is
    the same but the bursts follow the rhythm, not the upcoming onset."""
    cyc = motion.cycles.get(name, [])
    if not cyc:
        return []
    lo, lc = _leads_s(spec)
    a_task = motion.task[name][0]
    ends = [a_task] + [c[2] for c in cyc[:-1]]                  # previous closing ends
    pause = float(np.mean([c[0] - e for c, e in zip(cyc[1:], ends[1:])])) if len(cyc) > 1 \
        else 0.4
    ends[0] = cyc[0][0] - pause
    d_open = float(np.mean([pk - s0 for s0, pk, _ in cyc]))
    jit = (np.random.default_rng([spec.seed + 15485863, zlib.crc32(name.encode())])
           .normal(0.0, spec.lead_jitter_ms * 1e-3, (len(cyc), 2))
           if spec.lead_jitter_ms > 0 else np.zeros((len(cyc), 2)))
    rel = 0.0 if spec.closer_release_ms is None else spec.closer_release_ms * 1e-3
    prev_lock = spec.emg_lock == "previous_event"
    out = []
    for k, ((s0, pk, s1), e) in enumerate(zip(cyc, ends)):
        jo, jc = jit[k]
        if prev_lock:
            o0, c0, r = e + pause - lo - jo, s0 + d_open - lc - jc, e + pause - rel
        else:
            o0, c0, r = s0 - lo - jo, pk - lc - jc, s0 - rel
        tonic = ((max(e, a_task) - RAMP_S, r + RAMP_S / 2) if spec.closer_tonic_uv > 0
                 else None)
        out.append({"cycle": (s0, pk, s1), "anchor_open": e, "opener": (o0, pk),
                    "closer": (c0, s1), "release": r if tonic else np.nan, "tonic": tonic})
    return out


def _alpha(motion: Motion, t: np.ndarray) -> np.ndarray:
    """Opening angular acceleration (deg/s^2) of the analytic motion at ``t`` (s)."""
    th = motion.angles(t)[0]
    return np.gradient(np.gradient(th, t), t) if t.size > 2 else np.zeros_like(t)


def _envelopes(spec: Spec, t: np.ndarray, motion: Motion | None,
               mvc: list[tuple[str, float, float]] | None, rng: np.random.Generator) -> np.ndarray:
    """(n, 6) RMS amplitude (uV) per muscle over time."""
    env = np.full((t.size, 6), 3.0) * rng.uniform(0.8, 1.2, 6)
    io = [MUSCLES.index(m) for m in OPENERS]
    ic = [MUSCLES.index(m) for m in CLOSERS]

    def burst(cols: list[int], a: float, b: float, amp: float) -> None:
        ramp = RAMP_S
        w = np.clip(np.minimum((t - a) / ramp, (b - t) / ramp), 0, 1)
        env[:, cols] = np.maximum(env[:, cols], amp * w[:, None])

    if motion is not None and (spec.emg_decoupled or spec.emg_lock == "none"):
        _decoupled_bursts(spec, motion, burst, io, ic)
    elif motion is not None and spec.emg_model == "torque":
        _torque_envelopes(spec, motion, t, env, io, ic)
    elif motion is not None:
        drop = (np.random.default_rng(spec.seed + 7877) if spec.burst_dropout > 0 else None)

        def keep(cols: list[int]) -> list[int]:
            return cols if drop is None else [c for c in cols
                                              if drop.random() >= spec.burst_dropout]

        for name in ("open_close", "chewing"):
            amp_o, amp_c = (60.0, 50.0) if name == "open_close" else (40.0, 90.0)
            for b in burst_schedule(spec, motion, name):
                burst(keep(io), *b["opener"], amp_o)
                burst(keep(ic), *b["closer"], amp_c)
                if b["tonic"] is not None:
                    burst(ic, *b["tonic"], spec.closer_tonic_uv)
    for name, a, b in mvc or []:
        for k in range(3):
            s = a + 0.6 + k * (b - a - 1.0) / 3
            burst(io if name == "mvc_opening" else ic, s, s + 1.6,
                  250.0 if name == "mvc_opening" else 300.0)
    return env


def _torque_envelopes(spec: Spec, motion: Motion, t: np.ndarray, env: np.ndarray,
                      io: list[int], ic: list[int]) -> None:
    """Torque model (X6 P5): in open_close and chewing the closer envelope follows the closing
    share of the angular acceleration, -alpha > 0, and the opener envelope +alpha > 0, both
    ``emd_ms`` early (EMG leads the acceleration by the electromechanical delay; ``reflex_ms``
    makes it late). Peak amplitudes as the burst model."""
    emd = (-spec.reflex_ms if spec.reflex_ms is not None else spec.emd_ms) * 1e-3
    for name in ("open_close", "chewing"):
        a, b = motion.task[name]
        m = (t >= a) & (t <= b)
        if m.sum() < 3:
            continue
        al = _alpha(motion, t[m] + emd)
        scale = float(np.percentile(np.abs(al), 99)) or 1.0
        amp_o, amp_c = (60.0, 50.0) if name == "open_close" else (40.0, 90.0)
        env[np.ix_(m, ic)] = np.maximum(env[np.ix_(m, ic)],
                                        (amp_c * np.clip(-al / scale, 0, 1))[:, None])
        env[np.ix_(m, io)] = np.maximum(env[np.ix_(m, io)],
                                        (amp_o * np.clip(al / scale, 0, 1))[:, None])


def _check_spec(spec: Spec) -> None:
    if spec.emg_lock not in EMG_LOCKS:
        raise ValueError(f"emg_lock {spec.emg_lock!r}; known: {EMG_LOCKS}")
    if spec.emg_model not in EMG_MODELS:
        raise ValueError(f"emg_model {spec.emg_model!r}; known: {EMG_MODELS}")


def write_participant(root: str | Path, spec: Spec, stamp: str = "20250302T101000Z") -> Truth:
    """Write the motion and MVC takes of one synthetic participant under ``root``.

    The IMU and artefact extensions of :data:`TIMING_ONLY` exist only in the in-memory
    simulator (:func:`timing_arrays`); a spec that sets one is refused here."""
    _check_spec(spec)
    bad = [k for k in TIMING_ONLY if getattr(spec, k)]
    if bad:
        raise ValueError(f"{bad} are honoured by timing_arrays only, not by the HDF5 writer")
    root = Path(root)
    rng = np.random.default_rng(spec.seed)
    sched = _schedule(task_s=spec.task_s)
    duration = sched[-1][3] + 1.0
    motion = Motion(spec, sched, rng)

    # Fixed placements: world = reference camera, gravity "up" is -y_cam.
    R_w_hp = _rand_rot(rng, 25.0) @ Rotation.from_euler("x", -90, degrees=True).as_matrix()
    T_w_hp = _T(R_w_hp, np.array([40.0, -65.0, 700.0]))
    T_max_hp = _T(_rand_rot(rng, 20.0), np.array([35.0, 0.0, 70.0]))    # forehead array
    T_max_mp0 = _T(_rand_rot(rng, 20.0), np.array([15.0, 0.0, -45.0]))  # chin array, closed
    T_w_max = T_w_hp @ _inv(T_max_hp)
    up_w = np.array([0.0, -1.0, 0.0])
    R_gz_w = Rotation.align_vectors([[0, 0, 1]], [up_w])[0].as_matrix()
    R_hp_sc, R_mp_sm = _rand_rot(rng), _rand_rot(rng)
    psi_c, psi_m = rng.uniform(-np.pi, np.pi, 2)
    r_sm_mp = np.array([10.0, 5.0, -8.0])          # IMU position on the chin array (MP frame)

    # Fine time grid for the analytic motion, then everything is sampled from it.
    fine_hz = 1000.0
    tf = np.arange(0.0, duration, 1.0 / fine_hz)
    M = motion.pose(tf)                                                  # max <- mand
    T_w_mp = T_w_max[None] @ M @ T_max_mp0[None]

    def at(times: np.ndarray, T: np.ndarray) -> np.ndarray:
        R = Slerp(tf, Rotation.from_matrix(T[:, :3, :3]))(np.clip(times, tf[0], tf[-1]))
        p = np.column_stack([np.interp(times, tf, T[:, k, 3]) for k in range(3)])
        return _T(R.as_matrix(), p)

    # --- the motion take ----------------------------------------------------------------
    mdir = root / _session_name(spec, False, stamp)
    mdir.mkdir(parents=True, exist_ok=True)
    write_mock_session(mdir / "session.h5", duration_s=duration, emg_fs=spec.emg_fs,
                       write_jts_config=False, seed=spec.seed)
    with h5py.File(mdir / "session.h5", "r+") as f:
        f["meta"].attrs["experiment_name"] = f"{spec.pid}_{spec.sex}_{spec.age}"
        f["meta"].attrs["created_unix_utc"] = "1791464472.0"
        f["meta"].attrs["emg_check_json"] = json.dumps(_emg_check(spec, 1))
        _write_phases(f, sched)
        # Optical (40 Hz trigger grid of the skeleton).
        t_opt_us = np.asarray(f["optical/t_master_us"][()], dtype=np.int64)
        to = t_opt_us * 1e-6
        noise_t = lambda n: rng.normal(0, spec.pose_noise_mm, (n, 3))          # noqa: E731
        noise_R = lambda n: Rotation.from_rotvec(                                # noqa: E731
            rng.normal(0, np.radians(spec.pose_noise_deg), (n, 3))).as_matrix()
        n = to.size
        P_hp = np.repeat(T_w_hp[None], n, axis=0)
        P_hp[:, :3, :3] = noise_R(n) @ P_hp[:, :3, :3]
        P_hp[:, :3, 3] += noise_t(n)
        P_mp = at(to, T_w_mp)
        P_mp[:, :3, :3] = noise_R(n) @ P_mp[:, :3, :3]
        P_mp[:, :3, 3] += noise_t(n)
        # CT pointer: its tip rests on the landmark during the hold, untracked otherwise.
        P_ct = np.full((n, 4, 4), np.nan)
        R_ct = _rand_rot(rng)
        lm_max = {**{f"upper_{k}": v for k, v in UPPER.items()},
                  **{f"lower_{k}": v for k, v in LOWER.items()}, **CONDYLES}
        for (name, _, jaw, role), (_, _, a, b) in zip(HOLDS, sched[:len(HOLDS)]):
            m = (to >= a) & (to <= b)
            key = role if jaw == "condyle" else f"{jaw}_{role}"
            tip_w = _apply(T_w_max, np.array(lm_max[key]))     # jaw closed during holds
            P_ct[m] = _T(R_ct, tip_w + rng.normal(0, 0.05, (m.sum(), 3)))
        for body, P in (("cra", P_hp), ("mta", P_mp), ("ct", P_ct)):
            _replace(f["optical"], f"pose_{body}", P.astype(np.float32))
            ok = np.isfinite(P.reshape(n, -1)).all(1)
            q = f["optical/quality"][body]
            _replace(q, "n_tags", np.where(ok, 3, 0).astype(np.uint8))
            _replace(q, "n_cameras", np.where(ok, 2, 0).astype(np.uint8))
            _replace(q, "reprojection_rms", np.where(ok, 0.3, np.nan).astype(np.float32))

        # IMUs: game rotation vector = Rz(psi) R_gz<-w R_w<-body R_body<-sensor.
        for slot, R_body_s, psi in (("imu_cra", R_hp_sc, psi_c), ("imu_mta", R_mp_sm, psi_m)):
            g = f[slot]
            tq = np.arange(0.0, duration, 1 / 390.0)
            ta = np.arange(0.0, duration, 1 / 256.0)
            Rz = Rotation.from_rotvec([0, 0, psi]).as_matrix()
            if slot == "imu_cra":
                R_ws = np.repeat((T_w_hp[:3, :3] @ R_body_s)[None], tq.size, 0)
                acc_lin_w = np.zeros((ta.size, 3))
                R_ws_a = np.repeat((T_w_hp[:3, :3] @ R_body_s)[None], ta.size, 0)
            else:
                R_ws = at(tq, T_w_mp)[:, :3, :3] @ R_body_s
                R_ws_a = at(ta, T_w_mp)[:, :3, :3] @ R_body_s
                p = _apply(T_w_mp, r_sm_mp) * 1e-3                   # m, on the fine grid
                acc = np.gradient(np.gradient(p, tf, axis=0), tf, axis=0)
                acc_lin_w = np.column_stack([np.interp(ta, tf, acc[:, k]) for k in range(3)])
            R_g = Rz @ R_gz_w
            q = _quat_wxyz(R_g @ R_ws)
            # Specific force in the sensor frame: R_s<-w (a + g up).
            f_w = acc_lin_w + G * up_w
            a_s = np.einsum("nji,nj->ni", R_ws_a, f_w) + rng.normal(0, 0.03, (ta.size, 3))
            for name in list(g.keys()):
                del g[name]
            g.create_dataset("quat", data=q.astype(np.float32))
            g.create_dataset("t_master_us", data=np.round(tq * 1e6).astype(np.uint64))
            g["t_master_us_quat"] = g["t_master_us"]
            del g["t_master_us"]
            g.create_dataset("quat_report_id", data=np.full(tq.size, 0x08, np.uint8))
            g.create_dataset("quat_accuracy", data=np.full(tq.size, 3, np.uint8))
            ds = g.create_dataset("accel", data=a_s.astype(np.float32))
            ds.attrs["units"] = "m/s^2"
            g.create_dataset("t_master_us_accel", data=np.round(ta * 1e6).astype(np.uint64))
            g.create_dataset("accel_accuracy", data=np.full(ta.size, 3, np.uint8))

        # EMG.
        te = np.arange(0.0, duration, 1.0 / spec.emg_fs)
        env = _envelopes(spec, te, motion, None, rng)
        fm = motion.task["free_motion"]
        counts, uv, lead = _emg(spec, te, env, rng, (fm[0] + 3.0, fm[0] + 4.5))
        e = f["emg"]
        for name, data in (("samples", counts), ("samples_uv", uv), ("leadoff_status", lead),
                           ("t_master_us", np.round(te * 1e6).astype(np.uint64)),
                           ("seq", np.arange(te.size, dtype=np.uint32))):
            _replace(e, name, data)

    # --- the MVC take -----------------------------------------------------------------------
    vdir = root / _session_name(spec, True, stamp.replace("T13", "T12"))
    vdir.mkdir(parents=True, exist_ok=True)
    vsched = [("mvc_opening", "mvc", 1.0, 9.0), ("mvc_closing", "mvc", 10.0, 18.0)]
    vdur = 19.0
    write_mock_session(vdir / "session.h5", duration_s=vdur, emg_fs=spec.emg_fs,
                       write_jts_config=False, seed=spec.seed + 1000)
    with h5py.File(vdir / "session.h5", "r+") as f:
        f["meta"].attrs["experiment_name"] = f"{spec.pid}_{spec.sex}_{spec.age}_mvc"
        f["meta"].attrs["created_unix_utc"] = "1791464292.0"       # 3 min before the motion
        f["meta"].attrs["emg_check_json"] = json.dumps(_emg_check(spec, 0))
        _write_phases(f, vsched)
        if "optical" in f:
            del f["optical"]
        te = np.arange(0.0, vdur, 1.0 / spec.emg_fs)
        env = _envelopes(spec, te, None, [(n_, a, b) for n_, _, a, b in vsched], rng)
        counts, uv, lead = _emg(spec, te, env, rng, None)
        e = f["emg"]
        for name, data in (("samples", counts), ("samples_uv", uv), ("leadoff_status", lead),
                           ("t_master_us", np.round(te * 1e6).astype(np.uint64)),
                           ("seq", np.arange(te.size, dtype=np.uint32))):
            _replace(e, name, data)

    # --- ground truth -----------------------------------------------------------------------
    oc = motion.cycles["open_close"]
    peaks = [spec.theta_max_deg * k for k in motion.jit["open_close"]]
    inc0 = np.array(LOWER["incisor"])
    peak_inc, peak_th = {}, {}
    for name, cyc in motion.cycles.items():
        if not cyc:
            continue
        Tp = motion.pose(np.array([c[1] for c in cyc]))
        peak_inc[name] = (_apply(Tp, inc0) - inc0).tolist()
        peak_th[name] = np.degrees(Rotation.from_matrix(Tp[:, :3, :3]).as_rotvec()[:, 1]).tolist()
    Mpk = motion.pose(np.array([oc[0][1]]))[0]
    inc0 = np.array(LOWER["incisor"])
    opening_mm = float(inc0[2] - _apply(Mpk, inc0)[2])
    truth = Truth(
        spec=asdict(spec), hinge_axis_n=motion.n.tolist(), hinge_axis_s=motion.s.tolist(),
        translation_dir=motion.u.tolist(),
        landmarks_max={**{f"upper_{k}": list(v) for k, v in UPPER.items()},
                       **{f"lower_{k}": list(v) for k, v in LOWER.items()},
                       **{k: list(v) for k, v in CONDYLES.items()}},
        open_close_peaks_deg=peaks, opening_mm=opening_mm,
        X_mta_mp=R_mp_sm.T.tolist(), A_max_cra=(T_max_hp[:3, :3] @ R_hp_sc).tolist(),
        psi_rad=float(np.angle(np.exp(1j * (psi_c - psi_m)))),
        motion_dir=mdir.name, mvc_dir=vdir.name,
        cycles={k: v for k, v in motion.cycles.items() if v},
        peak_incisor_mm=peak_inc, peak_theta_deg=peak_th,
    )
    (mdir / "truth.json").write_text(json.dumps(asdict(truth), indent=1), encoding="utf-8")
    return truth


def write_cohort(root: str | Path, specs: list[Spec]) -> dict[str, Truth]:
    """Several participants plus nothing else; use ``biomech cohort draft`` for the manifest."""
    return {s.pid: write_participant(root, s) for s in specs}


DEFAULT_SPECS = (
    Spec(pid="s01", sex="f", age=24, translation_mm=0.0, lead_ms=60.0, seed=1),
    Spec(pid="s02", sex="m", age=31, translation_mm=4.0, lead_ms=90.0, seed=2,
         leadoff_channel="digastric_r"),
    Spec(pid="s03", sex="f", age=45, translation_mm=6.0, lead_ms=120.0, seed=3,
         theta_max_deg=28.0),
    Spec(pid="s04", sex="m", age=27, translation_mm=3.0, lead_ms=75.0, seed=4,
         mains_hz=60.0, emg_band_hz=(20.0, 250.0)),
)


def write_study(root: str | Path, specs: list[Spec] | tuple[Spec, ...] = DEFAULT_SPECS,
                stages: dict[str, Any] | None = None) -> tuple[Path, dict[str, Truth]]:
    """Synthetic cohort + manifest + study file under ``root``; returns (study.yaml, truths).

    The default stage overrides keep the run short (fewer Monte-Carlo draws, a
    reduced ML grid) without changing what is computed.
    """
    import yaml

    from biomech.cohort import draft_manifest

    root = Path(root)
    data = root / "data"
    truths = write_cohort(data, list(specs))
    (root / "cohort.yaml").write_text(yaml.safe_dump(draft_manifest(data, "synthetic")),
                                      encoding="utf-8")
    fast = {
        "uncertainty": {"n_sims": 20, "n_boot": 20},
        "variability": {},
    }
    doc = {"study": "synthetic", "seed": 7,
           "paths": {"data_dir": "data", "out_dir": "out", "cohort": "cohort.yaml"},
           "stages": {k: v for k, v in {**fast, **(stages or {})}.items() if v}}
    path = root / "study.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path, truths


# -- in-memory timing simulator (X6 synthetic gates) ----------------------------------------
TIMING_PHASES = ("rest", "open_close", "chewing")
LEVER_M = 0.1                         # hinge-to-jaw-IMU lever for the |a_rel| model (m)
HEAD_HZ = 0.25                        # head-motion frequency


@dataclass
class _PhaseArrays:
    """The fields of :class:`biomech.emg.prepare.PhaseEmg` the timing analysis reads."""

    t_us: np.ndarray
    fs: float
    cleaned: np.ndarray
    mask: np.ndarray
    env: np.ndarray


class SynthPrep:
    """Duck-typed :class:`biomech.emg.prepare.EmgPrep` for :mod:`biomech.timing`: ``names``,
    ``phase(name)`` and ``causal(name)`` (the causal chain is the cleaned EMG itself, NaN at
    masked samples)."""

    def __init__(self, names: list[str], phases: dict[str, _PhaseArrays]) -> None:
        self.names = list(names)
        self._phases = phases

    def phase(self, name: str) -> _PhaseArrays:
        return self._phases[name]

    def causal(self, name: str) -> np.ndarray:
        ph = self._phases[name]
        return np.where(ph.mask, np.nan, ph.cleaned)


@dataclass
class TimingArrays:
    """What the timing stage reads, without HDF5: ``prep`` (:class:`SynthPrep`),
    ``imu_arrays`` (as the ``imu`` stage writes them; empty without IMU), ``kin_cycles`` (as
    ``kinematics.cycles``), ``kin_arrays`` (optical angle per task, 40 Hz) and ``truth``
    (per task the :func:`burst_schedule` rows; times in s)."""

    prep: SynthPrep
    imu_arrays: dict[str, np.ndarray]
    kin_cycles: pd.DataFrame
    kin_arrays: dict[str, np.ndarray]
    truth: dict[str, list[dict[str, Any]]]


def _imu_defaults() -> dict[str, float]:
    """The imu stage's grid rate and angular-velocity smoothing (package defaults)."""
    from biomech.config import package_default

    return {k: float(package_default("stages", "imu", k)) for k in ("grid_hz", "omega_smooth_s")}


def _rot_noise(rng: np.random.Generator, n: int, deg: float) -> Rotation:
    return Rotation.from_rotvec(rng.normal(0.0, np.radians(deg), (n, 3)) if deg > 0
                                else np.zeros((n, 3)))


def _imu_phase(spec: Spec, motion: Motion, a: float, b: float, grid_hz: float,
               smooth_s: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """IMU arrays of one phase on the ``grid_hz`` grid: the relative-quaternion opening angle
    (head rotation, orientation noise and stream latency applied; heading-free), its angular
    velocity, the jaw-minus-head |a_rel| (lever ``LEVER_M``, gravity leak of a jaw-IMU tilt
    error that turns with the jaw) and the optical angle (40 Hz, interpolated)."""
    from biomech import timebase

    step = int(round(1e6 / grid_hz))
    t_us = np.arange(int(round(a * 1e6)), int(round(b * 1e6)), step, dtype=np.int64)
    t = t_us * 1e-6
    th = motion.angles(t - spec.quat_latency_ms * 1e-3)[0]
    n = t.size
    ax = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    head = Rotation.from_rotvec(np.outer(np.radians(spec.head_motion_deg)
                                         * np.sin(2 * np.pi * HEAD_HZ * t), ax))
    jaw = head * Rotation.from_rotvec(np.outer(np.radians(th), [0.0, 1.0, 0.0]))
    rel = (_rot_noise(rng, n, spec.quat_noise_deg) * head).inv() \
        * (_rot_noise(rng, n, spec.quat_noise_deg) * jaw)
    theta_quat = np.degrees(rel.as_rotvec()[:, 1])
    omega = timebase.gradient(t_us, theta_quat,
                              smooth_n=timebase.odd_window(smooth_s, grid_hz))
    # |a_rel|: the jaw point on the lever in the head frame (true motion, latency applied),
    # plus the leak of a gravity removal tilted by tilt_error_deg in the jaw-IMU frame.
    thr = np.radians(th)
    p = LEVER_M * np.column_stack([np.cos(thr), np.zeros(n), -np.sin(thr)])
    acc = np.gradient(np.gradient(p, t, axis=0), t, axis=0) if n > 2 else np.zeros((n, 3))
    tilt = Rotation.from_rotvec([np.radians(spec.tilt_error_deg), 0.0, 0.0]).as_matrix()
    R_jaw = Rotation.from_rotvec(np.outer(thr, [0.0, 1.0, 0.0])).as_matrix()
    leak = G * np.einsum("nij,j->ni", R_jaw, (tilt - np.eye(3)) @ np.array([0.0, 0.0, 1.0]))
    acc_rel = np.linalg.norm(acc + leak + rng.normal(0.0, 0.03 * np.sqrt(2), (n, 3)), axis=1)
    to = np.arange(a, b, 1.0 / 40.0)
    th_o = motion.angles(to)[0] + rng.normal(0.0, 0.05, to.size)
    return {"t_us": t_us, "theta_quat": theta_quat, "omega_dps": omega,
            "acc_rel_ms2": acc_rel, "theta_opt": np.interp(t, to, th_o),
            "opt_t_us": np.round(to * 1e6).astype(np.int64), "opt_theta_deg": th_o}


def timing_arrays(spec: Spec, *, imu: bool = True,
                  tasks: tuple[str, ...] = ("open_close", "chewing")) -> TimingArrays:
    """Fast in-memory simulator of what the timing analysis needs (X6 synthetic gates).

    The motion schedule, cycles and EMG envelopes are those of :func:`write_participant` for
    the same ``spec`` (same :class:`Motion`, :func:`burst_schedule`, :func:`_envelopes`); the
    EMG is band-limited noise x envelope at ``spec.emg_fs``, already "cleaned" (no mains, DC
    or lead-off), unmasked, plus the closing-acceleration artefact ``emg_artifact_uv`` on the
    closers; the causal chain is the cleaned EMG. ``imu`` False: no IMU arrays (a participant without jaw IMU).
    Phases: ``rest`` and ``tasks``.
    """
    _check_spec(spec)
    from biomech.emg.artifacts import masked_envelope

    rng = np.random.default_rng(spec.seed)
    sched = _schedule(task_s=spec.task_s)
    motion = Motion(spec, sched, rng)
    span = {name: (a, b) for name, _, a, b in sched}
    names = list(TIMING_PHASES[:1]) + [t for t in tasks if t != "rest"]
    fs = float(spec.emg_fs)
    grids = {n: np.arange(span[n][0], span[n][1], 1.0 / fs) for n in names}
    t_all = np.concatenate([grids[n] for n in names])
    env_all = _envelopes(spec, t_all, motion, None, np.random.default_rng(spec.seed + 31))
    lo, hi = spec.emg_band_hz
    sos = butter(4, [lo, min(hi, 0.45 * fs)], btype="band", fs=fs, output="sos")
    noise_rng = np.random.default_rng(spec.seed + 37)
    ic = [MUSCLES.index(m) for m in CLOSERS]
    phases, k0 = {}, 0
    for n in names:
        t = grids[n]
        env = env_all[k0:k0 + t.size]
        k0 += t.size
        z = sosfiltfilt(sos, noise_rng.normal(size=(t.size, 6)), axis=0)
        x = z / z.std(axis=0) * env
        if spec.emg_artifact_uv and n != "rest":
            al = _alpha(motion, t)
            scale = float(np.percentile(np.abs(al), 99)) or 1.0
            x[:, ic] += (spec.emg_artifact_uv * np.clip(-al / scale, 0, None))[:, None]
        mask = np.zeros(x.shape, bool)
        phases[n] = _PhaseArrays(np.round(t * 1e6).astype(np.int64), fs, x, mask,
                                 masked_envelope(x, mask, fs, 0.15))
    imu_arrays: dict[str, np.ndarray] = {}
    kin_arrays: dict[str, np.ndarray] = {}
    defaults = _imu_defaults()
    imu_rng = np.random.default_rng(spec.seed + 41)
    for n in names:
        sig = _imu_phase(spec, motion, *span[n], defaults["grid_hz"],
                         defaults["omega_smooth_s"], imu_rng)
        kin_arrays[f"{n}__t_us"] = sig.pop("opt_t_us")
        kin_arrays[f"{n}__theta_deg"] = sig.pop("opt_theta_deg")
        if imu:
            imu_arrays.update({f"{n}__{k}": v for k, v in sig.items()})
    rows = [{"pid": spec.pid, "task": n, "cycle": k, "side": "", "t_start_us": c[0] * 1e6,
             "t_peak_us": c[1] * 1e6, "t_end_us": c[2] * 1e6, "ok": True}
            for n in names if n != "rest" for k, c in enumerate(motion.cycles.get(n, []))]
    truth = {n: burst_schedule(spec, motion, n) for n in names if n != "rest"}
    return TimingArrays(SynthPrep(list(MUSCLES), phases), imu_arrays, pd.DataFrame(rows),
                        kin_arrays, truth)
