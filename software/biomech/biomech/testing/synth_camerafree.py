# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic jaw-in-head motion with a known truth for the camera-free study.

A standalone companion of :mod:`biomech.testing.synth` (whose ``DEFAULT_SPECS`` and file
writer stay untouched): no session file, only the two IMU streams (as
:class:`biomech.imu.preprocess.ImuStream`) and the truth, in memory. One physical model:

* **frames:** world W (z up; the accelerometer reads +g along z at rest), maxillary frame
  ``max`` (x anterior, y left, z up, origin = upper incisor, as in ``synth``), head plate
  ``hp``, mandibular plate ``mp`` and the two IMU chips ``cs`` (head) and ``ms`` (jaw);
* **head:** ``H(t) = R_W<-max``, a base posture plus slow nod (pitch), tilt (roll) and yaw
  (sums of two sines per axis) and a small sway of the head origin;
* **jaw** (maxillary frame, relative to the canonical closed pose): a rotation vector
  ``r(t)`` per task -- rest, open_close (pure opening about y), protrusion_retrusion and
  left_right (translation-dominated, with a small opening and a Bennett-like rotation about
  z), chewing (opening plus a lateral swing, no pauses), free_motion (slow sines) -- and the
  condyle-midpoint translation ``tau = B f(r, theta_dot) + tau_free``: ``f`` is
  :func:`biomech.imu.camerafree.translation_features` (rotation vector, its squares and
  the hysteresis term ``tanh(theta_dot / w) theta``: the condyle path differs between
  opening and closing), ``tau_free`` the part no rotation explains (protrusion, laterotrusion,
  optionally free_motion). With ``free_translation_mm = 0`` the E4 model family contains
  the truth in every task except protrusion_retrusion and left_right. Cycles are ``sin⁴``
  excursions and every window or blend a septic smoothstep, so the motion is C³: the
  coupling depends on θ̇, and anything less smooth puts velocity jumps (acceleration
  impulses) into the condyle path that no accelerometer samples correctly;
* **closed (intercuspal) poses** between movements, each with its own small offset
  (``jitter_deg`` / ``jitter_mm``, blended smoothly through the movement in between; the
  first closure is the canonical closed pose itself);
* **IMUs:** game rotation vectors ``q = Rz(h(t)) R_W<-chip(t - lag)`` with an independent
  heading ``h(t) = h0 + drift t + random walk`` per IMU, white body-frame noise, an optional
  tilt drift of the jaw IMU's fusion (about a horizontal axis of its world; the
  accelerometer stays true) and an optional plate slip (the jaw plate turns on the
  mandible); accelerometers ``f = (1 + s) R_W<-chipᵀ (p̈ + g e_z) + b + noise`` at the
  IMU sites (lever arms in the plate frames), with ``p̈`` by central differences (1 ms) of
  the analytic site path.

Mountings follow :mod:`biomech.imu.calibration`: ``A = R_max<-cs = R_max<-hp R_mount,cra``,
``X = R_ms<-mp = R_mount,mtaᵀ`` (``R_mount`` = body<-chip, the hand-eye convention of
``validation.fusion.load_mounting``), and the true heading offset is
``psi(t) = h_cra(t) - h_mta(t)``. Every error source is off by default; :meth:`CFSpec.realistic`
switches on the plain sensor non-idealities (no drift, no lag).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jtsmm.imu.tilt import G

from biomech.imu.camerafree import HYST_DPS, translation_features
from biomech.imu.preprocess import ImuStream
from biomech.testing.synth import CONDYLES, LOWER

TASKS = ("rest", "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
# Per cyclic task: (movement s, closed pause s, cycles). Chewing has no pause (no anchor).
CYCLES = {"open_close": (1.0, 0.8, 6), "protrusion_retrusion": (1.2, 0.8, 5),
          "left_right": (1.2, 0.8, 6), "chewing": (0.8, 0.0, 12)}
FREE_S = 15.0
LEAD_S = 0.6                   # closed at the start and the end of every task
GRID_HZ = 200.0                # truth grid (the analysis grid of imu/analysis.py)
_DD_S = 1e-3                   # central-difference step of the site accelerations
_DTH_S = 1e-4                  # central-difference step of theta_dot inside the coupling


@dataclass
class CFSpec:
    """What to synthesize. Angles in degrees, lengths in mm, rates per minute."""

    seed: int = 0
    tasks: tuple[str, ...] = TASKS
    n_rounds: int = 1                    # the task sequence repeated (longer takes for drift)
    rest_s: float = 10.0
    gap_s: float = 2.0                   # closed between tasks
    # jaw
    theta_max_deg: float = 25.0
    chew_theta_deg: float = 12.0
    chew_swing_deg: float = 2.0          # lateral rotation (about z) per chewing cycle
    protrusion_mm: float = 7.0
    lateral_mm: float = 8.0
    free_translation_mm: float = 1.5     # translation in free_motion no rotation explains
    # coupling tau = B f (mm per deg, per deg²); u = anterior-inferior condyle path
    tau_lin: float = 0.20
    tau_quad: float = 0.004
    tau_hyst: float = 0.02
    bennett: float = 0.30                # lateral condyle shift per deg of rotation about z
    hyst_dps: float = HYST_DPS
    jitter_deg: float = 0.0              # closed-pose offset per closure (SD per axis)
    jitter_mm: float = 0.0
    # head (amplitudes of the slow motion about the maxillary axes)
    head_pitch_deg: float = 0.0
    head_roll_deg: float = 0.0
    head_yaw_deg: float = 0.0
    head_freq_hz: float = 0.12
    head_sway_mm: float = 0.0
    # IMUs (pairs: cra, mta)
    q_hz: float = 390.0
    a_hz: float = 255.0
    q_noise_deg: float = 0.0
    q_lag_s: tuple[float, float] = (0.0, 0.0)
    a_lag_s: tuple[float, float] = (0.0, 0.0)
    heading_drift_dpm: tuple[float, float] = (0.0, 0.0)
    heading_rw_deg: tuple[float, float] = (0.0, 0.0)     # random-walk SD after 1 min
    tilt_drift_dpm: float = 0.0          # jaw IMU fusion tilt drift (accelerometer true)
    slip_deg: float = 0.0                # jaw plate slip on the mandible (smooth step)
    slip_t_s: float | None = None        # None: the middle of the movement nearest mid-take
    slip_dur_s: float = 0.3
    a_noise_mps2: float = 0.0
    a_bias_mps2: float = 0.0             # SD per axis
    a_scale: float = 0.0                 # SD of the per-axis scale error
    lever_cra_mm: float = 30.0           # |IMU site| in the head plate frame
    lever_mta_mm: float = 24.0           # |IMU site| in the jaw plate frame (H4.2: 24 mm)

    @classmethod
    def realistic(cls, **kw: Any) -> "CFSpec":
        """Plain sensor non-idealities and natural head motion; no drift, no lag."""
        base = dict(q_noise_deg=0.02, jitter_deg=0.05, jitter_mm=0.03, head_pitch_deg=6.0,
                    head_roll_deg=3.0, head_yaw_deg=4.0, head_sway_mm=3.0, a_noise_mps2=0.02,
                    a_bias_mps2=0.03, a_scale=0.005)
        return cls(**{**base, **kw})

    def with_(self, **kw: Any) -> "CFSpec":
        return replace(self, **kw)


@dataclass
class CFTruth:
    """Everything an estimator should recover, on the 200 Hz grid ``t_us`` (int64 µs)."""

    t_us: np.ndarray
    R_rel: np.ndarray                    # (N, 3, 3) mandible rotation since the closed pose
    rotvec_deg: np.ndarray               # (N, 3) its rotation vector (maxillary frame)
    tau_mm: np.ndarray                   # (N, 3) condyle-midpoint translation
    incisor_mm: np.ndarray               # (N, 3) lower incisor, maxillary frame
    condyle_mm: np.ndarray               # (N, 3) condyle midpoint, maxillary frame
    psi_rad: np.ndarray                  # (N,) true heading offset h_cra - h_mta (wrapped)
    H: np.ndarray                        # (N, 3, 3) head orientation R_W<-max
    closures: list[tuple[float, float]]  # closed intervals (s)
    tasks: list[tuple[str, float, float]]  # (task, t0, t1) s, in order (rounds repeat names)
    A: np.ndarray                        # R_max<-cs
    X: np.ndarray                        # R_ms<-mp
    R_mount_cra: np.ndarray              # R_hp<-cs (body<-chip)
    R_mount_mta: np.ndarray              # R_mp<-ms
    T_max_hp: np.ndarray                 # (4, 4) landmark frame of the head plate
    T_max_mp_closed: np.ndarray          # (4, 4) jaw plate at the canonical closed pose
    lever_cra_mm: np.ndarray             # IMU site, head plate frame
    lever_mta_mm: np.ndarray             # IMU site, jaw plate frame
    incisor0_mm: np.ndarray              # closed landmarks (maxillary frame)
    condyle0_mm: np.ndarray
    site_cra_mm: np.ndarray              # head IMU site, maxillary frame
    site_mta0_mm: np.ndarray             # jaw IMU site at the closed pose, maxillary frame
    B: np.ndarray                        # (3, 7) coupling tau = B f
    hyst_dps: float
    heading_cra_rad: np.ndarray          # (N,) per-IMU heading (truth of the drift)
    heading_mta_rad: np.ndarray
    tilt_drift_axis: np.ndarray          # horizontal axis of the jaw IMU's tilt drift
    slip_axis: np.ndarray                # jaw plate frame
    slip_t_s: float
    spec: dict[str, Any] = field(default_factory=dict)

    @property
    def t_s(self) -> np.ndarray:
        return self.t_us * 1e-6

    def mask(self, *names: str) -> np.ndarray:
        """Grid samples inside any task named ``names`` (all rounds)."""
        t = self.t_s
        out = np.zeros(t.size, bool)
        for name, a, b in self.tasks:
            if name in names:
                out |= (t >= a) & (t <= b)
        return out

    def closed_mask(self) -> np.ndarray:
        t = self.t_s
        out = np.zeros(t.size, bool)
        for a, b in self.closures:
            out |= (t >= a) & (t <= b)
        return out


@dataclass
class CFSynth:
    spec: CFSpec
    cra: ImuStream
    mta: ImuStream
    truth: CFTruth


# -- small helpers ---------------------------------------------------------------------------
def _rot(rv_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.radians(rv_deg)).as_matrix()


def _rz(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _rand_rot(rng: np.random.Generator, max_deg: float = 180.0) -> np.ndarray:
    v = rng.normal(size=3)
    v /= np.linalg.norm(v)
    return Rotation.from_rotvec(v * np.radians(rng.uniform(0.3, 1.0) * max_deg)).as_matrix()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _smooth(s: np.ndarray) -> np.ndarray:
    """0 -> 1 step on s in [0, 1] with zero 1st-3rd derivatives at both ends (septic)."""
    s = np.clip(s, 0.0, 1.0)
    return s ** 4 * (35.0 - 84.0 * s + 70.0 * s ** 2 - 20.0 * s ** 3)


def _profile(t: np.ndarray, s0: float, dur: float) -> np.ndarray:
    """0 -> 1 -> 0 excursion ``sin⁴(π s)`` of length ``dur`` from ``s0``.

    C³ at both ends: the coupling depends on θ̇, so the condyle acceleration contains θ⃛; a
    raised cosine (C¹) would make the translation velocity jump at every cycle edge, an
    impulse no accelerometer samples correctly (and no jaw makes)."""
    s = (t - s0) / dur
    out = np.zeros_like(t)
    m = (s >= 0) & (s <= 1)
    out[m] = np.sin(np.pi * s[m]) ** 4
    return out


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(R).as_quat()
    return np.column_stack([q[:, 3], q[:, :3]])


# -- schedule --------------------------------------------------------------------------------
def _schedule(spec: CFSpec) -> tuple[list[tuple[str, float, float]],
                                     list[tuple[str, int, float, float]],
                                     list[tuple[float, float]], float]:
    """Tasks (name, t0, t1), movements (task, cycle, s0, s1), closed intervals, duration."""
    tasks, moves = [], []
    t = 1.0
    for _ in range(spec.n_rounds):
        for name in spec.tasks:
            if name == "rest":
                t1 = t + spec.rest_s
            elif name == "free_motion":
                t1 = t + FREE_S
                moves.append((name, 0, t, t1))
            else:
                mv, pause, n = CYCLES[name]
                s = t + LEAD_S
                for i in range(n):
                    moves.append((name, i, s, s + mv))
                    s += mv + pause
                t1 = s - pause + LEAD_S
            tasks.append((name, t, t1))
            t = t1 + spec.gap_s
    duration = t - spec.gap_s + 1.0
    # closed = everything outside the movements (adjacent chewing cycles merge)
    closed, cur = [], 0.0
    for _, _, a, b in sorted(moves, key=lambda m: m[2]):
        if a - cur > 1e-9:
            closed.append((cur, a))
        cur = max(cur, b)
    if duration - cur > 1e-9:
        closed.append((cur, duration))
    return tasks, moves, closed, duration


# -- the motion model --------------------------------------------------------------------------
class _Model:
    def __init__(self, spec: CFSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.tasks, self.moves, self.closed, self.duration = _schedule(spec)
        self.u = _unit([1.0, 0.0, -0.5])              # anterior-inferior condyle path
        self.c0 = 0.5 * (np.array(CONDYLES["tmj_left"]) + np.array(CONDYLES["tmj_right"]))
        self.p0 = np.array(LOWER["incisor"], float)
        sp = spec
        B = np.zeros((3, 7))
        B[2, 0] = 0.05                                 # roll -> small vertical shift
        B[:, 1] = sp.tau_lin * self.u                  # opening
        B[1, 2] = sp.bennett                           # rotation about z -> lateral shift
        B[:, 4] = sp.tau_quad * self.u
        B[0, 5] = 0.02                                 # lateral swing -> protrusive shift
        B[:, 6] = sp.tau_hyst * self.u                 # opening/closing hysteresis
        self.B = B
        self.jit_amp = 1.0 + 0.08 * rng.uniform(-1, 1, len(self.moves))
        self.free = {"f": rng.uniform(0.15, 0.5, 8), "ph": rng.uniform(0, 2 * np.pi, 8)}
        k = len(self.closed)
        self.j_rot = rng.normal(0, sp.jitter_deg, (k, 3))
        self.j_tr = rng.normal(0, sp.jitter_mm, (k, 3))
        self.j_rot[0] = 0.0                            # the first closure is the canonical pose
        self.j_tr[0] = 0.0
        # head
        self.H0 = _rz(rng.uniform(-np.pi, np.pi)) @ Rotation.from_euler(
            "y", rng.uniform(-10, 10), degrees=True).as_matrix()
        self.h0 = np.array([0.0, 0.0, 1500.0]) + rng.uniform(-50, 50, 3)
        self.head_ph = rng.uniform(0, 2 * np.pi, (3, 2))
        self.sway_ph = rng.uniform(0, 2 * np.pi, 3)

    # jaw ---------------------------------------------------------------------------------
    def _task_rot(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Task rotation vector (deg) and the translation no rotation explains (mm)."""
        sp = self.spec
        r = np.zeros(t.shape + (3,))
        tr = np.zeros(t.shape + (3,))
        for (name, i, s0, s1), k in zip(self.moves, self.jit_amp):
            if name == "free_motion":
                continue
            prof = _profile(t, s0, s1 - s0)
            if not prof.any():
                continue
            if name == "open_close":
                r[:, 1] += sp.theta_max_deg * k * prof
            elif name == "protrusion_retrusion":
                tr += np.outer(sp.protrusion_mm * k * prof, _unit([1.0, 0.0, -0.35]))
                r[:, 1] += 2.0 * prof
            elif name == "left_right":
                sd = 1.0 if i % 2 == 0 else -1.0            # left, right, left, ...
                tr[:, 1] += sd * sp.lateral_mm * k * prof
                tr[:, 2] -= 1.0 * prof
                r[:, 2] += sd * 4.0 * prof
                r[:, 1] += 2.0 * prof
            elif name == "chewing":
                r[:, 1] += sp.chew_theta_deg * k * prof
                r[:, 2] += sp.chew_swing_deg * np.sin(2 * np.pi * (t - s0) / (s1 - s0)) * prof
        f, ph = self.free["f"], self.free["ph"]
        for name, _, a, b in self.moves:
            if name != "free_motion":
                continue
            m = (t >= a) & (t <= b)
            if not m.any():
                continue
            w = _smooth(np.minimum((t[m] - a) / 1.0, (b - t[m]) / 1.0))
            tt = t[m]
            sn = lambda j: np.sin(2 * np.pi * f[j] * tt + ph[j])          # noqa: E731
            r[m, 1] += w * (2.0 + 7.0 * (1.0 + sn(0)))
            r[m, 0] += w * 3.0 * sn(1)
            r[m, 2] += w * 5.0 * sn(2)
            amp = sp.free_translation_mm
            tr[m] += (w * amp)[:, None] * np.column_stack([sn(3), sn(4), 0.5 * sn(5)])
        return r, tr

    def _jitter(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Closed-pose offsets: constant in closure k, a smooth step to k+1 in between."""
        c = np.array(self.closed)
        k = np.clip(np.searchsorted(c[:, 0], t, side="right") - 1, 0, len(c) - 1)
        nxt = np.minimum(k + 1, len(c) - 1)
        e, s = c[k, 1], c[nxt, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where((t > e) & (nxt > k), _smooth((t - e) / np.maximum(s - e, 1e-9)), 0.0)
        jr = self.j_rot[k] + w[:, None] * (self.j_rot[nxt] - self.j_rot[k])
        jt = self.j_tr[k] + w[:, None] * (self.j_tr[nxt] - self.j_tr[k])
        return jr, jt

    def _rotvec(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, tr = self._task_rot(t)
        jr, jt = self._jitter(t)
        return r + jr, tr + jt

    def jaw(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rotation vector (deg), R_rel (N, 3, 3) and tau (mm) at ``t``."""
        r, tr = self._rotvec(t)
        th_d = (self._rotvec(t + _DTH_S)[0][:, 1] - self._rotvec(t - _DTH_S)[0][:, 1]) / (
            2 * _DTH_S)
        tau = translation_features(r, th_d, self.spec.hyst_dps) @ self.B.T + tr
        return r, _rot(r), tau

    # head --------------------------------------------------------------------------------
    def head(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sp = self.spec
        amp = np.array([sp.head_roll_deg, sp.head_pitch_deg, sp.head_yaw_deg])
        fr = sp.head_freq_hz * np.array([1.0, 0.8, 0.6])
        ph = self.head_ph
        g = (0.75 * np.sin(2 * np.pi * fr[None] * t[:, None] + ph[:, 0])
             + 0.25 * np.sin(2 * np.pi * 2.3 * fr[None] * t[:, None] + ph[:, 1]))
        H = self.H0 @ _rot(g * amp)
        h = self.h0 + sp.head_sway_mm * np.sin(
            2 * np.pi * 0.2 * t[:, None] + self.sway_ph) * np.array([1.0, 1.0, 0.3])
        return H, h


def generate(spec: CFSpec | None = None) -> CFSynth:
    """Synthesize the two IMU streams and the truth of one take."""
    spec = spec or CFSpec()
    rng = np.random.default_rng(spec.seed)
    mdl = _Model(spec, rng)
    dur = mdl.duration

    # Fixed placements (maxillary frame), mountings and lever arms.
    T_max_hp = np.eye(4)
    T_max_hp[:3, :3] = _rand_rot(rng, 20.0)
    T_max_hp[:3, 3] = [35.0, 0.0, 70.0]                  # forehead array
    T_mp0 = np.eye(4)
    T_mp0[:3, :3] = _rand_rot(rng, 20.0)
    T_mp0[:3, 3] = [15.0, 0.0, -45.0]                    # chin array, closed
    R_mount_cra, R_mount_mta = _rand_rot(rng), _rand_rot(rng)
    A = T_max_hp[:3, :3] @ R_mount_cra
    X = R_mount_mta.T
    lev_c = spec.lever_cra_mm * _unit(rng.normal(size=3))
    lev_m = spec.lever_mta_mm * _unit(rng.normal(size=3))
    site_c = T_max_hp[:3, :3] @ lev_c + T_max_hp[:3, 3]
    site_m0 = T_mp0[:3, :3] @ lev_m + T_mp0[:3, 3]

    # Heading of each IMU's world (drift + random walk on a 10 Hz grid).
    h0 = rng.uniform(-np.pi, np.pi, 2)
    t_rw = np.arange(0.0, dur + 1.0, 0.1)
    rw = [np.cumsum(rng.normal(0, np.radians(s) * np.sqrt(0.1 / 60.0), t_rw.size))
          for s in spec.heading_rw_deg]

    def heading(i: int, t: np.ndarray) -> np.ndarray:
        return (h0[i] + np.radians(spec.heading_drift_dpm[i]) * t / 60.0
                + np.interp(t, t_rw, rw[i]))

    a_tilt = rng.uniform(0, 2 * np.pi)
    n_tilt = np.array([np.cos(a_tilt), np.sin(a_tilt), 0.0])
    n_slip = _unit(rng.normal(size=3))
    if spec.slip_t_s is not None:
        t_slip = float(spec.slip_t_s)
    else:
        mids = np.array([0.5 * (a + b) for _, _, a, b in mdl.moves])
        t_slip = float(mids[np.argmin(np.abs(mids - 0.5 * dur))])

    def slip(t: np.ndarray) -> np.ndarray:
        s = _smooth((t - t_slip + 0.5 * spec.slip_dur_s) / spec.slip_dur_s)
        return Rotation.from_rotvec(np.outer(np.radians(spec.slip_deg) * s, n_slip)).as_matrix()

    def sensors(t: np.ndarray) -> dict[str, np.ndarray]:
        """Chip orientations (W<-chip) and IMU sites (W, mm) at ``t``."""
        _, R_rel, tau = mdl.jaw(t)
        t_rel = mdl.c0 + tau - R_rel @ mdl.c0
        H, h = mdl.head(t)
        S = slip(t)
        R_mp = R_rel @ T_mp0[:3, :3] @ S
        q_m = np.einsum("nij,nj->ni", R_rel, np.einsum("nij,j->ni", T_mp0[:3, :3] @ S, lev_m)
                        + T_mp0[:3, 3]) + t_rel
        return {"R_cs": H @ A, "R_ms": H @ R_mp @ X.T,
                "p_cs": H @ site_c + h, "p_ms": np.einsum("nij,nj->ni", H, q_m) + h}

    def accel(t: np.ndarray, key: str) -> np.ndarray:
        return (sensors(t + _DD_S)[key] - 2 * sensors(t)[key] + sensors(t - _DD_S)[key]) / (
            _DD_S ** 2)

    streams = {}
    for i, slot in enumerate(("cra", "mta")):
        rk, pk = ("R_cs", "p_cs") if slot == "cra" else ("R_ms", "p_ms")
        tq = np.arange(0.0, dur, 1.0 / spec.q_hz)
        Rq = sensors(tq - spec.q_lag_s[i])[rk]
        Rq = _rz(heading(i, tq)) @ Rq
        if slot == "mta" and spec.tilt_drift_dpm:
            beta = np.radians(spec.tilt_drift_dpm) * tq / 60.0
            Rq = Rotation.from_rotvec(np.outer(beta, n_tilt)).as_matrix() @ Rq
        if spec.q_noise_deg > 0:
            Rq = Rq @ Rotation.from_rotvec(
                rng.normal(0, np.radians(spec.q_noise_deg), (tq.size, 3))).as_matrix()
        ta = np.arange(0.0, dur, 1.0 / spec.a_hz)
        tl = ta - spec.a_lag_s[i]
        Ra = sensors(tl)[rk]
        f_w = accel(tl, pk) * 1e-3 + np.array([0.0, 0.0, G])          # m/s², specific force
        f = np.einsum("nji,nj->ni", Ra, f_w)
        f = f * (1.0 + rng.normal(0, spec.a_scale, 3)) + rng.normal(0, spec.a_bias_mps2, 3)
        f = f + rng.normal(0, spec.a_noise_mps2, f.shape)
        t_q, t_a = np.round(tq * 1e6).astype(np.int64), np.round(ta * 1e6).astype(np.int64)
        q, a = _quat_wxyz(Rq), f
        for arr in (t_q, q, t_a, a):
            arr.flags.writeable = False
        streams[slot] = ImuStream(slot, t_q, q, t_a, a, 0, 0)

    # Truth on the 200 Hz grid.
    tg = np.arange(0.0, dur, 1.0 / GRID_HZ)
    r, R_rel, tau = mdl.jaw(tg)
    H, _ = mdl.head(tg)
    hc, hm = heading(0, tg), heading(1, tg)
    truth = CFTruth(
        t_us=np.round(tg * 1e6).astype(np.int64), R_rel=R_rel, rotvec_deg=r, tau_mm=tau,
        incisor_mm=np.einsum("nij,j->ni", R_rel, mdl.p0 - mdl.c0) + mdl.c0 + tau,
        condyle_mm=mdl.c0 + tau, psi_rad=(hc - hm + np.pi) % (2 * np.pi) - np.pi, H=H,
        closures=[(float(a), float(b)) for a, b in mdl.closed], tasks=mdl.tasks, A=A, X=X,
        R_mount_cra=R_mount_cra, R_mount_mta=R_mount_mta, T_max_hp=T_max_hp,
        T_max_mp_closed=T_mp0, lever_cra_mm=lev_c, lever_mta_mm=lev_m,
        incisor0_mm=mdl.p0.copy(), condyle0_mm=mdl.c0.copy(), site_cra_mm=site_c,
        site_mta0_mm=site_m0, B=mdl.B, hyst_dps=spec.hyst_dps, heading_cra_rad=hc,
        heading_mta_rad=hm, tilt_drift_axis=n_tilt, slip_axis=n_slip, slip_t_s=t_slip,
        spec=asdict(spec))
    return CFSynth(spec, streams["cra"], streams["mta"], truth)


def head_tilt_range_deg(truth: CFTruth, mask: np.ndarray | None = None) -> float:
    """Largest angle between two head verticals (maxillary frame) over ``mask`` (default: the
    closures): the head-tilt range that makes psi identifiable (H4.8)."""
    m = truth.closed_mask() if mask is None else np.asarray(mask, bool)
    v = np.einsum("nji,j->ni", truth.H[m], np.array([0.0, 0.0, 1.0]))
    v = v[:: max(1, len(v) // 600)]
    return float(np.degrees(np.arccos(np.clip((v @ v.T).min(), -1.0, 1.0))))


__all__ = ["CFSpec", "CFSynth", "CFTruth", "TASKS", "generate", "head_tilt_range_deg"]
