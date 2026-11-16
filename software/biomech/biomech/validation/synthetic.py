# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""A synthetic JTS-MM + Qualisys take with a known clock model and registration.

Used by the validation tests (``tests/test_v_*.py``); no subject data involved.

* Three bodies (Qualisys names): ``CT`` (hand-held tool: visibility bursts, then irregular
  rotation and translation in a start and an end sync window, a static hold after the
  start motion), ``HP`` (head: slow sway) and ``MP`` (mandible = head ∘ jaw motion: a hinge
  plus translation in the movement phases, still otherwise).
* JTS-MM: 40-Hz trigger stamps on the master clock (exposure start), poses averaged over
  the 12-ms exposure, dropped frames, white pose noise; the CT is detected
  ``threshold_s`` late on reappearance and early on disappearance.
* Qualisys: 200 Hz on its own clock ``t_Q = t_M + c + α (t_M − t_ref)``, short occlusions,
  smaller noise, detected ``threshold_q_s`` early / late (it needs only 3 markers).
* World and body transforms: ``T_Q,b = X · T_S,b · Y_b`` with known ``X``, ``Y_b``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.validation import registration as REG
from biomech.validation.sync import TimeModel


@dataclass(frozen=True)
class Phase:
    """Duck-typed stand-in for ``jtsmm`` ``PhaseInterval`` (name, start/end on the master µs)."""
    name: str
    t_start_us: int
    t_end_us: int


@dataclass
class SynthTake:
    t_trig_s: np.ndarray                 # JTS-MM trigger (exposure start) times, master s
    S: dict[str, np.ndarray]             # body -> (N, 4, 4) JTS-MM poses, NaN = not detected
    t_q_s: np.ndarray                    # Qualisys frame times (Qualisys clock)
    Q: dict[str, np.ndarray]             # body -> (M, 4, 4) Qualisys poses
    phases: list[Phase]
    model: TimeModel                     # the true clock model (t_ref = ``t_ref_s``)
    X: np.ndarray
    Y: dict[str, np.ndarray]
    noise: dict[str, float] = field(default_factory=dict)


def _bandlimited(rng: np.random.Generator, n_comp: int, f_lo: float, f_hi: float,
                 amp: float, dims: int = 3):
    f = rng.uniform(f_lo, f_hi, (n_comp, dims))
    ph = rng.uniform(0, 2 * np.pi, (n_comp, dims))
    a = rng.uniform(0.3, 1.0, (n_comp, dims))
    a = amp * a / np.sqrt((a ** 2).sum(axis=0, keepdims=True) / 2.0)

    def fn(t: np.ndarray) -> np.ndarray:
        return np.einsum("kd,nkd->nd", a, np.sin(2 * np.pi * f[None] * t[:, None, None]
                                                  + ph[None]))
    return fn


def _smooth_box(t: np.ndarray, a: float, b: float, ramp: float = 0.5) -> np.ndarray:
    """1 inside [a, b], 0 outside, with raised-cosine ramps of ``ramp`` s."""
    up = np.clip((t - a) / ramp, 0, 1)
    dn = np.clip((b - t) / ramp, 0, 1)
    return (0.5 - 0.5 * np.cos(np.pi * up)) * (0.5 - 0.5 * np.cos(np.pi * dn))


def make_take(*, seed: int = 0, t0: float = 1000.0, duration: float = 300.0,
              offset_s: float = -957.3215, alpha: float = 30e-6, fs_s: float = 40.0,
              fs_q: float = 200.0, exposure_s: float = 0.012, drop_frac: float = 0.03,
              noise_pos_s: float = 0.2, noise_rot_s_deg: float = 0.1,
              noise_pos_q: float = 0.05, noise_rot_q_deg: float = 0.03,
              threshold_s: float = 0.010, threshold_q_s: float = 0.005,
              q_occlusion_frac: float = 0.01) -> SynthTake:
    """One synthetic take (see module docstring). Noise values are RMS magnitudes (3-D)."""
    rng = np.random.default_rng(seed)
    T = duration
    # -- protocol (seconds from t0) -------------------------------------------------------
    ph = [("mand_point_1", 5, 60), ("rest", 62, 75), ("open_close", 77, 95),
          ("protrusion_retrusion", 97, 112), ("left_right", 114, 128), ("chewing", 130, 145),
          ("free_motion", 147, T - 1)]
    phases = [Phase(n, int((t0 + a) * 1e6), int((t0 + b) * 1e6)) for n, a, b in ph]
    # burst times jittered so that the 40-Hz quantisation of the edges is not systematic
    js = rng.uniform(0.0, 0.3, (11, 2))
    bursts_start = [(7.0 + 2.0 * k + js[k, 0], 7.6 + 2.0 * k + js[k, 1]) for k in range(6)]
    bursts_end = [(T - 45.0 + 2.0 * k + js[6 + k, 0], T - 44.4 + 2.0 * k + js[6 + k, 1])
                  for k in range(5)]
    ct_vis = bursts_start + [(20.0, 58.0)] + bursts_end + [(T - 34.0, T - 8.0)]
    ct_motion = [(20.0, 45.0), (T - 34.0, T - 8.0)]

    # -- true world motion (Qualisys world, Qualisys body frames) ---------------------------
    ct_rot = _bandlimited(rng, 8, 0.2, 2.0, np.radians(30.0))
    ct_pos = _bandlimited(rng, 8, 0.2, 2.0, 40.0)
    ct_rot2 = _bandlimited(rng, 4, 3.0, 6.0, np.radians(4.0))   # bursts: small wiggles
    hp_rot = _bandlimited(rng, 5, 0.05, 0.5, np.radians(2.0))
    hp_pos = _bandlimited(rng, 5, 0.05, 0.5, 3.0)
    jaw_open = _bandlimited(rng, 6, 0.5, 2.5, 1.0, dims=1)
    jaw_lat = _bandlimited(rng, 6, 0.3, 2.0, 1.0, dims=3)
    R_ct0 = Rotation.from_rotvec(rng.normal(size=3))
    R_hp0 = Rotation.from_rotvec(rng.normal(size=3) * 0.3)
    p_ct0 = np.array([150.0, -50.0, 900.0])
    p_hp0 = np.array([100.0, 30.0, 950.0])
    hinge_axis = np.array([0.0, 1.0, 0.1]) / np.linalg.norm([0.0, 1.0, 0.1])
    hinge_pt = np.array([-90.0, 0.0, -20.0])             # in the head frame
    jaw_off = np.array([60.0, 0.0, -60.0])               # MP origin in the head frame (closed)
    tasks = {n: (a, b) for n, a, b in ph}

    def ct_world(t: np.ndarray) -> np.ndarray:
        tr = t - t0
        env = sum(_smooth_box(tr, a, b, 1.0) for a, b in ct_motion)
        wig = sum(_smooth_box(tr, a, b, 0.2) for a, b in bursts_start + bursts_end)
        rv = ct_rot(tr) * env[:, None] + ct_rot2(tr) * wig[:, None]
        R = (R_ct0 * Rotation.from_rotvec(rv)).as_matrix()
        p = p_ct0 + ct_pos(tr) * env[:, None] + 2.0 * ct_rot2(tr) * wig[:, None] * 50
        return REG.make_pose(R, p)

    def hp_world(t: np.ndarray) -> np.ndarray:
        tr = t - t0
        return REG.make_pose((R_hp0 * Rotation.from_rotvec(hp_rot(tr))).as_matrix(),
                             p_hp0 + hp_pos(tr))

    def jaw_rel(t: np.ndarray) -> np.ndarray:
        tr = t - t0
        act = np.zeros_like(tr)
        for n in ("open_close", "protrusion_retrusion", "left_right", "chewing"):
            act += _smooth_box(tr, *tasks[n], 0.5)
        act += _smooth_box(tr, tasks["free_motion"][0], T - 47.0, 0.5)
        ang = np.radians(12.0) * (1.0 + jaw_open(tr)[:, 0]) * act
        R = Rotation.from_rotvec(ang[:, None] * hinge_axis[None])
        trans = jaw_lat(tr) * np.array([3.0, 3.0, 1.0]) * act[:, None]
        p = hinge_pt + R.apply(jaw_off - hinge_pt) + trans
        return REG.make_pose(R.as_matrix(), p)

    def world(b: str, t: np.ndarray) -> np.ndarray:
        if b == "CT":
            return ct_world(t)
        if b == "HP":
            return hp_world(t)
        return np.einsum("nij,njk->nik", hp_world(t), jaw_rel(t))

    # -- transforms -------------------------------------------------------------------------
    def rand_pose(rot_scale: float, trans: np.ndarray) -> np.ndarray:
        return REG.make_pose(Rotation.from_rotvec(rng.normal(size=3) * rot_scale).as_matrix(),
                             trans)
    X = rand_pose(1.0, np.array([350.0, -1200.0, 400.0]))
    Y = {"CT": rand_pose(0.5, np.array([-12.0, 95.0, 4.0])),     # incl. the tip offset
         "MP": rand_pose(0.5, np.array([5.0, -8.0, 12.0])),
         "HP": rand_pose(0.5, np.array([-6.0, 3.0, 20.0]))}
    Xi = np.linalg.inv(X)
    Yi = {b: np.linalg.inv(v) for b, v in Y.items()}

    def noisy(T_: np.ndarray, pos: float, rot_deg: float) -> np.ndarray:
        n = len(T_)
        rv = rng.normal(size=(n, 3)) * np.radians(rot_deg) / np.sqrt(3)
        out = T_.copy()
        out[:, :3, :3] = np.einsum("nij,njk->nik", T_[:, :3, :3],
                                   Rotation.from_rotvec(rv).as_matrix())
        out[:, :3, 3] += rng.normal(size=(n, 3)) * pos / np.sqrt(3)
        return out

    # -- JTS-MM -----------------------------------------------------------------------------
    t_trig = t0 + np.arange(0.0, T, 1.0 / fs_s)
    keep = rng.random(len(t_trig)) >= drop_frac
    t_trig = t_trig[keep]
    sub = np.linspace(0.0, exposure_s, 13)
    tt = (t_trig[:, None] + sub[None, :]).ravel()
    S: dict[str, np.ndarray] = {}
    for b in ("CT", "MP", "HP"):
        W = world(b, tt).reshape(len(t_trig), len(sub), 4, 4)
        Wm = W.mean(axis=1)                                  # exposure blur
        U, _, Vt = np.linalg.svd(Wm[:, :3, :3])
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, :, 2] *= d[:, None]
        Wm[:, :3, :3] = U @ Vt
        Ts = np.einsum("ij,njk,kl->nil", Xi, Wm, Yi[b])
        S[b] = noisy(Ts, noise_pos_s, noise_rot_s_deg)
    tc = t_trig + 0.5 * exposure_s - t0
    vis_s = np.zeros(len(t_trig), bool)
    for a, b in ct_vis:
        vis_s |= (tc >= a + threshold_s) & (tc <= b - threshold_s)
    S["CT"][~vis_s] = np.nan

    # -- Qualisys ---------------------------------------------------------------------------
    t_ref = t0 + 0.5 * T
    model = TimeModel(offset_s, alpha, t_ref)
    tq_lo, tq_hi = model.to_q(t0 - 5.0), model.to_q(t0 + T + 5.0)
    t_q = np.arange(np.ceil(tq_lo * fs_q), np.floor(tq_hi * fs_q)) / fs_q
    t_qm = model.to_s(t_q) - t0                               # master time of each frame
    Q: dict[str, np.ndarray] = {}
    for b in ("CT", "MP", "HP"):
        Q[b] = noisy(world(b, t_qm + t0), noise_pos_q, noise_rot_q_deg)
        if b != "CT" and q_occlusion_frac > 0:
            n_occ = int(q_occlusion_frac * len(t_q) / 10)
            for s in rng.integers(0, len(t_q) - 20, n_occ):
                Q[b][s:s + rng.integers(2, 15)] = np.nan
    vis_q = np.zeros(len(t_q), bool)
    for a, b in ct_vis:
        vis_q |= (t_qm >= a - threshold_q_s) & (t_qm <= b + threshold_q_s)
    Q["CT"][~vis_q] = np.nan
    return SynthTake(t_trig_s=t_trig, S=S, t_q_s=t_q, Q=Q, phases=phases, model=model, X=X,
                     Y=Y, noise={"pos_s": noise_pos_s, "rot_s_deg": noise_rot_s_deg,
                                 "pos_q": noise_pos_q, "rot_q_deg": noise_rot_q_deg})


# -- file writers (QTM .mat, Trigno CSV) with the real layouts, synthetic values ---------
def write_qtm_mat(path, take: str, fs_hz: float, bodies: dict[str, np.ndarray],
                  start_frame: int = 1, markers: dict[str, np.ndarray] | None = None,
                  timestamp: str = "2000-01-01, 00:00:00.000\t0.0") -> None:
    """A minimal QTM 2024-style ``.mat`` export: one struct variable named ``take`` with
    File, Timestamp, StartFrame, Frames, FrameRate, Trajectories.Labeled and RigidBodies
    (Positions (B, 3, N) mm, Rotations (B, 9, N) column-major, Residual (B, 1, N); NaN where
    untracked)."""
    from scipy.io import savemat

    names = list(bodies)
    n = len(next(iter(bodies.values())))
    pos = np.full((len(names), 3, n), np.nan)
    rot = np.full((len(names), 9, n), np.nan)
    res = np.full((len(names), 1, n), np.nan)
    for k, b in enumerate(names):
        T = bodies[b]
        ok = np.isfinite(T.reshape(n, -1)).all(axis=1)
        pos[k, :, ok] = T[ok, :3, 3]
        rot[k, :, ok] = T[ok, :3, :3].transpose(0, 2, 1).reshape(-1, 9)   # column-major
        res[k, 0, ok] = 0.5
    markers = markers or {}
    labels = np.empty((1, len(markers)), dtype=object)
    data = np.full((len(markers), 4, n), np.nan)
    for k, (lab, xyz) in enumerate(markers.items()):
        labels[0, k] = lab
        data[k, :3] = xyz.T
        data[k, 3] = 0.3
    name_cell = np.empty((1, len(names)), dtype=object)
    for k, b in enumerate(names):
        name_cell[0, k] = b
    st = {
        "File": f"C:\\synthetic\\{take}.qtm", "Timestamp": timestamp,
        "StartFrame": float(start_frame), "Frames": float(n), "FrameRate": float(fs_hz),
        "Trajectories": {"Labeled": {"Count": float(len(markers)), "Labels": labels,
                                     "Data": data}},
        "RigidBodies": {"Bodies": float(len(names)), "Name": name_cell, "Positions": pos,
                        "Rotations": rot, "RPYs": np.zeros_like(pos), "Residual": res},
    }
    savemat(str(path), {take: st}, do_compression=True)


def write_trigno_csv(path, channels: list[tuple[str, str, int, float, np.ndarray]],
                     start: str = "2000-01-01 00:00:00", trailer: int = 6) -> None:
    """A Trigno Discover 2.x CSV: ``channels`` = (label, sensor name, sensor mode, rate Hz,
    values in mV). Each channel gets its own time column; a slower channel's cells end
    early; ``trailer`` empty analysis columns follow, as in the real exports."""
    n = max(len(v) for *_, v in channels)
    length = max(len(v) / fs for *_, fs, v in channels)
    lines = ["Application:, Trigno Discover (synthetic)", f"Date/Time:, {start}",
             f"Collection Length (seconds):, {length:.4f}",
             ", , ".join(s for _, s, _, _, _ in channels),
             ", , ".join(f"sensor mode: {m}" for _, _, m, _, _ in channels),
             ", ".join(f"{lab} Time Series (s), {lab} (mV)" for lab, *_ in channels),
             ", " + ", , ".join(f"{fs:.4f} Hz" for *_, fs, _ in channels),
             ", " + ", , ".join(f"{1 / fs} s" for *_, fs, _ in channels)
             + ", , Type, Name, Label, Time (s), , "]
    pad = ", " * trailer
    for i in range(n):
        cells = []
        for *_, fs, v in channels:
            if i < len(v):
                cells += [f"{i / fs:.7f}", f"{v[i]:.7f}"]
            else:
                cells += ["", ""]
        lines.append(", ".join(cells) + ", " + pad.rstrip())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
