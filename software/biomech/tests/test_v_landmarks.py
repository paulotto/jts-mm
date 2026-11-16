# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Point-marking accuracy (validation plan VD4, ``validation.landmarks``) on the synthetic
two-system recording of ``test_v_anatomy``: a known JTS-MM digitization error at one tooth
is recovered in the Qualisys body frame, the frames tilt accordingly, and the frame-free
distances see it."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from biomech.io.reference import QtmBody, QtmData
from biomech.kinematics import se3
from biomech.validation import anatomy as AN
from biomech.validation.landmarks import run_ref_landmarks
from biomech.validation.selection import Interval

from test_v_anatomy import _recording

TM = {"HP": np.array([[0.0, 0, 0], [75.0, 0, 0], [26.0, 37.0, 0]]),
      "MP": np.array([[0.0, 0, 0], [30.0, 0, 0], [70.0, 50.0, 10.0]]),
      "CT": np.array([[0.0, 200.0, 0], [60.0, 200.0, 0], [28.0, 269.0, 0]])}


def _row(name: str, T: np.ndarray) -> dict:
    rx, ry, rz = np.degrees(Rotation.from_matrix(T[:3, :3]).as_rotvec())
    return {"transform": name, "rx_deg": rx, "ry_deg": ry, "rz_deg": rz,
            "tx_mm": T[0, 3], "ty_mm": T[1, 3], "tz_mm": T[2, 3]}


def _systems(err_mm=None, err_at="mand_point_2"):
    d_q = np.array([0.5, -1.0, 0.3])
    sys_, phases, rel = _recording(seed=2, d_q=d_q)
    S, Q = sys_["S"], sys_["Q"]
    T_ct_s = S["T_ct"].copy()
    if err_mm is not None:                 # JTS-MM marks this point off by err (S jaw frame)
        ph = next(p for p in phases if p.name == err_at)
        k = (S["t"] >= ph.t_start_us * 1e-6) & (S["t"] <= ph.t_end_us * 1e-6)
        Tb = S["T_jaw"] if AN.BODY_OF[err_at] == "mp" else S["T_head"]
        T_ct_s[k, :3, 3] += np.einsum("nij,j->ni", Tb[k, :3, :3], err_mm)
    bodies = {"CT": Q["T_ct"], "HP": Q["T_head"], "MP": Q["T_jaw"]}
    markers = {f"{b} - {i + 1}": se3.apply(T, TM[b][i]) for b, T in bodies.items()
               for i in range(3)}
    n = len(Q["t"])
    qtm = QtmData(path=None, take="t", file="", timestamp="", start_frame=1, n_frames=n,
                  fs_hz=50.0, t_s=Q["t"], markers=markers,
                  bodies={b: QtmBody(b, T, np.zeros(n)) for b, T in bodies.items()})
    qtm.label_info["CT"] = {"template_mm": TM["CT"].tolist()}
    Xr = Q["X"] @ se3.inv(S["X"])
    Y = {b: se3.inv(Xr @ Ts[0]) @ Tq[0] for b, Ts, Tq in
         (("HP", S["T_head"], Q["T_head"]), ("MP", S["T_jaw"], Q["T_jaw"]),
          ("CT", S["T_ct"], Q["T_ct"]))}
    reg = pd.DataFrame([_row("X", Xr)] + [_row(f"Y_{b}", Y[b]) for b in ("HP", "MP", "CT")])
    t = S["t"]
    sy = SimpleNamespace(
        pid="sX", tS=t, t_qm=t, S={"CT": T_ct_s, "HP": S["T_head"], "MP": S["T_jaw"]},
        qtm=qtm, phases=phases, quality_s={},
        windows=[Interval("sync_start", 0.0, t[-1] - 40.0)],     # CT moving: X from the tips
        held=[Interval("rest", t[-1] - 40.0, t[-1] - 30.0)],
        q_at_s=lambda b: bodies[b])
    dec = pd.Series({"dx_mm": d_q[0], "dy_mm": d_q[1], "dz_mm": d_q[2]})
    est = pd.DataFrame([{"method": "pivot_qualisys", "se_x_mm": 0.1, "se_y_mm": 0.1,
                         "se_z_mm": 0.1}])
    return sy, reg, dec, est, Y


def test_exact_marking_gives_zero_error():
    sy, reg, dec, est, _ = _systems()
    t, _ = run_ref_landmarks(sy, reg, dec, est, pd.DataFrame(), pd.DataFrame(), n_boot=40)
    e = t["landmark_errors"]
    assert len(e) >= 8 and e["e_mm"].max() < 0.05
    assert t["frame_errors"]["angle_deg"].max() < 0.05
    assert t["marking_summary"]["rest_incisor_offset_mm"].iloc[0] < 0.05
    assert (t["landmark_errors"]["sigma_ref_mm"] > 0).all()
    r = t["registration"].set_index("transform")
    assert r.loc["X", "source"] == "tips" and r.loc["X", "tip_rms_mm"] < 1e-6


def test_known_marking_error_is_recovered():
    err = np.array([3.0, -2.0, 1.0])
    sy, reg, dec, est, Y = _systems(err_mm=err, err_at="mand_point_2")
    t, _ = run_ref_landmarks(sy, reg, dec, est, pd.DataFrame(), pd.DataFrame(), n_boot=40)
    e = t["landmark_errors"].set_index("landmark")
    assert abs(e.loc["mand_point_2", "e_mm"] - np.linalg.norm(err)) < 0.05
    assert e.drop(index="mand_point_2")["e_mm"].max() < 0.05
    fr = t["frame_errors"].set_index("frame")
    assert fr.loc["F_mand", "angle_deg"] > 1.0 and fr.loc["F_max", "angle_deg"] < 0.05
    d = t["landmark_distances"]
    moved = d[(d["a"] == "mand_point_2") | (d["b"] == "mand_point_2")]
    assert moved["diff_mm"].abs().max() > 1.0
    assert d.drop(index=moved.index)["diff_mm"].abs().max() < 0.05
