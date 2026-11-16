# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Readers for the reference systems of the validation study (plan D1).

* **Qualisys QTM** ``.mat`` export (QTM 2024.2, MAT v5): rigid bodies as (N, 4, 4)
  poses in mm with NaN where the body is not tracked, residuals, frame times and the
  labelled marker trajectories. A minimal port of ``jts.qualisys.QualysisData``
  (biomech must not depend on the ``jts`` package).
* **Delsys Trigno Discover** CSV export: one time column per sensor (rates may differ
  per sensor), values in mV, channels mapped to muscles by their *label* (the sensor
  order differs between recording days), returned in µV.

Both readers are read-only and keep every sample; selecting windows is the caller's job.
Times are each instrument's own clock in seconds (QTM: frame index / frame rate from
``StartFrame``; Delsys: the per-sensor time column).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

# -- Qualisys ---------------------------------------------------------------------------


@dataclass
class QtmBody:
    name: str
    T: np.ndarray            # (N, 4, 4) mm; NaN where untracked
    residual: np.ndarray     # (N,) mm; NaN where untracked

    def valid(self) -> np.ndarray:
        return np.isfinite(self.T.reshape(len(self.T), -1)).all(axis=1)


@dataclass
class QtmData:
    path: Path
    take: str                                   # MATLAB variable name (the take label)
    file: str                                   # original .qtm path stored by QTM
    timestamp: str                              # QTM "Timestamp" string (local wall clock)
    start_frame: int
    n_frames: int
    fs_hz: float
    t_s: np.ndarray                             # (N,) QTM time of each frame
    bodies: dict[str, QtmBody]
    markers: dict[str, np.ndarray]              # label -> (N, 3) mm, NaN when missing
    marker_residual: dict[str, np.ndarray] = field(default_factory=dict)
    # per-frame trajectory type code of the labelled markers (QTM ``Type``): 0 = missing,
    # 1 = measured, 2 = gap-filled (QTM MATLAB export; other codes kept as they are)
    marker_type: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    # label correction of label-ambiguous bodies (:func:`fix_label_swaps`): per body the
    # per-frame flag "QTM labels were permuted" and a summary (margin, counts)
    label_swapped: dict[str, np.ndarray] = field(default_factory=dict)
    label_info: dict[str, dict[str, Any]] = field(default_factory=dict)

    def body_markers(self, body: str) -> list[str]:
        """Labels of the markers of a rigid body (QTM labels ``"<body> - <k>"``)."""
        return sorted(k for k in self.markers if k.split(" - ")[0].strip() == body)

    def pose(self, body: str) -> np.ndarray:
        return self.bodies[body].T

    @property
    def wall_start(self) -> str:
        """The wall-clock part of ``Timestamp`` (QTM PC local time)."""
        return self.timestamp.split("\t")[0].strip()


def _scalar(v: Any) -> Any:
    a = np.asarray(v)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.reshape(-1)[0])
    return a.reshape(-1)[0] if a.size else None


def _strings(v: Any) -> list[str]:
    """MATLAB cellstr (1, n) object array -> list of str."""
    a = np.asarray(v, dtype=object).reshape(-1)
    return [str(_scalar(x)) for x in a]


def _field(s: Any, name: str) -> Any:
    return s[name] if name in (s.dtype.names or ()) else None


def read_qtm_mat(path: str | Path, fix_labels: bool = True) -> QtmData:
    """Read one QTM ``.mat`` export.

    Rotations are stored row-wise as 9 values in column-major order (QTM convention),
    i.e. ``R = r.reshape(3, 3).T``. A body is untracked in a frame when its residual is
    NaN, any pose element is NaN, or position and rotation are all zero (older exports).
    ``fix_labels``: correct QTM label swaps of label-ambiguous bodies
    (:func:`fix_label_swaps`; bodies whose marker geometry is unambiguous are unchanged).
    """
    q = _read_qtm_mat(path)
    if fix_labels:
        for b in list(q.bodies):
            fix_label_swaps(q, b)
    return q


# -- label swaps of near-symmetric rigid bodies (erratum EV1) -----------------------------
def body_template(q: QtmData, body: str, frames: np.ndarray | None = None
                  ) -> np.ndarray | None:
    """The marker template of ``body`` in its QTM body frame: per marker the median of
    ``T⁻¹ m_k`` over frames with a valid pose and all markers (``frames``: only these,
    e.g. the frames whose labels are known to be right). QTM fits its body definition to
    the *labels*, so this approximates the definition even with some label-swapped frames;
    restricted to correctly labelled frames it is the definition itself."""
    names = q.body_markers(body)
    if len(names) < 3 or body not in q.bodies:
        return None
    P = np.stack([q.markers[k] for k in names], axis=1)               # (N, M, 3)
    T = q.bodies[body].T
    ok = np.isfinite(T.reshape(len(T), -1)).all(axis=1) & np.isfinite(P).all(axis=(1, 2))
    if frames is not None:
        ok &= np.asarray(frames, bool)
    if ok.sum() < 10:
        return None
    R, t = T[ok, :3, :3], T[ok, :3, 3]
    loc = np.einsum("nji,nmj->nmi", R, P[ok] - t[:, None, :])         # R^T (m - t)
    return np.median(loc, axis=0)


def _pairs(m: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def label_ambiguity(template: np.ndarray, margin_mm: float = 3.0
                    ) -> tuple[float, list[tuple[int, ...]]]:
    """Label ambiguity of a marker template: the smallest largest-pairwise-distance change
    over the non-identity permutations (mm) and the permutations within ``margin_mm``.
    A nearly isosceles triangle (e.g. the CT, sides 60.0/74.3/75.9 mm) has a margin of
    ~1.6 mm: swapped labels still fit the body, rotated ~180°."""
    import itertools

    m = len(template)
    D = np.linalg.norm(template[:, None] - template[None], axis=2)
    best, cand = np.inf, []
    for perm in itertools.permutations(range(m)):
        if perm == tuple(range(m)):
            continue
        mis = max(abs(D[i, j] - D[perm[i], perm[j]]) for i, j in _pairs(m))
        best = min(best, mis)
        if mis < margin_mm:
            cand.append(perm)
    return float(best), cand


def _kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``R, t`` minimizing ``|R A_k + t - B_k|`` (rows are points)."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([False], ok, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def fix_label_swaps(q: QtmData, body: str, *, margin_mm: float = 3.0,
                    informative_mm2: float = 2.5, min_frames: int = 10,
                    min_share: float = 0.8, max_gap_s: float = 1.0,
                    jump_deg: tuple[float, float] = (45.0, 135.0)) -> np.ndarray | None:
    """Correct QTM label swaps of a label-ambiguous ``body`` in place (erratum EV1).

    * **Ambiguous** when a non-identity permutation of the markers changes no template
      distance by more than ``margin_mm`` (:func:`label_ambiguity`); otherwise nothing
      changes. QTM fits its definition to the labels, so a swap rotates the body ~180°
      while the fit still looks clean.
    * **Runs.** QTM keeps the labels while it tracks every marker, so the labelling is
      decided per run of frames with all markers present.
    * **Distance evidence.** Each candidate labelling is scored per frame by the squared
      mismatch of the observed inter-marker distances with the *measured* geometry (the
      median over frames the definition labels best; QTM's stored definition can be less
      asymmetric). A frame is informative when the best labelling fits
      (< ``informative_mm2``) and beats the others by ``informative_mm2``; a distorted
      cluster (a marker displaced by more than the asymmetry) is not. A run is decided
      when ≥ ``min_frames`` informative frames agree to ≥ ``min_share``.
    * **Continuity.** An undecided run takes the labelling under which its pose continues
      a decided neighbour across a gap ≤ ``max_gap_s`` (rotation jump < ``jump_deg[0]``
      against > ``jump_deg[1]`` for the alternative); passes run until nothing changes.
      Runs still undecided keep the QTM labels and are flagged (``label_undetermined``).
    * **Refit.** Frames with a non-identity labelling get the markers relabelled and the
      pose refit to the definition (Kabsch), so the body frame (and a tip defined in it)
      is unchanged; the QTM pose is kept elsewhere.

    Uses only the Qualisys data, never the system under test. Returns the per-frame swap
    flag (also ``q.label_swapped[body]``; summary in ``q.label_info[body]``)."""
    L = body_template(q, body)
    if L is None:
        return None
    margin, cand = label_ambiguity(L, margin_mm)
    names = q.body_markers(body)
    n = q.n_frames
    flag = np.zeros(n, bool)
    q.label_info[body] = {"margin_mm": margin, "ambiguous": bool(cand), "n_frames": 0,
                          "n_swapped": 0, "n_undetermined": 0, "n_runs": 0,
                          "runs_by_continuity": 0, "permutations": [list(p) for p in cand]}
    if not cand:
        q.label_swapped[body] = flag
        return flag
    P = np.stack([q.markers[k] for k in names], axis=1)               # (N, M, 3)
    T = q.bodies[body].T
    ok = np.isfinite(P).all(axis=(1, 2))
    perms = [tuple(range(len(names)))] + list(cand)
    pr = _pairs(len(names))
    d_obs = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in pr], axis=1)

    def expected(D: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
        return np.array([D[perm[i], perm[j]] for i, j in pr])

    def costs(D: np.ndarray) -> np.ndarray:                          # (N, n_perm)
        return np.stack([((d_obs - expected(D, pm)) ** 2).sum(axis=1) for pm in perms], 1)

    c = costs(np.linalg.norm(L[:, None] - L[None], axis=2))
    first = ok & (np.argmin(c, axis=1) == 0)              # measured geometry, identity frames
    if first.sum() >= 10:
        E = np.zeros((len(names), len(names)))
        med = np.median(d_obs[first], axis=0)
        for k, (i, j) in enumerate(pr):
            E[i, j] = E[j, i] = med[k]
        c = costs(E)
    best = np.argmin(c, axis=1)
    srt = np.sort(c, axis=1)
    informative = ok & (srt[:, 0] < informative_mm2) & (srt[:, 1] - srt[:, 0] > informative_mm2)

    def pose(i: int, j: int) -> np.ndarray:
        inv = np.argsort(perms[j])          # observed label i is definition marker perm[i]
        R, t = _kabsch(L, P[i][inv])
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = R, t
        return out

    runs = _runs(ok)
    state: list[int | None] = []
    for a, b in runs:
        votes = best[a:b][informative[a:b]]
        if len(votes) >= min_frames:
            j = int(np.bincount(votes, minlength=len(perms)).argmax())
            state.append(j if np.mean(votes == j) >= min_share else None)
        else:
            state.append(None)
    by_cont = 0
    changed = True
    while changed:
        changed = False
        for r, (a, b) in enumerate(runs):
            if state[r] is not None:
                continue
            for nb, mine, theirs in ((r - 1, a, None), (r + 1, b - 1, None)):
                if not 0 <= nb < len(runs) or state[nb] is None:
                    continue
                na, nbb = runs[nb]
                gap = (a - nbb) if nb < r else (na - b)
                if gap / q.fs_hz > max_gap_s:
                    continue
                ref_i = nbb - 1 if nb < r else na
                R_ref = pose(ref_i, state[nb])[:3, :3]
                jumps = [_rot_angle_deg(R_ref, pose(mine, j)[:3, :3]) for j in range(len(perms))]
                j = int(np.argmin(jumps))
                rest = [x for k, x in enumerate(jumps) if k != j]
                if jumps[j] < jump_deg[0] and min(rest) > jump_deg[1]:
                    state[r] = j
                    by_cont += 1
                    changed = True
                    break
    choice = np.zeros(n, int)
    undetermined = np.zeros(n, bool)
    for (a, b), st in zip(runs, state):
        if st is None:
            undetermined[a:b] = True
        else:
            choice[a:b] = st
    swapped = ok & (choice > 0)
    # the definition from the frames decided as correctly labelled (the all-frame median is
    # pulled towards the swapped geometry); the refit uses it
    L2 = body_template(q, body, ok & ~undetermined & (choice == 0))
    if L2 is not None:
        L = L2
    q.label_info[body]["template_mm"] = L.tolist()
    newT = T.copy()
    Pn = P.copy()
    for i in np.flatnonzero(swapped):
        newT[i] = pose(i, choice[i])
        Pn[i] = P[i][np.argsort(perms[choice[i]])]
    for k, name in enumerate(names):
        q.markers[name] = Pn[:, k]
    q.bodies[body] = QtmBody(body, newT, q.bodies[body].residual)
    q.label_swapped[body] = swapped
    q.label_swapped[f"{body}__undetermined"] = undetermined
    q.label_info[body].update(n_frames=int(ok.sum()), n_swapped=int(swapped.sum()),
                              n_undetermined=int(undetermined.sum()), n_runs=len(runs),
                              runs_by_continuity=by_cont)
    return swapped


def _read_qtm_mat(path: str | Path) -> QtmData:
    from scipy.io import loadmat

    path = Path(path)
    d = loadmat(path, struct_as_record=True, squeeze_me=False)
    keys = [k for k in d if not (k.startswith("__") and k.endswith("__"))]
    if not keys:
        raise ValueError(f"{path}: no QTM data variable")
    take = keys[0]
    m = d[take][0, 0]
    start = int(_scalar(m["StartFrame"]))
    n = int(_scalar(m["Frames"]))
    fs = float(_scalar(m["FrameRate"]))
    t = (start - 1 + np.arange(n, dtype=np.float64)) / fs

    bodies: dict[str, QtmBody] = {}
    rb_f = _field(m, "RigidBodies")
    if rb_f is not None:
        rb = rb_f[0, 0]
        names = _strings(rb["Name"])
        pos = np.asarray(rb["Positions"], dtype=np.float64)        # (B, 3, N)
        rot = np.asarray(rb["Rotations"], dtype=np.float64)        # (B, 9, N)
        res = np.asarray(rb["Residual"], dtype=np.float64)         # (B, 1, N)
        for b, name in enumerate(names):
            p = pos[b].T                                           # (N, 3)
            r = rot[b].T.reshape(n, 3, 3).transpose(0, 2, 1)       # column-major -> R
            T = np.zeros((n, 4, 4))
            T[:, :3, :3] = r
            T[:, :3, 3] = p
            T[:, 3, 3] = 1.0
            rr = res[b].reshape(-1)
            zero = (np.abs(p).sum(axis=1) == 0) & (np.abs(r).reshape(n, -1).sum(axis=1) == 0)
            bad = ~np.isfinite(rr) | ~np.isfinite(T.reshape(n, -1)).all(axis=1) | zero
            T[bad] = np.nan
            rr = rr.copy()
            rr[bad] = np.nan
            bodies[name] = QtmBody(name=name, T=T, residual=rr)

    markers: dict[str, np.ndarray] = {}
    mres: dict[str, np.ndarray] = {}
    mtype: dict[str, np.ndarray] = {}
    tr_f = _field(m, "Trajectories")
    if tr_f is not None:
        tr = tr_f[0, 0]
        lab_f = _field(tr, "Labeled")
        if lab_f is not None and lab_f.size:
            lab = lab_f[0, 0]
            labels = _strings(lab["Labels"])
            data = np.asarray(lab["Data"], dtype=np.float64)       # (M, 4, N): x, y, z, residual
            ty_f = _field(lab, "Type")
            types = (np.asarray(ty_f, dtype=np.float64).reshape(len(labels), -1)
                     if ty_f is not None and np.asarray(ty_f).size == len(labels) * n
                     else None)
            for k, name in enumerate(labels):
                xyz = data[k, :3].T.copy()
                miss = ~np.isfinite(xyz).all(axis=1) | (np.abs(xyz).sum(axis=1) == 0)
                xyz[miss] = np.nan
                markers[name] = xyz
                if data.shape[1] > 3:
                    mr = data[k, 3].copy()
                    mr[miss] = np.nan
                    mres[name] = mr
                if types is not None:
                    mtype[name] = types[k].astype(np.int8)

    events: list[dict[str, Any]] = []
    ev_f = _field(m, "Events")
    if ev_f is not None and ev_f.size:
        for e in np.asarray(ev_f, dtype=object).reshape(-1):
            try:
                events.append({k: _scalar(e[k]) for k in e.dtype.names})
            except Exception:                       # unknown layout: keep going
                continue

    return QtmData(path=path, take=take, file=str(_scalar(m["File"])),
                   timestamp=str(_scalar(m["Timestamp"])), start_frame=start, n_frames=n,
                   fs_hz=fs, t_s=t, bodies=bodies, markers=markers, marker_residual=mres,
                   marker_type=mtype, events=events)


@dataclass
class QtmCalibration:
    """The QTM calibration report (``qca_*.txt``): camera positions (mm, QTM world),
    points used and average residual per camera, and the wand-length SD."""
    cameras: dict[str, dict[str, float]]
    wand_sd_mm: float
    when: str


def read_qca_txt(path: str | Path) -> QtmCalibration:
    cams: dict[str, dict[str, float]] = {}
    wand, when = float("nan"), ""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) >= 6 and cells[0].isdigit():
            x, y, z, pts, res = (float(c) for c in cells[1:6])
            cams[cells[0]] = {"x_mm": x, "y_mm": y, "z_mm": z, "points": pts,
                              "avg_residual_mm": res}
        elif line.lower().startswith("standard deviation of wand length"):
            m = re.search(r"([-+0-9.eE]+)\s*\(mm\)", line)
            wand = float(m.group(1)) if m else float("nan")
        elif line.lower().startswith("calibration carried out"):
            when = line.split(":", 1)[1].strip()
    if not cams:
        raise ValueError(f"{path}: no camera rows")
    return QtmCalibration(cams, wand, when)


# -- Delsys -----------------------------------------------------------------------------

_MUSCLE = {"dig": "digastric", "digastric": "digastric", "mas": "masseter",
           "mass": "masseter", "masseter": "masseter", "tem": "temporalis",
           "temp": "temporalis", "temporalis": "temporalis"}
_SIDE = {"l": "l", "left": "l", "r": "r", "right": "r"}
_LABEL_RE = re.compile(r"^\s*(?P<a>[A-Za-z]+)[\s_\-]*(?P<b>[A-Za-z]+)\s*$")


def muscle_from_label(label: str) -> str | None:
    """'Dig_Left' / 'Mas Right' / 'Left_Temporalis' / 'TEM_L' -> 'digastric_l' etc.;
    None when the label names no known muscle and side."""
    m = _LABEL_RE.match(label)
    if not m:
        return None
    a, b = m["a"].lower(), m["b"].lower()
    for mus, side in ((a, b), (b, a)):
        if mus in _MUSCLE and side in _SIDE:
            return f"{_MUSCLE[mus]}_{_SIDE[side]}"
    return None


@dataclass
class DelsysChannel:
    label: str               # as exported, e.g. "Dig_Left"
    muscle: str              # e.g. "digastric_l" (the label itself, lowercased, if unknown)
    sensor: str              # e.g. "Avanti Sensor 1 (67700)"
    mode: str                # e.g. "84" (the "sensor mode" header)
    fs_hz: float             # header rate (falls back to the time column)
    t_s: np.ndarray          # (N,) the sensor's own time column
    uv: np.ndarray           # (N,) µV


@dataclass
class DelsysData:
    path: Path
    header: dict[str, str]                   # "Application", "Date/Time", ...
    channels: dict[str, DelsysChannel]       # muscle -> channel
    unmapped: list[str] = field(default_factory=list)

    def __getitem__(self, muscle: str) -> tuple[np.ndarray, np.ndarray]:
        ch = self.channels[muscle]
        return ch.t_s, ch.uv

    def __iter__(self) -> Iterator[str]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def keys(self) -> list[str]:
        return list(self.channels)

    def items(self) -> list[tuple[str, tuple[np.ndarray, np.ndarray]]]:
        return [(k, self[k]) for k in self.channels]

    @property
    def start(self) -> str:
        return self.header.get("Date/Time", "")

    @property
    def length_s(self) -> float:
        try:
            return float(self.header.get("Collection Length (seconds)", "nan"))
        except ValueError:
            return float("nan")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in next(csv.reader([line]))]


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_delsys_csv(path: str | Path, muscles: list[str] | None = None,
                    max_header_lines: int = 40) -> DelsysData:
    """Read a Trigno Discover CSV export.

    Layout (Trigno Discover 2.x): ``key:, value`` lines, then one line of sensor names
    (one per channel pair), one of ``sensor mode: N``, the column labels
    (``<label> Time Series (s)``, ``<label> (mV)``), the rates (``, 4370.37 Hz``), the
    sample periods, then the data. Trailing analysis columns (``Type, Name, ...``) are
    ignored. A slower sensor's columns end earlier (blank cells); its samples are
    trimmed to its own length. ``muscles`` limits which channels are parsed.
    """
    path = Path(path)
    head: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for _ in range(max_header_lines):
            line = fh.readline()
            if not line:
                break
            head.append(line.rstrip("\r\n"))
    i_lab = next((i for i, ln in enumerate(head) if "Time Series" in ln), None)
    if i_lab is None:
        raise ValueError(f"{path}: no 'Time Series' column header in the first "
                         f"{max_header_lines} lines (not a Trigno Discover CSV?)")
    i_data = next((i for i in range(i_lab + 1, len(head))
                   if head[i].strip() and _is_number(_cells(head[i])[0])), None)
    if i_data is None:
        raise ValueError(f"{path}: no data rows found after the column header")

    header: dict[str, str] = {}
    for ln in head[:i_lab]:
        c = _cells(ln)
        if len(c) >= 2 and c[0].endswith(":"):
            header[c[0][:-1].strip()] = c[1]
    sensors_line = next((head[i] for i in range(i_lab) if "Sensor" in head[i]
                         and not head[i].lower().startswith("sensor mode")), "")
    modes_line = next((head[i] for i in range(i_lab) if "sensor mode" in head[i].lower()), "")
    labels = _cells(head[i_lab])
    rates = _cells(head[i_lab + 1]) if i_lab + 1 < i_data else []

    # Channel pairs: (time col, value col) where the time label ends with "Time Series (s)".
    pairs: list[tuple[int, int, str]] = []
    for j, lab in enumerate(labels):
        mt = re.match(r"^(?P<name>.+?)\s+Time Series\s*\(s\)$", lab)
        if mt and j + 1 < len(labels):
            name = mt["name"].strip()
            val = labels[j + 1]
            unit = re.search(r"\((?P<u>[^)]*)\)\s*$", val)
            if not val.startswith(name) or not unit:
                raise ValueError(f"{path}: column {j + 1} ({val!r}) does not follow {lab!r}")
            pairs.append((j, j + 1, name))
    sensors = [s for s in _cells(sensors_line)] if sensors_line else []
    modes = _cells(modes_line) if modes_line else []

    def at(seq: list[str], j: int) -> str:
        return seq[j] if j < len(seq) else ""

    keep: list[tuple[int, int, str, str]] = []
    unmapped: list[str] = []
    for jt, jv, name in pairs:
        mus = muscle_from_label(name)
        if mus is None:
            unmapped.append(name)
            mus = re.sub(r"\W+", "_", name.strip().lower())
        if muscles is not None and mus not in muscles:
            continue
        keep.append((jt, jv, name, mus))
    if len({k[3] for k in keep}) != len(keep):
        raise ValueError(f"{path}: two channels map to the same muscle: "
                         f"{[(k[2], k[3]) for k in keep]}")

    import pandas as pd

    cols = sorted({c for jt, jv, _, _ in keep for c in (jt, jv)})
    df = pd.read_csv(path, skiprows=i_data, header=None, usecols=cols, dtype=np.float64,
                     engine="c", skipinitialspace=True, encoding="utf-8-sig")
    out: dict[str, DelsysChannel] = {}
    for jt, jv, name, mus in keep:
        t = df[jt].to_numpy()
        v = df[jv].to_numpy()
        n = int(np.flatnonzero(np.isfinite(t))[-1]) + 1 if np.isfinite(t).any() else 0
        t, v = t[:n], v[:n]
        unit = re.search(r"\((?P<u>[^)]*)\)\s*$", labels[jv])["u"].strip().lower()
        scale = {"mv": 1e3, "v": 1e6, "uv": 1.0, "µv": 1.0}.get(unit)
        if scale is None:
            raise ValueError(f"{path}: unknown unit {unit!r} for {name}")
        rate = re.match(r"^\s*([0-9.eE+-]+)\s*Hz", at(rates, jv) or at(rates, jt) or "")
        fs = float(rate.group(1)) if rate else (
            1.0 / float(np.median(np.diff(t))) if n > 1 else float("nan"))
        mode = at(modes, jt)
        mode = mode.split(":", 1)[1].strip() if ":" in mode else mode
        out[mus] = DelsysChannel(label=name, muscle=mus, sensor=at(sensors, jt), mode=mode,
                                 fs_hz=fs, t_s=t, uv=v * scale)
    return DelsysData(path=path, header=header, channels=out, unmapped=unmapped)
