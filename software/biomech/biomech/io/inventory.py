# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""What one recorded take contains, without loading its signals.

``inventory(folder)`` answers the questions every study script asks before it
touches a take: which streams exist (EMG, optical, cameras, sync report, IMU),
how many frames each camera wrote, and what kind of take it is. The kind comes
from the folder name (``biomech.cohort.parse_session_name``); the phase table
is only used to *check* it. A disagreement (an MVC folder holding motion
phases, or the reverse) is reported as a ``kind_mismatch`` finding, never
silently re-labelled.

``scan`` runs it over a data directory; ``select_takes`` picks the first or
latest take per participant (``first`` is the pre-specified rule of an01 §2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import h5py

from biomech.cohort import SessionInfo, discover_sessions, parse_session_name
from biomech.io.session import SessionData

MOTION_PHASES: tuple[str, ...] = (
    "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion",
)
MVC_PHASES: tuple[str, ...] = ("mvc_opening", "mvc_closing")


@dataclass
class TakeInventory:
    name: str
    path: Path
    pid: str | None
    subject_id: str | None          # "<pid>_<sex>_<age>", the legacy key of the scripts
    sex: str | None
    age: int | None
    take_kind: str                  # folder kind: "motion" | "mvc" | "unknown"
    phase_kind: str                 # "motion" | "mvc" | "other" | "none"
    phases: list[str] = field(default_factory=list)
    config_hash: str | None = None
    software_version: str | None = None
    recording_date: str | None = None
    has_emg: bool = False
    has_optical: bool = False
    has_cameras: bool = False
    has_sync: bool = False
    has_imu_mta: bool = False
    has_imu_cra: bool = False
    has_jts_config: bool = False
    n_emg_samples: int = 0
    n_optical_frames: int = 0
    n_cam_frames: dict[str, int] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)

    @property
    def h5(self) -> Path:
        return self.path / "session.h5"

    @property
    def complete(self) -> bool:
        """Has a phase table (an unfinished take does not)."""
        return bool(self.phases)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["h5"] = str(self.h5)
        return d


def phase_kind(phase_names: Iterable[str], recording_mode: str | None = None) -> str:
    names = set(phase_names)
    if recording_mode == "mvc" or names & set(MVC_PHASES):
        return "mvc"
    if names & set(MOTION_PHASES) or "rest" in names:
        return "motion"
    return "other" if names else "none"


def _decode(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if hasattr(v, "item") and type(v).__module__ == "numpy":
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


def _opt_str(meta: dict[str, Any], key: str) -> str | None:
    v = meta.get(key)
    return str(v) if v not in (None, "") else None


def inventory(folder: str | Path) -> TakeInventory:
    """Inventory of one session folder (or its ``session.h5``)."""
    folder = Path(folder)
    if folder.name == "session.h5":
        folder = folder.parent
    h5 = folder / "session.h5"
    if not h5.is_file():
        raise FileNotFoundError(h5)
    info = parse_session_name(folder.name)
    findings: list[str] = []
    if info is None:
        findings.append("unparseable_name")

    phases: list[str] = []
    meta: dict[str, Any] = {}
    try:
        with SessionData(folder) as sd:
            phases = sd.phase_names()
            meta = {str(k): _decode(v) for k, v in dict(sd.meta).items()}
    except Exception as exc:  # noqa: BLE001 - a broken take is a finding, not a crash
        findings.append(f"unreadable: {type(exc).__name__}: {exc}")

    n_cam: dict[str, int] = {}
    with h5py.File(h5, "r") as f:
        if "cameras" in f:
            for cid, g in f["cameras"].items():
                if isinstance(g, h5py.Group) and "t_master_us" in g:
                    n_cam[str(cid)] = int(g["t_master_us"].shape[0])
        emg = f.get("emg")
        has_emg = emg is not None and "samples_uv" in emg
        n_emg = int(emg["samples_uv"].shape[0]) if has_emg else 0
        opt = f.get("optical")
        has_opt = opt is not None and "t_master_us" in opt
        n_opt = int(opt["t_master_us"].shape[0]) if has_opt else 0
        has_sync = "sync" in f and "report_json" in f["sync"].attrs
        has_mta = "imu_mta" in f and "quat" in f["imu_mta"]
        has_cra = "imu_cra" in f and "quat" in f["imu_cra"]
        if not meta and "meta" in f:
            meta = {str(k): _decode(v) for k, v in f["meta"].attrs.items()}

    pk = phase_kind(phases, _opt_str(meta, "recording_mode"))
    take_kind = info["kind"] if info else "unknown"
    if not phases:
        findings.append("no_phases")
    elif take_kind in ("motion", "mvc") and pk in ("motion", "mvc") and pk != take_kind:
        findings.append(f"kind_mismatch: folder says {take_kind}, phases say {pk}")

    return TakeInventory(
        name=folder.name,
        path=folder,
        pid=info["pid"] if info else None,
        subject_id=f"{info['pid']}_{info['sex']}_{info['age']}" if info else None,
        sex=info["sex"] if info else None,
        age=info["age"] if info else None,
        take_kind=take_kind,
        phase_kind=pk,
        phases=phases,
        config_hash=_opt_str(meta, "config_hash"),
        software_version=_opt_str(meta, "software_version"),
        recording_date=_opt_str(meta, "recording_date"),
        has_emg=has_emg,
        has_optical=has_opt,
        has_cameras=bool(n_cam),
        has_sync=has_sync,
        has_imu_mta=has_mta,
        has_imu_cra=has_cra,
        has_jts_config=any(folder.glob("*_jts_config.json")),
        n_emg_samples=n_emg,
        n_optical_frames=n_opt,
        n_cam_frames=n_cam,
        findings=findings,
    )


def scan(data_dir: str | Path, *, pids: Iterable[str] | None = None,
         kind: str | Iterable[str] | None = None) -> list[TakeInventory]:
    """Inventory of every well-named take under ``data_dir`` (oldest first per pid).

    ``pids`` matches participant ids exactly (``s1`` never matches ``s10``);
    ``kind`` filters on the folder kind.
    """
    want = {p.lower() for p in pids} if pids else None
    kinds = None if kind is None else ({kind} if isinstance(kind, str) else set(kind))
    out: list[TakeInventory] = []
    for s in discover_sessions(data_dir):
        if want is not None and s.pid not in want:
            continue
        if kinds is not None and s.kind not in kinds:
            continue
        out.append(inventory(s.path))
    return out


def subject_pids(subject: str | None) -> list[str] | None:
    """Exact pid list from a ``--subject`` value: ``s03`` or ``s03_m_33`` (comma-separated ok)."""
    if not subject:
        return None
    out = []
    for part in str(subject).split(","):
        part = part.strip().lower()
        if part:
            out.append(part.split("_")[0])
    return out or None


def select_takes(takes: list[Any], rule: Literal["first", "latest", "all"] = "first"
                 ) -> list[Any]:
    """One take per (pid, kind): the oldest (``first``, an01 §2), the newest, or all.

    Works on :class:`TakeInventory` and :class:`biomech.cohort.SessionInfo` alike.
    """
    if rule == "all":
        return list(takes)

    def key(t: Any) -> tuple[str, str]:
        kind = getattr(t, "take_kind", None) or getattr(t, "kind", "")
        return (str(t.pid), str(kind))

    def stamp(t: Any) -> Any:
        info = parse_session_name(t.name)
        return info["recorded_utc"] if info else t.name

    best: dict[tuple[str, str], Any] = {}
    for t in takes:
        k = key(t)
        cur = best.get(k)
        if cur is None or (stamp(t) > stamp(cur) if rule == "latest" else stamp(t) < stamp(cur)):
            best[k] = t
    return [t for t in takes if any(t is b for b in best.values())]


__all__ = [
    "MOTION_PHASES", "MVC_PHASES", "SessionInfo", "TakeInventory", "inventory",
    "phase_kind", "scan", "select_takes", "subject_pids",
]
