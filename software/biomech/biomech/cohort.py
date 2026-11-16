# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Session discovery and the cohort manifest.

Session folders are named ``session_<id>_<sex>_<age>[_mvc][_<tag>]_<UTC>_<hash>``. Sex
and age exist only there, so the manifest copies them once and every later
reader takes them from the manifest (an01 §1). ``validate`` re-parses the folder
names and fails on any disagreement.

A ``<tag>`` marks a take recorded for another purpose (``qualisys_comparison``: the
simultaneous JTS-MM + Qualisys/Delsys takes of the validation study). Tagged takes are
invisible by default -- to parsing, discovery and manifests -- and a study opts in with
``session_tags`` in its cohort manifest, so the dataset study can never pick one up.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

_SESSION_RE = re.compile(
    r"^session_(?P<pid>s\d+)_(?P<sex>[mf])_(?P<age>\d+)(?P<mvc>_mvc)?"
    r"(?:_(?P<tag>qualisys_comparison))?_"
    r"(?P<ts>\d{8}T\d{6}Z)_(?P<hash>[0-9a-f]+)$",
    re.I,
)
# Known take tags (opt-in per study via ``session_tags``); "" = an ordinary take.
SESSION_TAGS: tuple[str, ...] = ("qualisys_comparison",)


@dataclass(frozen=True)
class SessionInfo:
    name: str
    path: Path
    pid: str
    sex: str
    age: int
    kind: str  # "motion" | "mvc"
    recorded_utc: datetime
    tag: str = ""  # "" or one of SESSION_TAGS

    @property
    def h5(self) -> Path:
        return self.path / "session.h5"


def parse_session_name(name: str, tags: Iterable[str] = ()) -> dict[str, Any] | None:
    """Fields of a session folder name; None when it does not parse or carries a tag not
    listed in ``tags`` (tagged takes are opt-in, see the module docstring)."""
    m = _SESSION_RE.match(name)
    if not m:
        return None
    tag = (m["tag"] or "").lower()
    if tag and tag not in {t.lower() for t in tags}:
        return None
    out = {
        "pid": m["pid"].lower(),
        "sex": m["sex"].lower(),
        "age": int(m["age"]),
        "kind": "mvc" if m["mvc"] else "motion",
        "recorded_utc": datetime.strptime(m["ts"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc),
    }
    if tag:
        out["tag"] = tag
    return out


def discover_sessions(data_dir: str | Path, tags: Iterable[str] = ()) -> list[SessionInfo]:
    """All well-named session folders that contain a ``session.h5``, oldest first (tagged
    takes only when their tag is in ``tags``)."""
    root = Path(data_dir)
    out: list[SessionInfo] = []
    if not root.is_dir():
        return out
    for leaf in sorted(root.iterdir()):
        info = parse_session_name(leaf.name, tags) if leaf.is_dir() else None
        if info is None or not (leaf / "session.h5").is_file():
            continue
        out.append(SessionInfo(name=leaf.name, path=leaf, **info))
    return sorted(out, key=lambda s: (s.pid, s.recorded_utc))


@dataclass
class Participant:
    pid: str
    sex: str
    age: int
    motion: str                      # primary motion session folder name
    mvc: str | None                  # primary MVC session folder name
    include: bool = True
    exclusion_reason: str = ""
    secondary: list[str] = field(default_factory=list)  # other takes, never pooled
    flags: list[str] = field(default_factory=list)      # e.g. reference_subset
    notes: str = ""


@dataclass
class Cohort:
    study: str
    participants: list[Participant]
    data_dir: Path
    source: Path | None = None
    # Take tags this study may list (``session_tags`` in the manifest; default none).
    session_tags: tuple[str, ...] = ()
    # False: an included participant may lack an MVC take (``require_mvc`` in the manifest;
    # the validation study reads its MVC data from elsewhere). Default True.
    require_mvc: bool = True

    # -- access -----------------------------------------------------------------
    def included(self, pids: Iterable[str] | None = None) -> list[Participant]:
        want = {p.lower() for p in pids} if pids else None
        return [p for p in self.participants if p.include and (want is None or p.pid in want)]

    def get(self, pid: str) -> Participant:
        for p in self.participants:
            if p.pid == pid.lower():
                return p
        raise KeyError(pid)

    def session_path(self, name: str) -> Path:
        return self.data_dir / name

    def demographics(self) -> dict[str, Any]:
        """Aggregate-only demographics of the included participants (an01 §1)."""
        inc = self.included()
        ages = [p.age for p in inc]
        n = len(ages)
        mean = sum(ages) / n if n else float("nan")
        sd = (sum((a - mean) ** 2 for a in ages) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        return {
            "n": n,
            "n_female": sum(p.sex == "f" for p in inc),
            "n_male": sum(p.sex == "m" for p in inc),
            "age_mean": mean,
            "age_sd": sd,
            "age_min": min(ages) if ages else None,
            "age_max": max(ages) if ages else None,
        }

    # -- integrity ----------------------------------------------------------------
    def listed_sessions(self) -> set[str]:
        """Every session folder the manifest names (primary, MVC, secondary; all rows)."""
        return {n for p in self.participants for n in [p.motion, p.mvc, *p.secondary] if n}

    def unlisted_sessions(self) -> list[SessionInfo]:
        """Session folders under ``data_dir`` (with this manifest's ``session_tags``) that
        the manifest does not name: takes recorded but never assigned (A16)."""
        listed = self.listed_sessions()
        return [s for s in discover_sessions(self.data_dir, self.session_tags)
                if s.name not in listed]

    def validate(self) -> list[str]:
        """Problems that make the manifest unusable; empty when consistent."""
        problems: list[str] = []
        seen: set[str] = set()
        for p in self.participants:
            if p.pid in seen:
                problems.append(f"{p.pid}: listed twice")
            seen.add(p.pid)
            if not p.include and not p.exclusion_reason:
                problems.append(f"{p.pid}: excluded without a reason")
            for name, kind in [(p.motion, "motion"), (p.mvc, "mvc")] + [
                (s, None) for s in p.secondary
            ]:
                if not name:
                    if kind == "motion" or (kind == "mvc" and p.include and self.require_mvc):
                        problems.append(f"{p.pid}: no {kind} session")
                    continue
                info = parse_session_name(name, self.session_tags)
                if info is None:
                    tagged = parse_session_name(name, SESSION_TAGS)
                    if tagged:
                        problems.append(f"{p.pid}: {name} is a {tagged['tag']} take, not "
                                        f"allowed by this manifest's session_tags")
                    else:
                        problems.append(f"{p.pid}: unparseable session name {name}")
                    continue
                if (info["pid"], info["sex"], info["age"]) != (p.pid, p.sex, p.age):
                    problems.append(f"{p.pid}: {name} disagrees with pid/sex/age")
                if kind and info["kind"] != kind:
                    problems.append(f"{p.pid}: {name} is a {info['kind']} take, listed as {kind}")
                if not (self.session_path(name) / "session.h5").is_file():
                    problems.append(f"{p.pid}: missing {name}/session.h5")
        return problems


def _as_bool(v: object, pid: object, key: str) -> bool:
    """YAML booleans only: the string "false" would otherwise read as True."""
    if isinstance(v, bool):
        return v
    raise ValueError(f"cohort manifest: {pid}: {key} must be true or false, got {v!r}")


def load_cohort(path: str | Path, data_dir: str | Path) -> Cohort:
    path = Path(path)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parts = [
        Participant(
            pid=str(row["pid"]).lower(),
            sex=str(row["sex"]).lower(),
            age=int(row["age"]),
            motion=str(row.get("motion") or ""),
            mvc=row.get("mvc") or None,
            include=_as_bool(row.get("include", True), row.get("pid"), "include"),
            exclusion_reason=str(row.get("exclusion_reason") or ""),
            secondary=list(row.get("secondary") or []),
            flags=list(row.get("flags") or []),
            notes=str(row.get("notes") or ""),
        )
        for row in doc.get("participants") or []
    ]
    tags = tuple(str(t).lower() for t in (doc.get("session_tags") or []))
    unknown = sorted(set(tags) - set(SESSION_TAGS))
    if unknown:
        raise ValueError(f"cohort manifest: unknown session_tags {unknown}; known: "
                         f"{list(SESSION_TAGS)}")
    return Cohort(study=str(doc.get("study") or ""), participants=parts,
                  data_dir=Path(data_dir), source=path, session_tags=tags,
                  require_mvc=_as_bool(doc.get("require_mvc", True), "manifest", "require_mvc"))


def draft_manifest(data_dir: str | Path, study: str = "") -> dict[str, Any]:
    """Starting point for a manifest: the first motion and MVC take per participant.

    "First complete session" is the pre-specified rule for repeated takes
    (an01 §2); later takes go to ``secondary``. Exclusions are a human decision
    and are never drafted.
    """
    by_pid: dict[str, list[SessionInfo]] = {}
    for s in discover_sessions(data_dir):
        by_pid.setdefault(s.pid, []).append(s)
    rows = []
    for pid in sorted(by_pid, key=lambda x: int(x[1:])):
        takes = by_pid[pid]
        motion = [s for s in takes if s.kind == "motion"]
        mvc = [s for s in takes if s.kind == "mvc"]
        first = motion[0] if motion else takes[0]
        rows.append({
            "pid": pid,
            "sex": first.sex,
            "age": first.age,
            "motion": motion[0].name if motion else None,
            "mvc": mvc[0].name if mvc else None,
            "include": True,
            "secondary": [s.name for s in motion[1:] + mvc[1:]],
        })
    return {"study": study, "participants": rows}


def dump_manifest(doc: dict[str, Any]) -> str:
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=120)
