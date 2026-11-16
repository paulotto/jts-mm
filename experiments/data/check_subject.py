#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Plausibility of trajectory, EMG and IMU for one subject's motion and MVC takes.

Looks under the data directory (and its ``tmp/``, and one level of nested
folders) for session folders of exactly that participant. Every motion take is
checked against the MVC take recorded immediately before it.

EMG is checked on the ``biomech.emg.prepare.EmgPrep`` chain (mains fit at the
measured line frequency, band-pass, *masked* envelope): rest noise, mains share
and SNR use unmasked samples only; MVC is the p95 of the masked MVC envelope.

Exit status is 0 when nothing failed, 1 when a check failed, 2 when no session
was found.

Examples
--------
  ./check_subject.py s01_f_31
  ./check_subject.py s03 --data-dir ~/jtsmm_data
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biomech.cohort import discover_sessions
from biomech.emg.prepare import EmgParams, EmgPrep
from biomech.io.inventory import MOTION_PHASES
from biomech.io.session import OPENERS, SessionData
from biomech.kinematics.se3 import relative
from biomech.paths import resolve_data_dir

OPENING_MUSCLES = OPENERS
_SUBJECT_RE = re.compile(r"s\d+(_[mf]_\d+)?", re.IGNORECASE)
Z_MAX_KOHM = 50.0


def mask_phase(t_us: np.ndarray, t0: int, t1: int) -> np.ndarray:
    t = np.asarray(t_us, dtype=np.int64)
    return (t >= int(t0)) & (t <= int(t1))


def snr_std(signal: np.ndarray, noise: np.ndarray) -> float:
    s = np.asarray(signal, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    s, n = s[np.isfinite(s)], n[np.isfinite(n)]
    if s.size < 8 or n.size < 8 or float(np.std(n)) < 1e-12:
        return float("nan")
    return float(np.std(s) / np.std(n))


@dataclass
class Finding:
    level: str  # PASS, WARN, FAIL
    text: str


@dataclass
class Take:
    path: Path
    kind: str  # motion | mvc
    stamp: str

    @property
    def h5(self) -> Path:
        return self.path / "session.h5"


def _session_roots(root: Path) -> list[Path]:
    """``root``, ``root/tmp`` and their direct sub-folders that are not sessions themselves."""
    out: list[Path] = []
    for base in (root, root / "tmp"):
        if not base.is_dir():
            continue
        out.append(base)
        out.extend(p for p in sorted(base.iterdir())
                   if p.is_dir() and not (p / "session.h5").is_file())
    return out


def find_takes(subject: str, roots: list[Path]) -> list[Take]:
    """Takes of exactly this participant (``s01`` or ``s01_f_31``; ``s1`` never matches ``s10``)."""
    key = subject.lower()
    found: list[Take] = []
    seen: set[Path] = set()
    for root in roots:
        for base in _session_roots(root):
            for info in discover_sessions(base):
                full = f"{info.pid}_{info.sex}_{info.age}"
                if key not in (info.pid, full) or info.path.resolve() in seen:
                    continue
                seen.add(info.path.resolve())
                found.append(Take(path=info.path, kind=info.kind,
                                  stamp=info.recorded_utc.strftime("%Y%m%dT%H%M%SZ")))
    return sorted(found, key=lambda t: t.stamp)


def pair_mvc(motion: Take, mvcs: list[Take]) -> Take | None:
    earlier = [m for m in mvcs if m.stamp and motion.stamp and m.stamp <= motion.stamp]
    pool = earlier or mvcs
    if not pool:
        return None
    return max(pool, key=lambda m: m.stamp)


def _level_band(value: float, *, ok: tuple[float, float], warn: tuple[float, float]) -> str:
    if ok[0] <= value <= ok[1]:
        return "PASS"
    if warn[0] <= value <= warn[1]:
        return "WARN"
    return "FAIL"


def _add(findings: list[Finding], level: str, text: str) -> None:
    findings.append(Finding(level, text))


def check_impedance(take: Take, findings: list[Finding]) -> None:
    path = take.path / "integrity.json"
    if not path.is_file():
        _add(findings, "WARN", f"{take.kind}: no integrity.json, contact not checked")
        return
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        channels = doc["checks"]["emg"]["detail"]["channels"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        _add(findings, "WARN", f"{take.kind}: integrity.json unreadable ({exc})")
        return
    for ch in channels:
        name = str(ch.get("name") or ch.get("channel"))
        z = ch.get("z_kohm")
        if z is None:
            continue
        z = float(z)
        if z > Z_MAX_KOHM or ch.get("contact_hard_fail"):
            _add(findings, "FAIL", f"{take.kind} {name}: |Z| {z:.0f} kΩ > {Z_MAX_KOHM:.0f}")
        else:
            _add(findings, "PASS", f"{take.kind} {name}: |Z| {z:.0f} kΩ")


def check_emg(motion: Take, mvc: Take | None, findings: list[Finding]) -> None:
    from jtsmm.emg.mains import mains_share

    with ExitStack() as stack:
        sd = stack.enter_context(SessionData(motion.path))
        sdv = stack.enter_context(SessionData(mvc.path)) if mvc is not None else None
        if sd.phase("rest") is None:
            _add(findings, "FAIL", "EMG: no rest phase")
            return
        prep = EmgPrep(sd, sdv, EmgParams())
        names = list(prep.names)
        fs = sd.emg_fs
        rest = prep.phase("rest")
        acts = {m: prep.phase(m) for m in MOTION_PHASES if sd.phase(m) is not None}
        refs = prep.ref_uv("p95")
        have_mvc_phases = sdv is not None and all(
            sdv.phase(p) is not None for p in ("mvc_opening", "mvc_closing"))

    for i, name in enumerate(names):
        rest_ok = rest.cleaned[~rest.mask[:, i], i]
        rest_std = float(np.std(rest_ok)) if rest_ok.size else float("nan")
        level = "PASS" if rest_std <= 15 else "WARN" if rest_std <= 40 else "FAIL"
        _add(findings, level, f"EMG {name}: rest {rest_std:.1f} µV")
        blocks = [rest.cleaned[:, i]] + [pe.cleaned[:, i] for pe in acts.values()]
        share = max(max(mains_share(b, fs, 50.0), mains_share(b, fs, 60.0)) for b in blocks)
        level = "PASS" if share < 0.1 else "WARN" if share < 0.5 else "FAIL"
        _add(findings, level, f"EMG {name}: mains share after cleaning {share:.2f} (worst phase)")
        snrs = []
        for motion_name, pe in acts.items():
            ok = ~pe.mask[:, i]
            if int(ok.sum()) < 8:
                continue
            snrs.append((motion_name, snr_std(pe.cleaned[ok, i], rest_ok)))
        if not snrs:
            _add(findings, "FAIL", f"EMG {name}: no motion phase")
            continue
        best_name, best = max(snrs, key=lambda item: item[1])
        level = "PASS" if best >= 2.0 else "WARN" if best >= 1.3 else "FAIL"
        _add(findings, level, f"EMG {name}: best SNR {best:.1f} ({best_name})")

    if mvc is None:
        _add(findings, "FAIL", "EMG: no MVC recording, %MVC not checked")
        return
    if not have_mvc_phases:
        _add(findings, "FAIL", "EMG: MVC is missing mvc_opening or mvc_closing")
        return
    for i, name in enumerate(names):
        scale = float(refs[i])
        kind = "opening" if name in OPENING_MUSCLES else "closing"
        floor = 25.0 if name in OPENING_MUSCLES else 50.0
        level = "PASS" if scale >= floor else "WARN" if scale >= floor * 0.5 else "FAIL"
        _add(findings, level, f"MVC {name} ({kind}): {scale:.0f} µV")
        for motion_name, pe in acts.items():
            env = pe.env[:, i]
            env = env[np.isfinite(env)]
            if env.size < 8 or not np.isfinite(scale) or scale <= 0:
                continue
            peak = float(np.percentile(env, 95))
            ratio = peak / scale
            level = "PASS" if ratio <= 1.0 else "WARN" if ratio <= 1.3 else "FAIL"
            _add(
                findings,
                level,
                f"%MVC {name} {motion_name}: p95 {peak:.0f} µV = {100 * ratio:.0f}% of MVC",
            )


def _ptp(xyz: np.ndarray) -> np.ndarray:
    return np.nanmax(xyz, axis=0) - np.nanmin(xyz, axis=0)


def check_trajectory(motion: Take, findings: list[Finding]) -> None:
    with SessionData(motion.path) as sd:
        opt = sd.optical()
        phases = {p.name: p for p in sd.phases}
    if opt is None or "mta" not in opt.poses or "cra" not in opt.poses:
        _add(findings, "FAIL", "trajectory: no pose_mta/pose_cra")
        return
    t = opt.t_us.astype(np.int64)
    rel = relative(opt.poses["cra"], opt.poses["mta"])[:, :3, 3]
    world = opt.poses["mta"][:, :3, 3]
    cra = opt.poses["cra"][:, :3, 3]
    if "rest" not in phases:
        _add(findings, "FAIL", "trajectory: no rest phase")
        return
    rest_m = mask_phase(t, phases["rest"].t_start_us, phases["rest"].t_end_us) & np.isfinite(rel).all(1)
    if int(rest_m.sum()) < 10:
        _add(findings, "FAIL", "trajectory: rest has no valid relative pose")
        return
    origin = np.median(rel[rest_m], axis=0)
    rest_dist = np.linalg.norm(rel[rest_m] - origin, axis=1)
    rest_p95 = float(np.percentile(rest_dist, 95))
    level = _level_band(rest_p95, ok=(0.0, 3.0), warn=(0.0, 8.0))
    _add(findings, level, f"trajectory rest: p95 {rest_p95:.1f} mm about the rest pose")

    expect = {
        "open_close": ((20.0, 55.0), (12.0, 70.0), "open/close"),
        "protrusion_retrusion": ((4.0, 18.0), (2.0, 25.0), "protrusion"),
        "left_right": ((6.0, 22.0), (3.0, 30.0), "laterotrusion"),
        "chewing": ((4.0, 25.0), (2.0, 35.0), "chewing"),
    }
    for name, (ok, warn, label) in expect.items():
        phase = phases.get(name)
        if phase is None:
            _add(findings, "FAIL", f"trajectory: phase {name} is missing")
            continue
        mask = mask_phase(t, phase.t_start_us, phase.t_end_us)
        finite = mask & np.isfinite(rel).all(1)
        frac = float(finite.sum() / mask.sum()) if mask.any() else 0.0
        level = "PASS" if frac >= 0.98 else "WARN" if frac >= 0.90 else "FAIL"
        _add(findings, level, f"trajectory {label}: {100 * frac:.0f}% valid frames")
        if int(finite.sum()) < 10:
            continue
        dist = np.linalg.norm(rel[finite] - origin, axis=1)
        p95 = float(np.percentile(dist, 95))
        rom = float(np.nanmax(_ptp(rel[finite])))
        level = _level_band(p95, ok=ok, warn=warn)
        _add(findings, level, f"trajectory {label}: p95 {p95:.0f} mm from rest, range {rom:.0f} mm")
        xyz = rel[finite]
        step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        step = step[np.isfinite(step)]
        if step.size:
            jump = float(step.max())
            level = "PASS" if jump <= 15 else "WARN" if jump <= 40 else "FAIL"
            _add(findings, level, f"trajectory {label}: largest frame step {jump:.1f} mm")

    if "open_close" in phases:
        rest_end = phases["rest"].t_end_us
        oc_start = phases["open_close"].t_start_us
        a = mask_phase(t, rest_end - 1_000_000, rest_end) & np.isfinite(world).all(1)
        b = mask_phase(t, oc_start, oc_start + 1_000_000) & np.isfinite(world).all(1) & np.isfinite(rel).all(1)
        if a.any() and b.any():
            d_world = float(np.linalg.norm(np.median(world[b], 0) - np.median(world[a], 0)))
            d_head = float(np.linalg.norm(np.median(cra[b], 0) - np.median(cra[a], 0)))
            d_jaw = float(np.linalg.norm(np.median(rel[b], 0) - np.median(rel[a], 0)))
            level = "PASS" if d_jaw <= 3 else "WARN" if d_jaw <= 8 else "FAIL"
            _add(
                findings,
                level,
                f"trajectory rest→open/close: jaw {d_jaw:.1f} mm, "
                f"head {d_head:.0f} mm, world {d_world:.0f} mm",
            )


def _quat_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)[:, [1, 2, 3, 0]]
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] *= -1
    return q


def _rotation_on(t_src: np.ndarray, q_wxyz: np.ndarray, t_dst: np.ndarray):
    from scipy.spatial.transform import Rotation, Slerp

    keep = np.ones(len(t_src), dtype=bool)
    keep[1:] = np.diff(t_src) > 0
    rot = Rotation.from_quat(_quat_xyzw(q_wxyz)[keep])
    return Slerp(t_src[keep].astype(np.float64), rot)(t_dst.astype(np.float64))


def _geodesic_from_first(rots) -> np.ndarray:
    delta = rots[0].inv() * rots
    return np.degrees(delta.magnitude())


def check_imu(motion: Take, findings: list[Finding]) -> None:
    import h5py
    from scipy.spatial.transform import Rotation

    with h5py.File(motion.h5, "r") as handle:
        missing = [name for name in ("imu_mta", "imu_cra") if name not in handle]
        if missing:
            _add(findings, "FAIL", "IMU missing: " + ", ".join(missing))
            return
        streams = {}
        for name, label in (("imu_mta", "mandible"), ("imu_cra", "cranium")):
            g = handle[name]
            quat = np.asarray(g["quat"], dtype=np.float64)
            t_quat = np.asarray(g["t_master_us_quat"], dtype=np.int64)
            accel = np.asarray(g["accel"], dtype=np.float64)
            t_accel = np.asarray(g["t_master_us_accel"], dtype=np.int64)
            qacc = np.asarray(g["quat_accuracy"]) if "quat_accuracy" in g else None
            streams[name] = (label, quat, t_quat, accel, t_accel, qacc)
            norm = np.linalg.norm(quat, axis=1)
            if np.nanmax(np.abs(norm - 1.0)) > 0.02 or not np.isfinite(quat).all():
                _add(findings, "FAIL", f"IMU {label}: quaternion is not unit length")
            else:
                _add(findings, "PASS", f"IMU {label}: quaternion unit length, n={len(quat)}")
            if qacc is not None:
                high = float(np.mean(qacc == 3))
                level = "PASS" if high >= 0.95 else "WARN" if high >= 0.8 else "FAIL"
                _add(findings, level, f"IMU {label}: {100 * high:.0f}% of quaternions at high accuracy")
            dt = np.diff(t_quat.astype(np.int64))
            back = dt[dt < 0]
            if back.size == 0:
                _add(findings, "PASS", f"IMU {label}: timestamps monotonic")
            else:
                worst = int(-back.min())
                level = "WARN" if worst <= 1000 and back.size / dt.size < 0.05 else "FAIL"
                _add(findings, level, f"IMU {label}: {back.size} backward steps, max {worst} µs")
            med = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
            gaps = int(np.sum(dt > 4 * med)) if med else 0
            level = "PASS" if gaps == 0 else "FAIL"
            gap_text = (
                f"IMU {label}: {gaps} gaps > 4 sample periods ({1e6 / med:.0f} Hz)"
                if med
                else f"IMU {label}: no sample period"
            )
            _add(findings, level, gap_text)

        with SessionData(motion.path) as sd:
            phases = {p.name: p for p in sd.phases}
        if "rest" not in phases:
            return
        for name, (label, _q, _tq, accel, t_accel, _qa) in streams.items():
            mask = mask_phase(t_accel, phases["rest"].t_start_us, phases["rest"].t_end_us)
            if int(mask.sum()) < 20:
                _add(findings, "FAIL", f"IMU {label}: no rest acceleration")
                continue
            mag = float(np.median(np.linalg.norm(accel[mask], axis=1)))
            level = _level_band(mag, ok=(9.4, 10.2), warn=(9.0, 10.6))
            _add(findings, level, f"IMU {label}: |a| at rest {mag:.2f} m/s²")

        if "open_close" not in phases or "optical" not in handle:
            _add(findings, "WARN", "IMU: open/close not compared with optical")
            return
        t_opt = np.asarray(handle["optical/t_master_us"], dtype=np.int64)
        rot_m = np.asarray(handle["optical/pose_mta"][:, :3, :3], dtype=np.float64)
        rot_c = np.asarray(handle["optical/pose_cra"][:, :3, :3], dtype=np.float64)
        phase = phases["open_close"]
        mask = mask_phase(t_opt, phase.t_start_us, phase.t_end_us)
        finite = mask & np.isfinite(rot_m).all((1, 2)) & np.isfinite(rot_c).all((1, 2))
        if int(finite.sum()) < 30:
            _add(findings, "FAIL", "IMU: too few optical poses during open/close")
            return
        t_dst = t_opt[finite]
        rel_opt = Rotation.from_matrix(np.einsum("nji,njk->nik", rot_c[finite], rot_m[finite]))
        ang_opt = _geodesic_from_first(rel_opt)
        _label_m, quat_m, t_m, *_rest_m = streams["imu_mta"]
        _label_c, quat_c, t_c, *_rest_c = streams["imu_cra"]
        try:
            rel_imu = _rotation_on(t_c, quat_c, t_dst).inv() * _rotation_on(t_m, quat_m, t_dst)
        except Exception as exc:  # noqa: BLE001
            _add(findings, "FAIL", f"IMU: orientation could not be interpolated ({exc})")
            return
        ang_imu = _geodesic_from_first(rel_imu)
        err = float(np.sqrt(np.mean((ang_imu - ang_opt) ** 2)))
        a = ang_imu - ang_imu.mean()
        b = ang_opt - ang_opt.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = float(a @ b / denom) if denom > 0 else float("nan")
        level = "PASS" if corr >= 0.75 and err <= 8 else "WARN" if corr >= 0.5 and err <= 15 else "FAIL"
        _add(
            findings,
            level,
            f"IMU open/close: r={corr:.2f} vs optical, error {err:.1f}° RMS",
        )


def report(subject: str, motion: Take, mvc: Take | None) -> list[Finding]:
    findings: list[Finding] = []
    print(f"\n{subject}")
    print(f"  motion  {motion.path}")
    print(f"  MVC     {mvc.path if mvc else '— missing —'}")
    check_impedance(motion, findings)
    if mvc is not None:
        check_impedance(mvc, findings)
    check_emg(motion, mvc, findings)
    check_trajectory(motion, findings)
    check_imu(motion, findings)
    for finding in findings:
        print(f"  {finding.level:4s}  {finding.text}")
    n_fail = sum(f.level == "FAIL" for f in findings)
    n_warn = sum(f.level == "WARN" for f in findings)
    print(f"  result: {n_fail} failed, {n_warn} warnings")
    return findings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subject", help="participant id, exact: s01 or s01_f_31")
    ap.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help="directory to scan, repeatable (default: $JTSMM_DATA_DIR, repo pilot data, "
        "~/jtsmm_data; each with tmp/ and one level of sub-folders)",
    )
    args = ap.parse_args(argv)
    if not _SUBJECT_RE.fullmatch(args.subject):
        print(f"error: expected a subject id like s01 or s01_f_31, got {args.subject}",
              file=sys.stderr)
        return 2

    roots = [resolve_data_dir(p) for p in args.data_dir] if args.data_dir else [resolve_data_dir()]
    takes = find_takes(args.subject, roots)
    motions = [t for t in takes if t.kind == "motion"]
    mvcs = [t for t in takes if t.kind == "mvc"]
    if not motions and not mvcs:
        print(f"error: no session for {args.subject} under {', '.join(str(r) for r in roots)}",
              file=sys.stderr)
        return 2
    if not motions:
        print(f"error: MVC found for {args.subject}, but no motion session", file=sys.stderr)
        return 2

    failed = False
    for motion in motions:
        findings = report(args.subject, motion, pair_mvc(motion, mvcs))
        failed = failed or any(f.level == "FAIL" for f in findings)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
