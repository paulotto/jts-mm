#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Inter-subject variability on time-normalised cycles.

Kinematics from the ``cycles_*.json`` of ``5_4_2_motion_trajectory_analysis``. EMG envelopes (%MVC) are warped onto
the same optical cycle windows; they come from ``biomech.emg.prepare.EmgPrep``
(masked envelope / p95 MVC reference of the participant's first MVC take), the
same chain ``5_4_1_emg_properties`` reports, so no ``mvc_scales.json`` hand-over is needed.

Three analysis choices (kept side by side):

- Cycle-as-rep SPM ANOVA (original; anticonservative).
- Subject as unit: one mean cycle per subject, between-subject SD (no fake F).
- Amplitude-normalised cycles (shape, 0–1) with both of the above.

Usage
-----
  python eval_spm.py
  python eval_spm.py --cycles-dir ~/experiments/5_4_2_motion_trajectory_analysis/results/derived
  python eval_spm.py --data-dir ~/jtsmm_data --subject s01,s02,s03
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from biomech.cohort import parse_session_name
from biomech.emg.prepare import EmgParams, EmgPrep
from biomech.io.inventory import TakeInventory, scan, subject_pids
from biomech.io.session import CLOSERS, OPENERS, SessionData
from biomech.kinematics.cycles import time_normalize
from biomech.kinematics.peak_cycles import peak_to_peak_cycles
from biomech.kinematics.traces import load_kinematic_trace, opening_1d, sample_rate_hz
from biomech.paths import resolve_data_dir, results_dir, results_root
from biomech.results.provenance import write_json, write_provenance

STUDY = "5_4_3_inter_subject_variability"
CLOSING_MUSCLES = CLOSERS
OPENING_MUSCLES = OPENERS
TRAJ_MOTIONS = ("open_close", "protrusion_retrusion", "left_right", "chewing")
CYCLE_N = 101


def default_cycles_dir() -> Path:
    return results_root() / "5_4_2_motion_trajectory_analysis" / "results" / "derived"


def cluster_x_pct(extent: Any, n_z: int) -> tuple[float, float] | None:
    if not extent or len(extent) < 2:
        return None
    a, b = float(extent[0]), float(extent[-1])
    n = max(int(n_z), 2)
    return 100.0 * a / (n - 1), 100.0 * b / (n - 1)


def extents_pct(res: dict[str, Any]) -> list[tuple[float, float]]:
    """SPM-significant intervals as % cycle (gray bars in Pataky-style figures)."""
    z = np.asarray(res.get("z") or [], dtype=float)
    n_z = int(z.size) if z.size else CYCLE_N
    out: list[tuple[float, float]] = []
    for c in res.get("clusters") or []:
        xy = cluster_x_pct(c.get("extent"), n_z)
        if xy:
            out.append(xy)
    if out:
        return out
    zstar = res.get("zstar")
    if z.size and zstar is not None and np.isfinite(zstar):
        above = z > float(zstar)
        i = 0
        while i < len(above):
            if above[i]:
                j = i
                while j < len(above) and above[j]:
                    j += 1
                out.append((100.0 * i / (len(z) - 1), 100.0 * (j - 1) / (len(z) - 1)))
                i = j
            else:
                i += 1
    return out


def draw_cluster_bars(ax: Any, res: dict[str, Any]) -> None:
    """Gray bar at the top of the trajectory: SPM cluster F > F* (α=0.05)."""
    import matplotlib.transforms as mtransforms
    from matplotlib.patches import Rectangle

    extents = extents_pct(res)
    if not extents:
        return
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_xlim(0, 100)
    for x0, x1 in extents:
        w = max(x1 - x0, 0.6)
        ax.add_patch(
            Rectangle(
                (x0, 0.96),
                w,
                0.028,
                transform=trans,
                facecolor="#5d5d5d",
                edgecolor="none",
                clip_on=False,
                zorder=6,
            )
        )


def load_cycle_docs(d: Path) -> list[dict[str, Any]]:
    docs = []
    if not d.is_dir():
        return docs
    for p in sorted(d.glob("cycles_*.json")):
        docs.append(json.loads(p.read_text(encoding="utf-8")))
    return docs


def amp_normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    lo = float(np.nanmin(y))
    hi = float(np.nanmax(y))
    rng = hi - lo
    if not np.isfinite(rng) or rng < 1e-9:
        return np.full_like(y, np.nan)
    return (y - lo) / rng


def collect_groups(
    docs: list[dict[str, Any]], motion: str, *, ampnorm: bool
) -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    groups: list[np.ndarray] = []
    for sid in sorted({d.get("subject_id") for d in docs if d.get("subject_id")}):
        reps = []
        for doc in docs:
            if doc.get("subject_id") != sid:
                continue
            for cyc in (doc.get("cycles") or {}).get(motion) or []:
                y = np.asarray(cyc, dtype=np.float64)
                if ampnorm:
                    y = amp_normalize(y)
                if y.size and np.isfinite(y).all():
                    reps.append(y)
        if reps:
            groups.append(np.vstack(reps))
            labels.append(str(sid))
    return labels, groups


def _spm_anova1(groups: list[np.ndarray], labels: list[str], motion: str, alpha: float, note: str) -> dict[str, Any]:
    if len(groups) < 2:
        return {"ok": False, "reason": "need ≥2 subjects with cycles", "subjects": labels, "motion": motion}
    means = {lab: np.nanmean(g, axis=0).tolist() for lab, g in zip(labels, groups)}
    sds = {lab: np.nanstd(g, axis=0).tolist() for lab, g in zip(labels, groups)}
    try:
        import spm1d
    except ImportError:
        return {
            "ok": False,
            "reason": "spm1d not installed (pip install 'software/biomech[stats]')",
            "subjects": labels,
            "n_groups": len(groups),
            "motion": motion,
            "means": means,
            "sd": sds,
            "note": note,
        }
    try:
        spm = spm1d.stats.anova1(groups, equal_var=True)
        spi = spm.inference(alpha, interp=False)
    except Exception as exc:  # noqa: BLE001 - e.g. SPM1DError: zero variance at a node
        return {
            "ok": False,
            "reason": f"spm1d: {type(exc).__name__}: {str(exc).strip()[:200]}",
            "subjects": labels,
            "n_groups": len(groups),
            "motion": motion,
            "means": means,
            "sd": sds,
            "note": note,
        }
    clusters = []
    for c in getattr(spi, "clusters", None) or []:
        ep = getattr(c, "endpoints", None)
        if ep is None:
            xs = getattr(c, "x", None)
            if xs is not None and len(np.asarray(xs)):
                arr = np.asarray(xs).ravel()
                ep = [arr[0], arr[-1]]
        clusters.append(
            {
                "p": float(getattr(c, "P", getattr(c, "p", np.nan))),
                "extent": [float(ep[0]), float(ep[-1])] if ep is not None and len(ep) >= 2 else None,
            }
        )
    z = np.asarray(getattr(spi, "z", spm.z), dtype=np.float64)
    zstar = float(getattr(spi, "zstar", np.nan))
    return {
        "ok": True,
        "motion": motion,
        "subjects": labels,
        "n_reps": [int(g.shape[0]) for g in groups],
        "alpha": alpha,
        "zstar": zstar,
        "z": z.tolist(),
        "clusters": clusters,
        "means": means,
        "sd": sds,
        "note": note,
    }


def run_spm_reps(docs: list[dict[str, Any]], motion: str, alpha: float, *, ampnorm: bool) -> dict[str, Any]:
    labels, groups = collect_groups(docs, motion, ampnorm=ampnorm)
    note = (
        "pilot n; cycles treated as independent reps within subject (anova1)"
        + ("; amplitude-normalised 0–1" if ampnorm else "")
    )
    out = _spm_anova1(groups, labels, motion, alpha, note)
    out["ampnorm"] = ampnorm
    out["unit"] = "cycle"
    return out


def run_spm_emg_reps(
    recs: list[tuple[TakeInventory, TakeInventory | None]],
    motion: str,
    channels: tuple[str, ...],
    group_name: str,
    alpha: float,
    *,
    ampnorm: bool,
    span_cache: dict[str, dict[str, list[tuple[int, int]]]] | None = None,
    emg_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels, groups = collect_emg_groups(
        recs, motion, channels, span_cache=span_cache, emg_cache=emg_cache
    )
    if ampnorm:
        groups = [np.vstack([amp_normalize(row) for row in g]) for g in groups if len(g)]
    note = (
        "pilot n; EMG cycles as independent reps (anova1)"
        + ("; amplitude-normalised 0–1" if ampnorm else "")
    )
    out = _spm_anova1(groups, labels, motion, alpha, note)
    out["group"] = group_name
    out["ampnorm"] = ampnorm
    out["unit"] = "cycle"
    out["channels"] = list(channels)
    return out


def subject_unit_summary(docs: list[dict[str, Any]], motion: str, *, ampnorm: bool) -> dict[str, Any]:
    """One mean cycle per subject; between-subject SD of those means."""
    labels, groups = collect_groups(docs, motion, ampnorm=ampnorm)
    if len(groups) < 2:
        return {"ok": False, "reason": "need ≥2 subjects", "subjects": labels, "motion": motion}
    means = {lab: np.nanmean(g, axis=0) for lab, g in zip(labels, groups)}
    sds = {lab: np.nanstd(g, axis=0) for lab, g in zip(labels, groups)}
    stack = np.vstack([means[lab] for lab in labels])
    between = np.nanstd(stack, axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])
    grand = np.nanmean(stack, axis=0)
    return {
        "ok": True,
        "motion": motion,
        "unit": "subject",
        "ampnorm": ampnorm,
        "subjects": labels,
        "n_reps": [int(g.shape[0]) for g in groups],
        "means": {lab: means[lab].tolist() for lab in labels},
        "sd": {lab: sds[lab].tolist() for lab in labels},
        "grand_mean": grand.tolist(),
        "between_sd": between.tolist(),
        "note": "subject mean cycle is the unit; between-subject SD of n means (no cycle-level ANOVA)",
    }


def plot_spm(results: list[dict[str, Any]], out: Path, *, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    n = max(len(results), 1)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    xi = np.linspace(0, 100, CYCLE_N)
    for axrow, res in zip(axes, results):
        ax0, ax1 = axrow
        means = res.get("means") or {}
        sds = res.get("sd") or {}
        for sid, mu in means.items():
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sds.get(sid, np.zeros_like(mu)), dtype=float)
            ax0.plot(xi[: len(mu)], mu, label=sid)
            ax0.fill_between(xi[: len(mu)], mu - sd, mu + sd, alpha=0.15)
        ax0.set_title(res.get("motion"))
        ax0.set_xlabel("% cycle")
        ax0.set_ylabel(ylabel)
        ax0.legend(fontsize=8, loc="lower left")
        draw_cluster_bars(ax0, res)
        z = np.asarray(res.get("z") or [], dtype=float)
        if z.size:
            ax1.plot(np.linspace(0, 100, z.size), z, color="#c0392b")
            zstar = res.get("zstar")
            if zstar is not None and np.isfinite(zstar):
                ax1.axhline(zstar, color="#7f8c8d", ls="--")
                ax1.axhline(-zstar, color="#7f8c8d", ls="--")
            ax1.set_ylabel("SPM F")
        else:
            ax1.text(0.5, 0.5, res.get("reason") or "no SPM", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("% cycle")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_spm_paper(results: list[dict[str, Any]], out: Path, *, ylabel: str, title: str) -> None:
    """Paper-style: mean cycles with gray SPM-significance bars (no F panel)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), constrained_layout=True)
    xi = np.linspace(0, 100, CYCLE_N)
    fig.suptitle(title, fontsize=11)
    for ax, res in zip(axes.ravel(), results):
        means = res.get("means") or {}
        sds = res.get("sd") or {}
        for sid, mu in means.items():
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sds.get(sid, np.zeros_like(mu)), dtype=float)
            ax.plot(xi[: len(mu)], mu, label=sid, lw=1.4)
            ax.fill_between(xi[: len(mu)], mu - sd, mu + sd, alpha=0.12)
        ax.set_title(res.get("motion"))
        ax.set_xlabel("% cycle")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="lower left")
        draw_cluster_bars(ax, res)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_spm_paper_row(results: list[dict[str, Any]], out: Path, *, ylabel: str, title: str) -> None:
    """Paper 1×4: mean cycles with gray SPM bars along the top."""
    import matplotlib.pyplot as plt

    n = max(len(results), 1)
    fig, axes = plt.subplots(1, n, figsize=(3.15 * n, 2.85), sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    xi = np.linspace(0, 100, CYCLE_N)
    fig.suptitle(title, fontsize=11)
    for ax, res in zip(axes, results):
        means = res.get("means") or {}
        sds = res.get("sd") or {}
        for sid, mu in means.items():
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sds.get(sid, np.zeros_like(mu)), dtype=float)
            ax.plot(xi[: len(mu)], mu, label=sid, lw=1.4)
            ax.fill_between(xi[: len(mu)], mu - sd, mu + sd, alpha=0.12)
        ax.set_title(res.get("motion"))
        ax.set_xlabel("% cycle")
        ax.legend(fontsize=6, loc="lower left")
        draw_cluster_bars(ax, res)
    axes[0].set_ylabel(ylabel)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_subject_unit(results: list[dict[str, Any]], out: Path, *, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    n = max(len(results), 1)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.2 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    xi = np.linspace(0, 100, CYCLE_N)
    fig.suptitle(title, fontsize=11)
    for axrow, res in zip(axes, results):
        ax0, ax1 = axrow
        means = res.get("means") or {}
        sds = res.get("sd") or {}
        for sid, mu in means.items():
            mu = np.asarray(mu, dtype=float)
            sd = np.asarray(sds.get(sid, np.zeros_like(mu)), dtype=float)
            ax0.plot(xi[: len(mu)], mu, label=sid)
            ax0.fill_between(xi[: len(mu)], mu - sd, mu + sd, alpha=0.15)
        ax0.set_title(res.get("motion"))
        ax0.set_xlabel("% cycle")
        ax0.set_ylabel(ylabel)
        ax0.legend(fontsize=8, loc="lower left")
        draw_cluster_bars(ax0, res)
        bsd = np.asarray(res.get("between_sd") or [], dtype=float)
        if bsd.size:
            ax1.plot(xi[: len(bsd)], bsd, color="#1a5276")
            ax1.set_ylabel("between-subject SD")
        else:
            ax1.text(0.5, 0.5, res.get("reason") or "n<2", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xlabel("% cycle")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def kinematic_cycle_spans(rec: TakeInventory) -> dict[str, list[tuple[int, int]]]:
    """Peak-to-peak cycle windows (t_master_us) per motion phase, as in 5_4_2_motion_trajectory_analysis."""
    packed = load_kinematic_trace(rec.path)
    if packed is None:
        return {}
    t, xyz, _source = packed
    with SessionData(rec.path) as sd:
        phases = list(sd.phases)

    def mask(ph: Any) -> np.ndarray:
        return (t >= int(ph.t_start_us)) & (t <= int(ph.t_end_us))

    rest = next((p for p in phases if p.name == "rest"), None)
    amp = opening_1d(xyz, xyz[mask(rest)] if rest is not None else None)
    fs = sample_rate_hz(t, 60.0)
    out: dict[str, list[tuple[int, int]]] = {}
    for p in phases:
        if p.name not in TRAJ_MOTIONS:
            continue
        m = mask(p)
        if int(m.sum()) < 8:
            continue
        t_ph = t[m]
        segs = peak_to_peak_cycles(amp[m], fs, t_s=t_ph.astype(np.float64) * 1e-6)
        spans = [(int(t_ph[a]), int(t_ph[b])) for a, b in segs
                 if 0 <= a < b < len(t_ph) and int(t_ph[b]) > int(t_ph[a])]
        if spans:
            out[p.name] = spans
    return out


def emg_pct_by_phase(rec: TakeInventory, mvc_rec: TakeInventory | None
                     ) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
    """Motion phase -> ``(t_us float, %MVC (N, C) NaN where masked, channel names)``.

    Empty without an MVC take or a rest phase (no reference, no artifact rules).
    """
    if mvc_rec is None:
        return {}
    with ExitStack() as stack:
        sd = stack.enter_context(SessionData(rec.path))
        sdv = stack.enter_context(SessionData(mvc_rec.path))
        if sd.phase("rest") is None:
            return {}
        prep = EmgPrep(sd, sdv, EmgParams())
        out = {}
        for motion in TRAJ_MOTIONS:
            if sd.phase(motion) is None:
                continue
            pe = prep.phase(motion)
            out[motion] = (pe.t_us.astype(np.float64), pe.norm, list(pe.names))
    return out


def collect_emg_groups(
    recs: list[tuple[TakeInventory, TakeInventory | None]],
    motion: str,
    channels: tuple[str, ...],
    span_cache: dict[str, dict[str, list[tuple[int, int]]]] | None = None,
    emg_cache: dict[str, Any] | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    """%MVC envelopes time-normalised onto optical cycle windows, stacked per subject.

    A wave is the mean over the group's channels; waves that are not finite
    everywhere (masked edges) cannot enter SPM and are counted in
    ``emg_cache[("_incomplete", motion, channels)]`` instead.
    """
    by_sid: dict[str, list[np.ndarray]] = {}
    cache = span_cache if span_cache is not None else {}
    ecache = emg_cache if emg_cache is not None else {}
    for rec, mvc_rec in recs:
        sid = rec.subject_id or ""
        if not sid:
            continue
        if rec.name not in ecache:
            try:
                ecache[rec.name] = emg_pct_by_phase(rec, mvc_rec)
            except Exception as exc:  # noqa: BLE001
                print(f"EMG skip {rec.name}: {exc}", file=sys.stderr)
                ecache[rec.name] = {}
        block = ecache[rec.name].get(motion)
        if block is None:
            continue
        t_emg, pct, names = block
        cols = [names.index(c) for c in channels if c in names]
        if not cols:
            continue
        if rec.name not in cache:
            cache[rec.name] = kinematic_cycle_spans(rec)
        for t0, t1 in cache[rec.name].get(motion) or []:
            m = (t_emg >= t0) & (t_emg <= t1)
            waves = []
            for j in cols:
                y = pct[m, j]
                if int(np.isfinite(y).sum()) < 4:
                    continue
                waves.append(time_normalize(y, CYCLE_N))
            if not waves:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns
                wave = np.nanmean(np.vstack(waves), axis=0)
            if not np.isfinite(wave).all():
                ecache.setdefault(("_incomplete", motion, tuple(channels)), set()).add(
                    (rec.name, t0, t1))
                continue
            by_sid.setdefault(sid, []).append(wave)
    labels, groups = [], []
    for sid in sorted(by_sid):
        labels.append(sid)
        groups.append(np.vstack(by_sid[sid]))
    return labels, groups


def emg_subject_unit(
    recs: list[tuple[TakeInventory, TakeInventory | None]],
    motion: str,
    channels: tuple[str, ...],
    group_name: str,
    span_cache: dict[str, dict[str, list[tuple[int, int]]]] | None = None,
    emg_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    labels, groups = collect_emg_groups(
        recs, motion, channels, span_cache=span_cache, emg_cache=emg_cache
    )
    if len(groups) < 2:
        return {
            "ok": False,
            "reason": "need ≥2 subjects with EMG cycles",
            "subjects": labels,
            "motion": motion,
            "group": group_name,
        }
    means = {lab: np.nanmean(g, axis=0) for lab, g in zip(labels, groups)}
    sds = {lab: np.nanstd(g, axis=0) for lab, g in zip(labels, groups)}
    stack = np.vstack([means[lab] for lab in labels])
    between = np.nanstd(stack, axis=0, ddof=1)
    spm_pack = _spm_anova1(
        groups,
        labels,
        motion,
        0.05,
        "cycle-as-rep ANOVA on %MVC waves (bars only; means remain subject-unit)",
    )
    return {
        "ok": True,
        "motion": motion,
        "group": group_name,
        "unit": "subject",
        "channels": list(channels),
        "subjects": labels,
        "n_reps": [int(g.shape[0]) for g in groups],
        "means": {lab: means[lab].tolist() for lab in labels},
        "sd": {lab: sds[lab].tolist() for lab in labels},
        "grand_mean": np.nanmean(stack, axis=0).tolist(),
        "between_sd": between.tolist(),
        "clusters": spm_pack.get("clusters") or [],
        "z": spm_pack.get("z"),
        "zstar": spm_pack.get("zstar"),
        "note": "optical cycle windows; mean %MVC of group channels; subject-mean unit; gray bars from cycle-level SPM",
    }


def plot_emg_subject_unit(closers: list[dict[str, Any]], openers: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    n = max(len(closers), 1)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 3.1 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    xi = np.linspace(0, 100, CYCLE_N)
    fig.suptitle("EMG %MVC  —  subject-mean cycles (optical windows)", fontsize=11)
    for axrow, rc, ro in zip(axes, closers, openers):
        for ax, res, title in ((axrow[0], rc, "closers"), (axrow[1], ro, "openers")):
            means = res.get("means") or {}
            sds = res.get("sd") or {}
            for sid, mu in means.items():
                mu = np.asarray(mu, dtype=float)
                sd = np.asarray(sds.get(sid, np.zeros_like(mu)), dtype=float)
                ax.plot(xi[: len(mu)], mu, label=sid)
                ax.fill_between(xi[: len(mu)], mu - sd, mu + sd, alpha=0.15)
            if not means:
                ax.text(0.5, 0.5, res.get("reason") or "no data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{res.get('motion')}  {title}")
            ax.set_xlabel("% cycle")
            ax.set_ylabel("%MVC")
            ax.legend(fontsize=7, loc="lower left")
            draw_cluster_bars(ax, res)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_emg_between_sd(closers: list[dict[str, Any]], openers: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    n = max(len(closers), 1)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 3.1 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])
    xi = np.linspace(0, 100, CYCLE_N)
    fig.suptitle("EMG %MVC  —  between-subject SD of subject means", fontsize=11)
    for axrow, rc, ro in zip(axes, closers, openers):
        for ax, res, title in ((axrow[0], rc, "closers"), (axrow[1], ro, "openers")):
            bsd = np.asarray(res.get("between_sd") or [], dtype=float)
            if bsd.size:
                ax.plot(xi[: len(bsd)], bsd, color="#1a5276")
            else:
                ax.text(0.5, 0.5, res.get("reason") or "n<2", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{res.get('motion')}  {title}")
            ax.set_xlabel("% cycle")
            ax.set_ylabel("between-subject SD (%MVC)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _doc_pid(doc: dict[str, Any]) -> str | None:
    if doc.get("pid"):
        return str(doc["pid"]).lower()
    info = parse_session_name(str(doc.get("session") or ""))
    return info["pid"] if info else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--cycles-dir", type=Path, default=None,
                    help="5_4_2_motion_trajectory_analysis derived/ folder (default: <results root>/5_4_2_.../results/derived)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=None,
                    help=f"default: $BIOMECH_RESULTS_DIR or ~/experiments, then /{STUDY}/results")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    cycles_dir = (args.cycles_dir or default_cycles_dir()).expanduser()
    docs = load_cycle_docs(cycles_dir)
    pids = subject_pids(args.subject)
    if pids:
        docs = [d for d in docs if _doc_pid(d) in set(pids)]

    data_dir = resolve_data_dir(args.data_dir)
    takes = scan(data_dir, pids=pids)
    first_mvc: dict[str, TakeInventory] = {}
    for t in takes:
        if t.take_kind == "mvc" and t.has_emg:
            first_mvc.setdefault(str(t.pid), t)
    doc_names = {d.get("session") for d in docs}
    recs = [(t, first_mvc.get(str(t.pid))) for t in takes
            if t.take_kind == "motion" and t.has_emg and t.has_optical and t.name in doc_names]

    cycle_mm = [run_spm_reps(docs, m, args.alpha, ampnorm=False) for m in TRAJ_MOTIONS]
    cycle_amp = [run_spm_reps(docs, m, args.alpha, ampnorm=True) for m in TRAJ_MOTIONS]
    subj_mm = [subject_unit_summary(docs, m, ampnorm=False) for m in TRAJ_MOTIONS]
    subj_amp = [subject_unit_summary(docs, m, ampnorm=True) for m in TRAJ_MOTIONS]
    span_cache: dict[str, dict[str, list[tuple[int, int]]]] = {}
    emg_cache: dict[Any, Any] = {}
    emg_kw = dict(span_cache=span_cache, emg_cache=emg_cache)
    emg_c_cycle = [run_spm_emg_reps(recs, m, CLOSING_MUSCLES, "closers", args.alpha,
                                    ampnorm=False, **emg_kw) for m in TRAJ_MOTIONS]
    emg_o_cycle = [run_spm_emg_reps(recs, m, OPENING_MUSCLES, "openers", args.alpha,
                                    ampnorm=False, **emg_kw) for m in TRAJ_MOTIONS]
    emg_c_amp = [run_spm_emg_reps(recs, m, CLOSING_MUSCLES, "closers", args.alpha,
                                  ampnorm=True, **emg_kw) for m in TRAJ_MOTIONS]
    emg_o_amp = [run_spm_emg_reps(recs, m, OPENING_MUSCLES, "openers", args.alpha,
                                  ampnorm=True, **emg_kw) for m in TRAJ_MOTIONS]
    emg_c = [emg_subject_unit(recs, m, CLOSING_MUSCLES, "closers", **emg_kw)
             for m in TRAJ_MOTIONS]
    emg_o = [emg_subject_unit(recs, m, OPENING_MUSCLES, "openers", **emg_kw)
             for m in TRAJ_MOTIONS]
    incomplete = {f"{k[1]}:{'closers' if k[2] == tuple(CLOSING_MUSCLES) else 'openers'}": len(v)
                  for k, v in emg_cache.items() if isinstance(k, tuple) and k[0] == "_incomplete"}

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_docs": len(docs),
        "cycle_reps_mm": cycle_mm,
        "cycle_reps_ampnorm": cycle_amp,
        "subject_unit_mm": subj_mm,
        "subject_unit_ampnorm": subj_amp,
        "emg_mvc_closers": emg_c,
        "emg_mvc_openers": emg_o,
        "emg_cycle_closers": emg_c_cycle,
        "emg_cycle_openers": emg_o_cycle,
        "emg_cycle_closers_ampnorm": emg_c_amp,
        "emg_cycle_openers_ampnorm": emg_o_amp,
        # EMG cycle waves with masked (non-finite) samples left after warping cannot
        # enter SPM; counted here per motion:group instead of being dropped silently.
        "emg_incomplete_waves": incomplete,
    }
    write_json(out / "spm_summary.json", summary)
    write_json(derived / "spm_results.json", summary)
    write_provenance(derived, STUDY, "eval_spm.py", [r for r, _ in recs],
                     data_dir=data_dir,
                     extra={"cycles_dir": str(cycles_dir), "cycle_docs": sorted(doc_names),
                            "mvc_sessions": {r.name: (m.name if m else None) for r, m in recs}})
    if not args.no_plot and docs:
        import matplotlib

        matplotlib.use("Agg")
        plot_spm(cycle_mm, out / "spm_fields.png", ylabel="opening (mm)")
        plot_spm(cycle_amp, out / "spm_fields_ampnorm.png", ylabel="opening (0–1)")
        plot_spm_paper(
            cycle_mm,
            out / "spm_bars.png",
            ylabel="opening (mm)",
            title="SPM clusters (gray) — F > F*, α=0.05  (cycle-as-rep)",
        )
        plot_spm_paper(
            cycle_amp,
            out / "spm_bars_ampnorm.png",
            ylabel="opening (0–1)",
            title="SPM clusters (gray) — amplitude-normalised, F > F*, α=0.05",
        )
        plot_spm_paper_row(
            cycle_amp,
            out / "spm_bars_ampnorm_row.png",
            ylabel="opening (0–1)",
            title="SPM clusters (gray) — amplitude-normalised kinematics, F > F*, α=0.05",
        )
        plot_spm(emg_c_cycle, out / "spm_emg_fields_closers.png", ylabel="%MVC")
        plot_spm(emg_o_cycle, out / "spm_emg_fields_openers.png", ylabel="%MVC")
        plot_spm(emg_c_amp, out / "spm_emg_fields_closers_ampnorm.png", ylabel="%MVC (0–1)")
        plot_spm(emg_o_amp, out / "spm_emg_fields_openers_ampnorm.png", ylabel="%MVC (0–1)")
        plot_spm_paper(
            emg_c_cycle,
            out / "spm_emg_bars_closers.png",
            ylabel="%MVC",
            title="EMG closers — SPM clusters (gray), cycle-as-rep, F > F*",
        )
        plot_spm_paper(
            emg_o_cycle,
            out / "spm_emg_bars_openers.png",
            ylabel="%MVC",
            title="EMG openers — SPM clusters (gray), cycle-as-rep, F > F*",
        )
        plot_spm_paper(
            emg_c_amp,
            out / "spm_emg_bars_closers_ampnorm.png",
            ylabel="%MVC (0–1)",
            title="EMG closers — amplitude-normalised, SPM clusters (gray)",
        )
        plot_spm_paper(
            emg_o_amp,
            out / "spm_emg_bars_openers_ampnorm.png",
            ylabel="%MVC (0–1)",
            title="EMG openers — amplitude-normalised, SPM clusters (gray)",
        )
        plot_spm_paper_row(
            emg_c_amp,
            out / "spm_emg_bars_closers_ampnorm_row.png",
            ylabel="%MVC (0–1)",
            title="EMG closers — amplitude-normalised, F > F* (cycle-as-rep)",
        )
        for dst, src in ((subj_mm, cycle_mm), (subj_amp, cycle_amp)):
            for a, b in zip(dst, src):
                a["clusters"] = b.get("clusters")
                a["z"] = b.get("z")
                a["zstar"] = b.get("zstar")
        plot_subject_unit(
            subj_mm,
            out / "spm_subject_unit.png",
            ylabel="opening (mm)",
            title="Kinematics — subject as unit (mean ± within-subject SD)",
        )
        plot_subject_unit(
            subj_amp,
            out / "spm_subject_unit_ampnorm.png",
            ylabel="opening (0–1)",
            title="Kinematics — amplitude-normalised, subject as unit",
        )
        plot_emg_subject_unit(emg_c, emg_o, out / "spm_emg_mvc.png")
        plot_emg_between_sd(emg_c, emg_o, out / "spm_emg_mvc_between_sd.png")
    print(f"cycle_docs={len(docs)} emg_sessions={len(recs)}")
    for r in cycle_mm:
        print(f"  cycle-mm {r.get('motion')}: ok={r.get('ok')} {r.get('reason', '')}")
    for r in emg_c:
        print(f"  emg-closers {r.get('motion')}: ok={r.get('ok')} n={r.get('n_reps')} {r.get('reason', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
