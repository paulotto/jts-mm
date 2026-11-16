#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""EMG SNR and MVC-normalised amplitude.

Signals come from ``biomech.emg.prepare.EmgPrep`` (the chain the ICORR
pipeline uses): mains fitted at the measured grid frequency of the take ->
20-450 Hz zero-phase band-pass -> 150 ms RMS envelope. Artifacts (rail,
lead-off, envelope > 3x MVC, movement low-frequency) are *masked*, never
zeroed; masked samples are left out of every statistic below.

* SNR = std(cleaned activation) / std(cleaned rest) over samples free of the
  physical artifact rules (rail, lead-off, movement low-frequency). The
  ``envelope`` rule (> 3x the MVC reference) is left out here: it depends on
  the MVC, and with a weak MVC it would mask the very activation SNR measures.
* MVC = 95th percentile of the masked envelope in the matching MVC phase
  (digastric <- ``mvc_opening``, masseter/temporalis <- ``mvc_closing``) of the
  participant's first MVC take.
* %MVC = envelope / MVC. When the MVC is below the channel's maximum in the
  motion take, it is still the denominator; the row carries
  ``mvc_below_recording_max = True`` and ``recording_max_uv`` instead of the
  former silent substitution.
* ``pct_mvc_p50/p95`` use the full EmgPrep mask (as the dataset-study pipeline);
  ``pct_mvc_*_physmask`` is the sensitivity value with the physical rules only,
  ``mask_frac`` (dilated, all rules) and ``mask_frac_<rule>`` (undilated) say
  how much was removed and by which rule.

Usage
-----
  python eval_emg_properties.py
  python eval_emg_properties.py --data-dir ~/jtsmm_data --subject s03
  python eval_emg_properties.py --legacy path/to/commercial_raw.h5   # optional
"""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from biomech.emg.artifacts import RULES, masked_envelope
from biomech.emg.prepare import EmgParams, EmgPrep
from biomech.io.inventory import MOTION_PHASES, TakeInventory, scan, subject_pids
from biomech.io.session import CLOSERS, MUSCLES, OPENERS, SessionData
from biomech.paths import resolve_data_dir, results_dir
from biomech.results.provenance import write_csv, write_json, write_provenance

STUDY = "5_4_1_emg_properties"
PHYSICAL_RULES = ("rail", "leadoff")             # D14: lowfreq is a QC flag, not an exclusion
MUSCLE_CHANNELS = MUSCLES
CLOSING_MUSCLES = CLOSERS
OPENING_MUSCLES = OPENERS


def snr_std(signal: np.ndarray, noise: np.ndarray) -> float:
    """std(signal) / std(noise) over finite samples; NaN with < 8 samples or a flat noise floor."""
    s = np.asarray(signal, dtype=np.float64)
    n = np.asarray(noise, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = n[np.isfinite(n)]
    if s.size < 8 or n.size < 8:
        return float("nan")
    den = float(np.std(n))
    return float(np.std(s) / den) if den >= 1e-12 else float("nan")


def pair_mvc(motion: list[TakeInventory], mvc: list[TakeInventory]
             ) -> list[tuple[TakeInventory, TakeInventory | None]]:
    """Every motion take with its participant's first MVC take (an01 §2), or None."""
    first: dict[str, TakeInventory] = {}
    for t in mvc:                                  # scan() is oldest-first per pid
        first.setdefault(str(t.pid), t)
    return [(m, first.get(str(m.pid))) for m in motion]


def eval_motion(rec: TakeInventory, mvc_rec: TakeInventory | None,
                params: EmgParams = EmgParams()) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Rows per (channel, motion phase) and the channel MVC references (µV)."""
    from jtsmm.emg.mains import mains_share

    with ExitStack() as stack:
        sd = stack.enter_context(SessionData(rec.path))
        sd_mvc = stack.enter_context(SessionData(mvc_rec.path)) if mvc_rec else None
        if sd.phase("rest") is None:
            raise LookupError("no rest phase")
        prep = EmgPrep(sd, sd_mvc, params)
        names = list(prep.names)
        ref = np.asarray(prep.ref_uv("p95"), dtype=np.float64)
        mvc_mask = np.asarray(prep.refs["mask_frac"], dtype=np.float64)
        line_hz = float(prep.line_hz(sd))
        fs = sd.emg_fs
        rest = prep.phase("rest")
        pes = {m: prep.phase(m) for m in MOTION_PHASES if sd.phase(m) is not None}

    dil = max(int(round(params.mask.dilate_s * fs)), 0)
    phys = {k: pe.masks.rules(*PHYSICAL_RULES, dilate=dil) for k, pe in pes.items()}
    rest_phys = rest.masks.rules(*PHYSICAL_RULES, dilate=dil)
    env_phys = {k: masked_envelope(pe.cleaned, phys[k], fs, params.envelope_s)
                for k, pe in pes.items()}
    # Channel maximum in the motion take, on the physically masked envelope (the
    # MVC-relative envelope rule would cap it at 3x the MVC it is compared with).
    rec_max = np.full(len(names), -np.inf)
    for env in env_phys.values():
        rec_max = np.fmax(rec_max, np.max(np.where(np.isfinite(env), env, -np.inf), axis=0))
    rec_max[~np.isfinite(rec_max)] = np.nan
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        rest_sig = rest.cleaned[~rest_phys[:, i], i]
        mvc = float(ref[i])
        has_mvc = np.isfinite(mvc) and mvc > 0
        below = bool(has_mvc and np.isfinite(rec_max[i]) and mvc < rec_max[i])
        if below:
            print(f"{rec.name} {name}: MVC {mvc:.1f} µV < recording max {rec_max[i]:.1f} µV "
                  "(flagged, MVC kept as denominator)")
        for motion, pe in pes.items():
            sig = pe.cleaned[~phys[motion][:, i], i]
            pct = pe.norm[:, i] if has_mvc else np.full(pe.norm.shape[0], np.nan)
            fin = np.isfinite(pct)
            pct_p = (100.0 * env_phys[motion][:, i] / mvc if has_mvc
                     else np.full(pe.norm.shape[0], np.nan))
            fin_p = np.isfinite(pct_p)
            rows.append(
                {
                    "session": rec.name,
                    "pid": rec.pid,
                    "subject_id": rec.subject_id,
                    "channel": name,
                    "motion": motion,
                    "snr": snr_std(sig, rest_sig),
                    "rest_std_uv": float(np.std(rest_sig)) if rest_sig.size else float("nan"),
                    "act_std_uv": float(np.std(sig)) if sig.size else float("nan"),
                    "mains_share_raw": float(mains_share(pe.raw[:, i], fs, line_hz)),
                    "mains_share_clean": float(mains_share(pe.cleaned[:, i], fs, line_hz)),
                    "line_hz": line_hz,
                    "mvc_uv": mvc if has_mvc else float("nan"),
                    "mvc_source": "mvc" if has_mvc else "missing",
                    "mvc_session": mvc_rec.name if mvc_rec else None,
                    "mvc_mask_frac": float(mvc_mask[i]),
                    "recording_max_uv": float(rec_max[i]),
                    "mvc_below_recording_max": below,
                    "pct_mvc_p50": float(np.median(pct[fin])) if fin.any() else float("nan"),
                    "pct_mvc_p95": float(np.percentile(pct[fin], 95)) if fin.any()
                    else float("nan"),
                    "pct_mvc_p50_physmask": float(np.median(pct_p[fin_p])) if fin_p.any()
                    else float("nan"),
                    "pct_mvc_p95_physmask": float(np.percentile(pct_p[fin_p], 95))
                    if fin_p.any() else float("nan"),
                    "mask_frac": float(pe.mask[:, i].mean()) if pe.mask.size else float("nan"),
                    "mask_frac_physical": float(phys[motion][:, i].mean())
                    if pe.mask.size else float("nan"),
                    # per rule, undilated (biomech.emg.artifacts.RULES)
                    **{f"mask_frac_{r}": float(pe.masks.hits[r][:, i].mean())
                       if r in pe.masks.hits and pe.mask.size else 0.0 for r in RULES},
                    "n_rest": int(rest_sig.size),
                    "n_act": int(sig.size),
                }
            )
    refs = {n: float(ref[i]) for i, n in enumerate(names)}
    return rows, refs


def plot_snr(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    subjects = sorted({r["subject_id"] for r in rows if r["subject_id"]})
    muscles = [c for c in MUSCLE_CHANNELS]
    # mean SNR over motions that exist
    mat = np.full((len(subjects), len(muscles)), np.nan)
    for i, sid in enumerate(subjects):
        for j, ch in enumerate(muscles):
            vals = [r["snr"] for r in rows if r["subject_id"] == sid and r["channel"] == ch]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            if vals:
                mat[i, j] = float(np.median(vals))
    fig, ax = plt.subplots(figsize=(9, 3.5 + 0.4 * len(subjects)), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(muscles)))
    ax.set_xticklabels(muscles, rotation=30, ha="right")
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects)
    ax.set_title("EMG SNR (median over motions)  std(act)/std(rest)")
    fig.colorbar(im, ax=ax, label="SNR")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_mvc_pooled(rows: list[dict[str, Any]], out: Path) -> None:
    """Appendix: all channels in one box — not the primary %MVC figure."""
    import matplotlib.pyplot as plt

    motions = sorted({r["motion"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    data = []
    labels = []
    for motion in motions:
        vals = [r["pct_mvc_p50"] for r in rows if r["motion"] == motion]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        data.append(vals)
        labels.append(motion)
    if any(data):
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(labels)
    ax.set_ylabel("%MVC (session median of envelope)")
    ax.set_title("Pooled across all channels (appendix — mixed closers/openers)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_mvc_uv_bars(mvc_json: dict[str, dict[str, float]], out: Path) -> None:
    import matplotlib.pyplot as plt

    subjects = sorted(mvc_json)
    muscles = list(MUSCLE_CHANNELS)
    x = np.arange(len(muscles), dtype=np.float64)
    width = 0.8 / max(len(subjects), 1)
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    for i, sid in enumerate(subjects):
        ys = [float(mvc_json[sid].get(ch, np.nan)) for ch in muscles]
        ax.bar(x + (i - (len(subjects) - 1) / 2) * width, ys, width=width, label=sid)
    ax.set_xticks(x)
    ax.set_xticklabels(muscles, rotation=25, ha="right")
    ax.set_ylabel("MVC envelope (µV), 95th percentile")
    ax.set_title("MVC scale per muscle (denominator of %MVC)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _box_by_motion(ax, rows: list[dict[str, Any]], channels: tuple[str, ...] | list[str], title: str) -> None:
    motions = sorted({r["motion"] for r in rows})
    allow = set(channels)
    data, labels = [], []
    for motion in motions:
        vals = [
            r["pct_mvc_p50"]
            for r in rows
            if r["motion"] == motion and r["channel"] in allow
        ]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        data.append(vals)
        labels.append(motion)
    if any(data):
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("%MVC")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def plot_mvc_by_group(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True, sharey=False)
    _box_by_motion(ax[0], rows, CLOSING_MUSCLES, "Closers (masseter + temporalis)")
    _box_by_motion(ax[1], rows, OPENING_MUSCLES, "Openers (digastric)")
    fig.suptitle("%MVC by movement  —  split by muscle group", fontsize=11)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_mvc_by_muscle(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    muscles = list(MUSCLE_CHANNELS)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True, sharey=False)
    for ax, ch in zip(axes.ravel(), muscles):
        _box_by_motion(ax, rows, (ch,), ch)
    fig.suptitle("%MVC by movement  —  one panel per muscle", fontsize=11)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_mvc_primary(
    rows: list[dict[str, Any]],
    mvc_json: dict[str, dict[str, float]],
    out: Path,
) -> None:
    """Primary figure: MVC µV scales beside closer/opener %MVC."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    subjects = sorted(mvc_json)
    muscles = list(MUSCLE_CHANNELS)
    fig = plt.figure(figsize=(11.4, 7.6), constrained_layout=True)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.15])
    ax_bar = fig.add_subplot(gs[0, :])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_o = fig.add_subplot(gs[1, 1])

    x = np.arange(len(muscles), dtype=np.float64)
    width = 0.8 / max(len(subjects), 1)
    for i, sid in enumerate(subjects):
        ys = [float(mvc_json[sid].get(ch, np.nan)) for ch in muscles]
        ax_bar.bar(x + (i - (len(subjects) - 1) / 2) * width, ys, width=width, label=sid)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(muscles, rotation=20, ha="right")
    ax_bar.set_ylabel("MVC (µV), 95th pct.")
    ax_bar.set_title("MVC scale (denominator)")
    ax_bar.set_yscale("log")
    ax_bar.legend(loc="upper right", fontsize=8)
    ax_bar.grid(True, axis="y", alpha=0.3)

    _box_by_motion(ax_c, rows, CLOSING_MUSCLES, "Closers %MVC")
    _box_by_motion(ax_o, rows, OPENING_MUSCLES, "Openers %MVC")
    fig.suptitle("MVC-normalised EMG  —  scales and %MVC by muscle group", fontsize=11)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def maybe_legacy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"skipped": True, "reason": "no --legacy path"}
    if not path.exists():
        return {"skipped": True, "reason": f"missing {path}"}
    # Commercial comparison needs raw EMG + rest/activity labels, which the old
    # feature files of the commercial system did not contain.
    if path.suffix == ".json" or path.name == "metadata.json":
        return {
            "skipped": True,
            "reason": "legacy metadata/features only — no raw EMG for SNR",
        }
    return {"skipped": True, "reason": "pass a session-like HDF5 with /emg and rest phases"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=None,
                    help=f"default: $BIOMECH_RESULTS_DIR or ~/experiments, then /{STUDY}/results")
    ap.add_argument("--legacy", type=Path, default=None, help="Optional commercial raw EMG HDF5")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    takes = [t for t in scan(data_dir, pids=subject_pids(args.subject)) if t.has_emg]
    motion = [t for t in takes if t.take_kind == "motion"]
    mvc = [t for t in takes if t.take_kind == "mvc"]
    pairs = pair_mvc(motion, mvc)

    rows: list[dict[str, Any]] = []
    mvc_json: dict[str, dict[str, float | None]] = {}
    for rec, mvc_rec in pairs:
        try:
            r, refs = eval_motion(rec, mvc_rec)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {rec.name}: {exc}", file=sys.stderr)
            continue
        rows.extend(r)
        sid = rec.subject_id or rec.name
        if mvc_rec is not None and sid not in mvc_json:
            mvc_json[sid] = {ch: (v if np.isfinite(v) else None) for ch, v in refs.items()}

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    write_csv(out / "emg_properties.csv", rows)
    write_json(derived / "emg_properties.json", rows)
    write_json(derived / "mvc_scales.json", mvc_json)
    legacy = maybe_legacy(args.legacy)
    n_flag = sum(bool(r["mvc_below_recording_max"]) for r in rows)
    write_json(out / "emg_summary.json", {"n_rows": len(rows), "mvc_subjects": list(mvc_json),
                                          "n_rows_mvc_below_recording_max": n_flag,
                                          "legacy": legacy})
    used = [t for pair in pairs for t in pair if t is not None]
    write_provenance(derived, STUDY, "eval_emg_properties.py", used, data_dir=data_dir,
                     extra={"legacy": legacy, "emg_params": "biomech.emg.prepare.EmgParams()"})
    if not args.no_plot and rows:
        import matplotlib

        matplotlib.use("Agg")
        mvc_plot = {sid: {ch: (np.nan if v is None else v) for ch, v in bag.items()}
                    for sid, bag in mvc_json.items()}
        plot_snr(rows, out / "snr_heatmap.png")
        plot_mvc_primary(rows, mvc_plot, out / "pct_mvc.png")
        plot_mvc_uv_bars(mvc_plot, out / "mvc_uv_bars.png")
        plot_mvc_by_group(rows, out / "pct_mvc_by_group.png")
        plot_mvc_by_muscle(rows, out / "pct_mvc_by_muscle.png")
        plot_mvc_pooled(rows, out / "pct_mvc_box_pooled.png")
    print(f"motion={len(motion)} mvc={len(mvc)} rows={len(rows)} "
          f"mvc_below_recording_max={n_flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
