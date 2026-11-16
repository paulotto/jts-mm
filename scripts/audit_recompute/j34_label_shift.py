# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J34.4: why the label-shift null of the C3b permutation test lies above 1/7
(descriptive, post hoc; definitions fixed in J34 before this ran).

A predictor independent of the test labels scores an expected balanced accuracy of exactly
1/7, so a null mean above it means the shifted labels still carry the true classes. The 100
label shifts of ``ml_eval``'s permutation test are redrawn exactly (study seed,
``ml.protocols.evaluate.shift_labels`` on the windows pooled as ``ml_eval`` pools them).

Reproduction first: the first two permutations are refitted (LDA, EMG + IMU, LOSO, the stage's
``inner_folds`` and ``max_train``, seed + k) and must equal the stored
``permutation_null[0:2]``; the script stops otherwise.

Per permutation: the cohort mean over participants of the balanced accuracy of the shifted
labels against the true labels (``align_ba``), i.e. the score of a decoder that recovered the true
class, next to the stored null value. J34.4b (added after that result, before computing): the
cohort mean balanced accuracy of a position decoder (``pos_ba``) that predicts a participant's
shifted label from the other participants' shifted labels at the same relative position in the
session (100 bins), and the same decoder on the unshifted labels as reference. J34.4c: per
participant the expected balanced accuracy over the restricted shift range ([n/4, 3n/4)) of the
best class per window (``ceiling_ba``; with shifts over the whole sequence every decoder averages
exactly 1/7), with the best constant class as sanity check (exactly 1/7).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j34_label_shift.py

Output: ``<adults store>/audit/j34_label_shift_<snap>.parquet`` (one row per permutation),
``j34_shift_ceiling_<snap>.parquet`` (one row per participant) and ``.json`` (settings,
reproduction, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
N_REPRO = 2
N_BINS = 100            # J34.4b: relative-position bins of the position decoder


def position_ba(y: np.ndarray, idx: dict, pos: dict, ba) -> float:
    """J34.4b: cohort mean balanced accuracy of the position decoder. For each participant the
    class predicted in a relative-position bin is the most frequent label (>= 0) of the other
    participants in that bin (ties: the smaller class id; empty bin: no prediction, -1)."""
    k = int(y.max()) + 1
    counts = {q: np.zeros((N_BINS, k), np.int64) for q in idx}
    for q, i in idx.items():
        ok = y[i] >= 0
        np.add.at(counts[q], (pos[q][ok], y[i][ok]), 1)
    total = sum(counts.values())
    per = []
    for q, i in idx.items():
        c = total - counts[q]
        pred_bin = np.where(c.sum(1) > 0, c.argmax(1), -1)     # argmax: first (smaller) id on ties
        ok = y[i] >= 0
        per.append(ba(y[i][ok], pred_bin[pos[q][ok]]))
    return float(np.mean(per))


def shift_ceiling(y: np.ndarray, n_classes: int) -> tuple[float, float]:
    """J34.4c: one participant's expected balanced accuracy over the shifts of ``shift_labels``
    (uniform in [n/4, 3n/4)) for the best class per window, and for the best constant class
    (sanity check: exactly 1 / n_classes). Windows labelled -1 count in n but in no class."""
    n = len(y)
    a, b = n // 4, 3 * n // 4                         # rng.integers(a, b): b exclusive
    Y = np.zeros((n, n_classes))
    ok = y >= 0
    Y[np.flatnonzero(ok), y[ok]] = 1
    cum = np.vstack([np.zeros((1, n_classes)), np.vstack([Y, Y, Y]).cumsum(0)])
    hi = (np.arange(n) - a) % n + n                   # labels y[t - s], s in [a, b)
    q = (cum[hi + 1] - cum[hi + 1 - (b - a)]) / (b - a)
    nk = Y.sum(0)
    score = np.divide(q, nk, out=np.zeros_like(q), where=nk > 0)
    present = int((nk > 0).sum())
    return float(score.max(1).sum() / present), float(score.sum(0).max() / present)


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    a = ap.parse_args()
    from biomech.io.context import DataContext
    from ml.biomech_stages import _pool
    from ml.evaluation.metrics import balanced_accuracy
    from ml.protocols import evaluate as ev

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        shim = SimpleNamespace(participants=[SimpleNamespace(pid=q) for q in dc.pids()],
                               upstream=lambda s, t: dc.table(s, t),
                               upstream_arrays=lambda s, q: dc.arrays(s, q))
        ds = _pool(shim)
        stored = np.asarray(dc.arrays("ml_eval", "_cohort")["permutation_null"], float)
        p = dc.config.params("ml_eval")
    finally:
        dc.close()
    seed = int(cfg.seed)
    rng = np.random.default_rng(seed)                 # as ev.permutation
    perms = [ev.shift_labels(ds, rng) for _ in range(int(p["n_perm"]))]
    if len(perms) != stored.size:
        raise SystemExit(f"{len(perms)} permutations, {stored.size} stored")

    repro = []
    for kk in range(N_REPRO):
        rs = ev.run(ds, "emg_imu", "lda", "loso_subject", seed=seed + kk,
                    inner=p["inner_folds"], max_train=p["max_train"], y=perms[kk])
        v = float(np.mean([balanced_accuracy(perms[kk][r.fold.test], r.y_pred) for r in rs]))
        repro.append({"k": kk, "refit": v, "stored": float(stored[kk]),
                      "equal": bool(np.isclose(v, stored[kk], rtol=1e-12, atol=1e-12))})
    if not all(r["equal"] for r in repro):
        raise SystemExit(f"reproduction failed: {json.dumps(repro, indent=1)}")

    pids = np.unique(ds.pid)
    idx = {q: np.flatnonzero(ds.pid == q) for q in pids}
    pos = {q: np.minimum((np.arange(len(i)) * N_BINS) // len(i), N_BINS - 1)
           for q, i in idx.items()}
    rows = []
    for kk, yp in enumerate(perms):
        per = []
        for q in pids:
            i = idx[q]
            ok = (ds.y[i] >= 0) & (yp[i] >= 0)                  # untracked windows (-1) left out
            per.append(balanced_accuracy(ds.y[i][ok], yp[i][ok]))
        rows.append({"k": kk, "align_ba": float(np.mean(per)),
                     "pos_ba": position_ba(yp, idx, pos, balanced_accuracy),
                     "null": float(stored[kk]), "n_participants": int(len(pids))})
    out = pd.DataFrame(rows)
    pos_true = position_ba(ds.y, idx, pos, balanced_accuracy)
    k = int(ds.y.max()) + 1
    ceil = pd.DataFrame([{"pid": q, "n_windows": int(len(i)),
                          **dict(zip(("ceiling_ba", "constant_ba"), shift_ceiling(ds.y[i], k)))}
                         for q, i in idx.items()])
    present = [int(np.unique(ds.y[i][ds.y[i] >= 0]).size) for i in idx.values()]
    if not np.allclose(ceil.constant_ba, 1 / np.asarray(present), rtol=1e-12):
        raise SystemExit(f"ceiling sanity check failed: {ceil.to_string()}")
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"j34_label_shift_{a.snapshot}.parquet"
    out.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    fc = out_dir / f"j34_shift_ceiling_{a.snapshot}.parquet"
    ceil.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(fc)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "variant": VARIANT, "seed": seed,
            "n_perm": int(len(perms)), "inner_folds": p["inner_folds"],
            "max_train": p["max_train"], "reproduction": repro, "n_bins": N_BINS,
            "pos_ba_unshifted": pos_true,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            "ceiling_sha256": hashlib.sha256(fc.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(repro, indent=1))
    print(out.describe().to_string())
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
