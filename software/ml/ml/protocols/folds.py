# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Outer folds of the evaluation protocols.

=================  ==================================================================
``loso_subject``   leave one *subject* out: inter-subject generalization (headline)
``loso_session``   leave one *session* (take) out: inter-session; with several
                   subjects the other sessions of the test subject stay in training
``blocked``        within subject: each phase is cut into ``k`` contiguous time blocks
                   (classes are confined to their task, so blocks are cut per phase);
                   block b of every phase is tested together and training windows
                   closer than ``guard_s`` to a test window of the same take are
                   dropped, because 75 %-overlapping windows share samples
=================  ==================================================================

Training uses the structured phases only; ``free_motion`` windows of the held-out
unit are predicted and scored separately. ``groups`` gives the unit by which every
inner split (hyper-parameter search, SVM calibration, MLP/GRU early stopping) is
made: subjects, sessions or blocks, never single windows.

The old name ``loso`` is ambiguous (the legacy CLI meant sessions, the biomech
stage subjects) and is rejected with a message naming both.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ml.config.classes import FREE_PHASE
from ml.core.dataset import WindowDataset

PROTOCOLS = ("loso_subject", "loso_session", "blocked")
_ALIASES = {"leave_one_subject_out": "loso_subject", "losubo": "loso_subject",
            "leave_one_session_out": "loso_session"}


def normalize_protocol(name: str) -> str:
    key = str(name).strip().lower()
    if key == "loso":
        raise ValueError("protocol 'loso' is ambiguous; use 'loso_subject' (leave one subject "
                         "out) or 'loso_session' (leave one session out)")
    if key in ("groupkfold", "group_kfold"):
        raise ValueError("protocol 'groupkfold' (bout-grouped k-fold without a guard band) was "
                         "replaced by 'blocked' (per-phase time blocks with a guard)")
    key = _ALIASES.get(key, key)
    if key not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}; expected one of {PROTOCOLS}")
    return key


@dataclass
class Fold:
    unit: str                 # held-out subject / session / "pid|b<k>"
    pid: str                  # subject of the test rows ("" when several)
    train: np.ndarray         # row indices
    test: np.ndarray          # structured rows of the held-out unit
    free: np.ndarray          # free-motion rows of the held-out unit
    groups: np.ndarray        # inner group per training row


def make_folds(ds: WindowDataset, protocol: str, *, k: int = 5, guard_s: float = 0.35
               ) -> list[Fold]:
    protocol = normalize_protocol(protocol)
    structured = ds.phase != FREE_PHASE
    free = ~structured
    if protocol in ("loso_subject", "loso_session"):
        unit = ds.pid if protocol == "loso_subject" else ds.session
        units = np.unique(unit)
        if units.size < 2:
            raise ValueError(f"{protocol} needs at least 2 distinct "
                             f"{'subjects' if protocol == 'loso_subject' else 'sessions'}, "
                             f"found {units.tolist()}")
        if protocol == "loso_session" and np.unique(ds.pid).size > 1:
            multi = [p for p in np.unique(ds.pid) if np.unique(ds.session[ds.pid == p]).size > 1]
            if multi:
                warnings.warn(f"loso_session: subjects {multi} have several sessions, so their "
                              "other sessions stay in training; use loso_subject for "
                              "inter-subject claims", UserWarning, stacklevel=2)
        out = []
        for u in units:
            tr = np.flatnonzero((unit != u) & structured)
            te = np.flatnonzero((unit == u) & structured)
            fr = np.flatnonzero((unit == u) & free)
            pids = np.unique(ds.pid[unit == u])
            out.append(Fold(str(u), str(pids[0]) if pids.size == 1 else "", tr, te, fr,
                            unit[tr]))
        return out

    out = []
    for pid in np.unique(ds.pid):
        sel = np.flatnonzero((ds.pid == pid) & structured)
        blk = np.zeros(sel.size, dtype=int)
        for key in {(s, p) for s, p in zip(ds.session[sel], ds.phase[sel])}:
            m = np.flatnonzero((ds.session[sel] == key[0]) & (ds.phase[sel] == key[1]))
            rank = np.argsort(np.argsort(ds.t_end_us[sel][m], kind="stable"), kind="stable")
            blk[m] = (rank * k) // m.size
        for b in range(k):
            te = sel[blk == b]
            cand = sel[blk != b]
            if te.size == 0 or cand.size == 0:
                continue
            keep = np.ones(cand.size, dtype=bool)
            for sess in np.unique(ds.session[te]):
                t_te = np.sort(ds.t_end_us[te][ds.session[te] == sess])
                c = ds.session[cand] == sess
                tc = ds.t_end_us[cand][c]
                j = np.clip(np.searchsorted(t_te, tc), 1, max(t_te.size - 1, 1))
                near = np.minimum(np.abs(tc - t_te[j - 1]), np.abs(tc - t_te[np.minimum(
                    j, t_te.size - 1)]))
                keep[np.flatnonzero(c)[near <= guard_s * 1e6]] = False
            tr = cand[keep]
            out.append(Fold(f"{pid}|b{b}", str(pid), tr, te, np.zeros(0, dtype=np.int64),
                            blk[np.searchsorted(sel, tr)]))
    if not out:
        raise ValueError("blocked: no folds (empty dataset?)")
    return out
