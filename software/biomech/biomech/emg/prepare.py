# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-participant EMG preparation shared by the qc, emg and timing stages.

``EmgPrep`` owns the order of operations so every stage sees identical signals:

1. grid frequency per take (``take_line_hz``);
2. rest statistics from the motion take's ``rest`` phase (noise RMS, envelope
   median, 1-20 Hz level for the movement-artifact rule);
3. MVC/RVC references: p95 and max 1-s mean of the envelope in
   ``mvc_closing`` (masseter, temporalis) or ``mvc_opening`` (digastric);
4. per-phase signals: raw, cleaned (mains fit + zero-phase band-pass), the
   artifact mask, the masked envelope and %MVC/%RVC.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d

from biomech.emg.artifacts import MaskParams, MaskResult, artifact_mask, lowfreq_level, \
    masked_envelope
from biomech.emg.pipeline import clean, refine_line_hz, rms_envelope, take_line_hz
from biomech.io.session import OPENERS, SessionData


def max_window_mean(x: np.ndarray, n: int, min_valid: float = 0.5) -> np.ndarray:
    """Largest moving mean over ``n`` samples, per column, NaN-aware.

    Masked samples (NaN) are left out of each window's mean instead of counting
    as zero, and a window needs ``min_valid`` of its samples valid; a column
    without such a window is NaN.
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    valid = np.isfinite(x)
    s = uniform_filter1d(np.where(valid, x, 0.0), size=n, axis=0, mode="constant")
    c = uniform_filter1d(valid.astype(float), size=n, axis=0, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(c >= min_valid, s / np.where(c > 0, c, np.nan), np.nan)
    if len(mean) > n:                       # drop the zero-padded edge windows
        h = n // 2
        mean = mean[h:len(mean) - (n - 1 - h)]
    out = np.full(x.shape[1], np.nan)
    has = np.isfinite(mean).any(axis=0)
    out[has] = np.nanmax(mean[:, has], axis=0)
    return out


@dataclass
class PhaseEmg:
    name: str
    t_us: np.ndarray            # (N,) int64
    fs: float
    names: list[str]
    raw: np.ndarray             # (N, C) uV
    cleaned: np.ndarray         # (N, C) uV, unmasked
    masks: MaskResult
    env: np.ndarray             # (N, C) uV, NaN where masked
    norm: np.ndarray            # (N, C) % of reference

    @property
    def mask(self) -> np.ndarray:
        return self.masks.mask


REF_EFFORT_FRAC = 0.25         # envelope share of the reference p95 that counts as effort
# Motion-take phases whose activity can set the normalization reference (D11).
REFERENCE_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")


def effort_mask_frac(mask: np.ndarray, env: np.ndarray, p95: np.ndarray,
                     frac: float = REF_EFFORT_FRAC) -> np.ndarray:
    """Masked share of the effort samples (envelope >= ``frac`` x p95) per channel.

    A reference (p95, max-1-s) is set by the efforts, not by the pauses between them,
    so artifacts masked in the pauses must not make it unusable.
    """
    effort = env >= frac * np.asarray(p95)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        return (mask & effort).sum(axis=0) / effort.sum(axis=0)


@dataclass(frozen=True)
class EmgParams:
    bandpass_hz: tuple[float, float] = (20.0, 450.0)
    filter_order: int = 4
    envelope_s: float = 0.15
    nominal_line_hz: float = 50.0
    line_hz_window_s: float = 60.0
    robust_mains: bool = True
    line_hz_per_phase: bool = True
    mains_amp_order: int | str = "auto"
    reference: str = "max_recorded"     # max_recorded (D11) | p95 (frozen MVC/RVC p95) | max1s
    mask: MaskParams = MaskParams()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EmgParams":
        mk = dict(d.get("mask") or {})
        if "lf_band_hz" in mk:
            mk["lf_band_hz"] = tuple(mk["lf_band_hz"])
        if "analysis_rules" in mk:
            mk["analysis_rules"] = tuple(str(r) for r in mk["analysis_rules"])
        return cls(
            bandpass_hz=tuple(d.get("bandpass_hz", (20.0, 450.0))),
            filter_order=int(d.get("filter_order", 4)),
            envelope_s=float(d.get("envelope_s", 0.15)),
            nominal_line_hz=float(d.get("nominal_line_hz", 50.0)),
            line_hz_window_s=float(d.get("line_hz_window_s", 60.0)),
            robust_mains=bool(d.get("robust_mains", True)),
            line_hz_per_phase=bool(d.get("line_hz_per_phase", True)),
            mains_amp_order=(int(d["mains_amp_order"]) if str(d.get("mains_amp_order", "auto"))
                             .isdigit() else "auto"),
            reference=str(d.get("reference", "max_recorded")),
            mask=MaskParams(**mk),
        )


def _line_hz(sd: SessionData, p: EmgParams) -> float:
    t0 = min(ph.t_start_us for ph in sd.phases) if sd.phases else 0
    t, uv, _ = sd.emg_window(t0, t0 + int(p.line_hz_window_s * 1e6))
    return take_line_hz(uv, sd.emg_fs, t, p.nominal_line_hz)


class EmgPrep:
    def __init__(self, motion: SessionData, mvc: SessionData | None, p: EmgParams) -> None:
        self.motion = motion
        self.mvc = mvc
        self.p = p
        self.names = motion.muscle_names()
        self._line: dict[int, float] = {}
        self._cleaned: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._phase_line: dict[tuple[int, str], float] = {}
        # B1: PhaseEmg per (take, phase, normalization) and the causal chain per (take, phase);
        # both depend on the mask rules, so with_mask_rules() does not share them.
        self._phases: dict[tuple[str, str, str], PhaseEmg] = {}
        self._causal: dict[tuple[str, str], np.ndarray] = {}

    def with_mask_rules(self, rules: tuple[str, ...]) -> "EmgPrep":
        """The same participant under other analysis mask rules (D14 sensitivity).

        Cleaning does not depend on the mask, so the cleaned phases (the costly mains fit)
        and grid frequencies are shared; references, masks and envelopes are recomputed.
        """
        other = EmgPrep(self.motion, self.mvc,
                        replace(self.p, mask=replace(self.p.mask, analysis_rules=tuple(rules))))
        other._line, other._cleaned = self._line, self._cleaned
        other._phase_line = self._phase_line
        return other

    def release(self) -> None:
        """Drop the memoized per-phase results (PhaseEmg, causal chain); the cleaned phases
        and references stay. Results computed afterwards are identical."""
        self._phases.clear()
        self._causal.clear()

    def line_hz(self, sd: SessionData) -> float:
        key = id(sd)
        if key not in self._line:
            self._line[key] = _line_hz(sd, self.p)
        return self._line[key]

    def _clean(self, sd: SessionData, phase: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (id(sd), phase)                  # each phase is cleaned once (mains fit is costly)
        if key not in self._cleaned:
            self._cleaned[key] = self._clean_uncached(sd, phase)
        return self._cleaned[key]

    def _clean_uncached(self, sd: SessionData, phase: str
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t, uv, _ = sd.emg_phase(phase, self.names)
        lf = self.line_hz(sd)
        if self.p.line_hz_per_phase:
            lf = refine_line_hz(uv, sd.emg_fs, t, lf)
        cl, _ = clean(uv, sd.emg_fs, t, line_hz=lf, band_hz=self.p.bandpass_hz,
                      order=self.p.filter_order, robust_mains=self.p.robust_mains,
                      amp_order=self.p.mains_amp_order)
        self._phase_line[(id(sd), phase)] = float(lf)
        return t.astype(np.int64), uv, cl

    def phase_line_hz(self, sd: SessionData, phase: str) -> float:
        """The grid frequency the offline chain used for this phase (take value, refined
        per phase when ``line_hz_per_phase``)."""
        key = (id(sd), phase)
        if key not in self._phase_line:
            self._clean(sd, phase)
            if key not in self._phase_line:           # cleaned elsewhere (shared cache)
                t, uv, _ = self._cleaned[key]
                lf = self.line_hz(sd)
                self._phase_line[key] = (refine_line_hz(uv, sd.emg_fs, t, lf)
                                         if self.p.line_hz_per_phase else lf)
        return self._phase_line[key]

    # -- rest ---------------------------------------------------------------------------
    @cached_property
    def _rest_lf(self) -> np.ndarray:
        """Median 1-20 Hz level of the rest phase (scale of the low-frequency rule)."""
        _, raw, _ = self._clean(self.motion, "rest")
        return np.median(lowfreq_level(raw, self.motion.emg_fs, self.p.mask), axis=0)

    @cached_property
    def rest(self) -> dict[str, np.ndarray]:
        """Rest statistics. ``rms_uv`` (rest noise) uses the analysis-masked samples only
        (E22): a rail/lead-off stretch must not inflate the noise level; NaN when every
        sample of a channel is masked. ``analysis_mask`` is that mask."""
        t, raw, cl = self._clean(self.motion, "rest")
        fs = self.motion.emg_fs
        env = rms_envelope(cl, fs, self.p.envelope_s)
        m = self.phase("rest").mask                  # the analysis mask of the rest phase
        valid = ~m
        n = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = np.where(valid, cl, 0.0).sum(axis=0) / n
            rms = np.sqrt(np.where(valid, (cl - mu) ** 2, 0.0).sum(axis=0) / n)
        return {
            "rms_uv": rms,
            "env_median_uv": np.median(env, axis=0),
            "lf_uv": self._rest_lf,
            "raw": raw, "cleaned": cl, "analysis_mask": m,
        }

    # -- references ---------------------------------------------------------------------
    @cached_property
    def refs(self) -> dict[str, np.ndarray]:
        c = len(self.names)
        out = {"p95_uv": np.full(c, np.nan), "max1s_uv": np.full(c, np.nan),
               "max_uv": np.full(c, np.nan),
               "mask_frac": np.full(c, np.nan), "mask_frac_effort": np.full(c, np.nan),
               "phase": np.array(
                   ["mvc_opening" if n in OPENERS else "mvc_closing" for n in self.names])}
        if self.mvc is None:
            return out
        fs = self.mvc.emg_fs
        for phase in ("mvc_opening", "mvc_closing"):
            if self.mvc.phase(phase) is None:
                continue
            t, raw, cl = self._clean(self.mvc, phase)
            env = rms_envelope(cl, fs, self.p.envelope_s)
            sel = out["phase"] == phase
            p95 = np.percentile(env, 95, axis=0)
            m = self._mask(self.mvc, phase, raw, cl, ref=np.where(sel, p95, np.nan))
            envm = masked_envelope(cl, m.mask, fs, self.p.envelope_s)
            out["p95_uv"][sel] = np.nanpercentile(envm, 95, axis=0)[sel]
            out["max1s_uv"][sel] = max_window_mean(envm, int(fs))[sel]
            with np.errstate(invalid="ignore"):
                mx = np.nanmax(np.where(np.isfinite(envm), envm, -np.inf), axis=0)
            out["max_uv"][sel] = np.where(np.isfinite(mx), mx, np.nan)[sel]   # fully masked: NaN
            out["mask_frac"][sel] = m.mask.mean(axis=0)[sel]
            out["mask_frac_effort"][sel] = effort_mask_frac(m.mask, env, p95)[sel]
        return out

    @cached_property
    def task_peaks(self) -> dict[str, np.ndarray]:
        """Largest masked envelope (and 1-s mean) per channel over the motion-take tasks.

        Masks here omit the envelope rule (its scale is the reference being determined);
        the other analysis rules (D14 default: rail, lead-off) apply as everywhere.
        """
        c = len(self.names)
        out = {"max_uv": np.full(c, -np.inf), "max1s_uv": np.full(c, -np.inf),
               "phase": np.array([""] * c, dtype=object)}
        fs = self.motion.emg_fs
        for phase in REFERENCE_TASKS:
            if self.motion.phase(phase) is None:
                continue
            t, raw, cl = self._clean(self.motion, phase)
            m = self._mask(self.motion, phase, raw, cl, None)
            envm = masked_envelope(cl, m.mask, fs, self.p.envelope_s)
            with np.errstate(invalid="ignore"):
                mx = np.nanmax(np.where(np.isfinite(envm), envm, -np.inf), axis=0)
            m1 = np.nan_to_num(max_window_mean(envm, int(fs)), nan=-np.inf)
            better = mx > out["max_uv"]
            out["phase"][better] = phase
            out["max_uv"] = np.maximum(out["max_uv"], mx)
            out["max1s_uv"] = np.maximum(out["max1s_uv"], m1)
        for k in ("max_uv", "max1s_uv"):
            out[k] = np.where(np.isfinite(out[k]), out[k], np.nan)
        return out

    @cached_property
    def reference(self) -> dict[str, np.ndarray]:
        """The normalization reference (analysis_plan D11): the largest activity recorded.

        ``max_recorded_uv`` = max(MVC/RVC-phase max, task max) of the masked envelope, so
        normalized activity is <= 100 %. ``source`` = "mvc" when the reference contraction
        holds the maximum, "task" when a task exceeded it (a failed or submaximal reference
        contraction: flagged, not excluded).
        """
        mvc, task = self.refs["max_uv"], self.task_peaks["max_uv"]
        ref = np.fmax(mvc, task)
        src = np.where(np.isfinite(mvc) & ~(task > mvc), "mvc",
                       np.where(np.isfinite(task), "task", "none"))
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = mvc / task
        return {"max_recorded_uv": ref, "source": src, "mvc_over_task": ratio,
                "max1s_recorded_uv": np.fmax(self.refs["max1s_uv"], self.task_peaks["max1s_uv"]),
                "task_phase": self.task_peaks["phase"]}

    def ref_uv(self, normalization: str | None = None) -> np.ndarray:
        """Reference per channel: ``max_recorded`` (default, D11), ``p95`` (the frozen MVC/RVC
        p95) or ``max1s`` (largest 1-s mean over the MVC/RVC phase and the tasks)."""
        normalization = normalization or self.p.reference
        if normalization == "p95":
            return self.refs["p95_uv"]
        if normalization == "max1s":
            return self.reference["max1s_recorded_uv"]
        if normalization == "max_recorded":
            return self.reference["max_recorded_uv"]
        raise ValueError(f"unknown EMG normalization {normalization!r}")

    def envelope_rule_ref(self) -> np.ndarray:
        """Scale of the envelope artifact rule: the MVC/RVC p95 where the reference
        contraction holds the maximum; off (NaN) where a task exceeded it, because there the
        rule would cut physiological activity (D11)."""
        return np.where(self.reference["source"] == "mvc", self.refs["p95_uv"], np.nan)

    # -- phases -------------------------------------------------------------------------
    def _mask(self, sd: SessionData, phase: str, raw: np.ndarray, cl: np.ndarray,
              ref: np.ndarray | None) -> MaskResult:
        ph = sd.phase(phase)
        lead = sd.leadoff_window(ph.t_start_us, ph.t_end_us, self.names)
        if lead.shape[0] != raw.shape[0]:
            lead = None
        return artifact_mask(raw, cl, sd.emg_fs, full_scale_uv=sd.emg_full_scale_uv(self.names),
                             leadoff=lead, ref_uv=ref, rest_lf_uv=self._rest_lf,
                             p=self.p.mask)

    def phase(self, phase: str, *, take: str = "motion",
              normalization: str | None = None) -> PhaseEmg:
        """Signals of one phase, memoized per (take, phase, normalization) (B1).

        The arrays of the returned object are shared between callers and read-only;
        ``release()`` drops the memo.
        """
        key = (take, phase, normalization or self.p.reference)
        hit = self._phases.get(key)
        if hit is None:
            hit = self._phases[key] = self._phase_uncached(phase, take, normalization)
        return hit

    def _phase_uncached(self, phase: str, take: str, normalization: str | None) -> PhaseEmg:
        sd = self.motion if take == "motion" else self.mvc
        t, raw, cl = self._clean(sd, phase)
        ref = self.ref_uv(normalization)
        m = self._mask(sd, phase, raw, cl, self.envelope_rule_ref())
        env = masked_envelope(cl, m.mask, sd.emg_fs, self.p.envelope_s)
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = 100.0 * env / ref[None, :]
        for a in (env, norm, m.mask):
            a.setflags(write=False)
        return PhaseEmg(phase, t, sd.emg_fs, list(self.names), raw, cl, m, env, norm)

    def causal(self, phase: str, *, take: str = "motion") -> np.ndarray:
        """Causally cleaned phase (E27, :func:`biomech.emg.causal.causal_clean`), NaN at the
        analysis-masked samples; memoized per (take, phase), read-only."""
        key = (take, phase)
        if key not in self._causal:
            from biomech.emg.causal import causal_clean

            x = causal_clean(self, phase, take=take)
            x.setflags(write=False)
            self._causal[key] = x
        return self._causal[key]
