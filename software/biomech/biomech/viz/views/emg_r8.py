# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG gallery views (J2.3): signals, spectra, timing and feature space in new forms.

Twelve dashboards for the dataset paper gallery; the authors pick forms for the papers. Each is
built from the stored results (the paper store) and reads the recorded session only for the
per-participant signal forms (the raw EMG of one channel), falling back to the stored chain
example when no session is available:

* ``emg_signal_spectrum`` -- raw → band-passed → TKEO → envelope of one channel with the rest
  threshold, both onset rules and the mechanical onsets, a short-time spectrum and the Welch
  PSD (form ``chain``); a stratified random sample of channel-cycles per participant as the
  onset check (form ``onset_qc``);
* ``emg_cycle_spectrogram`` -- cycle-normalised spectral shape (cycle phase × frequency) per
  muscle group or muscle, the activity mask in grey, the mean frequency over the cycle;
* ``emg_onset_raster`` -- channel-cycles × time envelope image around the mechanical onset,
  sorted by lead, with the onset ticks of both rules;
* ``emg_motion_coupling`` -- lagged correlation and coherence of the closer envelope with the
  closing angular acceleration (the stored coupling lags marked);
* ``emg_activation_clock`` -- openers vs closers over the cycle phase as a clock (x/y loop)
  with the SPM clusters as arcs, or unrolled with significance bars;
* ``emg_surface`` -- muscle × cycle-phase activation surface per task, or a 3-D waterfall of
  the cycle-phase spectra;
* ``emg_on_path`` -- the incisor path coloured by the opener / closer envelope (3-D: where
  each group is active; 2-D: colour lines in the sagittal and frontal planes);
* ``emg_story`` -- one chewing excerpt on one time axis: raw EMG, spectrogram strip,
  envelopes, IMU and optical opening angle, intent label, decoder output and release;
* ``emg_feature_embedding`` -- principal components of the decoding feature windows,
  coloured by class, highlighted per participant, or fitted within each participant;
* ``emg_ridgelines`` -- per-participant spectra or envelopes as ridgelines with the cohort
  median on top and each participant's amplitude printed;
* ``emg_usability_flow`` -- recorded channel-cycles → usable ones through the timing gates
  (Sankey-style), or the outcome shares per participant;
* ``kin_phase_portrait`` -- cohort density of opening angle vs angular velocity (or angular
  velocity vs acceleration) per task.

Every method parameter in a caption is read from the configuration the stages ran with
(:func:`blocks.cfg`) or derived from the stored arrays; the display constants of this module
(short-time window, phase bins, sample sizes) are named here and quoted from these names.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.caption import CaptionSpec, attach
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series
from biomech.viz.views import blocks as B
from biomech.viz.views import vocab as V
from biomech.viz.views.registry import ViewRequest, register
from biomech.viz.views.vocab import CLOSERS, MUSCLE_LABEL, MUSCLES, OPENERS, TASK_LABEL

# -- shared vocabulary of this module ---------------------------------------------------------
CYCLE_TASKS = ("open_close", "chewing")
GROUPS = (("openers", OPENERS), ("closers", CLOSERS))
GROUP_ROLE = {"openers": "group:openers", "closers": "group:closers"}
GROUP_LABEL = {"openers": "openers (digastric)", "closers": "closers (masseter, temporalis)"}
GROUP_SHORT = {"openers": "openers", "closers": "closers"}
DIR_GROUP = {"opening": "openers", "closing": "closers"}
RULES = ("first", "drive")
RULE_LABEL = {"first": "first onset", "drive": "driving burst"}
RULE_ROLE = {"first": "secondary", "drive": "accent"}
RULE_MS = {"first": "v", "drive": "d"}
CLASSES = ("rest", "open", "close", "protrude", "retrude", "left", "right")
CLASS_ROLE = ("context", "cat:0", "cat:1", "cat:2", "cat:3", "cat:4", "cat:5")

# Display constants (quoted in the captions from these names)
EXCERPT_CYCLES = {"open_close": 2, "chewing": 3}   # cycles in the signal excerpt
STFT_S = 0.064               # short-time spectrum window (s), 75 % overlap
STFT_OVERLAP = 0.75
QC_PER_STRATUM = 2           # sampled channel-cycles per task x usability stratum
PHASE_BINS = 20              # cycle-phase bins of the cycle spectrogram
EMBED_PER_CLASS = 40         # feature windows per participant and class in the embedding
PORTRAIT_BINS = 48           # bins per axis of the phase-portrait density
SURF_STEP = 5                # cycle points per surface column (% of the cycle)
SURF_ROW = 2.0               # spacing of the muscle rows on the surface (display units)
COH_SEG_S = 2.0              # Welch segment of the coherence (s)
COH_MIN = 0.1                # coherence below which the phase lag is not shown


# == helpers ===================================================================================
def _t(ctx: Any, stage: str, name: str, pids: list[str] | None = None) -> pd.DataFrame:
    """A stored table, empty when the stage or table is missing."""
    try:
        d = ctx.table(stage, name, pids)
    except Exception:
        return pd.DataFrame()
    return d if isinstance(d, pd.DataFrame) else pd.DataFrame()


def _a(ctx: Any, stage: str, unit: str) -> dict[str, np.ndarray]:
    """Stored arrays of one unit, empty when missing."""
    try:
        return ctx.arrays(stage, unit) or {}
    except Exception:
        return {}


def _pids(ctx: Any) -> list[str]:
    try:
        return list(ctx.pids())
    except Exception:
        return []


def _pid(req: ViewRequest) -> str | None:
    pids = _pids(req.ctx)
    return req.pid if req.pid in pids else (pids[0] if pids else None)


def _who(ctx: Any, pid: str) -> str:
    return f"P{ctx.pseudonym(pid)}"


def _sig(ctx: Any, block: str, *keys: str, default: Any = None) -> Any:
    """A signal-chain parameter (``config.signals.<block>.<keys>``) or ``default``."""
    try:
        d: Any = ctx.config.signals.get(block) or {}
    except AttributeError:
        return default
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _tp(ctx: Any, *keys: str, default: Any = None) -> Any:
    return B.cfg(ctx, "timing", *keys, default=default)


def _band(ctx: Any) -> tuple[float, float]:
    b = _sig(ctx, "emg", "bandpass_hz", default=(20.0, 450.0))
    return float(b[0]), float(b[1])


def _env_ms(ctx: Any) -> float:
    return 1e3 * float(_sig(ctx, "emg", "envelope_s", default=0.15))


def _primary_rule(ctx: Any) -> str:
    try:
        r = str(ctx.onset_rule())
    except Exception:
        r = str(_tp(ctx, "onset_rule", default="drive"))
    return r if r in RULES else "drive"


def _detector() -> str:
    try:
        from biomech.stats.cohort_misc import PRIMARY

        return str(PRIMARY["detector"])
    except Exception:                                # pragma: no cover - stats missing
        return "tkeo"


def _det_label() -> str:
    return V.term("detector", _detector())


def _names(ctx: Any, pid: str, e: dict[str, np.ndarray] | None = None) -> list[str]:
    e = _a(ctx, "emg", pid) if e is None else e
    if "profile__names" in e:
        return [str(n) for n in e["profile__names"]]
    r = _t(ctx, "emg", "refs", [pid])
    return list(r.muscle) if not r.empty and "muscle" in r else []


def _cycles(ctx: Any, pid: str | None, task: str) -> pd.DataFrame:
    c = _t(ctx, "kinematics", "cycles", [pid] if pid else None)
    if c.empty or "task" not in c:
        return c
    c = c[c.task == task]
    if "ok" in c:
        c = c[c.ok.astype(bool)]
    return c.sort_values(["pid", "t_start_us"] if "pid" in c else "t_start_us")


def _q(x: Any, q: float) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def _nice(v: float) -> float:
    if not np.isfinite(v) or v <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(v))
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * mag >= v:
            return float(m * mag)
    return float(10 * mag)


def _nanmean(a: Any, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(a, float), axis=axis)


def _nanmedian(a: Any, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.asarray(a, float), axis=axis)


def _empty_panel(title: str, msg: str) -> Panel:
    p = Panel(title, Axis("", lim=(0.0, 1.0)), Axis("", lim=(0.0, 1.0)),
              [Series("text", np.array([0.5]), np.array([0.45]), text=[msg],
                      style={"anchor": "above"})], legend="none")
    p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
    return p


def _empty_spec(fid: str, title: str, msg: str, what: str) -> FigureSpec:
    """A one-panel figure saying why there is nothing to show, with a structured caption."""
    spec = FigureSpec(fid, title, [_empty_panel("", msg)], width="column", height_in=1.2)
    return attach(spec, CaptionSpec(
        f"{what}; nothing to show here because {msg}: the stage outputs this figure reads are "
        "missing from the result store or hold no rows for the request",
        [("text", "the reason the panel is empty")], "n = 0", evidence="descriptive"))


def _grid_labels(panels: list[Panel], cols: int) -> None:
    """Small multiples with common axes: x label on the bottom row, y label on the left."""
    n = len(panels)
    for k, p in enumerate(panels):
        if k + cols < n and p.x.label:
            p.hidden_labels["x"] = (p.x.label, p.x.unit)
            p.x.label, p.x.unit = "", ""
        if k % cols and p.y.label:
            p.hidden_labels["y"] = (p.y.label, p.y.unit)
            p.y.label, p.y.unit = "", ""


def _bottom_x(panels: list[Panel]) -> None:
    for p in panels[:-1]:
        if p.x.label:
            p.hidden_labels["x"] = (p.x.label, p.x.unit)
        p.x.label, p.x.unit = "", ""


def _vlines(p: Panel, xs: list[float], role: str, label: str | None = None) -> None:
    xs = [float(x) for x in xs if np.isfinite(x)]
    if xs:
        p.series.append(Series("vline", x=np.array(xs), role=role, label=label))


def _thin(x: np.ndarray, *ys: np.ndarray, budget: int = 6000) -> np.ndarray:
    from biomech.viz.downsample import thin_index

    return thin_index(np.asarray(x, float), *[np.asarray(y, float) for y in ys], budget=budget)


def _shape_ok(ctx: Any) -> pd.DataFrame:
    """``emg:spectral`` rows (pid, task, muscle, usable_shape)."""
    s = _t(ctx, "emg", "spectral")
    if s.empty or "usable_shape" not in s:
        return pd.DataFrame(columns=["pid", "task", "muscle", "usable_shape"])
    return s


def _rest_p95(ctx: Any) -> pd.DataFrame:
    """Rest-phase 95th percentile of the envelope (% of max recorded), pid x muscle."""
    a = _t(ctx, "emg", "activation")
    if a.empty or not {"task", "p95_pct"} <= set(a.columns):
        return pd.DataFrame()
    return a[a.task == "rest"].pivot_table(index="pid", columns="muscle", values="p95_pct",
                                           aggfunc="first")


def _env_grid(e: dict[str, np.ndarray], task: str, names: list[str], muscles: tuple[str, ...]
              ) -> tuple[np.ndarray, np.ndarray] | None:
    """(t_us, mean envelope over ``muscles`` in % of max recorded, NaN = masked) on the
    stored envelope grid of ``task``."""
    if f"{task}__t_us" not in e or f"{task}__norm_pct" not in e:
        return None
    t = np.asarray(e[f"{task}__t_us"], np.int64)
    N = np.asarray(e[f"{task}__norm_pct"], float).copy()
    if f"{task}__masked" in e:
        N[np.asarray(e[f"{task}__masked"], bool)] = np.nan
    cols = [names.index(m) for m in muscles if m in names]
    if not cols:
        return None
    return t, _nanmean(N[:, cols], axis=1)


def _env_channels(e: dict[str, np.ndarray], task: str, names: list[str],
                  muscles: tuple[str, ...]) -> list[tuple[np.ndarray, np.ndarray]]:
    """(t_us, envelope in % of max recorded, NaN = masked) of each channel of ``muscles`` on
    the stored envelope grid of ``task`` (the inputs of :func:`_env_grid`'s mean)."""
    if f"{task}__t_us" not in e or f"{task}__norm_pct" not in e:
        return []
    t = np.asarray(e[f"{task}__t_us"], np.int64)
    N = np.asarray(e[f"{task}__norm_pct"], float).copy()
    if f"{task}__masked" in e:
        N[np.asarray(e[f"{task}__masked"], bool)] = np.nan
    return [(t, N[:, names.index(m)]) for m in muscles if m in names]


def _phase_ticks(p: Panel) -> None:
    p.x.ticks = [0.0, 25.0, 50.0, 75.0, 100.0]


# == 1. signal + spectrum, onset check =========================================================
def _signal(ctx: Any, pid: str, task: str, muscle: str) -> dict[str, Any] | None:
    """One channel of a task: the recorded session when it can be read, else the stored
    chain example (a 2-s window). Keys: t_us, fs, raw, cleaned, mask, norm, rest, src."""
    try:
        prep = ctx.emg(pid)
        if muscle not in prep.names or prep.motion.phase(task) is None:
            raise LookupError(muscle)
        pe, rest = prep.phase(task), prep.phase("rest")
        i = list(prep.names).index(muscle)
        raw = np.asarray(pe.raw[:, i], float)
        return {"t_us": np.asarray(pe.t_us, np.int64), "fs": float(pe.fs),
                "raw": raw - np.median(raw), "cleaned": np.asarray(pe.cleaned[:, i], float),
                "mask": np.asarray(pe.mask[:, i], bool), "norm": np.asarray(pe.norm[:, i], float),
                "rest": np.where(np.asarray(rest.mask[:, i], bool), np.nan,
                                 np.asarray(rest.cleaned[:, i], float)),
                "src": "session:emg", "session": True}
    except Exception:
        pass
    e = _a(ctx, "emg", pid)
    k = f"chain__{task}__"
    names = _names(ctx, pid, e)
    if f"{k}t_us" not in e or muscle not in names:
        return None
    i = names.index(muscle)
    t = np.asarray(e[f"{k}t_us"], np.int64)
    rest = None
    if "chain__rest__cleaned" in e:
        rest = np.where(np.asarray(e["chain__rest__masked"][:, i], bool), np.nan,
                        np.asarray(e["chain__rest__cleaned"][:, i], float))
    raw = np.asarray(e[f"{k}raw"][:, i], float)
    return {"t_us": t, "fs": 1e6 / float(np.median(np.diff(t))), "raw": raw - np.median(raw),
            "cleaned": np.asarray(e[f"{k}cleaned"][:, i], float),
            "mask": np.asarray(e[f"{k}masked"][:, i], bool),
            "norm": np.asarray(e[f"{k}norm"][:, i], float), "rest": rest,
            "src": "emg:chain", "session": False}


def _onset_params(ctx: Any):
    from jtsmm.emg.onset import OnsetParams

    return OnsetParams(method=_detector(), k_sd=float(_tp(ctx, "k_sd", default=3.0)))


def _threshold(rest: np.ndarray | None, fs: float, op: Any) -> float:
    from jtsmm.emg.onset import baseline_threshold

    if rest is None or not np.isfinite(rest).any():
        return float("nan")
    return float(baseline_threshold(rest, fs, op))


def _detector_signal(x: np.ndarray, fs: float, op: Any) -> np.ndarray:
    from jtsmm.emg.onset import detector_signal

    return np.asarray(detector_signal(np.nan_to_num(np.asarray(x, float)), fs, op), float)


def _onset_rows(ctx: Any, pid: str | None, task: str | None = None) -> pd.DataFrame:
    """Channel-cycles of the timing onsets (primary detector): one row per (pid, task,
    direction, cycle, muscle) with t_mech, t_omega, burst, active-at-start, and per rule the
    onset time, lead and usability (``t_on_<rule>``, ``lead_<rule>``, ``usable_<rule>``)."""
    o = _t(ctx, "timing", "onsets", [pid] if pid else None)
    need = {"pid", "task", "cycle", "direction", "muscle", "detector", "rule", "t_onset_us",
            "t_mech_us", "usable"}
    if o.empty or not need <= set(o.columns):
        return pd.DataFrame()
    o = o[o.detector == _detector()]
    if task is not None:
        o = o[o.task == task]
    if o.empty:
        return pd.DataFrame()
    key = ["pid", "task", "direction", "cycle", "muscle"]
    base = o.groupby(key, sort=False).first().reset_index()
    keep = key + [c for c in ("t_mech_us", "t_omega_us", "burst_db", "active_at_start")
                  if c in base]
    out = base[keep].copy()
    for r in RULES + ("drive_taskbase",):
        s = o[o.rule == r].set_index(key)
        if s.empty:
            continue
        idx = pd.MultiIndex.from_frame(out[key])
        out[f"t_on_{r}"] = s["t_onset_us"].reindex(idx).to_numpy(float)
        out[f"usable_{r}"] = s["usable"].reindex(idx).fillna(False).to_numpy(bool)
        out[f"lead_{r}"] = (out["t_mech_us"].to_numpy(float) - out[f"t_on_{r}"]) / 1e3
    return out


def _reason(r: Any, rule: str, burst_min: float) -> str:
    """Why a channel-cycle is not usable under ``rule`` (the timing gates, in order)."""
    if bool(getattr(r, f"usable_{rule}", False)):
        return "usable"
    if bool(getattr(r, "active_at_start", False)):
        return "active at window start"
    b = float(getattr(r, "burst_db", np.nan))
    if not np.isfinite(b) or b < burst_min:
        return f"burst below {burst_min:g} dB"
    if not np.isfinite(float(getattr(r, f"t_on_{rule}", np.nan))):
        return "no onset found"
    return "masked samples in the window"


@register("emg_signal_spectrum", "dashboard", "EMG signal, onsets and spectrum",
          params=("pid", "phase"), phase="open_close", phases=CYCLE_TASKS, domain="emg_time",
          choices={"form": ("chain", "onset_qc"), "muscle": MUSCLES}, form="chain",
          muscle="digastric_r")
def emg_signal_spectrum(req: ViewRequest) -> FigureSpec:
    """Raw → band-passed → TKEO → envelope with both onset rules, a short-time spectrum and
    the Welch PSD of one channel; or a stratified random sample of channel-cycles (onset
    check)."""
    fid, title = "emg_signal_spectrum", "EMG signal, onsets and spectrum"
    pid = _pid(req)
    if pid is None:
        return _empty_spec(fid, title, "there are no participants", "EMG signal and spectrum")
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    if req.options.get("form") == "onset_qc":
        return _onset_qc(req, pid, task)
    muscle = req.options.get("muscle") if req.options.get("muscle") in MUSCLES else "digastric_r"
    return _signal_chain(req, pid, task, muscle)


def _signal_chain(req: ViewRequest, pid: str, task: str, muscle: str) -> FigureSpec:
    from scipy.signal import spectrogram

    ctx = req.ctx
    fid, title = "emg_signal_spectrum", "EMG signal, onsets and spectrum"
    who = _who(ctx, pid)
    sg = _signal(ctx, pid, task, muscle)
    if sg is None:
        return _empty_spec(fid, title, f"no recorded session and no stored example for {who}",
                           "EMG signal and spectrum")
    t, fs = sg["t_us"], sg["fs"]
    rule0 = _primary_rule(ctx)
    pre_s = float(_tp(ctx, "emg_pre_s", default=0.6))
    rows = _onset_rows(ctx, pid, task)
    rows = rows[rows.muscle == muscle] if not rows.empty else rows
    cyc = _cycles(ctx, pid, task)
    cyc = cyc[(cyc.t_start_us >= t[0]) & (cyc.t_end_us <= t[-1])] if not cyc.empty else cyc
    n_cyc = EXCERPT_CYCLES.get(task, 2)
    if sg["session"] and len(cyc) >= 1:
        starts = cyc.t_start_us.to_numpy(float)
        ends = cyc.t_end_us.to_numpy(float)
        use = (rows.set_index("cycle")[f"usable_{rule0}"] if not rows.empty
               and f"usable_{rule0}" in rows else pd.Series(dtype=bool))
        cyc_ids = cyc.cycle.to_numpy(int)
        score = [sum(bool(use.get(c, False)) for c in cyc_ids[k:k + n_cyc])
                 for k in range(max(len(cyc) - n_cyc + 1, 1))]
        k0 = int(np.argmax(score))
        k1 = min(k0 + n_cyc, len(cyc)) - 1
        t0 = max(float(starts[k0]) - pre_s * 1e6, float(t[0]))
        t1 = min(float(ends[k1]), float(t[-1]))
        shown = cyc_ids[k0:k1 + 1]
    else:
        t0, t1 = float(t[0]), float(t[-1])
        shown = cyc.cycle.to_numpy(int) if not cyc.empty else np.array([], int)
    sel = (t >= t0) & (t <= t1)
    ts = (t[sel] - t0) / 1e6
    raw, cl, msk, nrm = (sg[k][sel] for k in ("raw", "cleaned", "mask", "norm"))
    x = np.where(msk, np.nan, cl)
    op = _onset_params(ctx)
    thr = _threshold(sg["rest"], fs, op)
    det = _detector_signal(x, fs, op)
    rs = rows[rows.cycle.isin(shown)] if not rows.empty else rows
    t_mech = ((rs.t_mech_us.to_numpy(float) - t0) / 1e6) if not rs.empty else np.array([])
    t_om = ((rs.t_omega_us.to_numpy(float) - t0) / 1e6) if not rs.empty and "t_omega_us" in rs \
        else np.array([])
    ons = {r: ((rs[f"t_on_{r}"].to_numpy(float) - t0) / 1e6) if not rs.empty
           and f"t_on_{r}" in rs else np.array([]) for r in RULES}
    role = f"entity:{muscle}"
    ml = MUSCLE_LABEL.get(muscle, muscle)
    with np.errstate(all="ignore"):
        amp_raw = _nice(float(np.nanpercentile(np.abs(raw), 99.9)) * 1.1) if raw.size else 1.0
        amp = _nice(float(np.nanpercentile(np.abs(cl), 99.9)) * 1.1) if cl.size else 1.0
    panels: list[Panel] = []

    def time_panel(ttl: str, y: np.ndarray, ylab: str, unit: str, rl: str,
                   lim: tuple[float, float] | None = None) -> Panel:
        k = _thin(ts, y)
        p = Panel(ttl, Axis("Time", V.SEC, lim=(0.0, float(ts[-1]) if ts.size else 1.0)),
                  Axis(ylab, unit, lim=lim), [Series("line", ts[k], y[k], role=rl,
                                                     style={"width": 0.45})],
                  sharex="t", legend="none")
        return p

    pa = time_panel("(a) Raw EMG (median removed)", raw, "EMG", V.UV, "context",
                    (-amp_raw, amp_raw))
    pb = time_panel(f"(b) Band-passed, mains removed: {ml}", cl, "EMG", V.UV, role, (-amp, amp))
    from biomech.viz.views.data import _runs

    runs = _runs(msk)
    if runs:
        pb.series.insert(0, Series("hspan", np.array([ts[a] for a, _ in runs]),
                                   y2=np.array([ts[b - 1] for _, b in runs]), role="threshold",
                                   style={"alpha": 0.15}))
    with np.errstate(divide="ignore", invalid="ignore"):
        ddb = (10 * np.log10(np.maximum(det / thr, 1e-2)) if np.isfinite(thr) and thr > 0
               else np.full(det.shape, np.nan))
    top = max(20.0, _nice(_q(ddb, 99.9))) if np.isfinite(ddb).any() else 20.0
    pc = time_panel(f"(c) {_det_label()} detector signal re rest threshold", ddb,
                    _det_label(), "\\decibel", role, (-20.0, top))
    # unlabelled: the in-plot label covered the trace; the caption names the line
    pc.series.append(Series("hline", y=np.array([0.0]), role="threshold"))
    pd_ = Panel("(d) Envelope with onsets", Axis("Time", V.SEC, lim=pa.x.lim),
                Axis("Envelope", V.PCT), [], sharex="t", legend="below")
    k = _thin(ts, nrm)
    pd_.series.append(Series("line", ts[k], nrm[k], role=role))
    for r in RULES:
        v = ons[r][np.isfinite(ons[r])]
        if v.size:
            yv = np.interp(v, ts, np.nan_to_num(nrm))
            pd_.series.append(Series("scatter", v, yv, role=RULE_ROLE[r],
                                     label=f"{RULE_LABEL[r]} rule",
                                     style={"ms": RULE_MS[r], "size": 1.5}))
            pc.series.append(Series("scatter", v, np.zeros(v.size), role=RULE_ROLE[r],
                                    style={"ms": RULE_MS[r], "size": 1.3}))
    for p in (pa, pb, pc, pd_):
        _vlines(p, list(t_mech), "reference")
    _vlines(pd_, list(t_om), "muted")
    # legend entries for the onset lines (reference lines carry none)
    if np.isfinite(t_mech).any():
        pd_.series.append(Series("line", np.array([np.nan]), np.array([np.nan]),
                                 role="reference", label="mechanical onset"))
    if np.isfinite(t_om).any():
        pd_.series.append(Series("line", np.array([np.nan]), np.array([np.nan]),
                                 role="muted", label="IMU velocity onset"))
    panels += [pa, pb, pc, pd_]
    # (e) short-time spectrum of the band-passed EMG
    lo, hi = _band(ctx)
    nper = max(int(round(STFT_S * fs)), 16)
    pe_ = Panel("(e) Short-time spectrum", Axis("Time", V.SEC, lim=pa.x.lim),
                Axis("Frequency", V.HZ, lim=(lo, hi)), [], sharex="t", legend="none",
                height=1.35)
    if x.size > nper:
        f, tt, S = spectrogram(np.nan_to_num(x), fs, nperseg=nper,
                               noverlap=int(round(nper * STFT_OVERLAP)), scaling="density")
        keep = (f >= lo) & (f <= hi)
        f, S = f[keep], S[keep]
        with np.errstate(divide="ignore"):
            Z = 10 * np.log10(np.where(S > 0, S, np.nan))
        half = nper / fs / 2
        bad = np.array([msk[(ts >= c - half) & (ts <= c + half)].any() for c in tt])
        zl = (np.floor(_q(Z, 5)), np.ceil(_q(Z, 99.5)))
        df = float(f[1] - f[0]) if f.size > 1 else 1.0
        dt = float(tt[1] - tt[0]) if tt.size > 1 else nper / fs
        img = Z[::-1]
        pe_.series.append(Series("image", img=img, extent=(
            float(tt[0] - dt / 2), float(tt[-1] + dt / 2), float(f[0] - df / 2),
            float(f[-1] + df / 2)), style={"cmap": "sequential", "lim": zl,
                                          "mask": np.broadcast_to(bad[None, :], img.shape)
                                          .copy()}))
        pe_.colorbar = Colorbar("PSD", "\\decibel", lim=zl)
        _vlines(pe_, list(t_mech), "reference")
    else:
        pe_.note = "excerpt too short"
    panels.append(pe_)
    # (f) Welch PSD of the whole task (stored by the emg stage) with MNF / MDF
    e = _a(ctx, "emg", pid)
    names = _names(ctx, pid, e)
    pf = Panel("(f) Power spectral density, whole task", Axis("Frequency", V.HZ, lim=(0.0, hi)),
               Axis("PSD", "\\decibel"), [], legend="none", height=1.1)
    spec_src = []
    mnf = mdf = float("nan")
    if f"spectral__{task}__psd" in e and muscle in names:
        fq = np.asarray(e["spectral__freq"], float)
        P = np.asarray(e[f"spectral__{task}__psd"], float)[names.index(muscle)]
        with np.errstate(divide="ignore", invalid="ignore"):
            pf.series.append(Series("line", fq, 10 * np.log10(np.where(P > 0, P, np.nan)),
                                    role=role))
        st = _t(ctx, "emg", "spectral", [pid])
        if not st.empty:
            r = st[(st.task == task) & (st.muscle == muscle)]
            if not r.empty:
                mnf, mdf = float(r.mnf_hz.iloc[0]), float(r.mdf_hz.iloc[0])
        _vlines(pf, [mnf], "reference")
        _vlines(pf, [mdf], "muted")
        for v, rl, lab in ((mnf, "reference", "mean frequency"), (mdf, "muted", "median frequency")):
            if np.isfinite(v):
                pf.series.append(Series("line", np.array([np.nan]), np.array([np.nan]), role=rl,
                                        label=f"{lab} ({v:.0f} Hz)"))
        pf.legend = "below"
        spec_src = ["emg:arrays", "emg:spectral"]
    else:
        pf.note = "no stored spectrum"
    panels.append(pf)
    _bottom_x(panels[:5])
    n_on = {r: int(np.isfinite(ons[r]).sum()) for r in RULES}
    excerpt = (f"{len(shown)} {TASK_LABEL.get(task, task)} cycles" if sg["session"]
               else "the stored example window")
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text", height_in=0.62,
                      sources=[sg["src"], "timing:onsets", "kinematics:cycles"] + spec_src,
                      perspective=f"{who}, {ml}, {TASK_LABEL.get(task, task)}",
                      meta={"pid": pid, "muscle": muscle, "threshold": thr,
                            "mnf_hz": mnf, "mdf_hz": mdf, "cycles": [int(c) for c in shown]})
    k_sd = float(_tp(ctx, "k_sd", default=3.0))
    return attach(spec, CaptionSpec(
        f"{who}, {ml}, {excerpt}: the raw EMG; the band-passed ({lo:g}–{hi:g} Hz) EMG after mains "
        f"removal; the {_det_label()} detector signal with the rest threshold (rest mean + "
        f"{k_sd:g} SD; in dB re the threshold); the envelope ({_env_ms(ctx):g} ms RMS, % of max "
        "recorded) with the onsets of both rules; the short-time spectrum of the band-passed EMG "
        f"({1e3 * STFT_S:g} ms windows, {100 * STFT_OVERLAP:g} % overlap); and the Welch power "
        "spectral density of the whole task with its mean (MNF) and median (MDF) frequency",
        [("down triangles", "first-onset rule"), ("diamonds", "driving-burst rule"),
         ("dashed vertical lines", "earliest mechanical onset"),
         ("dashed horizontal line in (c)", "rest threshold (0 dB)"),
         ("grey lines in (d)", "IMU angular-velocity onset"),
         ("shading in (b), grey columns in (e)", "samples excluded by the analysis mask"),
         ("colour in (e)", "power spectral density")],
        f"n = 1 participant, one channel; onsets found: {n_on['first']} (first onset), "
        f"{n_on['drive']} (driving burst)",
        "single channel-cycles, no summary statistic; MNF and MDF of the whole task",
        evidence="descriptive",
        notes=["The EMG onsets and the mechanical onsets are those of the timing analysis; "
               "the detector signal and threshold are recomputed here by the same routine "
               "for display"]
        + ([] if sg["session"] else ["No recorded session: the stored example window is "
                                     "shown and the rest threshold is computed from the "
                                     "stored rest example only"])))


def _onset_qc(req: ViewRequest, pid: str, task: str) -> FigureSpec:
    """A stratified random sample (study seed) of channel-cycles of one participant: per
    task and usability stratum ``QC_PER_STRATUM`` channel-cycles, each with the detector
    signal around the mechanical onset, the threshold and both rule onsets."""
    ctx = req.ctx
    fid, title = "emg_signal_spectrum", "EMG onset check"
    who = _who(ctx, pid)
    rule0 = _primary_rule(ctx)
    burst_min = float(_tp(ctx, "burst_min_db", default=6.0))
    pre_s = float(_tp(ctx, "emg_pre_s", default=0.6))
    post_s = float(_tp(ctx, "emg_post_s", default=0.2))
    rows = _onset_rows(ctx, pid)
    if rows.empty:
        return _empty_spec(fid, title, f"no stored onsets for {who}", "EMG onset check")
    rng = np.random.default_rng([B.seed_of(ctx), int(ctx.pseudonym(pid))])
    picks = []
    for tk in CYCLE_TASKS:
        for usable in (True, False):
            g = rows[(rows.task == tk) & (rows[f"usable_{rule0}"] == usable)]
            if len(g):
                picks.append(g.iloc[np.sort(rng.choice(len(g), min(QC_PER_STRATUM, len(g)),
                                                       replace=False))])
    sample = pd.concat(picks, ignore_index=True) if picks else rows.iloc[:0]
    op = _onset_params(ctx)
    cache: dict[tuple[str, str], Any] = {}
    panels: list[Panel] = []
    n_session = 0
    for r in sample.itertuples():
        key = (r.task, r.muscle)
        if key not in cache:
            sg = _signal(ctx, pid, r.task, r.muscle)
            cache[key] = (sg, _threshold(sg["rest"], sg["fs"], op) if sg else np.nan)
        sg, thr = cache[key]
        tl_ = V.task_label(r.task)
        ttl = (f"{tl_[:1].upper() + tl_[1:]}, {V.term('direction', r.direction)}: "
               f"{MUSCLE_LABEL.get(r.muscle, r.muscle)}, cycle {int(r.cycle) + 1}")
        outcome = _reason(r, rule0, burst_min)
        p = Panel(ttl, Axis("Time re mechanical onset", V.MSEC, lim=(-1e3 * pre_s, 1e3 * post_s)),
                  Axis(f"{_det_label()} re threshold", "\\decibel"), [], legend="none")
        tm = float(r.t_mech_us)
        if sg is None or not (sg["t_us"][0] <= tm - pre_s * 1e6
                              and tm + post_s * 1e6 <= sg["t_us"][-1]):
            panels.append(_empty_panel(ttl, "window not in the recorded signal"))
            continue
        n_session += int(sg["session"])
        sel = (sg["t_us"] >= tm - pre_s * 1e6) & (sg["t_us"] <= tm + post_s * 1e6)
        x = np.where(sg["mask"][sel], np.nan, sg["cleaned"][sel])
        det = _detector_signal(x, sg["fs"], op)
        tt = (sg["t_us"][sel] - tm) / 1e3
        with np.errstate(divide="ignore", invalid="ignore"):
            y = 10 * np.log10(np.maximum(det / thr, 1e-2)) if np.isfinite(thr) and thr > 0 \
                else np.full(tt.shape, np.nan)
        k = _thin(tt, y, budget=1500)
        p.series.append(Series("line", tt[k], y[k], role=f"entity:{r.muscle}",
                               style={"width": 0.6}))
        p.series.append(Series("hline", y=np.array([0.0]), role="threshold"))
        p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        xo = (float(getattr(r, "t_omega_us", np.nan)) - tm) / 1e3
        if np.isfinite(xo) and -1e3 * pre_s <= xo <= 1e3 * post_s:      # inside the window
            _vlines(p, [xo], "muted")
        for rr in RULES:
            t_on = float(getattr(r, f"t_on_{rr}", np.nan))
            if np.isfinite(t_on) and -1e3 * pre_s <= (t_on - tm) / 1e3 <= 1e3 * post_s:
                p.series.append(Series("scatter", np.array([(t_on - tm) / 1e3]), np.array([0.0]),
                                       role=RULE_ROLE[rr], style={"ms": RULE_MS[rr],
                                                                  "size": 1.6}))
        b = float(getattr(r, "burst_db", np.nan))
        p.note = f"{outcome}; " + (f"burst {b:.0f} dB" if np.isfinite(b)
                                   else "burst not measured")
        p.y.lim = (-20.0, max(20.0, _nice(float(np.nanmax(y)) if np.isfinite(y).any() else 1.0)))
        panels.append(p)
    if not panels:
        return _empty_spec(fid, title, f"no channel-cycles for {who}", "EMG onset check")
    cols = 2
    _grid_labels(panels, cols)
    spec = FigureSpec(fid + "_qc", title, panels, rows=int(np.ceil(len(panels) / cols)),
                      cols=cols, width="text", height_in=1.05,
                      sources=["timing:onsets", "session:emg"],
                      perspective=f"{who}, random channel-cycles",
                      meta={"pid": pid, "sample": sample[["task", "cycle", "muscle"]]
                            .to_dict("records")})
    spec.id = "emg_signal_spectrum"
    return attach(spec, CaptionSpec(
        f"{who}: a stratified random sample of channel-cycles (per task, {QC_PER_STRATUM} usable "
        f"and {QC_PER_STRATUM} not usable under the {RULE_LABEL[rule0]} rule, drawn with the "
        f"study seed) for checking the onsets by eye: the {_det_label()} detector signal in "
        f"dB re the channel's rest threshold, in the onset search window ({-1e3 * pre_s:g} to "
        f"+{1e3 * post_s:g} ms around the mechanical onset); each panel gives the cycle's "
        "usability or the first gate it fails (in the panel's corner, with the burst size)",
        [("down triangles", "first-onset rule"), ("diamonds", "driving-burst rule"),
         ("dashed vertical line", "earliest mechanical onset"),
         ("grey vertical line", "IMU angular-velocity onset"),
         ("dashed horizontal line", "rest threshold")],
        f"n = {len(panels)} channel-cycles of 1 participant ({len(rows)} recorded)",
        "single channel-cycles, no summary statistic", evidence="descriptive",
        notes=[f"Usable: an onset found, not active at the window start, a burst at least "
               f"{burst_min:g} dB above rest, and enough unmasked samples in the window"]))


# == 2. cycle-normalised spectrogram ===========================================================
def _cycle_spectra(ctx: Any, task: str, pids: list[str] | None = None
                   ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Per participant the stored per-window spectral shape (log10 power share on the cohort
    frequency grid) averaged within ``PHASE_BINS`` cycle-phase bins: ``(grid_hz, {pid:
    (6 muscles in MUSCLES order, PHASE_BINS, bins)}, info)``. The window centre gives its
    cycle phase; window length from the frequency resolution, hop from the window count and
    the phase duration. Channels not usable for the spectral shape stay NaN."""
    st = _a(ctx, "stats_emg", "_cohort")
    grid = np.asarray(st.get("spectral__grid_hz", np.array([])), float)
    shape_ok = _shape_ok(ctx)
    out: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"win_s": np.nan, "hop_s": np.nan, "dur_s": [], "n_excluded": 0}
    for pid in pids if pids is not None else _pids(ctx):
        e = _a(ctx, "emg", pid)
        key = f"spectral__{task}__win"
        if key not in e or f"{task}__t_us" not in e or "spectral__freq" not in e:
            continue
        W = np.asarray(e[key], float)                       # (C, S, bins)
        if not grid.size:
            grid = np.arange(W.shape[2], dtype=float)
        names = _names(ctx, pid, e)
        fq = np.asarray(e["spectral__freq"], float)
        L = 1.0 / float(np.median(np.diff(fq)))
        tg = np.asarray(e[f"{task}__t_us"], np.int64)
        T = (tg[-1] - tg[0]) / 1e6
        S = W.shape[1]
        hop = (T - L) / (S - 1) if S > 1 else L
        info["win_s"], info["hop_s"] = L, hop
        tc = tg[0] + (np.arange(S) * hop + L / 2) * 1e6
        cyc = _cycles(ctx, pid, task)
        if cyc.empty:
            continue
        ph = np.full(S, np.nan)
        for c in cyc.itertuples():
            m = (tc >= c.t_start_us) & (tc < c.t_end_us)
            ph[m] = (tc[m] - c.t_start_us) / (c.t_end_us - c.t_start_us)
            info["dur_s"].append((c.t_end_us - c.t_start_us) / 1e6)
        b = np.minimum(np.floor(ph * PHASE_BINS), PHASE_BINS - 1)
        A = np.full((len(MUSCLES), PHASE_BINS, W.shape[2]), np.nan)
        ok = shape_ok[(shape_ok.pid == pid) & (shape_ok.task == task)].set_index("muscle") \
            ["usable_shape"] if not shape_ok.empty else pd.Series(dtype=bool)
        for mi, m in enumerate(MUSCLES):
            if m not in names:
                continue
            if len(ok) and not bool(ok.get(m, True)):
                info["n_excluded"] += 1
                continue
            Wm = W[names.index(m)]
            for k in range(PHASE_BINS):
                sel = (b == k) & np.isfinite(Wm).all(axis=1)
                if sel.any():
                    A[mi, k] = Wm[sel].mean(axis=0)
        out[pid] = A
    return grid, out, info


def _group_activation(ctx: Any, task: str, muscles: tuple[str, ...]
                      ) -> tuple[np.ndarray, float]:
    """Cohort mean of the per-participant mean cycle envelope of ``muscles`` (101 points,
    % of max recorded) and the cohort mean of their rest-phase 95th percentile."""
    curves, rest = [], []
    rp = _rest_p95(ctx)
    for pid in _pids(ctx):
        e = _a(ctx, "emg", pid)
        k = f"profile__{task}__pct"
        if k not in e:
            continue
        names = [str(n) for n in e.get("profile__names", _names(ctx, pid, e))]
        cols = [names.index(m) for m in muscles if m in names]
        if not cols:
            continue
        P = np.asarray(e[k], float)[:, :, cols]
        curves.append(_nanmean(_nanmean(P, axis=0), axis=1))
        if not rp.empty and pid in rp.index:
            rest.append(float(_nanmean(rp.loc[pid].reindex(list(muscles)).to_numpy(float))))
    act = _nanmean(np.array(curves), axis=0) if curves else np.full(101, np.nan)
    return act, float(_nanmean(np.array(rest))) if rest else float("nan")


def _mnf(grid: np.ndarray, logshare: np.ndarray) -> np.ndarray:
    """Mean frequency of each row of log10 power shares on ``grid``."""
    s = 10 ** np.asarray(logshare, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (s * grid[None, :]).sum(axis=1) / s.sum(axis=1)


@register("emg_cycle_spectrogram", "dashboard", "Cycle-normalised EMG spectrogram",
          params=("phase",), phase="open_close", phases=CYCLE_TASKS, domain="emg_freq",
          choices={"form": ("group", "change", "muscle")}, form="group")
def emg_cycle_spectrogram(req: ViewRequest) -> FigureSpec:
    """Spectral shape over the normalised cycle (cycle phase × frequency), mean over the
    cycles of a participant and then over participants, per muscle group or muscle."""
    ctx = req.ctx
    fid, title = "emg_cycle_spectrogram", "Cycle-normalised EMG spectrogram"
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    form = req.options.get("form", "group")
    grid, per, info = _cycle_spectra(ctx, task)
    if not per:
        return _empty_spec(fid, title, "no per-window spectra are stored",
                           "cycle-normalised spectrogram")
    pids = sorted(per)
    Aall = np.stack([per[p] for p in pids])               # (P, 6, bins, F)
    units = ([(m, (m,)) for m in MUSCLES] if form == "muscle"
             else [(g, ms) for g, ms in GROUPS])
    change = form == "change"
    xs = (np.arange(PHASE_BINS) + 0.5) * 100.0 / PHASE_BINS
    df = float(np.median(np.diff(grid))) if grid.size > 1 else 1.0
    imgs, mnfs, n_used = [], [], []
    for name, ms in units:
        idx = [MUSCLES.index(m) for m in ms]
        G = 10.0 * _nanmean(Aall[:, idx], axis=1)          # (P, bins, F) dB, group mean per pid
        n_used.append(int(np.isfinite(G).any(axis=(1, 2)).sum()))
        mnfs.append(_mnf(grid, _nanmean(G, axis=0) / 10.0))
        if change:                                         # re the participant's cycle mean
            G = G - _nanmean(G, axis=1)[:, None, :]
        imgs.append(_nanmean(G, axis=0))
    if change:
        m = max(abs(_q(z, 1)) for z in imgs) if imgs else 1.0
        m = max(m, max(abs(_q(z, 99)) for z in imgs))
        zl = (-round(m + 0.05, 1), round(m + 0.05, 1))
        ramp, cb_lab = "diverging", "Change re cycle mean"
    else:
        zl = (np.floor(min(_q(z, 2) for z in imgs)), np.ceil(max(_q(z, 99.5) for z in imgs)))
        ramp, cb_lab = "sequential", "Power share"
    panels = []
    letters = "abcdef"
    n_quiet = 0
    for k, ((name, ms), Z) in enumerate(zip(units, imgs)):
        act, rest = _group_activation(ctx, task, ms)
        ab = np.array([_nanmean(act[int(round(100 * i / PHASE_BINS)):
                                    int(round(100 * (i + 1) / PHASE_BINS)) + 1])
                       for i in range(PHASE_BINS)])
        quiet = (ab < rest) if np.isfinite(rest) else np.zeros(PHASE_BINS, bool)
        img = Z.T[::-1]                                     # rows: high frequency first
        lab = MUSCLE_LABEL[name] if form == "muscle" else GROUP_LABEL[name]
        y0, y1 = float(grid[0] - df / 2), float(grid[-1] + df / 2)
        p = Panel(f"({letters[k]}) {lab[:1].upper() + lab[1:]}",
                  Axis("Movement cycle", V.PCT, lim=(0.0, 100.0)),
                  Axis("Frequency", V.HZ, lim=(y0, y1)),
                  [Series("image", img=img, extent=(0.0, 100.0, y0, y1),
                          style={"cmap": ramp, "lim": zl})],
                  legend="below", colorbar=Colorbar(cb_lab, "\\decibel", lim=zl, ramp=ramp))
        p.series.append(Series("line", xs, mnfs[k], role="accent",
                               label="mean frequency" if k == len(units) - 1 else None,
                               style={"width": 1.3}))
        from biomech.viz.views.data import _runs

        runs = _runs(quiet)
        if runs:
            n_quiet += 1
            p.series.append(Series("sigbar", np.array([100.0 * a / PHASE_BINS for a, _ in runs]),
                                   y2=np.array([100.0 * b / PHASE_BINS for _, b in runs]),
                                   role="muted", label="no activity" if n_quiet == 1 else None,
                                   style={"row": 0}))
            p.y.lim = (y0, y1 + 0.12 * (y1 - y0))      # room for the bar above the image
        _phase_ticks(p)
        panels.append(p)
    cols = 3 if form == "muscle" else 2
    _grid_labels(panels, cols)
    # One scale for all panels: the colour bar only right of each row's last panel, the same
    # y top (room for the no-activity bar) and frequency ticks below that bar.
    top = max(p.y.lim[1] for p in panels)
    step_hz = 100.0 if y1 - y0 > 250 else 50.0
    for k, p in enumerate(panels):
        p.y.lim = (y0, top)
        p.y.ticks = [float(v) for v in np.arange(0.0, y1 - 0.3 * step_hz, step_hz)
                     if v >= y0]
        if (k + 1) % cols and k + 1 < len(panels):
            p.colorbar = None
    dur = np.median(info["dur_s"]) if info["dur_s"] else np.nan
    win, hop = info["win_s"], info["hop_s"]
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=1.2 if form == "muscle" else 1.45,
                      sources=["emg:arrays", "emg:spectral", "kinematics:cycles",
                               "stats_emg:arrays", "emg:activation"],
                      perspective=f"cohort, {TASK_LABEL.get(task, task)}",
                      meta={"n": n_used, "win_s": win, "hop_s": hop, "median_cycle_s": dur})
    coarse = (f"The windows ({win:.2g} s) are long against the median cycle ({dur:.2g} s), so "
              "the phase axis is strongly smoothed" if np.isfinite(dur) and win > 0.4 * dur
              else f"Windows of {win:.2g} s against a median cycle of {dur:.2g} s")
    return attach(spec, CaptionSpec(
        f"{TASK_LABEL.get(task, task).capitalize()}: spectral shape of the cleaned EMG over the "
        "time-normalised cycle (power share per frequency bin in dB, from the stored "
        "per-window spectra placed at their window centre's cycle phase in "
        f"{PHASE_BINS} phase bins); mean over the cycles of a participant, then over "
        "participants" + ("" if form == "muscle" else
                          "; per participant the group's muscles averaged first")
        + ("; shown as the change re each participant's mean over the cycle, so a shift of "
           "the spectrum across the cycle stands out" if change else ""),
        [("colour", "change of the power share re the cycle mean" if change
          else "power share per frequency bin"),
         ("black line", "mean frequency of the mean spectrum"),
         ("grey bar above a panel", "phases where the mean envelope stays below its "
                                    "rest-phase 95th percentile (no activity: the shape there "
                                    "is the noise floor)")],
        f"n = {max(n_used) if n_used else 0} participants with spectral-shape-usable channels"
        + (f" ({info['n_excluded']} channels not usable for the shape left out)"
           if info["n_excluded"] else ""),
        "mean over cycles, then mean over participants", evidence="exploratory",
        notes=[f"{coarse}; window step {hop:.2g} s"]))


# == 3. onset raster ===========================================================================
def _aligned_env(ctx: Any, rows: pd.DataFrame, lags_ms: np.ndarray) -> np.ndarray:
    """Envelope (% of max recorded, NaN = masked) of each channel-cycle row sampled at
    t_mech + lags (stored envelope grid, linear interpolation)."""
    out = np.full((len(rows), lags_ms.size), np.nan)
    for pid, g in rows.groupby("pid", sort=False):
        e = _a(ctx, "emg", pid)
        names = _names(ctx, pid, e)
        for task, gt in g.groupby("task", sort=False):
            if f"{task}__t_us" not in e:
                continue
            t = np.asarray(e[f"{task}__t_us"], float)
            N = np.asarray(e[f"{task}__norm_pct"], float).copy()
            if f"{task}__masked" in e:
                N[np.asarray(e[f"{task}__masked"], bool)] = np.nan
            for i, r in zip(gt.index, gt.itertuples()):
                if r.muscle not in names:
                    continue
                tq = float(r.t_mech_us) + lags_ms * 1e3
                y = np.interp(tq, t, N[:, names.index(r.muscle)], left=np.nan, right=np.nan)
                out[rows.index.get_loc(i)] = y
    return out


@register("emg_onset_raster", "dashboard", "EMG onset raster", params=("pid", "phase"),
          phase="open_close", phases=CYCLE_TASKS, domain="timing",
          choices={"form": ("cohort", "participant")}, form="cohort")
def emg_onset_raster(req: ViewRequest) -> FigureSpec:
    """Channel-cycles × time image of the envelope around the mechanical onset, sorted by the
    lead, with the onsets of both rules (cohort, or one participant)."""
    ctx = req.ctx
    fid, title = "emg_onset_raster", "EMG onset raster"
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    form = req.options.get("form", "cohort")
    pid = _pid(req) if form == "participant" else None
    rows = _onset_rows(ctx, pid, task)
    if rows.empty:
        return _empty_spec(fid, title, "no stored onsets", "onset raster")
    rule0 = _primary_rule(ctx)
    other = [r for r in RULES if r != rule0][0]
    pre = 1e3 * float(_tp(ctx, "emg_pre_s", default=0.6))
    post = 1e3 * float(_tp(ctx, "emg_post_s", default=0.2))
    e0 = _a(ctx, "emg", rows.pid.iloc[0])
    tg = np.asarray(e0.get(f"{task}__t_us", np.array([0, 10000])), float)
    step = float(np.median(np.diff(tg))) / 1e3 if tg.size > 1 else 10.0
    lags = np.arange(-pre, post + step / 2, step)
    panels, ns = [], {}
    for k, (direction, grp) in enumerate((("opening", "openers"), ("closing", "closers"))):
        d = rows[rows.direction == direction].copy()
        ttl = f"({'ab'[k]}) {direction.capitalize()}: {GROUP_SHORT[grp]}"
        if d.empty:
            panels.append(_empty_panel(ttl, "no channel-cycles"))
            continue
        u = d[f"usable_{rule0}"].to_numpy(bool)
        lead0 = d[f"lead_{rule0}"].to_numpy(float)
        lead1 = d[f"lead_{other}"].to_numpy(float)
        order = np.lexsort((-np.nan_to_num(lead1, nan=-1e9), -np.nan_to_num(lead0, nan=-1e9),
                            ~u))
        d = d.iloc[order].reset_index(drop=True)
        u = u[order]
        Y = _aligned_env(ctx, d, lags)
        with np.errstate(invalid="ignore", divide="ignore"):
            pk = np.nanmax(np.where(np.isfinite(Y), Y, -np.inf), axis=1)
            Yn = 100.0 * Y / np.where(pk > 0, pk, np.nan)[:, None]
        n = len(d)
        ns[direction] = (n, int(u.sum()))
        p = Panel(ttl, Axis("Time re mechanical onset", V.MSEC, lim=(-pre, post)),
                  Axis("Channel-cycles, sorted by lead", lim=(-0.5, n - 0.5)),
                  [Series("image", img=Yn, extent=(-pre - step / 2, post + step / 2, -0.5,
                                                   n - 0.5),
                          style={"cmap": "sequential", "lim": (0.0, 100.0)})],
                  legend="below", colorbar=Colorbar("Envelope re row peak", V.PCT,
                                                    lim=(0.0, 100.0)))
        p.y.ticks, p.y.ticklabels = [], []
        # image row 0 is drawn at the top: row i sits at y = n - 1 - i in data units
        ydat = n - 1 - np.arange(n, dtype=float)
        for r in RULES:
            x = -d[f"lead_{r}"].to_numpy(float)
            m = np.isfinite(x)
            if m.any():
                p.series.append(Series("scatter", x[m], ydat[m], role=RULE_ROLE[r],
                                       style={"ms": "|" if r == "drive" else "v",
                                              "size": 0.55 if n > 60 else 1.0}))
                # legend key at full size: the data marks are shrunk for dense rasters
                p.series.append(Series("scatter", np.array([np.nan]), np.array([np.nan]),
                                       role=RULE_ROLE[r], label=f"{RULE_LABEL[r]} rule",
                                       style={"ms": "|" if r == "drive" else "v",
                                              "size": 1.2}))
        if "t_omega_us" in d:
            xo = (d.t_omega_us.to_numpy(float) - d.t_mech_us.to_numpy(float)) / 1e3
            m = np.isfinite(xo)
            if m.any():
                p.series.append(Series("scatter", xo[m], ydat[m], role="faint",
                                       style={"ms": "|", "size": 0.5 if n > 60 else 0.9}))
                p.series.append(Series("scatter", np.array([np.nan]), np.array([np.nan]),
                                       role="faint", label="IMU velocity onset",
                                       style={"ms": "|", "size": 1.2}))
        p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
        nu = int(u.sum())
        if 0 < nu < n:
            # unlabelled: an in-plot label covers rows; the caption names the line
            p.series.append(Series("hline", y=np.array([n - nu - 0.5]), role="reference"))
        p.y.lim = (-0.5, n - 0.5)
        p.y.label = "Channel-cycles, sorted by lead"
        panels.append(p)
    who = _who(ctx, pid) if pid else "cohort"
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text",
                      height_in=3.2 if pid is None else 2.0,
                      sources=["timing:onsets", "emg:arrays"],
                      perspective=f"{who}, {TASK_LABEL.get(task, task)}",
                      meta={"n": ns, "rule": rule0})
    n_txt = "; ".join(f"{dr}: {a} channel-cycles, {b} usable" for dr, (a, b) in ns.items())
    n_p = rows.pid.nunique()
    return attach(spec, CaptionSpec(
        f"{TASK_LABEL.get(task, task).capitalize()}, {who}: the envelope of every channel-cycle "
        f"({_env_ms(ctx):g} ms RMS, % of its own peak in the onset search window, "
        f"{-pre:g} to +{post:g} ms around the earliest mechanical onset) as one image row; "
        f"openers for the opening, closers for the closing. Rows usable under the "
        f"{RULE_LABEL[rule0]} rule come first, sorted by their lead under that rule (earliest "
        "EMG at the top), then the rows not usable, sorted the same way",
        [("colour", "envelope re the row's peak"), ("down triangles", "first-onset rule onset"),
         ("dark ticks", "driving-burst rule onset"), ("light ticks", "IMU angular-velocity onset"),
         ("dashed vertical line", "earliest mechanical onset"),
         ("dashed horizontal line", f"border between the rows usable under the "
                                    f"{RULE_LABEL[rule0]} rule (above) and the others")],
        f"n = {n_p} participants; {n_txt}",
        f"single channel-cycles, no summary statistic ({_det_label()} detector)",
        evidence="descriptive",
        notes=["The two rules differ where the triangle and the tick of a row are apart: an "
               "early burst unrelated to the movement moves the first onset, not the driving "
               "burst"]))


# == 4. EMG-kinematics coupling =================================================================
def _coupling_series(ctx: Any, pid: str, task: str) -> dict[str, Any] | None:
    """Closer envelope and -alpha of one participant and task on the IMU grid, both through
    the coupling stage's low-pass and residualised over cycle phase (``timing.coupling``),
    with the occlusions masked; the EMG feature is the stored envelope (not the rectified
    raw EMG of the stored test)."""
    from biomech import timebase
    from biomech.timing import coupling as CP

    imu = _a(ctx, "imu", pid)
    e = _a(ctx, "emg", pid)
    if f"{task}__theta_quat" not in imu or f"{task}__t_us" not in e:
        return None
    t = np.asarray(imu[f"{task}__t_us"], np.int64)
    th = np.asarray(imu[f"{task}__theta_quat"], float)
    if t.size < 50:
        return None
    alpha = timebase.gradient(t, timebase.gradient(t, th))
    env = _env_grid(e, task, _names(ctx, pid, e), CLOSERS)
    if env is None:
        return None
    te, ye = env
    emg = np.interp(t, te, ye, left=np.nan, right=np.nan)
    cyc = _cycles(ctx, pid, task)
    if len(cyc) < 3:
        return None
    c2e = _tp(ctx, "c2e", default={}) or {}
    p = CP.CouplingParams.from_dict(c2e.get("coupling"))
    fc, order = float(c2e.get("alpha_cutoff_hz", 15.0)), int(c2e.get("alpha_order", 4))
    st = cyc.t_start_us.to_numpy(float)
    en = np.r_[st[1:], cyc.t_end_us.to_numpy(float)[-1:]]
    a = -np.asarray(alpha, float)
    for t_occ in cyc.t_end_us.to_numpy(float):
        a[np.abs(t - t_occ) <= p.occlusion_mask_s * 1e6] = np.nan
    # The stored envelope is a centred moving RMS: the acceleration gets the same centred
    # window (mask-aware moving mean), so both carry about the same smoothing and the lag of
    # the correlation peak is not biased by unequal filters.
    from scipy.ndimage import uniform_filter1d

    fs = 1e6 / float(np.median(np.diff(t)))
    w = max(int(round(_env_ms(ctx) / 1e3 * fs)) // 2 * 2 + 1, 1)
    valid = np.isfinite(a)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = uniform_filter1d(np.where(valid, a, 0.0), w, mode="nearest")
        den = uniform_filter1d(valid.astype(float), w, mode="nearest")
        a = np.where(den >= 0.5, num / den, np.nan)
    eg = CP.lowpass(t, emg, fc, order)
    ag = CP.lowpass(t, a, fc, order)
    er = CP.residualise(t, eg, st, en, p.n_phase_bins)
    ar = CP.residualise(t, ag, st, en, p.n_phase_bins)
    dt_ms = float(np.median(np.diff(t))) / 1e3
    return {"t": t, "emg": er, "acc": ar, "dt_ms": dt_ms, "p": p, "fc": fc}


def _coupling_participant(req: ViewRequest) -> FigureSpec:
    """One participant, the coupling test's own feature: the rectified closer EMG of the
    recorded session (``timing.coupling.emg_feature``) against -alpha through the test's
    low-pass, occlusion mask and residualisation; the whole lag curve (residualised and raw)
    with the stored peak lag."""
    from biomech import timebase
    from biomech.timing import coupling as CP

    ctx = req.ctx
    fid, title = "emg_motion_coupling", "EMG–motion coupling"
    pid = _pid(req)
    if pid is None:
        return _empty_spec(fid, title, "there are no participants", "EMG–motion coupling")
    who = _who(ctx, pid)
    c2e = _tp(ctx, "c2e", default={}) or {}
    p = CP.CouplingParams.from_dict(c2e.get("coupling"))
    fc, order = float(c2e.get("alpha_cutoff_hz", 15.0)), int(c2e.get("alpha_order", 4))
    cp = _t(ctx, "timing", "coupling", [pid])
    imu = _a(ctx, "imu", pid)
    panels, peaks = [], {}
    try:
        prep = ctx.emg(pid)
    except Exception:
        prep = None
    for k, task in enumerate(CYCLE_TASKS):
        tl = TASK_LABEL.get(task, task)
        ttl = f"({'ab'[k]}) {tl[:1].upper() + tl[1:]}"
        cyc = _cycles(ctx, pid, task)
        if prep is None or f"{task}__theta_quat" not in imu or len(cyc) < 3:
            panels.append(_empty_panel(ttl, "needs the recorded session and the IMU angle"))
            continue
        pe = prep.phase(task)
        names = list(prep.names)
        cols = [names.index(m) for m in CLOSERS if m in names]
        x = np.where(pe.mask, np.nan, pe.cleaned)
        t = np.asarray(imu[f"{task}__t_us"], np.int64)
        alpha = timebase.gradient(t, timebase.gradient(t, np.asarray(imu[f"{task}__theta_quat"],
                                                                     float)))
        emg = CP.emg_feature(np.asarray(pe.t_us), x[:, cols], t, float(pe.fs), p.smooth_s)
        st = cyc.t_start_us.to_numpy(float)
        en = np.r_[st[1:], cyc.t_end_us.to_numpy(float)[-1:]]
        e = CP.lowpass(t, emg, fc, order)
        a = -np.asarray(alpha, float)
        for t_occ in cyc.t_end_us.to_numpy(float):
            a[np.abs(t - t_occ) <= p.occlusion_mask_s * 1e6] = np.nan
        a = CP.lowpass(t, a, fc, order)
        dt_ms = float(np.median(np.diff(t))) / 1e3
        kk = int(round(p.max_lag_ms / dt_ms))
        er = CP.residualise(t, e, st, en, p.n_phase_bins)
        ar = CP.residualise(t, a, st, en, p.n_phase_bins)
        lg, c = CP.lagged_corr(er, ar, kk, p.min_pairs, center=True)
        _, c_raw = CP.lagged_corr(e, a, kk, p.min_pairs, center=True)
        pn = Panel(ttl, Axis("Lag (> 0: EMG first)", V.MSEC), Axis("Correlation"),
                   [Series("line", lg * dt_ms, c_raw, role="context",
                           label="without the cycle-phase mean removed"),
                    Series("line", lg * dt_ms, c, role="group:closers",
                           label="cycle-phase mean removed (the test)"),
                    Series("vline", x=np.array([0.0]), role="reference")], legend="below")
        r = cp[cp.task == task] if not cp.empty and "task" in cp else pd.DataFrame()
        if not r.empty and np.isfinite(float(r.emd_lag_ms.iloc[0])):
            lag0 = float(r.emd_lag_ms.iloc[0])
            peaks[task] = lag0
            pn.series.append(Series("vline", x=np.array([lag0]), role="group:closers"))
            null = float(r.emd_r_null_median.iloc[0]) if "emd_r_null_median" in r else np.nan
            if np.isfinite(null):
                pn.series.append(Series("hline", y=np.array([null]), role="threshold"))
                pn.series.append(Series("line", np.array([np.nan]), np.array([np.nan]),
                                        role="threshold", label="null median of the peak"))
            pv = float(r.p_coupling.iloc[0]) if "p_coupling" in r else np.nan
            pn.note = f"stored peak {lag0:+.0f} ms" + (f", p = {pv:.3f}" if np.isfinite(pv) else "")
        panels.append(pn)
    _grid_labels(panels, 2)
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=1.7,
                      sources=["session:emg", "imu:arrays", "kinematics:cycles", "timing:coupling"],
                      perspective=f"{who}, open-close and chewing", meta={"pid": pid,
                                                                          "stored_lag": peaks})
    return attach(spec, CaptionSpec(
        f"{who}: lagged Pearson correlation of the rectified closer EMG (mean of the closer "
        f"channels, centred {1e3 * p.smooth_s:g} ms moving mean) with the closing angular "
        f"acceleration (minus the second derivative of the IMU opening angle), both through the "
        f"same zero-phase low-pass ({fc:g} Hz), the acceleration masked ±{1e3 * p.occlusion_mask_s:g}"
        " ms around occlusion, with and without the mean waveform over cycle phase removed "
        f"({p.n_phase_bins} phase bins), as in the stored coupling test",
        [("blue line", "cycle-phase mean removed (the tested curve)"),
         ("grey line", "without removing it"), ("blue vertical line", "stored peak lag"),
         ("dashed horizontal line", "median peak correlation of the circular-shift null")],
        "n = 1 participant, all cycles of each task",
        "lag of the peak correlation; Monte-Carlo p of the peak against the circular-shift null "
        "(stored)", evidence="exploratory"))


@register("emg_motion_coupling", "dashboard", "EMG–motion coupling", domain="timing",
          params=("pid",), choices={"form": ("xcorr", "coherence", "participant")},
          form="xcorr")
def emg_motion_coupling(req: ViewRequest) -> FigureSpec:
    """Lagged correlation (or coherence and lag over frequency) of the closer envelope with
    the closing angular acceleration, per task, with the stored coupling lags; or one
    participant's lag curve from the rectified EMG as in the coupling test."""
    from scipy.signal import csd, welch

    from biomech.timing.coupling import lagged_corr, peak_lag

    ctx = req.ctx
    fid, title = "emg_motion_coupling", "EMG–motion coupling"
    form = req.options.get("form", "xcorr")
    if form == "participant":
        return _coupling_participant(req)
    cp = _t(ctx, "timing", "coupling")
    seed = B.seed_of(ctx)
    panels, ns, lag_txt, peak_txt = [], {}, [], []
    same_band = False
    max_lag = None
    for k, task in enumerate(CYCLE_TASKS):
        curves, lags_ms, cohs, phlags, fgrid = [], None, [], [], None
        for pid in _pids(ctx):
            s = _coupling_series(ctx, pid, task)
            if s is None:
                continue
            p = s["p"]
            max_lag = p.max_lag_ms
            kk = int(round(p.max_lag_ms / s["dt_ms"]))
            if form == "xcorr":
                lg, c = lagged_corr(s["emg"], s["acc"], kk, p.min_pairs, center=True)
                if np.isfinite(c).any():
                    curves.append(c)
                    lags_ms = lg * s["dt_ms"]
            else:
                ok = np.isfinite(s["emg"]) & np.isfinite(s["acc"])
                if ok.sum() < 4 * p.min_pairs:
                    continue
                fs = 1e3 / s["dt_ms"]
                xa = np.where(ok, s["emg"], 0.0)
                xb = np.where(ok, s["acc"], 0.0)
                nper = int(round(COH_SEG_S * fs))
                if ok.sum() < 2 * nper:
                    continue
                f, Pab = csd(xa, xb, fs, nperseg=nper)
                _, Paa = welch(xa, fs, nperseg=nper)
                _, Pbb = welch(xb, fs, nperseg=nper)
                m = (f > 0) & (f <= s["fc"])
                with np.errstate(invalid="ignore", divide="ignore"):
                    coh = np.abs(Pab[m]) ** 2 / (Paa[m] * Pbb[m])
                    # phase of the cross spectrum: acceleration after the EMG by lag(f)
                    lag = -np.angle(Pab[m]) / (2 * np.pi * f[m]) * 1e3
                cohs.append(coh)
                phlags.append(np.where(coh >= COH_MIN, lag, np.nan))
                fgrid = f[m]
        stored = cp[cp.task == task] if not cp.empty and "task" in cp else pd.DataFrame()
        sl = stored.emd_lag_ms.to_numpy(float) if not stored.empty else np.array([])
        sl = sl[np.isfinite(sl)]
        tl = TASK_LABEL.get(task, task)
        if form == "xcorr":
            p = Panel(f"({'ab'[k]}) {tl[:1].upper() + tl[1:]}",
                      Axis("Lag (> 0: EMG first)", V.MSEC),
                      Axis("Correlation"), [], legend="below")
            if curves:
                C = np.array(curves)
                ns[task] = len(C)
                B.add_curves_ci(p, lags_ms, C, centre="median", seed=seed, label="median")
                bands = [q for q in p.series if q.kind == "band"]
                if len(bands) == 2 and np.allclose(bands[0].y, bands[1].y, equal_nan=True) \
                        and np.allclose(bands[0].y2, bands[1].y2, equal_nan=True):
                    # with ~13 participants the bootstrap 95 % CI of the median falls on the
                    # same order statistics as the IQR: one band, named as both
                    p.series.remove(bands[0])
                    same_band = True
                p.series.append(Series("vline", x=np.array([0.0]), role="reference"))
                if sl.size:
                    top = float(np.nanmax(C)) * 1.05
                    p.series.append(Series("scatter", sl, np.full(sl.size, top), role="group:closers",
                                           label="stored lag per participant",
                                           style={"size": 0.9}))
                    p.series.append(Series("vline", x=np.array([float(np.median(sl))]),
                                           role="group:closers"))
                    lag_txt.append(f"{tl}: median {B.snum(np.median(sl), 0)} ms")
                # in the caption: a panel note sits on the lag dots at the top
                peak_txt.append(f"{tl} {B.snum(peak_lag(lags_ms, _nanmedian(C))[0], 0)} ms")
            else:
                p = _empty_panel(p.title, "no participant with the inputs")
            panels.append(p)
        else:
            pa = Panel(f"({'abcd'[2 * k]}) {tl[:1].upper() + tl[1:]}: coherence",
                       Axis("Frequency", V.HZ), Axis("Coherence", lim=(0.0, 1.0)), [],
                       legend="none")
            pb = Panel(f"({'abcd'[2 * k + 1]}) {tl[:1].upper() + tl[1:]}: lag over frequency",
                       Axis("Frequency", V.HZ), Axis("Lag (> 0: EMG first)", V.MSEC), [],
                       legend="below")
            if cohs:
                ns[task] = len(cohs)
                B.add_curves(pa, fgrid, np.array(cohs), centre="median")
                L = np.array(phlags)
                med = _nanmedian(L)
                for row in L:
                    pb.series.append(Series("line", fgrid, row, role="context"))
                pb.series.append(Series("line", fgrid, med, role="accent", label="median"))
                pb.series.append(Series("hline", y=np.array([0.0]), role="reference"))
                if sl.size:
                    lo_, hi_ = np.percentile(sl, [25, 75])
                    pb.series.append(Series("band", np.array([fgrid[0], fgrid[-1]]),
                                            np.array([lo_, lo_]), y2=np.array([hi_, hi_]),
                                            role="group:closers",
                                            label="stored lag, interquartile range"))
                    lag_txt.append(f"{tl}: median {B.snum(np.median(sl), 0)} ms")
                pb.y.lim = (-2 * (max_lag or 150.0), 2 * (max_lag or 150.0))
                panels += [pa, pb]
            else:
                panels += [_empty_panel(pa.title, "no participant with the inputs"),
                           _empty_panel(pb.title, "no participant with the inputs")]
    if not ns:
        return _empty_spec(fid, title, "the IMU opening angle or the envelopes are missing",
                           "EMG–motion coupling")
    cols = 2
    _grid_labels(panels, cols)
    c2e = _tp(ctx, "c2e", default={}) or {}
    fc = float(c2e.get("alpha_cutoff_hz", 15.0))
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=1.6,
                      sources=["imu:arrays", "emg:arrays", "kinematics:cycles", "timing:coupling"],
                      perspective="cohort, open-close and chewing", meta={"n": ns})
    n_txt = ", ".join(f"{TASK_LABEL.get(t, t)} {v}" for t, v in ns.items())
    what = ("lagged Pearson correlation" if form == "xcorr"
            else "magnitude-squared coherence and the lag implied by the cross-spectral phase "
                 f"(shown where the coherence is at least {COH_MIN:g}; Welch segments of {COH_SEG_S:g} s)")
    return attach(spec, CaptionSpec(
        f"Coupling of the closer envelope (mean of the closer channels, {_env_ms(ctx):g} ms RMS) "
        f"with the closing angular acceleration (minus the second derivative of the IMU "
        f"opening angle, smoothed with the same centred window as the envelope): both through "
        f"the same zero-phase low-pass ({fc:g} Hz), the acceleration masked around occlusion, "
        f"both with the mean waveform over cycle phase removed; per participant the {what}",
        ([("grey lines", "participants"), ("dark line", "median")]
         + ([("band", "95 % bootstrap CI of the median (participants resampled); with this n "
                      "it coincides with the IQR")] if form == "xcorr" and same_band else
            list(tuple(x.split(": ", 1)) for x in B.curve_ci_caption("median").split("; "))
            if form == "xcorr" else [])
         + [("blue dots and line" if form == "xcorr" else "blue band",
             "the stored peak lag of the coupling test, cycle-phase mean removed (per "
             "participant and their median)" if form == "xcorr" else
             "the stored peak lag of the coupling test, interquartile range over participants"),
            ("dashed line", "zero lag" if form == "xcorr" else "zero lag in (b, d)")]),
        f"n = {n_txt} participants",
        "median over participants" + (" with its 95 % bootstrap CI" if form == "xcorr" else "")
        + (f"; peak of the median curve: {', '.join(peak_txt)}" if peak_txt else ""),
        evidence="exploratory",
        notes=["Recomputed here from the stored envelopes for display; the stored lags come "
               "from the rectified EMG of the coupling test"
               + (f" ({'; '.join(lag_txt)})" if lag_txt else "")]))


# == 5. activation clock =======================================================================
def _group_curves(ctx: Any, task: str) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Per participant the peak-normalised mean cycle envelope of the openers and of the
    closers (rows = participants, 101 points, % of the curve's peak), as the openers-vs-
    closers SPM test builds them (channel mean, cycle mean, spectral-shape-usable channels,
    then the peak), and the cohort mean opening angle. The stored test arrays when present."""
    st = _a(ctx, "stats_emg", "_cohort")
    th = []
    for pid in _pids(ctx):
        e = _a(ctx, "emg", pid)
        if f"profile__{task}__theta_deg" in e:
            th.append(_nanmean(np.asarray(e[f"profile__{task}__theta_deg"], float), axis=0))
    theta = _nanmean(np.array(th), axis=0) if th else np.full(101, np.nan)
    if task == "open_close" and "spm2__openers" in st and "spm2__closers" in st:
        return ({"openers": 100.0 * np.asarray(st["spm2__openers"], float),
                 "closers": 100.0 * np.asarray(st["spm2__closers"], float)}, theta,
                "stats_emg:arrays")
    ok = _shape_ok(ctx)
    out: dict[str, list[np.ndarray]] = {"openers": [], "closers": []}
    for pid in _pids(ctx):
        e = _a(ctx, "emg", pid)
        k = f"profile__{task}__pct"
        if k not in e:
            continue
        names = [str(n) for n in e.get("profile__names", _names(ctx, pid, e))]
        good = set(ok[(ok.pid == pid) & ok.usable_shape.astype(bool)].muscle) \
            if not ok.empty else set(names)
        P = np.asarray(e[k], float)
        cur = {}
        for g, ms in GROUPS:
            cols = [names.index(m) for m in ms if m in names and m in good]
            if cols:
                y = _nanmean(_nanmean(P[:, :, cols], axis=2), axis=0)
                pk = float(np.nanmax(y)) if np.isfinite(y).any() else np.nan
                cur[g] = 100.0 * y / pk if pk > 0 else None
        if all(cur.get(g) is not None for g, _ in GROUPS):
            for g, _ in GROUPS:
                out[g].append(cur[g])
    return ({g: np.array(v) if v else np.empty((0, 101)) for g, v in out.items()}, theta,
            "emg:arrays")


def _polar(phase_pct: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clock coordinates: 0 % at the top, clockwise."""
    a = 2 * np.pi * np.asarray(phase_pct, float) / 100.0
    return np.asarray(r, float) * np.sin(a), np.asarray(r, float) * np.cos(a)


@register("emg_activation_clock", "dashboard", "Activation clock", params=("phase",),
          phase="open_close", phases=CYCLE_TASKS, domain="emg_time",
          choices={"form": ("clock", "curves")}, form="clock")
def emg_activation_clock(req: ViewRequest) -> FigureSpec:
    """Openers vs closers over the cycle phase as a clock (radius = activation), with the
    SPM clusters of the openers-vs-closers test as arcs; or unrolled with significance bars."""
    ctx = req.ctx
    fid, title = "emg_activation_clock", "Activation clock"
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    form = req.options.get("form", "clock")
    curves, theta, src = _group_curves(ctx, task)
    if not any(len(v) for v in curves.values()):
        return _empty_spec(fid, title, "no cycle envelopes are stored", "activation clock")
    seed = B.seed_of(ctx)
    x = np.linspace(0.0, 100.0, 101)
    spm = _t(ctx, "stats_emg", "spm")
    test = f"openers_vs_closers_{task}"
    clusters = []
    if not spm.empty and "test" in spm and (spm.test == test).any():
        clusters = B.parse_clusters(spm[spm.test == test].clusters.iloc[0])
    ns = {g: int(len(v)) for g, v in curves.items()}
    peak = float(x[int(np.nanargmax(theta))]) if np.isfinite(theta).any() else np.nan
    if form == "curves":
        p = Panel(f"{TASK_LABEL.get(task, task).capitalize()}: openers vs closers",
                  Axis("Movement cycle", V.PCT, lim=(0.0, 100.0)),
                  Axis("Activation re peak", V.PCT), [], legend="below")
        for g, _ in GROUPS:
            if len(curves[g]):
                B.add_curves_ci(p, x, curves[g], label=GROUP_LABEL[g], role=GROUP_ROLE[g],
                                show_subjects=False, seed=seed)
        B.add_spm_bars(p, [("openers ≠ closers", clusters, "accent")])
        _vlines(p, [peak], "reference")
        _phase_ticks(p)
        panels = [p]
        width, h = "column", 1.9
    else:
        rmax = _nice(max(float(np.nanmax(_nanmean(v, axis=0))) for v in curves.values()
                         if len(v)) * 1.05)
        p = Panel(f"{TASK_LABEL.get(task, task).capitalize()}: openers vs closers",
                  Axis("", lim=(-1.45 * rmax, 1.45 * rmax)), Axis("", lim=(-1.4 * rmax, 1.5 * rmax)),
                  [], legend="below", equal=True)
        ring = np.linspace(0.0, 100.0, 181)
        for r in (rmax / 2, rmax):
            cx, cy = _polar(ring, np.full(ring.size, r))
            p.series.append(Series("line", cx, cy, role="faint"))
        p.series.append(Series("text", np.array([0.0, 0.0]), np.array([rmax / 2, rmax]),
                               text=[f"{rmax / 2:g} % of peak", f"{rmax:g} % of peak"],
                               style={"anchor": "right", "color": "ink2"}))
        for ph in (0.0, 25.0, 50.0, 75.0):
            cx, cy = _polar(np.array([ph, ph]), np.array([0.0, rmax]))
            p.series.append(Series("line", cx, cy, role="faint"))
        tx, ty = _polar(np.array([0.0, 25.0, 50.0, 75.0]), np.full(4, 1.22 * rmax))
        p.series.append(Series("text", tx, ty - np.array([0, 0.06, 0.12, 0.06]) * rmax,
                               text=["cycle 0 % (start)", "25 %", "50 %", "75 %"],
                               style={"anchor": "above", "color": "ink2"}))
        for g, _ in GROUPS:
            S = curves[g]
            if not len(S):
                continue
            mu = _nanmean(S, axis=0)
            lo, hi = B.boot_curve_ci(S, centre="mean", seed=seed)
            if np.isfinite(lo).all() and np.isfinite(hi).all():
                ox, oy = _polar(x, hi)
                ix, iy = _polar(x[::-1], lo[::-1])
                p.series.append(Series("fill", np.r_[ox, ix], np.r_[oy, iy], role=GROUP_ROLE[g],
                                       style={"alpha": 0.22, "edge": False}))
            cx, cy = _polar(x, mu)
            p.series.append(Series("line", cx, cy, role=GROUP_ROLE[g], label=GROUP_LABEL[g],
                                   style={"width": 1.4}))
        for j, (a, b, _) in enumerate(clusters):
            arc = np.linspace(a, b, max(int(abs(b - a) * 2), 3))
            cx, cy = _polar(arc, np.full(arc.size, 1.1 * rmax))
            p.series.append(Series("line", cx, cy, role="accent", style={"width": 2.6},
                                   label="SPM cluster: openers ≠ closers" if j == 0 else None))
        if np.isfinite(peak):
            cx, cy = _polar(np.array([peak, peak]), np.array([0.0, 1.1 * rmax]))
            p.series.append(Series("line", cx, cy, role="reference", label="maximum opening"))
        p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
        panels = [p]
        width, h = "column", 3.0
    spec = FigureSpec(fid, title, panels, width=width, height_in=h,
                      sources=[src, "stats_emg:spm"],
                      perspective=f"cohort, {TASK_LABEL.get(task, task)}",
                      meta={"n": ns, "clusters": clusters})
    corr = "" if not clusters else "paired SPM{t}, cluster-level p < 0.05"
    return attach(spec, CaptionSpec(
        f"{TASK_LABEL.get(task, task).capitalize()}: the activation of the openers and the "
        "closers over the time-normalised cycle, as the openers-vs-closers test compares them: "
        "per participant the envelope averaged over the group's channels usable for the shape "
        "and over cycles, in % of its own peak"
        + (", drawn as a clock: the angle is the cycle phase (start at the top, clockwise), the "
           "radius the activation" if form == "clock" else ""),
        [("lines", "cohort mean per group"),
         ("bands", "95 % bootstrap CI of the mean (participants resampled, pointwise)"),
         ("dashed line", "phase of the maximum mean opening angle")]
        + ([("dark arc" if form == "clock" else "bar above the curves",
             "supra-threshold cluster of the openers-vs-closers SPM test")] if clusters else []),
        f"n = {max(ns.values())} participants",
        "mean over participants with its pointwise bootstrap CI"
        + (f"; {corr}" if corr else ""),
        evidence="pre-specified" if clusters else "descriptive",
        notes=[] if clusters else ["No openers-vs-closers SPM test is stored for this task"]))


# == 6. surface and waterfall ==================================================================
@register("emg_surface", "dashboard", "EMG activation surface", params=("phase",),
          phase="open_close", phases=CYCLE_TASKS, domain="emg_time",
          choices={"form": ("phase_heatmap", "surface", "waterfall")}, form="phase_heatmap")
def emg_surface(req: ViewRequest) -> FigureSpec:
    """Muscle × cycle-phase activation per task: a heatmap (default: muscles are categories,
    so a surface between them suggests a continuum that is not there), the 3-D surface
    (surface3d), or a 3-D waterfall of the cycle-phase spectra per muscle group."""
    ctx = req.ctx
    fid, title = "emg_surface", "EMG activation surface"
    form = req.options.get("form", "phase_heatmap")
    task = req.phase if req.phase in CYCLE_TASKS else "open_close"
    panels: list[Panel] = []
    if form == "waterfall":
        grid, per, info = _cycle_spectra(ctx, task)
        if not per:
            return _empty_spec(fid, title, "no per-window spectra are stored", "spectral waterfall")
        A = np.stack([per[p] for p in sorted(per)])
        df = float(np.median(np.diff(grid))) if grid.size > 1 else 1.0
        fb = np.arange(1, grid.size + 1, dtype=float)       # frequency bins from the lowest
        step = 100.0 / PHASE_BINS
        for k, (g, ms) in enumerate(GROUPS):
            Z = 10.0 * _nanmean(_nanmean(A[:, [MUSCLES.index(m) for m in ms]], axis=1), axis=0)
            p = Panel(f"({'ab'[k]}) {GROUP_LABEL[g][:1].upper() + GROUP_LABEL[g][1:]}",
                      Axis(f"Frequency bin ({df:g} Hz from {grid[0] - df / 2:g} Hz)"),
                      Axis(f"Cycle phase bin ({step:g} %)"), [], projection="3d",
                      z=Axis("Power share", "\\decibel"), legend="none", view=(-60.0, 28.0))
            for i in range(PHASE_BINS):
                p.series.append(Series("line3d", fb, np.full(fb.size, i + 1.0), z=Z[i],
                                       role=GROUP_ROLE[g], style={"width": 0.7}))
            panels.append(p)
        n = len(per)
        spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.4,
                          sources=["emg:arrays", "emg:spectral", "kinematics:cycles",
                                   "stats_emg:arrays"],
                          perspective=f"cohort, {TASK_LABEL.get(task, task)}", meta={"n": n})
        return attach(spec, CaptionSpec(
            f"{TASK_LABEL.get(task, task).capitalize()}: waterfall of the spectral shape over the "
            f"cycle, one line per {100 / PHASE_BINS:g} % of the cycle (power share per frequency "
            "bin in dB from the stored per-window spectra placed at their cycle phase; mean "
            "over the cycles of a participant and the group's channels, then over participants)",
            [("lines", "mean spectrum at one cycle phase, front to back through the cycle")],
            f"n = {n} participants", "mean over cycles, then mean over participants",
            evidence="exploratory",
            notes=[f"Stored windows of {info['win_s']:.2g} s smooth the phase axis"]))
    ns = []
    heat: list[tuple[str, np.ndarray]] = []
    for k, tk in enumerate(CYCLE_TASKS):
        rows = []
        for pid in _pids(ctx):
            e = _a(ctx, "emg", pid)
            key = f"profile__{tk}__pct"
            if key not in e:
                continue
            names = [str(n) for n in e.get("profile__names", _names(ctx, pid, e))]
            P = _nanmean(np.asarray(e[key], float), axis=0)
            rows.append(np.array([P[:, names.index(m)] if m in names else np.full(101, np.nan)
                                  for m in MUSCLES]))
        tl = TASK_LABEL.get(tk, tk)
        ttl = f"({'ab'[k]}) {tl[:1].upper() + tl[1:]}"
        if not rows:
            panels.append(_empty_panel(ttl, "no cycle envelopes"))
            continue
        ns.append(len(rows))
        if form == "phase_heatmap":
            heat.append((ttl, _nanmean(np.array(rows), axis=0)))     # (6 muscles, 101 %)
            continue
        Z = _nanmean(np.array(rows), axis=0)[:, ::SURF_STEP]  # (6, phases)
        xph = np.arange(Z.shape[1], dtype=float)
        ypos = SURF_ROW * np.arange(len(MUSCLES), dtype=float)
        p = Panel(ttl, Axis(f"Cycle ({SURF_STEP:g} % steps)"),
                  Axis("Muscle"),
                  [Series("surface3d", x=xph, y=ypos, z=Z, style={"alpha": 0.95})],
                  projection="3d", z=Axis("Activation", V.PCT), legend="none",
                  view=(-62.0, 32.0))
        for i, m in enumerate(MUSCLES):           # each muscle's ridge in its own colour
            p.series.append(Series("line3d", xph, np.full(xph.size, ypos[i]), z=Z[i],
                                   role=f"entity:{m}", label=MUSCLE_LABEL[m] if k == 1 else None,
                                   style={"width": 1.1}))
        p.y.ticks = [float(v) for v in ypos]
        p.y.ticklabels = [MUSCLE_LABEL[m] for m in MUSCLES]
        if k == 1:
            p.legend = "below"
        panels.append(p)
    if form == "phase_heatmap":
        top = max((float(np.nanmax(Z)) for _, Z in heat if np.isfinite(Z).any()), default=1.0)
        hp: list[Panel] = []
        for ttl, Z in heat:
            p = Panel(ttl, Axis("Cycle", V.PCT), Axis(""),
                      [Series("heatmap", z=Z)], legend="none",
                      colorbar=Colorbar("Activation", V.PCT, lim=(0.0, top)))
            p.x.ticks = [float(v) for v in (0, 25, 50, 75, 100)]
            p.x.ticklabels = ["0", "25", "50", "75", "100"]
            p.y.ticks = [float(i) for i in range(len(MUSCLES))]
            p.y.ticklabels = [MUSCLE_LABEL[m] for m in MUSCLES]
            hp.append(p)
        panels = [q for q in panels if q.title not in {t for t, _ in heat}] or []
        spec = FigureSpec(fid, title, hp + panels, rows=1, cols=2, width="text", height_in=1.6,
                          sources=["emg:arrays", "emg:cycles"], perspective="cohort, open-close and chewing",
                          meta={"n": ns})
        return attach(spec, CaptionSpec(
            "Activation over muscle and time-normalised cycle for open-close and chewing: the "
            "cohort mean of the per-participant mean envelope over cycles (% of max recorded), "
            "one row per muscle",
            [("colour", "mean activation, one scale for both tasks")],
            f"n = {max(ns) if ns else 0} participants",
            "mean over cycles, then over participants", evidence="descriptive"))
    spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text", height_in=2.5,
                      sources=["emg:arrays", "emg:cycles"], perspective="cohort, open-close and chewing",
                      meta={"n": ns})
    return attach(spec, CaptionSpec(
        "Activation surface over muscle and time-normalised cycle for open-close and chewing: "
        "the height and colour are the cohort mean of the per-participant mean envelope over "
        "cycles (% of max recorded)",
        [("surface height", "mean activation"),
         ("surface colour", "the same activation, light to dark within each task"),
         ("coloured lines", "each muscle's curve along the surface")],
        f"n = {max(ns) if ns else 0} participants", "mean over cycles, then over participants",
        evidence="descriptive"))


# == 7. EMG on the jaw path ====================================================================
@register("emg_on_path", "dashboard", "EMG on the jaw path", params=("pid", "phase"),
          phase="chewing", phases=("open_close", "chewing", "free_motion"), domain="emg_time",
          choices={"form": ("2d", "3d")}, form="2d")
def emg_on_path(req: ViewRequest) -> FigureSpec:
    """The incisor path of one participant and task, where the openers and the closers are
    active (3-D) or coloured by their envelope in the sagittal and frontal planes (2-D)."""
    ctx = req.ctx
    fid, title = "emg_on_path", "EMG on the jaw path"
    pid = _pid(req)
    if pid is None:
        return _empty_spec(fid, title, "there are no participants", "EMG on the jaw path")
    who = _who(ctx, pid)
    task = req.phase if isinstance(req.phase, str) and req.phase in V.TASKS else "chewing"
    form = req.options.get("form", "2d")   # 3-D: legend over the path at print size
    k = _a(ctx, "kinematics", pid)
    e = _a(ctx, "emg", pid)
    if f"{task}__incisor" not in k or f"{task}__t_us" not in e:
        return _empty_spec(fid, title, f"no incisor path or envelopes for {who}",
                           "EMG on the jaw path")
    tk = np.asarray(k[f"{task}__t_us"], float)
    P = np.asarray(k[f"{task}__incisor"], float)
    names = _names(ctx, pid, e)
    rp = _rest_p95(ctx)
    env, thr = {}, {}
    for g, ms in GROUPS:
        eg = _env_grid(e, task, names, ms)
        if eg is None:
            continue
        env[g] = np.interp(tk, eg[0].astype(float), eg[1], left=np.nan, right=np.nan)
        thr[g] = (float(_nanmean(rp.loc[pid].reindex(list(ms)).to_numpy(float)))
                  if not rp.empty and pid in rp.index else np.nan)
    ok = np.isfinite(P).all(axis=1)
    panels: list[Panel] = []
    if form == "2d":
        lim = (0.0, _nice(max(_q(v, 99) for v in env.values()))) if env else (0.0, 1.0)
        planes = (("Sagittal", 0, 2, "Anterior"), ("Frontal", 1, 2, "Left"))
        L = 0
        for g, _ in GROUPS:
            if g not in env:
                continue
            for pl, i, j, xl in planes:
                m = ok & np.isfinite(env[g])
                p = Panel(f"({'abcd'[L]}) {GROUP_SHORT[g].capitalize()}, {pl.lower()} plane",
                          Axis(xl, V.MM), Axis("Superior", V.MM),
                          [Series("line", P[ok, i], P[ok, j], role="faint"),
                           Series("colorline", P[m, i], P[m, j], c=env[g][m],
                                  style={"lim": lim})],
                          legend="none", equal=True,
                          colorbar=Colorbar(f"{GROUP_SHORT[g].capitalize()} activation", V.PCT,
                                            lim=lim))
                panels.append(p)
                L += 1
        _grid_labels(panels, 2)
        cols, rows_, h = 2, int(np.ceil(len(panels) / 2)), 1.8
        marks = [("colour", "group envelope along the path (% of max recorded)"),
                 ("light line", "the whole path")]
    else:
        p = Panel(f"{who}, {TASK_LABEL.get(task, task)}", Axis("Anterior", V.MM),
                  Axis("Left", V.MM), [Series("line3d", P[ok, 0], P[ok, 1], z=P[ok, 2],
                                              role="context", label="incisor path")],
                  projection="3d", z=Axis("Superior", V.MM), legend="inside:north west",
                  view=(-60.0, 18.0))
        for g, _ in GROUPS:
            if g not in env or not np.isfinite(thr.get(g, np.nan)):
                continue
            m = ok & (env[g] > thr[g])
            if m.any():
                p.series.append(Series("scatter3d", P[m, 0], P[m, 1], z=P[m, 2],
                                       role=GROUP_ROLE[g], label=f"{GROUP_SHORT[g]} active",
                                       style={"size": 0.9}))
        panels = [p]
        cols, rows_, h = 1, 1, 2.8
        marks = [("grey line", "incisor path"),
                 ("vermilion dots", "openers active (mean envelope above its rest 95th "
                                     "percentile)"),
                 ("blue dots", "closers active (same rule)")]
    act = {g: float(np.mean(env[g][ok & np.isfinite(env[g])] > thr[g])) if g in env
           and np.isfinite(thr.get(g, np.nan)) else np.nan for g, _ in GROUPS}
    spec = FigureSpec(fid, title, panels, rows=rows_, cols=cols,
                      width="text" if form == "2d" else "column", height_in=h,
                      sources=["kinematics:arrays", "emg:arrays", "emg:activation"],
                      perspective=f"{who}, {TASK_LABEL.get(task, task)}",
                      meta={"pid": pid, "active_share": act})
    act_txt = ", ".join(f"{GROUP_SHORT[g]} active in {100 * v:.0f} % of the samples"
                        for g, v in act.items() if np.isfinite(v))
    lp = _sig(ctx, "kinematics", "lowpass_hz", default=6.0)
    return attach(spec, CaptionSpec(
        f"{who}, {TASK_LABEL.get(task, task)}: the lower-incisor path in the maxillary frame "
        f"(optical poses low-passed at {float(lp):g} Hz) with the mean envelope of the openers "
        "and of the closers (% of max recorded) sampled at the pose times"
        + (": where in the workspace each group is active" if form == "3d" else
           ", in the sagittal and frontal planes"),
        marks, f"n = 1 participant, {int(ok.sum())} pose samples",
        "per-sample values, no summary statistic" + (f"; {act_txt}" if act_txt else ""),
        evidence="descriptive"))


# == 8. multimodal story =======================================================================
@register("emg_story", "dashboard", "Multimodal chewing excerpt", params=("pid", "phase"),
          phase="chewing", phases=CYCLE_TASKS, domain="timing",
          choices={"muscle": CLOSERS}, muscle="masseter_r")
def emg_story(req: ViewRequest) -> FigureSpec:
    """One excerpt on one time axis: raw EMG, its spectrogram strip, the group envelopes, the
    IMU and optical opening angle, the optical intent label, the decoder's stream decisions
    and the causal release decisions."""
    from scipy.signal import spectrogram

    from biomech.results import COHORT_UNIT

    ctx = req.ctx
    fid, title = "emg_story", "Multimodal chewing excerpt"
    pid = _pid(req)
    if pid is None:
        return _empty_spec(fid, title, "there are no participants", "multimodal excerpt")
    who = _who(ctx, pid)
    task = req.phase if req.phase in CYCLE_TASKS else "chewing"
    muscle = req.options.get("muscle") if req.options.get("muscle") in MUSCLES else "masseter_r"
    sg = _signal(ctx, pid, task, muscle)
    cyc = _cycles(ctx, pid, task)
    if sg is None or cyc.empty:
        return _empty_spec(fid, title, f"no EMG or no cycles for {who}", "multimodal excerpt")
    t = sg["t_us"]
    cyc = cyc[(cyc.t_start_us >= t[0]) & (cyc.t_end_us <= t[-1])]
    n_c = EXCERPT_CYCLES.get(task, 3) + 1
    if sg["session"] and len(cyc) > n_c:
        k0 = max((len(cyc) - n_c) // 2, 0)
        t0 = float(cyc.t_start_us.iloc[k0]) - 0.3e6
        t1 = float(cyc.t_end_us.iloc[min(k0 + n_c, len(cyc)) - 1]) + 0.2e6
    else:
        t0, t1 = float(t[0]), float(t[-1])
    t0, t1 = max(t0, float(t[0])), min(t1, float(t[-1]))
    span = (0.0, (t1 - t0) / 1e6)
    sel = (t >= t0) & (t <= t1)
    ts = (t[sel] - t0) / 1e6
    role = f"entity:{muscle}"
    ml = MUSCLE_LABEL.get(muscle, muscle)
    panels: list[Panel] = []
    raw = sg["cleaned"][sel]
    k = _thin(ts, raw)
    amp = _nice(_q(np.abs(raw), 99.9) * 1.1)
    panels.append(Panel(f"(a) EMG, {ml} (band-passed)", Axis("Time", V.SEC, lim=span),
                        Axis("EMG", V.UV, lim=(-amp, amp)),
                        [Series("line", ts[k], raw[k], role=role, style={"width": 0.4})],
                        sharex="t", legend="none", height=0.8))
    lo, hi = _band(ctx)
    fs = sg["fs"]
    nper = max(int(round(STFT_S * fs)), 16)
    ps = Panel("(b) Short-time spectrum", Axis("Time", V.SEC, lim=span),
               Axis("Frequency", V.HZ, lim=(lo, hi)), [], sharex="t", legend="none", height=0.8)
    x = np.where(sg["mask"][sel], np.nan, raw)
    if x.size > nper:
        f, tt, S = spectrogram(np.nan_to_num(x), fs, nperseg=nper,
                               noverlap=int(round(nper * STFT_OVERLAP)))
        m = (f >= lo) & (f <= hi)
        f, S = f[m], S[m]
        with np.errstate(divide="ignore"):
            Z = 10 * np.log10(np.where(S > 0, S, np.nan))
        zl = (np.floor(_q(Z, 5)), np.ceil(_q(Z, 99.5)))
        df = float(f[1] - f[0]) if f.size > 1 else 1.0
        dt = float(tt[1] - tt[0]) if tt.size > 1 else nper / fs
        ps.series.append(Series("image", img=Z[::-1], extent=(
            float(tt[0] - dt / 2), float(tt[-1] + dt / 2), float(f[0] - df / 2),
            float(f[-1] + df / 2)), style={"cmap": "sequential", "lim": zl}))
        ps.colorbar = Colorbar("PSD", "\\decibel", lim=zl)
    panels.append(ps)
    e = _a(ctx, "emg", pid)
    names = _names(ctx, pid, e)
    pe = Panel("(c) Envelopes", Axis("Time", V.SEC, lim=span), Axis("Envelope", V.PCT), [],
               sharex="t", legend="below")
    # the single channels thin and light behind their group mean (chewing differs
    # between the working and the balancing side, which the mean over both sides hides)
    for g, ms in GROUPS:
        for t_, y_ in _env_channels(e, task, names, ms):
            m = (t_ >= t0) & (t_ <= t1)
            pe.series.append(Series("line", (t_[m] - t0) / 1e6, y_[m], role=GROUP_ROLE[g],
                                    style={"width": 0.45, "alpha": 0.35}))
    for g, ms in GROUPS:
        eg = _env_grid(e, task, names, ms)
        if eg is not None:
            m = (eg[0] >= t0) & (eg[0] <= t1)
            pe.series.append(Series("line", (eg[0][m] - t0) / 1e6, eg[1][m], role=GROUP_ROLE[g],
                                    label=GROUP_LABEL[g]))
    panels.append(pe)
    imu = _a(ctx, "imu", pid)
    pa = Panel("(d) Opening angle", Axis("Time", V.SEC, lim=span), Axis("Angle", V.DEG), [],
               sharex="t", legend="below")
    if f"{task}__t_us" in imu:
        ti = np.asarray(imu[f"{task}__t_us"], float)
        m = (ti >= t0) & (ti <= t1)
        for key, lab, rl in (("theta_opt", "optical", "accent"), ("theta_quat", "IMU", "set:imu")):
            if f"{task}__{key}" in imu:
                pa.series.append(Series("line", (ti[m] - t0) / 1e6,
                                        np.asarray(imu[f"{task}__{key}"], float)[m], role=rl,
                                        label=lab))
    panels.append(pa)
    pi = Panel("(e) Intent: label, decoder and release", Axis("Time", V.SEC, lim=span),
               Axis(""), [], sharex="t", legend="below", height=1.1)
    lab_a = _a(ctx, "labels", pid)
    if f"{task}__label" in lab_a:
        tl_ = np.asarray(lab_a[f"{task}__t_us"], float)
        yl = np.asarray(lab_a[f"{task}__label"], float)
        yl[yl < 0] = np.nan
        m = (tl_ >= t0) & (tl_ <= t1)
        pi.series.append(Series("line", (tl_[m] - t0) / 1e6, yl[m], role="context",
                                label="optical label", style={"width": 2.2}))
    on = _a(ctx, "ml_online", COHORT_UNIT)
    n_rel = 0
    if f"online__{pid}__t_us" in on:
        to = np.asarray(on[f"online__{pid}__t_us"], float)
        yh = np.asarray(on[f"online__{pid}__yhat"], float)
        yh[yh < 0] = np.nan
        m = (to >= t0) & (to <= t1)
        pi.series.append(Series("scatter", (to[m] - t0) / 1e6, yh[m] + 0.15, role="set:emg_imu",
                                label="decoder (smoothed stream)", style={"size": 1.1,
                                                                          "ms": "s"}))
        rel = np.asarray(on.get(f"online__{pid}__release_us", []), float)
        rel = rel[(rel >= t0) & (rel <= t1)]
        n_rel = int(rel.size)
        if rel.size:
            for p in panels + [pi]:
                p.series.append(Series("vline", x=(rel - t0) / 1e6, role="group:closers",
                                       style={"width": 0.8}))
            pi.series.append(Series("line", np.array([np.nan]), np.array([np.nan]),
                                    role="group:closers", label="release decision"))
    pi.y.ticks, pi.y.ticklabels = [float(i) for i in range(3)], list(CLASSES[:3])
    pi.y.lim = (-0.5, 2.6)
    panels.append(pi)
    _bottom_x(panels)
    spec = FigureSpec(fid, title, panels, rows=len(panels), cols=1, width="text", height_in=0.9,
                      sources=[sg["src"], "emg:arrays", "imu:arrays", "labels:arrays",
                               "ml_online:arrays"],
                      perspective=f"{who}, {TASK_LABEL.get(task, task)}",
                      meta={"pid": pid, "window_us": (t0, t1), "n_release": n_rel})
    return attach(spec, CaptionSpec(
        f"{who}, a {span[1]:.1f}-s {TASK_LABEL.get(task, task)} excerpt on one time axis: the "
        f"band-passed EMG of the {ml} and its short-time spectrum ({1e3 * STFT_S:g} ms windows); "
        f"the mean envelopes of the openers and closers ({_env_ms(ctx):g} ms RMS, % of max "
        "recorded); the opening angle from the optical system and from the IMUs; the optical "
        "intent label (rest, open, close), the simulated real-time decoder's smoothed "
        "decisions and the causal closer-release decisions",
        [("vermilion and blue lines in (c)", "openers and closers"),
         ("dark and sky lines in (d)", "optical and IMU opening angle"),
         ("thick grey line in (e)", "optical label"), ("green squares", "decoder decisions"),
         ("blue vertical lines", "release decisions")],
        f"n = 1 participant, {n_rel} release decisions in the excerpt",
        "single excerpt, no summary statistic", evidence="descriptive"))


# == 9. feature-space embedding ================================================================
def _embed_data(ctx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """EMG feature windows of every participant: a per participant and class random sample
    (study seed) of ``EMBED_PER_CLASS`` windows: (X, class, participant index, pids)."""
    rng = np.random.default_rng(B.seed_of(ctx))
    X, Y, P, pids = [], [], [], []
    for pid in _pids(ctx):
        a = _a(ctx, "ml_windows", pid)
        if "X_emg" not in a or "y" not in a:
            continue
        x = np.asarray(a["X_emg"], float)
        y = np.asarray(a["y"], int)
        keep = []
        for c in range(len(CLASSES)):
            idx = np.flatnonzero((y == c) & np.isfinite(x).all(axis=1))
            if idx.size:
                keep.append(rng.choice(idx, min(EMBED_PER_CLASS, idx.size), replace=False))
        if not keep:
            continue
        idx = np.sort(np.concatenate(keep))
        X.append(x[idx])
        Y.append(y[idx])
        P.append(np.full(idx.size, len(pids)))
        pids.append(pid)
    if not X:
        return np.empty((0, 0)), np.empty(0, int), np.empty(0, int), []
    return np.vstack(X), np.concatenate(Y), np.concatenate(P), pids


def _pca(X: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Scores on the first ``k`` principal components of the column-standardised X and the
    explained variance shares."""
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    var = s ** 2 / max(np.sum(s ** 2), 1e-12)
    return U[:, :k] * s[:k], var[:k]


@register("emg_feature_embedding", "dashboard", "EMG feature space", domain="ml",
          choices={"form": ("class", "participant", "within")}, form="class")
def emg_feature_embedding(req: ViewRequest) -> FigureSpec:
    """Principal components of the decoding EMG feature windows: pooled over participants and
    coloured by class, each participant highlighted, or fitted within each participant."""
    ctx = req.ctx
    fid, title = "emg_feature_embedding", "EMG feature space"
    form = req.options.get("form", "class")
    X, Y, P, pids = _embed_data(ctx)
    if not pids:
        return _empty_spec(fid, title, "no decoding feature windows are stored",
                           "EMG feature space")
    who = B.pseudo_rows(ctx, pids)
    panels: list[Panel] = []
    present = [c for c in range(len(CLASSES)) if (Y == c).any()]
    if form == "within":
        for j, pid in enumerate(pids):
            m = P == j
            S, var = _pca(X[m])
            p = Panel(who[j], Axis("PC 1"), Axis("PC 2"), [], legend="none",
                      note=f"{100 * var[0]:.0f} % / {100 * var[1]:.0f} %")
            for c in present:
                mc = Y[m] == c
                if mc.any():
                    p.series.append(Series("scatter", S[mc, 0], S[mc, 1], role=CLASS_ROLE[c],
                                           style={"size": 0.6}))
            p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
            panels.append(p)
        panels[-1].legend = "below"
        for c in present:            # legend entries on the last panel
            panels[-1].series.append(Series("scatter", np.array([np.nan]), np.array([np.nan]),
                                            role=CLASS_ROLE[c], label=CLASSES[c]))
        for s_ in panels[0].series:
            s_.label = None
        cols = 5
    else:
        S, var = _pca(X)
        xa = Axis(f"PC 1 ({100 * var[0]:.0f} %)")
        ya = Axis(f"PC 2 ({100 * var[1]:.0f} %)")
        xl = (_q(S[:, 0], 0.5), _q(S[:, 0], 99.5))
        yl = (_q(S[:, 1], 0.5), _q(S[:, 1], 99.5))
        if form == "class":
            pa = Panel("(a) Windows by class", xa, ya, [], legend="below")
            for c in present:
                m = Y == c
                # light, small points (the mass overplots); the legend keys are drawn at the
                # size of the class centres in (b), so they stay readable
                pa.series.append(Series("scatter", S[m, 0], S[m, 1], role=CLASS_ROLE[c],
                                        style={"size": 0.45, "alpha": 0.45}))
            for c in present:
                pa.series.append(Series("scatter", np.array([np.nan]), np.array([np.nan]),
                                        role=CLASS_ROLE[c], label=CLASSES[c],
                                        style={"size": 1.1}))
            pb = Panel("(b) Class centre per participant", Axis(xa.label), Axis(ya.label), [],
                       legend="none")
            for c in present:
                cx = [S[(Y == c) & (P == j)].mean(axis=0) for j in range(len(pids))
                      if ((Y == c) & (P == j)).any()]
                if cx:
                    C = np.array(cx)
                    pb.series.append(Series("scatter", C[:, 0], C[:, 1], role=CLASS_ROLE[c],
                                            style={"size": 1.1}))
            for p in (pa, pb):
                p.x.lim, p.y.lim = xl, yl
            panels = [pa, pb]
            cols = 2
        else:
            for j in range(len(pids)):
                p = Panel(who[j], Axis(xa.label, lim=xl), Axis(ya.label, lim=yl),
                          [Series("scatter", S[P != j, 0], S[P != j, 1], role="faint",
                                  style={"size": 0.35})], legend="none")
                m = P == j
                for c in present:
                    mc = m & (Y == c)
                    if mc.any():
                        p.series.append(Series("scatter", S[mc, 0], S[mc, 1], role=CLASS_ROLE[c],
                                               style={"size": 0.6}))
                p.x.ticks, p.x.ticklabels, p.y.ticks, p.y.ticklabels = [], [], [], []
                panels.append(p)
            panels[-1].legend = "below"
            for c in present:
                panels[-1].series.append(Series("scatter", np.array([np.nan]),
                                                np.array([np.nan]), role=CLASS_ROLE[c],
                                                label=CLASSES[c]))
            cols = 5
    if cols == 5:
        _grid_labels(panels, cols)
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=2.2 if cols == 2 else 1.05,
                      sources=["ml_windows:arrays", "ml_windows:refs"],
                      perspective="cohort, decoding windows",
                      meta={"n_windows": int(len(Y)), "n": len(pids)})
    n_feat = X.shape[1]
    what = {"class": "(a) pooled over participants, coloured by the class; (b) the centre of "
                     "each class per participant",
            "participant": "pooled over participants; one panel per participant with that "
                           "participant's windows coloured by class over everyone else's "
                           "(light grey)",
            "within": "fitted within each participant, coloured by class"}[form]
    return attach(spec, CaptionSpec(
        f"The decoding EMG feature windows ({n_feat} features per window, computed per channel "
        "relative to the participant's rest) on their "
        f"first two principal components (columns standardised), {what}. How far the class "
        "centres of different participants scatter, compared with how far the classes lie "
        "apart within one participant, is what a decoder trained on other participants cannot "
        "anticipate",
        [("colour and marker shape", "classes (rest grey)")]
        + ([("markers in (b)", "one participant's mean of a class")] if form == "class"
           else [])
        + ([("numbers in a panel", "variance share of the two components")]
           if form == "within" else []),
        f"n = {len(pids)} participants, up to {EMBED_PER_CLASS} random windows per participant "
        f"and class (study seed), {len(Y)} windows",
        "principal component scores (no inference)", evidence="exploratory",
        notes=["Principal components (a linear embedding); a nonlinear embedding is not shown"]))


# == 10. ridgelines ============================================================================
@register("emg_ridgelines", "dashboard", "EMG ridgelines per participant", params=("phase",),
          phase="chewing", phases=CYCLE_TASKS, domain="emg_freq",
          choices={"form": ("spectra", "envelopes"), "muscle": MUSCLES}, form="spectra",
          muscle="masseter_r")
def emg_ridgelines(req: ViewRequest) -> FigureSpec:
    """Per-participant spectra (or cycle envelopes) of one muscle as ridgelines, shape
    normalised, each participant's amplitude printed, the cohort median on top."""
    ctx = req.ctx
    fid, title = "emg_ridgelines", "EMG ridgelines per participant"
    task = req.phase if req.phase in CYCLE_TASKS else "chewing"
    form = req.options.get("form", "spectra")
    muscle = req.options.get("muscle") if req.options.get("muscle") in MUSCLES else "masseter_r"
    shape_ok = _shape_ok(ctx)
    rows, amps, labels, flagged = [], [], [], []
    xgrid = None
    lo, hi = _band(ctx)
    for pid in _pids(ctx):
        e = _a(ctx, "emg", pid)
        names = _names(ctx, pid, e)
        if muscle not in names:
            continue
        i = names.index(muscle)
        if form == "spectra":
            if f"spectral__{task}__psd" not in e:
                continue
            f = np.asarray(e["spectral__freq"], float)
            m = (f >= lo) & (f <= hi)
            P = np.asarray(e[f"spectral__{task}__psd"], float)[i, m]
            if not np.isfinite(P).any() or np.nanmax(P) <= 0:
                continue
            with np.errstate(divide="ignore"):
                db = 10 * np.log10(np.where(P > 0, P, np.nan))
            amp = 10 * np.log10(np.trapezoid(np.nan_to_num(P), f[m]))
            rows.append(db - np.nanmax(db))                 # shape: dB re own peak
            amps.append(amp)
            xgrid = f[m]
            r = shape_ok[(shape_ok.pid == pid) & (shape_ok.task == task)
                         & (shape_ok.muscle == muscle)] if not shape_ok.empty else shape_ok
            flagged.append(bool(len(r)) and not bool(r.usable_shape.iloc[0]))
        else:
            k = f"profile__{task}__pct"
            if k not in e:
                continue
            y = _nanmean(np.asarray(e[k], float)[:, :, i], axis=0)
            if not np.isfinite(y).any() or np.nanmax(y) <= 0:
                continue
            rows.append(100.0 * y / np.nanmax(y))
            amps.append(float(np.nanmax(y)))
            xgrid = np.linspace(0.0, 100.0, 101)
            flagged.append(False)
        labels.append(_who(ctx, pid) + ("*" if flagged[-1] else ""))
    if not rows:
        return _empty_spec(fid, title, "no spectra or envelopes are stored", "ridgelines")
    R_ = np.array(rows)
    if form == "spectra":
        floor = min(-5.0, 5.0 * np.floor(_q(R_, 2) / 5.0))   # dB re peak, from the data
        Rn = (np.clip(R_, floor, 0.0) - floor) / -floor     # 0 .. 1 per ridge
        amp_txt = [f"{B.snum(a - np.nanmedian(amps), 1)} dB" for a in amps]
        xa = Axis("Frequency", V.HZ, lim=(lo, hi))
        # ridges that never come near the floor: a flatter spectrum, named in the caption so a
        # raised, flat ridge does not read as missing data
        flat = [(labels[j], float(-np.nanmin(R_[j])), float(xgrid[np.nanargmax(R_[j])]))
                for j in range(len(R_)) if np.nanmin(Rn[j]) > 0.3]
    else:
        Rn = R_ / 100.0
        amp_txt = [f"{a:.0f} %" for a in amps]
        flat = []
        xa = Axis("Movement cycle", V.PCT, lim=(0.0, 100.0))
    gap = 0.55
    hr = 1.5 * gap                                        # ridge height: neighbours overlap a little
    n = len(Rn)
    p = Panel(f"{MUSCLE_LABEL.get(muscle, muscle)}, {TASK_LABEL.get(task, task)}", xa,
              Axis(""), [], legend="none")
    role = f"entity:{muscle}"
    med = _nanmedian(Rn, axis=0)
    for j in range(n):
        base = (n - j) * gap
        y = base + hr * np.nan_to_num(Rn[j])
        p.series.append(Series("fill", np.r_[xgrid, xgrid[::-1]],
                               np.r_[y, np.full(xgrid.size, base)], role=role,
                               style={"alpha": 0.12, "edge": False}))
        p.series.append(Series("line", xgrid, y, role=role, style={"width": 0.8}))
    top = (n + 1) * gap + 0.4 * gap
    p.series.append(Series("fill", np.r_[xgrid, xgrid[::-1]],
                           np.r_[top + hr * np.nan_to_num(med), np.full(xgrid.size, top)],
                           role="accent", style={"alpha": 0.15, "edge": False}))
    p.series.append(Series("line", xgrid, top + hr * med, role="accent", style={"width": 1.4}))
    p.y.ticks = [float(top)] + [float((n - j) * gap) for j in range(n)]
    p.y.ticklabels = ["median"] + [f"{lb} ({t})" for lb, t in zip(labels, amp_txt)]
    p.y.lim = (gap * 0.6, top + hr * 1.1)
    spec = FigureSpec(fid, title, [p], width="column", height_in=0.2 * n + 1.0,
                      sources=["emg:arrays", "emg:spectral"],
                      perspective=f"cohort, {MUSCLE_LABEL.get(muscle, muscle)}, "
                                  f"{TASK_LABEL.get(task, task)}",
                      meta={"n": n, "amplitude": dict(zip(labels, amps))})
    if form == "spectra":
        shows = (f"{TASK_LABEL.get(task, task).capitalize()}, "
                 f"{MUSCLE_LABEL.get(muscle, muscle)}: the Welch power spectral density of the "
                 "cleaned EMG of each participant as one ridge, in dB re the participant's own "
                 f"peak ({lo:g}–{hi:g} Hz, ridges clipped at {B.num(floor, 0)} dB), so the shapes can "
                 "be compared; the number next to a participant is its in-band power re the "
                 "cohort median")
        marks = [("ridges", "participants"), ("dark ridge on top", "cohort median shape"),
                 ("numbers in brackets", "in-band power re the cohort median")]
    else:
        shows = (f"{TASK_LABEL.get(task, task).capitalize()}, "
                 f"{MUSCLE_LABEL.get(muscle, muscle)}: each participant's mean envelope over "
                 "the time-normalised cycle as one ridge, scaled to its own peak so the shapes "
                 "can be compared; the number next to a participant is its peak in % of max "
                 "recorded")
        marks = [("ridges", "participants"), ("dark ridge on top", "cohort median shape"),
                 ("numbers in brackets", "the participant's peak activation")]
    return attach(spec, CaptionSpec(
        shows, marks, f"n = {n} participants",
        "per participant the mean over the task; cohort median of the normalised shapes",
        evidence="exploratory",
        notes=(["* Channel not usable for the spectral shape (shown, not used in the shape "
                "statistics)"] if any(flagged) else [])
        + ([("; ".join(f"{lb}: a flatter spectrum (at most {B.num(r, 0)} dB below its peak at "
                       f"{B.num(fp, 0)} Hz)" for lb, r, fp in flat))
            + ", so the ridge stays high across the band; recorded data, not a gap"]
           if flat else [])))


# == 11. usability flow ========================================================================
GATES = ("active at window start", "burst below", "no onset found", "masked samples in the window")


def _gate_counts(d: pd.DataFrame, rule: str, burst_min: float) -> dict[str, int]:
    out = {g: 0 for g in GATES}
    out["usable"] = 0
    for r in d.itertuples():
        why = _reason(r, rule, burst_min)
        key = "burst below" if why.startswith("burst below") else why
        out[key] += 1
    return out


@register("emg_usability_flow", "dashboard", "Timing usability flow", domain="qc",
          choices={"form": ("flow", "heatmap"), "rule": RULES, "task": ("both",) + CYCLE_TASKS},
          form="flow", rule="drive", task="both")
def emg_usability_flow(req: ViewRequest) -> FigureSpec:
    """Recorded channel-cycles → usable ones through the timing gates in order (active at the
    window start, burst size, onset found, unmasked share), per direction; or the outcome
    shares per participant."""
    ctx = req.ctx
    fid, title = "emg_usability_flow", "Timing usability flow"
    form = req.options.get("form", "flow")
    rule = req.options.get("rule") if req.options.get("rule") in RULES else _primary_rule(ctx)
    task = req.options.get("task", "both")
    rows = _onset_rows(ctx, None, None if task == "both" else task)
    if rows.empty:
        return _empty_spec(fid, title, "no stored onsets", "usability flow")
    burst_min = float(_tp(ctx, "burst_min_db", default=6.0))
    tl = "open-close and chewing" if task == "both" else TASK_LABEL.get(task, task)
    panels: list[Panel] = []
    counts: dict[str, dict[str, int]] = {}
    if form == "heatmap":
        cols_ = ["usable"] + list(GATES)
        for k, (direction, grp) in enumerate((("opening", "openers"), ("closing", "closers"))):
            d = rows[rows.direction == direction]
            pids = [p for p in _pids(ctx) if (d.pid == p).any()]
            Z = np.full((len(pids), len(cols_)), np.nan)
            for i, pid in enumerate(pids):
                c = _gate_counts(d[d.pid == pid], rule, burst_min)
                tot = sum(c.values())
                Z[i] = [100.0 * c[g] / tot for g in cols_]
            counts[direction] = _gate_counts(d, rule, burst_min)
            p = B.heat(Z, B.pseudo_rows(ctx, pids),
                       ["usable", "active at start", f"burst < {burst_min:g} dB", "no onset",
                        "masked"], f"({'ab'[k]}) {direction.capitalize()}: {GROUP_SHORT[grp]}",
                       cb=Colorbar("Share of channel-cycles", V.PCT, lim=(0.0, 100.0)))
            panels.append(p)
        spec = FigureSpec(fid, title, panels, rows=1, cols=2, width="text",
                          height_in=B.rows_height(len(_pids(ctx)), per=0.17, extra=0.7),
                          sources=["timing:onsets"], perspective=f"per participant, {tl}",
                          meta={"counts": counts, "rule": rule})
        marks = [("colour and numbers", "share of the participant's channel-cycles (%) with "
                                        "that outcome; each row sums to 100 %")]
    else:
        for k, (direction, grp) in enumerate((("opening", "openers"), ("closing", "closers"))):
            d = rows[rows.direction == direction]
            c = _gate_counts(d, rule, burst_min)
            counts[direction] = c
            N = len(d)
            p = Panel(f"({'ab'[k]}) {direction.capitalize()}: {GROUP_SHORT[grp]}",
                      Axis("", lim=(-0.3, len(GATES) + 0.35)),
                      Axis("Channel-cycles", lim=(0.0, max(N, 1) * 1.08)), [], legend="none")
            role = GROUP_ROLE[grp]
            w = 0.16
            left = N
            xs = np.arange(len(GATES) + 1, dtype=float)
            for j, g in enumerate(GATES):
                drop = c[g]
                keep = left - drop
                x0 = xs[j]
                p.series.append(Series("fill", np.array([x0 - w, x0 + w, x0 + w, x0 - w]),
                                       np.array([0, 0, left, left], float), role=role,
                                       style={"alpha": 0.85, "edge": False}))
                # link: kept flow to the next bar, dropped flow to its box
                p.series.append(Series("fill", np.array([x0 + w, xs[j + 1] - w, xs[j + 1] - w,
                                                         x0 + w]),
                                       np.array([0, 0, keep, keep], float), role=role,
                                       style={"alpha": 0.25, "edge": False}))
                if drop:
                    xm = x0 + 0.5
                    yy = np.linspace(0.0, 1.0, 12)
                    sx = x0 + w + (xm - x0 - w) * yy
                    p.series.append(Series("fill", np.r_[sx, sx[::-1]],
                                           np.r_[np.full(sx.size, keep),
                                                 np.full(sx.size, left)], role="muted",
                                           style={"alpha": 0.35, "edge": False}))
                    p.series.append(Series("fill", np.array([xm, xm + 0.08, xm + 0.08, xm]),
                                           np.array([keep, keep, left, left], float),
                                           role="muted", style={"alpha": 0.8, "edge": False}))
                    p.series.append(Series("text", np.array([xm + 0.1]),
                                           np.array([0.5 * (keep + left)]),
                                           text=[f"−{drop}"], style={"anchor": "right"}))
                left = keep
            p.series.append(Series("fill", np.array([xs[-1] - w, xs[-1] + w, xs[-1] + w,
                                                     xs[-1] - w]),
                                   np.array([0, 0, left, left], float), role=role,
                                   style={"alpha": 0.85, "edge": False}))
            p.series.append(Series("text", np.array([xs[0], xs[-1]]),
                                   np.array([N, left], float),
                                   text=[f"{N}", f"{left} ({100 * left / max(N, 1):.0f} %)"],
                                   style={"anchor": "above", "color": "ink"}))
            p.x.ticks = list(xs)
            p.x.ticklabels = ["recorded", "not active at start", f"burst ≥ {burst_min:g} dB",
                              "onset found", "usable"]
            panels.append(p)
        spec = FigureSpec(fid, title, panels, rows=2, cols=1, width="text", height_in=1.55,
                          sources=["timing:onsets"], perspective=f"cohort, {tl}",
                          meta={"counts": counts, "rule": rule})
        marks = [("bars", "channel-cycles left after each gate"),
                 ("grey boxes and numbers", "channel-cycles dropped at that gate"),
                 ("light bands", "flow to the next gate")]
    n_tot = {d: sum(v.values()) for d, v in counts.items()}
    n_use = {d: v["usable"] for d, v in counts.items()}
    return attach(spec, CaptionSpec(
        f"{tl.capitalize()}: how the recorded channel-cycles (openers for the opening, closers "
        f"for the closing; {_det_label()} detector, {RULE_LABEL[rule]} rule) become usable for "
        "the onset lead. The gates in order: the channel not active at the start of the onset "
        f"search window, a burst at least {burst_min:g} dB above rest, an onset found, and "
        "enough unmasked samples in the window; a channel-cycle is counted at the first gate "
        "it fails",
        marks,
        "n = " + "; ".join(f"{d}: {n_tot[d]} channel-cycles, {n_use[d]} usable "
                           f"({100 * n_use[d] / max(n_tot[d], 1):.0f} %)" for d in n_tot)
        + f"; {rows.pid.nunique()} participants",
        "counts and shares of channel-cycles", evidence="descriptive",
        notes=["Channel-cycles that fail a gate stay in the dataset and are flagged; they only "
               "do not enter the onset lead"]))


# == 12. kinematic phase portrait ==============================================================
@register("kin_phase_portrait", "dashboard", "Cohort phase portrait", domain="kinematics",
          choices={"form": ("velocity", "acceleration")}, form="velocity")
def kin_phase_portrait(req: ViewRequest) -> FigureSpec:
    """Density of the opening angle vs its angular velocity (or the velocity vs the
    acceleration) per task, every participant weighted equally."""
    ctx = req.ctx
    fid, title = "kin_phase_portrait", "Cohort phase portrait"
    form = req.options.get("form", "velocity")
    kx, ky = (("theta_deg", "theta_vel") if form == "velocity" else ("theta_vel", "theta_acc"))
    lab = {"theta_deg": ("Opening angle", V.DEG), "theta_vel": ("Angular velocity", V.DPS),
           "theta_acc": ("Angular acceleration", V.DPS2)}
    tasks = list(V.TASKS)
    data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {t: [] for t in tasks}
    for pid in _pids(ctx):
        a = _a(ctx, "kinematics", pid)
        for tk in tasks:
            if f"{tk}__{kx}" in a and f"{tk}__{ky}" in a:
                x = np.asarray(a[f"{tk}__{kx}"], float)
                y = np.asarray(a[f"{tk}__{ky}"], float)
                m = np.isfinite(x) & np.isfinite(y)
                if m.sum() > 10:
                    data[tk].append((x[m], y[m]))
    tasks = [t for t in tasks if data[t]]
    if not tasks:
        return _empty_spec(fid, title, "no stored opening-angle series", "phase portrait")
    allx = np.concatenate([x for t in tasks for x, _ in data[t]])
    ally = np.concatenate([y for t in tasks for _, y in data[t]])
    xl = (_q(allx, 0.5), _q(allx, 99.5))
    yl = (_q(ally, 0.5), _q(ally, 99.5))
    ex = np.linspace(*xl, PORTRAIT_BINS + 1)
    ey = np.linspace(*yl, PORTRAIT_BINS + 1)
    dens, ns = {}, {}
    for tk in tasks:
        H = []
        for x, y in data[tk]:
            h, _, _ = np.histogram2d(x, y, bins=[ex, ey])
            if h.sum() > 0:
                H.append(h / h.sum())
        D = 100.0 * np.mean(H, axis=0)                     # % of samples per cell
        dens[tk] = D
        ns[tk] = len(H)
    with np.errstate(divide="ignore"):
        logs = {t: np.log10(np.where(D > 0, D, np.nan)) for t, D in dens.items()}
    zl = (np.floor(min(_q(z, 5) for z in logs.values())),
          np.ceil(max(_q(z, 99.9) for z in logs.values()) * 2) / 2)
    panels = []
    for k, tk in enumerate(tasks):
        img = logs[tk].T[::-1]                             # rows: high y first
        t_ = TASK_LABEL.get(tk, tk)
        p = Panel(f"({'abcdef'[k]}) {t_[:1].upper() + t_[1:]}",
                  Axis(*lab[kx], lim=xl), Axis(*lab[ky], lim=yl),
                  [Series("image", img=img, extent=(xl[0], xl[1], yl[0], yl[1]),
                          style={"cmap": "sequential", "lim": zl})], legend="none",
                  colorbar=Colorbar("Sample share (log10 %)", "", lim=zl,
                                    ticks=[float(v) for v in np.arange(np.ceil(zl[0]),
                                                                       np.floor(zl[1]) + 1)]))
        p.series.append(Series("hline", y=np.array([0.0]), role="faint"))
        panels.append(p)
    cols = 3
    _grid_labels(panels, cols)
    for k, p in enumerate(panels):          # one shared scale: one colour bar, top right
        if k != min(cols, len(panels)) - 1:
            p.colorbar = None
    lp = float(_sig(ctx, "kinematics", "lowpass_hz", default=6.0))
    spec = FigureSpec(fid, title, panels, rows=int(np.ceil(len(panels) / cols)), cols=cols,
                      width="text", height_in=1.45,
                      sources=["kinematics:arrays", "kinematics:cycles"],
                      perspective="cohort, every task", meta={"n": ns})
    return attach(spec, CaptionSpec(
        ("Phase portrait of the opening: the opening angle against its angular velocity"
         if form == "velocity" else
         "Dynamics of the opening: the angular velocity against the angular acceleration")
        + f" per task (optical poses low-passed at {lp:g} Hz, all samples of the task), as a "
        f"density on a {PORTRAIT_BINS} × {PORTRAIT_BINS} grid: per participant the share of "
        "its samples in each cell, averaged over participants so every participant weighs the "
        "same; the dynamic envelope an exoskeleton has to follow",
        [("colour", "share of samples per cell (log10 of %)"), ("white", "no samples")],
        f"n = {max(ns.values())} participants", "mean of the per-participant densities",
        evidence="descriptive",
        notes=["Axes cover the 0.5th to 99.5th percentile of all samples; velocities and "
               "accelerations from the low-passed poses are lower bounds of the peaks"]))
