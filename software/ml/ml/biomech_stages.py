# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``biomech`` pipeline stages provided by the ml package (entry point ``biomech.stages``).

Dependencies point downstream only: ml imports biomech, never the reverse.
The stages run through ``biomech.pipeline.run`` like any other, so they are
cached, appear in the status matrix and can be started from the viewer.

Both stages are thin adapters around the shared core: ``ml_windows`` hands
biomech's measured line frequency, rail/lead-off masks and IMU streams to
:func:`ml.core.dataset.build_windows`, with a normalization reference of the
definition biomech uses (``signals.emg.reference``) but computed on the *causal*
chain the windows use (:func:`_causal_reference`), so reference and features have
the same gain; ``ml_eval`` pools the stored windows (feature names per
participant, checked by :func:`~ml.core.dataset.concat`) into a
:class:`~ml.core.dataset.WindowDataset` and runs :mod:`ml.protocols.evaluate`, plus
the exploratory ``fusion_diagnostics``, ``permutation_importance``, ``variants``
and ``variants_summary`` tables (:mod:`ml.protocols.variants`).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

from biomech.emg.prepare import EmgPrep
from biomech.imu.preprocess import load as load_imu
from biomech.io.session import SessionData
from biomech.pipeline import CohortContext, ParticipantContext, StageResult, stage
from biomech.stats.core import describe, holm

from ml.config.classes import PHASES, REST_PHASE
from ml.core import signals as S
from ml.core.dataset import (Imu, WindowConfig, WindowDataset, build_windows, concat,
                             filter_phases)
from ml.evaluation import diagnostics as dg
from ml.protocols import evaluate as ev
from ml.protocols import variants as va

log = logging.getLogger(__name__)
TASK_PHASES = tuple(p for p in PHASES if p != REST_PHASE)     # biomech REFERENCE_TASKS


def ref_rules(prep: EmgPrep) -> tuple[str, ...]:
    """Artifact rules for the reference segments: biomech's analysis rules (D14) without the
    envelope rule, whose scale is the reference itself."""
    return tuple(r for r in prep.p.mask.analysis_rules if r != "envelope")
NORM_KEYS = ("emg", "seq")


def _causal_segment(sd: SessionData, phase: str, names: list[str], line: float,
                    mask: np.ndarray | None, cfg: WindowConfig) -> np.ndarray | None:
    """Phase EMG through the windows' causal chain (warm-up discarded), masked -> NaN."""
    ph = sd.phase(phase)
    if ph is None:
        return None
    t, uv, _ = sd.emg_window(max(ph.t_start_us - int(cfg.warmup_s * 1e6), 0), ph.t_end_us,
                             names)
    keep = t >= ph.t_start_us
    if mask is not None and mask.shape != (int(keep.sum()), len(names)):
        mask = None                                   # misaligned: unmasked, never misplaced
    full = np.zeros(uv.shape, dtype=bool)
    if mask is not None:
        full[keep] = mask
    x = S.causal_emg(t, S.hold_fill(uv, full), sd.emg_fs, line, hpf_hz=cfg.emg_band_hz[0],
                     lpf_hz=cfg.emg_band_hz[1])
    return np.where(full, np.nan, x)[keep]


def _causal_reference(prep: EmgPrep, sd: SessionData, mvc: SessionData | None,
                      names: list[str], mode: str, cfg: WindowConfig, dil: int,
                      task_masks: dict[str, np.ndarray], envelope_s: float,
                      filtered: dict[str, np.ndarray] | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Reference of biomech's definition ``mode`` on the causal chain of the windows.

    MVC phases come from the MVC take, tasks (``max_recorded`` / ``max1s``) from the
    motion take. Excluded samples: :func:`ref_rules` = biomech's analysis rules minus the
    envelope rule (its scale is the reference being determined), i.e. rail + lead-off
    under D14 (+ low-frequency in the ``lowfreq_mask`` sensitivity). ``filtered`` holds
    task phases the windows already filtered with the *same* mask
    (:func:`ml.core.dataset.filter_phases`); they are reused, not filtered again.
    """
    segs = {}
    if mvc is not None:
        for ph in ("mvc_opening", "mvc_closing"):
            if mvc.phase(ph) is not None:
                m = prep.phase(ph, take="mvc").masks.rules(*ref_rules(prep), dilate=dil)
                segs[ph] = _causal_segment(mvc, ph, names, prep.line_hz(mvc), m, cfg)
    tasks = {}
    if mode != "p95":
        for ph in TASK_PHASES:
            if (filtered or {}).get(ph) is not None:
                tasks[ph] = filtered[ph]
            elif sd.phase(ph) is not None:
                m = task_masks.get(ph)
                tasks[ph] = _causal_segment(sd, ph, names, prep.line_hz(sd), m, cfg)
    return S.emg_reference(segs, tasks, names, sd.emg_fs, mode=mode, envelope_s=envelope_s)


def _reusable(filt: dict[str, tuple[np.ndarray, np.ndarray]], masks: dict[str, np.ndarray],
              ref_masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Filtered window phases whose decoding mask equals the reference mask: the same
    signal :func:`_causal_segment` would produce (same samples, mask, line, chain)."""
    out = {}
    for ph, (_, x) in filt.items():
        a, b = masks.get(ph), ref_masks.get(ph)
        if (a is None and b is None) or (a is not None and b is not None
                                         and a.shape == b.shape and np.array_equal(a, b)):
            out[ph] = x
    return out


@stage("ml_windows", depends=("labels", "qc"), version="3",
       description="Causal EMG + IMU window features with intent labels (C3b)")
def ml_windows_stage(ctx: ParticipantContext) -> StageResult:
    cfg = WindowConfig.from_dict(ctx.params.get("window", {}))
    emg_sig = ctx.signals["emg"]
    ref_mode = str(emg_sig.get("reference", "max_recorded"))
    lab = ctx.upstream_arrays("labels")
    mvc_path = ctx.mvc_path()
    # Shared per-participant session and EMG preparation (cleaned once per chain).
    sd, mvc = ctx.session("motion"), ctx.session("mvc")
    prep = ctx.emg_prep()
    names = list(prep.names)
    # The rest phase is windowed even if unlabelled: thresholds and D13 statistics.
    phases = [(ph.name, ph.t_start_us, ph.t_end_us) for ph in sd.phases
              if ph.name in PHASES and (f"{ph.name}__label" in lab
                                        or ph.name == REST_PHASE)]
    dil = int(round(prep.p.mask.dilate_s * sd.emg_fs))
    mres = {ph: prep.phase(ph).masks for ph in
            dict.fromkeys([p for p, _, _ in phases]
                          + [p for p in TASK_PHASES if sd.phase(p) is not None])}
    # Only physically missing data (rail, lead-off) is masked for decoding:
    # movement artifacts stay, as a wearable decoder would see them.
    masks = {name: mres[name].rules("rail", "leadoff", dilate=dil)
             for name, _, _ in phases}
    t0 = max(min(t for _, t, _ in phases) - int(cfg.warmup_s * 1e6), 0)
    t1 = max(t for _, _, t in phases)
    emg_t, emg_uv, _ = sd.emg_window(t0, t1, names)
    # One causal filter pass per windowed phase: the windows and the reference share it.
    filt = filter_phases(phases, emg_t, emg_uv, sd.emg_fs, prep.line_hz(sd), masks, names,
                         cfg)
    ref_masks = {ph: m.rules(*ref_rules(prep), dilate=dil) for ph, m in mres.items()}
    reuse = _reusable(filt, masks, ref_masks)
    ref, ref_src = _causal_reference(
        prep, sd, mvc, names, ref_mode, cfg, dil, ref_masks,
        float(emg_sig.get("envelope_s", 0.15)), filtered=reuse)
    bad = ~(np.isfinite(ref) & (ref > 0))
    if bad.any():
        if cfg.emg_norm != "rest_relative":
            raise ValueError(f"no causal {ref_mode} reference for "
                             f"{[n for n, b in zip(names, bad) if b]} "
                             f"(MVC take: {mvc_path}); emg_norm 'mvc' needs one")
        # D13 features depend on the reference only through EPS = 1e-6 in log(amp + EPS)
        # and rounding: rest RMS keeps them defined.
        rest = filt[REST_PHASE][1] if REST_PHASE in filt else _causal_segment(
            sd, REST_PHASE, names, prep.line_hz(sd),
            mres[REST_PHASE].rules("rail", "leadoff", dilate=dil)
            if REST_PHASE in mres else None, cfg)
        if rest is not None:
            ref = np.where(bad, S.rest_reference(rest), ref)
            ref_src = np.where(bad, "rest", ref_src)
    mta, cra = load_imu(sd, "mta"), load_imu(sd, "cra")
    imu = None if mta is None or cra is None else Imu(
        (mta.t_q, mta.q), (cra.t_q, cra.q), (mta.t_a, mta.a), (cra.t_a, cra.a))
    labels = {name: (lab[f"{name}__t_us"], lab[f"{name}__label"])
              for name, _, _ in phases if f"{name}__label" in lab}
    w = build_windows(pid=ctx.pid, session=sd.folder.name, phases=phases,
                      emg_t=emg_t, emg_uv=emg_uv, emg_fs=sd.emg_fs,
                      line_hz=prep.line_hz(sd), ref_uv=ref, masks=masks,
                      channels=names, labels=labels, imu=imu, cfg=cfg, filtered=filt)
    meta = pd.DataFrame([{"pid": ctx.pid, "n_windows": len(w), **w.meta["rejected"],
                          "emg_norm": cfg.emg_norm, "n_rest_windows": w.meta["n_rest_windows"],
                          "ref_chain": "causal", "ref_mode": ref_mode}])
    refs = pd.DataFrame({"pid": ctx.pid, "muscle": names, "ref_uv": ref,
                         "source": [str(v) for v in ref_src], "thr": w.meta["thr"]})
    rows = [(b, n) for b in ("emg", "imu", "seq", "seq_imu") for n in w.names.get(b, [])]
    names_df = pd.DataFrame({"block": [b for b, _ in rows], "feature": [n for _, n in rows]})
    arrays = {f"X_{b}": v.astype(np.float32) for b, v in w.blocks.items()}
    arrays.update({"y": w.y.astype(np.int8), "t_end_us": w.t_end_us,
                   "phase": np.array([PHASES.index(p) for p in w.phase], dtype=np.int8)})
    for b, st in w.meta["norm"].items():
        arrays[f"norm_{b}_center"] = np.asarray(st["center"])
        arrays[f"norm_{b}_scale"] = np.asarray(st["scale"])
    return StageResult(tables={"meta": meta, "features": names_df, "refs": refs},
                       arrays=arrays)


def _cancel_fn(ctx):
    """``ctx.cancelled`` (cohort contexts of the pipeline) or None."""
    fn = getattr(ctx, "cancelled", None)
    return fn if callable(fn) else None


def check_cancelled(ctx) -> None:
    """Raise when the run was cancelled: the unit is then recorded as failed, never stored
    with partial tables."""
    fn = _cancel_fn(ctx)
    if fn is not None and fn():
        raise RuntimeError("cancelled")


def _pool(ctx: CohortContext) -> WindowDataset:
    """Stored windows of every participant, each with its *own* feature names.

    :func:`~ml.core.dataset.concat` then refuses participants whose names differ
    (a shared name list would hide a channel-order or feature-set mismatch).
    """
    feats = ctx.upstream("ml_windows", "features")
    unit = "unit" if "unit" in feats.columns else "pid"
    parts = []
    for p in ctx.participants:
        a = ctx.upstream_arrays("ml_windows", p.pid)
        if not a:
            continue
        f = feats[feats[unit] == p.pid] if not feats.empty else feats
        blocks = {k[2:]: v.astype(np.float64) for k, v in a.items() if k.startswith("X_")}
        names = {b: f[f.block == b].feature.tolist() for b in blocks
                 if not f.empty and (f.block == b).any()}
        y = a["y"].astype(np.int64)
        n = len(y)
        phase = np.array(PHASES)[a["phase"].astype(int)]
        norm = {b: {"center": a[f"norm_{b}_center"].tolist(),
                    "scale": a[f"norm_{b}_scale"].tolist()}
                for b in NORM_KEYS if f"norm_{b}_center" in a}
        parts.append(WindowDataset(
            blocks=blocks, y=y, pid=np.full(n, p.pid), session=np.full(n, p.pid), phase=phase,
            t_end_us=a["t_end_us"].astype(np.int64), groups=np.zeros(n, np.int64),
            names=names, meta={"pid": p.pid, "session": p.pid, "emg_norm": "rest_relative" if norm else "mvc",
                               "norm": norm}))
    if not parts:
        raise ValueError("no ml_windows output for any participant")
    for p in parts:
        missing = sorted(set(p.blocks) - set(p.names))
        if missing:
            raise ValueError(f"{p.meta['pid']}: ml_windows has no feature names for blocks "
                             f"{missing}; re-run ml_windows")
    return concat(parts)


def model_cache_dir(ctx) -> str | None:
    """``<out>/model_cache`` of the run's result store (:mod:`ml.models.cache`), or None."""
    root = getattr(getattr(ctx, "store", None), "root", None)
    return None if root is None else str(root / "model_cache")


def _online_models(ctx) -> set[tuple[str, str]]:
    """``(model, feature set)`` of the fold models ``ml_online`` will need."""
    from ml.online.stage import config as online_config

    try:
        o = online_config(ctx.config.params("ml_online"))
    except Exception:                       # ml_online misconfigured: it reports that itself
        return set()
    return {(str(o["model"]), str(f)) for f in o["feature_sets"]}


@stage("ml_eval", scope="cohort", depends=("ml_windows",), version="7",
       description="LOSO / blocked intent decoding, ablations, latency, permutation (C3b)")
def ml_eval_stage(ctx: CohortContext) -> StageResult:
    p = ctx.params
    d = _pool(ctx)
    seed = ctx.config.seed
    diag = dict(p.get("diagnostics") or {})
    imp = diag.get("importance")                   # {model, feature_set, n_repeats} | None
    want_imp = (str(imp["model"]), str(imp["feature_set"])) if imp else None
    configs = {m: list(f) for m, f in p["configs"].items()}
    if want_imp and want_imp[1] not in configs.get(want_imp[0], []):
        configs.setdefault(want_imp[0], []).append(want_imp[1])
    plan = [(model, fset, int(imp.get("n_repeats", 5)) if want_imp == (model, fset) else 0)
            for model, fsets in configs.items() for fset in fsets]
    # Every configuration's nested LOSO in one pool (longest fits first, no barriers); the
    # fold models ml_online uses go to the model cache (it loads instead of refitting).
    check_cancelled(ctx)
    t_loso = time.perf_counter()
    rows = [r for rs in ev.loso_many(d, plan, seed=seed, inner=p["inner_folds"],
                                     max_train=p["max_train"], n_jobs=p["n_jobs"],
                                     cancelled=_cancel_fn(ctx), cache_dir=model_cache_dir(ctx),
                                     cache_models=_online_models(ctx))
            for r in rs]
    log.info("ml_eval nested LOSO: %d configurations in %.0f s wall", len(plan),
             time.perf_counter() - t_loso)
    importance_df = ev.importance_table(rows)
    # Exploratory decoding variants (SVM, same LOSO folds, fold choices of the main run;
    # few-shot k = 0 reuses the main run's predictions, which are that very model's).
    check_cancelled(ctx)
    var_cfg = va.config(p.get("variants", {}))
    t_var = time.perf_counter()
    variants_df = va.run_variants(d, var_cfg, va.main_params(rows), seed=seed,
                                  inner=p["inner_folds"], max_train=p["max_train"],
                                  n_jobs=p["n_jobs"], main_pred=va.main_predictions(rows),
                                  cancelled=_cancel_fn(ctx))
    if not variants_df.empty:
        log.info("ml_eval variants: %d rows in %.0f s wall", len(variants_df),
                 time.perf_counter() - t_var)
    variants_summary = va.summarize_variants(variants_df)
    fusion_df = dg.fusion_diagnostics(d) if diag.get("fusion_variance", True) else \
        pd.DataFrame()
    loso_df, conf = ev.summarize(rows)
    if var_cfg is not None and var_cfg.get("muscle_ablation") is not None:
        # X10: per left-out muscle group, paired with the full feature set's main run
        ab = va.ablation_deltas(variants_df, loso_df,
                                feature_set=str(var_cfg["muscle_ablation"]["feature_set"]))
        variants_summary = pd.concat([variants_summary, ab], ignore_index=True)
    # Blocked CV with the hyper-parameters the nested LOSO search chose for the fold that
    # held the subject out (like-for-like with LOSO); the r9 default-hyper-parameter run is
    # kept as a sensitivity (table blocked_default, summary protocol "blocked_default").
    # Both runs of every configuration in one pool (fold fits they share run once).
    check_cancelled(ctx)
    specs = [s for model, fset in p["blocked"]
             for s in ((fset, model, va.main_params(rows, model=model).get(fset) or None),
                       (fset, model, None))]
    blk_all = ev.blocked_many(d, specs, seed=seed, n_jobs=p["n_jobs"],
                              cancelled=_cancel_fn(ctx)) if specs else []
    blocked_df = pd.DataFrame([r for t in blk_all[0::2] for r in t])
    blocked_default_df = pd.DataFrame([r for t in blk_all[1::2] for r in t])

    # Cohort summaries per configuration.
    summ = []
    for (fset, model), g in loso_df.groupby(["feature_set", "model"]):
        for m in ("balanced_accuracy", "macro_f1", "free_balanced_accuracy", "latency_ms_median",
                  "detected_frac", "fa_frac", "fa_per_min"):
            if m in g:
                summ.append({"protocol": "loso", "feature_set": fset, "model": model,
                             "metric": m, **describe(g[m].to_numpy(), seed=seed, stat="median")})
    for proto, bdf in (("blocked", blocked_df), ("blocked_default", blocked_default_df)):
        for (fset, model), g in bdf.groupby(["feature_set", "model"]) if not bdf.empty \
                else []:
            summ.append({"protocol": proto, "feature_set": fset, "model": model,
                         "metric": "balanced_accuracy",
                         **describe(g.balanced_accuracy.to_numpy(), seed=seed, stat="median")})
    summary = pd.DataFrame(summ)

    # Feature-set comparison for the primary model (Friedman + Holm-Wilcoxon vs emg_imu).
    tests = []
    prim = p["primary_model"]
    w = loso_df[loso_df.model == prim].pivot(index="pid", columns="feature_set",
                                             values="balanced_accuracy").dropna()
    if w.shape[1] >= 3 and len(w) >= 5:
        fr = sps.friedmanchisquare(*[w[c] for c in w.columns])
        tests.append({"test": f"friedman|{prim}", "stat": float(fr.statistic),
                      "p": float(fr.pvalue), "n": int(len(w))})
        others = [c for c in w.columns if c != "emg_imu"]
        ps = [float(sps.wilcoxon(w["emg_imu"], w[c]).pvalue) for c in others]
        for c, pv, pa in zip(others, ps, holm(ps)):
            tests.append({"test": f"wilcoxon|{prim}|emg_imu_vs_{c}",
                          "stat": float(np.median(w["emg_imu"] - w[c])), "p": pv, "p_holm": pa,
                          "n": int(len(w))})
    null = np.array([])
    if p["n_perm"] > 0:
        # The observed statistic first: without it the test is skipped (no row), never
        # p = 1 / (n + 1) against a NaN.
        obs = loso_df[(loso_df.model == "lda") & (loso_df.feature_set == "emg_imu")] \
            if not loso_df.empty else loso_df
        m_obs = float(obs.balanced_accuracy.mean()) if len(obs) else float("nan")
        if np.isfinite(m_obs):
            check_cancelled(ctx)
            null = ev.permutation(d, "emg_imu", "lda", p["n_perm"], inner=p["inner_folds"],
                                  max_train=p["max_train"], seed=seed, n_jobs=p["n_jobs"],
                                  cancelled=_cancel_fn(ctx))
        else:
            log.warning("ml_eval: no finite observed lda|emg_imu balanced accuracy "
                        "(%d rows); permutation test skipped", len(obs))
    if null.size:
        tests.append({"test": "permutation|lda|emg_imu", "stat": m_obs,
                      "p": float((1 + np.sum(null >= m_obs)) / (1 + null.size)),
                      "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                      "n": int(null.size)})
    # Few-shot EMG + IMU vs IMU per calibration amount (paired, Holm over the k), appended
    # after the existing rows.
    tests += va.few_shot_tests(variants_df)
    # Per-subject accuracy by sex/age (aggregate only, analysis_plan §2.1 item 7).
    demo = ctx.demographics()
    best = loso_df[(loso_df.model == prim) & (loso_df.feature_set == "emg_imu")].merge(demo,
                                                                                         on="pid")
    gen = []
    if len(best) >= 5:
        rho = sps.spearmanr(best.age, best.balanced_accuracy)
        gen.append({"item": "spearman_age_vs_ba", "value": float(rho.statistic),
                    "p": float(rho.pvalue)})
        for sex, g in best.groupby("sex"):
            gen.append({"item": f"median_ba_{sex}", "value": float(g.balanced_accuracy.median()),
                        "n": int(len(g))})
    arrays = {f"confusion__{k}": v for k, v in conf.items()}
    # Window-level decoder output per participant (primary model): the viewer's decision
    # timeline and the video overlay.
    for r in rows:
        if r.get("model") == prim and "_pred" in r:
            q = r["_pred"]
            base = f"pred__{r['pid']}__{r['feature_set']}"
            arrays[f"{base}__t_us"] = np.asarray(q["t_end_us"], np.int64)
            arrays[f"{base}__y"] = np.asarray(q["y"], np.int8)
            arrays[f"{base}__yhat"] = np.asarray(q["yhat"], np.int8)
    if null.size:
        arrays["permutation_null"] = null
    return StageResult(tables={"loso": loso_df, "blocked": blocked_df,
                               "blocked_default": blocked_default_df, "summary": summary,
                               "tests": pd.DataFrame(tests), "generalization": pd.DataFrame(gen),
                               "fusion_diagnostics": fusion_df,
                               "permutation_importance": importance_df,
                               "variants": variants_df, "variants_summary": variants_summary},
                       arrays=arrays)


# The pseudo-online evaluation (X4) registers its cohort stage on import (ml.online.stage).
from ml.online import stage as _online_stage  # noqa: E402,F401
