# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Built-in stages. Each one wires a context to pure analysis functions.

Keep logic out of this file: a stage reads its inputs, calls functions from the
analysis subpackages, and returns tables. Stage names are the public IDs used
by the CLI, the cache layout and the viewer.
"""

from __future__ import annotations

import pandas as pd

from biomech.emg.analysis import analyse as analyse_emg
from biomech.imu.analysis import analyse as analyse_imu
from biomech.timing.lead import analyse as analyse_timing
from biomech.labels.auto_labels import CLASSES
from biomech.uncertainty import montecarlo as mc
from biomech.kinematics.analysis import analyse as analyse_kinematics
from biomech.kinematics.pose_chain import KinParams
from biomech.pipeline import CohortContext, ParticipantContext, StageResult, stage
from biomech.qc import report as qc


@stage("qc", version="4", description="Per-participant data quality (an01 §3)")
def qc_stage(ctx: ParticipantContext) -> StageResult:
    p = ctx.params
    tables: dict[str, pd.DataFrame] = {}
    motion, mvc = ctx.session("motion"), ctx.session("mvc")
    sessions = [qc.session_table(motion, ctx.pid, "motion")]
    phases = [qc.phase_table(motion, ctx.pid, "motion")]
    if mvc is not None:
        sessions.append(qc.session_table(mvc, ctx.pid, "mvc"))
        phases.append(qc.phase_table(mvc, ctx.pid, "mvc"))
    tables["sessions"] = pd.concat(sessions, ignore_index=True)
    tables["phases"] = pd.concat(phases, ignore_index=True)
    tables["emg"] = qc.emg_table(ctx.emg_prep(), ctx.pid, p["usability"])
    from biomech.qc import impedance

    tables["emg_check"] = impedance.check_table([("motion", motion), ("mvc", mvc)], ctx.pid)
    tables["imu"] = qc.imu_table(motion, ctx.pid, p["imu"])
    tables["optical"] = qc.optical_table(motion, ctx.pid, p["optical"])
    # Stream-level products of the data views (G2): they render from the store alone.
    from biomech.qc import streams

    tables["imu_intervals"], tables["imu_coverage"] = streams.imu_streams(motion, ctx.pid)
    tables["timeline_lanes"], tables["timeline_segments"], arrays = streams.timeline(motion,
                                                                                      ctx.pid)
    arrays.update(streams.scene(motion))
    return StageResult(tables=tables, arrays=arrays)


@stage("qc_cohort", scope="cohort", depends=("qc",), version="3",
       description="Cohort QC overview, EMG usability flags, electrode impedance")
def qc_cohort_stage(ctx: CohortContext) -> StageResult:
    emg = ctx.upstream("qc", "emg")
    imu = ctx.upstream("qc", "imu")
    opt = ctx.upstream("qc", "optical")
    demo = pd.DataFrame([ctx.cohort.demographics()])
    flag_cols = ["usable_shape", "usable_amplitude", "rest_noise_limited"]
    flags = emg.loc[~emg["usable_shape"] | ~emg["usable_amplitude"] | emg["rest_noise_limited"]
                    | (emg["mask_frac_task"] > 0.01),
                    ["pid", "muscle", *flag_cols, "mask_frac_task", "snr_db",
                     "rest_rms_uv"]].reset_index(drop=True)
    motion_lock = (opt[opt["phase"].isin(qc.MOTION_PHASES)]
                   .groupby(["pid", "body"])["lock_frac"].mean().unstack("body")
                   .add_prefix("lock_").reset_index()) if not opt.empty else pd.DataFrame()
    quat_cov = (imu[imu["stream"] == "quat"].pivot(index="pid", columns="slot", values="coverage")
                .add_prefix("quat_cov_").reset_index()) if not imu.empty else pd.DataFrame()
    per_pid = emg.groupby("pid").agg(
        emg_unusable_amplitude_n=("usable_amplitude", lambda s: int((~s).sum())),
        emg_rest_noise_limited_n=("rest_noise_limited", "sum"),
        emg_mask_frac_task=("mask_frac_task", "mean")).reset_index()
    for extra in (motion_lock, quat_cov):
        if not extra.empty:
            per_pid = per_pid.merge(extra, on="pid", how="outer")
    from biomech.qc import impedance

    rec = ctx.upstream("qc", "emg_check")
    # B1 bench scale per channel (null unless the bench check asks for the erratum); a
    # cohort-stage parameter, so setting it re-runs this stage only
    chk = impedance.rescale(rec, (ctx.params or {}).get("impedance_scale"))
    tables = {"demographics": demo, "emg_flags": flags, "overview": per_pid,
              "impedance": impedance.summary(chk),
              "impedance_assoc": impedance.associations(chk, emg, seed=ctx.config.seed)}
    if chk is not rec:                    # rescaled: the recorded values stay reported
        tables["impedance_recorded"] = impedance.summary(rec)
    return StageResult(tables=tables)


@stage("kinematics", depends=("qc",), version="3",
       description="Jaw kinematics in the maxillary frame: cycles, clinical metrics, C1")
def kinematics_stage(ctx: ParticipantContext) -> StageResult:
    kp = KinParams.from_dict(ctx.signals.get("kinematics"))
    tables, arrays, jm = analyse_kinematics(ctx.session("motion"), ctx.pid, kp, ctx.params)
    ctx.memo("jaw_motion:" + repr(kp), lambda: jm)      # shared with `uncertainty`
    return StageResult(tables=tables, arrays=arrays)


@stage("emg", depends=("qc", "kinematics"), version="2",
       description="Activation maps, rest activity, clench indices, cycle profiles")
def emg_stage(ctx: ParticipantContext) -> StageResult:
    tables, arrays = analyse_emg(ctx.emg_prep(), ctx.pid, ctx.upstream("kinematics", "cycles"),
                                 ctx.upstream_arrays("kinematics"),
                                 ctx.upstream("qc", "emg"), ctx.params)
    return StageResult(tables=tables, arrays=arrays)


@stage("emg_relevance", depends=("emg",), version="1",
       description="EMG activity relevance per task and muscle (D19, descriptive)")
def emg_relevance_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.emg.relevance import relevance

    p = ctx.params
    return StageResult(tables={"relevance": relevance(
        ctx.emg_prep(), ctx.upstream("emg", "activation"), ctx.pid,
        k_sd=float(p.get("k_sd", 3.0)), min_pct=float(p.get("min_pct", 5.0)))})


@stage("acquisition", version="1",
       description="Acquisition quality of the motion take (X11, an20): camera frame drops "
                   "and the association counters, camera time base, IMU-to-camera lag, "
                   "stereo calibration in use, recording day")
def acquisition_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.qc import acquisition as acq
    from biomech.qc import stereo_cal as sc

    sd = ctx.session("motion")
    tables = acq.analyse(ctx.motion_path(), ctx.pid, "motion", sd.phases, sd.imu("mta"),
                         sd.optical(), sd.meta, ctx.params)
    h5 = ctx.motion_path() / "session.h5"
    cal, pairs = sc.session_calibration(h5), sc.corner_pairs(h5)
    row = {"pid": ctx.pid, "stored": cal is not None and pairs is not None}
    if row["stored"]:
        p = {**sc.DEFAULTS, **(ctx.params.get("stereo") or {})}
        row.update(sc.in_use(cal, sc.thin(pairs, int(p["max_pairs"]), int(p["seed"]))))
    tables["calibration"] = pd.DataFrame([row])
    return StageResult(tables=tables)


@stage("acquisition_cohort", scope="cohort", depends=("acquisition",), version="1",
       description="Acquisition quality over the cohort (X11, an20): pooled frame drops, time "
                   "base, IMU lag per participant, recording days, calibration cross test and "
                   "the trajectory sensitivity to the extrinsic calibration")
def acquisition_cohort_stage(ctx: CohortContext) -> StageResult:
    from biomech.qc import acquisition_cohort as ac

    paths = {p.pid: ctx.cohort.session_path(p.motion) for p in ctx.participants}
    tables = ac.analyse(ctx.upstream("acquisition", "cameras"),
                        ctx.upstream("acquisition", "counters"),
                        ctx.upstream("acquisition", "lag"),
                        ctx.upstream("acquisition", "lag_windows"),
                        ctx.upstream("acquisition", "recording"),
                        ctx.upstream("acquisition", "calibration"), paths, ctx.params,
                        int(ctx.config.seed or 0))
    return StageResult(tables=tables)


@stage("imu", depends=("qc", "kinematics", "acquisition"), version="3",
       description="IMU calibration, opening-angle estimators, onset signals (C3a)")
def imu_stage(ctx: ParticipantContext) -> StageResult:
    p = dict(ctx.params)
    # X11 sensitivity `imu_latency`: the IMU time base shifted by the participant's measured
    # IMU-to-camera lag (acquisition stage); off (null) in the pre-specified analysis.
    if p.get("latency_correction") == "measured":
        lag = ctx.upstream("acquisition", "lag")
        ms = float(lag["lag_ms"].iloc[0]) if not lag.empty else float("nan")
        p["latency_shift_us"] = int(round(ms * 1e3)) if ms == ms else 0
    tables, arrays = analyse_imu(ctx.session("motion"), ctx.pid,
                                 ctx.upstream_arrays("kinematics"),
                                 ctx.upstream("kinematics", "cycles"),
                                 ctx.upstream("kinematics", "opening_model"), p,
                                 ctx.upstream("kinematics", "opening_model_loco"))
    if p.get("latency_shift_us"):
        tables["latency_shift"] = pd.DataFrame([{"pid": ctx.pid,
                                                 "shift_ms": p["latency_shift_us"] * 1e-3}])
    return StageResult(tables=tables, arrays=arrays)


@stage("timing", depends=("kinematics", "imu", "emg"), version="4",
       description="Mechanical and EMG onsets, EMG-to-motion lead and its chance null (D23), "
                   "causal budget (C2); post hoc family C2-E with its surrogates (X6)")
def timing_stage(ctx: ParticipantContext) -> StageResult:
    tables, arrays = analyse_timing(ctx.emg_prep(), ctx.pid, ctx.upstream_arrays("imu"),
                                    ctx.upstream("kinematics", "cycles"), ctx.params,
                                    ctx.seed)
    if (ctx.params.get("c2e") or {}).get("enabled", False):
        from biomech.timing.c2e import analyse as analyse_c2e

        tables.update(analyse_c2e(ctx.emg_prep(), ctx.pid, ctx.upstream_arrays("imu"),
                                  ctx.upstream("kinematics", "cycles"), ctx.params, ctx.seed,
                                  ctx.upstream_arrays("kinematics")))
    return StageResult(tables=tables, arrays=arrays)


# detach: own task after the participant's chain in parallel runs (scheduling only).
@stage("uncertainty", depends=("kinematics",), version="3", detach=True,
       description="Monte-Carlo noise floor and CI for the hinge residual (G'1), "
                   "Monte-Carlo p, the D24/D25 null variants and the D33 condyle-hinge "
                   "nulls (bench_tmj, invivo_tmj: a perfect hinge about the axis through "
                   "the condyle points)")
def uncertainty_stage(ctx: ParticipantContext) -> StageResult:
    import numpy as np

    from biomech.kinematics.clinical import find_task_cycles
    from biomech.kinematics.coupling import hinge_data, hinge_fits, openings
    from biomech.kinematics.pose_chain import build

    p = ctx.params
    kp = KinParams.from_dict(ctx.signals.get("kinematics"))
    kparams = ctx.config.params("kinematics")
    anat = ctx.upstream("kinematics", "anatomy").iloc[0]
    sd = ctx.session("motion")
    jm = ctx.memo("jaw_motion:" + repr(kp), lambda: build(sd, kp))
    ph = sd.phase("open_close")
    tc = find_task_cycles(jm, "open_close", ph.t_start_us, ph.t_end_us, kparams["cycles"])[0]
    ops = openings(jm, tc)
    mf = kparams["hinge"]["max_frames_per_cycle"]
    A, B, _ = hinge_data(jm, ops, mf)
    import pandas as pd
    if len(ops) < 2 or A.shape[0] < 10:
        # Too few open-close cycles for the hinge models (as in kinematics): a NaN row,
        # the participant stays in the cohort tables with the reason.
        return StageResult(tables={"hinge_noise": pd.DataFrame([{
            "pid": ctx.pid, "n_openings": len(ops), "n_frames": int(A.shape[0]),
            "note": "too few open-close cycles for the hinge models"}])})
    fits = hinge_fits(A, B)
    observed = {k: r.rms(0) for k, r in fits.items()}
    h = fits["hinge_fit"]
    seed = ctx.seed                       # per participant: no shared draws
    rest = sd.phase("rest")
    T_rest = jm.T_raw[jm.window(rest.t_start_us, rest.t_end_us)] if rest is not None else None
    null = mc.hinge_nulls(
        jm, ops, h.n, h.s, T_rest,
        # no rest phase: no floor_* columns (kinematics), NaN floors, participant untested
        np.array([float(anat.get(f"floor_trans_sd_mm_raw_{a}", np.nan)) for a in "xyz"]),
        np.array([float(anat.get(f"floor_rot_sd_deg_raw_{a}", np.nan)) for a in "xyz"]),
        p, kp.lowpass_hz, kp.lowpass_order, mf, seed)
    ci = mc.observed_ci(jm, ops, p["n_boot"], p["tooth_sd_mm"], p["tmj_sd_mm"], mf, seed + 2)
    row = {"pid": ctx.pid, "n_openings": len(ops), "n_frames": int(A.shape[0]),
           **mc.summarize(observed, null, ci)}
    return StageResult(tables={"hinge_noise": pd.DataFrame([row])},
                       arrays={f"null__{k}": v for k, v in null.items()}
                       | {f"ci__{k}": v for k, v in ci.items()})


@stage("labels", depends=("kinematics",), version="2",
       description="Intent labels from optical incisal velocity (C3b ground truth)")
def labels_stage(ctx: ParticipantContext) -> StageResult:
    import numpy as np
    import pandas as pd

    from biomech.labels.auto_labels import compare_axis_scale

    kin = ctx.upstream_arrays("kinematics")
    arrays, rows, agree = {}, [], []
    phases = sorted({k.split("__")[0] for k in kin if k.endswith("__t_us")
                     and not k.startswith("coupling")})
    for ph in phases:
        t = kin[f"{ph}__t_us"].astype(np.int64)
        inc = kin[f"{ph}__incisor"].astype(np.float64)
        fs = 1e6 / np.median(np.diff(t))
        # D16: labels with the axis-scale floor, and their agreement with the pre-D16 rule.
        _, lab, ag = compare_axis_scale(ph, t, inc, fs, ctx.params)
        agree.append({"pid": ctx.pid, "phase": ph, "n": ag["n"],
                      "changed_frac": ag["changed_frac"], "kappa": ag["kappa"]})
        arrays[f"{ph}__t_us"] = t
        arrays[f"{ph}__label"] = lab
        for code, name in enumerate(CLASSES):
            rows.append({"pid": ctx.pid, "phase": ph, "label": name,
                         "seconds": float((lab == code).sum() / fs)})
        rows.append({"pid": ctx.pid, "phase": ph, "label": "UNTRACKED",
                     "seconds": float((lab < 0).sum() / fs)})
    return StageResult(tables={"counts": pd.DataFrame(rows),
                               "label_agreement": pd.DataFrame(agree)}, arrays=arrays)


@stage("labels_check", scope="cohort", depends=("labels", "kinematics"), version="1",
       description="Validity check of the intent labels: Cohen's kappa against a second, "
                   "independent rule on random windows (fixed and rest-floor thresholds)")
def labels_check_stage(ctx: CohortContext) -> StageResult:
    from biomech.labels.kappa import DEFAULTS, run_labels_check

    data = {}
    for p in ctx.participants:
        try:
            data[p.pid] = (ctx.upstream_arrays("labels", p.pid),
                           ctx.upstream_arrays("kinematics", p.pid))
        except (KeyError, FileNotFoundError, LookupError):
            continue
    return StageResult(tables=run_labels_check(data, {**DEFAULTS, **ctx.params}))


# -- cohort statistics ---------------------------------------------------------------------
@stage("stats_kinematics", scope="cohort", depends=("kinematics", "uncertainty"), version="6",
       description="S1 kinematic reference values, reliability, C1 hinge/FHA summaries, "
                   "C1 null tests (D24) and condyle sensitivities (D25, D29)")
def stats_kinematics_stage(ctx: CohortContext) -> StageResult:

    from biomech.stats import cohort_kinematics as ck

    cyc = ctx.upstream("kinematics", "cycles")
    anat = ctx.upstream("kinematics", "anatomy").drop(columns=["unit"], errors="ignore")
    geo = ctx.upstream("kinematics", "geometry").drop(columns=["unit"], errors="ignore")
    if not geo.empty:
        anat = anat.merge(geo, on="pid", how="left")
    subj = ck.subject_table(cyc, anat, ctx.demographics())
    kin_arr = {p.pid: ctx.upstream_arrays("kinematics", p.pid) for p in ctx.participants}
    for extra in (ck.dynamics_envelope(kin_arr), ck.posterior_excursion(kin_arr)):
        if not extra.empty:                              # X7
            subj = subj.merge(extra, on="pid", how="left")
    metrics = list(ck.METRICS) + ["chewing_freq_hz", "opening_cv", "lateral_symmetry_pct",
                                  "opening_per_intermolar", "max_observed_opening_mm",
                                  "upper_intermolar_mm", "intercondylar_mm",
                                  "condyle_incisor_left_mm", "condyle_incisor_right_mm",
                                  "floor_incisor_rms_mm_smooth", *ck.GEOMETRY_METRICS]
    tables = {"subjects": subj.drop(columns=["sex", "age"]),
              "reference_values": ck.reference_values(subj, metrics, ctx.config.seed),
              "influence": ck.influence(subj, ["opening_selfpaced_mm", "protrusion_mm",
                                               "laterotrusion_left_mm", "peak_open_vel_dps"],
                                        ctx.params.get("older_pids", [])),
              "reliability": ck.reliability(cyc)}
    from biomech.norms.literature import compare

    cmp = compare(subj)
    tables["norms_subjects"] = cmp
    if not cmp.empty:
        tables["norms_summary"] = (cmp.groupby(["metric", "ref_mean", "source"])
                                   .agg(n=("diff", "size"), mean_diff=("diff", "mean"),
                                        sd_diff=("diff", "std")).reset_index())
    plausible = set(subj.loc[pd.to_numeric(subj.condyles_plausible, errors="coerce").eq(1),
                             "pid"]) if "condyles_plausible" in subj else None
    tables.update(ck.hinge_tables(ctx.upstream("kinematics", "hinge"),
                                  ctx.upstream("uncertainty", "hinge_noise"),
                                  ctx.upstream("kinematics", "fha"), ctx.config.seed,
                                  plausible))
    tables["condyle_sensitivity"] = ck.condyle_sensitivity(
        subj, ctx.upstream("kinematics", "hinge"), ctx.config.seed,
        tables.get("fha_subject"))
    curves, grid = {}, None
    for p in ctx.participants:
        a = ctx.upstream_arrays("kinematics", p.pid)
        grid = a.get("coupling__grid_deg", grid)
        for k in ("opening_mm", "tmj_mid_ant_mm", "tmj_mid_disp_mm"):
            curves.setdefault(k, []).append(a.get(f"coupling__{k}"))
    arrays = ck.coupling_band(curves, grid) if grid is not None else {}
    if grid is not None:                                 # X9
        tables["coupling_summary"] = ck.coupling_summary(
            curves, grid, [p.pid for p in ctx.participants], plausible)
    return StageResult(tables=tables, arrays=arrays)


@stage("stats_emg", scope="cohort", depends=("emg", "qc"), version="1",
       description="Cohort activation, rest activity, indices, co-activation, SPM tests")
def stats_emg_stage(ctx: CohortContext) -> StageResult:
    from biomech.stats import cohort_emg as ce

    seed = ctx.config.seed
    refs = ctx.upstream("emg", "refs")
    tables = ce.activation_tables(ctx.upstream("emg", "activation"), refs, seed)
    tables["indices"] = ce.indices_table(ctx.upstream("emg", "indices"), seed)
    tables["cci"] = ce.cci_table(ctx.upstream("emg", "cycles"), seed)
    tables["mask_validation"] = ce.mask_validation_table(ctx.upstream("emg", "mask_validation"))
    profiles, shape_ok = {}, {}
    for p in ctx.participants:
        a = ctx.upstream_arrays("emg", p.pid)
        profiles[p.pid] = {k.replace("profile__", ""): v for k, v in a.items()
                           if k.startswith("profile__")}
        r = refs[refs.pid == p.pid]
        shape_ok[p.pid] = set(r[r.usable_shape].muscle)
    spm, arrays = ce.spm_tests(profiles, shape_ok)
    tables["spm"] = spm
    # Frequency domain (X1, exploratory).
    from biomech.emg.analysis import COARSE_EDGES

    windows, names = {}, {}
    for p in ctx.participants:
        a = ctx.upstream_arrays("emg", p.pid)
        windows[p.pid] = {k.split("__")[1]: v for k, v in a.items()
                          if k.startswith("spectral__") and k.endswith("__win")}
        names[p.pid] = [str(n) for n in a.get("profile__names", [])] or None
        if names[p.pid] is None:
            r = refs[refs.pid == p.pid]
            names[p.pid] = list(r.muscle)
    grid = 0.5 * (COARSE_EDGES[:-1] + COARSE_EDGES[1:])
    tables["spectral_cohort"], tables["spectral_spm"], sarr = ce.spectral_tables(
        ctx.upstream("emg", "spectral"), windows, names, shape_ok, grid)
    arrays.update(sarr)
    return StageResult(tables=tables, arrays=arrays)


@stage("stats_timing", scope="cohort", depends=("timing",), version="4",
       description="C2: cohort EMG-to-motion lead, its test against the chance null (D23), "
                   "detector/source sensitivity, causal budget; C2-E cohort tests (X6)")
def stats_timing_stage(ctx: CohortContext) -> StageResult:
    from biomech.stats.cohort_misc import timing_tables

    tables = timing_tables(ctx.upstream("timing", "summary"),
                           ctx.upstream("timing", "onsets"), ctx.config.seed,
                           ctx.upstream("timing", "ensemble"),
                           ctx.upstream("timing", "release_summary"),
                           params=ctx.params,
                           onset_rule=ctx.config.params("timing").get("onset_rule", "drive"),
                           lead_null=ctx.upstream("timing", "lead_null"))
    c2e = ctx.config.params("timing").get("c2e") or {}
    if c2e.get("enabled", False):
        from biomech.stats.cohort_c2e import c2e_cohort

        tables["c2e_cohort"] = c2e_cohort(ctx.upstream("timing", "c2e_null"),
                                          ctx.upstream("timing", "coupling"), c2e,
                                          ctx.config.seed)
    return StageResult(tables=tables)


@stage("stats_imu", scope="cohort", depends=("imu", "kinematics"), partial=("imu",),
       version="2",
       description="C3a: IMU angle agreement, opening (LOSO vs own model) Bland-Altman")
def stats_imu_stage(ctx: CohortContext) -> StageResult:
    # E32: ``imu`` is partial -- the IMU tables come from the participants with IMU data,
    # the opening models (generic LOSO model) from every participant with optics.
    from biomech.stats.cohort_misc import imu_tables

    return StageResult(tables=imu_tables(ctx.upstream("imu", "agreement"),
                                         ctx.upstream("imu", "calibration"),
                                         ctx.upstream("imu", "cycles"),
                                         ctx.upstream("kinematics", "opening_model"),
                                         ctx.config.seed))


@stage("requirements", scope="cohort",
       depends=("kinematics", "imu", "uncertainty", "stats_kinematics", "stats_timing",
                "stats_imu"), partial=("imu",), version="4",
       description="Derived exoskeleton requirements with statistic, CI and evidence level")
def requirements_stage(ctx: CohortContext) -> StageResult:
    # E32: ``imu`` is partial -- optical rows use every participant with optics, the IMU
    # bandwidth those with IMU data (each row's own n).
    from biomech.stats.requirements import (coverage_table, requirements_sensitivity,
                                            requirements_table)

    bandwidth = ctx.upstream("imu", "bandwidth")
    req = requirements_table(
        ctx.upstream("stats_kinematics", "subjects"), ctx.upstream("kinematics", "hinge"),
        ctx.upstream("uncertainty", "hinge_noise"), bandwidth,
        ctx.upstream("stats_timing", "timing_cohort"),
        ctx.upstream("stats_timing", "ensemble_cohort"),
        ctx.upstream("stats_timing", "release_cohort"),
        ctx.upstream("stats_imu", "imu_agreement_cohort"),
        ctx.upstream("stats_imu", "imu_opening_ba"), ctx.config.seed,
        c2e=ctx.upstream("stats_timing", "c2e_cohort"))
    return StageResult(tables={"requirements": req,
                               "requirements_sensitivity": requirements_sensitivity(
                                   ctx.upstream("kinematics", "cycles"), bandwidth, req),
                               "max_coverage": coverage_table()})


@stage("variability", scope="cohort",
       depends=("kinematics", "emg", "stats_kinematics", "stats_timing", "qc"), version="1",
       description="S2: specification curve, cycles needed, protocol heterogeneity")
def variability_stage(ctx: CohortContext) -> StageResult:
    import pandas as pd

    from biomech.stats import cohort_misc as cm

    spec = pd.concat([ctx.upstream("kinematics", "spec"), ctx.upstream("emg", "spec"),
                      ctx.upstream("stats_timing", "timing_spec")], ignore_index=True)
    lead = f"detector=tkeo|mech=lead_ms|onset={ctx.config.params('timing').get('onset_rule', 'drive')}"
    norm = ctx.config.signals.get("emg", {}).get("reference", "max_recorded")
    primary = {"mio_mm": "cycle_start|vertical|lowpass", "median_pct": f"norm={norm}",
               "open_close": lead, "chewing": lead}
    rel = ctx.upstream("stats_kinematics", "reliability")
    # Keyed by side too: laterotrusion has one MDC per side (an07, reliability per side).
    mdc = {(r.task, r.metric, str(getattr(r, "side", "") or "")): r.mdc95
           for r in rel.itertuples()}
    qc_phases = ctx.upstream("qc", "phases")
    phases = qc_phases.rename(columns={"name": "phase"}) if "name" in qc_phases else qc_phases
    return StageResult(tables={
        "spec_curve": cm.spec_curve(spec, primary),
        "spec_values": spec,
        "cycles_needed": cm.cycles_needed_table(ctx.upstream("kinematics", "cycles"), mdc,
                                                seed=int(ctx.config.seed or 0)),
        "protocol": cm.protocol_heterogeneity(phases, ctx.upstream("kinematics", "cycles"),
                                              ctx.upstream("emg", "refs")),
    })


@stage("validation", scope="cohort", depends=("qc",), version="1",
       description="G: reference-system agreement (Qualisys/Delsys subset) or backup status")
def validation_stage(ctx: CohortContext) -> StageResult:
    from biomech.stats.cohort_misc import validation_status

    return StageResult(tables={"status": validation_status(ctx.participants)})


# Stages of the system-validation study (group "validation"; experiments/jtsmm_validation).
import biomech.validation.stages  # noqa: E402,F401  (registers ref_sync)
import biomech.imu.stage_camerafree  # noqa: E402,F401  (X5: camerafree, camerafree_cohort)
import biomech.imu.stage_dyncal  # noqa: E402,F401  (I3c audit stored: imu_dyncal, J4.6)
