# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reference-system stages of the validation study beyond the sync: EMG comparability
(``ref_emg``, plan D5), trueness/precision (``ref_trueness``, D6) the IMU lever arm
from the hand-eye sessions (``ref_imu_lever``, VD15), the accelerometer against Qualisys
(``ref_imu_accel``, VD17), the hinge models on both systems (``ref_hinge``, VD21), the error
budget (``ref_budget``, VD22), the EMG side check (``ref_emg_side``, VD23) and the stored
diagnostics (``ref_diagnostics``, VD24). Registered by the
import at the end of :mod:`biomech.validation.stages`.

``ref_emg`` uses the dataset pipeline's own parameters (``signals.emg``,
``signals.kinematics``, ``stages.kinematics.cycles``, ``stages.timing``,
``stages.emg.cycle_mask_max``, ``stages.qc.usability``, ``stages.emg_relevance``: the package defaults unless the
study overrides them). Only ``signals`` and the ``ref_emg`` block are part of its cache key,
and the reference files (Delsys CSV, QTM ``.mat``) and the dataset takes read by path are
not: re-run with ``--force ref_emg`` after changing one of them. The same holds for the
ladder files of ``ref_trueness`` and the hand-eye sessions of ``ref_imu_lever``.
"""

from __future__ import annotations

from typing import Any

from biomech.pipeline import CohortContext, ParticipantContext, StageResult, stage
from biomech.validation.stages import load_reference, participant_reference, reference_path


def dataset_params(config: Any) -> dict[str, Any]:
    """The dataset pipeline's parameters that ``ref_emg`` applies to both systems."""
    return {"emg": dict(config.signals.get("emg") or {}),
            "kinematics": config.signals.get("kinematics"),
            "cycles": dict(config.params("kinematics").get("cycles") or {}),
            "timing": config.params("timing"),
            "cycle_mask_max": float(config.params("emg").get("cycle_mask_max", 0.10)),
            "usability": dict(config.params("qc").get("usability") or {}),
            "relevance": dict(config.params("emg_relevance") or {})}


@stage("ref_emg", depends=["ref_sync"], version="2", group="validation",
       description="V-D5: EMG comparability, JTS-MM dataset take vs Delsys comparison take "
                   "(each to its own MVC): MVC levels, profiles (r, CMC, RMS, SPM), onsets, "
                   "signal quality; Delsys clock via δ check and the ref_sync model")
def ref_emg_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.emg import DEFAULTS, run_ref_emg

    p = {**DEFAULTS, **ctx.params}
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    ref = participant_reference(doc, ctx.pid, ctx.motion_path())
    if not doc.get("dataset_dir"):
        raise LookupError("reference.yaml: dataset_dir not set")
    root = (ctx.config.data_dir / str(doc["dataset_dir"])).resolve()
    sync = ctx.upstream("ref_sync", "sync").iloc[0]
    tables, arrays = run_ref_emg(ctx.session("motion"), ref, root, sync, p,
                                 dataset_params(ctx.config), ctx.pid, ctx.seed)
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_imu_lever", scope="cohort", version="1", group="validation",
       description="VD15: IMU lever arm on its plate from the hand-eye "
                   "mounting sessions (accelerometer vs optical rigid-body acceleration), "
                   "cut-off sweep, block bootstrap, between-session repeatability")
def ref_imu_lever_stage(ctx: CohortContext) -> StageResult:
    from biomech.validation.lever import DEFAULTS, run_ref_imu_lever

    p = {**DEFAULTS, **ctx.params}
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    tables, arrays = run_ref_imu_lever(doc, ctx.config.data_dir, p, int(ctx.config.seed or 0))
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_imu_accel", depends=["ref_sync"], version="1", group="validation",
       description="VD17: IMU accelerometer vs the Qualisys acceleration of the "
                   "IMU site (hand-eye mounting and lever arm), leave-one-task-out calibration; "
                   "sensor, own-orientation and no-lever variants")
def ref_imu_accel_stage(ctx: ParticipantContext) -> StageResult:
    import numpy as np
    from scipy.spatial.transform import Rotation

    from biomech.validation import lever as LV
    from biomech.validation.accel import DEFAULTS, run_ref_imu_accel
    from biomech.validation.fusion import DEFAULTS as FD
    from biomech.validation.fusion import load_mounting
    from biomech.validation.stages import _systems, imu_mounting_files

    sy, p, *_ = _systems(ctx)
    pa = {**DEFAULTS, **(ctx.params.get("accel") or {})}
    files = imu_mounting_files(ctx, p)
    mounting, levers = {}, {}
    for name, f in files.items():
        slot = FD["imu_slots"][name]
        R = load_mounting(f, slot)
        d = LV.read_slot(f, slot) if f.suffix == ".h5" else None
        if R is None or d is None:                  # the lever arm needs a hand-eye session
            continue
        M, y, t = LV.design(d, float(pa["fc_hz"]), LV.DEFAULTS)
        lv = LV.fit(M, y, t, {**LV.DEFAULTS, "n_boot": 0}, int(ctx.config.seed or 0))
        if "r_x_mm" in lv:
            mounting[name] = R
            levers[name] = np.array([lv[f"r_{ax}_mm"] for ax in "xyz"])
    reg = ctx.upstream("ref_sync", "registration")
    Y = {}
    for name in FD["imu_slots"]:
        r = reg[reg["transform"] == f"Y_{name}"]
        if not r.empty:
            r = r.iloc[0]
            T = np.eye(4)
            T[:3, :3] = Rotation.from_rotvec(np.radians([r.rx_deg, r.ry_deg, r.rz_deg])).as_matrix()
            T[:3, 3] = [r.tx_mm, r.ty_mm, r.tz_mm]
            Y[name] = T
    tables, arrays = run_ref_imu_accel(sy, ctx.session("motion"), mounting, levers, Y,
                                       FD["imu_slots"], pa)
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_trueness", scope="cohort", version="1", group="validation",
       description="V-D6: trueness (step deviation, CI), precision (static SD, 30-s takes "
                   "detrended, drift) and accuracy on the hand-measured ladder; Qualisys, "
                   "JTS-MM once jtsmm_ladder_dir is set")
def ref_trueness_stage(ctx: CohortContext) -> StageResult:
    from biomech.validation.trueness import DEFAULTS, run_ref_trueness

    p = {**DEFAULTS, **ctx.params}
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    tables, arrays = run_ref_trueness(doc, ctx.config.data_dir, p)
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_hinge", depends=["ref_kinematics", "ref_agreement"], version="1",
       group="validation",
       description="VD21 (exploratory): the dataset's C1 hinge models (best-fit fixed axis, "
                   "axis through the condyle points) on the JTS-MM and the Qualisys "
                   "trajectories of the same openings")
def ref_hinge_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.hinge import run_ref_hinge

    mf = int(ctx.config.params("kinematics")["hinge"]["max_frames_per_cycle"])
    return StageResult(tables=run_ref_hinge(ctx.upstream_arrays("ref_kinematics"),
                                            ctx.upstream("ref_agreement", "cycles"), ctx.pid,
                                            max_frames=mf))


@stage("ref_budget", scope="cohort", depends=["ref_kinematics", "ref_landmarks", "ref_quality"],
       version="2", group="validation",
       description="VD22 (descriptive): error budget per participant (marking and motion "
                   "error split into the reference part σ_ref and the JTS-MM part) and the "
                   "marking frame-error gap between the participants with its causes")
def ref_budget_stage(ctx: CohortContext) -> StageResult:
    from biomech.validation.budget import DEFAULTS, run_ref_budget

    return StageResult(tables=run_ref_budget(
        ctx.upstream("ref_kinematics", "budget"), ctx.upstream("ref_landmarks", "marking_summary"),
        ctx.upstream("ref_landmarks", "hold_attribution"),
        ctx.upstream("ref_quality", "consistency"), ctx.upstream("ref_quality", "marker_geometry"),
        {**DEFAULTS, **ctx.params}))


@stage("ref_emg_side", depends=["ref_emg"], version="1", group="validation",
       description="VD23 (descriptive): EMG side check of the JTS-MM montage against Delsys -- "
                   "shape correlations of the mean cycle profiles per muscle pair and the side "
                   "index (+2 sides as in the code, -2 swapped)")
def ref_emg_side_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.emg_side import DEFAULTS, run_ref_emg_side

    return StageResult(tables=run_ref_emg_side(ctx.upstream_arrays("ref_emg"), ctx.pid,
                                               {**DEFAULTS, **ctx.params}))


@stage("ref_diagnostics", depends=["ref_kinematics", "ref_fusion"], version="1",
       group="validation",
       description="VD24 (descriptive): stored diagnostics quoted in the docs -- JTS-MM "
                   "frame timing, the IMU mounting audit (hand-eye residual per calibration "
                   "date, remount angle, excitation, fold-to-fold spread, imu-stage "
                   "estimator) and the validation part of the IMU dynamic-calibration audit")
def ref_diagnostics_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation import diagnostics as DG
    from biomech.validation.fusion import DEFAULTS as FD
    from biomech.validation.fusion import load_mounting
    from biomech.validation.stages import DEFAULTS as SD
    from biomech.validation.stages import imu_mounting_files

    p = {**SD, **DG.DEFAULTS, **ctx.params}
    session = ctx.session("motion")
    tables: dict[str, Any] = {}
    ka = ctx.upstream_arrays("ref_kinematics")
    tables["frame_timing"], tables["frame_gaps"] = DG.frame_timing(
        ctx.pid, ka["t_s"], float(p["long_interval_factor"]))
    # hand-eye sessions (reference.yaml imu_mounting_sessions) and the take's own (VD14)
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    he: dict[str, dict[str, Any]] = {}
    for rel in doc.get("imu_mounting_sessions") or []:
        f = ctx.config.data_dir / str(rel)
        if not f.exists():
            continue
        M = {sl: load_mounting(f, sl) for sl in FD["imu_slots"].values()}
        he[DG.session_label(f)] = {sl: R for sl, R in M.items() if R is not None}
    own_files = imu_mounting_files(ctx, p)
    own = DG.session_label(next(iter(own_files.values()))) if own_files else None
    if he:
        tables.update(DG.run_mounting(session, ctx.pid, he, own, p))
    tables.update(DG.run_dyncal(session, ctx.pid, ctx.upstream_arrays("ref_fusion"), p))
    return StageResult(tables=tables)


from biomech.validation import stages_camerafree  # noqa: E402,F401  (VD20: ref_camerafree)


@stage("ref_assoc", version="1", group="validation",
       description="VD26 (J14): acquisition quality of the comparison take and the "
                   "dataset take (drops, time base, IMU lag) and the frame-trigger "
                   "association under injected faults (replay through the recorder's "
                   "ClockAligner)")
def ref_assoc_stage(ctx: ParticipantContext) -> StageResult:
    from pathlib import Path

    import pandas as pd

    from biomech.io.session import SessionData
    from biomech.qc import acquisition as acq
    from biomech.validation import assoc_faults as AF

    p = {**AF.DEFAULTS, **(ctx.params.get("faults") or {})}
    doc = load_reference(reference_path(ctx.config, ctx.params.get("reference",
                                                                   "reference.yaml")))
    ref = participant_reference(doc, ctx.pid, ctx.motion_path())
    takes = {"comparison": Path(ctx.motion_path())}
    if doc.get("dataset_dir") and ref.get("dataset_take"):
        takes["dataset"] = (ctx.config.data_dir / str(doc["dataset_dir"])
                            / ref["dataset_take"]).resolve()
    out: dict[str, list[pd.DataFrame]] = {}
    for take, path in takes.items():
        sd = ctx.session("motion") if take == "comparison" else SessionData(path)
        try:
            tabs = acq.analyse(path, ctx.pid, take, sd.phases, sd.imu("mta"), sd.optical(),
                               sd.meta, ctx.params.get("acquisition"))
        finally:
            if take != "comparison":
                sd.close()
        for k, v in tabs.items():
            out.setdefault(k, []).append(v)
        cams, tid, tt = AF.load_recording(path / "session.h5")
        runs = AF.run_recording(cams, tid, tt, p)
        runs.insert(0, "take", take)
        runs.insert(0, "pid", ctx.pid)
        out.setdefault("fault_runs", []).append(runs)
    return StageResult(tables={k: pd.concat(v, ignore_index=True) for k, v in out.items()})


@stage("ref_assoc_cohort", scope="cohort", depends=["ref_assoc"], version="1",
       group="validation",
       description="VD26: fault-injection summary over the participants' takes (wrong and "
                   "unassigned frames per fault level, tolerance limits, clean replay and the "
                   "ablation without the cadence step)")
def ref_assoc_cohort_stage(ctx: CohortContext) -> StageResult:
    import pandas as pd

    from biomech.validation import assoc_faults as AF

    runs = ctx.upstream("ref_assoc", "fault_runs")
    levels, limits = AF.summarise(runs)
    base = runs[runs["fault"].isin(["clean", "cadence_off"])][
        ["pid", "take", "fault", "n", "correct_pct", "wrong_pct", "unassigned_pct"]
        + [c for c in runs.columns if c.startswith("cam") and c.endswith(("_wrong", "_n"))]]
    clean = runs[runs["fault"] == "clean"]
    summ = pd.DataFrame([{"n_recordings": int(clean[["pid", "take"]].drop_duplicates().shape[0]),
                          "n_runs": int(len(runs)),
                          "seeds": int(runs["seed"].max() + 1),
                          "clean_correct_pct_min": float(clean["correct_pct"].min()),
                          "cadence_off_wrong_pct_cam0_max": float(
                              (100 * runs[runs["fault"] == "cadence_off"]["cam0_wrong"]
                               / runs[runs["fault"] == "cadence_off"]["cam0_n"]).max())}])
    return StageResult(tables={"levels": levels, "limits": limits, "baseline": base,
                               "summary": summ})


@stage("ref_cal_repeat", scope="cohort", version="1", group="validation",
       description="VD27 (J14): repeatability of the calibration-tool pivot "
                   "calibration and of both IMU mountings over ten consecutive runs each, "
                   "every run refitted with the project's code")
def ref_cal_repeat_stage(ctx: CohortContext) -> StageResult:
    from biomech.validation.calrepeat import DEFAULTS, run_ref_cal_repeat

    return StageResult(tables=run_ref_cal_repeat(ctx.config.data_dir,
                                                 {**DEFAULTS, **ctx.params}))
