<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Analysis plan: JTS-MM system validation (frozen 2026-10-22)

> **Note of the public release.** This is the analysis plan as it was frozen, before any agreement result was looked at. Its text is unchanged, except for passages marked as shortened for participant privacy and for the neutral names that this release uses throughout. Pointers to internal working notes are left out. "The user" in the text is the responsible author. Every later change is listed in [change_log.md](change_log.md), with its reason and with whether results had been seen.

> **Status: frozen 2026-10-22 (see "Freeze"); was: DRAFT.** Sections 2 (selection rules) and 4 (tolerances) must be confirmed
> and the file tagged (`git tag validation-plan-frozen`) **before any agreement result is looked at**.
> The sync stage (`ref_sync`) only establishes the common time base and frames and reports no
> agreement outcome. Everything marked *proposed* needs confirmation.

## 1. Question and design

- How well does JTS-MM agree with Qualisys (motion) and Delsys Trigno (sEMG)? This is a
  technical validation (user: n = 2 is sufficient). Agreement is estimated within
  participant; there are no population claims.
  - Kinematic agreement uses simultaneous recordings.
  - EMG comparability is cross-session (only the Delsys electrodes were on the skin in the comparison takes).
- **Participants:** s10 and s14, one comparison take each (JTS-MM + Qualisys + Delsys at once).
  With n = 2, every inferential statement is per participant, with cycles as repetitions.
  Group-level claims are out of scope.
- **Separate study:** no data products are shared with `icorr2027`. The JTS-MM dataset takes of
  the same subjects are read by path (`reference.yaml`), never from the icorr2027 store.

## 2. Which data each metric uses (D0b): *fixed before results*

### 2.1 Sync and registration fit
- **Windows:** only the CT sync windows.
  - **Start:** the first CT episode inside `mand_point_1` (on/off bursts + continuous motion).
    The landmark hold that follows after a pause of more than 2.5 s is not used.
  - **End:** the first CT episode inside the last `free_motion` (end sync bursts + continuous motion).
- **Window bounds:** from the CT visibility runs, padded by 0.5 s and clipped to the phase.
- **Bodies:** the CT alone fits the clock (`c`, `α`) and `X` (V1). The joint CT+MP+HP fit is a sensitivity result.
- **Outliers:**
  - gross outliers (> 10 robust SD, e.g. a pose in another branch) get weight 0 and are counted (`n_gated`);
  - the remaining frames are Huber-weighted.

### 2.2 Kinematic agreement (D4): held out from the fit
- **Phases used:** only the movement tasks `open_close`, `protrusion_retrusion`, `left_right`, `chewing`, `free_motion`.
  Reported per task and pooled. `rest` is reported separately (static agreement and noise).
- **Excluded:**
  - the calibration and landmark holds (`mand_point_*`, `max_point_*`, `tmj_*`);
  - the CT sync windows;
  - in `free_motion`, everything from 0.5 s before the first CT appearance (a visibility run of at least 3 frames) on;
  - frames where either system lacks the body (no gap bridging: the Qualisys spline is evaluated only inside runs of valid frames);
  - the first and last 0.5 s of each phase.
- **Per-cycle features:** only cycles that are complete in both systems (`ok` cycles, the dataset rules).

### 2.3 Landmarks for the shared points
- Only the static part of each hold. The `mand_point_1` landmark uses the hold after the CT sync motion.

### 2.4 EMG comparability (D5)
- **Data sources:**
  - the EMG of the comparison takes is not used;
  - JTS-MM side: the dataset take of the same subject and its MVC take;
  - Delsys side: the comparison take and its MVC CSVs.
- **Per system:**
  - task activation: the same five movement tasks, cycle-based;
  - rest noise and SNR: the `rest` phase;
  - MVC: the MVC phases or segments only;
  - spectra: the task phases, mask-aware.
- **Normalization:** each system to its own MVC (largest-activity rule, D11).

### 2.5 Trueness and precision (D6)
- **Trueness:** static slot windows only, trimmed to the central 80 % of each 10-s take.
- **Precision:** from the 30-s takes, after removing a linear drift trend (the drift is reported separately).

## 3. Time base and frames (stage `ref_sync`)

- **Clock model:** `t_Q = t_S + c + α (t_S − t_ref)`.
  - `t_S` = JTS-MM cam0 trigger time + 6 ms, the exposure centre of the 12-ms exposure.
  - `t_ref` = the mean time of the fitted CT frames.
- **Registration:** `T_Q,b = X · T_S,b · Y_b`, b ∈ {CT, MP, HP}.
  `Y_CT` absorbs the CT tip offset: JTS-MM reports the tip, Qualisys the marker-cluster origin.
- **Estimation steps:**
  1. vote over the CT visibility edges;
  2. least-squares fit on the edges, with the rise/fall threshold bias as a parameter;
  3. CT `|ω|²` cross-correlation per window;
  4. robust Gauss–Newton fit of `c`, `α`, `X` and `Y_CT` on the CT alone (Qualisys spline at 200 Hz).
     The joint fit with MP/HP is stored as a sensitivity result (`fine_joint`, `joint_d_*`; V1).
  5. `Y_MP` and `Y_HP` for the agreement metrics: leave-one-task-out (§2.2, V2).
- **Uncertainty:**
  - `Σ = σ² (JᵀJ)⁻¹`, also reported inflated for the lag-1 residual autocorrelation (`*_ac`);
  - the spread of the sliding-window lags.
- **Checks:**
  - held-out residuals per movement phase, including the residual lag that best explains them;
  - sliding-window lags (15-s windows, 5-s step) from `|ω|²` (frame-free) and from the registered poses: both must lie on a line of slope `α`;
  - steps > 10 ms between neighbouring windows indicate dropped or re-associated frames;
  - a significant quadratic term indicates unstable clocks.

## 4. Tolerances for the equivalence view: *proposed — confirm before results*

| Quantity | Tolerance | Rationale |
|---|---|---|
| Translation (incisor, condyle points, MP/HP origin) | ±0.5 mm | bench precision scale of both systems |
| Rotation (opening angle, relative pose) | ±0.5° | same |
| Sync residual (held-out lag) | ±1 ms | ≤ 0.1 mm at 100 mm/s (cv05 §3.1 target) |
| Clock drift consistency (sliding slope vs α) | within 2 SE | — |

SPM tests the difference between the systems; it cannot show equivalence. Equivalence is
claimed only where the bootstrap 95 % band of the mean difference lies inside the tolerance band.

## 5. Statistics (D4–D5, to be detailed before the freeze)

- **Continuous:**
  - per-axis bias and RMS (mm, °);
  - error vs speed and vs position in the volume;
  - tracking availability.
- **Cycle features:** repeated-measures Bland–Altman with cluster-bootstrap CIs, Lin's CCC, ICC(A,1), CMC.
- **SPM** (spm1d, time-normalized cycles, per participant):
  - paired t-test JTS-MM vs Qualisys; two-sample t-test for the EMG profiles;
  - mean difference curve with a bootstrap 95 % band against the tolerances in §4.

## Decisions before the freeze (2026-10-22; read-only sync diagnostics had been seen)

The sync and registration diagnostics were seen. These are sync/registration diagnostics
(per-phase residuals), not the agreement outcomes of §5.

**V1. Clock and X from the CT alone.**
- **Observation:** while the CT moves next to the face, the head and jaw bodies disagree
  between the systems:
  - s14 HP 7–8° in the start sync, 3.5° in the end sync;
  - s10 MP 4–5° in the start sync.
- **Likely causes:** QTM label swaps or occlusion. s10 also showed a CT flip.
- **Rule:**
  - the CT-only fit is primary;
  - the joint fit is reported;
  - `joint_disagrees` flags a clock offset more than 2 SE (autocorrelation-inflated) away.

**V2. `Y_MP` / `Y_HP` by leave-one-task-out** (author decision; replaces "Y re-estimated on rest",
2026-10-21).
- **Method:**
  - each movement task, and `rest`, is evaluated with the robust closed-form `Y`
    (`registration.y_given_x`) fitted on the *other* movement tasks;
  - no sample is evaluated with a `Y` fitted on itself.
- **Sensitivity variants:** `Y` from rest, `Y` from the sync windows, and `Y` in-sample on
  all tasks.
- **Why:** the sync-window `Y` carries the corruption of V1. The s14 rest phase is itself
  off (HP 3° against 0.8° in `open_close`), so a rest-based `Y` does not transfer to the
  tasks.

**V3. The jaw-in-head error is reported at the lower incisor.**
- **Point used:** `mand_point_1`, always the incisor (user), digitized by the still-hold rule
  of §2.3 (`landmarks.digitize(hold="still")`).
- **Frame:** the maxillary frame of the comparison take.
- **Condyle points:** secondary.
- **Body-origin error:** a diagnostic only. The Qualisys MP origin lies 150–235 mm from the
  HP origin, so any rotation error turns into millimetres there.
- **Bug fixed:** the phase-median rule had put `mand_point_1` 300+ mm off, because the sync
  motion lies inside the phase.

**V4. Optical + IMU fusion is exploratory**, with these constraints:
- estimators F0 (optical only), F1 (SO(3) complementary filter) and F2 (error-state Kalman
  filter with a smoother);
- the time constant τ, the IMU calibration (Z, X_imu), the lag and the noise levels are
  fitted only on the sync windows and the training tasks of each leave-one-task-out fold;
- evaluated with the same metric as V2/V3.

**Amendments and new decisions (2026-10-22, user: compare anatomical frames).** *Proposed;
they must be confirmed at the freeze.*

**V5. Primary outcome from anatomical frames, built per system.**
- **Frames per system:** each system builds a maxillary frame from its own measurement of
  `max_point_1..3` (in its head body) and a mandibular frame from `mand_point_1..3` (in its
  jaw body). Same construction for both: origin at the incisor, x anterior, y left, z up;
  `*_point_1` is the incisor, `_2` the left molar, `_3` the right molar.
- **Jaw-in-head motion:** `M(t) = F_max⁻¹ · T_head(t)⁻¹ · T_jaw(t) · F_mand`.
- **Primary outcome** (§2.2 samples): the lower-incisor position and the mandibular rotation
  (rotation vector relative to rest), JTS-MM minus Qualisys.
- **Nothing is registered between the head or jaw bodies** of the two systems (no Y_HP/Y_MP),
  and X cancels.
- **The constant offset from the two frame definitions** is part of the result. It is
  reported as bias, and a bias-removed dynamic agreement (SD, LoA) is reported next to it.

**V2 (amended): leave-one-task-out Y is the secondary tracking-only analysis.**
- It removes the frame-definition difference.
- End-to-end (V5) minus tracking-only ≈ the digitization contribution.
- It is not the primary outcome.

**V6. Clock selection, fixed before agreement results.**
- **Primary:** the CT-only fit (the clock comes from an object that is never compared).
- **Sensitivity variants:** joint, two-stage (MP/HP frames within 3 robust SD of the
  stage-1 solution), hybrid, per-window, and Tukey loss. The sliding |ω| lag line is a
  check only.
- **Criteria for keeping the CT-only fit** (none uses the agreement outcome):
  - synthetic recovery with corrupted MP/HP (τ < 0.5 ms, α < 2 ppm);
  - start and end window clocks within 2 SE;
  - sliding |ω| slope consistent with α within 2 SE.
  - Failing one while the two-stage fit passes makes the two-stage fit primary.
- **Reported:** the primary outcome under every variant (robustness).

**V7 (revised). CT tip: each system's own verified tip is primary.**
- **Finding (inferred):** the `Y_CT` translation is 2.9 mm (s10) and 8.4 mm (s14) from
  the old joint fit, but both tips were pivot-calibrated (user).
- **Verification:** by the following records and checks.
  - Existing records: the JTS-MM tip per session in `/meta`, the Pi pivot JSON, the QTM
    body definition, the QTM pivot recordings.
  - From the recordings:
    - `Y_CT` stability and identifiability;
    - a hold pivot per system;
    - a cross-system landmark regression `Δp_h = R_CT,h·d`;
    - a frame-free inter-landmark distance estimator;
    - the Qualisys CT body rebuilt from the raw trajectories;
    - a trueness/precision recording check.
  - A fresh pivot only if the tool geometry is verified unchanged.
  - The data-based estimates of `d` must agree within their SEs.
- **Primary if verified:** each system's own tip, with acceptance `|Y_CT translation|`
  < 0.5 mm and within 2 SE after correction.
- **Otherwise:** the shared tip (JTS-MM tip via `Y_CT`), stated as a limitation.
- **Either way:** the other variant is reported as sensitivity.
- **Still holds:** each system detects its own still holds with the same rule.
- **Operationalised before the freeze (2026-10-22, after the tip diagnostics, before any V5
  number):**
  - **Uncertainty:** the uncertainty of every data-based tip estimate comes from a
    bootstrap over the landmark holds. The frames within a hold are not independent; the
    frame-level least-squares SEs had been 10–100× too small.
  - **"Verified"** means:
    - at least one data-based estimate is informative (bootstrap SE < 2 mm on every axis);
    - every informative estimate contains d = 0 (the system's own tip) in its 95 % region
      (χ²₃).
  - **Primary tip:** the own tip if verified, otherwise the shared tip.
  - **Reported alongside, descriptive:**
    - the QTM CT body origin relative to its marker centroid (both days 200.8 mm, the same
      definition: the origin was moved to a pivot-calibrated tip, as the user stated);
    - the within-body landmark-distance agreement of each tip variant.

**V8. Reference-quality gate and a second tolerance** (proposed).
- **Gate:** frames are excluded only by Qualisys's own quality, never by disagreement.
  Results are reported with and without the gate. JTS-MM quality is a covariate only.
  - **Thresholds** (fixed before results): a frame is excluded when, for HP or
    MP, any marker is missing or gap-filled, an inter-marker distance deviates from its
    median by more than 0.5 mm (≈ 5× the task SD of the head body), or a marker residual
    exceeds 1.0 mm.
  - The frame-type code meaning (missing / measured / gap-filled) is verified against the
    QTM export documentation before use.
- **Tolerance:** besides ±0.5 mm / ±0.5° (metrology), an application-derived bound from
  the dataset paper's requirements, and the relative error (% of each task's range of
  motion).
  - **Application bound (fixed 2026-10-22 from the frozen dataset snapshot `r5`, before
    any agreement result):** ±2.0 mm for the incisor and condyle points, ±1.2° for
    rotation.
  - **Derivation:**
    - 2.0 mm is the design value of the fixed-hinge misalignment (best-fit axis) and
      ≈ 5 % of the design incisal opening (39.2 mm);
    - 1.2° is 5 % of the design opening rotation (23.4°).
  - The authors approved the plan with this rule; the value is open to the user's confirmation.
- **QTM frame-type codes, verified:** code 0 coincides exactly with missing marker
  positions; 1 = measured; 2 = gap-filled.

**V9. Reference-uncertainty budget and adequacy** (proposed).
- **σ_ref per outcome** (incisor, rotation, condyle points) and segment comes from
  Qualisys-internal data only:
  - inter-marker distance deviations;
  - marker residuals;
  - viewing direction per camera in the body frame;
  - camera-support steps;
  - Monte-Carlo propagation through the marker geometry to the anatomical-frame outcome;
  - cross-checked against the rest-phase static noise and the D6 bench ladder.
- **Calibration quality** (wand-length SD, camera residuals) and the setup (3 cameras;
  3 flat ≈ 10 mm markers per body, on the JTS-MM plates, not coplanar with the ArUco
  markers) are reported.
- **Budget:** σ_obs² = σ_JTS² + σ_ref². The corrected σ̂_JTS = √(σ_obs² − σ_ref²) is
  reported next to σ_obs, never alone.
- **Adequacy:** the reference is adequate for an outcome when σ_obs / σ_ref ≥ 3 (≤ 5 %
  inflation). Otherwise the outcome is flagged, and the corrected estimate carries the
  interpretation.
- **Attribution analysis** (descriptive, after the freeze): the disagreement against both
  systems' quality metrics, orientation and speed.
- **Paper wording:** the flat-marker choice is described truthfully, as practical, with its
  advantages. The marker-geometry insensitivity follows from V5 (constant marker offsets
  cancel in per-system anatomical frames).

**V4 (amended): fusion settings fixed in advance.**
- The IMU calibration and the lag are fitted against JTS-MM optical only.
- τ and the noise levels come from the sensor characteristics (rest-phase noise, increment
  noise, drift), written down before the comparison.
- The Qualisys comparison is evaluation only. The τ curve is descriptive.

## Freeze (2026-10-22)

V1–V9 and §2/§4 are frozen with this commit (tag `validation-plan-frozen`). The user
approved the plan as a whole.

**What had been seen before the freeze:**
- the sync and registration diagnostics;
- the per-body and Y-registered jaw-in-head errors of the R5b diagnostics (superseded as
  primary by V5);
- the Qualisys-internal quality numbers (R5d.0).

No anatomical-frame (V5) agreement number had been computed.

**Code realising the frozen rules:**
- `biomech/validation/{anatomy,tip,quality,evaluate}.py`;
- stages `ref_tip`, `ref_quality`, `ref_kinematics` v2;
- `ref_sync` v2 with the clock variants and criteria.

## Deviations

(none yet)
