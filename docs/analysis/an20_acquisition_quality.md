<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an20 — Acquisition quality of the dataset (X11) and the integration of the earlier evaluation of the recording system (J14)

The acquisition quality of every dataset take: camera frame drops and the frame-trigger
association, the camera time base, the IMU-to-camera lag, the stereo calibration in use, the
recording days, and how much the trajectories depend on the stereo calibration. Stages
`acquisition` (per participant) and `acquisition_cohort` (dataset study); the systems-study
counterparts are VD26–VD28 (an14 §11e–§11h).

**Origin.** These analyses were first done in the earlier evaluation of the recording system
(change-log entry J14). They were ported into
`biomech` after a review of each method, not copied: every number below was recomputed with
the project's code from the recordings, and checked against the original evaluation scripts
(§6). Where the port deviates, the reason is given.

**Recording days.** The dataset was not recorded in one session: the 14 takes were recorded on
`\numAcqDays` (6) days within `\numAcqSpanDays` (13) days (the dates are
withheld from the artefacts, which use day 1 … 6), and the stereo rig was recalibrated between
sessions, also within a day (four distinct calibrations among the five stored ones, three of
them on day 5). Firmware (`1.0.0-prod`), recorder configuration and protocol were the same on
every day.

## 1. Frame drops and the frame-trigger association

- **Definition** (`qc/acquisition.camera_drops`). A camera's frames carry the trigger id the
  recorder assigned; an id step of k is k − 1 dropped frames. The start-up of the cameras
  (no frames for about 0.5 s after the start; cam0 first delivers one or two frames) is
  counted separately: every gap that ends within 1 s of the first trigger. The drop rate is
  the dropped share of the triggers of regular operation.
- **Association paths** (`association_counters`, from the recorder's `/sync` report): every
  drop must have been resolved by the cadence step (the SensorTimestamp advanced by more than
  one period), and every other path (slip, prediction, re-lock, lost trigger) unused. The
  check: `cadence_advance` = drops + start-up ids after the camera's first frame, per camera
  (`acquisition_cohort:counter_check`).
- **Result** (the 13 adults of the papers; all 14 takes in §6). Pooled drop rate `\numAcqDropRate` (3.0 %) of the triggers (per camera stream
  `\numAcqDropRateMin` (1.7 %) to `\numAcqDropRateMax` (5.5 %)); `\numAcqBurstSingle` (83 %) of the drop events are single
  frames, the longest run `\numAcqBurstMax` (4) frames. Every drop of all 28 camera streams (14 takes) was
  resolved by the cadence step; no other association path was used.

## 2. Camera time base

- **Definition** (`timebase_residuals`). The live clock fit of each camera
  (`jtsmm.sync.clock_align.ClockAligner._update_fit`: least squares of the trigger master time
  on the SensorTimestamp over a trailing window of 30 frames including the current one),
  replayed offline in vectorised form; the residual of each frame from its own window's fit.
  The replay equals the live fit to 0.001 µs (`tests/test_r8_j14_acquisition.py`). A frame paired
  with a neighbouring trigger would sit a whole period (25 ms) off; a constant whole-period
  shift of a take would be absorbed by the offset and is excluded by the IMU lag (§3), which
  is stamped independently of the association.
- **Result** (13 adults). Pooled RMS `\numAcqResidRms` (4.6 µs) (per camera stream `\numAcqResidRmsMin` (2.4 µs) to
  `\numAcqResidRmsMax` (9.3 µs)); `\numAcqResidSpikeShare` (0.03 %) of the frames above 50 µs; the largest residual
  `\numAcqResidMax` (267 µs) (a step of the Pi system-clock rate in one take, outside the tasks,
  followed by the fit within one window); inside the tasks at most `\numAcqResidTasksMax` (66 µs). cam1
  sits consistently above cam0 (about 5 vs 2.5 µs).

## 3. IMU-to-camera lag (per take; feeds the C2 latency chain, an05 §2)

- **Definition** (`imu_camera_lag`). The angular speed of the mandibular array from the optics
  (frame-to-frame rotation, frames at the exposure centre = trigger + 6 ms) and from the IMU game
  rotation vector (sample-to-sample), each body-frame component interpolated onto a 1 kHz grid
  and smoothed with the same centred Savitzky–Golay filter (175 ms, order 2) before the
  magnitude is taken; Pearson correlation over ±100 ms, parabolic peak refinement, over the
  movement tasks (open-close, protrusion, laterotrusion, chewing, free motion; never across
  task boundaries), and per task for the constancy. |ω| is frame-independent, so no hand-eye
  calibration enters. Positive: the IMU shows the motion later.
- **Deviations from the original evaluation script, and why.** (a) It smoothed |ω| itself; the 40 Hz
  pose differences then keep a positive noise floor and the peak correlation fell to 0.35–0.5
  in some takes; smoothing the components restores 0.86–0.98. (b) The IMU stream contains
  isolated quaternion jumps (up to 10⁵ °/s in single samples); sample-to-sample rotations
  faster than 1000 °/s are masked in both streams (counted per take). (c) 20-s windows held no
  data in most takes (the tasks last 10–24 s), so constancy is checked per task.
- **Verification.** On synthetic rotations with a known 7.3 ms delay and injected jumps the
  estimator returns it within 0.6 ms (`tests/test_r8_j14_acquisition.py`).
- **Result.** Median `\numAcqImuLag` (3.9 ms) over the 13 takes with jaw-IMU data (quartiles
  `\numAcqImuLagQOne` (3.5 ms) to `\numAcqImuLagQThree` (5.1 ms)), peak correlation at least `\numAcqImuLagRMin` (0.86), constant
  within a take (largest spread over its tasks `\numAcqImuLagTaskRange` (6.6 ms)). `\numAcqImuLagOutliers` (3) takes lag
  `\numAcqImuLagOutlierMin` (12.7 ms) to `\numAcqImuLagOutlierMax` (13.4 ms), `\numAcqImuLagExcess` (9.1 ms) above the others (`\numAcqImuLagRegularMin` (2.9 ms) to
  `\numAcqImuLagRegularMax` (5.1 ms)). They were recorded on the same day (day 2), but a fourth take of that day lies
  in the normal range, so it is not a day effect; firmware, IMU configuration and report rate
  are identical, and the cause is not identified. Against the trigger edge the lags are 6 ms
  larger (the E23 figure of ≈ 9.5 ms is the trigger-referenced median).
- **Consequence.** The pre-specified C2 does not correct the latency chain (E34). The X11
  sensitivity variant `imu_latency` (study.yaml) moves each participant's IMU stamps earlier by
  its own measured lag in the `imu` stage, so the IMU onsets of C2 and C3a are read on the
  optical time base: `\numLeadOpenFirstImuLat` / `\numLeadOpenDriveImuLat` against the primary
  `\numLeadOpenFirst` / `\numLeadOpenDrive` (an05 §2).

## 4. Stereo calibration in use (s10 on) and the cross test

- **Data.** From day 5 on the recorder stored the live tracker's camera model and every
  detected tag corner (`/optical/corners`); s01–s09 stored neither, so their calibration cannot
  be re-checked or the corners re-triangulated (a limitation of the released data, stated in
  the data descriptor). For them only the tracker's per-frame rigid-body fit residual remains
  (`qc:optical`, `optical/quality/*/reprojection_rms`, mm in triangulation mode), a coarse
  check.
- **Definition** (`qc/stereo_cal.py`). Symmetric epipolar distance of the corners seen by both
  cameras in the same frame (distance to the epipolar line in normalised coordinates times the
  focal length), on the recording the calibration was used for; it needs no scene knowledge
  and is blind to the baseline length (scale). Reprojection error after linear triangulation.
  Cross test: every stored calibration on every stored recording.
- **Result.** `\numAcqCalStored` (5) takes stored their calibration, `\numAcqCalDistinct` (4) distinct
  calibrations (labelled A–D in order of use). With its own calibration a take's median
  epipolar error is at most `\numAcqEpiOwnMax` (0.56) px; with any other take's calibration at
  least `\numAcqEpiForeignMin` (6.5) px: the metric separates matching from foreign calibrations, and the
  geometry changed between the calibrations (recalibration was necessary).
- **Shared calibration.** P10 and P11 (consecutive takes of day 5) carry the identical camera
  model (`\numAcqCalSharedN` takes; no recalibration between them); the other takes of days
  5–6 each have their own. The intrinsics of A–C are identical (only the extrinsics were
  recalibrated on day 5); D (day 6) has new intrinsics.
- **Takes of days 1–4: indirect check (J21).** The live tracker's rigid-fit residual
  is stored for every take. On the stored takes the corners re-triangulated with foreign
  calibrations raise the cranial residual from ≤ `\numCalFitOwnMaxCra` to ≥
  `\numCalFitForeignMinCra`. The mandibular residual does not separate. The days 1–4 takes'
  cranial residual (`\numCalFitEarlyMinCra`–`\numCalFitEarlyMaxCra`) puts
  `\numCalFitConsistentN` of `\numAcqCalEarlyN` within the own-calibration range: consistent
  with a matching calibration, but a coarse check.

## 5. Sensitivity of the trajectories to the extrinsic calibration (side note)

- **Definition** (`stereo_cal.sensitivity`). The recorded corners re-triangulated with the cam1
  pose perturbed by a random rotation and translation (per-axis N(0, σ²)), both arrays refitted
  per frame (weighted Kabsch on the tag model of `optical.yaml`), and the mandibular-array
  origin in the cranial frame compared with the unperturbed refit (RMS change over the take,
  rotation change, and the change of the range of motion per task: P99 distance from the rest
  position). 50 draws per take and scenario, intrinsics fixed.
- **Uncertainty model.** The final extrinsics are the mean of the inlier board snapshots
  (`optical.calibration.extrinsics`: "inliers are averaged"), so the expected uncertainty is
  their standard error, σ = RMS(snapshot deviation) / √3 / √n per axis, from the one bundle that
  stores its snapshots (n = 41: 0.53° and 8.5 mm RMS). Snapshots of one board session
  are not independent, so this is optimistic; the single-snapshot scatter (no √n) is reported
  as the conservative bound.
- **Result** (`\numCalSensN` (5) takes). Expected: the trajectory changes by `\numCalSensTraj` (0.33 mm) (median RMS;
  P95 `\numCalSensTrajPNinetyFive` (0.89 mm)), the rotation by `\numCalSensRot` (0.04°), the range of motion by at most
  `\numCalSensRom` (0.13 mm) (P95; open-close `\numCalSensRomOpen` (0.17 mm)); most of the trajectory change is a constant
  offset. Even under the single-snapshot scatter the range of motion changes by at most
  `\numCalSensSnapRom` (0.9 mm) (P95). The calibration is therefore not a relevant error source for the
  ranges of motion and requirements of the dataset.

## 6. Checks against the original evaluation scripts (2026-11-06)

Recomputed with the project's code from the same recordings (values of the earlier evaluation
in the right column; it counts 16 recordings including the two Qualisys comparison takes, the
dataset 14):

| Quantity | Here | Earlier evaluation |
|---|---|---|
| Drop rate per take, e.g. s10 cam0 / cam1 | 5.53 / 5.29 % | 5.53 / 5.29 % |
| Longest burst; single-frame share | 4; 83 % | 4; 84 % |
| Every drop by the cadence step | yes (28 streams) | yes |
| Time-base RMS, pooled; cam0 / cam1 per take | 4.4 µs; 2.4–3.3 / 4.6–5.7 µs (s01 8–9) | 4.4 µs; 2.4–3.4 / 4.4–5.7 µs |
| Largest residual | 267 µs (s01) | 267 µs (s01) |
| IMU lag, regular takes; s02 / s04 / s05 | 2.9–5.1 ms; 12.7–13.4 ms | 0.1–4.6 ms; 12.5–13.3 ms |
| Epipolar error in use, s10 … s14 | 0.17, 0.41, 0.44, 0.46, 0.56 px | 0.17, 0.41, 0.45, 0.46, 0.56 px |
| Cross test, foreign calibrations | 6.5–42 px | 6.5–51 px (incl. the Qualisys takes) |
| Calibration sensitivity, median trajectory change | 0.33 mm | 0.38 mm (incl. the Qualisys takes) |

The numbers in this table are a verification record of 2026-11-06; the
current values are the macros above. <!-- num-ok: verification record -->

## 7. Code and outputs

- `software/biomech/biomech/qc/acquisition.py`, `qc/stereo_cal.py`, `qc/acquisition_cohort.py`;
  stages `acquisition`, `acquisition_cohort` (`stages.py`); the `imu` stage's
  `latency_correction` (variant `imu_latency`).
- Tables: `acquisition:{cameras,counters,lag,lag_windows,recording,calibration}`,
  `acquisition_cohort:{summary,counter_check,per_session,days,calibrations,cross_test,
  sensitivity_draws,sensitivity_rom,sensitivity_summary}` (etc/tables.yaml).
- Paper: supplementary table `acq_table_sessions` (`suppTabAcquisition`), the `acq…`, `calSens…` and
  `…ImuLat` macros (numbers_dataset.yaml).
