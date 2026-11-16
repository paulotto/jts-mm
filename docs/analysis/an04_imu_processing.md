<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an04 — IMU processing, wearable opening (C3a) and the camera-free study (X5)

How the two jaw-system inertial measurement units (IMUs) are preprocessed and calibrated, how
the IMU opening angle and incisal opening are derived and compared with the optics (the
wearable-opening outcome, C3a), which IMU signals feed timing and decoding, the movement
bandwidth, and the camera-free jaw-pose study (X5). This doc is the home of the IMU and
camera-free methods; symbols are listed in §0 (Notation).

**Abbreviations.** EMG = electromyography; BCa = bias-corrected and accelerated (bootstrap);
CI = confidence interval; RMS(E) = root mean square (error); SD = standard deviation; MAD =
median absolute deviation; QC = quality control; LOSO = leave one subject out; OLS = ordinary
least squares; MOVER = method of variance estimates recovery; RTS = Rauch–Tung–Striebel
(smoother); TMJ = temporomandibular joint; `cra` / `mta` = head / mandible IMU.

Stages and code:

| Stage | Scope | Code | Parameters |
|---|---|---|---|
| `imu` | participant | `biomech/imu/preprocess.py`, `calibration.py`, `analysis.py`; primitives `jtsmm.imu.batch`, `jtsmm.imu.tilt`, `jtsmm.imu.mounting` | `stages.imu` in `software/biomech/biomech/etc/biomech.yaml`: `calibration_phase` = `free_motion`, `grid_hz` = 200, `quasi_static_tol` = 0.05, `tau_s` = 1.0, `omega_smooth_s` = 0.02; `stages.qc.imu.gap_s` = 0.05 |
| `stats_imu` | cohort | `biomech/stats/cohort_misc.py:imu_tables` | — |
| `camerafree` | participant | `biomech/imu/camerafree.py` (methods), `biomech/imu/stage_camerafree.py` (stage), `biomech/imu/lever.py` (lever arms); `jtsmm.imu.position.position_kf` | `stages.camerafree` in `experiments/icorr2027/study.yaml` (code defaults `stage_camerafree.DEFAULTS`) |
| `camerafree_cohort` | cohort | `biomech/stats/cohort_camerafree.py` | `stages.camerafree_cohort` (`exclude_generic` = [], `n_boot` = 4000) |
| `imu_dyncal` | participant (depends on `camerafree`) | `biomech/imu/stage_dyncal.py`, `biomech/imu/dyncal.py` (helpers shared with the validation stage `ref_diagnostics`) | none: `dyncal.DEFAULTS` (hard-coded, §1) |
| `imu_dyncal_cohort` | cohort | `biomech/imu/stage_dyncal.py:transition_test` | `STILL_MIN_S` = 2 s (module constant) |

Related documents: rigid-body and SE(3) primitives, the maxillary frame, the opening angle
θ_y, `T_closed`, the pose low-pass and the opening model: [an02](an02_frames_and_kinematics.md).
Generic statistics (BCa bootstrap, Hodges–Lehmann, Wilcoxon, Holm, repeated-measures
Bland–Altman, MOVER, mixed model): [an07](an07_statistics.md). The validation-study fusion
filters (F1/F2), the hand-eye mounting (VD14), the lever-arm method (VD15) and the camera-free
estimators against Qualisys (VD20): [an14](an14_system_validation.md) §9, §11a, §11b.

## 0. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf R_{\mathrm{cra}}(t),\ \mathbf R_{\mathrm{mta}}(t)$ | orientation (world←chip) of the head (`cra`) and mandible (`mta`) BNO085 from its game rotation vector; each IMU has its own world (gravity-aligned $z$, arbitrary heading) |
| $\mathbf f_{\mathrm{cra}}(t),\ \mathbf f_{\mathrm{mta}}(t)$ | raw accelerometer (specific force, gravity included), chip frame, m/s² |
| $g = 9.80665$ m/s², $\mathbf e_z = (0,0,1)^\top$ | gravity; world vertical |
| $\mathbf R_z(\psi)$ | rotation by $\psi$ about $\mathbf e_z$; $\psi$ = heading offset that turns the jaw IMU's world into the head IMU's world |
| $\mathbf A = \mathbf R_{\mathrm{max}\leftarrow\mathrm{cra}}$ | head-IMU mounting composed with the maxillary landmark frame |
| $\mathbf X = \mathbf R_{\mathrm{mta}\leftarrow\mathrm{mp}}$ | jaw-IMU mounting on the mandibular plate (MP) |
| $\mathbf M(t) = \mathbf R_{\mathrm{cra}}^\top \mathbf R_z(\psi)\, \mathbf R_{\mathrm{mta}}$ | pure-IMU relative orientation (jaw chip in head chip) |
| $\mathbf R_{\mathrm{opt}}(t)$ | optical jaw orientation $\mathbf R_{\mathrm{max}\leftarrow\mathrm{mp}}$ (kinematics `<phase>__T`, 40 Hz, 6 Hz zero-phase low-pass) |
| $\mathbf T_{\mathrm{closed}}$ | kinematics closed reference pose (an open_close cycle start; an02 §4) |
| $\mathbf T_{\mathrm{ref}}$ | camera-free closed reference: optical mean pose over the rest phase (§7.1) |
| $\operatorname{Log}:SO(3)\to\mathbb R^3$, $\operatorname{Exp}$ | rotation vector of a rotation and its inverse (`scipy` `as_rotvec` / `from_rotvec`) |
| $[\mathbf v]_y$ | $y$ component (maxillary frame) |
| $\theta(\mathbf R) = [\operatorname{Log}\mathbf R]_y$ | opening angle (deg) of a rotation that is already relative to the closed pose: $\theta_y$ of an02 §3 (a rotation-vector component, not an Euler angle) |
| $\mathcal P(\mathbf S) = \mathbf U\operatorname{diag}(1,1,\det \mathbf U\mathbf V^\top)\mathbf V^\top$ | nearest rotation to $\mathbf S = \mathbf U\boldsymbol\Sigma\mathbf V^\top$ (Frobenius) |
| $\operatorname{cm}\{\mathbf R_i\} = \mathcal P(\sum_i \mathbf R_i)$ | chordal mean of rotations (`camerafree._chordal_mean`) |
| $\mathcal R$, $\mathcal C$ | grid samples of the rest phase; samples of detected closed runs |
| $\mathbf p_0$, $\mathbf c_0$ | lower incisor and condyle midpoint at the closed pose (maxillary frame, mm) |
| $\boldsymbol\tau(t)$ | condyle-midpoint translation since the closed pose (maxillary frame, mm) |
| $\mathbf r_{\mathrm{cra}}, \mathbf r_{\mathrm{mta}}$ | lever arms: IMU site in its plate's optical body frame (mm) |
| $\mathbf s_{\mathrm{cra}}, \mathbf s_{\mathrm{mta},0}$ | IMU sites in the maxillary frame (head; jaw at the closed pose) |
| $\Delta t$ | grid step (5 ms at 200 Hz) |
| $\operatorname{RMS}(x) = \sqrt{\tfrac1n\sum x_k^2}$ | over the finite samples |

## 1. What was recorded

Per BNO085 (mandible `mta`, head `cra`):
- **Game-rotation-vector quaternions:** about 390 Hz, world←body, $(w,x,y,z)$. No
  magnetometer, so the two IMUs have **independent heading references**; their worlds differ
  by a rotation $\mathbf R_z(\psi)$ about the common vertical.
- **Raw accelerometer including gravity:** about 255 Hz.
- **No gyroscope stream.** Angular rates come from differences of the quaternions.

**Known property: dynamic calibration was on in every recording.** The firmware
enables the BNO085 motion-engine dynamic calibration (accelerometer, gyroscope, magnetometer)
on every configuration apply (`firmware/jtsmm/src/domain/imu_config.c:436-439`), so the gyroscope
zero-rate offset and the accelerometer calibration may be updated while a sensor detects
stillness; quaternion accuracy stays 3, so an update is silent in the data. The possible
effect on the game rotation vector is a step, or a change of the heading drift rate, after
still periods. **Audit** (`scripts/audit_recompute/imu_dyncal.py`, read-only on the stored results
and the session files): still
periods = both IMUs below 2°/s for ≥ 2 s (13 dataset takes: 6–21 per take, 24–119 s; holds,
rest, idle gaps); error traces with the motion removed = the E2a opening error against the
optics at the IMU-detected closed samples (dataset, §7.4) and the IMU-alone rotation error against
Qualisys (validation, movement tasks only).

**Stored audit** (change-log entry J4.6; stages `imu_dyncal` / `imu_dyncal_cohort`,
`imu/stage_dyncal.py:70-88`): the dataset part of the audit runs in the pipeline, with the same
helpers as the validation part (`imu/dyncal.py:dyncal_trace`). Error trace =
`open_deg__E2a` − `open_deg__opt` (the optical angle after the 6 Hz pose low-pass) of the
`camerafree` arrays, set to NaN outside the closed runs of the primary E3 (`closed` array,
E3b; `stage_dyncal.py:40-46`, `stage_camerafree.py:731`). Parameters are hard-coded in
`dyncal.DEFAULTS` (`dyncal.py:17-28`; not configurable from study.yaml):
- *still* = both IMUs turn slower than 2°/s for ≥ 2 s (angular speed from the geodesic
  quaternion increments on a 100 Hz grid, 0.2-s centred running mean);
- *steps*: block medians of the error over 0.25-s blocks; changes between consecutive blocks at
  most 1 s apart; robust SD = 1.4826 MAD of the changes; a step is a change above
  max(0.25°, 5 robust SD); *aligned* = in a still period or up to 2 s after it, against the
  share of changes expected there without alignment; drift = OLS slopes (deg/min) inside the
  still periods (median of the per-period slopes), outside them and over the whole trace;
- *transition jump* per pair of consecutive phases with ≥ 5 error samples: median of the first
  1 s of the next phase minus the median of the last 1 s of the previous one; expected jump =
  the whole-trace OLS drift × gap; residual = jump − expected (`dyncal.py:phase_jumps`);
- *test* (`imu_dyncal_cohort`): |residual| of the transitions with ≥ `STILL_MIN_S` = 2 s of
  stillness in the gap against the others, two-sided Mann–Whitney (descriptive).

Tables: `imu_dyncal:dyncal_summary` (per take; a note when there is no camera-free result),
`dyncal_periods` (per still period), `dyncal_transitions`; `imu_dyncal_cohort:
dyncal_transition_test` (one row: medians, counts, `p_mannwhitney`, `n_participants`) and the
stacked `dyncal_summary`. The paper macros `dyncalJump*` / `dyncalStillN` / `dyncalOtherN` /
`dyncalStillMin` read `imu_dyncal_cohort:dyncal_transition_test`
(`experiments/icorr2027/numbers_dataset.yaml`). Audit only: no stage reads these tables.
`scripts/audit_recompute/imu_dyncal.py` stays the independent recomputation.
- **No step aligned with stillness:** no block-to-block change of the dataset traces exceeded
  the step threshold ($\max(0.25°, 5\,\mathrm{robust\ SD})$ = 0.7–4.2°; largest change per take 0.37–2.4°);
  the validation traces had 3 steps (s14), none within 2 s of a still period (the error trace
  covers still periods for only about 2 % of its blocks, so the validation check is weak).
- **No transition effect:** the error jump across task boundaries, minus the take's drift
  over the gap, is |0.48|° (median, 26 boundaries with ≥ 2 s of stillness in the gap) against
  |0.34|° (27 without; Mann–Whitney p = 0.20), at a window SD of 0.34°.
- **Drift rate:** the opening drift of E2a at the closed samples is small over the take
  (median slope 0.11°/min outside still periods); the within-still slopes (median −0.81°/min,
  4 of 13 takes positive, Wilcoxon p = 0.11 against outside) and the rest-phase change
  (|0.24|° over a median 5.5 s) are noise-dominated short windows, not a resolved rate change.
  The relative heading $\psi$, re-fitted at the closed-pose anchors (rest and idle gaps), drifts by
  −9.4…+5.6°/min (median −2.6°/min, residual SD 2.1°); the audit cannot tell how much of it is
  gyroscope bias the dynamic calibration did or did not correct (every take had it on).
- **Relative IMU orientation inside still periods** (pure IMU, $\mathbf M(t)$; includes
  micro-motion of the jaw and head), median change per period by kind: rest 0.35° (2.7°/min,
  21 periods), idle gaps 0.39° (7.2°/min, 87), landmark holds 0.48° (7.4°/min, 79); per
  participant the median is 0.27–0.66° (4.4–9.7°/min).

**What it affects:** analyses fitted per take absorb it (C3a / E1: the in-take calibration;
the E3 rungs reset at every closed pose); the camera-free drift (E2a / E2b) and the
IMU-alone results of the validation study (VD10 drift, VD20) carry whatever the calibration updates added, together
with the ordinary gyroscope-bias drift; the audit bounds a stillness-aligned effect below
about 0.3–0.5° per step at the closed pose. **Setting since 2026-11-02 (decision of the authors):** the
dynamic calibration is a per-sensor config switch, `mcu.imu_<slot>.dynamic_calibration: {accel,
gyro, mag}`, **default on** (the behaviour of every recording so far); each take stores it in
`/meta/mcu_config_json` (no key = on, all takes before 2026-11-02). No biomech code reads
`mcu.imu_*.dynamic_calibration` (only the jtsmm HDF5 writer and integrity checker do), so the
per-take mode is not in an analysis table. Additions to this dataset
keep it on, so the dataset stays homogeneous; new studies calibrate and save (Save DCD) before
the session and switch all three off for the takes, so the bias model is fixed within a take
(procedure [cv02 §7.2](../calibration_validation/cv02_imu_calibration.md)). The analysis needs no
IMU calibration step in either mode. The existing data stay as recorded.

*Paper wording:* "The IMUs ran the manufacturer's dynamic sensor calibration throughout the
recordings. An audit of the IMU-only orientation errors at closed-jaw instants found no
error step aligned with still periods (largest change per recording 0.4–2.4°, below the
noise-derived step threshold)."

## 2. Preprocessing (`imu/preprocess.py`)

**Accuracy and timestamps.** Samples with accuracy 0 are dropped first (quaternions and
acceleration separately). Timestamps are sorted (stable)
and de-duplicated, first occurrence kept (`jtsmm.imu.batch.sanitize_stream`, 1–1158
non-monotonic stamps per stream), so that every stream is strictly increasing.
`calibration.dropped_*` counts both: the accuracy-0 samples plus the removed duplicate stamps
(raw count minus kept samples, `imu/preprocess.py:80-86`).

**Resampling** onto a `grid_hz` = 200 Hz grid per phase, $t_k = t_{\mathrm{start}} + k\cdot 5000\,\mu$s,
end exclusive:
- quaternions are normalized and made sign-continuous ($\mathbf q_i \to -\mathbf q_i$ when
  $\mathbf q_{i-1}\cdot\mathbf q_i < 0$; `quat_prepare`), then interpolated by SLERP between the
  bracketing samples $j, j+1$:
  $$
  \mathbf q(t) = \mathbf q_j \otimes \operatorname{Exp}_q\!\bigl(u\,\operatorname{Log}_q(\mathbf q_j^{*}\otimes\mathbf q_{j+1})\bigr),\qquad u = \frac{t - t_j}{t_{j+1}-t_j}
  $$
  (`quat_resample`; $\operatorname{Log}_q$ = rotation vector of a quaternion);
- acceleration linearly per axis (`ImuStream.accel_at`).

Both are NaN where the bracketing source samples are more than `stages.qc.imu.gap_s` = 0.05 s
apart and outside the stream's span (`GAP_S` / `GAP_US`, read from the package default of
the same parameter that sets the QC coverage; E33.7, formerly a hard-coded 50 000 µs).
Rotation matrices from quaternions: `jtsmm.imu.batch.quat_to_matrix` (Hamilton convention).

*Paper wording:* "IMU samples flagged as uncalibrated were discarded, timestamps were made
strictly increasing, and both streams were resampled to 200 Hz (quaternions by spherical
linear interpolation, accelerations linearly), leaving gaps longer than 50 ms unfilled."

## 3. One-time in-take calibration (`imu/calibration.py`; the C3a model, camera-free rung E1)

**What and why.** The two game rotation vectors have independent headings, and the IMU
mountings were not known when the dataset was processed. The jaw orientation in the maxillary
frame is therefore modelled from the two IMUs with three unknown rotations, fitted per
participant against the optics on one task and tested on all others:
$$
\hat{\mathbf R}_{\mathrm{max}\leftarrow\mathrm{mp}}(t) = \mathbf A\,\mathbf R_{\mathrm{cra}}(t)^\top\,\mathbf R_z(\psi)\,\mathbf R_{\mathrm{mta}}(t)\,\mathbf X = \mathbf A\,\mathbf M(t)\,\mathbf X .
$$
Parameters $\mathbf x = (\boldsymbol a, \psi, \boldsymbol\xi)\in\mathbb R^7$ with
$\mathbf A = \operatorname{Exp}(\boldsymbol a)$, $\mathbf X = \operatorname{Exp}(\boldsymbol\xi)$.

**Data.** On `calibration_phase` = `free_motion` (about 15 s): the optical
$\mathbf R_{\mathrm{opt}}$ (`<phase>__T` of the kinematics stage) and both quaternions
resampled at the optical frame times; frames where all three are finite (≥ 30 needed), evenly
subsampled to ≤ 800 ($\lfloor\mathrm{linspace}\rfloor$ indices).

**Residual and cost.** Per sample $i$ the rotation vector of the error rotation (rad),
$$
\mathbf r_i(\mathbf x) = \operatorname{Log}\bigl(\hat{\mathbf R}_i(\mathbf x)^\top\mathbf R_{\mathrm{opt},i}\bigr)\in\mathbb R^3,\qquad
\min_{\mathbf x}\ \tfrac12\sum_i\lVert\mathbf r_i(\mathbf x)\rVert^2 ,
$$
reported as `rms_deg` $=\frac{180}{\pi}\sqrt{\frac1n\sum_i\lVert\mathbf r_i\rVert^2}$.

**Starts.**
- $\mathbf X$ from Park–Martin $\mathbf A_{ij}\mathbf X' = \mathbf X'\mathbf B_{ij}$
  (`jtsmm.imu.mounting.park_martin_axxb`) on random sample pairs (seed 0) with optical
  relative rotation $\mathbf A_{ij} = \mathbf R_{\mathrm{opt},i}^\top\mathbf R_{\mathrm{opt},j}$ of
  at least 3° and IMU relative rotation $\mathbf B_{ij} = \mathbf R_{\mathrm{mta},i}^\top\mathbf R_{\mathrm{mta},j}$
  (≤ 400 pairs, ≥ 20 needed): $\mathbf X' = \mathcal P\bigl(\sum \operatorname{Log}\mathbf A_{ij}\,\operatorname{Log}\mathbf B_{ij}^\top\bigr)$
  ($\mathbf R_{\mathrm{mp}\leftarrow\mathrm{mta}}$; refused when the IMU rotation axes are
  nearly collinear), $\mathbf X = \mathbf X'^\top$. Plus 6 random rotations for $\mathbf X$.
- Each $\mathbf X$ start × $\psi\in\{0, 45, \dots, 315\}°$, with $\mathbf A$ initialized as the
  chordal mean (`Rotation.mean`) of $\mathbf R_{\mathrm{opt},i}\,(\mathbf M_i(\psi)\mathbf X)^\top$.

**Refinement.** The 4 starts with the lowest mean squared residual are refined by
Levenberg–Marquardt (`scipy.optimize.least_squares(method="lm")`); the lowest final cost wins
(`init` = `park_martin` or `random`). $\mathbf A$ and $\psi$ are separable only when the head
rotates; the prediction needs only their product.

**Drift** (`calibration.drift_quat_deg_per_min`): OLS slope $\times 60$ of
$\theta_{\mathrm{quat}}(t)-\theta_{\mathrm{opt}}(t)$ (§4) against time (s), over all finite
grid samples of all experiment phases pooled, the calibration phase included
(`numpy.polyfit`, degree 1; > 10 samples).

**Stored:** table `imu:calibration` (`rms_deg`, `n`, `init`, `psi_deg`, `X_rv{x,y,z}_deg`,
`A_rv{x,y,z}_deg`, `dropped_*`, `drift_quat_deg_per_min`). Code: `calibration.py:fit`,
`_residual`, `_park_martin_X`; `analysis.py:analyse`.

*Paper wording:* "Per participant, the jaw orientation was predicted from the two IMU
orientations with three unknown constant rotations (head-IMU mounting with the anatomical
frame, jaw-IMU mounting, and the heading offset between the IMUs), fitted by nonlinear least
squares on the geodesic orientation error against the optical jaw orientation during a 15-s
free-movement task and evaluated on all other tasks."

## 4. Opening-angle estimators (about $+y$ of the maxillary frame, vs the optical closed pose)

All three are relative to $\mathbf T_{\mathrm{closed}}$ (an02 §4; $\mathbf R_{\mathrm{closed}}$
its rotation) and use the opening angle $\theta$ of an02 §3. Code: `analysis.py:phase_signals`.

| Estimator | Definition | Character |
|---|---|---|
| `quat` | $\theta_{\mathrm{quat}}(t) = \bigl[\operatorname{Log}(\hat{\mathbf R}(t)\,\mathbf R_{\mathrm{closed}}^\top)\bigr]_y$, $\hat{\mathbf R}$ = calibrated prediction (§3) | smooth; drift 0.1–1.0 °/min in absolute value (`r17a`: −1.00…+0.61) |
| `tilt` | gravity seen by both IMUs, $\mathbf g_h = \mathbf A\,\hat{\mathbf f}_{\mathrm{cra}}$, $\mathbf g_m = \mathbf R_{\mathrm{closed}}\mathbf X^\top\hat{\mathbf f}_{\mathrm{mta}}$ ($\hat{\mathbf f} = \mathbf f/\lVert\mathbf f\rVert$); $\theta_{\mathrm{tilt}} = -\angle_y(\mathbf g_h\to\mathbf g_m)$ | heading-free, drift-free; physically meaningful only while both IMUs are quasi-static |
| `comp` | complementary fusion: increments of `quat`, level of `tilt` ($\tau$ = `tau_s` = 1 s) | causal |

- **Signed angle about $y$** (`jtsmm.imu.tilt.signed_angle_about`): both vectors projected onto
  the plane normal to $\mathbf e_y$, $\mathbf u_\perp = \mathbf u - (\mathbf u\cdot\mathbf e_y)\mathbf e_y$;
  $\angle_y(\mathbf a\to\mathbf b) = \operatorname{atan2}\bigl((\mathbf a_\perp\times\mathbf b_\perp)\cdot\mathbf e_y,\ \mathbf a_\perp\cdot\mathbf b_\perp\bigr)$.
  A body rotation by $+\theta$ turns the gravity it sees by $-\theta$, hence the sign.
- **Quasi-static** (`jtsmm.imu.tilt.quasi_static`, stored per sample as `quasi_static`):
  $\bigl\lvert\lVert\mathbf f\rVert - g\bigr\rvert \le$ `quasi_static_tol`$\cdot g$ (0.05) for **both** IMUs.
- **`comp`** (`jtsmm.imu.tilt.complementary`): starts at $\theta_{\mathrm{tilt}}[k_0]$, $k_0$ the
  first sample where $\theta_{\mathrm{quat}}$ is finite and the tilt is valid; then
  $$
  \theta[k] = \begin{cases}\alpha\,(\theta[k-1] + \Delta\theta_{\mathrm{quat}}[k]) + (1-\alpha)\,\theta_{\mathrm{tilt}}[k], & \text{quasi-static},\\ \theta[k-1] + \Delta\theta_{\mathrm{quat}}[k], & \text{otherwise},\end{cases}\qquad \alpha = \frac{\tau}{\tau + \Delta t}
  $$
  with $\Delta\theta_{\mathrm{quat}}[k] = 0$ when either neighbour is NaN.
- **Optical reference $\theta_{\mathrm{opt}}$:** the kinematics `<phase>__theta_deg` ($\theta$ vs
  $\mathbf T_{\mathrm{closed}}$), linearly interpolated onto the grid; NaN where the bracketing
  optical frames are more than 2.5 × the median optical frame interval apart
  (`_optical_gap_us`, E23; 62.5 ms at 40 Hz, formerly a fixed 60 ms).

**Agreement (`imu:agreement`, per experiment phase; `is_calibration` marks `free_motion`):**
- `rmse_<est>_deg` $=\operatorname{RMS}(\theta_{\mathrm{est}}-\theta_{\mathrm{opt}})$ and
  `bias_<est>_deg` = mean of the same difference over the samples where both are finite
  (`agreement`). For `tilt` only the quasi-static samples count (E23).
- `quasi_static_frac`; `lag_ms`: lag $L^\ast$ of $\theta_{\mathrm{quat}}$ behind
  $\theta_{\mathrm{opt}}$ (> 0 = IMU late) at the maximum of the uncentred normalized
  cross-correlation of their derivatives $a = \dot\theta_{\mathrm{opt}}$, $b = \dot\theta_{\mathrm{quat}}$,
  $$
  c(L) = \frac{\sum_i a_i\,b_{i+L}}{\sqrt{\sum_i a_i^2\,\sum_i b_{i+L}^2}},\qquad \lvert L\rvert\,\Delta t \le 100\ \mathrm{ms},
  $$
  each sum over the pairs where both are finite (≥ 50 pairs per lag and ≥ 50 common samples;
  else NaN). The derivatives are taken per gap-free segment (`timebase.gradient`, NaN stays
  NaN); `_lag_ms`, `timing.coupling.lagged_corr` / `peak_lag` (E23; formerly gaps were
  zero-filled before the gradient).
- The cohort (`stats_imu.imu_agreement_cohort`, `cohort_misc.imu_tables`) excludes the
  calibration phase: per task and estimator the median RMSE over participants with its 95 %
  BCa CI (`stats.core.describe`, an07); `rms_deg` and `drift_quat_deg_per_min` likewise;
  `lag_ms` pooled over all test phases.

**Cycle values (`imu:cycles`):** per `ok` kinematics cycle of the phase with ≥ 10 grid samples
in $[t_{\mathrm{start}}, t_{\mathrm{end}}]$:
- $\theta_{\mathrm{peak}} = \max_t\bigl(x(t) - x(t_{\mathrm{start}})\bigr)$ for $\theta_{\mathrm{opt}}$ and each
  estimator (`theta_peak_<est>_deg`); for `tilt`, $x - x(t_{\mathrm{start}})$ is masked to the
  quasi-static samples (the start value itself is taken unmasked);
- `rmse_cycle_<est>_deg` $=\operatorname{RMS}\bigl[(x - x_{\mathrm{start}}) - (\theta_{\mathrm{opt}} - \theta_{\mathrm{opt,start}})\bigr]$;
  `opening_opt_mm` = the kinematics cycle's `opening_mm`.

*Paper wording:* "The IMU opening angle was the sagittal (maxillary $y$) component of the
rotation vector of the calibrated jaw-in-head rotation relative to the closed pose. Its
agreement with the optical angle was summarized per task as the root-mean-square difference
over time, and its delay as the lag maximizing the cross-correlation of the two angular
velocities within ±100 ms."

## 4a. IMU opening model and the C3a outcome

**What and why.** The wearable outcome is incisal opening in mm, not an angle. Each cycle's
IMU peak angle is converted to opening with the rotation-to-opening coupling of the optics
and compared with the optical opening of the same cycle.

**Coupling model** (kinematics `opening_model`, an02 §6; `kinematics/analysis.py:_opening_fit`):
OLS without intercept over all finite frames of the participant's `ok` open_close opening
halves, $\theta$ and opening ($-\Delta z$ of the incisor) relative to each cycle start,
$$
\widehat{O}(\theta) = a_1\,\theta + a_2\,\theta^2,\qquad (a_1,a_2) = \arg\min\sum_k\bigl(O_k - a_1\theta_k - a_2\theta_k^2\bigr)^2 .
$$
Applied to $\theta_{\mathrm{peak}}$ of each IMU cycle (§4):
- **Own model, out of sample** (`opening_subject_<est>_mm`, D26; `analysis.py:own_model_openings`):
  an open_close cycle whose frames entered the fit is predicted by the leave-one-cycle-out
  coefficients $(a_1^{(-c)}, a_2^{(-c)})$ (`kinematics:opening_model_loco`,
  `kinematics/analysis.py:opening_model_loco`; NaN when the other cycles hold < 10 frames);
  every other cycle (chewing; open_close cycles not in the fit) never entered the fit and gets
  the full model. All three estimators.
- **Own model, in sample** (`opening_subject_insample_<est>_mm`): the full model for every
  cycle (the value reported before D26; kept as a sensitivity).
- **Generic model, leave one subject out** (`opening_loso_<est>_mm`, `stats_imu`, `quat` and
  `comp` only; table `stats_imu:imu_opening_cycles`): for participant $i$,
  $\bar a_m^{(-i)} = \frac{1}{N-1}\sum_{j\ne i} a_{m,j}$ over every other participant **with
  optics** (E32; `loso_n_others`), $\widehat O = \bar a_1^{(-i)}\theta + \bar a_2^{(-i)}\theta^2$.
  This is the pre-specified C3a outcome; it is reported first.

**Agreement** (`stats_imu:imu_opening_ba`, `cohort_misc.imu_tables`): per task (open_close,
chewing), estimator (`quat`, `comp`) and model (`loso`, `subject`, `subject_insample`), with
$d = \widehat O - O_{\mathrm{opt}}$ per cycle:
- repeated-measures Bland–Altman (`stats.core.bland_altman_rm`; one-way ANOVA with
  participants as groups, an07): bias $\mu$, $s^2 = s_b^2 + s_w^2$, limits
  $\mu \pm 1.96\,s$, with participant-cluster bootstrap CIs (`bland_altman_rm_ci`);
- the user-facing error, the larger absolute limit $\max(\lvert\mu - 1.96s\rvert, \lvert\mu + 1.96s\rvert) = \lvert\mu\rvert + 1.96s$
  (`max_abs_loa`), with the MOVER 95 % CI (`max_abs_loa_ci_lo/hi`, reported;
  `cohort_misc.max_abs_loa_mover`, an07) and, for comparison, the cluster-bootstrap percentile
  CI (`max_abs_loa_boot_ci_*`, under-covers with 13 clusters);
- `subject_bias_sd` (SD of the per-participant mean differences),
  `subject_abs_bias_median`.

The requirement rows (`requirements`, `stats/requirements.py:_imu_loa_rows`) take open_close,
`quat`: `imu_opening_loa_loso_mm` (generic, first) and `imu_opening_loa_subject_mm` (own,
leave-one-cycle-out; in-sample value in the note); `imu_angle_rmse_deg` = the open_close
`quat` median RMSE with BCa CI. Values (paper store `r17a`, 13 adults, all with IMU): generic
3.83 mm [MOVER 2.95, 6.26], own 3.05 mm [2.45, 4.28], angle RMSE 0.76° [0.53, 1.23]. In the
analysis of record (`r17`) the generic model is also trained on the optics of the participant
recorded outside the protocol: 3.92 mm [3.03, 6.34]; the other two values are equal (an07; D26,
change log).

*Paper wording:* "Each cycle's IMU peak opening angle was converted to incisal opening with a
quadratic angle-to-opening model without intercept, either the participant's own model fitted
without the predicted cycle, or a generic model averaging the coefficients of all other
participants. Agreement with the optical opening was assessed by repeated-measures
Bland–Altman analysis; we report the larger absolute 95 % limit of agreement with its MOVER
confidence interval."

## 5. Signals for timing and decoding

- **ω** (`<phase>__omega_dps`): `timebase.gradient` of $\theta_{\mathrm{quat}}$: `numpy.gradient` per finite run
  (central differences inside, one-sided at the ends), then a centred moving mean of
  `odd_window`(`omega_smooth_s`, `grid_hz`) = 5 samples (25 ms; `uniform_filter1d`,
  `mode="nearest"`; a run shorter than that uses its largest odd length). The window is odd so
  the smoother is symmetric (E17; formerly 4 samples, which shifted ω by half a sample, 2.5 ms).
  Linear resampling, central differences and the centred mean are all zero-phase, as required
  for the lead (an02 §1 filter-phase policy); only `comp` is causal and it does not enter the
  lead.
- **Opening angle $\theta_{\mathrm{quat}}$** (`<phase>__theta_quat`, deg; §4): the calibrated
  jaw-in-head rotation from the two game rotation vectors; head rotation cancels. ω and α are
  its derivatives.
- **Acceleration magnitude** (`<phase>__acc_rel_ms2`): gravity is removed per IMU in its own world,
  $\mathbf a^{\mathrm{lin}} = \mathbf R\,\mathbf f - g\,\mathbf e_z$, with $\mathbf R$ from that IMU's own game rotation vector; then
  $\lVert\mathbf R_z(\psi)\,\mathbf a^{\mathrm{lin}}_{\mathrm{mta}} - \mathbf a^{\mathrm{lin}}_{\mathrm{cra}}\rVert$ (the mandible's value rotated into the head IMU's world). It is
  unsigned and unfiltered, with no lever-arm or rotation (centripetal, tangential) correction;
  a tilt error $\delta$ of either IMU leaks $g\sin\delta$ (≈ 0.17 m/s² per degree) into it. This is why it can fire early as a
  mechanical onset (an05 §2, E34).
- **Signed angular acceleration α** (`timing/lead.py`, `alpha_signal`; computed in the
  `timing` stage, not stored by `imu`): $\theta_{\mathrm{quat}}$ through a 4th-order Butterworth low-pass at
  `stages.timing.c2e.alpha_cutoff_hz` = 15 Hz (10 / 20 Hz as sensitivity), zero-phase
  (`sosfiltfilt`) per finite segment (segments not longer than the filter padding,
  3·(2·n_sections + 1) samples, stay NaN), then the gap-aware second derivative
  (`timebase.gradient` twice), deg/s², > 0 = accelerating the opening. Its rest noise is 3·SD
  over the `rest` phase. Used only by the post hoc family C2-E (an05 §3.3). It never enters
  the pre-specified t_mech.
- **Decoding inputs (an05 §5):** calibration-free. Body-frame angular velocity by backward
  differences, and gravity-free acceleration of both IMUs.

## 6. Movement bandwidth (X1, exploratory)

`imu:bandwidth` (`analysis.py:movement_bandwidth`, `spectral.py`): per experiment phase (all of them,
including `rest` and `free_motion`), the frequencies below which 50 / 95 / 99 % of the power of
ω (§5) lie (`f50_hz`, `f95_hz`, `f99_hz`).
- A phase needs ≥ 4 s of finite samples (≥ 4·`grid_hz`, not necessarily contiguous).
- **PSD** (`welch_masked`): 4-s segments ($L$ = 800 samples, $\Delta f$ = 0.25 Hz), 50 % overlap, Hann
  window $w$, mean removed per segment, one-sided density
  $P(f) = \frac{2\,\lvert\mathcal F\{(x - \bar x)\,w\}(f)\rvert^2}{f_s\sum w^2}$; segments
  containing any NaN are skipped; the PSD is the mean over the remaining segments (`n_windows`).
- **Quantiles** (`power_quantiles`): over bins $f \ge 0.05$ Hz (drops the DC bin),
  $f_q = \min\{f_b : \sum_{b'\le b}P_{b'} \ge q\sum_{b'}P_{b'}\}$.
- **Noise check (D12):** the IMU spectrum carries sensor-specific content that appears only during
  motion (10–30 Hz; tooth-contact transients, sensor fusion). Subtracting the rest-phase noise
  spectrum does not remove it (pilot: f95 unchanged), and repeated quaternion samples and timestamp
  jitter were ruled out as its cause. An independent sensor does: the unsmoothed optical opening
  angle (`kinematics` array `<phase>__theta_raw_deg`, 40 Hz, before the 6 Hz low-pass) and the IMU ω
  are compared on a common 40 Hz grid over the overlap of both streams. Optical: `numpy.gradient`
  of the linearly interpolated angle (needs > 8 s of finite frames). IMU: ω over its finite
  samples, low-passed first (zero-phase 4th-order Butterworth at 16 Hz, `_antialias`, E29: no
  aliasing of content above 20 Hz), then linearly interpolated; ≥ 320 grid samples.
  Magnitude-squared coherence $C(f) = \lvert P_{oi}\rvert^2/(P_{oo}P_{ii})$ and the PSDs
  $P_{\mathrm{imu}}$, $P_{\mathrm{opt}}$ (scipy `coherence` / `welch`, 160-sample = 4-s Hann segments,
  50 % overlap). $C$ is bias-corrected for $K = 2\lfloor N/160\rfloor - 1$ averaged segments
  (unrelated signals give $E[C]\approx 1/K$):
  $$
  C'(f) = \operatorname{clip}\Bigl(\frac{C(f) - 1/K}{1 - 1/K},\,0,\,1\Bigr).
  $$
  Only the shared power counts as movement: $P_{\mathrm{coh}} = C'\,P_{\mathrm{imu}}$ over
  $0 < f < 20$ Hz → `f95_coh_hz`, `f99_coh_hz` (`power_quantiles`, $f\ge 0.05$ Hz). `coh_band_hz` =
  the highest frequency of the contiguous run of bins, from the lowest bin up, with $C' \ge 0.5$
  (`COHERENCE_MIN`; 0 if the lowest bin fails); `noise_limited` when it is below 2 Hz
  (`MIN_COHERENT_BAND_HZ`: the opening angle does not carry the task's motion, e.g. protrusion) or
  when no coherence could be computed. D27 sensitivity: `f95_opt_hz`, the f95 of
  $P_{\mathrm{opt}}$ on the same grid and band (the optics alone). Pilot: IMU–optical coherence
  0.8–0.99 below 3 Hz, 0.3–0.9 at 3–6 Hz and 0.1–0.4 above 6 Hz.
- **Requirement** (`requirements` stage, `stats/requirements.py:bandwidth_per_participant`,
  `_cap`; D12, E31): tasks whose motion is the opening rotation (`open_close`, `chewing`,
  `free_motion`; `BANDWIDTH_TASKS`) and that are not noise-limited (`coh_band_hz` ≥ 2.0 Hz, NaN
  counts as noise-limited); per participant the largest `f95_coh_hz`; the design value is the
  Harrell–Davis P95 across participants with the exact one-sided lower 95 % order-statistic
  bound (an07 §13). Value (`r17a`, n = 13): 8.43 Hz, lower bound 5.25 Hz.
  The raw `f95_hz`/`f99_hz` and `f95_opt_hz` stay in the table and in
  `requirements_sensitivity` (variants `imu_only`, `optics_only`, `min_band_1.5hz`,
  `min_band_3hz`).
- **Arrays** (stage `imu`, per task, the curves behind the numbers): `bandwidth__<task>__freq` /
  `__psd` (Welch PSD of ω, ≤ 50 Hz) and, where the coherence was computed,
  `bandwidth__<task>__coh_freq` / `__coh` ($C'$) / `__psd_imu40` ($P_{\mathrm{imu}}$ on the 40 Hz grid),
  0 < f < 20 Hz.
- The IMU is used because the optical poses are low-passed at 6 Hz.

*Paper wording:* "The movement bandwidth was the frequency below which 95 % of the jaw
angular-velocity power lay. To exclude IMU content not seen by the cameras, the IMU power
spectrum was weighted by the bias-corrected magnitude-squared coherence with the unfiltered
optical angular velocity; tasks with a coherent band below 2 Hz were treated as noise-limited.
The requirement is the 95th percentile across participants of each participant's largest
value over the opening tasks."

## 7. Camera-free jaw pose (X5, exploratory; D28 path 2)

**Question.** What do the cameras contribute once the IMU mountings and the anatomical frame
are known? An estimator ladder, from the optics (reference, E0) through IMU rotation with
decreasing optical input to IMU-only translation, each scored against the optics. Optics
enter only as **static calibration**: the hand-eye IMU mountings (one session), the landmark
frame of the holds ($\mathbf T_{\mathrm{max}\leftarrow\mathrm{hp}}$, incisor, condyles), the
optical closed pose of the rest phase (the reference all rungs and the ground truth refer to),
and, for E2a only, the heading against the optical closed pose; the translation models (E4own,
E4gen) are trained on optical free-motion data. During the scored tasks only the IMUs are
used. All values below were fixed before any real-data run and checked only on synthetic data
(`biomech.testing.synth_camerafree`, `software/biomech/tests/test_x5_camerafree_synth.py`).

| Rung (code) | Reader name | Rotation from | Translation from |
|---|---|---|---|
| E0 | optics (reference) | optics | optics |
| E1 | in-take fit | §3 model, fitted on the take's free_motion against the optics | — |
| E2a | hand-eye mountings, optical heading | hand-eye $\mathbf A$; $\psi$ from the rest phase vs the optical closed pose | — |
| E2b | hand-eye mountings, IMU heading | hand-eye $\mathbf A$; $\psi$ from IMU-detected closed poses (no optics) | — |
| E3a / E3b (`_slerp`) | closed-pose resets | E2a / E2b reset at every IMU-detected closed pose | — |
| E4hinge | condyle hinge | primary E3 | none (rotation about $\mathbf c_0$) |
| E4own / E4gen | own / generic model | primary E3 | ridge model rotation → $\boldsymbol\tau$ (own optics / other participants) |
| E5 (`_causal`) | accelerometer bridging | primary E3 | double-integrated accelerometers between closed-pose anchors |
| E6 (`_causal`) | accelerometers + own model | primary E3 | E5 + E4own as pseudo-measurements |
| E7 | movement-class model | — | not computed (§7.12) |
| X5b, X5c | oracle analyses | §7.13 | §7.13 |

Primary E3 = `primary_e3` = E3b (camera-free $\psi$; falls back to E2a's $\psi$ when not
identifiable). Hold (causal) is primary; `_slerp` is offline.

### 7.1 Grid, reference and inputs (`stage_camerafree.py:run_camerafree`)

- **Grid:** one 200 Hz grid (`grid_hz`) from the first to the last experiment phase of the
  kinematics, inter-phase gaps included (`phase` = −1 there): the IMU estimators run over the
  whole take; scoring is per phase.
- **Rest mask $\mathcal R$:** `rest_mask` = `phase` in the study (the protocol rest phase;
  `labels` = rest ∩ optical intent label REST or untracked, code default).
- **Optical closed pose** $\mathbf T_{\mathrm{ref}}$ (`closed_pose`): chordal mean of the rest
  phase's jaw rotations and mean of its translations (≥ 5 valid frames; else
  $\mathbf T_{\mathrm{closed}}$). Its angle to $\mathbf T_{\mathrm{closed}}$ is
  `cf_summary.closed_vs_tclosed_deg`. Every rung and the ground truth refer to
  $\mathbf T_{\mathrm{ref}}$ (a reference at the open_close cycle start would add a constant offset
  that is no IMU error).
- **Optics on the grid** (`optical_grid`): per phase, never across,
  $\boldsymbol\rho_{\mathrm{opt}}(t) = \operatorname{Log}(\mathbf R_{\mathrm{opt}}(t)\,\mathbf R_{\mathrm{ref}}^\top)$ (deg; `se3.rotvec_rel`),
  the lower incisor $\mathbf p_{\mathrm{opt}}$ and condyle midpoint $\mathbf c_{\mathrm{opt}} = \tfrac12(\mathbf c_L + \mathbf c_R)$,
  each linearly interpolated per component; NaN where the bracketing optical frames are more than
  `max_opt_gap_frames` = 2.5 median frame intervals apart. $\mathbf R_{\mathrm{opt}}^{\mathrm{grid}} = \operatorname{Exp}(\boldsymbol\rho_{\mathrm{opt}})$.
- **Landmarks** (`mp_points`, `landmarks0`): each mandibular point in the jaw-plate frame is the
  median over all frames of all phases of $\mathbf R(t)^\top(\mathbf x(t) - \mathbf t(t))$; then
  $\mathbf p_0 = \mathbf T_{\mathrm{ref}}\,\mathbf p^{\mathrm{mp}}_{\mathrm{inc}}$,
  $\mathbf c_0 = \mathbf T_{\mathrm{ref}}\,\tfrac12(\mathbf c^{\mathrm{mp}}_L + \mathbf c^{\mathrm{mp}}_R)$;
  scale $s = \lVert\mathbf p_0 - \mathbf c_0\rVert$ (`scale_mm`).
- **Hand-eye mountings** (`mounting`, D28: the hand-eye session
  `session_imu_mounting_20250314T110000Z_0000000d/session.h5`, valid for every dataset take; read
  by `load_handeye` through `jtsmm.imu.mounting`, accepted proper rotations only):
  $\mathbf R_{\mathrm{mount}}$ = body←chip per slot; then (`camerafree.mounting_prior`)
  $$
  \mathbf A = \mathbf R_{\mathrm{max}\leftarrow\mathrm{hp}}\,\mathbf R_{\mathrm{mount,cra}},\qquad \mathbf X = \mathbf R_{\mathrm{mount,mta}}^\top ,
  $$
  with $\mathbf R_{\mathrm{max}\leftarrow\mathrm{hp}}$ the rotation of the kinematics `T_max_hp` (landmark frame
  of the head plate).
- **Lever arms** (E5/E6 only; `lever` = `session`, `handeye_levers`): estimated from the same
  hand-eye session's tumbling phases with the validation-study method (`imu/lever.py:design`,
  `fit`; an14 §11a, VD15) at the primary cut-off 6 Hz, without bootstrap. Model (chip frame):
  $$
  \mathbf f = \mathbf R_{\mathrm{mount}}^\top\mathbf R_{wb}^\top\bigl(\ddot{\mathbf p}_b + \ddot{\mathbf R}_{wb}\,\mathbf r - \mathbf g_w\bigr) + \mathbf b ,
  $$
  linear in $(\mathbf r, \mathbf g_w, \mathbf b)$, solved by OLS on all frames of the plate's
  tumbling phase; $\ddot{\mathbf p}_b$, $\ddot{\mathbf R}_{wb}$ by central second differences of the
  4th-order zero-phase Butterworth-filtered (6 Hz) optical pose on gap-free runs (≥ 1.5 s,
  0.25 s dropped at each end), the accelerometer filtered at the same cut-off. The IMU sites
  (`camerafree.jaw_geometry`): $\mathbf s_{\mathrm{cra}} = \mathbf T_{\mathrm{max}\leftarrow\mathrm{hp}}\,\mathbf r_{\mathrm{cra}}$,
  $\mathbf s_{\mathrm{mta},0} = \mathbf T_{\mathrm{ref}}\,\mathbf r_{\mathrm{mta}}$ (homogeneous points).

*Paper wording:* "The camera-free estimators used the optics only as static calibration: the
IMU mountings from a separate hand-eye calibration session, the anatomical landmark frame from
the landmark holds, and the optical jaw pose averaged over the rest phase as the closed
reference."

### 7.2 Relative formulation (`camerafree.py`, module docstring, `_rel`)

With the model of §3, the jaw rotation since a closed pose with IMU product $\mathbf M_c$ is
$$
\mathbf R(t) = \hat{\mathbf R}(t)\,\hat{\mathbf R}_c^\top = \mathbf A\,\mathbf M(t)\,\mathbf M_c^\top\,\mathbf A^\top ,
$$
so the jaw mounting $\mathbf X$ cancels. An error $\delta$ of $\mathbf A$ only conjugates the result
(the rotation axis tilts: opening error $\le \theta(1-\cos\delta)$, off-axis components leak
$\approx\theta_{\mathrm{off}}\sin\delta$). An error $\delta$ of $\psi$ cancels while the head keeps
the tilt it had at the reference; head tilt $\varphi$ since then adds a rotation of
$\approx\delta\sin\varphi$ ($\psi$ turns about the head IMU's world vertical, which moves with
the head). This is why E3 re-fits $\psi$ at every anchor and why E2b needs head tilt to
identify $\psi$.

The **IMU closed reference** of E2 (`closed_reference`):
$\mathbf M_c = \operatorname{cm}\{\mathbf M(t;\psi) : t\in\mathcal R\}$. Opening:
$\theta(t) = [\operatorname{Log}\mathbf R(t)]_y$ (`opening_deg`).

### 7.3 In-take fit (E1)

$\mathbf R_{\mathrm{E1}}(t) = \hat{\mathbf R}(t)\,\mathbf R_{\mathrm{ref}}^\top$ with $\hat{\mathbf R}$ the
§3 prediction read back from `imu:calibration` (`_calibration_e1`). The same model as C3a:
`cf_summary.e1_vs_c3a_max_deg` is the largest difference of its opening (vs
$\mathbf T_{\mathrm{closed}}$) to `imu` `<phase>__theta_quat` on the first phase checked (should be ≈ 0);
`e1_calibration_rms_deg` = `rms_deg`.

*Paper wording:* "As an upper reference for IMU-only rotation, the jaw orientation was also
predicted by the in-recording calibration fitted against the optics (in-take fit)."

### 7.4 Hand-eye mountings with optical heading (E2a; `psi_static`)

$\psi_a$ makes the IMU product at rest match the optical closed pose: over the rest samples,
$$
\psi_a = \arg\min_\psi\sum_{t\in\mathcal R}\bigl\lVert\mathbf R_{\mathrm{cra}}^\top\mathbf R_z(\psi)\mathbf R_{\mathrm{mta}} - \mathbf T\bigr\rVert_F^2,\qquad \mathbf T = \mathbf A^\top\mathbf R_{\mathrm{closed}}\mathbf X^\top ,
$$
with $\mathbf R_{\mathrm{closed}}$ the rotation of the kinematics $\mathbf T_{\mathrm{closed}}$ (as fixed
for X5; not $\mathbf T_{\mathrm{ref}}$). **Closed form** (`_heading_fit`): with
$\mathbf B = \sum_t\mathbf R_{\mathrm{mta}}\mathbf T^\top\mathbf R_{\mathrm{cra}}^\top$,
$\lVert\cdot\rVert_F^2 = \mathrm{const} - 2\operatorname{tr}(\mathbf R_z(\psi)\mathbf B)$ and
$\operatorname{tr}(\mathbf R_z(\psi)\mathbf B) = \cos\psi\,(B_{00}+B_{11}) + \sin\psi\,(B_{01}-B_{10}) + B_{22}$, so
$$
\psi = \operatorname{atan2}\bigl(B_{01}-B_{10},\ B_{00}+B_{11}\bigr).
$$
An error of $\mathbf X$ enters $\psi_a$ through its component about the vertical. Then
$\mathbf M_{c,a} = \operatorname{cm}\{\mathbf M(t;\psi_a)\}_{\mathcal R}$ and
$\mathbf R_{\mathrm{E2a}}(t) = \mathbf A\,\mathbf M(t;\psi_a)\,\mathbf M_{c,a}^\top\mathbf A^\top$
(`relative_rotation`). Stored: `cf_summary.psi_a_deg`.

*Paper wording:* "With the hand-eye mountings, the jaw rotation since the closed pose was
computed from the two IMU orientations alone; the heading offset between the IMUs was fitted
once, in closed form, to the optical closed pose of the rest phase (hand-eye mountings with
optical heading)."

### 7.5 Hand-eye mountings with IMU heading (E2b; `psi_consistency`)

**Idea.** At every closed pose the true $\mathbf M$ is the same. With $\psi$ off by $\delta$ the
closed poses scatter by rotations $\delta$ about each pose's head vertical
$\mathbf v_t = \mathbf R_{\mathrm{cra}}(t)^\top\mathbf e_z$, so $\psi$ minimizes their chordal scatter.
**Cost** (`_consistency_parts`, `_consistency_fit`): $\mathbf R_{\mathrm{cra}}^\top\mathbf R_z(\psi)\mathbf R_{\mathrm{cra}}$ is the
rotation by $\psi$ about $\mathbf v_t$ (Rodrigues, $\mathbf K_t = [\mathbf v_t]_\times$), so with
$\mathbf M^0_t = \mathbf R_{\mathrm{cra}}^\top\mathbf R_{\mathrm{mta}}$,
$$
\mathbf S(\psi) = \sum_t \mathbf M_t(\psi) = \mathbf S_0 + \sin\psi\,\mathbf S_1 + (1-\cos\psi)\,\mathbf S_2,\quad
\mathbf S_0 = \sum_t\mathbf M^0_t,\ \mathbf S_1 = \sum_t\mathbf K_t\mathbf M^0_t,\ \mathbf S_2 = \sum_t\mathbf K_t^2\mathbf M^0_t ,
$$
$$
J(\psi) = \min_{\bar{\mathbf M}\in SO(3)}\sum_t\lVert\mathbf M_t(\psi) - \bar{\mathbf M}\rVert_F^2 = 6N - 2\,(\sigma_1 + \sigma_2 + d\,\sigma_3)\bigl(\mathbf S(\psi)\bigr),
$$
$\sigma_i$ the singular values, $d = \operatorname{sign}\det(\mathbf U\mathbf V^\top)$ (`_nuc_so3`). 1-D
search: grid with ≤ 1° spacing and ≥ 9 points on the interval (`camerafree.py:253`), then bounded Brent around the best grid point
(`minimize_scalar`, `xatol` = 1e-7).

**Closed poses unknown** (the case in the stage): a wrong $\psi$ turns head tilt into apparent
jaw rotation, so for each $\psi$ on a 10° grid over $[-180°, 180°)$ the detector of §7.6 runs on
E2 with that $\psi$ (rest reference $\mathbf M_c(\psi)$) with one common threshold, the smallest
rest-phase P90 over the grid + 1°; the $\psi$ with the longest detected closed time is kept. The
pose of each detected run is its most closed 0.3-s window (lowest mean opening,
`_core_start`). Two refinements: fit on these pose samples (thinned evenly to ≤ 3000) within
$\psi \pm 10°$, re-detect with the plan's own threshold, poses again; second fit within
$\pm 2°$, re-detect.

**Identifiability:** tilt range = largest angle between two head verticals over the pose
samples, $\max_{s,t}\arccos(\mathbf v_s\cdot\mathbf v_t)$ (verticals thinned to ≈ 600);
$\psi_b$ is used only when it is ≥ `PSI_MIN_TILT_RANGE_DEG` = 10°; otherwise E2b/E3b fall back to
$\psi_a$ (`flags["psi_b_fallback"]`). Spread: RMS geodesic angle of the pose samples about their
chordal mean. Stored: `cf_summary.psi_b_deg` (the optimum even when not identifiable),
`psi_b_identifiable`, `psi_b_fallback`, `tilt_range_deg`, `psi_b_spread_deg`.

*Paper wording:* "Without optics, the heading offset was estimated from the requirement that
all IMU-detected closed poses correspond to one jaw-in-head orientation: a heading error
scatters the closed poses about the varying head vertical, so the offset minimizing their
chordal scatter was taken, provided the head tilt across the closed poses spanned at least 10°
(hand-eye mountings with IMU heading)."

### 7.6 Closed-pose detection (`detect_closed`, `closed_threshold`, `relative_speed_dps`)

Parameters `stages.camerafree.closed` (fixed; `camerafree.CLOSED`): `rest_quantile` =
0.90, `margin_deg` = 1.0, `speed_max_dps` = 5.0, `min_s` = 0.3, `speed_smooth_s` = 0.1.
- **Threshold:** $\theta^\ast = Q_{0.90}\{\theta(t): t\in\mathcal R\} + 1°$ (`numpy.quantile`,
  linear) of the E2 opening.
- **Relative angular speed:** $\boldsymbol\omega$ of $\mathbf M(t)$ by central differences of its
  quaternion $\mathbf q_{\mathrm{cra}}^{*}\otimes\mathbf q_z(\psi)\otimes\mathbf q_{\mathrm{mta}}$,
  $\boldsymbol\omega_k = \operatorname{Log}_q(\mathbf q_{k-1}^{*}\otimes\mathbf q_{k+1})/(t_{k+1}-t_{k-1})$
  (body frame; `jtsmm.imu.batch.angular_velocity`), averaged causally (vector first) over the last
  $n_\omega = \operatorname{round}(0.1\cdot 200) = 20$ samples, then
  $\lVert\bar{\boldsymbol\omega}\rVert$ in deg/s (white quaternion noise averages out, motion does not).
- **Rule:** closed where $\theta < \theta^\ast$ and $\lVert\bar{\boldsymbol\omega}\rVert < 5$°/s for at
  least $n_{\min} = \lceil 0.3\cdot 200\rceil = 60$ consecutive samples; each run is confirmed causally at
  its sample $a + n_{\min} - 1$.

**Detection metrics** (`cf_summary.det_sensitivity`, `det_precision`; `closed_runs`, `_overlap`):
the optical closed runs are the same rule applied to the optical opening and the norm of the
causal 0.1-s mean of $d\boldsymbol\rho_{\mathrm{opt}}/dt$ (threshold from the optical rest P90). Two runs
match when $\min(b_1, b_1') + t_{\mathrm{tol}} \ge \max(a_0, a_0')$ with `detect_tol_s` = 0.3 s.
Sensitivity = share of optical runs in scored tasks matched by a detected run; precision = share
of detected runs (primary E3) in scored tasks matched by an optical run (run midpoint decides the
task).

*Paper wording:* "Closed-jaw instants were detected from the IMUs alone as periods of at least
0.3 s in which the IMU opening angle stayed below the 90th percentile of the rest phase plus 1°
and the relative angular speed (0.1-s causal mean) stayed below 5°/s."

### 7.7 Closed-pose resets (E3; `anchored_rotation`)

**Causal detection on the anchored estimate itself.** From each anchor on, the opening is
measured against that anchor, so drift since the reference cannot hide a closure. The threshold
$\theta^\ast$ and the speed signal are those of E2 (speed with the initial $\psi_0$). When a run of
the §7.6 rule is confirmed at sample $c = a + n_{\min} - 1$, an anchor is set from the window
$W = [a, c]$:
$$
\psi_k = \arg\min_\psi\sum_{t\in W}\lVert\mathbf M(t;\psi) - \mathbf M_{\mathrm{ref}}\rVert_F^2\ \ (\text{closed form of §7.4, target } \mathbf M_{\mathrm{ref}}),\qquad
\mathbf M_k = \operatorname{cm}\{\mathbf M(t;\psi_k)\}_{t\in W},
$$
with $\mathbf M_{\mathrm{ref}}$ the E2 rest reference (so only the rotation about the IMU world's
vertical changes, and head tilt after the anchor does not meet a stale $\psi$), and the
estimate becomes $\mathbf R(t) = \mathbf A\,\mathbf M(t;\psi_k)\,\mathbf M_k^\top\mathbf A^\top$ (a full reset, like a
zero-velocity update; each anchor stands alone, so closure-to-closure scatter does not random-walk).
While the run lasts, the anchor moves to every later $n_{\min}$-window whose mean opening (in
the frame of the run's first anchor, `camerafree.py:anchored_rotation`) is a new running
minimum by more than
$$
m = 2\,\hat\sigma_w/\sqrt{n_{\min}},\qquad \hat\sigma_w = \frac{1.4826\,\operatorname{MAD}(\Delta\theta)}{\sqrt2}\ \text{over } \mathcal R
$$
(`_white_sd`: the white noise of the E2 opening from successive differences); the minimum of
the opening is the teeth's hard stop, and the run's edges may still be closing (protrusion,
laterotrusion). Each re-anchor takes effect from the window's last sample (causal). The run ends
at the first sample not closed in the final anchor's frame.
- **`hold`** (primary, causal): each anchor holds until the next.
- **`slerp`** (offline, `E3*_slerp`): each run uses its final anchor throughout; between run $k$
  (ending at $b_k$) and run $k+1$ (starting at $a_{k+1}$), with
  $u = (t - t_{b_k-1})/(t_{a_{k+1}} - t_{b_k-1})$,
  $$
  \psi(t) = \psi_k + u\,\operatorname{wrap}(\psi_{k+1}-\psi_k),\qquad
  \mathbf R(t) = \operatorname{Exp}\bigl(-u\,\operatorname{Log}\boldsymbol\Delta_k\bigr)\,\mathbf A\,\mathbf M(t;\psi(t))\,\mathbf M_k^\top\mathbf A^\top,\quad
  \boldsymbol\Delta_k = \mathbf A\,\mathbf M_{k+1}\mathbf M_k^\top\mathbf A^\top .
  $$
- Before the first anchor both are E2 ($\psi_0$, $\mathbf M_{\mathrm{ref}}$).

**Anchor table** (`cf_anchors`, per E3a/E3b one row per closed run, holding the run's final
anchor; `cf_summary.n_anchors` = the number of runs of the primary E3): `t0_s`, `t_confirm_s`, `t1_s`,
`duration_s`, `psi_deg`, `heading_deg` $=\operatorname{wrap}(\psi_k - \psi_0)$ (heading part,
cumulative), `tilt_deg` $=\lVert\operatorname{Log}(\mathbf A\mathbf M_k\mathbf M_{\mathrm{ref}}^\top\mathbf A^\top)\rVert$
(tilt part), `resid_deg` / `resid_open_deg`: the previous anchor's estimate read at this
closed window, $\operatorname{Log}(\mathbf A\,\operatorname{cm}\{\mathbf M(t;\psi_{k-1})\}_W\,\mathbf M_{k-1}^\top\mathbf A^\top)$
(norm and $y$ component; what the estimate had drifted to before the reset); optical scatter of
the anchor window `opt_rot_deg`, `opt_open_deg` (mean $\boldsymbol\rho_{\mathrm{opt}}$) and `opt_inc_mm`
($\lVert\bar{\mathbf p}_{\mathrm{opt}} - \mathbf p_0\rVert$); `matched` (§7.6).

**Drift signature** (diagnostic; `drift_signature`; thresholds `SIGNATURE`, synthetic only):
per anchor the heading and tilt parts above and, per IMU, the *fusion tilt*: horizontal part
(deg) of the mean gravity direction $\operatorname{mean}_W(\mathbf R\,\mathbf f/\lVert\mathbf f\rVert)$ (zero for a
correct fusion, whatever the heading). Net changes = median of the last 3 anchors minus the
first 3. Label, rules in order: net $< 0.3°$ → none; fusion-tilt change $\ge 0.5\times$ tilt
change and $\ge 0.3°$ → tilt; largest anchor-to-anchor step $\ge 0.5\times$ net → slip; heading
$\ge 2\times$ tilt → heading; else unexplained. Stored: `cf_summary.drift_label`,
`drift_heading_deg`, `drift_tilt_deg`, `drift_fusion_tilt_deg`, `drift_step_frac`,
`drift_net_deg`; anchor and closed masks in the arrays `anchor`, `closed`.

*Paper wording:* "At every IMU-detected closed pose the rotation estimate was reset: the heading
offset was re-fitted to the reference closed pose and the current closed pose became the new
reference, taken from the most closed 0.3-s window of the closed period (closed-pose resets;
causal)."

### 7.8 Condyle hinge (E4hinge; `incisor_position`)

The incisor from the primary E3 rotation about the condyle midpoint, no translation:
$$
\hat{\mathbf p}(t) = \mathbf R(t)\,(\mathbf p_0 - \mathbf c_0) + \mathbf c_0 \ \bigl(+\,\hat{\boldsymbol\tau}(t)\ \text{for E4own/E4gen}\bigr).
$$

*Paper wording:* "The incisor position was first obtained by rotating the closed-pose incisor
about the condyle midpoint with the IMU rotation (condyle hinge)."

### 7.9 Own and generic rotation-to-translation models (E4own, E4gen; `TranslationModel`)

**Features** (`translation_features`, `FEATURES`): from the rotation vector
$\boldsymbol\rho = (\rho_x,\rho_y,\rho_z)$ (deg, maxillary frame, since the closed pose) and the opening rate
$\dot\theta$ (deg/s),
$$
\mathbf F(t) = \bigl(\rho_x,\ \rho_y,\ \rho_z,\ \rho_x^2,\ \rho_y^2,\ \rho_z^2,\ \tanh(\dot\theta/h)\,\rho_y\bigr)\in\mathbb R^7,\qquad h = \texttt{HYST\_DPS} = 20\ ^\circ/\mathrm s ,
$$
the last term a smooth hysteresis (opening vs closing branch; a hard sign would jump at every
turning point). $\dot\theta$ (`theta_rate_dps`): `timebase.gradient` of $\rho_y$ with a centred
moving mean of `odd_window`(0.05 s, 200 Hz) = 11 samples, per finite segment.

**Model** (no intercept: $\boldsymbol\tau = 0$ at the closed pose where $\mathbf F = 0$):
$\hat{\boldsymbol\tau} = \mathbf F\,\mathbf C$, $\mathbf C\in\mathbb R^{7\times3}$. **Ridge** with the features scaled to unit
RMS, $\mathbf Z = \mathbf F\,\operatorname{diag}(\mathbf s)^{-1}$, $s_j = \sqrt{\overline{F_j^2}}$:
$$
\mathbf C = \operatorname{diag}(\mathbf s)^{-1}\bigl(\mathbf Z^\top\mathbf Z + \lambda\,n\,\mathbf I\bigr)^{-1}\mathbf Z^\top\mathbf Y ,
$$
$n$ = training samples, solved with `numpy.linalg.solve` (`_solve`). **λ** from
`ridge_log10_lambda` = [−8, 2, 21] → 21 log-spaced values $10^{-8}\dots10^{2}$, chosen by blocked
CV inside the training data: $\operatorname{CV}(\lambda) = \sqrt{\frac1n\sum_{\text{folds}}\sum_{i\in\text{test}}\lVert\mathbf F_i\mathbf C_{-\text{fold}} - \mathbf Y_i\rVert^2}$
(3-D RMS), minimum, ties to the larger λ; then refit on all data (`cv_rmse`, `cv_curve`).
- **E4own** (`TranslationModel.fit`): training data = the participant's optical free_motion
  on the grid (`calibration_phase`), $\boldsymbol\rho = \boldsymbol\rho_{\mathrm{opt}}$, $\dot\theta$ from
  $\rho_{\mathrm{opt},y}$, $\mathbf Y = \boldsymbol\tau_{\mathrm{opt}} = \mathbf c_{\mathrm{opt}}(t) - \mathbf c_0$; folds:
  `ridge_folds` = 5 contiguous blocks of the valid samples (neighbouring samples are not
  independent). Prediction from the primary E3: $\boldsymbol\rho_{\mathrm{E3}} = \operatorname{Log}\mathbf R_{\mathrm{E3}}$ and its
  $\dot\theta$ (unfiltered E3; array `rotvec_deg__E3`). Stored: `cf_summary.e4own_lambda`,
  `e4own_cv_rmse_mm`.
- **E4gen** (`TranslationModel.fit_generic`, `cohort_camerafree.generic`; leave one subject
  out): training blocks = the optical free_motion couplings of **every other participant with
  optics** (`stage_camerafree.optical_block`; the participant recorded outside the protocol is included in the n = 14 store, E32; none
  excluded: `exclude_generic` = []), $\mathbf Y_j = \boldsymbol\tau_{\mathrm{opt},j}/s_j$ (translation per mm of
  incisor–condyle distance); λ by leave-one-participant-out CV inside the training set;
  prediction $\hat{\boldsymbol\tau}_i = s_i\,\mathbf F(\boldsymbol\rho_{\mathrm{E3},i})\,\mathbf C$. Stored: `cf_generic`
  (scores), `cf_generic_models` (`n_train`, `lambda`, `cv_rmse_rel`, `n_samples`).

*Paper wording:* "Condylar translation was predicted from the IMU rotation by a ridge
regression without intercept on the rotation-vector components, their squares and a smooth
opening/closing hysteresis term, with the penalty chosen by blocked cross-validation; the own
model was trained on the participant's optical free-movement data, the generic model on all
other participants' data scaled by the incisor–condyle distance."

### 7.10 Accelerometer bridging (E5; `accel_bridge`)

**State and input.** The relative IMU-site position $\mathbf e(t) = \mathbf S_{\mathrm{mta}}(t) - \mathbf S_{\mathrm{cra}}(t)$
in the head IMU's world. Gravity is removed per IMU with its own orientation, the jaw IMU's
world is turned into the head IMU's, and the difference is $\ddot{\mathbf e}$ directly (no
rotating-frame terms; the lever arms are the sites themselves), at the jaw accelerometer's
sample times (head accelerometer and both quaternions interpolated there):
$$
\mathbf u(t) = 10^3\bigl[\mathbf R_z(\psi(t))\bigl(\mathbf R_{\mathrm{mta}}\mathbf f_{\mathrm{mta}} - g\,\mathbf e_z\bigr) - \bigl(\mathbf R_{\mathrm{cra}}\mathbf f_{\mathrm{cra}} - g\,\mathbf e_z\bigr)\bigr]\ \ \mathrm{mm/s^2},
$$
$\psi(t)$ = the primary E3's heading in use, unwrapped and interpolated.

**Anchors (position measurements).** At every sample of the anchor windows (the most closed
0.3 s of each detected run, `Anchored.anchor_mask`; dense, so the velocity is pinned too):
$$
\mathbf y = \mathbf W(t)\,(\mathbf s_{\mathrm{mta},0} - \mathbf s_{\mathrm{cra}}),\qquad \mathbf W(t) = \mathbf R_{\mathrm{cra}}(t)\,\mathbf A^\top\ (\text{head world}\leftarrow\text{maxillary}).
$$
Whole runs would be wrong anchors: their edges may still slide by up to a millimetre.

**Filter** (`jtsmm.imu.position.position_kf`, per axis, the three axes sharing the dynamics):
state $\mathbf x = (p, v, b)$ with a world-frame accelerometer bias $b$; between events
($\Delta = t_{k}-t_{k-1}$, input $u$ held from the last acceleration sample):
$$
\mathbf x_k = \mathbf F\mathbf x_{k-1} + \mathbf B\,u,\quad
\mathbf F = \begin{pmatrix}1 & \Delta & -\tfrac12\Delta^2\\ 0 & 1 & -\Delta\\ 0 & 0 & 1\end{pmatrix},\quad
\mathbf B = \begin{pmatrix}\tfrac12\Delta^2\\ \Delta\\ 0\end{pmatrix},\quad
\mathbf Q = q_a\begin{pmatrix}\Delta^3/3 & \Delta^2/2 & 0\\ \Delta^2/2 & \Delta & 0\\ 0 & 0 & 0\end{pmatrix} + \begin{pmatrix}0&0&0\\0&0&0\\0&0&q_b\Delta\end{pmatrix};
$$
measurement $y = p + \varepsilon$, $\varepsilon\sim\mathcal N(0, r^2)$, standard Kalman update
($\mathbf H = (1,0,0)$, $\mathbf K = \mathbf P\mathbf H^\top/(\mathbf H\mathbf P\mathbf H^\top + r^2)$); start at the first measurement
($p$ = measurement, $v = b = 0$, $\mathbf P_0 = \operatorname{diag}(r^2, 200^2, 500^2)$); Rauch–Tung–Striebel
smoother $\hat{\mathbf x}^s_k = \hat{\mathbf x}_k + \mathbf C_k(\hat{\mathbf x}^s_{k+1} - \hat{\mathbf x}^-_{k+1})$,
$\mathbf C_k = \mathbf P_k\mathbf F_{k+1}^\top(\mathbf P^-_{k+1})^{+}$ (E5, primary) and the forward filter
(E5_causal). Each acceleration sample is placed at $t_i - \tfrac12\Delta t_a$ so that it holds over
the interval centred on its stamp (midpoint rule; the zero-order hold would otherwise delay the
integration by half a sample).

**Noise from the rest phase** (no tuning; floors `stages.camerafree.bridge`):
$q_a = \sigma_a^2\,\Delta t_a$ with $\sigma_a = \max(\hat\sigma_w(\mathbf u\ \text{on}\ \mathcal R),$
`sigma_acc_floor_mm_s2` = 1.0) and $\Delta t_a$ the median accelerometer interval;
$r = \max(\hat\sigma_w(\mathbf y\ \text{on}\ \mathcal R),$ `r_pos_floor_mm` = 0.05); $q_b$ =
`bias_rw`² = 20² (mm/s³)²·s. $\hat\sigma_w$ = white SD from successive differences,
$1.4826\,\operatorname{MAD}(\Delta x)/\sqrt2$, RMS over axes (`_white_sd`).

**Back to the maxillary frame** (`to_max`): jaw site $\mathbf q = \mathbf s_{\mathrm{cra}} + \mathbf W^\top\hat{\mathbf e}$,
incisor $\hat{\mathbf p} = \mathbf q + \mathbf R_{\mathrm{E3}}(\mathbf p_0 - \mathbf s_{\mathrm{mta},0})$, condyle-midpoint
translation $\hat{\boldsymbol\tau} = \mathbf q - \mathbf c_0 - \mathbf R_{\mathrm{E3}}(\mathbf s_{\mathrm{mta},0} - \mathbf c_0)$.

**Why it diverges** (error model of the code; the data show the outcome, E5 − E4own about
+333 mm in the cohort, an07, not the split between the terms):
- $\mathbf u$ is the small difference of two gravity-removed specific forces of magnitude ≈ $g$. A
  tilt error $\delta$ of either game rotation vector leaks $g\sin\delta \approx 171$ mm/s² per
  degree; the jaw-in-head acceleration of interest is of the same order. A heading error
  $\delta\psi$ leaves $\approx\delta\psi\,\lVert\mathbf a_{\mathrm{head}}\rVert$ of the common head
  acceleration.
- Between anchors the double integrator is unobservable: a constant acceleration error $\delta a$
  gives a position error $\tfrac12\delta a\,T^2$ (1° over $T$ = 3 s: ≈ 0.8 m), and white noise grows as
  $\sqrt{q_a T^3/3}$. The bias state can absorb only an offset that is constant over several
  anchors; the RTS smoother constrains both ends of a gap but not its middle, and the causal
  filter has only the start.
- The anchor variance $r^2$ is the small rest scatter (tenths of a mm), so the filter trusts the
  anchors and the integration between them fully; the noise floors keep it regular, they do not
  limit the drift. Dynamic calibration updates (§1) add steps to the bias.

*Paper wording:* "Accelerometer bridging integrated the difference of the two gravity-compensated
accelerations twice with a Kalman filter (position, velocity, accelerometer bias) and
Rauch–Tung–Striebel smoother, using the IMU sites at the detected closed poses as position
measurements; all noise parameters were taken from the rest phase."

### 7.11 Accelerometers + own model (E6; `fuse_translation`)

E5 plus the E4own prediction as further position measurements:
$$
\mathbf y^{\mathrm{ps}}(t) = \mathbf W(t)\bigl(\mathbf R_{\mathrm{E3}}(\mathbf s_{\mathrm{mta},0} - \mathbf c_0) + \mathbf c_0 + \hat{\boldsymbol\tau}_{\mathrm{own}}(t) - \mathbf s_{\mathrm{cra}}\bigr),
$$
thinned to `pseudo_hz` = 5 Hz outside the anchor samples (the model error is correlated over a
movement, so denser samples would overweight it). Each measurement enters the update of §7.10
with its own variance (`position_kf(..., r_meas_mm=)`): the anchors exactly as in E5 (every
anchor sample, SD $r$), the pseudo-measurements with the per-axis SD
$$
r_{\mathrm{ps}} = \sigma_{\mathrm{ps}}/\sqrt3,\qquad
\sigma_{\mathrm{ps}} = \sqrt{\tfrac1N\textstyle\sum_k\lVert\hat{\boldsymbol\tau}_k-\boldsymbol\tau_k\rVert^2}
$$
($\sigma_{\mathrm{ps}}$ = E4own's inner-CV `cv_rmse`, a 3-D RMS, so $r_{\mathrm{ps}}^2$ is its mean
per-axis variance), i.e. $S_k = \mathbf H\mathbf P\mathbf H^\top + r_k^2$ with $r_k = r$ or
$r_{\mathrm{ps}}$. $\mathbf P_0$ uses the anchor SD $r$. Smoother (E6) and forward filter
(E6_causal). Stored: `pseudo_sigma_3d_mm` ($\sigma_{\mathrm{ps}}$), `pseudo_sigma_mm`
($r_{\mathrm{ps}}$), `pseudo_hz`, `n_pseudo`. (Erratum E40: before, the
pseudo-measurements had SD $\max(\sigma_{\mathrm{ps}}, r)$, √3 too large per axis, and the
anchors, thinned to 20 Hz, were repeated up to 50 times to emulate their smaller variance,
because `position_kf` had one common measurement variance. E5 is unchanged.)

*Paper wording:* "The fused estimate added the own-model translation as pseudo-measurements at
5 Hz, each axis weighted by the model's cross-validated error (3-D RMS divided by √3), while the
closed-pose measurements kept their own rest-phase variance (accelerometers + own model)."

### 7.12 Movement-class model (E7): not computed

The rule mapping a predicted movement class (from the decoders' leave-one-subject-out
classifiers) to a translation model is not fixed in the plan, and `ml_eval` is a plugin stage
that the D22 sensitivity variants skip, so a dependency would drop the cohort stage from those
runs. Rows with NaN and the reason (`cohort_camerafree.E7_NOTE`) stand in `cf_cohort`.

### 7.13 Oracle analyses next to the own model (X5b, X5c; never camera-free results)

**Decomposition (X5b; `oracle_incisors`, `stage_camerafree.decomposition_rows`; `cf_decomposition`):**
- *optical rotation + own model* (`X5b_model`, component `translation_model`):
  $\hat{\mathbf p} = \mathbf R_{\mathrm{opt}}(\mathbf p_0 - \mathbf c_0) + \mathbf c_0 + \hat{\boldsymbol\tau}_{\mathrm{own}}(\boldsymbol\rho_{\mathrm{opt}}, \dot\theta_{\mathrm{opt}})$:
  the translation-model error alone;
- *IMU rotation + optical translation* (`X5b_rotation`, component `rotation`):
  $\hat{\mathbf p} = \mathbf R_{\mathrm{E3}}(\mathbf p_0 - \mathbf c_0) + \mathbf c_0 + \boldsymbol\tau_{\mathrm{opt}}$, error
  $(\mathbf R_{\mathrm{E3}} - \mathbf R_{\mathrm{opt}})(\mathbf p_0 - \mathbf c_0)$: the rotation-induced error alone;
- the full own model (`E4own` rows copied, component `full`). Its error is the sum of the two plus
  the model's response to the rotation error, $\hat{\boldsymbol\tau}(\boldsymbol\rho_{\mathrm{E3}}) - \hat{\boldsymbol\tau}(\boldsymbol\rho_{\mathrm{opt}})$, so the
  three RMSEs do not add.

**Movement-class model (X5c; `fit_task_models`, `predict_by_task`, `task_model_rows`;
`cf_task_model`):** the E4own family fitted per movement class on the free_motion samples whose
optical intent label (labels stage; label of the first optical frame at or after each grid time,
`labels_on_grid`) is in the class (`x5c_task_labels`: open_close and chewing → OPEN+CLOSE,
protrusion_retrusion → PROTRUSION+RETRUSION, left_right → LEFT+RIGHT; free-motion labels cannot
tell chewing from open-close); a class with fewer than `x5c_min_cal_s` = 2.0 s (400 samples) is not
fitted (row with the reason). Applied in each scored task with that task's class model (the
true task label: the upper bound of the deferred E7), on the E3 rotation; NaN between tasks.

*Paper wording:* "To attribute the incisor error, the IMU rotation was replaced by the optical
rotation (translation-model error alone) and, separately, the model translation by the optical
condylar translation (rotation-induced error alone). A further oracle analysis fitted one
translation model per movement class and applied it with the true task label."

### 7.14 Scoring (`camerafree.score`, `stage_camerafree.score_rows`, `lowpass`)

**Low-pass.** Every estimate (rotation and incisor) goes through the ground truth's pose
low-pass on the 200 Hz grid: `se3.lowpass` (an02 §1) with `signals.kinematics.lowpass_hz` = 6 Hz,
`lowpass_order` = 2, Butterworth, zero-phase (`sosfiltfilt`), per finite segment of ≥ 15
samples; rotation filtered as rotation vectors about the segment's mean rotation,
$\tilde{\mathbf R} = \bar{\mathbf R}\operatorname{Exp}\bigl(\mathrm{LP}[\operatorname{Log}(\bar{\mathbf R}^\top\mathbf R)]\bigr)$, positions per
component. The optics are not filtered again (`fc_hz=None` in `score_rows`).

**Metrics** per task mask (≥ 20 samples finite in both), with
$\mathbf e(t) = \tfrac{180}{\pi}\operatorname{Log}(\tilde{\mathbf R}_{\mathrm{est}}\mathbf R_{\mathrm{opt}}^\top)$:

| Column | Definition |
|---|---|
| `rmse_open_deg`, `bias_open_deg`, `max_open_deg` | RMS, mean, max abs of $\theta(\tilde{\mathbf R}_{\mathrm{est}}) - \theta(\mathbf R_{\mathrm{opt}})$ |
| `rmse_geo_deg`, `max_geo_deg` | RMS and max of $\lVert\mathbf e\rVert$ (geodesic) |
| `rmse_r{x,y,z}_deg` | RMS of each component of $\mathbf e$ (per axis, maxillary frame) |
| `rmse_inc_mm`, `max_inc_mm` | RMS and max of $\lVert\tilde{\mathbf p}_{\mathrm{est}} - \mathbf p_{\mathrm{opt}}\rVert$ |
| `rmse_inc_{x,y,z}_mm`, `bias_inc_{x,y,z}_mm` | per-axis RMS and mean of $\tilde{\mathbf p}_{\mathrm{est}} - \mathbf p_{\mathrm{opt}}$ |

Rows per task (every experiment phase except `rest`, plus the pooled `scored` mask); `scored`
marks the scored tasks (all except `exclude_scored` = [rest, free_motion]: open_close,
chewing, protrusion_retrusion, left_right), `task_group` = rotational (open_close, chewing) or
translational (protrusion_retrusion, left_right). Tables `cf_rotation` (E1…E3b_slerp),
`cf_translation` (E4hinge, E4own, E5, E5_causal, E6, E6_causal), `cf_generic` (E4gen).

**Opening-error drift** (`cf_drift`, `cf_drift_slopes`, `cf_drift_cohort`): signed
$\theta_{\mathrm{est}} - \theta_{\mathrm{opt}}$ of E1, E2a, E2b, E3a, E3b at 1 Hz (`drift_hz`) over the
scored samples, against the time since the reference (midpoint of the rest phase) and since the
end of the last anchor window of the primary E3. Per participant, estimator and clock the OLS
slope (°/min; ≥ 10 samples): $\hat\beta = \arg\min\sum(\varepsilon_k - \alpha - \beta\,t_k/60)^2$; per
estimator and clock a random-intercept mixed model $\varepsilon_{ik} = \alpha + a_i + \beta t_{ik} + e_{ik}$
(REML, `stats.core.mixed_model`; descriptive, 1 Hz samples are autocorrelated so its SE is
optimistic).

**Cycle amplitudes** (`cf_cycles`, `cf_cycles_ba`): per `ok` kinematics cycle (≥ 10 grid
samples) the opening amplitude $\max(\theta - \theta_{\mathrm{start}})$ and the incisor amplitude
$\max\lVert\mathbf p - \mathbf p_{\mathrm{start}}\rVert$, estimate and optics; repeated-measures Bland–Altman with
participant-cluster bootstrap CIs per estimator, quantity and task group (≥ 3 cycles);
`loa_within_bound` = $\max\lvert\mathrm{LoA}\rvert \le$ 1.2° / 2.0 mm.

*Paper wording:* "Each estimate was passed through the same 6-Hz zero-phase pose low-pass as the
optical reference and scored per task by the root-mean-square opening-angle error, the geodesic
rotation error and the 3-D incisor error against the optics. Drift was the least-squares slope
of the signed opening error against time since the reference, or since the last reset."

**Free motion (X5d, exploratory; J15, 2026-11-09).**
- **What:** the stored `free_motion` rows (`scored` False), summarised per estimator by
  `derived:cf_free_motion` (`viz/derived_dataset.py`, unhashed). Free motion is the
  fastest and most varied phase, the closest to unscripted device use.
- **Leakage-free estimators:**
  - opening angle: E2a, E2b and the primary E3;
  - incisor: E4hinge, E4gen and E5.
- **In-sample references (optimistic):** E1, E4own and E6, which are fitted on free motion
  or use a model fitted on it.
- **Statistic:** per participant the one free-motion RMSE; cohort median with 95 % BCa CI
  (4000 resamples, study seed). No bound, no test.
- **Macros:** `\numCfFreeOpen…` (opening angle) and `\numCfFreeInc…` (incisor).
- **Caveat:** E2b and E3 detect closed poses over the whole take, including free motion, but
  from the IMUs only. The jaw-plate landmark positions are a median over all phases, a rigid
  constant rather than a motion fit.

*Paper wording:* "As an exploratory addition, the estimators that do not use the participant's
free-motion optics were also scored on the free-motion phase, the fastest and most varied
movement."

### 7.15 Cohort statistics, bounds and verdicts (`stats/cohort_camerafree.py:run`)

- **Participant value** (`participant_values`, `cf_participant`): for participant $i$ and estimator
  $e$, $v_{ie} = \operatorname{median}_{\text{scored tasks}}\mathrm{RMSE}_{ie,\text{task}}$ for `rmse_open_deg`,
  `rmse_geo_deg`, `rmse_inc_mm` (all scored tasks), `rmse_inc_mm_rot` (rotational tasks),
  `rmse_inc_mm_trans` (translational tasks); the primary E3 also appears as `E3`. Also the
  detection sensitivity/precision and the drift slopes.
- **Cohort** (`cf_cohort`, `describe_rows`): median of the $v_{ie}$ with the 95 % BCa CI
  (`stats.core.describe`, `n_boot` = 4000, seed = study seed; an07).
- **Bounds** (`BOUNDS`): opening 1.2°, incisor 2.0 mm (all three incisor outcomes); geodesic none.
  The plan's primary outcomes: opening RMSE of the anchored rotation (E3) and incisor RMSE of the
  own model in open_close and chewing (`rmse_inc_mm_rot`).
- **Verdict** (`verdict`): *within* if CI$_{\mathrm{hi}}$ < bound; else *exceeds* if CI$_{\mathrm{lo}}$ > bound;
  else *compatible* if median < bound; else *inconclusive*.
- **Paired contrasts** (`contrasts`, `cf_contrasts`; `stats.families.paired_contrast`, an07):
  $d_i = v_{ia} - v_{ib}$; exact two-sided Wilcoxon signed-rank on the non-zero $d_i$ (normal
  approximation with ties), Hodges–Lehmann estimate with its exact 95 % CI, `n_a_better` = #$\{d_i<0\}$;
  Holm within the family. Rotation family on `rmse_open_deg`: E2a−E1, E2b−E2a, E3−E2a;
  translation family on `rmse_inc_mm`: E4own−E4gen, E4gen−E4hinge, E5−E4own, E6−E4own; drift
  family (H2): $\lvert\beta_{\mathrm{E3,anchor}}\rvert - \lvert\beta_{\mathrm{E2a,ref}}\rvert$.
- **Oracle** (`oracle`; `cf_oracle_participant`, `cf_oracle_cohort`, `cf_oracle_contrasts`): the
  same participant values (and each movement task's own incisor RMSE, `rmse_inc_mm__<task>`) for
  X5b_model, X5b_rotation, X5c and E4own; cohort medians with BCa CI and no bound verdict;
  contrasts X5b_rotation − X5b_model (on `rmse_inc_mm_rot`, `rmse_inc_mm_trans`) and X5c − E4own
  (on `rmse_inc_mm`, `_rot`, `_trans`), Holm within each.
- Values: an07 §15.

*Paper wording:* "Per participant, the task RMSEs were summarized by their median over the four
movement tasks; the cohort median is reported with a 95 % BCa bootstrap interval and judged
against the bound (1.2° for opening, 2.0 mm for the incisor) as within (interval below the bound),
exceeding (interval above), or compatible / inconclusive otherwise. Estimators were compared
pairwise by exact Wilcoxon signed-rank tests with Hodges–Lehmann estimates, Holm-adjusted within
the rotation and the translation family."

### 7.16 Stored outputs and parameters

- **`camerafree` tables:** `cf_summary` (one row: configuration, lever arms `lever_<slot>_<axis>_mm`,
  `rest_s`, `closed_vs_tclosed_deg`, $\psi$ values and flags, `rest_p90_deg`, `threshold_deg`,
  `n_anchors`, `n_closed_opt`, detection metrics, drift signature, E4own λ and CV RMS, E1 checks,
  `incisor0_*`, `condyle0_*`, `scale_mm`, `duration_s`, `note`), `cf_rotation`,
  `cf_translation`, `cf_cycles`, `cf_anchors`, `cf_drift`, `cf_decomposition`, `cf_task_model`.
  Without a `mounting` only `cf_summary` with the reason. **Arrays:** `t_s`, `phase`,
  `phase_names`, `anchor`, `closed`, `rotvec_deg__opt`, `rotvec_deg__E3` (before the low-pass),
  `incisor_mm__opt`, `open_deg__opt`, `open_deg__<rotation estimator>`, `incisor_mm__<translation estimator>`.
- **`camerafree_cohort` tables:** `cf_participant`, `cf_cohort`, `cf_contrasts`, `cf_generic`,
  `cf_generic_models`, `cf_drift_slopes`, `cf_drift_cohort`, `cf_cycles_ba`, `cf_oracle_participant`,
  `cf_oracle_cohort`, `cf_oracle_contrasts`.
- **Parameters** (`stages.camerafree`, study file): `mounting` (above), `lever` = session,
  `rest_mask` = phase, `calibration_phase` = free_motion, `exclude_scored` = [rest, free_motion],
  `primary_e3` = E3b, `grid_hz` = 200, `closed` (§7.6), `bridge` (`sigma_acc_floor_mm_s2` = 1.0,
  `r_pos_floor_mm` = 0.05, `bias_rw` = 20), `pseudo_hz` = 5, `ridge_log10_lambda` = [−8, 2, 21],
  `ridge_folds` = 5, `detect_tol_s` = 0.3, `drift_hz` = 1, `x5c_task_labels`, `x5c_min_cal_s` = 2.0;
  `max_opt_gap_frames` = 2.5 (code default). Module constants: `PSI_MIN_TILT_RANGE_DEG` = 10,
  `HYST_DPS` = 20, `SIGNATURE`. Every sensitivity variant with `camerafree` in its `skip_stages`
  (`labels_p95`, `c2_symmetric_window`, `burst_3db`, `burst_10db`, `onset_ksd_5`,
  `lowfreq_mask`, `lowpass_*hz`, `imu_latency`) skips the stage and its dependants `camerafree_cohort`,
  `imu_dyncal` and `imu_dyncal_cohort` (`sensitivity.py:skipped_stages`).
