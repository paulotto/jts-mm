<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an14 — System validation against Qualisys and Delsys (systems paper)

How the validation study compares JTS-MM (the open-source multimodal jaw tracking system of
this project) with a Qualisys Oqus system (200 Hz) and a Delsys Trigno Avanti electromyography <!-- num-ok: instrument setting -->
(EMG) system (≈ 4.37 kHz), and the exact rule and mathematics behind each number.
Every computation that yields a stored or reported quantity has: what and why, the formula
as the code computes it, the parameters in force, the code pointer and a **Paper wording**
paragraph for the Methods of the systems paper (plain, past tense, no internal plan codes).
Symbols are listed in §0 (Notation and conventions). Further abbreviations: IMU = inertial
measurement unit; MVC = maximal voluntary contraction; TMJ = temporomandibular joint; RMS(E) =
root mean square (error); SD = standard deviation; CI = confidence interval; SPM = statistical
parametric mapping; CCC = concordance correlation coefficient; ICC = intraclass correlation
coefficient; CMC = coefficient of multiple correlation; TKEO = Teager–Kaiser energy operator;
SNR = signal-to-noise ratio; HP / MP / CT = head plate, mouthpiece, calibration tool.

It is a separate study from the dataset study (an01–an13):

- **Study files:** `experiments/jtsmm_validation/`: `study.yaml`, `cohort.yaml`,
  `reference.yaml`, `analysis_plan.md` (frozen 2026-10-22, tag `validation-plan-frozen`),
  `paper_systems.yaml`.
- **Store:** `data/analysis/jtsmm_validation/`; the paper reads the frozen snapshot `v15` (2026-11-16: the re-run under the released code, every value equal to `v14`; `v14`, 2026-11-06: `v13` plus VD26–VD28, no `v13` value changed; `v13`, 2026-11-03: `v12` with the JTS-MM digastric sides as cabled, EV12; `v12` = the `v11` values plus VD24), **pinned** in
  `paper_systems.yaml` and the dataset `cite:` block. Each entry of the change log of the
  validation plan (`experiments/jtsmm_validation/change_log.md`) names the snapshot its
  results come from; the list is in "Snapshots (history)" below.
- **Reporting rule for n = 2 (EV8, decision of the authors 2026-10-29 and 2026-10-30):** every result per
  participant (P1 = s10, P2 = s14) plus their mean as a descriptive summary ("mean (P1, P2)");
  no interval, test or equivalence verdict on the mean; the frames or cycles of the two
  participants are never pooled into one bootstrap or RMS; where the mean is quoted, both
  participants' values are stated with it. One exception: the jaw-IMU lever arm (VD15) is one
  physical constant estimated from both hand-eye sessions. P1 and P2 are reported as two
  separate reference measurements, without ranking; differences are explained by the error
  budget (VD22).
- **Stage group:** `validation`.

**Code:**
- `software/biomech/biomech/validation/`: `selection`, `sync`, `registration`, `anatomy`,
  `tip`, `quality`, `evaluate`, `landmarks`, `kinematics`, `agreement`, `fusion`, `emg`,
  `trueness`, `sampling` (VD9, EV7), `lever` (VD15; re-exports `biomech/imu/lever.py`),
  `accel` (VD17), `hinge` (VD21), `budget` (VD22), `emg_side` (VD23), `synthetic` (synthetic
  takes), `stages`, `stages_ref`, `stages_camerafree` (VD20);
- the estimators in `software/jtsmm/jtsmm/imu/fusion.py` (F0–F2) and
  `software/jtsmm/jtsmm/imu/position.py` (F2P);
- the readers in `biomech/io/reference.py`;
- views `viz/views/validation*.py`.

Generic statistics (Bland–Altman, CCC, ICC(A,1), bootstrap variants, SPM, CMC) are defined
in an07; SE(3) primitives, the maxillary frame and the hinge fits in an02; the IMU
preprocessing and the camera-free estimators in an04. This document gives the validation-
specific use and parameters.

Added after the plan was frozen: the CT label-swap erratum and the point-marking analysis,
the calculation-audit errata and additions EV6–EV11, VD18–VD21, the error budget VD22 and the
EMG side check VD23 (the entries with dates and reasons are in the change log of the
validation plan).

## Snapshots (history)

Each entry of the change log of the validation plan names the snapshot its results come from.

- `v1` (2026-10-22): before the errata EV1/EV2, kept as the record;
- `v2` (2026-10-23): after them, with the VD4 marking accuracy and the VD5 fusion in
  both frame forms;
- `v3` (2026-10-23): plus the VD6 position fusion and the VD7 alignment sensitivity
  (re-frozen after VD8, labels only);
- `v4` (2026-10-23): VD9–VD11, EV4; `v5`: VD12; `v6` (2026-10-27): VD13–VD16; `v7`
  (2026-10-28): VD17;
- `v8` (2026-10-29): the calculation-audit errata and additions EV6, EV7,
  EV9–EV11, VD18–VD21 (every changed value mapped to an entry);
- `v10` (2026-10-31): EV9 (CCC with 1/n moments), the
  additions VD22 and VD23, the VD22 separability flag; every other value equals v8;
- `v11` (2026-11-01): EV13 (condyle row of the error budget)
  and EV14 (waveform band with the effective sample size); every other value equals v10;
- `v12`: the `v11` values plus VD24; `v13` (2026-11-03): `v12` with the JTS-MM digastric sides
  as cabled (EV12); `v15` (2026-11-06): `v13` plus VD26–VD28, no `v13` value changed;
  `v15` (2026-11-16): the re-run under the released code, every value equal to `v14`;
  **pinned** in `paper_systems.yaml` and the dataset `cite:` block (the snapshot the paper
  reads).

## 0. Notation and conventions

| Symbol | Meaning |
|---|---|
| $t_S$, $t_Q$, $t_D$ | time on the JTS-MM master clock (exposure centre of a frame), on the Qualisys clock, on the Delsys clock (s) |
| $c$, $\alpha$, $t_{\mathrm{ref}}$ | clock offset (Qualisys minus master at $t_{\mathrm{ref}}$, s), relative clock-rate error (reported in ppm), centring time |
| $T_{S,b}(t)$, $T_{Q,b}(t)$ | pose (4×4, mm) of body $b$ in JTS-MM's / Qualisys' own world; $b$ ∈ {HP head plate, MP mouthpiece, CT calibration tool} |
| $R(T)$, $p(T)$ | rotation and translation part of a pose |
| $X$, $Y_b$ | world registration and body registration, $T_{Q,b} \approx X\,T_{S,b}\,Y_b$ |
| $\mathrm{Log}$, $\mathrm{Exp}$ | rotation vector of a rotation matrix and its inverse (rad) |
| $\angle(R)$ | geodesic angle $\lVert\mathrm{Log}\,R\rVert$ |
| $F_{\max}$, $F_{\mathrm{mand}}$ | anatomical frames: head ← maxillary, jaw ← mandibular (from the digitized teeth) |
| $M(t)$ | jaw-in-head motion $F_{\max}^{-1}\,T_{\mathrm{head}}^{-1}\,T_{\mathrm{jaw}}\,F_{\mathrm{mand}}$ of one system |
| $A$, $B$ | constant maxillary and mandibular frame alignment, $M_Q \approx A\,M_S\,B$ |
| $p_{\mathrm{inc}}$, $p_{\mathrm{tmj}}$ | lower incisor ($= p(M)$, origin of the mandibular frame) and condyle points in the maxillary frame (mm) |
| $\theta$ | opening angle: $y$ component (maxillary lateral axis) of $\mathrm{Log}(R(M)\,R(M_{\mathrm{rest}})^\top)$ (deg); the $\theta_y$ of an02 §3, here relative to the rest pose |
| $e_i$, $d_j$ | per-frame error JTS-MM minus Qualisys; per-cycle feature difference JTS-MM minus Qualisys |
| $\sigma_{\mathrm{obs}}$, $\sigma_{\mathrm{ref}}$, $\sigma_{\mathrm{JTS}}$ | observed disagreement, reference (Qualisys) uncertainty, the JTS-MM part |
| $n$, $n_{\mathrm{eff}}$, $\rho_1$ | number of cycles, effective sample size, lag-1 autocorrelation |

Conventions. The RMS of 3-D errors is $\sqrt{\tfrac1N\sum_i\lVert e_i\rVert^2}$; the
"bias-removed" (dynamic) RMS is $\sqrt{\sum_{a\in\{x,y,z\}} s_a^2}$ with $s_a$ the SD (ddof 1)
of the error component $a$. All intervals are 95 %; a percentile bootstrap uses the 2.5 and <!-- num-ok: definition -->
97.5 % quantiles of the replicates. Segments: the held-out movement tasks (`open_close`, <!-- num-ok: definition -->
`protrusion_retrusion`, `left_right`, `chewing`, `free_motion`) and `rest`, each trimmed by
0.5 s at both ends and cut 0.5 s before the first CT appearance (`selection.movement_intervals`, <!-- num-ok: definition -->
D0b); the landmark holds and the two sync windows are reported, never pooled.

## 1. Design in one paragraph

Two participants (s10 → P1, s14 → P2) each performed the dataset protocol once (the task
battery of an01 §1 / an02 §2,
with the landmark holds and the two sync windows) while both systems tracked the headpiece
(HP), the mouthpiece (MP) and the calibration tool (CT). Ethics: TU Darmstadt ethics approval
EK 3/2025 (consent and data handling, `docs/operations/op02_subject_preparation.md`).

**No hardware sync between the systems.** The clock is an affine model fitted to the CT
motion of two sync windows (start in `mand_point_1`, end in the last `free_motion`); the CT
is never one of the compared bodies.

**Comparison in anatomical frames.**
- Each system builds its own maxillary and mandibular frames from its own measurement of
  the same digitized teeth (`*_point_1` incisor, `_2` left, `_3` right second molar).
- The jaw-in-head motion $M = F_{\max}^{-1}\,T_{\mathrm{head}}^{-1}\,T_{\mathrm{jaw}}\,F_{\mathrm{mand}}$
  of the two systems is compared at the lower incisor, the condyle points and in rotation.
- No transform between the systems' head or jaw bodies is fitted on the evaluation data,
  and the world transform cancels in $T_{\mathrm{head}}^{-1}T_{\mathrm{jaw}}$. Constant marker
  offsets (flat disks, spheres) are rigid in the body frame and cancel too.

**Two separate verifications** (VD4):
- **(I) Point marking** (§10): how accurately JTS-MM digitizes the tooth and condyle points,
  and the anatomical frames built from them, with Qualisys as the reference.
- **(II) Motion** (§6–7): the jaw-in-head trajectory agreement after the constant frame
  alignment (VD3), which removes the marking difference on purpose.
- The end-to-end result in each system's own frames (V5) is their combination.

**What is reported alongside:**
- the reference uncertainty, propagated from Qualisys-internal data;
- the error budget;
- robustness to every clock and tip variant;
- a tracking-only decomposition.

**EMG** is compared cross-session (JTS-MM dataset take vs Delsys comparison take of the
same subject), each system normalized to its own maximum (D11; here the MVC).

**Paper wording.** Two participants performed the full recording protocol once while the
wearable system and an optical motion-capture system (Qualisys Oqus, 200 Hz) tracked the <!-- num-ok: instrument setting -->
head plate, the mouthpiece and the calibration tool simultaneously. Each system built its
own maxillary and mandibular coordinate frames from its own digitization of the same incisor
and second-molar points, and the jaw-in-head motion of the two systems was compared in these
frames, so that no transformation between the systems' bodies was fitted on the evaluation
data. Results are reported per participant together with their descriptive mean.

## 2. Stages

| Stage | Reads | Stores | Rule |
|---|---|---|---|
| `ref_sync` | comparison take, QTM `.mat` | `sync`, `sync_methods`, `registration`, `heldout`, `sliding_lag`, `windows`, `clock_variants`, `clock_criteria`; arrays (edges, sliding lags, CT visibility, CT \|ω\|, cross-correlation curves, CT sync paths) | §3 |
| `ref_tip` | `ref_sync` | `tip_estimates`, `tip_decision`, `landmarks` (both systems on the common still window; Qualisys with the primary, shared and own-origin tip), `landmark_distances`, `landmark_symmetry`, `ct_geometry`, `hold_residuals` | §4 |
| `ref_quality` | `ref_sync`, `ref_tip`, QTM calibration report | `calibration`, `consistency`, `viewing`, `marker_geometry`, `label_correction`, `sigma_ref`, `availability`; arrays (per-frame consistency, gate, CT label flags, `gate_at_s`) | §5 |
| `ref_landmarks` | `ref_sync`, `ref_tip`, `ref_quality`, `ref_kinematics` | `landmark_errors`, `landmark_precision`, `landmark_distances`, `frame_errors`, `landmark_sigma_ref`, `marking_summary`, `hold_attribution`, `registration` | §10 |
| `ref_kinematics` | the three above | `segments` (gated and ungated, own frames), `segments_aligned` and `frame_offsets` (VD3), `segments_aligned_freemotion` (VD7), `segments_filtered` (VD12), `robustness`, `budget`, `decomposition`, `attribution`, `tracking_only_*`; per-frame arrays of both systems | §6 |
| `ref_agreement` | `ref_kinematics` | `cycles`, `features`, `waveforms` (own frames) and `cycles_aligned`, `features_aligned`, `waveforms_aligned` (VD3); mean/SD/difference curves | §7 |
| `ref_emg` | `ref_sync`, dataset takes, Delsys CSVs | `mvc`, `mvc_segments`, `activation`, `relevance`, `quality`, `spectra`, `profiles`, `spm_clusters`, `timing_*`, `timing_compare`, `delta_check`, `timeline`, `cycles`; profile arrays | §8a |
| `ref_trueness` (cohort) | ladder recordings | `slots`, `trueness`, `precision`, `summary`, `files`, `notes`; `scale`, `pnp_trueness`, `pnp_precision` (VD28) | §8b, §11h |
| `ref_fusion` | `ref_sync`, `ref_tip`, `ref_quality`, IMU streams (quaternions, accelerometer) | `segments`, `paired`, `dropout`, `tau`, `calibration`, `diagnostics`, `imu_opening` (VD10), `smoothness` (VD16, VD18) | §9, §11, §11a |
| `ref_sampling` | `ref_sync`, `ref_tip`, `ref_kinematics`, `ref_agreement` | `peaks`, `bias`, `peak_ratio_open` (EV7), `spectrum_unfiltered` (EV7, primary), `spectrum` (pre-EV7) | §11 |
| `ref_imu_lever` (cohort), `ref_imu_accel` | hand-eye sessions; `ref_sync` | `lever`, `repeatability` (VD15); `accel`, `calibration` (VD17) | §11a |
| `ref_camerafree` | `ref_sync`, `ref_tip`, `ref_quality` | `cf_val_summary`, `cf_val_segments`, `cf_val_noninferiority`, `cf_val_drift` (VD20) | §11b |
| `ref_hinge` | `ref_kinematics`, `ref_agreement` | `hinge`, `hinge_diff` (VD21) | §11b |
| `ref_budget` (cohort) | `ref_kinematics`, `ref_landmarks`, `ref_quality` | `budget`, `marking_holds`, `marking_causes`, `marking_gap` (VD22) | §11b |
| `ref_emg_side` | `ref_emg` | `side_check` (VD23) | §11b |
| `ref_assoc`, `ref_assoc_cohort` (cohort) | comparison take, dataset take | `cameras`, `counters`, `lag`, `lag_windows`, `recording`, `fault_runs`; `levels`, `limits`, `baseline`, `summary` (VD26) | §11e |
| `ref_cal_repeat` (cohort) | the ten pivot and ten hand-eye runs | `pivot_runs`, `pivot_poses`, `imu_runs`, `summary`, `notes` (VD27) | §11f |

## 3. Clock and camera frame timing (V1, V6, VD19)

### 3.1 Camera frame times

**What and why.** The JTS-MM camera trigger marks the start of the global-shutter exposure;
the pose belongs to the middle of the exposure.

$$
t_S = t_{\mathrm{trig,cam0}} + \tfrac12\,t_{\exp}.
$$

Next to the fitted drift $\alpha$, the stage reports the JTS-MM camera frame-assignment
residual drift from the session's own sync report (`sync:cam_assign_drift_ppm`).

- **Parameters:** `stages.ref_sync.exposure_s` = 0.012 (so +6 ms); `frame_s` = 0.025. <!-- num-ok: method parameter -->
- **Code:** `validation/sync.py:frame_times`; `validation/stages.py:_assign_drift`.

**Paper wording.** Optical frame times of the wearable system were taken at the centre of
the 12-ms exposure.

### 3.2 Time model and sync windows

**What and why.** The two systems run on independent clocks. An affine model (offset plus
rate error) maps master time onto Qualisys time; it is centred on $t_{\mathrm{ref}}$ so that
$c$ and $\alpha$ are nearly uncorrelated.

$$
t_Q = t_S + c + \alpha\,(t_S - t_{\mathrm{ref}}),
\qquad
t_S = \frac{t_Q - c + \alpha\,t_{\mathrm{ref}}}{1+\alpha},
$$

with $t_{\mathrm{ref}}$ the mean time of the CT-visible JTS-MM frames inside the two sync
windows. Stored: `offset_s` $= c$, `alpha_ppm` $= 10^6\alpha$, `tau_s` $= t_{\mathrm{ref}} + c$
(the form $t_Q = (1+\alpha)(t_S - t_{\mathrm{ref}}) + \tau$), `t_ref_s`.

**Sync windows.** The first CT visibility episode (visibility runs separated by less than
2.5 s of invisibility form one episode: on/off bursts plus ≈ 25 s of continuous motion) <!-- num-ok: method parameter -->
inside the first `mand_point_1` phase (`sync_start`) and inside the last `free_motion` phase
(`sync_end`), padded by 0.5 s and clipped to the phase. <!-- num-ok: method parameter -->

- **Parameters:** `reference.yaml: sync.start_phase` = `mand_point_1`, `sync.end_phase` =
  `free_motion`; `stages.ref_sync.sync_pad_s` = 0.5; episode gap 2.5 s (code default). <!-- num-ok: method parameter -->
- **Code:** `validation/sync.py:TimeModel`; `validation/selection.py:sync_windows`.

**Paper wording.** The systems were not hardware-synchronized. Qualisys time was mapped onto
the wearable system's clock by an affine model (offset and clock-rate difference) estimated
from the motion of the calibration tool in two synchronization windows at the start and at
the end of each recording; the calibration tool was never one of the compared bodies.

### 3.3 Coarse estimates: visibility edges and angular-speed correlation

**Edge vote.** Rise (reappearance) and fall (disappearance) times of the CT are the
midpoints between the last invisible and the first visible frame (and vice versa); edges
whose bracketing frames are more than 1.5 frame periods apart are skipped. The coarse lag is
the median of the largest cluster (±`edge_tol_s`) of same-type differences $t_{Q,j} - t_{S,i}$,
searched within `lag_hint_s` ± `hint_range_s` (`sync_methods`, method `edge_vote`).

**Edge least squares.** Over mutually nearest matched edges $k$,

$$
t_{Q,k} - t_{S,k} = c + \alpha\,(t_{S,k} - t_{\mathrm{ref}}) + s_k\,\delta,
\qquad s_k = +1 \text{ (rise)},\ -1 \text{ (fall)},
$$

where $\delta$ absorbs the threshold bias of the two visibility detectors (rises late, falls
early). Ordinary least squares with 3.5 robust-SD trimming ($1.4826\cdot$MAD), 5 re-matching
rounds, $\mathrm{Cov} = s^2(A^\top A)^{-1}$; $\alpha$ only when the edges span > 30 s <!-- num-ok: method parameter -->
(method `edges`).

**Angular-speed cross-correlation.** Per sync window, the CT angular speed of each system over
a fixed difference quotient, squared,

$$
w(t) = \left(\frac{\angle\!\left(R(t+\Delta)\,R(t)^\top\right)}{\Delta}\right)^{2},
$$

on a 1-ms grid with a centred 0.1-s moving mean, and the Pearson correlation
$\rho(L)$ of $w_S(t)$ with $w_Q(t+L)$ over $L$ in the edge lag ± `xcorr_half_range_s`; the lag
is the parabolic peak (method `xcorr_ct_<window>`; a line through the window lags gives
`xcorr_ct`). $|\omega|^2$ is used because white pose noise adds a constant to
$\mathrm{E}|\omega|^2$ that the correlation removes. The approximate SE
$\mathrm{var}(\hat L) \approx (1-\rho_{\max})/(-\rho''\,N_{\mathrm{eff}})$ only screens windows.

- **Parameters:** `stages.ref_sync`: `hint_range_s` = 5.0, `edge_tol_s` = 0.05 (edge LS
  match tolerance $\max(2\cdot0.05, 0.08)$ = 0.1 s), `xcorr_half_range_s` = 0.3, <!-- num-ok: method parameter -->
  `speed_baseline_s` $=\Delta$ = 0.1 (both systems); grid gaps 3.5 frame periods.
- **Code:** `validation/sync.py:visibility_edges`, `coarse_offset_from_edges`, `fit_edges`,
  `speed_series`, `xcorr_lag`, `synchronise`.

### 3.4 Joint fine fit with the registration (primary clock)

**What and why.** The clock and the world registration are fitted together on the CT alone
(V1: the head and jaw bodies are unsafe in the sync windows, where the CT moves next to
them). The Qualisys pose stream is interpolated with a $C^2$ spline per run of valid frames
(cubic for positions, `RotationSpline` for rotations; no gap is bridged; queries within
2.5 ms of a run end are dropped). The residual of JTS-MM frame $i$ in a sync window is <!-- num-ok: method parameter -->

$$
r_i =
\begin{bmatrix}
\big(p(X\,T_{S,\mathrm{CT}}(t_i)\,Y_{\mathrm{CT}}) - p(\hat T_{Q,\mathrm{CT}}(t_i'))\big)/\sigma_p \\[2pt]
\mathrm{Log}\big(R(\hat T_{Q,\mathrm{CT}}(t_i'))^\top R(X\,T_{S,\mathrm{CT}}(t_i)\,Y_{\mathrm{CT}})\big)/\sigma_R
\end{bmatrix},
\qquad
t_i' = t_i + c + \alpha\,(t_i - t_{\mathrm{ref}}),
$$

and $(c, \alpha, X, Y_{\mathrm{CT}})$ minimise $\sum_i \lVert W_i^{1/2} r_i\rVert^2$
(left perturbations $X \leftarrow \mathrm{Exp}(\xi_R)X$, $p \leftarrow p + \xi_p$). Robust IRLS
in up to 5 rounds: round 1 plain least squares behind a loose gate ($3\cdot10\,\sigma$); later
rounds Huber weights $w = \min(1, k/\lVert r\rVert)$ applied separately to the position and the
rotation part, frames beyond $10\,\sigma$ get weight 0 (gross outliers such as a flipped pose
branch); from round 2 the scales are re-estimated per body,
$\sigma \leftarrow \sigma\cdot\mathrm{median}\lVert r\rVert/1.538$ (the median norm of a 3-D
standard normal; floor $0.2\,\sigma_0$). The time columns of the Jacobian are
$\partial r/\partial t'$ (central difference, step 0.2 ms) times $\partial t'/\partial(c,\alpha)$. <!-- num-ok: method parameter -->
Parameter covariance $\Sigma = s^2 (J^\top J)^{-1}$, $s^2 = \sum r^2/(N - P)$ (`se_offset_ms`,
`se_alpha_ppm`, and per transform `se_rot_deg`, `se_trans_mm` in `registration`). The
pre-VD19 autocorrelation-inflated SEs multiply these by
$\sqrt{(1+\bar\rho_1)/(1-\bar\rho_1)}$, $\bar\rho_1$ the lag-1 autocorrelation of the whitened
residuals in time order, clipped to [0, 0.95] (`se_inflation_ar1`, `*_ar1`, `*_ac`).

Start values: the edge or correlation model; $X$ and $Y_{\mathrm{CT}}$ in closed form from
$A_i = X B_i Y$ (rotation of $X$ from RANSAC/Kabsch on paired relative rotations, which satisfy
$\mathrm{Log}(R_{A_i}R_{A_j}^\top) = R_X\,\mathrm{Log}(R_{B_i}R_{B_j}^\top)$; $R_Y$ the robust
chordal mean of $(R_X R_{B_i})^\top R_{A_i}$; translations by linear least squares with 3-SD
trimming). After the fit, the head and jaw registrations are
$Y_b = \operatorname{robust\,mean}_i\big[(X\,T_{S,b}(t_i))^{-1}\,\hat T_{Q,b}(t_i')\big]$ over the
sync-window frames (rotation: chordal mean of the largest cluster within 3°; translation: <!-- num-ok: method parameter -->
median).

- **Parameters:** `stages.ref_sync`: `sigma_pos_mm` = 0.3, `sigma_rot_deg` = 0.15,
  `huber_k` = 2.5, `fit_bodies` = `ct`; gate 10 σ, 5 IRLS rounds (code defaults).
- **Code:** `validation/sync.py:PoseSpline`, `fit_fine`, `synchronise`;
  `validation/registration.py:init_xy`, `y_given_x`, `pose_residuals`, `huber_weights`.

**Paper wording.** Clock offset, clock-rate difference and the rigid transformation between
the two systems' world frames were estimated jointly by robust (Huber-weighted)
least squares, comparing the calibration-tool poses of the wearable system with a spline
interpolation of the Qualisys poses evaluated at the mapped times; position and rotation
residuals were scaled by their own robustly estimated noise levels.

### 3.5 Clock uncertainty (VD19)

**Per-window offsets.** For each sync window $w$, with $X$, $Y$ and the clock fixed, the
residual time shift that best explains the whitened 6-DoF CT residuals is one Gauss–Newton
step,

$$
\hat\delta_w = -\frac{\sum_i j_i^\top r_i}{\sum_i j_i^\top j_i},
\qquad j_i = \frac{\partial r_i}{\partial t'},
$$

with frames beyond $10\,\sigma$ left out. Its SE is a Newey–West (HAC) SE of the scores
$u_i = j_i^\top(r_i + \hat\delta_w j_i)$ in time order,

$$
\mathrm{SE}_{\mathrm{HAC}}(\hat\delta_w) = \frac{\sqrt{\hat V}}{\sum_i j_i^\top j_i},
\qquad
\hat V = \sum_i u_i^2 + 2\sum_{l=1}^{L}\Big(1 - \frac{l}{L+1}\Big)\sum_i u_i\,u_{i-l},
$$

with the Bartlett bandwidth
$L = \max\big(\lceil 1.1447\,(\hat a\,n)^{1/3}\rceil,\ \lfloor 4\,(n/100)^{2/9}\rfloor\big)$,
$\hat a = 4\hat\rho^2/\big((1-\hat\rho)^2(1+\hat\rho)^2\big)$, $\hat\rho$ the lag-1 autocorrelation of
$u$ clipped to [0, 0.97], $L \le n-1$ (Andrews AR(1) plug-in, at least the Newey–West fixed
rule).

**Offset and drift SEs.** Weighted least squares of $\hat\delta_w$ on $t_w - t_{\mathrm{ref}}$
($t_w$ the mean frame time of the window) with weights $1/\mathrm{SE}_w^2$ and no $\chi^2$
rescaling, $\mathrm{Cov} = (A^\top W A)^{-1}$; with two windows exactly

$$
\mathrm{SE}_\alpha = \frac{\sqrt{\mathrm{SE}_1^2 + \mathrm{SE}_2^2}}{t_2 - t_1};
$$

with three or more windows also the delete-one-window jackknife SE (`se_alpha_ppm_jk`).
Stored as `se_offset_ms_hac`, `se_alpha_ppm_hac` for the chosen clock (`sync`) and each
variant (`clock_variants`).

**Position-only window lags** (used by the V6 criteria, §3.7):
$\ell_w = \sum_i e_i\cdot v_i / \sum_i v_i\cdot v_i$, $e_i$ the CT tip position error and $v_i$ the
Qualisys spline velocity; `se_ms_hac` from the scores $u_i = (e_i - \ell_w v_i)\cdot v_i$ as above;
`se_ms_ar1` the white-noise SE $\sqrt{s^2/\sum v\cdot v}$ times the AR(1) factor of the residual
norms.

- **Code:** `validation/sync.py:window_offsets`, `drift_from_windows`, `window_lags`,
  `newey_west_var`, `hac_bandwidth`.

**Paper wording.** The uncertainty of the clock model was obtained from the two
synchronization windows separately: in each window the residual time offset was estimated
from the calibration-tool residuals, its standard error was computed with a
heteroskedasticity- and autocorrelation-consistent (Newey–West) estimator, and the offset and
drift standard errors followed from a weighted line through the window offsets.

### 3.6 Sliding-window lag lines (checks of the affine model)

Across the whole take, in 15-s windows every 5 s, the lag is re-estimated around the model <!-- num-ok: method parameter -->
two ways: (i) the $|\omega|^2$ correlation of the MP and HP (frame-free; a window counts when
$\rho \ge 0.5$, peak ratio ≥ 1.2, the peak is not at the edge of the ±0.1-s search and the
approximate SE ≤ 3 ms); (ii) the registered poses, Gauss–Newton on a shift $\delta$ with <!-- num-ok: method parameter -->
$\delta \mathrel{+}= \sum e\cdot v/\sum v\cdot v$ (a window counts when its SE ≤ 3 ms and the RMS <!-- num-ok: method parameter -->
speed ≥ 5 mm/s). The residual lags are fitted by a weighted line in $t - t_{\mathrm{ref}}$ <!-- num-ok: method parameter -->
(weights $1/\mathrm{SE}^2$, > 4 robust-SD outliers dropped, covariance scaled by
$\max(\chi^2/\mathrm{dof}, 1)$): its slope (ppm) must equal $\alpha$, its intercept ≈ 0; a
quadratic term (curvature = unstable clocks) and jumps > 10 ms between neighbouring windows <!-- num-ok: method parameter -->
(dropped frames) are reported (`sync:sliding_omega_*`, `sliding_pose_*`, `sliding_lag`).

- **Parameters:** `stages.ref_sync.sliding_win_s` = 15.0, `sliding_step_s` = 5.0.
- **Code:** `validation/sync.py:sliding_lags`, `sliding_pose_lags`, `lag_line`.

### 3.7 Clock variants, pre-registered criteria and the choice

**Variants** (`clock_variants`; each later re-scored with the same primary outcome):
- `ct` (primary): the CT-only fit of §3.4;
- `joint`: CT + MP + HP fitted jointly;
- `two_stage`: CT-only, then the MP/HP frames of the sync windows whose position and rotation
  errors under the CT-only solution lie within 3 robust SD of their median, refitted jointly;
- `hybrid`: CT-only clock with the `two_stage` registration;
- `tukey`: CT-only with the Tukey biweight ($c$ = 4.685) instead of Huber;
- `sliding_omega`, `sliding_pose` (VD19 sensitivity, `stages.ref_sync.sliding_variant: both`):
  offset $c + $ intercept and drift = slope of the respective lag line, with the CT-only
  registration. They never choose the clock.

**Criteria** (`clock_criteria`; none uses an agreement outcome):
1. `windows_within_tol`: $|\ell_w| \le 1$ ms in every window;
2. `windows_agree`: $|\ell_1 - \ell_2| \le 2\sqrt{\mathrm{SE}_1^2 + \mathrm{SE}_2^2}$ with the
   AR(1)-inflated window SEs;
3. `sliding_consistent`: $|b_\omega - \alpha| \le 2\sqrt{\mathrm{SE}_{b_\omega}^2 + \mathrm{SE}_\alpha^2}$
   with the fit-covariance SE of $\alpha$;
4. synthetic recovery: `tests/test_v_sync.py`.

`pass` = 1 ∧ 2 ∧ 3. VD19 adds the same checks with the HAC window SEs and the HAC drift SE of
§3.5 (`windows_agree_hac`, `sliding_consistent_hac`, `pass_hac`); these are reported, but the
choice keeps the pre-registered SEs.

**Rule** (`choose_clock`): `ct` if it passes; otherwise `two_stage` if it passes; otherwise
`ct`. In the pinned store (`v15`) no variant passes for P2 (sliding check), so `ct` stays primary. The P2
sliding-window sensitivity was fixed before the run: an erratum only if a P2 headline error
moves by more than 0.3 mm or 0.3° <!-- num-ok: pre-specified criterion -->
(the P2 rows of `robustness` under both sliding variants stay far inside it; the time model
stays).

- **Code:** `validation/sync.py:clock_variants`, `sliding_variant`, `clock_criteria`,
  `choose_clock`; `validation/stages.py:run_ref_sync`.

**Paper wording.** Pre-specified checks of the clock model, none of which used the agreement
outcome, required the residual offset of each synchronization window to stay within 1 ms, <!-- num-ok: pre-specified criterion -->
the two windows to agree within two standard errors, and the drift to agree with the slope of
lags estimated in sliding windows across the recording. Alternative clock estimates
(joint fit of all bodies, two-stage, redescending loss, and the sliding-window time model)
were used as sensitivity analyses.

### 3.8 Held-out validation of the clock and registration

On the held-out movement intervals, per interval and body: the position RMS and 95th
percentile of $p(X T_S Y_b) - p(\hat T_Q)$, the bias per axis, the rotation RMS, and the
residual lag $\sum e\cdot v/\sum v\cdot v$ (≈ 0 when the time model holds); plus the frame-free
jaw-in-head pose (`body` = `MP|HP`), $Y_{\mathrm{HP}}^{-1}T_{S,\mathrm{HP}}^{-1}T_{S,\mathrm{MP}}Y_{\mathrm{MP}}$
against $T_{Q,\mathrm{HP}}^{-1}T_{Q,\mathrm{MP}}$ ($X$ cancels). Pooled over the movement tasks as
$\sqrt{\sum_k n_k\,\mathrm{RMS}_k^2/\sum_k n_k}$ (`sync:heldout_*_rms_*`).

**What the jaw-in-head residual measures, and why it is large for P2.** Each body against
Qualisys agrees closely on the held-out intervals: position RMS at the body origin, mandible
plate / head plate, `\numHeldoutMpPosPOne` / `\numHeldoutHpPosPOne` (2.44 / 2.38 mm) for P1 and
`\numHeldoutMpPosPTwo` / `\numHeldoutHpPosPTwo` (0.58 / 0.73 mm) for P2. The jaw-in-head
position is `\numHeldoutRelPosPOne` / `\numHeldoutRelPosPTwo` (4.08 / 16.67 mm) and its rotation
`\numHeldoutRelRotPOne` / `\numHeldoutRelRotPTwo` (2.33 / 2.17 °) for P1 / P2. The jaw-in-head
position is the mandible-plate *origin* expressed in the *head-plate frame*, so it carries the
orientation difference of the head plate between the systems times the distance between the two
body origins. For P2 the residual is almost entirely a constant x/y offset per interval (RMS ≈
|bias|), and per interval it equals the head-plate rotation residual times a constant
(`ref_sync:heldout`, six intervals: 3.8 to 4.1 mm per degree, i.e. a lever of about 0.2 m; the <!-- num-ok: derived diagnostic ratio, not a paper result -->
recorder's integrity summary of the take puts the two plate origins 0.18 to 0.22 m apart); <!-- num-ok: integrity.json HP_MP dist_mm, diagnostic -->
the head-plate rotation residual pooled over the movement tasks is 1.4° for P1 and 4.1° for P2, <!-- num-ok: pooled in val_table_methods' note -->
while its position stays below 1 mm. <!-- num-ok: pooled in val_table_methods' note -->

*Corrected 2026-11-13 (VD31).* The earlier conclusion "the Qualisys head-plate body is
consistent in every interval (marker deviation below 0.3 mm), so the difference lies on the <!-- num-ok: quoted withdrawn claim -->
JTS-MM side" does not hold: a rigid rotation of a nearly collinear cluster about its long axis
$u_1$ changes the inter-marker distances only in second order, so the consistency check cannot
see it. VD31 splits the difference: about $u_1$ lie `\numRefHeadHeldoutLongAxisPOne` /
`\numRefHeadHeldoutLongAxisPTwo` of `\numRefHeadHeldoutTotalPOne` / `\numRefHeadHeldoutTotalPTwo`
(share of the mean square `\numRefHeadHeldoutLongSharePOne` / `\numRefHeadHeldoutLongSharePTwo`),
mostly constant within each interval (`\numRefHeadHeldoutConstSharePOne` /
`\numRefHeadHeldoutConstSharePTwo` of the mean square; same sign in every interval, size changing
between the P2 intervals). It is 10–20 times what random Qualisys marker noise gives about $u_1$ <!-- num-ok: ratio of the VD31 values quoted beside it -->
(`\numRefHeadMcLongAxisPOne` / `\numRefHeadMcLongAxisPTwo`), so it is a slowly varying bias, not
marker noise. Which system carries it is **not identified**: the weak axis of the JTS-MM head
array (its two tags lie on one baseline, so rotation about the baseline rests on the 30-mm tag <!-- num-ok: tag side, hardware definition -->
side alone) lies within `\numRefHeadAxisAnglePOne` / `\numRefHeadAxisAnglePTwo`° of $u_1$
and carries the same share (`\numRefHeadHeldoutJtsAxisSharePOne` /
`\numRefHeadHeldoutJtsAxisSharePTwo`). Both systems are weakest about nearly the same axis
(the Qualisys flat markers sit on the JTS-MM plate), and P1, whose Qualisys cluster is well
conditioned ($s_2$ = `\numRefHeadSvTwoPOne` mm against `\numRefHeadSvTwoPTwo` mm for P2), shows the
same pattern at a third of the size. So the cause is open on both sides; the VD29 / VD30 pose
noise of the JTS-MM arrays bounds the JTS-MM random part. The maxillary frame error of the
marking is not about this axis (share about $u_1$ `\numMarkFrameMaxLongSharePOne` /
`\numMarkFrameMaxLongSharePTwo`). The time model is not involved: the residual
lag of the mandible plate is about −1 ms in every interval, and the clock's quality is the <!-- num-ok: heldout lag_resid_ms, diagnostic -->
offset SE (§3.5). The jaw-in-head residual is therefore a frame diagnostic of the sync stage's
body registrations, not a measure of the clock or of the motion agreement: the motion outcome
(§6) uses the anatomical frames from the points marked in each system, not $Y$, and P2's incisor
error on the movement tasks stays at a few millimetres, as P1's, not at the jaw-in-head residual's
size (`ref_kinematics:segments` and `segments_aligned`). `val_table_methods` reports the per-body position residuals next to the
jaw-in-head pose and states this in its note (2026-11-05). No stage change is needed for the
table (the per-body columns are stored in `ref_sync:sync`; the pooled head-plate rotation is
computed in the view from `ref_sync:heldout`). The per-frame head-plate residual is recomputed
from the stage inputs by `scripts/audit_recompute/vd31_head_cluster.py` (VD31; reproduces every
stored interval), not stored by `ref_sync`.

- **Code:** `validation/sync.py:evaluate_intervals`, `_err_row`;
  `validation/registration.py:relative_pose_pair`.

## 4. CT tip (V7)

The JTS-MM CT pose is the pivot-calibrated tip (`/meta ct_tip_body_mm`). The Qualisys CT
origin was also moved to a pivot-calibrated tip in QTM; its offset from the marker centroid
is `\numCtOriginCentroidPOne` / `\numCtOriginCentroidPTwo` (200.7 / 200.8 mm, P1 / P2).

**Estimates** of the tip $d$ in the Qualisys CT frame (`tip_estimates`):
- `shared`: the JTS-MM tip is the origin of $T_{S,\mathrm{CT}}$, so with
  $T_{Q,\mathrm{CT}} = X T_{S,\mathrm{CT}} Y_{\mathrm{CT}}$ it lies at
  $d = -R(Y_{\mathrm{CT}})^\top p(Y_{\mathrm{CT}})$; covariance
  $R_Y^\top(\mathrm{SE}_t^2 I)R_Y$ with the inflated translation SE of $Y_{\mathrm{CT}}$.
- `hold_pivot_qualisys`, `hold_pivot_jtsmm`: within each landmark hold $h$ the tip rests on a
  tooth fixed in the body, so the CT pose relative to that body,
  $G_{h,i} = T_{\mathrm{body}}^{-1}T_{\mathrm{CT}}$, satisfies $R(G_{h,i})\,d + p(G_{h,i}) = q_h$;
  linear least squares in $(d, q_1, \dots, q_H)$ over all still frames.
- `tooth_holds` (replaces `cross_system`): per hold the chordal-mean Qualisys CT
  rotation $\bar R_{Q,h}$ and the median of $X p_S - p(T_{Q,\mathrm{CT}})$ over the common still
  frames, then $\bar R_{Q,h}\,d - c = \mathrm{med}_h(X p_S - p_Q)$ by least squares in $(d, c)$
  with a free world offset $c$ (≥ 3 holds); per-hold residuals in `hold_residuals`.
- `distances` (frame-free): the Qualisys landmark is $q_n(d) = \bar R_n d + \bar t_n$ (mean CT
  pose in the body over the still frames); $d$ minimises
  $\sum_{(a,b)}\big(\lVert q_a(d) - q_b(d)\rVert - \lVert p^S_a - p^S_b\rVert\big)^2$ over the
  same-body landmark pairs (Levenberg–Marquardt).
- `pivot_qualisys`: the Qualisys pivot calibration (`reference.yaml: ct_pivot`), covariance
  $\mathrm{rms}^2/n_{\mathrm{poses}}\,I$.

The data-based estimates carry a bootstrap covariance over the holds (the holds are the
sampling units; 400 replicates, at least 3 distinct holds each).

**Decision** (`tip_decision`):
1. if the Qualisys pivot recording used the same QTM body definition as the take (marker
   centroid within 1 mm), the own tip is primary with $d$ = the pivot offset (route a); <!-- num-ok: pre-specified criterion -->
2. else the own tip ($d = 0$) is primary if verified: at least one data-based estimate is
   informative (bootstrap SE < 2 mm on every axis) and every informative one contains <!-- num-ok: pre-specified criterion -->
   $d = 0$ in its 95 % region ($d^\top\Sigma^{-1}d$ against $\chi^2_3$); <!-- num-ok: pre-specified criterion -->
3. else the shared tip. The other tips are sensitivity variants (`robustness`, factor `tip`).

**Applied (V7, route a):** the own tip is verified by the pivot recordings of both systems
(`reference.yaml: ct_pivot`; the same QTM body definition, marker centroids
`\numTipPivotCentroidPOne` / `\numTipPivotCentroidPTwo` (0.01 / 0.05 mm) apart; origin
`\numCtOriginCentroidPOne` / `\numCtOriginCentroidPTwo` (200.7 / 200.8 mm) from the marker centroid;
Qualisys pivot offset d = `\numTipPivotDx`, `\numTipPivotDy`, `\numTipPivotDz` (−0.51, 0.97, 0.37 mm)), so it is primary.
- After EV1 the CT definition re-estimated from correctly labelled take frames matches the
  pivot recording's marker by marker (largest difference `\numTipPivotTemplatePOne` / `\numTipPivotTemplatePTwo` (0.13 / 0.12 mm)),
  and the pivot recording has no swaps.
- The tooth-hold estimate, `\numTipToothDxPOne`, `\numTipToothDyPOne`, `\numTipToothDzPOne` (−1.2, −1.2, 0.2 mm) for P1 and
  `\numTipToothDxPTwo`, `\numTipToothDyPTwo`, `\numTipToothDzPTwo` (−1.3, −1.5, 2.9 mm) for P2, lies within a few
  millimetres of the pivot tip. Its per-hold residuals (P1 `\numTipHoldResidMinPOne` to `\numTipHoldResidMaxPOne` (1.4 to 8.1 mm),
  P2 `\numTipHoldResidMinPTwo` to `\numTipHoldResidMaxPTwo` (1.7 to 9.2 mm)) are hold-specific errors, not a tip offset.

- **Code:** `validation/tip.py:shared_tip`, `hold_pivot`, `tooth_holds`, `distance_tip`,
  `hold_bootstrap`, `verify_own`, `decide`; `validation/evaluate.py:run_ref_tip`,
  `pivot_record`.

**Paper wording.** The calibration-tool tip was pivot-calibrated in both systems; the
Qualisys tip definition was verified against a separate pivot recording with the same rigid-
body definition and, independently, against tip estimates from the landmark holds.

## 5. Reference quality (V8, V9)

### 5.1 Rigid-body consistency and the gate

**What and why.** A rigid marker cluster has constant inter-marker distances; deviations,
large marker residuals and missing or gap-filled markers flag frames where the Qualisys pose
itself is unreliable. Per Qualisys frame and body, with $D_{jk}(t)$ the distance of markers
$j, k$,

$$
\Delta_{jk}(t) = D_{jk}(t) - \mathrm{median}_t\,D_{jk},
\qquad
\Delta_{\max}(t) = \max_{j<k}|\Delta_{jk}(t)|.
$$

**Gate.** A frame is gated when, for HP or MP, any marker is missing (type code 0) or
gap-filled (code 2), $\Delta_{\max} > 0.5$ mm, or a marker residual exceeds 1.0 mm. The gate is <!-- num-ok: method parameter -->
carried to the JTS-MM frames by the nearest Qualisys frame (`gate_at_s`). Results are reported
with and without the gate (`segments.gated`).

**Steps.** Frames where an inter-marker distance jumps between consecutive frames by more than
$\max(5\cdot1.4826\,\mathrm{MAD}, 0.3\ \mathrm{mm})$ (cameras starting or stopping to see a
marker; `consistency.n_steps`).

- **Parameters:** `quality.GATE_DIST_MM` = 0.5, `GATE_RESID_MM` = 1.0 (code constants).
- **Code:** `validation/quality.py:consistency`, `steps`; `validation/evaluate.py:run_ref_quality`.

**Paper wording.** Qualisys frames were flagged when a head- or mouthpiece marker was missing
or gap-filled, an inter-marker distance deviated from its median by more than 0.5 mm, or a <!-- num-ok: pre-specified criterion -->
marker residual exceeded 1 mm; agreement was reported with and without these frames. <!-- num-ok: pre-specified criterion -->

### 5.2 Label correction (EV1)

- **The swap.** The CT marker triangle is nearly isosceles (`ref_tip:ct_geometry`, sides 1–2 / 1–3 / 2–3:
  `\numCtSideOneTwoPOne` / `\numCtSideOneThreePOne` / `\numCtSideTwoThreePOne` (60.1 / 74.4 / 75.9 mm) in P1's take,
  `\numCtSideOneTwoPTwo` / `\numCtSideOneThreePTwo` / `\numCtSideTwoThreePTwo` (60.1 / 74.5 / 76.2 mm) in P2's). QTM
  swaps the labels of markers 1 and 2 in about a quarter to a third of the CT frames (counts below) and still fits
  its body to them, turned by half a turn, which moves the tip by several millimetres (EV1, change log).
- **The correction** (`io.reference.fix_label_swaps`, applied when the take is read):
  - a body is label-ambiguous when a marker permutation changes no template distance by
    > 3 mm; HP and MP are not, so only the CT is corrected; <!-- num-ok: method parameter -->
  - per tracked run, the labelling is decided by the informative frames of the
    inter-marker distances against the measured geometry;
  - distorted runs take the labelling that continues a decided neighbour across a gap
    ≤ 1 s (rotation jump < 45° vs > 135°); runs that stay undecided keep the QTM labels and <!-- num-ok: method parameter -->
    are flagged;
  - swapped frames are refit to the definition (Kabsch).
- **Counts** (`label_correction`), swapped / all / undecided CT frames: P1
  `\numCtSwappedPOne` / `\numCtFramesPOne` / `\numCtUndecidedPOne` (7083 / 26787 / 661), P2
  `\numCtSwappedPTwo` / `\numCtFramesPTwo` / `\numCtUndecidedPTwo` (8128 / 26978 / 113).

**Paper wording.** Label swaps of the nearly isosceles calibration-tool marker triangle were
detected from the inter-marker distances and corrected before analysis.

### 5.3 Viewing direction and marker geometry

**Viewing direction.** Each camera's unit line of sight from the body origin, expressed in the
body frame, as azimuth $\arctan2(v_y, v_x)$ and elevation $\arcsin v_z$, with $\Delta_{\max}$ and
the residual binned in 30° × 30° cells over the movement frames (`viewing`; the signature of <!-- num-ok: method parameter -->
the angle-dependent centroid bias of flat markers).

**Marker geometry** (`marker_geometry`): singular values $s_1 \ge s_2 \ge s_3$ of the centred
body-frame marker template, the triangle sides and the smallest triangle height
$2\cdot\mathrm{area}/\text{longest side}$. A nearly collinear cluster (small $s_2$; P2's head
body: smallest triangle height `\numRefHeadHeightPTwo` (6.4 mm)) determines rotation about its long axis poorly
($\sigma_{\mathrm{rot}} \propto \sigma_m/s_2$).

- **Code:** `validation/quality.py:viewing_directions`, `viewing_bins`, `template`,
  `marker_geometry`.

### 5.4 Marker noise and the reference uncertainty $\sigma_{\mathrm{ref}}$

**Marker noise.** For independent isotropic marker noise $\sigma_m$ per axis, the variance of
an inter-marker distance along the pair is $2\sigma_m^2$, so per body and segment, over the
non-gated frames,

$$
\hat\sigma_m = \frac{\mathrm{SD}\{\Delta_{jk}(t)\}_{j<k,\,t}}{\sqrt2}
\quad(\texttt{sigma\_marker\_mm}).
$$

**Monte Carlo.** Per segment (up to 100 Qualisys frames, evenly subsampled) and draw
$k = 1..200$: Gaussian noise $\mathcal N(0, \hat\sigma_m^2)$ is added to each marker of the HP and
MP templates, both bodies are refit by Kabsch, giving small pose errors $\Delta H_k$,
$\Delta J_k$, and

$$
M_k = F_{\max}^{-1}\,\Delta H_k^{-1}\,(T_{Q,\mathrm{HP}}^{-1}T_{Q,\mathrm{MP}})\,\Delta J_k\,F_{\mathrm{mand}}.
$$

$\sigma_{\mathrm{ref}}$ is the RMS over draws and frames of the 3-D incisor displacement
$\lVert p(M_k) - p(M_0)\rVert$, of the condyle displacement (both condyles pooled) and of the
rotation angle $\angle(R(M_k)R(M_0)^\top)$ (`sigma_ref`: `sigma_ref_incisor_mm`,
`sigma_ref_tmj_mm`, `sigma_ref_rot_deg`), in the Qualisys anatomical frames with the primary
tip.

- **Parameters:** 200 draws, ≤ 100 frames per segment, seed 0 (code defaults).
- **Code:** `validation/quality.py:sigma_marker`, `sigma_ref`, `_kabsch`.

**Paper wording.** The uncertainty of the Qualisys reference was propagated from Qualisys-
internal data only: the marker noise was estimated from the fluctuation of the inter-marker
distances of each rigid body, and its effect on the jaw-in-head outcome was obtained by Monte
Carlo simulation with rigid refits of both bodies.

### 5.5 Availability

Per segment and system: the fraction of frames in which HP and MP are both valid, the number
of dropout runs and their median and maximum duration (`availability`).

- **Code:** `validation/quality.py:availability`.

## 6. Primary outcome (V5), error budget (V9) and frame alignment (VD3)

### 6.1 Per-frame outcome

At the JTS-MM frame times, both systems' jaw-in-head motion in their own anatomical frames,
Qualisys through the chosen clock (spline, no gap bridging):

$$
M_S = F_{\max,S}^{-1}\,T_{S,\mathrm{HP}}^{-1}T_{S,\mathrm{MP}}\,F_{\mathrm{mand},S},
\qquad
M_Q = F_{\max,Q}^{-1}\,\hat T_{Q,\mathrm{HP}}^{-1}\hat T_{Q,\mathrm{MP}}\,F_{\mathrm{mand},Q},
$$

where each $F$ is the maxillary-type frame built from that system's incisor and second-molar
points (origin at the incisor, $y$ from the right to the left molar, $x$ anterior orthogonal to
$y$, $z = x \times y$; an02 §3). Per frame:
- incisor error $e_i = p(M_S) - p(M_Q)$ (maxillary axes);
- condyle errors $e^{\mathrm{tmj}}_i = M_S\,c_S - M_Q\,c_Q$, $c$ the condyle point in the
  system's mandibular frame;
- rotation error $\angle\big(R(M_S)R(M_Q)^\top\big)$;
- rotation vectors since each system's own mean rest pose,
  $\phi_X(t) = \mathrm{Log}\big(R(M_X(t))R(\bar M_{X,\mathrm{rest}})^\top\big)$ (chordal mean).

- **Code:** `validation/anatomy.py:anatomical_frames`, `jaw_in_head`, `points_of`,
  `rotation_error_deg`; `validation/evaluate.py:outcome`, `rot_rel_rest`.

### 6.2 Per-segment agreement

Per segment, with and without the gate (`segments`, column `gated`), over the frames where
both incisors and the rotation error are finite:
- per-axis bias $\bar e_a$ and SD $s_a$ (ddof 1);
- 3-D RMS $\sqrt{\overline{\lVert e\rVert^2}}$ (`incisor_rms_mm`);
- bias-removed RMS $\sqrt{s_x^2+s_y^2+s_z^2}$ (`incisor_dyn_rms_mm`);
- rotation RMS of the geodesic error (`rot_rms_deg`); bias-removed rotation RMS
  $\sqrt{\sum_a \mathrm{var}(\phi_{S,a}-\phi_{Q,a})}$ (`rot_dyn_rms_deg`) and the rotation bias
  per axis;
- condyle RMS and bias-removed RMS per side (`tmj_left_*`, `tmj_right_*`);
- incisor range of motion $\lVert P_{95}(p_Q) - P_5(p_Q)\rVert$ (per-axis percentiles) and the
  RMS as % of it (`incisor_rel_pct`).

Movement tasks are also pooled (`movement_pooled`); holds and sync windows are reported,
never pooled.

- **Code:** `validation/evaluate.py:segment_stats`, `segment_list`, `run_ref_kinematics`.

**Paper wording.** For every frame of the wearable system, the lower-incisor position, the
condyle points and the mandibular orientation in the maxillary frame were compared with the
time-mapped Qualisys values. Per task we report the bias, the root-mean-square (RMS) error,
the bias-removed RMS error and the rotation error, each with and without the flagged
reference frames.

### 6.3 Error budget per segment (V9)

**What and why.** If the reference noise and the JTS-MM error are independent, the observed
disagreement splits in quadrature. Per gated segment and outcome (incisor: `incisor_dyn_rms_mm`
vs `sigma_ref_incisor_mm`; rotation: `rot_dyn_rms_deg` vs `sigma_ref_rot_deg`; condyle:
`tmj_dyn_rms_mm` vs `sigma_ref_tmj_mm`, where both condyles are pooled on both sides,
$\sigma_{\mathrm{obs}}^{\mathrm{TMJ}} = \sqrt{\tfrac12\big(\sigma_{L}^2 + \sigma_{R}^2\big)}$ with
$\sigma_{L,R}$ = `tmj_{left,right}_dyn_rms_mm` (one side alone when the other is missing), like
$\sigma_{\mathrm{ref}}$ of §5.4; EV13: before, the left condyle alone was compared
with the pooled $\sigma_{\mathrm{ref}}$),

$$
\sigma_{\mathrm{obs}}^2 = \sigma_{\mathrm{JTS}}^2 + \sigma_{\mathrm{ref}}^2,
\qquad
\hat\sigma_{\mathrm{JTS}} =
\begin{cases}
\sqrt{\sigma_{\mathrm{obs}}^2 - \sigma_{\mathrm{ref}}^2} & \sigma_{\mathrm{ref}} < \sigma_{\mathrm{obs}} \ (\text{separable}),\\
\text{NaN} & \text{otherwise (not resolvable)},
\end{cases}
$$

`ratio` $=\sigma_{\mathrm{obs}}/\sigma_{\mathrm{ref}}$, `adequate` = ratio ≥ 3 (`budget`). §11b adds
the share $\sigma_{\mathrm{ref}}^2/\sigma_{\mathrm{obs}}^2$ and the marking rows.

- **Parameters:** ratio 3 (`quality.budget` default; `stages.ref_budget.adequacy_ratio` = 3.0
  for the cohort table).
- **Code:** `validation/quality.py:budget`; `validation/evaluate.py:run_ref_kinematics`.

### 6.4 Frame alignment (VD3; of equal value with the own-frame result, VD5)

**Model.** The two systems' anatomical frames differ by constant rigid offsets, because their
markings of the same teeth differ:

$$
M_Q(t) \approx A\,M_S(t)\,B,
$$

$A$ maxillary, $B$ mandibular. **Leave-one-task-out:** for each movement task $h$, $A_h$ and
$B_h$ are fitted on the gated-clean frames of the *other* movement tasks (≥ 50 frames): closed
form (`init_xy`, §3.4) then robust refinement

$$
(A_h, B_h) = \arg\min \sum_i \big\lVert W_i^{1/2}\,r(A\,M_{S,i}\,B,\ M_{Q,i})\big\rVert^2
$$

with the whitened 6-DoF residual of §3.4 ($\sigma_p$ = 0.3 mm, $\sigma_R$ = 0.15°, Huber <!-- num-ok: method parameter -->
$k$ = 2.5, 4 IRLS rounds), and applied to task $h$, so a task never shapes its own alignment.
After alignment both systems live in the Qualisys anatomical frames: the condyle points are
those of the Qualisys mandibular frame carried by both, and the rotation vectors refer to one
shared Qualisys rest pose. The segment statistics of §6.2 are then recomputed
(`segments_aligned`). Fold stability and the in-sample fit are in `frame_offsets` (rotation
angle and translation norm of $A$ and $B$); the same offsets follow independently from the
marked points (§10).

**What it removes.** The alignment removes the marking difference, so it measures motion
tracking only. The marking itself is verified in §10.

**Sensitivity (VD7).** $A$, $B$ fitted on `free_motion` alone, a fixed calibration segment that
is never scored, and applied to the four protocol tasks (`segments_aligned_freemotion`, pooled
as `protocol_pooled`).

- **Code:** `validation/registration.py:init_xy`, `register_paired`;
  `validation/evaluate.py:run_ref_kinematics`, `aligned_pooled`.

**Paper wording.** Because the two systems' anatomical frames differ by the small, constant
difference of their tooth digitizations, motion tracking was additionally assessed after a
constant rigid alignment of the maxillary and mandibular frames. The alignment was estimated
by robust least squares on the other movement tasks and applied to the held-out task
(leave-one-task-out), so no task contributed to its own alignment; the digitization itself was
assessed separately.

### 6.5 Robustness, decomposition and attribution

**Robustness** (`robustness`): the pooled gated movement outcome (incisor RMS and bias-removed
RMS, rotation RMS, and the frame-aligned incisor and rotation RMS with the alignment refitted
leave-one-task-out on that outcome) under every clock variant (§3.7, factor `clock`) and the
shared and own-origin Qualisys tips (§4, factor `tip`).

**Decomposition** (`decomposition`, `tracking_only_*`): end-to-end (own anatomical frames)
against tracking-only, where the Qualisys jaw-in-head pose is mapped into the JTS-MM body
frames, $Y_{\mathrm{HP}}\,T_{Q,\mathrm{HP}}^{-1}T_{Q,\mathrm{MP}}\,Y_{\mathrm{MP}}^{-1}$, with
$Y_b$ fitted leave-one-task-out (`y_given_x` on the other movement tasks) and both expressed in
the JTS-MM maxillary frame at the JTS-MM landmarks.

**Attribution** (`attribution`, descriptive): Spearman correlation over the pooled movement
frames of the per-frame incisor error with the Qualisys $\Delta_{\max}$, CT visibility, the
JTS-MM tag count and reprojection RMS of HP and MP, and the Qualisys incisor speed.

- **Code:** `validation/evaluate.py:run_ref_kinematics`, `aligned_pooled`;
  `validation/kinematics.py:run_ref_kinematics` (tracking-only).

## 7. Cycle agreement (§5 of the validation plan)

### 7.1 Cycles and features

**Cycles** are found on the *Qualisys* incisor signal of each held-out movement task (opening
$-z$ in `open_close` and `chewing`, protrusion $+x$, laterotrusion $\pm y$ per side) with the
dataset's rule (`kinematics.cycles.detect`: peaks with prominence ≥ 0.3 of the 5–95 % range, <!-- num-ok: method parameter -->
≥ 0.4 s apart; start and end at the minima between peaks; open-close and chewing flagged when <!-- num-ok: method parameter -->
the start-to-end difference exceeds 20 % of the amplitude, EV10). Only `ok` cycles complete <!-- num-ok: method parameter -->
(finite) in both systems and with no gated frame are kept, and the same windows are cut from
both systems.

**Features** per cycle $[a, b]$ with signal $s(t) = \pm p_{\mathrm{inc},\mathrm{axis}}(t)$, both
systems:
- amplitude $\max_t s(t) - s(a)$ (`amp_mm`);
- incisor excursion $\max_t\lVert p_{\mathrm{inc}}(t) - p_{\mathrm{inc}}(a)\rVert$ (`excursion_mm`);
- peak speed $\max_t\lVert\Delta p_{\mathrm{inc}}\rVert/\Delta t$ over first differences at 40 Hz <!-- num-ok: method parameter -->
  (`peak_speed_mm_s`);
- rotation excursion $\max_t\lVert\phi(t) - \phi(a)\rVert$ (`rot_excursion_deg`);
- condyle excursion, the larger of the two sides (`tmj_excursion_mm`).

- **Parameters:** `stages.kinematics.cycles` (dataset): `min_period_s` = 0.4,
  `prominence_frac` = 0.3, `max_nan_frac` = 0.1, `return_frac` = 0.2 (`find_cycles` uses the
  same values as its defaults).
- **Code:** `validation/agreement.py:find_cycles`, `cycle_features`, `run_ref_agreement`.

**Paper wording.** Movement cycles were detected on the Qualisys signal with the cycle rule of
the dataset and cut from both systems; per cycle we computed the amplitude, the incisor
excursion, the peak incisal speed, the rotation excursion and the condyle excursion.

### 7.2 Feature agreement per participant and task

With $d_j$ = JTS-MM minus Qualisys of cycle $j$ (in time order of the cycle starts),
$\bar d$ and $s$ = SD (ddof 1):

**Bland–Altman.** Bias $\bar d$, limits of agreement $\bar d \pm 1.96\,s$; LoA CIs from the
cycle percentile bootstrap (2000 replicates, $\bar d^* \pm 1.96\,s^*$).

**Bias CI (EV6, primary).** A $t$-interval with the effective sample size of the
autocorrelated cycle differences, $\bar d \pm t_{0.975,\nu}\,s/\sqrt{n_{\mathrm{eff}}}$ with
$n_{\mathrm{eff}} = n(1-\hat\rho_1)/(1+\hat\rho_1)$, $\hat\rho_1$ the lag-1 autocorrelation
clipped to $[0, 0.99]$ and $\nu = \max(n_{\mathrm{eff}} - 1, 1)$; formulas in
[an07](an07_statistics.md) §17.1 (columns `bias_ci_lo/hi`, `rho1`, `n_eff`, `t_df`).
**Sensitivities:** the independent-cycle percentile bootstrap (the pre-EV6 interval, `*_boot`)
and the moving-block bootstrap with overlapping blocks of 2 successive cycles (`*_mbb`).

**Concordance and reliability** (formulas in an07 §17.3): Lin's CCC with $1/n$ moments (EV9,
implemented 2026-10-31; before, $n-1$ moments), ICC(A,1) (McGraw & Wong, two-way, absolute
agreement, single measurement; $k$ = 2 systems) and Pearson $r$. The $n-1$ CCC value is not
stored: the old → new comparison of the validation re-run documents the difference.
`tests/test_r10_reference_impl.py` checks the formula, including a worked example
($\rho_c$ = 5/6).

**Equivalence.** For features in mm or deg, the bias CI must lie strictly inside $\pm\Delta$:
metrology $\Delta$ = 0.5 mm / 0.5°, application $\Delta$ = 2.0 mm / 1.2° (from the dataset <!-- num-ok: pre-specified criterion -->
requirements as frozen; `agreement.TOL`) (`equivalent_metrology`,
`equivalent_application`). The bound-free **equivalence margin** (VD13) is the smallest
symmetric bound containing the CI, $\max(|\mathrm{lo}|, |\mathrm{hi}|)$ (`equiv_margin`; the
EV6 sensitivities `equiv_margin_boot`, `equiv_margin_mbb`). The verdict changes that EV6 caused
are listed in the change log of the validation plan (entry of 2026-10-29, the results of EV6), and
the pre-specified rule is kept.

- **Parameters:** 2000 bootstrap replicates, block 2 cycles, seed = study `seed` (42).
- **Code:** `validation/agreement.py:ba_boot`, `n_effective`, `bias_t_interval`,
  `moving_block_means`, `ccc`, `icc_a1`, `margin`.

**Paper wording.** Per participant and task, the agreement of the cycle features was assessed
with Bland–Altman analysis, Lin's concordance correlation coefficient and the intraclass
correlation ICC(A,1). Because successive cycles are not independent, the 95 % confidence <!-- num-ok: definition -->
interval of the bias was a t-interval based on the effective number of cycles derived from the
lag-1 autocorrelation of the differences; a cycle bootstrap and a moving-block bootstrap
served as sensitivity analyses. Equivalence was declared when the confidence interval of the
bias lay within ±0.5 mm / ±0.5° (metrological bound) or within ±2.0 mm / ±1.2° (application <!-- num-ok: pre-specified criterion -->
bound derived from the dataset requirements); in addition we report the smallest symmetric
bound that contains the interval.

### 7.3 Waveforms

Per task and channel (incisor $x$, $y$, $z$; rotation-vector components $x$, $y$, $z$), each
cycle linearly time-normalized to 101 samples, as absolute values (`abs`, the frame-definition
offset included) and as displacement from the cycle start (`disp`, node 0 is zero by
construction and is excluded):
- **paired SPM $t$-test** JTS-MM vs Qualisys over cycles (spm1d, $\alpha$ = 0.05, two-tailed;
  ≥ 5 cycles), cluster ranges stored as `"[(start %, end %, p)]"` (significance bars);
- **mean difference** $\bar D(\tau)$ with a 95 % band that is, at every node, the same <!-- num-ok: definition -->
  effective-sample-size $t$-interval as the cycle-feature bias (§7.1; EV14): with
  the $n$ cycle differences $D_i(\tau)$ in time order, lag-1 autocorrelation
  $\hat\rho_1(\tau)$ clipped to $[0, 0.99]$ and
  $n_{\mathrm{eff}}(\tau) = n\,\frac{1-\hat\rho_1(\tau)}{1+\hat\rho_1(\tau)}$,

  $$
  \bar D(\tau) \pm t_{0.975,\,\max(n_{\mathrm{eff}}-1,\,1)}\,
  \frac{s_D(\tau)}{\sqrt{n_{\mathrm{eff}}(\tau)}}
  $$

  (`pointwise_t_band`; a node without spread gets $\bar D$ itself). The independent-cycle
  bootstrap band used before EV14 (2000 replicates) stays as the sensitivity
  `equiv_margin_boot`. `mean_diff_rms` $=\sqrt{\overline{\bar D(\tau)^2}}$;
- **share of the cycle inside each bound**: fraction of nodes whose band lies inside $\pm\Delta$
  (`frac_inside_metrology`, `frac_inside_application`); the band's equivalence margin
  $\max_\tau\max(|\mathrm{lo}(\tau)|, |\mathrm{hi}(\tau)|)$ (VD13);
- **CMC between the systems** (Kadaba, an07) per cycle on the pair of curves, median over
  cycles (`cmc_median`).

**n = 2:** participant is a fixed factor; nothing is pooled across participants.

- **Code:** `validation/agreement.py:run_ref_agreement`, `pointwise_t_band`, `normalise`, `cmc_pair`;
  `biomech/stats/waveforms.py:paired_t`, `cmc`.

**Paper wording.** Time-normalized cycle waveforms of the two systems were compared with
paired statistical parametric mapping t-tests; we also report the mean difference curve with
a pointwise 95 % confidence band (t-interval with the effective number of cycles, accounting <!-- num-ok: definition -->
for the correlation of successive cycles), the fraction of the cycle within each bound, and
the coefficient of multiple correlation between the systems.

**Limitation (state it).** The SPM test is a parametric paired t-test that treats the cycles as
independent; successive cycles are correlated (the reason for EV6 / EV14), so the SPM clusters
are anti-conservative (too many). Read them as the location of systematic differences, with the
effective-n band (EV14) as the inference; there is no n_eff correction inside SPM.

## 8. EMG and bench metrology (sections (III) and (IV) of the paper)

### 8a. EMG comparability (D5, stage `ref_emg`)

**Design.** Cross-session: the JTS-MM dataset take and the Delsys comparison take of the same
subject (the comparison takes carry no usable JTS-MM EMG). Interval between the two sessions:
P1 7 days, P2 1 day. Differences therefore include
electrode type and placement, day-to-day variation and hardware chain; the comparison shows
method comparability of normalized shapes, levels, onset distributions and signal quality,
not agreement of the same contraction.

**Delsys timeline.** The Trigno base triggers QTM: $t_Q = t_D - \delta$, and QTM time maps to the
master clock through the `ref_sync` model, so

$$
t_S = \frac{(t_D - \delta) - c + \alpha\,t_{\mathrm{ref}}}{1+\alpha}.
$$

$\delta$ (QTM start latency; candidates 0 and the hidden QTM trigger event at 0.74 s) is chosen <!-- num-ok: method parameter -->
by the pre-stated plausibility test `delta_check`: per task window (`open_close`, `chewing`)
and muscle group, the group-mean envelope (each channel scaled by its window median, 100-Hz
grid, $\delta$ = 0 assumed) is correlated with the Qualisys jaw speed and angle at lags
$L \in [-1.5, 1.5]$ s; a candidate implies the lead $L_{\mathrm{peak}} + \delta$; the candidate
whose implied lead lies in [0, 250] ms in most *opener* correlations wins (ties: higher mean
windowed peak $r$; no vote: inconclusive, $\delta$ = 0). Clock-rate differences between Trigno
and QTM are not modelled. Stored: `delta_check`, `timeline`.

**Front ends as used (2026-11-13).** Delsys: Trigno Avanti sensors (four parallel
silver bars, 10 mm inter-electrode distance, Delsys specification), sensor modes 84 (4370 Hz) <!-- num-ok: hardware specification -->
and 40 (2148 Hz, `Dig_Right`) per the CSV headers, both with the hardware band 20–450 Hz <!-- num-ok: hardware specification -->
(the authors, from the Trigno Discover sensor-mode settings; the band is not written into the files).
JTS-MM: Ambu BlueSensor P pre-gelled Ag/AgCl snap electrodes, the two of a channel next to each
other along the fibres (about 20–30 mm centre to centre, photographs; op02), ADS1299 at 4 kSPS, <!-- num-ok: hardware specification -->
gain 6, DC-coupled. Both then pass the common band-pass `\numEmgBandLow`–`\numEmgBandHigh` Hz.

**Resampling.** Each Delsys channel (4370 Hz EMG sensors, one sensor at 2148 Hz) is put on one <!-- num-ok: method parameter -->
uniform grid at the JTS-MM EMG rate: zero-phase 8th-order Butterworth low-pass at
$0.45\,f_{\mathrm{out}}$ when downsampling, then cubic-spline interpolation (flat to ≪ 1 % up to <!-- num-ok: method parameter -->
450 Hz; linear interpolation of the 2148-Hz sensor would cost 1.3 dB at 450 Hz). <!-- num-ok: method parameter -->

**Same chain for both systems.** Both systems go through the unchanged dataset EMG code
(`EmgPrep`: mains fit, 20–450 Hz zero-phase band-pass, analysis masks, masked 150-ms RMS <!-- num-ok: method parameter -->
envelope, D11 reference = largest activity recorded over the MVC phases and the tasks; an03 §2.7).
Each Delsys MVC file (opening then closing effort) is split where the active group changes: with
$\hat E_c(t)$ the smoothed (1 s) envelope of channel $c$ scaled to its file maximum and <!-- num-ok: method parameter -->
$g(t) = \overline{\hat E}_{\mathrm{openers}} - \overline{\hat E}_{\mathrm{closers}}$, the split is the
extremum of $\sum_{t'\le t} g(t')$ (maximum for opening first, minimum for closing first). With
several MVC files the per-channel reference comes from the file with the largest MVC activity.

- **Parameters:** `validation/emg.py:DEFAULTS` (no study overrides): `delta_candidates_s` =
  [0, 0.74], `xcorr_half_range_s` = 1.5, `xcorr_grid_hz` = 100, `lead_window_s` = [0, 0.25],
  `speed_baseline_s` = 0.05, `full_scale_uv` = 11000, `ct_guard_s` = 0.5, `mvc_smooth_s` = 1.0;
  the dataset's `signals.emg`, `stages.emg.cycle_mask_max` = 0.10, `stages.timing`,
  `stages.qc.usability`, `stages.emg_relevance`.
- **Code:** `validation/emg.py:delta_check`, `qualisys_jaw`, `resample_channel`,
  `delsys_take`, `segment_mvc`, `MultiMvcPrep`, `prepare_sides`.

**Paper wording.** Because only the reference electrodes could be placed during the comparison
recordings, EMG was compared across sessions: the wearable system's recording of the same
participant during the dataset protocol against the Delsys recording of the comparison session.
Delsys signals were resampled to the wearable system's sampling rate and both systems were
processed with the identical pipeline and normalized to their own maximal voluntary
contraction. The Delsys clock was linked to Qualisys through the trigger and a pre-specified
plausibility test of the start latency, and to the wearable system through the clock model.

**Cycle profiles.** Per system, cycle and channel, the normalized envelope (% of that system's
own maximum) between cycle start and end, time-normalized to 101 points; a channel-cycle with a
masked share above `cycle_mask_max` is NaN and dropped. JTS-MM cycles come from the dataset
take's own kinematic cycles, Delsys cycles from the comparison take's JTS-MM optics (same
detection; optical times +6 ms). Channels are paired **by name** (JTS-MM montage of an03 §1 <!-- num-ok: method parameter -->
against the side-labelled Delsys sensors), so the side check VD23 (§11b) bears on every
per-side value.

**Profile agreement** (`profiles`, per participant, profile key = task (and movement side in
`left_right`), muscle; ≥ 2 cycles per system), with $\bar y_J(\tau)$, $\bar y_D(\tau)$ the
ensemble means and $D(\tau) = \bar y_D(\tau) - \bar y_J(\tau)$ (percentage points of each
system's own maximum, Delsys minus JTS-MM):
- shape correlation `r` = Pearson $r(\bar y_J, \bar y_D)$ over the 101 points (level- and
  gain-free);
- between-system CMC of the two mean curves (Kadaba): with $m$ = 2 curves of $T$ = 101 points,
  $\mathrm{CMC} = \sqrt{1 - \dfrac{\sum_{c,\tau}(y_{c\tau}-\bar y_\tau)^2/(T(m-1))}{\sum_{c,\tau}(y_{c\tau}-\bar y)^2/(mT-1)}}$,
  undefined when the radicand is negative (`cmc_defined` = false; §11);
- `rms_diff_pct` $=\sqrt{\overline{D(\tau)^2}}$, `mean_diff_pct` $=\overline{D(\tau)}$, and the
  within-system CMCs;
- two-sample SPM $t$-test (Welch, unequal variances; cycles as repetitions, ≥ 3 per system;
  $\alpha$ = 0.05): `spm_h0reject`, `spm_zstar`, clusters in `spm_clusters`.

- **Code:** `validation/emg.py:cycle_profiles`, `compare_profiles`, `profile_r`,
  `ttest2_spm`; `biomech/stats/waveforms.py:cmc`.

**Paper wording.** For each muscle and task, the ensemble-mean activation profiles of the two
systems (time-normalized cycles, in percent of each system's own maximum) were compared by the
Pearson correlation (shape), the coefficient of multiple correlation, the RMS and mean
difference in percentage points, and a two-sample statistical parametric mapping t-test.

**Timing comparison** (`timing_*`, `timing_compare`). Both systems go through the dataset's
onset analysis (`timing.lead.analyse`: TKEO, Hodges and envelope detectors, `first` and `drive`
rules, the event-aligned ensemble and the release detector; an05). The mechanical onset is the
optical velocity onset for both systems: the optical opening angle (40-Hz frames shifted +6 ms) <!-- num-ok: method parameter -->
is interpolated onto a 200-Hz grid (gaps ≤ 0.1 s) and differentiated there (`mech_arrays`); for <!-- num-ok: method parameter -->
Delsys the Qualisys opening angle (rotation vector projected on the principal axis of the
open-close motion) is a second source. Cross-session, so no per-cycle pairing: per task,
direction and source, the median lead of each system, their IQRs and the difference of medians
$\Delta = \mathrm{med}(\ell_D) - \mathrm{med}(\ell_J)$ with a percentile bootstrap CI (cycles
resampled within each system, 2000 replicates). `timing_compare` uses the TKEO detector and the
primary rule (`stages.timing.onset_rule` = `drive`); the onsets and summaries of every detector
and both rules are in `timing_onsets`, `timing_summary`.

- **Code:** `validation/emg.py:mech_arrays`, `timing_tables`, `lead_comparison`.

**Paper wording.** EMG-to-motion lead times were computed with the onset detectors of the
dataset analysis, using the optical opening-velocity onset as the mechanical event in both
systems, and the two systems' lead distributions were compared by the difference of their
medians with a bootstrap 95 % confidence interval. <!-- num-ok: definition -->

**Signal quality, spectra, levels** (`quality`, `spectra`, `mvc`, `activation`): the dataset's
EMG quality table (rest noise, SNR = MVC p95 / rest envelope, mains share, mask shares,
usability flags), MNF/MDF and band shares, MVC levels per file, and task activation per system.

### 8b. Bench ladder: trueness, precision, accuracy (D6, stage `ref_trueness`)

**Data.** Qualisys: the `EvalArraz` rigid body in the QTM exports
`trueness-{orientation}-{distance}-d{mm}.mat` and `precision-{distance}.mat` (the stray
`trueness-lateral-far-d2000015` ignored). JTS-MM: capture sessions (`session.h5`) of the array;
the array position is the centroid of all corners of tags 35 and 41 (`markers_3d`; only frames
with every corner), the slots are the still plateaus (rolling 1-s SD ≤ 0.25 mm on every axis <!-- num-ok: method parameter -->
for ≥ 3 s; neighbours closer than 5 mm merged), projected on the first principal axis of the <!-- num-ok: method parameter -->
plateau means and mapped to the 20-mm grid from the first slot.

**Static samples.** Per slot take the central 80 % (D0b); slot mean $\bar p_k$ with nominal <!-- num-ok: method parameter -->
(hand-measured) position $s_k$.

**Metric line.** $\bar p_k \approx c + s_k\,u$ with $\lVert u\rVert = 1$ and the scale fixed to the
ruler (direction free, so a rigid tilt of the ladder is absorbed); the least-squares solution is

$$
u = \frac{\sum_k (s_k - \bar s)\,\bar p_k}{\big\lVert\sum_k (s_k - \bar s)\,\bar p_k\big\rVert},
\qquad c = \bar{\bar p} - \bar s\,u .
$$

**Trueness** = mean step error with a $t$-CI over the steps (consecutive steps share a slot, so
the CI is approximate):

$$
\varepsilon_k = \lVert\bar p_{k+1} - \bar p_k\rVert - (s_{k+1} - s_k),
\qquad
\text{trueness} = \bar\varepsilon \pm t_{0.975,\,K-1}\,\frac{s_\varepsilon}{\sqrt K}.
$$

Also: line residuals $\bar p_k - (c + s_k u)$ (RMS, max; `slot_mean_resid_sd_mm` with $S-2$ dof),
the spacing error from the reference slot $\lVert\bar p_k - \bar p_0\rVert - |s_k - s_0|$, and the
**scale error** of a free-scale fit, $100\,(\beta - 1)$ % with $\beta$ the OLS slope of
$(\bar p_k - \bar{\bar p})\cdot u$ on $s_k$.

**Static precision** (repeatability) pooled over slots, per axis $a$:
$\sigma_a = \sqrt{\sum_k (n_k - 1)\,s_{k,a}^2 / \sum_k (n_k - 1)}$, and
$\sigma_{3\mathrm{D}} = \sqrt{\sigma_x^2 + \sigma_y^2 + \sigma_z^2}$.

**Accuracy** = RMS of the total error of every static sample against its nominal position on
the line, $\sqrt{\overline{\lVert p_{k,i} - (c + s_k u)\rVert^2}}$ (trueness and precision together).

**30-s takes and drift.** Per take a per-axis linear trend $p_a(t) = \beta_{0a} + \beta_{1a}(t - \bar t)$
is removed; precision = SD of the residuals (ddof 2), 3-D $\sqrt{\sum_a\sigma_a^2}$, next to the raw
SD; drift $60\,\lVert\beta_1\rVert$ mm/min and $\lVert\beta_1\rVert\,T$ over the take (`precision`).

- **Parameters:** `validation/trueness.py:DEFAULTS`: `static_central` = 0.8,
  `precision_central` = 1.0, `ci` = 0.95, `reference_mm` = 0, `qualisys_body` = `EvalArraz`;
  `reference.yaml`: `qualisys_ladder_dir`, `jtsmm_ladder_dir`; plateau rule 1 s / 0.25 mm / 3 s / <!-- num-ok: method parameter -->
  20 mm (code defaults). <!-- num-ok: method parameter -->
- **Code:** `validation/trueness.py:metric_line`, `t_ci`, `ladder_metrics`,
  `precision_metrics`, `summary_table`, `plateaus`, `array_centroid`, `jtsmm_session_ladder`.

**Paper wording.** Trueness, precision and accuracy (ISO 5725) were measured on a
hand-measured 20-mm step ladder at near and far distances in frontal and lateral orientation.
Trueness was the mean deviation of the measured from the nominal step length, with the slot
positions fitted to a line of fixed metric scale; precision was the pooled within-slot standard
deviation of the static samples and, in 30-s static recordings, the standard deviation after
removal of a linear drift, which was reported separately; accuracy was the RMS distance of all
static samples from their nominal positions.

## 9. Optical + IMU fusion (V4, VD6; secondary result, VD8)

`jtsmm/imu/fusion.py`, `jtsmm/imu/position.py` and `biomech/validation/fusion.py`.

**Estimators:**

| Estimator | Orientation | Position |
|---|---|---|
| F0 | optical, SLERP between valid frames (F0_hold: last valid frame) | optical, linear between frames |
| F1 | SO(3) complementary filter (causal) | optical, linear |
| F2 / F2_causal | error-state Kalman filter with RTS smoother / forward only | optical, linear |
| F2P / F2P_causal (VD6) | F2 / F2_causal | accelerometer-fused (Kalman + RTS / forward) |
| F1_precal / F2_precal | as F1 / F2, with the IMU mounting fixed from the hand-eye calibration | optical, linear |
| I (VD10) | IMU alone through the calibration, no optical correction | optical, linear |

The `_precal` estimators only run when `reference.yaml: imu_mounting` is set, or when the take
embeds a mounting.

### 9.1 Calibration (per leave-one-task-out fold, JTS-MM optical only)

**Model** per body, with $R_{\mathrm{opt}}$ the optical rotation (camera ← marker body) and
$R_{\mathrm{imu}}$ the chip's game rotation vector (IMU world ← chip):

$$
R_{\mathrm{opt}}(t) = \mathrm{Exp}\big(w\,(t - t_{\mathrm{ref}})\big)\;Z\;R_{\mathrm{imu}}(t + \lambda)\;X,
$$

$Z$ camera ← IMU world, $X$ chip ← marker body (mounting), $\lambda$ the IMU lag, $w$ the slow
drift of the IMU world (mostly heading). Fit on the training frames only (the sync windows plus
the other movement tasks; disjointness is asserted): $X$ from body-frame relative rotations of
nearby frame pairs (RANSAC Kabsch; the heading drift cancels), $Z$ the robust chordal mean of
$R_{\mathrm{opt}}X^\top R_{\mathrm{imu}}^\top$, a coarse lag scan (2-ms grid, truncated mean square),
then robust Gauss–Newton (Huber on the residual angle, hard rejection beyond
$\max(5\cdot\mathrm{median}, 1°)$) over $Z$, $X$, $w$ (only when the training span ≥ 60 s) and <!-- num-ok: method parameter -->
$\lambda$ on the residual $\mathrm{Log}(R_{\mathrm{opt}}R_{\mathrm{pred}}^\top)$. The lag is pooled
over the two IMUs by inverse-variance weighting and the calibration refitted at it. `_precal`:
$X = R_{\mathrm{mount}}^\top$ fixed.

**Settings from the sensors** (never tuned against the reference):
- optical rotation noise: the structure function of the calibration residual on the training
  frames, $D(\Delta) = 2\sigma^2 + q\Delta + c\Delta^2$ per axis (robust variance of
  $r(t+\Delta) - r(t)$, non-negative LS), $\Sigma_{\mathrm{opt}} = \mathrm{diag}(\sigma)\,C\,\mathrm{diag}(\sigma)$
  with $\sigma_a^2$ at least the rest value (rest second differences / 6); IMU increment noise
  $q_\theta = \max(\bar q, q_{\mathrm{rest}})$; drift prior $\mathrm{SD}(b_0)^2 = \mathrm{SE}_w^2 + \max(\bar c, f^2/3)$;
  drift random walk $q_b = \max(\lVert w\rVert, f)^2/(3T)$ ($f$ = 0.25°/min floor, $T$ the training <!-- num-ok: method parameter -->
  span, ≥ 60 s); <!-- num-ok: method parameter -->
- per-frame measurement scale
  $s_k = \operatorname{clip}\big(\operatorname{clip}((\mathrm{rms}_k/\mathrm{rms}_{\mathrm{ref}})^2, \tfrac14, 16)\cdot\tfrac{n_{\mathrm{tags,ref}}}{n_{\mathrm{tags},k}}\cdot\tfrac{n_{\mathrm{cam,ref}}}{n_{\mathrm{cam},k}},\ \tfrac14, 50\big)$;
- gate: $\chi^2_3$ quantile at 0.999.

- **Parameters:** `validation/fusion.py:DEFAULTS` (no study override): `lag_range_s` =
  (−0.05, 0.08), `lag_mode` = `pooled`, `noise_source` = `train_structure`, `gate_prob` = 0.999,
  `reset_s` = 1.0, `rest_trim_s` = 0.5, `imu_max_gap_s` = `qc.imu.gap_s`.
- **Code:** `jtsmm/imu/fusion.py:calibrate`, `residual_structure`, `settings_from_sensors`,
  `quality_scale`; `validation/fusion.py:prepare`.

### 9.2 F0, F1, F2

**Propagation.** Between optical frames $a$ and $b$ the IMU increment in the marker frame is
$D_{ab} = X^\top R_{\mathrm{imu}}(t_a+\lambda)^\top R_{\mathrm{imu}}(t_b+\lambda)\,X$; nothing is
propagated across an IMU gap.

**F0** (offline): $R(t) = R_a\,\mathrm{Exp}\big(f\,\mathrm{Log}(R_a^\top R_b)\big)$,
$f = (t - t_a)/(t_b - t_a)$ (SLERP); F0_hold: $R(t) = R_a$.

**F1** (causal complementary filter):

$$
R^- = R\,D_{ab},\qquad y = \mathrm{Log}\big(R_{\mathrm{opt},b}\,(R^-)^\top\big),\qquad
R \leftarrow \mathrm{Exp}(\kappa\,y)\,R^-,\quad \kappa = 1 - e^{-\Delta t/\tau},
$$

with $\tau = \arg\min_\tau J(\tau)$ on a log grid in [0.02, 5] s,
$J(\tau) = \sigma^2\frac{\kappa}{2-\kappa} + q\,\Delta t\,\frac{(1-\kappa)^2}{1-(1-\kappa)^2} + \frac13\Big(b\,\Delta t\,\frac{1-\kappa}{\kappa}\Big)^2$
(optical noise pass-through, IMU random-walk tracking error, lag error of an uncompensated drift
$b$). Innovations with $\lVert y\rVert^2 > \chi^2_3\,v$ (predicted spread $v$) are rejected;
consistent rejection for 1 s re-initializes. <!-- num-ok: method parameter -->

**F2** (error-state Kalman filter on SO(3) with a drift-rate state). Nominal state $(R, b)$, error
state $x = [\delta\theta, \delta b]$ with $R_{\mathrm{true}} = \mathrm{Exp}(\delta\theta)\,R$ (camera
frame); between consecutive valid optical frames ($\Delta t$, IMU path angle $\vartheta$):

$$
R^- = \mathrm{Exp}(b\,\Delta t)\,R\,D_{ab},\qquad
F = \begin{bmatrix} I & \Delta t\,I\\ 0 & I\end{bmatrix},\qquad
Q = \begin{bmatrix}\big(q_\theta\Delta t + s_f^2\vartheta^2/3 + q_b\Delta t^3/3\big)I & q_b\tfrac{\Delta t^2}{2}I\\ q_b\tfrac{\Delta t^2}{2}I & q_b\Delta t\,I\end{bmatrix}
$$

(no IMU link: $D = I$ and the attitude variance grows by $(\text{lost\_rate}\,\Delta t)^2$);
measurement $y = \mathrm{Log}(R_{\mathrm{opt}}(R^-)^\top) = H x + v$, $H = [I\ 0]$,
$\mathrm{Cov}(v) = s_k\,\Sigma_{\mathrm{opt}}$; $S = P_{\theta\theta} + s_k\Sigma_{\mathrm{opt}}$,
$\mathrm{NIS} = y^\top S^{-1}y$; frames with NIS > $\chi^2_{3,0.999}$ are gated (logged, not used);
update $K = P H^\top S^{-1}$, $\delta x = K y$, $R \leftarrow \mathrm{Exp}(\delta x_\theta)R$,
$b \leftarrow b + \delta x_b$, Joseph-form covariance. An RTS smoother gives F2:
$G_k = P_k F_{k+1}^\top (P^-_{k+1})^{-1}$, applied to the error between the smoothed and the
predicted state. Queries between frames interpolate the smoothed states (forward and backward
IMU propagation blended geodesically). In F0–F2 the position stays optical (linear
interpolation).

- **Code:** `jtsmm/imu/fusion.py:f0_optical`, `f1_complementary`, `tau_cost`, `f2_kalman`.

### 9.3 F2P position fusion (VD6)

- **Gravity.** The BNO085 records the raw `accelerometer` (250 Hz, chip frame, **gravity <!-- num-ok: method parameter -->
  included**). Gravity is removed with the chip's own game-rotation-vector orientation
  (world $z$ up), $a_{\mathrm{iw}} = R_{\mathrm{imu}}(t)\,f(t) - g$, $g$ = (0, 0, 9.80665) m/s², so
  its heading drift does not matter.
- **World mapping.** $a = \mathrm{Exp}\big(w(t' - t_{\mathrm{ref}})\big)\,Z\,a_{\mathrm{iw}}$ at the
  optical-clock time $t' = t_{\mathrm{imu}} - \lambda$, in mm/s².
- **Filter.** Per axis (shared dynamics) a Kalman filter on $x = [p, v, b_a]$ (position,
  velocity, world-frame accelerometer bias), driven by the latest acceleration sample $u$
  (zero-order hold) and corrected by the optical positions:

$$
x_{k+1} = \begin{bmatrix}1 & \Delta t & -\tfrac12\Delta t^2\\ 0 & 1 & -\Delta t\\ 0 & 0 & 1\end{bmatrix} x_k
+ \begin{bmatrix}\tfrac12\Delta t^2\\ \Delta t\\ 0\end{bmatrix} u_k,
\qquad
Q = q_a\begin{bmatrix}\Delta t^3/3 & \Delta t^2/2 & 0\\ \Delta t^2/2 & \Delta t & 0\\ 0 & 0 & 0\end{bmatrix} + \mathrm{diag}(0, 0, q_b\Delta t),
$$

  measurement $z = p + \nu$, $\nu \sim \mathcal N(0, r_p^2)$; $P_0 = \mathrm{diag}(r_p^2, 200^2, 500^2)$;
  RTS smoother (F2P) or forward only (F2P_causal); run in a window of the fold ± 5 s. <!-- num-ok: method parameter -->
- **Settings from the JTS-MM data:** $r_p$ = RMS over axes of the optical position SD about its
  median at rest (≥ 0.05 mm); $q_a = \hat\sigma_a^2\,\Delta t_{\mathrm{frame}}$ with $\hat\sigma_a$ the <!-- num-ok: method parameter -->
  robust SD (1.4826 MAD, median removed) of IMU minus optical acceleration on the training frames
  (optical: local quadratic fit over ±3 frames; IMU averaged over each frame interval);
  $q_b = 20^2$ (mm/s³)²·s (fixed physical scale).
- **Limitations.**
  - A tilt error of $\delta$ leaks $g\sin\delta$ (≈ 171 mm/s² per degree). The bias state absorbs <!-- num-ok: physical constant (g sin 1°) -->
    its slowly varying part.
  - The lever arm of the accelerometer from the body origin ($\dot\omega\times r + \omega\times(\omega\times r)$)
    is not modelled.

- **Code:** `jtsmm/imu/position.py:world_acceleration`, `settings_from_data`, `position_kf`;
  `validation/fusion.py:fuse`.

### 9.4 Evaluation

- Two metrics per estimator, of equal value (VD5): the §6 metric in each system's own frames
  (incisor RMS, rotation RMS, bias-removed rotation RMS = RMS of
  $\lVert(\phi_S - \phi_Q) - \overline{(\phi_S - \phi_Q)}\rVert$ over the fold's evaluated frames), and
  frame-aligned ($A$, $B$ fitted on the *optical-only* poses of the other tasks, so fusion never
  shapes the alignment).
- At the optical frame times (`query` = `frames`) and at the 200-Hz Qualisys frames inside the
  fold (`query` = `ref200`, exact reference, no spline).
- **Paired against F0** (`paired`): per-frame differences
  $\Delta_i = \mathrm{err}_{E,i} - \mathrm{err}_{F0,i}$ pooled over the folds, mean and 95 % CI from a <!-- num-ok: method parameter -->
  bootstrap over 1-s blocks of frames (1000 replicates):
  $\bar\Delta^* = \sum_{\text{blocks}}\sum_i\Delta_i / \sum_{\text{blocks}} n_b$.
- **Dropouts** (`dropout`): optical gaps of 0.1, 0.25, 0.5, 1 and 2 s placed back to back inside <!-- num-ok: method parameter -->
  each held-out task (0.25 s from its ends, 1 s of untouched frames between gaps), removed from <!-- num-ok: method parameter -->
  every estimator's input (including F0), scored on the dropped frames only; the calibration is
  not refitted.
- **τ curve** (`tau`, descriptive, never used for selection): F1 with τ ∈ {0.05, 0.1, 0.2, 0.5,
  1, 2} s.
- **Diagnostics:** NIS mean and share inside the $\chi^2_{3,0.95}$ quantile, gated frames,
  calibration lag, RMS, inlier fraction, heading drift, τ, $r_p$, $\hat\sigma_a$.

- **Code:** `validation/evaluate.py:run_ref_fusion`, `_block_ci`;
  `validation/fusion.py:dropout_masks`.

**Paper wording.** As a secondary analysis, optical and inertial data were fused by a
complementary filter and by an error-state Kalman filter with Rauch–Tung–Striebel smoothing on
the rotation group, and, for position, by a Kalman filter on position, velocity and
accelerometer bias driven by the gravity-compensated accelerometer. The IMU-to-marker
calibration, the clock lag and all filter noise settings were derived from the wearable
system's own optical and inertial data in a leave-one-task-out scheme, never from the
reference. Each estimator was compared with optical-only tracking by paired per-frame
differences with 1-s block-bootstrap confidence intervals, and under simulated optical
dropouts of 0.1–2 s. <!-- num-ok: method parameter -->

## 10. Point-marking accuracy (VD4; section (I) of the paper)

`biomech/validation/landmarks.py`, stage `ref_landmarks`. Per participant, not pooled.

### 10.1 Registration of the body frames between the systems

The sync fit's $X$ trades translation against $Y_{\mathrm{CT}}$, because the CT is waved about
a point near its tip; its $X$ differs by `\numRegXSyncShiftPOne` / `\numRegXSyncShiftPTwo` (16 / 9 mm, P1 / P2) from the tip-based one. So
$X$ is refit from the two pivot-calibrated tips over the sync-window motion (robust Kabsch,
$\mathrm{tip}_Q \approx X\,\mathrm{tip}_S$, frames beyond median + 4·1.4826 MAD dropped, 5 rounds;
tip RMS `\numRegTipRmsPOne` / `\numRegTipRmsPTwo` (1.8 / 3.4 mm)), and the body registration given that $X$ from the body poses outside the
sync windows, $Y_b = \operatorname{robust\,mean}_i (X T_{S,b,i})^{-1}T_{Q,b,i}$ (`y_given_x`).
Independent of the landmarks (`registration`).

- **Code:** `validation/landmarks.py:x_from_tips`, `y_from_bodies`.

### 10.2 Point error with CIs

Per landmark $n$ (median CT tip in the body frame over the common still window, EV2), the JTS-MM
point mapped into the Qualisys body frame minus the Qualisys point, along the Qualisys
anatomical axes $F$ (anterior, left, superior):

$$
e_n = F^\top\big(Y_b^{-1}\,p^S_n - p^Q_n\big)\quad(\texttt{e\_anterior/left/superior\_mm}),\qquad
\lVert e_n\rVert\ (\texttt{e\_mm}).
$$

95 % CIs by a joint bootstrap (400 replicates): moving-block resampling (0.25-s blocks) of each <!-- num-ok: method parameter -->
hold's frames in both systems (median of the resample) combined with draws of $Y_b$ (chordal means
of moving-block resamples, 1-s blocks, of the per-frame estimates $(X T_{S,b,i})^{-1}T_{Q,b,i}$
outside the sync windows). The absolute errors include the registration uncertainty (a few
mm). **Accuracy:** $\mathrm{RMS}_{\mathrm{teeth}} = \sqrt{\overline{\lVert e_n\rVert^2}}$ over the six
teeth and over the two condyle points (`marking_summary`).

### 10.3 Registration-free summaries

- frame-free within-body distance differences $\lVert p^S_a - p^S_b\rVert - \lVert p^Q_a - p^Q_b\rVert$
  (`landmark_distances`; RMS over tooth pairs `rms_dist_diff_teeth_mm`);
- the angle errors of the anatomical frames,
  $\angle\big(R(F_Q)^\top R(Y_b^{-1}F_S)\big)$, and their origin distance, with bootstrap CIs
  (200 replicates), checked against the VD3 offsets estimated from motion (median over the
  leave-one-task-out folds; `frame_errors`);
- the incisor offset at rest the frames cause: with $\bar G$ the mean Qualisys
  $T_{\mathrm{HP}}^{-1}T_{\mathrm{MP}}$ at rest,
  $\lVert p(F_{\max,Q}^{-1}\bar G F_{\mathrm{mand},Q}) - p(\tilde F_{\max}^{-1}\bar G\tilde F_{\mathrm{mand}})\rVert$,
  $\tilde F = Y^{-1}F_S$ (`rest_incisor_offset_mm`, `rest_rotation_offset_deg`).

### 10.4 Static precision, $\sigma_{\mathrm{ref}}$, attribution

- **Static precision** (`landmark_precision`), both systems: the 3-D RMS deviation of the hold's
  tip positions from their mean (`sd_mm`) and the median distance from their median (`mad_mm`).
- **$\sigma_{\mathrm{ref}}$ per landmark:** Monte Carlo (300 draws) at the hold's mean poses: marker
  noise $\hat\sigma_m$ (§5.4, of that hold segment) on the CT and the body template, Kabsch refits,
  and the pivot tip drawn from $\mathcal N(d, \mathrm{diag}(\mathrm{SE}^2))$; $\sigma_{\mathrm{ref}}$ = RMS 3-D
  displacement of the Qualisys landmark (conservative: no averaging over the hold frames);
  `ratio` $=\lVert e_n\rVert/\sigma_{\mathrm{ref}}$; teeth summary
  $\sqrt{\overline{\sigma_{\mathrm{ref},n}^2}}$ and `adequacy_teeth`.
- **Attribution** (`hold_attribution`, descriptive), per hold:
  - the CT orientation disagreement $\vartheta = \angle\big(\overline{R(X T_{S,\mathrm{CT}}Y_{\mathrm{CT}})^\top R(T_{Q,\mathrm{CT}})}\big)$
    (mean rotation over the hold) and the tip error it predicts over the lever,
    $\vartheta\cdot\lVert\bar m_{\mathrm{CT}}\rVert$ ($\bar m$ the CT marker centroid in the tip frame);
  - the JTS-MM CT and body tag counts and reprojection RMS;
  - the Qualisys CT cluster deviation;
  - the smallest Qualisys viewing obliquity of the CT plate.

- **Code:** `validation/landmarks.py:run_ref_landmarks`, `y_bootstrap`, `_sigma_ref_hold`,
  `hold_attribution`; `validation/anatomy.py:digitize`, `common_windows`.

**Paper wording.** Point-marking accuracy was assessed by mapping each tooth and condyle point
digitized by the wearable system into the Qualisys body frame, using a registration obtained
from the pivot-calibrated tool tips independently of the landmarks, and comparing it with the
Qualisys digitization of the same still hold. Confidence intervals combined a moving-block
bootstrap of the hold frames with a bootstrap of the registration. Registration-free measures
(within-body distances, the angular errors of the anatomical frames) and the static precision
of both systems are reported alongside; the reference uncertainty of each Qualisys point was
obtained by Monte Carlo propagation of the marker noise and the tip uncertainty.

## 11. Additions VD9–VD12 and sampling notes

### 11.1 IMU alone (VD10, stage `ref_fusion`, estimator `I`)

The calibrated head and mouth IMUs alone, with no optical correction (the same
leave-one-task-out calibration as F1/F2, fitted on JTS-MM optics only):
$R_I(t) = \mathrm{Exp}(w(t - t_{\mathrm{ref}}))\,Z\,R_{\mathrm{imu}}(t+\lambda)\,X$ per body, combined
into $M_I$ with the optical positions. Table `ref_fusion:imu_opening`, per participant and
task, compares the opening angle $\theta = [\mathrm{Log}(R(M)\,R(\bar M_{\mathrm{rest}})^\top)]_y$ since
each system's own mean rest pose (the dataset convention, `_opening_deg`) for IMU alone vs
Qualisys, JTS-MM optics vs Qualisys and IMU alone vs optics: with $\Delta\theta_i$ the per-frame
difference, RMSE $\sqrt{\overline{\Delta\theta^2}}$, bias $\overline{\Delta\theta}$, SD (ddof 1) and the
reference range. The IMU-alone rotation error is also in the fusion arrays (`err_rot_I_deg`,
`err_rot_aligned_I_deg`). The IMU error is mostly a constant offset that grows in task order
(descriptive; untested against time since calibration).

- **Code:** `validation/evaluate.py:run_ref_fusion`, `_opening_deg`;
  `jtsmm/imu/fusion.py:Calibration.predict`.

**Paper wording.** The inertial sensors alone, calibrated on the wearable system's optical data
of the other tasks, were compared with Qualisys and with the optical tracking on the opening
angle (rotation about the maxillary lateral axis since the rest pose) by the RMS error, bias and
standard deviation of the differences.

### 11.2 Rate and filter bias (VD9, stage `ref_sampling`, `validation/sampling.py`)

On the cycles of `ref_agreement`, three pipelines compute the per-cycle peak opening and
closing angular velocity ($\max_t\dot\theta$, $\max_t(-\dot\theta)$; open_close and chewing) and
the peak incisal speed $\max_t\lVert\dot p_{\mathrm{inc}}\rVert$, each from jaw-in-head poses through a
zero-phase 2nd-order Butterworth low-pass (`se3.lowpass`: translations and rotation vectors
about the segment mean rotation) and a Savitzky–Golay derivative (`derivatives`), $\theta$
relative to the first valid rest frame:
- `ref200`: Qualisys at 200 Hz, 20 Hz low-pass, 35 ms Savitzky–Golay (the reference); <!-- num-ok: method parameter -->
- `ref40`: Qualisys at the JTS-MM frame times (40 Hz) through the dataset chain (6 Hz <!-- num-ok: method parameter -->
  low-pass, 0.15 s Savitzky–Golay); <!-- num-ok: method parameter -->
- `jtsmm`: JTS-MM through the dataset chain.

`bias` holds per task, quantity and pipeline the median over cycles of the ratio
$\rho_j = \mathrm{peak}_{j}^{\mathrm{pipe}}/\mathrm{peak}_{j}^{\mathrm{ref200}}$ (finite, > 0, ≥ 3 cycles) with a
cycle-bootstrap CI of the median (2000 replicates). **EV7:** the reported peak ratio is the
dataset chain on the reference (`ref40 / ref200`) for the opening angular velocity in
open_close, per participant (`peak_ratio_open`); the other ratios stay in `bias`.

**Movement bandwidth** (`spectrum_unfiltered`, primary): the opening rate from the
**unfiltered** 200-Hz Qualisys opening angle by central differences,
$\dot\theta_i = (\theta_{i+1} - \theta_{i-1})/(t_{i+1} - t_{i-1})$; per task the Welch PSD (Hann, 512
samples = 2.56 s, 50 % overlap, mean removed per run, averaged over the finite runs weighted by <!-- num-ok: method parameter -->
their segment counts) minus the noise floor $N(f)$ measured the same way on the rest phase,
$P_c(f) = \max(P(f) - N(f), 0)$; from the cumulative $C(f) = \sum_{f'\le f}P_c/\sum P_c$:
$f_{50}$, $f_{95}$, $f_{99}$ and the share above 20 Hz (the Nyquist frequency of 40-Hz optics) <!-- num-ok: method parameter -->
$\sum_{f>20}P_c/\sum P_c$. `spectrum` keeps the pre-EV7 values, computed on the filtered `ref200`
chain, which removes most power above 20 Hz by itself. <!-- num-ok: method parameter -->

**Reading (EV7 addendum, 2026-10-31):** the unfiltered spectrum is noise-dominated above the movement
band (the rest-phase floor underestimates the marker noise during movement), so the unfiltered
$f_{95}$ is not quoted as a movement bandwidth. Reported are the share of power above 20 Hz in <!-- num-ok: definition -->
the movement tasks, a value that includes noise and therefore overstates the movement content
(largest over open-close, chewing and free motion `\numBandAboveTwentyMaxPOne` / `\numBandAboveTwentyMaxPTwo` (0.068 / 0.329), P1 / P2,
P2's the chewing value; corrected 2026-11-01 from `\numBandAboveTwentyFreePTwo` (0.16), which is P2's
free_motion value), and $f_{50}$, with $f_{95}$ bracketed by the filtered spectrum and the dataset's
coherence-weighted IMU bandwidth; the peak-velocity ratio is unaffected. A caveat next to the
dataset's velocity requirement (n = 2), never a correction; the dataset study tests the cutoff
with its own sensitivity runs (D20).

- **Parameters:** `sampling.PIPELINES` = {`ref200`: (20 Hz, 0.035 s), `ref40`: (6 Hz, 0.15 s), <!-- num-ok: method parameter -->
  `jtsmm`: (6 Hz, 0.15 s)}; Welch `nperseg` = 512, split 20 Hz (code constants). <!-- num-ok: method parameter -->
- **Code:** `validation/sampling.py:_chain`, `cycle_peaks`, `bias`, `peak_ratio_open`,
  `spectrum_unfiltered`, `central_rate`, `_welch_runs`, `spectrum`.

**Paper wording.** To quantify what 40-Hz optics and the dataset's 6-Hz filter chain miss, peak
opening angular velocities per cycle were computed from Qualisys at 200 Hz (20-Hz low-pass) and <!-- num-ok: method parameter -->
from Qualisys resampled to the wearable system's frame times and processed with the dataset
chain; the median ratio of the two with a cycle-bootstrap confidence interval is reported. The
movement bandwidth was estimated from the Welch spectrum of the unfiltered 200-Hz opening rate
after subtraction of the rest-phase noise floor.

### 11.3 EMG relevance and CMC (VD11, stage `ref_emg`)

`relevance` applies the dataset rule D19 per system: a muscle × task cell is active when its
task p95 ≥ rest mean + 3 SD of the rest envelope and ≥ 5 % of the largest activity recorded. <!-- num-ok: method parameter -->
Profile agreement is interpreted for active cells. Kadaba's CMC has no real value when the
curves differ more in level than they vary in shape (the radicand $1 - \mathrm{num}/\mathrm{den} < 0$):
`cmc_defined` = false and the views print "undefined" with a footnote, never "n/a". The Pearson
$r$ already in `profiles` is the offset-free companion (it removes each curve's mean).

- **Parameters:** `stages.emg_relevance`: `k_sd` = 3.0, `min_pct` = 5.0 (EV11).
- **Code:** `validation/emg.py:compare`; `biomech/emg/relevance.py:relevance`.

### 11.4 Agreement after the dataset chain (VD12, stage `ref_kinematics`, table `segments_filtered`)

The jaw-in-head poses go through the dataset's pose processing (`evaluate.dataset_chain`: gaps
≤ 0.5 s bridged by SE(3) interpolation, then the zero-phase 2nd-order 6-Hz Butterworth on the <!-- num-ok: method parameter -->
translations and on the rotation vectors relative to each valid segment's mean rotation,
`se3.lowpass`; `signals.kinematics`), applied to both systems (`both`: the measurement error of
the processed signal) or to JTS-MM only (`jtsmm_only`: noise reduction and filter attenuation
together). The gated agreement (`incisor_rms_mm`, `incisor_dyn_rms_mm`, `rot_rms_deg`,
`rot_dyn_rms_deg`) is recomputed per movement task and pooled, in own frames and with the
primary leave-one-task-out alignment (not refitted).

- **Parameters:** `signals.kinematics` (dataset; `evaluate.DATASET_CHAIN` defaults
  `max_gap_s` = 0.5, `lowpass_hz` = 6.0, `lowpass_order` = 2).
- **Code:** `validation/evaluate.py:dataset_chain`, `filtered_agreement`.

### 11.5 Sampling rates and resolutions

- **Delsys** sensors sample at 4370 Hz (EMG) or 2148 Hz; resampling as in §8a <!-- num-ok: method parameter -->
  (`validation/emg.py:resample_channel`; the rates are in `timeline:delsys_rates_hz`).
- **Onsets:** EMG onsets are found at the EMG rate (4 kHz JTS-MM, the resampled Delsys grid),
  so their resolution is below 1 ms. Mechanical onsets come from the optical opening angle <!-- num-ok: method parameter -->
  (40 Hz frames, shifted 6 ms to the exposure centre), interpolated onto a 200 Hz grid <!-- num-ok: method parameter -->
  (`mech_arrays`: gaps ≤ 0.1 s) and differentiated there; the onset is the velocity-threshold <!-- num-ok: method parameter -->
  crossing on that grid (5 ms steps), from a signal measured every 25 ms. The dataset uses <!-- num-ok: method parameter -->
  the IMU angular velocity at 200 Hz instead (an05). <!-- num-ok: method parameter -->
- **Cycle features:** the peak incisal speed of `ref_agreement:cycles` is the maximum of first
  differences at 40 Hz for both systems (`agreement.py`), so both are biased alike; the <!-- num-ok: method parameter -->
  per-cycle peaks at 200 Hz are in `ref_sampling:peaks` (VD9) as the sensitivity. <!-- num-ok: method parameter -->

## 11a. Equivalence margin, hand-eye mounting, lever arm, smoothing, accelerometer (VD13–VD18)

### Equivalence margin (VD13, `ref_agreement`)

`equiv_margin` = the smallest symmetric bound containing the 95 % CI of a feature bias, <!-- num-ok: definition -->
$\max(|\mathrm{lo}|, |\mathrm{hi}|)$, or the whole 95 % band of a waveform's mean difference, <!-- num-ok: definition -->
$\max_\tau\max(|\mathrm{lo}(\tau)|, |\mathrm{hi}(\tau)|)$ (§7). The pre-specified verdicts against
±0.5 mm / ±0.5° and ±2.0 mm / ±1.2° stay. <!-- num-ok: pre-specified criterion -->

### Hand-eye mounting (VD14, `ref_fusion`)

Each take uses the latest hand-eye session before it (`reference.yaml`: global `imu_mounting`,
per-participant override); the `_precal` estimators F1/F2 keep that mounting fixed
($X = R_{\mathrm{mount}}^\top$). `calibration:mounting_vs_intake_deg` is
$\angle(X_{\mathrm{intake}}^\top X_{\mathrm{precal}})$. Between the two hand-eye sessions the head IMU
was remounted (`\numLeverMountChangeHead` (176°)), the jaw IMU not (`\numLeverMountChangeJaw` (1.2°); `ref_imu_lever:repeatability`).

- **Code:** `validation/stages.py:imu_mounting_files`; `validation/fusion.py:load_mounting`.

### IMU lever arm (VD15, stage `ref_imu_lever`, `biomech/imu/lever.py`)

**What and why.** The hand-eye calibration fits only the mounting rotation; the accelerometer
comparison and a position fusion that models the IMU site also need the IMU position $r$ on its
plate (body frame, from the optical body origin). In each plate's tumbling phase of the hand-eye
session, the accelerometer's specific force (chip frame, gravity included) is

$$
f = R_{cb}\,R_{wb}^\top\big(\ddot p_b + \ddot R_{wb}\,r - g_w\big) + b_f,
\qquad R_{cb} = R_{\mathrm{mount}}^\top,
$$

with the optical body pose $(R_{wb}, p_b)$, the unknown gravity vector $g_w$ in the (unlevelled)
camera frame and an accelerometer bias $b_f$; linear in $(r, g_w, b_f)$, so per frame three rows

$$
\big[\,R_{cb}R_{wb}^\top\ddot R_{wb}\ \ \big|\ -R_{cb}R_{wb}^\top\ \ \big|\ \ I\,\big]
\begin{bmatrix} r\\ g_w\\ b_f\end{bmatrix} = f - R_{cb}R_{wb}^\top\ddot p_b,
$$

solved by ordinary least squares over all frames. Derivatives: optical positions and the nine
rotation-matrix entries low-passed (4th-order Butterworth, zero phase, at $f_c$) on gap-free
40-Hz runs (≥ 1.5 s, steps ≤ 30 ms, 0.25 s dropped at each end) and differentiated twice by <!-- num-ok: method parameter -->
central second differences, $\ddot x_i = (x_{i+1} - 2x_i + x_{i-1})/\Delta t^2$; $R_{wb}$ is the
nearest rotation of the filtered matrix; the accelerometer low-passed at the same $f_c$ and
interpolated. **Uncertainty:** OLS SEs from $s^2(M^\top M)^{-1}$ and a bootstrap over 1-s blocks of
frames (1000 replicates, percentile CI). **Checks:** $\lVert g_w\rVert/9806.65$ (`g_ratio`), the
cut-off sweep spread (`fc_spread_mm`: largest per-axis range over the sweep), the 95 % angular- <!-- num-ok: method parameter -->
rate band (`motion_f95_hz`), and the between-session difference of each IMU's lever arm with the
mounting change (`repeatability`). Validated on a synthetic session (`tests/test_v_lever.py`).
Both lever arms repeat between the sessions: jaw `\numLeverJawRepeat` (5 mm) at a `\numLeverMountChangeJaw` (1.2°) mounting
change, head `\numLeverHeadRepeat` (4 mm) after the `\numLeverMountChangeHead` (176°) remount.

- **Parameters:** `biomech/imu/lever.py:DEFAULTS`: `fc_hz` = 6.0, `fc_sweep_hz` = [4, 8],
  `n_boot` = 1000, `block_s` = 1.0, `edge_s` = 0.25, `min_run_s` = 1.5, `max_gap_s` = 0.03;
  `reference.yaml: imu_mounting_sessions`.
- **Code:** `biomech/imu/lever.py:design`, `fit`, `motion_band_hz`, `run_ref_imu_lever`.

**Paper wording.** The position of each inertial sensor on its plate was estimated from the
hand-eye calibration recordings by linear least squares, comparing the accelerometer signal
with the rigid-body acceleration of the optically tracked plate at the unknown sensor position,
with the gravity vector and an accelerometer bias as additional unknowns; uncertainty was
obtained by a 1-s block bootstrap and the estimate was checked across filter cut-offs and
between two calibration sessions.

### Smoothing and jitter (VD16, VD18; `ref_fusion:smoothness`, `evaluate.smoothness_row`)

For raw optics (F0), optics + 6 Hz low-pass (`F0_lp6`: zero-phase 4th-order Butterworth on runs <!-- num-ok: method parameter -->
≥ 1 s at 40 Hz, interpolated to 200 Hz), F2, F2P and Qualisys, at the 200-Hz reference frames of <!-- num-ok: method parameter -->
the frame-aligned incisor trajectory $x(t)$, with $\mathrm{HP}_{6}(x) = x - \mathrm{LP}_{6\,\mathrm{Hz}}(x)$
(4th-order zero-phase Butterworth per finite run ≥ 1 s): <!-- num-ok: method parameter -->

$$
\texttt{jitter\_rms\_mm} = \mathrm{RMS}\,\lVert\mathrm{HP}_6(x)\rVert,
\qquad
\texttt{jitter\_err\_rms\_mm} = \mathrm{RMS}\,\lVert\mathrm{HP}_6(x - x_Q)\rVert,
\qquad
\texttt{err\_rms\_mm} = \mathrm{RMS}\,\lVert x - x_Q\rVert .
$$

**Reported jitter (VD18):** the content above 6 Hz of the estimator's **error** against <!-- num-ok: method parameter -->
Qualisys (`jitter_err_rms_mm`), per participant; the estimator's own content above 6 Hz stays as <!-- num-ok: method parameter -->
"smoothness" (`jitter_rms_mm`; a low-pass "wins" it by construction). The Qualisys noise sets a
floor under the error jitter (P2: Qualisys's own content `\numSmoothQualisysPTwo` (0.40 mm)), stated with it. View `val_fusion_smoothing`.

- **Code:** `validation/evaluate.py:smoothness_row`, `_jitter_rms`, `_lowpass_runs`,
  `run_ref_fusion`.

**Paper wording.** High-frequency jitter was quantified as the RMS of the error against Qualisys
above 6 Hz (the dataset's pose low-pass), evaluated at the 200-Hz reference frames. <!-- num-ok: method parameter -->

### Accelerometer against Qualisys (VD17, stage `ref_imu_accel`, `validation/accel.py`)

**Reference.** The Qualisys acceleration of the IMU site: the VD15 lever arm (JTS-MM body frame)
carried into the Qualisys body frame, $r_Q = Y_b^{-1}r_J$; the site $p(t) = T_{Q,b}(t)\,r_Q$ at
200 Hz, low-passed at $f_c$ (4th-order zero-phase, gap-free runs ≥ 1 s) and differentiated twice <!-- num-ok: method parameter -->
by central differences, $\ddot p$.

**Model** (chip frame, gravity included):

$$
f(t + \lambda) = R\,R_{Q,b}(t)^\top\big(\ddot p(t) - g_Q\big) + b_f,
\qquad \lVert g_Q\rVert = 9806.65\ \mathrm{mm/s^2}\ \text{(direction fitted)},
$$

$R$ constant (chip ← Qualisys body), started from $R_{\mathrm{mount}}^\top R(Y_b)$. Fitted
leave-one-task-out on the other movement tasks plus rest (training frames thinned to 50 Hz): <!-- num-ok: method parameter -->
alternating least squares ($g_Q, b_f$ linear given $R$, then $\lVert g_Q\rVert$ fixed; $R$ by Kabsch given
$g_Q, b_f$; 15 rounds) on a lag grid (−40 to 80 ms, 5-ms steps), then a joint Levenberg–Marquardt <!-- num-ok: method parameter -->
refinement at the best lag. **Outcomes** per held-out task, dynamic acceleration (gravity removed,
chip frame), reference $a_{\mathrm{ref}} = R R_Q^\top\ddot p$:
- `sensor`: $a_{\mathrm{meas}} = f - b_f + R R_Q^\top g_Q$ (isolates the accelerometer);
- `own_orientation`: $f - b_f - R_{\mathrm{imu}}^\top\hat g$ with the IMU's own orientation and
  $\hat g$ the training mean of $R_{\mathrm{imu}}(f - b_f)$ (adds the gravity leak of the IMU's tilt
  error);
- `no_lever`: `sensor` with $r = 0$ (refitted; checks VD15 on independent data).

Per outcome: $\mathrm{RMS}\lVert a_{\mathrm{meas}} - a_{\mathrm{ref}}\rVert$, $\mathrm{RMS}\lVert a_{\mathrm{ref}}\rVert$,
$r^2_{\mathrm{dyn}} = 1 - \sum\lVert e\rVert^2/\sum\lVert a_{\mathrm{ref}} - \bar a_{\mathrm{ref}}\rVert^2$, the band-mean 3-D
magnitude-squared coherence over 0.5–$f_c$ Hz ($|\sum_a S_{xy,a}|^2/(\sum_a S_{xx,a}\sum_a S_{yy,a})$,
Welch 256 samples), and the rest-phase RMS of both as noise floors. Views `val_imu_accel`,
`val_table_imu_accel`.

- **Parameters:** `validation/accel.py:DEFAULTS`: `fc_hz` = 6.0, `fc_sweep_hz` = [4, 10],
  `lag_grid_ms` = [−40, 80, 5], `n_iter` = 15, `fit_rate_hz` = 50, `g_mm_s2` = 9806.65,
  `fit_scale` = false, `coh_band_lo_hz` = 0.5, `coh_nperseg` = 256.
- **Code:** `validation/accel.py:site_acceleration`, `fit`, `evaluate_body`, `_scores`,
  `_coherence`; `validation/stages_ref.py:ref_imu_accel_stage`.

**Paper wording.** The accelerometer was compared with the Qualisys acceleration of the sensor
site, obtained by double differentiation of the tracked plate pose at the estimated sensor
position; sensor orientation, gravity direction, bias and lag were fitted on the other tasks
and the held-out task was scored by the RMS error, the explained variance and the coherence of
the dynamic acceleration.

## 11b. Additions VD20–VD23 and EV11

**EMG relevance parameters (EV11).** `validation/emg.py` calls `relevance()` with the study's
`stages.emg_relevance` block (k_sd 3, min_pct 5; formerly the function defaults, identical
values).

### Camera-free estimators against Qualisys (VD20)

Stage `ref_camerafree`, `validation/stages_camerafree.py`; exploratory; a systems-paper item,
"fitness for purpose", cited by the dataset paper (D28 routing). Per participant (n = 2; the
mean is formed in the views, descriptive). The estimators themselves are defined in an04 and
`biomech/imu/camerafree.py`; this section gives the validation scoring.

- **Inputs:** the take's IMU streams on a 200 Hz master-clock grid; the hand-eye mounting of <!-- num-ok: method parameter -->
  the take by the VD14 rule; the lever arms from the same hand-eye session (VD15 method,
  primary cut-off, point estimate); the landmark frame from the JTS-MM holds (`ref_tip`
  landmarks); the closed pose $T_c$ = the chordal-mean JTS-MM optical pose over the rest phase
  (after the dataset chain).
- **Rungs:** F0 (JTS-MM optics), E1 (in-take 7-parameter fit on the take's free_motion against
  the JTS-MM optics), E2a / E2b, E3a / E3b (causal; primary E3 = E3b), E4hinge, E4own (ridge
  model on the JTS-MM optical free_motion), E5 (accelerometer bridging) and E6 (accelerometers +
  own model; per-axis model SD = 3-D CV RMS / √3 and per-measurement SDs since
  erratum E40 of the dataset study (change log); an04 §7.10–§7.11). E4gen is not
  computed (it would need the dataset participants' optical couplings).
- **Scoring:** every trajectory — estimates (2nd-order 6-Hz low-pass at 200 Hz, then <!-- num-ok: method parameter -->
  interpolated to the JTS-MM frame times), F0 and Qualisys — through the dataset chain; per
  held-out movement task (free_motion not scored) the frame alignment $(A, B)$ fitted on the
  optical-only poses of the other movement tasks (VD3 / VD10 rule), gated frames left out. With
  $\tilde M = A M B$ and $\tilde T_c = A T_c B$, the opening error and the geodesic rotation error
  since each system's own closed pose are

$$
e_\theta = \big[\mathrm{Log}(R(\tilde M)R(\tilde T_c)^\top)\big]_y - \big[\mathrm{Log}(R(M_Q)R(Q_c)^\top)\big]_y,
\qquad
e_{\mathrm{geo}} = \angle\big(R(\tilde M)R(\tilde T_c)^\top\,(R(M_Q)R(Q_c)^\top)^\top\big),
$$

  and the incisor error $p(\tilde M) - p(M_Q)$ (translation rungs and F0 only); per segment
  `rmse_open_deg`, `bias_open_deg`, `rmse_geo_deg`, `rmse_inc_mm` (`cf_val_segments`).
- **Non-inferiority to F0** (`cf_val_noninferiority`): per-frame paired difference of the
  absolute error, $\Delta_i = \lvert e_{\theta,E,i}\rvert - \lvert e_{\theta,F0,i}\rvert$ for the rotation rungs and
  $\Delta_i = \lVert e_{\mathrm{inc},E,i}\rVert - \lVert e_{\mathrm{inc},F0,i}\rVert$ for the translation rungs,
  pooled over the scored tasks; mean and 95 % CI by resampling 1-s blocks of frames with <!-- num-ok: method parameter -->
  replacement (1000 replicates); non-inferior when the upper CI bound is below the margin.
- **VD10 drift** (`cf_val_drift`): OLS slope (deg/min) of E2's signed opening error against
  time since calibration (the rest-phase centre) and of E3's against time since the end of the
  last anchor, with a 1-s block-bootstrap CI of the slope; the anchor-residual diagnostic splits
  E3's residuals into heading and tilt parts (`cf_val_summary`).
- Nothing is tuned on Qualisys. The IMU dynamic calibration (an04 §1) was on in these takes
  too; the IMU-alone results carry it.
- Values (VD20, change log), P1 / P2:
  - opening RMSE optics `\numCfOpenRmseOpticalPOne` / `\numCfOpenRmseOpticalPTwo` (1.06 / 1.29°),
    anchored (E3b) `\numCfOpenRmseAnchoredPOne` / `\numCfOpenRmseAnchoredPTwo` (1.02 / 1.08°),
    calibrated (E2b) `\numCfOpenRmseCalibratedPOne` / `\numCfOpenRmseCalibratedPTwo` (0.68 / 1.50°);
  - every rotation estimator non-inferior at the opening margin; E3b − F0 in degrees:
    P1 `\numCfOpenDiffAnchoredPOne` (0.05), CI `\numCfOpenDiffAnchoredPOneCI` ([−0.07, 0.15]);
    P2 `\numCfOpenDiffAnchoredPTwo` (−0.19), CI `\numCfOpenDiffAnchoredPTwoCI` ([−0.30, −0.09]);
  - incisor RMSE optics `\numCfIncRmseOpticalPOne` / `\numCfIncRmseOpticalPTwo` (1.5 / 3.6 mm),
    E4own `\numCfIncRmseOwnPOne` / `\numCfIncRmseOwnPTwo` (7.1 / 5.5 mm);
  - no translation estimator non-inferior at the incisor margin; E4own − F0 in mm:
    P1 `\numCfIncDiffOwnPOne` (5.3), CI `\numCfIncDiffOwnPOneCI` ([4.6, 6.0]);
    P2 `\numCfIncDiffOwnPTwo` (1.5), CI `\numCfIncDiffOwnPTwoCI` ([1.0, 2.0]).

- **Parameters:** `stages.ref_camerafree`: `margin_open_deg` = 0.5, `margin_inc_mm` = 1.0,
  `primary_e3` = `E3b`, `calibration_task` = `free_motion`; code defaults `grid_hz` = 200,
  `block_s` = 1.0, `n_boot` = 1000, `ridge_log10_lambda` = [−8, 2, 21], `ridge_folds` = 5,
  `pseudo_hz` = 5.
- **Code:** `validation/stages_camerafree.py:run_ref_camerafree`, `_block_ci`, `_slope`,
  `ref_camerafree_stage`.

**Paper wording.** The camera-free estimators were applied to the comparison recordings and
scored against Qualisys after the dataset's filter chain and the leave-one-task-out frame
alignment. Non-inferiority to optical tracking was tested on the per-frame paired difference of
the absolute error with a 1-s block-bootstrap 95 % confidence interval, with margins of 0.5° for <!-- num-ok: pre-specified criterion -->
the opening angle and 1.0 mm for the incisor position; drift was assessed by the slope of the <!-- num-ok: pre-specified criterion -->
opening error over time since calibration or since the last anchor.

### The dataset's C1 hinge models on both systems (VD21)

Stage `ref_hinge`, `validation/hinge.py`; exploratory. **What and why.** The dataset fits a
fixed hinge to the JTS-MM openings; fitting the same models to the Qualisys trajectories of the
same openings separates anatomical non-hinge motion (present in both) from what JTS-MM tracking
error adds. **Data:** `ref_agreement` cycles of open_close and chewing, start → peak, each system
in its own maxillary frame, three points per frame (incisor $a_1$ and both condyle points
$a_2, a_3$); per opening the closed pose $A$ = the start frame and up to 30 subsampled later
frames $B$; an opening needs a tracked start and ≥ 4 tracked frames, a fit ≥ 10 stacked frames.
**Models** (an02 §6): the best-fit fixed axis $(n, s)$ with per-frame angles $\vartheta_i$,

$$
\min_{n,s,\{\vartheta_i\}}\sum_{i,k}\big\lVert \mathrm{Rot}_{n}(\vartheta_i)(a_k - s) + s - b_{ik}\big\rVert^2
$$

(started at the condyle midpoint and the condyle line), and the axis fixed through the two
condyle points (angles only). **Outcomes** per participant and task (table `hinge`): the incisor
RMS residual of both models (`hinge_fit_mm`, `hinge_tmj_mm`), the best-fit axis's distance from
the condyle midpoint $\lVert(\bar c - s)\times n\rVert$ (`axis_to_condyles_mm`), the largest fitted
opening angle; `hinge_diff`: JTS-MM minus Qualisys. The Qualisys residual shows the anatomical
non-hinge motion; the difference is what JTS-MM tracking error adds to the dataset's C1 residual
(an06 §4). Descriptive, per participant plus the mean; cited by the dataset paper. Values
(VD21, change log), open_close, JTS-MM / Qualisys:
- best-fit hinge P1 `\numHingeFitJtsmmOpenPOne` / `\numHingeFitQualisysOpenPOne` (3.02 / 1.82 mm),
  P2 `\numHingeFitJtsmmOpenPTwo` / `\numHingeFitQualisysOpenPTwo` (2.68 / 1.19 mm);
- hinge through the condyle points P1 `\numHingeTmjJtsmmOpenPOne` / `\numHingeTmjQualisysOpenPOne` (8.58 / 8.57 mm),
  P2 `\numHingeTmjJtsmmOpenPTwo` / `\numHingeTmjQualisysOpenPTwo` (5.67 / 4.61 mm);
- best-fit axis from the condyle midpoint P1 `\numHingeAxisJtsmmOpenPOne` / `\numHingeAxisQualisysOpenPOne` (39 / 47 mm),
  P2 `\numHingeAxisJtsmmOpenPTwo` / `\numHingeAxisQualisysOpenPTwo` (26 / 12 mm).

Chewing: P1 only (P2's chewing openings all contain gated frames).

**Ungated sensitivity** (read-only audit `scripts/audit_recompute/hinge_ungated.py`, run on the
store frozen at the time, `v10`; VD21, 2026-11-01; the stored `ref_hinge` values above are the
same in `v15`): the same cycle rule (`agreement.find_cycles`) on the same
held-out intervals once with and once without the reference-quality gate, both through
`hinge.run_ref_hinge`; the gated rows reproduce the stored table. In P2's chewing
`\numRefGatedChewPTwo` (53 %) of the frames are gated (jaw plate: inter-marker distances up to
`\numRefDevMaxChewPTwo` (0.94 mm) from their median, `ref_quality:consistency`), so every
chewing opening is dropped with the gate. The ungated values (P2 chewing, P1 chewing and
open_close of both) are an audit, not a stored result: they are recorded in the change log of
the validation plan (VD21, 2026-11-01). The conclusions hold without the gate.

- **Parameters:** `stages.kinematics.hinge.max_frames_per_cycle` = 30 (dataset);
  `min_frames` = 10 (code default).
- **Code:** `validation/hinge.py:opening_points`, `run_ref_hinge`;
  `biomech/kinematics/hinge_fit.py:fit_hinge`, `hinge_through`.

**Paper wording.** The fixed-hinge models of the dataset analysis (a best-fit fixed axis and an
axis through the digitized condyle points) were fitted to the same openings tracked by both
systems; the Qualisys residual reflects the anatomical deviation from a pure hinge, and the
difference between the systems the contribution of tracking error. The reported values use
the reference-quality gate; an ungated read-only sensitivity
(`scripts/audit_recompute/hinge_ungated.py`, VD21) changes no conclusion and is
not stored as a result (say "with the reference-quality gate; without it the conclusions are
unchanged", no numbers).

### Error budget per participant (VD22)

Stage `ref_budget`, cohort, `validation/budget.py`; descriptive. Built from tables the
participant stages already store; nothing is recomputed from the recordings.

- `budget`: per participant, part and segment the observed JTS-MM vs Qualisys error split in
  quadrature, $\sigma_{\mathrm{obs}}^2 = \sigma_{\mathrm{JTS}}^2 + \sigma_{\mathrm{ref}}^2$ (V9, `quality.budget`,
  §6.3): motion rows from the `ref_kinematics` budget (bias-removed RMS per gated segment;
  incisor and condyle in mm, rotation in deg), marking rows from the VD4 marking summary (RMS
  point error of the teeth and of the condyle points, with $\sigma_{\mathrm{ref}}$ of those
  landmarks). Columns: `sigma_jts_corrected` $=\sqrt{\sigma_{\mathrm{obs}}^2 - \sigma_{\mathrm{ref}}^2}$, `ratio`
  $=\sigma_{\mathrm{obs}}/\sigma_{\mathrm{ref}}$, `adequate` (ratio ≥ `adequacy_ratio` = 3) and

$$
\texttt{ref\_var\_share} = \min\!\Big(\frac{\sigma_{\mathrm{ref}}^2}{\sigma_{\mathrm{obs}}^2},\ 1\Big),
$$

  the share of the observed variance the reference alone accounts for. `separable` =
  $\sigma_{\mathrm{ref}} < \sigma_{\mathrm{obs}}$ (both finite): only then can the JTS-MM part be resolved;
  otherwise `sigma_jts_corrected` is NaN (not 0; `quality.budget` and `budget._separability`
  both set NaN, so `ref_kinematics:budget` carries NaN too), the share is 1 and `reason` =
  "reference noise >= observed difference: JTS-MM part not resolvable" (2026-10-31; the pinned
  store has such rows for P2: four segments, `\numBudgetUnresolvedRowsPTwo` (12) rows, hold and sync segments: `mand_point_1`,
  `max_point_1`, `max_point_2`, `sync_start`). Views and macros skip these rows (task medians
  over the resolvable tasks; a single row prints "--"; the budget table states "resolvable",
  the figure marks the line with † and draws no JTS-MM bar).
- `marking_holds`: per participant and VD4 hold the marking error $\lVert e_n\rVert$, the CT
  orientation disagreement $\vartheta$ and the tip error it predicts, and whether the hold's CT
  labels were swapped (share of relabelled CT frames ≥ `swap_frac_min` = 0.5, EV1).
- `marking_causes`: per participant the marking frame errors next to their candidate causes:
  median over the jaw's holds of $\vartheta$ and of the predicted tip error, the lever
  $\mathrm{median}(\text{predicted tip}/\vartheta)$ (`ref_landmarks:hold_attribution`), the Qualisys head
  cluster geometry (smallest triangle height; $s_2/s_1$, near 0 = nearly collinear;
  `ref_quality:marker_geometry`), label-swapped holds per jaw and $\sigma_{\mathrm{ref}}$ of the teeth.
- `marking_gap`: each of those quantities for the participant with the smaller and the one
  with the larger maxillary frame error, their difference and ratio. Descriptive (n = 2), no
  test. Columns `pid_smaller` / `pid_larger`, `value_smaller` / `value_larger`,
  `larger_minus_smaller`, `larger_over_smaller` (neutral names since 2026-10-31, `ref_budget`
  version 2; before `pid_ref` / `pid_stress`, `stress_minus_ref`, `stress_over_ref`); the
  reporting is neutral: two separate reference measurements, no ranking.
- Addendum listed in the change log of the validation plan (VD22 addendum, 2026-10-30, before
  reporting): the per-hold CT orientation does not explain the larger maxillary frame error of P2;
  what differs for P2 is the nearly collinear Qualisys head cluster, the label-swapped holds
  and the teeth $\sigma_{\mathrm{ref}}$, and P2's motion budget is reference-heavy.

- **Parameters:** `stages.ref_budget`: `adequacy_ratio` = 3.0, `swap_frac_min` = 0.5,
  `head_body` = `HP`.
- **Code:** `validation/budget.py:error_budget`, `_share`, `_separability`, `marking_holds`,
  `marking_causes`, `marking_gap`, `run_ref_budget`.

**Paper wording.** For each participant, the observed disagreement was split in quadrature into
the propagated reference uncertainty and the part attributable to the wearable system, assuming
independent errors; where the reference uncertainty reached the observed disagreement, the
wearable system's part was reported as not resolvable. For the digitization, candidate causes
of the difference in frame error between the two participants (calibration-tool orientation
disagreement, geometry of the Qualisys head cluster, corrected marker-label swaps, reference
uncertainty of the tooth points) were tabulated side by side.

### EMG side check against Delsys (VD23)

Stage `ref_emg_side`, `validation/emg_side.py`; descriptive. Read from the stored `ref_emg`
arrays only. Per participant, profile (task, and side of the movement in `left_right`) and
muscle pair (`pairs`: masseter, temporalis, digastric), with $\bar y^J_a$ and $\bar y^D_b$ the
ensemble-mean cycle profiles (% of each system's own maximum, cycles with a masked point
dropped as in D5, ≥ `min_cycles` = 2 cycles per system and side) of JTS-MM side $a$ and Delsys
side $b$, the four shape correlations $r(a, b) = \mathrm{Pearson}(\bar y^J_a, \bar y^D_b)$,
$a, b \in \{R, L\}$, and the side index

$$
\mathrm{SI} = \tfrac12\big[r(R,R) + r(L,L) - r(R,L) - r(L,R)\big] \in [-2, 2]:
$$

+2 the sides as in the code, −2 swapped, ≈ 0 not identifiable. Only tasks in which the two
sides act differently are informative (`side_tasks`: `left_right`; column `side_informative`).
Table `side_check` (columns `n_jtsmm_*`, `n_delsys_*`, `r_rr`, `r_ll`, `r_rl`, `r_lr`,
`side_index`, `reason`). Values of `v15` (VD23, change
log): the sides are confirmed where the data can tell them apart, and never clearly swapped
(left_right, side index left / right side of the movement): P1 temporalis
`\numSideIndexTemporalisLeftPOne` / `\numSideIndexTemporalisRightPOne` (1.60 / 1.63),
P2 masseter `\numSideIndexMasseterLeftPTwo` / `\numSideIndexMasseterRightPTwo` (1.71 / 1.67),
P2 temporalis `\numSideIndexTemporalisLeftPTwo` / `\numSideIndexTemporalisRightPTwo` (0.66 / 0.73).
P1 masseter is not identifiable (`\numSideIndexMasseterLeftPOne` / `\numSideIndexMasseterRightPOne`
(-0.09 / 0.02), ≈ 0: both sides co-activate in that participant's left_right task), so its side
rests on P2 and on the Delsys comparison (corrected 2026-11-02 from "masseter and temporalis
sides confirmed");
the digastrics co-activate
(index ≈ 0), so their side is not verifiable from the data. The cable-label check on the
hardware (the authors, 2026-11-03) confirmed the closers and found the digastrics swapped: CH3 is the
right, CH4 the left digastric (an03 §1). The code and the readers were corrected the same day;
before the correction the digastric values of `ref_emg_side` and of the JTS-MM ↔ Delsys
digastric pairing were computed with the swapped names (EV12).

- **Parameters:** `stages.ref_emg_side`: `pairs` = [masseter, temporalis, digastric],
  `min_cycles` = 2, `side_tasks` = [left_right].
- **Code:** `validation/emg_side.py:side_index`, `side_check`, `run_ref_emg_side`.

**Paper wording.** The side assignment of the wearable EMG montage was checked against the
side-labelled Delsys sensors: in the laterotrusion task, where the two sides act differently, a
side index was computed from the shape correlations of each wearable channel's mean profile
with the same-side and the opposite-side Delsys profile (+2 sides as assigned, −2 swapped,
≈ 0 not identifiable).

## 11d. J14: integration of the earlier evaluation of the recording system (VD26–VD28, §11e–§11h)

Ported from the earlier evaluation of the recording system after a review of each method and
recomputed with the project's code (VD26–VD28, change log; the dataset counterpart is X11 with change-log entry J14, an20).
Every check against the original evaluation scripts is listed per section.

### 11e. Frame-trigger association under injected faults (VD26, stages `ref_assoc`, `ref_assoc_cohort`)

- **What.** The recorded camera and trigger streams of the comparison take and of the dataset
  take of each participant, replayed in host arrival order (trigger events 2 ms after their <!-- num-ok: method parameter (fault levels, replay timing) -->
  edge, frames 40 ms after their trigger) through the recorder's own <!-- num-ok: method parameter (fault levels, replay timing) -->
  `jtsmm.sync.clock_align.ClockAligner` (unchanged since the recordings), with injected faults:
  extra single camera drops (0–50 %), ten bursts of 1–60 dropped frames per camera, lost <!-- num-ok: method parameter (fault levels, replay timing) -->
  trigger events (0–10 %), Gaussian jitter on the trigger time (SD 0–4 ms) and on the sensor <!-- num-ok: method parameter (fault levels, replay timing) -->
  time stamps (SD 0–6 ms), late trigger events (0–500 ms plus an exponential share). Ten seeds <!-- num-ok: method parameter (fault levels, replay timing) -->
  per level and take; the first frame of each camera anchored to its recorded trigger (the
  start-up anchoring is not under test). Each delivered frame: correct, wrong trigger, or
  unassigned. Code: `validation/assoc_faults.py`.
- **Checks.** Without faults the replay reproduces the recorded assignment of every frame
  (`\numFaultCleanCorrect` (100 %)); without the cadence step `\numFaultCadenceOffWrong` (100 %) of cam0's frames are assigned to a
  wrong trigger (cam0 delivers its first frames before the start-up pause), as in the earlier
  evaluation;
  synthetic streams in `tests/test_r8_j14_acquisition.py`.
- **Result** (`\numFaultTakes` (4) takes, `\numFaultSeeds` (10) seeds). No wrong assignment with up to `\numFaultDropTol` (50 %) extra drops,
  `\numFaultLossTol` (10 %) lost trigger events (their frames stay unassigned, by design) or trigger events
  `\numFaultDelayTol` (500 ms) late. Bursts up to `\numFaultBurstTol` (20) frames are tolerated; from `\numFaultBurstFail` (29) frames on (the
  gap reaches the 30 periods the cadence step bridges) `\numFaultBurstWrong` (87 %) of the frames are
  assigned wrongly on average and the offset persists. Jitter on the trigger time is tolerated
  up to SD `\numFaultTrigJitTol` (2.0 ms) and on the sensor time stamps up to SD `\numFaultSensJitTol` (1 ms). All limits are far
  above the recorded conditions (longest burst 4 frames, an20; trigger period jitter a few µs).
- **Against the earlier evaluation** (other takes). Same limits for drops, lost and late
  events, bursts (≤ 20 tolerated) and sensor jitter (1 ms); the trigger-jitter limit was <!-- num-ok: comparison with the earlier evaluation, verification record -->
  0.5 ms there (single runs at 1 ms failed), so this limit depends on the take: quote it as <!-- num-ok: comparison with the earlier evaluation, verification record -->
  "at least 0.5 ms, up to 2 ms in the comparison participants' takes". <!-- num-ok: value of the earlier evaluation, verification record -->
- **Also stored** (`ref_assoc:cameras`, `lag`): the X11 acquisition quality of both takes. Drop
  rates of the comparison takes 3.23 / 2.82 % (P1) and 3.36 / 3.03 % (P2), as in the earlier evaluation. <!-- num-ok: verification record (equal to the earlier evaluation) -->
  IMU-to-camera lag of the jaw IMU (|ω|, exposure centre) `\numImuCamLagPOne` / `\numImuCamLagPTwo` (3.4 / 4.9 ms). The fusion
  calibration's lag (`\numImuOptLagPOne` / `\numImuOptLagPTwo` (7.7 / 2.6 ms); orientation, both IMUs pooled, per held-out task) is a
  different estimator: the head IMU barely rotates and pulls the pooled value; for the
  system's IMU latency quote the |ω| lag. <!-- num-ok: drop rates as verification record -->

### 11f. Calibration repeatability (VD27, stage `ref_cal_repeat`; side note)

- **What.** Ten consecutive pivot calibrations of the calibration tool (19–20 poses
  each) and ten consecutive hand-eye calibrations of both IMU mountings, each refitted
  with the project's code: pivot least squares (residual per pose), and
  `jtsmm.imu.mounting.fit_session_slot` (Park–Martin) on each run's `session.h5`. Code:
  `validation/calrepeat.py`.
- **Result.** Tip SD over the runs `\numTipRepeatSd` (0.97 mm) (3-D), largest deviation from the mean tip `\numTipRepeatMax` (1.6 mm);
  pivot residual `\numTipRepeatRmsMin` (1.6 mm) to `\numTipRepeatRmsMax` (2.2 mm) per run, rising with the tool's tilt (Spearman
  `\numTipTiltRho` (0.32)): oblique views of the markers, amplified by the 229-mm lever to the tip. IMU
  mountings: RMS deviation from the mean `\numMountRepeatJaw` (0.9°) (jaw) and `\numMountRepeatHead` (0.8°) (head), largest
  `\numMountRepeatJawMax` (1.2°) / `\numMountRepeatHeadMax` (1.1°).
- **Against the earlier evaluation.** Pivot: identical (it used the same fit). IMU: 0.82 / 0.80° there, from <!-- num-ok: value of the earlier evaluation, verification record -->
  the runs' stored fits; refitted with today's fit code (revised since those runs) single runs move
  by up to 0.6°, the spread is 0.86 / 0.77°. <!-- num-ok: values of the earlier evaluation, verification record -->

### 11h. Single camera against stereo; the ladder scale error (VD28, stage `ref_trueness`; side note)

- **What.** The ladder and 30-s takes of §8b also evaluated with single-camera PnP (cam0
  images, the tracker's PnP; the slots are the plateaus of the triangulated positions of the
  same frames, since single-camera depth noise defeats the plateau detector at far range), in
  separate tables (`pnp_trueness`, `pnp_precision`); the paper's ladder tables and figures keep
  stereo and Qualisys. New for all systems (`ref_trueness:scale`): the scale error of the
  free-scale fit with its 95 % CI, the scale error relative to Qualisys on the same, <!-- num-ok: CI level -->
  simultaneously recorded ladder (an error of the hand-measured spacing cancels), and the
  share of the squared residual to the metric line explained by the scale error.
- **Result.** Residual to the metric line: single camera `\numBenchTruenessPnpMin` (1.4 mm) to `\numBenchTruenessPnpMax` (2.8 mm), stereo
  `\numBenchTruenessTriMin` (0.35 mm) to `\numBenchTruenessTriMax` (0.79 mm); 30-s precision along the camera axis at far range `\numBenchPrecPnpFarZ` (1.3 mm) (single
  camera) vs `\numBenchPrecTriFarZ` (0.10 mm) (stereo). The stereo scale error relative to Qualisys is `\numBenchScaleTriMin` (−1.5 %) to
  `\numBenchScaleTriMax` (−0.7 %) and explains `\numBenchScaleShareMin` (65 %) to `\numBenchScaleShareMax` (99 %) of the squared stereo residual; without it at most
  `\numBenchNoScaleTriMax` (0.35 mm) remain, close to Qualisys. Distances are measured systematically short by about 0.7 % <!-- num-ok: rounded from the macros above -->
  (a baseline calibration effect; 0.3 mm over a 40-mm opening). <!-- num-ok: derived: 0.7 % of a 40-mm opening -->
- **Against the earlier evaluation.** Stereo and PnP precision and the scale errors equal its values;
  PnP trueness 1.35–2.84 mm (earlier evaluation 1.36–2.82 mm). <!-- num-ok: values of the earlier evaluation, verification record -->

## 11i. J30: arrays, tags, head cluster, hinge floor (VD29–VD33)

Post hoc, descriptive, on `v15` without a re-run of the pipeline (audit scripts
`scripts/audit_recompute/vd29_array_noise.py`, `vd30_tag_viewing.py`, `vd31_head_cluster.py`,
`vd32_hinge_floor.py`; outputs in `<store>/audit/`, derived tables `array_noise`,
`tag_viewing`, `head_cluster`, `hinge_floor`). Definitions: VD29–VD33 and the
two VD29 / VD30 amendments (change log of the validation plan; J30 is an entry of the change
log of the dataset study). Every script reproduces the stored stage values it starts from
(refit poses with the tracker's trimmed Kabsch to ≤ 0.001° / 0.001 mm, the own-frame points <!-- num-ok: reproduction tolerance -->
exactly, the bench SDs exactly, the held-out head-plate RMS to 1e-14°, the hinge residuals <!-- num-ok: reproduction tolerance -->
exactly, `sigma_ref` exactly).
- **VD29:** lever arms, pose noise of the arrays on three scales (Monte Carlo from the corner
  residual = not a noise estimate, bench ratio k `\numBenchKTheta`; bench-calibrated = lower;
  rest jitter = upper), $\sigma_p$ at the incisor `\numSigmaPIncisorBenchPOne`–`\numSigmaPIncisorRestPOne`
  mm (P1) and `\numSigmaPIncisorBenchPTwo`–`\numSigmaPIncisorRestPTwo` mm (P2), CRA share
  `\numCraShareIncisorRestPOne` / `\numCraShareIncisorRestPTwo`; tag-model corner offsets up to
  `\numTagCornerOffsetMaxMtaPOne` mm.
- **VD30:** head tags at `\numTagIncidenceHpPOne` / `\numTagIncidenceHpPTwo`° incidence against
  `\numTagIncidenceMpPOne` / `\numTagIncidenceMpPTwo`°; rule verdict "not supported" ((iii) not
  evaluable, confounded with the image size).
- **VD31:** §3.8 (corrected).
- **VD32:** Qualisys hinge floor; best-fit residual above its P99 floor in P1 only
  (`\numHingeFloorFitRatioPOne` / `\numHingeFloorFitRatioPTwo`), condyle axis in both
  (`\numHingeFloorTmjRatioPOne` / `\numHingeFloorTmjRatioPTwo`).
- **VD33:** the comparison wording of every export (rename list `an14_r6_rename_list.md`).

## 12. Views

Paper set: `experiments/jtsmm_validation/paper_systems.yaml`, in metrology order (bench,
synchronisation, (I) point marking, (II) motion with the error budget, fitness for purpose,
(III) EMG; an13). Dashboards: the "Validation" domain of the viewer when it is opened on the
validation study.

| View | Shows |
|---|---|
| `val_sync_overview`, `val_sync_windows`, `val_sync_lag`, `val_sync_ct3d`, `val_sync_residuals` | sync diagnostics (§3) |
| `val_kin_timeseries`, `val_kin_planes`, `val_kin_orient`, `val_kin_3d` | both systems over the take, every segment coloured and labelled (§6) |
| `val_kin_aligned_timeseries` | frame-aligned result (VD3; equal value with the own-frame `val_kin_timeseries`, VD5): JTS-MM after the leave-one-task-out frame alignment vs Qualisys, with position and rotation disagreement |
| `val_fusion_timeseries`, `val_fusion_dropout`, `val_fusion_segments`, `val_fusion_tau`, `val_fusion_spectrum` | fusion estimators (F0, F0 hold, F1, F2, F2 causal, F2P, F2P causal; own frames and aligned) against Qualisys, dropout bridging of rotation and incisor (§9) |
| `val_table_segments_aligned_freemotion` | alignment sensitivity: offsets from free motion only (VD7) |
| `val_kin_waveforms`, `val_kin_agreement` | cycles with SPM bars and equivalence bands; feature bias (§7) |
| `val_robustness`, `val_decomposition` | clock/tip robustness; end-to-end vs tracking-only |
| `val_ref_quality`, `val_ref_angle`, `val_error_budget`, `val_availability` | reference quality per body with the gates, budget, availability (§5–6); `val_table_calibration`: the QTM calibration report |
| `val_tip` | CT tip estimates and decision (§4) |
| `val_marking_accuracy`, `val_marking_frames`, `val_marking_precision`, `val_tip_holds`, `val_table_marking` | point-marking accuracy (§10, VD4) |
| `val_emg_profiles`, `val_emg_timing`, `val_emg_mvc`, `val_emg_quality`, `val_emg_delta` | EMG comparison (§8a, D5) |
| `val_trueness`, `val_precision` | bench ladder trueness, precision, drift (§8b, D6) |
| `val_imu_alone`, `val_sampling`, `val_table_filtered` | IMU alone vs Qualisys (VD10), rate and filter bias with the reference bandwidth (VD9), agreement after the dataset chain (VD12) (§11) |
| `val_fusion_smoothing`, `val_table_imu_lever` | raw 40-Hz optics vs a 6-Hz low-pass (the dataset's cut-off, per finite run) vs IMU fusion vs Qualisys, error jitter (VD16, VD18); IMU lever arms (VD15) (§11a) |
| `val_imu_accel`, `val_table_imu_accel` | accelerometer vs Qualisys at the IMU site (VD17) (§11a) |
| `val_camerafree`, `val_table_camerafree` | camera-free estimators vs Qualisys next to the optics, non-inferiority to F0, drift (VD20) (§11b) |
| `val_hinge`, `val_table_hinge` | the dataset's hinge models on both systems, JTS-MM minus Qualisys (VD21) (§11b) |
| `val_budget`, `val_table_budget`, `val_table_marking_gap` | error budget per participant (reference vs JTS-MM part), the marking frame-error gap and its candidate causes (VD22) (§11b) |
| `val_emg_side`, `val_table_emg_side` | EMG side check of the montage against Delsys: side index per task and muscle pair (VD23) (§11b) |
| `val_bench`, `val_table_ladder`, `val_setup`, `val_table_methods`, `val_kin_error_ecdf`, `val_digitization`, `val_table_segments_both`, `val_table_features_both`, `val_table_emg_agreement` | bench trueness and precision; setup and clock link; methods table; per-frame error own vs aligned; digitization symmetry; own and aligned agreement side by side; EMG agreement table |
| `val_marking_combined`, `val_motion_agreement`, `val_sync_combined` | condensed paper figures (2026-11-05, `viz/views/validation_condensed.py`): point marking (landmark error, point budget, frame errors), motion agreement (per-frame error own vs aligned, incisor budget, feature bias vs bounds), clock link (lane, CT visibility and edges, start-window speed, residual lag); the source views stay |
| `val_table_integrity`, `val_table_emg_bench`, `val_table_fitness` | recording integrity per take (dropped frames, assignment drift, valid poses; sEMG / IMU sample loss not stored), the sEMG front end on the bench per channel (B1, B2), fitness for purpose in one table (hinge models, camera-free opening, sampling chain) |
| `val_table_*` | the stored tables (`val_table_emg_<table>` and `val_table_trueness_<table>` for `ref_emg` / `ref_trueness`) |

