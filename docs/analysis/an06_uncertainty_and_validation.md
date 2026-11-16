<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an06 — Uncertainty and validation

How measurement uncertainty is quantified and propagated in the dataset pipeline, and how it
enters the claims: the Monte-Carlo noise nulls of the hinge residual (C1), the in-vivo noise
floor, the observed-residual CI, and the cited Qualisys check. The hinge models themselves
(fits, residual definitions) are in an02 §6; the statistical primitives used here (Monte-Carlo
p, signed-rank test of log ratios, BCa CI) are defined once in an07 (§1, §2.1, §3.1, §4.3,
§14). The reference-system validation proper is the systems study (an14). Symbols are listed
in the Notation table below; rigid-body notation follows an02 §0.

**Abbreviations.** JTS-MM = the open-source multimodal jaw tracking system of this project;
EMG = electromyography; IMU = inertial measurement unit; TMJ = temporomandibular joint; BCa =
bias-corrected and accelerated (bootstrap); CI = confidence interval; RMS(E) = root mean square
(error); SD = standard deviation; P99 = 99th percentile; MAD = median absolute deviation.

## Notation

| Symbol | Meaning |
|---|---|
| $T_t = \begin{bmatrix} R_t & \mathbf t_t \\ 0 & 1\end{bmatrix}$ | mandibular pose in the maxillary frame at frame $t$ (SE(3)); $T^{\text{raw}}$ unsmoothed, $T$ low-passed |
| $\theta_t$ | measured opening angle $\theta_y$ (an02 §3) at frame $t$ of an opening, relative to its start pose $T_0$ |
| $(\hat{\mathbf n}, \mathbf s)$ | hinge axis: unit direction and a point on it |
| $\operatorname{Exp}(\mathbf r)$ | rotation matrix of the rotation vector $\mathbf r$ (Rodrigues; an02 §0) |
| $P_\ell$ | hinge points in the mandibular frame, $\ell \in$ {incisor, left molar, right molar, left condyle, right condyle} (`HINGE_POINTS`) |
| $F$ | stacked frames of the openings used by a fit (≤ `max_frames_per_cycle` = 30 per opening after its start) |
| $r^{(M)}$ | incisor RMS residual of hinge model $M$ (`hinge_fit`, `hinge_tmj`, `rot_trans`, `hinge_teeth`) |
| $r^*_b$, $b = 1..N$ | the same residual on null simulation $b$; $N$ = `stages.uncertainty.n_sims` = 1000 |
| $\hat q_{0.99}$ | P99 of the finite null draws, linear interpolation (`numpy.percentile`) |
| $\boldsymbol\epsilon^{\mathrm{rot}}_t$, $\boldsymbol\epsilon^{\mathrm{tr}}_t$ | pose noise: rotation vector (deg) and translation (mm) per frame (the rest residuals of an02 §4) |
| $f_s$ | pose sampling rate (40 Hz) |
| $w$ | odd window length in frames, $w$ = `odd_window(h, f_s)` $=2\lfloor h f_s/2\rfloor+1$ (an02 §1; 41 frames for $h$ = 1 s) |
| $s$ | participant seed, `participant_seed(seed, pid)` (an07, Seeds) |

## 1. Reference validation (G, stage `validation`)

**Data:** participants recorded with Qualisys and Delsys (cv05 block B5) carry
`flags: [reference_subset]` in `cohort.yaml`.

**Analysis** (implemented once data exist):
- clock registration with `docs/calibration_validation/cv05_sync_estimation.py`;
- agreement of the paper's own outcomes: incisal opening, ROM and hinge residual against Qualisys
  (RMSE, Bland–Altman); EMG envelope correlation and onset differences against Delsys.

**Status:** no dataset participant is flagged, so this stage reports `no_reference_data`
(`stats/cohort_misc.py:validation_status`) and the backup below applies within the dataset
study. The comparison against Qualisys and Delsys is the separate **validation study**
(`experiments/jtsmm_validation/`, two participants, an14, the systems paper); the dataset paper
cites it for the accuracy of JTS-MM and never pools its results.

## 2. Backup G′ (always run)

| Item | Where | What it gives |
|---|---|---|
| G′1 Monte Carlo: residual of a perfect hinge under bench and in-vivo noise vs the observed residual with a CI (§3); D24 / D25 variants and the D33 condyle-hinge nulls | stage `uncertainty`, `uncertainty/montecarlo.py`; cohort `stats_kinematics.hinge_noise_tests` | per participant "above the null's P99", ratio and Monte-Carlo p; cohort k of n and a signed-rank test of log(observed / P99) (an07 §14) |
| G′2 In-vivo stability floor (quietest 1 s of rest, `floor_incisor_rms_mm_smooth`, an02 §4; the null resamples the rest residuals, E28; §3a) | `kinematics.anatomy` (`kinematics/clinical.py:noise_floor`) | the participant's own pose fluctuation at rest |
| G′3 Step-ladder trueness | the systems study (an14 §8, D6) | trueness and precision of both systems on the 20-mm ladder |
| G′4 Cross-modal validity (IMU vs optical angle; onset sources) | `imu`, `timing` | angle RMSE per estimator (an04 §4); lead per mechanical source (an05 §2, §3) |
| G′5 EMG onset robustness (3 detectors × 4 mechanical sources × 3 onset rules, specification curve) | `stats_timing.timing_spec`, `variability.spec_curve` | spread / between-participant SD of the opening lead (36 variants; an07 §11) |
| G′6 Qualisys check of the hinge models (VD21, §4) | the systems study, stage `ref_hinge` | how much of the C1 residual is anatomy and how much is JTS-MM tracking error (cited, n = 2) |

Values (paper store `r17a`, 13 adults): above the bench noise in 13 of 13 and above
the in-vivo noise in 11 of 13 (median ratio 1.18 [1.09, 1.46], cohort p 0.009; `r17`, n = 14:
12 of 14, 1.20); specification-curve spread / SD of the open_close opening lead 3.64 (36
variants).

**Rule for claims:** C1 and C2 are claimed only where they exceed the propagated noise floor.

**How uncertainty enters the tests.**
- *C1, per participant:* the decision is a comparison with the null distribution of the same
  statistic under measurement noise, $r_{\text{obs}} > \hat q_{0.99}$ (§3); the noise is
  propagated through the whole chain (pose noise → low-pass → frame subsample → hinge refit),
  not added to the residual afterwards. The Monte-Carlo p and the ratio
  $r_{\text{obs}}/\hat q_{0.99}$ grade the decision.
- *C1, cohort:* the count k of n and the signed-rank test of $\log(r_{\text{obs}}/\hat q_{0.99})$ treat each participant's null-normalized residual as one observation (an07 §14).
- *C2:* the chance null of the lead (window bias, an05 §3.0) and the C2-E surrogate nulls
  (an05 §3.3) play the role of the noise floor for timing: a lead counts only against its
  null.
- *Observed-residual CI* (§3, landmark and cycle uncertainty): reported next to the
  residual; it is not part of the decision rule.
- *Design values:* sampling uncertainty of the cohort percentiles by order-statistic bounds
  (an07 §13).

## 3. G′1 Monte Carlo (`uncertainty/montecarlo.py`, stage `uncertainty`)

**What and why.** A rigid-body hinge fitted to noisy poses never has zero residual: tracking
noise alone produces a residual. The null distribution of the residual under "the jaw is a
perfect hinge, measured with this participant's noise" tells how large a residual noise alone
produces; only a residual above it is evidence of non-hinge motion.

Per participant, on the same openings and hinge data as the kinematics stage (an02 §6: `ok`
open_close cycles (`kinematics/coupling.py:openings`; an opening needs ≥ 5 valid frames and a
valid start), cycle parameters from `stages.kinematics.cycles`, ≤
`hinge.max_frames_per_cycle` = 30 frames per opening after the start, spread evenly over its
valid frames).

**Observed:** `obs_<model>_mm` = incisor RMS of `hinge_fit`, `hinge_tmj`, `rot_trans` and the
D25 teeth-only `hinge_teeth` (`hinge_fits`, an02 §6):

$$
r^{(M)}_{\text{obs}} = \sqrt{\frac{1}{F}\sum_{f=1}^{F}\big\lVert \hat{\mathbf p}^{(M)}_f - \mathbf p_f\big\rVert^2},
$$

$\mathbf p_f$ the incisor position at stacked frame $f$ and $\hat{\mathbf p}^{(M)}_f$ the
model's prediction from the opening's start pose.

**Null H0, a perfect hinge** (`null_distributions`):
- *Ideal poses:* per opening, with the **measured** opening angle $\theta_t$ (radians) and the
  fitted `hinge_fit` axis $(\hat{\mathbf n}, \mathbf s)$,

  $$
  T^{\text{ideal}}_t = H(\theta_t)\,T_0,\qquad
  H(\theta) = \begin{bmatrix} R(\theta) & \mathbf s - R(\theta)\mathbf s \\ 0 & 1 \end{bmatrix},\qquad
  R(\theta) = \operatorname{Exp}(\theta\,\hat{\mathbf n}),
  $$

  $T_0$ the measured start pose. Frames with NaN θ (untracked) stay NaN; an opening whose start
  pose is invalid gives no ideal poses.
- *Noise* (`_noisy`), left-multiplied per frame, $T'_t = \delta T_t\,T_t$:

  $$
  R'_t = \operatorname{Exp}(\boldsymbol\epsilon^{\mathrm{rot}}_t)\,R_t,\qquad
  \mathbf t'_t = \operatorname{Exp}(\boldsymbol\epsilon^{\mathrm{rot}}_t)\,\mathbf t_t + \boldsymbol\epsilon^{\mathrm{tr}}_t .
  $$

  Bench: white, $\boldsymbol\epsilon^{\mathrm{rot}}_t \sim \mathcal N(0, \sigma_r^2 I_3)$,
  $\boldsymbol\epsilon^{\mathrm{tr}}_t \sim \mathcal N(0, \sigma_t^2 I_3)$ independently per frame. In
  vivo (E28): $(\boldsymbol\epsilon^{\mathrm{rot}}_t, \boldsymbol\epsilon^{\mathrm{tr}}_t)$ is a block bootstrap of
  the participant's own rest residuals (`ResidualNoise`, below).
- *Padding and low-pass (E19):* the measured poses were low-passed as long continuous segments
  of the whole phase (an02 §1), so no opening sits at a filter edge. The null matches this:
  each ideal opening is built with `PAD_S` = 1 s (round(1 s·$f_s$) frames) of hinge motion
  continued on both sides, θ reflected evenly about the first and last frame (`np.pad`,
  `mode="symmetric"`, repeated if the opening is shorter than the pad; start and peak are
  turning points, so position and velocity stay continuous). Noise is added to every frame
  (padding included), the padded sequence is low-passed once with `se3.lowpass` (same
  `signals.kinematics.lowpass_hz` and `lowpass_order` as the data), and only then is the
  opening cut out. Formerly each bare opening was filtered as its own segment, which left
  openings shorter than 15 frames unfiltered and put edge transients into all.
- *Guards (A3):* no opening, or fewer than `MIN_FRAMES` = 10 valid frames in total → all
  `n_sims` draws NaN; a draw whose stacked frames are fewer than 10 stays NaN.
- *Refit:* the same frame subsample, then **`hinge_fit` only**, started at the true
  $(\hat{\mathbf n}, \mathbf s)$; the incisor RMS is one draw $r^*_b$.
  `uncertainty.n_sims` = **1000** draws per noise level (D24; was 100, whose P99 carried a large
  Monte-Carlo error). All noise draws are made first (simulation, then opening), then every
  opening is low-passed for all simulations at once (`se3.lowpass_batch`, identical to
  filtering per draw).

**Noise levels:**

| Tag | H0 hinge axis | Noise | Compared with | Plan |
|---|---|---|---|---|
| `bench` | best-fit (`hinge_fit`) | white: $\sigma_t = 0.182/\sqrt3$ mm and $\sigma_r = 0.126°/\sqrt3$ per axis | `obs_hinge_fit_mm` | pre-specified |
| `invivo` | best-fit | the participant's rest residuals (1-s running-median high-pass), resampled in 1-s blocks (E28) | `obs_hinge_fit_mm` | **pre-specified primary** |
| `invivo_slow` | best-fit | rest residuals high-passed by a 3-s running median (`SLOW_HP_S`), resampled in 3-s blocks: noise down to about 0.3 Hz kept | `obs_hinge_fit_mm` | D24 sensitivity (`slow_noise: true`) |
| `invivo_motion` | best-fit | the 1-s rest residuals × `motion_noise_scale` = 1.78 | `obs_hinge_fit_mm` | D24 sensitivity |
| `invivo_teeth` | best-fit, refitted with the teeth-only hinge | the same draws as `invivo` | `obs_hinge_teeth_mm` | D25 sensitivity (`teeth_null: true`) |
| `bench_tmj` | through the two condyle points | bench, as `bench` | `obs_hinge_tmj_mm` | D33, post hoc (`tmj_null: true`) |
| `invivo_tmj` | through the two condyle points | in vivo, as `invivo` | `obs_hinge_tmj_mm` | D33, post hoc |

- Bench values (`BENCH_T_MM` = 0.182 mm, `BENCH_R_DEG` = 0.126°) are the 3-D precisions from
  the bench (IEEE SENSORS 2025), split evenly over the three axes ($\sigma_{\text{axis}} = \sigma_{3D}/\sqrt3$, so that $\mathbb E\lVert\boldsymbol\epsilon\rVert^2 = \sigma_{3D}^2$).
- **In-vivo residuals** (E28; `kinematics/clinical.py:rest_residuals`) from the **unsmoothed**
  rest poses $T^{\text{raw}}_k$ (valid frames of the rest phase), in the same
  left-multiplied parametrization as the injected noise: with $\bar R$ the rotation mean and
  $\bar{\mathbf t}$ the mean translation,

  $$
  \delta R_k = R_k\bar R^{\top},\qquad
  \mathbf e_k = \big[\operatorname{rotvec}(\delta R_k)\ (\text{deg}),\ \ \mathbf t_k - \delta R_k\,\bar{\mathbf t}\ (\text{mm})\big] \in \mathbb R^6,
  $$

  and per run of consecutive tracked frames (runs shorter than $w$ dropped) the centred running
  median is subtracted, $\tilde{\mathbf e}_k = \mathbf e_k - \operatorname{med}_{\lvert j-k\rvert \le (w-1)/2}\,\mathbf e_j$ (`scipy.ndimage.median_filter`, edges `nearest`), $w$ =
  `odd_window(REST_HP_S = 1 s, f_s)`. Slow real movement (swallowing, posture drift) is
  removed while the fast, temporally correlated fluctuation stays.
- **Block bootstrap** (`ResidualNoise.draw`): blocks of $w$ consecutive residual frames,
  each from a run chosen with probability proportional to its length and a uniform start
  (runs shorter than $w$ used whole), concatenated until the opening (with padding) is
  covered and cut to length; every drawn residual × `scale` (1, or `motion_noise_scale` for
  `invivo_motion`; the random numbers do not depend on it). The null so has the measured size
  *and* correlation (lag-1 autocorrelation 0.6–0.8 at 40 Hz; white noise would make the null
  too narrow). The `floor_*_sd_*` columns report the residuals' per-axis SD (ddof 1;
  `pose_noise_residual`).
- **Motion-noise scale (D24 addendum).** 1.78 is the largest of the four motion-to-rest ratios
  of the system-to-reference difference in the systems study (`ref_kinematics:segments`,
  movement pooled vs rest: incisor 1.25 / 1.46, rotation 1.30 / 1.78, P1 / P2), a cited
  constant (`uncertainty.motion_noise_scale`; null = not computed). The ratio contains
  Qualisys noise and pose-dependent error, so the variant is conservative.
- **Slow and motion variants are conservative beyond measurement noise (D33).** They resample
  the participant's rest-pose fluctuation, whose P99 (0.4–2.8 mm across participants, `r17a`) is far
  above the bench tracking precision (0.13 mm P99): mostly physiological motion of the resting
  jaw, which the motion variant multiplies further.
- **Condyle-hinge nulls (D33, post hoc; `condyle_axis`, `null_distributions(...,
  through_condyles=True)`).** The pre-specified C1 outcome is the residual of the *best-fit*
  axis, which may sit anywhere (12–47 mm from the condyle points in the Qualisys takes, VD21),
  where a rotation reproduces most of the forward condylar translation: it tests "no fixed
  axis at all". The anatomically specific form tests "not a hinge at the TMJ": H0 = a perfect
  hinge about the axis through the condyle points. With $\mathbf c_L$, $\mathbf c_R$ the
  condyle points at each opening's closed (start) pose, averaged over the $F$ stacked frames of
  the hinge data (`hinge_data`, so each opening weighs by its picked frames, exactly as
  `hinge_through` gets the axis of the observed `hinge_tmj`, an02 §6),

  $$
  \hat{\mathbf n}_{\text{TMJ}} = \frac{\mathbf c_L - \mathbf c_R}{\lVert\mathbf c_L - \mathbf c_R\rVert},\qquad
  \mathbf s_{\text{TMJ}} = \tfrac12(\mathbf c_L + \mathbf c_R),
  $$

  directed left minus right like the maxillary $+y$ axis, so a positive measured opening angle
  rotates the ideal jaw open as in the data (E38; before that fix the axis was right minus
  left and the ideal motion closed the jaw, a mirrored geometry under anisotropic noise),

  rebuilt with the measured opening angles, the same bench / primary in-vivo noise, padding
  and filter; each draw refitted about the fixed axis through the (noisy) condyle points with
  only the angles free (`hinge_through`), compared with the observed `hinge_tmj` residual. No
  condyle-hinge null without usable hinge data (no opening with a valid start and ≥ 3 valid
  frames).
- **Without usable rest residuals.** Since E28 the primary in-vivo null is block-resampled
  from the 1-s rest residuals (`rest_noise`, `montecarlo.py:337-348`). The stored per-axis
  floor SDs (`floor_trans_sd_mm_raw_{x,y,z}`, `floor_rot_sd_deg_raw_{x,y,z}`, §3a) come from
  the same 1-s residuals (`clinical.pose_noise_residual`, `kinematics/clinical.py:283-289`), so
  they are no independent fallback: without usable 1-s rest residuals the floor SDs are NaN,
  the white-noise draws of `invivo`, `invivo_teeth`, `invivo_motion` (floor × motion factor)
  and `invivo_tmj` are NaN, and the participant counts as untested for those nulls (no
  in-vivo null; `above_*` False, `p_mc_*` NaN). Only `invivo_slow` genuinely falls back to
  white noise of the floor SDs, when the 1-s residuals exist but no rest run reaches the 3-s
  window (`SLOW_HP_S`). With no rest phase at all `kinematics` writes no `floor_*` columns
  (`kinematics/analysis.py:126-128`); the `uncertainty` stage then takes NaN floors and the
  participant counts as untested (`stages.py:232-235`).
- History: before E28 the SD of the quietest 1 s was used (a minimum over windows, biased
  low); a first-difference estimator tried next assumed white noise and fell below the bench
  precision in the quietest participants, so it was withdrawn before reporting.

**Decision per tag** (`summarize`; the observed model per tag as in the table, `model_of`):

$$
\texttt{above} = \mathbb 1\big[r_{\text{obs}} > \hat q_{0.99}\big],\qquad
\texttt{ratio} = \frac{r_{\text{obs}}}{\hat q_{0.99}}\ (\text{NaN if } \hat q_{0.99} \le 0),\qquad
p_{\text{MC}} = \frac{1 + \#\{b : r^*_b \ge r_{\text{obs}}\}}{1 + N},
$$

over the $N$ finite draws (`n_sims_<tag>`); $\hat q_{0.99}$ is the linear-interpolation P99
(position $0.99(N-1)$ of the sorted draws: between the 990th and 991st of 1000), so it rests on
about the 10 largest draws. `above_<tag>` is the pre-specified rule; `null_<tag>_p50_mm` and
`_p99_mm` are stored; $p_{\text{MC}}$ (D24) is never 0, its smallest value is 1/1001, and its
Monte-Carlo SE is about $\sqrt{p(1-p)/N}$ (an07 §3.1). Without any finite draw: `null_*`,
`ratio_*`, `p_mc_*` NaN, `above_*` False.

**Paper wording.** "For each participant, a perfect hinge was simulated with the measured
opening angles about the fitted axis, perturbed frame by frame with that participant's own
pose noise at rest (block-resampled to keep its temporal correlation), low-pass filtered and
refitted like the data (1000 simulations). The observed residual was called above noise when
it exceeded the 99th percentile of the simulated residuals; the Monte-Carlo p value is
$(k+1)/(N+1)$."

**CI of the observed residual** (`observed_ci`), `n_boot` = 100 draws, seed $s + 2$, each:
- landmark perturbation: each of the five hinge points
  $P^{(b)}_\ell = P_\ell + \sigma_\ell\,\mathbf z$, $\mathbf z \sim \mathcal N(0, I_3)$, with
  $\sigma_\ell = \max(\texttt{tooth\_sd\_mm} = 2\ \text{mm},\ \mathrm{MAD}_\ell)$ for the lower
  incisor and molars and $\max(\texttt{tmj\_sd\_mm} = 5\ \text{mm},\ \mathrm{MAD}_\ell)$ for the
  condyles ($\mathrm{MAD}_\ell$ the landmark's 3-D digitization MAD, used as a per-axis SD; 0
  when unknown);
- cycle bootstrap: openings resampled with replacement;
- all three models refitted, **warm-started** from the full-data fits (`hinge_fits(A, B, init)`,
  an02 §6: `hinge_fit` from the full-data axis, `rot_trans` from the full-data n, s, u; the
  full-data fits are computed once inside `observed_ci` from the unperturbed landmarks); the
  incisor RMS recorded per model. Same optimum as a cold start, fewer optimizer steps (E15).

`obs_<model>_ci_lo/hi` = $[Q_{0.025}, Q_{0.975}]$ of the draws (percentile interval; with 100
draws each limit is an interpolation between the 3rd and 4th most extreme draws, so it is
coarse). No CI for `hinge_teeth`. With no opening or fewer than 10 stacked frames the draws
are empty (no CI columns); a resample with fewer than 10 frames is skipped.

**Paper wording.** "The uncertainty of the observed residual was obtained by perturbing the
hinge landmarks (SD 2 mm for teeth, 5 mm for the palpated condyles, at least their
digitization MAD) and resampling the openings (100 draws; percentile interval)."

**Outputs:** table `uncertainty.hinge_noise` (one row per participant: `n_openings`,
`n_frames`, `obs_*`, `null_*`, `above_*`, `ratio_*`, `p_mc_*`, `n_sims_*`, CI columns; a
participant with fewer than 2 openings or 10 frames gets a NaN row with a `note`); arrays
`null__<tag>` (the null draws of every tag) and `ci__<model>` (the CI draws). Within one
participant chain the stage reuses the pose chain of the `kinematics` unit (`ctx.memo`, an08).

**Seeds** (`hinge_nulls`): $s$ (bench), $s + 1$ (in vivo; `invivo_teeth` reuses its draws),
$s + 2$ (CI), $s + 3$ (slow), $s + 4$ (motion), $s + 5$ (bench_tmj), $s + 6$ (invivo_tmj), so
adding a variant never changes another null; $s$ is the participant seed derived from the
study `seed` (top of `etc/biomech.yaml`) and the participant id, `crc32("<seed>:<pid>") &
0x7FFFFFFF` (`pipeline.py:participant_seed`; an08 §1), so participants never share draws.

**Cohort** (`stats/cohort_kinematics.py`; formulas in an07 §14):
- `stats_kinematics.hinge_noise_summary` (the pre-specified row): n, `n_tested_bench`,
  `n_tested_invivo`, `above_bench_n`, `above_invivo_n` and `describe` (median, BCa CI) of
  `ratio_invivo`. A participant counts as tested only when its null exists (finite ratio).
- `stats_kinematics.hinge_noise_tests` (D24, D25, D33; one row per tag × group `all` /
  `plausible_condyles` (D18, D25)): `n_tested`, `above_n` (the "k of n"), the median ratio
  with its BCa CI, the one-sided Wilcoxon signed-rank test of $\log(r_{\text{obs}}/\hat q_{0.99}) > 0$ over the tested participants (`wilcoxon_log_ratio_p`, exact without zeros or ties, normal approximation otherwise; an07 §2.1,
  `stats.families.wilcoxon_greater`) and the median and largest per-participant `p_mc`.
- Reporting: the primary "above the in-vivo noise in k of n" stays; the variants are reported
  next to it. For the paper the D33 condyle-hinge rows lead C1 (block A3 of the dataset paper,
  D33), with all participants and the D18-plausible subset. Values (`r17a`, 13 adults; with
  the E38 direction fix): hinge through the condyles above its in-vivo null in 12 of 13 (median
  ratio 3.91 [2.29, 10.18], p 0.0002), above its bench null in 13 of 13 (40.9), D18-plausible
  10 of 10 (3.37, p 0.0010); best-fit hinge: primary 11 of 13 (p 0.009), slow 6 of 13 (p 0.55),
  motion 2 of 13 (p 0.99), teeth-only 11 of 13 (p 0.0085). Including the participant recorded outside the protocol
  (`r17`): condyle hinge 13 of 14 (3.91, p 0.0001).
- Calibration test (`tests/test_r10_c1_null.py`, slow): a perfect hinge with noise from the
  same `ResidualNoise` gives an approximately uniform Monte-Carlo p (KS p > 0.01, median
  0.3–0.7, P(p ≤ 0.05) < 0.15 over 80 replicates).

**Parameters** (`stages.uncertainty`): `n_sims` (1000, null draws per tag; D24), `slow_noise`
(true), `motion_noise_scale` (1.78), `teeth_null` (true), `tmj_null` (true; D33), `n_boot`
(100, CI draws), `tooth_sd_mm` (2.0), `tmj_sd_mm` (5.0). Module constants: `PAD_S` = 1.0 s,
`MIN_FRAMES` = 10, `SLOW_HP_S` = 3.0 s, `BENCH_T_MM` = 0.182, `BENCH_R_DEG` = 0.126;
`kinematics/clinical.py:REST_HP_S` = 1.0 s.

## 3a. In-vivo noise floor (G′2; `kinematics/clinical.py:noise_floor`, table `kinematics.anatomy`)

Computed on the rest phase twice, for the smoothed (`_smooth`) and the unsmoothed (`_raw`)
poses; $\mathbf p_k$ the incisor position, $w_1 = \max(\operatorname{round}(1\,\text{s}\cdot f_s), 5)$ frames:
- `rest_incisor_rms_mm_*`: $\sqrt{\frac1K\sum_k\lVert\mathbf p_k - \bar{\mathbf p}\rVert^2}$ over
  the whole rest phase (includes swallowing and posture shifts); `rest_theta_sd_deg_*` the SD
  of the opening angle;
- `floor_incisor_rms_mm_*`: the same RMS in the **quietest** window of $w_1$ frames (windows
  stepped by $\lfloor w_1/4\rfloor$, fully tracked windows only, least incisor RMS);
  `floor_theta_sd_deg_*` the angle SD in that same window (E24). A descriptive lower bound of
  rest stability (a minimum over windows, so biased low); reported, not used by the null;
- `floor_rot_sd_deg_*_{x,y,z}`, `floor_trans_sd_mm_*_{x,y,z}`: the per-axis SD (ddof 1) of the
  rest residuals $\tilde{\mathbf e}_k$ of §3 over the whole phase (`pose_noise_residual`); the
  `_raw` values are the white-noise fallback of the in-vivo nulls (§3).

**Paper wording.** "The in-vivo noise floor was the incisor RMS in the quietest 1-s window of
the rest recording; the fast pose fluctuation at rest (after removing motion slower than a
1-s running median) defined each participant's measurement-noise model."

## 4. Qualisys check of the hinge models (VD21; systems study, cited)

The dataset cannot separate anatomical non-hinge motion from its own tracking error. The
systems study fits the same two hinge models (the best-fit fixed axis and the axis through the
condyle points, an02 §6) to both systems on the same open_close and chewing openings, each
system in its own maxillary frame with three points per frame (incisor and both condyle
points; the dataset fit uses five), ≤ 30 frames per opening as in the dataset (stage
`ref_hinge`, `validation/hinge.py`; method, data rules and parameters in an14 §11b, "The
dataset's C1 hinge models on both systems"):
- **Reading:** Qualisys tracks with sub-millimetre precision, so its residual bounds the
  anatomical non-hinge motion; the JTS-MM minus Qualisys difference is what tracking error
  adds to the dataset's C1 residual. Descriptive, n = 2, per participant plus the mean (an07
  §17.5). The dataset paper cites these values (from the systems paper's snapshot), never
  re-reports them. Values (`v15`; validation plan VD21), open_close, JTS-MM / Qualisys:
  best-fit hinge residual P1 3.02 / 1.82 mm, P2 2.68 / 1.19 mm; hinge through the condyle
  points P1 8.58 / 8.57 mm, P2 5.67 / 4.61 mm (difference mean 1.34 and 0.54 mm); Qualisys
  best-fit axis 47 / 12 mm from the condyle midpoint (P1 / P2). Chewing: P1 only (P2 has no
  usable opening).

## 5. Error budget and reference uncertainty (systems study, cited; an14 §5–§6, VD22)

The decomposition of an observed system-to-reference difference into the reference's own
uncertainty and the JTS-MM part lives in the systems study, not in the dataset pipeline; it is
summarized here because the dataset paper cites it (motion-noise scale above; marking errors).
For an outcome with observed difference SD $\sigma_{\text{obs}}$ and reference uncertainty
$\sigma_{\text{ref}}$ (independent errors, `validation/quality.py:budget`):

$$
\sigma_{\text{obs}}^2 = \sigma_{\text{JTS}}^2 + \sigma_{\text{ref}}^2,\qquad
\hat\sigma_{\text{JTS}} = \sqrt{\sigma_{\text{obs}}^2 - \sigma_{\text{ref}}^2}\ \ (\text{only if } \sigma_{\text{ref}} < \sigma_{\text{obs}}),
$$

`adequate` when $\sigma_{\text{obs}}/\sigma_{\text{ref}} \ge 3$ (`adequacy_ratio`),
`ref_var_share` = $\min(\sigma_{\text{ref}}^2/\sigma_{\text{obs}}^2, 1)$, `separable` =
$\sigma_{\text{ref}} < \sigma_{\text{obs}}$; otherwise `sigma_jts_corrected` is NaN, not 0
(`validation/budget.py:error_budget`, table `ref_budget:budget`). Descriptive, n = 2. In the
condyle row both $\sigma_{\text{obs}}$ and $\sigma_{\text{ref}}$ pool the two condyles (an14 §6.3;
erratum EV13).

**Paper wording (systems).** "The observed difference was decomposed in quadrature into the
reference uncertainty and the JTS-MM error, $\sigma_{\text{obs}}^2 = \sigma_{\text{JTS}}^2 + \sigma_{\text{ref}}^2$; the JTS-MM part was resolved only where the reference uncertainty was
smaller than the observed difference."
