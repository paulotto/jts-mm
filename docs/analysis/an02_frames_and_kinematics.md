<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an02 — Frames, landmarks and jaw kinematics

Stage `kinematics` (`biomech/kinematics/`, `biomech/frames/`), its cohort statistics in stage
`stats_kinematics` (`biomech/stats/cohort_kinematics.py`, `biomech/norms/`), and the kinematic
inputs of stage `requirements`. All positions are in mm, all angles in degrees (radians inside
formulas unless stated), frames at the optical trigger rate $f_s$ (40 Hz). This doc is the home
of the rigid-body primitives the other docs link to (§0). Statistics primitives (BCa bootstrap
CIs, Wilcoxon, Monte-Carlo p-values, ICC) live in an07; uncertainty propagation and the hinge
noise nulls in full in an06 §3 (Monte Carlo).

Every computation below gives: what and why, the math, the parameters in force, the code
pointer and a **Paper wording** block (plain Methods text, past tense, no plan codes). Symbols
are listed in §0.

**Abbreviations.** EMG = electromyography; IMU = inertial measurement unit; BCa =
bias-corrected and accelerated (bootstrap); CI = confidence interval; ICC = intraclass
correlation coefficient; RMS = root mean square; SD = standard deviation; QC = quality control;
TMJ = temporomandibular joint; FHA = finite helical axis (§5); MIO = maximal incisal opening
(§4); HP, MP, CT = headpiece, mandibular marker array and calibration tool (§0).

## 0. Notation and rigid-body primitives (`kinematics/se3.py`, `kinematics/derivatives.py`)

| Symbol | Meaning |
|---|---|
| W | camera (world) frame of the optical tracker |
| HP | headpiece marker array (`/optical/pose_cra`), rigid with the skull |
| MP | mandibular marker array (`/optical/pose_mta`), rigid with the mandible |
| CT | calibration-tool (pointer) tip (`/optical/pose_ct`, translation = tip) |
| Max | subject-specific maxillary frame (§3): origin upper incisor, $x$ anterior, $y$ left, $z$ up |
| $T_{A\leftarrow B}=\begin{bmatrix}R&\mathbf t\\ \mathbf 0^\top&1\end{bmatrix}\in SE(3)$ | pose of frame B in frame A: $\mathbf p_A = R\,\mathbf p_B + \mathbf t$ |
| $T(t)$, $T^{\mathrm{raw}}(t)$ | $T_{\mathrm{Max}\leftarrow\mathrm{MP}}(t)$ after smoothing / before smoothing (after gap bridging) |
| $\mathbf p_r$ | landmark of role $r$ in MP coordinates (constant) |
| $\mathbf x_r(t)=T(t)\odot\mathbf p_r$ | the same point tracked in maxillary coordinates ($T\odot\mathbf p := R\mathbf p+\mathbf t$) |
| $I,\;M_L,\;M_R,\;C_L,\;C_R$ | lower incisor, lower left/right second molar, left/right skin-palpated condyle point; $C_{\mathrm{mid}}=\tfrac12(C_L+C_R)$ |
| $I^u,\;M^u_L,\;M^u_R$ | the upper-jaw counterparts (HP coordinates) |
| $[\boldsymbol\omega]_\times$ | skew matrix, $[\boldsymbol\omega]_\times\mathbf v=\boldsymbol\omega\times\mathbf v$ |
| $\operatorname{Exp}(\mathbf r)$, $\operatorname{Log}(R)$ | rotation vector ↔ rotation matrix: $\operatorname{Exp}(\phi\hat{\mathbf n})=I_3+\sin\phi[\hat{\mathbf n}]_\times+(1-\cos\phi)[\hat{\mathbf n}]_\times^2$; $\operatorname{Log}$ returns $\phi\hat{\mathbf n}$ with $\phi\in[0,\pi]$ (`scipy` `as_rotvec`) |
| $\theta_y$, $\theta_{\mathrm{tot}}$ | sagittal opening angle and total rotation angle since a reference pose (§3) |
| $\Delta(t)$, $\mathbf C(t)$ | incisor / condyle-point displacement from the reference pose (§4) |
| $\hat{\mathbf n},\;\mathbf s,\;\phi,\;\tau,\;\hat{\mathbf u}$ | axis direction, axis point, rotation angle, translation, translation direction (§5, §6) |
| $F,\;K$ | number of stacked frames and of points in a hinge fit (§6) |
| $Q_p(\cdot)$ | empirical $p$-quantile (P5, P95, …) |

**Primitives** (all vectorised over frames; a pose with any non-finite entry is invalid, `valid`):

- Inverse (`inv`): $T^{-1}=\begin{bmatrix}R^\top&-R^\top\mathbf t\\ \mathbf 0^\top&1\end{bmatrix}$.
- Composition (`compose`): $T_{A\leftarrow C}=T_{A\leftarrow B}\,T_{B\leftarrow C}$. Point
  transform (`apply`): $T\odot\mathbf p=R\mathbf p+\mathbf t$.
- Relative pose (`relative`): $T_{\mathrm{ref}\leftarrow\mathrm{body}}=T_{W\leftarrow\mathrm{ref}}^{-1}\,T_{W\leftarrow\mathrm{body}}$, invalid where either input is invalid.
- Rotation since a reference, in the parent frame (`rotvec_rel`):
  $$
  \mathbf r(t)=\operatorname{Log}\!\left(R(t)\,R_{\mathrm{ref}}^\top\right).
  $$
  $R R_{\mathrm{ref}}^\top$ is the left (spatial) increment, so $\mathbf r$ is expressed in the
  parent (maxillary) axes, not in the moving body's axes.
- Mean rotation (`scipy` `Rotation.mean`): the chordal $L_2$ mean, i.e. the unit quaternion
  $\bar q=\arg\max_{\lVert q\rVert=1}\sum_k (q^\top q_k)^2$, the principal eigenvector of
  $\sum_k q_k q_k^\top$ (Markley et al. 2007). Used for the smoothing linearisation (§1) and the
  rest residuals (§4).
- Spatial angular velocity (`derivatives.angular_velocity`, stored as `<phase>__omega`,
  deg/s): central log differences per time segment of ≥ 3 frames,
  $$
  \boldsymbol\omega_k=\frac{\operatorname{Log}\!\left(R_{k+1}R_{k-1}^\top\right)}{t_{k+1}-t_{k-1}},
  $$
  the first and last sample of a segment copy their neighbour. Segments split at untracked
  frames and at time gaps $>2.5$ sample intervals (`timebase.time_segments`).
- Linear derivatives (`derivatives.derivatives`): Savitzky–Golay, see §4 (Derivatives).

## 1. Pose chain (`kinematics/pose_chain.py:build`, `kinematics/se3.py`)

**What and why.** The mandible's pose is expressed in a skull-fixed anatomical frame, so head
motion cancels and every mandibular point can be tracked in anatomical axes.

$$
T_{\mathrm{Max}\leftarrow\mathrm{MP}}(t)=T_{\mathrm{Max}\leftarrow\mathrm{HP}}\;T_{W\leftarrow\mathrm{HP}}(t)^{-1}\;T_{W\leftarrow\mathrm{MP}}(t)
$$

with $T_{\mathrm{Max}\leftarrow\mathrm{HP}}=T_{\mathrm{HP}\leftarrow\mathrm{Max}}^{-1}$ the constant
maxillary frame of §3. $f_s = 1/\operatorname{median}(\Delta t)$ of the optical timestamps.

1. **Relative pose** (`se3.relative`): headpiece (`pose_cra`) to mandibular array (`pose_mta`);
   NaN where either body is untracked.
2. **Gap bridging** (`se3.bridge_gaps` → `jtsmm.tools.run_jts_analysis.interpolate_se3_gaps`):
   interior gaps of at most $N_{\mathrm{gap}}=\operatorname{round}(\texttt{max\_gap\_s}\cdot f_s)$
   frames (0.5 s = 20 frames) between valid poses $a<b$ (valid = finite with
   $\lvert\det R\rvert>10^{-6}$) are filled, for $a<k<b$,
   $$
   \mathbf t_k=\mathbf t_a+\lambda(\mathbf t_b-\mathbf t_a),\qquad
     R_k=R_a\operatorname{Exp}\!\left(\lambda\operatorname{Log}(R_a^\top R_b)\right),\qquad
     \lambda=\frac{k-a}{b-a}
   $$
   (linear translation, Slerp rotation). Longer gaps and gaps at either end stay NaN.
   - The bridged mask is kept in memory (`JawMotion.bridged`) but is **not stored and not used
     by any flag**. Bridged frames count as tracked everywhere downstream: the cycle `untracked`
     flag (§4) only sees gaps > 0.5 s, and the `raw` noise floor (§4) includes bridged frames.
3. **Smoothing** (`se3.lowpass`): a digital Butterworth low-pass of order $N$ =
   `lowpass_order` = 2 at $f_c$ = `lowpass_hz` = 6 Hz (`scipy.signal.butter`, SOS form),
   applied forward–backward (`sosfiltfilt`, default odd-extension padding): zero phase, effective
   order 4, magnitude
   $$
   \lvert H_{\mathrm{eff}}(f)\rvert=\lvert H(f)\rvert^2=\frac{1}{1+\left(\dfrac{\tan(\pi f/f_s)}{\tan(\pi f_c/f_s)}\right)^{2N}},
   $$
   i.e. −6 dB at 6 Hz and −3 dB at 4.94 Hz for $f_s$ = 40 Hz.
   - Per contiguous valid segment; segments shorter than `min_len` = 15 frames are left
     unfiltered.
   - Translation $\mathbf t(t)$ is filtered directly, per axis.
   - Rotation is filtered in the tangent space at the segment's mean rotation $\bar R$ (§0):
     $\mathbf r(t)=\operatorname{Log}(\bar R^\top R(t))$, filter each component, map back as
     $R'(t)=\bar R\operatorname{Exp}(\mathbf r'(t))$. Within a jaw-motion segment the rotations
     stay within a few tens of degrees of $\bar R$, where the rotation-vector chart is nearly
     linear.
   - `T_raw` = poses after bridging, before smoothing; `T` = smoothed. Every metric uses `T`
     unless stated otherwise.

Parameters (`signals.kinematics` in `etc/biomech.yaml`): `max_gap_s` = 0.5, `lowpass_hz` = 6.0,
`lowpass_order` = 2, plus the landmark keys of §2 (`hold_trim`, `min_condyle_incisor_mm`,
`min_condyle_superior_mm`, `min_condyle_posterior_mm`). Defaults identical in
`pose_chain.KinParams`. Cut-off sensitivity: study variants `lowpass_8hz`, `lowpass_10hz`,
`lowpass_12hz` (`experiments/icorr2027/study.yaml`, `signals.kinematics.lowpass_hz`).

**Filter-phase policy.** Every offline filter whose output feeds a timing quantity (the
EMG-to-motion lead, an05) is zero-phase or symmetric: forward–backward Butterworth
(`sosfiltfilt`, here and in an03/an05), Savitzky–Golay derivatives (§4) and centred moving means
or medians over an **odd** number of samples (`biomech.timebase.odd_window`: the odd number
$2\lfloor x/2\rfloor+1$ for $x$ = window × $f_s$; an even-length centred mean puts the extra
sample on one side and shifts its output by half a sample, E17/E26). Only the real-time chain
(causal twins and the causal closer-release detector, an05; decoding, docs/ml) is causal, by
design.

**Paper wording.** Mandibular motion was expressed relative to the skull as the pose of the
mandibular marker array in the headpiece frame, transformed into a subject-specific maxillary
frame. Tracking gaps of up to 0.5 s were bridged by linear interpolation of the translation and
spherical linear interpolation of the rotation; longer gaps were left missing. Poses were
low-pass filtered with a zero-phase second-order Butterworth filter (6 Hz, forward–backward),
the rotation as rotation vectors about the segment's mean orientation.

## 2. The eight landmarks (`frames/landmarks.py`)

Eight landmarks are digitized with the CT pointer, one hold each (2–5 s) before the first motion
phase. Left and right are the participant's (operator checklist op01).

| Phase | Landmark | Frame | Used for |
|---|---|---|---|
| `mand_point_1` | lower central incisor (incisal edge, midline) | MP | incisal point (opening, protrusion, laterotrusion), hinge fits |
| `mand_point_2` / `_3` | lower left / right second molar | MP | hinge fits, lower intermolar width |
| `max_point_1` | upper central incisor (incisal edge, midline) | HP | maxillary-frame origin |
| `max_point_2` / `_3` | upper left / right second molar | HP | maxillary-frame y axis, upper intermolar width |
| `tmj_left` / `tmj_right` | left / right condyle, **skin-palpated** | MP and HP | condylar paths, hinge through the condyles, frame and role checks, coupling |

**CT tip (pivot calibration;** `optical/tracking/pivot.py:solve_pivot_ols`,
`jtsmm/capture/ct_pivot.py`**).** `pose_ct` reports the pointer *tip*: the recorder applies a
calibrated tip offset $\mathbf d$ (pointer-body coordinates) to the pointer body pose,
$T_{\mathrm{tip}}=[R,\;R\mathbf d+\mathbf t]$ (`apply_tip_offset`). $\mathbf d$ and the fixed
pivot point $\mathbf c$ (world) come from $N\ge3$ poses pivoting about a fixed point,
$R_i\mathbf d+\mathbf t_i=\mathbf c$, solved by ordinary least squares
$$
\begin{bmatrix}R_1&-I_3\\ \vdots&\vdots\\ R_N&-I_3\end{bmatrix}\begin{bmatrix}\mathbf d\\ \mathbf c\end{bmatrix}=-\begin{bmatrix}\mathbf t_1\\ \vdots\\ \mathbf t_N\end{bmatrix},\qquad e_i=\lVert R_i\mathbf d+\mathbf t_i-\mathbf c\rVert .
$$
The worst pose is dropped and the system re-solved while $\max_i e_i>4$ mm (`MAX_PER_POSE_ERR_MM`,
at least 3 poses kept); a solve with RMS $e$ > 5 mm is not trusted (`CT_TIP_TRUST_MAX_RMS_MM`),
and a condition number of the design matrix > 100 (`MAX_COND_A`) warns of too similar
orientations. Tip verification against Qualisys: an14 §4. (The general JTS library in
`software/jaw_tracking/jts` has its own landmark-coordinate-system calibration; `biomech` does not
use it, it digitizes the landmarks itself as below.)

**Digitization** (`digitize`, `_hold`). A landmark is the median tip position in its body's
frame over the central part of the hold. With $\mathcal K$ the frames in
$[t_0+\rho\,(t_1-t_0),\;t_1-\rho\,(t_1-t_0)]$ ($\rho$ = `hold_trim` = 0.2, so the central 60 %
of the hold) where both tip and body are tracked ($\lvert\mathcal K\rvert\ge5$ needed):
$$
\mathbf q_k=R_{\mathrm{body},k}^\top\left(\mathbf t_{\mathrm{tip},k}-\mathbf t_{\mathrm{body},k}\right),\qquad
\mathbf p=\operatorname{median}_k\mathbf q_k\ \text{(per axis)},\qquad
\mathrm{MAD}_{3D}=\operatorname{median}_k\lVert\mathbf q_k-\mathbf p\rVert .
$$
$\mathrm{MAD}_{3D}$ is kept as digitization noise (`mad_mm`, used in an06). Table `landmarks`: one
row per digitized point (`name`, `landmark`, `jaw`, `description`, `role_used`, `body`, `x`, `y`,
`z`, `mad_mm`, `n` frames). The validation comparison takes (an14) use `hold="still"` for
`mand_point_1` (`_still_hold`: the longest run ≥ 1 s whose centred rolling 1-s SD stays below
1.5 mm on every axis; without one, the frames within 6 mm of the median of the phase's last
fifth), because that phase also contains the CT sync motion.

**The condyles in both frames:**
- *MP frame:* moves with the mandible, which gives the condylar paths and the hinge models.
- *HP frame* (`tmj_*_hp`, the same holds digitized against the headpiece): a skull-fixed
  reference used for the frame check and the upper-jaw role check.

**Tooth roles** (`dental_roles`, `_geometric`): the roles follow the convention of the table
and are **verified** from geometry, per jaw, against that jaw's condyle points (MP condyles for
the lower teeth, HP for the upper).
- For each tooth point $\mathbf p$: $a(\mathbf p)=\lVert\mathbf p-C_L\rVert-\lVert\mathbf p-C_R\rVert$
  (negative = closer to the left condyle) and $d(\mathbf p)=\lVert\mathbf p-C_{\mathrm{mid}}\rVert$.
- Geometric assignment: incisor $=\arg\min_{\mathbf p}\,(\lvert a(\mathbf p)\rvert-d(\mathbf p))$
  (most midline and farthest from the condyle midpoint); of the other two, the one with the
  smaller $a$ is left, the other right. It is *unambiguous* (`geometry_ok`) when
  $a(\text{left})<0<a(\text{right})$ and
  $\lvert a(\text{incisor})\rvert<\tfrac12\min(\lvert a(\text{left})\rvert,\lvert a(\text{right})\rvert)$.
- `roles_{upper,lower}_geometry_agrees` = 1 when the geometric triple equals the convention.
- **Relabelling:** the incisor is never relabelled (`*_point_1` is always the incisor, digitized
  first). Only the two molars can be swapped, and only when the convention's sides are
  unambiguously reversed: $a(\text{conv. left})>0>a(\text{conv. right})$. Then
  `roles_relabelled` = 1. Any other disagreement keeps the convention (with
  `roles_*_geometry_agrees` = 0).
- Analysis of record (`r17`, 14 participants): the geometry agrees with the convention for both
  jaws in all 14; no role was relabelled.

**Condyle plausibility** (`condyle_check`, `pose_chain.condyle_level_check`,
`pose_chain.condyle_depth_check`; applied in this order, the first failure sets the reason):
skin palpation can land far from the joint.
- **Distance:** $\min(\lVert I-C_L\rVert,\lVert I-C_R\rVert)\ge$ `min_condyle_incisor_mm` = 70 mm
  (MP frame), else `condyles_plausible` = 0, `condyles_reason` = `too_close`. Adult
  condylion–incisor distances are around 100 mm.
- **Level (E13):** both condyle points (HP, in maxillary coordinates, `frame_tmj_*_z`; origin =
  upper incisor) must satisfy $\min(z_L,z_R)\ge$ `min_condyle_superior_mm` = 5 mm, else
  `below_incisal_level`. Both levels must be finite (E24), else `condyle_level_unknown`
  (formerly `min(left, right)` ignored a NaN in the second argument only). The frame check
  `frame_superior_ok` existed before but does not enter the plausibility flag.
- **Depth (D18, E33.10):** the side-mean depth behind the upper incisor
  $-\tfrac12(x_L+x_R)$ (`frame_tmj_depth_mm`) must be ≥ `min_condyle_posterior_mm` = 40 mm, else
  `too_shallow`; a non-finite depth gives `condyle_depth_unknown`. The side mean cancels a yaw of
  the dental frame to first order. Tolerance budget: 75 mm anatomical − 15 mm anterior marking −
  2 × 10 mm scatter = 40 mm, which is also about the depth of the second molars. The 3-D distance
  check misses shallow points because the lateral offset adds to it.
- Condyle outcomes are reported for all participants and for the plausible subset
  (`stats_kinematics.condyle_sensitivity`, §7; A4, change log): 11 of 14 in the analysis of
  record `r17` (s02 `too_shallow`, s03 `too_close`, s11 `below_incisal_level`); 10 of 13 adults in
  the paper store `r17a`.
- **Occlusal pitch QC** (`occlusal_pitch_deg`, `kinematics/analysis.py:analyse`): with
  $\mathbf d=\tfrac12(M_L+M_R)-I$ at the closed reference pose in maxillary coordinates,
  $\text{pitch}=\operatorname{atan2}(d_z,-d_x)$. Near 0° when upper and lower tooth points agree;
  a large pitch points at misplaced tooth points (a tilted frame) rather than at a low condyle
  palpation.
- **Values (`r17`, n = 14):** condyle–incisor distances 70.0–123.0 mm, intercondylar distances
  104.8–150.8 mm (skin points lie lateral to the joints). s03 sits exactly at the 70 mm threshold
  (70.0 / 71.4 mm, flagged); the level check flags s11, whose condyle points lie 3–4 mm *below*
  the incisal level (occlusal pitch −12.4°, all others −10.5…10.0°: probably partly a tilted
  frame).
- The primary hinge outcome is taken at the incisor, where the palpation error does not enter.

**Paper wording.** Eight landmarks were digitized with a pivot-calibrated pointer before the
motion tasks: the lower and upper central incisors and second molars, and both condyles,
located by skin palpation. Each landmark was the median pointer-tip position, in the frame of
the respective marker array, over the central 60 % of a 2–5 s hold; the median absolute 3-D
deviation served as digitization noise. Tooth labels were verified against the condyle
geometry, and condyle points were flagged as implausible when they lay less than 70 mm from the
lower incisor, less than 5 mm above the incisal level, or on average less than 40 mm behind the
upper incisor.

## 3. Maxillary frame (`frames/maxillary.py:maxillary_frame`, `check`)

**What and why.** A subject-specific anatomical frame from the three upper teeth (HP
coordinates), so that opening, protrusion and laterotrusion are measured along anatomical axes.

| Axis | Definition |
|---|---|
| Origin | $\mathbf o=I^u$ (upper incisor) |
| $\hat{\mathbf y}$ (left) | $\hat{\mathbf y}=\dfrac{M^u_L-M^u_R}{\lVert M^u_L-M^u_R\rVert}$ |
| $\hat{\mathbf x}$ (anterior) | $\mathbf x_0=I^u-\tfrac12(M^u_L+M^u_R)$, $\hat{\mathbf x}=\dfrac{\mathbf x_0-(\mathbf x_0\cdot\hat{\mathbf y})\hat{\mathbf y}}{\lVert\cdot\rVert}$ |
| $\hat{\mathbf z}$ (up) | $\hat{\mathbf z}=\hat{\mathbf x}\times\hat{\mathbf y}$ (right-handed) |

$$
T_{\mathrm{HP}\leftarrow\mathrm{Max}}=\begin{bmatrix}\hat{\mathbf x}&\hat{\mathbf y}&\hat{\mathbf z}&\mathbf o\\0&0&0&1\end{bmatrix},\qquad T_{\mathrm{Max}\leftarrow\mathrm{HP}}=T_{\mathrm{HP}\leftarrow\mathrm{Max}}^{-1}.
$$
The $xy$ plane is the upper occlusal plane through the incisor and the two second molars.

**Checks** (`frame_*` columns of `anatomy`; HP condyles in maxillary coordinates
$(x_L,y_L,z_L)$, $(x_R,y_R,z_R)$): `lr_ok` = $[y_L>y_R]$, `posterior_ok` = $[\max(x_L,x_R)<0]$,
`superior_ok` = $[\min(z_L,z_R)>0]$, `tmj_depth_mm` = $-\tfrac12(x_L+x_R)$. A left/right label
swap flips $\hat{\mathbf y}$ and with it $\hat{\mathbf z}$, which the superior check catches. In
`r17` all 14 participants pass left/right and posterior; 13 of 14 pass superior (s11).

**Opening angle** (`clinical.theta_deg`, `se3.rotvec_rel`): with the rotation since the
reference pose in maxillary axes, $D(t)=R(t)R_{\mathrm{ref}}^\top$ and
$\mathbf r(t)=\operatorname{Log}D(t)$,
$$
\theta_y(t)=\frac{180}{\pi}\,r_y(t),\qquad \theta_{\mathrm{tot}}(t)=\frac{180}{\pi}\,\lVert\mathbf r(t)\rVert .
$$
Positive $\theta_y$ = opening (a positive rotation about $+\hat{\mathbf y}$ moves an anterior
point down). $\theta_y$ is a rotation-vector component (rotation angle times the $y$ component of
the rotation axis), not an Euler angle; for a pure sagittal rotation the two coincide.

**Paper wording.** A maxillary coordinate frame was constructed from the digitized upper
teeth: origin at the upper central incisor, the y axis from the right to the left second molar,
the x axis anterior (incisor minus molar midpoint, orthogonalized to y) and z = x × y superior.
The opening angle was the component about the maxillary y axis of the rotation vector of the
mandible's rotation relative to the reference pose.

## 4. Cycles and clinical metrics (`kinematics/cycles.py`, `clinical.py`)

**Excursion signal by task** (`clinical.find_task_cycles`; incisor $\mathbf x_I(t)$ in maxillary
coordinates):

| Task | Excursion signal $y(t)$ |
|---|---|
| open_close, chewing | $-x_{I,z}$ |
| protrusion_retrusion | $x_{I,x}$ |
| left_right | $+x_{I,y}$ (side `left`) and $-x_{I,y}$ (side `right`), detected separately |

**Cycle detection** (`cycles.detect`, keys `stages.kinematics.cycles`). A cycle is
start (rest) → peak (maximal excursion) → end (rest).
1. NaN samples are linearly interpolated for the detection only ($\ge8$ finite samples
   needed). Range $\rho=Q_{95}-Q_5$ of the finite samples of the phase (floored at $10^{-9}$).
2. Peaks: `scipy.signal.find_peaks` with prominence $\ge\pi_f\rho$ ($\pi_f$ =
   `prominence_frac` = 0.3) and a minimum distance of $\lfloor\texttt{min\_period\_s}\cdot f_s\rfloor$
   samples (0.4 s → 16 frames).
3. For peak $k$ at sample $p_k$: start $s_k=\arg\min_{[p_{k-1},\,p_k]}y$, end
   $e_k=\arg\min_{[p_k,\,p_{k+1}]}y$ (phase start/end for the first/last peak). Neighbouring
   cycles can share a minimum.
4. **Silently discarded** (no row, no flag): candidates with $s_k=p_k$ or $e_k=p_k$, and
   candidates with amplitude
   $$
   y(p_k)-\max\big(y(s_k),y(e_k)\big)<\max\!\left(\texttt{min\_amplitude},\;0.5\,\pi_f\,\rho\right).
   $$
   With `min_amplitude` = 0 the floor is 15 % of the range; the yaml comment "0 = prominence
   rule only" refers to this derived floor.
5. **Kept but flagged** (not `ok`, reason in `flags`, comma-separated):
   - `untracked`: NaN fraction over $[s,e]$ > `max_nan_frac` (0.1). NaN here means a gap
     > 0.5 s (§1; bridged frames count as tracked);
   - `short`: $(e-s)/f_s<$ `min_period_s`;
   - `edge`: $s$ is the first or $e$ the last sample of the phase;
   - `no_return` (open_close and chewing only):
     $\lvert y(s)-y(e)\rvert>\texttt{return\_frac}\cdot\big(y(p)-\min(y(s),y(e))\big)$,
     `return_frac` = 0.2.

Only `ok` cycles enter summaries, coupling, hinge fits and the IMU/timing stages. (The original
evaluation scripts of the recording system cut peak-to-peak instead,
`kinematics/peak_cycles.py`; never pool the two.)

**Reference pose:**
- open_close and chewing: the cycle's own start frame, $T_{\mathrm{ref}}=T(s)$.
- protrusion and laterotrusion: the **closed reference pose** $T_c$ (`T_closed`), because their
  cycle starts are the opposite excursion. $T_c$ is an actual recorded pose: among the starts of
  the `ok` open_close cycles with a valid pose, the one whose incisor height $x_{I,z}$ is the
  median (sorted, index $\lfloor n/2\rfloor$, the upper median for even $n$). Without any such
  start these tasks fall back to the cycle start (`reference` column: `closed` /
  `cycle_start`).

**Per-cycle metrics** (`clinical.cycle_metrics`, `cycles` table).
$\Delta(t)=\mathbf x_I(t)-\mathbf x_I^{\mathrm{ref}}$ and
$\mathbf C_{L,R}(t)=\mathbf x_{C_{L,R}}(t)-\mathbf x_{C_{L,R}}^{\mathrm{ref}}$ (MP condyle points
tracked like the incisor), $\theta_y,\theta_{\mathrm{tot}}$ relative to $T_{\mathrm{ref}}$; $p$ is
the detected peak frame, $[s,e]$ the cycle window.

| Column | Definition |
|---|---|
| `opening_mm` | $-\Delta_z(p)$ |
| `incisal_disp_mm` | $\lVert\Delta(p)\rVert$ |
| `protrusion_mm`, `lateral_mm` | $\Delta_x(p)$, $\Delta_y(p)$ (signed; right laterotrusion is negative) |
| `deviation_mm` | $\max_{t\in[s,p]}\lvert\Delta_y(t)\rvert$ |
| `retrusion_mm` | $-\min_{t\in[s,e]}\Delta_x(t)$ (protrusion_retrusion only) |
| `theta_y_deg`, `theta_total_deg` | $\theta_y(p)$, $\theta_{\mathrm{tot}}(p)$ (§3) |
| `condyle_{left,right}_disp_mm` | $\lVert\mathbf C_{L,R}(p)\rVert$ (straight-line excursion, not path length) |
| `condyle_{left,right}_ant_mm` | $C_{L,R,x}(p)$ |
| `peak_open_vel_dps` | $\max_{t\in[s,p]}\dot\theta_y(t)$ |
| `peak_close_vel_dps` | $-\min_{t\in[p,e]}\dot\theta_y(t)$ |
| `peak_incisal_speed_mm_s` | $\max_{t\in[s,e]}\lVert\dot{\mathbf x}_I(t)\rVert$ |
| `t_on_us`, `t_off_us` | movement bounds (below) |
| `duration_s`, `rise_s` | $t_e-t_s$, $t_p-t_s$ |

Also stored: `t_start_us`, `t_peak_us`, `t_end_us`, `nan_frac`, `flags`, `ok`, `reference`.

- **Derivatives** (`kinematics/derivatives.py:derivatives`): Savitzky–Golay, the local
  least-squares cubic over a centred window of $n_w$ samples, differentiated analytically:
  $$
  \hat x^{(m)}_k=\frac{m!}{\Delta t^m}\,\beta_{m,k},\qquad \boldsymbol\beta_k=\arg\min_{\boldsymbol\beta}\sum_{j=-h}^{h}\Big(x_{k+j}-\sum_{q=0}^{3}\beta_q j^q\Big)^2,\quad n_w=2h+1,
  $$
  $n_w=$ round(`WINDOW_S`·$f_s$) forced odd and ≥ 5 (`WINDOW_S` = 0.15 s, `POLYORDER` = 3: 7
  samples at 40 Hz), $\Delta t$ = median sample interval, `mode="interp"` (the ends use the
  polynomial of the first/last full window). Per finite segment (split at NaN and at time gaps
  > 2.5 $\Delta t$); segments shorter than $n_w$ (but ≥ 3) use `numpy.gradient` for the first
  derivative and NaN for the second. For the cycle metrics they are computed on the cycle window
  only ($\theta_y$ referenced as in the table, and the incisor position); cycles with ≤ 3 frames
  get NaN. The stage also stores per phase `<phase>__incisor_vel/_acc`, `__theta_vel/_acc` (from
  $\theta_y$ relative to $T_c$) and `__omega` (§0).
- **Movement bounds** (`clinical.movement_bounds`): on the cycle's incisal speed $v_k$ (NaN → 0),
  the opening speed peak $k_\uparrow=\arg\max_{[s,p]}v$ and the closing speed peak
  $k_\downarrow=\arg\max_{[p,e]}v$;
  $$
  k_{\mathrm{on}}=\max\{k\le k_\uparrow: v_k<0.05\,v_{k_\uparrow}\},\qquad k_{\mathrm{off}}=\min\{k\ge k_\downarrow: v_k<0.05\,v_{k_\downarrow}\},
  $$
  the window edges if the speed never drops that low (`frac` = 0.05, module default).

**Per-participant `summary`** (`kinematics/analysis.py:_summary`): per task, side and metric
(`SUMMARY_METRICS`) over the `ok` cycles with a finite value: `n_cycles`, `median`, `max`,
`mean`, `sd` (ddof = 1, NaN for n = 1) and `cv` = SD/mean, only when n > 1 and **all** values
> 0 (ratio-scale metrics), else NaN.

**What the protocol can and cannot measure** (Deviations D1–D3):
- `open_close` was self-paced, so it measures *self-paced opening*, not the maximal incisal
  opening (MIO). The largest opening seen in any task is a lower bound for MIO:
  $$
  \texttt{max\_observed\_opening\_mm}=x_{I,z}(T_c)-\min_{t\in\text{experiment phases}}x_{I,z}(t)
  $$
  on the smoothed poses (`max_observed_opening_task` names the phase).
- The incisor landmarks are on labial surfaces, so overbite and overjet are not measurable;
  `incisor_landmark_dx/dz_mm` (lower incisor at $T_c$, maxillary coordinates, i.e. relative to
  the upper incisor) are QC only.

**Noise floor and stability** (`clinical.noise_floor`, phase `rest`; computed for `T` (suffix
`_smooth`) and `T_raw` (`_raw`, includes bridged frames); needs at least $w$ valid rest frames,
$w=\max(\operatorname{round}(f_s),5)$ = 40 = 1 s):
- `rest_incisor_rms_mm_*` ("stability"): $\sqrt{\tfrac1K\sum_k\lVert\mathbf x_{I,k}-\bar{\mathbf x}_I\rVert^2}$
  over all valid rest frames; `rest_theta_sd_deg_*`: SD (ddof = 0) of $\theta_y$ relative to the
  first valid rest frame.
- Windows of $w$ samples, stride $\lfloor w/4\rfloor$, only fully tracked windows.
  `floor_incisor_rms_mm_*`: the minimum windowed 3-D incisor RMS (the "quietest window");
  `floor_theta_sd_deg_*`: SD of $\theta_y$ in **that same window** (E24; it used to be the minimum
  over all windows, possibly another one). Both are descriptive lower bounds of rest stability (a
  minimum over many windows is biased low as a noise estimate). The `_raw` and `_smooth` floors
  each select their own quietest window.
- **Rest residuals** (`clinical.rest_residuals`, E28; the noise the in-vivo Monte-Carlo null
  resamples, §6b and an06 §3). With $\bar R$ the mean rotation (§0) and $\bar{\mathbf t}$ the mean
  translation of the valid rest frames, the left-multiplied perturbation $T_k=\delta T_k\,\bar T$:
  $$
  \delta R_k=R_k\bar R^\top,\quad \boldsymbol\epsilon^{\mathrm{rot}}_k=\tfrac{180}{\pi}\operatorname{Log}\delta R_k,\quad \boldsymbol\epsilon^{\mathrm{tr}}_k=\mathbf t_k-\delta R_k\bar{\mathbf t},
  $$
  per run of consecutive tracked frames minus its centred running median over
  `odd_window(1 s)` = 41 frames (`scipy.ndimage.median_filter`, `mode="nearest"`; runs shorter
  than the window are dropped). This removes slow real movement (swallowing, posture drift) and
  keeps the fast, temporally correlated fluctuation (lag-1 autocorrelation 0.6–0.8 at 40 Hz).
  `floor_rot_sd_deg_*_{x,y,z}`, `floor_trans_sd_mm_*_{x,y,z}` = per-axis SD (ddof = 1) of the
  pooled residuals (`pose_noise_residual`). (The previous SD of the quietest window was a minimum
  over windows, biased low; a first-difference estimator assumed white noise and was withdrawn
  before reporting.) `pose_noise` (SD of the same perturbation without the running median) is
  kept for other callers.

**Size proxies** (`clinical.anatomy`, `anatomy` table): `upper_intermolar_mm` =
$\lVert M^u_L-M^u_R\rVert$, `lower_intermolar_mm` = $\lVert M_L-M_R\rVert$, `intercondylar_mm` =
$\lVert C_L-C_R\rVert$, `condyle_incisor_{left,right}_mm` = $\lVert I-C_{L,R}\rVert$ and
`mandibular_length_mm` = $\lVert I-C_{\mathrm{mid}}\rVert$ (MP frame). The condyle-based ones
inherit the palpation error (A4). Landmark MADs, role checks, frame checks (`frame_*`) and
`condyles_plausible` / `condyles_reason` are in the same table.

**MIO specification variants** (`spec` table, an07 §11): over the `ok` open_close cycles, the
maximum of `opening_mm` (`cycle_start|vertical|lowpass`, primary), of `incisal_disp_mm`
(`cycle_start|3d|lowpass`), of $x_{I,z}(s)-\min_{[s,e]}x_{I,z}$ on `T_raw`
(`cycle_start|vertical|raw`), and of $\operatorname{median}_{\text{rest}}x_{I,z}-\min_{[s,e]}x_{I,z}$
(`rest|vertical|lowpass`).

**Paper wording.** Movement cycles were detected on the incisal excursion (vertical for
opening and chewing, anterior for protrusion, lateral for laterotrusion) as peaks with a
prominence of at least 30 % of the 5th–95th percentile range and at least 0.4 s apart, each
bounded by the minima to the neighbouring peaks. Cycles with more than 10 % untracked samples,
shorter than 0.4 s, cut by the recording edge, or (opening and chewing) not returning to within
20 % of their amplitude were flagged and excluded from the summaries. Displacements were taken
at the peak relative to the cycle start (opening, chewing) or to a recorded closed pose
(protrusion, laterotrusion); velocities were obtained with a cubic Savitzky–Golay
differentiator (0.15 s window). Movement onset and offset were the times at which the incisal
speed fell below 5 % of the opening and closing speed peaks.

## 5. Helical axes (`kinematics/helical.py`, `coupling.fha_rows`)

**What and why.** The finite helical axis (FHA) of the mandible between two poses shows where
the momentary rotation centre of opening lies and how it migrates; a fixed hinge would give one
fixed axis.

For two poses $A,B$ (both $T_{\mathrm{Max}\leftarrow\mathrm{MP}}$), the displacement
$D=B\,A^{-1}=[R_D,\mathbf d]$ maps every mandibular point from its position at $A$ to its
position at $B$. Its screw parameters (Spoor & Veldpaus 1980; Woltring et al. 1985;
`helical.screw`):
$$
\phi\hat{\mathbf n}=\operatorname{Log}R_D,\qquad \tau=\hat{\mathbf n}\cdot\mathbf d,\qquad (I_3-R_D)\,\mathbf s=\mathbf d-\tau\hat{\mathbf n}.
$$
$(I_3-R_D)$ has rank 2 (null space $\hat{\mathbf n}$), so $\mathbf s$ is taken as the
minimum-norm least-squares solution (`numpy.linalg.lstsq`), projected orthogonal to
$\hat{\mathbf n}$: the point of the axis closest to the origin. Equivalent closed form:
$$
\mathbf s=\tfrac12\left(\mathbf d-\tau\hat{\mathbf n}\right)+\tfrac12\cot\tfrac{\phi}{2}\;\hat{\mathbf n}\times\mathbf d .
$$
Pairs with $\phi<10^{-6}$ rad or a non-finite $D$ are skipped. The axis position error scales as
pose noise / $\phi$.

- **Pairs** (`stepped_screws`): along each opening half (start → peak of an `ok` open_close
  cycle, $\theta_y$ relative to the start), a non-overlapping chain: from frame $i$, $j$ is the
  first later frame with finite $\theta_y$ and $\lvert\theta_y(j)-\theta_y(i)\rvert\ge$
  `stages.kinematics.fha.step_deg` = 5°, then $i\leftarrow j$.
- **Piercing point** with the mid-sagittal plane $y=0$ (`Screw.pierce`):
  $\mathbf s+\lambda\hat{\mathbf n}$, $\lambda=-s_y/n_y$, only if $\lvert n_y\rvert>0.5$ (as for the
  hinge axis, §6, and the geometry, §6a), else NaN.
- **`fha` table:** per pair `theta_from_deg`/`theta_to_deg`, `phi_deg`, `t_mm` ($\tau$),
  `n_x/n_y/n_z`, `pierce_x_mm`/`pierce_z_mm` and `dist_tmj_mid_mm` = 2-D $(x,z)$ distance from the
  piercing point to the condyle midpoint $\mathbf x_{C_{\mathrm{mid}}}$ at the opening's start.
- **Cohort** (`stats_kinematics.fha_subject` / `fha_summary`, `cohort_kinematics.hinge_tables`):
  per cycle the range (max − min) of `pierce_x_mm` and `pierce_z_mm` and the mean
  `dist_tmj_mid_mm`; per participant the median over cycles; cohort median with BCa CI
  (`describe`, an07 §1.1).
- **Instantaneous helical axis (animations only, no stored result;**
  `anim/layers.py:Pose.prepare`**):** from the spatial angular velocity $\boldsymbol\omega$ (§0) and
  the incisor velocity $\mathbf v_I$, direction $\hat{\mathbf n}=\boldsymbol\omega/\lVert\boldsymbol\omega\rVert$
  and axis point $\mathbf q=\mathbf x_I+\boldsymbol\omega\times\mathbf v_I/\lVert\boldsymbol\omega\rVert^2$
  (then moved along the axis to the point nearest the condyle midpoint), drawn only while
  $\lVert\boldsymbol\omega\rVert\ge$ 20°/s. This is the $\phi\to0$ limit of the FHA.

**Paper wording.** Finite helical axes of the mandible were computed (Spoor and Veldpaus) for
successive pose pairs at least 5° of opening apart along each opening, and their intersection
with the mid-sagittal plane was compared with the palpated condyle midpoint.

## 6. Hinge models (C1, `kinematics/hinge_fit.py`, `coupling.py`)

**What and why.** How far is jaw opening from rotation about a fixed hinge? A device with a
fixed hinge imposes the residual of the best fixed-axis model as misalignment at the incisor
and the condyles.

**Data** (`coupling.openings`, `hinge_data`): the opening halves of the `ok` open_close cycles
that have ≥ 5 valid frames and a valid start. $K=5$ mandibular points $\mathbf p_k$ (`HINGE_POINTS`
= incisor, left and right second molar, left and right skin-palpated condyle; MP coordinates,
each tracked with `T`). For a picked frame $f$ of opening $c$:
$$
\mathbf A_{fk}=T_c(\text{start})\odot\mathbf p_k,\qquad \mathbf B_{fk}=T_c(f)\odot\mathbf p_k\qquad(\text{maxillary coordinates}).
$$
- Per opening, of its $n$ valid frames (index 0 = start), the picked ones are
  `valid[unique(int(linspace(1, n − 1, min(max_frames, n − 1))))]`, `max_frames` =
  `stages.kinematics.hinge.max_frames_per_cycle` = 30 (an even subsample, so long openings do
  not dominate). Openings with < 3 valid frames are skipped.
- The fits run only with $F\ge10$ stacked frames in total (`analyse`).

**Models** (`hinge_fits`):

| Model | Fitted parameters |
|---|---|
| `hinge_fit` | fixed axis $(\hat{\mathbf n},\mathbf s)$ over all cycles, plus one angle per frame |
| `hinge_tmj` | axis through the mean start positions of the two condyle points, $\hat{\mathbf n}=\operatorname{unit}(\bar C_L-\bar C_R)$, $\mathbf s=\tfrac12(\bar C_L+\bar C_R)$ (means over the $F$ rows of $\mathbf A$); angles only |
| `rot_trans` | axis $(\hat{\mathbf n},\mathbf s)$ plus a translation $\tau$ along a fitted direction $\hat{\mathbf u}$, one angle and one $\tau$ per frame |
| `hinge_teeth` | as `hinge_fit`, but fitted to the three tooth points only (D25 sensitivity: the skin-palpated condyle points carry no weight); residuals still reported at all five points |

- **Rotation about an axis** (`rotate`, Rodrigues): for $\mathbf a=\mathbf A-\mathbf s$,
  $$
  R(\theta)\mathbf a=\mathbf a\cos\theta+(\hat{\mathbf n}\times\mathbf a)\sin\theta+\hat{\mathbf n}(\hat{\mathbf n}\cdot\mathbf a)(1-\cos\theta),\qquad \hat{\mathbf B}=R(\theta)\mathbf a+\mathbf s .
  $$
- **Angle per frame** (`best_angles`, closed form, the 1-D Wahba problem): with $\mathbf a_k,\mathbf b_k$
  the points relative to $\mathbf s$ projected onto the plane $\perp\hat{\mathbf n}$
  ($\mathbf a^\perp=\mathbf a-(\mathbf a\cdot\hat{\mathbf n})\hat{\mathbf n}$),
  $$
  \theta_f=\operatorname{atan2}\Big(\sum_k\hat{\mathbf n}\cdot(\mathbf a^\perp_{fk}\times\mathbf b^\perp_{fk}),\;\sum_k\mathbf a^\perp_{fk}\cdot\mathbf b^\perp_{fk}\Big),
  $$
  which maximizes $\sum_k\mathbf b_{fk}\cdot R(\theta)\mathbf a_{fk}$ and so minimizes
  $\sum_k\lVert R(\theta)\mathbf a_{fk}-\mathbf b_{fk}\rVert^2$.
- **Optimisation problem** (`fit_hinge`): axis direction by azimuth $\alpha$ (about $z$, from
  $+y$) and elevation $\varepsilon$, $\hat{\mathbf n}(\alpha,\varepsilon)=(-\sin\alpha\cos\varepsilon,\;\cos\alpha\cos\varepsilon,\;\sin\varepsilon)$,
  so $\hat{\mathbf n}=+\hat{\mathbf y}$ at $(0,0)$; $\mathbf s\in\mathbb R^3$ free. With the angles
  profiled out,
  $$
  \min_{\alpha,\varepsilon,\mathbf s}\;\sum_{f=1}^{F}\sum_{k}\big\lVert R\big(\theta_f(\alpha,\varepsilon,\mathbf s)\big)(\mathbf A_{fk}-\mathbf s)+\mathbf s-\mathbf B_{fk}\big\rVert^2+\big(10^{-3}\,\mathbf s\cdot\hat{\mathbf n}\big)^2 .
  $$
  Equal weights for all points and frames (plain sum of squared 3-D errors). The cost is
  invariant to sliding $\mathbf s$ along the axis; the tiny regularizer selects the axis point
  closest to the origin. Solver: `scipy.optimize.least_squares`, trust-region reflective
  (`method="trf"`), 2-point finite-difference Jacobian, default tolerances
  (`ftol` = `xtol` = `gtol` = $10^{-8}$), `x_scale` = (0.1, 0.1, 10, 10, 10) (rad, rad, mm, mm,
  mm). Start: $\mathbf s_0=\tfrac12(\bar C_L+\bar C_R)$, $\hat{\mathbf n}_0=\operatorname{unit}(\bar C_L-\bar C_R)$,
  $\alpha_0=\operatorname{atan2}(-n_{0x},n_{0y})$, $\varepsilon_0=\arcsin n_{0z}$. `hinge_teeth`
  runs the same problem over $k\in\{I,M_L,M_R\}$ (angles from these points too), started from
  the `hinge_fit` solution.
- **`rot_trans`** (`fit_rot_trans`): parameters $(\alpha,\varepsilon,\mathbf s,\alpha_u,\varepsilon_u)\in\mathbb R^7$,
  $\hat{\mathbf u}=(\cos\alpha_u\cos\varepsilon_u,\;\sin\alpha_u\cos\varepsilon_u,\;\sin\varepsilon_u)$;
  prediction $\hat{\mathbf B}_{fk}=R(\theta_f)\mathbf a_{fk}+\mathbf s+\tau_f\hat{\mathbf u}$, same
  objective and regularizer over all five points, `least_squares` (TRF), `x_scale` = (0.1, 0.1,
  10, 10, 10, 0.3, 0.3). Start: the `hinge_fit` axis and $\hat{\mathbf u}$ 30° anterior-inferior
  ($\alpha_u=0$, $\varepsilon_u=-30°$), all 7 parameters refitted.
- **`rot_trans` per frame** (`_rt_predict`, exact for given $\hat{\mathbf n},\mathbf s,\hat{\mathbf u}$):
  for a fixed angle the optimal translation is closed-form,
  $\tau(\theta)=\frac1K\sum_k\hat{\mathbf u}\cdot\big(\mathbf b_k-R(\theta)\mathbf a_k\big)$, so the
  frame objective depends on $\theta$ only:
  $$
  f(\theta)=\sum_k\lVert R(\theta)\mathbf a_k-\mathbf b_k\rVert^2-K\,\tau(\theta)^2,\qquad K=5 .
  $$
  With $C=\sum_k\mathbf a^\perp_k\cdot\mathbf b_k$, $S=\sum_k(\hat{\mathbf n}\times\mathbf a_k)\cdot\mathbf b_k$
  and $\tau(\theta)=\alpha-\beta\cos\theta-\gamma\sin\theta$ ($\alpha,\beta,\gamma$ the means of
  $\hat{\mathbf u}\cdot(\mathbf b-\mathbf a^\parallel)$, $\hat{\mathbf u}\cdot\mathbf a^\perp$,
  $\hat{\mathbf u}\cdot(\hat{\mathbf n}\times\mathbf a)$), $f$ is trigonometric in $\theta$. Newton
  steps $\theta\leftarrow\theta-f'/f''$ (a frame with $f''\le0$ is not stepped; ≤ 12 steps, until
  every $\lvert\text{step}\rvert\le10^{-13}$ rad), vectorised over frames, start from the
  pure-hinge angle $\operatorname{atan2}(S,C)$ and reach the minimum to machine precision; then
  $\tau_f=\tau(\theta_f)$. This replaced an angle/translation alternation (E10: to a $10^{-6}$ mm
  tolerance, ≤ 50 iterations) whose tolerance made the finite-difference Jacobian noisy, so
  `least_squares` stopped before the minimum (E15; same fixed point to $10^{-13}$ mm,
  `tests/test_perf_equivalence.py`).
- **Warm start** (`hinge_fits(A, B, init=)`, `fit_rot_trans(start=)`): the bootstrap resamples
  of the observed-residual CI (`montecarlo.observed_ci`, an06 §3; 2.5/97.5 percentiles of
  `n_boot` = 100 cycle-bootstrap × landmark-perturbation draws) start `hinge_fit` from the
  full-data axis $(\hat{\mathbf n},\mathbf s)$ and `rot_trans` from the full-data
  $(\hat{\mathbf n},\mathbf s,\hat{\mathbf u})$ (the CI covers `hinge_fit`, `hinge_tmj` and
  `rot_trans`, not `hinge_teeth`).

**Outputs** (`hinge_summary`, `hinge` table, one row per model). Residual per point and frame
$r_{fk}=\lVert\hat{\mathbf B}_{fk}-\mathbf B_{fk}\rVert$ (mm);
$$
\mathrm{RMS}_{\mathcal P}=\sqrt{\frac{1}{F\lvert\mathcal P\rvert}\sum_f\sum_{k\in\mathcal P}r_{fk}^2}\quad(\text{NaN-skipping}).
$$
`rms_incisor_mm` and `max_incisor_mm` ($\mathcal P=\{I\}$), `rms_molars_mm` (both molars
pooled), `rms_condyles_mm` and `max_condyles_mm` (both condyle points pooled). Also `n_frames`
($F$), axis `axis_x/y/z` ($\hat{\mathbf n}$), mid-sagittal piercing point `axis_pierce_x/z_mm`
($\mathbf s-(s_y/n_y)\hat{\mathbf n}$, only if $\lvert n_y\rvert>0.5$), `theta_max_deg` =
$\max_f\theta_f$, and for `rot_trans` `translation_range_mm` = $Q_{95}(\tau)-Q_5(\tau)$ and
`trans_dir_x/z` ($u_x,u_z$). Arrays: `hinge__residual_incisor`, `hinge__theta_deg` (both
`hinge_fit`), `hinge__tau_mm` (`rot_trans`).

> A translation that is strictly proportional to the angle is indistinguishable from a rotation
> about a shifted axis. The hinge residual therefore measures **axis migration**, not translation
> as such. The synthetic tests use the physiological pattern of rotation first, then translation.
>
> In vivo the two are nearly proportional, so `rot_trans` is **not identifiable**: on the pilot,
> moving the axis up by 35–90 mm and doubling the translation range changes the incisal RMS by
> ≈ 0.03 mm (D10, change log). `rot_trans` is reported for its residual only. The translation
> an exoskeleton must allow is taken model-free from the condyle-point excursion (larger side,
> closed to peak opening; §7a). The synthetic ground truth, with a clear rotate-then-translate
> pattern, is still recovered by the fit.

**Paper wording (hinge models).** For each participant, a single fixed rotation axis was
fitted to all opening movements by nonlinear least squares, minimizing the summed squared 3-D
prediction errors of the incisor, both second molars and both condyle points, with the
rotation angle of each frame solved in closed form. As a second model, the axis was fixed
through the two palpated condyle points and only the angles were fitted. The fixed-hinge
misalignment was the root-mean-square residual at the incisor over all frames (up to 30 frames
per opening).

**Coupling curves** (`coupling_curves`): per opening half, $\theta_y$ is monotonised by its
running maximum $\tilde\theta(t)=\max_{t'\le t}\theta_y(t')$; at repeated $\tilde\theta$ values
the first frame is used (so untracked frames, which inherit the running maximum, never enter);
the curves are linearly interpolated onto the grid $0,1,\dots,40°$ (`coupling.max_deg` = 40,
`coupling.step_deg` = 1), NaN outside the cycle's range (needs ≥ 2 distinct angles). Curves (all
relative to the opening start): `opening_mm` = $-\Delta_z$ of the incisor, `tmj_mid_ant_mm` =
$\Delta x$ of the condyle midpoint, `tmj_mid_disp_mm`, `tmj_{left,right}_disp_mm` = $\lVert\Delta\rVert$
of the midpoint / condyle points. Stored as arrays `coupling__<curve>` (cycles × grid) and
`coupling__grid_deg`.
- **Cohort band** (`stats_kinematics.coupling_band`, arrays): per participant the median over
  cycles at each grid point with ≥ 3 finite cycles (else NaN); at each grid point the cohort
  median, Q1 and Q3 (`numpy.nanpercentile`, linear) of the participant medians where
  ≥ $\max(3,\lfloor N/2\rfloor)$ of the $N$ participants have a value. Only `opening_mm`,
  `tmj_mid_ant_mm` and `tmj_mid_disp_mm` are banded.

**Opening model** for the IMU stage (`analysis._opening_fit`, `opening_model` table):
$$
\text{opening}\approx a_1\theta_y+a_2\theta_y^2\quad(\text{no intercept}),
$$
ordinary least squares (`numpy.linalg.lstsq`) over all finite frames of all opening halves used
above ($\theta_y$ in degrees and opening $=-\Delta_z$ in mm, both relative to each opening start).
Reports `a1_mm_per_deg`, `a2_mm_per_deg2`, `rmse_mm` = $\sqrt{\operatorname{mean}(\text{residual}^2)}$
and `n` (frames). **Leave-one-cycle-out** (`opening_model_loco`, table `opening_model_loco`, D26):
the same fit once per opening with that opening's frames left out (NaN coefficients when the
rest has < 10 frames), so the IMU stage can predict each cycle out of sample (an04).

**Paper wording (coupling and opening model).** Incisal opening and condyle-point
displacement were expressed as functions of the opening angle on a 1° grid and summarized as
the cohort median and interquartile range of the participant medians. For the wearable sensor,
incisal opening was modelled per participant as a quadratic function of the opening angle
without intercept, fitted by least squares.

## 6a. Jaw geometry per participant (`kinematics/geometry.py:jaw_geometry`, D21)

Table `kinematics:geometry`: one row per participant. Static values use the rigid landmark sets
(MP frame for the lower jaw, maxillary frame for the upper); the closed-pose values use the
maxillary frame at the closed reference pose $T_c$. Notation as in §0;
$\hat{\mathbf m}(\mathbf a,\mathbf b,\mathbf c)=\operatorname{unit}\big((\mathbf b-\mathbf a)\times(\mathbf c-\mathbf a)\big)$
is the unit normal of the plane through three points, and
$\operatorname{dist}(\mathbf p;\mathbf a,\mathbf b)=\lVert\mathbf d-(\mathbf d\cdot\hat{\mathbf e})\hat{\mathbf e}\rVert$,
$\mathbf d=\mathbf p-\mathbf a$, $\hat{\mathbf e}=\operatorname{unit}(\mathbf b-\mathbf a)$, the
distance of a point from a line.

| Column | Definition |
|---|---|
| `bonwill_side_mm` | $\ell=\tfrac12\big(\lVert C_L-I\rVert+\lVert C_R-I\rVert\big)$, the side of the Bonwill triangle (Bonwill's ideal: 4 in ≈ 102 mm, equilateral; references: Glossary of Prosthodontic Terms, 9th ed., 2017; Bonwill 1899) |
| `bonwill_ratio` | $\ell/\lVert C_L-C_R\rVert$ (1 for an equilateral triangle; the skin points lie lateral to the joints, so the ratio is < 1) |
| `balkwill_deg` | $\arccos\big\lvert\hat{\mathbf m}(I,C_L,C_R)\cdot\hat{\mathbf m}(I,M_L,M_R)\big\rvert$: the Bonwill plane against the lower occlusal plane (literature 18–26° depending on the condylar reference point: Balkwill 1865–67; Bergström 1950; Ohm & Silness 1982; Shan et al. 2025) |
| `lower_arch_depth_mm`, `upper_arch_depth_mm` | $\operatorname{dist}(I;M_L,M_R)$ and $\operatorname{dist}(I^u;M^u_L,M^u_R)$ |
| `condyle_height_mm` | $(C_{\mathrm{mid}}-I)\cdot\hat{\mathbf m}_{\mathrm{occ}}$ at $T_c$, $\hat{\mathbf m}_{\mathrm{occ}}=\pm\hat{\mathbf m}(I,M_L,M_R)$ with the sign chosen so that its $z$ component is ≥ 0 (pointing up) |
| `hinge_radius_mm` | distance of the closed incisor from the `hinge_fit` axis of §6: $\lVert\mathbf d-(\mathbf d\cdot\hat{\mathbf n})\hat{\mathbf n}\rVert$, $\mathbf d=\mathbf x_I(T_c)-\mathbf s$ |
| `hinge_axis_dx_mm`, `hinge_axis_dz_mm`, `hinge_axis_offset_mm` | piercing point $\mathbf s-(s_y/n_y)\hat{\mathbf n}$ of the axis with $y=0$ (if $\lvert n_y\rvert>0.5$) minus $\mathbf x_{C_{\mathrm{mid}}}(T_c)$, $x$ and $z$ components (anterior, up), and $\sqrt{dx^2+dz^2}$ |
| `hinge_axis_angle_deg` | $\arccos\big\lvert\hat{\mathbf n}\cdot\operatorname{unit}(C_L-C_R)\big\rvert$ (condyle points at $T_c$) |

**Reading the hinge-axis values.** The best-fit fixed axis reproduces rotation plus condylar
translation with a single rotation. For an opening of angle $\theta$, a point at the condyle that
translates by $t$ moves as if the jaw rotated about a centre about $t/\theta$ away from it,
perpendicular to its path. That puts the equivalent axis below and behind the condyles; it
is not the anatomical axis. The offset is therefore a kinematic property of the opening
(C1), not a palpation error.

**Cohort summaries:** the geometry enters `stats_kinematics:reference_values` (all
participants, §7). The condyle-based columns (all except the arch depths and the hinge radius,
`cohort_kinematics.CONDYLE_METRICS`) also enter `condyle_sensitivity` (all vs plausible
condyles).

**Not estimated:** guidance and Bennett angles. The protocol did not control tooth contact,
and the closed pose is the rest position with the freeway space. The trial values
(incisor rising during protrusion in 9 of 14 participants, balancing condyle points up to
144° from the sagittal plane) are protocol and palpation effects (D21, change log).

**Paper wording.** From the landmarks we derived the Bonwill triangle (lower incisor and both
condyle points), the angle between the Bonwill and lower occlusal planes, the arch depths, the
condyle height above the occlusal plane, and, from the fitted fixed axis, the incisor's radius
about it and the axis' offset from the palpated condyle midpoint in the sagittal plane.

## 6b. Hinge residual against measurement noise (`uncertainty/montecarlo.py`, stage `uncertainty`)

**What and why.** A perfect hinge measured with the system's noise and processed by the same
chain does not give a zero residual; the observed residual of §6 is therefore compared with the
distribution of residuals of a perfect hinge under the participant's own noise. The method is
specified in full in [an06](an06_uncertainty_and_validation.md) §3 (home of the uncertainty
methods); the cohort statistics are in [an07](an07_statistics.md) §14. In short: per opening,
ideal poses $T^0(f)=H(\theta_y(f))\,T(\text{start})$ about the fitted axis (the `hinge_fit`
axis; for the condyle-hinge nulls the `hinge_tmj` axis of §6, direction
$\operatorname{unit}(\bar C_L-\bar C_R)$, left minus right, condyle points averaged over the
$F$ stacked rows of $\mathbf A$), with the measured opening angles, are perturbed by left-multiplied
pose noise (bench: white; in vivo: block bootstrap of the rest residuals of §4), low-passed as in
§1, subsampled and refitted as in §6 (1000 draws per null); the observed incisor RMS is called
above noise when it exceeds the 99th percentile of the draws. Parameters: §8 and an06 §3.

## 7. Participant-level metrics and literature norms (`stats/cohort_kinematics.py`, `norms/`)

**`stats_kinematics.subjects`** (`subject_table`; one row per participant, from the `ok` cycles):
- S1 metrics (`METRICS`): the **median** over cycles of open_close `opening_mm`
  (`opening_selfpaced_mm`), `theta_y_deg` (`opening_angle_deg`), `deviation_mm`
  (`opening_deviation_mm`), `condyle_{left,right}_disp_mm`, `peak_open_vel_dps`,
  `peak_close_vel_dps`, and of chewing `opening_mm` (`chewing_amplitude_mm`) and `theta_y_deg`
  (`chewing_angle_deg`); the **max** over cycles for `protrusion_mm`, `retrusion_mm`,
  laterotrusion left (`lateral_mm`, side left) and right ($-$`lateral_mm`, side right),
  `peak_open_vel_max_dps` (open_close `peak_open_vel_dps`), `peak_close_vel_max_dps` (open_close
  `peak_close_vel_dps`; the $X_i$ of the `close_vel_dps` requirement, §7a) and
  `chewing_peak_vel_dps` (chewing `peak_open_vel_dps`).
- `chewing_freq_hz` = $1/\operatorname{median}(\texttt{duration\_s})$ of the chewing cycles.
- `opening_cv` = SD(ddof = 1)/mean of the open_close `opening_mm` (n > 1).
- `lateral_symmetry_pct` = $100\,(L-R)/(L+R)$, $L,R$ = laterotrusion left/right (both positive).
- `opening_per_intermolar` = `opening_selfpaced_mm` / `upper_intermolar_mm` (dimensionless).
- Plus the `anatomy` and `geometry` columns (merged by pid); sex and age from the demographics
  are used for the groups below and dropped from the stored table.
- **X7 columns** (post hoc description; merged by pid, `stages.py:306-309`). These are
  computed from the stored whole-phase signals of the `kinematics` arrays (6 Hz pose low-pass,
  Savitzky–Golay derivatives, opening positive), not only from the `ok` cycles.
  `dynamics_envelope` (`cohort_kinematics.py:90-118`): `free_peak_open_vel_dps` = max and
  `free_peak_close_vel_dps` = $-$min of the free_motion $\dot\theta_y$ (`free_motion__theta_vel`;
  NaN when the sign never occurs); `peak_acc_dps2` = max $\lvert\ddot\theta_y\rvert$ over the
  `ENVELOPE_TASKS` open_close, chewing, free_motion (`<task>__theta_acc`), `peak_acc_task` the
  task that holds it, `peak_acc_p99_dps2` = the largest per-task 99th percentile of
  $\lvert\ddot\theta_y\rvert$ (robustness column: one sample decides the maximum).
  `posterior_excursion` (`cohort_kinematics.py:121-153`): with $\tilde x_I$ the median rest-phase
  incisor $x$ (maxillary frame, anterior positive), the posterior displacement
  $\max(0,\tilde x_I - x_I)$ over all finite samples of the five `WORKSPACE_TASKS` (open_close,
  protrusion_retrusion, left_right, chewing, free_motion); `posterior_excursion_mm` = its
  `POSTERIOR_Q` = 99th percentile (robust to a single tracking glitch),
  `posterior_excursion_max_mm` its maximum (NaN without rest or task samples).

**Cohort tables** (statistics in an07 §5, §8): `reference_values` (metric set
`stages.py:310-314`: the `METRICS` above, `chewing_freq_hz`, `opening_cv`,
`lateral_symmetry_pct`, `opening_per_intermolar`, `max_observed_opening_mm`,
`upper_intermolar_mm`, `intercondylar_mm`, `condyle_incisor_{left,right}_mm`,
`floor_incisor_rms_mm_smooth` and the `GEOMETRY_METRICS`; the X7 columns are not included;
`describe` per metric for all, f, m:
mean ± SD, median [IQR], BCa CI; Hedges' g f − m with CI; Spearman ρ with age when > 3 values);
`influence` (leave-one-out change of the cohort mean for the older participants
`stages.stats_kinematics.older_pids` = [s02, s06, s08]: $100(\bar x_{-i}-\bar x)/\bar x$);
`reliability` (ICC/SEM/MDC across cycles, `RELIABILITY` metrics; laterotrusion per side, as
the absolute `lateral_mm`); `hinge_summary`, `hinge_noise_summary`, `hinge_noise_tests`,
`fha_subject`, `fha_summary` (§5, §6b), `condyle_sensitivity` and `coupling_summary` (below).

**`coupling_summary`** (X9, post hoc description; `cohort_kinematics.py:336-378`,
written when the coupling grid exists, `stages.py:343-347`): one row per participant from the
`coupling__tmj_mid_ant_mm` cycle curves (anterior condyle-midpoint translation on the
$\theta_y$ grid of §6). The participant's median curve keeps the angles with ≥ 3 cycles.
`slope_mm_per_deg` = Theil–Sen slope of that curve over `COUPLING_SLOPE_DEG` = 5–15° (≥ 3 grid
points, else NaN); `generic_rms_mm` = RMS difference from the leave-one-subject-out cohort
median curve (an angle enters the generic curve when ≥ max(3, ⌊(N−1)/2⌋) of the other
participants define it; ≥ 3 shared angles, else NaN), i.e. how much a generic coupling model
misses; `angle_max_deg` (largest angle of the participant's curve), `n_angles` (shared
angles), `condyles_plausible` (D18). Descriptive, no test; it rests on the palpated condyle
points (§2), so the `condyles_plausible` flag travels with it.

**`condyle_sensitivity`** (flag, don't exclude): every condyle-based metric, the incisal RMS of
every hinge model (`hinge_rms_incisor_<model>`), `translation_max_mm` =
$\max(\tilde C_L,\tilde C_R)$ and `translation_mean_mm` = $\tfrac12(\tilde C_L+\tilde C_R)$ with
$\tilde C_{L,R}$ the participant's median `condyle_{left,right}_disp_mm`, the `rot_trans`
`translation_range_mm` and `rms_condyles_mm`, and the FHA `dist_tmj`; each as `describe` (median,
BCa CI) for the groups `all`, `plausible_condyles`, `condyles_unknown` (no anatomy, only if any)
and `plausible_condyles_30mm` / `_50mm` (D29: the depth threshold at fixed neighbours,
`plausible_at_depth`).

**Literature norms** (`norms/literature.compare`, values in `norms/norms.yaml`; an07 §12): per
participant and listed metric, the reference entry of the participant's sex, falling back to
`sex: all`; $\text{diff}=x-\mu_{\mathrm{ref}}$ and $z=\text{diff}/\sigma_{\mathrm{ref}}$ (NaN when
`sd` is null). Tables `norms_subjects` (one row per participant and metric) and `norms_summary`
(per metric × `ref_mean` × source: n, mean and SD (ddof = 1) of diff). In the current yaml only
protrusion and laterotrusion have entries, all pooled and without an SD.

**Paper wording.** Per participant, the median over valid cycles was used for opening,
opening angle, deviation, condylar excursion and velocities, and the maximum over cycles for
protrusion, retrusion and laterotrusion. Cohort values are reported as median with
bias-corrected and accelerated bootstrap confidence intervals.

## 7a. Kinematic inputs of the device requirements (`stats/requirements.py`, stage `requirements`)

The percentile policy (design value = 95th percentile across participants, Harrell–Davis
estimate, exact one-sided lower bound, sample maximum) is specified in an07 §13 (Derived
requirements); the bandwidth $f_{95}$ of the coherence-weighted IMU spectrum in an04 §6. This
section defines the participant value $X_i$ each kinematic row is computed from.

| Row (`key`) | Participant value $X_i$ | Statistic | Evidence |
|---|---|---|---|
| `opening_mm` (incisal opening) | `max_observed_opening_mm` (§4; a lower bound of MIO) | P95 rule | descriptive |
| `protrusion_mm` | max over protrusion cycles of `protrusion_mm` | P95 rule | descriptive |
| `laterotrusion_mm` (larger side) | $\max(\text{laterotrusion left},\text{laterotrusion right})$ of §7 | P95 rule | descriptive |
| `opening_angle_deg` (opening rotation) | `opening_angle_deg` = median over open_close cycles of $\theta_y(p)$ | P95 rule | descriptive |
| `translation_mm` (passive translational DoF) | $\max(\tilde C_L,\tilde C_R)$, the larger side's median condyle-point excursion closed → peak opening (model-free, D10; = `translation_max_mm`) | P95 rule | descriptive |
| `peak_vel_dps` | `peak_open_vel_max_dps` = max over open_close cycles of $\max\dot\theta_y$ | P95 rule | descriptive |
| `close_vel_dps` (X7) | `peak_close_vel_max_dps` = max over open_close cycles of the closing peak | P95 rule | post_hoc_descriptive |
| `free_peak_open_vel_dps` (X7) | `free_peak_open_vel_dps` (§7, whole free_motion phase) | P95 rule | post_hoc_descriptive |
| `peak_acc_dps2` (X7) | `peak_acc_dps2` (§7, max $\lvert\ddot\theta_y\rvert$ over open_close, chewing, free_motion) | P95 rule | post_hoc_descriptive |
| `posterior_excursion_mm` (X7) | `posterior_excursion_mm` (§7, P99 of the posterior incisal displacement behind rest) | P95 rule | post_hoc_descriptive |
| `bandwidth_hz` | max over the not noise-limited opening tasks (open_close, chewing, free_motion) of `f95_coh_hz` (an04 §6) | P95 rule | exploratory |
| `misalign_hinge_fit_mm`, `misalign_hinge_tmj_mm` | `rms_incisor_mm` of `hinge_fit` / `hinge_tmj` (§6) | median, BCa CI | confirmatory / secondary_post_hoc |

Evidence: the stored `evidence` column (`stats/evidence.py:REQUIREMENT_EVIDENCE`, an07 §13;
reader words from `etc/vocab.yaml` domain `evidence`). The X7 rows (`requirements.py:_envelope_rows`
and the `posterior_excursion_mm` row, `requirements.py:186-205, 383-389`) are only written when
the participant column has a finite value; their `note` is "lower bound (40-Hz optics, 6-Hz
low-pass); post hoc" for the three dynamics rows and "lower bound; post hoc" for
`posterior_excursion_mm` (group `workspace`).

**P95 rule** (`p95_design`; defined in [an07](an07_statistics.md) §13): the Harrell–Davis
estimate of the 95th percentile across participants (`value`; `value_linear` the
linear-interpolation P95 next to it), the exact distribution-free one-sided lower 95 %
order-statistic bound (`ci_lo` = $X_{(k)}$ with `lower_k` = $k$ and its achieved confidence
`lower_conf`; $X_{(11)}$ of 13, $X_{(12)}$ of 14), no finite upper bound (`ci_hi` NaN,
`upper_note`) and the sample maximum with its coverage statement (`max_q` = `MAX_Q` = 0.80,
`max_conf` = $1-0.8^{n}$; `requirements.py:137-172`). D27 sensitivity
(`requirements_sensitivity`, `requirements.py:524-594`): variant `largest_cycle` with $X_i$ =
the largest `ok` cycle of the task (`uniform_per_participant`) for the five `UNIFORM` rows only,
`opening_mm`, `protrusion_mm`, `laterotrusion_mm`, `opening_angle_deg`, `peak_vel_dps` (not
`translation_mm` nor the X7 rows); the same table holds the bandwidth variants `imu_only`,
`optics_only`, `min_band_1.5hz` and `min_band_3hz` (an04 §6, an07 §13).

**Paper wording.** Workspace and speed requirements were derived from participant-level values
(the largest incisal opening observed in any task, the largest protrusion and laterotrusion,
the median opening angle, the larger side's condyle-point excursion and the fastest opening)
as the 95th percentile across participants (Harrell–Davis estimator), reported with the exact
distribution-free one-sided lower 95 % bound and the sample maximum. Post hoc, the closing and
free-motion angular velocities, the peak angular acceleration and the posterior incisal
excursion behind the rest position were added with the same rule; the dynamics rows are lower
bounds, since the optical chain (40-Hz sampling, 6-Hz low-pass) attenuates fast transients.

## 8. Parameters

**Config** (`etc/biomech.yaml`; the study file `experiments/icorr2027/study.yaml` overrides none
of them for the primary run):

| Key | Value | Meaning |
|---|---|---|
| `signals.kinematics.max_gap_s` | 0.5 | longest bridged gap (§1) |
| `signals.kinematics.lowpass_hz`, `lowpass_order` | 6.0, 2 | pose low-pass (§1); variants 8/10/12 Hz in the study file |
| `signals.kinematics.hold_trim` | 0.2 | fraction cut from each end of a landmark hold (§2) |
| `signals.kinematics.min_condyle_incisor_mm` | 70.0 | condyle distance check (§2) |
| `signals.kinematics.min_condyle_superior_mm` | 5.0 | condyle level check (§2) |
| `signals.kinematics.min_condyle_posterior_mm` | 40.0 | condyle depth check (§2); D29 neighbours 30 / 50 mm evaluated in `condyle_sensitivity` |
| `stages.kinematics.cycles.min_period_s` | 0.4 | peak distance and the `short` flag |
| `stages.kinematics.cycles.prominence_frac` | 0.3 | peak prominence, as a fraction of the P5–P95 range; half of it is the amplitude floor |
| `stages.kinematics.cycles.min_amplitude` | 0.0 | absolute amplitude floor (signal units); the effective floor is max(this, 0.5·prominence_frac·range) |
| `stages.kinematics.cycles.max_nan_frac` | 0.1 | `untracked` flag |
| `stages.kinematics.cycles.return_frac` | 0.2 | `no_return` flag (open_close, chewing) |
| `stages.kinematics.fha.step_deg` | 5.0 | FHA pair spacing in $\theta_y$ |
| `stages.kinematics.hinge.max_frames_per_cycle` | 30 | hinge-fit subsample per opening |
| `stages.kinematics.coupling.max_deg`, `step_deg` | 40.0, 1.0 | coupling grid |
| `stages.stats_kinematics.older_pids` | [s02, s06, s08] | leave-one-out influence |
| `stages.uncertainty.n_sims` | 1000 | noise-null draws (§6b; other keys in an06) |

**Module constants** (not configurable): smoothing `min_len` = 15 frames (`se3.lowpass`); bridge
validity $\lvert\det R\rvert>10^{-6}$; ≥ 5 hold frames (`_hold`); still-hold 1 s / 1.5 mm / 6 mm
(`_still_hold`); ≥ 8 finite samples for cycle detection; Savitzky–Golay `WINDOW_S` = 0.15 s,
`POLYORDER` = 3, segment split at gaps > 2.5 $\Delta t$; movement-bound fraction 0.05; noise-floor
window 1 s, stride ¼; rest-residual running median `REST_HP_S` = 1 s; FHA $\phi$ floor
$10^{-6}$ rad, piercing $\lvert n_y\rvert>0.5$ (FHA, hinge axis and geometry alike); hinge: openings ≥ 5
valid frames, ≥ 10 stacked frames, regularizer $10^{-3}$, `x_scale` as in §6, Newton ≤ 12 steps
to $10^{-13}$ rad; coupling band ≥ 3 cycles and ≥ max(3, ⌊N/2⌋) participants; opening-model
LOCO ≥ 10 frames (`MIN_LOCO_FRAMES`); noise null pad 1 s (`PAD_S`), ≥ 10 frames (`MIN_FRAMES`),
bench 0.182 mm / 0.126° (`BENCH_T_MM`, `BENCH_R_DEG`). X7 / X9 (`stats/cohort_kinematics.py`):
`ENVELOPE_TASKS` = open_close, chewing, free_motion; `WORKSPACE_TASKS` = the five movement
tasks; `POSTERIOR_Q` = 99; `COUPLING_SLOPE_DEG` = (5, 15)°. Requirements
(`stats/requirements.py:57-58, 116-117`): `CONF` = 0.95 (one-sided bound), `MAX_Q` = 0.80;
`COVERAGE_N` = 13–30 and `COVERAGE_Q` = 0.80 / 0.90 (X8 planning table `max_coverage`, an07 §13).
