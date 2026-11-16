<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an05 — Onsets, EMG-to-motion lead (C2, the post hoc family C2-E), intent labels and decoding (C3b)

**Scope.** Every calculation behind the timing results (muscle onsets, mechanical onsets, the
EMG-to-motion lead and its chance null, the detector-free ensemble timing, the causal closer
release, the post hoc event-order family), the intent labels with their validity check, and the
intent decoding with its pseudo-online replay. Plan codes, used in the technical text only:
C2 = the pre-specified lead claim; C2-E (X6) = the post hoc event-order family;
X2 = ensemble timing; X3 = causal release detector; X4 = pseudo-online replay; C3b = intent
decoding; D-numbers are analysis-plan decisions, E-numbers erratum entries; they and the
X- and J-numbers are listed in the [change log](../../experiments/icorr2027/change_log.md) of
the analysis plan.

**Abbreviations.** EMG = electromyography; IMU = inertial measurement unit; TKEO =
Teager–Kaiser energy operator; RMS = root mean square; SD = standard deviation; IQR =
interquartile range; BCa = bias-corrected and accelerated (bootstrap); CI = confidence interval;
IUT = intersection-union test; KDE = kernel density estimate; MVC = maximal voluntary
contraction; LOSO = leave one subject out; SVM = support vector machine; LDA = linear
discriminant analysis; QC = quality control. Symbols are listed in §0 (Notation).

**Elsewhere.** EMG cleaning, envelopes and masks: [an03](an03_emg_processing.md) §2. Optical
opening angle, cycles: [an02](an02_frames_and_kinematics.md) §3, §4. IMU opening angle, ω and
the relative acceleration: [an04](an04_imu_processing.md) §4–§5. Generic statistics (BCa
bootstrap CI of the median, Wilcoxon signed-rank, Holm, Monte-Carlo p values,
intersection-union, Friedman): [an07](an07_statistics.md). Decoding specification:
[docs/ml](../ml/README.md) (ml03 features, ml04 models, ml05 evaluation, ml08 online replay).

## 0. Notation

| Symbol | Meaning |
|---|---|
| $f_s$ | EMG sampling rate (4 kHz); $n$ sample index; $N$ samples of a phase; $T = N/f_s$ phase length (s) |
| $x[n]$ | cleaned (band-passed, mains-free) EMG of one channel, µV; NaN = masked (an03 §2) |
| $\Psi[n]$ | Teager–Kaiser energy, $x[n]^2 - x[n-1]\,x[n+1]$ |
| $s[n]$ | detector signal (non-negative) that a detector thresholds |
| $\mathcal R$ | the channel's labelled `rest` phase; $\bar s_{\mathcal R}$, $\mathrm{SD}_0(s_{\mathcal R})$ mean and SD (ddof 0) of $s$ over $\mathcal R$ |
| $\eta$ | detector threshold; $k_{\mathrm{sd}} = 3$ |
| $w_\tau$ | odd window length of a centred smoother of duration $\tau$: $w_\tau = 2\lfloor \tau f_s/2 \rfloor + 1$ (`odd_window`); $h_\tau = (w_\tau-1)/2$ |
| $n_{\mathrm{on}}, n_{\mathrm{off}}, n_{\mathrm{edge}}$ | $\mathrm{round}(\tau f_s)$ of 25, 50, 50 ms |
| $\theta_{\mathrm{quat}}(t)$, $\theta_{\mathrm{opt}}(t)$ | IMU (calibrated relative-quaternion) and optical opening angle $\theta_y$ relative to the closed pose, deg (an04 §4, an02 §3) |
| $\omega(t)$ | opening angular velocity $\mathrm d\theta_{\mathrm{quat}}/\mathrm dt$ (odd 5-sample centred mean), deg/s, 200 Hz grid |
| $\alpha(t)$ | signed angular acceleration $\mathrm d^2\,\mathrm{LP}_{15}(\theta_{\mathrm{quat}})/\mathrm dt^2$, deg/s² |
| $a_{\mathrm{rel}}(t)$ | jaw-minus-head linear acceleration magnitude, m/s² (§2) |
| $\sigma_d$ | direction sign, $+1$ opening, $-1$ closing |
| $t_\omega, t_{\mathrm{acc}}, t_{\mathrm{opt}}, t_\alpha$ | mechanical onsets from $\omega$, $a_{\mathrm{rel}}$, $\theta_{\mathrm{opt}}$, $\alpha$; $t_{\mathrm{mech}} = \min(t_\omega, t_{\mathrm{acc}})$ |
| $t_{\mathrm{on}}$, $t_{\mathrm{dec}}$ | offline EMG onset; causal decision time |
| $L$ | lead $= t_{\mathrm{event}} - t_{\mathrm{on}}$ (ms; $> 0$: EMG first) |
| $\Delta_j$ | $j$-th circular shift of the EMG against the motion (samples), $\Delta_j = \mathrm{round}(U_j f_s)$, $U_j \sim \mathcal U[1\,\mathrm s,\ T - 1\,\mathrm s]$ |
| $M$ | number of surrogates (200) |
| $a_k$ | anchor of cycle $k$ for the cycle-shuffle surrogate |
| $\mathrm{IQR}(\cdot)$ | $Q_{75} - Q_{25}$ (numpy linear percentiles) |
| $\pi, \mu, \sigma$ | share, location and SD of the time-locked mixture component |
| $\delta$ | bias margin of the location test, 17 ms |
| $p_{\mathrm{MC}}$ | Monte-Carlo p value $(1 + \#\{\text{null} \preceq \text{obs}\})/(1 + M')$, $M'$ finite null values |
| $E(t)$ | closer group envelope, dB re rest |
| $\kappa$, $p_o$, $p_e$ | Cohen's kappa, observed and chance agreement |
| $\mathbf x$ | a decoding window's feature vector; $y$, $\hat y$ true and predicted class (7 classes) |

## 1. Onset detectors (`jtsmm/emg/onset.py`)

**Input.** One channel in the analysed window (§3), with masked samples set to NaN. The offline
detector reads the zero-phase cleaned signal (mains removed, band-passed, an03 §2); the causal
twin reads the causally cleaned signal of the same samples (E27, an03 §2.9).
`detect_onsets` sets NaN to 0 (`nan_to_num`) before it computes the detector signal, and a NaN
sample never counts as above threshold. A window shorter than 16 samples gives no onsets.

**Detector signals** (`detector_signal`). What is thresholded, offline (centred, zero-phase) and
causal (trailing, single pass):

$$
\begin{aligned}
\text{tkeo:}\quad & \Psi[n] = x[n]^2 - x[n-1]\,x[n+1]\ \ (\Psi[0]=\Psi[1],\ \Psi[N-1]=\Psi[N-2]),\\
& s[n] = \frac{1}{w_{20}} \sum_{j=-h_{20}}^{h_{20}} \lvert\Psi[n+j]\rvert \quad\text{(offline)},\qquad
  s[n] = \frac{1}{n_{20}} \sum_{j=0}^{n_{20}-1} \lvert\Psi[n-j]\rvert \quad\text{(causal)},\\
\text{hodges\_bui:}\quad & s = H_{50} * \lvert x\rvert,\ \ H_{50} = \text{2nd-order Butterworth low-pass at 50 Hz (sosfiltfilt / sosfilt)},\\
\text{envelope:}\quad & s[n] = \Big(\frac{1}{w_{50}} \sum_{j=-h_{50}}^{h_{50}} x[n+j]^2\Big)^{1/2}\ \text{(offline)},\ \text{trailing over } n_{50} \text{ samples (causal)},
\end{aligned}
$$

with $n_\tau = \mathrm{round}(\tau f_s)$. The centred smoothers use the odd length $w_\tau$
(E17/E26: an even centred boxcar shifts the signal by half a sample); edges use `mode="nearest"`;
the causal boxcar (`lfilter`, zero initial state) keeps $n_\tau$. The TKEO looks one sample ahead
($x[n+1]$), also in the causal twin.

**Threshold.** Mean plus $k_{\mathrm{sd}}$ SD of the detector signal over the channel's
**labelled `rest` phase** (`baseline_signal`; offline: zero-phase rest signal and offline
detector signal; causal twin: causally cleaned rest and causal detector signal):

$$
\eta = \bar s_{\mathcal R} + k_{\mathrm{sd}}\,\mathrm{SD}_0(s_{\mathcal R}),\qquad
\eta_\Psi = \overline{\lvert\Psi\rvert}_{\mathcal R} + k_{\mathrm{sd}}\,\mathrm{SD}_0(\lvert\Psi\rvert_{\mathcal R}),
$$

$\eta_\Psi$ being the unsmoothed TKEO threshold of the refinement below. Masked rest samples are
NaN; each NaN-free run is processed on its own and the detector values are pooled
(`jtsmm.emg.onset.baseline_values`, E18; runs shorter than $\max(\mathrm{round}(0.05\,f_s), 16)$
samples are skipped). Formerly the runs were concatenated, and the seams' TKEO / filter
transients raised the threshold. A channel needs ≥ 0.5 s of valid rest in both chains
(`MIN_REST_S`, A3; `biomech.timing.lead.build_detector`), else it gets no threshold and is
skipped; with fewer than 8 pooled values there are no onsets.

**Run rules.** A run is a maximal stretch $[a, b)$ with $s[n] > \eta$ and $x[n]$ finite. Runs are
scanned in order; `last_end` is the end of the last run that counted (initially
$-n_{\mathrm{off}}-1$, so the first run passes rule 4).
1. A run that starts in the first `edge_s` ($a < n_{\mathrm{edge}}$, 50 ms) is a filter edge. It
   is dropped and sets `last_end` $= b$.
2. A run shorter than `min_on_s` ($b - a < n_{\mathrm{on}}$, 25 ms) is dropped and does **not**
   update `last_end`, so blips do not restart the quiet period.
3. Offline only: a run that ends in the last 50 ms ($b > N - n_{\mathrm{edge}}$) *and* is shorter
   than $2n_{\mathrm{on}}$ (50 ms) is dropped, and `last_end` stays unchanged. The causal
   detector keeps such runs.
4. Any other run is an onset only if $a - \texttt{last\_end} \ge n_{\mathrm{off}}$ (50 ms). Either
   way it sets `last_end` $= b$, so a run rejected for being too close still extends the busy
   period.

**Onset index of an accepted run.**

| Detector | Offline onset | Causal decision |
|---|---|---|
| `tkeo` (primary) | $\hat n = \min\{n \in [a - h_{20},\ a + h_{20}] : \lvert\Psi[n]\rvert > \eta_\Psi\}$, $h_{20}$ = 10 ms; $a$ if the set is empty (synthetic error ≤ 10 ms) | $a + n_{\mathrm{on}} - 1$ |
| `hodges_bui` | $a$ (about 10 ms early from the zero-phase spread); `smooth_s` is not used | $a + n_{\mathrm{on}} - 1$ |
| `envelope` | $\min(a + h_{50},\ b - 1)$, $h_{50}$ = 25 ms | $a + n_{\mathrm{on}} - 1$ |

The causal decision is the sample at which the run has lasted 25 ms (`min_on_s`), with no
refinement or shift. The residual biases differ by detector on purpose; the spread across the
three is reported as a sensitivity (specification curve, an07).

**Parameters.** `stages.timing.k_sd` = 3 (the only yaml value). Hard-coded `OnsetParams`
defaults: `min_on_s` 25 ms, `min_off_s` 50 ms, `smooth_s` 20 ms, `envelope_s` 50 ms, `edge_s`
50 ms.

**Code.** `jtsmm/emg/onset.py:detector_signal`, `baseline_values`, `baseline_stats`,
`detect_onsets`.

**Paper wording.** Muscle onsets were detected with the Teager–Kaiser energy operator: its
absolute value was smoothed with a centred 20-ms moving average and thresholded at the mean
plus three standard deviations of the same signal in the channel's rest recording. An onset
required at least 25 ms above threshold after at least 50 ms below it and was refined to the
first sample within ±10 ms whose unsmoothed energy exceeded the corresponding rest threshold.
A low-pass (Hodges–Bui) and an RMS-envelope detector and causal versions of all three, which
report the sample at which the 25-ms criterion is met, served as sensitivity analyses.

## 2. Mechanical onset (`biomech/timing/lead.py`)

**What and why.** The time the jaw starts to move, per `ok` kinematic cycle (an02 §4) and
direction, on the IMU 200 Hz grid, inside a search window:
- opening: $[t_{\mathrm{start}} - \texttt{pre\_s},\ t_{\mathrm{peak}}]$, `pre_s` = 0.3 s;
- closing: $[t_{\mathrm{peak}} - 0.1\,\mathrm s,\ t_{\mathrm{end}}]$ (0.1 s hard-coded).

A window with fewer than 10 samples gives no onset.

**Backward rule** (`_backward_onset`). For samples $v_i$ of the window, peak index $i_p$ and
threshold $\eta$:

$$
i^\ast = \max\{\, i \le i_p : v_i < \eta \ \text{or}\ v_i = \mathrm{NaN} \,\} + 1,
\qquad t = t_{i^\ast}\ \text{if}\ i^\ast \le i_p,
$$

i.e. the **first sample at or above** $\eta$ after the last sample below it, at or before the
peak. There is no onset if no sample up to the peak is below $\eta$, or if the peak itself is
below $\eta$.

| Source | Signal $v$ and peak | Threshold $\eta$ |
|---|---|---|
| $t_\omega$ (`t_omega`) | $v = \sigma_d\,\omega$; $i_p = \arg\max v$ in the window | $\max(\texttt{vel\_frac}\cdot v_{i_p},\ 3\,\mathrm{SD}(\omega_{\mathcal R}))$, `vel_frac` = 0.05; SD = `nanstd` over the `rest` phase (`mechanical_baselines`: `omega_noise`) |
| $t_{\mathrm{acc}}$ (`acc_rel_ms2`, `t_acc`) | $v = a_{\mathrm{rel}}$ restricted to $i \le i_p^{\omega}$; $i_p = \arg\max$ of that; evaluated only if $\omega$ has a finite sample | $\mathrm{med}(a_{\mathcal R}) + \texttt{acc\_k}\cdot 1.4826\,\mathrm{MAD}(a_{\mathcal R})$, `acc_k` = 3, over the whole `rest` phase (`acc_thr`) |
| $t_{\mathrm{opt}}$ (`t_opt`) | $v = \sigma_d\,\mathrm d\theta_{\mathrm{opt}}/\mathrm dt$ (`timebase.gradient` of the optical angle on the IMU grid, per gap-free segment, NaN stays NaN; needs > 5 finite samples) | $\texttt{vel\_frac}\cdot v_{i_p}$ (no noise floor); cross-check only |
| $t_\alpha$ (C2-E only, never part of $t_{\mathrm{mech}}$) | $v = \sigma_d\,\alpha$ in the search span below (needs ≥ 3 samples, $v_{i_p} > 0$) | $\max(\texttt{alpha\_frac}\cdot v_{i_p},\ 3\,\mathrm{SD}(\alpha_{\mathcal R}))$, `alpha_frac` = 0.05 (`stages.timing.c2e`) |

Here $\omega = \mathrm d\theta_{\mathrm{quat}}/\mathrm dt$ is the gap-aware derivative of the IMU opening angle
with an odd 5-sample centred mean (`imu.omega_smooth_s` = 20 ms at 200 Hz; E17; an04 §5).

**The primary mechanical onset** (pre-specified, unchanged):

$$
t_{\mathrm{mech}} = \min\big(t_\omega,\ t_{\mathrm{acc}}\big)\quad\text{over the finite values}
$$

(`timing/lead.py:analyse`). If both are NaN, that cycle and direction are skipped.

**Relative acceleration** (defined in an04 §5, notation of an04 §0; `imu/analysis.py:phase_signals`):
with $\mathbf R_{\mathrm{mta}}, \mathbf R_{\mathrm{cra}}$ the game-rotation-vector orientations of
the jaw and head IMUs, $\mathbf f_{\mathrm{mta}}, \mathbf f_{\mathrm{cra}}$ their accelerometer
readings and $\mathbf R_z(\psi)$ the calibrated heading offset (an04 §3),

$$
a_{\mathrm{rel}} = \big\lVert \mathbf R_z(\psi)\,(\mathbf R_{\mathrm{mta}}\mathbf f_{\mathrm{mta}} - g\,\mathbf e_z) - (\mathbf R_{\mathrm{cra}} \mathbf f_{\mathrm{cra}} - g\,\mathbf e_z) \big\rVert .
$$

**Known artefact of $t_{\mathrm{mech}}$ (E34, documented; the pre-specified C2 keeps it).**
$a_{\mathrm{rel}}$ is **unsigned and unfiltered**. There is no low-pass, no lever-arm or rotation
(centripetal, tangential) correction, and every tilt error of either IMU leaks a fraction of $g$
into it; head motion and the residual motion of the previous closing add to it. In back-to-back
cycles it can stay above its rest threshold from the previous closing, so the last
sub-threshold sample, and with it the onset, lies long before the movement. In the audit $a_{\mathrm{rel}}$ was the earliest source in 58 % of open_close cycles, typically
15–60 ms before ω, and more than 200 ms before ω in about 20 % (D31). An early mechanical onset
shortens every lead (towards "EMG after motion"); the X2 events (§3.1) are aligned to the same
$t_{\mathrm{mech}}$. The source rows `lead_omega_ms` and `lead_opt_ms` (§3) show the leads
without it; C2-E (§3.3) uses $t_\omega$ as its primary event.

**α onset (C2-E, X6 B1; `mechanical_onsets` with `alpha`).**

$$
\alpha(t) = \frac{\mathrm d^2}{\mathrm dt^2}\,\big(H_{15}^{(4)} \ast\ast\ \theta_{\mathrm{quat}}\big)(t),
$$

$H_{15}^{(4)} \ast\ast$ = 4th-order Butterworth low-pass at 15 Hz applied forward and backward
(`sosfiltfilt`) per finite segment (segments not longer than the filter padding
$3(2n_{\mathrm{sos}}+1)$ stay NaN), then two gap-aware gradients (`lead.py:alpha_signal`). The
search spans are:
- opening ($+\alpha$): from `alpha_guard_s` (0.05 s) after the last sample before the ω peak at
  which the closing velocity $-\omega$ still exceeds the ω onset threshold (the end of the
  previous closing, so the tooth impact at occlusion is excluded; the window start if there is
  no such sample), to the ω peak;
- closing ($-\alpha$, the braking of the opening): from the same cycle's opening $t_\omega$ (the
  window start without one) to the closing ω peak; this span starts before the closing search
  window.

The rest noise `alpha_noise` = $3\,\mathrm{SD}(\alpha_{\mathcal R})$ (0 without a rest opening
angle). Cut-off `alpha_cutoff_hz` = 15 Hz, `alpha_order` = 4, sensitivity
`alpha_sensitivity_hz` = [10, 20].

**Latency chain (E34, firmware and host-code audit; the lead is not corrected for it).** The
measured lead is $\hat L = L + b$ with
- EMG stamped per sample at DRDY (ADS1299, 4 kSPS, not batched); the sinc group delay of
  ≈ 0.4 ms is not corrected (EMG late → $b$ −0.4 ms);
- BNO085 accelerometer ≈ 5 ms late (VD17), game rotation vector ≈ 9.5 ms late against the camera
  trigger (E23); a late mechanical stamp makes the lead too large;
- the 200 Hz onset grid ≈ +2.5 ms (half a step).

Net $b \approx +5 \ldots +8$ ms (in favour of "EMG first"), with about ±4–5 ms participant-level
variation. The 6 ms exposure-centre offset of the optics affects only the optical cross-check
`lead_opt_ms`. C2-E includes the 8 ms in its bias margin $\delta$ (§3.3).

**Per take (X11, an20 §3; 2026-11-06).** The IMU part of $b$ is now measured per take (stage
`acquisition`: lag of the jaw IMU's angular speed behind the optics at the exposure centre):
median `\numAcqImuLag` (3.9 ms) (6 ms more against the trigger edge, the E23 figure), but
`\numAcqImuLagOutliers` (3) takes lag `\numAcqImuLagOutlierMin` (12.7 ms) to `\numAcqImuLagOutlierMax` (13.4 ms), about `\numAcqImuLagExcess` (9.1 ms) more than the others,
constant within the take; cause not identified (an20 §3). For these participants $b$ is about
9 ms larger. The pre-specified C2 keeps the uncorrected chain; the sensitivity variant
`imu_latency` removes each participant's measured IMU lag (the `imu` stage shifts the IMU
stamps), so the IMU onsets sit on the optical time base: open-close opening lead
`\numLeadOpenFirstImuLat` (first onset) and `\numLeadOpenDriveImuLat` (driving burst) against
`\numLeadOpenFirst` / `\numLeadOpenDrive` (robustness row). The leads move by more than the
shift (r17a: first onset up, driving burst down, by 20–30 ms) because the EMG search window <!-- num-ok: rounded difference of the macros above -->
is anchored at the mechanical onset: moving it by a few milliseconds changes which burst is
the first one in the window (the window and rule dependence of D23 / D34). The inference
pattern is unchanged (first onset Holm p < 0.05, driving burst not; `\numLeadOpenFirstImuLatPHolm` /
`\numLeadOpenDriveImuLatPHolm`).

**Filter-phase policy.** Every offline signal that enters a lead is zero-phase or symmetric
(EMG: an03 §2; IMU ω and acceleration: an04 §5; optical angle: an02 §1), so neither side of
$L = t_{\mathrm{mech}} - t_{\mathrm{on}}$ is delayed by its filter. The causal twins and the X3
detector use the causal EMG chain on purpose (E27); the mechanical onset they are compared with
stays the offline one (the reference event).

**Parameters.** `stages.timing`: `vel_frac` = 0.05, `acc_k` = 3, `pre_s` = 0.3;
`stages.timing.c2e`: `alpha_cutoff_hz` = 15, `alpha_order` = 4, `alpha_frac` = 0.05,
`alpha_guard_s` = 0.05.

**Code.** `timing/lead.py:mechanical_onsets`, `_backward_onset`, `_alpha_onset`,
`alpha_signal`, `mechanical_baselines`.

**Paper wording.** Mechanical onsets were determined per cycle on the 200-Hz IMU grid by a
backward search from the peak opening (or closing) angular velocity to the last sample below
the larger of 5 % of the peak and three rest standard deviations; the onset was the following
sample. The same search on the jaw-minus-head acceleration magnitude, with a threshold at the
rest median plus three robust standard deviations, gave a second onset, and the pre-specified
mechanical onset was the earlier of the two. In the post hoc analyses the angular-velocity onset
alone served as the reference event, with the onset of the signed angular acceleration
(opening angle low-passed at 15 Hz, zero-phase) as a mechanistic cross-check.

## 3. Lead and real-time budget

**Window.** For each cycle and direction,
$[t_{\mathrm{mech}} - \texttt{emg\_pre\_s},\ t_{\mathrm{mech}} + \texttt{emg\_post\_s}] =
[t_{\mathrm{mech}} - 0.6\,\mathrm s,\ t_{\mathrm{mech}} + 0.2\,\mathrm s]$. A window with fewer
than $0.2 f_s$ EMG samples is skipped. $i_m$ is the first EMG sample at or after
$t_{\mathrm{mech}}$. Both detector runs (offline and causal) see only this window.

**Which onset** (`select_onset`, D6). With $O$ the offline onsets of the window and
$I_k = [k,\ \max(i_m, k+1))$:

$$
k^\ast = \max\Big\{ k \in O,\ k \le i_m :\ \frac{1}{\lvert I_k\rvert}\sum_{n \in I_k} \mathbb 1[s[n] > \eta] \ \ge\ 0.5 \Big\},
\qquad\text{else}\quad k^\ast = \min\{k \in O : k > i_m\},
$$

the *last* onset up to the movement whose burst is still on for ≥ 50 % of the samples to
$i_m$, else the first onset after it (a negative lead). The first-onset-in-window rule picked
earlier, unrelated bursts: 16 % of pilot leads lay within 150 ms of the window start.

**Causal decision:** $t_{\mathrm{dec}}$ = the first causal decision index $\ge k^\ast$, so that
it belongs to the same burst; NaN if there is none. The causal detector runs on the causally
cleaned EMG of the window (E27; formerly it ran on the zero-phase signal, so the "causal"
decision still looked ahead by the band-pass and mains-fit spread).

**Both rules are always computed.** The `onsets`, `summary` and cohort tables carry a `rule`
column: `first` = the frozen plan's first offline onset anywhere in the window, also after
$t_{\mathrm{mech}}$, $k = \min O$; `drive` = $k^\ast$. `stages.timing.onset_rule` (default
`drive`) only picks the primary rule for figures, tables and the specification curve; both rules
are confirmatory (*Aggregation*). The opening lead depends strongly on the rule (§3.0: under
`first` it is mostly the window bias), and the C2 verdict is "not confirmed" (D23,
D34). Values (`r17a`, 13 adults; median [BCa CI], Holm p): open_close first 108 [74, 248]
ms (n = 9, p 0.023), drive 49 [−38, 67] ms (n = 8, p 0.23); chewing first 245 [1, 364] ms
(n = 9, p 0.023), drive 38 [−10, 137] ms (n = 8, p 0.055); the D23 null medians 216–327 ms put
every observed − null difference below 0.

**Third, sensitivity rule `drive_taskbase`.** The driving-burst selection $k^\ast$ with the
threshold taken from this cycle's own pre-movement samples, the first `TASKBASE_S` = 0.3 s of the
window (−600 … −300 ms): $\eta_{\mathrm{tb}} = \bar s_{\mathrm{pre}} + k_{\mathrm{sd}}\,
\mathrm{SD}_0(s_{\mathrm{pre}})$, the detector signal pooled over the NaN-free runs of those
samples (and $\eta_\Psi$ likewise); needs ≥ $0.2 f_s$ finite samples. The elevated in-task
baseline then no longer counts as "already active". The causal twin takes its threshold from the
same 0.3 s of the causal chain. It enters the specification curve (3 detectors × 4 mechanical
sources × 3 rules) and the timing tables, not Fig. 2c. Under this rule the active-at-start gate
(below) compares the first 50 ms of the window with a threshold built from the first 0.3 s of the
same window, so it practically never fires (E34).

**Muscles:** digastric for opening (primary); masseter and temporalis for closing (secondary,
D5).

**Per channel-cycle** (table `timing.onsets`, all in ms; $> 0$: EMG first):

$$
\begin{aligned}
\texttt{lead\_ms} &= t_{\mathrm{mech}} - t_{\mathrm{on}}, &
\texttt{lead\_omega\_ms},\ \texttt{lead\_acc\_ms},\ \texttt{lead\_opt\_ms} &= t_\omega,\ t_{\mathrm{acc}},\ t_{\mathrm{opt}} - t_{\mathrm{on}},\\
\texttt{causal\_delay\_ms} &= t_{\mathrm{dec}} - t_{\mathrm{on}}, &
\texttt{anticipation\_ms} &= t_{\mathrm{mech}} - t_{\mathrm{dec}}
\end{aligned}
$$

(and `lead_alpha_ms` = $t_\alpha - t_{\mathrm{on}}$, C2-E P4). The anticipation is the real-time
budget: the margin a controller that decides causally would have before the movement.

**A channel-cycle is usable when all of the following hold** (`WindowOnsets.__call__`):
- $t_{\mathrm{on}}$ is finite;
- not `active_at_start`: $\mathrm{median}\big(s[0 : n_0]\big) \le \eta$, $n_0 = \lfloor 0.05 f_s
  \rfloor$, with the rule's threshold (rest threshold under `first` and `drive`,
  $\eta_{\mathrm{tb}}$ under `drive_taskbase`). Under `first` and `drive` this gate removes
  59–70 % of the opener channel-cycles (E34); under `drive_taskbase` it practically
  never fires;
- $\texttt{burst\_db} = 20\log_{10}(A/R) \ge$ `burst_min_db` (6 dB), with
  - $A$ = max over the *whole* −600…+200 ms window of the centred 50 ms masked envelope of the
    window (`masked_envelope`, odd window: masked samples left out of the RMS, NaN where < 50 %
    of the window is valid; E18, formerly masked samples were set to 0, which pulled the
    envelope down next to every mask);
  - $R$ = median over the unmasked `rest` samples of the channel's 150 ms masked envelope
    (`signals.emg.envelope_s`).

  The two envelope lengths differ (a 50 ms peak against a 150 ms rest median; E34), and $A$ is
  not tied to the selected burst (E33.13);
- finite share of the window (samples outside the analysis mask, an03 §2.4) $> 0.9$
  (`stages.timing.min_finite_share` = 0.9, a parameter since 2026-11-03; the change-point
  check of C2-E keeps the literal 0.9, `timing/c2e.py:248`; `stages.emg.cycle_mask_max` is not
  used here).

**Aggregation.**
- `timing.summary`: per pid × task × direction × muscle × detector × rule, the median of each
  column over the usable cycles (`n`, plus `n_total` = all channel-cycles). No minimum count here.
- `stats_timing.timing_cohort` (`stats/cohort_misc.py:per_participant_timing`), per task ×
  direction × detector × rule. **Per-participant value (D15, `stages.stats_timing.aggregation:
  pooled`):** $\tilde L_i = \mathrm{median}$ of the column over participant $i$'s usable
  channel-cycles of the direction's muscle group (digastric L/R for opening; masseter and
  temporalis L/R for closing), pooled over channels, so every usable channel-cycle weighs the
  same. A participant needs `stages.stats_timing.min_cycles` (3) usable channel-cycles in the
  group; participants with onset rows but fewer usable ones are excluded and counted
  (`n_excluded_min_cycles`). Before D15 the value was the unweighted mean over the group's
  muscles of the `summary` medians, so a muscle with one usable cycle weighed as much as one with
  twenty; this is kept as the sensitivity variant `c2_muscle_mean` (`aggregation: muscle_mean`,
  `min_cycles: 1`, `biomech sensitivity`, an08).
- The cohort gets `describe(stat="median")`: the median of $\tilde L_i$ with a BCa 95 % CI of the
  median (an07), for each of the six columns above. Only `lead_ms` and `anticipation_ms` also get
  a one-sided Wilcoxon signed-rank test of $\mathrm{median}(\tilde L) > 0$ (n ≥ 5;
  `stats.core.wilcoxon_one_sample`, effect size $r = z/\sqrt n$).
- **Families (D15)**, columns `family` and `p_holm`: *confirmatory* = the opening lead
  (`lead_ms`, direction opening) of `open_close` and `chewing` under the primary detector
  (`tkeo`) and both onset rules, the frozen `first` and the D6 `drive` (`CONFIRMATORY_RULES`,
  `stages.stats_timing.confirmatory_rules`): Holm across these four tests, so a C2 claim needs
  both rules; *exploratory* = every other row with a test, Holm within its metric (`lead_ms` or
  `anticipation_ms`) across all its task × direction × detector × rule rows; *descriptive* = the
  untested metrics (`p_holm` NaN).
- `timing_usability`: the mean of `usable` and of `active_at_start` over all channel-cycles, per
  pid × task × direction × detector × rule.
- `timing_spec` feeds the specification curve (an07).
- Sensitivity variants that touch the lead (`experiments/icorr2027/study.yaml`):
  `c2_symmetric_window` (window ±0.4 s), `d15_min_cycles_2` / `_5`, `burst_3db` / `burst_10db`,
  `c2_muscle_mean`, `onset_ksd_5` (onset threshold `stages.timing.k_sd` = 5 instead of 3, §3.4).

In cyclic, self-paced tasks with co-activation, about a quarter of the channel-cycles qualify
(TKEO, driving-burst rule, `r17a`; the usability flow `emg_usability_flow` gives the counts per
gate).

**Why so few, and the bias it causes.** *History (pilot, 12 participants, after D14; kept for the mechanism, not as a result; the current counts are in `timing_usability` and
an07 "Thresholds and exclusions" (end of an07)):* open_close opening, digastric, TKEO, D6 rule; 234
channel-cycles = 117 cycles × 2 digastrics; each channel-cycle counted once, first failing
criterion in the order of the table:

| Outcome | Channel-cycles |
|---|---|
| usable | 71 (30 %) |
| already above threshold at the window start (in-task baseline 5–15 dB above the rest phase that sets the threshold; pilot estimate, see below) | 132 (56 %) |
| burst < 6 dB above rest | 17 (7 %) |
| > 10 % of the window masked | 0 (pre-D14: 34) |
| burst ≥ 6 dB but no onset in the window (starts after +200 ms or never clears the threshold cleanly) | 14 (6 %) |

*Cohort measurement (change-log entry J34.2, r16a, 2026-11-14):* over the window's first 0.3 s the
openers' TKEO detector signal sits at about the rest threshold (rest mean + 3 SD), not 5–15 dB
above it: per-participant median `\numCTwoTaskbaseDbMedian` (+0.4 dB; range −3.8 to +6.0 dB);
the gate trips because the first 50 ms lie slightly above a threshold the in-task level already
reaches. Current counts: `c2_gate_flow` (301 of 472 at the start gate).

46 of 117 cycles (39 %) have at least one usable digastric; all 12 participants contribute,
three with only 1–2 cycles. The signals themselves are fine (all 24 digastric channels pass the
shape-usability QC); the restriction is the single-cycle detector. The last row biases the
usable cycles toward early onsets under both rules. Other detectors and rules: 27–52 % usable;
closers in closing: 26 %. Before D14 (low-frequency hits masked): 43 usable (18 %), 33 cycles,
11 participants.

**Code.** `timing/lead.py:analyse`, `WindowOnsets`, `select_onset`, `build_detector`;
`stats/cohort_misc.py:per_participant_timing`, `timing_tables`, `_timing_family`.

**Paper wording.** EMG onsets were searched from 600 ms before to 200 ms after each mechanical
onset. The pre-specified rule took the first onset in this window; a second rule took the last
onset before the movement whose burst was still active on at least half of the samples up to
the movement onset, and otherwise the first onset after it. A channel-cycle was analysed when an
onset was found, the median detector signal over the first 50 ms of the window did not exceed
the threshold, the peak of the 50-ms RMS envelope anywhere in the window exceeded the median
150-ms rest envelope by at least 6 dB and more than 90 % of the window was artefact-free. Per participant, the lead was the median over all usable channel-cycles of the
muscle group (at least three), and the cohort median was tested against zero with a one-sided
Wilcoxon signed-rank test, Holm-corrected across both tasks and both onset rules.

### 3.0 Chance null of the lead (D23, D23a) and what it tests (D34)

**Why.** The window is asymmetric (−600 … +200 ms), so a burst unrelated to the movement still
yields a positive lead under the `first` rule; the frozen one-sample Wilcoxon test against 0
does not test the procedure's own null. The null is reported next to the frozen test, which
stays primary and unchanged.

**Null per participant** (`timing/lead.py:pseudo_onset_null`; table `timing.lead_null`).
- **Construction (D23a), circular shift:** both protocol tasks run as continuous cycling, so they
  are in `NULL_SHIFT_TASKS` = (`chewing`, `open_close`). With the sorted real mechanical onsets
  $t^{\mathrm{real}}_0 < \dots < t^{\mathrm{real}}_{n-1}$ of the direction, pseudo-onset
  $j = 0, \dots, M-1$ ($M$ = `N_NULL` = 200 per task) is

  $$
  t^{(j)} = t^{\mathrm{real}}_{j \bmod n},\qquad
  \text{window samples}\ \ \big(i^{(j)} - \Delta_j + o\big) \bmod N,\quad o = -n_{\mathrm{pre}}, \dots, n_{\mathrm{post}},
  $$

  $i^{(j)}$ the first EMG sample at or after $t^{(j)}$, $n_{\mathrm{pre}}, n_{\mathrm{post}} =
  \mathrm{round}(0.6 f_s), \mathrm{round}(0.2 f_s)$, i.e. the EMG (zero-phase signal, causal
  chain, cleaned signal and mask together) circularly shifted within the phase by $\Delta_j$
  (`surrogates.uniform_shifts`, then `surrogates.shifted_windows`: index arithmetic). One shift
  per pseudo-onset, shared by both directions. The original D23 branch (pseudo-onsets drawn
  uniformly ≥ 1 s, `NULL_GAP_S`, from every real onset, from the start and peak of every cycle
  flagged not ok, and from the phase edges) is kept for tasks with pauses; none exists in this
  protocol (with it, 12 of 13 participants had no eligible span).
- **Same analysis:** the window, detectors, rules, causal twin and usability gates of the
  per-cycle analysis (`WindowOnsets`, built once per participant by `build_detector`).
- **Seed:** `surrogates.null_seed(participant seed, task)` = `default_rng([seed & 0x7FFFFFFF, crc32(task)])`.
- **Stored per pid × task × direction × detector × rule:** `null_kind` (`circular_shift`),
  `lead_null_ms` $= L_{0,i}$ = median lead over the usable pseudo channel-cycles of the
  direction's muscle group, pooled over channels as the D15 primary;
  `lead_null_muscle_mean_ms` (mean of the per-muscle medians, the `muscle_mean` aggregation),
  `anticipation_null_ms`, `usable_null_frac`, `n_null`, `n_null_rows`, `n_null_usable`,
  `null_span_s` ($T - 2$ s).

**Cohort test** (`stats/cohort_misc.py:null_pairs`, `null_test`; table
`stats_timing.timing_null` for the pairs, columns on the `lead_ms` rows of `timing_cohort`):
- per participant $d_i = \tilde L_i - L_{0,i}$ (`lead_minus_null_ms`); a null with fewer than
  `min_cycles` (3) usable pseudo channel-cycles is NaN;
- one-sided Wilcoxon signed-rank of $d > 0$ (`W_null`, `p_null`, `r_null`; n ≥ 5; an07 §2.1)
  and a one-sided sign-flip permutation test of the mean difference (`p_null_perm`; an07 §3.2:
  $B$ = `N_PERM_NULL` = 10 000 random sign vectors, study seed, NaN below 5 pairs);
- Holm within the same families as the frozen test (`p_null_holm`, `p_null_perm_holm`), so the
  confirmatory family is again the four opening-lead tests.

**What it tests (D34).** A burst unrelated to the motion lands anywhere in the asymmetric
window, so the null median is a fixed window bias (`r17a`: 216–327 ms). For a true,
time-locked lead $L$ the expected difference observed − null is about $L - 250$ ms, so the
one-sided test has essentially no power for physiological leads of tens of milliseconds: it is
calibrated, but tests "lead > the window bias of random bursts". It is therefore reported as a
**window-bias quantification**: the `first` rule's lead is a window artefact; $p_{\mathrm{null}}
\approx 1$ also shows that the observed onsets sit *closer* to the movement than random bursts
(coupling, not order). The synthetic regression case `Spec(lead_ms=+50)`
(`tests/test_r10_c2_null.py::test_positive_lead_has_no_power_against_the_d23_null`) documents
the missing power: the drive rule recovers +50 ms, but observed − null < 0 and $p_{\mathrm{null}}
> 0.5$. The coupling test the C2 question needs is C2-E (§3.3).

**Onset sources.** For both tasks the pseudo-onsets are the real primary onsets
$t_{\mathrm{mech}}$, so the null belongs to the primary source; the source rows
(`lead_omega_ms`, `lead_opt_ms`) only re-reference the same EMG onset and get no null of their
own (D31 addendum; the D31 promise of per-source nulls is withdrawn by D34, and the requirements
table carries no such columns).

**Paper wording.** To quantify the bias of the asymmetric search window, the per-cycle analysis
was repeated 200 times per participant and task with the EMG circularly shifted against the
motion by a random 1 s to T − 1 s (T, task duration), which keeps the EMG's own burst statistics
and removes any coupling to the movement. The median lead of these surrogate onsets (216–327 ms)
measured the lead an unrelated burst produces; observed minus surrogate leads were tested with
one-sided Wilcoxon signed-rank and sign-flip permutation tests.

### 3.1 Detector-free ensemble timing (`biomech/timing/ensemble.py`, X2)

**What and why.** Every cycle counts: instead of a single-cycle threshold, the envelope is
averaged over cycles aligned to the mechanical onset, and two events are read off the
participant's median curve.

**Curves.**
- Rest level $R_c$: the median over the `rest` phase of channel $c$'s 50 ms centred masked
  envelope (`ensemble.envelope_s`, odd window; masked samples left out of the RMS, NaN where
  < 50 % of the window is valid, and NaN at masked samples; E18, formerly masked samples were set
  to 0 before the RMS).
- Task envelope $\mathrm{env}_c(t)$: built the same way per task.
- Aligned dB curve of cycle $k$ on the grid $\tau \in \{-1000, -995, \dots, 1000\}$ ms
  (`window_ms`, `step_ms` = 5):

  $$
  y_{kc}(\tau) = 20\log_{10}\frac{\max(\mathrm{env}_c(t_{\mathrm{mech},k} + \tau),\ 10^{-9})}{R_c},
  $$

  the nearest EMG sample within half a grid step (2.5 ms), else NaN.
- Group curve $y_k(\tau) = \mathrm{nanmean}_{c \in G}\, y_{kc}(\tau)$ (in dB) over the group's
  channels; participant curve $Y(\tau) = \mathrm{nanmedian}_k\, y_k(\tau)$.
- Events come from every `ok` cycle with a mechanical onset, including cycles whose single-cycle
  window was skipped.

**Events** (`rise_time`, `release_time`; a NaN sample is never "below"):

| Event | Group | Definition (`stages.timing.ensemble`) |
|---|---|---|
| agonist **rise** | openers for opening, closers for closing | $b = \mathrm{med}_{\tau \in [-600, -300]} Y$; $\tau_{pk} = \arg\max_{\tau \in (-300, 800)} Y$ (open interval; it may precede the onset); $\Delta = Y(\tau_{pk}) - b$; $t = g_{i^\ast + 1}$ with $i^\ast = \max\{i < i_{pk} : g_i \ge -600,\ Y(g_i) < b + 0.25\,\Delta\}$ |
| antagonist **release** | closers for opening, openers for closing | $b = \mathrm{med}_{\tau \in [-600, -300]} Y$; $\Delta = b - \min_{\tau \in [-200, 300]} Y$; $t = \min\{g_i \ge -400 : Y(g_i) < b - 0.5\,\Delta\}$, $-400 = $ `base_ms[1]` − `release_lookback_ms` (100 ms, formerly hard-coded) |

If $\Delta < 3$ dB (`min_change_db`) there is no event (NaN). `change_db` = $\Delta$ (rise) or
$-\Delta$ (release). $\texttt{lead\_ms} = -t$ ($> 0$: EMG event first), as for the single-cycle
lead.

**Bootstrap.** Percentile 95 % CI: the cycles are resampled with replacement (`n_boot` 500,
`numpy.random.default_rng(participant seed)`), and the median curve and event are recomputed for
each resample. It runs only with ≥ 3 cycles. The CI is NaN unless > 50 % of the resamples find an
event; `boot_found` is that share.

**Outputs.**
- Table `timing.ensemble`.
- Arrays `ensemble__<task>__<direction>__<role>` (median curves) and `ensemble__grid_ms`.
- Cohort table `stats_timing.ensemble_cohort` (`cohort_misc.ensemble_cohort`): median with the
  BCa CI of the median, IQR, `n_found`, `n_before` (lead > 0), and a one-sided Wilcoxon test
  (> 0, n ≥ 5).
- Dashboard `timing_ensemble`, and rows in Fig. 2c.

The synthetic ground truth (opener lead 60–120 ms) is recovered within ±40 ms.

**Limits.**
- The events are aligned to the primary $t_{\mathrm{mech}} = \min(t_\omega, t_{\mathrm{acc}})$,
  so they inherit its early bias (§2, E34).
- X2 has no chance null: an event "before the onset" is not tested against what an unrelated
  envelope gives. C2-E P3 (§3.3) re-aligns the closer release to $t_\omega$ and tests it against
  the circular-shift surrogate.
- In back-to-back cycles the base window −600 … −300 ms can fall into the previous closing, so
  "release before opening" is not distinguishable from "end of the previous closing" without the
  cycle-shuffle check (§3.3).

**Reading.** Self-paced opening starts with the elevators letting go (gravity and the inferior
lateral pterygoid, which surface EMG does not record, start the motion); the digastric burst
follows. The earliest EMG intent cue for opening is therefore the closer release.
Values (`r17a`, 13
adults, `stats_timing.ensemble_cohort`, median [BCa CI]): open_close opener rise 80 ms after the
onset [20, 190] (12 of 13 after), closer release 70 ms before it [10, 150] (7 of 8 before,
p = 0.008); chewing closer release 265 ms before [50, 325] (n = 7, p = 0.008).

**Code.** `timing/ensemble.py:analyse`, `aligned_db`, `rise_time`, `release_time`,
`ensemble_events`; `stats/cohort_misc.py:ensemble_cohort`.

**Paper wording.** Without single-cycle detection, the 50-ms RMS envelope (in dB re the
channel's rest median) was aligned to every mechanical onset, averaged over the channels of a
muscle group and summarized per participant by the median over cycles. The agonist rise was the
time after which the median curve stayed above 25 % of the way from its pre-movement level
(median of −600 to −300 ms) to its peak, searched from −600 ms up to the peak; the antagonist release was the first time after −400 ms it fell below
the pre-movement level minus half of the drop to its minimum between −200 and +300 ms. Changes
below 3 dB counted as no event; 95 % confidence intervals bootstrapped the cycles.

### 3.2 Causal closer-release detector (`biomech/timing/release.py`, X3)

**What and why.** What a real-time controller could use from X2: a detector that sees only past
samples. Parameters (`stages.timing.release`) were fixed before the pilot data were analysed.

- **Input (E27):** the causally cleaned EMG of each task (an03 §2.9: tracking mains
  canceller and causal band-pass from a 2-s warm-up before the phase, analysis mask applied after
  filtering), so cleaning, envelope, baseline and decision all use past samples only.
- **Envelope** (`causal_db`): per closer $c$ (masseters, temporales), trailing moving RMS over
  $n_w = \mathrm{round}(0.05 f_s)$ samples, masked samples excluded,

  $$
  \mathrm{env}_c[n] = \Big(\frac{\sum_{j=0}^{n_w-1} v_c[n-j]\,x_c[n-j]^2}{\sum_{j=0}^{n_w-1} v_c[n-j]}\Big)^{1/2},\qquad
  E[n] = \frac{1}{\lvert C\rvert}\sum_{c} 20\log_{10}\frac{\max(\mathrm{env}_c[n], 10^{-9})}{R^{\mathrm{causal}}_c},
  $$

  $v_c$ = 1 for valid samples; NaN where $\sum v_c < 0.5\,n_w$; the mean over the closers with a
  value (`nanmean`); $R^{\mathrm{causal}}_c$ = median of the 50 ms masked envelope of the
  causally cleaned `rest` phase.
- **Baseline** (`detect`): a lagged running median on a 10 ms decimated copy $E_d[m] = E[m\,q]$,
  $q = \mathrm{round}(0.01 f_s)$:

  $$
  E_{\mathrm{base}}[n] = \mathrm{median}\big\{E_d[m] :\ m_n - l - W + 1 \le m \le m_n - l\big\},\quad
  m_n = \lfloor n/q \rfloor,\ l = \lfloor n_{0.3}/q \rfloor,\ W = \lfloor (n_{0.6} - n_{0.3})/q \rfloor,
  $$

  i.e. about $[t - 0.59, t - 0.30]$ s in 10 ms steps (`base_lo_s` 0.3, `base_hi_s` 0.6; lagged,
  so the drop does not lower it). NaN values of $E$ (all closers masked) are held at the last
  valid value (causal fill), values before the first valid one take that value, and a baseline
  window ending before the first valid value gives no baseline and no decision (E21; formerly NaN
  was replaced by the median of the whole phase, which looked ahead).
- **Decision:** $b[n] = \mathbb 1\big[E[n] \le E_{\mathrm{base}}[n] - \texttt{drop\_db}\big]$
  (3 dB); a decision at the sample where a run of $b = 1$ reaches $n_h = \mathrm{round}(
  \texttt{hold\_s}\,f_s)$ samples (30 ms), provided it is more than `refractory_s` (0.3 s) after
  the previous decision.
- **Per opening cycle** (mechanical onsets $t_{\mathrm{mech}}$ of §2): the first decision in
  $[t_{\mathrm{mech}} - 0.6\,\mathrm s,\ t_{\mathrm{mech}} + 0.2\,\mathrm s]$ (`search_pre_s`,
  `search_post_s`); $\texttt{anticipation\_ms} = t_{\mathrm{mech}} - t_{\mathrm{dec}}$ (> 0: known
  before the jaw moved); `detected`.
- **False releases:** decisions with no mechanical opening onset in $[t_{\mathrm{dec}} - 0.2,\
  t_{\mathrm{dec}} + 0.6]$ s; $\texttt{false\_per\_min} = n_{\mathrm{false}} / ((t_{\mathrm{last}}
  - t_{\mathrm{first}})/60\,\mathrm s)$ of the task phase.
- **Chance level** (`chance_level`; `chance_detected_frac`, `chance_anticipation_ms`): the
  decision train is shifted circularly within the task phase, $t \mapsto t_0 + ((t - t_0 + U_j)
  \bmod T)$, by $M$ = 200 uniform shifts $U_j \in [1\,\mathrm s, T - 1\,\mathrm s]$
  (`surrogates.uniform_shifts`, `shift_times`), and scored against the real opening onsets with
  the same search window; `chance_detected_frac` = detected share over all shifts × onsets,
  `chance_anticipation_ms` = median anticipation over the detected ones (D23a: the continuous
  open_close and chewing cycling leaves no span ≥ 1 s from every onset, so the former
  pseudo-onsets ≥ 1 s from every real onset could not be drawn; the pseudo-onset branch is kept
  for tasks with pauses). Drawn with `numpy.random.default_rng(participant seed)` (derived from
  the study `seed`, an08 §1; E21, formerly a fixed seed 0; the same seed for every task and sweep
  setting). With ~20 false releases/min a chance decision often falls in the 0.8-s window, so
  `detected_frac` is judged against this level and the anticipation is biased early by up to that
  share.
- **Cohort** (`stats/cohort_misc.py:release_cohort`, table `stats_timing.release_cohort`, per
  task × setting): describe of the per-participant median anticipation (median, BCa CI) with a
  one-sided Wilcoxon test > 0; `detected_frac_median`, `chance_detected_frac_median`,
  `detected_minus_chance_median` (median of the per-participant differences), `false_per_min_median`;
  and, per participant pair, the **two-sided** paired signed-rank test of the anticipation
  against `chance_anticipation_ms` (`_vs_chance`: `chance_anticipation_ms_median`,
  `anticipation_minus_chance_ms_median`, `W_vs_chance`, `p_vs_chance`, `r_vs_chance`; D23).
- **What carries a statement (D34).** The causal anticipation is *below* its chance level (`r17a`: 103 vs 229 ms, paired p = 0.017): a decision unrelated to the opening
  anticipates more than the real one, so the anticipation is not evidence of an early cue. Only
  the detection rate above chance (`r17a`: 80 vs 61 %) carries a statement ("detection, not
  anticipation").
- **Sensitivity sweep:** `drop_db` ∈ {2, 3, 6} dB × `hold_s` ∈ {20, 30, 50} ms; the frozen
  setting (3 dB, 30 ms) is marked `primary`.
- **Tables:** `timing.release` (per cycle, primary setting), `timing.release_summary` (per
  participant × task × setting: detected share, chance level, median anticipation, false
  releases/min), `stats_timing.release_cohort`. Shown as the supporting row "causal release" in
  Fig. 2c and in the requirements table.
- **Relation to X4:** formerly the cleaning here was the offline (zero-phase) chain and only the
  envelope and the decision were causal. Since E27 the chain is causal too; the stream replay of
  X4 (`ml_online`, docs/ml/ml08 §4) differs in running one filter continuously from the stream
  start (mains frequency from the pre-roll) instead of restarting per task, and in its rest
  level (ml08 §4). Its chance level follows the same rule as here, the circular shift in
  open_close and chewing (`NULL_SHIFT_TASKS`) with the participant seed (E42; before,
  the replay drew pseudo-onsets ≥ 1 s from every opening in every task, §5).

**Code.** `timing/release.py:causal_db`, `detect`, `_decisions`, `evaluate`, `chance_level`,
`analyse`; `stats/cohort_misc.py:release_cohort`, `_vs_chance`.

**Paper wording.** A causal detector of closer relaxation computed the 50-ms trailing RMS
envelope of the causally filtered masseter and temporalis EMG (dB re rest, averaged over the
four muscles) and declared a release when the envelope stayed at least 3 dB below its running
median 300–600 ms earlier for 30 ms (refractory period 300 ms). An opening was detected when a
release fell between 600 ms before and 200 ms after its mechanical onset; the chance level was
obtained by circularly shifting the release times 200 times within the task. Parameters were
fixed before the data were analysed; drops of 2 and 6 dB and hold times of 20 and 50 ms were
reported as sensitivity analyses.

### 3.3 C2-E: the post hoc timing family (X6)

**Status.** Post hoc, specified before computing (2026-10-30) and validated on synthetic data
before any real-data run. The pre-specified C2 rows and verdict ("not confirmed") are unchanged.
Seen before the entry: the closing leads of both tasks and the X2 release values (printed during
the audit); the primary endpoint is therefore defined task-pooled. Evidence label in every
artefact: post hoc. Code: `timing/c2e.py` (per participant, in the `timing` stage),
`timing/surrogates.py`, `timing/locking.py`, `timing/coupling.py`, `timing/changepoint.py`,
`stats/cohort_c2e.py` and `stats/families.py` (cohort, in `stats_timing`). Parameters:
`stages.timing.c2e` in `etc/biomech.yaml`. Reader name of the family: the *event-order family*.

**Why it exists.** The pre-specified test probes the digastric for opening, but self-paced
opening starts with closer relaxation (and the unrecorded inferior lateral pterygoid); its
mechanical onset is biased early (§2, E34); its gates differ between rules (§3); and its chance
null measures the window bias, not coupling (§3.0, D34). C2-E asks the physiologically predicted
questions — closers before closing, closer release before opening — with a primary event free of
$a_{\mathrm{rel}}$ and a test of time-locking against surrogates.

#### Building blocks

- **B1, mechanical events** (`c2e.cycle_events`, per `ok` cycle of the task in time order;
  `mechanical_onsets`, §2):
  - $t_\omega$, the relative-quaternion angular-velocity onset: the **primary** event (no
    $a_{\mathrm{rel}}$);
  - $t_\alpha$, the signed angular-acceleration onset at `alpha_cutoff_hz` = 15 Hz (4th order,
    zero-phase; `alpha_order` 4), with 10 / 20 Hz as sensitivity (`alpha_sensitivity_hz`);
  - $t_{\mathrm{opt}}$, the optical onset (cross-check);
  - cycle-shuffle anchors: $a_k^{\mathrm{open}}$ = `opening_anchor` = the end (`t_end_us`) of the
    cycle just before in time order among **all** detected cycles of the task, `ok` or flagged
    (`prev_end_us`; NaN for the first cycle of the task), i.e. the end of the previous closing;
    $a_k^{\mathrm{close}}$ = `closing_anchor` = the same cycle's opening $t_\omega$. (Erratum E41: the anchor was the end of the previous *`ok`* cycle, one cycle too early after a
    flagged cycle, 11 of 231 openings; only the descriptive structure rows changed.)
- **Onsets** (`WindowOnsets.restricted()`: TKEO, `drive_taskbase`, offline only; §1, §3): window
  of $n_{\mathrm{pre}} + n_{\mathrm{post}} + 1$ samples around $i_{\mathrm{ref}}$ = the first EMG
  sample at or after $t_{\mathrm{event}}$ ($n_{\mathrm{pre}} = \mathrm{round}(0.6 f_s)$,
  $n_{\mathrm{post}} = \mathrm{round}(0.2 f_s)$; `emg_pre_s`, `emg_post_s`), the same usability
  gates as C2 (onset found, not active at the window start, `burst_db` ≥ 6 dB, > 90 % finite).
  With $k$ the window sample of the selected onset,

  $$
  L = \big(t_{\mathrm{event}} - t_{i_{\mathrm{ref}}}\big) + \frac{n_{\mathrm{pre}} - k}{f_s}.
  $$

  NaN-free windows are evaluated row-wise in arrays (`WindowOnsets.batch`,
  `lead._tkeo_taskbase_rows`, values identical to the scalar path,
  `tests/test_r11_c2e_units.py`); windows with masked samples take the scalar path. Every
  observed window, including the symmetric one, must lie inside the phase ($i_{\mathrm{ref}} -
  \mathrm{round}(0.6 f_s) \ge 0$ and $i_{\mathrm{ref}} + \mathrm{round}(0.4 f_s) < N$); cycles
  whose window would leave it are not evaluated (observed or surrogate).
- **Further onset sources in the same windows:**
  - re-referenced leads (P4, cross-check): the same onsets against $t_\alpha$ (15 / 10 / 20 Hz)
    and $t_{\mathrm{opt}}$, $L_{\mathrm{src}} = L + (t_{\mathrm{src}} - t_\omega)$, usable only
    where $t_{\mathrm{src}}$ is finite;
  - P6 changepoint onsets (below);
  - the symmetric window ±`symmetric_window_s` = ±0.4 s (the pre-stated fallback statistic).
- **B2, circular-shift surrogate S1** (`surrogates.s1_sample_shifts`): $M$ = `n_surrogates` = 200
  whole-phase circular shifts $\Delta_j = \mathrm{round}(U_j f_s)$, $U_j \sim \mathcal U[$`gap_s`,
  $T -$ `gap_s`$] = \mathcal U[1\,\mathrm s, T - 1\,\mathrm s]$, of the EMG phase (zero-phase EMG,
  causal chain, cleaned signal and analysis mask together) per task; every real cycle is
  re-evaluated in every surrogate with the same windows and gates, reading sample $(i - \Delta_j)
  \bmod N$ (index arithmetic). One shift set serves both directions, every source and the
  release; seed `null_seed(participant seed, "c2e:<task>")`. It keeps the EMG's own statistics
  (burst rate, durations, amplitudes, masks) and destroys every coupling to the motion: the
  per-participant null of any statistic. Surrogate $j = 0$ is the observed EMG ($\Delta_0 = 0$).
- **B3, cycle-shuffle surrogate S2** (`surrogates.s2_permutations`, `s2_shuffle`, `n_s2` = 200):
  with anchor $a_k$ and EMG event $e_k$ of cycle $k$ (all channels of a cycle together) and a
  random permutation $\pi$ of the cycles,

  $$
  e^\ast_k = a_k + \big(e_{\pi(k)} - a_{\pi(k)}\big).
  $$

  It keeps everything locked to the rhythm and destroys only the locking to the upcoming onset:
  the null of "locked to the rhythm, not to the movement". Seeds `"c2e:s2:<task>:<direction>"`
  and `"c2e:s2rel:<task>"`.
- **B4, bias margin** $\delta$ = the synthetic location bias of the mixture at a true lead of 0
  (G4: 8.6 ms; 8.4–9.1 ms over the simulated leads) plus the 8 ms latency chain (§2, E34),
  rounded up: **$\delta$ = 17 ms** (parameter `delta_ms`, stored in the `c2e_cohort` column
  `bias_margin_ms`, before 2026-11-03 `delta_ms`; fixed by the synthetic gate before any
  real-data run; without it the decision reads "delta not fixed").
- **B5, intersection-union test** (`families.iut`): a hypothesis holds only if its locking part
  (A) and its location part (B) both reject; $p_H = \max(p_A, p_B)$ (NaN if either is NaN). The
  IUT is a level-α test of the intersection of the two alternatives without further correction
  (an07).

#### Per-participant statistics (`c2e._stats`, `timing/locking.py`; table `timing.c2e_null`)

One row per participant × scope (each task, and `pooled` = both tasks: the channel-cycles
concatenated, surrogate $j$ of each task joined, $\min$ of the tasks' surrogate counts) ×
direction (closing: the closers masseter and temporalis, primary; opening: the digastrics) ×
source (`omega`, `alpha15`, `alpha10`, `alpha20`, `opt`, `cp`, `sym`, and `release`). Units ms;
lead > 0 = EMG first. Let $\mathcal L_0$ be the usable observed leads (channel-cycles) and
$\mathcal L_j$ those of surrogate $j$.

- **Usable count and flag** (`c2e._flag`): `ok` with $\lvert\mathcal L_0\rvert \ge$ `min_usable`
  = 5; `few_usable` with 3–4 (`flag_usable`: counted, not tested); `too_few` below; plus
  `no_phase_or_cycles` / `no_mechanical_signal` rows.
- **P1 locking, part A** (`locking.locking`). A time-locked EMG has a smaller spread than chance:

  $$
  \lambda = \log\frac{\mathrm{IQR}(\mathcal L_0)}{\mathrm{median}_{j \in J}\ \mathrm{IQR}(\mathcal L_j)},\qquad
  p_{\mathrm{lock}} = \frac{1 + \#\{j \in J : \mathrm{IQR}(\mathcal L_j) \le \mathrm{IQR}(\mathcal L_0)\}}{1 + \lvert J\rvert},
  $$

  $J$ = surrogates with ≥ 5 usable leads; IQR needs ≥ 5 values. Columns `iqr_obs`,
  `iqr_null_median`, `log_ratio` ($\lambda < 0$: locked), `p_lock`, `n_surr`.
- **P1 location, part B** (`locking.mixture_location`). Maximum-likelihood fit to $\mathcal L_0$
  (restricted to $[l_o, l_h]$) of

  $$
  f(l) = \pi\,\mathrm{TN}(l; \mu, \sigma, [l_o, l_h]) + (1 - \pi)\,f_{\mathrm{bg}}(l),\qquad
  \mathrm{TN}(l; \mu, \sigma, [l_o, l_h]) = \frac{\phi\big((l - \mu)/\sigma\big)/\sigma}{\Phi\big((l_h - \mu)/\sigma\big) - \Phi\big((l_o - \mu)/\sigma\big)},
  $$

  with the background density of the pooled surrogate leads $\mathcal L_{1..M}$ on a 1-ms grid,

  $$
  f_{\mathrm{bg}}(l) \propto (1 - \varepsilon)\,\frac{\sum_{v} \big[K_h(l - v) + K_h(l - (2l_o - v)) + K_h(l - (2l_h - v))\big]}{\text{norm}} + \frac{\varepsilon}{l_h - l_o},
  $$

  $K_h$ a Gaussian kernel of bandwidth $h$ = 20 ms (edge reflection), $\varepsilon$ = 0.02 (the
  uniform floor), normalised to sum 1 over the 1-ms grid and linearly interpolated at the leads
  (uniform without ≥ 2 surrogate leads). $[l_o, l_h]$ = the lead range of the window, $[-200,
  +600]$ ms for $t_\omega$, widened by $\min$ / $\max$ of the re-reference offsets for α /
  optical, $[-400, +400]$ ms for the symmetric window; $\sigma \in [5, 150]$ ms. Start: a grid of
  $\mu$ every 5 ms × 8 log-spaced $\sigma$, $\pi$ from 25 EM fixed-point steps
  $\pi \leftarrow \frac{1}{n}\sum_i \frac{\pi\,\mathrm{TN}(l_i)}{\pi\,\mathrm{TN}(l_i) + (1-\pi) f_{\mathrm{bg}}(l_i)}$
  from $\pi = 0.5$; then L-BFGS-B over $(\mu, \log\sigma, \pi)$ within the bounds (the grid point
  is kept if the refinement does not improve on it). $\pi <$ `pi_min` = 0.1 → $\mu$ NaN
  (`no_locked_component`); fewer than 5 leads → `few_leads`. `loglik_gain` $= \ell(\hat\theta) -
  \sum_i \log f_{\mathrm{bg}}(l_i)$. $\mu$ is the location of the time-locked bursts, free of the
  window bias.
- **Fallback statistic:** `median_ms` $= \mathrm{median}\,\mathcal L_0$, `median_null_ms` =
  median of the pooled surrogate leads, `median_minus_null_ms` = their difference. The synthetic
  gate showed it unusable as a location estimate (bias 96–179 ms), so it is a supporting row
  only.
- **Cycle-shuffle locking** of the event source per task (`locking.s2_iqr`): the lead under S2 is
  $L^\ast_{kc} = (t_{\mathrm{event},k} - a_k) - (t_{\mathrm{on},\pi(k)c} - a_{\pi(k)})$, usable
  flags permuted with the onsets; `iqr_s2_median` = median S2 IQR, `p_s2` =
  $p_{\mathrm{MC}}$ of an S2 IQR ≤ the observed.
- **P3, closer release before opening** (`c2e._release`, `ensemble.release_s1`; source
  `release`, table `timing.c2e_release` per cycle): the X2 closer envelope (50 ms masked RMS, dB
  re rest; X2 parameters unchanged) aligned to the opening $t_\omega$ (grid points outside the
  phase at the real event are NaN for every shift); from the median curve the release time $t$
  and depth $D = b - \min_{[-200, 300]} Y$ (`release_time`), and per cycle the release times
  $t_k$ (`release_per_cycle`, the same rule on each cycle's own group curve). Statistics:
  - $\texttt{log\_depth\_ratio} = \log\big(D_0 / \mathrm{median}_j D_j\big)$, `p_depth` =
    $p_{\mathrm{MC}}$ of $D_j \ge D_0$ (greater), $D_j$ the median-curve depth under surrogate
    $j$;
  - locking of the per-cycle release leads $-t_k$ against S1 (as P1 A; usable = cycles with a
    release, flag from their count);
  - `median_ms` $= -t$ (the release lead of the median curve);
  - cycle structure: the Theil–Sen slope

    $$
    b = \mathrm{median}_{i < j,\ x_i \ne x_j}\ \frac{y_j - y_i}{x_j - x_i},\qquad x_k = t_{\omega,k} - a_k,\ \ y_k = t_{\mathrm{rel},k} - a_k,
    $$

    $a_k$ = the opening anchor, against S2 (offsets $y$ permuted across cycles, $x$ fixed;
    `b_s2`, `b_null_median`, `p_b_s2` = $p_{\mathrm{MC}}$ of a null slope ≥ $b$); $b \approx 1$:
    locked to the upcoming opening, $b \approx 0$: locked to the end of the previous closing;
    `x_sd_ms` = $\mathrm{SD}(x)$ (ddof 1; S2 has power only with enough variation; G6).
- **P4, α-referenced leads:** descriptive per participant; the mechanistic event.
- **P5, continuous coupling** (`timing/coupling.py:coupling`, table `timing.coupling`, per
  task). Electromechanical delay between closer activity and the closing acceleration:
  - EMG feature $e(t)$: per closer the rectified EMG $\lvert x\rvert$ through a mask-aware centred
    moving mean of `smooth_s` = 5 ms (NaN where < 50 % valid), mean over the closers, sampled on
    the IMU 200 Hz grid by nearest sample (`emg_feature`);
  - kinematic feature $-\alpha(t)$, $\alpha$ the unfiltered second derivative of $\theta_{\mathrm{quat}}$,
    masked ±`occlusion_mask_s` = 50 ms around each occlusion (cycle end);
  - both through the **same** 4th-order 15 Hz zero-phase Butterworth per finite segment
    (identical $\lvert H\rvert^2$, so the relative lag is unbiased): $\tilde e$, $\tilde\alpha$;
  - both residualised: $r_z(t) = z(t) - \bar z(\varphi(t))$, $\varphi = (t - s_c)/(s_{c+1} -
    s_c)$ the cycle phase (cycles from start to the next start, the last to its end), $\bar z$ the
    mean over all cycles in `n_phase_bins` = 50 phase bins, linearly interpolated between bin
    centres;
  - Pearson correlation at every lag (`lagged_corr`, `center=True`, the one lagged-correlation
    helper also behind the C3a IMU lag):

    $$
    \rho(\ell) = \mathrm{corr}\big(r_{e}[i],\ r_{\alpha}[i + \ell]\big)\ \text{over the pairs where both are finite},\quad
    \lvert\ell\rvert \le \mathrm{round}(150\,\mathrm{ms}/\Delta t),
    $$

    NaN with fewer than `min_pairs` = 200 pairs; `emd_lag_ms` $= \Delta t \cdot \arg\max_\ell
    \rho(\ell)$ (> 0: the acceleration follows the EMG, EMG first), `emd_r` $= \max_\ell
    \rho(\ell)$;
  - circular-shift null: `n_null` = 200 shifts (uniform [1 s, T − 1 s]) of $\tilde e$ (`np.roll`
    on the IMU grid, then residualised), `p_coupling` = $p_{\mathrm{MC}}$ of a null peak
    correlation ≥ the observed;
  - raw (not residualised) `emd_lag_raw_ms`, `emd_r_raw` are stored and pre-stated as not
    interpretable as onset order.
  - **Paper wording (J11).** The quotable lag is this stored per-participant
    `emd_lag_ms` (rectified, residualised EMG; P5 median over participants
    `\numCTwoECouplingOpenClose` / `\numCTwoECouplingChewing` (10 / 15 ms), open-close /
    chewing). The supplement figure `emg_motion_coupling` (form `xcorr`) correlates the
    smoothed closer *envelope* with the closing acceleration over the cohort; its peak (about <!-- num-ok: read off the supplement figure, not a stored result -->
    −30 ms, envelope smoothing included) is a picture of the coupling, not a lag estimate, and
    is not quoted.
- **P6, variance changepoint** (`timing/changepoint.py:variance_changepoints`, AGLR family after
  Staude et al. 2001): the window's maximum-likelihood change time of a variance *increase*
  (zero-mean Gaussian model), threshold-free. With $n$ valid samples, $n_1(k), n_2(k)$ the valid
  counts before / from sample $k$ and the mean squares $s_0^2, s_1^2(k), s_2^2(k)$ over the
  window, before and from $k$ (masked samples left out):

  $$
  \mathrm{LLR}(k) = \tfrac12\big[n\log s_0^2 - n_1\log s_1^2(k) - n_2\log s_2^2(k)\big],\qquad
  \hat k = \arg\max_{k \in [m,\ n_w - m],\ s_2^2 > s_1^2,\ n_1, n_2 \ge \lfloor m/2\rfloor} \mathrm{LLR}(k),
  $$

  $m = \max(\mathrm{round}(\texttt{cp\_min\_seg\_s}\,f_s), 2)$ (50 ms); usable when $10\log_{10}
  (s_2^2/s_1^2) \ge$ `burst_min_db` (6 dB) and > 90 % of the window is valid. Threshold-free, so
  it recovers channel-cycles that the threshold detectors reject as active at the window start:
  a detector sensitivity for P2 / P3 and the digastric re-test.

#### Perspectives, hypotheses and multiplicity (`stats/cohort_c2e.py`, table `stats_timing.c2e_cohort`)

Participants with ≥ 5 usable channel-cycles enter a test; `n_flagged` counts those with 3–4.
Every test is a `stats/families.py` building block: signed-rank tests with the exact null
distribution where there are no zeros or ties, else the normal approximation
(`wilcoxon_signed`), BCa CIs of the median (`stats.core.describe`). The location test of values
$v_i$ against a margin $m$ is the signed-rank test of $v_i - m$ (`location_test`); the locking
test is the signed-rank test of $\lambda_i$ (`log_ratio_test`).

| Key | Family | Hypothesis | Rows used | Test |
|---|---|---|---|---|
| `h1` | primary | **H1** closers lead closing, task-pooled (P1 / P2) | scope `pooled`, closing, source `omega` | IUT of A: signed-rank of `log_ratio` < 0 and B: signed-rank of $\mu - \delta > 0$ (BCa CI of the median $\mu$); rejected at α = 0.05 |
| `h2`, `h3` | secondary | **H2 / H3** closer release before opening in open_close / chewing (P3) | task scope, opening, source `release` | IUT of A: `log_depth_ratio` > 0 and B: release lead (`median_ms`) $- \delta > 0$; **serial gatekeeping**: tested (Holm over the two) only if H1 is rejected, else NaN and "not tested" |
| `p4_alpha` | supporting | closers lead the α onset (P4) | pooled, closing, `alpha15` | signed-rank of $\mu - \delta > 0$ |
| `p6_changepoint` | supporting | changepoint onsets lead closing (P6) | pooled, closing, `cp` | signed-rank of $\mu - \delta > 0$ |
| `optical` | supporting | closers lead the optical onset | pooled, closing, `opt` | signed-rank of $\mu - \delta > 0$ |
| `symmetric_window` | supporting | symmetric window: median − circular-shift null > 0 | pooled, closing, `sym` | signed-rank of `median_minus_null_ms` > 0 |
| `p5_open_close`, `p5_chewing` | supporting | EMG-to-α lag > 0 (P5) | `timing.coupling` | signed-rank of `emd_lag_ms` > 0 (margin 0, all participants with a lag); note: participants with `p_coupling` ≤ 0.05 |
| `digastric` | supporting | digastric locked to opening (expected: locked, negative lead) | pooled, opening, `omega` | signed-rank of `log_ratio` < 0; median $\mu$ with BCa CI |
| `s2_h1` | structure | closer locking beyond the rhythm (cycle shuffle) | per task, closing, `omega` | descriptive: median [BCa CI] of the per-participant mean over the tasks of $\log(\mathrm{IQR}_{\mathrm{obs}}/\mathrm{IQR}_{\mathrm{S2}})$; **cohort p withheld** |
| `s2_release_<task>` | structure | release slope $b$ (cycle shuffle) | task, opening, `release` | descriptive: median Theil–Sen $b$ [BCa CI]; note: median SD$(t_\omega - a)$, and the pre-stated wording when it is below `s2_min_sd_ms` = 232 ms; **cohort p withheld** |
| `decision` | decision | the decision rule below | — | — |

**Multiplicity.**
- H1 alone at α = 0.05 (`p_adj` = `p`).
- H2 / H3 (`families.gatekeeping`): if $p_{H1} \le \alpha$, Holm over $\{p_{H2}, p_{H3}\}$;
  otherwise both `p_adj` NaN ("not tested").
- Post hoc multiplicity sensitivity (decision of the authors 2026-10-31, after the results;
  `cohort_c2e.holm_family`): `p_holm_family` on the H1, H2 and H3 rows = Holm over the three
  IUT p values, always computed, also when the gatekeeping stops; labelled "post hoc
  sensitivity: Holm over H1–H3 (not the pre-specified procedure)". The pre-specified gatekeeping
  (`p_adj`, `rejected`, the decision) is unchanged.
- The supporting block gets Holm within itself (`families.block_holm`); no claim rests on it. A
  supporting row in `supporting_veto` (P4, both P5 rows, P6, optical, symmetric window; not the
  digastric, whose lead is expected to be negative) with a CI entirely below 0 ($\mathrm{ci\_hi}
  < 0$) is a veto.

**Why the cycle-shuffle cohort p is withheld (gate decision 2026-10-31).** Under rhythm-locked
synthetic EMG the cohort S2 row rejected in 0.095 of 201 null cohorts at nominal 0.05 (not
calibrated), and S2 has no power for back-to-back cycles (G6: power ≥ 0.8 only for
SD$(t_{\mathrm{mech}} - \text{anchor}) \ge 232$ ms). The rows report the effect size only; the
per-participant S2 values stay in `c2e_null`. No claim used them; the decision rule is unchanged.

**Decision rule** (`cohort_c2e.decide`):
- **"shown (post hoc)"** requires all of: H1 rejected; the BCa lower bound of the cohort median
  $\mu$ > $\delta$; no veto row; the synthetic gates passed (`gates_passed`);
- **"inconclusive (underpowered)"** when H1 is not rejected and the gate power at 40 ms
  (`gate_power_40ms`) is below 0.5;
- otherwise **"not shown"** ("delta not fixed" while $\delta$ is missing).

**Requirements rows** (`stats/requirements.py:_c2e_rows`; evidence `post_hoc`): `c2e_h1_ms`
"closer activation before closing" (the median $\mu$ of H1 with its BCa CI) and `c2e_h2_ms`
"closer release before opening (null-calibrated)" (H2), each with the IUT p, the gatekeeping
state and the decision in the note.

**Paper wording (methods).** As a post hoc analysis specified before computing, we tested the
event order within the cycle: closer activity before the closing onset and closer relaxation
before the opening onset, using the angular-velocity onset as the reference event. Per
participant, onsets (Teager–Kaiser detector, threshold from the first 300 ms of each window) were
compared with 200 circular-shift surrogates of the EMG. Time-locking was the log ratio of the
interquartile range of the leads to the median surrogate interquartile range; the location of
the time-locked component was the location parameter of a normal distribution truncated to the
analysis window, fitted by maximum likelihood as a mixture with the surrogate lead density. Closer activation before closing (hypothesis 1, both tasks
pooled) required both a cohort locking ratio below 0 and a cohort location above a bias margin
of 17 ms (one-sided Wilcoxon signed-rank tests, intersection-union p = the larger of the two);
closer release before opening in each task (hypotheses 2 and 3) was tested only if hypothesis 1
was rejected (serial gatekeeping, Holm), with the release depth against its surrogates and the
release lead above the margin.

#### Synthetic gates (fixed before any real-data run; `tests/test_r11_c2e_synth.py`, slow)

Synthetic participants were made from the cycle timing and usable shares of the store frozen at
the time (`r10`) only; no participant lead was read. The gates are a record of that date and
are not recomputed with later stores. Results as listed in the change log (X6 entries of
2026-10-30 and 2026-10-31) and in `etc/biomech.yaml` (`timing.c2e`); test functions `test_g1_calibration` …
`test_g6_release_and_s2_identifiability`:

| Gate | Criterion | Result | Status |
|---|---|---|---|
| G1 calibration | uniform per-participant p; cohort type I error ≤ 0.07 over ≥ 200 cohorts of 13 | locking p uniform (KS p = 0.16, P(p ≤ 0.05) = 0.045); H1 type I error 0.055 (locking part 0.070; the location part alone rejects in 0.40 of null cohorts, so only the IUT protects) | pass |
| G2 cycle structure | S2 not anti-conservative under rhythm-locked EMG | per task 0.046; cohort S2 row 0.095 over 201 cohorts | per task pass; cohort S2 p withheld (above) |
| G3 power | L ∈ {0, 20, 40, 80} ms, jitter 20 / 40 ms, usable shares as in that store | H1 power 0.0 at L = 0; 1.0 at 20 / 40 / 80 ms (jitter 20 ms); 0.95 / 1.0 / 1.0 (jitter 40 ms); `gate_power_40ms` = 1.0 (likely optimistic: clean synthetic bursts) | pass |
| G4 bias and coverage | bias → δ; coverage of the bias-corrected BCa CI ≥ 0.90 at every L | bias 8.4–9.1 ms → δ = 17 ms; coverage 0.888 / 0.892 / 0.903 / 0.933 at L = 0 / 20 / 40 / 80 (SE ≈ 0.02) | documented deviation: mixture kept (fallback unusable, bias 96–179 ms) |
| G5 artefacts | ω / α leads move < 5 ms under head motion and tilt error; artefact → lag ≈ 0; reflex → lag < 0 | ω and α move 0.0 ms (\|a_rel\| onset: 5 ms median, up to 15.5 ms); reflex lag −70 ms; torque model with 40 ms delay recovered; motion-artefact lag −12.5 ms (criterion \|lag\| ≤ 10 ms) | documented deviation: accepted (biases the supporting P5 towards "EMG after motion", the conservative direction) |
| G6 release | release truth within ±15 ms; S2 identifiability limit | median error −10 ms, P90 \|error\| 10 ms; depth beats S1 in 96 %; S2 power ≥ 0.8 only for SD(t_mech − anchor) ≥ 232 ms (`s2_min_sd_ms`) | pass; S2 expected powerless on real (back-to-back) cycles |

`gates_passed: true` (2026-10-31: G1, G3, G6 pass; G4 and G5 documented deviations; G2 handled
as above).

**Wording rule.** Under rhythm-locked synthetic EMG (locked to the previous event, not to the
movement) H1 rejects in 100 % of cohorts: in back-to-back cycles "closers lead closing" cannot be
separated from "closer activity keeps the rhythm". Every statement is therefore about the
**temporal order within the cycle** ("closer activity precedes the closing onset"), never a
causal drive. For H2 / H3 the pre-stated wording "precedes, not distinguishable from the end of
closing" applies whenever the cycle-shuffle check has no power (median SD$(t_\omega -
\text{anchor})$ < 232 ms).

**Reporting population.** The paper reads the `reporting_population` store (13 adults, D35). The participant recorded
outside the protocol has no jaw IMU: its `imu` unit fails and the `timing` stage (which depends on it) is skipped, so that participant has
no C2-E rows in either store and the C2-E values are identical with and without it. (The
optical-event branch of `cycle_events` for a participant without IMU arrays is therefore not
reached in this study.)

**Results (`r17a`, 13 adults; macros `\numCTwoE…`).** H1 (closers lead
closing, task-pooled) −26 ms [−52, 167], n = 12 (1 flagged), IUT p 0.615 (locking 0.575, location
0.615): not rejected, so H2 / H3 are "not tested" and the decision is **"not shown"**. H2 closer
release before opening (open_close) 118 ms [48, 142], n = 12; H3 (chewing) 282 ms [68, 360], n = 7.
Post hoc Holm over H1–H3 (labelled sensitivity, `p_holm_family`): 0.615 / 0.004 / 0.031.
Supporting (Holm within the block): digastric −38 ms [−60, −22], p 0.028 (recruited after the
onset, the predicted sign); α onset −8 ms [−59, 82]; changepoint 118 ms [−45, 344] (p 0.39);
optical −37 ms; symmetric window 5 ms; P5 lag open_close 10 ms [−20, 20], chewing 15 ms [5, 95].
Structure (descriptive): log(IQR / S2 IQR) −0.41 [−0.69, −0.31]; release slope b 0.94 / 1.02.

### 3.4 Final look at the onset analysis (J3, 2026-11-02)

What was checked one last time before the re-run, and what it found. The C2 verdict
stays "not confirmed" (pre-specified rule dependence); nothing below is a rescue.

- **Independent recompute.** An independent recomputation re-derives the four C2 leads (first onset and driving burst, openers and
  closers), the Holm-adjusted p values and H1–H3 from the stored onsets: every row PASS.
- **Code re-read** (`timing/lead.py: select_onset`, `WindowOnsets.__call__`,
  `stats/cohort_misc.py: per_participant_timing`, `timing/c2e.py: _release`): no defect.
  - The burst gate (D15) looks at the whole analysis window, not only the selected burst. This
    is the gate's definition (a channel-cycle without a burst above the rest level anywhere in
    the window is unusable), now stated so here and in the caption of the usability flow.
  - The finite-share gate (more than 90 % finite samples in the window, `timing/lead.py`) was
    a literal; since 2026-11-03 it is the parameter `stages.timing.min_finite_share`
    (0.9, the same value). <!-- num-ok: gate definition in the code -->
- **Wording.** The specification curve's reference specification uses the driving-burst rule;
  the pre-specified C2 rule is the first onset. The caption no longer calls the driving-burst
  rule "primary".
- **Threshold sensitivity** (post hoc, change-log entry "C2 onset sensitivity"): variant
  `onset_ksd_5` (onset threshold rest mean + 5 SD instead of 3 SD, `stages.timing.k_sd`), on the
  13-adult base. Reported as how the leads move, in the robustness table. <!-- num-ok: detector threshold definitions (3 SD primary, 5 SD variant; study.yaml) -->
- **Mechanical onset from ω alone** (t_mech = t_ω, because the acceleration term fires early,
  E34): no extra run; the specification curve stores it (`stats_timing:timing_spec`, mechanical
  source ω) and the text quotes it from there.
- **Latency chain** (§3, E34): stated as a bound on the lead, not subtracted. <!-- num-ok: see §3 for the chain -->
- **Onset QC figure** (`emg_signal_spectrum`, form `onset_qc`; decision of the authors: figure only, no manual onset labels): a
  stratified random sample of cycles per participant with the raw, filtered and TKEO traces, the
  rest threshold, both onset rules, t_ω and t_mech. The open acceptance item of s04 (TKEO
  within a tolerance of a visual onset) stays open; the figure lets a reader judge the onsets.
- **Montage** (X10): the authors' label check (2026-11-03) found the code's
  digastric sides swapped; corrected in the code and readers (an03 §1, E37). The opener onsets
  pool both digastrics and are unaffected.

## 4. Intent labels (`biomech/labels/auto_labels.py`, stage `labels`)

**What and why.** Ground truth for decoding: the direction of jaw motion at every optical frame
(40 Hz), from the incisal point $\mathbf p(t)$ in the maxillary frame (x anterior, y left, z up;
an02 §3). No manual labels exist; validity is checked in §4.1.

**Velocity** (`_velocity`), per phase and axis:
1. untracked samples are filled by linear interpolation over the sample index;
2. a 2nd-order Butterworth low-pass at `vel_cutoff_hz` = 4 Hz, zero-phase (`sosfiltfilt`), only
   for more than 15 samples: $\tilde{\mathbf p} = H_4 \ast\ast\ \mathbf p$;
3. $\mathbf v = \mathrm d\tilde{\mathbf p}/\mathrm dt$ by `np.gradient` over the frame times,
   set back to NaN where the point was untracked.

**Axis.** The task context sets the axis and sign (`AXES`): open_close / chewing $u = -v_z$
(OPEN for $u > 0$, CLOSE for $u < 0$), protrusion_retrusion $u = +v_x$ (PROTRUSION / RETRUSION),
left_right $u = +v_y$ (LEFT / RIGHT). Phases outside `AXES` (rest, MVC, …) are all REST.

**Motion** (`_hysteresis`). With $r = P_{95}(\lvert u\rvert)$ over the phase:

$$
m[i] = \begin{cases}
0 & u[i] = \mathrm{NaN},\\
0 & m[i-1] = 1,\ \lvert u[i]\rvert < \texttt{lo\_frac}\cdot r,\\
1 & m[i-1] = 0,\ \lvert u[i]\rvert > \texttt{hi\_frac}\cdot r,\\
m[i-1] & \text{otherwise},
\end{cases}
\qquad \text{class} = \text{pos if } u > 0,\ \text{neg if } u < 0\ \text{where } m = 1,
$$

`hi_frac` = 0.20, `lo_frac` = 0.10, $m[-1] = 0$. In `free_motion` all three axes compete
(D16, `axis_scales`): with $r_a = P_{95}(\lvert u_a\rvert)$,

$$
c_a = \max\big(r_a,\ \texttt{axis\_scale\_floor}\cdot \max_b r_b\big),\qquad
z_a = \lvert u_a\rvert / c_a,\qquad a^\ast = \arg\max_a z_a,
$$

and the hysteresis runs on $\max_a z_a$ with thresholds 0.20 / 0.10; the class is the sign of
$u_{a^\ast}$. `axis_scale_floor` = 0.25; 0 = the previous rule (own P95 per axis, sensitivity
variant `labels_p95`), under which a hardly used axis was amplified into spurious
LEFT / RIGHT / PROTRUSION labels. The labels stage stores `label_agreement` (share of samples
changed and Cohen's κ vs the previous rule, per phase).

**Segments** (`_merge_short`). Segments shorter than `min_seg_s` (0.1 s, round(0.1 $f_{\mathrm
{opt}}$) frames) take the label of their left neighbour (the right one for the first segment),
one at a time, until none is left. Untracked samples are then set to −1.

**Parameters.** `stages.labels`: `vel_cutoff_hz` 4, `hi_frac` 0.20, `lo_frac` 0.10, `min_seg_s`
0.10, `axis_scale_floor` 0.25.

**Code.** `labels/auto_labels.py:label_phase`, `axis_scales`, `compare_axis_scale`,
`label_agreement`; `stages.py:labels_stage`.

**Paper wording.** Intent labels were derived from the optical incisal-point trajectory in the
maxillary frame, low-passed at 4 Hz (zero-phase) and differentiated. Movement along the task
axis (vertical for opening and closing, anteroposterior for protrusion, mediolateral for
laterotrusion) was entered above 20 % and left below 10 % of the phase's 95th-percentile speed;
the sign gave the class, and segments shorter than 0.1 s were merged into their neighbour. In
free motion the axis with the largest speed, normalised by its 95th percentile (floored at a
quarter of the largest axis' value), set the class.

### 4.1 Label validity: kappa against a second rule (stage `labels_check`, `biomech/labels/kappa.py`)

**What and why.** No manual labels exist (the manual Dash session, `ml label`, was not done), so
the stored labels are compared with a **second, independent rule** that shares no code with
`auto_labels` and uses a different signal and derivative. Since
2026-11-01 this is the cohort pipeline stage `labels_check` (depends on `labels`, `kinematics`;
table `labels_check.kappa`); before that it was an audit file
(the audit script
`scripts/audit_recompute/labels_kappa.py` is now a wrapper of the same code for older
snapshots). Macros `labelKappa…` in `experiments/icorr2027/numbers_dataset.yaml`.

**Second rule** (`second_rule`), per phase, on the frames of the stored labels:
- derivative: Savitzky–Golay, order 2, first derivative, window $w$ = 0.25 s rounded up to an
  odd frame count, $w = \max\big(5,\ \mathrm{round}(0.25 f_{\mathrm{opt}}) \mathbin{\mathrm{OR}} 1\big)$
  (bitwise OR with 1; 11 frames = 0.275 s at 40 Hz; `np.gradient` if the phase is not longer than
  $w$), on linearly gap-filled signals (`_fill`);
- open_close / chewing: the optical **opening angle** $\theta$ (`theta_raw_deg`, without the pose
  low-pass; a rotation, not the incisor translation): OPEN if $\dot\theta > +\eta_\theta$, CLOSE if
  $\dot\theta < -\eta_\theta$, else REST (no hysteresis);
- protrusion_retrusion / left_right: the incisor velocity along the task axis, PROTRUSION /
  RETRUSION if $\pm v_x > \eta_x$, LEFT / RIGHT if $\pm v_y > \eta_y$;
- free_motion and rest: moving if $\lVert\mathbf v\rVert > \eta_{\mathrm{speed}}$, the class from
  the component with the largest $\lvert v_a\rvert$ and its sign ($+z$ = CLOSE, $-z$ = OPEN; no
  per-axis normalisation);
- runs shorter than round(0.10 $f_{\mathrm{opt}}$) frames take the label of the preceding run
  (`_merge_runs`); untracked frames −1.

**Thresholds.** `fixed` (set before the first comparison): $\eta_\theta$ = 10 deg/s,
$\eta_x = \eta_y$ = 5 mm/s, $\eta_{\mathrm{speed}}$ = 8 mm/s (`FIXED`). `rest_floor` (declared
after that run showed the fixed 3-D speed threshold inside the rest-phase tracking noise of some
participants): each threshold raised to the participant's rest-phase noise floor, never lowered,
$\eta^{\mathrm{rf}} = \max\big(\eta^{\mathrm{fixed}},\ P_{99}(\lvert\text{rule signal}\rvert
\text{ in the rest phase})\big)$ (`rest_floor`).

**Cohen's kappa** (`kappa`; formula in an07 §16.3). Over the frames labelled by both rules
($\ge 0$), from the normalised 7 × 7 confusion matrix (stored labels × second rule), with
$\kappa = 1$ when $p_e = 1$ (both labellings constant and equal). The stage labels the whole
rest phase REST, so its $\kappa$ is 0 whenever the second rule marks any rest frame as moving
(1 only if both label every rest frame REST).

**Sampling and levels** (`run_variant`). Per participant and phase, 10 random 2-s windows
(`windows`, `window_s`; starts uniform integers in $[0, N - n_w)$, drawn with replacement, union
of the frames; a phase not longer than 2 s is used whole), RNG `default_rng(20261029)` (the
stage `DEFAULTS` seed, not the study seed; the same windows for both variants). Rows
(`variant` × `scope` × `level`):
- `participant_phase` (sampled frames; also `agreement` = $p_o$ and `kappa_all_frames`);
- `participant` (phases pooled) and `overall_sampled` (participants pooled, with
  `kappa_median_participants`, `kappa_min_participants`, `kappa_max_participants`,
  `n_participants`), each for scope `all_phases` and `without_rest`;
- `overall_all_frames` (no sampling);
- `phase` (participants pooled, sampled);
- `boundary_tolerant`: all frames except the rest phase and except frames within ±2 frames
  (±50 ms, `EDGE_FRAMES`) of a label change in either labelling (`near_transition`: for a change
  between frames $i$ and $i+1$, frames $i-2 \dots i+3$), pooled and per phase: onset timing
  separated from class identity.

Planned target: $\kappa \ge 0.8$.

**Values (r17a, the 13 adults; paper store).** Fixed thresholds: overall 0.734,
per-participant median 0.720 (range 0.618–0.847), without the rest phase 0.757,
boundary-tolerant 0.926; rest-floor variant 0.671; per phase open-close 0.835, laterotrusion
0.784, protrusion 0.760, free motion 0.652, chewing 0.553. With the participant recorded outside the protocol (`r17`,
n = 14): 0.736, 0.711 (0.645–0.848), 0.754, 0.924, 0.678. Reading: the
sampled overall κ is below the planned 0.8, the boundary-tolerant κ above it, so the
disagreement lies mainly in the timing of transitions (±50 ms), not in the class.

**Code.** `labels/kappa.py:second_rule`, `rest_floor`, `kappa`, `near_transition`,
`run_variant`, `run_labels_check`; `stages.py:labels_check_stage`.

**Paper wording.** Because no manual labels were available, the automatic labels were compared
with a second, independent rule: the opening-angle rate (Savitzky–Golay derivative) beyond
±10 deg/s for opening and closing, the incisor velocity along the task axis beyond ±5 mm/s for
protrusion and laterotrusion, and the incisor speed beyond 8 mm/s with the dominant axis in free
motion. Agreement was quantified with Cohen's κ on ten random 2-s windows per participant and
phase. κ was 0.73 overall (participant median 0.72, range 0.62–0.85) and 0.76 without the
rest phase; over all frames outside the rest phase, excluding frames within ±50 ms of a label
transition raised it to 0.93, so disagreement concerned mainly the timing of transitions rather
than the class. *(Values of the paper store `r17a`, 13 adults; in the text
use the `labelKappa…` macros.)*

## 5. Decoding (`software/ml`, stages `ml_windows`, `ml_eval`, `ml_online`)

One implementation (`ml/core`) serves these stages and the `ml` CLI; the full specification is
in docs/ml (ml03–ml08). Configuration: `stages.ml_windows`, `stages.ml_eval`,
`stages.ml_online` in `etc/biomech.yaml`.

**Windows** (`ml/core/windows.py`, `ml/core/dataset.py:build_windows`). Trailing windows of
$L_w = \mathrm{round}(0.2 f_s)$ EMG samples (200 ms), stride 50 ms, cut per phase (they never
cross a phase edge); a window ending at sample $e$ covers $e - L_w + 1 \dots e$. Label: the label
in force at the window **end** (`label_at: end`), ignored when the latest label sample is older
than 0.1 s. Rejected: windows spanning a time gap ($t_e - t_{e - L_w + 1} > 1.1 \times$ 200 ms),
with a per-channel masked share > `max_masked` = 0.2, unlabelled, or with non-finite features.

**EMG chain.** jtsmm's real-time chain (`CausalEmgFilter`: tracking mains canceller, causal
20–450 Hz band-pass) from a 2-s warm-up before each phase (`warmup_s`, discarded), in % of the
channel reference $x \leftarrow 100\,x/\mathrm{ref}_c$ (`signals.emg.reference`, computed on the
same causal chain, E14). Masked samples are zeroed inside the window for the feature computation.

**EMG features per channel** (`ml/core/features.py`; $W$ the window, $d_i = W_{i+1} - W_i$,
$\epsilon = 10^{-6}$, $\vartheta_c$ = `thr_sigma` × SD of the rest-phase sample differences of
channel $c$, `thr_sigma` = 1, E14):

$$
\begin{aligned}
\mathrm{MAV} &= \tfrac{1}{L_w}\textstyle\sum_i \lvert W_i\rvert, &
\mathrm{RMS} &= \big(\tfrac{1}{L_w}\textstyle\sum_i W_i^2\big)^{1/2}, &
\mathrm{WL} &= \tfrac{1}{L_w - 1}\textstyle\sum_i \lvert d_i\rvert, \quad \text{features } \log(\cdot + \epsilon),\\
\mathrm{ZC} &= \mathrm{mean}_i\ \mathbb 1[W_i W_{i+1} < 0 \wedge \lvert d_i\rvert \ge \vartheta_c], &
\mathrm{SSC} &= \mathrm{mean}_i\ \mathbb 1[d_i d_{i+1} < 0 \wedge (\lvert d_i\rvert \ge \vartheta_c \vee \lvert d_{i+1}\rvert \ge \vartheta_c)], &
\mathrm{WAMP} &= \mathrm{mean}_i\ \mathbb 1[\lvert d_i\rvert \ge \vartheta_c],
\end{aligned}
$$

with the Hann-windowed periodogram $P(f) = \lvert\mathrm{FFT}(W \cdot \mathrm{hann})(f)\rvert^2$ and
$P_{\mathrm{tot}} = \sum_{20 \le f \le 450} P(f) + \epsilon$:

$$
\mathrm{band}_{[l, h)} = \frac{\sum_{l \le f < h} P(f)}{P_{\mathrm{tot}}}\ \ \text{for } [20,50), [50,100), [100,200), [200,450),\qquad
\mathrm{MNF} = \frac{\sum_{20 \le f \le 450} f\,P(f)}{100\,P_{\mathrm{tot}}}\ \ (\text{Hz}/100).
$$

**Rest-relative normalisation** (D13, `ml_windows.window.emg_norm: rest_relative`;
`rest_relative_stats`, `apply_rest_relative`). From the participant's valid windows of the
unlabelled `rest` phase (labels not used): log-amplitude features $g \leftarrow g -
\mathrm{median}_{\mathrm{rest}}(g)$; count and spectral features $g \leftarrow (g -
\overline{g}_{\mathrm{rest}}) / \max(\mathrm{SD}_{\mathrm{rest}}(g), 10^{-3})$. `mvc` = no rest
normalisation (comparison).

**IMU features** (`ml/core/signals.py:imu_signals`, `features.imu_window_features`): on a
200 Hz grid, for the mandible (MTA) and cranium (CRA) IMUs the body-frame angular velocity from
the backward quaternion difference, $\boldsymbol\omega_k = \mathrm{rotvec}(q_{k-1}^{-1} q_k)/\Delta
t$ (deg/s, causal), and the gravity-free acceleration $\mathbf a - R^\top (0, 0, g)$ (m/s²): 12
signals. Per window (last 40 grid samples): mean and SD (ddof 0) of each, plus the mean mandible
angular speed $\overline{\lVert\boldsymbol\omega_{\mathrm{MTA}}\rVert}$: 25 features.
Calibration-free (no hand-eye calibration).

**Scaling** (`ml/fusion/early.py:BlockScaler`, inside every pipeline, fitted on the training
fold): $z_j = (x_j - \bar x_j)/s_j \cdot w_{b(j)}$ with block weight $w_b = \sqrt{d/(B\,d_b)}$
($d$ features, $B$ blocks, $d_b$ features of block $b$; $w = 1$ for one block), so every block
contributes the same total variance $d/B$.

**Models** (`ml/models/registry.py`). Primary model SVM (`stages.ml_eval.primary_model: svm`):

$$
\hat y(\mathbf z) = \text{one-vs-one vote of } \operatorname{sign}\Big(\sum_i \alpha_i y_i K(\mathbf z_i, \mathbf z) + b\Big),\qquad
K(\mathbf z, \mathbf z') = \exp\big(-\gamma\lVert\mathbf z - \mathbf z'\rVert^2\big),
$$

soft margin $C \in \{0.01, 0.1, 1, 10\}$, $\gamma = m/(d\,\mathrm{Var}(\mathbf Z))$ with $m \in
\{0.25, 1, 4\}$ (`gamma_mult`; $m = 1$ is sklearn's "scale", Var over all scaled training
entries), `class_weight="balanced"`: per-class weight $n/(K n_k)$ multiplies $C$ (E14). LDA:
`lsqr` solver, Ledoit–Wolf shrinkage (`"auto"`), uniform priors (E14). MLP (64, 32) with early
stopping on grouped validation macro-F1; GRU and late fusion (`late_mean|late_weighted|
late_stack:<model>`) are available but not in the paper configuration.

**Training and protocols** (`ml/protocols/evaluate.py`, `folds.py`):
- **LOSO (`loso_subject`):** fold $u$ trains on the structured tasks of all other subjects and
  tests on subject $u$'s structured tasks and, separately, on their `free_motion`. Training rows:
  a random subsample of `max_train` = 12 000 windows (`subsample`, `default_rng(seed)`).
  Hyper-parameters: inner `GroupKFold` with $\min(3, \#\text{subjects})$ folds over the training
  subjects, mean inner macro-F1, the first grid point (grid order) with the highest mean
  (`grouped_search`); refit on all training rows. SVM probability calibration and MLP early
  stopping also split by subjects.
- **Blocked within-subject:** per take and phase the windows are ranked by time and assigned
  block $\lfloor \mathrm{rank}\cdot k / m\rfloor$, $k$ = 5; block $b$ of every phase is tested
  together, training windows within 0.35 s of a test window of the same take are dropped (75 %
  overlap; D4). Each subject's folds use the hyper-parameters that the nested LOSO search chose
  for the fold that held that subject out (D32; like-for-like with LOSO, chosen without the
  subject's data); the default-hyper-parameter run is stored as the sensitivity
  `ml_eval.blocked_default` (summary protocol `blocked_default`).

**Metrics** (`fold_scores`). LOSO: one row per held-out subject, on that subject's structured
test windows in time order. Blocked: predictions of all the subject's folds pooled; only balanced
accuracy and macro-F1.

$$
\mathrm{BA} = \frac{1}{\lvert\mathcal C_y\rvert}\sum_{c \in \mathcal C_y} \frac{\mathrm{TP}_c}{\mathrm{TP}_c + \mathrm{FN}_c},\qquad
\text{macro-F1} = \frac{1}{\lvert\mathcal C_y\rvert}\sum_{c \in \mathcal C_y} \frac{2\,\mathrm{TP}_c}{2\,\mathrm{TP}_c + \mathrm{FP}_c + \mathrm{FN}_c},
$$

$\mathcal C_y$ = the classes present in $y_{\mathrm{true}}$ (sklearn `balanced_accuracy_score`;
F1 with `labels = unique(y_true)`, `zero_division=0`), so a fold without some class is not
penalised for it. The same macro-F1 drives the inner search and early stopping.
- Confusion matrix: fixed 7 × 7 (`CLASSES`), summed over the subjects per feature set × model.
- Latency (`latency`): onsets are test windows labelled non-REST whose preceding window is REST,
  within the same take, phase and gap-free segment (a step > 2 strides starts a new segment). For
  an onset window with end time $t_{\mathrm{on}}$ the search runs over the windows of that segment
  from $t_{\mathrm{on}} - 0.5$ s to the end of the labelled run; the hit is the first run of 2
  consecutive windows predicted as that class; latency $= t_{\mathrm{end}}(\text{first window of
  the run}) - t_{\mathrm{on}}$ (ms; negative = early; NaN = not detected).
  `latency_ms_median` = median over detected onsets, `detected_frac` = detected share.
- False activations over the test windows of the labelled `rest` phase: `fa_frac` = share
  predicted non-REST; `fa_per_min` = (REST → movement transitions in the prediction, a movement
  in the first window counting as one) / ((last − first $t_{\mathrm{end}}$)/60 s).
- `free_balanced_accuracy`, `free_macro_f1`: the same scorers on the held-out subject's
  `free_motion` windows (LOSO only).

The cohort table (`ml_eval.summary`) gives `describe(stat="median")` (median, BCa CI, an07) over
the subjects per feature set × model × protocol: for LOSO the metrics `balanced_accuracy`,
`macro_f1`, `free_balanced_accuracy`, `latency_ms_median`, `detected_frac`, `fa_frac`,
`fa_per_min`; for `blocked` / `blocked_default` only `balanced_accuracy`
(`ml/biomech_stages.py:317-329`).

**Permutation test** (`permutation`, `shift_labels`). Each subject's label sequence (all its
windows in table order) is circularly shifted by an integer drawn uniformly from $[\lfloor N_u/4
\rfloor, \lfloor 3N_u/4\rfloor)$, which keeps the label autocorrelation and destroys the
feature–label pairing; 100 permutations (`n_perm`), each a full nested LOSO of LDA on EMG+IMU with
the inner folds and `max_train` of the observed run (E20). Statistic: the cohort mean LOSO
balanced accuracy $\overline{\mathrm{BA}}$;

$$
p = \frac{1 + \#\{b : \overline{\mathrm{BA}}{}^{\ast}_b \ge \overline{\mathrm{BA}}_{\mathrm{obs}}\}}{1 + 100}.
$$

Skipped when the observed score is missing (`ml_eval.tests`, row `permutation|lda|emg_imu`).

**Ablations** (feature sets `emg`, `imu`, `emg_imu`, `emg_nodig` (EMG without the digastrics),
`emg_nodig_imu`). For `primary_model` they are compared on the per-subject LOSO balanced accuracy,
complete cases, with ≥ 3 feature sets and ≥ 5 subjects: Friedman test, then a two-sided Wilcoxon
signed-rank test of EMG+IMU against each other set, Holm-corrected (`stat` = median paired
difference) (`ml/biomech_stages.py:ml_eval_stage`).

**Few-shot per-user calibration** (`ml/protocols/variants.py`, `stages.ml_eval.variants.few_shot`;
exploratory). Per LOSO fold and feature set (`imu`, `emg_imu`), mode `per_phase` (a guided
calibration routine): the calibration set is the first $k \in \{0, 20, 50, 100\}$ windows of every
phase of the held-out user (1, 2.5, 5 s per movement at 50 ms stride); the evaluation set, common
to all $k$, the windows of each phase ending more than 0.35 s (`guard_s`) after its $\max(k)$-th
window. The SVM (fold hyper-parameters of the main run) is refit on the fold's training subsample
plus the $n_{\mathrm{cal}} = k \times \#\text{phases}$ calibration windows, each weighted

$$
w_{\mathrm{user}} = \frac{n_{\mathrm{src}}}{K_{\mathrm{src}}\, n_{\mathrm{cal}}}\quad(\texttt{weight: balanced}),
$$

so all calibration windows together weigh as one average training class ($n_{\mathrm{src}}$
training windows, $K_{\mathrm{src}}$ classes; training windows weigh 1; `class_weight="balanced"`
multiplies both). $k = 0$ is the plain LOSO model. Tests (`few_shot_tests`, rows
`fewshot|svm|emg_imu_vs_imu|k=<k>`): per $k$ the paired per-participant difference
$\mathrm{BA}_{\mathrm{emg\_imu}} - \mathrm{BA}_{\mathrm{imu}}$, median, two-sided Wilcoxon (n ≥ 5),
Holm over the four $k$; `variants_summary` adds the paired gains $\mathrm{BA}(k) - \mathrm{BA}(0)$.

**Pseudo-online replay** (stage `ml_online`, X4, exploratory; `ml/online/`,
docs/ml/ml08). Per held-out participant the stored take is replayed in chunks of `chunk_s` =
20 ms of EMG samples: causal chain running continuously from the stream start
(`filter_mode: continuous`; mains frequency from the 60 s before the stream start, `line_hz:
preroll`), causal masks, the same window features, the fold SVM of `ml_eval` (other
participants' windows, `best_params`, `max_train`, `inner_folds`), and the X3 release detector on
the stream. A window is emitted at the chunk in which its last sample arrives (waiting at most
`imu_max_wait_s` = 0.1 s for its IMU samples); $t_{\mathrm{dec}}$ = the stream time at emission
(`t_dec_us`; for EMG-only feature sets the time the EMG part was complete, `t_dec_emg_us`).
- Calibration modes (`calibration.modes`): `none`; `first` (the first 100 decodable labelled
  windows after the rest block); `per_phase` (progressive: after each phase's 20th window a refit
  with the calibration windows of that and all earlier phases); weights as $w_{\mathrm{user}}$;
  scored from 0.35 s after the last calibration window.
- Smoothing (`decoder.smooth`, causal, reset to REST at every phase start): majority over the last
  $n$ = 3 decodable window decisions with counts $c_k$; a movement class $k$ is entered when $c_k
  \ge 2$ (`enter`) and $c_k$ exceeds the count of the current class, held while it keeps ≥ 1
  (`hold`), REST otherwise; undecodable windows hold the state.
- Stream vs offline accuracy: `ba` (smoothed) and `ba_raw` over the scored windows, against
  `balanced_accuracy_offline` of `ml_eval`; `ba_free`.
- Decision latency (`decision_latency`): per REST → class onset of the optical label track
  ($t_{\mathrm{on}}$, class held to $t_{\mathrm{off}}$, REST since $t_{\mathrm{rest}}$) the first
  window ending in $[\max(t_{\mathrm{on}} - 0.5\,\mathrm s, t_{\mathrm{rest}}),\ t_{\mathrm{off}}]$
  at which the smoothed decision *enters* that class; latency $= t_{\mathrm{dec}} -
  t_{\mathrm{on}}$; `latency_ms_median`, quartiles, `frac_le_150ms` (budget `latency.budget_ms`
  = 150), `frac_anticipatory`; the offline window-end latency alongside (`latency_win_ms_median`).
- False activations: in the `rest` phase onsets per minute (`fa_per_min_rest`, as offline); in
  `free_motion` smoothed onsets at REST-labelled windows with no labelled onset within
  `fa_guard_s` = 0.5 s after them, per minute of REST-labelled free-motion windows (count ×
  stride) (`fa_per_min_free`).
- Real-time factor (`compute_stats`): $\mathrm{RTF} = \sum_c \tau_c / (N_c \cdot 20\,\mathrm{ms})$
  over the per-chunk processing times $\tau_c$ (the one-off calibration chunk excluded and
  reported as `calibration_ms`), with median / P99 / max chunk time; `emit_delay_ms` =
  $t_{\mathrm{dec}} - t_{\mathrm{end}}$.
- `feature_deviation`: stream features against the stored offline `ml_windows` features.
- Stream release detector (`release_metrics`, columns `x3_*` and table `ml_online.release`):
  the offline X3 definitions on the stream decisions, including the chance level:
  `biomech.timing.release.chance_detection` with `shift=True` in the rhythmic tasks of
  `lead.NULL_SHIFT_TASKS` (open_close, chewing: the circular shift of the decision train, §3.2)
  and pseudo-onsets otherwise, seeded with the participant seed as offline
  (`ml/online/metrics.py:release_metrics`). Erratum E42: the replay used the
  pseudo-onset branch in every task, which continuous cycling leaves with few or no eligible
  times, so `x3_chance_detected_frac` was not comparable with `chance_detected_frac_offline`;
  it now is.

**Code.** `ml/core/dataset.py:build_windows`, `ml/core/features.py`, `ml/core/signals.py`,
`ml/models/registry.py:SvmClassifier`, `ml/fusion/early.py:BlockScaler`,
`ml/protocols/evaluate.py:fit`, `grouped_search`, `fold_scores`, `latency`,
`false_activations`, `permutation`, `blocked_many`; `ml/protocols/folds.py:make_folds`;
`ml/protocols/variants.py:few_shot_plan`, `user_weight`, `few_shot_tests`;
`ml/biomech_stages.py:ml_eval_stage`; `ml/online/stage.py`, `decoder.py`, `metrics.py`.

**Paper wording.** Decoding used trailing 200-ms windows with a 50-ms stride, labelled by the
intent label at the window end. Per EMG channel, log mean absolute value, log RMS and log
waveform length, zero-crossing, slope-sign-change and Willison-amplitude rates (threshold: one
standard deviation of the rest-phase sample differences), four relative band powers (20–450 Hz)
and the mean frequency were computed on the causally filtered EMG and expressed relative to the
participant's rest phase; IMU features were the window mean and standard deviation of the
angular velocity and gravity-free acceleration of both IMUs and the mean jaw angular speed. An
RBF-kernel support vector machine with balanced class weights was evaluated leave-one-subject-out;
C and the kernel width were selected by an inner three-fold cross-validation grouped by subject
(macro-F1). Performance was the balanced accuracy per held-out participant (cohort median with
BCa confidence interval). Above-chance decoding was tested for linear discriminant analysis on
the EMG and IMU features: the cohort mean leave-one-subject-out balanced accuracy was compared
with a null distribution from 100 circular shifts of each participant's label sequence, each
refitted with the same nested cross-validation. A pseudo-online replay fed the recordings in 20-ms chunks through
the causal processing chain and the same classifiers and measured the decision latency from the
labelled movement onset, false activations at rest and the real-time factor.
