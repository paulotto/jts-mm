<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ICORR 2027 — analysis plan

> **Note of the public release.** This is the analysis plan as it was frozen, before any outcome was computed. Its text is unchanged, except for passages marked as shortened for participant privacy and for the neutral names that this release uses throughout. Every later change (amendments, errata, deviations, and added analyses) is listed in [change_log.md](change_log.md), with its reason and with whether results had been seen. The study identifier `icorr2027` is historical and names the dataset study of both papers.

**Version 1.0 — frozen 2026-10-15** (git tag `icorr-plan-frozen`).

## 0. Why this plan is frozen, and what that means

**What could go wrong without it.** Twelve participants and many defensible analysis choices (frames,
thresholds, filters, normalizations, onset detectors, exclusions, tests) mean that choosing among them
*after* seeing results would inflate false positives and effect sizes. That is the "garden of forking
paths".

**What freezing does.** It fixes the primary outcomes, the data-handling rules, the signal processing
and the statistics **before any outcome metric is computed**. The paper can then call these analyses
pre-specified, and confirmatory results stay separate from exploratory ones.

**What it does not do.** Freezing does not forbid change. Every change after the freeze is logged
under **Amendments** (before the first outcome stage runs) or **Deviations** (after it), each with a
date, a reason, and whether results had been seen.

**Honest scope.** This is a *pre-analysis* plan, not a pre-data registration.
- The data were recorded before this plan existed.
- Signal quality was inspected (`qc` stage and the checks listed in §7).
- No outcome metric (§1) has been computed on this cohort.

An external timestamp (OSF) can be added; the git tag is the internal one.

## 1. Questions and outcomes

The paper derives requirements for jaw exoskeletons from synchronized kinematics, sEMG and IMU data
recorded in asymptomatic adults.

### Primary (confirmatory)

| Claim | Primary outcome | Unit of analysis |
|---|---|---|
| **C1** Jaw motion is not a hinge | Residual of the best-fit fixed hinge axis at the incisal point (mm, RMS over opening cycles), compared with its propagated noise floor | Participant (median over cycles) |
| **C2** Muscle activity precedes motion | EMG onset → mechanical onset lead (ms). Opening: digastric; closing: masseter/temporalis | Participant (median over cycles) |
| **C3a** Wearable sensing recovers opening | IMU-estimated vs. optical incisal opening (mm): bias and 95 % limits of agreement, leave-one-subject-out coupling model | Participant |
| **C3b** Wearable sensing decodes intent | LOSO balanced accuracy; detection latency vs. mechanical onset; false activations during rest | Participant (fold) |

### Secondary (descriptive, pre-specified)

- **S1 Reference values:**
  - clinical kinematics: DC/TMD-conformant MIO, protrusion, laterotrusion, deviation, condylar translation, velocities, chewing frequency;
  - activation maps (%MVC / %RVC per muscle × task), rest activity, asymmetry and activity indices, co-activation.

  All with SEM/MDC, raw and size-normalized, with sex-specific literature z-scores.
- **S2 Variability and data-collection requirements** (§5.3). How much of the observed variance is
  within-participant (cycle to cycle), between-participant, and induced by protocol or processing
  choices, and how many cycles a stable estimate needs. This supports the paper's case for
  standardized multimodal data collection.

### Exploratory

Anything not listed above, e.g. sex/age effects, s02 between-day comparison, IMU-only translation.
Labelled as exploratory wherever reported.

## 2. Cohort

- **Who:** asymptomatic adult volunteers recruited under
  `docs/operations/op04_emg_study_safety_sop.md`. Only sex and age are known. The source of truth is
  `cohort.yaml`.
- **Excluded:** *(Shortened in the public version: participant privacy.)*
- **Repeated takes:** the first complete motion and MVC takes are primary. *(Shortened in the
  public version: recording dates are not published.)*
- **Additions:** s14 is added under the same operator config. The analysis is re-run unchanged.
- **Reporting:** demographics in aggregate only. Figures may colour by sex, never label individuals
  with age or sex.

## 3. Data handling: flag, mask, never delete

**Principles.**
1. **Raw data are never altered or dropped.** Quality is expressed as *flags* at sample, cycle and
   channel level.
2. **Each analysis family uses the data its question needs.** A channel unusable for one question
   (e.g. absolute rest level) may be fully usable for another (e.g. onset timing).
3. Every primary outcome is also reported **with all data** as a sensitivity analysis. Counts of
   flagged data are reported per family.
4. Rules use only information that cannot reveal the outcome (rest, MVC, signal properties). They
   never use the outcome itself.

### 3.1 EMG

**Sample-level masks** (set to NaN, never zeroed):
- ADC saturation or rail;
- lead-off bit set;
- **artifact candidates:** envelope above 3× the channel's own MVC/RVC reference, or a raw
  low-frequency (1–20 Hz) power burst above 10× the channel's rest level. The second rule targets
  movement artifacts, e.g. submandibular skin stretch on the digastric during opening.

**Cycle-level flag:** more than 10 % of a channel-cycle masked → that channel-cycle is left out of
analyses of that channel.

**Usability per analysis family:**

| Family | Requirement for a channel(-cycle) | Rationale |
|---|---|---|
| Timing (C2 onsets, lead) | Burst detectable *in that cycle*: burst-to-baseline ≥ 6 dB | Timing needs a detectable onset, not low absolute noise |
| Waveform shape (SPM on normalized profiles, co-activation) | Channel SNR (reference vs. rest) ≥ 6 dB | Normalization removes gain; noise must not dominate the profile |
| Amplitude levels (%MVC/%RVC, indices) | SNR ≥ 6 dB **and** valid reference (MVC/RVC phase present, < 10 % masked) | Level outcomes need a trustworthy denominator |
| Rest activity (S1) | All channels. Channels with cleaned rest RMS > 20 µV are reported **separately** as "noise-floor limited" | Rest RMS is the outcome itself, so it cannot be an exclusion criterion |
| Intent decoding (C3b) | All channels, masks applied; sensitivity without flagged channels | A wearable faces real-world noise |

### 3.2 Optical
- Gaps ≤ 0.5 s are bridged by SE(3) interpolation and flagged.
- For a longer gap, only the metrics that need the missing interval are set missing for that cycle,
  e.g. maximum opening if the gap covers the peak. The rest of the cycle is kept.
- Lock per phase is reported.

### 3.3 IMU
- Timestamps are sorted and de-duplicated (count reported).
- Accuracy-0 samples and gaps > 50 ms are masked.
- Phase coverage is reported, not used to drop phases.

## 4. Definitions (detailed in docs/analysis/an02–an05)

### 4.1 Kinematics
- **Frame:** subject-specific maxillary frame from `max_point_1..3`. The incisal point is
  `mand_point_3`.
- **Closed reference:** the per-cycle minimum opening, not `rest`.
- **MIO:** DC/TMD-conformant, i.e. interincisal distance at maximum opening plus the vertical incisal
  overlap measured in the closed position.
- **Size proxies:** intercanine width, intercondylar width, condyle–incisor distance.

### 4.2 sEMG processing (offline)

1. **Mains removal:** least-squares fit and subtraction of the line fundamental **and its harmonics up
   to 450 Hz** (k = 1…9 at 50 Hz).
   - **Frequency:** the per-take measured grid frequency (`measure_line_hz` on 60 s), falling back to
     50.000 Hz.
   - **Blocks:** 1-s blocks at the real sample times, with intercept and linear-trend nuisance terms.
     Only the sinusoids are subtracted.
   - **Short final remainder:** a remainder shorter than one block is merged into the preceding
     block. The current code fits it on its own, which is ill-conditioned (see §7).

   **Why fitted subtraction rather than notches:** notches remove EMG power around every harmonic and
   ring at transients.

   **Harmonics above 450 Hz are not fitted:**
   - they carry a median 0.2–3 % of the in-band power;
   - fitting every harmonic up to Nyquist changed cleaned RMS by ≤ 0.3 % in all unaffected
     channel-phases;
   - the band-pass attenuates them.
2. **Band-pass:** 4th-order Butterworth 20–450 Hz applied forward and backward. This is zero-phase:
   −6 dB at both corners and an effective 8th-order magnitude roll-off. It removes motion-artefact
   and DC drift below 20 Hz.
3. **Artifact handling:** masking as in §3.1. **No automatic zeroing of "spikes".** The former
   150 µV / ≤ 0.4 s rule lowered MVC references by up to 9 % (5 of 72 channels) and zeroed up to 19 %
   of an open–close phase (§7).
4. **Envelope:** centred 150 ms moving RMS, used for amplitude levels and profiles.
5. **Normalization:**
   - masseter/temporalis: 95th percentile of the envelope in `mvc_closing` → %MVC;
   - digastric: the same statistic in `mvc_opening`, reported as **%RVC**, because the effort is
     submaximal against ~10 N.

   The 95th percentile is robust to isolated artifacts. Sensitivity: maximum 1-s moving average.
6. **Onsets:**
   - primary detector: TKEO on the band-passed signal, then a double threshold (baseline mean +
     k·SD, minimum duration);
   - sensitivity detectors: Hodges–Bui and envelope 3·SD.
7. **Causal variant** (real-time budget, ML): same corners as a single-pass filter, plus the causal
   harmonic canceller (`jtsmm.viewer.processing`). Its group delay is reported. The ML branch's IIR
   notch (50 Hz + 3 harmonics) is replaced by this shared implementation.

### 4.3 IMU
- **Inputs:** game-rotation-vector quaternions (≈ 390 Hz) and raw accelerometer **including
  gravity** (≈ 255 Hz). There is no gyroscope stream.
- **Angular velocity:** ω from the Savitzky–Golay derivative of the relative quaternion.
- **Linear acceleration:** gravity removed per IMU with its own quaternion, then mandible relative to
  head.
- **Mechanical onset:** the earliest of the ω-based and acceleration-based onsets. Optical onset is a
  cross-check only.
- **Real-time budget:** lead minus the delay of the causal detector.

## 5. Statistics (an07)

### 5.1 General
- **Descriptives:** participant-level metrics as mean ± SD and median [IQR], with 95 % BCa bootstrap
  CIs.
- **Cycle-level models:** linear mixed models with a subject random intercept.
- **Multiplicity:** Holm correction within each claim family. Effect sizes are always reported.
- **Sex and age:** exploratory only. Sex effect sizes with CIs, an age-adjusted sensitivity model,
  leave-one-out influence checks.

### 5.2 Confirmatory SPM (on subject means, α = 0.05, two tests, Holm)
- L vs. R masseter %MVC over the time-normalized chewing cycle (paired).
- Digastric vs. elevator normalized activation over the open–close cycle (paired).

### 5.3 Variability and data-collection requirements (S2, descriptive)
1. **Inter-participant SPM:** one-way SPM ANOVA across participants, with cycles as repetitions, for
   opening-angle and EMG profiles. Interpreted as "between-participant differences exceed cycle-to-cycle
   variability". Inference about the population would require subject means, so this is descriptive.
2. **Single-participant SPM:** nonparametric (permutation) SPM per participant, with cycles as
   observations:
   - L vs. R masseter in chewing;
   - left vs. right laterotrusion paths.

   Report how many participants show a supra-threshold cluster and where in the cycle.
   - **Caveat:** cycles of one session are not independent samples of a population. This is
     within-session evidence for that individual only.
   - **Message:** group means hide individual patterns, so exoskeleton calibration must be individual.
3. **Variance decomposition of waveforms:** pointwise between- vs. within-participant variance ratio
   and the coefficient of multiple correlation (CMC) within vs. between participants.
4. **Cycle-count stability:** bootstrap estimates of each S1/C1/C2 metric from n = 1…N cycles. The
   number of cycles needed for the CI half-width to fall below the MDC goes into the protocol
   recommendation.
5. **Specification curve:** primary and S1 outcomes recomputed under the plausible alternatives:
   - template vs. subject frame;
   - rest vs. cycle-minimum closed reference;
   - p95 vs. max 1-s normalization;
   - 100/150/250 ms envelope;
   - the three onset detectors;
   - with vs. without flagged data.

   Report the spread induced by these choices next to the between-participant SD.
6. **Protocol heterogeneity actually observed in this data set**, tabulated. Examples:
   - self-paced phases of 10–31 s with varying cycle counts;
   - MVC contractions without the specified rests;
   - landmark holds digitized once;
   - metadata missing (instructions, cabling, dental and TMD status).

   This becomes a minimal data-collection checklist.

### 5.4 Reliability
- Within-session: cycle CV, ICC(3,1), SEM, MDC.
- Between-day: s02 only, descriptive.

### 5.5 Intent decoding
- LOSO is primary; blocked within-subject CV is secondary.
- Hyper-parameters tuned in nested CV.
- Permutation test against chance.
- Per-subject accuracies compared by Friedman/Wilcoxon (Holm).

## 6. Validation

- **Primary:** agreement with Qualisys/Delsys on the reference subset (cv05 B5). Go/no-go on
  **10 Nov 2026**.
- **Always run (backup if no-go):**
  - Monte-Carlo propagation of bench noise and landmark error through C1;
  - the in-vivo rest noise floor;
  - step-ladder trueness;
  - IMU/optical cross-modal agreement;
  - EMG onset robustness across detectors.

  C1 and C2 are claimed only where they exceed the propagated noise floor.

## 7. Inspections made before the freeze (signal quality only, no outcomes)

*(Shortened in the public version: recording dates are not published.)*

- **`qc` stage:**
  - rest noise 2.4–8.7 µV in 10 of 12 participants after cleaning;
  - s03 masseters at 28–33 µV;
  - optical lock ≥ 0.98 in motion phases;
  - IMU coverage 1.0.
- **Harmonics above 450 Hz:** median 0.2–3 % of in-band power. Fitting all harmonics up to Nyquist
  changed cleaned RMS by ≤ 0.3 % (unaffected channel-phases), so they are not fitted.
- **Trailing-block defect in `jtsmm.emg.mains.remove_mains`:** a final block shorter than 1 s is
  fitted on its own and produces a large transient. It inflated whole-window RMS by > 10 % in 10 of
  576 channel-phases (up to 28×, e.g. s09 `mvc_closing` digastric_r). Envelope percentiles moved by
  ≤ 2.4 % in all but one case (11 %). QC exclusions were unchanged. To be fixed in `jtsmm` with a
  regression test before any outcome stage runs.
- **The former spike suppression (150 µV, ≤ 0.4 s)** zeroed samples in 20 task/MVC channel-phases
  and lowered 5 MVC references by 2–9 %. It is replaced by masking (§3.1, §4.2).
- **s11 `open_close` digastric_l:** 11 mV raw peak-to-peak, alternating with near-baseline seconds.
  This is a suspected movement artifact, to be adjudicated by the §3.1 low-frequency rule, not by
  hand.

## Amendments (before the first outcome stage)

_None yet._

## Deviations (after the first outcome stage)

_None yet._
