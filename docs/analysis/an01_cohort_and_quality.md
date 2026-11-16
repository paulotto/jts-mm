<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an01 — Cohort and data quality

Who was recorded and analysed, which takes entered, and how the quality of every EMG, IMU and
optical recording and of the pre-take electrode check was measured and flagged (flag, don't
exclude). Symbols are listed in the Notation table below.

Stages `qc` (per participant) and `qc_cohort` (cohort); code `biomech/cohort.py`,
`biomech/qc/` (`report.py`, `metrics.py`, `impedance.py`, `streams.py`); parameters
`stages.qc.*` in `biomech/etc/biomech.yaml` (the icorr2027 `study.yaml` overrides none of them).
The EMG cleaning chain and its symbols are in [an03](an03_emg_processing.md) §2; statistics
primitives (Spearman, bootstrap, Holm, Wilcoxon) in [an07](an07_statistics.md).

**Abbreviations.** QC = quality control; EMG / sEMG = (surface) electromyography; MVC / RVC =
maximal / reference voluntary contraction (an03 §2.6); SNR = signal-to-noise ratio; RMS = root
mean square; SD = standard deviation; |Z| = electrode–skin impedance magnitude; DRL = driven
right leg (bias/reference electrode); IMU = inertial measurement unit; LSB = least significant bit.

## Notation

| Symbol | Meaning |
|---|---|
| $t_k$ | sample time stamp of a stream (µs or s), $k$ in write order |
| $[t_0,t_1]$ | a phase window (master clock) |
| $\mathcal T$ | the motion phases `rest` … `free_motion` of the motion take (§3) |
| $N_\mathcal T$ | number of EMG samples in the motion phases |
| $y_n$, $x_n$ | cleaned / raw EMG sample (an03 notation) |
| $E^u_n$, $E_n$ | unmasked / masked 150-ms RMS envelope (an03 §2.4–2.5) |
| $M_n$ | dilated analysis mask (an03 §2.4); $h^r_n$ undilated hit of rule $r$ |
| $f_0$ | grid (line) frequency of the motion take |
| $P(f_j)$ | periodogram bin at frequency $f_j$ |
| $Q_p(\cdot)$ | $p$-th percentile (linear interpolation) |
| $\rho$, $r(\cdot)$ | Spearman correlation, (average) rank |
| $Z$ | electrode–skin impedance magnitude (kΩ) |
| $p$, $m$ | participant, muscle |

## 1. Cohort manifest

The participant information we have is **sex and age, encoded in the session folder names**
(`session_<id>_<sex>_<age>[_mvc][_<tag>]_<UTC>_<hash>`). There are no anthropometric, clinical,
dental or handedness data. The study's `cohort.yaml` copies sex and age once. `biomech cohort
check` re-parses the folder names and fails on any disagreement, on a missing `session.h5`, on a
missing motion take (or a missing MVC take of an included participant while the manifest's
`require_mvc` is true), on a take listed under the wrong kind, on a participant listed twice, and on an exclusion without a
reason (`cohort.py:Cohort.validate`).

```bash
biomech cohort draft --data-dir data/pilot_study/jtsmm_data --study icorr2027   # starting point
biomech cohort check --config experiments/icorr2027/study.yaml
```

**Privacy:** report demographics in aggregate only (`Cohort.demographics()`: n, n female, n male,
age mean, SD (ddof 1), min, max of the included participants; `qc_cohort.demographics`). With
n ≈ 13, age plus sex identifies people.

> **Paper wording.** Participant information was limited to sex and age, which are reported in
> aggregate only.

## 2. Inclusion and repeated takes

- **Primary takes:** the first complete motion take and the first complete MVC take per
  participant. Later takes are listed under `secondary` and never pooled.
- **Exclusions** are a human decision recorded in the manifest with a reason. They are never
  drafted automatically.
- **Dataset (icorr2027):** nobody is excluded since D17 (2026-10-21): n = 14. The participant recorded
  outside the protocol (flag `outside_protocol`) is included; its missing mandible-IMU orientation is a documented
  unavailable unit, and the `reporting_population` sensitivity run drops it. Until D17, that participant was excluded.
  s02's later takes are secondary. Reporting population of the papers: an07 *Reporting
  population*.
- **Recording days (X11, an20).** The takes were not recorded in one session: the 14 motion
  takes span six recording days within two weeks, one to four participants
  per day, with the same firmware, recorder configuration and protocol, and the stereo rig
  recalibrated between sessions (also within a day). From the fifth day on the recorder stored
  the calibration in use and the tag corners (s10–s14); s01–s09 did not. The artefacts give
  days as day 1 … 6 (dates withheld); per take: `acquisition_cohort:per_session`.

## 3. QC stage (`qc`, `qc_cohort`)

`qc` runs per participant on both takes (`biomech/qc/report.py`, `biomech/qc/metrics.py`). It reads
only the phase windows; ≈ 5–13 s per participant (pilot), most of it the EMG cleaning, which the
participant's `emg` and `timing` units then reuse (`ctx.emg_prep`, an08). Parameters: `stages.qc.*`;
the EMG chain uses `signals.emg.*` (an03).

| Table | One row per | Main columns |
|---|---|---|
| `sessions` | take | `duration_s`, `n_phases`, `missing_phases`, `n_mcu_logs` (firmware log events), integrity-report counts (`integrity_overall`, `_errors`, `_warnings`, `emg_precheck`, `hdf5_check`) |
| `phases` | phase | `phase`, `kind`, `t_start_us`, `t_end_us` (master µs), `phase_s` (`report.py:phase_table`) |
| `emg` | muscle | `line_hz`; `rest_rms_uv`, `rest_env_median_uv`, `rest_lf_uv`; `ref_phase`, `ref_p95_uv`, `ref_max1s_uv`, `ref_mask_frac`, `ref_mask_frac_effort`; `ref_uv`, `ref_from_task`, `ref_source`, `mvc_over_task` (D11); `snr_db`; `mains_share_raw/clean`; `leadoff_frac`; `mask_frac_task`, `mask_<rule>_frac`; `usable_shape`, `usable_amplitude`, `rest_noise_limited`, `lowfreq_flag` (D14) |
| `emg_check` | take × muscle | the pre-take EMG check (§3.4) |
| `imu` | IMU slot × stream | `n`, `rate_hz`, `nonmono_n`, `coverage`, `coverage_min_phase`, `acc0_frac` |
| `optical` | body × phase (every phase, calibration holds included) | `n_frames`, `lock_frac` |
| `imu_intervals` | IMU stream (quaternion, accelerometer per slot) | `rate_hz`, `interval_median_ms`, `interval_p01_ms`, `interval_p99_ms`, `interval_max_ms`, `long_interval_pct` (> 1.5 × median) |
| `imu_coverage` | IMU stream × experiment phase | `coverage_pct` (samples present / expected at the median interval, capped at 100 %) |
| `timeline_lanes` | stream lane | `lane`, `kind`, `rate_hz`, `lock_pct` (optical), `t0_us` (time origin = first phase start) |
| `timeline_segments` | covered interval | `lane`, `t0_s`, `t1_s` (a gap > 3 sample intervals splits a bar; optical: lock runs; `phase` for the protocol lane) |

The last four tables and the arrays `timeline__<body>__t_s` / `__fit_rms_mm` (the optical fit
residual, stored by the capture as `reprojection_rms` with `rms_kind = kabsch_mm`: RMS of the
rigid tag fit in mm, E31; min/max per bin, ≤ 4000 points; `streams.py:172-173`) and
`scene__<phase>__<body>__xyz` / `__key` (body origins in the reference camera frame, thinned
with stride ⌊n/1500⌋ (`SCENE_MAX_FRAMES`, `streams.py:197`; a phase with n ≥ 1500 frames keeps
1500 to < 3000), three key poses) are what the data views
`data_imu_quality`, `data_timeline` and `data_tracking_scene` draw (`biomech/qc/streams.py`), so
they render from the result store or a snapshot without the recorded session.

`qc_cohort` adds `demographics`, `emg_flags` (channels not usable for shape or amplitude,
rest-noise limited, or with `mask_frac_task` > 0.01), `overview` (per participant:
`emg_unusable_amplitude_n`, `emg_rest_noise_limited_n`, mean `emg_mask_frac_task` over channels,
`lock_<body>` = unweighted mean of `lock_frac` over the motion phases, `quat_cov_<slot>`), and the
impedance tables `impedance` and `impedance_assoc` (§3.4).

"Motion phases" $\mathcal T$ below are `rest`, `open_close`, `protrusion_retrusion`,
`left_right`, `chewing`, `free_motion` of the motion take (`report.py:MOTION_PHASES`).

### 3.1 EMG metrics (`report.py:emg_table`)

- **Grid frequency** `line_hz`: the motion take's value from `take_line_hz` (an03 §2.1).
  Window: the first `signals.emg.line_hz_window_s` = 60 s from the start of the take's first
  phase; channel: the one with the largest raw SD there. The nominal grid
  (`signals.emg.nominal_line_hz`, 50 Hz) is tried first, then the other grid (60 Hz); if
  neither tone passes the acceptance test, `nominal_line_hz` is used. The MVC take gets its own
  value the same way (used for its cleaning, not stored here). Estimating it on the short,
  clean rest window latched onto noise (48.7 Hz on s01), and `remove_mains` then fitted the
  wrong frequency.
- **Cleaning:** as in an03 (robust mains fit → 20–450 Hz zero-phase → artifact masks; no spike
  suppression).
- **Rest statistics** (motion take, `rest` phase; `EmgPrep.rest`, an03 §2.8):

  $$
  \texttt{rest\_rms\_uv}=\sqrt{\frac{1}{\lvert V\rvert}\sum_{n\in V}(y_n-\bar y_V)^2},\ V=\{n\in\text{rest}:\lnot M_n\},\qquad \texttt{rest\_env\_median\_uv}=\operatorname{median}_{n\in\text{rest}}E^u_n,\qquad \texttt{rest\_lf\_uv}=\operatorname{median}_{n\in\text{rest}}\ell_n .
  $$

  - `rest_rms_uv`: SD (ddof 0) of the cleaned rest signal over the samples **outside** the rest
    phase's analysis mask (E22; formerly over all samples); NaN if all are masked;
  - `rest_env_median_uv`: median over the phase of the 150-ms moving RMS of the cleaned signal
    **without** any mask (an03 §2.4, $E^u$);
  - `rest_lf_uv`: median over the phase of the 1–20 Hz level $\ell$ of the **raw** signal
    (an03 §2.4), the scale of the low-frequency rule (a QC flag since D14).
- **Reference amplitude:** `ref_p95_uv` = 95th percentile of the masked 150 ms envelope in
  `mvc_opening` (digastric) or `mvc_closing` (masseter, temporalis) (`ref_phase`); `ref_max1s_uv`,
  `ref_mask_frac` and `ref_mask_frac_effort` as defined in an03 §2.6. NaN without an MVC take.
  `ref_uv` (the normalization reference = largest activity recorded), `ref_source` (`mvc` or
  `task`: a task exceeded the reference contraction) and `mvc_over_task` as in an03 §2.7.
  The SNR and the usability flags keep the frozen MVC/RVC p95.
- **SNR** (`metrics.py:snr_db`): reference level over resting envelope level,

  $$
  \texttt{snr\_db}=20\log_{10}\frac{A_\text{ref}}{\texttt{rest\_env\_median\_uv}},\qquad A_\text{ref}=\begin{cases}\texttt{ref\_p95\_uv}&\text{if finite}\\ \texttt{ref\_uv}&\text{otherwise (}\texttt{ref\_from\_task}=\text{True)}\end{cases}
  $$

  NaN when either operand is non-finite or ≤ 0. Without an MVC/RVC reference (no MVC take, or
  the reference phase missing) the D11 reference `ref_uv` (the largest activity recorded,
  i.e. the task maximum) sets the SNR and `ref_from_task` is True; the amplitude family still
  needs a valid MVC/RVC reference.
- **Mains share** (`jtsmm.emg.mains.mains_share`, via `metrics.py:channel_mains_share`), on the
  motion take's `rest` phase only: `mains_share_raw` on the raw signal minus its per-channel
  mean, `mains_share_clean` on the cleaned signal. Per channel: subtract the mean, Hann window
  $w_n$ over the sample index (no resampling onto the timestamps; $N$ = phase samples),
  $P(f_j)=\lvert\mathrm{rfft}(x\cdot w)\rvert^2_j$, $f_j=jf_s/N$; then

  $$
  S=\frac{\sum_{f_j\in B\cap H}P(f_j)}{\sum_{f_j\in B}P(f_j)},\qquad B=\{f_j\in[20,450]\ \text{Hz}\},\qquad H=\bigcup_{k\in\mathcal K}\{f_j:\lvert f_j-kf_0\rvert\le 1.5\ \text{Hz}\},
  $$

  $\mathcal K=\{k\le 9: kf_0<\min(452\ \text{Hz},f_s/2)\}$ (the fitted harmonics, an03 §2.2),
  $f_0$ = the motion take's `line_hz` (not the per-phase value). $S=0$ for < 16 samples or zero
  in-band power. ≈ 0.9 means the band is dominated by interference, < 0.1 is clean.
- **Lead-off:** $\texttt{leadoff\_frac}=\frac{1}{N_\mathcal T}\sum_{n\in\mathcal T}\mathbf{1}[\text{P or N comparator bit set}]$.
  Lead-off monitoring was on during the pilot (`mcu.emg.leadoff.during_recording: true`).
- **Mask shares** (masks are built per phase window, an03 §2.4):

  $$
  \texttt{mask\_frac\_task}=\frac{1}{N_\mathcal T}\sum_{n\in\mathcal T}M_n,\qquad \texttt{mask\_<rule>\_frac}=\frac{1}{N_\mathcal T}\sum_{n\in\mathcal T}h^{r}_n .
  $$

  - $M$ = the dilated **analysis mask**, the union of the rules in
    `signals.emg.mask.analysis_rules` (default `rail`, `leadoff`, `envelope`; D14);
  - $h^r$ (`rail`, `leadoff`, `envelope`, `lowfreq`) = undilated hits of that rule, for every
    rule whether it excludes samples or only flags them (0 for a rule that was not evaluated).
    Rules overlap and dilation adds samples, so the rule shares do not add up to
    `mask_frac_task`; `mask_lowfreq_frac` is not part of it at all under the default rules.

> **Paper wording.** Signal quality was described per channel by the resting noise (standard
> deviation of the cleaned signal at rest), the SNR (20·log10 of the 95th percentile of the
> reference-contraction envelope over the median resting envelope), the share of resting
> in-band power within ±1.5 Hz of the line harmonics, the share of samples with electrode
> lead-off, and the share of masked samples.

### 3.2 Usability instead of exclusion (analysis_plan v1.0 §3, A1)

Nothing is excluded. Each channel gets flags (`stages.qc.usability.*`), and each analysis family
takes what it needs:

| Flag | Definition | Used by |
|---|---|---|
| `usable_shape` | $\texttt{snr\_db}$ finite and $\ge$ `snr_min_db` (6 dB) | waveform shape, co-activation, SPM, spectral cohort |
| `usable_amplitude` | `usable_shape` and `ref_p95_uv` finite and `ref_mask_frac_effort` ≤ `ref_mask_max` (0.10): ≤ 10 % of the MVC/RVC effort samples in the analysis mask (effort = unmasked envelope ≥ 25 % of the unmasked reference p95; D9) | normalized levels (% of max recorded, D11: activation map, LMM, rest %), clench indices |
| `rest_noise_limited` | `rest_rms_uv` > `rest_noise_uv` (20 µV) | QC reporting only: counted in `qc_cohort` (`emg_flags`, `emg_rest_noise_limited_n`). No analysis filters on it; the cohort `rest_activity` table includes these channels |
| `lowfreq_flag` | `mask_lowfreq_frac` > `lowfreq_flag_frac` (0.10): > 10 % of the task samples hit by the low-frequency rule (D14) | QC reporting only; the viewer shows these samples as "flagged (kept)". No analysis filters on it; its effect is the sensitivity analysis (`emg.mask_sensitivity`, `biomech sensitivity lowfreq_mask`; an03 §2.4) |
| `mask_*_frac` | share of task samples hit per rule (§3.1) | reporting |

A NaN `ref_mask_frac_effort` (no effort samples) makes `usable_amplitude` False.

Timing usability is decided per cycle (an05 §3); a channel-cycle with more than
`stages.emg.cycle_mask_max` (0.10) of its samples in the analysis mask is NaN in the cycle
profiles (an03 §3). Decoding uses all channels, with only rail and lead-off masked (A3).

> **Paper wording.** No channel was excluded. A channel entered waveform-shape analyses when its
> SNR was at least 6 dB, and amplitude analyses when, in addition, no more than 10 % of its
> reference-contraction effort was masked.

### 3.3 IMU and optical metrics (`report.py:imu_table`, `optical_table`; `metrics.py`)

- **`rate_hz`:** $1/\operatorname{median}(\Delta t)$ over the differences of the unique,
  sorted timestamps.
- **`coverage`:** the share of motion-phase time not inside a sample gap > `stages.qc.imu.gap_s`
  (50 ms). Per phase, with $t_{(1)}<\dots<t_{(K)}$ the unique stamps inside $[t_0,t_1]$ and
  gaps $g$ the differences of the sequence $(t_0,t_{(1)},\dots,t_{(K)},t_1)$:

  $$
  \mathrm{cov}=1-\frac{\sum_{g>\texttt{gap\_s}}g}{t_1-t_0},
  $$

  then averaged over the motion phases weighted by duration; `coverage_min_phase` is the worst
  phase. The stretch before the first and after the last sample counts as a gap, so a stream
  that stops shows up. On the participant recorded outside the protocol the mandible quaternion coverage is 0.0, while its accelerometer
  stays at 1.0.
- **`nonmono_n`:** $\#\{k: t_{k+1}-t_k\le 0\}$, timestamps not strictly increasing in write
  order. These are sanitized in IMU preprocessing (an04).
- **`acc0_frac`:** share of motion-phase samples with sensor accuracy 0.
- **`lock_frac`:** the share of optical frames with a finite pose, per body
  (`stages.qc.optical.bodies` = `mta`, `cra`) and phase. The headpiece (`cra`) is not tracked
  during the mandibular landmark holds by design.

### 3.4 Electrode impedance: the pre-take EMG check (`qc/impedance.py`, D21)

Before each take the recorder ran `jtsmm emg check`. The result is stored in the session
(`/meta/emg_check_json`, `/meta/emg_check_passed`); the values are those computed by the
recorder at the time (`jtsmm/emg/impedance.py`, `jtsmm/emg/check.py`).

**Per muscle channel the check records:**
- **|Z|** (`jtsmm.emg.impedance.estimate_impedance`): the ADS1299 AC lead-off current is a
  ±$I$ square wave at $f_t$ = $f_\text{CLK}/2^{16}$ = 31.25 Hz (independent of the sample rate;
  evaluated at $31.25\ \text{Hz}\times$ the ADC clock ratio measured from the timestamps). Its
  fundamental across the electrode pair has amplitude $V_\text{tone}=\tfrac{4}{\pi}IZ_\text{loop}$, so

  $$
  \lvert Z\rvert=\frac{V_\text{tone}}{\tfrac{4}{\pi}\,I_\text{lead-off}}-R_\text{protect},\qquad R_\text{protect}=0.2\ \text{k}\Omega\ (2\times 100\ \Omega\text{ series resistors}),
  $$

  $I_\text{lead-off}$ = `mcu.emg.leadoff.current_na` (6 nA default; 6 or 24 nA allowed).
  $V_\text{tone}$ is the least-squares amplitude $\sqrt{a^2+b^2}$ of a sine/cosine pair at
  $f_t$ fitted at the real sample times jointly with an intercept, a linear trend and the
  mains harmonics below $\min(450\ \text{Hz},0.45f_s)$ as nuisance terms. The tone must exceed
  4 × the median amplitude at 12 neighbouring frequencies ($f_t\pm k/T$, $k$ = 3…8, $T$ =
  capture duration, > 2 bins away from mains), and $\lvert Z\rvert\ge 1$ kΩ; otherwise
  |Z| = n/a. All six pairs are measured at once, each with its own current source and sink.
  The contact limit is 50 kΩ (`CONTACT_Z_MAX_KOHM`).
- **Noise** (`check.py:_bandpass_rms_no_mains`): from a capture of a few seconds with lead-off
  off, mains removed (`remove_mains` at the capture's measured grid frequency, constant-amplitude
  model), then the 20–450 Hz RMS by Parseval,
  $\sqrt{\sum_{f_j\in[20,450]}\gamma_j\lvert X_j\rvert^2/N^2}$ ($X$ = rfft of the mean-removed
  capture, $\gamma_j=2$ except DC and Nyquist); limit 10 µV, 15 µV on channels 5–6
  (`NOISE_LIMIT_UV`). **Line peak:** least-squares amplitude of the line tone (limit 50 µV,
  `LINE_PEAK_LIMIT_UV`).

The DRL/reference electrode cannot be measured (CH8 reads the bias node, IN8P is unwired).

**Bench check of the |Z| scale (B1, 2026-11-04; change log of the dataset study, entry written
before the measurement).** 1 % resistors of nominal 10, 22, 47 and 100 kΩ (multimeter values used as the
truth) across INxP–INxN of each channel in turn, with cables as for a participant, the channel
alone powered (`--active-channels n`), the bias drive on and sensing that channel only
(`--bias-channels n`) and the DRL lead on the resistor's N side as the body's common mode; two
captures per resistor (re-seated), at 6 nA (the dataset's current) and 24 nA. A settings
comparison before the series (all channels powered, no DRL lead) read the same |Z| within
0.01 kΩ. <!-- num-ok: bench settings comparison (pre1 captures), printed by the bench script -->
Result at 6 nA: channel scale $k=\sum\lvert Z\rvert R/\sum R^2$ median `\numBenchImpK`
(0.95), range `\numBenchImpKMin`–`\numBenchImpKMax` (0.92–0.99), largest single error
`\numBenchImpMaxErr` (8.5 %), the two captures within `\numBenchImpRepeat` (0.2 %): the stored
|Z| stand (decision rule ±10 %). The error is not a pure scale: it falls with the resistance
(a few percent high at 10 kΩ, up to `\numBenchImpMaxErr` (8.5 %) low at 100 kΩ), so near the
contact limit |Z| reads a few percent low. Sensitivity (descriptive, `--store`): with |Z|
divided by the channel's k, the median would be `\numBenchImpMedianScaled` (23 kΩ) (recorded
`\numImpMedian` (22 kΩ)). The counts of channels above the check's former contact limit are no
longer reported in the dataset paper (the limit was not a recording criterion and has no
source; 2026-11-09); the systems paper keeps its bench sensitivity counts. Tables:
`hardware/bench/tables/emg_impedance_{captures,channels,sensitivity}.csv`
(`scripts/bench/emg_impedance_bench.py`).

**Bench check of the board's own noise (B2, 2026-11-04; change log of the dataset study).** Every
channel's inputs shorted with separate short cables and joined to the DRL input, dataset
settings, the check's noise measurement (`jtsmm emg check --no-impedance`, four captures). The
board's noise RMS is `\numBenchNoiseMedian` (0.5 µV) (at most `\numBenchNoiseChannelMax`
(0.6 µV) per channel), no measurable line peak: `\numBenchNoiseRatio` (0.03) of the
participants' check noise `\numBenchNoiseVivoMedian` (11.8 µV) (same metric). The in-vivo check
noise is therefore electrode, cable and participant noise (the check's noise and line limits
are design values without a source and are not reported in the dataset paper; change-log entry J26); the worn electrode cables were not part of the bench
check (they cannot be shorted). Tables `hardware/bench/tables/emg_noise_{captures,channels}.csv`
(`scripts/bench/emg_noise_bench.py`).

**Tables:**
- **`qc:emg_check`** (`impedance.py:check_table`): one row per participant, take and muscle
  (`z_kohm`, `z_quality`, `contact_ok` = |Z| ≤ 50 kΩ, `noise_rms_uv`, `noise_total_uv` (without
  mains removal), `noise_ok`, `line_peak_uv`, `line_ok`, `check_passed` for the whole check,
  `line_hz_check`). The takes are ordered by session creation time (`order`, `minutes` since
  the first take's check), and the electrodes stay on between them.
- **`qc_cohort:impedance`** (`impedance.py:summary`): per take and muscle (and over all
  muscles) the median, quartiles and range of |Z|, `above_limit_n` (measured channels with
  |Z| > 50 kΩ; a channel without an impedance value is not counted — formerly,
  `contact_ok` False also counted unmeasured channels), `noise_fail_n`, `line_fail_n`, `takes_passed_n` / `takes_n`. Second vs
  first take (same electrodes; E30), unit = participant:

  $$
  \Delta_{pm}=Z^{(2)}_{pm}-Z^{(1)}_{pm},\qquad \tilde\Delta_p=\operatorname{median}_m\Delta_{pm},
  $$

  rows `second_minus_first` also store `minutes_median` (median time of the second check after
  the first, over participants) and `n_channels` (channel pairs with both values;
  `impedance.py:141-147`). Row `second_minus_first` / `all`: spread of $\tilde\Delta_p$ over participants, `n_negative`,
  and the two-sided Wilcoxon signed-rank test of $\tilde\Delta_p$ against 0 (`wilcoxon_p`;
  zeros dropped; exact null distribution without ties, else normal approximation; NaN below 6
  participants; an07). Row `second_minus_first` / `all_channels`: the $\Delta_{pm}$ pairs,
  descriptive only (the six channels of a participant are not independent).
- **`qc_cohort:impedance_assoc`** (`impedance.py:associations`): for the motion take, the
  check against the recorded rest signal of `qc:emg`. Relations (x → y): |Z| vs `snr_db`,
  `rest_rms_uv`, `rest_lf_uv`, `mains_share_raw` (Holm family `check_vs_recorded`, 4 tests);
  check noise vs `rest_rms_uv` and line peak vs `mains_share_raw` (family `check_vs_check`,
  2 tests). Per relation, over the $n$ channels with both values (≥ 6 and non-constant):
  - pooled Spearman $\rho$ = Pearson correlation of the average ranks (`rho_pooled`), with a
    95 % percentile CI (`rho_pooled_lo/hi`) from $B$ = 2000 resamples of whole participants with
    replacement (seed = config `seed`, 42; resamples with constant x or y skipped);
  - within-group rank correlations (`rho_within_pid`, `rho_within_muscle`): pooled ranks centred
    on their group mean, $\tilde r_i=r_i-\bar r_{g(i)}$, then
    $\operatorname{corr}(\tilde r^x,\tilde r^y)$;
  - the adjusted **rank slope** (`rho_adj`; a slope, not a correlation coefficient) from the OLS
    rank regression with participant and muscle as fixed effects:

    $$
    r(y_i)=\beta_0+\beta\,r(x_i)+\sum_{p\ne p_1}\gamma_p\,\mathbf{1}[p_i=p]+\sum_{m\ne m_1}\delta_m\,\mathbf{1}[m_i=m]+\varepsilon_i,
    $$

    $\widehat{\mathrm{SE}}(\hat\beta)=\sqrt{s^2\,[(A^\top A)^{+}]_{\beta\beta}}$,
    $s^2=\mathrm{RSS}/\nu$, $\nu=n-\operatorname{rank}(A)$ ($A$ = design matrix),
    `p_adj` = $2\,(1-F_{t,\nu}(\lvert\hat\beta/\widehat{\mathrm{SE}}\rvert))$, and `p_holm` =
    Holm within its family (`holm_family`; an07 "Families and multiplicity").

  The pooled ρ mixes the differences between participants and between muscles into the
  association. The adjusted slope asks whether a channel with a higher |Z| than its
  participant's and its muscle's usual one also records a different rest signal; both ranks run
  over $1..n$, so $\beta$ is the change in the rank of y per rank of x.

**Nothing is excluded on the check.** The check judged the contact; the recorded data are
judged by `qc:emg` (usability flags above). Almost every check "failed" on the strict
noise limit, without any consequence for the recorded channels.

> **Paper wording.** Electrode–skin impedance was measured before each recording from the
> amplifier's 31.25-Hz AC lead-off current (least-squares tone amplitude, corrected for the
> square-wave fundamental and the on-board series resistance). Its association with the
> recorded resting signal was assessed with Spearman correlations and with a rank regression
> including participant and muscle as fixed effects (Holm-corrected); the change between the
> first and second recording was tested per participant with the Wilcoxon signed-rank test.
