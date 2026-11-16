<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an03 — sEMG processing, normalization and outcomes

How the surface EMG is cleaned (mains removal, band-pass, artefact masks), enveloped and
normalized, which channels are usable for which analysis, and how the activation, co-activation,
waveform and spectral outcomes are computed. Symbols are listed in the Notation table below.

Stages `qc`, `emg`, `emg_relevance`, `timing` (onsets: an05), cohort `stats_emg`. Code:
`biomech/emg/` (`pipeline.py`, `prepare.py`, `artifacts.py`, `analysis.py`, `relevance.py`,
`causal.py`), `biomech/spectral.py`, `biomech/stats/cohort_emg.py`; primitives in `jtsmm.emg`
(`mains.py`). Parameters: `signals.emg.*` and `stages.emg.*` / `stages.emg_relevance.*` in
`biomech/etc/biomech.yaml` (the icorr2027 `study.yaml` overrides none of them); hard-coded
constants in §6. Statistics primitives (BCa bootstrap, Holm, mixed model, SPM) are defined in
[an07](an07_statistics.md); data-quality metrics and usability flags in
[an01](an01_cohort_and_quality.md) §3.

**Abbreviations.** sEMG = surface electromyography; MVC = maximal voluntary contraction (closers:
clench); RVC = reference voluntary contraction (openers: opening against about 10 N, not a true
maximum); RMS = root mean square; PSD = power spectral density; MNF / MDF = mean / median
frequency; SPM = statistical parametric mapping (spm1d); CCI = co-contraction index; LMM = linear
mixed model; BIC = Bayesian information criterion; LSB = least significant bit of the ADC; IQR =
interquartile range; SD = standard deviation; BCa = bias-corrected and accelerated (bootstrap);
CI = confidence interval; SNR = signal-to-noise ratio; QC = quality control; JTS-MM = the
multimodal jaw tracking system of this project.

## Notation

| Symbol | Meaning |
|---|---|
| $c$, $n$ | channel (muscle) index, sample index within a phase window |
| $t_n$ | real sample time (s; from `t_master_us`), $\bar t$ its mean over the block/window |
| $f_s$ | nominal EMG rate of `/emg` (`sample_rate_sps`; 4000 Hz in every pilot session) |
| $x_n$ | raw sample (µV) |
| $\tilde x_n$ | mains-removed sample (after step 2) |
| $y_n$ | cleaned sample: mains removed and band-passed (after step 3), unmasked |
| $f_0$, $f_n$ | grid (line) frequency used, nominal grid frequency |
| $\mathcal K$, $K$ | set and number of fitted mains harmonics |
| $P_m(\tau)$ | Legendre polynomial of order $m$ on block time $\tau\in[-1,1]$ |
| $\operatorname{odd}(T)$ | centred window length for duration $T$: $2\lfloor T f_s/2\rfloor+1$ samples (odd; an exact even product is rounded up, 600 → 601; `biomech.timebase.odd_window`, an02 §1) |
| $W_E$, $W_L$ | $\operatorname{odd}(0.15\ \text{s}) = 601$, $\operatorname{odd}(0.25\ \text{s}) = 1001$ samples at 4 kHz |
| $E^{u}_n$ | unmasked moving RMS envelope of $y$ over $W_E$ |
| $E_n$ | masked moving RMS envelope of $y$ (NaN where too few valid samples) |
| $\ell_n$ | 1–20 Hz level of the raw signal (moving RMS over $W_L$) |
| $h^{r}_n$ | hit of artifact rule $r\in\{\text{rail},\text{leadoff},\text{envelope},\text{lowfreq}\}$ (undilated) |
| $M_n$ | analysis mask (dilated union of the analysis rules); $v_n=\lnot M_n\wedge \text{finite}(y_n)$ valid |
| $Q_p(\cdot)$ | $p$-th percentile (linear interpolation; NaN skipped where stated) |
| $P_u$ | $Q_{95}(E^u)$ in the reference phase |
| $R_c$ | normalization reference of channel $c$ (µV; default: largest activity recorded) |
| $e_n$ | normalized envelope $100\,E_n/R_c$ (% of the largest activity recorded) |
| $I_i$ | EMG samples of kinematic cycle $i$ |
| $P(f)$ | one-sided PSD (µV²/Hz) |

## 1. Montage

This is the one montage table of the project; the setup and hardware docs (cv03 §3, h03 §2)
point here. It is the mapping of the electrode cables and the one the code uses:

| CH | Muscle (name in the code) | Status |
|---|---|---|
| CH1 | masseter_r | cable label; side also confirmed by the Delsys side check (VD23) |
| CH2 | temporalis_r | cable label; VD23 |
| CH3 | digastric_r | cable label (hardware check 2026-11-03); **stored as digastric_l before** |
| CH4 | digastric_l | cable label (hardware check 2026-11-03); **stored as digastric_r before** |
| CH5 | masseter_l | cable label; VD23 |
| CH6 | temporalis_l | cable label; VD23 |

CH7 = bias_debug, CH8 = DRL (driven right leg, the bias/reference electrode).

- **Where the names come from.** The recorder writes a fixed name list into every `session.h5`
  (`/emg` attribute `channel_names`, from `jtsmm.config.catalog.FULL_CHANNEL_NAMES`); it is not
  entered per session. Channels are selected **by name**. Lead-off bit *i* belongs to CH(*i*+1).
  Openers = digastric L/R (`io.session.OPENERS`); closers = masseter and temporalis L/R
  (`CLOSERS`).
- **Sessions recorded before 2026-11-03 (all 33 with names).** Their files name CH3
  `digastric_l` and CH4 `digastric_r`, the reverse of the cables. The files are not changed:
  every reader maps exactly that stored list to the table above
  (`jtsmm.io.session_reader.montage_corrected`, used by the replay viewer, biomech, the ML loader
  and the integrity checker; `EmgStream.montage_note` says so), and the impedance table takes
  the muscle of a pre-take check channel from its channel number (`qc/impedance.py`), not from
  the name stored in `emg_check_json`. The 9 older files without names fall back to the
  recorder's list (`biomech.io.session.MUSCLES` = `FULL_CHANNEL_NAMES[:6]`). Results computed
  before this correction carry the old digastric names: their per-side digastric values are
  swapped (errata E37 / EV12).
- **Side check against Delsys (VD23, an14 §11b; `validation/emg_side.py`).** In the
  side-informative laterotrusion task (`left_right`) the side index of the JTS-MM profiles
  against the side-labelled Delsys sensors is clearly positive for the closers wherever the
  two sides can be told apart (`v15`: P1 temporalis +1.60 / +1.63, P2 masseter +1.71 / +1.67,
  P2 temporalis +0.66 / +0.73; P1 masseter is not identifiable, −0.09 / +0.02; never clearly
  negative), i.e. the code's masseter and temporalis sides are right, provided the Delsys
  sensors were placed per their labels. The digastrics co-activate on both sides (index ≈ 0),
  so their side cannot be checked from the data.
- **Hardware check (the authors, 2026-11-03).** Every cable label matches the name the viewer showed
  (the recorder's list) except the digastrics, which were swapped: CH3 is the right, CH4 the
  left digastric. The closers agree with VD23, so the cable labels give the participant's sides.
  The recorder's list and the readers were corrected the same day (above). The list noted
  earlier as the cable labels (1 M_L, 2 T_L, 3 D_L, 4 D_R, 5 M_R, 6 T_R) does not
  describe the cables as checked. Pooled opener / closer results are unaffected; per-side
  digastric values and the JTS-MM ↔ Delsys digastric pairing changed with the correction.

## 2. Offline chain (`emg/pipeline.py`, `emg/prepare.py`)

Each phase window is processed on its own (`EmgPrep._clean_uncached`, memoized per take and
phase): raw µV, real sample times `t_master_us`, $f_s$ = the nominal `/emg` rate. Order:
grid frequency → mains removal → band-pass → artifact masks → masked envelope → references →
normalization. (Older references to "an03 §2 step N" mean §2.N below.)

> **Paper wording (chain overview).** sEMG was sampled at 4 kHz. Each protocol phase was
> processed separately: power-line interference was removed by fitting and subtracting the
> line frequency and its harmonics up to 450 Hz, the signal was band-pass filtered (20–450 Hz,
> fourth-order Butterworth, zero phase), artifacts were masked rather than removed or
> replaced, and the amplitude envelope was the 150-ms moving RMS of the unmasked samples.

### 2.1 Grid frequency

*What and why.* The mains model needs the line frequency to ≈ 0.01 Hz: the grid wanders
(pilot: 50.016 vs 49.986 Hz within an hour), and a frequency error $\delta f$ becomes
$k\,\delta f$ at harmonic $k$ (×9 at 450 Hz).

- **Take value**: window = the first `signals.emg.line_hz_window_s` = 60 s from the start of
  the take's first phase (chosen in `emg/prepare.py:_line_hz`); channel = the one with the
  largest raw SD there (`pipeline.py:take_line_hz`, which also holds the 50 ↔ 60 Hz fallback
  below). `measure_line_hz` is tried at $f_n$ = `signals.emg.nominal_line_hz` (50 Hz), then at
  the other grid (60 ↔ 50 Hz); if neither is accepted, $f_0 = f_n$. Motion and MVC take each get
  their own value.
- **Estimate** (`jtsmm/emg/mains.py:estimate_line_hz`): on a uniform grid (step $1/f_s$ from the
  first to the last stamp, linear interpolation of $x$; used for the FFT only), first
  $N\le 120\ \text{s}\cdot f_s$ samples, mean removed, Hann window $w_n$:

  $$
  P_j=\Bigl|\sum_{n=0}^{N-1} w_n\,(x_n-\bar x)\,e^{-i2\pi jn/N}\Bigr|^2,\qquad f_j=j\,\frac{f_s}{N},\qquad \Delta f=\frac{f_s}{N}.
  $$

  Peak bin $j^{\star}=\arg\max_{|f_j-f_n|\le 2\,\text{Hz}}P_j$; parabolic vertex through the log power
  (a Hann main lobe is close to a Gaussian, whose log is a parabola), with
  $a,b,c=\ln P_{j^{\star}-1},\ln P_{j^{\star}},\ln P_{j^{\star}+1}$:

  $$
  \hat f=f_{j^{\star}}+\operatorname{clip}\!\Bigl(\tfrac12\,\frac{a-c}{a-2b+c},\,-\tfrac12,\,\tfrac12\Bigr)\Delta f\quad(\text{offset }0\text{ unless }a-2b+c<0).
  $$

  Precision ≈ 0.002 Hz from 5 s of data.
- **Acceptance** (`mains.py:measure_line_hz`, max deviation $\Delta$): reject if
  $|\hat f-f_n|>\Delta$ ($\Delta$ = 0.3 Hz for the take value). Tone test at the real sample
  times over the whole window: least squares

  $$
  x_n=a\sin(2\pi f t_n)+b\cos(2\pi f t_n)+c_0+d\,(t_n-\bar t)+\varepsilon_n,\qquad A(f)=\sqrt{a^2+b^2},
  $$

  noise $\tilde A=\operatorname{median}\{A(\hat f\pm k/T)\}_{k=3,\dots,8}$ (12 values; $T$ =
  uniform-grid duration $N_\text{grid}/f_s$). Accept iff $A(\hat f)>0$ and $A(\hat f)\ge 5\,\tilde A$.
  Fewer than 64 samples or $f_n\ge f_s/2$ → not accepted.
- **Per phase** (`pipeline.py:refine_line_hz`, `signals.emg.line_hz_per_phase` = true):
  `measure_line_hz` on the phase's largest-SD channel with $f_n$ = the take value and
  $\Delta$ = 0.1 Hz; otherwise the take value. With 0.05 Hz drift over 5 min the residual late
  in a take was 1.44 µV with the take value, 0.55 µV with the per-phase value (synthetic).

Estimating on the short, clean rest window alone latched onto noise (48.7 Hz on s01); hence the
60-s take value with the tone test.

> **Paper wording.** The line frequency was estimated per recording from the first minute of
> data (log-parabolic interpolation of the Hann-windowed periodogram peak near 50 Hz, accepted
> only if the fitted tone exceeded its spectral neighbourhood five-fold) and refined per
> protocol phase within ±0.1 Hz.

### 2.2 Mains removal

*What and why.* Fit and subtract the coherent interference instead of notching: only the
sinusoids leave, broadband EMG at the harmonic frequencies stays, and refitting per block tracks
amplitude and phase drift (`jtsmm.emg.mains.remove_mains`; called with `robust=True`,
`amp_order="auto"` from `pipeline.py:clean`).

- **Blocks.** Per channel and abutting block, at the real sample times. Block length
  $B=\max(\operatorname{round}(1\ \text{s}\cdot f_s),\,2n_\text{sin}+4)$ samples with
  $n_\text{sin}=2K(M_\text{max}+1)$ (= 4000 samples at 4 kHz); a remainder shorter than $B$ is
  merged into the previous block; a block with $<n_\text{sin}+4$ samples is left unchanged.
- **Model.** $t$ = time since the block's first sample (s), $\tau=2t/t_\text{last}-1\in[-1,1]$:

  $$
  \tilde x_n=x_n-\hat h_n,\qquad x_n=\underbrace{\sum_{k\in\mathcal K}\sum_{m=0}^{M}P_m(\tau_n)\bigl[a_{km}\sin(2\pi kf_0t_n)+b_{km}\cos(2\pi kf_0t_n)\bigr]}_{h_n}+c_0+d\,(t_n-\bar t)+e_n .
  $$

  - Harmonics (`harmonic_numbers`): $\mathcal K=\{k\in 1,\dots,9: k f_0<\min(450+2\ \text{Hz},\,f_s/2)\}$
    (9 at 50 Hz; at 50.02 Hz the 9th sits at 450.18 Hz and is still fitted). Higher harmonics
    lie above the band-pass and are not fitted (fitting them as well changed RMS by ≤ 0.3 %).
  - $c_0$ and $d$ are nuisance terms: fitted, **not subtracted** (electrode DC offset is not
    orthogonal to a sinusoid at gappy timestamps); only $\hat h_n$ is subtracted.
- **Robust weights** (`signals.emg.robust_mains` = true, D8): starting from the ordinary fit, 2
  reweighted passes (`_robust_weights`, `_reweighted_fit`). Each pass, with residual $r_n$ of the
  previous fit and $W_r=\operatorname{odd}(0.05\ \text{s})$ = 201 samples:

  $$
  \rho_n=\sqrt{\frac{1}{W_r}\sum_{|j|\le (W_r-1)/2} r_{n+j}^2}\ \ (\text{zeros beyond the block}),\qquad \rho_\text{fl}=Q_{10}(\rho),\qquad w_n=\frac{1}{\max(\rho_n,\rho_\text{fl})^2},
  $$

  then the weighted least-squares refit $\hat\beta=\arg\min_\beta\sum_n w_n\,(x_n-D_n\beta)^2$
  ($D$ = design matrix). The channel stays unweighted if $\rho_\text{fl}$ is non-finite or ≤ 0.
  An ordinary fit also "explains" a burst's own energy at the harmonic frequencies and
  subtracts that error as a steady tone over the whole block, so quiet samples next to a burst
  gained a residual hum (synthetic: baseline 3 → 5 µV, onsets early by 30–130 ms).
- **Amplitude modulation** (`signals.emg.mains_amp_order` = `auto`; integer = fixed $M$; D8).
  With `auto` ($M_\text{max}$ = 3): the weights are computed once per block from the robust fit
  of the order-0 model (nuisance + order-0 columns); with these weights, per channel, the
  columns are nested (nuisance; order 0 for all $k$; order 1; …) and

  $$
  \mathrm{BIC}_M=n\ln\!\frac{\mathrm{RSS}_M}{n}+q_M\ln n,\qquad q_M=2+2K(M+1),\qquad M=0,\dots,3,
  $$

  $n$ = block samples, $\mathrm{RSS}_M$ = weighted residual sum of squares; $\hat M=\arg\min\mathrm{BIC}_M$.
  All RSS come from one Cholesky factor of the weighted Gram matrix: $D_w^\top D_w=LL^\top$,
  $Lz=D_w^\top x_w$, $\mathrm{RSS}_q=\lVert x_w\rVert^2-\lVert z_{1..q}\rVert^2$ (floored at
  $10^{-300}$); if the factorization fails, a ridge $10^{-10}\operatorname{tr}(G)/p$ is added.
  The chosen order's coefficients solve $L_{1..q,1..q}^\top\beta=z_{1..q}$. Hum modulated by
  movement (cable/electrode motion) otherwise leaves a residual locked to the movement: at
  1 Hz / 30 % modulation 2.41 µV with $M=0$ and 0.88 µV with BIC selection, while steady hum
  costs nothing extra (0.42 µV either way; synthetic 3-µV EMG).
- **Remaining limit:** strong, fast modulation within a block (≥ 1.5 Hz at 50 % depth) is only
  partly captured (≈ 1.4–3.8 µV residual in synthetic tests); such stretches are what the
  envelope mask, the low-frequency flag and the QC `mains_share_clean` report.

> **Paper wording.** Power-line interference was removed per channel in consecutive 1-s blocks
> by weighted least-squares fitting of sinusoids at the line frequency and its harmonics up to
> 450 Hz at the actual sample times, with an intercept and a linear trend as nuisance terms; the
> harmonic amplitudes could vary within a block as Legendre polynomials of order 0–3 chosen by
> BIC, and EMG bursts were down-weighted by the inverse squared local residual RMS (two
> iterations). Only the fitted sinusoids were subtracted.

### 2.3 Band-pass

*What and why.* Limits the signal to the sEMG band; the 20-Hz high-pass is the standard remedy
for movement artifact (De Luca et al. 2010, J Biomech 43:1573).

$$
y=\mathcal F^{\leftarrow}\bigl\{\mathcal F^{\rightarrow}\{\tilde x\}\bigr\},\qquad |H_\text{eff}(f)|=|H_\text{BW}(f)|^2,
$$

$H_\text{BW}$ = `butter(4, [lo, hi], 'band')` (a 4th-order Butterworth band-pass: 4 poles per
edge, 8 in total) in second-order sections, applied forward and backward (`sosfiltfilt`, zero
phase, −6 dB at the corners). $lo=\max(20,0.1)$ Hz, $hi=\min(450\ \text{Hz},\,0.98\,f_s/2)$.
Not applied when $hi\le lo$ or the window has $<3(2\cdot\text{order}+1)$ samples.
Parameters: `signals.emg.bandpass_hz` = [20, 450], `filter_order` = 4.
Code: `pipeline.py:bandpass`.

> **Paper wording.** The signal was band-pass filtered between 20 and 450 Hz with a
> fourth-order Butterworth filter applied forward and backward (zero phase).

### 2.4 Artifact masks

*What and why.* Samples that are physically missing, saturated or grossly corrupted are
excluded (masked samples become NaN, never zero or interpolated values), so no statistic uses
invented data (`emg/artifacts.py:artifact_mask`, `signals.emg.mask.*`). Per channel and phase
window, every rule is evaluated:

| Rule | Hits sample $n$ when |
|---|---|
| `rail` | $\lvert x_n\rvert\ge \rho_\text{rail}\cdot\text{LSB}_c\cdot(2^{23}-1)$, $\rho_\text{rail}$ = `rail_frac` = 0.99, per-channel LSB from `/emg` (ADC saturated) |
| `leadoff` | the channel's P or N lead-off comparator bit is set |
| `envelope` | $E^u_n>k_E\,R^\text{env}_c$, $k_E$ = `env_factor` = 3 |
| `lowfreq` | $\ell_n>k_\ell\,\tilde\ell^\text{rest}_c$ **and** $\ell_n>k_r\,E^{lf}_n$, $k_\ell$ = `lf_factor` = 10, $k_r$ = `lf_ratio` = 3 (Amendment A1) |

with

$$
E^u_n=\sqrt{\frac{1}{W_E}\sum_{|j|\le (W_E-1)/2}y_{n+j}^2},\qquad \ell_n=\sqrt{\frac{1}{W_L}\sum_{|j|\le (W_L-1)/2}\bigl(\mathcal B_{1\text{–}20}\{x-\operatorname{med}(x)\}\bigr)_{n+j}^2},\qquad E^{lf}_n=\sqrt{\frac{1}{W_L}\sum_{|j|\le (W_L-1)/2}y_{n+j}^2},
$$

nearest-value padding at the window edges, $W_E$ = odd(`mask.envelope_s` = 0.15 s),
$W_L$ = odd(`mask.lf_window_s` = 0.25 s).

- **Envelope rule scale** $R^\text{env}_c$ (`EmgPrep.envelope_rule_ref`): the channel's
  MVC/RVC-phase `ref_p95_uv` (§2.6) in every phase served by `EmgPrep.phase`, **only where the
  reference contraction holds the channel's maximum** (`ref_source = mvc`, D11); off where a
  task exceeded it (there the rule would cut physiological activity) and while the task maxima
  are determined (§2.7). While the MVC/RVC references are computed: $P_u$, the unmasked p95 of
  $E^u$ in that phase. Off where the scale is non-finite or ≤ 0.
- **Low-frequency level** (`artifacts.py:lowfreq_level`): $\mathcal B_{1\text{–}20}$ =
  `butter(2, mask.lf_band_hz = [1, 20] Hz, 'band')` (2 poles per edge) forward–backward on the
  raw signal minus its per-window median; $\tilde\ell^\text{rest}_c=\operatorname{median}_n\ell_n$
  over the motion take's `rest` (`rest_lf_uv`). Off where $\tilde\ell^\text{rest}_c$ is
  non-finite or ≤ 0, and for windows < 64 samples. $E^{lf}$ is unmasked.

**Roles (D14):** every rule is computed and kept (`MaskResult.hits`, per-rule counts), but only
the rules in $\mathcal R_a$ = `mask.analysis_rules` (default `[rail, leadoff, envelope]`:
physically missing, saturated or grossly corrupted samples) form the **analysis mask**:

$$
m_n=\bigvee_{r\in\mathcal R_a}h^r_n,\qquad M_n=\bigvee_{|j|\le k_d}m_{n+j},\qquad k_d=\operatorname{round}(\texttt{dilate\_s}\cdot f_s)=400\ \text{samples},
$$

i.e. dilated by ±0.1 s within the phase window (cumulative-sum window, `artifacts.dilate_rows`,
identical to a box `binary_dilation`). Per-rule counts (`mask_<rule>_frac`, an01 §3) use the
undilated hits. The low-frequency rule is a **QC flag** (`mask_lowfreq_frac`; `lowfreq_flag`
when it exceeds 10 % of the motion-take samples, i.e. all samples of the six motion phases
including rest, `qc/report.py:15-16, 80-91, 119-131`, an01 §3; marked "flagged (kept)" in the
viewer) and a sensitivity analysis (`biomech sensitivity lowfreq_mask`, an08 §1: the whole
pipeline with `analysis_rules: [rail, leadoff, envelope, lowfreq]`, run with `base:
reporting_population` and `skip_stages: [ml_windows, camerafree]` (study.yaml), every cohort table
compared; and the
per-participant table `emg.mask_sensitivity`, §4). Rationale: movement artifact sits below
20 Hz and the standard remedy is the 20 Hz high-pass of §2.3, plus rejecting clearly corrupted
data; sample-level masking by a low-frequency ratio is not standard, it removed 34 % of the
channel-cycles, and the samples it hit carried no contamination signature (§7). The former
150 µV spike zeroing is gone: it lowered MVC references by up to 9 %.

> **Paper wording.** Samples were masked (excluded, not replaced) when the ADC was saturated
> (≥ 99 % of full scale), when the amplifier's lead-off detection flagged the electrode, or
> when the 150-ms RMS envelope exceeded three times the channel's reference-contraction level
> (only for channels whose reference contraction held their largest activity; where a task
> exceeded it, this rule was not applied); masks were extended by 0.1 s on both sides. Samples with strong 1–20 Hz content (more than
> ten times the resting level and three times the concurrent 20–450 Hz envelope) were flagged
> but kept; excluding them was analysed as a sensitivity analysis.

### 2.5 Envelope and normalization

*What and why.* The amplitude measure is the centred moving RMS over the valid samples only, so
a masked stretch neither enters nor zeroes the envelope (`artifacts.py:masked_envelope`,
`signals.emg.envelope_s` = 0.15 s):

$$
E_n=\sqrt{\frac{\sum_{|j|\le (W_E-1)/2}v_{n+j}\,y_{n+j}^2}{\sum_{|j|\le (W_E-1)/2}v_{n+j}}},\qquad E_n=\text{NaN if }\frac{1}{W_E}\sum_{|j|\le (W_E-1)/2}v_{n+j}<0.5,
$$

$v_n=\lnot M_n\wedge\text{finite}(y_n)$, nearest-value padding. Normalized:

$$
e_n=100\,\frac{E_n}{R_c}\ \ [\%],\qquad e_n\le 100\,\%\text{ by construction (§2.7).}
$$

Code: `prepare.py:EmgPrep._phase_uncached` (fields `env`, `norm` of `PhaseEmg`).

> **Paper wording.** The amplitude envelope was the centred 150-ms moving RMS over the unmasked
> samples (undefined where fewer than half of the window was valid) and was expressed in percent
> of the largest envelope value recorded for the channel.

### 2.6 Reference contractions (MVC take)

*What and why.* Per channel, the reference-contraction level: phase `mvc_closing` for masseter
and temporalis (MVC), `mvc_opening` for digastric (RVC) (`prepare.py:EmgPrep.refs`). In the
reference phase: clean (§2.1–2.3), unmasked envelope $E^u$ (150 ms) and $P_u=Q_{95}(E^u)$, mask
(§2.4, envelope rule against $3P_u$), masked envelope $E$ (§2.5); then

| Column | Definition |
|---|---|
| `ref_p95_uv` | $Q_{95}\{E_n:\ E_n\text{ finite}\}$ (`nanpercentile`, linear) |
| `ref_max1s_uv` (sensitivity) | $\max_i \bar E_{[i,\,i+L)}$, $L=\lfloor f_s\rfloor$ samples (1 s), over windows lying fully inside the phase; $\bar E$ = mean over the window's finite $E_n$, defined if ≥ 50 % valid (`max_window_mean`) |
| `ref_mask_frac` | $\frac{1}{N}\sum_n M_n$ over the phase (dilated analysis mask) |
| `ref_mask_frac_effort` | $\dfrac{\lvert\{n: M_n\}\cap\mathcal E\rvert}{\lvert\mathcal E\rvert}$, $\mathcal E=\{n: E^u_n\ge 0.25\,P_u\}$ (`REF_EFFORT_FRAC`): artifacts in the pauses between efforts do not count (D9); NaN if $\mathcal E=\emptyset$ |
| `mvc_max_uv` | $\max_n E_n$ over the finite $E_n$ of the reference phase; NaN when every sample is masked (formerly $-\infty$, which `fmax` in §2.7 also ignored) |

All are NaN without an MVC take or without the reference phase.

> **Paper wording.** Reference contractions were a maximal clench (masseter, temporalis) and
> mouth opening against a resistance of about 10 N (digastric; a submaximal reference
> contraction). For each channel the 95th percentile and the maximum of the masked envelope in
> the reference phase were stored.

### 2.7 Normalization reference: largest activity recorded

*What and why* (`signals.emg.reference` = `max_recorded`, D11; `EmgPrep.reference`,
`EmgPrep.ref_uv`). A frozen reference-contraction value is valid only if the person contracted
maximally; in the pilot some did not (s12 masseter R: clench maximum 44 µV, median 20 µV, rest
16 µV, while chewing reached 95 µV → 270 % of the frozen reference). The reference is therefore
the largest activity recorded for the channel:

$$
R_c=\operatorname{fmax}\bigl(\texttt{mvc\_max\_uv},\ \texttt{task\_max\_uv}\bigr),\qquad \texttt{task\_max\_uv}=\max_{\phi\in\Phi_T}\max_n E^{(\phi)}_n,
$$

$\Phi_T$ = `REFERENCE_TASKS` = `open_close`, `protrusion_retrusion`, `left_right`, `chewing`,
`free_motion` of the motion take; the task envelopes $E^{(\phi)}$ use the analysis rules
**without** the envelope rule (its scale is the reference being determined). fmax ignores a NaN
operand. Hence every $e_n\le 100\,\%$.

- `ref_source` = `mvc` when the reference contraction holds the maximum
  ($\texttt{mvc\_max}$ finite and not exceeded), `task` when a task exceeded it — a failed or
  submaximal reference contraction (flagged, not excluded) — or when there is no reference
  value; `none` when neither exists. `task_max_phase` names the task.
- `mvc_over_task` = $\texttt{mvc\_max\_uv}/\texttt{task\_max\_uv}$.
- Options: `p95` ($R_c$ = `ref_p95_uv`, frozen; kept as a specification-curve variant) and
  `max1s` ($R_c$ = fmax of the largest 1-s mean over the reference phase and the tasks).

> **Paper wording.** Because some participants did not reach their maximum in the reference
> contractions, each channel was normalized to the largest envelope value recorded in the
> reference contraction or in any task; channels whose maximum occurred in a task were
> flagged. Normalization to the 95th percentile of the reference contraction and to the largest
> 1-s mean were reported as alternatives.

### 2.8 Rest statistics (`EmgPrep.rest`, motion take `rest` phase)

$$
\sigma^\text{rest}_c=\sqrt{\frac{1}{\lvert V\rvert}\sum_{n\in V}\bigl(y_n-\bar y_V\bigr)^2},\quad V=\{n:\lnot M_n\},\qquad \tilde E^\text{rest}_c=\operatorname{median}_n E^u_n,\qquad \tilde\ell^\text{rest}_c=\operatorname{median}_n\ell_n .
$$

- `rms_uv` = $\sigma^\text{rest}$ (`rest_rms_uv` in `qc.emg` and `emg.activation`): SD (ddof 0)
  of the cleaned rest signal over the samples **outside** the rest phase's analysis mask (E22;
  formerly over all samples, so a rail or lead-off stretch inflated the noise level), NaN when
  every sample of a channel is masked;
- `env_median_uv` = $\tilde E^\text{rest}$: median of the **unmasked** 150-ms envelope over all
  rest samples (the SNR denominator, an01 §3);
- `lf_uv` = $\tilde\ell^\text{rest}$: median 1–20 Hz level of the raw rest signal (the `lowfreq`
  rule's scale).

> **Paper wording.** Resting noise was the standard deviation of the cleaned signal during the
> rest phase, excluding masked samples.

### 2.9 Centred windows, filter phase, causal chain

**Centred windows (E17/E26).** Every centred moving window of the chain (envelopes $E$ and
$E^u$, the low-frequency level $\ell$ and $E^{lf}$, the robust mains burst envelope $\rho$) has
the odd sample count $\operatorname{odd}(T)=2\lfloor Tf_s/2\rfloor+1$
(`biomech.timebase.odd_window`; jtsmm twin `jtsmm.emg.onset.odd_window`; an exact even product
is raised by one, e.g. 0.15 s at 4 kHz → 601 samples). An even-length centred mean is
asymmetric and shifted its output by half a sample.

**Filter-phase policy.** The offline chain (§2.2–2.5) is zero-phase: fitted harmonic
subtraction, forward–backward band-pass, symmetric centred windows. It reads future samples and
is used for all offline outcomes, including the offline onsets (an05). Quantities that stand for
what a real-time controller could do use the causal chain below instead.

**Causal chain (E27, `emg/causal.py:causal_clean`, `EmgPrep.causal(phase)`).** For the causal
onset twins (`t_decision_us`, `causal_delay_ms`, `anticipation_ms`) and the X3 release detector
(an05 §3, §3.2): `jtsmm.viewer.processing.CausalEmgFilter(mains_mode="track")` fed through
`process_timed` (DC block, tracking mains canceller, causal 20 Hz high-pass and 450 Hz low-pass;
docs/ml/ml02 §4), with the band edges of `signals.emg.bandpass_hz` and the grid frequency the
offline chain used for that phase (take value refined per phase, `EmgPrep.phase_line_hz`). The
filter starts `WARMUP_S` = 2 s before the phase start and the warm-up is discarded, as in
`ml.core.signals.causal_emg`. Before filtering, masked raw samples (the phase's analysis mask;
in the warm-up rail hits and non-finite samples) are replaced by the last valid sample of the
channel (`hold_fill`, causal; 0 before any), so an artifact does not ring through the IIR
sections; after filtering the analysis mask is applied (NaN). The rows are those of
`EmgPrep.phase(phase)`, memoized per (take, phase).

## 3. Usability per analysis family (analysis_plan §3.1)

Flags from the `qc.emg` table (definitions and equations in an01 §3): `usable_shape`
(SNR ≥ 6 dB), `usable_amplitude` (also `ref_mask_frac_effort` ≤ 10 %), `rest_noise_limited`
(rest RMS > 20 µV; reporting only), `lowfreq_flag` (> 10 % of the motion-take samples, rest
included, hit by the low-frequency rule, D14; reporting only).

| Family | Data used |
|---|---|
| Rest activity | `rest_rms_uv`: all channels; rest level (% of max recorded): `usable_amplitude` only. `rest_noise_limited` is not applied |
| Activation map, LMM, clench indices | `usable_amplitude` |
| Activity relevance (§4.1) | all channels (descriptive; shown with the SNR) |
| Co-activation, SPM, spectral cohort | `usable_shape` |
| Timing | decided per cycle (an05) |
| Decoding | all channels; only rail and lead-off masked (A3); the causal reference segments use the analysis rules without the envelope rule (D14: rail, lead-off) |

**Channel-cycles.** For kinematic cycle $i$ with EMG samples $I_i$ (§4, cycle profiles):

$$
\phi_{ci}=\frac{1}{\lvert I_i\rvert}\sum_{n\in I_i}M_{nc},\qquad \phi_{ci}>\texttt{stages.emg.cycle\_mask\_max}\ (0.10)\ \Rightarrow\ \text{channel-cycle set to NaN}.
$$

The low-frequency flag does not count; $\phi_{ci}$ is stored as `masked_<muscle>` in
`emg.cycles`. Pilot: 1308 of 1308 channel-cycles of `open_close` and `chewing` kept (pre-D14
masking: 867, 66 %). Below that threshold its masked
samples stay NaN, and `time_normalize` (`kinematics/cycles.py`) **bridges them by linear
interpolation** onto the 101-point grid; masked samples at the start or end of a cycle stay NaN
at the profile ends. So profiles and SPM contain interpolated values over gaps ≤ 10 % of a
cycle. Per-cycle means and the CCI use the samples at the EMG rate and skip NaN (no bridging).

> **Paper wording.** A channel's cycle was excluded when more than 10 % of its samples were
> masked; shorter masked gaps were bridged by linear interpolation during time normalization.

## 4. Outcomes (stage `emg`, cohort `stats_emg`)

Cohort summaries use `describe` (`stats/core.py`; an07 §1): n, mean ± SD (ddof 1),
median, IQR, min/max and a 95 % BCa bootstrap CI (4000 resamples, config seed 42; only for n ≥ 4
with non-zero range; percentile interval where BCa fails or returns NaN bounds, flagged
`ci_percentile`, E25) of the mean or median as stated.

- **Activation map** (`activation`; `emg/analysis.py:_analyse`): per motion experiment phase
  (incl. `rest`) and muscle, over the phase samples at the EMG rate (masked samples NaN,
  skipped):

  $$
  \texttt{median\_pct}=\operatorname{median}_n e_n,\qquad \texttt{p95\_pct}=Q_{95}(e_n),\qquad \texttt{mask\_frac}=\tfrac1N\textstyle\sum_n M_n .
  $$

  Cohort `activation_cohort`: `usable_amplitude` channels, per task × muscle,
  describe(`median_pct`), CI of the median. Arrays per phase on a 100 Hz grid (each grid time
  takes the first EMG sample at or after it; display only): `<phase>__t_us`, `__norm_pct`,
  `__env_uv`, `__masked`.
  *Paper wording:* Task activation was the median and the 95th percentile of the normalized
  envelope over each task.
- **References** (`refs`): per muscle `ref_phase`, `p95_uv`, `max1s_uv`, `mask_frac`, `ref_uv`,
  `ref_source`, `mvc_max_uv`, `task_max_uv`, `task_max_phase`, `mvc_over_task` and the two
  usability flags (§2.6–2.7).
- **Mask sensitivity** (`mask_sensitivity`, D14; `analysis.py:mask_sensitivity`): per task (not
  `rest`) and muscle, under the primary `analysis_rules` (`rules`) and the same rules with
  `lowfreq` switched in or out (`alt_rules`, `EmgPrep.with_mask_rules`, which reuses the cleaned
  phases): `mask_frac` / `mask_frac_alt`, `median_pct` / `median_pct_alt` (each variant
  normalized with its own reference, since the reference depends on the mask), and for
  `open_close` and `chewing` the `ok` cycles with ≥ max(round(`MIN_CYCLE_EMG_S`·$f_s$), 2)
  samples (0.1 s; `n_cycles`) and how many pass `cycle_mask_max` (`kept` / `kept_alt`).
- **LMM** (`activation_lmm`, `stats/core.py:mixed_model`; an07 §6): rows =
  `usable_amplitude`, task ≠ rest, finite `median_pct`; fitted only with ≥ 5 participants:

  $$
  \ln\bigl(\max(\texttt{median\_pct}_{pjm},\,0.1)\bigr)=\mu+\alpha_j+\gamma_m+(\alpha\gamma)_{jm}+u_p+\varepsilon_{pjm},\quad u_p\sim\mathcal N(0,\sigma_u^2),\ \varepsilon\sim\mathcal N(0,\sigma^2),
  $$

  $p$ participant, $j$ task, $m$ muscle (formula `log_pct ~ C(task) * C(muscle)`, random
  intercept per participant). statsmodels `mixedlm`, REML. Optimizers tried in order lbfgs →
  bfgs → powell → nm; a fit is accepted only with a finite log-likelihood and all fixed-effect
  SEs finite and < 10³ (lbfgs had returned llf = +inf with an intercept SE of 4.7·10⁶ flagged as
  converged). Output: term, coef, se, z, p, CI, optimizer, group variance, scale, llf,
  `status = converged`. None accepted → one row `status = not converged` with `formula`,
  `n_participants`, `n_obs`, `note` (the error text) instead of the fixed-effect table (E25;
  formerly the error dropped every `stats_emg` table).
  *Paper wording:* Log-transformed task activation (floored at 0.1 %) was modelled with task,
  muscle and their interaction as fixed effects and a random intercept per participant (REML).
- **Rest activity** (`rest_activity`): per muscle, `rest_rms_uv` (µV, all channels) and
  `rest_pct` (rest `median_pct`, `usable_amplitude` channels only); describe, CI of the median.
- **Specification curve** (`spec`): `median_pct` in `open_close` and `chewing` normalized by
  the max recorded (primary), the frozen MVC/RVC p95 and the largest 1-s mean (same masks and
  envelope; only $R_c$ differs).
- **Clench indices** (`indices`, from the `ref_p95_uv` values of `mvc_closing`, µV):

  $$
  \mathrm{asym}_m=100\,\frac{R_m-L_m}{R_m+L_m}\ [\%]\ (m=\text{masseter, temporalis}),\qquad \mathrm{AcI}=100\,\frac{\bar M-\bar T}{\bar M+\bar T}\ [\%],
  $$

  $\bar M$, $\bar T$ = NaN-skipping means of the right and left masseter / temporalis p95.
  `usable` only if all four closers are `usable_amplitude` **and** their clench holds their
  maximum (`ref_source = mvc`, D11: the indices compare the clench contractions themselves);
  cohort = describe over usable participants, CI of the mean.
  *Paper wording:* During clenching, the asymmetry index of each closer pair was
  100·(R − L)/(R + L) and the activity index 100·(M − T)/(M + T), with M and T the mean
  masseter and temporalis levels.
- **Cycle profiles** (101 points): kinematic cycles of `open_close` and `chewing` with `ok`;
  EMG samples $I_i=\{n: t^\text{start}_i\le t_n\le t^\text{end}_i\}$ with
  $\lvert I_i\rvert\ge\max(\operatorname{round}(0.1\ \text{s}\cdot f_s),2)$ (`MIN_CYCLE_EMG_S`;
  formerly 20 samples, a rate-dependent rule); channel-cycle rule of §3; `time_normalize`
  (linear over the sample index, not the time stamps):

  $$
  \hat e_i(p)=\operatorname{interp}\Bigl(\tfrac{p}{100};\ \bigl\{\tfrac{k}{\lvert I_i\rvert-1},\,e_{n_k}\bigr\}_{k:\,e_{n_k}\text{ finite}}\Bigr),\qquad p=0,\dots,100,
  $$

  NaN outside the first/last finite sample; the opening angle is normalized the same way.
  *Paper wording:* EMG envelopes were time-normalized to 101 points per kinematic cycle.
- **Co-activation** (Winter CCI, per cycle, `cycles`; `analysis.py:cci`): $O_n$ = mean of the
  shape-usable digastrics, $C_n$ = mean of the shape-usable closers (normalized %,
  NaN-skipping across channels);

  $$
  \mathrm{CCI}=200\,\frac{\sum_{n\in S}\min(O_n,C_n)}{\sum_{n\in S}(O_n+C_n)}\ [\%],\qquad S=\{n\in I_i:\ O_n,C_n\text{ finite}\},
  $$

  opening half = samples with $t_n\le t^\text{peak}_i$ (cycle's peak opening), closing half =
  after; both `open_close` and `chewing`. NaN without a usable opener and closer or with a zero
  denominator. Cohort `cci`: per participant the median over cycles, then describe per task (CI
  of the mean).
  *Paper wording:* Co-activation of openers and closers was quantified per opening and closing
  half-cycle with Winter's co-contraction index, 200·Σ min(O, C)/Σ(O + C), on the normalized
  envelopes.
- **SPM** (`spm`; `stats/cohort_emg.py:spm_tests`; spm1d, α = 0.05, two-tailed; a curve with
  any NaN node is dropped; tests and inference in an07 §10):
  - Confirmatory, parametric paired t on subject means (NaN-skipping mean over cycles), n ≥ 5,
    Holm over the two tests on each test's smallest cluster p (1 if no cluster); `h0reject` =
    Holm-adjusted p (`p_holm`) < 0.05, `h0reject_raw` = the test's own (unadjusted) spm1d
    decision (E25):
    - test 1: masseter L vs R over the chewing cycle (both shape-usable);
    - test 2: openers vs closers over the open-close cycle: per cycle the mean of the
      shape-usable openers (closers), mean over cycles, then divided by its own maximum
      (peak-normalized): $\bar e^\text{pk}(p)=\bar e(p)/\max_{p'}\bar e(p')$.
  - S2: one-way ANOVA across participants (cycles as repetitions; participants with ≥ 3
    complete cycles, ≥ 3 participants) on the opening angle (`open_close`) and on masseter_r
    activity (% of max recorded, `chewing`, shape-usable); single-participant nonparametric
    paired t, masseter L vs R in chewing (≥ 6 complete cycles, min(1000, 2ⁿ) permutations,
    seeded per participant, E16, an07 §10), with a count of significant participants.
  - Variance ratio and CMC (an07).
  *Paper wording:* Normalized EMG profiles were compared with SPM paired t-tests on participant
  mean curves (masseter left vs right during chewing; peak-normalized openers vs closers during
  open–close), α = 0.05, with Holm correction across the two tests.

### 4.1 Activity relevance (stage `emg_relevance`, `emg/relevance.py:relevance`, D19)

*What and why.* In the unloaded tasks some muscles stay near rest (the closers during
open-close); a low normalized value there is noise, not activation. Each participant × task ×
muscle cell is classified as **active** or **near rest** — marked in figures and tables, never
dropped. The rule was fixed before any value was computed. Per channel, over the rest-phase
samples $n\in\mathcal V_\text{rest}$ that are finite and outside the analysis mask:

$$
\mu_c=\frac{1}{\lvert\mathcal V_\text{rest}\rvert}\sum_{n\in\mathcal V_\text{rest}}e_n,\qquad s_c=\sqrt{\frac{1}{\lvert\mathcal V_\text{rest}\rvert-1}\sum_{n\in\mathcal V_\text{rest}}(e_n-\mu_c)^2},\qquad \theta_c=\max\bigl(\mu_c+k\,s_c,\ \pi_\text{min}\bigr),
$$

$$
\text{active}_{cj}\iff Q_{95}\bigl(e^{(j)}\bigr)_c\ge\theta_c,
$$

$Q_{95}(e^{(j)})$ = `emg:activation.p95_pct` of task $j$ (not rest). Parameters:
`stages.emg_relevance.k_sd` = $k$ = 3, `min_pct` = $\pi_\text{min}$ = 5 %. Table `relevance`:
`rest_mean_pct`, `rest_sd_pct`, `threshold_pct`, `p95_pct`, `active`, `reason` ("below rest +
k SD", "below 5 %", "no data" when the p95 or the threshold is non-finite). The paper views
(`emg_landscape`, `table_emg_activation`) show per task × muscle the median `p95_pct` over the
participants and the count of participants in whom the muscle is active, and per muscle the
SNR (`qc:emg.snr_db`) and the reference contraction relative to the largest task activity
(`emg:refs.mvc_over_task`).

> **Paper wording.** A muscle was considered active in a task when the 95th percentile of its
> normalized envelope during the task reached both the resting mean plus three standard
> deviations and 5 % of the largest recorded activity; otherwise it was reported as near rest.

## 5. Frequency domain (`biomech/spectral.py`; X1, exploratory)

- **Spectra** (`spectral.py:welch_masked`, called from `analysis.py:_spectra`): per motion
  experiment phase and MVC phase, cleaned EMG $y$. Segments of $L=\operatorname{round}(0.5\ \text{s}\cdot f_s)$
  = 2000 samples, hop $L/2$, starting at sample 0; a segment $s$ is used for a channel only if
  none of its samples is masked ($M$) or non-finite. Per segment, with Hann window $w$:

  $$
  P_s(f_j)=\gamma_j\,\frac{\bigl|\sum_{l=0}^{L-1}w_l\,(y_{s+l}-\bar y_s)\,e^{-i2\pi jl/L}\bigr|^2}{f_s\sum_l w_l^2},\qquad \gamma_j=\begin{cases}1&j=0\text{ or }j=L/2\\2&\text{else}\end{cases},\qquad P(f_j)=\frac{1}{\lvert\mathcal S\rvert}\sum_{s\in\mathcal S}P_s(f_j),
  $$

  $\mathcal S$ = clean segments ($P$ = NaN without any), resolution $f_s/L$ = 2 Hz. Stored for
  10–500 Hz (`emg.spectral`, arrays `spectral__<phase>__psd`, `spectral__freq`).
- **Descriptors** (`mean_median_freq`, `band_shares`; bins $\mathcal F=\{f_j\in[20,450]\ \text{Hz}\}$):

  $$
  \mathrm{MNF}=\frac{\sum_{\mathcal F}f_j\,P(f_j)}{\sum_{\mathcal F}P(f_j)},\qquad \mathrm{MDF}=\min\Bigl\{f_k\in\mathcal F:\ \frac{\sum_{f_j\le f_k}P(f_j)}{\sum_{\mathcal F}P(f_j)}\ge 0.5\Bigr\}\ (\text{no interpolation}),
  $$

  $$
  \texttt{share\_<lo>\_<hi>}=\frac{\sum_{lo\le f_j<hi}P(f_j)}{\sum_{\mathcal F}P(f_j)}\quad\text{for }[20,50),[50,100),[100,200),[200,450)\ \text{Hz}
  $$

  (half-open, so the 450 Hz bin is only in the denominator); `mnf_sd_within_hz` = SD (ddof 1)
  of the per-segment MNF (MNF of each $P_s$), `n_windows` = clean segments. NaN where the
  in-band power is zero or NaN.
- **Cohort** (`stats_emg.spectral_cohort`, `cohort_emg.py:spectral_tables`; shape-usable
  channels, per task × muscle, over participants $p$):

  $$
  \mathrm{SD}_\text{between}=\operatorname{SD}_{p}(\mathrm{MNF}_p)\ (\text{ddof }1),\qquad \mathrm{SD}_\text{within}=\sqrt{\operatorname{mean}_p\,\texttt{mnf\_sd\_within}_p^2},\qquad \texttt{between\_over\_within}=\frac{\mathrm{SD}_\text{between}}{\mathrm{SD}_\text{within}}
  $$

  plus `mnf_mean_hz`, `mdf_mean_hz`. Spectral shape depends on electrode placement,
  inter-electrode distance and tissue as much as on the muscle, which argues for standardized
  electrode protocols (S2).
- **SPM along frequency** (`stats_emg.spectral_spm`) on the per-segment spectral shape of
  `open_close`, `chewing` and `mvc_closing` (arrays `spectral__<phase>__win`,
  `analysis.py:coarse_log_shape`): 44 bins $b$ of 10 Hz from 20 to 460 Hz, each $[lo,lo+10)$
  (`COARSE_EDGES`); per segment

  $$
  \bar P_{s,b}=\operatorname{mean}_{f_j\in b}P_s(f_j),\qquad S_{s,b}=\log_{10}\frac{\bar P_{s,b}}{\sum_{b'}\bar P_{s,b'}} ,
  $$

  participant curve = NaN-skipping mean of $S_{s,\cdot}$ over the segments. Tests: masseter L
  vs R in chewing (paired t on participant means, n ≥ 5) and the participants ANOVA on
  masseter_r in chewing (complete segments as repetitions, ≥ 3 per participant: where in
  frequency participants differ). Clusters are reported in Hz at bin centres (25…455 Hz) and
  drawn as gray bars above the curves.
- **Mains-removal spectra and chain examples** (`emg.analysis.chain_products`, for the data
  views `data_emg_spectra` and `data_emg_chain`; display only): per experiment phase the
  Welch PSD (`scipy.signal.welch`: Hann, 1-s segments, 50 % overlap, constant detrend; ≤ 500 Hz)
  of the raw EMG (median removed) and after mains removal at the phase's grid frequency
  (`mains__<phase>__raw_psd`, `__demained_psd`, `__line_hz`), and for `rest`, `open_close` and
  `chewing` a 2-s example of every channel at each step (raw, mains removed, band-passed, masked
  envelope in % of max recorded, analysis mask; `chain__<phase>__*`), starting 0.4 s before the
  first movement onset of the phase, or 1 s into the phase when it has no cycles (rest;
  `analysis.py:chain_window`). The "mains removed" step here is recomputed with a
  constant-amplitude model per block (`amp_order` 0, robust weights as configured), not the
  BIC-selected order of §2.2; the band-passed step is the analysis signal $y$.

> **Paper wording.** Power spectra were estimated with Welch's method (0.5-s Hann windows, 50 %
> overlap) using only windows without masked samples; mean and median frequency and relative
> band powers were computed between 20 and 450 Hz. Between-participant variability of the mean
> frequency was compared with the within-participant variability across windows.

## 6. Hard-coded parameters

| Parameter | Value | Where |
|---|---|---|
| Mains block length | 1 s (`DEFAULT_BLOCK_S`) | `jtsmm/emg/mains.py` |
| Max harmonics; band edge + margin | 9 (`DEFAULT_HARMONICS`); 450 + 2 Hz (`EMG_BAND`, `HARMONIC_MARGIN_HZ`) | `mains.py` |
| Robust envelope window | 0.05 s (`robust_env_s`) | `mains.py` `remove_mains` |
| Robust floor quantile; passes | 10th percentile (`floor_q`); 2 (`n_iter`) | `mains.py` `_robust_weights`, `_reweighted_fit` |
| Max Legendre order for `auto` | 3 (`AUTO_MAX_ORDER`) | `mains.py` |
| Line-frequency peak search; FFT length | ±2 Hz; ≤ 120 s (`max_seconds`) | `mains.py` `estimate_line_hz` |
| Tone acceptance | `min_snr` 5; neighbours 3…8 bins | `mains.py` `measure_line_hz` |
| Max deviation | 0.3 Hz from nominal (take); 0.1 Hz from take value (phase) | `measure_line_hz`; `biomech/emg/pipeline.py` `refine_line_hz` |
| Mains-share half-width; band | ±1.5 Hz (`_SHARE_HALF_HZ`); 20–450 Hz | `mains.py` `mains_share` |
| Band-pass upper cap | 0.98·fs/2 | `pipeline.py` `bandpass` |
| Low-frequency filter order | 2 | `biomech/emg/artifacts.py` `lowfreq_level` |
| Envelope / max-1-s minimum valid share | 0.5 | `artifacts.py` `masked_envelope`; `prepare.py` `max_window_mean` |
| Effort threshold for reference usability | 0.25 × p95 (`REF_EFFORT_FRAC`) | `biomech/emg/prepare.py` |
| Tasks that can set the reference | open_close, protrusion_retrusion, left_right, chewing, free_motion (`REFERENCE_TASKS`) | `biomech/emg/prepare.py` |
| Mask-validation window | 0.25 s (`MASK_WIN_S`, ≥ 16 samples) | `biomech/emg/analysis.py` |
| Minimum EMG per cycle | 0.1 s (`MIN_CYCLE_EMG_S`, ≥ 2 samples), for the profiles and for `mask_sensitivity` | `emg/analysis.py` |
| Causal-chain warm-up | 2 s (`WARMUP_S`) | `biomech/emg/causal.py` |
| Welch segment; overlap | 0.5 s; 50 % | `analysis.py` `_spectra`; `spectral.py` `welch_masked` |
| Coarse spectral bins | 20–460 Hz, 10 Hz (`COARSE_EDGES`) | `analysis.py` |
| Activation arrays grid | 100 Hz (`GRID_HZ`) | `analysis.py` `to_grid` |
| Chain example | 2 s (`CHAIN_WINDOW_S`) from 0.4 s (`CHAIN_LEAD_S`) before the first onset | `analysis.py` |
| LMM floor; minimum participants | 0.1 %; 5 | `stats/cohort_emg.py` `activation_tables` |

## 7. Low-frequency mask validation (`mask_validation`, exploratory)

Does the A1 low-frequency rule hit samples whose band-passed EMG is still contaminated?
(`analysis.py:mask_validation`, cohort `cohort_emg.py:mask_validation_table`.) Per channel and
task (not `rest`; only where the rule hits at all), consecutive non-overlapping windows of
$n_w=\max(\lfloor 0.25\,f_s\rfloor,16)$ = 1000 samples of the cleaned signal from the phase start
(a trailing remainder is dropped; ≥ 4 windows needed): *masked* windows have > 50 % of their
samples hit by the low-frequency rule, *clean* windows no sample hit by **any** rule, analysis or
flag-only, and none inside the dilated (±0.1 s) analysis mask (`MaskResult.any_hit` =
undilated hits ∪ analysis mask, `emg/artifacts.py:84-89`; since D14 the low-frequency hits are
not in the analysis mask, so "not masked" would no longer mean clean). A channel-task needs ≥ 2 masked and ≥ 4 clean
windows. Per window $w$, with $P_w$ the Hann-windowed periodogram of the mean-removed window:

$$
\mathrm{sh}_w=\frac{\sum_{20\le f_j<40}P_w(f_j)}{\sum_{20\le f_j\le 450}P_w(f_j)},\qquad \mathrm{MNF}_w=\frac{\sum_{20\le f_j\le 450}f_j\,P_w(f_j)}{\sum_{20\le f_j\le 450}P_w(f_j)} .
$$

Pre-stated rule: the rule is *conservative* for a channel-task when
$\operatorname{median}_\text{masked}(\mathrm{sh})\in[Q_{25},Q_{75}]_\text{clean}(\mathrm{sh})$ **and**
$\operatorname{median}_\text{masked}(\mathrm{MNF})\in[Q_{25},Q_{75}]_\text{clean}(\mathrm{MNF})$.
Cohort table `stats_emg.mask_validation`: overall (`muscle = all`) and per muscle the
conservative share and the median masked − clean differences (of the per-channel-task
medians). The comparison is confounded by activity (masked windows fall in movement, clean
ones include holds).

Pilot: 127 channel-tasks, 52 %
conservative, median differences +0.004 (20–40 Hz share) and +3.5 Hz (MNF): no contamination
signature. The check behind D11/D14 (pre-D14 run) gave 126, 50 %, +0.005 and +4 Hz. Together with
the share of channel-cycles the rule removed, this is why the rule only flags since D14 (§2.4).
