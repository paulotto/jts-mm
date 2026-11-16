<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an07 — Statistics

Code: `biomech/stats/` (`core.py`, `families.py`, `waveforms.py`, `requirements.py`,
`cohort_*.py`), stages `stats_*`, `requirements`, `variability` and `camerafree_cohort`
(`biomech/stages.py`, `imu/stage_camerafree.py`); the decoding tests in the `ml` plugin
(`ml/biomech_stages.py`, `ml/protocols/`); the agreement statistics of the systems study in
`biomech/validation/` (an14); the impedance associations in `biomech/qc/impedance.py`. This
document is the reference for **every statistical procedure** of the code base: the other
analysis docs link here for the formulas. The unit of analysis is the participant. Symbols
are listed in the Notation table below; the Index maps each method to its section.

**Abbreviations.** BCa = bias-corrected and accelerated (bootstrap); CI = confidence interval;
SD = standard deviation; RMS(E) = root mean square (error); IQR = interquartile range; LoA =
limits of agreement; MOVER = method of variance estimates recovery; ICC = intraclass
correlation coefficient; SEM = standard error of measurement; MDC = minimal detectable change;
CCC = concordance correlation coefficient; SPM = statistical parametric mapping; RFT = random
field theory; CMC = coefficient of multiple correlation; LMM = linear mixed model; REML =
restricted maximum likelihood; HAC = heteroskedasticity and autocorrelation consistent; KDE =
kernel density estimate; IUT = intersection-union test; LOSO = leave one subject out; EMG =
electromyography; IMU = inertial measurement unit; MVC / RVC = maximal / reference voluntary
contraction; TKEO = Teager–Kaiser energy operator; TMJ = temporomandibular joint; SNR =
signal-to-noise ratio; QC = quality control; JTS-MM = the multimodal jaw tracking system of
this project.

**Software** (analysis environment `software/jtsmm/.venv`): SciPy 1.18.1, NumPy 2.5.1,
statsmodels 0.15.0, pingouin 0.7.0, spm1d 0.4.54. Where the result depends on a library
default, the default is named.

**Seeds.** Cohort bootstraps and permutation tests use the study `seed` (`seed: 42` in
`etc/biomech.yaml` and both `experiments/*/study.yaml`) through `ctx.config.seed`, with a
fresh `numpy.random.default_rng(seed)` per call (so every table is reproducible and does not
depend on the order of the calls). Exceptions:

| Procedure | Seed |
|---|---|
| participant stages (Monte Carlo nulls, observed-residual CI; an06) | `participant_seed(seed, pid)` = `crc32("<seed>:<pid>") & 0x7FFFFFFF` (`pipeline.py`), plus a fixed offset per null |
| single-participant SPM permutation test | `crc32("masseter_L_vs_R\|<pid>")` (independent of the study seed) |
| C2-E and chance-null surrogates (an05) | `surrogates.null_seed(seed, key)` = `default_rng([seed & 0x7FFFFFFF, crc32(key)])` |
| cycles needed (`stats.waveforms.cycles_needed`) | the study seed, passed by `cycles_needed_table` (formerly the function default 0); the same stream for every participant and outcome |
| label check (`labels/kappa.py`) | `stages.labels_check.seed`, default 20261029 |
| C1 misalignment requirement rows (`misalign_hinge_fit_mm`, `misalign_hinge_tmj_mm`; BCa of the median) | study seed + 6 (`stats/requirements.py:403-405`) |
| impedance association CIs (`qc/impedance.py:associations`, §16.1) | one `default_rng(seed)` shared by the six relations in `RELATIONS` order (`impedance.py:207-223`): a relation's CI depends on the relations before it (a skipped relation shifts the stream) |
| IMU in-take calibration starts (`imu/calibration.py`, Park–Martin pairs and random rotations; an04 §3) | **0** (function default) |
| Qualisys reference uncertainty Monte Carlo (`validation/quality.py:sigma_ref`; an14 §5.4) | **0** (function default) |

## Notation

| Symbol | Meaning |
|---|---|
| $i = 1,\dots,n$ | participants entering a test (finite values only; $n$ is counted after dropping NaN) |
| $x_i$, $d_i$ | per-participant value; per-participant paired difference ($a_i - b_i$) |
| $\bar x$, $s$ | mean; SD with divisor $n-1$ (`ddof = 1`) unless "1/$n$ moments" is stated |
| $\tilde x$, $Q_q(\cdot)$ | median; sample $q$-quantile by linear interpolation (`numpy.percentile` / `scipy.stats.quantile`, method `linear`: position $q(n-1)$ in the sorted 0-based sample) |
| $\hat\theta$, $\hat\theta^*_b$, $\hat\theta_{(-i)}$ | statistic on the data; on bootstrap resample $b = 1,\dots,B$; on the data without observation $i$ (jackknife) |
| $\Phi$, $z_q$ | standard normal CDF; $z_q = \Phi^{-1}(q)$ |
| $t_{q,\nu}$, $\chi^2_{q,\nu}$ | $q$-quantile of Student's $t$ / chi-square with $\nu$ degrees of freedom |
| $R_i$, $T^+$ | rank of $\lvert d_i\rvert$ among the non-zero $\lvert d\rvert$ (average ranks for ties); $T^+ = \sum_{d_i>0} R_i$ |
| $W_{(1)} \le \dots \le W_{(M)}$ | ordered Walsh averages $(d_i + d_j)/2$, $i \le j$, $M = n(n+1)/2$ |
| $N$, $k$ | Monte-Carlo draws (simulations, surrogates, permutations); draws at least as extreme as the observed |
| $p_{(1)} \le \dots \le p_{(m)}$ | ordered p values of a family of $m$ finite p values |
| $\alpha$ | 0.05 (two-sided intervals at 95 %) |
| $X_{(1)} \le \dots \le X_{(n)}$, $Q_p$ | order statistics; population $p$-quantile |
| $\mathbb{1}[\cdot]$, $\#\{\cdot\}$ | indicator; count |

## Index: method → section → where it is used

| Method | Section | Claims / outcomes |
|---|---|---|
| Descriptives, BCa bootstrap CI of the mean or median | §1 | S1 reference values; every cohort median (C1 ratios, C2 leads, X2 / X3, C2-E locations, camera-free, IMU agreement, requirement performance rows, decoding summary) |
| Percentile, cluster, block and moving-block bootstraps | §1.3 | Hedges *g*; Bland–Altman CIs (C3a, camera-free); max \|LoA\| sensitivity; cycles needed (S2); observed-residual CI (C1, an06); systems agreement; impedance ρ |
| Wilcoxon signed-rank, one sample / paired | §2.1 | C2 confirmatory and exploratory leads; C2 test against the chance null; X2, X3 (and X3 vs chance); C1 cohort log-ratio test; C2-E H1–H3 and supporting rows; camera-free contrasts; impedance second − first take; decoding feature-set and few-shot tests |
| Effect size $r = z/\sqrt{n}$ | §2.2 | C2, X2, X3 rows (`r`, `r_null`, `r_vs_chance`) |
| Hodges–Lehmann estimate with exact CI | §2.3 | camera-free ladder contrasts, X5b / X5c contrasts |
| Monte-Carlo p, sign-flip permutation | §3 | C1 per participant (`p_mc_*`); C2 chance null (`p_null_perm`); C2-E surrogate p values (an05); decoding label-shift permutation |
| Holm | §4.1 | C2 confirmatory (four tests) and exploratory families; confirmatory SPM (two tests); decoding ablations and few-shot; camera-free families; C2-E secondary and supporting blocks; impedance families |
| Intersection-union test, serial gatekeeping | §4.2 | C2-E (H1 → H2, H3) |
| Log-ratio and location tests | §4.3 | C1 cohort test (log observed / P99); C2-E locking and location |
| Hedges *g*, Spearman ρ, leave-one-out influence | §5 | S1 (sex, age, older participants; exploratory); decoding accuracy vs age |
| Linear mixed model | §6 | EMG activation by task × muscle; camera-free drift |
| Friedman test | §6 | decoding feature-set ablations |
| ICC(3,1), SEM, MDC95 | §8 | reliability (S2); target of cycles needed |
| Bland–Altman (simple, repeated measures, cluster bootstrap, MOVER) | §9 | C3a IMU opening; camera-free cycle amplitudes; requirement rows of the IMU opening |
| Within-participant Bland–Altman (effective-n *t*, moving block), equivalence margin, Lin's CCC, ICC(A,1) | §17 | systems study cycle agreement (an14 §7) |
| SPM paired *t* (random field theory), permutation SPM, SPM ANOVA, Welch SPM; CMC, variance ratio | §10 | confirmatory EMG waveforms; single-participant masseter test; S2 variability; spectral SPM (X1); systems waveforms and EMG profiles |
| Specification curve, cycles-needed bootstrap | §11 | S2; G′5 (an06) |
| Literature *z* | §12 | S1 norms |
| Harrell–Davis P95, order-statistic bounds | §13 | requirement capacity rows (E31), including the post hoc X7 envelope rows (closing and free-motion velocity, peak acceleration, posterior excursion) |
| Coverage of the sample maximum, $1 - q^n$ (no data) | §13 | X8 planning table `requirements:max_coverage` (post hoc) |
| Order statistics of the null (P99), Monte-Carlo p, signed-rank of log ratios | §14, an06 | C1 against the noise nulls |
| Verdict against a bound, OLS drift slope | §15 | camera-free (X5) |
| Rank association with fixed effects | §16.1 | impedance check vs recorded signal (D21) |
| IQR locking, truncated-normal mixture, Theil–Sen slope | §16.2 | C2-E per participant (an05 §3.3) |
| Cohen's κ | §16.3 | label validity (C3b ground truth) |
| Theil–Sen slope, leave-one-subject-out median curve (description, no test) | §16.4 | X9 coupling summary (post hoc) |
| Newey–West HAC variance, weighted LS, jackknife | §17.4 | systems clock (an14 §3) |
| n = 2 descriptive rule | §17.5 | every systems-study result |
| Circular statistics | — | none: "circular shift" in an05 is a surrogate (the EMG shifted against the motion), not a statistic on angles |
| Binomial sign test | — | none: the only binomial computations are the order-statistic bounds (§13) |

## Reporting population (D35)

- **Analysis of record:** the primary store, all 14 participants (n = 14), unchanged. The participant
  recorded outside the protocol stays in the analysis (flag, don't exclude). No paper item reads it
  (D36).
- **Dataset paper:** every artefact reports the **13 adults**. The paper set names the
  `reporting_population` sensitivity variant (`variant: reporting_population` in `paper_dataset.yaml`; an13), whose
  store recomputes every stage and every cohort statistic without that participant (medians, CIs, tests,
  Holm, the C1 counts, requirements, decoding LOSO folds; the variant skips no stage). It is
  frozen as its own snapshot (`biomech snapshot --variant reporting_population <name>`; current: `r17a`, pinned in
  `paper_dataset.yaml`; the analysis of record is `r17`, n = 14).
- **Sensitivity variants (D36):** every variant a paper quotes has `base: reporting_population`
  (`experiments/icorr2027/study.yaml`): its cohort filter and overrides are the base's plus its
  own, its cache is seeded from the `reporting_population` store and it is compared with that 13-adult
  store (`sensitivity.py:variant_config`, `_changes`, `base_config`). The robustness table's
  primary column is the 13 adults; no n = 14 value appears in the papers.
- **IMU-based outcomes** are identical with and without that participant (no jaw IMU, so no IMU, timing,
  C2-E or camera-free unit), except where a generic leave-one-subject-out model also trains on
  that participant's optical data (the C3a generic opening model and the camera-free E4gen); in the
  13-adult store that participant is in no training set, so the papers are not affected.
- **Systems paper:** unaffected (two adults, P1 and P2; n = 2, per participant plus the mean,
  EV8; §17.5).
- **Checks:** `check-outputs` bounds every participant count of a paper set by the cohort of
  the store it reads (dataset paper: 13; an12).

## Evaluation stance

- **A near-miss is judged by its sampling error, not by the digit.** A criterion estimated
  from $N$ simulations or resamples (a coverage, a type I error rate, a Monte-Carlo p) is a
  binomial proportion $\hat p$ with standard error

  $$
  \mathrm{SE}(\hat p) = \sqrt{\hat p(1-\hat p)/N}.
  $$

  A miss within about one SE is not read as a failure until the estimate has been made
  precise enough to decide: more replicates with the same method and parameters, never a
  changed method. Example (X6, G2 / G4): a cohort S2 type I error of 0.10 over 30 cohorts and
  a coverage of 0.85 over 40 cohorts were re-run with ≥ 200 cohorts (0.095; 0.888–0.933 at
  SE ≈ 0.02) before the decisions recorded in the change log.
- **The decision follows the precise estimate and is recorded before the real-data run**:
  either the criterion passes, or the deviation is documented with its direction (conservative
  or not) and what it changes in the reading (e.g. the S2 cohort p withheld; the mixture kept
  with δ = 17 ms).
- **Monte-Carlo p values carry their resolution:** $(k+1)/(N+1)$ is never 0; with
  $N = 1000$ (C1) or 200 (C2-E surrogates) the smallest attainable value is 1/1001 or 1/201
  (§3).
- **Observed values are read with their interval.** A cohort median whose BCa CI includes the
  bound or margin is reported as "compatible" or "inconclusive", not as passed or failed
  (camera-free verdicts §15, C2-E decision rule §4.2).

## Families and multiplicity (overview)

| Family | Tests | Multiplicity | Doc |
|---|---|---|---|
| C1 (pre-specified) | per participant: observed > P99 of the in-vivo null; cohort: "k of n" | none (one decision rule); the D24 / D25 / D33 variants and the signed-rank test of log(observed / P99) are reported next to it, unadjusted | §14 below, an06 |
| C2 confirmatory (D15) | one-sided Wilcoxon of the opening lead, TKEO, rules `first` and `drive`, tasks open_close and chewing | Holm over the four | §7 below, an05 §3 |
| C2 chance null (D23) | per-participant observed − null: one-sided Wilcoxon and sign-flip permutation | Holm within the same families as the frozen tests | an05 §3.0; §3.2 below |
| C2 exploratory | every other tested timing row | Holm within its metric | §7 below |
| C2-E (X6, post hoc) | H1 (IUT of locking and location) → H2, H3 (IUT) | **serial gatekeeping**: H1 at α = 0.05; H2 / H3 Holm only if H1 rejected; supporting block Holm within itself; structure rows descriptive (cohort p withheld); post hoc sensitivity `p_holm_family` = Holm over H1–H3 | an05 §3.3; §4.2 below |
| X3 vs chance | detected share vs chance (descriptive); two-sided paired signed-rank of anticipation vs chance | none | an05 §3.2 |
| Confirmatory SPM | masseter L vs R (chewing), openers vs closers (open_close) | Holm over the two (minimum cluster p) | §10 |
| Decoding ablations | Friedman, then two-sided Wilcoxon of EMG + IMU against each other set; few-shot EMG + IMU vs IMU per calibration amount | Holm over the pairwise tests; Holm over the amounts k | §6, an05 §5 |
| Camera-free (X5) | paired ladder contrasts: rotation (E2a−E1, E2b−E2a, E3−E2a), translation (E4own−E4gen, E4gen−E4hinge, E5−E4own, E6−E4own), H2 drift; X5b, X5c | Holm within each family | §15 |
| Impedance (D21, E30) | second − first take per participant (Wilcoxon over participants); associations | Holm within `check_vs_recorded` and `check_vs_check` separately | §16.1, an03 |

**Shared building blocks** (`stats/families.py`, one test path for C1, C2-E, the camera-free
contrasts and the impedance test): `wilcoxon_signed` (signed-rank, exact null distribution
when there are no zeros or ties, else the normal approximation; NaN below the minimum n),
`wilcoxon_greater` (its one-sided "greater" form, D24), `log_ratio_test` (per-participant log
ratios with the BCa CI of their median), `location_test` (per-participant locations against a
margin: BCa CI of the median and the signed-rank test of value − margin), `iut`
(intersection-union p = the largest part p, NaN if any part is NaN), `gatekeeping` (secondary
Holm only when the primary is rejected, else NaN), `block_holm`, and `paired_contrast` with
`hodges_lehmann` and `signed_rank_null` (the camera-free ladder contrasts, §2.3; formerly a
routine of its own in `stats/cohort_camerafree.py`). The C2 timing tests (`stats/cohort_misc.py`)
use `stats.core.wilcoxon_one_sample`, which leaves the choice of the exact or approximate p to
SciPy (§2.1).

## 1. Descriptives and bootstrap intervals

### 1.1 `describe` (`stats/core.py:describe`)

**What and why.** One summary per participant set: location, spread and a 95 % interval of
the cohort mean or median, without a normality assumption.

On the finite values $x_1,\dots,x_n$:
- $n$, mean $\bar x$, SD $s = \sqrt{\sum_i (x_i - \bar x)^2/(n-1)}$ (NaN for $n = 1$), median
  $\tilde x$, quartiles $q_1 = Q_{0.25}$, $q_3 = Q_{0.75}$ (linear interpolation), min, max;
- a 95 % **BCa** bootstrap CI of the statistic `stat` (mean or median; §1.2), $B = 4000$
  resamples (`n_boot`), `scipy.stats.bootstrap(method="BCa")` with
  `default_rng(seed)`;
- the CI only for $n \ge 4$ and non-constant values ($\max - \min > 0$); otherwise NaN;
- the **percentile** bootstrap (same resamples: a new generator with the same seed) is the
  fallback when BCa raises an exception **or returns NaN bounds** (E25: SciPy returns NaN with a
  suppressed `DegenerateDataWarning` when $\hat z_0$ or $\hat a$ cannot be computed, e.g. a
  median whose bootstrap distribution is mostly one value, so that $\hat z_0 = \pm\infty$; that
  NaN used to be reported as the CI). `ci_percentile` (boolean) marks the rows that use it.

**Which statistic the CI bootstraps:**

| `stat` | Tables |
|---|---|
| mean | `stats_kinematics.reference_values` (S1), `stats_emg.indices`, `stats_emg.cci` |
| median | `stats_emg.activation_cohort`, `rest_activity`; `stats_kinematics.hinge_summary`, `hinge_noise_summary`, `hinge_noise_tests`, `fha_summary`, `condyle_sensitivity`; `stats_timing.timing_cohort`, `ensemble_cohort`, `release_cohort`, `c2e_cohort`; `stats_imu.imu_agreement_cohort`; `camerafree_cohort`; `requirements` (performance rows); `ml_eval.summary` |

The decoding variants (`ml/protocols/variants.py:summarize_variants`) report median [IQR] only
(no CI).

**Paper wording.** "Cohort values are reported as median [interquartile range] (or mean ± SD)
across participants, with 95 % bias-corrected and accelerated (BCa) bootstrap confidence
intervals (4000 resamples; Efron 1987); where the BCa interval was undefined, the percentile
interval is given."

### 1.2 BCa interval (SciPy's implementation, Efron & Tibshirani 1993, §14.3 and Eq. 15.36)

With $B$ resamples drawn with replacement from $x_1,\dots,x_n$ and $\hat\theta^*_b$ the
statistic on resample $b$:

$$
\hat z_0 = \Phi^{-1}\!\left(\frac{\#\{b:\hat\theta^*_b < \hat\theta\} + \#\{b:\hat\theta^*_b \le \hat\theta\}}{2B}\right)
$$

(the mid-rank of $\hat\theta$ among the replicates), the acceleration from the leave-one-out
jackknife, $\bar\theta_{(\cdot)} = \frac1n\sum_i \hat\theta_{(-i)}$:

$$
\hat a = \frac{\sum_{i=1}^n \left(\bar\theta_{(\cdot)} - \hat\theta_{(-i)}\right)^3}
              {6\left[\sum_{i=1}^n \left(\bar\theta_{(\cdot)} - \hat\theta_{(-i)}\right)^2\right]^{3/2}}
$$

(SciPy computes it from $U_i = (n-1)(\bar\theta_{(\cdot)} - \hat\theta_{(-i)})$; the factors
$(n-1)^3/n^3$ cancel), and the adjusted levels

$$
\alpha_1 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{\alpha/2}}{1 - \hat a(\hat z_0 + z_{\alpha/2})}\right),\qquad
\alpha_2 = \Phi\!\left(\hat z_0 + \frac{\hat z_0 + z_{1-\alpha/2}}{1 - \hat a(\hat z_0 + z_{1-\alpha/2})}\right).
$$

The interval is $[Q_{\alpha_1}(\hat\theta^*), Q_{\alpha_2}(\hat\theta^*)]$ (linear
interpolation). The **percentile** interval is the special case $\hat z_0 = \hat a = 0$:
$[Q_{0.025}(\hat\theta^*), Q_{0.975}(\hat\theta^*)]$.

*Assumptions:* the participants are exchangeable draws from the population; with $n$ = 13
the interval of a median is coarse (it moves in steps between order statistics), which is why
results near a bound are read as "compatible" rather than decided (Evaluation stance).

### 1.3 Other bootstraps in the code

All are percentile intervals $[Q_{0.025}, Q_{0.975}]$ of the replicates unless stated.

| Bootstrap | What is resampled | $B$ | Code | Used for |
|---|---|---|---|---|
| independent groups | each group with replacement, separately | 4000 | `stats/core.py:hedges_g` | CI of Hedges *g* (§5) |
| participant cluster | participants with replacement, **all their cycles kept**; a participant drawn twice enters as two participants | 2000 | `stats/core.py:bland_altman_rm_ci`, `stats/cohort_misc.py:max_abs_loa_ci` | CIs of bias, LoA, proportional bias (§9); max \|LoA\| sensitivity |
| participant cluster (rank correlation) | participants with replacement, all their channels | 2000 | `qc/impedance.py:associations` | CI of the pooled Spearman ρ (§16.1) |
| cycles within a participant | cycles with replacement, $n$ of the available | 2000 per $n$ | `stats/waveforms.py:cycles_needed` | cycles needed (§11) |
| openings × landmark perturbation | openings with replacement, landmarks perturbed | 100 | `uncertainty/montecarlo.py:observed_ci` | CI of the observed hinge residual (an06 §3) |
| cycles (systems) | cycles with replacement | 2000 | `validation/agreement.py:ba_boot`, `run_ref_agreement` | LoA CIs; bias sensitivity; waveform band sensitivity `equiv_margin_boot` (§17) |
| moving block (systems) | overlapping blocks of 2 successive cycles (Künsch 1989) | 2000 | `validation/agreement.py:moving_block_means` | bias sensitivity (§17.1) |
| time blocks (systems) | 1-s blocks of frames | 1000 | `validation/evaluate.py:_block_ci`, `validation/stages_camerafree.py:_block_ci` | CIs of frame-level mean differences (fusion, camera-free check; an14) |
| time blocks (lever arm) | blocks of `block_s` of frames, OLS refitted | `n_boot` | `imu/lever.py` | lever-arm CI (an04, an14 VD15) |
| cycles (ensemble events) | cycles with replacement, median curve re-evaluated | `n_boot` | `timing/ensemble.py:ensemble_events` | CI of an ensemble event time (an05 X2); NaN unless more than half of the resamples find the event |

The moving-block bootstrap draws $\lceil n/\ell\rceil$ block starts uniformly from
$\{0,\dots,n-\ell\}$, concatenates the blocks $(d_s,\dots,d_{s+\ell-1})$, cuts to $n$ and
takes the mean ($\ell = 2$).

**Paper wording (cluster bootstrap).** "Confidence intervals of the limits of agreement were
obtained by a participant-cluster bootstrap (2000 resamples of participants with all their
cycles), so that they reflect the between-participant variability."

## 2. Signed-rank tests and the Hodges–Lehmann estimate

### 2.1 Wilcoxon signed-rank test

**What and why.** Tests whether the per-participant values (or paired differences) are
centred at 0 (or at a margin $\delta$, testing $x_i - \delta$), without a normality
assumption; the standard test for a median across 5–14 participants (Wilcoxon 1945).

Zeros are dropped (`zero_method="wilcox"`); $n'$ = the number of non-zero values. The
statistic is $T^+ = \sum_{d_i > 0} R_i$ with $R_i$ the (average) rank of $\lvert d_i\rvert$.
Under $H_0$ (symmetric about 0) without ties, $T^+$ has the exact distribution of
$\sum_{r=1}^{n'} r\,B_r$ with $B_r$ independent Bernoulli(1/2): the counts follow from the
generating function $\prod_{r=1}^{n'}(1 + x^r)$, i.e. the recursion $c_r(t) = c_{r-1}(t) + c_{r-1}(t - r)$ (`families.signed_rank_null`), $P_0(T^+ = t) = c_{n'}(t)/2^{n'}$. p values:

$$
p_{\text{greater}} = P_0(T^+ \ge t^+),\quad p_{\text{less}} = P_0(T^+ \le t^+),\quad
p_{\text{two-sided}} = \min\{1,\ 2\min(p_{\text{greater}}, p_{\text{less}})\}.
$$

**Normal approximation** (no continuity correction, SciPy's default `correction=False`):

$$
z = \frac{T^+ - n'(n'+1)/4}{\sqrt{\dfrac{n'(n'+1)(2n'+1)}{24} - \dfrac{\sum_g (t_g^3 - t_g)}{48}}},
$$

with $t_g$ the size of tie group $g$ among the $\lvert d\rvert$; $p = 1 - \Phi(z)$
("greater"), $\Phi(z)$ ("less"), $2(1 - \Phi(\lvert z\rvert))$ (two-sided).

Three call paths decide between exact and approximate p differently:

| Function | Exact p when | Otherwise | Minimum n | Used by |
|---|---|---|---|---|
| `stats/core.py:wilcoxon_one_sample` (SciPy `method="auto"`) | $n \le 50$, no zeros, no ties | ties or zeros and $n \le 13$: exact **permutation** p over all $2^n$ sign vectors (`PermutationMethod`, deterministic); $n > 13$: normal approximation | 5 finite values (zeros counted) | C2 leads and anticipation (`p`), chance null (`p_null`), X2, X3, X3 vs chance |
| `stats/families.py:wilcoxon_signed` | no zeros and no ties (checked by the code: `method="exact"`) | normal approximation (`method="asymptotic"`, tie-corrected variance) | `min_n` (2 for C1; 5 for C2-E; 2 for the camera-free contrasts; 6 for impedance) | C1 cohort test, C2-E, camera-free, impedance |
| `ml/protocols/variants.py:_wilcoxon_p`, `ml/biomech_stages.py` (SciPy default) | as `wilcoxon_one_sample` | as `wilcoxon_one_sample` | 5 pairs and not all zero (few-shot); the Friedman gate $n \ge 5$ (ablations) | decoding tests |

The stored statistic `W` is $T^+$ for one-sided tests and $\min(T^+, T^-)$ for two-sided
tests (SciPy). `exact` / `method` columns (C1, camera-free) record which p was used.

*Assumption:* the distribution of the differences is symmetric under $H_0$ (the test is then
a test of the median); participants are independent.

**Paper wording.** "Cohort effects were tested with the Wilcoxon signed-rank test over
participants (one-sided where the hypothesis was directional), using the exact null
distribution when there were no ties or zero differences and the normal approximation with
tie-corrected variance otherwise (Wilcoxon 1945)."

### 2.2 Effect size $r$ (`stats/core.py:signed_rank_z`)

$r = z/\sqrt{n'}$ with $z$ the normal-approximation statistic of §2.1 (zeros dropped, average
ranks, tie-corrected variance) computed **from $T^+$ itself**, so it is valid for one-sided
tests (a one-sided p cannot be turned back into $z$ with $\Phi^{-1}(1 - p/2)$). Signed (positive
when the values lie above 0); $\lvert r\rvert \le \sqrt{3(n'+1)/(2(2n'+1))} < 1$.

**Paper wording.** "Effect sizes are given as $r = z/\sqrt{n}$, with $z$ the standardized
signed-rank statistic (Rosenthal 1991)."

### 2.3 Hodges–Lehmann estimate and its exact interval (`stats/families.py:hodges_lehmann`, `paired_contrast`)

**What and why.** The location estimate that goes with the signed-rank test: the median of
the Walsh averages, with a distribution-free interval obtained by inverting the test (Hodges &
Lehmann 1963; Hollander, Wolfe & Chicken 2014). Used for the camera-free paired
contrasts, where a difference in RMSE between two estimators is reported with its interval.

$$
\hat\Delta_{\mathrm{HL}} = \operatorname{median}\left\{\frac{d_i + d_j}{2} : 1 \le i \le j \le n\right\}
$$

over the finite differences (zeros included). Interval: with
$c = \max\{t : P_0(T^+ \le t) \le \alpha/2\}$ from the exact null of §2.1 (no ties assumed),

$$
\mathrm{CI} = \left[W_{(c+1)},\ W_{(M-c)}\right] \quad (1\text{-based order statistics}),
$$

with coverage $1 - 2P_0(T^+ \le c) \ge 1 - \alpha$ (0.952 at $n = 13$, 0.951 at $n = 14$);
NaN when no $t$ qualifies ($2^{-n} > \alpha/2$, i.e. $n \le 5$).

> **Erratum E39.** The code used to cut one more Walsh average at each
> end, $[W_{(c+2)}, W_{(M-c-1)}]$, coverage $1 - 2P_0(T^+ \le c+1)$: 0.943 at $n = 13$ instead
> of 0.952.
> The intervals are now slightly wider; the estimates, p values and Holm decisions are
> unchanged.

`paired_contrast(d)`: `p` = two-sided `wilcoxon_signed` of the finite **non-zero**
differences (exact without ties; `method` "exact" / "approx" / "" below 2 values), `hl`,
`hl_ci_lo`, `hl_ci_hi` from all finite differences.

**Paper wording.** "Paired differences between estimators were summarized by the
Hodges–Lehmann estimate with its exact 95 % confidence interval and tested with the two-sided
Wilcoxon signed-rank test; p values were Holm-adjusted within each family of contrasts."

## 3. Monte-Carlo and permutation p values

### 3.1 Monte-Carlo p (`uncertainty/montecarlo.py:mc_pvalue`, `timing/surrogates.py:mc_p`)

**What and why.** When the null distribution is only available by simulation (a perfect hinge
plus noise, EMG surrogates, label permutations), the p value counts the simulated statistics
at least as extreme as the observed one, with the observed value counted as one of the null
draws so that the test is exact for exchangeable draws and never returns 0 (Davison & Hinkley
1997; Phipson & Smyth 2010):

$$
p = \frac{1 + \#\{b : T^*_b \ge T_{\text{obs}}\}}{N + 1}
\qquad\left(\text{"less": } T^*_b \le T_{\text{obs}}\right),
$$

over the $N$ finite null draws (NaN without a finite draw or observation). The smallest
attainable value is $1/(N+1)$; its Monte-Carlo standard error is about
$\sqrt{p(1-p)/N}$.

| Use | Statistic | $N$ | Direction |
|---|---|---|---|
| C1 per participant `p_mc_<tag>` (an06 §3) | incisor RMS of the refitted hinge | `stages.uncertainty.n_sims` = 1000 (min p 1/1001) | greater |
| C2-E locking `p_lock`, S2 `p_s2` (an05 §3.3) | IQR of the leads | `n_surrogates` = 200, `n_s2` = 200 (min 1/201) | less |
| C2-E release slope `p_s2` | Theil–Sen slope | `n_s2` = 200 | greater |
| C2-E coupling `p_coupling` | peak EMG-to-α cross-correlation | `coupling.n_null` = 200 | greater |
| decoding permutation `permutation\|lda\|emg_imu` (`ml/biomech_stages.py`) | cohort mean LOSO balanced accuracy | `ml_eval.n_perm` = 100 (min 1/101) | greater |

The decoding null refits the whole nested LOSO pipeline on labels whose sequence is
circularly shifted per participant by a uniform integer in $[L/4, 3L/4)$ ($L$ the
participant's number of windows; `ml/protocols/evaluate.py:shift_labels`), which keeps the
label autocorrelation.

**Paper wording.** "Monte-Carlo p values were computed as $(k+1)/(N+1)$, with $k$ the number of
the $N$ null simulations at least as extreme as the observed value (Phipson & Smyth 2010)."

### 3.2 Sign-flip permutation test (`stats/cohort_misc.py:sign_flip_p`)

**What and why.** The second, distribution-free test of the C2 chance null (observed lead −
null lead per participant, an05 §3.0): under $H_0$ the sign of each difference is
exchangeable.

$$
\bar d^*_b = \frac1n\sum_{i=1}^n s_{bi}\,\lvert d_i\rvert,\quad s_{bi} \in \{-1, +1\}\ \text{i.i.d. equiprobable},
\qquad p = \frac{1 + \#\{b : \bar d^*_b \ge \bar d - \epsilon\}}{1 + N},
$$

$N$ = `N_PERM_NULL` = 10 000 random sign vectors (drawn with replacement, not the $2^n$
enumeration), $\epsilon = 10^{-9}\max(1, \lvert\bar d\rvert)$ (ties at the observed mean count
as extreme), seed = study seed; NaN below 5 pairs. One-sided (> 0).

**Paper wording.** "As a distribution-free check, the mean paired difference was also tested
with a one-sided sign-flip permutation test (10 000 random sign assignments)."

## 4. Multiplicity

### 4.1 Holm (`stats/core.py:holm`)

**What and why.** Controls the family-wise error rate within a claim family without
assumptions on the dependence between tests (Holm 1979).

With the $m$ finite p values ordered $p_{(1)} \le \dots \le p_{(m)}$ (NaN kept as NaN and not
counted in $m$):

$$
\tilde p_{(j)} = \max_{l \le j}\ \min\left\{1,\ (m - l + 1)\,p_{(l)}\right\},
$$

(step-down; the running maximum makes the adjusted p monotone). A hypothesis is rejected when
$\tilde p \le \alpha$ (the confirmatory SPM decision uses $< \alpha$, §10).

**Paper wording.** "Within each family of tests, p values were adjusted with the Holm
step-down procedure (Holm 1979)."

### 4.2 Intersection-union test and serial gatekeeping (`stats/families.py:iut`, `gatekeeping`, `block_holm`; `stats/cohort_c2e.py`)

**Intersection-union test (IUT).** A hypothesis that holds only if each part holds (C2-E H1:
the closer onsets are *locked* to closing **and** *located* before it) is rejected when
every part is rejected at α; the IUT p is the largest part p (Berger 1982; Berger & Hsu 1996):

$$
p_{\mathrm{IUT}} = \max(p_{\text{lock}},\ p_{\text{loc}}) \quad(\text{NaN if any part is NaN}).
$$

No multiplicity adjustment of the parts is needed (the IUT is level α by construction).

**Serial gatekeeping** (Westfall & Krishen 2001; Dmitrienko et al. 2003). H1 is tested at α;
only if $p_{\mathrm{IUT},1} \le \alpha$ are H2 and H3 tested, Holm-adjusted between them:

$$
(\tilde p_2, \tilde p_3) =
\begin{cases}
\mathrm{Holm}(p_2, p_3) & p_1 \le \alpha\\
(\text{NaN}, \text{NaN}) \ (\text{not tested}) & \text{otherwise.}
\end{cases}
$$

Supporting rows: Holm within their block (`block_holm`), no claim rests on them; a supporting
row in `supporting_veto` with its upper CI below 0 vetoes the decision. Post hoc sensitivity
(decision of the authors, 2026-10-31): `p_holm_family` = Holm over the three IUT p values of H1–H3,
always computed. Decision rule (`stats/cohort_c2e.py:decide`): "shown (post hoc)" when H1 is
rejected, the BCa lower bound of the median μ exceeds δ (`stages.timing.c2e.delta_ms` = 17),
no veto and `gates_passed`; "inconclusive (underpowered)" when H1 is not rejected and
`gate_power_40ms` < 0.5; "delta not fixed" without δ; else "not shown".

**Paper wording.** "The primary post hoc hypothesis required both time-locking and a
location before the movement onset and was tested as an intersection-union test (p = the
larger of the two p values; Berger 1982). The two secondary hypotheses were tested only if the
primary was rejected (serial gatekeeping), Holm-adjusted between them."

### 4.3 Log-ratio and location tests (`stats/families.py:log_ratio_test`, `location_test`)

- `log_ratio_test(ℓ, alternative)`: per participant $\ell_i = \log(\text{observed}_i / \text{reference}_i)$; reports $n$, $\tilde\ell$ with its BCa CI (§1.2) and the signed-rank
  test of $\ell_i$ against 0 (§2.1, `wilcoxon_signed`, `min_n` = 5 for C2-E). Used for C2-E
  locking ($\log(\mathrm{IQR}_{\text{obs}}/\operatorname{median}\mathrm{IQR}_{S1}) < 0$) and
  release depth (> 0). The C1 cohort test applies the same idea inline
  (`stats/cohort_kinematics.py:hinge_noise_tests`: `wilcoxon_greater` of $\log(r_{\text{obs}}/ \hat q_{0.99})$ over participants with a finite positive ratio, `min_n` = 2).
- `location_test(x, δ)`: $n$, $\tilde x$ with its BCa CI and the signed-rank test of
  $x_i - \delta$ (default "greater"; NaN p without a margin).

The log makes the ratio symmetric (a doubling and a halving are equally far from 1), so the
signed-rank symmetry assumption is more plausible on the log scale.

**Paper wording (cohort test of the hinge residual).** "Across participants, the log ratio of the observed
residual to the 99th percentile of its null distribution was tested against 0 with a one-sided
Wilcoxon signed-rank test."

## 5. Descriptives and effect sizes (S1, exploratory)

- **Hedges' *g*** for sex (F − M) (`stats/core.py:hedges_g`; Hedges 1981):

  $$
  g = J\,\frac{\bar x_F - \bar x_M}{s_p},\qquad
  s_p = \sqrt{\frac{(n_F - 1)s_F^2 + (n_M - 1)s_M^2}{n_F + n_M - 2}},\qquad
  J = 1 - \frac{3}{4(n_F + n_M) - 9},
  $$

  ($J = 1 - 3/(4\nu - 1)$ with $\nu = n_F + n_M - 2$, the small-sample bias correction); it
  needs ≥ 2 per group; NaN if $s_p = 0$. CI: percentile bootstrap, 4000 resamples, each group
  resampled separately (§1.3; the replicates with $s_p = 0$ are NaN and ignored by
  `nanpercentile`).
- **Spearman correlation with age** (`scipy.stats.spearmanr`, ρ and its p from the
  *t*-approximation $t = \rho\sqrt{(n-2)/(1-\rho^2)}$ on $n - 2$ df): needs > 3 participants
  with both values. The decoding stage reports the same for balanced accuracy vs age
  (`spearman_age_vs_ba`, ≥ 5 participants).
- **Leave-one-out influence** (`stats/cohort_kinematics.py:influence`, table
  `stats_kinematics.influence`):

  $$
  \text{change\_pct}_j = 100\,\frac{\bar x_{(-j)} - \bar x}{\bar x},
  $$

  the plain cohort mean without participant $j$ against the mean of all; computed for 4
  metrics (`opening_selfpaced_mm`, `protrusion_mm`, `laterotrusion_left_mm`,
  `peak_open_vel_dps`) and each of `stages.stats_kinematics.older_pids` (s02, s06, s08).
- Hedges' *g*, Spearman and influence are all *exploratory*.
- `holm` adjusts p-values within a claim family (§4.1).

**Paper wording.** "Sex differences are described by Hedges' *g* with a bootstrap 95 %
confidence interval and age associations by Spearman's ρ (both exploratory); the influence
of the oldest participants is given as the change of the cohort mean when each is left out."

## 6. Models

- **Linear mixed model** (`stats/core.py:mixed_model`; table `stats_emg.activation_lmm`;
  Laird & Ware 1982):

  $$
  \log y_{ijk} = \beta_0 + \tau_j + \gamma_k + (\tau\gamma)_{jk} + u_i + \varepsilon_{ijk},\qquad
  u_i \sim \mathcal N(0, \sigma_u^2),\ \varepsilon_{ijk} \sim \mathcal N(0, \sigma^2),
  $$

  participant $i$, task $j$ (treatment coding, `C(task)`), muscle $k$ (`C(muscle)`), with
  $y = \max(\texttt{median\_pct}, 0.1)$; statsmodels formula `log_pct ~ C(task) * C(muscle)`,
  random intercept per participant, fitted by **REML**. Fixed-effect tests are Wald tests,
  $z = \hat\beta/\mathrm{SE}$ with normal p and CI $\hat\beta \pm 1.96\,\mathrm{SE}$
  (statsmodels `MixedLMResults`).
  - Data: amplitude-usable channels and tasks other than rest; it runs only with ≥ 5
    participants.
  - Optimizers are tried in the order lbfgs → bfgs → powell → nm (`MIXED_METHODS`). A fit is
    accepted only with a finite log-likelihood and finite fixed-effect SEs that are all
    < 10³. statsmodels can flag a diverged fit as converged, so this check is needed. If no
    optimizer passes, `mixed_model` raises `RuntimeError`; `activation_tables` then writes one
    row `status = not converged` (`formula`, `n_participants`, `n_obs`, `note` = the error)
    instead of the fixed effects (E25; formerly the error took every other `stats_emg` table
    with it).
  - Columns: `term`, `coef`, `se`, `z`, `p`, `ci_lo`, `ci_hi`, `optimizer`, `group_var`
    ($\hat\sigma_u^2$), `scale` ($\hat\sigma^2$), `llf`, `status` (`converged`).
  - The camera-free drift model uses the same function with `err_open_deg ~ t_min` (§15).
- **One-sample Wilcoxon** (`stats/core.py:wilcoxon_one_sample`): needs $n \ge 5$ finite
  values; p from SciPy `wilcoxon` (`method="auto"`, §2.1; alternative as stated: "greater"
  for leads); effect size $r = z/\sqrt{n'}$ (§2.2).
- **Friedman, then Wilcoxon with Holm, for the decoding ablations** (`ml/biomech_stages.py`;
  an05 §5): per participant the LOSO balanced accuracy of the primary model
  (`ml_eval.primary_model` = svm) for each feature set; with ≥ 3 sets and ≥ 5 complete
  participants, `scipy.stats.friedmanchisquare` (Friedman 1937; chi-square approximation on
  $K - 1$ df for $K$ sets), then two-sided Wilcoxon (SciPy default) of emg_imu against each
  other set, Holm over those tests (`stat` = median difference). Few-shot
  (`ml/protocols/variants.py:few_shot_tests`): emg_imu − imu per calibration amount $k$,
  two-sided Wilcoxon, Holm over the $k$ with a finite p (`n_amounts`).

**Paper wording (LMM).** "Normalized activation was modelled on the log scale with a linear
mixed model (task × muscle fixed effects, random intercept per participant, REML)."

## 7. C2 timing families (D15, `stats/cohort_misc.py`, stage `stats_timing`)

- Per-participant value (`per_participant_timing`): with `stages.stats_timing.aggregation:
  pooled` (primary) the median over the participant's usable channel-cycles of the direction's
  muscle group, pooled over channels; participants with < `min_cycles` (3) usable
  channel-cycles are excluded and counted (`n_excluded_min_cycles`). `muscle_mean` (pre-D15,
  sensitivity `c2_muscle_mean`, with `min_cycles: 1`) = unweighted mean of the per-muscle
  medians. Details in an05 §3.
- Tests: one-sided Wilcoxon ("greater", `wilcoxon_one_sample`, §2.1) of `lead_ms` and
  `anticipation_ms` per task × direction × detector × rule, with the median and its BCa CI
  (`describe`).
- `family` / `p_holm` (`_timing_family`): **confirmatory** = `lead_ms`, opening, `tkeo`, both
  onset rules (`first`, `drive`; `stages.stats_timing.confirmatory_rules`, default
  `CONFIRMATORY_RULES`), tasks `open_close` and `chewing`: Holm across these four tests (D15).
  **Exploratory** = every other tested row, Holm within its metric (all task × direction ×
  detector × rule rows of `lead_ms`, respectively `anticipation_ms`). **Descriptive** = the
  untested metrics (`lead_omega_ms`, `lead_acc_ms`, `lead_opt_ms`, `causal_delay_ms`; `p_holm`
  NaN). A tested row with n < 5 has p NaN, which Holm keeps as NaN and does not count. Before
  D15 no p-value of this table was adjusted.
- **Chance null (D23, D23a, D34; an05 §3.0):** the `lead_ms` rows also carry the calibrated
  test of observed − null per participant (`p_null`: one-sided Wilcoxon > 0; `p_null_perm`:
  sign-flip permutation, 10 000, §3.2; Holm within the same families, `p_null_holm`,
  `p_null_perm_holm`). A participant's null needs ≥ `min_cycles` usable pseudo channel-cycles.
  It quantifies the window bias of the asymmetric search window and has no power for
  physiological leads (D34); the frozen test against 0 stays the primary.
- **C2 verdict:** "not confirmed" (both rules needed); the post hoc family C2-E (an05 §3.3)
  does not change it.
- **X3 vs chance** (`release_cohort`, `_vs_chance`): per participant anticipation − chance
  anticipation, two-sided signed-rank test (`wilcoxon_one_sample`) over the participants with
  both values; only the detection rate above chance carries a statement (D34).

**Paper wording.** "The opening lead was tested against zero with one-sided Wilcoxon
signed-rank tests over participants for both onset rules and both tasks; the four
confirmatory p values were Holm-adjusted."

## 8. Reliability (`stats/core.py:icc_sem_mdc`, `stats_kinematics.reliability`, `ok` cycles)

- Metrics: open_close `opening_mm`, `theta_y_deg`, `peak_open_vel_dps`; chewing `opening_mm`;
  protrusion_retrusion `protrusion_mm`; left_right \|`lateral_mm`\| **per side**. Laterotrusion
  cycle numbers restart per side.
- **ICC(3,1)** consistency (two-way mixed, single measurement; pingouin `ICC3` /
  `ICC(C,1)`; Shrout & Fleiss 1979, McGraw & Wong 1996) across the first $k = 8$ cycles by
  cycle number, cycles in the role of raters:

  $$
  \mathrm{ICC}(3,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E},
  $$

  with $MS_R$ the between-participant (rows) and $MS_E$ the residual mean square of the
  participant × cycle two-way ANOVA; its 95 % CI from the *F* distribution (pingouin).
  Participants with < 8 cycles are dropped; it needs ≥ 3 participants.
- $\mathrm{SEM} = s\sqrt{\max(1 - \mathrm{ICC}, 0)}$, with $s$ the SD (ddof 1) of all
  participant × cycle values that entered the ICC (Weir 2005).
- $\mathrm{MDC}_{95} = 1.96\sqrt2\,\mathrm{SEM}$.
- **Bland–Altman** limits of agreement for repeated cycles per participant (`bland_altman_rm`,
  §9; Bland & Altman 2007, "true value varies"; D7 in the change log).
- **Between-day (S5.4):** sensitivity variant `s02_day2` (`base: reporting_population`, `cohort_swap:
  {s02: 1}`: s02 analysed from its second session in place of the first; only s02's units
  re-run; `sensitivity.py:_changes`), compared with the 13-adult store like every variant.

**Paper wording.** "Cycle-to-cycle reliability was quantified by the consistency ICC(3,1)
over the first eight valid cycles, the standard error of measurement
$\mathrm{SEM} = \mathrm{SD}\sqrt{1 - \mathrm{ICC}}$ and the minimal detectable change
$\mathrm{MDC}_{95} = 1.96\sqrt{2}\,\mathrm{SEM}$ (Weir 2005)."

## 9. Bland–Altman limits of agreement (dataset)

**Simple** (`stats/core.py:bland_altman`): $d_j = \text{est}_j - \text{ref}_j$, bias $\bar d$,
LoA $\bar d \pm 1.96\,s_d$, RMSE $\sqrt{\overline{d^2}}$; ≥ 3 pairs (Bland & Altman 1986).

**Repeated measures** (`stats/core.py:bland_altman_rm`; Bland & Altman 2007, "true value
varies"; also `stats/cohort_misc.py:_ba_anova`). Participant $i = 1,\dots,k$ with $m_i$ cycles
$d_{ij}$, $N = \sum_i m_i$, participant means $\bar d_i$, cycle-weighted grand mean $\bar d$:

$$
MS_B = \frac{\sum_i m_i(\bar d_i - \bar d)^2}{k-1},\qquad
MS_W = \frac{\sum_i\sum_j (d_{ij} - \bar d_i)^2}{N-k},\qquad
D = \frac{N^2 - \sum_i m_i^2}{(k-1)N},
$$

$$
\hat\sigma_B^2 = \max\!\left(\frac{MS_B - MS_W}{D},\ 0\right),\qquad
s = \sqrt{\hat\sigma_B^2 + MS_W},\qquad
\text{LoA} = \bar d \pm 1.96\,s.
$$

Needs $k \ge 2$ and $N - k \ge 1$. Pooling all cycles as if independent would make the limits
too narrow. Columns `bias`, `loa_lo`, `loa_hi`, `rmse`, `sd_within` ($\sqrt{MS_W}$),
`sd_between` ($\hat\sigma_B$).

**Participant-cluster bootstrap CIs** (`stats/core.py:bland_altman_rm_ci`, §1.3): bias, both
LoA and the **proportional bias** — the OLS slope of $d$ on $(\text{est} + \text{ref})/2$ at
cycle level (`np.polyfit`, degree 1; `prop_slope`) — get 95 % percentile CIs from 2000
resamples of participants (all their cycles; a participant drawn twice counts as two in the
variance components); needs ≥ 3 participants; stored in `stats_imu.imu_opening_ba`
(`*_ci_lo/hi`, `prop_slope`) and `camerafree_cohort.cf_cycles_ba`.

**Largest absolute limit with the MOVER interval** (`stats/cohort_misc.py:max_abs_loa_mover`;
Zou 2013 for limits of agreement with multiple observations per individual; Donner & Zou 2012
for the method of variance estimates recovery). The quantity a user must expect is

$$
\theta = \max(\lvert\bar d - 1.96 s\rvert, \lvert\bar d + 1.96 s\rvert) = \lvert\bar d\rvert + 1.96\,s .
$$

$s^2 = MS_B/D + (1 - 1/D)\,MS_W$ is a sum of scaled chi-square terms $c_r$ with $\nu_1 = k-1$
and $\nu_2 = N-k$ degrees of freedom; each term's exact limits are
$[l_r, u_r] = \text{sort}\big(c_r\nu_r/\chi^2_{1-\alpha/2,\nu_r},\ c_r\nu_r/\chi^2_{\alpha/2,\nu_r}\big)$
and

$$
L_{s^2} = s^2 - \sqrt{\textstyle\sum_r (c_r - l_r)^2},\qquad
U_{s^2} = s^2 + \sqrt{\textstyle\sum_r (u_r - c_r)^2},
$$

$l_s = \sqrt{\max(L_{s^2}, 0)}$, $u_s = \sqrt{U_{s^2}}$. When $\hat\sigma_B^2 = 0$ only the
$MS_W$ term is used ($s^2 = MS_W$). The bias has a *t* interval on $k - 1$ df with
$\mathrm{Var}(\bar d) = \hat\sigma_B^2\sum_i m_i^2/N^2 + MS_W/N$; the interval of
$\lvert\bar d\rvert$ is $[l_{\lvert\mu\rvert}, u_{\lvert\mu\rvert}]$ with
$u_{\lvert\mu\rvert} = \max(\lvert l_\mu\rvert, \lvert u_\mu\rvert)$ and
$l_{\lvert\mu\rvert} = 0$ if the bias interval contains 0, else $\min(\lvert l_\mu\rvert, \lvert u_\mu\rvert)$. Then

$$
\theta_L = \theta - \sqrt{(\lvert\bar d\rvert - l_{\lvert\mu\rvert})^2 + 1.96^2(s - l_s)^2},\qquad
\theta_U = \theta + \sqrt{(u_{\lvert\mu\rvert} - \lvert\bar d\rvert)^2 + 1.96^2(u_s - s)^2}.
$$

The participant-cluster percentile bootstrap of $\theta$ is stored as a sensitivity
(`max_abs_loa_boot_ci_*`): with 13 clusters it under-covers (coverage simulation: 85 %, 14 %
upper misses against 2.5 % nominal; MOVER 97 %, 2.3 %;
`tests/test_r10_requirements.py::test_max_abs_loa_interval_coverage_13_clusters`).

**Paper wording.** "Agreement over repeated cycles was quantified by Bland–Altman limits of
agreement for multiple observations per participant (one-way ANOVA estimator; Bland & Altman
2007), with participant-cluster bootstrap confidence intervals; the largest absolute limit
of agreement was given with a MOVER confidence interval (Zou 2013)."

## 10. Waveforms (spm1d 0.4, `stats/waveforms.py`, `stats/cohort_emg.py:spm_tests`)

Profiles: (cycles × 101) per participant, from the `emg` stage. Cycles with any NaN node are
dropped per test.

**SPM paired *t* (random field theory; Pataky 2012, Pataky et al. 2015).** At each node
$q = 0,\dots,100$ of the $n$ paired subject-mean curves:

$$
t(q) = \frac{\bar d(q)}{s_d(q)/\sqrt n},\qquad \nu = n - 1 .
$$

spm1d estimates the smoothness (FWHM) of the residual fields and sets the critical threshold
$t^*$ (`zstar`) so that $P\big(\max_q t(q) > t^*\big) = \alpha/2$ for a smooth Gaussian-based
*t* field of that FWHM (two-tailed: α/2 per tail; `withBonf=True`: the survival probability
is the smaller of the RFT and the Bonferroni value, so for very rough fields $t^*$ never
exceeds the Bonferroni threshold). Supra-threshold clusters are the intervals with
$\lvert t(q)\rvert > t^*$ (endpoints interpolated); each cluster's p is
$\min\{1, 2\,P_{\mathrm{RFT}}(\text{extent} \ge \text{observed resels} \mid t^*)\}$. A test
rejects $H_0$ when any cluster exists.

- **Paired t on subject means** (`paired_t`, the two confirmatory tests):
  - It needs n ≥ 5 participants with complete pairs; α = 0.05, two-tailed.
  - Test 1: masseter L vs R in chewing, normalized activity (% of max recorded, D11) averaged
    over cycles. Only participants with both masseters shape-usable enter.
  - Test 2: open_close, openers vs closers. Each curve is the mean over the shape-usable
    channels of the group, then over cycles, then divided by its maximum (peak = 1).
  - Holm across the two tests uses each test's minimum cluster p. A test that ran but found no
    cluster counts as p = 1; a test that did not run counts as NaN.
  - `h0reject` is the confirmatory decision, Holm-adjusted p (`p_holm`) < 0.05 (None if the test
    did not run, False if it ran but `p_holm` is NaN); `h0reject_raw` keeps spm1d's own
    per-test decision (E25; formerly `h0reject` was the unadjusted decision next to the
    adjusted p).
- **One-way ANOVA across participants** (`anova1_participants`), cycles as repetitions,
  descriptive: $F(q)$ of the one-way model with participants as groups, RFT threshold $F^*$:
  - It needs ≥ 3 participants with ≥ 3 complete cycles.
  - Data: the open_close opening angle and the chewing masseter_r (shape-usable).
  - $\texttt{supra\_frac} = \frac{1}{101}\#\{q : F(q) > F^*\}$.
- **Nonparametric paired t per participant** (`paired_t_nonparam`; Nichols & Holmes 2002):
  masseter L vs R in chewing, cycles paired. The same $t(q)$; the null of $\max_q\lvert t(q)\rvert$
  from random sign flips of the cycle differences; $t^*$ its $1 - \alpha$ quantile; cluster p
  from the permutation distribution of the cluster integral (spm1d default
  `MaxClusterIntegral`).
  - It needs n ≥ 6 complete cycles.
  - Permutations: min(1000, 2ⁿ); two-tailed, α = 0.05.
  - Seeded (E16): spm1d draws its permutations from numpy's global random state, so each test
    runs inside `seeded_global_rng(seed)` (`stats/waveforms.py`), which seeds the global state,
    restores it afterwards and holds a lock so concurrent stages cannot interleave draws. The
    seed is derived from the participant, `zlib.crc32("masseter_L_vs_R|<pid>")` (not the study
    `seed`), so a participant's result does not depend on which others are in the run.
  - The summary row counts the participants tested and those significant. The evidence is
    within-session and applies to that individual only.
- **Spectral SPM** (`stats/cohort_emg.py:spectral_tables`, X1, exploratory): the same
  `paired_t` and `anova1_participants` along the frequency axis (log power share per 10-Hz
  bin; clusters converted to Hz).
- **Two-sample Welch SPM** (systems study, `validation/emg.py:ttest2_spm`):
  `spm1d.stats.ttest2(equal_var=False)`, cycles of the two sessions as independent groups,
  ≥ 3 per group.
- **Pointwise variance ratio** (`variance_ratio`):
  $\mathrm{between}(q) = \operatorname{Var}_i\big(\bar y_i(q)\big)$ (ddof 1, participant mean
  curves), $\mathrm{within}(q) = \frac1n\sum_i \operatorname{Var}_j\big(y_{ij}(q)\big)$ (ddof 1),
  ratio = between / max(within, 10⁻¹²); it needs ≥ 3 cycles per participant and ≥ 3
  participants.
- **CMC** (Kadaba et al. 1989; `cmc`), on the open_close angle, for $m$ curves $Y_{jq}$ of $n$
  nodes, node means $\bar Y_q$, grand mean $\bar Y$:

  $$
  \mathrm{CMC} = \sqrt{1 - \frac{\sum_j\sum_q (Y_{jq} - \bar Y_q)^2 / \big(n(m-1)\big)}
                             {\sum_j\sum_q (Y_{jq} - \bar Y)^2 / (mn - 1)}},
  $$

  NaN if the radicand is < 0 or $m < 2$. within = the median over participants of each
  participant's CMC over its cycles; between = the CMC of the participant mean curves. The
  systems study uses the CMC of the two systems' curves per cycle (`validation/agreement.py:
  cmc_pair`).

**Paper wording.** "Waveforms were compared with paired *t*-tests in statistical parametric
mapping (spm1d; Pataky 2012), with the critical threshold from random field theory
(α = 0.05, two-tailed); the minimum cluster p values of the two confirmatory tests were
Holm-adjusted. Single-participant comparisons used the nonparametric permutation version."

## 11. Variability and data-collection requirements (S2, stage `variability`)

- **Specification curve** (`stats/cohort_misc.py:spec_curve`; in the spirit of Simonsohn,
  Simmons & Nelson 2020): one value per participant and processing variant.
  - Variants: MIO (`kinematics.spec`): cycle start vs rest reference, 3D vs vertical, raw vs
    low-pass; activation `median_pct` in open_close and chewing (`emg.spec`): normalized by the
    max recorded, the frozen MVC/RVC p95 or the largest 1-s mean (D11); leads
    (`stats_timing.timing_spec`): 3 onset detectors × 4 mechanical sources (combined, ω, \|a\|,
    optical) × 3 onset rules (first, driving burst, in-task threshold `drive_taskbase`) = 36;
    the per-participant lead values follow `stages.stats_timing.aggregation` (D15: pooled
    channel-cycles, participants below `min_cycles` left out).
  - With $\mu_v$ the plain mean over the participants that have variant $v$:

    $$
    \text{spread} = \max_v \mu_v - \min_v \mu_v,\qquad
    \text{spread\_over\_sd} = \frac{\text{spread}}{s_{\text{primary}}},
    $$

    $s_{\text{primary}}$ the between-participant SD (ddof 1) of the **primary** variant.
  - Primary variants are explicit: `mio_mm` → `cycle_start|vertical|lowpass`; `median_pct` →
    `norm=<signals.emg.reference>` (`max_recorded`); timing (keys `open_close`, `chewing`) →
    `detector=tkeo|mech=lead_ms|onset=<stages.timing.onset_rule>` (`drive`). If no variant
    starts with the primary string, the alphabetically first variant is used.
  - Tables: `variability.spec_curve` (per outcome: `n_variants`, `primary_variant`,
    `primary_mean`, `variant_min/max`, `spread`, `sd_between`, `spread_over_sd`) and
    `spec_values` (the per-participant values).
- **Cycles needed** (`stats/cohort_misc.py:cycles_needed_table`,
  `stats/waveforms.py:cycles_needed`), per participant:
  - Keys: MDC95 per (task, metric, side) from `reliability`. Laterotrusion has its own side's
    cycles and MDC; for left_right the values are \|·\|.
  - It needs ≥ 4 finite `ok` cycles.
  - For each $n = 2,\dots,n_{\text{avail}}$, $B = 2000$ resamples draw $n$ of the
    participant's cycles with replacement; the half-width is
    $h(n) = \tfrac12\big(Q_{0.975} - Q_{0.025}\big)$ of the resampled means.
  - $n_{\text{needed}} = \min\{n : h(n) \le \mathrm{MDC}_{95}/2\}$, or NaN if it never gets
    there (no extrapolation beyond the available cycles). Study seed (see Seeds).
- **Protocol heterogeneity** (`variability.protocol`): phase durations (experiment and MVC
  phases), `ok` cycles per task, median cycle duration per task, and per participant the largest
  masked share of an MVC/RVC reference phase (`emg.refs.mask_frac`, analysis mask); min /
  median / max over participants.

**Paper wording.** "Analytical flexibility was quantified by a specification curve: the
spread of the cohort mean across processing variants relative to the between-participant SD
of the primary variant. The number of cycles needed per participant was the smallest number
for which the bootstrap 95 % confidence interval of the participant mean had a half-width
of at most half the MDC95."

## 12. Literature norms (`norms/literature.py`, stage `stats_kinematics`)

- Each metric in `norms/norms.yaml` that exists in the subjects table is compared per
  participant with the reference entry of the participant's sex. If there is none, the pooled
  entry (`sex: all`) is used.
- $\text{diff} = x - \mu_{\text{ref}}$, $z = (x - \mu_{\text{ref}})/\sigma_{\text{ref}}$; $z$
  is NaN if the entry has no SD.
- Tables:
  - `norms_subjects`: pid, metric, value, `ref_mean`, `ref_sd`, `ref_sex`, `diff`, `z`,
    `source`;
  - `norms_summary`: per metric × `ref_mean` × source, n, `mean_diff` and `sd_diff` (ddof 1).

## 13. Derived requirements (stage `requirements`, `stats/requirements.py`)

One table, one row per requirement, so Table I (conference) / Table II (journal) are
formatted from stored numbers only. Columns (`COLUMNS`): `key, group, requirement, value,
ci_lo, ci_hi, median, unit, statistic, n, derivation, evidence, implication, note`; the E31
capacity columns `value_linear, lower_k, lower_conf, upper_note, sample_max, max_q,
max_conf`; the C2 latency-budget columns `lead_null_ms, lead_minus_null_ms, p_null,
p_null_holm, chance_distinguishable, lead_omega_ms, lead_opt_ms`; and the D26 columns of the
IMU opening rows `ci_method, ci_lo_boot, ci_hi_boot, value_insample`. Each input keeps its own
participant set (E32): optical rows use every participant with optics, IMU rows those with
IMU data (`n` per row).

**Percentile policy (capacity rows: incisal opening, protrusion, laterotrusion, opening
rotation, translation to allow, peak angular velocity, control bandwidth, and the post hoc X7
rows `close_vel_dps`, `free_peak_open_vel_dps`, `peak_acc_dps2`, `posterior_excursion_mm`
(participant values in an02 §7, §7a; lower bounds); E31, `p95_design`).** The design value is the 95th percentile across participants of the
participant-level metric (a device should accommodate 95 % of users). At n = 13 or 14 P95
lies between the two largest participants and **no finite two-sided 95 % interval exists**:
$P(X_{(n)} > Q_{0.95}) = 1 - 0.95^{n}$ = 0.51 at n = 14, and a percentile bootstrap of so
extreme a quantile can never reach above the sample maximum. Per row:

- `value`: the **Harrell–Davis** estimate (`harrell_davis`, `scipy.stats.mstats.hdquantiles`;
  Harrell & Davis 1982), a weighted mean of all order statistics,

  $$
  \hat Q_p^{\mathrm{HD}} = \sum_{i=1}^n w_i X_{(i)},\qquad
  w_i = I_{i/n}(a, b) - I_{(i-1)/n}(a, b),\quad a = p(n+1),\ b = (1-p)(n+1),
  $$

  $I_x(a,b)$ the regularized incomplete beta function, $p = 0.95$; `value_linear`: the
  linear-interpolation P95 (`numpy.percentile`, the reported value before E31) stored next to it;
- `ci_lo`: the exact distribution-free **one-sided lower 95 % bound** (`order_stat_lower`;
  David & Nagaraja 2003). Because $X_{(k)} \le Q_p$ exactly when at least $k$ of the $n$
  values fall at or below $Q_p$,

  $$
  P(X_{(k)} \le Q_p) = P\big(\mathrm{Bin}(n, p) \ge k\big) = \sum_{j=k}^{n}\binom{n}{j}p^j(1-p)^{n-j},
  $$

  and the bound is $X_{(k^*)}$ with $k^* = \max\{k : P(\mathrm{Bin}(n, 0.95) \ge k) \ge 0.95\}$ (`lower_k`, achieved confidence `lower_conf`): X(12) of 14 with 0.970, X(11) of 13
  with 0.975;
- `ci_hi` NaN with `upper_note` "none finite at n = …": a one-sided upper bound $X_{(k)}$
  needs $P(\mathrm{Bin}(n, p) \le k - 1) \ge 0.95$ (`order_stat_upper`); even $k = n$ gives
  $1 - 0.95^n$, which reaches 0.95 only for n ≥ 59;
- `sample_max` with its coverage statement: $P\big(X_{(n)} > Q_{0.80}\big) = 1 - 0.8^{n}$
  (`max_conf`; 0.956 at n = 14, 0.945 at n = 13): with that confidence the maximum exceeds the
  `max_q` = 80th percentile of the population;
- `median` for orientation.
Reported as "P95 (lower 95 % bound; maximum)". Workspace values are lower bounds (no maximal
opening, D1).

**Paper wording.** "Design values are the 95th percentile across participants (Harrell–Davis
estimator), reported with the exact distribution-free one-sided lower 95 % confidence bound
from the order statistics; with 13 participants no finite upper bound exists, and the sample
maximum exceeds the population 80th percentile with 94.5 % confidence."

**Performance rows** (fixed-hinge misalignment, the C2 leads under both onset rules with the
D15 minimum and their D23 null columns, closer release, causal detection, IMU angle RMSE):
cohort medians with BCa CIs (`describe`, §1). **IMU opening agreement (C3a, D26):** the largest
absolute repeated-measures limit of agreement $\lvert\bar d\rvert + 1.96 s$ with the **MOVER**
interval (§9; `ci_method` = `mover_zou2013`), the participant-cluster bootstrap in
`ci_lo_boot` / `ci_hi_boot`. The pre-specified C3a model is the generic leave-one-subject-out
model, reported first; the participant's own model is predicted leave-one-cycle-out (D26;
the in-sample fit is a sensitivity, `value_insample`).

**Post hoc rows (C2-E, X6):** `c2e_h1_ms` "closer activation before closing" and `c2e_h2_ms`
"closer release before opening (null-calibrated)", evidence `post_hoc` (an05 §3.3).

Evidence levels stored (`evidence` column, one map `stats/evidence.py:REQUIREMENT_EVIDENCE`,
`evidence.py:23-48`; an unlisted key fails): `descriptive` (S1 workspace and speed rows
`opening_mm`, `protrusion_mm`, `laterotrusion_mm`, `opening_angle_deg`, `translation_mm`,
`peak_vel_dps`; C3a `imu_angle_rmse_deg`, `imu_opening_loa_subject_mm`), `confirmatory`
(`misalign_hinge_fit_mm`), `secondary_post_hoc` (`misalign_hinge_tmj_mm`, D33), `primary`
(`imu_opening_loa_loso_mm`, the pre-specified C3a outcome), `exploratory` (`bandwidth_hz`,
`closer_release_ms`, `causal_release_ms`), `not_confirmed` (`lead_first_ms`),
`not_confirmed_posthoc_rule` (`lead_drive_ms`), `post_hoc` (X6: `c2e_h1_ms`, `c2e_h2_ms`) and
`post_hoc_descriptive` (the four X7 rows). The reader words come from `etc/vocab.yaml` domain
`evidence`; the domain `requirement_evidence` is only the paper Table I override of the shown
level (`vocab.yaml:253-261`). Decoding (C3b) is
added by the table view from `ml_eval.summary` (ml is an optional plugin). **D27 sensitivity**
(`requirements_sensitivity`): variant `largest_cycle`, the five `UNIFORM` rows `opening_mm`,
`protrusion_mm`, `laterotrusion_mm`, `opening_angle_deg` and `peak_vel_dps` with one uniform
per-participant statistic (the largest `ok` cycle of the task; not `translation_mm` nor the X7
rows), and the bandwidth f95 from the IMU alone (`imu_only`) and from the optics alone
(`optics_only`) next to the coherence-weighted primary; D29: the bandwidth with the
noise-limited rule at 1.5 / 3 Hz (`min_band_1.5hz`, `min_band_3hz`).

**X8 planning table** (`requirements:max_coverage`, `coverage_table`,
`requirements.py:116-134`, `stages.py:451`; post hoc, planning only, uses no data): for
$n$ = `COVERAGE_N` = 13…30 and $q$ ∈ `COVERAGE_Q` = {0.80, 0.90} the chance that the largest
of $n$ participants exceeds the population $q$-quantile, `p_max_exceeds` = $1 - q^n$, and per
$q$ the smallest $n$ with $1 - q^n \ge$ `CONF` = 0.95, $n = \lceil\log 0.05/\log q\rceil$ (row
with `smallest_n` True): what more participants buy.

**Bland–Altman CIs** (`stats.core.bland_altman_rm_ci`): §9.

## 14. C1 against the noise nulls (D24, D25, D33; stages `uncertainty`, `stats_kinematics`)

- **Per participant** (`uncertainty.hinge_noise`, `uncertainty/montecarlo.py`; an06 §3 has the
  simulation): for every null (`bench`; `invivo`, the pre-specified primary; the sensitivities
  `invivo_slow`, `invivo_motion`, `invivo_teeth`; the post hoc condyle-hinge nulls
  `bench_tmj`, `invivo_tmj`) the median and P99 $\hat q_{0.99}$ (linear interpolation) of
  `uncertainty.n_sims` = 1000 draws (was 100), `above_<null>` = $r_{\text{obs}} > \hat q_{0.99}$ (primary rule, unchanged), `ratio_<null>` = $r_{\text{obs}}/\hat q_{0.99}$ and the
  Monte-Carlo p `p_mc_<null>` = $(k+1)/(N+1)$, $k$ the draws ≥ the observed residual, $N$ the
  finite draws (§3.1; smallest attainable p = 1/1001).
  The nulls (noise models, axes, padding, refits) and their seeds are specified in an06 §3;
  `bench_tmj` / `invivo_tmj` (D33, post hoc) test "not a hinge at the TMJ" (axis through the
  condyle points, left minus right), the pre-specified best-fit axis "no fixed axis at all".
  For the dataset paper the condyle-hinge rows lead C1 (D33), with the D18-plausible subset
  and the Qualisys check (VD21, an06 §4).
- **Cohort** (`stats_kinematics.hinge_noise_tests`, one row per null × group `all` /
  `plausible_condyles`): `n_tested`, `above_n` (the primary "k of n"; a participant is tested
  only when its ratio is finite, `hinge_noise_counts`), the median ratio with its BCa CI, and
  the one-sided Wilcoxon signed-rank test of $\log(r_{\text{obs}}/\hat q_{0.99}) > 0$ over the
  tested participants (`wilcoxon_log_ratio_p`; exact distribution when there are no zeros or
  ties, else the normal approximation, `wilcoxon_log_ratio_exact`; `stats.families.wilcoxon_greater`,
  §2.1, §4.3), plus the median
  and largest per-participant Monte-Carlo p (`p_mc_median`, `p_mc_max`). The
  `hinge_noise_summary` row (primary) is unchanged. No multiplicity adjustment: the primary
  decision is the pre-specified k of n; every other row is reported as a sensitivity. Values
  (`r17a`, 13 adults): primary 11 of 13 (ratio 1.18 [1.09, 1.46], p 0.009); slow 6 of
  13 (p 0.55); motion 2 of 13 (p 0.99); teeth-only 11 of 13 (p 0.0085); condyle hinge in-vivo
  12 of 13 (3.91, p 0.0002), bench 13 of 13, plausible 10 of 10 (p 0.0010). an06 §3 lists the
  n = 14 values.
- **Calibration test** of the null: an06 §3 (`tests/test_r10_c1_null.py`).
- **Condyle dependence** (D25, `stats_kinematics.condyle_sensitivity`): every condyle-based
  row (condyle metrics and jaw geometry, the incisal residual of each hinge model including
  `hinge_teeth`, the translation as `translation_max_mm` (larger side, the requirement's
  value), `translation_mean_mm` (mean of the sides) and `translation_range_mm` (τ of the
  rotation-plus-translation fit; D10: not identifiable, a bracket only), the condylar
  residual and the helical-axis distance `dist_tmj`) as median with BCa CI for all
  participants, the D18-plausible ones, those without anatomy (`condyles_unknown`, when any)
  and the D29 depth neighbours (`plausible_condyles_30mm`, `_50mm`).

**Paper wording.** "For each participant, the observed hinge residual was called above noise
when it exceeded the 99th percentile of 1000 simulations of a perfect hinge under that
participant's own measurement noise; Monte-Carlo p values are given alongside. Across
participants we report the number above noise and a one-sided Wilcoxon signed-rank test of the
log ratio of observed residual to the null's 99th percentile."

## 15. Camera-free study (X5; stage `camerafree_cohort`, `stats/cohort_camerafree.py`)

- **Per participant:** the median over the scored tasks of each task's RMSE (opening angle,
  geodesic rotation; incisor over all scored tasks, the rotational and the translational
  tasks).
- **Cohort:** median with 95 % BCa CI (`describe`, `n_boot` = 4000, `DEFAULTS`) and the
  verdict against the bound $b$ (opening 1.2°, incisor 2.0 mm; geodesic without bound),
  checked in this order (`verdict`):

  $$
  \text{verdict} =
  \begin{cases}
  \textit{within} & U < b\\
  \textit{exceeds} & L > b\\
  \textit{compatible} & \tilde x < b\\
  \textit{inconclusive} & \text{otherwise,}
  \end{cases}
  $$

  with $[L, U]$ the BCa CI and $\tilde x$ the median.
- **Paired ladder contrasts** (`contrasts`, `stats.families.paired_contrast`, §2.3):
  two-sided Wilcoxon signed-rank (exact without ties, normal approximation with ties),
  Hodges–Lehmann estimate with its exact 95 % CI, `n_a_better` = #(a − b < 0), Holm within the
  rotation family (E2a−E1, E2b−E2a, E3−E2a on the opening RMSE) and the translation family
  (E4own−E4gen, E4gen−E4hinge, E5−E4own, E6−E4own on the incisor RMSE); H2 (E3's drift slope
  against the anchor clock vs E2a's against the reference clock, |slope|) as its own family.
- **Drift** (`drift`): per participant and clock the OLS slope of the signed opening error
  against time (°/min), $\hat\beta = \sum_t (t - \bar t)(e_t - \bar e)/\sum_t (t - \bar t)^2$
  (`np.polyfit`, ≥ 10 samples and a non-zero time span), time since the reference and since
  the last anchor; cohort medians; a random-intercept mixed model per estimator and clock
  (`err_open_deg ~ t_min`, §6; descriptive: 1 Hz samples are autocorrelated, so its SE is
  optimistic).
- **Cycle amplitudes:** repeated-measures Bland–Altman with the participant-cluster bootstrap
  CIs per estimator, quantity and task group (§9); `loa_within_bound` = max(\|LoA\|) ≤ bound.
- **X5b / X5c** (oracle analyses, never camera-free results): the same per-participant task
  medians for the decomposition rows (optical rotation + E4own model; E3 + optical
  translation) and the task-conditioned model; cohort medians with BCa CI (no bound verdict);
  paired contrasts X5b_rotation − X5b_model and X5c − E4own, Holm within each.
- **E4gen** is trained leave-one-subject-out on every other participant of the store with an
  optical calibration block (`stats/cohort_camerafree.py:85-109`; `exclude_generic` empty; E32):
  in the analysis of record that includes the participant recorded outside the protocol, in the 13-adult paper store the 12 other adults. E7 is not computed (rows say why). Values (`r17a`, 13 adults, median
  [BCa CI]; pre-specified translation primary E4own in open-close and chewing 4.4 [3.7, 6.5] mm,
  exceeds; E6 5.0 [4.0, 6.0] mm after E40): opening RMSE E1 0.83 [0.57, 0.93]°, E2a 0.79 [0.72, 0.87]°, E3b 0.90 [0.79, 1.05]°
  (all *within* 1.2°); incisor RMSE E4own 5.3 [3.6, 6.1], E4gen 5.2 [4.5, 6.1], E4hinge 5.0
  [4.5, 6.8] mm (all *exceed* 2.0 mm); contrasts E3−E2a 0.18° (Holm p 0.051), E5−E4own 333 mm
  (p 0.001), drift E3−E2a 2.78 °/min (p 0.001); X5b model 2.5 [1.9, 2.8] mm, rotation 2.0
  [1.4, 2.6] mm; X5c 4.8 [3.7, 5.9] mm, X5c − E4own 0.04 mm (p 1.0). E4gen with that participant in the
  training set (`r17`): 5.5 mm.

**Paper wording.** "An estimator was called within the accuracy bound when the upper limit of
the 95 % BCa confidence interval of the cohort median RMSE lay below the bound, and exceeding
it when the lower limit lay above; otherwise the median is reported as compatible with the
bound or inconclusive."

## 16. Other procedures

### 16.1 Rank association with fixed effects (impedance, D21; `qc/impedance.py:associations`)

**What and why.** Does the pre-recording impedance check predict the recorded rest signal?
Channels are nested in participants and muscles, so a pooled correlation mixes between-
participant and between-muscle differences into the association; the slope with both held
fixed is the one to read.

Over the $n$ channels of the `motion` take with both values ($\ge 6$, ≥ 2 distinct values
each):
- pooled Spearman $\rho$ (`scipy.stats.spearmanr`) with a 95 % participant-cluster percentile
  CI (2000 resamples, §1.3; one generator from the study seed shared by the relations in
  order, see Seeds);
- within-participant and within-muscle rank correlation: the pooled ranks $r(x)$, $r(y)$
  centred by their group means, then the Pearson correlation of the centred ranks;
- the **rank slope with fixed effects** (`_adjusted`; `rho_adj`, a slope, not a correlation):
  OLS of $r(y)$ on $r(x)$ with participant and muscle dummies (drop-first),

  $$
  r(y)_c = \beta_0 + \beta_1\,r(x)_c + \sum_{p} \gamma_p\,\mathbb 1[\text{pid}_c = p] + \sum_{m} \eta_m\,\mathbb 1[\text{muscle}_c = m] + e_c,
  $$

  $\mathrm{SE}(\hat\beta_1) = \sqrt{\hat\sigma^2 [(A^\top A)^{+}]_{11}}$,
  $\hat\sigma^2 = \mathrm{RSS}/(n - \operatorname{rank} A)$, two-sided *t* test on
  $n - \operatorname{rank} A$ df (`p_adj`), Holm within `check_vs_recorded` and
  `check_vs_check` (`holm_family`, `p_holm`).

Second − first take: per participant the median over channels of $Z_2 - Z_1$, two-sided
Wilcoxon over participants (`wilcoxon_exact` → `families.wilcoxon_signed`, zeros dropped;
exact when there are no ties, normal approximation otherwise; NaN below 6 values because the
smallest attainable two-sided exact p at n = 5 is 1/16).

**Paper wording.** "Associations between the impedance check and the recorded signal were
estimated as rank-regression slopes with participant and muscle as fixed effects, with
Holm-adjusted p values; the pooled Spearman correlation is reported for comparison."

### 16.2 C2-E per-participant primitives (`timing/locking.py`; definitions in an05 §3.3)

- **Locking:** $\mathrm{IQR} = Q_{0.75} - Q_{0.25}$ of the usable per-cycle leads;
  $\ell = \log(\mathrm{IQR}_{\text{obs}}/\operatorname{median}_s \mathrm{IQR}^{S1}_s)$ over the S1
  surrogates with ≥ `min_n` leads; `p_lock` Monte-Carlo ("less", §3.1).
- **Location (mixture):** maximum likelihood of

  $$
  f(\ell) = \pi\,\mathrm{TN}(\ell;\mu,\sigma,[\ell_{lo},\ell_{hi}]) + (1-\pi)\,f_{bg}(\ell),
  $$

  TN the normal truncated to the search window, $f_{bg}$ the density of the S1 surrogate leads
  (Gaussian KDE, bandwidth 20 ms, reflected at the window edges, mixed with 2 % uniform);
  $\sigma \in [5, 150]$ ms; start from a grid (μ every 5 ms, 8 log-spaced σ, π by 25 EM steps),
  then L-BFGS-B; μ is NaN when $\hat\pi < 0.1$ (`no_locked_component`) or with < 5 leads.
- **Release slope:** Theil–Sen slope $b = \operatorname{median}_{i<j,\ x_i \ne x_j} (y_j - y_i)/(x_j - x_i)$ (Sen 1968) of $(t_{\text{EMG}} - a)$ on $(t_{\text{mech}} - a)$,
  $a$ the S2 anchor; ≥ 3 cycles.

### 16.3 Cohen's κ (`labels/kappa.py:kappa`)

From the joint proportions $m_{gh}$ of the two labellings over frames labelled by both:
$\kappa = (p_o - p_e)/(1 - p_e)$, $p_o = \sum_g m_{gg}$, $p_e = \sum_g m_{g\cdot}m_{\cdot g}$
(Cohen 1960); κ = 1 when $p_e = 1$.

### 16.4 Coupling summary (X9; `stats/cohort_kinematics.py:coupling_summary`, stage `stats_kinematics`)

Post hoc description, no test (definitions in an02 §7). Per participant, from the median
curve of the anterior condyle-midpoint translation `tmj_mid_ant_mm` on the opening-angle grid
(angles with ≥ 3 cycles): `slope_mm_per_deg` = Theil–Sen slope (as §16.2) over
`COUPLING_SLOPE_DEG` = 5–15° (≥ 3 grid points); `generic_rms_mm` = RMS difference from the
leave-one-subject-out cohort median curve (an angle needs ≥ max(3, ⌊(N−1)/2⌋) of the other
participants; ≥ 3 shared angles); with `angle_max_deg`, `n_angles` and `condyles_plausible`.
It rests on the palpated condyle points (D18 flag carried along).

## 17. Agreement statistics of the systems study (`validation/agreement.py`, `validation/sync.py`; an14)

### 17.1 Within-participant Bland–Altman over cycles (`ba_boot`, `bias_t_interval`, `n_effective`)

Per participant, task and feature, $d_j$ = JTS-MM − Qualisys of cycle $j$ in time order (cycle
start frames); ≥ 3 cycles. Bias $\bar d$, LoA $\bar d \pm 1.96 s_d$. **Bias CI (EV6,
primary):** with the lag-1 autocorrelation
$\hat\rho_1 = \sum_j (d_j - \bar d)(d_{j+1} - \bar d)/\sum_j (d_j - \bar d)^2$ clipped to
$[0, 0.99]$,

$$
n_{\text{eff}} = n\,\frac{1 - \hat\rho_1}{1 + \hat\rho_1},\qquad
\bar d \pm t_{0.975,\ \max(n_{\text{eff}} - 1,\ 1)}\ \frac{s_d}{\sqrt{n_{\text{eff}}}},
$$

the sample size of independent cycles with the same variance of the mean under AR(1) errors
(a negative $\hat\rho_1$ never narrows the interval). Sensitivities: the cycle percentile
bootstrap that treats cycles as independent (`bias_ci_lo/hi_boot`, 2000) and the moving-block
bootstrap with blocks of 2 cycles (`bias_ci_lo/hi_mbb`, §1.3; drawn from the same generator
after the cycle bootstrap). LoA CIs: cycle percentile bootstrap of $\bar d^* \pm 1.96 s^*_d$.
Seed = study seed.

### 17.2 Equivalence and the equivalence margin

Equivalent to bound $\Delta$ when the 95 % bias CI lies strictly inside it:
$-\Delta < L$ and $U < \Delta$, with $\Delta$ = 0.5 mm / 0.5° (metrology) and 2.0 mm / 1.2°
(application) (`TOL`). This is two one-sided tests (Schuirmann 1987) at level 0.025 each, i.e.
more conservative than the conventional 90 %-CI version. The bound-free **equivalence margin**
(VD13) is the smallest symmetric bound the interval lies inside, $\max(\lvert L\rvert, \lvert U\rvert)$; for waveforms, the maximum over nodes of the 95 % band of the mean difference, which
at every node is the same effective-$n$ $t$-interval as the feature bias (§17.1;
EV14; the cycle-bootstrap band is the sensitivity `equiv_margin_boot`; node 0 excluded for
displacements), and `frac_inside_*` the share of nodes
whose band lies inside the bound.

### 17.3 Lin's CCC, ICC(A,1), Pearson r

- **Lin's concordance correlation** (`ccc`; Lin 1989), with **1/n moments** (EV9):

  $$
  \rho_c = \frac{2 s_{ab}}{s_a^2 + s_b^2 + (\bar a - \bar b)^2},\qquad
  s_{ab} = \frac1n\sum_j (a_j - \bar a)(b_j - \bar b),\ s_a^2 = \frac1n\sum_j (a_j - \bar a)^2,
  $$

  pairs with a NaN dropped, NaN below 3 pairs.
- **ICC(A,1)** (`icc_a1`; two-way, absolute agreement, single measurement; McGraw & Wong 1996),
  $n$ cycles × $k = 2$ systems:

  $$
  \mathrm{ICC}(A,1) = \frac{MS_R - MS_E}{MS_R + (k-1)MS_E + \frac{k}{n}(MS_C - MS_E)},
  $$

  $MS_R = k\sum_j(\bar Y_{j\cdot} - \bar Y)^2/(n-1)$, $MS_C = n\sum_s(\bar Y_{\cdot s} - \bar Y)^2/(k-1)$, $MS_E$ the residual mean square on $(n-1)(k-1)$ df; NaN below 3 cycles.
- Pearson $r$ (pandas `corr`).

### 17.4 Clock standard errors (`validation/sync.py`; an14 §3)

Newey–West HAC variance of a score sum in time order (Newey & West 1987),

$$
\widehat{\mathrm{Var}}\Big(\sum_i u_i\Big) = \sum_i u_i^2 + 2\sum_{l=1}^{L}\Big(1 - \frac{l}{L+1}\Big)\sum_i u_i u_{i-l},
$$

bandwidth $L = \min\big(n - 1,\ \max(\lceil 1.1447(\hat\alpha n)^{1/3}\rceil,\ \lfloor 4(n/100)^{2/9}\rfloor)\big)$,
$\hat\alpha = 4\hat\rho^2/((1-\hat\rho)^2(1+\hat\rho)^2)$, $\hat\rho$ the lag-1 autocorrelation
of $u$ clipped to $[0, 0.97]$ (Andrews 1991; Newey & West 1994). Per window lag
$\hat\delta = -\sum j_i r_i/\sum j_i^2$ with SE $\sqrt{\widehat{\mathrm{Var}}}/\sum j_i^2$; the
clock offset and drift from a weighted least-squares line of the window lags on time with
weights $1/\mathrm{SE}^2$ (no rescaling); with ≥ 3 windows also the delete-one-window
jackknife SE of the drift, $\sqrt{\frac{K-1}{K}\sum_w(\hat b_{(-w)} - \bar b)^2}$.

### 17.5 The n = 2 rule (EV8)

Every systems-study result is reported per participant (P1, P2) plus their mean as a
descriptive summary ("mean (P1, P2)"); no interval, test or equivalence verdict on the mean;
frames or cycles of the two participants are never pooled into one bootstrap or RMS; where the
mean is quoted, the worse participant is stated with it. Participant is a fixed factor. One
exception: the jaw-IMU lever arm (VD15), one physical constant estimated from both sessions.

**Paper wording (systems).** "With two participants, all results are reported per
participant, with their mean as a descriptive summary. Within each participant, the bias CI
used the effective number of cycles under lag-1 autocorrelation; agreement is also given as
Lin's concordance correlation coefficient (Lin 1989) and ICC(A,1) (McGraw & Wong 1996), and
equivalence as the smallest symmetric bound containing the 95 % CI."

## References (standard methods named above)

Andrews DWK (1991) Econometrica 59:817–858. · Berger RL (1982) Technometrics 24:295–300. ·
Berger RL, Hsu JC (1996) Stat Sci 11:283–319. · Bland JM, Altman DG (1986) Lancet
327:307–310; (1999) Stat Methods Med Res 8:135–160; (2007) J Biopharm Stat 17:571–582. ·
Cohen J (1960) Educ Psychol Meas 20:37–46. · David HA, Nagaraja HN (2003) Order Statistics,
3rd ed., Wiley. · Davison AC, Hinkley DV (1997) Bootstrap Methods and their Application,
CUP. · Dmitrienko A, Offen WW, Westfall PH (2003) Stat Med 22:2387–2400. · Donner A, Zou GY
(2012) Stat Methods Med Res 21:347–359. · Efron B (1987) J Am Stat Assoc 82:171–185. · Efron
B, Tibshirani RJ (1993) An Introduction to the Bootstrap, Chapman & Hall. · Friedman M (1937)
J Am Stat Assoc 32:675–701. · Harrell FE, Davis CE (1982) Biometrika 69:635–640. · Hedges LV
(1981) J Educ Stat 6:107–128. · Hodges JL, Lehmann EL (1963) Ann Math Stat 34:598–611. ·
Hollander M, Wolfe DA, Chicken E (2014) Nonparametric Statistical Methods, 3rd ed., Wiley. ·
Holm S (1979) Scand J Stat 6:65–70. · Kadaba MP et al. (1989) J Orthop Res 7:849–860. ·
Künsch HR (1989) Ann Stat 17:1217–1241. · Laird NM, Ware JH (1982) Biometrics 38:963–974. ·
Lin LI (1989) Biometrics 45:255–268. · McGraw KO, Wong SP (1996) Psychol Methods 1:30–46. ·
Newey WK, West KD (1987) Econometrica 55:703–708; (1994) Rev Econ Stud 61:631–653. · Nichols
TE, Holmes AP (2002) Hum Brain Mapp 15:1–25. · Pataky TC (2012) Comput Methods Biomech Biomed
Engin 15:295–301. · Pataky TC, Vanrenterghem J, Robinson MA (2015) J Biomech 48:1277–1285. ·
Phipson B, Smyth GK (2010) Stat Appl Genet Mol Biol 9:39. · Rosenthal R (1991) Meta-Analytic
Procedures for Social Research, Sage. · Schuirmann DJ (1987) J Pharmacokinet Biopharm
15:657–680. · Sen PK (1968) J Am Stat Assoc 63:1379–1389. · Shrout PE, Fleiss JL (1979)
Psychol Bull 86:420–428. · Simonsohn U, Simmons JP, Nelson LD (2020) Nat Hum Behav
4:1208–1214. · Weir JP (2005) J Strength Cond Res 19:231–240. · Westfall PH, Krishen A (2001)
J Stat Plan Inference 99:25–40. · Wilcoxon F (1945) Biometrics Bull 1:80–83. · Zou GY (2013)
Stat Methods Med Res 22:630–642.

## Thresholds and exclusions (D29)

Every rule that drops, masks or flags data. **F** = flags and keeps the data (outcomes also
exist with all data), **R** = removes the data from that analysis. "As coded" gives the
comparison exactly as the code applies it. Status: pre-specified (§ of
`experiments/icorr2027/analysis_plan.md` before the amendments), amendment, deviation code,
or not in the plan. **n affected: history** — counted read-only at the time of the D29
inventory (2026-10-29) on the store frozen then (`r9`; 14 participants, 13 with jaw-IMU data),
before E32–E34 and D23a. These counts are a record of that date, not the current result; the
current counts are in the pinned stores (`r17` / `r17a`: `stats_timing.timing_usability`,
the stage `coverage` tables). Re-counted on the pinned stores and equal to the inventory: the
start gate (301 / 472) and the burst criterion (28 / 472) of the TKEO opener channel-cycles and
the D15 exclusions (`r17a`), and the 3 / 14 implausible condyles (`r17`). The camera-free and
C2-E rows were added later and give the `r17a` values. Neighbours: the D29 sensitivity, never
tuned. The rows marked X6 are the C2-E gates (an05 §3.3), fixed before any real-data run.

| Area | Rule | As coded | Status | Code; setting | F/R | n affected (2026-10-29, history) | Neighbours (D29) |
|---|---|---|---|---|---|---|---|
| IMU | accuracy-0 samples | sample kept if `accuracy != 0` (quaternion and accelerometer streams separately), dropped before resampling | §3.3 ("masked"); no rationale | `imu/preprocess.py` (`load`); hard-coded | R | 0 % of samples in every stream (`qc.imu.acc0_frac` = 0) | — |
| IMU | quaternion / accelerometer gap | NaN where the gap between samples `> gap_s` (0.05 s) | §3.3 "gaps > 50 ms are masked"; no rationale; E33.7 (one parameter) | `imu/preprocess.py` (`GAP_S`, `GAP_US`; `quat_at`, `accel_at`), `validation/fusion.py` (`imu_max_gap_s`); `stages.qc.imu.gap_s` (package default; also the QC coverage) | R | coverage 1.00 in every phase of the 13 participants with IMU data (the participant recorded outside the protocol: no jaw IMU) | — |
| IMU | quasi-static (tilt estimator) | `\| \|a\| − g \| <= 0.05 g`, both IMUs | not in the plan (an04, E23) | `jtsmm/imu/tilt.py` (`quasi_static`); `stages.imu.quasi_static_tol` | F (tilt estimate only) | median share of quasi-static samples per task 0.68 (free motion) – 0.87 (laterotrusion) | — |
| IMU | noise-limited bandwidth task | `coh_band_hz < 2.0` Hz, band = contiguous bins from 0 Hz with bias-corrected C′ `>= 0.5` (0.25 Hz bins); also True when no coherence can be computed | D12 ("< 2 Hz"); the 0.5 coherence threshold only in an04 | `imu/analysis.py` (`MIN_COHERENT_BAND_HZ`, `COHERENCE_MIN`, module constants) | F; the task leaves the bandwidth requirement | 32 / 78 participant × task rows (13 rest rows); in the bandwidth tasks (open-close, chewing, free motion) 3 / 39 | < 1.5 Hz: 2 / 39; < 3 Hz: 13 / 39 — in-stage from `coh_band_hz`, no re-run |
| Optics | gap bridging | gaps of `<= round(0.5 · fs)` frames bridged by SE(3) interpolation | §3.2 "≤ 0.5 s … flagged" | `kinematics/pose_chain.py` (`se3.bridge_gaps`); `signals.kinematics.max_gap_s` | F | lock ≥ 0.93 in every experiment phase (median 1.00); untracked label time 0 s of 1287 s | — |
| Optics | dataset optical quality | none: `lock_frac` is reported only | §3.2 "reported" | `qc/report.py` | report | — | — |
| Optics | systems reference gate (V8) | HP/MP marker missing or gap-filled, `dev_max_mm > 0.5`, `resid_max_mm > 1.0` | validation plan V8 (0.5 mm with a rationale, 1.0 mm without) | `validation/quality.py`; constants | R in the gated analysis, both reported | systems study (n = 2), not in this store | planned "alternative stated in VD13": VD13 states none (see below) |
| EMG | low-frequency rule (D14) | sample hit: 1–20 Hz RMS `> 10 ×` own rest level **and** `> 3 ×` concurrent in-band envelope; channel flag: `mask_lowfreq_frac > 0.10` | §3.1, Amendment A1, D14 (flag since D14) | `emg/artifacts.py`, `qc/report.py`; `signals.emg.mask.lf_*`, `stages.qc.usability.lowfreq_flag_frac` | F | 21 / 84 channels flagged; hit share median 3.2 %, max 54 % | sensitivity run `lowfreq_mask` (exists) |
| EMG | envelope artifact | `env > 3 ×` own MVC/RVC p95 (off where the reference is a task, D11) | §3.1 | `emg/artifacts.py`; `signals.emg.mask.env_factor` | R (samples, dilated 0.1 s) | 0 % of task samples | — |
| EMG | rest usability | `rest_rms_uv > 20` µV → rest family reported separately | §3.1 (rationale given) | `qc/report.py`; `stages.qc.usability.rest_noise_uv` | F | 2 / 84 channels | — |
| EMG | SNR (shape and amplitude) | usable if `snr_db >= 6` dB (20·log10 of amplitudes) | §3.1 "≥ 6 dB" | `qc/report.py`; `stages.qc.usability.snr_min_db` | R from the shape and amplitude families | 0 / 84 | — |
| EMG | reference masking (amplitude) | usable if `ref_mask_frac_effort <= 0.10` (effort samples: envelope ≥ 25 % of the p95) | §3.1 ("< 10 %", whole phase), then D9 (effort samples) | `qc/report.py`; `stages.qc.usability.ref_mask_max` | R from the amplitude family | 0 / 84 | — |
| EMG | channel-cycle masking | `mask_frac > 0.10` → NaN | §3.1 "more than 10 %" | `emg/analysis.py`; `stages.emg.cycle_mask_max` | R | 0 / 1524 channel-cycles | — |
| EMG timing | burst | usable if `burst_db >= 6.0` dB (20·log10 of the 50-ms peak envelope over the window / rest median 150-ms envelope) | §3.1 "≥ 6 dB"; no rationale for 6 | `timing/lead.py`; `stages.timing.burst_min_db` | R (row kept, `usable` False) | TKEO, opening, openers: 28 / 472 channel-cycles below 6 dB | 3 / 10 dB: variants `burst_3db`, `burst_10db` |
| EMG timing | window finite | usable if the finite share of the window `> 0.9` | not in the plan; wording E33.1 ("more than 90 %") | `timing/lead.py` (`WindowOnsets`); `stages.timing.min_finite_share` = 0.9 (a literal until 2026-11-03, a parameter since; not `cycle_mask_max`); the change-point check of C2-E keeps the literal 0.9 (`timing/c2e.py:252`) | R | with the onset-not-found cases: 14 / 472 (first), 26 / 472 (drive) | — |
| EMG timing | active at the window start | median detector signal over the first 50 ms `> threshold` → unusable (threshold of the rule: rest under first / drive, the window's first 0.3 s under drive_taskbase) | not in §3.1, but in the code from the first pipeline commit (before any C2 result; change-log entry J34); first described in the D6 verification; E34 | `timing/lead.py`; 50 ms hard-coded, `stages.timing.k_sd` | R | 301 / 472 opener channel-cycles (64 %) under first and drive; 0 under drive_taskbase | variant `onset_ksd_5` (rest + 5 SD, the same threshold this gate uses; post hoc, J3); drive_taskbase covers it in part |
| EMG timing | D15 minimum per participant | participant in the test if `n_use >= 3` usable channel-cycles, pooled over the group's muscles; the D23 null needs `n_null_usable >= 3` too | D15 (results seen; no rationale for 3) | `stats/cohort_misc.py`; `stages.stats_timing.min_cycles` | R from the test, counted (`n_excluded_min_cycles`) | open-close / chewing, TKEO opening lead: first 4 / 4 of 13 excluded, drive 5 / 5, drive_taskbase 0 / 2 | ≥ 2: first 12 / 13 kept (open-close), 13 / 13 (chewing); ≥ 5: 8 / 3 — variants `d15_min_cycles_2`, `d15_min_cycles_5` |
| EMG | relevance (D19; VD11 in the systems study) | cell active if task p95 `>= max(rest mean + 3 SD, 5 %)` | D19 (fixed before values); VD11; EV11 | `emg/relevance.py`; `stages.emg_relevance` (the validation call reads the same block since EV11) | F | 50 / 420 participant × task × muscle cells inactive (32 below 5 %, 18 below rest + 3 SD) | — |
| Kinematics | cycle validity (`ok`) | flagged if untracked share `> 0.1`, duration `< 0.4` s, at a phase edge, or `\|start − end\| > 0.2 ×` (peak − lower end) | E9 (erratum), an02 | `kinematics/cycles.py`; `stages.kinematics.cycles` | F, left out of the statistics | 35 / 698 cycles (edge 19, no return 11, short 3, edge + no return 2) | — |
| Kinematics | condyle plausibility (D18 and checks) | plausible if side-mean depth `>= 40` mm and `min(dist) >= 70` mm and `min(height) >= 5` mm | 70 mm pre-specified; 5 mm E13; 40 mm D18 (results seen; tolerance budget) | `kinematics/pose_chain.py`, `frames/landmarks.py`; `signals.kinematics.min_condyle_*` | F (all + plausible subset) | 3 / 14 implausible (s02 too shallow, s03 too close, s11 below the incisal level) | 30 mm: 3 / 14; 50 mm: 5 / 14 (s06, s08 added) — in-stage groups in `condyle_sensitivity`, `hinge_noise_tests` |
| Decoding | guard 0.35 s | blocked CV: training windows `near <= 0.35` s dropped; few-shot and online: scored if `> 0.35` s after the last calibration window | D4 (blocked CV only; no rationale); few-shot and online not in the plan | `ml/protocols/folds.py` (hard-coded), `ml/protocols/variants.py`, `ml/online/decoder.py`; `stages.ml_eval…few_shot.guard_s`, `stages.ml_online.calibration.guard_s` | R | not stored | — |
| Decoding | label smoothing | enter motion `> 0.20 ×` the phase's p95 speed, leave `< 0.10 ×`, segments `< 0.10` s merged | not in the plan (an05) | `labels/auto_labels.py`; `stages.labels` | relabels | — (D16 agreement: `labels.label_agreement`) | sensitivity run `labels_p95` (axis floor) |
| Decoding | online smoothing | enter with `>= 2`, hold with `>= 1` of the last 3 windows | X4 (fixed before) | `ml/online/decoder.py`; `stages.ml_online.smoothing` | relabels | — | — |
| Decoding | untracked labels (−1) | non-finite incisor → −1; windows with label `< 0` dropped | not in the plan | `labels/auto_labels.py`, `ml/core/dataset.py` | R | 0 s untracked label time; 1 / 23 754 windows dropped for the label | — |
| Decoding | masked windows | window dropped if any channel has a masked share `> 0.2`; also non-finite features | not in the plan | `ml/core/dataset.py`; `stages.ml_windows.window.max_masked` | R | 0 masked; 1999 / 23 754 non-finite | — |
| Camera-free | closed-pose detector | closed if θ `<=` rest-phase P90 + 1°, relative angular speed (causal 0.1 s mean) `< 5` °/s, for `>= 0.3` s | X5 (the synthetic gate; method details in the change log) | `imu/camerafree.py` (`CLOSED`); `stages.camerafree.closed` | F (anchors only) | — (`r17a`: median precision over participants 0.125, `cf_summary.det_precision`) | — |
| Camera-free | ψ identifiability | relative heading not fitted when the head-tilt range `< 10°` | X5 (an04 §7) | `imu/camerafree.py` (`PSI_MIN_TILT_RANGE_DEG`) | F | — | — |
| Camera-free | X5c class model | a movement class with `< 2 s` of free-motion frames is not fitted (row with the reason) | X5c (fixed before computing) | `imu/stage_camerafree.py` | R (that class) | — (`r17a`: every class fitted) | — |
| C2-E (X6) | usable channel-cycles per participant | test if `n_usable >= 5`; 3–4 flagged and counted (`n_flagged`) | X6 (before computing) | `timing/c2e.py`, `stats/cohort_c2e.py`; `stages.timing.c2e.min_usable`, `flag_usable` | R from the cohort test, counted | `r17a`: H1 12 tested + 1 flagged of 13; H2 12 + 0; H3 7 + 4 | — |
| C2-E (X6) | mixture locked share | μ NaN if π `< 0.1` (`no_locked_component`); fewer than 5 leads: `few_leads` | X6 | `timing/locking.py`; `stages.timing.c2e.mixture.pi_min` | R (μ only) | `r17a`, H1 (ω): `no_locked_component` 2 of 13, `few_leads` 1 | — |
| C2-E (X6) | changepoint usable (P6) | change found `>= 50` ms from both edges, variance ratio `>= burst_min_db` (6 dB), finite share `> 0.9` | X6 | `timing/changepoint.py`, `timing/c2e.py`; `cp_min_seg_s`, `stages.timing.burst_min_db` | R (that row) | `r17a`: 316 of 944 closer closing channel-cycles usable (ω rule: 279) | burst 3 / 10 dB variants reach it |
| C2-E (X6) | observed window inside the phase | cycles whose window [t − 0.6 s, t + 0.4 s] (asymmetric and symmetric windows) leaves the phase are not evaluated | X6 (implementation) | `timing/c2e.py` (`inside`) | R | not counted (cycles not evaluated are not stored) | — |
| Systems | cycle return rule | `\|start − end\| > 0.2 ×` amplitude → flagged, in `ref_agreement` and `ref_emg` | validation plan (dataset rules); EV10 (was 0.3 in `ref_agreement`) | `validation/agreement.py` (`find_cycles`, `return_frac` 0.2) | F, left out | at the time of EV10: the cycle tables stayed identical, no comparison cycle dropped | — |

**Consistency check (D29: same number, unit and direction in code, settings, plan, captions).** Consistent: burst ≥ 6 dB, D18 ≥ 40 mm (code, settings, D18, D29), noise-limited
< 2 Hz (D12, D29, an04, catalog), bridging ≤ 0.5 s, rest > 20 µV, SNR ≥ 6 dB, cycle masking
> 10 %, low-frequency flag > 10 %, V8 > 0.5 / > 1.0 mm, D19 "at least", the 0.35 s guard.
The 16 mismatches found by the inventory (2026-10-29) and their status (erratum E33 of the
dataset plan, EV10 / EV11 of the validation plan):

| # | Mismatch | Status |
|---|---|---|
| 1 | window finite: "at least 90 %" vs the code's `> 0.9` | E33.1: wording "more than 90 %" |
| 2 | D15 unit: "cycles" vs the code's channel-cycles (also the D23 null gate on pseudo channel-cycles) | E33.2: wording "channel-cycles" |
| 3 | hard-coded "3", "6 dB" in captions and catalog text | E33.3: generated from the parameters |
| 4 | reference masking: "< 10 % of the phase" vs ≤ 0.10 of the effort samples (D9) | E33.4 |
| 5 | low-frequency rule: power vs RMS amplitude | E33.5 |
| 6 | accuracy-0 samples "masked" vs dropped | E33.6 (an04 was correct) |
| 7 | 50 ms IMU gap: two unlinked constants | E33.7: one parameter `stages.qc.imu.gap_s` (code change, no value change) |
| 8 | E9 return rule: "baseline" vs amplitude | E33.8 |
| 9 | systems study: `ref_agreement` return fraction 0.3 vs 0.2 | EV10: 0.2 in both (no comparison cycle was dropped by the change) |
| 10 | long optical gaps: metric vs whole cycle | E33.9 |
| 11 | D18 depth check not documented | E33.10 recorded |
| 12 | V8 neighbour "stated in VD13" does not exist | E33.14: sensitivity dropped |
| 13 | noise-limited definition incomplete | E33.11 recorded |
| 14 | few-shot k: "per class" vs windows per phase | E33.12 |
| 15 | D19 and VD11 not linked | EV11: `validation/emg.py` reads `stages.emg_relevance` |
| 16 | burst wording: "in that cycle" vs the whole-window peak | E33.13 |

**Largest exclusion** (E33, E34): the active-at-start gate under `first` and `drive` (`r17a`:
301 of 472 TKEO opener channel-cycles, 64 %; 59 % in open-close and 70 % in chewing, E34); it
practically never fires under `drive_taskbase`; the threshold-free changepoint onsets of
C2-E (P6) recover these channel-cycles as a detector sensitivity.

Without a stated rationale: accuracy-0, the 50 ms gap, quasi-static tolerance, bridging, the
6 dB value, cycle masking, window finite, active-at-start, D15's 3, the cycle rules, the
guard, label smoothing, the untracked and masked-window rules, the 0.5 coherence threshold and
the 1.0 mm part of V8.
