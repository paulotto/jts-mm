<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an10 — Data atlas and participant report

The data atlas shows the recorded data itself, next to the analysis plots (22 views). Each view is a
`FigureSpec` builder in `biomech/viz/views/data.py`, registered with kind `data`. The same spec
renders in three places:

- the viewer's *Atlas* tab (Plotly);
- quick-look PNG/PDF (matplotlib);
- TikZ/pgfplots for LaTeX (`biomech report`, export).

```bash
biomech report --config experiments/icorr2027/study.yaml --pid s03 --compile
# -> data/analysis/icorr2027/reports/s03/report_s03.tex (+ .pdf, figures/, tables/)
```

## Conventions

- **Frame.** Coordinates are in the maxillary frame: x anterior, y left, z up, with the origin at
  the upper incisor. This frame is occlusal-plane-like, not Frankfort. Condyle points are
  skin-palpated (an02 §2; A4, change log).
- **Pseudonyms.** Participants appear only as P1…Pn (`DataContext.pseudonym`). Figures never
  show pid strings, age or sex.
- **Perspectives.** Views with trials take `perspective` / `align` / `stat`:
  - `trials`: every cycle, coloured by order on the sequential ramp (early = light); flagged
    cycles are dashed grey.
  - `trials+summary`: the cycles in grey plus the participant's centre and spread.
  - `participant`: one centre per participant, with the selected participant highlighted.
  - `cohort`: participant centres in grey plus the cohort centre and spread.

  Every summary comes from `ensembles.aggregate` over `ok` cycles. Cohort statistics are taken
  over participant centres. The caption states the perspective and n.
  - **Participant centre:** the mean over the participant's cycles; the median for
    `median_iqr`.
  - **Statistics:**
    - `mean_sd`: mean ± SD (ddof 1);
    - `median_iqr`: median with the 25th–75th percentiles;
    - `mean_ci`: mean with a percentile bootstrap 95 % CI, 1000 resamples over cycles
      (participant level) or over participant centres (cohort).
  - **Cohort coverage:** a cohort value at a grid point and channel needs finite centres from at
    least max(2, ⌈0.5·P⌉) participants, with P the participants in that ensemble. Otherwise it
    is NaN, so the plot never shows one person's curve as the cohort.
  - **Within-participant SD:** √(mean over participants of the within-participant variance),
    returned alongside the cohort value.
- **Cycles.** There are two cuts. Both are resampled to 101 points.
  - Kinematic ensembles (`ensembles.kinematic`, align `cycle` / `path` / `time`) cut at movement
    onset..offset, `t_on_us`..`t_off_us`. These are the points where the incisal speed crosses
    5 % of its peak, searched outward from the opening and closing speed peaks. If they are
    missing, the cycle start..end is used. `data.custom_ensemble` applies the same rule to any
    per-participant signal, so several signals of one cycle stay aligned (e.g. a path coloured
    by activation).
  - EMG profiles (`emg_profiles`, and with them the SPM tests, ANOVA, CMC and variance ratio of
    an07) and the `angle` alignment cut at the cycle start..end (`t_start_us`..`t_end_us`).
  - Fig. 2a and `data_emg_cycle` take the opening angle from `ensembles.profile_ensembles`.
    That is the angle stored with the EMG profiles, so the angle and the activation panels show
    the same cycles with the same start..end cut.
- **Colour.**
  - One sequential ramp for magnitudes: speed, activation, density, trial order.
  - The fixed categorical order for tasks.
  - Side hues for left/right.
  - Accent over grey for summaries.
  - Reference lines are the only dashed lines.
- **Exploratory numbers.** Numbers outside the frozen plan are marked *exploratory* in the
  caption: demand percentiles, condylar inclinations, median frequencies and the optical-IMU
  residual.
- **Missing data.** A missing modality, array or session gives an annotated empty panel, never
  an exception. For example, a store from before the derivative arrays shows "not in the store
  (re-run kinematics)".

## Reproducible without the session

Every data view except two draws only from the result store, so it renders from a frozen
snapshot without the recorded sessions (`biomech view --snapshot NAME`). The processing-chain
and spectra views (`data_emg_chain`, `data_emg_spectra`) read the products the `emg` stage
stores (`mains__*`, `chain__*`; an03 §5), the IMU-quality, timeline and tracking-scene views read
the `qc` stream products (`imu_intervals`, `imu_coverage`, `timeline_*`, `scene__*`; an01 §3).
When a request is outside what is stored (e.g. a chain window other than the stored 2 s, or
an older store) they fall back to the session if it is available. `data_imu_accel` and
`data_landmarks` read raw samples and are registered `needs_session=True`: without the session
they show a card saying so, and `check-outputs` lists them. Checked on a snapshot with the
session directory unreachable: every other view builds, the two session views show the card.

## Views

| id | Form | Perspectives |
|---|---|---|
| `data_orthographic` | sagittal (Posselt-type convex-hull envelope), frontal, horizontal (gothic-arch-type) projections of the incisal point, one shared mm scale | trials (raw, all tasks) · trials+summary (mean path + 1-SD ellipses at 10 % steps) · participant · cohort |
| `data_jaw3d` | 3-D paths of incisor (by task), molars, condyle points; maxillary landmarks; ghost mandible at closure and maximal opening | as above |
| `data_speed_path` | sagittal + frontal incisal path coloured by tangential speed (colorbar mm/s) | trials (raw) · participant · cohort (mean path coloured by mean speed) |
| `data_condylar_paths` | axiography-style condyle-point displacement: opening and protrusion (sagittal, with inclination), laterotrusion (horizontal, side shift); left/right hues | trials · trials+summary · cohort |
| `data_cohort_atlas` | small multiples of every participant's sagittal envelope on identical axes, plus an overlay with the median envelope (median hull radius per direction) | compare |
| `data_activation_path` | sagittal path coloured by digastric (openers) and masseter (closers) activation (% of max recorded); the stroke in which a group acts is drawn on top | trials (raw) · trials+summary · participant · cohort |
| `data_kin_timeseries` | incisal point and mandibular rotation over the whole take, every recorded segment coloured and labelled (phase lane on top) | trials (time) |
| `data_orient_pairs` | rotation components against each other (opening vs lateral and vs tilt), one colour per recorded segment | trials |
| `data_kinematic_stack` | θ, θ̇, θ̈ and incisal displacement, speed, \|a\| as shared-x rows, with opening segments shaded | trials (time) · all perspectives on the cycle grid |
| `data_cycle_profiles` | cycle-normalized θ, peak-normalized θ̇ with the minimum-jerk reference, θ̈ | all (default cohort, median/IQR) |
| `data_phase_portrait` | θ vs θ̇ and opening vs opening velocity, one loop per cycle | all |
| `data_demand_map` | per task, the 2-D density of velocity vs acceleration (angular; incisal speed vs tangential acceleration), with the 95 %-of-frames iso-line and the p95 limits | trials (participant) · cohort (equal participant weight; p95 = median of the participants' p95) |
| `data_imu_accel` | relative linear acceleration of the mandible: IMU (gravity-free, mandible minus head) vs optical prediction, residual, raw \|a\| − 1 g | time |
| `data_emg_chain` | raw → mains removed → band-passed → envelope (% of max recorded), one channel over 2 s, masked samples shaded | time |
| `data_emg_spectra` | Welch PSD per channel, raw vs mains removed, mains harmonics, median frequency | time |
| `data_activation_raster` | muscles × time heatmap of the activation (% of max recorded, mean per bin) over every task, masked bins grey | time |
| `data_emg_cycle` | opening angle and L/R envelopes of digastric, masseter and temporalis over the stored EMG cycle profiles | all |
| `data_imu_estimators` | IMU opening angle (quaternion, tilt, complementary) vs optical, the difference, dθ/dt | time |
| `data_imu_quality` | sample-interval distribution (1st–99th percentile, median, longest; % long intervals) and coverage per phase | time |
| `data_timeline` | swimlanes on the master clock: protocol phases, EMG, IMU streams, optical lock per body with the reprojection RMS, camera previews, triggers | time |
| `data_tracking_scene` | camera-frame poses of CRA, MTA and CT pointer tip with triads; reference camera at the origin (no second camera: no calibration in the session) | time |
| `data_landmarks` | CT-tip cloud per hold along its principal axes, median and MAD (matches the kinematics stage); optional camera frame per hold (`filmstrip=True`) | time |

**Several tasks in one view.** `data_orthographic` and `data_jaw3d` are registered with
`multi_phase=True`. Their phase can be one task, a list of tasks or `"all"` (every task
present); each task is drawn in its task colour (`TASK_ROLE`), with a legend entry per task. The
cycle-ensemble perspectives (trials+summary, participant, cohort) still use one task: the first
of the list. The registry normalizes the argument (`registry.normalize_phase`): duplicates are
dropped, the tasks are put in task order, a list of every task becomes `"all"`, and a
one-element list becomes the plain name. The exported id therefore stays stable whatever the
order of selection: `data_jaw3d--phase-all`, `data_jaw3d--phase-open_close+chewing`. The `+`
is kept literally in the phase token; other parameters still spell it `plus`. A view without
`multi_phase` that is given a list gets its first phase. The explorer's *Atlas* tab passes
the phase selector's choice (several phases, "all motions") to these two views.

## Design choices

- **Orthographic set.** The planes and the 3-D view are two views (`data_orthographic` 1 × 3,
  `data_jaw3d`), not one 2 × 2 figure. With one shared mm scale, a 2 × 2 grid leaves the narrow
  frontal and horizontal cells mostly empty and the 3-D panel too small.
- **Optical vs IMU acceleration.** The IMU calibration holds only rotations (no sensor lever
  arm), so the IMU's specific force at the sensor cannot be predicted. The view therefore
  compares the IMU's gravity-free *relative* linear acceleration (mandible minus head sensor)
  with the second derivative of the mandibular-array origin. Both are low-passed at 6 Hz, and
  the residual includes the unknown lever-arm terms. No gyroscope was recorded, so the rate row
  is dθ/dt of the quaternion estimate.
- **Sample intervals.** These are shown as a percentile strip per stream rather than a
  histogram, because stable clocks give one-bin histograms.
- **Filmstrip.** The camera frames of the landmark holds show the participant's face. They are
  therefore opt-in (`filmstrip=True`) and enabled only in the participant report, which stays
  under the git-ignored `data/analysis/`.

## Participant report

`biomech report --pid sXX [--out DIR] [--compile]` (`biomech/report.py`) writes the following
under `<out_dir>/reports/<pid>/`:

- every data view (TikZ), in sections: kinematics in space, kinematics in time, EMG, IMU,
  tracking and synchronization (views not assigned in `report.SECTIONS`, e.g.
  `data_kin_timeseries` and `data_orient_pairs`, go to "Other views");
- the key tables (clinical kinematics per task, landmark MAD, EMG references and quality, IMU
  agreement, EMG onset lead per detector);
- `report_<pid>.tex`.

`--compile` (default off) compiles it with LuaLaTeX. A view that fails is listed under "Views
that could not be built" instead of stopping the report. The landmark view carries the
camera frames of the holds (`filmstrip=True`), so the report is identifiable subject data and
stays within the study team.

## Tests

`software/biomech/tests/test_views_data.py` runs on the synthetic study (`synth_study`
fixture). It checks the following:

- every view × perspective renders with no text overlaps and a clean layout, in matplotlib and
  Plotly, without pid strings;
- a pgf sample compiles with LuaLaTeX within 3.5 / 7.16 in;
- the plane projections equal the 3-D coordinates;
- the envelope contains every path point and spans the true opening;
- the speed colours equal ‖v‖;
- the minimum-jerk overlay integrates to the amplitude;
- the demand-map p95 equals `np.percentile`, and the cohort p95 equals the median of the
  participants' p95;
- the cohort perspective equals `ensembles.aggregate`;
- the landmark MAD equals the kinematics stage;
- a missing modality (kinematics, EMG, IMU, derivative arrays, session files) gives annotated
  panels.
