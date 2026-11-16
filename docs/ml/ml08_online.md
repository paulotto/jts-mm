<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml08 — Pseudo-online evaluation (`ml_online`, X4)

How well would the intent decoder and the X3 closer-release detector work if they ran in real
time? No exoskeleton exists, so this evaluates *intent detection only*: recorded takes are
replayed chunk by chunk through a strictly causal chain ("simulated real-time"). It is an
exploratory analysis (analysis_plan §1; X4 in the change log).

> **Status:** implemented — `ml/online/` (`stream`, `decoder`, `release`, `metrics`, `stage`);
> biomech cohort stage `ml_online` v1 (depends on `ml_windows`, `ml_eval`, `timing`, `labels`);
> tests `software/ml/tests/test_online.py`; views on the viewer's Results page (domain *Intent
> decoding*: `online_accuracy`, `online_timeline`, `online_compute`, `online_release`,
> `online_feature_deviation`). Pilot results: §8.
>
> `biomech run --config experiments/icorr2027/study.yaml --stage ml_online`

## 1. What is shown, and what is not

Shown: the decisions an online decoder would have made on the recorded sensor streams, with the
information it would have had at each moment, and the compute time per chunk on this machine.

Not shown:

* **Closed loop.** Nobody wore an actuated device; there is no feedback of the decisions on the
  movement, no actuation delay, no user adaptation to the decoder. The latencies are those of the
  decision, not of an assistance.
* **Transport.** Samples "arrive" at their master-clock stamp; radio/USB delays and the BNO085
  report jitter are not simulated. The IMU streams are the sanitized ones (sorted, duplicate
  stamps dropped) as offline; online a sample with an out-of-order stamp would arrive late.
* **Refit time.** Calibration refits are idealized as instantaneous at the end of the
  calibration windows (their fit time is reported, `cal_fit_s`).
* **Embedded hardware.** Compute times are from the analysis PC (one thread, but other
  processes share the machine); they bound the order of magnitude, not a Raspberry Pi.

## 2. The causal chain (`ml.online.stream`)

The take is cut into chunks of `chunk_s` worth of EMG samples (20 ms = 80 samples at 4 kHz).
Each chunk carries the IMU samples stamped at or before its last EMG sample. The processor
(`OnlineProcessor`) only ever receives chunks; it never holds the stored arrays.

| Step | Online | Offline (`ml_windows`) |
|------|--------|------------------------|
| Stream | starts `warmup_s` (2 s) before the rest phase, runs to the end of the last windowed phase | per phase, from `warmup_s` before it |
| Mains frequency | measured on up to 60 s **before** the stream start (`line_hz: preroll`; nominal if < 1 s) | measured on the first 60 s of the take's phases (`offline`) |
| EMG filter | one stateful `CausalEmgFilter` (tracking mains canceller + 20-450 Hz) from the stream start (`filter_mode: continuous`) | a fresh filter per phase, 2 s warm-up discarded |
| Canceller re-lock | at recording gaps only (as `process_timed` on a whole block) — makes the output independent of the chunk size | same |
| Masks | rail (`|x| >= 0.99` full scale, non-finite) and lead-off, pointwise; grown **causally** by `dilate_s` (a masked sample masks the next 0.1 s, never earlier ones) | biomech's rail/lead-off masks grown on **both** sides |
| Masked raw samples | held at the last valid value before filtering, NaN after | same |
| Calibration | at the end of the rest phase: rest RMS (the "% reference"), ZC/SSC/WAMP thresholds, D13 rest-relative statistics from the rest windows, closer rest level for X3. Checked: the rest phase ends before every task starts (`rest_precedes_tasks`) | same statistics from the rest phase; reference from the MVC/task data (D13 features do not depend on it) |
| Windows | trailing, on the offline grid (ends `win - 1 + k stride` samples after a phase's first sample, inside the phase); emitted once their EMG has arrived and every IMU stream has a sample at or after the last IMU grid point (`imu_max_wait_s` at most, then `imu_late`) | all at once |
| Features | `emg_window_features`, `imu_signals` (on the phase-anchored 200 Hz grid, interpolated between bracketing samples), `imu_window_features`, `apply_rest_relative` — the offline functions on the same samples | same |
| Labels | attached afterwards for scoring (`window_labels`, end-of-window rule); never an input except the calibration windows' labels | same |

The rest windows are produced (and decoded) when the rest phase ends: before that the statistics
that normalize them do not exist. They are `retro` and are used only for the rest-phase false
activations, which are therefore in-sample (as offline). Task windows that would need a
calibration that does not exist yet are emitted undecodable (`n_uncalibrated_task`).

`filter_mode: phase_warmup` runs, in addition, a filter per phase from `warmup_s` before it and
computes the windows of a phase from it. It knows the phase schedule 2 s ahead, so it is **not**
causal; it exists to show that the stream and the offline windows are the same computation.

### Stream vs offline features (what differs and why)

* With `phase_warmup` and the offline mains frequency the stream windows equal `build_windows`
  to < 1e-9 (test) and the stored `ml_windows` features to ~2e-6 SD (float32 storage and the
  `EPS` in `log(f + EPS)` under a different % reference).
* `continuous` (the online chain): the offline filter is restarted 2 s before every phase, and
  its 0.5-Hz mains tracker has not fully converged by the phase start; the continuous filter
  has. The difference is a transient at phase starts, largest in the spectral features
  (`band_50_100`, `mnf`, and the count features): on the synthetic study (4 participants) up to
  0.5-0.9 SD in the first windows of a phase, p99 0.08-0.15 SD over all windows, p99 0.03-0.14 SD
  from 2.2 s after a phase start, median 4e-4-8e-3 SD (`feature_deviation`). The rest phase starts
  identically in both. The IMU features agree to < 1e-6 SD.
* `line_hz: preroll` vs the offline value (a few 1e-5 to 1e-4 Hz apart on the synthetic takes)
  adds a small persistent difference: with the offline value the late p99 on synthetic participant s01 drops from
  0.035 to 2e-4 SD. Both use biomech's estimator (`take_line_hz`, nominal grid first, then the
  other of 50/60 Hz).
* Effect on decisions: the uncalibrated stream decoder reproduces `ml_eval`'s balanced accuracy
  (synthetic study: identical on s01, s02, s04; 0.403 vs 0.397 on s03).
* Masks: causal vs two-sided dilation (only where rail/lead-off events occur).

## 3. Decoder (`ml.online.decoder`)

Per LOSO fold (held-out participant) the model is trained offline on the other participants'
stored `ml_windows` windows, with the fold's hyper-parameters from the main `ml_eval` LOSO run
(`best_params`, column `hyper` = `main_loso`; a grouped inner search with
`stages.ml_eval.inner_folds` when missing, `hyper` = `inner_search`) and the same subsample
(`stages.ml_eval.max_train`; `ml_online` has no `max_train` of its own, setting one is a config
error) and seed — it is `ml_eval`'s fold model, so `ba_raw` of mode `none` differs from
`balanced_accuracy_offline` only through the feature differences above. The primary feature
set's model decodes inside the replay loop (its cost is in the chunk time); all modes are then
decoded from the stored stream windows with the model in force at each window, which gives the
same decisions (`inloop_equal_batch`).

| Mode | Calibration windows (stream features, optical labels) | Model in force | Scored windows |
|------|------|------|------|
| `none` | — | fold model | all structured phases (every phase except `free_motion`; the rest phase included) |
| `first` | the first `first_k` (100 = 5 s) decodable labelled windows **after the rest block** (the realistic calibration block; the rest block itself holds REST only) | fold model, refit with them from `guard_s` after the last one | after that |
| `per_phase` | progressive guided calibration: the first `per_phase_k` (20 = 1 s) decodable labelled windows of every structured phase (the labelled rest phase and the four tasks; a phase with fewer is skipped) | refit after each phase's k-th window with the calibration windows of that and all **earlier** phases | each phase from `guard_s` after its k-th window |

Refits add the calibration windows to the fold's training subsample with the few-shot weights
(`weight: balanced`, ml05 §6.1). `variants.few_shot` `per_phase` uses the windows of every phase
at once; in a stream that would need windows of later phases, hence the progressive form. A mode
without enough calibration windows is reported with `applicable = false` and no scores.

Smoothing (`smoothing: {n: 3, enter: 2, hold: 1}`): majority over the last 3 decodable windows
with hysteresis — a movement class is entered when it has at least 2 of them (and more than the
current class) and held while it has at least 1; REST otherwise. Undecodable windows hold the
state; the state starts at REST at every phase start. Onset events (`decisions` table) are the
entries into a movement class.

## 4. Release detector (`ml.online.release`)

The X3 detector (an05 §3.2, `biomech.timing.release`, frozen parameters of `timing.release`)
fed block by block (`StreamRelease`) by the stream's causal chain: trailing RMS per closer
(50 ms) in dB re its rest level, mean over closers; baseline = trailing median of the 10-ms
decimated envelope over [t − 0.6, t − 0.3] s; release when ≥ 3 dB below it for 30 ms; 0.3 s
refractory. A NaN envelope value (all closers masked) is held at the last valid value in the
baseline, values before the first valid one take that value, and a baseline window ending
before it gives no baseline, exactly as offline (E21; formerly the stream used the median of the
values seen so far and the offline detector the median of the whole phase). Fed from its first
sample it equals `detect(causal_db(...))` on the same samples, also with masked stretches
(test). Since E27 the offline X3 also runs on a causal chain (an05 §3.2), so the remaining
differences are the chain's start and the rest level: the stream detector starts at the end of
the rest phase and runs continuously on one filter from the stream start (mains frequency from
the pre-roll; offline: per task, from a 2-s warm-up, at the offline per-phase grid frequency),
so its baseline at a task start is not padded; its rest level is the median of the trailing
50 ms RMS of the stream's rest phase (offline: of the centred masked 50 ms envelope of the
causally cleaned rest). The chance level is drawn with the study `seed` (offline: the
participant seed).

## 5. Metrics (`ml.online.metrics`; `summary` per participant × feature set × mode)

| Column | Definition |
|--------|------------|
| `ba_raw`, `ba` | balanced accuracy of the raw / smoothed window decisions over the scored windows (decodable, labelled) |
| `ba_free_raw`, `ba_free` | the same on `free_motion` (never trained on) |
| `ba_k0_same_rows` | mode `none` on the same rows (paired calibration gain) |
| `latency_ms_median` (`_q1`, `_q3`), `detected_frac` | per REST→class onset of the optical label track (preceded by REST since `t_rest`), the first window ending in `[max(t_on − 0.5 s, t_rest), t_off]` at which the smoothed decision **enters** that class; latency = its *decision time* (the chunk at which it was emitted) − `t_on`; negative = anticipatory; a state held over from the previous movement is not a detection |
| `frac_le_150ms`, `frac_anticipatory` | detected onsets with latency ≤ `latency.budget_ms` (150 ms, literature budget; the column name follows the value) / ≤ 0 |
| `latency_win_ms_median` | `ml_eval`'s definition (`evaluate.latency`: raw decisions, 2 consecutive windows, window-end times, onsets of the window labels) on the stream; compare `latency_ms_median_offline` |
| `fa_per_min_free` (`fa_free_n`, `fa_free_minutes`) | smoothed onsets in `free_motion` at a REST-labelled window with no labelled onset within `fa_guard_s` (0.5 s) after it, per minute of REST-labelled, calibrated free-motion windows |
| `fa_per_min_rest` (`_raw`) | `evaluate.false_activations` on the smoothed (raw) decisions in the rest phase (in-sample normalization, see §2), inside the mode's scored portion (none for `first`: NaN); compare `fa_per_min_offline` |
| `x3_*` | release detector on the first timing task (`open_close`): detected share, chance level at random pseudo-onsets, median anticipation, false releases/min, and the offline X3 values (`*_offline`); all tasks in the `release` table |
| `chunk_ms_median`, `chunk_ms_p99`, `chunk_ms_max`, `realtime_factor` | per-chunk processing time (`time.perf_counter_ns`, one BLAS thread) of masks, filter, IMU buffers, window features, in-loop decoding and the release detector, without the one calibration chunk (`calibration_ms`); real-time factor = compute / stream time |
| `emit_delay_ms_median` / `_max` | decision time − window end (chunk quantization and the IMU wait) |
| info | `line_hz`, `line_source` (`preroll`, `nominal` or `offline`), `line_hz_preroll`, `line_hz_offline`, `stream_start_us`, `stream_s`, `ref_source` (`rest_rms`), `rest_precedes_tasks`, `calibration_error`, `mvc_precedes_motion` (take stamps; the MVC take is not used), `calibrated_after_rest_ms`, `c2_lead_ms` (C2 opening lead, TKEO, primary rule `timing.onset_rule`, for the latency budget) |
| bookkeeping | `n_chunks`, `replay_wall_s`, `n_windows`, `n_decodable`, `n_imu_late`, `n_uncalibrated_task`, `applicable`, `hyper`, `params`, `fit0_s` (fold-model fit), `n_cal` (calibration windows of the last refit), `n_models` (1 + refits), `cal_fit_s`, `n_eval`, `n_onsets`, `inloop_equal_batch` (mode `none`, primary feature set) |
| offline | `balanced_accuracy_offline`, `latency_ms_median_offline`, `fa_per_min_offline`, `free_balanced_accuracy_offline`: the participant's `ml_eval` `loso` row (same model and feature set) |

Other tables: `decisions` (every onset event per feature set and mode: decision time `t_us`,
window end, class, phase, true label, `kind` = hit / false / other, `scored` (inside the mode's
scored portion or in free motion), latency of hits), `release` (per task), `feature_deviation`
(stream vs stored `ml_windows` features per block, |Δ| in offline SDs, overall and from
`late_after_s` after the phase start). Arrays per participant for the viewer:
`online__{pid}__t_us` (decision times), `__t_end_us`, `__y`, `__yhat` (smoothed, primary feature
set, mode `none`), `__yhat_raw`, `__phase` (index into PHASES), `__release_us` (stream release
decisions).

## 6. Parameters (`stages.ml_online`, `software/biomech/biomech/etc/biomech.yaml`)

`chunk_s` 0.02, `model` svm, `feature_sets` [emg_imu, imu], `primary_feature_set` emg_imu,
`line_hz` preroll, `filter_mode` continuous, `imu_max_wait_s` 0.1,
`smoothing` {n 3, enter 2, hold 1}, `calibration` {modes [none, first, per_phase], first_k 100,
per_phase_k 20, guard_s 0.35, weight balanced}, `latency` {pre_s 0.5, budget_ms 150},
`fa_guard_s` 0.5, `pids` [] (all), `n_jobs` 0 (every core, `evaluate.resolve_jobs`; model fits
in parallel; replays run one at a time with one BLAS thread so the chunk times are not inflated by
the stage itself). Unknown keys are errors; `line_hz` must be `preroll` or `offline`,
`primary_feature_set` one of `feature_sets`, `smoothing` must satisfy 1 ≤ `hold` ≤ `enter` ≤ `n`.
Window settings come from `stages.ml_windows.window`, mask settings (`dilate_s`, `rail_frac`)
from `signals.emg.mask`, the nominal mains frequency from `signals.emg.nominal_line_hz`, the inner
folds from `stages.ml_eval.inner_folds`, the release parameters (`timing.release`), the X3 tasks
(`timing.tasks`, the first one in the summary) and the C2 onset rule from `stages.timing`.

Run time: the replay costs about the stream duration × `realtime_factor` per participant; the
fits are `len(feature_sets)` fold models + 1 (`first`) + up to 5 (`per_phase`) refits per
participant and feature set. To shorten a run, restrict `pids` or `calibration.modes`. Pilot
(12 participants): about 6 min for the whole stage.

## 7. Optional end-to-end path

`jtsmm.viewer.sources.ReplaySource` + `Hub` replay a take on the wall clock into the live
viewer; the `OnlineProcessor` could subscribe to the hub's blocks. Not wired: the stage replays
as fast as possible, which gives the same decisions (the chain depends on sample order and
timestamps, not on wall time).

## 8. Pilot (12 participants)

Medians over participants, model SVM, mode `none` unless stated. X3 rows: values of an earlier
pipeline version (offline X3 on the zero-phase chain, NaN baseline fill by medians); the current
values follow the table.

| Quantity | Stream | Offline |
|----------|--------|---------|
| Balanced accuracy, EMG + IMU (`ba_raw`) | 0.585 | 0.583 (`ml_eval`) |
| Balanced accuracy with per-phase calibration (`ba_raw`, `per_phase`) | EMG + IMU 0.690, IMU 0.662 | — |
| Decision latency (`latency_ms_median`, EMG + IMU, smoothed) | ≈ 145 ms | — |
| X3 release, `open_close`: detected | 89 % (chance 40 %) | 89 % |
| X3 anticipation (median) | 99 ms | 149 ms |
| X3 false releases | 25 /min | 21 /min |
| Compute per 20-ms chunk | median 0.7 ms, P99 4.5 ms | — |

The stream decoder reproduces the offline accuracy; the in-loop decisions equal the batch
decoding for every participant (`inloop_equal_batch`). The online compute is faster than real time
(real-time factor ≈ 0.06 on the pilot: 0.7 ms per 20-ms chunk, P99 4.5 ms) on the analysis PC;
this bounds the order of magnitude, not an embedded target (§1). In this pilot run the release
detector on the stream's causal chain kept its detection rate but anticipated less and fired more
false releases than the offline X3, which then still ran on the zero-phase cleaned EMG; since
E27 the offline X3 uses a causal chain as well. *Values of the paper store (`r17a`, 13 adults,
medians over participants; E42 chance level):* open-close stream 89 % detected vs chance 62 % (offline 80 vs
61 %), anticipation 102 ms (offline 103 ms), 26 false releases/min (offline 26); chewing stream
67 vs 66 % (offline 78 vs 67 %), 179 ms (188 ms), 23 /min (21 /min). Stream and offline agree:
above chance in open-close only. Compute (wall-clock time of that run on the analysis PC, so it
differs from run to run): median 1.0 ms, P99 4.5 ms per 20-ms chunk (real-time factor 0.080).
These are intent-detection results of a simulated real-time replay; no exoskeleton and no closed
loop are involved.
