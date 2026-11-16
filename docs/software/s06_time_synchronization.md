<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# s06 — Time Synchronization (software side)

This document complements [`../hardware/h06_synchronization.md`](../hardware/h06_synchronization.md). It states the **target** clock model, then clearly marks what the **current code** does.

Hardware vs software (short):

- **Hardware sync** — shared wire / MCU edge stamp (XTRIG, polled DRDY, INT IRQ).
- **Software sync** — map foreign timestamps onto `t_master_us` (trigger match, affine fit, LED, PPS).

## 0. Implementation status (read this first)

Clock tree: [`01_architecture.md` §4](../01_architecture.md#4-timing-architecture); hardware side: [`h06`](../hardware/h06_synchronization.md) §1.

| Path | Target | **Today (`jtsmm record`)** |
|------|--------|----------------------------|
| EMG / IMU → HDF5 with MCU `t_master_us` | yes | **Done** — no affine needed |
| `TRIGGER_EVENT` → `/triggers` | yes | **Done** |
| Camera metadata → `/cameras/<id>/` | yes | **Partial** — timestamps + ids; **no video pixels** |
| Match camera frame ↔ `trigger_id` | exact id or inferred | **Done** — exact id when provided (mock); real picamera2 path (`trigger_id=None`) uses a PPS-anchored first frame + **cadence tracking**, with prediction / slip detection as fallbacks (see §4.1) |
| Affine fit for cameras | optional / residual / offline | Fit **updated**; stored `t_master_us` = **trigger time**, not `a·t+b` |
| Pi PPS affine | start + end of take | **Done** — `pps` ×`host.capture.pps_samples` (8) at take start and end (§4.3) |
| Track A LED-flash affine | webcams | **Not implemented** on default GS HAT |
| `jtsmm sync rebuild` offline re-stamp | yes | **Not implemented** as a CLI |
| Separate `clock_align` process + queues | s02 design | **In-process** `ClockAligner` inside `SessionSupervisor` |

Code: [`jtsmm/sync/clock_align.py`](../../software/jtsmm/jtsmm/sync/clock_align.py), [`jtsmm/capture/supervisor.py`](../../software/jtsmm/jtsmm/capture/supervisor.py), [`jtsmm/capture/capture_cam.py`](../../software/jtsmm/jtsmm/capture/capture_cam.py).

---

## 1. The clock model (target)

For every **non-master** clock `k`:

```
t_master_us = a_k · t_local + b_k
```

- `t_local` — foreign clock (Pi monotonic ns, camera `SensorTimestamp` ns, UVC PTS, …).
- `a_k` — scale ≈ 1 (drift).
- `b_k` — offset.

**EMG and both IMUs do not use this model** — the MCU already stamped them.

The map is local in time (crystal drift). A production estimator would re-fit over a sliding window (target algorithm in §6).

## 2. Clocks we care about

| Clock | Sync mechanism | Needs affine? | Live today? |
|-------|----------------|---------------|-------------|
| MCU `t_master_us` | Master (`time_us_64`) | no | yes |
| EMG / IMU records | MCU stamp at DRDY/INT | no | yes |
| Cam Track B | HW XTRIG + `TRIGGER_EVENT` match | residual / offline only | metadata path partial |
| Pi `CLOCK_MONOTONIC` | PPS handshake | **yes** | yes (start + end pass, §4.3) |
| Cam Track A UVC PTS | LED-flash pairs | **yes** | no |

## 3. PPS handshake (Pi monotonic)

```
Pi side                                   MCU side
-------------------------------           --------------------------------
t_send = clock_gettime(MONOTONIC)
send PPS request with id = N
                          --->            t_master_a = t_master_us()
                                          send PPS_REPLY{ id=N, t_master_a }
t_recv = clock_gettime(MONOTONIC)
record midpoint ((t_send+t_recv)/2, t_master_a) → update _pi_fit
```

**Status:** implemented as a start and an end pass, not a 1 Hz client (§4.3). Fusion of EMG/IMU/cameras does not depend on it while those streams already carry `t_master_us`.

## 4. Hardware-triggered camera mapping (Track B)

### 4.0 Intended flow

Each trigger produces:

1. MCU `TRIGGER_EVENT{trigger_id=N, t_master_us=T_N}` over USB.
2. Both cameras expose on the shared XTRIG edge.
3. Pi frame metadata includes `SensorTimestamp` (and, in the **target** design, a way to recover `N`).
4. Aligner stores camera row with `t_master_us = T_N`.

Different modality rates are fine: all streams share the **same time axis**, not a common sample grid.

### 4.1 Camera↔trigger association (implemented)

libcamera / picamera2 for IMX296 expose **`SensorTimestamp`** (and exposure/gain controls), but **not** the MCU’s monotonic `trigger_id` counter. There is no standard metadata key that echoes the RP2350 trigger index — so we do **not** wait for a fictional camera `trigger_id` field.

`capture_cam.py` today:

- sets `trigger_id=frame_idx` only on the **mock** path (exact-id lookup);
- on the real picamera2 path leaves `trigger_id=None`.

`ClockAligner.ingest_cam` then associates frames as follows:

1. **Exact id** when `trigger_id` is a concrete int (mock / tests).
2. **First-frame anchor + cadence tracking** when `trigger_id is None` (real path): each camera keeps a cursor `next_trigger_id`. When a PPS fit is available, the first frame is anchored on the trigger nearest the PPS-mapped `SensorTimestamp`. Otherwise the first frame locks to the oldest buffered `TRIGGER_EVENT` only when the buffer is short (USB slightly ahead). A long buffer (≥6 events spanning >4 trigger periods) is treated as **pipeline fill** (IMX296 pulses before libcamera returns a request) and the first frame locks to a *recent* trigger instead — nearest the frame's host arrival, typically the newest. Each later frame advances the cursor by the trigger count implied by its `SensorTimestamp` delta (cadence tracking, `host.capture.camera_delta_tracking`); a one-shot anchor refinement after ~90 frames removes a residual constant anchor error.
3. **Start order (since 2026-09):** `jtsmm record` starts the trigger only *after* picamera2 is streaming (`host.capture.trigger_after_cameras: true`), so the oldest buffered `TRIGGER_EVENT` is pulse 1 and the camera's first frame pairs with it. With the old trigger-first order the oldest buffered trigger predated the camera by its start-up latency (typically 10–40 frames) and every camera row inherited that offset — undetectable in the file, because a camera row carries the trigger's time. `sync/report_json.capture_order` records which order ran.
4. **Cross-check:** with the `pps` host-clock fit, `ClockAligner` records per camera `assigned trigger time − PPS-mapped SensorTimestamp`. `sync/report_json.cameras.<id>.assign_residual.median_frames` is ≈ a small constant (camera pipeline latency) when the pairing is right and ≈ −N when it is off by N pulses; `drift_us_per_s` is ≈ 0 when trigger stamps track the pulses (the pre-2026-09 PIO period bug reads as ≈ −240 µs/s at 60 Hz). `jtsmm integrity` warns beyond 1.5 frames / 50 µs/s and errors beyond 3 frames / 500 µs/s (configurable).

   > **The anchor is load-bearing.** A first frame that takes `_triggers[0]` while six trigger pulses have already fired (libcamera pipeline fill) shifts **every** later pose by a constant. That bias is self-consistent: the per-camera affine fit is seeded from the bad pair, so step 4 validates the offset and `residual_rms_us` stays at a few µs. Defences: arm `trig_start` **after** `cam.start()`; `reset_trigger_backlog()` at camera start; first-frame lock treats a long buffer as pipeline fill and pairs with a recent trigger. Detection is cross-modal — see §4.3.
3. **Predict-then-match recovery** (fallback) once the affine fit is warm: the aligner estimates the expected elapsed trigger ticks from `SensorTimestamp -> t_master_us` and the measured trigger period. If that predicts a skipped camera frame, the cursor jumps forward (capped by `predict_max_jump`) and `camera_drop_recovered` increments. Wild jumps are refused when the affine residual is noisy.
4. **Validation before trust**: the chosen trigger time is checked against the affine prediction. If it is too far away, the frame is rejected (`t_master_us=None`) and `validation_fail_count` increments instead of silently writing a wrong stamp.
5. If the expected id is missing but a greater id is already buffered, the cursor jumps forward and `slip_count` increments. If no id ≥ cursor is buffered yet (USB lag), the frame is **held** (not dropped): `last_ingest_waiting` / `waiting_for_trigger` increment and the supervisor keeps the frame in a short hold queue until MCU triggers catch up.
6. The matched id (never `None`) is written into the aligned payload — this also prevents the native HDF5 writer from crashing on `int(None)`.

Latest-trigger fallback is **not** used in the steady-state cursor. The first-frame pipeline-fill path **does** prefer a recent trigger (see step 2). Absolute nearest-time match in MCU units is **not** the primary path without a live PPS bridge (`SensorTimestamp` is Pi-domain).

### 4.2 Affine fit for Track B cameras — when to use it

| Use | Recommendation |
|-----|----------------|
| Primary science stamp for HW-triggered frames | **No** — keep `t_master_us` from the matched `TRIGGER_EVENT` |
| Residual / quality metric | **Yes** — already computed as `residual_us` when pairs exist |
| Offline refine / missing trigger_id | **Yes** — once PPS or a coarse offset exists, map `SensorTimestamp → t_master` |
| EMG / IMU | **Never** — already on master |

**Today:** `ingest_cam` updates a per-camera least-squares fit and sets `residual_us = t_master - fit.map(SensorTimestamp)`, but **writes `t_master` from the trigger lookup**, not from the fit. So the learner is **not unused** (it runs for residuals / `sync_report`), but it is **not the authority** for stored camera time.

**Recommendation:** keep trigger-match as primary for Track B; use the PPS/affine fit for (a) Pi host events, (b) residuals in the viewer/doctor/integrity, (c) offline rebuild when association is ambiguous. Do **not** replace EMG/IMU stamps with an affine map.

### 4.3 PPS host-clock handshake (implemented 2026-09)

`jtsmm record` sends `pps <id>` over CDC (`host.capture.pps_samples`, default 8, `0` disables) once after the sensor streams settle and once after the last record is pumped. The firmware answers with `PPS_REPLY{id}` on the bulk stream, stamped with `t_master_us` at parse time; the host attributes it to the midpoint of its `CLOCK_MONOTONIC` send/receive times and rejects samples with a round trip above 30 ms. `ClockAligner.ingest_pps` fits `t_master ≈ a·t_pi + b` (`sync/report_json.pi_to_master`, with `n_samples`, `mode`, `span_s`). Every PPS sample is kept (start and end pass together). While they span less than 30 s the fit is offset-only (`a = 1`, `mode: offset_only`) — a free slope over the ~0.2 s start pass was off by hundreds of ppm — and a free slope afterwards (`mode: free_slope`). Stamps taken live during a take therefore use the start offset only; `ClockAligner.map_pi_ns(t_host_ns)` called after the end pass uses the final fit, which is how stored `t_host_ns` values are re-mapped at close. Without any PPS reply, `map_pi_ns` falls back to an offset-only estimate from record arrival times (quality `mcu_arrival`, never `pi_fit`), and the camera PPS anchor is not used. libcamera `SensorTimestamp` is `CLOCK_BOOTTIME`, which equals `CLOCK_MONOTONIC` on a Pi that never suspends; the recorder stores `pi_clock.boottime_minus_monotonic_ns` so the assumption can be checked. A firmware without `pps` yields `accepted = 0` and a warning — never a failed take.

### 4.3 Optical↔IMU timing check (cross-modal, mount-free)

A constant camera↔trigger misassignment is invisible to every single-stream metric, so it is checked **against another modality**. [`integrity/optical_imu_sync.py`](../../software/jtsmm/jtsmm/integrity/optical_imu_sync.py) correlates angular **speed** |ω| from `/optical/pose_*` against |ω| from `/imu_*/quat`.

Speed is invariant to the unknown IMU mounting rotation *and* to the camera-world vs game-RV-earth offset, so this needs no calibration and runs on any session that has both streams plus enough motion (median optical |ω| ≥ 5 °/s, correlation ≥ 0.5; otherwise reported as `inconclusive`).

| Lag | Result |
|-----|--------|
| > 120 ms | **error** — frames paired with the wrong `TRIGGER_EVENT`; poses unusable for kinematics |
| > 40 ms | warn |
| ≤ 40 ms | pass (fusion latency + exposure midpoint live here) |

The estimate lands in `report.summary["optical_imu_sync"]` as `{lag_ms, correlation, n}` per slot.

## 5. LED-flash mapping (Track A) — target

For each LED flash:

- MCU emits `LED_FLASH_EVENT` with `t_master_us`.
- Pi detects LED-on in the frame and records UVC PTS.
- Pairs feed an affine fit (noisy ±½ frame; many flashes → ~ms residual).

Default GS HAT: sync LED **DNP** — Track B path only.

## 6. Online affine estimation (target algorithm)

Target estimator (windowed WLS + Huber) as previously specified; **current** `ClockAligner._update_fit` is a simpler sliding-window least squares (last ~30 points), good enough for residuals.

Parameters (target):

- `window_us = 60 × 1e6`, `decay_us = 30 × 1e6`, `min_observations = 8`, Huber `δ = 3 × MAD`.

## 7. Latency between estimation and use

When affine **is** used to assign stamps (Pi events, Track A, offline rebuild):

1. Online assignment may differ slightly from a full-session offline re-fit.
2. Target: store fit history under `/sync/<clock>_to_master` for `jtsmm sync rebuild`.

**Today:** session gets a JSON snapshot in sync report attrs; no rebuild CLI.

## 8. Diagnostics

| Metric | Target | Today |
|--------|--------|-------|
| Per-clock residual RMS | viewer + doctor | in `ClockAligner.sync_report()` at end of record |
| Camera drop count | yes | `aligner.drops` (inconsistent / discarded frames) |
| Waiting for trigger | yes | `aligner.waiting_for_trigger` (USB lag; frame held, not dropped) |
| Camera hold overflow / leftover | yes | `sync.attrs.report_json` → `cam_hold_overflow`, `cam_hold_leftover` |
| Camera slip count | yes | `aligner.slip_count` (cursor jumped over missing ids) |
| Camera-side drops recovered from `SensorTimestamp` gap | yes | `aligner.camera_drop_recovered` |
| Prediction validation rejects | yes | `aligner.validation_fail_count` |
| Prediction anomalies (`expected_ticks <= 0`) | yes | `aligner.anomaly_count` |
| Optical pipeline counters | yes | `sync.attrs.report_json` → `optical` (`rgb_frames`, `enqueued`, `written`, …) |
| Live viewer gauges | yes | trigger rate notes in viewer; full residual UI incomplete |

Acceptance (target):

| Clock | Acceptable residual RMS |
|-------|-------------------------|
| Pi monotonic | <500 µs |
| Cam Track B | <100 µs |
| Cam Track A | <2 ms |

## 9. Worked example (PPS)

After many PPS pairs, typical `a ≈ 1 ± few ppm`, large `b`. A Pi event at `t_local` maps to `a·t_local + b` on the MCU axis. EMG/IMU already on that axis without this step.

## 10. Edge cases

| Case | Handling |
|------|----------|
| MCU clock wraps | 64-bit µs — not a practical concern |
| Pi NTP step / slew | PPS uses `CLOCK_MONOTONIC` (`time.monotonic_ns`), deliberately not `_RAW`: it is never stepped, and it is the clock libcamera's `SensorTimestamp` (`CLOCK_BOOTTIME`) runs on, so the camera anchor maps through the same fit. An NTP slew (≤ 500 ppm) during a take is absorbed by the free-slope fit over start + end pass; it cannot be told apart from crystal drift, and shows in `pi_to_master.residual_rms_us` only if the slew rate changes mid-take |
| Camera frame missed | prediction path can jump the cursor from `SensorTimestamp` gap; `camera_drop_recovered++`; if prediction/trigger disagree beyond tolerance, reject frame and `validation_fail_count++` |
| Trigger not yet over USB | frame **held** in supervisor queue; `waiting_for_trigger++`; cursor unchanged until matching `TRIGGER_EVENT` arrives (or hold overflows) |
| Real path `trigger_id is None` | PPS anchor + cadence tracking (§4.1) — not latest-trigger |
| AGC on during Track B | fights XTRIG exposure — keep `host.cameras.list[].ae_enable: false` ([`s08`](s08_runtime_configuration.md)) |
| MCU disconnect | stop streams; do not invent stamps |

## 11. Reviewer checklist

- [ ] EMG/IMU still carry MCU `t_master_us` with no affine rewrite.
- [ ] Track B camera rows prefer matched `TRIGGER_EVENT` time over `fit.map()`.
- [ ] Any new association logic documents drop/slip/hold/recovery behaviour.
- [ ] Docs mark target vs implemented when changing sync.
- [ ] No `CLOCK_REALTIME` in the data path.
