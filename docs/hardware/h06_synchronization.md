<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h06 — Hardware/Software Synchronization

> **Building the trigger / sync LED schematic?** See [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md).

This is the load-bearing document of the project. Tight, traceable cross-modal timing is what turns a pile of sensors into a usable dataset for multimodal sensor fusion. We commit to a single rule:

> **There is one master clock. It lives on the RP2350A. Every record in every stream carries `t_master_us` (microseconds since MCU boot). All other clocks are mapped to it.**

Software-side status and affine-fit details: [`../software/s06_time_synchronization.md`](../software/s06_time_synchronization.md).

## Hardware vs software synchronization (definitions)

| Term | Meaning in this project |
|------|-------------------------|
| **Hardware sync** | A shared physical signal (or an MCU `t_master_us()` stamp of a sensor edge) forces devices onto the master timeline without post-hoc clock regression. Example: MCU **XTRIG** → both IMX296 cameras; EMG stamped when the sensor core's poll sees DRDY fall, IMU on the INT edge IRQ. |
| **Software sync** | Devices keep separate clocks; host software maps foreign timestamps onto `t_master_us` (trigger matching, affine fit, LED-flash pairing, PPS). |

**Default Track B (2× CSI GS):** load-bearing camera sync is **hardware-triggered**. Software still labels frames and stores them on `t_master_us`. Affine fits are for **foreign clocks** (Pi monotonic, Track A PTS) and diagnostics — not for EMG/IMU.

## 1. Architecture

The clock tree is drawn once, in [`01_architecture.md` §4](../01_architecture.md#4-timing-architecture): the RP2350 hardware timer is the master clock; the ADS1299 is stamped on DRDY, both BNO085 on their INT line; the PIO trigger generator drives both cameras' XTRIG and reports every edge to the Pi as a `TRIGGER_EVENT` with its `trigger_id`; the cameras' `SensorTimestamp` and the Pi's `CLOCK_MONOTONIC` are the foreign clocks that get mapped.

There are two camera synchronization paths depending on capability:

- **Hardware-triggered (Track B + optional Arducam UC-599 USB3)**: the MCU emits a trigger; cameras expose on that edge. Primary association is MCU `TRIGGER_EVENT.trigger_id` ↔ camera frame (see [`s06`](../software/s06_time_synchronization.md) for what is implemented vs target).
- **Software-synced (Track A webcams)**: the MCU drives a **visible sync LED**; the Pi detects LED-on in the image and pairs UVC PTS with LED `t_master_us`. **Not** an IR ring for retroreflective markers.

### 1.1 Implementation status (Track B default)

| Piece | Status |
|-------|--------|
| MCU `t_master_us` on EMG / IMU / `TRIGGER_EVENT` | **Implemented** |
| PIO `trigger_gen` + dual CSI XTRIG fan-out | **Implemented** (reload period each wrap; `pulse.width_us` = IMX296 low-time when `imx296_fast`) |
| `TRIGGER_EVENT` stamp source | **PIO IRQ** at the exposure-start edge (`flags[0]` = 1); the `repeating_timer` is only a watchdog fallback (`flags[0]` = 2), see §3.1 |
| Pi records trigger + EMG + IMU to HDF5 | **Implemented** |
| Camera **metadata** rows (`t_master_us`, `SensorTimestamp`, …) | **Partial** — `jtsmm record` polls picamera2; the first frame is anchored on the trigger nearest its PPS-mapped `SensorTimestamp`, then cadence tracking ([`s06`](../software/s06_time_synchronization.md) §4.1) |
| Extract real `trigger_id` from camera metadata | **Not available** — libcamera/IMX296 does not expose MCU `trigger_id`; host associates by PPS anchor + cadence instead |
| Camera **video** / pixel sidecars (`.mkv`) | **Not implemented** — metadata only |
| Affine fit applied to stored timestamps | `pps` ×`host.capture.pps_samples` (8) at take start + end; the final fit re-maps `/events` and `/phases` stamps and anchors camera pairing (camera rows keep the trigger stamp) |
| Track A LED-flash sync | **Target** (HAT LED DNP on default GS build) |

## 2. Timing budget per modality

| Modality | Source clock | Sync mechanism | Budget |
|----------|--------------|----------------|--------|
| EMG | ADS1299 internal oscillator, DRDY edge timestamped by MCU | Direct (MCU is the master) | <1 µs |
| IMU MTA | BNO085 internal clock, INT edge timestamped by MCU | Direct | <50 µs (INT-to-read jitter from RP2350A IRQ scheduling) |
| IMU CRA | same | same | <50 µs |
| Cameras Track B | IMX296 sensor exposure clock, **XTRIG** (XTR+) fast trigger | Hardware trigger | <100 µs (libcamera queue jitter) |
| Cameras Track A | Webcam internal clock + UVC PTS | LED-flash protocol + affine fit | <2 ms |
| Pi monotonic | Linux `CLOCK_MONOTONIC` | `pps` handshakes at take start + end | <1 ms |

The headline number for the validation experiment is **<100 µs camera-vs-EMG jitter for Track B**, which is achievable on any modern Pi 5 running picamera2 with sensor metadata enabled.

## 3. Camera trigger path (Track B)

### 3.1 Pulse generation

The RP2350A PIO state machine `trigger_gen` drives a single GPIO at the configured frame rate. With the PIO clock at 1 MHz:

- Pulse edges are generated in hardware by the PIO (sub-µs edge placement relative to the PIO clock).
- Pulse width: programmable (default short pulse; for IMX296, **low-time** sets exposure — see Pi imx296 `trigger_mode` docs).
- **`TRIGGER_EVENT` stamping (since 2026-09):** the PIO raises IRQ flag 0 on the very instruction that drives the exposure-start edge (`irq set 0 side 0` in `trigger_gen.pio`) and `trigger_pio.c` latches `t_master_us()` in that IRQ — the stamp is a *measurement* of the edge, not a prediction. The same-period `repeating_timer` still runs, but only as a **watchdog**: if the PIO IRQ is silent for two periods it takes over stamping for the rest of the run, back-fills the ids it missed, and the comms core logs `TRIGGER STAMP FALLBACK` (CDC + `REC_LOG`). `TRIGGER_EVENT.flags[0]` records the source (`1` PIO IRQ, `2` timer, `0` firmware before 2026-09) and `flags[1]` the fallback bit. `trig_stamp timer` selects the legacy timer-only path for an A/B; `trig_status` shows both.
- **Period compensation:** the PIO loop costs extra cycles per period (`mov` + `jmp y--` reload, `irq`). Before 2026-09 the FIFO templates were *not* compensated, so at 60 Hz the pin period was 16670 µs against a 16666 µs software timer — 4 µs/frame of drift, a full frame every ~69 s, and invisible in the file because a camera row carries the trigger's own time. `trigger_timing.c` now subtracts the overhead so the pin period equals `period_us` exactly (host-tested in `firmware/jtsmm/tests/test_trigger_timing.c`); `trig_comp off` restores the old templates for a scope comparison on **`TP_TRIG`**.
- **Lost edges (contract K4):** one PIO IRQ flag cannot count pulses — if the IRQ is not serviced within a period, the next edge merges into it. The ISR counts edges against the pin schedule, not the previous (possibly late) stamp: `k = max(1, floor((t − ref + T/16)/T))`, where `ref` is re-anchored on a prompt IRQ and otherwise advanced by `k·T` (`trigger_timing_edge_track`, host-tested in `test_trigger_timing.c`; a single IRQ served up to a period late is still one edge and is stamped at its scheduled time) and advances `trigger_id` by that many, so the host sees a gap (missing trigger) instead of every later frame pairing with the wrong id; `trig_status` shows `lost_edges` and the comms core logs `TRIGGER EDGES LOST`. The PIO IRQ runs above every other IRQ on the comms core (`platform.irq.trigger` 0–3 → NVIC 0x00–0x60, all others at the SDK default 0x80).
- **What the stamp marks:** with `polarity: imx296_fast` (exposure on the LOW phase) the IRQ edge is exposure **start**. With the other polarity the same edge is exposure **end** — `trig_status` prints `edge=…`.

**IMX296 fast trigger (Raspberry Pi / Sony):** integration starts when **XTRIG** goes **low**, ends when it goes **high**; exposure time ≈ **low pulse width + ~14.26 µs**. Frame rate follows pulse rate. Firmware and PIO polarity must match this (invert if the scope shows the wrong edge starting exposure). Enable: `echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode`.

**Idle level:** between runs the line rests at the **inactive** level of `pulse.polarity` — **HIGH** for `imx296_fast` (the 74LVC2G241 is non-inverting and the divider only level-shifts; the camera pulls XTR up itself), **LOW** for `active_high`. It is driven there from boot, by `config apply`, and on `trig_stop`; the PIO hands over edge-free (latch preloaded, template pulls do not side-set), and every run starts with the non-exposure (HIGH) phase. A `trig_stop` during the LOW phase ends that exposure early: one short frame whose `TRIGGER_EVENT` already exists. (Before 2026-09 the line idled LOW for both polarities — the IMX296 *exposing* level — and a stop mid-HIGH-phase added a stray falling edge.) The TRIG LED shares the pin, so with `imx296_fast` it is **lit while idle** and blinks dark with the exposures.

### 3.2 Pulse fan-out

The trigger GPIO goes through **2× 74LVC2G241** buffers (four outputs total) to a 4-way fan-out. **Default GS build** uses **2× CSI + scope test pads** only; **USB opto**, **BNC jack**, and **sync LED** branches are **DNP** ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md)).

| HAT net | To | Wiring | Default build |
|---------|-----|--------|---------------|
| `TRIG_CSI1` | Camera 1 | **1.5 kΩ + 1.8 kΩ** divider → **XTR+**; **XTR−** → **GND** (2-wire cable) | **Populate** |
| `TRIG_CSI2` | Camera 2 | Same | **Populate** |
| `TRIG_BNC` | Bench scope (**`TP_TRIG`** / optional BNC) | **50 Ω** | **Populate** (test pads default; **J4** optional) |
| `TRIG_USB` | USB3 GS cam (optional) | Optoisolator LED + **220 Ω** (separate return) | **DNP** |

**XTR+ / XTR−:** vendor boards label the pair **XTR+** and **XTR−** (InnoMaker: **Trig+** / **Trig−**). **XTR− is ground**, not a second trigger input. Both cameras can share the same buffered edge on their **XTR+** lines (star at the buffer); each **XTR−** ties to local module ground.

**Buffer enables:** on each **74LVC2G241**, tie **1OE → GND** and **2OE → +3V3D** so both outputs drive ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 2).

**1.8 V at the camera:** IMX296 **XTRIG** is **1.8 V** logic. The buffer outputs **3.3 V** — the HAT uses the Pi GS **1.5 kΩ + 1.8 kΩ** divider per **`TRIG_CSI`** channel (**R51/R53 + R57/R59** on `trigger_io`), **directly** from **U1** with **no** extra **100 Ω** ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 2b).

**Not used:** **XVS**, **XHS**, **MAS** on Waveshare/Pi-style IMX296 boards — those are for sensor master/slave sync, not our MCU fan-out ([`h01`](h01_camera_subsystem.md) §3.1.1).

All CSI trigger outputs see the same edge within &lt;2 ns of each other, far below our timing budget.

### 3.3 Camera reaction

- **IMX296 (Pi GS, Waveshare, InnoMaker, etc.)**: in **fast trigger mode**, each valid **XTRIG** pulse captures one frame. `request.metadata.SensorTimestamp` (libcamera / picamera2) is the CSI start-of-frame time on the **Pi monotonic** domain (not the MCU counter). Characterize the fixed offset between the **XTRIG exposure-start edge** (scope: **`TP_TRIG`**) / MCU `TRIGGER_EVENT` stamp (PIO IRQ at the edge, §3.1) and `SensorTimestamp` once per rig.
- **Arducam UC-599 (OV9281 USB3)**: similar; onboard FPGA + V4L2 PTS — different pin name, same HAT opto path.

### 3.4 Trigger profiles and host binding

MCU trigger settings live under **`mcu.trigger`** in `~/jtsmm.yaml`. Resolved by `firmware/jtsmm/src/domain/trigger_config.c` + PIO driver `trigger_pio.c`. Full key reference: [`s08_runtime_configuration.md`](../software/s08_runtime_configuration.md) §`mcu.trigger`.

| Profile | `rate_hz` | `pulse.width_us` | Outputs | Use |
|---------|-----------|------------------|---------|-----|
| **`imx296_dual_csi`** | **30** | **12000** | csi1+csi2+scope | **Default Track B** |
| `imx296_high_fps` | 100 | 2000 | csi1+csi2+scope | High-speed motion |
| `bench_scope` | 10 | 1000 | scope only | Oscilloscope check |
| `track_a_led` | 30 | 200 | — (LED via `flash on`) | Webcam + LED sync *(HAT LED DNP on GS build)* |
| `custom` | explicit | explicit | explicit | Full control |

`outputs` are read-only: PIN_TRIG drives both buffers with their enables tied on, so CSI1, CSI2 and the scope tap always pulse; the firmware refuses any other combination (and `sync_led: true`).

**Host cross-check** (enforced in `software/jtsmm/jtsmm/config/validate.py` before record):

- `mcu.trigger.rate_hz` must equal each `host.cameras.list[].trigger.expect_rate_hz`.
- `host.cameras.list[].exposure_us` ≤ `mcu.trigger.pulse.width_us + 14.26` (IMX296 fast-trigger margin).

**Session start:** enable `imx296` `trigger_mode` sysfs → `config apply` (includes trigger block) → EMG/IMU streams + `pps` → cameras start (picamera2 external trigger) → `trig_start` (`host.capture.trigger_after_cameras`, default true) → verify `TRIGGER_EVENT` count vs camera `SensorTimestamp` ([`s03_camera_pipeline.md`](../software/s03_camera_pipeline.md)).

**Trigger-only bench** (no EMG/IMU, no full session): prefer `jtsmm trigger [--rate 30] [--width-us 12000]` on the host (sends CDC `trig_start …` only). Use `jtsmm trigger --stop` / `--status` to disarm or query. This does **not** run `config apply` and does **not** start EMG/IMU streams.

### 3.5 Polarity verification checklist (before first capture)

Run on the bench with a scope on **`TP_TRIG`** (or BNC **J4**):

1. Flash **production** firmware (`firmware/jtsmm/build/jtsmm.uf2`).
2. `config apply` with `mcu.trigger.profile: imx296_dual_csi`, then start the pulse with `jtsmm trigger` (or raw `trig_start` / `trig_start 30` over CDC).
3. Confirm pulse rate = the configured `rate_hz` (**40 Hz**, 12 ms low time with `etc/jtsmm.yaml` as shipped; raw `trig_start`, the firmware default and the `imx296_dual_csi` profile are **30 Hz** / 12 ms).
4. Confirm **low-time** on XTRIG matches IMX296 fast-trigger expectation (integration starts on falling edge) — invert `pulse.polarity` in YAML if exposure starts on the wrong edge.
5. Confirm the **idle level**: HIGH before `trig_start` and after `trig_stop` with `imx296_fast` (LOW with `active_high`), no glitch at `trig_start` (first edge is the falling exposure-start edge one HIGH phase later), and after a `trig_stop` issued mid-exposure the line returns HIGH without an extra falling edge; the camera must deliver no frame while idle.
6. On Pi: `echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode` before `jtsmm record`.
7. During a short record, `TRIGGER_EVENT.trigger_id` must increase monotonically and match camera frame count ±1.

See [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §4.1 (`trigger_gen` PIO) and [`firmware/jtsmm/README.md`](../../firmware/jtsmm/README.md) bench commands.

## 4. LED-flash protocol (Track A — optional on HAT)

For webcams there is no electrical trigger path. We use **photonic synchronization**: a **visible** high-brightness LED in the cameras' shared field of view is pulsed at known `t_master_us` instants; the capture process detects the LED-on event in each camera's image stream.

**Default GS build (2× IMX296 on XTRIG):** this block is **not populated** on the HAT — camera time comes from **`TRIGGER_EVENT`** + **`SensorTimestamp`** (§3). Populate the sync LED only for **Track A** or an optional cross-check.

### 4.1 LED hardware (on HAT, not an IR ring) — **optional**

- **White** LED (e.g. 5 mm, 15–25 mA), on a 2-pin cable from the HAT `SYNC_LED` JST ([`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §7).
- Driven on GPIO14 by a firmware `repeating_timer` + one-shot alarm (`led_flash.pio` exists but is unused) via a small N-FET on the HAT.
- Mount near the ChArUco board or above the subject so **all** cameras see it without blocking AprilTags on the face.
- Pulse width: 200 µs–2 ms (tune so at least one camera frame captures the flash at 30–60 fps).
- Pulse rate: 1 Hz nominal; burst at session start for initial affine fit.

**Do not** confuse this with 850 nm IR illumination for retroreflective spheres — that hardware is **not** in the default BOM when using AprilTag tracking, and draws **no power from the HAT** even when used (external 5 V rings; see [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §1.2).

### 4.2 LED-on detection

The capture process maintains a small ROI around the LED position. For each frame:

1. Sum the pixel intensities in the ROI.
2. Compute a running baseline and threshold.
3. Detect a "high-intensity" event = LED-on frame.
4. The frame's PTS is paired with the LED-on `t_master_us` (sent in a `LED_FLASH_EVENT` record by the firmware).

### 4.3 LED-on pairing precision

The webcam captures a single frame at the LED-on event with no information about *when in the exposure window* the LED actually turned on. The single-pairing precision is therefore ±0.5 frame intervals: **±8 ms at 60 fps**, and **±16 ms at 30 fps** (the common Logitech C920 rate — see [`../software/s02_data_pipeline.md`](../software/s02_data_pipeline.md) §7). We then **fit an affine model over many flash events** (typically tens to hundreds of paired observations over a session) to drive the residual down to ~1–3 ms standard deviation; running Track A webcams at 60 fps (1280×720) rather than 30 fps (1080p) halves the per-pairing error and is recommended when sync matters more than resolution.

The fit is online and ridge-regularized to be robust to occasional false-positive LED detections.

## 5. Clock-disciplining algorithm

**Target:** the Pi maintains an affine map from each *foreign* clock to the master:

```
t_master_us = a_k * t_local + b_k
```

for clock `k` ∈ {Pi monotonic, Cam0 SensorTimestamp, Cam1 SensorTimestamp, Track A UVC PTS, …}.

**Important:** EMG and both IMUs do **not** use this fit — they already carry MCU `t_master_us`.

| Mechanism | Target behaviour | Implemented today? |
|-----------|------------------|--------------------|
| Pi PPS handshake | `pps` ↔ `PPS_REPLY`, midpoint RTT → `_pi_fit` | **Yes** — `host.capture.pps_samples` (8) at take start + end; offset-only fit until ≥ 30 s span |
| Track B trigger match | Match frame ↔ `TRIGGER_EVENT`; store trigger `t_master_us` | **Yes** — exact id when provided; PPS-anchored first frame + cadence tracking (prediction / slip detection as fallbacks) when camera `trigger_id` is missing ([`s06`](../software/s06_time_synchronization.md) §4.1) |
| Track B affine residual | Fit `(SensorTimestamp, t_master)` for diagnostics | **Partial** — fit updated; **stored** camera time is still the trigger stamp, not `fit.map()` |
| Track A LED-flash fit | Pair UVC PTS with `LED_FLASH_EVENT` | **Target** |

Code: [`software/jtsmm/jtsmm/sync/clock_align.py`](../../software/jtsmm/jtsmm/sync/clock_align.py). Full software narrative: [`s06`](../software/s06_time_synchronization.md).

## 6. Validation experiment

This is the headline measurement of the project. Procedure:

### 6.1 Setup

- Connect a 4-channel oscilloscope (≥100 MHz) to:
  - Ch1: MCU trigger output (**`TP_TRIG`** on HAT, or optional BNC **J4**).
  - Ch2: Pi GS Camera 1 exposure-active pin (a fly-wire to the camera board's CTL pin during validation only; remove for normal operation).
  - Ch3: Pi GS Camera 2 exposure-active pin.
  - Ch4: sync visible LED drive line.
- Start `jtsmm record --mode validation --duration 60` on the Pi.

### 6.2 Measurements

For 1 minute, the system logs:

1. From the scope (via SCPI to a small Python helper): per-trigger Δt between Ch1 rising edge and Ch2/Ch3 rising edges. Expected ~10 µs constant offset for the IMX296, jitter <5 µs.
2. From the Pi capture process: per-frame `(t_master_us from TRIGGER_EVENT) - (a · SensorTimestamp + b)`, where `(a, b)` are from `clock_align`. Expected jitter <100 µs.
3. From Track A cameras: per-frame (a · UVC_PTS + b) - (LED-flash t_master_us), at the rate of LED flash events. Expected jitter <2 ms.

### 6.3 Outputs

Plots emitted by the validation tool:

- Histograms of (trigger → sensor exposure) latency for each Track B camera.
- Histogram of (Pi-computed t_master_us from PTS) − (MCU-emitted t_master_us at trigger). Reports mean (the constant offset) and standard deviation (the jitter we care about).
- For Track A: same plot but using LED flashes. Reports the residual after the affine fit.

These plots are the project's headline figure for the publication on the new system.

## 7. Time-coherent storage

The implication of the clock model is that the HDF5 file holds **only `t_master_us`**, not the local clocks. The local clocks are kept in a separate group `/sync/` for debugging and reproducibility:

```
/sync/
├── pi_to_master           (S, 4) [t_pi_ns, t_master_us, a, b]   — one row per fit update
├── cam0_to_master         (S, 4) [t_cam_ns, t_master_us, a, b]  — same per camera
├── cam1_to_master         (S, 4) — same
├── cam2_to_master         (S, 4) — same (only for 3-camera setups)
└── attrs: residual_rms_us per stream, n_used per stream
```

Each `*_to_master` dataset is a **time series** of affine-fit snapshots (so drift is auditable), not a single scalar fit. This is the exact layout written by the `hdf5_writer`; the canonical schema lives in [`../software/s02_data_pipeline.md`](../software/s02_data_pipeline.md) §8. Downstream analysis code never has to think about clock domains; it indexes everything by `t_master_us`.

## 8. Failure modes

| Symptom | Likely cause | Diagnostic + remedy |
|---------|--------------|---------------------|
| Camera frames have NaN `t_master_us` after `clock_align` | No trigger events received yet | Check **`TP_TRIG`** / optional BNC; **XTR+ / XTR−** cable; `trigger_mode`; Pi **R11** mod on official GS board |
| Jitter histogram has 16 ms spikes | UVC driver dropped frames; PTS does not advance monotonically | Increase USB bandwidth allocation; lower camera resolution; pin capture process to a core |
| Affine fit residual >5 ms on Track A | LED detection misfires (false positives) | Increase detection threshold; move LED away from windows; reduce ambient IR |
| `clock_align` reports negative `a` for Pi monotonic | Pi clock stepped backwards (NTP correction) | Use `CLOCK_MONOTONIC_RAW`; disable chrony slewing during a session |
| MCU `t_master_us` wraps | We use 64-bit; would take ~580 000 years; if you ever see this, you have a bug | Read the firmware fix in `clock.c` |

## 9. Future extensions

- **GPS PPS input** (future hardware on a spare GPIO; no PIO program exists yet) for absolute UTC time alignment. **v1:** no GNSS wire on the HAT; Pi↔MCU sync uses the USB **`pps`** handshake ([`s06`](../software/s06_time_synchronization.md)).
- **Wired PTP slave** on the Pi 5 if a lab requires IEEE-1588 alignment with other equipment.
- **Hardware-trigger for Track A** via a USB cable that exposes the camera's internal trigger pin — requires opening the webcam, not a serious option in v1.
