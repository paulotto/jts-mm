<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# op03 — Troubleshooting

A symptom-indexed troubleshooting guide for the most common (and a few less common but instructive) failure modes encountered during bring-up and operation. Organized by symptom; each entry lists likely causes and recommended actions.

## 0. Test pad map (LEDs vs probes)

**DMM** = digital multimeter (voltage, resistance, continuity). **Scope** = oscilloscope (waveforms, timing).

Status **LEDs** show rail presence and firmware activity; **test pads** allow exact voltage and timing checks. Canonical table: [`h02_custom_hat_shield.md`](../hardware/h02_custom_hat_shield.md) §6.1.

**Test pads on the board (14, top side):** `TP_5VIN1`, `TP_3V3D1`, `TP_5VA1`, `TP_DGND_PWR1` (power); `TP_ADS_DRDY1`, `TP_VREFP1`, `TP_AGND1`, `TP_BIASOUT1`, `TP_VCAP1` (EMG); `TP_TRIG1`, `TP_DGND_TRIG1`, `TP_CSI1`, `TP_CSI2`, `TP_TRIG_PIO1` (trigger). The HAT+ pads `GND1` / `WP1` are on the bottom side. The MCU and IMU nets have no test pads; the table names the probe point to use where a pad does not exist.

| Symptom / check | Test pad | Alternative probe point | LED hint |
|-----------------|----------|-------------------------|--------|
| Short to GND before power | `TP_5VIN1` → `TP_DGND_PWR1` (Ω) | Pi header 5 V (pins 2/4) vs GND | — |
| Power sequence | `TP_5VIN1`, `TP_3V3D1`, `TP_5VA1` (ground: `TP_DGND_PWR1`) | `J_IMU_MTA1` pin 1 vs pin 2 for 3V3; Pi header pins 2/4 for 5 V | 5V→3V3→6.5V-PG→5VA chain |
| MCU core | No pad (`DVDD` is not exposed) | `LED_HB1` blink / USB enumerate | `LED_3V3D_OK1` on only |
| Pre-SWD flash | `TP_3V3D1` ≈ 3.3 V | `J_IMU_MTA1` pin 1 vs pin 2 | 3V3 LED |
| ADS1299 does not start | `TP_VREFP1` ≈ 4.5 V (return: `TP_AGND1`); scope `TP_ADS_DRDY1` @ 4 kHz (default rate) | `selftest` / `ads_regs` over USB-CDC | `LED_EMG_LO1` |
| IMU does not start (`imu_status`) | No pad on the IMU nets | Logic analyzer at `J_IMU_MTA1` pin 3 (SDA), pin 4 (SCL), pin 5 (INT), harness off; CRA / DP behind the mux at `J_IMU_CRA1` / `J_IMU_DP1` pins 3–5 | `LED_MTA1` flicker |
| Camera sync / NaN timestamps | Scope: `TP_TRIG_PIO1` vs `TP_CSI1` / `TP_CSI2` (~3.3 V vs ~1.8 V); ground: `TP_DGND_TRIG1` | `TP_TRIG1` vs `J_TRIG_CSI1` pin 1 | `LED_TRIG1` amber |

Use **`TP_AGND1`** (not `DGND`) as scope return for `TP_VREFP1` and `+5VA` ripple. Do not probe USB D+/D− — use `J_USB1`.

## 1. Boot / enumeration issues

### `/dev/jtsmm_hat` does not appear after plugging in

- **Check USB-C cable**: confirm it is a data cable (some USB-C cables are charge-only). Try a known-good cable.
- **Check power LED on HAT**: if off, the 5 V rail is not reaching the HAT. Re-seat the HAT on the Pi 5 header.
- **Check Pi 5 dmesg**: `dmesg | tail` — look for USB-CDC enumeration errors.
- **Check udev rule installation**: `ls -l /etc/udev/rules.d/99-jtsmm.rules`; `sudo udevadm control --reload-rules`.
- **Power-cycle**: `sudo shutdown -h now`, wait 10 s, power up. The MCU sometimes ends up in a state where it needs a full power-cycle, not just a USB unplug.

### HAT enumerates but `jtsmm info` returns no response

- **Firmware corrupted or absent**: confirm **`TP_3V3D1`** ≈ 3.3 V (or **`J_IMU_MTA1` pin 1** vs pin 2), then re-flash via **3-wire SWD** (Raspberry Pi Debug Probe — GND/SWCLK/SWDIO only; see [`../hardware/h08_rp2350_flashing.md`](../hardware/h08_rp2350_flashing.md)), or hold BOOTSEL and drag the `.uf2` onto the **RP2350** volume.
- **Wrong baud / framing**: `pyserial` settings should be 115200 baud × 8N1 by convention (USB-CDC ignores baud but pyserial wants it). Check `software/jtsmm/jtsmm/capture/serial_util.py`.
- **Reset loop**: CDC `info` prints `last reset:` (HardFault / watchdog / power-on). The hardware watchdog exists only in builds with `-DJTSMM_WATCHDOG_MS=4000` (default off).

### `BulkReaderError: No jtsmm USB device` or `Could not claim vendor interface`

The binary record stream is read from a **USB vendor bulk** endpoint via libusb (not the CDC serial port), so it needs device access beyond the `/dev/ttyACM0` node:

- **udev/libusb permission**: the vendor interface needs a udev rule granting the user access. Confirm [`software/jtsmm/etc/udev/99-jtsmm.rules`](../../software/jtsmm/etc/udev/99-jtsmm.rules) is installed (`ATTRS{idVendor}=="2e8a", ... MODE="0666"` or `plugdev` group), then `sudo udevadm control --reload-rules && sudo udevadm trigger`. Without it, `open()` fails even though CDC text still works.
- **Interface busy**: another process (a previous `jtsmm record`, or a lingering capture service) is holding the vendor interface. Stop it: `pkill -f "jtsmm record"`.
- **libusb missing**: `python -c "import usb.core"` must succeed; install `libusb-1.0-0` and the `pyusb` dependency (`pip install -e software/jtsmm[dev]`).
- **CDC works but bulk does not**: `jtsmm info` uses CDC and will still respond; a capture that produces *no* records but a healthy `info` points squarely at bulk permissions.

### USER LED stays blue / CAPTURE LED on after `jtsmm record` or bench

- **Streams still running**: blue USER RGB = `PANEL_STATE_STREAMING`; CAPTURE LED = EMG stream active. Send teardown over **J_USB1**:
  ```text
  trig_stop
  imu_stream cra off
  imu_stream mta off
  ads_stop
  ```
  Or run `./scripts/hil/run_bench_hw.sh` (H8 teardown) / host `stop_mcu_streams()` after session end.
- **Bench left H5 streams running**: Phase 7 bench uses `trap` + H8 to stop streams; re-flash Phase 7+ firmware if `ads_stop` does not return panel to idle green.

## 2. EMG quality issues

### Heavy noise appears and disappears abruptly, and touching the electrode pins changes nothing

This is the signature of **corrupted SPI framing, not an analog problem**. If the recorded trace does not respond at all to touching the electrode pins, the samples are no longer a function of any electrode: the ADS1299 RDATA byte stream has lost alignment, which shifts the 24-bit channel words by one or more bytes. A one-byte shift scales values by ~256 and rotates data between channels, so it looks like a sudden burst of full-scale hash.

Confirm it from the counters rather than from the trace:

- Live: the viewer health line shows **`FRAMING SLIP <n>`** in red and the EMG chip goes red.
- On the USB console: `EMG FRAMING SLIP: … This is corrupted SPI framing, not electrode noise.` once per second while it is happening, plus `EMG framing recovered` when it stops.
- After the session: `jtsmm record` prints a `WARNING: ADS1299 RDATA framing slipped …` line, and `jtsmm integrity` **fails** with `ADS1299 RDATA framing slipped`.
- Any time: `emg_guard` over CDC prints the counters (`misaligned`, `dropped`, `worst_streak`, `resyncs`, `spi_wedges`).

Causes, most common first:

- **A bench command was issued while the stream was running.** `ads_status`, `imu_status`, `selftest` and `ads_drdy_watch` drive SPI0/I²C from the comms core while the sensor core owns those buses, which slips the byte stream. These now refuse with `… REFUSED: EMG stream active` unless you append `force`; if you see the FORCED warning in the log, that was the cause. Run `ads_stop` first.
- **SPI signal integrity**: check the SCLK/MISO/CS routing and that `platform.spi0.hz` is not raised above 4 MHz.
- **SPI DMA wedging**: a non-zero `spi_wedges` means transfers are timing out and being aborted; investigate bus contention before trusting the data.

Misaligned frames are dropped rather than published, so the affected span shows up as *missing* samples (an EMG continuity gap) instead of plausible-looking garbage. Data outside the slip window is unaffected.

### An IMU stream stops in the middle of a recording

- The comms core logs `IMU WATCHDOG: mta stalled 3s, recovery attempt 1/6` (CDC + session log), retries every 10 s, and after six attempts logs `… giving up; check cable/mux/power`. Before 2026-09 it tried exactly once and then went quiet.
- `imu_status` shows `i2c_tx_aborts=`: a non-zero count means the BNO085 NACKed reads (reset, cable, mux). Each one costs at most one packet now; on older firmware a single NACK wedged the DMA reader until the watchdog ran.
- `jtsmm integrity` fails the session with `imu_…: stream ends N s before the end of the session -- it stopped early`. The same check catches an EMG stream stopped by the panel **USER1** button (`PANEL: USER1 pressed -> capture STOPPED from the panel` in the log).

### Camera frames land on the wrong EMG time / integrity warns about camera pairing

- `trigger stamps: pio_irq=N` at the end of `jtsmm record` (and `sync/report_json.trigger_stamp`) says how `TRIGGER_EVENT` was timed. `timer=…` with a `fallback` count means the PIO IRQ went silent and the software timer took over (`TRIGGER STAMP FALLBACK` in the log) — run `trig_status` on the HAT; `pio_pulses=0` with `timer_events>0` means the IRQ never fired on this board.
- `camera cam0: assigned-trigger minus PPS-mapped SensorTimestamp: median … (+X frames), drift Y us/s` — |X| ≳ 1.5 means the frame↔trigger sequence association is off by that many pulses (check `capture_order` is `sensors,cameras,trigger`); |Y| ≳ 50 µs/s means the trigger stamps drift against the camera clock (check `trig_comp` is `on`). Both are also `jtsmm integrity` warnings.
- `pps[start]: accepted 0/8` — the firmware does not answer `pps` (older than 2026-09) or the CDC/bulk link is broken; the cross-check above is unavailable, everything else still works.
- `WARNING: host capture queues overflowed` — the Pi fell behind and dropped records (integrity error); reduce load on the Pi, not the HAT.

### Clean with the debug probe or the HAT's USB on a laptop, noisy on the Pi's own supply

**Cause: leakage from the Pi's mains power supply**, not software. A switch-mode supply has a
small safety capacitor from its mains side to its 5 V output, so the Pi and HAT ground swing
tens of volts of 50 Hz against earth, driven only by a few µA. The subject is near earth, the DRL
can push only a few µA through its 220 kΩ, and it saturates. Anything that ties the circuit ground
to a large reference drains that leakage and hides the problem: the debug probe (its GND wire,
`J_SWD` pin 1, goes to the laptop), the HAT's USB on a laptop (its chassis, charger earth, your hand
on it). A power bank has no mains inside the system and needs none of that.

Measured 2026-09-28, same board, a finger on CH5 and DRL:

| | HAT USB on the laptop | HAT USB on the Pi (mains supply) |
|---|---|---|
| CH5 mains (50 Hz + harmonics) | 80–108 µV rms | ≈ 17 600 µV rms |
| CH5 noise 20–450 Hz, mains removed | 4–9 µV rms | ≈ 900 µV rms |
| DRL bias node, mains swing | 1.8 mV rms | 163 mV rms |

Symptoms: good with the debug probe connected and bad as soon as it is unplugged; good on a power
bank; bad on the Pi supply with nothing else attached. Software, firmware and config are the same in
all cases — the recorded samples already differ, so the viewer is not the cause.

Fix: **run the Pi from a USB-C PD power bank** (`op04` E1) and use the viewer over WiFi. With
electrodes on a person, never connect the HAT or the Pi to mains-powered equipment by cable (laptop
USB, debug probe, monitor, Ethernet to an earthed device) and do not ground the circuit on purpose:
the HAT is not isolated, so that path runs through the subject (`op04` §2). For mains operation an
isolated power path is needed (`h07` §4).

### The trace is all noise, touching the electrodes changes nothing, and the reference does not help

This is **mains interference with a saturated DRL loop**, not a firmware fault. `jtsmm integrity`
now names it directly:

```
error: DRL/bias amplifier saturated: the CH8 BIAS_DRN readback sits at its rail in 45% of the session...
error: EMG masseter_l: mains interference dominates 100% of the session (100% of 20-450 Hz power
       at 50 Hz and harmonics, 7600 µV RMS in-band vs 5-20 µV for resting skin EMG)...
```

CH8 reads back the bias node. **Judge its AC content, not its absolute level:** `IN8P` is unwired
on this board (`h03` §5), so a healthy, quiescent loop reads *pinned at full scale*. Measured on
one subject, minutes apart, same electrodes:

| | bias-node AC | rail fraction | muscle channels |
|---|---|---|---|
| MVC (excellent) | **3.9 mV** | 0.99 | 166–231 µV |
| take1 (unusable) | **219 mV** | 0.04 | 405–3179 µV |

A node swinging hundreds of mV is a loop being driven *by* the common-mode instead of rejecting
it, so the reference electrode adds interference rather than removing it — which is why it appears
not to help. In take1 every muscle channel correlated with that node (|r| up to 0.85) and was
61–84 % mains, with the worst channel being the one with the highest electrode impedance.

It also explains the two symptoms that look like a bug:

- **Touching the electrode changes nothing** — the trace is driven by a saturated loop fighting
  several hundred mV of common-mode; a fingertip cannot move that.
- **Only connect/disconnect changes the trace** — that is the one action that alters the loop's
  impedance and operating point.
- **Sometimes excellent, sometimes pure noise** — the loop is effectively bistable: inside its
  linear range CMRR works and the EMG is clean; once it clips you get ~100× hum, and it flips
  abruptly. `emg_signal.channels.<name>.bad_window_frac` reports what fraction of the session was
  affected, so an intermittent fault reads as a fraction rather than an average.

Fix, in order:

1. **Re-prep the DRL/reference site** (abrade, fresh gel, firm contact) and check continuity from
   the HAT DRL pin to the electrode with a multimeter. Getting the bias amp out of clipping is the
   highest-value action.
2. **Remove the mains coupling**: run the Pi from a power bank (its own mains supply leaks 50 Hz
   into the circuit ground, see the entry above), unplug mains-powered laptops from the Pi (or use
   an isolated hub), move away from mains cabling and metal chair frames, check the patient-cable
   shield.
3. **Re-prep the muscle electrode.** A DC offset of ~9 mV with 6 nA lead-off implies ~1.5 MΩ of
   source impedance, 30–300× a prepped electrode. Confirm with `jtsmm emg impedance`; target
   < 50 kΩ. Note the lead-off comparator can read *connected* at that impedance — it detects an
   open, not a bad contact.
4. **Do not skip `jtsmm emg check`** (`--no-emg-check`): it measures these numbers in ~15 s.

`jtsmm emg check` now measures this directly too, as a `bias amp (DRL)` row carrying the rail
percentage — so you see it in ~15 s instead of after a 60 s take.

To confirm it is not the SPI path: `frame_misalign` stays 0, and with channels powered down
(`--active-channels`) their columns in `/emg/samples` are *exactly* zero — any byte slip would
scatter data into them. On the session above that held for all 733,026 values.

#### Why grounding the cable shield does not fix it

A shield stops **capacitive pickup into the cable**. The dominant path here is different: mains
couples capacitively into the **subject's body**, producing a common-mode voltage on everything
attached to them, which then becomes differential through **electrode impedance mismatch**. A
shield cannot touch that. The measured bias-amp swing (≥ 390 mV of 50 Hz, clipped) is body-level
common-mode, far more than cable pickup would produce.

Two ways a shield can make things *worse*:

- **Grounded at both ends** it forms a ground loop. Ground the shield at the **amplifier end only**.
- **Tied to a dirty ground.** If the Pi runs from a non-isolated mains supply, its ground already
  carries mains leakage, and the shield then drives that into the front end. An isolated supply
  fixes the cause; a shield tied to a noisy reference just distributes it.

#### Why notch filtering does not rescue the recording

Four separate reasons, all visible in the data:

1. **The interference is not additive any more.** Once the DRL clips, the corruption is a
   *nonlinear*, signal-dependent product, not a clean sinusoid. On the measured session, removing
   the fundamental and nine harmonics with a per-window least-squares fit still left ~1030 µV RMS —
   50–200× resting EMG. You cannot subtract what a nonlinearity has smeared across the band.
2. **The SNR is hopeless.** 7800 µV of interference against 5–20 µV of EMG is about −52 dB. A
   notch that removes 30–40 dB at one frequency still leaves interference far above the signal.
3. **Mains is not exactly 50.000 Hz and it wanders.** Measured 49.979 Hz; a single global sine fit
   explained only 67% of the variance over 61 s while per-2 s fits explained 98% — that gap *is*
   the wander. Fixed notches drift off, and the error grows with harmonic number.
4. **The viewer used to notch only the fundamental.** `CausalEmgFilter` now defaults to
   `notch_harmonics=4`, each notch ~5 Hz wide, built at `host.emg_check.line_hz` (so a 60 Hz lab
   notches 60/120/180 Hz; replay uses whatever the session was recorded with).
   On the session above that takes the displayed trace from 2351 µV RMS to 748 µV — better, and
   still ~40–150× resting EMG, which is the point: the harmonics were never the real problem.

Filtering is the last step, not the fix. Get the DRL out of clipping first; then a notch removes a
small residue instead of being asked to recover a signal that was never resolved.

#### Reducing mains influence — what actually works, in order of effect

**Reduce the common-mode at the source** (largest lever):

- Run the Pi from a **battery or medical-grade isolated supply**; a mains-powered laptop plugged
  into the Pi is the most common single cause.
- Remove mains-powered equipment from contact with the subject (laptop PSU, monitor, metal chair
  frame, adjustable bed).
- Move away from mains wiring, fluorescent/LED ballasts and power strips; cross power cables at
  90° rather than running parallel.

**Reduce common-mode → differential conversion** (this is the actual mechanism):

- **Lower and match the electrode impedances.** Conversion scales with the *mismatch*, so this is
  the highest-value electrode action: abrade, fresh gel, let it hydrate 1–2 min, firm contact.
  Target < 10 kΩ and both inputs of a pair within ~20% of each other. Verify with
  `jtsmm emg impedance`. (The rig session implied ~1.5 MΩ.)
- Keep leads short and **twist each P/N pair** so both wires see the same field.

**Make the DRL work:**

- Low-impedance DRL contact on a bony, electrically quiet site, as close to the recording
  electrodes as practical.
- Ensure `bias.enabled: true` and that `bias.channels` senses the channels you are actually
  recording — with `--bias-channels 1` the loop references a single electrode.
- Confirm it is not clipping (the `bias amp (DRL)` row, or `emg_signal.drl_rail_frac`).

**Wiring / hardware:**

- Shield grounded at the amplifier end only; a **driven/guard shield** (shield driven at the
  common-mode potential) is the proper fix for cable leakage without loading the input.
- Ferrites on the USB and power leads.

**Acquisition settings:**

- Prefer `leadoff.during_recording: false` for real takes once contact is verified: 6 nA through a
  high-impedance electrode adds a DC offset (9 mV on the rig) and the current sources add noise.
- If hum is large, a lower PGA gain preserves headroom in the analog front end.

Only after the above: notch the fundamental **and its harmonics**, or use the spectral method in
[`../software/s04_emg_signal_processing.md`](../software/s04_emg_signal_processing.md) §4.

### High 50 Hz noise on EMG

- **DRL electrode not connected** or has poor skin contact: re-prep the mastoid site; verify continuity from the **DRL lead** (DRL pin header **`J_EMG_DRL1`** on the HAT) to the electrode with a multimeter.
- **Pi on its mains power supply**: its leakage drives 50 Hz into the circuit ground and saturates the DRL — run it from a power bank (entry *Clean with the debug probe…* above).
- **Mains-powered laptop connected to the Pi 5**: disconnect or interpose an isolated USB hub.
- **Subject's chair has a metal frame touching mains-tied equipment**: move the chair.
- **EMG cable shield damaged**: replace patient cable.

### EMG channel saturated

- **Electrode loose or fallen off**: re-prep the site.
- **Open/disconnected input, not a real fault**: DC lead-off is on by default during `jtsmm record`/`jtsmm view` and intentionally pegs an unconnected input to a rail — check the channel's lead-off LED/`emg/leadoff_status` bit before assuming saturation. Reconfirm with `--no-leadoff` (pegging stops once the input floats instead) if you need to rule this out.
- **AC lead-off left on**: the 31.2 Hz impedance tone (`leadoff.ac_tone`) can also mimic saturation on a connected channel; this should only ever be on inside `jtsmm emg impedance`/`jtsmm emg check`, never during `jtsmm record`.
- **PGA gain too high**: lower the channel's gain from 12 to 6 via the YAML config.

### One EMG channel reads pure noise (no muscle signal)

- **Wrong polarity**: verify the active and reference electrodes are correctly assigned (check the wire colors against the cable manufacturer's diagram).
- **Electrode dried out**: shelf life of disposable Ag/AgCl is ~12 months; check the box's expiry.
- **Electrode placed off the muscle belly**: re-palpate the muscle while the subject activates it; re-place the electrode.

### Lead-off detection reports a channel "off" but you can see good contact

- **Cable break**: do a continuity test with a multimeter.
- **ADS1299 register misconfigured**: power-cycle the HAT to reset register state, or re-run `jtsmm config apply`; inspect with `ads_regs`.

## 3. IMU issues

### IMU quaternion frozen at last value

- **SHTP transport hang**: firmware automatically retries; if persistent, pulse `nRST` (or power-cycle the HAT, which cycles power to the IMU breakout).
- **Cable detached at 7-pin JST-SH connector**: check both ends (HAT **SH** and breakout **GH**).
- **Shield open / noisy I2C**: verify cable **pin 7** continuity to braid; HAT pin 7 must land on **`AGND`**; breakout pin 7 must **not** be soldered to breakout **GND** ([`h04_imu_cable_assembly.md`](../hardware/h04_imu_cable_assembly.md)).
- **I2C not ACKing**: verify the 2.2 kΩ SDA/SCL pull-ups and that `nRST` is released high ([`../hardware/h04_imu_subsystem.md`](../hardware/h04_imu_subsystem.md) §4).

### IMU quaternion jumps 180° at a specific orientation

- **Sign ambiguity in game-rotation-vector**: q and −q are the same rotation; enforce sign continuity (q·q_prev ≥ 0) in the analysis step before differencing or interpolating.
- **Mounting transform incorrect**: re-do the §3 procedure in [`../calibration_validation/cv02_imu_calibration.md`](../calibration_validation/cv02_imu_calibration.md).

### IMU drift much higher than expected

- **Magnetometer interference**: check for ferromagnetic objects near the MTA / CRA; switch to game RV (which doesn't use mag) if not already.
- **Gyro bias not subtracted**: ensure `gyro_bias` from calibration is being applied; in particular, BNO085's on-chip bias estimator may need a few seconds at session start before it stabilizes.
- **Temperature effect**: warm the IMU breakout for ~5 min before calibrating mounting transform.

## 4. Camera issues

### `capture_cam` exits at startup

- **libcamera daemon not running**: `sudo systemctl restart camera-app` or `pkill libcamerify`.
- **Camera not detected**: `libcamera-hello --list-cameras` should show all CSI cameras. If not, check the ribbon cable seating.
- **Permission denied**: ensure the user is in the `video` group; re-login.

### Frames have NaN `t_master_us` even though `clock_align` is running

- **Trigger events not arriving**: check **`TP_TRIG1`** on the scope (ground: **`TP_DGND_TRIG1`**); compare **`TP_TRIG_PIO1`** (PIO, ~3.3 V) vs **`TP_CSI1`** / **`TP_CSI2`** (~1.8 V); the firmware logs each trigger.
- **Trigger cable not wired**: each CSI camera needs **2 wires** — **XTR+** (signal) and **XTR−** (ground). InnoMaker labels them **Trig+ / Trig−**; same thing.
- **3.3 V on XTRIG (divider not effective):** confirm the series resistors **R57 / R59 = 1.5 kΩ** and the resistors to `DGND` **R58 / R60 = 1.8 kΩ**; measure **~1.81 V** on **`TP_CSI1`** / **`TP_CSI2`** or on pin 1 of **`J_TRIG_CSI1`** / **`J_TRIG_CSI2`** (harness off).
- **Official Pi GS board:** external trigger may require **R11** removed — see [Pi external-trigger guide](https://www.raspberrypi.com/documentation/accessories/global-shutter-camera.html).
- **Not in fast-trigger mode:** `echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode`; in picamera2/libcamera use fixed gain/exposure (`external_sync` / pipeline flags per your OS image).
- **No trigger pulse on scope:** confirm firmware is running (`jtsmm info`), then arm trigger only with `jtsmm trigger` (rate from the applied config: 40 Hz / 12 ms with the template, the 30 Hz boot preset if none was applied; or CDC `trig_start 40 12000`) and check `jtsmm trigger --status` shows `running=1`. This does **not** start EMG/IMU. Stop with `jtsmm trigger --stop`.
- **Wrong edge / exposure:** IMX296 fast trigger uses **low pulse width** for shutter time; confirm on scope and match PIO polarity ([`h06_synchronization.md`](../hardware/h06_synchronization.md) §3.1).

### AprilTag / marker detection fails

- **Tags too small in image**: move cameras closer or use larger tags (≥30 px per side in the preview); see [`../04_component_catalog.md`](../04_component_catalog.md#9-fiducials--scene-lighting).
- **Glare or blown highlights on tags**: add diffuse light from the sides; lock exposure; avoid direct ceiling spots on the cheek.
- **Motion blur**: shorten exposure (`exposure_us` in `~/jtsmm.yaml`); prefer Track B global-shutter cameras.
- **Low contrast / wrong dictionary**: the printed tags must match `markers.*.dictionary` in `software/optical/etc/optical.yaml` (currently ArUco `DICT_4X4_50`); check print quality and contrast.
- **Tags occluded by hair / hands**: re-position cameras or subject; IMU bridging handles brief drop-outs but analysis needs stable optical pose.
- **Pose mismatch in `triangulate`**: re-check calibration; re-run `optical calibrate`.

*Optional retroreflective kit only:* if you enabled `marker_detector: retroreflective`, check IR ring power (~250 mA) and 850 nm filters — not used in the default build.

### Jitter histogram has a 16 ms or 33 ms periodic spike

- **Capture process is being preempted by another OS task**: close other CPU-heavy processes (`jtsmm record` sets no real-time priority or core pinning).
- **libcamera buffer underrun**: increase the buffer count in the picamera2 config.
- **Storage write blocking on rotational disk**: NVMe should not exhibit this; if running on SD card, switch to NVMe.

## 5. Sync issues

### `clock_align` reports residual >1 ms for Track B cameras

- **Trigger event not arriving from MCU**: check firmware log; verify `trig_start [rate_hz] [width_us]` was sent and acknowledged, and `trig_status` shows it running.
- **Trigger_id mismatch**: counter rolled over on one side but not the other (very unlikely; counters are 64-bit). Restart.
- **Frame drop in libcamera**: increases noise in the fit. Check `dropped_frames` in libcamera stats.
- **CSI cable too long or marginal**: replace with a shorter (≤30 cm) cable.

### `clock_align` for Pi monotonic shows scale `a` very far from 1.0

- **NTP slew enabled during session**: disable chrony/timesyncd for the duration of recording.
- **CLOCK_MONOTONIC slewed**: the host uses `CLOCK_MONOTONIC`, which NTP can slew; keep NTP from slewing during a take.
- **Too few PPS samples**: check the `pps[start]` / `pps[end]: accepted n/8` lines of `jtsmm record`; takes < 30 s get an offset-only fit (a = 1).

### `integrity` reports "optical vs IMU timing offset" (poses lag the IMU by a constant)

The camera frames were paired with the wrong `TRIGGER_EVENT`. Every pose is stamped a fixed number of trigger periods early, so ArUco looks fine, `residual_rms_us` looks perfect (the affine fit is seeded from the bad pairing), and only a cross-modal check sees it ([`s06`](../software/s06_time_synchronization.md) §4.3).

Symptom downstream: `jtsmm imu calibrate --mode fit` fails with a large residual (10–20° RMS) even though the IMUs are rigidly glued and the motion was slow.

- **Trigger armed before the cameras were delivering**: the offset equals `settle_ms` (or the libcamera startup time) × trigger rate. Fixed in the capture path — `trig_start` is now issued *after* `cam.start()` and `ClockAligner.reset_trigger_backlog()` drops anything buffered during startup. Check `sync.attrs.report_json` → `discarded_trigger_backlog`.
- **Recorded before that fix**: the session cannot be repaired; the true exposure time of each frame was never stored. Re-record.
- **Offset near one frame period only**: expected. Exposure midpoint versus trigger edge plus BNO fusion latency live in the ≤40 ms warn band.

### Track A LED-flash not detected in frames

- **LED out of frame**: re-aim the camera to include the LED ROI.
- **Detection threshold too high**: lower it (in the YAML config).
- **LED too bright and saturates the sensor**: lower drive current.
- **Sunlight in FOV**: close blinds; the sync LED must remain visible but not saturate the sensor.

## 5b. Capture drops / `integrity --strict` failures

### `integrity --strict` fails on EMG continuity or IMU drop rate

Start by reading the `mcu_stats` line (from `REC_STATS`, printed by the bench and `jtsmm record`) and ask **which counter moved**:

- **`missed_drdy` climbing** → the MCU could not read the ADS1299 in time (SPI DMA busy or EMG ring full). Healthy is ~0/s at 1 kHz, ~0.2% at 2 kHz, a handful per take at the default 4 kHz (8 in 274 s with both IMUs and cameras). If sustained, the panel FAULT LED latches and a `watchdog: sustained missed_drdy` `LOG` is emitted. Lower the EMG rate (`sample_rate_sps: 2000` doubles the read margin) or check the SPI bus.
- **`emg_drops` / `imu_*_drops` climbing** → the comms core is not draining the SPSC rings fast enough. Usually a downstream (USB/host) stall backing up.
- **`emg_drops` climbing with clean IMU** → the **host is not reading the USB vendor bulk pipe fast enough**, so the MCU's vendor TX FIFO fills. By design EMG backs off first (a `HOST_LINK_CRITICAL_RESERVE` protects the ~390 Hz IMU/trigger frames): EMG records wait in their ring until the FIFO has room and are lost only when that ring overflows, so gappy EMG with clean IMU is the expected shape of host-side jitter. `tx_drops` (`_emg` / `_imu` / `_other`) counts frames lost at the USB boundary itself (no host mounted, or a log/ack/stats frame with no FIFO room). This is the known limiter at high EMG rates — see [`h05_sensor_pipeline_architecture.md`](../hardware/h05_sensor_pipeline_architecture.md) §6–§7.

Mitigations: lower `mcu.emg.sample_rate_sps` (2000 halves the EMG volume of the 4000 default); close other USB-heavy processes on the Pi. The native capture backend (required for real recording) keeps several async USB transfers in flight and parses off the GIL. `JTSMM_NATIVE=0` only works with `--mock`; a real recording then errors.

### IMU drop rate flaps just above the 0.5% budget at 400 Hz

The BNO085 game-rotation-vector native rate is ~398 Hz, so a declared 400 Hz under-delivers by ~0.5% and flaps against the strict budget. The template declares **390 Hz** so the sensor over-delivers; if you hand-edit the config back to 400 you will see this flapping. See [`s08` `mcu.imu_*`](../software/s08_runtime_configuration.md).

### `integrity --strict` fails on optical lock / SE(3)

`jtsmm integrity check` also validates `/optical` when `meta.optical_tracking` is set or pose rows exist ([`s02` §8.5](../software/s02_data_pipeline.md)). The report table shows `opt.HP` / `opt.MP` / `opt.CT` lock %. For a plots-only pass with the same optical budgets:

```bash
python -m jtsmm.tools.eval_optical_session SESSION.h5 --no-show --strict
```

- **Both HP and MP stereo lock < 5% (error) or one body < 30% (warn):** lighting, glare, occlusion, stale calibration bundle, or lores too small for the tag size (`optical eval-quality`). Confirm cameras share `t_master_us` (stereo pairing).
- **Optical rows cover < 20% of stereo camera timestamps:** ArUco worker falling behind (`queue_dropped` / `pair_dropped` in `sync.attrs.report_json` → `optical`). Close other CPU load; keep `--optical-track` on lores only.
- **Phase lock below floor** (`mand_*` needs MP+CT ≥50%; `max_*` HP+CT ≥50%; motion HP+MP ≥40%): the body was not in view during that SPACE interval — re-run the phase, not the whole take if EMG is fine.
- **Finite poses not rigid SE(3) / jumps ≥250 mm / HP–MP &gt;800 mm:** bad PnP or swapped mounts; check `tracker.pnp.max_reprojection_rms_px` and marker IDs in `optical.yaml`.
- **`optical quality group missing`:** older `session.h5` without `/optical/quality/` — warn only; re-record if you need fail reasons.
- **`optical tracking enabled but /optical/t_master_us is missing`:** `--optical-track` ran but the Python writer never got rows (bridge crash or calibration bundle missing).

Live gates (NaN pose, session still records) are separate from these post-session fail/warn budgets — see [`docs/optical/03`](../optical/03_pose_estimation_and_output.md) §3.3.

## 6. Storage / writer issues

### Recording stops with "disk full"

- **NVMe full**: `df -h` to confirm; manually delete or move old sessions to NAS.
- **HDF5 writer crashed without flushing**: the writer flushes at most every 1 s, so at most the last ~1 s is lost; check the file with `jtsmm integrity check <session_dir>`.

### Session HDF5 file size much larger than expected

- **Streams duplicated**: check there are no two `capture_mcu` instances running.
- **Trigger rate too high**: trigger events accumulate; verify configured frame rate.

### Sidecar video file does not exist after a successful recording

- **Expected**: video sidecars are planned (not implemented); `jtsmm record` writes only `session.h5` (+ `integrity.json`, `reproduce.*`) under `~/jtsmm_data` (`host.paths.data_dir`).

## 7. Pi 5 / OS issues

### Pi 5 throttling (clock frequency reduced)

- **Active cooler not seated correctly**: re-seat; check that the fan spins (`pinctrl get FAN_PWM`). Tune `dtparam=fan_temp*` — see [`s01` §5.3](../software/s01_pi5_software_stack.md#53-thermal-and-fan).
- **Insufficient PSU**: use the official **27 W** USB-C PSU (5 V / 5 A). Without PD negotiation, USB is limited to 600 mA — see [`s01` §5.1](../software/s01_pi5_software_stack.md#51-psu-and-usb-current-budget) for `PSU_MAX_CURRENT=5000` and `usb_max_current_enable=1`.
- **Excessive ambient temperature**: improve room airflow.
- **Sustained CPU load from capture**: check `htop`. Try `performance` governor only after cooling is confirmed — [`s01` §5.2](../software/s01_pi5_software_stack.md#52-cpu-frequency-governor).

Check with: `vcgencmd get_throttled` — `0x0` is healthy.

### Pi 5 USB-CDC disconnects randomly

- **USB cable degraded**: replace.
- **HAT power glitch**: check the 5 V rail's bypass capacitors are populated; check no shorts.
- **Pi 5 USB controller saturating**: reduce camera bandwidth (resolution or fps); use the dedicated USB3 controller for the HAT and not a shared hub.

### `journalctl` shows segfaults in libcamera

- **libcamera version mismatch**: pin to the version listed in `pyproject.toml`; check `apt list --installed | grep libcamera`.
- **Recently updated Pi OS**: roll back `libcamera-dev` to the known-good version, or update `picamera2` to match.

## 8. When all else fails

1. Stop the recording cleanly (Ctrl-C). Do not yank cables.
2. Run `jtsmm doctor` and `jtsmm info`. Save the output.
3. Run `journalctl --since "1 hour ago" --no-pager > debug.log`.
4. Take a photo of the bench setup, especially the cables.
5. File an issue in the project tracker with the above bundle.

## 9. Known bugs / gotchas (kept up to date with the codebase)

These are project-specific quirks; they will be removed once fixed.

- `picamera2` with `external_sync=true` requires the `LIBCAMERA_LOG_LEVELS=ERROR` environment variable, otherwise log spam interferes with `capture_cam`'s stdout JSON.
- The first 100 ms of EMG data after `ads_start` may show settling transients; we discard the first 1 s of every session in offline analysis.
- The Pi 5's USB3 port shares bandwidth with the PCIe lanes; if the NVMe HAT is saturating PCIe, the USB3 camera (if used) may drop frames. Mitigation: prefer two CSI GS cameras instead of mixing CSI + USB3.
- BNO085 occasionally enters "calibration" mode autonomously (re-tunes its biases) which can introduce a brief discontinuity in the quaternion; we mark these via the `SystemCalStatus` field but cannot prevent them.

## 10. Where to ask for help

- The project's GitHub issue tracker.
- For OpenBCI / ADS1299 questions: <https://openbci.com/forum/>.
- For libcamera / picamera2 questions: <https://github.com/raspberrypi/picamera2/issues>.
