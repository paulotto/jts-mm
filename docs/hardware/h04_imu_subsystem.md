<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# h04 — IMU Subsystem

> **Building the schematic in KiCad?** See [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md) and [`h04_imu_kicad_implementation.md`](h04_imu_kicad_implementation.md) (7-pin JST-SH on `imu_connectors.kicad_sch`; BNO085 is off-board). **Harness build:** [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md).

The IMU subsystem provides a second, optical-independent estimate of mandibular and cranial orientation. It serves three purposes:

1. **Cross-validation** of the optical pose. An IMU quaternion that drifts away from the optical solution is a strong indicator of marker occlusion or calibration drift.
2. **Bridging optical drop-outs**. A few hundred milliseconds of optical occlusion (a hand passing through the FOV, a marker briefly hidden by the lip) become a non-event because the IMU pose continues uninterrupted.
3. **Higher-rate angular velocity** than the camera can deliver. The gyro signal at 200–400 Hz is the cleanest source of angular velocity for feature extraction in the classifier.

## 1. Why two IMUs

We instrument **both** the mandibular tracking array (MTA) and the cranial reference array (CRA). The reason is the same as for the optical system: only the *relative* mandible-to-cranium orientation matters for jaw kinematics. Subtracting the CRA quaternion from the MTA quaternion removes head motion from the jaw signal, which is exactly the equivalent of the optical pipeline's "subtract CRA from MTA" step described in §II.B of the IEEE SENSORS 2025 paper.

A single-IMU configuration would only work if the head were perfectly still, which it never is.

The published jaw tracking system also includes a **digitizing pointer (DP)** for anatomical landmark registration ([IEEE SENSORS 2025](https://github.com/paulotto/jaw_tracking_system); paper component (3)). That pointer is **optically** tracked during digitization; it is **not** part of the continuous jaw-motion recording stream. An **optional third IMU on the DP** is documented in §3.4 for labs that want pointer attitude on the MCU time base — it is **not required** for the default capture path (MTA + CRA).

## 2. IMU choice: BNO085

The **BNO085** (Bosch / CEVA) is a 9-DoF IMU with on-chip sensor fusion (BSX3.0). Its outputs include:

- **Game rotation vector** (accel + gyro fusion, no magnetometer; drift-free in roll/pitch, drift in yaw): perfect for our use because the magnetic environment near a metal-rich lab bench is unreliable.
- **Rotation vector** (full 9-DoF fusion including magnetometer): supported but not relied upon.
- **Linear acceleration** (gravity removed).
- **Gyro calibrated** at up to 400 Hz.
- **Accelerometer** at up to 500 Hz.

We use the **game rotation vector** as the primary mandibular/cranial orientation output. Production default (`mcu.imu_*.profile: kinematics_accel`) configures it at **390 Hz** with the **accelerometer @ 250 Hz** (gravity included; removed offline with the optical orientation, [`s05`](../software/s05_imu_processing.md) §3.1). `kinematics_default` takes **linear_acceleration @ 250 Hz** in the same slot; measured on hardware, that pair runs at **~350 Hz (game_rv) + ~220 Hz (linear_accel)**: the BNO085 hub prioritizes the fusion report and quantizes the two into interleaved slots. linear_accel is requested at **250, not 200**, because 200 Hz aligns badly with the fusion schedule and collapses to ~120 Hz, while 250 Hz lands at ~220 Hz. **Gyro is not in the default** (fusion + gyro + lin throttles all three to ~331/331/79 Hz); enable it via `raw_only` / `datasheet_motion` / `custom` — see §6. (Alone, i.e. `fused_only`, the fusion report reaches its native ~398 Hz — see [`s08`](../software/s08_runtime_configuration.md).)

### 2.1 Why not raw IMU + on-host fusion

We considered ICM-42688-P + Madgwick / Mahony / EKF fusion on the Pi 5. Advantages: fully transparent algorithm. Disadvantages:

- Adds ~1–2 weeks of firmware/software effort.
- The on-chip BSX3.0 fusion is empirically excellent (Adafruit's [test docs](https://learn.adafruit.com/adafruit-9-dof-orientation-imu-fusion-breakout-bno085) measure ~0.5° static accuracy).
- We can request both fused **and** raw outputs from the BNO085, so anyone who wants to redo the fusion has the raw data on disk.

The architecture is therefore: BNO085 *primary*, ICM-42688-P *alternative* documented in §5, raw + fused both stored on disk.

## 3. Electrical and mechanical mounting

### 3.1 MTA-mounted IMU

The MTA in the IEEE SENSORS 2025 paper is a printable mouthpiece attached to the lower teeth with dental adhesive. We add a **dorsal mounting platform** to the existing CAD: a flat 12 × 10 mm pad on the outside (cheek-side) of the MTA, just behind the marker carrier, to hold an IMU breakout board.

Design constraints:

- Total added mass: ≤2 g (light enough to not interfere with mandible dynamics; the original MTA + markers is ~3 g).
- Cable strain relief: the JST-SH connector is glued to the MTA so that cable tension is transferred to the MTA shell, not to the breakout's SMD pads.
- Cable run: ~30 cm from MTA to the cheek, then taped along the cheek and routed behind the ear to a strain-relief clip on the patient's collar; from there a 1 m shielded cable goes to the HAT.

A future iteration could integrate a bare BNO085 LGA chip into a flex PCB embedded in the MTA itself, but for v1 we keep the breakout-board approach for replaceability and serviceability.

### 3.2 CRA-mounted IMU

The CRA is a forehead band; mounting the IMU is mechanically simpler. We add an analogous flat 12 × 10 mm pad on the dorsal side of the CRA, near the back of the band where there is no marker. The cable runs along the band to the back of the head, joins the MTA cable in a single strain-relief clip on the collar, and continues to the HAT.

### 3.3 Cabling

Each IMU link uses a **7-conductor shielded cable** (24-AWG): **six signals** plus a **dedicated shield/drain** on **pin 7**. Signals: VCC (3.3 V), GND, **SDA**, **SCL**, **INT**, **nRST**. SDA/SCL and INT/nRST are twisted pairs inside the bundle; the foil/braid is **not** pin 2 — pin 2 is **DGND** (sensor return), while pin 7 is **`IMU_SHLD`** → **AGND** at the HAT only ([`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2.3).

**Connectors:** **JST-SH 7-pin** (`SM07B-SRSS-TB`) at the HAT; **JST-GH 7-pin** (`GHR-07V-S`) on the breakout pigtail. The older **6-pin** SH/GH pair cannot crimp the shield into the netlist — see [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md).

> **Why not a circular shielded connector here (like EMG)?** EMG uses pin headers for bring-up (**Option 0**) or shielded circular connectors for production ([`h03_emg_frontend.md`](h03_emg_frontend.md) §10.2). The IMU link stays on **JST** — digital I2C, lightweight body end (MTA **≤2 g**).

Cable length: **1.5 m** typical on the HAT jumper. Total path is longer on the subject (§3.1–§3.2); see **§3.3.1** for I2C capacitance vs length.

**7-pin pinout (HAT end — identical on `J_IMU_MTA`, `J_IMU_CRA`, `J_IMU_DP`):**

| Pin | Net | Direction (HAT view) | Pair / notes |
|-----|-----|----------------------|--------------|
| 1 | VCC (+3V3D) | out | with GND |
| 2 | GND (DGND) | — | signal return; **not** the cable shield |
| 3 | SDA | bidir | with SCL |
| 4 | SCL | out | with SDA |
| 5 | INT | in | with nRST |
| 6 | nRST | out | with INT |
| **7** | **`IMU_SHLD`** | — | **→ AGND @ HAT**; breakout end **floating** (pin 7 not tied to breakout GND) |

Full assembly steps: [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md).

#### 3.3.1 Cable parasitics and maximum length (I2C)

**Scope.** Pull-ups are on the HAT (§4). Cable **capacitance** loads **SDA** and **SCL** and sets how fast edges rise; **resistance** causes small DC drops; **inductance** is usually secondary at **400 kHz**.

**Total bus capacitance (Fast-mode budget):**

```text
C_bus = C_HAT + C_cable + C_breakout + C_mux + C_ESD
```

Only **`C_cable`** scales with length:

```text
C_cable ≈ C' × L_total
```

`L_total` = entire I2C path on that bus (breakout pigtail + subject routing + collar segment + HAT jumper). Example MTA: **~0.3 m** on head + **~1.0 m** to collar + **~1.5 m** to HAT ≈ **2.8 m** (round up in planning).

**Typical `C'` (SDA or SCL to GND/shield):** **30–80 pF/m** for **24 AWG** twisted pair inside a shielded 7-conductor cable (measure your SKU @ 1 kHz).

**I2C Fast-mode rise-time limit** (NXP / UM10204 style, 400 kHz):

```text
t_rise ≤ 300 ns
t_rise ≈ 0.8473 × R_pullup × C_bus          (single time-constant approximation)
```

Solve for maximum capacitance at given pull-up `R_pu`:

```text
C_bus,max ≈ t_rise,max / (0.8473 × R_pu)
```

With **`R_pu = 2.2 kΩ`** (this HAT) and **`t_rise,max = 300 ns`**:

```text
C_bus,max ≈ 300×10⁻⁹ / (0.8473 × 2200) ≈ 161 pF
```

**Worked example** (`C' = 50 pF/m`, `L_total = 2.8 m`, `C_HAT+breakout+mux ≈ 40 pF`):

```text
C_cable = 50×10⁻¹² × 2.8 ≈ 140 pF
C_bus   ≈ 40 + 140 ≈ 180 pF   → slightly over 161 pF
t_rise  ≈ 0.8473 × 2200 × 180×10⁻¹² ≈ 335 ns
```

At **335 ns** you are still often **marginally OK** on the bench (BNO085 tolerates moderate rounding), but you are **near the edge**. Mitigations (in order):

1. Shorten **`L_total`** (target **≤ 2.5 m** per bus for planning).
2. Stronger pull-ups: **`R_pu = 1.5 kΩ`** → `C_bus,max ≈ 236 pF` (check **3 mA** sink budget on RP2350A and breakout).
3. Lower I2C clock to **100 kHz** in firmware (last resort; costs latency).

**Resistance (usually not the limiter):**

```text
R_wire = R' × L_total        (per conductor, 24 AWG: R' ≈ 0.04–0.05 Ω/m)
ΔV_SDA ≈ I_pullup × R_wire   (I_pullup ≈ 3.3 V / R_pu ≈ 1.5 mA with 2.2 kΩ)
```

For `L_total = 3 m`, `ΔV ≈ 1.5 mA × 0.15 Ω ≈ 0.2 mV` — negligible.

**Inductance:**

```text
L_wire = L' × L_total       (L' ~ 0.3–0.6 µH/m)
|Z_L| @ 400 kHz ≈ 2π f L_wire  → hundreds of mΩ
```

Still small compared with `R_pu`; ignore unless you run **> 1 MHz** on long harnesses.

**Recommended cable (v1):**

| Item | Specification |
| ---- | ------------- |
| Conductors | **7** — VCC, GND, SDA, SCL, INT, nRST, **SHLD** (pin 7) |
| Gauge | **24 AWG** minimum |
| Layout | **Twisted pairs**: (SDA,SCL), (INT,nRST), (VCC,GND); braid/drain → **pin 7** |
| Shield | **Pin 7 → AGND** at HAT only; far end floating ([`h07`](h07_power_grounding_safety.md) §2.3) |
| Connectors | **JST-SH 7-pin** `SM07B` (HAT), **JST-GH 7-pin** `GHR-07` (breakout) |
| HAT jumper length | **1.5 m** nominal (**I-02** in catalog) |
| Planning `L_total` | **≤ 2.5 m** per bus without stronger pull-ups; **≤ 3.5 m** only after ACK/scope check |

**EMG harness** parasitics are different (high-Z biopotential); see [`h03_emg_frontend.md`](h03_emg_frontend.md) §10.1.

### 3.4 Optional digitizing-pointer (DP) IMU

**Role (aligned with the original JTS).** During setup, the clinician touches six dental landmarks with the DP tip while OMoCap (or JTS-MM cameras) records marker positions. The IMU does **not** measure tip position — only the **orientation** of the pointer body, which is rigidly related to the marker frame by CAD. Uses include:

- Logging pointer attitude at **200 Hz** on **`t_master_us`** during digitization (same clock as MTA/CRA).
- Comparing fused orientation to the optical rigid-body fit of DP markers / AprilTags.
- Detecting operator motion or magnetic disturbance before accepting landmark sets.

**Default build:** omit — two IMUs on MTA and CRA are sufficient for jaw kinematics and intent fusion.

**Hardware when fitted:**

| Item | Detail |
|------|--------|
| Sensor | Third **BNO085** breakout (same as I-01 in [`04_component_catalog.md`](../04_component_catalog.md)) |
| Mount | 12 × 10 mm pad on DP handle, ~20–40 mm from marker cluster; cable strain relief on DP shell |
| HAT connector | Third **JST-SH 7-pin** `J_IMU_DP` (same pinout as §3.3) |
| Bus | **I2C1** through **PCA9546A** mux (**H-IC-13**, DNP): channel 0 = CRA, channel 1 = DP; both may use **0x4A** |
| INT | **GPIO12** (DP); CRA remains **GPIO17** |
| Reset | Shared **GPIO22** `IMU_NRST` with MTA/CRA |

**Firmware / data:** DP support is planned (not implemented in firmware or `jtsmm`); record types `0x17`–`0x19` are reserved — see [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1.2 and §3.2. Enable **game rotation vector @ 50–200 Hz** only during digitization to limit USB load; full 400 Hz gyro is unnecessary on the DP.

**Schematic capture:** [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md) Step 8 (optional).

**I2C mux performance:** A **PCA9546A** on I2C1 does **not** slow the bus electrically (400 kHz Fast-mode capable, ~6 Ω Ron). The practical limit is **logical**: only one downstream channel is active at a time, so CRA and DP cannot each sustain **200 Hz** SH2 traffic concurrently on the same RP2350A I2C1. Use the DP IMU only during **landmark digitization** (pause or slow CRA), at **50–100 Hz** on the pointer. Channel-select writes add ~20–40 µs when switching; that is negligible if you do not alternate channels every sample. **Default builds omit the mux** — no overhead. Details: build guide §*I2C mux — does it hurt performance?*

## 4. Bus choice: I2C (v1)

The BNO085 supports I2C, SPI, and UART (SH2 protocol). For **v1 we use I2C**, because it keeps the patient-side cable to **six signal conductors** plus **shield on pin 7** (a 4-pin JST cannot carry SPI + INT + WAKE; a full SPI harness needs eight signals plus shield → **9-pin** variant). I2C trade-offs we accept:

- 400 kHz Fast-mode is plenty for our per-IMU report budget (~26 kB/s, see §6) when each IMU is on its **own** I2C bus.
- INT-driven reads keep latency jitter low (<50 µs budget); we do **not** poll.
- The RP2350A has two I2C peripherals, so we give **each IMU its own bus** (no shared-bus bandwidth contention, no address-collision juggling). EMG stays on SPI0 (high-speed, time-critical DRDY).

Wiring (each IMU on a dedicated I2C bus):

| Signal | RP2350A pin | MTA IMU | CRA IMU |
|--------|-----------|---------|---------|
| SDA | GPIO20 (I2C0) / GPIO18 (I2C1) | I2C0_SDA | I2C1_SDA |
| SCL | GPIO21 (I2C0) / GPIO19 (I2C1) | I2C0_SCL | I2C1_SCL |
| INT | GPIO16 / GPIO17 | INT_MTA | INT_CRA |
| nRST | GPIO22 (shared) | active low | active low |
| VCC / GND | +3V3D / DGND | — | — |

- **Pull-ups**: 2.2 kΩ to +3V3D on each SDA/SCL pair, placed on the HAT (not the breakout) so cable capacitance is the only added load.
- **I2C address**: with separate buses both BNO085s use the default address (0x4A); no `DI`/`ADR` strapping needed.
- **nRST** is a hard reset (held low to reset). The SHTP-level `WAKE`/PS pin is tied on the breakout for I2C mode, so no per-device WAKE GPIO is needed (this is the I2C-mode simplification vs the SPI wiring of earlier revisions).

The authoritative full-HAT RP2350A pin map is in [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1.

**Third IMU (optional):** The RP2350A has only two I2C peripherals. A populated DP IMU shares **I2C1** with CRA via **PCA9546** (§3.4); firmware must select the mux channel before each CRA or DP access. Default v1 PCBs wire CRA **direct** to I2C1 with mux **DNP**.

> **SPI alternative.** If a future revision needs lower latency or higher report rates, the BNO085 can run SPI (SCLK/MOSI/MISO/CS/INT/WAKE/nRST = 8 signals + shield → **9-pin** JST-SH/GH). The on-disk data format is unchanged. Document the harness change in this section if you take that path.

## 5. Alternative: ICM-42688-P

For replicators who insist on a fully open processing chain, **ICM-42688-P** is the closest functional alternative:

- 6-DoF only (no magnetometer in -P variant; ICM-20948 has one if needed).
- Raw 16-bit accel + 16-bit gyro at up to 32 kHz (we'd run at 1 kHz).
- No on-chip fusion: a Madgwick filter on the RP2350A produces a usable quaternion at ~1 kHz with ~3 ms latency.
- ~3 EUR / chip in single quantities (vs. ~22 EUR for a BNO085 breakout).

If chosen, the cable/connector stays identical (the ICM-42688-P also speaks I2C; just a different chip and a firmware fusion module). The on-disk data format is unchanged because we already store raw accel + gyro + quaternion in both cases.

Other alternatives we considered and rejected:

- **MPU-9250**: end-of-life, magnetometer cross-axis issues.
- **BMI270 + BMM150 + custom fusion**: more PCB work, no clear advantage over ICM-42688-P.
- **BNO055** (predecessor of BNO085): older, lower update rate (~100 Hz), more drift; superseded by BNO085.

## 6. Sample-rate strategy and runtime profiles

Production firmware and host software resolve IMU settings from **`mcu.imu_mta`** / **`mcu.imu_cra`** in `~/jtsmm.yaml` ([`s08_runtime_configuration.md`](../software/s08_runtime_configuration.md)). Implementation: `firmware/jtsmm/src/domain/imu_config.c`, `software/jtsmm/jtsmm/config/profiles.py`.

### 6.1 Recording profiles

| Profile | Enables (resolved `reports` map) | Typical use |
|---------|----------------------------------|-------------|
| `fused_only` | Primary fusion report only | Minimal USB bandwidth |
| `raw_only` | `accelerometer`, `gyroscope` (+ optional `magnetometer`) | Host-side re-fusion research |
| **`kinematics_accel`** | Primary fusion + `accelerometer` @ 250 | **Default capture** (orientation + gravity-included accel; gravity removed with the optical orientation) |
| `kinematics_default` | Primary fusion + `linear_acceleration` @ 250 | Orientation + the BNO085's gravity-free accel (measured ~350 Hz game_rv + ~220 Hz linear_accel) |
| `all` | Every catalog report at default rates | Lab max-data (bus budget check required) |
| `custom` | Explicit `reports:` map | Full operator control |

`fusion.primary` selects the orientation stream when not overridden by `custom`:

| `fusion.primary` | SH-2 report | Notes |
|------------------|-------------|-------|
| **`game_rotation_vector`** | `0x08` | **Default** — robust near metal benches |
| `rotation_vector` | `0x05` | Full 9-DoF; needs reliable mag |
| `geomagnetic_rotation_vector` | `0x09` | Mag-aided yaw |
| `arvr_stabilized_game_rotation_vector` | `0x29` | Visually stabilized 6-DoF |
| `arvr_stabilized_rotation_vector` | `0x28` | Visually stabilized 9-DoF |
| `gyro_integrated_rotation_vector` | `0x2A` | Highest-rate orientation candidate (~1 kHz bench target) |

### 6.2 Sync modes (`mcu.imu_*.sync.mode`)

| Mode | Hardware path | When to use |
|------|---------------|-------------|
| **`int_driven`** *(default)* | BNO085 `/INT` → GPIO16/17 → **I2C DMA** on each report | Non-integer EMG/IMU rate ratios (e.g. 1000 Hz EMG + 400 Hz IMU) |
| `emg_drdy_divisor` | Every *N*th EMG `/DRDY` kicks I2C DMA | Only when `emg_fs % imu_rate == 0` (host rejects otherwise) |

PIO bit-bang I2C is **out of scope** for v1 — hardware I2C0/I2C1 + DMA only ([`h05`](h05_microcontroller_firmware.md) §1.1.3).

**Timestamp source (`int_driven`).** Each IMU record's `t_master_us` is its sample time:

1. **t_int** — a falling-edge IRQ on the INT pin, handled on the sensor core (core 1, own raw GPIO handler; the core-0 `gpio_irq.c` dispatcher is not used), latches `time_us_64()`. The loop takes that edge when it kicks the read and clears it first, so an INT re-assertion during the read stamps the *next* packet. Edges from the gap between the 4-byte header read and the full re-read, and edges whose INT is high again, are dropped. No edge, one older than max(3 report periods, 20 ms), or one in the future → **poll time** (the old behaviour), counted.
2. **Per report** — `t_int − base_delta + rebase + delay` (×100 µs) from the SH-2 Base Timestamp Reference (0xFB), Timestamp Rebase (0xFA) and the report's 14-bit delay (status[7:2]:byte 3), as in CEVA's `sh2.c`. Never later than t_int (clamped); a result > 100 ms before t_int keeps t_int (counted). Channel-5 gyro RV and reports before any 0xFB keep t_int.

`emg_drdy_divisor` keeps the DRDY-paced read time (no edge IRQ, no SH-2 correction). Revert without code edits: `-DJTSMM_IMU_TS_USE_INT_EDGE=0` and/or `-DJTSMM_IMU_TS_USE_SH2_DELAY=0`. The HAT has **no pull-up on INT**; the firmware enables the RP2350 internal one. Diagnostics: `imu_status` (`ts:` lines: t_int source, edge vs. poll-fallback counts, edge→kick latency, base delta) and `imu_int_watch [ms]` (IRQ path without an IMU: tap INT to GND).

### 6.3 Resolved rates (`kinematics_accel`, default)

| Report | Rate | Why |
|--------|------|-----|
| Game rotation vector (`fusion.primary`) | **390 Hz req → ~350 Hz** (measured with lin active) | Primary orientation; jaw-vs-cranium relative attitude |
| Accelerometer (calibrated, gravity included) | **250 Hz req** (not yet measured; ~220 Hz expected, as linear accel in the same slot) | Chewing / motion onset + translation; gravity removed with the optical orientation |
| Linear acceleration | *(off; `kinematics_default`: 250 Hz req → ~220 Hz)* | The BNO085's own gravity removal (inherits its tilt error) |
| Calibrated gyro | *(off)* | Enable via `raw_only` / `datasheet_motion` / `custom` |
| Magnetic field | *(off by default)* | Sanity check only when enabled |

**Why no gyro in the default?** MTA and CRA are on **separate** 400 kHz buses
(§4), but each BNO085 sensor hub prioritizes its fusion report and has a per-hub
scheduling ceiling (~740 reports/s raw; I2C clock-stretching + SHTP overhead).
Measured, `game_rv@390 + linear_accel@250` gives game_rv ~350 Hz + linear_accel
~220 Hz (the hub interleaves the two; requesting linear_accel@200 instead is worse
— it collapses to ~120 Hz because 5 ms aligns badly with the fusion schedule).
Adding `gyro@400` is worse still: the hub throttles all three to ~331/331/79 Hz. A
single report reaches full rate one at a time (game_rv ~398, gyro ~399, accel
~500).

### 6.4 Legacy fixed-rate table (bring-up reference)

Bring-up firmware used a fixed 200 Hz game-RV plan. Production defaults are **profile-driven** (§6.1–6.3); the table below is retained for comparison:

| Report | Bring-up rate | `kinematics_default` |
|--------|---------------|-----------------------------------|
| Game rotation vector | 200 Hz | **390 Hz** (~350 with lin active) |
| Linear acceleration | 200 Hz | **250 Hz** (~220 measured) |
| Calibrated gyro | 400 Hz | off (unless `raw_only` / `custom`) |
| Accelerometer raw | 200 Hz | off (unless `custom`) |
| Magnetic field | 50 Hz | off (unless `custom`) |

## 7. Calibration

A one-time calibration per IMU produces:

1. **Gyro bias** (mean static reading, ~0.01 rad/s); subtracted in firmware.
2. **Accel bias + scale** via 6-position static (cube faces).
3. **Magnetometer hard- and soft-iron** ellipsoid fit (optional, only if we use the full rotation vector instead of the game RV).
4. **Mounting transform** from IMU chip axes to MTA/CRA optical body axes. **Rotation now** (Park–Martin `AX=XB`, figure-eight of each array). **Translation (lever arm) later**, optional, for origin-frame accelerations only. Protocol and CLI: [`cv02`](../calibration_validation/cv02_imu_calibration.md) §6.

The full protocol lives in [`../calibration_validation/cv02_imu_calibration.md`](../calibration_validation/cv02_imu_calibration.md).

## 8. Software interface

**Production path:** `firmware/jtsmm` streams framed records over USB (`0x55 0xAA`); `software/jtsmm/jtsmm/capture/capture_mcu.py` parses them via `jtsmm.protocol.FrameParser` and `jtsmm.capture.session.run_session()` writes HDF5.

The MCU firmware emits **one self-describing tagged record per parsed BNO085
report** (so accel/gyro/mag/fusion are independent and combinable at their own
rates):

- `IMU_MTA_QUAT` — 17 B: `[report_id:u8][w,x,y,z:f32×4]`. `report_id` is the SH-2
  fusion id (`0x05` rotation vector, `0x08` game, `0x09` geomagnetic) so multiple
  fusion types coexist unambiguously.
- `IMU_MTA_RAW` — 7 B: `[sensor_id:u8][x,y,z:int16×3]`, one motion vector per
  record; `sensor_id` ∈ {`0x01` accelerometer, `0x02` gyroscope, `0x03`
  magnetometer}. Counts are SH-2 Q-point fixed point (accel Q8, gyro Q9, mag Q4),
  converted to SI on the host ([`jtsmm/imu/units.py`](../../software/jtsmm/jtsmm/imu/units.py)).
- `IMU_MTA_LIN_ACCEL` — 12 B: `[x,y,z:f32×3]` (gravity-removed, m/s²).
- Same for `IMU_CRA_*` and, when fitted, optional `IMU_DP_*` (§3.4).

The firmware catalog ([`bno085_reports.c`](../../firmware/jtsmm/src/domain/bno085_reports.c))
now covers all seven datasheet outputs, including **magnetometer (`0x03`)** and
**geomagnetic rotation vector (`0x09`)**. Host decoders branch on payload length
for legacy fallback (16 B untagged QUAT ⇒ configured fusion; 18 B RAW ⇒ merged
accel+gyro). The authoritative record-type table (codes, LEN) is in
[`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §3.2.

The host process `capture_mcu` (or inline reader in `run_session`) decodes these into NumPy arrays and forwards them to the HDF5 writer ([`s02_data_pipeline.md`](../software/s02_data_pipeline.md) §3).

For ROS 2 deployment (optional), each IMU stream maps cleanly to a `sensor_msgs/Imu` message; the mapping is documented in [`../software/s05_imu_processing.md`](../software/s05_imu_processing.md).

## 9. Failure modes and diagnostics

| Symptom | Likely cause | Diagnostic |
|---------|--------------|-----------|
| All zeros from IMU | Power not reaching the breakout, I2C not ACKing (pull-ups missing), nRST stuck low | Scope SDA/SCL for ACK + 3.3 V at breakout; check nRST is released high |
| Quaternion jumps | Magnetic disturbance near IMU (a metal table, a power supply) | Switch to game RV (already default), move setup |
| Quaternion frozen | SHTP transport failed, BNO085 in suspended state | Firmware sends `RESET` SHTP packet; if persistent, power-cycle |
| INT line never asserts | Report not subscribed; firmware bug | Use the firmware `imu_status` command to dump subscribed reports |
| Latency >5 ms between INT and I2C read | RP2350A IRQ priority too low | Bump IMU I2C ISR priority above background USB-CDC work |

## 10. Performance acceptance criteria

The IMU subsystem is accepted when:

- [ ] Both IMUs deliver game-RV at 200 Hz continuously for 10 minutes with no dropped samples.
- [ ] Quaternion noise (rotating IMU statically) is <0.2° RMS in pitch / roll, <1°/min drift in yaw (game RV).
- [ ] Relative quaternion between MTA-IMU and CRA-IMU during static head + jaw is within 1° of the relative quaternion derived from optical pose (after the mounting transform calibration).
- [ ] During controlled jaw motion (mouth opening 30 mm), the mandibular pitch angle measured by IMU and by optical agree within 2° RMS.

These tests are part of [`../calibration_validation/cv02_imu_calibration.md`](../calibration_validation/cv02_imu_calibration.md).
