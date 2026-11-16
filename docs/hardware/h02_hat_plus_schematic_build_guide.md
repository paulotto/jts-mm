<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h02 — HAT+ / Top-Level Schematic Build Guide (KiCad capture walkthrough)

Companion to [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §8 for **`hat_plus.kicad_sch`** (Pi 40-pin + HAT ID EEPROM) and **root bus wiring** on `jtsmm_hat.kicad_sch`.

> **Subsystem sheets:** use the per-sheet build guides listed in [`_schematic_build_guide_conventions.md`](_schematic_build_guide_conventions.md).

## How to read the diagrams

Same legend as [`_schematic_build_guide_conventions.md`](_schematic_build_guide_conventions.md).

---

## Part A — `hat_plus.kicad_sch`

### Step 0 — Parts

> **Footprint audit:** [`h09_footprint_master_table.md`](h09_footprint_master_table.md)

| ID | Item | Notes |
|------|------|--------|
| H-CON-07 | 2×20 **Samtec HLE** socket **HLE-120-02-L-DV-BE-A** (SMD bottom-entry; LCSC **C6207203**; hand-install) | HAT+ pass-through for M.2 HAT+ |
| — | **AT24CS64** (HAT ID EEPROM) | HAT ID / vendor EEPROM on I2C ID pins; MPN **AT24CS64-SSHM-T** (SOIC **U1**) or **AT24CS64-MAHM-T** (DFN **U2**, ALT) |
| — | **R1–R4** 3.9 kΩ (ID pull-ups) | **RC0603FR-073K9L** when populated per HAT+ template |
| — | 2.2 kΩ on `ID_SDA` / `ID_SCL` if not on Pi side | Per HAT+ spec |

Project template may already include `Connector:Conn_02x20_Odd_Even` and EEPROM symbols.

### Step 1 — Pi 40-pin header

- Place **2×20 stacking header** per HAT+ mechanical spec ([`h02`](h02_custom_hat_shield.md) §2).
- Route **pin 2 / pin 4** (5 V) to hierarchical **`HAT_5V0`** (output toward root).
- Route **GND** pins to hierarchical **`HAT_GND`**.
- **Do not** tap 3.3 V from the header for the analog front-end — system 3.3 V is **`+3V3D`** from the on-board LDO ([`h07`](h07_power_grounding_safety.md)).

### Step 2 — HAT ID EEPROM (if populated)

```text
GPIO0 / GPIO1 (ID_SDA / ID_SCL) ──► AT24CS64
                                    VCC ── +3V3 (from Pi 3V3 or local)
                                    GND ── GND
```

Verify against the Raspberry Pi **HAT+ ID EEPROM** application note and your template schematic. WP / address pins per datasheet.

**Programming U1 after assembly:** [`h02_hat_plus_eeprom_flashing.md`](h02_hat_plus_eeprom_flashing.md) — `eepmake` settings in [`hardware/jtsmm_hat/eeprom/`](../../hardware/jtsmm_hat/eeprom/) (`jtsmm_hat_eeprom_vert.txt` / `jtsmm_hat_eeprom_hori.txt`: PCB with vertical / horizontal EMG pins, each with its prebuilt `.eep`).

### Step 3 — hat_plus hierarchical labels

Export at minimum:

| Label | Role |
|-------|------|
| `HAT_5V0` | 5 V to power sheet |
| `HAT_GND` | Return to power sheet |
| `HAT_3V3` | Pi 3.3 V (logic reference only if needed) |
| `GPIO0_ID_SDA0` / `GPIO1_ID_SCL0` | EEPROM (if used) |

Other `GPIOxx` labels on the template are for future HAT features — leave unconnected on root unless used.

---

## Part B — Root sheet `jtsmm_hat.kicad_sch`

### Rule

**No ICs on the root page** — only **sheet symbols** and **wires** (or global labels) that connect hierarchical pins ([`h02`](h02_custom_hat_shield.md) §8).

### Suggested sheet order on root

| Page | Sheet | File |
|------|-------|------|
| 2 | hat_plus | `sheets/hat_plus.kicad_sch` |
| 3 | power | `sheets/power.kicad_sch` |
| 4 | emg | `sheets/emg.kicad_sch` |
| 5 | mcu | `sheets/mcu.kicad_sch` |
| 6 | trigger_io | `sheets/trigger_io.kicad_sch` |
| 7 | imu_connectors | `sheets/imu_connectors.kicad_sch` |

### Root integration — wire these hierarchical nets

Use **wires** on the root sheet so each row is one electrical net (all pins with the same name on different sheets connect only when wired on the parent).

#### Power (from `hat_plus` → `power` → consumers)

| Net | Connect |
|-----|---------|
| `HAT_5V0` | `hat_plus` → `power` `+5V_IN` |
| `HAT_GND` | `hat_plus` → `power` `GND_IN` |
| `+5VA` | `power` out → `emg` in |
| `+3V3D` | `power` out → `mcu`, `emg`, `trigger_io`, `imu_connectors` in |
| `AGND` | `power` out → `emg` in (and `imu_connectors` for shield) |
| `DGND` | `power` out → `mcu`, `emg`, `trigger_io`, `imu_connectors` in |

```text
hat_plus: HAT_5V0 ──────────────► power: +5V_IN
hat_plus: HAT_GND ──────────────► power: GND_IN

power: +5VA ────────────────────► emg: +5VA
power: +3V3D ──┬───────────────► mcu: +3V3D
               ├───────────────► emg: +3V3D
               ├───────────────► trigger_io: +3V3D
               └───────────────► imu_connectors: +3V3D

power: AGND ──┬────────────────► emg: AGND
              └────────────────► imu_connectors: AGND

power: DGND ──┬────────────────► mcu: DGND
              ├────────────────► emg: DGND
              ├────────────────► trigger_io: DGND
              └────────────────► imu_connectors: DGND
```

#### SPI / ADS1299 control (`mcu` ↔ `emg`)

| Net | `mcu` | `emg` |
|-----|-------|-------|
| `ADS_SPI_CLK` | out | in |
| `ADS_SPI_MOSI` | out | in |
| `ADS_SPI_MISO` | in | out |
| `ADS_SPI_CS` | out | in |
| `ADS_DRDY` | in | out |
| `ADS_RESET` | out | in |
| `ADS_PWDN` | out | in |
| `ADS_START` | out | in |

#### Trigger (`mcu` ↔ `trigger_io`)

| Net | `mcu` | `trigger_io` |
|-----|-------|----------------|
| `TRIG_PIO` | out | in |
| `SYNC_LED_PIO` | out | in |

#### IMU (`mcu` ↔ `imu_connectors`)

| Net | `mcu` | `imu_connectors` | Notes |
|-----|-------|------------------|-------|
| `IMU_MTA_SDA` | ↔ | ↔ |
| `IMU_MTA_SCL` | ↔ | ↔ |
| `IMU_MTA_INT` | ↔ | ↔ |
| `IMU_CRA_SDA` | ↔ | ↔ |
| `IMU_CRA_SCL` | ↔ | ↔ |
| `IMU_CRA_INT` | ↔ | ↔ |
| `IMU_NRST` | out | in | MCU GPIO22 drives shared IMU reset |
| `IMU_DP_INT` | in | out | Optional DP kit |
| `I2C_MUX_RST` | out | in | Optional GPIO23 → PCA9546 RESET; or tie RESET to +3V3D on `imu` only |
| `LED_IMU_MTA_ACT` | out | in | **D_MTA** on `imu_connectors` |
| `LED_IMU_CRA_ACT` | out | in | **D_CRA** on `imu_connectors` |
| `LED_IMU_DP_ACT` | out | in | **D_DP** on `imu_connectors` (Topology B, DNP default) |

Direction on child sheets is arbitrary for bidirectional I2C/GPIO; root wires define connectivity. **DP I2C does not add MCU pins** — it shares `IMU_CRA_SDA` / `IMU_CRA_SCL` (I2C1) through `U_MUX`.

**Panel LEDs / buttons** are local to `mcu`, `trigger_io`, and `power` sheets (no extra root hierarchicals except the three IMU LED nets above).

### ASCII bus diagram (root)

```text
                    ┌─────────────┐
   Pi ─────────────►│  hat_plus   │ HAT_5V0, HAT_GND
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   power     │ +5VA, +3V3D, AGND, DGND
                    └──┬───┬───┬──┘
           +5VA/AGND   │   │   └────────── +3V3D / DGND fan-out
                  ┌────▼───┐    ┌────────┐ ┌────────────┐ ┌────────────┐
                  │  emg   │    │  mcu   │ │ trigger_io │ │ imu_conn.  │
                  └───┬────┘    └───┬────┘ └────────────┘ └────────────┘
                      │             │
                      │    ADS_SPI*, ADS_*
                      └─────────────┘
                            mcu ◄──► emg

                      mcu ◄── TRIG/SYNC ──► trigger_io
                      mcu ◄── IMU_* ──────► imu_connectors
```

---

## Pre-capture checklist (root + hat_plus)

| Check | Pass criteria |
|-------|----------------|
| Stacking header | Clears M.2 HAT+ below ([`h02`](h02_custom_hat_shield.md) §11.5) |
| 5 V path | `HAT_5V0` → `+5V_IN` → regulators |
| No ICs on root | Only sheets + wires |
| All subsystem guides | emg, power, mcu, imu, trigger captured per guides |
| ERC on root | No dangling hierarchical pins |

---

## Suggested project capture order

1. `hat_plus` (5 V in)  
2. `power` ([`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md))  
3. `mcu` + `imu_connectors` ([`h05`](h05_mcu_schematic_build_guide.md), [`h04`](h04_imu_schematic_build_guide.md))  
4. `emg` ([`h03`](h03_emg_schematic_build_guide.md))  
5. `trigger_io` ([`h06`](h06_trigger_io_schematic_build_guide.md))  
6. **Root wires** (this document § Part B)  
7. ERC entire project
