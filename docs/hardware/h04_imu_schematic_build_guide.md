<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h04 — IMU Connectors Schematic Build Guide (KiCad capture walkthrough)

Step-by-step companion to [`h04_imu_subsystem.md`](h04_imu_subsystem.md) for the **on-HAT IMU interface only** (three **7-pin JST-SH** connectors, pull-ups, I2C mux). **BNO085** breakouts are off-board. Capture on `hardware/jtsmm_hat/sheets/imu_connectors.kicad_sch`. **Harness / shielding:** [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md). **KiCad checklist for the 7-pin connectors:** [`h04_imu_kicad_implementation.md`](h04_imu_kicad_implementation.md).

> **MCU GPIO mapping:** [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1 (canonical). **MCU sheet:** [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) Step 4b.

## How to read the diagrams

Same legend as [`_schematic_build_guide_conventions.md`](_schematic_build_guide_conventions.md).

---

## Topology of the sheet

`imu_connectors.kicad_sch` (revision 0.0.1) carries three IMU connectors and one I2C mux:

| IMU | Connector | I2C path to the RP2350A | INT |
|-----|-----------|-------------------------|-----|
| MTA | `J_IMU_MTA1` | **direct** on **I2C0** (GPIO20 SDA / GPIO21 SCL) | GPIO16 |
| CRA | `J_IMU_CRA1` | **I2C1** (GPIO18 SDA / GPIO19 SCL) through **`U_MUX1` channel 0** | GPIO17 |
| DP (digitizing pointer) | `J_IMU_DP1` | **I2C1** through **`U_MUX1` channel 1** | GPIO12 |

`U_MUX1` is fitted. There are no bypass resistors around it: the CRA connector reaches the MCU only through mux channel 0. Mux channels 2 and 3 are not connected.

The firmware uses two IMU slots (MTA and CRA). It selects mux channel 0 once (one write of `0x01` to address `0x70`); the selection latches, so CRA reads need no further mux write. The firmware does not address a DP IMU and does not drive GPIO12 or GPIO23 ([`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.4).

An alternative without the mux is described in [Step 8.1](#step-81--alternative-without-the-mux-not-captured); it is not captured on the sheet.

---

## I2C mux — effect on the bus

| Effect | Magnitude | Note |
|--------|-----------|------|
| **Bus speed** | PCA9546A supports **400 kHz** Fast-mode; Ron ≈ 6 Ω | One pair of 2.2 kΩ pull-ups on the upstream side (`R48`, `R49`) |
| **Extra capacitance** | ~3–5 pF per enabled channel | Small against the cable; see [`h04`](h04_imu_subsystem.md) §3.3.1 for `C_bus` against length |
| **Channel-select overhead** | One I2C write to the mux at **0x70** (~20–40 µs) when the channel changes | The selection latches; with CRA only, channel 0 is selected once |
| **CRA and DP at 200 Hz at the same time** | Not possible on one I2C1 — the two channels would have to be time-multiplexed | [`h04`](h04_imu_subsystem.md) §3.4 |
| **INT lines** | Not through the mux — separate GPIO17 (CRA) and GPIO12 (DP) | |

MTA stays on I2C0 and is not affected. The mux exists because the RP2350A has two I2C blocks, not three.

---

## Step 0 — Symbols and connectors

> **Footprints:** [`h09_footprint_master_table.md`](h09_footprint_master_table.md) · all three IMU jacks use the SM07B 7-pin horizontal footprint.

Parts on the sheet (all fitted):

| Ref | Item | Notes |
|-----|------|-------|
| `J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1` | **JST-SH 7-pin** 1.0 mm, **SM07B-SRSS-TB** | Same pinout on all three; pin 7 = `IMU_SHLD` → `AGND` (catalog `H-CON-02`) |
| `U_MUX1` | **PCA9546APW,118** (TSSOP-16) | Catalog `H-IC-13`; KiCad symbol and value [`Interface_Expansion:TCA9546APW`](#kicad-10--u_mux1-pca9546-symbol--footprint) (pin-compatible) |
| `R42`, `R46` | **2.2 kΩ** | I2C0 pull-ups (MTA): `R42` on SDA, `R46` on SCL, to `+3V3D` |
| `R48`, `R49` | **2.2 kΩ** | I2C1 pull-ups on the upstream side of the mux: `R48` on SDA, `R49` on SCL, to `+3V3D` |
| `R50` | **10 kΩ** | `IMU_NRST` → `+3V3D` |
| `R47` | **10 kΩ** | `U_MUX1` RESET (`I2C_MUX_RST`) → `+3V3D` |
| `LED_MTA1`, `LED_CRA1`, `LED_DP1` + `R43`, `R44`, `R45` | green 0603 LED (**LTST-C190KGKT**) + **470 Ω** | [Step 6b](#step-6b--imu-link-leds-led_mta1-led_cra1-led_dp1) |
| `C53`, `C55`, `C57` | **100 nF** (0402) | `+3V3D` → `DGND` at pin 1 of `J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1` |
| `C54`, `C56`, `C58` | **1 µF** (0603) | Bulk at each connector |
| `C59` | **100 nF** (0402) | `U_MUX1` VCC bypass |

### KiCad 10 — symbol + footprint (not in one library)

KiCad **does not** ship a schematic symbol named `SM07B-SRSS` (there is no `Connector_JST` symbol library in the default install). The **footprint** is in **`Connector_JST`**. The sheet uses the same pattern as the Debug Probe jack on [`mcu.kicad_sch`](../../hardware/jtsmm_hat/sheets/mcu.kicad_sch) (generic symbol, JST footprint):

| Field | Library ID |
|-------|------------|
| **Symbol** | `Connector:Conn_01x07_Pin` |
| **Footprint** | `Connector_JST:JST_SH_SM07B-SRSS-TB_1x07-1MP_P1.00mm_Horizontal` |
| **Value** | `Conn_01x07_Pin` |
| **MPN** (property) | JST **SM07B-SRSS-TB** (1.0 mm pitch, horizontal SMD) |

**Why 7 pins?** BNO085 on **I2C** needs **six signals** (VCC, GND, SDA, SCL, INT, nRST) plus a **dedicated shield contact**. A **6-pin** SH housing has no crimp position for the cable braid ([`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md)). Pin **7** = **`IMU_SHLD`** → **`AGND`** at the HAT; breakout pin 7 is **not** tied to sensor GND. Patient-side cable uses **JST-GH 7-pin** (`GHR-07V-S`); HAT end is **JST-SH** (different latch — do not mix series on one plug).

**Checklist for the 7-pin connectors:** [`h04_imu_kicad_implementation.md`](h04_imu_kicad_implementation.md).

### KiCad 10 — `U_MUX1` (PCA9546) symbol + footprint

There is **no** `Interface:PCA9546A` (or `PCA9546APW`) in the default KiCad symbol set. The **4-channel I2C mux with RESET** in TSSOP-16 (`PW`) is under **`Interface_Expansion`**, as the TI part **TCA9546A** — **pin-compatible** with NXP **PCA9546A** (same `PW` pinout as [Step 8.2](#step-82--u_mux1-pca9546apw-tssop-16)).

| Field | On the sheet |
|-------|--------------|
| **Symbol** | `Interface_Expansion:TCA9546APW` |
| **Footprint** | `Package_SO:TSSOP-16_4.4x5mm_P0.65mm` |
| **Value** | `TCA9546APW` |
| **MPN** (BOM property) | **PCA9546APW,118** (NXP; pin-compatible with TI **TCA9546APW**) |
| **LCSC** | **C149507** |

**Symbol chooser:** Add Symbol → library **`Interface_Expansion`** → search **`TCA9546`** or **`9546`** (not `Interface` — that library has USB/Ethernet, not muxes). Do **not** pick **PCA9544** (only 2 downstream channels) or **PCA9548** (8 channels, different pinout).

---

## Step 1 — 7-pin JST pinout (all jacks)

Identical on **`J_IMU_MTA1`**, **`J_IMU_CRA1`**, and **`J_IMU_DP1`** ([`h04`](h04_imu_subsystem.md) §3.3):

| Pin | Net name (local) | To breakout |
|-----|------------------|-------------|
| 1 | `+3V3D` | VCC |
| 2 | `DGND` | GND (signal return — **not** the cable shield) |
| 3 | `…_SDA` | SDA |
| 4 | `…_SCL` | SCL |
| 5 | `…_INT` | INT → RP2350A (per device) |
| 6 | `IMU_NRST` | nRST (shared bus) |
| **7** | **`IMU_SHLD`** | Cable braid/drain only — **NC** on breakout PCB |

**Shield wiring on HAT:** pin **7** → label **`IMU_SHLD`** → **`jtsmm:AGND`**. The shield is not connected to **`DGND`** or pin 2. Breakout end: pin 7 present in the plug for the drain wire; **no** copper to breakout GND ([`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md), [`h07`](h07_power_grounding_safety.md) §2.3).

---

## Step 2 — MTA (`J_IMU_MTA1`) — I2C0, direct

No mux. Nets tie straight to hierarchical labels.

```text
+3V3D ──┬── C53 (100nF) ── DGND
        ├── C54 (1µF) ──── DGND
        └──► J_IMU_MTA1 pin 1

DGND ──► J_IMU_MTA1 pin 2

J_IMU_MTA1 pin 3 (SDA) ──┬── R42 (2.2k) ── +3V3D
                         └──► hierarchical IMU_MTA_SDA  → mcu GPIO20 (I2C0_SDA)

J_IMU_MTA1 pin 4 (SCL) ──┬── R46 (2.2k) ── +3V3D
                         └──► hierarchical IMU_MTA_SCL  → mcu GPIO21 (I2C0_SCL)

J_IMU_MTA1 pin 5 (INT) ──► hierarchical IMU_MTA_INT  → mcu GPIO16

J_IMU_MTA1 pin 6 ──► IMU_NRST (shared, Step 6)
J_IMU_MTA1 pin 7 ──► IMU_SHLD ──► AGND
```

---

## Step 3 — CRA (`J_IMU_CRA1`) — I2C1 through mux channel 0

CRA **SDA/SCL** connect to **`U_MUX1`** pins **SD0** / **SC0** (channel 0). **INT and power are direct** (not through the mux):

```text
+3V3D ──┬── C55 (100nF) ── DGND
        ├── C56 (1µF) ──── DGND
        └──► J_IMU_CRA1 pin 1

DGND ──► J_IMU_CRA1 pin 2

J_IMU_CRA1 pin 3 ──► U_MUX1 pin 4 (SD0)
J_IMU_CRA1 pin 4 ──► U_MUX1 pin 5 (SC0)

J_IMU_CRA1 pin 5 (INT) ──► hierarchical IMU_CRA_INT  → mcu GPIO17

J_IMU_CRA1 pin 6 ──► IMU_NRST
J_IMU_CRA1 pin 7 ──► IMU_SHLD ──► AGND
```

**Upstream I2C1** (mux to MCU):

```text
IMU_CRA_SDA ──┬── R48 (2.2k) ── +3V3D
              ├──► hierarchical IMU_CRA_SDA  → mcu GPIO18 (I2C1_SDA)
              └──► U_MUX1 pin 15 (SDA)

IMU_CRA_SCL ──┬── R49 (2.2k) ── +3V3D
              ├──► hierarchical IMU_CRA_SCL  → mcu GPIO19 (I2C1_SCL)
              └──► U_MUX1 pin 14 (SCL)
```

The HAT has pull-ups only on the upstream side of the mux; there are none on the connector side of channels 0 and 1. The two nets between `U_MUX1` and each connector carry no label on the sheet.

> There are no hierarchical labels `IMU_DP_SDA` / `IMU_DP_SCL` — DP shares upstream I2C1; only **`IMU_CRA_SDA` / `IMU_CRA_SCL`** connect to the RP2350A.

---

## Step 4 — Hierarchical labels (`imu_connectors` → root → `mcu`)

| Label | RP2350A | Notes |
|-------|---------|-------|
| `IMU_MTA_SDA` | GPIO20 | I2C0 |
| `IMU_MTA_SCL` | GPIO21 | I2C0 |
| `IMU_MTA_INT` | GPIO16 | |
| `IMU_CRA_SDA` | GPIO18 | I2C1, upstream side of the mux |
| `IMU_CRA_SCL` | GPIO19 | I2C1, upstream side of the mux |
| `IMU_CRA_INT` | GPIO17 | |
| `IMU_DP_INT` | GPIO12 | INT of the DP connector |
| `IMU_NRST` | GPIO22 | Shared reset; MCU drives, the sheet fans out to the three JSTs |
| `I2C_MUX_RST` | GPIO23 | `U_MUX1` RESET; held high by `R47` |
| `LED_IMU_MTA_ACT` | GPIO26 | `LED_MTA1` |
| `LED_IMU_CRA_ACT` | GPIO27 | `LED_CRA1` |
| `LED_IMU_DP_ACT` | GPIO28 | `LED_DP1` |

Power on the sheet: global symbols `+3V3D`, `DGND`, and `AGND` (shield ties only).

Root: each label is wired `mcu` ↔ `imu_connectors` per [`h02_hat_plus_schematic_build_guide.md`](h02_hat_plus_schematic_build_guide.md) §Root nets.

---

## Step 5 — I2C addresses

| Device | Bus | Address | Strap on HAT |
|--------|-----|---------|----------------|
| MTA BNO085 | I2C0 | **0x4A** | None (dedicated bus) |
| CRA BNO085 | I2C1, mux channel 0 | **0x4A** | None |
| DP BNO085 | I2C1, mux channel 1 | **0x4A** | None — the mux isolates the channels |
| `U_MUX1` | I2C1 | **0x70** | A0, A1, A2 → `DGND` |

Mux control: [Step 8.4](#step-84--firmware-mux-control).

---

## Step 6b — IMU link LEDs (`LED_MTA1`, `LED_CRA1`, `LED_DP1`)

One green **0603** LED per IMU, all three fitted.

| Ref | Series R | Hierarchical (from `mcu`) | RP2350A | Footprint |
|-----|----------|---------------------------|--------|-----------|
| `LED_MTA1` | `R43` (470 Ω) | `LED_IMU_MTA_ACT` | GPIO26 | `LED_SMD:LED_0603_1608Metric` |
| `LED_CRA1` | `R44` (470 Ω) | `LED_IMU_CRA_ACT` | GPIO27 | same |
| `LED_DP1` | `R45` (470 Ω) | `LED_IMU_DP_ACT` | GPIO28 | same |

```text
LED_IMU_MTA_ACT ──[R43 470Ω]──► LED_MTA1 anode ── cathode ──► DGND   (MPN LTST-C190KGKT)
LED_IMU_CRA_ACT ──[R44 470Ω]──► LED_CRA1 anode ── cathode ──► DGND
LED_IMU_DP_ACT  ──[R45 470Ω]──► LED_DP1 anode ─── cathode ──► DGND
```

On **`mcu`** the **`LED_IMU_*_ACT`** labels are hierarchical **outputs** (MCU drives); on **`imu_connectors`** they are **inputs**.

---

## Step 6 — Shared `IMU_NRST`

```text
IMU_NRST ──┬── R50 (10k) ── +3V3D
           ├──► J_IMU_MTA1 pin 6
           ├──► J_IMU_CRA1 pin 6
           ├──► J_IMU_DP1 pin 6
           └──► hierarchical IMU_NRST → mcu GPIO22
```

All connected BNO085s reset together when firmware pulls **GPIO22** low.

---

## Step 7 — ERC

- [ ] Pull-ups: **two** on I2C0 (`R42`, `R46`), **two** on I2C1 **upstream** (`R48`, `R49`); none on the mux channels.
- [ ] `U_MUX1` VCC, GND, address pins, RESET wired; channels 2 and 3 marked not connected.
- [ ] `+3V3D` not shorted to `DGND`.
- [ ] INT nets reach `mcu` hierarchicals.
- [ ] Pin **7** `IMU_SHLD` → `AGND` on all jacks; not on `DGND`.

---

## Step 8 — Mux and DP path

### Step 8.1 — Alternative without the mux (not captured)

A build for MTA and CRA only could omit the mux and connect the CRA connector directly to I2C1 (two 0 Ω links from upstream SDA/SCL to `J_IMU_CRA1` pins 3 and 4). **Revision 0.0.1 has no footprints for such links**; on this board `U_MUX1` must be fitted for the CRA IMU to work. The alternative needs a change of the schematic and the PCB.

### Step 8.2 — `U_MUX1` (PCA9546APW, TSSOP-16)

KiCad: **`Interface_Expansion:TCA9546APW`** + **`Package_SO:TSSOP-16_4.4x5mm_P0.65mm`** ([capture notes](#kicad-10--u_mux1-pca9546-symbol--footprint)). Part: **PCA9546APW,118** (NXP); pin-compatible **TCA9546APW** (TI). Catalog: `H-IC-13`.

**Pin wiring (TSSOP `PW` pin numbers):**

| Pin | Name | Connected to |
|-----|------|--------------|
| 1 | A0 | **DGND** (address bit 0) |
| 2 | A1 | **DGND** |
| 3 | RESET | `I2C_MUX_RST`: `R47` (10 kΩ) → **+3V3D** and hierarchical label → mcu **GPIO23** |
| 4 | SD0 | `J_IMU_CRA1` pin 3 |
| 5 | SC0 | `J_IMU_CRA1` pin 4 |
| 6 | SD1 | `J_IMU_DP1` pin 3 |
| 7 | SC1 | `J_IMU_DP1` pin 4 |
| 8 | GND | **DGND** |
| 9 | SD2 | not connected |
| 10 | SC2 | not connected |
| 11 | SD3 | not connected |
| 12 | SC3 | not connected |
| 13 | A2 | **DGND** |
| 14 | SCL | `IMU_CRA_SCL` (upstream, `R49`) |
| 15 | SDA | `IMU_CRA_SDA` (upstream, `R48`) |
| 16 | VCC | **+3V3D** + `C59` (100 nF) → **DGND** |

**7-bit mux address:** A2=A1=A0=0 → **0x70**.

### Step 8.3 — `J_IMU_DP1` (digitizing pointer)

Same pinout as Step 1. **INT** is **not** muxed:

```text
+3V3D ──┬── C57 (100nF) ── DGND
        ├── C58 (1µF) ──── DGND
        └──► J_IMU_DP1 pin 1

DGND ──► J_IMU_DP1 pin 2

J_IMU_DP1 pin 3 ──► U_MUX1 pin 6 (SD1)
J_IMU_DP1 pin 4 ──► U_MUX1 pin 7 (SC1)

J_IMU_DP1 pin 5 (INT) ──► hierarchical IMU_DP_INT  → mcu GPIO12

J_IMU_DP1 pin 6 ──► IMU_NRST
J_IMU_DP1 pin 7 ──► IMU_SHLD ──► AGND
```

Off-board: a third **BNO085** breakout (catalog **I-01** +1) on the DP handle ([`h04`](h04_imu_subsystem.md) §3.4); shielded **7-conductor** cable (**H-CON-02** +1).

### Step 8.4 — Firmware mux control

The PCA9546A control register selects the channels: **`0x01`** = channel 0 (CRA), **`0x02`** = channel 1 (DP), **`0x00`** = none.

The firmware writes `0x01` to address `0x70` once when the CRA slot is prepared (`imu_bus.c`, `imu_mux_select_cra`); the selection latches, so the CRA reads on I2C1 need no further mux write. It does not select channel 1.

```text
i2c_write(0x70, [0x01])          // select SC0/SD0 (CRA) — done by the firmware
i2c_read(0x4A, ...)              // CRA BNO085

i2c_write(0x70, [0x02])          // select SC1/SD1 (DP) — not used by the firmware
```

**GPIO23** (`I2C_MUX_RST`) is not driven by the firmware; `R47` holds RESET high.

The recorder defines the USB record types `0x17`–`0x19` for a DP IMU ([`h05`](h05_microcontroller_firmware.md) §3.2); the firmware does not send them.

### Step 8.5 — Signal flow (ASCII)

```text
                    ┌── R48 / R49 (2.2k) ── +3V3D
  mcu GPIO18/19 ────┤ hierarchical IMU_CRA_SDA / IMU_CRA_SCL
                    │
              ┌─────▼─────┐
              │  U_MUX1   │
              │  PCA9546A │
              └──┬────┬───┘
           SD0/SC0    SD1/SC1
                 │         │
           J_IMU_CRA1   J_IMU_DP1
           + INT→GPIO17   + INT→GPIO12
           + nRST─────────┴── IMU_NRST → GPIO22

  mcu GPIO20/21 ──────── J_IMU_MTA1  (I2C0, no mux; R42 / R46 2.2k)
```

---

## Step 9 — PCB layout

On the board (revision 0.0.1), all parts of this sheet are on the **top** side:

- The three JSTs sit in a row on the long edge opposite the 40-pin header (`J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1`, from x ≈ 20 mm to 41 mm at y ≈ 53 mm), away from the EMG pin headers on the analog extension.
- `U_MUX1` sits about 9 mm inboard of `J_IMU_CRA1`.
- The link LEDs `LED_MTA1`, `LED_CRA1`, `LED_DP1` are in the LED column next to the MCU status LEDs (x ≈ 15 mm), not at the connectors.
- Pin **7** (`IMU_SHLD`) of each connector is on **`AGND`**.

### Step 9b — Probe points

Revision 0.0.1 has no test pads on **`imu_connectors`**. The nets are probed at the connectors, harness off:

| Net | Probe point |
|-----|-------------|
| `IMU_MTA_SDA` (GPIO20) | `J_IMU_MTA1` pin 3 |
| `IMU_MTA_SCL` (GPIO21) | `J_IMU_MTA1` pin 4 |
| `IMU_MTA_INT` (GPIO16) | `J_IMU_MTA1` pin 5 |
| CRA / DP SDA, SCL, INT (connector side of the mux) | `J_IMU_CRA1` / `J_IMU_DP1` pins 3, 4, 5 |

Logic-analyzer ground: **`J_SWD1` pin 1** or **`TP_DGND_PWR1`**. Full map: [`h02`](h02_custom_hat_shield.md) §6.1.

---

## Checklist of the captured sheet

| Check | Revision 0.0.1 |
|-------|----------------|
| JST count | 3 (`J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1`) |
| `U_MUX1` | Fitted; no bypass links |
| I2C0 pull-ups | `R42`, `R46` — 2.2 kΩ |
| I2C1 pull-ups | `R48`, `R49` — 2.2 kΩ, upstream of the mux |
| CRA I2C | Via SD0/SC0 |
| DP I2C | Via SD1/SC1 |
| INT | GPIO16 (MTA), GPIO17 (CRA), GPIO12 (DP) |
| nRST | GPIO22 + `R50` 10 kΩ pull-up |
| Mux reset | GPIO23 + `R47` 10 kΩ pull-up |
| Mux address | A0–A2 → `DGND` → 0x70 |

---

## Schematic status

[`imu_connectors.kicad_sch`](../../hardware/jtsmm_hat/sheets/imu_connectors.kicad_sch) (revision 0.0.1) has three **7-pin** jacks — `J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1` — each `Connector:Conn_01x07_Pin` with footprint `Connector_JST:JST_SH_SM07B-SRSS-TB_1x07-1MP_P1.00mm_Horizontal`, pin 7 on **`AGND`**. `U_MUX1` is populated (CRA on mux channel 0, DP on channel 1) and there are no bypass resistors. Checklist: [`h04_imu_kicad_implementation.md`](h04_imu_kicad_implementation.md).

**Pinout sanity:** With `Conn_01x07_Pin` at **0°** (pins on the right), KiCad pin **1** is the top pad → matches [Step 1](#step-1--7-pin-jst-pinout-all-jacks) (`+3V3D` on pin 1, **`IMU_SHLD` on pin 7**). Confirm in the footprint editor against the JST land.

---

## Suggested capture order

1. Power symbols (`+3V3D`, `DGND`, `AGND`).
2. `J_IMU_MTA1` + I2C0 pull-ups + bypass caps (Step 2).
3. Upstream I2C1 (`IMU_CRA_SDA` / `IMU_CRA_SCL`) + pull-ups + hierarchical labels.
4. `U_MUX1` with address, RESET and VCC bypass (Step 8.2).
5. `J_IMU_CRA1` on channel 0 (Step 3) and `J_IMU_DP1` on channel 1 (Step 8.3).
6. Shared `IMU_NRST` (Step 6) and the link LEDs (Step 6b).
7. ERC (Step 7).

---

## Hierarchical labels (`imu_connectors` ↔ `mcu`)

| `imu_connectors` | `mcu` |
|------------------|-------|
| `IMU_MTA_SDA` | `IMU_MTA_SDA` |
| `IMU_MTA_SCL` | `IMU_MTA_SCL` |
| `IMU_MTA_INT` | `IMU_MTA_INT` |
| `IMU_CRA_SDA` | `IMU_CRA_SDA` |
| `IMU_CRA_SCL` | `IMU_CRA_SCL` |
| `IMU_CRA_INT` | `IMU_CRA_INT` |
| `IMU_NRST` | `IMU_NRST` |
| `IMU_DP_INT` | `IMU_DP_INT` |
| `I2C_MUX_RST` | `I2C_MUX_RST` |
| `LED_IMU_MTA_ACT` | `LED_IMU_MTA_ACT` |
| `LED_IMU_CRA_ACT` | `LED_IMU_CRA_ACT` |
| `LED_IMU_DP_ACT` | `LED_IMU_DP_ACT` |
