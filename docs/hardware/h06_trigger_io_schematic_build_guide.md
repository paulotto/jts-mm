<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h06 — Trigger / Sync I/O Schematic Build Guide (KiCad capture walkthrough)

Step-by-step companion for the **camera trigger fan-out** and **optional visible sync LED** on KiCad sheet `hardware/jtsmm_hat/sheets/trigger_io.kicad_sch`.

> **Spec:** [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §7 (BOM), [`h06_synchronization.md`](h06_synchronization.md) §3–§4 (timing). **Default build:** **2× IMX296 GS cameras** on **XTR+ / XTR−** (hardware trigger) — **no IR ring drivers** on the HAT.  
> **MCU GPIO:** [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) Step 4b — **GPIO13** → `TRIG_PIO`, **GPIO14** → `SYNC_LED_PIO` (only wired on this sheet if the sync-LED block is populated).  
> **As captured (revision 0.0.1):** `trigger_io.kicad_sch` holds the **default GS trigger path** (74LVC2G241 fan-out, 1.8 V dividers, CSI JSTs, scope TPs, trigger LED). The optional blocks below (sync LED, USB opto, BNC) are **not on the sheet**. The guide uses short working designators (**U1**, **R4**, **J1**, …); the designators on the sheet are listed in [Step 2b](#reference-designators-on-trigger_iokicad_sch) — see also [`h10_schematic_logical_review.md`](h10_schematic_logical_review.md).

## Default build vs optional blocks

**Populate for the validated GS-camera rig** (Track B default in [`00_overview.md`](../00_overview.md)):

| Block | Refs (example) | Role |
|-------|----------------|------|
| **Camera trigger fan-out** | **U1**, **U2**, **R4**, **C1**–**C6**, **J1**, **J2**, **TP1**, **TP2** | **GPIO13** → buffers → **2× CSI XTRIG** + scope test pads |
| **Trigger panel LED** | **D_TRIG**, **R8** | Operator sees **`TRIG_PIO`** activity (amber) |
| **XTRIG dividers (on-HAT)** | **R51**, **R53**, **R57**, **R59** | **1.5 kΩ + 1.8 kΩ** per CSI channel — Pi GS values; **~1.81 V** at **J1/J2** |

**Options, not on revision 0.0.1** (no symbols on the sheet, no footprints on the board):

| Block | Refs (example) | Use |
|-------|----------------|------------------|
| **USB3 camera trigger** | **U3**, **R3**, **R7** (+ optional **JST** for isolated return) | Optional **3rd** camera — **Arducam UC-599** ([`h01`](h01_camera_subsystem.md)); **not** used for Pi CSI IMX296 |
| **Visible sync LED** | **Q1**, **R5**, **R6**, **J3**, **D_REMOTE**, **D_FLASH**, **R9** | **Track A webcams** (no HW trigger) or an optional photonic cross-check; **not required** when both GS cameras use **XTRIG** |
| **BNC scope jack** | **J4** | Repeatable coax hookup for multi-board labs; **TP1** / **TP2** serve the same purpose |

On revision 0.0.1 **`SYNC_LED_PIO`** remains on the **`mcu`** sheet (GPIO14 / firmware `led_flash`) and is **not connected** on **`trigger_io`**; **U2 pin 6** (**1Y**, `U_TP_1`) is not connected.

```text
Default (GS only):     TRIG_PIO ──► U1 ──► [1.5k/1.8k ÷] ──► J1/J2 (~1.8 V) ──► IMX296 XTR+/XTR−
                                    └── U2 pin 3 (2Y) ──[R4 50Ω]──► TP1 ; TP2 → DGND

Optional BNC:          same net ──► J4 center (shell → DGND) — parallel to TP1

Optional USB:          U2 1Y ──► U3 opto ──► TRIG_USB (isolated)

Optional sync LED:     SYNC_LED_PIO ──► Q1 ──► J3 ──► white LED in FOV (Track A)
```

## How to read the diagrams

Same legend as [`_schematic_build_guide_conventions.md`](_schematic_build_guide_conventions.md).

**Pin names** on the **74LVC2G241** follow the TI / Nexperia datasheet (not arbitrary):

| Name | Package pin (TSSOP-8) | Role |
|------|------------------------|------|
| **1OE** | 1 | Output enable, **gate 1** — **active LOW** (tie to **`DGND`** to turn gate 1 **on**) |
| **1A** | 2 | Data **input**, gate 1 |
| **2Y** | 3 | Data **output**, gate 2 |
| **GND** | 4 | IC ground → **`DGND`** |
| **2A** | 5 | Data **input**, gate 2 |
| **1Y** | 6 | Data **output**, gate 1 |
| **2OE** | 7 | Output enable, **gate 2** — **active HIGH** (tie to **`+3V3D`** to turn gate 2 **on**) |
| **VCC** | 8 | Supply → **`+3V3D`** |

Gate I/O shorthand in wiring diagrams: **`1A`** = input of gate 1, **`1Y`** = output of gate 1, **`2A`** / **`2Y`** = gate 2.  
**`Ux`** = buffer IC reference (**U1** or **U2**). **`Ux` pin 8** = **`VCC`** (same as “**`Ux` VCC pin**” in decoupling notes).

```text
TRIG_PIO ──┬──► U1 pin 2 (1A)     star: one net fans to all four buffer inputs
           ├──► U1 pin 5 (2A)
           ├──► U2 pin 2 (1A)
           └──► U2 pin 5 (2A)
[50Ω]                       series resistor on U2 scope tap (2Y → TP1) only
+3V3D / DGND / +5V_IN       power symbols (global nets on this sheet)
```

---

## Step 0 — Symbols, footprints, and BOM

> **Footprint audit:** [`h09_footprint_master_table.md`](h09_footprint_master_table.md) · test points use `TestPoint:TestPoint_Pad_D1.0mm` (stock KiCad).

### Required (default GS build)

| Ref (example) | Catalog | Part / value | KiCad symbol | Footprint | Qty | Notes |
|---------------|---------|--------------|--------------|-----------|-----|--------|
| **U1**, **U2** (`U_CSI_1`, `U_TP_1`) | **H-IC-07** | **SN74LVC2G241DCUR** (TI **74LVC2G241**) | `74xGxx:74LVC2G241` or `Logic_74xx:SN74LVC2G241` | `Package_SO:VSSOP-8_2.3x2mm_P0.5mm` | 2 | **LCSC C10430**; **3 units each**; **U1** → CSI; **U2 pin 3** → **R4** → **TP1** / optional **J4**; **TP2** on **`DGND`** |
| **R51**, **R53** | — | **1.5 kΩ** 1 % | `Device:R` | 0603 | 2 | XTRIG divider **series** — **U1** **1Y** / **2Y** → **`TRIG_CSI1/2`** (Step 2b) |
| **R57**, **R59** | — | **1.8 kΩ** 1 % | `Device:R` | 0603 | 2 | XTRIG divider **shunt** — **`TRIG_CSI1/2`** → **`DGND`** |
| **R4** | — | **50 Ω** | `Device:R` | 0603 | 1 | **`TRIG_BNC`** net — between **U2 pin 3** (**2Y**) and **TP1** / optional **J4** |
| **R8** | **H-PAS-05** | **470 Ω–1 kΩ** | `Device:R` | 0603 | 1 | Panel **D_TRIG** (amber) |
| **C1**, **C2** | — | **100 nF** | `Device:C` | 0603 | 2 | **100 nF** from **`+3V3D`** to **`DGND`**, one at **U1 pin 8** and one at **U2 pin 8** |
| **C61**, **C63** | **H-PAS-04** | **1 µF** | `Device:C` | 0603 | 2 | Bulk **`+3V3D`** bypass, within **3 mm** of **U1** / **U2** **VCC** — **GRM188R71E105KA12D** (same as EMG 1 µF jellybean) |
| **D_TRIG** | **H-PAS-05** | Amber **0603** | `Device:LED` | `LED_SMD:LED_0603_1608Metric` | 1 | Panel copy of **`TRIG_PIO`** |
| **J1**, **J2** | **H-CON-03** | **JST-SH 2-pin** `SM02B-SRSS-TB` | `Connector:Conn_01x02_Pin` | `Connector_JST:JST_SH_SM02B-SRSS-TB_1x02-1MP_P1.00mm_Horizontal` | 2 | **`TRIG_CSI1/2`** (~**1.8 V** at pin 1) → camera **XTR+**; pin 2 → **XTR−** |
| **TP1** | — | **`TP_TRIG`** | `TestPoint:TestPoint` | `TestPoint:TestPoint_Pad_D1.0mm` | 1 | Scope **signal** — on **`TRIG_BNC`** after **R4** (Step 2c) — **~3.3 V** MCU reference |
| **TP2** | — | **`TP_DGND`** | `TestPoint:TestPoint` | `TestPoint:TestPoint_Pad_D1.0mm` | 1 | Scope **ground** — on **`DGND`** (11.8 mm from **TP1** on the board) |
| **TP3**, **TP4** | — | **`TP_CSI1`**, **`TP_CSI2`** | `TestPoint:TestPoint` | `TestPoint:TestPoint_Pad_D1.0mm` | 2 | Scope **camera-side** trigger (~**1.8 V**) — on **`TRIG_CSI1/2`** at **J1/J2 pin 1** (Step 2b); **default capture** ([`h02`](h02_custom_hat_shield.md) §6.1) |
| **TP5** | — | **`TP_TRIG_PIO`** | `TestPoint:TestPoint` | `TestPoint:TestPoint_Pad_D1.0mm` | 1 | **PIO source** — on **`TRIG_PIO`** before **U1/U2** buffers; separates firmware from 74LVC fan-out when **`LED_TRIG1`** blinks but cameras do not sync |

### Options (not on revision 0.0.1)

| Ref (example) | Catalog | Part / value | KiCad symbol | Footprint | Qty | Use |
|---------------|---------|--------------|--------------|-----------|-----|----------------|
| **U3** | — | **TLP291** (Toshiba SO4) or **PC817** | `Coupling:TLP291` or `Coupling:PC817` | `Package_SO:SOP-4_4.4x2.3mm_P1.27mm` / `Package_DIP:SMD-4` | 1 | **3rd USB3 GS camera** — **`TRIG_USB`** (Step 3) |
| **R3** | — | **220 Ω** | `Device:R` | 0603 | 1 | With **U3** — opto LED current limit |
| **R7** | — | **10 kΩ** | `Device:R` | 0603 | 1 | With **U3** — opto collector pull-up |
| **Q1** | **H-IC-11** | **AO3400A** / **2N7002** | `Transistor_FET:AO3400A` or `Device:Q_NMOS_GSD` | `Package_TO_SOT_SMD:SOT-23` | 1 | **Track A webcams** or optional flash cross-check (Step 4) |
| **R5** | — | **1 kΩ** | `Device:R` | 0603 | 1 | With **Q1** — FET gate from `SYNC_LED_PIO` |
| **R6** | — | **150–330 Ω** (tune) | `Device:R` | 0603 | 1 | With **Q1** — remote sync LED current (~20 mA @ 5 V) |
| **R9** | **H-PAS-05** | **470 Ω–1 kΩ** | `Device:R` | 0603 | 1 | With sync LED — panel **D_FLASH** (white) |
| **D_FLASH** | **H-PAS-07** | White **0603** | `Device:LED` | `LED_SMD:LED_0603_1608Metric` | 1 | With sync LED — panel copy of **`SYNC_LED_PIO`** |
| **D_REMOTE** | — | **5 mm white** (or 0603 on pigtail) | — | Off-board | 1 | With sync LED — in FOV; mates **J3** |
| **J3** | **H-CON-04** | **JST-SH 2-pin** | `Connector:Conn_01x02_Pin` | same as **J1** | 1 | With sync LED — remote **`SYNC_LED`** harness |
| **J4** | **H-CON-05** | **BNC** jack (or **SMB**) | `Connector:BNC` / `Conn_Coaxial:BNC_JACK` | Vendor BNC edge footprint | 1 | Optional coax scope tap — same **`TRIG_BNC`** net as **TP1** (Step 2c) |

Order example: **TLP291(GB-TP,SE)** (CTR rank **GB**, tape **TP**). There is **no** separate base part **TLP291L** — “L” in some order codes is tape/reel notation, not a different IC.

### KiCad 10 — symbol notes

- **74LVC2G241:** use **`74xGxx:74LVC2G241`** or **`Logic_74xx:SN74LVC2G241`**. The library symbol is a **multi-unit De Morgan** graphic — **not** a single 8-pin rectangle:
  - **Unit A** — gate 1 only: pins **1** (**1OE**), **2** (**1A**), **6** (**1Y**) → looks like “3 pins”
  - **Unit B** — gate 2 only: pins **3** (**2Y**), **5** (**2A**), **7** (**2OE**)
  - **Unit C** — power: pins **4** (**GND**), **8** (**VCC**)
  - **You must place all three units** for each IC (**U1**, **U2**). After the first unit: select the symbol → **`A`** (Add another unit) twice, or place with **“All units”** if your KiCad version offers it. ERC will flag missing **VCC**/**GND** if unit C is omitted.
  - Enable **Preferences → Schematic Editor → Display → Pin number** and **Pin name** so **1OE** / **1Y** show on the sheet.
  - Footprint: **`Package_SO:VSSOP-8_2.3x2mm_P0.5mm`** (TI **DCU** / **SN74LVC2G241DCUR**, LCSC **C10430**). Same **pin 1–8** map as TSSOP — only the land pattern changes; **re-place `U_CSI_1` / `U_TP_1` on the PCB** after updating the schematic footprint. Do **not** substitute **SN74LVC2G17** (inverting Schmitt — wrong part).

> **PCB layout (2026-06):** After this footprint change, open **`jtsmm_hat.kicad_pcb`** → **Update PCB from schematic** → for **`U_CSI_1`** and **`U_TP_1`**, accept the **VSSOP-8** footprint swap, re-place at the same locations, and re-route the eight pads. MPN/LCSC fields are **SN74LVC2G241DCUR / C10430**.
- **JST-SH:** same pattern as IMU jacks — generic `Conn_01x0N_Pin` symbol + **`Connector_JST`** footprint ([`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md) Step 0).
- **Test points:** `TestPoint:TestPoint` + **`TestPoint:TestPoint_Pad_D1.0mm`** (1.0 mm round SMD pad). `TestPoint_1.0mm` is **not** in default KiCad libraries.
- **Power on this sheet:** use project globals **`jtsmm:+3V3D`**, **`jtsmm:DGND`**, **`jtsmm:+5V_IN`** (same as `imu_connectors` / `emg` DVDD — nets exported from [`power.kicad_sch`](../../hardware/jtsmm_hat/sheets/power.kicad_sch) per [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md)).

---

## Step 1 — Sheet ports and power

### Hierarchical inputs (from `mcu` sheet)

| Label | Direction on `trigger_io` | RP2350A / firmware |
|-------|---------------------------|-------------------|
| `TRIG_PIO` | **in** | **GPIO13** — PIO `trigger_gen` |
| `SYNC_LED_PIO` | **in** (only with the sync-LED block, Step 4) | **GPIO14** — PIO `led_flash`; not on this sheet on revision 0.0.1 (the label ends at the `mcu` sheet port) |

On the root sheet `jtsmm_hat.kicad_sch` the **trigger_io** block has one pin, **`TRIG_PIO`**, wired **`mcu` out → `trigger_io` in**; a sync-LED block would need a second pin for **`SYNC_LED_PIO`**.

### Power symbols (on `trigger_io` — not hierarchical)

Place **`jtsmm:+3V3D`**, **`jtsmm:DGND`**, and **`jtsmm:+5V_IN`** power symbols wherever needed. They join the project-wide nets from the **power** sheet (no extra root pins required).

| Net | Use on this sheet |
|-----|-------------------|
| `+3V3D` | **U1** / **U2** pin **8** (**VCC**); **U1** / **U2** pin **7** (**2OE**); opto collector pull-up (**R7**) |
| `DGND` | **U1** / **U2** pin **4** (**GND**); **U1** / **U2** pin **1** (**1OE**); FET source; panel LED cathodes; JST pin 2 (**XTR−**); opto LED cathode |
| `+5V_IN` | Remote sync LED anode supply (via **R6**) — only with the sync-LED block (Step 4); not used on revision 0.0.1 |

> **Do not** return sync or trigger currents on **`AGND`**. Patient analog ground stays on the EMG sheet ([`h07`](h07_power_grounding_safety.md) §2).

---

## Step 2 — Camera trigger fan-out (required — 2× 74LVC2G241)

**One `74LVC2G241` = two independent non-inverting buffers (gates 1 and 2).** Four fan-out branches need **two** ICs: **U1** (both CSI cameras) and **U2** (scope tap + optional USB opto).

### TSSOP-8 pinout (both **U1** and **U2**)

```text
        ┌─ pin 1  1OE  (active LOW enable, gate 1)
        │  pin 2  1A   (input, gate 1)
        │  pin 3  2Y   (output, gate 2)
  GND ──┤  pin 4  GND
        │  pin 5  2A   (input, gate 2)
        │  pin 6  1Y   (output, gate 1)
        │  pin 7  2OE  (active HIGH enable, gate 2)
  VCC ──┴─ pin 8  VCC
```

Both gates must be **enabled** on each IC:

| Pin | Net | Why |
|-----|-----|-----|
| **1OE** (pin 1) | **`DGND`** | **LOW** → gate 1 drives (**1Y** follows **1A**) |
| **2OE** (pin 7) | **`+3V3D`** | **HIGH** → gate 2 drives (**2Y** follows **2A**) |

Do **not** tie **both** **1OE** and **2OE** to **`DGND`** — gate 2 stays **high-Z** and **2Y** will not drive the scope or USB path.

### Complete wiring — **U1** (CSI cameras)

| IC pin | Name | Connect to |
|--------|------|------------|
| 1 | **1OE** | **`DGND`** |
| 2 | **1A** | **`TRIG_PIO`** (from **`mcu`**) |
| 3 | **2Y** | **R53** (1.5 kΩ) → **`TRIG_CSI2`** (+ **R59** 1.8 kΩ shunt, Step 2b) → **J2** pin 1 |
| 4 | **GND** | **`DGND`** |
| 5 | **2A** | **`TRIG_PIO`** (same net as pin 2) |
| 6 | **1Y** | **R51** (1.5 kΩ) → **`TRIG_CSI1`** (+ **R57** 1.8 kΩ shunt, Step 2b) → **J1** pin 1 |
| 7 | **2OE** | **`+3V3D`** |
| 8 | **VCC** | **`+3V3D`** + **C1** (100 nF) to **`DGND`** at the pin |

**J1** / **J2** pin 2 → **`DGND`** (camera **XTR−**).

### Complete wiring — **U2** (scope + optional USB)

| IC pin | Name | Connect to |
|--------|------|------------|
| 1 | **1OE** | **`DGND`** |
| 2 | **1A** | **`TRIG_PIO`** |
| 3 | **2Y** | **R4** (50 Ω) → net **`TRIG_BNC`** → **TP1** (+ optional **J4** center) — Step 2c |
| 4 | **GND** | **`DGND`** |
| 5 | **2A** | **`TRIG_PIO`** |
| 6 | **1Y** | not connected on revision 0.0.1; with the USB opto option: **R3** (220 Ω) → **U3** LED anode (Step 3) |
| 7 | **2OE** | **`+3V3D`** |
| 8 | **VCC** | **`+3V3D`** + **C2** (100 nF) to **`DGND`** at the pin |

**Scope test pads (default build)** — not IC pins; place next to **U2** (Step 2c):

| Ref | Net | Connect to |
|-----|-----|------------|
| **TP1** | **`TRIG_BNC`** | Same node as **U2 pin 3** after **R4** (50 Ω) — scope **signal** |
| **TP2** | **`DGND`** | **`DGND`** (same domain as **U2 pin 4**) — scope **ground** (11.8 mm from **TP1** on the board) |

Optional **J4** (BNC) center ties to **`TRIG_BNC`** with **TP1**; **J4** shell → **`DGND`**.

### Decoupling (per IC)

Place at **U1 pin 8** / **U2 pin 8** (**VCC**), return to **`DGND`** at **pin 4**:

```text
+3V3D ──┬── Cx 100nF ── DGND     (Cx = C1 at U1, C2 at U2 — at the VCC pin)
        └── Cy 1µF ──── DGND     (Cy = C61 near U1, C63 near U2 — bulk 0603, ≤3 mm from IC)
```

### Signal topology (summary)

Single net **`TRIG_PIO`** stars to all **four** inputs (**1A** and **2A** on **U1** and **U2**). **`D_TRIG`** is a separate tap on **`TRIG_PIO`** (Step 4b). **Each buffer output has its own series resistor** — never share one resistor across branches.

```text
                    TRIG_PIO (from mcu, GPIO13)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    U1: 1A, 2A       U2: 1A, 2A     D_TRIG ──[R8]──► amber LED → DGND
    1OE→DGND         1OE→DGND
    2OE→+3V3D        2OE→+3V3D

U1 pin 6 (1Y) ──[R51 1.5k]──► TRIG_CSI1 ──► J1 pin 1     (R57 1.8k TRIG_CSI1→DGND; J1 pin 2 → DGND)
U1 pin 3 (2Y) ──[R53 1.5k]──► TRIG_CSI2 ──► J2 pin 1     (R59 1.8k TRIG_CSI2→DGND; J2 pin 2 → DGND)

U2 pin 6 (1Y) ── not connected (with the Step 3 option: ──[R3 220Ω]──► U3 LED)
U2 pin 3 (2Y) ──[R4  50Ω]──► TRIG_BNC ──► TP1 (+ optional J4)
TP2 (TP_DGND) ──► DGND          (scope ground — pair with TP1)
```

### Step 2b — XTRIG 1.8 V divider (on-HAT, default IMX296 build)

The buffer outputs **3.3 V**. Sony **IMX296 XTRIG** is **1.8 V** logic. Use the **Raspberry Pi Global Shutter external-trigger divider** directly from **U1** outputs — **no extra 100 Ω** in series (the **1.5 kΩ** leg already limits fault current and sets the level).

#### Reference designators on `trigger_io.kicad_sch`

The sheet (revision 0.0.1) implements the default circuit below. Its reference designators, against the working designators of this guide:

| This guide | On the sheet | Value / role |
|------------|--------------|--------------|
| **U1** | **`U_CSI_1`** | 74LVC2G241 — **1Y** → CSI1, **2Y** → CSI2 |
| **U2** | **`U_TP_1`** | 74LVC2G241 — **2Y** → scope tap; **1Y** (pin 6) not connected |
| **R51** + **R57** (CSI1 series + shunt) | **`R57`** + **`R58`** | **1.5 kΩ** + **1.8 kΩ** |
| **R53** + **R59** (CSI2 series + shunt) | **`R59`** + **`R60`** | **1.5 kΩ** + **1.8 kΩ** |
| **R4** | **`R55`** | **50 Ω** — **U2 2Y** → `TP_TRIG1` |
| **R8**, **D_TRIG** | **`R52`**, **`LED_TRIG1`** | **470 Ω**, amber LED on **`TRIG_PIO`** |
| **C1**, **C2** (100 nF) | **`C60`**, **`C62`** | at `U_CSI_1` / `U_TP_1` (0402) |
| **C61**, **C63** (1 µF) | **`C61`**, **`C63`** | at `U_CSI_1` / `U_TP_1` |
| **J1**, **J2** | **`J_TRIG_CSI1`**, **`J_TRIG_CSI2`** | JST-SH 2-pin |
| **TP1** … **TP5** | **`TP_TRIG1`**, **`TP_DGND_TRIG1`**, **`TP_CSI1`**, **`TP_CSI2`**, **`TP_TRIG_PIO1`** | scope test pads |

There is no further series resistor between **`U_CSI_1`** and the dividers — the Pi doc is **1.5 kΩ + 1.8 kΩ** only.

> **Ref collision:** **R51**, **R53** and **R56** are used on the **`mcu`** sheet — hence the divider designators **R57**–**R60** on this sheet.

**U2 → TP_TRIG1** is **50 Ω** at **~3.3 V** (scope reference only — **no** camera divider on **U2**).

#### Default circuit (matches Pi GS doc)

Per channel (**CSI1:** **R51** + **R57**; **CSI2:** **R53** + **R59**):

```text
U1 1Y/2Y ──[R51/R53 1.5kΩ]──► TRIG_CSIx ──► Jx pin 1 (XTR+)
                                   │            (~1.81 V high)
                              [R57/R59 1.8kΩ]
                                   │
                                 DGND         Jx pin 2 (XTR−)
```

| Ref | Value | Net(s) | Notes |
|-----|-------|--------|-------|
| **R51**, **R53** | **1.5 kΩ** 1 % | **U1 1Y/2Y** → **`TRIG_CSI1/2`** | Divider **top** — [Pi GS external-trigger doc](https://www.raspberrypi.com/documentation/accessories/global-shutter-camera.html) |
| **R57**, **R59** | **1.8 kΩ** 1 % | **`TRIG_CSI1/2`** → **`DGND`** | Divider **bottom** — same node as **Jx pin 1** |
| **J1**, **J2** pin 1 | — | **`TRIG_CSI1/2`** | Camera **XTR+** — **~1.81 V** high, **~0 V** low |
| **J1**, **J2** pin 2 | — | **`DGND`** | Camera **XTR−** |

**Expected levels (3.3 V buffer high):**  
`V_XTR+ ≈ 3.3 V × 1.8 / (1.5 + 1.8) ≈ 1.81 V`

**Harness:** plain **2-conductor** JST-SH per camera — no extra parts at the module.

#### Scope bring-up

| Probe point | Net | Expected |
|-------------|-----|----------|
| **TP1** | **`TRIG_BNC`** | **~3.0–3.3 V** — MCU/buffer reference (unchanged) |
| **TP3** / **TP4** (`TP_CSI1/2`) | **`TRIG_CSI1/2`** at **Jx pin 1** | **~1.8 V** high — camera-side XTRIG |
| **TP5** (`TP_TRIG_PIO`) | **`TRIG_PIO`** (before **U1/U2**) | **~3.3 V** — raw PIO output |
| **Jx pin 1** | **`TRIG_CSIx`** | Same node as **TP3/TP4** |

Compare **TP5** (PIO) → **TP1** (buffered 3.3 V) → **TP3** (~1.8 V camera) to localize fan-out bugs. **`LED_TRIG1`** mirrors PIO activity but cannot measure voltage levels or fall time.

Compare **TP1** (3.3 V domain) to **J1 pin 1** (1.8 V domain) for fan-out and polarity before locking the harness.

See also [`h01_camera_subsystem.md`](h01_camera_subsystem.md) §3.1.1 and [`h06_synchronization.md`](h06_synchronization.md) §3.2.

### IMX296 polarity

Fast trigger: exposure starts on **XTRIG low** ([`h06_synchronization.md`](h06_synchronization.md) §3.1). Scope **TP1** (or optional **J4**) + `trigger_mode=1` on Pi before locking PCB revision.

---

## Step 2c — Scope test pads (`TP1` / `TP2`) — **required for default build**

Bench-only tap of the buffered trigger — **not** used during normal recordings. Replaces a **BNC jack** for bring-up and timing checks unless you populate **J4** as well.

### Schematic

```text
U2 pin 3 (2Y) ──[R4 50Ω]──► net TRIG_BNC ──► TP1 (TP_TRIG)
                              └──► J4 center (optional, same net)

TP2 (TP_DGND) ──► DGND     (local scope return — not AGND)
```

| Ref | Value | Net | Notes |
|-----|-------|-----|--------|
| **TP1** | `TP_TRIG` | **`TRIG_BNC`** | **After** **R4** — MCU-side trigger reference (~3.0–3.3 V swing) |
| **TP2** | `TP_DGND` | **`DGND`** | Scope ground clip — **never** **`AGND`** |

**J4 (optional):** wire **J4 center** to the same **`TRIG_BNC`** node as **TP1**; **J4 shell → DGND**. Revision 0.0.1 has no **J4**; probe **TP1** / **TP2**.

**Is R4 (50 Ω) required for test pads only?** No — not for **function** with a normal **high‑impedance** scope probe. **Keep R4 populated anyway** for fault limiting, optional **J4**, and validation at **TP1**. **50 Ω** is **only** on the **U2** scope tap — CSI camera paths use the **1.5 kΩ / 1.8 kΩ** divider, not 100 Ω.

### PCB placement ([`h02`](h02_custom_hat_shield.md) §7.4)

On the board (revision 0.0.1) the five trigger test pads are on the **top** copper layer in the trigger region at the short edge of the HAT+ core, so probes reach them with the HAT on the Pi:

| Pad (sheet ref) | Position (x, y in mm) | Distance to the ground pad `TP_DGND_TRIG1` |
|-----------------|-----------------------|--------------------------------------------|
| **TP2** `TP_DGND_TRIG1` | (51.0, 21.8) | — |
| **TP5** `TP_TRIG_PIO1` | (51.6, 27.1) | 5.3 mm |
| **TP3** `TP_CSI1` | (55.1, 25.8) | 5.7 mm |
| **TP4** `TP_CSI2` | (55.8, 28.0) | 7.8 mm |
| **TP1** `TP_TRIG1` | (50.6, 33.6) | 11.8 mm |

| Property | On the board |
|----------|--------------|
| **Ground proximity** | **TP1** is 11.8 mm from **TP2**; use a ground lead of matching length or the `DGND` pin 2 of **J1** / **J2** |
| **Edge access** | 9–14 mm inside the short edge (x = 65 mm), next to **U1** / **U2** and **J1** / **J2**; not under the Pi GPIO header |
| **Ground domain** | All pads in the `DGND` part of the board; no `AGND` pour or patient-analog traces nearby |
| **Silkscreen** | Reference designators (`TP_TRIG1`, `TP_DGND_TRIG1`, …) next to each pad |
| **Soldermask** | Exposed copper on all pads (KiCad test-point footprint) |

### Bench use

1. Scope **Ch1** → **TP1**; scope **ground** → **TP2** (or **J1/J2 pin 2**).
2. Send `trig_start` over USB-CDC (or `jtsmm trigger`) — expect the rate of the applied configuration (**40 Hz** with a **12 ms** low time with the template `software/jtsmm/etc/jtsmm.yaml`; **30 Hz** / **12 ms** with the firmware defaults), **3.0–3.3 V** amplitude, fall time **&lt;100 ns** ([`h07`](h07_power_grounding_safety.md) §10).
3. Compare **TP1** to **J1 pin 1** / **J2 pin 1** (or **TP3** / **TP4**) for polarity and fan-out — camera nets are **~1.8 V** after the on-HAT divider.

---

## Step 3 — USB camera optoisolated path (`TRIG_USB`) — **optional**

> Not on revision 0.0.1: **U3**, **R3**, **R7** have no symbols or footprints, and **U2 pin 6** (**1Y**) is not connected. The path is for an optional **Arducam UC-599** USB3 camera.

```text
U2 pin 6 (1Y) ──[R3 220Ω]──► U3 pin 1 (LED anode)
                               U3 pin 2 (LED cathode) ──► DGND
U3 pin 4 (phototransistor emitter) ──► TRIG_USB_GND  (isolated return — harness only)
U3 pin 3 (collector) ──┬──[R7 10kΩ]──► +3V3D
                       └──► TRIG_USB_SIG ──► USB camera trigger input (open-collector style)

Do not connect LED cathode to the phototransistor.
Do not tie TRIG_USB_GND to HAT DGND at the camera end.
```

Purpose: USB3 camera ground does not couple into HAT **`DGND`** / analog domain ([`h02`](h02_custom_hat_shield.md) §7). Confirm **Arducam UC-599** trigger polarity before fab.

Optional: second **2-pin JST** for `TRIG_USB` + return if the harness is not wired inside the opto cable assembly.

---

## Step 4 — Visible sync LED (`SYNC_LED` / `J3`) — **optional**

> Not on revision 0.0.1: **Q1**, **R5**, **R6**, **J3**, **D_REMOTE**, **D_FLASH**, **R9** have no symbols or footprints. With the GS cameras, IMX296 **`SensorTimestamp`** + **`TRIGGER_EVENT`** ([`h06_synchronization.md`](h06_synchronization.md) §3) is sufficient. The block is for **Track A webcams** (no electrical trigger) or an optional photonic cross-check.

Low-side switch on **`DGND`**; anode fed from **`+5V_IN`** (~20 mA).

```text
SYNC_LED_PIO ──[R5 1kΩ]──► Q1 gate
                             Q1 source ──► DGND
                             Q1 drain  ──► net LED_CATHODE

+5V_IN ──[R6 150–330Ω]──► J3 pin 1 ── cable ──► D_REMOTE anode
                             D_REMOTE cathode ── cable ──► J3 pin 2 ──► LED_CATHODE (→ Q1 drain)
```

| Item | Value / rule |
|------|----------------|
| **R6** | **150 Ω** ≈ 23 mA @ 5 V; **220 Ω** ≈ 16 mA; **330 Ω** ≈ 11 mA — stay **≤ 25 mA** |
| **Q1** | **AO3400A** logic-level NMOS; **1 kΩ** gate resistor |
| **Return** | **`DGND` only** — not **`AGND`** |
| **Silkscreen** | **SYNC LED** (remote) vs **TRIG** (camera path) |

Firmware: `flash on` / `flash off` ([`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md)).

---

## Step 4b — Panel indicator LEDs

Place on the **trigger corner** ([`h02`](h02_custom_hat_shield.md) §7.4), visible to the operator.

| Ref | Footprint | Color (silk) | Default build | Driven from | Wiring |
|-----|-----------|--------------|---------------|-------------|--------|
| **D_TRIG** | `LED_SMD:LED_0603_1608Metric` | **TRIG** (amber) | fitted (`LED_TRIG1`) | **`TRIG_PIO`** (same net as buffer inputs) | `TRIG_PIO ──[R8 470Ω RC0603FR-07470RL]──► anode; cathode ──► DGND`; LED MPN **LTST-C190KFKT** |
| **D_FLASH** | `LED_SMD:LED_0603_1608Metric` | **FLASH** (white) | not on the board (Step 4) | **`SYNC_LED_PIO`** | `SYNC_LED_PIO ──[R9 470Ω–1kΩ]──► anode; cathode ──► DGND` |

**No extra GPIO** — local taps of existing PIO nets. Do **not** hang **D_FLASH** only on the FET gate (gate sees **R5**); tap **`SYNC_LED_PIO`** directly when the sync block is populated.

---

## Step 5 — Connector pinouts

| Ref | Catalog | Pin 1 | Pin 2 | Shell / notes |
|-----|---------|-------|-------|----------------|
| **J1** `J_TRIG_CSI1` | H-CON-03 | **`TRIG_CSI1`** (~**1.8 V**) → **XTR+** | **DGND** → **XTR−** | Horizontal **SM02B** on PCB edge |
| **J2** `J_TRIG_CSI2` | H-CON-03 | **`TRIG_CSI2`** (~**1.8 V**) → **XTR+** | **DGND** → **XTR−** | Same |
| **J3** `J_SYNC_LED` | H-CON-04 | `+5V_IN` side (after **R6**) | `LED_CATHODE` → FET drain | Not on the board — ~30 cm cable to **D_REMOTE** |
| **J4** `J_TRIG_BNC` | H-CON-05 *(optional)* | `TRIG_BNC` (via **R4**, same net as **TP1**) | — | **Shell → DGND** — not on the board |

### Scope test pads (default)

| Ref | Net | Notes |
|-----|-----|--------|
| **TP1** `TP_TRIG1` | **`TRIG_BNC`** (after **R4**) | Scope **signal** — see Step 2c |
| **TP2** `TP_DGND_TRIG1` | **`DGND`** | Scope **ground** |
| **TP3** `TP_CSI1` | **`TRIG_CSI1`** | Camera 1 XTRIG (~1.8 V) |
| **TP4** `TP_CSI2` | **`TRIG_CSI2`** | Camera 2 XTRIG (~1.8 V) |
| **TP5** `TP_TRIG_PIO1` | **`TRIG_PIO`** | PIO source before buffers |

Full map + alternative probe points: [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §6.1.

Label every JST on silkscreen. Mate: **SHR-02V-S-B** (or GH series at cable end — latch differs; do not mix on one plug).

---

## Step 6 — ESD / bench protection

| Net | Default protection |
|-----|-------------------|
| `TRIG_CSI1/2` | **1.5 kΩ + 1.8 kΩ** divider at **U1** outputs (**R51/R53 + R57/R59**) — **no** 100 Ω |
| `TRIG_BNC` / **TP1** | **50 Ω** series at **U2 pin 3**; scope ground → **TP2** or **`DGND`** |
| `TRIG_USB` | Isolation via **U3** |
| JST pins (optional) | **USBLC6-2SC6** or **TPD2E001** on external pins if cables leave the bench — no such footprint on revision 0.0.1 ([`h02`](h02_custom_hat_shield.md) §7.3) |

---

## Step 7 — Blocks not on revision 0.0.1

| Block | Status |
|-------|--------|
| IR LED ring constant-current driver | no footprint |
| `RING_EN` / `RING_PWR` high-current FET | no footprint ([`h07`](h07_power_grounding_safety.md) §1.4) |
| USB opto **`TRIG_USB`** (**U3**, **R3**, **R7**) | no footprint; for an optional 3rd USB3 camera |
| BNC jack **J4** | no footprint (test pads **TP1** / **TP2** are on the board) |
| Visible sync LED (**Q1**, **J3**, **D_FLASH**, …) | no footprint; for Track A webcams or a flash cross-check |

---

## Step 8 — ERC

- **U1**, **U2**: all **8** package pins connected — especially **pin 4** (**GND**) and **pin 8** (**VCC**); **1OE → DGND**, **2OE → +3V3D**; **1A** / **2A** on **`TRIG_PIO`**; outputs not floating (**NC** only on **U2 pin 6** when USB opto **DNP**).
- **KiCad:** each **74LVC2G241** has **three schematic units** (gate 1, gate 2, power) — if ERC reports missing **VCC**/**GND**, add units **B** and **C** (**`A`** hotkey).
- **No** output-to-output shorts; **four** independent series resistors (**R1**–**R4**).
- **`TRIG_CSI1` ≠ `TRIG_CSI2`** (separate nets and connectors).
- **Opto:** LED side on **DGND**; phototransistor return **not** tied to **`AGND`**.
- **Sync LED** *(if populated):* FET source on **DGND**; **`+5V_IN`** only for LED anode path.
- **PWR_FLAG** not required on child sheet if globals are driven from **power** sheet (project ERC at root).

---

## Hierarchical labels (`trigger_io` ↔ root)

| `trigger_io` | Root / `mcu` connection |
|--------------|-------------------------|
| `TRIG_PIO` | **`mcu`** hierarchical **out** → **`trigger_io`** hierarchical **in** |
| `SYNC_LED_PIO` | **`mcu`** hierarchical **out**; not connected to **`trigger_io`** on revision 0.0.1 (only with the sync-LED block) |
| `+3V3D`, `DGND`, `+5V_IN` | Global nets from **`power`** / **`hat_plus`** (symbols on sheet — no root pins) |

---

## Pre-capture / pre-fab checklist

| Check | Pass criteria |
|-------|----------------|
| Default build | **2× CSI** + **TP1** / **TP2** from **`TRIG_PIO`**; **J4**, **USB opto**, and **sync LED** not on the sheet |
| Fan-out (full PCB) | Schematic has **2× CSI** + scope tap (**TP1**) from one **`TRIG_PIO`**; the fourth buffer output (**U2** **1Y**) is free |
| Buffers | **2× 74LVC2G241**; **all 3 units** placed per IC; **pin 1 → DGND**, **pin 7 → +3V3D**; **pins 4/8** wired |
| 1.8 V divider | **R51/R53** @ **1.5 kΩ** + **R57/R59** @ **1.8 kΩ**; **~1.81 V** on **J1/J2 pin 1**; **no** CSI **100 Ω** |
| Sync LED | If populated: FET on **DGND**; **R6** sized for ≤25 mA; **J3** labeled |
| Panel LEDs | **D_TRIG** on **`TRIG_PIO`**; no **D_FLASH** on the board |
| IR ring | **Not populated** |
| Silkscreen | reference designators of the trigger parts |
| Scope pads | **TP1** / **TP2** on the top layer (Step 2c); **3.0–3.3 V** on **TP1**, fall time &lt;100 ns ([`h07`](h07_power_grounding_safety.md) §10) |
| Series R | **50 Ω** scope path on **U2** only — **no** further series resistor in the CSI path |

---

## Step 9 — PCB layout (short)

- The trigger region is on the **top** side at the short edge of the HAT+ core, between the USB-C connector and the HAT ID EEPROM ([`h02`](h02_custom_hat_shield.md) §7.4); the RP2350A is on the bottom side.
- **TP1** / **TP2** are on the top layer, 11.8 mm apart (Step 2c).
- No digital trigger pour under **ADS1299** / **`AGND`** region.
- Place **R51/R53** and **R57/R59** between **U1** and **J1/J2** (short **`TRIG_CSI1/2`** runs).
- **TP3/TP4** on **`TRIG_CSI1/2`** near **J1/J2** for camera-side scope (~1.8 V).
- Buffer decoupling (100 nF + 1 µF) at the VCC pins of **U1** / **U2**.

---

## Suggested capture order

1. Hierarchical **`TRIG_PIO`** (+ **`SYNC_LED_PIO`** if Step 4 populated) + global **`+3V3D`**, **`DGND`**, **`+5V_IN`**
2. **U1** + **U2** — … → **Step 2b** divider (**R51/R53** @ 1.5 kΩ, **R57/R59** @ 1.8 kΩ) → **R4** → **J1**, **J2**, **TP1**, **TP2** …
3. *(Optional)* **J4** on **`TRIG_BNC`** (parallel to **TP1**)
4. *(Optional)* **U3** opto + **R3**, **R7** → **`TRIG_USB`**
5. *(Optional)* **Q1** + **R5**, **R6** → **J3** + **D_FLASH** / **R9**
6. **TP3/TP4** (`TP_CSI1`, `TP_CSI2`) and **TP5** (`TP_TRIG_PIO1`) for camera-side and PIO-side scope
7. Assign footprints → **ERC** → layout

---

## Common mistakes

| Mistake | Fix |
|---------|-----|
| Both **OE** pins to GND | **2OE** (pin 7) must go **HIGH** (**+3V3D**) |
| Only **unit A** placed for buffer | Add **units B + C** (**`A`**) — need **all 8** pins |
| Shared series R for two outputs | One **R** per **1Y** / **2Y** (pins 6 / 3) |
| **1.8 V** divider off-HAT | Divider on **HAT** (**R51/R53 + R57/R59**); plain 2-wire harness |
| **100 Ω on CSI path** | Remove — **1.5 kΩ** replaces it; do not stack 100 Ω + 1.5 kΩ |
| **3.3 V on XTR+** | IMX296 — **R51/R53 + R57/R59** populated; measure **~1.81 V** on **Jx pin 1** |
| **SN74LVC2G17** substituted | Use **74LVC2G241** only |
| Sync LED on **AGND** | **DGND** + **`+5V_IN`** |
| **D_FLASH** on FET gate only | Tap **`SYNC_LED_PIO`** net |
| Opto LED ↔ transistor shorted | Four separate **U3** pins per datasheet |
