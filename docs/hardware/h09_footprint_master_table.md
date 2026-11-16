<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h09 — Footprint master table (JTS-MM HAT+)

Cross-sheet index of the **footprints** of the JTS-MM HAT+ schematic (`hardware/jtsmm_hat/`, revision 0.0.1) and of their **KiCad availability**: **Easy** = footprint from the stock KiCad libraries, **Project lib** = footprint from `hardware/jtsmm_hat/libs/`.

**EMG connector (revision 0.0.1):** **H-CON-01CY** — 13 pins on 2.54 mm pin headers ([`04_component_catalog.md`](../04_component_catalog.md) §5).

---

## Summary

| Metric | Value |
|--------|------:|
| Components on the 6 sheets (reference designators, excl. power symbols) | 217 |
| Without footprint | 0 |
| Footprint differs between schematic and PCB | 0 |
| Not placed on the PCB | `U2` (alternative EEPROM package, DNP) |
| 100 nF caps | **28×** **0402** + `C1` (**0603**, `hat_plus`) — see [100 nF capacitors](#100-nf-capacitors-0402) |

---

## Footprints that the stock defaults get wrong

A default footprint is wrong for these parts; the schematic carries the footprint in the fourth column.

| Refs | Sheet | Role | Footprint | KiCad | Note |
|------|-------|------|-----------|-------|------|
| **C9** | emg | VCAP1 100 µF | `Capacitor_Tantalum_SMD:CP_EIA-3528-21_Kemet-B` | Easy | A 0603 land cannot hold 100 µF; tantalum |
| **FL1–FL7** | emg | Murata CM choke | `jtsmm:FIL_Murata_DLW21SN900SQ2L` | **Project lib** (`jtsmm.pretty`) | MPN **DLW21SN900SQ2L** |
| **RN1–RN3** | emg | CAY16 R-array | `jtsmm:Bourns_RESCAF80P320X160X60-8N` | **Project lib** | Land of the CAY16-1000F4LF datasheet |
| **L2** | mcu | RP2350 3.3 µH | **`jtsmm:AOTAB201610S3R3101T`** | **Project lib** (RP2350 minimal) | **AOTA-B201610S3R3-101-T** |
| **L1** | power | TPS61252 1 µH | **`jtsmm:AOTAB201610S3R3101T`** | **Project lib** | **AOTA-B201610S1R0MT** (LCSC **C25914177**) |

---

## Easy — stock KiCad footprints with a size choice

| Refs | Sheet | Footprint |
|------|-------|-----------|
| **C39** | mcu | 10 µF bulk: **0805** |
| **C24,C26,C29,C34,C37** | power | 10 µF: **0805** |
| **C32** | power | 100 pF CFF: **0603**, **C0G** in the Dielectric field |
| **J_EMG_N1, J_EMG_P1** | emg | **2× 1×6** pin header (H-CON-01CY bring-up), channel N/P blocks: `PinHeader_1x06_P2.54mm_Vertical` |
| **J_EMG_DRL1** | emg | **1×1** DRL pin, with the two 1×6 blocks: `PinHeader_1x01_P2.54mm_Vertical` |

---

## Manual — custom footprints (`libs/jtsmm.pretty`)

| Footprint name | Part | Source | Status |
|----------------|------|--------|--------|
| `FIL_Murata_DLW21SN900SQ2L` | H-IC-10 CM choke | SnapEDA / Murata DLW21SN900SQ2L | **In `jtsmm.pretty`** |
| `TPD4E1B06DRLR_SOT50P160X60-6N` | H-IC-09 ESD array | SnapEDA SOT-563 land for **TPD4E1B06DRLR** | **In `jtsmm.pretty`** |
| `Bourns_RESCAF80P320X160X60-8N` | H-PAS-02 CAY16 array | SnapEDA Bourns CAY16 land | **In `jtsmm.pretty`** |
| `AOTAB201610S3R3101T` | H-PAS-08 L2 (3.3µH) + L1 boost (1.0µH) shared land | Abracon B201610 | **In `jtsmm.pretty`** — polarity dot |
| `TDK_CKCA43_C0G_4x470pF` | H-PAS-02 optional CM array | TDK CKCA43 | Not in the library — optional part, not on the schematic |
| `Tantalum_CaseD_100uF` | H-PAS-04 VCAP1 alt | Vendor MPN | Not in the library — alternative to Kemet-B if 100 µF needs Case-D |
| `DIN42802_Jack_1.5mm` | H-CON-01F | Panel jack drawing | Not in the library — H-CON-01F is not used on revision 0.0.1 |

Samtec HAT header already in [`Connector_Samtec_Reverse.pretty`](../../hardware/jtsmm_hat/libs/Connector_Samtec_Reverse.pretty/).

---

## Spec-only — footprint OK, spec in the symbol fields

| Refs | Sheet | Required spec |
|------|-------|---------------|
| **C2–C8** (4.7 nF) | emg | **C0G** in Description / Dielectric field |
| **C41,C42** (15 pF) | mcu | **C0G** crystal load caps — **ABM8-272-T3** (default); 18 pF for the upgrade crystal (not on the schematic) |
| **R53** (1 kΩ) | mcu | XOUT series damping — Pi *Hardware design with RP2350* §4 |
| **C9** (100 µF) | emg | **C0G/tantalum** + role `VCAP1` |

See [`h09_component_field_conventions.md`](h09_component_field_conventions.md).

---

## 100 nF capacitors (0402)

One jellybean MPN for the board's 100 nF caps — **GRM155R71C104KA88D** (LCSC **C71629**), footprint `Capacitor_SMD:C_0402_1005Metric` — **28×**:

| Refs | Sheet |
|------|-------|
| **C47,C48,C50,C51,C64,C65,C67,C68,C69,C70,C71** | mcu (RP2350 pad cluster) |
| **C12,C13,C14,C22** | emg |
| **C25,C27,C33,C38** | power |
| **C43,C49,C52** | mcu (flash VCC, USB VBUS at `U_USB_ESD1`, `+5V_IN`) |
| **C53,C55,C57,C59** | imu_connectors |
| **C60,C62** | trigger_io |

`C1` (bypass of the ID EEPROM on `hat_plus`) is the only **0603** 100 nF (GRM188R71C104KA01D, LCSC C77055).

The 0402 size is mandatory only for the RP2350 pad-adjacent cluster (Pi HW guide); for the other bus HF bypass caps it saves ~10–15 mm² and unifies the BOM line, with **no** electrical difference.

**Not 0402:** the **1 µF+** caps (except the 4.7 µF **C45/C66** at the RP2350 regulator), **C0G 4.7 nF** CFilt, crystal load caps, tantalum **C9**.

---

## Per-sheet footprint quick reference

### `hat_plus`

| Part | Footprint | KiCad |
|------|-----------|-------|
| J1 Pi HLE socket | `Connector_Samtec_Reverse:Samtec_HLE-120-02-xxx-DV-BE-A_2x20_P2.54mm_Horizontal_Reverse` — MPN **HLE-120-02-L-DV-BE-A**, LCSC **C6207203** | Project |
| U1 EEPROM | `Package_SO:SOIC-8_3.9x4.9mm_P1.27mm` | Easy |
| Passives | `R_0603` / `C_0603` | Easy |
| H1–H4 | `MountingHole:MountingHole_3.2mm_M3_DIN965` | Easy |

### `power`

| Part | Footprint | KiCad |
|------|-----------|-------|
| U_BOOST1 TPS61252 | `Package_SON:WSON-8-1EP_2x2mm_P0.5mm_EP0.9x1.6mm` | Easy |
| U_ALDO1 ADP7142 | `Package_TO_SOT_SMD:TSOT-23-5` | Easy |
| **C35,C36** ADP7142 CIN/COUT | `Capacitor_SMD:C_0603_1608Metric` | Easy — **GRM188R61A225KE34D** (LCSC C86013) |
| U_DLDO1 TPS73633 | `Package_TO_SOT_SMD:SOT-23-5` | Easy |
| L1 1 µH | **`jtsmm:AOTAB201610S3R3101T`** | **Project lib** — **AOTA-B201610S1R0MT** |
| Bulk 10 µF | `Capacitor_SMD:C_0805_2012Metric` | Easy |

### `emg`

| Part | Footprint | KiCad |
|------|-----------|-------|
| U_ADS1 | `Package_QFP:TQFP-64_10x10mm_P0.5mm` | Easy |
| NT_GND_TIE1 | `NetTie:NetTie-2_SMD_Pad0.5mm` | Easy — AGND/DGND star tie; PCB feature (no BOM) |
| D5–D8 TPD4E1B06 | `jtsmm:TPD4E1B06DRLR_SOT50P160X60-6N` | **Project lib** |
| FL1–FL7 DLW21 | `jtsmm:FIL_Murata_DLW21SN900SQ2L` | **Project lib** |
| RN1–RN3 CAY16 | `jtsmm:Bourns_RESCAF80P320X160X60-8N` | **Project lib** |
| CFilt 4.7 nF | `Capacitor_SMD:C_0603_1608Metric` + **C0G** spec | Easy |
| **C78** 1.5 nF C0G | `Capacitor_SMD:C_0603_1608Metric` | Easy |
| **R8** 1 MΩ | `Resistor_SMD:R_0603_1608Metric` | Easy |
| Test pads | `TestPoint:TestPoint_Pad_D1.0mm` | Easy — `TP_ADS_DRDY1`, `TP_AGND1`, `TP_VREFP1`, `TP_BIASOUT1`, `TP_VCAP1` |
| C9 VCAP1 | `Capacitor_Tantalum_SMD:CP_EIA-3528-21_Kemet-B` | Easy |
| J_EMG_N1, J_EMG_P1 | `Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical` | Easy |
| J_EMG_DRL1 | `Connector_PinHeader_2.54mm:PinHeader_1x01_P2.54mm_Vertical` | Easy |

### `mcu`

| Part | Footprint | KiCad |
|------|-----------|-------|
| U_MCU1 RP2350A | `Package_DFN_QFN:QFN-60-1EP_7x7mm_P0.4mm_EP3.4x3.4mm` | Easy |
| U_FLASH1 | `Package_SON:WSON-8-1EP_6x5mm_P1.27mm_EP3.4x4.3mm` | Easy |
| L2 3.3 µH | **`jtsmm:AOTAB201610S3R3101T`** | **Project lib** — Abracon AOTA-B201610S3R3-101-T |
| Y1 12 MHz | `Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm` — **ABM8-272-T3** (default) | Easy |
| Upgrade crystal **ABM8-12-B1U-T** | same footprint as Y1 | Easy — not on the schematic |
| J_USB1 | `Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal` (MPN **USB4105-GF-A**) | Easy — 3D model ships with KiCad |
| LED_USER1 | `LED_SMD:LED_WS2812B-2020_PLCC4_2.0x2.0mm` | Easy |

Capacitors at the RP2350A:

| Refs | Role | Footprint | Part |
|------|------|-----------|------|
| **C45** | **VREG CIN 4.7 µF** | **`C_0402_1005Metric`** | **GRM155R61E475ME15D** (C2858031) — Pi HW guide |
| **C66** | **DVDD COUT 4.7 µF** | **`C_0402_1005Metric`** | **GRM155R61E475ME15D** (C2858031) — Pi HW guide |
| **C40,C46** | **4.7 µF** at the regulator (`C40` on VREG_AVDD, `C46` on DVDD) | **`C_0603_1608Metric`** | **GRM188R61A475KE15D** (C71633) |
| **C47,C48,C50,C51,C64,C65,C67–C71** | **RP2350 pad 100 nF** (`+3V3D`; `C64`, `C65` on DVDD) | **`C_0402_1005Metric`** | **11×** at RP2350 pads F/B — [`h05`](h05_mcu_schematic_build_guide.md) §1.1.2 |

The other 100 nF caps: [100 nF capacitors](#100-nf-capacitors-0402).

### `imu_connectors`

| Part | Footprint | KiCad |
|------|-----------|-------|
| J_IMU_* | `Connector_JST:JST_SH_SM07B-SRSS-TB_1x07-1MP_P1.00mm_Horizontal` | Easy |
| U_MUX1 | `Package_SO:TSSOP-16_4.4x5mm_P0.65mm` | Easy |

### `trigger_io`

| Part | Footprint | KiCad |
|------|-----------|-------|
| U_CSI_1, U_TP_1 | `Package_SO:VSSOP-8_2.3x2mm_P0.5mm` | Easy — **SN74LVC2G241DCUR** (LCSC **C10430**) |
| **C61,C63** bulk 1 µF | `Capacitor_SMD:C_0603_1608Metric` | Easy — **GRM188R71E105KA12D** (LCSC C15849) |
| R57, R59 (1.5 kΩ) + R58, R60 (1.8 kΩ) | `Resistor_SMD:R_0603_1608Metric` | Easy |
| J_TRIG_* | `Connector_JST:JST_SH_SM02B-SRSS-TB_1x02-1MP_P1.00mm_Horizontal` | Easy |
| TP_* | `TestPoint:TestPoint_Pad_D1.0mm` | Easy |

---

## Related docs

| Doc | Role |
|-----|------|
| [`h09_component_field_conventions.md`](h09_component_field_conventions.md) | BOM fields, Description template |
| [`04_component_catalog.md`](../04_component_catalog.md) | MPNs and catalog IDs |
| Per-sheet build guides Step 0 | Capture walkthrough (link here for footprints) |
