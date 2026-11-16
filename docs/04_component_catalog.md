<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 04 — Component Catalog

This document is the **only line-item parts list** for JTS-MM: stable IDs (`H-IC-01`, …), MPNs, quantities, unit costs, specifications, and **approved alternatives**.

| Need | Document |
|------|----------|
| Subsystem totals, cost scenarios, JLC panel notes, **ordering checklist** | [`02_components_bom.md`](02_components_bom.md) |
| Schematics, layout rules, signal chains | `hardware/h01`–`h07` |

**Do not duplicate** these tables in `02_components_bom.md` — link here instead.

**Pricing:** EUR, incl. 19 % German VAT where applicable, late 2025 / early 2026 (Mouser DE, Digi-Key DE, Reichelt, BerryBase, Pi-Shop, TME). Treat as **±20 %** until the as-built BOM is frozen.

**Quantity column:** `Qty` = count **per one complete system** unless noted (e.g. PCB panel of 5).

---

## 1. Catalog index

| § | Subsystem | ~EUR (Track B) | Primary doc |
|---|-----------|----------------|-------------|
| [§2](#2-compute--raspberry-pi-5) | Compute (Pi 5 + storage) | 181 | — |
| [§3](#3-custom-hat--pcb--manufacturing) | HAT PCB + fab | 109 | [`hardware/h02_custom_hat_shield.md`](hardware/h02_custom_hat_shield.md) |
| [§4](#4-custom-hat--active-ics) | HAT active ICs | ~57 | [`h03`](hardware/h03_emg_frontend.md), [`h05`](hardware/h05_microcontroller_firmware.md), [`h07`](hardware/h07_power_grounding_safety.md) |
| [§5](#5-custom-hat--connectors-passives-mechanical) | HAT connectors & passives | ~35 (revision 0.0.1, pin headers H-CON-01CY); ~51–103 with another EMG connector option | [`h02`](hardware/h02_custom_hat_shield.md), [`h03`](hardware/h03_emg_frontend.md) |
| [§6](#6-emg-accessories-off-hat) | EMG electrodes & leads | 55–175 | [`h03`](hardware/h03_emg_frontend.md), [`op02`](operations/op02_subject_preparation.md) |
| [§7](#7-imu-subsystem-off-hat) | IMU (2× daughterboards) | 60 | [`hardware/h04_imu_subsystem.md`](hardware/h04_imu_subsystem.md) |
| [§8](#8-cameras) | Cameras + lenses + tripods | 148–287 | [`hardware/h01_camera_subsystem.md`](hardware/h01_camera_subsystem.md) |
| [§9](#9-fiducials--scene-lighting) | AprilTag + lighting | 31–56 | MTA/CRA CAD notes in §9 below |
| [§10](#10-cables-lab--enclosure) | Cables, hub, enclosure | 90 | — |
| [§11](#11-lab-consumables) | Skin prep, adhesive, tape | 40 | [`op02`](operations/op02_subject_preparation.md) |
| [§12](#12-development--bring-up-only) | Dev boards | ~77 | — |
| [§13](#13-optional--not-in-default-build) | Optional kits | 0–65 | — |

---

## 2. Compute — Raspberry Pi 5

| ID | Component | MPN / product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------------|-----|------|-----|-------------------|------|--------------|
| C-01 | Raspberry Pi 5 | SC1111 (8 GB) | 1 | 89 | 89 | 4× Cortex-A76 @ 2.4 GHz; 8 GB LPDDR4X; 2× CSI-2 (4-lane); PCIe 2.0 ×1; dual USB 3 | Host: capture, HDF5, vision, JTS pipeline | **Pi 5 4 GB** (~75 EUR): OK for capture-only; tight if JTS kinematics run live. **Pi 4 8 GB**: viable for Track A + lighter ML; no native dual CSI GS. |
| C-02 | USB-C PSU | SC1112 (27 W) | 1 | 13 | 13 | 5.1 V / 5 A; PD profile for Pi 5 | Stable 5 V under full USB + GPIO load | Generic 5 V/3 A: risks undervoltage / SD corruption. |
| C-03 | Active cooler | SC1148 | 1 | 6 | 6 | PWM fan on SoC | Sustained 60 fps + encoding | Passive heatsink only: thermal throttle under load. |
| C-04 | M.2 HAT+ | SC1206 | 1 | 14 | 14 | PCIe to M.2 2230/2242; pass-through 40-pin header | NVMe below custom HAT | USB SSD: higher latency, not recommended for 60 fps × 2 write. |
| C-05 | NVMe SSD | WD SN740 256 GB / Samsung PM991a | 1 | 35 | 35 | M.2 2230/2242; ≥500 MB/s write | Session HDF5 + FFV1/H.264 | SATA USB enclosure: works but more CPU for USB stack. |
| C-06 | microSD | SanDisk Extreme 32 GB A2 | 1 | 12 | 12 | A2 application class | Pi OS boot only (data on NVMe) | Any A1+ 16 GB+ card. |
| C-07 | Case (optional) | Open frame / official tray | 1 | 12 | 12 | Clears M.2 + HAT stack | Bench mechanical protection | 3D-printed open frame ([§10](#10-cables-lab--enclosure)). |

---

## 3. Custom HAT — PCB & manufacturing

| ID | Component | Spec | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|------|-----|------|-----|-------------------|------|--------------|
| H-PCB-01 | Fabricated PCB | 100 × 56.5 mm (HAT+ core 65 × 56.5 mm plus a 35 mm extension for the EMG front end); 4-layer FR4 1.6 mm; ENIG | 5 panel | 8 | 40 | Top: EMG front end, connectors, LEDs, buttons; bottom: RP2350A, flash, crystal, regulators; min trace 0.15 mm, min via drill 0.3 mm | One board per system (+ spares) | **JLCPCB** (used for the recorded systems, §6.1); alternative **AISLER Beautiful Boards** (EU). |
| H-PCB-02 | Stencil | 0.1 mm laser steel | 1 | 14 | 14 | For hand-paste of ADS1299 | Rework / hand assembly | Order with fab or separate. |
| H-PCB-03 | Factory SMT assembly | Jelly-bean placement only | 5 | 11 | 55 | RP2350A, passives, USB-C, LDOs, etc. | Factory SMT | **JLCPCB** or **AISLER Amazing Assembly** (Push plugin). |
| H-PCB-04 | Hand assembly (local) | ADS1299, header, patient connectors | 1 | — | — | TQFP-64 + THT connectors | Analog-critical / mechanical | **AISLER Simple Supply** or Mouser. |

**Note:** **BNO085 is not on the HAT** — only three **7-pin JST-SH** connectors (`SM07B`, pin 7 = shield → AGND) ([§7](#7-imu-subsystem-off-hat)).

---

## 4. Custom HAT — active ICs

KiCad footprints: [`h09_footprint_master_table.md`](hardware/h09_footprint_master_table.md). Quantities are those of the production BOM of revision 0.0.1 ([`production/bom.csv`](../hardware/jtsmm_hat/production/bom.csv)); a quantity of 0 means the part is not on the board. Sum of the `Sub` column: **55.3 EUR**.

| ID | Component | MPN | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|-----|-----|------|-----|-------------------|------|--------------|
| H-IC-01 | EMG AFE | **ADS1299IPAG** (TQFP-64) | 1 | 38 | 38 | 8 ch; 24-bit Σ-Δ; 250 SPS–16 kSPS; PGA 1–24; SPI; DRDY; internal DRL | 6 EMG + 2 spare ch | **ADS1298**: similar, firmware tweak. **ADS131M08**: 32 kSPS, no integrated DRL. **INA333 + MCP3564**: cheaper, more design risk ([`h03`](hardware/h03_emg_frontend.md) §12). |
| H-IC-02 | Voltage reference | **ADS1299 internal** (4.5 V) | 0 | 0 | 0 | **VREF = 4.5 V**; full datasheet dynamic range; `CONFIG3` internal ref buffer **on**; **≥10 µF + 0.1 µF** on VREFP–VREFN ([`h03`](hardware/h03_emg_frontend.md) §9) | No external ref IC on default BOM | **REF5045AIDR** (4.5 V external) if internal buffer disabled; **REF5025** is **not** used (2.5 V = partial range only) |
| H-IC-03 | Analog LDO (+5VA) | **ADP7142AUJZ-5.0** | 1 | 3 | 3 | Fixed **5.0 V**; KiCad `Regulator_Linear:ADP7142AUJZ-5.0`; **SENSE → +5VA** at load; Vin from boost (~6.5 V) | ADS1299 **AVDD = 5 V** (SBAS499 §7.3) | **ADP7142ARDZ-5.0** (SOIC-8 EP); **TPS7A4700**; **LT1763-5**. |
| H-IC-03b | Analog-rail boost | **TPS61252DSG** | 1 | 1.2 | 1.2 | Per TI Table 3: **C29** 10 µF @VIN, **C30/C31** 2×22 µF @VOUT, **C32** 100 pF, **L1** 1 µH, **R13/R14** 787 kΩ / 180 kΩ (≈6.45 V), **R15** 20 kΩ (ILIM), **R32** 1 MΩ PG↔VOUT; **R33** 15 kΩ PG↔**+5V_IN** + **LED_6V5D_PG1** (Step 8) | Pre-regulator for +5VA | Not TPS61299/TLV61070 (5.5 V max). |
| H-IC-04 | Digital LDO | **TPS73633DBV** | 1 | 1.5 | 1.5 | 3.3 V fixed; KiCad `Regulator_Linear:TPS73633DBV`; **EN→VIN**; **NR→10 nF→GND** | ADS1299 DVDD + RP2350A 3.3 V domain (IOVDD/QSPI_IOVDD/USB_OTP_VDD/ADC_AVDD + core-regulator VREG_VIN) + flash, trigger buffers, I2C mux and IMU connectors | **AP2112**: cheaper, noisier. |
| H-IC-05 | MCU | **RP2350A** (QFN-60, 7×7 mm) | 1 | 1.3 | 1.3 | Dual Cortex-M33 (or Hazard3 RISC-V) @ 150 MHz; 520 KB SRAM (10 banks); 8 KB OTP; 12 PIO SM (3×4); USB 1.1 device; **on-chip SMPS core regulator** (needs H-PAS-08) | Master clock, SPI, triggers | **RP2350B** (QFN-80, 48 GPIO): only if >30 GPIO needed. **RP2354A** (QFN-60, 2 MB in-package flash): drops H-IC-06. **RP2040** (QFN-56, legacy): 264 KB SRAM, internal-LDO core. **STM32F4/F7**: rewrite firmware. **ESP32-S3**: rejected (RF vs EMG). |
| H-IC-06 | QSPI flash | **W25Q128JVP** (WSON-8) | 1 | 1.0 | 1.0 | **128 Mbit (16 MiB)**; 3.3 V quad-SPI; **EP→GND** | UF2 / firmware + headroom | **W25Q16JV** (2 MiB, Pico default); **W25Q32** mid-size SOIC/WSON |
| H-IC-07 | Trigger buffer | **2× SN74LVC2G241DCUR** (`74LVC2G241`) | 2 | 0.5 | 1.0 | **`U_CSI_1`**, **`U_TP_1`**; three outputs used: `U_CSI_1` 1Y / 2Y → CSI **XTR+** via the **H-PAS-10** dividers (**~1.8 V**), `U_TP_1` 2Y → **R55** 50 Ω → **`TP_TRIG1`**; `U_TP_1` 1Y not connected; **1OE→DGND, 2OE→+3V3D**; 100 nF + 1 µF per IC; **VSSOP-8** LCSC **C10430** | Camera **XTR+** drive + scope tap | One 2G241 has only **two** outputs. TSSOP PWR not stocked at JLC. |
| H-IC-08 | Level shifter | **SN74LVC1T45** / **TXS0108** | 0 | 0.6 | 0 | 3.3 V ↔ 1.8/3.3 V | **Not used** — on-HAT **1.5 kΩ / 1.8 kΩ** dividers (**R57–R60**) for IMX296 XTRIG | No footprint on revision 0.0.1; needed only if a camera requires active level translation |
| H-IC-09 | ESD / TVS (inputs) | **TPD4E1B06DRLR** | 4 | 1 | 4 | 4-ch **bidirectional** TVS; **0.7 pF** typ (0.95 pF max); **0.5 nA** max leakage; ±12 kV contact / ±15 kV air; GND-referenced (no VCC pin); fitted package SOT-563 (DRL); **`D5`–`D8`** | Electrode jack protection — precision-analog ESD (**same part as OpenBCI**); 3 arrays (`D5`–`D7`) cover the 12 EMG lines (6 pairs); the fourth (`D8`) protects the DRL line with one channel | **TPD4E05U06DQA** (~0.5 pF/ch, USB3-oriented, GND-ref); **TPD4E004** (~1.6 pF/ch, ±8 kV, VCC-ref — needs VCC→+5VA + 0.1 µF); **SP724AHT**. All must stay <5 pF/ch ([`h03`](hardware/h03_emg_frontend.md) §4). |
| H-IC-10 | CM choke (EMG) | **Murata DLW21SN900SQ2L** | 7 | 0.5 | 3.5 | 90 Ω @ 100 MHz CM (RF) | RF common-mode suppression on patient lines | Alt **DLW21SN121SQ2L** (120 Ω). Omit on poor benches: may worsen 50 Hz pickup slightly. Legacy **801SQ2** invalid. |
| H-IC-11 | Sync LED driver *(not on revision 0.0.1)* | **AO3400** / **2N7002** | 0 | 0.2 | 0 | Low-side switch ~20 mA | Bench **`SYNC_LED`** JST for **Track A** — no footprint on the board; `SYNC_LED_PIO` (GPIO14) ends at the port of the `mcu` sheet | GPIO direct: OK for single 3 mm LED. |
| H-IC-12 | IR ring driver *(not on the HAT)* | — | 0 | 0 | 0 | No footprint; **no HAT 5 V** for rings | Not part of the board (AprilTag default) | Optional kit: rings on **external 5 V** (~250 mA each); see [`h07`](hardware/h07_power_grounding_safety.md) §1.2. |
| H-IC-13 | I2C mux | **PCA9546APW,118** (TSSOP-16; KiCad symbol and value `TCA9546APW`) | 1 | 0.8 | 0.8 | **`U_MUX1`**, fitted; 4-ch switch on **I2C1**: **ch.0 = CRA**, **ch.1 = DP**, ch.2/3 not connected; addr **0x70** (A0–A2→DGND); RESET on GPIO23 with 10 kΩ pull-up (**R47**); LCSC **C149507** | CRA and DP IMU connectors behind one I2C1 ([`h04`](hardware/h04_imu_subsystem.md) §3.4, [`h04_imu_schematic_build_guide.md`](hardware/h04_imu_schematic_build_guide.md) Step 8) | **TCA9546APW** (TI, pin-compatible). The board has no bypass around the mux. |
| H-IC-14 | HAT ID EEPROM | **AT24CS64-SSHM-B** (SOIC-8) | 1 | 1.2 | 1.2 | 64 kbit I²C EEPROM with a factory-programmed serial number; HAT+ ID EEPROM (`U1`) | Raspberry Pi HAT+ identification (settings in `hardware/jtsmm_hat/eeprom/`) | DFN-8 variant AT24CS64-MAHM (footprint `U2` not placed) |
| H-IC-15 | USB ESD | **USBLC6-2SC6** (SOT-23-6) | 1 | 0.05 | 0.05 | 2-line USB ESD array (`U_USB_ESD1`) on D+/D− and VBUS | USB-C protection | — |

Unit prices of H-IC-14, H-IC-15 and H-CON-08 are LCSC list prices for one piece (June 2026, USD read as EUR, rounded; the parts are C615478, C2687116 and C160403 in [`production/bom.csv`](../hardware/jtsmm_hat/production/bom.csv)). Fitted without a catalog ID and not in the totals: the general 0603 resistors (pull-ups, series and LED resistors).

---

## 5. Custom HAT — connectors, passives, mechanical

Footprints and dielectric rules: [`h09_footprint_master_table.md`](hardware/h09_footprint_master_table.md) · [`h09_component_field_conventions.md`](hardware/h09_component_field_conventions.md).

Revision 0.0.1 carries the pin headers **H-CON-01CY** as its EMG connector; the rows **H-CON-01T / R / C / F** are connector options with no footprint on this board. Sum of the `Sub` column for revision 0.0.1 (H-CON-01CY): **35.1 EUR**; with another connector option instead: T 51.1, R 81.1, F 59.1, C 103.1 EUR.

| ID | Component | MPN / spec | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|------------|-----|------|-----|-------------------|------|--------------|
| H-CON-01CY | EMG HAT connector — **bring-up (Option 0)** | **2×** **1×6** + **1×** **1×1** **2.54 mm** pin header (`PinHeader_1x06` + `PinHeader_1x01`) | **13** pins total | 0.15 | **2** | **`J_EMG_N1` / `J_EMG_P1`** (6-pin blocks) + **`J_EMG_DRL1`** (DRL); no shield pin ([`h03`](hardware/h03_emg_frontend.md) §10.2.0) | **First PCB iteration** | Fitted on revision 0.0.1 (the only EMG connector footprints on the board). OpenBCI Cyton-class; alt layout 1×13 or 13×1×1 |
| H-CON-01T | EMG HAT connector — **trunk (Option A)** | **1×** shielded circular trunk — **LEMO `ECG.1B.314.CLN`** (14-pin) / **ODU MEDI-SNAP** size 2 / **Binder 423 `99-..-15-16`** | **1** | 18 | **18** | All 13 conductors in one keyed receptacle; shell → **AGND** ([`h03`](hardware/h03_emg_frontend.md) §10.2.1) | CH7/CH8: no contact | Best single-mate; custom **E-02T** harness |
| H-CON-01R | EMG HAT connector — **regional (Option B, default procurement)** | **3× LEMO `ECG.1B.304.CLN`** (4-pin) + **1× `ECG.1B.302.CLN`** (2-pin DRL) / Binder M16-4 ×3 + M16-2 | **4** | 12 | **48** | Anatomical dual-channel cables + DRL; each shell → **AGND** ([`h03`](hardware/h03_emg_frontend.md) §10.2.2) | CH7/CH8: no contact | **LEMO-compatible** (Witacon): sample before batch |
| H-CON-01C | EMG HAT connector — **per-channel (Option C)** | **7× LEMO `ECG.1B.302.CLN`** (2-pin) / Binder M16-2 ×7 | **7** | 10 | **70** | One receptacle per bipolar channel + DRL ([`h03`](hardware/h03_emg_frontend.md) §10.2.3) | CH7/CH8: no contact | Pairs with stock Witacon 2-snap cables |
| H-CON-01F | *(production alt.)* Touch-proof jack | **DIN 42802** 1.5 mm (Multi-Contact / Lemo compatible) | **13** | 2 | **26** | Shielded lead option; touch-proof at device ([`h03`](hardware/h03_emg_frontend.md) §10.2.4); does **not** fit 56.5 mm edge | CH7/CH8: no jack | Not first-iteration — use **H-CON-01CY** instead |
| H-CON-02 | IMU cable assembly | **JST-SH 7-pin** `SM07B` / plug **SHR-07V-S-B** (HAT) + **JST-GH 7-pin** `GHR-07V-S` (breakout) | 3 on the HAT; cables 2 (+1 opt.) | 1 | 3 | HAT receptacles **`J_IMU_MTA1`**, **`J_IMU_CRA1`**, **`J_IMU_DP1`** (all fitted); cables for MTA + CRA, optional 3rd for **DP** ([`h04`](hardware/h04_imu_subsystem.md) §3.4) | **7×24 AWG** shielded; pin **7** = shield → **AGND** @ HAT ([`h04_imu_cable_assembly.md`](hardware/h04_imu_cable_assembly.md)) | **9-pin** if SPI; legacy **6-pin** lacks shield contact |
| H-CON-03 | Trigger fan-out | **JST-SH 2-pin** `SM02B-SRSS-TB` ×2 on the HAT (**`J_TRIG_CSI1`**, **`J_TRIG_CSI2`**) + harness | 1 set | 1 | 1 | **2×** plain 2-wire to CSI **XTR+/XTR−** (~**1.8 V** set on HAT) | Camera trigger | Divider **R57/R59 + R58/R60** (H-PAS-10) on HAT, not in cable |
| H-CON-04 | Sync LED connector *(not on revision 0.0.1)* | **JST-SH 2-pin** | 0 | 0.5 | 0 | Visible LED in FOV | **Track A** time sync — no footprint on the board | — |
| H-CON-05 | Bench test pads | **Test pads** (copper only) | 14 | 0 | 0 | **14 pads:** power `TP_5VIN1`, `TP_3V3D1`, `TP_5VA1`, `TP_DGND_PWR1`; EMG `TP_ADS_DRDY1`, `TP_VREFP1`, `TP_AGND1`, `TP_BIASOUT1`, `TP_VCAP1`; trigger `TP_TRIG1`, `TP_DGND_TRIG1`, `TP_CSI1`, `TP_CSI2`, `TP_TRIG_PIO1` (**no pads** on the MCU and IMU sheets). Map: [`h02`](hardware/h02_custom_hat_shield.md) §6.1. No BNC / SMB footprint on the board. | Bring-up / validation | **`TestPoint:TestPoint_Pad_D1.0mm`** (1.0 mm round SMD) — copper only, no part cost. The HAT+ template pads `GND1` / `WP1` (bottom side) are not counted. |
| H-CON-06 | USB-C | **GCT USB4105-GF-A** (horizontal, top-mount) | 1 | 1 | 1 | USB 2.0 FS; device to Pi; cable exits board edge | USB-CDC stream | KiCad: `Connector:USB_C_Receptacle_USB2.0_16P` + `Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal`. LCSC **C3020560**. |
| H-CON-07 | Pi header socket | **Samtec HLE-120-02-L-DV-BE-A** (2×20, 2.54 mm, SMD bottom-entry) | 1 | ~17 | ~17 | HAT+ 40-pin pass-through; mates pins from Pi / M.2 HAT+ below | Mechanical + 5 V | **Hand-install**; footprint `Samtec_HLE-120-02-xxx-DV-BE-A_…`; LCSC **C6207203**. Male pins are on the stack **below** — do **not** order **TSW** for **J1**. Mates **0.025″ square** posts; generic 13.3 mm stacking headers with **0.64×0.4 mm** rectangular pins may be **intermittent** — prefer **TSW**/**MTSW** or square-pin stacking headers ([`h02`](hardware/h02_custom_hat_shield.md) §11.5.1). |
| H-CON-08 | SWD connector | **JST SM03B-SRSS-TB** (JST-SH 3-pin, right angle) | 1 | 0.2 | 0.2 | `J_SWD1`: SWCLK, SWDIO, GND for the Raspberry Pi Debug Probe | Flashing and debugging the RP2350 | — |
| H-MECH-01 | Shield can *(optional)* | **Default:** Würth WE-SHC frame **`36103305S`** + cover **`36003300S`** (30×30×3 mm SMT). **Compact:** **`36103255S`** + **`36003250S`** (25×25). **JLC alt:** TE **`2118717-2`** + **`2118717-4`** (LCSC C5167708 + C28601796). Layout: [`h10` §5.4](hardware/h10_kicad_pcb_layout_guide.md#54-optional-emi-shield-can-h-mech-01) | 0 | 3 | 0 | EMI cage over `U_ADS1` + input filters; **no footprint on revision 0.0.1** | <1 µVrms margin | Omit: slightly higher pickup; not unsafe |
| H-PAS-01 | 12 MHz crystal (Pi default) | **Abracon ABM8-272-T3** (±30 ppm, **C_L 10 pF**) | 1 | 0.3 | 0.3 | **15 pF** C0G load caps (**C41/C42**); **1 kΩ** **R53** in series on **XOUT**; footprint `Crystal_SMD_3225-4Pin_3.2x2.5mm`; LCSC **C20625731** — per *Hardware design with RP2350* §4.1 / Pico 2 | RP2350A XOSC / USB boot | **ECS-120-8-33B-CTN-TR** (8 pF CL → **10–12 pF** caps, retune **R53**) if 272-T3 unavailable |
| H-PAS-01-UPG | 12 MHz crystal (upgrade, *not on revision 0.0.1*) | **Abracon ABM8-12.000MHZ-12-B1U-T** (±10 ppm, **C_L 12 pF**) | 0 | 0.6 | 0 | No symbols or footprints on the sheet; would replace **Y1** with **18 pF** C0G load caps and **no** series resistor; LCSC **C1986854**; only instead of the **H-PAS-01** tier ([`h05`](hardware/h05_mcu_schematic_build_guide.md) §1.3) | Stricter ppm MCU clock | Do not populate **both** crystal tiers; never pair **272-T3** with **18 pF** or **12-B1U-T** with **15 pF** |
| H-PAS-02 | EMG input network | 100 Ω/leg + **4.7 nF C0G differential `CFilt`/pair**: 3× Bourns **CAY16-1000F4LF** (`RN1`–`RN3`, 4×100 Ω) + **R5** 100 Ω (DRL) + 7× **GRM1885C1H472JA01D** (`C3`–`C8` differential, `C2` on the DRL path to AGND) | 6 ch + DRL → 3 arrays + 1 R + 7 C | — | 4 | Per [`h03`](hardware/h03_emg_frontend.md) §4 / SBAS499 §10.2.2: **differential** CFilt, not per-leg-to-ground; no common-mode capacitor footprints on the board; CM choke is `H-IC-10` (1/pair) | Anti-alias + CMRR | Resistor arrays for P/N matching ([`h03`](hardware/h03_emg_frontend.md) §4.1); discrete 100 Ω ±1 % per leg also possible. Keep CFilt **C0G** — **not** X7R. Topology fixed by ADS1299 DS. |
| H-PAS-03 | DRL + bias network | `R5`/`R6` 100 Ω + 220 kΩ; `C2` 4.7 nF C0G; `R8` 1 MΩ + `C78` 1.5 nF C0G (BIASOUT–BIASINV) | 1 set | — | — | SBAS499 Fig. 33/34 per [`h03`](hardware/h03_emg_frontend.md) §5 — no BIASOUT shunt cap | Patient safety margin | Lower R: more bias current. |
| H-PAS-03B | AGND/DGND star tie | **`NT_GND_TIE1`** — `Device:NetTie_2` / `NetTie:NetTie-2_SMD_Pad0.5mm` | 1 | — | — | PCB copper bridge at `U_ADS1` pads 32/31; **not orderable** — exclude from assembly BOM | Single ground-domain choke | No 0 Ω resistor substitute |
| H-PAS-04 | Decoupling | **100 nF:** **GRM155R71C104KA88D** **0402** (LCSC **C71629**) — **28×** total ([`h09`](hardware/h09_footprint_master_table.md) § 100 nF capacitors). **1 µF** + **10 µF** per IC **0603/0805**; **ADS1299:** 1 µF/AVDD+DVDD pin, **VCAP1 = 100 µF** (C0G/NPO or tantalum), VCAP2–4 + VREFP bulk per SBAS499 §11 | bag (+1× 100 µF) | 4 | 4 | As close to the package as possible | PDN | VCAP1 non-ferroelectric; do not shrink µF bulk to 0402. |
| H-PAS-05 | Panel LEDs + resistors | **0603** LED (mostly **Lite-On LTST-C190** series) + **AO3400A** (`Q1`, CAP switch) | 12 + 1 FET | 0.05 | 0.65 | **mcu:** **LTST-C190KGKT** (HB), **LTST-C190TBKT** (CAP @ **+5V_IN** + **Q1**/**GPIO24**), **LTST-C193KRKT-5A** (fault), **NCD0603Y1** (EMG_LO, yellow); **imu:** **LTST-C190KGKT** ×3 (MTA, CRA, DP); **trigger_io:** **LTST-C190KFKT** (TRIG amber); **power:** **LTST-C190KGKT** ×4 ([`h07`](hardware/h07_power_schematic_build_guide.md) Step 8) | Bench status | **470 Ω** @ **3.3 V** (**RC0603FR-07470RL**); **470 Ω** @ **+5V_IN** (CAP + boost PG LED **R34**); **15 kΩ** boost PG pull-up **R33** → **+5V_IN** (**RC0603FR-0715KL**); **2.2 kΩ** on **`power`** 5V/5VA greens (**RC0603FR-072K2L**); gate **R56** **100 Ω**, **R54** **100 kΩ** |
| H-PAS-06 | Buttons | Tactile **RS-282G05A3-SMRT** (footprint `SW_SPST_CK_RS282G05A3`) | 4 | 0.3 | 1.2 | **`SW_RESET1`**, **`SW_BOOTSEL1`**, **`SW_USER1`**, **`SW_USER2`** | UF2 + bench control | SWD-only build: omit BOOTSEL |
| H-PAS-07 | *(alias)* | Same as the H-PAS-05 row on `trigger_io` | — | — | — | **`LED_TRIG1`** (counted in H-PAS-05) | — | — |
| H-PAS-08 | MCU core regulator + RP2350 decoupling | **L2** 3.3 µH (**AOTA-B201610S3R3-101-T**) + **4.7 µF 0402** **GRM155R61E475ME15D**: **C45** (CIN at VREG_VIN), **C66** (COUT at the inductor) + **4.7 µF 0603** **GRM188R61A475KE15D**: **C46** (DVDD), **C40** (VREG_AVDD, with **R39** 33 Ω from +3V3D); **100 nF 0402** **GRM155R71C104KA88D**: **9×** at the **+3V3D** pads (**C47, C48, C50, C51, C67–C71**), **2×** on **DVDD** (**C64, C65**), **C43** at the flash, **C49** on USB **VBUS** (at `U_USB_ESD1`), **C52** on **+5V_IN** (at `LED_USER1`) | **L2** + 18 caps | — | 0.6 | MCU capacitors on **B.Cu** around `U_MCU1` per [`h05`](hardware/h05_mcu_schematic_build_guide.md) §1.1.2 (**C49**, **C52** on F.Cu); bulk **C39** 10 µF **0805** | RP2350A **1.1 V** SMPS + **+3V3D** PDN | Figure 21 external **1.1 V LDO**; all 100 nF → **GRM155** 0402 per [`h09`](hardware/h09_footprint_master_table.md) |
| H-PAS-09 | User RGB LED (addressable) | **WS2812B-2020** (Worldsemi) | 1 | 0.15 | 0.15 | **2.0×2.0 mm** PLCC4 (`LED_SMD:LED_WS2812B-2020_PLCC4_2.0x2.0mm`); **VDD → +5V_IN** (3.3–5.5 V); **DATA** on **GPIO4** @ 3.3 V via **R25** (220 Ω); **100 nF** on VDD; **DOUT** NC; **GRB** | Programmable debug colour (ESP32-style) | **SK6812-2020** (same footprint class); **SK6812MINI-E** reverse-mount; discrete 0603 RGB ×3 |
| H-PAS-10 | IMX296 XTRIG divider | **RC0603FR-071K5L** (1.5 kΩ) + **RC0603FR-071K8L** (1.8 kΩ) | 4 | — | — | **`trigger_io`:** **R57/R59** 1.5 kΩ (series from **`U_CSI_1`**) + **R58/R60** 1.8 kΩ (to **DGND**); Pi GS doc → **~1.81 V** at pin 1 of **`J_TRIG_CSI1`** / **`J_TRIG_CSI2`**; **R55** **50 Ω** scope tap on **`U_TP_1`** only (**~3.3 V**) | 3.3 V buffer → 1.8 V **XTR+** | No further series resistor in the CSI path ([`h06`](hardware/h06_trigger_io_schematic_build_guide.md) Step 2b) |

---

## 6. EMG accessories (off-HAT)

> **Note on IDs:** `E-0x` are **internal catalog codes**, not product names. Pick **one** lead-set row (+ matching **H-CON-01\***) per lifecycle phase ([`h03`](hardware/h03_emg_frontend.md) §10.2). **Typical totals (incl. E-01 electrodes):** **0 (bring-up)** ≈ **55 EUR**; **B** ≈ 193 EUR; **A** ≈ 82 EUR; **C** ≈ 212 EUR; **F** ≈ 84 EUR.

| ID | Component | MPN / product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------------|-----|------|-----|-------------------|------|--------------|
| E-01 | Disposable electrode | **Ambu BlueSensor P-00-S** (used; planned: BlueSensor N) | 100 | 0.4 | 40 | Ag/AgCl; pre-gelled | Per-session contact | **Skintact F-301**: longer sessions. |
| E-02CY | Lead set — **bring-up (Option 0)** | [OpenBCI EMG/ECG snap cables](https://shop.openbci.com/collections/frontpage/products/emg-ecg-snap-electrode-cables) — unshielded, snap + **Dupont female** | **13** | 6 | **78** | **10-pack + 3 singles** (or 2×10 − 7 spares); mates **H-CON-01CY** ([`h03`](hardware/h03_emg_frontend.md) §10.2.0) | **First PCB test** | DIY: snap + wire + Dupont crimp |
| E-02T | Lead set — **trunk (Option A)** | **6+1 shielded trunk** (6 individually-shielded pairs + DRL), 1.5 m; HAT plug mates **H-CON-01T**; collar spider; electrode **snap / DIN 42802** | 1 set | 28 | 28 | Custom bionen / OT-Bioelettronica class ([`h03`](hardware/h03_emg_frontend.md) §10.2.1) | Subject to HAT | DIY: Belden 9841 + LEMO 314 plug |
| E-02TP | Trunk cable plug | **LEMO `FGG.1B.314.CLAD`** / ODU MEDI-SNAP / **Binder `99-5455-15-16`** | 1 | 14 | 14 | Mate to **H-CON-01T**; shield → shell | — | — |
| E-02R | Lead set — **regional (Option B)** | **3×** dual-bipolar cables (2 indiv. shielded pairs each, Y-fork) + **1×** DRL; **~1.2 m** collar→HAT; **4×** LEMO 1B.304 / 1B.302 plugs; **4×** snap per dual cable | 1 set | 105 | **105** | Anatomical grouping: dig L+R, face L, face R ([`h03`](hardware/h03_emg_frontend.md) §10.2.2) | **Default procurement** | Witacon/Heymed custom quote; DIY: Belden 8723 |
| E-02C | Lead set — **per-channel (Option C)** | **6×** LEMO 1B → 2-snap bipolar (~150 cm) + **1×** DRL snap cable | 1 set | 132 | **132** | Witacon [2-snap cable](https://www.witacon.com/Circular-Connector-Cable/LEMO-1B-Connector-To-Electrode-Snap-Cable/) ×6 + DRL ([`h03`](hardware/h03_emg_frontend.md) §10.2.3) | Fastest stock path | Bundle at collar in velcro sleeve |
| E-02F | Lead set — **DIN (Option F)** | **13×** shielded wires or bionen **0044.xxx** snap + DIN TP | 1 set | 18 | 18 | Fan-out to **H-CON-01F** ([`h03`](hardware/h03_emg_frontend.md) §10.2.4) | Shielded + touch-proof device end | bionen singles by color |
| E-03 | Extension cable | TRS ↔ DIN 42802 1.5 mm | 6 | 2 | 12 | Shielded; **+1.5 m**/ch — [`h03`](hardware/h03_emg_frontend.md) §10.1 | **H-CON-01F** only | Shorter runs: less flex fatigue |

---

### 6.1 As used in the recordings (both studies, record of the authors 2026-11-06)

The planning rows above list options; the dataset and validation recordings used:

| Item | Product (manufacturer, MPN) | Notes |
|---|---|---|
| Electrodes | **Ambu BlueSensor P-00-S** (Ambu; bag of 50) | Ag/AgCl, pre-gelled, snap; single use |
| EMG leads (3 cables) | **Belden 8723** (2 × 2 × 22 AWG, Cu, PVC, grey) | shielded, shield to **AGND** at the HAT; one cable per electrode pair group |
| DRL / bias lead (1 cable) | **HELU SiF 0.25 mm²** (MPN 23205; single core, silicone, white) | unshielded single conductor |
| IMU cables (2) | **LAPP UNITRONIC LiYCY (TP) 3 × 2 × 0.14 mm²** (MPN 0035141, PVC, grey) | shielded, twisted pairs, shield to **AGND** (HAT pin 7) |
| HAT PCB (H-PCB-01) | **JLCPCB** | the recorded systems' boards were fabricated by JLCPCB; the AISLER route below is an alternative, not used |

Cables bought from TME (Germany). The bring-up snap leads (E-02CY, unshielded) were not used
in the reported recordings.

## 7. IMU subsystem (off-HAT)

| ID | Component | MPN / product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------------|-----|------|-----|-------------------|------|--------------|
| I-01 | IMU breakout | **Adafruit 4754** / **SparkFun BOB-22857** (BNO085) | 2 (+1 opt.) | 22 | 44 (+22) | 9-DoF; BSX fusion; SPI/I2C; game RV @ 200 Hz | MTA + CRA; **optional 3rd on DP** ([`h04`](hardware/h04_imu_subsystem.md) §3.4) | **ICM-42688-P** breakout (~8 EUR): raw only, host fusion ([`h04`](hardware/h04_imu_subsystem.md) §5). **BNO080** module: older, slower. **MPU-9250**: EOL. |
| I-02 | Shielded cable | **7-conductor** twisted **24 AWG**, **1.5 m** | 2 (+1 opt.) | 4 | 8 (+4) | 6 signals + **drain on pin 7**; [`h04`](hardware/h04_imu_subsystem.md) §3.3 | MTA/CRA (DP +1) to HAT | Shorter if routing allows; length budget §3.3.1 |
| I-03 | Crimp kit | **SHR-07V-S-B** + **GHR-07V-S** + **SSH-003T-P0.2-H** contacts | 2 (+1 opt.) | 1 | 2 (+1) | Per [`h04_imu_cable_assembly.md`](hardware/h04_imu_cable_assembly.md) | Pluggable IMU harness | Pre-made cable if available |
| I-04 | Mounting clip | PETG/TPU 3D print | 2 | 2 | 4 | ≤2 g added mass on MTA | Strain relief | Tape-only: pad lift risk. |
| I-05 | *(optional)* Pigtail adapter PCB | 12×10 mm; **BM07B-GHS-TB** + BNO085 pads | 0 | 2 | 0 | Pin 7 **NC** on copper; [`h04_imu_cable_assembly.md`](hardware/h04_imu_cable_assembly.md) | Easier harness build | Hand-solder GH pigtail to breakout |

---

## 8. Cameras

### 8.1 Track B (recommended)

| ID | Component | MPN / product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------------|-----|------|-----|-------------------|------|--------------|
| CAM-B-01 | Pi GS camera | **IMX296** (Pi Global Shutter Camera) | 2 | 55 | 110 | GS; 1456×1088 @ 60 fps; **XTR+ / XTR−** fast trigger | Stereo pose | [§8.3](#83-camera-alternatives-matrix); Waveshare/InnoMaker alts |
| CAM-B-01b | GS camera (alt) | **Waveshare** IMX296 Global Camera (A) | 0–2 | ~55 | — | Same sensor; **XTR±** + XVS/XHS/MAS on header (use XTR± only) | Alt. to CAM-B-01 | Berrybase / Waveshare |
| CAM-B-01c | GS camera (alt) | **InnoMaker** IMX296 MIPI | 0–2 | ~45 | — | **Trig+ / Trig−**; verify Pi OS + ribbon | Budget alt. | Amazon / InnoMaker |
| CAM-B-02 | Lens | M12 visible **4–6 mm** C/CS | 2 | 22 | 44 | ~46° HFOV @ 80 cm; no IR-pass coating | Sharp AprilTag edges | 8 mm: narrower FOV, more distance. |
| CAM-B-03 | CSI ribbon | 22-pin FFC, 30 cm | 2 | 4 | 8 | ≤30 cm for signal integrity | Pi ↔ camera | Longer ribbon: CSI errors. |
| CAM-B-04 | Tripod | Manfrotto Pixi / equiv. | 2 | 15 | 30 | Stable aim | Extrinsics stability | Gorilla pod: OK for bench. |
| CAM-B-05 | 3rd camera (optional) | **Arducam UC-599** (OV9281 USB3) | 1 | 95 | 95 | GS; USB3; HW trigger via HAT | Occlusion robustness | Fixed stereo only: save 95 EUR. |

### 8.2 Track A (budget)

| ID | Component | MPN / product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------------|-----|------|-----|-------------------|------|--------------|
| CAM-A-01 | Webcam | **Logitech C920/C920s** | 2 | 55 | 110 | Rolling shutter; 1080p30; UVC | Cheapest optical | **C922**: similar. |
| CAM-A-02 | USB GS (upgrade) | **Arducam B0298** (OV9281) | 2 | 80 | 160 | GS mono ~100 fps; no HW trigger | Better motion freeze | See matrix. |
| CAM-A-03 | USB adapter | USB-A → USB-C | 2 | 4 | 8 | Pi 5 host ports | Physical fit | Powered hub ([§10](#10-cables-lab--enclosure)). |
| CAM-A-04 | Tripod | Same as CAM-B-04 | 2 | 15 | 30 | — | — | — |

### 8.3 Camera alternatives matrix

| Product | Shutter | HW trigger | ~EUR | Markers | When to choose |
|---------|---------|------------|------|---------|----------------|
| Logitech C920 | Rolling | SW + sync LED | 55 | AprilTag | Minimum budget |
| Arducam B0298 USB | Global | SW + sync LED | 80 | AprilTag | Track A but need GS |
| **Pi GS (IMX296)** | **Global** | **XTR+ / XTR−** | **55** | **AprilTag** | **Default** — best Pi docs |
| Waveshare IMX296 Global (A) | Global | **XTR+ / XTR−** | ~55 | AprilTag | Pre-populated headers |
| InnoMaker IMX296 | Global | **Trig+ / Trig−** | ~45 | AprilTag | Verify cable + support |
| Arducam UC-599 USB3 | Global | HAT TRIG (opto) | 95 | AprilTag | Third view / redundancy |

Full discussion: [`hardware/h01_camera_subsystem.md`](hardware/h01_camera_subsystem.md).

---

## 9. Fiducials & scene lighting

| ID | Component | Spec | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|------|-----|------|-----|-------------------|------|--------------|
| F-01 | PETG filament | Matte black | 1 spool | 20 | 20 | MTA/CRA pads + enclosure | 3D prints | PLA: gloss hurts tags. |
| F-02 | AprilTag sheet | **36h11** printable adhesive | 2 | 3 | 6 | 15–25 mm tags; ≥30 px in image | Default tracking | Classic ArUco: worse occlusion tolerance. |
| F-03 | Cleaning kit | IPA + microfiber | 1 | 5 | 5 | Lens + tag maintenance | — | — |
| F-04 | LED panel (optional) | ~5000 K dimmable softbox | 1 | 25 | 25 | Diffuse visible light | Glare reduction | Ceiling fluorescents only: often uneven. |
| F-05 | Matte tape (optional) | Black gaffer | 1 | 4 | 4 | Cheek glare control | — | — |

**MTA/CRA CAD:** 3× flat tag pads per array; sphere posts **unpopulated** in default build. Dictionary: OpenCV `DICT_APRILTAG_36h11`.

### Optional — retroreflective kit (Qualisys comparison)

| ID | Component | Qty | ~EUR | Notes |
|----|-----------|-----|------|-------|
| O-RETRO-01 | 3M Scotchlite + ⌀6 mm spheres + IR rings + 850 nm filters | 1 kit | 65 | Not on HAT power; not default software |

---

## 10. Cables, lab & enclosure

| ID | Component | Product | Qty | Unit | Sub | Key specifications | Role | Alternatives |
|----|-----------|---------|-----|------|-----|-------------------|------|--------------|
| L-01 | Powered USB hub | Plugable USBC-HUB7 / equiv. | 1 | 40 | 40 | USB 3; isolated preferred | Cameras + peripherals off Pi | Direct to Pi: port limit / noise. |
| L-02 | BNC/SMB cables | Scope leads | 2 | 4 | 8 | Trigger debug | Bring-up | — |
| L-03 | Dupont / headers | Assortment | 1 | 12 | 12 | Bench wiring | Bring-up | — |
| L-04 | 3D-printed enclosure | PETG open frame | 1 | 8 | 8 | Pi + HAT + cable relief | Bench safety | Commercial Pi case: may not fit stack. |
| L-05 | Cable clips / velcro | — | 1 | 12 | 12 | Strain relief | — | — |
| L-06 | Heat-shrink kit | — | 1 | 10 | 10 | Cable repairs | — | — |

---

## 11. Lab consumables

| ID | Component | Product | Qty | Unit | Sub | Role | Alternatives |
|----|-----------|---------|-----|------|-----|------|--------------|
| X-01 | Skin prep | Weaver **Nuprep** | 1 | 12 | 12 | Lower impedance | Fine sandpaper: not recommended. |
| X-02 | Alcohol wipes | — | 1 box | 6 | 6 | Cleaning | — |
| X-03 | Dental adhesive | IBT / medical PETG compatible | 1 | 12 | 12 | MTA bond | Per original JTS protocol. |
| X-04 | Tape | 3M Micropore | 1 | 5 | 5 | Cable / electrode secure | — |
| X-05 | Gloves | Disposable nitrile | 1 box | 5 | 5 | Hygiene | — |

---

## 12. Development & bring-up only

Not required for a production bench after the HAT is validated; essential for the parallel software path.

| ID | Component | MPN | Qty | Unit | Sub | Role | Alternatives |
|----|-----------|-----|-----|------|-----|------|--------------|
| D-01 | **RP2350 Pico 2** | Raspberry Pi Pico 2 | 1 | 6 | 6 | Early firmware on breakouts | HAT RP2350A chip: same binary, different board. Original **Pico (RP2040)** also works for early USB-CDC/I2C bring-up. |
| D-02 | ADS1299 eval / breakout | TI ADS1299EEG-FE or OpenBCI-derived | 1 | 45 | 45 | SPI bring-up before HAT | Wire to Pico directly. |
| D-03 | Debug Probe | **Raspberry Pi Debug Probe** | 1 | 12 | 12 | CMSIS-DAP: **3-wire SWD** (GND, SWCLK, SWDIO) + UART; kit includes JST↔JST and JST↔0.1″ cables — **does not power the target** ([`h08`](hardware/h08_rp2350_flashing.md) §2–§3) | Spare **Pico** as Picoprobe ([`D-01`](#12-development--bring-up-only)); UF2-only (no probe) works once USB is up |
| D-04 | ChArUco board | A2/A3 printed | 1 | 15 | 15 | Camera intrinsics | Printed PDF + foam board: cheaper. |

---

## 13. Optional / not in default build

| ID | Item | When needed | ~EUR |
|----|------|-------------|------|
| O-01 | Medical-grade isolation (ADuM3151 + isolated DC-DC) | Patient safety v2 | +30 per [`h07`](hardware/h07_power_grounding_safety.md) |
| O-02 | RP2354A (2 MB in-package flash) or RP2350B (QFN-80, 48 GPIO) | Drop external QSPI flash / need >30 GPIO | ~+1 |
| O-03 | Force sensors on spare ADS ch 7–8 | Exoskeleton interface | TBD |
| O-04 | Qualisys retroreflective kit | Side-by-side validation | 65 ([O-RETRO-01](#optional--retroreflective-kit-qualisys-comparison) in §9) |

---

## 14. MCU alternatives (summary)

| MCU | Spec highlight | Use when | Trade-off |
|-----|----------------|----------|-----------|
| **RP2350A** (default) | Dual M33 @150 MHz; 520 KB SRAM; 12 PIO SM; FPU/DSP; PIO triggers; USB-CDC; ~1 EUR; QFN-60 | This project | On-chip SMPS core regulator needs a 3.3 µH inductor (H-PAS-08) |
| **RP2350B** | Same as A but QFN-80, 48 GPIO | Need >30 GPIO | Larger footprint; no extra I2C |
| **RP2040** (legacy) | Dual M0+ @133 MHz; 264 KB SRAM; 8 PIO SM; internal-LDO core; QFN-56 | Prior revisions / stock fallback | Half the SRAM; firmware identical except platform |
| **STM32F4/F7** | Mature ST ecosystem | Lab policy mandates ST | Firmware rewrite |
| **Teensy 4.x** | High performance | Rapid Arduino prototype | Module integration on HAT awkward |
| **ESP32-S3** | Wi-Fi/BLE | — | **Rejected:** RF noise on EMG front-end |

Detail: [`hardware/h05_microcontroller_firmware.md`](hardware/h05_microcontroller_firmware.md) §6.

---

## 15. Pre-existing lab assets (not priced)

Assume available at TU Darmstadt; add cost if replicating from scratch:

| Asset | Used for |
|-------|----------|
| Oscilloscope ≥100 MHz, 2 ch | Trigger / DRDY validation |
| Function generator (mV level) | EMG channel test |
| ChArUco / calibration target | Camera intrinsics ([`cv01`](calibration_validation/cv01_camera_calibration.md)) |
| 3D printer (PETG + biocompatible resin) | MTA/CRA/DP |
| Soldering station + hot air | ADS1299 assembly |
| Qualisys Oqus (optional) | Validation ([`cv04`](calibration_validation/cv04_phantom_and_qualisys_validation.md)) |
| Workstation with KiCad 8+ | PCB design |

---

## 16. Revision log

| Rev | Date | Change |
|-----|------|--------|
| 1.1 | 2026-05 | Sole line-item source; procurement moved to `02` v2.0 |
| 1.0 | 2026-05 | Initial catalog; AprilTag default; BNO085 off-HAT only |
