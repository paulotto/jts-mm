<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 02 — Bill of Materials (procurement)

This file is the **procurement and budgeting** view: subsystem totals, cost scenarios, ordering checklist, and fab notes.

**Line-item detail** (MPN, specifications, alternatives, stable IDs like `H-IC-01`) lives in one place only: [`04_component_catalog.md`](04_component_catalog.md). Do not duplicate those tables here.

Prices are EUR including 19 % German VAT (late 2025 / early 2026, DE distributors). Treat as **±20 %** until the as-built BOM is frozen.

---

## 1. Subsystem cost summary (one benchtop system)

| Subsystem | Track A (webcams) | Track B (2× Pi GS, default) |
|-----------|-------------------|-----------------------------|
| Compute (Pi 5 + storage) | 181 | 181 |
| Custom HAT — fab (5-board JLC panel) | 109 | 109 |
| Custom HAT — components (1 populated board) | 92 | 92 |
| EMG accessories (electrodes, leads) | 55–175 | 55–175 |
| IMU (2× BNO085 off-HAT) | 60 | 60 |
| Cameras + lenses + tripods | 148–198 | 192–287 |
| Fiducials + scene lighting (AprilTag) | 31–56 | 31–56 |
| Cables, hub, enclosure | 90 | 90 |
| Lab consumables | 40 | 40 |
| **Total (typical)** | **~820–900** | **~870–990** |

**How this maps to the catalog:** HAT silicon + connectors = catalog [§3–§5](04_component_catalog.md#3-custom-hat--pcb--manufacturing); everything else = matching catalog section in [§1 index](04_component_catalog.md#1-catalog-index).

The totals price every item; with **lab reuse** (no Pi case, shared filament, etc.) a system costs less. Use [§2](#2-cost-scenarios) for explicit builds.

**Benchmarks:**

- OpenBCI Cyton + Daisy (~16 ch): ~2050 EUR  
- OpenBCI Cyton-GS (8 ch): ~930 EUR  
- Qualisys Oqus (3 cam): ≥45 000 EUR  

---

## 2. Cost scenarios

Summed from [`04_component_catalog.md`](04_component_catalog.md) (one system, default AprilTag, no retroreflective kit).

| What you buy | Approx. EUR |
|--------------|-------------|
| **Full system — Track B, 2× Pi GS, minimal lighting** | **860–890** |
| **Full system — Track A, 2× C920** | **~820** |
| **Full system — Track B + LED panel + 3rd camera** | **~990–1000** |
| **One populated HAT** (1/5 JLC panel + Mouser parts for one board) | **~114** |
| **One populated HAT** (single-board JLC + Mouser; no panel economy) | **~161** |
| **JLC fab + SMT only** (one board’s share of 5-panel order) | **~22** |
| Dev bring-up only (Pico + ADS1299 EVM + debug probe) | +77 |
| Optional Qualisys retroreflective kit | +65 |

**HAT stack detail (one board):** JLC panel share 22 + catalog [§4–§5](04_component_catalog.md#4-custom-hat--active-ics) parts ~92 (§4: 56.6 + §5 with the pin headers H-CON-01CY: 35.1) ≈ **114 EUR** (ADS1299 hand-soldered locally; not in JLC SMT quote).

---

## 3. Procurement index → catalog

| Order from | Catalog § | ~EUR (1 system) | Primary sources |
|------------|-----------|-----------------|-----------------|
| Pi + storage | [§2](04_component_catalog.md#2-compute--raspberry-pi-5) | 181 | BerryBase, Pi-Shop, Reichelt |
| HAT PCB + SMT | [§3](04_component_catalog.md#3-custom-hat--pcb--manufacturing) | 109 (5× panel) | JLCPCB (used for the recorded systems) **or** [AISLER](https://aisler.net) Beautiful Boards + Amazing Assembly |
| HAT ICs + connectors | [§4](04_component_catalog.md#4-custom-hat--active-ics) + [§5](04_component_catalog.md#5-custom-hat--connectors-passives-mechanical) | 92 | Mouser / Digi-Key |
| EMG leads + electrodes | [§6](04_component_catalog.md#6-emg-accessories-off-hat) | 55–175 | **Bring-up ~55 EUR** (E-01 + E-02CY); production **B** ≈ 193 EUR |
| IMU breakouts + cables | [§7](04_component_catalog.md#7-imu-subsystem-off-hat) | 60 | Mouser (Adafruit 4754) |
| Cameras Track A | [§8.2](04_component_catalog.md#82-track-a-budget) | 148–198 | Amazon, Arducam |
| Cameras Track B | [§8.1](04_component_catalog.md#81-track-b-recommended) | 192–287 | Pi-Shop, BerryBase |
| AprilTag + lighting | [§9](04_component_catalog.md#9-fiducials--scene-lighting) | 31–56 | print shop, Amazon |
| Bench cables / hub | [§10](04_component_catalog.md#10-cables-lab--enclosure) | 90 | local / Plugable |
| Consumables | [§11](04_component_catalog.md#11-lab-consumables) | 40 | pharmacy, Weaver |
| Development / bring-up only | [§12](04_component_catalog.md#12-development--bring-up-only) | ~77 | Pi-Shop, Mouser |

---

## 4. Procurement notes (not in the catalog tables)

### 4.1 Long-lead / schedule-critical

- **ADS1299IPAG** — often allocated; try Mouser, Digi-Key, Arrow, TI Store.  
- **Pi Global Shutter cameras** — stock varies in EU (Pi-Shop, BerryBase, Pimoroni).  
- **JLCPCB** — reserve an express slot early; **do not submit** Gerbers before the schematic review. Panel of **5 boards** (~109 EUR) is cheaper per board than a one-off.
- **AISLER** (EU) — alternative fab (not used for the recorded systems; their boards came from JLCPCB) + **Amazing Assembly** (SMT) + **Simple Supply** (hand parts kit). Use KiCad **AISLER Push** (`https://packages.aisler.net`).

### 4.2 HAT assembly model (default)

- **Factory SMT (JLC or AISLER):** jelly-bean parts (RP2350A + 3.3 µH core-regulator inductor, passives, USB-C, LDOs, trigger buffers, I2C mux).
- **Hand local / Simple Supply:** ADS1299 (TQFP-64), **Pi stacking header**, **EMG patient connector(s)**; **do not** upload hand lines to JLC assembly.
- **Off-HAT:** BNO085 on MTA/CRA daughterboards — **not** on the HAT PCB.  
- **Not on the HAT:** IR ring drivers (no footprints; AprilTag default); the optional retro kit uses **external 5 V** ([`hardware/h07_power_grounding_safety.md`](hardware/h07_power_grounding_safety.md) §1).

### 4.3 Optical default (AprilTag)

- No IR-pass lenses, IR rings, or Scotchlite in the default BOM.  
- Diffuse **visible** lighting + locked exposure matter more than extra cameras.  
- Fiducial CAD, tag size, optional retro kit: [`04_component_catalog.md` §9](04_component_catalog.md#9-fiducials--scene-lighting).

### 4.4 Alternatives

Per-part alternatives are in the catalog (last column). Subsystem trade-offs:

| Topic | Doc |
|-------|-----|
| ADS1299 / EMG front-end | [`hardware/h03_emg_frontend.md`](hardware/h03_emg_frontend.md) §12 |
| IMU | [`hardware/h04_imu_subsystem.md`](hardware/h04_imu_subsystem.md) §5 |
| MCU | [`hardware/h05_microcontroller_firmware.md`](hardware/h05_microcontroller_firmware.md) §6; catalog [§14](04_component_catalog.md#14-mcu-alternatives-summary) |
| Cameras | [`hardware/h01_camera_subsystem.md`](hardware/h01_camera_subsystem.md) §6; catalog [§8.3](04_component_catalog.md#83-camera-alternatives-matrix) |

---

## 5. Pre-existing lab assets (not priced)

Assume available at TU Darmstadt; add cost if replicating from scratch:

- Oscilloscope ≥100 MHz (sync / DRDY)  
- Function generator, mV level (EMG bench test)  
- ChArUco board (intrinsics) — or catalog [D-04](04_component_catalog.md#12-development--bring-up-only)  
- 3D printer (MTA/CRA/DP, enclosure)  
- Soldering station + hot air (ADS1299)  
- Qualisys Oqus (validation only)  
- KiCad 8+ workstation  

---

## 6. Ordering checklist

Items to order:

- [ ] ADS1299IPAG  
- [ ] Pi Global Shutter cameras (Track B) or chosen Track A cameras — see catalog [§8](04_component_catalog.md#8-cameras)  
- [ ] **Bring-up:** **H-CON-01CY** (pin headers) + **E-02CY** (OpenBCI snap cables) + **E-01** electrodes — first PCB
- [ ] **Production (later):** **H-CON-01R** + **E-02R** (or **T/C** per [`h03`](hardware/h03_emg_frontend.md) §10.2)  
- [ ] 2× BNO085 breakouts (Adafruit / SparkFun)  
- [ ] JLCPCB: panel + stencil + SMT  
- [ ] Mouser bag: HAT §4–§5 line items for **one** board (+ spare passives optional)  
- [ ] AprilTag sheets + PETG — catalog [§9](04_component_catalog.md#9-fiducials--scene-lighting)  

All other lines are usually next-day from Mouser / Reichelt / Amazon DE.

---

## 7. Revision log

| Rev | Date | Change |
|-----|------|--------|
| 2.0 | 2026-05 | Slim procurement doc; line items only in `04_component_catalog.md` |
| 1.0 | 2026-05 | Initial combined BOM + AprilTag default |
