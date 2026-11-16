<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h09 — Component field conventions (BOM / procurement)

Companion to the footprint master table ([`h09_footprint_master_table.md`](h09_footprint_master_table.md)).

---

## Project custom fields

| Field | Purpose | Example |
|-------|---------|---------|
| **CatalogID** | Link to [`04_component_catalog.md`](../04_component_catalog.md) | `H-PAS-02`, `H-IC-10` |
| **Dielectric** | Capacitor class (Digi-Key / Mouser filter) | `C0G`, `X7R`, `X5R`, `C0G/tantalum` |
| **Tolerance** | R/C tolerance | `1%`, `5%`, `10%` |
| **Voltage** | Capacitor voltage rating | `50V`, `25V`, `10V` |
| **Power** | Resistor power rating | `63mW`, `100mW`, `125mW` |
| **Spec** | **Comprehensive one-liner** for procurement search | `4.7nF 50V C0G 5% 0603` |
| **MPN** | Manufacturer part number (when fixed) | `GRM1885C1H472JA01D` |
| **LCSC Part #** | JLCPCB / LCSC warehouse SKU | `C85980` |
| **Populate** | Build option | `YES`, `DNP`, `ALT` |

### JLCPCB assembly (what you need beyond KiCad)

JLC’s SMT line uses **LCSC part numbers** (`Cxxxxx`) in the uploaded BOM — that is their warehouse SKU, not necessarily the same string as the manufacturer MPN. In practice:

| Deliverable | Source |
|-------------|--------|
| Gerbers + drill | KiCad fab export |
| Centroid / pick-and-place | KiCad → **Fabrication outputs → Footprint positions** |
| BOM with **LCSC** or matchable **MPN** | KiCad BOM export after fields are filled; map MPN → LCSC on [jlcpcb.com](https://jlcpcb.com) part search |

**Basic vs Extended parts:** jelly-bean passives and common ICs (RP2350A, LDOs, USB-C) should be **Basic** parts in JLC’s library for lowest assembly cost. **ADS1299**, Murata DLW21, LEMO connectors, and other specialty lines are often **Extended** (higher fee, longer review) or **not placed by JLC** in the default build ([`02_components_bom.md`](../02_components_bom.md) §4.2).

### AISLER assembly (Amazing Assembly / Simple Supply)

AISLER reads **MPN** from schematic symbols (and PCB after **Update PCB from Schematic**). Supported field names: `MPN`, `mpn`, `Mpn`, `AISLER_MPN` — this project uses **`MPN`** only.

| Deliverable | Source |
|-------------|--------|
| PCB + Gerbers | **AISLER Push for KiCad** (repo `https://packages.aisler.net`) |
| Factory assembly parts | Push → assign matched parts in Amazing Assembly |
| Hand-install parts (header, patient connectors) | **Simple Supply** or Mouser — **not** the factory assembly BOM |

### PCB features vs assembly BOM (test points, mounting holes)

`TestPoint:TestPoint_Pad_D1.0mm` pads are **copper on the fabricated PCB** — not a distributor line item. They appear automatically from Gerbers/layout; they are **not** on the assembly BOM.

---

## Spec field format (comprehensive)

The **Spec** field is the single string you can paste into distributor search when MPN is not yet fixed:

| Component | Spec pattern | Example |
|-----------|--------------|---------|
| Capacitor (C0G signal) | `{value} {voltage} {dielectric} {tolerance} {package}` | `4.7nF 50V C0G 5% 0603` |
| Capacitor (X7R power) | `{value} {voltage} X7R {tolerance} {package}` | `100nF 25V X7R 10% 0402` (`0603` for `C1` on `hat_plus`) |
| Capacitor (100 nF jellybean) | same pattern, **0402** | `100nF 25V X7R 10% 0402` — **GRM155R71C104KA88D** (LCSC **C71629**); **28×** on the board |
| Capacitor (RP2350 +3V3 pad) | same **0402** MPN | refs **C47,C48,C50,C51,C64,C65,C67–C71** — mandatory at pads per Pi HW guide |
| Capacitor (tantalum VCAP) | `{value} {voltage} tantalum {tolerance}` | `100uF 10V tantalum 10%` |
| Resistor | `{value} {power} {tolerance} {package}` | `100 63mW 1% 0603` |
| Resistor array | `{value} {power} {tolerance} {package}` | `100 125mW 1% 1206-array` |

### Default passive ratings (0603 unless noted)

**100 nF:** the 100 nF bypass caps of the `mcu`, `emg`, `power`, `imu_connectors` and `trigger_io` sheets are **GRM155R71C104KA88D**, footprint `C_0402_1005Metric` (**28×**; [`h09_footprint_master_table.md`](h09_footprint_master_table.md) § 100 nF capacitors). **1 µF+** (except the 4.7 µF **C45/C66** at the RP2350 regulator) and **C0G signal** caps are **0603/0805**.

| Type | Voltage | Dielectric | Tolerance | Power |
|------|---------|------------|-----------|-------|
| C0G signal (CFilt, XOSC load) | **50V** | C0G | **5%** | — |
| X7R / X5R bypass | **25V** | X7R / X5R | **10%** | — |
| VCAP1 / bulk pre-reg | **10V** | C0G/tantalum | **10%** | — |
| Resistor 0603 | — | — | **5%** ( **1%** on EMG series R) | **63mW** |
| Resistor 0805 / 1206 array | — | — | per role | **100mW** / **125mW** |

---

## Value vs Description policy

| Field | Content | Rationale |
|-------|---------|-----------|
| **Value** | Numeric / short identity only | `4.7nF`, `100`, `ADS1299IPAG` — stable BOM grouping |
| **Description** | Human-readable role + spec | Assembly review; mirrors Spec with role prefix |
| **Spec** | Distributor search string | No role prefix — pure part spec |
| **Footprint** | Full `Library:Name` | Always include library prefix |

### Description template

```text
{role}: {value} {voltage} {dielectric} {package} ±{tolerance} [{CatalogID}]
```

Examples:

| Part | Description | Spec |
|------|-------------|------|
| EMG CFilt | `CFilt / DRL 4.7 nF: 4.7nF 50V C0G 0603 ±5% [H-PAS-02]` | `4.7nF 50V C0G 5% 0603` |
| VCAP1 | `ADS1299 VCAP1: 100uF 10V C0G/tantalum ±10% [H-PAS-04]` | `100uF 10V tantalum 10%` |
| EMG series R | `EMG input series R: 100 63mW 0603 ±1% [H-PAS-02]` | `100 63mW 1% 0603` |
| DVDD bypass | `HF decouple: 100nF 25V X7R 0603 ±10% [H-PAS-04]` | `100nF 25V X7R 10% 0603` |

---

## Dielectric rules (must appear in Dielectric and/or Spec)

| Role | Required dielectric |
|------|---------------------|
| EMG CFilt, DRL 4.7 nF, crystal load caps (15 pF default / 18 pF upgrade) | **C0G / NPO only** |
| ADS1299 VCAP1 100 µF | **C0G / NPO or tantalum** — never X7R/X5R |
| TPS61252 C2 100 pF | **C0G** |
| Power / digital bypass (100 nF, 10 µF bulk) | **X7R / X5R OK** |
| EMG series R | **±1%**, **63mW** |

Generic KiCad descriptions (`Unpolarized capacitor`, `Resistor`) are **not sufficient**.

---

## DNP / optional parts

| Catalog | Populate | Notes |
|---------|----------|-------|
| H-IC-02 REF5025 | not on the schematic | Internal ADS1299 ref used |
| H-IC-13 PCA9546 (`U_MUX1`) | **YES** | Fitted on revision 0.0.1: CRA and DP share I2C1 through the mux ([`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md)) |
| H-PAS-02 CM 470 pF arrays | not on the schematic | Optional CKC array |
| H-IC-11 / H-CON-04 sync LED | not on the schematic | Track A only |
| `U2` AT24CS64-MAHM | **ALT** (DNP) | Alternative package of the ID EEPROM `U1` |

