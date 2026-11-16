<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Schematic build guide conventions

All KiCad capture walkthroughs in `docs/hardware/` follow the same pattern (see [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md) as the reference implementation).

## Document pairing

| Build guide | Spec (“what / why”) | KiCad sheet |
|-------------|---------------------|-------------|
| [`h02_hat_plus_schematic_build_guide.md`](h02_hat_plus_schematic_build_guide.md) | [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §8 | `hardware/jtsmm_hat/sheets/hat_plus.kicad_sch` + root `jtsmm_hat.kicad_sch` |
| [`h02_hat_plus_eeprom_flashing.md`](h02_hat_plus_eeprom_flashing.md) | [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) bring-up | `hardware/jtsmm_hat/eeprom/jtsmm_hat_eeprom_vert.txt` / `jtsmm_hat_eeprom_hori.txt` |
| [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md) | [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) | `sheets/power.kicad_sch` |
| [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) | [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md), [`h08_rp2350_flashing.md`](h08_rp2350_flashing.md) | `sheets/mcu.kicad_sch` |
| [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md) | [`h04_imu_subsystem.md`](h04_imu_subsystem.md) | `sheets/imu_connectors.kicad_sch` |
| [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md) | Harness / shielding (off-sheet) | — |
| [`h04_imu_kicad_implementation.md`](h04_imu_kicad_implementation.md) | KiCad checklist (7-pin connectors) | `imu_connectors` (manual) |
| [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md) | [`h03_emg_frontend.md`](h03_emg_frontend.md) | `sheets/emg.kicad_sch` |
| [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) | [`h02`](h02_custom_hat_shield.md) §7, [`h06_synchronization.md`](h06_synchronization.md) §3 | `sheets/trigger_io.kicad_sch` |
| [`h10_schematic_logical_review.md`](h10_schematic_logical_review.md) | All h02–h07 specs (review record) | All sheets in `hardware/jtsmm_hat/` |
| [`h10_kicad_pcb_layout_guide.md`](h10_kicad_pcb_layout_guide.md) | [`h02`](h02_custom_hat_shield.md) §3–§5, [`h07`](h07_power_grounding_safety.md) §2 | `hardware/jtsmm_hat/jtsmm_hat.kicad_pcb` |

## Standard sections

1. Title + link to spec + target `.kicad_sch`
2. **How to read the diagrams** (ASCII legend)
3. **Step 0** — symbols / footprints ([`04_component_catalog.md`](../04_component_catalog.md))
4. **Steps 1–N** — capture order with ASCII wiring blocks
5. **Hierarchical labels** — table (child ↔ root sheet pins)
6. **ERC / PWR_FLAG** — KiCad 10 notes
7. **Pre-capture checklist**
8. **Layout** (optional short pointer to h02 / h07)

## Footprint audit (project-wide)

After Step 0 symbol/footprint choices, cross-check against:

| Doc / tool | Role |
|------------|------|
| [`h09_footprint_master_table.md`](h09_footprint_master_table.md) | Cross-sheet footprint index |
| [`h09_component_field_conventions.md`](h09_component_field_conventions.md) | BOM fields: **Spec**, **Voltage**, **Power**, CatalogID, Dielectric |

## Net naming (project-wide)

| Net | Meaning |
|-----|---------|
| `+5V_IN` | Raw 5 V from Pi header (hat_plus) into power sheet |
| `+5VA` | Clean 5.0 V analog (ADS1299 AVDD) |
| `+3V3D` | 3.3 V digital (RP2350A, DVDD, buffers, IMU JST VCC) |
| `AGND` | Analog ground domain |
| `DGND` | Digital ground domain |
| `IMU_SHLD` | IMU cable shield/drain (JST pin **7** → **AGND** @ HAT only) |

Do **not** use `+3V3A` for the analog rail — analog is **`+5VA`**.

## Diagram legend (reuse in every guide)

```text
A ──┬── 100nF ── B      capacitor from A to B
    └──► PIN            A also connects to PIN
[100Ω]                  series component
```
