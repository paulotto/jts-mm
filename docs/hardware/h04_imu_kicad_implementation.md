<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h04 — IMU KiCad implementation checklist

Checklist for the **7-pin JST-SH** IMU connectors (`SM07B-SRSS-TB`) on [`imu_connectors.kicad_sch`](../../hardware/jtsmm_hat/sheets/imu_connectors.kicad_sch).

> **Primary capture guide:** [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md)  
> **Footprint master:** [`h09_footprint_master_table.md`](h09_footprint_master_table.md)

---

## 7-pin connector (SM07B)

| Item | Value |
|------|-------|
| HAT receptacle | **SM07B-SRSS-TB** (a 6-pin SM06B has no shield pin) |
| Footprint | `Connector_JST:JST_SH_SM07B-SRSS-TB_1x07-1MP_P1.00mm_Horizontal` |
| Symbol | `Connector:Conn_01x07_Pin` |
| Pin 7 | **`IMU_SHLD` → `AGND`** @ HAT only |

Checklist:

- [ ] All three jacks (`J_IMU_MTA1`, `J_IMU_CRA1`, `J_IMU_DP1`) use **7-pin** symbol + SM07B footprint
- [ ] Pin table labels match [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md) Step 1
- [ ] Pin **7** net is **`IMU_SHLD`**, not `DGND`
- [ ] I2C pull-ups only on upstream bus (see build guide — no duplicate on `mcu`)

---

## Footprints

Footprints on the IMU sheet: JST-SH 7-pin horizontal, 0603 passives and LEDs (0402 for the 100 nF bypass capacitors), TSSOP-16 mux.

---

## Cable / harness

Off-board assembly: [`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md) — HAT **SH** plug, breakout **GH** receptacle, pin 7 = drain only.
