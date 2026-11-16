<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h04 — IMU cable assembly and shielding

> **Subsystem spec:** [`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.3. **HAT capture:** [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md). **Grounding policy:** [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2.3.

## What cable shielding is (and why we terminate it)

A **shield** is a conductive layer around the signal wires — usually a **metal braid** or **aluminium foil** plus an optional **drain wire** that is electrically tied to that layer. It is not one of the six I2C signals; it exists to:

1. **Reflect or shunt high-frequency electric fields** (USB noise, switchers, RF, fluorescent ballasts) so they couple less into SDA/SCL and INT.
2. **Reduce crosstalk** between conductors in long harnesses (collar → HAT).
3. **Give ESD and EMI currents a controlled return path** at the bench end instead of through `DGND` on pin 2.

A shield only works if it is **connected to a defined reference** at **one end** of the cable (here: the HAT). If both ends are tied to local grounds, **ground loops** can form (especially 50/60 Hz) — bad for EMG, still undesirable for IMU I2C. If neither end is tied, the shield **floats** and does little good.

**Why AGND on the HAT (not DGND on pin 2):** Pin 2 is **DGND** — the digital/sensor return for I2C current. The shield is a **separate** return for interference current. Project policy ([`h07`](h07_power_grounding_safety.md) §2.1–§2.3) routes all cable shields to **AGND** at the HAT; shield current crosses to **DGND** only at the **ADS1299 star tie**. Do **not** bond the shield to **DGND** at the jack “for convenience.”

## How the shield reaches AGND (7-pin connector)

The legacy **6-pin JST-SH** (`SM06B`) has **no crimp position** for the braid — only six signal contacts. Shield termination via footprint **MP** pads is layout-only and not in the netlist.

**v1 fix:** **7-pin JST-SH** on the HAT (`SM07B-SRSS-TB`) and **7-pin JST-GH** on the breakout cable (`GHR-07V-S`). **Pin 7** is **`IMU_SHLD`**: the drain wire and foil are crimped into contact 7 at both ends of the cable assembly; on the HAT PCB pin 7 routes to **`AGND`**; on the breakout PCB **pin 7 is not connected to the BNO085 GND pad** (far-end shield floating).

```text
  HAT SM07B                         Cable                         Breakout GHR-07
  pin 1-6  signals  ◄──────────────►  same  ◄──────────────────►  solder to BNO085
  pin 7    IMU_SHLD ──► AGND pour     braid/drain on pin 7 only   pin 7 NC (no GND plane)
```

## 7-pin pinout (HAT and breakout plug — identical order)

| Pin | Net | HAT | Breakout harness |
|-----|-----|-----|------------------|
| 1 | `+3V3D` | Power | 3V3 |
| 2 | `DGND` | Signal return | GND |
| 3 | SDA | I2C | SDA |
| 4 | SCL | I2C | SCL |
| 5 | INT | → MCU | INT |
| 6 | `IMU_NRST` | shared reset | nRST |
| **7** | **`IMU_SHLD`** | **`AGND` only** | **Do not wire to breakout GND** |

**Keying:** JST SH and GH housings are keyed; do not mix **SH plug into GH receptacle** or vice versa on the same end.

## Cable and connector BOM (per IMU link)

| Item | MPN / part family | Notes |
|------|-------------------|--------|
| HAT receptacle | **SM07B-SRSS-TB** | 7p, 1.0 mm, horizontal SMD |
| Cable plug (HAT side) | **SHR-07V-S-B** + **SSH-003T-P0.2-H** | Crimp; verify tool for SH |
| Breakout plug | **GHR-07V-S** + **SSH-003T-P0.2-H** (or GH-rated contact per JST sheet) | 1.25 mm pitch |
| Cable | **7×24 AWG** in **shielded** bundle | Twisted pairs: (SDA,SCL), (INT,nRST), (VCC,GND) |
| Length | **1.5 m** HAT jumper nominal (**I-02**) | See [`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.3.1 for total path |

> **Circular connector (optional, HAT/collar end only).** EMG patient cables use shielded circular connectors ([`h03_emg_frontend.md`](h03_emg_frontend.md) §10.2). The IMU link **keeps JST-GH at the body end** (MTA **≤2 g** budget). A small shielded circular may replace the JST **only at the HAT or collar**; drain → **AGND**.

## Assembly procedure (bench / harness shop)

1. **Strip** outer jacket; fan out seven insulated conductors + expose braid/foil.
2. **Bond drain to shield:** solder or fold drain wire onto braid/foil (one point near breakout end of cable).
3. **Crimp** conductors 1–6 into SH (HAT) and GH (breakout) contacts; **crimp conductor 7 (drain/shield)** into position 7 on **both** housings.
4. **HAT PCB:** pin 7 lands on net **`IMU_SHLD`** → short trace to **AGND** pour (see layout guide).
5. **Breakout:** solder pigtail pins 1–6 to Adafruit **4754** / SparkFun **BOB-22857** pads (3V3, GND, SDA, SCL, INT, RST). **Leave pin 7 unconnected to the breakout GND plane** — use a short insulated extension if the contact must be seated in the housing.
6. **Continuity check:** pin 7 ↔ braid at cable; **no** continuity pin 7 ↔ pin 2 at breakout PCB.
7. **Label** cables MTA / CRA / DP (heat-shrink).

## BNO085 breakout pigtail (Adafruit / SparkFun)

Both breakouts ship with **STEMMA QT** (4-pin I2C) on-board; full IMU use needs **INT** and **RST** from nearby pads.

| Cable pin | Typical breakout pad / label |
|-----------|------------------------------|
| 1 | **3V** / **VIN** (3.3 V) |
| 2 | **GND** |
| 3 | **SDA** |
| 4 | **SCL** |
| 5 | **INT** |
| 6 | **RST** / **RESET** |
| 7 | **NC at PCB** — cable shield only |

Strain relief: glue the **GH housing** to the MTA/CRA/DP shell so tension does not pull SMD pads ([`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.1).

## Optional adapter PCB (**I-04**)

Small **12 × 10 mm** daughterboard (optional catalog line):

- **J1:** `BM07B-GHS-TB` or vertical **GHR-07** footprint (cable side).
- **Castellated / 0.1″ pads** to mate with BNO085 breakout holes.
- Copper: pin 7 pad **isolated** (no tie to GND pour); silk **“SHLD: cable only — NC”**.

Reduces hand-wiring errors when building multiple harnesses.

## SPI / future harness

SPI + INT + WAKE + nRST needs **eight signals** plus shield → plan **9-pin** SH/GH variant; document in [`h04_imu_subsystem.md`](h04_imu_subsystem.md) when adopted.
