<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h10 — KiCad PCB Layout Guide (JTS-MM HAT)

> **Prerequisite:** `[h10_schematic_logical_review.md](h10_schematic_logical_review.md)`  
> **PCB file:** `hardware/jtsmm_hat/jtsmm_hat.kicad_pcb` (KiCad 10)

4-layer layout guide: **AGND/DGND split**, **single star tie**, and **ADS1299** placement rules.

---

## How to use this guide

| You are…              | Start                        | Then                                         |
| --------------------- | ---------------------------- | -------------------------------------------- |
| First open of the PCB | **§0** Board map             | **§1.1** Setup                               |
| Placing parts         | **§1.2** steps 1–5           | `[h02` §4](h02_custom_hat_shield.md)         |
| Pouring planes        | **§3** Ground · **§4** Power | —                                            |
| Routing EMG           | **§5.1** checklist           | `[h03` §4/§9](h03_emg_frontend.md)           |
| Routing MCU/USB       | **§1.2** step 8 · **§5.2**   | `[h05` §1](h05_mcu_schematic_build_guide.md) |
| Before fab            | **§7** Deliverables          | `[h02` §11](h02_custom_hat_shield.md)        |

### Contents

| §      | Topic                                                     |
| ------ | --------------------------------------------------------- |
| **§0** | Board map — template vs staging, coordinates, Pi keep-out |
| **§1** | Workflow — setup + 10-step layout pipeline                |
| **§2** | 4-layer stack-up                                          |
| **§3** | AGND/DGND split, zone pours, return paths                 |
| **§4** | L3 power zones + ADS1299 supply routing                   |
| **§5** | Pre-fab checklists (+ optional EMI can **§5.4**)          |
| **§6** | Tips, test pads, manufacturing notes                      |
| **§7** | Deliverables before fab                                   |
| **§8** | References                                                |
| **§9** | Star-ground cheat sheet                                   |

### KiCad hotkeys

| Key        | Action                     |
| ---------- | -------------------------- |
| **F8**     | Update PCB from schematic  |
| **X**      | Route track                |
| **V**      | Place via                  |
| **B**      | Fill all zones             |
| **F**      | Flip board view (bottom)   |
| **~**      | Highlight net under cursor |
| **Alt+F3** | 3D viewer                  |

### Glossary

| Term             | Meaning                                                             |
| ---------------- | ------------------------------------------------------------------- |
| **Staging**      | Footprints at **x < −5 mm** after F8 — move onto board before fab   |
| **Floor plan**   | Footprint positions **before** routing and zone pours               |
| **Domain guide** | `Eco1.User` line at **x ≈ 15** — planning only, not fabricated      |
| **Pour / zone**  | **Place → Add Filled Zone**; press **B** to fill                    |
| **Keep-out**     | ~**51 × 35 mm** over Pi body — **tall parts on rim only** (§0.5)    |
| **Star tie**     | Single **`NT_GND_TIE1`** (`Device:NetTie_2`) at `U_ADS1` pads 32/31 — only AGND↔DGND bridge |
| **Stitch via**   | GND via to inner plane — **same domain only** (AGND or DGND)        |

**Layers:** `F.Cu` (L1) · `In1.Cu` (L2 ground) · `In2.Cu` (L3 power) · `B.Cu` (L4)

---

## 0. Board map

The file contains the **Pi 5 HAT+ template** (locked) plus **staging footprints** from F8. Do not confuse the two regions.

### 0.1 Two regions

```text
  STAGING (not fab)              HAT+ TEMPLATE (real board)
  x ≈ −115 … −5                  x ≈ −35 … +65
  ┌─────────────────┐            ┌───────┬─────────────────────┐
  │ U_ADS1 U_MCU1   │  drag ──►  │ tab   │ J1 · holes · EEPROM │
  │ J_USB1 passives │            │ EMG   │ Dwgs.User guides    │
  └─────────────────┘            └───────┴─────────────────────┘
```

| Region   | X (mm)        | Action                                                |
| -------- | ------------- | ----------------------------------------------------- |
| Staging  | **< −5**      | **Move** onto outline (**§1.2** step 3)               |
| Template | **−35 … +65** | **Verify only** — locked outline, `J1`, holes, EEPROM |

### 0.2 Locked template (do not move)

| Ref                  | Position (mm)               | Notes                                                                                 |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------------- |
| `H1` / `H2`          | (3.5, 4) / (61.5, 4)        | M2.5 mounting                                                                         |
| `H3` / `H4`          | (3.5, 53) / (61.5, 53)      | M2.5 mounting                                                                         |
| `J1`                 | **(32.5, 4)** on `**F.Cu`** | Samtec HLE 2×20 stacking header — `[h02 HAT+](h02_hat_plus_schematic_build_guide.md)` |
| `Edge.Cuts`          | §0.3                        | Board outline                                                                         |
| `U1`/`U2` + passives | ~(61, 36–48)                | HAT ID EEPROM — display-corner keep-out                                               |

### 0.3 Outline and coordinates

**Bounding box:** **100.0 × 56.5 mm** (x **−35 … +65**, y **0 … 56.5**). KiCad: **+X right**, **+Y down**.

| Zone          | X (mm)                         | Use                                                 |
| ------------- | ------------------------------ | --------------------------------------------------- |
| Patient tab   | **−35 … 0**                    | EMG connectors, `U_ADS1`, analog LDO                |
| HAT+ core     | **0 … 65**                     | Standard 65 × 56.5 mm; `J1` on **top** edge (y ≈ 4) |
| Display notch | **~47 … 58**, y **~40 … 56.5** | `Dwgs.User` `Pi5 J3` / `Pi5 J4` — no tall parts     |

```text
  x=−35      x=0    x=15         x=47      x=65
    │◄─ tab ─►│◄─ digital ───────►│◄EEPROM──►│  y=0  top (J1)
    │  AGND   │     DGND          │          │
    │         │  Pi keep-out ↓    │  notch   │  y=56.5 bottom
    └─────────┴───────────────────┴──────────┘
              ▲ split x≈15 (Eco1.User)
```

**Floor-plan placement (summary):**

| Block                  | Region (x mm) | GND net |
| ---------------------- | ------------- | ------- |
| `U_ADS1`, EMG, ADP7142 | **−35 … 15**  | AGND    |
| MCU, boost, USB, JSTs  | **15 … 47**   | DGND    |
| EEPROM only            | **47 … 65**   | —       |

### 0.4 Pi keep-out (tall parts)

On the **65 × 56.5 mm** HAT+ core, **~51 × 35 mm ≈ half the area** sits over the Pi when stacked. That is normal.

|       | Core    | Keep-out                      | Left for tall parts                           |
| ----- | ------- | ----------------------------- | --------------------------------------------- |
| Width | 65 mm   | ~51 mm                        | **~7 mm** rim each side                       |
| Depth | 56.5 mm | ~35 mm inboard from `J1` (+Y) | **~17 mm** bottom strip (minus display notch) |

**Rules:**

- **Inside keep-out:** low-profile SMD only (MCU, passives) if stack height ≤ HAT+ clearance (~11 mm default; **15–16 mm** spacers recommended).
- **Rim + patient tab:** tall connectors (USB, JST, EMG headers).
- **Approx. rectangle:** x **7 … 58**, y **4 … 39** (from `J1` at 32.5, 4).
- Also respect `**Pi5 J3`/`J4`** boxes (x **47–56**, y **40–56**).

`J1` stays on `**F.Cu`** — do not flip to `B.Cu`. Optional: mark keep-out on `Eco1.User` or a rule area (not fabricated).

---

## 1. Workflow

### 1.1 One-time setup

1. Open `jtsmm_hat.kicad_pro` → **F8** (Update PCB from Schematic).
2. Fix missing footprints in schematic; F8 again. New parts land in **staging** — expected.
3. Audit footprints → `[h09_footprint_master_table.md](h09_footprint_master_table.md)`.
4. **File → Board Setup** — configure stack-up, constraints, net classes (tables below).

> **As set in `jtsmm_hat.kicad_pro`:** the DRC floors are those of the HAT+ template (minimum track **0.15 mm**, minimum clearance **0.12 mm**), below the 0.2 mm of the constraints table; the net classes are as in the table further down, except `Power` clearance (**0.2 mm**). The routed board has **0.2 mm** signal tracks and a smaller number of **0.15 mm** segments — `[h02` §3](h02_custom_hat_shield.md).

#### Physical stack-up

| Setting       | Value                                         |
| ------------- | --------------------------------------------- |
| Thickness     | **1.6 mm** (file shows 1.616 mm — OK for JLC) |
| Copper layers | **4**                                         |
| Finish        | **ENIG**                                      |
| `In1.Cu` type | **Ground**                                    |
| `In2.Cu` type | **Power**                                     |

#### Constraints (DRC floors)

| Setting           | Value                   |
| ----------------- | ----------------------- |
| Min clearance     | **0.2 mm**              |
| Min track width   | **0.2 mm**              |
| Min hole          | **0.3 mm**              |
| Min via / annular | **0.5 mm** / **0.1 mm** |
| Copper to edge    | **0.3 mm**              |

#### Pre-defined sizes

| Tracks (mm) | Use                               |
| ----------- | --------------------------------- |
| 0.2         | Default signal                    |
| 0.25        | Wider signal                      |
| 0.4 / 0.6   | Power branches / `+5VA`, `+5V_IN` |

| Via Ø / hole (mm) | Use                |
| ----------------- | ------------------ |
| 0.55 / 0.3        | Signal             |
| 0.7 / 0.4         | Power / GND stitch |

| Diff pair W / gap (mm) | Use       |
| ---------------------- | --------- |
| 0.2 / 0.4              | USB D+/D− |

#### Net classes

| Class     | Clearance | Track | Via Ø/hole | DP W/gap |
| --------- | --------- | ----- | ---------- | -------- |
| `Default` | 0.2       | 0.2   | 0.55/0.3   | —        |
| `Power`   | 0.25      | 0.4   | 0.7/0.4    | —        |
| `USB`     | 0.2       | 0.2   | 0.55/0.3   | 0.2/0.4  |
| `Analog`  | 0.25      | 0.2   | 0.55/0.3   | —        |

**Assign:** `USB_DP`/`USB_DM` → `USB` · `+5VA`/`+3V3D`/`+5V_IN` → `Power` · EMG inputs → `Analog` (optional).

**KiCad 10:** **Board Setup → Net Classes → Assignments** (or right-click pad → Properties → Net class).

---

### 1.2 Layout pipeline (10 steps)

Planes before most traces — EMG is sensitive to return paths (`[h07` §2](h07_power_grounding_safety.md)).

```mermaid
flowchart LR
  S1[1 Verify] --> S2[2 J1] --> S3[3 Move] --> S4[4 Floor plan]
  S4 --> S5[5 Fine place] --> S6[6 Planes] --> S7[7 Analog]
  S7 --> S8[8 Digital] --> S9[9 Star tie] --> S10[10 DRC]
```

| Step   | Task                        | KiCad                              | Detail         |
| ------ | --------------------------- | ---------------------------------- | -------------- |
| **1**  | Verify outline              | Solo `Edge.Cuts`                   | §0.3           |
| **2**  | Verify `J1`                 | (32.5, 4) on `F.Cu`, locked        | §0.4           |
| **3**  | Move staging                | **M** / **E** → inside outline     | Below          |
| **4**  | Floor plan                  | `Eco1.User` split **x ≈ 15**       | Below          |
| **5**  | Fine placement              | Bypass at IC pins                  | Below          |
| **6**  | Inner planes                | Zones on `In1.Cu`, `In2.Cu`        | **§3**, **§4** |
| **7**  | Route analog                | Class `Analog`, no cap vias        | **§5.1**       |
| **8**  | Route digital + L1/L4 fills | SPI, USB, then pours               | Below          |
| **9**  | Star tie + 3D               | `NT_GND_TIE1`, **Alt+F3**            | **§3.1**       |
| **10** | DRC                         | **Inspect → Design Rules Checker** | Below          |

---

#### Step 3 — Move staging onto the board

**Goal:** All footprints inside `Edge.Cuts`; nothing at **x < −5 mm**.

1. Box-select staging cluster (exclude locked `J1`, `H1`–`H4`, EEPROM).
2. **M** (Move) → anchor on `U_ADS1` → place near **(−10, 28)**.
3. Fine-tune: **E** → Position, or **right-click → Positioning → Move Exactly**.
4. **DRC** — courtyard overlaps only.

---

#### Step 4 — Floor plan (no copper yet)

**Goal:** Mark AGND/DGND regions and place parts on the correct side. **Zone pours come in step 6** (§3, §4).

**Draw split on `Eco1.User`:**

1. Appearance → enable `**Eco1.User`** → set active layer.
2. **Add → Line:** **(15, 0)** to **(15, 56.5)**.
3. Optional labels: `AGND — analog` (left), `DGND — digital` (right), `★ NT_GND_TIE1` near `U_ADS1`.

**Place footprints:**

| Ref        | Target x (mm)     | Notes                                                 |
| ---------- | ----------------- | ----------------------------------------------------- |
| `U_ADS1`   | **< 10**, y 15–40 | ~5 mm for passives around package                     |
| `J_EMG_*`  | **−30 … −5**      | Shell pads → **AGND**                                 |
| ADP7142    | Analog tab        | Full LDO on **AGND**; **~6.5 V** feed from boost only |
| TPS61252   | **> 15**          | Boost + switch node on **DGND**                       |
| `U_MCU1`   | **18–40**         | Low profile OK inside Pi keep-out                     |
| USB / JSTs | Digital rim       | Clear §0.4 keep-out and display boxes                 |

**Verify:** `**~`** on GND pads — `U_ADS1` → **AGND**, `U_MCU1` → **DGND**. **Inspect → Net Inspector** — nets still separate.

> Do not place GND vias on **x ≈ 15** except at the **`NT_GND_TIE1`** site (under `U_ADS1`).

---

#### Step 5 — Fine placement

| Analog (`F.Cu`) | Rule                           |
| --------------- | ------------------------------ |
| `U_ADS1` bypass | Same layer, **no via** cap↔pin |
| VCAP1 100 µF    | At pin                         |
| VREFP           | ≥10 µF + 0.1 µF; VREFN → AGND  |
| INx 4.7 nF C0G  | Across INxP–INxN at package    |

| Digital            | Rule                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| RP2350A SMPS loop  | `VREG_LX`–inductor–`DVDD` minimal                                             |
| `C47`…`C71` 100 nF | < 2 mm to pad, 2+ GND vias — `[h05` §1.1.2](h05_mcu_schematic_build_guide.md) |
| Crystal `Y1`       | Traces < 6 mm; nothing under crystal                                          |

**KiCad:** **Edit → Align and Distribute** · lock critical parts after this step.

---

#### Step 8 — Digital routing and outer ground fills

**Route (order):** DVDD/SMPS → SPI (`ADS_`*) → USB D+/D− (diff pair) → I2C/triggers → `J1` escape.

**Pour last:** `F.Cu` + `B.Cu` zones — **AGND** left, **DGND** right (same **x ≈ 15** split) · **B** · stitch vias **5–10 mm** (domain-local only). Details: **§3.4**.

---

#### Step 9 — Star tie

Place **`NT_GND_TIE1`** (`Device:NetTie_2`, footprint **`NetTie:NetTie-2_SMD_Pad0.5mm`**) at **`U_ADS1`**: **pin 1 → `AGND`** (pad **32 / `AVSS`)**, **pin 2 → `DGND`** (pad **31 / `RESV1`)**. Short traces on `F.Cu`; one via per net into `In1.Cu`.

**Inspect → Net Inspector** — exactly **one** AGND↔DGND path. **Alt+F3** — clearance vs Pi / M.2 HAT+.

---

#### Step 10 — DRC

**Inspect → Design Rules Checker** after every major step.

| Symptom           | Fix                                         |
| ----------------- | ------------------------------------------- |
| AGND ↔ DGND short | ≥ 0.5 mm zone gap; only `NT_GND_TIE1` bridges |
| Cap far from IC   | Same-layer placement                        |
| USB errors        | `USB` class, 0.2 / 0.4 mm pair              |
| Header collision  | Move tall part to rim                       |

---

## 2. Four-layer stack-up

> Spec: `[h02` §3](h02_custom_hat_shield.md) · Power: `[h07` §1](h07_power_grounding_safety.md)

```text
  F.Cu   L1  signal   ADS1299 · EMG · J1 · silk
  In1.Cu L2  ground   AGND ║ gap ║ DGND
  In2.Cu L3  power    +5VA · +3V3D · +5V_IN
  B.Cu   L4  signal   MCU · USB · digital
         └── 1.6 mm FR4, ENIG (JLC 4L) ──┘
```

| Layer | KiCad    | Role                                 |
| ----- | -------- | ------------------------------------ |
| L1    | `F.Cu`   | Analog signals, top connectors, `J1` |
| L2    | `In1.Cu` | Split ground — **AGND** / **DGND**   |
| L3    | `In2.Cu` | Power pours                          |
| L4    | `B.Cu`   | Digital signals, optional `U_MCU1`   |

**Solo-view while working:** `In1.Cu` for ground split · `In2.Cu` for power · `F.Cu`+`In1.Cu`+highlighted net for return-path checks.

**Fab:** JLCPCB 4L, 1.6 mm FR4, ENIG — min 0.127 mm trace; we use **0.2 mm**. No castellations in v1.

---

## 3. Ground planes (AGND / DGND)

> Rationale: `[h07` §2](h07_power_grounding_safety.md) · TI SBAS499 §12.1

### 3.1 Two nets, one tie

| Net      | Connects to                                                      |
| -------- | ---------------------------------------------------------------- |
| **AGND** | `U_ADS1` AVSS, VREFN, EMG shells, IMU pin 7 shield, ADP7142 GND  |
| **DGND** | `U_MCU1`, USB, boost, TPS73633, Pi `J1` GND, IMU pin 2, triggers |

**Only bridge:** **`NT_GND_TIE1`** at `**U_ADS1` pads 32/31**. Do not merge nets or stitch across the split.

### 3.2 Split geometry

Vertical split at **x ≈ 15** (full board height). Boost stays on **DGND**; filtered `**+5VA`** crosses to analog tab after ADP7142 (`[h07` §1](h07_power_grounding_safety.md)).

### 3.3 Pour `In1.Cu` (L2)

**Prerequisite:** Step 4 floor plan done.

1. **Board Setup** — `In1.Cu` = **Ground**.
2. Solo `**In1.Cu`** + `Eco1.User` reference line.
3. **Place → Add Filled Zone** → polygon **x −35 … 14.75**, y 0 … 56.5 → net `**AGND`**.
4. Second zone → **x 15.25 … 65** → net `**DGND`**. Gap **≥ 0.5 mm** between pours.
5. Zone properties: equal **priority**; **solid** pad connection under `U_ADS1` AVSS.
6. **B** (fill all) → verify two separate islands.
7. Place **`NT_GND_TIE1`** (§1.2 step 9).
8. Stitch vias every **5–10 mm** — **AGND vias in analog region only**, **DGND in digital**.

> Re-run **B** after any footprint move — stale pours hide DRC errors.

[KiCad copper zones](https://docs.kicad.org/10.0/en/pcbnew/pcbnew_copper_zones.html)

### 3.4 Pour `F.Cu` / `B.Cu` (after routing)

Same **x ≈ 15** boundary as L2 · enable **remove islands** · stitch vias (domain-local) · solid AGND under `U_ADS1`.

### 3.5 Return paths and shields

| Signal           | Route over | Notes                              |
| ---------------- | ---------- | ---------------------------------- |
| EMG inputs       | **AGND**   | No parallel digital in analog zone |
| SPI to `U_ADS1`  | **DGND**   | Short; meets chip at pads          |
| USB D+/D−        | **DGND**   | Away from analog                   |
| Pi `J1` GND      | **DGND**   | All GND pins                       |
| IMU pin 2        | **DGND**   | Signal return                      |
| IMU pin 7 (SHLD) | **AGND**   | Shield; far end floating           |

**EMG connector shells** → **AGND** (one tie each). **Never** bond cable shields to **DGND** — `[h07` §2.3](h07_power_grounding_safety.md).

---

## 4. Power planes (L3) and ADS1299 feeds

> Schematic: `[h07_power_schematic_build_guide.md](h07_power_schematic_build_guide.md)`

### 4.1 `In2.Cu` zones

Solo `**In2.Cu`** → three separate polygons → clearance **0.25 mm** (Power class) → **B**.

| Net      | Region           | Feeds                                 |
| -------- | ---------------- | ------------------------------------- |
| `+5VA`   | Analog (x < 15)  | ADP7142 → `U_ADS1` **AVDD**           |
| `+3V3D`  | Digital (x > 15) | MCU, IMU, triggers, `U_ADS1` **DVDD** |
| `+5V_IN` | Near `J1`        | Pi 5 V → boost + TPS73633             |

Do **not** merge `+5VA` and `+3V3D` pours.

### 4.2 Routing `+5VA` and `+3V3D` to `U_ADS1`

| Rail        | Source side             | ADS1299 pins | Return                   |
| ----------- | ----------------------- | ------------ | ------------------------ |
| `**+5VA`**  | ADP7142 on **AGND** tab | **AVDD**     | **AVSS → AGND**          |
| `**+3V3D`** | TPS73633 on **DGND**    | **DVDD**     | **DGND** pins → **DGND** |

- **Boost (TPS61252)** and all switcher caps stay on **DGND**. Only a **~6.5 V trace** crosses to ADP7142 **VIN**; full LDO + caps on **AGND**.
- `**+3V3D`** trace crosses split over **DGND** on L2 — not a ground bridge.
- At the chip: **1 µF per AVDD/DVDD pin** on **same layer**, **no via** between cap and pin.
- Parallel `+5VA` / `+3V3D` traces OK for short runs; keep **≥ 0.5 mm** spacing; avoid running alongside EMG inputs.
- Ground connects domains only at **`NT_GND_TIE1`**, not via supply traces.

### 4.3 MCU buck copper relief

> **RP2350 datasheet** (Raspberry Pi, RP-008373-DS-2) **pp. 448–457** (§6.3.8 external components and PCB layout). **Critical:** **pp. 454–456** (§6.3.8.1 layout recommendations, Figure 23 switch-loop, **Figure 24** inner-layer void under `VREG_LX` / inductor).

On this HAT the buck loop is **`U_MCU1` `VREG_LX` → `L2` (3.3 µH) → `DVDD`** with **`C45`** (CIN), **`C66`** (COUT at `L2`) and **`C46`** on `DVDD`, and **`R39`** (33 Ω) + **`C40`** (4.7 µF) on **`VREG_AVDD`**. **`L2`** and the LX routing sit on **`B.Cu`** (digital half, **DGND**). Schematic: [`h05` §1.1](h05_mcu_schematic_build_guide.md).

Raspberry Pi’s rules (Figure 24): **no plane copper directly under the inductor body or the `VREG_LX` switch node** on any inner layer; on the **component side** of the loop, **pull `DGND` back** from under **`L2`** and along the LX trace.

**KiCad 10 — after routing, before final fill (`B`):**

1. Solo **`B.Cu`** — confirm **`VREG_LX`** is a short intentional pour/trace; **`L2`** pad 2 (dot) → LX, pad 1 → `DVDD` / `VREG_FB` cluster ([`h05` §1.1](h05_mcu_schematic_build_guide.md)).
2. Select the **`DGND`** zone → **right-click → Zones → Add Zone Cutout** → rectangle **~6 × 4 mm** over **`L2`** courtyard (+ ~1 mm margin).
3. Add a second cutout in the same **`DGND`** zone along **`VREG_LX`** from the MCU pad to **`L2`** (trace width + ~1 mm each side) so ground does not hug the switch node.
4. Repeat cutouts on **`In1.Cu`** and **`In2.Cu`** for the same footprint (vertical column through the board — Figure 24). Cut matching voids in any **`+3V3D`** pour on **`In2.Cu`** if it overlaps the LX column.
5. If **`DGND`** is poured on **`F.Cu`** above the MCU region, add the same void under the projected **`L2` + LX** area.
6. **`B`** (fill all zones) · **3D viewer** — no fill under **`L2`** / LX on inner layers.

**Also from §6.3.8.1 (pp. 454–456):** keep CIN/COUT switching loops tiny and on one side of the board; **`VREG_FB`** from COUT output — **do not route under `L2`**; dedicated GND return for **`VREG_AVDD`** filter cap (do not share CIN/COUT high-current GND vias).

---

## 5. Pre-fab checklists

### 5.1 ADS1299 / EMG (highest priority)

- [ ] Every bypass cap **same layer as pin** — **no via** cap↔pin
- [ ] **VCAP1 = 100 µF** at pin
- [ ] **VREFP:** ≥10 µF + 0.1 µF (VREFN → AGND)
- [ ] **4.7 nF C0G** across each INxP–INxN
- [ ] **100 Ω** series — short input traces
- [ ] Patient connectors on analog edge; shells → **AGND**
- [ ] **Single star tie** — Net Inspector: one AGND↔DGND path

### 5.2 Power / MCU

- [ ] Boost TPS61252 on **DGND**, digital half
- [ ] ADP7142 full layout on **AGND** tab; **~6.5 V** from boost only
- [ ] **VREG_LX → 3.3 µH → DVDD** loop minimal
- [ ] **Copper relief** under **`L2` / `VREG_LX`** on **`In1.Cu`**, **`In2.Cu`**, and **`DGND`** cutouts on **`B.Cu`** — **§4.3**
- [ ] **11× 100 nF** (`C47`…`C71`) < 2 mm, 2+ GND vias
- [ ] **≥ 9 vias** on RP2350A EP → DGND

### 5.3 Digital / mechanical

- [ ] USB D+/D− short pair; ESD at connector
- [ ] Crystal traces < 6 mm
- [ ] SPI away from analog inputs
- [ ] Staging moved inside outline (**§1.2** step 3)
- [ ] `J1` at **(32.5, 4)** on `**F.Cu`**; tall parts on rim only (**§0.4**)
- [ ] No tall parts in **Pi5 J3/J4** boxes
- [ ] `H1`–`H4` unchanged; IMU/trigger JSTs on digital rim

### 5.4 Optional EMI shield can (H-MECH-01)

**DNP on first bring-up.** Tinplate cage over `U_ADS1` + input filters — complements star ground; does not replace it.

| Tier            | Frame             | Cover       | Size           |
| --------------- | ----------------- | ----------- | -------------- |
| **A (default)** | Würth `36103305S` | `36003300S` | 30 × 30 × 3 mm |
| **A compact**   | `36103255S`       | `36003250S` | 25 × 25 × 3 mm |
| **B (JLC)**     | TE `2118717-2`    | `2118717-4` | 26.7 × 26.7 mm |

**Layout if populated:** frame on `F.Cu` analog zone · all frame pads → **AGND** (solid) · stitch to L2 · SPI exits toward digital side · **VCAP1** may need placement **outside** 3 mm cage — see `[04_component_catalog.md](../04_component_catalog.md)` **H-MECH-01** and `[h02` §2](h02_custom_hat_shield.md).

---

## 6. Tips and test access

- Pour outer zones **after** critical routes.
- **DRC** after placement, pours, and before Gerbers.
- **3D viewer** for M.2 HAT+ stack clearance.
- Thermals on DGND OK for hand-solder; **solid AGND** under `U_ADS1`.
- USB FS — no controlled impedance fab note; match 0.2 / 0.4 mm pair anyway.
- USB shield → **DGND** at connector.

**Test pads:** `[h02` §6.1](h02_custom_hat_shield.md) — stubs < 3–5 mm; no TPs on USB D+/D− or EMG INxP/N. Scope return: analog → `TP_AGND1`; digital → `DGND`.

---

## 7. Deliverables before fab

| Item          | Tool                                                      |
| ------------- | --------------------------------------------------------- |
| ERC clean     | `kicad-cli sch erc jtsmm_hat.kicad_sch`                   |
| DRC clean     | **Inspect → Design Rules Checker**                        |
| BOM / CPL     | `hardware/jtsmm_hat/production/bom.csv` (BOM), `positions_correct.csv` (placement), `designators.csv` |
| Layout review | This doc §5 + `[h02` §11](h02_custom_hat_shield.md)       |
| Gerbers       | `hardware/jtsmm_hat/production/JTS-MM_EMG_IMU_HAT_0.0.1.zip` (Gerber and drill files) |

---

## 8. References

| Doc                                                                                           | Topic                       |
| --------------------------------------------------------------------------------------------- | --------------------------- |
| `[h02](h02_custom_hat_shield.md)`                                                             | Mechanical, floor plan, fab |
| `[h03](h03_emg_frontend.md)`                                                                  | ADS1299 input network       |
| `[h05` MCU](h05_mcu_schematic_build_guide.md)                                                 | RP2350A, USB, crystal       |
| RP2350 datasheet §6.3.8 (Raspberry Pi, RP-008373-DS-2)                                        | Buck layout pp. 448–457 (454–456) |
| `[h07](h07_power_grounding_safety.md)`                                                        | Star ground, power tree     |
| `[h09` footprints](h09_footprint_master_table.md)                                             | Footprint assignments       |
| ADS1299 datasheet (TI SBAS499)                                                                | §12.1 layout                |
| [KiCad 10 PCB Editor](https://docs.kicad.org/10.0/en/pcbnew/pcbnew.html)                      | Zones, stack-up, DRC        |
| [Pi HAT+ spec](https://pip.raspberrypi.com/documents/raspberry-pi-hat-plus-specification.pdf) | Mechanical                  |

---

## 9. Star-ground cheat sheet

```text
 ┌─────────────────────────────────────────────────────────────┐
 │  STAR GROUND — six checks before Gerbers                    │
 ├─────────────────────────────────────────────────────────────┤
 │  1. AGND and DGND separate in schematic        ✓ (done)     │
 │  2. In1.Cu:  AGND zone | GAP ≥0.5 mm | DGND    ← pour + B   │
 │  3. NT_GND_TIE1 at U_ADS1 pads 32/31 only     ← one bridge │
 │  4. F.Cu/B.Cu fills match split at x ≈ 15                   │
 │  5. Stitch vias: AGND-only or DGND-only        ← no cross   │
 │  6. DRC → zero AGND–DGND shorts (except R)                  │
 └─────────────────────────────────────────────────────────────┘
```

When in doubt: *“Does this copper let digital return current flow through AGND?”* — If yes, fix it. See `[h07` §2](h07_power_grounding_safety.md).