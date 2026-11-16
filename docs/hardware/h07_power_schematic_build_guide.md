<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h07 — Power Schematic Build Guide (KiCad capture walkthrough)

Step-by-step companion to [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) for capturing the **power tree** in KiCad 10. Work inside `hardware/jtsmm_hat/sheets/power.kicad_sch`.

> **Authoritative sources:** [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §1–§2, part IDs in [`04_component_catalog.md`](../04_component_catalog.md) (H-IC-03, H-IC-03b, H-IC-04). ADS1299 analog requirements: SBAS499 §7.3 (AVDD−AVSS = 4.75–5.25 V).

## How to read the diagrams

```text
+5V_IN ──┬── 10µF ── DGND     bulk / HF bypass on a rail
         └──► VIN              connection to regulator input
[L]                         inductor (boost output filter)
```

---

## Step 0 — Library and parts

> **Footprint audit:** [`h09_footprint_master_table.md`](h09_footprint_master_table.md) · **L1** boost inductor → `jtsmm:AOTAB201610S3R3101T`.

| Catalog ID | Part | Symbol / notes |
|------------|------|----------------|
| H-IC-03b | **TPS61252DSG** | KiCad: `Regulator_Switching:TPS61252DSG`; boost **5 V → 6.5 V**; place on **DGND** |
| H-IC-03 | **ADP7142AUJZ-5.0** | KiCad: `Regulator_Linear:ADP7142AUJZ-5.0`; fixed **5.0 V**; **SENSE/ADJ → +5VA** at load |
| H-IC-04 | **TPS73633DBV** | KiCad: `Regulator_Linear:TPS73633DBV`; fixed **3.3 V**; see Step 2 pin table |
| — | **L1**, **C1/C2/C3**, **R1/R2/R3/R4** per Table 3 | See Step 3 — TI Figure 9 / Table 3 designators |
| — | **C_NR** 10 nF (TPS73633 NR pin); 100 nF + 10 µF per rail | X7R; 2.2 µF min on ADP7142 ([`h07`](h07_power_grounding_safety.md) §1.5) |
| H-PAS-05 | Panel power LEDs ×4 on **`power`** | `Device:LED` · `LED_SMD:LED_0603_1608Metric`; Value e.g. `LED_5VIN_Green`, `LED_3V3D_OK_Green`, `LED_6.5VA_PG_Green`, `LED_5VA_OK_Green` — Step 8 |

**Footprints:** both LDOs `Package_TO_SOT_SMD:TSOT-23-5`; boost `Package_SON:WSON-8-1EP_2x2mm_P0.5mm_EP0.9x1.6mm`; **L1** `jtsmm:AOTAB201610S3R3101T`; 0603/0805 passives.

**H-IC-03 alternate:** **ADP7142ARDZ-5.0** (SOIC-8 EP) — same 5.0 V fixed die; use only if layout already targets SOIC-8.

### Why TPS61252 (not TPS61299 / TLV61070)

The +5VA tree needs a **~6.5 V** pre-regulator so the ADP7142 has dropout headroom when the Pi 5 V sags ([`h07`](h07_power_grounding_safety.md) §1.1). **TPS61252DSG** is the catalog pick because it:

- Is in the **default KiCad** library (`Regulator_Switching:TPS61252DSG`) with the WSON-8 footprint above.
- Meets the electrical need: **2.3–6.0 V** in, **3.0–6.5 V** adjustable out, 1.5 A switch — far above the ~65 mA ADS1299 AVDD budget.
- Uses an integrated synchronous rectifier (no external Schottky).

Earlier catalog alternates **TPS61299** and **TLV61070** are **not** suitable as drop-in replacements for this design: both are limited to **Vout ≤ 5.5 V** (TI datasheets), so they cannot produce the documented **~6.5 V** boost without changing the LDO headroom strategy. They are also absent from the default KiCad symbol set.

---

## Step 1 — Hierarchical ports and PWR_FLAG

**Inputs (west):**

| Label | From root / hat_plus |
|-------|----------------------|
| `+5V_IN` | Pi 5 V (via `HAT_5V0` on hat_plus sheet) |
| `GND_IN` | Pi GND (via `HAT_GND`) |

**Outputs (east) — use these exact names:**

| Label | Consumers |
|-------|-----------|
| `+5VA` | `emg` sheet (ADS1299 AVDD) |
| `+3V3D` | `mcu`, `emg` (DVDD), `trigger_io`, `imu_connectors` |
| `AGND` | Analog domain (emg, jack shields, **`IMU_SHLD`** on JST pin 7) |
| `DGND` | Digital domain (RP2350A incl. core-buck VREG_PGND, boost, digital LDO GND) |

Place a **`PWR_FLAG`** on each exported rail at the regulator output (KiCad: power output not driven otherwise).

> **Do not** export `+3V3A` — that was a stub typo. Analog supply is **`+5VA`** at 5.0 V, not 3.3 V.

---

## Step 2 — Digital 3.3 V rail (TPS73633DBV)

```text
+5V_IN ──┬── 10µF ── DGND
         ├── 100nF ── DGND
         └──► IN (TPS73633DBV, pin 1)
                EN (pin 3) ──► +5V_IN / IN   (tie to VIN — regulator on with Pi 5 V)
                NR (pin 4) ──┬── 10nF ── DGND   (C_NR noise-reduction, not optional)
                GND (pin 2) ──► DGND
                │
                OUT (pin 5) ──┬── 10µF ── DGND
                              ├── 100nF ── DGND
                              └──► +3V3D ──► hierarchical out
```

### TPS73633DBV — pin wiring (SOT-23-5)

| Pin | Name | Connect to |
|-----|------|------------|
| 1 | IN | `+5V_IN` (with 10 µF + 100 nF to **DGND** at the IC) |
| 2 | GND | **DGND** |
| 3 | EN | **`+5V_IN` or IN** — high = on. TI allows **EN = IN** if shutdown is unused ([TPS736](https://www.ti.com/lit/ds/symlink/tps736.pdf) §7.3). Do **not** leave floating. |
| 4 | NR | **10 nF (`C_NR`) to DGND** — bypasses the internal bandgap for lower output noise (~30 µV rms typ with 0.1 µF; **10 nF** is the usual BOM). Do **not** tie NR to OUT or leave unconnected. |
| 5 | OUT | **`+3V3D`** + 10 µF + 100 nF to **DGND** |

Typical headroom: 5 V in → 3.3 V out @ ≤160 mA ([`h07`](h07_power_grounding_safety.md) §1.1 budget). Optional **0.1 µF** on IN if the Pi 5 V trace is long (TI good-practice note).

> **RP2350A core regulator is on the `mcu` sheet, not here.** `+3V3D` also feeds the RP2350A `VREG_VIN`; the **1.1 V core SMPS** (3.3 µH inductor `L_VREG`, **CIN/COUT 4.7 µF**, **33 Ω + 100 nF** on VREG_AVDD, VREG_LX→DVDD, VREG_PGND→DGND) is captured next to the MCU in [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.1. Keep its switching loop tight and on the digital side ([`h07`](h07_power_grounding_safety.md) §2.4, §5).

---

## Step 3 — Boost + analog 5 V rail (switcher on DGND)

A linear regulator cannot produce 5.0 V from a 5.0 V input. **Boost first, then LDO.**

Capture must match TI **Figure 9** and **Table 3** in the **TPS61252** datasheet. Use **Table 3 refdes** on the schematic (`C1`…`R4`), not aliases like `C_IN` / `C_FF`.

> **TI naming quirk:** §11.2.2.1 calls the feed-forward cap “C1”, but **Table 3** is **C1 = input**, **C2 = 100 pF (VOUT–FB)**, **C3 = 2× 22 µF output**. Follow **Table 3**.

> **Not in Table 3:** extra **100 nF** on the boost rails. Omit for a strict TI BOM, or add per [`h07`](h07_power_grounding_safety.md) §1.5.

### Figure 9 topology (Table 3 refdes)

Same structure as the PDF; **R1/R2** are for **6.5 V** on this HAT (TI’s figure is the **5 V** example).

> **Ground in this block = `DGND` only.** The TI figure says “GND”; on this board that is **`DGND`** (switcher domain). It is **not** `AGND` and it is **not** a separate net from `GND_IN` — see below.

```text
  +5V_IN  (from Pi, 2.3 V … 6.0 V)  ─── input node (same net as VIN pin)
      │
      ├──── C1 ──── DGND              Table 3: 4.7 µF; Figure 9 draws 10 µF
      │
      ├──── VIN (8)     EN (6) ── this node (+5V_IN)
      │
      └──── L1 ──── SW (7)            L1 only between input node and SW — not VOUT

      TPS61252:  GND (1), EP (9) ── DGND
                   ILIM (4) ── R3 ── DGND
                   VOUT (2) ──┬── C3 ──── DGND    Table 3: 2× 22 µF
                              ├──── C3 ──── DGND
                              ├──── R1 ──── FB (3) ──── R2 ──── DGND
                              │         └── C2 100 pF (VOUT to FB)
                              ├──── R4 1 MΩ ──── PG (5) ────► open-drain PG (Step 8 LED)
                              └── ~6.5 V ──► ADP7142 VIN   (LDO GND → AGND)
```

> **Common capture mistake:** Do **not** wire **SW ↔ VIN** with only **L1** between them *and* a separate wire from **+5V_IN** to that same junction (that shorts **L1**). **+5V_IN** and **VIN (8)** share one node; **L1** goes from that node to **SW (7)** only — per TI §11.2.2.4 (“between the **VIN** pin and **SW** pin”).

| Net | Role |
|-----|------|
| **`GND_IN`** | Hierarchical **in** from Pi `HAT_GND` — **return reference** for `+5V_IN`, not a third ground domain |
| **`DGND`** | **All TPS61252 returns** (pins 1/EP, C1, C3, R2, R3); TPS73633; RP2350A (incl. core-buck VREG_PGND); boost switching current |
| **`AGND`** | **ADP7142 GND** and caps only — **after** the ~6.5 V node, on the quiet LDO side |

On the schematic: tie **`GND_IN`** to **`DGND`** at the sheet entry (same as Step 1 / hat_plus). Do **not** tie the boost to **`AGND`**.

| Check | Datasheet |
|-------|-----------|
| **L1** | **Input node (VIN pin) → SW (7)** only — **not** SW↔VOUT, **not** a bypass from +5V_IN around L1 |
| **C1** | **Input** (VIN node) → **DGND** — **never** the 22 µF parts |
| **C3** | Both **22 µF** on **VOUT (2)** → **DGND** |
| **C2** | **VOUT → FB**, 100 pF COG (parallel **R1**) |
| **R1 / R2** | **R1:** VOUT → FB; **R2:** FB → **DGND** |
| **R3** | **ILIM → DGND** (or ILIM → VIN for max) |
| **R4** | **PG ↔ VOUT** (1 MΩ) per TI Table 3; **PG** is open-drain **out** — **D_6V5_PG** on **PG** (Step 8) |

### Table 3 — component values

| Ref | Table 3 (TI) | This HAT (6.5 V) | Figure 9 (5 V) |
|-----|----------------|-------------------|----------------|
| **C1** | 1× **4.7 µF**, 10 V, X7R @ **VIN** | **4.7 µF** or **10 µF** | 10 µF drawn |
| **C2** | 1× **100 pF**, COG, **VOUT–FB** | **100 pF** | CFF 100 pF |
| **C3** | **2× 22 µF**, 10 V @ **VOUT** | **2× 22 µF** | one “COUT 22 µF” |
| **L1** | **1.0 µH**, ~2.1 A Isat typ. | **1.0 µH** (1.5 µH OK §11.2.2.4) | 1 µH |
| **R1** | sets VOUT | **787 kΩ** (nearest E24; was 795 kΩ) | **768 kΩ** |
| **R2** | sets VOUT | **180 kΩ** | **243 kΩ** |
| **R3** | ILIM | **20 kΩ** (~500 mA) or ILIM=VIN | RILIM 20 kΩ |
| **R4** | **1 MΩ**, PG–VOUT | **1 MΩ** | 1 MΩ |

**6.5 V (Eq. 5):** `VOUT = VFB × (1 + R1/R2)` with **VFB ≈ 1.2 V**. **R2 = 180 kΩ**, **R1 = 787 kΩ** → **~6.45 V** (nearest stocked E24 to 795 kΩ); **R1 + R2 ≤ ~1 MΩ**.

### TPS61252DSG — pin wiring (WSON-8)

| Pin | Name | Connect to |
|-----|------|------------|
| 1 | GND | **DGND** |
| 2 | VOUT | **C3** (2× 22 µF) + **R1/R2/C2** + **R4**; then **ADP7142 VIN** |
| 3 | FB | **R1** from VOUT, **R2** to **DGND**, **C2** from VOUT to FB |
| 4 | ILIM | **R3** to **DGND** (Eq. 6) or **VIN** for max |
| 5 | PG | **R4** (1 MΩ) to **VOUT** (required for PG function); open-drain **out** — **high-Z when VOUT in regulation**, **pulls to GND** when out of regulation (TPS61252 datasheet §10.3.6). Wire **D_6V5_PG** here (Step 8) or tie **PG → DGND** only if that LED is **DNP** |
| 6 | EN | **VIN** or **`+5V_IN`** — must be terminated |
| 7 | SW | **L1** to **input node** (other end is **VIN** / `+5V_IN`) |
| 8 | VIN | **`+5V_IN`** + **C1** — same net as **L1** input side |
| EP | GND | **DGND** (must solder) |

### ADP7142AUJZ-5.0 — Figure 1 (fixed 5 V)

Capture must match **Figure 1** and **Figure 4** / **Table 5** in the Analog Devices **ADP7142** datasheet (Rev. K). The **AUJZ-5.0** suffix is the **fixed 5.0 V** option — **no** external R1/R2 divider on **SENSE/ADJ**.

> **Input headroom:** spec tests use `VIN = VOUT + 1 V` (whichever is greater vs 2.7 V). For **5.0 V** out you need **VIN ≥ ~6.0 V** — the **~6.5 V** boost output satisfies this with margin when the Pi 5 V sags.

> **Ground in this block = `AGND` only.** Boost switching returns stay on **`DGND`**; the LDO is the boundary into the analog domain ([`h07`](h07_power_grounding_safety.md) §2).

```text
  ~6.5 V (TPS61252 VOUT) ──┬── CIN ≥2.2 µF ── AGND     Figure 1 / CAPACITOR SELECTION
                           ├── 10 µF ── AGND             project bulk ([`h07`](h07_power_grounding_safety.md) §1.5)
                           ├── 100 nF ── AGND
                           └── VIN (1)

  EN (3) ──► same ~6.5 V node as VIN (1)     Table 5: “connect EN to VIN” for auto start

  GND (2) ── AGND

  VOUT (5) ──┬── COUT ≥2.2 µF ── AGND        at IC, close to VOUT + GND pins
             ├── 10 µF ── AGND
             ├── 100 nF ── AGND
             ├── +5VA ──► hierarchical → emg
             │
  SENSE/ADJ (4) ──► +5VA at ADS1299 / load   Table 5: “Connect to load” (Kelvin)
                    (same net as VOUT at the ADC — not AGND, not a divider)
```

| Check | Datasheet |
|-------|-----------|
| **CIN / COUT** | **≥ 2.2 µF** ceramic each, **ESR ≤ 0.3 Ω**, **X5R/X7R** (Rev. K §CAPACITOR SELECTION; Table 2 **CMIN** ≥ 1.5 µF effective) |
| **CIN vs COUT** | If total **COUT** > 2.2 µF, increase **CIN** to **match** (we use ~12 µF out → **2.2 µF + 10 µF** on **VIN** is appropriate) |
| **EN** | **Tie to VIN** (~6.5 V) — do **not** float. **`+5V_IN` (5 V)** is above **ENHIGH** (~1.2 V typ.) but is **not** the Figure 1 connection |
| **SENSE/ADJ** | **Kelvin** to **`+5VA` at the load** (ADS1299 AVDD bank) — improves accuracy vs sensing only at the LDO pin |
| **SS** | **N/A** on TSOT-5 (soft-start pin is LFCSP/SOIC only) |
| **VIN range** | **2.7–40 V**; this design uses **~6.5 V** in, **5.0 V** out |

### ADP7142AUJZ-5.0 — pin wiring (TSOT-23-5, Figure 4)

| Pin | Name | Connect to |
|-----|------|------------|
| 1 | VIN | Boost **~6.5 V** + **≥2.2 µF** + **10 µF** + **100 nF** to **AGND** at IC |
| 2 | GND | **AGND** |
| 3 | EN | **Same net as pin 1 (VIN)** — ~6.5 V from boost |
| 4 | SENSE/ADJ | **`+5VA` at load** (ADS1299 AVDD / local caps) — Kelvin; **not** GND; **no** divider |
| 5 | VOUT | **`+5VA`** + **≥2.2 µF** + **10 µF** + **100 nF** to **AGND** at IC |

**Fixed 5.0 V:** internal reference — do not use the adjustable **Figure 43** divider. Optional **Figure 2** “bump” (5 V → 6 V via divider) is **not** used on this HAT.

**Bench check:** **+5VA** within **4.75–5.25 V** at ADS1299 (SBAS499 §7.3) and within ADP7142 accuracy (±0.8% typ. initial, ±1.5% over −40°C to +85°C for 5 V).

**Layout (Figure 66 / PCB LAYOUT CONSIDERATIONS):** place **CIN** as close as possible to **VIN** and **GND**; **COUT** as close as possible to **VOUT** and **GND**; use **0805/1206** on the same layer as the TSOT when possible. Boost and its caps remain on **DGND**; only **`+5VA`** and the LDO quiet side use **AGND** ([`h07`](h07_power_grounding_safety.md) §2).

---

## Step 4 — AGND and DGND on the schematic

On the **schematic**, export both `AGND` and `DGND` as separate power symbols.

On the **`emg`** sheet, place **`NT_GND_TIE1`** (`Device:NetTie_2`): **pin 1 → `AGND`**, **pin 2 → `DGND`**. Footprint **`NetTie:NetTie-2_SMD_Pad0.5mm`** (PCB feature — exclude from BOM). On the **PCB**, route it between **`U_ADS1` pads 32 and 31** ([`h07`](h07_power_grounding_safety.md) §2.2, [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md) §1e). Do **not** add a second tie elsewhere.

```text
   DGND ◄── boost, TPS73633, RP2350A region (digital, incl. core-buck switch node)
   AGND ◄── ADP7142AUJZ-5.0 quiet side, emg analog, IMU pin 7 IMU_SHLD (HAT end)

   (PCB only) AGND ──★── DGND   single tie at ADS1299 DGND pin area
```

---

## Step 5 — Bench test points

> The four power test pads are on the `power` sheet as **`TP_5VIN1`**, **`TP_3V3D1`**, **`TP_5VA1`** and **`TP_DGND_PWR1`** (full pad map: [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §6.1). Power **LEDs** (`Step 8`) show rail presence only; TPs allow DMM voltage and ripple checks before SWD flash.

The pads are on the **`power`** sheet and sit next to the power panel LEDs (`Step 8`). Footprint: `TestPoint:TestPoint_Pad_D1.0mm`, top copper, silkscreen net label. Catalog: **H-CON-05**.

| Ref | Net | Tap | Use | Accept (DMM) |
|-----|-----|-----|-----|--------------|
| **TP_5VIN1** | `+5V_IN` | After input bulk cap (Pi-side) | Confirm Pi header 5 V under load | 4.8–5.2 V |
| **TP_3V3D1** | `+3V3D` | TPS73633 **OUT** / hierarchical export | **Required before SWD** ([`h08_rp2350_flashing.md`](h08_rp2350_flashing.md)) | 3.135–3.465 V |
| **TP_5VA1** | `+5VA` | ADP7142 **VOUT** (ADS AVDD feed) | ADS1299 needs 4.75–5.25 V analog | 4.75–5.25 V |
| **TP_DGND_PWR1** | `DGND` | Power cluster | DMM / scope return for power measurements | 0 V vs rails |

**Alternative probe points:** Pi header pins 2/4 vs GND (`+5V_IN`); `J_IMU_MTA1` pin 1 vs pin 2 (`+3V3D`); rail side of `R31`, the series resistor of `LED_5VA_OK1` (`+5VA`, less ideal).

**No test pad** on the `+6.5V` boost output — `LED_6V5D_PG1` (boost PG) + downstream `TP_5VA1` are sufficient. RP2350 **DVDD (~1.1 V)** has **no bench TP** — VREG loop is layout-sensitive ([`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.6b).

---

## Step 8 — Power panel LEDs (no GPIO)

Place on the **`power`** sheet near the regulators, visible on the **digital edge** silk ([`h02`](h02_custom_hat_shield.md) §4).

**KiCad:** `Device:LED` · **Footprint:** `LED_SMD:LED_0603_1608Metric` · **Reference** `D?` · **Value** = descriptive name (below) · **silk** = short rail label · **MPN:** **LTST-C190KGKT** (all four greens).

| Ref (doc) | Value (example) | Silk | Wiring | Series R MPN |
|-----------|-----------------|------|--------|--------------|
| **D_5VIN** | `LED_5VIN_Green` | **5V** | `+5V_IN ──[2.2 kΩ]──► anode; cathode ──► DGND` — Pi **5 V** present | **RC0603FR-072K2L** (**R29**) |
| **D_3V3D_OK** | `LED_3V3D_OK_Green` | **3V3** | `+3V3D ──[470 Ω]──► anode; cathode ──► DGND` — **TPS73633** output | **RC0603FR-07470RL** |
| **D_6V5_PG** | `LED_6.5VA_PG_Green` | **6.5V** | **TPS61252 PG** (pin 5) — see below | **RC0603FR-07470RL** (**R34**) |
| **D_5VA_OK** | `LED_5VA_OK_Green` | **5VA** | `+5VA ──[2.2 kΩ]──► anode; cathode ──► DGND` — **ADP7142** **+5VA** | **RC0603FR-072K2L** (**R31**) |

### **D_6V5_PG** (`LED_6.5VA_PG_Green`) on **TPS61252 PG** (pin 5)

The **PG** pin is an **open-drain** output: **released (high-Z)** when boost **VOUT** is within the PG window (~95–105 % of setpoint per the TPS61252 datasheet §10.3.6), **pulled low to GND** when VOUT is out of regulation or the part is in fault/UVLO.

Keep TI **R4 = 1 MΩ** from **PG ↔ VOUT** (Step 3). Add **R33** + LED with pull-up to **`+5V_IN`** (do not drive the indicator from **~6.5 V VOUT**):

```text
VOUT (boost) ──[R4 1 MΩ]── PG (pin 5) ──[R33 15 kΩ]── +5V_IN
                              │
                              └──[R34 470 Ω]──► D_6V5_PG anode ── cathode ──► DGND
```

| State | PG pin | **D_6V5_PG** |
|-------|--------|----------------|
| Boost **OK** (~6.5 V in regulation) | High-Z (pulled ~5 V via **R33**) | **On** |
| Boost fault / VOUT low / UVLO | Driven **low** | **Off** |

### **D_5VA_OK** (`LED_5VA_OK_Green`) on **`+5VA`**

Tie the LED to the **`+5VA`** net at the **ADP7142 VOUT** side (same net as hierarchical export to `emg`). This confirms the **5.0 V analog rail** is present after the LDO — what the ADS1299 **AVDD** uses.

```text
+5VA ──[2.2 kΩ]──► D_5VA_OK anode ── cathode ──► DGND
```

Use **2.2 kΩ** (same as **D_5VIN**) for ~1–2 mA at 5 V — negligible vs the ~35–65 mA **+5VA** budget. This LED does **not** prove regulation margin (only that the net is high); pair it with **D_6V5_PG** (boost OK) for the full chain: Pi **5 V** → boost **~6.5 V** → **+5VA**.

**Bring-up order (typical):** **D_5VIN** → **D_3V3D_OK** → **D_6V5_PG** → **D_5VA_OK**. If **D_6V5_PG** is on but **D_5VA_OK** is off, suspect **ADP7142**, boost headroom, or a short on **`+5VA`**.

**PG caveat:** If **EN** is low while **`+5V_IN`** is already up, **PG** can float high and **D_6V5_PG** may glow with the boost off — on v1 **EN** is tied to **VIN**.

All four LED cathode returns go to **`DGND`**, not **`AGND`** (panel indicators only; ~1–2 mA each).

---

## Step 6 — ERC

- All power pins driven; `PWR_FLAG` on `+5VA`, `+3V3D`, `+5V_IN` outputs.
- No short between `+5VA` and `+3V3D`.
- Boost **GND** on `DGND`, not `AGND`.
- **TPS73633 EN** driven high (tied to `+5V_IN` / IN); **NR** has **10 nF** to **DGND**.
- No floating **EN** pins on TPS61252, TPS73633, or ADP7142.

---

## Hierarchical labels (power sheet ↔ root)

| power sheet | Direction | Root `jtsmm_hat` connection |
|-------------|-----------|------------------------------|
| `+5V_IN` | in | From `hat_plus` `HAT_5V0` |
| `GND_IN` | in | From `hat_plus` `HAT_GND` |
| `+5VA` | out | To `emg` `+5VA` |
| `+3V3D` | out | To `mcu`, `emg`, `trigger_io`, `imu_connectors` |
| `AGND` | out | To `emg` `AGND` (and IMU shield policy) |
| `DGND` | out | To `mcu`, `emg`, `trigger_io`, `imu_connectors` |

---

## Optional / not v1 (DNP)

Per [`h07`](h07_power_grounding_safety.md) §1.4 — **do not populate** on default BOM:

- `RING_PWR`, `RING_EN`, high-current IR ring MOSFETs
- Any 5 V **ring** supply routed through the HAT

Leave footprints **DNP** on assembly drawing if present on a dev spin.

---

## Pre-capture checklist

| Check | Pass criteria |
|-------|----------------|
| Net names | `+5VA` and `+3V3D` exported (not `+3V3A`) |
| Boost | Table 3: **C1@VIN**, **C3×2@VOUT**, **C2**, **L1 VIN–SW**, **R1/R2** 795k/180k, **R3/R4** |
| ADP7142AUJZ-5.0 | **5.0 V** fixed; SENSE tied to **+5VA** at load |
| TPS73633DBV | `+3V3D` @ 3.3 V; **EN = IN**; **NR — 10 nF — DGND** |
| GND domains | `AGND` / `DGND` separate; one PCB star documented |
| PWR_FLAG | On each generated rail |
| Panel LEDs | **D_5VIN**, **D_3V3D_OK**, **D_6V5_PG** (PG + **R33** → **`+5V_IN`**), **D_5VA_OK** on **`+5VA`** — all cathodes **`DGND`** (Step 8) |
| Current budget | ~220 mA typical @ 5 V in ([`h07`](h07_power_grounding_safety.md) §1.1) |

---

## Step 7 — PCB layout (short)

- Regulator input/output caps **at the IC** ([`h07`](h07_power_grounding_safety.md) §1.5).
- Boost in **digital corner**; `+5VA` route short and quiet into analog zone ([`h02`](h02_custom_hat_shield.md) §3).
- Star tie location: under ADS1299 — see [`h07`](h07_power_grounding_safety.md) §2.2.

---

## Suggested capture order

1. Hierarchical labels + `PWR_FLAG`
2. TPS73633DBV → `+3V3D` (wire **EN**, **NR** first)
3. Boost → ADP7142AUJZ-5.0 → `+5VA` (**R4** on **PG**)
4. Step 8 panel LEDs (after `+3V3D` and **`+5VA`** nets exist): **D_6V5_PG** on **PG**, **D_5VA_OK** on **`+5VA`**
5. ERC → annotate boost FB resistor values on schematic
