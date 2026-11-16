<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h03 — EMG Schematic Build Guide (KiCad capture walkthrough)

A practical, step-by-step companion to [`h03_emg_frontend.md`](h03_emg_frontend.md) for **capturing the EMG front-end schematic in KiCad 10**. `h03` is the *what and why* (topology, values, rationale); this file is the *how* (symbols, pin-by-pin wiring, decoupling diagrams, ERC). Work inside `hardware/jtsmm_hat/sheets/emg.kicad_sch`.

> **Authoritative sources.** Values come from [`h03_emg_frontend.md`](h03_emg_frontend.md) §3–§9, the RP2350A pin map in [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1, and grounding/power in [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2. Exact ADS1299 capacitor values are governed by the **ADS1299 datasheet (SBAS499)** typical-connection figures; where this guide cannot verify a value it is flagged **“confirm SBAS499.”**

## How to read the diagrams

```text
A ──┬── 100nF ── B      means: a 100 nF capacitor from node A to node B
    └──► PIN            means: node A also connects to PIN
[100Ω]                  a series component in-line
╫                       the two windings of the 4-terminal common-mode choke
```

ADS1299 pin numbers below are the standard **TQFP-64** assignments (same as the OpenBCI symbol and the KiCad `Analog_ADC:ADS1299` symbol). Verify against the symbol in your sheet before wiring.

---

## Step 0 — Library and footprint setup (do this first)

> **Footprint audit:** cross-check assignments against [`h09_footprint_master_table.md`](h09_footprint_master_table.md).

Resolve symbols before capturing so nothing rescues to a placeholder.

**Symbols already working** (KiCad global libs): `Analog_ADC:ADS1299`, `Reference_Voltage:REF5025AD`, `Power_Protection:TPD4E05U06DQA`, `Device:R` / `Device:C` / `Device:L`, `Device:Filter_EMI_CommonMode`.

> **ESD default is now `TPD4E1B06`** (precision-analog, the OpenBCI part). It may not ship in the KiCad global libs — import the symbol from the OpenBCI library (`OpenBCI_hardware_KiCAD`) or draw a 6-pin one (IO1–IO4 + GND + NC), or fall back to the global-lib `TPD4E05U06DQA` (also GND-referenced, drop-in topology).

**Symbols to sort out:**

| Part (catalog) | Symbol to use | Note |
|----------------|---------------|------|
| REF5025AIDR (`H-IC-02`) | **DNP — do not populate** | Default is **ADS1299 internal 4.5 V ref** ([`h03`](h03_emg_frontend.md) §9). If `U3` is already on the sheet, mark **DNP** in schematic and BOM. Optional alt: **REF5045AIDR** (4.5 V external) + `CONFIG3.PD_REFBUF = 0` (power down on-chip ref buffer). |
| EMG patient connector | **Pick one** ([`h03`](h03_emg_frontend.md) §10.2): **0** `PinHeader` ×13 (bring-up) · **A** trunk · **B** `J_L`/`J_R`/`J_DIG`/`J_DRL` · **C** `J_CH1`…`J_DRL` · **F** `DIN42802_Jack` ×13 | **First iteration: Option 0.** Filter nets unchanged. See §“Option 0 — connector capture” below. |
| DLW21SN900SQ2L (`H-IC-10`) | `jtsmm:DLW21SN900SQ2L` or `Device:Filter_EMI_CommonMode` | Footprint **`jtsmm:FIL_Murata_DLW21SN900SQ2L`**. |
| AGND/DGND star tie | `Device:NetTie_2` | Ref **`NT_GND_TIE1`**; pin **1 → `AGND`**, pin **2 → `DGND`**. PCB between **`U_ADS1` pads 32 and 31** ([`h07`](h07_power_grounding_safety.md) §2.2). |
| ESD array (`H-IC-09`) | `Power_Protection:TPD4E1B06` (4-ch bidirectional TVS; same part as OpenBCI) | **Default** per [`h03`](h03_emg_frontend.md) §4 — 0.7 pF, 0.5 nA leak, GND-referenced (GND→AGND, no VCC). `TPD4E05U06DQA` is a drop-in GND-ref alternative; `TPD4E004` is **VCC-referenced** (needs VCC→+5VA + 0.1 µF) — not pin-for-pin swappable. |
| Series-R array (`H-PAS-02`, opt.) | Bourns `CAY16-1000F4LF` (4× 100 Ω ±1 %) — generic `Device:R_Pack04` if no exact symbol | **Optional density/CMRR option** ([`h03`](h03_emg_frontend.md) §4.1): 3 arrays = 12 EMG legs. 4 *isolated* resistors. Convex 1206 land. |
| C0G CM-cap array (`H-PAS-02`, opt., DNP) | TDK `CKCA43C0G1H471K100AA` (4× 470 pF C0G) — generic `Device:C_Pack04` | Only for the **optional CM caps** to AGND; **C0G**, DNP by default. The differential `CFilt` stays **C0G** (discrete 4.7 nF or 1 nF C0G array) — **never X7R**. |

**Footprint mapping** (assign in Step 5, decide now):

| Part | Footprint |
|------|-----------|
| ADS1299IPAG | `Package_QFP:TQFP-64_10x10mm_P0.5mm` (already set) |
| NT_GND_TIE1 | `NetTie:NetTie-2_SMD_Pad0.5mm` (alt `NetTie-2_THT_Pad0.3mm`) — **PCB feature, no BOM line** |
| REF5025AIDR (DNP) | `Package_SO:SOIC-8_3.9x4.9mm_P1.27mm` — leave unpopulated on default build |
| TPD4E1B06DRLR | **`jtsmm:TPD4E1B06DRLR_SOT50P160X60-6N`** (default MPN suffix DRLR) |
| CAY16-1000F4LF (R array) | **`jtsmm:Bourns_RESCAF80P320X160X60-8N`** |
| CKCA43 / CKCL44 (C array) | TDK 4-element array land (1206 = CKCA43, 0805 = CKCL44); create in `libs/jtsmm.pretty` if not in stock libs |
| DLW21SN900SQ2L | **`jtsmm:FIL_Murata_DLW21SN900SQ2L`**. Alt MPN **DLW21SN121SQ2L** (120 Ω). LCSC **C97856** (900). |
| 100 µF VCAP1 | **`Capacitor_Tantalum_SMD:CP_EIA-3528-21_Kemet-B`** (C0G/tantalum) — **never 0603** |
| 100 Ω / 4.7 nF / 1 µF | `R_0603` / `C_0603` (0603 for hand-rework) |
| 100 nF HF bypass | **`C_0402_1005Metric`** — **GRM155R71C104KA88D** (C12–C14,C22 target; optional **0603→0402**) |
| 10 µF bulk | `Capacitor_SMD:C_0805_2012Metric` |
| EMG patient connector | **H-CON-01CY:** **`PinHeader_1x06`** ×2 (`J_EMG_N1`, `J_EMG_P1`) + **`PinHeader_1x01`** ×1 (`J_EMG_DRL1`). Alternatives: **H-CON-01T/R/C:** LEMO/ODU/Binder panel lands. **H-CON-01F:** DIN 42802 jack land |

The project `sym-lib-table` / `fp-lib-table` already register the `jtsmm` libraries; add custom footprints into `libs/jtsmm.pretty`.

---

## Step 1 — Capture the ADS1299 core (power + reference + control)

Build outward from the chip. Use the **unipolar 0/5 V** scheme: `AVDD = +5 V`, `AVSS = AGND = 0 V` — **there is no negative rail**. The 5 V analog supply is a **hard requirement** (AVDD−AVSS = 4.75–5.25 V, SBAS499 §7.3). **VREF = 4.5 V** from the **internal** ADS1299 reference (full dynamic range — see [`h03`](h03_emg_frontend.md) §1/§3/§9, [`h07`](h07_power_grounding_safety.md) §1).

### 1a. Analog supply + decoupling (AVDD / AVSS)

**AVDD** pins **19, 21, 22, 56, 59** tie to `+5VA` (boost + **ADP7142AUJZ-5.0**; **SENSE/ADJ** at the ADC). **AVDD1** pin **54** star-ties to the AVDD cluster ([`h03`](h03_emg_frontend.md) §9). **AVSS** pins **20, 23, 32, 57** and **AVSS1** pins **53, 58** tie to `AGND`.

Per the SBAS499 **Pin Functions** table, **each** AVDD pin gets its own **1 µF to AVSS** (the §11 text adds a 10 µF + 0.1 µF bulk per supply). `AVDD1`/`AVSS1` (charge-pump supply) must be **star-connected** to the `AVDD`/`AVSS` pins.

```text
   +5VA ──┬── 10µF ── AGND          (bulk per analog rail, §11)
          ├── 0.1µF ── AGND         (HF bypass)
          ├── 1µF ── AGND           (at pin 19)
          ├── 1µF ── AGND           (at pin 21)
          ├── 1µF ── AGND           (at pin 22)
          ├── 1µF ── AGND           (at pin 56)
          ├── 1µF ── AVSS (58)      (pin 59 charge-pump AVDD, per §9 table)
          └──► AVDD 19,21,22,56,59

   +5VA ──★── 1µF ── AGND  ──► AVDD1 (54)    (star tie to AVDD cluster)

   AVSS  (20,23,32,57)  ──► AGND
   AVSS  (58, charge-pump) ──► AGND   (star tie to AVSS)
   AVSS1 (53)            ──► AGND     (= 0 V; no negative rail in the unipolar scheme)
```

> **Datasheet rule (SBAS499 §11):** “star connect AVDD1 to the AVDD pins and AVSS1 to the AVSS pins.” On the schematic the redundant AVDD/AVSS pins are stacked/hidden on one stub, so you wire `+5VA`/`AGND` once — but on the **PCB** place a **1 µF at each physical AVDD pad** and route the AVDD1/AVSS1 star tie explicitly. Per SBAS499 §12.1, place each bypass cap on the same layer at the pin with **no via between cap and pin**.

### 1b. Digital supply + decoupling (DVDD / DGND)

Per SBAS499, **each** DVDD pin gets **1 µF to DGND** (plus a 10 µF + 0.1 µF bulk on the rail).

```text
   +3V3D ──┬── 10µF ── DGND          (bulk, §11)
           ├── 0.1µF ── DGND         (HF bypass)
           ├── 1µF ── DGND           (ONE PER DVDD pin: 48, 50)
           └──► DVDD   (ADS1299 48, 50)

   DGND (33, 49, 51) ──► DGND
```

### 1c. Reference — internal 4.5 V (VREFP / VREFN)

**Default:** use the ADS1299 **on-chip 4.5 V reference** for **full datasheet dynamic range** (±VREF/gain with **VREF = 4.5 V**). No external reference IC.

- Tie **VREFN (pin 25)** → **AGND**.
- Decouple **VREFP (pin 24)** with **≥10 µF + 0.1 µF** to **VREFN** (SBAS499: “minimum 10-µF capacitor to VREFN”).
- In firmware: **`CONFIG3.PD_REFBUF = 1`** → internal reference buffer **enabled** (SBAS499 Table 15: `0` = power-down, `1` = enable — the bit name is misleading). Power-up flow uses `WREG CONFIG3, 0xE0` for internal ref (Figure 67).
- **Do not populate REF5025 (U3)** — 2.5 V would waste ~44 % of the ADC span; mark **DNP** on the schematic BOM.

```text
   VREFN (25) ─────────────────────────────► AGND

   VREFP (24) ──┬── 10µF ──┐
                ├── 0.1µF ┴── VREFN (= AGND)     place AT pin 24 (no vias between cap and pin)
                └── (on-chip buffer → 4.5 V across VREFP–VREFN)

   (no REF5025 / REF5045 — U3 DNP on default build)
```

> **Optional external 4.5 V:** populate **REF5045AIDR** instead of REF5025, drive `VREFP` from its output, set **`CONFIG3.PD_REFBUF = 0`** (power down the on-chip buffer), and decouple REF5045 Vin from `+5VA` per its datasheet. Only use if you have a measured reason to disable the internal buffer.

### 1d. Internal-LDO / charge-pump reservoirs (VCAP1–VCAP4)

These bypass the ADS1299’s internal regulators/charge pump. **Values are now fixed from the SBAS499 Pin Functions table** — each cap goes to **AVSS** and must sit as close to the package as possible:

```text
   VCAP1 (pin 28) ──┬── 100µF ── AGND    (analog bypass; SBAS499 = 100 µF to AVSS)
   VCAP2 (pin 30) ──┬── 1µF   ── AGND    (1 µF to AVSS)
   VCAP3 (pin 55) ──┬── 1µF ∥ 0.1µF ─ AGND   (parallel 1 µF + 0.1 µF to AVSS)
   VCAP4 (pin 26) ──┬── 1µF   ── AGND    (1 µF to AVSS)
```

> **SBAS499 exact values (Pin Functions):** VCAP1 = **100 µF**, VCAP2 = **1 µF**, VCAP3 = **1 µF ∥ 0.1 µF**, VCAP4 = **1 µF**, all to AVSS.
> - **VCAP1 dielectric:** use a non-ferroelectric type — **C0G/NPO (class 1) or tantalum** — not X7R/X5R. SBAS499 §11 warns the piezoelectric effect of class-2/3 ceramics injects noise on VCAP1 (matters most with the internal reference; harmless to over-specify).
> - **Power-up timing:** VCAP1’s 100 µF sets the reference RC ramp. Issue `RESET` only **after tPOR (218 × tCLK) _or_ after VCAP1 > 1.1 V, whichever is longer** (SBAS499 §11.1, Fig. 76). Capture this in the firmware bring-up ([`h05`](h05_microcontroller_firmware.md)).

### 1e. AGND/DGND star tie (`NT_GND_TIE1`)

The **only** on-board connection between **`AGND`** and **`DGND`** is a KiCad net tie at the ADS1299 ([`h07`](h07_power_grounding_safety.md) §2.2). Capture it on the **`emg`** sheet:

```text
   AGND ──┬── NT_GND_TIE1 (Device:NetTie_2) pin 1
          │
   DGND ──┴── NT_GND_TIE1 pin 2
```

- Ref: **`NT_GND_TIE1`**
- Footprint: **`NetTie:NetTie-2_SMD_Pad0.5mm`** (preferred for pad pitch at `U_ADS1`)
- On the **PCB**, place between **`U_ADS1` pad 32 (`AVSS` / `AGND`)** and **pad 31 (`RESV1` / `DGND`)** — short traces on `F.Cu`, one via per net into the `In1.Cu` planes ([`h10`](h10_kicad_pcb_layout_guide.md) §3.1).
- **Do not** add a second tie or a 0 Ω resistor elsewhere.

### 1f. Control + SPI → hierarchical labels

The hierarchical labels already exist on the sheet (the inter-sheet contract). Attach each ADS1299 pin to its label:

| ADS1299 pin (No.) | Hierarchical label | RP2350A (per [`h05`](h05_microcontroller_firmware.md) §1.1) |
|-------------------|--------------------|-----------------------------------------------------------|
| SCLK (40) | `ADS_SPI_CLK` | GPIO2 |
| DIN (34) | `ADS_SPI_MOSI` | GPIO3 |
| DOUT (43) | `ADS_SPI_MISO` | GPIO0 |
| ~CS~ (39) | `ADS_SPI_CS` | GPIO5 |
| ~DRDY~ (47) | `ADS_DRDY` | GPIO6 (polled input) |
| ~RESET~ (36) | `ADS_RESET` | GPIO7 |
| ~PWDN~ (35) | `ADS_PWDN` | GPIO8 |
| START (38) | `ADS_START` | GPIO9 |

Per SBAS499, the two-state mode pins must be strapped **through a ≥10 kΩ resistor** (not tied hard):

```text
   CLKSEL (52) ──[10kΩ]── +3V3D     (HIGH = internal oscillator; SBAS499 needs ≥10kΩ strap)
   CLK    (37) ── leave open for internal-clock mode (driven only if using an external clock)
```

**Unused pins** — terminate exactly per the SBAS499 Pin Functions table so ERC is clean:

```text
   GPIO1–4 (42,44,45,46) ──[10kΩ]── DGND   (SBAS499: "Connect to DGND with a ≥10-kΩ resistor if unused")
   RESV1 (31)            ── DGND            (SBAS499: "connect directly to DGND")
   DAISY_IN (41)         ── DGND            (SBAS499 §10.1.4.2: short to DGND if no daisy-chain)
   RESERVED (64)         ── open            (SBAS499: "leave as open circuit")
   NC (27, 29)           ── open            (no-connect flags)
   SRB1 (17), SRB2 (18)  ── leave unless using an SRB reference montage
```

---

## Step 2 — Build ONE input channel, then array it

[`h03`](h03_emg_frontend.md) §4 is the authoritative topology. Build **CH1** (IN1P/IN1N) fully, verify, then copy for CH2–CH6.

> **SBAS499 §10.2.2 input-filter rule (this changed the topology).** The datasheet is explicit: *“form the filter using a **differential capacitor** CFilt that shunts the inputs **rather than individual RC filters whose capacitors shunt to ground**. The differential configuration significantly improves common-mode rejection because it removes dependence on component mismatch.”* For high-frequency EMI, optional common-mode caps to ground are allowed but *“should be **10 or 20× smaller** than the differential capacitor.”*

So the primary anti-alias/RFI cap is **differential across INxP–INxN**, not per-leg-to-ground:

```text
  J_EMG pin (IN1P) ──[100Ω 1%]──┬──[ESD ch]──── choke in1 ╫ out1 ──┬──► ADS1299 IN1P (16)
                                 │    GND──AGND   DLW21SN900SQ2   │
                              (470pF)                            4.7nF C0G   ← DIFFERENTIAL CFilt
                            opt. CM──AGND                            │
  J_EMG pin (IN1N) ──[100Ω 1%]──┬──[ESD ch]──── choke in2 ╫ out2 ──┴──► ADS1299 IN1N (15)
                                 │    GND──AGND
                              (470pF)
                            opt. CM──AGND

  J_EMG shell ── AGND   (one tie per patient connector; all architectures)
```

> **Connector note:** **`J_EMG pin`** / **`INxP` header pin** = the HAT contact for that signal. Pin maps: [`h03`](h03_emg_frontend.md) §10.2. `emg.kicad_sch` carries **Option 0**: two 1×6 pin headers (`J_EMG_P1` = `IN1P`…`IN6P`, `J_EMG_N1` = `IN1N`…`IN6N`) and a 1-pin header (`J_EMG_DRL1`) — see the table below.

---

## Option 0 — connector capture (OpenBCI / Cyton-style bring-up)

Option 0 is the connector variant of the **first populated PCB** (revision 0.0.1). The connector options differ only in the **patient-side connector symbols**, **footprints**, **silkscreen**, and **SHLD nets**; the ADS1299, filter, ESD, choke, and DRL/bias networks are the same. The table lists Option 0 as captured on `emg.kicad_sch` against the DIN-jack variant (Option **F**).

| # | Schematic / PCB item | Option **0** (as captured) | Option **F** (13× DIN 42802 jacks) | Same in both |
|---|----------------------|----------------------------|------------------------------------|--------------|
| 1 | Patient connector symbol | **`Connector:Conn_01x06_Pin`** ×2 + **`Connector:Conn_01x01_Pin`** ×1 | `jtsmm:DIN42802_Jack` ×13 | ADS1299 `U_ADS1`, decoupling, SPI, `DRDY` |
| 2 | Connector pins per signal | **1 pin** — on the **`SIG`** net (`IN1P`, `IN1N`, … `DRL`) | `SIG` + `SHLD` (2 pins each) | `IN1P`→`IN1N` filter: 100 Ω, TPD4E1B06, DLW21, 4.7 nF `CFilt` |
| 3 | Shield net at connector | none (no cable shield in Option 0) | Each `SHLD` → `AGND` | Star **`AGND`** under ADS1299; digital split ([`h07`](h07_power_grounding_safety.md) §2.2) |
| 4 | Footprint | **`PinHeader_1x06_P2.54mm_Vertical`** ×2 (`J_EMG_N1`, `J_EMG_P1`) + **`PinHeader_1x01_P2.54mm_Vertical`** ×1 (`J_EMG_DRL1`) | DIN 42802 panel jack (THT) | Murata DLW21, TPD4E1B06 lands |
| 5 | Reference designators | `J_EMG_P1` (pins 1–6 = `IN1P`…`IN6P`), `J_EMG_N1` (pins 1–6 = `IN1N`…`IN6N`), `J_EMG_DRL1` | one jack per signal | `R`, `C`, `FL`, ESD refs |
| 6 | Silkscreen | Reference designators only; signal per pin as in the [`h03`](h03_emg_frontend.md) §10.2.0 table | — | — |
| 7 | DRL path | `J_EMG_DRL1` → 100 Ω → TPD → DLW21 → 220 kΩ → **`IN8N`** (net **`IN8N_BIASOUT`**) | **Same** topology from the DRL jack | Bias loop at IC: `BIASIN↔BIASOUT`, 1 MΩ∥1.5 nF, `BIASREF→AGND` |
| 8 | CH7 / CH8 | **IN8N** = DRL; **IN7P/IN7N** and **IN8P** not connected; **no +5VA tie** | Same | `MUX8=111` for DRL |
| 9 | BOM field | **`H-CON-01CY`** — 2.54 mm pin header strip | `H-CON-01F` / DIN jack MPN | All `H-IC-*`, `H-PAS-*` analog parts |
| 10 | 3D / panel | Two parallel 1×6 pin headers and one single pin on the patient edge ([`h02`](h02_custom_hat_shield.md) §2.1) | Large DIN jack keep-out | Pi 40-pin keep-out |

**Bench wiring:** plug [OpenBCI-class snap cables](https://shop.openbci.com/collections/frontpage/products/emg-ecg-snap-electrode-cables) (Dupont female) onto the headers; map colors to channels in a bring-up note. Keep leads **short** (≤ 1 m) and connect **DRL** to a mastoid electrode before judging 50 Hz noise.

**Option B** has **4× LEMO 1B.304 + 1× 1B.302** in place of the 13 header pins — filter nets and ADS1299 pinout stay the same; only connector symbols and pin grouping change.

---
- **R 100 Ω 1 %** in each leg (ESD current limit + forms the RC roll-off with CFilt). For density/CMRR you can fold the 12 legs into **3× CAY16 4-element resistor arrays** ([`h03`](h03_emg_frontend.md) §4.1) instead of discretes.
- **TPD4E1B06** channel tap on each leg; tie the array **GND pin → AGND** (GND-referenced; no VCC pin). One 4-channel array covers 4 lines. (Place it as close to the **patient trunk connector** as possible, per its datasheet.)
- **CFilt 4.7 nF C0G — differential, one per pair** across IN_P/IN_N (this is the main change vs. the old per-leg-to-ground caps). C0G to avoid voltage-coefficient distortion.
- **Optional CM caps**: if bench EMI demands it, add **≤470 pF** (≈ CFilt/10) from each leg to AGND. **Default = DNP**, since the CM choke already rejects common-mode RF and to-ground caps re-introduce a CMRR-vs-mismatch dependence.
- **DLW21SN900SQ2**: both legs of the pair pass through the one choke into `IN1P`/`IN1N`.

**Then array it:** select the whole CH1 group (**two patient connector pins: P and N**) → **Copy/Paste ×5** for CH2–CH6 → reassign inputs to `IN2…IN6 P/N` and update connector pin labels per [`h03`](h03_emg_frontend.md) §10.2 (trunk pin map for **A**; `J_L`/`J_R`/`J_DIG` for **B**; etc.).

**ESD array allocation (3× quad covers 12 EMG lines):**

| Array | Protects |
|-------|----------|
| ESD-A | IN1P, IN1N, IN2P, IN2N |
| ESD-B | IN3P, IN3N, IN4P, IN4N |
| ESD-C | IN5P, IN5N, IN6P, IN6N |

**CH7 / CH8** (as-built): **IN8N** = DRL electrode (`J_EMG_DRL1` → filter → **IN8N**; firmware **`MUX8=111`**). **IN8P** = **NC**. **IN7P/IN7N** = **NC** (no filter, **no patient pin**, **no +5VA tie**) — BIAS readback on CH7 is an internal mux setting (h03 §5.3). Power down unused channels in **`CHnSET`** when not in use; do **not** use the old “tie spare inputs to +5VA” recipe on this rev.

**Quantities as captured** (`H-PAS-02` / `H-IC-09` / `H-IC-10`): **13× 100 Ω** (12 EMG legs in 3× CAY16 arrays `RN1`–`RN3` + `R5` on DRL), **7× 4.7 nF C0G** (6 differential CFilt `C3`–`C8` + `C2` on the DRL path to AGND), **7 chokes** (6 EMG `FL1`–`FL6` + DRL `FL7`), **4 ESD arrays** (`D5`–`D7` for the 12 EMG lines, `D8` with one channel for DRL), plus **H-CON-01CY** (12 EMG + **DRL** on IN8N).

---

## Step 3 — DRL / bias-drive network

From [`h03`](h03_emg_frontend.md) §5 (SBAS499 Fig. 33/34). KiCad refs on `emg.kicad_sch`: `BIASOUT` (63), `BIASIN` (62), `BIASREF` (60), `BIASINV` (61). **`BIAS_SENSP` / `BIAS_SENSN` are registers** — enable CH1–CH6 in firmware only.

**Firmware (no extra parts):** after `RESET`, set:

- `CONFIG3 = 0xEC`:
  - `PD_BIAS = 1` — bias amplifier on (`0` leaves it powered down)  
  - `BIASREF_INT = 1` — bias reference = internal (AVDD+AVSS)/2 = 2.5 V; pin 60 (→ **AGND**) is then unused. **Not** `0xE4`: with `BIASREF_INT = 0` the amp drives the body to 0 V, the negative rail of this unipolar supply (SBAS499C §10.1.3)  
  - `PD_REFBUF = 1` — internal 4.5 V reference buffer on (PGA ref, separate from bias)  
- `BIAS_SENSP` / `BIAS_SENSN` — CH1–CH6 only; CH7/CH8 off for bias derivation  
- **`CH8SET.MUX = 111`** — route bias drive to **IN8N** (DRL electrode) in normal acquisition  
- Optional bench: **`CONFIG3.BIAS_MEAS = 1`**, **`CH7SET.MUX = 010`/`111`** — read **BIASIN** on CH7 vs **BIASREF** (Fig. 34)

### Full signal path (as-built `emg.kicad_sch`)

Patient DRL lands on **`IN8N`**, not **`BIASOUT`**. On-chip loop: **BIASIN ↔ BIASOUT** with **1 MΩ ∥ 1.5 nF** to **BIASINV**.

```text
   J_EMG_DRL1
   DRL ──[100Ω R5]──┬──[TPD4E1B06]──── DLW21 FL7 pin1 ╫ pin2 ──┬──[220kΩ R6]──► IN8N (pin 1)
                    │              (GND→AGND)  (pins 3–4 NC)  └── 4.7nF C2 ── AGND
               TP_BIASOUT1 (tap on IN8N net)

   At U_ADS1 (bias pins):
   BIASREF (60) ── AGND
   BIASOUT (63) ──┬──[1MΩ R8 ∥ 1.5nF C78]──► BIASINV (61)
                  └──wire────► BIASIN (62)

   IN8P ── NC     IN7P/IN7N ── NC
```

**Parts (as-built refs):**

| Ref | Value | Connection |
|-----|-------|------------|
| **`J_EMG_DRL1`** | 1-pin header | Patient DRL electrode |
| **R5** | 100 Ω | DRL pin → TPD |
| **D8** | TPD4E1B06 (1 IO) | DRL ESD — one channel of a quad array, the others unused |
| **FL7** | DLW21SN900SQ2 | **1→2** in series; **3–4** NC |
| **R6** | 220 kΩ | Choke output → **IN8N** |
| **C2** | 4.7 nF C0G | Patient-side node → **AGND** |
| **R8** | 1 MΩ | **BIASOUT** → **BIASINV** (parallel with C78) |
| **C78** | 1.5 nF | **BIASOUT** → **BIASINV** |
| **TP_BIASOUT1** | test pad | **IN8N / DRL** net — not raw **BIASOUT** |

**KiCad wiring checks before ERC:**

1. **DRL path** terminates at **IN8N** — **not** at **BIASOUT**.
2. **Wire `BIASIN` (62)** directly to **`BIASOUT`** (same net).
3. **`BIASREF` (60)** → **AGND** (not NC).
4. **`BIASINV` (61)** — **1 MΩ ∥ 1.5 nF** to **`BIASOUT`**.
5. **`BIASOUT` net** includes **R8**, **C78**, and **BIASIN** — **not** the DRL patient path.
6. **IN7P/IN7N/IN8P** — no-connect flags, no tie to **+5VA**; **IN8N** wired to DRL network.

---



1. Swap **every** remaining placeholder (`Conn_01x02`, `AudioJack2`) for real symbols first — leftovers skew the netlist and BOM.
2. **Tools → Annotate Schematic** (assign refs).
3. **Inspect → Electrical Rules Checker.** Expect to resolve:
   - `GPIO1–4` → **≥10 kΩ to DGND** (not bare no-connect flags); `RESV1` → **direct to DGND**; `DAISY_IN` → DGND; `NC`/`RESERVED` → **no-connect flags**; `CLKSEL` → ≥10 kΩ to +3V3D (per SBAS499 Pin Functions).
   - “Power input pin not driven” on `+5VA`/`+3V3D`/`AGND`/`DGND` → add a **`PWR_FLAG`** where each rail originates (likely the `power` sheet).
   - Pin-name typos / unintended shorts.
   Target **zero errors**; keep only intentional warnings.

---

## Step 5 — Footprints + push to PCB

1. **Tools → Assign Footprints** using the Step 0 table. Double-check ADS1299 land, Murata choke land, and **patient connector** land(s) per **H-CON-01CY/T/R/C/F** ([`h03`](h03_emg_frontend.md) §10.2).
2. **Tools → Update PCB from Schematic** to pull the EMG group onto the board.

---

## Step 6 — EMG PCB layout (analog-critical)

Driven by [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §3–§4 and [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2.

- **Stackup:** 4-layer (L1 analog / L2 ground / L3 power / L4 digital) per `H-PCB-01`. EMG front-end on the analog side.
- **Outline / patient connector:** per architecture ([`h02`](h02_custom_hat_shield.md) §2.1, [`h03`](h03_emg_frontend.md) §10.2) — **0:** 1×13 or 13×1 pin header (~33 mm); **A:** one ~M12; **B:** 4× M12; **C:** 7× M12; **F:** DIN row + tab. Silkscreen **`IN1P` … `DRL`**.
- **Partition:** ADS1299 analog inputs, REF, and input networks in one contiguous analog zone; DVDD/SPI on the digital side. Keep the SPI bus and RP2350A away from the input network.
- **Ground:** single `AGND`/`DGND` split with the **one** star tie **under the ADS1299 at the DGND pin region** ([`h07`](h07_power_grounding_safety.md) §2.2). Do **not** add a second tie.
- **Place** each per-channel network within ~5 mm of the ADS1299 input pins: **trunk pin** → R → ESD → [opt. CM] → choke → **differential 4.7 nF C0G** → `INxP`/`INxN`; short and **symmetric P/N** for CMRR. Route **`J_EMG` shell → AGND** at the connector.
- **Decoupling:** every cap from Step 1 at its pin (SBAS499: “place the capacitors for supply, reference, and VCAP1–VCAP4 as close to the package as possible”); a **1 µF at each physical AVDD/DVDD pad**; the **100 µF VCAP1** (C0G/tantalum) and **≥10 µF VREFP–VREFN** bank at the package; route the **AVDD1/AVSS1 star tie** explicitly.
- **Shield can (optional):** reserve analog keep-out / fence per **`H-MECH-01`** — **DNP on first bring-up**; part numbers and layout rules: [`h10`](h10_kicad_pcb_layout_guide.md) §5.4.
- **Route**, then **Inspect → Design Rules Checker** to zero DRC. Prioritize short, symmetric input pairs over controlled impedance (high-Z, quasi-DC). Off-board harness length and cable capacitance: [`h03`](h03_emg_frontend.md) §10.1.
- **Mounting holes:** switch the HAT holes to **M2.5** (Pi spec, [`h02`](h02_custom_hat_shield.md)); the template currently uses M3.

**SBAS499 §12.1 layout rules (apply throughout):**

- **Separate analog/digital** sections; route digital lines (SPI, USB, trigger) away from the analog inputs.
- The L2 plane **may** be split AGND/DGND (we do) — place analog signals over AGND, digital over DGND, and **join the split at the ADC** (the single star tie). 
- **Fill voids** on signal layers with ground fill; via-stitch to the adjacent plane.
- **Keep ground return paths continuous** — no plane cut or crossing trace that forces return current to detour under a signal.
- **Bypass caps on the same layer as the device, at the pin, with no via between the cap and the pin.**
- **Differential C0G cap (4.7 nF) across each INxP–INxN** at the package (high-quality C0G/NPO only).
- **Keep the +5VA boost (switcher) in the digital half**; only the LDO-cleaned 5 V crosses into the analog region.

Full point-by-point compliance: [`h07`](h07_power_grounding_safety.md) §2.4.

---

## Per-pin decoupling summary (quick reference)

All values from the **SBAS499 Pin Functions table** and §11 (caps to AVSS = AGND in the unipolar scheme):

| ADS1299 pin(s) | Net | Cap(s) to | Value | Source |
|----------------|-----|-----------|-------|--------|
| AVDD (19,21,22,56) | +5VA (5.0 V) | AVSS/AGND | 1 µF per pin (+ 10 µF + 0.1 µF rail bulk) | SBAS499 Pin Fn / §11 |
| AVDD (59, charge pump) | +5VA | AVSS pin 58 | 1 µF | SBAS499 Pin Fn |
| AVDD1 (54) | +5VA (★star) | AVSS1 | 1 µF | SBAS499 Pin Fn / §11 |
| AVSS/AVSS1 (20,23,32,57,58,53) | AGND | — | — | unipolar: 0 V |
| DVDD (48,50) | +3V3D | DGND | 1 µF per pin (+ 10 µF + 0.1 µF bulk) | SBAS499 Pin Fn |
| DGND (33,49,51) | DGND | — | — | — |
| VREFP (24) | (internal 4.5 V) | VREFN/AGND | ≥10 µF + 0.1 µF | SBAS499 Pin Fn; `PD_REFBUF=1` |
| BIASREF (60) | AGND | — | tie to AGND (unused) | `BIASREF_INT=1` (internal mid-supply) |
| BIASINV (61) | BIASOUT | — | 1 MΩ ∥ 1.5 nF (Fig. 33) | h03 §5 |
| BIASIN (62) | BIASOUT | — | direct tie | §10.1.2.2 |
| BIASOUT (63) | BIAS | — | no shunt cap | h03 §5 |
| VREFN (25) | AGND | — | — | tie to AGND |
| **VCAP1 (28)** | — | AVSS/AGND | **100 µF** (C0G/NPO or tantalum) | SBAS499 Pin Fn |
| VCAP2 (30) | — | AVSS/AGND | 1 µF | SBAS499 Pin Fn |
| VCAP3 (55) | — | AVSS/AGND | 1 µF ∥ 0.1 µF | SBAS499 Pin Fn |
| VCAP4 (26) | — | AVSS/AGND | 1 µF | SBAS499 Pin Fn |
| CLKSEL (52) | +3V3D | — | ≥10 kΩ strap HIGH | SBAS499 Pin Fn |
| GPIO1–4 (42,44,45,46) | DGND | — | ≥10 kΩ pull-down (if unused) | SBAS499 Pin Fn |
| RESV1 (31) | DGND | — | direct tie | SBAS499 Pin Fn |
| DRL (`J_EMG_DRL1`) | via filter | AGND | 100 Ω → TPD → **DLW21 (1→2)** → 220 kΩ → **IN8N**; 4.7 nF C0G to AGND | h03 §5 |
| **`J_EMG` shell** | AGND | — | one shield tie for all channels | h07 §2.3 |
| CFilt (per input pair) | INxP–INxN | differential | 4.7 nF C0G | SBAS499 §10.2.2 |
| REF5025 / REF5045 (U3) | — | — | **DNP** (default) | internal 4.5 V ref used |

---

## Suggested order to start

1. **Step 0** library/footprint setup — add patient connector symbol(s) in `jtsmm` ([`h03`](h03_emg_frontend.md) §10.2). **Option 0:** standard KiCad `PinHeader` — no custom symbol required.
2. **Step 1 + Step 3** — ADS1299 core (power, reference, VCAPs, control) and DRL: small, high-value, gets the chip “alive” in the netlist.
3. **Step 2** — one channel → array (the bulk of the capture).
4. **ERC → footprints → layout.**

> ✅ All decoupling, reference, VCAP, and termination values above are taken from **SBAS499** (Pin Functions, §11, Figure 77, §10.2.2). **VREF = 4.5 V internal** is the default reference (full dynamic range). Deliberately-open choices: **optional CM input caps** (DNP default); **optional external REF5045** instead of internal ref.

---

## Step 11 — EMG lead-off indicator (on `mcu` sheet)

**D_EMG_LO** is **not** captured on `emg.kicad_sch` — it lives on **`mcu`** next to the other panel LEDs ([`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) Step 6).

| Ref | GPIO | Footprint | Meaning |
|-----|------|-----------|---------|
| **D_EMG_LO** | **29** | `LED_SMD:LED_0603_1608Metric` + **470 Ω–1 kΩ** | Firmware lights when **`ADS_STATUS`** reports any lead-off bit |

Optional: add a **summary** LED on the **patient-connector edge** wired to hierarchical **`LED_EMG_LO`** from `mcu` if you extend the root net to a remote 0603 — default v1 keeps it on the **digital corner** only.

---

## Pre-capture verification checklist

Use this before you trust the netlist for PCB layout:

| Check | Pass criteria |
|-------|----------------|
| **Analog supply** | `AVDD` pins → `+5VA`; `AVSS`/`AVSS1` → `AGND`; measured target **4.75–5.25 V** (SBAS499 §7.3). |
| **Digital supply** | `DVDD` → `+3V3D`; `DGND` pins → `DGND`; **`NT_GND_TIE1`** bridges `AGND`↔`DGND` once at `U_ADS1` pads 32/31. |
| **Reference** | `VREFN` → `AGND`; `VREFP` has **≥10 µF + 0.1 µF** to `VREFN`; **U3 REF5025 DNP**; firmware will set **`PD_REFBUF = 1`**. |
| **VCAP bank** | VCAP1 **100 µF** (C0G/tantalum, not X7R); VCAP2/4 **1 µF**; VCAP3 **1 µF ∥ 0.1 µF** — all to `AGND`, at the package. |
| **Per-pin bypass** | **1 µF** at each `AVDD` and `DVDD` pad (+ rail bulk 10 µF / 0.1 µF). |
| **Input filter** | Per leg: **100 Ω → ESD (GND→AGND) → CM choke →** pair ends at **4.7 nF C0G differential** across `INxP`–`INxN` (not per-leg X7R to ground). |
| **CH1–CH6 pins** | IN1P/N = 16/15 … IN6P/N = 6/5 (see symbol). |
| **CH7–CH8** | **IN8N** = DRL (`MUX8=111`); **IN7P/IN7N** and **IN8P** NC; **no +5VA tie** (h03 §5). |
| **DRL / bias** | **J_EMG_DRL1 → 100 Ω → TPD → DLW21 → 220 kΩ → IN8N** + **4.7 nF**; at IC: **BIASREF→AGND**, **BIASIN↔BIASOUT**, **1 MΩ∥1.5 nF→BIASINV**, **`TP_BIASOUT1`** on IN8N net. |
| **Control straps** | `CLKSEL` → **10 kΩ → +3V3D**; `GPIO1–4` → **10 kΩ → DGND**; `RESV1`/`DAISY_IN` → `DGND`; `NC`/`RESERVED` open. |
| **SPI labels** | `ADS_SPI_*`, `ADS_DRDY`, `ADS_RESET`, `ADS_PWDN`, `ADS_START` match [`h05`](h05_microcontroller_firmware.md) §1.1. |
| **Lead-off LED** | **D_EMG_LO** on `mcu` (GPIO29) per Step 11 — not required on `emg` sheet |

**Common mistakes:** swapping **`PD_REFBUF` polarity** (`1` = on-chip ref **on**); tying spare inputs to **AGND** instead of **AVDD**; placing **CFilt** per-leg to ground (kills CMRR); populating **REF5025** while firmware enables the internal ref; routing **ESD array VCC** parts (`TPD4E004`) without a `+5VA` bypass; **via between bypass cap and ADS1299 pin**.
