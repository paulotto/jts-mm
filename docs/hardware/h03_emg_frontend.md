<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h03 — EMG Analog Front-End

> **Building the schematic in KiCad?** See the step-by-step capture walkthrough with pin-by-pin wiring and decoupling diagrams in [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md).

The EMG front-end is the highest-stakes circuit on the HAT. EMG signals at the surface of the masticatory muscles are 10 µV to ~1 mV peak with most useful energy in 20–450 Hz, embedded in 50/60 Hz line noise that can be three orders of magnitude larger. We adopt the OpenBCI / Cyton topology — a single TI ADS1299 with active driven-right-leg (DRL) bias — because it is well-proven, well-documented, and we can buy the chip directly instead of buying the whole OpenBCI board.

## 1. ADS1299 in one paragraph

The ADS1299 is an 8-channel, 24-bit, simultaneous-sampling Σ-Δ ADC purpose-built for biopotential measurements. Each channel has its own programmable-gain amplifier (PGA) with gain 1, 2, 4, 6, 8, 12, or 24. There is a single bias-drive amplifier shared by all channels (for the DRL electrode), a built-in **4.5 V** voltage reference (which we use — see §9), lead-off detection, on-chip oscillator, and SPI host interface with a `DRDY` "data ready" pin.

**Supply voltage (hard requirement, SBAS499 §7.3):** the ADS1299 needs an **analog supply of AVDD − AVSS = 4.75–5.25 V (nominal 5 V)** and a digital supply of DVDD = 1.8–3.6 V. We operate it **unipolar: AVDD = +5 V, AVSS = AGND = 0 V, DVDD = +3.3 V** (a bipolar ±2.5 V option exists but needs a negative rail). The analog rail **must be 5 V** — it is **not** a 3.3 V part on the analog side.

**Reference voltage (default):** we enable the **on-chip 4.5 V reference** so **VREF = 4.5 V** (full datasheet dynamic range: differential full-scale **±VREF / gain**). Supply and reference are related but not the same: **+5VA** powers AVDD; the internal ref buffer generates **4.5 V** between `VREFP` and `VREFN`. All ADS1299 electrical specs in SBAS499 are characterized at **AVDD−AVSS = 5 V, DVDD = 3.3 V, VREF = 4.5 V** — our default matches that corner exactly (see §3, §9).

## 2. Channel assignments

We have 8 ADC channels and 6 EMG muscles.

**CH1–CH6 → muscle:** see [an03 §1](../analysis/an03_emg_processing.md#1-montage), the one
montage table of the project: CH1 masseter_r, CH2 temporalis_r, CH3 digastric_r,
CH4 digastric_l, CH5 masseter_l, CH6 temporalis_l, as the cables are labelled (hardware check
2026-11-03). Sessions recorded before stored CH3/CH4 as digastric_l/digastric_r; the readers
correct that on read (an03 §1).

Placement per muscle (both sides mirrored):

| Muscle              | SENIAM-like position                                                                    |
| ------------------- | --------------------------------------------------------------------------------------- |
| Masseter            | Center of muscle belly, 2 cm above mandibular angle, fibers parallel to bite axis       |
| Anterior Temporalis | 2 cm above + 2 cm posterior of frontozygomatic suture, fibers vertical                  |
| Anterior Digastric  | Submandibular, 1 cm lateral of midline, anterior belly                                  |

Non-muscle channels:

| Channel | Use | Side  | Notes |
| ------- | --- | ----- | ----- |
| CH7     | BIAS debug (optional) | —   | No patient pin — IN7P/IN7N are not connected on the HAT; the ADS1299 MUX could route **BIASIN** here for readback vs **BIASREF** (SBAS499 Fig. 34); not implemented in firmware (§5.3) |
| CH8     | DRL electrode (IN8N) | —    | **Patient DRL** lands on **IN8N** via `J_EMG_DRL1`; firmware **MUX8 = 111** routes bias drive to IN8N (Fig. 33). **IN8P** is NC |

The bias / DRL electrode is **separate** from the six EMG pairs and routes through **`IN8N`** (`J_EMG_DRL1`); firmware **`CH8SET.MUX = 111`** drives the body from the internal bias loop (SBAS499 Fig. 33). Place the DRL on the **mastoid process** or behind the ear — bony tissue, away from the masticatory muscles, gives a stable driven-body reference.

For the rationale and exact placement diagrams, see `[../operations/op02_subject_preparation.md](../operations/op02_subject_preparation.md)` §3.

### 2.1 HAT connector vs. patient harness (13 conductors, not “13 muscles”)

Each **differential EMG channel** uses **two** electrode contacts → **two** conductors (**INxP** and **INxN**); the **DRL** electrode adds **one** more. That is **12 + 1 = 13 conductors**. The question is **how those 13 conductors land on the HAT** and **what cable type** runs to the electrodes ([§10.2](#102-patient-harness-and-hat-connector-architectures)).

| Item | Count |
|------|-------|
| EMG **channels** (muscles) | 6 |
| **Conductors** in the harness | 13 (12 signal + DRL) |
| **HAT-side mate points** | **13** (bring-up) or **1 / 4 / 7 / 13** (production paths) — see [§10.2](#102-patient-harness-and-hat-connector-architectures) |

Do **not** confuse this with “one connector per muscle” (that would be six and would not match bipolar wiring). **CH8 (IN8N)** carries the **DRL** electrode; **CH7** has no patient pin (optional BIAS debug only) — see §5 and [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md) Step 2.

**How the 13 conductors land on the HAT (full trade study in [§10.2](#102-patient-harness-and-hat-connector-architectures)):**

| Architecture | Catalog IDs | HAT connectors | Patient harness | Best when |
|--------------|-------------|----------------|-----------------|-----------|
| **0 — Bring-up (OpenBCI / Cyton-style)** | **H-CON-01CY**, **E-02CY** | **13×** 2.54 mm pin | **13×** unshielded snap→Dupont leads (12 + DRL) | **First PCB iteration** — fastest bench test; same topology as [OpenBCI Cyton](https://docs.openbci.com/Cyton/CytonSpecs/) |
| **A — Single trunk** | **H-CON-01T**, **E-02T** | **1×** 14-pin circular | One 6+1 shielded + collar spider | One mate point; custom trunk assembly |
| **B — Regional (anatomical)** | **H-CON-01R**, **E-02R** | **4×** (3 dual-ch + DRL) | Three regional Y-cables + DRL | **Default production procurement** — muscle geography |
| **C — Per-channel** | **H-CON-01C**, **E-02C** | **7×** (6 bipolar + DRL) | Six shielded LEMO→snap + DRL | Stock 2-lead snap cables (Witacon) |
| **F — DIN (shielded)** | **H-CON-01F**, **E-02F** | **13×** DIN 42802 | 13× shielded fan-out | Touch-proof device-side jacks; not first-iteration default |

**First iteration (PCB bring-up):** use **Option 0** — thirteen **0.1″ pin headers** on the HAT and **unshielded** OpenBCI-class snap cables. On-HAT analog front-end (100 Ω, TPD4E1B06, DLW21, differential `CFilt`) stays **identical** to production; only the **patient-side connector and cable** are simplified.

**Touch-proof policy:** Options **A/B/C** — snap or DIN 42802 at the **electrode end** only; keyed circular connectors at the HAT. **Option 0** — snap at electrode; **bare 2.54 mm pins** at the HAT (**bench / short cable runs only**; see [§10.2.0](#1020-option-0--bring-up-openbci--cyton-style-h-con-01cy--e-02cy)). **Option F** — touch-proof DIN at **both** ends if bionen leads are used.

**Schematic symbol `J_EMG`:** in filter diagrams below, **`J_EMG pin (INxP/N)`** means “the HAT contact for that signal” — one pin per conductor whether the physical connector is a trunk, a regional 4-pin, or a per-channel 2-pin.

## 3. PGA gain choice

EMG amplitudes:

- At rest: ~5–20 µV peak-to-peak.
- Light activation: 20–200 µV peak.
- MVC clench: 0.5–1.5 mV peak.

We use the **internal 4.5 V reference** (see §9), so the full-scale differential input range is **±VREF / gain** with **VREF = 4.5 V** — the same as the ADS1299 datasheet's characterized corner. With gain = **6** that is **±4.5 V / 6 = ±0.75 V** differential, which is far above any expected EMG amplitude and leaves large margin for electrode offset and common-mode transients without saturating. With 24-bit resolution, the LSB at gain 6 is:

```text
LSB = (2 × VREF) / (gain × 2^24) = (2 × 4.5 V) / (6 × 2^24) ≈ 89.4 nV
```

(At gain = 24 the LSB is ≈ 22.4 nV.) The LSB is still well below the thermal noise of the electrode–skin interface, so the system remains electrode-limited, not ADC-limited. Gain = **6** is the default; one `emg.default_gain` (or `--gain`) applies to CH1–CH6, CH7/CH8 are fixed at gain 24; higher gains (12, 24) are available when you need finer quantization on very small signals, but they are no longer required just to avoid wasting dynamic range.

**Why not default to gain = 24 for masticatory work?** It is tempting — the §11 noise-floor spec (0.98 µVrms) was characterized *at* gain 24, and a higher gain gives more raw ADC counts per µV. But raw counts are not the same as signal quality: `jtsmm` always reports data in µV (`lsb_uv() = V_ref / (gain · 2^23)`), so a clean signal looks identical on screen at any gain — only the *headroom* and *noise floor* change, and both cut the wrong way for this protocol:

- **Headroom.** The front end is **DC-coupled all the way to the PGA** — the 20 Hz high-pass is done digitally on the Pi (§4, [`s04_emg_signal_processing.md`](../software/s04_emg_signal_processing.md)), not in analog hardware before the ADS1299. That means electrode half-cell offset, sweat/gel drift over a session, and motion artifact from active jaw movement all land on the PGA input undivided. At gain 6 the differential full-scale is ±0.75 V — ~500× the largest expected MVC clench (≤1.5 mV) — leaving huge margin before these artifacts saturate the PGA. At gain 24 that margin shrinks to ±187.5 mV, only ~125×, and dynamic mastication protocols (active jaw motion, cable/skin movement between phases) are exactly the case where transient offsets of tens to a few hundred mV are plausible.
- **Noise floor.** At gain 6 the ADC's LSB (≈89 nV) is already far below the electrode–skin interface's own thermal/contact noise (~1 µVrms typical, §11) — the system is electrode-limited, not ADC-limited. Going to gain 24 lowers the LSB further (≈22 nV) but does **not** lower the actual measured noise floor, because the electrode interface — not ADC quantization — is the dominant noise source. So gain 24 buys no real SNR improvement here, only less headroom.

Net: **gain = 6 stays the production default for the acquisition profile.** Use `jtsmm record --gain 24` (or `view --gain 24`) deliberately for controlled, low-motion recordings (e.g. quiet baseline/rest captures with fresh gel and no jaw movement) where the extra quantization resolution on very small signals is worth the reduced saturation margin — but it is an explicit trade-off, not a universal upgrade.

> **Why not 2.5 V?** A 2.5 V external reference (e.g. REF5025) would work electrically on a 5 V supply but only uses **2.5/4.5 ≈ 56 %** of the ADC's designed span and forces you to scale every datasheet noise/CMRR number. We standardize on **VREF = 4.5 V** for **full dynamic range** and direct comparison to TI specs.

<a id="4-input-network"></a>

## 4. Input protection and filter network
Each EMG input pair (e.g. INxP, INxN) is filtered by the following network, placed within 5 mm of the ADS1299 pin:

```text
        J_EMG pin (INxP)           J_EMG pin (INxN)
              |                            |
       R_series 100 Ω 1%            R_series 100 Ω 1%      (per leg; may be a resistor array)
              |                            |
       ESD array  ───────┬───────  ESD array               (TPD4E1B06: 0.7 pF, 0.5 nA leak, bidirectional)
              |          │                |
       (470pF opt CM)    │         (470pF opt CM)           (≤ CFilt/10, to AGND, DNP default)
              |          │                |
              +────  common-mode choke  ──+                 (one DLW21 per PAIR)
              |   (Murata DLW21SN900SQ2L, 90 Ω @ 100 MHz)   |
              ├──────── CFilt 4.7 nF C0G ─┤                 (DIFFERENTIAL, one per PAIR)
              o (ADS1299 INxP)             o (ADS1299 INxN)

  J_EMG shell ── AGND   (one tie per patient connector; see §2.1, §10.2)
```

The series resistor and ESD tap are **per leg** (INxP and INxN each get their own). The main anti-alias/RFI capacitor `CFilt` is a **differential cap across INxP–INxN — one per pair**, and the **common-mode choke is one 4-terminal device per pair**. This follows the ADS1299 datasheet (SBAS499 §10.2.2), which advises forming the filter with a **differential** capacitor *"rather than individual RC filters whose capacitors shunt to ground,"* because the differential configuration removes the CMRR dependence on component mismatch. If bench EMI requires common-mode caps to ground, the datasheet says they must be **10–20× smaller** than `CFilt` (hence ≤470 pF, and DNP by default since the choke already handles common-mode RF).

**Per-channel / total quantities (6 EMG pairs):**

| Part              | Per channel  | Total (6 EMG)           | + DRL                 | Catalog           |
| ----------------- | ------------ | ----------------------- | --------------------- | ----------------- |
| 100 Ω 1 % series  | 2 (per leg)  | 12 (= 3× 4-element array, §4.1) | —             | `H-PAS-02`        |
| 4.7 nF C0G `CFilt` (differential) | 1 (per pair) | 6          | —                     | `H-PAS-02`        |
| 470 pF C0G CM (optional, DNP) | 2 (per leg) | 12 (DNP)         | —                     | `H-PAS-02`        |
| DLW21 CM choke    | 1 (per pair) | 6                       | +1 on DRL             | `H-IC-10` (qty 7) |
| TPD4E1B06 quad ESD | shared      | 3 arrays cover 12 lines | (DRL on a spare line) | `H-IC-09` (qty 3) |

Functions:

- **R_series 100 Ω**: limits ESD current and forms the RC anti-alias / EMI filter with `CFilt`. With the differential 4.7 nF across 2×100 Ω the cutoff is ≈ 170 kHz, well above EMG bandwidth and below fMOD.
- **ESD diode array**: handles transient discharges from the patient cable. Pick a part that is both **low-capacitance** (so it does not degrade CMRR) and **ultra-low-leakage** (so it does not add input-bias/offset error to a precision DC-coupled front-end). Default is the **TPD4E1B06** (the same part OpenBCI uses): a **4-channel bidirectional** TVS array, **0.7 pF** typ (0.95 pF max), **0.5 nA** max leakage, ±12 kV contact / ±15 kV air, GND-referenced flow-through — tie its GND pin to AGND, no VCC bypass needed. The bidirectional clamp (±5.5 V working, referenced to AGND) suits inputs biased near mid-supply, and the 0.5 nA leakage is orders of magnitude below ordinary ESD diodes (which can leak µA). Approved alternatives: **TPD4E05U06DQA** (quad, ~0.5 pF/ch, GND-referenced — a USB3-oriented part, perfectly usable here) and **TPD4E004** (quad, ~1.6 pF/ch, VCC-referenced — needs VCC tied to +5VA with a 0.1 µF bypass). All three stay well under the 5 pF guideline.
- **`CFilt` 4.7 nF C0G (differential, across INxP–INxN)**: the primary anti-alias / RFI filter. SBAS499 §10.2.2 specifies a *differential* cap here (not per-leg-to-ground) so CMRR does not depend on cap matching. C0G dielectric to avoid voltage-coefficient distortion.
- **DLW21 CM choke**: **RF common-mode** suppression (≈90 Ω @ 100 MHz typ on **DLW21SN900SQ2L**). It does **not** reject 50/60 Hz line frequency — that is handled by the ADS1299 **differential front-end + DRL bias**, electrode/cable practice, and **digital notch filtering** ([§4.3](#43-line-frequency-5060-hz-vs-rf-cm), [`s04`](../software/s04_emg_signal_processing.md) §4).
- **Optional CM caps (≤470 pF to AGND)**: only if bench EMI demands it; must be 10–20× smaller than `CFilt` per SBAS499. DNP by default.

The full network adds about 200 Ω of series impedance per pair (2 × 100 Ω) plus the choke DCR, into a high-impedance differential input. The ADS1299 PGA input impedance is >1 GΩ, so the gain error from this added impedance is negligible.

### 4.1 Using resistor / capacitor arrays (OpenBCI-style, recommended for density)

The per-channel passives can be built from **monolithic 4-element arrays** instead of discretes. This is what the OpenBCI Cyton does, and on a 6-channel front-end it saves substantial board area. The choices are not equal, though — match the dielectric/type to the job:

- **Series resistors → 4-element resistor array (recommended).** Use a Bourns **CAY16-1000F4LF** (4 *isolated* resistors, 100 Ω, **±1 %**, ±200 ppm/°C, 1206; the `…-1000…` code = 100 Ω). Equivalents: Yageo **YC164** (e.g. `YC164-FR-07100RL`), Panasonic **EXB-28V**. The 12 EMG legs (6 pairs × 2) fit in **3 arrays**. Beyond density, the four elements share one substrate and are laser-trimmed together, so they **track closely in value and temperature** — the INxP / INxN legs stay better matched than two independent ±1 % discretes, which directly **helps CMRR**. (The absolute ±1 % / ±200 ppm/°C spec is irrelevant for a series resistor feeding a >1 GΩ input; only the leg-to-leg *match* matters.) OpenBCI Cyton uses **2.2 kΩ discretes** and a different cap topology — see [§4.2](#42-openbci-cyton-input-filter-vs-this-design).
- **Capacitors → 4-element MLCC array *only if you keep the dielectric right*.** The same TDK **CKC** family (CKCL44 = 0805, CKCA43 = 1206) comes in C0G, X5R and X7R. The OpenBCI part `CKCL44X7R1H102M085AA` is **X7R 1 nF** — acceptable for plain bypass, **but the ADS1299 datasheet (§12.1) explicitly requires the differential input cap to be C0G/NPO** (low voltage-coefficient, low noise). So:
  - **Keep `CFilt` (the differential cap) C0G.** C0G in this array family tops out at **1 nF** (`CKCA43C0G1H102K100AA`, 1206), so either (a) keep the **4.7 nF C0G as a discrete** (no C0G array exists at 4.7 nF — preferred, best filtering) or (b) drop `CFilt` to **1 nF C0G array** (corner rises to ≈ 800 kHz, still fine). **Do not put `CFilt` in X7R.**
  - **The optional common-mode caps to AGND are the place where a cap *array* genuinely helps**, because the matched elements keep CMRR high — but still use a **C0G** array (e.g. `CKCA43C0G1H471K100AA`, 470 pF), not X7R. These remain DNP by default.

**Summary:** adopt the **resistor array** and the **TPD4E1B06** (both clear wins — OpenBCI uses the same ESD part); for capacitors, keep the differential `CFilt` in **C0G** (discrete 4.7 nF or 1 nF C0G array) and reserve arrays for the optional matched CM caps. Avoid the X7R cap array on the signal-critical differential cap.

### 4.2 OpenBCI Cyton input filter vs this design

OpenBCI and this HAT share the **ADS1299** and **TPD4E1B06**, but **not** the same R/C filter topology or values. Do not copy Cyton’s **2.2 kΩ** into this schematic without redesigning the cap network.

| | **OpenBCI Cyton** ([schematic](https://docs.openbci.com/Cyton/CytonSpecs/#openbci-cyton-board-circuit-schematic)) | **This HAT (§4)** |
| --- | --- | --- |
| **Series R** | **2.2 kΩ** per leg (discrete) | **100 Ω** per leg (CAY16 array) |
| **Filter cap** | **1 nF X7R per leg → AGND** (common-mode RC) | **4.7 nF C0G differential** across INxP–INxN |
| **Topology** | Per-leg RC to ground | SBAS499 §10.2.2 differential `CFilt` |
| **Leg RC corner** | ≈ **72 kHz** (2.2 kΩ × 1 nF) | Differential corner ≈ **170 kHz** (2×100 Ω × 4.7 nF) |
| **CM choke** | No | **DLW21** per pair |
| **Design goal** | RFI at Σ-Δ sinc “holes” (~1.024 MHz multiples); CMRR trade-off vs cap size ([forum](https://openbci.com/forum/index.php?p=%2Fdiscussion%2F154%2Ffront-end-filtering-esd-cmrr-rfi)) | CMRR + EMG bandwidth (§11); differential filter avoids cap-mismatch CM degradation |

**Why Cyton uses 2.2 kΩ:** the resistor pairs with a **1 nF leg-to-ground** cap. With only **100 Ω**, that RC would corner near **1.6 MHz** — too weak for the RFI suppression OpenBCI targets. The **2.2 kΩ** value was chosen after bench CMRR sweeps (100 pF vs 1 nF vs 4.7 nF): **1 nF** was the compromise between CMRR and attenuation at high frequency. That filter is **not** a tight Nyquist anti-alias for 250 Hz sampling; it cooperates with the ADS1299 internal sinc response.

**Why we use 100 Ω:** filtering is done by the **differential** 4.7 nF cap (both legs’ R appear in series: $f_c \approx 1/(2\pi \cdot 2R \cdot C)$). The **100 Ω** mainly **limits ESD fault current** into the TPD4E1B06 and adds negligible drop into the >1 GΩ PGA input. TI’s ADS1299 EEG-FE eval kit uses an even **more conservative** **~5 kΩ + 4.7 nF** differential network (~3.4 kHz corner) — we sit between Cyton’s per-leg topology and TI’s eval kit, biased toward **CMRR** and **110 dB-class** noise targets.

**BOM note:** reverse-engineered Cyton KiCad notes sometimes list a Bourns **RN-CAY16-F4** resistor array; the **published Cyton schematic** shows **2.2 kΩ discretes** on the input legs, not 100 Ω CAY16 arrays. OpenBCI’s **CKCL44 1 nF X7R** caps are on the **per-leg** filter — we do **not** use X7R on the differential `CFilt`.

### 4.3 Line frequency (50/60 Hz) vs RF CM

| Mechanism | What it rejects | Where |
|-----------|-----------------|-------|
| **ADS1299 differential INxP−INxN** | Common-mode voltage that appears equally on both legs | On-chip PGA |
| **DRL (BIASOUT) closed-loop bias** | Residual body/common-mode pickup vs the reference electrode | ADS1299 bias amp + mastoid (or similar) DRL electrode |
| **Matched 100 Ω legs + differential `CFilt`** | CMRR degradation from component mismatch; high-frequency RFI | On-HAT analog filter (§4) |
| **DLW21 CM choke** | **RF** common-mode (100 MHz-rated Z<sub>CM</sub>; FM/Wi-Fi/cable RFI) | On-HAT, **not** 50 Hz |
| **Shielded harness + shell→AGND** | Capacitive pickup from mains fields | Patient cable (production options) |
| **Digital 50 Hz (+ harmonics) notch** | Residual line hum after digitization | Pi / [`s04`](../software/s04_emg_signal_processing.md) §4 |

At 50/60 Hz the DLW21 presents **milliohm-class DCR** only — it is invisible to line frequency. Do **not** expect the choke to substitute for DRL placement, shielding, or software notch filtering. On a noisy bench, omitting the DLW21 can **reduce** capacitive 50 Hz pickup slightly (documented in the catalog); keep it populated for RF/ESD cable immunity in normal use.

**If line noise dominates after bring-up:** verify DRL contact and mastoid placement; use shielded cables (Options A–C); enable the adaptive 50 Hz notch in post-processing; during scientific captures disable lead-off tone (31.2 Hz injection, §6); re-bench CMRR per §13 if cable length exceeds nominal.

## 5. Bias-drive (DRL) circuit

The driven-right-leg (DRL) electrode is a **single-ended** patient lead. Hardware follows **TI SBAS499 §9.3.2.4** (Fig. 33 — MUX-routed bias to **IN8N**) rather than the older OpenBCI-style **220 kΩ → BIASOUT** tap. The patient still gets the same **RFI/ESD** blocks as one EMG leg (**100 Ω**, **TPD4E1B06**, **DLW21SN900SQ2L**) before the **220 kΩ** current limit and the **IN8N** ADC input.

> **Why the choke is in the BOM:** quantity **6 chokes (EMG) + 1 (DRL)** — the seventh Murata part is on the **DRL patient path** (`FL7` on `emg.kicad_sch`).

### 5.1 On-chip bias network (SBAS499 Fig. 33 / 34)

| Pin | Hardware | Firmware |
|-----|----------|----------|
| **BIAS_SENSP / BIAS_SENSN** | no wires | Enable **CH1–CH6** only; **CH7/CH8 off** for bias derivation |
| **BIASREF (60)** | **→ AGND** (unused) | **`CONFIG3.BIASREF_INT = 1`**: bias reference is the internal (AVDD+AVSS)/2 = **2.5 V**, so the pin is ignored. **`PD_BIAS = 1`** powers the amplifier → **`CONFIG3 = 0xEC`**. Never `0xE4` (`BIASREF_INT = 0`): the amp would drive the body to the pin's 0 V, the negative rail of this unipolar supply (SBAS499C §10.1.3) |
| **BIASIN (62)** | **direct tie → BIASOUT (63)** | Close the loop at the pin (no external filter) |
| **BIASINV (61)** | **1 MΩ ∥ 1.5 nF** to **BIASOUT** | Feedback network per Fig. 33/34 (`R8`, `C78`) |
| **BIASOUT (63)** | **no shunt cap** | TI Fig. 33/34 — feedback is **R8 ∥ C78** only |

### 5.2 Patient DRL path → IN8N (MUX topology)

The **DRL electrode** connects to **`IN8N`**, not to **BIASOUT**. Firmware sets **`CH8SET.MUX = 111`** so the chip routes the internal **BIASIN** signal to the **N-side** of channel 8 (Fig. 33).

1. **`J_EMG_DRL1`** (Option 0 pin header) — patient mastoid electrode; no shell/shield in bring-up.
2. **100 Ω 1 %** (`R5`) — ESD current limit (same as an EMG leg).
3. **TPD4E1B06** — one spare quad channel; **GND → AGND**.
4. **DLW21SN900SQ2L** (`FL7`) — pins **1 → 2** in series; **3–4 NC**.
5. **220 kΩ** (`R6`) — between choke and **`IN8N`** (patient current limit; ≤ 5 V / 220 kΩ ≈ **23 µA**).
6. **4.7 nF C0G** (`C2`) — patient-side node → **AGND** (RFI on the electrode lead).

**`TP_BIASOUT1`** — scope pad on the **`IN8N` / DRL** net (after the 220 kΩ), **not** on the raw **BIASOUT** pin. With **`MUX8 = 111`** and the loop closed, it tracks the driven-body voltage; for the internal bias-amp output, probe the **`BIASOUT`** net at **`R8`/`C78`** or pin 63 instead.

```text
   CH1..CH6 ── (BIAS_SENSP/N in firmware only)

   BIASREF (60) ── AGND (unused)        (BIASREF_INT = 1: internal 2.5 V; CONFIG3 = 0xEC)

   BIASOUT (63) ──┬──[1 MΩ ∥ 1.5 nF]──► BIASINV (61)
                  └──► BIASIN (62)     (direct tie BIASIN↔BIASOUT)

   J_EMG_DRL1 ──[100Ω]──┬── TPD4E1B06 ── DLW21 1──╫──2 ──[220kΩ]──► IN8N (pin 1)
                         │                              └── 4.7 nF ── AGND
                    TP_BIASOUT1 (tap on IN8N net)

   IN8P ── NC          IN7P / IN7N ── NC (§5.3)
```

**BOM (DRL + bias discretes, as-built refs):** `R5` 100 Ω, `R6` 220 kΩ, `R8` 1 MΩ, `C78` 1.5 nF C0G, `C2` 4.7 nF C0G, `FL7` DLW21, `D8` TPD4E1B06 (DRL ESD) — **`H-PAS-03`** / **`H-IC-09`** / **`H-IC-10`** / **H-CON-01\*** ([`04_component_catalog.md`](../04_component_catalog.md)).

### 5.3 CH7 — optional BIAS readback (Fig. 34, debug only)

**IN7P / IN7N** are **not connected** on the HAT (no filter, **no patient connector**, **no tie to +5VA**). The ADS1299 allows routing **BIASIN** to CH7 internally on the bench (**`MUX7 = 010`** or **`111`**, **`CONFIG3.BIAS_MEAS = 1`**, CH7 excluded from **BIAS_SENSP/N**) to read the bias voltage **vs the internal `BIASREF`** (mid-supply, since `BIASREF_INT = 1`). The firmware does **not** implement it (`bias.ch7_readback: true` is refused); CH7 stays powered down.

## 6. Lead-off detection

The ADS1299's built-in lead-off detection can inject either a **DC** or an **AC** current (LOFF register `FLEAD_OFF[1:0]`, SBAS499C Table 16) into each enabled input and senses the resulting voltage:

- **DC lead-off** (`FLEAD_OFF = 00`) pegs a disconnected input to a supply rail; the comparator trips and a status bit is set in the SPI status word (binary connected/disconnected only — tissue polarization makes the *voltage* meaningless for a numeric |Z|).
- **AC lead-off** (`FLEAD_OFF = 10`, the fixed **31.2 Hz** tone derived from the ADS1299 master clock — *not* from the configured sample rate) lets the host compute a real electrode-skin impedance: the tone's amplitude at the channel output is `V_tone ≈ I_lead_off × Z_total`, and `jtsmm emg impedance` recovers `|Z|_electrode ≈ V_tone / I − R_protect` (de-embedding the ~200 Ω on-board series network, §4).

Both modes share `LOFF_SENSP`/`LOFF_SENSN` (which inputs are exercised) and `ILEAD_OFF[1:0]` (current magnitude — only **6 nA** and **24 nA** are usable for skin |Z|; the 6/24 µA options would swing the PGA into the rails on a normal electrode pair). The firmware reports per-channel lead-off in real time from the SPI status word embedded in every EMG frame.

We **enable DC lead-off by default everywhere** — `jtsmm view`, `jtsmm record`, and `jtsmm record --viewer` all keep the DC comparator running (`emg.leadoff.during_recording: true` in the shipped `etc/jtsmm.yaml`, `emg.leadoff.ac_tone: false`, and the same in the firmware boot defaults; only `noise_characterization` forces `false`) so the operator can always see which electrodes have lost contact. This is safe for the scientific signal: the DC lead-off current is a *constant* bias, so on a connected input it only adds a fixed DC offset that the 20 Hz high-pass removes, and on a disconnected input it pulls the (already meaningless) reading to a rail — it never puts anything inside the EMG passband. The *AC tone* is the one that would contaminate a recording, because it sits at a fixed 31.2 Hz inside the EMG band; `emg.leadoff.ac_tone` therefore stays **false** during `record`/`view` regardless of `during_recording`, and `validate.py` warns if a custom config ever sets both `during_recording` and `ac_tone` together.

Pass `--no-leadoff` to `jtsmm record` or `jtsmm view` to disable the DC comparator too (sets `during_recording: false`), e.g. for a session that must be bit-for-bit free of any injected current, or to compare against. When disabled, the firmware programs the LOFF registers off at config apply and the viewer shows **n/a** for lead-off LEDs (status bits in the SPI word are stale). Lead-off current is selectable via `emg.leadoff.current_na` (**6** or **24** nA). `jtsmm emg check` and `jtsmm emg impedance` run their own short, dedicated AC-tone session independent of this recording-time setting.

Recorded sessions persist the per-sample DC comparator bits in `emg/leadoff_status` (columns `[statp_mask, statn_mask]`, muscle CH1-6 only, bit *i* = CH(*i*+1); see [`s02`](../software/s02_data_pipeline.md) §8). DRL (CH8/IN8N) is deliberately **not** included in that dataset — see the next paragraph for why its comparator only means something when read in isolation.

Each muscle channel drives its own independent current source (P) / sink (N) pair (SBAS499C Fig. 35b), so all six can be measured in parallel without cross-talk; the DRL/mastoid reference (CH8/IN8N) cannot be measured this way because ADC channel 8 is multiplexed to the internal `BIASIN` node (`MUX8=111`, SBAS499C Table 17 "BIAS_DRN") during normal operation and `IN8P` is not wired to an electrode (§5) — DRL connectivity is only available as the binary DC comparator result.

**DRL's comparator must run in isolation from the six muscle channels.** DRL has only a current *sink* on `IN8N` (`IN8P` is unwired, so there is no source leg on that channel). Each muscle channel, by contrast, drives *both* a source (P) and a sink (N) through the same patient skin/tissue network at the same time. If both run together, the six muscle currents share DRL's own body-return path and keep the shared node from ever reaching the comparator's rail-side threshold — the DRL comparator then never trips, even with the electrode genuinely disconnected (this is why the DRL row of `jtsmm emg check` used to always read `PASS`). `jtsmm emg check` fixes this with a short (`DRL_ISOLATION_DURATION_S = 2 s`) sub-session that powers down `CH1-6` (`emg.pd_channels = 0x3F`) so the ADS1299 excludes them from `LOFF_SENSP`/`LOFF_SENSN`, leaving only DRL's sink active on the shared comparator. The live viewer does **not** run this isolation (it would mean periodically dropping the six muscle channels while viewing), so its DRL LED is always shown as **n/a** — use `jtsmm emg check` before recording for an authoritative DRL contact result.

## 7. Sample rate and filtering

The ADS1299 supports 250 SPS to 16 kSPS via a single CONFIG1 register bit field. We choose:

- **fs = 4000 SPS** (CONFIG1 `0x92`) for normal acquisition — the default.
- **fs = 2000 SPS** (CONFIG1 `0x93`, the former default) when the DRDY read margin matters more than the passband (500 µs instead of 250 µs per sample).
- **fs = 1000 SPS** when running with all 8 channels and lots of additional processing on the MCU.
- **fs = 8 / 16 kSPS** are not supported (the firmware rejects them; max 4000). `noise_characterization` runs at 4000 SPS.

The on-chip Σ-Δ modulator + decimating digital filter give a 3-stage Sinc filter response (datasheet §8.4.4). At fs = 4000 it is −0.7 dB at 500 Hz and rejects the first alias by 51 dB; at fs = 2000 the same figures are −2.7 dB and 31 dB, which is why the default moved to 4000. In-band noise density is unchanged (gain 6: 0.93 µVrms over 1048 Hz at 4000 vs 0.66 µVrms over 524 Hz at 2000). Combined with the analog RFI filter above, the anti-alias filtering is excellent.

Additional digital filtering (high-pass 20 Hz, notch 50 Hz, etc.) is done on the Pi 5 in post-processing — see `[../software/s04_emg_signal_processing.md](../software/s04_emg_signal_processing.md)`.

## 8. SPI interface

The ADS1299 is on **SPI0**. The authoritative RP2350A pin map for the whole HAT lives in `[h05_microcontroller_firmware.md](h05_microcontroller_firmware.md)` §1.1; the EMG-relevant subset is repeated here for convenience.

| Signal | RP2350A pin (reference) | Notes                                               |
| ------ | ---------------------- | --------------------------------------------------- |
| SCLK   | GPIO2 (SPI0_SCLK)      | up to 20 MHz; stream capped at 4 MHz (`cfg_hz` for register access) |
| MOSI   | GPIO3 (SPI0_TX)        | SPI mode 1 (CPOL=0, CPHA=1)                         |
| MISO   | GPIO0 (SPI0_RX)        |                                                     |
| /CS    | GPIO5                  | active low                                          |
| /DRDY  | GPIO6                  | data-ready, polled by the RP2350A sensor core        |
| /RESET | GPIO7                  | active low, held high for normal operation          |
| /PWDN  | GPIO8                  | active low; tied high in firmware (kept on)         |
| START  | GPIO9                  | active high; toggled to start continuous conversion |

DRDY is the **only timing-critical signal**: every falling edge corresponds to a new sample. The MCU's sensor core polls DRDY, timestamps the falling edge and reads the RDATA command plus 27 bytes of conversion data (status + 8 × 24-bit channels) via SPI DMA. The 28-byte transfer at 4 MHz SPI takes ~56 µs (28 × 8 bit / 4 MHz), comfortably below the 250 µs inter-sample interval at the default 4 kSPS.

## 9. Power and reference

- **AVDD = 5 V** (AVSS = AGND = 0 V) from the low-noise **+5VA** rail — see [`h07`](h07_power_grounding_safety.md) §1; this 5 V is **required** (SBAS499 §7.3), not optional.
- **DVDD = 3.3 V** from the digital LDO (**TPS73633DBV**), separate path.
- **VREF = 4.5 V (internal, default):** enable the ADS1299 **on-chip reference buffer** in firmware (`CONFIG3.PD_REFBUF = 1`; SBAS499 Table 15: `0` = power-down, `1` = enable). Tie **VREFN** to **AVSS/AGND**. Decouple **VREFP** with the datasheet-mandated **≥10 µF + 0.1 µF** between **VREFP and VREFN** (SBAS499 Pin Functions). The chip generates **4.5 V** across `VREFP`–`VREFN` — **no external reference IC** on the default BOM.
- **Do not populate REF5025** (or any external ref) on the default build — the `REF5025` footprint on early schematics is **DNP**. Optional external **4.5 V** part: **REF5045AIDR** (see [`04_component_catalog.md`](../04_component_catalog.md) **H-IC-02**) if you ever disable the internal buffer and drive `VREFP` externally.

**Internal-reference schematic (unipolar 5 V, SBAS499 Figure 77 style):**

```text
   +5VA ──► AVDD (ADS1299)          VREFN (25) ──► AGND
                                    VREFP (24) ──┬── ≥10µF ──┐
                                                 ├── 0.1µF ──┴── VREFN
                                                 └── (driven by on-chip 4.5 V buffer)
```

> **VCAP1 with internal ref:** SBAS499 §11 stresses a **100 µF** on VCAP1 using **C0G/NPO or tantalum** (not X7R/X5R) — piezoelectric class-2 ceramics inject noise into the reference/charge-pump path. This is **more important** with the internal 4.5 V reference enabled.

**Decoupling — exact values from the ADS1299 datasheet (SBAS499) Pin Functions table + §11** (all caps to AVSS = AGND in our unipolar scheme; place as close to the package as possible):

| Node | Cap | Note |
|------|-----|------|
| Each AVDD pin (19, 21, 22, 56) | 1 µF to AVSS | + 10 µF + 0.1 µF bulk per analog rail (§11) |
| AVDD charge-pump (59) | 1 µF to AVSS (pin 58) | — |
| AVDD1 (54) | 1 µF to AVSS1 | **star-connect** AVDD1→AVDD, AVSS1→AVSS (§11) |
| Each DVDD pin (48, 50) | 1 µF to DGND | + 10 µF + 0.1 µF bulk |
| VREFP (24) | ≥10 µF + 0.1 µF to VREFN | **internal 4.5 V ref** (not an external IC) |
| **VCAP1 (28)** | **100 µF** to AVSS | **C0G/NPO or tantalum** — critical with internal ref |
| VCAP2 (30) | 1 µF to AVSS | — |
| VCAP3 (55) | 1 µF ∥ 0.1 µF to AVSS | parallel pair |
| VCAP4 (26) | 1 µF to AVSS | — |

> **Power-up sequencing (SBAS499 §11.1):** hold all digital/analog inputs low until the supplies are stable, start `CLK`, then issue `RESET` only **after tPOR (218 × tCLK) or after VCAP1 > 1.1 V, whichever is longer**. The 100 µF on VCAP1 makes this ramp the dominant term — see [`h05`](h05_microcontroller_firmware.md) §3.4. The two-state pins `CLKSEL`/`GPIO1–4` strap through **≥10 kΩ** and `RESV1` ties **directly to DGND**.

### 9.1 Bench test pads

The five pads are on the **`emg`** sheet, in the analog region at the ADS1299 (away from RP2350 buck/USB). Footprint: `TestPoint:TestPoint_Pad_D1.0mm`. Full map: [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §6.1.

| Ref | Net | Tap | Use | Alternative |
|-----|-----|-----|-----|-------------|
| `TP_ADS_DRDY1` | `ADS_DRDY` | ADS1299 DRDY pin | Scope 4 kHz data-ready (default rate) when the ADS1299 does not start | USB-CDC `selftest` / `ads_drdy_watch` |
| `TP_VREFP1` | `VREFP` | ADS1299 pin 24 | DMM ~4.5 V vs `TP_AGND1` before trusting counts | Trust counts if `selftest` passes |
| `TP_AGND1` | `AGND` | Star tie under ADS1299 | Scope return for analog checks — **not** `DGND` | — |
| `TP_BIASOUT1` | `IN8N` / DRL | After `R6` on DRL path | Driven-body / DRL voltage (with `MUX8=111`); **not** raw `BIASOUT` — probe **`R8`/`C78`** or pin 63 for that | CH8 SPI readback in BIAS-measure mode |
| `TP_VCAP1` | `VCAP1` | ADS1299 pin 28 | Verify charge-pump ramp before `RESET` | Follow power-up script only |

There are **no** TPs on patient `INxP/N` — pin headers (Option 0) are sufficient and TPs near inputs risk hum pickup. Stub of each pad **<3–5 mm**.

## 10. Electrode and patient-cable details

- **Disposable Ag/AgCl electrodes** (used: Ambu BlueSensor P-00-S; any pre-gelled Ag/AgCl snap electrode fits): excellent skin–metal contact, low DC drift, single use.
- **Patient harness:** **13 conductors** (six bipolar pairs + DRL); shielded in production paths **A/B/C/F**, **unshielded** in bring-up **Option 0** — see [§10.2](#102-patient-harness-and-hat-connector-architectures). Catalog codes **E-02CY / E-02T / E-02R / E-02C / E-02F** are internal lead-set families, not vendor SKUs.
- **Skin preparation**: alcohol wipe + Nuprep abrasive gel + dry, in that order. See [`../operations/op02_subject_preparation.md`](../operations/op02_subject_preparation.md) for the operator-facing procedure.

The standard cable runs **unshielded for 5–10 cm at the electrode end** (the "pigtail"). This is typical industry practice; fully shielded right up to the snap improves CMRR slightly but is uncomfortable and tangles easily.

### 10.1 Cable parasitics, length budget, and cable type

**Scope.** On-HAT parasitics are sized in §4 (100 Ω, 4.7 nF differential `CFilt`, ESD, choke). This section gives **lumped-element models** for the **off-board** shielded harness so you can estimate the effect of **length** and **cable type** before buying longer leads or extensions.

**Harness count (not 13 muscles).** v1 uses **6 differential EMG channels** → **12 signal conductors** plus **1 DRL** → **13 wires** total. They may travel as **thirteen individual unshielded leads** (Option **0**), **one trunk**, **four regional cables**, **seven per-channel cables**, or **thirteen shielded wires** ([§10.2](#102-patient-harness-and-hat-connector-architectures)). Optional **E-03** extensions apply only to **H-CON-01F** ([`04_component_catalog.md`](../04_component_catalog.md)).

#### Lumped model (one leg)

From the electrode snap to the ADS1299 input pin (simplified):

```text
electrode ──[pigtail]── shielded pair ── patient connector pin ── R_series ── ESD ── choke ── INxP/INxN
                              │                              (on HAT, §4)
                              └── C_leg, C_sh(L)  (per conductor to shield)
```

Treat each signal conductor as a **π- or T-section** with:

- **Series resistance** (wire): `R_wire = R' × L`
- **Shunt capacitance** (conductor ↔ shield): `C_sh = C' × L`
- **Series inductance** (wire): `L_wire = L' × L` (usually negligible below ~1 MHz for EMG)

`L` = **shielded length** in metres (patient harness + any **E-03** extension; pigtail unshielded segment is separate).

#### Typical cable parameters (use datasheet or bench measure)

| Parameter | Symbol | Shielded medical / lab cable (per conductor) | Notes |
| --------- | ------ | --------------------------------------------- | ----- |
| Capacitance to shield | `C'` | **40–120 pF/m** | Measure with LCR @ 1 kHz; depends on dielectric, twist, shield tightness |
| DC resistance | `R'` | **0.04–0.10 Ω/m** (24–26 AWG) | Round-trip drop is `2 R' L` per leg |
| Inductance | `L'` | **0.3–0.6 µH/m** | Relevant only for MHz-class EMI |

**Example (planning numbers):** `C' = 80 pF/m`, `R' = 0.05 Ω/m`, `L = 1.5 m` (nominal patient run, collar → HAT):

```text
C_sh  = C' × L = 80×10⁻¹² × 1.5 ≈ 120 pF
R_wire = R' × L = 0.05 × 1.5 ≈ 0.075 Ω   (≪ 100 Ω series on HAT)
L_wire = L' × L ≈ 0.5 µH                 (negligible vs 100 Ω at 50 Hz)
```

With **E-03** (+1.5 m extension on a channel): use **`L = 3.0 m`** → `C_sh ≈ 240 pF`, `R_wire ≈ 0.15 Ω`.

#### Equations — what actually limits length

**1) HF rolloff on each leg (cable C before `R_series`)**

The on-HAT **100 Ω** is **after** the patient connector pin; cable capacitance sits on the **patient side** of that resistor. A conservative single-pole model:

```text
f_c,leg ≈ 1 / (2π × R_series × C_sh)
         = 1 / (2π × R_series × C' × L)
```

With `R_series = 100 Ω`, `C' = 80 pF/m`:

| Total shielded length `L` | `C_sh` | `f_c,leg` |
| ------------------------- | ------ | --------- |
| 1.5 m (nominal) | 120 pF | **≈ 13 MHz** |
| 2.0 m | 160 pF | **≈ 10 MHz** |
| 3.0 m (+ E-03) | 240 pF | **≈ 6.6 MHz** |

This is **far above** the EMG band and above the on-HAT differential filter (§4); **alias rejection is still set by the 4.7 nF `CFilt` at the ADC**, not by the cable alone.

**2) Differential signal (wanted EMG)**

For a balanced pair, differential capacitance is roughly `C_diff ≈ |C_P − C_N|` (mismatch), not `C_P + C_N`. A crude high-frequency bound:

```text
|V_diff,error| / |V_CM| ≈ ω × |C_P − C_N| × Z_CM,source     (small-signal)
```

At **50 Hz**, even **ΔC = 20 pF** gives a tiny coupling compared with electrode offset and line interference; **prep, shield, and DRL** dominate, not `C'` at 1.5 m.

**3) DC / low-frequency — wire resistance**

Electrode–skin impedance is **kΩ–MΩ** at DC; cable `R_wire` adds in **series on each leg**:

```text
V_drop,leg = I_bias × R_wire ≈ I_bias × R' × L
```

With `I_bias ~ 10 nA` (ADS1299 input bias) and `R_wire = 0.15 Ω` at 3 m, `V_drop` is **sub-nV** — negligible.

**4) On-HAT filter (dominant HF network at the chip)**

After the patient connector pin, the designed differential pole is set by layout at the ADC (§4), not by the cable:

```text
f_c,onboard ≈ 1 / (2π × R_eff × C_Filt)
```

with `C_Filt = 4.7 nF` and effective `R_eff` set by the leg series resistors and source impedance — still **~100 kHz–1 MHz class**, i.e. **above EMG**.

**5) ESD capacitance (fixed on HAT, not length-dependent)**

Each leg sees **~0.7 pF** from **TPD4E1B06** (§4) after `R_series` — unchanged if you lengthen the cable.

#### Recommended lengths (all architectures)

| Segment | Nominal length | Max (engineering) | Notes |
| ------- | -------------- | ----------------- | ----- |
| Collar → HAT (each EMG cable) | **1.2–1.5 m** | **2 m** per cable | Re-bench noise/CMRR if longer ([§13](#13-validation-acceptance-criteria)) |
| Collar → electrode fork (regional) | **20–30 cm** (face); **~8 cm** (digastric) | — | Anatomy-dependent; [§10.2.2](#1022-option-b--regional--anatomical-h-con-01r--e-02r) |
| Electrode pigtail (unshielded) | **5–10 cm** | Keep short | Industry default |
| **E-03** extension | **+1.5 m**/ch | **≤ 3 m** total shielded/ch | **H-CON-01F** only |

Cable **type** choices are in [§10.2.7](#1027-cable-types-and-sourcing).

**IMU cables** use a different bus (I2C); see [`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.3.1.

### 10.2 Patient harness and HAT connector architectures

Every architecture carries the same **13 conductors** and the same on-HAT filter network (§4). They differ in **HAT connector type**, **cable shielding**, and **how cables are grouped on the subject**.

#### Overview

| Option | HAT IDs | Lead set | HAT mate points | Patient cables | Signal quality | Procurement | Panel fit (56.5 mm) |
| ------ | ------- | -------- | --------------- | -------------- | -------------- | ----------- | ------------------- |
| **0 — Bring-up** | **H-CON-01CY** | **E-02CY** | **13** | 13× unshielded snap→Dupont | Adequate for **bench / short runs**; not ≤1 µVrms class | **OpenBCI shop** (~2 days) | **~33 mm** pin row — fits one edge |
| **A — Trunk** | **H-CON-01T** | **E-02T** | **1** | 1× 6+1 + collar spider | Best shield continuity | Custom trunk; bionen/OT class | **1× M12** |
| **B — Regional** | **H-CON-01R** | **E-02R** | **4** | 3× dual-ch + 1× DRL | Production target | Custom 4-pin LEMO→4-snap | **4× M12** — split 2+2 |
| **C — Per-channel** | **H-CON-01C** | **E-02C** | **7** | 6× bipolar + 1× DRL | Production target | Witacon 2-snap ×6 | **7× M12** |
| **F — DIN** | **H-CON-01F** | **E-02F** | **13** | 13× shielded fan-out | Production if shielded | bionen + DIN jacks | **Does not fit** one edge |

**Lifecycle recommendation:**

| Phase | Use | Why |
| ----- | --- | --- |
| **First PCB bring-up** | **Option 0** | Same ADS1299 + **connector/cable** idea as Cyton/OpenBCI (pin headers + unshielded snap leads); on-HAT filter stays **§4** (100 Ω differential); buy cables today; prove register/SPI/DRL/lead-off before investing in LEMO |
| **Production sessions** | **Option B** (default) or **A/C** | Shielded pairs; meets noise / routing targets in §11 |

Reference: OpenBCI Cyton uses the same **ADS1299** + **TPD4E1B06** + **DRL** idea with **2.54 mm header pins** and **passive unshielded snap leads**, but a **different input R/C filter** (**2.2 kΩ + 1 nF per leg to AGND** — see [§4.2](#42-openbci-cyton-input-filter-vs-this-design)). [Cyton specs / schematic](https://docs.openbci.com/Cyton/CytonSpecs/#openbci-cyton-board-circuit-schematic); [EMG/ECG snap cables](https://shop.openbci.com/collections/frontpage/products/emg-ecg-snap-electrode-cables).

**Anatomical grouping rule:** group muscles that are **close on the body** into one cable; **never** pair bilateral homologues (e.g. masseter L + masseter R) or cross the face (digastric + masseter).

The grouping is by muscle; the channel numbers follow the montage of
[an03 §1](../analysis/an03_emg_processing.md#1-montage).

| Regional cable | Muscles | Why this grouping |
| -------------- | ------- | ----------------- |
| **`J_DIG`** | Digastric L + R | Both submandibular; ~6–10 cm apart |
| **`J_L`** | Masseter L + temporalis L | Same side of face; ~10–15 cm fork |
| **`J_R`** | Masseter R + temporalis R | Mirror of `J_L` |
| **`J_DRL`** | Mastoid bias | Single conductor |

**Do not group:** masseter L + masseter R (cheeks too far apart); temporalis L + temporalis R; digastric + cheek muscles on one cable. *(Applies to **B** only — Option **0** uses thirteen separate wires.)*

#### 10.2.0 Option 0 — Bring-up (OpenBCI / Cyton-style) (H-CON-01CY / E-02CY)

```text
electrode (E-01) ── snap ── unshielded single wire ── Dupont female ── 2.54 mm pin on HAT
                                                                                │
                                                                         (100 Ω → ESD → choke → INxP/N)
                                                                                │
                                                                         DRL pin → DRL filter → IN8N (MUX8=111)
```

- **HAT:** **2× 1×6** + **1× 1×1** **2.54 mm (0.1″) pin headers** (`J_EMG_N1`, `J_EMG_P1`, `J_EMG_DRL1`) — **13 pins** total, one pin per conductor (same electrical idea as OpenBCI Cyton **`N1P`…`N8N`** breakout pins). THT vertical is fine for bench. Alt: **1×13** strip or **13× 1×1**.
- **Harness:** **13×** passive **unshielded** leads — electrode **snap** at the body, **Dupont female** (0.1″ socket) at the HAT. Source: [OpenBCI EMG/ECG snap electrode cables](https://shop.openbci.com/collections/frontpage/products/emg-ecg-snap-electrode-cables) (10-pack) **+ 3 spare** single leads, or any equivalent unshielded snap→Dupont cable.
- **Electrodes:** **Ambu BlueSensor P-00-S** (E-01, as used) — snaps mate directly.
- **No cable shield:** there is **no shell → AGND** at the patient connector (unlike A/B/C/F). CMRR relies on **on-HAT** chokes + **DRL** + short bench runs. Keep cables **≤ 0.5–1 m** for first tests; expect more **50/60 Hz** than shielded harnesses.
- **Pin map** (`J_EMG_P1` pins 1–6 = IN1P…IN6P, `J_EMG_N1` pins 1–6 = IN1N…IN6N, `J_EMG_DRL1` = DRL; match OpenBCI color habit if using their 10-pack). The
  muscle on each channel is the montage of [an03 §1](../analysis/an03_emg_processing.md#1-montage).

| HAT pin | ADS1299 net | Channel | OpenBCI-style cable |
| ------------- | ----------- | ------- | ------------------- |
| **IN1P** | IN1P | CH1 + | assign one color → this pin |
| **IN1N** | IN1N | CH1 − | paired color |
| **IN2P** | IN2P | CH2 + | … |
| **IN2N** | IN2N | CH2 − | … |
| **IN3P** | IN3P | CH3 + | … |
| **IN3N** | IN3N | CH3 − | … |
| **IN4P** | IN4P | CH4 + | … |
| **IN4N** | IN4N | CH4 − | … |
| **IN5P** | IN5P | CH5 + | … |
| **IN5N** | IN5N | CH5 − | … |
| **IN6P** | IN6P | CH6 + | … |
| **IN6N** | IN6N | CH6 − | … |
| **DRL** | DRL (bias path) | — | one lead → **DRL** pin only (mastoid) |

- **Pros:** **fastest path** to validate a custom PCB (SPI, DRDY, lead-off, DRL, internal test signal); **no custom cable order**; smallest connector BOM; Cyton-proven topology.
- **Cons:** **not** for production noise targets (§11); bare pins on HAT — **do not** use with mains-powered bench clutter without care; no touch-proof at HAT; migrate to **B** before subject studies that need quantitative EMG.

**KiCad capture:** see [`h03_emg_schematic_build_guide.md`](h03_emg_schematic_build_guide.md) §“Option 0 — connector capture” — the connector options differ only in the connector symbols, not in the ADS1299 filter nets.

#### 10.2.1 Option A — Single trunk (H-CON-01T / E-02T)

```text
electrode (E-01) ──[5–10 cm pigtail]── snap / touch-proof DIN 42802 (per lead)
                                                  │  individually shielded twisted pair
                                          collar "spider" breakout
                                                  │  6 shielded pairs + DRL + overall braid
                                          ONE circular trunk plug (14-pin class)
                                                  │  360° shield → shell
                                          ONE HAT receptacle (J_EMG) ──► AGND
```

- **HAT:** one **14-pin** circular receptacle (LEMO `ECG.1B.314.CLN`, ODU MEDI-SNAP size 2, or Binder 423 M16-16).
- **Harness:** custom **6+1** cable — six **individually shielded** twisted pairs plus DRL, with a molded or lab-built **spider** at the collar ([`op02`](../operations/op02_subject_preparation.md) §6).
- **Example trunk pin map (silkscreen on HAT):** pins 1–2 = IN1P/N (CH1), 3–4 = IN2P/N (CH2), … 11–12 = IN6P/N (CH6; muscles per an03 §1), pin 13 = DRL (pin 14 NC/spare).
- **Pros:** one mate/demate; best 360° shield continuity; smallest HAT panel hole count.
- **Cons:** no ready-made 13-lead stock SKU; spider + trunk is a **custom harness-shop** job.

#### 10.2.2 Option B — Regional / anatomical (H-CON-01R / E-02R)

```text
                    [Temp L]───┐
                               ├── J_L (4-pin) ──1.2 m──► collar ──► HAT J_L
                    [Mass L]───┘

                    [Temp R]───┐
                               ├── J_R (4-pin) ──1.2 m──► collar ──► HAT J_R
                    [Mass R]───┘

                    [Dig L]────┐
                               ├── J_DIG (4-pin) ──1.0 m──► collar ──► HAT J_DIG
                    [Dig R]────┘

                    [Mastoid] ── J_DRL (2-pin) ──1.2 m──► HAT J_DRL
```

- **HAT:** **3× 4-pin** + **1× 2-pin** circular receptacles (e.g. LEMO `ECG.1B.304.CLN` ×3, `ECG.1B.302.CLN` ×1). Each **shell → AGND** ([`h07`](h07_power_grounding_safety.md) §2.3).
- **Harness:** three custom **dual-bipolar** cables (2× individually shielded pairs each, Y-fork at face/chin) + one **DRL** lead. Order as custom medical cable (Witacon, Heymed, cngoochain class) — see [§10.2.7](#1027-cable-types-and-sourcing).
- **Collar:** all four EMG cables + IMU cables meet at the strain-relief clip; run **bundled in one velcro sleeve** to the HAT ([`op02`](../operations/op02_subject_preparation.md) §6).
- **Pin map (silkscreen on HAT)**, with the montage of an03 §1 — `J_L`: masseter L (CH5) + temporalis L (CH6); `J_R`: masseter R (CH1) + temporalis R (CH2); `J_DIG`: digastric R (CH3) + digastric L (CH4):

| Receptacle | Pin 1 | Pin 2 | Pin 3 | Pin 4 |
| ---------- | ----- | ----- | ----- | ----- |
| **`J_L`** | IN5P | IN5N | IN6P | IN6N |
| **`J_R`** | IN1P | IN1N | IN2P | IN2N |
| **`J_DIG`** | IN3P | IN3N | IN4P | IN4N |
| **`J_DRL`** | DRL | NC | — | — |

- **Pros:** matches operator mental model and cable routing; **buyable** as 4-pin custom assemblies; per-channel shielding preserved.
- **Cons:** four mate points; **4× M12** holes need careful layout on 56.5 mm ([`h02`](h02_custom_hat_shield.md) §2.1).

#### 10.2.3 Option C — Per-channel (H-CON-01C / E-02C)

```text
[Mass R bipolar] ── J_CH1 (2-pin) ──┐
[Temp R bipolar] ── J_CH2 (2-pin) ──┤
[Dig R bipolar]  ── J_CH3 (2-pin) ──┤
[Dig L bipolar]  ── J_CH4 (2-pin) ──├── collar bundle (~1.2 m each) ──► 7× HAT receptacles
[Mass L bipolar] ── J_CH5 (2-pin) ──┤
[Temp L bipolar] ── J_CH6 (2-pin) ──┤
[DRL mastoid]    ── J_DRL (2-pin) ──┘
```

- **HAT:** **7× 2-pin** circular receptacles (LEMO `ECG.1B.302.CLN` or compatible), one per bipolar channel + DRL. Each **shell → AGND**.
- **Harness:** **6×** single-bipolar shielded cables (2 conductors + shell) + **1×** DRL cable. **Witacon** [LEMO 1B → 2-snap](https://www.witacon.com/Circular-Connector-Cable/LEMO-1B-Connector-To-Electrode-Snap-Cable/) is the closest off-the-shelf product (order six matching channel colors + one DRL).
- **Connector labels:** `J_CH1` … `J_CH6` = CH1 … CH6 (muscles per an03 §1, [§2](#2-channel-assignments)); `J_DRL` = mastoid bias.
- **Pros:** simplest procurement (no 4-pin custom fork); each channel is independently replaceable.
- **Cons:** **7 mate points**; more collar clutter unless bundled; **7× M12** on HAT needs panel extension or two rows ([`h02`](h02_custom_hat_shield.md) §2.1). No signal-quality advantage over B or A if pairs are individually shielded.

#### 10.2.4 Option F — DIN (shielded) (H-CON-01F / E-02F)

- **HAT:** **13×** touch-proof **DIN 42802** 1.5 mm jacks — **shielded** medical lead option (not the same as Option **0**).
- **Harness:** thirteen individual **shielded** wires (bionen **0044.xxx** snap + DIN TP at amp end) fanned at the bench.
- **Use when:** per-conductor **touch-proof at the device** side is required **and** shielded leads are already on hand. **Not** the recommended first-iteration path — prefer **Option 0** for speed, **Option B** for production. Needs panel extension ([`h02`](h02_custom_hat_shield.md) §2.1).

#### 10.2.6 Connector family trade study (premium vs. medical-mid vs. budget)

Pick the **same connector family** across all mate points in an architecture (do not mix LEMO plugs with Binder receptacles). Shield always lands on **shell/feedthrough → AGND** at the HAT.

| Family (example) | Class | Shielding | Touch-proof at device | Typical inserts | Locking | Notes |
| ---------------- | ----- | --------- | --------------------- | --------------- | ------- | ----- |
| **LEMO 1B** | **Premium** | 360° shell | Recessed, keyed | **2** (`1B.302`), **4** (`1B.304`), **14** (`1B.314`) | Push-pull | M12 panel hole; cable OD **5.3–6.2 mm** for 4-pin class |
| **ODU MEDI-SNAP** size 2 | **Medical mid** | 360° feedthrough in plastic | **Yes** (IEC 60601-1) | 2–16 contacts | Push-pull / break-away | Best if device-side touch-proof is required |
| **Binder 423 M16** | **Budget** | Shield ring + backshell | Keyed, not medical TP | **2**, **4**, **16**, … | Screw | Heavier; AWG24 max; fine for bench |
| **LEMO-compatible B-series** (Witacon, etc.) | **Value** | Varies — verify sample | Same as LEMO class | Match receptacle insert | Push-pull | **Sample before batch**; confirm mate force and shell continuity |

**By architecture:**

| Architecture | Premium (LEMO) | Budget (Binder 423) |
| ------------ | -------------- | ------------------- |
| **A Trunk** | `ECG.1B.314.CLN` / `FGG.1B.314.CLAD` | `99-5455-15-16` |
| **B Regional** | `ECG.1B.304.CLN` ×3 + `ECG.1B.302.CLN` ×1 | M16-4 ×3 + M16-2 ×1 |
| **C Per-channel** | `ECG.1B.302.CLN` ×7 | M16-2 ×7 |

#### 10.2.7 Cable types and sourcing

**Hard requirement (production A/B/C/F):** each EMG **channel** must use a **dedicated shielded twisted pair** (or equivalent — shield tied to shell at the HAT). **Option 0 explicitly waives this** for bring-up only.

| Cable / product | Construction | Best for | Avoid for |
| --------------- | ------------ | -------- | --------- |
| **OpenBCI EMG/ECG snap cables** | 1× unshielded wire, snap + Dupont | **E-02CY** Option **0** bring-up | Production / long runs |
| **Custom medical TPU**, 2× indiv. shielded pairs, 24 AWG | Per-pair foil/braid | **E-02R** regional (Witacon/Heymed/cngoochain quote) | — |
| **Custom 6+1 trunk** | 6 indiv. pairs + DRL + braid | **E-02T** | — |
| **Witacon LEMO 1B → 2-snap** | 1 shielded pair | **E-02C** per-channel (×6 + DRL) | Regional 4-snap fork (not stock) |
| **Belden 8723** (3-pair, **each pair indiv. shielded**) | DIY cut + LEMO crimp | **E-02R** DIY (one length per regional cable; use 2 of 3 pairs) | Trunk (only 3 pairs per cut) |
| **Belden 9841 / Alpha 5162C** (indiv. shield pairs) | DIY | Trunk or per-channel | — |
| **Belden 9506** (6-pair, **overall shield only**) | RS-232 class | — | **All EMG options** — no per-pair shield |
| **Cat6a S/FTP** (per-pair foil + braid) | DIY emergency | Per-channel only; awkward fork | Regional / trunk without careful pair assignment |
| **bionen 0044.xxx** (single snap, DIN TP amp end) | 1 conductor + shield | **E-02F** with DIN on HAT | Circular HAT without adapter |

**Regional custom order checklist (copy to vendor):**

1. Connector: **LEMO 1B.304** male (or mate to your receptacle), **shield → shell**.
2. Cable: **2× individually shielded twisted pairs**, **24 AWG**, medical **TPU**, OD ≤ **6.0 mm**.
3. Patient end: **4× female 3.5 mm snaps** (Ambu BlueSensor P-00-S), red/black × 2 channels.
4. Lengths: **digastric** — common **120 cm**, fork **~8 cm**; **face L/R** — common **130 cm**, forks **~20 cm** (masseter) and **~30 cm** (temporalis) from collar split.
5. Order **one sample** before batch; verify shell continuity and mate to **ECG.1B.304.CLN**.

**Why grouping does not hurt signal quality.** EMG noise is dominated by **common-mode** pickup (50/60 Hz), rejected by DRL + ADS1299 CMRR + well-terminated shields — not by inter-channel crosstalk in the EMG band (<450 Hz). A regional 4-pin cable with two isolated pairs is **electrically equivalent** to two separate 2-pin cables, as long as each pair stays shielded to the pigtail.

**Touch-proof, restated.** IEC 60601-1 touch-proof at the **electrode/lead** end (snap or DIN 42802). HAT-side circular connectors are keyed and not mains-pluggable. **ODU MEDI-SNAP** additionally provides touch-safe **device-side** housings with shield feedthrough.

**What we did *not* do in v1:** SPICE/lab verification of every harness SKU; numbers in §10.1 are **design estimates**. Before adopting **> 2 m** or a new cable vendor, measure **`C'`** on a sample and repeat the lead-off / 50 Hz noise bench tests (§13).

## 11. Expected performance

With this design and good electrode contact:

| Metric                                              | Target   | Typical (ADS1299 datasheet)    |
| --------------------------------------------------- | -------- | ------------------------------ |
| Input-referred noise (gain 24, bandwidth 0.1–70 Hz) | <1 µVrms | 0.98 µVrms                     |
| CMRR @ 50 Hz                                        | >110 dB  | 120 dB                         |
| Sample rate                                         | 4 kSPS   | 250 SPS – 16 kSPS configurable |
| Crosstalk between channels                          | <-90 dB  | -95 dB                         |
| Input bias current                                  | <200 pA  | 100 pA typical                 |

Real-world performance is electrode- and prep-limited and is typically 2–10× worse than the ideal numbers above. Skin-electrode noise dominates.

## 12. Component-level alternatives

| Reason to deviate                   | Alternative                                                                      | Trade-off                                                                                                                                      |
| ----------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| ADS1299 not available               | **ADS1298**                                                                      | 8-ch, 24-bit, slightly older but available; pin-compatible-ish, firmware needs small adjustments                                               |
| Need higher sample rate per channel | **ADS131M08**                                                                    | 8-ch, 24-bit, 32 kSPS, simultaneous sampling, lower per-channel current but no integrated bias-drive amp; would require external opamp for DRL |
| Cost-down                           | discrete **INA333** + **MCP3564**                                                | Cheaper but loses lead-off detection and integrated bias drive; more PCB area, more firmware work                                              |
| Need true galvanic isolation        | ADS1299 + **ADuM3151 isolated SPI** + **ADuM3160 isolated USB** + isolated DC-DC | +30 EUR BOM, larger PCB, complete medical-grade isolation; documented as v2 path                                                               |
| Need EEG (not EMG)                  | same ADS1299 with gain 24, low-pass at 70 Hz, active electrodes                  | Architecture supports it; firmware register settings change                                                                                    |

## 13. Validation acceptance criteria

The EMG front-end is accepted when, on the bench:

- All 8 channels read CHIPID = 0x3E and respond to register reads.
- Internal test signal (1 mV, 1 kHz square wave) appears on all 8 channels with the correct amplitude after gain.
- Input-referred noise with inputs shorted is <1.5 µVrms in the 0.1–70 Hz band (allow 50 % margin over datasheet ideal).
- Crosstalk between adjacent channels, with one channel driven by a 1 mV sine and the others shorted, is <-80 dB.
- Lead-off detection correctly flags a disconnected channel within 2 samples.
- Driven-right-leg, when connected to a body, drops the 50 Hz line noise by >40 dB compared to bias-off operation.

## 14. References

- TI ADS1299 datasheet (SBAS499) and reference design (TIDA-00922).
- Couby / OpenBCI hardware KiCAD: [https://github.com/Couby/OpenBCI_hardware_KiCAD](https://github.com/Couby/OpenBCI_hardware_KiCAD).
- SENIAM project — surface EMG sensor placement: [http://www.seniam.org/](http://www.seniam.org/). We use SENIAM-style placement for masseter and temporalis; the anterior digastric is not in SENIAM and we follow Castroflorio et al. 2005 for that.
- Castroflorio T. et al., *"Surface EMG of jaw-elevator muscles: effect of electrode location and inter-electrode distance"*, J Oral Rehabil 2005;32(6):411–417.
- Cooper D., *"Electrodes for surface electromyography"*, Delsys white paper, 2013.
