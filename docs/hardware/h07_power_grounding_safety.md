<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h07 — Power, Grounding, and Patient Safety

> **Building the schematic in KiCad?** See [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md).

This document specifies the power architecture of the HAT, the grounding discipline that lets the analog front-end achieve <1 µVrms input-referred noise, and the (research-grade) patient-safety practices we follow. The system is a **research instrument, not a medical device**; we do not pursue IEC 60601-1 certification, but we explicitly document where this prototype falls short of medical-device requirements so that a future v2 can address them.

<a id="1-power-tree"></a>

## 1. Power tree
**Default build (AprilTag / ArUco fiducials, 2× GS cameras on XTRIG):** the HAT powers EMG, MCU, and camera triggers. The **visible sync LED** and **USB opto** paths are **optional** ([`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §7). It does **not** power per-camera **IR rings** for retroreflective spheres. IR ring drivers are **DNP** on the PCB; optional rings use **external 5 V supplies** ([`../04_component_catalog.md`](../04_component_catalog.md#9-fiducials--scene-lighting) optional retro kit).

```
Mains 230 V AC
   |
   v
  Pi 5 27 W USB-C PSU (SC1112)
   |
   +-- 5 V/5 A to Pi 5 via USB-C
   |
   v
Pi 5 GPIO header (5 V, pin 2/4) ──> HAT  [DEFAULT BUILD]
   |
   +─> +3V3D digital LDO (TPS73633DBV; EN=VIN, NR+10nF)  ──> RP2350A 3.3 V domains (IOVDD/QSPI_IOVDD/USB_OTP_VDD/ADC_AVDD + VREG_VIN), level translators, ADS1299 DVDD (3.3 V)
   |       └─> RP2350A on-chip SMPS buck: VREG_VIN(3.3V) ──> VREG_LX ──> [3.3 µH] ──> DVDD = 1.1 V (MCU core)
   |
   +─> +5VA low-noise analog rail  ──> ADS1299 AVDD (5 V) only
   |       generation:  Pi 5 V ──> boost ~6.5 V ──> ADP7142AUJZ-5.0 ──> +5VA = 5.0 V
   |       (the ADS1299 REQUIRES AVDD−AVSS = 4.75–5.25 V, SBAS499 §7.3)
   |
   +─> trigger-out + IMU JST-SH (3.3 V logic; IMU breakouts powered from +3V3D)

   [OPTIONAL — DNP on default GS build, see h02 §7.1b]
   +─> SYNC_LED (~20 mA, visible white LED on JST, gated by GPIO)  ← Track A timing only

   [NOT IN DEFAULT BUILD — DNP on PCB, see §1.4]
   +─> RING_PWR / RING_EN (unpopulated) ──> would feed external IR rings ~250 mA each
```

> **RP2350A core supply (1.1 V DVDD).** Unlike the legacy RP2040 (internal LDO core), the RP2350A generates its **1.1 V core supply on-chip with an SMPS buck converter** (datasheet §6.3.7–6.3.8, Figure 19, the "combined supply" topology). It is fed from **+3V3D** at `VREG_VIN` and switches at `VREG_LX` into one external **3.3 µH** inductor, with **CIN/COUT = 4.7 µF** and a **33 Ω + 100 nF** RC on `VREG_AVDD`; values are in [`../04_component_catalog.md`](../04_component_catalog.md) **H-PAS-08** and [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.1. The switching node stays entirely on **DGND/+3V3D**, on the digital side of the star ground (§2), so its noise never reaches the analog domain. **Ultra-low-noise alternative:** the on-chip regulator can be disabled and `DVDD` fed from an external 1.1 V LDO (datasheet Figure 21, no inductor); not used in v1.
>
> **Why a 5 V analog rail (and why boost-then-LDO).** The ADS1299 is a 5 V analog part: its Recommended Operating Conditions (SBAS499 §7.3) require **AVDD − AVSS = 4.75–5.25 V**, and all datasheet specs are characterized at **AVDD−AVSS = 5 V, VREF = 4.5 V**. Running AVDD at 3.3 V is out of spec. We therefore generate a dedicated **+5VA** rail. Because a linear regulator cannot drop 5 V → 5 V (no headroom), we **boost the Pi's 5 V to ~6.5 V and then use a low-noise fixed 5.0 V LDO (ADP7142AUJZ-5.0)** to clean it up. The boost is a switcher and lives on the **digital/supply side** (DGND), with its output LC-filtered before the LDO so its switching noise never reaches AGND.
>
> **Reference vs supply:** **+5VA** feeds **AVDD** only. **VREF = 4.5 V** comes from the **ADS1299 internal reference buffer** (enabled in firmware), not from a separate LDO or REF5025. That gives **full datasheet dynamic range** (±VREF/gain) and matches TI's characterized corner — see [`h03`](h03_emg_frontend.md) §3 and §9.

**Not on this tree:** scene lighting for AprilTags (diffuse LED panels or room lights, mains-powered), camera modules (5 V from Pi USB or separate supply), and the optional retroreflective kit (IR rings + filters powered locally).

### 1.1 Current budget — default build (AprilTag)

| Load | Source | Typical | Max | Notes |
|------|--------|---------|-----|-------|
| **5 V HAT total** | Pi GPIO pin 2/4 | **~200 mA** | **~270 mA** | Well under Pi 5's ≥1 A header budget (add ~20 mA if optional **SYNC_LED** populated) |
| +3V3D rail | TPS73633DBV | 100 mA | 160 mA | RP2350A (incl. on-chip 1.1 V core SMPS, dual M33 @150 MHz) + USB-CDC + IMU JST (logic/supply to daughterboards). The core buck draws from +3V3D; reflected core current is small (1.1 V load ÷ 3.3 V in, η≈80 %), regulator can deliver up to ~200 mA on DVDD. |
| +5VA rail | boost + ADP7142AUJZ-5.0 | 35 mA | 65 mA | ADS1299 AVDD only (internal 4.5 V ref — no external ref IC) |
| SYNC_LED *(optional)* | 5 V via GPIO + FET | 0 | 25 mA | **DNP** on default GS build; Track A only ([`h06_synchronization.md`](h06_synchronization.md) §4) |
| Camera triggers | 3.3 V PIO | &lt;5 mA | 10 mA | **XTR+** via **R51/R53 + R57/R59** divider (~**1.8 V** at **J1/J2**); cameras powered elsewhere |

**Total at 5 V (default GS build):** ~1.0–1.2 W typical. **No IR ring** or **sync LED** current from the HAT unless those optional blocks are populated.

### 1.2 Current budget — optional retroreflective kit (not default)

If you populate `RING_PWR` on a dev spin **or** use the optional Qualisys comparison kit, IR rings are **not** included in the table above:

| Load | Source | Typical per ring | Notes |
|------|--------|------------------|-------|
| IR LED ring | **External 5 V** (wall adapter or lab supply) | ~200–250 mA | Not routed through HAT in default BOM; avoids EMI/heating near ADS1299 |
| 850 nm filters | Mechanical on lens | — | Camera/lens only; no HAT power |

Enabling IR rings does **not** change the EMG analog budget in the default layout because those loads are off-board. If you **do** populate the HAT ring driver footprints, add **~250 mA per ring** to the 5 V budget and re-check Pi header current (still usually OK for one ring; two rings may need external 5 V only).

### 1.3 Sequencing

The +3V3D and +5VA rails come up on 5 V arrival. On the **`power`** sheet, panel LEDs show **5 V in** (**D_5VIN**), **+3V3D** (**D_3V3D_OK**), boost regulation (**D_6V5_PG** on **TPS61252 PG**), and **+5VA** after the **ADP7142** (**D_5VA_OK**) — see [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md) Step 8. The ADS1299 tolerates AVDD and DVDD rising in either order, but it has a real **power-up timing requirement** (SBAS499 §11.1): after the supplies are stable you must wait for the internal LDO/charge-pump to settle — in practice until **VCAP1 has charged above ~1.1 V** and at least `tPOR` has elapsed — before issuing the first `/RESET`. With the datasheet-mandated **VCAP1 = 100 µF**, this ramp is on the order of tens of ms, so the firmware must not rush the reset. We add a 10 µF bulk capacitor on each rail to soften in-rush.

The RP2350A boots within ~20 ms of +3V3D rising; before then the ADS1299 SPI is unconfigured. The MCU's first action is the **datasheet power-up sequence** ([`h05`](h05_microcontroller_firmware.md) §3.4): hold `/RESET` low through the supply/VCAP1 ramp, wait `tPOR` (+ VCAP1 > 1.1 V), then issue `/RESET` and the `SDATAC` command so the ADS1299 wakes in a known state.

### 1.4 Optional IR ring path (DNP — reference only)

The v1 schematic may include **unpopulated** footprints `RING_PWR`, `RING_PWM`, and `RING_EN` for experiments with retroreflective spheres ([`../04_component_catalog.md`](../04_component_catalog.md#optional--retroreflective-kit-qualisys-comparison) optional kit). In the **shipping default**:

- Footprints are **DNP** (do not populate FET, inductor, or high-current connector).
- **No 5 V current** flows to IR rings through the HAT.
- Optical tracking uses **AprilTag 36h11** under visible light; marker dictionary set in `software/optical/etc/optical.yaml` (currently ArUco `DICT_4X4_50`) ([`../software/s03_camera_pipeline.md`](../software/s03_camera_pipeline.md)).

If populated on a lab spin, the electrical path would be: `Pi 5 V → PFET/MOSFET (RING_EN) → constant-current or resistor-limited 5 V bus → JST to ring(s)`. That path is intentionally omitted from §1.1 because it is not part of the validated build.

<a id="15-decoupling"></a>

### 1.5 Decoupling
Standard practice; we list it because it is the most common cause of analog board misbehavior. The **ADS1299-specific** values are fixed by the datasheet — see the per-pin table in [`h03`](h03_emg_frontend.md) §9 (1 µF at **each** AVDD/DVDD pin; **VCAP1 = 100 µF** C0G/NPO or tantalum; VCAP2 = 1 µF; VCAP3 = 1 µF ∥ 0.1 µF; VCAP4 = 1 µF; **VREFP ≥ 10 µF + 0.1 µF**). The general rules:

- 100 nF X7R + 10 µF X5R bulk on every IC supply rail (AVDD/AVSS at 5 V, DVDD/DGND at 3.3 V), **plus** the ADS1299 per-pin caps above.
- 100 nF + 10 µF at every regulator input and output (digital LDO, analog boost, analog LDO).
- **TPS73633DBV:** **10 nF** from **NR** to **DGND** (noise-reduction pin); **EN** tied to **VIN** if shutdown is unused.
- **VREFP gets ≥ 10 µF + 0.1 µF to AVSS** (SBAS499 Pin Functions) — *not* a small 47 pF cap; do not under-decouple the reference.
- VCAP1 (100 µF) must be **C0G/NPO or tantalum** (non-ferroelectric), placed right at the pin.
- Per SBAS499 §12.1: **no vias between a bypass cap and the device pin** — place each cap on the same layer, as close to the package as possible.
- 100 nF across the BNO085 supply, plus a 4.7 µF on the breakout (already populated).
- USB-C power: 4.7 µF + 100 nF immediately at the connector.

<a id="2-grounding-discipline"></a>

## 2. Grounding discipline
<a id="21-ground-domains"></a>

### 2.1 Ground domains
We split the L2 ground plane into **AGND** and **DGND** with a single connection point ("star tie") under the ADS1299. The split runs vertically across the board, separating the analog region (one short-side third) from the digital region (the rest).

| Domain | What lives on it |
|--------|-------------------|
| AGND | ADS1299 AVSS/AGND pins; VREFN; input filter caps; **analog LDO (ADP7142AUJZ-5.0) GND/output side**; **every EMG patient connector shell** (`J_EMG` / `J_L`… / `J_DRL` — one tie each); IMU cable shields (pin 7) |
| DGND | RP2350A GND/EP + **VREG_PGND (core-buck switching return)**; USB-C shell + signal GND; level translator GND; digital LDO GND; **analog-rail boost converter GND (switcher stays on the digital side)**; trigger-out connector GND; IMU JST-SH GND |

<a id="22-the-star-tie"></a>

### 2.2 The star tie
A single **KiCad net tie** (`NT_GND_TIE1`, `Device:NetTie_2` on the `emg` sheet) connects **AGND** and **DGND** at the ADS1299 — PCB placement between **pad 32 (`AVSS`)** and **pad 31 (`RESV1` / `DGND`)**, with short traces to those pads and vias into the `In1.Cu` planes. This is the **only** path between the two domains on the board itself. The Pi 5's GPIO header GND pins all connect to DGND only.

The reason: high-frequency switching currents from the RP2350A (incl. its 1.1 V core buck) / USB / cameras flow through DGND and return to the LDOs along the cheapest impedance path. If DGND and AGND are stitched at multiple points, those currents partially flow through AGND, contaminating the EMG reference. The single star tie forces all return currents back to the LDO through a known path.

<a id="23-shield-grounding"></a>

### 2.3 Shield grounding
Shielded cables (EMG patient cable, IMU cables, trigger fan-out) have their shields tied to **AGND at the HAT side only**. The far end of the shield is floating to avoid ground loops. This is a deliberate, well-known compromise: a shield grounded at both ends would catch more high-frequency RF but creates a ground loop that can pump 50 Hz line currents through the cable, swamping the ~µV EMG signal.

**IMU harness (v1):** shield is **not** on JST pin 2 (`DGND`). Use **7-pin** JST-SH/GH with **pin 7 = `IMU_SHLD` → `AGND`** on the HAT PCB ([`h04_imu_cable_assembly.md`](h04_imu_cable_assembly.md)). Do not rely on connector shell/`MP` pads alone.

> Note the deliberate asymmetry: the IMU and trigger **signal returns (GND conductor)** live on **DGND** (§2.1), but the **cable shields** of those same cables terminate on **AGND**. Shield current therefore returns to AGND and crosses to DGND only through the single star tie — this is intentional and keeps shield noise out of the digital return. Do not "fix" it by bonding shields to DGND or IMU pin 2.

### 2.4 Layout guidelines (ADS1299 SBAS499 §12.1 compliance)

TI's "good design can be ruined with a bad circuit layout" guidance (datasheet §12.1, Figure 79) maps onto our board as follows. Each datasheet bullet is paired with the concrete rule for this HAT:

| SBAS499 §12.1 guideline | How we comply on the JTS-MM HAT |
|--------------------------|----------------------------------|
| **Separate analog and digital sections; route digital lines away from analog** | The board is partitioned into an **analog third** (short side: ADS1299, VREF decoupling, input RC filters, **EMG patient connector(s)**, analog LDO) and a **digital region** (RP2350A + its **core-buck inductor and VREG_LX switch node**, USB-C, level translators, boost switcher, triggers). Digital traces (SPI, USB, trigger PIO) stay over the digital region; no digital net crosses the analog area. **Keep the VREG_LX → 3.3 µH → DVDD loop tight** (minimum loop area, on the digital side, away from the AGND/analog region). |
| **Ground plane *may* be split into AGND/DGND (not mandatory); place digital signals over digital plane, analog over analog; join the split at the ADC** | We **do** split the L2 plane into AGND and DGND and connect them at a **single star tie directly under the ADS1299** (§2.2). Analog signals sit over the AGND pour, digital over the DGND pour. The join-at-the-ADC rule is exactly our star tie. |
| **Fill void areas on signal layers with ground fill** | All signal layers (L1/L4) get **ground fill** stitched to the adjacent plane (AGND fill in the analog region, DGND fill in the digital region) with via stitching, leaving no large copper voids. |
| **Provide good ground return paths; don't cut the plane under signal traces** | The L2 reference plane is kept **continuous under every signal trace**; the only intentional break is the AGND/DGND split, and no trace is routed across that split except where it crosses at/near the star tie. No traces are run on the plane layer that would force return current to detour. |
| **Bypass caps: no vias between cap and device; same layer, as close as possible** | Every ADS1299 supply/VCAP/VREFP cap (1 µF/pin, VCAP1 = 100 µF, VREFP ≥ 10 µF — see [`h03`](h03_emg_frontend.md) §9) is placed on the **same layer as the device, directly at the pin, with no via between the cap pad and the pin**. Fan-out vias go on the far side of the cap. |
| **Differential cap across analog inputs; use high-quality C0G/NPO** | Each EMG channel has a **4.7 nF C0G (NPO) capacitor placed differentially across INxP–INxN** at the package ([`h03`](h03_emg_frontend.md) §4). C0G is mandatory for low noise and stable capacitance; no X7R/X5R on the differential input cap. |

Figure 79's recommended placement (supply generation and microcontroller on one side, signal conditioning + connector on the analog side, ground fill everywhere, optional split cuts) is the template the PCB follows; the specifics (geometry, layer use, shield-can keep-out) are in [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §3.

## 3. Patient-safety analysis (research-grade, not certified)

The system contacts the subject through:

- 6 EMG signal electrodes (single-fault isolation distance: shared with the bias electrode).
- 1 bias/DRL electrode (the current-limiting path).
- MTA mouthpiece (no electrical contact; it touches teeth only).
- CRA forehead band (no electrical contact).
- IMU breakouts (3.3 V on the breakout, but no electrical contact with skin: PETG casing).

The single failure mode that can deliver current to the subject is an EMG electrode shorted to a voltage source (mains via the HAT's power supply, in the limit). We mitigate as follows:

1. **Patient current limit via the 220 kΩ DRL series resistor**: the DRL/bias output swings within the analog supply, so the worst case is the full **5 V** AVDD rail through 220 kΩ ≈ **23 µA** — still well below the 100 µA AAMI ES60601-1 patient-leakage limit for type-BF applied parts. (At 3.3 V it would have been ~15 µA; the move to the required 5 V analog supply only changes this slightly and remains comfortably in spec.)
2. **Touch-proof at the electrode end:** **DIN 42802 1.5 mm** (or snap) on production leads. **Option 0 (bring-up):** snap only; **bare 2.54 mm pins** on the HAT — bench use with short cables ([`h03`](h03_emg_frontend.md) §10.2.0).
3. **Isolated USB hub** between the Pi 5 and any mains-tied peripheral (printer, monitor, second computer). Off-the-shelf USB 3.0 isolators are ~30 EUR and provide 1.5 kV isolation; cheap insurance.
4. **No mains-powered equipment within reach of the subject**. The Pi 5 PSU sits on the bench; the subject is on the other side of the desk; cabling is the only electrical bridge.

### 3.1 What this does **not** give us

- We are not certified for clinical use. No IEC 60601-1 compliance testing has been done.
- The HAT itself is not isolated: a fault in the Pi 5's PSU (5 V rail shorted to mains, vanishingly rare in a CE-marked PSU but theoretically possible) would propagate to the HAT and through the patient cable to the subject. The 220 kΩ DRL resistor limits the worst-case current to a few mA, which is uncomfortable but not lethal; nevertheless this is below medical-grade defense-in-depth.
- The system **must not** be used during electrosurgery, defibrillation, or any high-voltage transient procedure.

### 3.2 Recommended operator practices

> **Human-subjects SOP:** [`op04_emg_study_safety_sop.md`](../operations/op04_emg_study_safety_sop.md) — one-page checklist for ethics submissions and operator training.

- The operator runs the recording from a different mains-connected device (laptop) and accesses the Pi via SSH or a router. The subject is not connected to the operator's laptop directly.
- The subject sits on a non-conductive chair (standard office chair is fine).
- No conductive jewelry near the electrodes.
- Each session uses fresh, single-use Ag/AgCl electrodes.

## 4. Path to a v2 medically-isolated variant

If a future iteration requires medical-grade isolation, the steps are:

1. **Isolated DC-DC**: add a CUI PDS1-S5-S5-S (~5 EUR) or similar 5 V → 5 V 1 W isolated converter between the Pi 5's 5 V and the analog domain. Isolation rating: 1.5 kV.
2. **Isolated SPI** between the ADS1299 and the RP2350A: ADuM3151 (3-channel digital isolator, 5 kV).
3. **Isolated USB-CDC**: ADuM3160 (USB digital isolator) or replace USB with isolated UART (UART + ADuM1201, then UART-to-USB bridge on the Pi side).
4. **Move bias-drive series resistor to 470 kΩ** for additional patient-current margin under IEC 60601-1's "single-fault condition".

These changes increase the PCB area by ~40 %, the BOM by ~30 EUR, and the firmware effort by ~1 week. They are documented as the v2 path, not part of v1.

## 5. Layout-induced power problems we explicitly designed away

Each of these is a real problem we have seen in similar designs and pre-emptively avoid here:

1. **High-frequency currents looping under the ADS1299**: avoided by routing all digital signals through the digital region and crossing the AGND/DGND split only via the star tie.
2. **LDO ground bounce**: avoided by placing the LDO output cap pad right next to the LDO GND pin, with a wide direct trace to AGND.
3. **USB-C VBUS ringing**: avoided by placing the 4.7 µF + 100 nF caps within 2 mm of the connector.
4. **VREF noise pickup**: **VREFP/VREFN** decoupling caps sit at the ADS1299 pins (<3 mm); no long VREF traces (internal ref — no external REF IC).
5. **Crystal load mismatch**: production BOM uses Pi **ABM8-272-T3** (**C_L = 10 pF**) with **15 pF** NPO caps on **XIN**/**XOUT** and **1 kΩ** series on **XIN** (`C_L ≈ C_ext/2 + C_stray`, ~3 pF stray per Pi guide). Optional **±10 ppm** upgrade **ABM8-12-B1U-T** uses **18 pF** caps and **no** series resistor — mark **DNP** unless populated instead of the default tier ([`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.3). Never mix tiers (e.g. **272-T3** + **18 pF**) — detunes XOSC and can break USB boot.
6. **Core-buck switching loop (RP2350A)**: the 1.1 V SMPS draws pulsed current through `VREG_LX → 3.3 µH → DVDD` and returns via `VREG_PGND`. We keep this loop **physically tiny**, on the **digital side only**, with CIN at `VREG_VIN` and COUT at `DVDD` close to the pins, so the switching node never couples into AGND or the EMG input network.

## 6. Acceptance criteria (power and grounding)

The HAT's power and grounding pass when:

- [ ] All rails are within ±5 % of nominal under nominal load (multimeter).
- [ ] **`TP_5VA1`** (or ADP7142 VOUT): AVDD **5.0 V ±5 %** (4.75–5.25 V, SBAS499 §7.3). **`TP_3V3D1`**: ADS1299 DVDD **3.3 V ±5 %**. RP2350A core **1.1 V ±5 %** (on-chip SMPS): infer from **`+3V3D`**, **`LED_HB1`**, USB/SWD — no `DVDD` test pad ([`h02`](h02_custom_hat_shield.md) §6.1).
- [ ] Ripple on the **+5VA** (AVDD) rail is <500 µVpp from 10 Hz to 100 MHz (scope, AC-coupled) — confirm the boost switching frequency is not visible after the LDO.
- [ ] Ripple on DVDD is <5 mVpp (looser tolerance, OK for the digital side).
- [ ] With inputs shorted at the **EMG patient connector(s)** (each signal pin to its shell on the bench), ADS1299 reads <1.5 µVrms in the 0.1–70 Hz band (integrated grounding test).
- [ ] No 50 Hz pickup visible on the EMG spectrum with DRL connected to a 100 kΩ resistor + AGND mock.
- [ ] Trigger pulse amplitude on **`TP_TRIG`** (or optional BNC) is 3.0–3.3 V, fall time <100 ns.

## 7. Operator-visible interlocks

The firmware enforces a few interlocks to keep the operator from misconfiguring the system:

- The recording UI refuses to start a session if the ADS1299 lead-off detection reports >50 % electrodes disconnected. (Override available with a confirmation prompt.)
- The recording UI refuses to start if the affine fit residual on the master ↔ Pi clock map exceeds 5 ms. (Indicates a serious clock issue.)
- The firmware refuses to start the trigger generator if the configured frame rate is outside [10, 200] Hz.

These keep ordinary configuration mistakes from quietly producing useless data.
