<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h02 — Custom HAT/Shield Design

> **Building the schematic in KiCad?** Pi header / HAT ID EEPROM: [`h02_hat_plus_schematic_build_guide.md`](h02_hat_plus_schematic_build_guide.md). **Flashing the HAT ID EEPROM:** [`h02_hat_plus_eeprom_flashing.md`](h02_hat_plus_eeprom_flashing.md). Camera trigger: [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md). See also [`_schematic_build_guide_conventions.md`](_schematic_build_guide_conventions.md).  
> **Starting PCB layout?** Schematic review: [`h10_schematic_logical_review.md`](h10_schematic_logical_review.md). Layout workflow + AGND/DGND: [`h10_kicad_pcb_layout_guide.md`](h10_kicad_pcb_layout_guide.md).

The HAT is the single piece of custom electronics in the project. Everything else is off-the-shelf. This document specifies the **design philosophy, mechanical fit, electrical stack-up, and KiCad layout rules** for the HAT. The per-subsystem schematic detail lives in the dedicated docs: EMG in [`h03_emg_frontend.md`](h03_emg_frontend.md), IMU in [`h04_imu_subsystem.md`](h04_imu_subsystem.md), MCU + firmware in [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md), RP2350A flashing in [`h08_rp2350_flashing.md`](h08_rp2350_flashing.md), sync in [`h06_synchronization.md`](h06_synchronization.md), power and safety in [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md).

## 1. Goals

The HAT exists to do four things and nothing else:

1. Acquire **6 EMG channels** through an ADS1299, with the analog environment quiet enough to see ≤1 µVrms in-band.
2. Acquire **2 IMU streams** from external BNO085 daughterboards over shielded cables.
3. Generate the **camera hardware trigger** (Track B default) with sub-µs jitter relative to the MCU clock. The firmware can also generate a **visible sync LED pulse** (Track A) on GPIO14; revision 0.0.1 has no driver or connector for it.
4. Funnel all of the above through one **USB-CDC link to the Pi 5**, carrying a 64-bit master timestamp on every record.

It deliberately does *not*:

- Power the cameras (they get their own 5 V from the Pi or external supply).
- Drive **IR LED rings** or **retroreflective illumination** (optical tracking uses **AprilTag/ArUco** under visible lab light; see [`h01_camera_subsystem.md`](h01_camera_subsystem.md) and [`../04_component_catalog.md`](../04_component_catalog.md#9-fiducials--scene-lighting)).
- Provide on-board storage (the Pi has the NVMe).
- Provide wireless (RF would contaminate the EMG analog rails).
- Provide isolation (we are tethered to mains via the Pi's PSU; medical-grade isolation is a follow-on project, see [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §6).

<a id="2-form-factor-and-mechanical-fit"></a>

## 2. Form factor and mechanical fit
- **Outline**: **100 × 56.5 mm**. The HAT+ core (**65 × 56.5 mm**, x = 0…65 mm) has four **3.2 mm** mounting holes at the HAT+ positions (58 × 49 mm pitch) and the 2×20 header at the HAT+ position. A **35 mm extension** (x = −35…0 mm) beyond the pin-1 end of the header carries the EMG front end and the patient pin headers (see §2.1). A **10 × 16 mm notch** in the edge opposite the header (x = 47.5…57.5 mm) is at the position of the Pi 5 connectors J3 / J4 (marked on the drawing layer).
- **Stacking**: **`J1`**, a Samtec **HLE-120-02-L-DV-BE-A** 2×20 bottom-entry socket, soldered on the top side; the pins of the stack below enter from the bottom (§11.5.1).
- **Panel human interface** (digital corner + trigger corner; detail in subsystem build guides):
  - **Buttons:** **RESET** (RUN), **BOOTSEL**, **USER1** (GPIO10), **USER2** (GPIO11).
  - **MCU LEDs:** heartbeat (**GPIO15**), capture blue (**GPIO24**, **`+5V_IN`**), fault / EMG lead-off (0603 @ 3.3 V); **USER** programmable RGB on **GPIO4** (**WS2812B-2020** on **`+5V_IN`**, **H-PAS-09**).
  - **IMU LEDs:** **MTA** / **CRA** / **DP** link indicators (GPIO26/27/28) in the LED column next to the MCU LEDs.
  - **Trigger LED:** **`LED_TRIG1`** (amber) on the `trigger_io` sheet.
  - **Power LEDs:** **5V**, **3V3**, **6.5 V power-good**, **5VA** on the `power` sheet (hardware-only).
- **Top-side connectors**:
  - **EMG patient connector** ([`h03_emg_frontend.md`](h03_emg_frontend.md) §10.2): revision 0.0.1 carries **option 0** (`H-CON-01CY`): 13× 2.54 mm pins as **`J_EMG_P1`** / **`J_EMG_N1`** (1×6 each) and **`J_EMG_DRL1`** (1×1). The shielded circular options **A/B/C** and the DIN option **F** have no footprints on this board. **CH7/CH8:** no patient connection.
  - 1× **7-pin** JST-SH for IMU MTA cable (6 signals + **SHLD** on pin 7 → **AGND** — [`h04_imu_subsystem.md`](h04_imu_subsystem.md) §3.3).
  - 1× **7-pin** JST-SH for IMU CRA cable and 1× for **DP** (same pinout; both behind the I2C mux **`U_MUX1`** — [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md)).
  - Trigger harness: **2× 2-pin** JST-SH to CSI cameras (**XTR+** / **XTR−** each; **`J_TRIG_CSI1`**, **`J_TRIG_CSI2`**). The board has no opto-isolated output and no BNC footprint (see [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md)).
  - No connector for a bench sync LED (`SYNC_LED`); the Track A sync LED is an option described in [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 4.
  - **Scope test pads** **`TP_TRIG1`** + **`TP_DGND_TRIG1`** on the buffered trigger, plus **`TP_CSI1`**, **`TP_CSI2`**, **`TP_TRIG_PIO1`** ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 2c).
  - 1× USB-C connector (**`J_USB1`**) on the short edge at the pin-40 end of the header (x = 65 mm) for the direct host link (independent of the Pi 5 USB ports).
- **NVMe coexistence**: the official Pi 5 spec allows only one HAT on top of the Pi. To coexist with the M.2 HAT+ for NVMe storage, the HAT goes on top of the M.2 HAT+ via the M.2 HAT+'s pass-through header. We confirmed mechanical clearance: the HAT sits ~14 mm above the Pi 5 PCB which is enough for the active cooler underneath.

```
        +--------------------------+
        |        Custom HAT        |   ← top
        | ADS1299 + RP2350A + IMU   |
        |   JSTs (BNO085 off-board)|
        +--------------------------+
        |     M.2 HAT+ + NVMe      |
        +--------------------------+
        |     Active Cooler        |
        +--------------------------+
        |       Raspberry Pi 5     |   ← bottom
        +--------------------------+
```

- The **top side** carries the ADS1299 with its input network and the electrode pin headers, all connectors (40-pin socket, USB-C, SWD, IMU and trigger JSTs), the I2C mux, the trigger buffers, the HAT ID EEPROM, all LEDs, buttons and the 14 test pads. The **bottom side** carries the RP2350A with flash, crystal and core-regulator inductor, the three regulators (boost, analog LDO, digital LDO) with the boost inductor, the FET of the capture LED, and their passives; the HAT+ pads `GND1` / `WP1` are also on the bottom.

**RP2350A USB (not GPIO):** **Data:** **USB_DM/DP** → **U_USB_ESD1** → **J_USB1** D±. **Power:** HAT runs from **Pi 5 V header** → **`+3V3D`** → **USB_OTP_VDD (53)**. **J_USB1 VBUS** (cable 5 V) goes to **U_USB_ESD1** for **ESD only** — it does **not** feed **USB_OTP_VDD** or the board. Pin tables: [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.5. **SWD:** **SWCLK/SWDIO** → **J_SWD1** ([`h08_rp2350_flashing.md`](h08_rp2350_flashing.md)).
- Revision 0.0.1 has **no footprint for a shield can** over the ADS1299 and the input filters; the catalog describes the option (**H-MECH-01**).

<a id="21-connector-count-vs-board-area-rough-fit"></a>

### 2.1 Connector count vs. board area (rough fit)
**Electrical vs. mechanical count**

| What | Count | Notes |
|------|-------|--------|
| EMG **channels** (muscles) | 6 | Differential → **12** signal conductors |
| **DRL** | 1 | One conductor |
| **EMG HAT connector** | **13** pins (Option **0**) / **1**, **4**, or **7** circular / **13** DIN | [`h03`](h03_emg_frontend.md) §10.2 — revision 0.0.1 has option **0** |
| **IMU JST-SH** | 3 **7-pin** (MTA, CRA, DP) | Small; digital edge; pin 7 = shield |
| **CH7/CH8** | 0 | Spare ADC inputs — no patient connectors |

**Area budget (order of magnitude; the estimate that led to the extended outline)**

For the HAT+ core alone (**65 × 56.5 mm**), assume a **~51 × 35 mm** keep-out under the 2×20 stacking header (center). Usable “rim” per side is only **~7–12 mm** deep before you hit the header zone.

| Block | Rough footprint | Comment |
|-------|-----------------|--------|
| ADS1299 + input filters | **~25 × 20 mm** | TQFP-64 + six channel networks within ~5 mm of pins ([`h03`](h03_emg_frontend.md) §4) |
| RP2350A + flash + SMPS | **~18 × 15 mm** | Digital corner |
| USB-C + SWD + panel UI | **~15 × 10 mm** | Short edge near Pi front |
| Block | Rough footprint | Comment |
|-------|-----------------|--------|
| **0 — 13× pin header** (`H-CON-01CY`) | **~33 mm × 1 row** (2.54 mm pitch) | **First iteration** — fits 56.5 mm edge easily |
| **A — 1× trunk** (`H-CON-01T`) | **~1 M12 hole** | Production |
| **B — 4× regional** (`H-CON-01R`) | **4× M12** (~48 mm if tight) | Split **2+2** across two edges |
| **C — 7× per-channel** (`H-CON-01C`) | **7× M12** | Edge extension or staggered rows |
| **F — 13× DIN** (`H-CON-01F`) | **~130–156 mm** one row | Does not fit 56.5 mm short edge |

**Conclusion:** **Option 0 (pin headers)** is on revision 0.0.1. The options **A/B/C** use shielded circular connectors; **Option F** (DIN) is for shielded touch-proof device-side jacks only. Each of them needs its own footprints.

The EMG front end with its connector does not fit on the HAT+ rectangle next to the RP2350A, power, IMU and trigger blocks. Revision 0.0.1 therefore has a **patient-edge extension** of **35 mm** beyond the HAT+ outline on the side **opposite** the Pi USB/Ethernet stack; it carries the ADS1299, the input network, the analog LDO and the pin headers.

Extending the board outside the HAT+ rectangle is **normal** for lab/custom HATs as long as the **mounting holes and the 40-pin header** stay at the HAT+ positions and the overhang clears the M.2 HAT+ below. No Pi “HAT+” rule forbids a larger board; only the **mounting pattern** is fixed.

**IMU / trigger JSTs** (3× **7-pin** IMU + 2× **2-pin** trigger) are small (**~5 mm** each) and sit on the rim of the HAT+ core — not the limiting factor.

<a id="3-pcb-stack-up"></a>

## 3. PCB stack-up
**4-layer board**, JLCPCB standard 1.6 mm FR4:

| Layer | Use |
|-------|-----|
| L1 (top, signal) | ADS1299 with input filters, all connectors, I2C mux, trigger buffers, LEDs, buttons, test pads; analog and digital signal traces; top silkscreen |
| L2 (inner, ground) | Ground plane, **split**: an `AGND` zone under the extension (x < 0) and a `DGND` zone under the HAT+ core (it also covers the strip of the power LEDs at the top of the extension). The two are joined at exactly one point by the net tie **`NT_GND_TIE1`** on the top layer at `U_ADS1` pads 31/32 (star ground) |
| L3 (inner, power) | Power zones: **`+5VA`** (analog, under the extension) and **`+3V3D`** (digital, under the HAT+ core), plus a small zone of the 1.1 V core supply at the MCU. `+5V_IN` is routed as tracks, not as a plane |
| L4 (bottom, signal) | RP2350A, flash, crystal, regulators and their passives; digital signal traces (SPI, QSPI, GPIO); bottom silkscreen |

Trace/space rules:

- Trace width: signal tracks are **0.2 mm** (1153 of the 1699 track segments). **105 segments are 0.15 mm**: the USB D+/D− tracks, `ADS_RESET` / `ADS_PWDN`, and short stubs at the ADS1299 inputs and at a few supply pads. Power and ground tracks and the tracks of the HAT+ template (ID EEPROM) are 0.25–0.6 mm. The design-rule minimum in the project file is **0.15 mm**.
- Minimum drill: **0.3 mm** (vias 0.55 / 0.3 mm for signals; 0.7 / 0.4 mm for power, ground and the HAT+ template nets; 0.8 / 0.4 mm at the HAT+ template and on `+5V_IN`).
- USB D+/D−: routed as a pair with **0.15 mm** tracks (the `USB` net class of the project file sets 0.2 mm / 0.4 mm gap). Controlled impedance is **not** required for USB Full Speed (12 Mbit/s), which is what USB-CDC runs at.
- Crystal traces (12 MHz on RP2350A): on the bottom layer in the digital part; `XIN` about 12 mm of track (MCU → crystal → load capacitor), `XOUT` about 8 mm including the series resistor **R53**.

<a id="4-floor-plan"></a>

## 4. Floor plan
```
        +-------------------------------------------+
        |  [EMG pin headers: 2× 1×6 + 1× 1×1]        |
        |  ---------------------------------------- |
        |          [ESD + RFI filter array]         |
        |          [common-mode chokes]             |
        |          [ADS1299]                        |   ← analog part
        |          [ADS1299 VREF decap, +5VA LDO]  |
        |  ---------------------------------------- |
        |  star ground tie under ADS1299            |
        |  ---------------------------------------- |
        |   [RP2350A + flash + crystal]              |
        |   [LDO_D, +5VA boost (switcher), USB-C]   |   ← digital part
        |   [IMU JST-SH 7-pin × 3 + I2C mux (BNO085 off-board)] |
        |   [Trigger fan-out + scope TP]            |
        |   [40-pin Pi socket J1]                   |
        +-------------------------------------------+
```

The board is split into an **analog part** (the 35 mm extension with the ADS1299, about one third of the length) and a **digital part** (the HAT+ core). The L2 ground plane is split along the same line (x ≈ 0) and joined at a **single star tie at the ADS1299 (net tie `NT_GND_TIE1` at pads 31/32)** — see [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2.2 for the canonical tie location. The split forces all return currents through a known choke point and keeps digital switching noise out of the analog ground. The only switching regulator (the **+5VA boost** that feeds the analog LDO) sits in the **digital part** so its switching noise stays on DGND; only the clean LDO output crosses into the analog region. This floor plan implements TI's recommended placement (SBAS499 §12.1 / Figure 79); the full point-by-point compliance is in [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §2.4.

## 5. Signal isolation strategy

We do **not** use opto-isolators or digital isolators on the SPI bus between ADS1299 and the RP2350A. Reasons:

- The HAT sits on the Pi 5 and is powered by the Pi 5; there is no separate isolated domain to defend.
- The ADS1299 datasheet's reference design (Figure 75) explicitly uses non-isolated SPI to a host MCU with the same star-ground scheme we adopt.
- True medical-grade isolation (≥4 kV reinforced) requires an isolated DC-DC, isolated SPI bridge (e.g. ADuM3151), and isolated USB-CDC (e.g. ADuM3160). It is documented as future work in [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) but not part of v1.

Instead, we rely on:

- Star ground tie at the ADS1299.
- Common-mode chokes on every EMG input.
- Driven-right-leg (DRL) electrode driven by the ADS1299 internal bias buffer.
- Isolated USB hub between the Pi 5 and any mains-tied peripheral (this is enforced operationally, not on the HAT).

## 6. Power

The full power story is in [`h07_power_grounding_safety.md`](h07_power_grounding_safety.md). The HAT-level summary:

- Input: 5 V from the Pi 5's GPIO header (pin 2 / pin 4).
- 5 V → **+5VA (5.0 V) analog** via a small boost (~6.5 V) + **ADP7142AUJZ-5.0** fixed LDO — feeds **ADS1299 AVDD** only. **VREF = 4.5 V** from the **internal** ADS1299 reference ([`h03`](h03_emg_frontend.md) §9). The ADS1299 **requires** AVDD−AVSS = 4.75–5.25 V (SBAS499 §7.3); 3.3 V analog is out of spec, so a 5 V rail is mandatory (a linear reg can't drop 5 V→5 V, hence the boost-then-LDO).
- 5 V → **3.3 V digital** via **TPS73633DBV** (`EN` tied to `VIN`; **NR** + 10 nF to GND) — feeds the RP2350A with its flash, the trigger buffers, the I2C mux, the IMU connectors, and ADS1299 DVDD.
- Estimated total draw: **~200 mA** at 5 V typical (~1.0 W); well within the Pi 5 GPIO header's ~1 A budget. The HAT carries **no** sync-LED driver and **no** IR ring load. Optional IR rings use **external 5 V** ([`h07_power_grounding_safety.md`](h07_power_grounding_safety.md) §1.2–§1.4).

<a id="61-debug-access-leds-vs-test-pads"></a>

### 6.1 Debug access: LEDs vs test pads
Status LEDs give **qualitative** bring-up feedback (rail present, firmware alive, trigger toggling). **Test pads** give **quantitative** bench access (DMM voltage, scope timing, logic-analyzer buses) — especially **before firmware runs**. Capture spec: [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md) Step 5, [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md), [`h10_kicad_pcb_layout_guide.md`](h10_kicad_pcb_layout_guide.md) §6.

**DMM** = digital multimeter (DC voltage, resistance, continuity). **Scope** = oscilloscope (waveforms, edge timing).

#### LED panel

| Ref / group | Sheet | Indicates |
|-------------|-------|-----------|
| `LED_5VIN1`, `LED_3V3D_OK1`, `LED_6V5D_PG1`, `LED_5VA_OK1` | `power` | Pi 5 V in; `+3V3D` OK; boost PG; `+5VA` present |
| `LED_HB1`, `LED_CAP1`, `LED_FAULT1`, `LED_EMG_LO1`, `LED_USER1` | `mcu` | Heartbeat; streaming; fault; lead-off; programmable RGB |
| `LED_TRIG1` | `trigger_io` | GPIO13 / `TRIG_PIO` activity |
| `LED_MTA1`, `LED_CRA1`, `LED_DP1` | `imu_connectors` | IMU bus activity (firmware-driven) |

LEDs cannot confirm exact rail voltage under load, 4 kHz `ADS_DRDY` timing, or camera-side trigger level (~1.8 V).

#### Test pad map

Footprint for all pads: `TestPoint:TestPoint_Pad_D1.0mm` (1.0 mm round SMD), top copper, reference designator on the silkscreen. Catalog: **H-CON-05** ([`04_component_catalog.md`](../04_component_catalog.md)).

**Test pads in the design:** **`TP_TRIG1`**, **`TP_DGND_TRIG1`**, **`TP_CSI1`**, **`TP_CSI2`**, **`TP_TRIG_PIO1`** on `trigger_io`; **`TP_5VIN1`**, **`TP_3V3D1`**, **`TP_5VA1`**, **`TP_DGND_PWR1`** on `power`; **`TP_ADS_DRDY1`**, **`TP_AGND1`**, **`TP_VREFP1`**, **`TP_BIASOUT1`**, **`TP_VCAP1`** on `emg`; the `mcu` sheet has **no test pads**; HAT+ template `GND1` / `WP1` on `hat_plus` (bottom side; low bring-up value).

| Ref | Net | Sheet | Tap | Alternative probe point |
|-----|-----|-------|-----|-------------------------|
| `TP_TRIG1` | buffered trigger (no label; `TRIG_BNC` in the build guide) | `trigger_io` | After R55 (50 Ω) | — |
| `TP_DGND_TRIG1` | `DGND` | `trigger_io` | Local scope GND | Pi GND / `GND1` |
| `TP_CSI1` | camera 1 XTR+ (no label; `TRIG_CSI1` in the build guide) | `trigger_io` | After divider, before `J_TRIG_CSI1` | `J_TRIG_CSI1` pin 1 (harness off) |
| `TP_CSI2` | camera 2 XTR+ (no label; `TRIG_CSI2` in the build guide) | `trigger_io` | Same for camera 2 | `J_TRIG_CSI2` pin 1 |
| `TP_TRIG_PIO1` | `TRIG_PIO` | `trigger_io` | Before 74LVC buffers | `LED_TRIG1` + `TP_TRIG1` compare |
| `TP_5VIN1` | `+5V_IN` | `power` | After input bulk cap | Pi header pins 2/4 vs GND |
| `TP_3V3D1` | `+3V3D` | `power` | TPS73633 OUT | `J_IMU_MTA1` pin 1 vs pin 2 |
| `TP_5VA1` | `+5VA` | `power` | ADP7142 VOUT | Rail side of `R31` (series resistor of `LED_5VA_OK1`; less ideal) |
| `TP_DGND_PWR1` | `DGND` | `power` | Power cluster | `TP_DGND_TRIG1` |
| `TP_ADS_DRDY1` | `ADS_DRDY` | `emg` | ADS1299 DRDY pin | USB-CDC `selftest` / `ads_drdy_watch` |
| `TP_VREFP1` | ADS1299 `VREFP` pin (no label) | `emg` | ADS1299 VREFP | Trust counts if `selftest` passes |
| `TP_AGND1` | `AGND` | `emg` | Near the star tie at the ADS1299 | — |
| `TP_BIASOUT1` | `IN8N_BIASOUT` (IN8N / DRL) | `emg` | After `R6` on DRL path (not raw `BIASOUT`) | CH8 readback / probe **`R8`/`C78`** or pin 63 for bias amp |
| `TP_VCAP1` | ADS1299 `VCAP1` pin (no label) | `emg` | ADS1299 VCAP1 | Power-up script only |

The IMU nets have no test pads: probe `IMU_MTA_SDA` / `IMU_MTA_SCL` (I2C0) at **`J_IMU_MTA1` pins 3 / 4** and `IMU_MTA_INT` (GPIO16) at **pin 5**, harness off; CRA and DP at the same pins of **`J_IMU_CRA1`** / **`J_IMU_DP1`** (connector side of the mux).

**No test pads on:** USB D+/D− (use `J_USB1`), patient `INxP/N` (pin headers suffice), `VBUS` (ESD-only), **RP2350 `DVDD` / VREG loop** (probe stub couples into the on-chip SMPS — keep L2/C45/C66 cluster uninterrupted). **SWD** uses `J_SWD1`; **SPI** debug via USB-CDC `ads_status` / `ads_regs` + `TP_ADS_DRDY1`.

**Count:** **14** functional test pads (excl. the HAT+ template pads `GND1` / `WP1`).

#### Physical placement

```text
      [analog extension, x < 0]            [HAT+ core, x > 0]
    TP_5VIN1 TP_5VA1 TP_3V3D1 TP_DGND_PWR1   (top edge, x ≈ −12…−2)
    TP_BIASOUT1   (ADS1299, left)
    TP_ADS_DRDY1  (ADS1299, right, x ≈ 4)
    TP_VREFP1 TP_VCAP1 TP_AGND1  (below the ADS1299)
                                           TP_DGND_TRIG1  (x ≈ 51, y ≈ 22)
                                           TP_CSI1 TP_CSI2 TP_TRIG_PIO1  (y ≈ 26…28)
                                           TP_TRIG1  (x ≈ 51, y ≈ 34)
```

Scope ground: **`TP_DGND_TRIG1`** for the trigger pads (distance to `TP_TRIG_PIO1` 5.3 mm, `TP_CSI1` 5.7 mm, `TP_CSI2` 7.8 mm, `TP_TRIG1` 11.8 mm), **`TP_DGND_PWR1`** for the power pads (9–12 mm). Use **`TP_AGND1`** (not `DGND`) as scope return for `TP_VREFP1` and analog ripple checks.

#### Signal integrity (14 pads)

| Net type | Risk | Layout rule |
|----------|------|-------------|
| Power rails | Negligible | TP directly on rail |
| Slow digital (DRDY 4 kHz, I2C, INT, trigger) | Low | Stub **<3–5 mm** |
| Analog ref (`VREFP`, `AGND`) | Low if away from inputs | Analog edge only; **no TPs on `INxP/N`** |
| USB D+/D− | Stubs hurt FS USB | **No TPs** |

Pads are unpopulated copper (no extra BOM parts). Main cost is board-edge area and silkscreen — not measurable signal degradation when stubs stay short.

## 7. Trigger / sync I/O sheet (`trigger_io.kicad_sch`) — PCB scope for AprilTag tracking

Optical pose uses **printed fiducials**, not IR retroreflectors. The HAT still needs tight **time** infrastructure; it does **not** need high-current IR drivers.

### 7.1 On revision 0.0.1

| Block | Schematic | PCB notes |
|-------|-----------|-----------|
| Camera trigger buffer | **2×** `74LVC2G241` + **1.5 kΩ / 1.8 kΩ** divider ×2 + **50 Ω** (scope tap) | `U_CSI_1` → dividers **R57/R58**, **R59/R60** → **`J_TRIG_CSI1/2`** pin 1 (~**1.8 V**) → **XTR+**; pin 2 (`DGND`) → **XTR−**; `U_TP_1` → **R55** (50 Ω) |
| Scope test pads | **`TP_TRIG1`** (buffered trigger after **R55**) + **`TP_DGND_TRIG1`** (`DGND`); **`TP_CSI1`**, **`TP_CSI2`**, **`TP_TRIG_PIO1`** | Bench scope during bring-up and validation — see build guide Step 2c |
| PIO trigger source | RP2350A GPIO (`trigger_gen` PIO) → buffers | Single `trigger_gen` PIO fans out; see [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1 / §4.1 |
| Trigger panel LED | **`LED_TRIG1`** + **R52** (470 Ω) on **`TRIG_PIO`** | Amber operator indicator ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 4b) |

The fan-out on the board is **2× CSI + scope test pads** from **`TRIG_PIO`**. The branches below are options of the build guide ([`h06_synchronization.md`](h06_synchronization.md) §3.2); **revision 0.0.1 has no footprints for them**.

### 7.1b Options not on revision 0.0.1

| Block | Schematic | Use |
|-------|-----------|---------------|
| BNC / SMB scope jack | Coax tap parallel to **`TP_TRIG1`** | Multi-board lab or repeated scope hookup |
| USB3 camera trigger | optoisolator (anode via **220 Ω**) | Optional **Arducam UC-599** — `TRIG_USB`; isolated so USB camera ground does not couple into the HAT |
| Visible sync LED driver | GPIO → N-FET (e.g. AO3400) → **2-pin JST** | **Track A webcams** (no HW trigger) or optional photonic cross-check — **not** needed when both GS cameras use **XTRIG** |
| Sync LED series resistor + panel **D_FLASH** | Sized for 5 V LED anode (~20 mA) | With sync LED block only; return on **DGND** |

With a sync LED, the remote LED is mounted **visible to all cameras** (near the ChArUco board or top of head frame). Firmware: `flash on` / `flash off` ([`h05`](h05_microcontroller_firmware.md)); the signal is available on GPIO14 (`SYNC_LED_PIO`), which ends at the port of the `mcu` sheet.

### 7.2 Not on the HAT

| Block | Why omitted |
|-------|-------------|
| IR LED ring constant-current driver (e.g. Meanwell 350 mA module) | AprilTag detection uses visible light; rings add cost, heat, and EMI risk near EMG analog |
| 5 V high-current MOSFET “RING_EN” from Pi GPIO | Per-camera IR rings are not part of the system |
| 850 nm IR-pass filter screw mounts | Camera/lens mechanical, not HAT |

The board has no footprints for IR ring drivers (`RING_PWR`, `RING_PWM`).

### 7.3 Schematic sheet checklist (`trigger_io`)

- [ ] The two CSI outputs never share a series resistor (**R57** for CSI1, **R59** for CSI2); each output has its own divider.
- [ ] **`TP_TRIG1`** on the buffered trigger (after **R55**); **`TP_DGND_TRIG1`** on **`DGND`** (11.8 mm from `TP_TRIG1` on the board); reference designators on the silkscreen.
- [ ] No USB opto path, BNC jack or sync-LED block on the sheet; **`U_TP_1` pin 6** (1Y) is not connected.
- [ ] Trigger returns (**`J_TRIG_CSI1/2`** pin 2, **`LED_TRIG1`** cathode) on **DGND**; no return path through **AGND**.
- [ ] External JST pins: the **1.5 kΩ** series resistors (**R57**, **R59**) are the only protection of the CSI trigger outputs; the board has no ESD array on them.
- [ ] Silkscreen: reference designators of **`J_TRIG_CSI1`**, **`J_TRIG_CSI2`**, **`LED_TRIG1`** and the five trigger test pads.

### 7.4 Layout (trigger region, top layer)

- The trigger buffers (**`U_CSI_1`**, **`U_TP_1`**), dividers and JSTs are on the **top side** at the short edge of the HAT+ core (x ≈ 49…62 mm, y ≈ 20…36 mm), between the USB-C connector and the HAT ID EEPROM.
- **`TP_TRIG1`** / **`TP_DGND_TRIG1`**: top layer, **11.8 mm** apart; `TP_TRIG_PIO1`, `TP_CSI1` and `TP_CSI2` lie between them, 5.3 mm, 5.7 mm and 7.8 mm from the ground pad ([`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 2c).
- No copper pour under the ADS1299 from the digital trigger traces.
- The trigger cables leave at the short edge opposite the electrode pin headers (about 90 mm away), which reduces patient-cable crosstalk.

## 8. KiCad project layout

The KiCad project lives under `hardware/jtsmm_hat/` in the source repository (not in `docs/`). Its tree:

```
hardware/jtsmm_hat/
├── jtsmm_hat.kicad_pro
├── jtsmm_hat.kicad_sch       # page 1 — top level (buses only)
├── jtsmm_hat.kicad_pcb
├── fp-lib-table
├── libs/                     # project footprints (e.g. Samtec HAT+ header)
├── sheets/
│   ├── hat_plus.kicad_sch    # page 2 — Pi 40-pin + HAT ID EEPROM (template)
│   ├── power.kicad_sch       # page 3 — 5 V → +5VA (TPS61252+ADP7142AUJZ-5.0) / +3V3D (h07)
│   ├── emg.kicad_sch         # page 4 — ADS1299 + patient connectors (13 header pins, 2× 1×6 + 1×1)
│   ├── mcu.kicad_sch         # page 5 — RP2350A + USB-C (h05)
│   ├── trigger_io.kicad_sch  # page 6 — camera TRIG fan-out (§7)
│   └── imu_connectors.kicad_sch  # page 7 — BNO085 JST-SH 7-pin ×3 + I2C mux (h04)
├── eeprom/                   # HAT ID EEPROM settings and images (h02_hat_plus_eeprom_flashing.md)
├── plot/                     # board renders (top, bottom)
└── production/               # Gerber and drill archive, BOM, placement, netlist
```

**Top sheet (`jtsmm_hat.kicad_sch`)** places the six hierarchical blocks and wires shared nets (5 V from Pi, `+5VA`/`+3V3D`, SPI to ADS1299, trigger GPIO). Do not put ICs on the root page — only in the subsystem sheets.

**IMU connectors** live on [`sheets/imu_connectors.kicad_sch`](../../hardware/jtsmm_hat/sheets/imu_connectors.kicad_sch) (capture: [`h04_imu_schematic_build_guide.md`](h04_imu_schematic_build_guide.md)).

We start the schematic from:

- TI ADS1299 datasheet (SBAS499) Figures 73 and 75 (typical EMG application reference).
- The Couby OpenBCI KiCad project at <https://github.com/Couby/OpenBCI_hardware_KiCAD> as a sanity check on the EMG input network.
- Adafruit's open KiCad designs for BNO085 and RP2350A modules.

<a id="9-design-rule-check-drc-policy"></a>

## 9. Design rule check (DRC) policy
Before fab submission the following must be clean:

- No DRC errors (electrical or footprint).
- No silkscreen overlap on pads (manufacturer-specific minor warnings are OK).
- All signal traces have a continuous return path on L2 (visualize with `Render → 3D` and a custom plugin or by inspection).
- All ground pours connected via thermal-relief patterns to ground pads (the ADS1299 TQFP-64 has no exposed pad).
- USB-C connector clearance to the PCB edge ≥0.3 mm.
- ENIG finish for the **patient connector** pads (HASL is fine elsewhere but ENIG is gentler for repeated mating).

<a id="10-manufacturing"></a>

## 10. Manufacturing
- **PCB fab**: JLCPCB 4-layer, ENIG, 1.6 mm FR4 (no castellated areas).
- **Assembly**: Hand-soldered: ADS1299, **EMG patient connector(s)** — **H-CON-01CY** (pin headers) ([`../04_component_catalog.md`](../04_component_catalog.md) §5–§6).
- **Stencil**: 0.1 mm laser stencil for hand-paste assembly of the hand-soldered parts.
- **Panelization**: a panel of 5 boards (~50 mm spacing) is cheaper per-board than singles and gives us spare boards for other labs.
- **Quality control**: after assembly, the first board goes through the bring-up checklist before the rest are powered up.

<a id="11-design-review-checklist"></a>

## 11. Design-review checklist
The checklist below must be ticked off before submitting to fab. The intent is to catch the issues that most often kill a project at the PCB respin gate.

### 11.1 Power and grounding

- [ ] Regulator capacitors at the pins: **TPS61252** **C29** (10 µF) at VIN, **C30/C31** (2× 22 µF) at VOUT; **ADP7142** **C33** (100 nF) + **C35** (2.2 µF) + **C34** (10 µF) at VIN, **C38** (100 nF) + **C36** (2.2 µF) + **C37** (10 µF) at VOUT; **TPS73633** **C25** (100 nF) + **C24** (10 µF) at IN, **C27** (100 nF) + **C26** (10 µF) at OUT, **C28** (10 nF) at NR. On the board the nearest capacitor of each regulator pin is 1.4–1.9 mm away (pad centre to pad centre); **C30/C31** are about 7 mm from the boost VOUT pin.
- [ ] **+5VA rail present:** boost (~6.5 V) + **ADP7142AUJZ-5.0** produce **5.0 V** for ADS1299 AVDD; AVDD−AVSS measures 4.75–5.25 V (SBAS499 §7.3). DVDD = 3.3 V.
- [ ] **+5VA boost (switcher) placed in the digital half**, on DGND; only its LDO output crosses to the analog region.
- [ ] 5 V input: `+5V_IN` comes directly from header pins 2/4; the board has **no** reverse-polarity protection — apply bench power with the correct polarity.
- [ ] Star ground tie is **`NT_GND_TIE1`** (`Device:NetTie_2`) on the `emg` sheet; AGND/DGND joined **at `U_ADS1` pads 32/31** (per SBAS499 §12.1).
- [ ] Analog and digital ground planes are clearly separated and not stitched anywhere except at the star.
- [ ] No high-current digital trace passes underneath the analog ground region; L2 reference plane is continuous under every signal trace (no plane cuts forcing return-current detours).
- [ ] Void areas on signal layers filled with ground fill, via-stitched to the adjacent plane (SBAS499 §12.1).
- [ ] **No via between any ADS1299 bypass cap and its pin** — bypass caps on the same layer, at the pin (SBAS499 §12.1).

### 11.2 ADS1299 input network

- [ ] Each EMG input has: 100 Ω series resistor per leg + **4.7 nF C0G differential `CFilt` across INxP–INxN** (per SBAS499 §10.2.2, *not* per-leg-to-ground) + ESD diode array (**TPD4E1B06** default — 0.7 pF / 0.5 nA / bidirectional, GND→AGND; `TPD4E05U06DQA` / `TPD4E004` alternatives) + common-mode choke. The board has no common-mode capacitors to AGND.
- [ ] Series resistors = 3× **CAY16-1000F4LF** (`RN1`–`RN3`, 4×100 Ω ±1 %, [`h03`](h03_emg_frontend.md) §4.1); the differential `CFilt` (`C3`–`C8`) is **C0G** 4.7 nF, **never X7R**.
- [ ] DRL patient pin (`J_EMG_DRL1`): **100 Ω → TPD4E1B06 (1 ch) → DLW21 (pins 1–2 in series; 3–4 NC) → 220 kΩ → IN8N**, plus **4.7 nF C0G** on the patient path to AGND; firmware **`MUX8 = 111`** ([`h03`](h03_emg_frontend.md) §5).
- [ ] **EMG patient connector(s)** body/cable exit does not interfere with bench cabling ([`h03`](h03_emg_frontend.md) §10.2).
- [ ] **BIASIN** tied directly to **BIASOUT**; **BIASINV** to **BIASOUT** via **`R8` (1 MΩ) ∥ `C78` (1.5 nF)** — no BIASOUT shunt cap (see h03 §5.1).
- [ ] **BIASREF → AGND** (pin unused); firmware **`BIASREF_INT = 1`** (CONFIG3 = `0xEC`, internal mid-supply).
- [ ] **`TP_BIASOUT1`** on **IN8N / DRL** net (not raw **BIASOUT**).
- [ ] **Internal 4.5 V reference:** `CONFIG3` enables on-chip ref buffer; **VREFN → AGND**; **≥10 µF + 0.1 µF** on **VREFP–VREFN** at the ADS1299 ([`h03`](h03_emg_frontend.md) §9). No external reference IC on the board (full **VREF = 4.5 V** range).
- [ ] **ADS1299 decoupling per SBAS499 Pin Functions:** 1 µF at each AVDD/DVDD pin (+ 10 µF + 0.1 µF rail bulk); **VCAP1 = 100 µF** (tantalum, **C9**); VCAP2 = 1 µF; VCAP3 = 1 µF ∥ 0.1 µF; VCAP4 = 1 µF; all to AVSS, placed as close to the package as possible.
- [ ] **AVDD1/AVSS1 star-connected** to the AVDD/AVSS pins (SBAS499 §11); charge-pump AVDD (pin 59) bypassed to AVSS pin 58.
- [ ] Two-state pins strapped through ≥10 kΩ (`CLKSEL` high, `GPIO1–4` to DGND if unused); `RESV1` tied directly to DGND.
- [ ] **AVDD = AVSS + 5 V = AGND + 5 V** (unipolar; AVDD−AVSS must be 4.75–5.25 V per SBAS499 §7.3) — fed from the **+5VA** rail (boost + ADP7142AUJZ-5.0).
- [ ] DVDD = 3.3 V from digital LDO.
- [ ] Measured AVDD−AVSS within 4.75–5.25 V and AVDD−DVDD within −2.1…+3.6 V (SBAS499 §7.3) before enabling the ADS1299.

### 11.3 MCU

- [ ] BOOTSEL button accessible.
- [ ] **USER1** / **USER2** buttons and **LED_USER** (GPIO4 RGB) per [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) Step 6; **`LED_TRIG1`** panel LED per [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 4b.
- [ ] Power panel LEDs per [`h07_power_schematic_build_guide.md`](h07_power_schematic_build_guide.md) Step 8: **5V** (input), **3V3** (`+3V3D`), **6.5V** (boost **PG**), **5VA** (`+5VA` after **ADP7142**).
- [ ] **SWD breakout (3 signals):** **GND, SWCLK, SWDIO** on **`J_SWD1`**, a **JST-SH 3-pin** (matches the Raspberry Pi Debug Probe kit cable) — see [`h08_rp2350_flashing.md`](h08_rp2350_flashing.md). **+3V3D** from the HAT LDO when **5 V** is on; the debug cable does **not** carry power.
- [ ] **RESET** button (**`SW_RESET1`**) on **RUN** (not on the 3-wire probe cable).
- [ ] No 2×5 **1.27 mm** ARM debug header on the board — **not** required for the Debug Probe ([`h08`](h08_rp2350_flashing.md) §4.3).
- [ ] 12 MHz crystal per Pi layout: **ABM8-272-T3** (**Y1**), **15 pF** load caps (**C41/C42**), **1 kΩ** in series with **XOUT** (**R53**); **18 pF** upgrade tier not fitted — `[h05](h05_mcu_schematic_build_guide.md)` §1.3; `XIN` ≈ 12 mm, `XOUT` ≈ 8 mm of track on the bottom layer.
- [ ] RP2350A decoupling per `[h05](h05_mcu_schematic_build_guide.md)` §1.1.2, all on **B.Cu** around `U_MCU1`: **9× 0402** 100 nF on **+3V3D** (**C47, C48, C50, C51, C67–C71**); **2× 0402** 100 nF on **DVDD** (**C64, C65**) with **C46** and **C66** (4.7 µF); **C45** (4.7 µF) at **VREG_VIN**; **C40** (4.7 µF) on **VREG_AVDD** behind **R39** (33 Ω); **4** vias in the exposed pad. **C49** (100 nF) is on USB **VBUS** at `U_USB_ESD1`; **C52** (100 nF) on **+5V_IN** at `LED_USER1`.
- [ ] USB D+ and D− differential pair routed together, lengths matched, away from the analog region.
- [ ] **RUN** uses the internal pull-up (no external resistor); the BOOTSEL / flash **~CS** net has **R51** (10 kΩ) to +3V3D; user buttons debounced in firmware (no caps).

### 11.4 IMU and trigger

- [ ] **7-pin** JST-SH connectors keyed so the IMU cable cannot be reversed without effort (mechanical mistake-proofing); pin **7** `IMU_SHLD` → **AGND** on PCB.
- [ ] Trigger fan-out buffers (**2× 74LVC2G241**, **`U_CSI_1`**, **`U_TP_1`**) populated; **R57/R59** @ **1.5 kΩ** (series) + **R58/R60** @ **1.8 kΩ** (to DGND); **R55** **50 Ω** on the scope path.
- [ ] Scope test pads **`TP_TRIG1`** / **`TP_DGND_TRIG1`** / **`TP_CSI1`** / **`TP_CSI2`** / **`TP_TRIG_PIO1`** present with their reference designators on the silkscreen.
- [ ] Verify trigger polarity: scope **`TP_TRIG1`** vs pin 1 of **`J_TRIG_CSI1`** / **`J_TRIG_CSI2`** (or **`TP_CSI1`** / **`TP_CSI2`**) with `trig_start` or `jtsmm trigger` (rate and pulse width of the applied configuration, §12 step 13).
- [ ] No sync LED block and no IR ring driver on the board.
- [ ] I2C mux **`U_MUX1`** populated (the CRA connector is reached through mux channel 0).

### 11.5 Mechanical

- [ ] Outline 100 × 56.5 mm: HAT+ core 65 × 56.5 mm plus the 35 mm extension; header and holes at the HAT+ positions.
- [ ] Four 3.2 mm mounting holes (non-plated) at the HAT+ positions (58 × 49 mm pitch); copper keep-out around them.
- [ ] Stacking header chosen with the correct height to clear the M.2 HAT+ underneath.
- [ ] **GPIO mate verified:** `i2cdetect -y 0` shows stable **`50`** at **0x50** when fully stacked (not only with flying leads) — see [stacking pins](#1151-stacking-pins-below-j1) below.

#### 11.5.1 Stacking pins (below **J1**)

**J1** is a **Samtec HLE-120-02-L-DV-BE-A** bottom-entry **socket** (LCSC **C6207203**). Male pins must come from **below** the HAT — do **not** solder a pin header on **J1**.

| Stack | Pins below **J1** |
|-------|-------------------|
| Pi 5 → **JTS-MM only** | Pi factory GPIO pins (direct mate) |
| Pi 5 → **M.2 HAT+** → JTS-MM | **13.3 mm** stacking extension on the **Pi** (replaces the M.2 kit’s flush ~10 mm header) — e.g. BerryBase RPI-CGST-133 class |

**Confirmed bench issue (rev 0.0.1):** Generic stacking headers with **0.64 × 0.40 mm** rectangular pins (BerryBase spec) gave **intermittent** contact in the **HLE** beam socket — idle DMM on pins **1 / 27 / 28** looked fine, but `i2cdetect` was empty and `dmesg` showed **`SDA stuck at low`**. Flying leads to the same nets worked. **Fix:** use **0.025″ (0.64 mm) square** posts (Samtec **TSW** / **MTSW**, or a stacking header rated for HAT+ / square pins), or short jumpers on critical nets until proper pins arrive.

Samtec HLE mates **square** posts per datasheet; thin rectangular pins are a common mismatch on third-party “13.3 mm” headers.

### 11.6 Documentation

- [ ] Schematic title block has the board name, revision, date, designer.
- [ ] Reference designators labeled on silkscreen, visible after assembly.
- [ ] Pin 1 markers on all polarized components.
- [ ] Test points labeled.
- [ ] BOM exported to CSV with manufacturer part numbers.
- [ ] HAT ID EEPROM **U1** programmed — [`h02_hat_plus_eeprom_flashing.md`](h02_hat_plus_eeprom_flashing.md).
- [ ] iBOM (interactive BOM) exported for assembly.

## 12. Bring-up bench script

The bring-up sequence for a newly assembled board. Commands are those of the firmware in [`firmware/jtsmm/`](../../firmware/jtsmm/README.md) (USB-CDC text commands) and of the host package [`software/jtsmm/`](../../software/jtsmm/README.md):

1. Visual inspection (loupe). Look for solder bridges on ADS1299, USB-C, RP2350A.
2. Short-to-ground check: DMM **`TP_5VIN1`** → **`TP_DGND_PWR1`** (or Pi header 5 V vs GND) — typically ≥1 kΩ, not 0 Ω.
3. Apply **5 V** to the 5 V pins of the 40-pin header (pins 2/4) from a bench supply (do not insert into the Pi yet; the USB-C connector does not power the board, §2). DMM **`TP_3V3D1`** ≈ 3.3 V (or **`J_IMU_MTA1` pin 1** vs pin 2). Watch power LED chain: **5V → 3V3 → 6.5V-PG → 5VA**; DMM **`TP_5VA1`** ≈ 5.0 V.
4. Connect the Debug Probe **3-wire** cable (**GND, SWCLK, SWDIO** only). Flash the firmware via SWD ([`h08_rp2350_flashing.md`](h08_rp2350_flashing.md)). Watch **`LED_HB1`** blink (firmware heartbeat).
5. Connect a USB cable from **`J_USB1`** to the host PC; the RP2350A should enumerate as a USB-CDC device within 2 s.
6. Open a terminal at the enumerated `/dev/ttyACM0`. Type `info\n`; the firmware should respond with its version string (`jtsmm <version>-prod`) and the last reset reason.
7. Send `selftest`, then `ads_regs`. The firmware should report the ADS1299 register reads matching the expected values (ID = 0x3E, etc.). If this fails: DMM **`TP_VREFP1`** ≈ 4.5 V vs **`TP_AGND1`**; check `DRDY` at **`TP_ADS_DRDY1`** with a scope, or count its edges with `ads_drdy_watch`.
8. Enable the ADS1299 internal test signal in the configuration (`mcu.emg.test_signal.enabled: true`, [`s08_runtime_configuration.md`](../software/s08_runtime_configuration.md)) and record a short take on the host: `jtsmm record --duration 10 --port /dev/ttyACM0`. The firmware sets `CONFIG2 = 0xD0` (`INT_CAL = 1`, `CAL_AMP = 0`, `CAL_FREQ = 00`) and switches all eight channels to the test input. Per the ADS1299 data sheet this is a square wave of ±(VREFP − VREFN) / 2400 ≈ ±1.9 mV (VREF = 4.5 V) at f_CLK / 2²¹ ≈ 1 Hz; the capture should show it on all channels.
9. Connect the function generator to one EMG channel. Verify the gain (gain × Vin should match observed count × LSB) and the noise floor (<1 µVrms with input shorted, AC-coupled).
10. Connect the BNO085 to its JST-SH connector. `selftest` probes both IMUs and `imu_status` reports each slot. Start the live viewer on the host (`jtsmm view`) and rotate the sensor; verify that the orientation (quaternion) changes smoothly.
11. If **U1** was not factory-programmed: flash the HAT ID EEPROM per [`h02_hat_plus_eeprom_flashing.md`](h02_hat_plus_eeprom_flashing.md) (short **WP1 → GND1**, `eepmake` + `eepflash.sh`, verify `/proc/device-tree/hat/product`).
12. Insert the HAT onto the Pi 5 (after powering down both). Re-test steps 6–10 via the on-Pi capture process.
13. Trigger output test: send `trig_start` (or `jtsmm trigger`); a bare `trig_start` uses the applied configuration — 40 Hz with a 12 ms low time with the template [`software/jtsmm/etc/jtsmm.yaml`](../../software/jtsmm/etc/jtsmm.yaml), 30 Hz / 12 ms with the firmware defaults when no configuration was applied. Scope **`TP_TRIG1`** (ground **`TP_DGND_TRIG1`**); compare **`TP_TRIG_PIO1`** vs **`TP_CSI1`** (~3.3 V vs ~1.8 V). Verify frame rate, IMX296 **low-pulse** exposure polarity, and `trigger_mode=1` on Pi.

If any step fails, [`../operations/op03_troubleshooting.md`](../operations/op03_troubleshooting.md) has the standard remedies.
