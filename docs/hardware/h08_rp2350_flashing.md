<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h08 — RP2350 flashing and debug (SWD)

How to program the **RP2350A** on the custom HAT: what **SWD** is, why the **Raspberry Pi Debug Probe** uses a **3-wire** cable (not 10 pins), how **+3V3D** is supplied, and what must be on the PCB.

**Related:** pin map and firmware build — [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §1.1, §7; design checklist — [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §11.3; bench tool — [`../04_component_catalog.md`](../04_component_catalog.md) **D-03**.

---

## 1. What SWD is

**SWD = Serial Wire Debug** — ARM’s 2-wire debug interface for Cortex-M (including the RP2350's dual **Cortex-M33** cores). The RP2350 also supports a RISC-V debug mode for its Hazard3 cores; this guide uses the **Arm/SWD** path, which the Raspberry Pi Debug Probe drives directly.

| Layer | Signals | Count |
|-------|---------|------:|
| **SWD protocol (minimum)** | **SWCLK**, **SWDIO**, **GND** | **3** |
| **Recommended on the board** | above + **RUN** (hardware reset) | 4 nets |
| **Power for the MCU** | **+3V3D** from the HAT LDO — **not** from the debugger cable | separate rail |

The debugger does **not** flash over USB to the RP2350A directly; it bit-bangs **SWCLK/SWDIO** while the chip is **powered by your board**.

---

## 2. Three pins vs ten pins (do not confuse them)

These are two different things that are easy to mix up:

| | **Debug Probe cable (what you plug in)** | **Optional 2×5 header on the PCB** |
|---|------------------------------------------|-------------------------------------|
| **Pin count** | **3** (per kit cable) | **10** (1.27 mm pitch) |
| **Signals** | GND, SWCLK, SWDIO only | ARM “Cortex debug” footprint: same SWD + multiple GNDs + optional VTref sense + nRESET + NC/key |
| **Required for our project?** | **Yes** — matches **D-03** | **No** — only if you want a ribbon cable / other probes later |

The **Raspberry Pi Debug Probe** ([`D-03`](../04_component_catalog.md)) ships with **three** debug cables, all carrying the **same three signals** with different connectors:

1. **3-pin JST male ↔ 3-pin JST male** — use if the HAT has a matching **JST-SH 3-pin** footprint.
2. **3-pin JST → 0.1″ female header** — plugs onto **2.54 mm male pins** you solder (GND, SWCLK, SWDIO).
3. **3-pin JST → 0.1″ male header** — plugs into a **female** socket on the board.

There is **no VCC / 3.3 V line** on any of these cables.

---

## 3. How +3V3D is provided (not over the debug cable)

The Debug Probe **does not power** the HAT. **+3V3D** comes from the normal power tree:

```text
  Pi 5 V (HAT 2×20 header)  →  TPS73633DBV  →  +3V3D  →  RP2350A, W25Q128JVP, USB PHY, …
```

On the RP2350A the **+3V3D** rail also feeds the **on-chip SMPS core regulator** (`VREG_VIN`), which produces the 1.1 V **DVDD** through the external **3.3 µH** inductor ([`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.1). There is **no** separate core rail to supply during flashing.

**Bring-up sequence:**

1. Apply **5 V** to the HAT (Pi on, or bench supply on the 5 V input).
2. Confirm **+3V3D ≈ 3.3 V** at **`TP_3V3D1`** (or **`J_IMU_MTA1` pin 1** vs pin 2). Core **`DVDD`** is not probed at a test pad — see [`h05_mcu_schematic_build_guide.md`](h05_mcu_schematic_build_guide.md) §1.6b.
3. Connect the Debug Probe **USB** to the PC.
4. Connect only **GND, SWCLK, SWDIO** from the probe cable to the HAT.
5. Flash firmware.

**Ground is mandatory:** probe **GND** must connect to HAT **DGND** (common ground with the powered board).

On a full **10-pin ARM** debug connector, pin 1 (**VTref**) is only a **voltage sense** input for the debugger (so it knows the I/O level). The Debug Probe’s **3-wire** cable omits it; with the HAT self-powered at 3.3 V, that is normal for RP2350 work.

---

## 4. What to put on the PCB

### 4.1 Required for flash / run

| Item | RP2350A connection | Notes |
|------|--------------------|--------|
| **RP2350A** + **W25Q128JVP** QSPI flash | — | Program stored in external flash (16 MiB). The in-package **RP2354A** would remove this chip. |
| **3.3 µH core-regulator inductor** | VREG_LX → DVDD | On-chip SMPS buck (H-PAS-08); not present on the legacy RP2040 |
| **12 MHz crystal** + load caps | XIN / XOUT | |
| **+3V3D** rail | IOVDD / VREG_VIN / QSPI_IOVDD / USB_OTP_VDD / ADC_AVDD | From **TPS73633DBV** when **5 V** present |
| **BOOTSEL** button | **QSPI_SS** (active low) | ROM UF2 bootloader |
| **USB-C** | USB D+ / D− | UF2 drag-and-drop + USB-CDC ([`H-CON-06`](../04_component_catalog.md)) |
| **SWD breakout** | **SWCLK**, **SWDIO**, **GND** | See §4.2 |

### 4.2 SWD breakout (pick one — Debug Probe–friendly)

**Default recommendation:** expose **GND, SWCLK, SWDIO** on a footprint that matches one of the Debug Probe kit cables (§2).

| Option | Footprint | Cable from Debug Probe kit |
|--------|-----------|----------------------------|
| **A (recommended)** | **JST-SH 3-pin** 1.0 mm (e.g. SM03B-SRSS land) | JST ↔ JST |
| **B** | **3× 2.54 mm male header** (label GND, SWCLK, SWDIO) | JST → 0.1″ **female** |
| **C** | **3× 2.54 mm female** socket | JST → 0.1″ **male** |

Silkscreen pin 1 (**GND**) and the net names. Verify pin order against the Debug Probe documentation before first use.

**Also strongly recommended (not on the 3-wire cable):**

| Item | Connection |
|------|------------|
| **RESET** tactile button | RP2350A **RUN** (active low) — reliable reset during SWD flash |
| **Status LED** | **GPIO15** (**LED_HB**) + resistor — blink bring-up test |

> **Erratum RP2350-E9 (internal pull-downs).** On affected RP2350 silicon, a GPIO configured as a digital **input with the internal pad pull-down enabled** can latch at roughly 2.2 V instead of being pulled to ground. **Do not rely on internal pull-downs for a guaranteed logic low.** Panel buttons use **internal pull-up + switch to GND** (§1.4); ADS1299 control lines and I2C still use **external** pulls (**10 kΩ** / **2.2 kΩ**).

### 4.3 Optional: 2×5 ARM debug header (10 pins)

A **2×5, 1.27 mm** (“0.05 inch”) header is the **ARM Cortex debug** mechanical standard. It is **optional** for this project:

- **Not** needed for the Raspberry Pi Debug Probe (3-wire only).
- Useful if you later use a **10-pin ribbon** to another CMSIS-DAP probe or a different debugger.

If populated, wire only the used pins (pin 1 = VTref sense to **+3V3D**, 2 = SWDIO, 3/5/9 = GND, 4 = SWCLK, 6/8 = NC, 7 = key, 10 = **RUN**). See §5.

---

## 5. Pinout reference

### 5.1 Three-signal SWD (Debug Probe)

| Net | RP2350A | To Debug Probe cable |
|-----|---------|----------------------|
| **GND** | DGND | GND |
| **SWCLK** | dedicated **SWCLK** pad (pin 24) | SWCLK |
| **SWDIO** | dedicated **SWDIO** pad (pin 25) | SWDIO |

Dedicated pads — **not** GPIO numbers ([`h05`](h05_microcontroller_firmware.md) §1.1). On the QFN-60 the SWD pins are **24 / 25** (unchanged from the RP2040 numbering).

### 5.2 Optional 2×5 ARM header (1.27 mm)

View into the **male header on the PCB**, pin **1** marked (dot / square pad):

| Pin | Signal | Connect to |
|-----|--------|------------|
| 1 | VTref | **+3V3D** (sense only — board still powered from 5 V/LDO) |
| 2 | SWDIO | SWDIO |
| 3 | GND | DGND |
| 4 | SWCLK | SWCLK |
| 5 | GND | DGND |
| 6 | SWO | **NC** (unused on RP2350 here) |
| 7 | — | Key (no pin) |
| 8 | NC | Leave unconnected |
| 9 | GND | DGND |
| 10 | nRESET | **RUN** |

---

## 6. Two ways to load firmware

| Method | When | Needs |
|--------|------|--------|
| **A. SWD** (Debug Probe) | Development, first bring-up, debug, broken USB | Powered HAT + 3-wire SWD + PC with `pico-sdk` / OpenOCD / VS Code Pico |
| **B. UF2 / BOOTSEL** | Field update once USB works | Hold **BOOTSEL**, plug **USB-C** to PC, copy `.uf2` to the **RP2350** volume |

Both program the **W25Q128JVP**; after reset the RP2350A boots from flash.

### 6.1 Build firmware

**Full procedure:** [`firmware/jtsmm/README.md`](../../firmware/jtsmm/README.md). Summary:

```bash
# Toolchain (Debian / Raspberry Pi OS) — once
sudo apt install cmake gcc-arm-none-eabi libnewlib-arm-none-eabi \
  libstdc++-arm-none-eabi-newlib build-essential git

# pico-sdk — once
git clone https://github.com/raspberrypi/pico-sdk.git ~/pico/pico-sdk
cd ~/pico/pico-sdk && git submodule update --init

# Build (from repo clone)
export PICO_SDK_PATH=~/pico/pico-sdk
cd firmware/jtsmm && rm -rf build && mkdir build && cd build
cmake .. -DPICO_PLATFORM=rp2350
cmake --build . -j"$(nproc)"
```

Artifacts: `build/jtsmm.elf` (SWD), `build/jtsmm.uf2` (BOOTSEL).

MCU spec cross-reference: [`h05_microcontroller_firmware.md`](h05_microcontroller_firmware.md) §7.

### 6.2 Flash to the HAT

Flashing needs an **RP2350-aware OpenOCD** — not the distro `openocd` package: the Raspberry Pi build (`pico-sdk-tools` releases), installed once to `~/pico/openocd/`. From the repository root:

```bash
./scripts/flash_swd.sh  # HAT powered, Debug Probe on J_SWD1
```

`scripts/install_openocd.sh` installs that OpenOCD build.

**SWD (manual, after build):**

```bash
~/pico/openocd/openocd -s ~/pico/openocd/scripts \
  -f interface/cmsis-dap.cfg -f target/rp2350.cfg \
  -c "adapter speed 1000" \
  -c "program build/jtsmm.elf verify reset exit"
```

(Run from `firmware/jtsmm/` after §6.1, or pass the full path to the `.elf`.)

**UF2:**

```bash
# Hold BOOTSEL, connect USB-C, then:
cp build/jtsmm.uf2 /media/$USER/RP2350/
```

**Bench test over USB serial:** `picocom -b 115200 /dev/ttyACM0` → `info`, `selftest`. Command list in [`firmware/jtsmm/README.md`](../../firmware/jtsmm/README.md) (Bench verification).

VS Code extension **“Raspberry Pi Pico”** → interface **Raspberry Pi Debug Probe** is the lowest-friction GUI.

**Free alternative to D-03:** a spare **Raspberry Pi Pico 2** (or original Pico) flashed as **Picoprobe** ([`D-01`](../04_component_catalog.md)) — same 3-wire SWD idea.

---

## 7. Bench shopping list (flash only)

| ID | Item | Qty | Role |
|----|------|-----|------|
| **D-03** | [Raspberry Pi Debug Probe](https://www.raspberrypi.com/products/debug-probe/) | 1 | CMSIS-DAP SWD + UART; includes **3-pin SWD cables** (§2) |
| — | USB-C data cable (probe → PC) | 1 | Usually in Debug Probe box |
| — | USB-C data cable (HAT → PC) | 1 | UF2 / USB-CDC tests |
| — | Multimeter | 1 | Verify **+3V3D** before first SWD session |

Everything in §4.1 is on the **HAT BOM** ([`04_component_catalog.md`](../04_component_catalog.md)); optional **JST-SH 3-pin** land can be added as a footprint on the `mcu` sheet.

---

## 8. Bring-up order (summary)

1. Visual inspection — RP2350A, USB-C, W25Q128JVP ([`h02`](h02_custom_hat_shield.md) §12).
2. Power — **5 V** on; measure **+3V3D** and **+5VA** ([`h07`](h07_power_grounding_safety.md) §1).
3. **SWD** — connect **GND, SWCLK, SWDIO**; flash blinky to **GPIO15** (**LED_HB**).
4. **USB** — enumerate CDC; `jtsmm` heartbeat ([`h05`](h05_microcontroller_firmware.md)).
5. **UF2** — confirm BOOTSEL path for field updates.
