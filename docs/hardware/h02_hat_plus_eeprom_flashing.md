<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h02 — HAT+ ID EEPROM configuration and flashing

How to build and program the **AT24CS64** HAT ID EEPROM (**U1**) on the JTS-MM board so Raspberry Pi OS can identify the board as a **HAT+**.

**Related:** schematic capture — [`h02_hat_plus_schematic_build_guide.md`](h02_hat_plus_schematic_build_guide.md) §Step 2; mechanical stack — [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §NVMe coexistence; first-board bring-up — [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §12.

**Source files in repo:** [`hardware/jtsmm_hat/eeprom/`](../../hardware/jtsmm_hat/eeprom/) (`jtsmm_hat_eeprom_vert.txt` / `jtsmm_hat_eeprom_hori.txt`: PCB with vertical / horizontal EMG pins, each with its prebuilt `.eep`). The commands in this guide name the `_vert` files; for a board with horizontal EMG pins use the `_hori` files instead.

**New to Pi HAT bring-up?** Read [§0 Terminology](#0-terminology) first — every specialised term used in this guide is defined there. The rest of the document assumes you can look up words like *runtime overlay*, *bit-banged I2C*, or *sysfs* in that section.

---

## 0. Terminology

This section defines vocabulary used throughout the guide. Definitions are precise; nothing here is “simplified away,” but each term is explained on first principles.

### 0.1 Boards, connectors, and schematic names

| Term | Meaning |
|------|---------|
| **HAT** | *Hardware Attached on Top* — a board that plugs onto the Raspberry Pi 40-pin GPIO header. |
| **HAT+** | Raspberry Pi’s updated HAT standard (2022+). Adds stricter EEPROM metadata, power reporting, and other rules. This project follows HAT+. |
| **HAT ID EEPROM** | A small I²C memory chip on the HAT that stores vendor name, product name, UUID, and optional configuration. The Pi reads it to know *which* board is attached. On JTS-MM this is **U1**. |
| **U1, R1, WP1, …** | **Reference designators** from the schematic — unique labels for each component (`U` = IC, `R` = resistor, etc.). |
| **40-pin header / J1** | The GPIO connector between Pi and HAT. Carries power, ground, and signals including **GPIO0** and **GPIO1** used for the ID EEPROM. |
| **HAT stack** | One or more HAT boards stacked on the Pi. The top board’s EEPROM is usually at I²C address **0x50**; the board below may use **0x51**. |
| **DNP** | *Do Not Populate* — footprint on the PCB, but no part installed (e.g. **U2** is an alternate package for the same EEPROM). |
| **SOIC-8** | Small 8-pin surface-mount package shape of **U1** (AT24CS64). |
| **Test pad (WP1, GND1)** | Exposed copper spots for probes or tweezers during bench work — not a connector. |
| **Short / strap** | Electrically connect two points (e.g. tweezers from **WP1** to **GND1**). |
| **Floating** | A pin with no deliberate connection — may pick up noise; for **WP**, floating often means “write-protected.” |
| **LDO** | *Low-Dropout regulator* — on-board 3.3 V supply for the MCU (**`+3V3D`**). The HAT EEPROM is **not** fed from this; it uses **`HAT_3V3`** from the Pi header. |
| **Bring-up** | First power-on and validation of a new board assembly. |
| **Fab house** | PCB assembler / manufacturer. |

### 0.2 Memory and the `.eep` image

| Term | Meaning |
|------|---------|
| **EEPROM** | *Electrically Erasable Programmable Read-Only Memory* — non-volatile chip; keeps data when power is off. Slower to write than RAM; used here for fixed product identity. |
| **AT24CS64** | Microchip I²C EEPROM: **64 Kbit** (8192 **bytes**) capacity. The **CS** variant includes a factory **serial number** (not used by HAT+ tools but same part family). |
| **Flash / program** | In this doc: write the prepared `.eep` **image** into **U1**. (Not the same as “flash memory” inside the RP2350 MCU.) |
| **`.eep` file** | Binary file in Raspberry Pi **HAT+ format** produced by **`eepmake`**. This is what you copy into the physical EEPROM. |
| **Blank / erased chip** | Factory state: every byte **`0xFF`**. No valid `R-Pi` header — **`eepdump`** on a readback will fail until a good image is written. |
| **Image** | The complete byte contents written to the EEPROM (header + data + padding). |
| **Signature `R-Pi`** | Four-byte magic at the start of a valid HAT EEPROM binary identifying a Raspberry Pi HAT-family image. |
| **Format version `0x02`** | HAT+ layout version (legacy HAT used `0x01`). **`eepdump`** expects `0x02` for this project. |
| **Atom** | A typed chunk inside the `.eep` file (vendor info, power draw, optional device-tree blob, custom text, CRC). |
| **CRC** | *Cyclic Redundancy Check* — checksum so the Pi can detect corrupted EEPROM data. |
| **`dt_blob`** | Optional atom containing a **device-tree overlay** binary for Pi GPIOs. JTS-MM omits it — sensing runs on the **RP2350**, not Pi pins. |
| **UUID** | *Universally Unique Identifier* — 128-bit product ID; should be fixed in the settings file (`jtsmm_hat_eeprom_vert.txt` / `jtsmm_hat_eeprom_hori.txt`) after first **`eepmake`** run. |
| **`product_id` / `product_ver`** | 16-bit vendor-assigned numbers; **`product_ver`** encodes PCB revision (e.g. rev 0.0.1 → `0x0001`). |

### 0.3 I2C — the bus to the EEPROM

**I2C** (*Inter-Integrated Circuit*; written **I²C** in datasheets) is a two-wire serial bus: one data line (**SDA**) and one clock line (**SCL**). Many chips share the same two wires; each chip has an **address** so the host talks to one device at a time.

| Term | Meaning |
|------|---------|
| **SDA** | *Serial Data* — bidirectional data line. On the Pi HAT header this is **GPIO0**, labelled **ID_SDA**. |
| **SCL** | *Serial Clock* — timing line driven by the bus master. **GPIO1** / **ID_SCL**. |
| **GPIO** | *General-Purpose Input/Output* — a Pi pin that can be software-controlled or assigned to a hardware peripheral. “GPIO0” is pin number 0 on the Broadcom numbering used by Pi docs. |
| **Bus** | The shared SDA+SCL wiring plus all devices on it. Linux exposes each controller as **`i2c-0`**, **`i2c-13`**, etc. |
| **Bus master / controller** | The device that generates SCL and initiates transfers — here, the Raspberry Pi SoC’s I²C hardware (or, in bit-bang mode, the CPU). |
| **Slave / target device** | A chip that only responds when addressed — **U1** at **0x50**. |
| **7-bit I²C address** | Address sent on the wire (range `0x03`–`0x77`). Written in docs as **`0x50`** or hex column **`50`** in **`i2cdetect`** grids. **`eepflash -a=50`** means address 0x50 (no `0x` prefix in the flag). |
| **ACK** | *Acknowledge* — the slave pulls SDA low for one clock cycle to say “I’m here.” **`i2cdetect`** shows a hex digit when something ACKs. |
| **NACK** | *Not Acknowledge* — no response; empty **`--`** in **`i2cdetect`**, or errors in **`dmesg`**. |
| **Pull-up resistor (R1–R4)** | Resistors (here 3.9 kΩ) from SDA/SCL to **3.3 V**. I²C is **open-drain**: devices only pull lines **low**; pull-ups release them to **high** when idle. Without pull-ups the bus does not work. |
| **Open-drain** | A pin type that can pull a line low or leave it floating (high via pull-up). Explains why “stuck low” on SDA blocks the whole bus. |
| **SDA stuck at low** | Fault: the data line never returns to idle high — often a short, missing pull-up, or a damaged chip holding the line. |
| **Write protect (WP)** | EEPROM pin: when **high**, the chip **ignores** program cycles but still **ACKs** on the bus — `eepflash` may print **`Done.`** while every byte stays **`0xFF`**. When **low** (tied to **GND**), writes stick. Bench flash requires **WP1 → GND1** for the entire write. |
| **Address strap (A1)** | Solder jumper that sets the EEPROM’s least significant address bit. Open → default **0x50** on this PCB. |

#### 0.3.1 Hardware I2C vs bit-banged I2C

| Term | Meaning |
|------|---------|
| **Hardware I²C controller** | Dedicated silicon inside the Pi SoC that generates correct SCL timing and handles ACK/NACK in hardware. Fast and reliable. |
| **`dtoverlay i2c0`** | Enables that **hardware** controller on **GPIO0/1** (HAT ID pins). Creates Linux bus **`i2c-0`**. **Use this on Pi 5.** |
| **Bit-banged I2C** (*bit-bang*) | Software toggles GPIO pins high/low to imitate SCL/SDA without dedicated I2C hardware. Works in a pinch but is slower and more fragile. |
| **`i2c-gpio` overlay** | Device-tree overlay that creates a **bit-banged** bus on chosen GPIO numbers. Old HAT docs used **`bus=9`** on GPIO0/1. On **Pi 5** this often **fails** or maps the wrong physical pins — do not use for JTS-MM. |

**Analogy:** Hardware I2C is a dedicated telegraph office with trained operators; bit-banging is Morse code tapped by hand on two wires — same alphabet, much easier to get wrong under load.

### 0.4 Device tree and overlays

The **device tree** is a data structure (loaded at boot) that describes hardware to the Linux **kernel**: which controllers exist, which pins they use, and which drivers attach to them.

| Term | Meaning |
|------|---------|
| **Device tree (DT)** | Tree of nodes (devices) and properties (settings). Replaces old PC-style “plug and play” on ARM boards. |
| **Device-tree overlay (DTO)** | A **patch** to the device tree — adds or changes nodes without rebuilding the entire OS image. Stored as **`.dtbo`** files. |
| **Boot-time overlay** | Listed in **`/boot/firmware/config.txt`** as `dtoverlay=i2c0`. Applied when the Pi **powers on**, before Linux fully starts. Persists across reboots until you edit `config.txt`. |
| **Runtime overlay** | Loaded **while Linux is already running** via `sudo dtoverlay i2c0`. Lasts until reboot or `sudo dtoverlay -r i2c0` (**remove**). Convenient for one-off tests; Pi 5 may log harmless warnings when removing overlays. |
| **`config.txt`** | Raspberry Pi firmware configuration file on the boot partition. Controls overlays, GPU, etc. |
| **`dtparam`** | Similar `config.txt` mechanism for smaller toggles (e.g. `i2c_arm=on` on a **different** bus) — not a substitute for **`i2c0`** on GPIO0/1. |
| **`/proc/device-tree/`** | Linux filesystem view of the **effective** device tree after firmware + overlays. **`/proc/device-tree/hat/*`** appears when the Pi successfully read a valid HAT EEPROM at boot. |
| **Metadata / HAT node** | Vendor, product, UUID strings the Pi extracted from **U1** and exposed under **`/proc/device-tree/hat/`**. |

**Why two ways to enable I²C?** Early boot: **Pi firmware** reads the EEPROM with its own ID-I²C path (GPIO0/1). Later: **Linux** needs the **`i2c0`** overlay so tools like **`i2cdetect`** and **`eepflash`** can use the same pins. Programming the EEPROM requires the **Linux** path; verifying product name after reboot uses the **firmware** path.

#### 0.4.1 Runtime overlay vs boot-time overlay (worked example)

Think of the device tree as a wiring diagram the kernel reads at startup.

1. **Boot-time** (`dtoverlay=i2c0` in `config.txt`): The Pi firmware patches the diagram **before** Linux starts, every time you power on. The I2C bus on GPIO0/1 is “part of the machine” until you edit `config.txt` and reboot.

2. **Runtime** (`sudo dtoverlay i2c0` in an already-running shell): You patch the diagram **on demand** while Linux is up — useful when experimenting. The patch disappears after reboot, or immediately if you run `sudo dtoverlay -r i2c0`. On Pi 5, removing runtime overlays can log “memory leak” warnings; that is a known quirk, not damage to your board.

Neither type programs the EEPROM by itself — both only tell Linux *how to reach* the chip. You still run **`eepflash`** afterward.

### 0.5 Linux kernel, drivers, and filesystem interfaces

| Term | Meaning |
|------|---------|
| **Kernel** | The core of the operating system: drivers, scheduling, hardware access. User commands talk to hardware **through** the kernel. |
| **Kernel module** | Loadable driver code (`.ko` file). **`modprobe i2c-dev`** and **`modprobe at24`** insert modules without rebooting. |
| **Driver** | Kernel code for one device class — e.g. **`at24`** driver speaks I²C to EEPROM chips and exposes their memory. |
| **Bind / probe** | When the kernel attaches a driver to a device at an address (**probe** = detect and initialise). **`new_device`** forces probe at **0x50**. |
| **Userspace** | Normal programs you run in the shell (`eepflash`, `i2cdetect`) — not the kernel. They open **`/dev`** or **`sysfs`** nodes instead of touching GPIO directly. |
| **`sudo`** | Run as **root** (administrator). Required for hardware access and **`eepflash`**. |
| **`/dev/i2c-0`** | **Character device** — an interface file representing I²C bus 0. Tools may use this path or sysfs instead. |
| **Character device** | A `/dev` node for byte-stream devices (as opposed to **block devices** like disks). |
| **sysfs** | Virtual filesystem under **`/sys/`** — kernel exports hardware status as files. Example: **`.../0-0050/eeprom`** is the raw 8192-byte EEPROM contents. |
| **`dd`** | Unix “copy bytes” tool. **`eepflash`** uses **`dd`** to copy between `.eep` file and sysfs **`eeprom`** file. |
| **`dmesg`** | Ring buffer of **kernel log** messages — driver errors, “SDA stuck at low,” overlay warnings. |
| **`modprobe` vs `insmod`** | **`modprobe`** loads a module by name and resolves dependencies; you only need the name (`i2c-dev`). |

### 0.6 Commands and utilities (names only — procedure in §3+)

| Term | Meaning |
|------|---------|
| **`eepmake`** | Compile text settings → binary `.eep` (host-side). |
| **`eepdump`** | Decode `.eep` → readable text; validates signature and version. |
| **`eepflash.sh`** | Shell script: load **`at24`**, bind device, **`dd`** read/write. |
| **`i2cdetect`** | Scan an I²C bus for responding addresses (`-l` list buses, `-y N` scan bus **N** without prompts). |
| **`i2cdetect -l`** | **List adapters** — kernel-registered I²C controllers (`i2c-0`, …). |
| **`dtoverlay`** | Load or remove ( **`-r`** ) a runtime device-tree overlay. |
| **`pinctrl`** | Raspberry Pi OS tool to read/set GPIO mux and level. |
| **`hat_info`** | Optional command that prints HAT metadata in friendly form. |
| **`hexdump`** | Show file bytes in hexadecimal — used to inspect **`readback.eep`**. |
| **`wc -c`** | Count bytes in a file — expect **8192** for AT24CS64 readback. |
| **`cmake` / `make`** | Build **`eepmake`** and **`eepdump`** from Raspberry Pi **`utils`** source. |

### 0.7 Numbers and flags you will see

| Term | Meaning |
|------|---------|
| **64 Kbit** | EEPROM capacity in bits → **8192 bytes** (8 KiB). |
| **`24c64`** | Linux driver name for 64 Kbit I²C EEPROMs; **`eepflash -t=24c64`**. |
| **`i2c-0`** | Linux name for bus number **0** — the HAT ID bus after **`i2c0`** overlay. |
| **`0-0050`** | Sysfs name for device at bus **0**, address **0x50** (`0050` = hex 50, zero-padded). |
| **`i2c-13` / `i2c-14`** | Other I²C controllers on Pi 5’s **RP1** southbridge — unrelated to HAT ID EEPROM. |
| **`-d=0`** | Tell **`eepflash`** to use bus **0**. |
| **`-a=50`** | Target I2C address **0x50**. |
| **`-y`** | Non-interactive / “assume yes” flag (`i2cdetect` and `eepflash`). |
| **`0xFF`** | Hex byte value 255 — erased EEPROM cell. |
| **SoC** | *System on Chip* — the main processor die in the Pi (CPU, GPIO, I2C controllers, etc.). |
| **Firmware** | Low-level code that runs before or alongside Linux on the Pi — reads the HAT EEPROM early in boot. Distinct from the **kernel** (Linux) and from **RP2350 firmware** on the HAT MCU. |
| **RP1** | I/O chip on Pi 5 that provides many GPIOs and extra I2C buses (`i2c-13`, `i2c-14`) — not the HAT ID bus. |
| **Power-cycle** | Turn power off and on (or reboot) so hardware re-initialises. |
| **Little-endian** | Byte order: least significant byte stored first in memory/files — affects how multi-byte values appear in **`hexdump`**. |
| **ASCII** | Standard text encoding for human-readable strings in the EEPROM (`vendor`, `product`). |

---

## 1. When you need this

| Situation | Action |
|-----------|--------|
| Fab house did **not** pre-program U1 | Flash once before shipping or during bring-up |
| Vendor/product strings or `product_id` change | Rebuild `.eep` and re-flash |
| Pi does not list the board under HAT metadata | Verify EEPROM contents (§6) |

The Pi reads this EEPROM at boot over **GPIO0 (ID_SDA)** and **GPIO1 (ID_SCL)** — see [§0.3 I2C](#03-i2c--the-bus-to-the-eeprom). It does **not** affect EMG, IMU, or RP2350 firmware — those use USB from the on-board **MCU** (*Microcontroller Unit*, the RP2350A).

---

## 2. Hardware on this board

| Ref | Part | Role |
|-----|------|------|
| **U1** | AT24CS64-SSHM-T (SOIC-8) | 64 Kbit HAT ID EEPROM |
| **U2** | AT24CS64-MAHM-T | DFN alternate — **DNP**; populate **U1** only |
| **R1–R4** | 3.9 kΩ | I2C pull-ups to `HAT_3V3` |
| **A1** | 2-pad solder jumper | EEPROM address strap (template default → **0x50**) |
| **WP1** | Test pad (bottom) | EEPROM **WP** — tie to **GND1** while writing |
| **GND1** | Test pad (bottom) | **DGND** — ground reference for WP strap |

**Power:** U1 `VCC` is **`HAT_3V3`** from the Pi 40-pin header (not `+3V3D` from the on-board LDO).

**I2C address (default build):**

| Strap | Value on this PCB | Result |
|-------|-------------------|--------|
| A2, A1 | Tied to **DGND** on **U1** | — |
| A0 | **A1** solder jumper (leave **open** unless changing address) | **0x50** expected |

Confirm with `sudo i2cdetect -y 0` on **Pi 5** (after `dtoverlay i2c0`; see §5.3). Use **0x50** for this HAT. An **M.2 HAT+** underneath uses **0x51** — only relevant when that board is in the stack; with **only** the JTS-MM HAT you should see **0x50** alone (or nothing at all if **U1** is blank and not yet responding — see §8).

**Write protect:** With **WP1** floating, the AT24CS64 treats **WP** as high and blocks writes. Short **WP1 → GND1** (tweezers or a brief jumper wire) for the whole flash operation, then remove the short.

---

## 3. Toolchain (Raspberry Pi official)

Use the current **HAT+** utilities in Raspberry Pi **`utils`** (supersedes the old `raspberrypi/hats` repo):

```bash
git clone https://github.com/raspberrypi/utils.git
cd utils/eeptools
cmake .
make
sudo make install   # installs eepmake, eepdump; eepflash.sh stays in this directory
```

On the Pi (or cross-build host with the same repo):

```bash
sudo apt install i2c-tools cmake build-essential
```

References:

- [raspberrypi/utils/eeptools](https://github.com/raspberrypi/utils/tree/master/eeptools)
- [HAT+ specification](https://pip.raspberrypi.com/documents/raspberry-pi-hat-plus-specification.pdf)

### 3.1 What the tools do (overview)

| Tool | Input → output | Role |
|------|----------------|------|
| **`eepmake`** | `.txt` settings → binary `.eep` | Builds a **HAT+ format v0x02** image on your PC or Pi (no hardware needed) |
| **`eepdump`** | `.eep` binary → human-readable text | Validates and inspects an image **before** or **after** flash |
| **`eepflash.sh`** | `.eep` ↔ physical **U1** over I2C | Low-level read/write via kernel `at24` driver + `dd` |
| **`i2cdetect`** | I2C bus number | Lists which addresses ACK (chip present) |
| **`dtoverlay`** | overlay name | Loads a **device-tree overlay** at runtime (enables `i2c-0` on GPIO0/1) |
| **`modprobe`** | kernel module name | Loads drivers (`i2c-dev`, `at24`) |

**Typical order:** edit `jtsmm_hat_eeprom_vert.txt` → `eepmake` → `eepdump` (sanity) → on Pi: enable bus → `i2cdetect` → `eepflash -w` → reboot → `cat /proc/device-tree/hat/*`.

### 3.2 Device tree overlays — what is `dtoverlay i2c0`?

Definitions: [§0.4 Device tree and overlays](#04-device-tree-and-overlays), [§0.3.1 Hardware I2C vs bit-banged](#031-hardware-i2c-vs-bit-banged-i2c).

A **device tree** tells the Pi kernel which hardware exists and which pins each peripheral uses. A **device-tree overlay** (`.dtbo` file) patches that tree **without** rebuilding the whole OS image.

| Command / setting | What it does |
|-------------------|--------------|
| `sudo dtoverlay i2c0` | **[Runtime overlay](#04-device-tree-and-overlays):** while Linux is already running, load the `i2c0` patch. Enables the SoC **hardware I²C controller** on **GPIO0 (SDA)** and **GPIO1 (SCL)**. Creates adapter **`i2c-0`** and `/dev/i2c-0`. Lost on reboot unless also in `config.txt`. |
| `dtoverlay=i2c0` in `/boot/firmware/config.txt` | **[Boot-time overlay](#04-device-tree-and-overlays):** firmware applies the same patch on every power-on. Preferred for a bench that flashes many boards. |
| `sudo dtoverlay -r i2c0` | **Remove** a previously loaded **runtime** overlay. Does not remove entries baked into `config.txt` (edit that file and reboot instead). |
| `sudo dtoverlay i2c-gpio … bus=9` | **[Bit-banged I2C](#031-hardware-i2c-vs-bit-banged-i2c)** on chosen GPIOs — legacy HAT docs; **unreliable on Pi 5** (wrong GPIO mapping / overlay load failures). **Do not use** for this project on Pi 5. |

Overlays are **not** the same as `config.txt` options like `dtparam=i2c_arm=on` (that enables a **different** I²C bus on other pins). For HAT ID EEPROM you specifically need **`i2c0`** on GPIO0/1.

The `dmesg` warnings about overlay removal and “memory leak” appear when dropping **runtime** overlays on Pi 5 — annoying but harmless for bench use; use **`config.txt`** for a stable setup.

### 3.3 Kernel modules used during flash

See [§0.5 Linux kernel, drivers, and filesystem interfaces](#05-linux-kernel-drivers-and-filesystem-interfaces).

| Command | Module | Purpose |
|---------|--------|---------|
| `sudo modprobe i2c-dev` | `i2c-dev` | Exposes `/dev/i2c-N` **character devices** and `/sys/class/i2c-dev/` so **userspace** tools can talk to the bus |
| `modprobe at24` | `at24` | **EEPROM driver**; `eepflash.sh` loads this automatically. **Binds** **24c64** at **0x50** and creates `.../0-0050/eeprom` **sysfs** file |

`eepflash.sh` internally runs `modprobe i2c_dev` and `modprobe at24`, then writes `24c64 0x50` to **`new_device`** if the chip is not already bound.

### 3.4 I2C discovery commands

Terminology: [§0.3 I2C](#03-i2c--the-bus-to-the-eeprom).

```bash
i2cdetect -l
```

Lists **adapters** (kernel-registered **I2C controllers** / buses). After `dtoverlay i2c0` you want **`i2c-0`**. Entries like `i2c-13` / `i2c-14` are other Pi 5 **RP1** peripherals — ignore them for EEPROM work.

```bash
sudo i2cdetect -y 0
```

**Probes bus 0** — sends read requests to addresses `0x03`–`0x77` and prints a grid:

| Cell | Meaning |
|------|---------|
| `--` | No device ACK at that address |
| `50` | Device responds at **0x50** (your EEPROM) |
| `UU` | Address in use — kernel driver already bound (OK for flash) |

You **must** see `50` or `UU` at column **0x50** before writing. An empty grid with `dmesg` **SDA stuck at low** is a hardware fault (§8.2).

```bash
pinctrl get 0-1
```

Shows Pi GPIO0/1 mux level and pull state (needs Pi OS `pinctrl` utility). Useful to confirm ID pins are not stuck low before blaming the HAT.

### 3.5 `eepmake` and `eepdump` (file-side, no Pi hardware)

**`eepmake`** parses the text settings file and writes a binary **`.eep`** file:

- Magic signature **`R-Pi`** (HAT family)
- Format version **`0x02`** (HAT+, not legacy HAT v1 `0x01`)
- **Atoms:** vendor block (UUID, strings), optional GPIO map, optional `dt_blob`, custom data, power supply, CRC

**`eepdump`** reverses that — prints `# vendor …`, `# product …`, etc. Always run on the **source** image before flashing:

```bash
eepdump jtsmm_hat_eeprom_vert.eep    # must succeed with no WARNING lines
```

A valid HAT+ image is small (often **~200–300 bytes** for a minimal JTS-MM image). **`eepflash`** copies **only the `.eep` file length** via `dd` — e.g. **238 bytes** — into the start of the chip. The rest of the 8192-byte AT24CS64 stays **`0xFF`** (erased), which is fine for HAT+.

### 3.6 `eepflash.sh` flags (full reference)

```bash
sudo ./eepflash.sh -y -w -d=0 -t=24c64 -a=50 -f=/path/to/jtsmm_hat_eeprom_vert.eep
```

| Flag | Meaning |
|------|---------|
| `-w` / `-r` | **Write** image to EEPROM, or **read** EEPROM to file |
| `-f=FILE` | Path to `.eep` file (`-f FILE` also works) |
| `-d=N` | I2C **adapter number** (`0` → `/dev/i2c-0`, sysfs `i2c-0`) |
| `-a=50` | 7-bit I2C address in **hex without `0x`** (`50` = `0x50`) |
| `-t=24c64` | EEPROM size/driver hint — **8192 bytes** for AT24CS64 |
| `-y` | Skip “Do you wish to continue?” prompt |

**What it does internally:** load `at24` → ensure `0-0050` exists on bus `N` → `dd` between file and `/sys/.../eeprom` → `delete_device` (removes forced binding when script exits).

**Write requires:** WP1 grounded, bus healthy, `i2cdetect` shows `50`/`UU`.

**Read requires:** same bus health; output file should be **8192 bytes** for `-t=24c64`.

---

## 4. Configure the EEPROM image

### 4.1 Edit the settings file

Copy or edit the project template:

```bash
cd hardware/jtsmm_hat/eeprom
cp jtsmm_hat_eeprom_vert.txt myhat_eeprom.txt   # optional working copy
$EDITOR myhat_eeprom.txt
```

Important fields:

| Field | Purpose |
|-------|---------|
| `product_uuid` | 128-bit UUID; leave all-zero once so `eepmake` generates one, then **paste the generated UUID back** into the file for reproducible builds |
| `product_id` / `product_ver` | 16-bit IDs for your product line / PCB revision |
| `vendor` / `product` | ASCII strings shown in `hat_info` / device tree metadata |
| `current_supply` | Optional max 5 V draw in mA from the Pi header |
| `dt_blob` | **Omit** for JTS-MM — no Pi GPIO device-tree overlay is required |

This project does **not** use `dt_blob` because sensing and timing run on the **RP2350A** and are exposed over **USB**, not as Pi peripheral overlays.

### 4.2 Build the binary

```bash
eepmake jtsmm_hat_eeprom_vert.txt jtsmm_hat_eeprom_vert.eep
```

`eepmake` prints the assigned `product_uuid` if you left it zeroed — **save it** in the `.txt` file.

Sanity-check the binary:

```bash
eepdump jtsmm_hat_eeprom_vert.eep
```

---

## 5. Flash procedure (on Raspberry Pi 5)

Perform this with **5 V present** (Pi powered, HAT assembled). The EEPROM is only powered from the Pi header (`HAT_3V3` on the 40-pin connector).

### 5.1 Stack configuration

| Stack | Expected EEPROM at 0x50 | Notes |
|-------|-------------------------|-------|
| Pi 5 → **JTS-MM HAT only** | **U1** on custom HAT | Normal bench bring-up (no M.2 board required) |
| Pi 5 → M.2 HAT+ → JTS-MM (top) | **U1** on custom HAT | M.2 EEPROM is at **0x51**; flash with `-a=50` only |

Flashing always targets **U1** at I2C address **0x50**.

### 5.2 Disable write protection

1. Locate **WP1** and **GND1** on the **bottom** side near **U1** (~60–62 mm, y ≈ 39 mm in KiCad coords).
2. Short **WP1** to **GND1** with a grounded clip or tweezers — **keep the short in place** for steps 5.3–5.4.

Write-protect only blocks **programming**; the chip should still **ACK** at 0x50 on `i2cdetect` even when WP is high. If the bus scan is completely empty, the problem is power, seating, or **U1** — not WP alone.

### 5.3 Expose the ID EEPROM bus (Pi 5)

GPIO0 (**ID_SDA**) and GPIO1 (**ID_SCL**) are the dedicated HAT ID I2C pins. On **Raspberry Pi 5**, use the hardware **`i2c0`** overlay — **not** `i2c-gpio bus=9` (that overlay often fails at runtime on Pi 5 or maps the wrong GPIO lines).

```bash
sudo modprobe i2c-dev
sudo dtoverlay i2c0
i2cdetect -l
```

Expect a new adapter:

```text
i2c-0   i2c    Synopsys DesignWare I2C adapter    I2C adapter
```

Other buses (`i2c-13`, `i2c-14`, …) are unrelated RP1 peripherals — **ignore them** for HAT EEPROM work.

Scan bus **0**:

```bash
sudo i2cdetect -y 0
```

Before first programming you may see **`50`** in the grid (blank but responding EEPROM) or **`UU`** (kernel driver already bound). An entirely empty row at 0x50 (`--`) means **no ACK** — fix hardware (§8) before running `eepflash`.

**Optional — persist across reboots** (recommended for production benches):

```ini
# /boot/firmware/config.txt
dtoverlay=i2c0
```

Then reboot and skip the manual **runtime overlay** command (`sudo dtoverlay i2c0`) — see [§0.4](#04-device-tree-and-overlays).

#### Pi 4 and earlier (legacy note)

Older docs used [bit-banged I2C](#031-hardware-i2c-vs-bit-banged-i2c) via `i2c-gpio` as bus **9**:

```bash
sudo dtoverlay i2c-gpio i2c_gpio_sda=0 i2c_gpio_scl=1 bus=9
sudo i2cdetect -y 9
```

Do **not** use this on Pi 5 unless `i2c0` is unavailable on your image.

### 5.4 Write the image

From the directory containing `eepflash.sh` (after building `utils/eeptools`):

```bash
sudo ./eepflash.sh -y -w -d=0 -t=24c64 -a=50 -f=/path/to/jtsmm_hat_eeprom_vert.eep
```

| Flag | Value | Why |
|------|-------|-----|
| `-d` | `0` | **Pi 5:** HAT ID bus after `dtoverlay i2c0` (not bus 9) |
| `-t` | `24c64` | Matches **AT24CS64** (64 Kbit) |
| `-a` | `50` | Default strap on this PCB |
| `-f` | `jtsmm_hat_eeprom_vert.eep` | Output of `eepmake` |
| `-y` | — | Skip interactive confirmation (scripts) |

**`238 bytes copied` on write is normal** — that is the `.eep` file size, not the full 8192-byte chip. After a good write, `hexdump -C readback.eep | head -1` must **not** be all `ff`; expect `52 2d 50 69` (`R-Pi`) at the start.

If you omit `-d`, `eepflash.sh` auto-picks `/dev/i2c-0` when present (correct on Pi 5 after `i2c0`).

### 5.5 Re-enable write protection

Remove the **WP1 → GND1** short. Power-cycle the Pi and confirm HAT metadata (§6).

### 5.6 Optional: read back and verify

Only meaningful **after a successful write** on a **healthy bus** (`i2cdetect` showed `50`/`UU`, write finished with `Done.` and no `Error doing I/O`).

```bash
sudo ./eepflash.sh -y -r -d=0 -t=24c64 -a=50 -f=readback.eep
wc -c readback.eep          # expect 8192 for AT24CS64
hexdump -C readback.eep | head -1   # valid HAT+ starts with signature bytes (see below)
eepdump readback.eep
```

Compare to the source image:

```bash
eepdump jtsmm_hat_eeprom_vert.eep > /tmp/expected.txt
eepdump readback.eep > /tmp/got.txt
diff -u /tmp/expected.txt /tmp/got.txt
```

**Valid HAT+ binary header** (first bytes on disk, [little-endian](#07-numbers-and-flags-you-will-see) `uint32` signature):

```text
00000000  52 2d 50 69 02 00 01 00  ...
          R  -  P  i  ^ver ...
```

#### If `eepdump readback.eep` reports errors

Example (EEPROM never programmed correctly or bus read failed):

```text
WARNING: Format signature mismatch
# WARNING: format version mismatch!!!
Unexpected EOF or error occurred
```

| Cause | What happened | What to do |
|-------|----------------|------------|
| **Blank / erased chip** | All `0xFF` — no `R-Pi` header | Normal **before** first good write; run `eepflash -w` with **WP grounded** |
| **Write never succeeded** | `dd` failed (permission, SDA stuck) **or** WP high (silent no-op) | Fix §8; for silent WP failure see row above; confirm first readback bytes are `52 2d 50 69` |
| **Truncated read** | `wc -c readback.eep` **< 8192** | Bus fault or sysfs node missing mid-read; fix hardware, re-read |
| **Corrupt partial write** | Size OK but garbage header | Re-flash with WP grounded; power-cycle between attempts |
| **Wrong `-t` type** | Size mismatch vs chip | Use `-t=24c64` for **AT24CS64** |

`eepdump` on **`jtsmm_hat_eeprom_vert.eep`** (built by `eepmake` on the host) should **always** succeed with no WARNING lines. If that file dumps cleanly but `readback.eep` shows signature errors (not the padding warnings below), the problem is **on the Pi/EEPROM side**, not the source settings.

#### Expected warnings on full-chip readback (harmless)

`eepflash -r` always saves **8192 bytes** (full AT24CS64). The HAT+ image inside is only **`eeplen`** bytes (here **238**); the rest of the chip is padding **`0xFF`**.

If the write succeeded, `eepdump readback.eep` will show the correct vendor/product atoms and may end with:

```text
WARNING: Dump finished before EOF
WARNING: EOF does not match length specified in header
WARNING: 8192 bytes of file not processed
```

| Message | Meaning |
|---------|---------|
| **Dump finished before EOF** | Valid HAT+ block ended at byte 238; file continues with unused `0xFF` fill |
| **EOF does not match length specified in header** | Header says image is 238 bytes; file is 8192 — normal for a raw chip read |
| **8192 bytes of file not processed** | Trailing erased cells — **not** part of the HAT+ structure |

**Ignore these three** if the atoms above match `jtsmm_hat_eeprom_vert.eep` and CRC lines are present. Compare content with:

```bash
eepdump jtsmm_hat_eeprom_vert.eep | grep -v WARNING > /tmp/expected.txt
eepdump readback.eep | grep -v WARNING > /tmp/got.txt
diff -u /tmp/expected.txt /tmp/got.txt
```

(`diff` should be empty through the last atom.)

To avoid the warnings, read only the image length: `dd if=readback.eep of=readback_238.eep bs=238 count=1` then `eepdump readback_238.eep`.

**Do not** use manual `echo 24c64 0x50 > .../new_device` as a substitute for a clean `i2cdetect` + `eepflash` sequence — it can create a sysfs node even when the bus cannot reliably read data (§8.2).

---

## 6. Verify on the Pi

After reboot (remove runtime overlays with `sudo dtoverlay -r i2c0` if you did not add `dtoverlay=i2c0` to `config.txt`):

```bash
# Device tree metadata (Pi 5 / Bookworm) — only present after valid EEPROM programmed
cat /proc/device-tree/hat/vendor 2>/dev/null || echo "(no hat node — EEPROM missing or invalid)"
cat /proc/device-tree/hat/product
cat /proc/device-tree/hat/product_uuid

# Alternative helper (if installed)
hat_info
```

| Check | Meaning |
|-------|---------|
| `/proc/device-tree/hat/*` exists | Pi firmware read **U1** at boot and accepted a **valid HAT+ v0x02** image |
| `(no hat node)` | Blank EEPROM, corrupt image, bus fault at boot, or **U1** not populated |
| `hat_info` | User-space pretty-printer for the same metadata |

Expected strings match the settings file (e.g. vendor `TU Darmstadt ExoFusion`, product `JTS-MM — Multimodal Jaw Tracking System HAT+`).

The Pi reads the EEPROM **once early in boot** over GPIO0/1 — you do not need `dtoverlay i2c0` in Linux for this metadata to appear; the firmware has its own ID I2C path. Runtime `i2c0` is for **Linux tools** (`i2cdetect`, `eepflash`) during bench programming.

---

## 7. Factory / bench scripting

Minimal scripted flow (run on the Pi after copying `jtsmm_hat_eeprom_vert.eep`):

```bash
#!/usr/bin/env bash
set -euo pipefail
EEPTOOLS="${EEPTOOLS:-$HOME/utils/eeptools}"
EEP_FILE="${1:?usage: flash_jtsmm_hat_eeprom.sh /path/to/jtsmm_hat_eeprom_vert.eep}"

echo "Attach WP1 to GND1, then press Enter"
read -r

sudo modprobe i2c-dev
sudo dtoverlay i2c0
sudo i2cdetect -y 0 | grep -qE ' 50|UU' || { echo "0x50 not found on i2c-0"; exit 1; }

sudo "$EEPTOOLS/eepflash.sh" -y -w -d=0 -t=24c64 -a=50 -f="$EEP_FILE"
echo "Remove WP1 short and reboot"
```

Store production images under version control or your release artifacts with the **UUID** and `product_ver` recorded.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| `No such file or directory` for `/dev/i2c-9` | Pi 5 does not use `i2c-gpio` bus 9 | Use `sudo dtoverlay i2c0` and bus **0** (§5.3) |
| `Failed to apply overlay 'i2c-gpio'` on Pi 5 | Runtime `i2c-gpio` broken / wrong GPIO mapping | Use `dtoverlay i2c0` instead |
| `i2cdetect -l` shows only `i2c-13` / `i2c-14` | ID bus not enabled yet | `sudo dtoverlay i2c0`; look for **`i2c-0`** |
| `i2cdetect -y 0` empty at **0x50** (`--`) | **U1** not populated, bad reflow, HAT not seated, no `HAT_3V3`, **SDA stuck low** | Inspect **U1**, reseat header, measure 3.3 V at EEPROM `VCC` vs **DGND**; see §8.1 |
| `dmesg`: `SDA stuck at low` on `i2c_designware` | SDA held low — no valid I2C idle state | Measure **ID_SDA** at header pin 27 (~3.3 V idle); isolate Pi vs HAT (§8.2) |
| `new_device` then `delete_device` for `24c64` | Driver bound but probe/read failed (bus fault) | Fix SDA/SCL first; do not rely on manual `new_device` alone |
| `at24 0-0050: supply vcc not found` | Normal when forcing `new_device` | Harmless warning — not the root cause |
| `eepflash`: `Permission denied` on `.../eeprom` | No device at 0x50 — `at24` driver did not bind | **Do not flash** until `i2cdetect -y 0` shows `50` or `UU`; fix hardware first |
| `eepflash` prints **`Done.`** but readback is all **`0xFF`** | **WP** high — chip write-protected; kernel `dd` “succeeds” but cells do not change | Short **WP1 → GND1** *before* `-w`; keep short until `Done.`; re-read and check header bytes |
| `eepflash` write fails with WP error | WP not grounded (some tools detect explicitly) | Short **WP1 → GND1** during entire write |
| Wrong product after flash | Wrote wrong address | Confirm `-a=50` and `-d=0`; re-read with `eepflash -r` |
| UUID changes every build | `product_uuid` left zero | Paste stable UUID into `.txt` after first `eepmake` |
| No `/proc/device-tree/hat/*` after flash | Invalid `.eep`, wrong address, or WP removed mid-write | `eepdump` the image; re-flash with WP grounded |
| `eepdump readback.eep`: signature / version mismatch, unexpected EOF | Read from blank, corrupt, or truncated EEPROM | See §5.6; check `wc -c readback.eep`; fix bus before re-flash |
| `50` with jumpers, `--` when fully stacked | **Intermittent J1 mate** (rectangular vs square stacking pins) or other header issue | §8.3; [`h02` §11.5.1](h02_custom_hat_shield.md#1151-stacking-pins-below-j1) — use **0.025″ square** posts |
| `new_device` then immediate `delete_device` | `at24` probe read failed (bus fault) | Fix transaction-level fault; DMM idle voltage is not enough |

### 8.1 Diagnose an empty bus scan

With `dtoverlay i2c0` loaded:

```bash
pinctrl get 0-1
# Expect: ID_SDA / ID_SCL on GPIO0–1 (often input + pull-up before overlay)

sudo i2cdetect -y 0
# Must show 50 or UU at column 0x50 before eepflash
```

Manual driver bind (debug only — should succeed if hardware is OK):

```bash
echo 24c64 0x50 | sudo tee /sys/bus/i2c/devices/i2c-0/new_device
dmesg | tail -10
ls -l /sys/bus/i2c/devices/0-0050/eeprom
```

| Sysfs path | Role |
|------------|------|
| `/sys/bus/i2c/devices/i2c-0/new_device` | Ask kernel to probe address **0x50** as **24c64** |
| `/sys/bus/i2c/devices/0-0050/eeprom` | Raw EEPROM contents — `eepflash` uses `dd` on this file |
| `.../delete_device` | Remove forced device (`eepflash` does this on exit) |

If `new_device` fails or `dmesg` reports NACK, the Pi cannot see **U1** — check soldering, address strap **A1** (open → 0x50), and `R1–R4` pull-ups.

**Note:** sysfs paths use `/sys/class/i2c-dev/i2c-0/` (character device side). There is no `/sys/class/i2c-adapter/` on some Pi 5 kernels — use `i2cdetect -l` instead.

### 8.2 Isolate “SDA stuck at low”

`dmesg` line:

```text
i2c_designware ... i2c_dw_handle_tx_abort: SDA stuck at low
```

means the controller could not release **ID_SDA** (header pin **27**) to idle high. Typical causes on this HAT:

| Check | What to look for |
|-------|------------------|
| **R1–R4** (3.9 kΩ to `HAT_3V3`) | All four populated; measure **~3.3 V** on SDA/SCL at **J1** pins 27/28 with Pi on |
| **U1** | Correct orientation; no solder bridge on pin 5 (SDA) or pin 6 (SCL) |
| **HAT off Pi** | `sudo dtoverlay i2c0` then `pinctrl get 0-1` — lines should not read **lo** with no HAT |
| **HAT on, Pi off** | Ohmmeter: SDA/SCL to **DGND** should not be a hard short |
| **WP1 → GND1** | Required for **writing** only — does **not** explain SDA stuck during a bus scan |

After the bus idles high, `sudo i2cdetect -y 0` should show **`50`** (or **`UU`**). Only then run `eepflash`.

**Spurious address `58`:** An extra entry in the `0x5x` row (e.g. **`58`**) with **`50`** present often means bus noise, a weak connection, or another device — focus on **`50`** for **U1**. If writes to `0x50` succeed, ignore `58` unless problems persist.

Manual `new_device` can create a sysfs node even when the bus is unhealthy; the first bind may be removed immediately (`delete_device`) if the driver cannot read the EEPROM:

```bash
echo 24c64 0x50 | sudo tee /sys/bus/i2c/devices/i2c-0/new_device
ls /sys/bus/i2c/devices/0-0050/eeprom 2>/dev/null || echo "probe failed / device gone"
sudo i2cdetect -y 0
```

### 8.3 EEPROM visible with jumper wires but not when fully stacked

**Symptom:** `i2cdetect -y 0` shows **`50`** when you power the HAT with flying leads to **3.3 V + GND + SDA + SCL**, but shows **`--`** at **0x50** when the full 40-pin header is mated — even if a DMM shows **~3.3 V** on pins **1 / 27 / 28** at idle.

**Important:** Idle voltage only proves pull-ups and power. **I2C can still fail during clocking** if something holds **SDA low** while the controller runs a transaction. That matches:

```text
i2c_designware ... i2c_dw_handle_tx_abort: SDA stuck at low
i2c i2c-0: new_device: Instantiated device 24c64 at 0x50
i2c i2c-0: delete_device: Deleting device 24c64 at 0x50
```

The kernel **binds** `at24`, **probe read fails**, then **removes** the device again. `i2cdetect` and `eepflash` will look broken even though the multimeter looked fine.

The schematic isolates **ID_SDA** / **ID_SCL** to **U1**, **R2–R3**, and header pins **27** / **28** only. Pass-through Pi GPIOs on the HAT template are **no-connect** on the root sheet — they should not load the ID bus.

**Confirmed root cause (bench):** **Intermittent mate** between **J1** (Samtec **HLE** socket) and generic **13.3 mm** stacking pins (**0.64 × 0.40 mm** rectangular — e.g. BerryBase RPI-CGST-133). DMM idle voltage on **1 / 27 / 28** was fine; I2C failed when fully stacked. **Square 0.025″ posts** or direct Pi pins fix it. Details: [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §11.5.1.

Other full-stack-only causes (if pins are already square): **(A)** another header pin disturbing ground, **(B)** +5 V domain noise, **(C)** socket vs jumper path.

#### Step 1 — Confirm the failure mode (fully stacked)

```bash
sudo dtoverlay -r i2c0 2>/dev/null
sudo dtoverlay i2c0
sudo i2cdetect -y 0          # note 0x50 and any ghosts (e.g. 58)
dmesg | tail -8
```

Do **not** rely on manual `new_device` for pass/fail — it forces the same probe that is already failing.

#### Step 2 — Minimal header mating (best isolation test)

With the HAT **physically seated**, electrically connect **only**:

| Pin | Signal |
|-----|--------|
| **1** | 3.3 V |
| **2** or **4** | 5 V *(if you need the HAT digital section on)* |
| **6** (+ more GND pins if you have them) | GND |
| **27** | ID_SDA |
| **28** | ID_SCL |

Isolate **all other pins** with tape / no-connect inserts on the **Pi** header or the HAT socket. Re-run `i2cdetect`.

| Result | Interpretation |
|--------|----------------|
| **`50` appears** with minimal pins | Another mated pin (or combination) disturbs the bus or ground — binary-search which pin group |
| Still **`--`** with only 1/2/4/6/27/28 | Focus on SDA/SCL path (socket vs jumper route), RP2350 activity (step 3), or Pi `config.txt` overlays |

Your jumper test likely differed in **two** ways: direct wires to known pads **and** most Pi↔HAT pins **not mated at all**.

#### Step 3 — Isolate the +5 V / RP2350 domain

With **full** header mated:

1. **5 V blocked** — tape pins **2 + 4**; keep **1 + 6 + 27 + 28**. EEPROM is on `HAT_3V3` (pin 1), so it can still work. If **`50` appears**, the fault correlates with powering the MCU/analog section.
2. **RP2350 held in reset** — pull **RUN** low on the HAT (if accessible) while scanning.
3. **USB unplugged** from the HAT MCU port during the scan.

If **`50` only when RP2350 is in reset / 5 V off**, capture `dmesg` under the failing case and inspect layout near **U1** ↔ header **27/28** (socket vias vs where your jumpers landed).

#### Step 4 — During a failed scan (not at idle)

If you have a scope or logic analyzer, probe **SDA** at **U1 pin 5** while running `i2cdetect`. Stuck-low during clock edges with idle high confirms a **dynamic** contention problem.

Without a scope:

```bash
pinctrl get 0-1    # immediately after a failed i2cdetect
```

#### What this is *not*

- **Not** “Pi GPIO0 vs RP2350 GPIO0” — different chips; RP2350 **GPIO0** is SPI MISO and is not on the Pi **ID_SDA** net.
- **Not** proof that **U1** is unprogrammed — you may have programmed it successfully via jumpers; full-stack detection can still fail independently.
- **`supply vcc not found, using dummy regulator`** on manual `new_device` — harmless; ignore.

---

## 9. Pre-ship checklist

- [ ] `jtsmm_hat_eeprom_vert.txt` / `jtsmm_hat_eeprom_hori.txt` committed with a **fixed** `product_uuid`
- [ ] `eepdump jtsmm_hat_eeprom_vert.eep` reviewed
- [ ] U1 programmed; WP short removed
- [ ] `/proc/device-tree/hat/product` correct on target Pi 5
- [ ] (If M.2 HAT+ stacked) its EEPROM at **0x51** unchanged — N/A when flashing with JTS-MM HAT only
