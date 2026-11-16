<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# h01 — Camera Subsystem

The camera subsystem replaces the Qualisys Oqus rig used in the IEEE SENSORS 2025 paper. We document two tracks because they correspond to two different price–performance points and to two distinct lab realities.

**Default optical markers:** **AprilTag 36h11** (OpenCV `DICT_APRILTAG_36h11`) printed on matte black PETG pads on the MTA and CRA. Retroreflective spheres and IR illumination are **optional** (Qualisys comparison only); see [`../04_component_catalog.md`](../04_component_catalog.md#9-fiducials--scene-lighting).

## 1. Requirements

The first-generation JTS records at 200 Hz with ~182 µm precision. We do not aim to match that; we aim to be **good enough for jaw exoskeleton intent recognition** while staying under 1 kEUR for the full system.

| # | Requirement | Track A (webcams) | Track B (Pi GS CSI) |
|---|-------------|-------------------|---------------------|
| R1 | 2–3 cameras with overlapping FOV covering ±200 mm around the head | yes | yes |
| R2 | Frame rate ≥30 fps | 30–60 fps | 60–120 fps |
| R3 | Tracking precision better than ±2 mm RMS | target 1–2 mm | target 0.5–1.5 mm (AprilTag, good lighting) |
| R4 | Hardware-trigger sync option | no | yes |
| R5 | Open-source software stack | OpenCV / V4L2 | libcamera / picamera2 / OpenCV |
| R6 | Total optical BOM ≤ 500 EUR | ~120 EUR | ~260 EUR |
| R7 | Tracking compatible with the existing MTA + CRA arrays | yes (AprilTag pads on MTA/CRA) | yes (same fiducials) |

## 2. Track A — USB webcams (cheapest path)

### 2.1 Hardware

- **2× Logitech C920 / C920s** (rolling shutter, 1920×1080 at 30 fps; or 1280×720 at 60 fps).
  - Pros: ubiquitous, cheap, V4L2-friendly, autofocus disable supported via UVC extension units.
  - Cons: rolling shutter (skew on fast motion), no hardware trigger, vendor-specific PTS quality.
- Alternative: **Arducam B0298** (OV9281 USB, global shutter, monochrome, ~100 fps). Pricier (~80 EUR) but global shutter; still no HW trigger over USB.

### 2.2 Optical setup (AprilTag)

- Disable autofocus, auto-exposure, auto-white-balance, autogain via `v4l2-ctl`.
- Lock exposure to freeze motion (e.g. 2–8 ms) and set gain so AprilTag borders are sharp (not blown out).
- Use **diffuse visible lighting** (LED panel or softboxes); avoid windows behind the subject and specular glare on tags.
- **3× AprilTag 36h11** per array (MTA + CRA), 15–25 mm tag size, high contrast (black tag on matte white or white tag on matte black PETG — pick one scheme and stay consistent).

### 2.3 Synchronization

There is no hardware trigger on Track A. We use a **software-sync protocol** described in detail in [`h06_synchronization.md`](h06_synchronization.md):

1. The HAT drives a **visible sync LED** in the shared camera FOV (not an IR ring).
2. The LED is pulsed at known `t_master_us` instants (every 1 s, short pulse).
3. The capture process detects the LED-on event in each camera frame and pairs `t_master_us` with the camera's UVC presentation timestamp (PTS).
4. The `clock_align` service fits an affine map `t_master = a · t_pts + b` per camera, updated continuously.

Achievable alignment is ~1–3 ms depending on UVC PTS quality.

### 2.4 Calibration

- Intrinsic: standard ChArUco workflow with OpenCV (`cv2.aruco.calibrateCameraCharuco`).
- Extrinsic: place a ChArUco board so all cameras can see it at once; solve `solvePnP` per camera.

See [`../calibration_validation/cv01_camera_calibration.md`](../calibration_validation/cv01_camera_calibration.md).

### 2.5 Marker detection

- **Primary:** `cv2.aruco.ArucoDetector` with the dictionary from `software/optical/etc/optical.yaml` (currently `DICT_4X4_50`; AprilTag 36h11 is the design target and only needs a dictionary change).
- Per frame: detect tags → `solvePnP` per tag → fuse to MTA/CRA rigid pose (quaternion average + outlier rejection).
- Minimum **~30 px** per tag side at runtime; verify in the `jtsmm view` camera preview before recording.

## 3. Track B — Pi Global Shutter CSI cameras (recommended)

### 3.1 Hardware

- **2× IMX296 global-shutter CSI modules** (1456×1088, 60 fps full resolution), native Pi 5 CSI-2.
  - **Default BOM:** [Raspberry Pi Global Shutter Camera](https://www.raspberrypi.com/products/global-shutter-camera/) (official module).
  - **Alternates:** Waveshare IMX296 Global Camera (A), InnoMaker IMX296 MIPI module — same sensor and **libcamera `imx296` driver**; see [§3.1.1](#311-imx296-trigger-and-sync-pins) for the **XTR+ / XTR−** header (not a single “XTR” pin).
  - **Visible-spectrum** C/CS or M12 lens (4–6 mm); do **not** use 850 nm IR-pass lenses with AprilTag.
- **Optional 3rd camera:** Arducam UC-599 (OV9281 USB3, global shutter, hardware trigger via HAT opto).

#### 3.1.1 IMX296 trigger and sync pins

Sony **IMX296** modules expose several **hardware** pads. Names vary by vendor; they wire the same sensor signals.

| Label (typical) | Role | Used in JTS-MM? |
|-----------------|------|-----------------|
| **XTR+** / **Trig+** / touchpad **XTR** | External trigger input (**XTRIG**, 1.8 V logic) | **Yes** — HAT `TRIG_CSI1/2` → **XTR+** |
| **XTR−** / **Trig−** / **GND** next to trigger | Ground reference for the trigger input | **Yes** — tie to **camera/module GND** (common with HAT **DGND**) |
| **XVS** | Vertical sync (master/slave camera chaining) | **No** — not used; we use one MCU fan-out to both **XTR+** pins |
| **XHS** | Horizontal sync (used with XVS in sink mode) | **No** |
| **MAS** | Master/slave strap (hardware mode select at boot) | **No** — leave open / factory default; do **not** bridge for our stack |

There is **not** a differential trigger pair in the sense of two driven signals: **XTR− is ground**, not a second trigger input. Use a **2-conductor** cable: **signal → XTR+**, **return → XTR−**.

**Fast trigger mode (what we use):** per [Raspberry Pi external-trigger documentation](https://www.raspberrypi.com/documentation/accessories/global-shutter-camera.html), integration is controlled by the **low time** on **XTRIG** (exposure ≈ low pulse width + ~14.26 µs). Frame rate follows how often you pulse. Enable in software with `echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode` and fixed gain/exposure in picamera2 (AGC must not fight the external pulse).

**Electrical notes:**

- **XTRIG is 1.8 V** — do not drive it from 3.3 V without a divider or level shifter (Pi doc: 1.5 kΩ series + 1.8 kΩ from **XTR+** to **GND** when sourcing from 3.3 V).
- **Official Pi GS board:** may require removing **R11** (and checking **Q2**) before external trigger works — see Pi doc “Preparation”. Third-party boards often ship with **Trig+/Trig−** headers already broken out.
- **HAT output:** 3.3 V via **74LVC2G241**, then on-HAT **1.5 kΩ / 1.8 kΩ** divider per **`TRIG_CSI`** → **~1.81 V** at **J1/J2** (Pi GS doc; **no** CSI **100 Ω** — see [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md) Step 2b).

**Module choice (short):**

| Module | Trigger header | Driver / docs | Notes |
|--------|----------------|-----------------|-------|
| **Raspberry Pi Global Shutter Camera** | Solder **XTR** + **GND** touchpads | Best documented; `imx296` in Pi OS | **Recommended default** |
| **Waveshare IMX296 Global Camera (A)** | **XTR+**, **XTR−**, XVS, XHS, MAS | Pi-compatible CSI; same sensor | Good if you want pre-populated headers; ignore unused sync pins |
| **InnoMaker IMX296 (Amazon etc.)** | **Trig+**, **Trig−** (same as XTR+/−) | Claims Pi OS / external trigger | Verify board revision, ribbon (15→22 pin), and GPL/support before bulk buy |

Do **not** use IMX296 **XVS/XHS master–slave** wiring unless you deliberately abandon the MCU trigger path — the Pi **imx296** stack is built around **external trigger on XTRIG**, not multi-camera XVS chaining.

### 3.2 Mounting and field of view

- Two tripods, ~60 cm apart, ~80 cm from the subject; baseline ~30°.
- Wall-mounted ChArUco board for calibration and optional extrinsic drift checks.
- Same **AprilTag pads** on MTA/CRA as Track A (identical software pipeline).

### 3.3 Synchronization

MCU hardware trigger → camera **XTR+** (XTRIG) → `SensorTimestamp` → `t_master_us` (see [`h06_synchronization.md`](h06_synchronization.md)). Target jitter &lt;100 µs.

**Trigger rate.** The operator template [`software/jtsmm/etc/jtsmm.yaml`](../../software/jtsmm/etc/jtsmm.yaml) triggers both cameras at **40 Hz** with a 12 ms low pulse (≈ 12 ms exposure, 25 ms period). The Oct 2026 pilot study was recorded this way; keep it for study sessions so all participants share one rate. `mcu.trigger.rate_hz` must equal every `host.cameras.list[].trigger.expect_rate_hz` (`jtsmm config validate`). Two other rates are deliberate and separate:

- The `imx296_dual_csi` profile preset and the firmware boot default (`trig_start` with no arguments) are **30 Hz** / 12 ms.
- `optical calibrate` pulses at **30 Hz** regardless of `jtsmm.yaml` (`CALIBRATE_TRIGGER_RATE_HZ`): full-resolution copies of both cameras overrun shorter periods, and a still board does not need the study rate.

The optional visible **sync LED** from the HAT is **not required** for Track B timestamping — use **`TRIGGER_EVENT`** + **`SensorTimestamp`** only.

### 3.4 Calibration

ChArUco intrinsics + multi-camera extrinsics (same as Track A). Global shutter improves PnP stability vs rolling-shutter webcams.

### 3.5 Marker detection

- **Primary:** AprilTag 36h11 on MTA/CRA (same as Track A).
- Multi-tag fusion → rigid-body pose → existing JTS kinematic pipeline unchanged downstream.

### 3.6 Triangulation

- If multiple tags are visible per array per camera, use tag corner 3D points from known tag geometry on the CAD model, or triangulate tag origins across cameras then Kabsch to the array frame.
- Software default: `marker_detector: apriltag` in `~/jtsmm.yaml` (a label; markers are configured in `software/optical/etc/optical.yaml`) ([`../software/s03_camera_pipeline.md`](../software/s03_camera_pipeline.md)).

## 4. Bandwidth and Pi 5 CSI feasibility

Two IMX296 @ 60 fps fit Pi 5 CSI bandwidth; encode to FFV1/H.264 on NVMe. AprilTag detection runs on **downscaled** previews or full frames depending on CPU budget (~5–10 ms per camera per frame).

## 5. Camera-trigger electrical interface (HAT)

The HAT provides camera triggers only (no IR ring power):

- `TRIG_CSI1` / `TRIG_CSI2` → **2-wire** each: **signal → XTR+**, **GND → XTR−** (see §3.1.1).
- `TRIG_USB` → optional USB3 GS camera (opto-isolated).
- `SYNC_LED` JST → **optional** bench visible LED for Track A only (**DNP** on default GS build).

See [`h02_custom_hat_shield.md`](h02_custom_hat_shield.md) §7 and [`h06_trigger_io_schematic_build_guide.md`](h06_trigger_io_schematic_build_guide.md).

## 6. Alternatives matrix (for replicators)

| Camera | Shutter | Trigger | Cost (EUR) | Markers | Notes |
|--------|---------|---------|-----------|---------|-------|
| Logitech C920 | rolling | software + sync LED | ~55 | AprilTag | Cheapest |
| Arducam B0298 USB GS | global | software + sync LED | ~80 | AprilTag | Better motion freeze |
| Pi GS Camera (IMX296) | global | XTR+ / XTR− | ~55 | AprilTag | **Recommended** |
| Waveshare IMX296 Global (A) | global | XTR+ / XTR− (+ unused XVS/XHS/MAS) | ~50–70 | AprilTag | Pre-broken-out headers |
| InnoMaker IMX296 MIPI | global | Trig+ / Trig− | ~40–60 | AprilTag | Verify driver + cable kit |
| Arducam UC-599 USB3 | global | HW TRIG (opto) | ~95 | AprilTag | 3rd view |

**Retroreflective + IR rings:** optional for Qualisys side-by-side only; not the default software path.

## 7. Cabling and physical install

- CSI ribbons ≤30 cm.
- Trigger fan-out: plain **2-wire** JST from HAT **J1/J2** to each camera **XTR+ / XTR−** (~**1.8 V** from on-HAT **R51/R53 + R57/R59**).
- *(If sync LED populated)* Cable from HAT `SYNC_LED` JST to a small white LED mounted in FOV (gaffer-tape to tripod or light stand).

## 8. Lens choice

Visible 6 mm M12 on Pi GS at ~80 cm → ~46° HFOV. Verify tag pixel size at that distance before gluing tags.

## 9. Future work

- Learned fiducial detector for motion blur edge cases.
- Optional retroreflective mode in software (blob detector) without changing HAT hardware.
- Bundle adjustment on recorded sessions to refine extrinsics post-hoc.
