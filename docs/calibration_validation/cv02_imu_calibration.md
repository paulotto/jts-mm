<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# cv02 — IMU Calibration

This document specifies the IMU calibration protocol that produces the values stored under `/calibration/imu_mta/` and `/calibration/imu_cra/` in the HDF5 schema. Calibration covers four quantities: gyro bias, accel bias + scale, magnetometer hard- and soft-iron, and the mounting transform (from IMU body frame to MTA/CRA anatomical frame).

The BNO085 has its own on-chip **motion-engine (ME) dynamic calibration** that we *also* use (see [§7.1](#71-on-chip-bno085-dynamic-calibration)); the firmware enables it at boot and surfaces its live status. The static + Allan + ellipsoid + mounting-transform calibration here gives us full transparency on top of that and lets the dataset be re-fused offline with any algorithm.

## 1. What gets calibrated

| Quantity | When estimated | Stored where |
|----------|----------------|--------------|
| Gyro bias | Each IMU power-up + once per session | `gyro_bias` (3,) rad/s |
| Accel bias + scale | One-time per IMU | `accel_bias` (3,) m/s², `accel_scale` (3,) |
| Magnetometer hard / soft iron | Per environment | `mag_hard_iron`, `mag_soft_iron` |
| Mounting transform | Once per rigid IMU–array pair (rotation now; `r` later) | `mounting_transform` (4, 4) |
| Allan variance noise model | One-time per IMU type | attrs: `gyro_arw`, `gyro_rrw`, `accel_vrw` |

## 2. Gyro bias (static, fast)

### 2.1 Procedure

1. Place IMU on a stable, vibration-free surface (the optical bench).
2. Let it rest for 60 seconds.
3. Compute the mean gyro reading over the last 30 seconds.
4. Subtract this mean from every subsequent gyro sample as bias.

This is the simplest of the calibrations. The BNO085 does it internally and reports a `GyroCalibrationStatus` field; we also do it ourselves on the Pi side for redundancy.

### 2.2 Acceptance

After bias subtraction, the gyro should read <0.005 rad/s (≈ 0.3°/s) per axis on a static IMU.

## 3. Accel bias + scale (6-position, slow)

The standard 6-position calibration. The IMU is oriented along each of the six principal axes (gravity along ±X, ±Y, ±Z). The expected reading is ±9.81 m/s² on the active axis, 0 on the others.

### 3.1 Procedure

1. Mount the IMU on a small cube (the cube is part of the calibration fixture; PETG-printed).
2. For each of 6 cube faces, place the IMU stationary for 5 s; record the mean accel reading.
3. The 6 readings give an overdetermined system; solve for per-axis bias and scale via least squares:

```
accel_corrected[i] = accel_scale[i] * (accel_raw[i] - accel_bias[i])
```

### 3.2 Acceptance

- After correction, the magnitude of the accel vector (with the IMU stationary in any orientation) is 9.81 ± 0.05 m/s².
- Per-axis scale within 0.99–1.01.

A guided `jtsmm imu calibrate-accel` is planned (not implemented); take the 6 positions manually.

## 4. Magnetometer hard- and soft-iron (optional)

We use the BNO085's game-rotation-vector mode by default, which does **not** use the magnetometer. The magnetometer calibration is only relevant if a future analysis wants to use full 9-DoF fusion (e.g. to recover absolute heading).

### 4.1 Procedure

1. With the IMU mounted in its final location (on the MTA / CRA, since metal nearby affects the magnetic field), rotate it slowly in all directions for 60 s, covering as much of the unit sphere as possible.
2. Plot the (x, y, z) magnetic field; ideally points lie on a sphere of radius |B_earth| (~50 µT in Darmstadt).
3. Fit an ellipsoid to the points using a least-squares ellipsoid fit; the center is the hard-iron offset, the semi-axes define the soft-iron matrix.

### 4.2 Acceptance

- After hard- and soft-iron correction, the corrected magnitude varies by <5 % across orientations.
- The corrected field magnitude is within ±10 % of the local geomagnetic field strength.

### 4.3 When to skip

If using only the game rotation vector (which we recommend), magnetometer calibration is unnecessary. The script will skip it by default and only run with `--include-mag`.

## 5. Allan variance noise characterization (one-time)

The Allan variance gives us the noise parameters needed for proper Kalman-filter fusion of the IMU. We do this once per IMU type (not per build).

### 5.1 Procedure

1. Mount the IMU on a stable, vibration-free surface.
2. Record raw gyro + accel at 400 Hz for at least 1 hour.
3. Compute the Allan deviation curve via `allantools.adev`.
4. Extract:
   - **Angular random walk (ARW)** from the slope-of-−1/2 region (white noise).
   - **Rate random walk (RRW)** from the slope-of-+1/2 region (slow drift).
   - **Velocity random walk (VRW)** for the accelerometer (analogous to ARW).

### 5.2 Acceptance

- ARW: <0.15 °/√h (BNO085 datasheet expectation).
- RRW: <10 °/h (datasheet expectation).
- VRW: <0.05 m/s/√h.

If a unit fails these, it is replaced. The Allan curve is committed to the project repository for the canonical IMUs so labs replicating the system can compare.

## 6. Mounting transform (rotation now; translation later)

This calibration aligns each IMU **chip** frame to the corresponding optical array body (`MP` / MTA, `HP` / CRA). Errors here show up as a constant rotation offset between IMU and optical attitude.

**Implemented now:** rotation `R_mount = body←chip` via Park–Martin hand-eye (`AX = XB`) on **relative** motions. Stored as `/calibration/imu_mta|cra/mounting_transform` (4×4) with **translation zero**.

**Optional later:** lever-arm `r` (chip position in the body frame) from optical origin acceleration vs IMU linear accel + gyro, or from CAD. Not required for mandible-vs-cranium **attitude** (`q_rel`). Classic SE(3) hand-eye is the wrong estimator for `r` because the IMU quaternion has no chip position.

Do **not** fit absolute `R_optical(t)` vs `R(q_imu(t))` in a shared world (Wahba / Kabsch on orientations). Game rotation vector lives in a gravity-aligned, yaw-unreferenced earth frame; optical pose lives in the camera world. Relative motions cancel that unknown world-to-world yaw.

### 6.1 Data

One session HDF5, one array at a time (or both in one take with two SPACE phases):

- `/optical/pose_mta` or `pose_cra` plus `t_master_us` (only **R** is used)
- `/imu_mta/quat` or `/imu_cra/quat` (game RV) plus `t_master_us_quat`
- Valid optical rows (finite SE(3)); tags visible

Gyro, linear accel, mag, and landmarks are **not** required for rotation.

### 6.2 Capture pipeline (figure-eight)

Use `jtsmm imu calibrate --mode record` (not a plain `jtsmm record`). Operator YAML often leaves `session.optical_track` unset, which **defaults to false**: cameras still dump JPEGs, but ArUco never runs and `/optical/pose_*` is missing. Record mode forces optical tracking on, skips `ads_start` (no EMG), ignores `session.ct_calibration` / `session.experiment` / `session.use_experiment_plan`, and loads `software/jtsmm/etc/imu_mounting.json`.

SPACE-bound two motion phases (cameras stay on; landmark digitization is skipped for `imu_mount_*` plans):

1. **`imu_mount_mta`** (~20–30 s). Hold the mouthpiece (IMU already glued). Slow figure-eight tumble about **~3 axes**. Not chewing. Keep MTA tags in view.
2. **`imu_mount_cra`** (~20–30 s). Whole **head + CRA** in space. Nodding only is degenerate.

Then (already run at the end of `--mode record` unless `--no-fit`):

```bash
jtsmm imu calibrate --mode fit SESSION.h5
# or: jtsmm imu calibrate --mode fit SESSION.h5 --imu mta
```

The fit SLERPs IMU quats onto optical times, gates frames (stereo `n_cameras` / `n_tags` / Kabsch-or-PnP RMS from `/optical/quality`, drop orientation glitches, prefer ~10–80°/s tumble), farthest-point samples **40** poses on SO(3), builds relative pairs `A_ij = R_opt(i)^T R_opt(j)`, `B_ij = R_imu(i)^T R_imu(j)`, solves `A X = X B` for `X = R_mount` (Park & Martin 1994), and writes `/calibration/imu_*/mounting_transform`. After the fit, the operator report prints whether optical and IMU relative motions covered **three axes** (SVD $s_2/s_0 \ge 0.28$ on both). This is a recap after the take, not a live print during capture.

With `session.persist_imu_calibration: true` (the default; `--no-persist-imu-calibration` skips it) the take is also copied to `session.imu_calibration_recordings_dir` (default `~/experiments/5_2_2_imu_optical_handeye/recordings/imu_mounting_{N}_{timestamp}/`) as `imu_mounting.json` + `session.h5` + `meta.json`. Extra takes of the **same** rigid pair are the repeatability study (`experiments/5_2_2_imu_optical_handeye`) — not a per-subject cal. Operatively one take after each remount is enough. Eval: `eval_handeye.py` (alias `eval_repeatability.py`).

**Every take carries its mounting.** An accepted fit also updates the stable file `session.imu_mounting_calibration` (default `software/jtsmm/etc/imu_mounting_calibration.json`); every later recording copies it into its own `session.h5` as `/calibration/imu_*/mounting_transform` (attrs `embedded=True`, `source`, `source_session`, calibration `date`, `rms_deg`). Analyses read it with `jtsmm.imu.mounting.load_mounting_rotation`. Takes recorded before had no mounting in the file.

Once per **rigid IMU–array pair**. Redo if the IMU is re-glued, the array is reprinted, or optical vs IMU shows a large constant offset (>5°) at session start. Not per subject. Per-session gyro-bias / rate checks are a different step.

### 6.3 Acceptance

- After applying `R_mount`, the AX=XB geodesic residual is **≤3° RMS** on the calibration take.
- At least **12** valid paired optical frames (`n_samples`) and **8** relative-rotation pairs above 8° (`n_pairs`).
- Relative-motion SVD not collinear ($s_1/s_0 \ge 0.12$). The end-of-fit report flags **3-axis** coverage when $s_2/s_0 \ge 0.28$ on both optical and IMU pairs (informational; not a reject).
- Residual should not grow linearly with time (that is gyro bias / fusion drift, not mount).

### 6.4 CLI

```bash
jtsmm imu calibrate --mode record --persist-imu-calibration
jtsmm imu calibrate --mode fit SESSION.h5 [--imu mta|cra|both]
```

## 7. Per-session calibration

Before each recording session:

```bash
jtsmm doctor imu-rates          # stream rates per slot
# CDC: imu_status                # per-report accuracy bits (§7.1)
```

What to verify (a combined `jtsmm imu check` is planned, not implemented):

- Re-estimates the gyro bias from 5 s of static data and warns if it differs from the calibrated value by >0.005 rad/s.
- Verifies the IMU streams are running at the configured rates with no dropped samples.
- Verifies the BNO085's on-chip calibration status (per-report accuracy bits) are at the expected level (see §7.1).

### 7.1 On-chip (BNO085) dynamic calibration

The BNO085 runs a background **motion-engine (ME) calibration** for the accelerometer, gyroscope, and magnetometer. It is *automatic* — there is no blocking "calibration mode" to enter; accuracy improves as the sensor experiences motion. Every SH-2 input report carries a 2-bit **accuracy / calibration status** in its status byte: `0=unreliable, 1=low, 2=medium, 3=high`.

**What the firmware does (no reflash needed to use):**

1. **Sets the ME calibration on every `config apply` (and after a device reset).** `imu_config_apply_slot()` issues an SH-2 *ME Calibration* command (`0xF2` cmd `0x07`) for each populated slot while the bus is idle, with the per-sensor flags of `mcu.imu_<slot>.dynamic_calibration` (`accel`, `gyro`, `mag`; [s08](../software/s08_runtime_configuration.md)). **Default: all three on**, the behaviour of every recording up to 2026-11-02 (pilot dataset and validation study); before 2026-11-02 the firmware always enabled all three, there was no switch. See §7.2 for when to switch it off.
2. **Surfaces the live accuracy** of each streaming report. The MCU parses the per-report accuracy bits (`bno085.c`) and `imu_status` prints them per slot:

   ```text
   imu_mta: enabled=1 streaming=1 int=0 reports=3
     i2c_hz=400000 (cfg=400000) read_us last=852 max=6162 len=21
     cal quat=high accel=med gyro=high mag=unreliable (0=unreliable..3=high)
   ```

   `quat` is the fusion (game/geomag/absolute rotation vector) accuracy; `accel`/`gyro`/`mag` reflect the raw motion-vector reports. `n/a` means that report is not currently streaming (accuracy is per-report).
3. **Persists calibration on demand** with `imu_savecal`, which issues an SH-2 *Save DCD* command (`0xF2` cmd `0x06`) so the Dynamic Calibration Data survives a power cycle. Save while the slot is **idle** (not streaming) — the sensor core owns the I²C bus during streaming:

   ```text
   imu_stream mta off
   imu_stream cra off
   imu_savecal            # -> "imu_<slot>: save DCD sent"
   ```

**Do we need to run it manually?** No. For jaw tracking we default to the **game rotation vector** (accel+gyro, no magnetometer), whose orientation accuracy reaches `high` within a couple of seconds of a resting sensor — the gyro zero-rate bias converges quickly when still. Verified on hardware (both slots, static): `quat=high` within ~2 s.

**Expected static behavior of the raw sensors** (matches the datasheet, verified on hardware with an accel+gyro+mag report set at rest):

| Sensor | Static accuracy | To reach `high` |
|--------|-----------------|-----------------|
| Gyro   | `high` (fast)   | Rest still for ~1–2 s (zero-rate cal) |
| Accel  | `med`           | Tumble through several orientations (≈ the 6-position pose set in §3) |
| Mag    | `unreliable`    | Slow figure-eight covering the unit sphere (§4) — only needed for 9-DoF/absolute heading |

Because the default fusion is magnetometer-free, `mag=unreliable` at rest is expected and harmless. If you switch to the absolute rotation vector (9-DoF), perform the figure-eight motion until `mag` reaches at least `med`, then `imu_savecal`.

### 7.2 Dynamic calibration on or off during takes

**Default and the current dataset: on.** `etc/jtsmm.yaml` keeps all three flags `true`. Recordings
that **add participants to the existing pilot dataset** (or repeat validation takes) keep this
setting, so the whole dataset is recorded the same way. The audit of its effect on those
recordings is in [an04 §1](../analysis/an04_imu_processing.md) (no stillness-aligned step above
about 0.3–0.5°).

**New studies: off during takes, with a saved calibration.** The manufacturer recommends
disabling gyroscope and magnetometer dynamic calibration during application use, because new
calibrations created mid-use can cause motion artefacts; the gyroscope zero-rate offset is still
corrected automatically whenever the sensor is stable. The accelerometer calibration needs
4–6 held orientations, which jaw and head movements rarely provide, so it is also better fixed
during a take (the analysis estimates the accelerometer bias itself: position filter, lever fit).
Pros of leaving it on: no operator step can be forgotten, and slow temperature-driven bias
changes are followed. Cons: silent updates mid-take (steps or drift-rate changes the per-take
analysis cannot see).

Procedure with dynamic calibration **off** (once per IMU and session day, and whenever `imu_status`
shows degraded accuracy):
1. Apply a config with all three flags `true` (the default) and stream the reports **through a
   host program that reads them**, i.e. `jtsmm view` (it applies the config and drains the binary
   USB endpoint). Do not start the streams from a serial terminal (`imu_stream … on` in picocom):
   nobody reads the binary endpoint, the IMU ring fills within seconds, and the firmware's IMU
   watchdog reads the stopped ring pushes as a stalled sensor ("IMU WATCHDOG: mta stalled 3s,
   recovery attempt 1/6") and resets healthy IMUs (firmware 1.1.0 reports
   "ring full -- the host is not reading the data endpoint" instead and leaves the IMUs alone).
2. Rest the IMU still for 2–3 s (gyroscope), then hold it in 4–6 distinct orientations for a
   few seconds each (accelerometer, ≈ the poses of §3); magnetometer figure-eight only if a
   magnetometer-based fusion is used (§4).
3. Stop the viewer (Ctrl-C; streams off), then in a serial terminal `imu_status` (the last
   accuracy per slot: quat should be `high`, accel at least `med`) and `imu_savecal` (Save DCD;
   survives power cycles). The accel accuracy comes only from the raw `accelerometer` report,
   which the template (`kinematics_accel`, the profile of every recorded take) enables; a config
   with `linear_acceleration` instead (`kinematics_default`) reads `n/a`. In a repository
   checkout every `jtsmm` command without `--config` loads the template
   `software/jtsmm/etc/jtsmm.yaml`, not `~/jtsmm.yaml`; make the off copy of step 4 from the
   template (a copy of an old, unused `~/jtsmm.yaml` with `kinematics_default`
   showed accel `n/a` and would have recorded without the raw accelerometer).
4. Set `mcu.imu_mta|imu_cra.dynamic_calibration: {accel: false, gyro: false, mag: false}` for the
   takes; every take then stores the setting in `/meta/mcu_config_json`.

**Analysis pipeline.** No IMU calibration step is needed in the analysis in either mode: the
mounting transform (§6, embedded in each take) and the per-take fits (an04) are unchanged.
Takes recorded before 2026-11-02 have no `dynamic_calibration` key in `/meta/mcu_config_json`;
they ran with all three on.

*Firmware state:* the switch is implemented and host-tested (firmware unit tests, config schema
and golden payload), flashed and **hardware-checked** with the procedure
above: after the calibration movements with the switch on, `imu_status` quat / accel MTA high /
high, CRA high / high; `imu_savecal`; off config applied (`config dump`: all three flags `false`
on both slots); after a power cycle with the switch off MTA high / high, CRA high / med, i.e. the
saved calibration is loaded. It may be used for new studies; additions to the current dataset
keep it on.

## 8. Calibration file structure

The calibration HDF5 has the same group structure as the session HDF5, restricted to the `/calibration/` subtree:

```
calibration_imu_<build>_<YYYY-MM-DD>.h5
├── /calibration/imu_mta/
│   ├── gyro_bias            (3,)
│   ├── accel_bias           (3,)
│   ├── accel_scale          (3,)
│   ├── mag_hard_iron        (3,)  (optional)
│   ├── mag_soft_iron        (3, 3) (optional)
│   ├── mounting_transform   (4, 4)
│   ├── allan/
│   │   ├── tau_s            (T,)
│   │   ├── adev_gyro        (3, T)
│   │   └── adev_accel       (3, T)
│   └── attrs:
│       ├── device            "BNO085"
│       ├── serial            ...
│       ├── arw_dps_sqrt_h    0.13
│       ├── rrw_dps_per_h     6.0
│       ├── vrw_mps_sqrt_h    0.04
│       ├── n_static_seconds  300
│       └── date              ISO 8601
└── /calibration/imu_cra/
    ... (same structure)
```

## 9. Recalibration triggers

| Trigger | What to re-calibrate |
|---------|----------------------|
| IMU re-glued | Mounting transform |
| MTA or CRA modified (geometry changed) | Mounting transform |
| New IMU unit (different chip) | All steps |
| Operator notices >5° offset between IMU and optical pose at session start | Mounting transform, gyro bias |
| Magnetic environment changed (e.g. moved to a different lab) | Magnetometer (only if magnetometer is used) |
| Dynamic calibration off (§7.2): new session day, or `imu_status` accuracy below `high` (quat) / `med` (accel) | On-chip calibration + `imu_savecal` (§7.2) |

A `jtsmm imu status` command reporting the calibration's age is planned (not implemented).

## 10. References

- Bosch Sensortec BNO085 datasheet and BNO085 Application Note.
- Tedaldi D., Pretto A., Menegatti E. — *"A robust and easy to implement method for IMU calibration without external equipments"*, ICRA 2014.
- IEEE Std 952-1997 — *"IEEE Standard Specification Format Guide and Test Procedure for Single-Axis Interferometric Fiber Optic Gyros"* (Allan variance methodology).
- Markley F. L. — *"Attitude determination using vector observations: a fast optimal matrix algorithm"*, J. Astronautical Sciences 36(3) 1988 (Wahba / Davenport on a **shared** world — not used for mounting; game-RV and optical worlds differ).
- Park F. C., Martin B. J. — *"Robot Sensor Calibration: Solving AX = XB on the Euclidean Group"*, IEEE Trans. Robotics and Automation 10(5) 1994.
