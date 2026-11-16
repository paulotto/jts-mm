<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# s05 — IMU Processing

This document specifies the IMU data path on the Pi 5: quaternion ingestion, calibration application, mounting-frame transformation, relative mandible-vs-cranium quaternion computation, drift correction against the optical pose, and the IMU-bridging mode that fills optical drop-outs.

## 1. Inputs

Per IMU (MTA + CRA), the enabled BNO085 outputs are **freely combinable at
independent rates** via `mcu.imu_mta.reports` / `imu_cra.reports` (the combinable
report model and the seven datasheet outputs are specified in
[`s08_runtime_configuration.md`](s08_runtime_configuration.md)). What is emitted
depends on the applied config/preset; there is no fixed four-stream set.

The default operator preset is **`kinematics_accel`** (fusion quaternion +
the gravity-included accelerometer; gravity is removed offline with the optical
orientation, §3.1). `kinematics_default` records the BNO085's own
`linear_accel` instead. Other streams (`gyro`, `mag`) are available via other
presets or `custom`:

| Stream | Report (`reports` key) | Default rate | Wire (legacy 17/7/12 B without `acc` accepted) | Host units |
|--------|------------------------|--------------|------|------------|
| `quat` | `game_rotation_vector` | 390 Hz¹ | tagged `QUAT` 18 B: `[report_id:u8][w,x,y,z:f32][acc:u8]` | quaternion (w, x, y, z) |
| `accel` | `accelerometer` | 250 Hz | tagged `RAW` 8 B: `[sensor_id:u8][x,y,z:int16][acc:u8]` (`sensor_id=0x01`) | m/s² (Q8), gravity included |
| `linear_accel` | `linear_acceleration` | 250 Hz² | `LIN_ACCEL` 13 B: `[x,y,z:f32][acc:u8]` | m/s² |
| `gyro` | `gyroscope` | 400 Hz² | tagged `RAW` 8 B (`sensor_id=0x02`) | rad/s (Q9) |
| `mag` | `magnetometer` | 100 Hz² | tagged `RAW` 8 B (`sensor_id=0x03`) | µT (Q4) |

² Not in `kinematics_accel`; `linear_accel` is in `kinematics_default`, the
others via `raw_only` / `datasheet_motion` / `custom`. The two IMUs sit on
**separate** I2C buses (MTA=i2c0, CRA=i2c1). The BNO085 hub interleaves the
fusion report and the 3-axis vector: measured with linear_accel **~350 Hz game_rv
+ ~220 Hz linear_accel** (the accelerometer takes the same 250 Hz slot; its own
rate there is not yet measured, run `jtsmm doctor imu-rates --preset
kinematics_accel`). The vector is requested at 250 (not 200) because 200 Hz / 5 ms
aligns badly with the fusion schedule and collapses to ~120 Hz, whereas 250 Hz /
4 ms quantizes to ~220 Hz. Adding `gyro@400` is worse
still — the hub then throttles all three (~331/331/79 Hz). Full per-stream rates
are only reached one report at a time.

¹ The BNO085 game rotation vector runs on an internal ~400 Hz (2.5 ms) schedule;
requesting 390 Hz quantizes up to the sensor's nearest slot, so the **measured**
rate is ~398–399 Hz. 390 preserves the integrity drop budget; the +2–3% overshoot
is the sensor delivering its nearest achievable rate.

**Units on the wire:** the accelerometer, gyroscope, and magnetometer are
transported as **int16 Q-point counts** (accel Q8 → m/s², gyro Q9 → rad/s, mag
Q4 → µT; see [`jtsmm/imu/units.py`](../../software/jtsmm/jtsmm/imu/units.py)),
**not** float32. Only `linear_acceleration` (and the fusion quaternions) are
float32 on the wire. The host converts RAW counts to SI at decode/persist time.

**Storage:** each enabled report is stored in its own HDF5 dataset with its own
timestamp vector, so independent rates fall out naturally:

```
/imu_mta/
  quat (N,4) f32  + t_master_us_quat  (+ quat_report_id, fusion_mode attrs)
  accel (Na,3)    + t_master_us_accel   (attrs: units=m/s^2, rate_hz, device=BNO085)
  gyro  (Ng,3)    + t_master_us_gyro    (units=rad/s)
  mag   (Nm,3)    + t_master_us_mag     (units=uT)
  linear_accel (Nl,3) + t_master_us_linear_accel (units=m/s^2)
  quat_accuracy, accel_accuracy, ... (N,) u8, one per stream, row-aligned
/imu_cra/ (same, for reports enabled on that slot)
```

`<stream>_accuracy` is the BNO085 calibration status of each sample (SH-2
status bits 1:0): 0 unreliable, 1 low, 2 medium, 3 high; 255 = the record
carried no status byte (older firmware). Reader: `Vec3Stream.accuracy`,
`ImuKinematicsStream.quat_accuracy`. Use it to drop or down-weight samples
from before the sensor settled, e.g. `accel_accuracy < 2`.

**What `t_master_us` of an IMU sample means:** the BNO085's own estimate of
when that sample was taken, on the MCU master clock. The firmware latches the
INT falling edge (t_int) and subtracts the SH-2 base delta / adds the report's
delay field, so each report of a batched packet has its own time, ≤ t_int. It
is *not* the time the MCU read the packet (that was the case before
2026-09; such data is late by the chip's sample→INT delay plus the poll
latency, with up to ~1 ms jitter). Exceptions: `emg_drdy_divisor` slots keep
the DRDY-paced read time, and the channel-5 gyro-integrated RV has no delay
field (t_int). Details and fallbacks: [`h04`](../hardware/h04_imu_subsystem.md) §6.2.

The analysis code resamples across streams as needed (`numpy.interp`, or SLERP
for the quaternion). See [`session_reader.ImuKinematicsStream`](../../software/jtsmm/jtsmm/io/session_reader.py)
for the reader surface.

> **Implementation status:** tagged wire format, host decode, HDF5
> persistence, session reader, viewer, `jtsmm doctor imu-rates`, and **rotation
> mounting** (`jtsmm imu calibrate --mode record` / `--mode fit` + Park–Martin) are implemented.
> Lever-arm translation, optical drift correction (§5), and IMU bridging (§6)
> remain future work **on the Pi (live)**.
>
> **Offline, validation study:**
> - `jtsmm/imu/fusion.py` fuses the orientation: F1 complementary filter, F2 error-state
>   Kalman filter with smoother. Its calibration (`Z`, `X`, drift, lag) is fitted against
>   the optical data; `X_fixed` uses the hand-eye mounting.
> - `jtsmm/imu/position.py` bridges the **position** through optical dropouts. It uses the
>   raw `accelerometer` report (recorded; `linear_acceleration` is off in the presets),
>   with gravity removed with the chip's own orientation; Kalman filter on position,
>   velocity and bias, RTS smoother.
> - These differ from §5–6 below: there is no ZUPT, the bias is a filter state, and both
>   gap ends constrain the bridge.
> - Rules and results: `docs/analysis/an14_system_validation.md` §9.

## 2. Calibration application

Each IMU has a calibration HDF5 group (see [`../calibration_validation/cv02_imu_calibration.md`](../calibration_validation/cv02_imu_calibration.md) for the procedure to generate it):

```
/calibration/imu_mta/
├── gyro_bias            (3,) rad/s              constant subtracted from gyro
├── accel_bias           (3,) m/s²               constant subtracted from accel
├── accel_scale          (3,) dimensionless      multiplied (after bias removal)
├── mag_hard_iron        (3,) µT                 subtracted from mag (optional)
├── mag_soft_iron        (3, 3)                  matrix multiplied (optional)
└── mounting_transform   (4, 4) float64          rigid transform from IMU body frame to MTA anatomical frame
```

These are applied at the **boundary** of the IMU processing module so downstream code always sees data in the array's anatomical frame, not the chip's body frame.

## 3. Mounting transform

The IMU chip is glued to a pad that is not the optical body origin. `mounting_transform` is the rigid map **chip → array body**.

**Now:** rotation `R_mount = body←chip` so `v_body = R_mount v_chip`; with the fusion output `q_chip = world←chip`, the body orientation is `q_body = world←body = q_chip ⊗ conj(q(R_mount))` (`R_body = R_chip R_mountᵀ`). Linear acceleration stays a chip-frame vector: `a_world = R_chip a_chip`. Estimated with Park–Martin `AX = XB` on relative optical vs IMU orientations after a quality/rate gate and SO(3) farthest-point sample of 40 poses. After the fit the operator report prints 3-axis coverage ($s_2/s_0$) — not during capture. Translation of the 4×4 is **zero**. Capture and CLI: [`cv02`](../calibration_validation/cv02_imu_calibration.md) §6 (`jtsmm imu calibrate --mode record`: `imu_mount_mta` / `imu_mount_cra` figure-eights with ArUco on and EMG off).

**Later (optional):** lever-arm `r` so origin-frame linear acceleration can be transported from the chip (`a_imu = a_origin + α × r + ω × (ω × r)`). Needed for origin translation / ZUPT bridging, not for `q_rel`. Do not fold `r` into a joint 6-DoF hand-eye on IMU “poses”.

After `R_mount`, IMU quaternions are in the MTA / CRA body frames used by §4.

### 3.1 Gravity removal with the optical orientation (`kinematics_accel`)

The accelerometer measures specific force `f_S = R_SW (a_W + g û)` in the chip
frame S (û = world "up"). The BNO085's `linear_accel` subtracts gravity with its
own tilt estimate, so its error is `g·sin(tilt error)`: 2.5° (datasheet, dynamic)
leaves ~0.43 m/s². With the optical marker orientation `R_WM` (world ← marker)
and the mounting `R_MS` (marker ← chip, §3):

```
a_W(t) = R_WM(t) R_MS (f_S(t) − b) − g û_W
```

1. Interpolate `R_WM` onto the accel timestamps by SLERP; take the optical
   time as mid-exposure (trigger edge + exposure/2, i.e. +6 ms at 12 ms).
2. Estimate `û_W` (gravity direction in the optical world) and the accel bias
   `b` from **all** samples: jaw motion is bounded, so the mean of `a_W` over a
   window is `Δv / T` ≈ 0 (0.1 m/s over 60 s → 0.002 m/s²), i.e.
   `mean(R_WM R_MS (f_S − b)) ≈ g û_W`. Use sliding windows (10–30 s) so slow
   head drift does not bias it. Reversal points are *not* still samples: there
   the acceleration peaks (`−ω² x`, ≈ 0.8 m/s² at ±20 mm and 1 Hz).
3. Drop samples with `accel_accuracy < 2` or without a valid optical pose.

**Recording:** normal takes (opening/closing, protrusion/retrusion, left/right,
cyclic) are enough. The part of `b` that they cannot separate from gravity (the
head barely rotates, the mandible mainly pitches about one axis) also barely
matters: a bias only causes an error through rotation, ≈ `|b| · 2 sin(Δθ/2)`
for an orientation range Δθ, about 0.4 |b| for 25° of mandible rotation and ~0
for the head; the constant part is removed with the mean. What matters more:
mid-exposure optical timing (a 6 ms offset at 2 rad/s leaks ≈ 0.12 m/s²) and
`R_MS` from the `imu_mount_*` calibration recordings (cv02 §6). Optional
cross-check: a few seconds still in two tilted head poses make `b` fully
observable.

## 4. Relative mandible-vs-cranium quaternion

The MTA-IMU quaternion `q_mta` and the CRA-IMU quaternion `q_cra` are in the MTA and CRA anatomical frames respectively. The relative quaternion expressing the mandible's orientation in the cranial frame is:

```
q_rel = q_cra^(-1) ⊗ q_mta
```

(where `⊗` is quaternion multiplication and `^(-1)` is the conjugate for unit quaternions).

This is the IMU-derived analog of the optical mandible-vs-cranium pose computed in §II.B of the IEEE SENSORS 2025 paper. The two should agree (modulo a constant offset that captures the relative IMU mounting on MTA vs. CRA) at every time step.

The relative-quaternion computation runs on every `quat` sample (game RV ~390 Hz configured), time-aligned to the most recent quaternion from each IMU. The live viewer computes the same `q_rel` per frame ([`jtsmm/imu/kinematics.py`](../../software/jtsmm/jtsmm/imu/kinematics.py)); the offline pipeline output goes into:

```
/derived/relative_orientation/
├── t_master_us          (T,) uint64
├── q_rel                (T, 4) float32 (w, x, y, z)
└── attrs: imu_only ("true"); see /derived/relative_orientation_optical/ for the optical equivalent
```

## 5. Drift correction against optical pose

The BNO085 game-rotation-vector drifts in yaw at ~1°/min. Over a 10-minute session, that is ~10° of integrated drift if uncorrected.

We correct using the optical signal as the ground truth:

1. Every time the optical pose is valid (no occlusion), we compute the optical-derived `q_rel_optical` and the IMU-derived `q_rel_imu`.
2. The residual rotation `q_drift = q_rel_optical ⊗ q_rel_imu^(-1)` is the IMU's drift error at that time.
3. A low-pass filter (Kalman-like: complementary filter with time constant 10 s) tracks `q_drift` over time.
4. The drift-corrected IMU quaternion is `q_imu_corrected = q_drift ⊗ q_imu`.

This is a one-line implementation of complementary filtering on SO(3). It works because optical and IMU errors are uncorrelated: optical has zero long-term drift (it is geometric) but suffers short occlusions; IMU has continuous output but slow drift. Their fusion gives the best of both worlds.

## 6. IMU-bridging mode (optical drop-out)

When the optical pose is invalid (marker occlusion), the IMU is the sole source of mandibular orientation. The pipeline emits a "bridged" pose:

```python
if optical_valid(t):
    pose_out[t] = pose_optical[t]
else:
    pose_out[t] = q_imu_corrected[t]  # using the latest drift estimate
```

The bridging mode also handles **translation**: the IMU does not measure absolute position, only orientation + linear acceleration. We integrate the linear acceleration twice with a zero-velocity update (ZUPT) on long stationary periods to bound translational drift; for jaw motion the integration window is typically <500 ms (until the next optical re-acquisition) which keeps translation error <2 mm.

A flag `pose_source` in the output HDF5 records whether each frame came from optical, IMU-bridged, or both:

```
/derived/pose/
├── t_master_us         (T,) uint64
├── pose_4x4            (T, 4, 4) float32
└── pose_source         (T,) uint8 (0 = optical, 1 = imu_bridged, 2 = fused)
```

## 7. Output to existing JTS pipeline

The existing JTS pipeline reads `/optical/pose_mta` and `/optical/pose_cra` directly. JTS-MM writes those datasets from the **camera triangulation step** (described in [`s03_camera_pipeline.md`](s03_camera_pipeline.md)) **and** writes the bridged version to `/derived/pose/` (described above).

This separation lets the existing JTS code keep working unchanged while exposing the JTS-MM fused pose to anyone who wants it.

## 8. Online vs. offline

The IMU pipeline is naturally online: every operation in §§2–6 is causal and stateless apart from the drift filter (which is a first-order LP).

The complete online pipeline runs on the Pi 5 in <0.5 ms per quaternion sample at 200 Hz (vectorized NumPy), so it can run inside the `clock_align` process without introducing latency.

## 9. Validation acceptance criteria

The IMU pipeline is accepted when:

- [ ] Static MTA + CRA on a bench: relative quaternion noise <0.2° RMS in pitch/roll, <1°/min in yaw.
- [ ] During controlled jaw opening (30 mm amplitude): mandibular pitch from IMU and from optical agree within 2° RMS.
- [ ] During a 60 s optical drop-out (simulated by covering the markers), IMU-bridged pose drifts <5° from the truth (re-acquired at 60 s).
- [ ] Drift filter time constant tunable: at 10 s it should not over-correct rapid intentional rotations.

## 10. ROS 2 mapping (optional)

For the optional ROS 2 bridge, each IMU stream maps to a standard `sensor_msgs/Imu` message with these conventions:

| ROS field | Source |
|-----------|--------|
| `header.stamp` | converted from `t_master_us` via `pi_to_master` |
| `header.frame_id` | `"mta"` or `"cra"` |
| `orientation` | `quat` (game RV) or `q_imu_corrected` after drift correction |
| `angular_velocity` | `gyro` rad/s |
| `linear_acceleration` | `accel` (with gravity) — note: the BNO085 also gives `linear_acceleration` (gravity-removed), but the `sensor_msgs/Imu` convention is **with** gravity |
| covariance fields | computed from `cv02` Allan variance |

## 11. Failure modes

| Symptom | Likely cause | Recovery |
|---------|--------------|----------|
| Quaternion frozen | BNO085 SHTP transport hang | MCU firmware automatically resets; if persistent, replace breakout |
| Quaternion jumps 180° | Sign ambiguity in game RV; rare but possible at certain orientations | Quaternion continuity check in `imu_processing`: enforce `dot(q[k], q[k-1]) >= 0`; flip sign if negative |
| Drift not corrected | Optical pose continuously invalid (long occlusion) | Bridging mode emits IMU-only pose; analysis must accept higher drift |
| Mounting transform wrong | IMU re-glued without re-calibration | Re-run §3; rotational error appears as a constant rotation offset between IMU and optical |
| Magnetic disturbance | Metal object close to IMU | Game RV is robust; if using full rotation vector, switch to game RV |
