<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# cv05 — Reference Validation Protocol (Qualisys Oqus + Delsys Trigno + Step Ladder)

This is the **executable protocol** for validating JTS-MM against commercial reference instruments. It defines how to set up, synchronize, and run the session, and **exactly what to record** so that every metric in the IEEE Sensors Journal paper (Sec. V-C/V-D, Tab. *protocol*) can be computed **offline** without returning to the lab.

It supersedes §2 of [`cv04`](cv04_phantom_and_qualisys_validation.md) (Qualisys side-by-side). The cv04 phantom rig (§3) is still future work, needed for *dynamic* trueness.

| Reference | What it validates | Metrology term (ISO 5725) |
|---|---|---|
| **3D-printed step ladder** with independently measured hole spacings | JTS-MM optical **trueness** (static, translational) and **precision**. The Qualisys system is measured on the same artifact. | trueness, precision |
| **Qualisys Oqus**, time-registered offline (§3) | optical and inertial **agreement**, and an independent check of the JTS-MM frame-to-trigger association | agreement |
| **Delsys Trigno** wireless sEMG, sensors adjacent to the JTS-MM electrodes | sEMG **agreement** of derived measures (envelope, RMS, MDF, onsets) | agreement |

> **Rule of thumb:** If a quantity can only be recovered by being in the lab (a distance, a photo, a GUI setting, which take belongs to which file), it goes into the session folder **during** the session. Everything else can be recomputed later.

> **No wiring between JTS-MM and the reference instruments.** The JTS-MM cameras stay on their electrical external-trigger pins (`TRIG_CSI1/2`). Qualisys and Delsys are registered to the JTS-MM master clock **offline**, from shared mechanical events and motion (§3). Qualisys and Delsys may or may not be hardware-synchronized with each other; both cases are covered.

---

## 1. Items to settle before the first session

Settle these in the pilot (§9), then record them in `manifest.yaml`.

| # | Item | Where it bites |
|---|---|---|
| 1 | Trigger rate `f_trig` (30 or 40 Hz) | exposure, blur, comparison rate |
| 2 | Marker family | which detector the session uses |
| 3 | Tag-to-body transforms of MTA, CRA, ladder array, sync wand | every pose |
| 4 | Landmark phase count (6 vs. 8) | registration |
| 5 | Qualisys ↔ Delsys hardware sync available? (trigger module / QTM integration) | case A vs. B in §3 |
| 6 | Qualisys model / camera count / QTM version; **timestamp convention** (exposure start or centre) | sub-ms bias |
| 7 | Delsys model (Trigno classic / Avanti / Avanti mini); does the sync sensor have a gyroscope? | sync signals, rates |
| 8 | sEMG cabling variant (bring-up vs. shielded harness) | noise results |

---

## 2. Equipment list

**JTS-MM**
- Pi 5 + HAT, 2× IMX296 GS cameras on tripods, triggered via their external-trigger pins (baseline per `optical.yaml`, configured 450 mm), MTA/CRA with IMUs, electrode kit (cv03).

**Qualisys**
- Oqus cameras + QTM PC, wand calibration kit (L-frame + wand).
- Retroreflective spheres (Ø 4–6.4 mm, pick by camera distance), mounted on the **same rigid bodies** as the fiducial tags: MTA, CRA, ladder array, co-registration object, sync wand.
- Case A only: Delsys trigger module, or the QTM Delsys integration (§3.5).

**Delsys**
- Trigno base station + EMGworks / Trigno Discover / QTM integration; ≥ 4–6 EMG sensors (masseter L/R, temporalis L/R, digastric L/R if space allows).
- **One extra sensor with an inertial unit** (Avanti-class: accelerometer, ideally gyroscope). It is the "sync sensor" and stays on the sync wand.

**Ground truth and fixtures**
- **Step ladder** (3D-printed) with holes and the **insertable marker array**. The array carries the fiducial tag(s) and ≥ 4 retroreflective spheres in a non-symmetric layout, and has a keyed pin that reproduces its orientation in every hole.
- **Gauge pins** (h6, matching the hole diameter) + calibrated calliper **or** access to a CMM (§5.1).
- **Co-registration object**: a rigid plate with ≥ 2 tags + ≥ 4 spheres at CAD-known positions.
- **Sync wand** (§3.3): a rigid, hand-held body with tags + ≥ 4 spheres + the Delsys sync sensor taped or clipped on rigidly. The ladder array or the co-registration object can double as the wand if it is light enough to swing. In the subject-free blocks, use the **MTA** (it carries a JTS-MM IMU) with the Delsys sync sensor taped to it.
- **Dual digitizing pointer**: the JTS-MM pointer with ≥ 4 retroreflective spheres added (or the Qualisys pointer with tags added), so that **one** placement of the tip is measured by both systems at the same time (§8.2).
- **Impact pad**: a hard, flat surface (e.g. an aluminium plate on the table) on which the wand is struck. A hard stop gives a sharper event than a soft one.
- **Clamp / stand** holding an MTA array rigidly for the static precision block (B3).
- Thermometer (room temperature, for the ladder: PLA ≈ 70 µm m⁻¹ K⁻¹).
- Camera or phone for photos, measuring tape, and a ruler for the electrode offsets.

---

## 3. Synchronization without a common clock

### 3.1 Model

JTS-MM is the reference time base (`t_master_us`, one µs counter for EMG, IMU, and trigger latches). Each reference instrument `X ∈ {Q (Qualisys), D (Delsys)}` keeps its own clock, modelled **per take** as affine:

```
t_X = (1 + ε_X) · t_M + θ_X          θ_X: offset,  ε_X: drift (ppm)
```

The drift is not negligible. Crystal tolerances are tens of ppm, and 50 ppm over a 10 min take is **30 ms**, i.e. 3 mm of apparent error at 100 mm/s jaw speed (δx = v·δt). Therefore:

- **Every take** starts and ends with the synchronization sequence **S** (§3.3). The two brackets give `θ_X` and `ε_X`.
- Between the brackets, lags estimated in hold-out windows of the motion (jaw, head) quantify the registration uncertainty and catch any change of the JTS-MM frame association during the take. They are diagnostics only and are never fed back into the fit (circularity rule, §3.4).

**Target:** registration uncertainty ≤ 1 ms for kinematics (≤ 0.1 mm at 100 mm/s, ≤ 0.1° at 100°/s) and ≤ 2 ms for sEMG. It is reported as part of the comparison overhead, not as an instrument property.

### 3.2 Why the paper-cover (occlusion) event is only a coarse anchor

Uncovering the markers is detected at the **first frame in which the target is visible**. That introduces two errors:

1. **Quantization.** The reveal falls uniformly within one frame period. Each system adds an error uniform on `[0, T]`, with mean `T/2` and SD `T/√12`: 25 ms → 7.2 ms at 40 Hz (9.6 ms at 30 Hz), but only 5 ms → 1.4 ms for Qualisys at 200 Hz. The different frame rates therefore produce both a **bias** (`(T_J − T_Q)/2` ≈ 10 ms) and random scatter.
2. **Threshold bias.** A sheet pulled away uncovers the target over 10–40 ms. JTS-MM needs the *whole* tag including its border, while Qualisys solves the body once 3 of 4 spheres are visible. The two systems therefore "see" the event at different moments, and this bias depends on how fast the sheet moves. It cannot be calibrated away.

Monte Carlo ([`cv05_sync_montecarlo.py`](cv05_sync_montecarlo.py); 40 Hz / 12 ms vs. 200 Hz) gives ≈ **26 ms bias and 8 ms SD per event**, and 3.7 ms SD for the mean of 5 events (the bias stays). At 100 mm/s that is 2–3 mm of apparent error. So use the reveal **at most** as a coarse anchor (it is not needed if S is recorded). A **motion onset** (wand at rest, then moved) is a better coarse anchor than an appearance, because the target is visible to both systems throughout.

### 3.3 Synchronization sequences S

S is done at the **start and end of every take**, in the overlapping field of view of both optical systems. The rest phases matter: they bound the motion, so each system sees a unique, aperiodic event, and they provide a reference pose that is **the same physical pose in both systems** without knowing the lag.

**S-wand (subject-free blocks; MTA + taped-on Delsys sync sensor as wand), ≈ 20 s**

1. **Rest** 2 s (wand held still, or lying on the impact pad).
2. **Three impacts.** Move the wand down at ~0.2–0.5 m/s onto the impact pad, hold still for 1 s, lift, and repeat. Keep the tags facing the cameras.
3. **Aperiodic rotation, 10 s.** Rotate about all three axes with irregular timing (no rhythm), ±30° amplitude, about 0.3–3 Hz, and peak angular speed ≲ 100°/s. Faster motion blurs the 12 ms exposure and can lose the tags.
4. **Rest** 2 s.

**S-jaw (subject blocks), ≈ 12 s**

1. **Rest** 2 s, teeth in occlusion, head still.
2. **Irregular jaw movements, ~5 s.** Fast, in random order and with uneven timing: open, close, left, right, protrude. **Not** rhythmic chewing; periodic motion makes the correlation ambiguous at the cycle period.
3. **Rest** 2 s in occlusion.
4. **One short maximal clench** (1 s), then relax. This links the two sEMG systems.

What each instrument sees:

| Instrument | Impacts (S-wand) | Rotation (S-wand: wand; S-jaw: R_CRAᵀ R_MTA) | Clench (S-jaw, B4) |
|---|---|---|---|
| JTS-MM optical | wand height: velocity drops to 0 (kink) | geodesic angle `φ(t)`, `‖ω‖` | — |
| JTS-MM IMU | acceleration spike (MTA as wand) | gyro `‖ω‖` | — |
| Qualisys | kink in height | `φ(t)`, `‖ω‖` | — |
| Delsys sync sensor | acceleration spike | gyro `‖ω‖` (if available) | — |
| JTS-MM / Delsys sEMG | — | — | envelope bursts on adjacent sites |

### 3.4 Estimators (offline)

**Frame-invariant signals first.** For the same rigid body, `R_Q(t) = A · R_J(t) · B`, with an unknown world-frame rotation `A` and an unknown tag-to-sphere body offset `B`. The geodesic angle from a reference instant,

```
φ(t) = angle( R(t) · R(t0)ᵀ )   →   R_Q(t) R_Q(t0)ᵀ = A · R_J(t) R_J(t0)ᵀ · Aᵀ
```

is **identical** in both systems, and so is `‖ω(t)‖`. The time lag can therefore be estimated **before** any spatial registration or hand–eye calibration. The same holds for the relative mandibular rotation `R_CRAᵀ R_MTA`, so the motion battery itself can be used to track drift.

1. **Motion cross-correlation (fine).** Minimize over `τ` the squared difference between JTS-MM `φ` at its **exposure centroids** (`t_trig + τ_exp/2`) and Qualisys `φ`, which is first **box-filtered over τ_exp** (to emulate the same blur) and then interpolated. The result is **sub-frame** even though the frame periods differ (25 ms vs. 5 ms), because a continuous signal is interpolated rather than an event detected. The Cramér–Rao bound is
   `σ_τ ≈ σ_φ / (rms(φ̇) · √N)`. With σ_φ = 0.1°, rms(φ̇) ≈ 250°/s, and N = 400 frames (10 s at 40 Hz), σ_τ ≈ 20 µs; the Monte Carlo gives 15–25 µs. In practice, model errors (outliers, interpolation, nonlinear blur, each system's timestamp convention) dominate, so expect sub-millisecond. The search window comes from the impacts, so the correlation peak is unambiguous; this is why the motion must be aperiodic.
2. **Impact (hinge) fit.** Per system, fit `h(t) = v · max(t_i − t, 0) + h0` to the wand height around each impact. For JTS-MM, the model is averaged over the exposure window. Monte Carlo (σ_p 0.2 mm / 0.05 mm): **σ(t_i) ≈ 0.46 ms** for JTS-MM and **0.07 ms** for Qualisys per impact; three impacts → ≈ 0.3 ms. In the IMUs (JTS-MM, Delsys sync sensor) the impact is an acceleration spike: take the onset by threshold plus interpolation. Its precision is ~1 sample at the Delsys IMU rate, which is coarser than the gyro cross-correlation.
3. **Angular-speed cross-correlation (IMU ↔ optical).** Correlate `‖ω‖` from the Delsys sync-sensor gyro (or JTS-MM IMU) with `‖ω‖` from Qualisys or JTS-MM optical. This is invariant to the sensor mounting.
4. **sEMG–sEMG.** Cross-correlate the clench-burst envelopes (adjacent sites, same muscle). This links Delsys to JTS-MM directly. The lag includes the Delsys on-sensor filter and RF delay, which is exactly what onset comparisons need.
5. **Frame-association check (independent of the camera stack).** In the subject-free blocks, the MTA carries both tags and a JTS-MM IMU. The lag of **JTS-MM optical vs. Qualisys** minus the lag of **JTS-MM IMU vs. Qualisys** isolates the camera exposure/association offset. It must be the calibrated constant (≈ τ_exp/2 handled by the centroid convention), with **no whole-frame component** (25 or 33 ms). Compare the start and end brackets as well.

**Computation per take.** This is the recipe implemented and tested in [`cv05_sync_estimation.py`](cv05_sync_estimation.py) (`register_take`, `envelope_lag`, `clock_model`).

1. **Signal.** Use `R_rel = R_CRAᵀ R_MTA` for S-jaw, or the wand rotation for S-wand, in both systems. Compute `φ(t) = angle(R(t) · R̄_restᵀ)`, where `R̄_rest` is the chordal mean over the rest phase that precedes the motion. Rest is static, so any instant in it is the same pose in both systems, even before the lag is known.
2. **Windows.** Detect rest → motion → rest independently in each stream from the smoothed angular speed (`find_sync_windows`; threshold ≈ 8°/s, 0.3 s smoothing, short pauses bridged).
3. **Coarse lag.** Normalized cross-correlation of `‖ω‖` on a 1 kHz grid within the wall-clock search range (e.g. ±1–2 s). The *peak ratio* compares the main peak with the best peak outside the main lobe; a ratio near 1 means ambiguous (redo, or check the wall clock).
4. **Fine lag.**
   `τ̂ = argmin_τ Σ_k [ φ_J(t_k) − φ̄_Q(t_k + τ) ]²`, with `t_k = t_trig,k + τ_exp/2` (JTS-MM exposure centroids) and
   `φ̄_Q(t) = (1/τ_exp) ∫_{t−τ_exp/2}^{t+τ_exp/2} φ_Q`. This is the blur-matched reference, computed exactly from the cumulative integral of the linear interpolant. Use a grid of 20 µs plus a parabolic refinement; the standard error comes from Gauss–Newton.
5. **Clock model.** With lags `τ̂_s`, `τ̂_e` at master times `t_s`, `t_e` (motion centres):
   `ε̂ = (τ̂_e − τ̂_s) / (t_e − t_s)`, `θ̂ = τ̂_s − ε̂ t_s`, and `σ_ε = √(σ_s² + σ_e²) / (t_e − t_s)` (0.15 ms over 300 s → 0.7 ppm). Then map with `t_Q = (1 + ε̂) t_M + θ̂`. With more than two windows, use a weighted least-squares line (`clock_model`).
   **Plausibility:** |ε̂| ≲ 100 ppm. A `|τ̂_e − τ̂_s|` close to a whole JTS-MM frame period (25/33 ms) is a frame-association change, not drift: check the IMU-vs-optical lag.
6. **sEMG.** Moving-RMS envelope (50 ms) of the clench on adjacent sites, cross-correlation with parabolic peak interpolation (`envelope_lag`). Use the same clock model, either per pair (case B) or as a check (case A).
7. **Uncertainty (hold-out).** Estimate lags in a few windows *inside* the take (e.g. 10 s of the motion battery, with the reference pose at a turning point of the motion) against the fitted model. The **spread** of these residuals is the reported registration uncertainty. The Gauss–Newton standard error was optimistic by about 2× in simulation.

Synthetic self-test (`python3 cv05_sync_estimation.py`): 300 s take, JTS-MM 40 Hz / 12 ms against Qualisys 200 Hz, 40 ppm drift, unknown tag-to-sphere offsets. Results:
- drift recovered to 0.4 ppm;
- mapping error ≤ 0.14 ms over the take;
- hold-out spread ≈ 0.15 ms;
- sEMG clench lag within 1 ms.

**Circularity rule.** The lag is fitted to data that are also evaluated.
- With one offset and one drift per take (`p = 2` against thousands of samples), the optimism of the residual is negligible: `E[RSS] = (N − p) σ²`.
- However, a global offset **absorbs any constant timing error of JTS-MM**, and per-window lags would also absorb dynamic errors (e.g. blur-related lag).
- Therefore: fit **only `θ`, `ε` per take**, and **only on the S sequences**.
- Hold-out windows are diagnostics and are **never fed back**.
- The frame association is checked independently by the IMU-vs-optical lag difference (item 5 above).

### 3.5 Case A — Qualisys and Delsys hardware-synchronized

Options, in order of preference:
- **QTM integration of the Delsys Trigno** (QTM records EMG and IMU of the Trigno sensors in the same `.qtm`). Check that your QTM and Trigno versions support it, and how QTM aligns the streams (start trigger and/or continuous sync).
- **Trigno trigger module** wired to the Qualisys sync/trigger I/O for **start and stop** (both instruments are mains-side; there is no subject connection on this link). Start + stop gives both offset and drift between Q and D.

Then only the JTS-MM ↔ Qualisys registration is needed (motion + impacts), and Delsys inherits it: `t_D = f_QD(t_Q)`. The **sEMG–sEMG lag (B4)** and the sync-sensor IMU then serve as an **independent closure check** of the whole chain.

### 3.6 Case B — no link between Qualisys and Delsys

Estimate the three pairwise lags independently in each bracket:

| Pair | Signals | Expected σ |
|---|---|---|
| δ_JQ (JTS-MM → Qualisys) | wand `φ` cross-correlation + impacts | ≤ 0.5 ms |
| δ_JD (JTS-MM → Delsys) | sEMG–sEMG clenches (B4), JTS-MM IMU vs. sync-sensor IMU (subject-free blocks) | 1–2 ms |
| δ_QD (Qualisys → Delsys) | sync-sensor `‖ω‖` / acceleration vs. Qualisys `‖ω‖` / impacts | ≤ 1 ms |

Three equations, two unknown offsets → weighted least squares. The **closure residual** `δ_JQ + δ_QD − δ_JD` is reported as the registration uncertainty. Do the same for the end bracket to get the drifts. Case B is fully sufficient for the paper; case A only makes it more convenient and adds redundancy.

### 3.7 Timestamp conventions (sub-ms biases)

JTS-MM stamps **exposure start** (`t_trig`), and the exposure lasts ≈ `width_us` + 14.26 µs. Find out whether QTM stamps frame start or exposure centre (Oqus marker exposure is ~0.1–1 ms), and what delay the Delsys software applies per sensor type. In the simulation, ignoring the Qualisys exposure centre alone gave a 0.25 ms bias. The motion-based lag absorbs all constant conventions into `θ_X`, which is fine for agreement metrics. Still, write the conventions in the manifest so that the estimated offsets can be explained.

### 3.8 Safety

- Subject-connected: the JTS-MM HAT (battery or CE-marked supply, isolated hub) and the Delsys sensors (battery, wireless).
- **No signal or ground connection** between JTS-MM and the Qualisys/Delsys equipment. The case-A link (Delsys trigger module ↔ Qualisys) connects only mains-side instruments.
- See [`h07`](../hardware/h07_power_grounding_safety.md). The system is neither galvanically isolated nor certified. The ethics approval must cover the reference instruments.

---

## 4. Marker preparation (once per rig, re-check per session)

1. **Spheres on the arrays.** Each rigid body (MTA, CRA, ladder array, co-registration object) gets ≥ 4 spheres in a non-symmetric layout, without occluding the tags. Define the body in QTM (*6DOF body*) from a static capture, and export the definition (`.xml`) into the session folder.
2. **Tag-to-body transforms.** Take them from CAD and check them by measurement. Write them into `optical.yaml`; these are the placeholders flagged in §1.
3. **Sphere-to-tag offsets `B`.** These are the constant transforms between the QTM body frame and the JTS-MM body frame of the same rigid body (MTA: `B_M`, CRA: `B_C`, pointer). Their CAD values are stored as priors; they are **re-estimated offline** from static poses in ≥ 3 orientations per body (B1, *before* the arrays are attached to the subject), see §8.2.
4. **Dual pointer.** Pivot-calibrate the tip **once, from one pivot recording seen by both systems** (offline: least-squares pivot on each system's 6DOF stream). Record the tip offsets and both pivot residuals.
5. **IR interference.** Strobes reflect off glossy prints and the spheres. Use matte tag prints, angle or baffle the Oqus away from the Pi lenses, and confirm in a pilot capture that the tag detection rate is unchanged with the strobes on versus off.

---

## 5. Step ladder

### 5.1 Ground truth: measure the actual hole spacings (once, before the first session)

The printed nominal pitch is **not** the ground truth (print tolerance ≈ 0.1–0.2 mm).

- **CMM (best):** measure each hole axis (centre at a defined depth), giving the hole centres `c_i` and all spacings `d*_ij`, with the CMM uncertainty.
- **Gauge-pin method (good):** insert gauge pins of the hole diameter into holes *i* and *j* and measure over the pins with the calliper: `d*_ij = L_over − d_pin`. Do ≥ 3 repetitions per pair, at the same depth as the array's reference point. Uncertainty ≈ calliper uncertainty plus the pin–hole fit.
- Record: instrument, its uncertainty, temperature, the full `d*_ij` table (CSV), hole numbering (photo with labels), the array's pin-axis point in the array CAD frame (`o`), and whether the key reproduces the orientation.
- Re-measure after any drop or deformation, and at least yearly.

### 5.2 Recording procedure (block B2)

- Fix the ladder to a rigid base so it cannot move while you insert the array.
- **Placements:** ≥ 3 per session, more if time allows: along the camera x axis (horizontal), along the y axis (vertical), along depth (z, toward the cameras), diagonally, and at the centre and edges of the working volume. Both systems must see the array in every hole.
- **Per placement:** insert the array in hole 1 → hold **≥ 3 s static** (hands off) → move to hole 2 → … → last hole. Then **repeat the full sequence 3×**, with re-insertion each time.
- Log each insertion as a phase. Press SPACE in `jtsmm record` (the `/phases` interval). Alternatively, the hold intervals are segmented offline from the pose velocity, so a missed key press is recoverable. Note the hole order on the sheet.
- Write down the placement label and a photo for each placement.

### 5.3 What it yields

For each hole `i` and system, the mean position `p̄_i` of the array point `T_i·o`. Using the pin-axis point makes the result robust to residual orientation play. Then
`e_ij = ‖p̄_i − p̄_j‖ − d*_ij` for all pairs (mean = trueness, plus SD, RMS, max), the scale error (slope of measured vs. true distance), the dependence on position, the re-insertion repeatability, and the relative rotation between holes (nominally 0; a rotation-trueness check). The same statistics are computed for Qualisys, giving **the uncertainty of the reference**.

---

## 6. Session plan (blocks)

Every block is one `jtsmm record` take, recorded **simultaneously** by all instruments and **bracketed by S** (start and end): S-wand without the subject, S-jaw with the subject. QTM and Delsys may record one file per block or one continuous file; the S brackets identify the takes either way. Log the wall-clock start time of each take on each PC as a coarse anchor (NTP on both PCs if possible).

| Block | Subject? | Content | Duration (+ 2 × S ≈ 40 s) | Purpose |
|---|---|---|---|---|
| **B0** | no | JTS-MM extrinsics check (cv01 static board, cross-view gate); QTM wand calibration; IMU mounting tumble if the arrays were re-glued (cv02). One take with the **MTA + Delsys sync sensor as wand**: 5× S | 10–15 min | valid calibrations; JTS-MM IMU ↔ optical ↔ Qualisys ↔ Delsys lags, and the frame-association check (§3.4 item 5) |
| **B1** | no | Co-registration object static, 10 s; each body (MTA, CRA, ladder array, pointer) static, 10 s, in ≥ 3 orientations; dual-pointer pivot, 20 s | 4 min | world-frame registration, offsets `B_M`, `B_C`, pointer tip in both systems |
| **B2** | no | Step ladder, ≥ 3 placements × all holes × 3 re-insertions, ≥ 3 s per hole | 20–30 min | trueness of both systems |
| **B3** | no | MTA array clamped at nominal working distance, static, 120 s (+ 2 further distances/angles if time allows) | 3–6 min | static precision of both systems |
| **B4** | yes, after electrode prep | 3 short maximal clenches (1 s on / 2 s off) | 1 min | sEMG–sEMG registration with the subject in place |
| **B5** | yes | Landmark digitization with the **dual pointer** (6/8 phases, both systems record the same placement); `jtsmm record --mvc` (`mvc_closing`, `mvc_opening`), recorded by Delsys too; motion battery: opening/closing, laterotrusion L/R, protrusion/retrusion, cyclic chewing (30 s × 3 each); graded opening at 3 speeds; gum chewing 30 s. **Every take starts and ends with S-jaw** | 30–40 min | kinematic, inertial, and sEMG agreement |
| **B5s** | yes | Subset (opening/closing + chewing) with the **short-exposure profile** (100 Hz / 2 ms) | 5 min | motion-blur share of the dynamic error |
| **B6** | yes, then no | Repeat B4; remove subject; repeat B1 and one B0-style wand take with the MTA | 3 min | stability of timing and registration over the session |

The order matters. **MVC comes before the motion battery**, and **B0/B4 and B6 bracket the subject blocks**. Keep takes **≤ 5 min**, so that a linear drift model per take is safe.

---

## 7. Settings and commands per instrument

### 7.1 JTS-MM

```bash
# once per session
jtsmm doctor
jtsmm emg check                         # lead-off, |Z|, baseline noise (cv03)

# artifact / wand blocks (no subject; EMG still runs, which is harmless)
jtsmm record --optical-track --optical-config software/optical/etc/optical.yaml \
  --protocol cv05_B2 --meta block=B2 --meta placement=x_axis --meta qtm_file=<name> \
  --meta delsys_file=<name> --meta operator=<id>

# subject blocks
jtsmm record --mvc --subject <hash> --protocol cv05_B5_mvc --meta block=B5 ...
jtsmm record --optical-track --optical-config ... --use-experiment-plan \
  --subject <hash> --protocol cv05_B5 --meta block=B5 ...

# after every take
jtsmm integrity check --strict <session_dir>
```

- The wand needs a tag ID and body definition known to the tracker. Add it as an extra body in `optical.yaml` (or reuse the ladder-array body).
- Mark each S as a phase (SPACE), or segment it offline from the rest/motion pattern.
- Keep `trigger_after_cameras: true` and the period compensation as shipped.
- **Keep the raw camera frames** (JPEG dump) for B0–B3 at least, so detection can be rerun offline.
- Do not change the camera exposure, gain, focus, or tripod position after B0. If anything moves, repeat B0 and B1 and note it.
- Gain 6 (default). Note the sEMG cabling variant in `--meta cabling=<bringup|shielded>`.

### 7.2 Qualisys (QTM)

- Free-running at a fixed rate (e.g. 200 Hz); short marker exposure. No external timebase is needed.
- Save the **raw `.qtm`** (keeps 2D data for re-tracking), the **calibration** (`.qca` + calibration report), the **6DOF body definitions** (incl. the wand), and the project settings export.
- Export per take: **TSV** of 6DOF bodies (position, rotation matrix, residual, *with frame number and time column*), **TSV/C3D** of 3D markers, and, in case A, the Delsys data. Disable gap filling and smoothing in the export, or export raw and filled separately.
- Note the QTM version, the frame rate, the marker exposure, the timestamp convention, the camera serials and positions, and the wand-calibration residual.

### 7.3 Delsys

- EMG sensors: one per muscle, **immediately adjacent** to the JTS-MM electrode pair, along the fibre direction. Measure and photograph the **centre-to-centre offset** per site (mm, direction). Skip the digastric if the two sensors do not fit, and write down that you did.
- **Sync sensor**: rigidly on the wand, IMU enabled at its highest rate (accelerometer, gyroscope if available).
- Native EMG rate (model-dependent, ≈ 2 kHz) and on-board filter setting.
- Save the native file **and** a CSV export with **per-channel time columns** (EMG and IMU channels have different rates), sensor serials, the channel-to-muscle map, and the software version. Note the per-sensor-type delay the software applies, if documented.

---

## 8. Offline evaluation (what the recorded data must allow)

### 8.1 Time registration

1. **JTS-MM internal:** `t_master_us` for EMG, IMU, and frames. The frame association is checked with `jtsmm integrity check` (gate: |median| < 1.5 frames, |slope| < 50 µs/s).
2. **Per take, per reference instrument:** run the computation of §3.4 (steps 1–7) on the start and end S. This gives `θ`, `ε`, the hold-out spread, and the peak ratios. Keep the impact hinge fits (S-wand) as a cross-check of the coarse lag.
3. **Case A:** apply the hardware Q↔D link; the sEMG clench lag is the closure check. **Case B:** weighted least squares over δ_JQ, δ_JD, δ_QD, plus the closure residual.
4. **Frame-association check** (B0, B6, S-wand with the MTA): lag(optical, Q) − lag(IMU, Q) = constant, with no whole-frame component.
5. **Map and compare:** evaluate Qualisys at `t_Q = (1 + ε̂) t_M + θ̂` for every JTS-MM exposure centroid, after the box filter over τ_exp (SLERP for rotations). Report static and dynamic phases separately, and report the registration uncertainty (hold-out spread) next to every dynamic agreement number.
6. Store per take in `sync/`: window times, coarse/fine lags, standard errors, peak ratios, `θ`, `ε`, hold-out residuals, closure residual.

### 8.2 Spatial registration

The aim is to keep **registration error** and **tracking error** apart.

- **Main comparison quantity:** the relative pose `^CRA T_MTA`. It is independent of both world frames and of the camera placement. The relative poses of the two systems differ by the marker-set offsets: `T_Q = B_C⁻¹ · T_J · B_M`.
- **Offsets `B_C`, `B_M`:** solve from the static B1 poses of each array (≥ 3 orientations; `T_Q,i = A · T_J,i · B` with the world-frame transform `A` and the offset `B`, solved jointly or via AX = XB). Report the residuals.
- **One anatomical frame for both systems.** Digitize the six dental landmarks once with the **dual pointer** (both systems record the same placement, so the operator's placement error is common-mode and cancels). Derive **one** set of anatomical frames, e.g. from the Qualisys digitization or from the mean of the two, and apply it to both systems through `B_C`, `B_M`. The registration is then identical by construction; the remaining difference is tracking error.
- **Cross-check:** the landmarks as digitized by each system, mapped through `B`, should agree within ~0.1–0.3 mm. Report this. Do **not** build separate frames per system: three points with spread `L` give an orientation error of `δθ ≈ σ_p / L` (0.1 mm / 35 mm ≈ 0.16° → 0.14 mm at 50 mm excursion). That would appear as a fixed or proportional bias in the regression, but it is registration, not tracking.
- **World frames** (ladder, absolute plots only): Procrustes (Kabsch) on the co-registration object (B1), cross-checked with the ladder hole centres (B2, both systems) and with B6. Report all three residuals and their differences.

### 8.3 Metrics (paper Tab. *reference-agreement* / *emg-agreement*)

- Kinematics: RMSE, bias, and 95th percentile per axis and norm; geodesic rotation error; Bland–Altman 95 % LoA; ICC (form stated); least-products regression with CIs (as Mostashiri 2018); error vs. incisal velocity, B5 vs. B5s; clinical measures (max. opening, laterotrusion, protrusion, chewing envelope); per participant or mixed model.
- Inertial: geodesic error of the IMU relative orientation vs. Qualisys, yaw drift rate, angular-speed lag.
- sEMG: envelope r and concordance ρc, Bland–Altman on the %MVC envelope, RMS (%MVC), MDF/MNF, TKEO onset difference, coherence 20–450 Hz, and resting noise on adjacent sites. Both signals go through the **identical** filter chain (s09 fit-and-subtract) after resampling.
- Multimodal: sEMG onset → motion onset lead, computed with JTS-MM alone and with Delsys + Qualisys as a cross-check.

Tooling: the sync estimators exist as a tested reference implementation in [`cv05_sync_estimation.py`](cv05_sync_estimation.py). The import of the reference recordings (QTM and Delsys exports) and the agreement metrics are stages of the analysis pipeline (`software/biomech/biomech/validation/`, [an14](../analysis/an14_system_validation.md)). The Monte Carlo used for the numbers in §3 is a starting point for the sync estimators.

---

## 9. Pilot session (no subject) — do this first

- [ ] Wand (MTA) and dual pointer defined in both systems; tags are detected during the S rotation at ≤ 100°/s (no blur losses).
- [ ] Dual-pointer pivot in both systems; tip residuals noted.
- [ ] Impacts are visible as clean kinks in both optical systems and as spikes in the Delsys sync sensor (and the JTS-MM IMU with the MTA as wand).
- [ ] Run `cv05_sync_estimation.register_take` on 3 pilot takes (one with a volunteer doing S-jaw): peak ratio clearly > 1, drift plausible (tens of ppm), hold-out spread ≤ 1 ms, case-B closure residual ≤ 1 ms.
- [ ] Frame-association check: lag(optical, Q) − lag(IMU, Q) is constant across takes, with no whole-frame steps.
- [ ] Case A: verify the Q↔D link (start/stop or QTM integration) against the sync-sensor estimate.
- [ ] Find out the QTM and Delsys timestamp conventions and per-sensor delays (§3.7).
- [ ] Tag detection rate with the Oqus strobes on equals the rate with them off.
- [ ] One ladder placement end-to-end, analysed with a quick script (does `e_ij` come out sensibly for both systems?).
- [ ] Fill in the open items in §1 and freeze `jtsmm.yaml` / `optical.yaml`; record their hashes.

---

## 10. Session folder and manifest

```
validation/
└── 2026-MM-DD_<subject-hash or BENCH>/
    ├── manifest.yaml            # see template below
    ├── notes.md                 # timeline: block, start/stop wall time, anything unusual
    ├── jtsmm/                   # one session dir per take (session.h5, frames/, integrity report)
    ├── qtm/                     # *.qtm, *.qca, calibration report, 6DOF body xml, exports/*.tsv|*.c3d
    ├── delsys/                  # native files + CSV exports
    ├── calib/                   # camera calibration bundle used, imu_mounting.json, optical.yaml, jtsmm.yaml
    ├── metrology/               # ladder d*_ij CSV (+ instrument, uncertainty, temperature), hole map photo
    ├── photos/                  # setup overview, camera positions, electrode sites (with ruler), ladder placements
    └── sync/                    # sync-estimator outputs per take (lags, clock model, closure residual)
```

```yaml
# manifest.yaml (template)
session: 2026-MM-DD_S01
operator: <id>
subject: {id_hash: <hash>, age: , sex: , max_opening_mm: , upper_intermolar_mm: , lower_intermolar_mm: }   # anthropometrics per paper Sec. V-E
jtsmm:
  git_commit: <hash>
  firmware: <version>
  config_hash: {jtsmm_yaml: , optical_yaml: }
  trigger: {rate_hz: , width_us: 12000, short_profile: {rate_hz: 100, width_us: 2000}}
  marker_detector: <apriltag|aruco>  # dictionary/family
  cabling: <bringup|shielded>
  camera_baseline_mm_measured:
qualisys:
  model: Oqus <x>, cameras: , qtm_version: , calibration_residual_mm:
  frame_rate_hz: , marker_exposure_us: , timestamp_convention: <frame start|exposure centre|unknown>
  marker_size_mm:
delsys:
  system: <Trigno/Avanti/...>, software: <EMGworks/Trigno Discover x.y>
  emg_rate_hz: , imu_rate_hz: , sync_sensor: {serial: , gyro: <yes|no>}
  per_sensor_delay_ms: , delay_source: <vendor doc|measured|unknown>
  sites: {masseter_L: {sensor: , offset_mm: , direction: }, ...}
sync:
  case: <A_hardware_QD|B_offline_only>
  qd_link: <qtm_integration|trigger_module_start_stop|none>
  wand: {body: , tags: , spheres: , delsys_sync_sensor: }
  wall_clock: {pi_ntp: <yes|no>, qtm_pc_ntp: , delsys_pc_ntp: }
  s_brackets_per_take: 2
ladder:
  id: , holes: , dstar_csv: metrology/ladder_dstar.csv, instrument: , uncertainty_um: , temperature_C:
  pin_axis_point_in_array_frame_mm: [x, y, z]
  placements: [x_axis, y_axis, z_axis, diagonal]
blocks:            # take name per instrument, so files can be joined without guessing
  - {block: B2, placement: x_axis, jtsmm: <dir>, qtm: <file>, delsys: <file>, note: }
deviations: []     # anything that differs from this protocol
```

---

## 11. Acceptance checklist (per session)

- [ ] Pilot items (§9) done for this hardware configuration; manifest complete.
- [ ] B0: JTS-MM cross-view gate passed; QTM calibration residual noted.
- [ ] Every JTS-MM take passes `jtsmm integrity check --strict` (association gate included).
- [ ] QTM exports contain frame number + time and 6DOF residuals; raw `.qtm` saved.
- [ ] Every take has S at start **and** end, visible in all running instruments (wand tracked by both optical systems, sync-sensor IMU active).
- [ ] Sync estimate per take: start/end lags consistent; case B closure residual ≤ 1 ms (kinematics) / ≤ 2 ms (sEMG).
- [ ] B4/B6 head motion visible in both optical systems; clenches visible in both sEMG systems.
- [ ] Ladder: ≥ 3 placements × all holes × 3 re-insertions, ≥ 3 s static per insertion, both systems tracking.
- [ ] Electrode offsets measured and photographed; MVC recorded on both systems before B5.
- [ ] Session folder backed up (NAS) with checksums.

---

## 12. References

- IEEE Sensors Journal paper: Sec. V (methodology, incl. time registration), Tab. *protocol*, Sec. VI (results structure).
- [`cv01`](cv01_camera_calibration.md), [`cv02`](cv02_imu_calibration.md), [`cv03`](cv03_emg_setup_mvc.md), [`cv04`](cv04_phantom_and_qualisys_validation.md); [`h06`](../hardware/h06_synchronization.md), [`h06_trigger_io`](../hardware/h06_trigger_io_schematic_build_guide.md), [`h07`](../hardware/h07_power_grounding_safety.md); [`s06`](../software/s06_time_synchronization.md), [`s09`](../software/s09_mains_removal.md); [`op01`](../operations/op01_setup_checklist.md).
- ISO 5725-1 (trueness/precision); Bland & Altman 1999; Koo & Li 2016 (ICC); Ludbrook 2010 (least-products regression); Mostashiri et al. 2018 (PFMS vs. Vicon).
- Qualisys QTM user manual (timestamps, Delsys integration, trigger I/O); Delsys Trigno user guide (trigger module, IMU sensors, per-sensor delays). **Check against the versions you use.**
