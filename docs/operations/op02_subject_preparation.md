<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# op02 — Subject Preparation

This document details how to prepare the subject for a JTS-MM session: skin preparation, electrode placement, MTA / CRA mounting, and the safety-related checks. The setup is non-invasive, takes ~15–20 minutes, and should be done with the subject seated, hair pulled back, and any cosmetic creams already removed.

## 1. Materials

Per subject (single-use unless noted):

- 7 disposable Ag/AgCl electrodes (Ambu BlueSensor P-00-S, as used in the recordings; any pre-gelled Ag/AgCl snap electrode fits).
- 1 jar of skin-prep gel (Nuprep), shared (apply with a fresh cotton swab per site).
- ~20 alcohol pads.
- 1 roll of 3M Micropore tape (~30 cm needed).
- Disposable gloves.
- Fresh MTA dental-resin printout (or a sterilized re-usable one) — biocompatible material (IBT Resin, medical-grade PETG).
- The CRA forehead band — fabric, wipeable, re-used between subjects after disinfection.
- Sterile dental cotton rolls (for MVC bite trials). *Not used in the 2026 dataset: the closers were clenched with the teeth in contact (cv03 §6).*
- Tongue depressor (gauze-wrapped) for digastric MVC.

## 2. Skin preparation procedure

For each of the 7 electrode sites (6 EMG + DRL):

1. **Identify the landmark** using the diagrams in §3 below.
2. **Wipe with alcohol pad**; let dry (~10 s).
3. **Abrade with Nuprep**: on a cotton swab tip, apply a small amount of Nuprep and rub in circles for ~5 s. Stop at slight pinkness; do not break the skin.
4. **Wipe off the gel** with a fresh alcohol pad; let dry (~10 s).
5. **Apply the electrode** firmly; press for 5 s.
6. **Connect the lead wire** with a gentle but firm push onto the snap.

Total time per site: ~1 min, so ~7 min for all electrodes.

## 3. Electrode placement (with landmarks)

Channel numbers follow the cable labels and the montage table of [an03 §1](../analysis/an03_emg_processing.md#1-montage) (sides are the participant's). The placements follow SENIAM where available and Castroflorio et al. 2005 for the digastric. The text below describes the placement.

### 3.1 Masseter (CH1 right, CH5 left)

- Ask the subject to clench. The masseter is the prominent muscle on the side of the cheek that bulges visibly.
- Place the **active electrode** in the middle of the muscle belly, approximately **2 cm above the lower edge of the mandibular angle**.
- Place the **reference electrode** directly next to it, anterior along the muscle fiber direction (fibers run nearly vertically): about 20–30 mm centre to centre, as recorded (photographs).
- Both electrodes should be firmly within the muscle belly; if you can feel the bone underneath, move up by 5 mm.

### 3.2 Anterior Temporalis (CH2 right, CH6 left)

- Ask the subject to clench. The temporalis fibers contract at the temple.
- Place the **active electrode** ~2 cm above and ~2 cm posterior to the frontozygomatic suture (where the cheekbone meets the temple).
- Place the **reference electrode** directly next to it, superior along the fiber direction (about 20–30 mm centre to centre, as recorded).
- Fibers are roughly vertical at this location; align the electrode pair accordingly.
- Note: the hair line is often close; trim a small patch of hair (with subject consent) if necessary for good contact.

### 3.3 Anterior Digastric (CH3 right, CH4 left)

- Ask the subject to open the mouth against resistance (operator gently pushes up under the chin). The digastric should be palpable just behind the chin, lateral to the midline.
- Place the **active electrode** ~10 mm lateral to the midline, in the anterior belly.
- Place the **reference electrode** directly next to it, anterior along the fiber direction (about 20–30 mm centre to centre, as recorded).
- The submandibular region has thin skin; press electrodes firmly but be careful of subject discomfort.

### 3.4 DRL / bias (separate lead)

> **As run in the 2026 dataset:** the DRL / bias electrode was on the **forehead**, not on the mastoid.

- Place a single electrode on the **mastoid process** (bony bump behind the ear) of either side.
- The mastoid is bony, away from muscle, and provides a stable common-mode reference.
- Alternative: earlobe (some operators prefer this); both work.
- With the bias drive on (`bias.enabled: true` in the config, the default, or `--bias-drive`), the DRL electrode **must** be connected — otherwise 50 Hz common-mode rejection is not improved.

## 4. CRA mounting

- The CRA is a soft fabric headband with the marker carrier on the forehead.
- Place it on the forehead such that the **three AprilTag pads** face the cameras (cheek/forehead arc unobstructed).
- Verify each tag is clean, unscratched, and matte (no specular tape); replace worn PETG pads before the session if needed.
- Tighten the strap behind the head; verify it does not slip when the subject nods.
- Verify the CRA-mounted IMU sits flat against the headband (not twisted).
- Route the IMU cable along the band to the back of the head, down the back, and to the patient cable strain-relief clip on the subject's collar.

## 5. MTA mounting

The MTA is the most invasive piece of the system (it sits on the teeth). Use the procedure from the IEEE SENSORS 2025 paper §II.A:

1. Subject is seated upright, mouth open.
2. Operator dries the lower-front-teeth surface with a tissue.
3. Operator applies a small amount of biocompatible dental adhesive to the MTA's flat tooth-contact surface.
4. Operator presses the MTA onto the lower teeth, holds for 30 s (per adhesive directions).
5. Verify the MTA does not move when the subject moves their tongue or smiles.
6. Confirm the **three AprilTag pads** on the MTA arc point toward the cameras and are not covered by the cheek or lips at rest.

The IMU breakout on the MTA sits on the cheek-side, outside the mouth. Its cable exits the corner of the mouth, runs along the cheek, and joins the patient cable at the collar.

## 6. Cable management

- All cables (EMG + DRL + 2 IMU + MTA-IMU) join at a single strain-relief clip on the subject's collar.
- **Regional harness (Option B):** three face/chin cables + DRL — velcro-bundle at collar ([`h03`](../hardware/h03_emg_frontend.md) §10.2.2).
- **Bring-up (Option 0):** thirteen separate unshielded leads to pin headers — **short bench runs only** ([`h03`](../hardware/h03_emg_frontend.md) §10.2.0).
- Provide a service loop of ~5 cm at the electrode end so head motion does not tug.
- Collar → HAT run: **~1.2–1.5 m** per EMG cable (length budget: [`h03_emg_frontend.md`](../hardware/h03_emg_frontend.md) §10.1); IMU I2C budget: [`h04_imu_subsystem.md`](../hardware/h04_imu_subsystem.md) §3.3.1.
- Tape any cable bend that the subject is likely to disturb (e.g. behind the ear) with Micropore.
- **IMU cables:** use the **7-pin** harness only at the HAT (**SH** plug). Do not solder the cable shield to the BNO085 breakout **GND** pad — pin **7** is for the braid only ([`h04_imu_cable_assembly.md`](../hardware/h04_imu_cable_assembly.md)).

## 7. Pre-recording subject checks

- Ask the subject: "Are you comfortable? Any cable tugging?". Adjust if needed.
- Ask the subject to make a few jaw movements (open, close, side-to-side); verify by looking at the live viewer that:
  - All 6 EMG channels show activity.
  - Both IMU quaternions update smoothly.
  - All optical markers are visible in all cameras.
- Confirm with the subject that they understand the upcoming protocol.

## 8. End-of-session removal

- Stop the recording first; do not remove anything live.
- Remove the MTA last (it requires careful prying); use a dental-release agent if needed. The subject's teeth may feel slightly residue-sticky for a few minutes after removal — provide a glass of water.
- Remove the CRA.
- Remove the EMG electrodes: peel slowly from one edge to avoid pulling skin / hair. Warn the subject before each one.
- Inspect the skin for irritation or redness; document if any.

## 9. Subject comfort and safety

- The whole setup is non-invasive and the patient sees only research-grade equipment (no high voltages, no needles).
- The maximum current the subject can ever see, in any fault condition we have identified, is ≈23 µA (full 5 V AVDD rail through the 220 kΩ DRL series resistor; see [`../hardware/h07_power_grounding_safety.md`](../hardware/h07_power_grounding_safety.md) §3). This is well below the 100 µA IEC 60601 / AAMI limit for patient leakage current.
- If the subject reports any discomfort, stop immediately, investigate, and document.
- Hypoallergenic electrodes are the default; if the subject reports skin irritation in a prior session, document and switch to a different brand (Skintact F-301 is a common alternative).

## 10. Data privacy reminder

- The session HDF5 contains the subject's anonymized ID, not their name.
- Video files contain the subject's face; treat as personal data, encrypt during transfer, delete from working copies once analysis is complete (per the consent form).
- The mapping subject-name ↔ anonymized-ID is held only by the study lead and is not part of the data directory.

## 11. Operator's body posture

A note for the operator's well-being: subject preparation involves leaning over the seated subject for ~15 min. Use an adjustable-height chair for the operator; do not work hunched over. Repetitive subject prep is the most common source of repetitive-strain injury in EMG labs.

## 12. References

- Original JTS paper (IEEE SENSORS 2025) §II.A — MTA attachment procedure.
- SENIAM project — electrode placement for masseter and temporalis.
- Castroflorio T. et al., 2005 — anterior digastric placement.
- TU Darmstadt Ethics Approval EK 3/2025 — consent and data handling.
