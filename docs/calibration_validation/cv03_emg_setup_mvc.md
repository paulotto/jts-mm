<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# cv03 — EMG Setup and MVC Calibration

This document specifies the per-subject EMG setup procedure: skin preparation, electrode placement, signal-quality checks, and the Maximum Voluntary Contraction (MVC) protocol used to normalize EMG amplitudes across subjects and sessions.

## 1. Why MVC normalization

Surface EMG amplitudes vary by 5–10× between subjects even for the same muscle and effort. Within a subject, they also drift between sessions (electrode placement, hydration, skin impedance, etc.). Normalizing each channel's signal to a reference percentage of MVC makes signals comparable across subjects and sessions, and is the SENIAM-recommended practice for surface EMG.

## 2. Skin preparation (operator-facing)

For each electrode site:

1. Identify the anatomical landmark (see [`../operations/op02_subject_preparation.md`](../operations/op02_subject_preparation.md) §3).
2. **Clean** the skin with alcohol wipe; let it dry.
3. **Abrade** lightly with Nuprep abrasive gel using a cotton swab in small circles; this removes the top layer of dead skin. Stop when the skin shows light redness.
4. Wipe off the gel with a fresh alcohol pad; let dry.
5. Apply the pre-gelled Ag/AgCl electrode firmly; press for 5 s to ensure good contact.
6. Connect the lead wire to the electrode snap.
7. Verify electrode-skin impedance via the firmware's lead-off detection (see §5).

Repeat for all six EMG electrodes plus the DRL/bias electrode (mastoid).

## 3. Electrode placement (per muscle)

**Channel → muscle mapping:** see [an03 §1](../analysis/an03_emg_processing.md#1-montage), the
one montage table of the project: CH1 masseter_r, CH2 temporalis_r, CH3 digastric_r,
CH4 digastric_l, CH5 masseter_l, CH6 temporalis_l, as the cables are labelled (hardware check
2026-11-03). Sessions recorded before stored CH3/CH4 as digastric_l/digastric_r; the readers
correct that on read (an03 §1). The channel numbers formerly given in the headings below were an older plan and
were not used in the recordings.

### 3.1 Masseter (L/R)

- Subject clenches; palpate the masseter belly (visibly bulges).
- Place the active electrode in the **center of the muscle belly, ~2 cm above the mandibular angle**.
- Place the reference electrode 1.5 cm anterior along the muscle fiber direction (the fibers run vertically with a slight anterior tilt).
- Inter-electrode distance: about 20–30 mm centre to centre (the two snap electrodes of a channel placed next to each other; as recorded per the photographs; SENIAM recommends 20 mm).

### 3.2 Anterior Temporalis (L/R)

- Subject clenches; palpate the temporalis at the temple.
- Place the active electrode **~2 cm posterior to the frontozygomatic suture, in the muscle belly**.
- Place the reference electrode 1.5 cm superior along the fiber direction.
- Fibers are roughly vertical; align the electrode pair accordingly.

### 3.3 Anterior Digastric (L/R)

- Subject opens against resistance to activate the digastric.
- Locate the submandibular triangle. Place the active electrode **1 cm lateral to the midline, in the anterior belly**.
- Reference electrode 1.5 cm anterior along the fiber direction.
- This is the trickiest placement; cross-talk from platysma and mylohyoid is common.

### 3.4 DRL/bias — separate jack

- Place on the **mastoid process** (the bony bump behind the ear) or earlobe.
- Bony locations minimize muscle artifact in the common-mode reference.

A diagram with photographs lives in `docs/operations/op02_subject_preparation.md` §3 to keep this doc free of subject images.

## 4. Cable management

- Run cables behind the subject; tape with hypoallergenic Micropore tape on the cheek and behind the ear.
- Bundle electrode leads at the **collar** (production) or keep **short** for bench bring-up (Option 0). See [`h03`](../hardware/h03_emg_frontend.md) §10.2.
- Provide enough slack at the electrode end (~5 cm "service loop") so head motion does not tug on the electrode.

## 5. Signal-quality checks before recording

The operator runs `jtsmm emg check` before starting the recording. It runs lead-off + |Z|, a DRL isolation capture (CH1–CH6 muted), then baseline noise and line frequency:

### 5.1 Lead-off detection

Enables the ADS1299's internal lead-off detection for 5 seconds and reports per-channel pass/fail. A channel fails on a rail-stuck input, a floating input, or |Z| above 50 kΩ. A lead-off comparator trip is reported as a WARN only (not yet bench-verified on this board), and comparator bits are ignored unless the MCU reports its comparators powered (older firmware left them off).

### 5.2 Baseline noise

With the subject relaxed, records 5 s of EMG with lead-off disabled and reports per-channel RMS noise in the 20–450 Hz band. Acceptable: <10 µV RMS per channel for masseter / temporalis; <15 µV RMS for digastric (often noisier due to submandibular skin movement).

### 5.3 Line-frequency check

Computes the 50 Hz amplitude in the spectrum. Acceptable: <50 µV peak at 50 Hz. If higher, suspect grounding issues, nearby switching electronics, or a missing DRL connection.

If any check fails, the operator re-preps the offending channel and re-runs `jtsmm emg check`.

For a quantitative skin impedance estimate (kΩ per channel, from the ADS1299 lead-off tone at `f_SPS/64`), run `jtsmm emg impedance` or toggle **|Z|** in the live viewer when `leadoff.during_recording=true` (bench profile).

## 6. MVC protocol

We elicit MVC for each muscle group separately:

> **As run in the 2026 dataset:** the closers were clenched with the teeth in contact, **without cotton rolls**. The MVC take has one `mvc_closing` and one `mvc_opening` phase of about 10 s each, not three 3-s trials with 60 s rest. The DRL / bias electrode was on the **forehead**, not on the mastoid (§2). The steps below are the original plan.

### 6.1 Bilateral masseter / temporalis (bite)

1. Subject sits upright with a sterile dental cotton roll between molars.
2. Operator instructs: "Bite down as hard as you can for 3 seconds, then relax."
3. Repeat 3 trials with 60 s rest between.
4. Record the peak envelope per channel; the MVC is the 95th percentile of peak envelope across trials (more robust than max-of-max).

The masseter and temporalis activate together during clench; their MVC values come from the same trial.

### 6.2 Bilateral anterior digastric (forced opening)

1. Subject opens their mouth against a manual resistance (operator gently pushes upward on the chin with a tongue depressor wrapped in gauze).
2. Operator instructs: "Try to open your mouth as hard as you can for 3 seconds against my resistance, then relax."
3. Repeat 3 trials with 60 s rest.

Care: the chin push must be firm but not painful; aim for ~10 N maximum.

### 6.3 Storage

On **`jtsmm record --mvc`**, two SPACE intervals are stored on `/events` and `/phases` with `kind: mvc` (`mvc_opening`, `mvc_closing`). Cameras and ArUco stay off so the take is EMG-only (MCU IMU/triggers still stream). This is independent of `session.experiment`. Offline, slice `/emg` to those `t_master_us` bounds and compute per-channel MVC (e.g. 95th percentile of the envelope). A dedicated `/calibration/mvc/` group is not written by the recorder.

```
/phases  name=mvc_opening|mvc_closing  kind=mvc  t_start_us .. t_end_us
/emg/samples_uv   (full session; use phase bounds)
```

## 7. Validation acceptance criteria

The MVC calibration is accepted when:

- [ ] All 6 channels' MVC values are above the noise floor (clearly identifiable peaks vs. baseline).
- [ ] The three MVC trials per muscle group agree within ±20 % (otherwise repeat).
- [ ] Lead-off detection passes on all 6 channels at the start and end of the MVC session.

## 8. Operator notes

- The MVC protocol is performed **before** the main motion-battery acquisitions, with the same electrodes in place; the electrodes are not removed and re-applied between MVC and recording.
- If the electrode falls off mid-session, the operator stops the recording and re-preps; MVC may need to be re-done if a masseter, temporalis, or digastric electrode is replaced (the MVC scale is electrode-position-dependent).
- The 95th percentile (rather than max) is used because brief motion artifacts at the peak of MVC can create spurious "peak" samples that misrepresent the true muscle activation.

## 9. References

- SENIAM project recommendations for surface EMG of the masseter and temporalis: <http://www.seniam.org/>.
- Castroflorio T., Bracco P., Farina D. — *"Surface electromyography in the assessment of jaw elevator muscles"*, J Oral Rehabil 35(8) 2008.
- Burnett D. R. — *"Electromyography of the digastric muscle during reflexive and voluntary movements"*, J Dent Res 56(8) 1977.
- Halaki M., Gi K. — *"Normalization of EMG signals: to normalize or not to normalize and what to normalize to?"*, in *Computational Intelligence in Electromyography Analysis*, IntechOpen 2012.
