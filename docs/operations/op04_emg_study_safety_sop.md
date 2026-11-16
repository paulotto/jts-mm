<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# op04 — EMG Study Safety SOP (human subjects)

> **Purpose:** One-page standard operating procedure for ethics submissions and operator training.  
> **Device:** JTS-MM custom HAT (ADS1299 EMG front-end on Raspberry Pi 5).  
> **Electrical design basis:** [`h07_power_grounding_safety.md`](../hardware/h07_power_grounding_safety.md) §3.  
> **Session workflow:** [`op01_setup_checklist.md`](op01_setup_checklist.md), [`op02_subject_preparation.md`](op02_subject_preparation.md).

---

## 1. Scope and device classification

| Item | Detail |
|------|--------|
| **Use** | Non-invasive surface EMG (+ multimodal jaw tracking) on **healthy adult volunteers** in a university research study |
| **Classification** | **Research instrument** — not a medical device; **not** IEC 60601-1 certified |
| **Patient contact** | 6 bipolar EMG sites + 1 driven-right-leg (DRL) reference on the face/jaw; no needles, no broken skin |
| **Ethics protocol** | TU Darmstadt ethics approval EK 3/2025 (covers optical motion capture and surface EMG in general terms) |
| **Study lead / session operator** | *[Name]* / *[Name, trained on this SOP]* |

**Out of scope (do not use):** clinical diagnosis, treatment, vulnerable patients without separate approval, environments with electrosurgery / defibrillation / high-voltage procedures.

---

## 2. Electrical safety rationale (summary for ethics file)

- **Leakage limiting:** A **220 kΩ** series resistor in the DRL/bias path limits worst-case patient current to **≈23 µA** if the full **5 V** analog rail were applied through that path (see h07 §3). This is below the **100 µA** AAMI ES60601-1 type-BF patient-leakage figure cited for comparison — **we do not claim compliance**.
- **No galvanic isolation (v1):** The HAT shares a ground reference with the Pi 5. Mains isolation is **not** implemented in hardware; operational measures below reduce risk.
- **Star AGND/DGND tie:** PCB noise-control measure only; **not** a safety barrier (see h07 §2, h10 schematic review).

---

## 3. Subject exclusion criteria (verify before each session)

Stop enrollment / do **not** proceed if the subject has any of:

- Pacemaker, ICD, or other **active implanted electronic device**
- Pregnancy (unless explicitly approved in ethics protocol)
- **Open wounds, eczema, or dermatitis** at electrode sites
- Known **severe allergy** to Ag/AgCl electrodes or skin-prep products
- Inability to give informed consent
- Dental condition preventing safe MTA use (if MTA is used in the protocol)

Document exclusions in the study log. When in doubt, consult the study lead.

---

## 4. Pre-session electrical and hardware checks (operator)

Complete **before** attaching electrodes. Initial each line in the session log.

| # | Check | Pass criterion |
|---|--------|----------------|
| E1 | **Power** | Pi 5 powered from **USB-C PD battery bank** (≥27 W capable) **or** CE-marked PSU on bench **≥1 m from subject**; operator laptop **not** in physical contact with subject |
| E2 | **USB** | No mains-tied USB peripherals on Pi except via **isolated USB hub** if required |
| E3 | **Cables** | **Shielded patient leads** with touch-proof or snap connectors at the electrode; **no bare 2.54 mm pin headers** at the subject interface |
| E4 | **Visual** | HAT, cables, and electrodes intact — no frayed insulation, exposed copper, or liquid on electronics |
| E5 | **Leakage sanity** (weekly or after any repair) | With system **off**, DMM: resistance from any EMG snap to Pi USB shell **>1 MΩ**; with system **on**, no tingling when operator touches electrode snaps with dry fingers |
| E6 | **Firmware / doctor** | `jtsmm doctor` all green; `jtsmm emg check` lead-off functional |

---

## 5. Session setup (subject present)

1. Subject seated on **non-conductive** chair; remove **metal jewelry** near electrodes.
2. Skin prep and electrode placement per [`op02`](op02_subject_preparation.md) §2–§3; **fresh single-use** Ag/AgCl electrodes.
3. DRL on **mastoid** (or approved alternative in op02 §3.4).
4. Route cables with strain relief at collar; subject confirms **no tugging** (op02 §6–§7).
5. Operator station **separate** from subject (Wi‑Fi/SSH to Pi); live viewer on operator display only.
6. Brief subject: *“Tell me immediately if you feel tingling, burning, or discomfort.”*

---

## 6. During recording

- Operator remains within **line of sight** of the subject.
- If subject reports **tingling, pain, or skin burning** → **STOP** (§7).
- Do not attach or detach electrodes while recording is active.
- Maximum continuous wear per protocol; if skin irritation appears, **end session** and document (op02 §8).

---

## 7. Emergency stop and incident response

| Situation | Action |
|-----------|--------|
| **Subject discomfort / tingling** | Press **Stop** in `jtsmm record` or pull Pi power; remove electrodes; inspect skin; log incident |
| **Equipment fault (smoke, heat, smell)** | Disconnect **battery/PSU**; evacuate if needed; do not reuse HAT until inspected |
| **Medical emergency** | Call local emergency number; do not remove MTA if airway concern — follow first-aid training |
| **Suspected electrical fault** | Tag HAT **out of service**; study lead + electrical check before reuse |

**Incident log fields:** date/time, subject ID, symptoms, equipment state, action taken, study lead notified (Y/N).

---

## 8. End of session

Per op02 §8: stop recording → remove MTA (if used) → CRA → EMG electrodes → inspect skin → disinfect reusable CRA band.

---

## 9. Limitations (must appear in consent form / paper Methods)

- Device is a **non-certified research prototype** without reinforced galvanic isolation from mains.
- Not for clinical diagnosis or therapy.
- Residual risk from extremely unlikely PSU fault; mitigated by current limiting and operational setup (§4).
- Future hardware (isolated DC-DC + isolated SPI) documented in h07 §4 — **not** required for this study if ethics accepts §2–§4 mitigations.

---

## 10. Operator sign-off (per session)

| Field | Entry |
|-------|--------|
| Date | |
| Subject ID | |
| Operator (print) | |
| E1–E6 completed (Y/N) | |
| Exclusion criteria cleared (Y/N) | |
| Incidents (Y/N — describe if Y) | |
| Study lead review required (Y/N) | |

---

## References

- [`h07_power_grounding_safety.md`](../hardware/h07_power_grounding_safety.md) — leakage analysis, operational rules
- [`h10_schematic_logical_review.md`](../hardware/h10_schematic_logical_review.md) — record of the schematic review (findings and changes)
- [`op01_setup_checklist.md`](op01_setup_checklist.md) — full session checklist
- [`op02_subject_preparation.md`](op02_subject_preparation.md) — electrode placement
