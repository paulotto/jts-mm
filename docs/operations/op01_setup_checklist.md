<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# op01 — Per-Session Setup Checklist

This is the **operator-facing checklist** for running a full multimodal recording session. Print it, laminate it, and keep it on the bench. The operator goes through it top-to-bottom every session. The whole procedure takes ~30 minutes for a trained operator (~45 minutes for the first few sessions).

The checklist assumes the hardware is fully built, calibrated, and the subject has consented per the TU Darmstadt ethics approval EK 3/2025. Electrical safety rules: [`op04_emg_study_safety_sop.md`](op04_emg_study_safety_sop.md).

## 1. T-30 — Pre-arrival

Done *before* the subject arrives:

- [ ] Lab temperature stable (within 2 °C of last calibration).
- [ ] Pi 5 on **27 W (5 V / 5 A)** PSU; `vcgencmd get_throttled` → `0x0` (see [`s01` §5](../software/s01_pi5_software_stack.md#5-pi-5-power-and-cpu-scaling)).
- [ ] Pi 5 on, NVMe drive has >100 GB free: `df -h ~/jtsmm_data`.
- [ ] No mains-powered laptops connected to Pi 5 USB (use the isolated hub for any peripheral).
- [ ] Cameras in their tripod positions; do not move during the session.
- [ ] **AprilTag default:** diffuse scene lighting on (LED panel or room lights); exposure/gain locked in `jtsmm.yaml`; **no IR rings** on cameras unless running optional `marker_detector: retroreflective`.
- [ ] *(Track A only, or if sync LED populated)* Visible **sync LED** (`SYNC_LED` JST) in all camera FOVs.
- [ ] HAT installed on Pi 5, USB connected, `/dev/jtsmm_hat` present: `ls -l /dev/jtsmm_hat`.
- [ ] `jtsmm config audit --strict` — parameter coverage OK on `~/jtsmm.yaml` ([`op06`](op06_config_hardware_validation.md)).
- [ ] Camera trigger rate: `mcu.trigger.rate_hz: 40` and `host.cameras.list[].trigger.expect_rate_hz: 40` on both cameras (study default, same as the Oct 2026 pilot). Do not change it between participants ([`h01`](../hardware/h01_camera_subsystem.md) §3.3).
- [ ] Run `jtsmm doctor` — all subsystems should report green.
- [ ] Latest firmware loaded on the HAT: `jtsmm info | grep firmware`.
- [ ] IMU breakouts mounted on MTA and CRA; cables routed to the HAT.
- [ ] Patient cable on the HAT; ready to receive electrodes.
- [ ] Disposable electrodes (×7 plus a few spares), skin-prep kit, hypoallergenic tape, gloves prepared on the bench.
- [ ] Camera calibration not stale (>30 days): verify `data/optical/calibrations/<rig_id>/calibration_bundle.h5` exists and matches [`optical.yaml`](../../software/optical/etc/optical.yaml) (see [`docs/optical/02_calibration_and_frames.md`](../optical/02_calibration_and_frames.md)).
- [ ] For **multimodal** sessions with ArUco: `jtsmm record --optical-track --optical-config software/optical/etc/optical.yaml` (half-res lores; poses in `session.h5` `/optical/`). Add `--use-experiment-plan` (or set `session.use_experiment_plan: true`) to walk SPACE phases from `session.experiment`.
- [ ] For **optical-only** kinematics: `optical track` (default `optical.yaml`); writes MCU `t_master_us` in the standalone tracking HDF5.

## 2. T-15 — Subject arrival and consent

- [ ] Greet subject; review the consent form they signed at enrollment.
- [ ] Verify subject is not wearing dental appliances that would interfere with the MTA.
- [ ] Confirm subject does not have a pacemaker, defibrillator, or other active implant (EMG is non-invasive and very low current but caution is policy).
- [ ] Subject sits in the recording chair, head positioned roughly in the cameras' shared FOV.

## 3. T-10 — Hardware on subject

Follow [`op02_subject_preparation.md`](op02_subject_preparation.md) in detail; the high-level steps:

- [ ] Skin-prep + electrode placement on the six EMG sites + DRL on mastoid.
- [ ] Verify electrode contact: `jtsmm emg check` — all 6 channels pass lead-off.
- [ ] Attach the CRA (forehead band, with IMU breakout) and verify it does not slip.
- [ ] Attach the MTA (mouthpiece, with IMU breakout) using dental adhesive per the published procedure.
- [ ] Run a quick optical check: `jtsmm view` (camera previews) and verify all **AprilTags** on MTA + CRA are detected in all cameras (`marker_detector: apriltag`).

## 4. T-5 — IMU and calibration check

- [ ] If this IMU–array pair has never been mount-calibrated (or was re-glued): `jtsmm imu calibrate --mode record --persist-imu-calibration` (optical + IMU figure-eights, EMG off), then confirm `/calibration/imu_*/mounting_transform` and the persist leaf under `~/experiments/5_2_2_imu_optical_handeye/recordings` (see [`cv02`](../calibration_validation/cv02_imu_calibration.md) §6). Once per rigid pair, not per subject. Do **not** use a plain `jtsmm record --experiment-plan` for this — ArUco stays off unless `session.optical_track` is true.
- [ ] Subject sits still for 5 s; BNO fusion accuracy reaches `high` (CDC `imu_status`). Gyro bias / rate checks are on-chip / session hygiene, not a new `R_mount`.
- [ ] Subject moves head slowly through a small figure-eight; verify both IMU quaternions update smoothly. After a valid mounting cal, optical vs mounted IMU attitude should agree within ~3° RMS on a similar tumble.
- [ ] Subject opens / closes mouth a few times; verify all 6 EMG channels show activation pattern.
- [ ] DC lead-off stays on during recording (`leadoff.during_recording: true`); set it to `false` or pass `--no-leadoff` only if you need it off.

## 5. T-2 — Sync check and final sanity

- [ ] `jtsmm doctor` is green; `jtsmm doctor imu-rates` if IMU rates are in doubt. Sync residuals and stream rates are checked after the take by `jtsmm integrity check <session_dir>`.
- [ ] Storage doctor: confirm >50 GB free on NVMe (for ~30 min of Track B recording).
- [ ] Operator's laptop has the live viewer open at `http://jtsmm.local:8770/`.

## 6. T=0 — MVC trials (if first session of the day)

If MVC trials have not been recorded today:

- [ ] Record MVC with `jtsmm record --mvc` (not as names in `session.experiment.phases`). Two **EMG-only** SPACE phases (`mvc_opening`, `mvc_closing`; cameras/ArUco off).
- [ ] Opening MVC (`mvc_opening`): digastric against resistance — 3 trials, 3 s each, 60 s rest (or one long SPACE interval covering the set).
- [ ] Closing MVC (`mvc_closing`): masseter/temporalis bite — 3 trials, 3 s each, 60 s rest.
- [ ] Confirm EMG envelopes look like MVC (not noise floor, not rail).

If MVC intervals are already in today's session `/phases`, skip this block and continue with landmark + motion phases (or a new take).

## 7. Main recording

- [ ] `jtsmm record --subject SUBJ_ID --protocol motion_battery_v1`.
- [ ] `jtsmm record` prints readiness once all streams are healthy (or use the web viewer URL on stderr).
- [ ] Operator starts recording; capture runs.
- [ ] Landmark digitizing (the eight calibration phases before the first motion phase: six teeth, then `tmj_left` / `tmj_right` on the skin-palpated condyles). Hold the CT tip still for 2–5 s between the two SPACE presses. The prompt shows what comes next. Left/right are the **participant's**:

  | Phase | Tooth |
  |---|---|
  | `mand_point_1` | lower central incisor (incisal edge, midline) |
  | `mand_point_2` | lower left second molar |
  | `mand_point_3` | lower right second molar |
  | `max_point_1` | upper central incisor (incisal edge, midline) |
  | `max_point_2` | upper left second molar |
  | `max_point_3` | upper right second molar |

  The incisor comes first because JTS puts the landmark origin at the first point (`coordinate_origin_index: 0`). Keep the same molar on every participant; if one cannot be reached, use a note event to record the tooth used.
- [ ] Operator drives the motion protocol from the laptop's web UI:
  - 30 s opening + closing.
  - 30 s left-right lateral excursion.
  - 30 s protrusion + retrusion.
  - 30 s cyclic chewing (no food; mime).
  - 30 s rest.
- [ ] At any moment, use the note bar to inject events ("subject reports jaw click", "left motion harder than right", etc.).
- [ ] If something goes wrong, stop capture (Ctrl-C or viewer Stop); `jtsmm record` flushes the partial HDF5 and exits cleanly. Restart with a new session ID.

## 8. T+session — Wrap-up

- [ ] Stop the recording (Ctrl-C in the SSH session or Stop in the UI).
- [ ] `jtsmm integrity check <session_dir>` — verify all expected modalities are present in the HDF5.
- [ ] Optical take: `jtsmm integrity check --strict <session_dir>` (or `python -m jtsmm.tools.eval_optical_session <session.h5> --strict`) — lock / SE(3) / phase floors ([`s02` §8.5](../software/s02_data_pipeline.md), [`op03`](op03_troubleshooting.md) §5b).
- [ ] Carefully remove the MTA (use the dental-adhesive release agent if applicable).
- [ ] Remove the CRA.
- [ ] Remove the EMG electrodes (gentle peel; warn the subject of the slight pull on facial hair).
- [ ] Dispose of single-use electrodes in the regulated-waste bin.
- [ ] Subject debriefing: ask about discomfort, jaw fatigue, electrode irritation.
- [ ] Backup the session directory to the NAS: `rsync -av ~/jtsmm_data/<session_dir>/ <nas>:/jtsmm/data/<session_dir>/`.

## 9. End of day

- [ ] All sessions passed `jtsmm integrity check`.
- [ ] Storage disk usage: report to `docs/operations/log.csv` (a simple CSV row).
- [ ] Power down the Pi 5: `sudo shutdown -h now`.
- [ ] Scene lighting off (LED panels); cameras power down with the Pi.

## 10. Things to remember

- The HAT-to-electrode patient cable should **never** be left dangling near floor-standing equipment. A pulled cable can pull on an electrode and irritate the subject's skin.
- The dental adhesive used for the MTA has a working time of ~2 min and an open time of ~8 min. Plan electrode placement before applying.
- Keep food / drink off the bench; spilled coffee into a Pi 5 HAT is an immediate session-ender.
- Take a 5-minute break between subjects for the operator's own sake; rushing causes errors.
- If anything feels off (a strange smell, an unexpected reset, a marker not detected after a known fix), stop and run `jtsmm doctor`. The 5 minutes you save by skipping the check are not worth losing a session.
