<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# jtsmm — production Pi 5 host package

Multimodal capture for JTS-MM: MCU USB reader, dual-camera trigger sync, HDF5 sessions, YAML config from [`schema/v1/`](../../schema/v1/).

**Status:** production host stack (Pi 5).

| Package | Role |
|---------|------|
| `jtsmm/config/` | Load monolithic YAML, strict validate, registry accessors, MCU `config apply` JSON |
| `jtsmm/protocol/` | `0x55 0xAA` frame encode/decode |
| `jtsmm/capture/` | MCU reader, cameras, [`OpticalTrackBridge`](jtsmm/capture/optical_bridge.py) for `--optical-track` |
| `jtsmm/native/` | C++/libusb capture + libhdf5 writer (required for real, non-mock recording) |
| `jtsmm/sync/` | Clock alignment (`t_master_us`) |
| `jtsmm/io/` | HDF5 writer, session open/close |
| `jtsmm/cli/` | `jtsmm record`, `ct-calibrate`, `trigger`, `config {init,validate,audit,show,apply}`, `emg {check,impedance}`, `imu calibrate`, `doctor`, `info`, `integrity`, `view`/`replay` |
| `jtsmm/viewer/` | Live/replay/mock web viewer (EMG + IMU kinematics + cameras) |
| `jtsmm/imu/` | BNO085 int16→SI units and derived jaw kinematics |

## Live viewer / replay

`jtsmm view` (live), `jtsmm replay <session.h5>`, or the mock source serve a web
UI with EMG traces, per-IMU orientation (fusion quaternion) and
accelerometer / gyroscope / magnetometer / linear-acceleration strips, and a
shared 3D dual-body (cranium + mandible) scene.

## Camera trigger only

Arm/disarm the MCU PIO trigger **without** applying config or starting EMG/IMU:

```bash
jtsmm trigger                 # trig_start using MCU's last-applied rate/pulse
jtsmm trigger --rate 30       # override rate for this start
jtsmm trigger --rate 100 --width-us 2000
jtsmm trigger --status
jtsmm trigger --stop
```

> **The 3D "estimated translation" trail is a drift-prone IMU estimate, not
> optical ground truth.** IMUs give orientation directly but not position;
> the relative translation is a short-term ZUPT-bounded double-integration of
> linear acceleration (auto-resets, labeled "est."). Use the optical/camera
> pipeline for measured position.

## IMU sample-rate validation

```bash
# apply a preset over J_USB1, capture, and report measured-vs-configured Hz + drops
jtsmm doctor --port /dev/jtsmm_hat imu-rates --all-presets --duration 10 \
    --report imu_rates_$(date +%F).md
# offline fallback from a recorded session:
python -m jtsmm.tools.imu_rate_report session.h5

# MP world-frame X/Y/Z from /optical (or bodies/MP):
python -m jtsmm.tools.plot_mp_trajectory session.h5
python -m jtsmm.tools.plot_mp_trajectory session.h5 --save mp_xyz.png --no-show
python -m jtsmm.tools.plot_mp_trajectory session.h5 --no-cycles --save mp_xyz.png --no-show

# Optical lock timeline and working volume:
python -m jtsmm.tools.eval_optical_session session.h5 --no-show
# Same lock / SE(3) / drop budgets as jtsmm integrity (exit 2 on errors or warnings):
python -m jtsmm.tools.eval_optical_session session.h5 --no-show --strict
jtsmm integrity check --strict session.h5   # EMG + IMU + cameras + /optical
```

## Install

Rig (Pi 5): [`scripts/install_pi.sh`](../../scripts/install_pi.sh). Development (repo root, in a venv):

```bash
pip install -e software/jtsmm -e software/optical
python -c "import jtsmm; print(jtsmm.__version__)"
```

Operator config: edit **`~/jtsmm.yaml`** — one file with every tunable parameter. Template: [`etc/jtsmm.yaml`](etc/jtsmm.yaml), installed via [`scripts/install_pi.sh`](../../scripts/install_pi.sh) / `build_install.sh --config`.

```bash
jtsmm config validate          # strict check before sessions
jtsmm config audit --strict    # parameter coverage gate
jtsmm config show --mcu-json   # inspect MCU apply payload
```

See [`docs/software/s08_runtime_configuration.md`](../../docs/software/s08_runtime_configuration.md) and [`docs/operations/op06_config_hardware_validation.md`](../../docs/operations/op06_config_hardware_validation.md).

## Native capture backend

Real recording always uses the C++/libusb USB capture (`NativeCapture`) and errors without the build.
HDF5 goes through the native **libhdf5** writer (`NativeWriter`) unless `--optical-track`, CT
calibration or an experiment plan is active (then the Python `Hdf5Writer`). `host.capture.native_prefer: false`
only selects the Python writer. `JTSMM_NATIVE=0` works for `--mock` only; a real recording then errors.

```bash
# system build deps (Debian/Ubuntu/Pi OS)
sudo apt install build-essential cmake ninja-build libusb-1.0-0-dev libhdf5-dev

# build the extension into jtsmm/native/
pip install -e ".[native]"        # pulls pybind11
./native/build.sh --python "$(command -v python)"

python -c "from jtsmm import native; print('native available:', native.is_available())"
```

## Optical tracking in a multimodal session

```bash
jtsmm record --optical-track \
  --optical-config software/optical/etc/optical.yaml
```

Or set under `session:` in `jtsmm.yaml`: `optical_track`, `optical_config`
(path to `optical.yaml`), `use_experiment_plan`, `experiment` (inline plan),
optional `experiment_plan` path override, `write_jts_config`, `ct_calibration`,
`ct_calibration_n_poses`, `ct_tip_calibration`.

ArUco runs on a half-res picamera2 **lores** stream so camera metadata and `/optical/` stay near trigger rate.
Poses that fail PnP RMS / Kabsch / SE(3) are stored as NaN; per-row `n_tags` / RMS live under `/optical/quality/`.
See [`docs/software/s03_camera_pipeline.md`](../../docs/software/s03_camera_pipeline.md), [`docs/software/s02_data_pipeline.md`](../../docs/software/s02_data_pipeline.md) §8.5, and [`software/optical/README.md`](../optical/README.md).

## CT tip pivot calibration

The calibration tool tip is removable, so its body-frame offset varies. With a
calibration cone:

```yaml
session:
  ct_calibration: true          # run cone + SPACE×N once (writes tip JSON)
  ct_calibration_n_poses: 20
  ct_tip_calibration: software/optical/etc/ct_tip_calibration.json
  persist_ct_calibration: false
  ct_calibration_recordings_dir: ~/experiments/5_2_1_pivot_calibration_repeatability/recordings
  persist_imu_calibration: false
  imu_calibration_recordings_dir: ~/experiments/5_2_2_imu_optical_handeye/recordings
  optical_config: software/optical/etc/optical.yaml
  optical_track: true
```

Or calibrate without a motion take:

```bash
jtsmm ct-calibrate
jtsmm ct-calibrate --persist-ct-calibration
```

Later takes (no re-calib): set `ct_calibration: false` and keep the same
`ct_tip_calibration` path — the tip offset is loaded and applied to
`/optical/pose_ct`. Each calib also writes a copy into the session dir as
`ct_tip_calibration.json`. With `persist_ct_calibration: true` (or
`--persist-ct-calibration`), SPACE poses and camera frames are also written
under `ct_calibration_recordings_dir` (default
`~/experiments/5_2_1_pivot_calibration_repeatability/recordings`, leaf folder
`{experiment_slug}_{timestamp}`).

## IMU mounting calibration (hand-eye)

`jtsmm imu calibrate --mode record` captures optical + IMU figure-eights (EMG
off) and fits `R_mount`. With `persist_imu_calibration: true` (or
`--persist-imu-calibration`), the session plus `imu_mounting.json` (`R_mount`,
`n_pairs`, RMS, `run`) are copied under `imu_calibration_recordings_dir`
(default `~/experiments/5_2_2_imu_optical_handeye/recordings`). Extra takes of
the same rigid pair feed the **repeatability** eval — one operational
cal after each remount is enough. Eval:
`python experiments/5_2_2_imu_optical_handeye/eval_handeye.py` (from the repository root).

## Experiment phases (interactive SPACE)

In `jtsmm.yaml`:

```yaml
session:
  use_experiment_plan: true    # false → duration-only capture, no /phases
  mvc: false                   # true → EMG MVC take (or pass --mvc)
  experiment:
    name: subject01_session1
    description: Jaw motion analysis experiment
    phases:
      - open_close
      - left_right
      - protrusion_retrusion
      - chewing
```

```bash
# Motion battery + landmark calibration
jtsmm record --use-experiment-plan \
  --optical-track \
  --optical-config software/optical/etc/optical.yaml

# Separate EMG MVC take (opening then closing SPACE)
jtsmm record --mvc
```

Or `--no-use-experiment-plan` to record without walking phases even if the YAML switch is on. Optional `--experiment-plan <json>` overrides `session.experiment`. `--mvc` cannot be combined with `--use-experiment-plan`.

Capture **keeps running** between SPACE presses on a motion plan. Landmark digitization is inserted before motion. MVC is a **separate** `--mvc` recording (EMG only: cameras and ArUco off). Edges are stamped onto MCU `t_master_us` under `/events` and `/phases`. The default 10 s timed-capture duration does **not** apply while walking phases; optional `--duration` is a safety timeout only. Optional JTS sidecar: `<session>/<slug>_jts_config.json` (motion/optical takes).

C/C++ layout and Doxygen comment conventions: [`docs/software/s09_c_cpp_api_comments.md`](../../docs/software/s09_c_cpp_api_comments.md) (`cd native && doxygen Doxyfile`).
