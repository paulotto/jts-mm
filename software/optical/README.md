<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# Optical: replacing Qualisys jaw tracking with ArUco camera tracking

| What | Where |
|------|-------|
| **Package** | `software/optical/` — installed by [`scripts/install_pi.sh`](../../scripts/install_pi.sh); manually: `sudo apt install python3-picamera2`, then `pip install -e software/optical` in a `--system-site-packages` venv |
| **Guides** | [`docs/optical/`](../../docs/optical/) |
| **Config** | [`software/optical/etc/`](etc/) — `optical.yaml` (auto 1/2-cam), `ct_tip_calibration.json`, `jts_config.json`; optional `~/optical.yaml` |
| **Artifacts** | `data/optical/` (gitignored; created by the first `optical track` / `optical calibrate` run) |

| Doc | Subpackage | Purpose |
|-----|------------|---------|
| [01 — Marker setup](../../docs/optical/01_marker_setup.md) | `optical.markers` | Dictionary, ID plan, printable tags, mount transforms |
| [02 — Calibration](../../docs/optical/02_calibration_and_frames.md) | `optical.calibration` | Intrinsics/extrinsics capture, HDF5 calibration bundle |
| [03 — Pose estimation](../../docs/optical/03_pose_estimation_and_output.md) | `optical.tracking` | Detection → PnP/triangulation → HP/MP/CT poses → HDF5 + quality/integrity |
| [04 — JTS migration](../../docs/optical/04_migration_from_qualisys.md) | `software/jaw_tracking/` | Feed tracker output into JTS for jaw kinematics analysis |

## Config (`software/optical/etc/`)

One YAML per rig profile (like `jtsmm.yaml`):

- `optical.yaml` — auto-selects 1-cam (PnP) or 2-cam (triangulation) profile at runtime

Sections: `markers`, `calibration`, `tracker`, `experiment`, `eval`, `eval_quality`, plus `paths` / `rig_id`.

## Quick start

```bash
optical markers --out-dir data/optical/markers_out
optical calibrate --mode intrinsics
optical calibrate --mode extrinsics   # writes extrinsics cache + calibration_bundle.h5
# optical calibrate --mode bundle     # re-export from caches without recapture

# Standalone tracking (experiment: from optical.yaml; optional --experiment-plan override)
optical track

# Persist session.h5 + lores frames (name/condition from optical.yaml experiment:)
# With use_experiment_plan: false → one SPACE take, no landmark calib
optical track --persist-frames
optical track --persist-frames --condition near

# Unified multimodal session
jtsmm record --optical-track \
  --optical-config software/optical/etc/optical.yaml \
  --duration 1800

optical eval-quality --save-frames
```

Downstream analysis uses the vendored JTS package in [`software/jaw_tracking/`](../jaw_tracking/README.md).

## Status

- **`optical track`:** MCU `TRIGGER_EVENT`s → per-frame `t_master_us` in the standalone tracking HDF5 (picamera2 + `jtsmm.sync.ClockAligner`). `--backend v4l2` skips the HAT pulse and master-clock stamps.
- **`jtsmm record --optical-track`:** same tracker into multimodal `session.h5` `/optical/` (poses + `quality/`). Uses a half-res picamera2 **lores** stream for ArUco (full-res `main` stays metadata-only) via [`OpticalTrackBridge`](../jtsmm/jtsmm/capture/optical_bridge.py). After a take: `jtsmm integrity check --strict` or `python -m jtsmm.tools.eval_optical_session session.h5 --strict`. See [`docs/software/s03_camera_pipeline.md`](../../docs/software/s03_camera_pipeline.md) and [`docs/software/s02_data_pipeline.md`](../../docs/software/s02_data_pipeline.md) §8.5.
- Calibration, tracking, and recording leave IMX296 external trigger enabled so the HAT remains the timing authority.
- Live `optical` commands (`calibrate` except `--mode bundle`, `track`, `eval`, `eval-quality`) run `jtsmm trigger --rate <mcu.trigger.rate_hz>` (jtsmm.yaml, default 30) and `jtsmm trigger --stop` when they finish. `track` arms the pulse only after picamera2 is streaming (as `jtsmm record` with `trigger_after_cameras`), starts no EMG/IMU stream, and pairs the two cameras by `SensorTimestamp` (≤ 4 ms apart, `DEFAULT_MAX_PAIR_SKEW_NS`).
- **`optical eval` / `optical eval-quality`:** choose marker size and lores resolution before changing the recorder.
