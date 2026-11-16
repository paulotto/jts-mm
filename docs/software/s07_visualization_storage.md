<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# s07 — Visualization and Storage

This document covers two operator-facing concerns: the live viewer used during recording, and the on-disk file layout used for storage. The HDF5 schema details are referenced from [`s02_data_pipeline.md`](s02_data_pipeline.md) §8 and not repeated here; this doc focuses on what the operator sees, how files are organized on disk, and how replay works.

## 1. Live viewer

### 1.1 Goals

- Confirm at a glance that every modality is producing data.
- Detect electrode lead-off or saturation in the EMG stream.
- See the optical pose and the IMU pose side by side.
- Never block recording. The viewer **drops frames** if it cannot keep up; the writer is never starved.

### 1.2 Architecture

A web UI served from the Pi 5 by a FastAPI + WebSocket server (`jtsmm/viewer/server.py`; static SPA in `software/jtsmm/viewer_web/`). The browser is on the operator's laptop. Choosing a web UI over a native PyQtGraph window:

- No X server on the Pi 5 (we run Lite, no desktop).
- Operator's laptop is more powerful than the Pi for rendering.
- Trivially shareable: a second observer can open the URL.

The viewer is fed in-process by the viewer hub inside `jtsmm record` / `jtsmm view`. Updates are coalesced and sent as msgpack snapshots over the WebSocket.

### 1.3 Panels

```
+-----------------------------------------------------------+
| Header: session_id, elapsed, modalities OK/WARN, doctor   |
+--------------------------+-------------------------------+
| EMG (6 + spare)          | Camera preview tile 2x2       |
| live waveform per ch     | (one preview per camera)      |
| envelope overlay         | overlay markers + IDs         |
| lead-off indicators      |                               |
+--------------------------+-------------------------------+
| IMU MTA quaternion       | IMU CRA quaternion            |
| 3D arrow + numerics      | 3D arrow + numerics           |
+--------------------------+-------------------------------+
| Relative jaw-vs-cranium  | Sync diagnostics              |
| (from optical + IMU)     | per-clock residual histogram  |
| Euler angles vs. time    | trigger pulse count vs. frame |
+--------------------------+-------------------------------+
| Bottom: operator note bar (timestamp + free text)         |
+-----------------------------------------------------------+
```

The bottom note bar lets the operator type text markers during recording (e.g. "subject reported jaw click"). Under `jtsmm record --viewer` each note is stamped with the Pi host clock when the server receives it, mapped to `t_master_us` like a phase edge and written to `/events` (`kind = "note"`, browser display time in `viewer_t_s`) at the next flush (≤ 0.25 s), so a killed recorder keeps it. At close it is re-mapped with the final PPS fit. The UI says "note saved to session file". `jtsmm view` and `jtsmm replay` have no file to write: notes stay in server memory only and the UI says so. Read them with `Session.notes()`.

### 1.4 Performance

- EMG waveform: render at 30 Hz from ~500 display rows/s at any EMG rate (8× decimation at the 4 kHz default; sample-and-hold for max amplitude) so the operator sees a smooth display. Filters run at the configured rate.
- Camera preview: 5 Hz at 320 × 240 (resized on the Pi); not the full-rate stream.
- IMU 3D arrows: WebGL via three.js, very cheap.
- Total network bandwidth: ~5 Mbit/s, well within Ethernet.

## 2. Storage layout on disk

```
~/jtsmm_data/                              -- host.paths.data_dir (JTSMM_DATA_DIR overrides)
└── session_[<slug>_]<YYYYMMDDTHHMMSSZ>_<id8>/
    ├── session.h5                         -- main HDF5 file
    ├── integrity.json, integrity.txt      -- post-session integrity verdict
    └── reproduce.sh, reproduce.json       -- how to re-run this take
```

`<slug>` is the experiment name (if any), the timestamp is UTC, `<id8>` a random 8-character id. Video sidecars and a manifest (§2.1–§2.2) are target only.

### 2.1 Video file rotation *(target — not implemented)*

Video files are rotated when they exceed `storage.rotate_at_gb` (default 8 GB). Each new file gets a sequence suffix (`cam0_002.mkv`, etc.). The HDF5 `frame_idx` is then a `(video_file_index, frame_in_file)` tuple stored as `uint64` (top 16 bits = file index, bottom 48 bits = frame number).

### 2.2 Hashing and manifest *(target — not implemented)*

`manifest.yaml` lists every file in the session directory with size and SHA-256:

```yaml
session_id: abc12345_motion_battery_v1_001
created: 2026-06-15T10:30:00Z
files:
  - path: session.h5
    size_bytes: 524288000
    sha256: 7c8f9a2b...
  - path: cam0_001.mkv
    size_bytes: 5368709120
    sha256: 1a2b3c4d...
  - path: cam0_001.json
    size_bytes: 4194304
    sha256: 5e6f7g8h...
total_size_bytes: 6_201_000_000
```

The manifest is updated at session start (skeleton) and rewritten at session end with final hashes. This makes data transfer to another machine verifiable.

### 2.3 Hardware serial linkage *(target — not implemented)*

`/meta/hardware_serial` would identify the physical build; it is not written yet.

## 3. Replay

### 3.1 `jtsmm replay`

```
jtsmm replay ~/jtsmm_data/<session_dir>/session.h5 [--speed X] [--start S]
```

Opens a web viewer identical to the live one but reads from the HDF5. Supports:

- Pause / resume / speed control (0.01× to 64×).
- Jump to time / event.
- Cycle through detected EMG onsets.
- Toggle between optical and IMU-bridged pose for the kinematic display.

### 3.2 Programmatic access

The Python package exposes:

```python
from jtsmm.io import Session

with Session.open("session.h5") as s:
    t_us, uv = s.emg.window(0, 1000)      # EmgStream
    q = s.imu_mta.quat                    # IMU kinematics (QuatStream-compatible)
    trig = s.triggers                     # TriggerStream
```

Accessors return NumPy arrays. `/optical` and a feature-matrix helper are not exposed by the reader yet (read `/optical/*` with `h5py`).

### 3.3 Compatibility with the original JTS code

The original JTS code reads `/optical/*` only and is unmodified. Running it on a JTS-MM HDF5 produces the same kinematic output as the original system, modulo whatever differences come from the different optical front-end. This is the JTS-MM project's contract with the existing code base (no automated compatibility test yet).

## 4. Data retention and privacy

- HDF5 files contain `/meta/subject_id` as the **anonymized hash**, not any personal information.
- The mapping from real subject identity to subject hash lives in a separate **password-protected encrypted file** held by the study lead, in line with TU Darmstadt ethics approval EK 3/2025.
- Raw video files contain the subject's face. They are treated as personal data: stored on encrypted disks, transferred only over TLS, and deleted from working copies once the analysis is complete (per the consent form).
- A future v2 will support face-anonymization on the camera output (blur or replace with synthetic face) if the dataset is to be released publicly.

## 5. Quick-start session

The complete recipe for a typical session:

```bash
# On the operator's laptop:
ssh pi@jtsmm.local

# On the Pi:
jtsmm doctor                                   # pre-flight check
jtsmm record --subject SUBJECT_42 --protocol motion_battery_v1
# (recording starts; viewer URL printed on stderr)

# On the operator's laptop, open http://jtsmm.local:8770/
# Drive the protocol from the laptop's note bar.

# At the end:
# Ctrl-C in the SSH session; `jtsmm record` flushes and exits.

jtsmm integrity check ~/jtsmm_data/<session_dir>
```

## 6. Storage budget

| Hour of recording | EMG | IMU | Cameras (Track B, FFV1) | Total |
|-------------------|-----|-----|--------------------------|-------|
| 8 ch × 4 kHz (HDF5 ≈ 18 MB/min) | ≈ 1.1 GB | 110 MB | ~180 GB (3 cam) / ~120 GB (2 cam) | ~120–180 GB |
| 8 ch × 4 kHz (HDF5 ≈ 18 MB/min) | ≈ 1.1 GB | 110 MB | ~30 GB (Track A H.264) | ~31 GB |

A 1 TB NVMe holds ~6 hours of 3-camera Track B FFV1 recording, or ~30 hours of Track A H.264. For long pilot studies we recommend offloading session data to a NAS at the end of each day; verify the copy with `jtsmm integrity check` on the NAS side.

## 7. Schema versioning

The HDF5 `/meta/jts_schema_version` attribute drives compatibility. Conventions:

- `1.0` = original JTS schema (paulotto/jaw_tracking_system, IEEE SENSORS 2025).
- `1.1` = JTS-MM (this project): adds `emg`, `imu_*`, `cameras/*`, `triggers`, `sync`, `events`.
- `1.x` = backwards-compatible additions only. Any code reading 1.0 must still read 1.x.
- `2.0` = would be reserved for a breaking change (e.g. renaming `optical` → `kinematics`). We do not anticipate this in the project's lifetime.

A `jts_schema_version` check on open is planned (not implemented); `Session.open` does not check it yet.
