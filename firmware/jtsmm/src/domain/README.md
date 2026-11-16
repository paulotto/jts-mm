<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# domain/ — catalogs, profile resolution, per-subsystem config apply

`config_defaults.c` (boot defaults), `emg_config.c` (+ `include/jtsmm/domain/emg_config_check.h`), `imu_config.c` (+ `imu_timing.c`: IMU sample timestamps), `bno085_reports.c`, `trigger_config.c`.

Apply order (via `runtime/config_apply.c`): platform → emg → imu_mta → imu_cra → trigger.
