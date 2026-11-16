<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Config hardware validation (op06)

Hardware validation matrix for the monolithic `jtsmm.yaml` contract on a Pi + HAT bench rig.

| Step | Command | Pass criteria |
|------|---------|---------------|
| H1 | `jtsmm config audit --strict` | exit 0 on rig's `~/jtsmm.yaml` |
| H1b | Remove one required key from `~/jtsmm.yaml` → `jtsmm record --duration 5` | fails with `missing required key: ...` before capture |
| H1c | Add `host.unknown_key: 1` → `jtsmm config validate` | fails with `unknown key not in schema` |
| H2 | `jtsmm config validate` | no hard errors |
| H3 | Change `host.capture.settle_ms: 300` → `jtsmm record --duration 20` | capture succeeds (timing observable in logs) |
| H4 | Change `mcu.emg.default_gain: 8` → `jtsmm config apply` → `jtsmm record --duration 10` | gain in HDF5 `/meta/mcu_config_json` matches |
| H5 | `host.viewer.enabled: true` + laptop browser | live plots during H4 |
| H6 | `scripts/hil/run_bench_hw.sh` + `JTSMM_HW_REQUIRED=1` | integrity pass |
| H7 | Change `mcu.trigger.rate_hz` + matching `host.cameras.list[].trigger.expect_rate_hz` → record 30s | `jtsmm doctor` + integrity pass |
| H8 | `host.emg_check.leadoff.duration_s: 3` → `jtsmm emg check` | leadoff phase ~3s |
| H9 | Set `session.subject_id` + `session.protocol` in YAML only → `jtsmm record --duration 10` | `/meta/subject_id` and `/meta/protocol` in HDF5 match YAML (no CLI flags) |

`scripts/hil/run_bench_hw.sh` runs **config audit --strict** (step H8 in the bench script) before optional soak capture.

Deprecated env overrides (last resort only): `JTSMM_SETTLE_MS`, `JTSMM_STREAM_RESTART`, `JTSMM_NATIVE`, `JTSMM_DATA_DIR`.
