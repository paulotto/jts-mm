<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# schema/ — shared configuration contract

Single source of truth for JTS-MM runtime configuration. Consumed by:

- **Host:** `software/jtsmm/jtsmm/config/` — load YAML, resolve profiles, validate
- **Firmware:** `firmware/jtsmm/include/jtsmm/config.h` — C struct mirror (hand-synced)
- **Docs:** `docs/software/s08_runtime_configuration.md`
- **Tests:** cross-validation between host and firmware apply paths

## Layout

```
schema/v1/
├── jtsmm.schema.json       # JSON Schema — machine validation
├── jtsmm.schema.yaml       # Field docs + defaults (human reference)
├── mcu_config.h.in         # C struct template (optional codegen)
├── config_coverage.yaml    # Every tunable key and where it is consumed (audit)
├── optical.schema.yaml     # Field outline of software/optical/etc/optical.yaml
├── golden/                 # mcu_apply_template.json (MCU apply payload), frames.json
└── profiles/               # Composable profile fragments
    ├── platform_default.yaml
    ├── emg_acquisition.yaml
    ├── imu_kinematics_default.yaml
    └── trigger_imx296_dual_csi.yaml
```

Operator-facing template: [`software/jtsmm/etc/jtsmm.yaml`](../software/jtsmm/etc/jtsmm.yaml) → installed as `~/jtsmm.yaml` (`jtsmm config init`, `scripts/install_pi.sh`).

A regular (non-editable) install cannot see this directory, so `software/jtsmm/setup.py` copies `schema/v1/` and the template into `jtsmm/_data/` at build time; the loader prefers the checkout copy. Nothing under `_data/` is committed.
