# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CLI overrides applied to a resolved config before MCU apply.

These mirror the bringup emg-lab flags (``--bias-drive``, ``--bias-channels``,
``--active-channels``) but patch the resolved ``mcu.emg`` block in place so the
existing config → ``build_mcu_apply_payload`` → firmware path handles the wire
translation (1-based lists → bitmasks).
"""

from __future__ import annotations

from typing import Any

from .catalog import EMG_GAINS, parse_channel_spec


def apply_emg_overrides(
    resolved: dict[str, Any],
    *,
    bias_drive: bool | None = None,
    bias_channels: str | None = None,
    active_channels: str | None = None,
    leadoff: bool | None = None,
    default_gain: int | None = None,
    monitoring: bool = False,
) -> dict[str, Any]:
    """Patch ``resolved['mcu']['emg']`` from CLI overrides (mutates and returns).

    ``bias_channels``/``active_channels`` accept the same specs as bringup
    (``"1,4"`` or ``"1-4"``); a ``ValueError`` is raised for out-of-range channels.

    ``default_gain`` (the ADS1299 PGA setting for all patient channels) must be
    one of :data:`jtsmm.config.catalog.EMG_GAINS`; a ``ValueError`` is raised
    otherwise.

    DC lead-off (comparator only, no AC tone) is on by default during both
    recording and live viewing (``leadoff.during_recording: true`` in the
    resolved profile) so the UI always has contact feedback without touching
    the EMG band. ``leadoff`` is the ``--leadoff``/``--no-leadoff`` CLI
    override and always wins when given. When it is *not* given and
    ``monitoring`` is true (``jtsmm view`` only), lead-off is still forced on
    so a custom profile that disabled it doesn't leave the viewer's contact
    LEDs silently blind.
    """
    mcu = resolved.setdefault("mcu", {})
    emg = mcu.setdefault("emg", {})

    if leadoff is not None:
        leadoff_cfg = emg.setdefault("leadoff", {})
        leadoff_cfg["during_recording"] = bool(leadoff)
        if leadoff:
            leadoff_cfg["enabled"] = True
    elif monitoring:
        leadoff_cfg = emg.setdefault("leadoff", {})
        leadoff_cfg["enabled"] = True
        leadoff_cfg["during_recording"] = True

    if bias_drive is not None:
        bias = emg.setdefault("bias", {})
        bias["enabled"] = bool(bias_drive)

    if bias_channels is not None:
        bias = emg.setdefault("bias", {})
        bias["channels"] = parse_channel_spec(bias_channels)
        if not bias.get("enabled", True):
            raise ValueError("--bias-channels requires bias drive enabled (--bias-drive)")

    if active_channels is not None:
        emg["active_channels"] = parse_channel_spec(active_channels)

    # Bias contributors must be powered-on muscle channels only.
    active_list = emg.get("active_channels")
    bias = emg.get("bias")
    if bias and active_list:
        bias_list = bias.get("channels")
        if bias_list:
            active_set = set(active_list)
            outside = sorted(set(bias_list) - active_set)
            if outside:
                raise ValueError(
                    f"bias channels {outside} are not in active_channels {active_list} "
                    "(powered-down channels cannot drive the DRL bias loop)"
                )

    if default_gain is not None:
        if int(default_gain) not in EMG_GAINS:
            raise ValueError(f"--gain {default_gain} not in {EMG_GAINS}")
        emg["default_gain"] = int(default_gain)

    return resolved
