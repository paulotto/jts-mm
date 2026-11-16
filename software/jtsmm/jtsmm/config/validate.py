# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cross-subsystem validation beyond JSON Schema."""

from __future__ import annotations

import os
from typing import Any

from .catalog import (
    BNO085_REPORT_CATALOG,
    CAMERA_GAIN_MAX,
    CAMERA_GAIN_MIN,
    CAMERA_NOISE_REDUCTION_MODES,
    CAMERA_PIXEL_FORMATS,
    DEFAULT_IMU_PROFILE,
    EMG_GAINS,
    EMG_SAMPLE_RATES_SPS,
    FIRMWARE_FUSION_PRIMARIES,
    FIRMWARE_IMU_REPORTS,
    FIRMWARE_TRIGGER_PROFILES,
    FUSION_PRIMARY_TO_REPORT,
    FUSION_REPORTS,
    IMU_BUS_BUDGET_BPS,
    IMU_MAX_REPORTS_PER_SLOT,
    IMU_PROFILE_REPORTS,
    IMU_SYNC_MODES,
    IMX296_EXPOSURE_MARGIN_US,
    trigger_period_us,
)
from .loader import validate_schema


def _usb_bytes_per_second(rate_hz: float, payload_len: int) -> float:
    frame_overhead = 4 + 8 + 2  # TYPE, LEN, t_master, CRC
    return rate_hz * (frame_overhead + payload_len)


def _validate_imu_slot(slot: dict[str, Any], *, label: str, emg_fs: int, errors: list[str]) -> None:
    if not slot.get("enabled", True):
        return

    sync = slot.get("sync") or {}
    mode = sync.get("mode", "int_driven")
    if mode not in IMU_SYNC_MODES:
        errors.append(f"mcu.{label}.sync.mode must be one of {sorted(IMU_SYNC_MODES)}")
        return

    reports = slot.get("reports") or {}
    profile = slot.get("profile", DEFAULT_IMU_PROFILE)
    fusion = slot.get("fusion") or {}
    primary = fusion.get("primary", "game_rotation_vector")
    if primary not in FUSION_PRIMARY_TO_REPORT:
        errors.append(f"mcu.{label}.fusion.primary unknown: {primary}")
    elif primary not in FIRMWARE_FUSION_PRIMARIES:
        errors.append(
            f"mcu.{label}.fusion.primary={primary} is not implemented by the firmware "
            f"(config_json.c would silently use game_rotation_vector); use one of "
            f"{sorted(FIRMWARE_FUSION_PRIMARIES)}"
        )

    bus_bps = 0.0
    enabled_fusion: list[str] = []
    enabled_names: set[str] = set()
    for name, cfg in reports.items():
        if not cfg.get("enabled", True):
            continue
        spec = BNO085_REPORT_CATALOG.get(name)
        if spec is None:
            errors.append(f"mcu.{label}.reports unknown report: {name}")
            continue
        enabled_names.add(name)
        if name not in FIRMWARE_IMU_REPORTS:
            errors.append(
                f"mcu.{label}.reports.{name} is not implemented by the firmware (it would "
                f"be skipped and emit nothing); implemented: {sorted(FIRMWARE_IMU_REPORTS)}"
            )
            continue
        if not spec.datasheet and profile not in ("all", "custom"):
            errors.append(
                f"mcu.{label}.reports.{name} is not a core datasheet output; enable it only "
                f"under profile: all or profile: custom (got profile: {profile})"
            )
        rate = int(cfg.get("rate_hz", spec.default_rate_hz))
        if name in FUSION_REPORTS:
            enabled_fusion.append((name, rate))
        if rate <= 0:
            errors.append(f"mcu.{label}.reports.{name}.rate_hz must be > 0")
            continue
        if rate > spec.max_rate_hz:
            errors.append(
                f"mcu.{label}.reports.{name}.rate_hz={rate} exceeds max {spec.max_rate_hz} Hz"
            )
        cfg["rate_hz"] = rate
        bus_bps += _usb_bytes_per_second(rate, spec.usb_payload_len)

        if mode == "emg_drdy_divisor" and emg_fs % rate != 0:
            errors.append(
                f"mcu.{label}.sync.mode=emg_drdy_divisor requires EMG rate ({emg_fs}) "
                f"divisible by IMU rate ({rate}) for report {name}"
            )

    if not enabled_names:
        # config_json.c keeps the slot's profile when no report is given, and the
        # MCU then expands its OWN preset (imu_config.c) -- not what the host shows.
        errors.append(
            f"mcu.{label} is enabled but has no enabled report; list the reports "
            "explicitly (profile names are labels) or set enabled: false"
        )
    elif len(enabled_names) > IMU_MAX_REPORTS_PER_SLOT:
        errors.append(
            f"mcu.{label} enables {len(enabled_names)} reports; the firmware keeps only "
            f"the first {IMU_MAX_REPORTS_PER_SLOT} (config.h IMU_MAX_REPORTS)"
        )
    elif profile in IMU_PROFILE_REPORTS:
        # The reports map is what runs; a preset label that says otherwise
        # (e.g. kinematics_accel over a linear_acceleration map) misleads.
        from .profiles import _resolve_imu_slot

        preset = {n for n, c in _resolve_imu_slot({**slot, "reports": {}})["reports"].items()
                  if c.get("enabled", True)}
        if enabled_names != preset:
            errors.append(
                f"warn: mcu.{label}.reports {sorted(enabled_names)} differ from profile "
                f"{profile} {sorted(preset)}; the reports run (rename the profile to custom "
                "or match the reports)"
            )

    # At most one fusion quaternion per slot: hard error outside custom (the wire
    # format supports multiple via quat_report_id, but ambiguous science defaults);
    # a warning under profile: custom where the operator opts in explicitly.
    if len(enabled_fusion) > 1:
        msg = (
            f"mcu.{label} enables multiple fusion quaternions "
            f"{sorted(n for n, _ in enabled_fusion)}; at most one is recommended per slot"
        )
        errors.append(f"warn: {msg}" if profile == "custom" else msg)
    elif len(enabled_fusion) == 1 and primary in FUSION_PRIMARY_TO_REPORT:
        # The firmware takes the explicit report map verbatim (config_json.c sets
        # the slot to custom), and fusion.* is written into the HDF5 metadata, so
        # a stale fusion block would describe a stream that was not recorded.
        name, rate = enabled_fusion[0]
        want = FUSION_PRIMARY_TO_REPORT[primary]
        if name != want:
            errors.append(
                f"mcu.{label}.fusion.primary={primary} but the enabled fusion report is "
                f"{name}; make them agree"
            )
        fusion_rate = fusion.get("rate_hz")
        if fusion_rate is not None and int(fusion_rate) != rate:
            errors.append(
                f"mcu.{label}.fusion.rate_hz={fusion_rate} but reports.{name}.rate_hz={rate}; "
                "make them agree"
            )

    # Geomagnetic RV fuses accel+mag, so the magnetometer report is normally
    # wanted alongside it for the viewer's raw mag strip.
    if "geomagnetic_rotation_vector" in enabled_names and "magnetometer" not in enabled_names:
        errors.append(
            f"warn: mcu.{label} uses geomagnetic_rotation_vector without enabling the "
            "magnetometer report (raw mag will be unavailable)"
        )

    # I2C/USB bus-bandwidth budget per slot. Hard error only for profile: all
    # (the documented bus-flood mode); a warning otherwise so operators notice
    # heavy custom combinations before the Phase 4 drop check fails on hardware.
    if bus_bps > IMU_BUS_BUDGET_BPS:
        msg = (
            f"mcu.{label} report set draws {bus_bps:.0f} B/s > I2C bus budget "
            f"{IMU_BUS_BUDGET_BPS} B/s"
        )
        errors.append(msg if profile == "all" else f"warn: {msg}")


def _channel_list(spec: Any, *, field: str, errors: list[str]) -> list[int] | None:
    from .catalog import parse_channel_spec

    if spec is None:
        return None
    try:
        return parse_channel_spec(spec)
    except (ValueError, TypeError) as exc:
        errors.append(f"mcu.emg.{field}: {exc}")
        return None


def _validate_emg_channels(emg: dict[str, Any], errors: list[str]) -> None:
    """Validate the 1-based DRL/active channel selections (CH1–CH6)."""
    bias = emg.get("bias") or {}
    bias_enabled = bias.get("enabled", False)
    bias_ch = _channel_list(bias.get("channels"), field="bias.channels", errors=errors)
    active = _channel_list(emg.get("active_channels"), field="active_channels", errors=errors)

    if bias_ch is not None:
        if not bias_ch:
            errors.append("mcu.emg.bias.channels is empty; specify at least one channel (1–6)")
        if not bias_enabled:
            errors.append("warn: emg.bias.channels set but bias.enabled=false (ignored)")
    if active is not None and not active:
        errors.append("mcu.emg.active_channels is empty; specify at least one channel (1–6)")
    if bias_ch and active and not set(bias_ch).issubset(active):
        outside = sorted(set(bias_ch) - set(active))
        errors.append(
            "mcu.emg.bias.channels "
            f"{outside} are not in active_channels {active} "
            "(powered-down channels cannot drive the DRL bias loop)"
        )


def _camera_device_key(device: Any) -> str:
    """picamera2 index alias: ``csi0`` and ``0`` name the same camera."""
    text = str(device).strip().lower()
    if text.startswith("csi") and text[3:].isdigit():
        return str(int(text[3:]))
    return str(int(text)) if text.isdigit() else text


def _validate_camera_identity(cam_list: list[Any], errors: list[str]) -> None:
    """Camera ids name HDF5 groups and devices name sensors: both must be unique."""
    seen_ids: dict[str, int] = {}
    seen_dev: dict[str, int] = {}
    for idx, cam in enumerate(cam_list):
        if not isinstance(cam, dict):
            continue
        cam_id = cam.get("id")
        if cam_id is not None:
            if str(cam_id) in seen_ids:
                errors.append(
                    f"host.cameras.list[{idx}].id={cam_id!r} duplicates "
                    f"host.cameras.list[{seen_ids[str(cam_id)]}]"
                )
            seen_ids.setdefault(str(cam_id), idx)
        device = cam.get("device")
        if device is not None:
            key = _camera_device_key(device)
            if key in seen_dev:
                errors.append(
                    f"host.cameras.list[{idx}].device={device!r} is the same camera as "
                    f"host.cameras.list[{seen_dev[key]}]"
                )
            seen_dev.setdefault(key, idx)


def _validate_data_dir_alias(host: dict[str, Any], errors: list[str]) -> None:
    """host.storage.base_dir is a deprecated alias of host.paths.data_dir."""
    base = (host.get("storage") or {}).get("base_dir")
    if base is None:
        return
    data_dir = (host.get("paths") or {}).get("data_dir")
    if data_dir is None:
        errors.append(
            "warn: host.storage.base_dir is deprecated; rename it to host.paths.data_dir"
        )
    elif os.path.expanduser(str(base)) != os.path.expanduser(str(data_dir)):
        errors.append(
            f"host.storage.base_dir={base} is deprecated and would be IGNORED; sessions are "
            f"written to host.paths.data_dir={data_dir}. Delete host.storage."
        )


def _validate_dead_emg_filter_keys(host: dict[str, Any], errors: list[str]) -> None:
    """host.emg_filter.* except mains_mode has no consumer; reject it when it looks effective."""
    f = host.get("emg_filter") or {}
    dead = [k for k in ("notch_hz", "highpass_hz", "lowpass_hz") if k in f]
    if f.get("enabled"):
        dead.insert(0, "enabled")
    if dead:
        errors.append(
            f"host.emg_filter.{', '.join(dead)} has no effect (deprecated); the viewer "
            "filter is fixed and only host.emg_filter.mains_mode is read. Remove it"
        )


# Keys with exactly one honest value: the firmware refuses any other (config_json.c
# fixed_value(), so the whole apply would fail) or nothing reads the key. Checked
# here so the host rejects them before sending, with the reason. `None` = must be
# absent. Wire or drop a key -> remove it here.
_INERT_KEYS: tuple[tuple[str, Any, str], ...] = (
    ("mcu.platform.system.xosc_hz", 12_000_000, "the board crystal; the firmware refuses others"),
    ("mcu.platform.pio.trigger.clock_hz", 1_000_000,
     "one PIO tick = 1 us in trigger_timing.c; the firmware refuses others"),
    ("mcu.platform.irq.emg", 0, "EMG is polled on core 1; the firmware refuses others"),
    ("mcu.platform.irq.imu", 1, "IMU is polled on core 1; the firmware refuses others"),
    ("mcu.emg.stream_channels", 6,
     "every EMG block carries 8 channels and no host code reads this key"),
    ("mcu.emg.bias.ch7_readback", False, "not implemented; the firmware refuses true"),
    ("mcu.emg.test_signal.amplitude_mv", 0,
     "not adjustable (0 = ADS1299 default test signal); the firmware refuses others"),
    ("mcu.trigger.outputs.csi1", True, "hard-wired on; the firmware refuses false"),
    ("mcu.trigger.outputs.csi2", True, "hard-wired on; the firmware refuses false"),
    ("mcu.trigger.outputs.scope", True, "hard-wired on; the firmware refuses false"),
    ("mcu.trigger.outputs.sync_led", False,
     "the trigger never drives the sync LED (`flash on` does); the firmware refuses true"),
    ("mcu.trigger.start_on", "stream_start",
     "not sent to the MCU; jtsmm record starts the trigger once the cameras stream"),
    ("host.cameras.marker_detector", "apriltag",
     "no reader; markers are configured in optical.yaml"),
    ("mcu.platform.custom", None, "custom blocks are only merged by `jtsmm doctor --preset`"),
    ("mcu.emg.custom", None, "custom blocks are only merged by `jtsmm doctor --preset`"),
    ("mcu.imu_mta.custom", None, "custom blocks are only merged by `jtsmm doctor --preset`"),
    ("mcu.imu_cra.custom", None, "custom blocks are only merged by `jtsmm doctor --preset`"),
    ("mcu.trigger.custom", None, "custom blocks are only merged by `jtsmm doctor --preset`"),
)

_ABSENT = object()


def _get_path(doc: dict[str, Any], dotted: str) -> Any:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return cur


def _validate_inert_keys(doc: dict[str, Any], errors: list[str]) -> None:
    for path, accepted, why in _INERT_KEYS:
        value = _get_path(doc, path)
        if value is _ABSENT:
            continue
        if accepted is None:
            errors.append(f"{path} has no effect ({why}); remove it")
        elif value != accepted:
            errors.append(f"{path}={value!r} has no effect ({why}); only {accepted!r} is accepted")


def validate_config(doc: dict[str, Any], *, include_schema: bool = True) -> list[str]:
    """Return human-readable validation errors (empty if valid).

    With ``include_schema`` a document that fails the JSON Schema gets only the
    schema errors: the cross-checks below assume its types (a gain of "high" or a
    null report entry would raise instead of being reported).
    """
    if include_schema:
        schema_errors = validate_schema(doc)
        if schema_errors:
            return list(schema_errors)
    errors: list[str] = []

    mcu = doc.get("mcu") or {}
    host = doc.get("host") or {}

    emg = mcu.get("emg") or {}
    fs = int(emg.get("sample_rate_sps", 1000))
    if fs not in EMG_SAMPLE_RATES_SPS:
        errors.append(f"mcu.emg.sample_rate_sps={fs} not in {EMG_SAMPLE_RATES_SPS}")

    gain = int(emg.get("default_gain", 6))
    if gain not in EMG_GAINS:
        errors.append(f"mcu.emg.default_gain={gain} not in {EMG_GAINS}")

    _validate_emg_channels(emg, errors)

    _validate_imu_slot(mcu.get("imu_mta") or {}, label="imu_mta", emg_fs=fs, errors=errors)
    _validate_imu_slot(mcu.get("imu_cra") or {}, label="imu_cra", emg_fs=fs, errors=errors)

    trigger = mcu.get("trigger") or {}
    trig_hz = trigger.get("rate_hz")
    pulse = trigger.get("pulse") or {}
    pulse_us = pulse.get("width_us")
    trig_profile = trigger.get("profile")
    if trig_profile is not None and trig_profile not in FIRMWARE_TRIGGER_PROFILES:
        errors.append(
            f"mcu.trigger.profile={trig_profile!r} is unknown to the firmware "
            f"(one of {sorted(FIRMWARE_TRIGGER_PROFILES)})"
        )
    if trig_hz is not None and pulse_us is not None and int(trig_hz) > 0:
        period = trigger_period_us(int(trig_hz))
        if int(pulse_us) >= period:
            # trigger_timing.c silently replaces such a width with period/2.
            errors.append(
                f"mcu.trigger.pulse.width_us={pulse_us} >= trigger period {period} us "
                f"at rate_hz={trig_hz} (the firmware would silently use {period // 2} us)"
            )

    cameras = (host.get("cameras") or {})
    cam_list = cameras.get("list") or []
    _validate_camera_identity(cam_list, errors)
    mcu_cams = [
        idx for idx, cam in enumerate(cam_list)
        if ((cam.get("trigger") or {}).get("source")) == "mcu"
    ]
    if mcu_cams and trigger.get("enabled") is False:
        # The firmware refuses trig_start while disabled (cmd_parser.c).
        errors.append(
            f"mcu.trigger.enabled=false but host.cameras.list{mcu_cams} use "
            "trigger.source: mcu -- those cameras would never be triggered"
        )
    for idx, cam in enumerate(cam_list):
        prefix = f"host.cameras.list[{idx}]"
        trig_cfg = cam.get("trigger") or {}
        expect = trig_cfg.get("expect_rate_hz")
        if trig_hz is not None and expect is not None and int(expect) != int(trig_hz):
            errors.append(
                f"{prefix}.trigger.expect_rate_hz={expect} "
                f"!= mcu.trigger.rate_hz={trig_hz}"
            )
        exposure = cam.get("exposure_us")
        if exposure is not None and pulse_us is not None:
            max_exp = float(pulse_us) + IMX296_EXPOSURE_MARGIN_US
            if float(exposure) > max_exp:
                errors.append(
                    f"{prefix}.exposure_us={exposure} exceeds "
                    f"trigger pulse budget ({max_exp:.2f} µs)"
                )
        gain = cam.get("gain")
        if gain is not None and not (CAMERA_GAIN_MIN <= float(gain) <= CAMERA_GAIN_MAX):
            errors.append(
                f"{prefix}.gain={gain} outside [{CAMERA_GAIN_MIN}, {CAMERA_GAIN_MAX}]"
            )
        pix_fmt = cam.get("format")
        if pix_fmt is not None and str(pix_fmt) not in CAMERA_PIXEL_FORMATS:
            errors.append(
                f"{prefix}.format={pix_fmt!r} not in {sorted(CAMERA_PIXEL_FORMATS)}"
            )
        nrm = cam.get("noise_reduction_mode")
        if nrm is not None and str(nrm) not in CAMERA_NOISE_REDUCTION_MODES:
            errors.append(
                f"{prefix}.noise_reduction_mode={nrm!r} "
                f"not in {sorted(CAMERA_NOISE_REDUCTION_MODES)}"
            )
        awb_gains = cam.get("awb_gains")
        if awb_gains is not None:
            if not isinstance(awb_gains, (list, tuple)) or len(awb_gains) != 2:
                errors.append(f"{prefix}.awb_gains must be [red, blue]")
            elif any(float(v) < 0 for v in awb_gains):
                errors.append(f"{prefix}.awb_gains values must be >= 0")
        if cam.get("ae_enable") and cameras.get("track") == "B":
            errors.append(
                f"warn: {prefix}.ae_enable=true on Track B — AGC may fight MCU "
                "XTRIG exposure; prefer ae_enable=false"
            )
        if expect is not None and exposure is not None and float(expect) > 0:
            period_us = 1_000_000.0 / float(expect)
            if float(exposure) >= period_us:
                errors.append(
                    f"{prefix}.exposure_us={exposure} >= frame period "
                    f"{period_us:.1f} µs at expect_rate_hz={expect}"
                )

    _validate_data_dir_alias(host, errors)
    _validate_dead_emg_filter_keys(host, errors)
    _validate_inert_keys(doc, errors)

    platform = mcu.get("platform") or {}
    buffers = platform.get("buffers") or {}
    # Ring sizes are byte budgets (firmware divides by each ring's slot size:
    # 48 B compact EMG slots, 256 B IMU/event slots).
    emg_ring = int(buffers.get("emg_ring", 98304))
    imu_ring = int(buffers.get("imu_ring", 16384))
    event_ring = int(buffers.get("event_ring", 8192))
    sram_est = emg_ring + 2 * imu_ring + event_ring
    if sram_est > 400_000:
        errors.append(
            f"mcu.platform.buffers SRAM estimate {sram_est} B exceeds ~400 KB budget"
        )

    spi_hz = int((platform.get("spi0") or {}).get("hz", 4_000_000))
    if fs >= 2000 and spi_hz < 2_000_000:
        errors.append(
            f"mcu.emg.sample_rate_sps={fs} with spi0.hz={spi_hz} may exceed ISR budget"
        )

    if emg.get("profile") == "bench" and not (emg.get("test_signal") or {}).get("enabled"):
        errors.append("warn: emg.profile=bench with test_signal disabled")

    leadoff = emg.get("leadoff") or {}
    if leadoff.get("during_recording") and leadoff.get("ac_tone"):
        errors.append("warn: emg.leadoff.during_recording+ac_tone injects a 31.2 Hz tone")
    cur = leadoff.get("current_na")
    if cur is not None and int(cur) not in (6, 24):
        errors.append(f"mcu.emg.leadoff.current_na={cur} not in (6, 24)")

    return errors
