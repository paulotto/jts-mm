# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Static catalogs for profile resolution and validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# The firmware rejects 16000 (emg_config_check.h: max 2000 SPS on this board).
EMG_SAMPLE_RATES_SPS = (250, 500, 1000, 2000, 4000)
# Default since 2026-09-25: the ADS1299's sinc3 filter is flat to 500 Hz at 4 kSPS
# (-0.7 dB vs -2.7 dB at 2 kSPS) and rejects aliases by 51 dB (31 dB); verified on
# the Pi with both IMUs and cameras (8 samples lost in 274 s).
DEFAULT_EMG_SAMPLE_RATE_SPS = 4000
EMG_GAINS = (1, 2, 4, 6, 8, 12, 24)

# ADS1299 muscle inputs are CH1–CH6 (1-based). CH7 is a spare/debug channel and
# CH8 is the DRL output pin; only CH1–CH6 can be power-managed or fed into the
# bias/DRL common-mode average.
MUSCLE_CHANNELS = (1, 2, 3, 4, 5, 6)
MUSCLE_CHANNEL_MASK = 0x3F
VIEWER_EMG_CHANNELS = len(MUSCLE_CHANNELS)


def parse_channel_spec(spec: object) -> list[int]:
    """Parse a 1-based muscle-channel spec into a sorted unique list (CH1–CH6).

    Accepts a comma/range string like ``"1,4"`` or ``"1-4"``, or an iterable of
    ints (``[1, 4]``). Raises ``ValueError`` for anything outside CH1–CH6.
    """
    channels: set[int] = set()
    if spec is None:
        return []
    if isinstance(spec, str):
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start, end = int(start_s), int(end_s)
                if start > end:
                    start, end = end, start
                channels.update(range(start, end + 1))
            else:
                channels.add(int(part))
    elif isinstance(spec, Iterable):
        for ch in spec:
            channels.add(int(ch))
    else:
        raise ValueError(f"channel spec must be a string or list, got {type(spec).__name__}")
    for ch in channels:
        if ch not in MUSCLE_CHANNELS:
            raise ValueError(f"muscle channel must be 1–6, got {ch}")
    return sorted(channels)


def channels_to_mask(channels: Sequence[int] | str | None) -> int:
    """Convert a 1-based channel list/spec into an ADS1299 bit0=CH1 bitmask."""
    mask = 0
    for ch in parse_channel_spec(channels):
        mask |= 1 << (ch - 1)
    return mask


def mask_to_channels(mask: int) -> list[int]:
    """Inverse of :func:`channels_to_mask` (bit0=CH1 → 1-based list)."""
    return [ch for ch in MUSCLE_CHANNELS if mask & (1 << (ch - 1))]

# ADS1299 analog front-end: 4.5 V reference, 24-bit bipolar output.
ADS1299_VREF_V = 4.5


def lsb_uv(gain: int) -> float:
    """LSB size in microvolts for a given PGA gain (ADS1299, 4.5 V ref, 24-bit).

    Shared by the HDF5 writer, the session reader, and the live viewer so counts
    are converted to microvolts identically everywhere.
    """
    g = gain if gain else 1
    return (ADS1299_VREF_V / (g * (1 << 23))) * 1e6


# CH7 (bias debug) and CH8 (DRL readback) are never set by the muscle gain: the
# firmware writes them with fixed CHnSET values whose GAIN field is 24
# (ADS_CH8SET_DRL 0x67, ADS_CH1SET_NORMAL 0x60 in firmware ads1299.c). Scaling
# them with the muscle gain overstated every DRL figure by 24/6 = 4x.
AUX_CHANNEL_GAIN = 24


def channel_lsb_uv(gain: int, n_channels: int = 8) -> list[float]:
    """Per-stream-column LSB in µV: muscle gain for CH1-6, fixed gain for CH7/CH8."""
    muscle = lsb_uv(gain)
    aux = lsb_uv(AUX_CHANNEL_GAIN)
    return [muscle if i < len(MUSCLE_CHANNELS) else aux for i in range(n_channels)]

IMU_SYNC_MODES = frozenset({"int_driven", "emg_drdy_divisor"})

# USB frame bytes per second budget on one I2C bus (plan §1.3.4).
IMU_BUS_BUDGET_BPS = 40_000

PROFILE_FILES = {
    "platform": {
        "default": "platform_default.yaml",
    },
    "emg": {
        "acquisition": "emg_acquisition.yaml",
        "high_rate": "emg_acquisition.yaml",
        "bench": "emg_acquisition.yaml",
        "noise_characterization": "emg_acquisition.yaml",
    },
    "imu": {
        "kinematics_accel": "imu_kinematics_default.yaml",
        "kinematics_default": "imu_kinematics_default.yaml",
        "fused_only": "imu_kinematics_default.yaml",
        "raw_only": "imu_kinematics_default.yaml",
        "datasheet_motion": "imu_kinematics_default.yaml",
        "all": "imu_kinematics_default.yaml",
    },
    "trigger": {
        "imx296_dual_csi": "trigger_imx296_dual_csi.yaml",
        "imx296_high_fps": "trigger_imx296_dual_csi.yaml",
        "bench_scope": "trigger_imx296_dual_csi.yaml",
        "track_a_led": "trigger_imx296_dual_csi.yaml",
    },
}

EMG_PROFILE_OVERRIDES: dict[str, dict] = {
    "acquisition": {"sample_rate_sps": DEFAULT_EMG_SAMPLE_RATE_SPS},
    "high_rate": {"sample_rate_sps": DEFAULT_EMG_SAMPLE_RATE_SPS},
    "bench": {"sample_rate_sps": 250, "leadoff": {"enabled": True, "during_recording": True}},
    # Amplifier noise-floor characterization must stay free of *any* injected
    # current (DC comparator included), so this profile keeps leadoff off
    # even though it is on by default for every other profile above. It used to
    # run at 16 kSPS, which the firmware no longer accepts (max 4000).
    "noise_characterization": {"sample_rate_sps": DEFAULT_EMG_SAMPLE_RATE_SPS,
                               "leadoff": {"during_recording": False}},
}

# Fusion quaternion rate. The BNO085 game/rotation vector tops out near its
# native ~398 Hz, so 390 keeps the 0.5% integrity-drop budget with headroom
# (docs/software/s08_runtime_configuration.md). Firmware config_defaults.c uses
# the same value; test_imu_rate_parity asserts the two stay in lockstep.
IMU_FUSION_RATE_HZ = 390

# Profile of a slot that names none (firmware config_defaults.c boots with it too).
DEFAULT_IMU_PROFILE = "kinematics_accel"

# Preset -> report map for a slot without enabled reports: resolve_profiles
# (doctor imu-rates) and the `config apply` payload (mcu_apply.py sends the
# preset's reports explicitly, so the MCU runs this set even for presets it has
# no copy of). An explicit `mcu.imu_*.reports` map wins over the preset. The
# firmware's own presets (imu_config.c) only matter for a boot without a host
# apply; raw_only differs there (SH-2 catalog default 100 Hz, not 400 Hz).
IMU_PROFILE_REPORTS: dict[str, dict[str, dict]] = {
    "fused_only": {
        "game_rotation_vector": {"enabled": True, "rate_hz": IMU_FUSION_RATE_HZ},
    },
    "raw_only": {
        "accelerometer": {"enabled": True, "rate_hz": 400},
        "gyroscope": {"enabled": True, "rate_hz": 400},
        "magnetometer": {"enabled": False, "rate_hz": 100},
    },
    # Default science capture: orientation + translation. Measured on hardware,
    # game_rv + linear_accel run at ~350 Hz + ~220 Hz (the BNO085 hub prioritizes
    # the fusion report and quantizes the pair to convenient interleaved slots).
    # linear_accel is requested at 250 (not 200): 200 Hz / 5 ms aligns badly with
    # the fusion schedule and collapses to ~120 Hz, whereas 250 Hz / 4 ms
    # quantizes cleanly to ~220 Hz — nearly 2x translation bandwidth for a small
    # orientation cost. Gyro is intentionally omitted:
    # adding it throttles all three (~331/331/79). Enable gyro explicitly via
    # `datasheet_motion`, `raw_only`, or `custom`.
    "kinematics_default": {
        "game_rotation_vector": {"enabled": True, "rate_hz": IMU_FUSION_RATE_HZ},
        "linear_acceleration": {"enabled": True, "rate_hz": 250},
    },
    # Default capture: orientation + the gravity-included accelerometer, so
    # gravity can be removed offline with the optical orientation instead of the
    # BNO085's own estimate (linear_acceleration inherits its ~2.5 deg tilt error:
    # 9.81 m/s^2 * sin(2.5 deg) ~ 0.43 m/s^2). Same 250 Hz slot as linear_accel
    # above; the BNO085 accelerometer and linear_accel share one sensor schedule.
    "kinematics_accel": {
        "game_rotation_vector": {"enabled": True, "rate_hz": IMU_FUSION_RATE_HZ},
        "accelerometer": {"enabled": True, "rate_hz": 250},
    },
    "datasheet_motion": {
        "accelerometer": {"enabled": True, "rate_hz": 400},
        "gyroscope": {"enabled": True, "rate_hz": 400},
        "magnetometer": {"enabled": True, "rate_hz": 100},
        "linear_acceleration": {"enabled": True, "rate_hz": 200},
    },
}

# `profile: all` = the firmware's boot preset (imu_config.c IMU_PROFILE_ALL):
# these four plus the fusion.primary quaternion. A slot holds at most
# IMU_MAX_REPORTS_PER_SLOT reports (firmware config.h IMU_MAX_REPORTS).
IMU_ALL_PRESET_REPORTS = ("accelerometer", "gyroscope", "magnetometer", "linear_acceleration")
IMU_MAX_REPORTS_PER_SLOT = 8

# Mirrors firmware trigger_config_resolve_profile (bench_scope: 10 Hz, 1 ms).
TRIGGER_PROFILE_OVERRIDES: dict[str, dict] = {
    "imx296_high_fps": {"rate_hz": 100, "pulse": {"width_us": 2000}},
    "bench_scope": {
        "rate_hz": 10,
        "pulse": {"width_us": 1000},
        "outputs": {"csi1": False, "csi2": False, "scope": True},
    },
    # h06 §3.4 / firmware trigger_config.c: webcams + sync LED at 30 Hz; the pulse
    # only lights the TRIG LED (no XTRIG camera), so it stays short.
    "track_a_led": {"rate_hz": 30, "pulse": {"width_us": 200}},
}


# Tagged USB record payload lengths (records.h / s05 §1): QUAT = 1 (report_id)
# + 4*f32 + 1 (accuracy) = 18 B; RAW = 1 (sensor_id) + 3*int16 + 1 = 8 B;
# LIN_ACCEL = 3*f32 + 1 = 13 B. Host decoders also accept the legacy lengths
# without the trailing accuracy byte (17 / 7 / 12).
IMU_QUAT_PAYLOAD_LEN = 18
IMU_RAW_PAYLOAD_LEN = 8
IMU_LIN_PAYLOAD_LEN = 13


@dataclass(frozen=True)
class ImuReportSpec:
    name: str
    usb_type: int
    usb_payload_len: int
    max_rate_hz: int
    default_rate_hz: int
    # kind: one of "raw" (int16 motion vector), "quat" (fusion), "lin" (float32).
    kind: str = "raw"
    # True for the seven core datasheet outputs freely selectable by operators;
    # False for reports gated behind ``profile: all`` (AR/VR, gravity, raw ADC).
    datasheet: bool = True


BNO085_REPORT_CATALOG: dict[str, ImuReportSpec] = {
    # --- The seven datasheet motion/rotation outputs (freely combinable). ---
    "accelerometer": ImuReportSpec(
        "accelerometer", 0x12, IMU_RAW_PAYLOAD_LEN, 500, 400, kind="raw"),
    "gyroscope": ImuReportSpec(
        "gyroscope", 0x12, IMU_RAW_PAYLOAD_LEN, 400, 400, kind="raw"),
    "magnetometer": ImuReportSpec(
        "magnetometer", 0x12, IMU_RAW_PAYLOAD_LEN, 100, 100, kind="raw"),
    "linear_acceleration": ImuReportSpec(
        "linear_acceleration", 0x13, IMU_LIN_PAYLOAD_LEN, 400, 200, kind="lin"),
    "rotation_vector": ImuReportSpec(
        "rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 400, 200, kind="quat"),
    "game_rotation_vector": ImuReportSpec(
        "game_rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 400, 400, kind="quat"),
    "geomagnetic_rotation_vector": ImuReportSpec(
        "geomagnetic_rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 400, 200, kind="quat"),
    # --- Optional extra fusion (implemented in firmware, off the core list). ---
    "gyro_integrated_rotation_vector": ImuReportSpec(
        "gyro_integrated_rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 1000, 1000,
        kind="quat", datasheet=False),
    # --- Gated behind profile:all (not yet emitted by firmware). ---
    "gravity": ImuReportSpec("gravity", 0x13, IMU_LIN_PAYLOAD_LEN, 200, 100, kind="lin",
                             datasheet=False),
    "raw_accelerometer": ImuReportSpec(
        "raw_accelerometer", 0x12, IMU_RAW_PAYLOAD_LEN, 500, 400, kind="raw", datasheet=False),
    "arvr_stabilized_rotation_vector": ImuReportSpec(
        "arvr_stabilized_rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 400, 200,
        kind="quat", datasheet=False),
    "arvr_stabilized_game_rotation_vector": ImuReportSpec(
        "arvr_stabilized_game_rotation_vector", 0x11, IMU_QUAT_PAYLOAD_LEN, 400, 400,
        kind="quat", datasheet=False),
}

# The seven core datasheet outputs, in datasheet order.
DATASHEET_OUTPUTS: tuple[str, ...] = tuple(
    name for name, spec in BNO085_REPORT_CATALOG.items() if spec.datasheet
)
# Fusion reports emit a quaternion; only one should be enabled per slot outside
# of profile: custom (validate.py enforces this).
FUSION_REPORTS: frozenset[str] = frozenset(
    name for name, spec in BNO085_REPORT_CATALOG.items() if spec.kind == "quat"
)

FUSION_PRIMARY_TO_REPORT = {
    "game_rotation_vector": "game_rotation_vector",
    "rotation_vector": "rotation_vector",
    "geomagnetic_rotation_vector": "geomagnetic_rotation_vector",
    "arvr_stabilized_game_rotation_vector": "arvr_stabilized_game_rotation_vector",
    "arvr_stabilized_rotation_vector": "arvr_stabilized_rotation_vector",
    "gyro_integrated_rotation_vector": "gyro_integrated_rotation_vector",
}

# What the firmware actually implements. config_json.c maps an unknown name to a
# default instead of failing (parse_fusion_primary -> game RV, parse_trigger_profile
# -> imx296_dual_csi) and parse_imu_reports silently skips a report missing from
# bno085_reports.c catalog[], so the host must reject anything outside these sets.
# tests/test_fix_H_validation.py parses the C sources to keep them in sync.
FIRMWARE_IMU_REPORTS: frozenset[str] = frozenset(
    {
        "accelerometer",
        "gyroscope",
        "magnetometer",
        "linear_acceleration",
        "rotation_vector",
        "game_rotation_vector",
        "geomagnetic_rotation_vector",
        "gyro_integrated_rotation_vector",
    }
)
FIRMWARE_FUSION_PRIMARIES: frozenset[str] = frozenset(
    {
        "game_rotation_vector",
        "rotation_vector",
        "geomagnetic_rotation_vector",
        "gyro_integrated_rotation_vector",
    }
)
FIRMWARE_TRIGGER_PROFILES: frozenset[str] = frozenset(
    {"imx296_dual_csi", "imx296_high_fps", "bench_scope", "track_a_led", "custom"}
)
# config.h trigger_config_t.rate_hz is uint16 and trigger_config_validate caps at 200.
TRIGGER_RATE_HZ_MIN = 1
TRIGGER_RATE_HZ_MAX = 200


def trigger_period_us(rate_hz: int) -> int:
    """Trigger period as the firmware computes it (trigger_timing.c: 1e6 / rate, integer)."""
    return 1_000_000 // int(rate_hz)


# ADS1299 CH1..CH8 -> muscle, as the electrode cables are labelled (hardware check 2026-11-03,
# docs/analysis/an03 §1). Written into every session as /emg channel_names.
FULL_CHANNEL_NAMES = [
    "masseter_r",
    "temporalis_r",
    "digastric_r",
    "digastric_l",
    "masseter_l",
    "temporalis_l",
    "bias_debug",
    "drl",
]

# The list recorders wrote until 2026-11-03: CH3/CH4 named digastric_l/digastric_r, the reverse
# of the cable labels. Sessions carrying exactly this list are read with FULL_CHANNEL_NAMES
# (jtsmm.io.session_reader); the files themselves are not changed.
LEGACY_CHANNEL_NAMES_DIGASTRIC_SWAPPED = (
    "masseter_r",
    "temporalis_r",
    "digastric_l",
    "digastric_r",
    "masseter_l",
    "temporalis_l",
    "bias_debug",
    "drl",
)

# Muscle-only labels for the live viewer (CH7/CH8 are not wired to electrodes).
MUSCLE_CHANNEL_NAMES = list(FULL_CHANNEL_NAMES[: len(MUSCLE_CHANNELS)])

# Patient DRL (driven-right-leg) on ADS1299 CH8 / IN8N — separate from muscle pairs.
DRL_CHANNEL = 8
DRL_STREAM_INDEX = 7
DRL_NAME = "drl"

# Live viewer plots: six muscles plus CH8 DRL (CH7 bias_debug is omitted).
VIEWER_EMG_PLOT_COUNT = VIEWER_EMG_CHANNELS + 1
VIEWER_EMG_STREAM_COLS = (0, 1, 2, 3, 4, 5, DRL_STREAM_INDEX)
VIEWER_EMG_PLOT_NAMES = MUSCLE_CHANNEL_NAMES + [DRL_NAME]

IMX296_EXPOSURE_MARGIN_US = 14.26

# Host camera (picamera2 / libcamera) bounds — mirrored in jtsmm.schema.json.
CAMERA_GAIN_MIN = 1.0
CAMERA_GAIN_MAX = 16.0
CAMERA_NOISE_REDUCTION_MODES = frozenset({"off", "fast", "high_quality"})
CAMERA_PIXEL_FORMATS = frozenset({"YUV420", "RGB888", "XBGR8888", "BGR888"})
