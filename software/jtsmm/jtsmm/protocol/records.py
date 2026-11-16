# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Decode TYPE-specific payloads."""

from __future__ import annotations

import struct
from typing import Any

from .types import RecordType


def _sign_extend24(raw: int) -> int:
    if raw & 0x800000:
        raw -= 1 << 24
    return raw


def _read_emg_count24(payload: bytes, channel: int) -> int:
    """Decode one ADS1299 24-bit channel word (MSB-first, per firmware/RDATA)."""
    off = 3 + channel * 3
    raw = (payload[off] << 16) | (payload[off + 1] << 8) | payload[off + 2]
    return _sign_extend24(raw)


# EMG_BLOCK: status(3) + 8 x 24-bit channels = 27 B. Newer firmware appends a
# u32 LE acquisition sequence number (31 B) that advances once per conversion,
# including frames the firmware knows it missed, so a host-side seq gap is the
# exact number of frames lost anywhere after the ADC.
EMG_BLOCK_LEN = 27
EMG_BLOCK_LEN_SEQ = 31


def decode_emg_block(payload: bytes) -> dict[str, Any]:
    if len(payload) not in (EMG_BLOCK_LEN, EMG_BLOCK_LEN_SEQ):
        raise ValueError(f"EMG_BLOCK len {len(payload)} not in (27, 31)")
    status = payload[0:3]
    channels = [_read_emg_count24(payload, i) for i in range(8)]
    out: dict[str, Any] = {"status": bytes(status), "channels": channels}
    if len(payload) == EMG_BLOCK_LEN_SEQ:
        (out["seq"],) = struct.unpack("<I", payload[27:31])
    return out


# SH-2 fusion report ids carried in the tagged QUAT record.
FUSION_REPORT_IDS = {
    0x05: "rotation_vector",
    0x08: "game_rotation_vector",
    0x09: "geomagnetic_rotation_vector",
    0x2A: "gyro_integrated_rotation_vector",
}

# RAW record sensor tags (match firmware records.h IMU_SENSOR_*).
IMU_SENSOR_ACCEL = 0x01
IMU_SENSOR_GYRO = 0x02
IMU_SENSOR_MAG = 0x03
_RAW_SENSOR_KEY = {IMU_SENSOR_ACCEL: "accel", IMU_SENSOR_GYRO: "gyro", IMU_SENSOR_MAG: "mag"}
# Map sensor_id -> calibration stream key in the viewer.
_RAW_SENSOR_CAL_KEY = {
    IMU_SENSOR_ACCEL: "accel",
    IMU_SENSOR_GYRO: "gyro",
    IMU_SENSOR_MAG: "mag",
}


def _decode_accuracy(payload: bytes, idx: int) -> int | None:
    if len(payload) > idx:
        acc = int(payload[idx]) & 0x03
        return acc
    return None


def decode_quat(payload: bytes) -> dict[str, Any]:
    """Decode a fusion quaternion record.

    Tagged form (17/18 B): ``[report_id:u8][w,x,y,z:f32x4][acc:u8?]``. Legacy
    form (16 B): untagged quaternion assumed to be the slot's configured fusion
    primary.
    """
    if len(payload) >= 17:
        report_id = payload[0]
        w, x, y, z = struct.unpack("<4f", payload[1:17])
        out: dict[str, Any] = {
            "quat": (w, x, y, z),
            "quat_report_id": report_id,
            "fusion": FUSION_REPORT_IDS.get(report_id),
        }
        acc = _decode_accuracy(payload, 17)
        if acc is not None:
            out["quat_accuracy"] = acc
        return out
    w, x, y, z = struct.unpack("<4f", payload[:16])
    return {"quat": (w, x, y, z), "quat_report_id": None, "fusion": None}


def decode_raw_imu(payload: bytes) -> dict[str, Any]:
    """Decode a raw motion-vector record.

    Tagged form (7/8 B): ``[sensor_id:u8][x,y,z:int16x3][acc:u8?]`` — one of
    accel/gyro/mag. Legacy form (18 B): combined ``accel(3) gyro(3) mag(3)`` int16.
    """
    if len(payload) >= 7 and len(payload) < 18:
        sensor_id = payload[0]
        x, y, z = struct.unpack("<3h", payload[1:7])
        key = _RAW_SENSOR_KEY.get(sensor_id)
        out: dict[str, Any] = {"sensor_id": sensor_id}
        if key is not None:
            out[key] = (x, y, z)
            cal_key = _RAW_SENSOR_CAL_KEY.get(sensor_id)
            acc = _decode_accuracy(payload, 7)
            if cal_key is not None and acc is not None:
                out[f"{cal_key}_accuracy"] = acc
        else:
            out["raw"] = (x, y, z)
        return out
    ax, ay, az, gx, gy, gz, mx, my, mz = struct.unpack("<9h", payload[:18])
    return {
        "accel": (ax, ay, az),
        "gyro": (gx, gy, gz),
        "mag": (mx, my, mz),
    }


def decode_lin_accel(payload: bytes) -> dict[str, Any]:
    x, y, z = struct.unpack("<3f", payload[:12])
    out: dict[str, Any] = {"linear_accel": (x, y, z)}
    acc = _decode_accuracy(payload, 12)
    if acc is not None:
        out["linear_accel_accuracy"] = acc
    return out


# TRIGGER_EVENT flags[0]: how the firmware obtained t_master_us (records.h).
TRIGGER_STAMP_UNKNOWN = 0  # pre-2026-09 firmware: software timer, untagged
TRIGGER_STAMP_PIO_IRQ = 1  # latched in the PIO IRQ at the exposure-start edge
TRIGGER_STAMP_TIMER = 2    # software repeating_timer (legacy mode or fallback)
TRIGGER_STAMP_NAMES = {
    TRIGGER_STAMP_UNKNOWN: "unknown",
    TRIGGER_STAMP_PIO_IRQ: "pio_irq",
    TRIGGER_STAMP_TIMER: "timer",
}


def decode_trigger(payload: bytes) -> dict[str, Any]:
    flags = tuple(payload[0:4])
    (trigger_id,) = struct.unpack("<Q", payload[4:12])
    return {
        "flags": flags,
        "trigger_id": trigger_id,
        # flags[1] == 1 means the timer watchdog took over from a silent PIO IRQ.
        "stamp_source": TRIGGER_STAMP_NAMES.get(flags[0], f"0x{flags[0]:02X}"),
        "stamp_fallback": bool(flags[1]) if len(flags) > 1 else False,
    }


def decode_led_flash(payload: bytes) -> dict[str, Any]:
    (flash_id,) = struct.unpack("<I", payload[:4])
    return {"flash_id": flash_id}


def decode_button(payload: bytes) -> dict[str, Any]:
    """Panel button edge (firmware panel.c push_button_event).

    Layout (5 B): ``[button_id:u8][pressed:u8][reserved:3]``. Today only USER1
    (id 1, the panel capture toggle) emits, and only on the press edge.
    """
    if len(payload) < 2:
        raise ValueError(f"BUTTON_EVENT len {len(payload)} < 2")
    return {"button_id": payload[0], "pressed": bool(payload[1])}


def decode_pps_reply(payload: bytes) -> dict[str, Any]:
    (req_id,) = struct.unpack("<Q", payload[:8])
    return {"pps_id": req_id}


# Positional map over the u32 array in REC_STATS. Append-only, and mirrored by
# kStatsFields in native/src/module.cpp and comms_core_send_stats() in the
# firmware -- all three must be changed together.
STATS_FIELDS = (
    "drdy_count",
    "missed_drdy",
    "emg_push",
    "emg_drops",
    "imu_mta_push",
    "imu_mta_drops",
    "imu_cra_push",
    "imu_cra_drops",
    "events_push",
    "events_drops",
    "usb_tx_bytes",
    "sensor_loop_hz",
    "comms_loop_hz",
    "tx_drops",
    "max_drdy_gap_us",
    "imu_svc_max_us",
    "leadoff_max_us",
    "leadoff_count",
    "poll_max_us",
    "frame_misalign",
    "emg_misalign_drops",
    "misalign_max_streak",
    "emg_resyncs",
    "spi_wedges",
    "trig_pio_events",
    "trig_timer_events",
    "imu_i2c_aborts",
    "imu_cra_i2c_aborts",
    # Appended 2026-09 (older firmware sends only the 28 above; decode is by count).
    "tx_drops_emg",
    "tx_drops_imu",
    "tx_drops_other",
    "imu_mta_seq_gaps",
    "imu_cra_seq_gaps",
    "imu_mta_resets",
    "imu_cra_resets",
    "stats_seq",
)


def decode_stats(payload: bytes) -> dict[str, Any]:
    count = len(payload) // 4
    values = struct.unpack(f"<{count}I", payload[: count * 4])
    return {name: values[i] for i, name in enumerate(STATS_FIELDS) if i < count}


def decode_record(record_type: int, payload: bytes) -> dict[str, Any]:
    if record_type == int(RecordType.EMG_BLOCK):
        return decode_emg_block(payload)
    if record_type in (
        int(RecordType.IMU_MTA_QUAT),
        int(RecordType.IMU_CRA_QUAT),
        int(RecordType.IMU_DP_QUAT),
    ):
        return decode_quat(payload)
    if record_type in (
        int(RecordType.IMU_MTA_RAW),
        int(RecordType.IMU_CRA_RAW),
        int(RecordType.IMU_DP_RAW),
    ):
        return decode_raw_imu(payload)
    if record_type in (
        int(RecordType.IMU_MTA_LIN_ACCEL),
        int(RecordType.IMU_CRA_LIN_ACCEL),
        int(RecordType.IMU_DP_LIN_ACCEL),
    ):
        return decode_lin_accel(payload)
    if record_type == int(RecordType.TRIGGER_EVENT):
        return decode_trigger(payload)
    if record_type == int(RecordType.LED_FLASH_EVENT):
        return decode_led_flash(payload)
    if record_type == int(RecordType.BUTTON_EVENT):
        return decode_button(payload)
    if record_type == int(RecordType.PPS_REPLY):
        return decode_pps_reply(payload)
    if record_type == int(RecordType.STATS):
        return decode_stats(payload)
    if record_type == int(RecordType.LOG):
        return {"text": payload.decode("utf-8", errors="replace")}
    if record_type == int(RecordType.CMD_ACK):
        return {"ok": bool(payload[0]) if payload else False}
    return {"raw": payload}
