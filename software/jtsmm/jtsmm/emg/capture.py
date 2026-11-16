# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Short EMG-only capture windows for check/impedance commands."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np

from jtsmm.config.catalog import channel_lsb_uv, lsb_uv
from jtsmm.protocol.types import RecordType
from jtsmm.viewer.leadoff import COMPARATOR_CHANNELS, decode_status_word

# Floating BNC inputs pick up mains/hash; connected skin is normally <15 µV RMS.
FLOATING_EMG_RMS_UV = 500.0

# --- DRL / bias-amplifier readback -----------------------------------------
# With bias drive on, CH8SET.MUX = BIAS_DRN and CH8 reads IN8P minus IN8N, where
# IN8N is the DRL electrode net and IN8P is permanently unwired on this board
# (h03 §5). So CH8 is a floating pin referenced to the DRL net: with the bias
# amplifier off it shows the passive body common-mode, with it on the amplifier
# output against a floating input. CH8 runs at fixed gain 24, not the muscle
# gain (catalog.channel_lsb_uv); every figure below is in that true scale.
#
# What this can and cannot say. Measured on one subject, minutes apart, same
# electrodes (numbers rescaled from the 4x-overstated gain-6 scaling used then):
#
#   MVC   (excellent recording): railed 96.8-99% of the time, AC ~0 mV
#   take1 (unusable, 0.6-3.4 mV of mains on the muscle channels):
#                               not railed, AC 37-56 mV (147-225 as then read)
#
# A railed ADC has no AC content *by construction*, so "AC 0 mV" on the MVC
# says nothing about the loop -- it is the clipping talking, and the check read
# PASS for the wrong reason. Only an unrailed column's AC content means
# anything, hence DRL_RAIL_NA_FRAC: above it the verdict is "n/a".
#
# The old firmware wrote CONFIG3 = 0xE8: internal mid-supply reference but
# PD_BIAS = 0, i.e. the bias amplifier was powered down and CH8 read a passive
# electrode against the unwired IN8P. It is now 0xEC (amplifier on). Until the
# driven loop is verified on the bench this metric is INFORMATIONAL: `jtsmm emg
# check` and `jtsmm integrity` report it, neither lets it fail anything.
# Open bench check: record CH8 with bias on after the CONFIG3 fix, on a subject with
# good contact and with the DRL lead removed; re-derive these thresholds from
# the unrailed AC levels before letting them gate anything.
#
# Shared by `jtsmm emg check` and `jtsmm integrity` so the two cannot drift.
# True CH8 units (gain 24, channel_lsb_uv): the old 10 / 50 mV were tuned while
# CH8 was read 4x too high, so the same discrimination is 2.5 / 12.5 mV.
DRL_AC_WARN_MV = 2.5
DRL_AC_ERR_MV = 12.5
# Rail occupancy above which the AC figure is meaningless (reported as n/a).
DRL_RAIL_NA_FRAC = 0.5
DRL_RAIL_LEVEL_FRAC = 0.97
_ADC_FULL_SCALE = float(1 << 23)


def drl_ac_mv(drl_counts: Any, lsb_uv: float) -> float:
    """AC (DC-removed) RMS of the BIAS_DRN readback, in millivolts.

    ``lsb_uv`` must be CH8's own LSB (``channel_lsb_uv(gain)[DRL_STREAM_INDEX]``,
    gain 24), not the muscle LSB. Only meaningful when bias drive is on, CH8SET
    is BIAS_DRN and the column is not railed (see :func:`drl_readback_status`);
    not in the isolated ``probe_drl`` session, where CH8 is an electrode input.
    """
    if lsb_uv <= 0:
        return 0.0
    return drl_ac_mv_from_uv(np.asarray(drl_counts, dtype=np.float64) * lsb_uv)


def drl_ac_mv_from_uv(drl_uv: Any) -> float:
    """Same quantity as :func:`drl_ac_mv`, for a column already scaled to µV."""
    arr = np.asarray(drl_uv, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0.0
    return float(np.std(arr) / 1000.0)


def drl_rail_fraction(drl_counts: Any) -> float:
    """Fraction of samples where the BIAS_DRN readback (ADC counts) is railed.

    Not a fault indicator on its own (IN8P is unwired, see above), but it
    decides whether the AC figure means anything: see drl_readback_status().
    """
    arr = np.asarray(drl_counts, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.mean(np.abs(arr) > DRL_RAIL_LEVEL_FRAC * _ADC_FULL_SCALE))


def drl_readback_status(ac_mv: float, rail_frac: float) -> str:
    """``"n/a"`` (railed), ``"ok"``, ``"warn"`` or ``"high"`` -- informational only."""
    if rail_frac > DRL_RAIL_NA_FRAC:
        return "n/a"
    if ac_mv >= DRL_AC_ERR_MV:
        return "high"
    if ac_mv >= DRL_AC_WARN_MV:
        return "warn"
    return "ok"

# Comparator must trip on at least this fraction of status frames.
LEADOFF_FRAME_FRACTION = 0.25

# --- lead-off comparators ------------------------------------------------------
# Until 2026-09 the firmware wrote CONFIG4 = 0x00, which leaves the ADS1299
# lead-off comparators POWERED DOWN (PD_LOFF_COMP, bit 1, is active-high "on").
# Every status word then read all-zero and every electrode looked "connected".
# Firmware now writes 0x02 when lead-off is wanted, but the host cannot assume
# which firmware it talks to, so:
#   - an all-zero status is "unknown", not "connected", unless the MCU reports
#     the comparators powered (CONFIG4 readback via `ads_status`) -- or some
#     comparator bit has been seen set, which proves they are running;
#   - CH7/CH8 bits (IN7x/IN8x) are never used: IN8P is unwired and CH7 is not
#     an electrode, so the firmware masks them and so does every host reader;
#   - a comparator hit is a WARNING in the contact gate, not a hard failure.
# Open bench check: with the new firmware, unplug one electrode lead at a time
# (P and N) with lead-off on, and confirm the matching LOFF_STATP/N bit trips
# within 2 s and clears when it is reconnected; also confirm no bit trips on a
# good skin contact at 6 nA. Only then promote comparator hits to hard failures.
CONFIG4_PD_LOFF_COMP = 0x02
_CONFIG4_RE = re.compile(rb"CONFIG4=0x([0-9A-Fa-f]{2})")


def parse_config4(text: bytes | str) -> int | None:
    """CONFIG4 value from an ``ads_status`` dump, or None."""
    raw = text.encode("utf-8", "replace") if isinstance(text, str) else bytes(text)
    m = _CONFIG4_RE.search(raw)
    return int(m.group(1), 16) if m else None


def query_comparators_enabled(ser: Any, *, timeout_s: float = 1.5) -> bool | None:
    """Ask the MCU whether the lead-off comparators are powered (CONFIG4 bit 1).

    Read-only (`ads_status` only reads registers); call it with streams stopped,
    after `config apply`. Returns None when the answer is not available (old
    firmware, busy bus, no reply) -- callers must then treat comparator status
    as unknown.
    """
    try:
        ser.reset_input_buffer()
        ser.write(b"ads_status\r\n")
        ser.flush()
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            chunk = ser.read(256)
            if chunk:
                buf.extend(chunk)
                val = parse_config4(bytes(buf))
                if val is not None:
                    return bool(val & CONFIG4_PD_LOFF_COMP)
            else:
                time.sleep(0.01)
    except Exception:  # noqa: BLE001 - a diagnostic query must never break a session
        return None
    return None


def comparator_state(config_enabled: bool | None, loff: dict[str, Any]) -> str:
    """``"on"`` (bits trustworthy), ``"off"`` (comparators powered down) or ``"unknown"``.

    ``config_enabled`` is :func:`query_comparators_enabled`'s answer. With no
    answer, any tripped muscle bit proves the comparators run; an all-zero
    window stays "unknown" -- it is what powered-down comparators look like.
    """
    if config_enabled is True:
        return "on"
    if config_enabled is False:
        return "off"
    if any(f > 0.0 for f in (loff.get("frac_off") or [])):
        return "on"
    return "unknown"


def capture_emg_uv(
    cap: Any,
    duration_s: float,
    *,
    gain: int = 6,
    n_channels: int = 8,
) -> tuple[np.ndarray, list[bytes], np.ndarray]:
    """Return ``(samples_uv (N,C), status_words, t_us (N,))`` from a native drain.

    Columns are scaled per channel: CH1-6 at ``gain``, CH7/CH8 at their fixed
    gain 24 (``catalog.channel_lsb_uv``).

    ``t_us`` is each row's ``t_master_us``. Every measurement on these rows
    needs it: the stream is packed around dropped samples, so row index is not
    time, and a mains fit on index time smears the tone across every gap (1 mV
    of hum with 0.9% dropouts read as 570 uV of "noise" instead of 3 uV).
    """
    t_list: list[int] = []
    ch_list: list[list[int]] = []
    status_list: list[bytes] = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        recs = cap.drain_records()
        if not recs:
            time.sleep(0.002)
            continue
        for rec in recs:
            if int(rec.record_type) != int(RecordType.EMG_BLOCK):
                continue
            dec = rec.decoded
            ch = dec.get("channels")
            if not ch:
                continue
            t_list.append(int(rec.t_master_us))
            ch_list.append(ch)
            st = dec.get("status")
            if st:
                status_list.append(bytes(st))
    if not ch_list:
        return (
            np.empty((0, n_channels), dtype=np.float32),
            status_list,
            np.empty(0, dtype=np.int64),
        )
    counts = np.asarray(ch_list, dtype=np.float32)
    lsb = np.asarray(channel_lsb_uv(gain, counts.shape[1]), dtype=np.float32)
    return counts * lsb, status_list, np.asarray(t_list, dtype=np.int64)


def leadoff_summary(status_words: list[bytes], n_channels: int = 6) -> dict[str, Any]:
    """Aggregate lead-off hits over a capture window (muscle channels only).

    ``n_channels`` is capped at COMPARATOR_CHANNELS: CH7/CH8 bits are masked.
    """
    n_channels = min(int(n_channels), COMPARATOR_CHANNELS)
    if not status_words:
        return {
            "any_off": [False] * n_channels,
            "p_off": [False] * n_channels,
            "n_off": [False] * n_channels,
            "frac_off": [0.0] * n_channels,
            "frames": 0,
        }
    p_any = [False] * n_channels
    n_any = [False] * n_channels
    p_count = [0] * n_channels
    n_count = [0] * n_channels
    for st in status_words:
        s = decode_status_word(st)
        for i in range(n_channels):
            if s.statp[i]:
                p_any[i] = True
                p_count[i] += 1
            if s.statn[i]:
                n_any[i] = True
                n_count[i] += 1
    n_frames = len(status_words)
    any_off = [p_any[i] or n_any[i] for i in range(n_channels)]
    frac_off = [
        max(p_count[i], n_count[i]) / float(n_frames) for i in range(n_channels)
    ]
    return {
        "any_off": any_off,
        "p_off": p_any,
        "n_off": n_any,
        "frac_off": frac_off,
        "frames": n_frames,
    }


def comparator_contact_off(loff: dict[str, Any], *, fraction: float = LEADOFF_FRAME_FRACTION) -> list[bool]:
    """Per-channel lead-off from ADS1299 comparator status (majority vote)."""
    frac = loff.get("frac_off") or [0.0] * len(loff.get("any_off", []))
    return [bool(f >= fraction) for f in frac]


def flat_rail_contact_off(
    samples_uv: np.ndarray,
    *,
    gain: int = 6,
    n_channels: int = 6,
) -> list[bool]:
    """Detect DC lead-off pulling a disconnected input to a flat supply rail.

    Open/floating inputs with DC excitation often sit at ±full-scale with ~0 AC
    energy while the LOFF_STAT bits in the SPI status word may still read 0.
    """
    arr = np.asarray(samples_uv, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        return [False] * n_channels
    limit = (1 << 23) * lsb_uv(gain) * 0.97
    ac_lim = 500.0
    out: list[bool] = []
    for i in range(n_channels):
        col = arr[:, i] if i < arr.shape[1] else np.zeros(arr.shape[0])
        if col.size < 8:
            out.append(False)
            continue
        std = float(np.std(col))
        hi = float(np.max(col))
        lo = float(np.min(col))
        stuck_hi = hi >= limit and std < ac_lim
        stuck_lo = lo <= -limit and std < ac_lim
        out.append(bool(stuck_hi or stuck_lo))
    return out


def drl_comparator_off(
    status_words: list[bytes],
    *,
    fraction: float = LEADOFF_FRAME_FRACTION,
) -> bool:
    """Always False: the CH8 (IN8N) comparator bit is masked (contract K8).

    It never tripped on the bench even with the lead unplugged (IN8P is
    unwired), and the firmware now masks CH7/CH8 bits. Kept for API
    compatibility; ``drl_probe_floating`` is the DRL contact test.
    """
    del status_words, fraction
    return False


# An open, high-impedance IN8N (nothing plugged into J_EMG_DRL1, or a plugged
# lead not actually touching skin) acts like a small antenna and picks up far
# more ambient interference (mains hum, RF) than a low-impedance skin contact
# damps out. Measured on bench hardware with the isolated DRL probe (bias
# off, CH8SET forced to normal input 0x60 = gain 24, 24 uA DC lead-off): open/no
# contact ≈ 0.6-0.9 mV RMS vs a real skin contact (mastoid or otherwise)
# ≈ 4-19 µV RMS -- over 30x separation either way. (Those runs scaled CH8 with
# gain 6 and read 4x higher: 2.4-3.7 mV vs 15-75 µV against a 500 µV threshold,
# which is the same decision as 125 µV in the true scale used now.) This is the *only*
# reliable signal available on this column: IN8P is permanently unwired on
# this board (h03 §5), so CH8's absolute/differential level is dominated by
# internal bias-network quirks and reads the same stable, current-independent
# offset regardless of whether IN8N is actually connected (verified on
# hardware: open and skin-contact settle to the same ~60% of full-scale).
DRL_PROBE_OPEN_RMS_UV = 125.0


def drl_probe_floating(drl_col_uv: np.ndarray) -> tuple[bool, float]:
    """Return (floating, rms_uv) for the isolated DRL probe's CH8 column.

    See ``DRL_PROBE_OPEN_RMS_UV`` for why RMS (not absolute level) is used.
    """
    arr = np.asarray(drl_col_uv, dtype=np.float64) if drl_col_uv is not None else np.empty(0)
    if arr.size < 8:
        return False, 0.0
    rms = float(np.std(arr))
    return rms >= DRL_PROBE_OPEN_RMS_UV, rms


def drl_contact_off(
    status_words: list[bytes],
    drl_col_uv: np.ndarray,
) -> tuple[bool, list[str]]:
    """Return (off, notes) for the mastoid DRL electrode (CH8 / IN8N).

    Must be called on data from the isolated DRL *probe* session (bias drive
    disabled, CH8SET forced out of BIAS_DRN back to a plain electrode input,
    `emg.leadoff.probe_drl=True`; see ``_check_config``/``_hardware_session``
    in ``jtsmm.emg.check``) with CH1-6 muted too (`pd_channels=0x3F`): each
    muscle channel sources *and* sinks current through the same patient
    skin/tissue network, and if left enabled they keep the shared body node
    from reaching either detector's threshold, masking a genuine DRL
    disconnection.

    During *normal* acquisition (bias on, MUX8=111 "BIAS_DRN") the ADS1299's
    own bias amplifier actively drives IN8N through a low-impedance closed
    loop (SBAS499 Fig. 33, h03 §5.2) -- both the comparator and the streamed
    ADC column then just see the amplifier's own output and never reflect the
    real electrode, which is why the DRL row of `jtsmm emg check` used to
    always read PASS regardless of the true connection state. The comparator
    bit for IN8N is masked (K8, see ``drl_comparator_off``), so
    ``drl_probe_floating`` (RMS-based) is the only DRL contact test.
    """
    del status_words
    notes: list[str] = []
    floating, rms = drl_probe_floating(drl_col_uv)
    if floating:
        notes.append(f"floating input ({rms:.0f} µV RMS > {DRL_PROBE_OPEN_RMS_UV:.0f})")
    return (len(notes) > 0, notes)
