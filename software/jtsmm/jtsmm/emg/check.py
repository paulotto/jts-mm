# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Pre-recording EMG signal-quality checks (cv03 §5)."""

from __future__ import annotations

import copy
import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from jtsmm.config.catalog import (
    DEFAULT_EMG_SAMPLE_RATE_SPS,
    DRL_CHANNEL,
    DRL_NAME,
    DRL_STREAM_INDEX,
    FULL_CHANNEL_NAMES,
    MUSCLE_CHANNELS,
    channel_lsb_uv,
)
from jtsmm.config.registry import (
    emg_check_drl_isolation_duration_s,
    emg_check_impedance_duration_s,
    emg_check_impedance_enabled,
    emg_check_leadoff_duration_s,
    emg_check_line_hz,
    emg_check_noise_duration_s,
    emg_check_settle_ms,
)
from jtsmm.emg.capture import (
    FLOATING_EMG_RMS_UV,
    capture_emg_uv,
    comparator_contact_off,
    comparator_state,
    drl_contact_off,
    flat_rail_contact_off,
    leadoff_summary,
)
from jtsmm.emg.capture import (
    DRL_AC_WARN_MV,
    DRL_RAIL_NA_FRAC,
    drl_ac_mv_from_uv,
    drl_rail_fraction,
    drl_readback_status,
)
from jtsmm.emg.impedance import CONTACT_Z_MAX_KOHM

__all__ = [
    "ChannelCheck",
    "CheckResult",
    "check_result_to_dict",
    "run_emg_check",
    "hard_contact_failures",
    "run_impedance",
]

# CH8/IN8N runs at a fixed gain 24 in every mode (ADS_CH1SET_NORMAL 0x60 while
# probing, ADS_CH8SET_DRL 0x67 with bias on), independent of the profile's
# `default_gain`; capture_emg_uv scales it per channel (catalog.channel_lsb_uv),
# so the probe session needs no gain override. (This used to force gain 6 and
# overstated every DRL figure 4x.)

# cv03 thresholds
NOISE_LIMIT_UV = {1: 10.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 15.0, 6: 15.0}
LINE_PEAK_LIMIT_UV = 50.0
LEADOFF_DURATION_S = 5.0
NOISE_DURATION_S = 5.0
# 31.2 Hz AC tone resolves in well under a second; 2 s gives ~62 cycles of
# margin for the Goertzel estimate without materially lengthening the check.
IMPEDANCE_DURATION_S = 2.0
# DRL comparator settle time once the 6 muscle channels are muted (see
# `mute_muscle` below) — the current sink on IN8N alone is uncontested, so it
# settles in well under a second; 2 s leaves ample margin.
DRL_ISOLATION_DURATION_S = 2.0

# Hardware-verified caveat (bench testing, this board revision): IN8P is
# permanently unwired (h03 §5), so CH8 has no valid P-side reference in any
# MUX mode. The isolated probe (bias off, MUX forced to normal input) at
# least reads the *real* IN8N pin instead of the ADS1299's own bias-loop
# output — fixing the "always PASS" bug — but neither the LOFF comparator
# (never observed to trip, even at 24 uA over 15+ s with the electrode
# genuinely unplugged) nor `drl_contact_off`'s noise/RMS heuristic (which
# varied 56-3700 µV RMS run-to-run for the *same* physical state in bench
# testing, i.e. an unreliable threshold) is a fully dependable PASS/FAIL on
# its own. Treat the DRL row as best-effort and confirm electrode placement
# visually too.
DRL_LEADOFF_CAVEAT = (
    "DRL leadoff is best-effort: IN8P is unwired on this board, so neither "
    "the comparator nor the ADC-noise heuristic reliably discriminates "
    "connected/disconnected in every condition (see docs/hardware/"
    "h03_emg_frontend.md §6) — also confirm electrode placement visually."
)


@dataclass
class ChannelCheck:
    channel: int
    name: str
    leadoff_ok: bool
    noise_rms_uv: float | None      # mains removed: the electrode's own noise
    noise_ok: bool
    line_peak_uv: float | None
    line_ok: bool
    z_kohm: float | None = None
    z_quality: str = ""
    notes: list[str] = field(default_factory=list)
    # True for the *contact* failures that make a channel's data unusable and
    # that the operator can fix in a couple of minutes: a channel stuck at full
    # scale, or |Z| above CONTACT_Z_MAX_KOHM. A comparator hit only warns until
    # the comparators are bench-verified (see jtsmm.emg.capture). Noise and
    # line-hum failures are deliberately excluded -- they are environmental,
    # often not fixable at the rig, and blocking on them would strand a session.
    # A real field, not a property: dataclasses.asdict() drops properties, so a
    # property here would vanish from /meta/emg_check_json.
    contact_hard_fail: bool = False
    # In-band RMS as measured, mains included. Reported for context only --
    # `noise_rms_uv` above (mains removed) is what `noise_ok` judges, and
    # `line_peak_uv` owns the verdict on the mains itself.
    noise_total_uv: float | None = None
    # Lead-off comparator for this channel: hit | clear | off | unknown.
    comparator: str = ""

    @property
    def ok(self) -> bool:
        return self.leadoff_ok and self.noise_ok and self.line_ok


@dataclass
class DrlSaturation:
    """State of the bias amplifier itself, independent of DRL *contact*.

    Contact (the ``drl`` row) asks "is the electrode on the subject". This asks
    "is the loop still regulating". They fail independently: a well-attached DRL
    on a rig with several hundred mV of mains common-mode will still clip, and a
    clipping loop actively makes the interference worse.
    """

    ac_mv: float
    # Always True from run_emg_check: the readback is informational until the
    # bias fix is bench-verified. `status` carries the actual reading.
    ok: bool
    rail_frac: float = 0.0
    notes: list[str] = field(default_factory=list)
    status: str = ""          # ok | warn | high | n/a (railed)


@dataclass
class CheckResult:
    channels: list[ChannelCheck]
    passed: bool
    drl: ChannelCheck | None = None
    note: str = ""
    drl_saturation: DrlSaturation | None = None
    # Grid frequency measured on the noise capture (nominal when too weak to
    # measure); the mains fit and the line-peak figure use it.
    line_hz_measured: float | None = None

    @property
    def failed_channels(self) -> list[int]:
        failed = [c.channel for c in self.channels if not c.ok]
        if self.drl is not None and not self.drl.leadoff_ok:
            failed.append(DRL_CHANNEL)
        return failed


def _bandpass_rms(
    samples_uv: np.ndarray,
    fs_hz: float,
    lo: float,
    hi: float,
    t_us: np.ndarray | None = None,
) -> float:
    """RMS of the ``lo``-``hi`` content, in the same units as the input.

    Parseval for a one-sided ``rfft`` is ``mean(x**2) = sum(|X|**2) / n**2``
    with the interior bins doubled. Dividing by ``n`` instead -- as this did
    until 2026-09 -- returns ``RMS * sqrt(n)``: at a 5 s, 2 kHz capture that is
    a factor of 128, which turned the 10 uV noise limit into an unreachable
    0.08 uV and made every channel of every session fail the noise check with
    implausible six-figure readings. It also must not zero-pad to a power of
    two: the padding dilutes ``mean(x**2)`` by ``size/n``.
    """
    x = np.asarray(samples_uv, dtype=np.float64)
    if x.size < 16 or fs_hz <= 0:
        return 0.0
    if t_us is not None and np.asarray(t_us).size == x.size:
        # An FFT needs uniform spacing; rows packed around dropped samples
        # would shift every frequency. Interpolate onto the real time base.
        from jtsmm.emg.mains import uniform_grid

        _, x = uniform_grid(np.asarray(t_us), x, fs_hz)
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size < 16:
            return 0.0
    x = x - np.mean(x)
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return 0.0
    power = (np.abs(spec) ** 2) / float(n * n)
    power[1:] *= 2.0                 # one-sided: fold the negative frequencies
    if n % 2 == 0:
        power[-1] /= 2.0             # ...except Nyquist, which has no partner
    return float(np.sqrt(np.sum(power[mask])))


def _bandpass_rms_no_mains(
    samples_uv: np.ndarray,
    fs_hz: float,
    lo: float,
    hi: float,
    line_hz: float,
    t_us: np.ndarray | None = None,
) -> float:
    """Band RMS with the line tone and its harmonics fitted out first.

    The 10 uV noise limit is about the *electrode and amplifier*: is this
    contact quiet enough to see 5-20 uV of resting EMG. Mains lives in the same
    band, so with hum present the plain band RMS measures the room instead --
    s01_f_30 reported "noise 232259.6 uV > 10" on a channel whose impedance was
    a perfectly good 24.8 kOhm, which reads as a dead electrode and sends the
    operator off to re-prep something that was fine.

    Mains keeps its own verdict through ``line_peak_uv``; this one answers the
    question the threshold was actually written for. Falls back to the
    unfiltered RMS if the fit cannot run.

    Pass ``t_us`` (the rows' ``t_master_us``) and the *measured* ``line_hz``
    (see ``_capture_line_hz``): without timestamps, 0.9% dropped samples turned
    a 3 uV electrode under 1 mV of hum into a 570 uV reading, and a fit at the
    nominal 50.000 Hz on a 50.016 Hz grid left 72 uV.
    """
    x = np.asarray(samples_uv, dtype=np.float64)
    if x.size < 16 or fs_hz <= 0:
        return 0.0
    try:
        from jtsmm.emg.mains import remove_mains

        cleaned, _ = remove_mains(x, fs_hz, t_us=t_us, line_hz=line_hz)
    except Exception:  # noqa: BLE001 - a measurement must never crash the check
        cleaned = x
    return _bandpass_rms(cleaned, fs_hz, lo, hi)


def _line_peak(
    samples_uv: np.ndarray,
    fs_hz: float,
    line_hz: float = 50.0,
    t_us: np.ndarray | None = None,
) -> float:
    """Peak amplitude of the line tone, by least squares at ``line_hz``.

    Until 2026-09 this read one zero-padded FFT bin and divided by the padded
    length instead of the sample count, so it reported ~50% of the real
    amplitude (and at the nearest bin, not at the grid frequency).
    """
    x = np.asarray(samples_uv, dtype=np.float64)
    if x.size < 16 or fs_hz <= 0:
        return 0.0
    from jtsmm.emg.mains import fit_tone_amplitude

    return fit_tone_amplitude(x, fs_hz, line_hz, t_us=t_us)


def _capture_line_hz(
    samples_uv: np.ndarray,
    fs_hz: float,
    nominal_hz: float,
    t_us: np.ndarray | None = None,
    n_channels: int = len(MUSCLE_CHANNELS),
) -> float:
    """Grid frequency measured on this capture's muscle channels, else nominal.

    Measured on the channel carrying the most hum (the interference is common,
    so the loudest channel gives the cleanest peak).
    """
    from jtsmm.emg.mains import fit_tone_amplitude, measure_line_hz

    arr = np.asarray(samples_uv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 64:
        return float(nominal_hz)
    cols = min(n_channels, arr.shape[1])
    if cols <= 0:
        return float(nominal_hz)
    amps = [fit_tone_amplitude(arr[:, i], fs_hz, nominal_hz, t_us=t_us) for i in range(cols)]
    best = int(np.argmax(amps))
    f = measure_line_hz(arr[:, best], fs_hz, nominal_hz, t_us=t_us)
    return float(f) if f is not None else float(nominal_hz)


class _Capture(NamedTuple):
    """One short hardware capture: µV rows, status words, applied EMG config, t_us."""

    samples: np.ndarray
    status: list[bytes]
    emg: dict[str, Any]
    t_us: np.ndarray | None = None
    # CONFIG4 readback after apply: comparators powered? None = not known.
    comparators_on: bool | None = None


def _check_config(
    resolved: dict[str, Any],
    *,
    leadoff_on: bool,
    ac_tone: bool = False,
    mute_muscle: bool = False,
    probe_drl: bool = False,
) -> dict[str, Any]:
    cfg = copy.deepcopy(resolved)
    emg = cfg.setdefault("mcu", {}).setdefault("emg", {})
    leadoff = emg.setdefault("leadoff", {})
    leadoff["enabled"] = True
    leadoff["during_recording"] = leadoff_on
    leadoff["ac_tone"] = bool(ac_tone)
    # DC comparator/rail check only: a slow rate is enough and settles faster.
    # The AC |Z| tone is a fixed 31.2 Hz (fCLK-derived, not fDR-derived — see
    # ads1299.c), so it does not need a particular sample rate to resolve; keep
    # the acquisition profile's own rate for that session.
    if leadoff_on and not ac_tone:
        emg["sample_rate_sps"] = 250
    emg.setdefault("test_signal", {})["enabled"] = False
    if mute_muscle:
        # Power down CH1-6 (ADS1299 PDn) so `ads_leadoff_channel_masks()` on
        # the MCU excludes them from LOFF_SENSP/LOFF_SENSN. Muscle channels
        # each drive their own P-source/N-sink pair through the patient's
        # skin/tissue; left enabled, their currents share the same body
        # return path as the DRL-only N-sink on IN8N and can mask a genuine
        # DRL disconnection (the six muscle sources keep the shared node from
        # ever reaching the comparator's rail-side threshold). Muting them
        # isolates the IN8N comparator so it only ever sees DRL's own current.
        # Sent as the wire-level mask: `active_channels: []` means "no
        # selection" to the payload builder, i.e. all six stayed powered.
        emg.pop("active_channels", None)
        emg["pd_channels"] = 0x3F
    if probe_drl:
        # During normal acquisition CH8SET.MUX=111 ("BIAS_DRN") makes the
        # ADS1299's own bias amplifier actively drive IN8N through a
        # low-impedance closed loop (h03 §5.2) -- the lead-off comparator and
        # the streamed ADC column for CH8 then just see *that* amplifier's
        # output, never the real external electrode, so they always look
        # "connected" no matter what. Disabling bias drive makes the MCU pull
        # CH8SET back to a plain electrode input (MUX=000); `leadoff.probe_drl`
        # tells it to still enable the LOFF_SENSN[7] excitation current on
        # IN8N even though bias is off, so we can read the *real* pin.
        emg["bias"] = {"enabled": False}
        leadoff["probe_drl"] = True
    return cfg


def _hardware_session(
    resolved: dict[str, Any],
    *,
    port: str | None,
    leadoff_on: bool,
    duration_s: float,
    settle_ms: int = 400,
    ac_tone: bool = False,
    mute_muscle: bool = False,
    probe_drl: bool = False,
) -> _Capture:
    from jtsmm import native
    from jtsmm.capture.native_backend import NativeCapture
    from jtsmm.capture.serial_util import find_hat_port
    from jtsmm.config import apply_mcu, open_hat_serial, start_mcu_streams, stop_mcu_streams

    if not native.use_native():
        raise RuntimeError("EMG check requires the native capture extension (jtsmm[native]).")

    cfg = _check_config(
        resolved, leadoff_on=leadoff_on, ac_tone=ac_tone, mute_muscle=mute_muscle, probe_drl=probe_drl,
    )
    emg = cfg.get("mcu", {}).get("emg", {})
    gain = int(emg.get("default_gain", 6) or 6)
    serial_port = find_hat_port(port or (resolved.get("hat") or {}).get("serial_port"))
    ser = open_hat_serial(str(serial_port))
    comparators_on: bool | None = None
    try:
        stop_mcu_streams(ser)
        time.sleep(0.2)
        applied = apply_mcu(ser, cfg, persist=False, strict=False)
        if not applied:
            raise RuntimeError("MCU config apply failed")
        if leadoff_on:
            from jtsmm.emg.capture import query_comparators_enabled

            comparators_on = query_comparators_enabled(ser)
        # Open the USB reader before the streams start: libusb takes ~1.5 s to
        # come up, and EMG produced meanwhile overflowed the MCU ring (~1000
        # emg_drops at 2 kHz on every start, measured on the HAT).
        cap = NativeCapture(port=str(serial_port))
        cap.start()
        try:
            # EMG only: an IMU stream with no IMU plugged in lights the FAULT LED
            # (panel.c), and neither the IMUs nor the camera trigger feed this check.
            start_mcu_streams(ser, trigger=False, imu_mta=False, imu_cra=False)
        except BaseException:
            cap.stop()
            raise
    finally:
        ser.close()

    try:
        cap.settle(quiet_ms=settle_ms)
        samples, status, t_us = capture_emg_uv(cap, duration_s, gain=gain)
    finally:
        cap.stop()
        try:
            from jtsmm.config import stop_mcu_streams_robust

            stop_mcu_streams_robust(str(serial_port))
        except Exception:
            try:
                with open_hat_serial(str(serial_port)) as ser2:
                    stop_mcu_streams(ser2)
            except Exception:
                pass
    return _Capture(samples, status, emg, t_us, comparators_on)


def run_emg_check(
    resolved: dict[str, Any],
    *,
    port: str | None = None,
    duration_s: float | None = None,
    line_hz: float | None = None,
    with_impedance: bool | None = None,
) -> CheckResult:
    """Run lead-off, |Z| estimate, baseline-noise, and line-frequency checks.

    Combines the DC comparator/rail lead-off decision with a quantitative AC
    |Z| estimate (same math as ``jtsmm emg impedance``) so a channel that
    slips past the comparator (e.g. a floating input the ADS1299 didn't trip
    on) but has an implausibly high impedance is still flagged.

    The DRL/mastoid electrode (CH8, IN8N) gets its own short, isolated probe
    sub-session with CH1-6 powered down (``mute_muscle``) *and* bias drive
    disabled (``probe_drl``, which also forces CH8SET out of BIAS_DRN back to
    a plain electrode input): during normal acquisition the ADS1299's own
    bias amplifier actively drives IN8N through a low-impedance loop, so
    without this the comparator/ADC only ever see the amplifier's own output,
    never the real electrode — this is why the DRL row of `jtsmm emg check`
    used to always read PASS. See ``jtsmm.emg.capture.drl_contact_off`` for
    the noise-based detection this enables (IN8P is permanently unwired on
    this board, so absolute/differential level alone can't distinguish
    connected from disconnected here).
    """
    leadoff_duration = (
        float(duration_s) if duration_s is not None else emg_check_leadoff_duration_s(resolved)
    )
    line_hz_val = float(line_hz) if line_hz is not None else emg_check_line_hz(resolved)
    impedance_on = (
        with_impedance if with_impedance is not None else emg_check_impedance_enabled(resolved)
    )
    settle_ms = emg_check_settle_ms(resolved)
    noise_duration = emg_check_noise_duration_s(resolved)
    impedance_duration = emg_check_impedance_duration_s(resolved)
    drl_duration = emg_check_drl_isolation_duration_s(resolved)

    names = FULL_CHANNEL_NAMES[: len(MUSCLE_CHANNELS)]

    try:
        loff_cap = _hardware_session(
            resolved, port=port, leadoff_on=True, duration_s=leadoff_duration, settle_ms=settle_ms,
        )
        loff_samples, loff_status, emg_lo = loff_cap.samples, loff_cap.status, loff_cap.emg
        loff_t = loff_cap.t_us
        loff = leadoff_summary(loff_status, n_channels=len(MUSCLE_CHANNELS))
        # Comparator bits count only when the comparators are known to run; an
        # all-zero window from powered-down comparators is not "connected".
        cmp_state = comparator_state(loff_cap.comparators_on, loff)
        hw_off = (
            comparator_contact_off(loff)
            if cmp_state == "on"
            else [False] * len(MUSCLE_CHANNELS)
        )
        loff_cols = (
            loff_samples[:, : len(MUSCLE_CHANNELS)]
            if loff_samples.size
            else np.empty((0, len(MUSCLE_CHANNELS)))
        )
        gain = int(emg_lo.get("default_gain", 6) or 6)
        rail_off = flat_rail_contact_off(loff_cols, gain=gain)

        z_results: list[Any] = []
        if impedance_on:
            z_results, _ = run_impedance(
                resolved, port=port, duration_s=impedance_duration, line_hz=line_hz_val,
            )

        noise_cap = _hardware_session(
            resolved, port=port, leadoff_on=False, duration_s=noise_duration, settle_ms=settle_ms,
        )
        samples, emg, noise_t = noise_cap.samples, noise_cap.emg, noise_cap.t_us
        fs = float(emg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS) or DEFAULT_EMG_SAMPLE_RATE_SPS)
        fs_lo = float(emg_lo.get("sample_rate_sps", 250) or 250)
    except RuntimeError as exc:
        # `drl` is the 3rd positional field (ChannelCheck | None) -- pass the
        # message as `note` explicitly, not positionally, or it lands in
        # `drl` and later `.leadoff_ok`/`.ok` accesses blow up.
        return CheckResult([], False, note=str(exc))

    z_by_channel = {r.channel: r for r in z_results}
    # Fit mains at the grid frequency each capture actually saw, not nominal.
    line_f = _capture_line_hz(samples, fs, line_hz_val, noise_t) if samples.size else line_hz_val
    line_f_lo = (
        _capture_line_hz(loff_cols, fs_lo, line_hz_val, loff_t) if loff_cols.size else line_hz_val
    )

    channels: list[ChannelCheck] = []
    all_ok = True
    # Channels powered down by active_channels carry no signal: listing them as
    # clean PASS (flat trace, tone absent) would hide that they were not checked.
    active_cfg = ((resolved.get("mcu") or {}).get("emg") or {}).get("active_channels")
    active_set = set(int(c) for c in active_cfg) if active_cfg else set(MUSCLE_CHANNELS)
    for i, ch in enumerate(MUSCLE_CHANNELS):
        if ch not in active_set:
            channels.append(ChannelCheck(
                channel=ch,
                name=names[i] if i < len(names) else f"CH{ch}",
                leadoff_ok=True, noise_rms_uv=None, noise_ok=True,
                line_peak_uv=None, line_ok=True, z_quality="n/a",
                notes=["powered down (active_channels), not checked"],
            ))
            continue
        col = samples[:, i] if samples.size and samples.shape[1] > i else np.array([])
        loff_col = loff_cols[:, i] if loff_cols.size and loff_cols.shape[1] > i else np.array([])
        noise_total = _bandpass_rms(col, fs, 20.0, 450.0, noise_t) if col.size else None
        # Judge the electrode on what is left once mains is fitted out; `line`
        # below still judges the mains itself.
        noise = (
            _bandpass_rms_no_mains(col, fs, 20.0, 450.0, line_f, noise_t) if col.size else None
        )
        loff_noise = (
            _bandpass_rms_no_mains(loff_col, fs_lo, 20.0, 450.0, line_f_lo, loff_t)
            if loff_col.size
            else None
        )
        line_loff = _line_peak(loff_col, fs_lo, line_f_lo, loff_t) if loff_col.size else None
        line = _line_peak(col, fs, line_f, noise_t) if col.size else None
        z_res = z_by_channel.get(ch)
        z_bad = bool(z_res is not None and z_res.quality != "n/a" and not z_res.ok)
        # Open BNCs without DC rail still pick up strong line hum during lead-off.
        # `loff_noise` is now mains-removed, so this needs BROADBAND noise *and*
        # hum: a dangling input has both, while a connected electrode in a hummy
        # room has only the hum and no longer trips. Genuine opens are caught
        # more reliably by |Z| (z_bad) anyway -- this is a further line of
        # defence, so making it specific is the right trade. It does not gate
        # recording (see contact_hard_fail).
        floating_open = (
            not rail_off[i]
            and loff_noise is not None
            and loff_noise >= FLOATING_EMG_RMS_UV
            and line_loff is not None
            and line_loff >= 80.0
        )
        # A comparator hit warns (note below) but does not fail the channel
        # until the comparators are bench-verified -- see jtsmm.emg.capture.
        loff_ok = not rail_off[i] and not floating_open and not z_bad
        noise_lim = NOISE_LIMIT_UV.get(ch, 10.0)
        noise_ok = noise is not None and noise < noise_lim
        line_ok = line is not None and line < LINE_PEAK_LIMIT_UV
        notes: list[str] = []
        if hw_off[i]:
            p, n = loff["p_off"][i], loff["n_off"][i]
            leg = []
            if p:
                leg.append("P")
            if n:
                leg.append("N")
            notes.append(f"WARN comparator lead-off ({'/'.join(leg) or '?'}), unverified")
        if not loff_ok:
            if floating_open:
                notes.append(f"floating input ({loff_noise:.0f} µV RMS)")
            if rail_off[i]:
                notes.append("stuck at full-scale (DC lead-off / open)")
            if z_bad:
                z_s = f"{z_res.z_kohm:.0f}" if z_res.z_kohm is not None else "n/a"
                notes.append(f"|Z| {z_s} kΩ > {CONTACT_Z_MAX_KOHM:.0f}")
        if noise is not None and not noise_ok:
            extra = (
                f", {noise_total:.0f} with mains"
                if noise_total is not None and noise_total > noise * 1.5
                else ""
            )
            notes.append(f"noise {noise:.1f} µV > {noise_lim:.0f}{extra}")
        if line is not None and not line_ok:
            notes.append(f"{line_hz_val:.0f} Hz {line:.1f} µV > {LINE_PEAK_LIMIT_UV:.0f}")
        cc = ChannelCheck(
            channel=ch,
            name=names[i] if i < len(names) else f"CH{ch}",
            contact_hard_fail=bool(rail_off[i] or z_bad),
            leadoff_ok=loff_ok,
            noise_rms_uv=noise,
            noise_ok=noise_ok,
            noise_total_uv=noise_total,
            line_peak_uv=line,
            line_ok=line_ok,
            z_kohm=z_res.z_kohm if z_res is not None else None,
            z_quality=z_res.quality if z_res is not None else "",
            notes=notes,
            comparator=("hit" if hw_off[i] else "clear") if cmp_state == "on" else cmp_state,
        )
        if not cc.ok:
            all_ok = False
        channels.append(cc)

    # --- bias-amplifier readback (informational) ----------------------------
    # Measured on the bias-ON noise capture (CH8 = BIAS_DRN readback there).
    # Deliberately NOT the probe_drl session below, where CH8 is routed to a
    # plain electrode input. Never gates `all_ok`: a railed column (the MVC
    # session: 96.8% railed) has no AC content by construction and used to read
    # PASS for that reason alone, and the thresholds are unverified until the
    # bias fix is bench-checked (see the open bench check in jtsmm.emg.capture).
    drl_sat: DrlSaturation | None = None
    bias_enabled = bool(
        ((resolved.get("mcu") or {}).get("emg") or {}).get("bias", {}).get("enabled", True)
    )
    if bias_enabled and samples.size and samples.shape[1] > DRL_STREAM_INDEX:
        col_uv = samples[:, DRL_STREAM_INDEX]  # `samples` is already in µV here
        # Back to ADC counts with CH8's own LSB (gain 24), not the muscle LSB.
        lsb8 = channel_lsb_uv(int(emg.get("default_gain", 6) or 6))[DRL_STREAM_INDEX]
        rail = drl_rail_fraction(col_uv / max(lsb8, 1e-12))
        ac = drl_ac_mv_from_uv(col_uv)
        status = drl_readback_status(ac, rail)
        sat_notes: list[str] = []
        if status == "n/a":
            sat_notes.append(
                f"CH8 railed {rail * 100:.0f}% of the time (> {DRL_RAIL_NA_FRAC * 100:.0f}%): "
                "AC content not measurable"
            )
        elif ac >= DRL_AC_WARN_MV:
            sat_notes.append(
                f"bias node swinging {ac:.1f} mV RMS -- the DRL loop may be driven by "
                "common-mode instead of rejecting it; re-prep the DRL site and remove "
                "mains-coupled equipment"
            )
        drl_sat = DrlSaturation(
            ac_mv=ac,
            ok=True,  # informational: never gates the check (see above)
            rail_frac=rail,
            notes=sat_notes,
            status=status,
        )

    drl_check: ChannelCheck | None = None
    note = ""
    bias_on = bool(((resolved.get("mcu") or {}).get("emg") or {}).get("bias", {}).get("enabled", True))
    if bias_on:
        try:
            drl_cap = _hardware_session(
                resolved,
                port=port,
                leadoff_on=True,
                duration_s=drl_duration,
                mute_muscle=True,
                probe_drl=True,
            )
            drl_samples, drl_status = drl_cap.samples, drl_cap.status
        except RuntimeError as exc:
            # Do NOT fall back to the non-isolated `loff_status` here: with
            # bias on and CH1-6 also active it always reads "connected" (see
            # drl_contact_off), which would silently turn a probe failure
            # into a false PASS. Report FAIL with the reason instead.
            drl_check = ChannelCheck(
                channel=DRL_CHANNEL,
                name=DRL_NAME,
                leadoff_ok=False,
                noise_rms_uv=None,
                noise_ok=True,
                line_peak_uv=None,
                line_ok=True,
                notes=[f"probe failed: {exc}"],
            )
            all_ok = False
            note = DRL_LEADOFF_CAVEAT
        else:
            drl_col = (
                drl_samples[:, DRL_STREAM_INDEX]
                if drl_samples.size and drl_samples.shape[1] > DRL_STREAM_INDEX
                else np.empty(0)
            )
            drl_off, drl_notes = drl_contact_off(drl_status, drl_col)
            drl_check = ChannelCheck(
                channel=DRL_CHANNEL,
                name=DRL_NAME,
                leadoff_ok=not drl_off,
                noise_rms_uv=None,
                noise_ok=True,
                line_peak_uv=None,
                line_ok=True,
                notes=drl_notes,
            )
            if not drl_check.ok:
                all_ok = False
            note = DRL_LEADOFF_CAVEAT
    else:
        note = "DRL check skipped: bias.enabled=false in config (DRL electrode not in use)"
    if cmp_state != "on":
        why = "powered down in firmware (CONFIG4)" if cmp_state == "off" else "state unknown"
        cmp_note = (
            f"lead-off comparators {why}: contact judged by |Z|, rail and noise only"
        )
        note = f"{note} | {cmp_note}" if note else cmp_note

    return CheckResult(
        channels, all_ok, drl=drl_check, note=note, drl_saturation=drl_sat,
        line_hz_measured=line_f if samples.size else None,
    )


def restore_mcu_config(resolved: dict[str, Any], *, port: str | None = None) -> str | None:
    """Re-apply ``resolved`` with streams stopped; return an error string or None.

    The check's sub-sessions leave the MCU in the last one's setup (the DRL
    probe: bias off, CH1-6 powered down; or the AC tone on). A standalone
    `jtsmm emg check` / `emg impedance` calls this afterwards.
    """
    try:
        from jtsmm.capture.serial_util import find_hat_port
        from jtsmm.config import apply_mcu, open_hat_serial, stop_mcu_streams

        serial_port = find_hat_port(port or (resolved.get("hat") or {}).get("serial_port"))
        with open_hat_serial(str(serial_port)) as ser:
            stop_mcu_streams(ser)
            time.sleep(0.2)
            if not apply_mcu(ser, resolved, persist=False, strict=False):
                return "config apply failed"
    except Exception as exc:  # noqa: BLE001 - report, never mask the check result
        return str(exc)
    return None


def hard_contact_failures(result: CheckResult) -> list[str]:
    """Channels whose *contact* is unusable and operator-fixable.

    Separated from ``CheckResult.passed`` so `jtsmm record` can block on the
    fixable half (bad electrode contact) while still only warning about the
    environmental half (mains noise, line hum, best-effort DRL detection).
    Returns the channel names, for the message.
    """
    return [c.name for c in result.channels if c.contact_hard_fail]


def check_result_to_dict(result: CheckResult) -> dict[str, Any]:
    """JSON-safe snapshot of a `CheckResult`, for embedding in HDF5 metadata."""
    return {
        "passed": result.passed,
        "note": result.note,
        "channels": [dataclasses.asdict(c) for c in result.channels],
        "drl": dataclasses.asdict(result.drl) if result.drl is not None else None,
        "drl_saturation": (
            dataclasses.asdict(result.drl_saturation)
            if result.drl_saturation is not None
            else None
        ),
        "line_hz_measured": result.line_hz_measured,
    }


def run_impedance(
    resolved: dict[str, Any],
    *,
    port: str | None = None,
    duration_s: float = 3.0,
    current_na: int | None = None,
    line_hz: float | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Capture with AC lead-off tone on and estimate |Z| per muscle + DRL.

    ``line_hz`` (default ``host.emg_check.line_hz``) is the grid whose
    harmonics are fitted out next to the 31.25 Hz tone.
    """
    from jtsmm.emg.impedance import AC_LEADOFF_HZ, estimate_drl_impedance, estimate_impedance

    emg_cfg = (resolved.get("mcu") or {}).get("emg") or {}
    if current_na is None:
        current_na = int((emg_cfg.get("leadoff") or {}).get("current_na", 6) or 6)
    else:
        # The MCU must inject the current the |Z| math divides by: `--current-na
        # 24` used to reach only the host math while the MCU kept the YAML value
        # (|Z| read 4x low).
        resolved = copy.deepcopy(resolved)
        lo_cfg = resolved.setdefault("mcu", {}).setdefault("emg", {}).setdefault("leadoff", {})
        lo_cfg["current_na"] = int(current_na)
    z_cap = _hardware_session(
        resolved,
        port=port,
        leadoff_on=True,
        duration_s=duration_s,
        ac_tone=True,
    )
    samples, emg = z_cap.samples, z_cap.emg
    fs = float(emg.get("sample_rate_sps", 250) or 250)
    i_na = float(current_na)
    if line_hz is None:
        from jtsmm.config.registry import emg_check_line_hz

        line_hz = emg_check_line_hz(resolved)
    results = estimate_impedance(
        samples, fs, current_na=i_na, n_channels=len(MUSCLE_CHANNELS), t_us=z_cap.t_us,
        line_hz=float(line_hz),
    )
    bias_on = bool((emg_cfg.get("bias") or {}).get("enabled", True))
    if bias_on:
        results.append(
            estimate_drl_impedance(
                samples,
                fs,
                stream_index=DRL_STREAM_INDEX,
                channel=DRL_CHANNEL,
                current_na=i_na,
            )
        )
    meta = {
        "fs_hz": fs,
        "tone_hz": AC_LEADOFF_HZ,
        "current_na": current_na,
        "duration_s": duration_s,
        "n_samples": int(samples.shape[0]),
        "drl_included": bias_on,
    }
    return results, meta


def format_check_table(result: CheckResult) -> str:
    lines = [
        f"{'CH':>3}  {'name':<14} {'leadoff':>7}  {'|Z| kΩ':>7}  {'noise µV':>9}  {'50Hz µV':>8}  status",
        "-" * 68,
    ]
    for c in result.channels:
        loff = "PASS" if c.leadoff_ok else "FAIL"
        z = f"{c.z_kohm:7.1f}" if c.z_kohm is not None else "    n/a"
        noise = f"{c.noise_rms_uv:9.1f}" if c.noise_rms_uv is not None else "      n/a"
        line = f"{c.line_peak_uv:8.1f}" if c.line_peak_uv is not None else "     n/a"
        status = "PASS" if c.ok else "FAIL"
        if c.notes:
            status += " — " + "; ".join(c.notes)
        lines.append(f"{c.channel:3d}  {c.name:<14} {loff:>7}  {z}  {noise}  {line}  {status}")
    if result.drl is not None:
        d = result.drl
        loff = "PASS" if d.leadoff_ok else "FAIL"
        status = "PASS" if d.ok else "FAIL"
        if d.notes:
            status += " — " + "; ".join(d.notes)
        lines.append(
            f"{'DRL':>3}  {d.name:<14} {loff:>7}  {'n/a':>7}  {'n/a':>9}  {'n/a':>8}  {status}"
        )
    if result.drl_saturation is not None:
        sat = result.drl_saturation
        if sat.status:
            status = f"INFO {sat.status}"
        else:
            status = "PASS" if sat.ok else "FAIL"
        if sat.notes:
            status += " — " + "; ".join(sat.notes)
        ac = "    n/a  " if sat.status == "n/a" else f"{sat.ac_mv:7.1f}mV"
        lines.append(
            f"{'—':>3}  {'bias amp (DRL)':<14} {'':>7}  {'n/a':>7}  "
            f"{ac}  {'n/a':>8}  {status}"
        )
    lines.append("-" * 68)
    lines.append(f"overall: {'PASS' if result.passed else 'FAIL'}")
    if result.note:
        lines.append(f"note: {result.note}")
    return "\n".join(lines)
