# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Build and send MCU `config apply` JSON over USB."""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from typing import Any

import serial

from .catalog import IMU_PROFILE_REPORTS, MUSCLE_CHANNEL_MASK, channels_to_mask


def _emg_to_wire(emg: dict[str, Any]) -> dict[str, Any]:
    """Translate the human-friendly EMG block into the firmware wire format.

    The config/CLI express channel selections as 1-based lists (``bias.channels``,
    ``active_channels``); the firmware parser expects integer bitmasks
    (``bias.channels`` = BIAS_SENSP/N mask, ``pd_channels`` = power-down mask over
    CH1–CH6). Mirrors the bringup ``--active-channels`` semantics: powering down
    the unused channels and, unless overridden, driving DRL from the active set.

    An explicit wire-level ``pd_channels`` mask (not a YAML key; e.g. the EMG
    check's DRL probe powering down all of CH1-CH6 = 0x3F) is passed through
    as-is and then neither derived from ``active_channels`` nor used to mask
    ``bias.channels``. An empty ``active_channels`` list keeps its historical
    meaning (no selection = all active): the schema does not define it.
    """
    emg = copy.deepcopy(emg or {})
    active = emg.pop("active_channels", None)
    explicit_pd = emg.pop("pd_channels", None)
    active_mask = channels_to_mask(active) if active else MUSCLE_CHANNEL_MASK

    bias = dict(emg.get("bias") or {})
    bias_list = bias.get("channels")
    if isinstance(bias_list, int):
        bias_mask = int(bias_list)  # already a wire mask
    elif bias_list is not None:
        bias_mask = channels_to_mask(bias_list)
    elif active:
        # No explicit DRL set → derive it from the active channels so the bias
        # loop is never fed by a floating/disconnected input.
        bias_mask = active_mask
    else:
        bias_mask = None

    if bias_mask is not None:
        if explicit_pd is None:
            # Powered-down muscle channels must never contribute to the DRL average.
            bias_mask &= active_mask
        bias["channels"] = bias_mask
        emg["bias"] = bias

    if explicit_pd is not None:
        emg["pd_channels"] = int(explicit_pd) & MUSCLE_CHANNEL_MASK
    elif active:
        emg["pd_channels"] = (~active_mask) & MUSCLE_CHANNEL_MASK
    return emg


def _slim_imu_for_wire(slot: dict[str, Any]) -> dict[str, Any]:
    """Wire form of one IMU slot: only enabled reports, always spelled out.

    Disabled entries are dropped (the firmware ignores them and they bloat CDC
    lines). A host preset without enabled reports is expanded to its report map
    (profiles.py, the same resolution doctor uses), so the MCU runs exactly the
    host's set -- also for presets it has no copy of -- and verify_mcu_config can
    check it. `all` and `custom` stay as they are.
    """
    slot = copy.deepcopy(slot or {})
    reports = slot.get("reports")
    if isinstance(reports, dict):
        reports = {
            name: cfg
            for name, cfg in reports.items()
            if isinstance(cfg, dict) and cfg.get("enabled", True)
        }
        slot["reports"] = reports
    if slot.get("enabled", True) and not reports:
        from .profiles import _resolve_imu_slot

        resolved = _resolve_imu_slot(slot)
        if resolved["profile"] in IMU_PROFILE_REPORTS:
            slot["profile"] = resolved["profile"]
            slot["reports"] = {
                name: cfg for name, cfg in resolved["reports"].items() if cfg.get("enabled", True)
            }
    return slot


def _slim_trigger_for_wire(trigger: dict[str, Any]) -> dict[str, Any]:
    """Omit schema-only keys the firmware parser does not consume."""
    trigger = copy.deepcopy(trigger or {})
    trigger.pop("start_on", None)
    return trigger


def build_mcu_apply_payload(resolved: dict[str, Any]) -> dict[str, Any]:
    """Extract MCU-owned blocks for firmware config_apply."""
    mcu = resolved.get("mcu") or {}
    return {
        "schema_version": resolved.get("schema_version", 1),
        "platform": copy.deepcopy(mcu.get("platform", {})),
        "emg": _emg_to_wire(mcu.get("emg", {})),
        "imu_mta": _slim_imu_for_wire(mcu.get("imu_mta", {})),
        "imu_cra": _slim_imu_for_wire(mcu.get("imu_cra", {})),
        "trigger": _slim_trigger_for_wire(mcu.get("trigger", {})),
    }


def canonical_applied_config(resolved: dict[str, Any]) -> dict[str, Any]:
    """The effective config actually persisted to a session (wire-form MCU blocks).

    ``config_hash`` must be computed over this shape, not the raw resolved doc, so
    the integrity checker (which can only see the stored wire ``mcu_config`` plus
    the host bundle) reproduces the same hash. Mirrors ``_check_config_hash``.
    """
    payload = build_mcu_apply_payload(resolved)
    return {
        "schema_version": payload["schema_version"],
        "hat": resolved.get("hat", {}),
        "mcu": {
            "platform": payload["platform"],
            "emg": payload["emg"],
            "imu_mta": payload["imu_mta"],
            "imu_cra": payload["imu_cra"],
            "trigger": payload["trigger"],
        },
        "host": resolved.get("host", {}),
        "session": resolved.get("session", {}),
    }


def format_apply_command(payload: dict[str, Any]) -> str:
    line = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return f"config apply {line}\n"


_FAIL_WORD = re.compile(r"\b(FAIL|ERR|ERROR)\b")


def _classify_config_line(line: str) -> str | None:
    """``ok`` / ``fail`` / ``busy`` for a final `config init|apply` line, else None.

    Matches whole lines: the firmware also prints ``config apply parse FAIL`` and
    ``config apply re-parse FAIL``, which a substring test for ``config apply
    FAIL`` never saw (the host then sat out the full 120 s timeout).
    """
    s = line.strip()
    if not (s.startswith("config apply") or s.startswith("config init")):
        return None
    if s.startswith(("config apply:", "config init:")):
        return None  # informational ("host overrides profile defaults ...")
    if s.endswith(" OK"):
        return "ok"
    if s.endswith(" BUSY"):
        return "busy"
    if _FAIL_WORD.search(s):
        return "fail"
    return None


def wait_config_result(
    ser: serial.Serial, *, timeout_s: float = 120.0, detail: dict[str, Any] | None = None
) -> str:
    """Wait for the end of a `config init/apply`: ``ok``, ``fail``, ``busy`` or ``timeout``.

    ``busy`` is the firmware refusing because an apply is already running (K6):
    nothing was applied. ``detail["line"]`` receives the final reply line, e.g.
    ``config apply FAIL (<reason>)``.
    """
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))
            text = bytes(buf).decode("utf-8", errors="replace")
            for line in text.splitlines():
                verdict = _classify_config_line(line)
                if verdict is not None:
                    if detail is not None:
                        detail["line"] = line.strip()
                    return verdict
        else:
            time.sleep(0.02)
    return "timeout"


def wait_config_done(ser: serial.Serial, *, timeout_s: float = 120.0) -> bool:
    """Wait for config init/apply completion on serial (True only on OK)."""
    return wait_config_result(ser, timeout_s=timeout_s) == "ok"


def read_config_dump(ser: serial.Serial, *, timeout_s: float = 3.0) -> dict[str, Any] | None:
    """Read the MCU's active config back via `config dump`.

    Returns the parsed JSON object, or None if the MCU did not answer in time.
    """
    ser.reset_input_buffer()
    ser.write(b"config dump\r\n")
    ser.flush()
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))
            text = bytes(buf).decode("utf-8", errors="replace")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
        else:
            time.sleep(0.02)
    return None


def verify_mcu_config(
    ser: serial.Serial,
    resolved: dict[str, Any],
    *,
    dump: dict[str, Any] | None = None,
) -> list[str]:
    """Compare the MCU's active config against what we asked it to apply.

    ``config apply OK`` only means the firmware finished its apply steps, not that
    it kept our values: profile expansion rewrites the fields a profile owns. That
    matters most for ``default_gain``, because the host derives ``lsb_uv`` — and
    therefore every microvolt written to the HDF5 — from the gain it *requested*.
    A silent disagreement rescales the entire recording with nothing in the file
    to show it happened. ``leadoff.during_recording`` is the other one: the host
    computes rail detection and impedance from its own belief, so if the firmware
    dropped it the lead-off indicators go green on a session that has none.
    The trigger rate / pulse width (the camera frame clock) and each IMU slot's
    report set and rates are checked too: the host labels streams by what it
    requested, so a silently different rate mis-describes the whole file.

    ``dump`` is a ``config dump`` already read by the caller (else it is read
    here). Fields missing on either side are skipped. Returns a list of
    human-readable mismatches; empty means verified.
    """
    requested = build_mcu_apply_payload(resolved)
    if dump is None:
        dump = read_config_dump(ser)
    if dump is None:
        return ["MCU did not answer `config dump`, so the applied config is unverified"]

    problems: list[str] = []
    compared: set[str] = set()

    def cmp_int(path: str, want: Any, got: Any) -> None:
        if want is None or got is None:
            return
        compared.add(path)
        if int(want) != int(got):
            problems.append(f"{path}: requested {int(want)}, MCU reports {int(got)}")

    def cmp_bool(path: str, want: Any, got: Any) -> None:
        if want is None or got is None:
            return
        compared.add(path)
        if bool(want) != bool(got):
            problems.append(f"{path}: requested {bool(want)}, MCU reports {bool(got)}")

    req_emg = requested.get("emg") or {}
    got_emg = dump.get("emg") or {}
    for key in ("sample_rate_sps", "default_gain", "pd_channels"):
        cmp_int(f"emg.{key}", req_emg.get(key), got_emg.get(key))

    req_bias = req_emg.get("bias") or {}
    got_bias = got_emg.get("bias") or {}
    cmp_bool("emg.bias.enabled", req_bias.get("enabled"), got_bias.get("enabled"))
    cmp_int("emg.bias.channels", req_bias.get("channels"), got_bias.get("channels"))

    req_lo = req_emg.get("leadoff") or {}
    got_lo = got_emg.get("leadoff") or {}
    cmp_bool("emg.leadoff.enabled", req_lo.get("enabled"), got_lo.get("enabled"))
    cmp_bool(
        "emg.leadoff.during_recording",
        req_lo.get("during_recording"),
        got_lo.get("during_recording"),
    )
    cmp_int("emg.leadoff.current_na", req_lo.get("current_na"), got_lo.get("current_na"))

    req_trig = requested.get("trigger") or {}
    got_trig = dump.get("trigger")
    if isinstance(got_trig, dict) and req_trig.get("enabled", True):
        want_rate = req_trig.get("rate_hz")
        cmp_int("trigger.rate_hz", round(float(want_rate)) if want_rate is not None else None,
                got_trig.get("rate_hz"))
        cmp_int(
            "trigger.pulse.width_us",
            (req_trig.get("pulse") or {}).get("width_us"),
            (got_trig.get("pulse") or {}).get("width_us"),
        )

    for slot in ("imu_mta", "imu_cra"):
        req_slot = requested.get(slot) or {}
        got_slot = dump.get(slot)
        if not isinstance(got_slot, dict) or req_slot.get("enabled") is False:
            continue
        req_reports = {
            str(n): c for n, c in (req_slot.get("reports") or {}).items() if isinstance(c, dict)
        }
        got_reports = got_slot.get("reports")
        if not req_reports or not isinstance(got_reports, dict):
            continue
        missing = sorted(set(req_reports) - set(got_reports))
        extra = sorted(set(got_reports) - set(req_reports))
        if missing:
            problems.append(f"{slot}.reports: requested {missing}, MCU does not run them")
        if extra:
            problems.append(f"{slot}.reports: MCU also runs {extra}, which were not requested")
        for name in sorted(set(req_reports) & set(got_reports)):
            got_r = got_reports.get(name)
            cmp_int(
                f"{slot}.reports.{name}.rate_hz",
                req_reports[name].get("rate_hz"),
                got_r.get("rate_hz") if isinstance(got_r, dict) else None,
            )

    # Every other leaf present in BOTH the request and the dump (platform bus
    # rates and ring sizes, fusion rates, and ac_tone / test_signal / probe_drl
    # once the firmware dumps them). Profiles are skipped: the firmware reports
    # "custom" as soon as the host overrides a profile field, which is expected;
    # ring buffers are dumped rounded down to whole records.
    def walk(req: dict[str, Any], got: dict[str, Any], path: str) -> None:
        for key, want in req.items():
            if key in ("profile", "reports", "schema_version", "buffers") or key not in got:
                continue
            p = f"{path}.{key}" if path else str(key)
            have = got[key]
            if isinstance(want, dict):
                if isinstance(have, dict):
                    walk(want, have, p)
                continue
            if p in compared or want is None or have is None:
                continue
            if isinstance(want, bool) or isinstance(have, bool):
                cmp_bool(p, want, have)
            elif isinstance(want, (int, float)) and isinstance(have, (int, float)):
                if abs(float(want) - float(have)) > 1e-6 * max(1.0, abs(float(want))):
                    problems.append(f"{p}: requested {want}, MCU reports {have}")
            elif isinstance(want, str) and isinstance(have, str) and want != have:
                problems.append(f"{p}: requested {want!r}, MCU reports {have!r}")

    walk(requested, dump, "")
    return problems


def apply_mcu(
    ser: serial.Serial,
    resolved: dict[str, Any],
    *,
    timeout_s: float = 120.0,
    persist: bool = False,
    strict: bool = True,
    verify: bool = True,
    report: dict[str, Any] | None = None,
    busy_wait_s: float = 10.0,
) -> bool:
    """Send config apply and wait for completion.

    ``config apply BUSY`` (another apply still running) means nothing was
    applied: stop the streams, wait for the running apply to finish, retry once,
    then give up loudly. ``report`` (optional dict) receives ``result``, the
    post-apply ``config_dump`` and ``verify_problems`` for the session metadata.
    """
    if strict:
        from .strict import raise_on_errors, validate_strict

        raise_on_errors(validate_strict(resolved))
    out = report if report is not None else {}
    cmd = format_apply_command(build_mcu_apply_payload(resolved))

    def _send() -> str:
        ser.reset_input_buffer()
        ser.write(cmd.encode("utf-8"))
        ser.flush()
        return wait_config_result(ser, timeout_s=timeout_s, detail=out)

    result = _send()
    if result == "busy":
        print(
            "warning: MCU answered `config apply BUSY` (an apply is already running); "
            "stopping streams, waiting for it, retrying once",
            file=sys.stderr,
        )
        stop_mcu_streams(ser)
        wait_config_result(ser, timeout_s=busy_wait_s)  # the running apply's OK/FAIL
        result = _send()
        if result == "busy":
            print(
                "error: MCU still answers `config apply BUSY` after a retry -- the config "
                "was NOT applied. Power-cycle the HAT if this persists.",
                file=sys.stderr,
            )
    out["result"] = result
    if result != "ok":
        if result == "timeout":
            print(
                f"error: MCU sent no `config apply OK/FAIL` within {timeout_s:g} s",
                file=sys.stderr,
            )
        elif result == "fail":
            line = out.get("line", "config apply FAIL")
            print(f"error: MCU rejected the config: {line}", file=sys.stderr)
        return False
    if verify:
        # Advisory only: a mismatch means the data will be mis-scaled, not that
        # the apply failed, and failing here would be a new way to break a
        # working session. Say it loudly and carry on.
        try:
            dump = read_config_dump(ser)
            out["config_dump"] = dump
            problems = verify_mcu_config(ser, resolved, dump=dump) if dump is not None else [
                "MCU did not answer `config dump`, so the applied config is unverified"
            ]
        except Exception as exc:  # noqa: BLE001 - verification must never break apply
            problems = [f"config verification could not run: {exc}"]
        out["verify_problems"] = list(problems)
        for problem in problems:
            print(f"warning: MCU config mismatch — {problem}", file=sys.stderr)
        if any(p.startswith("emg.default_gain") for p in problems):
            print(
                "warning: the host computes lsb_uv from the gain it requested, so "
                "recorded microvolts will be wrong by the ratio above. Fix the "
                "config or the firmware profile before recording.",
                file=sys.stderr,
            )
    if persist:
        ser.write(b"config save\r\n")
        ser.flush()
        deadline = time.monotonic() + 5.0
        buf = bytearray()
        while time.monotonic() < deadline:
            n = ser.in_waiting
            if n:
                buf.extend(ser.read(n))
                if b"config save OK" in buf:
                    return True
                if b"config save FAIL" in buf:
                    print("error: MCU reported `config save` FAIL", file=sys.stderr)
                    return False
            time.sleep(0.02)
        # Silence is not success: the flash may hold the old config (or none).
        print("error: MCU never answered `config save` -- persist unconfirmed", file=sys.stderr)
        return False
    return True


def open_hat_serial(port: str, baud: int = 115200) -> serial.Serial:
    return serial.Serial(port, baud, timeout=0.05)


_REPLY_PROBLEM = re.compile(r"\b(FAIL|BUSY|REFUSED|ERR|ERROR)\b|^unknown:")


def _command_problems(ser: Any, cmd: bytes, window_s: float) -> list[str]:
    """Send one CDC command; return any failure lines the firmware printed.

    The positive ACK of ``ads_start``/``imu_stream``/``trig_*`` is a CMD_ACK
    record on the vendor link, not CDC text; what CDC does carry is the failure
    text (``trig_start FAIL``, ``unknown: <cmd>`` on old firmware, ``... REFUSED``).
    Reading it costs nothing: the settle window below was a bare sleep before.
    """
    ser.write(cmd)
    ser.flush()
    if not hasattr(ser, "in_waiting") or not hasattr(ser, "read"):
        time.sleep(window_s)
        return []
    deadline = time.monotonic() + window_s
    buf = bytearray()
    while time.monotonic() < deadline:
        try:
            n = int(ser.in_waiting or 0)
        except Exception:  # noqa: BLE001 - a flaky port must not break stream start
            n = 0
        if n:
            buf.extend(ser.read(n))
        else:
            time.sleep(0.01)
    text = bytes(buf).decode("utf-8", errors="replace")
    name = cmd.decode("ascii", errors="replace").strip()
    return [
        f"`{name}`: {line.strip()}"
        for line in text.splitlines()
        if _REPLY_PROBLEM.search(line.strip())
    ]


def _warn_problems(problems: list[str]) -> list[str]:
    for p in problems:
        print(f"WARNING: MCU refused a stream command -- {p}", file=sys.stderr)
    return problems


def mcu_stream_enables(resolved: dict[str, Any]) -> dict[str, bool]:
    """``mcu.<block>.enabled`` for emg / imu_mta / imu_cra / trigger (default on).

    The firmware parses these flags but does not act on them (``imu_stream on``
    even force-enables a disabled slot), so the host must simply not start a
    disabled stream.
    """
    mcu = resolved.get("mcu") or resolved
    out: dict[str, bool] = {}
    for block in ("emg", "imu_mta", "imu_cra", "trigger"):
        cfg = mcu.get(block)
        out[block] = not (isinstance(cfg, dict) and cfg.get("enabled") is False)
    return out


def start_mcu_sensor_streams(
    ser: serial.Serial, *, emg: bool = True, imu_mta: bool = True, imu_cra: bool = True
) -> list[str]:
    """Start EMG + IMU streams (no camera trigger). Returns firmware failure lines."""
    cmds: list[bytes] = []
    if emg:
        cmds.append(b"ads_start\r\n")
    if imu_mta:
        cmds.append(b"imu_stream mta on\r\n")
    if imu_cra:
        cmds.append(b"imu_stream cra on\r\n")
    problems: list[str] = []
    for cmd in cmds:
        problems.extend(_command_problems(ser, cmd, 0.15))
    return _warn_problems(problems)


def start_mcu_trigger(ser: serial.Serial) -> list[str]:
    """Start the camera trigger pulse train.

    Kept separate from :func:`start_mcu_sensor_streams` so the recorder can
    arm the cameras *before* the first pulse. When the trigger ran first, the
    host's sequence-consume association paired a camera's first frame with the
    oldest trigger it had buffered -- one from before the camera was even
    streaming -- and every camera row inherited that offset (typically 10-40
    frames of picamera2 start-up latency) with nothing in the file to show it.

    Returns firmware failure lines (``trig_start FAIL`` = no camera frames).
    """
    return _warn_problems(_command_problems(ser, b"trig_start\r\n", 0.15))


def start_mcu_streams(
    ser: serial.Serial,
    *,
    emg: bool = True,
    trigger: bool = True,
    imu_mta: bool = True,
    imu_cra: bool = True,
) -> list[str]:
    """Start MCU streams after config apply.

    ``emg=False`` skips ``ads_start``. ``trigger=False`` skips ``trig_start``
    so the caller can arm the camera trigger after ``cam.start()``.
    Bench/viewer convenience; `jtsmm record` starts the trigger separately after
    the cameras are up (see :func:`start_mcu_trigger`).
    """
    problems = start_mcu_sensor_streams(ser, emg=emg, imu_mta=imu_mta, imu_cra=imu_cra)
    if trigger:
        problems = problems + start_mcu_trigger(ser)
    return problems


def stop_mcu_streams(ser: serial.Serial) -> None:
    """Stop acquisition streams and return panel to idle when all off."""
    for cmd in (
        b"trig_stop\r\n",
        b"imu_stream cra off\r\n",
        b"imu_stream mta off\r\n",
        b"ads_stop\r\n",
    ):
        ser.write(cmd)
        ser.flush()
        time.sleep(0.1)


def stop_mcu_streams_robust(port: str, *, attempts: int = 3) -> None:
    """Stop MCU streams after native capture releases the vendor USB interface."""
    time.sleep(0.15)
    last_exc: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            with open_hat_serial(port) as ser:
                stop_mcu_streams(ser)
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(0.25)
    if last_exc is not None:
        raise last_exc

