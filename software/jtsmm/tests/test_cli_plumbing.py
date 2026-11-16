# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every `jtsmm` CLI option reaches its effect (config-plumbing audit, CLI half).

The click tree of ``jtsmm.cli.main.main`` is walked recursively (group options
included). Every option must have an entry in ``EFFECTS``; an unclassified or
stale entry fails ``test_every_option_is_classified``, so a new option has to
be classified here. Each entry names a non-default value and the observable
change it must cause, measured by running the real command through CliRunner
with only the hardware stubbed:

- ``apply_mcu`` (in every namespace that imports it) records
  ``build_mcu_apply_payload(cfg)`` -- the exact wire payload;
- serial port lookup/open, MCU stream start/stop, native capture,
  ``capture_emg_uv``, ``run_preflight``, ``check_session`` and
  ``SessionSupervisor.run`` (which records the resolved config, the /meta
  ``mcu_config_json`` / ``config_hash`` it would write, and its arguments)
  are recorders.

Kinds:
  mcu_payload  the MCU `config apply` payload changes
  call_arg     an argument handed to the next layer changes (SessionSupervisor,
               run_preflight, find_hat_port, ReplaySource, run_case, ...)
  meta         a /meta attribute (the supervisor's extra_meta) changes
  output       stdout / exit code / a written file changes (commands whose
               effect is what they print or write)
  none         no observable effect; ``reason`` says why
The YAML half (every jtsmm.yaml leaf) is tests/test_cli_plumbing_yaml.py.
"""

from __future__ import annotations

import copy
import json
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import click
import numpy as np
import pytest
import serial
import yaml
from click.testing import CliRunner

import jtsmm.capture.native_backend as native_backend
import jtsmm.capture.serial_util as serial_util
import jtsmm.capture.session  # noqa: F401 - imported so its bindings get patched
import jtsmm.capture.supervisor as supervisor_mod
import jtsmm.cli.config_cmd  # noqa: F401
import jtsmm.cli.imu_cmd as imu_cmd
import jtsmm.cli.info as info_mod
import jtsmm.cli.integrity  # noqa: F401
import jtsmm.cli.view as view_mod
import jtsmm.config.mcu_apply as mcu_apply
import jtsmm.doctor as doctor_pkg
import jtsmm.doctor.imu_rates as imu_rates
import jtsmm.emg.capture as emg_capture
import jtsmm.emg.check as emg_check
import jtsmm.integrity as integrity_pkg
import jtsmm.integrity.summary as integrity_summary
import jtsmm.native as native
import jtsmm.viewer.sources as sources
from jtsmm.cli.main import main
from jtsmm.config import applied_config_hash, build_mcu_apply_payload, config_hash
from jtsmm.config.catalog import trigger_period_us
from jtsmm.config.loader import load_yaml, template_path
from jtsmm.doctor.mcu_info import McuInfo
from jtsmm.integrity import IntegrityReport

_NONE = object()  # "not observed" (no apply / no session / no call)
PORT = "/dev/ttyTEST9"


# --------------------------------------------------------------------------- #
# Observations and the hardware-stubbed harness
# --------------------------------------------------------------------------- #


@dataclass
class Obs:
    exit_code: int = 0
    output: str = ""
    exception: BaseException | None = None
    applies: list[dict] = field(default_factory=list)
    preflight: list[dict] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)
    hat_ports: list[Any] = field(default_factory=list)
    writes: list[bytes] = field(default_factory=list)
    captures: list[float] = field(default_factory=list)
    integrity: list[dict] = field(default_factory=list)
    calls: dict[str, list[dict]] = field(default_factory=dict)

    def record(self, name: str, **kw: Any) -> None:
        self.calls.setdefault(name, []).append(kw)


def _get(doc: Any, dotted: str) -> Any:
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _NONE
        cur = cur[part]
    return cur


def _set(doc: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = doc
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


class FakeSerial:
    """CDC port: records writes, answers the few queries the CLI reads back."""

    def __init__(self, harness: "Harness", port: Any, *args: Any, **kwargs: Any) -> None:
        self.h = harness
        self.port = str(port)
        self.dtr = True
        self._rx = bytearray()
        self._period = 33333

    def __enter__(self) -> "FakeSerial":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._rx.clear()

    def flush(self) -> None:
        pass

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, n: int = 1) -> bytes:
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def write(self, data: bytes) -> int:
        data = bytes(data)
        self.h.obs.writes.append(data)
        line = data.strip()
        if line.startswith(b"trig_start"):
            parts = line.split()
            if len(parts) > 1:
                self._period = trigger_period_us(int(parts[1]))
            self._rx += f"trig running=1 period_us={self._period}\r\n".encode()
        elif line == b"trig_status":
            self._rx += f"trig running=1 period_us={self._period}\r\npio_pulses=3\n".encode()
        elif line == b"config dump":
            self._rx += (json.dumps({"trigger": {"pulse": {"width_us": 1000}}}) + "\r\n").encode()
        return len(data)


class FakeCapture:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def settle(self, **kwargs: Any) -> None:
        pass

    def drain_records(self) -> list:
        return []


def _replace_everywhere(mp: pytest.MonkeyPatch, original: Any, fake: Any) -> None:
    """Rebind ``original`` in every loaded jtsmm module (``from x import f`` copies)."""
    for mod in list(sys.modules.values()):
        if not getattr(mod, "__name__", "").startswith("jtsmm"):
            continue
        for attr, val in list(vars(mod).items()):
            if val is original:
                mp.setattr(mod, attr, fake)


class Harness:
    def __init__(self, mp: pytest.MonkeyPatch, world: "World") -> None:
        self.mp = mp
        self.world = world
        self.obs = Obs()
        self.scenario: dict[str, Any] = {}
        self._install()

    # -- stubs ------------------------------------------------------------- #
    def _install(self) -> None:
        mp, h = self.mp, self

        mp.setattr(time, "sleep", lambda s: None)

        def find_hat_port(explicit: str | None = None) -> str:
            h.obs.hat_ports.append(explicit)
            return explicit or "/dev/fakehat"

        _replace_everywhere(mp, serial_util.find_hat_port, find_hat_port)
        _replace_everywhere(mp, mcu_apply.open_hat_serial, lambda port, *a, **k: FakeSerial(h, port))
        mp.setattr(serial, "Serial", lambda port=None, *a, **k: FakeSerial(h, port))

        def apply_mcu(ser: Any, resolved: dict, **kw: Any) -> bool:
            h.obs.applies.append(
                {
                    "payload": build_mcu_apply_payload(resolved),
                    "port": getattr(ser, "port", None),
                    "persist": bool(kw.get("persist", False)),
                }
            )
            return True

        _replace_everywhere(mp, mcu_apply.apply_mcu, apply_mcu)

        def read_config_dump(ser: Any, **kw: Any) -> dict:
            return copy.deepcopy(h.obs.applies[-1]["payload"]) if h.obs.applies else {}

        _replace_everywhere(mp, mcu_apply.read_config_dump, read_config_dump)
        for name in (
            "stop_mcu_streams",
            "stop_mcu_streams_robust",
            "start_mcu_streams",
            "start_mcu_sensor_streams",
            "start_mcu_trigger",
        ):
            def rec(*a: Any, _n: str = name, **k: Any) -> list:
                h.obs.record(_n, args=a, kw=k)
                return []

            _replace_everywhere(mp, getattr(mcu_apply, name), rec)

        mp.setattr(native, "use_native", lambda: True)
        mp.setattr(native_backend, "NativeCapture", FakeCapture)

        def capture_emg_uv(cap: Any, duration_s: float, *, gain: int = 6, n_channels: int = 8):
            h.obs.captures.append(float(duration_s))
            n = 500
            rng = np.random.default_rng(len(h.obs.captures))
            samples = rng.normal(0.0, 1.0, (n, n_channels))
            if h.scenario.get("emg_railed"):
                samples[:, :6] = 1e7  # every muscle input stuck at the rail
            return samples, [], np.arange(n, dtype=np.int64) * 4000

        _replace_everywhere(mp, emg_capture.capture_emg_uv, capture_emg_uv)
        _replace_everywhere(mp, emg_capture.query_comparators_enabled, lambda ser, **k: True)

        def capture_line_hz(samples: Any, fs: float, nominal: float, *a: Any, **k: Any) -> float:
            h.obs.record("line_hz", nominal=float(nominal))
            return float(nominal)

        mp.setattr(emg_check, "_capture_line_hz", capture_line_hz)

        def run_preflight(resolved: dict | None = None, **kw: Any) -> list[str]:
            h.obs.preflight.append({"resolved": copy.deepcopy(resolved), **kw})
            return list(h.scenario.get("preflight", ()))

        _replace_everywhere(mp, doctor_pkg.run_preflight, run_preflight)

        def sup_run(sup: Any, duration_s: float | None, *, mock_only: bool = False, **kw: Any) -> Path:
            mcu_cfg, _ = supervisor_mod.split_host_mcu(sup.resolved)
            rec = {
                "resolved": copy.deepcopy(sup.resolved),
                "payload": build_mcu_apply_payload(sup.resolved),
                "mcu_config_json": json.dumps(mcu_cfg, separators=(",", ":")),
                "config_hash": applied_config_hash(sup.resolved),
                "extra_meta": dict(sup.extra_meta),
                "duration_s": duration_s,
                "mock_only": mock_only,
                "port": str(sup.port),
                "applies_before": len(h.obs.applies),
            }
            for attr in (
                "viewer", "viewer_host", "viewer_port", "optical_track", "optical_config",
                "experiment_plan", "write_jts_config", "jts_config_path", "ct_calibration",
                "ct_calibration_n_poses", "ct_tip_calibration", "persist_ct_calibration",
                "ct_calibration_recordings_dir", "ct_calibration_only", "enable_emg", "reproduce",
            ):
                rec[attr] = getattr(sup, attr)
            h.obs.sessions.append(rec)
            out = h.world.root / "sessions" / f"take{id(rec)}"
            out.mkdir(parents=True, exist_ok=True)
            h5 = out / "session.h5"
            h5.touch()
            return h5

        mp.setattr(supervisor_mod.SessionSupervisor, "run", sup_run)

        def check_session(path: Any, **kw: Any) -> IntegrityReport:
            h.obs.integrity.append({"path": Path(path), **kw})
            return IntegrityReport(path=Path(path), warnings=list(h.scenario.get("warnings", ())))

        _replace_everywhere(mp, integrity_pkg.check_session, check_session)
        _replace_everywhere(mp, integrity_summary.load_emg_from_session_h5, lambda p: (None, None))

        def run_server(source: Any, host: str, port: int, open_browser: bool) -> None:
            h.obs.record("run_server", source=source, host=host, port=port, open=open_browser)
            if isinstance(source, sources.LiveSource):
                source.start(source.make_hub())

        mp.setattr(view_mod, "_run_server", run_server)
        mp.setattr(sources.LiveSource, "_spawn", lambda self, hub, fn: None)
        fake_cam = types.ModuleType("jtsmm.viewer.cam_preview")

        class _Feed:
            camera_ids: list = []

            def start_live(self, *a: Any, **k: Any) -> None:
                pass

        fake_cam.CamPreviewFeed = _Feed  # type: ignore[attr-defined]
        fake_cam.PREVIEW_SIZE = (320, 240)  # type: ignore[attr-defined]
        mp.setitem(sys.modules, "jtsmm.viewer.cam_preview", fake_cam)

        class _Replay:
            kind = "replay"

            def __init__(self, path: Any, *, speed: float = 1.0, start_s: float = 0.0) -> None:
                h.obs.record("replay", path=Path(path), speed=speed, start_s=start_s)

        mp.setattr(sources, "ReplaySource", _Replay)

        def run_case(*, port: Any, resolved: dict, duration_s: float = 10.0, label: str = "",
                     emg: bool = True, **k: Any):
            h.obs.record(
                "run_case", port=port, payload=build_mcu_apply_payload(resolved),
                duration_s=duration_s, label=label, emg=emg,
            )
            ok = imu_rates.StreamRate("mta", "game_rotation_vector", 10, 100.0, 100.0, 0.0, True)
            return imu_rates.CaseResult(label=label, duration_s=duration_s, streams=[ok])

        mp.setattr(imu_rates, "run_case", run_case)

        def fit_session(session: Path, which: str, max_rms_deg: float, output_path: Any):
            h.obs.record("fit", session=Path(session), which=which, max_rms_deg=max_rms_deg,
                         output=output_path)
            return 0, {}

        def persist_take(**kw: Any) -> None:
            h.obs.record("persist", **kw)

        mp.setattr(imu_cmd, "_fit_session", fit_session)
        mp.setattr(imu_cmd, "_persist_take", persist_take)

        def query_mcu_info(port: Any = None, **k: Any) -> McuInfo:
            h.obs.record("mcu_info", port=port)
            return McuInfo(port=port or "/dev/fakehat", firmware_version="1.2.3", raw_banner="jtsmm 1.2.3")

        mp.setattr(info_mod, "query_mcu_info", query_mcu_info)

    # -- running ----------------------------------------------------------- #
    def run(self, args: list[str], scenario: dict[str, Any] | None = None) -> Obs:
        self.obs = Obs()
        self.scenario = dict(scenario or {})
        cfg = self.world.config_for(self.scenario.get("yaml", ()))
        res = CliRunner().invoke(main, [str(a) for a in args], env={"JTSMM_CONFIG": str(cfg)})
        self.obs.exit_code = res.exit_code
        self.obs.output = res.output
        if res.exception is not None and not isinstance(res.exception, SystemExit):
            self.obs.exception = res.exception
        return self.obs


class World:
    """Files shared by the whole module (so baselines can be cached)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.baselines: dict[tuple, Obs] = {}
        doc = load_yaml(template_path())
        host = doc.setdefault("host", {})
        host.setdefault("paths", {})["data_dir"] = str(root / "data")
        if "storage" in host:
            host["storage"]["base_dir"] = str(root / "data")
        host["viewer"] = {"enabled": False, "bind_host": "127.0.0.9", "port": 9001}
        session = doc["session"]
        session.update(
            optical_track=False,
            use_experiment_plan=False,
            ct_calibration=False,
            ct_calibration_recordings_dir=str(root / "ct_rec"),
            imu_calibration_recordings_dir=str(root / "imu_rec"),
        )
        self.base_doc = doc
        self._configs: dict[tuple, Path] = {}
        self.base_cfg = self.config_for(())
        self.alt_cfg = self.config_for(
            (
                ("mcu.emg.default_gain", 12),
                ("hat.serial_port", "/dev/alt_hat"),
                ("host.viewer.port", 9555),
                ("host.integrity.thresholds.emg_gap_warn_periods", 3.0),
            ),
            name="alt.yaml",
        )
        gappy = copy.deepcopy(doc)
        del gappy["mcu"]["emg"]["leadoff"]["current_na"]
        self.gappy_cfg = self._write(gappy, "gappy.yaml")
        unknown = copy.deepcopy(doc)
        unknown["host"]["viewer"]["not_a_key"] = 1
        self.unknown_cfg = self._write(unknown, "unknown_key.yaml")
        self.plan_json = root / "plan.json"
        self.plan_json.write_text(json.dumps({"name": "from_file", "phases": ["rest"]}))
        self.session_h5 = root / "given_session" / "session.h5"
        self.session_h5.parent.mkdir(parents=True, exist_ok=True)
        self.session_h5.touch()
        self.existing = root / "existing.yaml"
        self.existing.write_text("old: true\n")

    def _write(self, doc: dict, name: str) -> Path:
        path = self.root / name
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    def config_for(self, overrides: tuple, name: str | None = None) -> Path:
        key = tuple(overrides)
        if key not in self._configs:
            doc = copy.deepcopy(self.base_doc)
            for dotted, value in key:
                _set(doc, dotted, value)
            self._configs[key] = self._write(doc, name or f"cfg{len(self._configs)}.yaml")
        return self._configs[key]


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return World(tmp_path_factory.mktemp("plumbing"))


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, world: World) -> Harness:
    return Harness(monkeypatch, world)


# --------------------------------------------------------------------------- #
# Getters and checks for the EFFECTS table
# --------------------------------------------------------------------------- #

Getter = Callable[[Obs], Any]


def LAST(path: str) -> Getter:
    """Leaf of the last `config apply` payload."""
    return lambda o: _get(o.applies[-1]["payload"], path) if o.applies else _NONE


def FIRST(path: str) -> Getter:
    return lambda o: _get(o.applies[0]["payload"], path) if o.applies else _NONE


def SESS(attr: str) -> Getter:
    """Argument the SessionSupervisor got."""
    return lambda o: o.sessions[-1][attr] if o.sessions else _NONE


def META(key: str) -> Getter:
    return lambda o: o.sessions[-1]["extra_meta"].get(key, _NONE) if o.sessions else _NONE


def PRE(fn: Callable[[dict], Any]) -> Getter:
    return lambda o: fn(o.preflight[-1]) if o.preflight else _NONE


def CALL(name: str, key: str, idx: int = -1) -> Getter:
    return lambda o: o.calls[name][idx].get(key, _NONE) if o.calls.get(name) else _NONE


def N(what: str) -> Getter:
    """How many applies / sessions / preflights / captures / integrity checks / <calls>."""
    return lambda o: len(getattr(o, what)) if hasattr(o, what) else len(o.calls.get(what, []))


def PORTS(o: Obs) -> Any:
    return set(o.hat_ports) or _NONE


def WROTE(cmd: bytes) -> Getter:
    return lambda o: cmd in o.writes


def OUT(text: str) -> Getter:
    return lambda o: text in o.output


def EXIT(o: Obs) -> int:
    return o.exit_code


Check = Callable[[Obs, Obs, "World"], None]


def changed(getter: Getter, want: Any) -> Check:
    """Variant observes ``want``; the baseline must not (so the value is non-default).

    ``want`` may be a callable taking the World (for per-module paths).
    """

    def check(base: Obs, var: Obs, wd: "World") -> None:
        if callable(want):
            want_v = Path(want(wd))
            norm = lambda v: v if v in (None, _NONE) else Path(v)  # noqa: E731
            got, was = norm(getter(var)), norm(getter(base))
        else:
            want_v, got, was = want, getter(var), getter(base)
        assert got == want_v, f"variant observed {got!r}, expected {want_v!r}\n{var.output[-2000:]}"
        assert was != want_v, f"baseline already observed {was!r}: pick a non-default value"

    return check


def all_of(*checks: Check) -> Check:
    def check(base: Obs, var: Obs, wd: "World") -> None:
        for c in checks:
            c(base, var, wd)

    return check


def json_output(base: Obs, var: Obs, wd: "World") -> None:
    json.loads(var.output[var.output.index("{"):])
    with pytest.raises(ValueError):
        json.loads(base.output)


@dataclass(frozen=True)
class Effect:
    kind: str
    argv: tuple = ()
    check: Callable[..., None] | None = None
    base: tuple = ()
    scenario: tuple = ()
    reason: str = ""


# argv entries / expected paths that are callables take the World (per-module paths).
def P_ALT(wd: World) -> Path:
    return wd.alt_cfg


def P_X(wd: World) -> Path:
    return wd.root / "x_path"


def P_PLAN(wd: World) -> Path:
    return wd.plan_json


def P_SESSION(wd: World) -> Path:
    return wd.session_h5


_OVERRIDE_EFFECTS = {
    "bias_drive": Effect("mcu_payload", ("--no-bias-drive",), changed(LAST("emg.bias.enabled"), False)),
    "bias_channels": Effect("mcu_payload", ("--bias-channels", "1,2"), changed(LAST("emg.bias.channels"), 0x03)),
    "active_channels": Effect(
        "mcu_payload", ("--active-channels", "4"),
        all_of(changed(LAST("emg.pd_channels"), 0x37), changed(LAST("emg.bias.channels"), 0x08)),
    ),
    "leadoff": Effect("mcu_payload", ("--no-leadoff",), changed(LAST("emg.leadoff.during_recording"), False)),
    "default_gain": Effect("mcu_payload", ("--gain", "24"), changed(LAST("emg.default_gain"), 24)),
}


def _overrides(cmd: str) -> dict:
    return {(cmd, k): v for k, v in _OVERRIDE_EFFECTS.items()}


_SESSION_EFFECTS = {
    "port": Effect(
        "call_arg", ("--port", PORT),
        all_of(changed(PORTS, {PORT}), changed(PRE(lambda p: p["resolved"]["hat"]["serial_port"]), PORT)),
    ),
    "optical_config": Effect("call_arg", ("--optical-config", P_X), changed(SESS("optical_config"), P_X)),
    "skip_apply": Effect(
        "call_arg", ("--skip-apply",),
        all_of(changed(N("applies"), 0), changed(META("mcu_apply_skip_reason"), "--skip-apply")),
    ),
    "skip_preflight": Effect("call_arg", ("--skip-preflight",), changed(N("preflight"), 0)),
    "no_doctor": Effect("call_arg", ("--no-doctor",), changed(N("preflight"), 0)),
    "viewer": Effect("call_arg", ("--viewer",), changed(SESS("viewer"), True)),
    "viewer_host": Effect("call_arg", ("--viewer-host", "10.1.2.3"), changed(SESS("viewer_host"), "10.1.2.3")),
    "viewer_port": Effect("call_arg", ("--viewer-port", "9100"), changed(SESS("viewer_port"), 9100)),
    "mock": Effect(
        "call_arg", ("--mock",),
        all_of(changed(SESS("mock_only"), True), changed(N("applies"), 0),
               changed(PRE(lambda p: p["check_serial"]), False)),
    ),
}

_CT_EFFECTS = {
    "ct_calibration_n_poses": Effect(
        "call_arg", ("--ct-calibration-n-poses", "7"), changed(SESS("ct_calibration_n_poses"), 7)
    ),
    "ct_tip_calibration": Effect(
        "call_arg", ("--ct-tip-calibration", P_X), changed(SESS("ct_tip_calibration"), P_X)
    ),
    "persist_ct_calibration": Effect(
        "call_arg", ("--persist-ct-calibration",), changed(SESS("persist_ct_calibration"), True)
    ),
    "ct_calibration_recordings_dir": Effect(
        "call_arg", ("--ct-calibration-recordings-dir", P_X),
        changed(SESS("ct_calibration_recordings_dir"), P_X),
    ),
}


def _with(cmd: str, table: dict) -> dict:
    return {(cmd, k): v for k, v in table.items()}


EFFECTS: dict[tuple[str, str], Effect] = {
    # ---- jtsmm (group) ----
    ("", "version"): Effect("output", ("--version",), changed(OUT(", version "), True)),
    # ---- record ----
    ("record", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(LAST("emg.default_gain"), 12)
    ),
    ("record", "duration_s"): Effect("call_arg", ("--duration", "3.5"), changed(SESS("duration_s"), 3.5)),
    **_with("record", _SESSION_EFFECTS),
    **_with("record", _CT_EFFECTS),
    ("record", "optical_track"): Effect("call_arg", ("--optical-track",), changed(SESS("optical_track"), True)),
    ("record", "ct_calibration"): Effect(
        "call_arg", ("--ct-calibration",), changed(SESS("ct_calibration"), True)
    ),
    ("record", "mvc"): Effect("meta", ("--mvc",), changed(META("recording_mode"), "mvc")),
    ("record", "use_experiment_plan"): Effect(
        "call_arg", ("--use-experiment-plan",),
        changed(lambda o: getattr(SESS("experiment_plan")(o), "name", None), "s03_m_33"),
    ),
    ("record", "experiment_plan"): Effect(
        "call_arg", ("--experiment-plan", P_PLAN),
        changed(lambda o: getattr(SESS("experiment_plan")(o), "name", None), "from_file"),
    ),
    ("record", "write_jts_config"): Effect(
        "call_arg", ("--no-write-jts-config",), changed(SESS("write_jts_config"), False)
    ),
    ("record", "jts_config_path"): Effect("call_arg", ("--jts-config", P_X), changed(SESS("jts_config_path"), P_X)),
    ("record", "no_integrity"): Effect("call_arg", ("--no-integrity",), changed(N("integrity"), 0)),
    **_overrides("record"),
    ("record", "subject"): Effect("meta", ("--subject", "s42"), changed(META("subject_id"), "s42")),
    ("record", "protocol"): Effect("meta", ("--protocol", "p7"), changed(META("protocol"), "p7")),
    ("record", "meta_kv"): Effect("meta", ("--meta", "operator=jd"), changed(META("operator"), "jd")),
    ("record", "emg_check"): Effect(
        "meta", ("--no-emg-check",),
        all_of(changed(N("captures"), 0), changed(META("emg_check_json"), _NONE)),
    ),
    ("record", "no_abort"): Effect(
        "call_arg", ("--no-abort",), changed(N("sessions"), 1), scenario=(("preflight", ("boom",)),)
    ),
    ("record", "force_emg_check"): Effect(
        "call_arg", ("--force-emg-check",), changed(N("sessions"), 1), scenario=(("emg_railed", True),)
    ),
    # ---- ct-calibrate ----
    ("ct-calibrate", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(LAST("emg.default_gain"), 12)
    ),
    **_with("ct-calibrate", _SESSION_EFFECTS),
    **_with("ct-calibrate", _CT_EFFECTS),
    # ---- doctor (group options) ----
    ("doctor", "config_path"): Effect(
        "call_arg", ("--config", P_ALT),
        changed(PRE(lambda p: (p["resolved"] or {}).get("hat", {}).get("serial_port")), "/dev/alt_hat"),
    ),
    ("doctor", "port"): Effect(
        "call_arg", ("--port", PORT),
        changed(PRE(lambda p: (p["resolved"] or {}).get("hat", {}).get("serial_port")), PORT),
    ),
    ("doctor", "no_serial"): Effect(
        "call_arg", ("--no-serial",),
        all_of(changed(PRE(lambda p: p["check_serial"]), False), changed(PRE(lambda p: p["require_hat"]), False)),
    ),
    ("doctor", "mcu_selftest"): Effect(
        "call_arg", ("--mcu-selftest",), changed(PRE(lambda p: p["check_mcu_selftest"]), True)
    ),
    # ---- doctor imu-rates ----
    ("doctor imu-rates", "preset"): Effect(
        "mcu_payload", ("--preset", "raw_only"),
        changed(lambda o: sorted(_get(CALL("run_case", "payload")(o), "imu_mta.reports")), ["accelerometer", "gyroscope"]),
    ),
    ("doctor imu-rates", "all_presets"): Effect("call_arg", ("--all-presets",), changed(N("run_case"), 5)),
    ("doctor imu-rates", "slot"): Effect(
        "mcu_payload", ("--slot", "mta"),
        changed(lambda o: _get(CALL("run_case", "payload")(o), "imu_cra.profile"), "kinematics_accel"),
        base=("--preset", "raw_only"),
    ),
    ("doctor imu-rates", "duration"): Effect(
        "call_arg", ("--duration", "3"), changed(CALL("run_case", "duration_s"), 3.0)
    ),
    ("doctor imu-rates", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT),
        changed(lambda o: _get(CALL("run_case", "payload")(o), "emg.default_gain"), 12),
    ),
    ("doctor imu-rates", "report_path"): Effect(
        "output", ("--report", P_X), changed(OUT("report written"), True)
    ),
    ("doctor imu-rates", "emg"): Effect(
        "call_arg", ("--no-emg",), changed(CALL("run_case", "emg"), False)
    ),
    ("doctor imu-rates", "sweep"): Effect(
        "call_arg", ("--sweep",), changed(CALL("run_case", "label", 0), "accelerometer@100")
    ),
    ("doctor imu-rates", "sweep_report"): Effect(
        "mcu_payload", ("--sweep-report", "gyroscope"),
        changed(lambda o: _get(CALL("run_case", "payload")(o), "imu_cra.reports.gyroscope.rate_hz"), 400),
        base=("--sweep",),
    ),
    ("doctor imu-rates", "min_fusion_hz"): Effect(
        "output", ("--min-fusion-hz", "123"), changed(OUT("fusion >= 123 Hz"), True),
        base=("--sweep",),
    ),
    ("doctor imu-rates", "sweep_rates"): Effect(
        "call_arg", ("--sweep-rates", "150,180"), changed(N("run_case"), 2), base=("--sweep",),
    ),
    # ---- emg check ----
    ("emg check", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(LAST("emg.default_gain"), 12)
    ),
    ("emg check", "port"): Effect("call_arg", ("--port", PORT), changed(PORTS, {PORT})),
    ("emg check", "duration"): Effect(
        "call_arg", ("--duration", "1.5"), changed(lambda o: o.captures[0] if o.captures else _NONE, 1.5)
    ),
    ("emg check", "line_hz"): Effect("call_arg", ("--line-hz", "60"), changed(CALL("line_hz", "nominal"), 60.0)),
    ("emg check", "with_impedance"): Effect(
        "mcu_payload", ("--no-impedance",),
        changed(lambda o: any(a["payload"]["emg"]["leadoff"].get("ac_tone") for a in o.applies), False),
    ),
    **_overrides("emg check"),
    # ---- emg impedance ----
    ("emg impedance", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(FIRST("emg.default_gain"), 12)
    ),
    ("emg impedance", "port"): Effect("call_arg", ("--port", PORT), changed(PORTS, {PORT})),
    ("emg impedance", "duration"): Effect(
        "call_arg", ("--duration", "1.25"), changed(lambda o: o.captures[0] if o.captures else _NONE, 1.25)
    ),
    ("emg impedance", "current_na"): Effect(
        "mcu_payload", ("--current-na", "24"),
        all_of(changed(FIRST("emg.leadoff.current_na"), 24), changed(OUT("I=24 nA"), True)),
    ),
    ("emg impedance", "export_path"): Effect(
        "output", ("--export", P_X), changed(OUT("exported:"), True)
    ),
    **_overrides("emg impedance"),
    # ---- imu calibrate ----
    ("imu calibrate", "mode"): Effect(
        "call_arg", ("--mode", "fit", P_SESSION),
        all_of(changed(N("sessions"), 0), changed(CALL("fit", "session"), P_SESSION)),
    ),
    ("imu calibrate", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(LAST("emg.default_gain"), 12)
    ),
    **_with("imu calibrate", _SESSION_EFFECTS),
    ("imu calibrate", "no_integrity"): Effect("call_arg", ("--no-integrity",), changed(N("integrity"), 0)),
    ("imu calibrate", "fit"): Effect("call_arg", ("--no-fit",), changed(N("fit"), 0)),
    ("imu calibrate", "which"): Effect("call_arg", ("--imu", "mta"), changed(CALL("fit", "which"), "mta")),
    ("imu calibrate", "max_rms_deg"): Effect(
        "call_arg", ("--max-rms-deg", "5.5"), changed(CALL("fit", "max_rms_deg"), 5.5)
    ),
    ("imu calibrate", "output_path"): Effect("call_arg", ("--output", P_X), changed(CALL("fit", "output"), P_X)),
    ("imu calibrate", "persist_imu_calibration"): Effect(            # default on: test the off switch
        "call_arg", ("--no-persist-imu-calibration",), changed(CALL("persist", "persist"), False)
    ),
    ("imu calibrate", "imu_calibration_recordings_dir"): Effect(
        "call_arg", ("--imu-calibration-recordings-dir", P_X), changed(CALL("persist", "recordings_dir"), P_X)
    ),
    ("imu calibrate", "run_index"): Effect("call_arg", ("--run", "3"), changed(CALL("persist", "run"), 3)),
    ("imu calibrate", "subject"): Effect(
        "call_arg", ("--subject", "s9"), changed(CALL("persist", "subject_id"), "s9")
    ),
    # ---- config ----
    ("config apply", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT), changed(LAST("emg.default_gain"), 12)
    ),
    ("config apply", "port"): Effect("call_arg", ("--port", PORT), changed(PORTS, {PORT})),
    ("config apply", "dry_run"): Effect(
        "output", ("--dry-run",), all_of(changed(N("applies"), 0), changed(OUT("config apply {"), True))
    ),
    ("config apply", "persist"): Effect(
        "call_arg", ("--persist",), changed(lambda o: o.applies[-1]["persist"] if o.applies else _NONE, True)
    ),
    ("config audit", "config_path"): Effect(
        "output", ("--config", P_ALT), changed(lambda o: "alt.yaml" in o.output, True)
    ),
    ("config audit", "strict_mode"): Effect(
        "output", ("--strict",), changed(EXIT, 1), base=("--config", (lambda wd: wd.gappy_cfg))
    ),
    ("config audit", "as_json"): Effect("output", ("--json",), json_output),
    ("config init", "dest"): Effect("output", ("--dest", P_X), changed(OUT("x_path"), True)),
    ("config init", "force"): Effect(
        "output", ("--force",), changed(OUT("wrote "), True), base=("--dest", (lambda wd: wd.existing))
    ),
    ("config show", "config_path"): Effect(
        "output", ("--config", P_ALT), changed(OUT("/dev/alt_hat"), True)
    ),
    ("config show", "mcu_json"): Effect("output", ("--mcu-json",), json_output),
    ("config validate", "config_path"): Effect(
        "output", ("--config", (lambda wd: wd.unknown_cfg)), changed(EXIT, 1)
    ),
    ("config validate", "no_strict"): Effect(
        "output", ("--no-strict",), changed(EXIT, 0), base=("--config", (lambda wd: wd.unknown_cfg))
    ),
    # ---- info ----
    ("info", "port"): Effect("call_arg", ("--port", PORT), changed(CALL("mcu_info", "port"), PORT)),
    ("info", "as_json"): Effect("output", ("--json",), json_output),
    # ---- integrity check (SESSION_PATH is given in BASE_ARGS) ----
    ("integrity check", "config_path"): Effect(
        "call_arg", ("--config", P_ALT),
        changed(
            lambda o: _get(o.integrity[-1].get("resolved_config"), "host.integrity.thresholds.emg_gap_warn_periods")
            if o.integrity else _NONE,
            3.0,
        ),
    ),
    ("integrity check", "verify_config_hash"): Effect(
        "call_arg", ("--verify-config-hash",),
        changed(lambda o: o.integrity[-1].get("expected_config") is not None if o.integrity else _NONE, True),
        base=("--config", P_ALT),
    ),
    ("integrity check", "strict_warn"): Effect(
        "output", ("--strict",), changed(EXIT, 1), scenario=(("warnings", ("slow imu",)),)
    ),
    ("integrity check", "plain"): Effect("output", ("--plain",), changed(OUT("integrity: OK"), True)),
    ("integrity check", "write_summary"): Effect(
        "output", ("--write-summary",), changed(OUT("Wrote integrity summary"), True)
    ),
    # ---- replay (PATH is given in BASE_ARGS) ----
    ("replay", "config_path"): Effect("call_arg", ("--config", P_ALT), changed(CALL("run_server", "port"), 9555)),
    ("replay", "host"): Effect("call_arg", ("--host", "10.9.9.9"), changed(CALL("run_server", "host"), "10.9.9.9")),
    ("replay", "http_port"): Effect("call_arg", ("--http-port", "9200"), changed(CALL("run_server", "port"), 9200)),
    ("replay", "speed"): Effect("call_arg", ("--speed", "2.5"), changed(CALL("replay", "speed"), 2.5)),
    ("replay", "start_s"): Effect("call_arg", ("--start", "1.5"), changed(CALL("replay", "start_s"), 1.5)),
    ("replay", "no_open"): Effect("call_arg", ("--no-open",), changed(CALL("run_server", "open"), False)),
    # ---- trigger ----
    ("trigger", "port"): Effect("call_arg", ("--port", PORT), changed(PORTS, {PORT})),
    ("trigger", "rate"): Effect("output", ("--rate", "60"), changed(WROTE(b"trig_start 60\r\n"), True)),
    ("trigger", "width_us"): Effect(
        "output", ("--width-us", "1000"), changed(WROTE(b"trig_start 60 1000\r\n"), True), base=("--rate", "60")
    ),
    ("trigger", "do_stop"): Effect(
        "output", ("--stop",),
        all_of(changed(WROTE(b"trig_stop\r\n"), True), changed(WROTE(b"trig_start\r\n"), False)),
    ),
    ("trigger", "do_status"): Effect("output", ("--status",), changed(WROTE(b"trig_start\r\n"), False)),
    # ---- view ----
    ("view", "config_path"): Effect(
        "mcu_payload", ("--config", P_ALT),
        all_of(changed(LAST("emg.default_gain"), 12), changed(CALL("run_server", "port"), 9555)),
    ),
    ("view", "serial_port"): Effect("call_arg", ("--port", PORT), changed(PORTS, {PORT})),
    ("view", "host"): Effect("call_arg", ("--host", "10.9.9.9"), changed(CALL("run_server", "host"), "10.9.9.9")),
    ("view", "http_port"): Effect("call_arg", ("--http-port", "9200"), changed(CALL("run_server", "port"), 9200)),
    ("view", "no_open"): Effect("call_arg", ("--no-open",), changed(CALL("run_server", "open"), False)),
    ("view", "mock"): Effect(
        "call_arg", ("--mock",),
        all_of(changed(lambda o: CALL("run_server", "source")(o).kind, "mock"), changed(N("applies"), 0)),
    ),
    **_overrides("view"),
}

BASE_ARGS: dict[str, tuple] = {
    "": (),
    "replay": (P_SESSION,),
    "integrity check": (P_SESSION,),
    "doctor imu-rates": (),
}


def _resolve_argv(argv: tuple, wd: World) -> list[str]:
    return [str(a(wd)) if callable(a) else str(a) for a in argv]


def _command_args(cmd: str, extra: tuple, wd: World) -> list[str]:
    head = cmd.split() if cmd else []
    return head + _resolve_argv(BASE_ARGS.get(cmd, ()) + extra, wd)


def _baseline(h: Harness, cmd: str, eff: Effect) -> Obs:
    key = (cmd, eff.base, eff.scenario)
    if key not in h.world.baselines:
        h.world.baselines[key] = h.run(_command_args(cmd, eff.base, h.world), dict(eff.scenario))
    return h.world.baselines[key]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def _walk(cmd: click.Command, path: tuple[str, ...] = ()):
    for p in cmd.params:
        if isinstance(p, click.Option):
            yield (" ".join(path), p.name)
    if isinstance(cmd, click.Group):
        for name, sub in sorted(cmd.commands.items()):
            yield from _walk(sub, path + (name,))


KINDS = {"mcu_payload", "call_arg", "meta", "output", "none"}


def test_every_option_is_classified():
    options = set(_walk(main))
    missing = sorted(options - set(EFFECTS))
    stale = sorted(set(EFFECTS) - options)
    assert not missing, f"unclassified CLI options (add them to EFFECTS): {missing}"
    assert not stale, f"EFFECTS entries for options that no longer exist: {stale}"
    for key, eff in EFFECTS.items():
        assert eff.kind in KINDS, key
        if eff.kind == "none":
            assert eff.reason, f"{key}: kind 'none' needs a reason"
        else:
            assert eff.argv and eff.check is not None, key


@pytest.mark.parametrize(
    "key",
    [k for k, e in EFFECTS.items() if e.kind != "none"],
    ids=lambda k: f"{k[0] or 'jtsmm'}:{k[1]}",
)
def test_option_reaches_its_effect(harness: Harness, key: tuple[str, str]):
    cmd, _ = key
    eff = EFFECTS[key]
    base = _baseline(harness, cmd, eff)
    var = harness.run(_command_args(cmd, eff.base + eff.argv, harness.world), dict(eff.scenario))
    for label, obs in (("baseline", base), ("variant", var)):
        assert obs.exception is None, f"{label} raised {obs.exception!r}\n{obs.output[-3000:]}"
    eff.check(base, var, harness.world)


@pytest.mark.parametrize(
    "cmd", sorted({k[0] for k in EFFECTS} - {"config validate", "config audit"})
)
def test_baseline_runs_clean(harness: Harness, cmd: str):
    """Each command's baseline succeeds, so an option's diff is not an error path."""
    obs = harness.run(_command_args(cmd, (), harness.world))
    assert obs.exception is None, repr(obs.exception)
    if cmd != "":
        assert obs.exit_code == 0, obs.output[-3000:]


# ---- /meta equals the last payload applied before the take ----------------- #


def _canonical_hash(sess: dict, payload: dict) -> str:
    r = sess["resolved"]
    return config_hash(
        {
            "schema_version": payload["schema_version"],
            "hat": r.get("hat", {}),
            "mcu": {k: payload[k] for k in ("platform", "emg", "imu_mta", "imu_cra", "trigger")},
            "host": r.get("host", {}),
            "session": r.get("session", {}),
        }
    )


@pytest.mark.parametrize(
    "args,scenario",
    [
        ((), {}),
        (("--active-channels", "4", "--bias-channels", "4", "--gain", "12", "--no-leadoff"), {}),
        ((), {"yaml": (("host.record.mcu_apply.enabled", False),)}),
        (("--no-bias-drive",), {"yaml": (("host.record.mcu_apply.enabled", False),)}),
    ],
    ids=["yaml", "overrides", "apply-disabled", "apply-disabled+override"],
)
def test_record_meta_is_the_last_applied_payload(harness: Harness, args, scenario):
    obs = harness.run(["record", *args], scenario)
    assert obs.exit_code == 0, obs.output[-3000:]
    sess = obs.sessions[-1]
    assert sess["applies_before"] >= 1
    last = obs.applies[sess["applies_before"] - 1]["payload"]
    assert json.loads(sess["mcu_config_json"]) == last
    assert sess["config_hash"] == _canonical_hash(sess, last)
    # Nothing touched the MCU config after the take started.
    assert len(obs.applies) == sess["applies_before"]
    if scenario:
        assert sess["extra_meta"]["mcu_apply_skipped"] == "true"
        assert sess["extra_meta"]["mcu_config_restored_after_emg_check"] == "true"
    else:
        assert sess["extra_meta"]["mcu_apply_skipped"] == "false"


@pytest.mark.parametrize(
    "args,scenario,reason",
    [
        (("--skip-apply",), {}, "--skip-apply"),
        (("--no-emg-check",), {"yaml": (("host.record.mcu_apply.enabled", False),)},
         "host.record.mcu_apply.enabled=false"),
    ],
)
def test_record_without_any_apply_says_so_in_meta(harness: Harness, args, scenario, reason):
    obs = harness.run(["record", *args], scenario)
    assert obs.exit_code == 0, obs.output[-3000:]
    assert obs.applies == []
    assert obs.captures == []  # the EMG check would have rewritten the MCU config
    meta = obs.sessions[-1]["extra_meta"]
    assert meta["mcu_apply_skipped"] == "true"
    assert meta["mcu_apply_skip_reason"] == reason
    assert "mcu_config_dump_json" in meta


# ---- EMG check sub-sessions ----------------------------------------------- #


@pytest.mark.parametrize("cmd", [["record"], ["emg", "check"]])
@pytest.mark.parametrize("extra", [(), ("--active-channels", "4", "--bias-channels", "4")])
def test_drl_probe_powers_down_all_muscle_channels(harness: Harness, cmd, extra):
    obs = harness.run([*cmd, *extra])
    assert obs.exception is None
    probes = [a["payload"]["emg"] for a in obs.applies if a["payload"]["emg"]["leadoff"].get("probe_drl")]
    assert len(probes) == 1, [a["payload"]["emg"] for a in obs.applies]
    assert probes[0]["pd_channels"] == 0x3F
    assert probes[0]["bias"]["enabled"] is False
    # The check's other sub-sessions and the restore keep the take's channel set.
    want_pd = 0x37 if extra else None
    others = [a["payload"]["emg"] for a in obs.applies if not a["payload"]["emg"]["leadoff"].get("probe_drl")]
    assert others and all(e.get("pd_channels") == want_pd for e in others)
    assert obs.applies[-1]["payload"]["emg"]["leadoff"].get("probe_drl") in (None, False)


def test_emg_impedance_current_reaches_the_mcu_and_the_math(harness: Harness):
    obs = harness.run(["emg", "impedance", "--current-na", "24"])
    assert obs.exception is None
    first = obs.applies[0]["payload"]["emg"]["leadoff"]
    assert first["current_na"] == 24 and first["ac_tone"] is True
    assert "I=24 nA" in obs.output
    # ... and the MCU is put back on the YAML value afterwards.
    assert obs.applies[-1]["payload"]["emg"]["leadoff"]["current_na"] == 6


@pytest.mark.parametrize(
    "cmd",
    [["record"], ["emg", "check"], ["emg", "impedance"], ["view"]],
)
def test_contradictory_overrides_are_usage_errors(harness: Harness, cmd):
    obs = harness.run([*cmd, "--no-bias-drive", "--bias-channels", "1"])
    assert obs.exception is None, f"traceback instead of a usage error: {obs.exception!r}"
    assert obs.exit_code != 0
    assert obs.applies == []


# ---- precedence: an explicit value equal to the click default still wins --- #


@pytest.mark.parametrize(
    "args,getter,want",
    [
        (["view", "--http-port", "8770"], CALL("run_server", "port"), 8770),
        (["view", "--host", "0.0.0.0"], CALL("run_server", "host"), "0.0.0.0"),
        (["replay", P_SESSION, "--http-port", "8770"], CALL("run_server", "port"), 8770),
        (["record", "--viewer-port", "8770"], SESS("viewer_port"), 8770),
        (["ct-calibrate", "--viewer-port", "8770"], SESS("viewer_port"), 8770),
        (["imu", "calibrate", "--viewer-port", "8770"], SESS("viewer_port"), 8770),
    ],
    ids=["view-port", "view-host", "replay-port", "record", "ct-calibrate", "imu-calibrate"],
)
def test_explicit_default_value_beats_yaml(harness: Harness, args, getter, want):
    obs = harness.run(_resolve_argv(tuple(args), harness.world))  # YAML: 127.0.0.9:9001
    assert obs.exception is None
    assert getter(obs) == want


_YAML_DEFAULTS = (
    ("host.emg_check.leadoff.duration_s", 2.5),
    ("host.emg_check.impedance.duration_s", 1.75),
    ("host.emg_check.impedance.enabled", False),
    ("host.emg_check.line_hz", 60.0),
)


@pytest.mark.parametrize(
    "args,getter,want",
    [
        (["emg", "check"], lambda o: o.captures[0], 2.5),
        (["emg", "check"], CALL("line_hz", "nominal"), 60.0),
        (["emg", "check"], lambda o: any(a["payload"]["emg"]["leadoff"]["ac_tone"] for a in o.applies), False),
        (["emg", "impedance"], lambda o: o.captures[0], 1.75),
        (["view"], CALL("run_server", "port"), 9001),
        (["view"], CALL("run_server", "host"), "127.0.0.9"),
        (["replay", P_SESSION], CALL("run_server", "port"), 9001),
        (["record"], SESS("viewer_port"), 9001),
    ],
    ids=["check-duration", "check-line-hz", "check-impedance", "impedance-duration",
         "view-port", "view-host", "replay-port", "record-viewer-port"],
)
def test_yaml_applies_when_the_option_is_omitted(harness: Harness, args, getter, want):
    """The other half of precedence: a click default must not shadow the YAML."""
    obs = harness.run(_resolve_argv(tuple(args), harness.world), {"yaml": _YAML_DEFAULTS})
    assert obs.exception is None
    assert getter(obs) == want


def test_doctor_group_options_reach_imu_rates(harness: Harness):
    obs = harness.run(["doctor", "--port", PORT, "--config", harness.world.alt_cfg, "imu-rates"])
    assert obs.exit_code == 0, obs.output
    case = obs.calls["run_case"][-1]
    assert case["port"] == PORT
    assert case["payload"]["emg"]["default_gain"] == 12


# ---- integrity check --config X grades with X's thresholds (real checker) -- #


def test_integrity_check_uses_the_given_configs_thresholds(tmp_path: Path, world: World):
    pytest.importorskip("h5py")
    from jtsmm.tools.mock_session import write_mock_session

    h5 = write_mock_session(
        tmp_path / "s.h5", duration_s=6.0, write_jts_config=False, emg_dropouts=[(2.0, 0.004)]
    )["session_h5"]
    lax = world.config_for(
        (
            ("host.integrity.thresholds.emg_gap_warn_periods", 1000.0),
            ("host.integrity.thresholds.emg_gap_err_periods", 1000.0),
            ("host.integrity.thresholds.emg_loss_warn_frac", 0.5),
            ("host.integrity.thresholds.emg_loss_err_frac", 0.5),
        ),
        name="lax_integrity.yaml",
    )
    runner = CliRunner()
    strict = runner.invoke(main, ["integrity", "check", "--plain", str(h5)])
    relaxed = runner.invoke(main, ["integrity", "check", "--plain", "--config", str(lax), str(h5)])

    def gap_lines(out: str) -> list[str]:
        return [ln for ln in out.splitlines() if ln.startswith("error:") and "EMG" in ln]

    assert gap_lines(strict.output), strict.output
    assert not gap_lines(relaxed.output), relaxed.output


@pytest.mark.parametrize("cmd", [["emg", "impedance"], ["emg", "check"]])
def test_emg_tools_start_emg_only(harness: Harness, cmd: list[str]):
    """No IMU streams (FAULT LED without IMUs) and no camera trigger for EMG tools."""
    obs = harness.run([*cmd, "--port", PORT])
    assert obs.exception is None, obs.output
    starts = obs.calls["start_mcu_streams"]
    assert starts and all(
        c["kw"] == {"trigger": False, "imu_mta": False, "imu_cra": False} for c in starts
    )
