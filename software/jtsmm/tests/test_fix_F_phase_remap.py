# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Phase/event stamps are re-mapped with the final host->master fit at close."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import h5py
import pytest

from jtsmm import native
from jtsmm.capture.phase_walker import PhaseWalker
from jtsmm.capture.supervisor import remap_phase_stamps


class _FinalFit:
    """map_pi_ns with the 'final' fit: master_us = host_us + 1_000_000."""

    def __init__(self, n_samples: int = 16, quality: str = "pi_fit") -> None:
        self._pi_fit = SimpleNamespace(n_samples=n_samples)
        self.quality = quality

    def map_pi_ns(self, t_host_ns):
        return int(t_host_ns // 1000) + 1_000_000, self.quality


def _walk(stamps: list[int]) -> PhaseWalker:
    """Two phases; the live stamp_fn drifts by 757 ppm like take1's start-only fit."""
    it = iter(stamps)

    def live_stamp():
        t_host = next(it)
        t_us = t_host // 1000
        return t_host, int(t_us * 1.000757) + 1_000_000, "pi_fit"

    plan = SimpleNamespace(all_phases=[SimpleNamespace(name="a", kind="experiment"),
                                       SimpleNamespace(name="b", kind="experiment")])
    w = PhaseWalker.from_plan(plan, live_stamp)
    for _ in range(4):
        w.poll(ord(" "))
    return w


def test_remap_restamps_events_and_intervals():
    host = [100_000_000_000, 200_000_000_000, 300_000_000_000, 400_000_000_000]  # 100..400 s
    w = _walk(host)
    live = [(p.t_start_us, p.t_end_us) for p in w.recorded]
    out = remap_phase_stamps(w, _FinalFit())
    assert out["n_events"] == 4 and out["n_remapped"] == 4
    want = [h // 1000 + 1_000_000 for h in host]
    assert [e.t_master_us for e in w.events] == want
    assert [(p.t_start_us, p.t_end_us) for p in w.recorded] == [(want[0], want[1]), (want[2], want[3])]
    assert out["live_intervals"] == live
    # 757 ppm over 400 s = 302.8 ms: the size of the error the live stamps had
    assert 302_000 <= out["max_abs_correction_us"] <= 303_500


def test_remap_skips_without_a_final_fit():
    w = _walk([1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])
    before = [e.t_master_us for e in w.events]
    out = remap_phase_stamps(w, _FinalFit(quality="mcu_arrival"))
    assert out["n_remapped"] == 0
    assert [e.t_master_us for e in w.events] == before


def test_remap_disabled_for_mock():
    w = _walk([1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000])
    assert remap_phase_stamps(w, _FinalFit(), enabled=False)["n_remapped"] == 0


@pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)
def test_supervisor_writes_remapped_events_and_phases(monkeypatch, tmp_path, resolved_config):
    import contextlib

    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.sync.clock_align import ClockAligner
    from tests.test_fix_F_supervisor import FakeMcu

    class LateFitAligner(ClockAligner):
        """Live stamps use offset 0; the fit 'improves' by +50 ms at close."""

        offset_us = 0

        def map_pi_ns(self, t_host_ns=None):
            return int(t_host_ns // 1000) + self.offset_us, "pi_fit"

        def sync_report(self):
            LateFitAligner.offset_us = 50_000  # the end PPS pass has happened by now
            return super().sync_report()

    presses = iter([None, None, ord(" "), None, None, ord(" ")] + [None] * 1000)

    class Keys:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return None

        def poll(self):
            return next(presses)

    for key, val in (("JTSMM_NATIVE", "1"), ("JTSMM_SETTLE_MS", "0"),
                     ("JTSMM_STREAM_RESTART", "0"), ("JTSMM_PPS_SAMPLES", "0")):
        monkeypatch.setenv(key, val)
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: FakeMcu())
    for name in ("start_mcu_streams", "start_mcu_trigger", "stop_mcu_streams",
                 "stop_mcu_streams_robust", "_send_stats_request"):
        monkeypatch.setattr(sup_mod, name, lambda *_a, **_k: None)
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _p: contextlib.nullcontext())
    monkeypatch.setattr(sup_mod, "ClockAligner", LateFitAligner)
    monkeypatch.setattr(sup_mod, "_FINAL_STATS_TIMEOUT_S", 0.05)
    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", Keys)
    monkeypatch.setitem(sys.modules, "jtsmm.viewer.cam_preview", SimpleNamespace(PREVIEW_SIZE=(320, 240)))
    resolved_config["host"].setdefault("storage", {})["base_dir"] = str(tmp_path / "data")

    plan = SimpleNamespace(
        name="p", slug="p", description="", needs_optical=False,
        all_phases=[SimpleNamespace(name="open_close", kind="experiment")],
        to_dict=lambda: {"name": "p"},
    )
    h5_path = sup_mod.SessionSupervisor(
        resolved_config, port="/dev/null", experiment_plan=plan, write_jts_config=False
    ).run(duration_s=20.0)

    with h5py.File(h5_path, "r") as f:
        t_ev = [int(x) for x in f["events"]["t_master_us"][:]]
        t_host = [int(x) for x in f["events"]["t_host_ns"][:]]
        t0 = int(f["phases"]["t_start_us"][0])
        t1 = int(f["phases"]["t_end_us"][0])
        sync = json.loads(f["sync"].attrs["report_json"])
    assert t_ev == [h // 1000 + 50_000 for h in t_host]
    assert (t0, t1) == (t_ev[0], t_ev[1])
    rm = sync["phase_stamp_remap"]
    assert rm["n_remapped"] == 2 and rm["max_abs_correction_us"] == 50_000
    ph = sync["phases"][0]
    assert ph["t_start_us"] - ph["t_start_us_live"] == 50_000
    assert sync["end_reason"] == "phases_complete"
