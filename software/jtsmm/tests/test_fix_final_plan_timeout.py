# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""--duration running out before the experiment plan finished is plan_timeout."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from jtsmm import native
from tests.test_fix_F_supervisor import _sync, env  # noqa: F401 - fixture

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)


class _NeverPressed:
    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return None

    def poll(self):
        return None


def test_duration_timeout_before_the_plan_finished_is_plan_timeout(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", _NeverPressed)
    monkeypatch.setitem(sys.modules, "jtsmm.viewer.cam_preview",
                        SimpleNamespace(PREVIEW_SIZE=(320, 240)))
    monkeypatch.setattr(env.sup_mod, "_WATCHDOG_SILENCE_S", 30.0)
    plan = SimpleNamespace(
        name="p", slug="p", description="", needs_optical=False,
        all_phases=[SimpleNamespace(name=n, kind="experiment") for n in ("a", "b")],
        to_dict=lambda: {"name": "p"},
    )
    h5_path = SessionSupervisor(
        env.resolved, port="/dev/null", experiment_plan=plan, write_jts_config=False
    ).run(duration_s=0.3)
    sync = _sync(h5_path)
    assert sync["end_reason"] == "plan_timeout"
    assert sync["phases_aborted"] is True
