# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Supervisor tags a failure with the phase it happened in (setup/take/finish)."""

from __future__ import annotations

import os
import signal

import pytest

from jtsmm import native
from tests.test_fix_F_supervisor import FakeCam, _sync, env  # noqa: F401 - fixture

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)


def test_failure_is_tagged_with_its_phase(env):
    from jtsmm.capture.supervisor import SessionSupervisor

    env.cams[0].start_raises = RuntimeError("picamera2 start failed")
    with pytest.raises(RuntimeError) as ei:
        SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=5.0)
    assert ei.value.jtsmm_phase == "setup"


def test_second_ctrl_c_in_the_loop_is_tagged_take(env):
    from jtsmm.capture.supervisor import SessionSupervisor

    class DoubleCtrlC(FakeCam):
        calls = 0

        def drain_frames(self, **_kw) -> list:
            self.calls += 1
            if self.calls == 3:  # stop request, then the hard abort
                os.kill(os.getpid(), signal.SIGINT)
                os.kill(os.getpid(), signal.SIGINT)
            return []

    env.cams = [DoubleCtrlC()]
    with pytest.raises(KeyboardInterrupt) as ei:
        SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=30.0)
    assert ei.value.jtsmm_phase == "take"
    assert _sync(ei.value.jtsmm_session_h5)["end_reason"] == "operator_abort"
