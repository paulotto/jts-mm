# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Unit tests for lead-off status gating in the viewer hub."""

from __future__ import annotations

from jtsmm.viewer.hub import Hub, LEADOFF_BAD_FRAMES, LEADOFF_GOOD_FRAMES


def test_hub_ignores_leadoff_when_inactive():
    hub = Hub(emg_fs=2000, emg_lsb_uv=0.1, leadoff_during_recording=False)
    status = bytes([0xC0, 0x10, 0x00])
    hub.set_leadoff(status)
    snap = hub.snapshot()
    assert not any(snap.leadoff_p)
    assert snap.leadoff_active is False


def test_hub_debounces_comparator_leadoff():
    hub = Hub(emg_fs=2000, emg_lsb_uv=0.1, leadoff_during_recording=True, n_channels=6)
    status = bytes([0xC0, 0x10, 0x00])  # CH0 P lead-off
    for _ in range(LEADOFF_BAD_FRAMES - 1):
        hub.set_leadoff(status)
        snap = hub.snapshot()
        assert not snap.leadoff_p[0]
    hub.set_leadoff(status)
    snap = hub.snapshot()
    assert snap.leadoff_p[0]
    assert snap.leadoff_active is True


def test_hub_clears_leadoff_after_good_streak():
    hub = Hub(emg_fs=2000, emg_lsb_uv=0.1, leadoff_during_recording=True, n_channels=6)
    off = bytes([0xC0, 0x10, 0x00])
    ok = bytes([0xC0, 0x00, 0x00])
    for _ in range(LEADOFF_BAD_FRAMES):
        hub.set_leadoff(off)
    assert hub.snapshot().leadoff_p[0]
    for _ in range(LEADOFF_GOOD_FRAMES):
        hub.set_leadoff(ok)
    assert not hub.snapshot().leadoff_p[0]


def test_hub_ignores_the_masked_drl_comparator_bit():
    """K8: CH7/CH8 comparator bits are invalid on this board (IN8P unwired)."""
    hub = Hub(emg_fs=2000, emg_lsb_uv=0.1, leadoff_during_recording=True, n_channels=6)
    off = bytes([0xC0, 0x08, 0x00])  # CH8N lead-off
    for _ in range(3 * LEADOFF_BAD_FRAMES):
        hub.set_leadoff(off)
    snap = hub.snapshot()
    assert snap.drl_leadoff is False
    assert not any(snap.leadoff_p) and not any(snap.leadoff_n)


def test_hub_clears_drl_when_leadoff_inactive():
    hub = Hub(emg_fs=2000, emg_lsb_uv=0.1, leadoff_during_recording=True, n_channels=6)
    off = bytes([0xC0, 0x08, 0x00])
    for _ in range(LEADOFF_BAD_FRAMES):
        hub.set_leadoff(off)
    hub.leadoff_during_recording = False
    hub.set_leadoff(off)
    snap = hub.snapshot()
    assert snap.drl_leadoff is False
    assert snap.leadoff_active is False


def test_railed_input_stays_red_while_comparator_status_is_all_zero():
    """With comparators off/unknown the status word is all zero; the rail
    detector is then the only contact indicator and must not be unlatched by
    those zero frames while the input is still stuck at full scale."""
    import numpy as np

    from jtsmm.config.catalog import lsb_uv

    lsb = lsb_uv(6)
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb, leadoff_during_recording=True, n_channels=6)
    zero = bytes([0xC0, 0x00, 0x00])
    railed = np.zeros((200, 6), dtype=np.float32)
    railed[:, 0] = 0.99 * (1 << 23) * lsb  # CH1 pinned near +full scale
    t = 0
    for _ in range(3 * LEADOFF_GOOD_FRAMES):
        hub.append_emg_raw(np.arange(t, t + 200) * 500, railed)
        t += 200
        hub.set_leadoff(zero)
    assert hub.snapshot().leadoff_p[0], "railed input must stay latched"
    clean = np.zeros((200, 6), dtype=np.float32)
    for _ in range(LEADOFF_GOOD_FRAMES):
        hub.append_emg_raw(np.arange(t, t + 200) * 500, clean)
        t += 200
        hub.set_leadoff(zero)
    assert not hub.snapshot().leadoff_p[0], "clears once the rail is gone"
