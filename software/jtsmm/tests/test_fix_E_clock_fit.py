# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Host <-> MCU clock fit (``ClockAligner._pi_fit``) and its fallbacks."""

from __future__ import annotations

import numpy as np

from jtsmm.sync.clock_align import ClockAligner

OFF_US = 7_000_000.0  # t_master - t_host at host time 0


def _pps(al: ClockAligner, t_host_us: float, *, ppm: float = 0.0, jitter_us: float = 0.0) -> None:
    t_master = t_host_us * (1.0 + ppm * 1e-6) + OFF_US
    al.ingest_pps(int((t_host_us + jitter_us) * 1000), int(round(t_master)))


def test_start_pass_alone_is_offset_only():
    """8 samples over ~170 ms: a free slope here was hundreds of ppm off."""
    al = ClockAligner()
    rng = np.random.default_rng(1)
    for k in range(8):
        _pps(al, 1e9 + k * 24_000, ppm=20.0, jitter_us=float(rng.normal(0, 300)))
    assert al.pi_fit_mode == "offset_only"
    assert al._pi_fit.a == 1.0  # noqa: SLF001
    # 10 minutes later the offset-only map is off only by the true 20 ppm drift.
    t_late = 1e9 + 600e6
    mapped, q = al.map_pi_ns(int(t_late * 1000))
    assert q == "pi_fit"
    truth = t_late * (1 + 20e-6) + OFF_US
    assert abs(mapped - truth) < 20e-6 * 600e6 + 2_000


def test_end_pass_keeps_start_samples_and_frees_the_slope():
    al = ClockAligner()
    for k in range(64):
        _pps(al, 1e9 + k * 20_000, ppm=25.0, jitter_us=float((-1) ** k * 200))
    for k in range(64):  # end pass must not evict the start pass
        _pps(al, 1e9 + 300e6 + k * 20_000, ppm=25.0, jitter_us=float((-1) ** k * 200))
    assert al._pi_fit.n_samples == 128  # noqa: SLF001
    assert al.pi_fit_mode == "free_slope"
    assert abs((al._pi_fit.a - 1.0) * 1e6 - 25.0) < 2.0  # noqa: SLF001
    rep = al.sync_report()["pi_to_master"]
    assert rep["mode"] == "free_slope" and rep["span_s"] > 299


def test_map_pi_ns_at_close_uses_the_final_fit():
    """Package F re-maps stored t_host_ns with map_pi_ns after the end PPS pass."""
    al = ClockAligner()
    for k in range(8):
        _pps(al, 1e9 + k * 24_000, ppm=40.0)
    t_event = 1e9 + 200e6
    live, _ = al.map_pi_ns(int(t_event * 1000))
    for k in range(8):
        _pps(al, 1e9 + 400e6 + k * 24_000, ppm=40.0)
    final, q = al.map_pi_ns(int(t_event * 1000))
    truth = t_event * (1 + 40e-6) + OFF_US
    assert q == "pi_fit"
    assert abs(final - truth) < 50
    assert abs(live - truth) > 5_000  # the live, offset-only stamp was 8 ms off


def test_short_span_below_threshold_stays_offset_only():
    al = ClockAligner(pps_slope_min_span_s=30.0)
    for k in range(4):
        _pps(al, 1e9 + k * 5e6, ppm=10.0)  # 15 s span
    assert al.pi_fit_mode == "offset_only"


def test_soft_pairs_are_offset_only_and_not_labelled_pi_fit():
    """Processing-time pairs carry a variable positive delay; a slope through
    them came out ~20 in simulation."""
    al = ClockAligner()
    rng = np.random.default_rng(3)
    for i in range(200):
        t_master = 1_000_000 + i * 33_333
        delay_us = float(rng.uniform(0, 30_000)) + (40_000 if i % 17 == 0 else 0)
        t_host_ns = int((t_master - OFF_US + delay_us) * 1000)
        al.note_host_mcu_pair(t_host_ns, t_master)
    t_host = (1_000_000 + 250 * 33_333 - OFF_US) * 1000
    mapped, q = al.map_pi_ns(int(t_host))
    assert q == "mcu_arrival"
    assert abs(mapped - (1_000_000 + 250 * 33_333)) < 2_000
    assert al._pi_fit.n_samples == 0  # noqa: SLF001
    # A later real handshake is not contaminated by the soft pairs.
    _pps(al, 5e9)
    assert al._pi_fit.n_samples == 1 and al.map_pi_ns(int(5e12))[1] == "pi_fit"  # noqa: SLF001


def test_camera_anchor_requires_real_pps_samples():
    al = ClockAligner(pps_anchor=True, delta_tracking=False)
    for i in range(200):
        al.note_host_mcu_pair(int((i * 1000 - OFF_US) * 1000), i * 1000)
    for i in range(1, 20):
        al.note_trigger(i, 2_000_000 + i * 33_333)
    al.ingest_cam("cam0", (2_000_000 + 5 * 33_333) * 1000, None)
    assert al.anchor_info["cam0"]["method"] == "oldest_buffered"
    assert al.assignment_residual_stats("cam0") is None
