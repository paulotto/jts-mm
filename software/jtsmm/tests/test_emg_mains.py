# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mains removal: the offline fit and the viewer's causal canceller.

Calibrated on s01_f_30, where ~90% of in-band power sat at the line frequency
and chewing was indistinguishable from rest (SNR 0.97-1.05) until the
interference was removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.emg.mains import (
    estimate_line_hz,
    mains_share,
    remove_mains,
    uniform_grid,
)
from jtsmm.viewer.processing import CausalEmgFilter, HarmonicCanceller

FS = 2000.0


def _signal(n_s=20.0, *, f0=50.0, hum=50.0, emg=5.0, seed=0, harmonics=(1, 2, 3)):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * FS)) / FS
    x = rng.standard_normal(t.size) * emg
    for k in harmonics:
        x = x + (hum / k) * np.sin(2 * np.pi * k * f0 * t + 0.3 * k)
    return t, x


# --- offline fit -------------------------------------------------------------


def test_removes_the_hum_down_to_the_noise_floor():
    _, x = _signal()
    y, f0 = remove_mains(x, FS)
    assert x.std() > 30.0
    assert y.std() == pytest.approx(5.0, abs=0.6)   # the EMG that was underneath
    assert f0 == pytest.approx(50.0, abs=0.05)


def test_broadband_emg_at_the_line_frequency_survives():
    """The point of fitting rather than notching: only the coherent tone goes."""
    _, x = _signal(hum=0.0, emg=20.0)
    y, _ = remove_mains(x, FS, line_hz=50.0)
    # A 5 Hz x 9-harmonic notch bank would take ~10% of a 20-450 Hz signal.
    assert y.std() > 0.97 * x.std()


def test_estimates_a_grid_that_is_not_exactly_50():
    _, x = _signal(f0=50.017, n_s=60.0)
    assert estimate_line_hz(x, FS) == pytest.approx(50.017, abs=0.01)


def test_sixty_hertz_grid_is_detected():
    _, x = _signal(f0=60.0)
    assert estimate_line_hz(x, FS) == pytest.approx(60.0, abs=0.05)


def test_share_falls_from_dominated_to_clean():
    _, x = _signal()
    y, f0 = remove_mains(x, FS)
    assert mains_share(x, FS, f0) > 0.9
    assert mains_share(y, FS, f0) < 0.15


def test_multichannel_shape_and_independence():
    _, a = _signal(seed=1)
    _, b = _signal(seed=2, hum=200.0)
    x = np.column_stack([a, b])
    y, _ = remove_mains(x, FS)
    assert y.shape == x.shape
    assert y[:, 0].std() == pytest.approx(5.0, abs=0.6)
    assert y[:, 1].std() == pytest.approx(5.0, abs=0.6)


def test_short_and_degenerate_inputs_do_not_raise():
    assert remove_mains(np.zeros((0, 3)), FS)[0].shape == (0, 3)
    assert remove_mains(np.ones(5), FS)[0].shape == (5,)
    assert remove_mains(np.ones(100), 0.0)[0].shape == (100,)


# --- the dropout trap --------------------------------------------------------


def test_dropped_samples_smear_the_line_frequency_unless_timestamps_are_used():
    """The failure that made filtering look useless on a real recording.

    A recorded stream is packed around dropped samples, so row index stops being
    proportional to time. On s01_f_30 take1 (2.92% missing) that put the
    apparent line frequency at 50.224 Hz against a true 50.020 -- far enough off
    that neither a notch nor a tracker helps.
    """
    t, x = _signal(n_s=30.0)
    t_us = (t * 1e6).astype(np.int64)
    # Drop 3% of samples in scattered short bursts, as the real fault did.
    rng = np.random.default_rng(3)
    drop = np.zeros(t.size, bool)
    for s in rng.integers(0, t.size - 20, size=int(0.03 * t.size / 8)):
        drop[s : s + 8] = True
    tp, xp = t_us[~drop], x[~drop]

    naive = estimate_line_hz(xp, FS)                  # index-as-time
    tu, xu = uniform_grid(tp, xp, FS)
    honest = estimate_line_hz(xu, FS)
    assert abs(naive - 50.0) > 0.4                    # smeared
    assert honest == pytest.approx(50.0, abs=0.05)    # recovered

    bad, _ = remove_mains(xp, FS)
    good, _ = remove_mains(xp, FS, t_us=tp)
    assert good.std() < 0.5 * bad.std()
    assert good.shape == xp.shape                     # mapped back onto input rows


# --- the viewer's causal canceller -------------------------------------------


def _tone_gain(fs, freq, n_s=20.0, **kw):
    t = np.arange(int(fs * n_s)) / fs
    x = np.sin(2 * np.pi * freq * t).reshape(-1, 1)
    f = CausalEmgFilter(fs, 1, **kw)
    y = np.vstack([f.process(x[i : i + 64])[0] for i in range(0, x.shape[0], 64)])
    tail = y[-int(fs * 4) :, 0]
    return float(np.sqrt(np.mean(tail**2)) * np.sqrt(2))


def test_tracking_beats_the_notch_on_a_drifted_grid():
    """Why 'track' is the default: the grid is never exactly 50.000 Hz.

    Measured 50.016 and 49.986 Hz in two sessions. A fixed notch is centred on
    the nominal frequency; the canceller locks onto the real one.
    """
    for f in (50.016, 49.986, 50.05):
        assert _tone_gain(500.0, f) < 0.01, f
        assert _tone_gain(500.0, f, mains_mode="notch") > _tone_gain(500.0, f)


def test_tracking_keeps_more_of_the_emg_band():
    for f in (45.0, 60.0, 75.0, 107.5):
        assert _tone_gain(500.0, f) > 0.98, f
    assert _tone_gain(500.0, 45.0) > _tone_gain(500.0, 45.0, mains_mode="notch")


def test_canceller_state_survives_block_boundaries():
    """Viewer frames arrive in small blocks; the estimate must not restart.

    With the frequency lock off the canceller core is a pure per-sample
    recursion, so chunking must not change the output. It is not bit-exact only
    because the phase accumulator is wrapped at block boundaries; 1e-7 uV on a
    30 uV tone is float noise, while a state reset would show as whole uV.
    """
    t = np.arange(int(20 * FS)) / FS
    x = (30 * np.sin(2 * np.pi * 50.0 * t)).reshape(-1, 1)
    one = HarmonicCanceller(FS, 1, [50.0], lock_hz=0.0)
    many = HarmonicCanceller(FS, 1, [50.0], lock_hz=0.0)
    y1 = one.process(x.copy())
    y2 = np.vstack([many.process(x[i : i + 37].copy()) for i in range(0, x.shape[0], 37)])
    assert np.allclose(y1[-2000:], y2[-2000:], atol=1e-7)


def test_lock_rate_does_not_depend_on_how_the_stream_is_chunked():
    """Regression: re-locking once per call made the lock rate a function of
    block size, so a loaded viewer would converge differently from an idle one."""
    t = np.arange(int(40 * FS)) / FS
    x = (30 * np.sin(2 * np.pi * 50.12 * t)).reshape(-1, 1)
    offsets = []
    for blk in (37, 128, 1024):
        hc = HarmonicCanceller(FS, 1, [50.0])
        for i in range(0, x.shape[0], blk):
            hc.process(x[i : i + blk].copy())
        offsets.append(hc.locked_offset_hz)
    assert max(offsets) - min(offsets) < 0.01, offsets
    assert all(o == pytest.approx(0.12, abs=0.02) for o in offsets), offsets


def test_canceller_locks_onto_the_real_grid_frequency():
    t = np.arange(int(40 * FS)) / FS
    x = (30 * np.sin(2 * np.pi * 50.12 * t)).reshape(-1, 1)
    hc = HarmonicCanceller(FS, 1, [50.0])
    for i in range(0, x.shape[0], 128):
        hc.process(x[i : i + 128])
    assert hc.locked_offset_hz == pytest.approx(0.12, abs=0.02)


def test_lock_is_clamped_so_it_cannot_walk_onto_the_emg_band():
    t = np.arange(int(30 * FS)) / FS
    x = (30 * np.sin(2 * np.pi * 57.0 * t)).reshape(-1, 1)   # far off, e.g. 60 Hz grid
    hc = HarmonicCanceller(FS, 1, [50.0], lock_hz=0.5)
    for i in range(0, x.shape[0], 128):
        hc.process(x[i : i + 128])
    assert abs(hc.locked_offset_hz) <= 0.5 + 1e-9


def test_modes_are_validated():
    with pytest.raises(ValueError):
        CausalEmgFilter(500.0, 1, mains_mode="bogus")


def test_off_mode_leaves_the_tone():
    assert _tone_gain(500.0, 50.0, mains_mode="off") > 0.9


def test_config_default_is_tracking():
    from jtsmm.config.registry import emg_filter_mains_mode

    assert emg_filter_mains_mode({}) == "track"
    assert emg_filter_mains_mode(
        {"host": {"emg_filter": {"mains_mode": "notch"}}}
    ) == "notch"
    # An unknown value must not silently disable mains removal.
    assert emg_filter_mains_mode({"host": {"emg_filter": {"mains_mode": "xx"}}}) == "track"


def test_recovers_known_emg_through_scattered_dropouts():
    """Ground truth: the only test that can tell the methods apart.

    A lower post-filter RMS is NOT evidence of better mains removal -- it can
    just as easily mean signal was destroyed. Resampling onto a uniform grid,
    fitting there and interpolating back looked better than this on a real
    recording (9-178 uV vs 63-454) while actually leaving 7.5x MORE mains
    residual against a known answer. Fitting at the true sample times needs no
    interpolator and lands on the truth.
    """
    fs = 2000.0
    rng = np.random.default_rng(0)
    t = np.arange(int(60 * fs)) / fs
    emg = rng.standard_normal(t.size) * 10.0
    hum = sum((3000.0 / k) * np.sin(2 * np.pi * k * 50.017 * t + 0.5 * k) for k in (1, 2, 3, 4, 5))
    t_us = (t * 1e6).astype(np.int64)

    drop = np.zeros(t.size, bool)
    for s0 in rng.integers(0, t.size - 20, size=int(0.03 * t.size / 8)):
        drop[s0 : s0 + 8] = True
    kept, noisy, truth = t_us[~drop], (emg + hum)[~drop], emg[~drop]

    def band_rms(x):
        from jtsmm.emg.check import _bandpass_rms

        return _bandpass_rms(x, fs, 20.0, 450.0)

    assert band_rms(noisy) > 1000.0
    recovered = band_rms(remove_mains(noisy, fs, t_us=kept)[0])
    assert recovered == pytest.approx(band_rms(truth), rel=0.15), recovered
    # and the naive index-as-time path must be visibly worse, not silently close
    assert band_rms(remove_mains(noisy, fs)[0]) > 10 * recovered


def test_dc_offset_cannot_leak_into_the_fit_through_dropouts():
    """The bug that masqueraded as unfixable interference.

    Real electrode pairs sit at 10-25 mV of DC offset. On a uniform grid whole
    cycles of a sinusoid are orthogonal to a constant; at real timestamps with
    dropped samples they are not, and ~2% of a 25 mV offset leaked into the
    fitted 50 Hz amplitude -- ~500 uV of tone subtracted that was never there.
    On s01_f_30 that RAISED resting in-band RMS (132 -> 288 uV) and was read as
    modulated interference no filter could remove. The zero-mean ground-truth
    test above never saw it, which is why this one exists.
    """
    from jtsmm.emg.check import _bandpass_rms

    fs = 2000.0
    rng = np.random.default_rng(5)
    t = np.arange(int(40 * fs)) / fs
    emg = rng.standard_normal(t.size) * 10.0
    hum = 150.0 * np.sin(2 * np.pi * 50.017 * t) + 40.0 * np.sin(2 * np.pi * 100.034 * t + 1.0)
    x = emg + hum + 25_000.0                       # a 25 mV offset, as recorded
    t_us = (t * 1e6).astype(np.int64)
    drop = np.zeros(t.size, bool)
    for s0 in rng.integers(0, t.size - 20, size=int(0.03 * t.size / 8)):
        drop[s0 : s0 + 8] = True
    kept, noisy, truth = t_us[~drop], x[~drop], emg[~drop]

    got = _bandpass_rms(remove_mains(noisy, fs, t_us=kept)[0], fs, 20.0, 450.0)
    want = _bandpass_rms(truth, fs, 20.0, 450.0)
    assert got == pytest.approx(want, rel=0.15), (got, want)
    # and it must not have made things worse than leaving the hum in
    assert got < 0.5 * _bandpass_rms(noisy, fs, 20.0, 450.0)


# --- the live tracker and dropped samples --------------------------------------


def _gapped_tone(seed=0, n_s=40.0, tone=300.0, emg=10.0, f0=50.0176):
    rng = np.random.default_rng(seed)
    t = np.arange(int(n_s * FS)) / FS
    x = tone * np.sin(2 * np.pi * f0 * t) + rng.standard_normal(t.size) * emg
    drop = np.zeros(t.size, bool)
    for s0 in rng.integers(0, t.size - 20, size=int(0.03 * t.size / 8)):
        drop[s0 : s0 + 8] = True
    return (t[~drop] * 1e6).astype(np.int64), x[~drop], x


def _tail_rms(y, seconds=10.0):
    from jtsmm.emg.check import _bandpass_rms

    return _bandpass_rms(np.asarray(y)[-int(seconds * FS) :], FS, 20.0, 450.0)


def test_causal_tracker_is_defeated_by_dropouts_fed_as_contiguous():
    """What happened on s01_f_30 take1: 3% missing samples in short bursts, the
    tracker never converged, and the viewer showed the hum untouched (rest RMS
    402 vs 403 uV raw) while the old notch bank still halved it."""
    t_us, xg, x = _gapped_tone()
    f = CausalEmgFilter(FS, 1, mains_mode="track", notch_hz=50.0176)
    clean = _tail_rms(f.process(x[:, None])[0][:, 0])
    f = CausalEmgFilter(FS, 1, mains_mode="track", notch_hz=50.0176)
    broken = _tail_rms(f.process(xg[:, None])[0][:, 0])
    assert clean < 12.0, clean
    assert broken > 10 * clean, (broken, clean)


def test_process_timed_advances_the_reference_across_gaps():
    t_us, xg, x = _gapped_tone()
    f = CausalEmgFilter(FS, 1, mains_mode="track", notch_hz=50.0176)
    # fed in viewer-sized blocks, gaps both inside blocks and at block joins
    out = np.vstack([f.process_timed(t_us[i : i + 256], xg[i : i + 256])[0]
                     for i in range(0, xg.shape[0], 256)])
    assert _tail_rms(out[:, 0]) < 12.0, _tail_rms(out[:, 0])


def test_process_timed_matches_process_when_there_are_no_gaps():
    _, _, x = _gapped_tone()
    t_us = (np.arange(x.size) / FS * 1e6).astype(np.int64)
    a = CausalEmgFilter(FS, 1, mains_mode="track").process(x[:, None])[0]
    b = CausalEmgFilter(FS, 1, mains_mode="track").process_timed(t_us, x[:, None])[0]
    assert np.allclose(a, b)


def test_hub_feeds_the_filter_with_timestamps():
    """The hub is where dropouts actually arrive; it must not undo the fix."""
    from jtsmm.config.catalog import lsb_uv
    from jtsmm.viewer.hub import Hub

    t_us, xg, _ = _gapped_tone()
    hub = Hub(emg_fs=FS, emg_lsb_uv=lsb_uv(6), n_channels=1, display_fs=FS, line_hz=50.0176)
    for i in range(0, xg.shape[0], 256):
        hub.append_emg_raw(t_us[i : i + 256].astype(np.uint64), xg[i : i + 256, None].astype(np.float32))
    snap = hub.snapshot()
    assert snap.emg_filt is not None
    assert _tail_rms(snap.emg_filt[:, 0]) < 12.0, _tail_rms(snap.emg_filt[:, 0])


@pytest.mark.parametrize("remainder", [25, 40, 60, 2000])
def test_short_trailing_block_leaves_no_transient(remainder):
    """A sub-block remainder used to be fitted alone and blow up (s09 mvc_closing).

    Before the fix a 25-sample tail produced ~1e9 uV after band-passing.
    """
    pytest.importorskip("scipy")
    from scipy.signal import butter, sosfiltfilt

    fs = 4000.0
    rng = np.random.default_rng(3)
    n = 11 * int(fs) + remainder
    t = np.arange(n) / fs
    x = -19000.0 + 40.0 * t + rng.standard_normal(n) * 5 + 80 * np.sin(2 * np.pi * 50.0 * t)
    y, _ = remove_mains(x, fs, line_hz=50.0)
    sos = butter(4, [20 / 2000, 450 / 2000], btype="band", output="sos")
    bp = sosfiltfilt(sos, y)
    assert bp[-int(fs):].std() < 1.5 * bp[: int(10 * fs)].std()


def test_robust_fit_keeps_quiet_stretches_quiet():
    """Bursts must not leak into the harmonic fit and back into quiet samples."""
    from scipy.signal import butter, sosfiltfilt

    fs = 2000.0
    rng = np.random.default_rng(5)
    t = np.arange(0, 20.0, 1 / fs)
    sos = butter(4, [20, 450], btype="band", fs=fs, output="sos")
    emg = sosfiltfilt(sos, rng.normal(size=t.size))
    emg /= emg.std()
    env = np.full(t.size, 3.0)
    on = (t % 1.0) < 0.5                         # 60 uV bursts in half of every block
    env[on] = 60.0
    hum = sum(a * np.sin(2 * np.pi * 50.0 * k * t + k) for k, a in ((1, 15.0), (3, 5.0), (5, 3.0)))
    x = emg * env + hum + 2000.0
    quiet = ~on & ((t % 1.0) > 0.6) & ((t % 1.0) < 0.95)
    truth = np.std((emg * env)[quiet])
    plain, _ = remove_mains(x, fs, line_hz=50.0)
    robust, _ = remove_mains(x, fs, line_hz=50.0, robust=True)
    band = lambda v: sosfiltfilt(sos, v)          # noqa: E731
    err_plain = np.std(band(plain)[quiet]) / truth
    err_robust = np.std(band(robust)[quiet]) / truth
    assert err_plain > 1.5                        # the plain fit adds hum next to bursts
    assert 0.85 < err_robust < 1.1                # the robust fit does not
    left = band(robust - emg * env - 2000.0)
    assert np.std(left) < 1.0                     # hum residual < 1 uV (plain: ~6 uV)
    # Burst energy is kept (the fit costs a few % at the harmonic bins).
    assert np.std(band(robust)[on]) == pytest.approx(np.std((emg * env)[on]), rel=0.06)


def test_auto_modulation_order_follows_modulated_hum():
    """Movement-modulated hum: the per-block BIC order removes it; steady hum costs nothing."""
    from scipy.signal import butter, sosfiltfilt

    fs = 2000.0
    t = np.arange(0, 12.0, 1 / fs)
    rng = np.random.default_rng(1)
    sos = butter(4, [20, 450], btype="band", fs=fs, output="sos")
    emg = sosfiltfilt(sos, rng.normal(size=t.size))
    emg *= 3.0 / emg.std()

    def residual(hum, order):
        y, _ = remove_mains(emg + hum, fs, line_hz=50.0, robust=True, amp_order=order)
        return np.std(sosfiltfilt(sos, y) - emg)

    steady = 15 * np.sin(2 * np.pi * 50 * t) + 5 * np.sin(2 * np.pi * 150 * t)
    mod = 1 + 0.3 * np.sin(2 * np.pi * 1.0 * t)
    moving = 15 * mod * np.sin(2 * np.pi * 50 * t) + 5 * mod * np.sin(2 * np.pi * 150 * t)
    assert residual(steady, "auto") <= residual(steady, 0) * 1.05
    assert residual(moving, "auto") < 0.6 * residual(moving, 0)
