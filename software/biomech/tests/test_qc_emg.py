# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from biomech.emg.pipeline import bandpass, clean, rms_envelope, take_line_hz
from biomech.qc import metrics as m

FS = 4000.0


def test_coverage_counts_stalls_and_edges():
    t = np.arange(0, 2_000_000, 2_500, dtype=np.int64)          # 400 Hz for 2 s, then stops
    assert m.coverage(t, 0, 2_000_000, 0.05) > 0.99
    assert m.coverage(t, 0, 4_000_000, 0.05) < 0.51               # second half lost
    assert m.coverage(t, 3_000_000, 4_000_000, 0.05) == 0.0       # nothing in window


def test_nonmonotonic_and_rate():
    t = np.array([0, 2500, 2500, 5000, 4000, 7500])
    assert m.nonmonotonic_count(t) == 2
    assert abs(m.stream_rate_hz(np.arange(0, 1_000_000, 2_500)) - 400.0) < 1e-6


def test_snr_db():
    assert m.snr_db(100.0, 10.0) == 20.0
    assert np.isnan(m.snr_db(float("nan"), 1.0))


def test_rms_envelope_of_sine():
    t = np.arange(int(2 * FS)) / FS
    env = rms_envelope(10 * np.sin(2 * np.pi * 100 * t), FS, 0.15)
    assert abs(np.median(env) - 10 / np.sqrt(2)) < 0.05


def test_line_hz_falls_back_to_nominal_on_clean_signal():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((int(10 * FS), 2))
    assert take_line_hz(x, FS, nominal_hz=50.0) == 50.0


def test_clean_removes_hum_and_keeps_emg_band():
    rng = np.random.default_rng(1)
    t = np.arange(int(10 * FS)) / FS
    emg = rng.standard_normal(t.size) * 5
    hum = 200 * np.sin(2 * np.pi * 50.01 * t)
    f0 = take_line_hz(emg + hum, FS)
    out, _ = clean((emg + hum + 5000)[:, None], FS, line_hz=f0)
    assert abs(f0 - 50.01) < 0.01
    ref = bandpass(emg[:, None], FS).std()          # in-band share of the white "EMG"
    assert abs(out.std() / ref - 1.0) < 0.05         # hum and DC gone, in-band EMG kept


def test_artifact_mask_rules_and_masked_envelope():
    from biomech.emg.artifacts import MaskParams, artifact_mask, masked_envelope

    rng = np.random.default_rng(2)
    n = int(4 * FS)
    clean_x = rng.standard_normal((n, 2)) * 5
    raw = clean_x.copy()
    raw[int(2 * FS):int(2.3 * FS), 0] += 3000 * np.sin(np.linspace(0, np.pi, int(0.3 * FS)))
    cl = clean_x.copy()
    cl[int(1 * FS):int(1.1 * FS), 1] += 400       # far beyond 3 x ref (ref = 20)
    res = artifact_mask(raw, cl, FS, ref_uv=np.array([20.0, 20.0]),
                        rest_lf_uv=np.array([1.0, 1.0]), p=MaskParams())
    # D14 default: the low-frequency rule is computed and flagged but does not exclude.
    assert res.analysis_rules == ("rail", "leadoff", "envelope")
    assert not res.mask[int(2.1 * FS), 0] and res.flagged("lowfreq")[int(2.1 * FS), 0]
    assert res.any_hit()[int(2.1 * FS), 0]
    assert res.mask[int(1.05 * FS), 1] and res.counts["envelope"][1] > 0
    assert res.counts["lowfreq"][0] > 0 and res.counts["lowfreq"][1] == 0
    # Sensitivity configuration: lowfreq excludes as before D14 (dilated).
    lf = artifact_mask(raw, cl, FS, ref_uv=np.array([20.0, 20.0]),
                       rest_lf_uv=np.array([1.0, 1.0]),
                       p=MaskParams(analysis_rules=("rail", "leadoff", "envelope", "lowfreq")))
    assert lf.mask[int(2.1 * FS), 0] and not lf.mask[int(0.5 * FS), 0]
    assert not lf.flagged("lowfreq").any()
    np.testing.assert_array_equal(lf.counts["lowfreq"], res.counts["lowfreq"])
    env = masked_envelope(cl, res.mask, FS)
    assert np.isnan(env[int(1.05 * FS), 1]) and np.isfinite(env[int(3 * FS), 1])
    assert abs(np.nanmedian(env[:, 1]) - 5) < 0.5


def test_dilate_rows_equals_binary_dilation():
    from scipy.ndimage import binary_dilation

    from biomech.emg.artifacts import dilate_rows

    rng = np.random.default_rng(5)
    for n, c, p in ((1, 1, 0.5), (7, 3, 0.2), (500, 4, 0.01), (2000, 2, 0.001)):
        m = rng.random((n, c)) < p
        m[0, 0] = True                                     # runs touching both edges
        m[-1, -1] = True
        for k in (0, 1, 3, 40, 3 * n):
            ref = (binary_dilation(m, structure=np.ones((2 * k + 1, 1), bool)) if k else m)
            np.testing.assert_array_equal(dilate_rows(m, k), ref)
    assert not dilate_rows(np.zeros((10, 2), bool), 3).any()


def test_mask_params_rules_validated_and_parsed():
    import pytest

    from biomech.emg.artifacts import MaskParams
    from biomech.emg.prepare import EmgParams

    with pytest.raises(ValueError, match="unknown mask rules"):
        MaskParams(analysis_rules=("rail", "lowfrequency"))
    p = EmgParams.from_dict({"mask": {"analysis_rules": ["rail", "lowfreq"]}})
    assert p.mask.analysis_rules == ("rail", "lowfreq")
    assert EmgParams.from_dict({}).mask.analysis_rules == ("rail", "leadoff", "envelope")
