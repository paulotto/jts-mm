# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 EMG preparation: E22 masked rest noise, B1 phase memo, A3 QC without an MVC take,
E27 causal chain of a real (synthetic) phase."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from biomech.emg.prepare import EmgParams, EmgPrep

U = {"snr_min_db": 6.0, "rest_noise_uv": 20.0, "ref_mask_max": 0.10, "lowfreq_flag_frac": 0.10}


@pytest.fixture(scope="module")
def takes(tmp_path_factory):
    from biomech.io.session import SessionData
    from biomech.testing.synth import Spec, write_participant

    root = tmp_path_factory.mktemp("r4_participant")
    tr = write_participant(root, Spec(pid="s01", emg_fs=2000, seed=11))
    motion = SessionData(Path(root) / tr.motion_dir)
    mvc = SessionData(Path(root) / tr.mvc_dir)
    yield motion, mvc
    for sd in (motion, mvc):
        close = getattr(sd, "close", None)
        if close:
            close()


def test_b1_phase_is_memoized_and_release_reproduces(takes):
    motion, mvc = takes
    prep = EmgPrep(motion, mvc, EmgParams())
    a = prep.phase("open_close")
    assert prep.phase("open_close") is a
    assert prep.phase("open_close", normalization="max_recorded") is a     # default reference
    b = prep.phase("open_close", normalization="p95")
    assert b is not a and b.mask is not a.mask
    np.testing.assert_array_equal(a.mask, b.mask)
    with pytest.raises(ValueError):
        a.env[0, 0] = 1.0                                            # shared: read-only
    prep.release()
    c = prep.phase("open_close")
    assert c is not a
    for k in ("env", "norm", "mask", "cleaned", "raw"):
        np.testing.assert_array_equal(getattr(a, k), getattr(c, k))
    # Another mask-rule set never sees this memo.
    other = prep.with_mask_rules(("rail",))
    assert other.phase("open_close") is not c


def test_e22_rest_rms_uses_unmasked_samples_only(takes):
    motion, mvc = takes
    prep = EmgPrep(motion, mvc, EmgParams())
    t, raw, cl = prep._clean(motion, "rest")
    # Inject an artifact into the cleaned rest phase and mask it (as the rail/lead-off rule
    # would): the rest noise must not see it.
    cl2 = cl.copy()
    cl2[1000:1400, 0] += 500.0
    prep._cleaned[(id(motion), "rest")] = (t, raw, cl2)
    pe = prep.phase("rest")
    m = np.array(pe.mask)
    m[1000:1400, 0] = True
    prep._phases[("motion", "rest", prep.p.reference)] = replace(
        pe, masks=replace(pe.masks, mask=m))
    rms = prep.rest["rms_uv"]
    expect = np.std(np.delete(cl2[:, 0], np.flatnonzero(m[:, 0])))
    assert rms[0] == pytest.approx(expect, rel=1e-9)
    assert rms[0] < 0.5 * cl2[:, 0].std()                             # the old, unmasked value
    assert "analysis_mask" in prep.rest


def test_a3_qc_without_mvc_take_uses_d11_reference(takes):
    from biomech.qc.report import emg_table

    motion, _ = takes
    prep = EmgPrep(motion, None, EmgParams())
    t = emg_table(prep, "s01", U)
    assert t.ref_from_task.all()
    assert np.isfinite(t.snr_db).all() and t.usable_shape.any()
    assert not t.usable_amplitude.any()                               # still needs an MVC/RVC
    t_mvc = emg_table(EmgPrep(*takes, EmgParams()), "s01", U)
    assert not t_mvc.ref_from_task.any()


def test_e27_causal_clean_of_a_phase(takes):
    motion, mvc = takes
    prep = EmgPrep(motion, mvc, EmgParams())
    pe = prep.phase("open_close")
    xc = prep.causal("open_close")
    assert xc.shape == pe.cleaned.shape
    np.testing.assert_array_equal(np.isnan(xc), pe.mask)              # NaN exactly where masked
    assert prep.causal("open_close") is xc
    # Same band, similar level as the offline chain (not identical: causal filters).
    ok = ~pe.mask
    r = np.std(xc[ok]) / np.std(pe.cleaned[ok])
    assert 0.7 < r < 1.3
    assert prep.phase_line_hz(motion, "open_close") == pytest.approx(50.0, abs=0.2)
